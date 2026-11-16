"""Retro chips, first batch: Konami SCC, Namco WSG and NEC HuC6280 (PC Engine).

Registers
    chip systems  `scc` (ch1 .. ch5), `wsg` (v1 .. v3), `huc6280` (ch1 .. ch6) for the `chip` layer of genny.chiptune
    instruments   `scc`, `namco_wsg`, `huc6280`
The Atari TIA lives in genny.chiptune2.
"""
from __future__ import annotations

import numba as nb
import numpy as np
from scipy.signal import butter, lfilter

from . import chiptune as CT
from . import filters as F
from .chiptune2 import _H, driver_rows, lanes_of
from .core import DEFAULT_SR, samples
from .instruments import _soften, instrument
from .retro import _fade, _to_sr
from .spec import SpecError


# ============================================================================ shared
def _note_track(t0, t1, hz, vel, p, clk, period_of, rate):
    """One note as per-sample arrays at `rate`: (first sample, register period, driver volume 0..15)."""
    c0, c1 = int(round(t0 * clk)), int(round(t1 * clk))
    if c1 <= c0:
        return None
    rows = []
    driver_rows(c0, c1, clk, hz, vel, p, period_of, rows)
    cuts = np.round((np.array([r[0] for r in rows] + [c1], np.float64) - c0) * rate / clk).astype(np.int64)
    n = np.diff(cuts)
    return (int(round(c0 * rate / clk)), np.repeat([r[1] for r in rows], n).astype(np.int64),
            np.repeat([r[2] for r in rows], n).astype(np.int64))


def _walk(inc):
    """Table position (in steps) of an oscillator that advances `inc` steps per sample, starting at 0."""
    return np.cumsum(inc) - inc


def _put(buf, start, x):
    stop = min(start + len(x), len(buf))
    if stop > start:
        buf[start:stop] += x[:stop - start]


def _notes(lanes, name):
    notes = sorted(lanes.get(name, []), key=lambda n: n[0])
    for i, nt in enumerate(notes):
        if i + 1 < len(notes) and nt[1] > notes[i + 1][0]:
            notes[i] = (nt[0], notes[i + 1][0]) + nt[2:]
    return notes


def _table(w, names, size, lo, hi, what):
    if isinstance(w, (list, tuple, np.ndarray)):
        a = np.asarray(w, np.float64)
        if a.shape != (size,):
            raise SpecError(f"{what}: a wave table has {size} values")
        return np.clip(np.round(a), lo, hi)
    if isinstance(w, (int, float)) and not isinstance(w, bool):
        k = int(w)
        if not 0 <= k < len(names):
            raise SpecError(f"{what}: wave number must be 0..{len(names) - 1}")
        w = list(names)[k]
    if str(w).lower() not in names:
        raise SpecError(f"{what}: wave must be one of {sorted(names)} or a list of {size} values")
    return names[str(w).lower()]


def _fit(y, n):
    return np.pad(y, (0, max(0, n - len(y))))[:n]


def _finish(y, freq, sr, vel, mode, bright, rel, dur, tonal=True):
    y = F.dc_block(_fit(y, samples(dur + rel, sr)), sr)
    if str(mode) != "hardware_accurate":
        if tonal:
            y = _soften(y, freq, sr, harm=6, lo=2200, hi=3800, bright=bright)
        else:
            fc = 3000.0 * 2.0 ** ((bright - 0.5) * 4.0)
            if fc < sr * 0.45:
                y = F.lowpass(F.lowpass(y, fc, sr, 0.707), fc, sr, 0.707)
    return _fade(y.copy(), sr, 1.0, rel * 1000.0) * (0.5 + 0.5 * vel)


_X32 = np.arange(32)
_PATCH = {"vol": None, "arp": None, "pitch": None, "vib": 0.0, "vib_hz": 6.0, "vib_delay": 0.15, "gate": 1.0}


# ============================================================================ Konami SCC (K051649)
SCC_CLOCK = 3579545.0
SCC_NAMES = ("ch1", "ch2", "ch3", "ch4", "ch5")
SCC_GAIN = 0.5
SCC_WAVES = {
    "sine": np.round(127 * np.sin(2 * np.pi * _X32 / 32)),
    "saw": np.round(-128 + 255 * _X32 / 31),
    "square": np.where(_X32 < 16, 127.0, -128.0),
    "pulse": np.where(_X32 < 8, 127.0, -128.0),
    "triangle": np.round(np.where(_X32 < 16, -128 + 255 * _X32 / 15, 127 - 255 * (_X32 - 16) / 15)),
    "organ": np.round(127 * (np.sin(2 * np.pi * _X32 / 32) + 0.5 * np.sin(4 * np.pi * _X32 / 32)
                             + 0.25 * np.sin(6 * np.pi * _X32 / 32)) / 1.6),
    "noise": np.random.default_rng(0).integers(-128, 128, 32).astype(np.float64),
}
SCC_PATCH = dict(_PATCH, volume=15, wave="saw")


def scc_period(hz: float) -> int:
    """12-bit frequency register: the tone is clock / (32 * (N + 1))."""
    return int(min(max(round(SCC_CLOCK / (32.0 * max(hz, 1e-3))) - 1, 0), 4095))


def scc_hz(hz: float) -> float:
    """The pitch the chip plays for a written note (0 when the register is 8 or less: the channel is stopped)."""
    n = scc_period(hz)
    return SCC_CLOCK / (32.0 * (n + 1)) if n > 8 else 0.0


def render_scc(lanes, dur, sr, plus=False):
    """`dur` seconds of an SCC. lanes: ch1..ch5 -> notes (t0, t1, hz, vel, patch). Without `plus` channel 5 plays channel 4's wave."""
    rate = 2 * sr
    out = np.zeros(int(dur * rate) + 2)
    shared = None
    for name in SCC_NAMES:
        notes = _notes(lanes, name)
        if name == "ch4" and notes:
            shared = notes[0][4]["wave"]
        for t0, t1, hz, vel, p in notes:
            tr = _note_track(t0, t1, hz, vel, p, SCC_CLOCK, scc_period, rate)
            if tr is None:
                continue
            s, per, vol = tr
            wv = shared if (name == "ch5" and not plus and shared is not None) else p["wave"]
            tab = _table(wv, SCC_WAVES, 32, -128, 127, "scc") / 128.0
            inc = np.where(per > 8, SCC_CLOCK / (rate * (per + 1.0)), 0.0)
            idx = np.floor(_walk(inc)).astype(np.int64) % 32
            _put(out, s, tab[idx] * vol / 15.0)
    y = _to_sr(out, rate, sr)
    y = lfilter(*butter(1, 20.0, "high", fs=sr), y)
    return _fit(y, samples(dur, sr)) * SCC_GAIN


def _scc_layer(layer, sr, q):
    lanes, end = lanes_of(layer, "scc", SCC_NAMES, SCC_PATCH, q)
    return render_scc(lanes, end, sr, plus=bool(layer.get("plus", False)))


CT.SYSTEM_LAYERS["scc"] = _scc_layer
CT.SYSTEMS["scc"] = SCC_NAMES


@instrument("scc", "Konami SCC (K051649) channel: a 32-byte signed wave, 4-bit volume, 12-bit frequency from a 3.58 MHz clock.",
            family="retro", span=("C1", "C7"),
            wave=("saw", "saw | sine | square | pulse | triangle | organ | noise, a number 0..6, or a list of 32 signed bytes (-128..127)"),
            volume=(15, "0..15, linear"), vol=(None, _H["vol"]), arp=(None, _H["arp"]), pitch=(None, _H["pitch"]),
            vib=(0.0, _H["vib"]), vib_hz=(6.0, _H["vib_hz"]), vib_delay=(0.15, _H["vib_delay"]), mode=CT._MODE, bright=CT._BRIGHT)
def scc(freq, dur, sr=DEFAULT_SR, vel=1.0, wave="saw", volume=15, vol=None, arp=None, pitch=None, vib=0.0, vib_hz=6.0,
        vib_delay=0.15, mode="retro_stylized", bright=0.5):
    rel = 0.01
    p = dict(SCC_PATCH, wave=wave, volume=volume, vol=vol, arp=arp, pitch=pitch, vib=vib, vib_hz=vib_hz, vib_delay=vib_delay)
    y = render_scc({"ch1": [(0.0, dur, float(freq), 1.0, p)]}, dur + rel, sr)
    return _finish(y, freq, sr, vel, mode, bright, rel, dur)


# ============================================================================ Namco WSG (Pac-Man, Galaga, Dig Dug)
WSG_RATE = 96000.0
WSG_NAMES = ("v1", "v2", "v3")
WSG_GAIN = 0.4
_wq = lambda a: np.clip(np.round(a), 0, 15)  # noqa: E731
WSG_WAVES = {
    "sine": _wq(7.5 + 7.5 * np.sin(2 * np.pi * _X32 / 32)),
    "saw": _wq(_X32 / 2.0),
    "triangle": _wq(np.where(_X32 < 16, _X32, 31 - _X32)),
    "square": np.where(_X32 < 16, 15.0, 0.0),
    "pulse": np.where(_X32 < 8, 15.0, 0.0),
    "organ": _wq(7.5 + 7.5 * (np.sin(2 * np.pi * _X32 / 32) + 0.5 * np.sin(4 * np.pi * _X32 / 32)
                              + 0.25 * np.sin(6 * np.pi * _X32 / 32)) / 1.6),
    "bell": _wq(7.5 + 7.5 * (np.sin(2 * np.pi * _X32 / 32) * 0.6 + 0.4 * np.sin(2 * np.pi * 5 * _X32 / 32))),
    "half": _wq(np.maximum(15 * np.sin(2 * np.pi * _X32 / 32), 0)),
}
PROM_DEFAULT = np.array([
    7, 9, 10, 11, 12, 13, 13, 14, 14, 14, 13, 13, 12, 11, 10, 9, 7, 5, 4, 3, 2, 1, 1, 0, 0, 0, 1, 1, 2, 3, 4, 5,
    7, 12, 14, 14, 13, 11, 9, 10, 11, 11, 10, 9, 6, 4, 3, 5, 7, 9, 11, 10, 8, 5, 4, 3, 3, 4, 5, 3, 1, 0, 0, 2,
    7, 10, 12, 13, 14, 13, 12, 10, 7, 4, 2, 1, 0, 1, 2, 4, 7, 11, 13, 14, 13, 11, 7, 3, 1, 0, 1, 3, 7, 14, 7, 0,
    7, 13, 11, 8, 11, 13, 9, 6, 11, 14, 12, 7, 9, 10, 6, 2, 7, 12, 8, 4, 5, 7, 2, 0, 3, 8, 5, 1, 3, 6, 3, 1,
    0, 8, 15, 7, 1, 8, 14, 7, 2, 8, 13, 7, 3, 8, 12, 7, 4, 8, 11, 7, 5, 8, 10, 7, 6, 8, 9, 7, 7, 8, 8, 7,
    7, 8, 6, 9, 5, 10, 4, 11, 3, 12, 2, 13, 1, 14, 0, 15, 0, 15, 1, 14, 2, 13, 3, 12, 4, 11, 5, 10, 6, 9, 7, 8,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
], np.float64)
WSG_PATCH = dict(_PATCH, volume=15, wave=0)


def wsg_period(hz: float, voice: str = "v1") -> int:
    """20-bit frequency word: the tone is word * 96 kHz / 2**20. Voices 2 and 3 only have the upper 16 bits."""
    f = int(min(max(round(hz * 2 ** 20 / WSG_RATE), 0), 2 ** 20 - 1))
    return f if voice == "v1" else f & ~0xF


def wsg_hz(hz: float, voice: str = "v1") -> float:
    return wsg_period(hz, voice) * WSG_RATE / 2 ** 20


def render_wsg(lanes, dur, sr, prom=None, generic=False):
    """`dur` seconds of the 3-voice WSG at its native 96 kHz. The `wave` of a note is one of the PROM's eight 32-step blocks
    (0..7), a name, or 32 values 0..15; `prom` replaces the whole 256-nibble ROM; `generic` numbers the named generic waves instead."""
    out = np.zeros(int(dur * WSG_RATE) + 2)
    rom = None if generic else PROM_DEFAULT
    if prom is not None:
        rom = np.asarray(prom, np.float64)
        if rom.shape != (256,):
            raise SpecError("wsg: prom must be 256 nibbles")
        rom = np.clip(np.round(rom), 0, 15)
    for name in WSG_NAMES:
        for t0, t1, hz, vel, p in _notes(lanes, name):
            tr = _note_track(t0, t1, hz, vel, p, WSG_RATE * 32, lambda f, n=name: wsg_period(f, n), WSG_RATE)
            if tr is None:
                continue
            s, per, vol = tr
            w = p["wave"]
            if rom is not None and isinstance(w, (int, float)) and not isinstance(w, bool):
                tab = rom[(int(w) & 7) * 32:(int(w) & 7) * 32 + 32]
            else:
                tab = _table(w, WSG_WAVES, 32, 0, 15, "wsg")
            acc = np.cumsum(per.astype(np.float64)) - per          # the 20-bit accumulator, one add per 96 kHz sample
            idx = (np.floor(acc / 2 ** 15).astype(np.int64)) % 32
            _put(out, s, (tab[idx] - 8.0) / 8.0 * vol / 15.0)
    y = _to_sr(out, WSG_RATE, sr)
    y = lfilter(*butter(1, 20.0, "high", fs=sr), y)
    return _fit(y, samples(dur, sr)) * WSG_GAIN


def _wsg_layer(layer, sr, q):
    lanes, end = lanes_of(layer, "wsg", WSG_NAMES, WSG_PATCH, q)
    return render_wsg(lanes, end, sr, prom=layer.get("prom"), generic=bool(layer.get("generic", False)))


CT.SYSTEM_LAYERS["wsg"] = _wsg_layer
CT.SYSTEMS["wsg"] = WSG_NAMES


@instrument("namco_wsg", "Namco WSG (Pac-Man family) voice: a 32-step, 4-bit wave from the ROM, 20-bit frequency word at 96 kHz, 4-bit volume.",
            family="retro", span=("C1", "C7"),
            wave=(0, "ROM block 0..7, a name (sine | saw | triangle | square | pulse | organ | bell | half), or a list of 32 values 0..15"),
            generic=(False, "true: numbers 0..7 pick the generic named waves (the former default) instead of the ROM blocks"),
            volume=(15, "0..15, linear"), vol=(None, _H["vol"]), arp=(None, _H["arp"]), pitch=(None, _H["pitch"]),
            vib=(0.0, _H["vib"]), vib_hz=(6.0, _H["vib_hz"]), vib_delay=(0.15, _H["vib_delay"]), mode=CT._MODE, bright=CT._BRIGHT)
def namco_wsg(freq, dur, sr=DEFAULT_SR, vel=1.0, wave=0, generic=False, volume=15, vol=None, arp=None, pitch=None, vib=0.0, vib_hz=6.0,
              vib_delay=0.15, mode="retro_stylized", bright=0.5):
    rel = 0.01
    p = dict(WSG_PATCH, wave=wave, volume=volume, vol=vol, arp=arp, pitch=pitch, vib=vib, vib_hz=vib_hz, vib_delay=vib_delay)
    y = render_wsg({"v1": [(0.0, dur, float(freq), 1.0, p)]}, dur + rel, sr, generic=generic)
    return _finish(y, freq, sr, vel, mode, bright, rel, dur)


# ============================================================================ NEC HuC6280 (PC Engine)
HU_CLOCK = 3579545.0
HU_NAMES = ("ch1", "ch2", "ch3", "ch4", "ch5", "ch6")
HU_GAIN = 0.35
HU_WAVES = {
    "sine": np.round(15.5 + 15.5 * np.sin(2 * np.pi * _X32 / 32)),
    "saw": np.minimum(_X32 + 0.0, 31),
    "square": np.where(_X32 < 16, 31.0, 0.0),
    "pulse": np.where(_X32 < 8, 31.0, 0.0),
    "triangle": np.where(_X32 < 16, _X32 * 2.0, 62.0 - _X32 * 2),
    "organ": np.clip(np.round(15.5 + 15.5 * (np.sin(2 * np.pi * _X32 / 32) + 0.5 * np.sin(4 * np.pi * _X32 / 32)
                                           + 0.25 * np.sin(6 * np.pi * _X32 / 32)) / 1.6), 0, 31),
}
HU_PATCH = dict(_PATCH, volume=15, wave="square", noise=False, noise_freq=16, bal=None, dda=None, dda_hz=7000.0)
HU_LFO_SHIFT = {1: 0, 2: 2, 3: 4}


def hu_period(hz: float) -> int:
    """12-bit frequency register: the tone is clock / (32 * N); 0 acts as 4096."""
    return int(min(max(round(HU_CLOCK / (32.0 * max(hz, 1e-3))), 1), 4095))


def hu_hz(hz: float) -> float:
    return HU_CLOCK / (32.0 * hu_period(hz))


HU_BAL = (0, 3, 5, 7, 9, 11, 13, 15, 16, 19, 21, 23, 25, 27, 29, 31)


def _hu_amp(v, mbal=15, cbal=15):
    """Channel volume: the 4-bit driver value on the chip's 5-bit register (1.5 dB a step, bit 0 set). The attenuation index adds
    the volume, master balance and channel balance deficits (balance nibbles through HU_BAL, 3 dB a step), silent from 31."""
    c = np.where(v > 0, 2 * v + 1, 0)
    idx = (31 - c) + (31 - HU_BAL[int(mbal) & 15]) + (31 - HU_BAL[int(cbal) & 15])
    return np.where((c > 0) & (idx < 31), 10.0 ** (-1.5 * idx / 20.0), 0.0)


@nb.njit(cache=True)
def _hu_lfsr():
    seed = 1
    s = seed
    bits = np.empty(262144, np.uint8)
    n = 0
    while n < 262144:
        bits[n] = s & 1
        n += 1
        fb = (s ^ (s >> 1) ^ (s >> 11) ^ (s >> 12) ^ (s >> 17)) & 1
        s = (s >> 1) | (fb << 17)
        if s == seed:
            break
    return bits[:n]


_HU_NOISE = None


def hu_noise_bits():
    """One period of the 18-bit noise register (taps 0, 1, 11, 12, 17)."""
    global _HU_NOISE
    if _HU_NOISE is None:
        _HU_NOISE = _hu_lfsr()
    return _HU_NOISE


def render_hu(lanes, dur, sr, lfo_mode=0, lfo_freq=1, master=(15, 15)):
    """`dur` seconds of a HuC6280, stereo (n, 2). Channels 5 and 6 can play noise (`noise`, `noise_freq` 0..31). With `lfo_mode`
    1..3 channel 2's wave steps every `period * lfo_freq` clocks and its sample (shifted by 0, 2 or 4 bits) is added to
    channel 1's period; a note written on ch2 then sets that rate as `hz / lfo_freq`. The channel balance is `bal` (left, right nibbles 0..15, 3 dB a step) or `pan` L | C | R;
    `master` is the master balance nibbles. A note with `dda` (values 0..31 played at `dda_hz`) bypasses the wave buffer."""
    if lfo_mode not in (0, 1, 2, 3):
        raise SpecError("huc6280: lfo_mode must be 0..3")
    lfo_freq = int(lfo_freq) if int(lfo_freq) > 0 else 256
    rate = 2 * sr
    n = int(dur * rate) + 2
    lr = np.zeros((n, 2))
    src = np.full(n, 16.0)
    bits = hu_noise_bits()
    for name in ("ch2", "ch1", "ch3", "ch4", "ch5", "ch6"):
        for t0, t1, hz, vel, p in _notes(lanes, name):
            tr = _note_track(t0, t1, hz, vel, p, HU_CLOCK, hu_period, rate)
            if tr is None:
                continue
            s, per, vol = tr
            if p.get("dda") is not None:
                d = np.clip(np.round(np.asarray(p["dda"], np.float64)), 0, 31)
                if d.ndim != 1 or len(d) < 1:
                    raise SpecError("huc6280: dda is a list of values 0..31")
                k = np.floor(np.arange(len(per)) * float(p.get("dda_hz", 7000.0)) / rate).astype(np.int64)
                wv = d[np.minimum(k, len(d) - 1)] - 16.0
            elif p["noise"]:
                if name not in ("ch5", "ch6"):
                    raise SpecError("huc6280: only ch5 and ch6 have a noise generator")
                step = max((int(p["noise_freq"]) ^ 0x1F) << 6, 1)
                idx = np.floor(_walk(np.full(len(per), HU_CLOCK / rate / step))).astype(np.int64) % len(bits)
                wv = np.where(bits[idx] == 1, 31.0, 0.0) - 16.0
            else:
                tab = _table(p["wave"], HU_WAVES, 32, 0, 31, "huc6280")
                if lfo_mode and name == "ch2":
                    inc = HU_CLOCK / (rate * per * float(lfo_freq))
                elif lfo_mode and name == "ch1":
                    seg = np.pad(src[s:s + len(per)], (0, max(0, len(per) - len(src[s:s + len(per)]))), constant_values=16.0)
                    pm = per + (seg - 16.0) * 2.0 ** HU_LFO_SHIFT[lfo_mode]
                    pm = np.where(pm < 1, 4096.0, pm)
                    inc = HU_CLOCK / (rate * pm)
                else:
                    inc = HU_CLOCK / (rate * per)
                wv = tab[np.floor(_walk(inc)).astype(np.int64) % 32]
                if name == "ch2" and lfo_mode:
                    m = max(0, min(len(wv), n - s))
                    src[s:s + m] = wv[:m]
            bal = p.get("bal")
            if bal is None:
                pan = str(p.get("pan", "C"))
                bal = (15 if pan != "R" else 0, 15 if pan != "L" else 0)
            for ch in (0, 1):                                  # column 0 left, 1 right
                _put(lr[:, ch], s, wv / 16.0 * _hu_amp(vol, master[ch], bal[ch]))
    out = np.stack([_to_sr(lr[:, 0], rate, sr), _to_sr(lr[:, 1], rate, sr)], axis=1)
    out = lfilter(*butter(1, 20.0, "high", fs=sr), out, axis=0)
    return np.stack([_fit(out[:, 0], samples(dur, sr)), _fit(out[:, 1], samples(dur, sr))], axis=1) * HU_GAIN


def _hu_layer(layer, sr, q):
    lanes, end = lanes_of(layer, "huc6280", HU_NAMES, HU_PATCH, q)
    return render_hu(lanes, end, sr, lfo_mode=int(layer.get("lfo_mode", 0)), lfo_freq=int(layer.get("lfo_freq", 1)),
                     master=tuple(layer.get("master", (15, 15))))


CT.SYSTEM_LAYERS["huc6280"] = _hu_layer
CT.SYSTEMS["huc6280"] = HU_NAMES


@instrument("huc6280", "NEC HuC6280 (PC Engine) channel: a 32-sample, 5-bit wave, 12-bit frequency, 5-bit log volume; noise on channels 5-6; "
            "optional LFO (channel 2's wave modulating the period).",
            family="retro", span=("C1", "C7"),
            wave=("square", "square | saw | pulse | triangle | sine | organ, or a list of 32 values 0..31"),
            volume=(15, "0..15 driver steps on the 5-bit register, 3 dB each"),
            noise=(False, "true: the 18-bit noise register instead of the wave (channels 5 and 6 only on the chip)"),
            noise_freq=(16, "0..31, higher = faster: clock / (64 * (noise_freq xor 31))"),
            lfo_mode=(0, "0 off | 1 | 2 | 3: the LFO wave is added to the period shifted by 0 | 2 | 4 bits"),
            lfo_hz=(6.0, "LFO rate in Hz (lfo_mode > 0)"),
            vol=(None, _H["vol"]), arp=(None, _H["arp"]), pitch=(None, _H["pitch"]),
            vib=(0.0, _H["vib"]), vib_hz=(6.0, _H["vib_hz"]), vib_delay=(0.15, _H["vib_delay"]), mode=CT._MODE, bright=CT._BRIGHT)
def huc6280(freq, dur, sr=DEFAULT_SR, vel=1.0, wave="square", volume=15, noise=False, noise_freq=16, lfo_mode=0, lfo_hz=6.0,
            vol=None, arp=None, pitch=None, vib=0.0, vib_hz=6.0, vib_delay=0.15, mode="retro_stylized", bright=0.5):
    rel = 0.01
    p = dict(HU_PATCH, wave=wave, volume=volume, noise=bool(noise), noise_freq=int(noise_freq), vol=vol, arp=arp, pitch=pitch,
             vib=vib, vib_hz=vib_hz, vib_delay=vib_delay)
    lanes = {"ch5" if noise else "ch1": [(0.0, dur, float(freq), 1.0, p)]}
    lf = 16
    if lfo_mode and not noise:
        lanes["ch2"] = [(0.0, dur, float(lfo_hz) * lf, 1.0, dict(HU_PATCH, wave="sine", volume=0))]
    y = render_hu(lanes, dur + rel, sr, lfo_mode=int(lfo_mode), lfo_freq=lf)
    return _finish(y[:, 0] + y[:, 1], freq, sr, vel, mode, bright, rel, dur, tonal=not noise) * 0.5
