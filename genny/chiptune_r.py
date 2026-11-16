"""Retro chips, second batch: NES expansions (Konami VRC6, Namco 163, Famicom Disk System), Commodore Amiga Paula with a
ProTracker MOD player, Yamaha OPL2 (YM3812) and OPL3 (YMF262).

Registers
    chip systems  `vrc6` (p1, p2, saw), `n163` (ch1 .. ch8), `fds` (fds), `paula` (ch1 .. ch4), `opl2` (ch1 .. ch9, rhythm
                  lanes bd sd tom tc hh with `rhythm`), `opl3` (ch1 .. ch18, same)
    instruments   `vrc6`, `n163`, `fds`, `paula`, `opl2`, `opl3`
    functions     `render_mod`, `mod_parse` (ProTracker file player, offline)
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np
from scipy.signal import butter, lfilter

from . import chiptune as CT
from . import filters as F
from .chiptune2 import _H, lanes_of
from .chiptune_q import _PATCH, _finish, _fit, _notes, _note_track, _put, _walk
from .core import DEFAULT_SR, samples
from .instruments import instrument
from .retro import _to_sr
from .spec import SpecError

CPU_NTSC = 1789772.7272727
_X = np.arange(256)


def _hp(y, sr, axis=0):
    return lfilter(*butter(1, 20.0, "high", fs=sr), y, axis=axis)


def _nib_wave(w, length, hi, what):
    """A wave of `length` steps with values 0..hi: from a name (see `names`), a number, or a list of values."""
    if isinstance(w, (list, tuple, np.ndarray)):
        a = np.asarray(w, np.float64)
        if a.shape != (length,):
            raise SpecError(f"{what}: this wave has {length} values")
        return np.clip(np.round(a), 0, hi)
    x = np.arange(length) / length
    tab = {"sine": 0.5 + 0.5 * np.sin(2 * np.pi * x), "saw": x, "square": (x < 0.5) * 1.0, "pulse": (x < 0.25) * 1.0,
           "triangle": 1.0 - np.abs(2 * x - 1.0), "organ": 0.5 + 0.5 * (np.sin(2 * np.pi * x) + 0.5 * np.sin(4 * np.pi * x)
                                                                       + 0.25 * np.sin(6 * np.pi * x)) / 1.75}
    if str(w).lower() not in tab:
        raise SpecError(f"{what}: wave must be one of {sorted(tab)} or a list of {length} values")
    return np.clip(np.round(tab[str(w).lower()] * hi), 0, hi)


# ============================================================================ Konami VRC6
VRC6_NAMES = ("p1", "p2", "saw")
VRC6_GAIN = 0.3
VRC6_PATCH = dict(_PATCH, volume=15, duty=4)


VRC6_SHIFT = {0: 1.0, 4: 16.0, 8: 256.0}


def vrc6_period(hz: float, voice: str = "p1", shift: int = 0) -> int:
    """12-bit period register: the pulses run at cpu / (16 (P + 1)), the saw at cpu / (14 (P + 1)); `shift` 4 or 8 is the
    frequency scaling register (x16, x256)."""
    div = 14.0 if voice == "saw" else 16.0
    return int(min(max(round(CPU_NTSC * VRC6_SHIFT[int(shift)] / (div * max(hz, 1e-3))) - 1, 0), 4095))


def vrc6_hz(hz: float, voice: str = "p1", shift: int = 0) -> float:
    return CPU_NTSC * VRC6_SHIFT[int(shift)] / ((14.0 if voice == "saw" else 16.0) * (vrc6_period(hz, voice, shift) + 1))


def render_vrc6(lanes, dur, sr, shift=0):
    """`dur` seconds of a VRC6. Pulses: 16-step sequencer, high for duty + 1 steps (duty 0..7, 4-bit volume); saw: 14 steps,
    a 6-bit rate added to an 8-bit accumulator on every even step (6 adds, then reset), top 5 bits out. The driver volume of
    the saw sets that rate (0..42, the largest that does not wrap)."""
    rate = 2 * sr
    out = np.zeros(int(dur * rate) + 2)
    for name in VRC6_NAMES:
        for t0, t1, hz, vel, p in _notes(lanes, name):
            tr = _note_track(t0, t1, hz, vel, p, CPU_NTSC, lambda f, n=name: vrc6_period(f, n, shift), rate)
            if tr is None:
                continue
            s, per, vol = tr
            inc = CPU_NTSC * VRC6_SHIFT[int(shift)] / (rate * (per + 1.0))
            step = np.floor(_walk(inc)).astype(np.int64)
            if name == "saw":
                a = np.round(vol / 15.0 * 42.0).astype(np.int64)
                x = (((a * ((step % 14) // 2)) & 255) >> 3) / 31.0
            else:
                d = int(p["duty"])
                if not 0 <= d <= 7:
                    raise SpecError("vrc6: duty must be 0..7")
                x = ((step % 16) <= d) * (vol / 15.0)
            _put(out, s, x)
    y = _hp(_to_sr(out, rate, sr), sr)
    return _fit(y, samples(dur, sr)) * VRC6_GAIN


def _vrc6_layer(layer, sr, q):
    lanes, end = lanes_of(layer, "vrc6", VRC6_NAMES, VRC6_PATCH, q)
    return render_vrc6(lanes, end, sr, shift=int(layer.get("shift", 0)))


CT.SYSTEM_LAYERS["vrc6"] = _vrc6_layer
CT.SYSTEMS["vrc6"] = VRC6_NAMES


@instrument("vrc6", "Konami VRC6 (Castlevania III, Japan) voice: pulse with 8 duty settings, or the 14-step saw whose level is a 6-bit rate.",
            family="retro", span=("C1", "C7"),
            wave=("pulse", "pulse | saw"), duty=(4, "0..7: the pulse is high for duty + 1 of 16 steps"),
            volume=(15, "0..15 (saw: sets the accumulator rate, 0..42)"), vol=(None, _H["vol"]), arp=(None, _H["arp"]),
            pitch=(None, _H["pitch"]), vib=(0.0, _H["vib"]), vib_hz=(6.0, _H["vib_hz"]), vib_delay=(0.15, _H["vib_delay"]),
            mode=CT._MODE, bright=CT._BRIGHT)
def vrc6(freq, dur, sr=DEFAULT_SR, vel=1.0, wave="pulse", duty=4, volume=15, vol=None, arp=None, pitch=None, vib=0.0,
         vib_hz=6.0, vib_delay=0.15, mode="retro_stylized", bright=0.5):
    if wave not in ("pulse", "saw"):
        raise SpecError("vrc6: wave must be pulse or saw")
    rel = 0.01
    p = dict(VRC6_PATCH, duty=duty, volume=volume, vol=vol, arp=arp, pitch=pitch, vib=vib, vib_hz=vib_hz, vib_delay=vib_delay)
    y = render_vrc6({"saw" if wave == "saw" else "p1": [(0.0, dur, float(freq), 1.0, p)]}, dur + rel, sr)
    return _finish(y, freq, sr, vel, mode, bright, rel, dur)


# ============================================================================ Namco 163
N163_NAMES = tuple(f"ch{i}" for i in range(1, 9))
N163_GAIN = 0.6
N163_PATCH = dict(_PATCH, volume=15, wave="sine", length=32)


def n163_freq_reg(hz: float, n: int = 1, length: int = 32) -> int:
    """18-bit frequency register: the wave repeats at cpu * F / (15 * 65536 * n * length) with `n` channels active."""
    return int(min(max(round(hz * 15.0 * 65536.0 * n * length / CPU_NTSC), 0), 2 ** 18 - 1))


def n163_hz(hz: float, n: int = 1, length: int = 32) -> float:
    return CPU_NTSC * n163_freq_reg(hz, n, length) / (15.0 * 65536.0 * n * length)


def render_n163(lanes, dur, sr, n=None):
    """`dur` seconds of an N163. Each channel is updated once every 15 * n cycles (n = channels in use, default the lanes
    given) by a 24-bit phase accumulator; wave RAM holds 4-bit samples, `length` 4..256 (multiple of 4) per channel.
    The channels are time-multiplexed, so the output is the mean of the channel outputs (level falls as 1 / n)."""
    rate = 2 * sr
    n = int(n) if n else max(1, len([k for k in N163_NAMES if lanes.get(k)]))
    if not 1 <= n <= 8:
        raise SpecError("n163: 1..8 channels")
    out = np.zeros(int(dur * rate) + 2)
    for name in N163_NAMES:
        for t0, t1, hz, vel, p in _notes(lanes, name):
            ln = len(p["wave"]) if isinstance(p["wave"], (list, tuple, np.ndarray)) else int(p["length"])
            if not 4 <= ln <= 256 or ln % 4:
                raise SpecError("n163: wave length is 4..256, a multiple of 4")
            tab = (_nib_wave(p["wave"], ln, 15, "n163") - 8.0) / 8.0
            tr = _note_track(t0, t1, hz, vel, p, CPU_NTSC, lambda f: n163_freq_reg(f, n, ln), rate)
            if tr is None:
                continue
            s, reg, vol = tr
            inc = reg * CPU_NTSC / (15.0 * n * 65536.0 * rate)
            _put(out, s, tab[np.floor(_walk(inc)).astype(np.int64) % ln] * (vol / 15.0))
    y = _hp(_to_sr(out / n, rate, sr), sr)
    return _fit(y, samples(dur, sr)) * N163_GAIN


def _n163_layer(layer, sr, q):
    lanes, end = lanes_of(layer, "n163", N163_NAMES, N163_PATCH, q)
    return render_n163(lanes, end, sr, n=layer.get("n"))


CT.SYSTEM_LAYERS["n163"] = _n163_layer
CT.SYSTEMS["n163"] = N163_NAMES


@instrument("n163", "Namco 163 channel: 4-bit wave RAM of 4..256 steps, 18-bit frequency, 15 * channels cycles per update; `channels` sets the multiplex count.",
            family="retro", span=("C1", "C7"),
            wave=("sine", "sine | saw | square | pulse | triangle | organ, or a list of 4..256 values 0..15 (multiple of 4)"),
            length=(32, "wave steps when `wave` is a name, 4..256 (multiple of 4)"),
            channels=(1, "channels in use on the chip, 1..8 (sets the pitch of a given register and the level)"),
            volume=(15, "0..15, linear"), vol=(None, _H["vol"]), arp=(None, _H["arp"]), pitch=(None, _H["pitch"]),
            vib=(0.0, _H["vib"]), vib_hz=(6.0, _H["vib_hz"]), vib_delay=(0.15, _H["vib_delay"]), mode=CT._MODE, bright=CT._BRIGHT)
def n163(freq, dur, sr=DEFAULT_SR, vel=1.0, wave="sine", length=32, channels=1, volume=15, vol=None, arp=None, pitch=None,
         vib=0.0, vib_hz=6.0, vib_delay=0.15, mode="retro_stylized", bright=0.5):
    rel = 0.01
    p = dict(N163_PATCH, wave=wave, length=length, volume=volume, vol=vol, arp=arp, pitch=pitch, vib=vib, vib_hz=vib_hz,
             vib_delay=vib_delay)
    y = render_n163({"ch1": [(0.0, dur, float(freq), 1.0, p)]}, dur + rel, sr, n=channels)
    return _finish(y * channels, freq, sr, vel, mode, bright, rel, dur)


# ============================================================================ Famicom Disk System
FDS_NAMES = ("fds",)
FDS_GAIN = 0.5
FDS_MASTER = (1.0, 2.0 / 3.0, 0.5, 0.4)
FDS_MOD_STEP = np.array([0, 1, 2, 4, 0, -4, -2, -1])
_FX = np.arange(32)
FDS_MOD_TABLES = {
    "none": np.zeros(32, np.int64),
    "tri": np.array([2] * 8 + [6] * 16 + [2] * 8, np.int64),
    "saw": np.array([2] * 16 + [4] + [0] * 15, np.int64),
}
FDS_PATCH = dict(_PATCH, volume=15, wave="sine", master=0, mod_gain=0, mod_hz=5.0, mod_table="tri", mod_counter=0)


def fds_freq_reg(hz: float) -> int:
    """12-bit frequency register: the 64-step wave repeats at cpu * F / 2**22."""
    return int(min(max(round(hz * 2 ** 22 / CPU_NTSC), 0), 4095))


def fds_hz(hz: float) -> float:
    return CPU_NTSC * fds_freq_reg(hz) / 2 ** 22


def fds_mod_pitch(f: int, counter: int, gain: int) -> int:
    """The pitch the modulation unit hands the accumulator: counter (-64..63) times the 6-bit gain, scaled, then F * that / 64."""
    t = counter * gain
    rem = t & 15
    t >>= 4
    if rem and not t & 0x80:
        t += -1 if counter < 0 else 2
    if t >= 192:
        t -= 256
    elif t < -64:
        t += 256
    t2 = f * t
    rem = t2 & 63
    t2 >>= 6
    if rem >= 32:
        t2 += 1
    return f + t2


@nb.njit(cache=True)
def _fds_core(n, freg, mreg, gain, wave, mtab, rate, cpu, counter0):
    out = np.empty(n)
    ph = 0.0
    mph = 0.0
    mstep = 0
    counter = counter0
    eff = freg[0]
    mstepinc = cpu / (65536.0 * rate)
    for i in range(n):
        f = freg[i]
        if gain > 0:
            mph += mreg[i] * mstepinc
            while mph >= 1.0:
                mph -= 1.0
                v = mtab[(mstep >> 1) & 31]
                mstep += 1
                if v == 0:
                    pass
                elif v == 1:
                    counter += 1
                elif v == 2:
                    counter += 2
                elif v == 3:
                    counter += 4
                elif v == 4:
                    counter = 0
                elif v == 5:
                    counter -= 4
                elif v == 6:
                    counter -= 2
                else:
                    counter -= 1
                counter = ((counter + 64) & 127) - 64
            t = counter * gain
            rem = t & 15
            t >>= 4
            if rem != 0 and (t & 0x80) == 0:
                t += -1 if counter < 0 else 2
            if t >= 192:
                t -= 256
            elif t < -64:
                t += 256
            t2 = f * t
            rem2 = t2 & 63
            t2 >>= 6
            if rem2 >= 32:
                t2 += 1
            eff = f + t2
        else:
            eff = f
        if eff < 0:
            eff = 0
        ph += eff * mstepinc
        out[i] = wave[int(ph) & 63]
    return out


def render_fds(lanes, dur, sr):
    """`dur` seconds of the FDS channel: 64-step 6-bit wave, 12-bit frequency, volume envelope 0..32 (the driver volume,
    doubled) with the master volume (`master` 0..3 = 100 / 67 / 50 / 40 %), and the modulator: a 32-entry table of 3-bit
    codes (0, +1, +2, +4, reset, -4, -2, -1) steps a 7-bit counter at `mod_hz` cycles per second, and counter * `mod_gain`
    (0..63) bends the pitch."""
    rate = 2 * sr
    out = np.zeros(int(dur * rate) + 2)
    for t0, t1, hz, vel, p in _notes(lanes, "fds"):
        tr = _note_track(t0, t1, hz, vel, p, CPU_NTSC, fds_freq_reg, rate)
        if tr is None:
            continue
        s, reg, vol = tr
        wave = (_nib_wave(p["wave"], 64, 63, "fds") - 32.0) / 32.0
        mt = p["mod_table"]
        if isinstance(mt, str):
            if mt not in FDS_MOD_TABLES:
                raise SpecError(f"fds: mod_table must be one of {sorted(FDS_MOD_TABLES)} or 32 values 0..7")
            mt = FDS_MOD_TABLES[mt]
        mt = np.asarray(mt, np.int64)
        if mt.shape != (32,) or mt.min() < 0 or mt.max() > 7:
            raise SpecError("fds: mod_table has 32 values 0..7")
        g = int(p["mod_gain"])
        if not 0 <= g <= 63 or int(p["master"]) not in (0, 1, 2, 3):
            raise SpecError("fds: mod_gain 0..63, master 0..3")
        mreg = np.full(len(reg), min(max(round(float(p["mod_hz"]) * 2 ** 22 / CPU_NTSC), 0), 4095), np.float64)
        c0 = int(p["mod_counter"])
        if not -64 <= c0 <= 63:
            raise SpecError("fds: mod_counter -64..63")
        y = _fds_core(len(reg), reg.astype(np.int64), mreg, g, wave, mt, float(rate), CPU_NTSC, c0)
        _put(out, s, y * (np.minimum(vol * 2, 32) / 32.0) * FDS_MASTER[int(p["master"])])
    y = _hp(_to_sr(out, rate, sr), sr)
    return _fit(y, samples(dur, sr)) * FDS_GAIN


def _fds_layer(layer, sr, q):
    lanes, end = lanes_of(layer, "fds", FDS_NAMES, FDS_PATCH, q)
    return render_fds(lanes, end, sr)


CT.SYSTEM_LAYERS["fds"] = _fds_layer
CT.SYSTEMS["fds"] = FDS_NAMES


@instrument("fds", "Famicom Disk System channel: a 64-step 6-bit wave, 12-bit frequency, volume 0..32, master volume and a table-driven pitch modulator.",
            family="retro", span=("C1", "C7"),
            wave=("sine", "sine | saw | square | pulse | triangle | organ, or a list of 64 values 0..63"),
            master=(0, "0..3: 100 / 67 / 50 / 40 % of the channel level"),
            mod_gain=(0, "0..63, modulator depth (0 = off)"), mod_hz=(5.0, "modulator table cycles per second (0 = the modulator halted)"),
            mod_counter=(0, "-64..63: the modulation counter's starting value"),
            mod_table=("tri", "none | tri | saw, or 32 codes 0..7 (0, +1, +2, +4, reset, -4, -2, -1 on the counter)"),
            volume=(15, "0..15 driver steps (doubled onto the chip's 0..32)"), vol=(None, _H["vol"]), arp=(None, _H["arp"]),
            pitch=(None, _H["pitch"]), vib=(0.0, _H["vib"]), vib_hz=(6.0, _H["vib_hz"]), vib_delay=(0.15, _H["vib_delay"]),
            mode=CT._MODE, bright=CT._BRIGHT)
def fds(freq, dur, sr=DEFAULT_SR, vel=1.0, wave="sine", master=0, mod_gain=0, mod_hz=5.0, mod_counter=0, mod_table="tri",
        volume=15, vol=None, arp=None, pitch=None, vib=0.0, vib_hz=6.0, vib_delay=0.15, mode="retro_stylized", bright=0.5):
    rel = 0.01
    p = dict(FDS_PATCH, wave=wave, master=master, mod_gain=mod_gain, mod_hz=mod_hz, mod_counter=mod_counter, mod_table=mod_table,
             volume=volume, vol=vol,
             arp=arp, pitch=pitch, vib=vib, vib_hz=vib_hz, vib_delay=vib_delay)
    y = render_fds({"fds": [(0.0, dur, float(freq), 1.0, p)]}, dur + rel, sr)
    return _finish(y, freq, sr, vel, mode, bright, rel, dur)


# ============================================================================ Commodore Amiga Paula
PAULA_CLOCK = {"pal": 3546895.0, "ntsc": 3579545.0}
PAULA_MIN_PERIOD = 113                          # ProTracker slide floor (B-3), the default clamp
PAULA_DMA_MIN = {"pal": 123, "ntsc": 124}       # shortest period the four-voice DMA can fetch
PAULA_NAMES = ("ch1", "ch2", "ch3", "ch4")
PAULA_SIDE = (0, 1, 1, 0)                      # channels 0 and 3 left, 1 and 2 right
PAULA_GAIN = 0.5
PAULA_FILTERS = {"a500": 4900.0, "a1200": 34000.0, "none": None}      # fixed RC low-pass, Hz
PAULA_LED_HZ = 3300.0                           # the switchable 12 dB/oct "LED" filter
_X32 = np.arange(32)
PAULA_WAVES = {
    "sine": np.round(127 * np.sin(2 * np.pi * _X32 / 32)),
    "saw": np.round(-128 + 255 * _X32 / 31),
    "square": np.where(_X32 < 16, 127.0, -128.0),
    "triangle": np.round(np.where(_X32 < 16, -128 + 255 * _X32 / 15, 127 - 255 * (_X32 - 16) / 15)),
    "noise": np.random.default_rng(1).integers(-128, 128, 32).astype(np.float64),
}
PAULA_PATCH = dict(_PATCH, volume=64, wave="saw", modulates=None, loop=True)


def paula_period(hz: float, length: int = 32, system: str = "pal", dma: bool = False) -> int:
    """Period register, in colour clocks: one sample byte every `period` cycles, so a `length`-byte cycle plays at
    clock / (period * length). `dma=True` floors the period at what the DMA can fetch (123 PAL, 124 NTSC) instead of 113."""
    pmin = PAULA_DMA_MIN[system] if dma else PAULA_MIN_PERIOD
    return int(min(max(round(PAULA_CLOCK[system] / (max(hz, 1e-3) * length)), pmin), 65535))


def paula_hz(hz: float, length: int = 32, system: str = "pal", dma: bool = False) -> float:
    return PAULA_CLOCK[system] / (paula_period(hz, length, system, dma) * length)


@nb.njit(cache=True)
def _paula_chan(n, per, vol, trig, toff, bank, offs, lens, ls, ll, clk, rate):
    """One Paula channel: byte data read at clk / period, held between reads (no interpolation). trig[i] = s + 1 starts sample s
    at its first byte; a sample with loop length > 0 repeats its loop, otherwise it ends and the channel goes silent."""
    out = np.zeros(n)
    cur = -1
    pos = 0.0
    for i in range(n):
        if trig[i] > 0:
            cur = trig[i] - 1
            pos = toff[i]
        if cur < 0 or per[i] <= 0.0:
            continue
        end = lens[cur] if ll[cur] <= 0 else ls[cur] + ll[cur]
        if pos >= end:
            if ll[cur] > 0:
                pos = ls[cur] + (pos - ls[cur]) % ll[cur]
            else:
                cur = -1
                continue
        out[i] = bank[offs[cur] + int(pos)] * vol[i]
        pos += clk / (per[i] * rate)
    return out


def paula_filter(y, sr, kind="a500", led=False):
    """The Paula output stage on `y` (axis 0): the model's fixed RC low-pass (a500 4.9 kHz, a1200 34 kHz), then the LED filter."""
    if kind not in PAULA_FILTERS:
        raise SpecError(f"paula: filter must be one of {sorted(PAULA_FILTERS)}")
    fc = PAULA_FILTERS[kind]
    if fc and fc < sr * 0.45:
        y = lfilter(*butter(1, fc, fs=sr), y, axis=0)
    if led:
        y = lfilter(*butter(2, PAULA_LED_HZ, fs=sr), y, axis=0)
    return y


def render_paula(lanes, dur, sr, system="pal", filter="a500", led=False):
    """`dur` seconds of Paula, stereo (n, 2). A note's `wave` is a name or a list of signed bytes (8-bit samples); its period
    register holds the pitch, `volume` 0..64 the level. `modulates` "am" or "fm" on a channel feeds its data, as 16-bit words,
    into the volume or the period of the next channel (the ADKCON bits); the modulating channel itself is silent."""
    if system not in PAULA_CLOCK:
        raise SpecError("paula: system pal | ntsc")
    clk = PAULA_CLOCK[system]
    rate = 2.0 * sr
    n = int(dur * rate) + 2
    chans = np.zeros((4, n))
    held = {}
    for ci, name in enumerate(PAULA_NAMES):
        for t0, t1, hz, vel, p in _notes(lanes, name):
            w = p["wave"]
            if isinstance(w, str):
                if w not in PAULA_WAVES:
                    raise SpecError(f"paula: wave must be one of {sorted(PAULA_WAVES)} or a list of bytes")
                w = PAULA_WAVES[w]
            data = np.round(np.asarray(w, np.float64))
            if data.ndim != 1 or len(data) < 2:
                raise SpecError("paula: wave is a list of at least 2 bytes")
            ln = len(data)
            mode = p.get("modulates")
            if mode not in (None, "am", "fm"):
                raise SpecError("paula: modulates is am | fm")
            tr = _note_track(t0, t1, hz, vel, p, clk, lambda f: paula_period(f, ln, system, bool(p.get("dma"))), rate)
            if tr is None:
                continue
            s, per, vol = tr
            m = len(per)
            src = None
            for ent in held.get(ci - 1, []):
                if ent[0] <= s < ent[1]:
                    src = ent
            vcur = vol / 15.0 * float(p["volume"]) / 64.0 / 128.0
            pcur = per.astype(np.float64)
            if src is not None:
                seg = src[3][s - src[0]:s - src[0] + m]
                seg = np.pad(seg, (0, m - len(seg)))
                if src[2] == "am":
                    vcur = np.clip(seg, 0, 64) / 64.0 / 128.0
                else:
                    pcur = np.where(seg >= PAULA_MIN_PERIOD, seg, pcur)
            trig = np.zeros(m, np.int64)
            trig[0] = 1
            raw = _paula_chan(m, pcur, np.ones(m) if mode else vcur, trig, np.zeros(m), data, np.array([0]), np.array([ln]),
                              np.array([0]), np.array([ln if p.get("loop", True) else 0]), clk, rate)
            if mode:
                held.setdefault(ci, []).append((s, s + m, mode, raw))
            else:
                _put(chans[ci], s, raw)
    left = chans[0] + chans[3]
    right = chans[1] + chans[2]
    y = np.stack([_to_sr(left, rate, sr), _to_sr(right, rate, sr)], axis=1)
    y = paula_filter(_hp(y, sr), sr, filter, led)
    return np.stack([_fit(y[:, 0], samples(dur, sr)), _fit(y[:, 1], samples(dur, sr))], axis=1) * PAULA_GAIN


def _paula_layer(layer, sr, q):
    lanes, end = lanes_of(layer, "paula", PAULA_NAMES, PAULA_PATCH, q)
    return render_paula(lanes, end, sr, system=str(layer.get("clock", "pal")), filter=str(layer.get("filter", "a500")),
                        led=bool(layer.get("led", False)))


CT.SYSTEM_LAYERS["paula"] = _paula_layer
CT.SYSTEMS["paula"] = PAULA_NAMES


@instrument("paula", "Commodore Amiga Paula voice: 8-bit sample data read by DMA at clock / period (PAL 3.546895 MHz, NTSC 3.579545 MHz), volume 0..64, A500 / A1200 filter and LED filter.",
            family="retro", span=("C1", "C6"),
            wave=("saw", "saw | sine | square | triangle | noise, or a list of signed bytes (-128..127), played as a loop"),
            volume=(64, "0..64"), clock=("pal", "pal | ntsc"), filter=("a500", "a500 | a1200 | none (fixed RC output stage)"),
            led=(False, "the 3.3 kHz LED filter"), vol=(None, _H["vol"]), arp=(None, _H["arp"]), pitch=(None, _H["pitch"]),
            vib=(0.0, _H["vib"]), vib_hz=(6.0, _H["vib_hz"]), vib_delay=(0.15, _H["vib_delay"]), mode=CT._MODE, bright=CT._BRIGHT)
def paula(freq, dur, sr=DEFAULT_SR, vel=1.0, wave="saw", volume=64, clock="pal", filter="a500", led=False, vol=None, arp=None,
          pitch=None, vib=0.0, vib_hz=6.0, vib_delay=0.15, mode="retro_stylized", bright=0.5):
    rel = 0.01
    p = dict(PAULA_PATCH, wave=wave, volume=volume, vol=vol, arp=arp, pitch=pitch, vib=vib, vib_hz=vib_hz, vib_delay=vib_delay)
    y = render_paula({"ch1": [(0.0, dur, float(freq), 1.0, p)]}, dur + rel, sr, system=clock, filter=filter, led=led)
    return _finish(y[:, 0] + y[:, 1], freq, sr, vel, mode, bright, rel, dur)


# ---------------------------------------------------------------------------- ProTracker MOD
PT_PERIODS = (
    (856, 808, 762, 720, 678, 640, 604, 570, 538, 508, 480, 453, 428, 404, 381, 360, 339, 320, 302, 285, 269, 254, 240, 226, 214, 202, 190, 180, 170, 160, 151, 143, 135, 127, 120, 113),
    (850, 802, 757, 715, 674, 637, 601, 567, 535, 505, 477, 450, 425, 401, 379, 357, 337, 318, 300, 284, 268, 253, 239, 225, 213, 201, 189, 179, 169, 159, 150, 142, 134, 126, 119, 113),
    (844, 796, 752, 709, 670, 632, 597, 563, 532, 502, 474, 447, 422, 398, 376, 355, 335, 316, 298, 282, 266, 251, 237, 224, 211, 199, 188, 177, 167, 158, 149, 141, 133, 125, 118, 112),
    (838, 791, 746, 704, 665, 628, 592, 559, 528, 498, 470, 444, 419, 395, 373, 352, 332, 314, 296, 280, 264, 249, 235, 222, 209, 198, 187, 176, 166, 157, 148, 140, 132, 125, 118, 111),
    (832, 785, 741, 699, 660, 623, 588, 555, 524, 495, 467, 441, 416, 392, 370, 350, 330, 312, 294, 278, 262, 247, 233, 220, 208, 196, 185, 175, 165, 156, 147, 139, 131, 124, 117, 110),
    (826, 779, 736, 694, 655, 619, 584, 551, 520, 491, 463, 437, 413, 390, 368, 347, 328, 309, 292, 276, 260, 245, 232, 219, 206, 195, 184, 174, 164, 155, 146, 138, 130, 123, 116, 109),
    (820, 774, 730, 689, 651, 614, 580, 547, 516, 487, 460, 434, 410, 387, 365, 345, 325, 307, 290, 274, 258, 244, 230, 217, 205, 193, 183, 172, 163, 154, 145, 137, 129, 122, 115, 109),
    (814, 768, 725, 684, 646, 610, 575, 543, 513, 484, 457, 431, 407, 384, 363, 342, 323, 305, 288, 272, 256, 242, 228, 216, 204, 192, 181, 171, 161, 152, 144, 136, 128, 121, 114, 108),
    (907, 856, 808, 762, 720, 678, 640, 604, 570, 538, 508, 480, 453, 428, 404, 381, 360, 339, 320, 302, 285, 269, 254, 240, 226, 214, 202, 190, 180, 170, 160, 151, 143, 135, 127, 120),
    (900, 850, 802, 757, 715, 675, 636, 601, 567, 535, 505, 477, 450, 425, 401, 379, 357, 337, 318, 300, 284, 268, 253, 238, 225, 212, 200, 189, 179, 169, 159, 150, 142, 134, 126, 119),
    (894, 844, 796, 752, 709, 670, 632, 597, 563, 532, 502, 474, 447, 422, 398, 376, 355, 335, 316, 298, 282, 266, 251, 237, 223, 211, 199, 188, 177, 167, 158, 149, 141, 133, 125, 118),
    (887, 838, 791, 746, 704, 665, 628, 592, 559, 528, 498, 470, 444, 419, 395, 373, 352, 332, 314, 296, 280, 264, 249, 235, 222, 209, 198, 187, 176, 166, 157, 148, 140, 132, 125, 118),
    (881, 832, 785, 741, 699, 660, 623, 588, 555, 524, 494, 467, 441, 416, 392, 370, 350, 330, 312, 294, 278, 262, 247, 233, 220, 208, 196, 185, 175, 165, 156, 147, 139, 131, 123, 117),
    (875, 826, 779, 736, 694, 655, 619, 584, 551, 520, 491, 463, 437, 413, 390, 368, 347, 328, 309, 292, 276, 260, 245, 232, 219, 206, 195, 184, 174, 164, 155, 146, 138, 130, 123, 116),
    (868, 820, 774, 730, 689, 651, 614, 580, 547, 516, 487, 460, 434, 410, 387, 365, 345, 325, 307, 290, 274, 258, 244, 230, 217, 205, 193, 183, 172, 163, 154, 145, 137, 129, 122, 115),
    (862, 814, 768, 725, 684, 646, 610, 575, 543, 513, 484, 457, 431, 407, 384, 363, 342, 323, 305, 288, 272, 256, 242, 228, 216, 203, 192, 181, 171, 161, 152, 144, 136, 128, 121, 114),
)
PT_VIBRATO = (0, 24, 49, 74, 97, 120, 141, 161, 180, 197, 212, 224, 235, 244, 250, 253, 255, 253, 250, 244, 235, 224, 212, 197, 180, 161, 141, 120, 97, 74, 49, 24)

_PT_BASE = PT_PERIODS[0]
_PT_SINE = np.array(PT_VIBRATO)
_MOD_CH = {b"M.K.": 4, b"M!K!": 4, b"FLT4": 4, b"4CHN": 4, b"6CHN": 6, b"8CHN": 8, b"FLT8": 8, b"CD81": 8, b"OKTA": 8}


def _pt_table(ft):
    """Period table of finetune ft (-8..7): one of the 16 tables of 36 periods."""
    return np.array(PT_PERIODS[int(ft) & 15])


def _pt_wave(kind, pos):
    """Magnitude 0..255 of the vibrato / tremolo waveform at byte position `pos`: 0 sine, 1 ramp, 2 and 3 square."""
    i = (pos >> 2) & 31
    if kind == 0:
        return int(PT_VIBRATO[i])
    if kind == 1:
        return 255 - i * 8 if pos & 128 else i * 8
    return 255


def mod_parse(src):
    """A ProTracker file (bytes or path) -> dict(title, nch, order, patterns [n][64][nch] of (sample, period, effect, param),
    samples [dict(name, data, volume, finetune, loop_start, loop_len)])."""
    blob = bytes(src) if isinstance(src, (bytes, bytearray, memoryview)) else open(src, "rb").read()
    tag = blob[1080:1084]
    nch = _MOD_CH.get(tag) or (int(tag[:2]) if tag[2:] == b"CH" and tag[:2].isdigit() else 0)
    if not nch:
        raise SpecError("mod: only 31-sample files are read (tag M.K., M!K!, FLT4, 4CHN, 6CHN, 8CHN, xxCH)")
    heads = []
    for i in range(31):
        h = blob[20 + 30 * i:50 + 30 * i]
        ft = h[24] & 15
        heads.append(dict(name=h[:22].split(b"\0")[0].decode("latin-1"), words=(h[22] << 8) | h[23], finetune=ft - 16 if ft > 7 else ft,
                          volume=min(h[25], 64), loop_start=2 * ((h[26] << 8) | h[27]), loop_len=2 * ((h[28] << 8) | h[29])))
    ordr = list(blob[952:952 + min(blob[950], 128)])
    npat = max(blob[952:1080]) + 1
    pats = []
    for pi in range(npat):
        base = 1084 + pi * 64 * nch * 4
        rows = []
        for r in range(64):
            row = []
            for c in range(nch):
                b = blob[base + (r * nch + c) * 4:base + (r * nch + c) * 4 + 4].ljust(4, b"\0")
                row.append(((b[0] & 0xF0) | (b[2] >> 4), ((b[0] & 15) << 8) | b[1], b[2] & 15, b[3]))
            rows.append(row)
        pats.append(rows)
    at = 1084 + npat * 64 * nch * 4
    for h in heads:
        ln = h["words"] * 2
        h["data"] = np.frombuffer(blob[at:at + ln], np.int8).astype(np.float64)
        at += ln
        if h["loop_len"] <= 2:
            h["loop_len"] = 0
        h["loop_start"] = min(h["loop_start"], len(h["data"]))
    return dict(title=blob[:20].split(b"\0")[0].decode("latin-1"), nch=nch, order=ordr, patterns=pats, samples=heads)


def render_mod(src, sr=DEFAULT_SR, system="pal", seconds=None, filter="a500", led=False, separation=1.0, hw_limit=False):
    """Offline render of a ProTracker MOD to stereo (n, 2), at most `seconds` long (default 300). Effects: 0 arpeggio, 1 and 2
    slides, 3 tone portamento, 4 vibrato, 5 and 6 (portamento / vibrato with volume slide), 7 tremolo, 8 panning, 9 sample offset,
    A volume slide, B position jump, C set volume, D pattern break, F speed / tempo, E1 / E2 fine slides, E3 glissando control,
    E4 / E7 vibrato / tremolo waveform (0 sine, 1 ramp, 2 square, +4 no retrigger), E5 finetune, E6 pattern loop, E8 pan,
    E9 retrigger, EA / EB fine volume, EC note cut, ED note delay, EE pattern delay. Finetune uses the 16 period tables; periods are
    held between 113 and 856 by the slides, and the output period is limited to 113, or to 123 (PAL) / 124 (NTSC) with `hw_limit` (what the four-voice
    DMA of the real chip can fetch). `separation` 1 = hard left / right, 0 = mono; a 8xx / E8x channel is panned freely."""
    m = src if isinstance(src, dict) else mod_parse(src)
    clk = PAULA_CLOCK[system]
    pmin = PAULA_DMA_MIN[system] if hw_limit else PAULA_MIN_PERIOD
    nch, smp = m["nch"], m["samples"]
    limit = float(seconds) if seconds else 300.0
    speed, tempo = 6, 125
    z = [0] * nch
    per, base, vol, cur, ft, tpt, tps = (list(z) for _ in range(7))
    vpos, vspd, vdep, wv, tpos, tspd, tdep, wt, sofs, lstart, lcnt = (list(z) for _ in range(11))
    gliss = [False] * nch
    pan = [None] * nch
    ticks = []                      # per tick: seconds, [(period, volume, sample id + 1 or 0, offset, pan)] per channel
    seen = set()
    o, r = 0, 0
    t = 0.0

    def idx_of(p, f):
        return int(np.argmin(np.abs(_pt_table(f) - p)))

    done = False
    while not done and t < limit:
        if o >= len(m["order"]):
            break
        if (o, r) in seen:
            break
        seen.add((o, r))
        row = m["patterns"][m["order"][o]][r]
        jump_o = brk = loop_to = None
        repeats = 1
        cmds = []
        for c, (s, p, fx, pa) in enumerate(row):
            if s:
                cur[c] = s
                vol[c] = smp[s - 1]["volume"]
                ft[c] = smp[s - 1]["finetune"]
            ex = pa >> 4 if fx == 14 else -1
            delay = pa & 15 if ex == 13 else 0
            newnote = bool(p)
            if ex == 5:
                ft[c] = (pa & 15) - 16 if pa & 8 else pa & 15
            elif ex == 3:
                gliss[c] = bool(pa & 15)
            elif ex == 4:
                wv[c] = pa & 15
            elif ex == 7:
                wt[c] = pa & 15
            if p:
                pp = int(_pt_table(ft[c])[idx_of(p, 0)])
                if fx in (3, 5):
                    tpt[c] = pp
                    newnote = False
                else:
                    if delay == 0:
                        per[c] = base[c] = pp
                    if not wv[c] & 4:
                        vpos[c] = 0
                    if not wt[c] & 4:
                        tpos[c] = 0
                    sofs[c] = pa * 256 if fx == 9 else 0
            if fx == 3 and pa:
                tps[c] = pa
            if fx == 4:
                vspd[c] = pa >> 4 or vspd[c]
                vdep[c] = pa & 15 or vdep[c]
            if fx == 7:
                tspd[c] = pa >> 4 or tspd[c]
                tdep[c] = pa & 15 or tdep[c]
            if fx == 8 and pa != 0xA4:
                pan[c] = pa / 255.0
            if fx == 12:
                vol[c] = min(pa, 64)
            elif fx == 11:
                jump_o = pa
            elif fx == 13:
                brk = (pa >> 4) * 10 + (pa & 15)
                brk = brk if brk < 64 else 0
            elif fx == 15 and pa:
                speed, tempo = (pa, tempo) if pa < 32 else (speed, pa)
            elif fx == 14:
                if ex == 1:
                    per[c] = base[c] = max(PAULA_MIN_PERIOD, per[c] - (pa & 15))
                elif ex == 2:
                    per[c] = base[c] = min(856, per[c] + (pa & 15))
                elif ex == 6:
                    if pa & 15 == 0:
                        lstart[c] = r
                    else:
                        if lcnt[c] == 0:
                            lcnt[c] = pa & 15
                        else:
                            lcnt[c] -= 1
                        if lcnt[c]:
                            loop_to = lstart[c]
                elif ex == 8:
                    pan[c] = (pa & 15) / 15.0
                elif ex == 10:
                    vol[c] = min(64, vol[c] + (pa & 15))
                elif ex == 11:
                    vol[c] = max(0, vol[c] - (pa & 15))
                elif ex == 14 and pa & 15:
                    repeats = 1 + (pa & 15)
            cmds.append((newnote and cur[c] > 0, delay))
        for kk in range(speed * repeats):
            k = kk % speed
            first = kk < speed
            rec = []
            for c, (s, p, fx, pa) in enumerate(row):
                trg = 0
                keep_off = True
                nn, delay = cmds[c]
                if first and nn and k == delay:
                    trg = cur[c]
                    keep_off = False
                    if delay:
                        per[c] = base[c] = int(_pt_table(ft[c])[idx_of(p, 0)])
                pe = per[c]
                vt = vol[c]
                ex = pa >> 4 if fx == 14 else -1
                if k > 0 or not first:
                    if fx == 0 and pa:
                        i0 = idx_of(base[c], ft[c])
                        semi = (0, pa >> 4, pa & 15)[k % 3]
                        pe = int(_pt_table(ft[c])[min(35, i0 + semi)])
                    elif fx == 1:
                        per[c] = base[c] = max(PAULA_MIN_PERIOD, per[c] - pa)
                        pe = per[c]
                    elif fx == 2:
                        per[c] = base[c] = min(856, per[c] + pa)
                        pe = per[c]
                    elif fx in (3, 5):
                        if tpt[c] and per[c] != tpt[c]:
                            per[c] = min(per[c] + tps[c], tpt[c]) if per[c] < tpt[c] else max(per[c] - tps[c], tpt[c])
                            base[c] = per[c]
                        pe = per[c]
                        if gliss[c]:
                            tab = _pt_table(ft[c])
                            pe = next((int(x) for x in tab if per[c] >= x), int(tab[-1]))
                    if fx in (4, 6):
                        d = (_pt_wave(wv[c] & 3, vpos[c]) * vdep[c]) >> 7
                        pe = per[c] + (-d if vpos[c] & 128 else d)
                        vpos[c] = (vpos[c] + vspd[c] * 4) & 255
                    if fx == 7:
                        d = (_pt_wave(wt[c] & 3, tpos[c]) * tdep[c]) >> 6
                        vt = min(64, max(0, vol[c] + (-d if tpos[c] & 128 else d)))
                        tpos[c] = (tpos[c] + tspd[c] * 4) & 255
                    if fx in (5, 6, 10):
                        vol[c] = vt = min(64, max(0, vol[c] + ((pa >> 4) if pa >> 4 else -(pa & 15))))
                    if ex == 12 and k == (pa & 15):
                        vol[c] = vt = 0
                    if ex == 9 and pa & 15 and k % (pa & 15) == 0 and cur[c] > 0:
                        trg = cur[c]
                if ex == 12 and (pa & 15) == 0 and k == 0 and first:
                    vol[c] = vt = 0
                p_out = min(max(pe, pmin), 1712) if per[c] else 0
                ps = pan[c]
                rec.append((p_out, vt, trg, 0 if keep_off else sofs[c], ps))
            sec = 2.5 / tempo
            ticks.append((sec, rec))
            t += sec
            if t >= limit:
                done = True
                break
        if loop_to is not None:
            for rr in range(loop_to, r + 1):
                seen.discard((o, rr))
            r = loop_to
        elif jump_o is not None or brk is not None:
            o, r = (jump_o if jump_o is not None else o + 1), (brk or 0)
        else:
            r += 1
            if r >= 64:
                o, r = o + 1, 0
    cuts = np.round(np.cumsum([0.0] + [tk[0] for tk in ticks]) * sr).astype(np.int64)
    n = max(int(cuts[-1]), 1)
    lens_ = np.diff(cuts)
    lr = np.zeros((2, n))
    bank = np.concatenate([s["data"] for s in smp]) if smp else np.zeros(1)
    offs = np.cumsum([0] + [len(s["data"]) for s in smp[:-1]]).astype(np.int64)
    lns = np.array([len(s["data"]) for s in smp], np.int64)
    ls = np.array([s["loop_start"] for s in smp], np.int64)
    ll = np.array([s["loop_len"] for s in smp], np.int64)
    sep = float(separation)
    for c in range(nch):
        per_a = np.repeat([tk[1][c][0] for tk in ticks], lens_).astype(np.float64)
        vol_a = np.repeat([tk[1][c][1] for tk in ticks], lens_) / 64.0 / 128.0
        trg_a = np.zeros(n, np.int64)
        tof_a = np.zeros(n)
        hit = np.array([tk[1][c][2] > 0 for tk in ticks])
        trg_a[cuts[:-1][hit]] = [tk[1][c][2] for tk in ticks if tk[1][c][2] > 0]
        tof_a[cuts[:-1][hit]] = [tk[1][c][3] for tk in ticks if tk[1][c][2] > 0]
        y = _paula_chan(n, per_a, vol_a, trg_a, tof_a, bank, offs, lns, ls, ll, clk, float(sr))
        side = PAULA_SIDE[c % 4]
        if all(tk[1][c][4] is None for tk in ticks):
            lr[side] += y
            lr[1 - side] += y * (1.0 - sep)
        else:
            home = (0.0 if side == 0 else 1.0) * sep + 0.5 * (1.0 - sep)
            pn = np.array([home if tk[1][c][4] is None else tk[1][c][4] for tk in ticks])
            pn = np.repeat(pn, lens_)
            lr[0] += y * np.minimum(1.0, 2.0 * (1.0 - pn))
            lr[1] += y * np.minimum(1.0, 2.0 * pn)
    y = paula_filter(_hp(lr.T, sr), sr, filter, led)
    return y * PAULA_GAIN * min(1.0, 4.0 / nch)



# ============================================================================ Yamaha OPL2 / OPL3
OPL_CLOCK = 14318180.0
OPL_RATE = OPL_CLOCK / 288.0
_i = np.arange(256)
OPL_LOGSIN = np.round(-np.log2(np.sin((_i + 0.5) * np.pi / 512.0)) * 256.0).astype(np.int64)
OPL_EXP = np.round((2.0 ** ((255 - _i) / 256.0) - 1.0) * 1024.0).astype(np.int64)
OPL_MULT2 = np.array([1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 20, 24, 24, 30, 30], np.int64)
OPL_KSL = np.array([0, 32, 40, 45, 48, 51, 53, 55, 56, 58, 59, 60, 61, 62, 63, 64], np.int64)
OPL_KSL_SHIFT = np.array([8, 1, 2, 0], np.int64)
OPL_EG_PAT = np.array([[0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 1, 1, 0, 1], [0, 1, 1, 1, 0, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1]], np.int64)
_OPK = ("mult", "ksl", "tl", "ar", "dr", "sl", "rr", "sus", "ksr", "wave", "am", "vib")
OPL_OP = dict(mult=1, ksl=0, tl=0, ar=15, dr=0, sl=0, rr=7, sus=1, ksr=0, wave=0, am=0, vib=0)
OPL_PATCHES = {
    "sine": dict(ops=[dict(tl=63), dict()], fb=0, cnt=0),
    "organ": dict(ops=[dict(mult=1, tl=0, wave=0), dict(mult=2, tl=8)], fb=0, cnt=1),
    "bell": dict(ops=[dict(mult=14, tl=22, ar=15, dr=6, sl=8, rr=6, sus=0), dict(mult=1, tl=0, ar=15, dr=4, sl=10, rr=5, sus=0)], fb=0, cnt=0),
    "bass": dict(ops=[dict(mult=1, tl=24, ar=15, dr=9, sl=4, rr=7, sus=0), dict(mult=1, tl=0, ar=15, dr=5, sl=3, rr=7, sus=0)], fb=5, cnt=0),
    "brass": dict(ops=[dict(mult=1, tl=16, ar=9, dr=2, sl=3, rr=4), dict(mult=1, tl=0, ar=10, dr=2, sl=2, rr=5)], fb=4, cnt=0),
    "pluck": dict(ops=[dict(mult=2, tl=26, ar=15, dr=10, sl=15, rr=8, sus=0), dict(mult=1, tl=0, ar=15, dr=6, sl=12, rr=7, sus=0)], fb=2, cnt=0),
    "wow4": dict(ops=[dict(mult=1, tl=18), dict(mult=2, tl=6), dict(mult=1, tl=12), dict(mult=1, tl=0)], fb=3, cnt=(0, 0)),
}
OPL_RHYTHM = {            # name -> (op list, fb): bd uses two ops, the others one
    "bd": ([dict(mult=1, tl=14, ar=15, dr=9, sl=6, rr=6, sus=0), dict(mult=1, tl=0, ar=15, dr=8, sl=4, rr=6, sus=0)], 6),
    "hh": ([dict(mult=1, tl=6, ar=15, dr=11, sl=15, rr=9, sus=0)], 0),
    "sd": ([dict(mult=1, tl=4, ar=15, dr=10, sl=15, rr=8, sus=0)], 0),
    "tom": ([dict(mult=1, tl=6, ar=15, dr=8, sl=8, rr=7, sus=0)], 0),
    "tc": ([dict(mult=1, tl=6, ar=15, dr=9, sl=15, rr=7, sus=0)], 0),
}
OPL_NAMES2 = tuple(f"ch{i}" for i in range(1, 10))
OPL_NAMES3 = tuple(f"ch{i}" for i in range(1, 19))
OPL_DRUMS = ("bd", "sd", "tom", "tc", "hh")
OPL_PATCH = dict(_PATCH, patch="organ", fb=None, wave=None, mod_wave=None, trem=False, vib_lfo=False, deep_trem=True,
                 deep_vib=True, nts=0, tail=0.3)


def opl_note(hz: float):
    """(fnum, block) for a pitch: f = fnum * 49716 / 2**(20 - block), the smallest block that keeps fnum below 1024."""
    for b in range(8):
        fn = int(round(hz / (OPL_RATE * 2.0 ** (b - 20))))
        if fn < 1024:
            return max(fn, 1), b
    return 1023, 7


def opl_hz(fnum: int, block: int, mult: float = 1.0) -> float:
    return fnum * OPL_RATE * 2.0 ** (block - 20) * mult


def opl_wave_table(w: int) -> np.ndarray:
    """One 1024-step cycle of waveform w (0..7) at full level, 13-bit signed (+-4084)."""
    return np.array([_opl_wave(w, p, 0) for p in range(1024)])


@nb.njit(cache=True)
def _opl_wave(w, p, att):
    p &= 0x3FF
    neg = 0
    if w == 0:
        neg = p >> 9
        q = p & 0x1FF
        ls = OPL_LOGSIN[(q & 0xFF) ^ 0xFF] if q & 0x100 else OPL_LOGSIN[q & 0xFF]
    elif w == 1:
        if p >> 9:
            return 0
        q = p & 0x1FF
        ls = OPL_LOGSIN[(q & 0xFF) ^ 0xFF] if q & 0x100 else OPL_LOGSIN[q & 0xFF]
    elif w == 2:
        q = p & 0x1FF
        ls = OPL_LOGSIN[(q & 0xFF) ^ 0xFF] if q & 0x100 else OPL_LOGSIN[q & 0xFF]
    elif w == 3:
        if p & 0x100:
            return 0
        ls = OPL_LOGSIN[p & 0xFF]
    elif w == 4:
        if p & 0x200:
            return 0
        p2 = (p << 1) & 0x3FF
        neg = p2 >> 9
        q = p2 & 0x1FF
        ls = OPL_LOGSIN[(q & 0xFF) ^ 0xFF] if q & 0x100 else OPL_LOGSIN[q & 0xFF]
    elif w == 5:
        if p & 0x200:
            return 0
        q = (p << 1) & 0x1FF
        ls = OPL_LOGSIN[(q & 0xFF) ^ 0xFF] if q & 0x100 else OPL_LOGSIN[q & 0xFF]
    elif w == 6:
        neg = p >> 9
        ls = 0
    else:
        neg = p >> 9
        x = (p & 0x1FF) ^ 0x1FF if neg else p & 0x1FF
        ls = x << 3
    a = ls + att
    sh = a >> 8
    if sh >= 13:
        return 0
    v = ((OPL_EXP[a & 0xFF] | 0x400) << 1) >> sh
    return -v if neg else v


@nb.njit(cache=True)
def _opl_op(S, P, k, kon, cnt, inc, ksr_kc, ksl_att, pm, ph_override, trem, volatt, dvol):
    """One operator for one sample. S[k]: acc, env, stage, last out, previous out, last key. Returns the 13-bit output."""
    if kon and not S[k, 5]:
        S[k, 2] = 0
        S[k, 0] = 0
        if P[k, 3] >= 15:
            S[k, 1] = 0
    elif not kon and S[k, 5] and S[k, 2] != 3:
        S[k, 2] = 3
    S[k, 5] = 1 if kon else 0
    st = S[k, 2]
    if st == 0:
        r = P[k, 3]
    elif st == 1:
        r = P[k, 4]
    elif st == 2:
        r = P[k, 6] if not P[k, 7] else 0
    else:
        r = P[k, 6]
    rate = 0
    if r > 0:
        rate = min(63, 4 * r + (ksr_kc >> (0 if P[k, 8] else 2)))
    if rate > 0:
        rh = rate >> 2
        rl = rate & 3
        sh = max(12 - rh, 0)
        mul = 1 << max(rh - 12, 0)
        env = S[k, 1]
        if st == 0 and rh >= 15:
            env = 0
        elif (cnt & ((1 << sh) - 1)) == 0:
            ic = OPL_EG_PAT[rl, (cnt >> sh) & 7] * mul
            if st == 0:
                env += ((-env - 1) * ic) >> 3
                if env < 0:
                    env = 0
            else:
                env += ic
                if env > 511:
                    env = 511
        S[k, 1] = env
    sl = P[k, 5]
    if st == 0 and S[k, 1] <= 0:
        S[k, 2] = 1
    if S[k, 2] == 1 and S[k, 1] >= (31 if sl == 15 else sl) << 4:
        S[k, 2] = 2
    if st == 3 and S[k, 1] >= 511:
        S[k, 1] = 511
    S[k, 0] += inc
    ph = (S[k, 0] >> 9) & 0x3FF
    if ph_override >= 0:
        ph = ph_override
    e = S[k, 1] + (P[k, 2] << 2) + ksl_att + (trem if P[k, 10] else 0) + volatt
    if e > 511:
        e = 511
    return _opl_wave(P[k, 9], ph + pm, e << 3), ph


@nb.njit(cache=True)
def _opl_phase_inc(P, k, fnum, blk, vibp, dvb):
    fn = fnum
    if P[k, 11]:
        rg = (fnum >> 7) & 7
        if vibp & 3 == 0:
            d = 0
        elif vibp & 1:
            d = rg >> 1
        else:
            d = rg
        d >>= dvb
        fn = fnum - d if vibp & 4 else fnum + d
    return (((fn << blk) >> 1) * OPL_MULT2[P[k, 0]]) >> 1


@nb.njit(cache=True)
def _opl_chan(n, kon, fnum, blk, volatt, trem, P, modmask, outmask, fb, nops, nts, dvb):
    out = np.zeros(n)
    S = np.zeros((4, 6), np.int64)
    S[:, 1] = 511
    outs = np.zeros(4, np.int64)
    for i in range(n):
        fn = fnum[i]
        b = blk[i]
        kc = (b << 1) | ((fn >> (9 - nts)) & 1)
        vibp = (i >> 10) & 7
        tot = 0
        for k in range(nops):
            pm = 0
            for j in range(k):
                if (modmask[k] >> j) & 1:
                    pm += outs[j] >> 1
            if k == 0 and fb > 0:
                pm += (S[0, 3] + S[0, 4]) >> (9 - fb)
            ks = (OPL_KSL[fn >> 6] << 2) - ((8 - b) << 5)
            if ks < 0:
                ks = 0
            ks >>= OPL_KSL_SHIFT[P[k, 1]]
            inc = _opl_phase_inc(P, k, fn, b, vibp, dvb)
            o, ph = _opl_op(S, P, k, kon[i] > 0, i, inc, kc, ks, pm, -1, trem[i], volatt[i], 0)
            S[k, 4] = S[k, 3]
            S[k, 3] = o
            outs[k] = o
        for k in range(nops):
            if (outmask >> k) & 1:
                tot += outs[k]
        out[i] = tot / 4096.0
    return out


@nb.njit(cache=True)
def _opl_rhythm(n, kon, fnum, blk, volatt, trem, P, fb):
    """Rhythm section: kon[0..4] = bd, sd, tom, tc, hh; fnum / blk rows 0 (bd), 1 (ch7: hh, sd), 2 (ch8: tom, tc). P rows: bd mod,
    bd car, hh, sd, tom, tc. hh, sd and tc take their phase from the phase bits of hh and tc and the 23-bit noise register."""
    out = np.zeros(n)
    S = np.zeros((6, 6), np.int64)
    S[:, 1] = 511
    rng = 1
    for i in range(n):
        nb_ = ((rng >> 14) ^ rng) & 1 | (1 if rng == 0 else 0)
        rng = (rng >> 1) | (nb_ << 22)
        noise = rng & 1
        chof = (0, 0, 1, 1, 2, 2)
        o6 = np.zeros(6, np.int64)
        hhp = (S[2, 0] >> 9) & 0x3FF
        tcp = (S[5, 0] >> 9) & 0x3FF
        rmx = (((hhp >> 2) ^ (hhp >> 7)) & 1) | (((hhp >> 3) ^ (tcp >> 5)) & 1) | (((tcp >> 3) ^ (tcp >> 5)) & 1)
        for k in range(6):
            c = chof[k]
            fn = fnum[c, i]
            b = blk[c, i]
            kc = (b << 1) | (fn >> 9)
            ks = (OPL_KSL[fn >> 6] << 2) - ((8 - b) << 5)
            if ks < 0:
                ks = 0
            ks >>= OPL_KSL_SHIFT[P[k, 1]]
            inc = _opl_phase_inc(P, k, fn, b, 0, 1)
            kn = (kon[0, i], kon[0, i], kon[4, i], kon[1, i], kon[2, i], kon[3, i])[k]
            ov = -1
            if k == 2:
                ov = (rmx << 9) | (0xD0 if (rmx ^ noise) else 0x34)
            elif k == 3:
                ov = (((hhp >> 8) & 1) << 9) | ((((hhp >> 8) & 1) ^ noise) << 8)
            elif k == 5:
                ov = (rmx << 9) | 0x100
            pm = 0
            if k == 1:
                pm = o6[0] >> 1
            elif k == 0 and fb > 0:
                pm = (S[0, 3] + S[0, 4]) >> (9 - fb)
            o, ph = _opl_op(S, P, k, kn > 0, i, inc, kc, ks, pm, ov, trem[i], volatt[c, i], 0)
            S[k, 4] = S[k, 3]
            S[k, 3] = o
            o6[k] = o
        out[i] = (o6[1] + 2 * (o6[2] + o6[3] + o6[4] + o6[5])) / 4096.0
    return out


def _op_row(d):
    dd = dict(OPL_OP)
    dd.update({k: v for k, v in d.items() if v is not None})
    for k, (lo, hi) in dict(mult=(0, 15), ksl=(0, 3), tl=(0, 63), ar=(0, 15), dr=(0, 15), sl=(0, 15), rr=(0, 15), sus=(0, 1),
                            ksr=(0, 1), wave=(0, 7), am=(0, 1), vib=(0, 1)).items():
        if not lo <= int(dd[k]) <= hi:
            raise SpecError(f"opl: {k} must be {lo}..{hi}")
    return [int(dd[k]) for k in _OPK]


def _opl_patch(p, chip):
    pt = p["patch"]
    pt = OPL_PATCHES.get(pt) if isinstance(pt, str) else pt
    if pt is None:
        raise SpecError(f"opl: patch must be one of {sorted(OPL_PATCHES)} or a dict(ops, fb, cnt)")
    ops = [dict(o) for o in pt["ops"]]
    cnt = pt.get("cnt", 0)
    four = len(ops) == 4
    if len(ops) not in (2, 4) or (four and chip != "opl3"):
        raise SpecError("opl: a patch has 2 operators (4 on opl3)")
    if p.get("wave") is not None:
        ops[-1]["wave"] = int(p["wave"])
    if p.get("mod_wave") is not None:
        ops[0]["wave"] = int(p["mod_wave"])
    for o in ops:
        o["am"] = o.get("am", int(bool(p.get("trem"))))
        o["vib"] = o.get("vib", int(bool(p.get("vib_lfo"))))
        if int(o.get("wave", 0)) > (7 if chip == "opl3" else 3):
            raise SpecError("opl2 has waveforms 0..3, opl3 0..7")
    P = np.zeros((4, 12), np.int64)
    for k, o in enumerate(ops):
        P[k] = _op_row(o)
    fb = pt["fb"] if p.get("fb") is None else p["fb"]
    if not 0 <= int(fb) <= 7:
        raise SpecError("opl: fb 0..7")
    if four:
        c0, c1 = cnt if isinstance(cnt, (tuple, list)) else (cnt & 1, cnt >> 1)
        mm = {(0, 0): ([0, 1, 2, 4], 8), (1, 0): ([0, 0, 2, 4], 9), (0, 1): ([0, 1, 0, 4], 10), (1, 1): ([0, 0, 2, 0], 13)}[(int(c0), int(c1))]
    else:
        mm = ([0, 1], 2) if int(cnt) == 0 else ([0, 0], 3)
    return P, np.array(mm[0], np.int64), int(mm[1]), int(fb), len(ops)


def _opl_lfo(n, deep):
    pos = (np.arange(n) // 64) % 210
    tri = np.where(pos < 105, pos, 210 - pos) >> 2
    return (tri if deep else tri >> 2).astype(np.int64)


def render_opl(lanes, dur, sr, chip="opl2", rhythm=False):
    """`dur` seconds of an OPL2 (YM3812, 9 two-operator voices, waveforms 0..3, mono) or OPL3 (YMF262, 18 voices, waveforms 0..7,
    4-operator voices, `pan` L | C | R per voice, stereo (n, 2)), at the chip's 49716 Hz sample rate. Note patches: `patch` a
    name (see OPL_PATCHES) or dict(ops [dict(mult ksl tl ar dr sl rr sus ksr wave am vib)], fb, cnt); `wave` and `mod_wave` override
    the carrier and modulator waveform, `trem` / `vib_lfo` switch both operators' tremolo / vibrato, `deep_trem` 4.8 dB else 1 dB
    and `deep_vib` 14 else 7 cents. 4-operator patches have `cnt` (first pair bit, second pair bit) and occupy the lane. With
    `rhythm` the lanes bd sd tom tc hh are drum notes (the pitch sets the F-number of voices 6, 7, 8) and ch6..ch9 are not used."""
    chip = str(chip).lower()
    if chip not in ("opl2", "opl3"):
        raise SpecError("opl: chip is opl2 or opl3")
    names = OPL_NAMES3 if chip == "opl3" else OPL_NAMES2
    n = int(dur * OPL_RATE) + 2
    mix = np.zeros((2, n))
    for name in names:
        notes = _notes(lanes, name)
        for ni, (t0, t1, hz, vel, p) in enumerate(notes):
            tr = _note_track(t0, t1, hz, vel, p, OPL_RATE, lambda f: (opl_note(f)[1] << 10) | opl_note(f)[0], OPL_RATE)
            if tr is None:
                continue
            s, reg, vv = tr
            nxt = int(notes[ni + 1][0] * OPL_RATE) if ni + 1 < len(notes) else n
            tail = max(0, min(int(float(p["tail"]) * OPL_RATE), nxt - (s + len(reg)), n - s - len(reg)))
            reg = np.concatenate([reg, np.full(tail, reg[-1])])
            kon = np.concatenate([np.ones(len(reg) - tail, np.int64), np.zeros(tail, np.int64)])
            vv = np.concatenate([vv, np.zeros(tail, np.int64)])
            va = np.where(vv[:len(kon)] > 0, (15 - vv) * 16, 511).astype(np.int64)
            va[len(kon) - tail:] = 0
            P, mm, om, fb, no = _opl_patch(p, chip)
            y = _opl_chan(len(kon), kon, reg & 1023, reg >> 10, va, _opl_lfo(len(kon), p["deep_trem"]), P, mm, om, fb, no,
                          int(p["nts"]), 0 if p["deep_vib"] else 1)
            pan = str(p.get("pan", "C"))
            for ch, side in ((0, "R"), (1, "L")):
                if chip == "opl3" and pan == side:
                    continue
                _put(mix[ch], s, y)
    if rhythm:
        kon = np.zeros((5, n), np.int64)
        fnum = np.full((3, n), 512, np.int64)
        blk = np.full((3, n), 2, np.int64)
        va = np.zeros((3, n), np.int64)
        vol_ = [None] * 5
        for di, dn in enumerate(OPL_DRUMS):
            for t0, t1, hz, vel, p in _notes(lanes, dn):
                tr = _note_track(t0, t1, hz, vel, p, OPL_RATE, lambda f: (opl_note(f)[1] << 10) | opl_note(f)[0], OPL_RATE)
                if tr is None:
                    continue
                s, reg, vv = tr
                m = min(len(reg), n - s)
                kon[di, s:s + m] = 1
                c = {"bd": 0, "hh": 1, "sd": 1, "tom": 2, "tc": 2}[dn]
                fnum[c, s:s + m] = reg[:m] & 1023
                blk[c, s:s + m] = reg[:m] >> 10
                va[c, s:s + m] = np.minimum(va[c, s:s + m], 0)
        P = np.zeros((6, 12), np.int64)
        for k, (dn, j) in enumerate((("bd", 0), ("bd", 1), ("hh", 0), ("sd", 0), ("tom", 0), ("tc", 0))):
            P[k] = _op_row(OPL_RHYTHM[dn][0][j])
        y = _opl_rhythm(n, kon, fnum, blk, va, _opl_lfo(n, True), P, OPL_RHYTHM["bd"][1])
        mix[0] += y
        mix[1] += y
    if chip == "opl2":
        y = _to_sr(mix[0], OPL_RATE, sr)
        return _fit(_hp(y, sr), samples(dur, sr)) * 0.3
    y = np.stack([_to_sr(mix[0], OPL_RATE, sr), _to_sr(mix[1], OPL_RATE, sr)], axis=1)
    y = _hp(y, sr)
    return np.stack([_fit(y[:, 0], samples(dur, sr)), _fit(y[:, 1], samples(dur, sr))], axis=1) * 0.3


def _opl_layer(chip):
    def run(layer, sr, q):
        names = (OPL_NAMES3 if chip == "opl3" else OPL_NAMES2) + (OPL_DRUMS if layer.get("rhythm") else ())
        lanes, end = lanes_of(layer, chip, names, OPL_PATCH, q)
        return render_opl(lanes, end + 0.3, sr, chip=chip, rhythm=bool(layer.get("rhythm")))
    return run


for _c, _nm in (("opl2", OPL_NAMES2), ("opl3", OPL_NAMES3)):
    CT.SYSTEM_LAYERS[_c] = _opl_layer(_c)
    CT.SYSTEMS[_c] = _nm + OPL_DRUMS


def _opl_instr(chip):
    def fn(freq, dur, sr=DEFAULT_SR, vel=1.0, patch="organ", fb=None, wave=None, mod_wave=None, trem=False, vib_lfo=False,
           volume=15, vol=None, arp=None, pitch=None, vib=0.0, vib_hz=6.0, vib_delay=0.15, mode="retro_stylized", bright=0.5):
        p = dict(OPL_PATCH, patch=patch, fb=fb, wave=wave, mod_wave=mod_wave, trem=trem, vib_lfo=vib_lfo, volume=volume, vol=vol,
                 arp=arp, pitch=pitch, vib=vib, vib_hz=vib_hz, vib_delay=vib_delay)
        y = render_opl({"ch1": [(0.0, dur, float(freq), 1.0, p)]}, dur + 0.01, sr, chip=chip)
        y = y if y.ndim == 1 else 0.5 * (y[:, 0] + y[:, 1])
        return _finish(y, freq, sr, vel, mode, bright, 0.01, dur)
    return fn


_OPL_PARAMS = dict(
    patch=("organ", "sine | organ | bell | bass | brass | pluck (opl3 adds wow4: 4 operators), or a dict(ops, fb, cnt)"),
    fb=(None, "0..7 operator-1 feedback, None = the patch's"), wave=(None, "carrier waveform 0..3 (opl3: 0..7)"),
    mod_wave=(None, "modulator waveform"), trem=(False, "tremolo 3.7 Hz on both operators"),
    vib_lfo=(False, "vibrato 6.1 Hz on both operators"), volume=(15, "0..15, 3 dB a step"),
    vol=(None, _H["vol"]), arp=(None, _H["arp"]), pitch=(None, _H["pitch"]), vib=(0.0, _H["vib"]), vib_hz=(6.0, _H["vib_hz"]),
    vib_delay=(0.15, _H["vib_delay"]), mode=CT._MODE, bright=CT._BRIGHT)
instrument("opl2", "Yamaha OPL2 (YM3812, AdLib) voice: two-operator FM, 10-bit log-sine and exponent tables, F-number / block pitch, "
           "ADSR rates, 4 waveforms, feedback, tremolo and vibrato.", family="retro", span=("C1", "C7"), **_OPL_PARAMS)(_opl_instr("opl2"))
instrument("opl3", "Yamaha OPL3 (YMF262, Sound Blaster 16) voice: as opl2 with 8 waveforms and 4-operator patches.", family="retro",
           span=("C1", "C7"), **_OPL_PARAMS)(_opl_instr("opl3"))


# ============================================================================ Yamaha YM2413 OPLL / Konami VRC7
OPLL_ROM = (
    ("violin", "71611E17D0780017"), ("guitar", "13411A0DD8F72313"), ("piano", "13019900F2C41123"),
    ("flute", "31610E07A8647027"), ("clarinet", "32211E06E0760028"), ("oboe", "31221605E0710018"),
    ("trumpet", "21611D0782811007"), ("organ", "23212D14A2720007"), ("horn", "61611B0664651017"),
    ("synth", "41610B1885F77107"), ("harpsichord", "13018311FAE41004"), ("vibraphone", "17C12407F8F82212"),
    ("synth_bass", "61500C05C2F52042"), ("acoustic_bass", "01015503C9950302"), ("electric_guitar", "61418903F1E44013"),
)


def opll_patch(name_or_bytes):
    """A YM2413 instrument (the ROM name, 1..15, or 8 register bytes 0x00..0x07) as an OPL2 patch: byte 0 and 1 the modulator and
    carrier AM / vibrato / sustain / KSR / multiple, byte 2 the modulator's KSL and level, byte 3 the carrier's KSL, the two
    rectify bits and the feedback, bytes 4-7 the attack-decay and sustain-release pairs. The carrier level is the channel volume."""
    if isinstance(name_or_bytes, str):
        d = dict(OPLL_ROM)
        if name_or_bytes not in d:
            raise SpecError(f"opll: patch must be one of {sorted(d)}, 1..15 or 8 bytes")
        b = bytes.fromhex(d[name_or_bytes])
    elif isinstance(name_or_bytes, int) and not isinstance(name_or_bytes, bool):
        if not 1 <= name_or_bytes <= 15:
            raise SpecError("opll: patch number 1..15")
        b = bytes.fromhex(OPLL_ROM[name_or_bytes - 1][1])
    else:
        b = bytes(int(v) & 255 for v in name_or_bytes)
        if len(b) != 8:
            raise SpecError("opll: a user patch is 8 bytes")

    def op(k, tl, ksl, wave):
        v = b[k]
        return dict(am=v >> 7 & 1, vib=v >> 6 & 1, sus=v >> 5 & 1, ksr=v >> 4 & 1, mult=v & 15, tl=tl, ksl=ksl, wave=wave,
                    ar=b[4 + k] >> 4, dr=b[4 + k] & 15, sl=b[6 + k] >> 4, rr=b[6 + k] & 15)
    return dict(ops=[op(0, b[2] & 63, b[2] >> 6, b[3] >> 3 & 1), op(1, 0, b[3] >> 6, b[3] >> 4 & 1)], fb=b[3] & 7, cnt=0)


@instrument("opll", "Yamaha YM2413 OPLL (MSX-Music, Master System Japan; the VRC7 of Lagrange Point): the 15 ROM instruments as two-operator FM on the OPL engine, or an 8-byte user patch.",
            family="retro", span=("C1", "C7"),
            patch=("violin", "violin | guitar | piano | flute | clarinet | oboe | trumpet | organ | horn | synth | harpsichord | vibraphone | synth_bass | acoustic_bass | electric_guitar, a number 1..15, or 8 register bytes"),
            volume=(15, "0..15, 3 dB a step"), vol=(None, _H["vol"]), arp=(None, _H["arp"]), pitch=(None, _H["pitch"]),
            vib=(0.0, _H["vib"]), vib_hz=(6.0, _H["vib_hz"]), vib_delay=(0.15, _H["vib_delay"]), mode=CT._MODE, bright=CT._BRIGHT)
def opll(freq, dur, sr=DEFAULT_SR, vel=1.0, patch="violin", volume=15, vol=None, arp=None, pitch=None, vib=0.0, vib_hz=6.0,
         vib_delay=0.15, mode="retro_stylized", bright=0.5):
    p = dict(OPL_PATCH, patch=opll_patch(patch), volume=volume, vol=vol, arp=arp, pitch=pitch, vib=vib, vib_hz=vib_hz, vib_delay=vib_delay)
    y = render_opl({"ch1": [(0.0, dur, float(freq), 1.0, p)]}, dur + 0.01, sr, chip="opl2")
    return _finish(y, freq, sr, vel, mode, bright, 0.01, dur)


# ============================================================================ Super Nintendo S-DSP (BRR voice)
SNES_RATE = 32000.0
SNES_ATTACK_S = (4.1, 2.6, 1.5, 1.0, 0.64, 0.38, 0.26, 0.16, 0.096, 0.064, 0.040, 0.024, 0.016, 0.010, 0.006, 0.0)
SNES_DECAY_S = (1.2, 0.74, 0.44, 0.29, 0.18, 0.11, 0.074, 0.037)
SNES_RATE_PERIOD = (0, 2048, 1536, 1280, 1024, 768, 640, 512, 384, 320, 256, 192, 160, 128, 96, 80,
                    64, 48, 40, 32, 24, 20, 16, 12, 10, 8, 6, 5, 4, 3, 2, 1)       # samples at 32 kHz per envelope step, by 5-bit rate


def snes_gain_env(gain, m):
    """The VxGAIN envelope over m samples, level / 2048: 0..127 direct (level = 16 N); 128 + 32 mode + rate with mode 0 linear
    decrease (-32), 1 exponential decrease (-(((L - 1) >> 8) + 1)), 2 linear increase (+32), 3 bent increase (+32 below 0x600, else +8);
    the level starts at 0 and stays within 0..0x7FF."""
    g = int(gain) & 255
    if not g & 128:
        return np.full(m, (g & 127) * 16 / 2048.0)
    p, mode = SNES_RATE_PERIOD[g & 31], (g >> 5) & 3
    out = np.zeros(m)
    if p == 0:
        return out
    lv = 0
    for i in range(0, m, p):
        if mode == 0:
            lv -= 32
        elif mode == 1:
            lv -= ((lv - 1) >> 8) + 1
        elif mode == 2:
            lv += 32
        else:
            lv += 32 if lv < 0x600 else 8
        lv = min(max(lv, 0), 0x7FF)
        out[i:i + p] = lv
    return out / 2048.0


SNES_GAUSS_HEX = (
    "000000000000000000000000000000000000000000000000001001001001001001001001001001001002002002002002"
    "00200200300300300300300400400400400400500500500500600600600600700700700800800800900900900A00A00A"
    "00B00B00B00C00C00D00D00E00E00F00F00F01001001101101201301301401401501501601701701801801901A01B01B"
    "01C01D01D01E01F02002002102202302402402502602702802902A02B02C02D02E02F030031032033034035036037038"
    "03A03B03C03D03E04004104204304504604704904A04C04D04E05005105305405605705905A05C05E05F061063064066"
    "06806A06B06D06F07107307507607807A07C07E08008208408608908B08D08F09109309609809A09C09F0A10A30A60A8"
    "0AB0AD0AF0B20B40B70BA0BC0BF0C10C40C70C90CC0CF0D20D40D70DA0DD0E00E30E60E90EC0EF0F20F50F80FB0FE101"
    "10410710B10E11111411811B11E12212512912C13013313713A13E14114514814C15015315715B15F16216616A16E172"
    "17617A17D18118518918D19119519A19E1A21A61AA1AE1B21B71BB1BF1C31C81CC1D01D51D91DD1E21E61EB1EF1F31F8"
    "1FC20120520A20F21321821C22122622A22F23323823D24124624B25025425925E26326726C27127627B28028428928E"
    "29329829D2A22A62AB2B02B52BA2BF2C42C92CE2D32D82DC2E12E62EB2F02F52FA2FF30430930E31331831D32232632B"
    "33033533A33F34434934E35335735C36136636B37037437937E38338838C39139639B39F3A43A93AD3B23B73BB3C03C5"
    "3C93CE3D23D73DC3E03E53E93ED3F23F63FB3FF40340840C41041541941D42142542A42E43243643A43E44244644A44E"
    "45245545945D46146546846C47047347747A47E48148548848C48F49249649949C49F4A24A64A94AC4AF4B24B54B74BA"
    "4BD4C04C34C54C84CB4CD4D04D24D54D74D94DC4DE4E04E34E54E74E94EB4ED4EF4F14F34F54F64F84FA4FB4FD4FF500"
    "50250350450650750850A50B50C50D50E50F510511511512513514514515516516517517517518518518518518519519"
)
SNES_GAUSS = np.array([int(SNES_GAUSS_HEX[k:k + 3], 16) for k in range(0, 1536, 3)], np.int64)
SNES_FIR = {"echo": (127, 0, 0, 0, 0, 0, 0, 0), "lowpass": (-1, 8, 23, 36, 36, 23, 8, -1), "none": (0,) * 8}
SNES_WAVES = {
    "sine": lambda x: np.sin(2 * np.pi * x), "saw": lambda x: 2 * (x % 1.0) - 1, "square": lambda x: np.where(x % 1.0 < 0.5, 1.0, -1.0),
    "triangle": lambda x: 2 * np.abs(2 * (x % 1.0) - 1) - 1,
    "pulse": lambda x: np.where(x % 1.0 < 0.25, 1.0, -1.0),
}


def brr_predict(filt, p1, p2):
    return (0, p1 + ((-p1) >> 4), (p1 << 1) + ((-p1 * 3) >> 5) - p2 + (p2 >> 4), (p1 << 1) + ((-p1 * 13) >> 6) - p2 + ((p2 * 3) >> 4))[filt]


def brr_encode(pcm):
    """Samples in -1..1 (a multiple of 16 long) to BRR blocks: per 16 samples the range (0..12) and predictive filter (0..3) with the
    smallest error, 4-bit residues, the encoder following the decoder's own history. Returns a list of (range, filter, nibbles)."""
    x = np.round(np.asarray(pcm, float) * 16384.0).astype(np.int64)
    p1 = p2 = 0
    blocks = []
    for b in range(0, len(x), 16):
        best = None
        for filt in ((0,) if b == 0 else range(4)):
            for rng_ in range(13):
                a1, a2, nib, err = p1, p2, [], 0
                for s in x[b:b + 16]:
                    pr = brr_predict(filt, a1, a2)
                    q = int(np.clip(np.round((int(s) - pr) * 2.0 / (1 << rng_)), -8, 7))
                    d = int(np.clip(((q << rng_) >> 1) + pr, -32768, 32767))
                    err += (d - int(s)) ** 2
                    nib.append(q)
                    a2, a1 = a1, d
                if best is None or err < best[0]:
                    best = (err, rng_, filt, nib, a1, a2)
        blocks.append((best[1], best[2], best[3]))
        p1, p2 = best[4], best[5]
    return blocks


def brr_decode(blocks):
    """BRR blocks back to samples (full scale 2.0 = 16384): (nibble << range) >> 1 plus the filter's prediction from the two previous samples."""
    p1 = p2 = 0
    out = []
    for rng_, filt, nib in blocks:
        for q in nib:
            d = int(np.clip(((q << rng_) >> 1) + brr_predict(filt, p1, p2), -32768, 32767))
            out.append(d)
            p2, p1 = p1, d
    return np.array(out, float) / 16384.0


def snes_gauss(frac):
    """Four interpolation taps (oldest, older, old, newest sample) for a fractional position 0..1, read from the S-DSP's 512-entry
    table at index int(frac * 256) and normalised to sum 1 (the table's four values sum to 2047..2049)."""
    i = int(np.clip(int(frac * 256.0), 0, 255))
    g = SNES_GAUSS[[255 - i, 511 - i, 256 + i, i]].astype(float)
    return g / g.sum()


def render_snes(freq, dur, sr, wave="saw", blocks=2, gauss=True, attack=15, decay=7, sustain=7, sustain_s=0.0,
                release_s=0.05, echo_ms=0.0, echo_fb=0.4, echo_mix=0.35, fir="echo", gain=None):
    n0 = 16 * int(np.clip(blocks, 1, 64))
    if isinstance(wave, (list, tuple, np.ndarray)):
        cyc = np.asarray(wave, float)
        if cyc.size != n0:
            raise SpecError(f"snes: a sample list has 16 x blocks = {n0} values")
    else:
        if wave not in SNES_WAVES:
            raise SpecError(f"snes: wave must be one of {sorted(SNES_WAVES)} or a list of {n0} values")
        cyc = SNES_WAVES[wave](np.arange(n0) / n0)
    cyc = cyc / (np.max(np.abs(cyc)) + 1e-12) * 0.9
    smp = brr_decode(brr_encode(cyc))
    pitch = int(np.clip(round(freq * n0 / SNES_RATE * 4096.0), 1, 0x3FFF))
    edl = int(np.clip(round(float(echo_ms) / 16.0), 0, 15))
    m = int((dur + release_s) * SNES_RATE + 4 * edl * 512) + 2
    cnt = np.arange(m, dtype=np.int64) * pitch
    i0 = cnt >> 12
    ix = (cnt >> 4) & 255
    if gauss:
        y = sum(SNES_GAUSS[tab] * smp[(i0 - k) % n0] for k, tab in ((3, 255 - ix), (2, 511 - ix), (1, 256 + ix), (0, ix))) / 2048.0
    else:
        y = smp[i0 % n0]
    t = np.arange(m) / SNES_RATE
    at = max(SNES_ATTACK_S[int(np.clip(attack, 0, 15))], 1.0 / SNES_RATE)
    sl = (int(np.clip(sustain, 0, 7)) + 1) / 8.0
    d = np.maximum(t - at, 0.0)
    env = np.where(t < at, t / at, np.maximum(sl, np.exp(-d / (SNES_DECAY_S[int(np.clip(decay, 0, 7))] / 2.3))))
    if gain is not None:
        env = snes_gain_env(gain, m)
    elif sustain_s > 0:
        env = env * np.exp(-d / sustain_s)
    env = env * np.where(t > dur, np.exp(-(t - dur) / (max(release_s, 1e-3) / 4.6)), 1.0)
    y = y * env
    if edl > 0:
        y = snes_echo(y, edl, echo_fb, echo_mix, fir)
    return _to_sr(y, SNES_RATE, sr)


def snes_echo(y, edl, efb, evol, fir="echo"):
    """The S-DSP echo: a ring buffer of EDL x 16 ms (512 samples a step) read, filtered by the 8-tap FIR (signed coefficients, 128 = unity;
    a name of SNES_FIR or eight values) and written back as the input plus the FIR output x `efb`; the output is the input plus the FIR
    output x `evol`."""
    c = np.asarray(SNES_FIR[fir] if isinstance(fir, str) else fir, float)
    if c.shape != (8,):
        raise SpecError("snes: fir is a name of echo | lowpass | none or 8 coefficients")
    c = c / 128.0
    d = int(edl) * 512
    e = np.concatenate([np.zeros(7), np.asarray(y, float)])
    wet = np.zeros(len(e))
    for s in range(d, len(e) - 7, d):
        ln = min(d, len(e) - 7 - s)
        w = np.correlate(e[s - d:s - d + ln + 7], c, "valid")
        wet[7 + s:7 + s + ln] = w
        e[7 + s:7 + s + ln] += float(efb) * w
    return np.asarray(y, float) + float(evol) * wet[7:]


@instrument("snes", "Super Nintendo S-DSP voice: a looped BRR sample (4-bit blocks of 16 samples with four predictive filters), 14-bit pitch, 4-tap Gaussian interpolation at 32 kHz, ADSR, and the echo.",
            family="retro", span=("C1", "C7"),
            wave=("saw", "saw | sine | square | triangle | pulse, or a list of 16 x blocks values: the cycle that is encoded to BRR"),
            blocks=(2, "1..64 BRR blocks in the loop (a cycle of 16 x blocks samples: fewer samples, coarser, more aliased)"),
            gauss=(True, "the Gaussian interpolation (False: nearest sample, the bright aliasing of a sampler without it)"),
            attack=(15, "0..15 attack rate (4.1 s .. instant)"), decay=(7, "0..7 decay rate (1.2 s .. 37 ms)"), sustain=(7, "0..7 sustain level in eighths"),
            sustain_s=(0.0, "s the sustained level takes to fall by 1/e (0 = held)"), release_s=(0.05, "s"),
            echo_ms=(0.0, "echo delay in ms, 0 = none (rounded to EDL x 16 ms, up to 240)"), echo_fb=(0.4, "EFB: the filtered echo fed back"),
            echo_mix=(0.35, "EVOL: the filtered echo in the output"),
            fir=("echo", "echo | lowpass | none, or the 8 signed FIR coefficients (128 = unity)"),
            gain=(None, "VxGAIN byte replacing the ADSR: 0..127 fixed level (16 N / 2048); 128..255 = 128 + 32 mode + rate, mode 0 linear decrease, 1 exponential decrease, 2 linear increase, 3 bent increase (from level 0)"),
            mode=CT._MODE, bright=CT._BRIGHT)
def snes(freq, dur, sr=DEFAULT_SR, vel=1.0, wave="saw", blocks=2, gauss=True, attack=15, decay=7, sustain=7, sustain_s=0.0,
         release_s=0.05, echo_ms=0.0, echo_fb=0.4, echo_mix=0.35, fir="echo", gain=None, mode="retro_stylized", bright=0.5):
    y = render_snes(freq, dur, sr, wave, blocks, gauss, attack, decay, sustain, sustain_s, release_s, float(echo_ms), echo_fb, echo_mix, fir, gain)
    return _finish(y, freq, sr, vel, mode, bright, release_s, dur)


# ============================================================================ Philips SAA1099 (Sam Coupe, Creative Music System)
SAA_CLOCK = 8.0e6
SAA_NOISE_HZ = (31250.0, 15625.0, 7812.5)


def saa_register(hz: float):
    """(octave 0..7, note 0..255) for a pitch: f = clock / 512 x 2**octave / (511 - note)."""
    hz = max(float(hz), 1e-3)
    o = int(np.clip(np.floor(np.log2(511.0 * hz / (SAA_CLOCK / 512.0))), 0, 7))
    d = int(np.clip(round(SAA_CLOCK / 512.0 * 2.0 ** o / hz), 256, 511))
    return o, 511 - d


def saa_hz(hz: float) -> float:
    o, n = saa_register(hz)
    return SAA_CLOCK / 512.0 * 2.0 ** o / (511 - n)


@instrument("saa1099", "Philips SAA1099 (Sam Coupe, Creative Music System) voice: a square wave from an 8-octave, 8-bit note divider with 4-bit amplitude, optionally mixed with one of the two noise generators.",
            family="retro", span=("C1", "C7"),
            noise=(0.0, "0..1 share of the noise generator in the channel"),
            noise_rate=(0, "0..2 noise clock: 31.25 kHz, 15.6 kHz, 7.8 kHz"), volume=(15, "0..15, linear"),
            decay=(0.0, "s to fall 60 dB (0 = held)"), mode=CT._MODE, bright=CT._BRIGHT)
def saa1099(freq, dur, sr=DEFAULT_SR, vel=1.0, noise=0.0, noise_rate=0, volume=15, decay=0.0, mode="retro_stylized", bright=0.5):
    rel = 0.01
    n = samples(dur + rel, sr)
    f = saa_hz(freq)
    t = np.arange(n) / sr
    tone = np.where((t * f) % 1.0 < 0.5, 1.0, -1.0)
    nz = float(np.clip(noise, 0.0, 1.0))
    if nz > 0:
        clk = SAA_NOISE_HZ[int(np.clip(noise_rate, 0, 2))]
        k = int((dur + rel) * clk) + 2
        bits = np.empty(k)
        sh = 0x3FFFF
        for i in range(k):
            bits[i] = 1.0 if sh & 1 else -1.0
            sh = (sh >> 1) | ((((sh >> 17) ^ (sh >> 10)) & 1) << 17)
        tone = (1.0 - nz) * tone + nz * bits[np.minimum((t * clk).astype(np.int64), k - 1)]
    amp = int(np.clip(volume, 0, 15)) / 15.0
    if decay > 0:
        amp = amp * 10.0 ** (-3.0 * t / decay)
    return _finish(tone * amp * 0.5, freq, sr, vel, mode, bright, rel, dur)


from . import chiptune_p2  # noqa: E402,F401  (WonderSwan, PPG Wave, Optigan, Chamberlin)
