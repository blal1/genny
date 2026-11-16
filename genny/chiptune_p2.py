"""Retro chips and tape instruments, lot P2: Bandai WonderSwan, PPG Wave wavetables, Optigan optical disc, Chamberlin tape keyboard.

Registers
    instruments   `wonderswan`, `ppg_wave`, `optigan`, `chamberlin`
    functions     `ws_period`, `ws_hz`, `ppg_table`, `PPG_TABLES`
"""
from __future__ import annotations

import numpy as np

from . import chiptune as CT
from .chiptune_q import _finish, _fit, _table
from .core import DEFAULT_SR, samples
from .instruments import instrument
from .retro import _fade
from .spec import SpecError

# ============================================================================ Bandai WonderSwan
WS_CLOCK = 3072000.0
_X32 = np.arange(32)
WS_WAVES = {
    "sine": np.clip(np.round(7.5 + 7.5 * np.sin(2 * np.pi * _X32 / 32)), 0, 15),
    "saw": np.clip(np.round(_X32 / 2.0), 0, 15),
    "square": np.where(_X32 < 16, 15.0, 0.0),
    "pulse": np.where(_X32 < 8, 15.0, 0.0),
    "triangle": np.clip(np.round(np.where(_X32 < 16, _X32, 31 - _X32)), 0, 15),
    "organ": np.clip(np.round(7.5 + 7.5 * (np.sin(2 * np.pi * _X32 / 32) + 0.5 * np.sin(4 * np.pi * _X32 / 32)) / 1.5), 0, 15),
}
WS_NOISE_TAPS = (14, 10, 13, 4, 8, 6, 9, 11)


def ws_period(hz: float) -> int:
    """11-bit frequency register: the wave steps at clock / (2048 - N), 32 steps a cycle."""
    return int(min(max(2048 - round(WS_CLOCK / (32.0 * max(float(hz), 1e-3))), 0), 2047))


def ws_hz(hz: float) -> float:
    return WS_CLOCK / (2048 - ws_period(hz)) / 32.0


@instrument("wonderswan", "Bandai WonderSwan voice: a 32-step 4-bit wave (or the 8-bit voice channel, or the noise channel), 11-bit frequency register from a 3.072 MHz clock, 4-bit volume, optional pitch sweep.",
            family="retro", span=("C1", "C7"),
            channel=("wave", "wave | voice | noise: channel 1 / 3 wavetable, channel 2 voice (8-bit samples, volume in quarters), channel 4 noise (15-bit LFSR)"),
            wave=("saw", "saw | sine | square | pulse | triangle | organ, or a list of 32 values 0..15 (0..255 for the voice channel)"),
            volume=(15, "0..15, linear (voice channel 0..3: 100 %, 50 %, 25 %, 12.5 %)"),
            sweep=(0, "register steps added every 32 ms, signed (channel 3 sweep)"),
            taps=(0, "0..7 noise LFSR tap set (short, buzzy patterns .. long, hissing)"),
            mode=CT._MODE, bright=CT._BRIGHT)
def wonderswan(freq, dur, sr=DEFAULT_SR, vel=1.0, channel="wave", wave="saw", volume=15, sweep=0, taps=0,
               mode="retro_stylized", bright=0.5):
    if channel not in ("wave", "voice", "noise"):
        raise SpecError("wonderswan: channel is wave, voice or noise")
    rel = 0.01
    n = samples(dur + rel, sr)
    t = np.arange(n) / sr
    reg = ws_period(freq) + int(sweep) * np.floor(t / 0.032)
    reg = np.clip(reg, 0, 2046)
    step = WS_CLOCK / (2048.0 - reg)
    acc = np.cumsum(step / sr)
    if channel == "noise":
        k = int(acc[-1]) + 2
        a = WS_NOISE_TAPS[int(np.clip(taps, 0, 7))]
        sh = 0
        bits = np.empty(k)
        for i in range(k):
            fb = 1 ^ (((sh >> 7) ^ (sh >> a)) & 1)
            sh = ((sh << 1) & 0x7FFF) | fb
            bits[i] = 1.0 if sh & 1 else -1.0
        y = bits[np.minimum(acc.astype(np.int64), k - 1)] * int(np.clip(volume, 0, 15)) / 15.0
    elif channel == "voice":
        tab = _table(wave, {k: v * 17.0 for k, v in WS_WAVES.items()}, 32, 0, 255, "wonderswan")
        y = (tab[(acc / 32.0 * 32).astype(np.int64) % 32] - 128.0) / 128.0 * 0.5 ** int(np.clip(volume, 0, 3))
    else:
        tab = _table(wave, WS_WAVES, 32, 0, 15, "wonderswan")
        y = (tab[(acc % 32).astype(np.int64)] - 7.5) / 7.5 * int(np.clip(volume, 0, 15)) / 15.0
    return _finish(y * 0.5, freq, sr, vel, mode, bright, rel, dur, tonal=channel != "noise")


# ============================================================================ PPG Wave
PPG_WAVES, PPG_LEN = 64, 64


def _ppg_wave(kind, k):
    h = np.arange(1, PPG_LEN // 2)
    x = np.arange(PPG_LEN) / PPG_LEN
    f = k / (PPG_WAVES - 1.0)
    if kind == "resonant":
        p = 1.0 + 14.0 * f
        a = (1.0 / h) * (1.0 + 6.0 * np.exp(-(((h - p) / 1.3) ** 2))) * np.where(h <= p * 1.5, 1.0, np.exp(-(h - p * 1.5) / 2.0))
    elif kind == "harmonics":
        a = np.where(h <= k + 1, 1.0 / h, 0.0)
    elif kind == "pulse":
        d = 0.5 - 0.47 * f
        a = np.abs(np.sin(np.pi * h * d)) / h
    elif kind == "formant":
        c1, c2 = 2.0 + 6.0 * f, 6.0 + 14.0 * f
        a = (np.exp(-(((h - c1) / 1.5) ** 2)) + 0.6 * np.exp(-(((h - c2) / 2.0) ** 2))) / np.sqrt(h) + 0.05 / h
    elif kind == "organ":
        bars = (1, 2, 3, 4, 6, 8)
        a = np.zeros(len(h))
        for b, hh in enumerate(bars):
            if (k >> b) & 1:
                a[hh - 1] = 1.0 / np.sqrt(hh)
        a[0] = max(a[0], 0.3)
    elif kind == "sync":
        r = 1.0 + 7.0 * f
        w = 2 * ((x * r) % 1.0) - 1.0
        return w
    else:
        raise SpecError(f"ppg: table must be one of {sorted(PPG_TABLES)}")
    w = (a[:, None] * np.sin(2 * np.pi * h[:, None] * x[None, :] + 0.0)).sum(0)
    return w


PPG_TABLES = ("resonant", "harmonics", "pulse", "formant", "organ", "sync")


def ppg_table(kind="resonant"):
    """A 64 x 64 wavetable of 8-bit values in -1..1 (steps of 1/127), each wave of the kind scaled to full range."""
    tab = np.stack([_ppg_wave(kind, k) for k in range(PPG_WAVES)])
    tab = tab / (np.max(np.abs(tab), axis=1, keepdims=True) + 1e-12)
    return np.round(tab * 127.0) / 127.0


@instrument("ppg_wave", "PPG Wave wavetable voice: 64 waves of 64 8-bit samples read without interpolation, the wave number swept over the note (stepped, as the digital oscillator does), analogue low-pass behind.",
            family="synth", span=("C2", "C6"),
            table=("resonant", "resonant | harmonics | pulse | formant | organ | sync, or a 64 x 64 array of -1..1"),
            pos=(0.0, "first wave 0..63"), pos_end=(32.0, "wave reached after `sweep_s`"), sweep_s=(0.6, "s to sweep from pos to pos_end"),
            smooth=(False, "interpolate between neighbouring waves (the stepped sweep is the PPG's)"),
            cutoff=(9000.0, "Hz of the analogue low-pass after the oscillator"), release=(0.15, "s"))
def ppg_wave(freq, dur, sr=DEFAULT_SR, vel=1.0, table="resonant", pos=0.0, pos_end=32.0, sweep_s=0.6, smooth=False,
             cutoff=9000.0, release=0.15):
    from . import filters as F
    tab = ppg_table(table) if isinstance(table, str) else np.asarray(table, float)
    if tab.shape != (PPG_WAVES, PPG_LEN):
        raise SpecError("ppg_wave: a table is 64 x 64")
    n = samples(dur + release, sr)
    t = np.arange(n) / sr
    w = np.clip(pos + (pos_end - pos) * np.minimum(t / max(sweep_s, 1e-3), 1.0), 0, PPG_WAVES - 1)
    ph = (np.cumsum(np.full(n, freq / sr)) % 1.0) * PPG_LEN
    i = ph.astype(np.int64) % PPG_LEN
    if smooth:
        k0 = np.floor(w).astype(np.int64)
        k1 = np.minimum(k0 + 1, PPG_WAVES - 1)
        fr = w - k0
        y = tab[k0, i] * (1 - fr) + tab[k1, i] * fr
    else:
        y = tab[np.round(w).astype(np.int64), i]
    cutoff = min(float(cutoff), float(np.clip(6.0 * freq, 2500.0, 3500.0)))     # 8-bit waves read without interpolation: the stepped alias sits high
    if cutoff < 0.45 * sr:
        y = F.lowpass(F.lowpass(y, float(cutoff), sr, 0.9), float(cutoff), sr, 0.7)
    y = _fade(y.copy(), sr, 3.0, 1.0)
    t_rel = np.maximum(t - dur, 0.0)
    y = y * np.exp(-t_rel / max(release / 4.6, 1e-3))
    return y * 0.4 * (0.4 + 0.6 * vel)


# ============================================================================ Optigan and Chamberlin
def _recorded_tone(freq, n, sr, harm, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    y = np.zeros(n)
    for v in (-0.0035, 0.0, 0.004):
        f = freq * (1 + v)
        for h, a in enumerate(harm, 1):
            if f * h < 0.45 * sr:
                y += a * np.sin(2 * np.pi * f * h * t + rng.uniform(0, 6.28)) / 3.0
    return y


OPTIGAN_SOUNDS = {"organ": (1.0, 0.55, 0.35, 0.2, 0.12, 0.08), "strings": (1.0, 0.7, 0.55, 0.4, 0.3, 0.22, 0.15, 0.1),
                  "accordion": (1.0, 0.25, 0.7, 0.2, 0.45, 0.15, 0.3), "flute": (1.0, 0.12, 0.05)}


@instrument("optigan", "Optigan: an optical disc of looped recordings read by a lamp and photocell: a spliced loop every turn of the disc, a dull photocell, wow and flutter of the turntable, hiss and disc noise.",
            family="keys", span=("C3", "C5"),
            sound=("organ", "organ | strings | accordion | flute: the recorded pattern"), disc_s=(1.1, "s of one turn of the disc: the loop and its splice"),
            wow=(0.25, "slow drift, percent"), flutter=(0.15, "fast wobble, percent"), photocell=(4500.0, "Hz of the photocell's high roll-off"),
            noise=(-46.0, "hiss and disc noise, dB"), seed=(0, "int"))
def optigan(freq, dur, sr=DEFAULT_SR, vel=1.0, sound="organ", disc_s=1.1, wow=0.25, flutter=0.15, photocell=4500.0, noise=-46.0, seed=0):
    from . import filters as F
    from .dsp import tape
    if sound not in OPTIGAN_SOUNDS:
        raise SpecError(f"optigan: sound must be one of {sorted(OPTIGAN_SOUNDS)}")
    rel = 0.2
    n = samples(dur + rel, sr)
    ln = max(int(disc_s * sr), 16)
    loop = _recorded_tone(freq, ln, sr, OPTIGAN_SOUNDS[sound], int(seed))
    y = np.resize(loop, n)
    t = np.arange(n) / sr
    y = y * np.minimum(t / 0.03, 1.0) * np.where(t > dur, np.exp(-(t - dur) / (rel / 4.6)), 1.0)
    y = F.lowpass(F.lowpass(y, float(min(photocell, 0.45 * sr)), sr, 0.7), float(min(photocell, 0.45 * sr)), sr, 0.7)
    rng = np.random.default_rng(int(seed) + 5)
    y = y + 10.0 ** (noise / 20.0) * rng.standard_normal(n) * (1.0 + 2.0 * (np.floor(t / disc_s) != np.floor((t - 0.004) / disc_s)))
    y = tape(y, sr, wow=wow, flutter=flutter, drive=0.8, bump=0.0, hf=float(min(photocell * 1.2, 0.45 * sr)), hiss=-120, seed=int(seed))
    return y * 0.35 * (0.4 + 0.6 * vel)


@instrument("chamberlin", "Chamberlin tape keyboard: one tape strip per key (8 s, then the note stops), a pinch-roller drive that takes 60 ms to come up to speed, heavy wow and flutter, hiss.",
            family="keys", span=("F2", "F5"),
            sound=("flute", "flute | strings | brass | voice: the recorded instrument"), tape_s=(8.0, "seconds of tape behind each key"),
            wow=(0.35, "slow drift, percent"), flutter=(0.3, "fast wobble, percent"), start_ms=(60.0, "run-up of the drive to full speed"),
            seed=(0, "int"))
def chamberlin(freq, dur, sr=DEFAULT_SR, vel=1.0, sound="flute", tape_s=8.0, wow=0.35, flutter=0.3, start_ms=60.0, seed=0):
    from .dsp import tape, varispeed
    from .instruments import REGISTRY
    names = {"flute": "flute", "strings": "strings", "brass": "brass", "voice": "choir"}
    if sound not in names or names[sound] not in REGISTRY:
        raise SpecError("chamberlin: sound is flute, strings, brass or voice")
    held = min(float(dur), float(tape_s))
    from .notes import midi_to_freq, note_to_midi
    low = midi_to_freq(note_to_midi(REGISTRY[names[sound]]["range"].split("-")[0]))
    k = max(0, int(np.ceil(np.log2(low / freq - 1e-9)))) if freq < low else 0     # a tape of a note below the instrument's range is a higher note played slower
    y = np.asarray(REGISTRY[names[sound]]["fn"](freq * 2 ** k, held / 2 ** k, sr, vel), float)
    if y.ndim == 2:
        y = y.mean(1)
    n = len(y) * 2 ** k
    t = np.arange(n) / sr
    rate = (1.0 - 0.2 * np.exp(-t / max(start_ms * 1e-3 / 7.0, 1e-4))) / 2 ** k
    y = varispeed(y, rate)
    y = tape(y, sr, wow=wow, flutter=flutter, drive=1.3, bump=1.0, hf=7500, hiss=-55, seed=int(seed))
    return _fit(y, samples(held + 0.1, sr)) * 0.6
