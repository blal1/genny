"""Retro sound generators: sfxr / Bfxr, ZzFX, an AY-3-8910 PSG voice and a tracker `song` layer.

Registers
    sfx         `sfxr`, `zzfx`
    instrument  `sfxr_voice`, `zzfx_voice`, `psg`
    layer type  `song`

Sources (findings in out/research):
    g17-retro-sfx-generators.md  §A sfxr engine (jsfxr `sfxr.js`) and Bfxr extras, §B presets,
                                 §C ZzFX 1.3.2 `buildSamples`, §D ZzFXM song format, §F design.
    g15-course-farnell-composition-misc.md §2  General Instrument AY-3-8910 datasheet.

sfxr and ZzFX are defined at 44100 Hz (period ticks, sample counts), so both kernels always run at
44100 and the result is resampled (`scipy.signal.resample_poly`) to the requested rate.
"""
from __future__ import annotations

import math
import re
import zlib
from fractions import Fraction

import numba as nb
import numpy as np
from scipy.signal import resample_poly

from . import drums as D
from . import filters as F
from . import instruments as I
from .core import DEFAULT_SR, mix, samples
from .instruments import _soften, instrument
from .notes import midi_to_freq, note_to_midi, parse_pitch_list, to_hz
from .sfx import sfx
from .spec import layer_type

RSR = 44100          # native rate of sfxr and ZzFX


# ============================================================================ shared helpers
def _seed(seed) -> int:
    """int, or a string hashed with crc32 (stable across runs, unlike hash())."""
    if isinstance(seed, str):
        return zlib.crc32(seed.encode("utf-8"))
    return int(seed) & 0xFFFFFFFF


def _to_sr(x: np.ndarray, sr_from: float, sr: int) -> np.ndarray:
    if x.size == 0 or sr_from == sr:
        return x
    fr = Fraction(sr / sr_from).limit_denominator(1000)
    return resample_poly(x, fr.numerator, fr.denominator)


def _fade(x: np.ndarray, sr: int, ms_in: float = 2.0, ms_out: float = 2.0) -> np.ndarray:
    a = min(int(sr * ms_in / 1000), x.shape[0] // 2)
    b = min(int(sr * ms_out / 1000), x.shape[0] // 2)
    if a > 0:
        x[:a] *= np.linspace(0.0, 1.0, a)
    if b > 0:
        x[-b:] *= np.linspace(1.0, 0.0, b)
    return x


def _round_top(x: np.ndarray, soften: float) -> np.ndarray:
    """`soften` 0 = raw, 1 = 12 dB/oct lowpass at 4 kHz (log-interpolated from 16 kHz)."""
    s = float(np.clip(soften, 0.0, 1.0))
    if s <= 0.0 or x.size < 8:
        return x
    return F.lowpass(x, 16000.0 * (4000.0 / 16000.0) ** s, RSR, 0.707)


def _finish(x: np.ndarray, sr: int, level: float, soften: float, cut: bool = False) -> np.ndarray:
    """DC removal, optional rounding, resample 44100 -> sr, 2 ms fades (20 ms when `cut`), peak = level."""
    if x.size == 0 or not np.any(np.abs(x) > 1e-9):
        return np.zeros(samples(0.01, sr))        # the model produced nothing (e.g. freq_limit above freq)
    x = _round_top(x - x.mean(), soften)
    x = _fade(_to_sr(x, RSR, sr).copy(), sr, 2.0, 20.0 if cut else 2.0)
    peak = float(np.max(np.abs(x)))
    if peak < 1e-9:
        return np.zeros(samples(0.01, sr))
    return x * (float(np.clip(level, 1e-3, 1.4)) / peak)


# ============================================================================ sfxr / Bfxr engine
# The 22 float knobs in jsfxr `params_order` (g17 §A.1), under readable names.
KNOBS = ["attack", "sustain", "punch", "decay", "freq", "freq_limit", "slide", "dslide", "vib_depth", "vib_speed",
         "arp_mod", "arp_speed", "duty", "duty_sweep", "repeat", "flange_offset", "flange_sweep", "lpf", "lpf_sweep",
         "resonance", "hpf", "hpf_sweep"]
_KDEF = dict.fromkeys(KNOBS, 0.0) | {"sustain": 0.3, "decay": 0.4, "freq": 0.3, "lpf": 1.0}
_SIGNED = {"slide", "dslide", "arp_mod", "duty_sweep", "flange_offset", "flange_sweep", "lpf_sweep", "hpf_sweep"}
_KHELP = {
    "attack": "envelope attack 0..1 (v^2 * 2.27 s)", "sustain": "envelope sustain 0..1 (v^2 * 2.27 s)",
    "punch": "sustain punch 0..1 (start of sustain up to 3x)", "decay": "envelope decay 0..1 (v^2 * 2.27 s)",
    "freq": "base pitch 0..1 (Hz = 3528*(v^2+0.001); 0.3517 = 440 Hz)", "freq_limit": "0..1, same scale as freq: the sound ends when the pitch falls to it (0 = off)",
    "slide": "pitch slide -1..1 (>0 up)", "dslide": "slide acceleration -1..1",
    "vib_depth": "vibrato depth 0..1", "vib_speed": "vibrato speed 0..1 (v^2 * 70 Hz)",
    "arp_mod": "one pitch jump -1..1 (>0 up; 0.745 = +12 st, -0.316 = -12 st)", "arp_speed": "0..1: delay before the jump, (1-v)^2 * 0.45 s",
    "duty": "0..1: square duty 50% -> 0%; saw: 0 = triangle, 1 = pure saw", "duty_sweep": "duty sweep -1..1",
    "repeat": "0..1 retrigger rate of pitch/duty/arp (0 = off, period (1-v)^2 * 0.45 s)",
    "flange_offset": "flanger delay -1..1 (v^2 * 23 ms)", "flange_sweep": "flanger sweep -1..1",
    "lpf": "low-pass cutoff 0..1 (1 = off)", "lpf_sweep": "low-pass sweep -1..1", "resonance": "low-pass resonance 0..1",
    "hpf": "high-pass cutoff 0..1", "hpf_sweep": "high-pass sweep -1..1",
}
SFXR_WAVES = ["square", "saw", "sine", "noise", "triangle", "breaker", "tan", "whistle", "bitnoise"]   # 4.. = Bfxr extras


def hz_to_knob(hz: float) -> float:
    """sfxr base-frequency knob for a pitch (g17 §A.3). The engine floors the period to whole 1/8-sample
    ticks, so the period is first rounded to the nearest tick (+0.5 so the floor lands on it)."""
    period = min(max(round(8.0 * RSR / max(float(hz), 3.6)), 8), 99000) + 0.5
    return math.sqrt(max(100.0 / period - 0.001, 0.0))


@nb.njit(cache=True)
def _wave(w, x, duty, noise, idx, lfsr):
    """One sfxr/Bfxr waveform sample; x = phase in [0, 1), idx = noise-buffer index for that phase."""
    if w == 0:
        return 0.5 if x < duty else -0.5
    if w == 1:                                    # sfxr saw: duty 0.5 = triangle, duty 0 = falling saw
        return -1.0 + 2.0 * x / duty if x < duty else 1.0 - 2.0 * (x - duty) / (1.0 - duty)
    if w == 2:
        return np.sin(x * 2.0 * np.pi)
    if w == 3:
        return noise[idx]
    if w == 4:                                    # Bfxr triangle without its DC offset (g17 §A.5)
        return 2.0 * abs(1.0 - 2.0 * x) - 1.0
    if w == 5:                                    # breaker, jfxr's zero-DC form
        q = (x + 0.8660254037844386) % 1.0
        return -1.0 + 2.0 * abs(1.0 - 2.0 * q * q)
    if w == 6:                                    # tangent, jfxr's bounded form scaled to +-1
        return 0.5 * min(max(0.3 * np.tan(np.pi * x), -2.0), 2.0)
    if w == 7:
        return 0.75 * np.sin(2.0 * np.pi * x) + 0.25 * np.sin(40.0 * np.pi * x)
    return (1.0 - float(lfsr & 1)) - 0.5          # 8: 1-bit LFSR noise


@nb.njit(cache=True)
def _init_repeat(P):
    period = 100.0 / (P[4] * P[4] + 0.001)
    period_max = 100.0 / (P[5] * P[5] + 0.001)
    cut_en = P[5] > 0
    pmult = 1.0 - P[6] ** 3 * 0.01
    pmult_slide = -(P[7] ** 3) * 0.000001
    duty = 0.5 - P[12] * 0.5
    duty_slide = -P[13] * 0.00005
    arp_mult = 1.0 - P[10] ** 2 * 0.9 if P[10] >= 0 else 1.0 + P[10] ** 2 * 10.0
    arp_time = np.floor((1.0 - P[11]) ** 2 * 20000.0 + 32.0)
    if P[11] == 1.0:
        arp_time = 0.0
    return period, period_max, cut_en, pmult, pmult_slide, duty, duty_slide, arp_mult, arp_time


@nb.njit(cache=True)
def sfxr_core(wave, P, sound_vol, seed, cap, nharm, hfall):
    """The jsfxr synthesis loop (g17 §A.2: validated against `jsfxr/sfxr.js`), 8x oversampled, at 44100 Hz.

    P = float64[22] in `KNOBS` order; returns unclipped samples with gain exp(sound_vol)-1.
    With wave 0..3 and nharm = 0 this is sample-exact jsfxr (envelope stages of length 0 give 0, not NaN).
    Extras from Bfxr (§A.5): waves 4..8 and `nharm` overtones, each `1 - hfall` times the previous;
    the overtone sum is normalised (jfxr) instead of clipped (Bfxr)."""
    OS = 8
    np.random.seed(seed)
    period, period_max, cut_en, pmult, pmult_slide, duty, duty_slide, arp_mult, arp_time = _init_repeat(P)
    fltw = P[17] ** 3 * 0.1
    lp_on = P[17] != 1.0
    fltw_d = 1.0 + P[18] * 0.0001
    fltdmp = min(0.8, 5.0 / (1.0 + P[19] ** 2 * 20.0) * (0.01 + fltw))
    flthp = P[20] ** 2 * 0.1
    flthp_d = 1.0 + P[21] * 0.0003
    vib_speed = P[9] ** 2 * 0.01
    vib_amp = P[8] * 0.5
    env_len = np.array([np.floor(P[0] * P[0] * 100000.0), np.floor(P[1] * P[1] * 100000.0), np.floor(P[3] * P[3] * 100000.0)])
    punch = P[2]
    fl_off = P[15] ** 2 * 1020.0
    if P[15] < 0:
        fl_off = -fl_off
    fl_slide = P[16] ** 2
    if P[16] < 0:
        fl_slide = -fl_slide
    rep_time = np.floor((1.0 - P[14]) ** 2 * 20000.0 + 32.0)
    if P[14] == 0.0:
        rep_time = 0.0
    gain = np.exp(sound_vol) - 1.0
    hnorm = 0.0
    st = 1.0
    for k in range(nharm + 1):
        hnorm += st
        st *= 1.0 - hfall
    fltp = 0.0
    fltdp = 0.0
    fltphp = 0.0
    noise = np.random.random(32) * 2 - 1
    lfsr = 1 << 14
    stage = 0
    elapsed = 0
    vib_phase = 0.0
    phase = 0
    ipp = 0
    fbuf = np.zeros(1024)
    out = np.empty(cap)
    n = 0
    since_rep = 0
    t = 0
    while n < cap:
        if rep_time != 0:
            since_rep += 1
            if since_rep >= rep_time:
                since_rep = 0
                period, period_max, cut_en, pmult, pmult_slide, duty, duty_slide, arp_mult, arp_time = _init_repeat(P)
        if arp_time != 0 and t >= arp_time:       # jsfxr compares with the global sample counter (§A.2 note 6)
            arp_time = 0.0
            period *= arp_mult
        pmult += pmult_slide
        period *= pmult
        if period > period_max:
            period = period_max
            if cut_en:
                break                             # frequency cutoff ends the sound
        rfp = period
        if vib_amp > 0:
            vib_phase += vib_speed
            rfp = period * (1.0 + np.sin(vib_phase) * vib_amp)
        iperiod = max(int(np.floor(rfp)), OS)
        duty = min(max(duty + duty_slide, 0.0), 0.5)
        elapsed += 1
        if elapsed > env_len[stage]:
            elapsed = 0
            stage += 1
            if stage > 2:
                break
        L = env_len[stage]
        envf = elapsed / L if L > 0 else 0.0
        if stage == 0:
            env_vol = envf
        elif stage == 1:
            env_vol = 1.0 + (1.0 - envf) * 2.0 * punch
        else:
            env_vol = 1.0 - envf
        fl_off += fl_slide
        iphase = min(abs(int(np.floor(fl_off))), 1023)
        flthp = min(max(flthp * flthp_d, 0.00001), 0.1)
        sample = 0.0
        for si in range(OS):
            phase += 1
            if phase >= iperiod:
                phase = phase % iperiod
                if wave == 3:
                    noise = np.random.random(32) * 2 - 1
                elif wave == 8:                    # 15-bit LFSR clocked once per period (§A.5 wave 9)
                    fb = ((lfsr >> 1) & 1) ^ (lfsr & 1)
                    lfsr = (lfsr >> 1) | (fb << 14)
            if nharm == 0:
                sub = _wave(wave, phase / iperiod, duty, noise, (phase * 32) // iperiod, lfsr)
            else:
                sub = 0.0
                st = 1.0
                for k in range(nharm + 1):
                    tp = (phase * (k + 1)) % iperiod
                    sub += st * _wave(wave, tp / iperiod, duty, noise, (tp * 32) // iperiod, lfsr)
                    st *= 1.0 - hfall
                sub /= hnorm
            pp = fltp
            fltw = min(max(fltw * fltw_d, 0.0), 0.1)          # swept every sub-sample
            if lp_on:
                fltdp += (sub - fltp) * fltw
                fltdp -= fltdp * fltdmp
            else:
                fltp = sub
                fltdp = 0.0
            fltp += fltdp
            fltphp += fltp - pp                               # high-pass on the low-pass output
            fltphp -= fltphp * flthp
            sub = fltphp
            fbuf[ipp & 1023] = sub                            # flanger: add own delayed copy
            sub += fbuf[(ipp - iphase + 1024) & 1023]
            ipp = (ipp + 1) & 1023
            sample += sub * env_vol
        out[n] = sample / OS * gain
        n += 1
        t += 1
    return out[:n]


@nb.njit(cache=True)
def _crush(x, amount, sweep):
    """Bfxr sample-and-hold 'bit crush' (g17 §A.5.4). The phase wraps by subtraction, so amount 0 is
    transparent (Bfxr resets it to 0, which halves the rate even when the effect is off)."""
    n = x.shape[0]
    freq = 1.0 - amount ** (1.0 / 3.0)
    rate = -sweep / max(n, 1)
    ph = 1.0
    last = 0.0
    y = np.empty(n)
    for i in range(n):
        ph += freq
        if ph >= 1.0:
            ph -= 1.0
            last = x[i]
        y[i] = last
        r = np.sqrt(freq)
        freq = min(max(freq + (1.0 + (50.0 * freq - 1.0) * r) * rate, 1e-5), 1.0)
    return y


# ---------------------------------------------------------------- presets (jsfxr Params.prototype.*, g17 §B.1)
def _sfxr_presets():
    SQ, SAW, SIN, NOI = 0, 1, 2, 3

    def pickup(p, U, rnd):
        p["wave"] = SAW
        p["freq"] = 0.4 + U(0.5)
        p["attack"] = 0
        p["sustain"] = U(0.1)
        p["decay"] = 0.1 + U(0.4)
        p["punch"] = 0.3 + U(0.3)
        if rnd(1):
            p["arp_speed"] = 0.5 + U(0.2)
            p["arp_mod"] = 0.2 + U(0.4)

    def laser(p, U, rnd):
        p["wave"] = rnd(2)
        if p["wave"] == SIN and rnd(1):
            p["wave"] = rnd(1)
        if rnd(2) == 0:
            p["freq"] = 0.3 + U(0.6)
            p["freq_limit"] = U(0.1)
            p["slide"] = -0.35 - U(0.3)
        else:
            p["freq"] = 0.5 + U(0.5)
            p["freq_limit"] = max(0.2, p["freq"] - 0.2 - U(0.6))
            p["slide"] = -0.15 - U(0.2)
        if rnd(1):                                  # (jsfxr sets duty 1 for saw and then always overwrites it)
            p["duty"] = U(0.5)
            p["duty_sweep"] = U(0.2)
        else:
            p["duty"] = 0.4 + U(0.5)
            p["duty_sweep"] = -U(0.7)
        p["attack"] = 0
        p["sustain"] = 0.1 + U(0.2)
        p["decay"] = U(0.4)
        if rnd(1):
            p["punch"] = U(0.3)
        if rnd(2) == 0:
            p["flange_offset"] = U(0.2)
            p["flange_sweep"] = -U(0.2)
        p["hpf"] = U(0.3)

    def explosion(p, U, rnd):
        p["wave"] = NOI
        if rnd(1):
            p["freq"] = (0.1 + U(0.4)) ** 2
            p["slide"] = -0.1 + U(0.4)
        else:
            p["freq"] = (0.2 + U(0.7)) ** 2
            p["slide"] = -0.2 - U(0.2)
        if rnd(4) == 0:
            p["slide"] = 0
        if rnd(2) == 0:
            p["repeat"] = 0.3 + U(0.5)
        p["attack"] = 0
        p["sustain"] = 0.1 + U(0.3)
        p["decay"] = U(0.5)
        if rnd(1):
            p["flange_offset"] = -0.3 + U(0.9)
            p["flange_sweep"] = -U(0.3)
        p["punch"] = 0.2 + U(0.6)
        if rnd(1):
            p["vib_depth"] = U(0.7)
            p["vib_speed"] = U(0.6)
        if rnd(2) == 0:
            p["arp_speed"] = 0.6 + U(0.3)
            p["arp_mod"] = 0.8 - U(1.6)

    def powerup(p, U, rnd):
        if rnd(1):
            p["wave"] = SAW
            p["duty"] = 1
        else:
            p["duty"] = U(0.6)
        p["freq"] = 0.2 + U(0.3)
        if rnd(1):
            p["slide"] = 0.1 + U(0.4)
            p["repeat"] = 0.4 + U(0.4)
        else:
            p["slide"] = 0.05 + U(0.2)
            if rnd(1):
                p["vib_depth"] = U(0.7)
                p["vib_speed"] = U(0.6)
        p["attack"] = 0
        p["sustain"] = U(0.4)
        p["decay"] = 0.1 + U(0.4)

    def hit(p, U, rnd):
        p["wave"] = rnd(2)
        if p["wave"] == SIN:
            p["wave"] = NOI
        if p["wave"] == SQ:
            p["duty"] = U(0.6)
        if p["wave"] == SAW:
            p["duty"] = 1
        p["freq"] = 0.2 + U(0.6)
        p["slide"] = -0.3 - U(0.4)
        p["attack"] = 0
        p["sustain"] = U(0.1)
        p["decay"] = 0.1 + U(0.2)
        if rnd(1):
            p["hpf"] = U(0.3)

    def jump(p, U, rnd):
        p["wave"] = SQ
        p["duty"] = U(0.6)
        p["freq"] = 0.3 + U(0.3)
        p["slide"] = 0.1 + U(0.2)
        p["attack"] = 0
        p["sustain"] = 0.1 + U(0.3)
        p["decay"] = 0.1 + U(0.2)
        if rnd(1):
            p["hpf"] = U(0.3)
        if rnd(1):
            p["lpf"] = 1 - U(0.6)

    def blip(p, U, rnd):
        p["wave"] = rnd(1)
        p["duty"] = U(0.6) if p["wave"] == SQ else 1
        p["freq"] = 0.2 + U(0.4)
        p["attack"] = 0
        p["sustain"] = 0.1 + U(0.1)
        p["decay"] = U(0.2)
        p["hpf"] = 0.1

    def synth(p, U, rnd):
        p["wave"] = rnd(1)
        p["freq"] = [0.2723171360931539, 0.19255692561524382, 0.13615778746815113][rnd(2)]
        p["attack"] = U(0.5) if rnd(4) > 3 else 0
        p["sustain"] = U(1)
        p["punch"] = U(1)
        p["decay"] = U(0.9) + 0.1
        p["arp_mod"] = [0, 0, 0, 0, -0.3162, 0.7454, 0.7454][rnd(6)]
        p["arp_speed"] = U(0.5) + 0.4
        p["duty"] = U(1)
        p["duty_sweep"] = U(1) if rnd(2) == 2 else 0
        p["lpf"] = [1, 0.9 * U(1) * U(1) + 0.1][rnd(1)]
        p["lpf_sweep"] = U(2) - 1
        p["resonance"] = U(1)
        p["hpf"] = U(1) if rnd(3) == 3 else 0
        p["hpf_sweep"] = U(1) if rnd(3) == 3 else 0

    def tone(p, U, rnd):
        p["wave"] = SIN
        p["freq"] = 0.35173364          # 440 Hz
        p["attack"] = 0
        p["sustain"] = 0.6641           # 1 s
        p["decay"] = 0
        p["punch"] = 0

    def click(p, U, rnd):
        (explosion, hit)[rnd(1)](p, U, rnd)
        if rnd(1):
            p["slide"] = -0.5 + U(1.0)
        if rnd(1):
            p["sustain"] = (U(0.4) + 0.2) * p["sustain"]
            p["decay"] = (U(0.4) + 0.2) * p["decay"]
        if rnd(3) == 0:
            p["attack"] = U(0.3)
        p["freq"] = 1 - U(0.25)
        p["hpf"] = 1 - U(0.1)

    def random(p, U, rnd):
        p["wave"] = rnd(3)
        p["freq"] = (U(2) - 1) ** 3 + 0.5 if rnd(1) else U(1) ** 2
        p["freq_limit"] = 0
        p["slide"] = (U(2) - 1) ** 5
        if p["freq"] > 0.7 and p["slide"] > 0.2:
            p["slide"] = -p["slide"]
        if p["freq"] < 0.2 and p["slide"] < -0.05:
            p["slide"] = -p["slide"]
        p["dslide"] = (U(2) - 1) ** 3
        p["duty"] = U(2) - 1
        p["duty_sweep"] = (U(2) - 1) ** 3
        p["vib_depth"] = (U(2) - 1) ** 3
        p["vib_speed"] = U(2) - 1
        p["attack"] = (U(2) - 1) ** 3
        p["sustain"] = (U(2) - 1) ** 2
        p["decay"] = U(2) - 1
        p["punch"] = U(0.8) ** 2
        if p["attack"] + p["sustain"] + p["decay"] < 0.2:
            p["sustain"] += 0.2 + U(0.3)
            p["decay"] += 0.2 + U(0.3)
        p["resonance"] = U(2) - 1
        p["lpf"] = 1 - U(1) ** 3
        p["lpf_sweep"] = (U(2) - 1) ** 3
        if p["lpf"] < 0.1 and p["lpf_sweep"] < -0.05:
            p["lpf_sweep"] = -p["lpf_sweep"]
        p["hpf"] = U(1) ** 5
        p["hpf_sweep"] = (U(2) - 1) ** 5
        p["flange_offset"] = (U(2) - 1) ** 3
        p["flange_sweep"] = (U(2) - 1) ** 3
        p["repeat"] = U(2) - 1
        p["arp_speed"] = U(2) - 1
        p["arp_mod"] = U(2) - 1

    return {"pickup": pickup, "laser": laser, "explosion": explosion, "powerup": powerup, "hit": hit, "jump": jump,
            "blip": blip, "synth": synth, "tone": tone, "click": click, "random": random}


SFXR_PRESETS = _sfxr_presets()
_ALIAS = {"coin": "pickup", "shoot": "laser", "hurt": "hit", "select": "blip"}
_MUTATED = [k for k in KNOBS if k != "freq_limit"]          # jsfxr mutate never touches freq_limit / wave / volume


def _wave_index(wave) -> int:
    if isinstance(wave, str):
        w = wave.lower()
        if w not in SFXR_WAVES:
            raise ValueError(f"unknown sfxr wave {wave!r}; use one of {SFXR_WAVES}")
        return SFXR_WAVES.index(w)
    return int(wave)


def sfxr_params(preset="none", seed=0, mutate=0, wave=None, **knobs) -> dict:
    """Knob dict {'wave': int, <22 knobs>}: defaults <- preset (seeded) <- `mutate` rounds <- explicit knobs.

    Preset draws are kept exactly as jsfxr makes them (the `random` preset leaves [0, 1] on purpose);
    mutated and explicit values are clamped to [-1, 1] (signed knobs) / [0, 1]."""
    rng = np.random.default_rng(_seed(seed))
    U = lambda r: float(rng.random()) * r                   # noqa: E731  frnd(r)
    rnd = lambda n: int(rng.integers(0, n + 1))             # noqa: E731  rnd(n): 0..n inclusive
    p = dict(_KDEF, wave=0)
    name = str(preset or "none").lower()
    name = _ALIAS.get(name, name)
    if name != "none":
        if name not in SFXR_PRESETS:
            raise ValueError(f"unknown sfxr preset {preset!r}; use one of {sorted(SFXR_PRESETS) + sorted(_ALIAS)}")
        SFXR_PRESETS[name](p, U, rnd)
    for _ in range(max(0, int(mutate))):                    # jsfxr mutate: each knob with p = 1/2, +-0.05
        for k in _MUTATED:
            if rnd(1):
                p[k] = float(np.clip(p[k] + U(0.1) - 0.05, -1.0, 1.0))
    if wave is not None:
        p["wave"] = _wave_index(wave)
        if p["wave"] == 1 and knobs.get("duty") is None:
            p["duty"] = 1.0                                 # a real saw (duty 0 would be a triangle, §A.2 note 3)
    for k, v in knobs.items():
        if v is not None:
            p[k] = float(np.clip(float(v), -1.0 if k in _SIGNED else 0.0, 1.0))
    return p


def _sfxr_decl():
    d = {
        "preset": ("none", "none|pickup|laser|explosion|powerup|hit|jump|blip|synth|tone|click|random (aliases coin, shoot, hurt, select): seeded jsfxr preset generator"),
        "seed": (0, "int or string: preset draw, mutation and the noise buffer"),
        "mutate": (0, "rounds of the sfxr mutate (each knob: 50% chance of +-0.05) applied after the preset"),
        "wave": (None, "square|saw|sine|noise; engine bfxr adds triangle|breaker|tan|whistle|bitnoise (default: the preset's, else square)"),
        "engine": ("sfxr", "sfxr = jsfxr-faithful | bfxr = also the extra waves, harmonics, crush, compression"),
        "level": (0.8, "peak level 0..1.4 of the result (raw sfxr gain is meaningless across seeds)"),
        "max_dur": (3.0, "s: longest sound, cut with a 20 ms fade"),
        "soften": (1.0, "0 = raw sfxr .. 1 = top end rounded (12 dB/oct lowpass at 4 kHz) so it is not piercing"),
        "hz": (None, "base pitch in Hz or as a note name; overrides `freq` (3.6..3500 Hz)"),
    }
    for k in KNOBS:
        d[k] = (None, f"{_KHELP[k]}; default {_KDEF[k]:g} or the preset's")
    d.update({
        "harmonics": (0, "bfxr: number of overtones 0..10 added to the wave"),
        "harmonics_falloff": (0.5, "bfxr: each overtone is (1 - falloff) times the previous, 0..1"),
        "crush": (0.0, "bfxr: sample-and-hold rate reduction 0..1"),
        "crush_sweep": (0.0, "bfxr: crush sweep over the sound -1..1"),
        "compression": (0.0, "bfxr: 0..1, lifts the quiet parts (|x|^(1/(1+4c)))"),
    })
    return d


@sfx("sfxr", "The sfxr engine (jsfxr-faithful, 8x oversampled): 22 knobs, seeded presets, mutate; engine bfxr adds waves, harmonics, crush, compression.",
     **_sfxr_decl())
def sfxr(sr=DEFAULT_SR, preset="none", seed=0, mutate=0, wave=None, engine="sfxr", level=0.8, max_dur=3.0, soften=1.0,
         hz=None, harmonics=0, harmonics_falloff=0.5, crush=0.0, crush_sweep=0.0, compression=0.0, **knobs):
    """sfxr by DrPetter as ported in jsfxr (`sfxr.js`), with optional Bfxr extensions.
    Source: g17-retro-sfx-generators.md §A (engine), §B.1 (presets), §F.1 (design)."""
    if engine not in ("sfxr", "bfxr"):
        raise ValueError(f"sfxr engine must be 'sfxr' or 'bfxr', not {engine!r}")
    if hz is not None:
        knobs["freq"] = hz_to_knob(to_hz(hz))
    p = sfxr_params(preset, seed, mutate, wave, **knobs)
    bf = engine == "bfxr"
    if p["wave"] > 3 and not bf:
        raise ValueError(f"wave {SFXR_WAVES[p['wave']]!r} needs engine 'bfxr'")
    if bf and p["wave"] == 1:
        p["duty"] = 1.0                                     # Bfxr's saw is a pure saw (no duty)
    P = np.array([p[k] for k in KNOBS], dtype=np.float64)
    cap = max(64, int(float(np.clip(max_dur, 0.01, 30.0)) * RSR))
    nh = int(np.clip(harmonics, 0, 10)) if bf else 0
    x = sfxr_core(p["wave"], P, 0.5, _seed(seed), cap, nh, float(np.clip(harmonics_falloff, 0.0, 1.0)))
    cut = x.shape[0] >= cap
    if bf and (crush > 0 or crush_sweep != 0):
        x = _crush(x, float(np.clip(crush, 0.0, 1.0)), float(np.clip(crush_sweep, -1.0, 1.0)))
    c = float(np.clip(compression, 0.0, 1.0)) if bf else 0.0
    if c > 0 and x.size:
        x = x / (np.max(np.abs(x)) + 1e-12)
        x = np.sign(x) * np.abs(x) ** (1.0 / (1.0 + 4.0 * c))
    return _finish(x, sr, level, soften, cut)


_SV_GAIN = 0.145     # level trim: K-weighted about -12 dB at vel 0.9 (measured in tests/test_retro.py)


@instrument("sfxr_voice", "Raw sfxr oscillator as a pitched voice: square/saw/sine + duty, vibrato, resonant low-pass (the harsher sibling of `chip`).",
            family="retro", span=("C2", "C6"),
            wave=("square", "square|saw|sine|noise|triangle|breaker|tan|whistle|bitnoise"),
            duty=(0.0, "sfxr duty knob 0..1: square 50% -> thin pulse; saw is always a pure saw"),
            duty_sweep=(0.0, "duty sweep -1..1 (PWM)"), attack=(0.03, "sfxr attack knob 0..1 (v^2 * 2.27 s)"),
            decay=(0.1, "sfxr decay knob 0..1 = release after the note (v^2 * 2.27 s)"), punch=(0.0, "accent at note start 0..1"),
            vib_depth=(0.0, "vibrato depth 0..1"), vib_speed=(0.5, "vibrato speed 0..1 (v^2 * 70 Hz)"),
            slide=(0.0, "pitch slide -1..1 (0 = in tune)"), arp_mod=(0.0, "pitch jump -1..1 (0.745 = +12 st)"),
            arp_speed=(0.6, "delay before the jump 0..1"), lpf=(1.0, "sfxr low-pass cutoff 0..1 (1 = off)"),
            resonance=(0.0, "low-pass resonance 0..1"), hpf=(0.0, "sfxr high-pass cutoff 0..1"),
            harmonics=(0, "overtones 0..10 (Bfxr)"), harmonics_falloff=(0.5, "overtone falloff 0..1"),
            bright=(0.5, "0 = dull .. 0.5 = default .. 1 = raw unfiltered"), seed=(0, "noise seed"))
def sfxr_voice(freq, dur, sr=DEFAULT_SR, vel=1.0, wave="square", duty=0.0, duty_sweep=0.0, attack=0.03, decay=0.1,
               punch=0.0, vib_depth=0.0, vib_speed=0.5, slide=0.0, arp_mod=0.0, arp_speed=0.6, lpf=1.0, resonance=0.0,
               hpf=0.0, harmonics=0, harmonics_falloff=0.5, bright=0.5, seed=0):
    """A note from the sfxr engine. The base period is set from `freq` (rounded to the engine's 1/8-sample
    period grid: at most 2.6 cents off at C6) and the sustain stage from `dur`.
    Source: g17-retro-sfx-generators.md §A.2, §F.3."""
    w = _wave_index(wave)
    p = sfxr_params(wave=w, duty=1.0 if w == 1 else duty, duty_sweep=duty_sweep, attack=attack, decay=max(decay, 0.03),
                    punch=punch, vib_depth=vib_depth, vib_speed=vib_speed, slide=slide, arp_mod=arp_mod,
                    arp_speed=arp_speed, lpf=lpf, resonance=resonance, hpf=hpf)
    p["freq"] = hz_to_knob(freq)
    hold = max(dur * RSR - math.floor(p["attack"] ** 2 * 1e5), 32.0)
    p["sustain"] = math.sqrt((hold + 0.5) / 1e5)            # unclamped: notes longer than 2.27 s still work
    P = np.array([p[k] for k in KNOBS], dtype=np.float64)
    x = sfxr_core(w, P, math.log(2.0), _seed(seed), int((dur + 3.0) * RSR), int(np.clip(harmonics, 0, 10)),
                  float(np.clip(harmonics_falloff, 0.0, 1.0)))
    if w == 0:
        x = x * 2.0                                         # sfxr's square is +-0.5
    x = _to_sr(x, RSR, sr)
    x = F.dc_block(_soften(x, freq, sr, harm=6, lo=2200, hi=3800, bright=bright), sr)
    return _fade(x, sr, 1.0, 2.0) * (_SV_GAIN * (0.5 + 0.5 * vel))


# ============================================================================ ZzFX
ZZ_SLOTS = ["volume", "randomness", "frequency", "attack", "sustain", "release", "shape", "shape_curve", "slide",
            "delta_slide", "pitch_jump", "pitch_jump_time", "repeat_time", "noise", "modulation", "bit_crush", "delay",
            "sustain_volume", "decay", "tremolo", "filter"]
ZZ_DEF = [1.0, 0.05, 220.0, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
_ZHELP = ["linear gain (the result is normalised to `level` anyway)", "pitch jitter +- fraction, drawn from `seed`", "Hz",
          "s", "s", "s", "0 sine, 1 triangle, 2 saw, 3 tan, 4 noise, 5 square (duty = shape_curve/2)",
          "waveform power (1 = plain; <1 fatter, >1 thinner)", "pitch slide (about kHz/s)", "slide of the slide",
          "Hz added at pitch_jump_time", "s", "s: resets pitch/slide and re-arms the jump; also the tremolo period",
          "phase-noise amount", "Hz of frequency modulation", "sample-and-hold period, x100 samples",
          "s: adds a delayed copy and lengthens the sound", "level of the sustain stage", "s: attack -> sustain_volume",
          "tremolo depth 0..1 (period = repeat_time)", "Hz: >0 high-pass, <0 low-pass (actual corner about 2x)"]


@nb.njit(cache=True)
def _zzfx_loop(P, sr, rnd01, cap, sin_i5):
    """ZzFX 1.3.2 `buildSamples` (g17 §C.2), slot for slot; P = float64[21], rnd01 = its one Math.random().
    Returns samples including the `volume` slot gain (not clipped). JS `%` is fmod (sign of the dividend).
    sin_i5[i] = sin(i**5), the phase-noise hash (only read when the noise slot is non-zero)."""
    PI2 = 2.0 * np.pi
    volume = P[0]
    frequency = P[2]
    attack = P[3]
    sustain = P[4]
    release = P[5]
    shape = P[6]
    shape_curve = P[7]
    slide = P[8]
    delta_slide = P[9]
    pitch_jump = P[10]
    pitch_jump_time = P[11]
    noise = P[13]
    modulation = P[14]
    delay = P[16]
    sustain_volume = P[17]
    decay = P[18]
    tremolo = P[19]
    filt = P[20]
    slide = slide * 500.0 * PI2 / sr / sr
    start_slide = slide
    frequency = frequency * (1.0 + P[1] * 2.0 * rnd01 - P[1]) * PI2 / sr
    start_frequency = frequency
    mod_offset = 0
    repeat = 0
    crush = 0
    jump = 1
    t = 0.0
    s = 0.0
    fs = -1.0 if filt < 0 else 1.0
    w = PI2 * abs(filt) * 2.0 / sr                       # RBJ biquad, Q = 2
    cw = np.cos(w)
    alpha = np.sin(w) / 2.0 / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cw / a0
    a2 = (1.0 - alpha) / a0
    b0 = (1.0 + fs * cw) / 2.0 / a0
    b1 = -(fs + cw) / a0
    b2 = b0
    x1 = 0.0
    x2 = 0.0
    y1 = 0.0
    y2 = 0.0
    attack = attack * sr
    if attack == 0.0:
        attack = 9.0                                     # minAttack: prevents a pop
    decay *= sr
    sustain *= sr
    release *= sr
    delay *= sr
    delta_slide *= 500.0 * PI2 / sr ** 3
    modulation *= PI2 / sr
    pitch_jump *= PI2 / sr
    pitch_jump_time *= sr
    repeat_time = int(P[12] * sr)
    length = min(int(attack + decay + sustain + release + delay), cap)
    b = np.zeros(length)                                 # raw (before the volume slot), so volume 0 is safe
    crush_n = int(P[15] * 100.0)
    for i in range(length):
        crush += 1
        if crush_n == 0 or crush % crush_n == 0:         # bit crush: recompute every N samples
            if shape == 0:
                s = np.sin(t)
            elif shape <= 1:
                s = 1.0 - 4.0 * abs(np.floor(t / PI2 + 0.5) - t / PI2)
            elif shape <= 2:
                s = 1.0 - np.fmod(np.fmod(2.0 * t / PI2, 2.0) + 2.0, 2.0)
            elif shape <= 3:
                s = max(min(np.tan(t), 1.0), -1.0)
            elif shape <= 4:
                s = np.sin(t ** 3)
            else:
                s = 1.0 if np.fmod(t / PI2, 1.0) < shape_curve / 2.0 else -1.0
            trem = 1.0 - tremolo + tremolo * np.sin(PI2 * i / repeat_time) if repeat_time != 0 else 1.0
            if shape <= 4:
                s = (-1.0 if s < 0 else 1.0) * abs(s) ** shape_curve
            if i < attack:
                env = i / attack
            elif i < attack + decay:
                env = 1.0 - ((i - attack) / decay) * (1.0 - sustain_volume)
            elif i < attack + decay + sustain:
                env = sustain_volume
            elif i < length - delay:
                env = (length - i - delay) / release * sustain_volume
            else:
                env = 0.0
            s = trem * s * env
            if delay != 0.0:
                tap = 0.0
                if not delay > i:
                    tap = (1.0 if i < length - delay else (length - i) / delay) * b[int(i - delay)] / 2.0
                s = s / 2.0 + tap
            if filt != 0.0:
                yn = b2 * x2 + b1 * x1 + b0 * s - a2 * y2 - a1 * y1
                x2 = x1
                x1 = s
                y2 = y1
                y1 = yn
                s = yn
        b[i] = s
        slide += delta_slide                             # delta slide is not reset by repeat
        frequency += slide
        f = frequency * np.cos(modulation * mod_offset)
        mod_offset += 1
        t += f + f * noise * sin_i5[i] if noise != 0.0 else f
        if jump != 0:
            jump += 1
            if jump > pitch_jump_time:
                frequency += pitch_jump
                start_frequency += pitch_jump            # so every repeat steps the pitch again
                jump = 0
        if repeat_time != 0:
            repeat += 1
            if repeat % repeat_time == 0:
                frequency = start_frequency
                slide = start_slide
                if jump == 0:
                    jump = 1
    return b * volume


_SIN_I5 = np.zeros(0)


def zzfx_core(P, sr, rnd01, cap):
    """`_zzfx_loop` with the sin(i**5) table. i**5 leaves float64's exact range at i = 1553; from there
    the value depends on how the platform's pow rounds (V8's differs from the C library's in about 1 of
    500 samples, and each difference re-rolls that sample's noise). The table uses exact integers rounded
    once: identical on every machine, sample-exact with `ZzFX.js` whenever the noise slot is 0 and for
    the first 1553 samples otherwise, the same noise statistically after that."""
    global _SIN_I5
    n = 0 if P[13] == 0 else min(int((max(P[3], 0) + max(P[18], 0) + max(P[4], 0) + max(P[5], 0) + max(P[16], 0)) * sr) + 16, cap)
    if n > _SIN_I5.shape[0]:
        _SIN_I5 = np.sin(np.array([float(i ** 5) for i in range(n)]))
    return _zzfx_loop(P, sr, rnd01, cap, _SIN_I5 if n else np.zeros(1))


def _zz_array(p=None, named=None, base=None) -> np.ndarray:
    """21 slots: defaults <- base <- `p` (None = hole) <- named slots; clamped where ZzFX would blow up."""
    a = list(base if base is not None else ZZ_DEF)
    for i, v in enumerate(list(p or [])[:21]):
        if v is not None:
            a[i] = float(v)
    for k, v in (named or {}).items():
        if v is not None:
            a[ZZ_SLOTS.index(k)] = float(v)
    a = np.array(a, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("zzfx parameters must be finite numbers")
    for i in (3, 4, 5, 11, 12, 15, 16, 18):              # times and the crush period are >= 0
        a[i] = max(a[i], 0.0)
    a[7] = max(a[7], 0.0)                                # a negative power of 0 is infinite
    a[20] = float(np.clip(a[20], -10500.0, 10500.0))     # filter: w = 4*pi*|f|/sr must stay below pi (stability)
    return a


def _zzfx_presets():
    """ZzFX designer presets (`index.html` AddPresetSound, g17 §B.5). R(a, b) = b + (a - b) * U."""
    def common(z, R):
        if R() < 0.5:
            z["filter"] = 99 + R() ** 2 * 900 if R() < 0.5 else R() ** 2 * 1e3 - 1500
        z["shape_curve"] = R() * 4

    def shape(R):
        return 5 if R() < 0.2 else int(R(2))

    def opt(R, p, f):                                    # "p% 0 else f()"
        return 0.0 if R() < p else f()

    def rand(z, R):
        C = lambda: R() if R() < 0.5 else 0.0            # noqa: E731
        S = lambda: 1 if C() else -1                     # noqa: E731
        for k in ("attack", "decay", "sustain", "release"):
            z[k] = R() ** 3 / 2
        length = z["attack"] + z["decay"] + z["sustain"] + z["release"]
        z["filter"] = 0 if C() else (99 + R() ** 2 * 900 if R() < 0.5 else R() ** 2 * 1e3 - 1500)
        z["frequency"] = 9 + R() ** 2 * 1e3
        z["shape"] = int(R() * 6)
        z["shape_curve"] = R() * 4
        z["slide"] = C() ** 3 * 99 * S()
        z["delta_slide"] = C() ** 3 * 99 * S()
        z["pitch_jump"] = C() ** 2 * 500 * S()
        z["pitch_jump_time"] = R() ** 2 * length
        z["repeat_time"] = C() * length / 4
        z["noise"] = C() ** 4
        z["modulation"] = R() * C() ** 2 * 500
        z["bit_crush"] = C() ** 4
        z["delay"] = C() ** 3 / 2
        z["sustain_volume"] = 1 - R() * 0.5
        z["tremolo"] = C() ** 2 * 0.5

    def plain(z):
        return not (z["tremolo"] or z["slide"] or z["delta_slide"])

    def pickup(z, R):
        common(z, R)
        z.update(frequency=R(200, 700), shape=shape(R), attack=R(.03), decay=R(.05), sustain=R(.1),
                 sustain_volume=R(.5, 1), release=R(.05, .3), noise=opt(R, .8, lambda: R(.5)),
                 slide=opt(R, .5, lambda: R(-1, 1) ** 3 * 10), delta_slide=opt(R, .5, lambda: R(-1, 1) ** 3 * 200),
                 repeat_time=opt(R, .5, lambda: R(.02, .1)), bit_crush=opt(R, .5, lambda: R(.1)),
                 delay=opt(R, .7, lambda: R(.1)), modulation=opt(R, .8, lambda: R() ** 2 * 50))
        if R() < 0.5 or plain(z):
            z.update(pitch_jump=R(99, 500), pitch_jump_time=R(.04, .1))

    def powerup(z, R):
        common(z, R)
        z.update(frequency=R(99, 700), shape=shape(R), attack=R(.1), decay=R(.1, .3), sustain=R(.1, .3),
                 sustain_volume=R(.5, 1), release=R(.05, .4), delay=opt(R, .8, lambda: R(.2)), repeat_time=R(.02, .2),
                 slide=opt(R, .5, lambda: R(-1, 1) ** 3 * 20), delta_slide=opt(R, .5, lambda: R(-1, 1) ** 3 * 400),
                 noise=opt(R, .8, lambda: R(.5)), bit_crush=opt(R, .5, lambda: R(.2)),
                 tremolo=opt(R, .5, lambda: R(.5)), modulation=opt(R, .8, lambda: R() ** 2 * 50))
        if R() < 0.5 or plain(z):
            z.update(pitch_jump=-R(50, 200) if R() < 0.5 else R(50, 500), pitch_jump_time=R(.05, .2))

    def jump(z, R):
        common(z, R)
        z.update(frequency=R(50, 500), shape=shape(R), attack=R(.05), decay=R(.05), sustain=R(.05),
                 sustain_volume=R(.5, 1), release=R(.05, .1), noise=opt(R, .5, lambda: R(1)),
                 slide=opt(R, .5, lambda: R() ** 2 * 50))
        z.update(delta_slide=0.0 if (R() < 0.5 and z["slide"]) else R(-50, 200), bit_crush=opt(R, .5, lambda: R(.1)),
                 delay=opt(R, .8, lambda: R(.05)))

    def shoot(z, R):
        common(z, R)
        z.update(frequency=R(50, 500), shape=shape(R), attack=R(.03), decay=R(.05, .1), sustain=R(.2),
                 sustain_volume=R(.5, 1), release=R(.05, .1), delay=opt(R, .5, lambda: R(.3)), slide=R(-1, 1) * 20,
                 delta_slide=R(-1, 1) * 50, noise=opt(R, .8, lambda: R(1)), modulation=opt(R, .8, lambda: R() ** 2 * 50),
                 bit_crush=opt(R, .5, lambda: R(.5)), tremolo=opt(R, .5, lambda: R(.3)))
        if R() < 0.4 or z["tremolo"]:
            z["repeat_time"] = R(.01, .2)

    def blip(z, R):
        rand(z, R)
        z.update(attack=R(.03), decay=R(.03), sustain=R(.04), release=R(.04))

    def hit(z, R):
        common(z, R)
        z.update(frequency=R(30, 500), shape=int(R(6)), attack=R(.03), decay=R(.1), sustain=R(.1),
                 sustain_volume=R(.4, 1), release=R(.2), delay=opt(R, .5, lambda: R(.2)),
                 slide=opt(R, .5, lambda: R(-1, 1) ** 3 * 10), delta_slide=opt(R, .5, lambda: R(-1, 1) ** 3 * 20),
                 noise=R(2), modulation=opt(R, .8, lambda: R() ** 2 * 50), bit_crush=R(.5),
                 tremolo=opt(R, .6, lambda: R(.5)))
        if R() < 0.5 or z["tremolo"]:
            z["repeat_time"] = R(.01, .1)
        z["filter"] = opt(R, .5, lambda: 99 + R() ** 2 * 2e3 if R() < 0.5 else R() ** 2 * 1e3 - 2500)

    def explosion(z, R):
        common(z, R)
        z.update(frequency=R(30, 99), shape=int(R(6)), attack=R(.1), decay=R(.05, .2), sustain=R(.3),
                 sustain_volume=R(.3, .5), release=R(.2, .6), slide=opt(R, .5, lambda: R(-9, 9)),
                 delta_slide=opt(R, .5, lambda: R(-9, 9)), delay=opt(R, .5, lambda: R(.5)), noise=R(2),
                 modulation=opt(R, .8, lambda: R() ** 2 * 99), bit_crush=R(1, .1), tremolo=opt(R, .5, lambda: R(.5)))
        if R() < 0.5 or z["tremolo"]:
            z["repeat_time"] = R(.05, .3)
        z["filter"] = opt(R, .5, lambda: 99 + R() ** 2 * 2e3 if R() < 0.5 else R() ** 2 * 2e3 - 3500)

    return {"random": rand, "pickup": pickup, "powerup": powerup, "jump": jump, "shoot": shoot, "blip": blip,
            "hit": hit, "explosion": explosion}


ZZFX_PRESETS = _zzfx_presets()


def zzfx_preset(name: str, rng) -> list:
    """Draw a designer preset -> 21-slot list, with the designer's post-processing (rounding, dead-slot cleanup)."""
    key = {"laser": "shoot", "coin": "pickup"}.get(str(name).lower(), str(name).lower())
    if key not in ZZFX_PRESETS:
        raise ValueError(f"unknown zzfx preset {name!r}; use one of {sorted(ZZFX_PRESETS)}")
    R = lambda a=1.0, b=0.0: b + (a - b) * float(rng.random())      # noqa: E731
    z = dict(zip(ZZ_SLOTS, ZZ_DEF))
    z["randomness"] = 0.0
    ZZFX_PRESETS[key](z, R)
    z["frequency"] = float(round(z["frequency"]))
    for k in ZZ_SLOTS[3:]:
        z[k] = round(float(z[k]), 2)
    if z["repeat_time"] > z["attack"] + z["sustain"] + z["delay"]:
        z["repeat_time"] = z["tremolo"] = 0.0
    if not (z["tremolo"] or z["pitch_jump"] or z["slide"] or z["delta_slide"]):
        z["repeat_time"] = 0.0
    if not z["pitch_jump"]:
        z["pitch_jump_time"] = 0.0
    if z["release"] == 0:
        z["release"] = round(R(.01), 3) + 0.001
    if z["shape"] == 5 and not 0.1 <= z["shape_curve"] <= 1.9:
        z["shape_curve"] = round(0.1 + R() * 1.8, 2)
    return [z[k] for k in ZZ_SLOTS]


def _zzfx_decl():
    d = {"p": (None, "the ZzFX array, up to 21 slots (null = default): [volume, randomness, frequency, attack, sustain, release, shape, shapeCurve, slide, deltaSlide, pitchJump, pitchJumpTime, repeatTime, noise, modulation, bitCrush, delay, sustainVolume, decay, tremolo, filter]"),
         "preset": (None, "random|pickup|powerup|jump|shoot|blip|hit|explosion: seeded ZzFX-designer preset"),
         "seed": (0, "int or string: preset draw and the `randomness` pitch jitter"),
         "level": (0.8, "peak level 0..1.4 of the result"), "max_dur": (5.0, "s: longest sound"),
         "soften": (1.0, "0 = raw .. 1 = top end rounded (12 dB/oct lowpass at 4 kHz)")}
    for k, dv, h in zip(ZZ_SLOTS, ZZ_DEF, _ZHELP):
        d[k] = (None, f"{h}; default {dv:g}")
    return d


@sfx("zzfx", "The ZzFX 1.3.2 sound function: paste a ZzFX array as `p`, or use named slots / a seeded designer preset.",
     **_zzfx_decl())
def zzfx(sr=DEFAULT_SR, p=None, preset=None, seed=0, level=0.8, max_dur=5.0, soften=1.0, **slots):
    """ZzFX by Frank Force (`ZzFX.js` buildSamples). Precedence: defaults <- preset <- `p` <- named slots.
    Source: g17-retro-sfx-generators.md §C.1-C.2, §B.5 (presets), §F.2."""
    rng = np.random.default_rng(_seed(seed))
    base = zzfx_preset(preset, rng) if preset not in (None, "", "none") else None
    a = _zz_array(p, slots, base)
    cap = max(64, int(float(np.clip(max_dur, 0.01, 30.0)) * RSR))
    x = zzfx_core(a, float(RSR), float(rng.random()), cap)
    return _finish(x, sr, level, soften, x.shape[0] >= cap)


_ZV_GAIN = 0.44      # level trim: about -12 dB at vel 0.9 for the default sine (ZzFX itself plays at 0.3)


@instrument("zzfx_voice", "A ZzFX sound played as a pitched voice: slot 2 (frequency) follows the note, randomness is off.",
            family="retro", span=("C2", "C6"),
            p=(None, "ZzFX array (up to 21 slots, null = default); frequency and randomness are overridden"),
            hold=(True, "true: the sustain slot follows the note length; false: the array's own envelope, cut at the note end (ZzFXM)"),
            shape=(None, "0 sine, 1 triangle, 2 saw, 3 tan, 4 noise, 5 square; default: from p, else 0"),
            shape_curve=(None, "waveform power / square duty x2; default: from p, else 1"),
            attack=(None, "s; default: from p, else 0"), decay=(None, "s; default: from p, else 0"),
            sustain_volume=(None, "level of the held part; default: from p, else 1"),
            release=(None, "s; default: from p, else 0.1"),
            bright=(0.5, "0 = dull .. 0.5 = default .. 1 = raw unfiltered"))
def zzfx_voice(freq, dur, sr=DEFAULT_SR, vel=1.0, p=None, hold=True, shape=None, shape_curve=None, attack=None,
               decay=None, sustain_volume=None, release=None, bright=0.5):
    """ZzFXM note semantics: the instrument array rendered at the note's frequency (the oscillator is a float
    phase accumulator, so it is exactly in tune unless the array slides or jumps).
    Source: g17-retro-sfx-generators.md §C.2, §D.1, §F.3."""
    a = _zz_array(p, dict(shape=shape, shape_curve=shape_curve, attack=attack, decay=decay,
                          sustain_volume=sustain_volume, release=release))
    a[1] = 0.0
    a[2] = freq
    if hold:
        a[4] = max(dur - a[3] - a[18], 0.0)
    gain = float(np.clip(a[0], 0.0, 3.0))                 # the volume slot keeps instruments' relative levels
    a[0] = 1.0
    x = zzfx_core(a, float(RSR), 0.5, int((dur + 4.0) * RSR))
    if not hold:
        x = x[:max(samples(dur, RSR), 32)]
    if x.size < 32:
        x = np.concatenate([x, np.zeros(32 - x.size)])
    x = _to_sr(x, RSR, sr)
    x = F.dc_block(_soften(x, freq, sr, harm=6, lo=2200, hi=3800, bright=bright), sr)
    return _fade(x, sr, 0.5, 2.0) * (_ZV_GAIN * gain * vel)


# ============================================================================ AY-3-8910 PSG
# 16 logarithmic DAC steps. The datasheet (g15 §2) says "16 logarithmic steps" but prints no table:
PSG_DAC = np.array([0.0] + [2.0 ** (-(15 - n) / 2.0) for n in range(1, 16)])   # UNSOURCED: 3 dB/step is the findings' design choice
PSG_CLOCK = 1_773_400.0   # UNSOURCED: the datasheet gives no clock; 1.7734 MHz is the ZX Spectrum 128 value (general knowledge)


def psg_tone_period(freq: float, clock: float = PSG_CLOCK) -> int:
    """12-bit tone period TP for a pitch: f = clock / (16 * TP) (datasheet, g15 §2)."""
    return int(min(max(round(clock / (16.0 * max(freq, 1e-3))), 1), 4095))


@nb.njit(cache=True)
def _psg_core(n, tp, npd, ep, shape, level, noise_mix, seed, dac):
    """One AY channel pair at fs = clock/8 (every divider is then a whole number of samples):
    tone flips every TP samples (period 16*TP clocks), the 17-bit LFSR steps every 2*NP samples
    (f = clock/(16*NP)), the envelope steps every 2*EP samples (16 steps per 256*EP clocks).
    Output is the unipolar DAC voltage: (1 - noise_mix) * tone channel + noise_mix * noise channel."""
    out = np.empty(n)
    tone = 1
    tc = 0
    nc = 0
    ec = 0
    es = 0
    lfsr = ((seed * 2 + 1) & 0x1FFFF) | 1
    cont = (shape >> 3) & 1
    att = (shape >> 2) & 1
    alt = (shape >> 1) & 1
    hld = shape & 1
    for i in range(n):
        if shape < 0:
            a = dac[level]
        else:
            cyc = es // 16
            pos = es % 16
            if cyc == 0:
                v = pos if att == 1 else 15 - pos
            elif cont == 0:
                v = 0
            elif hld == 1:
                v = (15 if att == 1 else 0) ^ (15 if alt == 1 else 0)
            else:
                up = att ^ (alt & (cyc & 1))
                v = pos if up == 1 else 15 - pos
            a = dac[v]
        out[i] = a * ((1.0 - noise_mix) * tone + noise_mix * (lfsr & 1))
        tc += 1
        if tc >= tp:
            tc = 0
            tone ^= 1
        nc += 1
        if nc >= 2 * npd:
            nc = 0
            # UNSOURCED: taps bit0 ^ bit3 of a 17-bit register (not in the datasheet; from memory of the chip)
            lfsr = (lfsr >> 1) | (((lfsr ^ (lfsr >> 3)) & 1) << 16)
        ec += 1
        if ec >= 2 * ep:
            ec = 0
            if es < 1 << 30:
                es += 1
    return out


_PSG_GAIN = 0.58      # level trim: about -12 dB at vel 0.9


@instrument("psg", "AY-3-8910 sound chip voice: square from a 12-bit clock divider, 16-step log DAC, LFSR noise, hardware envelope shapes.",
            family="retro", span=("C2", "C6"),
            clock=(PSG_CLOCK, "chip clock in Hz (1773400 ZX Spectrum, 2000000 Atari ST, 1789772 MSX): sets the pitch grid f = clock/(16*TP)"),
            level=(15, "fixed volume 0..15 on the 16-step log DAC (3 dB per step)"),
            noise=(0.0, "0 = tone channel only .. 1 = noise channel only (two chip channels mixed)"),
            noise_period=(8, "noise period NP 1..31: LFSR clock = clock/(16*NP) (1 = hiss, 31 = rumble)"),
            shape=(-1, "hardware envelope shape 0..15 (R15 bits Continue/Attack/Alternate/Hold); -1 = off, use `level`. 8 \\\\\\\\, 10 \\/\\/, 12 ////, 14 /\\/\\, 9 decay once, 13 attack and hold"),
            env_hz=(8.0, "envelope ramps per second (quantised to the 16-bit period EP = clock/(256*env_hz)); audio rates give buzz-bass"),
            bright=(0.5, "0 = dull .. 0.5 = default .. 1 = raw unfiltered"), seed=(0, "noise LFSR start state"))
def psg(freq, dur, sr=DEFAULT_SR, vel=1.0, clock=PSG_CLOCK, level=15, noise=0.0, noise_period=8, shape=-1, env_hz=8.0,
        bright=0.5, seed=0):
    """General Instrument AY-3-8910 voice. The pitch is quantised to the chip's divider, f = clock/(16*TP),
    so high notes are slightly off like on the real machine (under 10 cents up to about 1.2 kHz at 1.77 MHz).
    Source: g15-course-farnell-composition-misc.md §2 (datasheet). The findings derive one envelope ramp as
    4096*EP/clock; the datasheet's Fig. 3 labels the ramp itself 1/f_E = 256*EP/clock, which is used here."""
    clock = float(np.clip(clock, 2.0e5, 8.0e6))
    fs = clock / 8.0
    tp = psg_tone_period(freq, clock)
    ep = int(min(max(round(clock / (256.0 * max(float(env_hz), 1e-3))), 1), 65535))
    rel = 0.01
    x = _psg_core(int((dur + rel) * fs) + 8, tp, int(np.clip(noise_period, 1, 31)), ep, int(np.clip(shape, -1, 15)),
                  int(np.clip(level, 0, 15)), float(np.clip(noise, 0.0, 1.0)), _seed(seed) & 0xFFFF, PSG_DAC)
    x = _to_sr(x, fs, sr)[:samples(dur + rel, sr)]
    x = F.dc_block(_soften(x, freq, sr, harm=6, lo=2200, hi=3800, bright=bright), sr)   # the output is AC-coupled
    return _fade(x, sr, 1.0, rel * 1000.0) * (_PSG_GAIN * (0.5 + 0.5 * vel))


# ============================================================================ tracker `song` layer
_CELL = re.compile(r"^([A-Ga-g])([#b-]?)(-?\d)$")
_EMPTY = {"", ".", "..", "...", "---"}
_OFF = {"^^", "==", "off", "OFF"}
_HITS = {"x": 1.0, "X": 1.4, "o": 0.4}


def _resolve(name, defs: dict) -> dict:
    """Channel instrument -> {'drum': bool, 'name', 'params', 'gain', 'pan'}."""
    d = defs.get(name, name) if isinstance(name, str) else name
    if isinstance(d, str):
        d = {"drum": d} if d in D.REGISTRY and d not in I.REGISTRY else {"inst": d}
    r = {"drum": False, "params": dict(d.get("params") or {}), "gain": float(d.get("gain", 1.0)), "pan": float(d.get("pan", 0.0))}
    if "drum" in d:
        r.update(drum=True, name=d["drum"])
    elif "sfxr" in d:                                     # inline sfxr instrument: {"sfxr": {<sfxr_voice params>}}
        r.update(name="sfxr_voice", params=dict(d["sfxr"] or {}))
    elif "zzfx" in d:                                     # inline ZzFX instrument: {"zzfx": [array]} or {<zzfx_voice params>}
        z = d["zzfx"]
        r.update(name="zzfx_voice", params=dict(z) if isinstance(z, dict) else {"p": list(z or [])})
    elif "inst" in d:
        r["name"] = d["inst"]
    else:
        raise ValueError(f"song instrument {name!r} needs one of 'inst', 'drum', 'sfxr', 'zzfx'")
    if r["name"] not in (D.REGISTRY if r["drum"] else I.REGISTRY):
        raise ValueError(f"song instrument {name!r}: unknown {'drum' if r['drum'] else 'instrument'} {r['name']!r}")
    return r


def _cells(text) -> list[str]:
    return [c for c in (text.split() if isinstance(text, str) else [str(c) for c in text]) if c != "|"]


def _native_tracks(layer: dict):
    """Patterns of tracker cells -> ([events per channel], total_rows). An event is (row, freq, vel, res);
    freq None = drum hit, freq < 0 = note off."""
    defs = layer.get("instruments") or {}
    pats = layer.get("patterns") or {}
    if not pats:
        raise ValueError("song layer needs 'patterns': {name: {channel: 'C-4 . . ^^ ...'}}")
    order = layer.get("order") or list(pats)
    pans = layer.get("pan") or {}
    tracks: dict[str, list] = {}
    res: dict[str, dict] = {}
    row0 = 0
    for pname in order:
        if pname not in pats:
            raise ValueError(f"song order names unknown pattern {pname!r}; patterns: {list(pats)}")
        chans = {c: _cells(t) for c, t in pats[pname].items()}
        for c, cells in chans.items():
            if c not in res:
                res[c] = _resolve(c, defs)
                if c in pans:
                    res[c]["pan"] = float(pans[c])
            r, ev = res[c], tracks.setdefault(c, [])
            for i, cell in enumerate(cells):
                tok, _, v = cell.partition("@")
                if tok in _EMPTY:
                    continue
                vel = float(v) if v else 1.0
                if tok in _OFF:
                    ev.append((row0 + i, -1.0, 0.0, r))
                elif r["drum"]:
                    ev.append((row0 + i, None, vel * _HITS.get(tok, 1.0), r))
                elif tok in _HITS:
                    raise ValueError(f"song channel {c!r} is pitched: cell {cell!r} must be a note such as C-4")
                else:
                    m = _CELL.match(tok)
                    f = midi_to_freq(note_to_midi(m.group(1).upper() + m.group(2).replace("-", "") + m.group(3))) if m \
                        else parse_pitch_list(tok)[0]
                    ev.append((row0 + i, f, vel, r))
        row0 += max((len(c) for c in chans.values()), default=0)
    return list(tracks.values()), row0


def _zzfxm_tracks(data):
    """Raw ZzFXM song [instruments, patterns, sequence, bpm] (g17 §D.1) -> (tracks, total_rows, row_seconds).
    Note N.A: pitch = instrument frequency * 2^((N-12)/12), volume 1 - A; -1 = note off; 0 = keep ringing.
    Volume-only cells (0.A) are ignored; channel 0 sets a pattern's row count (longer channels are cut)."""
    insts, pats, seq = data[0], data[1], data[2]
    bpm = float(data[3]) if len(data) > 3 and data[3] else 125.0
    beat = int(RSR / bpm * 60.0) >> 2                    # samples per row at 44100 (a row is a 16th note)
    res = {}
    tracks = []
    for k in range(max(len(pats[s]) for s in seq)):
        ev = []
        row0 = 0
        for s in seq:
            pat = pats[s]
            rows = len(pat[0]) - 2
            ch = pat[k] if k < len(pat) else None
            for i in range(rows if ch else 0):
                note = float(ch[2 + i] or 0.0) if 2 + i < len(ch) else 0.0
                n = int(note)
                if n < 0:
                    ev.append((row0 + i, -1.0, 0.0, None))
                elif n > 0:
                    ii, pan = int(ch[0] or 0), float(ch[1] or 0.0)
                    if (ii, pan) not in res:
                        res[(ii, pan)] = {"drum": False, "name": "zzfx_voice", "gain": 1.0, "pan": pan,
                                          "params": {"p": list(insts[ii]), "hold": False}}
                    f0 = insts[ii][2] if len(insts[ii]) > 2 and insts[ii][2] is not None else 220.0
                    ev.append((row0 + i, float(f0) * 2.0 ** ((n - 12) / 12.0), 1.0 - (note - n), res[(ii, pan)]))
            row0 += rows
        tracks.append(ev)
    return tracks, sum(len(pats[s][0]) - 2 for s in seq), beat / RSR


@layer_type("song")
def song(layer: dict, sr: int, q: float) -> np.ndarray:
    """Tracker layer (g17 §F.4, §D). Patterns are rows of cells per channel:
    `C-4` / `C#4` note, `.` nothing (the note keeps ringing), `^^` note off, `@0.6` per-cell volume,
    and on drum channels `x` hit, `X` accent, `o` ghost. `rows` cells make one beat (default 4), `order`
    lists the patterns to play. Channels are genny instruments/drums rendered with `render_note` /
    `render_drum`. `"format": "zzfxm"` + `"data"` imports a raw ZzFXM song array instead.
    The result is exactly total_rows * row_length long (+ `tail` seconds of ring-out)."""
    if str(layer.get("format", "")).lower() == "zzfxm":
        tracks, total, row_s = _zzfxm_tracks(layer["data"])
    else:
        tracks, total = _native_tracks(layer)
        row_s = q / float(layer.get("rows", 4))
    parts = []
    cache: dict = {}
    for ev in tracks:
        for j, (row, f, vel, r) in enumerate(ev):
            if f is not None and f < 0:
                continue
            nxt = ev[j + 1][0] if j + 1 < len(ev) else total
            key = (id(r), f, vel, None if r["drum"] else nxt - row)
            if key not in cache:
                if r["drum"]:
                    y = D.render_drum(r["name"], sr, min(1.0, vel), **r["params"]) * vel
                else:
                    y = I.render_note(r["name"], f, (nxt - row) * row_s, sr, vel, **r["params"])
                y = y * r["gain"]
                cache[key] = np.stack([y * (1.0 - r["pan"]), y * (1.0 + r["pan"])], axis=1) if r["pan"] else y
            parts.append((cache[key], row * row_s))
    n = int(round(total * row_s * sr)) + int(round(float(layer.get("tail", 0.0)) * sr))
    y = mix(parts, sr) if parts else np.zeros(max(n, 1))
    out = np.zeros((n,) + y.shape[1:])
    m = min(n, y.shape[0])
    out[:m] = y[:m]
    if y.shape[0] > n:                                    # a note rings past the end: 4 ms fade instead of a step
        k = min(n, samples(0.004, sr))
        out[n - k:] *= np.linspace(1.0, 0.0, k)[:, None] if out.ndim == 2 else np.linspace(1.0, 0.0, k)
    return out
