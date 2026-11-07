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
SFXR_WAVES = ["square", "saw", "sine", "noise", "triangle", "breaker", "tan", "whistle", "bitnoise", "rasp", "fmsyn", "voice"]   # 4.. = Bfxr extras
_AKWF_HEX = {
    "granular_0044": (
        "80658312849e87d6893e8ccf8ddf91b1924e965e966f9ac19a139ed89ce1a35697e25b4857f655d6520550414c434ae14722463e430b42e840a841a340d24381"
        "44c749f04e2256925e666a8675b1846c914aa07eacb2ba3fc462cf58d753dfafe5dbebdcf0cef4abf906fac1ff0efde8fffffaabfe5af4bffbe7ea34fe5a8d72"
        "00d40ef800630616000101fe000000c500780223044606a10b550f4a17041da229253365434051fc65fb76998abb99daab01b64ec2e3c9a5d1c7d4bfd95cd9b2"
        "dbb1da57da5cd818d682d3fcd0eeced5ca31c969c278c4dfb8cac7c37147398a4df7448e4e374a2650d24f5e549754e9594b5af65ede61a2653a68d86c357054"
        "736d779f7a247dc47f3681a081ad828e814c80ca7fa5804b7fc080347fd680217fe980117ff7800380037ff8800b7ff180107fef80137fed80127fef80117ff1"
        "800e7ff4800b7ff780077ffa80057ffd80038000800080017fff80017ffe80027ffe80017fff80027fff80017fff800280008000800080008000800080008000"
        "80007fff7fff80007fff80007fff8000800180008000800080008000800080007fff80008000800080018000800080007fff800080008001800080007fff8001"
        "7fff8001800080017fff80007fff80008000800080007fff80027fff80027ffe80037ffd80047ffa80077ff580127f8d7e2f7d497d0b7d687eb27ff480007fff"),
    "fmsynth_0012": (
        "8127849d87ce8b0c8e2491359429970a99d39c839f1ba194a3f8a638a862aa69ac59ae24afd8b169b2e2b43bb578b699b7a4b890b968ba28bad5bb6cbbf1bc65"
        "bcc9bd20bd6abda7bdd9be03be24be3fbe53be60be69be6cbe6bbe65be58be4abe35be1bbdfbbdd3bda6bd71bd34bcecbc9cbc41bbdabb66bae6ba5ab9beb914"
        "b858b78fb6b6b5ccb4d1b3c7b2acb181b045aefaad9eac34aabca934a7a3a603a459a2a2a0e39f179d479b6f998f97aa95c193d391e48ff28dfe8c0a8a168824"
        "86348446825980727e917cb37adb7907773c757873ba720370576eb26d186b8769ff6882671165a8644e62ff61bc60845f5c5e405d325c325b405a5e598a58c7"
        "5813576f56db565955e75586553854fb54cf54b754b054bd54dc550f555555ae561c569c573057d7589559655a495b435c4f5d715ea55fee614c62bf644765e1"
        "679269556b2d6d1c6f1b7133735d759c77ef7a567cd27f5f820184b7877f8a588d43903d9347965a99799ca09fcda2f7a61fa93cac4caf43b21eb4ceb74db98c"
        "bb7cbd11be36bedabee5be43bcd4ba82b726b2a6accfa58c9c9b91e8851d763864ac520d4190335b27411d1d14c00e0808cc04ea023c00a60004003e013402d2"
        "04fd07a20aaf0e0f11b51592199e1dc6220826592ab32f11336e37c33c15405b449648c44ce550fa550058f95ce460b5647b68496bf66fac733a76de7a447dec"),
    "hvoice_0012": (
        "8136851d88e48ca7904a93ad96be99f29cfaa013a364a6ffaaf5aeceb239b5aab911bd4bc1acc4ecc836cba4cf3ed2d1d718dba0df45e1a2e46ae7beea79ec05"
        "ed23ee05ef08ef0fea6be1abda11d7c5d6ecd2acc8d1bbd9b157ad86ac4aa5fa99b689a0780969f2628e5cf6539a46723a3e319d2c62292f25f521711b6011f4"
        "06e5008300c9047a07e70aaf0ce30f4713b91bdb25f52e9a340c36d039823eac473b52335ec56a6d72e07947813b8b99950e9ba19fb6a254a4e5a8edae08b35d"
        "b80fbafebc88be82c183c305c204c04dbee9bd30bb29b918b736b60db4deb2c0b056aea0ad1eab1aa944a822a68ca40ca1799f4c9d9c9c889b6799d1987b9749"
        "95b9944693cd93c7930791678edd8aee860881457d757ac4787c757f71836d4b69da66a463545f865a09531a4bb5457940ad3d3b3a98386e369f354f346e3370"
        "32b631d1308d2f242ede3006328f36833b93416e47cd4e81552c5bb4620e68326e0b74007a1c80a287f490589918a142a877ae9bb413b90cbd82c144c3ffc5ba"
        "c6c8c79bc869c8ccc8b7c815c67cc3d5c00ebb63b644b123ac17a695a01f98c391388a3083e47dfb77b6711c6a88643b5e80595254814f904aa3463342463efb"
        "3c3039f8388437e5382638df39c83a703af83bc23d1b3ee340cd42ac44c647a54b704ff7549d58d45c485f48623f6559685b6b2f6d93701c72db76867a287db4"),
}
AKWF = np.array([[int(h[i:i + 4], 16) for i in range(0, 1024, 4)] for h in ("".join(v) for v in _AKWF_HEX.values())]) / 32768.0 - 1.0


_UNITS = {   # physical value -> knob, the inverse of jsfxr's slider and domain conversions
    "attack_s": ("attack", lambda v: math.sqrt(max(v, 0.0) * 44100.0 / 100000.0)),
    "sustain_s": ("sustain", lambda v: math.sqrt(max(v, 0.0) * 44100.0 / 100000.0)),
    "decay_s": ("decay", lambda v: math.sqrt(max(v, 0.0) * 44100.0 / 100000.0)),
    "slide_oct": ("slide", lambda v: float(np.cbrt((1.0 - math.exp(math.log(0.5) * v / 44100.0)) / 0.01))),
    "lpf_hz": ("lpf", lambda v: float(np.cbrt(max(v, 0.0) / (max(v, 0.0) + 8 * 44100.0) / 0.1))),
    "hpf_hz": ("hpf", lambda v: math.sqrt(max(v, 0.0) / (max(v, 0.0) + 8 * 44100.0) / 0.1)),
}


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
    if w >= 9:                                    # rasp, fmsyn, voice: 256-sample tables
        return AKWF[w - 9, int(x * 256.0) % 256]
    return (1.0 - float(lfsr & 1)) - 0.5          # 8: 1-bit LFSR noise


def bfxr_strict(P: np.ndarray) -> np.ndarray:
    """Bfxr's knob laws on a jsfxr knob vector: minimum frequency = limit^0.4 x start, sustain >= 0.01,
    attack + sustain + decay stretched to 0.18, decay length + 10 samples, flanger sweep^3 x 0.2."""
    P = np.array(P, dtype=np.float64)
    P[5] = P[5] ** 0.4 * P[4]
    P[1] = max(P[1], 0.01)
    tot = P[0] + P[1] + P[3]
    if tot < 0.18:
        P[[0, 1, 3]] *= 0.18 / tot
    P[3] = math.sqrt(P[3] ** 2 + 1e-4)
    P[16] = math.copysign(math.sqrt(0.2 * abs(P[16]) ** 3), P[16])
    return P


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
def sfxr_core(wave, P, sound_vol, seed, cap, nharm, hfall, j1=0.0, o1=1.0, j2=0.0, o2=1.0, jr=0.0):
    """The jsfxr synthesis loop (g17 §A.2: validated against `jsfxr/sfxr.js`), 8x oversampled, at 44100 Hz.

    P = float64[22] in `KNOBS` order; returns unclipped samples with gain exp(sound_vol)-1.
    With wave 0..3 and nharm = 0 this is sample-exact jsfxr (envelope stages of length 0 give 0, not NaN).
    Extras from Bfxr (§A.5): waves 4..8 and `nharm` overtones, each `1 - hfall` times the previous;
    the overtone sum is normalised (jfxr) instead of clipped (Bfxr). j1, j2 are Bfxr's two pitch jumps, undone and rearmed
    every W samples (Bfxr_DSP.js:141-177, 291-325); onset 1 = never."""
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
    jm1 = 1.0 - 0.9 * j1 * j1 if j1 > 0 else 1.0 + 10.0 * j1 * j1
    jm2 = 1.0 - 0.9 * j2 * j2 if j2 > 0 else 1.0 + 10.0 * j2 * j2
    jw = np.floor((env_len[0] + env_len[1] + env_len[2]) * (1.0 - jr) + RSR / 50.0 * jr + 32.0)
    jt1 = 0.0 if o1 >= 1.0 else np.floor(o1 * jw + 32.0)
    jt2 = 0.0 if o2 >= 1.0 else np.floor(o2 * jw + 32.0)
    jn = 0
    hit1 = False
    hit2 = False
    jumps = (j1 != 0.0 and jt1 != 0.0) or (j2 != 0.0 and jt2 != 0.0)
    while n < cap:
        if jumps:
            if jt1 != 0.0 and not hit1 and jn >= jt1:
                period *= jm1
                hit1 = True
            if jt2 != 0.0 and not hit2 and jn >= jt2:
                period *= jm2
                hit2 = True
            jn += 1
            if jn >= jw:
                jn = 0
                if hit1:
                    period /= jm1
                    hit1 = False
                if hit2:
                    period /= jm2
                    hit2 = False
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


def jump_knob(ratio: float) -> float:
    """Bfxr jump knob for a pitch ratio (tools/match/notes.py:34-46 of Bfxr2)."""
    r = float(ratio)
    return math.sqrt((1.0 - 1.0 / r) / 0.9) if r > 1.0 else -math.sqrt((1.0 / r - 1.0) / 10.0)


_BFXR_POW = {"attack": 4, "sustain": 2, "vib_depth": 3, "duty_sweep": 3, "flange_offset": 3, "lpf": 3, "hpf": 5, "dslide": 7}


def _bfxr_preset(p, name, U, rnd):
    """What Bfxr's generators do differently from jsfxr's."""
    W = lambda *a: SFXR_WAVES.index(a[rnd(len(a) - 1)])     # noqa: E731
    if name == "laser":
        s = 0.1 + U(0.5) if U(1.0) < 0.33 else 0.5 + U(0.5)
        p["freq"] = s
        if s < 0.15:
            p["freq_limit"], p["slide"] = 0.0, -0.1 - U(0.1)
    elif name == "explosion":
        p["wave"] = W("noise", "bitnoise")
    elif name == "powerup":
        p["freq"] = 0.2 + U(0.3)
    elif name == "hit":
        p["wave"] = W("noise", "bitnoise", "saw", "square", "voice")
    elif name == "jump":
        p["wave"] = W("square", "saw", "fmsyn")
    elif name == "blip":
        p["wave"] = W("square", "saw", "fmsyn", "whistle")
    elif name == "pickup":
        cn = 1 + rnd(6)
        p["jump1"], p["onset1"] = jump_knob((cn + 2 + rnd(6)) / cn), 0.2 + U(0.3)   # jump is a period ratio cnum/cden
    elif name == "random":
        p.update(_KDEF)
        p["wave"] = rnd(8)
        for k, e in _BFXR_POW.items():
            v = U(1.0) ** e
            p[k] = 1.0 - v if k == "lpf" else (v if k not in _SIGNED or rnd(1) else -v)
        p["punch"] = (U(1.0) * U(1.0)) ** 2
        p["freq"] = float(np.clip(0.3 + (2.0 * U(1.0) - 1.0) ** 3 * 0.3, 0.0, 0.6))
        p["decay"] = 0.1 + U(0.5)


def _wave_index(wave) -> int:
    if isinstance(wave, str):
        w = wave.lower()
        if w not in SFXR_WAVES:
            raise ValueError(f"unknown sfxr wave {wave!r}; use one of {SFXR_WAVES}")
        return SFXR_WAVES.index(w)
    return int(wave)


def sfxr_params(preset="none", seed=0, mutate=0, wave=None, presets="jsfxr", **knobs) -> dict:
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
        if presets == "bfxr":
            _bfxr_preset(p, name, U, rnd)
    for _ in range(max(0, int(mutate))):                    # jsfxr mutate: each knob with p = 1/2, +-0.05
        for k in _MUTATED:
            if rnd(1):
                p[k] = float(np.clip(p[k] + U(0.1) - 0.05, -1.0, 1.0))
        if presets == "bfxr" and U(1.0) < 0.1:              # Bfxr also changes the wave, p = .1
            p["wave"] = rnd(8)
    if wave is not None:
        p["wave"] = _wave_index(wave)
        if p["wave"] == 1 and knobs.get("duty") is None:
            p["duty"] = 1.0                                 # a real saw (duty 0 would be a triangle, §A.2 note 3)
    for k, v in knobs.items():
        if v is not None:
            p[k] = float(np.clip(float(v), -1.0 if k in _SIGNED else 0.0, 1.0))
    return p


_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def sfxr_link(text: str) -> dict:
    """Knob dict of a jsfxr share link (or just the part after '#'): base58 of the wave byte, then 22 little-endian floats in KNOBS order."""
    code = str(text).rsplit("#", 1)[-1].strip()
    v = 0
    for c in code:
        if c not in _B58:
            raise ValueError(f"not a jsfxr base58 link: bad character {c!r}")
        v = v * 58 + _B58.index(c)
    raw = b"\0" * (len(code) - len(code.lstrip("1"))) + v.to_bytes((v.bit_length() + 7) // 8, "big")
    if len(raw) < 89:
        raise ValueError(f"jsfxr link too short: {len(raw)} bytes, 89 needed")
    vals = np.frombuffer(raw, "<f4", 22, 1)
    if not np.isfinite(vals).all():
        raise ValueError("jsfxr link holds non-finite values")
    return {"wave": min(int(raw[0]), len(SFXR_WAVES) - 1)} | {k: float(x) for k, x in zip(KNOBS, vals)}


def _sfxr_decl():
    d = {
        "link": (None, "a jsfxr share link or its base58 code: replaces preset and knobs' defaults; explicit knobs still apply"),
        "presets": ("jsfxr", "jsfxr | bfxr: what Bfxr's generators do differently (random, mutate on the wave too)"),
        "jump1": (None, "bfxr: first pitch jump -1..1 (retro.jump_knob(ratio) gives it from a ratio)"), "onset1": (None, "bfxr: when in the window it jumps, 0..1; 1 = never"),
        "jump2": (0.0, "bfxr: second jump"), "onset2": (1.0, "bfxr: second onset"),
        "jump_repeat": (0.0, "bfxr: 0..1 jumps undone and rearmed every (lerp of the envelope length and 20 ms): the only arpeggio in an sfxr sound"),
        "preset": ("none", "none|pickup|laser|explosion|powerup|hit|jump|blip|synth|tone|click|random (aliases coin, shoot, hurt, select): seeded jsfxr preset generator"),
        "seed": (0, "int or string: preset draw, mutation and the noise buffer"),
        "mutate": (0, "rounds of the sfxr mutate (each knob: 50% chance of +-0.05) applied after the preset"),
        "wave": (None, "square|saw|sine|noise; engine bfxr adds triangle|breaker|tan|whistle|bitnoise|rasp|fmsyn|voice (default: the preset's, else square)"),
        "engine": ("sfxr", "sfxr = jsfxr-faithful | bfxr = also the extra waves, harmonics, crush, compression"),
        "level": (0.8, "peak level 0..1.4 of the result (raw sfxr gain is meaningless across seeds)"),
        "max_dur": (3.0, "s: longest sound, cut with a 20 ms fade"),
        "soften": (1.0, "0 = raw sfxr .. 1 = top end rounded (12 dB/oct lowpass at 4 kHz) so it is not piercing"),
        "hz": (None, "base pitch in Hz or as a note name; overrides `freq` (3.6..3500 Hz)"),
        "attack_s": (None, "attack in s; overrides `attack` (0..2.27)"), "sustain_s": (None, "sustain in s; overrides `sustain`"),
        "decay_s": (None, "decay in s; overrides `decay`"), "slide_oct": (None, "pitch slide in octaves per second (+ up); overrides `slide`"),
        "lpf_hz": (None, "low-pass in the Hz jsfxr displays (not the physical cutoff: a resonant 2-pole that peaks near 56 kHz x sqrt(hz / (hz + 352800))); overrides `lpf`"),
        "hpf_hz": (None, "high-pass in the Hz jsfxr displays (not the physical cutoff: one pole, about this / 2 pi); overrides `hpf`"),
    }
    for k in KNOBS:
        d[k] = (None, f"{_KHELP[k]}; default {_KDEF[k]:g} or the preset's")
    d.update({
        "harmonics": (0, "bfxr: number of overtones 0..10 added to the wave"),
        "harmonics_falloff": (0.5, "bfxr: each overtone is (1 - falloff) times the previous, 0..1"),
        "crush": (0.0, "bfxr: sample-and-hold rate reduction 0..1"),
        "crush_sweep": (0.0, "bfxr: crush sweep over the sound -1..1"),
        "compression": (0.0, "bfxr: 0..1, lifts the quiet parts (|x|^(1/(1+4c)))"),
        "strict": (False, "bfxr: also Bfxr's own knob laws (relative minimum frequency, sustain >= 0.01, 0.18 minimum length, +10 decay samples, cubic flanger sweep)"),
    })
    return d


@sfx("sfxr", "The sfxr engine (jsfxr-faithful, 8x oversampled): 22 knobs, seeded presets, mutate; engine bfxr adds waves, harmonics, crush, compression.",
     **_sfxr_decl())
def sfxr(sr=DEFAULT_SR, preset="none", seed=0, mutate=0, wave=None, engine="sfxr", level=0.8, max_dur=3.0, soften=1.0,
         hz=None, harmonics=0, harmonics_falloff=0.5, crush=0.0, crush_sweep=0.0, compression=0.0, link=None, presets="jsfxr",
         jump1=None, onset1=None, jump2=0.0, onset2=1.0, jump_repeat=0.0, strict=False, **knobs):
    """sfxr by DrPetter as ported in jsfxr (`sfxr.js`), with optional Bfxr extensions.
    Source: g17-retro-sfx-generators.md §A (engine), §B.1 (presets), §F.1 (design)."""
    if engine not in ("sfxr", "bfxr"):
        raise ValueError(f"sfxr engine must be 'sfxr' or 'bfxr', not {engine!r}")
    if hz is not None:
        knobs["freq"] = hz_to_knob(to_hz(hz))
    for k, (knob, conv) in _UNITS.items():
        v = knobs.pop(k, None)
        if v is not None:
            knobs[knob] = conv(float(v))
    if presets not in ("jsfxr", "bfxr"):
        raise ValueError(f"sfxr presets must be 'jsfxr' or 'bfxr', not {presets!r}")
    p = sfxr_params(preset, seed, mutate, wave, presets, **knobs)
    if link:
        p |= sfxr_link(link) | {k: float(v) for k, v in knobs.items() if v is not None}
        engine = "bfxr" if p["wave"] > 3 else engine
    bf = engine == "bfxr"
    j1 = p.get("jump1", 0.0) if jump1 is None else float(jump1)
    o1 = p.get("onset1", 1.0) if onset1 is None else float(onset1)
    if not bf:
        j1, jump2 = 0.0, 0.0
    if p["wave"] > 3 and not bf:
        raise ValueError(f"wave {SFXR_WAVES[p['wave']]!r} needs engine 'bfxr'")
    if bf and p["wave"] == 1:
        p["duty"] = 1.0                                     # Bfxr's saw is a pure saw (no duty)
    P = np.array([p[k] for k in KNOBS], dtype=np.float64)
    if bf and strict:
        P = bfxr_strict(P)
    cap = max(64, int(float(np.clip(max_dur, 0.01, 30.0)) * RSR))
    nh = int(np.clip(harmonics, 0, 10)) if bf else 0
    x = sfxr_core(p["wave"], P, 0.5, _seed(seed), cap, nh, float(np.clip(harmonics_falloff, 0.0, 1.0)), float(np.clip(j1, -1.0, 1.0)), float(np.clip(o1, 0.0, 1.0)),
                  float(np.clip(jump2, -1.0, 1.0)), float(np.clip(onset2, 0.0, 1.0)), float(np.clip(jump_repeat, 0.0, 1.0)))
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


@nb.njit(cache=True)
def _zzfxg_loop(P, sr, rnd01, cap):
    """The generator embedded in the ZzFXM player (`zzfxG`, 20 slots, no filter), statement for statement;
    returned without the global zzfxV factor, like `_zzfx_loop`."""
    b = 2.0 * np.pi
    q, k, c, e, t, u, r, F = P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7]
    v, z, w, A, l, B, x, G = P[8], P[9], P[10], P[11], P[12], P[13], P[14], P[15]
    d, y, m, C = P[16], P[17], P[18], P[19]
    v *= 500.0 * b / sr ** 2
    H = v
    I = (1.0 if x > 0 else -1.0) * b / 4.0
    c *= (1.0 + 2.0 * k * rnd01 - k) * b / sr
    D = c
    e = 99.0 + sr * e
    m *= sr
    t *= sr
    u *= sr
    d *= sr
    z *= 500.0 * b / sr ** 3
    x *= b / sr
    w *= b / sr
    A *= sr
    li = int(sr * l)
    hf = int(e + m + t + u + d)
    h = min(hf, cap)
    crush = int(100.0 * G)
    Z = np.zeros(h)
    g = 0.0
    E = 0.0
    n = 1
    J = 0
    K = 0
    f = 0.0
    for a in range(h):
        K += 1
        if crush == 0 or K % crush == 0:
            if r == 0:
                f = np.sin(g)
            elif r <= 1:
                f = 1.0 - 4.0 * abs(np.floor(g / b + 0.5) - g / b)
            elif r <= 2:
                f = 1.0 - np.fmod(np.fmod(2.0 * g / b, 2.0) + 2.0, 2.0)
            elif r <= 3:
                f = max(min(np.tan(g), 1.0), -1.0)
            else:
                f = np.sin(np.fmod(g, b) ** 3)
            if a < e:
                env = a / e
            elif a < e + m:
                env = 1.0 - (a - e) / m * (1.0 - y)
            elif a < e + m + t:
                env = y
            elif a < hf - d:
                env = (hf - a - d) / u * y
            else:
                env = 0.0
            f = (1.0 - C + C * np.sin(2.0 * np.pi * a / li) if li else 1.0) * (1.0 if f > 0 else -1.0) * abs(f) ** F * q * env
            if d != 0.0:
                f = f / 2.0 + (0.0 if d > a else (1.0 if a < hf - d else (hf - a) / d) * Z[int(a - d)] / 2.0)
        Z[a] = f
        v += z
        c += v
        p = c * np.sin(E * x - I)
        g += p - p * B * (1.0 - np.fmod(1e9 * (np.sin(float(a)) + 1.0), 2.0))
        E += p - p * B * (1.0 - np.fmod(1e9 * (np.sin(float(a)) ** 2 + 1.0), 2.0))
        if n:
            n += 1
            if n > A:
                c += w
                D += w
                n = 0
        if li:
            J += 1
            if J % li == 0:
                c = D
                v = H
                n = n or 1
    return Z


_SIN_I5 = np.zeros(0)


def zzfx_core(P, sr, rnd01, cap, gen=False):
    """`_zzfx_loop` with the sin(i**5) table. i**5 leaves float64's exact range at i = 1553; from there
    the value depends on how the platform's pow rounds (V8's differs from the C library's in about 1 of
    500 samples, and each difference re-rolls that sample's noise). The table uses exact integers rounded
    once: identical on every machine, sample-exact with `ZzFX.js` whenever the noise slot is 0 and for
    the first 1553 samples otherwise, the same noise statistically after that."""
    global _SIN_I5
    if gen:
        return _zzfxg_loop(P, sr, rnd01, cap)
    n = 0 if P[13] == 0 else min(int((max(P[3], 0) + max(P[18], 0) + max(P[4], 0) + max(P[5], 0) + max(P[16], 0)) * sr) + 16, cap)
    if n > _SIN_I5.shape[0]:
        _SIN_I5 = np.sin(np.array([float(i ** 5) for i in range(n)]))
    return _zzfx_loop(P, sr, rnd01, cap, _SIN_I5 if n else np.zeros(1))


ZZFXM_INSTRUMENTS = {
    "korgbass": [1, 0, 22, None, 0.07, 0.07, 2, 0, None, None, 0.5, 0.01],
    "bassdrum": [None, 0, 84, None, None, 0.1, None, 0.7, None, None, None, 0.5, None, 6.7, 1, 0.05],
    "hihat1": [2, 0, 4e3, None, None, 0.03, 2, 1.25, None, None, None, None, 0.02, 6.8, -0.3, None, 0.5],
    "hihat2": [2, 0, 800, None, None, 0.075, 2, 2, None, None, 1e3, -0.01, 0.02, 4.8, -0.3, None, 0.065],
    "hihat4": [2, 0, 2600, 0.01, None, 0.12, 3, 2, None, None, 800, 0.02, None, 4.8, None, 0.01, 0.1],
    "m1-hhato": [0.8, 0, 2100, None, None, 0.1, 3, 3, None, None, -400, None, None, 2],
    "nice": [1, 0, 209, None, 0.02, 0.25, 3],
    "house-snare": [1, 0, 655, 0, None, 0.09, 3, 1.65, None, None, None, None, 0.02, 3.8, -0.1, None, 0.2],
    "smba-wish-l": [2, 0, 426, 0.01, 0.2, 0.48, None, 44, None, None, 200, None, None, 0.1],
    "smba-wish-s": [2, 0, 426, 0, 0.02, 0.2, None, 44, None, None, 200, None, None, 0.1],
    "pizza": [1.5, 0, 155, 0.01, 0, 0.2, 2, 0.5, 0, None, -270, 0.2, 2, 0, 0.1, None, 0.15],
    "korgfilter": [1, 0, 245, 0, 0, 0.48, 2, None, 0, 0, 0, None, None, None, 0, 0.02, 0.01],
    "funbass": [0.9, 0, 38, 0.01, 0.05, 0.25, 2, 2, None, None, -8],
    "popsnare2": [1, 0, 219, None, None, None, None, 1.1, None, -0.1, -50, -0.05, -0.01, 1],
    "claps1": [None, 0, 220, None, None, None, 3, 0, None, None, 320, None, None, 4],
    "hallbrass": [1, 0, 232, 0.01, 0.09, 0.15, 2, None, None, None, None, None, 154.87, None, None, None, 0.26],
    "synbrass": [0.9, 0, 143, None, None, 0.35, 3],
    "digdug": [1, 0, 23, None, None, 0.2, 3, 5, None, None, 0],
    "ringpiano": [0.75, 0, 196, 0, 0.08, 0.18, 3],
    "muteclav": [1, 0, 192, None, 0.04, 0.2, 2, 2, -0.03, -0.04, -60, 0.02],
    "polysynth": [0.75, 0, 25, 0.002, 0.02, 0.08, 3, None, None, None, None, None, None, None, None, 0.1, 0.01],
    "bdrum1": [3, 0, 43, 0, None, 0.25, None, None, None, None, None, None, None, 2],
    "strings7": [1, 0, 137, 0.02, 0.2, 0.55, 2, 0.41, None, None, 0.1],
    "perco": [1.5, 0, 216, None, None, 0.45, 1, 4, None, None, 50],
    "shaker": [0.3, 0, 900, 0.02, None, 0.07, 4, 0, None, None, None, None, None, 4],
    "d.superbass4": [1.8, 0, 72, None, None, 0.2, None, 4, -2, 6, 50, 0.15, None, 6],
    "b.mono": [1.2, 0, 23, None, None, 0.2, 3, 4, None, None, 3, 0.9, 0.05],
    "i.flute2": [1.5, 0, 740, 0, None, 0.15, 2, 0.2, -0.1, -0.15, 9, 0.02, None, 0.1, 0.12, 0, 0.06],
    "64tom1": [None, 0, 220, None, None, 0.2, None, 0.34, -2, None, None, None, None, None, None, 0.08],
    "plop": [0.75, 0, 386, None, None, 0.25, 2],
    "flybass2": [1.75, 0, 64, 0.02, 0.03, 0.3, 2, 14.2, -0.1, None, None, None, None, 1.1],
    "monobass": [1, 0, 52, None, None, 0.2, 2, None, None, None, 0.5, 0.01],
    "peck": [3, 0, 238, 0.01, None, 0.4, 2, 2, None, None, -0.5, None, 0.1, None, None, None, 0.01, None, None, 0.2],
    "pulse": [2, 0, 26, 0.02, None, None, 2, 2, None, None, None, None, -0.03, None, None, None, None, None, None, 0.07],
}


def _zz_array(p=None, named=None, base=None) -> np.ndarray:
    """21 slots: defaults <- base <- `p` (None = hole) <- named slots; clamped where ZzFX would blow up."""
    a = list(base if base is not None else ZZ_DEF)
    if isinstance(p, str):
        if p not in ZZFXM_INSTRUMENTS:
            raise ValueError(f"unknown ZzFXM instrument {p!r}; use one of {sorted(ZZFXM_INSTRUMENTS)}")
        p = ZZFXM_INSTRUMENTS[p]
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


_ZZ_STEP = {"frequency": 1.0, "shape_curve": 0.1, "attack": 0.01, "decay": 0.01, "sustain": 0.01, "release": 0.01, "sustain_volume": 0.1, "slide": 0.1,
            "delta_slide": 0.1, "pitch_jump": 50.0, "pitch_jump_time": 0.01, "repeat_time": 0.01, "tremolo": 0.01, "noise": 0.1, "bit_crush": 0.1,
            "delay": 0.01, "modulation": 1.0, "filter": 1.0}


def zz_mutate(a, rng, strength=10):
    """The ZzFX designer's mutate (index.html:373-395, 1102-1175): six draws among the settings. Each moves by r steps, random sign, rounded to
    .01; r is 0..1 from its global button, 1..10 from a setting's own; `strength` picks r up to it. A zero filter is left alone, going below 0 jumps to -(1000..2000)."""
    a = np.array(a, float)
    names = list(_ZZ_STEP) + ["shape"]
    for k in rng.choice(names, 6):
        i = ZZ_SLOTS.index(k)
        if k == "shape":
            a[i] = float(rng.integers(0, 6))
        elif k != "filter" or a[i] != 0.0:
            v = round(a[i] + float(rng.choice((-1, 1))) * int(rng.integers(0, int(strength) + 1)) * _ZZ_STEP[k], 2)
            if k == "filter" and a[i] >= 0.0 > v:
                v = -float(rng.uniform(1000.0, 2000.0))
            a[i] = v if k in ("slide", "delta_slide", "pitch_jump", "filter", "noise", "modulation", "frequency") else max(v, 0.0)
    a[ZZ_SLOTS.index("frequency")] = max(a[ZZ_SLOTS.index("frequency")], 1.0)
    if a[ZZ_SLOTS.index("shape")] == 5.0:                   # the square's duty
        a[ZZ_SLOTS.index("shape_curve")] = float(np.clip(a[ZZ_SLOTS.index("shape_curve")], 0.1, 1.9))
    return a


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
def zzfx(sr=DEFAULT_SR, p=None, preset=None, seed=0, level=0.8, max_dur=5.0, soften=1.0, mutate=0, **slots):
    """ZzFX by Frank Force (`ZzFX.js` buildSamples). Precedence: defaults <- preset <- `p` <- named slots.
    Source: g17-retro-sfx-generators.md §C.1-C.2, §B.5 (presets), §F.2."""
    rng = np.random.default_rng(_seed(seed))
    base = zzfx_preset(preset, rng) if preset not in (None, "", "none") else None
    a = _zz_array(p, slots, base)
    for _ in range(max(0, int(mutate))):
        a = zz_mutate(a, rng)
    cap = max(64, int(float(np.clip(max_dur, 0.01, 30.0)) * RSR))
    x = zzfx_core(a, float(RSR), float(rng.random()), cap)
    return _finish(x, sr, level, soften, x.shape[0] >= cap)


_ZV_GAIN = 0.44      # level trim: about -12 dB at vel 0.9 for the default sine (ZzFX itself plays at 0.3)


@instrument("zzfx_voice", "A ZzFX sound played as a pitched voice: slot 2 (frequency) follows the note, randomness is off.",
            family="retro", span=("C2", "C6"),
            p=(None, "ZzFX array (up to 21 slots, null = default) or a ZzFXM instrument name (retro.ZZFXM_INSTRUMENTS: korgbass, house-snare, strings7...); frequency and randomness are overridden"),
            hold=(True, "true: the sustain slot follows the note length; false: the array's own envelope, cut at the note end (ZzFXM)"),
            shape=(None, "0 sine, 1 triangle, 2 saw, 3 tan, 4 noise, 5 square; default: from p, else 0"),
            shape_curve=(None, "waveform power / square duty x2; default: from p, else 1"),
            attack=(None, "s; default: from p, else 0"), decay=(None, "s; default: from p, else 0"),
            sustain_volume=(None, "level of the held part; default: from p, else 1"),
            release=(None, "s; default: from p, else 0.1"),
            bright=(0.5, "0 = dull .. 0.5 = default .. 1 = raw unfiltered"),
            engine=("zzfx", "zzfx (ZzFX 1.3.2) | zzfxg (the generator embedded in the ZzFXM player: 99-sample minimum attack, sine modulation, no filter slot)"))
def zzfx_voice(freq, dur, sr=DEFAULT_SR, vel=1.0, p=None, hold=True, shape=None, shape_curve=None, attack=None,
               decay=None, sustain_volume=None, release=None, bright=0.5, engine="zzfx"):
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
    x = zzfx_core(a, float(RSR), 0.5, int((dur + 4.0) * RSR), engine == "zzfxg")
    if not hold:
        x = x[:max(samples(dur, RSR), 32)]
    if x.size < 32:
        x = np.concatenate([x, np.zeros(32 - x.size)])
    x = _to_sr(x, RSR, sr)
    x = F.dc_block(_soften(x, freq, sr, harm=6, lo=2200, hi=3800, bright=bright), sr)
    return _fade(x, sr, 0.5, 2.0) * (_ZV_GAIN * gain * vel)


# ============================================================================ AY-3-8910 PSG
# 16 logarithmic DAC steps, 3 dB (x 0.707) per step: GI AY-3-8910 data manual, Fig. 3 "D/A converter output"
# (normalized 1 V at 15, ~0.5 at 13, ~0.25 at 11, ~0.12 at 9; map.grauw.nl/resources/sound/generalinstrument_ay-3-8910.pdf).
PSG_DAC = np.array([0.0] + [2.0 ** (-(15 - n) / 2.0) for n in range(1, 16)])
# MAME ay8910.cpp, lines 623-631 and build_single_table / build_3D_table (read at master, 2026-10): the pull-up and pull-down
# resistors of each chip, Ohm; RU is left out at level 0 on the AY, whose level 0 is off (zero_is_off).
PSG_CHIPS = {"ay8910": (800e3, 8e6, (15950, 15350, 15090, 14760, 14275, 13620, 12890, 11370, 10600, 8590, 7190, 5985, 4820, 3945, 3017, 2345), True),
             "ym2149": (630.0, 801.0, (73770, 37586, 27458, 21451, 15864, 12371, 8922, 6796, 4763, 3521, 2403, 1737, 1123, 762, 438, 251), False)}
PSG_WESTCOTT = (1.147, 1.162, 1.169, 1.178, 1.192, 1.213, 1.238, 1.299, 1.336, 1.457, 1.573, 1.707, 1.882, 2.06, 2.32, 2.58)   # V, RL 2000, Dec. 2001 (ay8910.cpp)


def psg_levels(chip="ay8910", load=1000.0) -> np.ndarray:
    """16 output levels of one channel into `load` Ohm, 0..1 (MAME build_single_table, without its offset and scale)."""
    ru, rd, res, off = PSG_CHIPS[chip]
    g = 1.0 / np.array(res, float)
    gu = np.full(16, 1.0 / ru)
    if off:
        gu[0] = 0.0
    v = (g + gu) / (g + gu + 1.0 / rd + 1.0 / float(load))
    return (v - v[0]) / (v[15] - v[0])


PSG_YM_ENV = (103350, 73770, 52657, 37586, 32125, 27458, 24269, 21451, 18447, 15864, 14009, 12371, 10506, 8922, 7787, 6796,
              5689, 4763, 4095, 3521, 2909, 2403, 2043, 1737, 1397, 1123, 925, 762, 578, 438, 332, 251)


def _ym_env_levels(load=1000.0) -> np.ndarray:
    ru, rd = PSG_CHIPS["ym2149"][:2]
    g = 1.0 / np.array(PSG_YM_ENV, float)
    v = (g + 1.0 / ru) / (g + 1.0 / ru + 1.0 / rd + 1.0 / float(load))
    return (v - v[0]) / (v[-1] - v[0])


_PSG_ENV = {"ay8910": lambda dac: (15, 2, dac), "ym2149": lambda dac: (31, 1, _ym_env_levels())}


def _choice_psg(v, opts):
    if v not in opts:
        raise ValueError(f"must be one of {', '.join(opts)}, got {v!r}")
    return v


PSG_MOSFET_KN = (0.00076, 0.80536, 1.13106, 1.65952, 2.42261, 3.60536, 5.34893, 8.96871, 10.97202, 19.32370, 29.01935,
                 38.82026, 55.50539, 78.44395, 109.49257, 153.72985)


def psg_mosfet_levels(rd=1000.0, vth=1.465385778, vg=4.9) -> np.ndarray:
    """16 source voltages of the AY's output NMOS in saturation into RD, Vs = Vg' + 1/(2 Kn RD) - sqrt((Vg' + 1/(2 Kn RD))^2 - Vg'^2), 0..1."""
    g = vg - vth
    p2 = 1.0 / (2.0 * np.array(PSG_MOSFET_KN) * 1e-6 * float(rd)) + g
    v = p2 - np.sqrt(p2 * p2 - g * g)
    return (v - v[0]) / (v[-1] - v[0])


PSG_DACS = {"log": PSG_DAC, "mosfet": psg_mosfet_levels(), "ay8910": psg_levels("ay8910"), "ym2149": psg_levels("ym2149"),
            "westcott": (np.array(PSG_WESTCOTT) - PSG_WESTCOTT[0]) / (PSG_WESTCOTT[15] - PSG_WESTCOTT[0])}
PSG_CLOCK = 1_773_450.0   # ZX Spectrum 128: 35.469 MHz crystal / 20 (MAME sinclair/spec128.h X1_128_SINCLAIR, spec128.cpp AY_SLOT); CPU 3.54690 MHz = /10 (WoS 128K reference); datasheet range 1-2 MHz


def psg_tone_period(freq: float, clock: float = PSG_CLOCK) -> int:
    """12-bit tone period TP for a pitch: f = clock / (16 * TP) (datasheet, g15 §2)."""
    return int(min(max(round(clock / (16.0 * max(freq, 1e-3))), 1), 4095))


def beeper_hz(freq: float) -> float:
    """ZX Spectrum ROM BEEP: delay count HL = round(437500 / f - 30.125); sounding f = 437500 / (HL + 30.125)."""
    hl = max(0, round(437500.0 / max(freq, 1.0) - 30.125))
    return 437500.0 / (hl + 30.125)


def beeper(freq, dur: float, sr: int = DEFAULT_SR) -> np.ndarray:
    """1-bit ZX beeper square (port FE bit 4) at the ROM routine's quantised pitch, levels 0/1 around their mean."""
    f = beeper_hz(to_hz(freq))
    t = np.arange(samples(dur, sr)) / sr
    return np.where(np.mod(t * f, 1.0) < 0.5, 0.5, -0.5)


PSG_DUTY = (0x80000000, 0xC0000000, 0xF0000000, 0xFF000000, 0xFFFF0000, 0xFFFFFF00, 0xFFFFFFF0, 0xFFFFFFFC, 0xFFFFFFFE)


@nb.njit(cache=True)
def _psg_core(n, tp, npd, ep, shape, level, noise_mix, seed, dac, mixer_and, emask, ediv, edac, duty_mask=0):
    """One AY channel pair at fs = clock/8 (every divider is then a whole number of samples):
    tone flips every TP samples (period 16*TP clocks), the 17-bit LFSR steps every 2*NP samples
    (f = clock/(16*NP)), the envelope takes emask+1 steps per ramp, one every ediv*EP samples.
    Output is the unipolar DAC voltage: (1 - noise_mix) * tone channel + noise_mix * noise channel,
    or with `mixer_and` the register-7 gate (tone | tone_off) & (noise | noise_off).
    A `duty_mask` (AY8930 expanded mode) splits each period into 32 steps of TP/16 samples, the tone being the mask's bit."""
    out = np.empty(n)
    di = 0
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
    ns = emask + 1
    t_off = 1 if noise_mix >= 1.0 else 0
    n_off = 1 if noise_mix <= 0.0 else 0
    for i in range(n):
        if shape < 0:
            a = dac[level]
        else:
            cyc = es // ns
            pos = es % ns
            if cyc == 0:
                v = pos if att == 1 else emask - pos
            elif cont == 0:
                v = 0
            elif hld == 1:
                v = (emask if att == 1 else 0) ^ (emask if alt == 1 else 0)
            else:
                up = att ^ (alt & (cyc & 1))
                v = pos if up == 1 else emask - pos
            a = edac[v]
        if mixer_and:
            out[i] = a * ((tone | t_off) & ((lfsr & 1) | n_off))
        else:
            out[i] = a * ((1.0 - noise_mix) * tone + noise_mix * (lfsr & 1))
        if duty_mask:
            tc += 16
            while tc >= tp:
                tc -= tp
                di = (di - 1) & 31
                tone = (duty_mask >> di) & 1
        else:
            tc += 1
            if tc >= tp:
                tc = 0
                tone ^= 1
        nc += 1
        if nc >= 2 * npd:
            nc = 0
            lfsr = (lfsr >> 1) | (((lfsr ^ (lfsr >> 3)) & 1) << 16)
        ec += 1
        if ec >= ediv * ep:
            ec = 0
            if es < 1 << 30:
                es += 1
    return out


_PSG_GAIN = 0.58      # level trim: about -12 dB at vel 0.9


@instrument("psg", "AY-3-8910 sound chip voice: square from a 12-bit clock divider, 16-step log DAC, LFSR noise, hardware envelope shapes.",
            family="retro", span=("C2", "C6"),
            clock=(PSG_CLOCK, "chip clock in Hz (1773450 ZX Spectrum, 2000000 Atari ST, 1789772 MSX): sets the pitch grid f = clock/(16*TP)"),
            level=(15, "fixed volume 0..15 on the 16-step log DAC (3 dB per step)"),
            noise=(0.0, "0 = tone channel only .. 1 = noise channel only (two chip channels mixed)"),
            noise_period=(8, "noise period NP 1..31: LFSR clock = clock/(16*NP) (1 = hiss, 31 = rumble)"),
            shape=(-1, "hardware envelope shape 0..15 (R15 bits Continue/Attack/Alternate/Hold); -1 = off, use `level`. 8 \\\\\\\\, 10 \\/\\/, 12 ////, 14 /\\/\\, 9 decay once, 13 attack and hold"),
            env_hz=(8.0, "envelope ramps per second (quantised to the 16-bit period EP = clock/(256*env_hz)); audio rates give buzz-bass"),
            bright=(0.5, "0 = dull .. 0.5 = default .. 1 = raw unfiltered"), seed=(0, "noise LFSR start state"),
            dac=("log", "log (datasheet, 3 dB a step) | ay8910 | ym2149 (MAME's resistor tables) | westcott (levels measured on an AY, 2001) | mosfet (the AY output transistor into 1 kOhm)"),
            mixer=("sum", "sum = tone and noise channels added by `noise` | and = one channel through the register-7 gate: noise 0 tone only, 1 noise only, between both gated (tone AND noise)"),
            chip=("ay8910", "envelope: ay8910 = 16 steps on `dac` | ym2149 = 32 finer steps on the YM's own envelope resistor table"),
            duty=(None, "AY8930 expanded-mode pulse width 0..8: 3.125, 6.25, 12.5, 25, 50, 75, 87.5, 93.75, 96.875 %; null = the plain square"))
def psg(freq, dur, sr=DEFAULT_SR, vel=1.0, clock=PSG_CLOCK, level=15, noise=0.0, noise_period=8, shape=-1, env_hz=8.0,
        bright=0.5, seed=0, dac="log", mixer="sum", chip="ay8910", duty=None):
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
                  int(np.clip(level, 0, 15)), float(np.clip(noise, 0.0, 1.0)), _seed(seed) & 0xFFFF, PSG_DACS[dac],
                  _choice_psg(mixer, ("sum", "and")) == "and", *_PSG_ENV[_choice_psg(chip, tuple(_PSG_ENV))](PSG_DACS[dac]),
                  0 if duty is None else PSG_DUTY[int(np.clip(duty, 0, 8))])
    x = _to_sr(x, fs, sr)[:samples(dur + rel, sr)]
    x = F.dc_block(_soften(x, freq, sr, harm=6, lo=2200, hi=3800, bright=bright), sr)   # the output is AC-coupled
    return _fade(x, sr, 1.0, rel * 1000.0) * (_PSG_GAIN * (0.5 + 0.5 * vel))


@instrument("psg3", "Three AY/YM channels tied to one load resistor, as on machines that wire them together: the mix is not a sum, loud channels squash each other and intermodulate.",
            family="retro", span=("C2", "C6"), third=(4, "semitones, second channel"), fifth=(7, "semitones, third channel"), level=(15, "0..15"),
            chip=("ay8910", "ay8910 | ym2149"), load=(1000.0, "Ohm"), clock=(PSG_CLOCK, "Hz"), bright=(0.5, "0..1"))
def psg3(freq, dur, sr=DEFAULT_SR, vel=1.0, third=4, fifth=7, level=15, chip="ay8910", load=1000.0, clock=PSG_CLOCK, bright=0.5):
    """MAME build_3D_table: V = (n/RU + sum 1/Ri) / (n/RU + 3/RD + 1/RL + sum 1/Ri), n = channels not at an off level 0."""
    ru, rd, res, off = PSG_CHIPS[chip]
    g = 1.0 / np.array(res, float)
    fs = float(np.clip(clock, 2.0e5, 8.0e6)) / 8.0
    n = int((dur + 0.01) * fs) + 8
    i = np.arange(n)
    lv = int(np.clip(level, 0, 15))
    on = [((i // psg_tone_period(float(freq) * 2.0 ** (s / 12.0), fs * 8.0)) & 1) * lv for s in (0, third, fifth)]
    gs = sum(g[j] for j in on)
    k = sum(((j != 0) | (not off)).astype(float) for j in on) / ru
    v = (k + gs) / (k + gs + 3.0 / rd + 1.0 / float(load))
    x = _to_sr(v, fs, sr)[:samples(dur + 0.01, sr)]
    x = F.dc_block(_soften(x, freq, sr, harm=6, lo=2200, hi=3800, bright=bright), sr)
    return _fade(x / (np.max(np.abs(x)) + 1e-12), sr, 1.0, 10.0) * (_PSG_GAIN * (0.5 + 0.5 * vel))


@nb.njit(cache=True)
def _soft_sync(n, master, slave, window):
    """The slave restarts when the master wraps only if it is within `window` of its own zero (Brandt, Hard Sync Without Aliasing, ICMC 2001)."""
    out = np.empty(n)
    m = 0.0
    s = 0.0
    for i in range(n):
        m += master
        s = (s + slave) % 1.0
        if m >= 1.0:
            m -= 1.0
            if min(s, 1.0 - s) < window:
                s = 0.0
        out[i] = 2.0 * s - 1.0
    return out


@instrument("soft_sync", "Oscillator soft sync: a saw that a master only restarts when it is near its own zero, so the tone is nearer the slave than hard sync's and less harsh.",
            family="synth", span=("C2", "C6"), ratio=(1.7, "slave pitch / note"), window=(0.35, "0..0.5 of a cycle around zero; 0.5 = hard sync. Must hold the slave phase at each master wrap or the pitch falls: ratio 1.7 needs > 0.3"), bright=(0.5, "0..1"))
def soft_sync(freq, dur, sr=DEFAULT_SR, vel=1.0, ratio=1.7, window=0.35, bright=0.5):
    fs = 8.0 * RSR                                      # naive, oversampled: Brandt's minBLEP is not used
    x = _soft_sync(int((dur + 0.02) * fs) + 8, float(freq) / fs, float(freq) * float(np.clip(ratio, 0.25, 16.0)) / fs, float(np.clip(window, 0.0, 0.5)))
    x = F.dc_block(_soften(_to_sr(x, fs, sr)[:samples(dur + 0.02, sr)], freq, sr, harm=8, lo=2000, hi=3000, bright=bright), sr)
    return _fade(x, sr, 2.0, 15.0) * (0.45 * (0.5 + 0.5 * vel))


_BYTEBEATS = (lambda t: t * (((t >> 12) | (t >> 8)) & (63 & (t >> 4))), lambda t: t * (((t >> 9) | (t >> 13)) & (25 & (t >> 6))),
              lambda t: t * (((t >> 5) | (t >> 8)) & 63), lambda t: t * (((t >> 11) & (t >> 8)) & (123 & (t >> 3))),
              lambda t: t * (t >> 8 * ((t >> 15) | (t >> 8)) & (20 | (t >> 19) * 5 >> t | (t >> 3))),
              lambda t: t * (t >> ((t >> 9) | (t >> 8)) & (63 & (t >> 4))), lambda t: (t >> 7 | t | t >> 6) * 10 + 4 * (t & t >> 13 | t >> 6))


@sfx("bytebeat", "Bytebeat: music from one integer formula of the sample counter (seven of them, from Alejandro Olarte's SuperCollider examples).",
     formula=(0, "0..6"), rate=(8, "1..10: the counter runs at the sample rate / this"), dur=(4.0, "s"), level=(0.7, "peak"), soften=(0.5, "0..1 rounds the top"))
def bytebeat(sr=DEFAULT_SR, formula=0, rate=8, dur=4.0, level=0.7, soften=0.5):
    """Bytebeats.scd:18-45: t is 16 bits, the value taken mod 2^16 to +-1. No user formulas: nothing is evaluated."""
    t = (np.arange(samples(dur, sr), dtype=np.int64) // int(np.clip(rate, 1, 10))) & 0xFFFF
    y = (_BYTEBEATS[int(np.clip(formula, 0, 6))](t) & 0xFFFF) / 32768.0 - 1.0
    return _finish(F.dc_block(y, sr), sr, level, soften)


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
        r.update(name="zzfx_voice", params=dict(z) if isinstance(z, dict) else {"p": z if isinstance(z, str) else list(z or [])})
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


def _zzfxm_tracks(data, engine="zzfx"):
    """Raw ZzFXM song [instruments, patterns, sequence, bpm] (g17 §D.1) -> (tracks, total_rows, row_seconds).
    Note N.A: pitch = instrument frequency * 2^((N-12)/12), volume 1 - A; -1 = note off; 0 = keep ringing.
    A volume-only cell 0.A sets the ringing note's volume to 1 - A without retriggering it (event f = -2, vel = 1 - A).
    Channel 0 sets a pattern's row count (longer channels are cut)."""
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
                elif n == 0 and note > 0.0:
                    ev.append((row0 + i, -2.0, 1.0 - note, None))
                elif n > 0:
                    ii, pan = int(ch[0] or 0), float(ch[1] or 0.0)
                    if (ii, pan) not in res:
                        res[(ii, pan)] = {"drum": False, "name": "zzfx_voice", "gain": 1.0, "pan": pan,
                                          "params": {"p": list(insts[ii]), "hold": False, "engine": engine}}
                    f0 = insts[ii][2] if len(insts[ii]) > 2 and insts[ii][2] is not None else 220.0
                    ev.append((row0 + i, float(f0) * 2.0 ** ((n - 12) / 12.0), 1.0 - (note - n), res[(ii, pan)]))
            row0 += rows
        tracks.append(ev)
    return tracks, sum(len(pats[s][0]) - 2 for s in seq), beat / RSR


FRAME_HZ = {"zx48": 3.5e6 / 69888, "zx128": 3.5469e6 / 70908, "pal": 50.0}
MOD_TAGS = {b"M.K.": 4, b"M!K!": 4, b"FLT4": 4, b"4CHN": 4, b"6CHN": 6, b"8CHN": 8, b"FLT8": 8}


def mod_note(period: int) -> int:
    """ProTracker period -> ZzFXM note number (12 = the instrument's own frequency)."""
    return int(70 - (math.log2(period) - 7) * 12) - 36


def _mod_tracks(layer: dict):
    """31-sample ProTracker MOD -> (tracks, total_rows, row_seconds), as the ZzFXM converter does it: notes from
    the period, volume (sample default or Cxx) as attenuation 1 - v/64 (at most 0.99, 2 decimals), volume slide Axy adds
    x/64 (or y/64 when x = 0) per row, whatever its direction, even channels left / odd right, Dxx ends the pattern, speed/tempo from the first row only,
    position jumps ignored. Samples map to `instruments` {"<n>": instrument} else a ZzFXM instrument whose name
    the sample name contains, else the plain ZzFX tone."""
    blob = layer.get("data")
    if blob is None:
        with open(layer["path"], "rb") as fh:
            blob = fh.read()
    blob = bytes(blob)
    tag = blob[1080:1084]
    nch = MOD_TAGS.get(tag) or (int(tag[:2]) if tag[2:] == b"CH" and tag[:2].isdigit() else 0)
    if not nch:
        raise ValueError("song mod: only 31-sample MOD files are read (tag M.K., FLT4, 6CHN, 8CHN, xxCH)")
    names = [blob[20 + 30 * i:42 + 30 * i].split(b"\0")[0].decode("latin-1").strip().lower() for i in range(31)]
    vols = [min(64, blob[45 + 30 * i]) for i in range(31)]
    order = list(blob[952:952 + blob[950]])
    defs = layer.get("instruments") or {}
    res: dict = {}

    def inst(s, ch):
        if (s, ch % 2) not in res:
            d = defs.get(str(s), defs.get(s))
            f0 = 220.0
            if d is None:
                hit = [k for k in ZZFXM_INSTRUMENTS if names[s - 1] and k in names[s - 1]]
                d = {"zzfx": {"p": hit[0], "hold": False}} if hit else {"zzfx": None}
                if hit and len(ZZFXM_INSTRUMENTS[hit[0]]) > 2 and ZZFXM_INSTRUMENTS[hit[0]][2] is not None:
                    f0 = float(ZZFXM_INSTRUMENTS[hit[0]][2])
            r = _resolve(d, {})
            r["pan"] = -1.0 if ch % 2 == 0 else 1.0
            res[(s, ch % 2)] = (r, f0)
        return res[(s, ch % 2)]

    speed, tempo = 6, 125
    tracks: list[list] = [[] for _ in range(nch)]
    att = [0.0] * nch
    cur = [0] * nch
    row0 = 0
    for k, o in enumerate(order):
        base = 1084 + o * 64 * nch * 4
        cut = 64
        for row in range(64):
            brk = False
            for ch in range(nch):
                b0, b1, b2, b3 = blob[base + (row * nch + ch) * 4:base + (row * nch + ch) * 4 + 4]
                s, per, fx = (b0 & 0xF0) | (b2 >> 4), ((b0 & 0x0F) << 8) | b1, b2 & 0x0F
                if fx == 0xF and k == 0 and row == 0 and b3:
                    speed, tempo = (b3, tempo) if b3 < 32 else (speed, b3)
                if s:
                    cur[ch] = s
                    att[ch] = min(0.99, round(1.0 - vols[s - 1] / 64.0, 2))
                if fx == 0xC:
                    att[ch] = min(0.99, round(1.0 - min(64, b3) / 64.0, 2))
                elif fx == 0xA:
                    att[ch] = round(min(1.0, att[ch] + ((b3 >> 4) if b3 > 15 else b3) / 64.0), 2)
                if per and cur[ch]:
                    r, f0 = inst(cur[ch], ch)
                    tracks[ch].append((row0 + row, f0 * 2.0 ** ((mod_note(per) - 12) / 12.0), 1.0 - att[ch], r))
                elif fx in (0xA, 0xC):
                    tracks[ch].append((row0 + row, -2.0, 1.0 - att[ch], None))
                brk = brk or fx == 0xD
            if brk:
                cut = row + 1
                break
        row0 += cut
    return tracks, row0, speed * 2.5 / tempo


@layer_type("song")
def song(layer: dict, sr: int, q: float) -> np.ndarray:
    """Tracker layer (g17 §F.4, §D). Patterns are rows of cells per channel:
    `C-4` / `C#4` note, `.` nothing (the note keeps ringing), `^^` note off, `@0.6` per-cell volume,
    and on drum channels `x` hit, `X` accent, `o` ghost. `rows` cells make one beat (default 4), `order`
    lists the patterns to play. Channels are genny instruments/drums rendered with `render_note` /
    `render_drum`. `"format": "zzfxm"` + `"data"` imports a raw ZzFXM song array instead.
    The result is exactly total_rows * row_length long (+ `tail` seconds of ring-out)."""
    fmt = str(layer.get("format", "")).lower()
    if fmt == "zzfxm":
        tracks, total, row_s = _zzfxm_tracks(layer["data"], layer.get("engine", "zzfx"))
    elif fmt == "mod":
        tracks, total, row_s = _mod_tracks(layer)
    else:
        tracks, total = _native_tracks(layer)
        row_s = q / float(layer.get("rows", 4))
        if layer.get("ticks"):
            row_s = float(layer["ticks"]) / FRAME_HZ[layer.get("frame", "zx128")]
    parts = []
    cache: dict = {}
    for ev in tracks:
        for j, (row, f, vel, r) in enumerate(ev):
            if f is not None and f < 0:
                continue
            rest = [e for e in ev[j + 1:] if e[1] != -2.0]
            nxt = rest[0][0] if rest else total
            vols = [(e[0], e[2]) for e in ev[j + 1:] if e[1] == -2.0 and e[0] < nxt]
            key = (id(r), f, vel, None if r["drum"] else nxt - row, tuple(vols))
            if key not in cache:
                if r["drum"]:
                    y = D.render_drum(r["name"], sr, min(1.0, vel), **r["params"]) * vel
                else:
                    y = I.render_note(r["name"], f, (nxt - row) * row_s, sr, vel, **r["params"])
                    for vr, vv in vols:                   # the note rings on at the new volume
                        y[samples((vr - row) * row_s, sr):] *= vv / max(vel, 1e-6)
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
