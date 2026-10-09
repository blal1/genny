# Adapted for Genny v0.3 (2026): reorganized under genny.physical and integrated with
# Genny high-level APIs. Original procedural/Klang code © 2025 Chris Nash.
# Licensed under Klang Open License 1.0; see KLANG_LICENSE.txt.
"""Voices: glottal sources, formant filters, a vocal tract, creature calls, choir and breath.

Every model here is a port of a named source (PLAN.md rule 1). Numbers not given by a source
are marked ``# UNSOURCED:`` with the reason.

Sources (research notes in ``synthgen/out/research``; originals where they exist locally)
------------------------------------------------------------------------------------------
* Farnell, *Designing Sound* Practical 29 "Mammals" (pdf p.604-614) and his thesis fig.
  20.33-20.39 (``out/text/content.pdf.txt`` L22080-22345): the "flapping" cord waveshaper,
  the parallel band-pass tract comb, the soft rise/fall articulation envelope, the lion roar
  arc 30 -> 240 -> 120 Hz, the Carmell/CSLU vowel table and the "schwa box" (``bp~ Q34``).
  Research 03 §3.1, 05 §7.1.
* Farnell Pd patches ``04-insects/main-insectsall.pd`` (``pd frog``, ``pd frog2``,
  ``pd frog reson``, ``circada2``, ``pd rattler``): research 03 §3.2, §3.4.
* Pd/Puckette ``F01.pulse.pd`` (narrowed raised cosine), ``F02.just.say.pd`` ("just say
  one word", the only complete speech-like Pd patch), ``F12.paf.pd`` (PAF), ``E03.octave.divider``:
  research 09 §B8, §B9; TTEM p.158-172, p.126 (research 10 §8.3, §8.6).
* Pd ``bp~``, ``vcf~``, ``hip~``, ``lop~``, ``noise~`` coefficient code:
  ``libpd/pure-data/src/d_filter.c``, ``d_osc.h`` (read directly).
* Pink Trombone (Neil Thapen 2017) via the Faust port ``faustlibraries/pinktrombone.lib``
  (David Braun): LF glottis, 44-section Kelly-Lochbaum tract, 28-section nose, turbulence,
  plosive transients, and the simplex drift of ``noises.lib`` (``simplex1_lf``).
  Research 08 §5.4.
* Faust ``physmodels.lib`` ``formantValues`` (Csound manual table, 5 voices x 5 vowels, extracted
  from the file), ``autobendFreq``, ``vocalEffort``, ``formantFilterBP``, ``fof``,
  ``skirtWidthMultiplier``. The same table in dB is in Csound ``fof2-2.csd`` (research 11 C8).
* STK ``Phonemes.cpp`` (32 phonemes, 4 formants: freq, pole radius, dB - radii tuned at 44.1 kHz),
  ``FormSwep`` (zeros at +-1, b0 = 0.5 - 0.5 r^2), ``VoicForm`` (OneZero -0.9, OnePole
  0.97 - 0.2 amp, parallel formants, noise envelope), ``SingWave`` + ``Modulate`` (6 Hz vibrato,
  sample-and-hold noise every 330*fs/22050 samples through a 0.999 one-pole, randomGain 0.005),
  ``rawwaves/impuls20.raw`` (checked: equals sum_{k=1..20} cos(2 pi k phase), corr 0.99999999).
  Research 07 §8. ChucK ``voic-o-form.ck`` vibratoGain 0.01 (research 11 C8).
* Cook, *Real Sound Synthesis* p.67-69: whispered vowels keep the formant peaks, so breath =
  noise through vowel formants; FOF grains per voice with independent jitter for choirs
  (research 04 §9.1).
* Smyth & Smith, "The sounds of the avian syrinx" (DAFx-02) eq. 1, 5, 6, 8-11, 14 and fig. 4,
  decoded from the mis-encoded font in ``out/text/...The_Sounds_Of_The_Avian_Syrinx...txt``;
  Kahrs & Avanzini (DAFx-01) raven tube dimensions. Research 06 §9.2-9.3.

All rates are 48 kHz (``SR`` from core). Per-sample loops use numba; every time-varying filter
recomputes its coefficients every sample.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numba
import numpy as np
from scipy import signal as _sig

from .core import SR, rate_radius

TWO_PI = 2.0 * math.pi

# ==========================================================================================
# Small utilities
# ==========================================================================================


def _n_of(dur: float | None, *envs) -> int:
    if dur is not None:
        return int(round(dur * SR))
    lens = [len(e) for e in envs if isinstance(e, np.ndarray) and e.ndim == 1]
    if not lens:
        raise ValueError("give dur or at least one per-sample envelope")
    return max(lens)


def _env(v, n: int) -> np.ndarray:
    """Scalar -> constant array; array of another length -> linearly stretched to n."""
    if np.isscalar(v):
        return np.full(n, float(v))
    a = np.asarray(v, dtype=np.float64)
    if a.ndim == 0:
        return np.full(n, float(a))
    if len(a) == n:
        return a.copy()
    if len(a) == 1:
        return np.full(n, float(a[0]))
    return np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, len(a)), a)


def keyframes(points, n: int, sr: float = SR) -> np.ndarray:
    """Piecewise-linear envelope from ``[(time_s, value), ...]`` (the Pd ``line``/Csound ``line``)."""
    t = np.array([p[0] for p in points], dtype=np.float64)
    v = np.array([p[1] for p in points], dtype=np.float64)
    return np.interp(np.arange(n) / sr, t, v)


def _phasor(f: np.ndarray, sr: float = SR, start: float = 0.0) -> np.ndarray:
    """Pd ``phasor~``: phase in [0,1), first sample = start."""
    ph = start + np.concatenate(([0.0], np.cumsum(f[:-1] / sr)))
    return ph - np.floor(ph)


def _edge_fade(x: np.ndarray, ms: float = 5.0) -> np.ndarray:
    """Squared-sine fade at both ends so a hard-started patch does not click.

    UNSOURCED: the Pd patches start and stop hard (dac~ on/off); 5 ms is the minimum that
    removes the click (PLAN rule 5)."""
    m = min(len(x) // 2, int(ms * 1e-3 * SR))
    if m > 1:
        w = np.sin(0.5 * np.pi * np.arange(m) / m) ** 2
        x[:m] *= w
        x[-m:] *= w[::-1]
    return x


# ==========================================================================================
# Pd filters (d_filter.c / d_osc.h), coefficients per sample
# ==========================================================================================


@numba.njit(cache=True)
def _pd_qcos(f):
    if -0.5 * 3.14159 <= f <= 0.5 * 3.14159:
        g = f * f
        return ((g * g * g * (-1.0 / 720.0) + g * g * (1.0 / 24.0)) - g * 0.5) + 1.0
    return 0.0


@numba.njit(cache=True)
def _pd_bp(x, f, q, sr):
    """Pd ``bp~`` (sigbp_docoef + sigbp_perform) with per-sample f and q."""
    n = len(x)
    y = np.zeros(n)
    last = 0.0
    prev = 0.0
    for i in range(n):
        ff = f[i]
        if ff < 0.001:
            ff = 10.0
        qq = q[i] if q[i] > 0.0 else 0.0
        omega = ff * (2.0 * 3.14159) / sr
        if qq < 0.001:
            omr = 1.0
        else:
            omr = omega / qq
        if omr > 1.0:
            omr = 1.0
        r = 1.0 - omr
        c1 = 2.0 * _pd_qcos(omega) * r
        c2 = -r * r
        gain = 2.0 * omr * (omr + r * omega)
        out = x[i] + c1 * last + c2 * prev
        y[i] = gain * out
        prev = last
        last = out
    return y


@numba.njit(cache=True)
def _pd_vcf(x, f, q, sr):
    """Pd ``vcf~`` (d_osc.h SIGVCFPERF): complex one-pole resonator, band-pass output."""
    n = len(x)
    y = np.zeros(n)
    re = 0.0
    im = 0.0
    isr = 6.28318 / sr
    for i in range(n):
        qq = q[i]
        qinv = 1.0 / qq if qq > 0.0 else 0.0
        ampcorrect = 2.0 - 2.0 / (qq + 2.0)
        cf = f[i] * isr
        if cf < 0.0:
            cf = 0.0
        r = 1.0 - cf * qinv if qinv > 0.0 else 0.0
        if r < 0.0:
            r = 0.0
        omr = 1.0 - r
        cr = r * math.cos(cf)
        ci = r * math.sin(cf)
        re2 = re
        re = ampcorrect * omr * x[i] + cr * re2 - ci * im
        im = ci * re2 + cr * im
        y[i] = re
    return y


@numba.njit(cache=True)
def _pd_hip(x, f, sr):
    """Pd ``hip~`` (sighip_ft1 + sighip_perform, normalised form)."""
    coef = 1.0 - f * (2.0 * 3.14159) / sr
    if coef < 0.0:
        coef = 0.0
    elif coef > 1.0:
        coef = 1.0
    n = len(x)
    y = np.zeros(n)
    if coef >= 1.0:
        for i in range(n):
            y[i] = x[i]
        return y
    normal = 0.5 * (1.0 + coef)
    last = 0.0
    for i in range(n):
        new = x[i] + coef * last
        y[i] = normal * (new - last)
        last = new
    return y


@numba.njit(cache=True)
def _pd_lop(x, f, sr):
    """Pd ``lop~``: coef = f*2pi/sr clipped to [0,1]; y += coef*(x - y)."""
    coef = f * (2.0 * 3.14159) / sr
    if coef < 0.0:
        coef = 0.0
    elif coef > 1.0:
        coef = 1.0
    n = len(x)
    y = np.zeros(n)
    last = 0.0
    for i in range(n):
        last = coef * x[i] + (1.0 - coef) * last
        y[i] = last
    return y


def _pd_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pd ``noise~`` is uniform in [-1, 1)."""
    return rng.uniform(-1.0, 1.0, n)


def _bp(x, f, q, sr=SR):
    n = len(x)
    return _pd_bp(np.ascontiguousarray(x, dtype=np.float64), _env(f, n), _env(q, n), float(sr))


def _vcf(x, f, q, sr=SR):
    n = len(x)
    return _pd_vcf(np.ascontiguousarray(x, dtype=np.float64), _env(f, n), _env(q, n), float(sr))


# ==========================================================================================
# Faust resonbp and STK FormSwep, per-sample coefficients
# ==========================================================================================


@numba.njit(cache=True)
def _resonbp_bank(x, fc, q, g, sr):
    """Sum of Faust ``fi.resonbp(fc, Q, gain)``: H(s) = gain*s/(s^2 + s/Q + 1), bilinear with
    prewarping (``tf2s``). Peak gain is gain*Q. fc, q, g have shape (n, K)."""
    n, K = fc.shape
    y = np.zeros(n)
    x1 = np.zeros(K)
    x2 = np.zeros(K)
    y1 = np.zeros(K)
    y2 = np.zeros(K)
    nyq = 0.49 * sr
    for i in range(n):
        xi = x[i]
        acc = 0.0
        for k in range(K):
            f = fc[i, k]
            if f <= 0.0:
                continue
            if f > nyq:
                f = nyq
            c = 1.0 / math.tan(math.pi * f / sr)
            qq = q[i, k]
            d = c * c + c / qq + 1.0
            b0 = g[i, k] * c / d
            a1 = 2.0 * (1.0 - c * c) / d
            a2 = (c * c - c / qq + 1.0) / d
            out = b0 * (xi - x2[k]) - a1 * y1[k] - a2 * y2[k]
            x2[k] = x1[k]
            x1[k] = xi
            y2[k] = y1[k]
            y1[k] = out
            acc += out
        y[i] = acc
    return y


@numba.njit(cache=True)
def _formswep_bank(x, fc, rad, g, sr):
    """Parallel STK ``FormSwep`` filters: a2 = r^2, a1 = -2 r cos(2 pi f/fs), zeros at +-1,
    b0 = 0.5 - 0.5 a2, input times gain. fc, rad, g shape (n, K)."""
    n, K = fc.shape
    y = np.zeros(n)
    x1 = np.zeros(K)
    x2 = np.zeros(K)
    y1 = np.zeros(K)
    y2 = np.zeros(K)
    for i in range(n):
        acc = 0.0
        for k in range(K):
            r = rad[i, k]
            a2 = r * r
            a1 = -2.0 * r * math.cos(TWO_PI * fc[i, k] / sr)
            b0 = 0.5 - 0.5 * a2
            xin = g[i, k] * x[i]
            out = b0 * xin - b0 * x2[k] - a1 * y1[k] - a2 * y2[k]
            x2[k] = x1[k]
            x1[k] = xin
            y2[k] = y1[k]
            y1[k] = out
            acc += out
        y[i] = acc
    return y


# ==========================================================================================
# Formant tables (verbatim)
# ==========================================================================================

# Faust physmodels.lib ``formantValues`` (from the Csound manual), index = voiceType*5 + vowel,
# voiceType 0 alto, 1 bass, 2 countertenor, 3 soprano, 4 tenor; vowel 0 a, 1 e, 2 i, 3 o, 4 u.
_FV_F = np.array([
    [800, 400, 350, 450, 325, 600, 400, 250, 400, 350, 660, 440, 270, 430, 370, 800,
     350, 270, 450, 325, 650, 400, 290, 400, 350],
    [1150, 1600, 1700, 800, 700, 1040, 1620, 1750, 750, 600, 1120, 1800, 1850, 820, 630,
     1150, 2000, 2140, 800, 700, 1080, 1700, 1870, 800, 600],
    [2800, 2700, 2700, 2830, 2530, 2250, 2400, 2600, 2400, 2400, 2750, 2700, 2900, 2700, 2750,
     2900, 2800, 2950, 2830, 2700, 2650, 2600, 2800, 2600, 2700],
    [3500, 3300, 3700, 3500, 3500, 2450, 2800, 3050, 2600, 2675, 3000, 3000, 3350, 3000, 3000,
     3900, 3600, 3900, 3800, 3800, 2900, 3200, 3250, 2800, 2900],
    [4950, 4950, 4950, 4950, 4950, 2750, 3100, 3340, 2900, 2950, 3350, 3300, 3590, 3300, 3400,
     4950, 4950, 4950, 4950, 4950, 3250, 3580, 3540, 3000, 3300]], dtype=np.float64)
# g(0) has only 20 entries in the Faust source; all formant-0 gains are 1 (Csound table: 0 dB).
_FV_G = np.array([
    [1.0] * 25,
    [0.630957, 0.063096, 0.100000, 0.354813, 0.251189, 0.446684, 0.251189, 0.031623,
     0.281838, 0.100000, 0.501187, 0.199526, 0.063096, 0.316228, 0.100000,
     0.501187, 0.100000, 0.251189, 0.281838, 0.158489, 0.501187, 0.199526, 0.177828,
     0.316228, 0.100000],
    [0.100000, 0.031623, 0.031623, 0.158489, 0.031623, 0.354813, 0.354813, 0.158489,
     0.089125, 0.025119, 0.070795, 0.125893, 0.063096, 0.050119, 0.070795,
     0.025119, 0.177828, 0.050119, 0.079433, 0.017783, 0.446684, 0.251189, 0.125893,
     0.251189, 0.141254],
    [0.015849, 0.017783, 0.015849, 0.039811, 0.010000, 0.354813, 0.251189, 0.079433,
     0.100000, 0.039811, 0.063096, 0.100000, 0.015849, 0.079433, 0.031623,
     0.100000, 0.010000, 0.050119, 0.079433, 0.010000, 0.398107, 0.199526, 0.100000,
     0.251189, 0.199526],
    [0.001000, 0.001000, 0.001000, 0.001778, 0.000631, 0.100000, 0.125893, 0.039811,
     0.010000, 0.015849, 0.012589, 0.100000, 0.015849, 0.019953, 0.019953,
     0.003162, 0.001585, 0.006310, 0.003162, 0.001000, 0.079433, 0.100000, 0.031623,
     0.050119, 0.050119]], dtype=np.float64)
_FV_BW = np.array([
    [80, 60, 50, 70, 50, 60, 40, 60, 40, 40, 80, 70, 40, 40, 40, 80, 60, 60, 40, 50,
     50, 70, 40, 70, 40],
    [90, 80, 100, 80, 60, 70, 80, 90, 80, 80, 90, 80, 90, 80, 60, 90, 100, 90, 80, 60,
     90, 80, 90, 80, 60],
    [120, 120, 120, 100, 170, 110, 100, 100, 100, 100, 120, 100, 100, 100, 100,
     120, 120, 100, 100, 170, 120, 100, 100, 100, 100],
    [130, 150, 150, 130, 180, 120, 120, 120, 120, 120, 130, 120, 120, 120, 120,
     130, 150, 120, 120, 180, 130, 120, 120, 130, 120],
    [140, 200, 200, 135, 200, 130, 120, 120, 120, 120, 140, 120, 120, 120, 120,
     140, 200, 120, 120, 200, 140, 120, 120, 135, 120]], dtype=np.float64)

VOICE_TYPES = ("alto", "bass", "countertenor", "soprano", "tenor")
VOWELS = ("a", "e", "i", "o", "u")
_FEMALE = {"alto", "soprano"}                    # Faust voiceGender: alto, soprano female

# Faust bwMultMins / bwMultMaxes (index gender*5 + vowel, gender 0 male, 1 female) and the
# gender pitch ranges used by skirtWidthMultiplier.
_BW_MULT_MIN = np.array([1.0, 1.25, 1.25, 1.0, 1.5, 2.0, 3.0, 3.0, 2.0, 2.0])
_BW_MULT_MAX = np.array([10.0, 2.5, 2.5, 10.0, 4.0, 15.0, 12.0, 12.0, 12.0, 12.0])
_GENDER_FMIN = (82.41, 174.61)
_GENDER_FMAX = (523.25, 1046.5)

# STK Phonemes.cpp (verbatim): name, (voiceGain, noiseGain), 4 x (freq Hz, radius@44.1k, dB).
STK_PHONEMES = {
    "eee": ((1.0, 0.0), ((273, 0.996, 10), (2086, 0.945, -16), (2754, 0.979, -12), (3270, 0.440, -17))),
    "ihh": ((1.0, 0.0), ((385, 0.987, 10), (2056, 0.930, -20), (2587, 0.890, -20), (3150, 0.400, -20))),
    "ehh": ((1.0, 0.0), ((515, 0.977, 10), (1805, 0.810, -10), (2526, 0.875, -10), (3103, 0.400, -13))),
    "aaa": ((1.0, 0.0), ((773, 0.950, 10), (1676, 0.830, -6), (2380, 0.880, -20), (3027, 0.600, -20))),
    "ahh": ((1.0, 0.0), ((770, 0.950, 0), (1153, 0.970, -9), (2450, 0.780, -29), (3140, 0.800, -39))),
    "aww": ((1.0, 0.0), ((637, 0.910, 0), (895, 0.900, -3), (2556, 0.950, -17), (3070, 0.910, -20))),
    "ohh": ((1.0, 0.0), ((637, 0.910, 0), (895, 0.900, -3), (2556, 0.950, -17), (3070, 0.910, -20))),
    "uhh": ((1.0, 0.0), ((561, 0.965, 0), (1084, 0.930, -10), (2541, 0.930, -15), (3345, 0.900, -20))),
    "uuu": ((1.0, 0.0), ((515, 0.976, 0), (1031, 0.950, -3), (2572, 0.960, -11), (3345, 0.960, -20))),
    "ooo": ((1.0, 0.0), ((349, 0.986, -10), (918, 0.940, -20), (2350, 0.960, -27), (2731, 0.950, -33))),
    "rrr": ((1.0, 0.0), ((394, 0.959, -10), (1297, 0.780, -16), (1441, 0.980, -16), (2754, 0.950, -40))),
    "lll": ((1.0, 0.0), ((462, 0.990, 5), (1200, 0.640, -10), (2500, 0.200, -20), (3000, 0.100, -30))),
    "mmm": ((1.0, 0.0), ((265, 0.987, -10), (1176, 0.940, -22), (2352, 0.970, -20), (3277, 0.940, -31))),
    "nnn": ((1.0, 0.0), ((204, 0.980, -10), (1570, 0.940, -15), (2481, 0.980, -12), (3133, 0.800, -30))),
    "nng": ((1.0, 0.0), ((204, 0.980, -10), (1570, 0.940, -15), (2481, 0.980, -12), (3133, 0.800, -30))),
    "ngg": ((1.0, 0.0), ((204, 0.980, -10), (1570, 0.940, -15), (2481, 0.980, -12), (3133, 0.800, -30))),
    "fff": ((0.0, 0.7), ((1000, 0.300, 0), (2800, 0.860, -10), (7425, 0.740, 0), (8140, 0.860, 0))),
    "sss": ((0.0, 0.7), ((0, 0.000, 0), (2000, 0.700, -15), (5257, 0.750, -3), (7171, 0.840, 0))),
    "thh": ((0.0, 0.7), ((100, 0.900, 0), (4000, 0.500, -20), (5500, 0.500, -15), (8000, 0.400, -20))),
    "shh": ((0.0, 0.7), ((2693, 0.940, 0), (4000, 0.720, -10), (6123, 0.870, -10), (7755, 0.750, -18))),
    "xxx": ((0.0, 0.7), ((1000, 0.300, -10), (2800, 0.860, -10), (7425, 0.740, 0), (8140, 0.860, 0))),
    "hee": ((0.0, 0.1), ((273, 0.996, -40), (2086, 0.945, -16), (2754, 0.979, -12), (3270, 0.440, -17))),
    "hoo": ((0.0, 0.1), ((349, 0.986, -40), (918, 0.940, -10), (2350, 0.960, -17), (2731, 0.950, -23))),
    "hah": ((0.0, 0.1), ((770, 0.950, -40), (1153, 0.970, -3), (2450, 0.780, -20), (3140, 0.800, -32))),
    "bbb": ((1.0, 0.1), ((2000, 0.700, -20), (5257, 0.750, -15), (7171, 0.840, -3), (9000, 0.900, 0))),
    "ddd": ((1.0, 0.1), ((100, 0.900, 0), (4000, 0.500, -20), (5500, 0.500, -15), (8000, 0.400, -20))),
    "jjj": ((1.0, 0.1), ((2693, 0.940, 0), (4000, 0.720, -10), (6123, 0.870, -10), (7755, 0.750, -18))),
    "ggg": ((1.0, 0.1), ((2693, 0.940, 0), (4000, 0.720, -10), (6123, 0.870, -10), (7755, 0.750, -18))),
    "vvv": ((1.0, 1.0), ((2000, 0.700, -20), (5257, 0.750, -15), (7171, 0.840, -3), (9000, 0.900, 0))),
    "zzz": ((1.0, 1.0), ((100, 0.900, 0), (4000, 0.500, -20), (5500, 0.500, -15), (8000, 0.400, -20))),
    "thz": ((1.0, 1.0), ((2693, 0.940, 0), (4000, 0.720, -10), (6123, 0.870, -10), (7755, 0.750, -18))),
    "zhh": ((1.0, 1.0), ((2693, 0.940, 0), (4000, 0.720, -10), (6123, 0.870, -10), (7755, 0.750, -18))),
}
STK_FS = 44100.0
STK_SWEEP_S = 1000.0 / STK_FS      # FormSwep sweepRate 0.001 per sample -> 1000 samples

# Farnell fig 20.38 (Carmell, CSLU), F1 F2 F3 Hz; the "schwa box" (fig 20.39) uses bp~ Q 34.
FARNELL_VOWELS = {
    "i:": (280, 2250, 2900), "I": (400, 1900, 2550), "E": (550, 1770, 2490),
    "ae": (690, 1660, 2490), "^": (640, 1190, 2390), "u": (310, 870, 2250),
    "U": (450, 1030, 2380), "@": (500, 1500, 2500), "A": (710, 1100, 2640),
}
FARNELL_Q = 34.0


def formant_table(voice_type: str, vowel) -> dict:
    """Formant parameters of one table entry.

    ``voice_type`` in VOICE_TYPES (Faust/Csound 5 formants: freqs, bws, gains), ``"stk"``
    (STK phoneme name: freqs, radii at 48 kHz, linear gains, voice/noise gains) or ``"farnell"``
    (IPA key of FARNELL_VOWELS: freqs, q=34)."""
    if voice_type == "stk":
        (vg, ng), fm = STK_PHONEMES[vowel]
        return {"freqs": np.array([f[0] for f in fm], float),
                "radii": np.array([rate_radius(f[1], STK_FS) for f in fm]),
                "gains": np.array([10.0 ** (f[2] / 20.0) for f in fm]),
                "voice": vg, "noise": ng}
    if voice_type == "farnell":
        return {"freqs": np.array(FARNELL_VOWELS[vowel], float), "q": np.full(3, FARNELL_Q)}
    vt = VOICE_TYPES.index(voice_type)
    vi = VOWELS.index(vowel) if isinstance(vowel, str) else vowel
    idx = vt * 5 + vi
    return {"freqs": _FV_F[:, idx].copy(), "bws": _FV_BW[:, idx].copy(), "gains": _FV_G[:, idx].copy()}


def _list_interp(table: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Faust ``ba.listInterp``: linear interpolation over a list at fractional index."""
    i0 = np.clip(np.floor(idx).astype(int), 0, table.shape[-1] - 1)
    i1 = np.clip(i0 + 1, 0, table.shape[-1] - 1)
    fr = idx - np.floor(idx)
    return table[..., i0] * (1 - fr) + table[..., i1] * fr


def _param_path(vowel, n: int, voice_type: str, key: str) -> np.ndarray:
    """Per-sample (n, K) array of one formant parameter for a vowel spec.

    vowel: a name; or a list of (time_s, name) keyframes (parameters morph linearly between
    keyframes, as the Csound fof example's ``line`` between vowels); for the Faust/Csound voice
    types also a float or per-sample array of vowel index 0..4 (``ba.listInterp``)."""
    if isinstance(vowel, (list, tuple)) and len(vowel) and isinstance(vowel[0], (list, tuple)):
        times = np.array([k[0] for k in vowel], float)
        vals = np.stack([formant_table(voice_type, k[1])[key] for k in vowel])     # (M, K)
        t = np.arange(n) / SR
        return np.stack([np.interp(t, times, vals[:, j]) for j in range(vals.shape[1])], axis=1)
    if isinstance(vowel, str):
        return np.tile(formant_table(voice_type, vowel)[key], (n, 1))
    if voice_type in VOICE_TYPES:
        vt = VOICE_TYPES.index(voice_type)
        idx = vt * 5 + np.clip(_env(vowel, n), 0.0, 4.0)
        table = {"freqs": _FV_F, "bws": _FV_BW, "gains": _FV_G}[key]
        return _list_interp(table, idx).T.copy()
    raise ValueError("numeric vowel index only for the Faust/Csound voice types")


def _autobend(F: np.ndarray, f0: np.ndarray, voice_type: str) -> np.ndarray:
    """Faust ``autobendFreq`` (Olsen, from chant-lib): formant 0 raised to f0 if below it;
    formant 1 (not countertenor) lowered when F1 >= 1300 and f0 >= 200, raised to 30 + 2 f0."""
    F = F.copy()
    F[:, 0] = np.where(F[:, 0] <= f0, f0, F[:, 0])
    if voice_type != "countertenor":
        f1 = F[:, 1]
        hi = (f1 >= 1300) & (f0 >= 200)
        lo_lim = 30 + 2 * f0
        F[:, 1] = np.where(hi, f1 - (f0 - 200) * (f1 - 1300) / 1050,
                           np.where(f1 <= lo_lim, lo_lim, f1))
    return F


def _vocal_effort(f0: np.ndarray, voice_type: str) -> np.ndarray:
    """Faust ``vocalEffort``: male x (3 + 1.1 (400 - f)/300), female x (0.8 + 1.05 (1000 - f)/1250)."""
    if voice_type in _FEMALE:
        return 0.8 + 1.05 * (1000.0 - f0) / 1250.0
    return 3.0 + 1.1 * (400.0 - f0) / 300.0


def formant_filter(x: np.ndarray, vowel="a", voice_type: str = "bass", f0=None, *,
                   formants: dict | None = None, formant_scale=1.0) -> np.ndarray:
    """Source/filter formant bank with per-sample smooth vowel transitions.

    * Faust/Csound voice types (``"alto" "bass" "countertenor" "soprano" "tenor"``): Faust
      ``formantFilterBP`` - 5 parallel ``fi.resonbp(F, Q=F/bw, gain=g)`` (peak gain g*Q, as in
      Faust), with ``autobendFreq`` and ``vocalEffort`` applied when ``f0`` is given.
    * ``"stk"``: VoicForm's 4 parallel FormSwep filters from the STK phoneme table (radii
      converted from 44.1 kHz, gains from dB). A phoneme change in STK glides linearly over
      ``STK_SWEEP_S``; give keyframes ``[(t, "ahh"), (t + STK_SWEEP_S, "ooo")]`` for that.
    * ``"farnell"``: the schwa box - 3 parallel Pd ``bp~`` at F1..F3, Q 34.
    * ``formants=dict(freqs=, bws=, gains=)``: explicit arrays, shape (K,) or (n, K); each band a
      resonbp with peak gain = gains (Q = F/bw).

    ``vowel``: name, keyframe list ``[(t_s, name), ...]`` (linear parameter morph) or, for the
    Faust types, a vowel index 0..4 (scalar or per-sample). ``formant_scale`` multiplies every
    formant frequency (STK VoicForm CC4 scales 0.9/1.0/1.1/1.2/1.4 = tract size); scalar or
    per-sample."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    n = len(x)
    sc = _env(formant_scale, n)[:, None]
    if formants is not None:
        F = np.asarray(formants["freqs"], float)
        B = np.asarray(formants["bws"], float)
        G = np.asarray(formants.get("gains", np.ones(F.shape[-1])), float)
        F = np.broadcast_to(F, (n, F.shape[-1])) * sc
        B = np.broadcast_to(B, F.shape)
        G = np.broadcast_to(G, F.shape)
        Q = F / B
        return _resonbp_bank(x, np.ascontiguousarray(F), np.ascontiguousarray(Q),
                             np.ascontiguousarray(G / Q), float(SR))
    if voice_type == "stk":
        F = _param_path(vowel, n, "stk", "freqs") * sc
        R = _param_path(vowel, n, "stk", "radii")
        G = _param_path(vowel, n, "stk", "gains")
        return _formswep_bank(x, np.ascontiguousarray(F), np.ascontiguousarray(R),
                              np.ascontiguousarray(G), float(SR))
    if voice_type == "farnell":
        F = _param_path(vowel, n, "farnell", "freqs") * sc
        y = np.zeros(n)
        for k in range(F.shape[1]):
            y += _pd_bp(x, np.ascontiguousarray(F[:, k]), np.full(n, FARNELL_Q), float(SR))
        return y
    F = _param_path(vowel, n, voice_type, "freqs")
    B = _param_path(vowel, n, voice_type, "bws")
    G = _param_path(vowel, n, voice_type, "gains")
    if f0 is not None:
        f0a = _env(f0, n)
        F = _autobend(F, f0a, voice_type)
        G = G * _vocal_effort(f0a, voice_type)[:, None]
    F = F * sc
    Q = F / B
    return _resonbp_bank(x, np.ascontiguousarray(F), np.ascontiguousarray(Q),
                         np.ascontiguousarray(G), float(SR))


# ==========================================================================================
# Glottal sources
# ==========================================================================================


def raised_cosine_pulse(f0_env, width_env=1.0, dur: float | None = None) -> np.ndarray:
    """Narrowed raised-cosine glottal pulse (Pd ``F01.pulse.pd``, research 09 B8a; Farnell's
    "narrowed raised cosine riding on the flow", research 03 §3.1).

    ``phasor~(f0) - 0.5`` x idx -> ``clip~ -0.5 0.5`` -> ``cos~`` -> +1 -> x0.5 -> ``hip~ 5``.
    ``width_env`` is the index (>= 1): 1 = full raised cosine, k = pulse of duty 1/k
    (tightening the cords narrows the pulse without changing f0)."""
    n = _n_of(dur, f0_env if isinstance(f0_env, np.ndarray) else None)
    f0 = _env(f0_env, n)
    idx = np.maximum(_env(width_env, n), 1.0)
    ph = _phasor(f0)
    y = 0.5 * (np.cos(TWO_PI * np.clip((ph - 0.5) * idx, -0.5, 0.5)) + 1.0)
    return _pd_hip(y, 5.0, float(SR))


def flapping_source(f0_env, ripple_env=1.0, width_env=4.0, noisiness_env=0.0,
                    dur: float | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
    """Farnell's "flapping" cord waveshaper (thesis fig. 20.33, book fig. 52.2).

    phasor(f0) - 0.5 = p;  window w = (cos(2 pi p) + 1)/2  (the raised first cosine);
    ripple c = cos(2 pi p ripple)  ("line scaled by cord ripple": more cycles per lobe);
    y = 1 / (1 + k (w c)^2)  with k = cord width smoothed by ``lop~ 10``;  ``hip~ 1``;
    out = (1 - noisiness) y + noisiness (noise x y)  (noise modulated by the pulses).
    "Narrow pulses with lots of ripple and some noise give a harsh, gritty, snarling excitation,
    whereas wide pulses with little or no ripple and noise give a smooth, humming source."
    The patch prints no default numbers; the object order is read from the figure."""
    rng = rng or np.random.default_rng(0)
    n = _n_of(dur, *[e for e in (f0_env, ripple_env, width_env, noisiness_env)
                     if isinstance(e, np.ndarray)])
    f0 = _env(f0_env, n)
    rip = _env(ripple_env, n)
    wv = _env(width_env, n)
    # lop~ 10 smoothing of the width; pre-rolled so it starts on the control value, not 0
    k = np.maximum(_pd_lop(np.r_[np.full(SR, wv[0]), wv], 10.0, float(SR))[SR:], 0.0)
    p = _phasor(f0) - 0.5
    w = 0.5 * (np.cos(TWO_PI * p) + 1.0)
    c = np.cos(TWO_PI * p * rip)
    m = w * c
    y = 1.0 / (1.0 + k * m * m)
    y = _pd_hip(y, 1.0, float(SR))
    nz = _env(noisiness_env, n)
    return (1.0 - nz) * y + nz * (_pd_noise(n, rng) * y)


# ---- Pink Trombone glottis (pinktrombone.lib ``glottis``) --------------------------------

_SIMPLEX_P = np.array([
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
    140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
    247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
    57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
    60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
    65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
    200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
    52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
    207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
    119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
    129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
    218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
    81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
    184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180], dtype=np.int64)
_GRAD_X = np.array([1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0], float)
_GRAD_Y = np.array([1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1], float)
_F2 = 0.5 * (math.sqrt(3.0) - 1.0)
_G2 = (3.0 - math.sqrt(3.0)) / 6.0


def _simplex1(seed: int, x: np.ndarray) -> np.ndarray:
    """``no.simplex1(seed, x) = simplex2(seed, 1.2 x, -0.7 x)`` (noisejs seeding, Gustavson)."""
    sd = seed if seed >= 256 else seed | (seed << 8)
    lo_b, hi_b = sd & 255, (sd >> 8) & 255

    def perm(k):
        return _SIMPLEX_P[k & 255] ^ np.where(k & 1, lo_b, hi_b)

    xin, yin = 1.2 * x, -0.7 * x
    s = (xin + yin) * _F2
    i = np.floor(xin + s).astype(np.int64)
    j = np.floor(yin + s).astype(np.int64)
    t = (i + j) * _G2
    x0, y0 = xin - i + t, yin - j + t
    ii, jj = i & 255, j & 255
    lower = (x0 > y0).astype(np.int64)
    g0 = perm(ii + perm(jj)) % 12
    g1 = perm(ii + lower + perm(jj + (1 - lower))) % 12
    g2 = perm(ii + 1 + perm(jj + 1)) % 12

    def corner(cx, cy, g):
        t0 = 0.5 - cx * cx - cy * cy
        return np.where(t0 < 0, 0.0, t0 ** 4 * (_GRAD_X[g] * cx + _GRAD_Y[g] * cy))

    n0 = corner(x0, y0, g0)
    n1 = corner(x0 - lower + _G2, y0 - (1 - lower) + _G2, g1)
    n2 = corner(x0 - 1.0 + 2.0 * _G2, y0 - 1.0 + 2.0 * _G2, g2)
    return 70.0 * (n0 + n1 + n2)


def _simplex_lf(seed: int, rate, n: int, fs: float) -> np.ndarray:
    """``no.simplex1_lf(seed, rate)``: x advances by rate/fs per sample from 0."""
    r = _env(rate, n)
    xs = np.concatenate(([0.0], np.cumsum(r[:-1] / fs)))
    return _simplex1(seed, xs)


@numba.njit(cache=True)
def _lf_coefs(rd):
    """pinktrombone.lib ``_lfCoefficients`` (the original's setupWaveform)."""
    if rd < 0.5:
        rd = 0.5
    elif rd > 2.7:
        rd = 2.7
    ra = -0.01 + 0.048 * rd
    rk = 0.224 + 0.118 * rd
    rg = (rk / 4.0) * (0.5 + 1.2 * rk) / (0.11 * rd - ra * (0.5 + 1.2 * rk))
    ta = ra
    tp = 1.0 / (2.0 * rg)
    te = tp + tp * rk
    eps = 1.0 / ta
    shift = math.exp(-eps * (1.0 - te))
    delta = 1.0 - shift
    rhs = ((1.0 / eps) * (shift - 1.0) + (1.0 - te) * shift) / delta
    tot_upper = -(rhs - (te - tp) / 2.0)
    omega = math.pi / tp
    s = math.sin(omega * te)
    alpha = math.log(-math.pi * s * tot_upper / (tp * 2.0)) / (tp / 2.0 - te)
    e0 = -1.0 / (s * math.exp(alpha * te))
    return eps, shift, delta, te, omega, alpha, e0


@numba.njit(cache=True)
def _lf_eval(t, eps, shift, delta, te, omega, alpha, e0):
    if t > te:
        return (shift - math.exp(-eps * (t - te))) / delta
    return e0 * math.exp(alpha * t) * math.sin(omega * t)


def lf_waveform(rd: float, n: int = 512) -> np.ndarray:
    """One period of the normalised LF glottal-flow derivative (``pt.lfWaveform``)."""
    c = _lf_coefs(float(rd))
    return np.array([_lf_eval(k / n, *c) for k in range(n)])


@numba.njit(cache=True)
def _glottis_kernel(freq_target, vib, tens_ui, tens_new, voiced, asp, simp199, fs, block_time):
    n = len(freq_target)
    out = np.zeros(n)
    nmod = np.zeros(n)
    up = 0.13 / block_time / fs
    dn = 0.05 / block_time / fs
    log_rate = math.log(1.1) / block_time / fs
    intensity = 0.0
    smooth = 0.0
    phase = 0.0
    latched = -1.0
    eps = shift = delta = te = omega = alpha = e0 = 0.0
    for k in range(n):
        tgt = 1.0 if voiced[k] > 0.0 else 0.0
        if k == 0:
            intensity = tgt                  # _moveTowards starts on its target
        else:
            d = tgt - intensity
            if d > up:
                d = up
            elif d < -dn:
                d = -dn
            intensity += d
        if intensity < 0.0:
            intensity = 0.0
        elif intensity > 1.0:
            intensity = 1.0
        tl = math.log(max(1.0, freq_target[k]))
        if k == 0 or intensity == 0.0:
            smooth = tl
        else:
            d = tl - smooth
            if d > log_rate:
                d = log_rate
            elif d < -log_rate:
                d = -log_rate
            smooth += d
        newf = math.exp(smooth) * (1.0 + vib[k])
        f = newf if latched <= 0.0 else latched
        nxt = phase + f / fs
        wrap = nxt >= 1.0
        if wrap:
            phase = nxt - 1.0
            latched = newf
        else:
            phase = nxt
            latched = f
        if wrap or k == 0:
            eps, shift, delta, te, omega, alpha, e0 = _lf_coefs(3.0 * (1.0 - tens_new[k]))
        t = tens_ui[k]
        voice = _lf_eval(phase, eps, shift, delta, te, omega, alpha, e0) * intensity * t ** 0.25
        vm = 0.1 + 0.2 * max(0.0, math.sin(TWO_PI * phase))
        nm = t * intensity * vm + (1.0 - t * intensity) * 0.3
        a = intensity * (1.0 - math.sqrt(t)) * nm * asp[k] * (0.2 + 0.02 * simp199[k])
        out[k] = voice + a
        nmod[k] = nm
    return out, nmod


def _resonbp_const(x, fc, q, gain, fs):
    """Faust resonbp with constant coefficients (bilinear, prewarped)."""
    c = 1.0 / math.tan(math.pi * fc / fs)
    d = c * c + c / q + 1.0
    b = np.array([gain * c / d, 0.0, -gain * c / d])
    a = np.array([1.0, 2.0 * (1.0 - c * c) / d, (c * c - c / q + 1.0) / d])
    return _sig.lfilter(b, a, x)


def _glottis(f0, tens, voiced, wobble, rng, fs, block_time, seed):
    """pinktrombone.lib ``glottis`` at rate ``fs``; returns (glottal output, noise modulator).

    Noise-level note (derived): the Faust/JS original draws white noise at 48 kHz; at another
    internal rate the band-passed noise power scales with 1/fs, so the white noise is scaled by
    sqrt(fs/48000) to keep the original level."""
    n = len(f0)
    w = _env(wobble, n)
    vib = (0.005 * np.sin(TWO_PI * 6.0 * np.arange(n) / fs)
           + 0.02 * _simplex_lf(seed, 4.07, n, fs) + 0.04 * _simplex_lf(seed, 2.15, n, fs)
           + w * (0.2 * _simplex_lf(seed, 0.98, n, fs) + 0.4 * _simplex_lf(seed, 0.5, n, fs)))
    tens_new = tens + 0.1 * _simplex_lf(seed, 0.46, n, fs) + 0.05 * _simplex_lf(seed, 0.36, n, fs)
    white = 0.5 * rng.uniform(-1.0, 1.0, n) * math.sqrt(fs / 48000.0)
    asp = _resonbp_const(white, 500.0, 0.5, 2.0, fs)          # _bandpassUnity(500, 0.5)
    s199 = _simplex_lf(seed, 1.99, n, fs)
    return _glottis_kernel(np.ascontiguousarray(f0), vib, np.ascontiguousarray(tens), tens_new,
                           np.ascontiguousarray(voiced), asp, s199, float(fs), float(block_time))


def lf_pulse(f0_env, tenseness_env=0.6, dur: float | None = None,
             rng: np.random.Generator | None = None, *, voiced_env=1.0, wobble=0.0,
             return_modulator: bool = False):
    """Pink Trombone glottis (``pt.glottis``): per-period LF pulse with Rd = 3(1 - tenseness),
    aspiration noise (500 Hz, Q 0.5 band) modulated by the glottal cycle, 6 Hz + simplex
    vibrato, simplex tenseness drift, voicing intensity ramp 0.13/-0.05 per 512-sample block.
    Low tenseness = breathy/lax, high = pressed. Returns the glottal output (and the noise
    modulator if asked)."""
    rng = rng or np.random.default_rng(0)
    n = _n_of(dur, *[e for e in (f0_env, tenseness_env, voiced_env) if isinstance(e, np.ndarray)])
    seed = int(rng.integers(0, 65536))
    out, nm = _glottis(_env(f0_env, n), _env(tenseness_env, n), _env(voiced_env, n), wobble,
                       rng, float(SR), 512.0 / SR, seed)
    return (out, nm) if return_modulator else out


# ==========================================================================================
# Pink Trombone tract (pinktrombone.lib ``tract2Ext``), any number of constrictions
# ==========================================================================================

_PT_N = 44
_PT_BLADE = 10
_PT_TIP = 32
_PT_LIP = 39
_PT_NOSE_LEN = 28
_PT_NOSE_START = _PT_N - _PT_NOSE_LEN + 1       # 17
_PT_MOVE = 15.0


@numba.njit(cache=True)
def _pt_rest(i, tongue_index, tongue_diam):
    if i < _PT_BLADE:
        if i < 7.0 * _PT_N / 44.0 - 0.5:
            return 0.6
        if i < 12.0 * _PT_N / 44.0:
            return 1.1
        return 1.5
    if i >= _PT_LIP:
        return 1.5
    t = 1.1 * math.pi * (tongue_index - i) / (_PT_TIP - _PT_BLADE)
    fixed = 2.0 + (tongue_diam - 2.0) / 1.5
    edge = 1.0
    if i == _PT_BLADE - 2 or i == _PT_LIP - 1:
        edge = 0.8
    elif i == _PT_BLADE or i == _PT_LIP - 2:
        edge = 0.94
    return 1.5 - (1.5 - fixed + 1.7) * math.cos(t) * edge


@numba.njit(cache=True)
def _pt_constrict(i, index, diameter, active, rest):
    d = diameter - 0.3
    if d < 0.0:
        d = 0.0
    if not (active > 0.0 and index >= 2.0 and index < _PT_N and diameter < 3.0 and d < rest):
        return rest
    if index < 25.0:
        width = 10.0
    elif index >= _PT_TIP:
        width = 5.0
    else:
        width = 10.0 - 5.0 * (index - 25.0) / (_PT_TIP - 25.0)
    relpos = abs(i - index) - 0.5
    if relpos <= 0.0:
        shrink = 0.0
    elif relpos > width:
        shrink = 1.0
    else:
        shrink = 0.5 * (1.0 - math.cos(math.pi * relpos / width))
    return d + (rest - d) * shrink


@numba.njit(cache=True)
def _clip01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


@numba.njit(cache=True)
def _pt_tract_kernel(glottal, nmod, fric, t_idx, t_diam, c_idx, c_diam, c_act, nasal, fs):
    n = len(glottal)
    N = _PT_N
    NL = _PT_NOSE_LEN
    ns = _PT_NOSE_START
    nc = c_idx.shape[0]
    out = np.zeros(n)
    R = np.zeros(N)
    L = np.zeros(N)
    jR = np.zeros(N + 1)
    jL = np.zeros(N + 1)
    NR = np.zeros(NL)
    NLf = np.zeros(NL)
    nJR = np.zeros(NL + 1)
    nJL = np.zeros(NL + 1)
    diam = np.zeros(N)
    refl = np.zeros(N)
    inj = np.zeros(N)
    # fixed nose (Tract.init): reflections from the initial nose diameters, noseA[0] = 0.16
    nA = np.zeros(NL)
    for i in range(NL):
        d = 2.0 * (i / NL)
        v = 0.4 + 1.6 * d if d < 1.0 else 0.5 + 1.5 * (2.0 - d)
        if v > 1.9:
            v = 1.9
        nA[i] = v * v
    nrefl = np.zeros(NL)
    for i in range(1, NL):
        nrefl[i] = (nA[i - 1] - nA[i]) / (nA[i - 1] + nA[i])
    slow = np.zeros(N)
    for i in range(N):
        if i < ns:
            slow[i] = 0.6
        elif i >= _PT_TIP:
            slow[i] = 1.0
        else:
            slow[i] = 0.6 + 0.4 * (i - ns) / (_PT_TIP - ns)
    dn_rate = 2.0 * _PT_MOVE / fs
    fric_i = np.zeros(nc)
    fric_rate = 10.0 / fs
    velum = 0.0
    B = 1024                                   # one original 512-sample block = 1024 ticks
    raw_hist = np.zeros(n, dtype=np.bool_)
    lo_hist = np.full(n, -1, dtype=np.int64)
    lo_prev = -1
    env0 = 0.0
    env1 = 0.0
    pos0 = -1
    pos1 = -1
    count = 0
    tdec = 2.0 ** (-200.0 / fs)
    for k in range(n):
        # ---- targets and movement (tractDiameters2 + _moveTowards) ----
        for i in range(N):
            tg = _pt_rest(i, t_idx[k], t_diam[k])
            for c in range(nc):
                tg = _pt_constrict(i, c_idx[c, k], c_diam[c, k], c_act[c, k], tg)
            if k == 0:
                diam[i] = tg
            else:
                dd = tg - diam[i]
                upr = slow[i] * _PT_MOVE / fs
                if dd > upr:
                    dd = upr
                elif dd < -dn_rate:
                    dd = -dn_rate
                diam[i] += dd
        vt = 0.4 if nasal[k] > 0.0 else 0.01
        if k == 0:
            velum = vt
        else:
            dd = vt - velum
            if dd > 0.25 * _PT_MOVE / fs:
                dd = 0.25 * _PT_MOVE / fs
            elif dd < -0.1 * _PT_MOVE / fs:
                dd = -0.1 * _PT_MOVE / fs
            velum += dd
        # ---- reflections (_tractReflections) ----
        for i in range(1, N):
            a0 = diam[i - 1] * diam[i - 1]
            a1 = diam[i] * diam[i]
            if a1 == 0.0:
                refl[i] = 0.999
            else:
                refl[i] = (a0 - a1) / max(1e-12, a0 + a1)
        a0 = diam[ns] * diam[ns]
        a1 = diam[ns + 1] * diam[ns + 1]
        av = velum * velum
        sA = a0 + a1 + av
        rl = (2.0 * a0 - sA) / sA
        rr = (2.0 * a1 - sA) / sA
        rn = (2.0 * av - sA) / sA
        # ---- plosive transients (released closure, one block late) ----
        lo = -1
        for i in range(N):
            if diam[i] <= 0.0:
                lo = i
        raw_hist[k] = (lo_prev > -1) and (lo == -1) and (k > 0)
        lo_hist[k] = lo_prev
        lo_prev = lo
        trig = False
        tpos = -1
        if k >= B:
            trig = raw_hist[k - B] and (velum * velum < 0.05)
            tpos = lo_hist[k - B]
        if trig:
            count += 1
        slot = count % 2
        env0 = env0 * tdec + (0.3 if (trig and slot == 0) else 0.0)
        env1 = env1 * tdec + (0.3 if (trig and slot == 1) else 0.0)
        if trig and slot == 0:
            pos0 = tpos
        if trig and slot == 1:
            pos1 = tpos
        # ---- injections: transients + turbulence at each constriction ----
        for i in range(N):
            inj[i] = 0.0
        if 0 <= pos0 < N:
            inj[pos0] += 0.5 * env0
        if 0 <= pos1 < N:
            inj[pos1] += 0.5 * env1
        for c in range(nc):
            tgt = 1.0 if c_act[c, k] > 0.0 else 0.0
            if k == 0:
                fric_i[c] = tgt
            else:
                dd = tgt - fric_i[c]
                if dd > fric_rate:
                    dd = fric_rate
                elif dd < -fric_rate:
                    dd = -fric_rate
                fric_i[c] += dd
            ci = c_idx[c, k]
            cd = c_diam[c, k]
            if ci >= 2.0 and ci <= N:
                turb = (0.66 * fric[k] * _clip01(fric_i[c]) * nmod[k]
                        * _clip01(8.0 * (0.7 - cd)) * _clip01(30.0 * (cd - 0.3)))
                ix = int(math.floor(ci))
                dl = ci - ix
                if ix + 1 < N:
                    inj[ix + 1] += 0.5 * turb * (1.0 - dl)
                if ix + 2 < N:
                    inj[ix + 2] += 0.5 * turb * dl
        # ---- one waveguide tick (_tractTick) ----
        for i in range(N):
            R[i] += inj[i]
            L[i] += inj[i]
        jR[0] = L[0] * 0.75 + glottal[k]
        for i in range(1, N):
            if i == ns:
                a = R[i - 1]
                b = L[i]
                c = NLf[0]
                jR[i] = rr * b + (1.0 + rr) * (a + c)
                jL[i] = rl * a + (1.0 + rl) * (c + b)
                nJR[0] = rn * c + (1.0 + rn) * (b + a)
            else:
                w = refl[i] * (R[i - 1] + L[i])
                jR[i] = R[i - 1] - w
                jL[i] = L[i] + w
        jL[N] = R[N - 1] * -0.85
        for j in range(1, NL):
            w = nrefl[j] * (NR[j - 1] + NLf[j])
            nJR[j] = NR[j - 1] - w
            nJL[j] = NLf[j] + w
        nJL[NL] = NR[NL - 1] * -0.85
        for i in range(N):
            v = 0.999 * jR[i]
            R[i] = v if abs(v) >= 1e-20 else 0.0
            v = 0.999 * jL[i + 1]
            L[i] = v if abs(v) >= 1e-20 else 0.0
        for j in range(NL):
            v = nJR[j]
            NR[j] = v if abs(v) >= 1e-20 else 0.0
            v = nJL[j + 1]
            NLf[j] = v if abs(v) >= 1e-20 else 0.0
        out[k] = R[N - 1] + NR[NL - 1]
    return out


def pink_trombone(f0_env, tenseness_env=0.6, tongue_index_env=12.9, tongue_diameter_env=2.43,
                  constrictions=(), tick_rate_scale: float = 1.0, dur: float | None = None,
                  rng: np.random.Generator | None = None, *, voiced_env=1.0, wobble=0.0,
                  nasal_env=0.0) -> np.ndarray:
    """Full Pink Trombone (Thapen 2017, Faust port ``pinktrombone.lib`` ``pinkTrombone2``).

    Envelopes are scalars or arrays (any length, stretched to the duration). ``constrictions``
    is a list of dicts ``{"index": 2..44, "diameter": 0..3 (<= 0.3 closes), "active": 0/1}``
    (the original is multi-touch; Faust keeps two). Ranges: tongue_index 12..29 (12.9),
    tongue_diameter 2.05..3.5 (2.43), tenseness 0..1, wobble 0..1, nasal 0/1 (velum).

    ``tick_rate_scale``: waveguide ticks per second = 96000 x scale. "Each waveguide section is
    one tick long, so the tract's acoustic length, and with it every formant, scales with the
    tick rate" (pinktrombone.lib). 1.0 = the original (2 ticks per 48 kHz sample); 0.5 = a
    tract twice as long, every formant an octave lower, f0 unchanged. Implementation: glottis
    and tract run together at the tick rate (1 tick per internal sample) and the output is
    resampled to 48 kHz (the original holds each glottal sample for both ticks; here the
    glottis is evaluated at the tick rate). Block constants follow Faust's
    ``equivSR = tick_rate / 2``."""
    rng = rng or np.random.default_rng(0)
    arrays = [e for e in (f0_env, tenseness_env, tongue_index_env, tongue_diameter_env,
                          voiced_env, nasal_env) if isinstance(e, np.ndarray)]
    n = _n_of(dur, *arrays)
    ratio = Fraction(2.0 * tick_rate_scale).limit_denominator(48)
    if ratio <= 0:
        raise ValueError("tick_rate_scale must be > 0")
    fs = SR * float(ratio)
    m = int(math.ceil(n * float(ratio))) + 8

    def to_int(e):
        return _env(_env(e, n) if not np.isscalar(e) else e, m)

    f0 = to_int(f0_env)
    tens = to_int(tenseness_env)
    voiced = to_int(voiced_env)
    ti = to_int(tongue_index_env)
    td = to_int(tongue_diameter_env)
    nas = to_int(nasal_env)
    nc = len(constrictions)
    c_idx = np.zeros((max(nc, 1), m))
    c_diam = np.full((max(nc, 1), m), 3.0)
    c_act = np.zeros((max(nc, 1), m))
    for c, spec in enumerate(constrictions):
        c_idx[c] = to_int(spec.get("index", 30.0))
        c_diam[c] = to_int(spec.get("diameter", 3.0))
        c_act[c] = to_int(spec.get("active", 1.0))
    seed = int(rng.integers(0, 65536))
    block_time = 512.0 / (fs / 2.0)
    g, nm = _glottis(f0, tens, voiced, wobble, rng, fs, block_time, seed)
    white = 0.5 * rng.uniform(-1.0, 1.0, m) * math.sqrt(fs / 48000.0)
    fric = _resonbp_const(white, 1000.0, 0.5, 2.0, fs)          # _bandpassUnity(1000, 0.5)
    y = _pt_tract_kernel(g, nm, fric, ti, td, c_idx, c_diam, c_act, nas, float(fs)) * 0.25
    if ratio != 1:
        y = _sig.resample_poly(y, ratio.denominator, ratio.numerator)
    return y[:n]


# ==========================================================================================
# Farnell tract comb, articulation, roar, frogs, insects
# ==========================================================================================


def articulation_env(n: int, rise_frac: float = 0.5, offset: float = 0.25) -> np.ndarray:
    """Farnell's soft rise and fall curve (thesis fig. 20.35, book fig. 52.4): a line scans a
    quarter cosine (x pi/2) up over the rise time, then reverses back to 0 over the fall time;
    ``offset`` 0.25 is added "so parameters never sit on zero" (use offset=0 for amplitude)."""
    nr = max(1, int(round(n * rise_frac)))
    nf = max(1, n - nr)
    u = np.concatenate((np.arange(nr) / nr, 1.0 - np.arange(nf) / nf))[:n]
    return np.cos((1.0 - u) * np.pi / 2.0) + offset


def vocal_tract_comb(x: np.ndarray, tract_length_m=0.17, *, n_bands: int = 8, q=4.0,
                     c: float = 340.0) -> np.ndarray:
    """Farnell's parallel band-pass tract comb (thesis fig. 20.34, book fig. 52.3): a bank of
    Pd ``bp~`` "that behave like a comb filter", peaks at the tract's pipe modes.

    Peaks: quarter-wave closed-open pipe F = c/4l (book: l = 0.17 m -> 500 Hz, c = 340),
    odd modes (2k-1) c/4l (the book prints 500, 1000, 1500; physics correction per research 03).
    ``tract_length_m`` may be per-sample (articulation moves the length; coefficients are
    recomputed every sample, avoiding the message-rate clicks Farnell mentions). 8 bands as in
    the figure (8 ``bp~ 1 1``); bands above 0.45 SR are dropped.
    UNSOURCED: default q (the figure's "res" inlet has no printed value)."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    n = len(x)
    ln = np.maximum(_env(tract_length_m, n), 1e-3)
    qa = _env(q, n)
    y = np.zeros(n)
    for k in range(1, n_bands + 1):
        f = (2 * k - 1) * c / (4.0 * ln)
        if f.min() > 0.45 * SR:
            break
        y += _pd_bp(x, np.minimum(f, 0.45 * SR), qa, float(SR))
    return y


def roar(dur: float = 2.5, rng: np.random.Generator | None = None, arc=(30.0, 240.0, 120.0),
         *, rise_frac: float = 0.45, tract_length_m: float = 0.4) -> np.ndarray:
    """Lion-like roar (Farnell Practical 29): "start soft and low with distinct glottal pulses,
    building to a dense roar. Frequency ... starting as low as 30 Hz, building to around 240 Hz,
    and then dropping to around 120 Hz." Flapping source -> tract comb, driven by the
    articulation envelope (fig. 20.36 drives F0 base + excursion, tract and noise from it).

    UNSOURCED: rise_frac (time of the 240 Hz peak), tract_length_m (no lion tract length in the
    sources), and the mappings of the envelope to ripple (1 -> 4), width (2 -> 12) and
    noisiness (0.05 -> 0.45); "distinct pulses early, dense later (more ripple, more noise)" is
    the only guidance."""
    rng = rng or np.random.default_rng(0)
    n = int(round(dur * SR))
    a = articulation_env(n, rise_frac, offset=0.0)              # 0 -> 1 -> 0
    nr = int(round(n * rise_frac))
    f0 = np.empty(n)
    f0[:nr] = arc[0] + (arc[1] - arc[0]) * a[:nr]
    f0[nr:] = arc[2] + (arc[1] - arc[2]) * a[nr:]
    src = flapping_source(f0, ripple_env=1.0 + 3.0 * a, width_env=2.0 + 10.0 * a,
                          noisiness_env=0.05 + 0.4 * a, rng=rng)
    y = vocal_tract_comb(src, tract_length_m, q=3.0 + 3.0 * a)
    y = _pd_hip(y, 20.0, float(SR)) * a
    return y / (np.max(np.abs(y)) + 1e-12) * 0.9


def frog_croak(kind: str = "a", rng: np.random.Generator | None = None,
               dur: float | None = None) -> np.ndarray:
    """Farnell's frog patches (``04-insects/main-insectsall.pd``), research 03 §3.2.

    * ``"a"`` (``pd frog``): v = 1 -> 0 in 400 ms, r = 1 - v, s = (2.3 r)^2, pulse
      1/(1 + (42 wrap(4s - 1))^2) (accelerating 0 -> ~106 Hz); ``vcf~`` F1 600 + 21 s Q80,
      F2 1500 + 42 s Q60, F3 2300 + 119 s Q40; (F1 + F3) x 0.45 + F2 -> hip~ 10 -> x 0.1.
    * ``"b"`` (``pd frog2``): 200 ms, s = 9 (v - 0.5)^2, pulse 1/(1 + (122 wrap(2s - 1))^2)
      (90 -> 0 -> 90 Hz); m = 3 (min(r, .5) - max(r, .9)); F 900 + 40m Q50, 1800 + 90m Q40,
      2900 + 240m Q30; x 0.2.
    * ``"c"`` (``pd frog reson``): noise x o^8, o = cos(2 pi 0.4444 t) -> bp 450 Q220 -> bp 510
      Q220 -> vcf 450 + 70 o^2 Q80 -> x60 x o^2 x cos(2 pi 12 t) -> x 0.4 (a 12 Hz trill);
      ``dur`` default 2.25 s (one o^8 cycle pair). The random gates are left out."""
    rng = rng or np.random.default_rng(0)
    sr = float(SR)
    if kind in ("a", "b"):
        ramp_ms = 400.0 if kind == "a" else 200.0
        tail = 0.15                                   # UNSOURCED: ring-out after the ramp
        n = int(round((ramp_ms / 1000.0 + tail) * SR))
        t = np.arange(n) / SR
        v = np.clip(1.0 - t / (ramp_ms / 1000.0), 0.0, 1.0)
        r = 1.0 - v
        if kind == "a":
            s = (2.3 * r) ** 2
            phi = np.mod(4.0 * s - 1.0, 1.0)
            pulse = 1.0 / (1.0 + (42.0 * phi) ** 2)
            f = [(600 + 21 * s, 80.0), (1500 + 42 * s, 60.0), (2300 + 119 * s, 40.0)]
            gain = 0.1
        else:
            s = 9.0 * (v - 0.5) ** 2
            phi = np.mod(2.0 * s - 1.0, 1.0)
            pulse = 1.0 / (1.0 + (122.0 * phi) ** 2)
            mm = 3.0 * (np.minimum(r, 0.5) - np.maximum(r, 0.9))
            f = [(900 + 40 * mm, 50.0), (1800 + 90 * mm, 40.0), (2900 + 240 * mm, 30.0)]
            gain = 0.2
        # after the ramp the phase stops moving: no more pulses, only the constant level
        # (Pd holds it; hip~ 10 removes it)
        b1 = _vcf(pulse, f[0][0], f[0][1], sr)
        b2 = _vcf(pulse, f[1][0], f[1][1], sr)
        b3 = _vcf(pulse, f[2][0], f[2][1], sr)
        y = _pd_hip((b1 + b3) * 0.45 + b2, 10.0, sr) * gain
        return _edge_fade(y, 2.0)
    n = int(round((dur or 2.25) * SR))
    t = np.arange(n) / SR
    o = np.cos(TWO_PI * 0.4444 * t)
    x = _pd_noise(n, rng) * o ** 8
    x = _bp(_bp(x, 450.0, 220.0), 510.0, 220.0)
    x = _vcf(x, 450.0 + 70.0 * o ** 2, 80.0) * 60.0 * o ** 2 * np.cos(TWO_PI * 12.0 * t) * 0.4
    return _edge_fade(x)


def chitin_clicks(rate: float = 120.0, dur: float = 1.0, rng: np.random.Generator | None = None,
                  *, kind: str = "cicada", res_scale: float = 1.0) -> np.ndarray:
    """Chitin click trains from Farnell's insect patches (research 03 §3.4).

    * ``"cicada"`` (``circada2``): P = phasor(-rate (1 + 190 bp(noise, 120, Q40))) (a wildly
      noise-FM'd saw, rate 120 in the patch) -> hip~ 12000 x2 (only the saw edges survive:
      clicks) -> 4 parallel bp~ 5000/6000/7000/8000 Q30, sum x 0.7 -> x clip(osc(-12 Hz), 0, 1)
      -> bp~ 6000 Q5. "A good chitin click-train template. For creatures, lower the
      resonances": ``res_scale`` multiplies every resonance and the hip~ corner (1 = verbatim).
    * ``"rattler"`` (``pd rattler``): pulse 1/(1 + (6 cos(2 pi (rate/2) t))^2) (narrow pulses at
      ``rate``, 20 Hz in the patch) x bp~(noise, 7000, Q13) x 0.1 -> bp~ 6000 Q16 ->
      x clip(3 osc(0.5 Hz), 0, 1) -> x 0.2.
    The random on/off gates of the patches are left out (the caller places calls)."""
    rng = rng or np.random.default_rng(0)
    n = int(round(dur * SR))
    t = np.arange(n) / SR
    sr = float(SR)
    rs = float(res_scale)
    if kind == "rattler":
        pulse = 1.0 / (1.0 + (6.0 * np.cos(TWO_PI * (rate / 2.0) * t)) ** 2)
        x = pulse * _bp(_pd_noise(n, rng), 7000.0 * rs, 13.0) * 0.1
        x = _bp(x, 6000.0 * rs, 16.0) * np.clip(3.0 * np.cos(TWO_PI * 0.5 * t), 0.0, 1.0) * 0.2
        return _edge_fade(x)
    fm = _bp(_pd_noise(n, rng), 120.0, 40.0)
    p = _phasor(-rate * (1.0 + 190.0 * fm))
    x = _pd_hip(_pd_hip(p, 12000.0 * rs, sr), 12000.0 * rs, sr)
    y = sum(_bp(x, fc * rs, 30.0) for fc in (5000.0, 6000.0, 7000.0, 8000.0)) * 0.7
    y = y * np.clip(np.cos(TWO_PI * -12.0 * t), 0.0, 1.0)
    return _edge_fade(_bp(y, 6000.0 * rs, 5.0))


# ==========================================================================================
# Pd "just say one word", PAF, FOF, octave divider
# ==========================================================================================


def almost_speech(dur: float = 3.0, rng: np.random.Generator | None = None, *,
                  f0_scale: float = 1.0) -> np.ndarray:
    """Pd ``F02.just.say.pd`` ("just say one word", a talking "No"), research 09 B8b, literally:

    * pitch: a metro whose period is re-set each tick to 100 + r^2 ms, r = ``random 35``; each
      tick picks f0 = 150 + ``random 300`` Hz and glides to it (``line``) over that period;
    * syllables: ``phasor~ -4`` squared -> ``lop~ 130`` -> x6 + 0.5 = pulse index;
    * voice: (phasor(f0) - 0.5) x index -> ``clip~ -0.5 0.5`` -> ``cos~`` -> +1 -> x 0.5.
    As each syllable ramp falls the pulse widens, the formant sweeps from bright to dull,
    heard as a diphthong. ``f0_scale`` transposes the pitch generator (1 = the patch)."""
    rng = rng or np.random.default_rng(0)
    n = int(round(dur * SR))
    f0 = np.empty(n)
    pos = 0
    cur = 150.0 + rng.integers(0, 300)
    while pos < n:
        r = int(rng.integers(0, 35))
        period = int(round((100.0 + r * r) * 1e-3 * SR))
        tgt = 150.0 + rng.integers(0, 300)
        seg = min(period, n - pos)
        f0[pos:pos + seg] = cur + (tgt - cur) * np.arange(seg) / period
        cur = cur + (tgt - cur) * seg / period
        pos += seg
    f0 *= f0_scale
    syl = _phasor(np.full(n, -4.0))
    idx = _pd_lop(syl * syl, 130.0, float(SR)) * 6.0 + 0.5
    ph = _phasor(f0)
    y = 0.5 * (np.cos(TWO_PI * np.clip((ph - 0.5) * idx, -0.5, 0.5)) + 1.0)
    y = _pd_hip(y, 5.0, float(SR))                # UNSOURCED: F01's hip~ 5 to drop the DC
    return _edge_fade(y)


def paf(f0_env, fc_env, bw_env, dur: float | None = None, *, gravel: float = 0.0,
        rng: np.random.Generator | None = None) -> np.ndarray:
    """Puckette's phase-aligned formant generator (TTEM p.158-172, Pd F12/F13; research 10 §8.3,
    09 B8f). One phasor drives a Gaussian waveshaper on the half-cycle ``cos(pi ph - pi/2)``
    (index b = bw/f0) and a two-cosine carrier at the centre ratio c = fc/f0 (k = floor c,
    q = c - k), with c sample-and-held at each phase wrap. Output x (1 + b) so the formant peak
    is ~1. Several PAFs on one phasor add coherently (sum them for vowels).

    ``gravel``: Puckette's embellishment "introducing gravel by irregularly modulating the
    phase" - a random carrier phase offset per period. UNSOURCED: its size (gravel = the
    standard deviation in cycles)."""
    n = _n_of(dur, *[e for e in (f0_env, fc_env, bw_env) if isinstance(e, np.ndarray)])
    f0 = _env(f0_env, n)
    fc = _env(fc_env, n)
    bw = _env(bw_env, n)
    ph = _phasor(f0)
    wrap = np.r_[True, np.diff(ph) < 0]
    starts = np.flatnonzero(wrap)
    ratio = fc / f0
    held = np.repeat(ratio[starts], np.diff(np.r_[starts, n]))
    k = np.floor(held)
    q = held - k
    off = 0.0
    if gravel > 0.0:
        rng = rng or np.random.default_rng(0)
        off = np.repeat(rng.normal(0.0, gravel, len(starts)), np.diff(np.r_[starts, n]))
    car = (1.0 - q) * np.cos(TWO_PI * (k * ph + off)) + q * np.cos(TWO_PI * ((k + 1) * ph + off))
    b = bw / f0
    x = b * np.abs(np.sin(np.pi * ph))
    return car * np.exp(-x * x) * (1.0 + b)


@numba.njit(cache=True)
def _fof_grains(starts, n, fc, bw, sw, g, sr):
    """Overlap-added FOF grains (Faust ``pm.fof``: tf2 IR with poles exp(-sw pi/SR),
    exp(-bw pi/SR), b0 = g (1 + a1 + a2), times a sine at fc restarted at the impulse).
    Each impulse starts one grain per formant (fofCycle with unlimited overlap)."""
    y = np.zeros(n)
    K = fc.shape[1]
    for s in starts:
        for k in range(K):
            f = fc[s, k]
            if f <= 0.0 or f >= 0.5 * sr:
                continue
            u1 = math.exp(-sw[s, k] * math.pi / sr)
            u2 = math.exp(-bw[s, k] * math.pi / sr)
            a1 = -(u1 + u2)
            a2 = u1 * u2
            b0 = g[s, k] * (1.0 + a1 + a2)
            e1 = 0.0
            e2 = 0.0
            w = TWO_PI * f / sr
            L = int(7.0 * sr / (math.pi * bw[s, k])) + 2   # until the decay is ~ -60 dB
            for i in range(L):
                idx = s + i
                if idx >= n:
                    break
                e = (b0 if i == 0 else 0.0) - a1 * e1 - a2 * e2
                e2 = e1
                e1 = e
                y[idx] += e * math.sin(w * i)
    return y


def fof_voice(f0_env, vowel="a", voice_type: str = "bass", dur: float | None = None,
              *, gain: float = 1.0) -> np.ndarray:
    """FOF (CHANT) singer, Faust ``SFFormantModelFofCycle`` (Olsen, SMC 2016): an impulse per
    period launches one FOF per formant; formant freq with ``autobendFreq``, gain with
    ``vocalEffort``, skirt width sw = bw x skirtWidthMultiplier(vowel, f0, gender); output x
    corrFactor 75. ``vowel`` as in ``formant_filter`` (name, keyframes or index path); the
    parameters are sampled at each grain start."""
    n = _n_of(dur, f0_env if isinstance(f0_env, np.ndarray) else None)
    f0 = _env(f0_env, n)
    F = _autobend(_param_path(vowel, n, voice_type, "freqs"), f0, voice_type)
    B = _param_path(vowel, n, voice_type, "bws")
    G = _param_path(vowel, n, voice_type, "gains") * _vocal_effort(f0, voice_type)[:, None]
    gender = 1 if voice_type in _FEMALE else 0
    if isinstance(vowel, str):
        vi = np.full(n, float(VOWELS.index(vowel)))
    elif isinstance(vowel, (list, tuple)):
        vi = keyframes([(p[0], VOWELS.index(p[1])) for p in vowel], n)
    else:
        vi = np.clip(_env(vowel, n), 0, 4)
    vi = np.clip(np.round(vi).astype(int), 0, 4)             # ba.selectn: integer index
    mmin = _BW_MULT_MIN[gender * 5 + vi]
    mmax = _BW_MULT_MAX[gender * 5 + vi]
    fmin, fmax = _GENDER_FMIN[gender], _GENDER_FMAX[gender]
    sp = np.clip((f0 - fmin) / (fmax - fmin), 0.0, 1.0)
    SW = B * ((mmax - mmin) * sp + mmin)[:, None]
    ph = _phasor(f0)
    starts = np.flatnonzero(np.r_[True, np.diff(ph) < 0]).astype(np.int64)
    y = _fof_grains(starts, n, np.ascontiguousarray(F), np.ascontiguousarray(B),
                    np.ascontiguousarray(SW), np.ascontiguousarray(G), float(SR))
    return y * 75.0 * gain


def octave_divider(x: np.ndarray, f0_env) -> np.ndarray:
    """Pd ``E03.octave.divider``: ring-modulate by a sinusoid at f0/2 (odd partials 1/2, 3/2 ...,
    an octave down, spectral envelope roughly kept) x 2, plus the original (research 09 B9,
    10 §8.6). f0 is known here, so no pitch tracker (and no 1024-sample alignment delay)."""
    n = len(x)
    f0 = _env(f0_env, n)
    return x + 2.0 * x * np.cos(TWO_PI * _phasor(0.5 * f0))


# ==========================================================================================
# STK SingWave/Modulate, choir, breath
# ==========================================================================================


def stk_modulate(n: int, rng: np.random.Generator, *, vibrato_rate: float = 6.0,
                 vibrato_gain: float = 0.04, random_gain: float = 0.005,
                 phase: float = 0.0) -> np.ndarray:
    """STK ``Modulate``: vibratoGain sin(2 pi 6 t) + OnePole(pole 0.999, gain randomGain) of a
    sample-and-hold white noise renewed every 330 fs/22050 samples (15 ms). Pole converted from
    44.1 kHz. SingWave uses vibrato 0.04, random 0.005; the result multiplies the rate."""
    hold = int(round(330.0 * SR / 22050.0))
    k = n // hold + 2
    sh = np.repeat(rng.uniform(-1.0, 1.0, k), hold)[:n]
    p = rate_radius(0.999, STK_FS)
    noise = _sig.lfilter([random_gain * (1.0 - p)], [1.0, -p], sh)
    return vibrato_gain * np.sin(TWO_PI * vibrato_rate * np.arange(n) / SR + phase) + noise


def singwave(f0_env, dur: float | None = None, rng: np.random.Generator | None = None, *,
             vibrato_gain: float = 0.04, random_gain: float = 0.005,
             vibrato_rate: float = 6.0) -> np.ndarray:
    """STK ``SingWave`` with ``impuls20.raw``: looped one-period pulse = sum of 20 cosine
    harmonics (verified against the rawwave file), rate x (1 + Modulate)."""
    rng = rng or np.random.default_rng(0)
    n = _n_of(dur, f0_env if isinstance(f0_env, np.ndarray) else None)
    f = _env(f0_env, n) * (1.0 + stk_modulate(n, rng, vibrato_rate=vibrato_rate,
                                             vibrato_gain=vibrato_gain, random_gain=random_gain,
                                             phase=rng.uniform(0, TWO_PI)))
    ph = _phasor(f)
    return sum(np.cos(TWO_PI * h * ph) for h in range(1, 21)) / 20.0


def _voice_type_for(f0: float) -> str:
    # UNSOURCED: split points between the table's voice types (standard choral ranges).
    if f0 < 150.0:
        return "bass"
    if f0 < 260.0:
        return "tenor"
    if f0 < 440.0:
        return "alto"
    return "soprano"


def choir(notes, vowel="a", voices_per_note: int = 4, dur: float = 4.0,
          rng: np.random.Generator | None = None, *, vibrato_gain: float = 0.01,
          random_gain: float = 0.005, detune_cents: float = 6.0, attack: float = 0.25,
          release: float = 0.6, voice_type: str | None = None, breath: float = 0.0) -> np.ndarray:
    """Wordless choir: per singer an FOF voice (Faust ``SFFormantModelFofCycle``; Cook: "per-voice
    grain streams with independent jitter") whose f0 carries its own STK SingWave/Modulate
    wobble - 6 Hz vibrato at gain 0.01 (ChucK ``voic-o-form.ck``) plus 15 ms sample-and-hold
    jitter (randomGain 0.005, SingWave default) - with an independent vibrato phase.

    ``notes``: f0 values in Hz. Voice type per note from its pitch unless given.
    ``breath``: level of Cook's breathy "fuzz" (noise through the same formants).
    UNSOURCED: detune_cents (per-singer static detune, gaussian sd), attack/release times
    (power-law shapes, PLAN rule 5), the 1/sqrt(N) sum (uncorrelated voices add as sqrt(sum A^2),
    Farnell thesis p.95)."""
    rng = rng or np.random.default_rng(0)
    n = int(round(dur * SR))
    out = np.zeros(n)
    na = int(attack * SR)
    nr = int(release * SR)
    total = 0
    for f in notes:
        vt = voice_type or _voice_type_for(f)
        for _ in range(voices_per_note):
            det = 2.0 ** (rng.normal(0.0, detune_cents) / 1200.0)
            mod = stk_modulate(n, rng, vibrato_gain=vibrato_gain, random_gain=random_gain,
                               phase=rng.uniform(0, TWO_PI))
            f0 = f * det * (1.0 + mod)
            y = fof_voice(f0, vowel, vt)
            if breath > 0.0:
                y += breath * formant_filter(rng.uniform(-1, 1, n), vowel, vt, f0=f0)
            env = np.ones(n)
            off = int(rng.uniform(0, 0.05) * SR)     # UNSOURCED: singers enter up to 50 ms apart
            a0 = min(n, off + na)
            env[:off] = 0.0
            env[off:a0] = (np.arange(a0 - off) / max(1, na)) ** 2
            env[n - nr:] *= (1.0 - np.arange(nr) / nr) ** 2
            out += y * env
            total += 1
    return out / math.sqrt(max(1, total))


def breath(dur: float = 1.2, in_or_out: str = "out", rng: np.random.Generator | None = None,
           *, formant_scale: float = 1.0, rise_frac: float | None = None) -> np.ndarray:
    """Breath = noise through whispered-vowel formants (Cook p.67-69: whispered vowels keep the
    formant peaks). Formants: STK VoicForm whispered rows ``hah`` (exhale) / ``hoo`` (inhale),
    voice gain 0, noise gain 0.1, F1 at -40 dB; noise level follows Farnell's soft rise/fall
    articulation curve. ``formant_scale`` = STK CC4 tract size (0.9..1.4; lower for big bodies).
    UNSOURCED: which whispered row for in/out, and the default rise fraction (inhale 0.6,
    exhale 0.25)."""
    rng = rng or np.random.default_rng(0)
    n = int(round(dur * SR))
    ph = "hah" if in_or_out == "out" else "hoo"
    rf = rise_frac if rise_frac is not None else (0.25 if in_or_out == "out" else 0.6)
    env = articulation_env(n, rf, offset=0.0)
    noise = rng.uniform(-1.0, 1.0, n) * STK_PHONEMES[ph][0][1] * env
    return formant_filter(noise, ph, "stk", formant_scale=formant_scale)


# ==========================================================================================
# Hybrid: recorded CC0 vocalisations through LPC analysis / resynthesis (SFX_HYBRID.md)
# ==========================================================================================

# recording groups (the fetcher's query words, recordings/PROVENANCE.toml)
REC_GRUNT = ("grunt", "effort grunt", "man grunting", "groan", "pain groan", "grunts", "grunting", "effort",
             "exertion", "lifting grunt", "punch grunt", "hurt", "pain", "groaning")
REC_STRAIN = ("strain", "effort grunt", "man grunting", "grunt", "grunts", "grunting", "exertion",
              "lifting grunt", "effort", "vocal effort")
REC_BREATH_IN = ("inhale breath", "gasp", "breathing", "heavy breathing", "wheeze")
REC_BREATH_OUT = ("exhale", "sigh", "breathing", "heavy breathing", "panting", "wheeze")
REC_VOICED = ("moan", "hum voice", "groan", "scream", "gibberish", "vowel", "pain groan", "moaning",
              "groaning", "humming", "vowels", "sustained vowel", "aah", "ooh", "babble", "mumble",
              "nonsense", "gibberish voice", "vocalization", "yell", "shout", "whimper", "sob")
REC_GROWL = ("growl", "dog growl", "tiger", "bear", "walrus", "pig grunt", "boar", "camel",
             "elephant", "cow moo", "howler monkey", "seal bark", "goat", "wolf howl",
             "lion", "roar", "big cat", "leopard", "jaguar", "cougar", "bear growl", "grizzly", "dog snarl",
             "wolf", "hyena", "pig", "hog", "wild boar", "bull", "cattle", "ox", "yak", "bison", "donkey bray",
             "deer", "elk bugle", "stag roar", "red deer rut", "moose", "camel grunt", "sea lion",
             "elephant seal", "gorilla", "baboon", "cat growl", "cat yowl", "tomcat", "rhino", "hippo",
             "crocodile", "gator", "animal growl", "animal roar", "snarl", "tapir", "koala", "ostrich",
             "emu", "cassowary")
REC_HISS = ("hiss", "cat hiss", "horse snort", "frog", "crow", "raven", "goose hiss", "vulture",
            "bullfrog", "toad", "swan", "heron")


def growl_ok(pr: dict) -> bool:
    """Screen for growl material: voiced enough, a low-to-mid f0, at least half a second active."""
    return pr["voiced_frac"] > 0.12 and 40.0 < pr["f0_median"] < 520.0 and pr["active_s"] > 0.5


def voiced_ok(pr: dict) -> bool:
    return pr["voiced_frac"] > 0.2 and pr["active_s"] > 0.4


def _rms_norm(x: np.ndarray) -> np.ndarray:
    return x / (np.sqrt(np.mean(x * x)) + 1e-12)


def _segment_cache():
    from functools import lru_cache
    from . import analysis

    @lru_cache(maxsize=64)
    def seg(path: str, a: int, b: int):
        x = analysis.load(path)[a:b]
        return analysis.analyse(x)
    return seg


_SEG = None


def recorded_segment(family: str, queries, pick: float, rng: np.random.Generator, *,
                     max_s: float = 3.0, min_s: float = 0.2, pred=None):
    """One analysed vocalisation: recording chosen by ``pick`` in [0, 1) (a fixed identity per
    creature), region and offset by ``rng`` (a new take each time). Returns (row, Track,
    whitened residual)."""
    global _SEG
    from . import analysis
    if _SEG is None:
        _SEG = _segment_cache()
    rows = analysis.rows_where(family, queries, pred) if pred else analysis.rows_matching(family, queries)
    row = rows[int(pick * len(rows)) % len(rows)]
    x = analysis.load(row["path"])
    regs = analysis.active_regions(x, min_s)
    a, b = regs[int(rng.integers(len(regs)))]
    m = int(max_s * SR)
    if b - a > m:
        a = a + int(rng.integers(0, b - a - m))
        b = a + m
    tr, e = _SEG(row["path"], int(a), int(b))
    return row, tr, e


def roughness(f0, rng: np.random.Generator, *, jitter: float = 0.03, shimmer: float = 0.3,
              subharmonic=0.0, sub_rate: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
    """Per-cycle irregularity of a pulse source, returned as (f0 with cycle jitter, amplitude
    multiplier per sample):

    * cycle-to-cycle jitter and shimmer: each glottal cycle gets its own relative period
      deviation (normal, sd ``jitter``) and its own amplitude (log-normal, sd ``shimmer``) -
      Farnell's rippling / flapping mechanism, "the intensity and density of the pulses depend on
      the air velocity over the membrane" (research 03 §3.1): the pulses are not identical;
    * period doubling: alternate cycles scaled by 1 -/+ d, i.e. the carrier ring-modulated at f0/2
      with a DC offset - Puckette's octave divider (TTEM p.126/p.135, research 10 §8.6: "odd
      partials 1/2, 3/2 ... an octave down, spectral envelope roughly kept"). ``subharmonic`` is
      the depth d (scalar or per sample); it is gated on and off in random episodes at about
      ``sub_rate`` per second (UNSOURCED: episode rate and lengths), so the doubling comes and goes
      like a real vocal-fold bifurcation instead of a steady octave.
    Amplitude changes happen at cycle starts only (held per cycle), so no clicks inside a pulse."""
    f = np.asarray(f0, float)
    n = len(f)
    ph = np.cumsum(f) / SR
    cyc = np.floor(ph).astype(np.int64)
    nc = int(cyc[-1]) + 2
    jit = rng.normal(0.0, jitter, nc)
    f2 = f * (1.0 + np.clip(jit[cyc], -3 * jitter, 3 * jitter))
    ph2 = np.cumsum(f2) / SR
    c2 = np.floor(ph2).astype(np.int64)
    nc2 = int(c2[-1]) + 2
    amp = np.exp(rng.normal(0.0, shimmer, nc2))[c2]
    d = np.broadcast_to(np.asarray(subharmonic, float), (n,))
    if np.any(d > 0):
        gate = np.zeros(n)
        t = rng.exponential(1.0 / sub_rate)
        while t < n / SR:
            ln = rng.uniform(0.08, 0.4)
            a, b = int(t * SR), min(n, int((t + ln) * SR))
            gate[a:b] = 1.0
            t += ln + rng.exponential(1.0 / sub_rate)
        gate = _pd_lop(np.ascontiguousarray(gate), 30.0, float(SR))
        alt = np.where(c2 % 2 == 0, 1.0, -1.0)
        amp = amp * (1.0 + d * gate * alt)
    return f2, np.maximum(amp, 0.0)


def micro_pitch(track, n: int, limit: float = 0.12) -> np.ndarray:
    """The recording's own pitch instability as a relative deviation per sample: the voiced f0
    track divided by its 150 ms running median, minus 1 (jitter / wobble of a real larynx),
    0 where unvoiced, clipped to +-limit. Imposed on a synthetic arc it removes the rigid
    "buzz" of a perfectly smooth glottal source."""
    f = np.asarray(track.f0, float)
    if not np.any(f > 0):
        return np.zeros(n)
    v = f > 0
    fi = np.where(v, f, np.interp(np.arange(len(f)), np.flatnonzero(v), f[v]))
    k = max(3, int(0.15 * SR / track.hop) | 1)
    from scipy.ndimage import median_filter
    med = median_filter(fi, size=k, mode="nearest")
    dev = np.where(v, fi / np.maximum(med, 1.0) - 1.0, 0.0)
    dev = np.clip(dev, -limit, limit)
    d = track.per_sample(dev, n)
    return _pd_lop(np.ascontiguousarray(d), 30.0, float(SR))


def syllable_track(n: int, rng: np.random.Generator, queries, *, syl=(0.14, 0.3), family: str = "voice",
                   used: list | None = None, return_bounds: bool = False, lengths=None, pred=None):
    """A tract gesture sequence for "almost speech": each syllable is a short window of a
    different recorded human vocalisation's LPC track (random recording, random place), stretched
    to the syllable length and concatenated; the lattice interpolates k between frames, so the
    formants glide from one recorded vowel shape to the next (Farnell research 03 §3.1: diphthongs
    are neighbours in vowel space, move between them by interpolation). Returns (Track,
    syllable amplitude envelope). UNSOURCED: syllable lengths (Pd F02: 4 Hz syllable phasor)."""
    from . import analysis
    Ks, gs, bounds = [], [], []
    pos = 0
    hop = None
    si = 0
    while pos < n:
        if lengths is not None and si < len(lengths):
            m = int(lengths[si] * SR)
        else:
            m = int(rng.uniform(*syl) * SR)
        si += 1
        m = max(m, int(0.03 * SR))
        m = min(m, n - pos) if n - pos > int(0.08 * SR) else n - pos
        row, tr, _ = recorded_segment(family, queries, float(rng.random()), rng, max_s=0.8, pred=pred)
        if used is not None:
            used.append(row["file"])
        hop = tr.hop
        w = max(4, int(rng.uniform(0.12, 0.3) * SR / tr.hop))
        a = int(rng.integers(0, max(1, tr.frames - w)))
        sub = analysis.Track(tr.K[a:a + w], tr.gain[a:a + w], tr.f0[a:a + w], tr.voicing[a:a + w], tr.hop,
                             (min(w, tr.frames - a)) * tr.hop).stretch(m)
        f_need = int(math.ceil(m / hop))
        Ks.append(sub.K[:f_need])
        gs.append(sub.gain[:f_need] / (sub.gain[:f_need].max() + 1e-12))
        bounds.append((pos, m))
        pos += m
    K = np.concatenate(Ks)
    g = np.concatenate(gs)
    F = int(math.ceil(n / hop)) + 1
    K = np.concatenate([K, np.repeat(K[-1:], max(0, F - len(K)), axis=0)])[:F]
    g = np.concatenate([g, np.repeat(g[-1:], max(0, F - len(g)), axis=0)])[:F]
    track = analysis.Track(K, g, np.zeros(F), np.zeros(F), hop, n)
    env = np.zeros(n)
    for a, m in bounds:
        env[a:a + m] = 0.2 + 0.8 * articulation_env(m, float(rng.uniform(0.2, 0.45)), 0.0)
    if return_bounds:
        return track, env, bounds
    return track, env


# ---- the rune tongue as material for "almost speech" (AUDIO_CHARTER.md §3.10, §8.5) ---------

RUNE_VOWELS = {                       # (F1, F2) Hz, adult male; Farnell/CSLU table (research 03 §3.1)
    "a": (710.0, 1100.0), "e": (550.0, 1770.0), "i": (280.0, 2250.0), "o": (450.0, 800.0),
    "u": (310.0, 870.0),
}   # "o" is not in Farnell's table: UNSOURCED (between U 450/1030 and u 310/870, rounded back vowel)
RUNE_ONSETS = ("m", "n", "l", "r", "s", "sh", "h", "v", "th", "t", "k", "d")
RUNE_CODAS = ("n", "m", "l", "s")
# syllables of the charter lexicon (§3.10.5), scrambled so no word or grammar comes through
RUNE_SYLLABLES = ("thal", "su", "mel", "nu", "shen", "vel", "mor", "lu", "kesh", "i", "ru", "nosh", "hal",
                  "tas", "hael", "nol", "sel", "ve", "lis", "e", "run", "va", "shen", "tal", "ter", "o", "run",
                  "ha", "ke", "li", "de", "sa", "ne", "ken", "ma", "hem", "o", "ma", "na")


def parse_syllable(s: str) -> tuple[str | None, str, str | None]:
    """'thal' -> ('th', 'a', 'l'); 'i' -> (None, 'i', None). Rune-tongue phonotactics (C)V(C)."""
    on = None
    for c in ("sh", "th") + RUNE_ONSETS:
        if s.startswith(c) and len(s) > len(c) and s[len(c)] in RUNE_VOWELS:
            on = c
            break
    rest = s[len(on):] if on else s
    v = rest[0]
    coda = rest[1:] or None
    return on, v, coda if coda in RUNE_CODAS else None


def rune_plan(rng: np.random.Generator, k: int, *, lexical: float = 0.7) -> list[tuple]:
    """k scrambled rune-tongue syllables: mostly syllables of real lexicon words in a random
    order (the player "may half-recognise a syllable of the songs"), the rest random legal
    (C)V(C) syllables. Returns [(onset, vowel, coda), ...]."""
    out = []
    for _ in range(k):
        if rng.random() < lexical:
            out.append(parse_syllable(RUNE_SYLLABLES[int(rng.integers(len(RUNE_SYLLABLES)))]))
        else:
            on = RUNE_ONSETS[int(rng.integers(len(RUNE_ONSETS)))] if rng.random() < 0.8 else None
            v = "aeiou"[int(rng.integers(5))]
            coda = RUNE_CODAS[int(rng.integers(len(RUNE_CODAS)))] if rng.random() < 0.3 else None
            out.append((on, v, coda))
    return out


def vowel_window(n_samples: int, rng: np.random.Generator, queries, target, *, tries: int = 10,
                 used: list | None = None, pred=None, family: str = "voice"):
    """The recorded tract window (of ``tries`` random candidates) whose (F1, F2) lies nearest the
    target vowel (log-frequency distance), stretched to n_samples. Recorded human vowels chosen
    by formants: the vowel is heard, the person is not (the excitation is replaced)."""
    from . import analysis
    best = None
    for _ in range(tries):
        row, tr, _ = recorded_segment(family, queries, float(rng.random()), rng, max_s=0.8, pred=pred)
        w = max(4, int(rng.uniform(0.08, 0.2) * SR / tr.hop))
        a = int(rng.integers(0, max(1, tr.frames - w)))
        K, g = tr.K[a:a + w], tr.gain[a:a + w]
        if len(K) < 2 or g.max() <= 0:
            continue
        f1, f2 = analysis.formant_pair(K, g)
        d = abs(math.log(f1 / target[0])) + abs(math.log(f2 / target[1]))
        if best is None or d < best[0]:
            best = (d, row, tr, a, w)
    _, row, tr, a, w = best
    if used is not None:
        used.append(row["file"])
    sub = analysis.Track(tr.K[a:a + w], tr.gain[a:a + w], tr.f0[a:a + w], tr.voicing[a:a + w], tr.hop,
                         min(w, tr.frames - a) * tr.hop)
    return sub.stretch(n_samples)


def phrase_pitch(n: int, bounds, rng: np.random.Generator, base: float, *, declination_st: float = -4.0,
                 accent_st=(4.0, 7.0), accent_prob: float = 0.6) -> np.ndarray:
    """Intonation of a wordless phrase: a declination line (the phrase's f0 falls ``declination_st``
    semitones from start to end) plus per-syllable pitch accents - each syllable, with probability
    ``accent_prob``, gets a rise-fall hat of 4-7 semitones up or down (up twice as likely), peaking
    40-70 % into the syllable; the others glide gently. Syllable joins are slewed (25 ms), so the
    contour is continuous. UNSOURCED: all numbers (the coordinator's brief: declination plus
    +-4-7 semitone accents per syllable)."""
    t = np.arange(n) / max(1, n - 1)
    st = declination_st * t
    for a, m in bounds:
        if rng.random() < accent_prob:
            mag = rng.uniform(*accent_st) * (1.0 if rng.random() < 0.67 else -1.0)
        else:
            mag = rng.uniform(-1.5, 1.5)
        pk = rng.uniform(0.4, 0.7)
        u = np.arange(m) / max(1, m - 1)
        hat = np.where(u < pk, np.sin(0.5 * np.pi * u / pk), np.cos(0.5 * np.pi * (u - pk) / (1 - pk)))
        st[a:a + m] += mag * hat
    st = _pd_lop(np.ascontiguousarray(np.r_[np.full(SR // 10, st[0]), st]), 6.0, float(SR))[SR // 10:]
    return base * 2.0 ** (st / 12.0)


def hybrid_voice(n: int, rng: np.random.Generator, *, family: str, queries, pick: float,
                 f0=None, source=None, source_mix: float = 0.5, noise_mix: float = 0.0,
                 gate_source: bool = True, alpha: float | None = None, target_f1: float | None = None,
                 morph=None, flatten: float = 0.0, env_follow: float = 1.0,
                 pitch_ratio: float | None = None, used: list | None = None, pred=None,
                 min_warp: float = 0.0, warp_mod=None) -> np.ndarray:
    """Recorded vocal tract, new excitation (Cook ch.8 p.87-92 LPC resynthesis / cross-synthesis;
    Farnell ex. 52.2 "isolate excitation from resonance, then impose ... gestures").

    * filter: LPC track of a recorded segment, time-stretched to n; optionally morphed with a
      second recording (``morph = (family, queries, pick, weight)``, log-envelope interpolation),
      pulled toward its mean (``flatten``, no vowel identity) and warped by ``alpha`` (or so its
      first formant lands on ``target_f1`` = Farnell's c/4l for the body's tract).
    * excitation: the recording's own whitened residual, PSOLA-shifted onto ``f0`` (or by
      ``pitch_ratio``) and stretched to n (Farnell §9.7 synchronous granular), mixed with a
      synthetic ``source`` (glottal / flapping pulses) where the recording is voiced
      (``gate_source``: voiced/unvoiced separation) and with white noise (``noise_mix``, Cook's
      whisper: noise through the same formants).
    * amplitude: the recording's residual-gain contour raised to ``env_follow`` (1 = the recorded
      articulation, 0 = flat; the caller applies its own gesture envelope on top).
    Returns the voice normalised to peak 1. ``used`` collects the recording rows (provenance)."""
    from . import analysis
    row, tr, e = recorded_segment(family, queries, pick, rng, pred=pred)
    if used is not None:
        used.append(row["file"])
    trn = tr.stretch(n)
    if morph is not None:
        mf, mq, mp, mw = morph
        row2, tr2, _ = recorded_segment(mf, mq, mp, rng, pred=pred if mf == family else None)
        if used is not None:
            used.append(row2["file"])
        trn = analysis.morph(trn, tr2, mw)
    if flatten > 0:
        trn = analysis.flatten(trn, flatten)
    if target_f1 is not None:
        alpha = float(np.clip(target_f1 / analysis.first_formant(tr), 0.3, 1.8))
    if alpha is not None and min_warp > 0 and abs(alpha - 1.0) < min_warp:
        alpha = 1.0 + min_warp if alpha >= 1.0 else 1.0 - min_warp     # charter §8.5: >= +-30 %
    if warp_mod is not None:
        # per-sample factor on the warp (e.g. formants moving against the pitch: Corrupted rank)
        fac = np.asarray(warp_mod, float)[:: trn.hop][: trn.frames]
        fac = np.pad(fac, (0, max(0, trn.frames - len(fac))), mode="edge")
        trn = analysis.warp(trn, np.clip((alpha or 1.0) * fac, 0.25, 2.0))
    elif alpha is not None and abs(alpha - 1.0) > 1e-3:
        trn = analysis.warp(trn, alpha)
    voiced = tr.f0[tr.f0 > 0]
    src_f0 = float(np.median(voiced)) if len(voiced) > 3 else None
    f_eff = None
    if f0 is not None:
        f_eff = _env(f0, n) * (1.0 + micro_pitch(trn, n))
    if f_eff is not None and src_f0:
        ratio = np.clip(f_eff / src_f0, 0.25, 4.0)
    else:
        ratio = pitch_ratio or 1.0
    res = _rms_norm(analysis.psola(e, n, ratio=ratio, rng=rng, src_f0=src_f0))
    ex = res
    if callable(source):
        source = source(f_eff if f_eff is not None else np.full(n, 110.0))
    if source is not None and source_mix > 0:
        s = _rms_norm(np.asarray(source, float)[:n])
        if gate_source:
            v = trn.per_sample(np.clip((trn.voicing - 0.3) / 0.4, 0.0, 1.0), n)
            v = _pd_lop(np.ascontiguousarray(v), 20.0, float(SR))
        else:
            v = np.ones(n)
        ex = res * (1.0 - source_mix * v) + s * (source_mix * v)
    if noise_mix > 0:
        ex = ex * (1.0 - noise_mix) + noise_mix * rng.standard_normal(n)
    y = analysis.resynth(ex, trn, gain=False)
    g = trn.per_sample(trn.gain, n)
    g = g / (g.max() + 1e-12)
    y = y * np.power(g, env_follow)
    y = _pd_hip(np.ascontiguousarray(y), 20.0, float(SR))
    return y / (np.max(np.abs(y)) + 1e-12)


# ==========================================================================================
# Syrinx: Smyth & Smith (DAFx-02) valve + trachea waveguide
# ==========================================================================================


@numba.njit(cache=True)
def _syrinx_kernel(ps, w1a, w1b, w2a, w2b, fs, a, h, x0, m, eps, kappa_q, gamma, zs, vol,
                   rho, c, z0, beta, lp_a, dl):
    """Two independent valves (bronchi) into one trachea. Per valve: bronchial pressure p0
    (eq. 1, integrated exactly as a relaxation for stiffness), Bernoulli flow U (eq. 6), two
    membrane modes x_n (eq. 11 with the mass law eq. 10 and closed-wall damping eq. 9),
    driving force F (eq. 8), trapezoid rule (eq. 14). Trachea: fig. 4 - p1 = Z0 U + beta x lower
    rail, upper rail delay, at the top LPF then -1 (open end) into the lower rail; output =
    upper - LPF(upper) (the transmitted part)."""
    n = len(ps)
    out = np.zeros(n)
    up = np.zeros(dl)
    lo = np.zeros(dl)
    ptr = 0
    lp = 0.0
    T = 1.0 / fs
    # state per valve v: p0, U, dU(prev), x1, v1, a1(prev), x2, v2, a2(prev)
    p0 = np.zeros(2)
    U = np.zeros(2)
    dU = np.zeros(2)
    xa = np.full(2, 0.0)
    va = np.zeros(2)
    aa = np.zeros(2)
    xb = np.full(2, 0.0)
    vb = np.zeros(2)
    ab = np.zeros(2)
    p1 = 0.0
    K = rho * c * c / vol
    for k in range(n):
        lower_arrive = lo[ptr]
        u_tot = 0.0
        for v in range(2):
            wa = w1a[k] if v == 0 else w2a[k]
            wb = w1b[k] if v == 0 else w2b[k]
            if wa <= 0.0:
                continue
            x = x0 + xa[v] + xb[v]
            # eq. 1: dp0/dt = (rho c^2/V)((ps - p0)/Zs - U); exact relaxation toward ps - Zs U
            target = ps[k] - zs * U[v]
            p0[v] = target + (p0[v] - target) * math.exp(-K / zs * T)
            # eq. 6 (x > 0), trapezoid; closed valve -> no flow
            if x > 0.0:
                d = (2.0 * a * x / (rho * h)) * (p0[v] - p1 - rho / (8.0 * a * a * x * x) * U[v] * U[v])
                U[v] += (d + dU[v]) * 0.5 * T
                dU[v] = d
                if U[v] < 0.0:
                    U[v] = 0.0
                # eq. 8
                F = a * h * (p0[v] + p1) - 2.0 * rho * U[v] * U[v] * h / (7.0 * (a * x) ** 1.5)
            else:
                U[v] = 0.0
                dU[v] = 0.0
                F = a * h * (p0[v] + p1)
            # eq. 9 (closed: damping x gamma), eq. 10 (mass grows with excursion)
            g = gamma if x <= 0.0 else 1.0
            mm = m * (1.0 + eps * ((x - x0) / h) ** 2)
            ka = g * wa / (2.0 * kappa_q)
            kb = g * wb / (2.0 * kappa_q)
            # eq. 11 per mode (alpha = 1), trapezoid on velocity and displacement
            acc_a = F / mm - 2.0 * ka * va[v] - wa * wa * xa[v]
            acc_b = F / mm - 2.0 * kb * vb[v] - wb * wb * xb[v]
            nva = va[v] + (acc_a + aa[v]) * 0.5 * T
            nvb = vb[v] + (acc_b + ab[v]) * 0.5 * T
            xa[v] += (nva + va[v]) * 0.5 * T
            xb[v] += (nvb + vb[v]) * 0.5 * T
            va[v] = nva
            vb[v] = nvb
            aa[v] = acc_a
            ab[v] = acc_b
            u_tot += U[v]
        p1 = z0 * u_tot + beta * lower_arrive
        top = up[ptr]
        up[ptr] = p1
        lp = lp + lp_a * (top - lp)
        lo[ptr] = -lp
        out[k] = top - lp
        ptr += 1
        if ptr >= dl:
            ptr = 0
    return out


def syrinx(pressure_env, f_env=(500.0, 505.0), dur: float | None = None, *,
           mode_ratio: float = 1.6, trachea_length_m: float = 0.07,
           trachea_radius_m: float = 0.0035, second_valve: bool = True,
           oversample: int = 4) -> np.ndarray:
    """Pressure-driven syrinx valve(s) feeding a trachea waveguide (Smyth & Smith DAFx-02,
    after Fletcher 1988). ``pressure_env``: air-sac pressure p_s in Pa (Fig. 5 shows ~1-2 kPa at
    the trachea base). ``f_env``: membrane mode-1 frequency of each valve (two valves at close
    frequencies beat: the "two-voice" syrinx); mode 2 at ``mode_ratio`` x.

    Sourced: equations 1, 5, 6, 8-11, 14 and fig. 4 (decoded from the mis-encoded PDF font:
    F = a h (p0 + p1) - 2 rho U^2 h / (7 (a x)^1.5); dU/dt = (2 a x / rho h)(p0 - p1 -
    rho U^2 / (8 a^2 x^2)); gamma in 10..100; mass m (1 + eps ((x - x0)/h)^2); wall loss
    alpha = 1.2e-5 sqrt(omega)/a); raven trachea 70 mm x 3.5 mm radius (Kahrs & Avanzini);
    rho = 1.2, c = 343.
    SIGN CHECK (flagged): the force sign convention (pressure pushes the membrane open, the
    Bernoulli term pulls it shut) and p1 = Z0 U + beta x lower are my reading of eq. 8 and fig. 4.
    The p0 equation is integrated as an exact relaxation, not the paper's trapezoid, because with
    any plausible bronchus volume it is too stiff at audio rates (deviation).
    UNSOURCED (the paper prints no parameter values): a, h, x0, membrane mass, eps, mode Q,
    gamma = 30, Zs, bronchus volume, mode_ratio, the beak LPF corner, oversample. The set used
    was picked by a parameter sweep as the one whose f0 follows the membrane frequency
    (150-900 Hz at p_s = 3 kPa); below ~1 kPa or with low Q it does not self-oscillate."""
    n = _n_of(dur, pressure_env if isinstance(pressure_env, np.ndarray) else None)
    os_ = int(oversample)
    fs = SR * os_
    m_ = n * os_
    ps = _env(_env(pressure_env, n), m_)
    fa = _env(_env(f_env[0], n), m_)
    fb = _env(_env(f_env[1] if len(f_env) > 1 else f_env[0], n), m_)
    w1a = TWO_PI * fa
    w1b = w1a * mode_ratio
    w2a = TWO_PI * fb if second_valve else np.zeros(m_)
    w2b = w2a * mode_ratio
    rho, c = 1.2, 343.0
    a = trachea_radius_m                     # UNSOURCED: valve half-width = trachea radius
    h = 0.003                                # UNSOURCED: membrane half-length 3 mm
    x0 = 0.0005                              # UNSOURCED: rest opening 0.5 mm
    m = 1000.0 * (2 * a) * (2 * h) * 1e-3    # UNSOURCED: 1 mm thick, density 1000 (K&A)
    eps = 1.0                                # UNSOURCED
    q = 30.0                                 # UNSOURCED: modal Q
    gamma = 30.0                             # within the paper's 10..100
    zs = 2.0e6                               # UNSOURCED: air-sac impedance
    vol = math.pi * 0.0015 ** 2 * 0.010      # oil-bird bronchus 10 mm x 1.5 mm radius (K&A)
    z0 = rho * c / (math.pi * a * a)         # characteristic impedance of the trachea
    dl = max(2, int(round(trachea_length_m / c * fs)))
    omega = TWO_PI * 1000.0                  # UNSOURCED: wall loss evaluated at 1 kHz
    alpha = 1.2e-5 * math.sqrt(omega) / a
    beta = math.exp(-alpha * 2.0 * trachea_length_m)
    fc_beak = 4000.0                         # UNSOURCED: beak LPF corner
    lp_a = 1.0 - math.exp(-TWO_PI * fc_beak / fs)
    y = _syrinx_kernel(ps, w1a, w1b, w2a, w2b, float(fs), a, h, x0, m, eps, q, gamma, zs, vol,
                       rho, c, z0, beta, lp_a, dl)
    y = _sig.resample_poly(y, 1, os_)[:n]
    return _pd_hip(y, 20.0, float(SR)) * 1e-3          # output pressure in kPa


# ==========================================================================================
# Self-test
# ==========================================================================================


def _lpc_formants(x: np.ndarray, order: int = 12, fs_an: int = 11025, fmax: float = 5000.0,
                  bw_max: float = 600.0) -> list[float]:
    """Formants by autocorrelation LPC on a Hamming-windowed signal resampled to fs_an."""
    y = _sig.resample_poly(x, fs_an, SR)
    y = np.append(y[0], y[1:] - 0.63 * y[:-1])            # mild pre-emphasis
    y = y * np.hamming(len(y))
    r = np.correlate(y, y, "full")[len(y) - 1:len(y) + order]
    from scipy.linalg import solve_toeplitz
    a = solve_toeplitz(r[:order], -r[1:order + 1])
    roots = np.roots(np.r_[1.0, a])
    roots = roots[np.imag(roots) > 0]
    f = np.angle(roots) * fs_an / TWO_PI
    bw = -np.log(np.abs(roots)) * fs_an / np.pi
    keep = (f > 90) & (f < fmax) & (bw < bw_max)
    return sorted(f[keep].tolist())


def _resp_peaks(y_ir: np.ndarray, fs: float = SR, fmax: float = 5000.0,
                prom: float = 1.0) -> list[float]:
    """Peaks of the magnitude response from an impulse response (exact filter formants)."""
    nfft = 1 << 18
    H = 20 * np.log10(np.abs(np.fft.rfft(y_ir, nfft)) + 1e-20)
    f = np.fft.rfftfreq(nfft, 1 / fs)
    pk, _ = _sig.find_peaks(H, prominence=prom)
    return [float(f[i]) for i in pk if 60 < f[i] < fmax]


def _harmonic_peaks(x: np.ndarray, f0: float, fmax: float = 4000.0) -> list[float]:
    """Formant estimate of a voiced sound: harmonic amplitudes (dB), local maxima over the
    harmonic index, parabolic refinement between harmonics."""
    nfft = 1 << 20
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)), nfft))
    f = np.fft.rfftfreq(nfft, 1 / SR)
    hs = np.arange(1, int(fmax / f0))
    amp = []
    for h in hs:
        m = (f > (h - 0.3) * f0) & (f < (h + 0.3) * f0)
        amp.append(20 * np.log10(X[m].max() + 1e-20))
    a = np.array(amp)
    out = []
    for i in range(1, len(a) - 1):
        if a[i] > a[i - 1] and a[i] >= a[i + 1]:
            den = a[i - 1] - 2 * a[i] + a[i + 1]
            d = 0.5 * (a[i - 1] - a[i + 1]) / den if den != 0 else 0.0
            out.append(float((hs[i] + d) * f0))
    return out


def _pt_ir(tongue_index: float, tongue_diameter: float, scale: float = 1.0,
           seconds_: float = 0.5, nasal: float = 0.0) -> tuple[np.ndarray, float]:
    """Impulse response of the Pink Trombone tract (constant shape, no turbulence: LTI, as the
    Faust port notes), at the tick rate. Returns (ir, tick_rate)."""
    fs = 96000.0 * scale
    m = int(seconds_ * fs)
    g = np.zeros(m)
    g[0] = 1.0
    z = np.zeros(m)
    y = _pt_tract_kernel(g, z, z, np.full(m, tongue_index), np.full(m, tongue_diameter),
                         np.zeros((1, m)), np.full((1, m), 3.0), np.zeros((1, m)),
                         np.full(m, nasal), fs)
    return y, fs


def _nearest(est, tab):
    return [min(est, key=lambda e: abs(e - t)) if len(est) else float("nan") for t in tab]


def _smoothed_peaks(x: np.ndarray, fmax: float = 5000.0, smooth_hz: float = 120.0) -> list[float]:
    f, p = _sig.welch(x, SR, nperseg=8192)
    db = 10 * np.log10(p + 1e-20)
    w = max(3, int(smooth_hz / (f[1] - f[0])))
    k = np.hanning(w)
    s = np.convolve(db, k / k.sum(), "same")
    pk, _ = _sig.find_peaks(s, prominence=3.0)
    return [float(f[i]) for i in pk if 90 < f[i] < fmax]


def _f0_track(x: np.ndarray, fmin: float = 25.0, fmax: float = 600.0, hop: float = 0.02,
              win: float = 0.06) -> tuple[np.ndarray, np.ndarray]:
    """YIN f0 per frame (de Cheveigne & Kawahara 2002: cumulative-mean-normalised difference,
    threshold 0.1, parabolic refinement), via FFT correlation. Returns (times, f0)."""
    nw = int(win * SR)
    nh = int(hop * SR)
    lmin, lmax = max(2, int(SR / fmax)), int(SR / fmin)
    nfft = 1 << int(math.ceil(math.log2(2 * nw + lmax)))
    ts, fs_ = [], []
    for s in range(0, len(x) - nw - lmax, nh):
        seg = x[s:s + nw + lmax]
        a = seg[:nw]
        r = np.fft.irfft(np.conj(np.fft.rfft(a, nfft)) * np.fft.rfft(seg, nfft), nfft)[:lmax + 1]
        c2 = np.concatenate(([0.0], np.cumsum(seg * seg)))
        e0 = c2[nw]
        et = c2[np.arange(lmax + 1) + nw] - c2[np.arange(lmax + 1)]
        d = e0 + et - 2.0 * r
        d[0] = 0.0
        cm = d[1:] * np.arange(1, lmax + 1) / (np.cumsum(d[1:]) + 1e-20)
        cm = np.r_[1.0, cm]
        cand = np.flatnonzero(cm[lmin:] < 0.1)
        i = (lmin + cand[0]) if len(cand) else lmin + int(np.argmin(cm[lmin:]))
        while i + 1 <= lmax and cm[i + 1] < cm[i]:
            i += 1
        if lmin < i < lmax:
            den = cm[i - 1] - 2 * cm[i] + cm[i + 1]
            dd = 0.5 * (cm[i - 1] - cm[i + 1]) / den if den != 0 else 0.0
        else:
            dd = 0.0
        ts.append((s + nw / 2) / SR)
        fs_.append(SR / (i + dd))
    return np.array(ts), np.array(fs_)


def _cents(a, b):
    return 1200.0 * np.log2(np.asarray(a) / np.asarray(b))


def _main() -> None:  # pragma: no cover - self-test
    import soundfile as sf
    from .core import DEV_DIR

    out = DEV_DIR / "voice"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    report = []

    def save(name, y, norm=True):
        y = np.asarray(y, float)
        assert np.all(np.isfinite(y)), name + " has NaN/inf"
        pk = float(np.max(np.abs(y)))
        if norm and pk > 0:
            y = y / pk * 0.5
        sf.write(out / f"{name}.wav", y.astype(np.float32), SR, subtype="PCM_24")
        return pk

    # ---- 1. formant filters vs tables -------------------------------------------------------
    imp = np.zeros(SR)
    imp[0] = 1.0
    report.append("== formant_filter: response peaks (impulse response) vs table F1-F3 ==")
    worst = 0.0
    for vt in VOICE_TYPES:
        for vw in VOWELS:
            tab = formant_table(vt, vw)["freqs"][:3]
            near = _nearest(_resp_peaks(formant_filter(imp, vw, vt), prom=0.5), tab)
            err = [100 * (e - t) / t for e, t in zip(near, tab)]
            worst = max(worst, max(abs(e) for e in err))
    report.append(f"all 25 Faust/Csound entries: worst |F1-F3 peak error| {worst:.1f}%")
    for vt, vw in (("bass", "a"), ("bass", "u"), ("tenor", "i"), ("soprano", "o")):
        tab = formant_table(vt, vw)["freqs"][:3]
        near = _nearest(_resp_peaks(formant_filter(imp, vw, vt), prom=0.5), tab)
        report.append(f"  {vt:8s} {vw}: table {tab.astype(int).tolist()} peaks {[int(e) for e in near]}")
    for label, vt, names, keyn in (("STK FormSwep", "stk", ("ahh", "eee", "ooo", "mmm"), None),
                                   ("Farnell bp~ Q34", "farnell", ("@", "i:", "A", "u"), None)):
        rows = []
        for ph in names:
            tab = formant_table(vt, ph)["freqs"][:3]
            near = _nearest(_resp_peaks(formant_filter(imp, ph, vt), prom=0.3), tab)
            rows.append(f"{ph} {tab.astype(int).tolist()}->{[int(e) for e in near]}")
        report.append(f"{label}: " + "; ".join(rows))

    report.append("== rendered vowels: noise-excited (whisper, Welch peaks) and voiced (harmonic envelope) ==")
    for vt, vw, f0 in (("bass", "a", 98.0), ("bass", "o", 98.0), ("tenor", "e", 131.0), ("tenor", "i", 131.0)):
        tab = formant_table(vt, vw)["freqs"][:3]
        nz = formant_filter(rng.uniform(-1, 1, SR * 4), vw, vt)
        nn = _nearest(_smoothed_peaks(nz, 4000, 40), tab)
        src = lf_pulse(f0, 0.6, 3.0, rng, wobble=0.0)
        v = formant_filter(src, vw, vt)
        hv = _nearest(_harmonic_peaks(v[SR:], f0), tab)
        save(f"vowel_{vt}_{vw}_f0_{int(f0)}", v)
        report.append(f"{vt:6s} {vw}: table {tab.astype(int).tolist()} whisper {[int(e) for e in nn]} "
                      f"({', '.join(f'{100*(e-t)/t:+.1f}%' for e, t in zip(nn, tab))}) voiced {[int(e) for e in hv]} "
                      f"({', '.join(f'{100*(e-t)/t:+.1f}%' for e, t in zip(hv, tab))})")
    src = lf_pulse(keyframes([(0, 110), (3, 90)], 3 * SR), 0.6, rng=rng)
    morph = formant_filter(src, [(0.2, "a"), (1.4, "i"), (2.6, "u")], "bass", f0=110.0)
    save("formant_morph_a_i_u", morph)
    jump = np.max(np.abs(np.diff(morph))) / (np.max(np.abs(morph)) + 1e-12)
    report.append(f"morph a->i->u: max |diff|/peak {jump:.3f} (no steps), finite {bool(np.all(np.isfinite(morph)))}")

    # ---- 2. f0 tracking ---------------------------------------------------------------------
    report.append("== f0 tracking (YIN; median / 90th percentile |error| in cents) ==")
    glide = keyframes([(0, 100), (2, 200)], 2 * SR)
    tgrid = np.arange(2 * SR) / SR

    def track(name, y, ref):
        t, f = _f0_track(y, 60, 400)
        refv = np.interp(t, tgrid[:len(ref)], ref)
        e = np.abs(_cents(f, refv))[2:-2]
        report.append(f"{name:34s}: {np.median(e):5.1f} / {np.percentile(e, 90):5.1f} c (n={len(e)})")
        return f

    track("raised_cosine_pulse w=3", raised_cosine_pulse(glide, 3.0), glide)
    track("flapping_source ripple 1", flapping_source(glide, 1.0, 4.0, 0.0, rng=rng), glide)
    # lf_pulse: the glottis adds 6 Hz + simplex vibrato; rebuild the expected curve from the seed
    r_a = np.random.default_rng(11)
    r_b = np.random.default_rng(11)
    y_lf = lf_pulse(glide, 0.6, rng=r_a)
    seed = int(r_b.integers(0, 65536))
    nn = len(glide)
    vib = (0.005 * np.sin(TWO_PI * 6.0 * np.arange(nn) / SR) + 0.02 * _simplex_lf(seed, 4.07, nn, SR)
           + 0.04 * _simplex_lf(seed, 2.15, nn, SR))
    track("lf_pulse vs target (raw)", y_lf, glide)
    track("lf_pulse vs target x (1 + vibrato)", y_lf, glide * (1 + vib))
    track("singwave, modulation off", singwave(150.0, 2.0, rng, vibrato_gain=0.0, random_gain=0.0),
          np.full(2 * SR, 150.0))
    t, f = _f0_track(singwave(150.0, 2.0, rng), 60, 400)
    c = _cents(f, 150.0)[2:-2]
    report.append(f"singwave default vibrato 0.04: measured peak-to-peak {np.percentile(c, 98) - np.percentile(c, 2):.0f} c "
                  f"(expected ~{2 * 1200 * math.log2(1.04):.0f} c)")
    r = roar(2.5, rng)
    save("roar", r)
    t, f = _f0_track(r, 25, 400)
    n = len(r)
    a = articulation_env(n, 0.45, 0.0)
    nr = int(round(n * 0.45))
    exp_f0 = np.where(np.arange(n) < nr, 30 + 210 * a, 120 + 120 * a)
    for tt in (0.3, 0.8, 1.125, 1.6, 2.2):
        report.append(f"  roar t={tt:.2f} s: measured {f[np.argmin(np.abs(t - tt))]:6.1f} Hz, "
                      f"arc {exp_f0[int(tt * SR)]:6.1f} Hz")

    # ---- 3. Pink Trombone -------------------------------------------------------------------
    report.append("== Pink Trombone ==")
    import time
    t0 = time.time()
    pt_a = pink_trombone(120.0, 0.6, 12.9, 2.43, dur=2.0, rng=rng)
    dt = time.time() - t0
    t0 = time.time()
    pink_trombone(120.0, 0.6, 12.9, 2.43, dur=2.0, rng=rng)
    dt2 = time.time() - t0
    report.append(f"render 2 s: {dt:.2f} s (incl. JIT) / {dt2:.2f} s warm; peak {np.max(np.abs(pt_a)):.3f}")
    shapes = (("rest (12.9, 2.43)", 12.9, 2.43), ("front-high (27, 2.2) ~/i/", 27.0, 2.2),
              ("back-low (13, 3.3)", 13.0, 3.3), ("back-high (20, 2.05) ~/u/", 20.0, 2.05))
    base = {}
    for name, ti_, td_ in shapes:
        ir, fsr = _pt_ir(ti_, td_)
        pk = _resp_peaks(ir, fsr, 4500, 3.0)[:4]
        base[name] = pk
        report.append(f"tract IR formants {name:26s}: {[int(p) for p in pk]}")
    for sc in (0.5, 0.25):
        ir, fsr = _pt_ir(12.9, 2.43, sc)
        pk = _resp_peaks(ir, fsr, 4500, 3.0)[:4]
        b0 = base["rest (12.9, 2.43)"]
        report.append(f"tick_rate_scale {sc}: rest formants {[int(p) for p in pk]}, ratio to 1.0 "
                      f"{[round(p / q, 3) for p, q in zip(pk, b0)]} (expect {sc})")
    irn, fsr = _pt_ir(12.9, 2.43, nasal=1.0)
    report.append(f"velum open (nasal) rest formants {[int(p) for p in _resp_peaks(irn, fsr, 4500, 3.0)[:5]]}")
    fr = _harmonic_peaks(pt_a[SR // 2:], 120.0 * 1.0)
    report.append(f"rendered rest vowel (f0 120) harmonic-envelope peaks {[int(x) for x in fr[:4]]}")
    pt_i = pink_trombone(120.0, 0.6, 27.0, 2.2, dur=2.0, rng=rng)
    pt_big = pink_trombone(120.0, 0.6, 12.9, 2.43, tick_rate_scale=0.5, dur=2.0, rng=rng)
    t, f = _f0_track(pt_big, 60, 400)
    report.append(f"tick_rate_scale 0.5 rendered f0 median {np.median(f[5:]):.1f} Hz (f0 set 120: pitch unchanged)")
    save("pt_rest_a", pt_a)
    save("pt_front_i", pt_i)
    save("pt_half_rate_giant", pt_big)
    # stability: 12 s of moving articulation, closures/releases, nasal toggles, voicing gaps
    n12 = 12 * SR
    tt = np.arange(n12) / SR
    ti = 20.5 + 8.5 * np.sin(TWO_PI * 0.37 * tt)
    tdm = 2.8 + 0.7 * np.sin(TWO_PI * 0.23 * tt + 1)
    cact = (np.sin(TWO_PI * 0.9 * tt) > 0.6).astype(float)
    cdia = np.where(cact > 0, 0.0, 3.0) + 0.45 * (np.sin(TWO_PI * 0.31 * tt) > 0)
    nas = (np.sin(TWO_PI * 0.13 * tt) > 0.5).astype(float)
    voiced = (np.sin(TWO_PI * 0.2 * tt) > -0.8).astype(float)
    f0s = 80 + 60 * (0.5 + 0.5 * np.sin(TWO_PI * 0.11 * tt))
    for sc in (1.0, 0.4):
        ptl = pink_trombone(f0s, 0.2 + 0.7 * (0.5 + 0.5 * np.sin(TWO_PI * 0.07 * tt)), ti, tdm,
                            [{"index": 38.0 + 3 * np.sin(TWO_PI * 0.5 * tt), "diameter": cdia, "active": cact},
                             {"index": 12.0, "diameter": 0.5, "active": (tt > 6).astype(float)}],
                            tick_rate_scale=sc, rng=rng, voiced_env=voiced, wobble=1.0, nasal_env=nas)
        rms = [float(np.sqrt(np.mean(ptl[i * SR:(i + 1) * SR] ** 2))) for i in range(12)]
        report.append(f"12 s stress run scale {sc}: finite {bool(np.all(np.isfinite(ptl)))}, peak {np.max(np.abs(ptl)):.3f}, "
                      f"per-second RMS min/max {min(rms):.4f}/{max(rms):.4f}, last/first-half RMS "
                      f"{np.sqrt(np.mean(ptl[6 * SR:] ** 2)) / np.sqrt(np.mean(ptl[:6 * SR] ** 2)):.2f}, DC {np.mean(ptl):+.1e}")
        save(f"pt_stress_12s_scale_{sc}", ptl)
    for scale in (0.35, 0.7, 1.3):
        yy = pink_trombone(90.0, 0.5, 20.0, 2.6, tick_rate_scale=scale, dur=1.5, rng=rng, wobble=1.0)
        report.append(f"tick_rate_scale {scale}: finite {bool(np.all(np.isfinite(yy)))}, peak {np.max(np.abs(yy)):.3f}")
        save(f"pt_scale_{scale}", yy)
    n3 = 3 * SR
    lip = keyframes([(0, 3), (0.3, 3), (0.35, 0.0), (0.5, 0.0), (0.55, 3), (1.2, 3), (1.25, 0), (1.4, 0),
                     (1.45, 3), (3, 3)], n3)
    pt_speech = pink_trombone(keyframes([(0, 110), (3, 85)], n3), 0.55,
                              keyframes([(0, 13), (1, 25), (2, 15), (3, 22)], n3),
                              keyframes([(0, 2.4), (1, 2.2), (2, 2.8), (3, 2.3)], n3),
                              [{"index": 41.0, "diameter": lip, "active": 1.0}], rng=rng, wobble=0.5)
    save("pt_gesture", pt_speech)
    report.append(f"closure/release gesture: finite {bool(np.all(np.isfinite(pt_speech)))}, peak {np.max(np.abs(pt_speech)):.3f}")

    # ---- 4. remaining models -----------------------------------------------------------------
    report.append("== other models (finite, peak before normalising) ==")
    items = {
        "almost_speech": almost_speech(3.0, rng),
        "frog_a": frog_croak("a", rng), "frog_b": frog_croak("b", rng), "frog_c": frog_croak("c", rng),
        "chitin_cicada": chitin_clicks(120, 1.5, rng),
        "chitin_cicada_low": chitin_clicks(60, 1.5, rng, res_scale=0.35),
        "chitin_rattler": chitin_clicks(20, 2.0, rng, kind="rattler"),
        "choir_a": choir([110, 165, 220, 277], "a", 4, 5.0, rng),
        "choir_morph": choir([98, 147], [(0.5, "o"), (2.5, "a"), (4.5, "u")], 5, 5.0, rng, breath=0.02),
        "breath_out": breath(1.4, "out", rng), "breath_in": breath(1.2, "in", rng),
        "breath_big": breath(1.8, "out", rng, formant_scale=0.6),
        "fof_bass_a": fof_voice(110.0, "a", "bass", 2.0),
        "paf_vowel": sum(paf(keyframes([(0, 100), (2, 80)], 2 * SR), fc, bw, gravel=0.01, rng=rng) * g
                         for fc, bw, g in ((600, 60, 1.0), (1040, 70, 0.45), (2250, 110, 0.35))),
        "octave_down_pt": octave_divider(pt_a, 120.0),
        "comb_beast": vocal_tract_comb(flapping_source(55.0, 3.0, 12.0, 0.3, dur=2.0, rng=rng), 0.6),
    }
    for name, y in items.items():
        pk = save(name, y)
        report.append(f"{name:18s} finite {bool(np.all(np.isfinite(y)))} len {len(y) / SR:.2f} s peak {pk:.3g}")
    t, f = _f0_track(items["fof_bass_a"], 60, 400)
    report.append(f"fof_bass_a f0 median {np.median(f):.1f} Hz (110)")
    ch = items["choir_a"]
    report.append(f"choir_a smoothed-spectrum peaks {[int(p) for p in _smoothed_peaks(ch[SR:4 * SR])[:5]]}")
    fr = items["frog_a"]
    report.append(f"frog_a spectral peaks {[int(p) for p in _smoothed_peaks(fr, 4000, 60)[:4]]} (600-711/1500-1722/2300-2929)")
    cc = items["chitin_cicada"]
    report.append(f"cicada peaks {[int(p) for p in _smoothed_peaks(cc, 12000, 200)[:5]]} (5-8 kHz bank)")

    # syrinx
    ps = keyframes([(0, 0), (0.05, 3000), (1.2, 3000), (1.3, 0)], int(1.4 * SR))
    sy = syrinx(ps, (500.0, 503.0), dur=1.4)
    ok = bool(np.all(np.isfinite(sy)))
    pk = save("syrinx_two_valves", sy) if ok else float("nan")
    t, f = _f0_track(sy, 100, 1200) if ok else (np.array([]), np.array([]))
    rmsv = float(np.sqrt(np.mean(sy[int(0.3 * SR):int(1.1 * SR)] ** 2))) if ok else float("nan")
    report.append(f"syrinx: finite {ok}, raw peak {pk:.3g}, steady RMS {rmsv:.3g}, "
                  f"f0 median {np.median(f) if len(f) else float('nan'):.0f} Hz (membrane 500 Hz), "
                  f"peaks {[int(p) for p in _smoothed_peaks(sy, 4000, 100)[:4]] if ok else []}")
    rows = []
    for fm in (200.0, 350.0, 600.0, 850.0):
        s1 = syrinx(3000.0, (fm,), dur=0.8, second_valve=False)
        t, f = _f0_track(s1[int(0.3 * SR):], 100, 1500)
        rows.append(f"{int(fm)}->{np.median(f):.0f}")
    report.append("syrinx single valve, membrane Hz -> sounding f0: " + ", ".join(rows))
    quiet = syrinx(500.0, (500.0,), dur=0.8, second_valve=False)
    report.append(f"syrinx at p_s 500 Pa: RMS {np.sqrt(np.mean(quiet[int(0.3 * SR):] ** 2)):.2e} kPa "
                  f"(below the oscillation threshold)")
    print("\n".join(report))
    (out / "report.txt").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    _main()
