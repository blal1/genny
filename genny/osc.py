"""Oscillators and noise. All accept a scalar frequency or a per-sample frequency array (for sweeps)."""
from __future__ import annotations

import math

import numba
import numpy as np
from numpy.polynomial import Polynomial as _Poly
from scipy.signal import lfilter

from .core import DEFAULT_SR, samples

WAVES = ["sine", "saw", "square", "pulse", "triangle", "supersaw", "fm", "noise", "pink", "brown", "organ", "wavefold", "gendyn",
         "jp8000", "cz", "gbuzz", "dpw_saw", "dpw_triangle", "dpw_square", "blit", "blit_saw", "blit_square", "sync",
         "pitched_noise", "held_noise", "vco2_saw", "vco2_square", "vco2_triangle", "vco2_pulse", "squine", "fm_tree"]


def _freq_array(freq, n: int) -> np.ndarray:
    if np.isscalar(freq):
        return np.full(n, float(freq))
    f = np.asarray(freq, dtype=np.float64)
    if f.shape[0] != n:
        f = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, f.shape[0]), f)
    return f


def phase(freq, n: int, sr: int = DEFAULT_SR, phase0: float = 0.0) -> np.ndarray:
    """Phase in cycles (0..1 wrapping) integrating a possibly time-varying frequency."""
    f = _freq_array(freq, n)
    ph = np.cumsum(f / sr) + phase0
    return ph


def sine(freq, n: int, sr: int = DEFAULT_SR, phase0: float = 0.0) -> np.ndarray:
    return np.sin(2 * np.pi * phase(freq, n, sr, phase0))


def _polyblep(t: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """PolyBLEP residual for discontinuities at t=0 (t in 0..1)."""
    out = np.zeros_like(t)
    m1 = t < dt
    tt = t[m1] / dt[m1]
    out[m1] = tt + tt - tt * tt - 1.0
    m2 = t > 1.0 - dt
    tt = (t[m2] - 1.0) / dt[m2]
    out[m2] = tt * tt + tt + tt + 1.0
    return out


def _kernel_pieces(deg: int):
    """Linear (deg 1) or cubic (deg 3) B-spline, as polynomials on unit intervals from -support."""
    if deg == 1:
        return 1, [_Poly([1.0, 1.0]), _Poly([1.0, -1.0])]
    return 2, [_Poly([8.0, 12.0, 6.0, 1.0]) / 6.0, _Poly([2 / 3, 0.0, -1.0, -0.5]),
               _Poly([2 / 3, 0.0, -1.0, 0.5]), _Poly([8.0, -12.0, 6.0, -1.0]) / 6.0]


def _kernel_integrals(deg: int):
    a, ks = _kernel_pieces(deg)
    i1, i2, c1, c2 = [], [], 0.0, 0.0
    for j, k in enumerate(ks):
        lo = j - a
        p1 = k.integ(lbnd=lo) + c1
        p2 = p1.integ(lbnd=lo) + c2
        i1.append(p1)
        i2.append(p2)
        c1, c2 = p1(lo + 1), p2(lo + 1)
    return a, i1, i2


_RESID = {d: _kernel_integrals(d) for d in (1, 3)}


def residual(t, deg: int = 3, kind: str = "step") -> np.ndarray:
    """Band-limited step (``step``) or ramp (``ramp``) minus the trivial one, at ``t`` samples from the break,
    built from the integrated B-spline of degree ``deg`` (1: 2-point polyBLEP/BLAMP, 3: 4-point)."""
    a, i1, i2 = _RESID[deg]
    t = np.asarray(t, dtype=np.float64)
    out = np.zeros_like(t)
    for j, (p1, p2) in enumerate(zip(i1, i2)):
        m = (t >= j - a) & (t < j - a + 1)
        if m.any():
            tm = t[m]
            out[m] = p1(tm) - (tm >= 0) if kind == "step" else p2(tm) - np.maximum(tm, 0.0)
    return out


def _blep(t: np.ndarray, dt: np.ndarray, f: np.ndarray) -> np.ndarray:
    """2-point polyBLEP, 4-point (integrated cubic B-spline) where the tone is above 2 kHz."""
    out = _polyblep(t, dt)
    hi = f > 2000.0
    if hi.any():
        th, dh = t[hi], dt[hi]
        out[hi] = 2.0 * (residual(th / dh) + residual((th - 1.0) / dh))
    return out


def saw(freq, n: int, sr: int = DEFAULT_SR, phase0: float = 0.0) -> np.ndarray:
    f = _freq_array(freq, n)
    dt = np.maximum(f / sr, 1e-9)
    t = phase(f, n, sr, phase0) % 1.0
    naive = 2.0 * t - 1.0
    return naive - _blep(t, dt, f)


def pulse(freq, n: int, sr: int = DEFAULT_SR, width: float = 0.5, phase0: float = 0.0) -> np.ndarray:
    f = _freq_array(freq, n)
    dt = np.maximum(f / sr, 1e-9)
    t = phase(f, n, sr, phase0) % 1.0
    naive = np.where(t < width, 1.0, -1.0)
    out = naive + _blep(t, dt, f) - _blep((t + 1.0 - width) % 1.0, dt, f)
    return out


def square(freq, n: int, sr: int = DEFAULT_SR, phase0: float = 0.0) -> np.ndarray:
    return pulse(freq, n, sr, 0.5, phase0)


def _polyblamp(x: np.ndarray) -> np.ndarray:
    """2-point polyBLAMP residual (1 - |x|)^3 / 6 at x samples from a corner; scaled by the change of slope."""
    a = np.abs(x)
    return np.where(a < 1.0, (1.0 - a) ** 3 / 6.0, 0.0)


def triangle(freq, n: int, sr: int = DEFAULT_SR, phase0: float = 0.0, points: int = 4) -> np.ndarray:
    """Triangle with polyBLAMP-rounded corners (``points`` 2 or 4: the 4-point residual aliases less)."""
    f = _freq_array(freq, n)
    dt = np.clip(f / sr, 1e-9, 0.5)
    t = phase(f, n, sr, phase0) % 1.0
    u = (t + 0.5) % 1.0
    x_top = np.where(t < 0.5, t, t - 1.0) / dt          # samples from the maximum (slope +4 -> -4 per cycle)
    x_bot = np.where(u < 0.5, u, u - 1.0) / dt          # samples from the minimum (slope -4 -> +4)
    r = _polyblamp if points == 2 else (lambda x: residual(x, 3, "ramp"))
    return 4.0 * np.abs(t - 0.5) - 1.0 - 8.0 * dt * r(x_top) + 8.0 * dt * r(x_bot)


def classic(freq, n: int, sr: int = DEFAULT_SR, jumps=((0.0, 1.0),), corners=(), phase0: float = 0.0) -> np.ndarray:
    """Band-limited classical waveform from its breaks: `jumps` = (position in the cycle 0..1, rise in value),
    `corners` = (position, change of slope per cycle; they must add to 0). Each jump is a sawtooth
    (1/pi) sum sin(k w)/k, each corner a parabolic wave (1/(2 pi^2)) sum cos(k w)/k^2, summed up to Nyquist."""
    f = _freq_array(freq, n)
    x = phase(f, n, sr, phase0)
    out = np.zeros(n)
    for k in range(1, int(0.5 * sr / max(float(np.max(f)), 1e-6)) + 1):
        b = sum(d * -1j * np.exp(-2j * np.pi * k * pos) for pos, d in jumps) / (np.pi * k)
        b = b - sum(c * np.exp(-2j * np.pi * k * pos) for pos, c in corners) / (2.0 * np.pi ** 2 * k * k)
        if b != 0:
            out += np.real(b * np.exp(2j * np.pi * k * x))
    return out


JP8000_CENTS = (-21.0, -14.0, -7.0, 0.0, 4.0, 8.0, 12.0)


def supersaw(freq, n: int, sr: int = DEFAULT_SR, voices: int = 7, detune: float = 0.4, spread: float = 1.0,
             mode: str = "linear") -> np.ndarray:
    """Detuned stack of saws. detune in semitones total spread; ``mode`` jp8000 = the seven fixed offsets of the
    Roland JP-8000 (asymmetric, in cents, scaled by ``spread``)."""
    f = _freq_array(freq, n)
    out = np.zeros(n)
    rng = np.random.default_rng(1234)
    if mode == "jp8000":
        ratios = [2 ** (c * spread / 1200.0) for c in JP8000_CENTS]
    else:
        ratios = [2 ** (((i / max(1, voices - 1) - 0.5) * 2 if voices > 1 else 0.0) * detune / 12.0 * spread)
                  for i in range(voices)]
    for ratio in ratios:
        out += saw(f * ratio, n, sr, phase0=rng.random())
    return out / np.sqrt(len(ratios))


def fm(freq, n: int, sr: int = DEFAULT_SR, ratio: float = 2.0, index=1.0, phase0: float = 0.0,
       ratio2: float = 3.0, index2=0.0) -> np.ndarray:
    """FM (phase modulation): a sine carrier, a sine modulator at ``ratio`` with ``index`` and an optional second
    one in parallel (``ratio2``, ``index2``). Each index can be a scalar or a per-sample envelope array; the
    sideband k of one modulator has the amplitude J_k(index)."""
    f = _freq_array(freq, n)
    idx = _freq_array(index, n) if not np.isscalar(index) else index
    mod = idx * np.sin(2 * np.pi * phase(f * ratio, n, sr))
    if not (np.isscalar(index2) and index2 == 0):
        idx2 = _freq_array(index2, n) if not np.isscalar(index2) else index2
        mod = mod + idx2 * np.sin(2 * np.pi * phase(f * ratio2, n, sr))
    return np.sin(2 * np.pi * phase(f, n, sr, phase0) + mod)


def organ(freq, n: int, sr: int = DEFAULT_SR, drawbars=(1.0, 0.6, 0.4, 0.3, 0.2, 0.1)) -> np.ndarray:
    f = _freq_array(freq, n)
    out = np.zeros(n)
    harmonics = [1, 2, 3, 4, 6, 8]
    for h, g in zip(harmonics, drawbars):
        out += g * sine(f * h, n, sr)
    return out / sum(drawbars)


def additive(freq, n: int, sr: int = DEFAULT_SR, partials=None, inharmonic: float = 0.0) -> np.ndarray:
    """partials: list of (harmonic_number, amplitude, decay_seconds). Inharmonic stretches upper partials (piano/bell)."""
    f = _freq_array(freq, n)
    t = np.arange(n) / sr
    out = np.zeros(n)
    nyq = sr / 2
    for h, amp, dec in partials or [(1, 1.0, 1.0)]:
        stretch = np.sqrt(1 + inharmonic * h * h)
        fh = f * h * stretch
        live = fh < nyq
        if not live.any():
            continue
        out += amp * live * np.exp(-t / max(dec, 1e-3)) * sine(fh, n, sr)
    return out


def harmonic(f0, amp, dist, n: int, sr: int = DEFAULT_SR) -> np.ndarray:
    """Harmonic bank with factored amplitudes: ``amp`` the overall amplitude A(n), ``dist`` the share of each
    harmonic c_k(n) (frames x harmonics, or one row). Frame values are interpolated to samples; at every sample
    the harmonics at or above Nyquist are dropped and the rest renormalised to sum to 1."""
    f = _freq_array(f0, n)
    a = _freq_array(amp, n) if not np.isscalar(amp) else np.full(n, float(amp))
    c = np.atleast_2d(np.asarray(dist, dtype=np.float64))
    K = c.shape[1]
    grid = np.linspace(0, 1, n)
    cs = np.stack([np.interp(grid, np.linspace(0, 1, c.shape[0]), c[:, k]) for k in range(K)], axis=1)
    k = np.arange(1, K + 1)
    cs = cs * (f[:, None] * k[None, :] < sr / 2)
    cs = cs / np.maximum(cs.sum(axis=1, keepdims=True), 1e-12)
    ph = 2 * np.pi * phase(f, n, sr)
    return a * np.sum(cs * np.sin(ph[:, None] * k[None, :]), axis=1)


def wavefold(freq, n: int, sr: int = DEFAULT_SR, amount: float = 2.0) -> np.ndarray:
    x = sine(freq, n, sr) * amount
    return np.abs(((x + 1) % 4) - 2) - 1


def reflect(x, lo, hi):
    """Fold x back into [lo, hi] at both walls."""
    w = hi - lo
    if w <= 0:
        return np.full_like(np.asarray(x, float), lo)
    y = np.mod(np.asarray(x, float) - lo, 2.0 * w)
    return lo + np.where(y > w, 2.0 * w - y, y)


GENDY_DISTS = ("uniform", "cauchy", "logist", "hyperbcos", "arcsine", "expon")
GENDY_PRESETS = {
    "xenakis": dict(amp_dist="cauchy", dur_dist="cauchy", amp_param=1.0, dur_param=1.0, dur_spread=0.3),
    "buzz": dict(amp_dist="logist", amp_param=0.6, amp_step=0.3, dur_spread=0.0),
    "drift": dict(amp_dist="arcsine", dur_dist="arcsine", amp_param=0.8, dur_param=0.8, amp_step=0.15, dur_spread=0.05),
    "growl": dict(amp_dist="expon", dur_dist="hyperbcos", amp_param=0.9, dur_param=0.5, dur_spread=0.5),
}


def xenakis_dist(u, kind: str = "uniform", a: float = 0.5) -> np.ndarray:
    """Maps uniform u in [0, 1] to [-1, 1] through one of the Xenakis distributions of dynamic stochastic
    synthesis; ``a`` (0..1) sets the spread."""
    u = np.asarray(u, float)
    a = float(np.clip(a, 1e-4, 1.0))
    if kind == "cauchy":
        return 0.1 * np.tan(np.arctan(10.0 * a) * (2.0 * u - 1.0)) / a
    if kind == "logist":
        c = 0.5 + 0.499 * a
        v = (u - 0.5) * 0.998 * a + 0.5
        return np.log((1.0 - v) / v) / np.log((1.0 - c) / c)
    if kind == "hyperbcos":
        t = np.tan(1.5692255 * a * u) / np.tan(1.5692255 * a)
        return 2.0 * np.log(t * 0.999 + 0.001) * -0.1447648 - 1.0
    if kind == "arcsine":
        return np.sin(np.pi * (u - 0.5) * a) / np.sin(0.5 * np.pi * a)
    if kind == "expon":
        return 2.0 * np.log(1.0 - u * 0.999 * a) / np.log(1.0 - 0.999 * a) - 1.0
    if kind == "uniform":
        return 2.0 * u - 1.0
    raise ValueError(f"distribution: {' | '.join(GENDY_DISTS)}")


def gendyn(freq, n: int, sr: int = DEFAULT_SR, points: int = 12, amp_step: float = 0.5, dur_spread: float = 0.0,
           seed: int = 0, curve: float = 0.0, amp_dist: str = "uniform", dur_dist: str = "uniform",
           amp_param: float = 0.5, dur_param: float = 0.5, minfreq: float | None = None, maxfreq: float | None = None,
           preset: str | None = None) -> np.ndarray:
    """Dynamic stochastic synthesis: a cycle of ``points`` breakpoints, each with an amplitude and a segment length
    that move every cycle by a second-order random walk (a random walk sets the step of another), both walls
    reflecting. Amplitudes stay in [-1, 1] with steps in [-amp_step, amp_step]; segment lengths in
    (1 +- dur_spread) of period / points, their steps in +- dur_spread / 2 of it, so the pitch wanders
    around ``freq`` (0: fixed pitch, the timbre alone walks). ``minfreq`` / ``maxfreq`` replace the length walls
    so the cycle stays between those pitches. Steps are drawn from ``amp_dist`` / ``dur_dist`` (uniform, cauchy,
    logist, hyperbcos, arcsine, expon; spread ``amp_param`` / ``dur_param``); ``preset`` xenakis, buzz, drift or
    growl sets them together. Segments are straight lines, or with ``curve`` > 0
    exponential bends a + (b - a)(1 - e^(c u)) / (1 - e^c), c drawn per segment in [-curve, curve]."""
    if preset:
        p = GENDY_PRESETS[preset]
        amp_dist, dur_dist = p.get("amp_dist", amp_dist), p.get("dur_dist", dur_dist)
        amp_param, dur_param = p.get("amp_param", amp_param), p.get("dur_param", dur_param)
        amp_step, dur_spread = p.get("amp_step", amp_step), p.get("dur_spread", dur_spread)
    f = _freq_array(freq, n)
    rng = np.random.default_rng(seed)
    P = max(2, int(points))
    a = rng.uniform(-1.0, 1.0, P)
    da = np.zeros(P)
    u = np.ones(P)
    du = np.zeros(P)
    s, b = float(dur_spread), float(amp_step)
    xs, ys = [0.0], [a[0]]
    t = 0.0
    while t < n:
        if amp_dist == "uniform":
            da = reflect(da + rng.uniform(-b, b, P), -b, b)
        else:
            da = reflect(da + b * xenakis_dist(rng.random(P), amp_dist, amp_param), -b, b)
        a = reflect(a + da, -1.0, 1.0)
        if dur_dist == "uniform":
            du = reflect(du + rng.uniform(-s / 2, s / 2, P), -s / 2, s / 2)
        else:
            du = reflect(du + s / 2 * xenakis_dist(rng.random(P), dur_dist, dur_param), -s / 2, s / 2)
        if minfreq or maxfreq:
            fc = max(f[min(int(t), n - 1)], 1e-3)
            lo = fc / float(maxfreq) if maxfreq else 1.0 - s
            hi = fc / float(minfreq) if minfreq else 1.0 + s
            u = reflect(u + du, min(lo, hi), max(lo, hi))
        else:
            u = reflect(u + du, 1.0 - s, 1.0 + s)
        seg = u * sr / (max(f[min(int(t), n - 1)], 1e-3) * P)
        xs += list(t + np.cumsum(seg))
        ys += list(np.roll(a, -1))
        t = xs[-1]
    if curve <= 0:
        return np.interp(np.arange(n), xs, ys)
    xs, ys = np.asarray(xs), np.asarray(ys)
    cs = np.random.default_rng(seed + 1).uniform(-curve, curve, len(xs))
    i = np.arange(n, dtype=np.float64)
    j = np.clip(np.searchsorted(xs, i, side="right") - 1, 0, len(xs) - 2)
    u = (i - xs[j]) / np.maximum(xs[j + 1] - xs[j], 1e-12)
    c = cs[j]
    flat = np.abs(c) < 1e-4
    shape = np.where(flat, u, (1.0 - np.exp(c * u)) / np.where(flat, 1.0, 1.0 - np.exp(c)))
    return ys[j] + (ys[j + 1] - ys[j]) * shape


def white(n: int, seed: int | None = None) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1, 1, n)


def pink(n: int, seed: int | None = None) -> np.ndarray:
    """White noise through a 3-pole / 3-zero filter whose response follows 1/f (-3 dB per octave)."""
    w = white(n, seed)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    return lfilter(b, a, w) * 3.5


def brown(n: int, seed: int | None = None, sr: int = DEFAULT_SR) -> np.ndarray:
    w = white(n, seed)
    x = lfilter([1], [1, -0.995], w)
    return x / (np.max(np.abs(x)) + 1e-9)


_PK3_POLES = (0.99886, 0.99332, 0.969, 0.8665, 0.55, -0.7616)
_PK3_GAINS = (0.0555179, 0.0750759, 0.153852, 0.3104856, 0.5329522, -0.016898)


def pink_pk3(w: np.ndarray) -> np.ndarray:
    """Six one-pole sections plus a direct and a one-sample-delayed tap (the 'pk3' refined pink filter), / 7."""
    y = sum(lfilter([g], [1.0, -p], w) for p, g in zip(_PK3_POLES, _PK3_GAINS))
    y = y + 0.5362 * w + 0.115926 * np.concatenate(([0.0], w[:-1]))
    return y / 7.0


def held_noise(n: int, rate: float, sr: int = DEFAULT_SR, seed: int | None = 0) -> np.ndarray:
    """Stepped noise: a new uniform value ``rate`` times a second, held in between."""
    m = int(np.ceil(n * rate / sr)) + 1
    v = np.random.default_rng(seed).uniform(-1, 1, m)
    return v[np.minimum((np.arange(n) * rate / sr).astype(int), m - 1)]


def pitched_noise(freq, n: int, sr: int = DEFAULT_SR, color: str = "white", interpolate: bool = True,
                  seed: int | None = 0) -> np.ndarray:
    """Noise clocked by a pitch: two new values per period (so its spectrum follows ``freq``), joined by
    straight lines or held. ``color`` white, pink (the pk3 filter run over the values) or brown
    (clamped walk x + 0.1 w)."""
    ph = 2.0 * phase(freq, n, sr)
    idx = np.floor(ph).astype(np.int64)
    idx -= idx[0]
    m = int(idx[-1]) + 2
    w = np.random.default_rng(seed).uniform(-1, 1, m)
    if color == "pink":
        v = pink_pk3(w)
    elif color == "brown":
        v = np.empty(m)
        x = 0.0
        for i in range(m):
            x = min(1.0, max(-1.0, x + 0.1 * w[i]))
            v[i] = x
    elif color == "white":
        v = w
    else:
        raise ValueError("color: white | pink | brown")
    if not interpolate:
        return v[idx]
    fr = ph - np.floor(ph)
    return v[idx] + (v[idx + 1] - v[idx]) * fr


def phase_distortion(freq, n: int, sr: int = DEFAULT_SR, amount: float = 0.5, bipolar: bool = False,
                     phase0: float = 0.0) -> np.ndarray:
    """Casio CZ phase distortion: the phase is bent in two straight pieces meeting at ``mid`` before a cosine
    is read. Unipolar: mid = (amount + 1) / 2, slopes 0.5 / mid then 0.5 / (1 - mid). Bipolar (phase in
    [-1, 1]): mid = amount, slopes 1 / (mid + 1) then 1 / (1 - mid). amount 0 = a plain cosine; towards +-1
    the wave turns saw-like."""
    a = float(np.clip(amount, -0.999, 0.999))
    t = phase(freq, n, sr, phase0) % 1.0
    if bipolar:
        x = 2.0 * t - 1.0
        y = np.where(x < a, (x + 1.0) / (a + 1.0) - 1.0, (x - a) / (1.0 - a))
        return np.cos(np.pi * y)
    mid = (a + 1.0) / 2.0
    y = np.where(t < mid, 0.5 * t / mid, 0.5 + 0.5 * (t - mid) / (1.0 - mid))
    return np.cos(2 * np.pi * y)


def gbuzz(freq, n: int, sr: int = DEFAULT_SR, nharm: int = 4, lharm: int = 1, mul: float = 0.1,
          phase0: float = 0.0) -> np.ndarray:
    """Closed-form buzz: ``nharm`` cosine harmonics from ``lharm``, each ``mul`` times the one below, scaled so the
    amplitudes sum to 1. The count follows the pitch sample by sample: a harmonic fades out linearly over its last
    f Hz below Nyquist, so a glissando adds and drops harmonics without clicks or aliasing."""
    f = np.maximum(_freq_array(freq, n), 1e-6)
    k = max(int(lharm), 1)
    x = sr / 2.0 / f - k + 1.0
    fl = np.floor(x)
    m = np.clip(fl, 1, max(int(nharm), 1)).astype(np.int64)
    g = np.where(fl > m, 1.0, np.clip(x - m, 0.0, 1.0))
    r = float(mul)
    p = 2 * np.pi * phase(f, n, sr, phase0)
    top = r ** (m - 1) * np.cos((k + m - 1) * p)
    if abs(abs(r) - 1.0) < 1e-6:
        s = sum(np.where(i < m, r ** i * np.cos((k + i) * p), 0.0) for i in range(int(m.max())))
        return (s - (1.0 - g) * top) / np.maximum(m - (1.0 - g), 1e-12)
    num = np.cos(k * p) - r * np.cos((k - 1) * p) - r ** m * np.cos((k + m) * p) + r ** (m + 1) * np.cos((k + m - 1) * p)
    s = num / (1.0 + r * r - 2.0 * r * np.cos(p))
    total = (1.0 - abs(r) ** m) / (1.0 - abs(r)) - (1.0 - g) * abs(r) ** (m - 1)
    return (s - (1.0 - g) * top) / np.maximum(total, 1e-12)


def _dpw_poly(order: int, wave: str):
    """Pieces on [-1, 0) and [0, 1) of the periodic (order - 1)-fold integral of the trivial wave, each integral
    made zero-mean so the next stays periodic."""
    if wave == "saw":
        neg = pos = _Poly([0.0, 1.0])
    else:
        neg, pos = _Poly([1.0, 2.0]), _Poly([1.0, -2.0])
    for _ in range(order - 1):
        neg, pos = neg.integ(lbnd=0), pos.integ(lbnd=0)
        mean = (pos.integ(lbnd=0)(1.0) - neg.integ(lbnd=0)(-1.0)) / 2.0
        neg, pos = neg - mean, pos - mean
    return neg, pos


def dpw(freq, n: int, sr: int = DEFAULT_SR, order: int = 3, wave: str = "saw", phase0: float = 0.0) -> np.ndarray:
    """Differentiated polynomial waveform: the (order - 1)-fold integral of a trivial saw or triangle, sampled,
    then differenced order - 1 times and scaled by (pi / (2 sin(pi f / sr)))^(order - 1). ``wave`` saw,
    triangle or square (two saws half a period apart). Order 1 = trivial; 2..6 alias less and less."""
    order = int(np.clip(order, 1, 6))
    if wave == "square":
        return dpw(freq, n, sr, order, "saw", phase0 + 0.5) - dpw(freq, n, sr, order, "saw", phase0)
    if wave not in ("saw", "triangle"):
        raise ValueError("dpw wave: saw | triangle | square")
    f = _freq_array(freq, n)
    m = order - 1
    fe = np.concatenate((np.full(m, f[0]), f))
    t = (phase0 - m * f[0] / sr + np.cumsum(fe / sr)) % 1.0
    x = 2.0 * t - 1.0
    neg, pos = _dpw_poly(order, wave)
    y = np.where(x < 0, neg(x), pos(x))
    for _ in range(m):
        y = np.diff(y)
    return y * (np.pi / (2.0 * np.sin(np.pi * np.clip(f / sr, 1e-9, 0.5)))) ** m


def _bspline3(t: np.ndarray) -> np.ndarray:
    a = np.abs(t)
    return np.where(a < 1, 2 / 3 - a * a + a ** 3 / 2, np.where(a < 2, (2 - a) ** 3 / 6, 0.0))


def blit(freq, n: int, sr: int = DEFAULT_SR, harmonics: int = 0, kernel: str = "sinc", phase0: float = 0.0) -> np.ndarray:
    """Band-limited impulse train, one unit-area pulse per period. ``kernel`` sinc: closed form
    sin(M pi x) / (P sin(pi x)) with M odd = 2 H + 1, H = floor(P / 2) harmonics (``harmonics`` > 0 sets fewer),
    the top one faded linearly over its last f Hz below Nyquist so a glissando changes H without clicks;
    bspline: each pulse a cubic B-spline at its exact fractional position (spectrum sinc^4, -3 dB at 10 kHz at
    44.1 kHz); thiran: the pulse on a sample, its fractional part D in [0.418, 1.418] by a first-order allpass
    a1 = (1 - D) / (1 + D)."""
    f = _freq_array(freq, n)
    ph = phase(f, n, sr, phase0)
    if kernel == "sinc":
        P = sr / np.maximum(f, 1e-6)
        H = np.maximum(np.floor(P / 2), 1.0)
        g = np.clip(P / 2 - H, 0.0, 1.0)
        if harmonics:
            g = np.where(H > int(harmonics), 1.0, g)
            H = np.minimum(H, int(harmonics))
        M = 2 * H + 1
        x = ph % 1.0
        den = np.sin(np.pi * x)
        small = np.abs(den) < 1e-9
        d = np.where(small, M, np.sin(M * np.pi * x) / np.where(small, 1.0, den))
        return (d - 2.0 * (1.0 - g) * np.cos(2 * np.pi * H * x)) / P
    out = np.zeros(n + 64)
    k0, k1 = int(np.ceil(ph[0])), int(np.floor(ph[-1]))
    s = np.arange(n, dtype=np.float64)
    for k in range(k0, k1 + 1):
        j = int(np.searchsorted(ph, k))
        t0 = s[j] - (ph[j] - k) / max(ph[j] - (ph[j - 1] if j else ph[j] - f[j] / sr), 1e-12)
        if kernel == "bspline":
            i = np.arange(int(np.floor(t0)) - 1, int(np.floor(t0)) + 3)
            i = i[(i >= 0) & (i < n)]
            out[i] += _bspline3(i - t0)
        else:
            m0 = int(np.floor(t0 - 0.418))
            D = t0 - m0
            a1 = (1.0 - D) / (1.0 + D)
            h = np.empty(64)
            h[0] = a1
            h[1:] = (1.0 - a1 * a1) * (-a1) ** np.arange(63)
            lo = max(m0, 0)
            out[lo:m0 + 64] += h[lo - m0:]
    return out[:n]


def blit_saw(freq, n: int, sr: int = DEFAULT_SR, harmonics: int = 0, kernel: str = "sinc") -> np.ndarray:
    """Saw from a leaky integral (0.995) of the BLIT minus its mean f / sr, scaled to about +-1."""
    f = _freq_array(freq, n)
    y = lfilter([1.0], [1.0, -0.995], blit(f, n, sr, harmonics, kernel) - f / sr)
    return 2.0 * y


def blit_square(freq, n: int, sr: int = DEFAULT_SR, harmonics: int = 0, kernel: str = "sinc") -> np.ndarray:
    """Square from the leaky integral of a bipolar BLIT (a pulse up, half a period later one down)."""
    b = blit(freq, n, sr, harmonics, kernel) - blit(freq, n, sr, harmonics, kernel, phase0=0.5)
    y = lfilter([1.0], [1.0, -0.995], b)
    return 2.0 * (y - np.mean(y))


_SINC_OVER, _SINC_ZC = 64, 16


def _sinc_blep_table() -> np.ndarray:
    """Band-limited step from a Blackman-windowed sinc (cutoff at Nyquist), 16 zero crossings each side,
    64 points per sample, integrated: index 0 = -16 samples from the step. Kept smooth (the ideal step is taken
    off after interpolation, so a step just before a sample is not smeared across the jump)."""
    t = np.arange(-_SINC_ZC * _SINC_OVER, _SINC_ZC * _SINC_OVER + 1) / _SINC_OVER
    h = np.sinc(t) * np.blackman(len(t))
    step = np.cumsum(h)
    return step / step[-1]


_SINC_BLEP = _sinc_blep_table()


def _add_blep(out: np.ndarray, t0: float, jump: float) -> None:
    """Adds ``jump`` x the sinc BLEP residual for a step at fractional sample time ``t0``: the band-limited step
    (linear interpolation in the table) minus the ideal step."""
    i = np.arange(int(np.floor(t0)) - _SINC_ZC + 1, int(np.floor(t0)) + _SINC_ZC + 1)
    i = i[(i >= 0) & (i < len(out))]
    pos = (i - t0 + _SINC_ZC) * _SINC_OVER
    smooth = np.interp(pos, np.arange(len(_SINC_BLEP)), _SINC_BLEP, left=0.0, right=1.0)
    out[i] += jump * (smooth - (i >= t0))


def hard_sync(freq, n: int, sr: int = DEFAULT_SR, ratio: float = 2.5, phase0: float = 0.0) -> np.ndarray:
    """Hard-synced saw: a slave saw at ``ratio`` x ``freq`` restarted at every period of the master ``freq``.
    Every wrap and every restart is a step corrected by a band-limited step from a windowed-sinc table."""
    f = _freq_array(freq, n)
    r = max(float(ratio), 1e-3)
    pm = phase(f, n, sr, phase0)
    u = r * (pm % 1.0)
    out = 2.0 * (u % 1.0) - 1.0
    pmp = np.concatenate(([pm[0] - f[0] / sr], pm[:-1]))
    up = r * (pmp % 1.0)
    restart = np.floor(pm) > np.floor(pmp)
    events = np.nonzero(restart | (np.floor(u) > np.floor(up)))[0]
    v_end = 2.0 * (r - np.ceil(r) + 1.0) - 1.0                  # slave value just before a restart
    for j in events:
        if restart[j]:
            sm = (np.floor(pm[j]) - pmp[j]) / max(pm[j] - pmp[j], 1e-12)
            for k in range(int(np.floor(up[j])) + 1, int(np.ceil(r))):
                _add_blep(out, j - 1 + sm * (k - up[j]) / max(r - up[j], 1e-12), -2.0)
            _add_blep(out, j - 1 + sm, -1.0 - v_end)
            for k in range(1, int(np.floor(u[j])) + 1):
                _add_blep(out, j - 1 + sm + (1.0 - sm) * k / max(u[j], 1e-12), -2.0)
        else:
            for k in range(int(np.floor(up[j])) + 1, int(np.floor(u[j])) + 1):
                _add_blep(out, j - 1 + (k - up[j]) / max(u[j] - up[j], 1e-12), -2.0)
    return out


DPW_ALIAS_FREE = {44100: (600.0, 2037.0, 4593.0, 7851.0, 12221.0), 88200: (1019.0, 3901.0, 6691.0, 9614.0, 12435.0)}
SAW_ALIAS_FREE = {"lane": 600.0, "dpw": 600.0, "dpw_2x": 1136.0, "blit_sws": 1203.0, "blit_fdf": 4593.0, "blep": 5135.0}
FRAC_DELAY_ALIAS_FREE = {"thiran1": 600.0, "lagrange1": 1269.0, "lagrange2": 2239.0, "bspline2": 3228.0,
                         "thiran2": 3780.0, "lagrange3": 4143.0, "bspline3": 5784.0}


def dpw_order_for(f0: float, sr: int = 44100) -> int:
    """Lowest DPW order (2..6) whose aliasing stays inaudible up to `f0` (limits measured at 44.1 and 88.2 kHz,
    scaled with sr elsewhere); 0 when even order 6 does not suffice."""
    base = 88200 if sr >= 66150 else 44100
    lim = np.asarray(DPW_ALIAS_FREE[base]) * (float(sr) / base)
    ok = np.nonzero(lim >= float(f0))[0]
    return int(ok[0] + 2) if ok.size else 0


WAVETABLE_N = 2048


def wavetable_from_spectrum(harmonics, n: int = WAVETABLE_N) -> np.ndarray:
    """One period of n points from harmonic amplitudes c_1..c_N (inverse real FFT of [0, c_1..c_N, 0...])."""
    c = np.zeros(n // 2 + 1)
    h = np.asarray(harmonics, float)[: n // 2 - 1]
    c[1:1 + h.size] = h
    return np.fft.irfft(c * (n / 2.0), n)


def table_read(tab, pos, interp: str = "lagrange", rows=None) -> np.ndarray:
    """Periodic read of a table (or of row ``rows[i]`` of a 2-D stack) at fractional index ``pos``: ``lagrange``
    = 4-point cubic Lagrange (points i-1..i+2), ``linear``."""
    tab = np.asarray(tab, float)
    N = tab.shape[-1]
    pos = np.asarray(pos, float)
    i = np.floor(pos).astype(np.int64)
    f = pos - i
    get = (lambda j: tab[j % N]) if rows is None else (lambda j: tab[rows, j % N])
    if interp == "linear":
        return get(i) + f * (get(i + 1) - get(i))
    fp, fm, f2 = f + 1.0, f - 1.0, f - 2.0
    return (-f * fm * f2 / 6.0 * get(i - 1) + fp * fm * f2 / 2.0 * get(i)
            - fp * f * f2 / 2.0 * get(i + 1) + fp * f * fm / 6.0 * get(i + 2))


MIP_PER_OCTAVE = 12
MIP_LIMIT = 0.45


def mipmap(shape=None, harmonics=None, coeffs=None, per_octave: int = MIP_PER_OCTAVE, size: int = WAVETABLE_N):
    """Band-limited family of one period: table j keeps the harmonics up to floor(2^(j / per_octave)). The period is
    given as samples (``shape``, any length, its harmonics found by FFT, DC dropped), as cosine amplitudes
    (``harmonics``) or as complex coefficients c_k of sum Re(c_k e^(2 pi i k t)) (``coeffs``)."""
    K = size // 2 - 1
    C = np.zeros(size // 2 + 1, complex)
    if shape is not None:
        s = np.asarray(shape, float)
        X = np.fft.rfft(s) * (2.0 / len(s))
        m = min(len(X) - 1, K + 1)
        C[1:m] = X[1:m]
    else:
        c = np.asarray(coeffs if coeffs is not None else harmonics, complex)[:K]
        C[1:1 + c.size] = c
    top = int(np.max(np.nonzero(np.abs(C) > 0)[0], initial=1))
    J = int(np.floor(per_octave * np.log2(max(top, 1)))) + 1
    H = np.minimum(np.floor(2.0 ** (np.arange(J) / per_octave)).astype(int), top)
    uniq, inv = np.unique(H, return_inverse=True)
    spec = np.where(np.arange(C.size)[None, :] <= uniq[:, None], C[None, :], 0.0)
    return np.fft.irfft(spec * (size / 2.0), size, axis=1), inv, per_octave


def mip_read(mm, freq, ph, sr: int, limit: float = MIP_LIMIT, interp: str = "lagrange") -> np.ndarray:
    """Reads a ``mipmap`` at phase ``ph`` (cycles) for pitch ``freq``: the richest table whose harmonics stay under
    limit x sr and the one below it, crossfaded by the fractional table index, so the bandwidth follows the pitch
    smoothly."""
    tabs, inv, ppo = mm
    J = len(inv)
    f = np.abs(np.asarray(freq, float)) + 1e-9
    c = np.clip(ppo * np.log2(np.maximum(limit * sr / f, 1.0)), 0.0, J - 1.0)
    j = np.floor(c).astype(np.int64)
    w = np.where(j >= J - 1, 1.0, c - j)
    pos = (np.asarray(ph, float) % 1.0) * tabs.shape[1]
    hi = table_read(tabs, pos, interp, rows=inv[j])
    lo = table_read(tabs, pos, interp, rows=inv[np.maximum(j - 1, 0)])
    return lo + w * (hi - lo)


def wavetable_osc(freq, n: int, sr: int = DEFAULT_SR, harmonics=(1.0,), phase0: float = 0.0, interp: str = "lagrange",
                  mip: bool = True) -> np.ndarray:
    """Table oscillator from cosine harmonic amplitudes. ``mip``: a band-limited table per 1/12 octave, the pair
    around the current pitch crossfaded, so harmonics above 0.45 sr drop out sample by sample (a sweep keeps its
    top end); off: one table cut at the highest frequency of ``freq``. Read by 4-point Lagrange (``interp``
    lagrange) or linearly."""
    f = _freq_array(freq, n)
    ph = phase(f, n, sr, phase0)
    if mip:
        return mip_read(mipmap(harmonics=harmonics), f, ph, sr, interp=interp)
    h = np.asarray(harmonics, float)
    keep = max(1, int(MIP_LIMIT * sr / max(float(np.max(f)), 1e-9)))
    return table_read(wavetable_from_spectrum(h[:keep]), (ph % 1.0) * WAVETABLE_N, interp)


def wavetable_morph(freq, n: int, sr: int = DEFAULT_SR, tables=(), position=0.0, phase0: float = 0.0,
                    per_octave: int = MIP_PER_OCTAVE) -> np.ndarray:
    """Morphing table oscillator: ``tables`` is a list of one-period shapes (any length each), ``position`` (scalar or
    per sample, 0 .. len - 1) slides between neighbours by linear crossfade. Each table is band-limited by its own
    mip-map, so the morph never aliases."""
    f = _freq_array(freq, n)
    ph = phase(f, n, sr, phase0)
    pos = np.clip(_freq_array(position, n), 0.0, len(tables) - 1.0)
    out = np.zeros(n)
    for k, tab in enumerate(tables):
        w = np.clip(1.0 - np.abs(pos - k), 0.0, 1.0)
        live = np.nonzero(w > 0)[0]
        if live.size:
            out[live] += w[live] * mip_read(mipmap(shape=tab, per_octave=per_octave), f[live], ph[live], sr)
    return out


def _vco2_coeffs(wave: str, K: int = WAVETABLE_N // 2 - 1) -> np.ndarray:
    k = np.arange(1, K + 1, dtype=float)
    odd = (k % 2) == 1
    if wave == "saw":
        return 2j / (np.pi * k)
    if wave == "square":
        return np.where(odd, -4j / (np.pi * k), 0.0)
    if wave == "triangle":
        return np.where(odd, 8.0 / (np.pi * k) ** 2, 0.0)
    raise ValueError("vco2 wave: saw | square | triangle | pulse")


_VCO2_CACHE: dict = {}


def vco2(freq, n: int, sr: int = DEFAULT_SR, wave: str = "saw", width=0.5, shape=None, phase0: float = 0.0,
         per_octave: int = MIP_PER_OCTAVE) -> np.ndarray:
    """Band-limited table oscillator: a family of tables per 1/``per_octave`` octave (``mipmap``) read at the pitch,
    from a classic ``wave`` (saw, square, triangle; pulse = two saws ``width`` of a period apart, so the width can
    move per sample without aliasing; zero mean, swing 2) or from any one-period ``shape``."""
    f = _freq_array(freq, n)
    ph = phase(f, n, sr, phase0)
    if shape is not None:
        return mip_read(mipmap(shape=shape, per_octave=per_octave), f, ph, sr)
    key = ("saw" if wave == "pulse" else wave, per_octave)
    if key not in _VCO2_CACHE:
        _VCO2_CACHE[key] = mipmap(coeffs=_vco2_coeffs(key[0]), per_octave=per_octave)
    mm = _VCO2_CACHE[key]
    if wave != "pulse":
        return mip_read(mm, f, ph, sr)
    w = np.clip(_freq_array(width, n), 0.01, 0.99)
    return mip_read(mm, f, ph, sr) - mip_read(mm, f, ph - (1.0 - w), sr)


def minblep_table(zc: int = _SINC_ZC, over: int = _SINC_OVER) -> np.ndarray:
    """Minimum-phase band-limited step residual (minBLEP): the windowed sinc made minimum phase through the real
    cepstrum, integrated, minus the ideal step. Index 0 = the step instant, `over` points per sample; it ends at 0."""
    t = np.arange(-zc * over, zc * over + 1) / over
    h = np.sinc(t) * np.blackman(len(t))
    N = 1 << int(np.ceil(np.log2(4 * len(h))))
    c = np.fft.ifft(np.log(np.maximum(np.abs(np.fft.fft(h, N)), 1e-12))).real
    w = np.zeros(N)
    w[0], w[1:N // 2], w[N // 2] = 1.0, 2.0, 1.0
    hm = np.fft.ifft(np.exp(np.fft.fft(c * w))).real[:len(h)]
    step = np.cumsum(hm)
    return step / step[-1] - 1.0


_MINBLEP = minblep_table()


def minblep_saw(freq, n: int, sr: int = DEFAULT_SR, phase0: float = 0.0) -> np.ndarray:
    """Rising saw (-1..1) whose wraps are corrected by the minBLEP: causal, nothing before the step, so the
    correction also suits a reset that cannot be foreseen (hard sync, retrigger)."""
    f = _freq_array(freq, n)
    pm = phase(f, n, sr, phase0)
    out = 2.0 * (pm % 1.0) - 1.0
    span = (len(_MINBLEP) - 1) // _SINC_OVER
    for j in np.nonzero(np.floor(pm[1:]) > np.floor(pm[:-1]))[0] + 1:
        dt = max(pm[j] - pm[j - 1], 1e-12)
        t0 = j - 1 + (np.floor(pm[j]) - pm[j - 1]) / dt
        i = np.arange(j, min(n, j + span))
        out[i] -= 2.0 * np.interp((i - t0) * _SINC_OVER, np.arange(len(_MINBLEP)), _MINBLEP, right=0.0)
    return out


def corner_clip(x, limit: float = 1.0, mode: str = "clip") -> np.ndarray:
    """Hard clipper (``clip``, at +-limit), half-wave (``half``) or full-wave (``full``) rectifier with its corners
    band-limited: each crossing is located by a cubic Lagrange fit on 4 samples solved by Newton from 1.5, its
    slope mu taken there, and a 2-point polyBLAMP residual x |mu| (x 2 for the full wave) laid on the two
    samples around it."""
    x = np.asarray(x, dtype=np.float64)
    lv = float(limit) if mode == "clip" else 0.0
    if mode == "clip":
        y = np.clip(x, -lv, lv)
        levels = ((lv, -1.0), (-lv, 1.0))
    elif mode == "half":
        y = np.maximum(x, 0.0)
        levels = ((0.0, 1.0),)
    elif mode == "full":
        y = np.abs(x)
        levels = ((0.0, 2.0),)
    else:
        raise ValueError("mode: clip | half | full")
    for L, sgn in levels:
        s = np.sign(x - L)
        for n in np.nonzero(s[1:-2] * s[2:-1] < 0)[0] + 1:     # crossing between n and n+1
            p = x[n - 1:n + 3] - L
            a = (-p[0] + 3 * p[1] - 3 * p[2] + p[3]) / 6.0
            b = (p[0] - 2 * p[1] + p[2]) / 2.0
            c = (-2 * p[0] - 3 * p[1] + 6 * p[2] - p[3]) / 6.0
            e = p[1]
            D = 1.5
            for _ in range(8):
                t = D - 1.0
                g = ((a * t + b) * t + c) * t + e
                dg = (3 * a * t + 2 * b) * t + c
                if abs(dg) < 1e-12:
                    break
                D = min(2.0, max(1.0, D - g / dg))
            d = D - 1.0
            mu = abs((3 * a * d + 2 * b) * d + c)
            y[n:n + 2] += sgn * mu * _polyblamp(np.array([d, 1.0 - d]))
    return y


def pulse_train(freq, n: int, sr: int = DEFAULT_SR, index=1.0, shape: str = "gauss", carrier=0.0, copies: int = 1,
                phase0: float = 0.0) -> np.ndarray:
    """Buzz source, a pulse of adjustable width once per period, mean removed.

    ``shape`` gauss: e^(index (cos w n - 1)), the Gaussian train whose width falls as the index rises (harmonics
    e^-a [I0(a) + 2 sum In(a) cos k w n]); hann: the raised cosine read ``index`` times faster, duty 1/index, index >= 1;
    sine: a comparison sine-shaped hump (1 + cos w n) / 2 raised to ``index``. ``carrier``: ratio of a cosine locked to
    the pulse (a formant at carrier x freq, a wave packet). ``copies`` 2: a second pulse half a period later with the
    carrier continuing in phase, for a band narrower than the fundamental."""
    from scipy.special import ive
    ph = np.mod(phase(freq, n, sr, phase0), 1.0)
    a = np.maximum(_freq_array(index, n), 0.0 if shape == "gauss" else 1.0)
    c = _freq_array(carrier, n)

    def one(p, shift):
        x = 2 * np.pi * (p - 0.5)
        if shape == "gauss":
            y = np.exp(-a * (1.0 + np.cos(x)))
        elif shape == "hann":
            y = 0.5 * (1.0 + np.cos(np.clip(a * x, -np.pi, np.pi)))
        elif shape == "sine":
            y = (0.5 * (1.0 - np.cos(x))) ** a
        else:
            raise ValueError(f"pulse_train shape {shape!r}: gauss, hann or sine")
        return y * np.cos(c * x + shift) if np.any(c) else y

    y = one(ph, 0.0)
    if int(copies) == 2:
        y = y + one(np.mod(ph + 0.5, 1.0), np.pi * c)
    if shape == "gauss" and not np.any(c):
        return y - ive(0, a) * (2 if int(copies) == 2 else 1)       # e^-a I0(a) is the mean of one pulse per period
    return y - np.mean(y)


def noise(n: int, color: str = "white", seed: int | None = None, sr: int = DEFAULT_SR) -> np.ndarray:
    if color == "white":
        return white(n, seed)
    if color == "pink":
        return pink(n, seed)
    if color == "brown":
        return brown(n, seed, sr)
    raise ValueError(f"unknown noise color {color}")


def osc(wave: str, freq, n: int, sr: int = DEFAULT_SR, **kw) -> np.ndarray:
    """Generic dispatcher used by the synth layer and the sfx generators."""
    wave = wave.lower()
    if wave == "sine":
        return sine(freq, n, sr)
    if wave == "saw":
        return saw(freq, n, sr)
    if wave == "square":
        return square(freq, n, sr)
    if wave == "pulse":
        return pulse(freq, n, sr, width=float(kw.get("width", 0.25)))
    if wave in ("triangle", "tri"):
        return triangle(freq, n, sr)
    if wave == "supersaw":
        return supersaw(freq, n, sr, voices=int(kw.get("voices", 7)), detune=float(kw.get("detune", 0.4)))
    if wave == "fm":
        return fm(freq, n, sr, ratio=float(kw.get("ratio", 2.0)), index=kw.get("index", 1.0))
    if wave == "organ":
        return organ(freq, n, sr)
    if wave == "wavefold":
        return wavefold(freq, n, sr, amount=float(kw.get("amount", 2.0)))
    if wave == "jp8000":
        return supersaw(freq, n, sr, mode="jp8000", spread=float(kw.get("detune", 0.4)) / 0.4)
    if wave == "cz":
        return phase_distortion(freq, n, sr, amount=float(kw.get("amount", 0.5)))
    if wave == "gbuzz":
        return gbuzz(freq, n, sr, nharm=int(kw.get("nharm", 8)), mul=float(kw.get("mul", 0.8)))
    if wave.startswith("dpw_"):
        return dpw(freq, n, sr, order=int(kw.get("order", 3)), wave=wave[4:])
    if wave == "blit":
        return blit(freq, n, sr, kernel=str(kw.get("kernel", "sinc")))
    if wave == "blit_saw":
        return blit_saw(freq, n, sr, kernel=str(kw.get("kernel", "sinc")))
    if wave == "blit_square":
        return blit_square(freq, n, sr, kernel=str(kw.get("kernel", "sinc")))
    if wave == "sync":
        return hard_sync(freq, n, sr, ratio=float(kw.get("ratio", 2.5)))
    if wave.startswith("vco2_"):
        return vco2(freq, n, sr, wave=wave[5:], width=kw.get("width", 0.5))
    if wave == "squine":
        return squinewave(freq, n, sr, clip=kw.get("clip", 0.5), skew=kw.get("skew", 0.0))
    # Noise is seeded (0 unless the caller passes one): the same spec always renders the same file.
    seed = kw.get("seed")
    seed = 0 if seed is None else int(seed)
    if wave == "gendyn":
        return gendyn(freq, n, sr, seed=seed)
    if wave == "fm_tree":
        return fm_tree(freq, n, sr, seed=seed, index=float(kw.get("index", 2.0)))
    if wave == "pitched_noise":
        return pitched_noise(freq, n, sr, color=str(kw.get("color", "white")), seed=seed)
    if wave == "held_noise":
        return held_noise(n, float(np.max(_freq_array(freq, n))), sr, seed=seed)
    if wave in ("noise", "white"):
        return white(n, seed)
    if wave == "pink":
        return pink(n, seed)
    if wave == "brown":
        return brown(n, seed, sr)
    raise ValueError(f"unknown wave {wave!r}; choose from {WAVES}")


def sweep(start_hz: float, end_hz: float, dur: float, sr: int = DEFAULT_SR, curve: str = "exp") -> np.ndarray:
    """Per-sample frequency array from start to end. curve: 'exp' (musical) or 'lin'."""
    n = samples(dur, sr)
    if curve == "lin":
        return np.linspace(start_hz, end_hz, n)
    start_hz = max(start_hz, 1e-3)
    end_hz = max(end_hz, 1e-3)
    return start_hz * (end_hz / start_hz) ** np.linspace(0, 1, n)


def sync_phasor(freq, n: int, sr: int = DEFAULT_SR, sync=None, phase0: float = 0.0):
    """Phasor (0..1) with a sync input and output: it restarts at 0 on every sample where ``sync`` > 0, and returns
    (phase, sync_out) with sync_out = 1 where it wrapped or restarted, ready to drive another phasor."""
    f = _freq_array(freq, n)
    E = np.concatenate(([0.0], np.cumsum(f / sr)[:-1]))
    reset = np.zeros(n, bool) if sync is None else pad_bool(sync, n)
    last = np.maximum.accumulate(np.where(reset, np.arange(n), -1))
    u = np.where(last >= 0, E - E[np.maximum(last, 0)], E + phase0)
    wrap = np.zeros(n, bool)
    wrap[1:] = np.floor(u[1:]) > np.floor(u[:-1])
    return u % 1.0, (wrap | reset).astype(np.float64)


def pad_bool(x, n: int) -> np.ndarray:
    x = np.asarray(x, float)[:n] > 0
    return np.concatenate((x, np.zeros(n - x.size, bool)))


def squinewave(freq, n: int, sr: int = DEFAULT_SR, clip=0.0, skew=0.0, minsweep: float | None = None,
               phase0: float = 0.0) -> np.ndarray:
    """Sine-square-saw morph: each cycle is a rising half-cosine sweep, a hold at +1, a falling sweep and a hold at
    -1. ``clip`` 0..1 shortens the sweeps (sine to square), ``skew`` -1..1 moves the turning point (towards a
    ramp); no sweep is shorter than ``minsweep`` samples (default clip(sr / 3000, 8, sr / 100)), which keeps the
    edges band-limited. Both accept per-sample arrays."""
    f = np.abs(_freq_array(freq, n))
    t = phase(f, n, sr, phase0) % 1.0
    ms = float(minsweep) if minsweep else float(np.clip(sr / 3000.0, 8.0, sr * 0.01))
    msw = np.clip(ms * f / sr, 1e-9, 0.5)
    c = np.clip(_freq_array(clip, n), 0.0, 1.0)
    s = np.clip(_freq_array(skew, n), -1.0, 1.0)
    h1 = np.clip((1.0 + s) / 2.0, msw, 1.0 - msw)
    h2 = 1.0 - h1
    w1 = np.clip((1.0 - c) * h1, msw, h1)
    w2 = np.clip((1.0 - c) * h2, msw, h2)
    return np.where(t < w1, -np.cos(np.pi * t / w1),
                    np.where(t < h1, 1.0, np.where(t < h1 + w2, np.cos(np.pi * (t - h1) / w2), -1.0)))


@numba.njit(cache=True)
def _vactrol(x, a_base, t_up, t_down):
    y = np.empty(x.size)
    s1 = 0.0
    for i in range(x.size):
        d = x[i] - s1
        if d >= 0.0:
            a = a_base / (1.0 + t_up * (1.0 - 0.999 * s1))
        else:
            a = a_base / (10.0 + t_down * (1.0 - 0.9 * s1))
        v = d * a / (1.0 + a)
        y[i] = v + s1
        s1 = y[i] + v
    return y


def vactrol(x, sr: int = DEFAULT_SR, up: float = 20.0, down: float = 3000.0) -> np.ndarray:
    """Vactrol (lamp + photoresistor) response of a control signal (0..1): a one-pole lowpass whose time constant
    depends on its own state, 1 + up (1 - 0.999 s) ms rising and 10 + down (1 - 0.9 s) ms falling, so it opens fast
    and lets go slowly with a long tail (the low-pass-gate decay)."""
    return _vactrol(np.ascontiguousarray(x, dtype=np.float64), 1000.0 * np.pi / sr, float(up), float(down))


def exp_shape(x, index: float = 3.0) -> np.ndarray:
    """Exponential waveshaper exp(index (x - 1)): 1 at x = 1, a sine in gives the modified Bessel spectrum
    I_k(index) (brighter as index grows); not DC-free."""
    return np.exp(float(index) * (np.asarray(x, float) - 1.0))


def quartic_line(levels, times, n: int, sr: int = DEFAULT_SR) -> np.ndarray:
    """Envelope through ``levels`` with segment durations ``times`` (s), drawn straight on the fourth root of the
    level and raised to the fourth power: quick, click-free rises and natural-sounding decays."""
    lv = np.asarray(levels, float) ** 0.25
    xs = np.concatenate(([0.0], np.cumsum(np.asarray(times, float)))) * sr
    return np.interp(np.arange(n), xs[:lv.size], lv) ** 4


FM_TREE_RATIOS = (0.5, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 7.0, 1.4142, 3.5, 0.25)


def fm_tree_spec(seed: int = 0, operators: int = 4, carriers: int = 1, index: float = 2.0) -> list:
    """Random phase-modulation tree: a list of (parent, ratio, level); carriers have parent -1 and level 1,
    every other operator modulates an earlier one with an index drawn in [0.2, 1] x ``index``."""
    rng = np.random.default_rng(int(seed))
    ops = []
    for i in range(max(1, int(operators))):
        if i < max(1, int(carriers)):
            ops.append((-1, 1.0 if i == 0 else float(rng.choice((0.5, 1.0, 2.0, 3.0))), 1.0))
        else:
            ops.append((int(rng.integers(0, i)), float(rng.choice(FM_TREE_RATIOS)), float(rng.uniform(0.2, 1.0) * index)))
    return ops


def fm_tree(freq, n: int, sr: int = DEFAULT_SR, tree=None, seed: int = 0, operators: int = 4, carriers: int = 1,
            index: float = 2.0, phase0: float = 0.0) -> np.ndarray:
    """Phase-modulation network from ``tree`` (see ``fm_tree_spec``, drawn from ``seed`` when None): each operator
    is a sine at ratio x freq plus the sum of its modulators; the carriers are averaged. One seed = one timbre
    family, the pitch free."""
    f = _freq_array(freq, n)
    tree = tree or fm_tree_spec(seed, operators, carriers, index)
    mod = [np.zeros(n) for _ in tree]
    out = np.zeros(n)
    for i in range(len(tree) - 1, -1, -1):
        parent, ratio, level = tree[i]
        y = np.sin(2 * np.pi * phase(f * ratio, n, sr, phase0) + mod[i])
        if parent < 0:
            out += y
        else:
            mod[parent] += level * y
    return out / sum(1 for p in tree if p[0] < 0)


CHAOS_MAPS = ("logistic", "henon", "lorenz", "latoocarfian", "cusp", "gingerbreadman", "standard", "quad", "fbsine",
              "lincong", "crackle")
CHAOS_PARAMS = {"logistic": (3.9,), "henon": (1.4, 0.3), "lorenz": (10.0, 28.0, 8.0 / 3.0, 0.01),
                "latoocarfian": (1.0, 3.0, 0.5, 0.5), "cusp": (1.0, 1.9), "gingerbreadman": (), "standard": (1.0,),
                "quad": (1.0, -1.0, -0.75), "fbsine": (1.0, 0.1, 1.1, 0.5), "lincong": (1.1, 0.13, 1.0),
                "crackle": (1.5,)}
CHAOS_INIT = {"logistic": (0.5, 0.0), "henon": (0.0, 0.0), "lorenz": (0.1, 0.0), "latoocarfian": (0.5, 0.5),
              "cusp": (0.0, 0.0), "gingerbreadman": (1.2, 2.1), "standard": (0.5, 0.0), "quad": (0.0, 0.0),
              "fbsine": (0.1, 0.1), "lincong": (0.0, 0.0), "crackle": (0.3, 0.0)}


@numba.njit(cache=True)
def _lorenz_d(x, y, z, s, r, b):
    return s * (y - x), x * (r - z) - y, x * y - b * z


@numba.njit(cache=True)
def _chaos_run(kind, m, p, x0, y0):
    out = np.empty(m)
    x, y, z = x0, y0, 0.0
    for i in range(m):
        if kind == 0:
            x = p[0] * x * (1.0 - x)
        elif kind == 1:
            xn = 1.0 - p[0] * x * x + p[1] * y
            y = x
            x = xn
        elif kind == 2:
            h = p[3]
            k1 = _lorenz_d(x, y, z, p[0], p[1], p[2])
            k2 = _lorenz_d(x + 0.5 * h * k1[0], y + 0.5 * h * k1[1], z + 0.5 * h * k1[2], p[0], p[1], p[2])
            k3 = _lorenz_d(x + 0.5 * h * k2[0], y + 0.5 * h * k2[1], z + 0.5 * h * k2[2], p[0], p[1], p[2])
            k4 = _lorenz_d(x + h * k3[0], y + h * k3[1], z + h * k3[2], p[0], p[1], p[2])
            x += h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
            y += h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
            z += h / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        elif kind == 3:
            xn = math.sin(p[1] * y) + p[2] * math.sin(p[1] * x)
            y = math.sin(p[0] * x) + p[3] * math.sin(p[0] * y)
            x = xn
        elif kind == 4:
            x = p[0] - p[1] * math.sqrt(abs(x))
        elif kind == 5:
            xn = 1.0 - y + abs(x)
            y = x
            x = xn
        elif kind == 6:
            y = (y + p[0] * math.sin(x) + math.pi) % (2.0 * math.pi) - math.pi
            x = (x + y + math.pi) % (2.0 * math.pi) - math.pi
        elif kind == 7:
            x = p[0] * x * x + p[1] * x + p[2]
        elif kind == 8:
            xn = math.sin(p[0] * y + p[1] * x)
            y = (p[2] * y + p[3]) % (2.0 * math.pi)
            x = xn
        elif kind == 9:
            x = (p[0] * x + p[1]) % p[2]
        else:
            xn = abs(x * p[0] - y - 0.05)
            y = x
            x = xn
        if not (abs(x) < 1e6) or not (abs(y) < 1e6) or not (abs(z) < 1e6):
            x, y, z = x0, y0, 0.0
        out[i] = x
    return out


def chaos(kind: str, rate, n: int, sr: int = DEFAULT_SR, interp: str = "cubic", params=None, x0=None,
          y0=None) -> np.ndarray:
    """Chaotic map iterated ``rate`` times a second (scalar or per sample) and read as audio: ``interp`` none
    (held), linear or cubic between iterations. ``kind`` logistic (r), henon (a, b), lorenz (s, r, b, h; RK4),
    latoocarfian (a, b, c, d), cusp (a, b), gingerbreadman, standard (k), quad (a, b, c), fbsine (im, fb, a, c),
    lincong (a, c, m), crackle (p); ``params`` overrides the defaults in order. Mean removed, peak 1."""
    if kind not in CHAOS_MAPS:
        raise ValueError(f"chaos kind: {' | '.join(CHAOS_MAPS)}")
    r = np.abs(_freq_array(rate, n))
    pos = np.concatenate(([0.0], np.cumsum(r / sr)[:-1]))
    m = int(pos[-1]) + 4
    p = list(CHAOS_PARAMS[kind]) + [0.0] * 4
    for i, v in enumerate(params or ()):
        p[i] = float(v)
    ix, iy = CHAOS_INIT[kind]
    seq = _chaos_run(CHAOS_MAPS.index(kind), m, np.asarray(p[:4], float),
                     float(ix if x0 is None else x0), float(iy if y0 is None else y0))
    seq = seq - seq.mean()
    seq /= max(float(np.max(np.abs(seq))), 1e-12)
    if interp == "none":
        return seq[pos.astype(np.int64)]
    return table_read(np.concatenate((seq[:1], seq, seq[-2:])), pos + 1.0, "linear" if interp == "linear" else "lagrange")


@numba.njit(cache=True)
def _scan_run(x, v, ei, ej, k, c, d, F):
    M = x.size
    frames = np.empty((F, M))
    acc = np.empty(M)
    for s in range(F):
        frames[s] = x
        for i in range(M):
            acc[i] = -c * x[i] - d * v[i]
        for e in range(ei.size):
            dx = k * (x[ej[e]] - x[ei[e]])
            acc[ei[e]] += dx
            acc[ej[e]] -= dx
        for i in range(M):
            v[i] += acc[i]
            x[i] += v[i]
    return frames


def scanned(freq, n: int, sr: int = DEFAULT_SR, masses: int = 128, rate: float = 200.0, stiffness: float = 0.3,
            centering: float = 0.0, t60: float = 2.0, shape: str = "hammer", position: float = 0.3,
            width: float = 0.15, connect: str = "ring", links: int = 0, seed: int = 0, top: float | None = None) -> np.ndarray:
    """Scanned synthesis: a network of ``masses`` unit masses on springs (``connect`` ring, line, or ring plus
    ``links`` random springs) moves slowly, ``rate`` updates per second (semi-implicit Euler, one step per
    update); its displacement profile, read in mass order, is the wavetable played at ``freq``. Damping opposes
    the velocity and is set by the motion's ``t60`` (s); ``stiffness`` and ``centering`` are fractions of the
    stability limit (k deg + c < 4). Initial ``shape``: hammer (raised cosine at ``position``, ``width``), pluck
    (triangle) or noise. Each frame is band-limited to the pitch (harmonics under 0.45 sr, or under ``top`` Hz) and crossfaded to the next."""
    M = max(8, int(masses))
    f = _freq_array(freq, n)
    idx = np.arange(M)
    ei, ej = (idx, (idx + 1) % M) if connect != "line" else (idx[:-1], idx[1:])
    rng = np.random.default_rng(int(seed))
    if links:
        ei = np.concatenate((ei, rng.integers(0, M, int(links))))
        ej = np.concatenate((ej, rng.integers(0, M, int(links))))
    deg = np.bincount(np.concatenate((ei, ej)), minlength=M).max()
    c = float(np.clip(centering, 0.0, 1.0)) * 0.5
    k = float(np.clip(stiffness, 0.0, 1.0)) * (3.9 - c) / max(deg, 1)
    d = min(13.8 / (max(float(t60), 1e-3) * float(rate)), 1.0)
    u = idx / M
    if shape == "hammer":
        x0 = 0.5 + 0.5 * np.cos(np.clip((u - position) / max(width, 1e-3), -1.0, 1.0) * np.pi)
    elif shape == "pluck":
        x0 = np.where(u < position, u / max(position, 1e-3), (1.0 - u) / max(1.0 - position, 1e-3))
    elif shape == "noise":
        x0 = rng.uniform(-1.0, 1.0, M)
    else:
        raise ValueError("scanned shape: hammer | pluck | noise")
    F = int(n * float(rate) / sr) + 3
    fr = _scan_run(x0.astype(np.float64), np.zeros(M), ei.astype(np.int64), ej.astype(np.int64), k, c, d, F)
    T = max(256, 1 << int(np.ceil(np.log2(M))))
    X = np.fft.rfft(fr - fr.mean(axis=1, keepdims=True), axis=1)
    lim = MIP_LIMIT * sr if top is None else min(MIP_LIMIT * sr, float(top))
    X[:, max(1, int(lim / max(float(np.max(f)), 1e-6))) + 1:] = 0.0
    tabs = np.fft.irfft(X, T, axis=1) * (T / M)
    tabs /= max(float(np.max(np.abs(tabs[0]))), 1e-12)
    pos = (phase(f, n, sr) % 1.0) * T
    q = np.arange(n) * float(rate) / sr
    i = q.astype(np.int64)
    w = q - i
    return (1.0 - w) * table_read(tabs, pos, rows=i) + w * table_read(tabs, pos, rows=i + 1)


def loop_sampler(x, n: int, rate=1.0, start: float = 0.0, loop_start: float | None = None, loop_end: float | None = None,
                 mode: str = "forward", release: int | None = None, xfade: int = 0) -> np.ndarray:
    """Plays mono sample ``x`` for ``n`` samples at ``rate`` (scalar or per sample) from index ``start``; once past
    ``loop_end`` it cycles between ``loop_start`` and ``loop_end`` (``forward`` or ``pingpong``). At sample ``release``
    it leaves the loop and plays on to the end of the sample. ``xfade`` samples (forward): the end of the loop is
    crossfaded (equal power) with the samples before ``loop_start``, so the joint does not click."""
    from .reverbs import read_frac
    x = np.asarray(x, float)
    L = len(x)
    ls = 0.0 if loop_start is None else float(loop_start)
    le = float(L - 1) if loop_end is None else float(loop_end)
    p = float(start) + np.concatenate(([0.0], np.cumsum(_freq_array(rate, n)[:-1])))
    seg = p.copy()
    W = max(le - ls, 1.0)
    over = p >= le
    if mode == "pingpong":
        u = (p[over] - ls) % (2.0 * W)
        seg[over] = ls + np.where(u < W, u, 2.0 * W - u)
    else:
        seg[over] = ls + (p[over] - ls) % W
    if release is not None and 0 <= int(release) < n:
        r = int(release)
        seg[r:] = seg[r] + (p[r:] - p[r])
    y = read_frac(x, seg)
    xf = min(int(xfade), int(ls), int(W))
    if xf > 0 and mode != "pingpong":
        d = le - seg
        m = (d < xf) & (seg >= ls) & (np.arange(n) < (n if release is None else int(release))) & (p >= ls)
        g = np.clip(d[m] / xf, 0.0, 1.0)
        y[m] = np.sqrt(g) * y[m] + np.sqrt(1.0 - g) * read_frac(x, seg[m] - W)
    return y
