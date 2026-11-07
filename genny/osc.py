"""Oscillators and noise. All accept a scalar frequency or a per-sample frequency array (for sweeps)."""
from __future__ import annotations

import numpy as np
from numpy.polynomial import Polynomial as _Poly
from scipy.signal import lfilter

from .core import DEFAULT_SR, samples

WAVES = ["sine", "saw", "square", "pulse", "triangle", "supersaw", "fm", "noise", "pink", "brown", "organ", "wavefold", "gendyn",
         "jp8000", "cz", "gbuzz", "dpw_saw", "dpw_triangle", "dpw_square", "blit", "blit_saw", "blit_square", "sync",
         "pitched_noise", "held_noise"]


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


def gendyn(freq, n: int, sr: int = DEFAULT_SR, points: int = 12, amp_step: float = 0.5, dur_spread: float = 0.0,
           seed: int = 0, curve: float = 0.0) -> np.ndarray:
    """Dynamic stochastic synthesis: a cycle of ``points`` breakpoints, each with an amplitude and a segment length
    that move every cycle by a second-order random walk (a random walk sets the step of another), both walls
    reflecting. Amplitudes stay in [-1, 1] with steps in [-amp_step, amp_step]; segment lengths in
    (1 +- dur_spread) of period / points, their steps in +- dur_spread / 2 of it, so the pitch wanders
    around ``freq`` (0: fixed pitch, the timbre alone walks). Segments are straight lines, or with ``curve`` > 0
    exponential bends a + (b - a)(1 - e^(c u)) / (1 - e^c), c drawn per segment in [-curve, curve]."""
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
        da = reflect(da + rng.uniform(-b, b, P), -b, b)
        a = reflect(a + da, -1.0, 1.0)
        du = reflect(du + rng.uniform(-s / 2, s / 2, P), -s / 2, s / 2)
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
    """Closed-form buzz: ``nharm`` cosine harmonics from ``lharm``, each ``mul`` times the one below, scaled by
    (1 - |r|) / (1 - |r^n|) (1 / n when |r| = 1). nharm is cut so the top harmonic stays under Nyquist."""
    f = _freq_array(freq, n)
    k = max(int(lharm), 1)
    nh = max(1, min(int(nharm), int(sr / 2 / max(float(np.max(f)), 1e-6)) - k + 1))
    r = float(mul)
    p = 2 * np.pi * phase(f, n, sr, phase0)
    if abs(abs(r) - 1.0) < 1e-6:
        return sum(r ** i * np.cos((k + i) * p) for i in range(nh)) / nh
    num = np.cos(k * p) - r * np.cos((k - 1) * p) - r ** nh * np.cos((k + nh) * p) + r ** (nh + 1) * np.cos((k + nh - 1) * p)
    return num / (1.0 + r * r - 2.0 * r * np.cos(p)) * (1.0 - abs(r)) / (1.0 - abs(r ** nh))


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
    sin(M pi x) / (P sin(pi x)) with M odd = 2 floor(P / 2) + 1 harmonics (``harmonics`` > 0 sets fewer);
    bspline: each pulse a cubic B-spline at its exact fractional position (spectrum sinc^4, -3 dB at 10 kHz at
    44.1 kHz); thiran: the pulse on a sample, its fractional part D in [0.418, 1.418] by a first-order allpass
    a1 = (1 - D) / (1 + D)."""
    f = _freq_array(freq, n)
    ph = phase(f, n, sr, phase0)
    if kernel == "sinc":
        P = sr / np.maximum(f, 1e-6)
        M = 2 * np.floor(P / 2) + 1
        if harmonics:
            M = np.minimum(M, 2 * int(harmonics) + 1)
        x = ph % 1.0
        den = np.sin(np.pi * x)
        small = np.abs(den) < 1e-9
        return np.where(small, M, np.sin(M * np.pi * x) / np.where(small, 1.0, den)) / P
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


def wavetable_osc(freq, n: int, sr: int = DEFAULT_SR, harmonics=(1.0,), phase0: float = 0.0) -> np.ndarray:
    """Table oscillator read with linear interpolation; harmonics above 0.45 sr are dropped from the table at the
    highest frequency of `freq`."""
    f = _freq_array(freq, n)
    h = np.asarray(harmonics, float)
    keep = max(1, int(0.45 * sr / max(float(np.max(f)), 1e-9)))
    tab = wavetable_from_spectrum(h[:keep])
    ph = (phase(f, n, sr, phase0) % 1.0) * WAVETABLE_N
    return np.interp(ph, np.arange(WAVETABLE_N + 1), np.append(tab, tab[0]))


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
    # Noise is seeded (0 unless the caller passes one): the same spec always renders the same file.
    seed = kw.get("seed")
    seed = 0 if seed is None else int(seed)
    if wave == "gendyn":
        return gendyn(freq, n, sr, seed=seed)
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
