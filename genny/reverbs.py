"""Algorithmic reverberation, spatial and delay-modulation effects.

Sources (findings files in ``out/research``; PASP = J.O. Smith, *Physical Audio Signal Processing*):

* Schroeder reverberators JCREV / SATREV / Samson-box JCREV with the delay lengths and gains read
  from PASP Figs. 3.5-3.7 (g13 §1.1-1.3), tuned at fs = 25 kHz and rescaled to mutually prime lengths.
* Feedback delay network (Jot): Householder / Hadamard feedback, mean-free-path prime lengths,
  first-order per-line damping, tonal correction (PASP ch. 3; findings 10 §2.4, g13 §13.5).
* Early reflections: shoebox image-source model (Allen & Berkley 1979) with Sabine absorption per
  material (Farnell *Designing Sound* Table 5.1, in ``genny/physical/space.py``), "first ~100 ms" as a
  tapped delay line (PASP/Early_Reflections, findings 10 §2.1); stereo pick-up = two opposite
  cardioids ``(1 + cos mu)`` (Takala & Hahn, *Sound Rendering*, findings 04 / g15 §11.10).
* Existing kernels exposed: ``zita_rev1``, ``puckette_rev``, ``farnell_room`` (``physical/space.py``).
* Propagation: 1/r spreading, air absorption (PASP Table B.2), Doppler from a time-varying delay
  (PASP ch. 5; g13 §3.2, findings 10 §10), occlusion (Farnell p.87; *Sound Rendering* "occluders damp
  smoothly rather than cut").
* Delay-line interpolation: 4th-order (even order) Lagrange for every swept delay (g13 §2.1 rule).
* Flanger, phaser, chorus, Leslie (PASP ch. 5 and 8.9; g13 §3.1, §3.3, §3.4, §4.1).
* Electric-guitar feedback chain ("Sullivan" model, PASP ch. 9.1 Fig. 9.4; g13 §8).

Numbers without a source in those files are marked ``# UNSOURCED``.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numba
import numpy as np
from scipy.signal import butter, fftconvolve, lfilter, resample_poly, sosfilt

from .core import DEFAULT_SR, samples, to_mono, to_stereo
from .fx import effect
from .instruments import instrument
from .physical import space as _space
from .physical.core import tv_one_pole_lp as _tv_one_pole

C_AIR = _space.C_AIR            # 345 m/s (PASP, 22 C)
_PHYS_SR = 48000                # rate of the kernels in physical/space.py


# ==========================================================================================
# small helpers
# ==========================================================================================

def _resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if int(sr_from) == int(sr_to):
        return np.asarray(x, float)
    g = math.gcd(int(sr_from), int(sr_to))
    return resample_poly(np.asarray(x, float), int(sr_to) // g, int(sr_from) // g, axis=0)


def _pad(x: np.ndarray, n_extra: int, front: int = 0) -> np.ndarray:
    shape = x.shape[1:]
    return np.concatenate([np.zeros((front,) + shape), x, np.zeros((max(0, n_extra),) + shape)], axis=0)


def _fade_end(y: np.ndarray, sr: int, ms: float = 5.0, head_ms: float = 0.0) -> np.ndarray:
    y = np.array(y, dtype=np.float64)
    k = min(y.shape[0] // 2, int(sr * ms / 1000))
    if k > 1:
        r = np.linspace(1.0, 0.0, k)
        y[-k:] *= r[:, None] if y.ndim == 2 else r
    h = min(y.shape[0] // 2, int(sr * head_ms / 1000))
    if h > 1:
        r = np.linspace(0.0, 1.0, h)
        y[:h] *= r[:, None] if y.ndim == 2 else r
    return y


def _lr(x: np.ndarray):
    x = np.asarray(x, float)
    if x.ndim == 1:
        return np.ascontiguousarray(x), np.ascontiguousarray(x)
    return np.ascontiguousarray(x[:, 0]), np.ascontiguousarray(x[:, 1])


def _ms_width(wl, wr, width):
    width = float(np.clip(width, 0.0, 1.5))
    mid, side = 0.5 * (wl + wr), 0.5 * (wl - wr) * width
    return np.stack([mid + side, mid - side], axis=1)


def _wetdry(dry: np.ndarray, wet: np.ndarray, mix: float) -> np.ndarray:
    mix = float(np.clip(mix, 0.0, 1.0))
    if wet.ndim == 2 and dry.ndim == 1:
        dry = to_stereo(dry)
    if wet.ndim == 1 and dry.ndim == 2:
        wet = to_stereo(wet)
    return dry * (1.0 - mix) + wet * mix


def _lfo(n: int, sr: int, rate: float, shape: str = "sine", phase: float = 0.0) -> np.ndarray:
    """LFO in [-1, 1]; ``phase`` in cycles. sine | triangle."""
    ph = (np.arange(n) * (float(rate) / sr) + phase) % 1.0
    if shape in ("triangle", "exp"):
        return 1.0 - 4.0 * np.abs(ph - 0.5)
    return np.sin(2 * np.pi * ph)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _nearest_prime(n: int, taken=()) -> int:
    n = max(2, int(n))
    for k in range(0, 100000):
        for c in (n - k, n + k):
            if c >= 2 and c not in taken and _is_prime(c):
                return c
    raise ValueError("no prime found")


def coprime_lengths(lengths, scale: float) -> list[int]:
    """Scale delay lengths and keep them mutually prime (PASP: "nearest prime is the safe choice").
    With ``scale == 1`` the published values are returned untouched."""
    if abs(scale - 1.0) < 1e-12:
        return [int(v) for v in lengths]
    out: list[int] = []
    for v in lengths:
        out.append(_nearest_prime(int(round(v * scale)), out))
    return out


# ==========================================================================================
# interpolation kernels (4th-order Lagrange, nodes -2..2 around the nearest sample; g13 §2.1)
# ==========================================================================================

@numba.njit(cache=True)
def _l4w(d, w):
    """Lagrange weights for evaluating at offset d in [-0.5, 0.5] from the centre node."""
    w[0] = (d + 1.0) * d * (d - 1.0) * (d - 2.0) / 24.0
    w[1] = -(d + 2.0) * d * (d - 1.0) * (d - 2.0) / 6.0
    w[2] = (d + 2.0) * (d + 1.0) * (d - 1.0) * (d - 2.0) / 4.0
    w[3] = -(d + 2.0) * (d + 1.0) * d * (d - 2.0) / 6.0
    w[4] = (d + 2.0) * (d + 1.0) * d * (d - 1.0) / 24.0


@numba.njit(cache=True)
def _read_l4(x, pos):
    """y[k] = x(pos[k]) (fractional index, zero outside the buffer)."""
    n_in = x.shape[0]
    m = pos.shape[0]
    y = np.zeros(m)
    w = np.zeros(5)
    for k in range(m):
        p = pos[k]
        ic = int(np.floor(p + 0.5))
        if ic < 2 or ic > n_in - 3:
            i0 = int(np.floor(p))
            f = p - i0
            a = x[i0] if (i0 >= 0 and i0 < n_in) else 0.0
            b = x[i0 + 1] if (i0 + 1 >= 0 and i0 + 1 < n_in) else 0.0
            y[k] = (1.0 - f) * a + f * b
        else:
            _l4w(p - ic, w)
            y[k] = (w[0] * x[ic - 2] + w[1] * x[ic - 1] + w[2] * x[ic] + w[3] * x[ic + 1] + w[4] * x[ic + 2])
    return y


@numba.njit(cache=True)
def _ring_l4(buf, L, wpos, d, w):
    """Read a circular buffer ``d`` samples (>= 3.5) behind the write index ``wpos``."""
    p = wpos - d
    ic = int(np.floor(p + 0.5))
    _l4w(p - ic, w)
    acc = 0.0
    for j in range(5):
        i = (ic - 2 + j) % L
        if i < 0:
            i += L
        acc += w[j] * buf[i]
    return acc


# ==========================================================================================
# Schroeder reverberators: JCREV, SATREV, Samson-box JCREV  (g13 §1.1-1.3)
# ==========================================================================================

SCHROEDER_FS = 25000.0   # "All three CCRMA reverbs were tuned by ear at fs = 25 kHz" (g13 §1)
SCHROEDER = {
    # PASP Fig. 3.5: AP(347, .7) AP(113, .7) AP(37, .7) -> 4 parallel FBCF -> mixing matrix
    "jcrev": {"ap": (347, 113, 37), "ap_g": 0.7, "comb": (1687, 1601, 2053, 2251),
              "comb_g": (0.773, 0.802, 0.753, 0.733)},
    # PASP Fig. 3.6: 4 parallel FBCF summed -> AP(125, .7) AP(42, .7) AP(12, .7)
    "satrev": {"ap": (125, 42, 12), "ap_g": 0.7, "comb": (901, 778, 1011, 1123),
               "comb_g": (0.805, 0.827, 0.783, 0.764)},
    # PASP Fig. 3.7: AP(1051, .7) AP(337, .7) AP(113, .7) -> 4 combs -> output delays .046/.057/.041/.054 s.
    # The figure labels the combs FFCF, JOS's text and STK JCRev use feedback combs: feedback is used here.
    "samson": {"ap": (1051, 337, 113), "ap_g": 0.7, "comb": (4799, 4999, 5399, 5801),
               "comb_g": (0.742, 0.733, 0.715, 0.697), "out_delay": (0.046, 0.057, 0.041, 0.054)},
}


def schroeder_lengths(name: str, sr: int = DEFAULT_SR) -> dict:
    """Allpass and comb lengths of ``jcrev`` | ``satrev`` | ``samson`` at ``sr``: the published values
    at 25 kHz, otherwise scaled by sr/25000 and moved to the nearest unused prime (so all seven
    lengths are pairwise coprime)."""
    spec = SCHROEDER[name]
    allv = coprime_lengths(spec["ap"] + spec["comb"], sr / SCHROEDER_FS)
    return {"ap": allv[:3], "comb": allv[3:]}


@numba.njit(cache=True)
def _ap_series(x, D, g):
    """Series Schroeder allpasses AP_N^g = (-g + z^-N) / (1 - g z^-N)."""
    y = x.copy()
    n = x.shape[0]
    for k in range(D.shape[0]):
        d = D[k]
        gk = g[k]
        buf = np.zeros(d)
        p = 0
        for i in range(n):
            v = buf[p]
            w = y[i] + gk * v
            y[i] = v - gk * w
            buf[p] = w
            p += 1
            if p >= d:
                p = 0
    return y


@numba.njit(cache=True)
def _fbcf_bank(x, D, g, damp):
    """Parallel feedback combs FBCF_N^g = 1 / (1 - g z^-N); ``damp`` adds a one-pole low-pass in the
    loop (0 = the published filter). Returns (len(D), n)."""
    n = x.shape[0]
    out = np.zeros((D.shape[0], n))
    for k in range(D.shape[0]):
        d = D[k]
        gk = g[k]
        buf = np.zeros(d)
        p = 0
        s = 0.0
        for i in range(n):
            s = (1.0 - damp) * buf[p] + damp * s
            v = x[i] + gk * s
            buf[p] = v
            out[k, i] = v
            p += 1
            if p >= d:
                p = 0
    return out


def _comb_gains(spec, combs, sr, t60):
    g = np.asarray(spec["comb_g"], float)
    if t60 is not None and t60 > 0:
        g = 10.0 ** (-3.0 * np.asarray(combs, float) / (sr * float(t60)))
    return np.clip(g, 0.0, 0.98)


def _schroeder_wet(mono, sr, variant, t60, damp):
    """Returns (A, B) decorrelated wet outputs, each with unit-energy impulse response."""
    spec = SCHROEDER[variant]
    L = schroeder_lengths(variant, sr)
    apd = np.array(L["ap"], np.int64)
    cd = np.array(L["comb"], np.int64)
    apg = np.full(3, spec["ap_g"])
    cg = _comb_gains(spec, cd, sr, t60)
    damp = float(np.clip(damp, 0.0, 0.95))
    norm = 1.0 / math.sqrt(float(np.sum(1.0 / (1.0 - cg ** 2))))   # comb IR energy = 1 / (1 - g^2)
    mono = np.ascontiguousarray(mono, dtype=np.float64)
    if variant == "satrev":
        c = _fbcf_bank(mono, cd, cg, damp).sum(axis=0)
        a = _ap_series(c, apd, apg) * norm
        return a, -a                                               # OutB = -OutA
    c = _fbcf_bank(_ap_series(mono, apd, apg), cd, cg, damp)
    if variant == "jcrev":
        s1, s2 = c[0] + c[2], c[1] + c[3]
        return (s1 + s2) * norm, (s1 - s2) * norm                  # OutA, OutD
    rev = c.sum(axis=0) * norm
    od = spec["out_delay"]
    base = min(od)                                                 # JOS: the shortest can be subtracted
    da, db = int(round((od[0] - base) * sr)), int(round((od[1] - base) * sr))
    n = rev.shape[0]
    return np.concatenate([np.zeros(da), rev])[:n], np.concatenate([np.zeros(db), rev])[:n]


def _schroeder_tail(variant, sr, t60):
    spec = SCHROEDER[variant]
    cd = np.asarray(schroeder_lengths(variant, sr)["comb"], float)
    g = _comb_gains(spec, cd, sr, t60)
    return float(min(8.0, np.max(-3.0 * cd / (sr * np.log10(g)))))


@effect("jcrev", "Chowning's JCRev Schroeder reverb (3 series allpasses into 4 parallel combs, exact CCRMA lengths); `variant=samson` is the Samson-box version (longer, smoother). Stereo output.",
        mix=(0.3, "wet amount 0..1"),
        variant=("mus10", "mus10 (1972: AP 347/113/37, combs 1687/1601/2053/2251 at 25 kHz) | samson (AP 1051/337/113, combs 4799/4999/5399/5801)"),
        t60=(None, "decay time s; omit for the published comb gains (about 2 s mus10, 4.5 s samson)"),
        damp=(0.0, "extra high-frequency damping in the comb loops 0..0.95 (0 = published)"),
        width=(1.0, "stereo width 0..1 (1 = the two decorrelated outputs)"),
        predelay=(0.0, "seconds before the reverb starts"),
        tail=(None, "seconds of tail appended (auto = decay time)"))
def jcrev(x, sr=DEFAULT_SR, mix=0.3, variant="mus10", t60=None, damp=0.0, width=1.0, predelay=0.0, tail=None):
    """JCREV (PASP Fig. 3.5) and Samson-box JCREV (Fig. 3.7), g13 §1.1 / §1.3. L = OutA, R = OutD
    (the only two decorrelated outputs of the mixing matrix); samson: L/R = the 46 and 57 ms taps."""
    key = "samson" if str(variant).lower().startswith("sam") else "jcrev"
    x = np.asarray(x, float)
    if tail is None:
        tail = _schroeder_tail(key, sr, t60)
    pd = samples(predelay, sr) if predelay > 0 else 0
    dry = _pad(x, samples(tail, sr) + pd)
    mono = _pad(to_mono(x), samples(tail, sr), front=pd)
    a, b = _schroeder_wet(mono, sr, key, t60, damp)
    return _fade_end(_wetdry(dry, _ms_width(a, b, width), mix), sr)


@effect("satrev", "Chowning's SATREV (1971): 4 parallel combs into 3 series allpasses; small, bright, vintage. Mono reverb (channel count kept).",
        mix=(0.3, "wet amount 0..1"),
        t60=(None, "decay time s; omit for the published comb gains (about 1.2 s)"),
        damp=(0.0, "extra high-frequency damping in the comb loops 0..0.95 (0 = published)"),
        invert_right=(False, "true = the published two-output form, right = -left (stereo out; cancels in mono)"),
        predelay=(0.0, "seconds before the reverb starts"),
        tail=(None, "seconds of tail appended (auto = decay time)"))
def satrev(x, sr=DEFAULT_SR, mix=0.3, t60=None, damp=0.0, invert_right=False, predelay=0.0, tail=None):
    """SATREV, PASP Fig. 3.6 (g13 §1.2): FBCF(901, .805) FBCF(778, .827) FBCF(1011, .783)
    FBCF(1123, .764) summed -> AP(125, .7) AP(42, .7) AP(12, .7); OutB = -OutA."""
    x = np.asarray(x, float)
    if tail is None:
        tail = _schroeder_tail("satrev", sr, t60)
    pd = samples(predelay, sr) if predelay > 0 else 0
    dry = _pad(x, samples(tail, sr) + pd)
    a, b = _schroeder_wet(_pad(to_mono(x), samples(tail, sr), front=pd), sr, "satrev", t60, damp)
    wet = np.stack([a, b], axis=1) if invert_right else a
    return _fade_end(_wetdry(dry, wet, mix), sr)


# ==========================================================================================
# Feedback delay network (Jot)  (findings 10 §2.4, g13 §13.5)
# ==========================================================================================

def hadamard_matrix(n: int) -> np.ndarray:
    """Sylvester Hadamard matrix scaled by 1/sqrt(n) (orthogonal; n a power of 2)."""
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / math.sqrt(h.shape[0])


def householder_matrix(n: int) -> np.ndarray:
    """A_N = I - (2/N) 1 1^T (PASP/Householder_Feedback_Matrix)."""
    return np.eye(n) - (2.0 / n) * np.ones((n, n))


def feedback_matrix(kind: str, n: int) -> np.ndarray:
    """``householder``: I - (2/N) 1 1^T; for N = 16 Jot's balanced nested form
    A16 = 1/2 [[A4, -A4, -A4, -A4], ...] = A4 (x) A4 (the plain N = 16 reflection "degenerates to
    decoupled combs"). ``hadamard``: Sylvester / sqrt(N)."""
    kind = str(kind).lower()
    if kind.startswith("had"):
        return hadamard_matrix(n)
    if n == 16:
        a4 = householder_matrix(4)
        return np.kron(a4, a4)
    return householder_matrix(n)


def fdn_lengths(sr: int, size: float, lines: int) -> np.ndarray:
    """Mutually prime line lengths (samples), exponentially spaced with mean = the mean free path
    ``size`` metres / c (PASP/Mean_Free_Path: d = 4V/S; "Faust uses lines exponentially spaced")."""
    mean = max(float(size), 0.2) / C_AIR * sr
    r = 1.9 ** np.linspace(-1.0, 1.0, lines)          # UNSOURCED: spread 1/1.9 .. 1.9 around the mean
    r = r / r.mean()
    out: list[int] = []
    for v in r * mean:
        out.append(_nearest_prime(max(7, int(round(v))), out))
    return np.array(out, np.int64)


def fdn_damping(M, sr: int, t60_low: float, t60_high: float, f_low: float = 250.0, f_high: float = 4000.0):
    """First-order damping H_i = g_i / (1 - p_i z^-1) for each line of length M_i so that the loop
    attenuation gives ``t60_low`` at ``f_low`` and ``t60_high`` at ``f_high``.

    PASP/First_Order_Delay_Filter_Design fixes the two T60s at DC and Nyquist
    (20 log10 |H_i| = -60 M_i T / T60(w)); here the same one-pole is solved through two audible
    frequencies instead: |H|^2 = g^2 / (1 + p^2 - 2 p cos w) at w_l, w_h gives
    p^2 - 2 b p + 1 = 0, b = (G_l^2 c_l - G_h^2 c_h) / (G_l^2 - G_h^2). The DC gain is clamped below 1
    (T60 at DC at most 2 x t60_low) so the loop stays stable for any parameters.
    Returns (g, p, alpha) with alpha = T60(Nyquist) / T60(DC) for the tonal-correction filter."""
    M = np.asarray(M, float)
    t60_low = float(np.clip(t60_low, 0.05, 60.0))
    t60_high = float(np.clip(t60_high, 0.03, t60_low))
    f_high = min(float(f_high), 0.45 * sr)
    f_low = min(float(f_low), 0.5 * f_high)
    Gl = 10.0 ** (-3.0 * M / (sr * t60_low))
    Gh = 10.0 ** (-3.0 * M / (sr * t60_high))
    cl, ch = math.cos(2 * math.pi * f_low / sr), math.cos(2 * math.pi * f_high / sr)
    lo_lim = math.sqrt((1.0 - cl) / (1.0 - ch)) * 1.05          # keeps b > 1 (a real pole exists)
    Gh = np.maximum(Gh, Gl * lo_lim)
    p = np.zeros_like(M)
    g = Gl.copy()
    m = (Gl - Gh) > 1e-12
    b = (Gl[m] ** 2 * cl - Gh[m] ** 2 * ch) / (Gl[m] ** 2 - Gh[m] ** 2)
    p[m] = b - np.sqrt(np.maximum(b * b - 1.0, 0.0))
    g[m] = Gl[m] * np.sqrt(1.0 + p[m] ** 2 - 2.0 * p[m] * cl)
    lim = np.sqrt(Gl)
    dc = g / (1.0 - p)
    g = np.where(dc > lim, lim * (1.0 - p), g)
    dc = g / (1.0 - p)
    ny = g / (1.0 + p)
    alpha = float(np.clip(np.mean(np.log(dc) / np.log(ny)), 0.1, 1.0))
    return g, p, alpha


@numba.njit(cache=True)
def _fdn_core(xl, xr, M, A, g, p, bl, br, cl, cr, estep):
    """x(n) = A . filt(delay outputs) + b u(n); returns (outL, outR, energy trace of the delay
    lines every ``estep`` samples; empty when estep <= 0)."""
    n = xl.shape[0]
    N = M.shape[0]
    mx = 0
    for i in range(N):
        mx = max(mx, M[i])
    buf = np.zeros((N, mx))
    ptr = np.zeros(N, np.int64)
    s = np.zeros(N)
    outl = np.zeros(n)
    outr = np.zeros(n)
    ne = (n + estep - 1) // estep if estep > 0 else 0
    en = np.zeros(ne)
    for k in range(n):
        ol = 0.0
        orr = 0.0
        for i in range(N):
            s[i] = g[i] * buf[i, ptr[i]] + p[i] * s[i]
            ol += cl[i] * s[i]
            orr += cr[i] * s[i]
        outl[k] = ol
        outr[k] = orr
        a = xl[k]
        c = xr[k]
        for i in range(N):
            acc = bl[i] * a + br[i] * c
            for j in range(N):
                acc += A[i, j] * s[j]
            buf[i, ptr[i]] = acc
        for i in range(N):
            ptr[i] += 1
            if ptr[i] >= M[i]:
                ptr[i] = 0
        if estep > 0 and k % estep == 0:
            e = 0.0
            for i in range(N):
                for j in range(M[i]):
                    e += buf[i, j] * buf[i, j]
            en[k // estep] = e
    return outl, outr, en


def _fdn_io(n: int):
    """Orthogonal +-1/sqrt(N) input (L, R) and output (L, R) vectors: four Hadamard rows."""
    h = hadamard_matrix(n)
    return h[1].copy(), h[2].copy(), h[3].copy(), h[n // 2 + 1].copy()


def fdn_impulse_energy(lines: int = 8, matrix: str = "householder", sr: int = DEFAULT_SR, size: float = 6.0,
                       seconds: float = 2.0, estep: int = 2048) -> np.ndarray:
    """Energy stored in the delay lines of the *lossless prototype* (no damping) after one impulse,
    sampled every ``estep`` samples: constant for a unitary feedback matrix."""
    M = fdn_lengths(sr, size, lines)
    A = np.ascontiguousarray(feedback_matrix(matrix, lines))
    bl, br, cl, cr = _fdn_io(lines)
    x = np.zeros(int(seconds * sr))
    x[0] = 1.0
    one, zero = np.ones(lines), np.zeros(lines)
    return _fdn_core(x, np.zeros_like(x), M, A, one, zero, bl, br, cl, cr, int(estep))[2]


_DIFF_AP = (337, 113, 37)   # at 25 kHz (g13 §1.5: mutually prime, spanning orders of magnitude; < 15 ms)


def _late_fdn(xl, xr, sr, t60_low, t60_high, size, lines=16, matrix="householder", diffusion=0.7, tonal=1.0):
    """Late reverberation of (xl, xr) -> (wl, wr); impulse response normalised to about unit energy
    per channel (the reverberant level of steady noise equals the dry level)."""
    lines = 16 if int(lines) >= 12 else 8
    M = fdn_lengths(sr, size, lines)
    A = np.ascontiguousarray(feedback_matrix(matrix, lines))
    g, p, alpha = fdn_damping(M, sr, t60_low, t60_high)
    bl, br, cl, cr = _fdn_io(lines)
    xl = np.ascontiguousarray(xl, dtype=np.float64)
    xr = np.ascontiguousarray(xr, dtype=np.float64)
    d = float(np.clip(diffusion, 0.0, 1.0))
    if d > 0:
        gl = np.full(3, 0.75 * d)
        xl = _ap_series(xl, np.array(coprime_lengths(_DIFF_AP, sr / 25000.0 * 1.0001), np.int64), gl)
        xr = _ap_series(xr, np.array(coprime_lengths(_DIFF_AP, sr / 25000.0 * 1.13), np.int64), gl)
    # tonal correction E(z) = (1 - b z^-1) / (1 - b), b = (1 - alpha) / (1 + alpha)  (PASP/Tonal_Correction_Filter)
    b = (1.0 - alpha) / (1.0 + alpha) * float(np.clip(tonal, 0.0, 1.0))
    if b > 1e-6:
        xl = lfilter([1.0 / (1.0 - b), -b / (1.0 - b)], [1.0], xl)
        xr = lfilter([1.0 / (1.0 - b), -b / (1.0 - b)], [1.0], xr)
    wl, wr, _ = _fdn_core(np.ascontiguousarray(xl), np.ascontiguousarray(xr), M, A, g, p, bl, br, cl, cr, 0)
    # energy injected |bL + bR|^2 = 2 (mono), stored energy decays with tau = T60 / 13.8, each read head
    # carries E / sum(M) per sample -> IR energy per channel = 2 T60 sr / (13.8 sum M)
    t_eff = float(np.clip(t60_low, 0.05, 60.0))
    norm = 0.6 * math.sqrt(13.8 * float(M.sum()) / (2.0 * t_eff * sr))   # 0.6: measured, filters + tonal correction
    return wl * norm, wr * norm


@effect("fdn_reverb", "Feedback-delay-network reverb (Jot): 8 or 16 mutually prime lines, Householder or Hadamard feedback, decay time set separately for lows and highs. Smooth, non-metallic tails. Stereo output.",
        t60_low=(2.2, "decay time at 250 Hz, seconds (0.1..30)"),
        t60_high=(1.1, "decay time at 4 kHz, seconds (<= t60_low; air and walls always damp highs more)"),
        size=(8.0, "mean free path of the room in metres (4 x volume / surface): 1 = booth, 4 = room, 8 = hall, 20 = cathedral, 50 = canyon"),
        predelay=(0.02, "seconds before the reverb starts"),
        diffusion=(0.7, "input allpass diffusion 0..1 (0 = grainy early echoes, 1 = instantly dense)"),
        mix=(0.3, "wet amount 0..1"),
        width=(1.0, "stereo width 0..1"),
        lines=(16, "delay lines: 8 or 16 (16 = denser)"),
        matrix=("householder", "feedback matrix: householder | hadamard"),
        tonal=(1.0, "tonal correction 0..1: restores the highs that the faster HF decay removes from the tail"),
        tail=(None, "seconds of tail appended (auto = 1.2 x t60_low, max 10)"))
def fdn_reverb(x, sr=DEFAULT_SR, t60_low=2.2, t60_high=1.1, size=8.0, predelay=0.02, diffusion=0.7, mix=0.3,
               width=1.0, lines=16, matrix="householder", tonal=1.0, tail=None):
    """Jot FDN (findings 10 §2.4): lossless prototype (orthogonal feedback) + first-order damping per
    line designed from the two decay times (``fdn_damping``) + tonal correction."""
    x = np.asarray(x, float)
    t60_low = float(np.clip(t60_low, 0.05, 30.0))
    if tail is None:
        tail = min(1.2 * t60_low + 0.1, 10.0)
    pd = samples(predelay, sr) if predelay > 0 else 0
    ne = samples(tail, sr) + pd
    xl, xr = _lr(x)
    wl, wr = _late_fdn(_pad(xl, ne - pd, front=pd), _pad(xr, ne - pd, front=pd), sr, t60_low, t60_high, size,
                       lines, matrix, diffusion, tonal)
    return _fade_end(_wetdry(_pad(x, ne), _ms_width(wl, wr, width), mix), sr)


# ==========================================================================================
# Rooms: Sabine per band, shoebox image sources, presets
# ==========================================================================================

BANDS = _space.ABSORPTION_BANDS            # 125 .. 4000 Hz
ABSORPTION = dict(_space.ABSORPTION)       # Farnell Table 5.1 (verbatim in physical/space.py)
ABSORPTION.update({
    "open": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),          # an opening: nothing comes back (definition of the sabin)
    "tile": ABSORPTION["marble"],                     # alias (glazed tile ~ marble in Table 5.1)
    "rock": ABSORPTION["brick"],                      # alias: UNSOURCED, rough stone taken as brick
    "foliage": (0.2, 0.3, 0.4, 0.5, 0.6, 0.7),        # UNSOURCED: dense vegetation edge, scattering counted as loss
    "soil": (0.15, 0.25, 0.4, 0.55, 0.6, 0.6),        # UNSOURCED: soft ground / grass
})

# Dimensions (lx across, ly along the view, lz height, metres), six surfaces (x0, x1, y0, y1, floor,
# ceiling), extra absorbers {material: m^2} and the default source-listener distance.
# UNSOURCED: the geometries and furnishing areas are plausible choices, not measurements; the decay
# times follow from them through Sabine's formula and Table 5.1.
ROOM_PRESETS = {
    "closet": dict(dims=(1.0, 1.5, 2.2), surf=("fabric",) * 4 + ("carpet", "plaster"), extra={"people": 3.0}, dist=0.6),
    "studio": dict(dims=(5.0, 6.0, 3.0), surf=("fabric",) * 4 + ("carpet", "fabric"), extra={"people": 40.0}, dist=2.0),
    "bedroom": dict(dims=(3.5, 4.0, 2.5), surf=("plaster",) * 4 + ("carpet", "plaster"), extra={"fabric": 12.0, "people": 14.0}, dist=2.0),
    "bathroom": dict(dims=(2.0, 2.5, 2.4), surf=("tile",) * 4 + ("tile", "plaster"), extra={"fabric": 5.0, "people": 2.0, "glass": 2.0}, dist=1.2),
    "hall": dict(dims=(20.0, 30.0, 10.0), surf=("plaster",) * 4 + ("wood", "plaster"), extra={"people": 600.0, "fabric": 600.0}, dist=12.0),
    "church": dict(dims=(15.0, 40.0, 18.0), surf=("brick",) * 4 + ("marble", "wood"), extra={"people": 200.0, "wood": 300.0}, dist=15.0),
    "cave": dict(dims=(15.0, 40.0, 8.0), surf=("rock",) * 6, extra={"soil": 60.0}, dist=12.0),
    "forest": dict(dims=(50.0, 50.0, 20.0), surf=("foliage",) * 4 + ("soil", "open"), extra={}, dist=10.0),
    "street": dict(dims=(12.0, 80.0, 18.0), surf=("brick", "brick", "open", "open", "concrete", "open"), extra={}, dist=10.0),
    "canyon": dict(dims=(40.0, 300.0, 80.0), surf=("rock", "rock", "open", "open", "soil", "open"), extra={}, dist=25.0),
}


def room_info(preset: str = "hall", lx=None, ly=None, lz=None, walls=None, floor=None, ceiling=None,
              humidity: int = 50, furnish=None) -> dict:
    """Geometry and Sabine decay of a preset or an explicit shoebox.

    T60(band) = 0.161 V / (sum S_i a_i + 4 m V): Sabine (Farnell §6.1, findings 01) with the air term
    4 m V, m = intensity attenuation per metre from PASP Table B.2 (``space.air_db_per_m``).
    A preset comes with its furnishing; as soon as a dimension or material is given the room is a bare
    shoebox of those materials (values not given come from the preset) plus ``furnish`` m^2 of soft
    absorbers. Returns dims, surfaces, volume, area, t60 (6 bands 125..4000 Hz), mfp = 4V/S (mean free path) and
    critical distance r_c = sqrt(A / (16 pi)) (follows from Sabine's absorption area A at 500-1000 Hz)."""
    if preset not in ROOM_PRESETS:
        raise ValueError(f"unknown room preset {preset!r}; choose from {sorted(ROOM_PRESETS)}")
    pr = ROOM_PRESETS[preset]
    dims = [float(v) for v in pr["dims"]]
    for i, v in enumerate((lx, ly, lz)):
        if v is not None:
            dims[i] = float(np.clip(float(v), 0.5, 1000.0))
    surf = list(pr["surf"])
    explicit = any(v is not None for v in (lx, ly, lz, walls, floor, ceiling))
    for name, sl in ((walls, slice(0, 4)), (floor, slice(4, 5)), (ceiling, slice(5, 6))):
        if name is not None:
            if name not in ABSORPTION:
                raise ValueError(f"unknown material {name!r}; choose from {sorted(ABSORPTION)}")
            surf[sl] = [name] * (sl.stop - sl.start)
    LX, LY, LZ = dims
    areas = [LY * LZ, LY * LZ, LX * LZ, LX * LZ, LX * LY, LX * LY]
    V = LX * LY * LZ
    S = float(sum(areas))
    A = np.zeros(len(BANDS))
    for a, m in zip(areas, surf):
        A += a * np.asarray(ABSORPTION[m])
    if furnish is not None:                       # explicit soft absorbers: m^2 of "people" (Table 5.1)
        A += float(np.clip(furnish, 0.0, 1e5)) * np.asarray(ABSORPTION["people"])
    elif not explicit:                            # the preset's own furniture / audience
        for m, a in pr["extra"].items():
            A += a * np.asarray(ABSORPTION[m])
    m_air = _space.air_db_per_m(np.asarray(BANDS, float), humidity) * math.log(10.0) / 10.0
    A_tot = A + 4.0 * m_air * V
    t60 = np.clip(0.161 * V / A_tot, 0.05, 30.0)
    a_mid = float(np.mean(A_tot[2:4]))
    return {"preset": preset, "dims": tuple(dims), "surfaces": tuple(surf), "volume": V, "area": S,
            "t60": t60, "mfp": 4.0 * V / S, "critical_distance": math.sqrt(a_mid / (16.0 * math.pi)),
            "dist": float(pr["dist"])}


def image_source_taps(dims, src, lis, surfaces, order: int = 3, tmax: float = 0.1):
    """Shoebox image-source model (Allen & Berkley 1979). Image positions
    (1 - 2p) s + 2 n L for n in Z^3, p in {0,1}^3; the path reflects |n_a - p_a| times on the wall at 0
    and |n_a| times on the wall at L_a; each reflection scales the pressure by sqrt(1 - a) per band.
    Returns (delay s after the direct sound, gains (taps, 6 bands) relative to the direct sound
    (1/r spreading), unit x component of the arrival direction = sin azimuth, direct distance)."""
    L = np.asarray(dims, float)
    s = np.asarray(src, float)
    r = np.asarray(lis, float)
    beta = np.sqrt(1.0 - np.clip(np.array([ABSORPTION[m] for m in surfaces], float), 0.0, 0.9999))
    d0 = max(float(np.linalg.norm(s - r)), 0.05)
    K = int(order)
    delays, gains, sx = [], [], []
    for nx in range(-K, K + 1):
        for ny in range(-K, K + 1):
            for nz in range(-K, K + 1):
                for px in (0, 1):
                    for py in (0, 1):
                        for pz in (0, 1):
                            cnt = (abs(nx - px), abs(nx), abs(ny - py), abs(ny), abs(nz - pz), abs(nz))
                            tot = sum(cnt)
                            if tot == 0 or tot > K:
                                continue
                            pos = (1 - 2 * np.array([px, py, pz])) * s + 2 * np.array([nx, ny, nz]) * L
                            v = pos - r
                            d = float(np.linalg.norm(v))
                            t = (d - d0) / C_AIR
                            if t > tmax or t <= 0:
                                continue
                            gb = np.ones(len(BANDS)) * (d0 / d)
                            for w in range(6):
                                if cnt[w]:
                                    gb = gb * beta[w] ** cnt[w]
                            delays.append(t)
                            gains.append(gb)
                            sx.append(v[0] / d)
    if not delays:
        return np.zeros(0), np.zeros((0, len(BANDS))), np.zeros(0), d0
    o = np.argsort(delays)
    return np.asarray(delays)[o], np.asarray(gains)[o], np.asarray(sx)[o], d0


def _room_positions(info, distance):
    LX, LY, LZ = info["dims"]
    d = float(info["dist"] if distance is None else distance)
    lis = np.array([0.43 * LX, 0.21 * LY, min(1.6, 0.45 * LZ)])          # UNSOURCED: off-centre, ear height
    src = np.array([0.55 * LX, min(lis[1] + max(d, 0.1), 0.95 * LY), min(1.4, 0.5 * LZ)])
    return src, lis


def early_ir(info: dict, sr: int, distance=None, order: int = 3, stereo: bool = True):
    """Early-reflection impulse response (without the direct sound) and its mid-band energy.
    Per-band tap gains are merged with zero-phase weights that interpolate linearly in log-frequency
    between the six band centres. Stereo: two opposite cardioids, L = 1 - sin(az), R = 1 + sin(az)
    (Sound Rendering s(mu) = (1 + cos mu)^k, k = 1; their sum is omnidirectional)."""
    src, lis = _room_positions(info, distance)
    tmax = float(np.clip(2.5 * info["mfp"] / C_AIR, 0.1, 0.6))           # at least PASP's "first ~100 ms"
    delays, gains, sx, d0 = image_source_taps(info["dims"], src, lis, info["surfaces"], order, tmax)
    pad = int(0.006 * sr)
    n = int(tmax * sr) + 2 * pad + 4
    nfft = 1 << int(np.ceil(np.log2(n)))
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    lf = np.log(np.maximum(f, 1.0))
    lb = np.log(np.asarray(BANDS, float))
    nch = 2 if stereo else 1
    H = np.zeros((nch, f.shape[0]), complex)
    pos = delays * sr + pad
    i0 = np.floor(pos).astype(int)
    fr = pos - i0
    for b in range(len(BANDS)):
        wgt = np.interp(lf, lb, np.eye(len(BANDS))[b])
        for c in range(nch):
            pan = np.ones_like(sx) if not stereo else (1.0 - sx if c == 0 else 1.0 + sx)
            h = np.zeros(nfft)
            np.add.at(h, i0, gains[:, b] * pan * (1.0 - fr))
            np.add.at(h, i0 + 1, gains[:, b] * pan * fr)
            H[c] += wgt * np.fft.rfft(h)
    ir = np.fft.irfft(H, nfft, axis=1)[:, :n].T
    e_mid = float(np.sum(np.mean(gains[:, 2:4], axis=1) ** 2)) if len(delays) else 0.0
    return (ir if stereo else ir[:, 0]), pad, e_mid, d0


def _apply_er(x, ir, pad):
    n = x.shape[0]
    if ir.ndim == 1:
        if x.ndim == 1:
            return fftconvolve(x, ir)[pad:pad + n]
        return np.stack([fftconvolve(x[:, c], ir)[pad:pad + n] for c in range(2)], axis=1)
    xl, xr = _lr(x)
    return np.stack([fftconvolve(xl, ir[:, 0])[pad:pad + n], fftconvolve(xr, ir[:, 1])[pad:pad + n]], axis=1)


_ROOM_PARAMS = dict(
    preset=("hall", "closet | studio | bedroom | bathroom | hall | church | cave | forest | street | canyon"),
    lx=(None, "room width in metres (overrides the preset)"),
    ly=(None, "room length in metres (the direction you look in)"),
    lz=(None, "room height in metres"),
    walls=(None, "wall material: carpet concrete marble wood brick glass plaster fabric metal people water tile rock foliage soil open"),
    floor=(None, "floor material (same list)"),
    ceiling=(None, "ceiling material (same list; `open` = no ceiling)"),
    distance=(None, "source to listener distance in metres (default from the preset); farther = more room, less direct"),
    furnish=(None, "m2 of soft absorbers (people, sofas, curtains). Presets have their own; a room with explicit dimensions or materials is bare unless this is set"),
)


@effect("early_reflections", "Early reflections of a box room from an image-source model: the first echoes off walls, floor and ceiling (size, shape and materials of the room) without a reverb tail.",
        mix=(0.5, "0 = dry, 1 = direct sound plus reflections at their physical level"),
        order=(3, "highest reflection order 1..4"),
        stereo=(True, "true = stereo output (reflections arrive from their directions)"),
        humidity=(50, "relative humidity % (40..70) for air absorption"),
        **_ROOM_PARAMS)
def early_reflections(x, sr=DEFAULT_SR, mix=0.5, order=3, stereo=True, humidity=50, preset="hall", lx=None, ly=None,
                      lz=None, walls=None, floor=None, ceiling=None, distance=None, furnish=None):
    """Tapped delay line of the image sources up to ``order`` (PASP/Early_Reflections: the first
    ~100 ms set the perceived room shape). Output = (1 - mix) dry + mix (dry + ER) / sqrt(1 + E_er)."""
    x = np.asarray(x, float)
    info = room_info(preset, lx, ly, lz, walls, floor, ceiling, humidity, furnish)
    ir, pad, e_er, _ = early_ir(info, sr, distance, int(np.clip(order, 1, 4)), bool(stereo) or x.ndim == 2)
    xp = _pad(x, ir.shape[0])
    er = _apply_er(xp, ir, pad)
    full = (to_stereo(xp) if er.ndim == 2 and xp.ndim == 1 else xp) + er
    return _fade_end(_wetdry(xp, full / math.sqrt(1.0 + e_er), mix), sr)


@lru_cache(maxsize=64)
def _kernel_norm(kind: str, key: tuple) -> float:
    """1 / sqrt(mean channel energy) of the impulse response of a 48 kHz space.py kernel."""
    kw = dict(key)
    secs = float(min(10.0, kw.pop("_secs", 4.0)))
    imp = np.zeros(int(secs * _PHYS_SR))
    imp[0] = 1.0
    if kind == "zita":
        ir = _space.zita_rev1(imp, **kw)
    elif kind == "puckette":
        ir = _space.puckette_rev(imp, **kw)
    else:
        ir = _space.farnell_room(imp, **kw)
    e = float(np.mean(np.sum(ir ** 2, axis=0)))
    return 1.0 / math.sqrt(max(e, 1e-12))


def _kernel_late(kind, x, sr, kw, secs):
    """Run a physical/space.py kernel (fixed 48 kHz) on ``x`` at ``sr``; unit-energy stereo wet."""
    x48 = _resample(x, sr, _PHYS_SR)
    key = tuple(sorted({**{k: (round(float(v), 4) if v is not None else None) for k, v in kw.items()},
                        "_secs": round(float(secs), 2)}.items()))
    norm = _kernel_norm(kind, key)
    fn = {"zita": _space.zita_rev1, "puckette": _space.puckette_rev, "farnell": _space.farnell_room}[kind]
    wet = _resample(fn(x48, **kw) * norm, _PHYS_SR, sr)
    n = x.shape[0]
    if wet.shape[0] < n:
        wet = _pad(wet, n - wet.shape[0])
    return wet[:n]


@effect("room", "A physical room: early reflections from its shape, decay per frequency band from its volume and wall materials (Sabine), direct-to-reverb balance from the listening distance. Presets from a closet to a canyon, or your own dimensions and materials. Stereo output.",
        mix=(0.5, "0 = dry, 1 = what a listener at `distance` hears (direct + reflections + reverberation)"),
        model=("fdn", "late reverb engine: fdn (16-line Jot network) | zita (zita-rev1) | puckette (Puckette G08, one decay time) | farnell (Farnell's tight prime-delay room, fixed about 0.5 s)"),
        width=(1.0, "stereo width of the reverberation 0..1"),
        predelay=(0.0, "extra seconds before the late reverb (the room adds its own: mean free path / c)"),
        humidity=(50, "relative humidity % (40..70) for air absorption"),
        tail=(None, "seconds of tail appended (auto = 1.2 x the longest decay, max 10)"),
        **_ROOM_PARAMS)
def room(x, sr=DEFAULT_SR, mix=0.5, model="fdn", width=1.0, predelay=0.0, humidity=50, tail=None, preset="hall",
         lx=None, ly=None, lz=None, walls=None, floor=None, ceiling=None, distance=None, furnish=None):
    """Direct + image-source early reflections + late reverberation.

    Late level: reverberant / direct energy = (r / r_c)^2 with the critical distance
    r_c = sqrt(A / (16 pi)) (Sabine absorption area A; capped at +10 dB). The FDN gets T60 at 250 Hz
    and 4 kHz from the Sabine bands and its line lengths from the mean free path 4V/S; ``zita`` maps
    the bands with ``space.zita_from_bands``; ``puckette`` uses the 500-1000 Hz T60."""
    x = np.asarray(x, float)
    info = room_info(preset, lx, ly, lz, walls, floor, ceiling, humidity, furnish)
    t60 = info["t60"]
    t_long = float(np.max(t60))
    if tail is None:
        tail = min(1.2 * t_long + 0.2, 10.0)
    ir, pad, e_er, d0 = early_ir(info, sr, distance, 3, True)
    xp = _pad(x, samples(tail, sr) + ir.shape[0])
    er = _apply_er(xp, ir, pad)
    pd = int(min(info["mfp"] / C_AIR, 0.25) * sr) + (samples(predelay, sr) if predelay > 0 else 0)
    xd = _pad(xp, 0, front=pd)[: xp.shape[0]]
    model = str(model).lower()
    t_mid = float(np.mean(t60[2:4]))
    if model == "fdn":
        xl, xr = _lr(xd)
        wl, wr = _late_fdn(xl, xr, sr, float(t60[1]), float(t60[5]), info["mfp"], 16, "householder", 0.7, 1.0)
    elif model == "zita":
        zp = _space.zita_from_bands(t60, rdel=1.0)
        zp["t60dc"], zp["t60m"] = float(np.clip(zp["t60dc"], 0.1, 30)), float(np.clip(zp["t60m"], 0.1, 30))
        w = _kernel_late("zita", xd, sr, zp, 1.3 * max(zp["t60dc"], zp["t60m"]) + 0.3)
        wl, wr = w[:, 0], w[:, 1]
    elif model == "puckette":
        # UNSOURCED mapping: the optional one-pole damping follows the 4 kHz / mid decay ratio
        kw = {"t60": t_mid, "damp_hz": float(np.clip(6000.0 * t60[5] / t_mid, 800.0, 12000.0))}
        w = _kernel_late("puckette", xd, sr, kw, 1.3 * t_mid + 0.3)
        wl, wr = w[:, 0], w[:, 1]
    elif model == "farnell":
        w = _kernel_late("farnell", xd, sr, {"lowpass_hz": 400.0, "stage1_fb": 0.0, "liveness": 0.33}, 2.0)
        wl, wr = w[:, 0], w[:, 1]
    else:
        raise ValueError(f"unknown room model {model!r}; choose fdn, zita, puckette or farnell")
    e_late = float(min((d0 / info["critical_distance"]) ** 2, 10.0))
    late = _ms_width(wl, wr, width) * math.sqrt(e_late)
    full = (to_stereo(xp) + er + late) / math.sqrt(1.0 + e_er + e_late)
    return _fade_end(_wetdry(xp, full, mix), sr)


@effect("zita", "zita-rev1 (Fons Adriaensen): the smooth 8-line reverb with separate low and mid decay times. Stereo output.",
        mix=(0.3, "wet amount 0..1"),
        preset=("default", "default | instrument | exterior_dry | stone_hall | cave (the values below override it)"),
        t60_low=(None, "decay time below `xover`, seconds (preset default 3)"),
        t60_mid=(None, "decay time in the mid band, seconds (preset default 2)"),
        xover=(None, "low/mid crossover Hz (200)"),
        damp=(None, "frequency where the decay time has fallen to half the mid value, Hz (6000)"),
        predelay=(None, "seconds before the reverb starts (0.06)"),
        width=(1.0, "stereo width 0..1"),
        tail=(None, "seconds of tail appended (auto = 1.2 x the longer decay, max 10)"))
def zita(x, sr=DEFAULT_SR, mix=0.3, preset="default", t60_low=None, t60_mid=None, xover=None, damp=None,
         predelay=None, width=1.0, tail=None):
    """``physical/space.py`` ``zita_rev1`` (Faust ``zita_rev1_stereo``; findings 08 §7.1, 10 §2.4),
    run at its native 48 kHz and resampled; wet normalised to a unit-energy impulse response."""
    if preset not in _space.ZITA_PRESETS:
        raise ValueError(f"unknown zita preset {preset!r}; choose from {sorted(_space.ZITA_PRESETS)}")
    kw = dict(_space.ZITA_PRESETS[preset])
    if t60_low is not None:
        kw["t60dc"] = float(np.clip(t60_low, 0.1, 30.0))
    if t60_mid is not None:
        kw["t60m"] = float(np.clip(t60_mid, 0.1, 30.0))
    if xover is not None:
        kw["f1"] = float(np.clip(xover, 30.0, 2000.0))
    if damp is not None:
        kw["f2"] = float(np.clip(damp, 500.0, 0.45 * _PHYS_SR))
    if predelay is not None:
        kw["rdel"] = float(np.clip(predelay, 0.0, 1.0)) * 1000.0
    x = np.asarray(x, float)
    t_long = max(kw["t60dc"], kw["t60m"])
    if tail is None:
        tail = min(1.2 * t_long + kw["rdel"] / 1000.0, 10.0)
    xp = _pad(x, samples(tail, sr))
    w = _kernel_late("zita", xp, sr, kw, 1.3 * t_long + 0.3)
    return _fade_end(_wetdry(xp, _ms_width(w[:, 0], w[:, 1], width), mix), sr)


# ==========================================================================================
# Propagation: distance, Doppler fly-by, occlusion
# ==========================================================================================

def _air_filter(x, sr, metres, rh):
    """Zero-phase magnitude filter of ``metres`` of air (PASP Table B.2 via space.air_db_per_m)."""
    n = x.shape[0]
    nfft = 1 << int(np.ceil(np.log2(n + int(0.02 * sr))))
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    mag = 10.0 ** (-_space.air_db_per_m(f, rh) * metres / 20.0)
    X = np.fft.rfft(x, nfft, axis=0)
    return np.fft.irfft(X * (mag if x.ndim == 1 else mag[:, None]), nfft, axis=0)[:n]


def air_cutoff(metres, rh: int = 50):
    """Frequency (Hz) at which ``metres`` of air removes 3 dB (table + f^2 law, inverted numerically)."""
    f = np.geomspace(100.0, 40000.0, 400)
    r3 = 3.0 / np.maximum(_space.air_db_per_m(f, rh), 1e-12)        # metres for 3 dB at f
    r3 = np.minimum.accumulate(r3)                                   # monotone decreasing in f
    return np.interp(np.asarray(metres, float), r3[::-1], f[::-1])


@effect("distance", "Put a sound some metres away: 1/r level drop, air absorption that dulls it with distance and humidity, and optionally more reverberation relative to the direct sound.",
        metres=(10.0, "distance to the source, metres (the input is the sound at `ref` metres)"),
        air=(True, "true = air absorption low-pass"),
        humidity=(50, "relative humidity % (40..70; drier air absorbs more highs)"),
        reverb=(0.0, "reverb send 0..1: reverberant level stays constant while the direct sound falls with distance (stereo out when > 0)"),
        reverb_time=(1.5, "decay time of that reverb, seconds"),
        ref=(1.0, "reference distance in metres where gain = 1 (closer than this the gain is capped at +6 dB)"),
        line_source=(False, "true = line source (road, river): 1/sqrt(r), -3 dB per doubling"),
        delay=(False, "true = prepend the travel time (r - ref) / 345 s"))
def distance(x, sr=DEFAULT_SR, metres=10.0, air=True, humidity=50, reverb=0.0, reverb_time=1.5, ref=1.0,
             line_source=False, delay=False):
    """Spreading ref/r (point source, -6 dB per doubling; Farnell rule 19) or sqrt(ref/r); air
    absorption over (r - ref) metres from PASP Table B.2 with the f^2 extension; c = 345 m/s."""
    x = np.asarray(x, float)
    r = float(np.clip(metres, 0.01, 1e5))
    ref = float(max(ref, 0.01))
    g = min(math.sqrt(ref / r) if line_source else ref / r, 2.0)      # closer than `ref`: at most +6 dB
    extra = max(r - ref, 0.0)
    y = _air_filter(x, sr, extra, humidity) * g if (air and extra > 0) else x * g
    if delay and extra > 0:
        y = _pad(y, 0, front=int(round(extra / C_AIR * sr)))
    send = float(np.clip(reverb, 0.0, 1.0))
    if send > 0:
        t = float(np.clip(reverb_time, 0.1, 20.0))
        xp = _pad(x, samples(min(1.2 * t, 10.0), sr) + (y.shape[0] - x.shape[0]))
        xl, xr = _lr(xp)
        wl, wr = _late_fdn(xl, xr, sr, t, 0.5 * t, 8.0, 16, "householder", 0.7, 1.0)
        y = to_stereo(_pad(y, xp.shape[0] - y.shape[0])) + send * 0.5 * np.stack([wl, wr], axis=1)
    return _fade_end(y, sr)


def flyby_path(n: int, sr: int, speed: float, closest: float, pass_at: float = 0.5):
    """Emission times (seconds, fractional) for each output sample of a straight fly-by.

    Source on the line y = closest, x = v (tau - t0); listener at the origin. Sound emitted at tau is
    heard at t = tau + r(tau) / c; solving for tau (u = tau - t0, T = t - t0):
    (c^2 - v^2) u^2 - 2 c^2 T u + c^2 T^2 - d^2 = 0, smaller root. The output clock starts when the
    first input sample arrives. Returns (tau, distance r(tau), x position, n_out)."""
    c = C_AIR
    v = float(np.clip(abs(speed), 0.0, 0.9 * c))
    d = float(max(closest, 0.05))
    dur = n / sr
    t0 = float(pass_at) * dur
    r_start = math.hypot(d, v * (0.0 - t0))
    r_end = math.hypot(d, v * (dur - t0))
    n_out = max(2, int(math.ceil((dur + (r_end - r_start) / c) * sr)))
    T = np.arange(n_out) / sr + r_start / c - t0
    u = (c * c * T - np.sqrt(np.maximum(c ** 4 * T * T - (c * c - v * v) * (c * c * T * T - d * d), 0.0))) / (c * c - v * v)
    xs = v * u
    return u + t0, np.hypot(d, xs), xs, n_out


@effect("doppler", "Fly-by: the sound passes the listener on a straight line, with the true Doppler pitch drop (c/(c-v) approaching, c/(c+v) leaving), 1/r swell, air absorption and a pan across. Uses the whole length of the input.",
        speed=(20.0, "source speed in m/s (20 = 72 km/h car, 60 = racing car, 250 = jet)"),
        closest=(5.0, "closest distance to the listener in metres (small = sudden pass, large = slow swell)"),
        pass_at=(0.5, "when it passes, as a fraction of the sound's duration 0..1"),
        direction=("lr", "lr = left to right, rl = right to left"),
        pan=(True, "true = stereo output panned by the source direction"),
        air=(True, "true = air absorption that opens up as it approaches"),
        humidity=(50, "relative humidity % (40..70)"),
        ref=(None, "distance in metres where gain = 1 (default = `closest`, so the pass is at full level)"))
def doppler(x, sr=DEFAULT_SR, speed=20.0, closest=5.0, pass_at=0.5, direction="lr", pan=True, air=True,
            humidity=50, ref=None):
    """Time-varying delay = exact Doppler (PASP/Doppler_Simulation; g13 §3.2): each output sample
    reads the input at its emission time with a 4th-order Lagrange interpolator (even order for
    swept delays, g13 §2.1). Pan: constant-power cos/sin law (g15 §6.1 item 7) on sin(azimuth)."""
    mono = np.ascontiguousarray(to_mono(np.asarray(x, float)))
    n = mono.shape[0]
    tau, r, xs, _ = flyby_path(n, sr, speed, closest, float(np.clip(pass_at, 0.0, 1.0)))
    y = _read_l4(mono, np.ascontiguousarray(tau * sr))
    rref = float(max(closest, 0.05) if ref is None else max(ref, 0.01))
    y = y * np.minimum(rref / r, 2.0)
    if air:
        fc = np.minimum(air_cutoff(r, humidity), 0.45 * sr)
        y = _tv_one_pole(np.ascontiguousarray(y), np.exp(-2 * np.pi * fc / sr))
    if pan:
        s = xs / r * (-1.0 if str(direction).lower() == "rl" else 1.0)
        th = (s + 1.0) * np.pi / 4
        y = np.stack([y * np.cos(th), y * np.sin(th)], axis=1)
    return _fade_end(y, sr, head_ms=2.0)


# Volume density kg/m^3. UNSOURCED: handbook values, not in the findings files.
_DENSITY = {"concrete": 2300.0, "brick": 1900.0, "marble": 2700.0, "rock": 2600.0, "tile": 2300.0, "wood": 600.0,
            "glass": 2500.0, "plaster": 850.0, "metal": 7800.0, "fabric": 300.0, "carpet": 300.0, "water": 1000.0,
            "soil": 1500.0, "foliage": 30.0, "people": 1000.0}


@effect("occlude", "Sound from behind an obstacle (a wall, a door, a window, a curtain): the part that goes through the material loses its highs by the mass law, the part that bends round the edge is dulled smoothly.",
        material=("brick", "concrete brick marble rock tile wood glass plaster metal fabric carpet water soil foliage people"),
        thickness=(0.1, "thickness of the obstacle in metres (0.004 window pane, 0.04 door, 0.1 wall, 0.3 thick wall)"),
        leak=(0.15, "level 0..1 of the path that diffracts round the obstacle (0 = sealed room, 1 = a screen you can nearly see past)"),
        edge=(600.0, "Hz above which the diffracted path falls off (smaller = deeper in the shadow)"))
def occlude(x, sr=DEFAULT_SR, material="brick", thickness=0.1, leak=0.15, edge=600.0):
    """Farnell p.87: a wall "transmits a filtered version of the sound, usually with less high
    frequencies"; Sound Rendering: "occluders damp smoothly rather than cut" (diffraction).
    Transmission: field-incidence mass law TL = 20 log10(f m) - 47 dB (m = surface density kg/m^2),
    i.e. a first-order low-pass with corner 224 / m Hz.  # UNSOURCED: standard building acoustics
    Diffraction: one-pole low-pass at ``edge`` scaled by ``leak``.  # UNSOURCED: shape only"""
    if material not in _DENSITY:
        raise ValueError(f"unknown material {material!r}; choose from {sorted(_DENSITY)}")
    x = np.asarray(x, float)
    m = _DENSITY[material] * float(np.clip(thickness, 1e-4, 10.0))
    fc = float(np.clip(224.0 / m, 0.5, 0.45 * sr))
    a = math.exp(-2 * math.pi * fc / sr)
    y = lfilter([1.0 - a], [1.0, -a], x, axis=0)
    lk = float(np.clip(leak, 0.0, 1.0))
    if lk > 0:
        e = math.exp(-2 * math.pi * float(np.clip(edge, 50.0, 0.45 * sr)) / sr)
        y = y + lk * lfilter([1.0 - e], [1.0, -e], x, axis=0)
    y = sosfilt(butter(1, 20.0, "high", fs=sr, output="sos"), y, axis=0)
    return _fade_end(y, sr)


# ==========================================================================================
# Rotary speaker (Leslie)  (PASP/Leslie, Rotating_Horn_Simulation, Rotating_Woofer_Port_Cabinet; g13 §3.3)
# ==========================================================================================

# PASP gives no numbers for the Leslie. Everything in this table is UNSOURCED (commonly quoted
# values for a model 122/147: chorale ~0.8 Hz, tremolo ~6.7 Hz, crossover 800 Hz).
_LESLIE_RATES = {"slow": (0.8, 0.67), "fast": (6.7, 5.7), "brake": (0.0, 0.0)}   # (horn Hz, drum Hz)
_LESLIE_TAU = (0.8, 3.5)       # UNSOURCED: horn / drum spin-up time constants, seconds
_HORN_R = 0.19                 # UNSOURCED: horn mouth radius from the axis, metres
_DRUM_R = 0.15                 # UNSOURCED: effective drum port radius, metres
_CAB = 0.35                    # UNSOURCED: distance to the reflecting cabinet wall, metres
_CAB_GAIN = 0.35               # UNSOURCED: level of that reflection


def _soft_clip(c):
    """f(x) = x - x^3/3 for |x| <= 1, +-2/3 beyond (PASP ch. 9.1 cubic soft clipper; g13 §8)."""
    c = np.clip(c, -1.0, 1.0)
    return c - c ** 3 / 3.0


def _rotor_angle(n, sr, f0, f1, tau):
    t = np.arange(n) / sr
    f = f1 + (f0 - f1) * np.exp(-t / tau)
    return 2 * np.pi * np.cumsum(f) / sr


@effect("leslie", "Rotary speaker cabinet (Leslie): a spinning treble horn and bass drum give Doppler vibrato, tremolo and a moving tone, split at 800 Hz; slow (chorale), fast (tremolo) or braking, with the rotors' own acceleration. Stereo output.",
        speed=("fast", "slow | fast | brake: the speed the rotors head for"),
        start=(None, "slow | fast | brake: speed at the start of the sound; if it differs from `speed` the rotors spin up or down (horn in about 1 s, drum in about 4 s)"),
        drive=(0.3, "amplifier overdrive 0..1 (the growl)"),
        mic=(0.6, "microphone distance from the rotor axis in metres 0.3..3 (close = deep tremolo, far = mostly vibrato)"),
        spread=(90.0, "angle between the two microphones in degrees 0..180"),
        mix=(1.0, "wet amount 0..1"),
        crossover=(800.0, "horn/drum crossover Hz"))
def leslie(x, sr=DEFAULT_SR, speed="fast", start=None, drive=0.3, mic=0.6, spread=90.0, mix=1.0, crossover=800.0):
    """Structure of PASP/Leslie (g13 §3.3): (5) soft-clip overdrive -> crossover -> (1) rotating horn
    as a source on a circle read through a delay distance(t)/c with 1/r gain, (2) one cabinet-wall
    image source with its own Doppler shift, (3) rotating drum as amplitude modulation plus a
    modulated low-pass cutoff. Far field: w_l = w_s [1 - (r_s w_m / c) sin(w_m t)]."""
    x = np.asarray(x, float)
    mono = to_mono(x)
    n = mono.shape[0]
    if speed not in _LESLIE_RATES or (start is not None and start not in _LESLIE_RATES):
        raise ValueError("leslie speed/start must be slow, fast or brake")
    dr = float(np.clip(drive, 0.0, 1.0))
    if dr > 0:
        pre = 1.0 + 8.0 * dr                                    # UNSOURCED: drive range
        mono = _soft_clip(pre * mono) / math.sqrt(pre)
    fc = float(np.clip(crossover, 100.0, 0.4 * sr))
    lp = butter(2, fc, "low", fs=sr, output="sos")
    hp = butter(2, fc, "high", fs=sr, output="sos")
    lo = np.ascontiguousarray(sosfilt(lp, sosfilt(lp, mono)))    # Linkwitz-Riley 4th order
    hi = np.ascontiguousarray(sosfilt(hp, sosfilt(hp, mono)))
    f1 = _LESLIE_RATES[speed]
    f0 = _LESLIE_RATES[start] if start is not None else f1
    th_h = _rotor_angle(n, sr, f0[0], f1[0], _LESLIE_TAU[0])
    th_d = -_rotor_angle(n, sr, f0[1], f1[1], _LESLIE_TAU[1])    # UNSOURCED: drum turns the other way
    rl = float(np.clip(mic, 0.3, 3.0))
    idx = np.arange(n, dtype=np.float64)
    half = math.radians(float(np.clip(spread, 0.0, 180.0))) / 2
    chans = []
    for phi in (-half, half):
        # horn: direct path
        cs = np.cos(th_h - phi)
        d = np.sqrt(rl * rl + _HORN_R ** 2 - 2 * rl * _HORN_R * cs)
        near = rl - _HORN_R
        h = _read_l4(hi, idx - (d - near) / C_AIR * sr) * (near / d)
        h = h * (0.4 + 0.6 * 0.5 * (1.0 + cs))                   # UNSOURCED: horn directivity (1 + cos)/2
        h = _tv_one_pole(np.ascontiguousarray(h), np.exp(-2 * np.pi * np.minimum(2000.0 + 7000.0 * 0.5 * (1.0 + cs), 0.45 * sr) / sr))
        # horn: image in the cabinet wall behind the rotor (opposite phase of the rotation)
        d2 = np.sqrt(rl * rl + _HORN_R ** 2 + 2 * rl * _HORN_R * cs) + 2 * _CAB
        if _CAB_GAIN > 0:
            h = h + _CAB_GAIN * _read_l4(hi, idx - (d2 - near) / C_AIR * sr) * (near / d2)
        # drum: Doppler (small), tremolo, moving low-pass
        cd = np.cos(th_d - phi)
        dd = np.sqrt(rl * rl + _DRUM_R ** 2 - 2 * rl * _DRUM_R * cd)
        b = _read_l4(lo, idx - (dd - (rl - _DRUM_R)) / C_AIR * sr) * (1.0 - 0.35 * 0.5 * (1.0 - cd))   # UNSOURCED depth
        b = _tv_one_pole(np.ascontiguousarray(b), np.exp(-2 * np.pi * (fc * (0.5 + 0.5 * 0.5 * (1.0 + cd)) + 100.0) / sr))
        chans.append(h + b)
    wet = np.stack(chans, axis=1)
    wet = sosfilt(butter(1, 25.0, "high", fs=sr, output="sos"), wet, axis=0)
    return _fade_end(_wetdry(x, wet, mix), sr)


# ==========================================================================================
# Flanger, phaser, chorus, vibrato  (PASP ch. 5 and 8.9; g13 §3.1, §3.4, §4.1)
# ==========================================================================================

@numba.njit(cache=True)
def _flanger_core(x, dly, fb):
    """v(n) = u(n - M(n)), u(n) = x(n) + fb v(n): interpolated delay with true recursive regeneration."""
    n = x.shape[0]
    dmax = 0.0
    for k in range(n):
        if dly[k] > dmax:
            dmax = dly[k]
    L = int(dmax) + 16
    buf = np.zeros(L)
    w = np.zeros(5)
    out = np.zeros(n)
    wp = 0
    for k in range(n):
        d = dly[k]
        if d < 3.5:
            d = 3.5
        v = _ring_l4(buf, L, float(wp), d, w)
        buf[wp] = x[k] + fb * v
        out[k] = v
        wp += 1
        if wp >= L:
            wp = 0
    return out


def _sweep(n, sr, rate, shape, phase, lo, hi):
    """Delay sweep lo..hi; ``exp`` = triangular on a log scale (g13 §3.1)."""
    u = 0.5 + 0.5 * _lfo(n, sr, rate, shape, phase)
    if shape == "exp" and lo > 0 and hi > lo:
        return lo * (hi / lo) ** u
    return lo + (hi - lo) * u


@effect("flanger", "Jet-like sweeping comb: a short delay swept by an LFO and mixed with the dry sound, with real feedback (regeneration) for the resonant whoosh. Best on noisy or inharmonic sounds.",
        rate=(0.3, "LFO Hz"),
        depth=(2.0, "sweep width in ms: the delay moves between `delay` and `delay + depth`"),
        feedback=(0.5, "regeneration -0.95..0.95 (negative = hollow)"),
        mix=(0.5, "wet amount 0..1 (0.5 = deepest notches)"),
        delay=(1.0, "shortest delay in ms (>= 0.2)"),
        invert=(False, "true = subtract the delayed copy: notches and peaks swap, first notch at DC (thin, high-passed)"),
        shape=("sine", "LFO shape: sine | triangle | exp (triangle on a log scale: even sweep in pitch)"),
        phase=(0.0, "LFO start phase 0..1 cycles"),
        stereo=(0.0, "LFO phase offset between left and right in cycles (0.25 = wide; makes stereo output)"))
def flanger(x, sr=DEFAULT_SR, rate=0.3, depth=2.0, feedback=0.5, mix=0.5, delay=1.0, invert=False, shape="sine",
            phase=0.0, stereo=0.0):
    """PASP/Flanging (g13 §3.1): y(n) = x(n) + g x[n - M(n)], notches at odd multiples of fs/(2M),
    spacing fs/M = 1/delay; invert: g -> -g; regeneration a_M from the delay output to its input."""
    x = np.asarray(x, float)
    n = x.shape[0]
    fb = float(np.clip(feedback, -0.95, 0.95))
    lo = max(float(delay), 0.2) * sr / 1000.0
    hi = lo + max(float(depth), 0.0) * sr / 1000.0
    sign = -1.0 if invert else 1.0
    mix = float(np.clip(mix, 0.0, 1.0))

    def one(sig, ph):
        d = np.maximum(_sweep(n, sr, rate, shape, ph, lo, hi), 3.5)
        v = _flanger_core(np.ascontiguousarray(sig, dtype=np.float64), np.ascontiguousarray(d), fb)
        return sig * (1.0 - mix) + sign * mix * v

    if x.ndim == 1 and not stereo:
        return _fade_end(one(x, phase), sr)
    xl, xr = _lr(x)
    return _fade_end(np.stack([one(xl, phase), one(xr, phase + float(stereo))], axis=1), sr)


@numba.njit(cache=True)
def _phaser_core(x, fbase, ratios, sr, fb):
    """Chain of first-order allpasses H(z) = (p - z^-1) / (1 - p z^-1),
    p = (1 - tan(w_b T / 2)) / (1 + tan(w_b T / 2)), break frequencies fbase[n] * ratios, with feedback."""
    n = x.shape[0]
    S = ratios.shape[0]
    x1 = np.zeros(S)
    y1 = np.zeros(S)
    out = np.zeros(n)
    last = 0.0
    fmax = 0.45 * sr
    for k in range(n):
        u = x[k] + fb * last
        for s in range(S):
            f = fbase[k] * ratios[s]
            if f > fmax:
                f = fmax
            t = np.tan(np.pi * f / sr)
            p = (1.0 - t) / (1.0 + t)
            y = p * u - x1[s] + p * y1[s]
            x1[s] = u
            y1[s] = y
            u = y
        last = u
        out[k] = u
    return out


@effect("phaser", "Sweeping phaser: a chain of 4, 6 or 8 first-order allpass stages mixed with the dry sound gives 2, 3 or 4 notches that glide up and down, unevenly spaced (so it suits harmonic sounds where a flanger does not).",
        rate=(0.5, "LFO Hz"),
        depth=(0.7, "sweep amount 0..1 (1 = the notches travel `sweep` octaves)"),
        stages=(4, "allpass stages 4 | 6 | 8 (notches = stages / 2)"),
        mix=(0.5, "wet amount 0..1 (0.5 = full-depth notches)"),
        feedback=(0.0, "resonance -0.9..0.9 (sharpens the notches' edges)"),
        freq=(100.0, "lowest break frequency in Hz at the bottom of the sweep; the others are `spacing` apart"),
        spacing=(2.0, "ratio between successive break frequencies (2 = octaves)"),
        sweep=(3.0, "sweep range in octaves at depth 1"),
        phase=(0.0, "LFO start phase 0..1 cycles"),
        stereo=(0.0, "LFO phase offset between left and right in cycles (makes stereo output)"))
def phaser(x, sr=DEFAULT_SR, rate=0.5, depth=0.7, stages=4, mix=0.5, feedback=0.0, freq=100.0, spacing=2.0,
           sweep=3.0, phase=0.0, stereo=0.0):
    """PASP ch. 8.9 classic first-order chain (g13 §4.1): y = (x + g AP_S(...AP_1(x))) / 2, break
    frequencies 100, 200, 400, 800 Hz (octave spaced), bilinear allpass with the break mapped
    exactly, swept together per sample; S sections give S/2 notches (chain phase = odd pi)."""
    x = np.asarray(x, float)
    n = x.shape[0]
    S = int(np.clip(2 * round(int(stages) / 2), 2, 12))
    ratios = float(np.clip(spacing, 1.05, 4.0)) ** np.arange(S)
    fb = float(np.clip(feedback, -0.9, 0.9))
    octs = float(np.clip(sweep, 0.0, 8.0)) * float(np.clip(depth, 0.0, 1.0))
    mix = float(np.clip(mix, 0.0, 1.0))
    f0 = float(np.clip(freq, 10.0, 0.2 * sr))

    def one(sig, ph):
        fbase = f0 * 2.0 ** (octs * (0.5 + 0.5 * _lfo(n, sr, rate, "sine", ph)))
        wet = _phaser_core(np.ascontiguousarray(sig, dtype=np.float64), np.ascontiguousarray(fbase), ratios, float(sr), fb)
        return sig * (1.0 - mix) + wet * mix

    if x.ndim == 1 and not stereo:
        return _fade_end(one(x, phase), sr)
    xl, xr = _lr(x)
    return _fade_end(np.stack([one(xl, phase), one(xr, phase + float(stereo))], axis=1), sr)


@effect("chorus", "Thickening chorus: several copies on slowly swept, interpolated delays, spread across the stereo field (stereo output).",
        rate=(0.8, "LFO Hz"),
        depth=(3.0, "modulation depth ms"),
        mix=(0.5, "wet amount 0..1"),
        voices=(2, "1..8 delayed copies"),
        delay=(15.0, "delay of the first voice in ms"),
        spacing=(5.0, "extra delay per further voice in ms"),
        spread=(0.5, "how far the voices are panned apart 0..1"),
        feedback=(0.0, "regeneration 0..0.7 (towards a flanger)"))
def chorus(x, sr=DEFAULT_SR, rate=0.8, depth=3.0, mix=0.5, voices=2, delay=15.0, spacing=5.0, spread=0.5, feedback=0.0):
    """PASP/Chorus_Effect (g13 §3.4): interpolating taps on one delay line, each oscillating about
    its own position, each placed separately in the stereo field (4th-order Lagrange reads).
    Voice v: delay + v spacing + depth (0.5 + 0.5 sin), rate x (1 + 0.1 v), left/right LFOs in quadrature."""
    x = np.asarray(x, float)
    mono = np.ascontiguousarray(to_mono(x))
    n = mono.shape[0]
    V = int(np.clip(voices, 1, 8))
    fb = float(np.clip(feedback, 0.0, 0.7))
    idx = np.arange(n, dtype=np.float64)
    l = np.zeros(n)
    r = np.zeros(n)
    sp = float(np.clip(spread, 0.0, 1.0))
    for v in range(V):
        ph = v / V
        base = max(float(delay) + float(spacing) * v, 0.5)
        outs = []
        for q in (0.0, 0.25):
            d = (base + max(float(depth), 0.0) * (0.5 + 0.5 * _lfo(n, sr, rate * (1 + 0.1 * v), "sine", ph + q))) * sr / 1000.0
            if fb > 0:
                outs.append(_flanger_core(mono, np.ascontiguousarray(d), fb))
            else:
                outs.append(_read_l4(mono, idx - d))
        pos = 0.0 if V == 1 else (2.0 * v / (V - 1) - 1.0) * sp           # voice pan, constant power
        th = (pos + 1.0) * np.pi / 4
        l += outs[0] * math.cos(th) * math.sqrt(2.0)
        r += outs[1] * math.sin(th) * math.sqrt(2.0)
    wet = np.stack([l, r], axis=1) / V
    return _fade_end(_wetdry(x, wet, mix), sr)


def vibrato(x, sr=DEFAULT_SR, rate=5.0, depth=0.3, delay=5.0):
    """Pitch vibrato = a flanger with the direct path removed (g13 §3.1): one interpolated delay
    swept +-``depth`` ms by a sine. Peak pitch deviation = 2 pi rate depth/1000 (ratio). Library
    function (the registered ``vibrato`` effect in fx.py is left alone)."""
    x = np.asarray(x, float)
    n = x.shape[0]
    d = (max(float(delay), float(depth) + 0.2) + float(depth) * _lfo(n, sr, rate)) * sr / 1000.0
    idx = np.arange(n, dtype=np.float64)
    if x.ndim == 1:
        return _read_l4(np.ascontiguousarray(x), idx - d)
    return np.stack([_read_l4(np.ascontiguousarray(x[:, c]), idx - d) for c in range(x.shape[1])], axis=1)


# ==========================================================================================
# Tape / ping-pong delay
# ==========================================================================================

@numba.njit(cache=True)
def _tape_core(xl, xr, dl, dr, fb, lp, hp, drive, pingpong):
    n = xl.shape[0]
    dmax = 0.0
    for k in range(n):
        if dl[k] > dmax:
            dmax = dl[k]
        if dr[k] > dmax:
            dmax = dr[k]
    L = int(dmax) + 16
    bl = np.zeros(L)
    br = np.zeros(L)
    w = np.zeros(5)
    ol = np.zeros(n)
    orr = np.zeros(n)
    sl = 0.0
    sr_ = 0.0
    hl = 0.0
    hr = 0.0
    wp = 0
    for k in range(n):
        a = _ring_l4(bl, L, float(wp), max(dl[k], 3.5), w)
        b = _ring_l4(br, L, float(wp), max(dr[k], 3.5), w)
        sl = (1.0 - lp) * a + lp * sl               # tape head roll-off
        sr_ = (1.0 - lp) * b + lp * sr_
        hl = hp * hl + (1.0 - hp) * sl               # low cut: v - lowpass(v)
        hr = hp * hr + (1.0 - hp) * sr_
        fl = sl - hl
        fr = sr_ - hr
        if drive > 0.0:
            fl = np.tanh(drive * fl) / drive
            fr = np.tanh(drive * fr) / drive
        ol[k] = fl
        orr[k] = fr
        if pingpong:
            bl[wp] = 0.5 * (xl[k] + xr[k]) + fb * fr
            br[wp] = fl
        else:
            bl[wp] = xl[k] + fb * fl
            br[wp] = xr[k] + fb * fr
        wp += 1
        if wp >= L:
            wp = 0
    return ol, orr


@effect("tape_delay", "Tape echo: each repeat is duller and thinner than the last (filters in the feedback loop), with wow and flutter on the delay time and tape saturation; ping-pong bounces the repeats left-right (stereo output).",
        time=(0.3, "delay time in seconds"),
        feedback=(0.45, "0..0.95: how many repeats"),
        mix=(0.35, "wet amount 0..1"),
        pingpong=(True, "true = repeats alternate left / right (stereo output); false = straight echo, channel count kept"),
        tone=(3500.0, "low-pass in the loop, Hz: each repeat loses highs above this"),
        lowcut=(120.0, "high-pass in the loop, Hz: each repeat loses lows below this"),
        wow=(0.3, "slow tape-speed drift 0..1 (pitch wobble of the repeats)"),
        flutter=(0.2, "fast tape-speed flutter 0..1"),
        drive=(0.3, "tape saturation 0..1"),
        tail=(None, "seconds of tail appended (auto from feedback, max 8)"))
def tape_delay(x, sr=DEFAULT_SR, time=0.3, feedback=0.45, mix=0.35, pingpong=True, tone=3500.0, lowcut=120.0,
               wow=0.3, flutter=0.2, drive=0.3, tail=None):
    """Delay line with filtered feedback and modulated, interpolated read (g13 §2.1: even-order
    Lagrange for swept delays). Ping-pong: input -> left line -> right line -> back to left.
    Wow 0.6 Hz up to 0.4 % and flutter 7.3 Hz up to 0.08 % of the delay time.  # UNSOURCED: rates/depths"""
    x = np.asarray(x, float)
    fb = float(np.clip(feedback, 0.0, 0.95))
    T = float(np.clip(time, 0.002, 4.0))
    if tail is None:
        per = 2 * T if pingpong else T
        tail = min(8.0, per * (1.0 + math.log(1000.0) / max(-math.log(max(fb, 1e-3)), 0.05)))
    xp = _pad(x, samples(tail, sr))
    n = xp.shape[0]
    xl, xr = _lr(xp)
    mod = (1.0 + 0.004 * float(np.clip(wow, 0, 1)) * _lfo(n, sr, 0.6)
           + 0.0008 * float(np.clip(flutter, 0, 1)) * _lfo(n, sr, 7.3, "sine", 0.3))
    dl = np.ascontiguousarray(T * sr * mod)
    dr = np.ascontiguousarray(T * sr * (2.0 - mod)) if pingpong else dl
    lp = math.exp(-2 * math.pi * float(np.clip(tone, 200.0, 0.45 * sr)) / sr)
    hp = math.exp(-2 * math.pi * float(np.clip(lowcut, 5.0, 2000.0)) / sr)
    dv = 3.0 * float(np.clip(drive, 0.0, 1.0))
    wl, wr = _tape_core(xl, xr, dl, dr, fb, lp, hp, dv, bool(pingpong))
    if pingpong or x.ndim == 2:
        wet = np.stack([wl, wr], axis=1)
        dry = to_stereo(xp)
    else:
        wet, dry = wl, xp
    return _fade_end(dry * (1.0 - 0.3 * float(mix)) + wet * float(mix), sr)


# ==========================================================================================
# Shimmer, gated and reverse reverbs (built on the FDN)
# ==========================================================================================

def _octave_up(x, sr, window=0.06):
    """Puckette's two-tap delay pitch shifter (TTEM p.202-208; findings 10 §10): sawtooth delays
    d0..d0 + s half a cycle apart, each enveloped by half a sine (powers sum to 1), f = (t - 1) R / s;
    t = 2 -> the delay shrinks one sample per sample. Window s = 60 ms (source: 30-100 ms)."""
    n = x.shape[0]
    s = window * sr
    ph = (np.arange(n) / s) % 1.0
    idx = np.arange(n, dtype=np.float64)
    out = np.zeros(n)
    x = np.ascontiguousarray(x, dtype=np.float64)
    for off in (0.0, 0.5):
        p = (ph + off) % 1.0
        out += np.sin(np.pi * p) * _read_l4(x, idx - s * (1.0 - p) - 3.0)
    return out


@effect("shimmer", "Shimmer reverb: the reverb tail is fed back through an octave-up pitch shifter, so it blooms into a rising halo. Stereo output.",
        mix=(0.35, "wet amount 0..1"),
        shimmer=(0.5, "amount of octave-up regeneration 0..1"),
        t60=(3.0, "decay time in seconds"),
        size=(10.0, "mean free path in metres"),
        tone=(4500.0, "low-pass on the shifted signal, Hz (keeps the halo from getting piercing)"),
        tail=(None, "seconds of tail appended (auto = 1.5 x t60, max 10)"))
def shimmer(x, sr=DEFAULT_SR, mix=0.35, shimmer=0.5, t60=3.0, size=10.0, tone=4500.0, tail=None):
    """Feedback through the shifter is unrolled into three generations (reverb -> shift -> reverb ...),
    each on an 8-line FDN; exact for the first three octaves of regeneration."""
    x = np.asarray(x, float)
    t = float(np.clip(t60, 0.2, 20.0))
    if tail is None:
        tail = min(1.5 * t, 10.0)
    xp = _pad(x, samples(tail, sr))
    xl, xr = _lr(xp)
    amt = float(np.clip(shimmer, 0.0, 1.0)) * 0.8
    sos = butter(2, float(np.clip(tone, 500.0, 0.45 * sr)), "low", fs=sr, output="sos")
    wl, wr = _late_fdn(xl, xr, sr, t, 0.5 * t, size, 8, "householder", 0.8, 1.0)
    tl, tr = wl.copy(), wr.copy()
    gl, gr = wl, wr
    for _ in range(3 if amt > 0 else 0):
        sl = sosfilt(sos, _octave_up(gl, sr)) * amt
        sr_ = sosfilt(sos, _octave_up(gr, sr)) * amt
        gl, gr = _late_fdn(sl, sr_, sr, t, 0.5 * t, size, 8, "householder", 0.8, 1.0)
        tl += gl
        tr += gr
    return _fade_end(_wetdry(xp, np.stack([tl, tr], axis=1), mix), sr)


@effect("gated_reverb", "Gated reverb (the 80s drum sound): a big dense reverb that is cut off abruptly a moment after the sound stops instead of fading. Stereo output.",
        mix=(0.5, "wet amount 0..1"),
        gate=(0.25, "seconds the reverb stays open after the dry sound falls below the threshold"),
        release=(0.03, "seconds for the gate to close"),
        threshold=(-30.0, "dB below the input peak at which the gate opens"),
        t60=(2.5, "decay time of the reverb under the gate, seconds (long = flat plateau)"),
        size=(9.0, "mean free path in metres"),
        boost=(6.0, "dB of gain on the gated reverb"))
def gated_reverb(x, sr=DEFAULT_SR, mix=0.5, gate=0.25, release=0.03, threshold=-30.0, t60=2.5, size=9.0, boost=6.0):
    """16-line FDN multiplied by a gate keyed from the dry signal's envelope (hold + linear release)."""
    x = np.asarray(x, float)
    hold = float(np.clip(gate, 0.0, 5.0))
    rel = float(np.clip(release, 0.002, 2.0))
    xp = _pad(x, samples(hold + rel + 0.02, sr))
    n = xp.shape[0]
    xl, xr = _lr(xp)
    t = float(np.clip(t60, 0.2, 20.0))
    wl, wr = _late_fdn(xl, xr, sr, t, 0.7 * t, size, 16, "hadamard", 1.0, 1.0)
    a = math.exp(-1.0 / (0.005 * sr))
    env = lfilter([1 - a], [1, -a], np.abs(to_mono(xp)))
    thr = max(float(np.max(env)), 1e-12) * 10.0 ** (float(threshold) / 20.0)
    last = np.maximum.accumulate(np.where(env > thr, np.arange(n), -10 ** 9))
    since = (np.arange(n) - last) / sr
    g = np.clip(1.0 - (since - hold) / rel, 0.0, 1.0)
    g = lfilter([1 - a], [1, -a], g)                              # 5 ms smoothing: no click at the edges
    wet = np.stack([wl, wr], axis=1) * g[:, None] * 10.0 ** (float(np.clip(boost, -20, 20)) / 20.0)
    return _fade_end(_wetdry(xp, wet, mix), sr)


@effect("reverse_reverb", "Reverse reverb: the reverb swells up *before* each sound and is sucked into it (ghostly pre-echo). The output is longer at the front by `t60` seconds. Stereo output.",
        mix=(0.5, "wet amount 0..1"),
        t60=(1.5, "length of the swell in seconds"),
        size=(8.0, "mean free path in metres"),
        dark=(0.6, "0..1: how much faster the highs die (1 = very dark swell)"))
def reverse_reverb(x, sr=DEFAULT_SR, mix=0.5, t60=1.5, size=8.0, dark=0.6):
    """Time-reverse, reverberate with the FDN, time-reverse again; the dry sound is delayed by the
    swell length so both stay aligned."""
    x = np.asarray(x, float)
    t = float(np.clip(t60, 0.1, 10.0))
    ne = samples(t, sr)
    xr_ = _pad(x[::-1], ne)
    xl, xr = _lr(xr_)
    th = t * (1.0 - 0.8 * float(np.clip(dark, 0.0, 1.0)))
    wl, wr = _late_fdn(xl, xr, sr, t, th, size, 16, "householder", 0.8, 1.0)
    wet = np.stack([wl, wr], axis=1)[::-1]
    dry = _pad(x, 0, front=ne)
    return _fade_end(_wetdry(dry, wet, mix), sr, head_ms=20.0)


# ==========================================================================================
# Electric guitar with amplifier feedback  (PASP ch. 9.1 Fig. 9.4 "Sullivan" model; g13 §8)
# ==========================================================================================

_GTR_SR = 48000


@numba.njit(cache=True)
def _fbg_core(exc, n, N, eta, rho, rho_rel, n_off, Mp, r, gpre, drive, asym, fbg, Dfb):
    """Karplus-Strong string (delay N + two-point average 0.5 + allpass) with the amplifier chain
    closed around it: pickup comb -> leaky integrator -> cubic soft clip (offset = asymmetry) ->
    [-> differentiator -> out] and [-> gain -> air delay -> back into the string]."""
    line = np.zeros(N)
    fbl = np.zeros(Dfb)
    out = np.zeros(n)
    ptr = 0
    fptr = 0
    d1 = 0.0
    apx = 0.0
    apy = 0.0
    u = 0.0
    yd1 = 0.0
    c0 = asym
    if c0 > 1.0:
        c0 = 1.0
    f0 = c0 - c0 * c0 * c0 / 3.0
    for k in range(n):
        d = line[ptr]
        on = k < n_off
        rh = rho if on else rho_rel
        lp = rh * 0.5 * (d + d1)                      # loss + 1/2 sample of delay at every frequency
        d1 = d
        ap = eta * lp + apx - eta * apy               # tuning allpass (eta + z^-1) / (1 + eta z^-1)
        apx = lp
        apy = ap
        w = ap + exc[k]
        if on:
            w += fbg * fbl[fptr]
        j = ptr - Mp
        if j < 0:
            j += N
        pu = w - line[j]                              # pickup-position comb (rigid, inverting end)
        line[ptr] = w
        ptr += 1
        if ptr >= N:
            ptr = 0
        u = gpre * pu + r * u                         # pre-distortion gain = integrator g / (1 - r z^-1)
        c = drive * u + asym
        if c > 1.0:
            f = 2.0 / 3.0
        elif c < -1.0:
            f = -2.0 / 3.0
        else:
            f = c - c * c * c / 3.0
        yd = f - f0
        fbl[fptr] = yd
        fptr += 1
        if fptr >= Dfb:
            fptr = 0
        out[k] = yd - r * yd1                         # post-distortion gain = differentiator (1 - r z^-1)
        yd1 = yd
    return out


@instrument("feedback_guitar", "Overdriven electric guitar standing in front of its amplifier: the note sustains and blooms into feedback (a harmonic takes over) instead of dying away.",
            family="plucked", span=("E2", "E5"),
            drive=(6.0, "pre-distortion gain 1..30 (more = more sustain and crunch)"),
            feedback=(0.5, "amplifier-to-string acoustic feedback 0..1 (0 = plain overdriven pluck, above about 0.3 the note never dies)"),
            pickup=(0.18, "pickup position as a fraction of the string from the bridge 0.05..0.5 (0.1 bridge = thin, 0.3 neck = round)"),
            distance=(1.2, "amp-to-guitar distance in metres 0.2..6: the air delay picks the harmonic that howls"),
            tone=(2400.0, "speaker cabinet low-pass in Hz"),
            asym=(0.15, "clipper asymmetry 0..0.6 (even harmonics, tube-like)"),
            seed=(0, "pluck noise seed (int)"))
def feedback_guitar(freq, dur, sr=DEFAULT_SR, vel=1.0, drive=6.0, feedback=0.5, pickup=0.18, distance=1.2,
                    tone=2400.0, asym=0.15, seed=0):
    """Sullivan's electric guitar (PASP Fig. 9.4): string -> [pre gain] -> soft clip f(x) = x - x^3/3
    -> [post gain] -> out, and clip -> [feedback gain] -> [feedback delay = distance / c] -> string.
    Pre = integrator g/(1 - r z^-1), post = differentiator 1 - r z^-1 (bass-heavy drive, flat net
    response); even harmonics from the offset f(x + c); pickup position = feed-forward comb.
    Kernel runs at 48 kHz and is resampled; loop delay N + 1/2 + allpass, the allpass phase delay
    solved exactly at the fundamental."""
    fs = _GTR_SR
    f = float(np.clip(freq, 30.0, 4000.0))
    vel = float(np.clip(vel, 0.0, 1.5))
    rel = 0.14
    n = int((max(dur, 0.01) + rel) * fs)
    n_off = int(max(dur, 0.01) * fs)
    P = fs / f
    N = int(math.floor(P - 0.5 - 0.3))
    N = max(N, 4)
    D = P - 0.5 - N
    w0 = 2 * math.pi * f / fs
    q = math.tan(w0 * D / 2) / math.tan(w0 / 2)
    eta = (1.0 - q) / (1.0 + q)
    t60 = 5.0                                                     # UNSOURCED: solid-body sustain of the fundamental, s
    rho = 10.0 ** (-3.0 * P / (fs * t60)) / math.cos(w0 / 2)
    rho = min(rho, 0.99995)
    rho_rel = 10.0 ** (-3.0 * P / (fs * 0.07))                    # damped by the hand at note-off
    Mp = int(np.clip(round(float(np.clip(pickup, 0.05, 0.5)) * N), 1, N - 1))
    r = 0.995                                                     # UNSOURCED: leak of the integrator / differentiator
    gpre = abs(1.0 - r * np.exp(-1j * w0))                        # unity pre-gain at the fundamental
    rng = np.random.default_rng(int(seed))
    burst = rng.uniform(-1.0, 1.0, N)
    a = math.exp(-2 * math.pi * (600.0 + 2600.0 * min(vel, 1.0)) / fs)   # UNSOURCED: pick brightness vs. velocity
    burst = lfilter([1 - a], [1, -a], burst)
    burst -= burst.mean()
    exc = np.zeros(n)
    exc[:N] = burst * (0.25 + 0.75 * min(vel, 1.0)) * 0.6 / (np.max(np.abs(burst)) + 1e-12)
    drv = float(np.clip(drive, 0.5, 40.0))
    fbk = float(np.clip(feedback, 0.0, 1.0))
    # Feedback path seen by the string: F(w) = (1 - e^{-jwMp}) gpre / (1 - r e^{-jw}) e^{-jw Dfb}. The air
    # delay picks the harmonic h (1..5) with the largest in-phase loop gain |F| cos(arg F) / (1 - rho_h);
    # the delay is then nudged (less than a quarter period of that harmonic) so that it is exactly in
    # phase - what a player does by leaning towards the amplifier.
    Dfb0 = float(np.clip(distance, 0.05, 10.0)) / C_AIR * fs
    best = None
    for h in range(1, 6):
        wh = w0 * h
        if wh > 0.35 * math.pi:
            break
        F = (1.0 - np.exp(-1j * wh * Mp)) * gpre / (1.0 - r * np.exp(-1j * wh))
        loss = 1.0 - rho * math.cos(wh / 2)
        ph = float(np.angle(F * np.exp(-1j * wh * Dfb0)))
        score = abs(F) * math.cos(ph) / loss
        if best is None or score > best[0]:
            best = (score, abs(F), loss, ph, wh)
    _, Fm, loss, ph, wh = best
    Dfb = max(1, int(round(Dfb0 + ph / wh)))
    # small-signal loop gain = feedback x 3.3 x drive / 6: the note stops dying at feedback ~ 0.3
    fbg = fbk * 3.3 / 6.0 * loss / max(Fm, 1e-3) / max(1.0 - float(np.clip(asym, 0.0, 0.6)) ** 2, 0.3)
    y = _fbg_core(exc, n, N, eta, rho, rho_rel, n_off, Mp, r, float(gpre), drv, float(np.clip(asym, 0.0, 0.6)), fbg, Dfb)
    y = y / gpre
    fc = float(np.clip(tone, 500.0, 8000.0))
    y = sosfilt(butter(4, fc, "low", fs=fs, output="sos"), y)
    y = sosfilt(butter(2, 70.0, "high", fs=fs, output="sos"), y)
    y = _resample(y, fs, sr)
    k = min(len(y) // 2, int(0.02 * sr))
    y[-k:] *= np.linspace(1.0, 0.0, k)
    return y * (0.35 + 0.65 * min(vel, 1.0)) * _FBG_GAIN


_FBG_GAIN = 0.247        # level-matched: about -12 dB K-weighted at vel 0.9 (measured)
