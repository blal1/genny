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
* Propagation: 1/r spreading, air absorption (ISO 9613-1), Doppler from a time-varying delay
  (PASP ch. 5; g13 §3.2, findings 10 §10), occlusion (Farnell p.87; *Sound Rendering* "occluders damp
  smoothly rather than cut").
* Delay-line interpolation: 4th-order (even order) Lagrange for every swept delay (g13 §2.1 rule).
* Flanger, phaser, chorus, Leslie (PASP ch. 5 and 8.9; g13 §3.1, §3.3, §3.4, §4.1).
* Electric-guitar feedback chain ("Sullivan" model, PASP ch. 9.1 Fig. 9.4; g13 §8).

Numbers without a source in those files are justified in research/SOURCES.md.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numba
import numpy as np
from scipy.signal import butter, fftconvolve, lfilter, resample_poly, sosfilt, sosfiltfilt

from .core import DEFAULT_SR, samples, to_mono, to_stereo
from . import filters as F
from .fx import effect
from .instruments import instrument
from .physical import space as _space
from .physical.core import tv_one_pole_lp as _tv_one_pole

C_AIR = _space.C_AIR            # 343.2 m/s (20 C)
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


def _stk_len(lengths, sr, base, always=False):
    """STK: lengths x sr/base, then the next odd prime; left as published when sr == base unless ``always`` (NRev)."""
    if not always and float(sr) == float(base):
        return np.array(lengths, np.int64)
    out = []
    for v in lengths:
        n = int(math.floor(v * sr / base)) | 1
        while not _is_prime(n):
            n += 2
        out.append(n)
    return np.array(out, np.int64)


def _stk_comb(x, lengths, sr, t60, damp=0.0):
    g = 10.0 ** (-3.0 * lengths / (float(t60) * sr))            # g = 10^(-3 L / (T60 fs))
    return _fbcf_bank(np.ascontiguousarray(x, dtype=np.float64), lengths, g, float(damp)), g


@effect("nrev", "Perry Cook's NRev (STK, after CCRMA): 6 parallel combs into 6 allpasses and a low-pass. Dense and a little gritty. Stereo output.",
        mix=(0.3, "wet amount 0..1"), t60=(1.0, "decay time, seconds"), tail=(None, "seconds of tail appended (auto = t60)"))
def nrev(x, sr=DEFAULT_SR, mix=0.3, t60=1.0, tail=None):
    """STK NRev: lengths at 25641 Hz, always rescaled to primes; 3 allpasses in series, then 0.7 s + 0.3 x low-pass, a
    fourth allpass, and the fifth to L and the sixth to R; all allpass g = 0.7."""
    t = float(np.clip(t60, 0.05, 30.0))
    x = np.asarray(x, float)
    dry = _pad(x, samples(t if tail is None else tail, sr))
    L = _stk_len((1433, 1601, 1867, 2053, 2251, 2399, 347, 113, 37, 59, 53, 43), sr, 25641.0, always=True)
    c, g = _stk_comb(to_mono(dry), L[:6], sr, t)
    w = _ap_series(c.sum(axis=0) * math.sqrt(1.0 / float(np.sum(1.0 / (1.0 - g ** 2)))), L[6:9], np.full(3, 0.7))
    w = _ap_series(lfilter([0.3], [1.0, -0.7], w), L[9:10], np.full(1, 0.7))
    return _fade_end(_wetdry(dry, np.stack([_ap_series(w, L[10:11], np.full(1, 0.7)), _ap_series(w, L[11:12], np.full(1, 0.7))], axis=1), mix), sr)


@effect("prcrev", "Perry Cook's PRCRev (STK): the smallest useful reverb, 2 allpasses into 2 combs, one per channel. Sparse and cheap. Stereo output.",
        mix=(0.5, "wet amount 0..1"), t60=(1.0, "decay time, seconds"), tail=(None, "seconds of tail appended (auto = t60)"))
def prcrev(x, sr=DEFAULT_SR, mix=0.5, t60=1.0, tail=None):
    """STK PRCRev: 341 and 613 samples allpass g = 0.7 and 1557, 2137 combs at 44.1 kHz."""
    t = float(np.clip(t60, 0.05, 30.0))
    x = np.asarray(x, float)
    dry = _pad(x, samples(t if tail is None else tail, sr))
    L = _stk_len((341, 613, 1557, 2137), sr, 44100.0)
    a = _ap_series(np.ascontiguousarray(to_mono(dry), dtype=np.float64), L[:2], np.full(2, 0.7))
    c, g = _stk_comb(a, L[2:], sr, t)
    return _fade_end(_wetdry(dry, np.stack([c[0] * math.sqrt(1.0 - g[0] ** 2), c[1] * math.sqrt(1.0 - g[1] ** 2)], axis=1), mix), sr)


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
        variant=("mus10", "mus10 (1972: AP 347/113/37, combs 1687/1601/2053/2251 at 25 kHz) | samson (AP 1051/337/113, combs 4799/4999/5399/5801) | stk (Cook's version: AP 225/341/441, combs 1116/1356/1422/1617 at 44.1 kHz with a low-pass in the loop)"),
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
    if str(variant).lower() == "stk":
        # STK JCRev: one pole 0.2 in each comb, output delays 211 and 179 samples
        t = float(np.clip(1.0 if t60 is None else t60, 0.05, 30.0))
        pd = samples(predelay, sr) if predelay > 0 else 0
        ne = samples(t if tail is None else tail, sr)
        dry = _pad(x, ne + pd)
        L = _stk_len((225, 341, 441, 1116, 1356, 1422, 1617, 211, 179), sr, 44100.0)
        a = _ap_series(np.ascontiguousarray(_pad(to_mono(x), ne, front=pd), dtype=np.float64), L[:3], np.full(3, 0.7))
        c, g = _stk_comb(a, L[3:7], sr, t, 0.2)
        w = c.sum(axis=0) * math.sqrt(1.0 / float(np.sum(1.0 / (1.0 - g ** 2))))
        wl, wr = _pad(w, 0, front=int(L[7]))[: w.shape[0]], _pad(w, 0, front=int(L[8]))[: w.shape[0]]
        return _fade_end(_wetdry(dry, _ms_width(wl, wr, width), mix), sr)
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


def weighted_matrix(w) -> np.ndarray:
    """Orthogonal reflection about a weight vector: A = 2 w w^T / |w|^2 - I (the lines with larger weights exchange more)."""
    w = np.asarray(w, float)
    return 2.0 * np.outer(w, w) / float(w @ w) - np.eye(len(w))


def _weights(weights, n: int) -> np.ndarray:
    v = [float(s) for s in str(weights).replace(" ", "").split(",") if s] if isinstance(weights, str) else list(weights)
    if not v:
        return np.ones(n)
    return np.resize(np.abs(np.array(v, float)) + 1e-9, n)


def feedback_matrix(kind: str, n: int, weights="") -> np.ndarray:
    """``householder``: I - (2/N) 1 1^T; for N = 16 Jot's balanced nested form
    A16 = 1/2 [[A4, -A4, -A4, -A4], ...] = A4 (x) A4 (the plain N = 16 reflection "degenerates to
    decoupled combs"). ``hadamard``: Sylvester / sqrt(N). ``weighted``: ``weighted_matrix`` of ``weights``
    (comma list, repeated to N; empty = all equal)."""
    kind = str(kind).lower()
    if kind.startswith("had"):
        return hadamard_matrix(n)
    if kind.startswith("weight"):
        return weighted_matrix(_weights(weights, n))
    if n == 16:
        a4 = householder_matrix(4)
        return np.kron(a4, a4)
    return householder_matrix(n)


_FIRST_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)


def prime_power_lengths(M) -> np.ndarray:
    """PASP 3.7.3 (Faust prime_power_delays): M_i = p_i^round(log M_i / log p_i), distinct primes: mutually prime whatever the rounding."""
    return np.array([p ** max(1, int(round(math.log(m) / math.log(p)))) for m, p in zip(M, _FIRST_PRIMES)], np.int64)


def fdn_lengths(sr: int, size: float, lines: int, t60: float | None = None) -> np.ndarray:
    """Mutually prime line lengths (samples), exponentially spaced with mean = the mean free path
    ``size`` metres / c (PASP/Mean_Free_Path: d = 4V/S; "Faust uses lines exponentially spaced").
    With ``t60``, the lines are lengthened until their sum reaches 0.15 t60 sr (enough modes per Hz for a smooth tail)."""
    mean = max(float(size), 0.2) / C_AIR * sr
    if t60 is not None:
        mean = max(mean, 0.15 * float(t60) * sr / lines)
    r = 1.9 ** np.linspace(-1.0, 1.0, lines)
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


@numba.njit(cache=True)
def _vector_allpass(X, Q, M, g):
    """Multichannel allpass H = (I + g U)^-1 (g I + U), U = Q D(z): y = g x + Q D s, s = x - g y."""
    n, N = X.shape
    L = int(M.max()) + 1
    buf = np.zeros((L, N))
    Y = np.zeros((n, N))
    d = np.zeros(N)
    for i in range(n):
        for c in range(N):
            d[c] = buf[(i - M[c]) % L, c]
        for r in range(N):
            acc = 0.0
            for c in range(N):
                acc += Q[r, c] * d[c]
            Y[i, r] = g * X[i, r] + acc
        for c in range(N):
            buf[i % L, c] = X[i, c] - g * Y[i, c]
    return Y


VECTOR_AP_MS = (4.7, 6.1, 7.9, 10.3)


def vector_diffuse(xl, xr, sr, g):
    """Stereo through a 4-channel Gerzon vector allpass (Hadamard mixing, prime-ish delays 4.7-10.3 ms); channels 0 and 1 out."""
    X = np.ascontiguousarray(np.stack([xl, xr, xl, -xr], axis=1))
    M = np.array([max(1, int(round(ms * 1e-3 * sr))) for ms in VECTOR_AP_MS], np.int64)
    Y = _vector_allpass(X, np.ascontiguousarray(hadamard_matrix(4)), M, float(g))
    return np.ascontiguousarray(Y[:, 0] + Y[:, 2]) * 0.5, np.ascontiguousarray(Y[:, 1] - Y[:, 3]) * 0.5


def _late_fdn(xl, xr, sr, t60_low, t60_high, size, lines=16, matrix="householder", diffusion=0.7, tonal=1.0, prime_power=False,
              diffuser="serial", weights=""):
    """Late reverberation of (xl, xr) -> (wl, wr); impulse response normalised to about unit energy
    per channel (the reverberant level of steady noise equals the dry level)."""
    lines = 16 if int(lines) >= 12 else 8
    M = fdn_lengths(sr, size, lines)
    if prime_power:
        M = prime_power_lengths(M)
    A = np.ascontiguousarray(feedback_matrix(matrix, lines, weights))
    g, p, alpha = fdn_damping(M, sr, t60_low, t60_high)
    bl, br, cl, cr = _fdn_io(lines)
    xl = np.ascontiguousarray(xl, dtype=np.float64)
    xr = np.ascontiguousarray(xr, dtype=np.float64)
    d = float(np.clip(diffusion, 0.0, 1.0))
    if d > 0 and diffuser == "vector":
        xl, xr = vector_diffuse(xl, xr, sr, 0.75 * d)
    elif d > 0:
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
        matrix=("householder", "feedback matrix: householder | hadamard | weighted (reflection about `weights`)"),
        tonal=(1.0, "tonal correction 0..1: restores the highs that the faster HF decay removes from the tail"),
        tail=(None, "seconds of tail appended (auto = 1.2 x t60_low, max 10)"),
        prime_power=(False, "round the line lengths to powers of distinct primes (PASP 3.7.3)"),
        diffuser=("serial", "input diffusion: serial (three allpasses per side) | vector (one 4-channel allpass mixing both sides)"),
        weights=("", "matrix weighted: comma list of line weights, repeated over the lines (e.g. '1,1,2,3')"))
def fdn_reverb(x, sr=DEFAULT_SR, t60_low=2.2, t60_high=1.1, size=8.0, predelay=0.02, diffusion=0.7, mix=0.3,
               width=1.0, lines=16, matrix="householder", tonal=1.0, tail=None, prime_power=False, diffuser="serial", weights=""):
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
                       lines, matrix, diffusion, tonal, bool(prime_power), str(diffuser), weights)
    return _fade_end(_wetdry(_pad(x, ne), _ms_width(wl, wr, width), mix), sr)


def lucier(x, sr=DEFAULT_SR, generations=8, t60=1.5, size=6.0, chain=False):
    """"I am sitting in a room": y(k+1) = normalize(room(y(k))). The room is one fixed impulse response
    (the late FDN, mono); each pass keeps the input length, so the room's modes take over the sound."""
    m = to_mono(np.asarray(x, float))
    n = len(m)
    imp = np.zeros(int(t60 * sr) + 1)
    imp[0] = 1.0
    ir = _late_fdn(imp, imp, sr, float(t60), 0.5 * float(t60), float(size), 16, "householder", 0.7, 1.0)[0]
    ir[0] += 0.5
    y, out = m, []
    for _ in range(int(np.clip(generations, 1, 64))):
        y = fftconvolve(y, ir)[:n]
        y = y / (np.max(np.abs(y)) + 1e-12)
        out.append(y)
    return np.concatenate(out) if chain else y


@effect("lucier", "Room feedback by re-recording (Lucier): the sound played into a room and re-recorded again and again until only the room's resonances remain.",
        generations=(8, "number of passes through the room (1..64)"), t60=(1.5, "room decay time, s"),
        size=(6.0, "room mean free path, m"), chain=(False, "true = all passes one after the other (the piece); false = the last one"))
def lucier_fx(x, sr=DEFAULT_SR, generations=8, t60=1.5, size=6.0, chain=False):
    return lucier(x, sr, generations, t60, size, bool(chain))


def velvet_noise(n: int, sr: int, density: float = 2205.0, rng=None) -> np.ndarray:
    """Velvet noise: one +-1 impulse per grid cell of Td = sr/density samples, at k(m) = round(m Td + r1 (Td - 1)).
    2205 /s (Td = 20 at 44.1 kHz) already sounds smoother than Gaussian noise."""
    rng = np.random.default_rng(0) if rng is None else rng
    td = sr / float(density)
    m = np.arange(int(n / td) + 1)
    k = np.round(m * td + rng.random(m.size) * (td - 1)).astype(int)
    k = k[k < n]
    s = np.zeros(n)
    s[k] = 2.0 * np.round(rng.random(k.size)) - 1.0
    return s


def velvet_ir(sr: int, t60_low: float, t60_high: float, length: float | None = None, density: float = 2205.0,
              seed: int = 0, dark: float | None = None, t60_late: float | None = None, knee_db: float = -20.0,
              t60_bands=None) -> np.ndarray:
    """Late-reverb impulse response: velvet noise split into octave bands (zero-phase, complementary
    raised-cosine crossovers), each band decaying as exp(-6.91 t / T60(f)); T60 is t60_low at 250 Hz and
    t60_high at 4 kHz, log-frequency interpolated and held outside. Energy-normalised to 1.
    ``t60_bands``: ((Hz, T60 s), ...) per octave instead, log-frequency interpolated and held outside.
    ``dark``: dark velvet noise (wide pulses). ``t60_late``: double slope, the tail turns to t60_late (scaled per
    band like the first slope) once it is ``knee_db`` down."""
    if t60_bands is not None:
        bf, bt = (np.asarray(v, float) for v in zip(*sorted(t60_bands)))
        t60_low, t60_high = float(np.interp(math.log2(250.0), np.log2(bf), bt)), float(np.interp(math.log2(4000.0), np.log2(bf), bt))
    t60_low = float(np.clip(t60_low, 0.05, 30.0))
    t60_high = float(np.clip(t60_high, 0.05, t60_low))
    n = samples(min(1.2 * t60_low + 0.05, 12.0) if length is None else length, sr)
    rng = np.random.default_rng(seed)
    v = velvet_noise(n, sr, density, rng) if dark is None else dark_velvet_noise(n, sr, density, float(dark), rng)
    nfft = 1 << int(np.ceil(np.log2(n)))
    V = np.fft.rfft(v, nfft)
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    lf = np.log2(np.maximum(f, 1.0))
    t = np.arange(n) / sr
    ir = np.zeros(n)
    for c in 2.0 ** np.arange(5, 15):                       # octave centres 32 Hz .. 16 kHz
        if c > 0.45 * sr:
            break
        w = np.cos(np.clip(lf - math.log2(c), -1.0, 1.0) * math.pi / 2) ** 2      # sums to 1 across bands
        if c == 32.0:
            w[f < c] = 1.0
        if 2 * c > 0.45 * sr:
            w[f > c] = 1.0
        if t60_bands is None:
            x = float(np.clip((math.log2(c) - math.log2(250.0)) / 4.0, 0.0, 1.0))
            t60 = t60_low * (t60_high / t60_low) ** x
        else:
            t60 = float(np.clip(np.interp(math.log2(c), np.log2(bf), bt), 0.05, 30.0))
        env = np.exp(-6.91 * t / t60) if t60_late is None else two_slope_envelope(t, t60, float(t60_late) * t60 / t60_low, knee_db)
        ir += np.fft.irfft(V * w, nfft)[:n] * env
    return ir / (np.sqrt(np.sum(ir * ir)) + 1e-12)


@effect("velvet_reverb", "Velvet-noise convolution reverb: a sparse +-1 noise shaped per octave into a decaying tail. Dense and smooth from the first milliseconds, no metallic ringing, no modulation. Stereo output.",
        t60_low=(2.0, "decay time at 250 Hz, seconds (0.1..20)"),
        t60_high=(1.0, "decay time at 4 kHz, seconds (<= t60_low)"),
        predelay=(0.02, "seconds before the tail starts"),
        mix=(0.3, "wet amount 0..1"), width=(1.0, "stereo width 0..1 (two independent velvet tails)"),
        density=(2205.0, "impulses per second (1000 = grainy, 2205 = smooth, 4000 = dense)"),
        seed=(0, "variation of the tail"),
        kind=("velvet", "velvet | dark (wider pulses: warmer) | edvn (filtered pulses fitted to the decay per band) | sparse (additions only, no FFT)"),
        dark=(0.5, "kind dark: pulse width 0..1"),
        t60_late=(None, "seconds: a second, slower slope after the knee (double decay); None = single slope"),
        knee=(-20.0, "dB where the second slope starts"),
        preset=(None, "pori | cologne | musikverein: measured halls (decay and early-reflection time)"))
def velvet_reverb(x, sr=DEFAULT_SR, t60_low=2.0, t60_high=1.0, predelay=0.02, mix=0.3, width=1.0, density=2205.0, seed=0,
                  kind="velvet", dark=0.5, t60_late=None, knee=-20.0, preset=None):
    """Velvet-noise late reverberation: octave-band exponential envelopes on velvet noise, convolved by FFT.
    A preset with a measured octave-band decay (``HALL_T60``) uses it band by band."""
    x = np.asarray(x, float)
    bands = None
    if preset is not None:
        if preset not in HALL_PRESETS:
            raise ValueError(f"velvet_reverb preset must be one of {sorted(HALL_PRESETS)}")
        rt, er = HALL_PRESETS[preset]
        t60_low, t60_high, predelay = rt, HALL_HF_RATIO * rt, er
        bands = HALL_T60.get(preset)
        if bands is not None:
            d = dict(bands)
            t60_low, t60_high = d[250.0], d[4000.0]
    dens = float(np.clip(density, 200.0, 0.25 * sr))
    if kind == "sparse":
        xl, xr = _lr(x)
        pd = samples(predelay, sr) if predelay > 0 else 0
        ne = x.shape[0] + samples(min(1.2 * t60_low + 0.05, 12.0), sr) + pd
        out = []
        for ch, sd in ((xl, int(seed)), (xr, int(seed) + 7919)):
            c = _pad(ch, ne - ch.shape[0])
            lo = sosfilt(butter(2, 1000.0, fs=sr, output="sos"), c)
            y = sparse_velvet_convolve(lo, sr, t60_low, dens, seed=sd) + sparse_velvet_convolve(c - lo, sr, t60_high, dens, seed=sd + 1)
            out.append(_pad(y, 0, front=pd)[:ne])
        return _fade_end(_wetdry(_pad(x, ne - x.shape[0]), _ms_width(out[0], out[1], width), mix), sr)
    if kind == "edvn":
        irl = edvn_ir(sr, t60_low, t60_high, seed=int(seed))
        irr = edvn_ir(sr, t60_low, t60_high, seed=int(seed) + 7919)
    elif kind in ("velvet", "dark"):
        dk = float(dark) if kind == "dark" else None
        irl = velvet_ir(sr, t60_low, t60_high, density=dens, seed=int(seed), dark=dk, t60_late=t60_late, knee_db=float(knee), t60_bands=bands)
        irr = velvet_ir(sr, t60_low, t60_high, density=dens, seed=int(seed) + 7919, dark=dk, t60_late=t60_late, knee_db=float(knee), t60_bands=bands)
    else:
        raise ValueError("velvet_reverb kind must be velvet, dark, edvn or sparse")
    pd = samples(predelay, sr) if predelay > 0 else 0
    ne = x.shape[0] + len(irl) + pd
    xl, xr = _lr(x)
    wl = _pad(fftconvolve(xl, irl), ne - pd - len(xl) - len(irl) + 1, front=pd)[:ne]
    wr = _pad(fftconvolve(xr, irr), ne - pd - len(xr) - len(irr) + 1, front=pd)[:ne]
    return _fade_end(_wetdry(_pad(x, len(wl) - x.shape[0]), _ms_width(wl, wr, width), mix), sr)


HALL_PRESETS = {"pori": (2.3, 0.110), "cologne": (1.9, 0.119), "musikverein": (3.2, 0.052)}
HALL_HF_RATIO = 0.75
HALL_T60 = {"pori": ((125.0, 2.7), (250.0, 2.5), (500.0, 2.4), (1000.0, 2.3), (2000.0, 2.1), (4000.0, 1.7), (8000.0, 1.1))}


# Costello's reverb as in Soundpipe modules/revsc.c: delay s, random variation s, its rate Hz, seed
_REVSC = ((2473.0 / 44100, 0.0010, 3.100, 1966), (2767.0 / 44100, 0.0011, 3.500, 29491), (3217.0 / 44100, 0.0017, 1.110, 22937),
          (3557.0 / 44100, 0.0006, 3.973, 9830), (3907.0 / 44100, 0.0010, 2.341, 20643), (4127.0 / 44100, 0.0011, 1.897, 22937),
          (2143.0 / 44100, 0.0017, 0.891, 29491), (1933.0 / 44100, 0.0006, 3.221, 14417))


@numba.njit(cache=True)
def _revsc_core(inl, inr, sr, delay, depth, rate, seed, feedback, damp):
    n = inl.shape[0]
    size = np.array([int((delay[i] + depth[i] * 1.125) * sr + 16.5) for i in range(8)])
    buf = np.zeros((8, int(size.max())))
    wp = np.zeros(8, np.int64)
    rp = np.zeros(8)
    inc = np.zeros(8)
    cnt = np.zeros(8, np.int64)
    fs = np.zeros(8)
    sd = seed.copy()

    def seg(i):
        if sd[i] < 0:
            sd[i] += 0x10000
        sd[i] = (sd[i] * 15625 + 1) & 0xFFFF
        if sd[i] >= 0x8000:
            sd[i] -= 0x10000
        cnt[i] = int(sr / rate[i] + 0.5)
        prv = wp[i] - rp[i]
        while prv < 0.0:
            prv += size[i]
        prv /= sr
        nxt = delay[i] + sd[i] * depth[i] / 32768.0
        inc[i] = (prv - nxt) / cnt[i] * sr + 1.0

    for i in range(8):
        rp[i] = size[i] - (delay[i] + sd[i] * depth[i] / 32768.0) * sr
        seg(i)
    ol = np.zeros(n)
    orr = np.zeros(n)
    for k in range(n):
        jp = 0.25 * np.sum(fs)                     # jpScale
        al = jp + inl[k]
        ar = jp + inr[k]
        l = 0.0
        r = 0.0
        for i in range(8):
            sz = size[i]
            buf[i, wp[i]] = (ar if i & 1 else al) - fs[i]
            wp[i] += 1
            if wp[i] >= sz:
                wp[i] -= sz
            if rp[i] >= sz:
                rp[i] -= sz
            ip = int(rp[i])
            fr = rp[i] - ip
            a2 = (fr * fr - 1.0) / 6.0
            a1 = (fr + 1.0) * 0.5
            am1 = a1 - 1.0
            a0 = 3.0 * a2
            a1 -= a0
            am1 -= a2
            a0 -= fr
            vm1 = buf[i, (ip - 1) % sz]
            v0 = buf[i, ip % sz]
            v1 = buf[i, (ip + 1) % sz]
            v2 = buf[i, (ip + 2) % sz]
            y = (am1 * vm1 + a0 * v0 + a1 * v1 + a2 * v2) * fr + v0
            rp[i] += inc[i]
            y *= feedback
            y = (fs[i] - y) * damp + y
            fs[i] = y
            if i & 1:
                r += y
            else:
                l += y
            cnt[i] -= 1
            if cnt[i] <= 0:
                seg(i)
        ol[k] = 0.35 * l                           # outputGain
        orr[k] = 0.35 * r
    return ol, orr


@effect("costello", "Sean Costello's reverb (Soundpipe revsc, Csound reverbsc): 8 slowly wandering delay lines at one scattering junction. Lush, no metallic ring. Stereo output.",
        mix=(0.3, "wet amount 0..1"), feedback=(0.97, "decay 0..0.999 (0.97 = about 8 s, 0.8 = about 1.5 s)"),
        lpfreq=(10000.0, "Hz of the low-pass in each line"), tail=(None, "seconds of tail appended (auto = from feedback, max 12)"))
def costello(x, sr=DEFAULT_SR, mix=0.3, feedback=0.97, lpfreq=10000.0, tail=None):
    """Port of revsc.c; a double read position replaces its 28-bit fixed point. Junction 0.25 and output 0.35 as printed."""
    x = np.asarray(x, float)
    fb = float(np.clip(feedback, 0.0, 0.999))
    if tail is None:
        tail = min(-3.0 * 0.075 / math.log10(max(fb, 1e-3)), 12.0)       # about 75 ms mean line: -60 dB after that many passes
    dry = _pad(x, samples(tail, sr))
    xl, xr = _lr(dry)
    c = 2.0 - math.cos(2.0 * math.pi * float(np.clip(lpfreq, 50.0, 0.45 * sr)) / sr)
    p = np.array(_REVSC)
    ol, orr = _revsc_core(np.ascontiguousarray(xl, dtype=np.float64), np.ascontiguousarray(xr, dtype=np.float64), float(sr),
                          p[:, 0], p[:, 1], p[:, 2], p[:, 3].astype(np.int64), fb, c - math.sqrt(c * c - 1.0))
    return _fade_end(_wetdry(dry, np.stack([ol, orr], axis=1), mix), sr)


# ==========================================================================================
# Rooms: Sabine per band, shoebox image sources, presets
# ==========================================================================================

BANDS = _space.ABSORPTION_BANDS            # 125 .. 4000 Hz
ABSORPTION = dict(_space.ABSORPTION)       # Farnell Table 5.1 (verbatim in physical/space.py)
ABSORPTION.update({
    "open": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),          # an opening: nothing comes back (definition of the sabin)
    "tile": ABSORPTION["marble"],                     # alias (glazed tile ~ marble in Table 5.1)
    "rock": (0.02, 0.02, 0.03, 0.04, 0.05, 0.05),     # limestone walls (pyroomacoustics 0.10 materials database, limestone_wall)
    "foliage": (0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
    "soil": (0.15, 0.25, 0.4, 0.55, 0.6, 0.6),
})

# Dimensions (lx across, ly along the view, lz height, metres), six surfaces (x0, x1, y0, y1, floor,
# ceiling), extra absorbers {material: m^2} and the default source-listener distance.
ROOM_PRESETS = {
    "closet": dict(dims=(1.0, 1.5, 2.2), surf=("fabric",) * 4 + ("carpet", "plaster"), extra={"people": 3.0}, dist=0.6),
    "studio": dict(dims=(5.0, 6.0, 3.0), surf=("fabric",) * 4 + ("carpet", "fabric"), extra={"people": 40.0}, dist=2.0),
    "bedroom": dict(dims=(3.5, 4.0, 2.5), surf=("plaster",) * 4 + ("carpet", "plaster"), extra={"fabric": 12.0, "people": 14.0}, dist=2.0),
    "bathroom": dict(dims=(2.0, 2.5, 2.4), surf=("tile",) * 4 + ("tile", "plaster"), extra={"fabric": 5.0, "people": 2.0, "glass": 2.0}, dist=1.2),
    "hall": dict(dims=(20.0, 30.0, 10.0), surf=("plaster",) * 4 + ("wood", "plaster"), extra={"people": 600.0, "fabric": 600.0}, dist=12.0),
    # church: Meyer ch. 6: Gothic, unoccupied, 5 s at 10000 m3 (mid band); "people" is fitted to that (was 200 m^2, 6.3 s)
    "priory": dict(dims=(18.0, 95.0, 20.0), surf=("rock",) * 4 + ("marble", "wood"), extra={"people": 800.0, "wood": 800.0}, dist=20.0),
    "church": dict(dims=(15.0, 40.0, 18.0), surf=("brick",) * 4 + ("marble", "wood"), extra={"people": 362.0, "wood": 300.0}, dist=15.0),
    "cave": dict(dims=(15.0, 40.0, 8.0), surf=("rock",) * 6, extra={"soil": 60.0}, dist=12.0),
    "forest": dict(dims=(50.0, 50.0, 20.0), surf=("foliage",) * 4 + ("soil", "open"), extra={}, dist=10.0),
    "street": dict(dims=(12.0, 80.0, 18.0), surf=("brick", "brick", "open", "open", "concrete", "open"), extra={}, dist=10.0),
    "canyon": dict(dims=(40.0, 300.0, 80.0), surf=("rock", "rock", "open", "open", "soil", "open"), extra={}, dist=25.0),
    # Decay targets below are sourced; `extra` is fitted to them through Sabine and is not a measurement.
    # practice_room: Meyer l.11105: individual practice 30-40 m3, 0.3-0.4 s (chunk_01)
    "practice_room": dict(dims=(3.5, 4.0, 2.5), surf=("plaster",) * 4 + ("wood", "plaster"), extra={"people": 31.0}, dist=1.5),
    # orchestra_studio: Gilford 1972 in Meyer l.11046: 1 s at 1000 m3 + 0.2 s per doubling, no rise in the bass, so 1.2 s at 2000 m3
    "orchestra_studio": dict(dims=(12.0, 16.5, 10.0), surf=("plaster",) * 4 + ("wood", "plaster"), extra={"people": 306.0, "wood": 1039.0}, dist=6.0),
    # opera: La Scala, Meyer l.15696: 1.2 s mid, 1.5 s low (Kuttruff Table 7.5 prints 1.2 at 125 Hz, 0.95 mid: the two disagree); volume 10000 m3 from Kuttruff
    "opera": dict(dims=(20.0, 25.0, 20.0), surf=("plaster",) * 4 + ("wood", "plaster"), extra={"audience_seated": 1289.0}, dist=12.0),
    # concert_hall: Musikverein, Meyer Table 6.1: 14600 m3, 2.05 s mid, 2.4 s low; 20 x 39.5 x 18.5 m (39.5 deduced from the volume, chunk_01)
    "concert_hall": dict(dims=(20.0, 39.5, 18.5), surf=("plaster",) * 4 + ("wood", "plaster"), extra={"audience_seated": 1007.0, "wood": 354.0}, dist=15.0),
    # arena: Verona, Meyer l.11198: 0.85 s mid (1.25 s low is out of reach with an open sky under Sabine). geometry
    "arena": dict(dims=(60.0, 75.0, 12.0), surf=("rock",) * 4 + ("audience_seated", "open"), extra={"audience_seated": 1500.0}, dist=30.0),
    # amphitheater: Epidaurus, Meyer l.11167: all reflections within 30 ms, no diffuse field outdoors. geometry
    "amphitheater": dict(dims=(30.0, 30.0, 8.0), surf=("audience_seated",) * 4 + ("hard", "open"), extra={}, dist=12.0, late=False),
}


def room_info(preset: str = "hall", lx=None, ly=None, lz=None, walls=None, floor=None, ceiling=None,
              humidity: int = 50, furnish=None, formula: str = "sabine") -> dict:
    """Geometry and Sabine decay of a preset or an explicit shoebox.

    T60(band) = 0.161 V / (sum S_i a_i + 4 m V): Sabine (Farnell §6.1, findings 01) with the air term
    4 m V, m = intensity attenuation per metre from ISO 9613-1 (``space.air_db_per_m``).
    A preset comes with its furnishing; as soon as a dimension or material is given the room is a bare
    shoebox of those materials (values not given come from the preset) plus ``furnish`` m^2 of soft
    absorbers. ``formula``: sabine | eyring | millington | kuttruff (``space.absorption_exponent``), which matter once
    the mean alpha passes about 0.2. Returns dims, surfaces, volume, area, t60 (6 bands 125..4000 Hz), mfp = 4V/S (mean free
    path), reflections_per_s = c S/4V (Kuttruff 4.5), schroeder_hz from the 500-1000 Hz T60 (3.32) and critical distance
    r_c = sqrt(A / (16 pi)) (follows from the absorption area A at 500-1000 Hz)."""
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
    A_walls = A.copy()
    if furnish is not None:                       # explicit soft absorbers: m^2 of "people" (Table 5.1)
        A += float(np.clip(furnish, 0.0, 1e5)) * np.asarray(ABSORPTION["people"])
    elif not explicit:                            # the preset's own furniture / audience
        for m, a in pr["extra"].items():
            A += a * np.asarray(ABSORPTION[m])
    if formula != "sabine":                       # furniture counts as a uniform lift of alpha over the walls
        alpha = np.array([ABSORPTION[m] for m in surf], float) + (A - A_walls) / S
        A = S * _space.absorption_exponent(np.minimum(alpha, 1.0), formula, areas, dims)
    m_air = _space.air_db_per_m(np.asarray(BANDS, float), humidity) * math.log(10.0) / 10.0
    A_tot = A + 4.0 * m_air * V
    t60 = np.clip(0.161 * V / A_tot, 0.05, 30.0)
    a_mid = float(np.mean(A_tot[2:4]))
    return {"preset": preset, "dims": tuple(dims), "surfaces": tuple(surf), "volume": V, "area": S,
            "t60": t60, "mfp": 4.0 * V / S, "critical_distance": math.sqrt(a_mid / (16.0 * math.pi)),
            "dist": float(pr["dist"]), "reflections_per_s": C_AIR * S / (4.0 * V), "late": bool(pr.get("late", True)),
            "schroeder_hz": _space.schroeder_frequency(float(np.mean(t60[2:4])), V)}


def image_source_taps(dims, src, lis, surfaces, order: int = 3, tmax: float = 0.1, full: bool = False):
    """Shoebox image-source model (Allen & Berkley 1979). Image positions
    (1 - 2p) s + 2 n L for n in Z^3, p in {0,1}^3; the path reflects |n_a - p_a| times on the wall at 0
    and |n_a| times on the wall at L_a; each reflection scales the pressure by sqrt(1 - a) per band.
    Returns (delay s after the direct sound, gains (taps, 6 bands) relative to the direct sound
    (1/r spreading), unit x component of the arrival direction = sin azimuth, direct distance); with ``full``
    also the number of reflections of each path and the y component of its direction (toward the source)."""
    L = np.asarray(dims, float)
    s = np.asarray(src, float)
    r = np.asarray(lis, float)
    beta = np.sqrt(1.0 - np.clip(np.array([ABSORPTION[m] for m in surfaces], float), 0.0, 0.9999))
    d0 = max(float(np.linalg.norm(s - r)), 0.05)
    K = int(order)
    delays, gains, sx, ords, sy = [], [], [], [], []
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
                            ords.append(tot)
                            sy.append(v[1] / d)
    if not delays:
        z = (np.zeros(0), np.zeros((0, len(BANDS))), np.zeros(0), d0)
        return z + (np.zeros(0, int), np.zeros(0)) if full else z
    o = np.argsort(delays)
    out = (np.asarray(delays)[o], np.asarray(gains)[o], np.asarray(sx)[o], d0)
    return out + (np.asarray(ords)[o], np.asarray(sy)[o]) if full else out


def _room_positions(info, distance):
    LX, LY, LZ = info["dims"]
    d = float(info["dist"] if distance is None else distance)
    lis = np.array([0.43 * LX, 0.21 * LY, min(1.6, 0.45 * LZ)])          # off-centre, ear height
    src = np.array([0.55 * LX, min(lis[1] + max(d, 0.1), 0.95 * LY), min(1.4, 0.5 * LZ)])
    return src, lis


def _scatter_bands(scattering):
    if scattering is None:
        return None
    if isinstance(scattering, str):
        if scattering not in _space.SCATTERING:
            raise ValueError(f"scattering must be a number or one of {sorted(_space.SCATTERING)}")
        return np.asarray(_space.SCATTERING[scattering], float)
    return np.full(len(BANDS), float(np.clip(scattering, 0.0, 1.0)))


def early_ir(info: dict, sr: int, distance=None, order: int = 3, stereo: bool = True, scattering=None, mic: float = 1.0):
    """Early-reflection impulse response (without the direct sound) and its mid-band energy.
    Per-band tap gains are merged with zero-phase weights that interpolate linearly in log-frequency
    between the six band centres. Stereo: two opposite cardioids, L = 1 - sin(az), R = 1 + sin(az)
    (Sound Rendering s(mu) = (1 + cos mu)^k, k = 1; their sum is omnidirectional).
    ``scattering`` s per band: the specular part of a path with n reflections keeps (1 - s)^n of its energy, the
    rest arrives as 8 random-sign pulses spread over the next mean free path. ``mic`` a: receiver pattern
    a + (1 - a) cos(angle from the source direction) (1 omni, 0.5 cardioid, 0 figure of eight)."""
    src, lis = _room_positions(info, distance)
    tmax = float(np.clip(2.5 * info["mfp"] / C_AIR, 0.1, 0.6))           # at least PASP's "first ~100 ms"
    delays, gains, sx, d0, ords, sy = image_source_taps(info["dims"], src, lis, info["surfaces"], order, tmax, full=True)
    a_mic = float(np.clip(mic, 0.0, 1.0))
    if a_mic < 1.0 and len(delays):
        gains = gains * (a_mic + (1.0 - a_mic) * sy)[:, None]
    sb = _scatter_bands(scattering)
    if sb is not None and len(delays):
        keep = (1.0 - sb[None, :]) ** ords[:, None]
        ed = gains ** 2 * (1.0 - keep)
        gains = gains * np.sqrt(keep)
        rng = np.random.default_rng(1)
        m = 8
        dd = (delays[:, None] + rng.random((len(delays), m)) * info["mfp"] / C_AIR).ravel()
        dg = (np.sqrt(ed / m)[:, None, :] * (2.0 * np.round(rng.random((len(delays), m, 1))) - 1.0)).reshape(-1, len(BANDS))
        ok = dd < tmax
        delays = np.concatenate([delays, dd[ok]])
        gains = np.concatenate([gains, dg[ok]])
        sx = np.concatenate([sx, rng.uniform(-1.0, 1.0, int(ok.sum()))])
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
    preset=("hall", "closet | practice_room | studio | bedroom | bathroom | orchestra_studio | opera | hall | concert_hall | church | cave | forest | street | canyon | arena | amphitheater | priory"),
    lx=(None, "room width in metres (overrides the preset)"),
    ly=(None, "room length in metres (the direction you look in)"),
    lz=(None, "room height in metres"),
    walls=(None, "wall material: carpet concrete marble wood brick glass plaster fabric metal people water tile rock foliage soil open hard light_panel vibrating_wood thin_carpet curtain foam acoustic_plaster audience_seated car_glass car_floor car_seat car_trim"),
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


def room_metrics(ir, sr: int = DEFAULT_SR) -> dict:
    """T20 from the Schroeder integral (-5..-25 dB, x 3), and clarity C50, C80 = 10 log10(E_early / E_late) (Meyer 5.4, chunk_01 l.8728)."""
    h = to_mono(np.asarray(ir, float)) if np.ndim(ir) == 2 else np.asarray(ir, float)
    start = int(np.argmax(np.abs(h) > 0.1 * np.max(np.abs(h))))
    e = h[start:] ** 2
    edc = np.cumsum(e[::-1])[::-1]
    db = 10.0 * np.log10(edc / edc[0] + 1e-30)
    i0, i1 = np.nonzero(db <= -5.0)[0][0], np.nonzero(db <= -25.0)[0][0]
    cl = lambda ms: 10.0 * math.log10(np.sum(e[: int(ms * sr / 1000)]) / (np.sum(e[int(ms * sr / 1000):]) + 1e-30))
    return {"t20": float(-60.0 / np.polyfit(np.arange(i0, i1) / sr, db[i0:i1], 1)[0]), "c50": cl(50), "c80": cl(80)}


def _modal_low_end(xd, wl, wr, info, sr, distance):
    """Late field below the Schroeder frequency as the box's modes (Kuttruff 3.15, damping delta = 6.91/T per band), above it the
    network. Each mode is scaled by its mode shape at source and listener. ponytail: 2nd-order crossover; the level is set so the
    modal IR has the energy a unit-energy white tail has in that band, which is flat only on average."""
    fs = float(np.clip(info["schroeder_hz"], 40.0, 0.2 * sr))
    f, order = _space.room_modes(*info["dims"], fs)
    if f.size == 0:
        return wl, wr
    src, lis = _room_positions(info, distance)
    L = np.asarray(info["dims"], float)
    shape = np.prod(np.cos(np.pi * order * src / L) * np.cos(np.pi * order * lis / L), axis=1)
    delta = 6.91 / np.interp(np.log(f), np.log(BANDS), info["t60"])
    n = samples(min(1.2 * float(np.max(info["t60"])), 6.0), sr)
    t = np.arange(n) / sr
    ir = np.zeros(n)
    for a, fk, dk in zip(shape, f, delta):
        ir += a * np.cos(2.0 * np.pi * fk * t) * np.exp(-dk * t)
    ir *= math.sqrt(fs / (0.5 * sr)) / (math.sqrt(np.sum(ir * ir)) + 1e-12)
    xl, xr = _lr(xd)
    mono = 0.5 * (xl + xr)
    low = fftconvolve(mono, ir)[: mono.shape[0]]
    hp = butter(2, fs, "high", fs=sr, output="sos")
    return sosfilt(hp, wl) + low, sosfilt(hp, wr) + low


@effect("room", "A physical room: early reflections from its shape, decay per frequency band from its volume and wall materials (Sabine), direct-to-reverb balance from the listening distance. Presets from a closet to a canyon, or your own dimensions and materials. Stereo output.",
        mix=(0.5, "0 = dry, 1 = what a listener at `distance` hears (direct + reflections + reverberation)"),
        model=("fdn", "late reverb engine: fdn (16-line Jot network) | zita (zita-rev1) | puckette (Puckette G08, one decay time) | farnell (Farnell's tight prime-delay room, fixed about 0.5 s) | moorer (6 low-passed combs and an allpass) | sdn (scattering delay network: one node per wall, exact first echoes) | raytrace (rays with diffuse walls, energy histogram turned into noise; slow)"),
        width=(1.0, "stereo width of the reverberation 0..1"),
        predelay=(0.0, "extra seconds before the late reverb (the room adds its own: mean free path / c)"),
        humidity=(50, "relative humidity % (40..70) for air absorption"),
        tail=(None, "seconds of tail appended (auto = 1.2 x the longest decay, max 10)"),
        formula=("sabine", "decay formula: sabine | eyring | millington | kuttruff (the last three give shorter, truer decays in dead rooms)"),
        source_gain=(1.0, "directivity g of the source aimed at the listener: the critical distance grows by sqrt(g), so less room (voice about 2.5 at 1 kHz)"),
        source_radius=(None, "or: radius in metres of a piston source (loudspeaker, horn mouth); g comes from its size at 1 kHz"),
        modal=(False, "true = low end from the room's own resonances below the Schroeder frequency (small rooms boom at their modes)"),
        audience=(False, "true = rows of seats between source and listener: a dip around 150 Hz in the direct sound, deeper with distance"),
        scattering=(None, "diffusing walls: 0..1, or rpg_skyline | rpg_qrd | theatre_audience | flat: early echoes smeared into a diffuse wash"),
        mic=(1.0, "microphone pattern 0..1: 1 omni, 0.5 cardioid aimed at the source, 0 figure of eight (less room)"),
        rays=(20000, "model raytrace: number of rays (1e4..1e6; more = smoother, slower)"),
        **_ROOM_PARAMS)
def room(x, sr=DEFAULT_SR, mix=0.5, model="fdn", width=1.0, predelay=0.0, humidity=50, tail=None, preset="hall",
         lx=None, ly=None, lz=None, walls=None, floor=None, ceiling=None, distance=None, furnish=None, formula="sabine",
         source_gain=1.0, source_radius=None, modal=False, audience=False, scattering=None, mic=1.0, rays=20000):
    """Direct + image-source early reflections + late reverberation.

    Late level: reverberant / direct energy = (r / r_c)^2 with the critical distance
    r_c = sqrt(g A / (16 pi)) (Kuttruff 5.41, absorption area A; capped at +10 dB), less 0.85 dB per 10 m from the source
    (Barron & Lee, in Meyer 5.4, chunk_01 l.8533-8538). The FDN gets T60 at 250 Hz
    and 4 kHz from the Sabine bands and its line lengths from the mean free path 4V/S; ``zita`` maps
    the bands with ``space.zita_from_bands``; ``puckette`` uses the 500-1000 Hz T60."""
    x = np.asarray(x, float)
    info = room_info(preset, lx, ly, lz, walls, floor, ceiling, humidity, furnish, str(formula))
    t60 = info["t60"]
    t_long = float(np.max(t60))
    if tail is None:
        tail = min(1.2 * t_long + 0.2, 10.0)
    ir, pad, e_er, d0 = early_ir(info, sr, distance, 3, True, scattering, mic)
    xp = _pad(x, samples(tail, sr) + ir.shape[0])
    er = _apply_er(xp, ir, pad)
    a_mic = float(np.clip(mic, 0.0, 1.0))
    diffuse_pick = a_mic ** 2 + (1.0 - a_mic) ** 2 / 3.0
    pd = int(min(info["mfp"] / C_AIR, 0.25) * sr) + (samples(predelay, sr) if predelay > 0 else 0)
    xd = _pad(xp, 0, front=pd)[: xp.shape[0]]
    model = str(model).lower()
    t_mid = float(np.mean(t60[2:4]))
    phys_e = None
    if model == "fdn":
        xl, xr = _lr(xd)
        wl, wr = _late_fdn(xl, xr, sr, float(t60[1]), float(t60[5]), info["mfp"], 16, "householder", 0.7, 1.0)
    elif model == "zita":
        zp = _space.zita_from_bands(t60, rdel=1.0)
        zp["t60dc"], zp["t60m"] = float(np.clip(zp["t60dc"], 0.1, 30)), float(np.clip(zp["t60m"], 0.1, 30))
        w = _kernel_late("zita", xd, sr, zp, 1.3 * max(zp["t60dc"], zp["t60m"]) + 0.3)
        wl, wr = w[:, 0], w[:, 1]
    elif model == "puckette":
        # mapping: the optional one-pole damping follows the 4 kHz / mid decay ratio
        kw = {"t60": t_mid, "damp_hz": float(np.clip(6000.0 * t60[5] / t_mid, 800.0, 12000.0))}
        w = _kernel_late("puckette", xd, sr, kw, 1.3 * t_mid + 0.3)
        wl, wr = w[:, 0], w[:, 1]
    elif model == "farnell":
        w = _kernel_late("farnell", xd, sr, {"lowpass_hz": 400.0, "stage1_fb": 0.0, "liveness": 0.33}, 2.0)
        wl, wr = w[:, 0], w[:, 1]
    elif model == "moorer":
        xl, xr = _lr(xd)
        wl, wr = moorer_late(0.5 * (xl + xr), sr, float(t60[1]), float(t60[5]))
        hh = moorer_late(_pad(np.array([1.0]), samples(min(t_long, 4.0), sr)), sr, float(t60[1]), float(t60[5]))[0]
        e = math.sqrt(float(np.sum(hh ** 2)) + 1e-20)
        wl, wr = wl / e, wr / e
    elif model in ("sdn", "raytrace"):
        mono = to_mono(xp)
        if model == "sdn":
            imp = np.zeros(samples(min(t_long, 4.0), sr))
            imp[0] = 1.0
            hl, hr = sdn_late(imp, sr, info, distance)
            wl, wr = sdn_late(mono, sr, info, distance)
        else:
            hist = raytrace_hist(info, distance, int(np.clip(rays, 1000, 1000000)), 0.1 if scattering is None else float(np.mean(_scatter_bands(scattering))))
            hl = histogram_ir(hist, 0.005, sr, info["volume"], seed=1)
            hr = histogram_ir(hist, 0.005, sr, info["volume"], seed=2)
            wl, wr = fftconvolve(mono, hl)[:mono.shape[0]], fftconvolve(mono, hr)[:mono.shape[0]]
        phys_e = 0.5 * float(np.sum(hl ** 2) + np.sum(hr ** 2))
        wl, wr = wl / math.sqrt(phys_e + 1e-20), wr / math.sqrt(phys_e + 1e-20)
        er = np.zeros_like(er)
        e_er = 0.0
    else:
        raise ValueError(f"unknown room model {model!r}; choose fdn, zita, puckette, farnell, moorer, sdn or raytrace")
    if modal:
        wl, wr = _modal_low_end(xd, wl, wr, info, sr, distance)
    g = _space.piston_gain(2.0 * math.pi * 1000.0 * float(source_radius) / C_AIR) if source_radius is not None else max(float(source_gain), 1e-3)
    e_late = (phys_e / g if phys_e is not None else float(min((d0 / info["critical_distance"]) ** 2 / g, 10.0)) * 10.0 ** (-0.085 * d0 / 10.0) * float(info["late"])) * diffuse_pick
    late = _ms_width(wl, wr, width) * math.sqrt(e_late)
    direct = to_stereo(xp)
    if audience:
        # Meyer 5.3, chunk_01 l.8481-8501: resonant dip at 130-170 Hz, 10-20 dB at 20-25 m, about nothing at 2 m.
        # linear law between those two distances and the log-Gaussian shape of the dip, -3 dB at its edges
        depth = float(np.interp(d0, (2.0, 22.5), (0.0, 15.0)))
        nf = 1 << int(np.ceil(np.log2(direct.shape[0] + int(0.05 * sr))))
        lf = np.log(np.maximum(np.fft.rfftfreq(nf, 1.0 / sr), 1.0) / 150.0)
        w = math.log(170.0 / 150.0) / math.sqrt(2.0 * math.log(depth / 3.0)) if depth > 3.0 else 0.13
        mag = 10.0 ** (-depth * np.exp(-0.5 * (lf / w) ** 2) / 20.0)
        direct = np.fft.irfft(np.fft.rfft(direct, nf, axis=0) * mag[:, None], nf, axis=0)[: direct.shape[0]]
    full = (direct + er + late) / math.sqrt(1.0 + e_er + e_late)
    return _fade_end(_wetdry(xp, full, mix), sr)


def coupled_decay(t60_near, t60_far, v_near, v_far, aperture, c=C_AIR):
    """Coupled rooms, listener and source in the deader one (Kuttruff 9.8-9.9a): delta = 6.91 / T, C = (c S'/8)^2 / (V1 V2 (d1 - d2)^2),
    the bend at L = 10 / (1 - d1/d2) log10(1/C) - 3 dB below the start. Returns (d2, d1, C, L); 1 = the livelier room."""
    d2, d1 = 6.91 / float(t60_near), 6.91 / float(t60_far)
    C = (c * float(aperture) / 8.0) ** 2 / (float(v_near) * float(v_far) * (d1 - d2) ** 2)
    return d2, d1, C, 10.0 / (1.0 - d1 / d2) * math.log10(1.0 / C) - 3.0


@effect("coupled_room", "Two rooms joined by an opening, heard from the deader one (side chapel off a nave, cave gallery, stage behind a door): a quick first decay, then the livelier room's long tail through the opening. Stereo output.",
        mix=(0.5, "wet amount 0..1"),
        t60_near=(1.0, "decay time of the room you are in, seconds"),
        t60_far=(5.0, "decay time of the other room, seconds (longer than t60_near)"),
        v_near=(500.0, "volume of the room you are in, m3"), v_far=(5000.0, "volume of the other room, m3"),
        aperture=(10.0, "area of the opening between them, m2 (the bigger, the louder the tail)"),
        seed=(0, "variation of the tails"))
def coupled_room(x, sr=DEFAULT_SR, mix=0.5, t60_near=1.0, t60_far=5.0, v_near=500.0, v_far=5000.0, aperture=10.0, seed=0):
    """Two velvet tails; impulse-response energy density exp(-2 d2 t) + C exp(-2 d1 t), which crosses at Kuttruff's bend (9.9)."""
    tn = float(np.clip(t60_near, 0.1, 20.0))
    tf = float(np.clip(t60_far, tn * 1.05, 30.0))
    d2, d1, C, _ = coupled_decay(tn, tf, v_near, v_far, aperture)
    C = min(C, 1.0)
    n = samples(min(1.3 * tf, 15.0), sr)
    x = np.asarray(x, float)
    wet = []
    for ch, xc in enumerate(_lr(_pad(x, n))):
        a = velvet_ir(sr, tn, tn, length=n / sr, seed=int(seed) + 2 * ch)
        b = velvet_ir(sr, tf, tf, length=n / sr, seed=int(seed) + 2 * ch + 1)
        ir = a * math.sqrt(1.0 / (2.0 * d2)) + b * math.sqrt(C / (2.0 * d1))    # velvet_ir has unit energy, total of exp(-2 d t) is 1/(2 d)
        ir /= math.sqrt(np.sum(ir * ir)) + 1e-12
        wet.append(fftconvolve(xc, ir)[: xc.shape[0]])
    return _fade_end(_wetdry(_pad(x, n), np.stack(wet, axis=1), mix), sr)


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

def _air_filter(x, sr, metres, rh, temp_c=20.0):
    """Zero-phase magnitude filter of ``metres`` of air (ISO 9613-1 via space.air_db_per_m)."""
    n = x.shape[0]
    nfft = 1 << int(np.ceil(np.log2(n + int(0.02 * sr))))
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    mag = 10.0 ** (-_space.air_db_per_m(f, rh, temp_c) * metres / 20.0)
    X = np.fft.rfft(x, nfft, axis=0)
    return np.fft.irfft(X * (mag if x.ndim == 1 else mag[:, None]), nfft, axis=0)[:n]


def air_cutoff(metres, rh: int = 50):
    """Frequency (Hz) at which ``metres`` of air removes 3 dB (ISO 9613-1, inverted numerically)."""
    f = np.geomspace(100.0, 40000.0, 400)
    r3 = 3.0 / np.maximum(_space.air_db_per_m(f, rh), 1e-12)        # metres for 3 dB at f
    r3 = np.minimum.accumulate(r3)                                   # monotone decreasing in f
    return np.interp(np.asarray(metres, float), r3[::-1], f[::-1])


_GROUND_SIGMA = {"grass": 300.0, "hard": 0.0, "snow": 10.0, "old_snow": 30.0, "forest": 50.0, "dirt": 550.0,
                 "gravel": 53.0, "packed_earth": 6000.0, "asphalt": 20000.0, "sand": 253.0, "sharp_sand": 906.0,
                 "dry_loam": 370.0, "wet_loam": 218.0, "clay": 106.0, "rolled_clay": 168.0, "silt": 110.0,
                 "compacted_silt": 1045.0}


def _ground_sigma(ground) -> float:
    sigma = _GROUND_SIGMA.get(str(ground))
    if sigma is None:
        try:
            sigma = float(ground)
        except ValueError:
            raise ValueError(f"ground must be none, {', '.join(_GROUND_SIGMA)} or a flow resistivity in cgs units, not {ground!r}")
    return sigma


@effect("distance", "Put a sound some metres away: 1/r level drop, air absorption that dulls it with distance and humidity, and optionally more reverberation relative to the direct sound.",
        metres=(10.0, "distance to the source, metres (the input is the sound at `ref` metres)"),
        air=(True, "true = air absorption low-pass"),
        humidity=(50, "relative humidity % (10..100; drier air absorbs more highs)"),
        temperature=(20.0, "air temperature, C (-20..50): sets the absorption and the speed of sound"),
        ground=("none", "reflection off the ground between them: none | grass (flow resistivity 300 cgs) | snow (fresh, 10 cgs) | old_snow (settled, 30 cgs) | forest (50) | dirt (550) | gravel (53) | packed_earth (6000) | asphalt (20000) | sand (253) | sharp_sand (906) | dry_loam (370) | wet_loam (218) | clay (106) | rolled_clay (168) | silt (110) | compacted_silt (1045) | hard (rigid) | a number = flow resistivity in cgs units; over grass the low end gains 6 dB and a dip opens at 300-800 Hz"),
        source_height=(1.5, "metres, for `ground`"), listener_height=(1.5, "metres, for `ground`"),
        reverb=(0.0, "reverb send 0..1: reverberant level stays constant while the direct sound falls with distance (stereo out when > 0)"),
        reverb_time=(1.5, "decay time of that reverb, seconds"),
        ref=(1.0, "reference distance in metres where gain = 1 (closer than this the gain is capped at +6 dB)"),
        line_source=(False, "true = line source (road, river): 1/sqrt(r), -3 dB per doubling"),
        delay=(False, "true = prepend the travel time (r - ref) / c, c = 343 m/s at 20 C"),
        source_radius=(None, "radius in metres of a piston source (loudspeaker, horn or muzzle mouth), with `angle`: off axis it loses its highs"),
        angle=(0.0, "degrees off the axis of that source, 0 = aimed at the listener, 90 = side on"),
        pipe=(False, "true = that source is the open end of a pipe (exhaust, horn bell, muzzle, mouth): it also radiates sideways and behind, the highs still beam forward"),
        wall_distance=(None, "metres from the source to a wall behind it: bass boost up to +3 dB and a dip above"),
        scatter_mfp=(None, "metres between obstacles along the path (forest, crowded hall): adds exp(-r / rs) in energy"),
        flat_height=(None, "height in metres of a flat room (open-plan office, car park) replacing 1/r spreading"),
        flat_alpha=(0.2, "mean absorption 0.05..0.95 of its floor and ceiling"),
        turbulence=(0.0, "m/s of turbulent air drifting across the path: scintillation, level and arrival time wander, more in the highs and further away (0 = still air)"),
        gradient=(0.0, "temperature gradient C/km: negative (day, -6 normal) bends sound up and opens a shadow zone far away, positive (night inversion) bends it down and carries it further"),
        rolling=(False, "true = rolling terrain: the ground dip is averaged over source heights +-2 m (softer, no deep notch)"),
        across_water=(False, "true = the source is under water (or the listener is): -29.5 dB through the surface"),
        seed=(0, "variation of the turbulence"))
def distance(x, sr=DEFAULT_SR, metres=10.0, air=True, humidity=50, reverb=0.0, reverb_time=1.5, ref=1.0,
             line_source=False, delay=False, temperature=20.0, ground="none", source_height=1.5, listener_height=1.5,
             source_radius=None, angle=0.0, wall_distance=None, scatter_mfp=None, flat_height=None, flat_alpha=0.2,
             turbulence=0.0, seed=0, gradient=0.0, rolling=False, across_water=False, pipe=False):
    """Spreading ref/r (point source, -6 dB per doubling) or sqrt(ref/r); air absorption over (r - ref)
    metres; c = 343.2 sqrt(T/293.15) m/s. Piston 2 J1(u)/u; open pipe end D F (``physical.space.pipe_directivity``).
    Wall: power factor 1 + sin(2kd)/(2kd). Obstacles: w = P/(4 pi c r^2) exp(-r/rs). Flat room: energy density
    over a diffuse floor and ceiling."""
    x = np.asarray(x, float)
    r = float(np.clip(metres, 0.01, 1e5))
    ref = float(max(ref, 0.01))
    if flat_height is not None:
        al = float(np.clip(flat_alpha, 0.05, 0.95))
        fh = float(max(flat_height, 0.1))
        g = min(math.sqrt(float(_space.flat_room_energy(r, fh, al) / _space.flat_room_energy(ref, fh, al))), 2.0)
    else:
        g = min(math.sqrt(ref / r) if line_source else ref / r, 2.0)  # closer than `ref`: at most +6 dB
    if scatter_mfp is not None:
        g *= math.exp(-0.5 * max(r - ref, 0.0) / max(float(scatter_mfp), 0.01))
    extra = max(r - ref, 0.0)
    y = _air_filter(x, sr, extra, humidity, float(temperature)) * g if (air and extra > 0) else x * g
    if source_radius is not None or wall_distance is not None:
        nf = 1 << int(np.ceil(np.log2(y.shape[0] + int(0.02 * sr))))
        k = 2.0 * math.pi * np.fft.rfftfreq(nf, 1.0 / sr) / _space.speed_of_sound(float(temperature))
        mag = np.ones_like(k)
        if source_radius is not None:
            law = _space.pipe_directivity if pipe else _space.piston_directivity
            mag = mag * law(k * float(source_radius), math.radians(float(angle)))
        if wall_distance is not None:
            u = np.maximum(2.0 * k * max(float(wall_distance), 1e-3), 1e-9)
            mag = mag * np.sqrt(np.maximum(1.0 + np.sin(u) / u, 0.0))
        y = np.fft.irfft(np.fft.rfft(y, nf, axis=0) * (mag if y.ndim == 1 else mag[:, None]), nf, axis=0)[:y.shape[0]]
    if str(ground) != "none":
        sigma = _ground_sigma(ground)
        nfft = 1 << int(np.ceil(np.log2(y.shape[0] + int(0.05 * sr))))
        fq = np.fft.rfftfreq(nfft, 1.0 / sr)
        if rolling:
            Hs = [np.abs(_space.ground_effect(fq, r, max(float(source_height) + dh, 0.05), float(listener_height), sigma,
                                              float(temperature))) ** 2 for dh in (-2.0, -1.0, 0.0, 1.0, 2.0)]
            H = np.sqrt(np.mean(Hs, axis=0))
        else:
            H = _space.ground_effect(fq, r, float(source_height), float(listener_height), sigma, float(temperature))
        Y = np.fft.rfft(y, nfft, axis=0)
        y = np.fft.irfft(Y * (H if y.ndim == 1 else H[:, None]), nfft, axis=0)[:y.shape[0] + int(0.05 * sr)]
    gr = float(gradient)
    if gr != 0.0 or across_water:
        if gr > 0:
            y = y * _space.duct_gain(r, float(source_height), gr, float(temperature))
        if across_water:
            y = y * math.sqrt(_space.interface_transmission())
        if gr < 0:
            nf = 1 << int(np.ceil(np.log2(y.shape[0] + int(0.02 * sr))))
            ex = _space.shadow_excess_db(np.fft.rfftfreq(nf, 1.0 / sr), r, float(source_height), float(listener_height), gr, float(temperature))
            if np.any(ex > 0):
                mag = 10.0 ** (-ex / 20.0)
                y = np.fft.irfft(np.fft.rfft(y, nf, axis=0) * (mag if y.ndim == 1 else mag[:, None]), nf, axis=0)[:y.shape[0]]
    if turbulence > 0:
        y = _scintillate(y, sr, r, float(turbulence), _space.speed_of_sound(float(temperature)), np.random.default_rng([abs(int(seed)), 911]))
    if delay and extra > 0:
        y = _pad(y, 0, front=int(round(extra / _space.speed_of_sound(float(temperature)) * sr)))
    send = float(np.clip(reverb, 0.0, 1.0))
    if send > 0:
        t = float(np.clip(reverb_time, 0.1, 20.0))
        xp = _pad(x, samples(min(1.2 * t, 10.0), sr) + (y.shape[0] - x.shape[0]))
        xl, xr = _lr(xp)
        wl, wr = _late_fdn(xl, xr, sr, t, 0.5 * t, 8.0, 16, "householder", 0.7, 1.0)
        y = to_stereo(_pad(y, xp.shape[0] - y.shape[0])) + send * 0.5 * np.stack([wl, wr], axis=1)
    return _fade_end(y, sr)


def _scintillate(y, sr, d, v_perp, c, rng):
    """``aero.scintillate`` at any rate, with the distance ``d`` and the crossing speed ``v_perp`` per sample: one
    sequence drawn at the fastest crossing, read at t_k = sum(tau_ref / tau0_i) / fs (moving source)."""
    from . import aero as A
    n = y.shape[0]
    d = np.broadcast_to(np.asarray(d, float), (n,))
    v = np.maximum(np.broadcast_to(np.asarray(v_perp, float), (n,)), 1e-3)
    vref = float(v.max())
    s, fs = A.turbulence_sequence(n / sr, A.TURB_L / vref, rng)
    s = np.interp(np.concatenate(([0.0], np.cumsum(v[:-1] / vref))) * fs / sr, np.arange(len(s)), s)
    t = np.arange(n, dtype=float)
    lag = np.sqrt(0.5 * math.sqrt(math.pi) * A.TURB_SIGMA_MU2 * d * A.TURB_L) / float(c) * s * sr
    edges = (0.0,) + A.TURB_BANDS + (0.5 * sr,)

    def one(x):
        rest, out = np.interp(t - lag, t, x), np.zeros(n)
        for i, fc in enumerate(A.TURB_BANDS + (None,)):
            last = fc is None or fc >= 0.45 * sr
            band = rest if last else sosfiltfilt(butter(2, fc, fs=sr, output="sos"), rest)
            centre = math.sqrt(max(edges[i], 62.5) * (0.5 * sr if last else edges[i + 1]))
            sig = A.scintillation_sigma(centre, d)
            out += band * np.exp(sig * s - sig * sig)
            if last:
                return out
            rest = rest - band
    return one(y) if y.ndim == 1 else np.stack([one(y[:, k]) for k in range(y.shape[1])], axis=1)


def flyby_path(n: int, sr: int, speed: float, closest: float, pass_at: float = 0.5, c: float = C_AIR):
    """Emission times (seconds, fractional) for each output sample of a straight fly-by.

    Source on the line y = closest, x = v (tau - t0); listener at the origin. Sound emitted at tau is
    heard at t = tau + r(tau) / c; solving for tau (u = tau - t0, T = t - t0):
    (c^2 - v^2) u^2 - 2 c^2 T u + c^2 T^2 - d^2 = 0, smaller root. The output clock starts when the
    first input sample arrives. Returns (tau, distance r(tau), x position, n_out)."""
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


def path_emission(path, n: int, sr: int, c: float = C_AIR):
    """Emission times for a trajectory, listener at the origin (x right, y ahead, z up). ``path`` rows (t, x, y, z), t in
    seconds of the input, positions linear between rows and held beyond them. Sound emitted at tau arrives at tau + r(tau)/c
    (DAFX 5.4-5.5 by inversion, offline), so the source may accelerate, climb or circle. Returns (tau, r, position (m, 3),
    velocity (m, 3), n_out); the output clock starts with the first sample's arrival."""
    p = np.asarray(path, float)
    if p.ndim != 2 or p.shape[1] != 4 or p.shape[0] < 2 or np.any(np.diff(p[:, 0]) <= 0):
        raise ValueError("path must be rows of [t, x, y, z] with increasing t")
    tau = np.arange(n) / sr
    pos = np.stack([np.interp(tau, p[:, 0], p[:, i]) for i in (1, 2, 3)], axis=1)
    vel = np.gradient(pos, 1.0 / sr, axis=0)
    r = np.maximum(np.linalg.norm(pos, axis=1), 0.05)
    arrive = tau + r / c
    if np.any(np.diff(arrive) <= 0):
        raise ValueError("the path comes at the listener at or above the speed of sound; doppler needs it slower")
    n_out = max(2, int(math.ceil((arrive[-1] - arrive[0]) * sr)))
    T = arrive[0] + np.arange(n_out) / sr
    tau_out = np.interp(T, arrive, tau)
    at = lambda a: np.stack([np.interp(T, arrive, a[:, i]) for i in range(3)], axis=1)
    return tau_out, np.interp(T, arrive, r), at(pos), at(vel), n_out


def _surface_blocks(y, r, sr, height, sigma, c, block=2048):
    """Complex ground reflection along a trajectory: space.ground_effect at each block's distance, Hann overlap-add
    (Damiano et al. 2024 eq. 9-14). Source and listener share ``height``. ponytail: one H per 1024-sample hop; a finer hop if
    the dips must glide faster than that at very close passes."""
    n = y.shape[0]
    hop = block // 2
    w = np.hanning(block + 1)[:-1]
    nfft = 2 * block
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    pad = np.concatenate([np.zeros(hop), y, np.zeros(block)])
    rp = np.concatenate([np.full(hop, r[0]), r, np.full(block, r[-1])])
    out = np.zeros(pad.shape[0] + nfft)
    temp = (float(c) / 343.2) ** 2 * 293.15 - 273.15       # ground_effect takes c through the temperature
    for k in range(0, pad.shape[0] - block + 1, hop):
        H = _space.ground_effect(f, float(rp[k + hop]), 0.5 * height, 0.5 * height, sigma, temp)
        out[k:k + nfft] += np.fft.irfft(np.fft.rfft(pad[k:k + block] * w, nfft) * H, nfft)
    return out[hop:hop + n]


def _chowning_reverb(z, r, rref, th, sr, t60, stereo):
    """Chowning 1971 p.144-147: reverberant amplitude 1/sqrt(D) (D = distance / reference, held at 1 inside it), split into global
    (1/D)(1/sqrt D), equal on all channels, and local (1 - 1/D)(1/sqrt D), panned like the direct sound; one reverberator per channel
    with its own delays. Local pan x sqrt 2 so a centred source gives exactly 1/sqrt D (his pan law is not printed)."""
    D = np.maximum(r / rref, 1.0)
    glob, loc = 1.0 / (D * np.sqrt(D)), (1.0 - 1.0 / D) / np.sqrt(D)
    pad = samples(min(1.2 * t60, 10.0), sr)
    ext = lambda a: np.concatenate([a, np.zeros(pad)])
    if not stereo:
        s = ext(z * (glob + loc))
        return _late_fdn(s, s, sr, t60, 0.5 * t60, 8.0, 16, "householder", 0.7, 1.0)[0]
    sl, sr_in = ext(z * (glob + loc * math.sqrt(2.0) * np.cos(th))), ext(z * (glob + loc * math.sqrt(2.0) * np.sin(th)))
    wl = _late_fdn(sl, sl, sr, t60, 0.5 * t60, 8.0, 16, "householder", 0.7, 1.0)[0]
    wr = _late_fdn(sr_in, sr_in, sr, t60, 0.5 * t60, 8.0 * 1.13, 16, "householder", 0.7, 1.0)[1]
    return np.stack([wl, wr], axis=1)


@effect("doppler", "Fly-by: the sound passes the listener on a straight line, with the true Doppler pitch drop (c/(c-v) approaching, c/(c+v) leaving), 1/r swell, air absorption and a pan across. Uses the whole length of the input.",
        speed=(20.0, "source speed in m/s (20 = 72 km/h car, 60 = racing car, 250 = jet)"),
        closest=(5.0, "closest distance to the listener in metres (small = sudden pass, large = slow swell)"),
        pass_at=(0.5, "when it passes, as a fraction of the sound's duration 0..1"),
        direction=("lr", "lr = left to right, rl = right to left"),
        pan=(True, "true = stereo output panned by the source direction"),
        air=(True, "true = air absorption that opens up as it approaches"),
        humidity=(50, "relative humidity % (40..70)"),
        ref=(None, "distance in metres where gain = 1 (default = `closest`, so the pass is at full level)"),
        ground=(0.0, "0..1 strength of the reflection off the ground: a comb that sweeps as the source passes (0.35 = asphalt)"),
        height=(1.8, "source height + listener height in metres, for the ground path"),
        c=(C_AIR, "speed of sound in m/s (1500 = under water)"),
        path=(None, "any trajectory instead of the straight pass: rows [t, x, y, z] in seconds and metres, listener at the origin, x right, y ahead, z up (accelerate, climb, circle); speed, closest, pass_at and direction are then ignored"),
        surface=(None, "real ground under the path instead of the `ground` comb: grass | asphalt | gravel ... | flow resistivity in cgs (same list as `distance`); dips that glide as it passes"),
        reverb=(0.0, "reverb send 0..1 (Chowning): the reverberation falls more slowly than the direct sound and, far away, comes from the source's side"),
        reverb_time=(1.5, "decay time of that reverb, seconds"),
        shadow=(False, "true = head shadow: the far ear loses its highs as the source goes off to one side"),
        horn=("none", "tyre horn directivity: none | low (wheel source: -2.5 dB beside it to +1.5 dB ahead and behind, 800-6300 Hz) | truck (high source, full band)"),
        turbulence=(False, "true = scintillation through turbulent air: level and arrival time wander, faster while the source crosses the line of sight, deeper far away"),
        law=("sincos", "pan law: sincos (constant power) | tangent (Chowning's: the image stays full in the middle)"),
        seed=(0, "variation of the turbulence"))
def doppler(x, sr=DEFAULT_SR, speed=20.0, closest=5.0, pass_at=0.5, direction="lr", pan=True, air=True,
            humidity=50, ref=None, ground=0.0, height=1.8, c=C_AIR, path=None, surface=None, reverb=0.0, reverb_time=1.5,
            shadow=False, horn="none", turbulence=False, law="sincos", seed=0):
    """Time-varying delay = exact Doppler (PASP/Doppler_Simulation; g13 §3.2): each output sample
    reads the input at its emission time with a 4th-order Lagrange interpolator (even order for
    swept delays, g13 §2.1). Pan: constant-power cos/sin law (g15 §6.1 item 7) on sin(azimuth)."""
    mono = np.ascontiguousarray(to_mono(np.asarray(x, float)))
    n = mono.shape[0]
    if path is not None:
        tau, r, pos, vel, _ = path_emission(path, n, sr, float(c))
        xs = pos[:, 0]
        rmin = float(np.min(r))
    else:
        tau, r, xs, _ = flyby_path(n, sr, speed, closest, float(np.clip(pass_at, 0.0, 1.0)), float(c))
        rmin = float(max(closest, 0.05))
        pos = np.stack([xs, np.full_like(xs, rmin), np.zeros_like(xs)], axis=1)
        vel = np.broadcast_to(np.array([1.0, 0.0, 0.0]), pos.shape)
    z = _read_l4(mono, np.ascontiguousarray(tau * sr))
    rref = float(rmin if ref is None else max(ref, 0.01))
    y = z * np.minimum(rref / r, 2.0)
    if surface is not None and str(surface) != "none":
        y = _surface_blocks(y, r, sr, float(height), _ground_sigma(surface), float(c))
    elif ground:
        # Image source below the ground: path longer by sqrt(r^2 + h^2) - r, phase-inverted (grazing
        # incidence on a hard surface; the -0.35 of Agent00PED/ice-simulator audio/spatial.js).
        # ponytail: reads the direct signal late instead of solving a second emission time, and one
        # broadband coefficient; a Delany-Bazley impedance filter if the surface must be audible.
        late = np.arange(len(y)) - (np.sqrt(r * r + float(height) ** 2) - r) / float(c) * sr
        y = y - float(ground) * (r / np.sqrt(r * r + float(height) ** 2)) * _read_l4(np.ascontiguousarray(y), np.ascontiguousarray(np.maximum(late, 0.0)))    # the longer path spreads more (Smith, PASP; Addendum c15 l.4047)
    if air:
        fc = np.minimum(air_cutoff(r, humidity), 0.45 * sr)
        y = _tv_one_pole(np.ascontiguousarray(y), np.exp(-2 * np.pi * fc / sr))
    if turbulence:
        vs = vel if path is not None else vel * float(np.clip(abs(speed), 0.0, 0.9 * float(c)))
        o = pos / r[:, None]
        vp = np.linalg.norm(vs - np.sum(vs * o, axis=1)[:, None] * o, axis=1)
        y = _scintillate(y, sr, r, vp, float(c), np.random.default_rng([abs(int(seed)), 911]))
    horn = str(horn)
    if horn != "none":
        # Damiano et al. 2024 eq. 15-16: phi = angle between the motion and the listener seen from the source, psi = elevation.
        # eq. 16 is folded onto phi in [0, pi/2] (fore and aft alike, as eq. 15 is by |cos|): the paper's own range is in Fig. 5.
        sp = np.maximum(np.linalg.norm(vel, axis=1), 1e-9)
        cphi = np.clip(np.abs(np.sum(pos * vel, axis=1)) / (sp * r), 0.0, 1.0)
        cpsi = np.sqrt(np.maximum(1.0 - (pos[:, 2] / r) ** 2, 0.0))
        if horn == "low":
            dl = (-2.5 + 4.0 * cphi) * np.sqrt(cpsi)
            band = sosfiltfilt(butter(2, [800.0, min(6300.0, 0.45 * sr)], "band", fs=sr, output="sos"), y)
            y = y + (10.0 ** (dl / 20.0) - 1.0) * band
        elif horn == "truck":
            u = np.arcsin(cphi)                                  # pi/2 - phi
            y = y * 10.0 ** ((1.546 * u ** 3 - 1.425 * u ** 2 + 0.22 * u + 0.6) * np.sqrt(cpsi) / 20.0)
        else:
            raise ValueError("horn must be none, low or truck")
    s = xs / r * (-1.0 if str(direction).lower() == "rl" and path is None else 1.0)
    th = (s + 1.0) * np.pi / 4
    if pan:
        yl, yr = y, y
        if shadow:
            # far ear: (1 - 0.8 |p|) s + 0.8 |p| LP(s), one pole at 1.5 kHz (Agent00PED/ice-simulator audio/spatial.js)
            a = math.exp(-2 * math.pi * 1500.0 / sr)
            m = 0.8 * np.abs(s)
            far = (1.0 - m) * y + m * lfilter([1.0 - a], [1.0, -a], y)
            yl, yr = np.where(s > 0, far, y), np.where(s < 0, far, y)
        if law == "tangent":
            tn = np.tan(th - np.pi / 4)
            y = np.stack([yl * np.sqrt(0.5 * (1.0 - tn)), yr * np.sqrt(0.5 * (1.0 + tn))], axis=1)
        else:
            y = np.stack([yl * np.cos(th), yr * np.sin(th)], axis=1)
    if reverb:
        w = _chowning_reverb(z, r, rref, th, sr, float(np.clip(reverb_time, 0.1, 20.0)), bool(pan))
        y = _pad(y, w.shape[0] - y.shape[0]) + float(np.clip(reverb, 0.0, 1.0)) * w
    return _fade_end(y, sr, head_ms=2.0)


# Volume density kg/m^3.
_DENSITY = {"concrete": 2300.0, "brick": 1922.0, "marble": 2563.0, "rock": 2600.0, "tile": 2300.0, "wood": 500.0,
            "glass": 2500.0, "plaster": 849.0, "metal": 7900.0, "fabric": 300.0, "carpet": 300.0, "water": 1000.0,
            "soil": 1500.0, "foliage": 30.0, "people": 1000.0}


# Rear of a brass bell against its axis, dB at Hz: Meyer ch. 4 and annex (lot B07); omni limit trumpet 500 Hz (c00-c02).
# trombone's start of the fall, and the (1 - cos)/2 law between axis and rear
_BELL_REAR = {"french_horn": ((300.0, 0.0), (1700.0, -20.0), (5000.0, -20.0)),
              "trumpet": ((500.0, 0.0), (2000.0, -10.0), (5000.0, -17.0), (10000.0, -22.0)),
              "trombone": ((500.0, 0.0), (1500.0, -10.0), (4000.0, -18.0), (8000.0, -25.0))}


_WALL_GAIN = {"french_horn": ((60.0, 6.0), (300.0, 6.0), (1500.0, 15.0), (2000.0, 15.0), (6000.0, 10.0)),
              "other": ((60.0, 6.0), (100.0, 6.0), (150.0, -3.0), (300.0, 1.0), (1000.0, 0.0))}


@effect("directivity", "What a player's instrument sounds like from off its axis: a brass bell seen from the side or behind loses its highs; a flute's two openings cancel harmonics by direction.",
        instrument=("trumpet", "trumpet | trombone | french_horn | flute"),
        angle=(180.0, "degrees: bell axis for brass (0 = aimed at you, 180 = behind); from the normal to the flute's axis (0 = in front of the player, 90 = along the tube)"),
        f0=(440.0, "flute only: the note in Hz"),
        wall=(False, "true = a wall right behind the player: +6 dB in the bass, up to +15 dB at 1.5-2 kHz for a horn whose bell faces it"),
        podium=(False, "true = the player stands on a wooden podium: +3 dB around 50 Hz"),
        split=(False, "true = the noisy part of the sound (breath, attack) has a lobe twice as wide as the tones, never more than 10 dB down"))
def directivity(x, sr=DEFAULT_SR, instrument="trumpet", angle=180.0, f0=440.0, wall=False, podium=False, split=False):
    """Zero-phase filter. Flute: D_n = |1 + (-1)^(n+1) exp(i n pi sin phi)| / 2, n = f / f0 (Meyer 4.2, chunk_01 l.6380)."""
    x = np.asarray(x, float)
    nf = 1 << int(np.ceil(np.log2(x.shape[0] + int(0.02 * sr))))
    f = np.fft.rfftfreq(nf, 1.0 / sr)
    a = math.radians(float(angle))
    if instrument == "flute":
        n = f / max(float(f0), 1.0)
        mag = np.abs(1.0 + np.cos(np.pi * (n + 1.0)) * np.exp(1j * n * np.pi * math.sin(a))) / 2.0
    elif instrument in _BELL_REAR:
        pts = np.array(_BELL_REAR[instrument])
        mag = 10.0 ** (np.interp(np.log(np.maximum(f, 1.0)), np.log(pts[:, 0]), pts[:, 1]) * 0.5 * (1.0 - math.cos(a)) / 20.0)
        from .physical.winds import OMNI_BELOW
        if instrument in OMNI_BELOW:
            mag = np.where(f < OMNI_BELOW[instrument], 1.0, mag)
    else:
        raise ValueError("instrument must be trumpet, trombone, french_horn or flute")
    lf = np.log(np.maximum(f, 1.0))
    if wall:
        pts = np.array(_WALL_GAIN["french_horn" if instrument == "french_horn" else "other"])
        mag = mag * 10.0 ** (np.interp(lf, np.log(pts[:, 0]), pts[:, 1]) / 20.0)
    if podium:
        mag = mag * 10.0 ** (np.interp(lf, np.log([25.0, 50.0, 100.0, 200.0]), [1.5, 3.0, 1.5, 0.0]) / 20.0)
    if not split:
        return np.fft.irfft(np.fft.rfft(x, nf, axis=0) * (mag if x.ndim == 1 else mag[:, None]), nf, axis=0)[: x.shape[0]]
    tonal, noisy = _hpss(x, sr)
    wide = directivity(noisy, sr, instrument, float(angle) / 2.0, f0, wall, podium, False)
    floor = 10.0 ** (-10.0 / 20.0) * math.sqrt(float(np.sum(noisy ** 2)))
    e_w = math.sqrt(float(np.sum(wide ** 2)))
    if 0.0 < e_w < floor:
        wide = wide * floor / e_w
    return directivity(tonal, sr, instrument, angle, f0, wall, podium, False) + wide


def _hpss(x, sr):
    """Tonal / noise split by median filtering of the STFT magnitude across time and across frequency."""
    from scipy.ndimage import median_filter
    from scipy.signal import istft, stft
    chans = [x] if x.ndim == 1 else [x[:, 0], x[:, 1]]
    m = x if x.ndim == 1 else x.mean(axis=1)
    Z = stft(m, sr, nperseg=1024, noverlap=768)[2]
    A = np.abs(Z)
    H = median_filter(A, size=(1, 17))
    P = median_filter(A, size=(17, 1))
    mask = H ** 2 / (H ** 2 + P ** 2 + 1e-20)
    out = []
    for c in chans:
        tc = istft(stft(c, sr, nperseg=1024, noverlap=768)[2] * mask, sr, nperseg=1024, noverlap=768)[1][: c.shape[0]]
        out.append(_pad(tc, c.shape[0] - tc.shape[0]))
    ton = out[0] if x.ndim == 1 else np.stack(out, axis=1)
    return ton, x - ton


@effect("occlude", "Sound from behind an obstacle (a wall, a door, a window, a curtain): the part that goes through the material loses its highs by the mass law, the part that bends round the edge is dulled smoothly.",
        material=("brick", "concrete brick marble rock tile wood glass plaster metal fabric carpet water soil foliage people"),
        thickness=(0.1, "thickness of the obstacle in metres (0.004 window pane, 0.04 door, 0.1 wall, 0.3 thick wall)"),
        leak=(0.15, "level 0..1 of the path that diffracts round the obstacle (0 = sealed room, 1 = a screen you can nearly see past)"),
        edge=(600.0, "Hz above which the diffracted path falls off (smaller = deeper in the shadow)"),
        path_diff=(None, "extra metres the sound travels over the top of a barrier: replaces `edge` with Maekawa's curve for the diffracted path"))
def occlude(x, sr=DEFAULT_SR, material="brick", thickness=0.1, leak=0.15, edge=600.0, path_diff=None):
    """Farnell p.87: a wall "transmits a filtered version of the sound, usually with less high
    frequencies"; Sound Rendering: "occluders damp smoothly rather than cut" (diffraction).
    Transmission: random-incidence mass law TL = 20 log10(f m) - 48 dB (m = surface density kg/m^2;
    Warnock 1985, NRC CBD-239 Eq 1), i.e. a first-order low-pass with corner 10^(48/20) / m = 251 / m Hz.
    Diffraction: one-pole low-pass at ``edge`` scaled by ``leak``."""
    if material not in _DENSITY:
        raise ValueError(f"unknown material {material!r}; choose from {sorted(_DENSITY)}")
    x = np.asarray(x, float)
    m = _DENSITY[material] * float(np.clip(thickness, 1e-4, 10.0))
    fc = float(np.clip(251.0 / m, 0.5, 0.45 * sr))
    a = math.exp(-2 * math.pi * fc / sr)
    y = lfilter([1.0 - a], [1.0, -a], x, axis=0)
    lk = float(np.clip(leak, 0.0, 1.0))
    if lk > 0 and path_diff is not None:
        # IL = 10 log10(3 + 20 N), N = 2 delta / lambda. Maekawa's empirical fit, seen only in the secondary sources of lot B30
        nf = 1 << int(np.ceil(np.log2(x.shape[0] + int(0.02 * sr))))
        N = 2.0 * max(float(path_diff), 0.0) * np.fft.rfftfreq(nf, 1.0 / sr) / C_AIR
        mag = 10.0 ** (-np.log10(3.0 + 20.0 * N) / 2.0)
        y = y + lk * np.fft.irfft(np.fft.rfft(x, nf, axis=0) * (mag if x.ndim == 1 else mag[:, None]), nf, axis=0)[:x.shape[0]]
    elif lk > 0:
        e = math.exp(-2 * math.pi * float(np.clip(edge, 50.0, 0.45 * sr)) / sr)
        y = y + lk * lfilter([1.0 - e], [1.0, -e], x, axis=0)
    y = sosfilt(butter(1, 20.0, "high", fs=sr, output="sos"), y, axis=0)
    return _fade_end(y, sr)


# ==========================================================================================
# Rotary speaker (Leslie)
# ==========================================================================================

_LESLIE_RATES = {"slow": (50 / 60, 40 / 60), "fast": (6.92, 5.9), "brake": (0.0, 0.0)}     # (horn Hz, drum Hz)
_LESLIE_TAU = (1.0, 3.5)
_HORN_R = 0.19
_DRUM_R = 0.15
_CAB = 0.265
_CAB_GAIN = 0.35
_HORN_FLOOR = 0.25


def _soft_clip(c):
    """Cubic soft clipper: f(x) = x - x^3/3 for |x| <= 1, +-2/3 beyond."""
    c = np.clip(c, -1.0, 1.0)
    return c - c ** 3 / 3.0


def _leslie_split(x, fc, sr):
    """(low, high) of a 12 dB/octave Linkwitz-Riley split; the high band is inverted so low + high is flat."""
    lp1, hp1 = (butter(1, fc, k, fs=sr, output="sos") for k in ("low", "high"))
    return (np.ascontiguousarray(sosfilt(lp1, sosfilt(lp1, x))),
            np.ascontiguousarray(-sosfilt(hp1, sosfilt(hp1, x))))


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
        crossover=(800.0, "horn/drum crossover Hz (12 dB/octave)"),
        leak=(-10.0, "dB of horn sound leaking straight out of the cabinet without rotating (-60 = none)"))
def leslie(x, sr=DEFAULT_SR, speed="fast", start=None, drive=0.3, mic=0.6, spread=90.0, mix=1.0, crossover=800.0, leak=-10.0):
    """Soft-clip overdrive -> 12 dB/octave crossover (Linkwitz-Riley 2, high band inverted so the sum
    stays flat) -> (1) rotating horn as a source on a circle read through a delay distance(t)/c with 1/r
    gain, (2) one cabinet-wall image source with its own Doppler shift, (3) rotating drum as amplitude
    modulation plus a modulated low-pass cutoff. Far field: w_l = w_s [1 - (r_s w_m / c) sin(w_m t)]."""
    x = np.asarray(x, float)
    mono = to_mono(x)
    n = mono.shape[0]
    if speed not in _LESLIE_RATES or (start is not None and start not in _LESLIE_RATES):
        raise ValueError("leslie speed/start must be slow, fast or brake")
    dr = float(np.clip(drive, 0.0, 1.0))
    if dr > 0:
        pre = 1.0 + 8.0 * dr
        mono = _soft_clip(pre * mono) / math.sqrt(pre)
    fc = float(np.clip(crossover, 100.0, 0.4 * sr))
    lo, hi = _leslie_split(mono, fc, sr)
    f1 = _LESLIE_RATES[speed]
    f0 = _LESLIE_RATES[start] if start is not None else f1
    th_h = _rotor_angle(n, sr, f0[0], f1[0], _LESLIE_TAU[0])
    th_d = -_rotor_angle(n, sr, f0[1], f1[1], _LESLIE_TAU[1])    # drum turns the other way
    rl = float(np.clip(mic, 0.3, 3.0))
    idx = np.arange(n, dtype=np.float64)
    half = math.radians(float(np.clip(spread, 0.0, 180.0))) / 2
    lk = 10.0 ** (float(np.clip(leak, -60.0, 0.0)) / 20.0) if leak > -60 else 0.0
    leak_path = lk * _read_l4(hi, idx - _HORN_R / C_AIR * sr)
    chans = []
    for phi in (-half, half):
        # horn: direct path
        cs = np.cos(th_h - phi)
        d = np.sqrt(rl * rl + _HORN_R ** 2 - 2 * rl * _HORN_R * cs)
        near = rl - _HORN_R
        h = _read_l4(hi, idx - (d - near) / C_AIR * sr) * (near / d)
        h = h * (_HORN_FLOOR + (1.0 - _HORN_FLOOR) * 0.5 * (1.0 + cs))
        h = _tv_one_pole(np.ascontiguousarray(h), np.exp(-2 * np.pi * np.minimum(2000.0 + 7000.0 * 0.5 * (1.0 + cs), 0.45 * sr) / sr))
        # horn: image in the cabinet wall behind the rotor (opposite phase of the rotation)
        d2 = np.sqrt(rl * rl + _HORN_R ** 2 + 2 * rl * _HORN_R * cs) + 2 * _CAB
        if _CAB_GAIN > 0:
            h = h + _CAB_GAIN * _read_l4(hi, idx - (d2 - near) / C_AIR * sr) * (near / d2)
        # drum: Doppler (small), tremolo, moving low-pass
        cd = np.cos(th_d - phi)
        dd = np.sqrt(rl * rl + _DRUM_R ** 2 - 2 * rl * _DRUM_R * cd)
        b = _read_l4(lo, idx - (dd - (rl - _DRUM_R)) / C_AIR * sr) * (1.0 - 0.35 * 0.5 * (1.0 - cd))   # depth
        b = _tv_one_pole(np.ascontiguousarray(b), np.exp(-2 * np.pi * (fc * (0.5 + 0.5 * 0.5 * (1.0 + cd)) + 100.0) / sr))
        chans.append(h + b + leak_path)
    wet = np.stack(chans, axis=1)
    wet = sosfilt(butter(1, 25.0, "high", fs=sr, output="sos"), wet, axis=0)
    return _fade_end(_wetdry(x, wet, mix), sr)


# ==========================================================================================
# Flanger, phaser, chorus, vibrato  (PASP ch. 5 and 8.9; g13 §3.1, §3.4, §4.1)
# ==========================================================================================

@numba.njit(cache=True)
def _flanger_core(x, dly, fb, damp=0.0):
    """v(n) = u(n - M(n)), u(n) = x(n) + fb v(n): interpolated delay with true recursive regeneration; `damp` is a one-pole low-pass on what is fed back."""
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
    lp = 0.0
    for k in range(n):
        d = dly[k]
        if d < 3.5:
            d = 3.5
        v = _ring_l4(buf, L, float(wp), d, w)
        lp = (1.0 - damp) * v + damp * lp
        buf[wp] = x[k] + fb * lp
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
        stereo=(0.0, "LFO phase offset between left and right in cycles (0.25 = wide; makes stereo output)"),
        damp=(0.0, "0..0.95 low-pass in the feedback loop: the regeneration dulls"),
        deep=(False, "true = notches stay at zero whatever the feedback (level rises); false = plain PASP regeneration, the notches fill in"))
def flanger(x, sr=DEFAULT_SR, rate=0.3, depth=2.0, feedback=0.5, mix=0.5, delay=1.0, invert=False, shape="sine",
            phase=0.0, stereo=0.0, damp=0.0, deep=False):
    """PASP/Flanging (g13 §3.1): y(n) = x(n) + g x[n - M(n)], notches at odd multiples of fs/(2M),
    spacing fs/M = 1/delay; invert: g -> -g; regeneration a_M from the delay output to its input. deep: the delay output is scaled by 1 +- k so the
    notches stay at zero whatever the feedback k at mix 0.5, peaks 1/(1 - |k|) (Zavalishin 11.2-11.3)."""
    x = np.asarray(x, float)
    n = x.shape[0]
    fb = float(np.clip(feedback, -0.95, 0.95))
    lo = max(float(delay), 0.2) * sr / 1000.0
    hi = lo + max(float(depth), 0.0) * sr / 1000.0
    sign = -1.0 if invert else 1.0
    mix = float(np.clip(mix, 0.0, 1.0))

    def one(sig, ph):
        d = np.maximum(_sweep(n, sr, rate, shape, ph, lo, hi), 3.5)
        v = _flanger_core(np.ascontiguousarray(sig, dtype=np.float64), np.ascontiguousarray(d), fb, float(np.clip(damp, 0.0, 0.95)))
        return sig * (1.0 - mix) + sign * mix * (1.0 + sign * fb if deep else 1.0) * v

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
        out[k] = u + fb * last           # y~[n] + k y~[n-1]: x~ goes into the mix, so the notches stay at zero (Zavalishin 11.2)
        last = u
    return out


@numba.njit(cache=True)
def _phaser2_core(x, fbase, ratios, R, sr, fb):
    """Second-order sections (a2 + a1 z^-1 + z^-2) / (1 + a1 z^-1 + a2 z^-2), a1 = -2 R cos(theta), a2 = R^2 (PASP 8.9): one notch per section."""
    n = x.shape[0]
    S = ratios.shape[0]
    x1 = np.zeros(S)
    x2 = np.zeros(S)
    y1 = np.zeros(S)
    y2 = np.zeros(S)
    out = np.zeros(n)
    last = 0.0
    for k in range(n):
        u = x[k] + fb * last
        for s in range(S):
            a1 = -2.0 * R[s] * np.cos(2.0 * np.pi * min(fbase[k] * ratios[s], 0.45 * sr) / sr)
            a2 = R[s] * R[s]
            y = a2 * u + a1 * x1[s] + x2[s] - a1 * y1[s] - a2 * y2[s]
            x2[s], x1[s] = x1[s], u
            y2[s], y1[s] = y1[s], y
            u = y
        out[k] = u + fb * last
        last = u
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
        stereo=(0.0, "LFO phase offset between left and right in cycles (makes stereo output)"),
        invert=(False, "subtract the allpass chain: notches and peaks swap"),
        mode=("allpass1", "allpass1 | notch (second-order sections: one notch per stage, independent width)"),
        notch_q=(4.0, "mode notch: the higher, the narrower"),
        style=("", "pd = Puckette's law for the first-order coefficient, c = 0.97 - 0.6 v^2, v = |phasor - 0.5| (his rate: 0.3 Hz); no depth, freq, spacing"))
def phaser(x, sr=DEFAULT_SR, rate=0.5, depth=0.7, stages=4, mix=0.5, feedback=0.0, freq=100.0, spacing=2.0,
           sweep=3.0, phase=0.0, stereo=0.0, invert=False, mode="allpass1", notch_q=4.0, style=""):
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
    sign = -1.0 if invert else 1.0
    if style == "pd":
        ratios = np.ones(S)
    if mode == "notch":                                       # pole radius from a bandwidth f / Q at the lowest stage
        ratios = float(np.clip(spacing, 1.05, 4.0)) ** np.arange(max(S // 2, 1))
        R = np.exp(-np.pi * f0 * ratios / (max(float(notch_q), 0.1) * sr))

    def one(sig, ph):
        if style == "pd":                                     # Puckette, Theory and Technique, 8.3.11 H14/H15: H = (P - z^-1) / (1 - P z^-1)
            v = np.abs(((np.arange(n) / sr * rate + ph) % 1.0) - 0.5)
            p = 0.97 - 0.6 * v * v
            fbase = np.arctan((1.0 - p) / (1.0 + p)) * sr / np.pi
        else:
            fbase = f0 * 2.0 ** (octs * (0.5 + 0.5 * _lfo(n, sr, rate, "sine", ph)))
        sig_c, fb_c = np.ascontiguousarray(sig, dtype=np.float64), np.ascontiguousarray(fbase)
        wet = _phaser2_core(sig_c, fb_c, ratios, R, float(sr), fb) if mode == "notch" else _phaser_core(sig_c, fb_c, ratios, float(sr), fb)
        return sig * (1.0 - mix) + sign * wet * mix

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
        feedback=(0.0, "regeneration -0.7..0.7 (towards a flanger)"),
        lfo=("sine", "sine | noise (low-passed noise at `rate` Hz: less regular, no audible cycle)"),
        preset=("", "white = Dattorro's white chorus: blend 0.7, feedforward 1, feedback -0.7, noise LFO (an allpass when the delay does not move, so it never colours)"),
        seed=(0, "noise LFO seed"))
def chorus(x, sr=DEFAULT_SR, rate=0.8, depth=3.0, mix=0.5, voices=2, delay=15.0, spacing=5.0, spread=0.5, feedback=0.0, lfo="sine", preset="", seed=0):
    """PASP/Chorus_Effect (g13 §3.4): interpolating taps on one delay line, each oscillating about
    its own position, each placed separately in the stereo field (4th-order Lagrange reads).
    Voice v: delay + v spacing + depth (0.5 + 0.5 sin), rate x (1 + 0.1 v), left/right LFOs in quadrature.
    Universal comb (DAFX 3.2-3.4): y = BL x_h + FF x_h[n-M], x_h = x + FB x_h[n-M], that is BL x + (FF + BL FB) v with v the output of the delay."""
    x = np.asarray(x, float)
    mono = np.ascontiguousarray(to_mono(x))
    n = mono.shape[0]
    V = int(np.clip(voices, 1, 8))
    white = preset == "white"
    if white:
        feedback, lfo, V = -0.7, "noise", 1 if voices == 2 else V
    fb = float(np.clip(feedback, -0.7, 0.7))
    rng = np.random.default_rng(int(seed))

    def mod(f, ph):
        if lfo != "noise":
            return _lfo(n, sr, f, "sine", ph)
        u = F.onepole(rng.standard_normal(n), "lowpass", max(float(f), 0.01), sr)
        return np.clip(u / (float(np.std(u)) * 2.0 + 1e-12), -1.0, 1.0)

    idx = np.arange(n, dtype=np.float64)
    l = np.zeros(n)
    r = np.zeros(n)
    sp = float(np.clip(spread, 0.0, 1.0))
    for v in range(V):
        ph = v / V
        base = max(float(delay) + float(spacing) * v, 0.5)
        outs = []
        for q in (0.0, 0.25):
            d = (base + max(float(depth), 0.0) * (0.5 + 0.5 * mod(rate * (1 + 0.1 * v), ph + q))) * sr / 1000.0
            if fb != 0:
                outs.append(_flanger_core(mono, np.ascontiguousarray(d), fb))
            else:
                outs.append(_read_l4(mono, idx - d))
        pos = 0.0 if V == 1 else (2.0 * v / (V - 1) - 1.0) * sp           # voice pan, constant power
        th = (pos + 1.0) * np.pi / 4
        l += outs[0] * math.cos(th) * math.sqrt(2.0)
        r += outs[1] * math.sin(th) * math.sqrt(2.0)
    wet = np.stack([l, r], axis=1) / V
    if white:
        return _fade_end(0.7 * to_stereo(x) + (1.0 + 0.7 * fb) * wet, sr)
    return _fade_end(_wetdry(x, wet, mix), sr)


@effect("decorrelate", "Mono into wide stereo with no delay or comb you can hear: each channel gets its own short allpass-like filter (Kendall), or a constant phase offset.",
        mode=("fir", "fir (random phase, the two channels unrelated) | phase (right channel turned by `angle`) | velvet (sparse 30 ms exponential velvet noise, one per channel)"), taps=(512, "fir length in samples, under about 20 ms"),
        angle=(90.0, "mode phase: degrees; correlation = cos(angle)"), seed=(0, "fir: first seed tried"), mix=(1.0, "against the mono input"))
def decorrelate(x, sr=DEFAULT_SR, mode="fir", taps=512, angle=90.0, seed=0, mix=1.0):
    """Kendall, The Decorrelation of Audio Signals (lots B11, B59): FIR of unit magnitude and random phase, response under about 20 ms, try several seeds and keep the
    closest to the target correlation (8 tried here, count). Constant phase: rho goes +1, 0, -1 as the offset goes 0, pi/2, pi."""
    m = to_mono(np.asarray(x, float))
    if mode == "velvet":
        # EVN30, Alary, Politis & Valimaki, DAFx-18 sec. 2, eq. 5-6 (lot B83): 30 ms, 1 impulse/ms, gamma = exp(-tau alpha), -60 dB at the end
        ns = samples(0.03, sr)
        env = 10.0 ** (-60.0 / 20.0 * np.arange(ns) / ns)
        fir = [velvet_noise(ns, sr, 1000.0, np.random.default_rng(int(seed) + i)) * env for i in (0, 1)]
        fir = [h / np.sqrt(np.sum(h * h)) for h in fir]
        l, r = fftconvolve(m, fir[0])[: m.shape[0]], fftconvolve(m, fir[1])[: m.shape[0]]
    elif mode == "phase":
        from scipy.signal import hilbert
        a = math.radians(float(angle))
        l, r = m, math.cos(a) * m + math.sin(a) * np.imag(hilbert(np.concatenate([m, np.zeros(sr // 10)])))[: m.shape[0]]
    else:
        N = int(np.clip(taps, 32, int(0.02 * sr)))

        def fir(s):
            ph = np.random.default_rng(s).uniform(-np.pi, np.pi, N // 2 + 1)
            ph[0] = ph[-1] = 0.0
            return np.fft.irfft(np.exp(1j * ph), N)

        bank = [fir(int(seed) + i) for i in range(8)]
        _, i, j = min((abs(float(np.dot(a, b))), i, j) for i, a in enumerate(bank) for j, b in enumerate(bank) if i < j)
        l, r = fftconvolve(m, bank[i])[: m.shape[0]], fftconvolve(m, bank[j])[: m.shape[0]]
    y = np.stack([l, r], axis=1)
    return _fade_end(float(mix) * y + (1.0 - float(mix)) * np.stack([m, m], axis=1), sr)


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
    Wow 0.6 Hz up to 0.4 % and flutter 7.3 Hz up to 0.08 % of the delay time. Flutter rate inside "above 6 Hz"
    (Zölzer 2002 DAFX glossary p.517); wow rate and depths are voicing choices."""
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
            seed=(0, "pluck noise seed (int)"),
            aperture=(0.0, "pickup sensing width in m or single|humbucker (0.015, 0.04), 0 = a point pickup: softens the highest harmonics"))
def feedback_guitar(freq, dur, sr=DEFAULT_SR, vel=1.0, drive=6.0, feedback=0.5, pickup=0.18, distance=1.2,
                    tone=2400.0, asym=0.15, seed=0, aperture=0.0):
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
    t60 = 5.0                                                     # solid-body sustain of the fundamental, s
    rho = 10.0 ** (-3.0 * P / (fs * t60)) / math.cos(w0 / 2)
    rho = min(rho, 0.99995)
    rho_rel = 10.0 ** (-3.0 * P / (fs * 0.07))                    # damped by the hand at note-off
    Mp = int(np.clip(round(float(np.clip(pickup, 0.05, 0.5)) * N), 1, N - 1))
    r = 0.995                                                     # leak of the integrator / differentiator
    gpre = abs(1.0 - r * np.exp(-1j * w0))                        # unity pre-gain at the fundamental
    rng = np.random.default_rng(int(seed))
    burst = rng.uniform(-1.0, 1.0, N)
    a = math.exp(-2 * math.pi * (600.0 + 2600.0 * min(vel, 1.0)) / fs)   # pick brightness vs. velocity
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
    if isinstance(aperture, str) or aperture > 0.0:
        y = np.convolve(y, pickup_window(P, aperture), mode="same")
    fc = float(np.clip(tone, 500.0, 8000.0))
    y = sosfilt(butter(4, fc, "low", fs=fs, output="sos"), y)
    y = sosfilt(butter(2, 70.0, "high", fs=fs, output="sos"), y)
    y = _resample(y, fs, sr)
    k = min(len(y) // 2, int(0.02 * sr))
    y[-k:] *= np.linspace(1.0, 0.0, k)
    return y * (0.35 + 0.65 * min(vel, 1.0)) * _FBG_GAIN


_FBG_GAIN = 0.247        # level-matched: about -12 dB K-weighted at vel 0.9 (measured)

PICKUP_WIDTH = {"single": 0.015, "humbucker": 0.04}   # m, sensing width under the string
SCALE_LENGTH = 0.648                                  # m
PICKUP_HAMMING = 0.54


def pickup_window(period, width, scale=SCALE_LENGTH, a=PICKUP_HAMMING):
    """Pickup aperture as a short FIR: a wave crosses the sensing width W in T_W = W T0 / L, the pickup
    averages over that time with h(t) = a + (1 - a) cos(2 pi t / T_W). `period` in samples, `width` in m
    or a PICKUP_WIDTH key; unit DC gain. Harmonics whose wavelength nears W are attenuated."""
    w = PICKUP_WIDTH[width] if isinstance(width, str) else float(width)
    tw = max(w * float(period) / scale, 1.0)
    m = int(tw) // 2
    t = np.arange(-m, m + 1, dtype=float)
    h = a + (1.0 - a) * np.cos(2.0 * np.pi * t / tw)
    return h / h.sum()


# Dattorro, "Effect Design Part 1", JAES 45(9), 1997, Fig. 1 and Tables 1-2, at 29761 Hz: 4 input diffusers, a figure-eight
# tank of two allpass-delay-damp-allpass-delay arms, 14 output taps. Lengths checked against Faust reverbs.lib, taps against
# superduper-dsp plate.rs; the paper itself is not in the research folder.
@numba.njit(cache=True)
def _dattorro_kernel(x, sr, decay, damping, bandwidth, d1, d2, i1, i2):
    sc = sr / 29761.0
    ex = 16.0 * sc
    # lines: 0-3 input diffusers, 4-5 modulated allpasses, 6-7 first delays, 8-9 second allpasses, 10-11 second delays
    L = np.round(np.array([142.0, 107.0, 379.0, 277.0, 908.0, 908.0, 4453.0, 4217.0, 1800.0, 2656.0, 3720.0, 3163.0]) * sc).astype(np.int64)
    L[4:6] += int(ex) + 4
    off = np.zeros(13, dtype=np.int64)
    off[1:] = np.cumsum(L)
    buf = np.zeros(off[12])
    h = np.zeros(12, dtype=np.int64)
    base = (672.0 * sc, 908.0 * sc)
    fb = np.zeros(2)                                   # end of each arm
    lp = np.zeros(2)
    bw = 0.0
    w = 2.0 * math.pi / sr                             # 1 Hz, quadrature between the arms
    c = (i1, i1, i2, i2)
    # (line, samples, sign) of the taps
    tl = ((7, 266.0, 1.0), (7, 2974.0, 1.0), (9, 1913.0, -1.0), (11, 1996.0, 1.0), (6, 1990.0, -1.0), (8, 187.0, -1.0), (10, 1066.0, -1.0))
    tr = ((6, 353.0, 1.0), (6, 3627.0, 1.0), (8, 1228.0, -1.0), (10, 2673.0, 1.0), (7, 2111.0, -1.0), (9, 335.0, -1.0), (11, 121.0, -1.0))
    out = np.zeros((x.shape[0], 2))
    for n in range(x.shape[0]):
        bw += bandwidth * (x[n] - bw)
        u = bw
        for i in range(4):
            z = buf[off[i] + h[i]]
            v = u - c[i] * z
            u = z + c[i] * v
            buf[off[i] + h[i]] = v
            h[i] = (h[i] + 1) % L[i]
        for a in range(2):
            t = u + decay * fb[1 - a]
            i = 4 + a                                  # modulated allpass: w = x + d1 z, y = z - d1 w (the paper's sign)
            ri = (h[i] - (base[a] + ex * math.sin(w * n + 0.5 * math.pi * a))) % L[i]
            k = int(ri)
            fr = ri - k
            z = (1.0 - fr) * buf[off[i] + k] + fr * buf[off[i] + (k + 1) % L[i]]
            v = t + d1 * z
            y = z - d1 * v
            buf[off[i] + h[i]] = v
            h[i] = (h[i] + 1) % L[i]
            i = 6 + a                                  # delay, then damping and the first gain
            z = buf[off[i] + h[i]]
            buf[off[i] + h[i]] = y
            h[i] = (h[i] + 1) % L[i]
            lp[a] += (1.0 - damping) * (z - lp[a])
            t = decay * lp[a]
            i = 8 + a                                  # second allpass
            z = buf[off[i] + h[i]]
            v = t - d2 * z
            y = z + d2 * v
            buf[off[i] + h[i]] = v
            h[i] = (h[i] + 1) % L[i]
            i = 10 + a                                 # second delay
            fb[a] = buf[off[i] + h[i]]
            buf[off[i] + h[i]] = y
            h[i] = (h[i] + 1) % L[i]
        l = 0.0
        r = 0.0
        for i, d, sg in tl:
            l += sg * buf[off[i] + (h[i] - 1 - int(d * sc)) % L[i]]
        for i, d, sg in tr:
            r += sg * buf[off[i] + (h[i] - 1 - int(d * sc)) % L[i]]
        out[n, 0] = 0.6 * l
        out[n, 1] = 0.6 * r
    return out


@effect("dattorro", "Dattorro's plate reverb: the classic figure-eight tank of the Lexicon 224 era, dense and smooth at once, stereo.",
        mix=(0.3, "wet amount 0..1"), decay=(0.5, "tank gain 0..0.95: longer tail"), damping=(0.0005, "0..0.9 high-frequency loss in the tank"),
        bandwidth=(0.9995, "0.05..1 input brightness"), predelay=(0.0, "s"), tail=(None, "s of tail appended (auto = decay time)"))
def dattorro(x, sr=DEFAULT_SR, mix=0.3, decay=0.5, damping=0.0005, bandwidth=0.9995, predelay=0.0, tail=None):
    """Published defaults: decay 0.5, damping 0.0005, bandwidth 0.9995, diffusion 0.75, 0.625, 0.70, 0.50, excursion 16 at 1 Hz."""
    x = np.asarray(x, float)
    g = float(np.clip(decay, 0.0, 0.95))
    if tail is None:
        tail = min(30.0, 3.0 * 0.67 / max(-math.log10(max(g, 1e-3) ** 4), 1e-3))   # 0.67 s = once round the eight, g^4 per trip
    pd = samples(predelay, sr) if predelay > 0 else 0
    dry = _pad(x, samples(tail, sr) + pd)
    mono = _pad(to_mono(x), samples(tail, sr), front=pd)
    wet = _dattorro_kernel(np.ascontiguousarray(mono, dtype=np.float64), float(sr), g, float(np.clip(damping, 0.0, 0.9)),
                           float(np.clip(bandwidth, 0.05, 1.0)), 0.70, 0.50, 0.75, 0.625)
    return _fade_end(_wetdry(dry, wet, mix), sr)


@numba.njit(cache=True)
def _gverb_kernel(x, L, g, damp, tap_d, tap_g, dif, dif_c):
    """4-line Householder FDN with damped lines, early taps, a chain of four allpasses per output."""
    n = x.shape[0]
    mx = int(L.max()) + 1
    fdn = np.zeros((4, mx))
    w = np.zeros(4, dtype=np.int64)
    lp = np.zeros(4)
    tl = np.zeros(int(tap_d.max()) + 2)
    tw = 0
    dmax = int(dif.max()) + 1
    db = np.zeros((2, 4, dmax))
    dw = np.zeros((2, 4), dtype=np.int64)
    out = np.zeros((n, 3))                             # L tail, R tail, early
    o = np.zeros(4)
    for i in range(n):
        tl[tw] = x[i]
        e = 0.0
        for k in range(4):
            e += tap_g[k] * tl[(tw - int(tap_d[k])) % tl.shape[0]]
        tw = (tw + 1) % tl.shape[0]
        for k in range(4):
            v = fdn[k, w[k]]
            lp[k] = v * (1.0 - damp) + lp[k] * damp
            o[k] = g[k] * lp[k]
        h = 0.5 * (o[0] + o[1] + o[2] + o[3])
        for k in range(4):
            fdn[k, w[k]] = o[k] - h + x[i]
            w[k] = (w[k] + 1) % L[k]
        tails = (o[0] + o[2], o[1] + o[3])
        for c in range(2):
            s = tails[c]
            for k in range(4):
                z = db[c, k, dw[c, k]]
                v = s - dif_c[k] * z
                s = z + dif_c[k] * v
                db[c, k, dw[c, k]] = v
                dw[c, k] = (dw[c, k] + 1) % dif[c, k]
            out[i, c] = s
        out[i, 2] = e
    return out


@effect("gverb", "GVerb (Griesinger-style FDN by Juhana Sadeharju): big natural halls, a room size in metres and a decay in seconds.",
        roomsize=(30.0, "m"), revtime=(5.0, "s"), damping=(0.8, "0..1 high-frequency loss"), spread=(15.0, "stereo spread of the diffusers, samples"),
        bandwidth=(0.5, "0..1 input brightness"), dry=(0.6, "linear"), early=(0.4, "linear"), tail_level=(0.5, "linear"))
def gverb(x, sr=DEFAULT_SR, roomsize=30.0, revtime=5.0, damping=0.8, spread=15.0, bandwidth=0.5, dry=0.6, early=0.4, tail_level=0.5):
    """Defaults, line ratios, taps, diffuser bases and coefficients from ChucK's GVerb (chugins). this port has not
    been compared with GVerb.cpp, whose source is not in the research folder: how the spread enters is a reading of its constants."""
    x = np.asarray(x, float)
    room = float(np.clip(roomsize, 1.0, 300.0))
    rt = float(np.clip(revtime, 0.1, 60.0))
    largest = sr * room / 340.0
    L = np.maximum(np.round(largest * np.array([1.0, 0.81649, 0.7071, 0.63245])).astype(np.int64), 2)
    alpha = 0.001 ** (1.0 / (sr * rt))
    tap_d = 5.0 + np.array([0.41, 0.3, 0.155, 0.0]) * largest
    scale = L[3] / 1341.0
    r = np.array([[0.125541, 0.854046], [-0.568366, -0.126815]])
    s1, s2 = float(spread), 3.0 * float(spread)
    dif = np.zeros((2, 4), dtype=np.int64)
    for c in range(2):
        for k, base in enumerate((210.0, 159.0, 562.0, 410.0)):
            dif[c, k] = max(2, int(scale * base + (s1 if k < 2 else s2) * r[c, k % 2]))
    mono = lfilter([float(np.clip(bandwidth, 0.01, 1.0))], [1.0, -(1.0 - float(np.clip(bandwidth, 0.01, 1.0)))], to_mono(x))
    wet = _gverb_kernel(np.ascontiguousarray(_pad(mono, samples(rt, sr))), L, -(alpha ** L.astype(float)), float(np.clip(damping, 0.0, 0.95)),
                        tap_d, alpha ** tap_d, dif, np.array([0.75, 0.75, 0.625, 0.625]))
    d = to_stereo(_pad(x, samples(rt, sr)))
    return _fade_end(float(dry) * d + float(early) * wet[:, 2:3] + float(tail_level) * wet[:, :2], sr)


# Brown & Duda, IEEE TSAP 6(5), 1998, Table I, n = 2..6: rho, A, B, D for subject PB (RD: its D).
_BD_RHO, _BD_A, _BD_B = (0.5, -1.0, 0.5, -0.25, 0.25), (1.0, 5.0, 5.0, 5.0, 5.0), (2.0, 4.0, 7.0, 11.0, 13.0)
_BD_D = {"PB": (1.0, 0.5, 0.5, 0.5, 0.5), "RD": (0.85, 0.35, 0.35, 0.35, 0.35)}


def _bd_ear(x, sr, theta, phi, D):
    """One ear; theta in degrees from that ear's axis. Head shadow + ITD (DAFX ch. 6, M-files 6.2-6.4), shoulder, pinna."""
    w0 = 334.0 / 0.08 / sr
    alfa = 1.0 + 0.025 + (1.0 - 0.025) * math.cos(theta / 150.0 * math.pi)
    y = lfilter([(alfa + w0) / (1.0 + w0), (-alfa + w0) / (1.0 + w0)], [1.0, -(1.0 - w0) / (1.0 + w0)], x)
    t = math.radians(theta)
    gd = (1.0 - math.cos(t)) / w0 if abs(theta) < 90.0 else (math.radians(abs(theta) - 90.0) + 1.0) / w0
    a = (1.0 - gd) / (1.0 + gd)
    y = lfilter([a, 1.0], [1.0, a], y)
    out = y.copy()
    def add(g, d):
        k = int(round(d))
        if 0 < k < y.size:
            out[k:] += g * y[:-k]
    ms = 1.2 * (180.0 - abs(theta)) / 180.0 * (1.0 - 0.00004 * ((phi - 80.0) * 180.0 / (180.0 + abs(theta))) ** 2)
    add(0.3, ms * 1e-3 * sr)                               # gain 0.3 (the article's varies with elevation)
    for rho, A, B, Dn in zip(_BD_RHO, _BD_A, _BD_B, D):
        add(rho, (A * math.cos(t / 2.0) * math.sin(math.radians(Dn * (90.0 - phi))) + B) * sr / 44100.0)
    return out


@effect("binaural", "Headphone position: Brown & Duda's structural HRTF (head shadow, interaural delay, shoulder, pinna echoes). Stereo.",
        azimuth=(0.0, "degrees, -90 left, 0 ahead, 90 right, 180 behind"), elevation=(0.0, "degrees, 0 level, 90 above"),
        pinna=("PB", "PB|RD: whose ears (Brown & Duda's two subjects)"))
def binaural(x, sr=DEFAULT_SR, azimuth=0.0, elevation=0.0, pinna="PB"):
    """Without room echoes it localises but stays in the head (their sec. V): chain it with `room`."""
    m = to_mono(np.asarray(x, float))
    D = _BD_D.get(str(pinna), _BD_D["PB"])
    wrap = lambda a: (a + 180.0) % 360.0 - 180.0
    az, el = float(azimuth), float(np.clip(elevation, -40.0, 90.0))
    return np.stack([_bd_ear(m, sr, wrap(az + 90.0), el, D), _bd_ear(m, sr, wrap(az - 90.0), el, D)], axis=1) * 0.7


# ==========================================================================================
# Velvet-noise variants, sparse convolution, histogram to impulse response
# ==========================================================================================

def dark_velvet_noise(n: int, sr: int, density: float = 2205.0, dark: float = 0.5, rng=None) -> np.ndarray:
    """Dark velvet noise: each +-1 impulse of a velvet sequence becomes a rectangle of width
    w(m) = round(r2 (w_max - w_min) + w_min), w_min 1, w_max = 1 + dark (Td - 1), amplitude 1/sqrt(w) (same energy).
    Wider pulses = a darker spectrum (a sinc low-pass), built from running sums."""
    rng = np.random.default_rng(0) if rng is None else rng
    td = sr / float(density)
    m = np.arange(int(n / td) + 1)
    k = np.round(m * td + rng.random(m.size) * (td - 1)).astype(int)
    keep = k < n
    k = k[keep]
    sgn = 2.0 * np.round(rng.random(k.size)) - 1.0
    w_max = 1.0 + float(np.clip(dark, 0.0, 1.0)) * max(td - 1.0, 0.0)
    w = np.maximum(np.round(rng.random(k.size) * (w_max - 1.0) + 1.0).astype(int), 1)
    d = np.zeros(n + int(w_max) + 2)
    np.add.at(d, k, sgn / np.sqrt(w))
    np.add.at(d, k + w, -sgn / np.sqrt(w))
    return np.cumsum(d)[:n]


def _band_masks(f, sr):
    lf = np.log2(np.maximum(f, 1.0))
    out = []
    for c in 2.0 ** np.arange(5, 15):
        if c > 0.45 * sr:
            break
        w = np.cos(np.clip(lf - math.log2(c), -1.0, 1.0) * math.pi / 2) ** 2
        if c == 32.0:
            w[f < c] = 1.0
        if 2 * c > 0.45 * sr:
            w[f > c] = 1.0
        out.append((c, w))
    return out


def two_slope_envelope(t, t60_early: float, t60_late: float, knee_db: float = -20.0) -> np.ndarray:
    """Amplitude envelope of a double-slope decay: -60 dB per t60_early until the level reaches knee_db,
    then -60 dB per t60_late (coupled rooms, edited tails)."""
    t = np.asarray(t, float)
    tk = -knee_db / 60.0 * t60_early
    db = np.where(t < tk, -60.0 * t / t60_early, knee_db - 60.0 * (t - tk) / t60_late)
    return 10.0 ** (db / 20.0)


def edvn_ir(sr: int, t60_low: float, t60_high: float, length: float | None = None, seed: int = 0,
            q_filters: int = 10, d0: float = 2000.0, d1: float = 500.0) -> np.ndarray:
    """Extended dark velvet noise: Q second-order band-pass filters F_q; each pulse is routed to one filter with
    probability p_q(m), found per 85 ms frame (50 % hop) by non-negative least squares so that sum |F_q| p_q matches
    the target band decay; density falls from d0 to d1 pulses/s and the pulse gain is gamma(m) sqrt(Td(m))."""
    from scipy.optimize import nnls
    from scipy.signal import freqz, iirpeak
    t60_low = float(np.clip(t60_low, 0.05, 30.0))
    t60_high = float(np.clip(t60_high, 0.05, t60_low))
    n = samples(min(1.2 * t60_low + 0.05, 12.0) if length is None else length, sr)
    rng = np.random.default_rng(seed)
    fc = np.geomspace(50.0, min(16000.0, 0.4 * sr), int(q_filters))
    filt = [iirpeak(f, 1.4, fs=sr) for f in fc]
    fg = np.geomspace(40.0, 0.45 * sr, 64)
    Fm = np.stack([np.abs(freqz(b, a, worN=fg, fs=sr)[1]) for b, a in filt], axis=1)
    xg = np.clip((np.log2(fg) - math.log2(250.0)) / 4.0, 0.0, 1.0)
    t60f = t60_low * (t60_high / t60_low) ** xg
    hop = int(0.0425 * sr)
    frames = max(1, n // hop + 1)
    P = np.zeros((frames, len(fc)))
    for j in range(frames):
        P[j] = nnls(Fm, np.exp(-6.91 * j * hop / sr / t60f))[0]
    dens = np.linspace(d0, d1, n)
    pos, t = [], 0.0
    while t < n:
        pos.append(int(t))
        t += sr / dens[min(int(t), n - 1)] * (0.5 + rng.random())
    pos = np.array(pos)
    fr = np.minimum(pos // hop, frames - 1)
    tot = P[fr].sum(axis=1)
    pick = np.array([rng.choice(len(fc), p=P[i] / s) if s > 0 else 0 for i, s in zip(fr, tot)])
    gain = np.sqrt(tot) * np.sqrt(sr / dens[pos]) * (2.0 * np.round(rng.random(pos.size)) - 1.0)
    ir = np.zeros(n)
    for q, (b, a) in enumerate(filt):
        u = np.zeros(n)
        sel = pick == q
        u[pos[sel]] = gain[sel]
        ir += lfilter(b, a, u)
    return ir / (np.sqrt(np.sum(ir * ir)) + 1e-12)


@numba.njit(cache=True)
def _sparse_conv(x, pos, sgn, seg, gseg):
    """y[n] = sum over segments g_s sum_{k in s} sign_k x[n - pos_k]: additions inside a segment, one multiply each."""
    n = x.shape[0]
    y = np.zeros(n)
    for i in range(n):
        acc = 0.0
        for s in range(gseg.shape[0]):
            part = 0.0
            for k in range(seg[s], seg[s + 1]):
                j = i - pos[k]
                if j < 0:
                    break
                part += sgn[k] * x[j]
            acc += gseg[s] * part
        y[i] = acc
    return y


def sparse_velvet_convolve(x, sr: int, t60: float, density: float = 2205.0, segment: float = 0.02, seed: int = 0):
    """Convolve with a velvet-noise tail without per-tap multiplications: the taps are +-1, grouped in segments of
    `segment` seconds each scaled once by its exponential decay."""
    n = samples(min(1.2 * t60 + 0.05, 12.0), sr)
    v = velvet_noise(n, sr, density, np.random.default_rng(seed))
    pos = np.nonzero(v)[0]
    sgn = v[pos]
    L = max(1, int(segment * sr))
    edges = np.searchsorted(pos, np.arange(0, n + L, L))
    gseg = np.exp(-6.91 * (np.arange(edges.size - 1) * L + 0.5 * L) / sr / max(t60, 0.05))
    gseg /= math.sqrt(float(np.sum(gseg ** 2 * np.diff(edges))) + 1e-12)
    return _sparse_conv(np.ascontiguousarray(x, dtype=np.float64), pos.astype(np.int64), sgn,
                        edges.astype(np.int64), gseg)


def histogram_ir(hist, bin_s: float, sr: int, volume: float = 1000.0, density: float = 10000.0, seed: int = 0) -> np.ndarray:
    """Energy histogram E(band, t_k) (6 bands 125-4000 Hz, bins of bin_s seconds) to an impulse response: a velvet
    noise of rising density 4 pi c^3 t^2 / V (capped at `density` /s) split into octave bands, each scaled per bin by
    sqrt(E / noise energy in the bin)."""
    hist = np.asarray(hist, float)
    nb = hist.shape[0]
    L = max(1, int(round(bin_s * sr)))
    n = nb * L
    rng = np.random.default_rng(seed)
    t = (np.arange(n) + 1.0) / sr
    dens = np.minimum(4.0 * math.pi * C_AIR ** 3 * t ** 2 / max(float(volume), 1.0), density)
    acc = np.cumsum(dens / sr)
    k = np.nonzero(np.diff(np.floor(acc + rng.random())) > 0)[0]
    v = np.zeros(n)
    v[k] = 2.0 * np.round(rng.random(k.size)) - 1.0
    nfft = 1 << int(np.ceil(np.log2(n)))
    V = np.fft.rfft(v, nfft)
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    lb = np.log(np.asarray(BANDS, float))
    ir = np.zeros(n)
    for c, w in _band_masks(f, sr):
        e_band = np.array([np.interp(math.log(c), lb, hist[j]) for j in range(nb)])
        nb_sig = np.fft.irfft(V * w, nfft)[:n]
        e_noise = np.add.reduceat(nb_sig ** 2, np.arange(0, n, L))
        ir += nb_sig * np.repeat(np.sqrt(np.maximum(e_band, 0.0) / np.maximum(e_noise, 1e-20)), L)[:n]
    return ir


# ==========================================================================================
# JPverb and Greyhole (rotating stereo diffusers)
# ==========================================================================================

_JP_EARLY = ((55, 240), (215, 85), (115, 190), (175, 145))
_JP_LOOP = ((61, 293), (229, 107), (149, 211), (197, 167), (89, 263))
_GREY_N = 4


@numba.njit(cache=True)
def _rot_ap(xl, xr, ang, g, dl, dr, bl, br, p):
    """One diffuser step: rotate the stereo pair by ang, then a Schroeder allpass per channel."""
    c, s = math.cos(ang), math.sin(ang)
    a = c * xl - s * xr
    b = s * xl + c * xr
    vl = bl[(p - dl) % bl.shape[0]]
    wl = a + g * vl
    bl[p % bl.shape[0]] = wl
    vr = br[(p - dr) % br.shape[0]]
    wr = b + g * vr
    br[p % br.shape[0]] = wr
    return vl - g * wl, vr - g * wr


@numba.njit(cache=True)
def _jp_core(xl, xr, early, loop, eg, lg, D, modd, modf, sr, gl, gm, gh, al, ah):
    n = xl.shape[0]
    yl = np.zeros(n)
    yr = np.zeros(n)
    ne, nl = early.shape[0], loop.shape[0]
    size = 4096
    bE = np.zeros((ne, 2, size))
    bL = np.zeros((nl, 2, size))
    dlen = int(D + modd) + 4
    dL = np.zeros((2, dlen))
    fl = 0.0
    fr = 0.0
    sl = np.zeros(2)
    sh = np.zeros(2)
    for i in range(n):
        a, b = xl[i], xr[i]
        for k in range(ne):
            a, b = _rot_ap(a, b, math.pi / 4, eg, early[k, 0], early[k, 1], bE[k, 0], bE[k, 1], i)
        a += fl
        b += fr
        m = modd * math.sin(2.0 * math.pi * modf * i / sr)
        for k in range(nl):
            ml = int(m) if k % 2 == 0 else -int(m)
            a, b = _rot_ap(a, b, math.pi / 4, lg, loop[k, 0] + ml, loop[k, 1] - ml, bL[k, 0], bL[k, 1], i)
        w = i % dlen
        dL[0, w] = a
        dL[1, w] = b
        out = np.zeros(2)
        for c in range(2):
            v = dL[c, (w - int(D)) % dlen]
            sl[c] += al * (v - sl[c])
            sh[c] += ah * (v - sh[c])
            low = sl[c]
            high = v - sh[c]
            out[c] = gl * low + gm * (v - low - high) + gh * high
        fl, fr = out[0], out[1]
        yl[i] = a
        yr[i] = b
    return yl, yr


@effect("jpverb", "JPverb (Julian Parker): rotating stereo diffusers into a modulated loop with separate decay for lows, mids and highs. Smooth, wide, lush. Stereo output.",
        t60=(1.0, "decay time of the mids, seconds (0.1..60)"), size=(1.0, "room size 0.5..5 (loop length)"),
        early_diff=(0.707, "early diffusion 0..0.99"), mod_depth=(0.1, "delay modulation 0..1"), mod_freq=(2.0, "modulation rate Hz"),
        low=(1.0, "decay multiplier below low_cut"), mid=(1.0, "decay multiplier of the mids"), high=(1.0, "decay multiplier above high_cut"),
        low_cut=(500.0, "Hz"), high_cut=(2000.0, "Hz"), damp=(0.0, "extra high damping 0..1"), mix=(0.3, "wet 0..1"))
def jpverb(x, sr=DEFAULT_SR, t60=1.0, size=1.0, early_diff=0.707, mod_depth=0.1, mod_freq=2.0, low=1.0, mid=1.0,
           high=1.0, low_cut=500.0, high_cut=2000.0, damp=0.0, mix=0.3):
    """Four input diffusers (rotation pi/4 + allpass pair), then a loop of five modulated diffusers and a delay;
    loop time total = 1.7 * 0.1 * (5/4 size - 1/4) s, feedback per band 10^(-3 total / (t60 x multiplier))."""
    x = np.asarray(x, float)
    t60 = float(np.clip(t60, 0.1, 60.0))
    sz = float(np.clip(size, 0.5, 5.0))
    total = 1.7 * 0.1 * (sz * 5.0 / 4.0 - 0.25)
    sc = sz * sr / 44100.0
    early = np.array([[_nearest_prime(int(a * sc)), _nearest_prime(int(b * sc))] for a, b in _JP_EARLY], np.int64)
    loop = np.array([[_nearest_prime(int(a * sc)), _nearest_prime(int(b * sc))] for a, b in _JP_LOOP], np.int64)
    D = max(1.0, total * sr - float(loop.sum()) / 2.0)
    g = lambda mlt: 10.0 ** (-3.0 * total / (t60 * max(float(mlt), 0.01)))
    hd = 1.0 - 0.5 * float(np.clip(damp, 0.0, 1.0))
    al = 1.0 - math.exp(-2 * math.pi * float(low_cut) / sr)
    ah = 1.0 - math.exp(-2 * math.pi * float(high_cut) / sr)
    tail = min(1.2 * t60 + 0.1, 10.0)
    xp = _pad(x, samples(tail, sr))
    xl, xr = _lr(xp)
    wl, wr = _jp_core(np.ascontiguousarray(xl), np.ascontiguousarray(xr), early, loop, float(np.clip(early_diff, 0, 0.99)),
                      0.5, D, 50.0 * float(np.clip(mod_depth, 0, 1)), float(mod_freq), float(sr),
                      g(low), g(mid), g(high) * hd, al, ah)
    wet = np.stack([wl, wr], axis=1)
    wet /= (np.max(np.abs(wet)) / (np.max(np.abs(xp)) + 1e-12) + 1e-12)
    return _fade_end(_wetdry(xp, wet, mix), sr)


@numba.njit(cache=True)
def _grey_core(xl, xr, dly, fb, lp, diff, lens, modd, modf, sr):
    n = xl.shape[0]
    yl = np.zeros(n)
    yr = np.zeros(n)
    nst = lens.shape[0]
    size = 8192
    bufs = np.zeros((nst, 2, size))
    dlen = int(dly + modd) + 4
    dl = np.zeros((2, dlen))
    s = np.zeros(2)
    for i in range(n):
        m = modd * (1.0 + math.sin(2.0 * math.pi * modf * i / sr))
        w = i % dlen
        rd = (w - int(dly + m)) % dlen
        a = xl[i] + fb * dl[0, rd]
        b = xr[i] + fb * dl[1, rd]
        s[0] += lp * (a - s[0])
        s[1] += lp * (b - s[1])
        a, b = s[0], s[1]
        for k in range(nst):
            g = diff if k % 2 == 0 else -diff
            a, b = _rot_ap(a, b, math.pi / 2, g, lens[k, 0], lens[k, 1], bufs[k, 0], bufs[k, 1], i)
        dl[0, w] = a
        dl[1, w] = b
        yl[i] = a
        yr[i] = b
    return yl, yr


@effect("greyhole", "Greyhole: a long modulated delay fed back through nested rotating diffusers; from a smeared echo to a dark, endless cloud. Stereo output.",
        delay=(0.2, "seconds 0.001..1.45"), feedback=(0.9, "0..0.99"), damping=(0.0, "0..0.99 (darker repeats)"),
        diffusion=(0.5, "0..0.99"), size=(1.0, "0.5..3"), mod_depth=(0.1, "0..1"), mod_freq=(2.0, "Hz"),
        mix=(0.4, "wet 0..1"), tail=(None, "seconds of tail appended (auto from the feedback)"))
def greyhole(x, sr=DEFAULT_SR, delay=0.2, feedback=0.9, damping=0.0, diffusion=0.5, size=1.0, mod_depth=0.1,
             mod_freq=2.0, mix=0.4, tail=None):
    """Delay up to 65533 samples -> one-pole damping -> 3 nested diffusers of 4 stages each (rotation pi/2,
    coefficient +-diffusion, lengths (10 + 19 i) x size, made prime) -> feedback."""
    x = np.asarray(x, float)
    fb = float(np.clip(feedback, 0.0, 0.99))
    dly = float(np.clip(delay, 0.001, 65533.0 / sr) * sr)
    sz = float(np.clip(size, 0.5, 3.0)) * sr / 44100.0
    lens = np.array([[_nearest_prime(max(2, int((10 + 19 * i) * sz))), _nearest_prime(max(2, int((10 + 19 * i) * sz * 1.1)))]
                     for i in range(3 * _GREY_N)], np.int64)
    if tail is None:
        tail = min(-3.0 * dly / sr / math.log10(max(fb, 1e-3)) + 0.2, 10.0)
    xp = _pad(x, samples(tail, sr))
    xl, xr = _lr(xp)
    lp = 1.0 - float(np.clip(damping, 0.0, 0.99))
    wl, wr = _grey_core(np.ascontiguousarray(xl), np.ascontiguousarray(xr), dly, fb, lp,
                        float(np.clip(diffusion, 0.0, 0.99)), lens, 50.0 * float(np.clip(mod_depth, 0, 1)),
                        float(mod_freq), float(sr))
    return _fade_end(_wetdry(xp, np.stack([wl, wr], axis=1), mix), sr)


# ==========================================================================================
# Acoustic feedback (Larsen) through a room
# ==========================================================================================

_HILBERT_A = (0.6923878, 0.9360654322959, 0.9882295226860, 0.9987488452737)
_HILBERT_B = (0.4021921162426, 0.8561710882420, 0.9722909545651, 0.9952884791278)


@numba.njit(cache=True)
def _larsen_core(x, td, tg, amp, shift, sr, pipe_d, pipe_g, ha, hb, lim):
    n = x.shape[0]
    y = np.zeros(n)
    nb = int(np.max(td)) + 2
    buf = np.zeros(nb)
    pd = max(pipe_d, 1)
    pbuf = np.zeros(pd)
    sa = np.zeros((4, 2, 2))
    sb = np.zeros((4, 2, 2))
    prev_b = 0.0
    for i in range(n):
        m = x[i]
        for k in range(td.shape[0]):
            m += tg[k] * buf[(i - td[k]) % nb]
        if shift != 0.0:
            ra = m
            rb = m
            for k in range(4):
                a2 = ha[k] * ha[k]
                o = a2 * (ra + sa[k, 1, 0]) - sa[k, 0, 1]
                sa[k, 0, 1] = sa[k, 0, 0]
                sa[k, 0, 0] = ra
                sa[k, 1, 0] = sa[k, 1, 1]
                sa[k, 1, 1] = o
                ra = o
                b2 = hb[k] * hb[k]
                o = b2 * (rb + sb[k, 1, 0]) - sb[k, 0, 1]
                sb[k, 0, 1] = sb[k, 0, 0]
                sb[k, 0, 0] = rb
                sb[k, 1, 0] = sb[k, 1, 1]
                sb[k, 1, 1] = o
                rb = o
            ph = 2.0 * math.pi * shift * i / sr
            m = ra * math.cos(ph) - prev_b * math.sin(ph)
            prev_b = rb
        if pipe_g > 0.0:
            v = pbuf[i % pd]
            m = m + pipe_g * v
            pbuf[i % pd] = m
        o = lim * math.tanh(amp[i] * m / lim)
        buf[i % nb] = o
        y[i] = o
    return y


@effect("room_feedback", "Acoustic feedback (Larsen): a microphone hears the loudspeaker through a room; near the howl point the room's own resonances ring and take over. A slow swell of the loop gain makes pitches rise and die (David Tudor / Earls style); an optional pipe picks the partial; a small frequency shift fights the howl.",
        gain=(0.9, "loop gain relative to the howl point (1 = it starts to howl; 0.5..1.5, a limiter holds it)"),
        swell=(0.0, "depth 0..1 of a slow sinusoidal swell of the loop gain"), swell_rate=(0.5, "Hz 0.2..2"),
        pipe=(None, "Hz: an open pipe in the loop picks that partial (30..400)"),
        shift=(0.0, "Hz of frequency shift in the loop (about 4 / T60 lets the gain rise a few dB before howling)"),
        mix=(1.0, "wet 0..1"), **_ROOM_PARAMS)
def room_feedback(x, sr=DEFAULT_SR, gain=0.9, swell=0.0, swell_rate=0.5, pipe=None, shift=0.0, mix=1.0, preset="hall",
                  lx=None, ly=None, lz=None, walls=None, floor=None, ceiling=None, distance=None, furnish=None):
    """Loop: mic = input + room(speaker) (image sources to order 4, mid band, from the speaker to the mic) ->
    optional frequency shifter (Hilbert allpass pair) -> optional pipe comb -> amplifier q(t) -> tanh limiter -> speaker.
    q is set so that `gain` = 1 is the peak of the room's open-loop magnitude response."""
    x = to_mono(np.asarray(x, float))
    info = room_info(preset, lx, ly, lz, walls, floor, ceiling, 50, furnish)
    src, lis = _room_positions(info, distance)
    delays, gains, _, d0 = image_source_taps(info["dims"], src, lis, info["surfaces"], 4, 0.6)
    td = np.concatenate([[d0 / C_AIR], delays + d0 / C_AIR])
    tg = np.concatenate([[1.0], np.mean(gains[:, 2:4], axis=1)]) if len(delays) else np.array([1.0])
    td = np.maximum(np.round(td * sr).astype(np.int64), 1)
    nf = 1 << int(np.ceil(np.log2(int(td.max()) * 4 + 16)))
    h = np.zeros(nf)
    np.add.at(h, td, tg)
    q = float(np.clip(gain, 0.0, 3.0)) / (np.max(np.abs(np.fft.rfft(h))) + 1e-12)
    t = np.arange(x.shape[0] + samples(2.0, sr)) / sr
    amp = q * (1.0 + float(np.clip(swell, 0.0, 1.0)) * np.sin(2 * math.pi * float(np.clip(swell_rate, 0.05, 5.0)) * t))
    pd, pg = (max(2, int(round(sr / float(np.clip(pipe, 30.0, 400.0))))), 0.9) if pipe else (1, 0.0)
    xp = _pad(x, samples(2.0, sr))
    y = _larsen_core(np.ascontiguousarray(xp), td, tg, np.ascontiguousarray(amp), float(shift), float(sr), pd, pg,
                     np.array(_HILBERT_A), np.array(_HILBERT_B), max(float(np.max(np.abs(x))), 1e-3))
    return _fade_end(_wetdry(xp, y, mix), sr)


# ==========================================================================================
# Reflections off structures, dichotic pitch, swinging bell, orchestra seats
# ==========================================================================================

def reflector_cutoff(b: float, a1: float, a2: float, theta: float = 0.0, c: float = C_AIR) -> float:
    """Low-frequency limit of a finite reflector of width b: f_u = 2 c / (b cos theta)^2 * a1 a2 / (a1 + a2)."""
    return 2.0 * c / (b * math.cos(theta)) ** 2 * a1 * a2 / (a1 + a2)


@effect("reflection", "One reflecting structure heard with the direct sound: flutter between two parallel walls, a picket fence or a staircase (a chirped echo, the 'singing' stairs), or a finite panel (only its highs come back).",
        kind=("flutter", "flutter | fence | stairs | panel"),
        width=(10.0, "flutter: metres between the walls (a buzz above c/2w, a rattle below 20 Hz when wider than 8.6 m)"),
        absorption=(0.05, "flutter: wall absorption 0..0.99"),
        spacing=(0.3, "fence/stairs: metres between elements (pitch c/2d: 572 Hz at 0.3 m)"),
        count=(30, "fence/stairs: number of elements"), distance=(5.0, "metres to the structure (first element, panel)"),
        size=(2.0, "panel: width in metres"), angle=(0.0, "panel: incidence in degrees"),
        level=(0.5, "reflection level relative to its geometric value 0..1"), mix=(1.0, "wet 0..1"))
def reflection(x, sr=DEFAULT_SR, kind="flutter", width=10.0, absorption=0.05, spacing=0.3, count=30, distance=5.0,
               size=2.0, angle=0.0, level=0.5, mix=1.0):
    """Source and listener together. Flutter: echoes every 2w/c, pressure x (1 - a) per round trip and 1/r.
    Fence/stairs: element k at sqrt(D^2 + (k d)^2) (stairs: D + k d), echo 2 r_k / c, gain D / r_k.
    Panel: one echo 2D/c, first-order high-pass at reflector_cutoff(size, D, D, angle)."""
    x = np.asarray(x, float)
    D = max(float(distance), 0.1)
    taps = []
    if kind == "flutter":
        w = max(float(width), 0.5)
        a = float(np.clip(absorption, 0.0, 0.99))
        k = np.arange(1, 400)
        r = 2.0 * w * k
        g = (1.0 - a) ** k * (w / (w + r))
        taps = [(rr / C_AIR, gg) for rr, gg in zip(r, g) if gg > 1e-3 and rr / C_AIR < 4.0]
    elif kind in ("fence", "stairs"):
        d = max(float(spacing), 0.02)
        k = np.arange(int(np.clip(count, 1, 400)))
        r = np.sqrt(D * D + (k * d) ** 2) if kind == "fence" else D + k * d
        taps = list(zip(2.0 * r / C_AIR, D / r))
    elif kind == "panel":
        taps = [(2.0 * D / C_AIR, 0.5)]
    else:
        raise ValueError("reflection kind must be flutter, fence, stairs or panel")
    n_add = int(max(t for t, _ in taps) * sr) + 2 if taps else 0
    h = np.zeros(n_add + 1)
    for t, gg in taps:
        p = t * sr
        i = int(p)
        h[i] += gg * (1.0 - (p - i))
        h[i + 1] += gg * (p - i)
    xp = _pad(x, n_add)
    if xp.ndim == 1:
        e = fftconvolve(xp, h)[:xp.shape[0]]
    else:
        e = np.stack([fftconvolve(xp[:, c], h)[:xp.shape[0]] for c in range(2)], axis=1)
    if kind == "panel":
        fu = float(np.clip(reflector_cutoff(max(float(size), 0.05), D, D, math.radians(float(angle))), 5.0, 0.45 * sr))
        e = sosfilt(butter(1, fu, "high", fs=sr, output="sos"), e, axis=0)
    full = xp + float(np.clip(level, 0.0, 1.0)) * e
    return _fade_end(_wetdry(xp, full, mix), sr)


@effect("huggins", "Dichotic pitch: a pitch that exists only in the head. Over headphones, noise whose phase turns once round a narrow band in one ear (Huggins pitch), or a copy delayed by T in the other ear (a faint pitch at 1/T). Stereo output.",
        freq=(600.0, "Hz of the phase transition (mode phase)"), bw=(0.16, "relative width of the transition"),
        mode=("phase", "phase | delay"), delay=(0.012, "seconds (mode delay): pitch near 1/T"))
def huggins(x, sr=DEFAULT_SR, freq=600.0, bw=0.16, mode="phase", delay=0.012):
    """phase: left = x, right = x through a second-order allpass at freq (its phase turns 2 pi across bw).
    delay: right = x delayed by T."""
    m = to_mono(np.asarray(x, float))
    if mode == "delay":
        d = samples(float(np.clip(delay, 0.0005, 0.05)), sr)
        r = _pad(m, 0, front=d)[: m.shape[0]]
    elif mode == "phase":
        w0 = 2 * math.pi * float(np.clip(freq, 50.0, 0.4 * sr)) / sr
        al = math.sin(w0) / (2.0 * max(1.0 / max(float(bw), 1e-3), 0.5))
        b = np.array([1 - al, -2 * math.cos(w0), 1 + al])
        r = lfilter(b, b[::-1], m)
    else:
        raise ValueError("huggins mode must be phase or delay")
    return np.stack([m, r], axis=1)


@effect("bell_swing", "A church bell swinging in its tower: the mouth sweeps toward you and away, so each stroke comes with a Doppler sway and a brighter, louder sound when the mouth faces you.",
        swing=(80.0, "degrees each side of rest (< 90)"), period=(2.0, "seconds per full swing"),
        radius=(0.7, "metres from the pivot to the mouth centre"), distance=(25.0, "metres to the listener"),
        height=(20.0, "metres the tower rises above the listener"))
def bell_swing(x, sr=DEFAULT_SR, swing=80.0, period=2.0, radius=0.7, distance=25.0, height=20.0):
    """Mouth position on a pendulum arc (angle = swing sin(2 pi t / period)), read through `doppler(path=...)`;
    mouth facing the listener: + up to 6 dB and an open low-pass, side on: dulled (bell directivity)."""
    x = to_mono(np.asarray(x, float))
    n = x.shape[0]
    tt = np.linspace(0.0, n / sr, 200)
    th = math.radians(float(np.clip(swing, 0.0, 89.0))) * np.sin(2 * math.pi * tt / max(float(period), 0.2))
    R = float(radius)
    path = np.stack([tt, np.zeros_like(tt), float(distance) + R * np.sin(th), float(height) - R * np.cos(th)], axis=1)
    y = doppler(x, sr, path=path, pan=False)
    y = to_mono(y) if y.ndim == 2 else y
    y = _pad(y, max(0, n - y.shape[0]))[:n]
    tc = np.arange(n) / sr
    thc = math.radians(float(np.clip(swing, 0.0, 89.0))) * np.sin(2 * math.pi * tc / max(float(period), 0.2))
    look = math.atan2(float(distance), float(height))
    face = 0.5 * (1.0 + np.cos(thc - look))
    y = y * 10.0 ** (6.0 * (face - 0.5) / 20.0)
    return _tv_one_pole(np.ascontiguousarray(y), np.exp(-2 * np.pi * np.minimum(1500.0 + 8000.0 * face, 0.45 * sr) / sr))


ORCHESTRA_SOUND_POWER_DB = 114.0


@effect("orchestra_seat", "Where a player sits in the orchestra: back desks of a string section are heard softer and duller from the hall; the pit adds a little body.",
        desk=(1, "desk number from the front 1..8"), pit=(False, "true = playing in an opera pit"))
def orchestra_seat(x, sr=DEFAULT_SR, desk=1, pit=False):
    """Back desks: -7 dB at desk 3 relative to desk 1, linear in dB per desk (screened by the players in front,
    more in the highs); pit: +2.5 dB below 500 Hz (reflections of the pit walls)."""
    x = np.asarray(x, float)
    k = float(np.clip(desk, 1, 8)) - 1.0
    y = x * 10.0 ** (-3.5 * k / 20.0)
    if k > 0:
        y = sosfilt(butter(1, min(max(12000.0 / (1.0 + k), 2000.0), 0.45 * sr), fs=sr, output="sos"), y, axis=0)
    if pit:
        lo = sosfilt(butter(2, 500.0, fs=sr, output="sos"), y, axis=0)
        y = y + (10.0 ** (2.5 / 20.0) - 1.0) * lo
    return y


# ==========================================================================================
# Phase-vocoder reverb (tonal / noise split) and soundboard-like FDN
# ==========================================================================================

@effect("pv_reverb", "Spectral reverb: every frequency bin of the sound decays at its own rate, steady partials ringing longer than noise (a frozen, singing tail). Stereo output.",
        t60=(3.0, "decay of the tonal part, seconds"), noise_t60=(1.0, "decay of the noisy part, seconds"),
        mix=(0.4, "wet 0..1"), seed=(0, "phase randomisation"))
def pv_reverb(x, sr=DEFAULT_SR, t60=3.0, noise_t60=1.0, mix=0.4, seed=0):
    """STFT 2048 / hop 512. Tonal vs noise per bin by median filtering across time (harmonic) and frequency
    (noise); each bin's magnitude is a leaky maximum decaying at its own T60, phases advance at the bin frequency
    plus a random walk (decorrelated left and right)."""
    from scipy.ndimage import median_filter
    from scipy.signal import istft, stft
    x = np.asarray(x, float)
    m = to_mono(x)
    t60 = float(np.clip(t60, 0.1, 30.0))
    tail = min(1.2 * max(t60, noise_t60) + 0.1, 10.0)
    xp = _pad(m, samples(tail, sr))
    hop, nper = 512, 2048
    f, t, Z = stft(xp, sr, nperseg=nper, noverlap=nper - hop)
    A = np.abs(Z)
    H = median_filter(A, size=(1, 17))
    P = median_filter(A, size=(17, 1))
    ton = H ** 2 / (H ** 2 + P ** 2 + 1e-20)
    dec_t = 10.0 ** (-3.0 * hop / sr / t60)
    dec_n = 10.0 ** (-3.0 * hop / sr / max(float(noise_t60), 0.05))
    dec = ton * dec_t + (1.0 - ton) * dec_n
    rng = np.random.default_rng(seed)
    out = []
    for ch in range(2):
        mag = np.zeros(A.shape[0])
        ph = rng.random(A.shape[0]) * 2 * np.pi
        Y = np.zeros_like(Z)
        for j in range(A.shape[1]):
            mag = np.maximum(mag * dec[:, j], A[:, j])
            ph = ph + 2 * np.pi * f * hop / sr + 0.3 * rng.standard_normal(A.shape[0])
            Y[:, j] = mag * np.exp(1j * ph)
        out.append(istft(Y, sr, nperseg=nper, noverlap=nper - hop)[1][: xp.shape[0]])
    wet = np.stack(out, axis=1)
    wet *= math.sqrt(float(np.sum(xp ** 2)) / (float(np.sum(wet ** 2)) / 2.0 + 1e-20)) * 0.5
    return _fade_end(_wetdry(_pad(x, xp.shape[0] - x.shape[0]), wet, mix), sr)


def soundboard_fdn(x, sr=DEFAULT_SR, t60_dc=3.84, t60_nyq=0.78, lines=8, size=1.5, seed=0):
    """Soundboard as a reverberator: a Householder FDN whose one-pole loop filters give T60 t60_dc at 0 Hz and
    t60_nyq at Nyquist (a wooden plate rings long in the bass, short in the treble); mono in, mono out."""
    x = to_mono(np.asarray(x, float))
    M = fdn_lengths(sr, size, lines)
    n = x.shape[0]
    A = householder_matrix(lines)
    gdc = 10.0 ** (-3.0 * M / (t60_dc * sr))
    gny = 10.0 ** (-3.0 * M / (t60_nyq * sr))
    p = (gdc - gny) / (gdc + gny)
    gk = gdc * (1.0 - p)
    bl = np.ones(lines) / math.sqrt(lines)
    cl = np.ones(lines) / math.sqrt(lines)
    yl, _, _ = _fdn_core(np.ascontiguousarray(x), np.zeros(n), M.astype(np.int64), A, gk, p, bl, np.zeros(lines), cl,
                         np.zeros(lines), 0)
    return yl


# ==========================================================================================
# Room engines: Moorer, scattering delay network, ray tracing
# ==========================================================================================

MOORER_COMBS_MS = (50.0, 56.0, 61.0, 68.0, 72.0, 78.0)
MOORER_AP_MS = 6.0


def _onepole_for(g_lo, g_hi, f_lo, f_hi, sr):
    """g, p of H = g (1 - p) / (1 - p z^-1) with |H(f_lo)| ~ g_lo (dc) and |H(f_hi)| = g_hi."""
    q = np.clip((np.asarray(g_hi, float) / np.asarray(g_lo, float)) ** 2, 1e-6, 0.999999)
    c = math.cos(2 * math.pi * min(f_hi, 0.45 * sr) / sr)
    p = ((1 - q * c) - np.sqrt(np.maximum((1 - q * c) ** 2 - (1 - q) ** 2, 0.0))) / (1 - q)
    return np.asarray(g_lo, float), p


@numba.njit(cache=True)
def _moorer_core(x, D, g, p):
    n = x.shape[0]
    y = np.zeros(n)
    for k in range(D.shape[0]):
        buf = np.zeros(D[k])
        s = 0.0
        ptr = 0
        for i in range(n):
            v = buf[ptr]
            s = (1.0 - p[k]) * v + p[k] * s
            w = x[i] + g[k] * s
            buf[ptr] = w
            y[i] += v
            ptr = (ptr + 1) % D[k]
    return y


def moorer_late(x, sr, t60_low, t60_high):
    """Moorer's reverberator: six parallel combs (50-78 ms) each with a one-pole low-pass in its loop, set so the
    loop gives t60_low at low frequency and t60_high at 4 kHz, into an allpass of 6 ms, g 0.7 (two decorrelated
    allpasses for left and right)."""
    D = np.array([int(round(m * sr / 1000.0)) for m in MOORER_COMBS_MS], np.int64)
    glo = 10.0 ** (-3.0 * D / (sr * float(t60_low)))
    ghi = 10.0 ** (-3.0 * D / (sr * float(min(t60_high, t60_low))))
    g, p = _onepole_for(glo, ghi, 0.0, 4000.0, sr)
    y = _moorer_core(np.ascontiguousarray(x, dtype=np.float64), D, g, p) / math.sqrt(len(D))
    out = []
    for ms in (MOORER_AP_MS, MOORER_AP_MS * 1.13):
        out.append(_ap_series(y, np.array([int(round(ms * sr / 1000.0))], np.int64), np.array([0.7])))
    return out[0], out[1]


@numba.njit(cache=True)
def _sdn_core(x, d_sn, g_sn, d_nn, d_nm, g_nm, wg, wp, sx):
    """Scattering delay network: 6 wall nodes, lines node->node (6x6), source->node, node->mic.
    Node: p = (2/K) sum(incoming + source/2), K = 5; outgoing to l = wall filter(p - incoming from l)."""
    n = x.shape[0]
    K = 5
    mx = 1
    for a in range(6):
        for b in range(6):
            mx = max(mx, d_nn[a, b] + 1)
    nn = np.zeros((6, 6, mx))
    mm = 1
    for a in range(6):
        mm = max(mm, d_nm[a] + 1)
    nm = np.zeros((6, mm))
    fs_ = np.zeros((6, 6))
    fm = np.zeros(6)
    yl = np.zeros(n)
    yr = np.zeros(n)
    pin = np.zeros(6)
    for i in range(n):
        for k in range(6):
            src = g_sn[k] * x[i - d_sn[k]] if i >= d_sn[k] else 0.0
            tot = 0.0
            for l in range(6):
                if l != k:
                    pin[l] = nn[l, k, (i - d_nn[l, k]) % mx] + 0.5 * src
                    tot += pin[l]
            pk = 2.0 / K * tot
            for l in range(6):
                if l != k:
                    o = pk - pin[l]
                    fs_[k, l] = wg[k] * (1.0 - wp[k]) * o + wp[k] * fs_[k, l]
                    nn[k, l, i % mx] = fs_[k, l]
            fm[k] = wg[k] * (1.0 - wp[k]) * pk + wp[k] * fm[k]
            nm[k, i % mm] = fm[k]
            v = g_nm[k] * nm[k, (i - d_nm[k]) % mm]
            yl[i] += v * (1.0 - sx[k])
            yr[i] += v * (1.0 + sx[k])
    return yl, yr


def sdn_late(x, sr, info, distance=None):
    """Scattering delay network of the box room: one node per wall at the first-order reflection point, lines
    between nodes from their distances, source->node gain 1/|xs - xk|, node->mic 1/(1 + |xk - xm|/|xs - xk|),
    wall reflectance sqrt(1 - alpha) at 250 Hz and 4 kHz through a one-pole. Output relative to the direct 1/d0
    (direct sound not included)."""
    src, lis = _room_positions(info, distance)
    L = np.asarray(info["dims"], float)
    nodes = []
    for ax in range(3):
        for wall in (0.0, L[ax]):
            ds, dl = abs(src[ax] - wall), abs(lis[ax] - wall)
            t = ds / max(ds + dl, 1e-9)
            pnt = src + t * (lis - src)
            pnt[ax] = wall
            nodes.append(pnt)
    nodes = np.array(nodes)
    d0 = max(float(np.linalg.norm(src - lis)), 0.05)
    dsn = np.linalg.norm(nodes - src, axis=1)
    dnm = np.linalg.norm(nodes - lis, axis=1)
    dnn = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=2)
    alpha = np.array([[_space.ABSORPTION[m][1], _space.ABSORPTION[m][5]] for m in info["surfaces"]])
    beta = np.sqrt(1.0 - np.clip(alpha, 0.0, 0.9999))
    wg, wp = _onepole_for(beta[:, 0], np.minimum(beta[:, 1], beta[:, 0]), 250.0, 4000.0, sr)
    smp = lambda d: np.maximum(np.round(d / C_AIR * sr).astype(np.int64), 1)
    yl, yr = _sdn_core(np.ascontiguousarray(x, dtype=np.float64), smp(dsn), d0 / dsn, smp(dnn), smp(dnm),
                       1.0 / (1.0 + dnm / dsn), wg, wp, (nodes[:, 0] - lis[0]) / np.maximum(dnm, 1e-9))
    return yl, yr


def raytrace_hist(info, distance=None, rays=20000, scattering=0.1, bin_s=0.005, tmax=None, seed=0):
    """Energy histogram (bins x 6 bands) relative to the direct sound, from `space.ray_trace`."""
    src, lis = _room_positions(info, distance)
    alpha = np.array([_space.ABSORPTION[m] for m in info["surfaces"]], float)
    m_air = _space.air_db_per_m(np.asarray(BANDS, float), 50.0) * math.log(10.0) / 10.0
    if tmax is None:
        tmax = float(min(1.2 * np.max(info["t60"]), 4.0))
    s = np.broadcast_to(np.asarray(scattering, float), (6,)).astype(float)
    return _space.ray_trace(np.asarray(info["dims"], float), src, lis, alpha, s, m_air, int(rays), float(tmax),
                            float(bin_s), int(seed))


def greedy_filter_assign(tau, r, p, eps=1.0):
    """Greedy assignment of one of Q filters to each of M taps or lines: phi(m) = argmax_q (tau_q(m) + eps r_q(m)) p_q(m),
    the energies of each filter normalised to 1 first. tau, r, p: arrays [Q, M]. Returns phi, length M."""
    tau, r, p = (np.asarray(a, float) for a in (tau, r, p))
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-30)
    return np.argmax((tau + float(eps) * r) * p, axis=0)


def scattering_delays(n: int = 4, stages: int = 2, delta: float | None = None) -> list:
    """Delay patterns of a filtered feedback matrix A(z) = D_mK U_K ... U_1 D_m0 built from N x N orthogonal stages:
    m_k = L_(k-1) m_1 with L_k = N^k (dense: every path a different delay), or L_k = ceil(N^k / delta) for a velvet
    matrix. Returns the K + 1 integer multipliers L_0..L_K."""
    return [int(math.ceil(n ** k / delta)) if delta else n ** k for k in range(stages + 1)]


def room_params(volume: float, t60: float, surface: float | None = None) -> dict:
    """Perceptual room settings from its volume (m^3) and decay: mean free path 4V/S (S of a cube of that volume if not
    given) as the FDN `size`, first-reflection delay = that path / c, and the decay itself."""
    v = max(float(volume), 1.0)
    s = float(surface) if surface else 6.0 * v ** (2.0 / 3.0)
    mfp = 4.0 * v / s
    return {"size": mfp, "predelay": mfp / C_AIR, "t60": float(t60)}
