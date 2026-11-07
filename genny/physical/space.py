# Adapted for Genny v0.3 (2026): reorganized under genny.physical and integrated with
# Genny high-level APIs. Original procedural/Klang code © 2025 Chris Nash.
# Licensed under Klang Open License 1.0; see KLANG_LICENSE.txt.
"""Space: reverberators, per-band T60 from room absorption, distance and air absorption.

Ports
-----
* **zita_rev1** - Fons Adriaensen's 8x8 FDN as ported to Faust by J.O. Smith
  (``faustlibraries/reverbs.lib`` ``zita_rev_fdn``, ``zita_in_delay``, ``zita_distrib2``,
  ``zita_rev1_stereo``; research 08 §7.1, 10 §2.4). Exact delays, allpass coefficients +-0.6,
  per-line low shelf (T60 below f1 = t60dc) and "special lowpass" (T60 at f2 = t60m / 2),
  unnormalised Hadamard with 1/sqrt(8) in the line filters, input x 0.3 after ``rdel`` ms,
  distribution L, R, -L, -R, ..., output 0.37 (y1 + y2), 0.37 (y1 - y2). Wet only.
* **Puckette G08** - ``reverb-echo.pd`` / ``G08.reverb.pd`` (TTEM p.193-196; research 09 B5,
  10 §2.3): 6 rotate-and-delay early stages 5.43216 ... 55.5437 ms, 4 recirculating lines
  60, 71.9345, 86.7545, 95.945 ms, two butterfly stages, gain fb/200 (0.5 = infinite), taps
  at n1 / n2. Optional one-pole low-pass per line (Puckette's own named extension).
* **Farnell tight prime-delay room** - MEX acid-composition ``pd reverb`` (research 09 C7.6,
  D5.1): lop~ 400 Hz input, stage 1 taps 13/17/19/23 ms x 0.24 fed back, stage 2 Hadamard
  FDN 31/37/41/43 ms with g = 0.33 (text) - "a reverb time of about half a second ... a fairly
  tight room type effect"; outputs 0.4 A / 0.4 B.
* **Per-band T60** - Sabine T60 = 0.161 V / (S a) with Farnell Table 5.1 absorption
  coefficients (research 01 §6.1); mapped onto zita's (t60dc, t60m, f2).
* **Distance** - spherical spreading 1/r (-6 dB per doubling) or line source 1/sqrt(r)
  (-3 dB per doubling) (Farnell rule 19, research 01 §5); ISO 9613-1 air absorption
  (temperature, humidity, pressure); propagation delay r / c(T), c = 343.2 m/s at 20 C.
"""

from __future__ import annotations

import math

import numba
import numpy as np

from .core import SR, DEV_DIR, bandpass, envelope_db, seconds, write_wav

# ==========================================================================================
# zita_rev1
# ==========================================================================================

ZITA_AP_DELAYS = (0.020346, 0.024421, 0.031604, 0.027333, 0.022904, 0.029291, 0.013458, 0.019123)
ZITA_T_DELAYS = (0.153129, 0.210389, 0.127837, 0.256891, 0.174713, 0.192303, 0.125000, 0.219991)


@numba.njit(cache=True)
def _hadamard8(v, y):
    # ro.hadamard(8) = butterfly(8) : (hadamard(4), hadamard(4)); butterfly(u) = [a + b, a - b]
    a0 = v[0] + v[4]; a1 = v[1] + v[5]; a2 = v[2] + v[6]; a3 = v[3] + v[7]
    b0 = v[0] - v[4]; b1 = v[1] - v[5]; b2 = v[2] - v[6]; b3 = v[3] - v[7]
    c0 = a0 + a2; c1 = a1 + a3; c2 = a0 - a2; c3 = a1 - a3
    d0 = b0 + b2; d1 = b1 + b3; d2 = b0 - b2; d3 = b1 - b3
    y[0] = c0 + c1; y[1] = c0 - c1; y[2] = c2 + c3; y[3] = c2 - c3
    y[4] = d0 + d1; y[5] = d0 - d1; y[6] = d2 + d3; y[7] = d2 - d3


@numba.njit(cache=True)
def _zita_core(inl, inr, apd, fbd, apc, g0, gm, pole, lb, la1):
    n = inl.shape[0]
    N = 8
    ap_max = 0
    fb_max = 0
    for i in range(N):
        ap_max = max(ap_max, apd[i])
        fb_max = max(fb_max, fbd[i] + 1)
    apbuf = np.zeros((N, ap_max))
    fbbuf = np.zeros((N, fb_max))
    api = np.zeros(N, np.int64)
    fbi = np.zeros(N, np.int64)
    x1 = np.zeros(N)
    y1 = np.zeros(N)
    sm = np.zeros(N)
    fb = np.zeros(N)
    v = np.zeros(N)
    y = np.zeros(N)
    outl = np.zeros(n)
    outr = np.zeros(n)
    inv_sqrt8 = 1.0 / np.sqrt(8.0)
    for k in range(n):
        L = inl[k]
        R = inr[k]
        for i in range(N):
            s = L if (i % 2 == 0) else R          # zita_distrib2: L, R, -L, -R, L, R, -L, -R
            if (i // 2) % 2 == 1:
                s = -s
            wd = apbuf[i, api[i]]                 # w[n - N]
            w = s + fb[i] - apc[i] * wd           # allpass_comb: w = x - a w[n-N]
            v[i] = wd + apc[i] * w                # y = w[n-N] + a w
            apbuf[i, api[i]] = w
            api[i] += 1
            if api[i] >= apd[i]:
                api[i] = 0
        _hadamard8(v, y)
        outl[k] = 0.37 * (y[1] + y[2])
        outr[k] = 0.37 * (y[1] - y[2])
        for i in range(N):
            lp = lb * y[i] + lb * x1[i] - la1 * y1[i]          # fi.lowpass(1, f1)
            x1[i] = y[i]
            y1[i] = lp
            s = gm[i] * (y[i] + (g0[i] / gm[i] - 1.0) * lp)    # gM * low_shelf1_l(g0/gM, f1)
            sm[i] = (1.0 - pole[i]) * s + pole[i] * sm[i]      # special_lowpass(gM, f2)
            L2 = fbd[i] + 1                                     # fbdelay + the "~" sample
            fb[i] = fbbuf[i, fbi[i]]
            fbbuf[i, fbi[i]] = sm[i] * inv_sqrt8 + 1e-20       # /sqrt(N) + staynormal
            fbi[i] += 1
            if fbi[i] >= L2:
                fbi[i] = 0
    return outl, outr


def zita_rev1(x: np.ndarray, rdel: float = 60.0, f1: float = 200.0, f2: float = 6000.0,
              t60dc: float = 3.0, t60m: float = 2.0) -> np.ndarray:
    """zita_rev1_stereo (wet only). ``x`` mono or (n, 2). Returns (n, 2).

    Defaults are ``dm.zita_rev1`` GUI defaults (rdel 60 ms, f1 200 Hz, t60dc 3 s, t60m 2 s,
    f2 6000 Hz). ``instrReverb``: rdel 20, t60dc = 0.72*3, t60m = 0.72*2, wet 0.137.
    """
    x = np.asarray(x, float)
    xl, xr = (x, x) if x.ndim == 1 else (x[:, 0], x[:, 1])
    N = 8
    tdel = np.array([int(np.floor(0.5 + SR * t)) for t in ZITA_T_DELAYS])
    apd = np.array([int(np.floor(0.5 + SR * t)) for t in ZITA_AP_DELAYS])
    fbd = tdel - apd
    apc = np.array([0.6 if i % 2 == 0 else -0.6 for i in range(N)])
    g0 = np.exp(-3 * np.log(10) * tdel / (t60dc * SR))
    gm = np.exp(-3 * np.log(10) * tdel / (t60m * SR))
    cw = np.cos(2 * np.pi * f2 / SR)
    gs = gm * gm
    mbo2 = (1 - gs * cw) / (1 - gs)
    pole = mbo2 - np.sqrt(np.maximum(0.0, mbo2 * mbo2 - 1))
    c = 1 / np.tan(np.pi * f1 / SR)
    lb = 1 / (1 + c)
    la1 = (1 - c) / (1 + c)
    pre = int(SR * rdel * 0.001)
    n = len(xl)
    inl = np.concatenate([np.zeros(pre), xl])[:n] * 0.3       # zita_in_delay: delay rdel ms, x 0.3
    inr = np.concatenate([np.zeros(pre), xr])[:n] * 0.3
    ol, or_ = _zita_core(np.ascontiguousarray(inl), np.ascontiguousarray(inr), apd.astype(np.int64),
                         fbd.astype(np.int64), apc, g0, gm, pole, lb, la1)
    return np.stack([ol, or_], axis=1)


# Presets: dm.zita_rev1 GUI defaults (research 08 §7.1); the others are the research note's
# suggested spaces (inference inside the GUI ranges, marked there as such).
ZITA_PRESETS = {
    "default": dict(rdel=60.0, f1=200.0, f2=6000.0, t60dc=3.0, t60m=2.0),
    "instrument": dict(rdel=20.0, f1=200.0, f2=6000.0, t60dc=0.72 * 3, t60m=0.72 * 2),
    "exterior_dry": dict(rdel=20.0, f1=200.0, f2=3000.0, t60dc=1.2, t60m=0.8),
    "stone_hall": dict(rdel=40.0, f1=200.0, f2=4000.0, t60dc=3.0, t60m=2.2),
    "cave": dict(rdel=70.0, f1=200.0, f2=2500.0, t60dc=5.0, t60m=3.5),
}


# ==========================================================================================
# Puckette G08
# ==========================================================================================

G08_ECHO_MS = (5.43216, 8.45346, 13.4367, 21.5463, 34.3876, 55.5437)
G08_LINES_MS = (60.0, 71.9345, 86.7545, 95.945)


@numba.njit(cache=True)
def _g08_core(a, b, L, g, lp, hf):
    n = a.shape[0]
    bufs = np.zeros((4, L.max()))
    idx = np.zeros(4, np.int64)
    s = np.zeros(4)
    d = np.zeros(4)
    w = np.zeros(4)
    yl = np.zeros(n)
    yr = np.zeros(n)
    for i in range(n):
        for k in range(4):
            d[k] = bufs[k, idx[k]]
            if lp > 0.0:                          # one-pole low-pass at the end of each line
                s[k] = (1.0 - lp) * d[k] + lp * s[k]
                d[k] = s[k] + hf * (d[k] - s[k])
        n1 = a[i] + d[0]
        n2 = b[i] + d[1]
        n6 = d[2] + d[3]
        n8 = d[2] - d[3]
        n5 = n1 + n2
        n7 = n1 - n2
        w[0] = g[i] * (n5 + n6)
        w[1] = g[i] * (n7 + n8)
        w[2] = g[i] * (n5 - n6)
        w[3] = g[i] * (n7 - n8)
        for k in range(4):
            bufs[k, idx[k]] = w[k]
            idx[k] += 1
            if idx[k] >= L[k]:
                idx[k] = 0
        yl[i] = n1
        yr[i] = n2
    return yl, yr


def puckette_rev(x: np.ndarray, feedback: float = 90.0, t60: float | None = None,
                 damp_hz: float | None = None, kind: str = "g08", damping: float = 0.0,
                 capture: float | None = None) -> np.ndarray:
    """Puckette G08 reverberator (mono in, stereo (n, 2) out, early + late, no dry).

    ``feedback`` 0-100 -> g = feedback/200 (100 = infinite). If ``t60`` is given, g is set from
    RT60 = -3 d / log10(2 g) (TTEM; d = mean loop length). ``damp_hz`` adds the one-pole
    low-pass per line. Early stages are scaled by sqrt(1/2) each (Puckette's deferred
    normalisation) so the early block has unit power gain.
    ``kind="rev2"``: rev2~ behaviour, ``damp_hz`` is a crossover above which ``damping`` (0..1) of the
    signal is removed on each pass (0 = all of it kept). ``capture`` seconds: the reverb as a source,
    the input is let in for that time, then the loop is frozen (g = 1/2, lossless) and rings on.
    """
    x = np.asarray(x, float)
    if x.ndim == 2:
        x = x.mean(axis=1)
    a = x.copy()
    b = np.zeros_like(x)
    for dms in G08_ECHO_MS:
        D = int(round(dms * SR / 1000))
        s, df = a + b, a - b
        dl = np.zeros_like(df)
        dl[D:] = df[:-D]
        a, b = s * np.sqrt(0.5), dl * np.sqrt(0.5)
    L = np.array([int(round(m * SR / 1000)) for m in G08_LINES_MS], dtype=np.int64)
    if t60 is not None:
        dmean = L.mean() / SR
        g = 0.5 * 10 ** (-3 * dmean / t60)
    else:
        g = np.clip(feedback, 0, 100) / 200.0
    lp = float(np.exp(-2 * np.pi * damp_hz / SR)) if damp_hz else 0.0
    hf = 1.0 - float(np.clip(damping, 0.0, 1.0)) if kind == "rev2" else 0.0
    garr = np.full(a.shape[0], float(g))
    if capture is not None:
        nc = int(max(float(capture), 0.001) * SR)
        ramp = np.clip((nc - np.arange(a.shape[0])) / max(int(0.01 * SR), 1), 0.0, 1.0)
        a, b = a * ramp, b * ramp
        garr[nc:] = 0.5
    yl, yr = _g08_core(np.ascontiguousarray(a), np.ascontiguousarray(b), L, garr, lp, hf)
    return np.stack([yl, yr], axis=1)


# ==========================================================================================
# Farnell tight prime-delay room (MEX)
# ==========================================================================================

@numba.njit(cache=True)
def _farnell_room_core(x, taps1, fb1, L2, g):
    n = x.shape[0]
    m1 = taps1.max() + 1
    buf1 = np.zeros(m1)
    i1 = 0
    bufs = np.zeros((4, L2.max()))
    idx = np.zeros(4, np.int64)
    d = np.zeros(4)
    t = np.zeros(4)
    yl = np.zeros(n)
    yr = np.zeros(n)
    for i in range(n):
        for k in range(4):
            j = i1 - taps1[k]
            if j < 0:
                j += m1
            t[k] = buf1[j]
        buf1[i1] = x[i] + fb1 * (t[0] + t[1] + t[2] + t[3])     # stage 1 closed: taps x 0.24
        i1 += 1
        if i1 >= m1:
            i1 = 0
        for k in range(4):
            d[k] = bufs[k, idx[k]]
        A = t[0] + t[1] + d[0]
        B = t[2] + t[3] + d[1]
        C = d[2] + d[3]
        D = d[2] - d[3]
        w0 = g * ((A + B) + C)
        w1 = g * ((A - B) + D)
        w2 = g * ((A + B) - C)
        w3 = g * ((A - B) - D)
        bufs[0, idx[0]] = w0
        bufs[1, idx[1]] = w1
        bufs[2, idx[2]] = w2
        bufs[3, idx[3]] = w3
        for k in range(4):
            idx[k] += 1
            if idx[k] >= L2[k]:
                idx[k] = 0
        yl[i] = 0.4 * A
        yr[i] = 0.4 * B
    return yl, yr


def farnell_room(x: np.ndarray, lowpass_hz: float = 400.0, stage1_fb: float = 0.0,
                 liveness: float = 0.33) -> np.ndarray:
    """Farnell MEX two-stage prime-delay reverb, stereo wet (n, 2).

    Input one-pole low-pass (lop~ 400); stage 1: one line tapped at 13, 17, 19, 23 ms. In the
    patch each tap x 0.24 goes to ``throw~ rvfb`` but no ``catch~ rvfb`` exists, so stage 1 is
    open (``stage1_fb=0``, as wired - this gives the text's "about half a second"); set
    ``stage1_fb=0.24`` to close it as the text describes (a much longer, boomy tail). Stage 2: lines 31, 37, 41, 43 ms, Hadamard +/- mix, g = 0.33 (the
    author's text; the figure shows 0.35, the final mix 0.154); out L = 0.4 A, R = 0.4 B.
    """
    x = np.asarray(x, float)
    if x.ndim == 2:
        x = x.mean(axis=1)
    # Pd lop~: y = y1 + k (x - y1), k = 2 pi f / SR
    k = min(2 * np.pi * lowpass_hz / SR, 1.0)
    y = np.empty_like(x)
    s = 0.0
    for i, v in enumerate(x):
        s += k * (v - s)
        y[i] = s
    taps1 = np.array([int(round(m * SR / 1000)) for m in (13, 17, 19, 23)], dtype=np.int64)
    L2 = np.array([int(round(m * SR / 1000)) for m in (31, 37, 41, 43)], dtype=np.int64)
    yl, yr = _farnell_room_core(y, taps1, float(stage1_fb), L2, float(liveness))
    return np.stack([yl, yr], axis=1)


# ==========================================================================================
# Per-band T60 (Sabine + Farnell Table 5.1)
# ==========================================================================================

ABSORPTION_BANDS = (125, 250, 500, 1000, 2000, 4000)
ABSORPTION = {  # Farnell Designing Sound Table 5.1 (research 01 §6.1), verbatim
    "carpet": (0.01, 0.02, 0.06, 0.15, 0.25, 0.45),
    "concrete": (0.01, 0.02, 0.04, 0.06, 0.08, 0.1),
    "marble": (0.01, 0.01, 0.01, 0.01, 0.02, 0.02),
    "wood": (0.15, 0.11, 0.1, 0.07, 0.06, 0.07),
    "brick": (0.03, 0.03, 0.03, 0.04, 0.05, 0.07),
    "glass": (0.18, 0.06, 0.04, 0.03, 0.02, 0.02),
    "plaster": (0.01, 0.02, 0.02, 0.03, 0.04, 0.05),
    "fabric": (0.04, 0.05, 0.11, 0.18, 0.3, 0.35),
    "metal": (0.19, 0.69, 0.99, 0.88, 0.52, 0.27),
    "people": (0.25, 0.35, 0.42, 0.46, 0.5, 0.5),
    "water": (0.008, 0.008, 0.013, 0.015, 0.02, 0.025),
    # Kuttruff, Room Acoustics 5th ed., Table 9.1
    "hard": (0.02, 0.02, 0.03, 0.03, 0.04, 0.05),
    "light_panel": (0.10, 0.07, 0.05, 0.04, 0.04, 0.05),
    "vibrating_wood": (0.40, 0.20, 0.12, 0.07, 0.05, 0.05),
    "thin_carpet": (0.02, 0.03, 0.05, 0.10, 0.30, 0.50),
    "curtain": (0.15, 0.45, 0.90, 0.92, 0.95, 0.95),
    "foam": (0.08, 0.22, 0.55, 0.70, 0.85, 0.75),
    "acoustic_plaster": (0.08, 0.15, 0.30, 0.50, 0.60, 0.70),
    # Kuttruff Table 6.3: audience on moderately upholstered chairs, 0.85 m x 0.63 m; band 6 kHz not kept
    "audience_seated": (0.72, 0.82, 0.91, 0.93, 0.94, 0.87),
    # Kuttruff Table 6.2: m^2 per person, so the "area" given with these is a head count
    "person_coat": (0.17, 0.41, 0.91, 1.30, 1.43, 1.47),
    "person": (0.12, 0.24, 0.59, 0.98, 1.13, 1.12),
    "musician": (0.60, 0.95, 1.06, 1.08, 1.08, 1.08),
    # Malbos, Bogdanski & Strauss 2015 Table 2 (Cox & D'Antonio), 125-1000 Hz. 2 and 4 kHz hold the 1 kHz value
    "car_glass": (0.35, 0.25, 0.18, 0.12, 0.12, 0.12),
    "car_floor": (0.03, 0.09, 0.20, 0.54, 0.54, 0.54),
    "car_seat": (0.45, 0.60, 0.73, 0.80, 0.80, 0.80),
    "car_trim": (0.04, 0.05, 0.11, 0.18, 0.18, 0.18),
}

# Kuttruff Table 5.1: relative variance gamma^2 of the free path by relative dimensions of a box
PATH_VARIANCE = {(1, 1, 1): 0.342, (1, 1, 2): 0.356, (1, 1, 5): 0.412, (1, 1, 10): 0.415, (1, 2, 2): 0.363,
                 (1, 2, 5): 0.403, (1, 2, 10): 0.465, (1, 5, 5): 0.464, (1, 5, 10): 0.510, (1, 10, 10): 0.613}


def path_variance(dims) -> float:
    """gamma^2 of the table entry nearest in log proportions. ponytail: nearest entry, no 3D interpolation."""
    a, b, c = sorted(float(v) for v in dims)
    key = np.log([b / a, c / a])
    best = min(PATH_VARIANCE, key=lambda s: float(np.sum((np.log([s[1], s[2]]) - key) ** 2)))
    return PATH_VARIANCE[best]


def absorption_exponent(alpha, formula: str = "sabine", areas=None, dims=None):
    """Absorption exponent a of T = 0.161 V / (S a + 4 m V), per band. ``alpha`` (surfaces, bands).
    Kuttruff: sabine (5.25) mean alpha; eyring (5.24) -ln(1 - mean); millington (5.26) mean of -ln(1 - alpha_i);
    kuttruff (5.33) -ln(1 - mean) [1 + gamma^2/2 ln(1 - mean)]. A fully open surface (alpha = 1) sends millington to
    infinity, as Kuttruff says just below (5.26): alpha is clipped to 0.999 there, so T stays small but not zero."""
    al = np.minimum(np.asarray(alpha, float), 0.999)
    w = np.asarray(areas, float)[:, None] / float(np.sum(areas))
    mean = np.sum(w * np.asarray(alpha, float), axis=0)
    log1 = np.log(1.0 - np.minimum(mean, 0.999))
    if formula == "sabine":
        return mean
    if formula == "eyring":
        return -log1
    if formula == "millington":
        return -np.sum(w * np.log(1.0 - al), axis=0)
    if formula == "kuttruff":
        return -log1 * (1.0 + 0.5 * path_variance(dims) * log1)
    raise ValueError("formula must be sabine, eyring, millington or kuttruff")


def schroeder_frequency(t60: float, volume: float) -> float:
    """fs = 2000 sqrt(T / V) Hz: above it modes overlap (Kuttruff 3.32)."""
    return 2000.0 * math.sqrt(float(t60) / float(volume))


def room_modes(lx: float, ly: float, lz: float, fmax: float, c: float = None):
    """Eigenfrequencies of a rigid box up to ``fmax``, f = c/2 sqrt((nx/Lx)^2 + (ny/Ly)^2 + (nz/Lz)^2) (Kuttruff 3.15).
    Returns (frequencies, integer orders (m, 3)), sorted, without the (0, 0, 0) term."""
    c = C_AIR if c is None else float(c)
    n = [np.arange(int(2.0 * fmax * L / c) + 1) for L in (lx, ly, lz)]
    g = np.stack(np.meshgrid(*n, indexing="ij"), axis=-1).reshape(-1, 3)[1:]
    f = 0.5 * c * np.sqrt(np.sum((g / np.array([lx, ly, lz], float)) ** 2, axis=1))
    keep = f <= fmax
    o = np.argsort(f[keep])
    return f[keep][o], g[keep][o]


def piston_gain(ka: float) -> float:
    """Directivity g of a baffled circular piston, (ka)^2/2 (1 - 2 J1(2ka)/(2ka))^-1 (Kuttruff 10.3); 1 at low ka."""
    from scipy.special import j1
    ka = max(float(ka), 1e-3)
    return 0.5 * ka * ka / (1.0 - j1(2.0 * ka) / ka)


def piston_directivity(ka, theta: float) -> np.ndarray:
    """|2 J1(u)/u|, u = ka sin(theta), far-field pressure of the piston (Kuttruff 10.1)."""
    from scipy.special import j1
    u = np.maximum(np.abs(np.asarray(ka, float) * math.sin(float(theta))), 1e-6)
    return np.abs(2.0 * j1(u) / u)


PIPE_END_KA = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.45, 3.6, 3.75, 3.82)
PIPE_END_LA = (0.6133, 0.603, 0.582, 0.556, 0.527, 0.499, 0.472, 0.445, 0.417, 0.390, 0.363, 0.337, 0.309, 0.280,
               0.251, 0.222, 0.183, 0.142)


def pipe_reflection(ka) -> np.ndarray:
    """Reflection coefficient R (e^{+iwt}) at the open end of an unflanged pipe of radius a:
    |R| = 1 + 0.01336 ka - 0.59079 (ka)^2 + 0.33576 (ka)^3 - 0.06432 (ka)^4 up to ka = 1.5, then
    sqrt(pi ka) e^(-ka) (1 + 3 / (32 (ka)^2)); end correction l/a read off ``PIPE_END_LA`` (0.6133 at ka = 0,
    falling to 0.14 at the first cross mode, ka = 3.832), R = -|R| exp(-2 i k l)."""
    ka = np.asarray(ka, float)
    kr = np.minimum(ka, 1.5)
    kh = np.maximum(ka, 1.5)
    mag = np.where(ka <= 1.5, 1.0 + kr * (0.01336 + kr * (-0.59079 + kr * (0.33576 - 0.06432 * kr))),
                   np.sqrt(np.pi * kh) * np.exp(-kh) * (1.0 + 3.0 / (32.0 * kh * kh)))
    l_a = np.interp(ka, PIPE_END_KA, PIPE_END_LA)
    return -mag * np.exp(-2j * ka * l_a)


def pipe_directivity(ka, theta: float) -> np.ndarray:
    """|D F| / |F(0)| of an unflanged open pipe: D = 2 J1(u)/u (u = ka sin theta), F = 1 + z cos theta with
    z = (1 + R) / (1 - R) the radiation impedance / rho c. Omni at low ka, |R| behind (theta = pi)."""
    r = pipe_reflection(ka)
    z = (1.0 + r) / (1.0 - r)
    th = float(theta)
    return piston_directivity(ka, th) * np.abs(1.0 + z * math.cos(th)) / np.abs(1.0 + z)


def membrane_directivity(m: int, elevation: float, azimuth: float) -> float:
    """|cos^m(elev) cos(m phi)| of a drumhead's mode with m nodal diameters: -3 dB at 45, -10 dB at 71.5 degrees for m = 1 (Meyer 4.4, chunk_01 l.7633)."""
    return abs(math.cos(float(elevation)) ** int(m) * math.cos(int(m) * float(azimuth)))


_FLAT_B = ((0.1, 3.052), (0.2, 2.425), (0.3, 2.154), (0.4, 2.002), (0.5, 1.903), (0.6, 1.840), (0.7, 1.806))  # Kuttruff Table 9.2


def flat_room_energy(r, h: float, alpha: float) -> np.ndarray:
    """Energy density x 4 pi c / P at ``r`` in a flat room of height ``h``, diffuse floor and ceiling (Kuttruff 9.13);
    source and listener half way up. alpha beyond Table 9.2 takes its end value of b."""
    rho = 1.0 - float(alpha)
    b = float(np.interp(alpha, *zip(*_FLAT_B)))
    r2 = np.asarray(r, float) ** 2
    u = r2 / float(h) ** 2
    return ((1.0 + u) ** -1.5 + b * rho / (1.0 - rho) * (b * b + u) ** -1.5) / (r2 + 4.0 * rho * float(h) ** 2)


def sabine_t60(volume: float, surfaces: dict[str, float]) -> np.ndarray:
    """Per-band T60 = 0.161 V / sum(S_i a_i) at 125 ... 4000 Hz; ``surfaces`` = {material: m^2}."""
    absorb = np.zeros(len(ABSORPTION_BANDS))
    for mat, area in surfaces.items():
        absorb += area * np.asarray(ABSORPTION[mat])
    return 0.161 * volume / absorb


def zita_from_bands(t60s: np.ndarray, rdel: float = 40.0, f1: float = 200.0) -> dict:
    """Map a per-band T60 (125-4000 Hz) onto zita_rev1: t60dc = mean(125, 250), t60m = mean(500,
    1000), f2 = the frequency where the T60 falls to t60m / 2 (log-interpolated; if it never
    does below 4 kHz, extrapolated along the 2-4 kHz slope). Clamped to zita's GUI ranges
    (T60 1-8 s is the GUI range; shorter values are allowed; f2 1500 Hz - 0.49 SR).
    """
    t = np.asarray(t60s, float)
    t60dc, t60m = float(np.mean(t[:2])), float(np.mean(t[2:4]))
    target = t60m / 2
    lf = np.log(np.asarray(ABSORPTION_BANDS, float))
    below = np.nonzero(t <= target)[0]
    if len(below) and below[0] > 0:
        j = below[0]
        w = (np.log(t[j - 1]) - np.log(target)) / (np.log(t[j - 1]) - np.log(t[j]))
        f2 = float(np.exp(lf[j - 1] + w * (lf[j] - lf[j - 1])))
    else:
        slope = (np.log(t[5]) - np.log(t[4])) / (lf[5] - lf[4])
        f2 = float(np.exp(lf[5] + (np.log(target) - np.log(t[5])) / slope)) if slope < 0 else 0.49 * SR
    return dict(rdel=rdel, f1=f1, f2=float(np.clip(f2, 1500.0, 0.49 * SR)), t60dc=t60dc, t60m=t60m)


# ==========================================================================================
# Distance: spreading, air absorption, delay
# ==========================================================================================

def speed_of_sound(temp_c: float = 20.0) -> float:
    """c = 343.2 sqrt(T / 293.15) m/s (ISO 9613-1, as in acoustic-toolbox's atmosphere module)."""
    return 343.2 * math.sqrt((float(temp_c) + 273.15) / 293.15)


C_AIR = speed_of_sound(20.0)       # 343.2 m/s


def air_db_per_m(freqs: np.ndarray, rh: float = 50.0, temp_c: float = 20.0, pressure_kpa: float = 101.325) -> np.ndarray:
    """Pure-tone air absorption in dB/m, ISO 9613-1:1993 (oxygen and nitrogen relaxation; formula as
    given by sengpielaudio "Damping of air", saturation pressure WMO-No.188). Valid 50 Hz-10 kHz,
    -20..+50 C, 10..100 % RH; used beyond 10 kHz as is. 20 C / 50 %: 4.66 dB/km at 1 kHz, 29.7 at 4 kHz.
    (The PASP Table B.2 used before read its 4 kHz column 3x too high.)"""
    T, T0, T01, pr = float(temp_c) + 273.15, 293.15, 273.16, 101.325
    pa = float(pressure_kpa)
    psat = pr * 10 ** (-6.8346 * (T01 / T) ** 1.261 + 4.6151)
    h = float(np.clip(rh, 0.0, 100.0)) * psat / pa                  # molar concentration of water vapour, %
    fro = (pa / pr) * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frn = (pa / pr) * (T / T0) ** -0.5 * (9.0 + 280.0 * h * math.exp(-4.170 * ((T / T0) ** (-1.0 / 3.0) - 1.0)))
    f = np.asarray(freqs, float)
    return 8.686 * f * f * (1.84e-11 * (pr / pa) * (T / T0) ** 0.5
                            + (T / T0) ** -2.5 * (0.01275 * math.exp(-2239.1 / T) / (fro + f * f / fro)
                                                  + 0.1068 * math.exp(-3352.0 / T) / (frn + f * f / frn)))


GAMMA_AIR, PRANDTL_AIR, RHO_AIR, C_GROUND = 1.4, 0.71, 1.2, 343.0
# name: (effective flow resistivity cgs, tortuosity, porosity), three-parameter fits
HESS_SOILS = {"sand": (42.0, 2.15, 0.36), "sand2": (19.058, 1.27, 0.48), "sandy_loam_dry": (24.05, 3.32, 0.52),
              "sandy_loam_moist": (9.38, 1.23, 0.39), "clay_wheeled": (14.085, 3.87, 0.53), "clay_unwheeled": (12.07, 1.94, 0.57),
              "clay_wheeled_2": (3.68, 2.68, 0.42), "clay_unwheeled_2": (6.28, 3.35, 0.57), "silt_cultivated": (8.67, 2.89, 0.56),
              "silt_cultivated_2": (6.04, 2.24, 0.49), "silt_compacted": (14.26, 2.41, 0.18), "snow": (3.47, 2.70, 0.88),
              "snow_b": (1.42, 1.11, 0.60)}
# name: layered_impedance arguments (humus sigma, porosity, depth m, mineral soil sigma, porosity)
FOREST_FLOORS = {"red_pine": (49.63, 0.51, 0.020, 1350.0, 0.28), "hardwood": (32.54, 0.5, 0.053, 486.0, 0.3)}


def ground_impedance(freqs: np.ndarray, sigma: float = 300.0, model: str = "miki", tortuosity: float = 1.0,
                     porosity: float = 1.0) -> np.ndarray:
    """Normalised characteristic impedance Z/rho c of a porous ground (e^{-iwt} convention), sigma = flow
    resistivity in cgs units (1 cgs = 1 kPa s/m^2), x = f/sigma (= 1000 f/sigma_MKS).

    ``miki`` (default): the Delany-Bazley data refitted so that Z stays positive-real:
    1 + 5.50 x^-0.632 + i 8.43 x^-0.632, physical below x = 10. ``delany``: 1 + 9.08 x^-0.75 + i 11.9 x^-0.73.
    Capped at x = 1000. ``attenborough3``: three parameters, sigma the effective flow resistivity, with
    ``tortuosity`` T and ``porosity`` W (``HESS_SOILS`` presets):
    kb/k0 = sqrt(g) [(4/3 - (g-1) Npr/g) T + i 2 sigma/(pi rho0 f)]^0.5, Z = (k0/kb) [(4/3) T + i 2 sigma/(pi rho0 f)] / W.
    ``slit``: identical slit-like pores, same three parameters, lambda = sqrt(3 rho0 w T / (W sigma)),
    rho = (rho0 T / W) / (1 - th(lambda s) / (lambda s)), C = (W / g P0) (1 + (g-1) th(lambda s') / (lambda s')),
    s = sqrt(-i), s' = sqrt(-i Npr), Z = sqrt(rho / C) / (rho0 c0)."""
    f = np.maximum(np.asarray(freqs, float), 1e-3)
    if model == "slit":
        T, W, s = float(tortuosity), float(porosity), 1000.0 * float(sigma)
        p0 = 101325.0
        lam = np.sqrt(3.0 * RHO_AIR * 2.0 * math.pi * f * T / (W * s))
        zv = lam * np.sqrt(-1j)
        zt = lam * np.sqrt(-1j * PRANDTL_AIR)
        rho = RHO_AIR * T / W / (1.0 - np.tanh(zv) / zv)
        comp = W / (GAMMA_AIR * p0) * (1.0 + (GAMMA_AIR - 1.0) * np.tanh(zt) / zt)
        return np.sqrt(rho / comp) / (RHO_AIR * math.sqrt(GAMMA_AIR * p0 / RHO_AIR))
    if model == "attenborough3":
        T, W, s = float(tortuosity), float(porosity), 1000.0 * float(sigma)
        visc = 2j * s / (math.pi * RHO_AIR * f)
        kb = math.sqrt(GAMMA_AIR) * np.sqrt((4.0 / 3.0 - (GAMMA_AIR - 1.0) / GAMMA_AIR * PRANDTL_AIR) * T + visc)
        return (4.0 / 3.0 * T + visc) / (kb * W)
    x = np.minimum(f / float(sigma), 1000.0)
    if model == "delany":
        return 1.0 + 9.08 * x ** -0.75 + 1j * 11.9 * x ** -0.73
    if model != "miki":
        raise ValueError("model must be miki, delany, attenborough3 or slit")
    return 1.0 + (5.50 + 8.43j) * x ** -0.632


def layered_impedance(freqs: np.ndarray, sigma1: float, porosity1: float, depth: float, sigma2: float,
                      porosity2: float) -> np.ndarray:
    """Normalised surface impedance (e^{-iwt}) of a porous layer ``depth`` m thick on a porous half-space.
    Each medium: Z = 0.436 (sigma_e/f)^0.5 (1 + i), k = 0.0112 (porosity^2 sigma_e)^0.5 f^0.5 (1 + i) m^-1,
    sigma_e in cgs (kPa s/m^2, taken x 1000 as N s/m^4); Z_s = Z1 (Z2 - i Z1 tan(k1 d)) / (Z1 - i Z2 tan(k1 d))."""
    f = np.maximum(np.asarray(freqs, float), 1e-3)

    def zk(sig, om):
        s = 1000.0 * float(sig)
        return 0.436 * np.sqrt(s / f) * (1 + 1j), 0.0112 * float(om) * math.sqrt(s) * np.sqrt(f) * (1 + 1j)

    z1, k1 = zk(sigma1, porosity1)
    z2, _ = zk(sigma2, porosity2)
    kd = k1 * float(depth)
    t = np.where(kd.imag > 20.0, 1j, np.tan(np.where(kd.imag > 20.0, 0.0, kd)))
    return z1 * (z2 - 1j * z1 * t) / (z1 - 1j * z2 * t)


def ground_effect(freqs: np.ndarray, d: float, h_source: float = 1.5, h_listener: float = 1.5, sigma: float = 300.0,
                  temp_c: float = 20.0, layer: tuple | None = None) -> np.ndarray:
    """Transfer function (direct + ground-reflected path) / (direct path) at ``freqs`` for a source and a
    listener at the given heights, ``d`` metres apart (numpy's e^{+iwt} convention, ready for rfft).

    Reflected wave = image source x the spherical-wave reflection coefficient Q = Rp + (1 - Rp) F(w): plane-wave
    Rp = (Z sin psi - 1)/(Z sin psi + 1), boundary loss F(w) = 1 + i sqrt(pi) w wofz(w),
    w = (1 + i)/2 sqrt(k r2) (sin psi + 1/Z) (Weyl-van der Pol). Over grass this gives the classic +6 dB
    at low frequencies, a dip of 10-15 dB around 300-800 Hz at 100 m, and a comb above. ``sigma`` <= 0:
    rigid ground (Q = 1). ``layer`` = (sigma1, porosity1, depth m, sigma2, porosity2): a crust or a soaked
    top layer through ``layered_impedance`` instead of ``sigma``."""
    from scipy.special import wofz
    f = np.maximum(np.asarray(freqs, float), 1e-3)
    r1 = math.hypot(d, h_source - h_listener)
    r2 = math.hypot(d, h_source + h_listener)
    sn = (h_source + h_listener) / r2
    k = 2 * math.pi * f / speed_of_sound(temp_c)
    if sigma <= 0:
        q = np.ones_like(f, dtype=complex)
    else:
        z = layered_impedance(f, *layer) if layer is not None else ground_impedance(f, sigma)
        rp = (z * sn - 1.0) / (z * sn + 1.0)
        w = 0.5 * (1 + 1j) * np.sqrt(k * r2) * (sn + 1.0 / z)
        q = rp + (1.0 - rp) * (1.0 + 1j * math.sqrt(math.pi) * w * wofz(w))
    return np.conj(1.0 + (r1 / r2) * q * np.exp(1j * k * (r2 - r1)))


def distance(x: np.ndarray, r: float, *, ref: float = 1.0, rh: float = 50, line_source: bool = False,
             delay: bool = False, air: bool = True, temp_c: float = 20.0) -> np.ndarray:
    """Place a (mono or stereo) sound ``r`` metres away (it was recorded at ``ref`` metres).

    Spreading: ref/r (point, -6 dB per doubling) or sqrt(ref/r) (line source, -3 dB).
    Air: zero-phase magnitude filter from ``air_db_per_m`` over (r - ref) metres.
    Delay: (r - ref) / c(T) s of silence prepended when ``delay`` is True.
    """
    x = np.asarray(x, float)
    r = max(r, 1e-3)
    g = np.sqrt(ref / r) if line_source else ref / r
    extra = max(r - ref, 0.0)
    n = x.shape[0]
    pad = 1 << int(np.ceil(np.log2(n + seconds(0.05))))
    fr = np.fft.rfftfreq(pad, 1 / SR)
    mag = 10 ** (-air_db_per_m(fr, rh, temp_c) * extra / 20) if air else np.ones_like(fr)
    X = np.fft.rfft(x, pad, axis=0)
    X = X * (mag if x.ndim == 1 else mag[:, None])
    y = np.fft.irfft(X, pad, axis=0)[:n] * g
    if delay and extra > 0:
        d = seconds(extra / speed_of_sound(temp_c))
        y = np.concatenate([np.zeros((d,) + y.shape[1:]), y])
    return y


# ==========================================================================================
# Self-test
# ==========================================================================================

def _schroeder_t60(ir: np.ndarray, lo: float, hi: float, start_db: float = -5.0, stop_db: float = -25.0) -> float:
    """T60 from the Schroeder backward-integrated decay of ``ir`` band-passed to [lo, hi]."""
    fc = np.sqrt(lo * hi)
    q = fc / (hi - lo)
    y = bandpass(bandpass(ir, fc, q), fc, q)
    e = np.cumsum((y ** 2)[::-1])[::-1]
    edc = 10 * np.log10(e / e[0] + 1e-30)
    i0 = np.nonzero(edc <= start_db)[0][0]
    i1 = np.nonzero(edc <= stop_db)[0][0]
    t = np.arange(i0, i1) / SR
    slope = np.polyfit(t, edc[i0:i1], 1)[0]
    return float(-60.0 / slope)


if __name__ == "__main__":
    out_dir = DEV_DIR / "space"
    ok = True

    def check(name, cond, msg):
        global ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}: {msg}")

    imp = np.zeros(seconds(10.0))
    imp[0] = 1.0
    for t60dc, t60m, f2 in ((3.0, 2.0, 6000.0), (1.5, 1.0, 6000.0)):
        ir = zita_rev1(imp, rdel=20, f1=200, f2=f2, t60dc=t60dc, t60m=t60m)
        write_wav(out_dir / f"zita_ir_{t60m}s.wav", 0.9 * ir / np.max(np.abs(ir)))
        m = ir.mean(axis=1)
        tm = _schroeder_t60(m, 700, 1400, -10, -40)
        tl = _schroeder_t60(m, 50, 110, -10, -40)
        th = _schroeder_t60(m, f2 * 0.8, f2 * 1.25, -10, -40)
        check(f"zita mid T60 ({t60m} s)", abs(tm / t60m - 1) < 0.15, f"measured {tm:.2f} s at 1 kHz")
        check(f"zita low T60 ({t60dc} s)", abs(tl / t60dc - 1) < 0.2, f"measured {tl:.2f} s at 50-110 Hz")
        check(f"zita f2 T60 (~{t60m / 2} s)", abs(th / (t60m / 2) - 1) < 0.3, f"measured {th:.2f} s at {f2:.0f} Hz")
    ir8 = zita_rev1(np.r_[1.0, np.zeros(seconds(30.0))], t60dc=8.0, t60m=8.0)
    _, db = envelope_db(ir8.mean(axis=1), hop=0.5)
    growth = np.max(np.diff(db[2:]))
    check("zita stable", np.all(np.isfinite(ir8)) and growth < 0.5 and db[-1] < db[2] - 40,
          f"t60 8 s over 30 s: level {db[2]:.1f} -> {db[-1]:.1f} dB, max step {growth:+.2f} dB per 0.5 s")

    g8 = puckette_rev(imp, t60=2.0, damp_hz=None)
    write_wav(out_dir / "puckette_ir.wav", 0.9 * g8 / np.max(np.abs(g8)))
    tp = _schroeder_t60(g8.mean(axis=1), 700, 1400)
    check("puckette T60", abs(tp / 2.0 - 1) < 0.2, f"g from t60=2 s -> measured {tp:.2f} s")
    g8d = puckette_rev(imp, feedback=95, damp_hz=3000)
    check("puckette damped finite", np.all(np.isfinite(g8d)), f"peak {np.max(np.abs(g8d)):.3f}")

    fr = farnell_room(imp)
    write_wav(out_dir / "farnell_room_ir.wav", 0.9 * fr / np.max(np.abs(fr)))
    tf = _schroeder_t60(fr.mean(axis=1), 150, 400)
    check("farnell room tight", 0.25 < tf < 1.0, f"measured T60 {tf:.2f} s (author: 'about half a second')")

    bands = sabine_t60(2000.0, {"marble": 1200.0, "people": 10.0})
    zp = zita_from_bands(bands)
    print(f"  sabine marble hall V=2000 m3: T60 per band {np.round(bands, 2).tolist()} -> zita {zp}")
    ad = air_db_per_m(np.array([500, 1000, 4000, 8000]), 50) * 1000
    print(f"  air absorption RH 50 %: {np.round(ad, 1).tolist()} dB/km at 0.5/1/4/8 kHz")
    noise = np.random.default_rng(2).standard_normal(seconds(1.0))
    far = distance(noise, 200.0)
    lvl = lambda x, fc: 10 * np.log10(np.mean(bandpass(bandpass(x, fc, 10), fc, 10) ** 2))
    d4 = (lvl(noise, 4000) - lvl(far, 4000)) - (lvl(noise, 400) - lvl(far, 400))
    check("air absorption", 4 < d4 < 7, f"200 m: extra loss at 4 kHz vs 400 Hz = {d4:.1f} dB (ISO 9613-1: (29.7 - 2.2) dB/km x 0.199 km = 5.5 dB)")
    sp = 20 * np.log10(np.std(distance(noise, 2.0, air=False)) / np.std(noise))
    check("spreading", abs(sp + 6.02) < 0.1, f"point source 1 m -> 2 m: {sp:.2f} dB")
    print("space self-test", "OK" if ok else "FAILED")


# ==========================================================================================
# Ray tracing of a box room (specular + Lambert scattering)
# ==========================================================================================

@numba.njit(cache=True)
def _ray_trace_core(L, src, lis, alpha, scat, m_air, rays, tmax, bin_s, rd, seed, c):
    np.random.seed(seed)
    nb = int(tmax / bin_s) + 1
    hist = np.zeros((nb, 6))
    d0 = 0.0
    for a in range(3):
        d0 += (src[a] - lis[a]) ** 2
    d0 = max(math.sqrt(d0), 0.05)
    vs = 4.0 / 3.0 * math.pi * rd ** 3
    w_hit = 4.0 * math.pi * d0 * d0 / (rays * vs)
    for r in range(rays):
        u = 2.0 * np.random.random() - 1.0
        ph = 2.0 * math.pi * np.random.random()
        s = math.sqrt(1.0 - u * u)
        d = np.array([s * math.cos(ph), s * math.sin(ph), u])
        p = src.copy()
        e = np.ones(6)
        t = 0.0
        first = True
        while t < tmax:
            tw = 1e30
            wi = 0
            for a in range(3):
                if d[a] > 1e-12:
                    tt = (L[a] - p[a]) / d[a]
                    if tt < tw:
                        tw, wi = tt, 2 * a + 1
                elif d[a] < -1e-12:
                    tt = -p[a] / d[a]
                    if tt < tw:
                        tw, wi = tt, 2 * a
            if not first:
                oc = p - lis
                bq = oc[0] * d[0] + oc[1] * d[1] + oc[2] * d[2]
                cq = oc[0] ** 2 + oc[1] ** 2 + oc[2] ** 2 - rd * rd
                disc = bq * bq - cq
                if disc > 0.0:
                    sq = math.sqrt(disc)
                    t0 = max(-bq - sq, 0.0)
                    t1 = min(-bq + sq, tw)
                    if t1 > t0:
                        ta = t + (t0 + t1) / 2.0 / c
                        k = int(ta / bin_s)
                        if k < nb:
                            for b in range(6):
                                hist[k, b] += w_hit * (t1 - t0) * e[b] * math.exp(-m_air[b] * c * ta)
            first = False
            p = p + tw * d
            t += tw / c
            ax = wi // 2
            emax = 0.0
            for b in range(6):
                e[b] *= 1.0 - alpha[wi, b]
                emax = max(emax, e[b])
            if emax < 1e-7:
                break
            if np.random.random() < scat[wi]:
                z1 = np.random.random()
                z2 = np.random.random()
                th = math.acos(math.sqrt(z1))
                fi = 2.0 * math.pi * z2
                loc = np.array([math.sin(th) * math.cos(fi), math.sin(th) * math.sin(fi), math.cos(th)])
                inward = 1.0 if wi % 2 == 0 else -1.0
                o1 = (ax + 1) % 3
                o2 = (ax + 2) % 3
                d[ax] = inward * loc[2]
                d[o1] = loc[0]
                d[o2] = loc[1]
            else:
                d[ax] = -d[ax]
    return hist


def ray_trace(dims, src, lis, alpha, scattering, m_air, rays: int = 20000, tmax: float = 2.0, bin_s: float = 0.005,
              seed: int = 0, c: float = None) -> np.ndarray:
    """Energy histogram (bins of bin_s, 6 bands) of a box room relative to the direct sound energy 1/(4 pi d0^2):
    rays leave the source uniformly, lose (1 - alpha) per wall and exp(-m c t) in air; at each wall a fraction s
    leaves diffusely (Lambert: theta = arccos(sqrt(z1)), phi = 2 pi z2), the rest specularly, so the specular part
    after n reflections is [(1 - s)(1 - alpha)]^n. A receiver sphere counts chord length x energy (direct excluded).
    `alpha` (6 walls x 6 bands) in the order x0, xL, y0, yL, floor, ceiling; `scattering` per wall."""
    c = C_AIR if c is None else float(c)
    L = np.asarray(dims, float)
    rd = float(np.clip(0.1 * (float(np.prod(L))) ** (1.0 / 3.0), 0.3, 1.5))
    return _ray_trace_core(L, np.asarray(src, float), np.asarray(lis, float), np.asarray(alpha, float),
                           np.asarray(scattering, float), np.asarray(m_air, float), int(rays), float(tmax),
                           float(bin_s), rd, int(seed), c)


# ==========================================================================================
# Room statistics, porous layers, absorbers, diffusers, sound reinforcement, partitions
# ==========================================================================================

SCATTERING = {"rpg_skyline": (0.01, 0.08, 0.45, 0.82, 1.0, 1.0), "rpg_qrd": (0.06, 0.15, 0.45, 0.95, 0.88, 0.91),
              "theatre_audience": (0.3, 0.5, 0.6, 0.6, 0.7, 0.7), "flat": (0.05, 0.05, 0.05, 0.05, 0.05, 0.05)}


def transfer_statistics(t60: float, bandwidth: float = None) -> dict:
    """Statistics of a room transfer function above the Schroeder frequency: |p|^2 exponentially distributed, mean
    spacing of maxima 4 / T Hz, the highest peak dL = 4.34 ln ln(B T) dB above the mean in a band B, phase slope
    0.455 T rad/Hz (its mean), spatial correlation distance about lambda / pi."""
    B = 1000.0 if bandwidth is None else float(bandwidth)
    T = float(t60)
    return {"max_spacing_hz": 4.0 / T, "peak_db": 4.34 * math.log(max(math.log(B * T), 1e-9)),
            "phase_slope": 0.455 * T}


def diffuse_coherence(f, d: float, dim: int = 3, c: float = None):
    """Coherence of a diffuse field between two points d metres apart: sin(kd)/(kd) in 3-D, J0(kd) in 2-D."""
    from scipy.special import j0
    c = C_AIR if c is None else float(c)
    kd = 2.0 * np.pi * np.asarray(f, float) * float(d) / c
    return j0(kd) if dim == 2 else np.sinc(kd / np.pi)


def delany_bazley(f, sigma: float):
    """Characteristic impedance Zc / (rho c) and propagation constant gamma (1/m) of a porous layer of flow
    resistivity sigma (Pa s / m^2): alpha c / w = 10.3 (f / sigma_cgs)^-0.59, beta c / w = 1 + 10.8 (f / sigma_cgs)^-0.70,
    Z = 1 + 0.0571 X^-0.754 - 0.087 i X^-0.732, X = rho f / sigma."""
    f = np.maximum(np.asarray(f, float), 1.0)
    X = 1.204 * f / float(sigma)
    zc = 1.0 + 0.0571 * X ** -0.754 - 0.087j * X ** -0.732
    s_cgs = float(sigma) / 1000.0
    w = 2.0 * np.pi * f / C_AIR
    a = w * 10.3 * (f / s_cgs) ** -0.59
    b = w * (1.0 + 10.8 * (f / s_cgs) ** -0.70)
    return zc, a + 1j * b


def layer_absorption(f, sigma: float, thickness: float):
    """Normal-incidence absorption of a porous layer on a rigid backing: Z = Zc coth(gamma l),
    alpha = 1 - |(Z - 1) / (Z + 1)|^2 (impedances relative to rho c)."""
    zc, g = delany_bazley(f, sigma)
    Z = zc / np.tanh(g * float(thickness))
    return 1.0 - np.abs((Z - 1.0) / (Z + 1.0)) ** 2


def absorption_8k(material: str) -> float:
    """Absorption at 8 kHz: the 4 kHz value of the table, rising to 1 for porous materials whose 4 kHz value is
    already above 0.6 (absorption of porous layers keeps rising with frequency)."""
    a4 = ABSORPTION[material][5]
    return min(1.0, a4 * 1.1) if a4 > 0.6 else a4


def panel_absorber_f0(mass: float, depth_cm: float, filled: bool = False) -> float:
    """Resonance of a panel absorber of mass `mass` kg/m^2 over an air gap depth_cm: 600 / sqrt(M d) (500 filled)."""
    return (500.0 if filled else 600.0) / math.sqrt(max(mass * depth_cm, 1e-9))


def helmholtz_absorber(f0: float, c: float = None) -> dict:
    """A single Helmholtz resonator at f0: maximum absorption area lambda0^2 / (2 pi), effective radius 55 / f0 m."""
    c = C_AIR if c is None else float(c)
    lam = c / float(f0)
    return {"area_max": lam * lam / (2.0 * math.pi), "radius": 55.0 / float(f0)}


def rayleigh_absorber(f, porosity: float, xi: float, depth: float, c: float = None):
    """Rayleigh porous layer on a rigid wall: k' = (w/c) sqrt(1 - i Xi / (rho w)), Z = -i Z0 cot(k' d) / porosity;
    alpha = 1 - |(Z - Z0)/(Z + Z0)|^2. Thick layer limit alpha = 4 s / (1 + s)^2."""
    c = C_AIR if c is None else float(c)
    w = 2.0 * np.pi * np.maximum(np.asarray(f, float), 1.0)
    k = (w / c) * np.sqrt(1.0 - 1j * float(xi) / (1.204 * w))
    z = -1j / np.tan(k * float(depth)) / float(porosity) * np.sqrt(1.0 - 1j * float(xi) / (1.204 * w))
    return 1.0 - np.abs((z - 1.0) / (z + 1.0)) ** 2


def perforated_mass(b: float, a: float, sigma: float, rho: float = 1.204) -> float:
    """Mass per unit area of a perforated panel: rho0 b' / sigma, b' = b + (pi / 2) a (end correction)."""
    return rho * (b + math.pi / 2.0 * a) / sigma


def fabric_absorption(f, r: float, d: float, c: float = None):
    """A fabric of flow resistance r (relative to rho c) hung d metres before a wall:
    alpha = 4 r / ((r + 1)^2 + cot^2(2 pi f d / c))."""
    c = C_AIR if c is None else float(c)
    ct = 1.0 / np.tan(2.0 * np.pi * np.maximum(np.asarray(f, float), 1e-3) * float(d) / c)
    return 4.0 * r / ((r + 1.0) ** 2 + ct ** 2)


def qrd_depths(N: int, f_design: float, c: float = None) -> np.ndarray:
    """Quadratic-residue diffuser: well n has phase chi_n = 2 pi (n^2 mod N) / N and depth h_n = lambda chi_n / (4 pi)."""
    c = C_AIR if c is None else float(c)
    n = np.arange(int(N))
    chi = 2.0 * np.pi * ((n * n) % int(N)) / int(N)
    return (c / float(f_design)) * chi / (4.0 * np.pi)


def qrd_lobes(N: int, width: float, f: float, c: float = None) -> np.ndarray:
    """Angles (degrees) of the diffraction lobes of a period of N wells of width `width`: sin th_m = m lambda / (N d)."""
    c = C_AIR if c is None else float(c)
    s = np.arange(-int(N), int(N) + 1) * (c / float(f)) / (int(N) * float(width))
    return np.degrees(np.arcsin(s[np.abs(s) <= 1.0]))


def scale_model(f, sigma: float, alpha_full=None, c_ratio: float = 1.0) -> dict:
    """Scale model 1 : sigma: frequencies f' = sigma (c'/c) f, walls must give alpha'(f') = alpha(f)."""
    return {"f_model": np.asarray(f, float) * float(sigma) * float(c_ratio), "alpha_model": alpha_full}


def pa_range(volume: float, t60: float, gamma: float = 1.0, gamma_rx: float = 1.0) -> dict:
    """Sound reinforcement in a hall: acoustic power for a mean energy density w, P = 13.8 V w / T; furthest
    listener with enough direct sound r_max = 0.057 sqrt(g g' V / T) 2^(1/T) m (gamma: loudspeaker directivity)."""
    V, T = float(volume), float(t60)
    return {"r_max": 0.057 * math.sqrt(gamma * gamma_rx * V / T) * 2.0 ** (1.0 / T), "power_per_w": 13.8 * V / T}


def schroeder_design(sr: int, t60: float = 2.0) -> dict:
    """Schroeder's rules: allpasses in series of about 100 ms / 3^i (i = 0..4) made mutually prime, g 0.708 (t60
    about 2 s); 4 parallel combs between 30 and 45 ms with open-loop gain g = 10^(-3 tau / T60), at most 0.85."""
    ap = []
    for i in range(5):
        v = int(round(0.1 / 3 ** i * sr))
        while any(math.gcd(v, a) != 1 for a in ap):
            v += 1
        ap.append(v)
    taus = np.linspace(0.030, 0.045, 4)
    combs = []
    for tau in taus:
        v = int(round(tau * sr))
        while any(math.gcd(v, a) != 1 for a in combs + ap):
            v += 1
        combs.append(v)
    g = np.minimum(10.0 ** (-3.0 * np.array(combs) / sr / float(t60)), 0.85)
    return {"allpass": ap, "allpass_g": 0.708, "combs": combs, "comb_g": g}


STC_CONTOUR = ((125, -16), (160, -13), (200, -10), (250, -7), (315, -4), (400, -1), (500, 0), (630, 1), (800, 2),
               (1000, 3), (1250, 4), (1600, 4), (2000, 4), (2500, 4), (3150, 4), (4000, 4))


def stc_rating(tl) -> int:
    """Sound transmission class from 16 third-octave transmission losses 125-4000 Hz: the highest contour (its
    500 Hz value) with at most 32 dB of total deficiency and at most 8 dB in any band."""
    tl = np.asarray(tl, float)
    ref = np.array([v for _, v in STC_CONTOUR], float)
    best = 0
    for stc in range(0, 120):
        d = np.maximum(stc + ref - tl, 0.0)
        if d.sum() <= 32.0 and d.max() <= 8.0:
            best = stc
    return best


def interface_transmission(z1: float = 413.3, z2: float = 1.48e6) -> float:
    """Intensity transmission across a plane interface at normal incidence: 4 z1 z2 / (z1 + z2)^2
    (air to water about 0.11 %, -29.5 dB)."""
    return 4.0 * z1 * z2 / (z1 + z2) ** 2


def refraction_radius(gradient_c_per_km: float, temp_c: float = 20.0) -> float:
    """Radius of curvature of a sound ray in a linear temperature gradient: R = c / |dc/dz|,
    dc/dz = (c / (2 T)) dT/dz (inf when there is no gradient)."""
    g = float(gradient_c_per_km) / 1000.0
    if g == 0.0:
        return math.inf
    c = speed_of_sound(temp_c)
    return c / abs(c / (2.0 * (temp_c + 273.15)) * g)


def shadow_excess_db(f, r: float, h_s: float, h_r: float, gradient_c_per_km: float, temp_c: float = 20.0):
    """Excess attenuation in the shadow of an upward-refracting atmosphere (temperature falling with height):
    the shadow starts at x_s = sqrt(2 R) (sqrt(h_s) + sqrt(h_r)); beyond it the path is bent past the limiting
    ray by the arc-chord difference delta = (r - x_s)^3 / (24 R^2), treated as a diffracting edge,
    10 log10(3 + 20 N), N = 2 delta f / c, from the grazing value (4.8 dB) up to 25 dB."""
    f = np.asarray(f, float)
    if gradient_c_per_km >= 0:
        return np.zeros_like(f)
    R = refraction_radius(gradient_c_per_km, temp_c)
    xs = math.sqrt(2.0 * R) * (math.sqrt(max(h_s, 0.0)) + math.sqrt(max(h_r, 0.0)))
    if r <= xs:
        return np.zeros_like(f)
    delta = (r - xs) ** 3 / (24.0 * R * R)
    N = 2.0 * delta * f / speed_of_sound(temp_c)
    return np.minimum(10.0 * np.log10(3.0 + 20.0 * N), 25.0)


def duct_gain(r: float, h_s: float, gradient_c_per_km: float, temp_c: float = 20.0) -> float:
    """Downward refraction (inversion): rays leaving the source horizontally return to the ground after
    x_c = sqrt(8 R h_s); beyond it spreading turns cylindrical, gain sqrt(r / x_c), at most +10 dB."""
    if gradient_c_per_km <= 0:
        return 1.0
    R = refraction_radius(gradient_c_per_km, temp_c)
    xc = math.sqrt(8.0 * R * max(h_s, 0.1))
    return float(min(math.sqrt(max(r / xc, 1.0)), 10.0 ** 0.5))


# ==========================================================================================
# Loudspeaker panning, Ambisonics, transaural
# ==========================================================================================

def bformat(x, azimuth: float, elevation: float = 0.0) -> np.ndarray:
    """First-order B-format (W, X, Y, Z): W = S / sqrt 2, X = cos th cos ph S, Y = sin th cos ph S, Z = sin ph S."""
    th, ph = math.radians(azimuth), math.radians(elevation)
    x = np.asarray(x, float)
    return np.stack([x / math.sqrt(2.0), math.cos(th) * math.cos(ph) * x, math.sin(th) * math.cos(ph) * x,
                     math.sin(ph) * x], axis=1)


def ambi_decode(b, speakers, g1: float = 1.0, g2: float = 1.0) -> np.ndarray:
    """Decode B-format to loudspeakers at (azimuth, elevation) degrees: g_i = 1/2 [G1 sqrt2 W + G2 (X, Y, Z) . l_i]."""
    b = np.asarray(b, float)
    out = []
    for az, el in speakers:
        th, ph = math.radians(az), math.radians(el)
        l = np.array([math.cos(th) * math.cos(ph), math.sin(th) * math.cos(ph), math.sin(ph)])
        out.append(0.5 * (g1 * math.sqrt(2.0) * b[:, 0] + g2 * (b[:, 1:] @ l)))
    return np.stack(out, axis=1)


def vbap_gains(azimuth: float, speakers=(-30.0, 30.0)) -> np.ndarray:
    """2-D vector-base amplitude panning between two loudspeakers: g = p^T L^-1, normalised to unit power."""
    p = np.array([math.cos(math.radians(azimuth)), math.sin(math.radians(azimuth))])
    Lm = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in speakers])
    g = p @ np.linalg.inv(Lm)
    return g / max(float(np.linalg.norm(g)), 1e-12)


def transaural_gain(theta: float, D: float = 2.0, delta: float = 0.18, c: float = None) -> dict:
    """Cross-talk path of a loudspeaker pair at +-theta degrees, distance D: relative gain D / (delta sin th + D),
    extra delay about 1.5 delta sin th / c."""
    c = C_AIR if c is None else float(c)
    s = math.sin(math.radians(theta))
    return {"gain": D / (delta * s + D), "delay": 1.5 * delta * s / c}
