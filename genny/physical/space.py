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
  (-3 dB per doubling) (Farnell rule 19, research 01 §5); air absorption from JOS PASP
  Table B.2 (dB/km at 1-4 kHz, RH 40-70 %, research 10 §2.5), extended outside 1-4 kHz with
  the f^2 (Stokes) law (Farnell rule 20); propagation delay r / c, c = 345 m/s (JOS).
"""

from __future__ import annotations

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
def _g08_core(a, b, L, g, lp):
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
                d[k] = s[k]
        n1 = a[i] + d[0]
        n2 = b[i] + d[1]
        n6 = d[2] + d[3]
        n8 = d[2] - d[3]
        n5 = n1 + n2
        n7 = n1 - n2
        w[0] = g * (n5 + n6)
        w[1] = g * (n7 + n8)
        w[2] = g * (n5 - n6)
        w[3] = g * (n7 - n8)
        for k in range(4):
            bufs[k, idx[k]] = w[k]
            idx[k] += 1
            if idx[k] >= L[k]:
                idx[k] = 0
        yl[i] = n1
        yr[i] = n2
    return yl, yr


def puckette_rev(x: np.ndarray, feedback: float = 90.0, t60: float | None = None,
                 damp_hz: float | None = None) -> np.ndarray:
    """Puckette G08 reverberator (mono in, stereo (n, 2) out, early + late, no dry).

    ``feedback`` 0-100 -> g = feedback/200 (100 = infinite). If ``t60`` is given, g is set from
    RT60 = -3 d / log10(2 g) (TTEM; d = mean loop length). ``damp_hz`` adds the one-pole
    low-pass per line. Early stages are scaled by sqrt(1/2) each (Puckette's deferred
    normalisation) so the early block has unit power gain.
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
    yl, yr = _g08_core(np.ascontiguousarray(a), np.ascontiguousarray(b), L, float(g), lp)
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
}


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

C_AIR = 345.0
AIR_BANDS = (1000.0, 2000.0, 3000.0, 4000.0)
AIR_DB_PER_KM = {  # JOS PASP Air_Absorption Table B.2, 20 C (research 10 §2.5), verbatim
    40: (5.6, 16.0, 30.0, 105.0),
    50: (5.6, 12.0, 26.0, 90.0),
    60: (5.6, 12.0, 24.0, 73.0),
    70: (5.6, 12.0, 22.0, 63.0),
}


def air_db_per_m(freqs: np.ndarray, rh: int = 50) -> np.ndarray:
    """Air absorption in dB/m: Table B.2 between 1 and 4 kHz (log-frequency interpolation),
    f^2 (Stokes) extension below 1 kHz and above 4 kHz from the edge values."""
    tab = np.asarray(AIR_DB_PER_KM[min(AIR_DB_PER_KM, key=lambda k: abs(k - rh))]) / 1000.0
    f = np.maximum(np.asarray(freqs, float), 1.0)
    inside = np.interp(np.log(f), np.log(AIR_BANDS), tab)
    lo = tab[0] * (f / AIR_BANDS[0]) ** 2
    hi = tab[-1] * (f / AIR_BANDS[-1]) ** 2
    return np.where(f < AIR_BANDS[0], lo, np.where(f > AIR_BANDS[-1], hi, inside))


def distance(x: np.ndarray, r: float, *, ref: float = 1.0, rh: int = 50, line_source: bool = False,
             delay: bool = False, air: bool = True) -> np.ndarray:
    """Place a (mono or stereo) sound ``r`` metres away (it was recorded at ``ref`` metres).

    Spreading: ref/r (point, -6 dB per doubling) or sqrt(ref/r) (line source, -3 dB).
    Air: zero-phase magnitude filter from ``air_db_per_m`` over (r - ref) metres.
    Delay: (r - ref) / 345 s of silence prepended when ``delay`` is True.
    """
    x = np.asarray(x, float)
    r = max(r, 1e-3)
    g = np.sqrt(ref / r) if line_source else ref / r
    extra = max(r - ref, 0.0)
    n = x.shape[0]
    pad = 1 << int(np.ceil(np.log2(n + seconds(0.05))))
    fr = np.fft.rfftfreq(pad, 1 / SR)
    mag = 10 ** (-air_db_per_m(fr, rh) * extra / 20) if air else np.ones_like(fr)
    X = np.fft.rfft(x, pad, axis=0)
    X = X * (mag if x.ndim == 1 else mag[:, None])
    y = np.fft.irfft(X, pad, axis=0)[:n] * g
    if delay and extra > 0:
        d = seconds(extra / C_AIR)
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
    check("air absorption", 15 < d4 < 21, f"200 m: extra loss at 4 kHz vs 400 Hz = {d4:.1f} dB (table 90 dB/km x 0.199 km = 17.9 dB)")
    sp = 20 * np.log10(np.std(distance(noise, 2.0, air=False)) / np.std(noise))
    check("spreading", abs(sp + 6.02) < 0.1, f"point source 1 m -> 2 m: {sp:.2f} dB")
    print("space self-test", "OK" if ok else "FAILED")
