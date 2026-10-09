"""Sounds organised by state of matter: liquids, gases, fire / plasma / electricity, phase changes.

Everything is rendered at the physical engine's internal rate (48 kHz, the rate of the reused
``genny.physical`` kernels) and resampled at the boundary with ``scipy.signal.resample_poly``.

Sources (findings files are in ``out/research``):

* **Bubbles** - Zheng & James, "Harmonic Fluids" (HF), findings g07 section 2: Minnaert frequency
  with surface tension, radius-dependent radiation + viscous + thermal damping (Appendix A, Table 3),
  attack window (eq. 39), depth-driven pitch rise (eq. 6, K = 72.95, eta = 1.24), bubble counts and
  frequency ranges per scene (Table 1). Moss et al. "Sounding Liquids" (findings 06 section 3):
  amplitude eps*r0, radius power laws, tube bubbles. Farnell, "Designing Sound" Practical 12-14
  (findings 02 sections 7-9): terminal rise speed (2/3)sqrt(gR), burst timing, pouring vessel.
  "The Sounding Object" ch. 9 (findings g05 section 5): recursive secondary drops, two-band sizes.
* **Fire** - Chadwick & James, "Animating Fire with Sound" (FIRE), findings g07 section 3: p = dI/dt,
  30 Hz high-pass, 180 Hz hand-over, f^-alpha bandwidth extension (Algorithm 1, Table 1). The flame
  flux I(t) is a stochastic surrogate (*adaptation*, g07 section 3.8). Crackle / hiss / lapping are
  Farnell Practical 11 (findings 02 section 6, 05 section 8.2).
* **Electricity** - Farnell Practical 16 (findings 02 section 11) and the ``spark6formant`` bank
  (findings 03, explosion "mid" layer). Thunder is ``physical.ambience.thunder``.
* **Wind** - Farnell Practical 18 through ``physical.ambience.wind_scene`` (findings 02 section 13).
* **Jets** - Strouhal peak f = St U / D with St = 0.2 (Selfridge, findings 06 section 4.1).
  The findings contain no jet-noise level law; the textbook velocity power laws are marked UNSOURCED.
* **Ice** - no findings file covers ice. The dispersive chirp is the textbook Kirchhoff thin-plate
  flexural wave (derivation in ``ice_chirp``); the material constants are marked UNSOURCED.
"""
from __future__ import annotations

import math

import numba
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import lfilter, resample_poly

from . import filters as F
from .core import DEFAULT_SR
from .physical import ambience as amb
from .physical import modal as M
from .physical import particles as P
from .physical.core import SR, contact_pulse, tv_bandpass
from .sfx import sfx

# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _n(dur) -> int:
    return max(256, int(round(float(dur) * SR)))


def _rng(seed, salt: int) -> np.random.Generator:
    return np.random.default_rng([abs(int(seed)), salt])


def _lp(x, fc, order=1):
    return F.lowpass(x, float(min(fc, 0.45 * SR)), SR, order=order)


def _hp(x, fc, order=1):
    return F.highpass(x, float(fc), SR, order=order)


def _bp(x, fc, q=1.0):
    return F.bandpass(x, float(min(fc, 0.45 * SR)), SR, q=q)


def _tvbp(x, f, q):
    n = len(x)
    return tv_bandpass(np.ascontiguousarray(x, dtype=np.float64),
                       np.ascontiguousarray(np.broadcast_to(np.asarray(f, float), (n,))),
                       np.ascontiguousarray(np.broadcast_to(np.asarray(q, float), (n,))))


def _lfn(rng, n: int, fc: float) -> np.ndarray:
    """Smooth band-limited noise (unit-ish variance): cubic spline through Gaussian knots spaced
    1/(2 fc) s. No filter warm-up, and smooth by construction (FIRE section 3.3: the flame driver
    must be smooth; they used a cubic B-spline)."""
    k = int(n / SR * 2.0 * fc) + 4
    return CubicSpline(np.arange(k), rng.standard_normal(k))(np.arange(n) * (2.0 * fc / SR))


def _place(buf, sig, i: int, g: float = 1.0) -> None:
    i = int(i)
    if 0 <= i < len(buf):
        m = min(len(sig), len(buf) - i)
        buf[i:i + m] += g * sig[:m]


def _unit(x):
    """Scale to unit RMS over the active part (for mixing components by stated weights)."""
    x = np.asarray(x, float)
    if x.ndim == 2:
        x = x.mean(axis=1)
    r = math.sqrt(float(np.mean(x * x)))
    return x / r if r > 1e-12 else x


def _fit(x, n):
    x = np.asarray(x, float)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


def _ar(n, attack, release):
    t = np.arange(n) / SR
    return np.clip(t / max(attack, 1e-4), 0, 1) * np.clip((n / SR - t) / max(release, 1e-4), 0, 1)


def _fin(y, sr, peak=0.7, fin=0.002, fout=0.02) -> np.ndarray:
    """Mono, DC removed, peak-normalised, resampled to ``sr``, zero at both ends."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 2:
        y = y.mean(axis=1)
    y = _hp(np.where(np.isfinite(y), y, 0.0), 20.0)
    sr = int(sr)
    if sr != SR:
        g = math.gcd(sr, SR)
        y = resample_poly(y, sr // g, SR // g)
    m = float(np.max(np.abs(y)))
    if m > 1e-12:
        y = y * (peak / m)
    a = min(max(2, int(fin * sr)), len(y) // 2)
    b = min(max(2, int(fout * sr)), len(y) // 2)
    y[:a] *= np.linspace(0.0, 1.0, a)
    y[-b:] *= np.linspace(1.0, 0.0, b)
    return y


# =============================================================================================
# LIQUIDS - the acoustic bubble (HF Appendix A, findings g07 section 2.2)
# =============================================================================================

G_GRAV, RHO_W, P_ATM, GAMMA = 9.8, 1000.0, 101325.0, 1.4          # HF Table 3
SIGMA_W, D_GAS, C_WATER, MU_W = 0.0726, 2.122e-5, 1497.0, 8.9e-4  # HF Table 3
G_TH = 3.0 * GAMMA * P_ATM / (4.0 * math.pi * RHO_W * D_GAS)      # 1.60e6 s/m
K_RISE, ETA_RISE = 72.95, 1.24                                    # HF eq. 6
MINNAERT_WR = math.sqrt(3.0 * GAMMA * P_ATM / RHO_W)              # w0 * r0 = 20.63 m/s (f0 r0 = 3.28)


def minnaert(r0):
    """Minnaert frequency in Hz: f = (1 / 2 pi r) sqrt(3 gamma P0 / rho)."""
    return MINNAERT_WR / (2.0 * np.pi * np.asarray(r0, float))


def bubble_params(r0):
    """HF eq. 36-38: returns (w0, beta, w_d, delta) for radius r0 in metres (rad/s, 1/s, rad/s, -).

    w0 = sqrt(3 gamma p0 - 2 sigma / r0) / (r0 sqrt(rho));  beta = w0 delta / sqrt(delta^2 + 4);
    delta = w0 r0 / c_f  +  4 mu / (rho w0 r0^2)  +  2 (sqrt(psi - 3) - (3g-1)/(3(g-1))) / (psi - 4),
    psi = 16 G_th g / (9 (gamma - 1)^2 w0).
    """
    r0 = np.asarray(r0, float)
    w0 = np.sqrt(3 * GAMMA * P_ATM - 2 * SIGMA_W / r0) / (r0 * math.sqrt(RHO_W))
    psi = 16 * G_TH * G_GRAV / (9 * (GAMMA - 1) ** 2 * w0)
    d = (w0 * r0 / C_WATER + 4 * MU_W / (RHO_W * w0 * r0 * r0)
         + 2 * (np.sqrt(psi - 3) - (3 * GAMMA - 1) / (3 * (GAMMA - 1))) / (psi - 4))
    beta = w0 * d / np.sqrt(d * d + 4)
    return w0, beta, np.sqrt(w0 * w0 - beta * beta), d


@numba.njit(cache=True)
def _beta_of(w):
    """Damping rate at angular frequency w, the radius taken as r0 = sqrt(3 gamma p0 / rho) / w
    (g07 section 2.8)."""
    r0 = MINNAERT_WR / w
    psi = 16.0 * G_TH * G_GRAV / (9.0 * (GAMMA - 1.0) ** 2 * w)
    d = (w * r0 / C_WATER + 4.0 * MU_W / (RHO_W * w * r0 * r0)
         + 2.0 * (np.sqrt(psi - 3.0) - (3.0 * GAMMA - 1.0) / (3.0 * (GAMMA - 1.0))) / (psi - 4.0))
    return w * d / np.sqrt(d * d + 4.0)


@numba.njit(cache=True)
def _bubble_bank(out, starts, w0s, amps, depths, vrise, phi0s, wmax):
    """Sum of HF bubbles. Each is q'' + 2 beta q' + w^2 q = impulse with time-varying w, beta.

    Integrator: exact amplitude/phase recursion (a *= exp(-beta dt), phase += w_d dt), which is the
    closed-form impulse response for constant coefficients. The findings' reference (g07 section 2.8)
    uses the paper's midpoint rule; checked numerically at 48 kHz that rule is 3 % sharp at 3.3 kHz
    and loses half of the damping (|mu| ~ 1 + (w dt)^4 / 8), so it is not used.
    Pitch rise per sample (HF eq. 6): dw = K w_d exp(-eta (phi/phi0 - 1)) dphi, clamped to
    wmax * w0 (adaptation, g07 section 2.4) and to 0.45 of the sample rate.
    Attack window (HF eq. 39): gain exp(-(e - 0.85)^2 / 0.0028125) while the envelope e >= 0.85.
    """
    n = out.shape[0]
    dt = 1.0 / 48000.0
    wnyq = 2.0 * np.pi * 0.45 * 48000.0
    for k in range(starts.shape[0]):
        s = starts[k]
        w0 = min(w0s[k], wnyq)
        w = w0
        b = _beta_of(w)
        phi = -abs(depths[k])
        m = min(int(6.91 / b * 48000.0) + 1, n - s)          # HF: stop at 1/1000
        a = 1.0
        ph = 0.0
        for i in range(m):
            wd = np.sqrt(max(w * w - b * b, 0.0))
            g = np.exp(-(a - 0.85) ** 2 / 0.0028125) if a >= 0.85 else 1.0
            out[s + i] += amps[k] * a * g * np.sin(ph)
            ph += wd * dt
            a *= np.exp(-b * dt)
            if a < 1e-3:
                break
            dphi = min(vrise[k] * dt, -phi)
            if dphi > 0.0:
                phi += dphi
                e = np.exp(-1.24 * (phi / phi0s[k] - 1.0))
                if e > 1e-6:
                    w = min(w + 72.95 * wd * e * dphi, wmax * w0, wnyq)
                    b = _beta_of(w)


def hf_bubble(r0: float, depth: float = 1.0, rise: float = 1.0, phi0: float = -0.016,
              dur: float | None = None) -> np.ndarray:
    """One Harmonic-Fluids bubble of radius ``r0`` (m) at 48 kHz, unit amplitude.

    depth: start depth below the surface (m). Deeper than a few |phi0| = constant pitch.
    rise: multiplier on Farnell's terminal rise speed (2/3) sqrt(g r) (findings 02 section 7).
    phi0: distance at which the pitch starts to move, -0.008 ... -0.025 m (HF eq. 6).
    """
    w0, beta, _, _ = bubble_params(r0)
    n = _n(dur if dur else 7.0 / float(beta))
    out = np.zeros(n)
    v = rise * (2.0 / 3.0) * math.sqrt(G_GRAV * r0)
    _bubble_bank(out, np.zeros(1, np.int64), np.array([float(w0)]), np.ones(1), np.array([float(depth)]),
                 np.array([v]), np.array([float(phi0)]), 3.0)
    return out


def _radii(rng, k: int, lo: float, hi: float, dist: str = "gauss", alpha: float = 2.9) -> np.ndarray:
    """Bubble radii in [lo, hi] metres.

    gauss: HF Appendix B draws radii from a Gaussian; mean and deviation are not printed.
      # UNSOURCED: mean = geometric mean of the range, sd = range / 6.
    two_band: two Gaussians (Sounding Object, g05 section 5: a two-band distribution "gives a more
      convincing result" for streaming; no numbers printed). # UNSOURCED: band centres.
    power: p(r) ~ r^-alpha (Moss: breaking waves 1.5-3.3, rain 2.9).
    """
    if k <= 0:
        return np.zeros(0)
    mid = math.sqrt(lo * hi)
    if dist == "power":
        return P._powerlaw(rng, k, lo, hi, -alpha)
    if dist == "two_band":
        low = rng.random(k) < 0.5
        r = np.where(low, rng.normal(math.sqrt(lo * mid), (mid - lo) / 4, k),
                     rng.normal(math.sqrt(mid * hi), (hi - mid) / 5, k))
    else:
        r = rng.normal(mid, (hi - lo) / 6, k)
    r = lo + np.abs(r - lo)                       # reflect at the lower bound
    return np.clip(r, lo, hi)


def bubble_times(n: int, rng, rate, cluster: float = 0.0) -> np.ndarray:
    """Sorted bubble onset samples: N ~ Poisson(integral of rate), placed by the rate density.

    cluster in [0, 1): that fraction of the bubbles are children trailing a parent bubble with
    exponential delays (mean 30 ms) - Farnell's "bursts that decay in frequency, followed by a period
    of few bubbles" (findings 02 section 7). # UNSOURCED: the 30 ms burst constant.
    Onsets land on arbitrary samples (HF / Moss: spread them to avoid a frame-rate buzz).
    """
    lam = np.broadcast_to(np.asarray(rate, float), (n,)) if np.ndim(rate) == 0 else _fit(rate, n)
    cdf = np.cumsum(np.maximum(lam, 0.0))
    k = int(rng.poisson(cdf[-1] / SR))
    if k == 0:
        return np.zeros(0, np.int64)
    t = np.interp(rng.random(k) * cdf[-1], cdf, np.arange(n))
    npar = max(1, int(round(k * (1.0 - min(max(cluster, 0.0), 0.95)))))
    kids = t[rng.integers(0, npar, k - npar)] + rng.exponential(0.03 * SR, k - npar)
    return np.sort(np.concatenate([t[:npar], kids % n])).astype(np.int64)


def _render_bubbles(n, rng, starts, r, depth=(0.002, 0.03), rise=1.0) -> np.ndarray:
    """Amplitude: Moss eps * r0, eps power law on [0.01, 0.1] (mu = 2 # UNSOURCED), times a
    log-uniform +-10 dB transfer gain (HF section 2.7: equal-gain banks sound "harsh and
    unrealistic"; # UNSOURCED distribution). Start depth uniform in ``depth`` (# UNSOURCED: HF takes
    it from the fluid solver), rise speed (2/3) sqrt(g r) (Farnell), phi0 uniform in HF's
    -8 ... -25 mm."""
    k = len(starts)
    out = np.zeros(n)
    if k == 0:
        return out
    w0 = bubble_params(r)[0]
    amp = P._powerlaw(rng, k, 0.01, 0.1, -2.0) * r * 1e3 * 10.0 ** (rng.uniform(-10, 10, k) / 20.0)
    _bubble_bank(out, np.ascontiguousarray(starts, np.int64), np.ascontiguousarray(w0), amp,
                 rng.uniform(depth[0], depth[1], k), rise * (2.0 / 3.0) * np.sqrt(G_GRAV * r),
                 -rng.uniform(0.008, 0.025, k), 3.0)
    return out


def bubble_cloud(dur: float, rng, rate, r_lo: float, r_hi: float, dist: str = "gauss", *,
                 alpha: float = 2.9, cluster: float = 0.0, depth=(0.002, 0.03), rise: float = 1.0,
                 r_scale=None, return_info: bool = False):
    """A population of HF bubbles at 48 kHz. ``rate`` is bubbles per second (number or per-sample
    array); ``r_scale`` an optional per-sample radius multiplier (pouring: sizes fall as it fills)."""
    n = _n(dur)
    starts = bubble_times(n, rng, rate, cluster)
    r = _radii(rng, len(starts), r_lo, r_hi, dist, alpha)
    if r_scale is not None and len(r):
        r = np.clip(r * np.asarray(r_scale)[starts], 1e-4, 0.03)
    y = _render_bubbles(n, rng, starts, r, depth, rise)
    if return_info:
        return y, dict(n=len(starts), starts=starts, r=r)
    return y


# HF Table 1 (findings g07 section 2.6): bubbles per second and radius range by scene
HF_POUR_RATE, HF_BABBLE_RATE, HF_SPLASH_COUNT, HF_DROP_BUBBLES = 1580.0, 3100.0, 127, 14.0 / 3.0
R_300, R_500, R_4K, R_5K, R_6K = 10.9e-3, 6.6e-3, 0.82e-3, 0.66e-3, 0.55e-3


def drop_times(dur: float, rng, rate: float, rate_end: float | None = None,
               jitter: float = 0.15) -> np.ndarray:
    """Drop onset times (s). A dripping source is a relaxation oscillator (surface tension against
    weight, Farnell thesis p.80-81): nearly periodic. jitter 0 = periodic, 1 = Poisson
    (exponential gaps); the mean gap is 1 / rate either way. ``rate_end`` ramps the rate linearly."""
    rate = max(float(rate), 1e-3)
    rate_end = rate if rate_end is None else max(float(rate_end), 1e-3)
    j = min(max(float(jitter), 0.0), 1.0)
    t = rng.uniform(0.0, 1.0 / rate)
    out = []
    while t < dur:
        out.append(t)
        r = rate + (rate_end - rate) * t / dur
        t += ((1.0 - j) * rng.uniform(0.88, 1.12) + j * rng.exponential(1.0)) / r
    return np.array(out if out else [rng.uniform(0.0, 0.5 * dur)])      # never an empty take


def _drop(rng, size: float, nb: float = HF_DROP_BUBBLES, secondary: float = 0.5, click: float = 0.15,
          level: int = 0) -> np.ndarray:
    """One drop falling into water, 0.9 s buffer.

    HF Table 1: 14 bubbles for 3 drops (4-5 each), 500 Hz - 4 kHz, i.e. radii 6.6 - 0.82 mm.
    Largest bubble first (Farnell: big bubbles arrive first). Impact: Farnell's parabolic pulse of
    width 0.1 + 12 U ms, kept quiet (Moss / HF: the impact is far quieter than the bubbles).
    Secondary drops: "each falling drop of sufficient size in turn causes following smaller ones"
    (Sounding Object, g05 section 5); # UNSOURCED: Poisson count, 12 ms mean bubble spacing, the
    size ratio 0.5 and the 0.12-0.35 s rebound delay.
    """
    n = _n(0.9)
    k = int(np.clip(rng.poisson(nb), 1, 9))
    rm = (0.9 + 3.6 * min(max(size, 0.0), 1.0)) * 1e-3
    r = np.sort(np.clip(rng.normal(rm, 0.35 * rm, k), R_4K * 0.7, R_500 * 1.1))[::-1].copy()
    w = max(2, int((0.1 + 12.0 * rng.random()) * 1e-3 * SR))
    starts = (w + np.cumsum(rng.exponential(0.012 * SR, k))).astype(np.int64)
    out = _render_bubbles(n, rng, starts, r, depth=(0.002, 0.012))
    x = np.linspace(0.0, 1.0, w)
    out[:w] += click * float(np.max(np.abs(out))) * (1.0 - 4.0 * (x - 0.5) ** 2)
    if level < 2 and rng.random() < secondary:
        _place(out, _drop(rng, size * 0.5, 1.5, secondary * 0.5, click, level + 1),
               rng.uniform(0.12, 0.35) * SR, 0.5)
    return out


@sfx("drip", "One water drop into a pool: quiet impact, then 4-5 ringing air bubbles (Harmonic Fluids).",
     size=(0.4, "0..1 drop size: mean bubble radius 0.9..4.5 mm (3.6 kHz..730 Hz)"),
     bubbles=(4.7, "mean bubbles per drop (measured: 4-5)"),
     secondary=(0.5, "0..1 chance of smaller rebound drops"),
     click=(0.15, "0..1 level of the impact pulse"), seed=(0, "variation"))
def drip(sr=DEFAULT_SR, size=0.4, bubbles=4.7, secondary=0.5, click=0.15, seed=0):
    y = _drop(_rng(seed, 1), float(size), float(bubbles), float(secondary), float(click))
    return _fin(y, sr, 0.7, 0.0005, 0.05)


@sfx("drops", "A dripping source: drops at a rate, nearly periodic (tap) to random (melt, leak).",
     dur=(4.0, "s"), rate=(2.0, "drops per second at the start, 0.05..40"),
     rate_end=(-1.0, "drops per second at the end (-1 = same): rising = thaw, falling = tap closing"),
     jitter=(0.15, "0 periodic .. 1 Poisson"), size=(0.4, "0..1 drop size"), seed=(0, "variation"))
def drops(sr=DEFAULT_SR, dur=4.0, rate=2.0, rate_end=-1.0, jitter=0.15, size=0.4, seed=0):
    rng = _rng(seed, 2)
    n = _n(dur)
    rate = float(np.clip(rate, 0.05, 40.0))
    times = drop_times(float(dur), rng, rate, None if rate_end < 0 else float(np.clip(rate_end, 0.05, 40.0)),
                       jitter)
    y = np.zeros(n)
    for t in times:
        _place(y, _drop(rng, float(np.clip(size * rng.uniform(0.8, 1.2), 0, 1))), t * SR,
               rng.uniform(0.6, 1.0))
    return _fin(y, sr, 0.7, 0.0005, 0.03)


@sfx("splash", "Object falling into water: slap, cavity collapse, bubble cloud, crown droplets falling back.",
     size=(0.5, "0..1 object size (cavity bubble 6..20 mm, bubble count)"), dur=(1.5, "s"),
     droplets=(6, "crown droplets that fall back"), seed=(0, "variation"))
def splash(sr=DEFAULT_SR, size=0.5, dur=1.5, droplets=6, seed=0):
    """HF Table 1 splash: 127 bubbles in 1.5 s "concentrated in the event", 300 Hz - 6 kHz.
    Cavity: Moss tube bubbles (``ambience.tube_bubble``). Crown droplets return after a ballistic
    flight t = 2 sqrt(2 h / g) from heights 5-50 cm. # UNSOURCED: 0.25 s bubble-rate decay, the
    cavity pinch-off delay 60-140 ms, heights, and the layer balance."""
    rng = _rng(seed, 3)
    size = float(np.clip(size, 0.0, 1.0))
    n = _n(dur)
    t = np.arange(n) / SR
    env = np.exp(-t / 0.25)
    count = HF_SPLASH_COUNT * (0.4 + 1.2 * size)
    y = _unit(bubble_cloud(dur, rng, env * count / max(env.sum() / SR, 1e-9), R_6K, R_300, "gauss",
                           cluster=0.4, depth=(0.002, 0.06)))
    slap = _lp(rng.uniform(-1, 1, n) * np.exp(-t / (0.004 + 0.012 * size)), 1500 + 2500 * (1 - size), 2)
    wash = _bp(rng.uniform(-1, 1, n), 1800.0, 0.6) * np.clip(t / 0.02, 0, 1) * np.exp(-t / 0.18)
    y = y + 0.5 * _unit(slap) * 0.5 + 0.35 * _unit(wash)
    cav = np.zeros(n)
    for _ in range(1 + int(size > 0.5)):
        _place(cav, amb.tube_bubble((6.0 + 14.0 * size) * 1e-3 * rng.uniform(0.7, 1.1), rng),
               rng.uniform(0.06, 0.14) * SR)
    y = y + 0.9 * _unit(cav) * math.sqrt(0.25 + size)
    for _ in range(int(np.clip(droplets, 0, 40))):
        _place(y, _drop(rng, 0.25 * rng.random(), 2.0, 0.2), 2.0 * math.sqrt(2.0 * rng.uniform(0.05, 0.5) / G_GRAV) * SR,
               float(np.max(np.abs(y))) * rng.uniform(0.15, 0.5))
    return _fin(y, sr, 0.75, 0.0005, 0.05)


# ---------------------------------------------------------------------------------------------
# vessels
# ---------------------------------------------------------------------------------------------

VESSEL_Q = dict(glass=30.0, metal=45.0, ceramic=22.0, plastic=12.0, paper=6.0)
# glass 30 is Farnell's value (findings 02 section 9: "resonance about 30").
# UNSOURCED: the other materials (ordered by wall stiffness / loss).


def vessel_resonance(fill, height: float = 0.15, radius: float = 0.035, neck: float = 0.0,
                     neck_len: float = 0.05, c: float = 343.0):
    """Air resonance (Hz) of a vessel filled to ``fill`` (0..1 of its height).

    Open cylinder (neck = 0): quarter-wave tube, f = c / (4 (L_air + 0.61 a)) (Farnell Practical 14:
    "the empty cavity above the liquid is a quarter-wave (semi-open) tube"; the 0.61 a unflanged end
    correction is textbook, # UNSOURCED in the findings). Odd harmonics 1, 3, 5, 7.
    Bottle (0 < neck < radius): Helmholtz, f = (c / 2 pi) sqrt(A / (V L')), A = pi neck^2,
    V = pi radius^2 L_air (findings 01 section 3.5), L' = neck_len + 1.7 neck (textbook end
    correction, # UNSOURCED in the findings). Both rise monotonically as the vessel fills.
    """
    L = height * (1.0 - np.clip(np.asarray(fill, float), 0.0, 1.0))
    if neck <= 0.0 or neck >= radius:
        return c / (4.0 * (L + 0.61 * radius))
    # + 2 mm of air: a brim-full bottle keeps a finite cavity (guard against V -> 0)
    return c / (2.0 * np.pi) * np.sqrt(neck * neck / (radius * radius * (L + 0.002) * (neck_len + 1.7 * neck)))


def _vessel(src, f0, q, harmonics=(1, 3, 5, 7)):
    """Farnell's vessel: parallel band-passes at the odd harmonics of the cavity resonance."""
    out = np.zeros(len(src))
    for k in harmonics:
        if float(np.max(f0)) * k < 0.45 * SR:
            out += _tvbp(src, np.asarray(f0) * k, q) / k
    return out


@sfx("pour", "Pouring liquid into a vessel: bubbles + plunging stream through an air resonance that rises as it fills.",
     dur=(4.0, "s"), flow=(1.0, "flow rate, 0.1..3 (1 = 1580 bubbles/s, a tap)"),
     fill_start=(0.05, "fill level at the start, 0..1"), fill_end=(0.9, "fill level at the end, 0..1"),
     height=(0.15, "vessel inner height, m"), radius=(0.035, "vessel inner radius, m"),
     neck=(0.0, "neck radius in m; 0 = open cup (quarter-wave), >0 = bottle (Helmholtz)"),
     neck_len=(0.05, "neck length, m"), material=("glass", "glass|metal|ceramic|plastic|paper (resonance sharpness)"),
     resonance=(0.6, "0..1 how readable the fill pitch is"), seed=(0, "variation"))
def pour(sr=DEFAULT_SR, dur=4.0, flow=1.0, fill_start=0.05, fill_end=0.9, height=0.15, radius=0.035,
         neck=0.0, neck_len=0.05, material="glass", resonance=0.6, seed=0):
    """Farnell Practical 14 with HF bubble numbers. Bubble rate 1580/s x flow, 300 Hz - 6 kHz (HF
    Table 1 pouring). Bubble size falls as the column height falls (Farnell); # UNSOURCED: the
    factor 1.3 -> 0.6, the stream:bubble balance 0.3:0.7 and the flow ramp min(0.8 s, 0.2 dur)
    (Farnell: 800 ms in a 9 s gesture). The resonance is deliberately readable (Sounding Object)."""
    rng = _rng(seed, 4)
    n = _n(dur)
    t = np.arange(n) / SR
    flow = float(np.clip(flow, 0.1, 3.0))
    fill = np.clip(fill_start + (fill_end - fill_start) * t / (n / SR), 0.0, 1.0)
    env = _ar(n, min(0.8, 0.2 * dur), min(0.8, 0.2 * dur))
    bub = bubble_cloud(dur, rng, HF_POUR_RATE * flow * env, R_6K, R_300, "gauss", cluster=0.2,
                       r_scale=1.3 - 0.7 * fill)
    jet = sum(_fit(amb.farnell_water_voice(n / SR, rng, centre=600.0 + 300.0 * v, slew_ms=1.0), n)
              for v in range(3))
    src = 0.7 * _unit(bub) + 0.3 * _unit(jet) * env
    f0 = vessel_resonance(fill, float(height), float(radius), float(neck), float(neck_len))
    q = VESSEL_Q.get(str(material), 30.0)
    bottle = 0.0 < neck < radius
    res = _vessel(src, f0, q, (1,) if bottle else (1, 3, 5, 7))
    amt = float(np.clip(resonance, 0.0, 1.0))
    return _fin((1.0 - 0.6 * amt) * src + amt * math.sqrt(q) * res, sr, 0.7, 0.01, 0.05)


@sfx("babble", "Babbling brook: thousands of chirping air bubbles a second over a small step.",
     dur=(4.0, "s"), flow=(1.0, "flow, 0.05..3 (1 = 3100 bubbles/s)"),
     size=(1.0, "bubble size scale, 0.4..2 (larger = deeper, slower water)"),
     turbulence=(0.5, "0..1 slow surging of the bubble rate"), seed=(0, "variation"))
def babble(sr=DEFAULT_SR, dur=4.0, flow=1.0, size=1.0, turbulence=0.5, seed=0):
    """HF Table 1 "water step": 26657 bubbles in 8.6 s (3100/s), 300 Hz - 5 kHz. Two-band radii
    (Sounding Object). The chirp is what makes it babble (HF: constant-frequency bubbles "sound
    more like computer-generated noise"). # UNSOURCED: the 1.5 Hz surge and cluster 0.3."""
    rng = _rng(seed, 5)
    n = _n(dur)
    s = float(np.clip(size, 0.4, 2.0))
    lam = HF_BABBLE_RATE * float(np.clip(flow, 0.05, 3.0)) * np.clip(
        1.0 + 0.45 * float(np.clip(turbulence, 0, 1)) * _lfn(rng, n, 1.5), 0.15, 3.0)
    y = bubble_cloud(dur, rng, lam, R_5K * s, R_300 * s, "two_band", cluster=0.3, depth=(0.002, 0.025))
    return _fin(y, sr, 0.7, 0.02, 0.05)


@sfx("bubbles", "Air bubbles in water with exact radius-dependent pitch and decay; heard from above or underwater.",
     dur=(2.0, "s"), rate=(12.0, "bubbles per second, 0.2..4000"),
     size=(2.0, "mean bubble radius in mm, 0.3..12 (1 mm = 3.3 kHz, 5 mm = 660 Hz)"),
     spread=(0.4, "0..1 radius spread"), cluster=(0.5, "0..1 bubbles arrive in bursts"),
     underwater=(0.0, "0 = heard at the surface (pitch rises as they surface); 1 = deep, muffled, steady pitch"),
     rise=(1.0, "rise-speed multiplier: faster = stronger 'bloop' chirp"), seed=(0, "variation"))
def bubbles(sr=DEFAULT_SR, dur=2.0, rate=12.0, size=2.0, spread=0.4, cluster=0.5, underwater=0.0, rise=1.0, seed=0):
    rng = _rng(seed, 6)
    r = float(np.clip(size, 0.3, 12.0)) * 1e-3
    sp = float(np.clip(spread, 0.0, 1.0))
    deep = float(underwater) >= 0.5
    y = bubble_cloud(dur, rng, float(np.clip(rate, 0.2, 4000.0)), r / (1 + 2 * sp), r * (1 + 2 * sp), "gauss",
                     cluster=float(cluster), depth=(0.5, 2.0) if deep else (0.002, 0.03), rise=float(rise))
    if not np.any(y):
        y = _fit(hf_bubble(r, 1.0 if deep else 0.01), _n(dur))
    if deep:
        y = _lp(y, 1800.0, 2)                     # UNSOURCED: underwater listening low-pass
    return _fin(y, sr, 0.7, 0.0005, 0.03)


@sfx("boil", "Boiling water: kettle rumble of collapsing vapour bubbles, then large bubbles bursting in a rolling boil.",
     dur=(4.0, "s"), intensity=(0.8, "0..1: 0.2 simmer rumble, 0.6 onset, 1 rolling boil"),
     ramp=(0.0, "1 = heat up from cold to `intensity` over the duration"),
     size=(0.5, "0..1 vessel size (kettle .. cauldron): lower rumble, larger bubbles"), seed=(0, "variation"))
def boil(sr=DEFAULT_SR, dur=4.0, intensity=0.8, ramp=0.0, size=0.5, seed=0):
    """Bubbles: Farnell Practical 12 bursts (cluster) of HF bubbles. # UNSOURCED (no findings file
    gives boiling numbers): the rumble law sin(pi i / 0.9) (a kettle is loudest before it boils and
    quietens at the boil), the rolling-boil rate 160 ((i - 0.4) / 0.6)^2 per second, radii 2-10 mm,
    and the vessel band 900 -> 250 Hz."""
    rng = _rng(seed, 7)
    n = _n(dur)
    t = np.arange(n) / SR
    size = float(np.clip(size, 0.0, 1.0))
    i = float(np.clip(intensity, 0.0, 1.0)) * (t / t[-1] if ramp >= 0.5 else np.ones(n))
    rumble = _bp(rng.uniform(-1, 1, n), 900.0 - 650.0 * size, 1.2) * (0.6 + 0.4 * np.abs(_lfn(rng, n, 9.0)))
    rumble = _unit(rumble) * (0.02 + np.sin(np.pi * np.clip(i / 0.9, 0.0, 1.0)))
    rate = 160.0 * np.clip((i - 0.4) / 0.6, 0.0, 1.0) ** 2
    s = 1.0 + size
    bub = bubble_cloud(dur, rng, rate, 2e-3 * s, 10e-3 * s, "gauss", cluster=0.6, depth=(0.001, 0.01))
    peak = float(np.max(np.abs(bub)))
    y = 0.35 * rumble + (2.5 * bub / peak if peak > 0 else 0.0)
    return _fin(y, sr, 0.7, 0.02, 0.05)


def _hiss(n, rng):
    """Farnell fire hiss (findings 02 section 6): noise x (slow noise)^4 x 10, high-pass 1 kHz.
    Pre-gain 1 / (3 sigma) as in ``physical.ambience.fire``."""
    return _hp(rng.uniform(-1, 1, n) * np.clip(_lfn(rng, n, 1.0) / 3.0, -1, 1) ** 4 * 10.0, 1000.0)


def _crackles(n, rng, rate, wood: float = 0.5, sigma: float = 0.7, return_count: bool = False):
    """Poisson crackles with log-normal sizes.

    Each crackle is Farnell's (findings 02 section 6): noise x (linear decay over U(0, 30) ms)^2
    through a Q 1 band-pass at 100-1000 Hz ("good for burning wood", fraction ``wood``) or
    1.5-16.5 kHz. # UNSOURCED: Poisson timing and log-normal amplitude (sigma) replace Farnell's
    level-window trigger, as the assignment specifies.
    """
    times = bubble_times(n, rng, rate)
    out = np.zeros(n)
    for s in times:
        T = max(8, int(rng.random() * 0.030 * SR))
        seg = np.concatenate([rng.uniform(-1, 1, T) * (1.0 - np.arange(T) / T) ** 2, np.zeros(int(0.012 * SR))])
        fc = 100.0 + 900.0 * rng.random() if rng.random() < wood else 1500.0 + 500.0 * rng.integers(0, 31)
        _place(out, _bp(seg, fc, 1.0), s, rng.lognormal(0.0, sigma))
    return (out, len(times)) if return_count else out


@sfx("sizzle", "Water or fat on a hot pan: dense micro-explosions, spitting pops and evaporation hiss.",
     dur=(3.0, "s"), intensity=(0.7, "0..1 density of micro-explosions (about 4000/s at 1)"),
     spit=(6.0, "larger pops per second"), decay=(0.0, "s for it to boil away to 1/e (0 = steady)"),
     hiss=(0.4, "0..1 level of the steady evaporation hiss"),
     bright=(0.5, "0..1 top end: low-pass 3.5..12 kHz"), seed=(0, "variation"))
def sizzle(sr=DEFAULT_SR, dur=3.0, intensity=0.7, spit=6.0, decay=0.0, hiss=0.4, bright=0.5, seed=0):
    """# UNSOURCED model (the findings name sizzling only as a fire component): Poisson impulses with
    log-normal amplitudes ringing a 2.5-7 kHz band (each a sub-millisecond vapour burst), plus
    Farnell crackles for the spits and Farnell's 4th-power hiss."""
    rng = _rng(seed, 8)
    n = _n(dur)
    t = np.arange(n) / SR
    env = np.exp(-t / decay) if decay > 0 else np.ones(n)
    k = int(rng.poisson(4000.0 * float(np.clip(intensity, 0.01, 1.0)) * n / SR))
    imp = np.zeros(n)
    np.add.at(imp, rng.integers(0, n, k), rng.lognormal(0.0, 0.9, k) * rng.choice([-1.0, 1.0], k))
    y = _unit(_bp(imp, 2500.0, 0.8) + _bp(imp, 6500.0, 1.0)) * env
    y = y + 0.6 * _unit(_crackles(n, rng, float(np.clip(spit, 0.0, 60.0)) * env + 1e-6, wood=0.3, sigma=0.5))
    y = y + float(np.clip(hiss, 0, 1)) * _unit(_hiss(n, rng)) * (0.3 + 0.7 * env) * 0.6
    return _fin(_lp(y, 3500.0 + 8500.0 * float(np.clip(bright, 0, 1)), 2), sr, 0.6, 0.005, 0.05)


@sfx("fizz", "Carbonation: tiny bubbles bursting at the surface, at a rate that dies away.",
     dur=(3.0, "s"), rate=(900.0, "bubbles per second at the start, 20..5000"),
     half_life=(1.5, "s for the rate to halve (0 = steady)"),
     size=(0.7, "mean bubble radius in mm, 0.3..1.2 (0.7 mm = 4.7 kHz, 0.5 mm = 6.6 kHz)"),
     bright=(0.35, "0..1 top end: low-pass 4..14 kHz"), seed=(0, "variation"))
def fizz(sr=DEFAULT_SR, dur=3.0, rate=900.0, half_life=1.5, size=0.7, bright=0.35, seed=0):
    """HF bubbles of 0.6-1.8 x ``size`` at the surface (T60 from the damping law: 8 ms at 0.5 mm,
    19 ms at 1 mm), Poisson with
    rate(t) = rate 2^(-t / half_life). # UNSOURCED: the rate and half-life defaults (HF lists fizz
    as not modelled); the pitch and decay of each bubble are the sourced law."""
    rng = _rng(seed, 9)
    n = _n(dur)
    t = np.arange(n) / SR
    lam = float(np.clip(rate, 20.0, 5000.0)) * (0.5 ** (t / half_life) if half_life > 0 else 1.0)
    r = float(np.clip(size, 0.3, 1.2)) * 1e-3
    y, info = bubble_cloud(dur, rng, lam, 0.6 * r, 1.8 * r, "gauss", depth=(0.0005, 0.004), return_info=True)
    if info["n"] == 0:
        y = _fit(hf_bubble(r, 0.002), n)
    return _fin(_lp(y, 4000.0 + 10000.0 * float(np.clip(bright, 0, 1)), 2), sr, 0.5, 0.002, 0.05)


@sfx("gurgle", "Glugging: large wobbling air bubbles gulped at a rate (bottle emptying, drain, swamp).",
     dur=(2.5, "s"), rate=(3.5, "glugs per second, 0.3..12"),
     size=(0.5, "0..1 bubble size: 6..20 mm radius (550..165 Hz)"), seed=(0, "variation"))
def gurgle(sr=DEFAULT_SR, dur=2.5, rate=3.5, size=0.5, seed=0):
    """Each glug: a Moss non-spherical "tube" bubble (``ambience.tube_bubble``, findings 06 section
    3.2: shape modes radiating at 2 f_n give the low, full body) followed by a burst of small HF
    bubbles. Timing is quantised flow (a relaxation oscillator, findings 01 section 3.8).
    # UNSOURCED: 8 trailing bubbles per glug and the 12 % timing jitter."""
    rng = _rng(seed, 10)
    n = _n(dur)
    size = float(np.clip(size, 0.0, 1.0))
    y = np.zeros(n)
    lam = np.zeros(n)
    for t in drop_times(float(dur), rng, float(np.clip(rate, 0.3, 12.0)), jitter=0.12):
        _place(y, amb.tube_bubble((6.0 + 14.0 * size) * 1e-3 * rng.uniform(0.8, 1.2), rng), t * SR,
               rng.uniform(0.6, 1.0))
        i = int(t * SR)
        lam[i:i + int(0.08 * SR)] += 8.0 / 0.08
    small = bubble_cloud(dur, rng, lam, 1e-3, 5e-3, "gauss", cluster=0.5)
    y = _unit(y) + 0.5 * _unit(small)
    return _fin(y, sr, 0.75, 0.001, 0.05)


@sfx("drain", "Water draining away: running bubbles, a sucking vortex whose pitch climbs, final glugs.",
     dur=(4.0, "s"), size=(0.5, "0..1 basin size (sink .. bath): lower suction pitch"), seed=(0, "variation"))
def drain(sr=DEFAULT_SR, dur=4.0, size=0.5, seed=0):
    """# UNSOURCED gesture (no findings file models a drain vortex): bubble flow at the HF pouring
    rate falling away, an air-core suction band (noise through a resonance sweeping
    (250 -> 1400 Hz) / (0.6 + 0.8 size), fluttered at 8 -> 22 Hz) growing over the last 45 %, and
    Moss tube-bubble glugs at the end."""
    rng = _rng(seed, 11)
    n = _n(dur)
    x = np.arange(n) / n
    size = float(np.clip(size, 0.0, 1.0))
    flow = bubble_cloud(dur, rng, HF_POUR_RATE * 0.6 * (1.0 - x) ** 1.5, R_6K, R_300, "two_band", cluster=0.3)
    grow = np.clip((x - 0.55) / 0.35, 0.0, 1.0) ** 2 * np.clip((1.0 - x) / 0.06, 0.0, 1.0)
    flut = 0.5 + 0.5 * np.sin(2 * np.pi * np.cumsum(8.0 + 14.0 * x) / SR + 0.8 * _lfn(rng, n, 6.0))
    suck = _tvbp(rng.uniform(-1, 1, n), (250.0 + 1150.0 * x ** 2) / (0.6 + 0.8 * size), 4.0 + 10.0 * x)
    y = _unit(flow) + 1.4 * _unit(suck) * grow * flut
    g = np.zeros(n)
    for t in (0.82, 0.9, 0.96):
        _place(g, amb.tube_bubble((8.0 + 10.0 * size) * 1e-3 * rng.uniform(0.8, 1.2), rng),
               (t + rng.uniform(-0.02, 0.02)) * n)
    return _fin(y + 1.2 * _unit(g), sr, 0.7, 0.01, 0.05)


@sfx("surf", "Waves breaking on a shore (Moss bubble populations following each crest).",
     dur=(12.0, "s"), distance=(120.0, "m to the break line, 10..500"),
     period_min=(8.0, "shortest gap between waves, s"), period_max=(13.0, "longest gap between waves, s"),
     bubble_rate=(3000.0, "bubbles per second at a crest"), seed=(0, "variation"))
def surf(sr=DEFAULT_SR, dur=12.0, distance=120.0, period_min=8.0, period_max=13.0, bubble_rate=3000.0, seed=0):
    """``physical.ambience.surf`` (findings g16 Table 1 row 2). Shorter than 10 s renders one wave."""
    pmin = max(float(period_min), 1.0)
    y = amb.surf(max(float(dur), 4.0), _rng(seed, 12), distance=float(np.clip(distance, 10.0, 500.0)),
                 period=(pmin, max(float(period_max), pmin + 0.1)),
                 bubble_rate=float(np.clip(bubble_rate, 100.0, 8000.0)), single=dur < 10.0)
    return _fin(_fit(y, _n(dur)), sr, 0.7, 0.05, 0.3)


@sfx("cave_drips", "Several dripping sources in a reverberant cave.",
     dur=(6.0, "s"), sources=(5, "number of dripping points, 1..12"),
     wet=(0.6, "0..1 cave reverb level"), seed=(0, "variation"))
def cave_drips(sr=DEFAULT_SR, dur=6.0, sources=5, wet=0.6, seed=0):
    """``physical.ambience.cave_drips`` (findings g16 Table 1 row 3). Its drip periods are 1.2-7 s,
    so a drop is added at the start when a short render would otherwise be silent."""
    rng = _rng(seed, 13)
    n = _n(dur)
    y = _fit(amb.cave_drips(n / SR, rng, sources=int(np.clip(sources, 1, 12)), wet=float(np.clip(wet, 0, 1))), n)
    if float(np.max(np.abs(y[: n // 2]))) < 1e-6:
        _place(y, P.drip(rng), 0.05 * SR, max(float(np.max(np.abs(y))), 0.1))
    return _fin(y, sr, 0.7, 0.002, 0.1)


@sfx("waterfall", "Waterfall: near-white roar of countless bubbles with a low rumble that grows with height.",
     dur=(4.0, "s"), height=(6.0, "fall height in m, 0.5..60 (rumble and roar)"),
     distance=(8.0, "m, 1..200: farther = duller, less bubble detail"), seed=(0, "variation"))
def waterfall(sr=DEFAULT_SR, dur=4.0, height=6.0, distance=8.0, seed=0):
    """Farnell (findings 02 section 8): the extremes of water are "a stormy sea or waterfall (about
    white noise) and a single bubble"; at a distance the detail is replaced by noise (findings 01
    section 4.3). Detail layer: HF bubbles at the babbling rate with Moss's breaking-wave power law.
    # UNSOURCED: the height and distance mappings and the layer balance."""
    rng = _rng(seed, 14)
    n = _n(dur)
    h = float(np.clip(height, 0.5, 60.0))
    d = float(np.clip(distance, 1.0, 200.0))
    near = 1.0 / (1.0 + d / 10.0)
    roar = _lp(_hp(rng.uniform(-1, 1, n), 150.0), 9000.0 / (1.0 + d / 60.0), 2) * (1.0 + 0.15 * _lfn(rng, n, 3.0))
    rumble = _lp(rng.uniform(-1, 1, n), 60.0 + 600.0 / h, 2) * (1.0 + 0.3 * _lfn(rng, n, 1.0))
    bub = bubble_cloud(dur, rng, HF_BABBLE_RATE, R_6K, R_300 * min(2.0, 1.0 + h / 30.0), "power", alpha=2.2,
                       cluster=0.2)
    y = _unit(roar) + min(1.2, 0.25 * math.sqrt(h)) * _unit(rumble) + 1.6 * near * _unit(bub)
    return _fin(y, sr, 0.7, 0.05, 0.1)


@sfx("underwater_ambience", "Underwater bed: muffled low rumble with slow swell and passing deep bubbles.",
     dur=(6.0, "s"), depth=(5.0, "m, 0.5..100: deeper = darker and quieter bubbles"),
     bubbles=(4.0, "deep bubbles per second, 0..200"), seed=(0, "variation"))
def underwater_ambience(sr=DEFAULT_SR, dur=6.0, depth=5.0, bubbles=4.0, seed=0):
    """Bubbles are HF bubbles far from the surface (steady pitch, radius-law decay).
    # UNSOURCED: the rumble band 40 + 400 / (1 + depth / 5) Hz and the swell rate."""
    rng = _rng(seed, 15)
    n = _n(dur)
    dp = float(np.clip(depth, 0.5, 100.0))
    fc = 40.0 + 400.0 / (1.0 + dp / 5.0)
    rum = _lp(rng.uniform(-1, 1, n), fc, 2) * (1.0 + 0.35 * _lfn(rng, n, 0.4))
    y = _unit(rum)
    if bubbles > 0:
        b = bubble_cloud(dur, rng, float(np.clip(bubbles, 0.0, 200.0)), 1.5e-3, 9e-3, "gauss", cluster=0.6,
                         depth=(0.5, 2.0))
        pk = float(np.max(np.abs(b)))
        if pk > 0:
            y = y + _lp(b / pk, 2500.0 / (1.0 + dp / 20.0), 2) * 2.0
    return _fin(y, sr, 0.6, 0.1, 0.3)


# =============================================================================================
# GASES - jets, leaks, whistles, wind
# =============================================================================================

STROUHAL = 0.2          # Selfridge: fixing St = 0.2 was rated as authentic (findings 06 section 4.1)
C_AIR = 343.0


def jet_noise(n: int, rng, velocity, diameter: float, power: float = 8.0, u_ref: float = 100.0,
              q: float = 0.7) -> np.ndarray:
    """Turbulent jet / orifice noise at 48 kHz.

    Spectral peak f_p = St U / D with St = 0.2 (Strouhal scaling; findings 06 section 4.1 for the
    aeolian case, findings 02 section 13: "centre frequency proportional to velocity / diameter").
    Shape: one Q 0.7 band-pass, i.e. 6 dB/octave skirts either side of the peak.
    Level: total power ~ (U / u_ref)^power; the amplitude is (U / u_ref)^(power / 2) divided by
    sqrt(f_p / 1 kHz) because a constant-Q band passes power in proportion to its centre.
    # UNSOURCED: the findings give no jet level law. power = 8 is Lighthill's free-jet law (textbook);
    # 6 is the dipole law Selfridge uses for solid obstacles (findings 06: "u^6 intensity").
    # UNSOURCED: the Q 0.7 skirt shape.
    ``velocity`` (m/s) may be a per-sample array; ``diameter`` in metres.
    """
    u = np.broadcast_to(np.asarray(velocity, float), (n,))
    fp = np.clip(STROUHAL * u / max(float(diameter), 1e-5), 20.0, 0.45 * SR)
    return _tvbp(rng.uniform(-1, 1, n), fp, q) * (np.maximum(u, 0.0) / u_ref) ** (power / 2.0) / np.sqrt(fp / 1000.0)


def _velocity(pressure: float) -> float:
    """# UNSOURCED mapping: pressure 0..1 -> exit speed 40..343 m/s (a jet chokes at the speed of sound)."""
    return 40.0 + (C_AIR - 40.0) * float(np.clip(pressure, 0.0, 1.0))


@sfx("steam", "Steam escaping under pressure: hiss whose pitch follows speed / hole size, with wet sputter.",
     dur=(2.5, "s"), pressure=(0.6, "0..1 (exit speed 40..343 m/s)"),
     orifice=(4.0, "hole diameter in mm, 0.5..50: larger = lower hiss"),
     wet=(0.4, "0..1 sputtering of water droplets"), tone=(9000.0, "Hz low-pass on the top end"),
     seed=(0, "variation"))
def steam(sr=DEFAULT_SR, dur=2.5, pressure=0.6, orifice=4.0, wet=0.4, tone=9000.0, seed=0):
    """``jet_noise`` + Farnell's 4th-power hiss modulator and crackle bursts for the wet sputter
    (outgassing of water vapour hisses, findings 09 D2.2). # UNSOURCED: sputter amounts."""
    rng = _rng(seed, 20)
    n = _n(dur)
    wet = float(np.clip(wet, 0.0, 1.0))
    u = _velocity(pressure) * (1.0 + 0.06 * _lfn(rng, n, 12.0))
    y = _unit(jet_noise(n, rng, u, float(np.clip(orifice, 0.5, 50.0)) * 1e-3))
    y = y * (1.0 - 0.6 * wet + 0.6 * wet * np.clip(np.abs(_lfn(rng, n, 7.0)), 0, 2))
    y = y + wet * 0.8 * _unit(_crackles(n, rng, 25.0 * wet + 0.01, wood=0.0)) * 0.5
    return _fin(_lp(y, float(np.clip(tone, 500.0, 20000.0)), 2) * _ar(n, 0.04, 0.2), sr, 0.6, 0.005, 0.05)


@sfx("air_leak", "Compressed air through a small hole: steady leak, or a pneumatic burst that sags as pressure falls.",
     dur=(2.0, "s"), pressure=(0.5, "0..1 (exit speed 40..343 m/s)"),
     orifice=(2.0, "hole diameter in mm, 0.3..30"),
     burst=(0.0, "s: 0 = steady leak; >0 = pneumatic release, pressure decaying with this time constant"),
     whistle=(0.0, "0..1 narrow edge tone at the Strouhal frequency"),
     tone=(10000.0, "Hz low-pass on the top end"), seed=(0, "variation"))
def air_leak(sr=DEFAULT_SR, dur=2.0, pressure=0.5, orifice=2.0, burst=0.0, whistle=0.0, tone=10000.0, seed=0):
    """``jet_noise``. A burst lets the exit speed fall as exp(-t / burst): the hiss drops in pitch
    (f ~ U) and dies as U^4 in amplitude. Whistle: the same noise through a Q 60 resonance at
    St U / D (Farnell's wire-whistle filter, findings 02 section 13)."""
    rng = _rng(seed, 21)
    n = _n(dur)
    t = np.arange(n) / SR
    d = float(np.clip(orifice, 0.3, 30.0)) * 1e-3
    u = _velocity(pressure) * (1.0 + 0.04 * _lfn(rng, n, 15.0))
    if burst > 0:
        u = np.maximum(u * np.exp(-t / float(burst)), 1.0)
    y = jet_noise(n, rng, u, d)
    if whistle > 0:
        fw = np.clip(STROUHAL * u / d, 100.0, 6000.0)
        y = y + float(np.clip(whistle, 0, 1)) * 6.0 * _tvbp(rng.uniform(-1, 1, n), fw, 60.0) * (u / 100.0) ** 4 / np.sqrt(fw / 1000.0)
    y = _lp(y, float(np.clip(tone, 500.0, 20000.0)), 2)
    return _fin(y * (1.0 if burst > 0 else _ar(n, 0.03, 0.1)), sr, 0.6, 0.001, 0.04)


@sfx("spray", "Aerosol spray can: valve tick, fine high hiss with a little can resonance, valve closing.",
     dur=(0.8, "s"), pressure=(0.7, "0..1"), nozzle=(0.5, "nozzle diameter in mm, 0.2..3"),
     can=(900.0, "Hz hollow can resonance"), tone=(9000.0, "Hz low-pass on the top end"),
     seed=(0, "variation"))
def spray(sr=DEFAULT_SR, dur=0.8, pressure=0.7, nozzle=0.5, can=900.0, tone=9000.0, seed=0):
    """``jet_noise`` at nozzle scale (f_p = St U / D lies above the audio band for a 0.5 mm nozzle,
    so the audible part is the rising skirt: a thin hiss). # UNSOURCED: the valve ticks, the can
    resonance level and the 3 kHz atomisation layer."""
    rng = _rng(seed, 22)
    n = _n(dur)
    env = _ar(n, 0.012, 0.03)
    y = _unit(jet_noise(n, rng, _velocity(pressure), float(np.clip(nozzle, 0.2, 3.0)) * 1e-3))
    y = y + 0.5 * _unit(_hp(rng.uniform(-1, 1, n), 3000.0, 2)) * (0.8 + 0.2 * np.abs(_lfn(rng, n, 40.0)))
    y = (y + 0.25 * _unit(_bp(y, float(np.clip(can, 200.0, 4000.0)), 4.0))) * env
    y = _lp(y, float(np.clip(tone, 500.0, 20000.0)), 2)
    tick = _bp(np.concatenate([[1.0], np.zeros(int(0.01 * SR))]), 2200.0, 3.0)
    _place(y, tick, 0, 6.0)
    _place(y, tick, n - int(0.03 * SR), 4.0)
    return _fin(y, sr, 0.55, 0.0005, 0.01)


@sfx("kettle", "Whistling kettle: steam jet whose edge tone locks onto the whistle cavity as pressure builds.",
     dur=(5.0, "s"), tone=(1800.0, "Hz whistle cavity resonance, 600..4000"),
     onset=(2.5, "s for the steam pressure to build"), breath=(0.4, "0..1 steam noise in the whistle"),
     rumble=(0.3, "0..1 boiling water underneath"), seed=(0, "variation"))
def kettle(sr=DEFAULT_SR, dur=5.0, tone=1800.0, onset=2.5, breath=0.4, rumble=0.3, seed=0):
    """Edge / hole tone f_e = St U / gap rising with jet speed, pulled onto the nearest cavity mode
    (f_res, 2 f_res) when within 30 %; inside the lock the pitch moves only 4 % of the detuning and
    the amplitude follows cos^2 of it. Below the lock there is breath noise and a weak unstable tone.
    # UNSOURCED: no findings file has a kettle or edge-tone model beyond "centre frequency
    proportional to velocity / diameter" (findings 02 section 13) and the jet-drive remark in
    findings 08; the lock range, pull, gap (chosen so the free tone reaches 1.25 f_res) and harmonic
    levels 1 : 0.25 : 0.08 are assumptions."""
    rng = _rng(seed, 23)
    n = _n(dur)
    t = np.arange(n) / SR
    fres = float(np.clip(tone, 600.0, 4000.0))
    x = np.clip(t / max(float(onset), 0.05), 0.0, 1.0)
    u = 60.0 * (x * x * (3 - 2 * x)) * (1.0 + 0.03 * _lfn(rng, n, 8.0)) + 0.5
    gap = STROUHAL * 60.0 / (1.25 * fres)
    fe = STROUHAL * u / gap
    mode = np.where(fe > 1.5 * fres, 2.0 * fres, fres)
    det = fe / mode - 1.0
    lock = np.abs(det) < 0.3
    f = np.where(lock, mode * (1.0 + 0.04 * det), fe) * (1.0 + 0.003 * _lfn(rng, n, 30.0))
    a = _lp(np.where(lock, np.cos(np.pi * det / 0.6) ** 2, 0.05) * (u / 60.0) ** 2, 12.0, 2)
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = a * (np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph))
    y = y + float(np.clip(breath, 0, 1)) * 2.5 * _tvbp(rng.uniform(-1, 1, n), np.clip(f, 100, 8000), 3.0) * (u / 60.0) ** 2
    if rumble > 0:
        r = _bp(rng.uniform(-1, 1, n), 500.0, 1.2) * (0.6 + 0.4 * np.abs(_lfn(rng, n, 9.0)))
        y = y + float(np.clip(rumble, 0, 1)) * 0.5 * _unit(r) * float(np.max(np.abs(y)) + 1e-9)
    return _fin(y, sr, 0.5, 0.02, 0.15)


@sfx("balloon", "Balloon letting its air out: flapping neck raspberry over escaping air, pitch sagging as it empties.",
     dur=(2.0, "s"), size=(0.5, "0..1 balloon size: lower, slower flapping"),
     flutter=(0.7, "0 = plain hiss .. 1 = full flapping raspberry"), seed=(0, "variation"))
def balloon(sr=DEFAULT_SR, dur=2.0, size=0.5, flutter=0.7, seed=0):
    """The neck is a relaxation oscillator (findings 01 section 3.8 lists "balloon neck"): pressure
    opens it, flow drops the pressure, it shuts. Each opening gates the escaping jet
    (``jet_noise``); the shrinking cavity is a Helmholtz resonance that rises as volume falls
    (findings 01 section 3.5). # UNSOURCED: flap rate (40-300 Hz), pressure curve, cavity band."""
    rng = _rng(seed, 24)
    n = _n(dur)
    x = np.arange(n) / n
    size = float(np.clip(size, 0.0, 1.0))
    p = np.clip(1.0 - x ** 3, 0.0, 1.0) * np.clip((1.0 - x) / 0.08, 0.0, 1.0) ** 0.5
    f = (300.0 - 200.0 * size) * (0.35 + 0.65 * p) * (1.0 + 0.08 * _lfn(rng, n, 25.0))
    ph = np.cumsum(f) / SR % 1.0
    pulse = np.clip(1.0 - ph / 0.45, 0.0, 1.0) ** 2
    fl = float(np.clip(flutter, 0.0, 1.0))
    jet = jet_noise(n, rng, 20.0 + 60.0 * np.sqrt(p), 0.006) * (1.0 - fl + fl * pulse * 2.5)
    cav = _tvbp(pulse - pulse.mean(), (350.0 + 900.0 * x ** 2) / (0.6 + 0.8 * size), 3.0)
    y = _unit(jet) + 1.5 * fl * _unit(cav) * p
    return _fin(y, sr, 0.65, 0.003, 0.03)


GUST_SOURCES = dict(gap=("howl1", "static"), doorway=("howl1", "howl2"), canyon=("howl2", "static"),
                    wire=("wire1", "wire2"), leaves=("leaves", "static"))


@sfx("gust", "A gust of wind through an opening, across wires or through leaves (Farnell's wind emitters).",
     dur=(4.0, "s"), through=("gap", "gap|wire|leaves|canyon|doorway"),
     strength=(0.5, "0..1 peak wind speed. Howls (gap, doorway, canyon) sound only between about 0.25 and 0.6"),
     tone=(1.0, "pitch scale of the wire whistles (thinner wire = higher), 0.3..3"), seed=(0, "variation"))
def gust(sr=DEFAULT_SR, dur=4.0, through="gap", strength=0.5, tone=1.0, seed=0):
    """``physical.ambience.wind_scene`` with its unexposed sources (findings g16 Table 1 row 1),
    driven by one windspeed hump instead of the scene's random walk. Each emitter keeps Farnell's
    law: static bed linear, wires square-law at 600 + 400 w and 1000 + 1000 w Hz (Q 60), doorway
    howls only inside a speed window, leaves lagging by seconds. The render is pre-rolled by the
    emitter's propagation delay (and 6 s for the leaves' 0.07 Hz lag) and cropped.
    # UNSOURCED: resting speed 0.12, the sin^2 hump and its 2 Hz flutter."""
    rng = _rng(seed, 25)
    srcs = GUST_SOURCES.get(str(through), GUST_SOURCES["gap"])
    lead = amb.WIND_SOURCES[srcs[0]]["delay"] / 1000.0
    pre = 6.0 if "leaves" in srcs else 0.5
    total = pre + float(dur) + lead + 0.1
    N = amb.Dom(total, False).m
    x = np.clip((np.arange(N) / SR - pre) / float(dur), 0.0, 1.0)
    hump = np.sin(np.pi * x) ** 2
    w = np.clip(0.12 + (float(np.clip(strength, 0, 1)) - 0.12) * hump * (1.0 + 0.08 * _lfn(rng, N, 2.0)), 0.0, 1.0)
    y = amb.wind_scene(total, rng, sources=srcs, w=w, wire_scale=float(np.clip(tone, 0.3, 3.0)))
    i = int((pre + (0.0 if "leaves" in srcs else lead)) * SR)
    return _fin(_fit(np.asarray(y)[i:], _n(dur)), sr, 0.6, 0.05, 0.2)


# =============================================================================================
# FIRE - combustion noise (FIRE, findings g07 section 3)
# =============================================================================================

def extend_fire(p, rng, alpha: float = 3.0, fcut: float = 180.0, sig: float = 10.0, lo: float = 165.0,
                hi: float = 195.0) -> np.ndarray:
    """FIRE Algorithm 1 (findings g07 section 3.4, Table 1): spectral bandwidth extension.

    Power-law noise with |N(f)| = f^(-alpha/2) and random phase, high-passed across 165-195 Hz, is
    amplitude-modulated by |p| inside triangular windows of half-width 11.34 ms, blended with the
    low band (F_low 1 -> 0, F_high 0 -> 1 over 165-195 Hz), with a per-window gain beta chosen so
    the power under a Gaussian (180 Hz, sigma 10 Hz) is preserved (positive root of a quadratic).
    alpha: 2.5 (Abugov & Obrezkov propane burner; brighter) ... 3.0 (authors' choice) ... 3.5.
    Stability guard (*adaptation*): beta is a ratio whose denominator is the noise power that
    happens to fall under the Gaussian in one 23 ms window; when that is small by chance the
    printed formula returns a huge beta and a click (measured kurtosis of the extended band up to
    567). beta is therefore capped at 3x its median over the signal.
    """
    wh = int(round(0.01134 * SR))
    n0 = len(p)
    p = np.concatenate([np.asarray(p, float), np.zeros(3 * wh)])
    n = len(p)
    fq = np.fft.rfftfreq(2 * wh, 1.0 / SR)
    fhigh = np.clip((fq - lo) / (hi - lo), 0.0, 1.0)
    flow = 1.0 - fhigh
    g2 = np.exp(-((fq - fcut) / sig) ** 2)
    ff = np.fft.rfftfreq(n, 1.0 / SR)
    mag = np.zeros_like(ff)
    mag[1:] = ff[1:] ** (-alpha / 2.0)
    mag *= np.clip((ff - lo) / (hi - lo), 0.0, 1.0)
    noise = np.fft.irfft(mag * np.exp(1j * rng.uniform(-np.pi, np.pi, len(ff))), n)
    noise /= np.std(noise) + 1e-30
    w = 1.0 - np.abs(np.arange(2 * wh) - wh) / wh
    out = np.zeros(n)
    frames = []
    for s in range(0, n - 2 * wh, wh):
        pw = p[s:s + 2 * wh] * w
        nw = noise[s:s + 2 * wh] * np.abs(pw)
        pf, nf = np.fft.rfft(pw), np.fft.rfft(nw)
        a = np.sum(np.abs(fhigh * nf) ** 2 * g2)
        b = np.sum(np.real(flow * pf * np.conj(fhigh * nf)) * g2)
        c = np.sum((np.abs(flow * pf) ** 2 - np.abs(pf) ** 2) * g2)
        frames.append((s, flow * pf, fhigh * nf, (-b + math.sqrt(max(b * b - a * c, 0.0))) / (a + 1e-30)))
    betas = np.array([fr[3] for fr in frames])
    cap = 3.0 * float(np.median(betas[betas > 0])) if np.any(betas > 0) else 0.0
    for s, lowf, highf, beta in frames:
        out[s:s + 2 * wh] += np.fft.irfft(lowf + min(beta, cap) * highf, 2 * wh)
    return out[:n0]


def combustion_roar(n: int, rng, area=1.0, depth=0.5, f_turb: float = 60.0, alpha: float = 3.0) -> np.ndarray:
    """Flame roar at 48 kHz: p = dI/dt (FIRE eq. 6, 9), 30 Hz high-pass, ``extend_fire``.

    *Adaptation* (findings g07 section 3.8; the paper takes I(t) from a fluid simulation):
    I(t) = area(t) (1 + depth(t) turbulence(t)), turbulence = smooth noise band-limited to
    ``f_turb`` (< 180 Hz). ``area`` and ``depth`` are numbers or per-sample arrays. A steady flame
    is quiet: only changes of I make sound. # UNSOURCED: this surrogate and every value fed to it.
    """
    area = np.broadcast_to(np.asarray(area, float), (n,))
    depth = np.broadcast_to(np.asarray(depth, float), (n,))
    flux = area * (1.0 + depth * _lfn(rng, n, min(float(f_turb), 150.0)))
    return extend_fire(_hp(np.gradient(flux) * SR, 30.0), rng, alpha)


def _lapping(n, rng):
    """Farnell lapping (findings 05 section 8.2): bp 30 Hz Q 5 x100, hip 25 x2, clip +-0.9, x0.6."""
    lap = amb.hip(amb.hip(amb.bp(rng.uniform(-1, 1, n), 30.0, 5.0) * 100.0, 25.0), 25.0)
    return np.clip(lap, -0.9, 0.9) * 0.6


# kind -> area(t) shape, turbulence depth, f_turb Hz, alpha, crackles/s, hiss, lapping, jet (U m/s, D m, level)
# UNSOURCED: every number below (FIRE Table 2 gives no per-flame numbers; "the differences are the
# shape of I(t)"). alpha 2.5 for the gas flames is the propane-burner value of Abugov & Obrezkov.
FLAME_KINDS = dict(
    candle=dict(area="gusts", depth=0.9, f_turb=25.0, alpha=3.0, crackle=0.0, hiss=0.0, lap=0.0, jet=None, level=0.25),
    torch=dict(area="swing", depth=0.8, f_turb=70.0, alpha=3.0, crackle=3.0, hiss=0.1, lap=0.3, jet=None, level=0.6),
    campfire=dict(area="wander", depth=0.5, f_turb=60.0, alpha=3.0, crackle=12.0, hiss=0.3, lap=0.5, jet=None, level=0.6),
    bonfire=dict(area="wander", depth=0.8, f_turb=90.0, alpha=3.0, crackle=40.0, hiss=0.4, lap=0.7, jet=None, level=0.75),
    gas_burner=dict(area="steady", depth=0.25, f_turb=140.0, alpha=2.5, crackle=0.0, hiss=0.0, lap=0.0,
                    jet=(30.0, 0.004, 0.35), level=0.45),
    blowtorch=dict(area="steady", depth=0.4, f_turb=150.0, alpha=2.5, crackle=0.0, hiss=0.0, lap=0.0,
                   jet=(120.0, 0.003, 0.9), level=0.65),
    fireball=dict(area="burst", depth=0.6, f_turb=80.0, alpha=3.0, crackle=0.0, hiss=0.15, lap=0.0, jet=None, level=0.8),
    flamethrower=dict(area="breath", depth=0.7, f_turb=120.0, alpha=2.8, crackle=0.0, hiss=0.1, lap=0.2,
                      jet=(60.0, 0.008, 0.5), level=0.75),
)


def _area(shape: str, n: int, rng, rate: float = 1.0):
    """Flame area / fuel control and a turbulence-depth multiplier (g07 section 3.8 gestures)."""
    t = np.arange(n) / SR
    dur = n / SR
    one = np.ones(n)
    if shape == "gusts":        # candle: mostly silent, ruffled when a draught passes
        g = np.clip(_lfn(rng, n, 0.8 * rate) - 0.4, 0.0, None) ** 2
        return 0.2 * one, g / (g.max() + 1e-9)
    if shape == "swing":        # torch: ruffling follows the speed of the swing
        return one, 0.15 + 0.85 * np.abs(np.sin(np.pi * rate * t + rng.uniform(0, np.pi))) ** 2
    if shape == "wander":
        return np.clip(1.0 + 0.3 * _lfn(rng, n, 0.5), 0.3, 2.0), np.clip(1.0 + 0.5 * _lfn(rng, n, 1.0), 0.2, 2.5)
    if shape == "burst":        # fireball: one fast rise of the area, then decay
        a = np.clip(t / 0.06, 0, 1) ** 2 * np.exp(-np.maximum(t - 0.06, 0) / (0.3 * dur))
        return a, one
    if shape == "breath":       # flame jet: on, held, shut off, burn-off tail of turbulence
        off = max(dur - min(0.7, 0.35 * dur), 0.1)
        a = np.clip(t / 0.08, 0, 1) * np.where(t < off, 1.0, np.exp(-(t - off) / 0.18))
        return a, np.where(t < off, 1.0, 1.8)
    return one, one


def _flame(kind: str, n: int, rng, rate: float = 1.0, crackle=None, wood: float = 0.5):
    k = FLAME_KINDS[kind]
    area, dmod = _area(k["area"], n, rng, rate)
    y = _unit(combustion_roar(n, rng, area, k["depth"] * dmod, k["f_turb"], k["alpha"]))
    amp = np.clip(area / (float(np.max(area)) + 1e-9), 0.0, 1.0)
    cr = k["crackle"] if crackle is None else crackle
    if cr > 0:
        y = y + 0.5 * _unit(_crackles(n, rng, cr * amp ** 2, wood)) * math.sqrt(min(cr, 40.0) / 12.0)
    if k["hiss"] > 0:
        y = y + k["hiss"] * _unit(_hiss(n, rng)) * amp
    if k["lap"] > 0:
        y = y + k["lap"] * _unit(_lapping(n, rng)) * amp
    if k["jet"] is not None:
        u, d, g = k["jet"]
        y = y + g * 2.0 * _unit(jet_noise(n, rng, u * (1.0 + 0.05 * _lfn(rng, n, 20.0)), d)) * amp
    return y


@sfx("flame", "A flame by kind: combustion roar from the changing flame surface, plus crackle, hiss or gas jet.",
     kind=("campfire", "candle|torch|campfire|bonfire|gas_burner|blowtorch|fireball|flamethrower"),
     dur=(3.0, "s"), rate=(1.0, "gesture rate: torch swings per second, candle draughts (x0.8 Hz)"),
     seed=(0, "variation"))
def flame(sr=DEFAULT_SR, kind="campfire", dur=3.0, rate=1.0, seed=0):
    kind = str(kind) if str(kind) in FLAME_KINDS else "campfire"
    n = _n(dur)
    y = _flame(kind, n, _rng(seed, 30), float(np.clip(rate, 0.05, 10.0)))
    one_shot = kind in ("fireball", "flamethrower")
    return _fin(y, sr, FLAME_KINDS[kind]["level"], 0.003 if one_shot else 0.05, 0.1)


@sfx("fire", "Wood fire: combustion roar, Poisson crackles, hiss of outgassing; size, crackle rate, damp wood, wind.",
     dur=(4.0, "s"), size=(0.4, "0..1: 0 small hearth fire, 1 big blaze (1..4 independent flame fronts, more roar)"),
     crackle=(12.0, "crackles per second, 0..150"),
     wet=(0.2, "0..1 damp wood: more hiss, louder and rarer low pops"),
     wind=(0.2, "0..1 wind ruffling the flames (more roar, gusting)"), seed=(0, "variation"))
def fire(sr=DEFAULT_SR, dur=4.0, size=0.4, crackle=12.0, wet=0.2, wind=0.2, seed=0):
    """Per flame front one shared slow control drives roar, crackle rate and hiss together (Farnell:
    one noise per generator "adds coherency"); a big fire is up to 4 fronts with their own noises
    (Farnell: "want incoherence"). Roar: ``combustion_roar`` (alpha 3). Crackles and hiss: Farnell.
    # UNSOURCED: the mappings of size / wet / wind onto levels (damp wood hissing follows findings
    09 D2.2 "outgassing of water vapour (hissing)"; the amounts are assumptions)."""
    rng = _rng(seed, 31)
    n = _n(dur)
    size, wet, wind = (float(np.clip(v, 0.0, 1.0)) for v in (size, wet, wind))
    units = 1 + int(round(3 * size))
    y = np.zeros(n)
    for _ in range(units):
        c = np.clip(1.0 + 0.4 * _lfn(rng, n, 1.0) + wind * 0.8 * np.clip(_lfn(rng, n, 0.5), 0, None) ** 2, 0.2, 3.0)
        roar = _unit(combustion_roar(n, rng, c, 0.35 + 0.3 * size + 0.5 * wind, 50.0 + 60.0 * size, 3.0))
        cr = _crackles(n, rng, float(np.clip(crackle, 0.0, 150.0)) * (1.0 - 0.5 * wet) / units * c ** 2,
                       wood=0.5 + 0.4 * wet, sigma=0.7 + 0.5 * wet)
        hs = _unit(_lp(_hiss(n, rng), 9000.0, 2)) * c
        y += (0.4 + 0.8 * size + 0.4 * wind) * roar + 0.35 * _unit(cr) * math.sqrt(min(crackle, 40.0) / 12.0 + 1e-6) \
            + (0.2 + 0.6 * wet) * hs + 0.3 * _unit(_lapping(n, rng))
    return _fin(y, sr, 0.6 + 0.15 * size, 0.05, 0.1)


@sfx("match", "Striking a match: scrape on the strip, flare of ignition, then a small flame.",
     dur=(1.6, "s"), strike=(0.12, "s length of the scrape"), rough=(0.6, "0..1 grain of the striking strip"),
     seed=(0, "variation"))
def match(sr=DEFAULT_SR, dur=1.6, strike=0.12, rough=0.6, seed=0):
    """Flare = the fireball gesture of ``combustion_roar`` plus Farnell hiss; the small flame is the
    candle. # UNSOURCED: the scrape (band-passed noise gated by a 400-900 Hz grain train; solid
    friction proper belongs to genny's friction module) and the layer timing."""
    rng = _rng(seed, 32)
    n = _n(dur)
    ns = int(float(np.clip(strike, 0.03, 0.5)) * SR)
    y = np.zeros(n)
    grain = (np.cumsum(650.0 * (1.0 + 0.5 * _lfn(rng, ns, 60.0))) / SR % 1.0) < 0.4
    r = float(np.clip(rough, 0.0, 1.0))
    scrape = _bp(rng.uniform(-1, 1, ns), 3200.0, 1.2) * (1.0 - r + r * grain) * np.sin(np.pi * np.arange(ns) / ns) ** 0.5
    _place(y, _unit(scrape), 0, 0.6)
    nf = min(int(0.5 * SR), n - ns)
    if nf > 256:
        tf = np.arange(nf) / SR
        a = np.clip(tf / 0.03, 0, 1) ** 2 * np.exp(-np.maximum(tf - 0.03, 0) / 0.12)
        flare = _unit(combustion_roar(nf, rng, a, 0.7, 90.0, 3.0)) + 0.8 * _unit(_hiss(nf, rng) + _hp(rng.uniform(-1, 1, nf), 2500.0)) * a
        _place(y, flare, int(0.8 * ns), 1.0)
        rest = n - ns - int(0.15 * SR)
        if rest > 2048:
            _place(y, _flame("candle", rest, rng) * _ar(rest, 0.2, 0.1), ns + int(0.15 * SR), 0.15)
    return _fin(y, sr, 0.65, 0.002, 0.05)


# =============================================================================================
# PLASMA / ELECTRICITY (Farnell Practical 16, findings 02 section 11)
# =============================================================================================

def _formant(x, f: float = 1.0):
    """Farnell ``spark6formant`` (findings 03, "mid" layer): parallel bp at 4600 Q5 (x2), 7200 Q5
    (x2), 480 Q7 (x1.2), 720 Q8 (x2.5), then bp 2500 Q0.5. ``f`` scales every centre."""
    y = (2.0 * _bp(x, 4600.0 * f, 5.0) + 2.0 * _bp(x, 7200.0 * f, 5.0) + 1.2 * _bp(x, 480.0 * f, 7.0)
         + 2.5 * _bp(x, 720.0 * f, 8.0))
    return _bp(y, 2500.0 * f, 0.5)


def _snap(rng, t_noise: float, t_sweep: float = 0.010, f_top: float = 7000.0) -> np.ndarray:
    """Farnell snap, before the formant: noise x 0.5 v^4 (v 1 -> 0 over ``t_noise``) plus a sine
    swept 20 + f_top w Hz times w (w 1 -> 0 over ``t_sweep``: 7 kHz -> 20 Hz in 10 ms)."""
    n = max(32, int(t_noise * SR))
    m = min(n, max(8, int(t_sweep * SR)))
    w = np.zeros(n)
    w[:m] = 1.0 - np.arange(m) / m
    v = 1.0 - np.arange(n) / n
    return rng.uniform(-1, 1, n) * 0.5 * v ** 4 + np.sin(2 * np.pi * np.cumsum(20.0 + f_top * w) / SR) * w


@sfx("spark", "Electric spark snap: 7 kHz to 20 Hz sweep with a quartic noise burst through a spark formant.",
     size=(0.3, "0..1: static tick (5 ms) .. heavy contactor snap (120 ms)"),
     tone=(1.0, "formant scale, 0.5..1.5 (lower = bigger, enclosed)"), seed=(0, "variation"))
def spark(sr=DEFAULT_SR, size=0.3, tone=1.0, seed=0):
    rng = _rng(seed, 40)
    size = float(np.clip(size, 0.0, 1.0))
    x = np.concatenate([_snap(rng, 0.005 + 0.115 * size, 0.003 + 0.007 * size), np.zeros(int(0.03 * SR))])
    return _fin(_formant(x, float(np.clip(tone, 0.5, 1.5))), sr, 0.75, 0.0003, 0.01)


def _gate(rng, n, duty: float, rate: float = 3.0):
    """Farnell random gate (findings 02 section 11): low-passed noise against a threshold, about
    3 Hz, slewed. ``duty`` is the fraction of time it is open (the threshold is that quantile)."""
    if duty >= 1.0:
        return np.ones(n)
    g = _lfn(rng, n, rate)
    return _lp((g > np.quantile(g, 1.0 - max(duty, 0.02))).astype(float), 60.0)


@sfx("arc", "Electric arc: broadband buzz re-striking on every half cycle of the mains, sputtering in and out.",
     dur=(2.0, "s"), mains=(50.0, "Hz mains frequency (50 or 60): the buzz is at twice this"),
     stability=(0.7, "0..1 fraction of time the arc holds (1 = steady welding arc)"),
     tone=(1.0, "formant scale 0.5..1.5"), seed=(0, "variation"))
def arc(sr=DEFAULT_SR, dur=2.0, mains=50.0, stability=0.7, tone=1.0, seed=0):
    """Noise x |sin(2 pi f t)|^3 (conduction around each current peak, so the buzz is at 2 f; Farnell:
    hum is "50-60 Hz plus 100-120 Hz") with a Farnell snap at every re-strike, the random gate, and
    the spark formant. # UNSOURCED: the cube law and the snap level."""
    rng = _rng(seed, 41)
    n = _n(dur)
    f0 = float(np.clip(mains, 20.0, 400.0))
    t = np.arange(n) / SR
    x = rng.uniform(-1, 1, n) * np.abs(np.sin(2 * np.pi * f0 * t)) ** 3
    for k in range(int(2 * f0 * n / SR)):
        _place(x, _snap(rng, 0.003, 0.002), (k + 0.15) / (2 * f0) * SR + rng.uniform(-20, 20), rng.uniform(0.2, 0.6))
    y = _formant(x * _gate(rng, n, float(stability)), float(np.clip(tone, 0.5, 1.5)))
    y = y + 0.08 * float(np.max(np.abs(y))) * np.sin(4 * np.pi * f0 * t)
    return _fin(y, sr, 0.65, 0.002, 0.03)


def mains_harmonics(n: int, rng, f0: float, bright: float = 0.3, beat: float = 0.4, top: float = 3000.0):
    """Harmonic series on exactly ``f0`` at 48 kHz. Amplitude 1 / k^(2.6 - 1.6 bright), the 2nd
    harmonic doubled (magnetostriction is rectified: Farnell, "50-60 Hz plus 100-120 Hz") and split
    into a pair ``beat`` Hz apart (Farnell's 99.8 / 100.2 Hz phasors beat at 0.4 Hz).
    # UNSOURCED: the roll-off exponent."""
    t = np.arange(n) / SR
    y = np.zeros(n)
    for k in range(1, int(min(top, 0.4 * SR) / f0) + 1):
        a = k ** -(2.6 - 1.6 * bright)
        if k == 2 and beat > 0:
            y += a * (np.sin(2 * np.pi * (2 * f0 - beat / 2) * t + rng.uniform(0, 6.28))
                      + np.sin(2 * np.pi * (2 * f0 + beat / 2) * t + rng.uniform(0, 6.28)))
        else:
            y += a * (2.0 if k == 2 else 1.0) * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 6.28))
    return y


@sfx("mains_hum", "Mains / transformer hum: exact 50 or 60 Hz fundamental with a harmonic series and slow beating.",
     dur=(3.0, "s"), mains=(50.0, "Hz fundamental (50 or 60)"),
     harmonics=(0.3, "0..1 buzziness: 0 = soft hum, 1 = harsh transformer buzz"),
     beat=(0.4, "Hz slow beating of the 2nd harmonic (0 = none)"),
     resonance=(0.4, "0..0.9 feedback of the casing comb tuned to one mains period"), seed=(0, "variation"))
def mains_hum(sr=DEFAULT_SR, dur=3.0, mains=50.0, harmonics=0.3, beat=0.4, resonance=0.4, seed=0):
    """Farnell Practical 16: slow random level (noise -> low-pass -> squared) and a positive-feedback
    comb of 10-30 ms (here exactly 1 / f0, which is 20 ms at 50 Hz). Low-passed at 4 kHz as in
    ``physical.ambience.hum``. # UNSOURCED: LFO depth 15 %."""
    rng = _rng(seed, 42)
    n = _n(dur)
    f0 = float(np.clip(mains, 20.0, 400.0))
    y = mains_harmonics(n, rng, f0, float(np.clip(harmonics, 0, 1)), float(max(beat, 0.0)))
    y = y * np.clip(1.0 + 0.15 * (_lfn(rng, n, 0.5) ** 2 - 1.0), 0.6, 1.5)
    d = int(round(SR / f0))
    a = np.zeros(d + 1)
    a[0], a[-1] = 1.0, -float(np.clip(resonance, 0.0, 0.9))
    return _fin(_lp(lfilter([1.0], a, y), 4000.0, 2), sr, 0.5, 0.03, 0.05)


@sfx("tesla", "Tesla coil / stun-gun zap: a rapid train of spark discharges with corona hiss.",
     dur=(1.0, "s"), rate=(120.0, "discharges per second, 10..600 (the pitch of the buzz)"),
     chaos=(0.4, "0..1 timing and level irregularity"), sweep=(0.0, "octaves the rate glides over the duration (+ up, - down)"),
     tone=(1.0, "formant scale 0.5..1.5"), seed=(0, "variation"))
def tesla(sr=DEFAULT_SR, dur=1.0, rate=120.0, chaos=0.4, sweep=0.0, tone=1.0, seed=0):
    """A train of Farnell snaps (2-6 ms) through the spark formant. # UNSOURCED: the corona layer
    (noise above 3 kHz following the discharge envelope) and the jitter laws."""
    rng = _rng(seed, 43)
    n = _n(dur)
    ch = float(np.clip(chaos, 0.0, 1.0))
    rate = float(np.clip(rate, 10.0, 600.0))
    x = np.zeros(n)
    t = 0.0
    while t < n / SR:
        _place(x, _snap(rng, rng.uniform(0.002, 0.006), 0.002), t * SR, rng.lognormal(0.0, 0.1 + 0.5 * ch))
        t += (1.0 + ch * rng.uniform(-0.5, 0.5)) / (rate * 2.0 ** (float(sweep) * t * SR / n))
    env = _lp(np.abs(x), 80.0)
    y = _formant(x, float(np.clip(tone, 0.5, 1.5)))
    y = _unit(y) + 0.25 * _unit(_hp(rng.uniform(-1, 1, n), 3000.0, 2) * env)
    return _fin(y * _ar(n, 0.004, 0.05), sr, 0.65, 0.0005, 0.01)


@sfx("neon", "Neon / fluorescent tube: bright mains buzz that flickers, with a small tick each time it re-strikes.",
     dur=(3.0, "s"), mains=(50.0, "Hz (50 or 60)"), flicker=(0.5, "0 steady .. 1 failing tube"),
     seed=(0, "variation"))
def neon(sr=DEFAULT_SR, dur=3.0, mains=50.0, flicker=0.5, seed=0):
    """A neon lamp is a relaxation oscillator (findings 01 section 3.8). Buzz = ``mains_harmonics``
    (bright, high-passed). # UNSOURCED: dwell times (lit: exponential, mean 0.05 + 0.8 (1 - flicker)
    s; dark: 20-150 ms), the re-strike tick (Farnell snap) and the 2.8 kHz glass ping."""
    rng = _rng(seed, 44)
    n = _n(dur)
    f0 = float(np.clip(mains, 20.0, 400.0))
    fl = float(np.clip(flicker, 0.0, 1.0))
    gate = np.ones(n)
    extra = np.zeros(n)
    t = rng.exponential(0.3) if fl > 0 else n / SR
    while t < n / SR and fl > 0:
        off = rng.uniform(0.02, 0.15)
        gate[int(t * SR):int((t + off) * SR)] = 0.0
        i = int((t + off) * SR)
        _place(extra, _formant(np.concatenate([_snap(rng, 0.004, 0.002), np.zeros(960)])), i, 0.5)
        m = int(0.08 * SR)
        _place(extra, np.sin(2 * np.pi * 2800.0 * rng.uniform(0.9, 1.1) * np.arange(m) / SR) * np.exp(-np.arange(m) / (0.015 * SR)), i, 0.15)
        t += off + rng.exponential(0.05 + 0.8 * (1.0 - fl))
    buzz = _hp(mains_harmonics(n, rng, f0, 0.75, 0.4), 80.0)
    y = _unit(buzz) * _lp(gate, 300.0) + extra * 2.0
    return _fin(y, sr, 0.5, 0.01, 0.03)


@sfx("lightning", "Lightning strike: close crack (electric snap), then Farnell's multi-layer thunder.",
     dur=(6.0, "s"), distance=(150.0, "m to the strike, 20..6000: close = sharp bang, far = rumble only"),
     delay=(0.0, "1 = insert the true sound travel time (distance / 343 s) before it starts"),
     seed=(0, "variation"))
def lightning(sr=DEFAULT_SR, dur=6.0, distance=150.0, delay=0.0, seed=0):
    """Crack: Farnell's snap scaled up (noise x 0.5 v^4 over 2 s + 7 kHz -> 20 Hz sweep in 10 ms
    through ``spark6formant``, the explosion "mid" layer of findings 03), used as he suggests for
    the "incredibly loud and sharp bang" within 500 m (findings 02 sections 11, 12.2). Thunder:
    ``physical.ambience.thunder`` (findings 02 section 12.2), its distance ramp set to
    distance / 3000. # UNSOURCED: crack gain (1 - distance / 500)^2 and its low-pass
    12 kHz / (1 + distance / 100)."""
    rng = _rng(seed, 45)
    n = _n(dur)
    d = float(np.clip(distance, 20.0, 6000.0))
    th = _fit(amb.thunder(rng, dur=max(n / SR, 1.25), distance=float(np.clip(d / 3000.0, 0.0, 1.0))), n)
    y = th / (float(np.max(np.abs(th))) + 1e-12)
    g = max(0.0, 1.0 - d / 500.0) ** 2
    if g > 0:
        crack = _formant(_snap(rng, min(2.0, n / SR), 0.010), 0.8)
        y = y + 1.6 * g * _fit(_lp(crack, 12000.0 / (1.0 + d / 100.0), 2) / (float(np.max(np.abs(crack))) + 1e-12), n)
    if delay >= 0.5:
        y = np.concatenate([np.zeros(min(int(d / C_AIR * SR), n // 2)), y])[:n]
    return _fin(y, sr, 0.85, 0.0005, 0.3)


# =============================================================================================
# PHASE TRANSITIONS - ice
# =============================================================================================

ICE_E, ICE_RHO, ICE_NU = 9.0e9, 917.0, 0.33   # UNSOURCED (not in the findings): textbook values for ice
ICE_C_MAX = 1900.0                            # UNSOURCED: shear-wave ceiling on the flexural group speed, m/s


def ice_chirp(distance: float = 60.0, thickness: float = 0.05, loss: float = 0.004,
              return_freq: bool = False):
    """The dispersive "pew" of a crack in lake ice, heard ``distance`` m away, at 48 kHz.

    Thin-plate (Kirchhoff) flexural waves: w = k^2 a with a = sqrt(D / (rho h)) =
    h sqrt(E / (12 rho (1 - nu^2))), so the group speed c_g = 2 sqrt(w a) grows with frequency and
    high frequencies arrive first. A component of frequency f arrives at t = d / c_g, hence the
    instantaneous frequency falls as

        f(t) = d^2 / (8 pi a t^2),   t >= t0 = d / c_max

    where c_max caps the group speed (thin-plate theory fails once the wavelength nears the
    thickness), which gives the top frequency c_max^2 / (8 pi a): about 3 kHz for 5 cm ice and
    500 Hz for 30 cm - thin ice sings, thick ice booms. Stationary-phase amplitude of a dispersed
    impulse: A ~ sqrt(|df/dt|) ~ (t0 / t)^1.5, times exp(-loss 2 pi f (t - t0)).
    # UNSOURCED: derived here from textbook plate theory (no findings file covers ice); E, rho, nu,
    # c_max and ``loss`` are assumptions.
    """
    h = float(np.clip(thickness, 0.005, 2.0))
    d = float(np.clip(distance, 2.0, 2000.0))
    a = h * math.sqrt(ICE_E / (12.0 * ICE_RHO * (1.0 - ICE_NU ** 2)))
    c = d * d / (8.0 * math.pi * a)
    t0 = d / ICE_C_MAX
    f_top = min(c / (t0 * t0), 0.4 * SR)
    t0 = math.sqrt(c / f_top)
    t_end = min(math.sqrt(c / 40.0), t0 + 3.0)              # down to 40 Hz
    t = t0 + np.arange(max(64, int((t_end - t0) * SR))) / SR
    f = c / (t * t)
    y = (t0 / t) ** 1.5 * np.exp(-loss * 2 * np.pi * f * (t - t0)) * np.sin(2 * np.pi * c * (1.0 / t0 - 1.0 / t))
    k = min(48, len(y) // 4)
    y[:k] *= np.linspace(0.0, 1.0, k)
    y[-k:] *= np.linspace(1.0, 0.0, k)
    return (y, f) if return_freq else y


def _tick(rng, ms: float, fc: float, q: float = 3.0):
    """Brittle micro-crack: a 4th-power noise burst of ``ms`` through a band-pass."""
    m = max(16, int(ms * 1e-3 * SR))
    return _bp(np.concatenate([rng.uniform(-1, 1, m) * (1.0 - np.arange(m) / m) ** 4, np.zeros(4 * m)]), fc, q)


@sfx("ice_crack", "Lake ice cracking: the falling 'pew' of the dispersed plate wave, then the sharp brittle crack.",
     dur=(1.5, "s"), thickness=(0.05, "ice thickness in m, 0.01..1: thin = high laser-like pew, thick = low boom"),
     distance=(60.0, "m to the crack, 5..1000: farther = longer chirp"),
     cracks=(3, "number of cracks in the event, 1..20"),
     brittle=(0.6, "0..1 level of the air-borne snap"), seed=(0, "variation"))
def ice_crack(sr=DEFAULT_SR, dur=1.5, thickness=0.05, distance=60.0, cracks=3, brittle=0.6, seed=0):
    """Each crack: ``ice_chirp`` through the ice, then the snap through the air, which arrives later
    (d / 343 s) because the plate wave is faster. # UNSOURCED: crack clustering (exponential gaps,
    mean 0.12 s), per-crack distance spread +-30 %, snap band 1.5-5 kHz."""
    rng = _rng(seed, 50)
    n = _n(dur)
    y = np.zeros(n)
    t = 0.0
    for k in range(int(np.clip(cracks, 1, 20))):
        d = float(np.clip(distance, 5.0, 1000.0)) * rng.uniform(0.7, 1.3)
        g = rng.uniform(0.4, 1.0) if k else 1.0
        _place(y, ice_chirp(d, float(thickness)), (t + d / ICE_C_MAX) * SR, g)
        snap = _lp(_tick(rng, rng.uniform(3.0, 10.0), rng.uniform(1500.0, 5000.0), 1.5), 12000.0 / (1.0 + d / 80.0), 2)
        _place(y, snap / (float(np.max(np.abs(snap))) + 1e-12), (t + d / C_AIR) * SR, g * float(np.clip(brittle, 0, 1)) * 0.8)
        t += rng.exponential(0.12)
    return _fin(y, sr, 0.75, 0.0005, 0.05)


@sfx("ice_cubes", "Ice cubes clinking in a glass of drink.",
     dur=(1.5, "s"), cubes=(3, "number of cubes, 1..12"), glass=(1.0, "glass pitch scale, 0.5..2 (larger glass = lower)"),
     liquid=(0.6, "0..1 fill: damps and lowers the glass, adds a little slosh"),
     energy=(0.7, "0..1 how hard the glass is swirled"), seed=(0, "variation"))
def ice_cubes(sr=DEFAULT_SR, dur=1.5, cubes=3, glass=1.0, liquid=0.6, energy=0.7, seed=0):
    """Collisions settle like a decaying swirl (rate falling exponentially). Each is a short contact
    pulse (``physical.core.contact_pulse``) driving a glass body (modal table ``sy_vase`` through
    ``physical.modal``) plus the cube's own click: ice is stiff and ringing, "a candle taken from
    the freezer will sound like a wood or metal bar" (findings 01 section 1.6). Slosh: HF bubbles.
    # UNSOURCED: collision rate 9 x cubes x energy per second decaying over 0.45 dur, the liquid
    loading (-18 % pitch, x2.5 damping at full), cube click 2.5-6 kHz."""
    rng = _rng(seed, 51)
    n = _n(dur)
    t = np.arange(n) / SR
    lq, en = float(np.clip(liquid, 0, 1)), float(np.clip(energy, 0.05, 1))
    times = bubble_times(n, rng, 9.0 * float(np.clip(cubes, 1, 12)) * en * np.exp(-t / (0.45 * n / SR)))
    if len(times) == 0:
        times = np.array([int(0.02 * SR)])
    force = np.zeros(n)
    clicks = np.zeros(n)
    for s in times:
        v = rng.lognormal(0.0, 0.6) * math.exp(-s / (0.6 * n))
        _place(force, contact_pulse(rng.uniform(0.15, 0.5)), s, v)
        _place(clicks, _tick(rng, rng.uniform(0.8, 2.5), rng.uniform(2500.0, 6000.0), 6.0), s, v)
    modes = M.resolve("sy_vase", f_scale=float(np.clip(glass, 0.5, 2.0)) * (1.0 - 0.18 * lq), d_scale=1.0 + 1.5 * lq)
    y = _unit(M.render_modes(force, modes)) + 0.7 * _unit(clicks)
    if lq > 0:
        y = y + 0.25 * lq * _unit(bubble_cloud(dur, rng, 25.0 * en * np.exp(-t / (0.5 * n / SR)), 0.8e-3, 4e-3, cluster=0.5))
    return _fin(y, sr, 0.65, 0.0005, 0.05)


@sfx("freeze", "Water freezing / ice under stress: accelerating brittle ticks, stick-slip creaks and faint ice pings.",
     dur=(4.0, "s"), rate=(8.0, "micro-cracks per second at the end, 0.5..80 (they start at a fifth of this)"),
     creak=(0.5, "0..1 level of the stick-slip creaking"), pings=(2, "dispersive ice pings in the take, 0..12"),
     seed=(0, "variation"))
def freeze(sr=DEFAULT_SR, dur=4.0, rate=8.0, creak=0.5, pings=2, seed=0):
    """Creaks: Farnell's stick-slip (``physical.ambience.creak``, findings 02 section 4) with the
    "pine" body scaled to 0.45 so its formants sit higher (stiffer, colder material; the size
    factor is # UNSOURCED). Ticks: Poisson, log-normal sizes, 1-4 ms bursts at 2-6 kHz
    (# UNSOURCED). Pings: ``ice_chirp`` in thin ice, far away."""
    rng = _rng(seed, 52)
    n = _n(dur)
    x = np.arange(n) / n
    y = np.zeros(n)
    ticks = bubble_times(n, rng, float(np.clip(rate, 0.5, 80.0)) * (0.2 + 0.8 * x), cluster=0.4)
    for s in (ticks if len(ticks) else [int(rng.uniform(0.1, 0.6) * n)]):
        _place(y, _tick(rng, rng.uniform(1.0, 4.0), rng.uniform(2000.0, 6000.0)), s, rng.lognormal(0.0, 0.5))
    y = _unit(y)
    if creak > 0:
        force = np.clip(0.25 + 0.55 * np.clip(_lfn(rng, n, 0.7), -0.4, 1.2), 0.0, 0.95)
        c = _fit(amb.creak(force, rng, material="pine", size=0.45), n)
        if np.any(c):
            y = y + 1.2 * float(np.clip(creak, 0, 1)) * _unit(c)
    for _ in range(int(np.clip(pings, 0, 12))):
        _place(y, ice_chirp(rng.uniform(30.0, 120.0), rng.uniform(0.02, 0.06)), rng.uniform(0.05, 0.85) * n,
               rng.uniform(4.0, 10.0))
    return _fin(y, sr, 0.65, 0.002, 0.05)
