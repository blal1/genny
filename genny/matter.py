"""Sounds organised by state of matter: liquids, gases, fire / plasma / electricity, phase changes.

Everything is rendered at the physical engine's internal rate (48 kHz, the rate of the reused
``genny.physical`` kernels) and resampled at the boundary with ``scipy.signal.resample_poly``.

Models (sources of every constant: ``research/SOURCES.md``):

* **Bubbles** - Minnaert frequency with surface tension, radius-dependent radiation + viscous + thermal
  damping, attack window, depth-driven pitch rise, bubble counts and frequency ranges per scene;
  amplitude eps*r0, radius power laws, tube bubbles; terminal rise speed (2/3)sqrt(gR), burst timing,
  pouring vessel; recursive secondary drops, two-band sizes.
* **Fire** - p = dI/dt, 30 Hz high-pass, 180 Hz hand-over, f^-alpha bandwidth extension. The flame
  flux I(t) is a stochastic surrogate, or follows a combustion-noise spectrum (peak St U / L,
  f^-2.5 above). Crackle, hiss, lapping.
* **Electricity** - snaps, random gate and the ``spark6formant`` bank. Thunder is ``physical.ambience.thunder``.
* **Wind** - through ``physical.ambience.wind_scene``.
* **Jets** - Strouhal peak f = St U / D with St = 0.2; level from the free-jet U^8 law.
* **Ice** - the dispersive chirp is the thin-plate flexural wave (derivation in ``ice_chirp``);
  E, rho, nu and the shear-speed ceiling of polycrystalline ice at -16 C.
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
    1/(2 fc) s. No filter warm-up, and smooth by construction (the flame driver
    must be smooth)."""
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


def _fin(y, sr, peak=0.7, fin=0.002, fout=0.02, soft=True) -> np.ndarray:
    """Mono, DC removed, peak-normalised, resampled to ``sr``, zero at both ends. ``soft=False``: the linear fades of the take of `arms.ricochet` validated by ear."""
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
    if not soft:
        y[:a] *= np.linspace(0.0, 1.0, a)
        y[-b:] *= np.linspace(1.0, 0.0, b)
        return y
    y[:a] *= np.sin(0.5 * np.pi * np.arange(a) / a) ** 2
    y[-b:] *= np.cos(0.5 * np.pi * (np.arange(b) + 1) / b) ** 2
    return y


# =============================================================================================
# LIQUIDS - the acoustic bubble
# =============================================================================================

G_GRAV, RHO_W, P_ATM, GAMMA = 9.8, 1000.0, 101325.0, 1.4
SIGMA_W, D_GAS, C_WATER, MU_W = 0.0726, 2.122e-5, 1497.0, 8.9e-4
G_TH = 3.0 * GAMMA * P_ATM / (4.0 * math.pi * RHO_W * D_GAS)      # 1.60e6 1/s
# psi = PSI_K / w0; PSI_K_HF only for soft=False (frozen arms.ricochet take)
PSI_K = 16.0 * G_TH * 2.0 * math.pi / (9.0 * (GAMMA - 1.0) ** 2)
PSI_K_HF = 16.0 * G_TH * G_GRAV / (9.0 * (GAMMA - 1.0) ** 2)
K_RISE, ETA_RISE = 72.95, 1.24
XUE_RISE = math.sqrt(0.8 * (4.0 / 3.0) / (2.0 * 0.2))
MINNAERT_WR = math.sqrt(3.0 * GAMMA * P_ATM / RHO_W)              # w0 * r0 = 20.63 m/s (f0 r0 = 3.28)


def minnaert(r0, depth: float = 0.0):
    """Minnaert frequency in Hz at ``depth`` m below the surface."""
    return MINNAERT_WR * math.sqrt(1.0 + RHO_W * G_GRAV * float(depth) / P_ATM) / (2.0 * np.pi * np.asarray(r0, float))


def bubble_params(r0, depth: float = 0.0, legacy: bool = False, radiation: bool = True):
    """Returns (w0, beta, w_d, delta) for radius r0 in metres (rad/s, 1/s, rad/s, -).

    w0 = sqrt(3 gamma p0 - 2 sigma / r0) / (r0 sqrt(rho));  beta = w0 delta / sqrt(delta^2 + 4);
    delta = w0 r0 / c_f  +  4 mu / (rho w0 r0^2)  +  2 (sqrt(psi - 3) - (3g-1)/(3(g-1))) / (psi - 4),
    psi = PSI_K p0 / (P_atm w0); ``legacy``: PSI_K_HF. depth (m) raises p0.
    ``radiation=False`` drops the w0 r0 / c_f term: a bubble just under the free surface radiates as a
    dipole, whose loss is negligible.
    """
    r0 = np.asarray(r0, float)
    p0 = P_ATM + RHO_W * G_GRAV * float(depth)
    w0 = np.sqrt(3 * GAMMA * p0 - 2 * SIGMA_W / r0) / (r0 * math.sqrt(RHO_W))
    psi = (PSI_K_HF if legacy else PSI_K) * p0 / P_ATM / w0
    d = ((w0 * r0 / C_WATER if radiation else 0.0) + 4 * MU_W / (RHO_W * w0 * r0 * r0)
         + 2 * (np.sqrt(psi - 3) - (3 * GAMMA - 1) / (3 * (GAMMA - 1))) / (psi - 4))
    beta = w0 * d / np.sqrt(d * d + 4)
    return w0, beta, np.sqrt(w0 * w0 - beta * beta), d


@numba.njit(cache=True)
def _beta_of(w, legacy=False, pr=1.0, rad=1.0):
    """Damping rate at angular frequency w, the radius taken as r0 = sqrt(3 gamma p0 / rho) / w.
    ``pr``: p0 / P_atm (``bubble_params(depth=)``). ``legacy``: PSI_K_HF. ``rad``: weight of the
    radiation term (0 = near-surface dipole)."""
    r0 = MINNAERT_WR * np.sqrt(pr) / w
    if legacy:
        psi = 16.0 * G_TH * G_GRAV * pr / (9.0 * (GAMMA - 1.0) ** 2 * w)
    else:
        psi = PSI_K * pr / w
    d = (w * r0 / C_WATER * rad + 4.0 * MU_W / (RHO_W * w * r0 * r0)
         + 2.0 * (np.sqrt(psi - 3.0) - (3.0 * GAMMA - 1.0) / (3.0 * (GAMMA - 1.0))) / (psi - 4.0))
    return w * d / np.sqrt(d * d + 4.0)


@numba.njit(cache=True)
def _bubble_bank(out, starts, w0s, amps, depths, vrise, phi0s, wmax, soft, pr, rad, drift):
    """Sum of bubbles. Each is q'' + 2 beta q' + w^2 q = impulse with time-varying w, beta.

    Integrator: exact amplitude/phase recursion (a *= exp(-beta dt), phase += w_d dt), which is the
    closed-form impulse response for constant coefficients. A midpoint rule
    (the usual reference); checked numerically at 48 kHz that rule is 3 % sharp at 3.3 kHz
    and loses half of the damping (|mu| ~ 1 + (w dt)^4 / 8), so it is not used.
    Pitch rise per sample: dw = K w_d exp(-eta (phi/phi0 - 1)) dphi, clamped to
    wmax * w0 and to 0.45 of the sample rate.
    Attack window: gain exp(-(e - 0.85)^2 / 0.0028125) while the envelope e >= 0.85.
    ``drift`` (dB/s per bubble, empty = none): slow change of the transfer gain while it rings.
    """
    n = out.shape[0]
    dt = 1.0 / 48000.0
    wnyq = 2.0 * np.pi * 0.45 * 48000.0
    has_drift = drift.shape[0] > 0
    for k in range(starts.shape[0]):
        s = starts[k]
        w0 = min(w0s[k], wnyq)
        w = w0
        b = _beta_of(w, not soft, pr, rad)
        phi = -abs(depths[k])
        m = min(int(6.91 / b * 48000.0) + 1, n - s)          # stop at 1/1000
        a = 1.0
        ph = 0.0
        for i in range(m):
            wd = np.sqrt(max(w * w - b * b, 0.0))
            g = np.exp(-(a - 0.85) ** 2 / 0.0028125) if a >= 0.85 else 1.0
            if soft and a < 4e-3:                            # closes on the way to the stop: -60 dB is a step on anything quieter
                g = np.sin(0.5 * np.pi * (a - 1e-3) / 3e-3) ** 2
            if has_drift:
                g *= 10.0 ** (drift[k] * i * dt / 20.0)
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
                    b = _beta_of(w, not soft, pr, rad)


def hf_bubble(r0: float, depth: float = 1.0, rise: float = 1.0, phi0: float = -0.016,
              dur: float | None = None) -> np.ndarray:
    """One Harmonic-Fluids bubble of radius ``r0`` (m) at 48 kHz, unit amplitude.

    depth: start depth below the surface (m). Deeper than a few |phi0| = constant pitch.
    rise: multiplier on the terminal rise speed (2/3) sqrt(g r);
      ``XUE_RISE / (2/3)`` selects XUE_RISE sqrt(g r).
    phi0: distance at which the pitch starts to move, -0.008 ... -0.025 m.
    """
    w0, beta, _, _ = bubble_params(r0)
    n = _n(dur if dur else 7.0 / float(beta))
    out = np.zeros(n)
    v = rise * (2.0 / 3.0) * math.sqrt(G_GRAV * r0)
    _bubble_bank(out, np.zeros(1, np.int64), np.array([float(w0)]), np.ones(1), np.array([float(depth)]),
                 np.array([v]), np.array([float(phi0)]), 3.0, True, 1.0, 1.0, np.zeros(0))
    return out


def _radii(rng, k: int, lo: float, hi: float, dist: str = "gauss", alpha: float = 2.9) -> np.ndarray:
    """Bubble radii in [lo, hi] metres.

    gauss: radii from a Gaussian.
      Chosen: mean = geometric mean of the range, sd = range / 6.
    two_band: two Gaussians (more convincing for streaming). Chosen: band centres.
    power: p(r) ~ r^-alpha (breaking waves 1.5-3.3, rain 2.9).
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
    exponential delays (mean 30 ms): bursts that decay in frequency, followed by a period
    of few bubbles. Onsets land on arbitrary samples (no frame-rate buzz).
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


def excitation_amp(r, law: str = "moss", eps=None):
    """Bubble source amplitude by radius. ``moss``: eps * r0 (r in mm). ``xue``: the radiated
    pressure of a bubble released by surface tension, S_b q = 2 sigma w0^2 / (r0^2 w_d), which goes as
    r0^-3; scaled to the moss value of a 1 mm bubble with eps = 0.1."""
    r = np.asarray(r, float)
    if law == "moss":
        return (0.1 if eps is None else eps) * r * 1e3
    if law == "pinch":
        return pinch_eps(r, eps) * r * 1e3
    if law != "xue":
        raise ValueError(f"unknown amp_law {law!r}: moss|pinch|xue")
    w0, _, wd, _ = bubble_params(r)
    w1, _, wd1, _ = bubble_params(1e-3)
    return 0.1 * (w0 ** 2 / (r * r * wd)) / (w1 ** 2 / (1e-6 * wd1))


PINCH_EPS_MAX = 0.5


def pinch_eps(r, eps=None):
    """Volume compression at pinch-off grown for small bubbles: eps (1 mm / r), clipped to 0.01..0.5."""
    e = 0.1 if eps is None else eps
    return np.clip(e * 1e-3 / np.asarray(r, float), 0.01, PINCH_EPS_MAX)


def impact_energy(radius_m, speed, rho: float = 1000.0):
    """Acoustic energy of a liquid impact: E = 4 pi K M^3, K = (2/3) pi rho R^3 U^2, M = U / c."""
    K = (2.0 / 3.0) * math.pi * rho * np.asarray(radius_m, float) ** 3 * np.asarray(speed, float) ** 2
    return 4.0 * math.pi * K * (np.asarray(speed, float) / C_WATER) ** 3


def impact_band(radius_m, speed) -> tuple[float, float]:
    """Band of the impact noise: U / R < f < c / R (Hz)."""
    return float(speed / radius_m), float(C_WATER / radius_m)


def _impact_noise(rng, size01: float, ref_peak: float) -> np.ndarray:
    D = 1.0 + 3.6 * size01
    R = 0.5e-3 * D
    U = float(terminal_speed(D))
    lo, hi = impact_band(R, U)
    m = max(8, int(max(R / U, 1e-4) * SR))
    x = rng.uniform(-1.0, 1.0, m) * np.hanning(m)
    x = _hp(x, min(lo, 0.4 * SR), 2)
    if hi < 0.45 * SR:
        x = _lp(x, hi, 2)
    Dr = 2.6
    g = math.sqrt(float(impact_energy(R, U) / impact_energy(0.5e-3 * Dr, float(terminal_speed(Dr)))))
    return x / (np.max(np.abs(x)) + 1e-12) * ref_peak * g


def image_gain(w0, d, theta: float = 0.0):
    """Pressure of a bubble ``d`` m under the free surface, heard at angle theta from the vertical,
    relative to the same bubble in open water: the inverted image source 2 d below makes a dipole,
    |1 - exp(2 i k d cos theta)| = 2 |sin(k d cos theta)|, k = w0 / c_water."""
    return 2.0 * np.abs(np.sin(np.asarray(w0, float) / C_WATER * np.asarray(d, float) * math.cos(theta)))


def _render_bubbles(n, rng, starts, r, depth=(0.002, 0.03), rise=1.0, soft=True, water_depth=0.0,
                    amp_law="moss", transfer_db=10.0, image=False, radiation=True, drift=0.0,
                    pitch=1.0) -> np.ndarray:
    """Amplitude: ``excitation_amp`` (moss: eps power law on [0.01, 0.1], mu = 2), times a
    log-uniform +-``transfer_db`` transfer gain (equal-gain banks sound harsh and unrealistic).
    Start depth uniform in ``depth``, rise speed (2/3) sqrt(g r), phi0 uniform in -8 ... -25 mm.
    ``water_depth`` (m): see ``bubble_params(depth=)``. ``image``: the free-surface dipole gain
    (``image_gain`` at the start depth). ``radiation=False``: no radiation loss (near-surface dipole).
    ``drift`` (dB/s): each bubble's transfer gain drifts at a rate uniform in +-drift while it rings."""
    k = len(starts)
    out = np.zeros(n)
    if k == 0:
        return out
    w0 = bubble_params(r, depth=water_depth)[0]
    if pitch != 1.0:
        w0 = w0 * pitch
    td = float(transfer_db)
    amp = excitation_amp(r, amp_law, P._powerlaw(rng, k, 0.01, 0.1, -2.0)) * 10.0 ** (rng.uniform(-td, td, k) / 20.0)
    dep = rng.uniform(depth[0], depth[1], k)
    phi = -rng.uniform(0.008, 0.025, k)
    if image:
        amp = amp * image_gain(w0, dep)
    dr = rng.uniform(-drift, drift, k) if drift > 0 else np.zeros(0)
    _bubble_bank(out, np.ascontiguousarray(starts, np.int64), np.ascontiguousarray(w0), amp,
                 dep, rise * (2.0 / 3.0) * np.sqrt(G_GRAV * r), phi, 3.0, soft,
                 1.0 + RHO_W * G_GRAV * float(water_depth) / P_ATM, 1.0 if radiation else 0.0, dr)
    return out


def bubble_cloud(dur: float, rng, rate, r_lo: float, r_hi: float, dist: str = "gauss", *,
                 alpha: float = 2.9, cluster: float = 0.0, depth=(0.002, 0.03), rise: float = 1.0,
                 r_scale=None, return_info: bool = False, soft: bool = True, water_depth: float = 0.0,
                 **render):
    """A population of bubbles at 48 kHz. ``rate`` is bubbles per second (number or per-sample
    array); ``r_scale`` an optional per-sample radius multiplier (pouring: sizes fall as it fills).
    ``dist="plume"``: ``plume_radii``. ``render``: options of ``_render_bubbles`` (amp_law,
    transfer_db, image, radiation, drift)."""
    n = _n(dur)
    starts = bubble_times(n, rng, rate, cluster)
    r = plume_radii(rng, len(starts)) if dist == "plume" else _radii(rng, len(starts), r_lo, r_hi, dist, alpha)
    if r_scale is not None and len(r):
        r = np.clip(r * np.asarray(r_scale)[starts], 1e-4, 0.03)
    y = _render_bubbles(n, rng, starts, r, depth, rise, soft, water_depth, **render)
    if return_info:
        return y, dict(n=len(starts), starts=starts, r=r)
    return y


# ---------------------------------------------------------------------------------------------
# bubble plumes, coupled bubbles, boundaries
# ---------------------------------------------------------------------------------------------

PLUME_VOID = 0.34                       # void fraction of a breaking-wave plume (0.3-0.4)
COUPLING_EPS = 2.0                      # regularisation of the coupled-bubble mass matrix (2 sqrt 2 if dense)
plume_radii, plume_skin_depth = P.plume_radii, P.plume_skin_depth


def plume_frequency(radius: float, void: float = PLUME_VOID, depth: float = 0.0) -> float:
    """Lowest collective (volume) mode of a spherical bubble plume of ``radius`` m:
    f = (1 / (2 pi R)) sqrt(3 p0 / (rho beta)), beta the void fraction (6.3 Hz for R = 0.75 m)."""
    p0 = P_ATM + RHO_W * G_GRAV * float(depth)
    return math.sqrt(3.0 * p0 / (RHO_W * float(void))) / (2.0 * math.pi * float(radius))


def coupled_frequencies(r, pos, eps: float = COUPLING_EPS, depth: float = 0.0):
    """Natural frequencies (Hz, ascending) of bubbles of radii r (m) at positions pos (k x 3, m)
    coupled through the liquid: K y = w^2 M y, K = diag(w_i^2), M_ii = 1,
    M_ij = 1 / sqrt(d_ij^2 / (r_i r_j) + eps^2). Far apart: w_i; all at one point: the
    collective mode w0 sqrt(eps / (eps - 1 + N)) and N - 1 modes at w0 sqrt(eps / (eps - 1))."""
    from scipy.linalg import eigh
    r = np.atleast_1d(np.asarray(r, float))
    pos = np.asarray(pos, float).reshape(len(r), -1)
    w = bubble_params(r, depth=depth)[0]
    d2 = np.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=-1)
    M = 1.0 / np.sqrt(d2 / np.outer(r, r) + eps * eps)
    np.fill_diagonal(M, 1.0)
    return np.sqrt(eigh(np.diag(w * w), M, eigvals_only=True)) / (2.0 * math.pi)


def collective_shift(n_active, eps: float = COUPLING_EPS):
    """Factor on the frequency of the lowest mode of ``n_active`` close bubbles: sqrt(eps / (eps - 1 + N))."""
    return np.sqrt(eps / (eps - 1.0 + np.maximum(np.asarray(n_active, float), 1.0)))


def bubble_capacitance(r0, surface: float = math.inf, wall: float = math.inf):
    """Electrostatic capacitance (m) of a bubble of radius r0 at ``surface`` m under a free surface
    and ``wall`` m from a rigid wall, first images: C = r0 / (1 - r0 / (2 s) + r0 / (2 w)).
    The pitch goes as sqrt(C / r0): up near the surface, down near a wall."""
    r0 = np.asarray(r0, float)
    return r0 / (1.0 - r0 / (2.0 * surface) + r0 / (2.0 * wall))


def capacitance_pitch(r0, surface: float = math.inf, wall: float = math.inf):
    """Frequency factor sqrt(C / r0) of ``bubble_capacitance``."""
    return np.sqrt(bubble_capacitance(r0, surface, wall) / np.asarray(r0, float))


def entrain_weight(tau):
    """Entrainment kernel over the normalised time tau in [0, 1]: 1 - 4 tau^3 up to 0.5,
    4 (1 - tau)^3 after, 0 outside."""
    t = np.asarray(tau, float)
    return np.where((t < 0) | (t > 1), 0.0, np.where(t <= 0.5, 1.0 - 4.0 * t ** 3, 4.0 * (1.0 - t) ** 3))


ENTRAIN_TAU = 1.45                       # s: mean rejection time of the stochastic entrainment scenes


def forcing(t, r0: float, kind: str = "pinch", theta: float = 0.6, t_lim: float | None = None):
    """Forcing that starts a bubble ringing, up to a constant factor (the internal pressure).
    pinch: -9 kappa sigma eta sqrt(1 + eta^2) / (4 rho R^3) t^2 with eta = tan(theta) of the
      collapsing neck (theta 20-55 degrees), active for min(T/2, 600 us), T the period.
    merge: [1/2 - atan(3 (t - t_lim) / t_lim) / pi] 6 kappa sigma t^2 / (rho R^3); t_lim the time the
      neck takes to reach a fraction of the smaller radius (capped at 600 us)."""
    t = np.asarray(t, float)
    w0 = float(bubble_params(r0)[0])
    if kind == "pinch":
        eta = math.tan(theta)
        on = min(math.pi / w0, 600e-6)
        return np.where(t < on, -9.0 * GAMMA * SIGMA_W * eta * math.sqrt(1 + eta * eta) / (4 * RHO_W * r0 ** 3) * t * t, 0.0)
    if kind == "merge":
        tl = min(t_lim if t_lim else merge_time(r0), 600e-6)
        return (0.5 - np.arctan(3.0 * (t - tl) / tl) / math.pi) * 6.0 * GAMMA * SIGMA_W * t * t / (RHO_W * r0 ** 3) * (t < 4 * tl)
    raise ValueError(f"unknown forcing {kind!r}: pinch|merge")


MERGE_NECK = (0.4, 0.8)                  # neck radius / smaller radius when the merge forcing ends
MERGE_C = 1.62                           # inertial neck growth r = C (sigma R / rho)^(1/4) sqrt(t)


def merge_time(r_small: float, frac: float = 0.6) -> float:
    """Time (s) for the neck between two merging bubbles to reach ``frac`` of the smaller radius."""
    return (frac * r_small) ** 2 / (MERGE_C ** 2 * math.sqrt(SIGMA_W * r_small / RHO_W))


def forced_bubble(r0: float, kind: str = "pinch", rng=None, dur: float | None = None) -> np.ndarray:
    """One bubble started by ``forcing`` instead of the attack window: the forced damped oscillator
    x'' + 2 beta x' + w0^2 x = f(t), solved by convolution with its impulse response. Unit peak, 48 kHz."""
    rng = rng if rng is not None else np.random.default_rng(0)
    w0, beta, wd, _ = (float(v) for v in bubble_params(r0))
    n = _n(dur if dur else 7.0 / beta)
    t = np.arange(n) / SR
    if kind == "pinch":
        f = forcing(t, r0, "pinch", math.radians(rng.uniform(20.0, 55.0)))
    else:
        f = forcing(t, r0, "merge", t_lim=merge_time(r0, rng.uniform(*MERGE_NECK)))
    h = np.exp(-beta * t) * np.sin(wd * t) / wd
    x = np.convolve(f, h)[:n] / SR
    m = float(np.max(np.abs(x)))
    return x / m if m > 0 else x


# bubbles per second and radius range by scene
HF_POUR_RATE, HF_BABBLE_RATE, HF_SPLASH_COUNT, HF_DROP_BUBBLES = 1580.0, 3100.0, 127, 14.0 / 3.0
R_300, R_500, R_4K, R_5K, R_6K = 10.9e-3, 6.6e-3, 0.82e-3, 0.66e-3, 0.55e-3


def drop_times(dur: float, rng, rate: float, rate_end: float | None = None,
               jitter: float = 0.15) -> np.ndarray:
    """Drop onset times (s). A dripping source is a relaxation oscillator (surface tension against
    weight): nearly periodic. jitter 0 = periodic, 1 = Poisson
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


def terminal_speed(d_mm):
    """Terminal fall speed (m/s) of a raindrop of diameter ``d_mm``: 9.65 - 10.3 exp(-0.6 D)."""
    return np.maximum(9.65 - 10.3 * np.exp(-0.6 * np.asarray(d_mm, float)), 0.0)


def crater_freq(d_mm, speed=None):
    """Frequency (Hz) of the crater-retraction sound of a drop of diameter ``d_mm`` landing at
    ``speed`` (default: terminal speed): f = V / L with the crater rim length L = D / 2."""
    v = terminal_speed(d_mm) if speed is None else speed
    return v / (0.5 * np.asarray(d_mm, float) * 1e-3)


CAVE_DROP_VOLUME = 0.1433e-6             # m^3 of one cave drip
CAVE_DROP_R = (3.0 * CAVE_DROP_VOLUME / (4.0 * math.pi)) ** (1.0 / 3.0)   # 3.25 mm
DOUBLE_CREST = (39e-6, math.radians(80.0), 0.2)   # s delay, phase, gain of the 2 f0 component


def _drop(rng, size: float, nb: float = HF_DROP_BUBBLES, secondary: float = 0.5, click: float = 0.15,
          level: int = 0, soft: bool = True, harmonic: float = 0.0, crater: float = 0.0,
          impact: str = "parabolic") -> np.ndarray:
    """One drop falling into water, 0.9 s buffer.

    4-5 bubbles per drop, 500 Hz - 4 kHz, i.e. radii 6.6 - 0.82 mm. Largest bubble first.
    Impact: a parabolic pulse of width 0.1 + 12 U ms, kept quiet (far quieter than the bubbles).
    Secondary drops: each falling drop of sufficient size causes following smaller ones; Chosen: Poisson count, 12 ms mean bubble spacing, the
    size ratio 0.5.
    ``soft``: first bubble DROP_DELAY after the impact, drop diameter D = 1 + 3.6 size mm;
    rebound drops after DROP_REBOUND. ``soft=False``: frozen arms.ricochet take.
    """
    n = _n(0.9)
    k = int(np.clip(rng.poisson(nb), 1, 9))
    s01 = min(max(size, 0.0), 1.0)
    rm = (0.9 + 3.6 * s01) * 1e-3
    r = np.sort(np.clip(rng.normal(rm, 0.35 * rm, k), R_4K * 0.7, R_500 * 1.1))[::-1].copy()
    w = max(2, int((0.1 + 12.0 * rng.random()) * 1e-3 * SR))
    gaps = rng.exponential(0.012 * SR, k)
    if soft:
        D = 1.0 + 3.6 * s01
        d0 = float(np.interp(D, *DROP_DELAY)) * 1e-3 * SR
        if D > 2.2:
            d0 = max(d0 + rng.normal(0.0, 0.005 * SR), 0.0)
        starts = (w + d0 + np.concatenate([[0.0], np.cumsum(gaps[:-1])])).astype(np.int64)
    else:
        starts = (w + np.cumsum(gaps)).astype(np.int64)
    out = _render_bubbles(n, rng, starts, r, depth=(0.002, 0.012), soft=soft)
    if harmonic > 0 and soft:
        w0, b, wd, _ = (float(v) for v in bubble_params(r[0]))
        i0 = int(starts[0] + DOUBLE_CREST[0] * SR)
        tt = np.arange(n - min(i0, n)) / SR
        pk = float(np.max(np.abs(out)))
        out[i0:] += harmonic * DOUBLE_CREST[2] * pk * np.exp(-2 * b * tt) * np.sin(2 * wd * tt + DOUBLE_CREST[1]) * np.minimum(tt / 2e-4, 1.0)
    if crater > 0 and soft:
        D = 1.0 + 3.6 * s01
        fr = min(crater_freq(D), 0.45 * SR)
        L = int(0.003 * SR)
        tt = np.arange(L) / SR
        _place(out, crater * float(np.max(np.abs(out))) * np.sin(2 * np.pi * fr * tt) * np.sin(np.pi * tt / tt[-1]) ** 2, w)
    if impact == "franz" and soft:
        _place(out, _impact_noise(rng, s01, click * float(np.max(np.abs(out)))), 0)
    else:
        x = np.linspace(0.0, 1.0, w)
        out[:w] += click * float(np.max(np.abs(out))) * (1.0 - 4.0 * (x - 0.5) ** 2)
    if level < 2 and rng.random() < secondary:
        sub = _drop(rng, size * 0.5, 1.5, secondary * 0.5, click, level + 1, soft, impact=impact)
        t = rng.uniform(*DROP_REBOUND) if soft else rng.uniform(0.12, 0.35)
        _place(out, sub, t * SR, 0.5)
    return out


DROP_DELAY = ((1.0, 2.2), (9.35, 65.0))     # drop diameter mm -> first bubble ms, drops at terminal speed
DROP_REBOUND = (2.0 * math.sqrt(2.0 * 0.025 / G_GRAV), 2.0 * math.sqrt(2.0 * 0.15 / G_GRAV))


@sfx("drip", "One water drop into a pool: quiet impact, then 4-5 ringing air bubbles.",
     size=(0.4, "0..1 drop size: mean bubble radius 0.9..4.5 mm (3.6 kHz..730 Hz)"),
     bubbles=(4.7, "mean bubbles per drop (measured: 4-5)"),
     secondary=(0.5, "0..1 chance of smaller rebound drops"),
     click=(0.15, "0..1 level of the impact pulse"),
     harmonic=(0.0, "0..1 the first bubble's 2 f0 component (air disc trapped under the drop)"),
     crater=(0.0, "0..1 faint tone of the crater closing, f = speed / rim length"),
     impact=("parabolic", "parabolic (a quiet pulse) | franz (noise between U/R and c/R lasting R/U, energy ~ R^3 U^5: "
                          "big fast drops click much louder)"), seed=(0, "variation"))
def drip(sr=DEFAULT_SR, size=0.4, bubbles=4.7, secondary=0.5, click=0.15, harmonic=0.0, crater=0.0, impact="parabolic", seed=0):
    if impact not in ("parabolic", "franz"):
        raise ValueError("impact must be parabolic or franz")
    y = _drop(_rng(seed, 1), float(size), float(bubbles), float(secondary), float(click),
              harmonic=float(np.clip(harmonic, 0, 1)), crater=float(np.clip(crater, 0, 1)), impact=impact)
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
     droplets=(6, "crown droplets that fall back"),
     entrain=("exp", "exp (bubble rate decays from the impact) | stochastic (entrainment kernel over the rejection time)"),
     seed=(0, "variation"))
def splash(sr=DEFAULT_SR, size=0.5, dur=1.5, droplets=6, entrain="exp", seed=0):
    return _splash(sr, size, dur, droplets, seed, entrain=str(entrain))


def _splash(sr, size, dur, droplets, seed, soft=True, entrain="exp"):
    """``soft=False``: bubbles and fades as they were when `arms.ricochet surface=water` was validated by ear (they fire detect_clicks).
    127 bubbles in 1.5 s "concentrated in the event", 300 Hz - 6 kHz.
    Cavity: tube bubbles (``ambience.tube_bubble``). Crown droplets return after a ballistic
    flight t = 2 sqrt(2 h / g) from heights 5-50 cm. Chosen: 0.25 s bubble-rate decay, the
    cavity pinch-off delay 60-140 ms, heights, and the layer balance."""
    rng = _rng(seed, 3)
    size = float(np.clip(size, 0.0, 1.0))
    n = _n(dur)
    t = np.arange(n) / SR
    env = entrain_weight(t / ENTRAIN_TAU) if entrain == "stochastic" else np.exp(-t / 0.25)
    count = HF_SPLASH_COUNT * (0.4 + 1.2 * size)
    y = _unit(bubble_cloud(dur, rng, env * count / max(env.sum() / SR, 1e-9), R_6K, R_300, "gauss",
                           cluster=0.4, depth=(0.002, 0.06), soft=soft))
    slap = _lp(rng.uniform(-1, 1, n) * np.exp(-t / (0.004 + 0.012 * size)), 1500 + 2500 * (1 - size), 2)
    wash = _bp(rng.uniform(-1, 1, n), 1800.0, 0.6) * np.clip(t / 0.02, 0, 1) * np.exp(-t / 0.18)
    y = y + 0.5 * _unit(slap) * 0.5 + 0.35 * _unit(wash)
    cav = np.zeros(n)
    for _ in range(1 + int(size > 0.5)):
        _place(cav, amb.tube_bubble((6.0 + 14.0 * size) * 1e-3 * rng.uniform(0.7, 1.1), rng, soft=soft),
               rng.uniform(0.06, 0.14) * SR)
    y = y + 0.9 * _unit(cav) * math.sqrt(0.25 + size)
    for _ in range(int(np.clip(droplets, 0, 40))):
        _place(y, _drop(rng, 0.25 * rng.random(), 2.0, 0.2, soft=soft), 2.0 * math.sqrt(2.0 * rng.uniform(0.05, 0.5) / G_GRAV) * SR,
               float(np.max(np.abs(y))) * rng.uniform(0.15, 0.5))
    return _fin(y, sr, 0.75, 0.0005, 0.05, soft)


# ---------------------------------------------------------------------------------------------
# vessels
# ---------------------------------------------------------------------------------------------

VESSEL_Q = dict(glass=30.0, metal=45.0, ceramic=22.0, plastic=12.0, paper=6.0)
# glass 30; the other materials ordered by wall stiffness / loss.


def vessel_resonance(fill, height: float = 0.15, radius: float = 0.035, neck: float = 0.0,
                     neck_len: float = 0.05, c: float = 343.0):
    """Air resonance (Hz) of a vessel filled to ``fill`` (0..1 of its height).

    Open cylinder (neck = 0): quarter-wave tube, f = c / (4 (L_air + 0.6133 a)) (end correction of
    an unflanged pipe). Odd harmonics 1, 3, 5, 7.
    Bottle (0 < neck < radius): Helmholtz, f = (c / 2 pi) sqrt(A / (V L')), A = pi neck^2,
    V = pi radius^2 L_air, L' = neck_len + 1.43 neck (end corrections: inner end into the cavity,
    flanged by the shoulder, 0.8216 a + outer end unflanged, 0.6133 a). Both rise monotonically as the vessel fills.
    """
    L = height * (1.0 - np.clip(np.asarray(fill, float), 0.0, 1.0))
    if neck <= 0.0 or neck >= radius:
        return c / (4.0 * (L + 0.6133 * radius))
    # + 2 mm of air: a brim-full bottle keeps a finite cavity (guard against V -> 0)
    return c / (2.0 * np.pi) * np.sqrt(neck * neck / (radius * radius * (L + 0.002) * (neck_len + 1.43 * neck)))


def vapour_fraction(temp_c) -> float:
    """Mole fraction of water vapour in air saturated over water at ``temp_c`` (Magnus formula,
    p_sat = 610.94 exp(17.625 T / (T + 243.04)) Pa), at atmospheric pressure."""
    T = float(temp_c)
    return min(610.94 * math.exp(17.625 * T / (T + 243.04)) / P_ATM, 0.95)


def _mix_speed(temp_c) -> float:
    x = vapour_fraction(temp_c)
    g = (1.0 - x) * 1.4 + x * 1.33
    M = (1.0 - x) * 0.02896 + x * 0.01802
    return math.sqrt(g * 8.314 * (273.15 + float(temp_c)) / M)


def air_speed_ratio(temp_c) -> float:
    """Speed of sound in the saturated air space over a liquid at ``temp_c``, relative to 20 C."""
    return _mix_speed(temp_c) / _mix_speed(20.0)


def vapour_bubble_scale(temp_c) -> float:
    """Radius factor of an entrained air bubble at ``temp_c`` against 20 C: vapour takes its partial
    pressure inside the bubble, so the same air needs the volume 1 / (1 - x_v)."""
    return ((1.0 - vapour_fraction(20.0)) / (1.0 - vapour_fraction(temp_c))) ** (1.0 / 3.0)


def _vessel(src, f0, q, harmonics=(1, 3, 5, 7)):
    """Vessel: parallel band-passes at the odd harmonics of the cavity resonance."""
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
     resonance=(0.6, "0..1 how readable the fill pitch is"),
     temperature=(20.0, "liquid temperature in C, 0..95: hot = vapour in the air space (higher pitch) and in the bubbles (larger)"),
     collective=(0.0, "0..1 level of the low collective mode of the bubbles active together"), seed=(0, "variation"))
def pour(sr=DEFAULT_SR, dur=4.0, flow=1.0, fill_start=0.05, fill_end=0.9, height=0.15, radius=0.035,
         neck=0.0, neck_len=0.05, material="glass", resonance=0.6, temperature=20.0, collective=0.0, seed=0):
    """Stream, bubbles and vessel resonance. Bubble rate 1580/s x flow, 300 Hz - 6 kHz. Bubble size
    falls as the column height falls; Chosen: the factor 1.3 -> 0.6, the stream:bubble balance
    0.3:0.7 and the flow ramp min(0.8 s, 0.2 dur). The resonance is deliberately readable."""
    rng = _rng(seed, 4)
    n = _n(dur)
    t = np.arange(n) / SR
    flow = float(np.clip(flow, 0.1, 3.0))
    fill = np.clip(fill_start + (fill_end - fill_start) * t / (n / SR), 0.0, 1.0)
    env = _ar(n, min(0.8, 0.2 * dur), min(0.8, 0.2 * dur))
    T = float(np.clip(temperature, 0.0, 95.0))
    hot = vapour_bubble_scale(T) if T != 20.0 else 1.0
    bub, info = bubble_cloud(dur, rng, HF_POUR_RATE * flow * env, R_6K, R_300, "gauss", cluster=0.2,
                             r_scale=(1.3 - 0.7 * fill) * hot, return_info=True)
    jet = sum(_fit(amb.farnell_water_voice(n / SR, rng, centre=600.0 + 300.0 * v, slew_ms=1.0), n)
              for v in range(3))
    src = 0.7 * _unit(bub) + 0.3 * _unit(jet) * env
    if collective > 0 and info["n"]:
        rm = float(np.exp(np.mean(np.log(info["r"]))))
        w0, beta = (float(v) for v in bubble_params(rm)[:2])
        n_act = HF_POUR_RATE * flow * env / beta
        fc = w0 / (2 * math.pi) * collective_shift(n_act)
        col = _tvbp(src, fc, w0 / (2 * beta))
        src = src + float(np.clip(collective, 0, 1)) * _unit(col) * env
    f0 = vessel_resonance(fill, float(height), float(radius), float(neck), float(neck_len),
                          c=343.0 * air_speed_ratio(T) if T != 20.0 else 343.0)
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
    """A water step: 3100 bubbles/s, 300 Hz - 5 kHz, two-band radii. The chirp is what makes it
    babble (constant-frequency bubbles sound like computer-generated noise). Chosen: the 1.5 Hz surge and cluster 0.3."""
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
     rise=(1.0, "rise-speed multiplier: faster = stronger 'bloop' chirp"),
     flow_lpm=(0.0, "gas flow in litres/min (> 0 sets the rate: flow / bubble volume)"),
     forcing=("window", "window | pinch (neck collapse at release) | merge (two bubbles coalescing): how each bubble starts"),
     amp_law=("moss", "moss (amplitude ~ radius) | pinch (compression at pinch-off grows as 1/radius up to 0.5: "
                      "level nearly flat across sizes) | xue (surface-tension release, ~ radius^-3: small bubbles louder)"),
     transfer=(10.0, "dB: spread of the per-bubble transfer to the listener (+-)"),
     surface=(0, "1 = bubbles near the free surface radiate as dipoles (image source)"),
     vessel=(0.0, "m: radius of the container (rigid wall); 0 = open water"), seed=(0, "variation"))
def bubbles(sr=DEFAULT_SR, dur=2.0, rate=12.0, size=2.0, spread=0.4, cluster=0.5, underwater=0.0, rise=1.0,
            flow_lpm=0.0, forcing="window", amp_law="moss", transfer=10.0, surface=0, vessel=0.0, seed=0):
    rng = _rng(seed, 6)
    r = float(np.clip(size, 0.3, 12.0)) * 1e-3
    sp = float(np.clip(spread, 0.0, 1.0))
    deep = float(underwater) >= 0.5
    rate = float(np.clip(rate, 0.2, 4000.0))
    if flow_lpm > 0:
        rate = float(np.clip(flow_lpm / 60000.0 / (4.0 / 3.0 * math.pi * r ** 3), 0.2, 4000.0))
    if forcing != "window":
        n = _n(dur)
        starts = bubble_times(n, rng, rate, float(cluster))
        y = np.zeros(n)
        for s0, rb in zip(starts, _radii(rng, len(starts), r / (1 + 2 * sp), r * (1 + 2 * sp))):
            _place(y, forced_bubble(float(rb), str(forcing), rng) * excitation_amp(rb, str(amp_law)), s0)
    else:
        y = bubble_cloud(dur, rng, rate, r / (1 + 2 * sp), r * (1 + 2 * sp), "gauss",
                         cluster=float(cluster), depth=(0.5, 2.0) if deep else (0.002, 0.03), rise=float(rise),
                         amp_law=str(amp_law), transfer_db=float(np.clip(transfer, 0.0, 40.0)), image=bool(surface),
                         pitch=float(capacitance_pitch(r, wall=max(float(vessel), 2 * r))) if vessel > 0 else 1.0)
    if not np.any(y):
        y = _fit(hf_bubble(r, 1.0 if deep else 0.01), _n(dur))
    if deep:
        y = _lp(y, 1800.0, 2)                     # underwater listening low-pass
    return _fin(y, sr, 0.7, 0.0005, 0.03)


@sfx("boil", "Boiling water: kettle rumble of collapsing vapour bubbles, then large bubbles bursting in a rolling boil.",
     dur=(4.0, "s"), intensity=(0.8, "0..1: 0.2 simmer rumble, 0.6 onset, 1 rolling boil"),
     ramp=(0.0, "1 = heat up from cold to `intensity` over the duration"),
     size=(0.5, "0..1 vessel size (kettle .. cauldron): lower rumble, larger bubbles"),
     burst=(0.5, "0..1 level of the cap-film bursts at the surface (rising Helmholtz chirps)"), seed=(0, "variation"))
def boil(sr=DEFAULT_SR, dur=4.0, intensity=0.8, ramp=0.0, size=0.5, burst=0.5, seed=0):
    """Bubbles: bursts (cluster) of bubbles. Chosen: the rumble law sin(pi i / 0.9) (a kettle is loudest before it boils and
    quietens at the boil), the rolling-boil rate 160 ((i - 0.4) / 0.6)^2 per second, radii 2-10 mm,
    and the vessel band 900 -> 250 Hz. Each bubble rises 1-10 mm at (2/3) sqrt(g r), drains BOIL_LIFE
    at the surface and bursts (``particles.cap_burst``)."""
    rng = _rng(seed, 7)
    n = _n(dur)
    t = np.arange(n) / SR
    size = float(np.clip(size, 0.0, 1.0))
    i = float(np.clip(intensity, 0.0, 1.0)) * (t / t[-1] if ramp >= 0.5 else np.ones(n))
    rumble = _bp(rng.uniform(-1, 1, n), 900.0 - 650.0 * size, 1.2) * (0.6 + 0.4 * np.abs(_lfn(rng, n, 9.0)))
    rumble = _unit(rumble) * (0.02 + np.sin(np.pi * np.clip(i / 0.9, 0.0, 1.0)))
    rate = 160.0 * np.clip((i - 0.4) / 0.6, 0.0, 1.0) ** 2
    s = 1.0 + size
    bub, info = bubble_cloud(dur, rng, rate, 2e-3 * s, 10e-3 * s, "gauss", cluster=0.6, depth=(0.001, 0.01), return_info=True)
    peak = float(np.max(np.abs(bub)))
    y = 0.35 * rumble + (2.5 * bub / peak if peak > 0 else 0.0)
    if burst > 0:
        cap = np.zeros(n)
        for s0, r in zip(info["starts"], info["r"]):
            rise = rng.uniform(0.001, 0.01) / (2.0 / 3.0 * math.sqrt(9.81 * r))
            life = rng.uniform(*BOIL_LIFE)
            _place(cap, P.cap_burst(float(r), life), s0 + int((rise + life) * SR), rng.lognormal(0.0, 0.3))
        y = y + 2.5 * float(np.clip(burst, 0.0, 1.0)) * cap / max(float(np.max(np.abs(cap))), 1e-9)
    return _fin(y, sr, 0.7, 0.02, 0.05)


BOIL_LIFE = (0.005, 0.05)          # s at the surface before the cap bursts


def _hiss(n, rng):
    """Fire hiss: noise x (slow noise)^4 x 10, high-pass 1 kHz.
    Pre-gain 1 / (3 sigma) as in ``physical.ambience.fire``."""
    return _hp(rng.uniform(-1, 1, n) * np.clip(_lfn(rng, n, 1.0) / 3.0, -1, 1) ** 4 * 10.0, 1000.0)


def _crackles(n, rng, rate, wood: float = 0.5, sigma: float = 0.7, return_count: bool = False):
    """Poisson crackles with log-normal sizes.

    Each crackle: noise x (linear decay over U(0, 30) ms)^2
    through a Q 1 band-pass at 100-1000 Hz ("good for burning wood", fraction ``wood``) or
    1.5-16.5 kHz. Chosen: Poisson timing and log-normal amplitude (sigma) replace the
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
    """Chosen: model: Poisson impulses with
    log-normal amplitudes ringing a 2.5-7 kHz band (each a sub-millisecond vapour burst), plus
    fire crackles for the spits and the 4th-power hiss."""
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
     bright=(0.35, "0..1 top end: low-pass 4..14 kHz"),
     pops=(0.0, "0..1 level of the cap bursts as the bubbles break the surface (rising Helmholtz chirps, mostly above hearing)"),
     seed=(0, "variation"))
def fizz(sr=DEFAULT_SR, dur=3.0, rate=900.0, half_life=1.5, size=0.7, bright=0.35, pops=0.0, seed=0):
    """Bubbles of 0.6-1.8 x ``size`` at the surface (T60 from the damping law: 8 ms at 0.5 mm,
    19 ms at 1 mm), Poisson with
    rate(t) = rate 2^(-t / half_life). Chosen: the rate and half-life defaults; the pitch and decay of each bubble are the sourced law."""
    rng = _rng(seed, 9)
    n = _n(dur)
    t = np.arange(n) / SR
    lam = float(np.clip(rate, 20.0, 5000.0)) * (0.5 ** (t / half_life) if half_life > 0 else 1.0)
    r = float(np.clip(size, 0.3, 1.2)) * 1e-3
    y, info = bubble_cloud(dur, rng, lam, 0.6 * r, 1.8 * r, "gauss", depth=(0.0005, 0.004), return_info=True)
    if info["n"] == 0:
        y = _fit(hf_bubble(r, 0.002), n)
    elif pops > 0:
        pk = float(np.max(np.abs(y)))
        cap = np.zeros(n)
        for s0, rb in zip(info["starts"], info["r"]):
            rise = rng.uniform(0.0005, 0.004) / (2.0 / 3.0 * math.sqrt(G_GRAV * rb))
            p = P.bubble_pop(float(rb), rng)
            _place(cap, p, s0 + int(rise * SR), pk)
        y = y + float(np.clip(pops, 0.0, 1.0)) * cap
    return _fin(_lp(y, 4000.0 + 10000.0 * float(np.clip(bright, 0, 1)), 2), sr, 0.5, 0.002, 0.05)


@sfx("gurgle", "Glugging: large wobbling air bubbles gulped at a rate (bottle emptying, drain, swamp).",
     dur=(2.5, "s"), rate=(3.5, "glugs per second, 0.3..12"),
     size=(0.5, "0..1 bubble size: 6..20 mm radius (550..165 Hz)"), seed=(0, "variation"))
def gurgle(sr=DEFAULT_SR, dur=2.5, rate=3.5, size=0.5, seed=0):
    """Each glug: a non-spherical "tube" bubble (``ambience.tube_bubble``: shape modes radiating at
    2 f_n give the low, full body) followed by a burst of small bubbles. Timing is quantised flow
    (a relaxation oscillator).
    Chosen: 8 trailing bubbles per glug and the 12 % timing jitter."""
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
    """Chosen: gesture: bubble flow at the pouring
    rate falling away, an air-core suction band (noise through a resonance sweeping
    (250 -> 1400 Hz) / (0.6 + 0.8 size), fluttered at 8 -> 22 Hz) growing over the last 45 %, and
    tube-bubble glugs at the end."""
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


@sfx("surf", "Waves breaking on a shore (bubble populations following each crest).",
     dur=(12.0, "s"), distance=(120.0, "m to the break line, 10..500"),
     period_min=(8.0, "shortest gap between waves, s"), period_max=(13.0, "longest gap between waves, s"),
     bubble_rate=(3000.0, "bubbles per second at a crest"),
     sizes=("breaking", "breaking (0.3-20 mm) | plume (surf-zone plume 0.1-5 mm: brighter)"),
     plume=(0.0, "m of bubble plume between the bubbles and the water surface (dulls the highs)"),
     puffs=(0.0, "air injections per second at a full crest (0 = continuous; under ~20/s bursts are heard)"),
     active=(0, "1 = bubbles only during the active window after each crest"), seed=(0, "variation"))
def surf(sr=DEFAULT_SR, dur=12.0, distance=120.0, period_min=8.0, period_max=13.0, bubble_rate=3000.0,
         sizes="breaking", plume=0.0, puffs=0.0, active=0, seed=0):
    """``physical.ambience.surf``. Shorter than 10 s renders one wave."""
    pmin = max(float(period_min), 1.0)
    y = amb.surf(max(float(dur), 4.0), _rng(seed, 12), distance=float(np.clip(distance, 10.0, 500.0)),
                 period=(pmin, max(float(period_max), pmin + 0.1)),
                 bubble_rate=float(np.clip(bubble_rate, 100.0, 8000.0)), single=dur < 10.0,
                 sizes=str(sizes), plume=float(np.clip(plume, 0.0, 2.0)), puffs=float(np.clip(puffs, 0.0, 200.0)),
                 active=bool(active))
    return _fin(_fit(y, _n(dur)), sr, 0.7, 0.05, 0.3)


@sfx("rain_water", "Rain on water drop by drop: each drop by its size class (silent, impact, impact then bubble), level growing with the drop count.",
     dur=(4.0, "s"), rate=(5.0, "rain rate in mm/h, 0.1..300 (drop sizes and count)"),
     wind=(0.0, "m/s, 0..20: the drizzle bubbles (13-16 kHz) die out by 9 m/s"),
     rough=(0, "1 = rough, agitated water (more large-drop bubbles)"),
     area=(1.0, "m2 of water heard, 0.01..10"), seed=(0, "variation"))
def rain_water(sr=DEFAULT_SR, dur=4.0, rate=5.0, wind=0.0, rough=0, area=1.0, seed=0):
    """``physical.ambience.rain_on_water``. Other surfaces: ``nature.rain_on``."""
    y = amb.rain_on_water(float(dur), _rng(seed, 60), rate_mm_h=float(np.clip(rate, 0.1, 300.0)),
                          area_m2=float(np.clip(area, 0.01, 10.0)), wind=float(np.clip(wind, 0.0, 20.0)), rough=bool(rough))
    return _fin(_fit(y, _n(dur)), sr, 0.7, 0.005, 0.05)


@sfx("cave_drips", "Several dripping sources in a reverberant cave.",
     dur=(6.0, "s"), sources=(5, "number of dripping points, 1..12"),
     wet=(0.6, "0..1 cave reverb level"),
     regime=("fast", "fast (a drip every 1.6-9 s per source) | medium (9-90 s) | slow (90-900 s): drip-rate classes of a cave"),
     drop_ml=(0.0, "drop volume in ml (a cave drip: 0.1433); > 0 = impact click lasting the contact time at terminal speed"),
     seed=(0, "variation"))
def cave_drips(sr=DEFAULT_SR, dur=6.0, sources=5, wet=0.6, regime="fast", drop_ml=0.0, seed=0):
    """``physical.ambience.cave_drips``. Its drip periods are 1.2-7 s,
    so a drop is added at the start when a short render would otherwise be silent (in the medium and
    slow regimes most sources do not drip at all within a few seconds: that is the class)."""
    rng = _rng(seed, 13)
    n = _n(dur)
    reg = regime if regime in amb.CAVE_DRIP_PERIODS else "fast"
    y = _fit(amb.cave_drips(n / SR, rng, sources=int(np.clip(sources, 1, 12)), wet=float(np.clip(wet, 0, 1)),
                            regime=reg, drop_ml=float(np.clip(drop_ml, 0.0, 1.0))), n)
    if float(np.max(np.abs(y[: n // 2]))) < 1e-6:
        _place(y, P.drip(rng), 0.05 * SR, max(float(np.max(np.abs(y))), 0.1))
    return _fin(y, sr, 0.7, 0.002, 0.1)


@sfx("waterfall", "Waterfall: near-white roar of countless bubbles with a low rumble that grows with height.",
     dur=(4.0, "s"), height=(6.0, "fall height in m, 0.5..60 (rumble and roar)"),
     distance=(8.0, "m, 1..200: farther = duller, less bubble detail"), seed=(0, "variation"))
def waterfall(sr=DEFAULT_SR, dur=4.0, height=6.0, distance=8.0, seed=0):
    """A waterfall is near-white noise; at a distance the detail is replaced by noise.
    Detail layer: bubbles at the babbling rate with the breaking-wave power law.
    Chosen: the height and distance mappings and the layer balance."""
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
     bubbles=(4.0, "deep bubbles per second, 0..200"),
     surf=(0.0, "0..1 breaking surf heard underwater (20 Hz-5 kHz, peak near 850 Hz)"),
     tide=(0.0, "0..1 tidal-flow band, 40-110 Hz"), wind=(0.0, "0..1 wind-driven sea noise, 50 Hz-20 kHz, -17 dB/decade"),
     ships=(0.0, "0..1 distant shipping, below 100 Hz with a blade-rate line"),
     biophony=(0.0, "0..1 snapping shrimp crackle, 1.5-6 kHz"), seed=(0, "variation"))
def underwater_ambience(sr=DEFAULT_SR, dur=6.0, depth=5.0, bubbles=4.0, surf=0.0, tide=0.0, wind=0.0, ships=0.0,
                        biophony=0.0, seed=0):
    """Bubbles far from the surface (steady pitch, radius-law decay), pitched for
    the listening depth.
    Chosen: the rumble band 40 + 400 / (1 + depth / 5) Hz and the swell rate."""
    rng = _rng(seed, 15)
    n = _n(dur)
    dp = float(np.clip(depth, 0.5, 100.0))
    fc = 40.0 + 400.0 / (1.0 + dp / 5.0)
    rum = _lp(rng.uniform(-1, 1, n), fc, 2) * (1.0 + 0.35 * _lfn(rng, n, 0.4))
    y = _unit(rum)
    if bubbles > 0:
        b = bubble_cloud(dur, rng, float(np.clip(bubbles, 0.0, 200.0)), 1.5e-3, 9e-3, "gauss", cluster=0.6,
                         depth=(0.5, 2.0), water_depth=dp)
        pk = float(np.max(np.abs(b)))
        if pk > 0:
            y = y + _lp(b / pk, 2500.0 / (1.0 + dp / 20.0), 2) * 2.0
    if max(surf, tide, wind, ships, biophony) > 0:
        rb = _rng(seed, 16)
        lay = sea_noise_bands(n, rb)
        for name, g in (("surf", surf), ("tide", tide), ("wind", wind), ("ships", ships), ("biophony", biophony)):
            if g > 0:
                y = y + float(np.clip(g, 0.0, 1.0)) * _unit(lay[name])
    return _fin(y, sr, 0.6, 0.1, 0.3)


SURF_SPECTRUM = ((700.0, 1000.0), (-40.0, 5.0, -50.0))     # Hz corners; dB/decade below, dB rise, dB/decade above
SEA_BANDS = dict(surf=(20.0, 5000.0), tide=(40.0, 110.0), wind=(50.0, 20000.0), ships=(5.0, 100.0), biophony=(1500.0, 6000.0))
WIND_SLOPE = -17.0                                         # dB/decade of wind-driven sea noise
SHIP_BLADE = (6.0, 20.0)                                   # Hz range of a propeller blade rate
SHRIMP_RATE = 400.0                                        # snaps per second of a shrimp bed


def surf_spectrum_db(f):
    """Level (dB re the 1 kHz peak) of breaking-surf noise under water: -40 dB/decade below 700 Hz,
    +5 dB from 700 Hz to 1 kHz, then -50 dB/decade."""
    f = np.maximum(np.asarray(f, float), 1.0)
    (f1, f2), (lo, rise, hi) = SURF_SPECTRUM
    return np.where(f < f1, -rise + lo * np.log10(f / f1),
                    np.where(f < f2, -rise + rise * np.log10(f / f1) / np.log10(f2 / f1), hi * np.log10(f / f2)))


def _shaped(n, rng, db_of_f):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / SR)
    return np.fft.irfft(spec * 10.0 ** (db_of_f(f) / 20.0), n)


def sea_noise_bands(n: int, rng) -> dict:
    """Shallow-water ambient noise layers (48 kHz, unit RMS each): surf, tide, wind, ships, biophony."""
    def band(lo, hi, slope=0.0):
        return lambda f: np.where((f >= lo) & (f <= hi), slope * np.log10(np.maximum(f, 1.0) / lo), -120.0)
    out = dict(surf=_shaped(n, rng, lambda f: np.where((f >= SEA_BANDS["surf"][0]) & (f <= SEA_BANDS["surf"][1]),
                                                        surf_spectrum_db(f), -120.0)))
    out["tide"] = _shaped(n, rng, band(*SEA_BANDS["tide"])) * (1.0 + 0.5 * _lfn(rng, n, 0.2))
    out["wind"] = _shaped(n, rng, band(*SEA_BANDS["wind"], WIND_SLOPE))
    t = np.arange(n) / SR
    blade = rng.uniform(*SHIP_BLADE)
    out["ships"] = _shaped(n, rng, band(*SEA_BANDS["ships"], -6.0)) * (1.0 + 0.6 * np.sin(2 * np.pi * blade * t)) \
        + 0.5 * np.sin(2 * np.pi * blade * 3 * t)
    snaps = np.zeros(n)
    k = int(rng.poisson(SHRIMP_RATE * n / SR))
    np.add.at(snaps, rng.integers(0, n, k), rng.lognormal(0.0, 0.8, k))
    lo, hi = SEA_BANDS["biophony"]
    out["biophony"] = _lp(_hp(snaps, lo, 2), hi, 2)
    return {k: _unit(v) for k, v in out.items()}


# =============================================================================================
# GASES - jets, leaks, whistles, wind
# =============================================================================================

STROUHAL = 0.2
C_AIR = 343.0


def jet_noise(n: int, rng, velocity, diameter: float, power: float = 8.0, u_ref: float = 100.0,
              q: float = 0.7) -> np.ndarray:
    """Turbulent jet / orifice noise at 48 kHz.

    Spectral peak f_p = St U / D with St = 0.2 (Strouhal scaling: centre frequency proportional to
    velocity / diameter).
    Shape: one Q 0.7 band-pass, i.e. 6 dB/octave skirts either side of the peak.
    Level: total power ~ (U / u_ref)^power; the amplitude is (U / u_ref)^(power / 2) divided by
    sqrt(f_p / 1 kHz) because a constant-Q band passes power in proportion to its centre.
    power = 8: free-jet turbulence is quadrupole, power ~ (U / c0)^8; 6 is the dipole law of solid obstacles.
    Chosen: the Q 0.7 skirt shape.
    ``velocity`` (m/s) may be a per-sample array; ``diameter`` in metres.
    """
    u = np.broadcast_to(np.asarray(velocity, float), (n,))
    fp = np.clip(STROUHAL * u / max(float(diameter), 1e-5), 20.0, 0.45 * SR)
    return _tvbp(rng.uniform(-1, 1, n), fp, q) * (np.maximum(u, 0.0) / u_ref) ** (power / 2.0) / np.sqrt(fp / 1000.0)


def _velocity(pressure: float) -> float:
    """Chosen: mapping: pressure 0..1 -> exit speed 40..343 m/s (a jet chokes at the speed of sound)."""
    return 40.0 + (C_AIR - 40.0) * float(np.clip(pressure, 0.0, 1.0))


@sfx("steam", "Steam escaping under pressure: hiss whose pitch follows speed / hole size, with wet sputter.",
     dur=(2.5, "s"), pressure=(0.6, "0..1 (exit speed 40..343 m/s)"),
     orifice=(4.0, "hole diameter in mm, 0.5..50: larger = lower hiss"),
     wet=(0.4, "0..1 sputtering of water droplets"), tone=(9000.0, "Hz low-pass on the top end"),
     seed=(0, "variation"))
def steam(sr=DEFAULT_SR, dur=2.5, pressure=0.6, orifice=4.0, wet=0.4, tone=9000.0, seed=0):
    """``jet_noise`` + a 4th-power hiss modulator and crackle bursts for the wet sputter
    (outgassing of water vapour hisses). Chosen: sputter amounts."""
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
    St U / D (wire-whistle filter)."""
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
    so the audible part is the rising skirt: a thin hiss). Chosen: the valve ticks, the can
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


KETTLE_HOLE, KETTLE_PLATE, KETTLE_DELTA = 3e-3, 1e-3, 0.12      # orifice m, plate m, end correction / orifice
KETTLE_ST = ((2.56, 3.41, 4.25), (0.50, 0.39, 0.43))              # gap / orifice -> Strouhal (on the orifice), Re > 2000
KETTLE_RE2, KETTLE_RE_ON, KETTLE_GAP_MAX = 2000.0, 1000.0, 5.0
KETTLE_DB = ((2000, 2050, 2100, 2150, 2200, 2250, 2300, 2400, 2500, 2600, 2700, 2800, 2850, 3000, 3100, 3200, 3250, 3350, 3400),
             (0.0, -3.2, -2.8, -1.5, -1.5, -0.7, 0.8, 3.2, 6.8, 7.8, 8.5, 10.5, 11.6, 11.2, 10.5, 10.5, 8.5, -8.5, -7.0))
KETTLE_H1_DB, KETTLE_HARM_DB = -6.0, -15.0
STEAM_C = math.sqrt(1.33 * 461.5 * 373.15)                         # m/s
STEAM_NU = 1.23e-5 / 0.59                                          # m2/s


def kettle_helmholtz(gap_mm: float = 10.2, cavity_mm: float = 25.0, c: float = STEAM_C) -> float:
    """Low-flow whistle: the cavity between the plates (diameter cavity, length gap) behind the exit orifice."""
    V = math.pi / 4 * (cavity_mm * 1e-3) ** 2 * gap_mm * 1e-3
    A = math.pi / 4 * KETTLE_HOLE ** 2
    l = KETTLE_PLATE + 2 * KETTLE_DELTA * KETTLE_HOLE
    return c / (2 * math.pi) * math.sqrt(A / (V * l))


def kettle_tone(re, gap_mm: float = 10.2, cavity_mm: float = 25.0, nu: float = STEAM_NU, c: float = STEAM_C):
    """Whistle frequency at orifice Reynolds number ``re``: Helmholtz below 2000, St U / orifice above."""
    re = np.asarray(re, float)
    st = float(np.interp(gap_mm * 1e-3 / KETTLE_HOLE, *KETTLE_ST))
    return np.where(re < KETTLE_RE2, kettle_helmholtz(gap_mm, cavity_mm, c), st * re * nu / KETTLE_HOLE ** 2)


@sfx("kettle", "Whistling kettle: hole tone of the two-plate whistle, a Helmholtz note at low steam flow, then a vortex tone rising with the flow.",
     dur=(5.0, "s"), onset=(2.5, "s for the steam flow to build"),
     flow=(3000.0, "orifice Reynolds number at full boil, 1000..3400 (below 2000: Helmholtz note only)"),
     gap=(10.2, "mm between the whistle's plates, 5..20 (5 x the 3 mm orifice and over: no vortex tone)"),
     cavity=(25.0, "mm whistle diameter, 6..40: the Helmholtz note"),
     breath=(0.4, "0..1 steam noise in the whistle"), rumble=(0.3, "0..1 boiling water underneath"), seed=(0, "variation"))
def kettle(sr=DEFAULT_SR, dur=5.0, onset=2.5, flow=3000.0, gap=10.2, cavity=25.0, breath=0.4, rumble=0.3, seed=0):
    """Re rises to ``flow`` over ``onset``. Below Re 2000 the whistle sounds its Helmholtz note, constant, no harmonics;
    above, a jet-instability tone at constant Strouhal, so its pitch follows the flow, with a second harmonic and
    a level that follows the measured sound power against Re."""
    rng = _rng(seed, 23)
    n = _n(dur)
    t = np.arange(n) / SR
    g, D = float(np.clip(gap, 5.0, 20.0)), float(np.clip(cavity, 6.0, 40.0))
    x = np.clip(t / max(float(onset), 0.05), 0.0, 1.0)
    re = float(np.clip(flow, 1000.0, 3400.0)) * (x * x * (3 - 2 * x)) * (1.0 + 0.01 * _lfn(rng, n, 8.0))
    f = kettle_tone(re, g, D) * (1.0 + 0.002 * _lfn(rng, n, 30.0))
    hi = re >= KETTLE_RE2
    db2 = np.interp(re, *KETTLE_DB) if g * 1e-3 / KETTLE_HOLE < KETTLE_GAP_MAX else np.full(n, -60.0)
    db = np.where(hi, db2, KETTLE_H1_DB + 20 * np.log10(np.clip((re - KETTLE_RE_ON) / 200.0, 1e-3, 1.0)))
    a = _lp(10 ** (db / 20), 12.0, 2)
    h2 = _lp(hi * 10 ** (KETTLE_HARM_DB / 20), 12.0, 2)
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = a * (np.sin(ph) + h2 * np.sin(2 * ph))
    jet = (re / 3000.0) ** 2
    y = y + float(np.clip(breath, 0, 1)) * 2.5 * _tvbp(rng.uniform(-1, 1, n), np.clip(f, 100, 8000), 3.0) * jet
    if rumble > 0:
        r = _bp(rng.uniform(-1, 1, n), 500.0, 1.2) * (0.6 + 0.4 * np.abs(_lfn(rng, n, 9.0)))
        y = y + float(np.clip(rumble, 0, 1)) * 0.5 * _unit(r) * float(np.max(np.abs(y)) + 1e-9)
    return _fin(y, sr, 0.5, 0.02, 0.15)


@sfx("balloon", "Balloon letting its air out: flapping neck raspberry over escaping air, pitch sagging as it empties.",
     dur=(2.0, "s"), size=(0.5, "0..1 balloon size: lower, slower flapping"),
     flutter=(0.7, "0 = plain hiss .. 1 = full flapping raspberry"), seed=(0, "variation"))
def balloon(sr=DEFAULT_SR, dur=2.0, size=0.5, flutter=0.7, seed=0):
    """The neck is a relaxation oscillator : pressure
    opens it, flow drops the pressure, it shuts. Each opening gates the escaping jet
    (``jet_noise``); the shrinking cavity is a Helmholtz resonance that rises as volume falls.
    Chosen: flap rate (40-300 Hz), pressure curve, cavity band."""
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


@sfx("gust", "A gust of wind through an opening, across wires or through leaves.",
     dur=(4.0, "s"), through=("gap", "gap|wire|leaves|canyon|doorway"),
     strength=(0.5, "0..1 peak wind speed. Howls (gap, doorway, canyon) sound only between about 0.25 and 0.6"),
     tone=(1.0, "pitch scale of the wire whistles (thinner wire = higher), 0.3..3"), seed=(0, "variation"))
def gust(sr=DEFAULT_SR, dur=4.0, through="gap", strength=0.5, tone=1.0, seed=0):
    """``physical.ambience.wind_scene`` with its unexposed sources,
    driven by one windspeed hump instead of the scene's random walk. Each emitter keeps its
    law: static bed linear, wires square-law at 600 + 400 w and 1000 + 1000 w Hz (Q 60), doorway
    howls only inside a speed window, leaves lagging by seconds. The render is pre-rolled by the
    emitter's propagation delay (and 6 s for the leaves' 0.07 Hz lag) and cropped.
    Chosen: resting speed 0.12, the sin^2 hump and its 2 Hz flutter."""
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
# FIRE - combustion noise
# =============================================================================================

def extend_fire(p, rng, alpha: float = 3.0, fcut: float = 180.0, sig: float = 10.0, lo: float = 165.0,
                hi: float = 195.0) -> np.ndarray:
    """Spectral bandwidth extension.

    Power-law noise with |N(f)| = f^(-alpha/2) and random phase, high-passed across 165-195 Hz, is
    amplitude-modulated by |p| inside triangular windows of half-width 11.34 ms, blended with the
    low band (F_low 1 -> 0, F_high 0 -> 1 over 165-195 Hz), with a per-window gain beta chosen so
    the power under a Gaussian (180 Hz, sigma 10 Hz) is preserved (positive root of a quadratic).
    alpha: 2.5 (propane burner; brighter) ... 3.0 ... 3.5.
    Stability guard: beta is a ratio whose denominator is the noise power that
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


COMB_STROUHAL, COMB_SPEED, COMB_BETA = 1.0, 10.0, 2.5


def combustion_peak(length, speed: float = COMB_SPEED, st: float = COMB_STROUHAL) -> float:
    """Peak frequency Hz of combustion noise for a flame ``length`` m long: f_p = St U / L."""
    return float(st) * float(speed) / max(float(length), 1e-3)


def combustion_spectrum(f, f_peak: float, beta: float = COMB_BETA) -> np.ndarray:
    """Power spectrum shape of combustion noise: x^2 / (1 + x^(2 + beta)), x = f / f_peak; rises as f^2
    (monopole, p = dI/dt) and falls as f^-beta above the peak. 1/2 at the peak."""
    x = np.asarray(f, float) / float(f_peak)
    return x * x / (1.0 + x ** (2.0 + float(beta)))


def combustion_roar(n: int, rng, area=1.0, depth=0.5, f_turb: float = 60.0, alpha: float = 3.0,
                    peak: float | None = None, beta: float = COMB_BETA) -> np.ndarray:
    """Flame roar at 48 kHz: p = dI/dt, 30 Hz high-pass, ``extend_fire``.

    Surrogate flux:
    I(t) = area(t) (1 + depth(t) turbulence(t)), turbulence = smooth noise band-limited to
    ``f_turb`` (< 180 Hz). ``area`` and ``depth`` are numbers or per-sample arrays. A steady flame
    is quiet: only changes of I make sound. ``peak`` (Hz): the turbulence instead has the spectrum
    1 / (1 + (f/peak)^(2+beta)), so p follows ``combustion_spectrum`` over the whole band (no extension). Chosen: this surrogate and every value fed to it.
    """
    area = np.broadcast_to(np.asarray(area, float), (n,))
    depth = np.broadcast_to(np.asarray(depth, float), (n,))
    if peak is not None:
        nf = n + 2 * SR // 10
        mag = 1.0 / np.sqrt(1.0 + (np.fft.rfftfreq(nf, 1.0 / SR) / float(peak)) ** (2.0 + float(beta)))
        turb = np.fft.irfft(np.fft.rfft(rng.standard_normal(nf)) * mag, nf)[:n]
        flux = area * (1.0 + depth * turb / (np.std(turb) + 1e-30))
        return _hp(np.gradient(flux) * SR, 30.0)
    flux = area * (1.0 + depth * _lfn(rng, n, min(float(f_turb), 150.0)))
    return extend_fire(_hp(np.gradient(flux) * SR, 30.0), rng, alpha)


def _lapping(n, rng):
    """Lapping: bp 30 Hz Q 5 x100, hip 25 x2, clip +-0.9, x0.6."""
    lap = amb.hip(amb.hip(amb.bp(rng.uniform(-1, 1, n), 30.0, 5.0) * 100.0, 25.0), 25.0)
    return np.clip(lap, -0.9, 0.9) * 0.6


# kind -> area(t) shape, turbulence depth, f_turb Hz, alpha, crackles/s, hiss, lapping, jet (U m/s, D m, level)
# the differences between flames are the shape of I(t); alpha 2.5 for the gas flames (propane burner).
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
    """Flame area / fuel control and a turbulence-depth multiplier (gestures)."""
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


def _flame(kind: str, n: int, rng, rate: float = 1.0, crackle=None, wood: float = 0.5, peak=None):
    k = FLAME_KINDS[kind]
    area, dmod = _area(k["area"], n, rng, rate)
    y = _unit(combustion_roar(n, rng, area, k["depth"] * dmod, k["f_turb"], k["alpha"], peak))
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
     length=(0.0, "flame length m: 0 = the kind's own roar; > 0 = combustion-noise spectrum peaking at 10 / length Hz (0.05 m: 200 Hz), falling as f^-2.5 above"),
     seed=(0, "variation"))
def flame(sr=DEFAULT_SR, kind="campfire", dur=3.0, rate=1.0, length=0.0, seed=0):
    kind = str(kind) if str(kind) in FLAME_KINDS else "campfire"
    n = _n(dur)
    peak = combustion_peak(float(np.clip(length, 0.005, 5.0))) if length > 0 else None
    y = _flame(kind, n, _rng(seed, 30), float(np.clip(rate, 0.05, 10.0)), peak=peak)
    one_shot = kind in ("fireball", "flamethrower")
    return _fin(y, sr, FLAME_KINDS[kind]["level"], 0.003 if one_shot else 0.05, 0.1)


@sfx("fire", "Wood fire: combustion roar, Poisson crackles, hiss of outgassing; size, crackle rate, damp wood, wind.",
     dur=(4.0, "s"), size=(0.4, "0..1: 0 small hearth fire, 1 big blaze (1..4 independent flame fronts, more roar)"),
     crackle=(12.0, "crackles per second, 0..150"),
     wet=(0.2, "0..1 damp wood: more hiss, louder and rarer low pops"),
     wind=(0.2, "0..1 wind ruffling the flames (more roar, gusting)"), seed=(0, "variation"),
     whine=(0.0, "0..1 pitched whistles of gas escaping through trapped sap"),
     woof=(0.0, "0..1 low air induction under 80 Hz, puffing with the flames"),
     clatter=(0.0, "0..1 burnt pieces settling and falling"),
     detail=("full", "full: every component; medium: hiss, roar and crackle only (a fire heard from afar)"))
def fire(sr=DEFAULT_SR, dur=4.0, size=0.4, crackle=12.0, wet=0.2, wind=0.2, seed=0, whine=0.0, woof=0.0, clatter=0.0,
         detail="full"):
    """Per flame front one shared slow control drives roar, crackle rate and hiss together (one noise per
    generator keeps them coherent); a big fire is up to 4 fronts with their own noises. Whine: relaxation
    oscillations of outgassing; woof: 3-80 Hz air induction following the flames; clatter: wooden impacts."""
    rng = _rng(seed, 31)
    n = _n(dur)
    size, wet, wind = (float(np.clip(v, 0.0, 1.0)) for v in (size, wet, wind))
    full = str(detail) != "medium"
    units = 1 + int(round(3 * size))
    y = np.zeros(n)
    cs = np.zeros(n)
    for _ in range(units):
        c = np.clip(1.0 + 0.4 * _lfn(rng, n, 1.0) + wind * 0.8 * np.clip(_lfn(rng, n, 0.5), 0, None) ** 2, 0.2, 3.0)
        cs += c / units
        roar = _unit(combustion_roar(n, rng, c, 0.35 + 0.3 * size + 0.5 * wind, 50.0 + 60.0 * size, 3.0))
        cr = _crackles(n, rng, float(np.clip(crackle, 0.0, 150.0)) * (1.0 - 0.5 * wet) / units * c ** 2,
                       wood=0.5 + 0.4 * wet, sigma=0.7 + 0.5 * wet)
        hs = _unit(_lp(_hiss(n, rng), 9000.0, 2)) * c
        lap = _unit(_lapping(n, rng))
        y += (0.4 + 0.8 * size + 0.4 * wind) * roar + 0.35 * _unit(cr) * math.sqrt(min(crackle, 40.0) / 12.0 + 1e-6) \
            + (0.2 + 0.6 * wet) * hs + (0.3 if full else 0.0) * lap
    if full:
        r2 = _rng(seed, 33)
        ref = float(np.sqrt(np.mean(y * y))) + 1e-12
        for amount, part in ((whine, _fire_whine), (woof, _fire_woof), (clatter, _fire_clatter)):
            a = float(np.clip(amount, 0.0, 1.0))
            if a > 0:
                p = part(n, r2, cs, size)
                y = y + a * FIRE_PART_GAIN[part.__name__] * ref * p / (float(np.sqrt(np.mean(p * p))) + 1e-12)
    return _fin(y, sr, 0.6 + 0.15 * size, 0.05, 0.1)


FIRE_WHINE_HZ, FIRE_WHINE_RISE, FIRE_WHINE_S, FIRE_WHINE_RATE = (500.0, 1800.0), 1.25, (0.3, 1.5), 0.4
FIRE_CLATTER_RATE = 0.3
FIRE_PART_GAIN = {"_fire_whine": 0.5, "_fire_woof": 1.0, "_fire_clatter": 0.8}


def _fire_whine(n, rng, c, size):
    """Episodes of a relaxation oscillator: pressure charging exponentially, released each cycle, the pitch
    rising as the pocket empties."""
    y = np.zeros(n)
    t = rng.uniform(0.0, min(1.0 / FIRE_WHINE_RATE, 0.5 * n / SR))
    while t < n / SR:
        m = int(rng.uniform(*FIRE_WHINE_S) * SR)
        f0 = rng.uniform(*FIRE_WHINE_HZ)
        f = f0 * FIRE_WHINE_RISE ** (np.arange(m) / m) * (1.0 + 0.01 * _lfn(rng, m, 8.0))
        ph = np.cumsum(f) / SR % 1.0
        s = (1.0 - np.exp(-4.0 * ph)) / (1.0 - math.exp(-4.0)) - 0.5
        _place(y, _hp(s, 200.0, 2) * np.sin(np.pi * np.arange(m) / m) ** 2, t * SR)
        t += m / SR + rng.exponential(1.0 / FIRE_WHINE_RATE)
    return y


def _fire_woof(n, rng, c, size):
    """Air drawn in under the flames: noise kept in 3-80 Hz, its level following the flame control."""
    return _hp(_lp(rng.standard_normal(n), 80.0, 2), 3.0, 2) * c ** 2


def _fire_clatter(n, rng, c, size):
    """A piece of burnt wood settling: a short cascade of wooden impacts, each lighter and quicker."""
    from . import contact as K
    y = np.zeros(n)
    t = rng.uniform(0.0, min(1.0 / FIRE_CLATTER_RATE, 0.5 * n / SR))
    while t < n / SR:
        dt, v = 0.0, rng.uniform(1.0, 2.5)
        for _ in range(int(rng.integers(2, 6))):
            hit = K.impact(SR, material="wood", shape="bar", size=rng.uniform(0.05, 0.15 + 0.25 * size), velocity=v,
                           seed=int(rng.integers(1 << 30)))
            _place(y, np.asarray(hit, float), (t + dt) * SR, v / 2.5)
            dt += rng.uniform(0.04, 0.15)
            v *= rng.uniform(0.4, 0.7)
        t += dt + rng.exponential(1.0 / FIRE_CLATTER_RATE)
    return y


@sfx("match", "Striking a match: scrape on the strip, flare of ignition, then a small flame.",
     dur=(1.6, "s"), strike=(0.12, "s length of the scrape"), rough=(0.6, "0..1 grain of the striking strip"),
     seed=(0, "variation"))
def match(sr=DEFAULT_SR, dur=1.6, strike=0.12, rough=0.6, seed=0):
    """Flare = the fireball gesture of ``combustion_roar`` plus fire hiss; the small flame is the
    candle. Chosen: the scrape (band-passed noise gated by a 400-900 Hz grain train; solid
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
# PLASMA / ELECTRICITY
# =============================================================================================

def _formant(x, f: float = 1.0):
    """``spark6formant``: parallel bp at 4600 Q5 (x2), 7200 Q5
    (x2), 480 Q7 (x1.2), 720 Q8 (x2.5), then bp 2500 Q0.5. ``f`` scales every centre."""
    y = (2.0 * _bp(x, 4600.0 * f, 5.0) + 2.0 * _bp(x, 7200.0 * f, 5.0) + 1.2 * _bp(x, 480.0 * f, 7.0)
         + 2.5 * _bp(x, 720.0 * f, 8.0))
    return _bp(y, 2500.0 * f, 0.5)


def _snap(rng, t_noise: float, t_sweep: float = 0.010, f_top: float = 7000.0) -> np.ndarray:
    """Snap, before the formant: noise x 0.5 v^4 (v 1 -> 0 over ``t_noise``) plus a sine
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
    """Random gate: low-passed noise against a threshold, about
    3 Hz, slewed. ``duty`` is the fraction of time it is open (the threshold is that quantile)."""
    if duty >= 1.0:
        return np.ones(n)
    g = _lfn(rng, n, rate)
    return _lp((g > np.quantile(g, 1.0 - max(duty, 0.02))).astype(float), 60.0)


@sfx("arc", "Electric arc: broadband buzz re-striking on every half cycle of the mains, sputtering in and out.",
     dur=(2.0, "s"), mains=(50.0, "Hz mains frequency (50 or 60): the buzz is at twice this"),
     stability=(0.7, "0..1 fraction of time the arc holds (1 = steady welding arc)"),
     tone=(1.0, "formant scale 0.5..1.5"),
     click=(0.0, "0..1 level of the chirp clicks driven at twice the mains (0 = none)"), seed=(0, "variation"))
def arc(sr=DEFAULT_SR, dur=2.0, mains=50.0, stability=0.7, tone=1.0, click=0.0, seed=0):
    """Noise x |sin(2 pi f t)|^3 (conduction around each current peak, so the buzz is at 2 f:
    hum is 50-60 Hz plus 100-120 Hz) with a snap at every re-strike, the random gate, and
    the spark formant. Chosen: the cube law and the snap level."""
    rng = _rng(seed, 41)
    n = _n(dur)
    f0 = float(np.clip(mains, 20.0, 400.0))
    t = np.arange(n) / SR
    x = rng.uniform(-1, 1, n) * np.abs(np.sin(2 * np.pi * f0 * t)) ** 3
    for k in range(int(2 * f0 * n / SR)):
        _place(x, _snap(rng, 0.003, 0.002), (k + 0.15) / (2 * f0) * SR + rng.uniform(-20, 20), rng.uniform(0.2, 0.6))
    gate = _gate(rng, n, float(stability))
    y = _formant(x * gate, float(np.clip(tone, 0.5, 1.5)))
    y = y + 0.08 * float(np.max(np.abs(y))) * np.sin(4 * np.pi * f0 * t)
    if click > 0:
        c = arc_chirps(n, 2 * f0 - CHIRP_DETUNE, _rng(seed, 43)) * gate
        y = y + float(np.clip(click, 0, 1)) * c * float(np.sqrt(np.mean(y ** 2)) / (np.sqrt(np.mean(c ** 2)) + 1e-12))
    return _fin(y, sr, 0.65, 0.002, 0.03)


CHIRP_OFFSET, CHIRP_SCALE, CHIRP_HP = 0.1, 12.0, 300.0
CHIRP_DETUNE, CHIRP_MOD_HZ, CHIRP_MOD_GAIN, CHIRP_MOD_FLOOR = 0.2, 0.1, 700.0, 3.0


def arc_chirps(n: int, f: float, rng) -> np.ndarray:
    """Chirp pulses: a reversed phasor at ``f`` times a slow random m > 1, the part below 1 kept,
    phase (x + offset)^2 x scale cycles, cosine, high-pass. m sets the pulse length, so the click colour."""
    m = CHIRP_MOD_FLOOR + CHIRP_MOD_GAIN * np.maximum(_lp(_lp(rng.uniform(-1, 1, n), CHIRP_MOD_HZ), CHIRP_MOD_HZ), 0.0)
    x = (1.0 - (f * np.arange(n) / SR + rng.uniform()) % 1.0) * m
    x = np.where(x < 1.0, x, 0.0)
    return _hp(np.cos(2 * np.pi * CHIRP_SCALE * (x + CHIRP_OFFSET) ** 2), CHIRP_HP)


def mains_harmonics(n: int, rng, f0: float, bright: float = 0.3, beat: float = 0.4, top: float = 3000.0):
    """Harmonic series on exactly ``f0`` at 48 kHz. Amplitude 1 / k^(2.6 - 1.6 bright), the 2nd
    harmonic doubled (magnetostriction is rectified: 50-60 Hz plus 100-120 Hz) and split
    into a pair ``beat`` Hz apart (99.8 / 100.2 Hz beat at 0.4 Hz).
    Chosen: the roll-off exponent."""
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
    """Slow random level (noise -> low-pass -> squared) and a positive-feedback
    comb of 10-30 ms (here exactly 1 / f0, which is 20 ms at 50 Hz). Low-passed at 4 kHz as in
    ``physical.ambience.hum``. Chosen: LFO depth 15 %."""
    rng = _rng(seed, 42)
    n = _n(dur)
    f0 = float(np.clip(mains, 20.0, 400.0))
    g = float(np.clip(resonance, 0.0, 0.9))
    # pre-roll until the comb has built up (g^k < 1e-4)
    pre = int(SR * max(0.1, (math.log(1e-4) / math.log(g) if g > 0 else 1.0) / f0))
    y = mains_harmonics(n + pre, rng, f0, float(np.clip(harmonics, 0, 1)), float(max(beat, 0.0)))
    y = y * np.clip(1.0 + 0.15 * (_lfn(rng, n + pre, 0.5) ** 2 - 1.0), 0.6, 1.5)
    d = int(round(SR / f0))
    a = np.zeros(d + 1)
    a[0], a[-1] = 1.0, -g
    return _fin(_lp(lfilter([1.0], a, y), 4000.0, 2)[pre:], sr, 0.5, 0.03, 0.05)


@sfx("tesla", "Tesla coil / stun-gun zap: a rapid train of spark discharges with corona hiss.",
     dur=(1.0, "s"), rate=(120.0, "discharges per second, 10..600 (the pitch of the buzz)"),
     chaos=(0.4, "0..1 timing and level irregularity"), sweep=(0.0, "octaves the rate glides over the duration (+ up, - down)"),
     tone=(1.0, "formant scale 0.5..1.5"), seed=(0, "variation"))
def tesla(sr=DEFAULT_SR, dur=1.0, rate=120.0, chaos=0.4, sweep=0.0, tone=1.0, seed=0):
    """A train of snaps (2-6 ms) through the spark formant. Chosen: the corona layer
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
    """A neon lamp is a relaxation oscillator. Buzz = ``mains_harmonics``
    (bright, high-passed). Chosen: dwell times (lit: exponential, mean 0.05 + 0.8 (1 - flicker)
    s; dark: 20-150 ms), the re-strike tick (snap) and the 2.8 kHz glass ping."""
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


PRECURSOR_S, PRECURSOR_RATE, PRECURSOR_LEVEL = 0.3, 60.0, 0.25


@sfx("lightning", "Lightning strike: close crack (electric snap), then multi-layer thunder.",
     dur=(6.0, "s"), distance=(150.0, "m to the strike, 20..6000: close = sharp bang, far = rumble only"),
     delay=(0.0, "1 = insert the true sound travel time (distance / 343 s) before it starts"),
     seed=(0, "variation"), strokes=(1, "1..3 return strokes down the same channel, about 60 ms apart"),
     model=(0, "thunder: 0 layered patch, 1 the tortuous channel of N-waves"),
     precursor=(0.0, "0..1 crackle of discharges in the 0.3 s before the strike"))
def lightning(sr=DEFAULT_SR, dur=6.0, distance=150.0, delay=0.0, seed=0, strokes=1, model=0, precursor=0.0):
    """Crack: a snap scaled up (noise x 0.5 v^4 over 2 s + 7 kHz -> 20 Hz sweep in 10 ms through the
    spark formant) for the bang within 500 m, then the thunder. Later strokes follow the first one's channel:
    the same signal, weaker, shorter and so higher (model 0), or the channel's own strokes (model 1)."""
    k_st = int(np.clip(strokes, 1, 3))
    if k_st > 1 and int(model) == 0:
        y1 = lightning(sr, dur, distance, delay, seed, 1, 0, precursor)
        y = np.zeros(y1.size)
        i_t = np.arange(y.size) / sr
        for e, a, dt in list(zip(amb.STROKE_E, amb.STROKE_A, amb.STROKE_DT))[:k_st]:
            y += a * np.interp((i_t - dt) / math.sqrt(e), i_t, y1, left=0.0, right=0.0)
        return y * 0.85 / (np.max(np.abs(y)) + 1e-12)
    rng = _rng(seed, 45)
    n = _n(dur)
    d = float(np.clip(distance, 20.0, 6000.0))
    if int(model) == 1:
        th = amb.thunder_channel(rng, dur=max(n / SR, 1.25), distance_m=d, strokes=k_st)
    else:
        th = amb.thunder(rng, dur=max(n / SR, 1.25), distance=float(np.clip(d / 3000.0, 0.0, 1.0)))
    th = _fit(th, n)
    y = th / (float(np.max(np.abs(th))) + 1e-12)
    g = max(0.0, 1.0 - d / 500.0) ** 2
    if g > 0:
        crack = _formant(_snap(rng, min(2.0, n / SR), 0.010), 0.8)
        y = y + 1.6 * g * _fit(_lp(crack, 12000.0 / (1.0 + d / 100.0), 2) / (float(np.max(np.abs(crack))) + 1e-12), n)
    pc = float(np.clip(precursor, 0.0, 1.0))
    if pc > 0:
        lead = int(PRECURSOR_S * SR)
        pre = np.zeros(n + lead)
        prng = _rng(seed, 46)
        t = 0.0
        while t < PRECURSOR_S:
            _place(pre, _formant(np.concatenate([_snap(prng, prng.uniform(0.001, 0.003), 0.001), np.zeros(240)]), 1.2),
                   t * SR, prng.uniform(0.3, 1.0) * (t / PRECURSOR_S) ** 2)
            t += prng.exponential(1.0 / (PRECURSOR_RATE * (0.2 + t / PRECURSOR_S)))
        pre = _lp(pre, 12000.0 / (1.0 + d / 100.0), 2)
        pre *= pc * PRECURSOR_LEVEL * float(np.max(np.abs(y))) / (float(np.max(np.abs(pre))) + 1e-12)
        pre[lead:] += y
        y = pre[:n]
    if delay >= 0.5:
        y = np.concatenate([np.zeros(min(int(d / C_AIR * SR), n // 2)), y])[:n]
    return _fin(y, sr, 0.85, 0.0005, 0.3)


# =============================================================================================
# PHASE TRANSITIONS - ice
# =============================================================================================

# Isotropic polycrystalline ice Ih at -16 C (Brillouin): rho 919.5 kg/m^3, V_L 3844.94 m/s, V_T 1956.9 m/s; nu = (V_L^2 - 2 V_T^2) /
# (2 (V_L^2 - V_T^2)) = 0.325, E = 2 rho V_T^2 (1 + nu) = 9.33 GPa (derived here).
ICE_E, ICE_RHO, ICE_NU = 9.33e9, 919.5, 0.325
ICE_C_MAX = 1957.0
ICE_DEBYE = {False: (6.9e-16, 54810.0, 0.009), True: (3.95e-10, 23305.0, 0.012)}
ICE_ETA_FLOOR, ICE_LOSS = 0.002, 0.0045
ICE_C_PARALLEL = 0.2


def ice_eta(f, temp_c: float = -10.0, salt: bool = False, c_parallel: bool = False):
    """Loss factor of ice at frequency f: a Debye relaxation tau = tau0 exp(Q / RT), pure or NaCl-doped, over a floor.
    ``c_parallel``: crystal c axis along the bending (neutral fibre), peak scaled by ICE_C_PARALLEL."""
    tau0, q, top = ICE_DEBYE[bool(salt)]
    top = top * (ICE_C_PARALLEL if c_parallel else 1.0)
    wt = 2 * np.pi * np.asarray(f, float) * tau0 * math.exp(q / (8.314 * (273.15 + float(temp_c))))
    return top * 2.0 * wt / (1.0 + wt * wt) + ICE_ETA_FLOOR


def ice_chirp(distance: float = 60.0, thickness: float = 0.05, loss: float = ICE_LOSS,
              return_freq: bool = False, temp: float | None = None, salt: bool = False, c_parallel: bool = False):
    """The dispersive "pew" of a crack in lake ice, heard ``distance`` m away, at 48 kHz.

    Thin-plate (Kirchhoff) flexural waves: w = k^2 a with a = sqrt(D / (rho h)) =
    h sqrt(E / (12 rho (1 - nu^2))), so the group speed c_g = 2 sqrt(w a) grows with frequency and
    high frequencies arrive first. A component of frequency f arrives at t = d / c_g, hence the
    instantaneous frequency falls as

        f(t) = d^2 / (8 pi a t^2),   t >= t0 = d / c_max

    where c_max caps the group speed (thin-plate theory fails once the wavelength nears the
    thickness), which gives the top frequency c_max^2 / (8 pi a): about 3 kHz for 5 cm ice and
    500 Hz for 30 cm - thin ice sings, thick ice booms. Stationary-phase amplitude of a dispersed
    impulse: A ~ sqrt(|df/dt|) ~ (t0 / t)^1.5, times exp(-loss 2 pi f (t - t0)), loss = eta / 2 at the
    relaxation peak, or ``ice_eta(f, temp, salt) / 2`` when ``temp`` (C) is given.
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
    if temp is not None:
        loss = ice_eta(f, temp, salt, c_parallel) / 2.0
    elif c_parallel:
        loss = loss * ICE_C_PARALLEL
    y = (t0 / t) ** 1.5 * np.exp(-loss * 2 * np.pi * f * (t - t0)) * np.sin(2 * np.pi * c * (1.0 / t0 - 1.0 / t))
    k = min(48, len(y) // 4)
    y[:k] *= np.linspace(0.0, 1.0, k)
    y[-k:] *= np.linspace(1.0, 0.0, k)
    return (y, f) if return_freq else y


# Floating lake ice as fitted to a skating recording: E 8.5 GPa, nu 0.33, rho 917, water 1000 kg/m^3 -> 725 Hz tone for 4.3 cm of ice at c = 329 m/s.
LAKE_E, LAKE_NU, LAKE_RHO, WATER_RHO = 8.5e9, 0.33, 917.0, 1000.0


def _acfw_g0(w, h, c, depth):
    """The floating plate's response denominator for a load moving at the speed of sound,
    G0(w) = D (w/c)^4 + rho_w g - rho_i h w^2 - c rho_w w coth(w H / c)."""
    D = LAKE_E * h ** 3 / (12.0 * (1.0 - LAKE_NU ** 2))
    x = w * depth / c
    deep = np.real(x) > 20.0                                                   # coth -> 1 (and no tanh overflow)
    coth = np.where(deep, 1.0, 1.0 / np.tanh(np.where(deep, 1.0, x)))
    return D * (w / c) ** 4 + WATER_RHO * 9.81 - LAKE_RHO * h * w * w - c * WATER_RHO * w * coth


def acfw_frequency(thickness: float, c: float = 329.0, depth: float = 10.0) -> float:
    """Frequency of the air-coupled flexural wave (the ice "sings" where the flexural phase speed equals c):
    the root of G0 with k = w/c. Thin ice sings high: ~725 Hz at 4.3 cm, ~60 Hz at 80 cm."""
    h = float(np.clip(thickness, 0.005, 3.0))
    lo, hi = 2 * math.pi * 1.0, 2 * math.pi * 40000.0
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if _acfw_g0(mid, h, c, depth) < 0 else (lo, mid)
    return math.sqrt(lo * hi) / (2 * math.pi)


def ice_sing(distance: float = 60.0, thickness: float = 0.05, c: float = C_AIR, depth: float = 10.0,
             lead: float = 0.3, tail: float = 0.1) -> np.ndarray:
    """The air-coupled flexural wave of an impulse ``distance`` m away on floating ice: a near-monochromatic
    tone that builds up ahead of the air-borne crack and stops on it (plate velocity
    i w e^{-i w r/c} / G0(w + i gamma), the air wave lands ``lead`` s into the output).

    gamma is a free damping; here 1/gamma = how far the tone runs ahead of the air wave,
    r (1/c - 1/c_g), with the flexural group speed c_g at the coupled frequency (derived here)."""
    h = float(np.clip(thickness, 0.005, 3.0))
    r = float(np.clip(distance, 2.0, 2000.0))
    f0 = acfw_frequency(h, c, depth)
    w0, dw = 2 * math.pi * f0, 2 * math.pi * f0 * 1e-3
    # c_g = dw/dk on the dispersion branch through (w0, w0/c): implicit derivative of G(k, w) = 0
    D = LAKE_E * h ** 3 / (12.0 * (1.0 - LAKE_NU ** 2))
    k0 = w0 / c
    dG_dk = 4 * D * k0 ** 3 + WATER_RHO * w0 * w0 / (k0 * k0)                  # deep water
    dG_dw = -2 * LAKE_RHO * h * w0 - 2 * WATER_RHO * w0 / k0
    cg = -dG_dk / dG_dw
    tau = float(np.clip(r * (1.0 / c - 1.0 / max(cg, c * 1.01)), 0.01, lead))
    n = 1 << int(np.ceil(np.log2((lead + tail) * SR)))
    w = 2 * math.pi * np.fft.rfftfreq(n, 1.0 / SR)
    w[0] = dw
    H = 1j * w * np.exp(-1j * w * lead) / _acfw_g0(w + 1j / tau, h, c, depth)
    H[w < 2 * math.pi * 20.0] = 0.0                                            # drop the buoyancy (static) part
    y = np.fft.irfft(H, n)[: int((lead + tail) * SR)]
    return y / (float(np.max(np.abs(y))) + 1e-12)


def _tick(rng, ms: float, fc: float, q: float = 3.0):
    """Brittle micro-crack: a 4th-power noise burst of ``ms`` through a band-pass."""
    m = max(16, int(ms * 1e-3 * SR))
    return _bp(np.concatenate([rng.uniform(-1, 1, m) * (1.0 - np.arange(m) / m) ** 4, np.zeros(4 * m)]), fc, q)


@sfx("ice_crack", "Lake ice cracking: the falling 'pew' of the dispersed plate wave, then the sharp brittle crack.",
     dur=(1.5, "s"), thickness=(0.05, "ice thickness in m, 0.01..1: thin = high laser-like pew, thick = low boom"),
     distance=(60.0, "m to the crack, 5..1000: farther = longer chirp"),
     cracks=(3, "number of cracks in the event, 1..20"),
     brittle=(0.6, "0..1 level of the air-borne snap"),
     sing=(0.5, "0..1 level of the air-coupled tone that rises just before each snap (the skaters' 'laser')"),
     seed=(0, "variation"), temp=(99.0, "ice temperature in C, -50..0; 99 = loss fixed at its relaxation peak"),
     salt=(0, "1 = sea or salted ice (NaCl relaxation, much lighter loss in the audible band when cold)"),
     c_axis=(0, "1 = crystal c axes lying along the ice: relaxation peak / 5, longer ring"))
def ice_crack(sr=DEFAULT_SR, dur=1.5, thickness=0.05, distance=60.0, cracks=3, brittle=0.6, sing=0.5, seed=0,
              temp=99.0, salt=0, c_axis=0):
    """Each crack: ``ice_chirp`` through the ice, then the snap through the air, which arrives later
    (d / 343 s) because the plate wave is faster, preceded by the air-coupled flexural tone ``ice_sing``."""
    tc = None if float(temp) > 50.0 else float(np.clip(temp, -50.0, 0.0))
    rng = _rng(seed, 50)
    n = _n(dur)
    y = np.zeros(n)
    t = 0.0
    for k in range(int(np.clip(cracks, 1, 20))):
        d = float(np.clip(distance, 5.0, 1000.0)) * rng.uniform(0.7, 1.3)
        g = rng.uniform(0.4, 1.0) if k else 1.0
        _place(y, ice_chirp(d, float(thickness), temp=tc, salt=bool(salt), c_parallel=bool(c_axis)), (t + d / ICE_C_MAX) * SR, g)
        snap = _lp(_tick(rng, rng.uniform(3.0, 10.0), rng.uniform(1500.0, 5000.0), 1.5), 12000.0 / (1.0 + d / 80.0), 2)
        _place(y, snap / (float(np.max(np.abs(snap))) + 1e-12), (t + d / C_AIR) * SR, g * float(np.clip(brittle, 0, 1)) * 0.8)
        if sing > 0:
            i0 = int((t + d / C_AIR - 0.3) * SR)                # the tone's air-wave instant (0.3 s in) on the snap
            _place(y, ice_sing(d, float(thickness))[max(0, -i0):], max(0, i0), g * float(np.clip(sing, 0, 1)) * 0.6)
        t += rng.exponential(0.12)
    return _fin(y, sr, 0.75, 0.0005, 0.05)


@sfx("ice_cubes", "Ice cubes clinking in a glass of drink.",
     dur=(1.5, "s"), cubes=(3, "number of cubes, 1..12"), glass=(1.0, "glass pitch scale, 0.5..2 (larger glass = lower)"),
     liquid=(0.6, "0..1 fill: damps and lowers the glass, adds a little slosh"),
     energy=(0.7, "0..1 how hard the glass is swirled"), seed=(0, "variation"))
def ice_cubes(sr=DEFAULT_SR, dur=1.5, cubes=3, glass=1.0, liquid=0.6, energy=0.7, seed=0):
    """Collisions settle like a decaying swirl (rate falling exponentially). Each is a short contact
    pulse (``physical.core.contact_pulse``) driving a glass body (modal table ``sy_vase`` through
    ``physical.modal``) plus the cube's own click: ice is stiff and ringing, like a wood or metal
    bar. Slosh: bubbles.
    Liquid loading: ``glass_fill_ratio``."""
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
    modes = M.resolve("sy_vase", f_scale=float(np.clip(glass, 0.5, 2.0)) * float(glass_fill_ratio(lq)), d_scale=1.0 + 1.5 * lq)
    y = _unit(M.render_modes(force, modes)) + 0.7 * _unit(clicks)
    if lq > 0:
        y = y + 0.25 * lq * _unit(bubble_cloud(dur, rng, 25.0 * en * np.exp(-t / (0.5 * n / SR)), 0.8e-3, 4e-3, cluster=0.5))
    return _fin(y, sr, 0.65, 0.0005, 0.05)


JUNDT_N, JUNDT_ALPHA = 5.5, (897.0 / 416.0) ** 2 - 1.0


def glass_fill_ratio(h, n: float = JUNDT_N, alpha: float = JUNDT_ALPHA):
    """Pitch ratio of a glass filled to fraction ``h``: 1 / sqrt(1 + alpha h^n)."""
    return 1.0 / np.sqrt(1.0 + alpha * np.clip(np.asarray(h, float), 0.0, 1.0) ** n)


@sfx("freeze", "Water freezing / ice under stress: accelerating brittle ticks, stick-slip creaks and faint ice pings.",
     dur=(4.0, "s"), rate=(8.0, "micro-cracks per second at the end, 0.5..80 (they start at a fifth of this)"),
     creak=(0.5, "0..1 level of the stick-slip creaking"), pings=(2, "dispersive ice pings in the take, 0..12"),
     seed=(0, "variation"))
def freeze(sr=DEFAULT_SR, dur=4.0, rate=8.0, creak=0.5, pings=2, seed=0):
    """Creaks: stick-slip (``physical.ambience.creak``) with the
    "pine" body scaled to 0.45 so its formants sit higher (stiffer, colder material; the size
    factor is chosen). Ticks: Poisson, log-normal sizes, 1-4 ms bursts at 2-6 kHz, chosen.
    Pings: ``ice_chirp`` in thin ice, far away."""
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
