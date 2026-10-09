"""Friction-driven sound and the Baschet / sound-sculpture instrument family.

Three friction solvers act on one private modal bank (exact step-invariant recurrence of a damped
oscillator under a force held over the sample, findings g08 §1.6, eq. A7):

* ``friction_modal_resonator`` — Couineaux, Ablitzer & Gautier, "Minimal physical model of the cristal
  Baschet", Acta Acustica 7:49 (2023): hyperbolic friction law, resonator load line, stick/slip state
  machine (g08 §1.4-1.8).
* ``elastoplastic_friction`` — Rocchesso & Fontana (eds.), *The Sounding Object* (2003) §8.3.2 and
  appendix 8.A: elasto-plastic bristle model solved with the K method + Newton (g04 §4.6, g05 §9).
* ``bowed_modal`` — Bilbao, *Numerical Sound Synthesis* (2009) §4.3.1 / §7.4 / Problem 7.17: the soft bow
  characteristic with the implicit scalar solve and its uniqueness bound (g01 §3.6, g02 §5, §11.3).

Baschet data (Ruiz i Carulla, "Escultura sonora Baschet", thesis + annexes; g09, g10): clamped-bar
ratios, the f ~ 1/L^2 length rule, measured partial tables, collector coupling, radiators.

Every mode is described by (frequency, damping ratio, contact mobility a = phi^2/m in 1/kg), so that
``y'' + 2 xi w y' + w^2 y = a F`` is the contact-point contribution of that mode.
The friction kernels run at a fixed 48 kHz (the rate of the cristal paper) and are resampled at the
boundary; struck banks and filters derive their coefficients from the requested rate.
"""
from __future__ import annotations

import math

import numba
import numpy as np
from scipy.signal import resample_poly

from . import filters as F
from .core import DEFAULT_SR, samples
from .fx import effect
from .instruments import instrument
from .notes import to_hz
from .sfx import sfx

FS = 48000                      # internal rate of the friction kernels (g08 §1.6: the paper's fs)
MU_S, MU_D, V0 = 2.0, 0.3, 0.02  # wet finger on glass, hyperbolic law (g08 §1.5)
XI_CRISTAL = 0.0009             # modal damping ratio of the cristal resonator (g08 §1.3)
MOBILITY_CRISTAL = 21.0         # phi^2/m [1/kg] of the playing mode: 20-22 reproduces both paper checks (g08 §1.3)

CLAMPED = (1.0, 6.267, 17.55, 34.39)     # cantilever partial ratios (g09 §9, thesis L7814-7819)
FREE = (1.0, 2.756, 5.404, 8.933)        # free-free bar (genny BarBody; g09 §26: not in the thesis)
_BETA = {"clamped": (1.8751, 4.6941, 7.8548, 10.9955), "free": (4.7300, 7.8532, 10.9956, 14.1372)}
ALU_BLADE = (1.0, 2.71, 4.87, 8.67, 12.76)   # measured Ikedaphone aluminium blade A (g09 §3)
# Kit Barres 750 mm M8 rod in its collector: measured partials / 125 Hz, (ratio, relative level) (g10 §8.1;
# "faint" / "very faint" / "attack only" entries of the table are given low levels)
KIT_BAR = ((1.0, 1.0), (1.32, 0.5), (2.02, 0.45), (2.06, 0.45), (2.8, 0.35), (3.12, 0.08), (3.26, 0.3),
           (3.44, 0.3), (3.9, 0.06), (4.22, 0.12), (4.26, 0.12), (4.6, 0.2), (5.15, 0.2), (6.9, 0.15), (7.3, 0.12))
KIT_LENGTHS_MM = (880.0, 830.0, 780.0, 750.0, 705.0, 670.0, 645.0, 610.0)   # Kit Barres long halves (g10 §8.1)
GLASS = (1.0, 2.32, 4.25, 6.63, 9.38)    # rubbed-glass mode ratios (STK BandedWG glass_harmonica, g08 status)
STEEL = (2.2e11, 7900.0, 0.3)            # E [Pa], rho [kg/m3], Poisson: threaded shafts / steel (g08 Table 1)


# ------------------------------------------------------------------------------------------------
# modal bank coefficients and kernels
# ------------------------------------------------------------------------------------------------
def _coefs(freqs, xi, mob, dt):
    """Exact one-step matrices of ``y'' + 2 xi w y' + w^2 y = a F`` for F held over dt (g08 eq. A7)."""
    w = 2 * np.pi * np.asarray(freqs, float)
    xi = np.broadcast_to(np.asarray(xi, float), w.shape)
    a = np.broadcast_to(np.asarray(mob, float), w.shape)
    wd = w * np.sqrt(1 - xi * xi)
    e, c, s = np.exp(-xi * w * dt), np.cos(wd * dt), np.sin(wd * dt)
    a11 = e * (c + xi * w / wd * s)
    return np.stack([a11, e * s / wd, -e * w * w / wd * s, e * (c - xi * w / wd * s),
                     a * (1 - a11) / (w * w), a * e * s / wd])


@numba.njit(cache=True)
def _hyperbolic_kernel(co, fn, uf, mus, mud, v0, og):
    """Cristal Baschet minimal model (g08 §1.6). Returns (pickup velocity, contact velocity, F_T, stick flag).

    Each sample the resonator is the load line F = c1*dv + c0 (dv = rod - finger velocity); it is
    intersected with stick (dv = 0, |F| <= mus*FN) or with the hyperbolic slip law
    F = -sign(dv) FN (mud + (mus-mud)/(1+|dv|/V0)), i.e. with x = |dv|, s = sign(dv):
        c1 x^2 + (c1 V0 + s c0 + mud FN) x + V0 (s c0 + mus FN) = 0.
    |c0| > mus FN: exactly one positive root in the direction s = -sign(c0) (this includes the forward
    slip of g08 §1.8). |c0| <= mus FN: stick, unless the resonator is so light that the line also cuts
    the slip branch (c1 V0 < (mus-mud) FN); then a slipping contact stays on the outer root
    (Friedlander / Woodhouse hysteresis rule).
    """
    n = fn.shape[0]
    m = co.shape[1]
    y = np.zeros(m)
    yd = np.zeros(m)
    c1 = 1.0 / co[5].sum()
    out = np.empty(n)
    vel = np.empty(n)
    force = np.empty(n)
    stick = np.zeros(n, np.uint8)
    slip_sign = 0.0                      # 0: sticking, +-1: sign of dv while slipping
    for i in range(n):
        vfree = 0.0
        for k in range(m):
            vfree += co[2, k] * y[k] + co[3, k] * yd[k]
        c0 = -(vfree - uf[i]) * c1
        lim = mus * fn[i]
        f = c0
        if abs(c0) > lim:
            s = -1.0 if c0 > 0 else 1.0
            b = c1 * v0 + s * c0 + mud * fn[i]
            x = (-b + math.sqrt(b * b - 4 * c1 * v0 * (s * c0 + lim))) / (2 * c1)
            f = c1 * s * x + c0
            slip_sign = s
        else:
            held = False
            if slip_sign != 0.0:
                b = c1 * v0 + slip_sign * c0 + mud * fn[i]
                disc = b * b - 4 * c1 * v0 * (slip_sign * c0 + lim)
                if b < 0 and disc >= 0:
                    f = c1 * slip_sign * (-b + math.sqrt(disc)) / (2 * c1) + c0
                    held = True
            if not held:
                slip_sign = 0.0
                stick[i] = 1
        v = 0.0
        o = 0.0
        for k in range(m):
            yn = co[0, k] * y[k] + co[1, k] * yd[k] + co[4, k] * f
            yd[k] = co[2, k] * y[k] + co[3, k] * yd[k] + co[5, k] * f
            y[k] = yn
            v += yd[k]
            o += og[k] * yd[k]
        out[i] = o
        vel[i] = v
        force[i] = f
    return out, vel, force, stick


@numba.njit(cache=True)
def _zdot(v, z, fc, fs, vs, s0, zba):
    """Elasto-plastic bristle rate z' = v (1 - alpha z / z_ss) (Sounding Object eq. 8.16-8.19, g04 §4.6)."""
    zss = (fc + (fs - fc) * math.exp(-(v / vs) ** 2)) / s0        # |z_ss(v)|
    az = abs(z)
    if v * z <= 0.0 or az < zba:
        return v
    if az < zss:
        alpha = 0.5 * (1.0 + math.sin(math.pi * (az - 0.5 * (zss + zba)) / (zss - zba)))
    else:
        alpha = 1.0
    return v * (1.0 - alpha * az / zss)


@numba.njit(cache=True)
def _bristle_kernel(co, fn, vb, fw, mus, mud, vs, s0, s1, s2, og):
    """Elasto-plastic friction between an exciter of prescribed velocity vb and a modal resonator.

    K method (g04 §4.6): with h = z'(n), z = z~ + (dt/2) h and v = v~ + K1 h, solve z'(v, z) - h = 0 by
    Newton; f = s0 z + s1 h + s2 v + noise acts as -f on the resonator. v = resonator - exciter velocity.
    """
    n = fn.shape[0]
    m = co.shape[1]
    dt = 1.0 / FS
    y = np.zeros(m)
    yd = np.zeros(m)
    b = co[5].sum()
    k1 = -b * (s0 * dt / 2 + s1) / (1 + s2 * b)
    k2 = dt / 2
    out = np.empty(n)
    vel = np.empty(n)
    force = np.empty(n)
    z = 0.0
    h = 0.0
    for i in range(n):
        vfree = 0.0
        for k in range(m):
            vfree += co[2, k] * y[k] + co[3, k] * yd[k]
        f = 0.0
        if fn[i] > 1e-9:
            fc = mud * fn[i]
            fs = mus * fn[i]
            zba = 0.7 * fc / s0          # eq. 8.28 with c = 0.7 (< 1)   # UNSOURCED: the book only requires c < 1
            zt = z + k2 * h
            vt = (vfree - vb[i] - b * (s0 * zt + fw[i])) / (1 + s2 * b)
            done = False
            for _ in range(8):
                g = _zdot(vt + k1 * h, zt + k2 * h, fc, fs, vs, s0, zba) - h
                # ponytail: finite-difference slope instead of the analytic eq. 8.44-8.48; swap if profiling asks
                e = 1e-7 + 1e-6 * abs(h)
                dg = (_zdot(vt + k1 * (h + e), zt + k2 * (h + e), fc, fs, vs, s0, zba) - (h + e) - g) / e
                if dg > -1e-3:
                    break
                step = g / dg
                h -= step
                if abs(step) < 1e-10:
                    done = True
                    break
            if not done:
                # Newton left the smooth region (alpha switches): g -> +inf / -inf as h -> -inf / +inf
                # (alpha = 0 there, g = vt + (k1 - 1) h), so a sign change always exists; bisect it.
                c = vt / (1 - k1)
                w = abs(c) + 1e-6
                lo = c - w
                hi = c + w
                for _ in range(60):
                    if _zdot(vt + k1 * lo, zt + k2 * lo, fc, fs, vs, s0, zba) - lo > 0:
                        break
                    lo -= w
                    w *= 2
                for _ in range(60):
                    if _zdot(vt + k1 * hi, zt + k2 * hi, fc, fs, vs, s0, zba) - hi < 0:
                        break
                    hi += w
                    w *= 2
                for _ in range(50):
                    h = 0.5 * (lo + hi)
                    if _zdot(vt + k1 * h, zt + k2 * h, fc, fs, vs, s0, zba) - h > 0:
                        lo = h
                    else:
                        hi = h
            z = zt + k2 * h
            f = s0 * z + s1 * h + s2 * (vt + k1 * h) + fw[i]
        else:
            z = 0.0
            h = 0.0
        v = 0.0
        o = 0.0
        for k in range(m):
            yn = co[0, k] * y[k] + co[1, k] * yd[k] - co[4, k] * f
            yd[k] = co[2, k] * y[k] + co[3, k] * yd[k] - co[5, k] * f
            y[k] = yn
            v += yd[k]
            o += og[k] * yd[k]
        out[i] = o
        vel[i] = v
        force[i] = f
    return out, vel, force


@numba.njit(cache=True)
def _bow_kernel(co, fb, vb, a, og):
    """Bilbao's bow on a modal bank: solve q + g*phi(q) - b = 0 for the relative velocity q each sample,
    phi(q) = sqrt(2a) q exp(-a q^2 + 1/2) (g01 §3.6 (c)), g = F_B * sum(B2), b = free velocity - v_B
    (g02 §5 step 2). F_B is clamped to the uniqueness bound g < e / (2 sqrt(2a)) (eq. 4.27 / 7.34).
    """
    n = fb.shape[0]
    m = co.shape[1]
    y = np.zeros(m)
    yd = np.zeros(m)
    bsum = co[5].sum()
    amp = math.sqrt(2 * a) * math.exp(0.5)
    fmax = 0.9 * math.e / (2 * math.sqrt(2 * a)) / bsum
    out = np.empty(n)
    vel = np.empty(n)
    force = np.empty(n)
    q = 0.0
    for i in range(n):
        vfree = 0.0
        for k in range(m):
            vfree += co[2, k] * y[k] + co[3, k] * yd[k]
        f_b = min(fb[i], fmax)
        g = f_b * bsum
        b = vfree - vb[i]
        for _ in range(30):
            ex = math.exp(-a * q * q)
            step = (q + g * amp * q * ex - b) / (1 + g * amp * (1 - 2 * a * q * q) * ex)
            q -= step
            if abs(step) < 1e-12:
                break
        f = -f_b * amp * q * math.exp(-a * q * q)
        v = 0.0
        o = 0.0
        for k in range(m):
            yn = co[0, k] * y[k] + co[1, k] * yd[k] + co[4, k] * f
            yd[k] = co[2, k] * y[k] + co[3, k] * yd[k] + co[5, k] * f
            y[k] = yn
            v += yd[k]
            o += og[k] * yd[k]
        out[i] = o
        vel[i] = v
        force[i] = f
    return out, vel, force


@numba.njit(cache=True)
def _ring(x, freqs, t60, gains, sr):
    """Sum of two-pole resonators driven by x (impulse response g * exp(-t 6.91/T60) sin(2 pi f t))."""
    n = x.shape[0]
    out = np.zeros(n)
    for k in range(freqs.shape[0]):
        if freqs[k] <= 0 or freqs[k] >= 0.45 * sr:
            continue
        r = math.exp(-6.9078 / (t60[k] * sr))
        th = 2 * math.pi * freqs[k] / sr
        a1 = 2 * r * math.cos(th)
        a2 = r * r
        b = gains[k] * r * math.sin(th)
        y1 = 0.0
        y2 = 0.0
        for i in range(n):
            y0 = a1 * y1 - a2 * y2
            out[i] += b * y1
            y2 = y1
            y1 = y0 + x[i]
    return out


# ------------------------------------------------------------------------------------------------
# public solvers
# ------------------------------------------------------------------------------------------------
def _arr(v, n):
    return np.ascontiguousarray(np.broadcast_to(np.asarray(v, float), (n,)))


def _og(out_gains, m):
    return np.ones(m) if out_gains is None else np.ascontiguousarray(out_gains, dtype=float)


def friction_modal_resonator(freqs, normal_force, finger_speed, dur, *, xi=XI_CRISTAL, mobility=MOBILITY_CRISTAL,
                             mu_s=MU_S, mu_d=MU_D, v0=V0, out_gains=None):
    """FrictionModalResonator: the cristal Baschet minimal model at 48 kHz (g08 §1.4-1.6).

    freqs/xi/mobility: per-mode Hz, damping ratio, contact mobility phi^2/m [1/kg]. normal_force [N] and
    finger_speed [m/s] are scalars or per-sample arrays. Returns a dict with ``out`` (pickup velocity),
    ``vel`` (rod velocity at the contact, m/s), ``force`` (F_T, N) and ``stick`` (1 while sticking).
    """
    n = int(round(dur * FS))
    freqs = np.atleast_1d(np.asarray(freqs, float))
    co = _coefs(freqs, xi, mobility, 1.0 / FS)
    out, vel, force, stick = _hyperbolic_kernel(co, _arr(normal_force, n), _arr(finger_speed, n), float(mu_s),
                                                float(mu_d), float(v0), _og(out_gains, freqs.size))
    return {"out": out, "vel": vel, "force": force, "stick": stick}


def fn_min(freq, finger_speed=0.1, xi=XI_CRISTAL, mobility=MOBILITY_CRISTAL, mu_s=MU_S, mu_d=MU_D, v0=V0):
    """Minimum normal force [N] for self-oscillation of one mode, from the linear stability analysis
    (g08 §1.7): 2 xi w V0 (1 + u_f/V0)^2 / ((mu_s - mu_d) phi^2/m)."""
    return 2 * xi * 2 * np.pi * freq * v0 * (1 + finger_speed / v0) ** 2 / ((mu_s - mu_d) * mobility)


def rise_time(freq, normal_force=1.0, finger_speed=0.1, xi=XI_CRISTAL, mobility=MOBILITY_CRISTAL, mu_s=MU_S,
              mu_d=MU_D, v0=V0):
    """LSA estimate of the time to the first stick phase (g08 eq. 23): ln(u_f / C) / Re(lambda), where C is
    the velocity amplitude of the step response to the sudden friction force."""
    w = 2 * np.pi * freq
    slope = (mu_s - mu_d) * normal_force / (v0 * (1 + finger_speed / v0) ** 2)
    growth = 0.5 * (slope * mobility - 2 * xi * w)
    c = (mu_d + (mu_s - mu_d) / (1 + finger_speed / v0)) * normal_force * mobility / w
    return float(np.log(finger_speed / c) / growth) if growth > 0 else float("inf")


def elastoplastic_friction(freqs, xi, mobility, normal_force, speed, dur, *, mu_s=0.975, mu_d=0.197, v_s=0.1,
                           sigma0=2.0e4, sigma1=None, sigma2=0.0, noise=None, out_gains=None):
    """Elasto-plastic (LuGre-family) bristle friction between an exciter moving at ``speed`` [m/s] and a
    modal resonator, at 48 kHz (*The Sounding Object* §8.3.2, g04 §4.6). mu_s, mu_d, v_s default to the
    book's Fig. 8.5 set. ``noise``: optional force-noise array [N] (the sigma3 w term).

    The book prints no sigma values.  # UNSOURCED: sigma0 default tuned here so 0.02-0.2 kg modes lock in
    sigma1 defaults to 0.3 mu_d max(F_N) / v_s, the usual LuGre scaling that keeps sigma1 z' well below
    the Coulomb force.  # UNSOURCED: scaling rule, not from the book
    """
    n = int(round(dur * FS))
    freqs = np.atleast_1d(np.asarray(freqs, float))
    co = _coefs(freqs, xi, mobility, 1.0 / FS)
    if sigma1 is None:
        sigma1 = 0.3 * mu_d * float(np.max(normal_force)) / v_s
    fw = np.zeros(n) if noise is None else _arr(noise, n)
    out, vel, force = _bristle_kernel(co, _arr(normal_force, n), _arr(speed, n), fw, float(mu_s), float(mu_d),
                                      float(v_s), float(sigma0), float(sigma1), float(sigma2), _og(out_gains, freqs.size))
    return {"out": out, "vel": vel, "force": force}


def bowed_modal(freqs, xi, weights, bow_force, bow_speed, dur, *, a=100.0, out_gains=None):
    """Bilbao's soft bow characteristic on a modal bank, implicit solve, at 48 kHz (g01 §3.6, g02 §5).

    weights: squared mass-normalised mode shapes at the bow point (1 for a bowed mass); bow_force is
    F_B = force / mass [m/s^2] (book examples 50-4000), bow_speed v_B [m/s] (0.2), a = 100.
    """
    n = int(round(dur * FS))
    freqs = np.atleast_1d(np.asarray(freqs, float))
    co = _coefs(freqs, xi, weights, 1.0 / FS)
    out, vel, force = _bow_kernel(co, _arr(bow_force, n), _arr(bow_speed, n), float(a), _og(out_gains, freqs.size))
    return {"out": out, "vel": vel, "force": force}


# ------------------------------------------------------------------------------------------------
# Baschet modal data and small helpers
# ------------------------------------------------------------------------------------------------
def rod_frequency(length_m, diameter_m=0.008, material=STEEL):
    """First mode [Hz] of a round rod clamped at one end: 0.5596 / L^2 * sqrt(E/rho) * d/4 (Euler-Bernoulli,
    as genny BarBody). Gives the Baschet length rule f ~ 1/L^2, i.e. length x 0.7071 per octave (g09 §3)."""
    return 0.5596 / (length_m * length_m) * math.sqrt(material[0] / material[1]) * diameter_m / 4


def rod_length(freq, diameter_m=0.008, material=STEEL):
    """Free length [m] of the clamped rod whose first mode is ``freq`` (inverse of rod_frequency)."""
    return math.sqrt(0.5596 * math.sqrt(material[0] / material[1]) * diameter_m / 4 / freq)


def beam_shape(kind, x):
    """Mass-normalised Euler-Bernoulli mode shapes (first four modes) at x in 0..1.
    kind 'free': free-free bar, x from one end; 'clamped': cantilever, x from the clamp."""
    bl = np.array(_BETA[kind])
    ch, c, sh, s = np.cosh(bl), np.cos(bl), np.sinh(bl), np.sin(bl)
    if kind == "free":
        return np.cosh(bl * x) + np.cos(bl * x) - (ch - c) / (sh - s) * (np.sinh(bl * x) + np.sin(bl * x))
    return np.cosh(bl * x) - np.cos(bl * x) - (ch + c) / (sh + s) * (np.sinh(bl * x) - np.sin(bl * x))


def _to_sr(x, sr):
    if sr == FS:
        return x
    g = math.gcd(int(sr), FS)
    return resample_poly(x, int(sr) // g, FS // g)


def _edges(x, sr, t_in=0.002, t_out=0.012):
    """DC removal and raised-cosine fades so the buffer starts and ends at zero."""
    x = F.dc_block(np.asarray(x, float), sr)
    n = x.shape[0]
    a, b = min(n // 2, max(2, int(t_in * sr))), min(n // 2, max(2, int(t_out * sr)))
    x[:a] *= np.sin(np.linspace(0, np.pi / 2, a)) ** 2
    x[-b:] *= np.cos(np.linspace(0, np.pi / 2, b)) ** 2
    return x


def _gate(n, n_on, t_att, t_rel):
    """Gesture envelope at FS: raised-cosine rise over t_att, hold to n_on, raised-cosine fall over t_rel."""
    e = np.ones(n)
    a, r = max(2, int(t_att * FS)), max(2, int(t_rel * FS))
    k = min(a, n)
    e[:k] = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a)))[:k]
    if n_on < n:
        k = min(r, n - n_on)
        e[n_on:n_on + k] *= (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, r)))[:k]
        e[n_on + k:] = 0.0
    return e


def _spectral_peak(x, f, fs=FS, span=0.15):
    """Frequency of the strongest spectral line within +-span of f (parabolic interpolation)."""
    n = x.shape[0]
    sp = np.abs(np.fft.rfft(x * np.hanning(n), 8 * n))
    lo, hi = int(f * (1 - span) * 8 * n / fs), int(f * (1 + span) * 8 * n / fs) + 2
    k = lo + int(np.argmax(sp[lo:hi]))
    if k <= lo or k >= hi - 1 or sp[k] < 1e-12:
        return 0.0
    d = 0.5 * (sp[k - 1] - sp[k + 1]) / (sp[k - 1] - 2 * sp[k] + sp[k + 1] + 1e-30)
    return (k + d) * fs / (8 * n)


def _autotune(run, freq, t=1.0):
    """Friction flattens (or the bristle spring sharpens) a self-oscillation by an amount that depends on
    force and speed (g01 §3.6: 'more bow force -> longer period'). Run a short pilot of the steady gesture
    with the modes at ``freq`` and return the factor that puts the sounding pitch on ``freq``."""
    f = _spectral_peak(run(1.0, t)[int(0.1 * t * FS):], freq)     # includes most of the attack: an average pitch
    return freq / f if f > 0 else 1.0


def _pulse(sr, ms):
    w = np.hanning(max(3, int(round(ms * 1e-3 * sr)) | 1) + 2)[1:-1]
    return w / w.sum()


def _struck(freqs, t60, gains, n, sr, contact_ms, amp=1.0):
    """Strike a bank with a Hann force pulse. The pulse is scaled so the first mode gets amplitude
    gains[0]*amp whatever the contact time: hardness then only changes the balance of the upper partials."""
    x = np.zeros(n)
    p = _pulse(sr, min(contact_ms, 500.0 / freqs[0]))[:n]      # a contact longer than half a period cannot drive the mode
    p0 = abs(np.sum(p * np.exp(-2j * np.pi * freqs[0] / sr * np.arange(p.size))))
    x[:p.size] = p * amp / max(p0, 0.05)
    return _ring(x, np.ascontiguousarray(freqs, float), np.ascontiguousarray(t60, float),
                 np.ascontiguousarray(gains, float), float(sr))


def _note_tail(y, dur, sr, tail):
    """Let a voice ring for ``dur`` and damp it over ``tail`` seconds (hand damping)."""
    n_on = samples(dur, sr)
    if n_on < y.shape[0]:
        k = y.shape[0] - n_on
        y[n_on:] *= np.exp(-np.arange(k) / (tail / 5.0 * sr))
    return y


def _tone(x, freq, sr, bright, harm=6.0, lo=900.0, hi=3200.0):
    """Key-tracked 12 dB/oct lowpass: keeps `harm` harmonics but never opens above hi*bright Hz."""
    fc = float(np.clip(harm * freq, lo, hi) * np.clip(bright, 0.25, 4.0))
    return F.lowpass(x, min(fc, 0.45 * sr), sr)


# ------------------------------------------------------------------------------------------------
# radiators (g09 §19-21) and sympathetic resonators (g09 §21)
# ------------------------------------------------------------------------------------------------
def _plate_modes(thickness_mm, size, n, lo, hi, rng):
    """n modes of a simply supported steel sheet (long side `size` m, short side 2/3 of it, like the
    150 x 100 cm Baschet sheets, g09 §20.6), log-spread between lo and hi Hz.
    f_mn = (pi/2) h sqrt(E / (12 rho (1 - nu^2))) (m^2/a^2 + n^2/b^2): thicker = higher."""
    e, rho, nu = STEEL
    c = (np.pi / 2) * thickness_mm * 1e-3 * math.sqrt(e / (12 * rho * (1 - nu * nu)))
    a, b = size, size * 2.0 / 3.0
    mm = np.arange(1, 500.0) ** 2
    f = (c * (mm[:, None] / (a * a) + mm[None, :] / (b * b))).ravel()
    f = np.sort(f[f < hi])
    f = f[f >= min(lo, f[-1])] if f.size else np.array([c * (1 / (a * a) + 1 / (b * b))])
    idx = np.searchsorted(f, np.geomspace(f[0], f[-1], n) * rng.uniform(0.97, 1.03, n))
    return np.unique(f[np.clip(idx, 0, f.size - 1)])


def _unit_bank(x, freqs, t60, gains, sr):
    """Resonator bank scaled to unit white-noise power gain overall (a tone on a partial is boosted)."""
    r = np.exp(-6.9078 / (t60 * sr))
    g = gains * 2 * np.sqrt(1 - r) / math.sqrt(max(1, freqs.size))
    return _ring(np.ascontiguousarray(x, float), np.ascontiguousarray(freqs, float), np.ascontiguousarray(t60, float),
                 np.ascontiguousarray(g, float), float(sr))


def _sheet(x, sr, thickness=0.5, size=1.5, drive=1.0, decay=1.5, mix=0.5, bright=1.0, seed=0, avoid=0.0):
    rng = np.random.default_rng(seed)
    th = float(np.clip(thickness, 0.2, 2.0))
    size = float(np.clip(size, 0.3, 4.0))
    dec = float(np.clip(decay, 0.05, 8.0))
    top = min(4500.0 * float(np.clip(bright, 0.25, 4.0)), 0.42 * sr)
    f = _plate_modes(th, size, 40, 90.0, top, rng)
    if avoid > 0:      # instruments: no sheet mode within 7 % of the note, so the sheet cannot beat against it
        f = f[np.abs(f / avoid - 1.0) > 0.07]
    g = rng.lognormal(0.0, 0.6, f.size)              # "uneven and unpredictable" response curve (g09 §20.6)
    t60 = dec * np.clip((400.0 / f) ** 0.5, 0.15, 1.5)
    # Level-dependent part. Thinner sheet = earlier saturation and more strident, thicker = fuller (g09 §20.6).
    # UNSOURCED: tanh shape, the (0.5 mm / thickness)^2 drive law and the quadratic "thunder" coupling are
    # this module's reading of the qualitative description; the source has no measured nonlinearity.
    k = max(1e-3, float(drive)) * (0.5 / th) ** 2
    s = np.tanh(k * x) / k
    wet = _unit_bank(s, f, t60, g, sr)
    ft = _plate_modes(th, size, 12, 18.0, 160.0, rng)   # thunder: the long low modes, driven by the envelope
    env = F.highpass(F.lowpass(s * s, 60.0, sr), 8.0, sr)
    wet += 2.0 * k * _unit_bank(env, ft, np.full(ft.size, 2.0 * dec), rng.lognormal(0.0, 0.4, ft.size), sr)
    wet = 1.2 * np.tanh(wet / 1.2)                    # the sheet cannot radiate more: it saturates
    m = float(np.clip(mix, 0.0, 1.0))
    return (1 - m) * s + m * wet


def _cone(x, sr, size=0.6, material="cardboard", seed=0):
    """Cone / balloon radiator: no tail of its own, a broad formant, low cut set by size (g09 §20.2, §20.5;
    g10 S3: a ~0.35 m cardboard cone is weak below 150-200 Hz, 2 m metal cones reach 30 Hz -> fc = 60/size)."""
    size = float(np.clip(size, 0.1, 3.0))
    fc = 60.0 / size
    y = F.highpass(x, fc, sr)
    if material == "balloon":
        # dry, "filters highs, inflates mids", saturates when driven hard (g10 §5)
        y = F.peak(y, 320.0, sr, q=0.8, gain_db=5.0)        # UNSOURCED: centre of the mid bump
        y = F.lowpass(y, min(2600.0, 0.45 * sr), sr)
        return 1.5 * np.tanh(y / 1.5)
    y = F.peak(y, min(3.2 * fc + 200.0, 0.4 * sr), sr, q=1.1, gain_db=4.0)    # UNSOURCED: formant position
    if material == "metal":
        # metal cones add a short ring and bright overtones; fibre / cardboard cones add nothing (g09 §20.5)
        rng = np.random.default_rng(seed)
        f = np.sort(rng.uniform(4 * fc, min(3800.0, 0.42 * sr), 10))
        return 0.75 * y + 0.5 * _unit_bank(y, f, np.full(f.size, 0.35), np.ones(f.size), sr)
    return F.lowpass(y, min(4200.0, 0.45 * sr), sr)


def _radiate(x, sr, kind, size=1.5, **sheet):
    if kind == "none":
        return x
    if kind == "sheet":
        return _sheet(x, sr, size=size, **sheet)
    if kind in ("cone", "metal", "balloon"):
        return _cone(x, sr, size, "cardboard" if kind == "cone" else kind)
    raise ValueError(f"radiator must be sheet|cone|balloon|metal|none, not {kind!r}")


def _mono_fx(fn, x):
    x = np.asarray(x, float)
    return fn(x) if x.ndim == 1 else np.stack([fn(np.ascontiguousarray(x[:, c])) for c in range(x.shape[1])], axis=1)


@effect("sheet_radiator", "Folded stainless-sheet radiator (Baschet): uneven modal colouring, long ring, "
        "level-dependent saturation and low 'thunder' modes.",
        thickness=(0.5, "sheet thickness, mm (0.3-0.8): thinner saturates earlier and is more strident"),
        size=(1.5, "long side of the sheet, m (0.3-4): bigger = lower, denser modes"),
        drive=(1.0, "input drive into the saturation (0.2-8)"), decay=(1.5, "ring time, s (0.05-8)"),
        mix=(0.5, "0 dry .. 1 sheet only"), bright=(1.0, "upper limit of the modes = 4500 Hz x bright"),
        seed=(0, "int: which modes and how uneven"))
def sheet_radiator(x, sr=DEFAULT_SR, thickness=0.5, size=1.5, drive=1.0, decay=1.5, mix=0.5, bright=1.0, seed=0):
    """g09 §20.6 / §25 ('a sheet of metal vibrates like thunder when injected with ... one of its partials')."""
    return _mono_fx(lambda c: _sheet(c, sr, thickness, size, drive, decay, mix, bright, int(seed)), x)


@effect("cone_radiator", "Baschet cone / balloon radiator: dry formant colouring, low cut set by the cone size.",
        size=(0.6, "cone seam length, m (0.1-3): low cut = 60/size Hz"),
        material=("cardboard", "cardboard (dry) | metal (short bright ring) | balloon (dry, mid-heavy, saturates)"),
        mix=(1.0, "0 dry .. 1 radiator only"), seed=(0, "int: metal ring modes"))
def cone_radiator(x, sr=DEFAULT_SR, size=0.6, material="cardboard", mix=1.0, seed=0):
    """g09 §20.2, §20.5."""
    if material not in ("cardboard", "metal", "balloon"):
        raise ValueError("material must be cardboard|metal|balloon")
    m = float(np.clip(mix, 0.0, 1.0))
    return _mono_fx(lambda c: (1 - m) * c + m * _cone(c, sr, size, material, int(seed)), x)


@effect("sympathetic", "Bank of sympathetic strings or free-end rods ('whiskers') excited by the input.",
        tuning=("D3 A3 D4 F#4 A4", "notes or Hz the resonators are tuned to (list or space-separated string)"),
        kind=("string", "string (harmonic partials) | rod (clamped-rod partials 1 : 6.27 : 17.55 : 34.39)"),
        decay=(3.0, "ring time of the fundamentals, s (0.1-12)"), mix=(0.3, "level of the resonators added to the input (0-1)"))
def sympathetic(x, sr=DEFAULT_SR, tuning="D3 A3 D4 F#4 A4", kind="string", decay=3.0, mix=0.3):
    """Resonance add-ons of g09 §21: strings fixed at both ends, or piano-wire whiskers clamped at one end
    that 'ring longer than the oscillator signal and add brilliant higher octaves'."""
    from .notes import parse_pitch_list
    if kind not in ("string", "rod"):
        raise ValueError("kind must be string|rod")
    ratios = np.arange(1.0, 9.0) if kind == "string" else np.array(CLAMPED)
    base = np.array(parse_pitch_list(tuning), float)
    f = (base[:, None] * ratios[None, :]).ravel()
    g = np.tile(1.0 / np.sqrt(np.arange(1, ratios.size + 1)), base.size)
    keep = f < min(0.42 * sr, 9000.0)
    f, g = f[keep], g[keep]
    t60 = float(np.clip(decay, 0.1, 12.0)) * np.clip((base.min() / f) ** 0.5, 0.05, 1.0)
    m = float(np.clip(mix, 0.0, 1.0))
    return _mono_fx(lambda c: c + m * 1.5 * np.tanh(_unit_bank(c, f, t60, g, sr) / 1.5), x)


# ------------------------------------------------------------------------------------------------
# instruments
# ------------------------------------------------------------------------------------------------
# output gains that put each voice near -12 dB K-weighted at vel 0.9 (the coordinator's level_match trims the rest)
_LEVEL = {"cristal": 0.6, "rubbed_glass": 0.35, "bowed_bar": 0.35, "bowed_sheet": 0.43, "rod_bank": 0.65,
          "whistling_blades": 0.5, "tuning_fork": 0.5, "coil_spring": 0.16}


def _friction_note(run, freq, dur, sr, tail, ref):
    """run(scale, seconds, n_on) -> kernel output at FS. Auto-tunes, renders dur + tail, resamples."""
    scale = _autotune(lambda s, t: run(s, t, int(t * FS) + 1), freq)
    return _note_tail(_to_sr(run(scale, dur + tail, int(dur * FS)) / ref, sr), dur, sr, tail)


@instrument("cristal", "Cristal Baschet: wet finger on a glass rod driving a clamped steel shaft by stick-slip; "
            "near-sine, attackless, sustained.", family="bowed", span=("C2", "C6"),
            pressure=(1.0, "finger normal force, N (0-6); under about 0.1 N the rod stays silent"),
            finger_speed=(0.15, "finger speed along the rod, m/s (0.02-0.5): sets loudness"),
            radiator=("sheet", "sheet (ringing, saturates) | cone (dry, clean) | balloon (dry, mid-heavy)"),
            bright=(1.0, "0.5 mellow .. 2 bright: friction harmonics and lowpass"),
            wetness=(1.0, "0 too slippery .. 1 well-wetted finger (the paper's friction law)"))
def cristal(freq, dur, sr=DEFAULT_SR, vel=1.0, pressure=1.0, finger_speed=0.15, radiator="sheet", bright=1.0, wetness=1.0):
    """Couineaux, Ablitzer & Gautier 2023 minimal model (g08 §1): one playing mode, xi = 0.09 %,
    hyperbolic friction mu_s 2 / mu_d 0.3 / V0 0.02 m/s. Rod velocity at the contact goes through
    collector and radiator (g09 §12, §20).
    The contact mobility is the paper's 21 1/kg at its C#4 = 277.2 Hz resonator and is scaled in proportion
    to the note, i.e. lighter resonators higher up, as the makers do to get 'the same gesture, equal
    response' in every register (g09 §12). With one fixed resonator mass the minimum force would rise with
    pitch instead (g08 §1.7).  # UNSOURCED: the exact proportionality"""
    mob = MOBILITY_CRISTAL * float(np.clip(freq / 277.2, 0.1, 8.0))
    uf = float(np.clip(finger_speed, 0.02, 0.5)) * (0.4 + 0.6 * float(np.clip(vel, 0.0, 1.5)))
    fn = float(np.clip(pressure, 0.0, 6.0))
    # UNSOURCED: mapping of `wetness` onto mu_s (the paper only says the film dries and the law drifts)
    mus = MU_D + (MU_S - MU_D) * (0.3 + 0.7 * float(np.clip(wetness, 0.0, 1.0)))
    b = float(np.clip(bright, 0.25, 4.0))

    def run(scale, t, n_on):
        n = int(t * FS)
        g = _gate(n, n_on, 0.01, 0.03)
        # A slow finger starts the note more easily (g08 §1.7): in pure slip the amplitude grows at the rate
        # c / (1 + u/V0)^2, c = mob (mu_s - mu_d) F_N / (2 V0). The gesture holds a quarter of the speed until
        # the rod has grown from its step response C into stick-slip (eq. 23, with a 1.5x margin), then
        # accelerates along u^2 ~ 2 c V0^2 t, the curve on which the rod can just follow, at a fifth of that rate.
        f_n = max(fn, 0.05)
        c = mob * (mus - MU_D) * f_n / (2 * V0)
        u0 = 0.25 * uf
        step = (MU_D + (mus - MU_D) / (1 + u0 / V0)) * f_n * mob / (2 * np.pi * freq)
        tt = np.arange(n) / FS - min(0.4, 1.5 * math.log(max(u0 / step, 1.5)) * (1 + u0 / V0) ** 2 / c)
        rate = max(0.4 * c * V0 * V0, (uf * uf - u0 * u0) / 1.0)          # full speed after 1 s at the latest
        speed = np.minimum(uf, np.sqrt(u0 * u0 + rate * np.maximum(tt, 0.0)))
        r = friction_modal_resonator(freq * scale, fn * g, speed, n / FS, mobility=mob, mu_s=mus)
        # the clamp also feels the friction force itself: a square-like wave that carries the harmonics
        # (g09 §12). UNSOURCED: its mix level (0.1 x bright).
        return r["vel"] + 0.1 * b * uf * F.highpass(r["force"], 2 * freq, FS) / (MU_S * max(fn, 0.05))

    y = _tone(_friction_note(run, freq, dur, sr, 0.5, 0.15), freq, sr, b)
    y = _radiate(y, sr, radiator, **({"mix": 0.35, "decay": 1.2, "bright": b, "avoid": freq} if radiator == "sheet" else {}))
    return _edges(y, sr) * _LEVEL["cristal"]


@instrument("rubbed_glass", "Wine glass / glass harmonica rubbed with a wet finger (elasto-plastic bristle friction).",
            family="bowed", span=("C4", "C7"),
            pressure=(0.3, "finger normal force, N (0.05-3)"), speed=(0.1, "finger speed, m/s (0.02-0.25); too fast and it stops singing"),
            roughness=(0.2, "0 smooth .. 1 gritty: surface noise in the friction force"),
            bright=(1.0, "0.5 mellow .. 2 bright"), seed=(0, "int: roughness noise"))
def rubbed_glass(freq, dur, sr=DEFAULT_SR, vel=1.0, pressure=0.3, speed=0.1, roughness=0.2, bright=1.0, seed=0):
    """*The Sounding Object* §8.4.2 wineglass: resonator tuned to glass modes with long decays, friction
    parameters of Fig. 8.5 (g04 §4.6). Mode ratios: STK glass harmonica preset."""
    v = float(np.clip(speed, 0.02, 0.25)) * (0.5 + 0.5 * float(np.clip(vel, 0.0, 1.5)))
    fn = float(np.clip(pressure, 0.05, 3.0))
    ratios = np.array(GLASS)
    xi = 0.0006 * np.sqrt(ratios)                      # UNSOURCED: glass loss (T60 about 3.5 s at 500 Hz)
    mob = 50.0 / ratios ** 2                           # UNSOURCED: 20 g effective rim mass, higher modes stiffer
    rough = float(np.clip(roughness, 0.0, 1.0)) * 0.08 * fn

    def run(scale, t, n_on):
        n = int(t * FS)
        g = _gate(n, n_on, 0.005, 0.04)
        noise = rough * g * F.lowpass(np.random.default_rng(seed).standard_normal(n), 3000.0, FS)
        return elastoplastic_friction(freq * scale * ratios, xi, mob, fn * g, v * (0.5 + 0.5 * g), n / FS,
                                      sigma0=1.0e5, noise=noise)["out"]

    y = _friction_note(run, freq, dur, sr, 0.7, 0.1)
    return _edges(_tone(y, freq, sr, bright, harm=5.0), sr) * _LEVEL["rubbed_glass"]


_BAR_MATERIAL = {   # (partial ratios, damping ratio of mode 1)   # UNSOURCED: loss values, only their ranking matters
    "aluminium": (ALU_BLADE[:4], 0.0004), "steel": (FREE, 0.0003), "glass": (FREE, 0.0008), "wood": (FREE, 0.004)}


def _bow_force(force, freq, vb, w1):
    # UNSOURCED: force scaling. F_B is set so the stick limit F_B w1 / (w v_B) stays near 0.1-0.3 (the bow
    # stays on mode 1) and the small-signal growth rate is at least ~30 1/s at any pitch.
    return float(np.clip(force, 0.2, 2.5)) * (0.1 * 2 * np.pi * freq * vb + 12.0) / w1


@instrument("bowed_bar", "Bowed free bar (vibraphone / Baschet blade played with a bow): Bilbao bow on bar modes.",
            family="bowed", span=("F3", "F6"),
            material=("aluminium", "aluminium (measured Baschet blade partials) | steel | glass | wood"),
            force=(1.0, "bow force, relative (0.2-2.5): more = faster attack, harder tone"),
            position=(0.03, "bow point from the bar end as a fraction of length (0-0.15)"),
            speed=(0.2, "bow speed, m/s (0.05-0.5): sets loudness"), bright=(1.0, "0.5 mellow .. 2 bright"))
def bowed_bar(freq, dur, sr=DEFAULT_SR, vel=1.0, material="aluminium", force=1.0, position=0.03, speed=0.2, bright=1.0):
    """Bilbao Problem 7.17 in modal form (g02 §11.3) with the soft friction characteristic, a = 100 (g01 §3.6).
    Free-bar partials: measured Ikedaphone aluminium blade 1 : 2.71 : 4.87 : 8.67 (g09 §3) or theory."""
    if material not in _BAR_MATERIAL:
        raise ValueError(f"material must be one of {sorted(_BAR_MATERIAL)}")
    ratios, xi0 = _BAR_MATERIAL[material]
    ratios = np.array(ratios)
    w = beam_shape("free", float(np.clip(position, 0.0, 0.15))) ** 2
    vb = float(np.clip(speed, 0.05, 0.5)) * (0.4 + 0.6 * float(np.clip(vel, 0.0, 1.5)))
    fb = _bow_force(force, freq, vb, w[0])

    def run(scale, t, n_on):
        n = int(t * FS)
        return bowed_modal(freq * scale * ratios, xi0 * ratios, w, fb * _gate(n, n_on, 0.003, 0.03), vb, n / FS)["out"]

    return _edges(_tone(_friction_note(run, freq, dur, sr, 0.6, 0.2), freq, sr, bright), sr) * _LEVEL["bowed_bar"]


# Lines that a rubbed centre-clamped stainless plate settles on: fundamental, fifth, octave, then a major third
# and another fifth above (Superball spectrogram, g09 §15). There is no steel-cello data in the sources.
_SHEET_RATIOS = np.array([1.0, 1.5, 2.0, 2.5, 3.0])


@instrument("bowed_sheet", "Bowed steel sheet (steel-cello-like): Bilbao bow on plate modes, radiated by the sheet itself.",
            family="bowed", span=("C2", "C5"),
            force=(1.0, "bow force, relative (0.2-2.5)"), speed=(0.2, "bow speed, m/s (0.05-0.5): sets loudness"),
            thickness=(0.5, "sheet thickness, mm (0.3-0.8): thinner = more saturation"),
            ring=(0.5, "0 dry bowed tone .. 1 mostly the sheet's own ringing modes"),
            bright=(1.0, "0.5 mellow .. 2 bright"), seed=(0, "int: sheet modes"))
def bowed_sheet(freq, dur, sr=DEFAULT_SR, vel=1.0, force=1.0, speed=0.2, thickness=0.5, ring=0.5, bright=1.0, seed=0):
    """Bowed plates 'scream' with a few pure frequencies standing out, depending on bow point and pressure
    (g09 §7). Friction: Bilbao bow (g01 §3.6); radiation: sheet_radiator (g09 §20.6)."""
    vb = float(np.clip(speed, 0.05, 0.5)) * (0.4 + 0.6 * float(np.clip(vel, 0.0, 1.5)))
    w = 1.0 / np.arange(1, _SHEET_RATIOS.size + 1) ** 1.5    # UNSOURCED: bow-point weights favouring the low mode
    fb = _bow_force(force, freq, vb, 1.0)
    xi = 0.0015 * np.sqrt(_SHEET_RATIOS)                     # UNSOURCED: sheet loss

    def run(scale, t, n_on):
        n = int(t * FS)
        return bowed_modal(freq * scale * _SHEET_RATIOS, xi, w, fb * _gate(n, n_on, 0.003, 0.04), vb, n / FS)["out"]

    y = _tone(_friction_note(run, freq, dur, sr, 0.8, 0.2), freq, sr, bright)
    y = _sheet(y, sr, thickness=thickness, size=2.0, decay=2.0, mix=0.6 * float(np.clip(ring, 0.0, 1.0)), bright=bright, seed=int(seed), avoid=freq)
    return _edges(y, sr) * _LEVEL["bowed_sheet"]


_ROD_SETS = {   # name -> ((ratio, level) ..., T60 factor of the upper partials, T60 factor of the fundamental)
    "kit": (KIT_BAR, 0.35, 1.0),                                                   # measured, in its collector (g10 §8.1)
    "clamped": (tuple(zip(CLAMPED, (1.0, 0.6, 0.4, 0.25))), 0.5, 1.0),             # textbook cantilever (g09 §9)
    "weighted": (((1.0, 1.0), (6.0, 0.5), (18.0, 0.3), (34.0, 0.2)), 0.5, 1.0),    # Peix de fusta re-harmonised rod A (g10 §17.5)
    "mushroom": (tuple(zip(CLAMPED, (1.0, 0.5, 0.3, 0.2))), 0.02, 0.15)}           # end weight: short, "one clear frequency" (g09 §10)


@instrument("rod_bank", "Baschet percussion: struck clamped rod whose vibration crosses a shared collector, wakes "
            "the neighbouring rods and leaves through a sheet or cone radiator.", family="mallet", span=("G2", "G5"),
            rod=("kit", "kit (measured Kit Barres partials, beating pairs) | clamped (1 : 6.27 : 17.55) | "
                        "weighted (re-harmonised 1 : 6 : 18) | mushroom (end weight: short, one clear tone)"),
            hardness=(0.5, "0 soft heavy rubber .. 1 hard wood: contact time 4 ms .. 0.3 ms"),
            decay=(3.0, "ring time of the struck rod, s (0.1-12)"),
            sympathetic=(0.35, "0-1: how much the other seven rods of the set ring along through the collector"),
            radiator=("sheet", "sheet | cone | balloon | metal | none"), seed=(0, "int: sheet modes"))
def rod_bank(freq, dur, sr=DEFAULT_SR, vel=1.0, rod="kit", hardness=0.5, decay=3.0, sympathetic=0.35, radiator="sheet", seed=0):
    """Struck threaded rods in a collector (g09 §9-10, §16; g10 §8.1). The other rods are the Kit Barres
    set: their pitches follow f ~ 1/L^2 from the lengths 880 ... 610 mm, the struck rod being the 750 mm one."""
    if rod not in _ROD_SETS:
        raise ValueError(f"rod must be one of {sorted(_ROD_SETS)}")
    table, upper, fund = _ROD_SETS[rod]
    ratios, levels = np.array(table).T
    dec = float(np.clip(decay, 0.1, 12.0))
    tail = float(min(1.5, dec))
    n = samples(dur + tail, sr)
    f = freq * ratios
    t60 = np.where(ratios == 1.0, dec * fund, dec * upper / np.sqrt(ratios))
    ms = 4.0 * (0.3 / 4.0) ** float(np.clip(hardness, 0.0, 1.0))
    # UNSOURCED: partial levels (the tables give frequencies only); upper partials are kept 8 dB down
    levels = np.where(ratios == 1.0, 1.0, 0.4 * levels)
    y = _struck(f, t60, levels / (1.0 + (f / 3000.0) ** 2) * (1.0 + (freq / 3000.0) ** 2), n, sr, ms, amp=float(vel))
    s = float(np.clip(sympathetic, 0.0, 1.0))
    if s > 0:
        # ponytail: one-way coupling (struck rod -> collector -> neighbours); a two-way junction only if
        # energy exchange / beating between rods has to be modelled.
        others = np.array([freq * (750.0 / length) ** 2 for length in KIT_LENGTHS_MM if length != 750.0])
        fo = (others[:, None] * np.array(CLAMPED[:2])[None, :]).ravel()
        go = np.tile([1.0, 0.4], others.size) / (1.0 + (fo / 3000.0) ** 2)
        y = y + s * 0.15 * _unit_bank(y, fo, dec * fund / np.sqrt(fo / fo.min()), go, sr)
    y = _radiate(_note_tail(y, dur, sr, tail), sr, radiator, size=0.8,
                 **({"mix": 0.15, "decay": 0.8, "seed": int(seed), "avoid": freq} if radiator == "sheet" else {}))
    return _edges(y, sr, t_in=0.0005) * _LEVEL["rod_bank"]


def struck_rod(length_m, dur=1.0, sr=DEFAULT_SR, diameter_m=0.008, contact_ms=0.5, t60=2.0):
    """One isolated clamped steel rod struck near its tip: cantilever partials on rod_frequency(length)."""
    r = np.array(CLAMPED)
    y = _struck(rod_frequency(length_m, diameter_m) * r, t60 / np.sqrt(r), 1.0 / np.arange(1, 5.0), samples(dur, sr), sr, contact_ms)
    return _edges(y, sr, t_in=0.0005)


def _whistle_ratio(freq):
    """Second strong partial of a whistling blade relative to the heard one. Along the 25-blade SAD set
    (1484 .. 5350 Hz) the interval shrinks from an augmented fourth to a crossing at the middle blade and
    opens again to a major third with the roles swapped (g09 §4, g10 §14.1)."""
    t = float(np.clip(np.log(freq / 1484.0) / np.log(5350.0 / 1484.0), 0.0, 1.0))
    return 2.0 ** (0.5 * (1 - 2 * t)) if t < 0.5 else 2.0 ** (-(2 * t - 1) / 3.0)


@instrument("whistling_blades", "Baschet whistling blades: iron blades struck on the narrow edge, two high partials "
            "whose interval changes along the set, beating at the middle.", family="mallet", span=("F#6", "E8"),
            decay=(3.0, "ring time, s (0.2-12)"),
            second=(0.2, "level of the second partial (0-1); the real blades are nearer 0.5-1, which blurs the pitch"),
            spin=(0.0, "rotation of the blade rack, turns/s (0-8): directivity tremolo 'waa-waa'"),
            hardness=(0.7, "0 soft .. 1 hard stick"))
def whistling_blades(freq, dur, sr=DEFAULT_SR, vel=1.0, decay=3.0, second=0.2, spin=0.0, hardness=0.7):
    """g09 §4, g10 §14.1: pitch range F#6 (1484 Hz) .. E8 (5350 Hz); at the middle blade the two partials
    are a few Hz apart (ombak-like beating, quoted as 4-6 Hz; the measured pair 2870/2917 Hz disagrees, see docs)."""
    f2 = freq * _whistle_ratio(freq)
    if abs(f2 - freq) < 5.0:
        f2 = freq + 5.0                                   # the 4-6 Hz beat of the middle blade
    dec = float(np.clip(decay, 0.2, 12.0))
    tail = float(min(1.5, dec))
    f = np.array([freq, f2])
    g = np.array([1.0, float(np.clip(second, 0.0, 1.0))])
    ms = 1.5 * (0.2 / 1.5) ** float(np.clip(hardness, 0.0, 1.0))
    y = _struck(f, np.array([dec, dec * 0.8]), g, samples(dur + tail, sr), sr, ms, amp=float(vel))
    if spin > 0:
        y = y * (0.65 + 0.35 * np.cos(2 * np.pi * 2 * float(min(spin, 8.0)) * np.arange(y.size) / sr))
    return _edges(_note_tail(y, dur, sr, tail), sr, t_in=0.0005) * _LEVEL["whistling_blades"]


@instrument("tuning_fork", "Aluminium tuning fork struck by a felt hammer, radiating through a balloon or cone "
            "(Nimbus / Clavinimbus).", family="mallet", span=("G3", "D7"),
            radiator=("balloon", "balloon | cone | none"),
            pressure=(0.7, "0-1: how hard the balloon presses on the fork: louder without shortening the note"),
            swell=(0.0, "s: time over which the balloon pressure is applied after the strike (crescendo); 0 = from the start"),
            hardness=(0.25, "0 thick felt (pure tone) .. 1 bare hammer (clang partial at 6.27 x)"),
            decay=(6.0, "ring time, s (0.3-20)"))
def tuning_fork(freq, dur, sr=DEFAULT_SR, vel=1.0, radiator="balloon", pressure=0.7, swell=0.0, hardness=0.25, decay=6.0):
    """g10 §20: 44 aluminium forks G3-D7; felt is added to the hammers 'until the partials vanish'; balloon
    pressure raises loudness without damping. Each prong is a clamped bar, so the clang mode is at 6.267 f."""
    if radiator not in ("balloon", "cone", "none"):
        raise ValueError("radiator must be balloon|cone|none")
    dec = float(np.clip(decay, 0.3, 20.0))
    tail = float(min(1.5, dec))
    h = float(np.clip(hardness, 0.0, 1.0))
    f = freq * np.array(CLAMPED[:2])
    g = np.array([1.0, 0.8 * h / (1.0 + (f[1] / 3000.0) ** 2)])
    y = _struck(f, np.array([dec, 0.06 * dec]), g, samples(dur + tail, sr), sr, 3.0 * (0.4 / 3.0) ** h, amp=float(vel))
    p = float(np.clip(pressure, 0.0, 1.0))
    contact = 0.3 + 0.7 * p
    if swell > 0:
        contact = 0.3 + 0.7 * p * np.clip(np.arange(y.size) / (float(swell) * sr), 0.0, 1.0) ** 2
    y = _note_tail(y * contact, dur, sr, tail)
    return _edges(_radiate(y, sr, radiator, size=0.8), sr, t_in=0.0005) * _LEVEL["tuning_fork"]


@instrument("coil_spring", "Baschet coil spring, struck or scraped: a dispersive train of chirped echoes around a "
            "sustained tone ('something submarine').", family="mallet", span=("C2", "C5"),
            excite=("strike", "strike | scrape (granular, rumbling)"),
            dispersion=(0.5, "0 nearly harmonic echoes .. 1 strong chirps"),
            tone=(0.7, "0 blurred pitch .. 1 the note's own mode clearly on top"),
            decay=(4.0, "ring time, s (0.3-12); real springs ring > 8 s"),
            hardness=(0.5, "0 soft mallet (dark) .. 1 hard stick (laser-like chirp)"), seed=(0, "int: scrape grains"))
def coil_spring(freq, dur, sr=DEFAULT_SR, vel=1.0, excite="strike", dispersion=0.5, tone=0.7, decay=4.0, hardness=0.5, seed=0):
    """Dispersive modal model. Sourced: long ring, very low and very high partials, blurred pitch, scraped
    = granular (g09 §11, g10 §9); a helical spring is bar-like (dispersive, chirping) above its cutoff
    (Bilbao §7.9, g02 §10).
    UNSOURCED: the mode law f_n = f_s n sqrt(1 + B n^2) (stiff-string dispersion standing in for the bar-like
    branch; the two-variable helix scheme of Bilbao §7.9 is not implemented), the ~30 ms round trip, and
    the resonance bump that keeps the note's own mode on top."""
    if excite not in ("strike", "scrape"):
        raise ValueError("excite must be strike|scrape")
    k0 = max(1, int(round(freq / 30.0)))                  # the note is mode k0 of a spring with a ~30 Hz round trip
    bb = 2e-4 * 30.0 ** float(np.clip(dispersion, 0.0, 1.0))
    nn = np.arange(1, 400.0)
    law = nn * np.sqrt(1 + bb * nn * nn)
    f = freq * law / law[k0 - 1]
    f = f[f < min(5000.0, 0.42 * sr)]
    dec = float(np.clip(decay, 0.3, 12.0))
    tail = float(min(1.5, dec))
    t = float(np.clip(tone, 0.0, 1.0))
    g = (0.25 + (4.0 if excite == "strike" else 16.0) * t / (1.0 + ((f - freq) / (0.02 * freq)) ** 2)) / (1.0 + (f / 2500.0) ** 2)
    t60 = dec / (1.0 + (f / 1500.0) ** 2) * np.where(np.abs(f - freq) < 1e-6, 1.0, 0.6)
    n = samples(dur + tail, sr)
    h = float(np.clip(hardness, 0.0, 1.0))
    if excite == "strike":
        y = _struck(f, t60, g, n, sr, 2.5 * (0.25 / 2.5) ** h, amp=float(vel))
    else:
        rng = np.random.default_rng(seed)
        m = min(n, samples(max(dur, 0.05), sr))
        grains = rng.standard_normal(m) * (rng.random(m) < 400.0 / sr) * np.sin(np.linspace(0, np.pi, m)) ** 0.5
        x = np.zeros(n)
        x[:m] = F.lowpass(grains, 600.0 + 3000.0 * h, sr) * float(vel) * 0.25
        y = _ring(x, np.ascontiguousarray(f), np.ascontiguousarray(t60), np.ascontiguousarray(g), float(sr))
        y *= 4.0 * float(vel) / (np.abs(y).max() + 1e-12)       # same peak as a strike, whatever the grains were
    return _edges(_note_tail(y, dur, sr, tail), sr, t_in=0.0005) * _LEVEL["coil_spring"]


# ------------------------------------------------------------------------------------------------
# sound effects: one bristle-friction engine, different bodies and gestures
# ------------------------------------------------------------------------------------------------
def _rub_engine(freqs, xi, mob, og, fn, v, rng, rough, mus=0.975, mud=0.197, vs=0.1, sigma0=1.0e5):
    n = fn.shape[0]
    noise = rough * fn * np.sqrt(np.abs(v) / (np.abs(v).max() + 1e-12)) * F.lowpass(rng.standard_normal(n), 3500.0, FS)
    return elastoplastic_friction(freqs, xi, mob, fn, v, n / FS, mu_s=mus, mu_d=mud, v_s=vs, sigma0=sigma0,
                                  noise=noise, out_gains=og)


def _wander(n, rng, depth, rate=6.0):
    """Slow random multiplier around 1 (hand unsteadiness)."""
    k = max(2, int(n / FS * rate) + 2)
    return 1.0 + depth * np.interp(np.linspace(0, k - 1, n), np.arange(k), rng.uniform(-1, 1, k))


def _sfx_out(out, sr, ref, top=4500.0):
    y = F.lowpass(_to_sr(out / ref, sr), min(top, 0.45 * sr), sr)
    return _edges(0.9 * np.tanh(y), sr, t_in=0.004, t_out=0.02)


@sfx("stick_slip", "Generic stick-slip creak driver: a body dragged through a spring over a rubbing contact.",
     mu_s=(0.975, "static friction coefficient (0.2-3)"), mu_d=(0.197, "dynamic friction coefficient (0.05-mu_s)"),
     stiffness=(2.0e4, "stiffness of the drive / hinge compliance, N/m (1e3-1e6): stiffer = faster creak"),
     speed=(0.01, "drag speed, m/s (0.001-0.5): slow = separate cracks, faster = groan"),
     pressure=(8.0, "normal force, N (0.5-50)"), freq=(420.0, "main resonance of the creaking body, Hz or note"),
     dur=(1.5, "s"), seed=(0, "int"))
def stick_slip(sr=DEFAULT_SR, mu_s=0.975, mu_d=0.197, stiffness=2.0e4, speed=0.01, pressure=8.0, freq=420.0, dur=1.5, seed=0):
    """Elasto-plastic friction (g04 §4.6) on a body whose first mode is the drive compliance: slips repeat
    roughly every 2 (mu_s - mu_d) F_N / (stiffness * speed) seconds and each one rings the body modes.
    (slow drags; once that rate nears the compliance resonance the creak turns into a groan at that resonance).
    UNSOURCED: body mass 0.5 kg, body mode ratios 1 : 1.9 : 3.1 and their damping; Stribeck velocity 0.02 m/s
    (the book's 0.1 m/s leaves the friction curve flat at creak speeds, so no slow stick-slip develops)."""
    rng = np.random.default_rng(seed)
    f0 = to_hz(freq)
    n = int(float(np.clip(dur, 0.05, 30.0)) * FS)
    mus = float(np.clip(mu_s, 0.2, 3.0))
    mud = float(np.clip(mu_d, 0.05, mus))
    mass = 0.5
    k = float(np.clip(stiffness, 1e3, 1e6))
    freqs = np.array([math.sqrt(k / mass) / (2 * np.pi), f0, 1.9 * f0, 3.1 * f0])
    g = _gate(n, n - int(0.03 * FS), 0.02, 0.03)
    v = float(np.clip(speed, 0.001, 0.5)) * _wander(n, rng, 0.35) * (0.3 + 0.7 * g)
    r = _rub_engine(freqs, np.array([0.01, 0.02, 0.025, 0.03]), np.array([1 / mass, 3.0, 2.0, 1.5]),
                    np.array([0.0, 1.0, 1.0, 1.0]), float(np.clip(pressure, 0.5, 50.0)) * g, v, rng, 0.02,
                    mus, mud, vs=0.02, sigma0=2.0e6)
    return _sfx_out(r["out"], sr, 0.01)


_SQUEAK = {   # surface -> (mode Hz, mode ratios, damping, mobility 1/kg, roughness)   # UNSOURCED: tuned by ear, as in the book
    "hinge": (1500.0, (1.0, 2.1), 0.004, 200.0, 0.03),
    "shoe": (1100.0, (1.0, 1.6, 2.7), 0.02, 400.0, 0.08),
    "chalk": (2400.0, (1.0, 1.45), 0.006, 300.0, 0.15)}


@sfx("squeak", "Squeak of a hinge, a rubber sole or chalk: friction locking onto one resonance, with glides.",
     surface=("hinge", "hinge | shoe | chalk"), pressure=(1.0, "normal force, relative (0.2-4): more = rougher, louder"),
     speed=(1.0, "rubbing speed, relative (0.2-3)"), pitch=(1.0, "resonance multiplier (0.5-2)"),
     dur=(0.6, "s"), seed=(0, "int"))
def squeak(sr=DEFAULT_SR, surface="hinge", pressure=1.0, speed=1.0, pitch=1.0, dur=0.6, seed=0):
    """*The Sounding Object* §8.4.2 door squeak: two exciter-resonator pairs for a hinge (one per shutter),
    one for the others; the transients and glides come from the gesture crossing the playable window."""
    if surface not in _SQUEAK:
        raise ValueError(f"surface must be one of {sorted(_SQUEAK)}")
    f0, ratios, xi, mob, rough = _SQUEAK[surface]
    ratios = np.array(ratios)
    rng = np.random.default_rng(seed)
    n = int(float(np.clip(dur, 0.05, 20.0)) * FS)
    g = np.sin(np.linspace(0, np.pi, n)) ** 0.6
    out = np.zeros(n)
    for det in ((1.0, 1.13) if surface == "hinge" else (1.0,)):
        fr = f0 * det * float(np.clip(pitch, 0.5, 2.0)) * ratios
        v = 0.08 * float(np.clip(speed, 0.2, 3.0)) * g * _wander(n, rng, 0.4, 9.0)
        fn = 0.5 * float(np.clip(pressure, 0.2, 4.0)) * (0.4 + 0.6 * g) * _wander(n, rng, 0.25, 5.0)
        out += _rub_engine(fr, xi * np.sqrt(ratios), mob / ratios ** 2, np.ones(ratios.size), fn, v, rng, rough, sigma0=3.0e5)["out"]
    return _sfx_out(out, sr, 0.12)


@sfx("brake_squeal", "Braking wheel: rubbing noise while it is fast, squeal as it slows through the stick-slip range.",
     speed=(1.5, "initial rubbing speed, m/s (0.3-5)"), pressure=(4.0, "brake force, N (0.5-20)"),
     pitch=(1900.0, "main disc resonance, Hz or note"), dur=(1.6, "s: time to stop"), seed=(0, "int"))
def brake_squeal(sr=DEFAULT_SR, speed=1.5, pressure=4.0, pitch=1900.0, dur=1.6, seed=0):
    """*The Sounding Object* §8.4.2: 'neat stick-slip is established only at sufficiently low velocities, and
    brake squeals are produced in the final stage of deceleration'. Nothing is switched on by hand here: the
    Stribeck curve is flat far above v_s = 0.1 m/s and only gets its negative slope as the wheel slows.
    UNSOURCED: disc mode ratios 1 : 1.37 : 2.2, damping, mobility."""
    rng = np.random.default_rng(seed)
    f0 = to_hz(pitch)
    n = int(float(np.clip(dur, 0.2, 20.0)) * FS)
    v = float(np.clip(speed, 0.3, 5.0)) * (1.0 - np.linspace(0.0, 1.0, n)) ** 1.3 + 1e-4
    fn = float(np.clip(pressure, 0.5, 20.0)) * _gate(n, n - int(0.05 * FS), 0.08, 0.05)
    ratios = np.array([1.0, 1.37, 2.2])
    r = _rub_engine(f0 * ratios, 0.002 * np.sqrt(ratios), 60.0 / ratios ** 2, np.ones(3), fn, v, rng, 0.05, sigma0=1.0e6)
    return _sfx_out(r["out"], sr, 0.15)


_RUB = {   # surface -> (mode Hz list, damping, mobility, roughness, mu_s, mu_d)   # UNSOURCED: tuned by ear
    "balloon": ((310.0, 540.0, 890.0, 1420.0), 0.03, 300.0, 0.05, 1.6, 0.4),
    "glass": ((1240.0, 2870.0), 0.002, 80.0, 0.03, 0.975, 0.197),
    "wood": ((240.0, 510.0, 1150.0, 1900.0), 0.06, 120.0, 3.0, 0.5, 0.35)}


@sfx("rub", "A hand rubbing a balloon, a glass pane or a wooden board, back and forth.",
     surface=("balloon", "balloon (squeaky) | glass (clean squeal) | wood (dry, noisy)"),
     pressure=(1.0, "relative normal force (0.2-4)"), speed=(1.0, "relative hand speed (0.2-3)"),
     strokes=(3, "number of back-and-forth strokes in dur"), dur=(1.2, "s"), seed=(0, "int"))
def rub(sr=DEFAULT_SR, surface="balloon", pressure=1.0, speed=1.0, strokes=3, dur=1.2, seed=0):
    """Elasto-plastic friction (g04 §4.6) with the hand velocity reversing at each stroke end; the surface
    roughness enters through the noise term sigma3 w of the friction force."""
    if surface not in _RUB:
        raise ValueError(f"surface must be one of {sorted(_RUB)}")
    freqs, xi, mob, rough, mus, mud = _RUB[surface]
    rng = np.random.default_rng(seed)
    n = int(float(np.clip(dur, 0.05, 30.0)) * FS)
    fr = np.array(freqs)
    stroke = np.sin(np.pi * max(1, int(strokes)) * np.linspace(0.0, 1.0, n))
    v = 0.12 * float(np.clip(speed, 0.2, 3.0)) * stroke * _wander(n, rng, 0.3, 7.0)
    fn = 0.6 * float(np.clip(pressure, 0.2, 4.0)) * (0.25 + 0.75 * np.abs(stroke)) * _wander(n, rng, 0.2, 4.0)
    r = _rub_engine(fr, xi * np.sqrt(fr / fr[0]), mob / (fr / fr[0]) ** 2, np.ones(fr.size), fn, v, rng, rough, mus, mud)
    return _sfx_out(r["out"], sr, 0.12)
