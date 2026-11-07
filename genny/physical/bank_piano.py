"""Modal piano: Steinway D strings, felt hammers and damping, solved as a modal synthesizer.

Blocks:
- main string: one impulse-invariant two-pole resonator per transverse mode, f_k = k f0 sqrt(1 + B k^2),
  A_k = 1 / (pi L mu f_k), decay sigma_k = b1 + b3 w_k^2;
- hammer: mass on a hereditary felt, F = K e^p + R_H d(e^p)/dt, coupled to the string at x_h through
  w_in = w_out = sin(k pi x_h / L); the one-sample delay of the resonators removes the delay-free loop;
- unison: the two or three strings of a key share the hammer force and the bridge. Each partial is solved as a
  coupled system (diagonal mistuning, bridge coupling all-to-all), giving the beating and the two-stage decay;
- longitudinal motion: tension T(t) = T0 + pi^2 ES / 4L^2 sum n^2 y_n^2 plus the dynamic response of the first K
  longitudinal modes, odd modes driven by F_xi,3, even by F_xi,4; free longitudinal decay 0.15 s;
- soundboard: a parallel second-order filter with log-spaced poles, damped mode by mode (sigma = a f^2 + b f).
Options: the second (horizontal) polarisation, the tension fed back into the transverse modes, a bridge
admittance with a resonance, the bridge angle, per-harmonic coupling weights, the sustain pedal (every undamped
string resonating from the bridge), the lid.
"""
from __future__ import annotations

import math

import numba
import numpy as np

from .piano_data import CHABASSIER_D, E_STEEL, steinway_B

_MIDI = np.array([r[0] for r in CHABASSIER_D], float)
TAU_LONG = 0.15
SB_A, SB_B = 2.0e-5, 0.07
Y_BRIDGE = 1.0e-3
AIR_ETA, AIR_NU = 1.832e-5, 1.52e-5
LONG_COUPLING = 1.0
STULOV_EPS, STULOV_TAU = 0.69, 250e-6
UNISON_CENTS = {1: (0.0,), 2: (-0.9, 0.9), 3: (-0.9, 0.0, 0.7)}
TREBLE_MIDI, TREBLE_CENTS = 78, (-1.8, 0.0, 1.6)
BRIDGE_RES_HZ = 470.0
HORIZ_SHARE = 0.1
HORIZ_Y = 0.1
HORIZ_CENTS = 0.3
ORTH_RADIATION = 10 ** (-10.0 / 20.0)
DAMPER_TAU = 0.12
DAMPER_TOP_MIDI = 89
LID_HZ = 2000.0
LID_DB = {"open": 0.0, "half": -5.0, "closed": -10.0}
ACTION_NOISE_S = (0.1, 0.04)
ACTION_BOARD_HZ = 90.0
ACTION_SHOCK_HZ = (300.0, 600.0)
RIB_HZ = 1150.0
RIB_DENSITY = 2.0
RIB_MOBILITY = 0.5
SYMP_FMAX = 4000.0
SYMP_PARTIALS = 10
SYMP_DC_HZ = 10.0


def unison_cents(e_p: float) -> np.ndarray:
    """Cents of a trichord whose tensions are T1, e_p T1 and (2 - e_p) T1 (f grows as sqrt(T))."""
    e = float(np.clip(e_p, 1e-3, 1.0))
    return 600.0 * np.log2(np.array([e, 1.0, 2.0 - e]))


def _row(midi: float) -> dict:
    """Steinway D row interpolated in log scale on the key (lengths, diameters, hammer values)."""
    m = float(np.clip(midi, _MIDI[0], _MIDI[-1]))
    cols = np.array([r[1:] for r in CHABASSIER_D], float)
    v = [float(np.exp(np.interp(m, _MIDI, np.log(cols[:, j])))) for j in range(cols.shape[1])]
    f0, L, d, rho, T0, p, mh_g, kh = v
    return dict(f0=f0, L=L, d=d * 1e-3, rho=rho, T0=T0, p=p, mh=mh_g * 1e-3, K=kh)


def strings_per_key(midi: float) -> int:
    """Singles to E1, bichords over the wound section (to E2), trichords above."""
    return 1 if midi < 29 else 2 if midi < 41 else 3


def damped(midi: float) -> bool:
    """False for the top keys, which have no damper."""
    return midi <= DAMPER_TOP_MIDI


def bridge_admittance(f, peak: float = 0.0, f0: float = BRIDGE_RES_HZ):
    """Complex bridge admittance (s/kg): the flat conductance Y_BRIDGE plus one soundboard mode at f0 whose
    conductance peaks at peak * Y_BRIDGE. The mode's Q comes from the soundboard damping law,
    Q = pi f0 / (SB_A f0^2 + SB_B f0). Its reactive part changes sign at f0."""
    f = np.asarray(f, float)
    if peak <= 0.0:
        return np.full(f.shape, Y_BRIDGE, np.complex128)
    q = math.pi * f0 / (SB_A * f0 * f0 + SB_B * f0)
    r = f / f0
    jw = 1j * r / q
    return Y_BRIDGE + peak * Y_BRIDGE * jw / (1.0 - r * r + jw)


def harmonic_weight(k, w0: float = 1.0, hc: float = 0.0, slope: float = 0.0):
    """Coupling weight per harmonic: w0 up to the cutoff index hc, then falling linearly to 0 over `slope`
    harmonics (hc <= 0: w0 everywhere)."""
    k = np.asarray(k, float)
    if hc <= 0.0:
        return np.full(k.shape, float(w0))
    fall = np.clip(1.0 - (k - hc) / max(float(slope), 1e-9), 0.0, 1.0) if slope > 0.0 else (k <= hc).astype(float)
    return float(w0) * np.where(k <= hc, 1.0, fall)


def long_coupling(angle_deg: float | None) -> float:
    """Gain of the longitudinal force at the bridge: tan(alpha) of a bridge angle alpha in degrees
    (None: the full longitudinal force, the default)."""
    return LONG_COUPLING if angle_deg is None else math.tan(math.radians(float(angle_deg)))


@numba.njit(cache=True)
def _hammer(n_max, sr, a1, a2, bk, win, K, p, RH, mh, v0, eps=0.0, tau0=1.0):
    """Hammer against the main string. Returns the contact force F_h[n] until the hammer has left.
    eps > 0: hereditary law by recursion, I[n] = I[n-1] exp(-dt / tau0) + u^p dt / tau0, F = K (u^p - eps I)."""
    N = a1.shape[0]
    y1 = np.zeros(N)
    y2 = np.zeros(N)
    F = np.zeros(n_max)
    yh, vh = 0.0, v0
    phi_prev = 0.0
    fprev = 0.0
    dt = 1.0 / sr
    left = 0
    mem = 0.0
    q0 = math.exp(-dt / tau0)
    for i in range(n_max):
        ys = 0.0
        for k in range(N):
            yk = -a1[k] * y1[k] - a2[k] * y2[k] + bk[k] * win[k] * fprev
            y2[k] = y1[k]
            y1[k] = yk
            ys += win[k] * yk
        vh -= fprev / mh * dt
        yh += vh * dt
        e = yh - ys
        phi = e ** p if e > 0.0 else 0.0
        mem = mem * q0 + phi * dt / tau0
        f = K * (phi - eps * mem) if eps > 0.0 else K * phi + RH * (phi - phi_prev) * sr
        if f < 0.0 or e <= 0.0:
            f = 0.0
        F[i] = f
        phi_prev = phi
        fprev = f
        if e <= 0.0 and vh < 0.0 and i > 8:
            left += 1
            if left > 32:
                return F[:i + 1]
    return F


@numba.njit(cache=True)
def _transverse(Fh, n, sr, lam, cw, wk):
    """Bridge force of the coupled unison: per partial k and coupled mode j a complex one-pole driven by F_h
    (impulse-invariant, one-sample delay), output Re(-i c_kj s_kj) weighted by w_k = A_k w_in w_b. Each mode
    stops once it has decayed by 100 dB."""
    K, J = lam.shape
    out = np.zeros(n)
    m = Fh.shape[0]
    for k in range(K):
        for j in range(J):
            z = np.exp(lam[k, j] / sr)
            c = -1j * cw[k, j] * wk[k] / sr
            s = 0j
            stop = n if lam[k, j].real > -1e-9 else min(n, m + int(11.5 / -lam[k, j].real * sr))
            for i in range(stop):
                s = z * s + (Fh[i - 1] if 0 < i <= m else 0.0)
                out[i] += (c * s).real
    return out


@numba.njit(cache=True)
def _fxi(y, nk, kq, cX):
    """F_xi,k = F+ + F- on longitudinal mode kq from transverse amplitudes y (mode nk[j] = j + 1)."""
    N = y.shape[0]
    sp = 0.0
    for j in range(min(kq - 1, N)):
        m2 = kq - 2 - j
        if m2 < N:
            sp += nk[m2] * y[m2] * nk[j] * y[j]
    sm = 0.0
    for j in range(N - kq):
        sm += nk[j + kq] * y[j + kq] * nk[j] * y[j]
    return 0.5 * cX * kq * (sp + 2.0 * sm)


@numba.njit(cache=True)
def _longitudinal(Fh, n, sr, a1, a2, bk, win, nk, ES, L, la1, la2, lb, ldc, lk, full=False, back=False, wb=None):
    """Main-string modal amplitudes y_n and the longitudinal bridge force. full: each longitudinal mode k driven
    by its own F+ (m + n = k) and F- (|m - n| = k) instead of F_3 / F_4. back: the tension T(x, t) acts back on
    the transverse modes (uniform part: pitch glide; longitudinal mode q couples modes m, n with q = m + n or
    |m - n|). Returns (longitudinal force, change of the transverse bridge force caused by the feedback)."""
    N = a1.shape[0]
    KL = la1.shape[0]
    y1 = np.zeros(N)
    y2 = np.zeros(N)
    z1 = np.zeros(N)
    z2 = np.zeros(N)
    l1 = np.zeros(KL)
    l2 = np.zeros(KL)
    out = np.zeros(n)
    corr = np.zeros(n)
    Q = np.zeros(N)
    tq = np.zeros(KL)
    m = Fh.shape[0]
    cT = math.pi ** 2 * ES / (4.0 * L * L)
    cX = -2.0 * ES * math.pi ** 3 / (8.0 * L * L)
    cB = ES * math.pi / L
    kp = math.pi / L
    fo = 0.0
    fe = 0.0
    for i in range(n):
        f = Fh[i - 1] if 0 < i <= m else 0.0
        tens = 0.0
        cs = 0.0
        for k in range(N):
            yk = -a1[k] * y1[k] - a2[k] * y2[k] + bk[k] * (win[k] * f + Q[k])
            y2[k] = y1[k]
            y1[k] = yk
            tens += (nk[k] * yk) ** 2
            if back:
                zk = -a1[k] * z1[k] - a2[k] * z2[k] + bk[k] * win[k] * f
                z2[k] = z1[k]
                z1[k] = zk
                cs += wb[k] * (yk - zk)
        corr[i] = cs
        so = 0.0
        se = 0.0
        for k in range(N - 4):
            a = nk[k] * y1[k]
            so += a * (nk[k] + 3.0) * y1[k + 3]
            se += a * (nk[k] + 4.0) * y1[k + 4]
        fo = cX * 3.0 * so
        fe = cX * 4.0 * se
        dyn = 0.0
        for q in range(KL):
            drive = fo if lk[q] % 2 == 1 else fe
            if full:
                drive = _fxi(y1, nk, int(lk[q]), cX)
            x = -la1[q] * l1[q] - la2[q] * l2[q] + lb[q] * drive
            l2[q] = l1[q]
            l1[q] = x
            tq[q] = cB * lk[q] * (x - ldc[q] * drive)
            dyn += lk[q] * (x - ldc[q] * drive)
        out[i] = cT * tens + cB * dyn
        if back:
            dT = cT * tens
            for j in range(N):
                nj = nk[j]
                Q[j] = -dT * (nj * kp) ** 2 * 0.5 * L * y1[j]
            for q in range(KL):
                kq = int(lk[q])
                c = tq[q] * kp * kp * 0.25 * L
                for j in range(N):
                    nj = int(nk[j])
                    s = 0.0
                    mm = kq - nj
                    if 1 <= mm <= N:
                        s += mm * y1[mm - 1]
                    mm = nj + kq
                    if mm <= N:
                        s += mm * y1[mm - 1]
                    mm = nj - kq
                    if mm >= 1:
                        s += mm * y1[mm - 1]
                    Q[j] -= c * nj * s
    return out, corr


def _resonator(f, sigma, A, sr):
    """Impulse-invariant two-pole of A e^{-sigma t} sin(2 pi f t): (a1, a2, b)."""
    p = np.exp(2j * np.pi * f / sr) * np.exp(-sigma / sr)
    return -2.0 * p.real, np.abs(p) ** 2, A / sr * p.imag


def soundboard_filter(sr: int, n_poles: int = 100, f_lo: float = 30.0, seed: int = 0, ribs: bool = False):
    """(a1, a2, d0, d1) of the fixed-pole parallel filter: poles log-spaced from f_lo to 0.45 sr, radii from the
    soundboard damping law. Without a measured response to fit, the numerators have unit magnitude and random
    sign (a flat, dense modal response). ``ribs``: above RIB_HZ the modes are confined between the ribs,
    RIB_DENSITY times as many poles there, each with RIB_MOBILITY of the gain."""
    f = np.geomspace(f_lo, 0.45 * sr, n_poles)
    g = np.ones(n_poles)
    if ribs and RIB_HZ < 0.45 * sr:
        lo = f[f < RIB_HZ]
        hi = np.geomspace(RIB_HZ, 0.45 * sr, int(RIB_DENSITY * (n_poles - lo.size)))
        f = np.concatenate((lo, hi))
        g = np.concatenate((np.ones(lo.size), np.full(hi.size, RIB_MOBILITY)))
        n_poles = f.size
    sigma = SB_A * f * f + SB_B * f
    p = np.exp(2j * np.pi * f / sr - sigma / sr)
    rng = np.random.default_rng(1234 + seed)
    d0 = g * rng.choice([-1.0, 1.0], n_poles) * sigma / sigma.max()   # peak gain ~ d / sigma: the same at every pole
    return -2.0 * p.real, np.abs(p) ** 2, d0, -d0 * p.real


def parallel_poles(sr: int, n_poles: int = 100, R: float = 0.98, f_lo: float = 30.0, f_hi: float | None = None):
    """(a1, a2) of the fixed poles: theta_k = 2 pi f_k / fs on a log scale, |p_k| = R^(theta_k / pi)."""
    th = 2.0 * np.pi * np.geomspace(f_lo, f_hi or 0.45 * sr, n_poles) / sr
    p = R ** (th / np.pi) * np.exp(1j * th)
    return -2.0 * p.real, np.abs(p) ** 2


def fit_parallel_filter(h: np.ndarray, sr: int, n_poles: int = 100, R: float = 0.98, M: int = 0,
                        f_lo: float = 30.0, f_hi: float | None = None):
    """Fixed-pole parallel filter fitted to an impulse response h: the section outputs to an
    impulse and z^-m (m = 0..M; M = -1: no direct path) are the basis, d_k0, d_k1, b_m by least squares.
    Returns (a1, a2, d0, d1, b)."""
    from scipy.signal import lfilter
    h = np.asarray(h, float)
    a1, a2 = parallel_poles(sr, n_poles, R, f_lo, f_hi)
    imp = np.zeros(h.size)
    imp[0] = 1.0
    cols = []
    for k in range(a1.size):
        u = lfilter([1.0], [1.0, a1[k], a2[k]], imp)
        cols += [u, np.concatenate(([0.0], u[:-1]))]
    for m in range(M + 1):
        cols.append(np.roll(imp, m))
    c = np.linalg.lstsq(np.stack(cols, 1), h, rcond=None)[0]
    K2 = 2 * a1.size
    return a1, a2, c[0:K2:2], c[1:K2:2], c[K2:]


def soundboard(x: np.ndarray, sr: int, seed: int = 0, fit=None, ribs: bool = False) -> np.ndarray:
    """Soundboard filter; ``fit`` = (a1, a2, d0, d1, b) from ``fit_parallel_filter`` on a measured response."""
    from scipy.signal import lfilter
    a1, a2, d0, d1, b = fit if fit is not None else (*soundboard_filter(sr, seed=seed, ribs=ribs), np.zeros(0))
    y = np.convolve(x, b)[:x.size] if b.size else np.zeros_like(x)
    for k in range(a1.shape[0]):
        y += lfilter([d0[k], d1[k]], [1.0, a1[k], a2[k]], x)
    return y


_board_filter = soundboard


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


def air_rate(f, d, rho_s):
    """Amplitude decay rate (1/s) of a string of diameter d, density rho_s, at frequency f, from air viscosity alone."""
    f = np.asarray(f, float)
    M = 0.25 * d * np.sqrt(2.0 * np.pi * f / AIR_NU)
    return np.pi * f * (AIR_ETA / AIR_NU) / rho_s * (2.0 * math.sqrt(2.0) * M + 1.0) / (M * M)


AIR_RHO = 1.2


def air_rate_vc(f, d, rho_s):
    """Valette-Cuesta air damping of a string, sigma = pi (eta + d sqrt(pi eta rho_air f)) / (rho_s S), S = pi d^2 / 4."""
    f = np.asarray(f, float)
    return np.pi * (AIR_ETA + d * np.sqrt(np.pi * AIR_ETA * AIR_RHO * f)) / (rho_s * np.pi * d * d / 4.0)


BANK_COND_MAX = 1e6


def _b3(midi: float) -> float:
    return float(np.exp(np.interp(midi, [60.0, 96.0], [np.log(6.25e-9), np.log(2.6e-10)])))


def place_longitudinal(fl, fk, sigma, guard: float = 3.0):
    """Longitudinal mode frequencies moved off the transverse peaks: a mode closer to a partial than `guard`
    half-power bandwidths (sigma / pi Hz) is pushed to that distance, on its own side."""
    fl = np.array(fl, float)
    fk = np.asarray(fk, float)
    bw = guard * np.asarray(sigma, float) / math.pi
    for i, f in enumerate(fl):
        j = int(np.argmin(np.abs(fk - f)))
        d = f - fk[j]
        if abs(d) < bw[j]:
            fl[i] = fk[j] + (bw[j] if d >= 0.0 else -bw[j])
    return fl


def _action(n, sr, level):
    """Key and key-bed transient at the bridge: a percussive component 20 ms before hammer contact, a wave package
    near 1000 Hz (key), then key-bed resonances near 100 and 250 Hz, all of the order of the string's pp level."""
    t = np.arange(n) / sr
    click = np.exp(-t / 0.001) * np.sin(2 * np.pi * 1000.0 * t)
    key = np.exp(-t / 0.008) * np.sin(2 * np.pi * 1000.0 * t)
    bed = (np.exp(-t / 0.03) * np.sin(2 * np.pi * 100.0 * t) + np.exp(-t / 0.02) * np.sin(2 * np.pi * 250.0 * t))
    i = int(0.02 * sr)
    y = click + np.concatenate((np.zeros(i // 2), key))[:n] + np.concatenate((np.zeros(i), bed))[:n]
    return level * y / np.max(np.abs(y))


def _action_noise(n, sr, level, midi, seed=0):
    """Attack noise as two filtered noises: the soundboard thump near ACTION_BOARD_HZ and the shock in
    ACTION_SHOCK_HZ, lasting ACTION_NOISE_S from the lowest key to the highest."""
    from ..filters import bandpass
    rng = np.random.default_rng(77 + seed)
    t = np.arange(n) / sr
    tau = float(np.interp(midi, (21.0, 108.0), ACTION_NOISE_S)) / 3.0
    env = np.exp(-t / tau) * np.minimum(t / 0.001, 1.0)
    lo, hi = ACTION_SHOCK_HZ
    fc = math.sqrt(lo * hi)
    y = (bandpass(rng.standard_normal(n), ACTION_BOARD_HZ, sr, 1.0)
         + bandpass(rng.standard_normal(n), fc, sr, fc / (hi - lo))) * env
    return level * y / (np.max(np.abs(y)) + 1e-30)


def sympathetic(force: np.ndarray, sr: int, struck_midi: float, pedal: bool = True, loss: float = 1.0) -> np.ndarray:
    """Bridge force of the undamped strings driven by the bridge (one way, so always stable): the bridge moves by
    d = Y_BRIDGE * integral(force); each mode n of a free string answers through its end with the generalised
    force T0 (n pi / L) d, and pushes back on the bridge with T0 (n pi / L) y_n. pedal: every key is free; otherwise
    only the keys without damper. The struck key itself is left out."""
    from scipy.signal import lfilter
    d = lfilter([Y_BRIDGE / sr], [1.0, -math.exp(-2.0 * math.pi * SYMP_DC_HZ / sr)], force)
    out = np.zeros_like(force)
    me = int(round(struck_midi))
    for m in range(21, 109):
        if m == me or (not pedal and damped(m)):
            continue
        r = _row(m)
        f0 = 440.0 * 2.0 ** ((m - 69) / 12.0)
        B = steinway_B(m)
        S = math.pi * r["d"] ** 2 / 4.0
        mu = r["rho"] * S
        k = np.arange(1, SYMP_PARTIALS + 1, dtype=float)
        fk = k * f0 * np.sqrt(1.0 + B * k * k) / math.sqrt(1.0 + B)
        keep = fk < min(SYMP_FMAX, 0.45 * sr)
        k, fk = k[keep], fk[keep]
        if not k.size:
            continue
        w = 2 * np.pi * fk
        b3 = _b3(m)
        sigma = float(loss) * (0.5 + b3 * w * w)
        ns = strings_per_key(m)
        eta = w * math.sqrt(r["T0"] * mu) * Y_BRIDGE / math.pi
        share = np.minimum(eta, np.maximum(sigma - float(loss) * b3 * w * w - air_rate(fk, r["d"], r["rho"]), 0.0) / ns)
        g = (r["T0"] * k * math.pi / r["L"]) ** 2 * share / eta
        cents = np.asarray(TREBLE_CENTS if ns == 3 and m >= TREBLE_MIDI else UNISON_CENTS[ns])
        for c in cents:
            a1, a2, b = _resonator(fk * 2.0 ** (c / 1200.0), sigma, 1.0 / (math.pi * r["L"] * mu * fk), sr)
            for j in range(k.size):
                out += g[j] * (lfilter([0.0, b[j]], [1.0, a1[j], a2[j]], d) - b[j] / (1.0 + a1[j] + a2[j]) * d)
    return out


def lid(x: np.ndarray, sr: int, state: str = "open") -> np.ndarray:
    """Lid open (unchanged), half-stick or closed: a high shelf above LID_HZ."""
    if state not in LID_DB:
        raise ValueError(f"lid must be one of {sorted(LID_DB)}, not {state!r}")
    if LID_DB[state] == 0.0:
        return x
    from ..filters import highshelf
    return highshelf(x, LID_HZ, sr, LID_DB[state])


def note(freq: float, dur: float, sr: int = 48000, vel: float = 0.8, *, tail: float = 1.0, loss: float = 1.0,
         hardness: float = 1.0, max_modes: int = 140, seed: int = 0, una_corda: bool = False,
         hammer: str = "chabassier", action: float = 0.0, long_full: bool = False, soundboard_ir=None,
         tuning_factor: float | None = None, soundboard: str = "filter", polarisations: int = 1,
         bidirectional: bool = False, bridge_res: float = 0.0, bridge_angle: float | None = None,
         weighting: tuple | None = None, long_place: bool = False, pedal: bool = False,
         sympathetic_strings: bool = False, lid_state: str = "open", ribs: bool = False,
         action_kind: str = "tones", air: str = "stokes") -> np.ndarray:
    """One piano key: ``vel`` 0..1 maps to a hammer speed of 0.5..4.5 m/s; ``loss`` scales every string decay rate,
    ``hardness`` the felt stiffness K.
    ``una_corda``: the action shifts and the hammer misses one string of a bichord or trichord.
    ``hammer``: "chabassier" (first-order felt) or "stulov" (hereditary memory). ``action`` > 0 adds the key and
    key-bed transient at that multiple of its measured level, and delays the string by the 20 ms that precede contact.
    ``long_full``: every longitudinal mode gets its own F+ and F- drive instead of the shared F_3 / F_4.
    ``tuning_factor`` e_p (0 < e_p < 1) on a trichord: tensions T1, e_p T1, (2 - e_p) T1, so cents = 600 log2(T/T1).
    ``soundboard_ir``: a measured soundboard impulse response, fitted by ``fit_parallel_filter``.
    ``soundboard``: "filter" (parallel modal filter) or "fdn" (the board as a reverberator: the bridge signal plus a
    feedback-delay-network tail ringing long in the bass and short in the treble, at the filter's RMS level).
    ``polarisations`` 2: the horizontal plane (hammer share 0.1, 0.3 cent sharp, a tenth of the bridge loss, radiated
    10 dB down): a slow second decay, and its slopes add to the tension (more phantom partials).
    ``bidirectional``: the tension acts back on the transverse modes (pitch glide of a hard blow, mode exchange).
    ``bridge_res``: conductance peak of a soundboard mode at 470 Hz, in multiples of the flat bridge conductance
    (0 = flat; about 9 for a strong resonance); the coupling of the unison then follows the frequency.
    ``bridge_angle``: degrees; the longitudinal force reaches the board as tan(angle) (None = full force).
    ``weighting`` (w0, hc, slope): per-harmonic weight of the bridge coupling, w0 up to harmonic hc, then a linear fall.
    ``long_place``: longitudinal modes moved off the transverse peaks (no metallic ringing).
    ``pedal``: sustain pedal: no damper at note-off, and every other string resonates through the bridge.
    ``sympathetic_strings``: without pedal, the undamped top strings resonate. ``lid_state``: open | half | closed. ``ribs``: soundboard modes confined between the
    ribs above RIB_HZ. ``action_kind``: tones (key and key-bed tones) | noise (two filtered noises)."""
    if soundboard not in ("filter", "fdn"):
        raise ValueError(f"soundboard must be 'filter' or 'fdn', not {soundboard!r}")
    midi = 69.0 + 12.0 * math.log2(max(freq, 1.0) / 440.0)
    r = _row(midi)
    L, T0 = r["L"], r["T0"]
    S = math.pi * r["d"] ** 2 / 4.0
    mu = r["rho"] * S
    B = steinway_B(midi)
    f0 = float(freq)
    nmax = int(min(max_modes, max(1, 0.45 * sr / f0)))
    k = np.arange(1, nmax + 1, dtype=float)
    fk = k * f0 * np.sqrt(1.0 + B * k * k) / math.sqrt(1.0 + B)
    keep = fk < 0.45 * sr
    k, fk = k[keep], fk[keep]
    wk_ = 2 * np.pi * fk
    b3 = _b3(midi)
    sigma = float(loss) * (0.5 + b3 * wk_ * wk_)
    A = 1.0 / (math.pi * L * mu * fk)
    xh = 0.12
    win = np.sin(k * np.pi * xh)
    wb = k * np.pi * T0 / L
    a1, a2, bk = _resonator(fk, sigma, A, sr)
    K = r["K"] * float(hardness)
    RH = 2.434e5 * (1.477e6 / 2.434e5) ** ((float(np.clip(midi, 24.0, 107.0)) - 24.0) / 83.0) * float(hardness)
    v0 = 0.5 + 4.0 * float(np.clip(vel, 0.0, 1.0)) ** 1.5
    os_ = 4
    h1, h2, hb = _resonator(fk, sigma, A, sr * os_)
    if hammer not in ("chabassier", "stulov"):
        raise ValueError(f"unknown hammer {hammer!r}; choose from ['chabassier', 'stulov']")
    eps, tau0 = (STULOV_EPS, STULOV_TAU) if hammer == "stulov" else (0.0, 1.0)
    Fh = _hammer(int(0.05 * sr * os_), float(sr * os_), h1, h2, hb, win, K, r["p"], RH, r["mh"], v0, eps, tau0)
    from scipy.signal import resample_poly
    Fh = np.maximum(resample_poly(Fh, 1, os_), 0.0)
    n = int((float(dur) + float(tail)) * sr)
    ns = strings_per_key(midi)
    cents = np.asarray(TREBLE_CENTS if ns == 3 and midi >= TREBLE_MIDI else UNISON_CENTS[ns])
    if tuning_factor is not None and ns == 3:
        cents = unison_cents(tuning_factor)
    lam = np.zeros((k.size, ns), np.complex128)
    cw = np.zeros((k.size, ns), np.complex128)
    one = np.ones(ns)
    struck = one.copy()
    if una_corda and ns > 1:
        struck[-1] = 0.0
    floor = float(loss) * b3 * wk_ * wk_ + (air_rate_vc if air == "vc" else air_rate)(fk, r["d"], r["rho"])
    if bridge_res > 0.0:
        Y = bridge_admittance(fk, float(bridge_res))
        eta = wk_ * math.sqrt(T0 * mu) * Y.real / math.pi
        zeta = np.minimum(eta, np.maximum(sigma - floor, 0.0) / ns)
        react = zeta * Y.imag / Y.real
    else:
        eta = wk_ * math.sqrt(T0 * mu) * Y_BRIDGE / math.pi
        zeta = np.minimum(eta, np.maximum(sigma - floor, 0.0) / ns)
        react = zeta * np.random.default_rng(4321 + seed).choice([-1.0, 1.0], k.size)
    if weighting is not None:
        wgt = harmonic_weight(k, *weighting)
        zeta, react = zeta * wgt, react * wgt
    for q in range(k.size):
        M = np.diag(-(sigma[q] - ns * zeta[q]) + 1j * wk_[q] * 2.0 ** (cents / 1200.0)).astype(np.complex128) \
            + (-zeta[q] + 1j * react[q]) * np.ones((ns, ns))
        ev, V = np.linalg.eig(M)
        if np.linalg.cond(V) > BANK_COND_MAX:
            ev, V = np.diag(M).copy(), np.eye(ns, dtype=np.complex128)
        U = np.linalg.inv(V)
        lam[q] = ev
        cw[q] = (one @ V) * (U @ struck) / struck.sum()
    y = _transverse(Fh, n, float(sr), lam, cw, A * win * wb)
    two = int(polarisations) >= 2
    if two:
        sig_h = floor + HORIZ_Y * np.maximum(sigma - floor, 0.0)
        w_h = wk_ * 2.0 ** (HORIZ_CENTS / 1200.0)
        lam_h = (-sig_h + 1j * w_h).astype(np.complex128)[:, None]
        cw_h = np.full((k.size, 1), float(struck.sum()), np.complex128)
        y = y + ORTH_RADIATION * _transverse(HORIZ_SHARE * Fh, n, float(sr), lam_h, cw_h, A * win * wb)
    ES = E_STEEL * S
    fl0 = math.sqrt(ES / mu) / (2.0 * L)
    Hf = np.abs(np.fft.rfft(Fh, 8192))
    band = np.fft.rfftfreq(8192, 1.0 / sr)[np.nonzero(Hf > 0.01 * Hf.max())[0].max()]
    lk = np.arange(1, 11, dtype=float)
    lk = lk[(lk * fl0 < 0.45 * sr) & (lk * fl0 < 2.0 * band)]
    if lk.size and nmax > 6:
        fl = lk * fl0
        if long_place:
            fl = place_longitudinal(fl, fk, sigma)
        la1, la2, lb = _resonator(fl, np.full(lk.size, 1.0 / TAU_LONG), 1.0 / (math.pi * L * mu * fl), sr)
        ldc = lb / (1.0 + la1 + la2)
        half = fk < 0.25 * sr
        back = bool(bidirectional)
        yl, _ = _longitudinal(Fh, n, float(sr), a1[half], a2[half], bk[half], win[half], k[half], ES, L,
                              la1, la2, lb, ldc, lk, bool(long_full), False, wb[half])
        if back:
            _, corr = _longitudinal(Fh / struck.sum(), n, float(sr), a1[half], a2[half],
                                    bk[half], win[half], k[half], ES, L, la1, la2, lb, ldc, lk, bool(long_full), True,
                                    wb[half])
            y = y + struck.sum() * corr
        if two:
            sh = sig_h[half]
            fh_ = fk[half] * 2.0 ** (HORIZ_CENTS / 1200.0)
            c1, c2, cb = _resonator(fh_, sh, A[half], sr)
            yh, _ = _longitudinal(HORIZ_SHARE * Fh, n, float(sr), c1, c2, cb, win[half], k[half], ES, L,
                                  la1, la2, lb, ldc, lk, bool(long_full), False, wb[half])
            yl = yl + yh
        y = y + long_coupling(bridge_angle) * np.diff(yl, prepend=yl[0])
    t_off = int(float(dur) * sr)
    if t_off < n and damped(midi) and not pedal:
        y[t_off:] *= np.exp(-np.arange(n - t_off) / (DAMPER_TAU * sr))
    if pedal or sympathetic_strings:
        y = y + sympathetic(y, sr, midi, pedal=bool(pedal), loss=loss)
    if action > 0.0:
        h = _hammer(int(0.05 * sr * os_), float(sr * os_), h1, h2, hb, win, K, r["p"], RH, r["mh"], 0.5, eps, tau0)
        h = np.maximum(resample_poly(h, 1, os_), 0.0)
        pp = np.max(np.abs(_transverse(h, int(0.1 * sr), float(sr), lam, cw, A * win * wb)))
        i = int(0.02 * sr)
        lev = float(action) * pp * v0 / 4.0
        if action_kind == "noise":
            a = _action_noise(y.size + i, sr, lev, midi, seed)
        elif action_kind == "tones":
            a = _action(y.size + i, sr, lev)
        else:
            raise ValueError("action_kind must be tones or noise")
        y = np.concatenate((np.zeros(i), y)) + a
    board = _board_filter(y, sr, seed, None if soundboard_ir is None else fit_parallel_filter(soundboard_ir, sr), ribs=ribs)
    if soundboard == "fdn":
        from ..reverbs import soundboard_fdn
        wet = soundboard_fdn(y, sr, seed=seed)
        wet = y + wet * _rms(y) / (_rms(wet) + 1e-30)
        board = wet * _rms(board) / (_rms(wet) + 1e-30)
    y = lid(board, sr, lid_state)
    return y * 10 ** (-1.0 / 20.0) if una_corda else y
