"""Modal piano after Bank, Zambon & Fontana, "A modal-based real-time piano synthesizer", IEEE TASLP 18(4),
809-821 (2010), with the Steinway D strings, hammers and damping of Chabassier & Durufle, INRIA RT-0425 (2012).

Blocks (Bank Fig. 2):
- main string: one impulse-invariant two-pole resonator per transverse mode (eqs 3-11), f_k = k f0 sqrt(1 + B k^2),
  A_k = 1 / (pi L mu f_k), decay sigma_k = b1 + b3 w_k^2 (Chaigne & Askenfelt 1994, Table I);
- hammer: mass on a hereditary felt, F = K e^p + R_H d(e^p)/dt (Chabassier appendix A, Stulov's dissipative law),
  coupled to the string at x_h through w_in = w_out = sin(k pi x_h / L) (eqs 12-15); the one-sample delay of the
  impulse-invariant resonators removes the delay-free loop (Sec. IV);
- unison: the two or three strings of a key share the hammer force and the bridge. Each partial is solved as
  Weinreich's coupled system (diagonal mistuning, bridge coupling all-to-all), giving the beating and the two-stage
  decay that Bank's secondary resonators imitate (Sec. VI-A);
- longitudinal motion: tension T(t) = T0 + pi^2 ES / 4L^2 sum n^2 y_n^2 (eq. 26) plus the dynamic response of the
  first K longitudinal modes (eqs 22b, 27, 28), odd modes driven by F_xi,3, even by F_xi,4 (Sec. V-B2); free
  longitudinal decay 0.15 s (Bank, JASA 118 (2005), Fig. 3);
- soundboard: Bank's parallel second-order filter (eqs 30-32) with log-spaced poles, damped mode by mode as the
  Chabassier soundboard (sigma = a f^2 + b f, a 2e-5, b 0.07: loss factor 2-3 %).
"""
from __future__ import annotations

import math

import numba
import numpy as np

from .piano_data import CHABASSIER_D, E_STEEL, steinway_B

_MIDI = np.array([r[0] for r in CHABASSIER_D], float)
TAU_LONG = 0.15            # s, free longitudinal mode (Bank 2005, F1)
SB_A, SB_B = 2.0e-5, 0.07  # Chabassier soundboard modal damping (eq. 1)
# Weinreich unison. UNSOURCED: half of the prompt decay goes through the bridge (the aftersound decays at half the
# prompt rate), and the unison spread (cents) is genny's earlier SympatheticPiano default
BRIDGE_SHARE = 0.5
# UNSOURCED: the soundboard takes the longitudinal (in-plane) bridge force 20 dB weaker than the transverse one;
# Bank & Sujbert 2005 only say it responds differently (their high-pass H_l)
LONG_COUPLING = 0.1
UNISON_CENTS = {1: (0.0,), 2: (-0.9, 0.9), 3: (-1.8, 0.0, 1.6)}


def _row(midi: float) -> dict:
    """Chabassier row interpolated in log scale on the key (lengths, diameters, hammer values)."""
    m = float(np.clip(midi, _MIDI[0], _MIDI[-1]))
    cols = np.array([r[1:] for r in CHABASSIER_D], float)
    v = [float(np.exp(np.interp(m, _MIDI, np.log(cols[:, j])))) for j in range(cols.shape[1])]
    f0, L, d, rho, T0, p, mh_g, kh = v
    return dict(f0=f0, L=L, d=d * 1e-3, rho=rho, T0=T0, p=p, mh=mh_g * 1e-3, K=kh)


def strings_per_key(midi: float) -> int:
    # UNSOURCED: Steinway D stringing, singles to E1, bichords over the wound section (to E2), trichords above
    return 1 if midi < 29 else 2 if midi < 41 else 3


@numba.njit(cache=True)
def _hammer(n_max, sr, a1, a2, bk, win, K, p, RH, mh, v0):
    """Hammer against the main string (Bank Fig. 3). Returns the contact force F_h[n] until the hammer has left."""
    N = a1.shape[0]
    y1 = np.zeros(N)
    y2 = np.zeros(N)
    F = np.zeros(n_max)
    yh, vh = 0.0, v0
    phi_prev = 0.0
    fprev = 0.0
    dt = 1.0 / sr
    left = 0
    for i in range(n_max):
        # string at the hammer: resonator output (one-sample delay inside, so it only depends on past force)
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
        f = K * phi + RH * (phi - phi_prev) * sr
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
def _longitudinal(Fh, n, sr, a1, a2, bk, win, nk, ES, L, la1, la2, lb, ldc, lk):
    """Main-string modal amplitudes y_n and the longitudinal bridge force (Bank eqs 22b, 26-28)."""
    N = a1.shape[0]
    KL = la1.shape[0]
    y1 = np.zeros(N)
    y2 = np.zeros(N)
    l1 = np.zeros(KL)
    l2 = np.zeros(KL)
    out = np.zeros(n)
    m = Fh.shape[0]
    cT = math.pi ** 2 * ES / (4.0 * L * L)
    cX = -2.0 * ES * math.pi ** 3 / (8.0 * L * L)
    cB = ES * math.pi / L
    fo = 0.0
    fe = 0.0
    for i in range(n):
        f = Fh[i - 1] if 0 < i <= m else 0.0
        tens = 0.0
        for k in range(N):
            yk = -a1[k] * y1[k] - a2[k] * y2[k] + bk[k] * win[k] * f
            y2[k] = y1[k]
            y1[k] = yk
            tens += (nk[k] * yk) ** 2
        # F_xi,k^- for k = 3 (odd modes) and k = 4 (even modes), eq. 22b
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
            x = -la1[q] * l1[q] - la2[q] * l2[q] + lb[q] * drive
            l2[q] = l1[q]
            l1[q] = x
            dyn += lk[q] * (x - ldc[q] * drive)     # dynamic part: resonator minus its static (DC) response, eq. 28
        out[i] = cT * tens + cB * dyn
    return out


def _resonator(f, sigma, A, sr):
    """Impulse-invariant two-pole of A e^{-sigma t} sin(2 pi f t) (Bank eq. 11): (a1, a2, b)."""
    p = np.exp(2j * np.pi * f / sr) * np.exp(-sigma / sr)
    return -2.0 * p.real, np.abs(p) ** 2, A / sr * p.imag


def soundboard_filter(sr: int, n_poles: int = 100, f_lo: float = 30.0, seed: int = 0):
    """(a1, a2, d0, d1) of Bank's fixed-pole parallel filter (eqs 30-32): poles log-spaced from f_lo to 0.45 sr,
    radii from the Chabassier soundboard damping. UNSOURCED: Bank fits d to a measured response; with none at hand
    the numerators have unit magnitude and random sign (a flat, dense modal response)."""
    f = np.geomspace(f_lo, 0.45 * sr, n_poles)
    sigma = SB_A * f * f + SB_B * f
    p = np.exp(2j * np.pi * f / sr - sigma / sr)
    rng = np.random.default_rng(1234 + seed)
    d0 = rng.choice([-1.0, 1.0], n_poles) * sigma / sigma.max()   # peak gain ~ d / sigma: the same at every pole
    return -2.0 * p.real, np.abs(p) ** 2, d0, -d0 * p.real


def soundboard(x: np.ndarray, sr: int, seed: int = 0) -> np.ndarray:
    from scipy.signal import lfilter
    a1, a2, d0, d1 = soundboard_filter(sr, seed=seed)
    y = np.zeros_like(x)
    for k in range(a1.shape[0]):
        y += lfilter([d0[k], d1[k]], [1.0, a1[k], a2[k]], x)
    return y


def note(freq: float, dur: float, sr: int = 48000, vel: float = 0.8, *, tail: float = 1.0, loss: float = 1.0,
         hardness: float = 1.0, max_modes: int = 140, seed: int = 0) -> np.ndarray:
    """One piano key: ``vel`` 0..1 maps to a hammer speed of 0.5..5.5 m/s (UNSOURCED: Askenfelt & Jansson's
    pp-ff range); ``loss`` scales every string decay rate, ``hardness`` the felt stiffness K."""
    midi = 69.0 + 12.0 * math.log2(max(freq, 1.0) / 440.0)
    r = _row(midi)
    i_key = float(np.clip(midi - 20.0, 1.0, 88.0))
    L, T0 = r["L"], r["T0"]
    S = math.pi * r["d"] ** 2 / 4.0
    mu = r["rho"] * S
    B = steinway_B(midi)
    f0 = float(freq)                                           # tuned to the asked pitch; geometry from the table
    nmax = int(min(max_modes, max(1, 0.45 * sr / f0)))
    k = np.arange(1, nmax + 1, dtype=float)
    fk = k * f0 * np.sqrt(1.0 + B * k * k) / math.sqrt(1.0 + B)
    keep = fk < 0.45 * sr
    k, fk = k[keep], fk[keep]
    wk_ = 2 * np.pi * fk
    # Chaigne & Askenfelt 1994 Table I, fitted to measured tones: b1 0.5 1/s, b3 6.25e-9 s (C2, C4) -> 2.6e-10 s (C7).
    # Chabassier's R, eta are internal losses only (their soundboard takes the rest) and leave A0 ringing for minutes.
    b3 = float(np.exp(np.interp(midi, [60.0, 96.0], [np.log(6.25e-9), np.log(2.6e-10)])))
    sigma = float(loss) * (0.5 + b3 * wk_ * wk_)
    A = 1.0 / (math.pi * L * mu * fk)
    xh = 0.12                                                  # x_H / L of every Steinway D key (appendix A)
    win = np.sin(k * np.pi * xh)
    wb = k * np.pi * T0 / L
    a1, a2, bk = _resonator(fk, sigma, A, sr)
    # hammer: K and p from the table; R_H from appendix A, geometric from 2.434e5 (C1) to 1.477e6 (B7) N m^-p s
    # (their eq. 5 as printed gives values up to 1e7 times larger)
    K = r["K"] * float(hardness)
    RH = 2.434e5 * (1.477e6 / 2.434e5) ** ((float(np.clip(midi, 24.0, 107.0)) - 24.0) / 83.0) * float(hardness)
    v0 = 0.5 + 4.0 * float(np.clip(vel, 0.0, 1.0)) ** 1.5
    # the felt is stiff in the treble (K 1e11, p 2.8): at 48 kHz the contact chatters into spikes. Solved at 4x
    # (Chaigne & Askenfelt ran C7 at 96 kHz) with the same modes, then band-limited back to sr.
    os_ = 4
    h1, h2, hb = _resonator(fk, sigma, A, sr * os_)
    Fh = _hammer(int(0.05 * sr * os_), float(sr * os_), h1, h2, hb, win, K, r["p"], RH, r["mh"], v0)
    from scipy.signal import resample_poly
    Fh = np.maximum(resample_poly(Fh, 1, os_), 0.0)
    n = int((float(dur) + float(tail)) * sr)
    # unison: Weinreich's coupled partials
    ns = strings_per_key(midi)
    cents = np.asarray(UNISON_CENTS[ns])
    lam = np.zeros((k.size, ns), np.complex128)
    cw = np.zeros((k.size, ns), np.complex128)
    one = np.ones(ns)
    for q in range(k.size):
        # the measured decay is the prompt (in-phase) mode: sigma_int + ns zeta = sigma, aftersound sigma_int
        zeta = BRIDGE_SHARE * sigma[q] / ns
        M = np.diag(-(1.0 - BRIDGE_SHARE) * sigma[q] + 1j * wk_[q] * 2.0 ** (cents / 1200.0)).astype(np.complex128) \
            - zeta * np.ones((ns, ns))
        ev, V = np.linalg.eig(M)
        U = np.linalg.inv(V)                                   # rows: left eigenvectors with U V = I
        lam[q] = ev
        cw[q] = (one @ V) * (U @ one) / ns                     # force shared by the strings, bridge sums them
    y = _transverse(Fh, n, float(sr), lam, cw, A * win * wb)
    # longitudinal: ES of the homogenised string, modes below Nyquist, at most 10 (Bank Sec. V-A)
    ES = E_STEEL * S
    fl0 = math.sqrt(ES / mu) / (2.0 * L)
    # only the longitudinal modes inside the band of F_xi, twice the transverse band (Bank Sec. V-A2): here the
    # transverse band ends where the hammer force spectrum is 40 dB down
    Hf = np.abs(np.fft.rfft(Fh, 8192))
    band = np.fft.rfftfreq(8192, 1.0 / sr)[np.nonzero(Hf > 0.01 * Hf.max())[0].max()]
    lk = np.arange(1, 11, dtype=float)
    lk = lk[(lk * fl0 < 0.45 * sr) & (lk * fl0 < 2.0 * band)]
    if lk.size and nmax > 6:
        la1, la2, lb = _resonator(lk * fl0, np.full(lk.size, 1.0 / TAU_LONG), 1.0 / (math.pi * L * mu * lk * fl0), sr)
        ldc = lb / (1.0 + la1 + la2)
        half = fk < 0.25 * sr                                  # products only below fs/4 (Sec. V-B1)
        yl = _longitudinal(Fh, n, float(sr), a1[half], a2[half], bk[half], win[half], k[half], ES, L,
                           la1, la2, lb, ldc, lk)
        y = y + LONG_COUPLING * (yl - yl[0])
    # dampers at note-off. UNSOURCED: 0.12 s time constant on the strings (the soundboard rings on)
    t_off = int(float(dur) * sr)
    if t_off < n:
        y[t_off:] *= np.exp(-np.arange(n - t_off) / (0.12 * sr))
    return soundboard(y, sr, seed)
