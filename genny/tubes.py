"""Acoustic tubes: Webster's equation, a dynamic beating reed, vocal-tract vowels, bottles and ducts.

One finite-difference kernel (`_webster`) solves Webster's horn equation for an arbitrary, optionally
time-varying bore S(x) with a passive radiating end, optional wall / viscothermal loss, side holes with
a continuous open-closed state and a lumped exciter at the left end (volume-velocity source or a
mass-spring-damper reed with collision and Bernoulli flow). `_response` is the exact frequency response
of the same scheme; it is what tunes every pipe (end correction, bore shape and grid included).

Everything runs at the fixed internal rate `FS` and is resampled at the boundary, so tuning and reed
behaviour do not depend on the output sample rate.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numba
import numpy as np
from scipy.signal import resample_poly

from . import filters as F
from .core import DEFAULT_SR, samples
from .fx import effect
from .instruments import instrument
from .sfx import sfx

FS = 44100            # internal rate of every kernel here
C_AIR = 343.0         # m/s at 20 C
RHO = 1.2             # kg/m^3
ETA = 1.846e-5        # Pa s, air shear viscosity at 300 K
FLANGED = 0.8216      # tube ending in an infinite plane (vocal tract)
UNFLANGED = 0.6133    # free pipe end (wind instruments)


def end_correction(a_over_b):
    """delta / a of an unflanged pipe end with a wall: inner radius a, outer radius b."""
    r = float(np.clip(a_over_b, 0.0, 1.0))
    return FLANGED + r * (UNFLANGED - FLANGED) + 0.057 * r * (1.0 - r ** 5)


def hole_rate(b, l_eff, f, c=343.0, eta=ETA, rho=RHO):
    """Decay rate (1/s) of the flow in an open side hole of radius b and effective length l_eff, at f Hz:
    viscous wall loss (2 / b) sqrt(pi f eta / rho) plus radiation c (k b)^2 / (4 l_eff), i.e. the series
    resistance R of the open-hole impedance over its inertance rho l_eff / S_T."""
    kb = 2.0 * math.pi * f * b / c
    return (2.0 / b) * math.sqrt(math.pi * f * eta / rho) + c * kb * kb / (4.0 * max(l_eff, 1e-4))


def hole_length_corrections(b, a, kind="unflanged"):
    """Length corrections (m) of an open side hole of radius b on a bore of radius a, delta = b/a:
    matching volume t_m = b delta (1 + 0.207 delta^3) / 8, inner t_i = b (0.82 - 0.193 delta - 1.09 delta^2
    + 1.27 delta^3 - 0.71 delta^4), radiation t_r = 0.6133 b (unflanged) or 0.8216 b (flanged); returns
    (t_m, t_i, t_r, sum)."""
    if kind not in ("unflanged", "flanged"):
        raise ValueError("hole_kind must be lumped, unflanged or flanged")
    d = float(b) / float(a)
    tm = b * d * (1.0 + 0.207 * d ** 3) / 8.0
    ti = b * (0.82 - 0.193 * d - 1.09 * d ** 2 + 1.27 * d ** 3 - 0.71 * d ** 4)
    tr = (UNFLANGED if kind == "unflanged" else FLANGED) * b
    return tm, ti, tr, tm + ti + tr


JET_GROWTH, JET_CONVECTION = 0.4, 0.5


def jet_receptivity(f, l, U0, R=0.0):
    """Gain and phase of a flue jet from the flue exit to the edge, distance l (m), jet speed U0 (m/s): the
    disturbance grows as exp(0.4 omega l / U0) and travels at 0.5 U0 (phase omega l / (0.5 U0)), plus pi/2 at the
    flue exit. Returns (gain, phase rad, rounded) with rounded = f R / U0 >= 1 (an edge radius R strongly
    accelerates the flow there; a sharp edge gives the pi/2 shift only)."""
    w = 2.0 * math.pi * np.asarray(f, float)
    return (np.exp(JET_GROWTH * w * l / U0), 0.5 * math.pi + w * l / (JET_CONVECTION * U0),
            np.asarray(f, float) * float(R) / U0 >= 1.0)


def causse_zr(ka):
    """Radiation impedance of an unflanged pipe end over Z_c, j = +i."""
    ka = np.asarray(ka, dtype=float)
    lg = np.log(ka)
    return (ka ** 2 / 4.0 + ka ** 4 * (0.0127 + 0.082 * lg - 0.023 * ka ** 2)
            + 1j * (0.6113 * ka - ka ** 3 * (0.036 - 0.034 * lg + 0.0187 * ka ** 2)))

# Wall vibration of the vocal tract. A lossless wall (sigma0 = 0) cannot broaden formants, so the values
# are omega0 = 0 (mass + resistance wall), sigma0 = 8125 1/s, eps = 2718 at S0 = 2.5 cm^2.
# `test_wall_broadens_f1` checks that reading numerically.
WALL_SIGMA, WALL_OMEGA, WALL_EPS, WALL_S0 = 8125.0, 0.0, 2718.0, 2.5e-4

# Clarinet-like reed
REED = dict(Mr=3.37e-6, omega0=23250.0, sigma0=1500.0, H0=4e-4, omega1=316.0, alpha=3.0, Sr=1.46e-4)
REED_S0 = 1.72e-4     # bore area at the reed, m^2 (same figure)
REED_W = 0.01315      # reed channel width w (m): clarinet reed width 13.15 mm

# Vocal tract area functions, non-dimensional [x, S] pairs with S(0) = 1 (S0 = 2.5 cm^2, L = 0.17 m).
# /a/ and /e/: measured Russian vowels.
BILBAO_VOWELS = {   # tables before the modal tract model
    "a": ((0, 1), (0.03, 0.60), (0.09, 0.4), (0.12, 1.6), (0.18, 0.6), (0.29, 0.2), (0.35, 0.4), (0.41, 0.8),
          (0.47, 1), (0.50, 0.6), (0.59, 2), (0.65, 3.2), (0.85, 3.2), (0.94, 2), (1, 2)),
    "e": ((0, 1), (0.09, 0.4), (0.11, 2.4), (0.24, 2.4), (0.26, 3.2), (0.29, 3.2), (0.32, 4.2), (0.41, 4.2),
          (0.47, 3.2), (0.59, 1.8), (0.65, 1.6), (0.71, 1.6), (0.74, 1), (0.76, 0.8), (0.82, 0.8), (0.88, 2),
          (0.91, 2), (0.94, 3.2), (1, 3.2)),
    # /i/, /o/, /u/ are not in the book. Shapes drawn from the textbook articulation (/i/ wide pharynx +
    # palatal constriction, /u/ velar constriction + rounded lips, /o/ in between); the scheme puts their F1/F2
    # at 245/2109, 460/987 and 277/793 Hz, within 10 % of Peterson & Barney's adult-male /i/ and /u/
    # (tests/test_tubes.py::test_vowel_formants).
    "i": ((0, 1), (0.06, 0.5), (0.12, 3.2), (0.40, 4.2), (0.50, 2.4), (0.60, 0.5), (0.80, 0.26), (0.88, 0.5),
          (0.94, 1.2), (1, 1.6)),
    "o": ((0, 1), (0.06, 0.5), (0.12, 1.2), (0.25, 0.5), (0.36, 0.36), (0.48, 1.2), (0.60, 3.2), (0.80, 4.0),
          (0.90, 2.0), (0.96, 0.5), (1, 0.36)),
    "u": ((0, 1), (0.06, 0.5), (0.12, 2.0), (0.30, 3.2), (0.44, 1.2), (0.54, 0.24), (0.62, 0.4), (0.72, 2.8),
          (0.86, 3.6), (0.93, 1.0), (0.97, 0.14), (1, 0.12)),
}
TRACT_L, TRACT_S0, TRACT_C = 0.17, 2.5e-4, 340.0     # reference vocal tract


# ------------------------------------------------------------------------------------------ kernel
@numba.njit(cache=True)
def _halves(S):
    """Areas at the half points l-1/2, l = 0..N+1; the two ghost values are linear extrapolations
    (Sr = 1.5 S_N - 0.5 S_N-1), floored so a steep flare on a coarse grid stays positive."""
    N = S.size - 1
    H = np.empty(N + 2)
    for l in range(1, N + 1):
        H[l] = 0.5 * (S[l] + S[l - 1])
    H[0] = max(1.5 * S[0] - 0.5 * S[1], 0.25 * S[0])
    H[N + 1] = max(1.5 * S[N] - 0.5 * S[N - 1], 0.25 * S[N])
    return H


@numba.njit(cache=True)
def _webster(n, k, gamma, HA, HB, wt, lbc, rbc, radL, radR, sig0, sig1, wall, uin, lin,
             reed, pm, emb, hidx, hpar, hphi, pick, diam=False):
    """Webster scheme / time-varying form with lumped ends. Scaled units: x/L, S/S(0),
    p/(rho c^2), u/(c S0).

    HA, HB   half-point areas of the two bore profiles (N+2); wt = morph weight per sample (size 1 = static)
    lbc      0 closed, driven by the volume velocity uin   1 radiating   2 ideal open (Psi = 0)   3 reed
    rbc      0 closed   1 radiating   2 ideal open
    radL/R   (alpha1, alpha2 * sqrt(S_end)) of the radiation condition
    wall     (eps k / 2, sigma_w, omega_w) of the locally reacting wall; eps = 0 turns it off
    lin      grid index where uin is injected when lbc != 0
    reed     (Q, R, S, omega0, sigma0, omega1, alpha, C) of the reed; pm mouth pressure, emb = H0(t)/H0;
             C = mouthpiece compliance u_c = C dp_in/dt (backward Euler: passive)
    pick     grid index of the interior pressure pickup
    hidx     grid indices of side holes; hpar[j] = (S_T, xi, xi_e[, r]); hphi[j, n] = openness (1 open);
             r > 0 (1/s): the open hole's series resistance as a decay rate of its flow, R / inertance
    Returns  p at the right end, u at the right end, p at the left end, reed displacement y, p at `pick`."""
    N = HA.size - 2
    h = 1.0 / N
    lam2 = (gamma * k / h) ** 2
    lam = math.sqrt(lam2)
    morph = wt.size > 1
    nh = hidx.size
    Hc = HA.copy()                      # flux areas
    Sa = np.empty(N + 1)
    Sb = np.empty(N + 1)
    q4 = np.zeros(N + 1)
    ishole = np.zeros(N + 1, np.bool_)
    for j in range(nh):
        ishole[hidx[j]] = True
    # diam: the two bores are blended in diameter, (sqrt A + w (sqrt B - sqrt A))^2 (the vowel modes
    # deform the diameter), else in area
    DA = np.sqrt(HA)
    DB = np.sqrt(HB)
    w0 = wt[0]
    for l in range(N + 2):
        Hc[l] = (DA[l] + w0 * (DB[l] - DA[l])) ** 2 if diam else HA[l] + w0 * (HB[l] - HA[l])
    for l in range(N + 1):
        Sa[l] = 0.5 * (Hc[l] + Hc[l + 1])
        Sb[l] = Sa[l]
        q4[l] = wall[0] * math.sqrt(math.sqrt(Sa[l]))
    s0k = sig0 * k
    s1 = 2.0 * sig1 * k / (h * h)
    wsk = wall[1] * k
    wok = wall[2] * wall[2] * k * k
    haswall = wall[0] > 0.0
    psi = np.zeros(N + 1)
    psm = np.zeros(N + 1)
    psn = np.zeros(N + 1)
    Fp = np.zeros(N + 1)
    ww = np.zeros(N + 1)
    wm = np.zeros(N + 1)
    y = 0.0
    ym = 0.0
    out_p = np.zeros(n)
    out_u = np.zeros(n)
    out_i = np.zeros(n)
    out_y = np.zeros(n)
    out_m = np.zeros(n)
    out_h = np.zeros(n)
    Uh = np.zeros(nh)
    hasr = hpar.shape[1] > 3
    pin_prev = 0.0
    alpha_h = 0.5                       # alpha <= 1/2
    for i in range(n):
        if morph:
            wp = wt[i - 1] if i > 0 else wt[0]
            wc = wt[i]
            wn = wt[i + 1] if i + 1 < wt.size else wt[i]
            w4 = 0.25 * (wp + 2.0 * wc + wn)
            wa = 0.5 * (wn + wc)
            wb = 0.5 * (wc + wp)
            for l in range(N + 2):
                Hc[l] = (DA[l] + w4 * (DB[l] - DA[l])) ** 2 if diam else HA[l] + w4 * (HB[l] - HA[l])
            for l in range(N + 1):
                if diam:
                    Sa[l] = 0.5 * ((DA[l] + wa * (DB[l] - DA[l])) ** 2 + (DA[l + 1] + wa * (DB[l + 1] - DA[l + 1])) ** 2)
                    Sb[l] = 0.5 * ((DA[l] + wb * (DB[l] - DA[l])) ** 2 + (DA[l + 1] + wb * (DB[l + 1] - DA[l + 1])) ** 2)
                    sc = 0.5 * ((DA[l] + wc * (DB[l] - DA[l])) ** 2 + (DA[l + 1] + wc * (DB[l + 1] - DA[l + 1])) ** 2)
                else:
                    a = 0.5 * (HA[l] + HA[l + 1])
                    d = 0.5 * (HB[l] + HB[l + 1]) - a
                    Sa[l] = a + wa * d
                    Sb[l] = a + wb * d
                    sc = a + wc * d
                if haswall:
                    q4[l] = wall[0] * math.sqrt(math.sqrt(sc))
        # ---- interior
        for l in range(1, N):
            Fl = Hc[l + 1] * (psi[l + 1] - psi[l]) - Hc[l] * (psi[l] - psi[l - 1])
            if ishole[l]:
                psn[l] = psi[l] + (Sb[l] * (psi[l] - psm[l]) + lam2 * Fl) / Sa[l]
                continue
            rhs = Sa[l] * psi[l] + Sb[l] * (psi[l] - psm[l]) + lam2 * Fl + s1 * (Fl - Fp[l])
            Fp[l] = Fl
            den = Sa[l]
            ex = s0k * Sa[l]
            if haswall:
                qq = q4[l]
                qp = qq / (1.0 + wsk)
                W0 = ((2.0 - wok) * ww[l] - (1.0 - wsk) * wm[l]) / (1.0 + wsk)
                rhs -= qq * (W0 - wm[l])
                ex += qq * qp
                psn[l] = (rhs + ex * psm[l]) / (den + ex)
                wm[l] = ww[l]
                ww[l] = W0 + qp * (psn[l] - psm[l])
            else:
                psn[l] = (rhs + ex * psm[l]) / (den + ex)
        if lbc != 0 and lbc != 3 and lin > 0 and lin < N:
            psn[lin] += lam2 * h * uin[i] / Sa[lin]
        for j in range(nh):             # side holes, solved explicitly
            l = hidx[j]
            ph = hphi[j, i]
            A = ph * gamma * gamma * hpar[j, 0] / hpar[j, 2]
            B = (1.0 - ph) * hpar[j, 1] * hpar[j, 0]
            cc = 1.0 / (h * Sa[l])
            G = (psn[l] - 2.0 * psi[l] + psm[l]) / (k * k)
            if hasr and hpar[j, 3] > 0.0:
                Uh[j] = Uh[j] * math.exp(-hpar[j, 3] * k) + A * (psi[l] - psm[l])
                drive = Uh[j]
            else:
                drive = A * psi[l]
            out_h[i] += drive
            dtt = (G - cc * drive) / (1.0 + cc * B + cc * A * (1.0 - alpha_h) * k * k * 0.5)
            psn[l] = 2.0 * psi[l] - psm[l] + k * k * dtt
        # ---- right end
        if rbc == 2:
            psn[N] = 0.0
        else:
            q1 = 0.0
            q2 = 0.0
            if rbc == 1:
                g = gamma * gamma * k * Hc[N + 1] / (Sa[N] * h)
                q1 = radR[1] / math.sqrt(Sa[N]) * g * k
                q2 = radR[0] * g
            psn[N] = (2.0 * (1.0 - lam2) * psi[N] + 2.0 * lam2 * psi[N - 1] - (1.0 + q1 - q2) * psm[N]) / (1.0 + q1 + q2)
            if rbc == 1:
                out_u[i] = Sa[N] * (radR[0] * (psn[N] - psm[N]) / (2.0 * k)
                                    + radR[1] / math.sqrt(Sa[N]) * 0.5 * (psn[N] + psm[N]))
        # ---- left end
        if lbc == 2:
            psn[0] = 0.0
        elif lbc == 1:
            g = gamma * gamma * k * Hc[0] / (Sa[0] * h)
            q1 = radL[1] / math.sqrt(Sa[0]) * g * k
            q2 = radL[0] * g
            psn[0] = (2.0 * (1.0 - lam2) * psi[0] + 2.0 * lam2 * psi[1] - (1.0 + q1 - q2) * psm[0]) / (1.0 + q1 + q2)
        elif lbc == 0:
            psn[0] = 2.0 * psi[0] - psm[0] + 2.0 * lam2 * (psi[1] - psi[0]) + 2.0 * lam2 * h * Hc[0] / Sa[0] * uin[i]
        else:                           # reed
            e_ = emb[i]
            Q = reed[0] / e_
            R = reed[1] * e_
            Sr = reed[2] * e_
            cl = min(y + 1.0, 0.0)
            g = reed[5] ** (reed[6] + 1.0) * abs(cl) ** (reed[6] - 1.0) if cl < 0.0 else 0.0
            a2 = 2.0 / k + 2.0 * reed[4] + k * (reed[3] * reed[3] + g)
            if cl < 0.0 and reed.size > 8:   # Hunt-Crossley: contact force g |cl| (1 + lambda dy/dt)
                a2 += reed[8] * g * abs(cl)
            a1 = -2.0 * (y - ym) / (k * k) + reed[3] * reed[3] * ym + g * (ym + 1.0)
            b3 = a2 / (Sr * Q)
            b1 = R * max(y + 1.0, 0.0) * b3
            b2 = a1 / Q
            lc = lam * Hc[0] / Sa[0]
            e0 = ((psi[0] - psm[0]) + lam2 * (psi[1] - psi[0])) / (k * gamma)   # p_in = e0 + lc * u_in
            c2 = (pm[i] - e0) / lc
            ck = reed[7] / k
            den = 1.0 + b3 / lc + b3 * ck
            d1 = b1 / den
            d2 = (b2 - b3 * c2 - b3 * ck * (pm[i] - pin_prev)) / den
            r = 0.5 * (-d1 + math.sqrt(d1 * d1 + 4.0 * abs(d2)))
            pD = -r * r if d2 > 0.0 else r * r
            v = -(a1 + Q * pD) / a2
            yn = ym + 2.0 * k * v
            ym = y
            y = yn
            pin_prev = pm[i] - pD
            psn[0] = psm[0] + 2.0 * k * gamma * pin_prev
            out_y[i] = y
        out_p[i] = (psn[N] - psm[N]) / (2.0 * k * gamma)
        out_i[i] = (psn[0] - psm[0]) / (2.0 * k * gamma)
        out_m[i] = (psn[pick] - psm[pick]) / (2.0 * k * gamma)
        tmp = psm
        psm = psi
        psi = psn
        psn = tmp
    return out_p, out_u, out_i, out_y, out_m, out_h


@numba.njit(cache=True)
def _response(f, k, gamma, H, rbc, radR, sig0, sig1, wall, hidx, hpar, hphi):
    """Exact steady-state response of `_webster` (static bore, closed left end driven by u_in):
    returns Z_in = p_in/u_in, u_out/u_in and p_out/u_in at the frequencies f (Hz)."""
    N = H.size - 2
    h = 1.0 / N
    lam2 = (gamma * k / h) ** 2
    nf = f.size
    Z = np.zeros(nf, np.complex128)
    Hu = np.zeros(nf, np.complex128)
    Hp = np.zeros(nf, np.complex128)
    hole = np.full(N + 1, -1)
    for j in range(hidx.size):
        hole[hidx[j]] = j
    s1 = 2.0 * sig1 * k / (h * h)
    for m in range(nf):
        z = np.exp(1j * 2.0 * math.pi * f[m] * k)
        zi = 1.0 / z
        D2 = z - 2.0 + zi
        D1 = z - zi
        Dw = z * (1.0 + wall[1] * k) - (2.0 - (wall[2] * k) ** 2) + (1.0 - wall[1] * k) * zi
        Lam = lam2 + s1 * (1.0 - zi)
        SN = 0.5 * (H[N] + H[N + 1])
        q1 = 0.0
        q2 = 0.0
        if rbc == 1:
            g = gamma * gamma * k * H[N + 1] / (SN * h)
            q1 = radR[1] / math.sqrt(SN) * g * k
            q2 = radR[0] * g
        if rbc == 2:
            pr = 0.0 + 0.0j
            pc = 1.0 + 0.0j             # Psi_{N-1}
            pN = 0.0 + 0.0j
            uo = 0.0 + 0.0j
            start = N - 1
        else:
            pN = 1.0 + 0.0j
            pr = pN
            pc = (z * (1.0 + q1 + q2) - 2.0 * (1.0 - lam2) + zi * (1.0 + q1 - q2)) * pN / (2.0 * lam2)
            uo = SN * (radR[0] * D1 / (2.0 * k) + radR[1] / math.sqrt(SN) * 0.5 * (z + zi)) * pN if rbc == 1 else 0.0j
            start = N - 1
        # pr = Psi_{l+1}, pc = Psi_l ; march l = N-1 .. 1
        for l in range(start, 0, -1):
            B = 0.5 * (H[l] + H[l + 1])
            j = hole[l]
            if j >= 0:
                ph = hphi[j]
                A = ph * gamma * gamma * hpar[j, 0] / hpar[j, 2]
                Bh = (1.0 - ph) * hpar[j, 1] * hpar[j, 0]
                cc = 1.0 / (h * B)
                Ae = A
                if hpar.shape[1] > 3 and hpar[j, 3] > 0.0:
                    Ae = A * (1.0 - zi) / (1.0 - math.exp(-hpar[j, 3] * k) * zi)
                T = B * (D2 * (1.0 + cc * Bh + cc * A * 0.25 * k * k) + k * k * cc * Ae)
                L_ = lam2 + 0.0j
            else:
                qq = wall[0] * math.sqrt(math.sqrt(B))
                T = B * D2 + sig0 * k * B * D1 + qq * qq * D1 * D1 / Dw
                L_ = Lam
            pl = pc - (H[l + 1] * (pr - pc) - T * pc / L_) / H[l]
            pr = pc
            pc = pl
        S0 = 0.5 * (H[0] + H[1])
        u = (D2 * pc - 2.0 * lam2 * (pr - pc)) / (2.0 * lam2 * h * H[0] / S0)
        Z[m] = D1 * pc / (2.0 * k * gamma) / u
        Hu[m] = uo / u
        Hp[m] = D1 * pN / (2.0 * k * gamma) / u
    return Z, Hu, Hp


# ------------------------------------------------------------------------------------------ class
def _profile(area, N):
    """Bore on the grid x = l/N from a callable S(x), [x, S] pairs or samples over [0, 1]; S(0) -> 1."""
    x = np.linspace(0.0, 1.0, N + 1)
    if callable(area):
        S = np.asarray(area(x), dtype=np.float64) * np.ones(N + 1)
    else:
        a = np.asarray(area, dtype=np.float64)
        if a.ndim == 2:
            S = np.interp(x, a[:, 0], a[:, 1])
        elif a.ndim == 0:
            S = np.full(N + 1, float(a))
        else:
            S = np.interp(x, np.linspace(0.0, 1.0, a.size), a)
    return np.maximum(S / S[0], 1e-3)


class WebsterTube:
    """Acoustic tube of arbitrary bore S(x): explicit finite-difference scheme with the passive radiation
    end, yielding-wall loss and an approximate viscothermal loss.

    Stability is `lambda = gamma k / h <= 1` whatever the bore: the grid is `N = floor(1/(gamma k))`,
    `h = 1/N`, so the tube has exactly the requested length for any N.

    area      callable S(x) on x in [0, 1], [x, S] pairs, samples, or a number (cylinder); rescaled to S(0) = 1
    area2     optional second profile for `run(morph=...)` (time-varying scheme)
    radius_m  bore radius at the left end (sets S0, hence the radiation and loss constants)
    left      "closed" (driven by u_in) | "open" (ideal, Psi = 0) | "flanged" | "unflanged" | "reed"
    right     "closed" | "open" | "flanged" | "unflanged"
    loss      viscothermal boundary-layer loss, 1 = smooth rigid pipe. The exact loss needs a
              fractional derivative; here it is the two-term loss -2 sigma0 Psi_t + 2 sigma1 Psi_txx
              fitted to the boundary-layer attenuation alpha = 1.045 sqrt(omega eta/rho)/(c a) at the
              quarter-wave frequency and at 3 kHz: exact at those two points, up to ~2x too little between.
    wall      yielding (vocal-tract) wall, 1 = the WALL_* values
    holes     [(x in (0,1), hole radius m, chimney height m), ...] side holes
    end_wall  outer / inner radius of an unflanged end (>= 1): its mass end correction, None = thin wall
    hole_kind "lumped" (one external correction b (1.4 - 0.58 (b/a)^2)) | "unflanged" | "flanged": the open hole's
              length as chimney + matching volume + inner correction + radiation (0.6133 b or 0.8216 b)
    """

    def __init__(self, length_m, area=1.0, area2=None, radius_m=0.0074, left="closed", right="unflanged",
                 loss=0.0, wall=0.0, holes=(), c=C_AIR, sr=FS, diam=False, end_wall=None, hole_kind="lumped",
                 hole_loss=False):
        self.diam = bool(diam)
        self.delta = UNFLANGED if end_wall is None else end_correction(1.0 / max(float(end_wall), 1.0))
        self.L = float(max(length_m, 4.0 * c / sr))          # at least 4 grid cells
        self.c, self.sr, self.k = float(c), int(sr), 1.0 / sr
        self.gamma = self.c / self.L
        self.S0 = math.pi * float(radius_m) ** 2
        mean_a = lambda S: math.sqrt(self.S0 * float(np.mean(S)) / math.pi)   # noqa: E731
        N0 = max(4, int(1.0 / (self.gamma * self.k)))
        self.sig0 = self.sig1 = 0.0
        if loss > 0.0:
            a = mean_a(_profile(area, N0))
            dec = lambda w: float(loss) * 1.045 * math.sqrt(w * ETA / RHO) / a   # noqa: E731  amplitude decay, 1/s
            w1 = 2 * math.pi * self.c / (4 * self.L)
            w2 = max(2 * math.pi * 3000.0, 4 * w1)
            self.sig1 = self.gamma ** 2 * (dec(w2) - dec(w1)) / (w2 * w2 - w1 * w1)
            self.sig0 = dec(w1) - self.sig1 * w1 * w1 / self.gamma ** 2
        # stability with the explicit sigma1 term: h^2 >= gamma^2 k^2 + 4 sigma1 k
        self.N = max(4, int(1.0 / math.sqrt((self.gamma * self.k) ** 2 + 4.0 * self.sig1 * self.k)))
        self.h = 1.0 / self.N
        self.lam = self.gamma * self.k / self.h
        self.S = _profile(area, self.N)
        self.S2 = self.S if area2 is None else _profile(area2, self.N)
        self.HA, self.HB = _halves(self.S), _halves(self.S2)
        self.lbc = {"closed": 0, "flanged": 1, "unflanged": 1, "open": 2, "reed": 3}[left]
        self.rbc = {"closed": 0, "flanged": 1, "unflanged": 1, "open": 2}[right]
        self.radL, self.radR = self._rad(left), self._rad(right)
        eps = float(wall) * WALL_EPS * (WALL_S0 / self.S0) ** 0.25 * (self.c / TRACT_C)
        self.wall = np.array([eps * self.k / 2.0, WALL_SIGMA, WALL_OMEGA])
        self.hidx = np.array([int(np.clip(round(x * self.N), 1, self.N - 1)) for x, _, _ in holes], dtype=np.int64)
        hp = []
        for (x, b, tau), l in zip(holes, self.hidx):
            a = math.sqrt(self.S0 * self.S[l] / math.pi)
            b = min(b, a)
            # S_T, xi, xi_e
            xe = (tau / self.L + (b / self.L) * (1.4 - 0.58 * b * b / (a * a)) if hole_kind == "lumped"
                  else (tau + hole_length_corrections(b, a, hole_kind)[3]) / self.L)
            row = (math.pi * b * b / self.S0, tau / self.L, xe)
            if hole_loss:
                row = row + (hole_rate(b, xe * self.L, self.c / (4.0 * self.L), self.c),)
            hp.append(row)
        self.hpar = np.array(hp, dtype=np.float64).reshape(-1, 4 if hole_loss and hp else 3)

    def _rad(self, kind):
        """(alpha1, alpha2 sqrt(S_end)) of the flanged / unflanged end; the 1/gamma in alpha1 is confirmed by the reference code
        (`bet = 0.7407/gamma` = 1/(2 * 0.8216^2 gamma))."""
        if kind == "flanged":
            return np.array([1.0 / (2.0 * FLANGED ** 2 * self.gamma), self.L / (FLANGED * math.sqrt(self.S0 / math.pi))])
        if kind == "unflanged":
            return np.array([1.0 / (4.0 * UNFLANGED ** 2 * self.gamma), self.L / (self.delta * math.sqrt(self.S0 / math.pi))])
        return np.zeros(2)

    def run(self, u_in=None, n=None, morph=None, src_pos=0.0, reed=None, pm=None, emb=None, hole_state=None,
            pick_pos=0.5):
        """Simulate n samples at `self.sr`. Returns dict p_out (pressure at the right end / rho c^2),
        u_out (volume velocity out of the right end / c S0), p_in, y (reed displacement, -1 = closed),
        p_pick (pressure at x = pick_pos).

        u_in volume velocity (left end if it is closed, else injected at `src_pos`); morph = weight 0..1 per
        sample between `area` and `area2`; reed = (Q, R, S, omega0, sigma0, omega1, alpha) from `reed_params`
        with pm (mouth pressure) and emb (reed opening H0(t)/H0); hole_state = openness 0..1 per hole
        (scalar or per-sample array each)."""
        n = int(n if n is not None else len(u_in if u_in is not None else pm))
        z = np.zeros(1)
        u = z if u_in is None else np.ascontiguousarray(u_in, dtype=np.float64)
        if u.size < n:
            u = np.concatenate([u, np.zeros(n - u.size)])
        wt = z if morph is None else np.clip(np.ascontiguousarray(morph, dtype=np.float64), 0.0, 1.0)
        nh = self.hidx.size
        hphi = np.zeros((nh, n))
        if hole_state is not None:
            for j in range(nh):
                hphi[j] = np.clip(hole_state[j], 0.0, 1.0)
        rp = np.zeros(9) if reed is None else np.asarray(reed, dtype=np.float64)
        pmv = z if pm is None else np.ascontiguousarray(pm, dtype=np.float64)
        ev = np.ones(n) if emb is None else np.clip(np.ascontiguousarray(emb, dtype=np.float64) * np.ones(n), 0.05, 4.0)
        lin = int(np.clip(round(src_pos * self.N), 1, self.N - 1))
        pick = int(np.clip(round(pick_pos * self.N), 0, self.N))
        p, uo, pi, y, pk, uh = _webster(n, self.k, self.gamma, self.HA, self.HB, wt, self.lbc, self.rbc, self.radL,
                                        self.radR, self.sig0, self.sig1, self.wall, u, lin, rp, pmv, ev, self.hidx,
                                        self.hpar, hphi, pick, self.diam)
        return {"p_out": p, "u_out": uo, "p_in": pi, "y": y, "p_pick": pk, "u_holes": uh}

    def response(self, freqs, hole_state=None):
        """Exact frequency response of the scheme for a closed, flow-driven left end:
        (Z_in, u_out/u_in, p_out/u_in) at `freqs` Hz."""
        ph = np.zeros(self.hidx.size) if hole_state is None else np.asarray(hole_state, dtype=np.float64)
        return _response(np.atleast_1d(np.asarray(freqs, dtype=np.float64)), self.k, self.gamma, self.HA, self.rbc,
                         self.radR, self.sig0, self.sig1, self.wall, self.hidx, self.hpar, ph)

    def resonances(self, fmax=None, kind="z", count=None, hole_state=None, comp=0.0):
        """Frequencies (Hz) of the peaks of |Z_in| (kind "z": what a reed, lip or slap at a closed end
        excites) or of |u_out/u_in| (kind "u": formants), refined by parabolic interpolation.
        `comp` (seconds) loads the input with a compliance u = comp * dp/dt, i.e. a closed volume
        S0 * c * comp: the reed's own volume flow, c * S * Q / omega0^2 from `reed_params`."""
        fmax = float(fmax or 0.45 * self.sr)
        df = min(2.0, self.gamma / 400.0)
        f = np.arange(df, fmax, df)
        r = self.response(f, hole_state)
        z = r[0] / (1.0 + 2j * math.pi * f * comp * r[0])
        m = np.log(np.abs(z if kind == "z" else r[1]) + 1e-300)
        i = np.nonzero((m[1:-1] > m[:-2]) & (m[1:-1] >= m[2:]))[0] + 1
        if count:
            i = i[:count]
        d = 0.5 * (m[i - 1] - m[i + 1]) / (m[i - 1] - 2 * m[i] + m[i + 1])
        return f[i] + d * df


def bessel_bore(x0=1.1, nu=0.63):
    """Bessel horn S(x) proportional to (x0 - x)^(-2 nu), x0 > 1 the throat of the flare past the bell, nu 0.63:
    For `WebsterTube(area=)`. x0 default."""
    return lambda x: (float(x0) - x) ** (-2.0 * float(nu))


def reed_params(S0=REED_S0, c=C_AIR, w=REED_W, volume=0.0, contact_damping=0.0, **over):
    """Non-dimensional reed constants (Q, R, S, omega0, sigma0, omega1, alpha, C, lambda) of the reed model from
    the physical reed (`REED`, overridable by keyword; keys not in `REED` are ignored, so a preset such as
    `winds.REED_PLAYED` or `winds.SAX_REED` goes in whole) and the bore area S0 at the reed. `volume` (m^3) is an
    extra mouthpiece cavity, C = volume/(c S0) (a lumped compliance in parallel with the
    reed flow; it stands in for the missing apex of a truncated cone). `contact_damping` lambda (s):
    Hunt-Crossley damping of the reed on the lay, contact force g |y + 1|^alpha (1 + lambda dy/dt)."""
    r = {**REED, **{k: v for k, v in over.items() if k in REED}}
    Q = RHO * c * c * r["Sr"] / (r["Mr"] * r["H0"])
    R = math.sqrt(2.0) * w * r["H0"] / S0
    S = r["Sr"] * r["H0"] / (c * S0)
    return np.array([Q, R, S, r["omega0"], r["sigma0"], r["omega1"], r["alpha"], volume / (c * S0),
                     float(contact_damping)])


def _reed_preset(name):
    from .physical import winds
    if name == "default":
        return {}
    if name == "played":
        return dict(winds.REED_PLAYED)
    if name == "sax":
        return dict(winds.SAX_REED)
    raise ValueError(f"reed must be 'default', 'played' or 'sax', got {name!r}")


def tune_length(make, freq, L0, mode=0, comp=0.0):
    """Tube whose (mode+1)-th input-impedance peak sits at `freq`: `make(L)` builds the tube, the length is
    iterated on the scheme's own response, so end correction, flare, holes, loss and grid are all included."""
    L = float(L0)
    t = make(L)
    for _ in range(14):
        r = t.resonances(fmax=(mode + 1.6) * 2.4 * freq, count=mode + 1, comp=comp)
        if r.size <= mode:
            break
        if abs(r[mode] / freq - 1.0) < 1e-5:
            break
        L *= r[mode] / freq
        t = make(L)
    return t


# ------------------------------------------------------------------------------------------ helpers
def _rs(x, sr_from, sr_to):
    if sr_from == sr_to:
        return x
    g = math.gcd(int(sr_from), int(sr_to))
    return resample_poly(x, sr_to // g, sr_from // g, axis=0)


def _edge(n, a, r):
    """Raised-cosine attack over a samples and release over the last r samples."""
    e = np.ones(n)
    a, r = int(min(a, n)), int(min(r, n))
    if a > 1:
        e[:a] = 0.5 - 0.5 * np.cos(np.pi * np.arange(a) / a)
    if r > 1:
        e[n - r:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(r) / r)
    return e


def _finish(y, sr, tone=None, peak=None):
    """Internal-rate signal -> output: DC removal, optional tone lowpass, optional peak level, resampling,
    zero ends."""
    y = F.highpass(y, 20.0, FS)
    if tone is not None:
        y = F.lowpass(y, min(float(tone), 0.45 * FS), FS, order=2)
    if peak is not None:
        y = y * (float(peak) / (float(np.max(np.abs(y))) + 1e-12))
    a = np.abs(y)                                   # safety knee for extreme parameter settings: never above 1.45
    y = np.where(a > 1.2, np.sign(y) * (1.2 + 0.25 * np.tanh((a - 1.2) / 0.25)), y)
    y = _rs(y, FS, sr)
    return y * _edge(y.size, samples(0.002, sr), samples(0.006, sr))


def _noise(n, seed, fc=3500.0):
    return F.lowpass(np.random.default_rng(int(seed)).uniform(-1.0, 1.0, n), fc, FS)


def _midi(freq):
    return 69.0 + 12.0 * math.log2(max(float(freq), 1.0) / 440.0)


# ------------------------------------------------------------------------------------------ reed instruments
# Mouth pressure / (rho c^2) = a + b * vel. mapping, placed inside the reed regime landmarks:
# threshold 0.0121, beating from ~0.0168, static reed closure at omega0^2/Q = 0.0354. Always beating, so the
# note speaks within ~50 ms.
REED_PM = (0.018, 0.010)
# Reed loading. The reed's volume flow u_r = S dy/dt makes it a compliance c*S*Q/omega0^2 = 9.6 mm of extra
# tube; the flow nonlinearity adds a little that depends on mouth pressure. Measured on the cylinder
# (`python -m genny.tubes`).
_DL = 0.0096
# Residual per-instrument calibration, measured at vel 0.9 (`python -m genny.tubes` regenerates it): MIDI
# notes, cents the impedance peak must be offset by, output gain, and how the pitch moves with mouth pressure
# (extra reed length in m per unit pm below 0.027, cents per 0.006 of pm below 0.027).
_CAL = {
    "reed_tube": ((44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90),
        (-0.1, 0.3, 0.9, 0.6, 0.4, 1.6, 1.6, 2.0, 2.1, 2.5, 3.1, 3.4, 4.4, 4.0, 5.1, 4.9, 2.7, 3.5, 10.7, 9.4, -2.8, -9.4, -11.1, -7.1),
        (628, 586, 541, 505, 470, 435, 399, 356, 316, 283, 255, 230, 208, 193, 176, 175, 166, 149, 142, 159, 156, 140, 121, 103),
        (0.277, -0.79)),
    "reed_cone": ((44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90),
        (0.0, 9.6, -0.1, 0.3, -0.6, -0.4, 0.3, -1.2, -0.1, -0.5, -1.0, -1.5, -2.6, -2.7, -3.6, -3.9, -4.2, -4.2, -3.6, -2.9, -1.6, 0.6, 6.2, 19.1),
        (658, 1141, 1158, 1015, 892, 791, 700, 618, 546, 484, 429, 381, 339, 302, 271, 251, 232, 213, 198, 183, 167, 151, 136, 129),
        (0.201, -6.09)),
    "bottle": ((30, 36, 42, 48, 60, 72, 90), (0, 0, 0, 0, 0, 0, 0), (8.0, 4.2, 2.9, 2.1, 1.6, 1.3, 1.3)),
}
TONEHOLE = (0.8, 0.003, 0.005)        # x_T, radius b, height tau (S_T 0.1644, xi 0.0075, xi_e 0.0134)
REGISTER = (0.078, 0.0015, 0.010)     # distance from the reed (m) and radius; height 10 mm: measured register holes are 13 mm, but this lumped hole stops overblowing at 13 mm


@lru_cache(maxsize=512)
def _reed_bore(freq, ratio, comp_s):
    """Bore (cylinder, or cone S = (1 + eps x)^2 of end-radius ratio `ratio`) with a closed register hole and
    tonehole, tuned so that its first resonance, loaded by the reed compliance `comp_s`, is `freq`."""
    eps = ratio - 1.0
    area = (lambda x: (1.0 + eps * x) ** 2) if eps > 0 else 1.0

    def make(L):
        reg = (float(np.clip(REGISTER[0] / L, 0.05, 0.4)), REGISTER[1], REGISTER[2])
        return WebsterTube(L, area, radius_m=math.sqrt(REED_S0 / math.pi), left="reed", right="unflanged", loss=1.0,
                           holes=(reg, TONEHOLE))

    return tune_length(make, freq, C_AIR / (4.0 * freq) * (1.0 + 0.3 * eps), comp=comp_s)


def _on_twelfth(y, freq):
    """H3 more than 6 dB over H1 on a held stretch (too short a stretch says nothing)."""
    if y.size < 2048:
        return False
    S = np.abs(np.fft.rfft(y * np.hanning(y.size), 1 << 17))
    f = np.fft.rfftfreq(1 << 17, 1.0 / FS)
    h1, h3 = (S[(f > k * freq * 0.97) & (f < k * freq * 1.03)].max() for k in (1, 3))
    return h3 > 2.0 * h1


def _reed_voice(freq, dur, sr, vel, cone, pressure, embouchure, breath, vibrato, attack, tongue, register,
                hole, bright, seed, reed="default", contact=0.0, mouthpiece=0.0):
    freq = float(np.clip(freq, 30.0, 2500.0))
    vel = float(np.clip(vel, 0.0, 1.5))
    pm0 = float(np.clip((REED_PM[0] + REED_PM[1] * vel) * pressure, 0.0, 0.034))
    emb = float(np.clip(embouchure, 0.7, 1.6))       # below ~0.7 the mouth pressure holds the reed shut
    cone = float(np.clip(cone, 0.0, 1.0))
    m = _midi(freq)

    def cal(i):                 # calibration blended between the cylinder's and the cone's
        a, b = _CAL["reed_tube"], _CAL["reed_cone"]
        va = np.interp(m, a[0], a[i]) if i < 3 else np.array(a[3])
        vb = np.interp(m, b[0], b[i]) if i < 3 else np.array(b[3])
        return (1.0 - cone) * va + cone * vb

    slope = cal(3)
    cents = float(cal(1)) + slope[1] * (0.027 - pm0) / 0.006
    dl = _DL + slope[0] * (0.027 - pm0)
    vol = 0.0
    if mouthpiece > 0.0:                             # equivalent mouthpiece volume, a compliance: V / S0 of tube
        from .physical import winds
        vol = float(mouthpiece) * float(winds.sax_equivalent_volume(freq))
    rp = reed_params(volume=vol, contact_damping=contact, **_reed_preset(reed))
    tube = _reed_bore(round(freq * 2.0 ** (cents / 1200.0), 4), 1.0 + cone,
                      round((dl + vol / REED_S0) / C_AIR, 9))
    n = int((dur + 0.14) * FS)
    hold = int((dur + 0.05) * FS)
    env = np.zeros(n)
    env[:hold] = _edge(hold, max(0.002, attack) * FS, 0.05 * FS)
    t = np.arange(n) / FS
    # tongued start: a short pressure overshoot (25 ms; without it a low note needs > 100 ms to
    # speak); the turbulence comes in once the regime has locked (noise during the first periods can tip the
    # reed into the next register - a squeak)
    mod = 1.0 + float(np.clip(tongue, 0.0, 1.0)) * np.exp(-t / 0.025) + 0.08 * float(vibrato) * np.sin(2 * np.pi * 5.0 * t)
    mod = mod + float(breath) * _noise(n, seed) * np.clip((t - 0.08) / 0.05, 0.0, 1.0)
    ph = [float(np.clip(register, 0.0, 1.0)), float(np.clip(hole, 0.0, 1.0))]
    out = tube.run(pm=pm0 * env * mod, reed=rp, emb=emb, hole_state=ph, n=n)["p_out"]
    # A slow rise crosses the third bore mode's threshold first and the reed stays there: measured at attack 0.2 s,
    # the fundamental 14-22 dB under H3, 52-69 dB at 418-600 Hz. Same answer as impedance.clarinet (a player tongues
    # it): speak on a short rise, the slow attack then on the level. H3 > H1 + 6 dB as the test.
    if attack > 0.02 and not any(ph) and _on_twelfth(out[int(attack * FS):hold - int(0.05 * FS)], freq):
        fast = np.zeros(n)
        fast[:hold] = _edge(hold, 0.005 * FS, 0.05 * FS)
        out = tube.run(pm=pm0 * fast * mod, reed=rp, emb=emb, hole_state=ph, n=n)["p_out"] * _edge(n, attack * FS, 0)
    out = out * float(cal(2)) * (0.45 + 0.55 * min(vel, 1.0))
    return _finish(out, sr, tone=float(np.clip(freq * 9.0, 1800.0, 4200.0)) * float(np.clip(bright, 0.25, 4.0)))


_REED_HELP = dict(
    pressure=(1.0, "mouth pressure scale 0.7..1.25 (below ~0.65 the reed does not speak; high = beating reed, brighter)"),
    embouchure=(1.0, "reed opening H0 scale 0.7..1.5 (tight lips < 1: softer, a little sharp; loose > 1: louder, a little flat)"),
    breath=(0.04, "0..0.5 turbulence noise on the mouth pressure"),
    vibrato=(0.0, "0..1 breath-pressure vibrato depth (5 Hz)"),
    attack=(0.005, "s, mouth-pressure rise time"),
    tongue=(0.5, "0..1 tongued start: pressure overshoot that makes the note speak fast (0 = slow breath attack)"),
    register=(0.0, "0..1 register hole openness (1 = overblown: the cylinder jumps a twelfth)"),
    hole=(0.0, "0..1 openness of a tonehole at 80 % of the bore: bends the note up (about a semitone at 0.5); fully open the cylinder may flip register. 0 keeps the note in tune"),
    bright=(1.0, "0.25..4 output lowpass scale"), seed=(0, "noise seed"),
    reed=("default", "reed preset: default (textbook), played (clarinet reed measured in play), sax (tenor sax reed + lip)"),
    contact=(0.0, "s, Hunt-Crossley damping of the reed on the lay (0 = elastic contact)"),
    mouthpiece=(0.0, "0..1 share of the saxophone mouthpiece equivalent volume added at the reed (pitch kept)"))


@instrument("reed_tube", "Clarinet-like cylinder blown by a dynamic beating reed (Webster tube + beating reed): hollow, odd harmonics.",
            family="wind", span=("D3", "G5"), **_REED_HELP)
def reed_tube(freq, dur, sr=DEFAULT_SR, vel=1.0, pressure=1.0, embouchure=1.0, breath=0.04, vibrato=0.0,
              attack=0.005, tongue=0.5, register=0.0, hole=0.0, bright=1.0, seed=0, reed="default", contact=0.0,
              mouthpiece=0.0):
    """Beating reed on a cylindrical Webster bore with the unflanged radiation end."""
    return _reed_voice(freq, dur, sr, vel, 0.0, pressure, embouchure, breath, vibrato, attack, tongue,
                       register, hole, bright, seed, reed, contact, mouthpiece)


@instrument("reed_cone", "Sax/oboe-like conical bore blown by the same dynamic reed: full harmonic series, reedy.",
            family="wind", span=("D3", "C6"),
            cone=(1.0, "0..1 taper: 0 = cylinder (odd harmonics), 1 = end radius twice the reed end (even harmonics appear)"),
            **_REED_HELP)
def reed_cone(freq, dur, sr=DEFAULT_SR, vel=1.0, cone=1.0, pressure=1.0, embouchure=1.0, breath=0.04, vibrato=0.0,
              attack=0.005, tongue=0.5, register=0.0, hole=0.0, bright=1.0, seed=0, reed="default", contact=0.0,
              mouthpiece=0.0):
    """Same reed on S(x) = (1 + cone x)^2. The taper stops at an
    end-radius ratio of 2: steeper truncated cones jump to the octave or stutter, as reported for this model
    (ratio 3 held its fundamental only with a mouthpiece cavity equal to the missing apex, `reed_params(volume=)`,
    and still broke in the low register with a tongued start)."""
    return _reed_voice(freq, dur, sr, vel, cone, pressure, embouchure, breath, vibrato, attack, tongue,
                       register, hole, bright, seed, reed, contact, mouthpiece)


# ------------------------------------------------------------------------------------------ vocal tract
def _tract(vowel, vowel2=None, size=1.0, wall=1.0, consonant=""):
    """Modal tract (physical/story.py): measured vowel areas and lengths. With ``vowel2`` the
    two shapes are blended in diameter, i.e. along q1, q2; with ``consonant`` the tract starts at the
    consonant's constriction laid over the vowel and opens to the vowel, linear in the activation m_c,
    which is linear in area."""
    from .physical import story
    A1 = story.vowel_area(vowel)
    if consonant:
        A0, A2, diam = story.consonant_area(consonant, A1), A1, False
    else:
        A0, A2, diam = A1, None if not vowel2 else story.vowel_area(vowel2), True
    L = story.vowel_length(vowel) if not vowel2 else 0.5 * (story.vowel_length(vowel) + story.vowel_length(vowel2))
    L *= float(np.clip(size, 0.4, 2.5))
    return WebsterTube(L, story.breakpoints(A0), None if A2 is None else story.breakpoints(A2),
                       radius_m=math.sqrt(max(float(A0[0]), 0.02) * 1e-4 / math.pi), left="closed", right="flanged",
                       wall=wall, c=TRACT_C, diam=diam)


ROSENBERG = (0.56, 0.4 / 0.56)     # open quotient, rising share of the open phase: Tp 0.40 T, Tn 0.16 T
CONSONANT_EVENT = (0.0, 0.05, 0.11)  # s: closure from the start, held to the peak, released by the end


def event(t, n01, npk, n02, mu=1.0):
    """Constriction event E: 0 before n01, raised cosine up to 1 at npk, raised cosine down to 0 at n02, times mu.
    Times in any unit; ``t`` an array. n01 == npk starts closed."""
    t = np.asarray(t, float)
    up = np.where(t < npk, 0.5 * (1.0 - np.cos(np.pi * np.clip((t - n01) / max(npk - n01, 1e-12), 0.0, 1.0))), 1.0)
    down = np.where(t > npk, 0.5 * (1.0 + np.cos(np.pi * np.clip((t - npk) / max(n02 - npk, 1e-12), 0.0, 1.0))), 1.0)
    return float(mu) * np.where(t < n01, 0.0, up * down)


def events(t, evs):
    """Sum of several events, each (n01, npk, n02, mu), held to 0..1."""
    return np.clip(sum(event(t, *e) for e in evs), 0.0, 1.0)


def glottal_pulses(f0, n, vibrato=0.0, vib_hz=5.0, duty=0.2, sr=FS, source="bilbao"):
    """Glottal volume-velocity source: y = sin(2 pi * integral f0 (1 + e_f sin 2 pi f1 t)),
    u = ([y - e]_+ / (1 - e))^2. `duty` = e in [0, 1): 0 is a half-wave (50 % open), larger = shorter, brighter
    pulses; the square makes the flow differentiable. `vibrato` 1 = e_f 0.03, entering after 0.3 s as a
    singer's does. ``source`` rosenberg: Rosenberg's C pulse instead (cosine rise over 0.40 T, quarter-cosine fall over
    0.16 T, closed the rest), ``duty`` unused."""
    t = np.arange(n) / sr
    f = float(f0) * (1.0 + 0.03 * float(vibrato) * np.sin(2 * np.pi * vib_hz * t) * np.clip((t - 0.3) / 0.3, 0.0, 1.0))
    if source == "rosenberg":
        from .speech import glottal_pulse
        return glottal_pulse(np.cumsum(f) / sr % 1.0, *ROSENBERG)[1]
    if source != "bilbao":
        raise ValueError(f"glottal source {source!r}: use bilbao or rosenberg")
    y = np.sin(2 * np.pi * np.cumsum(f) / sr)
    e = float(np.clip(duty, 0.0, 0.9))
    return (np.maximum(y - e, 0.0) / (1.0 - e)) ** 2


@instrument("vowel_tube", "Sung vowel: glottal pulse train through a Webster vocal tract (a e i o u), gliding between two vowels.",
            family="voice", span=("E2", "C5"),
            vowel=("a", "i ih eh ae ah aa ao uh ow uw (MRI vowels; a e o u = aa eh ow uw), tract shape at the start"),
            vowel2=("", "vowel to glide to, along the tract modes; '' = hold the first"),
            morph=(1.0, "0..1 how far the tract travels toward vowel2 during the note"),
            consonant=("", "p t k m n ng f th s sh (b d g v z...: their place): start from that constriction over the vowel and release into it"),
            breath=(0.08, "0..1 aspiration noise at the glottis"),
            vibrato=(0.4, "0..2 pitch vibrato depth (1 = +-3 %, 5 Hz, comes in after 0.3 s)"),
            size=(1.0, "tract length scale: 1 = 17 cm adult, 0.8 small/bright, 1.4 giant (formants scale as 1/size)"),
            tense=(0.2, "0..0.8 glottal closed fraction: 0 soft/breathy pulse, 0.6 pressed and buzzy"),
            source=("bilbao", "glottal flow: bilbao (squared sine, shaped by tense) | rosenberg (Rosenberg C pulse)"),
            seed=(0, "noise seed"),
            glide=("smooth", "vowel2 path: smooth (cubic) | minjerk (minimum-jerk, the speech-motor profile)"),
            model=("webster", "webster | kl: the Kelly-Lochbaum tract with subglottal system and nasal branch"),
            nasal=(0.0, "velum opening cm^2 (0..2; > 0 uses model kl)"))
def vowel_tube(freq, dur, sr=DEFAULT_SR, vel=1.0, vowel="a", vowel2="", morph=1.0, breath=0.08, vibrato=0.4,
               size=1.0, tense=0.2, seed=0, consonant="", source="bilbao", glide="smooth", model="webster", nasal=0.0):
    """Finite-difference Webster tract with a flanged radiation end and yielding walls, area functions of ``_tract``.
    Output is the radiated pressure. A consonant onset is one constriction event (`event`, `CONSONANT_EVENT`): held
    50 ms, released over 60 ms by a raised cosine; voiceless ones (p t k f th s sh) start voicing 40 ms after the
    release with aspiration in the gap."""
    if model == "kl" or float(nasal) > 0.0:
        return _kl_vowel(freq, dur, sr, vel, vowel, vowel2, morph, vibrato, float(nasal), seed)
    if model != "webster":
        raise ValueError("vowel_tube model: webster or kl")
    tr = _tract(vowel, vowel2, size, consonant=consonant)
    n = int((dur + 0.10) * FS)
    hold = int((dur + 0.04) * FS)
    env = np.zeros(n)
    env[:hold] = _edge(hold, 0.03 * FS, 0.06 * FS)
    u = glottal_pulses(freq, n, vibrato, duty=tense, source=source)
    asp = 0.25 * float(breath) * _noise(n, seed, 4000.0)
    w = None
    if consonant:
        t = np.arange(n) / FS
        w = 1.0 - event(t, *CONSONANT_EVENT)
        if str(consonant).lower() in ("p", "t", "k", "f", "th", "s", "sh", "θ", "ʃ"):
            voice = np.clip((t - 0.15) / 0.02, 0.0, 1.0)
            u = u * voice
            asp = asp + 0.6 * _noise(n, seed + 1, 4000.0) * ((t > 0.11) & (t < 0.16))
    u = env * (u + asp * (0.4 + u))
    if vowel2 and not consonant:  # smooth trajectory: "do not interpolate area functions linearly for speech"
        x = np.clip(np.arange(n) / max(hold, 1), 0.0, 1.0)
        w = float(np.clip(morph, 0.0, 1.0)) * (_min_jerk(x) if glide == "minjerk" else x * x * (3.0 - 2.0 * x))
    p = tr.run(u, morph=w)["p_out"]
    body = p[:hold]
    p = p * (0.23 * (0.35 + 0.65 * min(float(vel), 1.2)) / (math.sqrt(float(np.mean(body * body))) + 1e-12))
    return _finish(p, sr, tone=4500.0)


def _min_jerk(x):
    from .physical.story import min_jerk
    return min_jerk(x, 1.0)


def _kl_vowel(freq, dur, sr, vel, vowel, vowel2, morph, vibrato, nasal, seed):
    from .physical import story
    shapes, times = [vowel], None
    if vowel2 and float(morph) > 0.0:
        shapes, times = [vowel, vowel2], [0.0, float(dur) / float(np.clip(morph, 1e-3, 1.0))]
    n = int((dur + 0.05) * sr)
    t = np.arange(n) / sr
    f0 = float(freq) * (1.0 + 0.03 * float(vibrato) * np.sin(2 * np.pi * 5.0 * t) * np.clip((t - 0.3) / 0.3, 0.0, 1.0))
    p = story.kl_voice(shapes, f0, dur + 0.05, times=times, sr=sr, port=float(np.clip(nasal, 0.0, 2.0)), seed=seed)
    body = p[:int(dur * sr)]
    return p * (0.23 * (0.35 + 0.65 * min(float(vel), 1.2)) / (math.sqrt(float(np.mean(body * body))) + 1e-12))


@effect("formant_tract", "Impose a vowel's vocal-tract resonances on any sound (Webster tract, flow in at the glottis, flow out at the lips).",
        vowel=("a", "a|e|i|o|u"), shift=(1.0, "formant scale 0.5..2 (>1 = shorter tract, higher formants)"),
        mix=(1.0, "0..1 wet/dry"))
def formant_tract(x, sr=DEFAULT_SR, vowel="a", shift=1.0, mix=1.0):
    """Transfer u_out/u_in of the tract (the formant curves); unity gain at DC."""
    tr = _tract(vowel, size=1.0 / float(np.clip(shift, 0.4, 2.5)))
    return _through(x, sr, lambda u: tr.run(u)["u_out"], mix)


def _through(x, sr, fn, mix):
    """Run a mono kernel `fn` (at FS) over each channel, match the wet level to the dry one, mix."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        return np.stack([_through(x[:, c], sr, fn, mix) for c in range(x.shape[1])], axis=1)
    if x.size < 8:
        return x
    wet = _rs(fn(np.ascontiguousarray(_rs(x, sr, FS))), FS, sr)
    wet = np.concatenate([wet, np.zeros(max(0, x.size - wet.size))])[:x.size]
    wet = np.nan_to_num(wet) * (math.sqrt(float(np.mean(x * x))) / (math.sqrt(float(np.mean(wet * wet))) + 1e-12))
    m = float(np.clip(mix, 0.0, 1.0))
    return (1.0 - m) * x + m * wet


# ------------------------------------------------------------------------------------------ ducts and pipes
@effect("tube", "Through a pipe / vent / well / tunnel: the sound enters one end of a resonant duct and is heard at the other.",
        length_m=(1.0, "duct length 0.03..30 m (longer = lower, denser resonances and a later arrival)"),
        radius_m=(0.05, "duct radius 0.002..1 m (narrow = stronger wall loss and sharper resonances; 1-D model, valid below ~100/radius Hz)"),
        ends=("closed-open", "closed-open (source at a closed end: odd quarter-wave series) | open-open (half-wave series) | closed-closed (heard inside a sealed duct)"),
        loss=(1.0, "wall loss 0..20: 1 = smooth rigid pipe, higher = rough / lined / leaky"),
        mix=(1.0, "0..1 wet/dry"),
        steepen=(0.0, "kPa peak pressure in the duct: loud pulses steepen into shocks on the way (0 = linear)"),
        silencer=(0.0, "ms: absorptive silencer at the outlet, diffusive response peaking at this time (0 = none; 1-10)"))
def tube(x, sr=DEFAULT_SR, length_m=1.0, radius_m=0.05, ends="closed-open", loss=1.0, mix=1.0, steepen=0.0,
         silencer=0.0):
    """WebsterTube as an effect: cylinder with the passive radiation end(s) and
    viscothermal loss. The distance is a virtual tube
    reduced to one dimension: axial resonances and delay only, no transverse reverberant field.
    `steepen`: nonlinear travel time over the duct (`cyber.steepening`). `silencer`: convolution with
    `cyber.polack_ir`, D^2 = 1.5 x the peak time."""
    L = float(np.clip(length_m, 0.03, 30.0))
    a = float(np.clip(radius_m, 0.002, 1.0))
    left, right = {"closed-open": ("closed", "unflanged"), "open-open": ("unflanged", "unflanged"),
                   "closed-closed": ("closed", "closed")}[str(ends)]
    t = WebsterTube(L, 1.0, radius_m=a, left=left, right=right, loss=float(np.clip(loss, 0.0, 20.0)))
    key = "p_pick" if right == "closed" else "u_out"

    def run(u):
        y = t.run(u, src_pos=1.5 / t.N, pick_pos=0.72)[key]
        if steepen > 0.0:
            y = steepen_through(y, FS, L / C_AIR, 1e3 * float(steepen))
        if silencer > 0.0:
            y = silence_through(y, FS, 1e-3 * float(silencer))
        return y
    return _through(x, sr, run, mix)


def steepen_through(y, fs, travel, peak_pa):
    """Each sample of a pressure wave of peak `peak_pa` arrives travel x (k(p) - 1) late, k = cyber.steepening:
    compressions catch up, rarefactions lag; where they would overtake, they merge (a shock)."""
    from .cyber import steepening
    y = np.asarray(y, dtype=np.float64)
    m = float(np.max(np.abs(y)))
    if m <= 0.0 or y.size < 2:
        return y
    tg = np.arange(y.size) / fs
    ta = np.maximum.accumulate(tg + travel * (steepening(y / m * peak_pa) - 1.0)) + 1e-12 * np.arange(y.size) / fs
    return np.interp(tg, ta, y)


def silence_through(y, fs, peak_s):
    """Convolution with the diffusive response of an absorptive silencer whose peak is at `peak_s`."""
    from scipy.signal import fftconvolve
    from .cyber import polack_ir
    D = math.sqrt(1.5 * float(peak_s))
    h = polack_ir(D, max(8, int(100.0 * D * D * fs)), fs)
    return fftconvolve(np.asarray(y, dtype=np.float64), h)[:len(y)]


MESH_FS = 8000        # rate of the 3-D mesh: junction spacing sqrt(3) c / fs = 74.3 mm
MESH_REF_M = 0.42     # closest listening point: output level is referred to it


def mesh_grid(length_m, width_m, fs=MESH_FS):
    """(nx, ny, spacing_m) of the rectilinear 3-D mesh of a square duct; nz = ny."""
    dw = math.sqrt(3.0) * C_AIR / fs
    return max(2, int(length_m / dw)), max(3, int(round(width_m / dw)) - 1), dw


@numba.njit(cache=True)
def _mesh3d(src, nx, ny, nz, sy, sz, bx, bw, rec):
    """Leapfrog 3-D wave equation at Courant 1/sqrt(3) (the rectilinear waveguide mesh) with locally reacting
    walls of normalised admittance bx (the two ends) and bw (the four sides); source at x = 0."""
    lam2 = 1.0 / 3.0
    lam = math.sqrt(lam2)
    p0 = np.zeros((nx, ny, nz))
    p1 = np.zeros((nx, ny, nz))
    p2 = np.zeros((nx, ny, nz))
    m = rec.shape[0]
    out = np.zeros((src.shape[0], m))
    for t in range(src.shape[0]):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    s = 0.0
                    b = 0.0
                    if i > 0:
                        s += p1[i - 1, j, k]
                    else:
                        s += p1[i + 1, j, k]
                        b += bx
                    if i < nx - 1:
                        s += p1[i + 1, j, k]
                    else:
                        s += p1[i - 1, j, k]
                        b += bx
                    if j > 0:
                        s += p1[i, j - 1, k]
                    else:
                        s += p1[i, j + 1, k]
                        b += bw
                    if j < ny - 1:
                        s += p1[i, j + 1, k]
                    else:
                        s += p1[i, j - 1, k]
                        b += bw
                    if k > 0:
                        s += p1[i, j, k - 1]
                    else:
                        s += p1[i, j, k + 1]
                        b += bw
                    if k < nz - 1:
                        s += p1[i, j, k + 1]
                    else:
                        s += p1[i, j, k - 1]
                        b += bw
                    p2[i, j, k] = (lam2 * s + 2.0 * (1.0 - 3.0 * lam2) * p1[i, j, k]
                                   + (lam * b - 1.0) * p0[i, j, k]) / (1.0 + lam * b)
        p2[0, sy, sz] += src[t]
        for r in range(m):
            out[t, r] = p2[rec[r, 0], rec[r, 1], rec[r, 2]]
        p0, p1, p2 = p1, p2, p0
    return out


@effect("tube_mesh", "Heard down a long square duct (3-D mesh): the source sits at one end on the axis, two ears at `distance`; the level falls slower than in open air, the reverberation grows with distance.",
        distance=(1.68, "source-to-listener distance along the duct 0.1..length_m m"),
        length_m=(9.5, "duct length 1..20 m; both ends absorb completely"),
        width_m=(0.45, "side of the square section 0.25..1.5 m"),
        absorption=(0.2, "energy absorbed by the side walls at each reflection 0.01..1"),
        tail=(0.3, "seconds of ring-out added after the input"),
        mix=(1.0, "0..1 wet/dry"))
def tube_mesh(x, sr=DEFAULT_SR, distance=1.68, length_m=9.5, width_m=0.45, absorption=0.2, tail=0.3, mix=1.0):
    """Stereo out: right ear on the axis, left ear two junctions off it (about 15 cm). Level is referred to the closest point, 0.42 m, so a farther
    listener is quieter by what the duct takes away, not by a normalisation."""
    from scipy.signal import butter, sosfilt
    x = np.asarray(x, dtype=np.float64)
    mono = x.mean(axis=1) if x.ndim == 2 else x
    L = float(np.clip(length_m, 1.0, 20.0))
    nx, ny, dw = mesh_grid(L, float(np.clip(width_m, 0.25, 1.5)))
    n_out = mono.size + samples(float(np.clip(tail, 0.0, 5.0)), sr)
    u = _rs(np.concatenate([mono, np.zeros(n_out - mono.size)]), sr, MESH_FS)
    u = np.diff(u, prepend=0.0)
    c = ny // 2
    ix = int(np.clip(round(float(np.clip(distance, 0.1, L)) / dw), 1, nx - 1))
    ir = int(np.clip(round(MESH_REF_M / dw), 1, nx - 1))
    e = max(c - 2, 0)
    rec = np.array([[ix, e, c], [ix, c, c], [ir, e, c], [ir, c, c]], np.int64)
    R = math.sqrt(1.0 - float(np.clip(absorption, 0.01, 1.0)))
    out = _mesh3d(np.ascontiguousarray(u), nx, ny, ny, c, c, 1.0, (1.0 - R) / (1.0 + R), rec)
    g = math.sqrt(float(np.mean(mono * mono))) / (math.sqrt(float(np.mean(out[:, 2:] ** 2))) + 1e-12)
    wet = _rs(out[:, :2] * g, MESH_FS, sr)
    wet = np.concatenate([wet, np.zeros((max(0, n_out - wet.shape[0]), 2))])[:n_out]
    wet = sosfilt(butter(2, 20.0, "highpass", fs=sr, output="sos"), wet, axis=0)
    nf = min(n_out, samples(0.01, sr))
    wet[n_out - nf:] *= np.linspace(1.0, 0.0, nf)[:, None]
    dry = np.zeros((n_out, 2))
    dry[:mono.size] = x if x.ndim == 2 else mono[:, None]
    m = float(np.clip(mix, 0.0, 1.0))
    return (1.0 - m) * dry + m * wet


@lru_cache(maxsize=256)
def _pipe(freq, radius, loss):
    """Stopped (closed-open) cylinder whose first resonance, end correction included, is `freq`."""
    return tune_length(lambda L: WebsterTube(L, 1.0, radius_m=radius, right="unflanged", loss=loss), freq, C_AIR / (4.0 * freq))


@instrument("piston_pipe", "Baschet 'windless organ pipe': a membrane piston drives a quarter-wave pipe - deep, slow-speaking organ tone.",
            family="organ", span=("C2", "C4"),
            diameter=(0.2, "pipe diameter 0.12..0.36 m (Baschet); wider = more radiation loss, faster speech, duller ring"),
            rod=(0.35, "0..1 harmonic content of the driving rod (the pipe passes the odd harmonics)"),
            detune=(0.0, "cents the pipe is off the driving note: the tone weakens as they part"))
def piston_pipe(freq, dur, sr=DEFAULT_SR, vel=1.0, diameter=0.2, rod=0.35, detune=0.0):
    """A disc on a rubber membrane closes one end of a pipe one quarter-wavelength long and is driven
    at the note by a vibrating rod; "the air column amplifies when the oscillator frequency matches the pipe".
    Here: volume-velocity drive at the closed end of a WebsterTube, radiated pressure out."""
    a = float(np.clip(diameter, 0.05, 0.5)) / 2.0
    tb = _pipe(round(float(freq) * 2.0 ** (float(detune) / 1200.0), 4), a, 1.0)
    n = int((dur + 0.35) * FS)
    hold = int(dur * FS) + 1
    env = np.zeros(n)
    env[:hold] = _edge(hold, 0.012 * FS, 0.02 * FS)
    ph = 2 * np.pi * float(freq) * np.arange(n) / FS
    u = np.sin(ph)
    for h in range(2, 8):           # rod spectrum h^-1.5 (the thesis gives no spectrum of the drive)
        if h * freq < 3000.0:
            u += float(rod) * np.sin(h * ph) / h ** 1.5
    return _finish(tb.run(u * env)["p_out"], sr, tone=3500.0, peak=0.5 * (0.3 + 0.7 * min(float(vel), 1.2)))


@instrument("pvc_pipe", "Slapped plastic tube (thongophone / boomwhacker-like): a paddle slaps one end shut and the air column rings.",
            family="percussion", span=("C2", "C5"),
            radius=(0.0, "pipe radius in m, 0.004..0.12 (narrow = longer, purer ring); 0 = length/16, the same proportions on every note"),
            hardness=(0.5, "0..1 slap hardness: soft palm 0 (dull thud) .. hard paddle 1 (clicky, more overtones)"),
            loss=(1.5, "wall loss 0.5..10 (1 = smooth rigid pipe; higher = shorter ring)"),
            seed=(0, "seed for the slap's noise component"))
def pvc_pipe(freq, dur, sr=DEFAULT_SR, vel=1.0, radius=0.0, hardness=0.5, loss=1.5, seed=0):
    """End-slap excitation: a short volume-velocity pulse (the air the paddle pushes in) at the now-closed end of
    a WebsterTube with a radiating far end; the ring is the stopped pipe's odd-harmonic series."""
    # default proportions length/radius = 16 and the slap contact times below (no measured source)
    a = float(np.clip(radius if radius > 0 else C_AIR / (4.0 * freq) / 16.0, 0.004, 0.12))
    tb = _pipe(round(float(freq), 4), round(a, 5), float(np.clip(loss, 0.2, 20.0)))
    n = int((max(dur, 0.05) + 0.45) * FS)
    # contact time 0.9 .. 0.3 of the period (a bigger tube gets a bigger paddle), 0.3 .. 6 ms
    k = max(4, int(np.clip((0.9 - 0.6 * float(np.clip(hardness, 0.0, 1.0))) / freq, 3e-4, 6e-3) * FS))
    u = np.zeros(n)
    pulse = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(k) / k)
    u[:k] = pulse * (1.0 + 0.3 * np.random.default_rng(int(seed)).uniform(-1.0, 1.0, k))
    p = tb.run(u)["p_out"]
    off = int(max(dur, 0.05) * FS)                      # the paddle lifts: the column is let go
    p[off:] *= np.exp(-np.arange(n - off) / (0.09 * FS))
    return _finish(p, sr, tone=float(np.clip(freq * 14.0, 1500.0, 4500.0)), peak=1.15 * (0.25 + 0.75 * min(float(vel), 1.2)))


@sfx("pipe_blow", "Breath across the open end of a pipe: airy, noisy hoot that only hints at the pipe's pitch.",
     length_m=(0.3, "pipe length 0.03..5 m"), radius_m=(0.012, "pipe radius, m"),
     stopped=(True, "far end closed (panpipe / bottle neck, quarter-wave) or open (half-wave)"),
     dur=(1.2, "s"), breath=(0.7, "0..1 gustiness / roughness of the air stream"), seed=(0, "noise seed"))
def pipe_blow(sr=DEFAULT_SR, length_m=0.3, radius_m=0.012, stopped=True, dur=1.2, breath=0.7, seed=0):
    """Jet turbulence (band-limited noise, slowly gusting) injected just inside the radiating mouth of a
    WebsterTube; output is the pressure at that mouth. No jet feedback: it never locks into a steady tone."""
    L = float(np.clip(length_m, 0.03, 5.0))
    t = WebsterTube(L, 1.0, radius_m=float(np.clip(radius_m, 0.002, 0.2)), left="unflanged",
                    right="closed" if stopped else "unflanged", loss=1.5)
    n = int(max(dur, 0.05) * FS)
    rng = np.random.default_rng(int(seed))
    # gust rate (6 Hz lowpassed noise) and the jet-noise band (up to 2.5x the pipe's lowest
    # resonance: the radiated pressure rises 6 dB/octave, so a wider band buries the fundamental) are by ear
    gust = 1.0 + float(np.clip(breath, 0.0, 1.0)) * 0.6 * F.lowpass(rng.standard_normal(n), 6.0, FS) * 60.0
    fc = float(np.clip(2.5 * C_AIR / ((4.0 if stopped else 2.0) * L), 250.0, 2500.0))
    u = F.lowpass(rng.standard_normal(n), fc, FS, order=2) * np.clip(gust, 0.1, 3.0) * _edge(n, 0.12 * FS, 0.3 * FS)
    return _finish(t.run(u, src_pos=1.0 / t.N)["p_in"], sr, tone=4000.0, peak=0.7)


WAH_PARTIALS = 12


@sfx("wah_pipe", "Long pipe blown by a fan through a rubber flap that swells: the pressure ramp brings in the fundamental, then the partials one by one; a half-stopped end beats its open and stopped series.",
     length_m=(6.1, "pipe length, m (0.5..12)"), diameter=(0.254, "pipe diameter, m"), dur=(4.0, "s"),
     ramp=(2.5, "s for the flap pressure to rise to full"), pressure=(1.0, "0..1 final blowing pressure: how many partials sound"),
     stop=(0.0, "0 open end .. 1 stopped end; in between both series sound and beat"), seed=(0, "variation"))
def wah_pipe(sr=DEFAULT_SR, length_m=6.1, diameter=0.254, dur=4.0, ramp=2.5, pressure=1.0, stop=0.0, seed=0):
    """Partial n of the open pipe f = n c / (2 (L + 2 x 0.6133 a)) speaks once the pressure passes n / N of full scale
    (each register needs a higher blowing pressure), its level rising smoothly past threshold; the stopped series
    (2m - 1) c / (4 (L + 0.6133 a)) is mixed in by ``stop``. Fan turbulence rides on the pressure."""
    L = float(np.clip(length_m, 0.5, 12.0))
    a = 0.5 * float(np.clip(diameter, 0.01, 0.6))
    n = int(max(dur, 0.1) * FS)
    t = np.arange(n) / FS
    rng = np.random.default_rng(int(seed))
    p = float(np.clip(pressure, 0.0, 1.0)) * np.clip(t / max(float(ramp), 1e-3), 0.0, 1.0) ** 1.5
    p = p * (1.0 + 0.05 * F.lowpass(rng.standard_normal(n), 4.0, FS) * 30.0)
    s = float(np.clip(stop, 0.0, 1.0))
    f_open = C_AIR / (2.0 * (L + 2.0 * UNFLANGED * a))
    f_stop = C_AIR / (4.0 * (L + UNFLANGED * a))
    y = np.zeros(n)
    for k in range(1, WAH_PARTIALS + 1):
        on = np.clip((p - (k - 1) / WAH_PARTIALS) * WAH_PARTIALS, 0.0, 1.0) ** 2
        for f, g in ((k * f_open, 1.0 - s), ((2 * k - 1) * f_stop, s)):
            if g > 0 and f < 0.45 * FS:
                y += g * on / k ** 0.5 * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    y += 0.03 * p * F.bandpass(rng.standard_normal(n), 400.0, FS, 0.7)
    return _finish(y * _edge(n, 0.02 * FS, 0.3 * FS), sr, tone=3000.0, peak=0.7)


# ------------------------------------------------------------------------------------------ blown bottle
@numba.njit(cache=True)
def _bottle_kernel(f, r, bp, vib, noise, noise_gain):
    """Blown-bottle tick at 44.1 kHz: Helmholtz resonator = normalised 2-pole BiQuad in a
    loop with the cubic jet table; the turbulence is modulated by the flow."""
    n = bp.size
    out = np.zeros(n)
    b0 = 0.5 - 0.5 * r * r
    a2 = r * r
    x1 = x2 = y1 = y2 = 0.0
    dx = dy = 0.0
    for i in range(n):
        a1 = -2.0 * r * math.cos(2.0 * math.pi * f[i] / FS)
        b = bp[i] + vib[i]
        pd = b - y1
        rnd = noise_gain * noise[i] * b * (1.0 + pd)
        jt = pd * (pd * pd - 1.0)
        jt = min(1.0, max(-1.0, jt))
        x = b + rnd - jt * pd
        y = b0 * x - b0 * x2 - a1 * y1 - a2 * y2
        x2 = x1
        x1 = x
        y2 = y1
        y1 = y
        dy = pd - dx + 0.99 * dy
        dx = pd
        out[i] = 0.2 * dy
    return out


@instrument("bottle", "Blown bottle / jug (STK BlowBotl): Helmholtz resonator with flow-modulated breath noise, hooty and airy.",
            family="wind", span=("C3", "C6"),
            noise=(0.12, "0..1.5 breath noise = STK noiseGain / 40. 0.5 is the STK default (very airy: the pitch then wanders +-10 cents or more with the turbulence), 0.75 = creature breath"),
            vibrato=(0.0, "0..1 breath vibrato depth (5.925 Hz, STK gain 0.4 at 1)"),
            bright=(1.0, "0.25..4 output lowpass scale"), seed=(0, "noise seed"))
def bottle(freq, dur, sr=DEFAULT_SR, vel=1.0, noise=0.12, vibrato=0.0, bright=1.0, seed=0):
    """Blown-bottle model: resonance radius 0.999, ADSR 5 ms / 10 ms / 0.8 / 10 ms,
    breath pressure 1.1 + 0.2 amp, output 0.2 (amp + 0.001) dcBlock(pd)."""
    amp = float(np.clip(vel, 0.05, 1.2))
    n = int((dur + 0.12) * FS)
    hold = int(dur * FS) + 1
    a, d = int(0.005 * FS), int(0.010 * FS)
    env = np.full(n, 0.8)
    env[:a] = np.linspace(0.0, 1.0, a)
    env[a:a + d] = np.linspace(1.0, 0.8, d)
    env[hold:] = 0.8 * np.exp(-np.arange(n - hold) / (0.010 * FS))
    t = np.arange(n) / FS
    mi, _, gg = _CAL["bottle"]
    m = _midi(freq)
    f = np.full(n, float(np.clip(freq, 20.0, 0.4 * FS)))
    # STK's fixed radius 0.999 is Q = 36 at its default 500 Hz; below that it is held as a constant Q (a real
    # bottle's Q does not fall with its size), otherwise a low note is 14 Hz of noise around a 65 Hz centre
    r = max(0.999, math.exp(-math.pi * f[0] / (35.6 * FS)))
    nz = np.random.default_rng(int(seed)).uniform(-1.0, 1.0, n)
    y = _bottle_kernel(f, r, (1.1 + 0.2 * amp) * env, 0.4 * float(vibrato) * np.sin(2 * np.pi * 5.925 * t) * env,
                       nz, 40.0 * float(np.clip(noise, 0.0, 1.5)))
    return _finish(y * (amp + 0.001) * float(np.interp(m, mi, gg)), sr, tone=float(np.clip(freq * 8.0, 2000.0, 4500.0)) * float(np.clip(bright, 0.25, 4.0)))


# ------------------------------------------------------------------------------------------ calibration
def _cents_off(y, f0, sr=FS):
    """tests/test_instruments.py::cents_off (autocorrelation peak next to the expected period)."""
    y = y[int(0.05 * sr):int(0.05 * sr) + 16384]
    y = y - y.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(y * np.hanning(len(y)), 4 * len(y))) ** 2)
    p = sr / f0 * 4
    lo = int(p * 0.94)
    k = lo + int(np.argmax(ac[lo:int(p * 1.06) + 2]))
    k = k + 0.5 * (ac[k - 1] - ac[k + 1]) / (ac[k - 1] - 2 * ac[k] + ac[k + 1] + 1e-20)
    return 1200 * np.log2((sr * 4 / k) / f0)


def _calibrate(names=("reed_tube", "reed_cone", "bottle"), step=2, passes=4):
    """Measure-and-compensate: play every `step` semitones across (and a little beyond) each span at vel 0.9,
    correct the tuning target by the measured pitch error and the gain to -12 dB K-weighted, then fit the
    pressure slope of the reeds at vel 0.3. Prints the `_CAL` literal to paste above."""
    from .analysis import loudness
    from .instruments import REGISTRY, render_note
    from .notes import midi_to_freq, note_to_midi
    for name in names:
        lo, hi = (note_to_midi(s) for s in REGISTRY[name]["range"].split("-"))
        grid = np.arange(lo - 6.0, hi + 8.0, step)
        cc, gg = np.zeros(grid.size), np.full(grid.size, _CAL[name][2][0])
        extra = _CAL[name][3:]
        for _ in range(passes):
            _CAL[name] = (grid, cc.copy(), gg.copy()) + extra
            _reed_bore.cache_clear()
            for i, m in enumerate(grid):
                f = midi_to_freq(m)
                e = _cents_off(render_note(name, f, 0.6, FS, 0.9), f)
                if abs(e) < 60.0 and extra:      # only the reeds need a pitch correction
                    cc[i] -= e
                gg[i] *= 10.0 ** ((-12.0 - loudness(render_note(name, f, 0.5, FS, 0.9), FS, window=0.15)) / 20.0)
        if extra:                               # least-squares fit of the two pressure slopes at vel 0.3

            def soft(s0, s1):
                _CAL[name] = (grid, cc, gg, (s0, s1))
                _reed_bore.cache_clear()
                return np.array([_cents_off(render_note(name, midi_to_freq(m), 0.6, FS, 0.3), midi_to_freq(m))
                                 for m in grid if lo <= m <= hi])

            s0, s1 = extra[0]
            for _ in range(3):
                e0 = soft(s0, s1)
                d0 = (soft(s0 + 0.2, s1) - e0) / 0.2
                d1 = np.full(e0.size, (0.027 - REED_PM[0] - 0.3 * REED_PM[1]) / 0.006)
                ok = np.abs(e0) < 60.0
                sol = np.linalg.lstsq(np.stack([d0[ok], d1[ok]], axis=1), -e0[ok], rcond=None)[0]
                s0, s1 = s0 + float(sol[0]), s1 + float(sol[1])
            extra = ((round(s0, 3), round(s1, 2)),)
        _CAL[name] = (grid, cc, gg) + extra
        r = lambda a, d: "(" + ", ".join(f"{v:.{d}f}" for v in a) + ")"   # noqa: E731
        print(f'    "{name}": ({r(grid, 0)},\n        {r(cc, 1)},\n        {r(gg, 3 if gg.max() < 50 else 0)}'
              + (f",\n        {extra[0]}" if extra else "") + "),")



# ---------------------------------------------------------------------------------------------------------
# radiation: rational impedance and reflection of a pipe end
# ---------------------------------------------------------------------------------------------------------
RAD_RATIONAL = {   # (a1, a2), (b1, b2) of Zr(s) = (a1 s + a2 s^2) / (1 + b1 s + b2 s^2), s = j k a
    "rayleigh1": ((0.8488, 0.0), (0.8488, 0.0)),
    "rayleigh2": ((0.8488, 0.4), (1.0186, 0.4)),
    "levine2": ((0.6133, 0.21), (0.7, 0.21)),
}
END_REFLECTION = {"flanged": (0.182, 1.825, 0.649), "unflanged": (0.167, 1.393, 0.457)}   # n1, d1, d2


def radiation_impedance(ka, model="levine2"):
    """Normalised radiation impedance Zr / Zc of a pipe end, rational fit: "rayleigh1" / "rayleigh2" (flanged
    piston), "levine2" (unflanged pipe). Mass end correction = a1 (0.8488 flanged, 0.6133 unflanged)."""
    s = 1j * np.asarray(ka, dtype=np.float64)
    (a1, a2), (b1, b2) = RAD_RATIONAL[model]
    return (a1 * s + a2 * s * s) / (1.0 + b1 * s + b2 * s * s)


def end_reflection(ka, end="unflanged"):
    """Pressure reflection of a pipe end, R = -(1 + n1 x)/(1 + d1 x + d2 x^2), x = j k a. Low ka: R = -exp(-2jk delta)
    with delta / a = (d1 - n1) / 2 (0.82 flanged, 0.61 unflanged)."""
    x = 1j * np.asarray(ka, dtype=np.float64)
    n1, d1, d2 = END_REFLECTION[end]
    return -(1.0 + n1 * x) / (1.0 + d1 * x + d2 * x * x)


def end_reflection_filter(radius_m, sr, end="unflanged"):
    """Digital biquad (b, a) of `end_reflection` for a waveguide termination (bilinear transform, x = s a / c)."""
    from scipy.signal import bilinear
    n1, d1, d2 = END_REFLECTION[end]
    tau = float(radius_m) / C_AIR
    return bilinear([-n1 * tau, -1.0], [d2 * tau * tau, d1 * tau, 1.0], fs=float(sr))


# ---------------------------------------------------------------------------------------------------------
# 2-D rectilinear waveguide mesh of the vocal tract
# ---------------------------------------------------------------------------------------------------------
MESH2D_DX = 0.004      # m: junction spacing; fs = c sqrt(2) / dx = 121 kHz
MESH2D_WALL = 0.97     # pressure kept by a wall reflection per step (soft tissue loss)


def mesh2d_shape(area_cm2, length_m, dx=MESH2D_DX):
    """Boolean mask (columns along the tract, rows across): each column as wide as the diameter of the area there."""
    nx = max(4, int(round(float(length_m) / dx)))
    a = np.interp(np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, len(area_cm2)), np.asarray(area_cm2, float))
    w = np.maximum(1, np.round(2.0 * np.sqrt(np.maximum(a, 1e-3) * 1e-4 / math.pi) / dx).astype(np.int64))
    ny = int(w.max()) + 2
    mask = np.zeros((nx, ny), dtype=np.bool_)
    for i, wi in enumerate(w):
        lo = (ny - wi) // 2
        mask[i, lo:lo + wi] = True
    return mask


@numba.njit(cache=True)
def _mesh2d(src, mask, g):
    """Leapfrog 2-D mesh p(n+1) = (1/2) sum of the 4 neighbours - p(n-1): glottis rigid (column -1 mirrors),
    lips open (p = 0 beyond the last column), walls reflect g p. Source added on the first column; returns the
    mean pressure of the last column."""
    nx, ny = mask.shape
    p, p1, pn = np.zeros((nx, ny)), np.zeros((nx, ny)), np.zeros((nx, ny))
    out = np.zeros(src.size)
    for s in range(src.size):
        for i in range(nx):
            for j in range(ny):
                if not mask[i, j]:
                    continue
                c = p[i, j]
                acc = c if i == 0 else (p[i - 1, j] if mask[i - 1, j] else g * c)
                acc += 0.0 if i == nx - 1 else (p[i + 1, j] if mask[i + 1, j] else g * c)
                acc += p[i, j - 1] if (j > 0 and mask[i, j - 1]) else g * c
                acc += p[i, j + 1] if (j < ny - 1 and mask[i, j + 1]) else g * c
                pn[i, j] = 0.5 * acc - p1[i, j]
        m = 0
        for j in range(ny):
            if mask[0, j]:
                pn[0, j] += src[s]
            if mask[nx - 1, j]:
                out[s] += pn[nx - 1, j]
                m += 1
        out[s] /= max(m, 1)
        p1, p, pn = p, pn, p1
    return out


@instrument("mesh_vowel", "Sung vowel through a 2-D waveguide mesh of the vocal tract (as wide as the measured tract): "
            "the cross modes add the upper formant detail a 1-D tube lacks.", family="voice", span=("C2", "C5"),
            vowel=("a", "a e i o u"), vibrato=(0.4, "0..1"))
def mesh_vowel(freq, dur, sr=DEFAULT_SR, vel=1.0, vowel="a", vibrato=0.4):
    from .physical import story
    fs = C_AIR * math.sqrt(2.0) / MESH2D_DX
    mask = mesh2d_shape(story.vowel_area(vowel), story.vowel_length(vowel))
    n = int((float(dur) + 0.1) * fs)
    u = glottal_pulses(float(freq), n, vibrato=float(vibrato), sr=fs)
    y = np.diff(_mesh2d(u - u.mean(), mask, MESH2D_WALL), prepend=0.0)
    y = resample_poly(y, FS, int(round(fs)))
    return _finish(y, sr, peak=0.6 * (0.5 + 0.5 * vel))


# ---------------------------------------------------------------------------------------------------------
# bar over a stopped box resonator
# ---------------------------------------------------------------------------------------------------------
BOX_RADIUS = 0.078     # m: welded box resonator, 156 mm across
BOX_COUPLING = 0.03    # bar-air coupling (fraction of the product of the two angular frequencies)
BOX_Q = 30.0           # quality factor of the stopped resonator (radiation + wall loss)


def box_length(freq, radius=BOX_RADIUS):
    """Length of a stopped tube resonator tuned to `freq`: L = c/(4 f) - 0.6 r."""
    return C_AIR / (4.0 * float(freq)) - 0.6 * float(radius)


def bar_resonator(f_bar, f_res, sr, dur, coupling=BOX_COUPLING, t60_bar=4.0, q_res=BOX_Q):
    """Struck bar mode and its resonator as two coupled oscillators; returns (bar velocity, resonator velocity).
    Tuned close, they trade energy fast: loud but short; detuned, the bar keeps its energy and rings on."""
    n = samples(dur, sr)
    k = 1.0 / sr
    wb, wr = 2 * math.pi * f_bar, 2 * math.pi * f_res
    sb, sres = 6.91 / t60_bar, wr / (2.0 * q_res)
    mu = coupling * wb * wr
    x = y = 0.0
    vx, vy = 1.0, 0.0
    ox, oy = np.zeros(n), np.zeros(n)
    for i in range(n):
        vx += k * (-wb * wb * x - 2.0 * sb * vx + mu * y)
        vy += k * (-wr * wr * y - 2.0 * sres * vy + mu * x)
        x += k * vx
        y += k * vy
        ox[i], oy[i] = vx, vy
    return ox, oy


@instrument("box_bar", "Metal bar over a stopped box resonator: in tune the box takes the bar's energy (loud, short); "
            "`detune` the box and the bar rings on, quieter.", family="mallet", span=("C3", "C6"),
            detune=(20.0, "cents between box and bar"), decay=(4.0, "s, T60 of the bare bar"))
def box_bar(freq, dur, sr=DEFAULT_SR, vel=1.0, detune=20.0, decay=4.0):
    f = float(freq)
    ring = max(float(dur), float(decay))
    xb, xr = bar_resonator(f, f * 2.0 ** (float(detune) / 1200.0), sr, ring, t60_bar=float(decay))
    y = 0.15 * xb + xr
    t = np.arange(y.size) / sr
    for r, a in ((2.756, 0.3), (5.404, 0.12)):
        if r * f < 0.45 * sr:
            y = y + a * np.max(np.abs(xr)) * np.sin(2 * np.pi * r * f * t) * np.exp(-6.91 * t * r / float(decay))
    a0 = samples(0.001, sr)
    y[:a0] *= np.linspace(0.0, 1.0, a0)
    return y * (0.7 / max(float(np.max(np.abs(y))), 1e-12)) * (0.5 + 0.5 * vel)



if __name__ == "__main__":
    _calibrate()
