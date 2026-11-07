"""1D finite-difference strings and bars (Bilbao, *Numerical Sound Synthesis*, Wiley 2009, ch. 6-8).

Findings files: out/research/g01-bilbao-part1.md (operators, hammer, bow, rattle) and
g02-bilbao-part2.md sections 1-14 (bar, stiff string, loss, bow, hammer, courses, prepared strings,
varying bars, Kirchhoff-Carrier). Reference listings: book Appendix A.7 (bar), A.8 (stiff string),
A.9 (Kirchhoff-Carrier).

One explicit kernel advances  phi u_tt = gamma^2 u_xx - kappa^2 (phi^3 u_xx)_xx - 2 s0 u_t + 2 s1 u_txx
(scheme 7.30a; phi = 1 for strings, gamma = 0 for bars) with simply-supported / clamped / free ends,
plus, solved semi-implicitly at single grid points (Rule of Thumb #3): a power-law hammer shared by
1-3 strings (7.37/7.38), springs / cubic springs / dampers (7.42) and rattles (Problem 7.19), and the
conservative Kirchhoff-Carrier tension modulation (8.9). A second kernel bows the string (7.32-7.34).

Loss convention: amplitude decays as exp(-(s0 + s1 beta^2) t) and T60 is a true 60 dB, so
sigma = 3 ln(10) / T60. The book prints T60 = 6 ln(10)/sigma (eq 7.29) but its own listing A.8 enters
the loss terms with half weight, which is the same thing.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from numba import njit
from scipy.linalg import eig_banded
from scipy.signal import butter, cheby1, ellip, lfilter, resample_poly, sosfilt

from . import filters as F
from .core import DEFAULT_SR, samples
from .fx import effect
from .instruments import instrument
from .sfx import sfx

SS, CL, FR = 0, 1, 2            # simply supported, clamped, free
IMPLICIT_GRID = 4.0
THETA_4TH = 1.0 / 12.0
SECOND_CUT_W = 0.05
INTERCEPT_W, INTERCEPT_CLOTH_W = 3.0e4, 3.0e3
INTERCEPT_ARM = 20.0
SCHEMES = {"explicit": (0.0, 0.0), "theta4": (THETA_4TH, 0.0), "stable": (0.25, 0.0), "bensa": (0.25, 0.25)}
_BC = {"ss": SS, "supported": SS, "clamped": CL, "free": FR}
_LN1000 = 3.0 * np.log(10.0)


# ------------------------------------------------------------------ kernels
@njit(cache=True)
def _d2(u, q, N, bcl, bcr):
    """q = h^2 d_xx u; end values carry the boundary condition (clamped: ghost u[-1] = u[1])."""
    for l in range(1, N):
        q[l] = u[l + 1] - 2.0 * u[l] + u[l - 1]
    q[0] = 2.0 * u[1] if bcl == 1 else 0.0
    q[N] = 2.0 * u[N - 1] if bcr == 1 else 0.0


@njit(cache=True)
def _lin(u, u1, qv, qp, G, W, Bb, N, lam2, mu2, s0, s1, bcl, bcr, phi3, iphi, theta=0.0, r=0.0):
    """Linear part of the scheme, split as u+ = W + g*Bb (Bb is the tension term, g = 1 if linear).
    theta, r > 0: the explicit parts of the theta-weighted stiffness term theta u+ + (1 - 2 theta) u + theta u-
    and of the r-weighted tension term; the u+ parts are left to the implicit solve.

    The 4th difference is D2(phi^3 D2 u) = D2^T D2, symmetric for every end type, so the lossless
    scheme conserves 1/2|d_t- u|^2 + 1/2 <u, K e_t- u> exactly."""
    _d2(u, qv, N, bcl, bcr)
    for l in range(N + 1):
        G[l + 1] = phi3[l] * ((1.0 - 2.0 * theta) * qv[l] + theta * qp[l])
    lo = 0 if bcl == 2 else 1
    hi = N if bcr == 2 else N - 1
    inv = 1.0 / (1.0 + s0)
    for l in range(lo, hi + 1):
        d2 = qv[l]
        d2p = qp[l]
        if l == 0:                      # free end: Neumann difference in the tension / loss terms
            d2 = u[1] - u[0]
            d2p = u1[1] - u1[0]
        elif l == N:
            d2 = u[N - 1] - u[N]
            d2p = u1[N - 1] - u1[N]
        d4 = G[l + 2] - 2.0 * G[l + 1] + G[l]
        W[l] = (2.0 * u[l] - (1.0 - s0) * u1[l] + iphi[l] * (s1 * (d2 - d2p) - mu2 * d4)) * inv
        Bb[l] = iphi[l] * lam2 * ((1.0 - 2.0 * r) * d2 + r * d2p) * inv


@njit(cache=True)
def _band_factor(A, p):
    """In-place LU (no pivoting) of a banded matrix of half-bandwidth p, stored dense."""
    n = A.shape[0]
    for i in range(n):
        for rr in range(i + 1, min(i + p + 1, n)):
            f = A[rr, i] / A[i, i]
            A[rr, i] = f
            for c in range(i + 1, min(i + p + 1, n)):
                A[rr, c] -= f * A[i, c]


@njit(cache=True)
def _band_solve(A, p, x):
    n = A.shape[0]
    for i in range(n):
        for c in range(max(0, i - p), i):
            x[i] -= A[i, c] * x[c]
    for i in range(n - 1, -1, -1):
        for c in range(i + 1, min(i + p + 1, n)):
            x[i] -= A[i, c] * x[c]
        x[i] /= A[i, i]


@njit(cache=True)
def _hr(u, HI, HW):
    s = 0.0
    for i in range(HI.shape[0]):
        s += HW[i] * u[HI[i]]
    return s


@njit(cache=True)
def _dgrad(e, e1, wHa, al):
    """Discrete gradient of Psi(e) = wHa [e]+^(al+1) / (al+1) between e1 and e, and its derivative in e."""
    pe = wHa * max(e, 0.0) ** (al + 1.0) / (al + 1.0)
    d = e - e1
    if abs(d) < 1e-13:
        return wHa * max(e, 0.0) ** al, 0.5 * al * wHa * max(e, 0.0) ** (al - 1.0) if e > 0.0 else 0.0
    p1 = wHa * max(e1, 0.0) ** (al + 1.0) / (al + 1.0)
    g = (pe - p1) / d
    return g, (wHa * max(e, 0.0) ** al - g) / d


@njit(cache=True)
def _fd_kernel(U, U1, nstep, N, h, k, lam2, mu2, s0, s1, bcl, bcr, phi3, iphi, kc, ham, el, RS, pick, rig, want_e,
               LW, LC, LR2, LB, LX, LF, theta, rr, LU, COLS, HI, HW, PX):
    """U, U1: (strings, N+1) displacement at steps n, n-1. Returns (out[n,2], energy, hammer force, U, U1).

    ham = [on, grid index, mass ratio M, wH^(1+alpha), alpha, uH, uH_prev]
    el rows = [kind (1 spring/damper, 2 rattle), grid index, w0^2, w1^4, sigma_P, wR^(1+alpha), alpha, eps, M_rattle]
    RS[string, element] = (uR, uR_prev) rattle positions. kc = alpha^2 of Kirchhoff-Carrier (0 = linear).
    rig: acceleration per unit hammer force that cancels the rigid-body part of the blow (free-free bars).
    LW, LC, LR2, LB: longitudinal modes k = 1..K (omega, 2 r cos(omega k), r^2, gain); LX[string, k] = (xi_n, xi_n-1);
    LF[string, k] the last modal force. Their dynamic part sets the tension along the string (tension decomposition).
    theta, rr > 0: implicit stiffness / tension weights; LU[string] is the band LU of the implicit matrix and
    COLS[string, j] its inverse applied to the grid point of the hammer (j = 0) and of element j - 1.
    ham[7] = 1: energy-conserving hammer (discrete gradient of the felt potential, Newton), ham[8] its loss rH.
    HI, HW: grid points and weights (sum 1) of the hammer's contact; one point of weight 1 for a point hammer."""
    nq = U.shape[0]
    imp = theta > 0.0 or rr > 0.0
    cons = ham.shape[0] > 7 and ham[7] > 0.5
    rH = ham[8] if ham.shape[0] > 8 else 0.0
    cq = np.zeros(nq)
    xq = np.zeros(nq)
    eq = np.zeros(nq)
    e1q = np.zeros(nq)
    Jm = np.zeros((nq, nq))
    Rv = np.zeros(nq)
    Un = np.zeros_like(U)
    Q = np.zeros_like(U)
    Qp = np.zeros_like(U)
    G = np.zeros(N + 3)
    W = np.zeros(N + 1)
    Bb = np.zeros(N + 1)
    out = np.zeros((nstep, 2))
    en = np.zeros(nstep if want_e else 1)
    fh = np.zeros(nstep)
    for q in range(nq):
        _d2(U1[q], Qp[q], N, bcl, bcr)
    ham_on = ham[0] > 0.5
    lH = int(ham[1])
    Mr = ham[2]
    wHa = ham[3]
    alH = ham[4]
    uH = ham[5]
    uH1 = ham[6]
    c0 = k * k / ((1.0 + s0) * h)
    ne = el.shape[0]
    pi_ = np.zeros(2, np.int64)
    pf = np.zeros(2)
    for c in range(2):
        p = min(max(pick[c], 0.0), 1.0) * N
        pi_[c] = min(int(p), N - 1)
        pf[c] = p - pi_[c]
    A_ = np.zeros(nq)
    e_ = np.zeros(nq)
    ek = h / (2.0 * k * k)
    KL = LW.shape[0]
    CS = np.zeros((KL, N))
    for j in range(KL):
        for l in range(N):
            CS[j, l] = math.cos((j + 1) * math.pi * (l + 0.5) * h)
    tau = np.zeros(N)
    sx = np.zeros(N)
    for n in range(nstep):
        E = 0.0
        for q in range(nq):
            u = U[q]
            u1 = U1[q]
            qv = Q[q]
            qp = Qp[q]
            un = Un[q]
            _lin(u, u1, qv, qp, G, W, Bb, N, lam2[q], mu2[q], s0, s1, bcl, bcr, phi3, iphi, theta, rr)
            if KL > 0:
                for l in range(N):
                    sx[l] = 0.5 * ((u[l + 1] - u[l]) / h) ** 2
                    tau[l] = 0.0
                for j in range(KL):
                    sp = 0.0
                    for l in range(N):
                        sp += sx[l] * CS[j, l]
                    w2 = LW[j] * LW[j]
                    Fj = -2.0 * w2 / ((j + 1) * math.pi) * sp * h
                    xt = LX[q, j, 0] - Fj / w2
                    for l in range(N):
                        tau[l] += kc * math.pi * (j + 1) * xt * CS[j, l]
                    xn = LC[j] * LX[q, j, 0] - LR2[j] * LX[q, j, 1] + LB[j] * Fj
                    LX[q, j, 1] = LX[q, j, 0]
                    LX[q, j, 0] = xn
                    LF[q, j] = Fj
                cf = lam2[q] / (1.0 + s0)
                for l in range(1, N):
                    W[l] += iphi[l] * cf * (tau[l] * (u[l + 1] - u[l]) - tau[l - 1] * (u[l] - u[l - 1]))
            if imp:
                _band_solve(LU[q], 2, W)
                _band_solve(LU[q], 2, Bb)
            g = 1.0
            if kc > 0.0:                # (8.9): g = 1 + alpha^2/4 (<p,p+> + <p,p->), linear in u+
                sa = 0.0
                sb = 0.0
                for l in range(1, N):
                    sa += qv[l] * (W[l] + u1[l])
                    sb += qv[l] * Bb[l]
                g = (1.0 - kc / (4.0 * h) * sa) / (1.0 + kc / (4.0 * h) * sb)
            for l in range(N + 1):
                un[l] = W[l] + g * Bb[l]
            if want_e:
                ke = 0.0
                p2 = 0.0
                p4 = 0.5 * (phi3[0] * qv[0] * qp[0] + phi3[N] * qv[N] * qp[N])
                for l in range(N + 1):
                    ke += (u[l] - u1[l]) ** 2 / iphi[l]
                for l in range(N):
                    p2 += (u[l + 1] - u[l]) * (u1[l + 1] - u1[l])
                for l in range(1, N):
                    p4 += phi3[l] * qv[l] * qp[l]
                E += ek * (ke + lam2[q] * p2 + mu2[q] * p4) + lam2[q] * h * h / (k * k) * kc / 8.0 * (p2 / h) ** 2
                for j in range(ne):
                    l = int(el[j, 1])
                    if el[j, 0] == 1.0:
                        E += 0.25 * el[j, 2] * (u[l] ** 2 + u1[l] ** 2) + 0.25 * el[j, 3] * u[l] ** 2 * u1[l] ** 2
                    elif el[j, 8] > 0.0:
                        E += 0.5 / el[j, 8] * ((RS[q, j, 0] - RS[q, j, 1]) / k) ** 2
        if want_e:
            if ham_on:
                E += 0.5 * Mr * ((uH - uH1) / k) ** 2
            en[n] = E
        if ham_on:                      # (7.37)/(7.38): F_q = G_q mu_t.(eta_q), one hammer on all strings
            cr = k * k / (1.0 + s0)
            c = -cr * rig[lH]
            for i_ in range(HI.shape[0]):
                c += c0 * iphi[HI[i_]] * HW[i_] * HW[i_]
            for q in range(nq):
                cq[q] = c
                if imp:
                    cq[q] = -cr * rig[lH]
                    for i_ in range(HI.shape[0]):
                        cq[q] += c0 * HW[i_] * COLS[q, 0, HI[i_]]
            pred = 2.0 * uH - uH1
            S = 0.0
            if cons:
                for q in range(nq):
                    e1q[q] = uH1 - _hr(U1[q], HI, HW)
                    xq[q] = pred - _hr(Un[q], HI, HW)
                for it in range(30):
                    sa = 0.0
                    for q in range(nq):
                        gq, dq = _dgrad(xq[q], e1q[q], wHa, alH)
                        if xq[q] > 0.0 or e1q[q] > 0.0:
                            gq += rH * (xq[q] - e1q[q]) / (2.0 * k)
                            dq += rH / (2.0 * k)
                        else:
                            gq, dq = 0.0, 0.0
                        A_[q] = max(gq, 0.0)
                        eq[q] = dq if gq > 0.0 else 0.0
                        sa += A_[q]
                    err = 0.0
                    for q in range(nq):
                        Rv[q] = xq[q] - (pred - _hr(Un[q], HI, HW)) + k * k * sa + cq[q] * Mr * A_[q]
                        err = max(err, abs(Rv[q]))
                        for p_ in range(nq):
                            Jm[q, p_] = (1.0 if p_ == q else 0.0) + (k * k + (cq[q] * Mr if p_ == q else 0.0)) * eq[p_]
                    if err < 1e-15 * (1.0 + abs(pred)):
                        break
                    dx = np.linalg.solve(Jm, Rv)
                    for q in range(nq):
                        xq[q] -= dx[q]
                for q in range(nq):
                    S += A_[q]
            else:
                SA = 0.0
                SAe = 0.0
                for q in range(nq):
                    eta = uH - _hr(U[q], HI, HW)
                    Gq = min(wHa * eta ** (alH - 1.0), 1.0 / (k * k + cq[q] * Mr)) if eta > 0.0 else 0.0
                    A_[q] = 0.5 * Gq / (1.0 + 0.5 * Gq * cq[q] * Mr)
                    # [eta-]+ here and F >= 0 below: no adhesive pull at make/break. What remains is the energy
                    # credited at first detection, M G eta^2/4 ~ (k wH)^2/2 of the hammer energy for alpha = 1
                    e_[q] = pred - _hr(Un[q], HI, HW) + max(uH1 - _hr(U1[q], HI, HW), 0.0)
                    SA += A_[q]
                    SAe += A_[q] * e_[q]
                S0 = SAe / (1.0 + k * k * SA)
                for q in range(nq):
                    A_[q] = max(A_[q] * (e_[q] - k * k * S0), 0.0)
                    S += A_[q]
            for q in range(nq):
                Fq = Mr * A_[q]
                if imp:
                    for l in range(N + 1):
                        Un[q, l] += c0 * Fq * COLS[q, 0, l]
                else:
                    for i_ in range(HI.shape[0]):
                        Un[q, HI[i_]] += c0 * iphi[HI[i_]] * Fq * HW[i_]
                if rig[lH] != 0.0:
                    for l in range(N + 1):
                        Un[q, l] -= cr * Fq * rig[l]
            uH1 = uH
            uH = pred - k * k * S
            fh[n] = S
        for q in range(nq):
            for j in range(ne):
                l = int(el[j, 1])
                c = c0 * iphi[l] * (COLS[q, j + 1, l] if imp else 1.0)
                eta = U[q, l]
                eta1 = U1[q, l]
                old = Un[q, l]
                if el[j, 0] == 1.0:     # (7.42) spring + cubic spring + damper
                    K = 0.5 * (el[j, 2] + el[j, 3] * eta * eta)
                    D = el[j, 4] / k
                    Un[q, l] = (Un[q, l] + c * (D - K) * eta1) / (1.0 + c * (K + D))
                elif el[j, 0] == 3.0:   # point mass: F = -m (u+ - 2u + u-) / k^2, m in units of a cell's mass
                    mk = el[j, 2] * h / (k * k)
                    Un[q, l] = (Un[q, l] + c * mk * (2.0 * eta - eta1)) / (1.0 + c * mk)
                else:                   # Problem 7.19 rattle, free play eps
                    uR = RS[q, j, 0]
                    uR1 = RS[q, j, 1]
                    e2 = 0.5 * el[j, 7]
                    d = eta - uR
                    s = 0.0
                    m = 0.0
                    if d >= e2:
                        s = 1.0
                        m = d - e2
                    elif d <= -e2:
                        s = -1.0
                        m = -d - e2
                    Fr = 0.0
                    if s != 0.0:
                        # contact stiffness clamped to (0.6/k)^2: make/break errors grow as (k w_contact)^2
                        Gc = min(el[j, 5] * m ** (el[j, 6] - 1.0), 0.36 / (c + k * k * el[j, 8]))
                        d0 = Un[q, l] - (2.0 * uR - uR1)
                        pen = 0.5 * ((s * d0 - e2) + max(s * (eta1 - uR1) - e2, 0.0))
                        Fr = -s * max(Gc * pen / (1.0 + 0.5 * Gc * (c + k * k * el[j, 8])), 0.0)
                    Un[q, l] += c * Fr
                    RS[q, j, 1] = uR
                    RS[q, j, 0] = 2.0 * uR - uR1 - k * k * el[j, 8] * Fr - k * k * el[j, 2]
                if imp and Un[q, l] != old:
                    dl = (Un[q, l] - old) / COLS[q, j + 1, l]
                    Un[q, l] = old
                    for i_ in range(N + 1):
                        Un[q, i_] += dl * COLS[q, j + 1, i_]
        if PX.shape[0] > 0:
            p = min(max(PX[min(n, PX.shape[0] - 1)], 0.0), 1.0) * N
            pi_[0] = min(int(p), N - 1)
            pf[0] = p - pi_[0]
        for c in range(2):
            sm = 0.0
            for q in range(nq):
                sm += (1.0 - pf[c]) * Un[q, pi_[c]] + pf[c] * Un[q, pi_[c] + 1]
            out[n, c] = sm
        tmp = U1
        U1 = U
        U = Un
        Un = tmp
        tmp = Qp
        Qp = Q
        Q = tmp
    ham[5] = uH
    ham[6] = uH1
    return out, en, fh, U, U1


@njit(cache=True)
def _bow_kernel(u, u1, nstep, N, h, k, lam2, mu2, s0, s1, bcl, bcr, phi3, iphi, xb, fb, vb, a, pick):
    """Bowed string/bar (7.32): v_rel + g*phi(v_rel) - b = 0 at a moving, linearly interpolated point,
    phi(v) = sqrt(2a) v exp(-a v^2 + 1/2) (g01 3.6 curve c). g is clamped so the solve is unique (7.34)."""
    un = np.zeros(N + 1)
    qv = np.zeros(N + 1)
    qp = np.zeros(N + 1)
    G = np.zeros(N + 3)
    W = np.zeros(N + 1)
    Bb = np.zeros(N + 1)
    out = np.zeros((nstep, 2))
    _d2(u1, qp, N, bcl, bcr)
    sq = np.sqrt(2.0 * a) * np.exp(0.5)
    gmax = 0.95 * np.e / (2.0 * np.sqrt(2.0 * a))      # 1 + g*min(phi') >= 0.05
    inv = 1.0 / (1.0 + s0)
    pi_ = np.zeros(2, np.int64)
    pf = np.zeros(2)
    for c in range(2):
        p = min(max(pick[c], 0.0), 1.0) * N
        pi_[c] = min(int(p), N - 1)
        pf[c] = p - pi_[c]
    v = 0.0
    for n in range(nstep):
        _lin(u, u1, qv, qp, G, W, Bb, N, lam2, mu2, s0, s1, bcl, bcr, phi3, iphi)
        for l in range(N + 1):
            un[l] = W[l] + Bb[l]
        p = xb[n] * N
        lb = min(max(int(p), 1), N - 2)
        al = min(max(p - lb, 0.0), 1.0)
        Jn = ((1.0 - al) ** 2 + al ** 2) / h
        b = ((1.0 - al) * (un[lb] - u1[lb]) + al * (un[lb + 1] - u1[lb + 1])) / (2.0 * k) - vb[n]
        g = min(0.5 * k * Jn * fb[n] * inv, gmax)
        lo_ = b - g
        hi_ = b + g
        v = min(max(v, lo_), hi_)
        for it in range(60):            # Newton, kept inside the bracket (f is monotone)
            ex = np.exp(-a * v * v)
            fv = v + g * sq * v * ex - b
            if abs(fv) < 1e-13 * (1.0 + abs(b)):
                break
            if fv > 0.0:
                hi_ = v
            else:
                lo_ = v
            vn = v - fv / (1.0 + g * sq * (1.0 - 2.0 * a * v * v) * ex)
            if not (lo_ < vn < hi_):
                vn = 0.5 * (lo_ + hi_)
            v = vn
        f = 2.0 * k * g * sq * v * np.exp(-a * v * v) / (h * Jn)
        un[lb] -= f * (1.0 - al)
        un[lb + 1] -= f * al
        for c in range(2):
            out[n, c] = (1.0 - pf[c]) * un[pi_[c]] + pf[c] * un[pi_[c] + 1]
        tmp = u1
        u1 = u
        u = un
        un = tmp
        tmp = qp
        qp = qv
        qv = tmp
    return out


@njit(cache=True)
def _biquads(s, sos, z):
    for q in range(sos.shape[0]):
        y = sos[q, 0] * s + z[q, 0]
        z[q, 0] = sos[q, 1] * s - sos[q, 4] * y + z[q, 1]
        z[q, 1] = sos[q, 2] * s - sos[q, 5] * y
        s = y
    return s


@njit(cache=True)
def _spring_kernel(x, M, Ki, eta, a, g, taps_d, taps_g, mod, rdc, clip, sos_lo, sos_hi, dlow):
    """Feedback loop of a spring: input + g x (multi-tap, slowly modulated delay line) -> dc blocker ->
    [optional crossover: the high band goes through the chain, the low band through a plain delay] ->
    M stretched allpasses (a + D)/(1 + a D), D = z^-Ki times a first-order Thiran fraction (eta). Returns the
    chain output, sample by sample; each trip round the loop smears the pulse into a longer chirp."""
    n = x.shape[0]
    y = np.zeros(n)
    R = Ki + 2
    wb = np.zeros((M, R))
    v1 = np.zeros(M)
    nd = int(np.max(taps_d) + np.max(np.abs(mod))) + 4
    dl = np.zeros(nd)
    lb = np.zeros(max(dlow, 1) + 1)
    zl = np.zeros((sos_lo.shape[0], 2))
    zh = np.zeros((sos_hi.shape[0], 2))
    dx1 = 0.0
    dy1 = 0.0
    wp = 0
    p = 0
    lp = 0
    for i in range(n):
        fb = 0.0
        for t in range(taps_d.shape[0]):
            pos = wp - taps_d[t] - mod[i]
            ip = int(np.floor(pos))
            fr = pos - ip
            fb += taps_g[t] * ((1.0 - fr) * dl[ip % nd] + fr * dl[(ip + 1) % nd])
        s = x[i] + g * fb
        dy1 = 0.5 * (1.0 + rdc) * (s - dx1) + rdc * dy1
        dx1 = s
        s = dy1
        if clip > 0.0:
            s = min(max(s, -clip), clip)
        low = 0.0
        if sos_lo.shape[0] > 0:
            lo = _biquads(s, sos_lo, zl)
            s = _biquads(s, sos_hi, zh)
            lb[lp] = lo
            low = lb[(lp - dlow) % lb.shape[0]]
            lp = (lp + 1) % lb.shape[0]
        for j in range(M):
            u = wb[j, (p - Ki) % R]
            u1 = wb[j, (p - Ki - 1) % R]
            v = eta * u + u1 - eta * v1[j]
            v1[j] = v
            w = s - a * v
            wb[j, p] = w
            s = a * w + v
        s += low
        wp = (wp + 1) % nd
        dl[wp] = s
        y[i] = s
        p = (p + 1) % R
    return y


# ------------------------------------------------------------------ design
def loss_coeffs(gamma, kappa, fa, Ta, fb, Tb):
    """(sigma0, sigma1) from T60 = Ta at fa Hz and Tb at fb Hz (g02 section 4, eq 7.29; true-60 dB convention)."""
    fb = max(fb, 1.5 * fa)
    Tb = min(Tb, Ta)

    def xi(f):                          # beta^2 at frequency f; rationalised so kappa = 0 and gamma = 0 both work
        w = 2.0 * np.pi * f
        return 2.0 * w * w / (gamma ** 2 + np.sqrt(gamma ** 4 + 4.0 * kappa ** 2 * w * w))
    xa, xb = xi(fa), xi(fb)
    s0 = _LN1000 / (xb - xa) * (xb / Ta - xa / Tb)
    s1 = _LN1000 / (xb - xa) * (1.0 / Tb - 1.0 / Ta)
    if s0 < 0.0:                        # Tb too short for a two-term law: all loss goes to sigma1
        s0, s1 = 0.0, _LN1000 / (Ta * xa)
    return float(s0), float(s1)


def _hmin(gamma, kappa, sig1, k, theta=0.0, r=0.0):
    """Stability bound: lambda^2 (1 - 4 r) + 4 mu^2 (1 - 4 theta) + 4 sigma1 k / h^2 <= 1 (von Neumann); a weight
    of 1/4 or more makes its term unconditionally stable."""
    a = gamma * gamma * k * k * max(1.0 - 4.0 * r, 0.0) + 4.0 * sig1 * k
    return np.sqrt(0.5 * (a + np.sqrt(a * a + 16.0 * kappa * kappa * k * k * max(1.0 - 4.0 * theta, 0.0))))


def ss_mode(p, N, gamma, kappa, sig0, sig1, k, theta=0.0, r=0.0):
    """(frequency Hz, decay rate 1/s) of mode p of the simply-supported scheme (exact: sin modes diagonalise it)."""
    h = 1.0 / N
    s = np.sin(0.5 * p * np.pi * h) ** 2
    s0, s1 = sig0 * k, 2.0 * sig1 * k / h ** 2
    lt = 4.0 * (gamma * k / h) ** 2 * s
    st = 16.0 * (kappa * k / h ** 2) ** 2 * s * s
    den = 1.0 + s0 + theta * st + r * lt
    a1 = (2.0 - (1.0 - 2.0 * r) * lt - (1.0 - 2.0 * theta) * st - 4.0 * s1 * s) / den
    a2 = (1.0 - s0 - 4.0 * s1 * s + theta * st + r * lt) / den
    return np.arccos(np.clip(a1 / (2.0 * np.sqrt(a2)), -1.0, 1.0)) / (2.0 * np.pi * k), -np.log(a2) / (2.0 * k)


def _eig(N, lam2, mu2, bcl, bcr, phi, m=6):
    """Lowest m non-rigid eigenvalues and the largest eigenvalue of M^-1/2 K M^-1/2 for the scheme's K.
    Scheme mode frequency: f = asin(sqrt(eig)/2) / (pi k)   (g01 5.5, eq 6.53)."""
    D1 = np.diff(np.eye(N + 1), axis=0)
    D2 = np.diff(np.eye(N + 1), 2, axis=0)
    p3 = phi ** 3
    K = lam2 * D1.T @ D1 + mu2 * (D2.T @ (p3[1:N, None] * D2))
    if bcl == CL:
        K[1, 1] += 2.0 * mu2 * p3[0]
    if bcr == CL:
        K[N - 1, N - 1] += 2.0 * mu2 * p3[N]
    idx = np.arange(0 if bcl == FR else 1, (N if bcr == FR else N - 1) + 1)
    r = 1.0 / np.sqrt(phi[idx])
    A = K[np.ix_(idx, idx)] * r[:, None] * r[None, :]
    xi1 = np.nan
    if len(idx) <= 300:
        ev, V = np.linalg.eigh(A)
        j = int(np.argmax(ev > 1e-9 * ev[-1]))
        u = np.zeros(N + 1)
        u[idx] = V[:, j] * r
        xi1 = N * N * float(np.sum(np.diff(u) ** 2) / np.sum(phi * u * u))   # beta^2 of the first mode
    else:
        ab = np.zeros((3, len(idx)))
        for j in range(3):
            ab[j, :len(idx) - j] = np.diagonal(A, j)
        ev = eig_banded(ab, lower=True, eigvals_only=True, select="i", select_range=(0, m + 1))
        ev = np.append(ev, 4.0 * lam2 + 16.0 * mu2)
    low = ev[ev > 1e-9 * ev[-1]]
    return low[:m], ev[-1], xi1


@lru_cache(maxsize=1024)
def _design_string(f1, B, sr_i, loss, backoff, bcl, bcr, spread, theta=0.0, r=0.0):
    """gamma, kappa, sigma0, sigma1, N such that the *scheme's* first partial is f1 (grid rule, then gamma is
    corrected for numerical dispersion and stiffness). f_n = n f0 sqrt(1 + B n^2), f1 = partial 1.
    theta, r: implicit weights; the grid is then at most IMPLICIT_GRID times the explicit one."""
    k = 1.0 / sr_i
    g = 2.0 * f1 / np.sqrt(1.0 + B)
    N = 0
    for it in range(10):
        kap = g * np.sqrt(B) / np.pi
        s0, s1 = loss_coeffs(g, kap, *loss) if loss else (0.0, 0.0)
        ge = g * 2.0 ** (spread / 2400.0)
        Nn = int(backoff / max(_hmin(ge, kap, s1, k, theta, r), _hmin(ge, kap, s1, k) / IMPLICIT_GRID))
        N = Nn if it < 3 else min(N, Nn)
        if N < 3:
            raise ValueError("string too short for this sample rate")
        if bcl == SS and bcr == SS:
            f = ss_mode(1, N, g, kap, s0, s1, k, theta, r)[0]
        else:
            ev = _eig(N, (g * k * N) ** 2, (kap * k * N * N) ** 2, bcl, bcr, np.ones(N + 1), 1)[0]
            f = np.arcsin(min(1.0, 0.5 * np.sqrt(ev[0]))) / (np.pi * k)
        if it < 9:
            g *= f1 / f
    return g, kap, s0, s1, N


def arch(x, xs, b, phi_min):
    """Thickness profile of an undercut bar (g02 11.2, Prog. Ex. 7.9): 1 at the shoulders, phi_min in the arch."""
    x = np.minimum(x, 1.0 - x)
    t = np.clip((x - xs) / max(b, 1e-9), 0.0, 1.0)
    return phi_min + (1.0 - phi_min) * 0.5 * (1.0 + np.cos(np.pi * t))


@lru_cache(maxsize=64)
def _arch2(N, bcl, bcr, r1, r2):
    """Arch (shoulder, taper, depth) whose modes 2 and 3 sit at r1 and r2 times mode 1: grid, then simplex."""
    from scipy.optimize import minimize
    x = np.linspace(0.0, 1.0, N + 1)

    def err(p):
        xs, b, pm = p
        if not (0.02 < xs < 0.45 and 0.01 < b < 0.45 and 0.05 < pm <= 1.0):
            return 1e3
        ev = _eig(N, 0.0, 1.0, bcl, bcr, arch(x, xs, b, pm), 3)[0]
        return np.log(np.sqrt(ev[1] / ev[0]) / r1) ** 2 + np.log(np.sqrt(ev[2] / ev[0]) / r2) ** 2
    grid = [(xs, b, pm) for xs in (0.1, 0.15, 0.2) for b in (0.03, 0.08, 0.15, 0.25) for pm in (0.15, 0.25, 0.4, 0.6)]
    p0 = min(grid, key=err)
    r = minimize(err, p0, method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-10, "maxiter": 600})
    if r.fun < 1e-8:
        return arch(x, *r.x)

    def err2(p):
        xs, b, pm, x2, d2 = p
        if not (0.0 <= d2 < 0.8 and 0.2 < x2 < 0.5):
            return 1e3
        ph = arch(x, xs, b, pm) * second_cut(x, x2, d2)
        if not (0.02 < xs < 0.45 and 0.01 < b < 0.45 and 0.05 < pm <= 1.0) or ph.min() < 0.04:
            return 1e3
        ev = _eig(N, 0.0, 1.0, bcl, bcr, ph, 3)[0]
        return np.log(np.sqrt(ev[1] / ev[0]) / r1) ** 2 + np.log(np.sqrt(ev[2] / ev[0]) / r2) ** 2
    best = min(((xs, b, pm, x2, d2) for xs in (0.1, 0.15, 0.2, 0.25) for b in (0.05, 0.1, 0.2)
                for pm in (0.2, 0.3, 0.45, 0.6) for x2 in (0.22, 0.28, 0.34, 0.4, 0.46) for d2 in (0.1, 0.3, 0.5, 0.7)),
               key=err2)
    r2_ = minimize(err2, best, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-12, "maxiter": 2000})
    p = r2_.x if r2_.fun < r.fun else (*r.x, 0.3, 0.0)
    return arch(x, *p[:3]) * second_cut(x, p[3], p[4])


def second_cut(x, x2, depth, width=SECOND_CUT_W):
    """Second, narrow undercut (both sides, symmetric) of relative depth `depth` centred at x2 from the end."""
    x = np.minimum(x, 1.0 - x)
    return 1.0 - depth * np.exp(-((x - x2) / width) ** 2)


@lru_cache(maxsize=1024)
def _design_bar(f1, sr, bcl, bcr, N, loss, ratio, theta=0.0):
    """Bar grid: fixed N, internal rate sr*os chosen so the explicit scheme (7.11) is stable, mu set so the
    scheme's first mode is f1. ratio > 0: search the arch depth so mode2/mode1 = ratio (g02 11.2)."""
    x = np.linspace(0.0, 1.0, N + 1)
    phi = np.ones(N + 1)
    if isinstance(ratio, tuple):
        phi = _arch2(N, bcl, bcr, *ratio)
    elif ratio > 0.0:
        lo_, hi_ = 0.25, 1.0
        for _ in range(24):             # deeper arch -> larger ratio
            mid = 0.5 * (lo_ + hi_)
            ev = _eig(N, 0.0, 1.0, bcl, bcr, arch(x, 0.18, 0.2, mid), 2)[0]
            if np.sqrt(ev[1] / ev[0]) < ratio:
                hi_ = mid
            else:
                lo_ = mid
        phi = arch(x, 0.18, 0.2, 0.5 * (lo_ + hi_))
    ev, emax, xi1 = _eig(N, 0.0, 1.0, bcl, bcr, phi, 6)
    os_ = max(1, int(np.ceil(np.pi * f1 * np.sqrt(emax / ev[0]) / (0.99 * sr))))
    if theta >= 0.25:
        os_ = max(1, int(np.ceil(os_ / IMPLICIT_GRID)))
    while True:
        k = 1.0 / (sr * os_)
        C = np.cos(2.0 * np.pi * f1 * k)
        mu = np.sqrt((1.0 - C) / (ev[0] * (C * theta + 0.5 - theta)))
        kap = mu / (N * N * k)
        # loss law (7.29) assumes beta^2 = omega/kappa; use the first mode's own beta^2 (differs for arched bars)
        s0, s1 = loss_coeffs(0.0, 2.0 * np.pi * f1 / xi1, *loss) if loss else (0.0, 0.0)
        if 0.25 * mu * mu * emax * max(1.0 - 4.0 * theta, 0.0) + 4.0 * s1 * k * N * N <= 0.98:
            break
        os_ += 1
    x_ = mu * mu * ev
    modes = np.arccos(np.clip((1.0 - x_ * (1.0 - 2.0 * theta) / 2.0) / (1.0 + theta * x_), -1.0, 1.0)) / (2.0 * np.pi * k)
    return mu, kap, s0, s1, os_, phi, modes


# ------------------------------------------------------------------ thin classes
class _FD:
    """Shared state and excitation for the grid models. Displacements are in units of the length."""

    def _alloc(self, nq):
        N = self.N
        self.h = 1.0 / N
        self.k = 1.0 / self.sr_i
        self.U = np.zeros((nq, N + 1))
        self.U1 = np.zeros((nq, N + 1))
        self.ham = np.zeros(9)
        self.el = np.zeros((0, 9))
        self.kc = 0.0
        self.force = np.zeros(0)
        self._imp = None
        self.longitudinal(0)

    def _implicit(self):
        """(LU, COLS) of the implicit matrix M = I + (theta mu2 D4 - r lam2 D2) / (1 + s0) for each string, built
        from the kernel's own stencil; COLS[q, j] = M^-1 e_l for the hammer (j = 0) and each element."""
        theta, r = getattr(self, "theta", 0.0), getattr(self, "r", 0.0)
        nq, N = self.U.shape[0], self.N
        pts = [int(self.ham[1])] + [int(e[1]) for e in self.el]
        HI, HW = self._spread()
        key = (theta, r, tuple(pts), tuple(HI), tuple(HW))
        if theta <= 0.0 and r <= 0.0:
            return np.zeros((nq, 1, 1)), np.zeros((nq, len(pts), 1))
        if self._imp is not None and self._imp[0] == key:
            return self._imp[1]
        p3, ip = self.phi ** 3, 1.0 / self.phi
        qv, qp, G, W, Bb = (np.zeros(N + 1), np.zeros(N + 1), np.zeros(N + 3), np.zeros(N + 1), np.zeros(N + 1))
        D4 = np.zeros((N + 1, N + 1))
        D2 = np.zeros((N + 1, N + 1))
        z = np.zeros(N + 1)
        for j in range(N + 1):
            e = np.zeros(N + 1)
            e[j] = 1.0
            _lin(e, z, qv, qp, G, W, Bb, N, 1.0, 1.0, 0.0, 0.0, self.bcl, self.bcr, p3, ip, 0.0, 0.0)
            lo = 0 if self.bcl == FR else 1
            hi = N if self.bcr == FR else N - 1
            col = 2.0 * e - W
            col[:lo] = 0.0
            col[hi + 1:] = 0.0
            D4[:, j] = col
            b = Bb.copy()
            b[:lo] = 0.0
            b[hi + 1:] = 0.0
            D2[:, j] = b
        LU = np.zeros((nq, N + 1, N + 1))
        COLS = np.zeros((nq, len(pts), N + 1))
        for q in range(nq):
            M = np.eye(N + 1) + (theta * self.mu2[q] * D4 - r * self.lam2[q] * D2) / (1.0 + self.s0)
            for j, l in enumerate(pts):
                e = np.zeros(N + 1)
                if j == 0:
                    e[HI] = HW / self.phi[HI]
                else:
                    e[l] = 1.0
                COLS[q, j] = np.linalg.solve(M, e)
            _band_factor(M, 2)
            LU[q] = M
        self._imp = (key, (LU, COLS))
        return LU, COLS

    def longitudinal(self, K=10, tau=0.15):
        """Bidirectional coupling with the first K longitudinal modes (needs kc = ES/T0 > 0); 0 = off.
        Modes at or above 0.45 of the internal rate are left out. The tension term is explicit: large
        amplitudes need a lower backoff (A = 0.005 L at kc 225 runs at backoff 0.7, not 0.9)."""
        cl = self.gamma * math.sqrt(max(self.kc, 0.0)) if hasattr(self, "gamma") else 0.0
        w = np.pi * cl * np.arange(1, int(K) + 1)
        w = w[(w > 0.0) & (w * self.k < 0.9 * np.pi)]
        r = math.exp(-self.k / tau)
        self.LW = w
        self.LC = 2.0 * r * np.cos(w * self.k)
        self.LR2 = np.full(w.size, r * r)
        self.LB = (1.0 - self.LC + r * r) / np.maximum(w * w, 1e-30)
        self.LX = np.zeros((self.U.shape[0], w.size, 2))
        self.LF = np.zeros((self.U.shape[0], w.size))
        return self

    def _idx(self, pos):
        lo = 0 if self.bcl == FR else 1
        hi = self.N if self.bcr == FR else self.N - 1
        l = int(np.clip(round(pos * self.N), lo, hi))
        used = [int(self.ham[1])] * int(self.ham[0] > 0.5) + [int(r[1]) for r in self.el]
        while l in used:                # point elements must not share a grid point (solved one by one)
            l = l + 1 if l < hi else lo
        return l

    def shape(self, pos, halfwidth):
        """Raised cosine (g01 5.2, eq 6.7) of unit height, zero at fixed ends."""
        x = np.linspace(0.0, 1.0, self.N + 1)
        hw = max(halfwidth, 1.5 * self.h)
        c = np.where(np.abs(x - pos) <= hw, 0.5 * (1.0 + np.cos(np.pi * (x - pos) / hw)), 0.0)
        if self.bcl != FR:
            c[0] = 0.0
        if self.bcr != FR:
            c[-1] = 0.0
        return c

    def pluck(self, pos=0.2, halfwidth=0.05, amp=1.0):
        c = amp * self.shape(pos, halfwidth)
        self.U += c
        self.U1 += c
        return self

    def strike(self, pos=0.2, halfwidth=0.05, vel=1.0):
        self.U1 -= self.k * vel * self.shape(pos, halfwidth)
        return self

    def hammer(self, pos=0.12, mass=0.75, w=3000.0, alpha=2.5, v=1.5, conservative=False, loss=0.0, width=0.0):
        """Hammer / mallet of mass ratio `mass` (hammer / string), F = w^(1+alpha) [uH - u]+^alpha.
        conservative: the force is the discrete gradient of the felt potential (energy-conserving, no backoff
        needed), with a felt loss F += loss * d(eta)/dt while in contact.
        width > 0 (fraction of the length): the felt touches the grid over a raised cosine of that width, at the
        exact position (linear interpolation between the two nearest points when narrower than two cells);
        0 = one grid point."""
        self._hpos, self._hwidth = float(pos), float(max(width, 0.0))
        self.ham[:] = 0.0
        self.ham[:] = (1.0, self._idx(pos), mass, w ** (1.0 + alpha), max(alpha, 1.0), 0.0, -self.k * v,
                       1.0 if conservative else 0.0, float(loss))
        return self

    def _spread(self):
        """(grid points, weights summing to 1) of the hammer contact."""
        l = int(self.ham[1])
        w = getattr(self, "_hwidth", 0.0)
        if w <= 0.0:
            return np.array([l], np.int64), np.array([1.0])
        lo = 0 if self.bcl == FR else 1
        hi = self.N if self.bcr == FR else self.N - 1
        x = self._hpos * self.N
        if w * self.N < 2.0:
            i = int(np.clip(math.floor(x), lo, hi - 1))
            a = float(np.clip(x - i, 0.0, 1.0))
            return np.array([i, i + 1], np.int64), np.array([1.0 - a, a])
        g = np.arange(self.N + 1, dtype=float)
        c = np.where(np.abs(g - x) < 0.5 * w * self.N, 1.0 + np.cos(2.0 * np.pi * (g - x) / (w * self.N)), 0.0)
        c[:lo] = 0.0
        c[hi + 1:] = 0.0
        idx = np.nonzero(c)[0]
        return idx.astype(np.int64), c[idx] / c[idx].sum()

    def spring(self, pos, w0=0.0, w1=0.0, sigma=0.0):
        """Lumped spring w0, cubic spring w1 and damper sigma at pos (g02 8.1)."""
        self.el = np.vstack([self.el, [1.0, self._idx(pos), w0 ** 2, w1 ** 4, sigma, 0.0, 0.0, 0.0, 0.0]])
        return self

    def mass(self, pos, ratio=0.1):
        """Point mass fixed on the string (a capo, a bead, a weight), `ratio` = its mass / the string's."""
        self.el = np.vstack([self.el, [3.0, self._idx(pos), float(ratio) * self.N, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        return self

    def end(self, side="right", mass=0.0, spring=0.0, damp=0.0, absorb=0.0):
        """Terminate one end on a mass / spring / damper instead of a rigid support: the end becomes free and the
        lumped elements sit on its grid point. mass = end mass / string mass; spring = Hz, stiffness
        K = M_string (2 pi spring)^2; damp = R / M_string (1/s); absorb 0..1 adds the wave-matched damper
        R = sqrt(T rho) (1 = no reflection of the transverse wave: an absorbing, infinitely long string)."""
        l = 0 if side in ("left", 0) else self.N
        if l == 0:
            self.bcl = FR
        else:
            self.bcr = FR
        self._imp = None
        sig = 0.5 * float(damp) + 0.5 * float(np.clip(absorb, 0.0, 1.0)) * float(getattr(self, "gamma", 0.0))
        if spring > 0.0 or sig > 0.0:
            w0 = 2.0 * math.pi * float(spring)
            self.el = np.vstack([self.el, [1.0, l, w0 * w0, 0.0, sig, 0.0, 0.0, 0.0, 0.0]])
        if mass > 0.0:
            self.el = np.vstack([self.el, [3.0, l, float(mass) * self.N, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        return self

    def rattle(self, pos, w=3000.0, eps=0.05, mass_ratio=1.0, alpha=2.0, offset=0.0, gravity=0.0):
        """Rattle with free play eps (g02 8.2). mass_ratio = string / rattle mass; 0 = a rigid stop at `offset`.
        ``gravity``: weight on a free rattle, u_R'' = -F - g, in string lengths per s^2 (g / L: 9.81 / 0.65 on a
        guitar string)."""
        a = max(alpha, 1.0)
        gr = float(gravity) if mass_ratio > 0 else 0.0
        self.el = np.vstack([self.el, [2.0, self._idx(pos), gr, 0.0, 0.0, w ** (1.0 + a), a, eps, mass_ratio]])
        self._off = getattr(self, "_off", []) + [(len(self.el) - 1, offset)]
        return self

    def _rigid(self):
        """Free-free bar: the blow's net force and torque are taken up by an ideal suspension, i.e. the hammer
        force is applied minus its projection on the two rigid-body modes (which would only drift)."""
        r = np.zeros(self.N + 1)
        if self.bcl == FR and self.bcr == FR and self.ham[0] > 0.5 and not np.isin(self.el[:, 1], (0, self.N)).any():
            x = np.linspace(0.0, 1.0, self.N + 1)
            m = self.h * self.phi
            xc = float(np.sum(m * x) / np.sum(m))
            r = 1.0 / np.sum(m) + (x[int(self.ham[1])] - xc) * (x - xc) / np.sum(m * (x - xc) ** 2)
        return r

    def run(self, n, pickups=(0.3, 0.73), energy=False, scan=None):
        """Advance n internal steps; returns displacement at two pickups (n, 2) [and the discrete energy].
        scan: per-step position (0..1) of a moving readout that replaces the first pickup."""
        nq = self.U.shape[0]
        if getattr(self, "RS", None) is None or self.RS.shape[1] != len(self.el):
            self.RS = np.zeros((nq, len(self.el), 2))
            for j, off in getattr(self, "_off", []):
                self.RS[:, j, :] = off
        out, en, self.force, self.U, self.U1 = _fd_kernel(
            self.U, self.U1, int(n), self.N, self.h, self.k, self.lam2, self.mu2, self.s0, self.s1, self.bcl, self.bcr,
            self.phi ** 3, 1.0 / self.phi, float(self.kc), self.ham, np.ascontiguousarray(self.el), self.RS,
            np.asarray(pickups, dtype=np.float64), self._rigid(), bool(energy), self.LW, self.LC, self.LR2, self.LB,
            self.LX, self.LF, float(getattr(self, "theta", 0.0)), float(getattr(self, "r", 0.0)), *self._implicit(),
            *self._spread(), np.zeros(0) if scan is None else np.ascontiguousarray(scan, dtype=np.float64))
        return (out, en) if energy else out

    def bow(self, force, speed, pos, pickups=(0.3, 0.73), a=100.0):
        """Bow the (first) string with per-step arrays force (F_B), speed (v_B) and contact position."""
        return _bow_kernel(self.U[0].copy(), self.U1[0].copy(), len(force), self.N, self.h, self.k, self.lam2[0],
                           self.mu2[0], self.s0, self.s1, self.bcl, self.bcr, self.phi ** 3, 1.0 / self.phi,
                           np.ascontiguousarray(pos, dtype=np.float64), np.ascontiguousarray(force, dtype=np.float64),
                           np.ascontiguousarray(speed, dtype=np.float64), float(a), np.asarray(pickups, dtype=np.float64))

    def audio(self, y):
        """Internal-rate signal -> output rate."""
        return resample_poly(y, 1, self.os, axis=0) if self.os > 1 else y

    def render(self, dur, pickups=(0.3, 0.73)):
        """Stereo (n, 2): one pickup per channel (g02 1.3)."""
        return self.audio(self.run(int(dur * self.sr_i), pickups))


class StiffString(_FD):
    """Lossy stiff string(s), scheme (7.30a), tuned so partial 1 of the scheme is `f1` Hz.

    B: inharmonicity (f_n = n f0 sqrt(1 + B n^2)); t60 = (f_a, T_a, f_b, T_b) or None (lossless);
    n_strings / detune (cents, total spread, eq. of g02 section 7); bc: 'ss' | 'clamped' per end;
    backoff < 1 lowers the Courant number (needed by tension modulation, Rule of Thumb #4);
    tension = alpha^2 of the Kirchhoff-Carrier term (g02 section 12), 0 = linear."""

    def __init__(self, f1, sr=DEFAULT_SR, B=1e-4, t60=(100.0, 5.0, 2000.0, 3.0), n_strings=1, detune=0.0,
                 bc=("ss", "ss"), backoff=1.0, tension=0.0, oversample=0, theta=0.0, r=0.0):
        self.os = int(oversample) or max(1, int(np.ceil(30.0 * f1 / sr)))
        self.sr_i = sr * self.os
        self.bcl, self.bcr = _BC[bc[0]], _BC[bc[1]]
        self.theta, self.r = float(max(theta, 0.0)), float(max(r, 0.0))
        B = float(np.clip(B, 0.0, 0.05))
        self.gamma, self.kappa, self.sig0, self.sig1, self.N = _design_string(
            float(f1), B, int(self.sr_i), None if t60 is None else tuple(float(v) for v in t60), float(backoff),
            self.bcl, self.bcr, float(abs(detune)) if n_strings > 1 else 0.0, self.theta, self.r)
        self._alloc(n_strings)
        q = np.arange(1, n_strings + 1)
        gq = self.gamma * (2.0 ** ((2 * q - 1 - n_strings) * detune / (2400.0 * max(1, n_strings - 1))))
        self.lam2 = (gq * self.k / self.h) ** 2
        self.mu2 = np.full(n_strings, (self.kappa * self.k / self.h ** 2) ** 2)
        self.s0 = self.sig0 * self.k
        self.s1 = 2.0 * self.sig1 * self.k / self.h ** 2
        self.phi = np.ones(self.N + 1)
        self.kc = float(tension)

    def partial(self, p):
        """Scheme frequency (Hz) and T60 (s) of partial p (simply-supported ends)."""
        f, s = ss_mode(p, self.N, self.gamma, self.kappa, self.sig0, self.sig1, self.k, self.theta, self.r)
        return float(f), float(_LN1000 / s) if s > 0 else np.inf


class Bar(_FD):
    """Ideal (Euler-Bernoulli) bar, explicit scheme (7.11) run at sr*os on a fixed grid of N intervals so the
    upper modes stay close to their true ratios; tuned so mode 1 of the scheme is `f1`.

    bc: 'free' | 'clamped' | 'ss' per end. ratio > 0 cuts an arch (variable thickness, naive scheme 7.72,
    stable here because of the oversampling) so that mode 2 = ratio * mode 1."""

    def __init__(self, f1, sr=DEFAULT_SR, bc=("free", "free"), t60=(440.0, 1.0, 4000.0, 0.2), N=32, ratio=0.0,
                 max_rate=1.5e6, theta=0.0):
        self.bcl, self.bcr = _BC[bc[0]], _BC[bc[1]]
        self.N = int(N)
        self.theta, self.r = float(max(theta, 0.0)), 0.0
        while True:                     # keep the internal rate under max_rate by coarsening the grid
            mu, self.kappa, self.sig0, self.sig1, self.os, phi, self.modes = _design_bar(
                float(f1), int(sr), self.bcl, self.bcr, self.N, None if t60 is None else tuple(float(v) for v in t60),
                tuple(float(v) for v in ratio) if isinstance(ratio, (tuple, list)) else float(ratio), self.theta)
            if self.os * sr <= max_rate or self.N <= 8:
                break
            self.N = max(8, int(self.N * 0.85))
        self.phi = phi.copy()
        self.sr_i = sr * self.os
        self._alloc(1)
        self.lam2 = np.zeros(1)
        self.mu2 = np.array([mu * mu])
        self.s0 = self.sig0 * self.k
        self.s1 = 2.0 * self.sig1 * self.k / self.h ** 2


# ------------------------------------------------------------------ voicing helpers
def _finish(y, sr, dur, vel, level, rel=0.0, lp=5000.0, hp=30.0):
    """DC removal, brightness ceiling, level from velocity, damper after `dur`, short fades."""
    y = F.lowpass(F.highpass(np.asarray(y, dtype=np.float64), hp, sr), min(lp, 0.45 * sr), sr)
    pk = float(np.max(np.abs(y))) if y.size else 0.0
    if pk > 0.0:
        y = y * (level * (0.3 + 0.7 * float(np.clip(vel, 0.0, 1.0))) / pk)
    n = y.shape[0]
    if rel > 0.0:
        off = samples(dur, sr)
        if off < n:
            y[off:] *= np.exp(-np.arange(n - off) / (rel * sr))
    a, r = min(n // 2, 8), min(n // 2, samples(0.01, sr))
    if a > 1:
        y[:a] *= np.linspace(0.0, 1.0, a)
    if r > 1:
        y[-r:] *= np.linspace(1.0, 0.0, r)
    return y


def _vel_out(s, y):
    """Velocity at the pickups (first difference), mixed to mono at the output rate."""
    v = np.diff(y, axis=0, prepend=y[:1]) * s.sr_i
    return s.audio(v).mean(axis=1)


def _disp_out(s, y):
    return s.audio(y).mean(axis=1)


def _key_B(B, freq):
    """Inharmonicity of the note from its value at C4: B ~ f^2, as across a real string set (short treble
    strings are far stiffer). It keeps the stretch of a partial at a given *absolute* frequency constant."""
    return float(np.clip(B * (freq / 261.63) ** 2, 0.0, 2.5e-3))


def _flat(freq, B, cents_per_1e4):
    """Stretched upper partials pull the heard pitch (and genny's autocorrelation pitch test) sharp, as on a
    piano; tune partial 1 flat by the measured bias (see docs/fdstring.md), capped at 12 cents."""
    return freq * 2.0 ** (-min(12.0, cents_per_1e4 * B / 1e-4) / 1200.0)


def _lp(freq, bright):
    """Output ceiling in Hz: about 6 partials of the note, inside 1.6 .. 4.2 kHz, times `bright`."""
    return float(np.clip(bright, 0.25, 4.0)) * float(np.clip(6.0 * freq, 1600.0, 4200.0))


def _pick(pickup):
    p = float(np.clip(pickup, 0.03, 0.97))
    return (p, float(np.clip((p + 0.43) % 1.0, 0.03, 0.97)))


def _t60(freq, lo, hi):
    """T60 = lo at the note, hi at max(4 kHz, 3x the note)."""
    return (float(freq), float(max(lo, 0.02)), float(max(4000.0, 3.0 * freq)), float(np.clip(hi, 0.01, lo)))


def _bow_note(freq, dur, sr, vel, force, speed, pos, travel, B, t60_lo, t60_hi, pickup, tail):
    # measured: the bowed note reads 3..7 cents sharp across the span, so tune 5 cents flat; finer grid for the bow
    s = StiffString(freq * 2.0 ** (-5.0 / 1200.0), sr, B=_key_B(B, freq), t60=_t60(freq, t60_lo, t60_hi),
                    oversample=max(1, int(np.ceil(80.0 * freq / sr))))
    n = int((dur + tail) * s.sr_i)
    t = np.arange(n) / s.sr_i
    env = np.clip(t / 0.04, 0.0, 1.0) * np.clip((dur - t) / 0.03 + 1.0, 0.0, 1.0)
    # F_B is force / string mass; the uniqueness bound (7.34) scales as 1/(k/h) ~ gamma, so scale force with it
    fb = float(np.clip(force, 0.05, 4.0)) * (0.4 + 0.6 * vel) * 0.12 * s.gamma * env
    vb = np.full(n, 0.2 * float(np.clip(speed, 0.1, 4.0)) * (0.5 + 0.5 * vel))
    xb = np.clip(pos + travel * np.clip(t / max(dur, 1e-3), 0.0, 1.0), 0.04, 0.96)
    return s, s.bow(fb, vb, xb, _pick(pickup))


# ------------------------------------------------------------------ instruments
@instrument("fd_string", "Finite-difference stiff string (Bilbao): plucked, struck or bowed, with two-point decay control.",
            family="strings", span=("E1", "C6"),
            excitation=("pluck", "pluck | strike | bow"),
            position=(0.18, "excitation point along the string, 0..1 (0.5 = middle: hollow, odd partials only)"),
            B=(1e-4, "stiffness / inharmonicity at C4, 0 .. 2e-3 (partial n at n f0 sqrt(1 + B n^2)); scaled by (f/C4)^2 across the keyboard"),
            t60_lo=(4.0, "seconds of ring (60 dB) at the note"),
            t60_hi=(0.6, "seconds of ring at max(4 kHz, 3x the note): lower = duller, faster-fading top"),
            pickup=(0.3, "readout point 0..1 (a second one sits 0.43 further along)"),
            bright=(1.0, "0.5 mellow .. 2 open (scales the output ceiling, about 6 partials inside 1.6..4.2 kHz)"),
            bead=(0.0, "mass fixed on the string (a bead, a weight, a capo) as a fraction of the string's mass: "
                       "detunes the partials unevenly, 0 = none"),
            bead_pos=(0.5, "where the bead sits, 0..1"))
def fd_string(freq, dur, sr=DEFAULT_SR, vel=1.0, excitation="pluck", position=0.18, B=1e-4, t60_lo=4.0, t60_hi=0.6,
              pickup=0.3, bright=1.0, bead=0.0, bead_pos=0.5):
    pos = float(np.clip(position, 0.03, 0.97))
    if excitation == "bow":
        s, y = _bow_note(freq, dur, sr, vel, 1.0, 1.0, min(pos, 1.0 - pos), 0.0, B, t60_lo, t60_hi, pickup, 0.3)
        return _finish(_vel_out(s, y), sr, dur, vel, 0.55, 0.08, _lp(freq, bright))
    s = StiffString(_flat(freq, B, 3.0), sr, B=_key_B(B, freq), t60=_t60(freq, t60_lo, t60_hi))
    if bead > 0.0:
        s.mass(float(np.clip(bead_pos, 0.03, 0.97)), float(np.clip(bead, 0.0, 5.0)))
    hw = 0.03 + 0.06 * (1.0 - float(np.clip(vel, 0.0, 1.0)))       # soft touch = wide, dull excitation
    if excitation == "strike":
        s.strike(pos, hw, 1.0)
    elif excitation == "pluck":
        s.pluck(pos, hw, 1.0)
    else:
        raise ValueError("excitation must be pluck, strike or bow")
    y = s.run(int((dur + 0.4) * s.sr_i), _pick(pickup))
    return _finish(_disp_out(s, y), sr, dur, vel, 0.8, 0.07, _lp(freq, bright))


def _piano_key_B(B, freq):
    """Inharmonicity of the piano note from its value at C4, following the Steinway D scale (Chabassier &
    Durufle 2012): smallest at the bass/tenor break, 25x C4 at C7; capped at 2.5e-3 for the FD grid."""
    from .physical.piano_data import steinway_B
    midi = 69.0 + 12.0 * math.log2(max(float(freq), 1.0) / 440.0)
    return float(np.clip(B * steinway_B(midi) / steinway_B(60.0), 0.0, 2.5e-3))


def _stretch_cents(freq):
    """Fundamental of a freshly tuned Steinway D against equal temperament, normalised to A4 (Klouche, DAGA 2014,
    Fig. 5 / Results D): a U-shaped stretch, about +5 cents at A1 and rising from A4 to about +20 cents in the top
    octave; within 4 cents over the middle five octaves."""
    midi = 69.0 + 12.0 * math.log2(max(float(freq), 1.0) / 440.0)
    return float(np.interp(midi, (33.0, 69.0, 93.0), (5.0, 0.0, 20.0)))


def _piano_string(freq, sr, strings, detune, B, t60_lo, t60_hi, hammer_mass, hardness, position, vel, backoff=0.95,
                  theta=0.0, r=0.0, conservative=False, width=0.0):
    f1 = _flat(freq, B, 2.0)
    B = _piano_key_B(B, freq)
    if strings <= 0:
        strings = 1 if freq < 70.0 else 2 if freq < 180.0 else 3
    strings = int(np.clip(strings, 1, 3))
    # backoff < 1: the semi-implicit hammer still rings at lambda = 1 (g02 section 6 warning)
    s = StiffString(f1, sr, B=B, t60=_t60(freq, t60_lo, t60_hi), n_strings=strings, detune=detune,
                    backoff=1.0 if conservative else backoff, theta=theta, r=r)
    kt = (freq / 262.0) ** 0.5          # key-tracking of the Fig 7.11 preset (M 0.75, wH 3000, alpha 2.5 at C4)
    v = 0.5 * 10.0 ** float(np.clip(vel, 0.0, 1.0))                # 0.5 (pp) .. 5 (ff), Fig 7.11
    s.hammer(position, mass=float(np.clip(0.75 * hammer_mass * kt, 0.02, 4.0)) / strings,
             w=3000.0 * kt * float(np.clip(hardness, 0.2, 5.0)), alpha=2.5, v=v, conservative=conservative,
             width=width)
    return s, v


@instrument("fd_piano", "Felt hammer two-way coupled to 1-3 detuned stiff strings (finite difference): velocity changes the tone, not just the level.",
            family="keys", span=("A0", "C7"),
            strings=(0, "strings per note 1..3 (0 = by register)"),
            detune=(3.0, "total spread of the course in cents: beating and a two-stage decay"),
            hammer_mass=(1.0, "hammer mass scale (1 = 0.75 x string mass at C4): heavier = longer contact, darker"),
            hardness=(1.0, "felt stiffness scale 0.2..5: harder = shorter contact, brighter"),
            position=(0.12, "strike point 0..1"),
            B=(3e-4, "inharmonicity at C4; across the keyboard it follows a Steinway D (least at the bass/tenor break)"),
            decay=(6.0, "seconds of ring at the note"),
            t60_hi=(1.0, "seconds of ring at max(4 kHz, 3x the note)"),
            bright=(1.0, "0.5 mellow .. 2 open"),
            stretch=(0.0, "octave stretch of a concert tuning, 0 = equal temperament, 1 = measured (A1 +5, A7 +20 cents)"),
            tuning=("klouche", "curve scaled by `stretch`: klouche (measured Steinway D) | matched (octaves matched on "
                               "the m:n partials of each register, from the string's B) | railsback (Railsback 1938, -17..+40)"),
            restrike=("", "seconds after the onset at which the hammer strikes again the still-moving string, e.g. \"0.032,0.4\""),
            scheme=("explicit", "explicit | theta4 (stiffness weighted theta = 1/12: fourth order, less numerical "
                                "dispersion) | stable (theta = 1/4: stiffness unconditionally stable, finer grid) | "
                                "bensa (r = 1/4 on the tension too)"),
            hammer_model=("semi", "semi (semi-implicit felt law) | conservative (energy-conserving felt, no backoff)"),
            hammer_width=(0.0, "felt contact width as a fraction of the string (0 = one grid point; about 0.016 for a "
                               "real felt at C4): the touch is spread over the grid, softening the top"))
def fd_piano(freq, dur, sr=DEFAULT_SR, vel=1.0, strings=0, detune=3.0, hammer_mass=1.0, hardness=1.0, position=0.12,
             B=3e-4, decay=6.0, t60_hi=1.0, bright=1.0, stretch=0.0, tuning="klouche", restrike="", scheme="explicit",
             hammer_model="semi", hammer_width=0.0):
    freq = freq * 2.0 ** (float(np.clip(stretch, 0.0, 2.0)) * tuning_curve_cents(freq, tuning) / 1200.0)
    lo = decay * float(np.clip((262.0 / freq) ** 0.5, 0.25, 2.0))
    if scheme not in SCHEMES:
        raise ValueError(f"scheme {scheme!r}: {', '.join(SCHEMES)}")
    if hammer_model not in ("semi", "conservative"):
        raise ValueError(f"hammer_model {hammer_model!r}: semi or conservative")
    th, rr = SCHEMES[scheme]
    cons = hammer_model == "conservative"
    s, _ = _piano_string(freq, sr, strings, float(np.clip(detune, 0.0, 30.0)), B, lo, t60_hi, hammer_mass, hardness,
                         float(np.clip(position, 0.04, 0.5)), vel, theta=th, r=rr, conservative=cons,
                         width=float(np.clip(hammer_width, 0.0, 0.2)))
    total = int((dur + 0.45) * s.sr_i)
    times = sorted(float(t) for t in str(restrike).replace(";", ",").split(",") if t.strip()) if restrike != "" else []
    parts, done = [], 0
    for t in times:
        i = min(int(t * s.sr_i), total)
        if i > done:
            parts.append(s.run(i - done, (0.31, 0.77)))
            done = i
            ham = s.ham.copy()
            s.hammer(float(np.clip(position, 0.04, 0.5)), mass=ham[2], w=ham[3] ** (1.0 / (1.0 + ham[4])), alpha=ham[4],
                     v=-ham[6] / s.k if ham[6] else 0.5 * 10.0 ** float(np.clip(vel, 0.0, 1.0)),
                     conservative=ham[7] > 0.5, loss=ham[8], width=s._hwidth)
            s.ham[5] = s.U[0, int(s.ham[1])]
            s.ham[6] = s.ham[5] - s.k * 0.5 * 10.0 ** float(np.clip(vel, 0.0, 1.0))
    parts.append(s.run(total - done, (0.31, 0.77)))
    y = np.concatenate(parts, axis=0)
    return _finish(_vel_out(s, y), sr, dur, vel, 0.7, 0.09, _lp(freq, bright))


RAILSBACK = ((21.0, -17.0), (45.0, -4.0), (69.0, 0.0), (93.0, 12.0), (108.0, 40.0))


def tuning_curve_cents(freq, curve="klouche"):
    """Cents from equal temperament (A4 = 0) of a tuned piano: klouche (measured Steinway D), matched (built from the
    inharmonicity, `strings.matched_tuning_cents`) or railsback (aural tuning curve, -17 cents at A0 to +40 at C8)."""
    midi = 69.0 + 12.0 * math.log2(max(float(freq), 1.0) / 440.0)
    if curve == "klouche":
        return _stretch_cents(freq)
    if curve == "matched":
        from .strings import matched_tuning_cents
        return float(matched_tuning_cents(int(np.clip(round(midi), 21, 108))))
    if curve == "railsback":
        return float(np.interp(midi, [a for a, _ in RAILSBACK], [b for _, b in RAILSBACK]))
    raise ValueError(f"tuning {curve!r}: klouche, matched or railsback")


# the four names are mapped onto the book's element presets (g02 section 8); gap sizes are relative to
# the string's expected excursion (hammer momentum / 2 gamma), since the book gives them for unit-amplitude plucks.
_PREP = {"rattle": 0, "rubber": 1, "bolt": 2, "screw": 3}


@instrument("prepared_piano", "Hammered stiff string with an object on it: loose rattle (buzz), rubber wedge (thud), heavy bolt (clank), screw (gong-like).",
            family="keys", span=("C2", "C6"),
            preparation=("rattle", "rattle (light buzz, sitar-like) | rubber (damped, muted thud) | bolt (heavy rattle: clank, split partials) | screw (stiff spring: inharmonic, gong-like)"),
            position=(0.3, "where the object sits on the string, 0..1"),
            amount=(1.0, "strength of the preparation 0..3 (0 = plain string)"),
            decay=(4.0, "seconds of ring at the note"),
            bright=(1.0, "0.5 mellow .. 2 open"))
def prepared_piano(freq, dur, sr=DEFAULT_SR, vel=1.0, preparation="rattle", position=0.3, amount=1.0, decay=4.0, bright=1.0):
    if preparation not in _PREP:
        raise ValueError("preparation must be rattle, rubber, bolt or screw")
    s, _ = _piano_string(freq, sr, 1, 0.0, 1e-4, decay, 0.8, 1.0, 1.0, 0.12, vel)
    amt = float(np.clip(amount, 0.0, 3.0))
    pos = float(np.clip(position, 0.05, 0.95))
    exc = s.ham[2] * 1.6 / s.gamma      # excursion of a mezzo-forte strike (v = 1.6)
    w = np.pi * s.gamma                 # the string's own angular frequency sets the scale of the elements
    if amt > 0.0:
        if preparation == "rattle":     # after Fig 7.19b: light rattle, small gap (alpha 1, mass ratio 300: measured to stay within 5 cents)
            s.rattle(pos, w=min(20.0 * w, 0.5 * s.sr_i), eps=0.5 * exc / amt, mass_ratio=300.0, alpha=1.0)
        elif preparation == "rubber":   # Fig 7.16b damper, plus a little spring
            s.spring(pos, w0=0.5 * w * amt, sigma=0.012 * w * amt)
        elif preparation == "bolt":     # after Fig 7.19c: heavy rattle (detunes and splits partials by design)
            s.rattle(pos, w=min(10.0 * w, 0.5 * s.sr_i), eps=0.6 * exc / amt, mass_ratio=2.0, alpha=1.0)
        else:                           # Fig 7.15 / 7.16a: linear + cubic spring
            s.spring(pos, w0=1.6 * w * amt, w1=np.sqrt(1.5 * w * amt / max(exc, 1e-12)))
    y = s.run(int((dur + 0.4) * s.sr_i), (0.31, 0.8))
    return _finish(_vel_out(s, y), sr, dur, vel, 0.75, 0.08, _lp(freq, bright))


def _slack(freq, n_sec, sr, glide, vel, position, t60, pickup):
    """Kirchhoff-Carrier string at Courant number 0.75 (Rule of Thumb #4). glide = initial sharpness in semitones."""
    s = StiffString(freq, sr, B=1e-5, t60=_t60(freq, t60, 0.25 * t60), backoff=0.75)
    c = s.shape(position, 0.1)
    ux2 = float(np.sum(np.diff(c) ** 2)) / s.h
    G0 = 1.0 + (2.0 ** (max(glide, 0.0) / 6.0) - 1.0) * vel * vel   # pitch ~ sqrt(G), G - 1 ~ amplitude^2
    s.kc = 2.0 * (G0 - 1.0) / ux2
    s.U += c
    s.U1 += c
    return s, s.run(int(n_sec * s.sr_i), _pick(pickup))


@instrument("slack_string", "Loose string with tension modulation (Kirchhoff-Carrier): a hard pluck starts sharp and glides down, twangy.",
            family="strings", span=("E1", "E5"),
            glide=(0.7, "semitones sharp at the start of a full-velocity pluck (0 = linear string; above 1 the note reads out of tune)"),
            position=(0.8, "pluck point 0..1"),
            decay=(1.2, "seconds of ring"),
            pickup=(0.3, "readout point 0..1"),
            bright=(1.0, "0.5 mellow .. 2 open"))
def slack_string(freq, dur, sr=DEFAULT_SR, vel=1.0, glide=0.7, position=0.8, decay=1.2, pickup=0.3, bright=1.0):
    s, y = _slack(freq, dur + 0.35, sr, float(np.clip(glide, 0.0, 12.0)), float(np.clip(vel, 0.0, 1.0)),
                  float(np.clip(position, 0.1, 0.9)), decay, pickup)
    return _finish(_disp_out(s, y), sr, dur, vel, 0.95, 0.07, _lp(freq, bright))


def sitar_fd(freq, dur, sr=DEFAULT_SR, vel=1.0, buzz=0.45, B=1e-4):
    """Sitar: a dispersive string whose first points rest on the flat bridge, a one-sided obstacle that the string
    strikes and leaves every period (so the vibrating length changes and a pluck at a node no longer silences that
    partial). The stiff string of this module, the bridge a row of rigid one-sided stops over the first 1 % of the
    string, curved as depth (x / l)^2 (jawari), depth = 2e-3 (1 - buzz) string lengths; buzz 0 = no bridge; plucked
    at one fifth. Bridge length, curvature, contact stiffness and the 5 mm pluck are chosen. Measured at D3, buzz .45: 30 % of the
    energy above 8 f0 late in the note (free string 43 %); a 4 % bridge at 2e-4 pinned the string and left .5 %, and a
    coarser grid (oversample 1) has no point inside 1 %. Higher buzz presses harder and takes this share down again."""
    s = StiffString(_flat(freq, B, 3.0), sr, B=_key_B(B, freq), t60=_t60(freq, 8.0, 1.5), oversample=max(4, int(np.ceil(30.0 * freq / sr))))
    b = float(np.clip(buzz, 0.0, 1.0))
    if b > 0.0:
        depth = 2e-3 * (1.0 - b)
        for x in np.arange(s.h, 0.01, s.h):
            s.rattle(x, w=3e4, eps=2.0, mass_ratio=0.0, alpha=1.5, offset=1.0 - depth * (x / 0.01) ** 2)
    s.pluck(0.2, 0.002, 0.005 * (0.4 + 0.6 * float(np.clip(vel, 0.0, 1.0))))
    y = s.run(int((dur + 1.2) * s.sr_i), (0.73, 0.31))
    return _finish(_disp_out(s, y), sr, dur, vel, 0.57, 0.25, 5000.0)   # as loud as the waveguide sitar


@instrument("string_bridge_plate","Plucked string driving a plate through a bridge mass on (nonlinear) springs "
            "(van Walstijn & Mehes DAFx-17): from a string on a soundboard to buzzing, rattling and cymbal-like plates.",
            family="strings", span=("E1", "E4"),
            config=("body", "body (heavy bridge: the string sings, the plate colours) | coupled (light bridge: "
                            "plate-like, short) | plate (cubic springs: cymbal-like) | rattle (loose bridge, gravity: "
                            "snare-like buzz) | flat (one-sided string contact: sitar-like)"))
def string_bridge_plate(freq, dur, sr=DEFAULT_SR, vel=1.0, config="body"):
    from .physical import sbp
    if config not in sbp.PRESETS:
        raise ValueError(f"config must be one of {sorted(sbp.PRESETS)}")
    force = 0.1 * max(float(vel), 0.05)
    # the bridge springs and the plate pull the coupled string off its own pitch (the body preset: 4 semitones
    # sharp, the stiff bridge spring acting almost as a support): tune the string from two 0.5 s trial renders,
    # on the strongest spectral peak within half an octave of the note, then within 6 %
    fs = float(freq)
    for lo, hi in ((2 ** -0.5, 2 ** 0.5), (0.94, 1.06)):
        tr = sbp.render(fs, 0.5, float(sr), config, force=force)[int(0.1 * sr):]
        S = np.abs(np.fft.rfft(tr * np.hanning(tr.size), 1 << 17))
        f = np.fft.rfftfreq(1 << 17, 1.0 / sr)
        band = (f > freq * lo) & (f < freq * hi)
        if not np.any(S[band] > 0.05 * S.max()):
            break                                           # no clear pitch (plate, rattle): leave it
        fs *= freq / f[band][np.argmax(S[band])]
    y = sbp.render(fs, dur + 1.0, float(sr), config, force=force)
    return _finish(y, sr, dur, vel, 0.95, 0.4, 6000.0)


# order-of-magnitude decay times (T60 at the note for A4, T60 at 4 kHz); no table in the book
_MATERIAL = {"wood": (0.7, 0.06), "metal": (4.0, 0.9), "glass": (1.6, 0.35), "plastic": (0.25, 0.03),
             "rosewood": (13.8155 * 2.13e10 / ((2 * np.pi * 440.0) ** 2 * 5.0e3), 13.8155 * 2.13e10 / ((2 * np.pi * 4000.0) ** 2 * 5.0e3))}
_BOUNDARY = {"free": ("free", "free"), "clamped_free": ("clamped", "free"), "clamped": ("clamped", "clamped"),
             "supported": ("ss", "ss")}
_TUNING = {"none": 0.0, "xylophone": (3.0, 6.0), "marimba": (4.0, 10.0)}


def _bar_note(freq, n_sec, sr, vel, material, boundary, mallet, tuning, position, pickup, quality, decay=1.0,
              theta=0.0, scan=0.0):
    if material not in _MATERIAL or boundary not in _BOUNDARY or tuning not in _TUNING:
        raise ValueError("fd_bar: unknown material, boundary or tuning")
    lo, hi = _MATERIAL[material]
    lo *= decay * float(np.clip((440.0 / freq) ** 0.5, 0.3, 3.0))
    N = (20, 32, 48)[int(np.clip(quality, 0, 2))]
    ratio = _TUNING[tuning] if boundary == "free" else 0.0
    s = Bar(freq, sr, _BOUNDARY[boundary], _t60(freq, lo, min(hi, lo)), N, ratio, theta=theta)
    v = 0.5 * 10.0 ** float(np.clip(vel, 0.0, 1.0))
    # contact time 2.5 ms (soft) .. 0.25 ms (hard) at v = 1.5 on a rigid target, shorter when hit harder:
    # Tc ~ 3 eta_max / v, eta_max = ((alpha+1) v^2 / 2)^(1/(alpha+1)) / wH
    al, tc = 2.0, 2.5e-3 * 0.1 ** float(np.clip(mallet, 0.0, 1.0)) * float(np.clip((440.0 / freq) ** 0.4, 0.4, 2.5))
    wH = 3.0 * ((al + 1.0) * 1.5 ** 2 / 2.0) ** (1.0 / (al + 1.0)) / (1.5 * tc)
    s.hammer(position, mass=0.3, w=wH, alpha=al, v=v)   # mallet / bar mass ratio 0.3
    n = int(n_sec * s.sr_i)
    px = None
    if scan > 0.0:
        t = np.arange(n) / s.sr_i
        px = np.clip(pickup + 0.3 * np.sin(2.0 * np.pi * float(scan) * t), 0.03, 0.97)
    y = s.run(n, _pick(pickup), scan=px)
    return s, y


def amplitude_glide(y, sr, cents):
    """Pitch that falls with the amplitude: the time base runs faster by `cents` at the peak of the envelope and
    by cents x (envelope / peak)^2 after (the stiffening of a large vibration grows with its square)."""
    if cents <= 0.0 or y.size < 4:
        return y
    from scipy.signal import lfilter
    a = math.exp(-1.0 / (0.01 * sr))
    env = np.sqrt(lfilter([1.0 - a], [1.0, -a], y * y))
    rate = 2.0 ** (float(cents) * (env / (env.max() + 1e-30)) ** 2 / 1200.0)
    tau = np.concatenate(([0.0], np.cumsum(rate[:-1])))
    return np.interp(tau, np.arange(y.size), y, right=0.0)


@instrument("fd_bar", "Finite-difference bar struck by a mallet: wood/metal/glass, free (marimba, xylophone, glockenspiel) or clamped (tine, ruler).",
            family="mallet", span=("C3", "C7"),
            material=("wood", "wood | metal | glass | plastic | rosewood (E and viscosity of a measured xylophone bar) (sets the ring time)"),
            boundary=("free", "free (partials 1 : 2.76 : 5.40) | clamped_free (1 : 6.27 : 17.5) | clamped | supported (1 : 4 : 9)"),
            mallet=(0.5, "mallet hardness 0 (soft yarn, contact about 2.7 ms at A4) .. 1 (hard, 0.45 ms, clicky)"),
            tuning=("none", "undercut arch for free bars: none | xylophone (2nd partial at 3x) | marimba (2nd and 3rd at 4x and 10x)"),
            position=(0.5, "strike point 0..1"),
            pickup=(0.42, "readout point 0..1"),
            decay=(1.0, "ring-time scale"),
            quality=(1, "grid size 0 (20 points) | 1 (32) | 2 (48): finer = truer upper partials, slower"),
            bright=(1.0, "0.5 mellow .. 2 open"),
            glide=(0.0, "cents the note starts sharp when hit hard (amplitude stiffening), settling as it decays"),
            scan=(0.0, "Hz: the readout sweeps +-0.3 of the bar around `pickup` (a moving microphone; 0 = fixed)"),
            scheme=("explicit", "explicit | implicit (theta = 1/4: unconditionally stable, 4x lower internal rate)"))
def fd_bar(freq, dur, sr=DEFAULT_SR, vel=1.0, material="wood", boundary="free", mallet=0.5, tuning="none", position=0.5,
           pickup=0.42, decay=1.0, quality=1, bright=1.0, glide=0.0, scan=0.0, scheme="explicit"):
    if scheme not in ("explicit", "implicit"):
        raise ValueError("fd_bar scheme must be explicit or implicit")
    ring = max(dur, min(2.5, 1.2 * _MATERIAL.get(material, (1.0,))[0] * decay * float(np.clip((440.0 / freq) ** 0.5, 0.3, 3.0))))
    s, y = _bar_note(freq, ring + 0.05, sr, vel, material, boundary, mallet, tuning, position, pickup, quality, decay,
                     0.25 if scheme == "implicit" else 0.0, float(max(scan, 0.0)))
    out = _vel_out(s, y)
    out = amplitude_glide(out, sr, float(np.clip(glide, 0.0, 200.0)) * float(np.clip(vel, 0.0, 1.0)))
    return _finish(out, sr, ring, vel, 0.9, 0.05, _lp(freq, bright), hp=min(60.0, 0.4 * freq))


@instrument("bowed_string_fd", "Bowed stiff string (finite difference, implicit friction solve) with a bow that can travel along the string.",
            family="strings", span=("G2", "C6"),
            force=(1.0, "bow pressure scale 0.05..4: more = harder, slightly flat, quicker attack"),
            speed=(1.0, "bow speed scale 0.1..4"),
            position=(0.12, "bow point 0..1 from the bridge (small = glassy)"),
            travel=(0.03, "how far the bow point drifts along the string during the note (0 = fixed)"),
            B=(5e-5, "stiffness / inharmonicity"),
            decay=(2.5, "seconds of free ring at the note"),
            pickup=(0.3, "readout point 0..1"),
            bright=(1.0, "0.5 mellow .. 2 open"))
def bowed_string_fd(freq, dur, sr=DEFAULT_SR, vel=1.0, force=1.0, speed=1.0, position=0.12, travel=0.03, B=5e-5,
                    decay=2.5, pickup=0.3, bright=1.0):
    s, y = _bow_note(freq, dur, sr, vel, force, speed, float(np.clip(position, 0.04, 0.5)), travel, B, decay,
                     0.3 * decay, pickup, 0.3)
    return _finish(_vel_out(s, y), sr, dur, vel, 0.55, 0.08, _lp(freq, bright))


@instrument("interceptor_string", "Plucked string with an interceptor: a screw set near the string that it buzzes "
            "against (gap sets how much), or two screws on wooden arms knocking each other every cycle (`tuba`), "
            "with a cloth that softens the contact.",
            family="strings", span=("E1", "C5"),
            kind=("screw", "screw (fixed stop near the string: sitar-like buzz) | tuba (a light arm carrying a screw, "
                           "thrown against its partner every vibration: a brassy, rattling drone)"),
            gap=(0.6, "distance of the stop as a fraction of the pluck amplitude: small = more buzz; under ~0.5 the stop "
                      "locks the low notes off pitch (-80 cents at E1)"),
            cloth=(0.0, "0 bare metal .. 1 cloth over the contact: softer, darker touch"),
            position=(0.85, "where the interceptor sits along the string, 0..1"),
            decay=(4.0, "seconds of ring at the note"))
def interceptor_string(freq, dur, sr=DEFAULT_SR, vel=1.0, kind="screw", gap=0.6, cloth=0.0, position=0.85, decay=4.0):
    if kind not in ("screw", "tuba"):
        raise ValueError("interceptor_string kind must be screw or tuba")
    return _interceptor(freq, dur, sr, vel, kind, gap, cloth, position, decay)


def _interceptor(freq, dur, sr, vel, kind, gap, cloth, position, decay):
    s = StiffString(_flat(freq, 1e-4, 3.0), sr, B=_key_B(1e-4, freq), t60=_t60(freq, decay, 0.8),
                    oversample=max(2, int(np.ceil(30.0 * freq / sr))))
    amp = 0.005 * (0.4 + 0.6 * float(np.clip(vel, 0.0, 1.0)))
    c = float(np.clip(cloth, 0.0, 1.0))
    w = INTERCEPT_W * (INTERCEPT_CLOTH_W / INTERCEPT_W) ** c
    al = 1.5 + 1.5 * c
    x = float(np.clip(position, 0.05, 0.95))
    reach = amp * math.sin(math.pi * x) / max(math.sin(math.pi * 0.2), 1e-3) if x < 0.2 else amp * (1.0 - x) / 0.8
    g = float(np.clip(gap, 0.02, 2.0)) * max(reach, 1e-6)
    if kind == "screw":
        s.rattle(x, w=w, eps=2.0, mass_ratio=0.0, alpha=al, offset=g - 1.0)
    else:
        s.rattle(x, w=w, eps=2.0 * g, mass_ratio=INTERCEPT_ARM, alpha=al, offset=0.0)
    s.pluck(0.2, 0.01, amp)
    y = s.run(int((dur + 1.0) * s.sr_i), (0.73, 0.31))
    return _finish(_disp_out(s, y), sr, dur, vel, 0.6, 0.12, 4500.0 * (1.0 - 0.4 * c))


# ------------------------------------------------------------------ sfx / fx
@sfx("twang", "Twang: 'band' = rubber band / slack string that starts sharp and sags; 'ruler' = ruler flicked on a desk edge, buzzing against it.",
     kind=("band", "band | ruler"), freq=(140.0, "Hz, settled pitch"), dur=(0.9, "seconds"),
     amount=(1.0, "0..2: band = glide depth (4 semitones at 1), ruler = how hard it slaps the desk"),
     slide=(0.0, "ruler: -0.5..0.6 fraction of the free length pulled in over `dur` (> 0 shortens it: the pitch rises as 1/L^2)"))
def twang(sr=DEFAULT_SR, kind="band", freq=140.0, dur=0.9, amount=1.0, slide=0.0):
    freq = float(np.clip(freq, 30.0, 1500.0))
    dur = float(np.clip(dur, 0.1, 6.0))
    amt = float(np.clip(amount, 0.0, 2.0))
    sl = float(np.clip(slide, -0.5, 0.6))
    if kind == "band":
        s, y = _slack(freq, dur, sr, 4.0 * amt, 1.0, 0.7, 0.9 * dur, 0.3)
    elif kind == "ruler":
        s = Bar(freq, sr, ("clamped", "free"), _t60(freq, 0.8 * dur, 0.1 * dur), 24)
        s.pluck(1.0, 0.5, 1.0)
        if amt > 0.0:                   # rigid stop just under the ruler near the clamp: one-sided slap
            s.rattle(0.3, w=2 * np.pi * freq * 30.0, eps=4.0, mass_ratio=0.0, alpha=1.0, offset=2.0 - 0.02 / amt)
        y = s.run(int(dur / (1.0 - sl) * s.sr_i), (0.9, 0.6))
        if sl:
            return _finish(length_slide(_vel_out(s, y), sl, int(round(dur * sr))), sr, dur, 1.0, 0.7, 0.0, 6000.0, hp=25.0)
    else:
        raise ValueError("kind must be band or ruler")
    return _finish(_vel_out(s, y), sr, dur, 1.0, 0.7, 0.0, 6000.0, hp=25.0)


def length_slide(y, slide, n):
    """Read a clamped-free bar's sound as its free length goes L0 (1 - slide u), u = 0..1 over n samples: every
    mode scales as 1/L^2, so the time base runs at 1/(1 - slide u)^2; ``y`` must last n / (1 - slide)."""
    u = np.arange(n) / max(n, 1)
    tau = n * u / (1.0 - slide * u)                    # integral of 1/(1 - s u)^2 du, in samples
    return np.interp(tau, np.arange(len(y)), y, right=0.0)


STEEL_E = 2.0e11
STEEL_RHO = 7850.0

SPRING_PRESETS = {
    "leem1": dict(td=0.056, fc=4300.0, a=0.62, M=100, g=-0.8, gmod=8.0, c1=0.1, g_rip=0.1, g_echo=0.1, f_eq=95.0, b_eq=130.0),
    "leem2": dict(td=0.044, fc=4400.0, a=0.62, M=100, g=-0.8, gmod=12.0, c1=0.1, g_rip=0.1, g_echo=0.1, f_eq=95.0, b_eq=130.0),
    "leem3": dict(td=0.047, fc=4450.0, a=0.62, M=100, g=-0.8, gmod=10.0, c1=0.1, g_rip=0.1, g_echo=0.1, f_eq=95.0, b_eq=130.0),
    "sansui": dict(td=0.056, fc=3526.0, a=0.62, M=100, g=-0.8, gmod=6.0, c1=0.2, g_rip=-0.2, g_echo=0.1, f_eq=95.0, b_eq=130.0),
    "leem_auto": dict(td=0.056, fc=4275.0, a=0.63, M=100, g=-0.64, gmod=8.0, c1=0.1, g_rip=0.1, g_echo=0.1, f_eq=183.0, b_eq=146.0),
}
SPRING_TANKS = {
    "single": None,
    "two_spring": ((1.0e-4, 2.5e-3, 112), (1.0e-4, 2.5e-3, 126)),
    "three_spring": ((1.0e-4, 2.5e-3, 96), (1.0e-4, 2.5e-3, 129), (1.0e-4, 2.5e-3, 166)),
}
SPRING_HF = dict(a=-0.6, M=200, g=-0.77, ratio=2.3, level=0.001)
SPRING_RIPPLE_N = 0.5
SPRING_DC_HZ = 40.0
SPRING_MOD_HZ = 2.0
SPRING_WANDER_A = 0.93


def spring_wire_length(R: float, turns: float, height: float = 0.0) -> float:
    """Unwound wire length of a helix of radius R with `turns` turns over an axial `height`."""
    return math.sqrt((2.0 * math.pi * R * turns) ** 2 + height ** 2)


def spring_geometry(r: float, R: float, turns: float, height: float = 0.0, E: float = STEEL_E, rho: float = STEEL_RHO) -> dict:
    """Echo spacing TD = 4 L R / (r c) and chirp cutoff fC = 3 r c / (16 R^2 sqrt(5 pi)) of a helical spring of wire
    radius r and helix radius R (c = sqrt(E / rho))."""
    c = math.sqrt(E / rho)
    L = spring_wire_length(R, turns, height)
    return {"L": L, "td": 4.0 * L * R / (r * c), "fc": 3.0 * r * c / (16.0 * R * R * math.sqrt(5.0 * math.pi))}


def spring_dispersion(r: float, R: float, E: float = STEEL_E, rho: float = STEEL_RHO, n: int = 20000):
    """Dispersion of a helical spring at zero helix angle (two coupled equations, longitudinal u and transverse v):
    (w^2 - c^2 b^2)(w^2 - g^2 (b^2 - k^2)^2 - c^2 k^2) = c^4 k^2 b^2, c^2 = E/rho, g^2 = E r^2 / (4 rho), k = 1/R.
    Returns (beta rad/m, f Hz, group velocity m/s) on the lower (audio) branch below its maximum fC, and fC."""
    c2 = E / rho
    g2 = E * r * r / (4.0 * rho)
    k2 = 1.0 / (R * R)
    b = np.linspace(1e-6, 1.0, n) * math.sqrt(k2)
    b2 = b * b
    S = c2 * b2 + g2 * (b2 - k2) ** 2 + c2 * k2
    P = c2 * g2 * b2 * (b2 - k2) ** 2
    w = np.sqrt(0.5 * (S - np.sqrt(np.maximum(S * S - 4.0 * P, 0.0))))
    top = int(np.argmax(w))
    vg = np.gradient(w, b)
    return b[:top], w[:top] / (2.0 * np.pi), vg[:top], float(w[top] / (2.0 * np.pi))


def spring_echo_time(f, r: float, R: float, turns: float, height: float = 0.0, E: float = STEEL_E, rho: float = STEEL_RHO):
    """Round-trip time 2 L / v_g(f) of the audio branch, seconds (inf at and above fC)."""
    _, fb, vg, fc = spring_dispersion(r, R, E, rho)
    L = spring_wire_length(R, turns, height)
    f = np.asarray(f, float)
    return np.where(f < fc, 2.0 * L / np.interp(f, fb, np.maximum(vg, 1e-9)), np.inf)


def spring_chain_delay(f, M: int, K: float, a: float, sr: float):
    """Group delay in samples of M stretched allpasses (a + z^-K)/(1 + a z^-K): M K (1 - a^2)/(1 + 2 a cos(w K) + a^2)."""
    w = 2.0 * np.pi * np.asarray(f, float) / sr
    return M * K * (1.0 - a * a) / (1.0 + 2.0 * a * np.cos(w * K) + a * a)


def spring_cost(M_low: int = 100, M_high: int = 200, efficient: bool = False) -> float:
    """Multiplications per output sample of the parametric spring (chain sections cost 3 each at stretch K, 2 at K 1)."""
    if not efficient:
        return 6 * M_low + 1 + 18 + 2 * M_high + 1 + 3
    lf = 6 * (M_low / 2) / 4 + 1 / 4 + 18 + 10 + 2 + 20
    hf = 2 * (M_high / 2) + 1 + 24 + 2
    return lf + hf + 3


def _stretch(K: float):
    """Integer part and first-order Thiran coefficient of a (possibly fractional) stretch K >= 1."""
    if K < 1.5:
        return 0, 0.0
    Ki = max(int(math.floor(K - 0.5)), 1)
    d = K - Ki
    return Ki, (1.0 - d) / (1.0 + d)


def spring_eq(f_peak: float, B: float, K: float, sr: float):
    """Stretched resonator of the first chirp, (A0/2)(1 - z^-2Keq)/(1 + a1 z^-Keq + a2 z^-2Keq), Keq = floor(K)."""
    Keq = max(1, int(math.floor(K)))
    Rr = 1.0 - math.pi * B * Keq / sr
    ct = float(np.clip((1.0 + Rr * Rr) / (2.0 * Rr) * math.cos(2.0 * math.pi * f_peak * Keq / sr), -1.0, 1.0))
    b = np.zeros(2 * Keq + 1)
    a = np.zeros(2 * Keq + 1)
    b[0], b[-1] = (1.0 - Rr * Rr) / 2.0, -(1.0 - Rr * Rr) / 2.0
    a[0], a[Keq], a[-1] = 1.0, -2.0 * Rr * ct, Rr * Rr
    return b, a


def _uniform_wander(n, sr, rng):
    """Uniform noise (variance 1/3) through the one-pole (1 - a)/(1 - a z^-1), a = SPRING_WANDER_A at 44.1 kHz
    (variance about 0.012), bounded to +-1."""
    a = SPRING_WANDER_A ** (44100.0 / sr)
    v = lfilter([1.0 - a], [1.0, -a], rng.uniform(-1.0, 1.0, n))
    return np.clip(v, -1.0, 1.0)


def _slow_noise(n, sr, rng):
    v = rng.standard_normal(n)
    p = math.exp(-2.0 * math.pi * SPRING_MOD_HZ / sr)
    v = lfilter([1.0 - p], [1.0, -p], v)
    return v / (np.max(np.abs(v)) + 1e-12)


def _spring_loop(xin, fs, K, a, M, g, L, taps, gmod, clip, sos_lo, sos_hi, dlow, rng, wander="smooth"):
    Ki, eta = _stretch(K)
    s_abs = sum(abs(t[1]) for t in taps)
    g_eff = float(np.clip(g, -0.999, 0.999))          # taps sum to 1 in amplitude: the loop gain is g, the T60 holds
    lo_d = max(abs(gmod) + 2.0, 2.0)
    td = np.array([max(L + t[0], lo_d) - 1.0 for t in taps])
    tg = np.array([t[1] / s_abs for t in taps])
    noise = _uniform_wander if wander == "uniform" else _slow_noise
    mod = gmod * noise(xin.shape[0], fs, rng) if gmod > 0 else np.zeros(xin.shape[0])
    rdc = 1.0 - 2.0 * math.pi * SPRING_DC_HZ / fs
    xin = np.ascontiguousarray(xin)
    return _spring_kernel(xin, int(M), Ki, eta, a, g_eff, td, tg, mod, rdc, float(clip),
                          sos_lo, sos_hi, int(dlow))


def _spring_one(ch, sr, P, td, fc, M, decay, delay, hf, efficient, clip, seed, wander="smooth"):
    rng = np.random.default_rng(seed)
    a, g = P["a"], P["g"]
    K = sr / (2.0 * fc)
    none = np.zeros((0, 6))
    if efficient:
        aa = cheby1(10, 2.0, 0.9 * sr / 8.0, fs=sr, output="sos")
        xin, fs = sosfilt(aa, ch)[::4], sr / 4.0
        Kc, ac, Mc = K / 2.0, -a, max(1, M // 2)
        lo = butter(4, fc / 2.0, fs=fs, output="sos")
        hi = butter(4, fc / 2.0, "high", fs=fs, output="sos")
        sos_lo, sos_hi = np.vstack([lo, lo]), np.vstack([hi, hi])
        dlow = int(round(float(spring_chain_delay(fc / 2.0, Mc, Kc, ac, fs))))
        tau = float(dlow)
    else:
        xin, fs, Kc, ac, Mc, sos_lo, sos_hi, dlow = ch, float(sr), K, a, M, none, none, 0
        tau = float(spring_chain_delay(0.0, Mc, Kc, ac, fs))
    L = (delay * fs) if delay is not None else max(td * fs - tau, 4.0)
    trip = (L + tau) / fs
    if decay is not None:
        g = -(10.0 ** (-3.0 * trip / max(float(decay), 0.05)))
    taps = ((-L / 5.0, P["g_echo"]), (0.0, 1.0), (2.0 * Kc * SPRING_RIPPLE_N, P["g_rip"]))
    gmod = P["gmod"] * fs / 44100.0
    y = _spring_loop(xin, fs, Kc, ac, Mc, g, L, taps, gmod, clip, sos_lo, sos_hi, dlow, rng, wander)
    if efficient:
        up = np.zeros(ch.shape[0])
        m = min(up[::4].shape[0], y.shape[0])
        up[::4][:m] = 4.0 * y[:m]
        y = sosfilt(aa, up)
    L_full = L * sr / fs
    b, a_eq = spring_eq(P["f_eq"], P["b_eq"], K, sr)
    out = lfilter(b, a_eq, y)
    out = sosfilt(ellip(8, 0.5, 60.0, min(fc, 0.45 * sr), fs=sr, output="sos"), out)
    out *= math.sqrt(float(np.sum(y * y)) / (float(np.sum(out * out)) + 1e-30))
    if hf > 0:
        H = SPRING_HF
        xh = np.ascontiguousarray(ch + P["c1"] * y)
        if efficient:
            lo = butter(2, sr / 4.0, fs=sr, output="sos")
            hi = butter(2, sr / 4.0, "high", fs=sr, output="sos")
            Kh, Mh, sl, sh = 2.0, H["M"] // 2, np.vstack([lo, lo]), np.vstack([hi, hi])
            dl = int(round(float(spring_chain_delay(sr / 4.0, Mh, Kh, H["a"], sr))))
        else:
            Kh, Mh, sl, sh, dl = 1.0, H["M"], none, none, 0
        yh = _spring_loop(xh, float(sr), Kh, H["a"], Mh, H["g"], max(L_full / H["ratio"], 4.0), ((0.0, 1.0),), 0.0,
                          clip, sl, sh, dl, rng)
        out = out + H["level"] * float(hf) * yh
    return out


def spring_fd(x, sr=DEFAULT_SR, r=1.0e-4, R=2.5e-3, turns=112, height=0.0, decay=1.7, E=STEEL_E, rho=STEEL_RHO,
              fmax=5000.0):
    """Helical spring as two coupled PDEs (curved bar, zero helix angle):
    u_tt = c^2 (u_ss - k v_s) - 2 s u_t,  v_tt = -g^2 (v_ssss + 2 k^2 v_ss + k^4 v) + c^2 k (u_s - k v) - 2 s v_t,
    simply supported ends, energy-conserving staggered differences, implicit average-acceleration time step
    (unconditionally stable). Driven by a transverse force near one end, read as transverse velocity near the other."""
    from scipy import sparse
    from scipy.linalg import cholesky_banded, cho_solve_banded
    x = np.asarray(x, float)
    c2, g2, kap = E / rho, E * r * r / (4.0 * rho), 1.0 / R
    L = spring_wire_length(R, turns, height)
    beta = math.sqrt(2.0 * math.pi * fmax / math.sqrt(g2))
    N = int(np.clip(math.ceil(2.0 * beta * L / math.pi), 16, 4000))
    h, k = L / N, 1.0 / sr
    n = N - 1
    Dp = sparse.diags([np.ones(n), -np.ones(n)], [0, -1], shape=(N, n)) / h
    Mu = sparse.diags([np.ones(n), np.ones(n)], [0, -1], shape=(N, n)) * 0.5
    B = sparse.hstack([Dp, -kap * Mu])
    D2 = sparse.diags([np.ones(n - 1), -2.0 * np.ones(n), np.ones(n - 1)], [-1, 0, 1]) / (h * h)
    Lv = D2 + kap * kap * sparse.identity(n)
    A = c2 * (B.T @ B) + sparse.block_diag([sparse.csr_matrix((n, n)), g2 * (Lv.T @ Lv)])
    perm = np.empty(2 * n, int)
    perm[0::2], perm[1::2] = np.arange(n), n + np.arange(n)
    A = A.tocsr()[perm][:, perm]
    sig = 3.0 * math.log(10.0) / max(float(decay), 0.05)
    th = 0.25
    Lhs = (1.0 + sig * k) * sparse.identity(2 * n) + th * k * k * A
    bw = 5
    ab = np.zeros((bw + 1, 2 * n))
    Lc = Lhs.todia()
    for off, row in zip(Lc.offsets, Lc.data):
        if 0 <= off <= bw:
            ab[bw - off, off:] = row[off:]
    cb = cholesky_banded(ab)
    M1 = (2.0 * sparse.identity(2 * n) - (1.0 - 2.0 * th) * k * k * A).tocsr()
    M0 = ((1.0 - sig * k) * sparse.identity(2 * n) + th * k * k * A).tocsr()
    i_in, i_out = 2 * 1 + 1, 2 * (n - 2) + 1
    mono = x if x.ndim == 1 else x.mean(axis=1)
    w0 = np.zeros(2 * n)
    w1 = np.zeros(2 * n)
    y = np.zeros(mono.shape[0])
    for t in range(mono.shape[0]):
        rhs = M1 @ w1 - M0 @ w0
        rhs[i_in] += k * k * mono[t]
        w2 = cho_solve_banded((cb, False), rhs)
        y[t] = (w2[i_out] - w0[i_out]) / (2.0 * k)
        w0, w1 = w1, w2
    y /= (np.max(np.abs(y)) + 1e-30)
    return y * (np.max(np.abs(mono)) + 1e-30)


@effect("spring_reverb", "Spring-tank reverb: dispersive allpass-chain loops for the low chirps and the wide-band chirps, with the tank's pre-echo, ripple and wandering delay (chirping 'boing' echoes); or the helical spring itself solved as a curved bar (`model=fd`, slow).",
        mix=(0.3, "wet level 0..1"),
        preset=("leem1", "leem1 | leem2 | leem3 (the three springs of a Leem Pro KA-1210) | sansui (darker, wobblier) | leem_auto (automatically fitted leem1)"),
        tank=("single", "single | two_spring | three_spring: springs of the given geometry side by side (overrides the preset's delay and cutoff)"),
        decay=(None, "seconds of ring (default from the preset's loop gain: 1.7 s for leem1)"),
        delay=(None, "seconds per trip along the spring, chain delay excluded (default from the preset)"),
        cutoff=(None, "Hz: top of the low chirps (default from the preset, 4300 Hz)"),
        density=(None, "allpass sections 10..200: more = longer chirps (default 100)"),
        hf=(1.0, "level of the wide-band chirp series: 1 = the -60 dB of the measured spring; 0 = off, half the cost"),
        efficient=(False, "true = the low chain at a quarter of the rate (about a third of the cost, a small notch at fC/2)"),
        clip=(0.0, "hard limit inside the loop (0 = off; 0.1..1 = an overdriven tank crashing)"),
        model=("chain", "chain (allpass loops) | fd (wire as two coupled wave equations, slow)"),
        seed=(0, "variation of the delay wander"),
        wander=("smooth", "smooth (slow 2 Hz drift) | uniform (fast random jitter of the delay: uniform noise "
                          "smoothed by a one-pole, the published spring model)"))
def spring_reverb(x, sr=DEFAULT_SR, mix=0.3, preset="leem1", tank="single", decay=None, delay=None, cutoff=None,
                  density=None, hf=1.0, efficient=False, clip=0.0, model="chain", seed=0, wander="smooth"):
    """Parametric spring reverb: low-chirp loop C_lf (dc blocker, M stretched allpasses, three-tap modulated delay,
    elliptic low-pass and first-chirp resonator outside the loop) and wide-band loop C_hf (200 unit allpasses,
    fed by the input and c1 x C_lf). `tank` runs one loop pair per spring, with TD and fC from its geometry."""
    x = np.asarray(x, dtype=np.float64)
    if preset not in SPRING_PRESETS:
        raise ValueError(f"spring_reverb preset must be one of {sorted(SPRING_PRESETS)}")
    if tank not in SPRING_TANKS:
        raise ValueError(f"spring_reverb tank must be one of {sorted(SPRING_TANKS)}")
    P = SPRING_PRESETS[preset]
    M = int(np.clip(P["M"] if density is None else density, 10, 200))
    dl = None if delay is None else float(np.clip(delay, 0.005, 0.5))
    if tank == "single":
        springs = [(P["td"], P["fc"] if cutoff is None else float(cutoff))]
    else:
        springs = [(gm["td"], gm["fc"] if cutoff is None else float(cutoff))
                   for gm in (spring_geometry(*s) for s in SPRING_TANKS[tank])]

    def one(ch):
        ch = np.ascontiguousarray(ch)
        if model == "fd":
            out = 0.0
            for j, s in enumerate(SPRING_TANKS[tank] or ((1.0e-4, 2.5e-3, 112),)):
                out = out + spring_fd(ch, sr, *s, decay=1.7 if decay is None else float(decay))
            return out
        out = 0.0
        for j, (td, fc) in enumerate(springs):
            fc = float(np.clip(fc, 500.0, 0.45 * sr))
            out = out + _spring_one(ch, sr, P, td, fc, M, decay, dl, float(hf), bool(efficient), float(clip), seed + j,
                                    wander)
        return out / math.sqrt(len(springs))

    if model not in ("chain", "fd"):
        raise ValueError("spring_reverb model must be chain or fd")
    if wander not in ("smooth", "uniform"):
        raise ValueError("spring_reverb wander must be smooth or uniform")
    wet = one(x) if x.ndim == 1 else np.stack([one(x[:, 0]), one(x[:, 1])], axis=1)
    m = float(np.clip(mix, 0.0, 1.0))
    return (1.0 - m) * x + m * wet


def spring_fit(ir, sr=DEFAULT_SR, fc=None, td=None):
    """Fit the low-chirp chain (M, a) to a measured spring impulse response: the arrival time of the first chirp per
    frequency bin below fC (peak of a short-time spectrum), then nonlinear least squares on
    tau(f) = L + M K (1 - a^2)/(1 + 2 a cos(2 pi f K / sr) + a^2). Returns dict(M, a, L, K, fc, td)."""
    from scipy.optimize import least_squares
    from scipy.signal import stft
    ir = np.asarray(ir, float)
    if ir.ndim == 2:
        ir = ir.mean(axis=1)
    if td is None:
        ac = np.correlate(ir, ir, "full")[ir.shape[0] - 1:]
        lo = int(0.01 * sr)
        td = (lo + int(np.argmax(np.abs(ac[lo:int(0.2 * sr)])))) / sr       # g < 0: echoes alternate in sign
    f, t, Z = stft(ir[:int(1.2 * td * sr)], sr, nperseg=256, noverlap=224)
    mag = np.abs(Z)
    if fc is None:
        prof = mag.sum(axis=1)
        fc = float(f[int(np.argmax(prof * (f > 1000.0)))])
    K = sr / (2.0 * fc)
    sel = (f > 100.0) & (f < 0.95 * fc)
    ff, tt = f[sel], t[np.argmax(mag[sel], axis=1)] * sr

    def res(p):
        return p[2] + spring_chain_delay(ff, p[0], K, p[1], sr) - tt

    sol = least_squares(res, x0=[100.0, 0.6, float(np.min(tt))], bounds=([1.0, -0.95, 0.0], [400.0, 0.95, 10.0 * sr]))
    return {"M": float(sol.x[0]), "a": float(sol.x[1]), "L": float(sol.x[2]), "K": K, "fc": fc, "td": float(td)}
