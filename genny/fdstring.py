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

from functools import lru_cache

import numpy as np
from numba import njit
from scipy.linalg import eig_banded
from scipy.signal import resample_poly

from . import filters as F
from .core import DEFAULT_SR, samples
from .fx import effect
from .instruments import instrument
from .sfx import sfx

SS, CL, FR = 0, 1, 2            # simply supported, clamped, free
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
def _lin(u, u1, qv, qp, G, W, Bb, N, lam2, mu2, s0, s1, bcl, bcr, phi3, iphi):
    """Linear part of scheme (7.30a), split as u+ = W + g*Bb (Bb is the tension term, g = 1 if linear).

    The 4th difference is D2(phi^3 D2 u) = D2^T D2, symmetric for every end type, so the lossless
    scheme conserves 1/2|d_t- u|^2 + 1/2 <u, K e_t- u> exactly."""
    _d2(u, qv, N, bcl, bcr)
    for l in range(N + 1):
        G[l + 1] = phi3[l] * qv[l]
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
        Bb[l] = iphi[l] * lam2 * d2 * inv


@njit(cache=True)
def _fd_kernel(U, U1, nstep, N, h, k, lam2, mu2, s0, s1, bcl, bcr, phi3, iphi, kc, ham, el, RS, pick, rig, want_e):
    """U, U1: (strings, N+1) displacement at steps n, n-1. Returns (out[n,2], energy, hammer force, U, U1).

    ham = [on, grid index, mass ratio M, wH^(1+alpha), alpha, uH, uH_prev]
    el rows = [kind (1 spring/damper, 2 rattle), grid index, w0^2, w1^4, sigma_P, wR^(1+alpha), alpha, eps, M_rattle]
    RS[string, element] = (uR, uR_prev) rattle positions. kc = alpha^2 of Kirchhoff-Carrier (0 = linear).
    rig: acceleration per unit hammer force that cancels the rigid-body part of the blow (free-free bars)."""
    nq = U.shape[0]
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
    for n in range(nstep):
        E = 0.0
        for q in range(nq):
            u = U[q]
            u1 = U1[q]
            qv = Q[q]
            qp = Qp[q]
            un = Un[q]
            _lin(u, u1, qv, qp, G, W, Bb, N, lam2[q], mu2[q], s0, s1, bcl, bcr, phi3, iphi)
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
            c = c0 * iphi[lH] - cr * rig[lH]
            pred = 2.0 * uH - uH1
            SA = 0.0
            SAe = 0.0
            for q in range(nq):
                eta = uH - U[q, lH]
                Gq = min(wHa * eta ** (alH - 1.0), 1.0 / (k * k + c * Mr)) if eta > 0.0 else 0.0
                A_[q] = 0.5 * Gq / (1.0 + 0.5 * Gq * c * Mr)
                # [eta-]+ here and F >= 0 below: no adhesive pull at make/break. What remains is the energy
                # credited at first detection, M G eta^2/4 ~ (k wH)^2/2 of the hammer energy for alpha = 1
                e_[q] = pred - Un[q, lH] + max(uH1 - U1[q, lH], 0.0)
                SA += A_[q]
                SAe += A_[q] * e_[q]
            S0 = SAe / (1.0 + k * k * SA)
            S = 0.0
            for q in range(nq):
                A_[q] = max(A_[q] * (e_[q] - k * k * S0), 0.0)
                S += A_[q]
            for q in range(nq):
                Fq = Mr * A_[q]
                Un[q, lH] += c0 * iphi[lH] * Fq
                if rig[lH] != 0.0:
                    for l in range(N + 1):
                        Un[q, l] -= cr * Fq * rig[l]
            uH1 = uH
            uH = pred - k * k * S
            fh[n] = S
        for q in range(nq):
            for j in range(ne):
                l = int(el[j, 1])
                c = c0 * iphi[l]
                eta = U[q, l]
                eta1 = U1[q, l]
                if el[j, 0] == 1.0:     # (7.42) spring + cubic spring + damper
                    K = 0.5 * (el[j, 2] + el[j, 3] * eta * eta)
                    D = el[j, 4] / k
                    Un[q, l] = (Un[q, l] + c * (D - K) * eta1) / (1.0 + c * (K + D))
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
                    RS[q, j, 0] = 2.0 * uR - uR1 - k * k * el[j, 8] * Fr
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
def _spring_kernel(x, n_ap, K, a, D, fb, lp):
    """Feedback delay loop holding a chain of stretched allpasses (a + z^-K)/(1 + a z^-K) and a one-pole
    low-pass: each trip round the loop smears the pulse into a longer downward-to-upward chirp."""
    n = x.shape[0]
    y = np.zeros(n)
    xs = np.zeros((n_ap, K))
    ys = np.zeros((n_ap, K))
    dl = np.zeros(D)
    z = 0.0
    p = 0
    d = 0
    for i in range(n):
        s = x[i] + fb * dl[d]
        for j in range(n_ap):
            t = a * s + xs[j, p] - a * ys[j, p]
            xs[j, p] = s
            ys[j, p] = t
            s = t
        z += lp * (s - z)
        dl[d] = z
        y[i] = z
        d = (d + 1) % D
        p = (p + 1) % K
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


def _hmin(gamma, kappa, sig1, k):
    """Stability bound of (7.30a): lambda^2 + 4 mu^2 + 4 sigma1 k / h^2 <= 1 (von Neumann, checked numerically)."""
    a = gamma * gamma * k * k + 4.0 * sig1 * k
    return np.sqrt(0.5 * (a + np.sqrt(a * a + 16.0 * kappa * kappa * k * k)))


def ss_mode(p, N, gamma, kappa, sig0, sig1, k):
    """(frequency Hz, decay rate 1/s) of mode p of the simply-supported scheme (exact: sin modes diagonalise it)."""
    h = 1.0 / N
    s = np.sin(0.5 * p * np.pi * h) ** 2
    s0, s1 = sig0 * k, 2.0 * sig1 * k / h ** 2
    a1 = (2.0 - 4.0 * (gamma * k / h) ** 2 * s - 16.0 * (kappa * k / h ** 2) ** 2 * s * s - 4.0 * s1 * s) / (1.0 + s0)
    a2 = (1.0 - s0 - 4.0 * s1 * s) / (1.0 + s0)
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
def _design_string(f1, B, sr_i, loss, backoff, bcl, bcr, spread):
    """gamma, kappa, sigma0, sigma1, N such that the *scheme's* first partial is f1 (g02 1.4 grid rule,
    then gamma is corrected for numerical dispersion and stiffness). f_n = n f0 sqrt(1 + B n^2), f1 = partial 1."""
    k = 1.0 / sr_i
    g = 2.0 * f1 / np.sqrt(1.0 + B)
    N = 0
    for it in range(10):
        kap = g * np.sqrt(B) / np.pi
        s0, s1 = loss_coeffs(g, kap, *loss) if loss else (0.0, 0.0)
        Nn = int(backoff / _hmin(g * 2.0 ** (spread / 2400.0), kap, s1, k))
        N = Nn if it < 3 else min(N, Nn)
        if N < 3:
            raise ValueError("string too short for this sample rate")
        if bcl == SS and bcr == SS:
            f = ss_mode(1, N, g, kap, s0, s1, k)[0]
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


@lru_cache(maxsize=1024)
def _design_bar(f1, sr, bcl, bcr, N, loss, ratio):
    """Bar grid: fixed N, internal rate sr*os chosen so the explicit scheme (7.11) is stable, mu set so the
    scheme's first mode is f1. ratio > 0: search the arch depth so mode2/mode1 = ratio (g02 11.2)."""
    x = np.linspace(0.0, 1.0, N + 1)
    phi = np.ones(N + 1)
    if ratio > 0.0:
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
    while True:
        k = 1.0 / (sr * os_)
        mu = 2.0 * np.sin(np.pi * f1 * k) / np.sqrt(ev[0])
        kap = mu / (N * N * k)
        # loss law (7.29) assumes beta^2 = omega/kappa; use the first mode's own beta^2 (differs for arched bars)
        s0, s1 = loss_coeffs(0.0, 2.0 * np.pi * f1 / xi1, *loss) if loss else (0.0, 0.0)
        if 0.25 * mu * mu * emax + 4.0 * s1 * k * N * N <= 0.98:
            break
        os_ += 1
    modes = np.arcsin(0.5 * mu * np.sqrt(ev)) / (np.pi * k)
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
        self.ham = np.zeros(7)
        self.el = np.zeros((0, 9))
        self.kc = 0.0
        self.force = np.zeros(0)

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

    def hammer(self, pos=0.12, mass=0.75, w=3000.0, alpha=2.5, v=1.5):
        """Hammer / mallet of mass ratio `mass` (hammer / string), F = w^(1+alpha) [uH - u]+^alpha (g02 section 6)."""
        self.ham[:] = (0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.ham[:] = (1.0, self._idx(pos), mass, w ** (1.0 + alpha), max(alpha, 1.0), 0.0, -self.k * v)
        return self

    def spring(self, pos, w0=0.0, w1=0.0, sigma=0.0):
        """Lumped spring w0, cubic spring w1 and damper sigma at pos (g02 8.1)."""
        self.el = np.vstack([self.el, [1.0, self._idx(pos), w0 ** 2, w1 ** 4, sigma, 0.0, 0.0, 0.0, 0.0]])
        return self

    def rattle(self, pos, w=3000.0, eps=0.05, mass_ratio=1.0, alpha=2.0, offset=0.0):
        """Rattle with free play eps (g02 8.2). mass_ratio = string / rattle mass; 0 = a rigid stop at `offset`."""
        a = max(alpha, 1.0)
        self.el = np.vstack([self.el, [2.0, self._idx(pos), 0.0, 0.0, 0.0, w ** (1.0 + a), a, eps, mass_ratio]])
        self._off = getattr(self, "_off", []) + [(len(self.el) - 1, offset)]
        return self

    def _rigid(self):
        """Free-free bar: the blow's net force and torque are taken up by an ideal suspension, i.e. the hammer
        force is applied minus its projection on the two rigid-body modes (which would only drift)."""
        r = np.zeros(self.N + 1)
        if self.bcl == FR and self.bcr == FR and self.ham[0] > 0.5:
            x = np.linspace(0.0, 1.0, self.N + 1)
            m = self.h * self.phi
            xc = float(np.sum(m * x) / np.sum(m))
            r = 1.0 / np.sum(m) + (x[int(self.ham[1])] - xc) * (x - xc) / np.sum(m * (x - xc) ** 2)
        return r

    def run(self, n, pickups=(0.3, 0.73), energy=False):
        """Advance n internal steps; returns displacement at two pickups (n, 2) [and the discrete energy]."""
        nq = self.U.shape[0]
        if getattr(self, "RS", None) is None or self.RS.shape[1] != len(self.el):
            self.RS = np.zeros((nq, len(self.el), 2))
            for j, off in getattr(self, "_off", []):
                self.RS[:, j, :] = off
        out, en, self.force, self.U, self.U1 = _fd_kernel(
            self.U, self.U1, int(n), self.N, self.h, self.k, self.lam2, self.mu2, self.s0, self.s1, self.bcl, self.bcr,
            self.phi ** 3, 1.0 / self.phi, float(self.kc), self.ham, np.ascontiguousarray(self.el), self.RS,
            np.asarray(pickups, dtype=np.float64), self._rigid(), bool(energy))
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
                 bc=("ss", "ss"), backoff=1.0, tension=0.0, oversample=0):
        self.os = int(oversample) or max(1, int(np.ceil(30.0 * f1 / sr)))
        self.sr_i = sr * self.os
        self.bcl, self.bcr = _BC[bc[0]], _BC[bc[1]]
        B = float(np.clip(B, 0.0, 0.05))
        self.gamma, self.kappa, self.sig0, self.sig1, self.N = _design_string(
            float(f1), B, int(self.sr_i), None if t60 is None else tuple(float(v) for v in t60), float(backoff),
            self.bcl, self.bcr, float(abs(detune)) if n_strings > 1 else 0.0)
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
        f, s = ss_mode(p, self.N, self.gamma, self.kappa, self.sig0, self.sig1, self.k)
        return float(f), float(_LN1000 / s) if s > 0 else np.inf


class Bar(_FD):
    """Ideal (Euler-Bernoulli) bar, explicit scheme (7.11) run at sr*os on a fixed grid of N intervals so the
    upper modes stay close to their true ratios; tuned so mode 1 of the scheme is `f1`.

    bc: 'free' | 'clamped' | 'ss' per end. ratio > 0 cuts an arch (variable thickness, naive scheme 7.72,
    stable here because of the oversampling) so that mode 2 = ratio * mode 1."""

    def __init__(self, f1, sr=DEFAULT_SR, bc=("free", "free"), t60=(440.0, 1.0, 4000.0, 0.2), N=32, ratio=0.0,
                 max_rate=1.5e6):
        self.bcl, self.bcr = _BC[bc[0]], _BC[bc[1]]
        self.N = int(N)
        while True:                     # keep the internal rate under max_rate by coarsening the grid
            mu, self.kappa, self.sig0, self.sig1, self.os, phi, self.modes = _design_bar(
                float(f1), int(sr), self.bcl, self.bcr, self.N, None if t60 is None else tuple(float(v) for v in t60), float(ratio))
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
            bright=(1.0, "0.5 mellow .. 2 open (scales the output ceiling, about 6 partials inside 1.6..4.2 kHz)"))
def fd_string(freq, dur, sr=DEFAULT_SR, vel=1.0, excitation="pluck", position=0.18, B=1e-4, t60_lo=4.0, t60_hi=0.6,
              pickup=0.3, bright=1.0):
    pos = float(np.clip(position, 0.03, 0.97))
    if excitation == "bow":
        s, y = _bow_note(freq, dur, sr, vel, 1.0, 1.0, min(pos, 1.0 - pos), 0.0, B, t60_lo, t60_hi, pickup, 0.3)
        return _finish(_vel_out(s, y), sr, dur, vel, 0.55, 0.08, _lp(freq, bright))
    s = StiffString(_flat(freq, B, 3.0), sr, B=_key_B(B, freq), t60=_t60(freq, t60_lo, t60_hi))
    hw = 0.03 + 0.06 * (1.0 - float(np.clip(vel, 0.0, 1.0)))       # soft touch = wide, dull excitation
    if excitation == "strike":
        s.strike(pos, hw, 1.0)
    elif excitation == "pluck":
        s.pluck(pos, hw, 1.0)
    else:
        raise ValueError("excitation must be pluck, strike or bow")
    y = s.run(int((dur + 0.4) * s.sr_i), _pick(pickup))
    return _finish(_disp_out(s, y), sr, dur, vel, 0.8, 0.07, _lp(freq, bright))


def _piano_string(freq, sr, strings, detune, B, t60_lo, t60_hi, hammer_mass, hardness, position, vel, backoff=0.95):
    f1 = _flat(freq, B, 2.0)
    B = _key_B(B, freq)
    if strings <= 0:
        strings = 1 if freq < 70.0 else 2 if freq < 180.0 else 3
    strings = int(np.clip(strings, 1, 3))
    # backoff < 1: the semi-implicit hammer still rings at lambda = 1 (g02 section 6 warning)
    s = StiffString(f1, sr, B=B, t60=_t60(freq, t60_lo, t60_hi), n_strings=strings, detune=detune, backoff=backoff)
    kt = (freq / 262.0) ** 0.5          # UNSOURCED: key-tracking of the Fig 7.11 preset (M 0.75, wH 3000, alpha 2.5 at C4)
    v = 0.5 * 10.0 ** float(np.clip(vel, 0.0, 1.0))                # 0.5 (pp) .. 5 (ff), Fig 7.11
    s.hammer(position, mass=float(np.clip(0.75 * hammer_mass * kt, 0.02, 4.0)) / strings,
             w=3000.0 * kt * float(np.clip(hardness, 0.2, 5.0)), alpha=2.5, v=v)
    return s, v


@instrument("fd_piano", "Felt hammer two-way coupled to 1-3 detuned stiff strings (finite difference): velocity changes the tone, not just the level.",
            family="keys", span=("A0", "C7"),
            strings=(0, "strings per note 1..3 (0 = by register)"),
            detune=(3.0, "total spread of the course in cents: beating and a two-stage decay"),
            hammer_mass=(1.0, "hammer mass scale (1 = 0.75 x string mass at C4): heavier = longer contact, darker"),
            hardness=(1.0, "felt stiffness scale 0.2..5: harder = shorter contact, brighter"),
            position=(0.12, "strike point 0..1"),
            B=(1e-4, "inharmonicity at C4 (scaled by (f/C4)^2 across the keyboard)"),
            decay=(6.0, "seconds of ring at the note"),
            t60_hi=(1.0, "seconds of ring at max(4 kHz, 3x the note)"),
            bright=(1.0, "0.5 mellow .. 2 open"))
def fd_piano(freq, dur, sr=DEFAULT_SR, vel=1.0, strings=0, detune=3.0, hammer_mass=1.0, hardness=1.0, position=0.12,
             B=1e-4, decay=6.0, t60_hi=1.0, bright=1.0):
    lo = decay * float(np.clip((262.0 / freq) ** 0.5, 0.25, 2.0))
    s, _ = _piano_string(freq, sr, strings, float(np.clip(detune, 0.0, 30.0)), B, lo, t60_hi, hammer_mass, hardness,
                         float(np.clip(position, 0.04, 0.5)), vel)
    y = s.run(int((dur + 0.45) * s.sr_i), (0.31, 0.77))
    return _finish(_vel_out(s, y), sr, dur, vel, 0.7, 0.09, _lp(freq, bright))


# UNSOURCED: the four names are mapped onto the book's element presets (g02 section 8); gap sizes are relative to
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


# UNSOURCED: order-of-magnitude decay times (T60 at the note for A4, T60 at 4 kHz); no table in the book
_MATERIAL = {"wood": (0.7, 0.06), "metal": (4.0, 0.9), "glass": (1.6, 0.35), "plastic": (0.25, 0.03)}
_BOUNDARY = {"free": ("free", "free"), "clamped_free": ("clamped", "free"), "clamped": ("clamped", "clamped"),
             "supported": ("ss", "ss")}
_TUNING = {"none": 0.0, "xylophone": 3.0, "marimba": 4.0}


def _bar_note(freq, n_sec, sr, vel, material, boundary, mallet, tuning, position, pickup, quality, decay=1.0):
    if material not in _MATERIAL or boundary not in _BOUNDARY or tuning not in _TUNING:
        raise ValueError("fd_bar: unknown material, boundary or tuning")
    lo, hi = _MATERIAL[material]
    lo *= decay * float(np.clip((440.0 / freq) ** 0.5, 0.3, 3.0))
    N = (20, 32, 48)[int(np.clip(quality, 0, 2))]
    ratio = _TUNING[tuning] if boundary == "free" else 0.0
    s = Bar(freq, sr, _BOUNDARY[boundary], _t60(freq, lo, min(hi, lo)), N, ratio)
    v = 0.5 * 10.0 ** float(np.clip(vel, 0.0, 1.0))
    # contact time 2.5 ms (soft) .. 0.25 ms (hard) at v = 1.5 on a rigid target, shorter when hit harder:
    # Tc ~ 3 eta_max / v, eta_max = ((alpha+1) v^2 / 2)^(1/(alpha+1)) / wH
    al, tc = 2.0, 2.5e-3 * 0.1 ** float(np.clip(mallet, 0.0, 1.0)) * float(np.clip((440.0 / freq) ** 0.4, 0.4, 2.5))
    wH = 3.0 * ((al + 1.0) * 1.5 ** 2 / 2.0) ** (1.0 / (al + 1.0)) / (1.5 * tc)
    s.hammer(position, mass=0.3, w=wH, alpha=al, v=v)   # UNSOURCED: mallet / bar mass ratio 0.3
    y = s.run(int(n_sec * s.sr_i), _pick(pickup))
    return s, y


@instrument("fd_bar", "Finite-difference bar struck by a mallet: wood/metal/glass, free (marimba, xylophone, glockenspiel) or clamped (tine, ruler).",
            family="mallet", span=("C3", "C7"),
            material=("wood", "wood | metal | glass | plastic (sets the ring time)"),
            boundary=("free", "free (partials 1 : 2.76 : 5.40) | clamped_free (1 : 6.27 : 17.5) | clamped | supported (1 : 4 : 9)"),
            mallet=(0.5, "mallet hardness 0 (soft yarn, contact about 2.7 ms at A4) .. 1 (hard, 0.45 ms, clicky)"),
            tuning=("none", "undercut arch for free bars: none | xylophone (2nd partial at 3x) | marimba (4x)"),
            position=(0.5, "strike point 0..1"),
            pickup=(0.42, "readout point 0..1"),
            decay=(1.0, "ring-time scale"),
            quality=(1, "grid size 0 (20 points) | 1 (32) | 2 (48): finer = truer upper partials, slower"),
            bright=(1.0, "0.5 mellow .. 2 open"))
def fd_bar(freq, dur, sr=DEFAULT_SR, vel=1.0, material="wood", boundary="free", mallet=0.5, tuning="none", position=0.5,
           pickup=0.42, decay=1.0, quality=1, bright=1.0):
    ring = max(dur, min(2.5, 1.2 * _MATERIAL.get(material, (1.0,))[0] * decay * float(np.clip((440.0 / freq) ** 0.5, 0.3, 3.0))))
    s, y = _bar_note(freq, ring + 0.05, sr, vel, material, boundary, mallet, tuning, position, pickup, quality, decay)
    return _finish(_vel_out(s, y), sr, ring, vel, 0.9, 0.05, _lp(freq, bright), hp=min(60.0, 0.4 * freq))


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


# ------------------------------------------------------------------ sfx / fx
@sfx("twang", "Twang: 'band' = rubber band / slack string that starts sharp and sags; 'ruler' = ruler flicked on a desk edge, buzzing against it.",
     kind=("band", "band | ruler"), freq=(140.0, "Hz, settled pitch"), dur=(0.9, "seconds"),
     amount=(1.0, "0..2: band = glide depth (4 semitones at 1), ruler = how hard it slaps the desk"))
def twang(sr=DEFAULT_SR, kind="band", freq=140.0, dur=0.9, amount=1.0):
    freq = float(np.clip(freq, 30.0, 1500.0))
    dur = float(np.clip(dur, 0.1, 6.0))
    amt = float(np.clip(amount, 0.0, 2.0))
    if kind == "band":
        s, y = _slack(freq, dur, sr, 4.0 * amt, 1.0, 0.7, 0.9 * dur, 0.3)
    elif kind == "ruler":
        s = Bar(freq, sr, ("clamped", "free"), _t60(freq, 0.8 * dur, 0.1 * dur), 24)
        s.pluck(1.0, 0.5, 1.0)
        if amt > 0.0:                   # rigid stop just under the ruler near the clamp: one-sided slap
            s.rattle(0.3, w=2 * np.pi * freq * 30.0, eps=4.0, mass_ratio=0.0, alpha=1.0, offset=2.0 - 0.02 / amt)
        y = s.run(int(dur * s.sr_i), (0.9, 0.6))
    else:
        raise ValueError("kind must be band or ruler")
    return _finish(_vel_out(s, y), sr, dur, 1.0, 0.7, 0.0, 6000.0, hp=25.0)


@effect("spring_reverb", "Spring-tank reverb as a dispersive allpass-chain loop (chirping 'boing' echoes); not the helical-spring FD model.",
        mix=(0.3, "wet level 0..1"), decay=(2.0, "seconds of ring"), delay=(0.045, "seconds per trip along the spring"),
        cutoff=(4000.0, "Hz: top of the chirps"), density=(60, "allpass sections 10..200: more = longer chirps"))
def spring_reverb(x, sr=DEFAULT_SR, mix=0.3, decay=2.0, delay=0.045, cutoff=4000.0, density=60):
    """Dispersive allpass-chain spring (the structure of parametric spring reverbs), NOT Bilbao's helical-spring
    scheme (g02 section 10): that equation set is the least reliably extracted part of the source and its
    coupling/boundary terms could not be recovered, so it is not used. Every constant here is UNSOURCED."""
    x = np.asarray(x, dtype=np.float64)
    fc = float(np.clip(cutoff, 500.0, 0.45 * sr))
    K = max(1, int(round(sr / (2.0 * fc))))                    # stretch: dispersion confined below sr / 2K
    M = int(np.clip(density, 10, 200))
    D = max(2, int(float(np.clip(delay, 0.005, 0.5)) * sr))
    trip = D / sr + M * K * 0.3 / sr                           # allpass chain adds about M K (1-a)/(1+a) samples
    fb = -(10.0 ** (-3.0 * trip / max(float(decay), 0.05)))    # UNSOURCED: inverting loop, -60 dB after `decay`
    lp = 1.0 - np.exp(-2.0 * np.pi * fc / sr)

    def one(ch):
        return _spring_kernel(np.ascontiguousarray(ch), M, K, 0.62, D, fb, lp)   # UNSOURCED: allpass coefficient 0.62
    wet = one(x) if x.ndim == 1 else np.stack([one(x[:, 0]), one(x[:, 1])], axis=1)
    m = float(np.clip(mix, 0.0, 1.0))
    return (1.0 - m) * x + m * wet
