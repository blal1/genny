"""2D finite-difference membranes and plates (Bilbao, *Numerical Sound Synthesis*, Wiley 2009, ch.10-13).

Findings file: out/research/g03-bilbao-part3.md (cited below as "g03 §n"). What is here:

* 2D wave equation / stiff membrane with air cavity     g03 §2.3, §2.6, §3.6, listing A.11
* Kirchhoff plate, explicit 13-point scheme, loss        g03 §3.4, §3.6, §4, listing A.12
* orthotropic (wood) plate                               g03 §4.6
* mallet (power-law contact), force pulse, bow           g03 §4.1-4.3, listings A.2, A.3
* Berger and von Karman nonlinear plates                 g03 §6.1, §6.2, sketch §11.2
* plate reverberation, 2D "room"                         g03 §4.4, §2.8

All linear models share one explicit two-step recursion in matrix form (as listing A.12 does):

    u+ = B u - C u-  (+ k^2 J F)/(1 + sigma0 k),
    B = (2 I - k^2 K + 2 sigma1 k L)/(1 + sigma0 k),   C = ((1 - sigma0 k) I + 2 sigma1 k L)/(1 + sigma0 k)

K is the stiffness operator (kappa^2 BIH - gamma^2 LAP), L the Laplacian. Both are assembled from a discrete
strain energy on a rectangular grid (`_operators`), which gives every edge condition from one piece of
code: free edges are the natural condition (edge quadrature weight 1/2, corner 1/4: g03 §1.3), simply
supported / fixed edges delete the edge nodes, clamped edges add Bilbao's `u = dx- u = 0` ghost term. For
supported edges the result is exactly `kappa^2 (Dirichlet LAP)^2`, i.e. scheme 12.14 / listing A.12; this
is checked against Tables 11.1 and 12.1 in tests/test_plates.py.

Stability is not taken from a formula per case: for symmetric K, L the scheme above is stable iff
`rho(k^2 K - 4 sigma1 k L) <= 4 - k^2 * cavity` (energy argument; it reduces to lambda <= 1/sqrt2 (11.13),
mu <= 1/4 (12.15), eq 12.40, eq 11.34 and g03 §3.6's `h^2 >= 4k(sigma1 + sqrt(sigma1^2 + kappa^2))`).
`_model` starts from that closed form and coarsens the grid until the measured spectral radius obeys it.

Cost is controlled the way g03 §8.1 says: lower the internal sample rate (`quality`), never coarsen the grid
at a fixed rate. Output is resampled to the requested rate.
"""
from __future__ import annotations

import math
from functools import lru_cache
from types import SimpleNamespace

import numba
import numpy as np
import scipy.sparse as sp
from scipy.linalg import cholesky_banded
from scipy.signal import resample_poly
from scipy.sparse.linalg import eigsh

from . import filters as F
from .core import DEFAULT_SR, to_mono
from .drums import drum
from .fx import effect
from .physical.fdgrid import airy_dst, dst_factors
from .physics import MATERIALS
from .sfx import sfx

# The book (listings A.8, A.11, A.12; g03 §3.6) sets sigma = 6 ln10 / T60. With the loss term -2 sigma u_t
# the amplitude goes as exp(-sigma t), so that value decays 120 dB in "T60": measured on the plate scheme,
# a requested 4 s came out as 1.9 s. 3 ln10 / T60 makes the requested T60 the real 60 dB time.
_L10 = 3.0 * math.log(10.0)
RHO_AIR, C_AIR = 1.2, 343.0          # kg/m^3, m/s: air at room temperature (standard values)
MYLAR_EH = 4.0e9 * 0.19e-3            # N/m: in-plane stiffness E h of a Mylar head
HEAD_DENSITY = 0.26                  # kg/m^2: timpani head, 0.262 (Christian et al. 1984, sec. III)
BRONZE = (8800.0, 1.24e11, 0.33)     # rho, E, nu: E and nu of cymbal bronze (Chadwick et al. 2009, Table 1); rho of bell bronze
                                     # C91300 (Cu 81 Sn 19), 0.318 lb/in3 (Wieland Concast C91300 datasheet, CDA physical data)
WOOD_K = (36.5, 8.41, 18.1, 1.248)   # kx, ky, kxy, aspect: wooden plate of Table 12.2 (g03 §4.6)
SNARE_K, SNARE_ALPHA = 1.0e8, 1.3    # N/m^alpha, collision stiffness and exponent of the wires on the head
SNARE_N, SNARE_MASS = 8, 1.5e-3      # contact points, kg of wire per point
SNARE_F, SNARE_T60 = 250.0, 0.1      # Hz, s: wire spring and loss


# ---------------------------------------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------------------------------------
def _d2(N):
    """Second difference at the N-1 interior nodes of an (N+1)-node line."""
    return sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(N - 1, N + 1))


def _d1(N):
    """Forward difference on the N cells of an (N+1)-node line."""
    return sp.diags([-1.0, 1.0], [0, 1], shape=(N, N + 1))


def _trap(N):
    w = np.ones(N + 1)
    w[0] = w[-1] = 0.5
    return w


def _operators(Nx, Ny, hx, hy, bc, cxx, cyy, cn, cc, nu, gamma2, circle=False):
    """Stiffness operator K, Laplacian L, quadrature weights w and node map idx.

    Energy per unit area: cxx uxx^2 + cyy uyy^2 + 2 cn uxx uyy (grid nodes) + cc uxy^2 (cells, discrete
    bracket form of g03 §6.2) + gamma2 |grad u|^2. Isotropic plate: cxx = cyy = kappa^2, cn = nu kappa^2,
    cc = 2(1-nu) kappa^2. `bc` = 4 letters for the edges x=0, x=max, y=0, y=max: f(ree), s(upported/fixed),
    c(lamped). On a free edge the normal curvature is eliminated (uxx = -nu uyy, eq 12.11), which leaves
    (1-nu^2) utt^2 there - the same (1-nu^2) that appears in the book's free-edge row (Problem 12.6).
    Returns K, L with `u_tt = -K u`, both already divided by the mass weights.
    """
    wx, wy = _trap(Nx), _trap(Ny)
    ix, iy = sp.identity(Nx + 1), sp.identity(Ny + 1)
    n = (Nx + 1) * (Ny + 1)
    cell = hx * hy
    E = sp.csr_matrix((n, n))
    if cxx or cyy or cc:
        edge = lambda w: np.where(w < 1.0, 0.5 * (1.0 - nu * nu), 1.0)
        Ax = sp.kron(_d2(Nx), iy) / hx ** 2
        Ay = sp.kron(ix, _d2(Ny)) / hy ** 2
        Axy = sp.kron(_d1(Nx), _d1(Ny)) / cell
        Axi = sp.kron(_d2(Nx), sp.eye(Ny - 1, Ny + 1, k=1)) / hx ** 2
        Ayi = sp.kron(sp.eye(Nx - 1, Nx + 1, k=1), _d2(Ny)) / hy ** 2
        E = (cxx * Ax.T @ sp.diags(np.kron(np.ones(Nx - 1), edge(wy))) @ Ax
             + cyy * Ay.T @ sp.diags(np.kron(edge(wx), np.ones(Ny - 1))) @ Ay
             + cn * (Axi.T @ Ayi + Ayi.T @ Axi) + cc * Axy.T @ Axy)
        clamp = np.zeros((Nx + 1, Ny + 1))            # clamped: u = dx- u = 0 (g03 §3.4), ghost value 0
        if bc[0] == "c":
            clamp[1, :] += cxx * wy / hx ** 4
        if bc[1] == "c":
            clamp[Nx - 1, :] += cxx * wy / hx ** 4
        if bc[2] == "c":
            clamp[:, 1] += cyy * wx / hy ** 4
        if bc[3] == "c":
            clamp[:, Ny - 1] += cyy * wx / hy ** 4
        E = (E + sp.diags(clamp.ravel())) * cell
    Gx = sp.kron(_d1(Nx), iy) / hx
    Gy = sp.kron(ix, _d1(Ny)) / hy
    G = cell * (Gx.T @ sp.diags(np.kron(np.ones(Nx), wy)) @ Gx + Gy.T @ sp.diags(np.kron(wx, np.ones(Ny))) @ Gy)

    keep = np.ones((Nx + 1, Ny + 1), dtype=bool)
    if bc[0] != "f":
        keep[0, :] = False
    if bc[1] != "f":
        keep[Nx, :] = False
    if bc[2] != "f":
        keep[:, 0] = False
    if bc[3] != "f":
        keep[:, Ny] = False
    if circle:                                        # staircase mask, fixed rim (g03 §2.7 rule)
        ll, mm = np.meshgrid(np.arange(Nx + 1) / Nx - 0.5, np.arange(Ny + 1) / Ny - 0.5, indexing="ij")
        keep &= np.hypot(ll, mm) < 0.5 - 1e-9
    sel = np.nonzero(keep.ravel())[0]
    idx = np.full(n, -1, dtype=np.int64)
    idx[sel] = np.arange(sel.size)
    w = (cell * np.kron(wx, wy))[sel]
    winv = sp.diags(1.0 / w)
    K = (winv @ (E + gamma2 * G).tocsr()[sel][:, sel]).tocsr()
    L = (-(winv @ G.tocsr()[sel][:, sel])).tocsr()
    return K, L, w, idx.reshape(Nx + 1, Ny + 1)


def _rho(M, w):
    """Largest eigenvalue of W^-1-scaled operator M (symmetric in the w inner product)."""
    s = np.sqrt(w)
    S = sp.diags(s) @ M @ sp.diags(1.0 / s)
    S = (S + S.T) * 0.5
    if S.shape[0] < 60:
        return float(np.linalg.eigvalsh(S.toarray())[-1])
    v0 = np.random.default_rng(0).standard_normal(S.shape[0])
    return float(eigsh(S, k=1, which="LA", v0=v0, tol=1e-5, return_eigenvectors=False)[0])


def _csr(M):
    M = M.tocsr()
    M.sort_indices()
    return M.data.astype(np.float64), M.indices.astype(np.int64), M.indptr.astype(np.int64)


@lru_cache(maxsize=32)
def _model(fs, eps=1.0, bc="ssss", kappa=0.0, gamma=0.0, sig0=0.0, sig1=0.0, nu=0.3, ortho=None,
           circle=False, cav=0.0, wall=None):
    """Grid + update matrices for `u_tt = -kappa^2 BIH u + gamma^2 LAP u - cav*mean(u) - 2 sig0 u_t
    + 2 sig1 LAP u_t` on a unit-area rectangle of aspect `eps` (sides sqrt(eps) x 1/sqrt(eps)).
    `ortho` = (kx, ky, kxy) replaces kappa (g03 §4.6; supported/clamped edges only).
    `wall` = specific impedances zeta (Z / rho c, 0 = rigid) of the four free edges: u_n = -u_t / (gamma zeta)."""
    k = 1.0 / fs
    if ortho:
        kx, ky, kxy = ortho
        cxx, cyy, cn, cc, r = kx * kx, ky * ky, 0.0, kxy * kxy, math.sqrt(ky / kx)     # r = hy/hx, eq 12.41
    else:
        kx = ky = kappa
        cxx = cyy = kappa * kappa
        cn, cc, r = nu * cxx, 2.0 * (1.0 - nu) * cxx, 1.0
    cav = min(cav, 2.0 / (k * k))                              # eq 11.34 needs k^2 gamma^2 d^2 < 4
    lim = 0.995 * (4.0 - k * k * cav)
    # closed-form von Neumann bound (eqs 11.13, 12.15, 12.40 and g03 §3.6 in one quadratic in hx^2)
    b_ = (4.0 * gamma * gamma * k * k + 16.0 * sig1 * k) * (1.0 + 1.0 / r ** 2)
    c_ = 16.0 * k * k * (cxx + cyy / r ** 4 + (cc + 2.0 * cn) / (r * r))
    hx = math.sqrt((b_ + math.sqrt(b_ * b_ + 4.0 * lim * c_)) / (2.0 * lim))
    while True:
        Nx, Ny = int(math.sqrt(eps) / hx), int(1.0 / (math.sqrt(eps) * hx * r))
        if min(Nx, Ny) < 5:                                    # too stiff for this rate: clamp the stiffness
            sc = 0.8
            return _model(fs, eps, bc, kappa * sc, gamma * sc, sig0, sig1 * sc, nu,
                          tuple(v * sc for v in ortho) if ortho else None, circle, cav, wall)
        hxx, hyy = math.sqrt(eps) / Nx, 1.0 / (math.sqrt(eps) * Ny)
        K, L, w, idx = _operators(Nx, Ny, hxx, hyy, bc, cxx, cyy, cn, cc, nu, gamma * gamma, circle)
        rho = _rho(k * k * K - 4.0 * sig1 * k * L, w)
        if rho <= lim:
            break
        hx *= max(1.01, (rho / lim) ** 0.25)
    a = 1.0 / (1.0 + sig0 * k)
    I = sp.identity(K.shape[0])
    B = a * (2.0 * I - k * k * K + 2.0 * sig1 * k * L)
    C = a * ((1.0 - sig0 * k) * I + 2.0 * sig1 * k * L)
    if wall and any(wall):
        B, C = _wall_update(K, L, idx, Nx, Ny, hxx, hyy, gamma, wall, k, sig0, sig1)
    return SimpleNamespace(fs=fs, k=k, eps=eps, Nx=Nx, Ny=Ny, hx=hxx, hy=hyy, n=K.shape[0], K=K, L=L, w=w,
                           idx=idx, a=a, B=_csr(B), C=_csr(C), Lc=_csr(L), kappa=kappa, gamma=gamma,
                           sig0=sig0, sig1=sig1, cav=cav, rho=rho)


def loss_coeffs(f1, t1, f2, t2, kappa=0.0, gamma=0.0, fs=0.0):
    """sigma0, sigma1 from two (Hz, T60) pairs: eq 7.29 as coded in listing A.8 (g03 §9), sigma(w) = sigma0 +
    sigma1*zeta(w) with zeta = squared wavenumber. For the pure plate and the pure membrane zeta is taken
    from the *scheme's* dispersion relation at rate `fs` (zeta = (2/(k kappa)) sin(wk/2) resp.
    (4/(k gamma)^2) sin^2(wk/2)), so the requested T60 is met in spite of numerical dispersion."""
    k = 1.0 / fs if fs else 0.0

    def zeta(f):
        w = 2.0 * math.pi * f
        s = math.sin(min(w * k / 2.0, math.pi / 2.0)) if k else 0.0
        if kappa > 0.0 and gamma == 0.0:
            return 2.0 * s / (k * kappa) if k else w / kappa
        if kappa > 0.0:
            return (-gamma ** 2 + math.sqrt(gamma ** 4 + 4.0 * kappa ** 2 * w * w)) / (2.0 * kappa ** 2)
        return 4.0 * s * s / (k * gamma) ** 2 if k else (w / gamma) ** 2

    z1, z2 = zeta(f1), zeta(f2)
    s1 = max(0.0, _L10 * (1.0 / t2 - 1.0 / t1) / (z2 - z1)) if z2 > z1 else 0.0
    s0 = _L10 / t1 - s1 * z1
    if s0 < 0.0:                                               # the line would go negative at DC
        s0, s1 = 0.0, _L10 / (t2 * z2)
    return s0, s1


def plate_modes(kappa, eps=1.0, n=6, ortho=None):
    """Analytic simply supported modes, Hz, sorted: eq 12.13 `f = (pi kappa/2)(p^2/eps + eps q^2)`; with
    `ortho=(kx,ky,kxy)` the orthotropic dispersion relation of g03 §4.6."""
    p, q = np.meshgrid(np.arange(1, 4 * n), np.arange(1, 4 * n), indexing="ij")
    bx2, by2 = (p * math.pi) ** 2 / eps, (q * math.pi) ** 2 * eps
    if ortho:
        kx, ky, kxy = ortho
        w = np.sqrt(kx ** 2 * bx2 ** 2 + ky ** 2 * by2 ** 2 + kxy ** 2 * bx2 * by2)
    else:
        w = kappa * (bx2 + by2)
    return np.sort(w.ravel())[:n] / (2.0 * math.pi)


def membrane_modes(gamma, eps=1.0, n=6):
    """Analytic fixed rectangular membrane modes, Hz: eq 11.8 `f = (gamma/2) sqrt(p^2/eps + eps q^2)`."""
    p, q = np.meshgrid(np.arange(1, 4 * n), np.arange(1, 4 * n), indexing="ij")
    return np.sort((0.5 * gamma * np.sqrt(p * p / eps + eps * q * q)).ravel())[:n]


def scheme_modes(mdl, n=6):
    """Frequencies (Hz) the lossless scheme actually produces: eigenvalues w^2 of K mapped through
    cos(w_d k) = 1 - k^2 w^2 / 2."""
    s = np.sqrt(mdl.w)
    S = (sp.diags(s) @ mdl.K @ sp.diags(1.0 / s)).toarray()
    lam = np.clip(np.linalg.eigvalsh((S + S.T) * 0.5), 0.0, None)[:n]
    return np.arccos(np.clip(1.0 - 0.5 * mdl.k ** 2 * lam, -1.0, 1.0)) / (2.0 * math.pi * mdl.k)


def _spread(mdl, x, y, width=0.0):
    """Excitation distribution e (indices, weights) with integral 1 (spreading operator J, g03 §1.2): a 2D
    raised cosine of full width `width` (eq 11.3, scaled units) or, if narrower than the grid, bilinear.
    x, y are fractions 0..1 of the two sides."""
    X, Y = x * mdl.Nx, y * mdl.Ny
    ll, mm = np.nonzero(mdl.idx >= 0)
    d = np.hypot((ll - X) * mdl.hx, (mm - Y) * mdl.hy)
    r = 0.5 * width
    sel = d < r
    if sel.sum() >= 4:
        ji, e = mdl.idx[ll[sel], mm[sel]], 0.5 * (1.0 + np.cos(math.pi * d[sel] / r))
    else:
        l0 = int(np.clip(math.floor(X), 0, mdl.Nx - 1))
        m0 = int(np.clip(math.floor(Y), 0, mdl.Ny - 1))
        ax, ay = float(np.clip(X - l0, 0, 1)), float(np.clip(Y - m0, 0, 1))
        cand = [(l0, m0, (1 - ax) * (1 - ay)), (l0, m0 + 1, (1 - ax) * ay),
                (l0 + 1, m0, ax * (1 - ay)), (l0 + 1, m0 + 1, ax * ay)]
        cand = [(mdl.idx[l, m], c) for l, m, c in cand if mdl.idx[l, m] >= 0 and c > 1e-9]
        if not cand:                                           # on a fixed edge: nearest live node
            j = int(np.argmin(d))
            cand = [(mdl.idx[ll[j], mm[j]], 1.0)]
        ji, e = np.array([c[0] for c in cand]), np.array([c[1] for c in cand])
    ji = ji.astype(np.int64)
    return ji, e / np.sum(mdl.w[ji] * e)


# ---------------------------------------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------------------------------------
@numba.njit(cache=True)
def _lin(Bd, Bi, Bp, Cd, Ci, Cp, u, u1, o):
    for r in range(o.shape[0]):
        s = 0.0
        for j in range(Bp[r], Bp[r + 1]):
            s += Bd[j] * u[Bi[j]]
        for j in range(Cp[r], Cp[r + 1]):
            s -= Cd[j] * u1[Ci[j]]
        o[r] = s


@numba.njit(cache=True)
def _read(u, idx, X, Y):
    """Bilinear interpolant I1 (g03 §1.2); nodes on a fixed edge read zero."""
    Nx, Ny = idx.shape[0] - 1, idx.shape[1] - 1
    X = min(max(X, 0.0), Nx - 1e-9)
    Y = min(max(Y, 0.0), Ny - 1e-9)
    l, m = int(X), int(Y)
    ax, ay = X - l, Y - m
    s = 0.0
    i = idx[l, m]
    if i >= 0:
        s += (1 - ax) * (1 - ay) * u[i]
    i = idx[l, m + 1]
    if i >= 0:
        s += (1 - ax) * ay * u[i]
    i = idx[l + 1, m]
    if i >= 0:
        s += ax * (1 - ay) * u[i]
    i = idx[l + 1, m + 1]
    if i >= 0:
        s += ax * ay * u[i]
    return s


@numba.njit(cache=True)
def _run(Bd, Bi, Bp, Cd, Ci, Cp, u, u1, w, ji, jw, force, fgain, cav, mal, bow, k, idx, pk, out):
    """Linear scheme. u, u1 = state at n, n-1 (updated in place). Excitation, one of:
    force[n] spread by (ji, jw); mallet `mal` = [M, wH, a, uH, uH_prev, kc_max] (power-law contact, explicit
    as listing A.2); bow `bow` = [FB, vB, sig, v_last] with force[n] as the bow-force envelope (Newton
    solve as listing A.3). pk rows = [X, Y, rx, ry, dphi]: pickup on a circle (moving pickup, g03 §2.9)."""
    n = u.shape[0]
    t = np.empty(n)
    A = math.sqrt(2.0 * bow[2]) * math.exp(0.5)
    for s in range(out.shape[0]):
        _lin(Bd, Bi, Bp, Cd, Ci, Cp, u, u1, t)
        if cav != 0.0:                                         # air cavity: spring on the mean (g03 §2.6)
            m = 0.0
            for i in range(n):
                m += w[i] * u[i]
            m *= cav
            for i in range(n):
                t[i] -= m
        if mal[0] > 0.0:
            d = mal[3]
            for j in range(ji.shape[0]):
                d -= w[ji[j]] * jw[j] * u[ji[j]]
            f = 0.0
            if d > 0.0:
                f = min(mal[1] ** (1.0 + mal[2]) * d ** (mal[2] - 1.0), mal[5]) * d
            uh = 2.0 * mal[3] - mal[4] - k * k * f / mal[0]
            mal[4] = mal[3]
            mal[3] = uh
            for j in range(ji.shape[0]):
                t[ji[j]] += fgain * f * jw[j]
        elif bow[0] > 0.0:
            i = ji[0]
            # u+_i = t_i - 2k c phi(v), v = (u+_i - u-_i)/(2k) - vB, phi(v) = A v exp(-sig v^2)
            c = min(fgain * jw[0] * bow[0] * force[s] / (2.0 * k), 0.9 / (2.0 * A * math.exp(-1.5)))
            b = (t[i] - u1[i]) / (2.0 * k) - bow[1]
            v = bow[3]
            for it in range(40):
                e = math.exp(-bow[2] * v * v)
                dv = (v + c * A * v * e - b) / (1.0 + c * A * (1.0 - 2.0 * bow[2] * v * v) * e)
                v -= dv
                if abs(dv) < 1e-10:
                    break
            bow[3] = v
            t[i] -= 2.0 * k * c * A * v * math.exp(-bow[2] * v * v)
        elif force[s] != 0.0:
            f = fgain * force[s]
            for j in range(ji.shape[0]):
                t[ji[j]] += f * jw[j]
        for i in range(n):
            u1[i] = u[i]
            u[i] = t[i]
        for p in range(pk.shape[0]):
            ph = pk[p, 4] * s + 1.7 * p
            out[s, p] = _read(u, idx, pk[p, 0] + pk[p, 2] * math.cos(ph), pk[p, 1] + pk[p, 3] * math.sin(ph))


@numba.njit(cache=True)
def _bracket(a, b, cc, hx, hy, o):
    """Discrete Monge-Ampere bracket l(a,b) of eq 13.11 at the interior nodes; a, b are (Nx+1, Ny+1) with
    zero edges, cc is (Nx, Ny) scratch for the cell products dx+y+ a * dx+y+ b."""
    Nx, Ny = a.shape[0] - 1, a.shape[1] - 1
    ixx, iyy, ixy = 1.0 / (hx * hx), 1.0 / (hy * hy), 1.0 / (hx * hx * hy * hy)
    for l in range(Nx):
        for m in range(Ny):
            cc[l, m] = ((a[l + 1, m + 1] - a[l + 1, m] - a[l, m + 1] + a[l, m])
                        * (b[l + 1, m + 1] - b[l + 1, m] - b[l, m + 1] + b[l, m]) * ixy)
    r = 0
    for l in range(1, Nx):
        for m in range(1, Ny):
            axx = (a[l + 1, m] - 2.0 * a[l, m] + a[l - 1, m]) * ixx
            ayy = (a[l, m + 1] - 2.0 * a[l, m] + a[l, m - 1]) * iyy
            bxx = (b[l + 1, m] - 2.0 * b[l, m] + b[l - 1, m]) * ixx
            byy = (b[l, m + 1] - 2.0 * b[l, m] + b[l, m - 1]) * iyy
            o[r] = axx * byy + ayy * bxx - 0.5 * (cc[l, m] + cc[l - 1, m] + cc[l, m - 1] + cc[l - 1, m - 1])
            r += 1


@numba.njit(cache=True)
def _bracket_adj(a, phi, cc, hx, hy, g):
    """Gradient over b of sum_p hx hy phi_p l(a, b)_p (phi zero on the edges): the force l(phi, a) carried to every
    grid node, edges included, so <v, g> = <phi, l(a, v)> and the free-edge von Karman scheme keeps its energy."""
    Nx, Ny = a.shape[0] - 1, a.shape[1] - 1
    ixx, iyy, ixy = 1.0 / (hx * hx), 1.0 / (hy * hy), 1.0 / (hx * hy)
    hh = hx * hy
    g[:, :] = 0.0
    for l in range(1, Nx):
        for m in range(1, Ny):
            f = hh * phi[l, m]
            if f == 0.0:
                continue
            gy = f * (a[l + 1, m] - 2.0 * a[l, m] + a[l - 1, m]) * ixx * iyy
            gx = f * (a[l, m + 1] - 2.0 * a[l, m] + a[l, m - 1]) * iyy * ixx
            g[l, m + 1] += gy
            g[l, m] -= 2.0 * gy
            g[l, m - 1] += gy
            g[l + 1, m] += gx
            g[l, m] -= 2.0 * gx
            g[l - 1, m] += gx
    for l in range(Nx):
        for m in range(Ny):
            s = phi[l, m] + phi[l + 1, m] + phi[l, m + 1] + phi[l + 1, m + 1]
            if s == 0.0:
                continue
            c = -0.5 * hh * s * (a[l + 1, m + 1] - a[l + 1, m] - a[l, m + 1] + a[l, m]) * ixy * ixy
            g[l + 1, m + 1] += c
            g[l + 1, m] -= c
            g[l, m + 1] -= c
            g[l, m] += c


@numba.njit(cache=True)
def _fill(up, u, idx, free, Nx, Ny):
    if free:
        for l in range(Nx + 1):
            for m in range(Ny + 1):
                up[l, m] = u[idx[l, m]] if idx[l, m] >= 0 else 0.0
    else:
        up[1:Nx, 1:Ny] = u.reshape(Nx - 1, Ny - 1)


@numba.njit(cache=True)
def _nl_force(pp, up, cc, hx, hy, nl, free, gg, idx, w):
    if free:
        _bracket_adj(up, pp, cc, hx, hy, gg)
        for l in range(gg.shape[0]):
            for m in range(gg.shape[1]):
                if idx[l, m] >= 0:
                    nl[idx[l, m]] = gg[l, m] / w[idx[l, m]]
    else:
        _bracket(pp, up, cc, hx, hy, nl)


@numba.njit(cache=True)
def _chol_solve(ch, x):
    """In-place solve with a banded Cholesky factor, ch[j, i] = L[j+i, j]."""
    n, bw = ch.shape[0], ch.shape[1] - 1
    for j in range(n):
        x[j] /= ch[j, 0]
        xj = x[j]
        for i in range(1, min(bw, n - 1 - j) + 1):
            x[j + i] -= ch[j, i] * xj
    for j in range(n - 1, -1, -1):
        s = x[j]
        for i in range(1, min(bw, n - 1 - j) + 1):
            s -= ch[j, i] * x[j + i]
        x[j] = s / ch[j, 0]


@numba.njit(cache=True)
def _stress_bound(p, hx, hy):
    """Gershgorin bound on the operator v -> l(p, v): the largest stiffness the Airy function p can add."""
    Nx, Ny = p.shape[0] - 1, p.shape[1] - 1
    ixx, iyy, ixy = 1.0 / (hx * hx), 1.0 / (hy * hy), 1.0 / (hx * hy)
    g = 0.0
    for l in range(1, Nx):
        for m in range(1, Ny):
            s = 4.0 * iyy * abs(p[l + 1, m] - 2.0 * p[l, m] + p[l - 1, m]) * ixx
            s += 4.0 * ixx * abs(p[l, m + 1] - 2.0 * p[l, m] + p[l, m - 1]) * iyy
            for a in range(l - 1, l + 1):
                for b in range(m - 1, m + 1):
                    s += 2.0 * ixy * ixy * abs(p[a + 1, b + 1] - p[a + 1, b] - p[a, b + 1] + p[a, b])
            g = max(g, s)
    return g


@numba.njit(cache=True)
def _run_nl(Bd, Bi, Bp, Cd, Ci, Cp, Ld, Li, Lp, u, u1, ji, jw, force, fgain, mode, cnl, hx, hy, Nx, Ny, ch,
            ulim, idx, pk, out, curv, wq, cav, dst, dS, dlam):
    """Nonlinear simply supported plate; `t` below is the linear update B u - C u- + force.
    mode 1  Berger, eq 13.2 in the explicit form of g03 §6.1.
    mode 2  von Karman, the book's simple scheme (eqs 13.10 + 13.12): BIH Phi = -l(u,u), u+ = t + c l(Phi,u).
            Unstable at high amplitude, as the book warns.
    mode 3  von Karman, energy-conserving variant: same Phi, but u+ = t + (c/2)(l(Phi,u+) + l(Phi,u-)).
            l is bilinear and symmetric, so <delta_t. u, l(Phi, mu_t. u)> telescopes into
            (1/4k)(<Phi^n, BIH Phi^n+1> - <Phi^n-1, BIH Phi^n>) and the lossless scheme conserves
            E = |delta_t+ u|^2/2 + kappa^2 <u+, BIH u>/2 + kappa^2 <Phi+, BIH Phi>/4 (checked in the tests).
            Linear in u+, solved by fixed-point iteration, which contracts at rate (c/2) |l(Phi, .)|;
            the Airy function is scaled down when that rate would exceed 0.5 (stress saturation).
    Free edges (u on every node, n > (Nx-1)(Ny-1)): Phi stays clamped on the border and the force on u is the
    adjoint of the bracket (`_bracket_adj`), which equals l(Phi, u) inside and carries it onto the edge nodes.
    Returns how many steps a guard (saturation or the amplitude limit `ulim`) acted on."""
    n = u.shape[0]
    ni = (Nx - 1) * (Ny - 1)
    free = n != ni
    t, q, nl, rhs, qa = np.empty(n), np.empty(n), np.empty(n), np.empty(n), np.empty(ni)
    up, pp, cc = np.zeros((Nx + 1, Ny + 1)), np.zeros((Nx + 1, Ny + 1)), np.empty((Nx, Ny))
    gg = np.zeros((Nx + 1, Ny + 1))
    hh = hx * hy
    hits = 0
    for s in range(out.shape[0]):
        _lin(Bd, Bi, Bp, Cd, Ci, Cp, u, u1, t)
        if cav != 0.0:
            mm = 0.0
            for i in range(n):
                mm += wq[i] * u[i]
            for i in range(n):
                t[i] -= cav * mm
        if force[s] != 0.0:
            f = fgain * force[s]
            for j in range(ji.shape[0]):
                t[ji[j]] += f * jw[j]
        if mode == 1:
            sa, sq = 0.0, 0.0
            for r in range(n):
                v = 0.0
                for j in range(Lp[r], Lp[r + 1]):
                    v += Ld[j] * u[Li[j]]
                q[r] = v
                sa += (t[r] + u1[r]) * v
                sq += v * v
            m = -0.5 * hh * sa / (1.0 + 0.5 * cnl * hh * sq)
            for r in range(n):
                t[r] += cnl * m * q[r]
        elif mode >= 2:
            _fill(up, u, idx, free, Nx, Ny)
            _bracket(up, up, cc, hx, hy, qa)
            for r in range(ni):
                q[r] = -qa[r]
            if curv != 0.0:                                    # shell: BIH Phi = -l(u,u) + 2 q Lap u
                for i in range(1, Nx):
                    for j in range(1, Ny):
                        q[(i - 1) * (Ny - 1) + j - 1] += 2.0 * curv * ((up[i + 1, j] - 2.0 * up[i, j] + up[i - 1, j]) / (hx * hx)
                                                                       + (up[i, j + 1] - 2.0 * up[i, j] + up[i, j - 1]) / (hy * hy))
            qa[:] = q[:ni]
            if dst:
                qa[:] = airy_dst(qa.reshape(Nx - 1, Ny - 1), dS, dlam, hx).ravel()
            else:
                _chol_solve(ch, qa)
            pp[1:Nx, 1:Ny] = qa.reshape(Nx - 1, Ny - 1)
            if curv != 0.0:                                    # u+ gets -kappa^2 q Lap Phi
                for i in range(1, Nx):
                    for j in range(1, Ny):
                        t[(i - 1) * (Ny - 1) + j - 1] -= cnl * curv * ((pp[i + 1, j] - 2.0 * pp[i, j] + pp[i - 1, j]) / (hx * hx)
                                                                       + (pp[i, j + 1] - 2.0 * pp[i, j] + pp[i, j - 1]) / (hy * hy))
            if mode == 2:
                _nl_force(pp, up, cc, hx, hy, nl, free, gg, idx, wq)
                for r in range(n):
                    t[r] += cnl * nl[r]
            else:
                c2 = 0.5 * cnl
                g = c2 * _stress_bound(pp, hx, hy)
                if g > 0.5:
                    pp *= 0.5 / g
                    hits += 1
                _fill(up, u1, idx, free, Nx, Ny)
                _nl_force(pp, up, cc, hx, hy, nl, free, gg, idx, wq)
                for r in range(n):
                    rhs[r] = t[r] + c2 * nl[r]
                    q[r] = 2.0 * u[r] - u1[r]                  # predictor
                for it in range(80):
                    _fill(up, q, idx, free, Nx, Ny)
                    _nl_force(pp, up, cc, hx, hy, nl, free, gg, idx, wq)
                    d, mq = 0.0, 0.0
                    for r in range(n):
                        v = rhs[r] + c2 * nl[r]
                        d = max(d, abs(v - q[r]))
                        mq = max(mq, abs(v))
                        q[r] = v
                    if d <= 1e-11 * (1.0 + mq):
                        break
                for r in range(n):
                    t[r] = q[r]
        mx = 0.0
        for r in range(n):
            mx = max(mx, abs(t[r]))
        g = 1.0
        if not mx <= ulim:                                     # also catches NaN
            g = ulim / mx if mx > ulim else 0.0
            hits += 1
        for r in range(n):
            u1[r] = u[r] * g
            u[r] = t[r] * g if g > 0.0 else 0.0
        for p in range(pk.shape[0]):
            out[s, p] = _read(u, idx, pk[p, 0], pk[p, 1])
    return hits


# ---------------------------------------------------------------------------------------------------------
# simulation helpers
# ---------------------------------------------------------------------------------------------------------
_NOMAL = np.zeros(6)
_NOBOW = np.array([0.0, 0.0, 1.0, 0.0])


def wall_damping(idx, Nx, Ny, hx, hy, gamma, wall):
    """Per-node damping rate db (1/s) of the impedance edges, u_tt = ... - db u_t: the boundary flux
    gamma^2 u_n = -gamma u_t / zeta over the node's edge length, divided by its trapezoid mass weight,
    i.e. 2 gamma / (h_normal zeta) on each edge the node lies on."""
    d = np.zeros((Nx + 1, Ny + 1))
    for e, (z, h) in enumerate(zip(wall, (hx, hx, hy, hy))):
        if z > 0:
            sl = (0, slice(None)) if e == 0 else (Nx, slice(None)) if e == 1 else (slice(None), 0) if e == 2 \
                else (slice(None), Ny)
            d[sl] += 2.0 * gamma / (h * z)
    return d.ravel()[idx.ravel() >= 0]


def _wall_update(K, L, idx, Nx, Ny, hx, hy, gamma, wall, k, sig0, sig1):
    db = wall_damping(idx, Nx, Ny, hx, hy, gamma, wall)
    a = sp.diags(1.0 / (1.0 + sig0 * k + 0.5 * k * db))
    I = sp.identity(K.shape[0])
    B = a @ (2.0 * I - k * k * K + 2.0 * sig1 * k * L)
    C = a @ (sp.diags(1.0 - sig0 * k - 0.5 * k * db) + 2.0 * sig1 * k * L)
    return B, C


def wall_impedance(alpha):
    """Specific impedance zeta >= 1 of a resistive wall absorbing a fraction alpha of normal-incidence energy:
    the larger root of alpha = 4 zeta / (1 + zeta)^2; 0 (rigid) for alpha <= 0."""
    a = float(np.clip(alpha, 0.0, 1.0))
    return 0.0 if a <= 0.0 else (2.0 - a + 2.0 * math.sqrt(1.0 - a)) / a


def _pickups(mdl, pts, scan=0.0, radius=0.0):
    pk = np.zeros((len(pts), 5))
    for i, (x, y) in enumerate(pts):
        pk[i] = (x * mdl.Nx, y * mdl.Ny, radius * mdl.Nx, radius * mdl.Ny,
                 2.0 * math.pi * scan * mdl.k * (1.0 + 0.31 * i))
    return pk


def simulate(mdl, force, exc=(0.5, 0.5, 0.0), pickups=((0.6, 0.3),), scan=0.0, radius=0.0, mallet=None,
             bow=None, state=None):
    """Run the linear scheme for len(force) steps; returns displacement at the pickups, shape (n, P).
    `force` is force / total mass (scaled units of eq 12.25). `state` = (u, u_prev) to continue a run."""
    force = np.ascontiguousarray(force, dtype=np.float64)
    u, u1 = state if state is not None else (np.zeros(mdl.n), np.zeros(mdl.n))
    ji, jw = _spread(mdl, *exc)
    out = np.zeros((force.shape[0], len(pickups)))
    _run(*mdl.B, *mdl.C, u, u1, mdl.w, ji, jw, force, mdl.a * mdl.k ** 2, mdl.a * mdl.k ** 2 * mdl.cav,
         _NOMAL.copy() if mallet is None else mallet, _NOBOW.copy() if bow is None else bow, mdl.k, mdl.idx,
         _pickups(mdl, pickups, scan, radius), out)
    return out


@numba.njit(cache=True)
def _phi(x, kap, al):
    return kap / (al + 1.0) * x ** (al + 1.0) if x > 0.0 else 0.0


@numba.njit(cache=True)
def collision_step(a0, b, g, kap, al):
    """Solve r + g (Phi(r + a0) - Phi(a0)) / r + b = 0 by Newton, Phi = kap/(al+1) [eta]+^(al+1).
    Returns (r, F) with F = (Phi(a0 + r) - Phi(a0)) / r, the energy-conserving collision force."""
    p0 = _phi(a0, kap, al)
    r = 0.0
    q = 0.0
    for _ in range(50):
        if abs(r) > 1e-15:
            q = (_phi(r + a0, kap, al) - p0) / r
            dq = (kap * max(r + a0, 0.0) ** al * r - (_phi(r + a0, kap, al) - p0)) / (r * r)
        else:
            q = kap * max(a0, 0.0) ** al
            dq = 0.5 * kap * al * max(a0, 0.0) ** (al - 1.0)
        dr = (r + g * q + b) / (1.0 + g * dq)
        r -= dr
        if abs(dr) < 1e-16 + 1e-13 * abs(r):
            break
    if abs(r) > 1e-15:
        q = (_phi(r + a0, kap, al) - p0) / r
    else:
        q = kap * max(a0, 0.0) ** al
    return r, q


@numba.njit(cache=True)
def _run_snare(Bd, Bi, Bp, Cd, Ci, Cp, u, u1, w, ji, jw, force, fgain, cav, sj, sw, sp_, k, idx, pk, out, wout):
    """Linear head + force pulse, and snare wires under it: one lumped wire mass per node sj, spring to the
    rest gap, loss, energy-conserving collision with the head. sw rows = [u_w, u_w_prev];
    sp_ = [kap, alpha, 1/mu, omega^2, sigma, gap]."""
    n = u.shape[0]
    t = np.empty(n)
    kap, al, imu, om2, sg, gap = sp_[0], sp_[1], sp_[2], sp_[3], sp_[4], sp_[5]
    den = 1.0 + sg * k
    for s in range(out.shape[0]):
        _lin(Bd, Bi, Bp, Cd, Ci, Cp, u, u1, t)
        if cav != 0.0:
            m = 0.0
            for i in range(n):
                m += w[i] * u[i]
            m *= cav
            for i in range(n):
                t[i] -= m
        if force[s] != 0.0:
            f = fgain * force[s]
            for j in range(ji.shape[0]):
                t[ji[j]] += f * jw[j]
        vw = 0.0
        for c in range(sj.shape[0]):
            j = sj[c]
            uw, uw1 = sw[c, 0], sw[c, 1]
            tw = (2.0 * uw - (1.0 - sg * k) * uw1 - k * k * om2 * (uw + gap)) / den
            a0 = uw1 - u1[j]
            g = fgain / w[j] + k * k * imu / den
            r, F = collision_step(a0, a0 - (tw - t[j]), g, kap, al)
            t[j] += fgain / w[j] * F
            sw[c, 1] = uw
            sw[c, 0] = tw - k * k * imu / den * F
            vw += sw[c, 0] - uw
        for i in range(n):
            u1[i] = u[i]
            u[i] = t[i]
        wout[s] = vw
        for p in range(pk.shape[0]):
            out[s, p] = _read(u, idx, pk[p, 0], pk[p, 1])


@lru_cache(maxsize=8)
def _airy(Nx, Ny, hx, hy):
    """Banded Cholesky factor of the clamped biharmonic for the Airy stress function (eq 13.8)."""
    D2 = _operators(Nx, Ny, hx, hy, "cccc", 1.0, 1.0, 0.3, 1.4, 0.3, 0.0)[0]
    bw = 2 * (Ny - 1)
    ab = np.zeros((bw + 1, D2.shape[0]))
    for d in range(bw + 1):
        ab[d, :D2.shape[0] - d] = D2.diagonal(-d)
    return np.ascontiguousarray(cholesky_banded(ab, lower=True).T)


@numba.njit(cache=True)
def apnl(x, a1, a2):
    """Nonlinear allpass: a first-order allpass whose coefficient is a1 while the input is positive and a2 otherwise
    (switched at the input's zero crossings); a1 = a2 is the plain allpass."""
    y = np.empty_like(x)
    s = 0.0
    for i in range(x.shape[0]):
        a = a1 if x[i] > 0.0 else a2
        v = x[i] - a * s
        y[i] = a * v + s
        s = v
    return y


@numba.njit(cache=True)
def wgr(f, n, sr, r=1.0):
    """Waveguide resonator: t = C (x1 + x2), x1 <- r (t - x2), x2 <- r (t + x1), C = cos(2 pi f / fs). Returns the
    quadrature pair (x1, x2 / gi), gi = sqrt((1 + C) / (1 - C)), both of unit amplitude when r = 1."""
    C = math.cos(2.0 * math.pi * f / sr)
    gi = math.sqrt((1.0 + C) / (1.0 - C))
    c = np.empty(n)
    s = np.empty(n)
    x1 = 1.0
    x2 = 0.0
    for i in range(n):
        c[i] = x1
        s[i] = x2 / gi
        t = C * (x1 + x2)
        x1, x2 = r * (t - x2), r * (t + x1)
    return c, s


def simulate_nl(mdl, force, model="vk", exc=(0.5, 0.5, 0.0), pickups=((0.6, 0.3),), ulim=50.0, state=None, curv=0.0,
                cnl=None, airy="chol"):
    """Nonlinear plate (`mdl` all-supported "ssss", or all-free "ffff" with `curv` 0). `model`: "vk" (energy-conserving von
    Karman, mode 3 of `_run_nl`), "vk_simple" (the book's scheme 13.10), "berger", "linear". Displacement is
    in units of H/sqrt(6(1-nu^2)) for von Karman (eq 13.5) and H/sqrt(6) for Berger (eq 13.1); `ulim` is a
    last-resort amplitude limit in those units (the model itself is only valid to a few thicknesses).
    `cnl` overrides the nonlinear coefficient (a membrane under Berger tension modulation has kappa = 0).
    The air cavity of `mdl` acts as in the linear scheme. `airy`: "chol" (banded Cholesky, Phi clamped on the
    border) | "dst" (two sine-transform Poisson solves, Phi and Lap Phi zero on the border: faster on big grids).
    Returns (out, guard_hits)."""
    mode = {"linear": 0, "berger": 1, "vk_simple": 2}.get(model, 3)
    force = np.ascontiguousarray(force, dtype=np.float64)
    u, u1 = state if state is not None else (np.zeros(mdl.n), np.zeros(mdl.n))
    ji, jw = _spread(mdl, *exc)
    if airy not in ("chol", "dst"):
        raise ValueError("simulate_nl airy must be chol or dst")
    if mdl.n != (mdl.Nx - 1) * (mdl.Ny - 1):
        curv = 0.0
    dst = airy == "dst" and mode >= 2
    ch = _airy(mdl.Nx, mdl.Ny, mdl.hx, mdl.hy) if mode >= 2 and not dst else np.zeros((1, 1))
    dS, dlam = dst_factors(mdl.Nx, mdl.Ny, mdl.hx, mdl.hy) if dst else (np.zeros((1, 1)), np.zeros(1))
    out = np.zeros((force.shape[0], len(pickups)))
    hits = _run_nl(*mdl.B, *mdl.C, *mdl.Lc, u, u1, ji, jw, force, mdl.a * mdl.k ** 2, mode,
                   mdl.a * (mdl.kappa * mdl.k) ** 2 if cnl is None else float(cnl), mdl.hx, mdl.hy, mdl.Nx, mdl.Ny,
                   ch, ulim, mdl.idx, _pickups(mdl, pickups)[:, :2].copy(), out, float(curv), mdl.w,
                   mdl.a * mdl.k ** 2 * mdl.cav, dst, dS, dlam)
    return out, hits


def pulse(n, fs, fmax, ms, t0=0.0):
    """Raised-cosine force pulse, eq 12.28 (g03 §4.2)."""
    f = np.zeros(n)
    i0, m = int(t0 * fs), max(2, int(round(ms * 1e-3 * fs)))
    seg = 0.5 * fmax * (1.0 - np.cos(2.0 * math.pi * np.arange(m) / m))
    f[i0:i0 + m] = seg[:max(0, n - i0)]
    return f


def _rate(f, lo, hi):
    return int(round(float(np.clip(f, lo, hi)) / 500.0)) * 500


def _to_sr(y, fs, sr):
    if fs == sr:
        return y
    g = math.gcd(int(fs), int(sr))
    return resample_poly(y, int(sr) // g, int(fs) // g, axis=0)


def _finish(y, fs, sr, level, lp=None, hp=25.0, fade=0.01):
    """Resample, remove DC, optional lowpass, fades, peak -> `level`."""
    y = _to_sr(np.asarray(y, dtype=np.float64), fs, sr)
    y = F.highpass(y - y.mean(axis=0), hp, sr, order=2)
    if lp is not None and lp < 0.45 * sr:
        y = F.lowpass(y, lp, sr, order=2)
    n = y.shape[0]
    env = np.ones(n)
    a, r = min(int(0.0005 * sr), n // 4), min(int(fade * sr), n // 2)
    if a:
        env[:a] = np.linspace(0.0, 1.0, a)
    if r:
        env[-r:] *= np.linspace(1.0, 0.0, r)
    y = y * (env if y.ndim == 1 else env[:, None])
    y[0] = y[-1] = 0.0
    return y * (level / max(float(np.max(np.abs(y))), 1e-12))


def _vel(out, fs):
    """Velocity read-out (delta_t-): DC blocker for free bodies (g03 §2.3) and spectrally flat for a
    force-driven plate; the book's acceleration pickup (g03 §4.4) is this, differenced once more."""
    return np.diff(out, axis=0, prepend=np.zeros((1, out.shape[1]))) * fs


def _kappa(E, rho, nu, H, L):
    """kappa = sqrt(E H^2 / (12 rho (1-nu^2))) / L^2, L = sqrt(area) (g03 §3.1)."""
    return math.sqrt(E * H * H / (12.0 * rho * (1.0 - nu * nu))) / (L * L)


# ---------------------------------------------------------------------------------------------------------
# membrane drum
# ---------------------------------------------------------------------------------------------------------
def membrane(sr=DEFAULT_SR, vel=1.0, tension=1000.0, size=0.33, decay=0.6, damping=0.5, pos=0.35,
             hardness=0.5, air=0.5, depth=0.3, stiffness=0.0, contact=False, scan=0.0, quality=1.0,
             stereo=False, dur=None, berger=False, berger_force=200.0, detune=0.0, snare=False, snare_gap=0.0):
    """Circular drum head on a Cartesian grid with a staircase rim (g03 §2.3, §2.6, §2.7, §2.9, §3.6).
    `snare`: wires under a diameter (lumped masses on springs) collide with the head through the potential
    kap/(a+1) [eta]+^(a+1), solved energy-conserving by Newton each step; struck by `berger_force` N at vel 1.
    `snare_gap` m: rest gap between wires and head (negative = wires pressed in).
    `berger`: tension modulation, T = T0 + (E h / 2) int |grad u|^2 (Mylar head), driven by a strike of `force`
    newtons (`berger_force`) at vel 1: hard hits start sharp and settle. `detune`: tension anisotropy +-d (Tx/Ty = (1+d)/(1-d)),
    which splits each (m >= 1, n) mode into a beating doublet.
    Returns velocity at one pickup (mono) or two (stereo, shape (n, 2))."""
    vel = float(np.clip(vel, 0.02, 1.0))
    size = float(np.clip(size, 0.08, 1.5))
    gamma = float(np.clip(math.sqrt(max(tension, 1.0) / HEAD_DENSITY) / size, 30.0, 1500.0))   # c / D
    hardness, damping, pos = (float(np.clip(v, 0.0, 1.0)) for v in (hardness, damping, pos))
    f01 = 2.405 / math.pi * gamma                                # Bessel zero j01, radius 1/2
    fs = _rate(70.0 * gamma * float(np.clip(quality, 0.25, 3.0)), 6000, 32000)
    kappa = 0.02 * float(np.clip(stiffness, 0.0, 1.0)) * gamma
    decay = float(np.clip(decay, 0.05, 8.0))
    s0, s1 = loss_coeffs(f01, decay, 6.0 * f01, decay * (1.0 - 0.85 * damping), kappa, gamma,
                         fs if kappa == 0.0 else 0.0)
    # Morse cavity term: d^2 = rho c0^2 L^4 / (T0 V0), V0 = disc area * depth, L = D (g03 §2.6)
    d2 = RHO_AIR * C_AIR ** 2 * size ** 2 * 4.0 / (math.pi * (HEAD_DENSITY * (gamma * size) ** 2)
                                                  * float(np.clip(depth, 0.03, 2.0)))
    d = float(np.clip(detune, 0.0, 0.3))
    mdl = _model(fs, round(math.sqrt((1.0 + d) / (1.0 - d)), 6), "ssss", round(kappa, 6), round(gamma, 4), round(s0, 6),
                 round(s1, 12), 0.3, None, True, round(float(np.clip(air, 0.0, 1.0)) * d2 * gamma ** 2, 3))
    dur = float(np.clip(decay * 1.2 + 0.1 if dur is None else dur, 0.05, 10.0))
    n = int(dur * fs)
    exc = (0.5 + 0.46 * pos * math.cos(0.6), 0.5 + 0.46 * pos * math.sin(0.6), 0.3 - 0.22 * hardness)
    mallet = None
    force = pulse(n, fs, 1.0, (6.0 - 5.3 * hardness) * (1.3 - 0.6 * vel))        # harder + faster = shorter
    if contact:                                                  # g03 §4.1: M, wH, a as Fig 12.6
        ji, jw = _spread(mdl, *exc)
        M, v0 = 0.4, 0.2 + 3.0 * vel
        mallet = np.array([M, 400.0 * 10.0 ** hardness, 3.0, 0.0, -v0 / fs,
                           1.0 / (mdl.k ** 2 * (1.0 / M + float(np.sum(mdl.w[ji] * jw * jw))))])
    pts = ((0.71, 0.33), (0.25, 0.62)) if stereo else ((0.71, 0.33),)
    if berger:
        T0 = max(tension, 1.0)
        f = force * (max(float(berger_force), 0.0) * vel / (HEAD_DENSITY * size ** 3))
        cnl = mdl.a * (mdl.k * gamma) ** 2 * MYLAR_EH / (2.0 * T0)
        y = _vel(simulate_nl(mdl, f, "berger", exc, pts, ulim=1e9, cnl=cnl)[0], fs)
    elif snare:
        f = force * (max(float(berger_force), 0.0) * vel / (HEAD_DENSITY * size ** 3))
        y = _vel(simulate_snare(mdl, f, exc, pts, size, snare_gap)[0], fs)
    else:
        y = _vel(simulate(mdl, force, exc, pts, scan, 0.08 if scan else 0.0, mallet), fs)
    y = _finish(y if stereo else y[:, 0], fs, sr, 0.9 * (0.55 + 0.45 * vel))
    return y


def simulate_snare(mdl, force, exc, pickups, size, gap=0.0):
    """Head driven by `force` (N / (rho D^3)) with SNARE_N wire masses under a diameter. Returns
    (head displacement at the pickups, summed wire velocity * k)."""
    force = np.ascontiguousarray(force, dtype=np.float64)
    u, u1 = np.zeros(mdl.n), np.zeros(mdl.n)
    ji, jw = _spread(mdl, *exc)
    sj = []
    for x in np.linspace(0.2, 0.8, SNARE_N):
        j = int(_spread(mdl, float(x), 0.5)[0][0])
        if j not in sj:
            sj.append(j)
    sj = np.array(sj, dtype=np.int64)
    mu = SNARE_MASS / (HEAD_DENSITY * size * size)
    sp_ = np.array([SNARE_K * size ** SNARE_ALPHA / (HEAD_DENSITY * size ** 3), SNARE_ALPHA, 1.0 / mu,
                    (2.0 * math.pi * SNARE_F) ** 2, _L10 / SNARE_T60, float(gap) / size])
    sw = np.zeros((len(sj), 2))
    sw[:] = -float(gap) / size
    out = np.zeros((force.shape[0], len(pickups)))
    wout = np.zeros(force.shape[0])
    _run_snare(*mdl.B, *mdl.C, u, u1, mdl.w, ji, jw, force, mdl.a * mdl.k ** 2, mdl.a * mdl.k ** 2 * mdl.cav,
               sj, sw, sp_, mdl.k, mdl.idx, _pickups(mdl, pickups), out, wout)
    return out, wout


@drum("fd_drum", "Finite-difference drum head (2D wave equation, circular, air-loaded): strike position, "
      "mallet hardness and tension behave physically.",
      tension=(1000.0, "head tension N/m (300 slack .. 4000 tight); pitch ~ sqrt(tension)/size"),
      size=(0.33, "head diameter m (0.15 bongo .. 0.6 bass drum)"),
      decay=(0.6, "T60 of the fundamental, s"),
      damping=(0.5, "0..1 extra loss at high frequency (0 = ringing, 1 = dead head)"),
      pos=(0.35, "strike position 0 = centre (thud) .. 1 = rim (ringy overtones)"),
      hardness=(0.5, "mallet 0 = soft felt (wide, 6 ms) .. 1 = hard stick (narrow, <1 ms)"),
      air=(0.5, "0..1 air-cavity load: raises only the modes that move air (kettledrum)"),
      depth=(0.3, "shell depth m (cavity volume)"),
      stiffness=(0.0, "0..1 bending stiffness of the head (stretches upper partials)"),
      contact=(0, "1 = power-law mallet contact instead of a force pulse (brightness follows vel)"),
      scan=(0.0, "Hz: moving-pickup rate (slow phasing), 0 = fixed pickup"),
      quality=(1.0, "0.25..3 internal sample rate / grid size (cost ~ quality^3)"),
      berger=(0, "1 = tension modulation: a hard hit starts sharp and its pitch settles as it rings"),
      force=(200.0, "N peak stick force at vel 1 (with berger: how far the pitch is pushed up)"),
      detune=(0.0, "0..0.3 tension anisotropy: splits the modes into beating doublets (0.07 = a real head)"),
      under=(0.0, "0..1 level of a second microphone under the drum (hears the head inverted, later, darker)"),
      flip=(0, "1 = polarity of the under mic flipped back (the lows add up instead of partly cancelling)"),
      snare=(0, "1 = snare wires under the head: energy-conserving collisions give the buzz (struck by force)"),
      snare_gap=(0.0, "m rest gap wires-head (negative = wires pressed in, tighter buzz)"))
def fd_drum(sr=DEFAULT_SR, vel=1.0, tension=1000.0, size=0.33, decay=0.6, damping=0.5, pos=0.35, hardness=0.5,
            air=0.5, depth=0.3, stiffness=0.0, contact=0, scan=0.0, quality=1.0, berger=0, force=200.0, detune=0.0,
            under=0.0, flip=0, snare=0, snare_gap=0.0):
    y = membrane(sr, vel, tension, size, decay, damping, pos, hardness, air, depth, stiffness, bool(contact),
                 scan, quality, berger=bool(berger), berger_force=force, detune=detune, snare=bool(snare),
                 snare_gap=snare_gap)
    if under > 0.0:
        from .percussion import double_mic
        y = double_mic(y, sr, under=under, flip=bool(flip), depth=float(depth))
    return y


# ---------------------------------------------------------------------------------------------------------
# linear struck / bowed plates
# ---------------------------------------------------------------------------------------------------------
def struck_plate(sr=DEFAULT_SR, kappa=12.0, eps=1.3, bc="ffff", nu=0.3, ortho=None, t60=3.0, t60_high=1.0,
                 pos=(0.31, 0.37), hardness=0.6, vel=1.0, dur=None, bright=0.5, quality=1.0, level=0.8):
    """Linear Kirchhoff plate struck with force pulse 12.28, velocity pickup (g03 §3, §4.2)."""
    q = float(np.clip(quality, 0.25, 3.0))
    kmax = max(ortho) if ortho else kappa
    fs = _rate(4.0 * kmax * 1500.0 * q, 8000, 32000)
    sc = min(1.0, fs / 240.0 / kmax)                               # keep at least ~60 grid points
    kappa, ortho = kappa * sc, tuple(v * sc for v in ortho) if ortho else None
    t60 = float(np.clip(t60, 0.05, 20.0))
    kl = math.sqrt(ortho[0] * ortho[1]) if ortho else kappa
    f2 = min(2000.0, 0.3 * fs)
    s0, s1 = loss_coeffs(200.0, t60, f2, float(np.clip(t60_high, 0.02, t60)), kl, 0.0, 0.0 if ortho else fs)
    mdl = _model(fs, eps, "ssss" if ortho else bc, round(kappa, 5), 0.0, round(s0, 6), round(s1, 10), nu,
                 tuple(round(v, 5) for v in ortho) if ortho else None)
    dur = float(np.clip(t60 * 0.9 + 0.1 if dur is None else dur, 0.05, 12.0))
    hardness, vel = float(np.clip(hardness, 0.0, 1.0)), float(np.clip(vel, 0.02, 1.0))
    force = pulse(int(dur * fs), fs, 1.0, (5.0 - 4.5 * hardness) * (1.3 - 0.6 * vel))
    y = _vel(simulate(mdl, force, (pos[0], pos[1], 0.0), ((0.62, 0.29),)), fs)[:, 0]
    return _finish(y, fs, sr, level * (0.55 + 0.45 * vel), 1500.0 * 8.0 ** float(np.clip(bright, 0.0, 1.0)))


def _material_plate(sr, mat, size, thickness, eps, bc, t60, t60_high, pos, hardness, vel, bright, quality):
    m = MATERIALS[mat]
    size, H = float(np.clip(size, 0.05, 3.0)), float(np.clip(thickness, 0.1, 50.0)) * 1e-3
    return struck_plate(sr, _kappa(m.young_modulus, m.density, m.poisson, H, size), eps, bc, m.poisson, None,
                        t60, t60_high, (0.5 - 0.42 * pos, 0.5 - 0.3 * pos), hardness, vel, None, bright, quality)


_PLATE_PARAMS = dict(
    pos=(0.45, "strike position 0 = centre .. 1 = corner"),
    hardness=(0.6, "striker 0 = soft (5 ms contact) .. 1 = hard (0.5 ms)"),
    vel=(0.8, "strike strength 0..1 (harder = shorter contact, brighter)"),
    bright=(0.5, "0..1 output lowpass 1.5 kHz .. 12 kHz"),
    quality=(1.0, "0.25..3 internal sample rate / grid size"))


@sfx("metal_plate", "Struck steel plate with free edges (Kirchhoff plate, finite differences): clang of a hung "
     "sheet, hatch or sign.", size=(0.5, "side length m (square root of the area)"),
     thickness=(2.0, "mm (thicker = higher and purer)"), decay=(3.0, "T60 at 200 Hz, s"),
     aspect=(1.3, "side ratio 1..4"), edges=("free", "free | supported | clamped"), **_PLATE_PARAMS)
def metal_plate(sr=DEFAULT_SR, size=0.5, thickness=2.0, decay=3.0, aspect=1.3, edges="free", pos=0.45,
                hardness=0.6, vel=0.8, bright=0.5, quality=1.0):
    bc = {"free": "ffff", "supported": "ssss", "clamped": "cccc"}.get(str(edges), "ffff")
    return _material_plate(sr, "steel", size, thickness, float(np.clip(aspect, 1.0, 4.0)), bc, decay,
                           0.45 * decay, pos, hardness, vel, bright, quality)


@sfx("glass_pane", "Struck window pane held in its frame (supported glass plate): a dull ringing knock.",
     size=(0.5, "side length m"), thickness=(4.0, "mm"), decay=(0.9, "T60 at 200 Hz, s"),
     aspect=(1.4, "side ratio 1..4"), **_PLATE_PARAMS)
def glass_pane(sr=DEFAULT_SR, size=0.5, thickness=4.0, decay=0.9, aspect=1.4, pos=0.45, hardness=0.6, vel=0.8,
               bright=0.5, quality=1.0):
    return _material_plate(sr, "glass", size, thickness, float(np.clip(aspect, 1.0, 4.0)), "ssss", decay,
                           0.35 * decay, pos, hardness, vel, bright, quality)


@sfx("wood_panel", "Knock on a wooden panel / soundboard (orthotropic plate, stiff along the grain).",
     size=(1.0, "relative size (1 = the soundboard of Bilbao Table 12.2, lowest mode 56 Hz; 0.5 = 4x higher)"),
     thickness=(1.0, "relative thickness (pitch scales with thickness/size^2)"),
     decay=(0.35, "T60 at 200 Hz, s"), **_PLATE_PARAMS)
def wood_panel(sr=DEFAULT_SR, size=1.0, thickness=1.0, decay=0.35, pos=0.45, hardness=0.6, vel=0.8, bright=0.5,
               quality=1.0):
    sc = float(np.clip(thickness, 0.2, 5.0)) / float(np.clip(size, 0.2, 4.0)) ** 2
    return struck_plate(sr, 0.0, WOOD_K[3], "ssss", 0.3, tuple(v * sc for v in WOOD_K[:3]), decay, 0.3 * decay,
                        (0.5 - 0.42 * pos, 0.5 - 0.3 * pos), hardness, vel, None, bright, quality)


@sfx("plate_bow", "Bowed metal plate edge (bowed cymbal / waterphone / metal scrape): friction-driven plate.",
     dur=(2.0, "s"), size=(0.3, "side length m"), thickness=(1.7, "mm (bronze); size 0.3 + 1.7 mm = kappa 20"),
     force=(15.0, "bow force 5..60 (10 = pitched, 30 = higher harmonic, 50 = scrape then pitch)"),
     speed=(1.0, "bow velocity 0.2..3"), pos=(0.5, "0..1 position of the bow along the free edge"),
     decay=(4.0, "plate T60, s"), bright=(0.5, "0..1 output lowpass 1.5 kHz .. 12 kHz"))
def plate_bow(sr=DEFAULT_SR, dur=2.0, size=0.3, thickness=1.7, force=15.0, speed=1.0, pos=0.5, decay=4.0,
              bright=0.5):
    fs = 32000
    kappa = min(_kappa(BRONZE[1], BRONZE[0], BRONZE[2], float(np.clip(thickness, 0.2, 20.0)) * 1e-3,
                       float(np.clip(size, 0.1, 2.0))), fs / 240.0)
    decay = float(np.clip(decay, 0.1, 20.0))
    s0, s1 = loss_coeffs(200.0, decay, 2000.0, 0.6 * decay, kappa, 0.0, fs)
    mdl = _model(fs, 1.0, "fccc", round(kappa, 5), 0.0, round(s0, 6), round(s1, 10))   # bowed edge free (§4.3)
    dur = float(np.clip(dur, 0.1, 20.0))
    n = int((dur + min(1.0, 0.3 * decay)) * fs)
    env = np.zeros(n)                                              # bow-force envelope
    m, a = int(dur * fs), max(2, int(0.04 * fs))
    env[:m] = 1.0
    env[:a] = np.linspace(0.0, 1.0, a)
    env[m - a:m] = np.linspace(1.0, 0.0, a)
    ji, jw = _spread(mdl, 0.0, float(np.clip(pos, 0.05, 0.95)))
    j = int(np.argmax(jw))                                         # one node (J0 truncation, g03 §4.5)
    out = np.zeros((n, 1))
    # sig = 1 puts the friction peak at 0.7 vB for vB = 1  # Fig 12.8 gives FB and vB only
    bow = np.array([float(np.clip(force, 0.5, 200.0)), float(np.clip(speed, 0.05, 5.0)), 1.0, 0.0])
    _run(*mdl.B, *mdl.C, np.zeros(mdl.n), np.zeros(mdl.n), mdl.w, ji[j:j + 1], 1.0 / mdl.w[ji[j:j + 1]], env,
         mdl.a * mdl.k ** 2, 0.0, _NOMAL.copy(), bow, mdl.k, mdl.idx, _pickups(mdl, ((0.7, 0.7),)), out)
    return _finish(_vel(out, fs)[:, 0], fs, sr, 0.7, 1500.0 * 8.0 ** float(np.clip(bright, 0.0, 1.0)), fade=0.03)


# ---------------------------------------------------------------------------------------------------------
# nonlinear plates: cymbals, gongs, thunder sheet
# ---------------------------------------------------------------------------------------------------------
def nl_plate(sr=DEFAULT_SR, vel=1.0, size=0.4, thickness=1.0, force=80.0, contact_ms=16.0, pos=(0.2, 0.3),
             decay=10.0, damping=0.005, model="vk", eps=1.0, material=BRONZE, bright=0.6, quality=1.0,
             dur=2.0, points=900, drive=None, level=0.85, info=None, accel=True, hp=150.0, loss=None, scan=None, dome=0.0,
             scan_order=1, edges="supported"):
    """Nonlinear plate (g03 §6): `model` = "vk" (von Karman simple scheme, Airy stress solve every step),
    `edges` "supported" | "free" (free rectangle, Phi clamped on its border; no dome),
    "berger" (tension modulation: glide only) or "linear". `size` is the diameter of the disc of equal
    area, `force` the peak strike force in newtons, converted to the scaled units of eq 13.9 by
    F / (rho H L^2 u0). `drive` (callable n, fs -> force array in N) replaces the single pulse.
    `accel`: acceleration pickup (g03 §4.4; what a small radiator sends to the air) instead of velocity;
    `hp`: output highpass in Hz (the lowest modes of a big thin plate are subsonic flapping).
    `loss` = (T60 at 0 Hz, f_c, T60 at f_c) replaces decay / damping (Bilbao et al. DAFx-23, eq. 48);
    `dome` (m, 0 = flat): radius of curvature, a shallow spherical shell (Bilbao 13.3): q = L^2 / (Rs u0), and BIH Phi gets 2 q Lap u
    because the scaling of 13.16 leaves the two couplings unequal; their product gives w^2 = w0^2 + E / (rho Rs^2), which the
    book's equal q in 13.18 would halve (Chaigne & Kergomard 8.5: chi = 12 (1 - nu^2) a^4 / (R^2 h^2)). Rectangular simply supported
    edge, not the book's free circle.
    `scan` = (R, f_o): the output moves on their ellipse x = 1/2 + R/2 cos(2 pi f_o t), y = 1/2 + R/2 sin(...)
    (eq. 7), read from 48 fixed points with linear interpolation along the ellipse (`scan_order` 3: 4-point
    Lagrange)."""
    rho, E, nu = material
    L = 0.886 * float(np.clip(size, 0.1, 3.0))                     # sqrt(pi)/2 * diameter
    H = float(np.clip(thickness, 0.2, 20.0)) * 1e-3
    q = float(np.clip(quality, 0.25, 3.0))
    kappa = _kappa(E, rho, nu, H, L)
    fs = _rate(4.0 * kappa * points * q, 4000, 24000)
    kappa = min(kappa, fs / 240.0)
    decay = float(np.clip(decay, 0.1, 30.0))
    if loss is not None:
        s0, s1 = loss_coeffs(1.0, loss[0], loss[1], loss[2], kappa, 0.0, fs)
    else:
        s0, s1 = _L10 / decay, float(np.clip(damping, 0.0, 0.05))
    mdl = _model(fs, eps, "ffff" if edges == "free" else "ssss", round(kappa, 5), 0.0, round(s0, 6), round(s1, 10), nu)
    u0 = H / math.sqrt(6.0 if model == "berger" else 6.0 * (1.0 - nu * nu))
    vel = float(np.clip(vel, 0.02, 1.0))
    n = int(float(np.clip(dur, 0.05, 20.0)) * fs)
    if drive is None:
        f = pulse(n, fs, max(force, 1e-3) * vel * vel, float(np.clip(contact_ms, 0.3, 40.0)) * (1.25 - 0.5 * vel))
    else:
        f = drive(n, fs)
    if scan is None:
        pts = ((0.63, 0.41),)
    else:
        th = 2.0 * math.pi * np.arange(48) / 48
        pts = tuple(zip(0.5 + 0.5 * scan[0] * np.cos(th), 0.5 + 0.5 * scan[0] * np.sin(th)))
    curv = L * L / (float(dome) * u0) if dome > 0 and model.startswith("vk") else 0.0
    out, hits = simulate_nl(mdl, f / (rho * H * L * L * u0), model, (pos[0], pos[1], 0.0), pts, curv=curv)
    if scan is not None:                                           # the moving pickup, interpolated along the ellipse
        ph = (np.arange(n) * scan[1] / fs % 1.0) * 48
        i0 = ph.astype(int) % 48
        fr = ph - np.floor(ph)
        rows = np.arange(n)
        if scan_order == 3:
            w = (-fr * (fr - 1.0) * (fr - 2.0) / 6.0, (fr + 1.0) * (fr - 1.0) * (fr - 2.0) / 2.0,
                 -(fr + 1.0) * fr * (fr - 2.0) / 2.0, (fr + 1.0) * fr * (fr - 1.0) / 6.0)
            out = sum(wk * out[rows, (i0 + k - 1) % 48] for k, wk in enumerate(w))[:, None]
        else:
            out = ((1.0 - fr) * out[rows, i0] + fr * out[rows, (i0 + 1) % 48])[:, None]
    if info is not None:
        info.update(fs=fs, kappa=kappa, n=mdl.n, guard_hits=hits, peak_u=float(np.max(np.abs(out))))
    y = _vel(_vel(out, fs), fs) if accel else _vel(out, fs)
    return _finish(y[:, 0], fs, sr, level * (0.5 + 0.5 * vel), 1500.0 * 8.0 ** float(np.clip(bright, 0.0, 1.0)),
                   hp, fade=0.03)


CYMBAL_LOSS = (6.9, 1000.0, 4.15)    # T60 at 0 Hz, f_c, T60 at f_c of mu = 0.007 w^0.6 + 2
TAMTAM_LOSS = (75.0, 8000.0, 4.0)    # tam-tam: T60 of the lowest partials and at 8 kHz


def _law(decay, law):
    return (decay, law[1], decay * law[2] / law[0])


def _nl_params(size, thickness, force, decay, dur):
    return dict(
        size=(size, "diameter m (bigger = lower, denser, slower to build)"),
        thickness=(thickness, "mm (thinner = crashes at a lighter touch: nonlinearity ~ force/thickness^2)"),
        force=(force, "peak strike force N at vel 1 = crash intensity (low: clean ring; high: glide, then the "
                      "wash of new partials building up after the hit)"),
        decay=(decay, "T60 of the low modes, s"), dur=(dur, "rendered length s"),
        bright=(0.6, "0..1 output lowpass 1.5 kHz .. 12 kHz"),
        model=("vk", "vk (von Karman: crash build-up) | vk_simple (the book's explicit scheme, can blow "
                     "up when hit hard) | berger (pitch glide only, cheap) | linear"),
        quality=(1.0, "0.25..3 internal sample rate / grid size (cost ~ quality^2.5)"))


@drum("plate_crash", "Crash cymbal from a von Karman nonlinear plate: the high-frequency wash builds up after "
      "the hit, more with `force`.", **_nl_params(0.4, 1.0, 400.0, 6.9, 2.2))
def plate_crash(sr=DEFAULT_SR, vel=1.0, size=0.4, thickness=1.0, force=400.0, decay=6.9, dur=2.2, bright=0.6,
                model="vk", quality=1.0):
    return nl_plate(sr, vel, size, thickness, force, 12.0, (0.2, 0.3), decay, 0.005, model, 1.0, BRONZE, bright,
                    quality, dur, loss=_law(float(np.clip(decay, 0.1, 30.0)), CYMBAL_LOSS))


@drum("shell_crash", "Crash cymbal as a shallow spherical shell: the curvature stiffens the low modes and strengthens the crash.",
      dome=(1.3, "m radius of curvature (default: 15 mm rise on 40 cm)"), **_nl_params(0.4, 1.0, 400.0, 6.9, 2.2))
def shell_crash(sr=DEFAULT_SR, vel=1.0, dome=1.3, size=0.4, thickness=1.0, force=400.0, decay=6.9, dur=2.2, bright=0.6,
                model="vk", quality=1.0):
    return nl_plate(sr, vel, size, thickness, force, 12.0, (0.2, 0.3), decay, 0.005, model, 1.0, BRONZE, bright, quality, dur,
                    dome=float(dome), loss=_law(float(np.clip(decay, 0.1, 30.0)), CYMBAL_LOSS))


@drum("plate_ride", "Ride cymbal: heavier nonlinear plate, short stick contact, clear ping over a light wash.",
      **_nl_params(0.5, 2.0, 600.0, 7.0, 2.2))
def plate_ride(sr=DEFAULT_SR, vel=1.0, size=0.5, thickness=2.0, force=600.0, decay=7.0, dur=2.2, bright=0.6,
               model="vk", quality=1.0):
    return nl_plate(sr, vel, size, thickness, force, 2.5, (0.33, 0.4), decay, 0.003, model, 1.0, BRONZE, bright,
                    quality, dur, loss=_law(float(np.clip(decay, 0.1, 30.0)), CYMBAL_LOSS))


@drum("plate_china", "China cymbal: thin plate struck hard near the edge, trashy and short.",
      **_nl_params(0.45, 0.8, 400.0, 3.0, 1.6))
def plate_china(sr=DEFAULT_SR, vel=1.0, size=0.45, thickness=0.8, force=400.0, decay=3.0, dur=1.6, bright=0.6,
                model="vk", quality=1.0):
    return nl_plate(sr, vel, size, thickness, force, 6.0, (0.12, 0.2), decay, 0.006, model, 1.3, BRONZE, bright,
                    quality, dur, loss=_law(float(np.clip(decay, 0.1, 30.0)), CYMBAL_LOSS))


@drum("plate_gong", "Gong from a nonlinear plate struck at the centre with a soft mallet: pitch glides down "
      "as it rings, hard strikes bloom.", **_nl_params(0.5, 4.0, 6000.0, 9.0, 4.0))
def plate_gong(sr=DEFAULT_SR, vel=1.0, size=0.5, thickness=4.0, force=6000.0, decay=9.0, dur=4.0, bright=0.6,
               model="vk", quality=1.0):
    return nl_plate(sr, vel, size, thickness, force, 3.0, (0.5, 0.5), decay, 0.001, model, 1.0, BRONZE, bright,
                    quality, dur, accel=False, hp=30.0)


@drum("tam_tam", "Tam-tam: large thin nonlinear plate, soft off-centre mallet; the shimmer swells for a second "
      "after the hit and the lows ring on for a minute (the file fades out at `dur`).",
      **_nl_params(1.0, 2.5, 2000.0, 75.0, 8.0))
def tam_tam(sr=DEFAULT_SR, vel=1.0, size=1.0, thickness=2.5, force=2000.0, decay=75.0, dur=8.0, bright=0.6,
            model="vk", quality=1.0):
    return nl_plate(sr, vel, size, thickness, force, 10.0, (0.38, 0.44), decay, 0.004, model, 1.0, BRONZE,
                    bright, quality, dur, accel=False, hp=40.0, loss=_law(float(np.clip(decay, 0.1, 120.0)), TAMTAM_LOSS))


# Bilbao, Webb, Wang & Ducceschi, "Real-time gong synthesis", DAFx-23: steel (E 2e11 Pa, rho 7850, nu 0.3), 0.5 mm,
# aspect 1.4, T60 20 s at 0 Hz and 10 s at 1 kHz; a small plate (0.01 m^2, 2 ms strike) crashes when hit hard, a
# large one (0.16 m^2, 4 ms) swells; its force at vel 1 is 50 N, the strongest of their Fig. 4 (1-50 N).
# the strike point.
STEEL = (7850.0, 2e11, 0.3)

# rho, E, nu; specific heat C (J/kg K), expansion alpha (1/K), conductivity k (W/m K): the middle of each range
THERMO = {"steel": (7850.0, 2e11, 0.3, 540.0, 14e-6, 30.0), "aluminium": (2700.0, 7e10, 0.33, 900.0, 22e-6, 180.0),
          "glass": (2500.0, 7e10, 0.22, 700.0, 8e-6, 1.1), "brass": (8500.0, 1.0e11, 0.34, 380.0, 19e-6, 110.0)}
ROOM_K = 293.0


def thermoelastic(freqs, thickness: float, material: str = "steel", t0: float = ROOM_K):
    """Zener thermoelastic loss of a thin plate (h in m) at ``freqs`` Hz: (loss factor eta, decay rate d = eta w / 2 in 1/s).
    eta = Delta w tau / (1 + (w tau)^2), relaxation tau = rho C h^2 / (k pi^2), strength
    Delta = E alpha^2 T0 / (rho C) (1 + nu) / (1 - nu); the losses peak at f = 1 / (2 pi tau) and move up as 1 / h^2."""
    rho, E, nu, C, alpha, k = THERMO[material]
    tau = rho * C * thickness ** 2 / (k * math.pi ** 2)
    delta = E * alpha ** 2 * t0 / (rho * C) * (1.0 + nu) / (1.0 - nu)
    w = 2.0 * math.pi * np.asarray(freqs, float)
    eta = delta * w * tau / (1.0 + (w * tau) ** 2)
    return eta, 0.5 * eta * w


_DAFX23 = dict(eps=1.4, material=STEEL, loss=(20.0, 1000.0, 10.0))


def _dafx23_params(force, dur):
    return dict(force=(force, "peak strike force N at vel 1 (low: clean modes; higher: pitch glides; high: crash)"),
                dur=(dur, "rendered length s"), bright=(0.6, "0..1 output lowpass 1.5 kHz .. 12 kHz"),
                scan=(0.0, "Hz: the pickup circles an ellipse of R 0.4 at this rate (a gong swinging on its "
                            "rope; DAFx-23 use 1 Hz), 0 = fixed"),
                quality=(1.0, "0.25..3 internal sample rate / grid size (cost ~ quality^2.5)"))


@drum("steel_cymbal", "Small steel plate (DAFx-23 real-time gong model, 0.01 m^2): clear modes when tapped, "
      "uneven pitch glides harder, a crash of wideband noise at full force.", **_dafx23_params(80.0, 2.0))
def steel_cymbal(sr=DEFAULT_SR, vel=1.0, force=80.0, dur=2.0, bright=0.6, scan=0.0, quality=1.0):
    return nl_plate(sr, vel, 0.113, 0.5, force, 2.0, (0.38, 0.45), 20.0, 0.0, "vk", quality=quality, dur=dur,
                    bright=bright, scan=(0.4, scan) if scan > 0 else None, hp=60.0, **_DAFX23)


@drum("steel_gong", "Large steel plate (DAFx-23 real-time gong model, 0.16 m^2): the gong swell, energy "
      "migrating upwards over the first few hundred ms.", **_dafx23_params(50.0, 5.0))
def steel_gong(sr=DEFAULT_SR, vel=1.0, force=50.0, dur=5.0, bright=0.6, scan=1.0, quality=1.0):
    return nl_plate(sr, vel, 0.451, 0.5, force, 4.0, (0.38, 0.45), 20.0, 0.0, "vk", quality=quality, dur=dur,
                    bright=bright, scan=(0.4, scan) if scan > 0 else None, accel=False, hp=30.0, **_DAFX23)


@drum("thunder_sheet", "Thunder sheet: a big thin steel sheet shaken by hand; the nonlinear plate turns the slow "
      "shake into rolling rumble and crackle.",
      size=(1.1, "diameter-equivalent m"), thickness=(0.6, "mm"),
      force=(25.0, "shake force N at vel 1 (more = more crackle)"), rate=(5.0, "shakes per second"),
      dur=(3.0, "s of shaking"), decay=(4.0, "T60 s"), bright=(0.6, "0..1 output lowpass 1.5 kHz .. 12 kHz"),
      seed=(0, "int: shake pattern"), quality=(1.0, "0.25..3 internal sample rate / grid size"))
def thunder_sheet(sr=DEFAULT_SR, vel=1.0, size=1.1, thickness=0.6, force=25.0, rate=5.0, dur=3.0, decay=4.0,
                  bright=0.6, seed=0, quality=1.0):
    st = MATERIALS["steel"]
    dur = float(np.clip(dur, 0.2, 20.0))

    def shake(n, fs):                                              # irregular shake + a flick at the start
        rng = np.random.default_rng(int(seed))
        t = np.arange(n) / fs
        r = float(np.clip(rate, 0.5, 20.0))
        f = np.zeros(n)
        for i in range(4):
            f += rng.uniform(0.5, 1.0) * np.sin(2.0 * math.pi * r * rng.uniform(0.6, 1.7) * t + rng.uniform(0, 6.28))
        env = np.clip(t / 0.15, 0.0, 1.0) * np.clip((dur - t) / 0.2, 0.0, 1.0)
        return (f / 2.0 * env + pulse(n, fs, 1.5, 30.0)) * max(force, 1e-3) * vel
    return nl_plate(sr, vel, size, thickness, force, 16.0, (0.5, 0.12), decay, 0.004, "vk", 1.6,
                    (st.density, st.young_modulus, st.poisson), bright, quality, dur + min(1.5, 0.4 * decay),
                    1500, shake, accel=False, hp=40.0)


# ---------------------------------------------------------------------------------------------------------
# modal von Karman plates and shells (physical/vkmodal.py)
# ---------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _vk_model(size, thickness, profile, dome, edges, law, eps, material, modes, airy, basis):
    from .physical import vkmodal
    return vkmodal.build(size, thickness, eps, edges, material, profile, dome, modes, airy, basis=basis, law=law)


def vk_plate(sr=DEFAULT_SR, vel=1.0, size=0.4, thickness=1.0, profile="uniform", dome=0.0, edges="ffff", law="crash",
             force=400.0, contact_ms=1.0, pos=(0.3, 0.35), eps=1.0, material=BRONZE, modes=48, airy=24, scheme="sav",
             basis="grid", dur=2.0, bright=0.6, level=0.85, accel=True, hp=60.0, info=None, radiation=True,
             distance=0.0):
    """Modal von Karman plate or shallow shell (`physical.vkmodal`): `profile` = thickness law (a name of
    vkmodal.PROFILES, a (hc mm, he mm, rb) tuple, or "uniform" = `thickness` mm), `dome` = rise of the spherical cap
    in m (0 = flat: hardening, the pitch glides down as it rings; curved: softening, it glides up), `edges` 4 letters
    f/s/c, `law` = modal damping power law (vkmodal.DAMPING_LAWS), `scheme` sav | conservative | linear,
    `basis` grid | ritz (sine Rayleigh-Ritz basis, supported rectangle only). Force in N, vel^2 scaled like nl_plate.
    `radiation` adds the radiation loss of each mode (`radiation_damping`); `distance` (m, 0 = far) the dipole near field."""
    from .physical import vkmodal
    prof = tuple(profile) if isinstance(profile, (list, tuple)) else str(profile)
    m = _vk_model(round(float(np.clip(size, 0.05, 2.0)), 4), round(float(np.clip(thickness, 0.1, 20.0)), 4), prof,
                  round(float(np.clip(dome, 0.0, 0.2)), 5), str(edges), law, round(float(eps), 4), tuple(material),
                  int(np.clip(modes, 4, 160)), int(np.clip(airy, 2, 80)), basis)
    if radiation:
        from copy import copy
        m = copy(m)
        h = float(np.mean(m.h))
        m.mu = m.mu + radiation_damping(m.w / (2.0 * math.pi), h, m.Lx * m.Ly, 2.0 * (m.Lx + m.Ly), material)
    fs = _rate(2.5 * float(m.w[-1]) / (2.0 * math.pi), 8000, 48000)
    vel = float(np.clip(vel, 0.02, 1.0))
    n = int(float(np.clip(dur, 0.05, 30.0)) * fs)
    f = pulse(n, fs, max(force, 1e-3) * vel * vel, float(np.clip(contact_ms, 0.1, 40.0)) * (1.25 - 0.5 * vel))
    out = vkmodal.simulate(m, f, fs, pos, ((0.63, 0.41),), scheme)
    if info is not None:
        info.update(fs=fs, f_modes=m.w / (2.0 * math.pi), peak_u=float(np.max(np.abs(out))))
    y = near_field(_vel(_vel(out, fs), fs) if accel else _vel(out, fs), fs, distance)
    return _finish(y[:, 0], fs, sr, level * (0.5 + 0.5 * vel), 1500.0 * 8.0 ** float(np.clip(bright, 0.0, 1.0)), hp,
                   fade=0.03)


# size m, thickness profile, dome rise m, edges, damping law, contact ms, strike point, force N
VK_CYMBALS = {
    "crash": (0.4, "crash", 0.017, "ffff", "crash", 1.0, (0.3, 0.35), 400.0),
    "ride": (0.5, "ride", 0.017, "ffff", "crash", 0.6, (0.36, 0.4), 300.0),
    "splash": (0.25, "splash", 0.01, "ffff", "crash", 0.6, (0.3, 0.35), 150.0),
    "china": (0.45, "nguyen_thin", 0.0, "ffff", "crash", 1.0, (0.15, 0.2), 400.0),
}
VK_GONGS = {   # size m, thickness mm, dome rise m, edges, law, contact ms, strike, force N
    "opera": (0.3, 1.2, 0.0, "ffff", "gong", 4.0, (0.5, 0.5), 300.0),
    "curved": (0.2, 1.2, 0.02, "ffff", "gong", 4.0, (0.5, 0.5), 300.0),
    "tam_tam": (0.9, 2.0, 0.0, "ffff", "gong_hand", 8.0, (0.4, 0.45), 1500.0),
    "rectangle": (0.55, 1.2, 0.0, "ssss", "rectangle", 7.0, (0.38, 0.45), 300.0),
}

_VK_COMMON = dict(force=(0.0, "peak strike force N at vel 1 (0 = the kind's own)"), dur=(2.5, "rendered length s"),
                  bright=(0.6, "0..1 output lowpass 1.5 kHz .. 12 kHz"),
                  modes=(48, "transverse modes kept (more = brighter wash, slower; cost ~ modes^2)"),
                  scheme=("sav", "sav (explicit, energy-stable) | conservative (energy-conserving, implicit in the "
                                 "Airy amplitudes) | linear"))


@drum("vk_cymbal", "Modal von Karman cymbal: measured thickness profile (thick bell, thin edge), shallow-shell dome, "
      "free edge, power-law modal damping; the crash wash builds up from the strike.",
      kind=("crash", "crash | ride | splash | china"), dome=(-1.0, "rise of the cap in m (-1 = the kind's own; 0 = flat)"),
      **_VK_COMMON)
def vk_cymbal(sr=DEFAULT_SR, vel=1.0, kind="crash", dome=-1.0, force=0.0, dur=2.5, bright=0.6, modes=48, scheme="sav"):
    size, prof, d0, edges, law, ms, pos, f0 = VK_CYMBALS.get(str(kind), VK_CYMBALS["crash"])
    return vk_plate(sr, vel, size, 1.0, prof, d0 if dome < 0 else float(dome), edges, law,
                    f0 if not force else float(force), ms, pos, modes=modes, scheme=scheme, dur=dur, bright=bright)


@drum("vk_gong", "Modal von Karman gong: a flat opera gong glides down after the hit, a small curved gong glides "
      "up; tam_tam swells into a shimmer; rectangle = the supported test plate.",
      kind=("opera", "opera | curved | tam_tam | rectangle"), dome=(-1.0, "rise of the cap in m (-1 = the kind's own)"),
      **_VK_COMMON)
def vk_gong(sr=DEFAULT_SR, vel=1.0, kind="opera", dome=-1.0, force=0.0, dur=2.5, bright=0.6, modes=48, scheme="sav"):
    size, h, d0, edges, law, ms, pos, f0 = VK_GONGS.get(str(kind), VK_GONGS["opera"])
    return vk_plate(sr, vel, size, h, "uniform", d0 if dome < 0 else float(dome), edges, law,
                    f0 if not force else float(force), ms, pos, eps=1.5 if kind == "rectangle" else 1.0,
                    material=STEEL if kind == "rectangle" else BRONZE, modes=modes, scheme=scheme, dur=dur,
                    bright=bright, accel=False, hp=30.0)


# ---------------------------------------------------------------------------------------------------------
# effects
# ---------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=16)
def _ir_gain(fs, key, exc, pts):
    """1 / RMS-sum of the first 0.3 s of the velocity impulse response: makes wet level ~ dry level."""
    mdl = _model(fs, *key)
    imp = np.zeros(int(0.3 * fs))
    imp[0] = 1.0
    h = _vel(simulate(mdl, imp, exc, pts), fs)
    return 1.0 / max(float(np.sqrt(np.mean(np.sum(h * h, axis=0)))), 1e-30)


def _wet_fx(x, sr, fs, key, exc, pts, mix, width, predelay, tail):
    mdl = _model(fs, *key)
    mono = to_mono(x)
    n_out = x.shape[0] + int((tail + predelay) * sr)
    xin = _to_sr(mono - mono.mean(), sr, fs)
    drive = np.zeros(int(n_out * fs / sr) + 8)
    p = int(predelay * fs)
    m = min(xin.shape[0], drive.shape[0] - p)
    drive[p:p + m] = xin[:m]
    wet = _vel(simulate(mdl, drive, exc, pts), fs) * _ir_gain(fs, key, exc, pts)
    wet = _to_sr(wet, fs, sr)
    wet = F.highpass(np.concatenate([wet, np.zeros((max(0, n_out - wet.shape[0]), 2))])[:n_out], 30.0, sr, order=2)
    r = min(int(0.02 * sr), n_out // 2)
    if r:
        wet[-r:] *= np.linspace(1.0, 0.0, r)[:, None]
    mid, side = wet.mean(axis=1), 0.5 * (wet[:, 0] - wet[:, 1]) * float(np.clip(width, 0.0, 1.0))
    mix = float(np.clip(mix, 0.0, 1.0))
    dry = np.concatenate([x, np.zeros((n_out - x.shape[0],) + x.shape[1:])], axis=0)
    if x.ndim == 1 and width <= 0.0:
        return (1.0 - mix) * dry + mix * mid
    if dry.ndim == 1:
        dry = np.stack([dry, dry], axis=1)
    return (1.0 - mix) * dry + mix * np.stack([mid + side, mid - side], axis=1)


@effect("plate_reverb", "Plate reverberation: the signal drives a lossy steel plate (finite differences), two "
        "pickups give stereo. Dense and bright, no early reflections.",
        mix=(0.3, "wet amount 0..1"), t60_low=(3.0, "decay time at 500 Hz, s (0.3..10)"),
        t60_high=(1.5, "decay time at 2 kHz, s (<= t60_low)"),
        size=(1.0, "plate size 0.4..1.5 (1 = the 2 m plate, kappa 2: densest; smaller = thinner, more metallic)"),
        damping=(0.0, "0..1 damper pad: shortens both decay times"), predelay=(0.0, "seconds before the plate"),
        width=(1.0, "stereo width 0..1 (0 = mono out for a mono input)"),
        tail=(None, "seconds of tail to append (auto if omitted)"),
        quality=(1.0, "0.5..2 internal sample rate 16 kHz x quality (wet bandwidth, cost ~ quality^2)"))
def plate_reverb(x, sr=DEFAULT_SR, mix=0.3, t60_low=3.0, t60_high=1.5, size=1.0, damping=0.0, predelay=0.0,
                 width=1.0, tail=None, quality=1.0):
    fs = _rate(16000.0 * float(np.clip(quality, 0.5, 2.0)), 8000, 32000)
    kappa = 2.0 / float(np.clip(size, 0.4, 1.5)) ** 2              # reference plate: kappa = 2 (g03 §4.4)
    d = 1.0 - 0.85 * float(np.clip(damping, 0.0, 1.0))
    t1 = float(np.clip(t60_low, 0.3, 10.0)) * d
    t2 = float(np.clip(t60_high, 0.1, t1 / d)) * d
    s0, s1 = loss_coeffs(500.0, t1, 2000.0, min(t2, t1), kappa, 0.0, fs)
    key = (2.0, "ffff", round(kappa, 5), 0.0, round(s0, 6), round(s1, 10))     # aspect 2, free edges (Fig 12.9)
    tail = float(np.clip(0.8 * t1 if tail is None else tail, 0.05, 12.0))
    # input (0.807, 0.353), pickup (0.607, 0.303) of Fig 12.9, as fractions of the sides sqrt2 x 1/sqrt2
    return _wet_fx(x, sr, fs, key, (0.807 / 2 ** 0.5, 0.353 * 2 ** 0.5, 0.0),
                   ((0.607 / 2 ** 0.5, 0.303 * 2 ** 0.5), (0.24, 0.71)), mix, width,
                   float(np.clip(predelay, 0.0, 1.0)), tail)


@effect("room2d", "2D wave-equation room (rigid walls, source and two listeners on a membrane-like floor "
        "plan): real modes and flutter of a small room; wet band-limited to <= 4 kHz.",
        mix=(0.3, "wet amount 0..1"), size=(5.0, "room side m (2..15; sqrt of the floor area)"),
        aspect=(1.4, "length/width 1..3"), t60=(0.8, "decay time s"),
        damping=(0.5, "0..1 extra high-frequency absorption"), predelay=(0.0, "seconds"),
        width=(1.0, "stereo width 0..1"), tail=(None, "seconds of tail to append (auto if omitted)"),
        quality=(1.0, "0.5..2 internal rate / grid (cost ~ quality^3)"),
        absorb=(0.0, "wall absorption 0..1 at normal incidence (one value, or 4 for the walls x0, x1, y0, y1); "
                     "0 = rigid walls"))
def room2d(x, sr=DEFAULT_SR, mix=0.3, size=5.0, aspect=1.4, t60=0.8, damping=0.5, predelay=0.0, width=1.0,
           tail=None, quality=1.0, absorb=0.0):
    gamma = C_AIR / float(np.clip(size, 2.0, 15.0))                # gamma = c / L (g03 §2.8)
    q = float(np.clip(quality, 0.5, 2.0))
    fs = _rate(min(8000.0 * q, 150.0 * gamma * q), 2000, 16000)    # <= ~11000 q^2 grid points
    t60 = float(np.clip(t60, 0.1, 6.0))
    s0, s1 = loss_coeffs(125.0, t60, 0.2 * fs, t60 * (1.0 - 0.8 * float(np.clip(damping, 0.0, 1.0))), 0.0,
                         gamma, fs)
    key = (float(np.clip(aspect, 1.0, 3.0)), "ffff", 0.0, round(gamma, 4), round(s0, 6), round(s1, 12))
    ab = np.broadcast_to(np.asarray(absorb, dtype=float), (4,))
    if np.any(ab > 0):
        key = key + (0.3, None, False, 0.0, tuple(round(wall_impedance(v), 6) for v in ab))
    tail = float(np.clip(0.9 * t60 if tail is None else tail, 0.05, 8.0))
    return _wet_fx(x, sr, fs, key, (0.23, 0.31, 0.0), ((0.71, 0.62), (0.64, 0.83)), mix, width,
                   float(np.clip(predelay, 0.0, 1.0)), tail)


# ---------------------------------------------------------------------------------------------------------
# plate acoustics helpers: modal density, prestress, Rayleigh damping, radiation, near field, point force
# ---------------------------------------------------------------------------------------------------------
# rho, E, nu, loss factor eta at 1 kHz of shell plastics
PLASTICS = {"abs": (1050.0, 2.3e9, 0.35, 0.02), "polycarbonate": (1200.0, 2.4e9, 0.37, 0.012),
            "pvc": (1400.0, 3.0e9, 0.38, 0.03), "acrylic": (1180.0, 3.2e9, 0.37, 0.04)}


def modal_density(area, thickness, material=BRONZE):
    """Modes per Hz of a thin plate, independent of frequency: n = sqrt(3) S / (c_L h), c_L = sqrt(E / (rho (1 - nu^2)))."""
    rho, E, nu = material[:3]
    return math.sqrt(3.0) * area / (math.sqrt(E / (rho * (1.0 - nu * nu))) * thickness)


def prestressed_modes(lx, ly, thickness, material=BRONZE, tx=0.0, ty=0.0, n=6):
    """Supported rectangular plate under in-plane tensions tx, ty (N/m): w^2 = (D / rho h)(kx^2 + ky^2)^2
    + (tx kx^2 + ty ky^2) / (rho h), kx = p pi / lx, ky = q pi / ly. Hz, sorted."""
    rho, E, nu = material[:3]
    D = E * thickness ** 3 / (12.0 * (1.0 - nu * nu))
    p, q = np.meshgrid(np.arange(1, 4 * n), np.arange(1, 4 * n), indexing="ij")
    kx2, ky2 = (p * math.pi / lx) ** 2, (q * math.pi / ly) ** 2
    w2 = (D * (kx2 + ky2) ** 2 + tx * kx2 + ty * ky2) / (rho * thickness)
    return np.sort(np.sqrt(np.clip(w2, 0.0, None)).ravel())[:n] / (2.0 * math.pi)


def rayleigh_t60(freqs, plastic="abs"):
    """T60 (s) of the modes of a plastic shell with Rayleigh damping fitted to its loss factor at 1 kHz
    (stiffness-proportional, zeta = beta w / 2, beta = eta / w_1k): T60 = 6.91 / (zeta w)."""
    eta = PLASTICS[plastic][3]
    w = 2.0 * math.pi * np.asarray(freqs, float)
    beta = eta / (2.0 * math.pi * 1000.0)
    return 6.91 / (0.5 * beta * w * w)


def coincidence(thickness, material=BRONZE):
    """Coincidence frequency (Hz) of a thin plate in air: c^2 / (2 pi) sqrt(rho h / D)."""
    rho, E, nu = material[:3]
    D = E * thickness ** 3 / (12.0 * (1.0 - nu * nu))
    return C_AIR ** 2 / (2.0 * math.pi) * math.sqrt(rho * thickness / D)


def radiation_efficiency(f, thickness, area, perimeter, material=BRONZE, stiffeners=0.0):
    """Radiation efficiency sigma of a finite plate's resonant modes: 1 above coincidence, edge-mode law
    P lambda_c / (pi^2 S) sqrt(f / fc) below it (capped at 1). `stiffeners`: total length (m) of ribs
    fixed on the plate; below fc they multiply sigma by 1 + 2 stiffeners / perimeter."""
    fc = coincidence(thickness, material)
    f = np.asarray(f, float)
    below = perimeter * (C_AIR / fc) / (math.pi ** 2 * area) * np.sqrt(np.clip(f, 0.0, None) / fc)
    below = below * (1.0 + 2.0 * max(float(stiffeners), 0.0) / perimeter)
    return np.where(f >= fc, 1.0, np.minimum(below, 1.0))


def radiation_damping(f, thickness, area, perimeter, material=BRONZE, stiffeners=0.0):
    """Radiation loss of the modes, mu_rad = 2 zeta w = rho0 c sigma / (rho h) (1/s); T60 = 13.82 / mu."""
    rho = material[0]
    sigma = radiation_efficiency(f, thickness, area, perimeter, material, stiffeners)
    return RHO_AIR * C_AIR * sigma / (rho * thickness)


def near_field(y, sr, distance):
    """Dipole near field at `distance` m: the pressure rises as 1 / (k r) below c / (2 pi r),
    H(s) = (s + c / r) / (s + 2 pi 20) (the lift stops at 20 Hz)."""
    from scipy.signal import bilinear, lfilter
    if not distance or distance <= 0.0:
        return y
    b, a = bilinear([1.0, C_AIR / distance], [1.0, 2.0 * math.pi * 20.0], sr)
    g = 2.0 * math.pi * 20.0 / (C_AIR / distance)
    return lfilter(b * g, a, y, axis=0) if g < 1.0 else y


def point_force_pressure(force, sr, mass_area, distance=1.0):
    """On-axis pressure (Pa) of an infinite plate of `mass_area` kg/m^2 driven by a point force (N per sample):
    p = rho0 F / (2 pi r m) above the fluid-loading corner rho0 c / (2 pi m) Hz (65.7 Hz for 1 kg/m^2), rising
    as f below it: H(s) = s / (s + rho0 c / m)."""
    from scipy.signal import bilinear, lfilter
    b, a = bilinear([1.0, 0.0], [1.0, RHO_AIR * C_AIR / mass_area], sr)
    return RHO_AIR / (2.0 * math.pi * distance * mass_area) * lfilter(b, a, np.asarray(force, float))


@drum("bodhran", "Bodhran (Irish frame drum): a slack goatskin head with no shell cavity, played with a tipper; `pos` "
      "moves the stroke from the centre (deep thud) to the rim (ringing overtones), `damp` is the left hand pressing "
      "the back of the skin (shorter, duller, a little higher).",
      pos=(0.3, "0 centre .. 1 rim"), damp=(0.0, "0 open .. 1 hand pressed hard"), size=(0.46, "head diameter m"),
      tension=(400.0, "N/m"))
def bodhran(sr=DEFAULT_SR, vel=1.0, pos=0.3, damp=0.0, size=0.46, tension=400.0):
    d = float(np.clip(damp, 0.0, 1.0))
    return membrane(sr, vel, tension * (1.0 + 0.15 * d), size, 1.2 * (1.0 - 0.85 * d) + 0.05, 0.3 + 0.6 * d,
                    float(np.clip(pos, 0.0, 0.95)), 0.55, 0.0, 0.1, 0.0, False, 0.0, 1.0, dur=1.4 - 0.9 * d)
