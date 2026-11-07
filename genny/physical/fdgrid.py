"""Finite-difference grids beyond the rectangle of `plates`.

* polar grids: circular membrane, Kirchhoff plate and shallow spherical shell (von Karman with curvature `q`),
  rim free / clamped / fixed, centre free or clamped, implicit `alpha` weighting factorised once;
* isotropic Cartesian schemes: nine-point Laplacian (`alpha` mix of the 5-point and the diagonal stencil) and the
  implicit theta time weighting, with their exact numerical dispersion;
* sine-transform Poisson / biharmonic solver (Phi = Lap Phi = 0 on the border), also as a numba kernel.

Units are the scaled ones of `plates`: unit radius (polar) or unit area (Cartesian); `kappa` (plate), `gamma`
(membrane) in 1/s; displacement of the shell in units of H / sqrt(6 (1 - nu^2)).
"""
from __future__ import annotations

import math
from functools import lru_cache
from types import SimpleNamespace

import numba
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


# ---------------------------------------------------------------------------------------------------------
# polar grid
# ---------------------------------------------------------------------------------------------------------
def _ix(l, nt):
    return 0 if l == 0 else 1 + (l - 1) * nt


def _ang(nt):
    """Periodic second difference around a ring (no 1/h^2)."""
    return (sp.eye(nt, k=1) + sp.eye(nt, k=-1) + sp.eye(nt, k=nt - 1) + sp.eye(nt, k=1 - nt)
            - 2.0 * sp.eye(nt)).tocsr()


def _sel(l, rin, nt):
    """Rows picking ring l out of a vector holding rings 0..rin."""
    n = 1 + rin * nt
    if l == 0:
        return sp.csr_matrix(([1.0], ([0], [0])), shape=(1, n))
    return sp.csr_matrix((np.ones(nt), (np.arange(nt), _ix(l, nt) + np.arange(nt))), shape=(nt, n))


def _lap(rows, rin, nt, hr, ht):
    """Polar 5-point Laplacian evaluated on the rings `rows` (0 = centre), acting on rings 0..rin.
    Ring l: (1/(l hr^2)) [(l+1/2) u(l+1) - 2l u(l) + (l-1/2) u(l-1)] + (1/(l^2 hr^2 ht^2)) [u(m+1) - 2u(m) + u(m-1)];
    centre: (4/(Nt hr^2)) sum_m (u(1,m) - u00)."""
    D = _ang(nt).tocoo()
    ri, ci, vi = [], [], []
    ro = 0
    m = np.arange(nt)
    for l in rows:
        if l == 0:
            c = 4.0 / (nt * hr * hr)
            ri += [ro] * (nt + 1)
            ci += [0] + list(range(1, nt + 1))
            vi += [-c * nt] + [c] * nt
            ro += 1
            continue
        a = 1.0 / (l * hr * hr)
        for ring, coef in ((l + 1, a * (l + 0.5)), (l, -2.0 / (hr * hr)), (l - 1, a * (l - 0.5))):
            ri += list(ro + m)
            ci += [0] * nt if ring == 0 else list(_ix(ring, nt) + m)
            vi += [coef] * nt
        b = 1.0 / (l * l * hr * hr * ht * ht)
        ri += list(ro + D.row)
        ci += list(_ix(l, nt) + D.col)
        vi += list(b * D.data)
        ro += nt
    return sp.csr_matrix((vi, (ri, ci)), shape=(ro, 1 + rin * nt))


def _expand(nr, nt, hr, ht, edge, centre, nu):
    """E: unknowns -> rings 0..nr+2 (ghost rings included), the unknown rings, and the X index of each unknown.
    edge "free": bending moment and Kirchhoff shear vanish at r = 1 (two ghost rings solved from them);
    "clamped": u = u_r = 0 (mirror ghosts); "fixed": u = 0 (membrane). Centre "clamped": u00 = u(1,m) = 0."""
    lo = 0 if centre == "free" else 2
    hi = nr if edge == "free" else nr - 1
    rings = list(range(lo, hi + 1))
    nx = 1 + (nr + 2) * nt
    cols = np.concatenate([[0] if l == 0 else _ix(l, nt) + np.arange(nt) for l in rings]).astype(np.int64)
    E = np.zeros((nx, cols.size))
    E[cols, np.arange(cols.size)] = 1.0
    if edge == "clamped":
        for g, s in ((nr + 1, nr - 1), (nr + 2, nr - 2)):
            E[_ix(g, nt):_ix(g, nt) + nt] = E[_ix(s, nt):_ix(s, nt) + nt]
    elif edge == "free":
        rin = nr + 2
        S = lambda l: _sel(l, rin, nt)
        A = _ang(nt) / (ht * ht)
        dr = (S(nr + 1) - S(nr - 1)) / (2.0 * hr)
        moment = (S(nr + 1) - 2.0 * S(nr) + S(nr - 1)) / (hr * hr) + nu * (dr + A @ S(nr))
        L = _lap([nr - 1, nr + 1], rin, nt, hr, ht)
        shear = (L[nt:] - L[:nt]) / (2.0 * hr) + (1.0 - nu) * (A @ (dr - S(nr)))
        C = sp.vstack([moment, shear]).toarray()
        g = np.arange(_ix(nr + 1, nt), nx)
        rest = np.arange(_ix(nr + 1, nt))
        E[g] = -np.linalg.solve(C[:, g], C[:, rest]) @ E[rest]
    return sp.csr_matrix(E), rings, cols


def _area(rings, nt, hr, ht, edge, nr):
    a = []
    for l in rings:
        if l == 0:
            a.append([math.pi * (0.5 * hr) ** 2])
        else:
            a.append(np.full(nt, l * hr * hr * ht * (0.5 if (l == nr and edge == "free") else 1.0)))
    return np.concatenate(a)


def _bih(rings, E, nr, nt, hr, ht):
    """Biharmonic = Laplacian of the Laplacian; the centre row is the dedicated formula
    (16/(3 Nt hr^4)) sum_m (u(2,m) - 4 u(1,m) + 3 u00)."""
    big = _lap(range(0, nr + 2), nr + 2, nt, hr, ht)
    small = _lap(rings, nr + 1, nt, hr, ht)
    B = (small @ big @ E).tolil()
    if rings[0] == 0:
        c = 16.0 / (3.0 * nt * hr ** 4)
        row = c * (_sel(2, nr + 2, nt).sum(axis=0) - 4.0 * _sel(1, nr + 2, nt).sum(axis=0))
        row[0, 0] += 3.0 * nt * c
        B[0] = sp.csr_matrix(row) @ E
    return B.tocsr()


@numba.njit(cache=True)
def _g(x, l, m, nt):
    return x[0] if l == 0 else x[1 + (l - 1) * nt + m]


@numba.njit(cache=True)
def _pbracket(a, b, nr, nt, hr, ht, out):
    """Polar von Karman bracket L(a, b) = a_rr (b_r/r + b_tt/r^2) + b_rr (a_r/r + a_tt/r^2)
    - 2 (a_rt/r - a_t/r^2)(b_rt/r - b_t/r^2) on rings 1..nr (a, b hold rings 0..nr+1); centre = mean of ring 1."""
    s = 0.0
    for l in range(1, nr + 1):
        r = l * hr
        for m in range(nt):
            mp, mm = (m + 1) % nt, (m - 1) % nt
            v = np.empty(10)
            for w in range(2):
                x = a if w == 0 else b
                c0, rp, rm = _g(x, l, m, nt), _g(x, l + 1, m, nt), _g(x, l - 1, m, nt)
                tp, tm = _g(x, l, mp, nt), _g(x, l, mm, nt)
                v[5 * w] = (rp - rm) / (2.0 * hr)
                v[5 * w + 1] = (rp - 2.0 * c0 + rm) / (hr * hr)
                v[5 * w + 2] = (tp - tm) / (2.0 * ht)
                v[5 * w + 3] = (tp - 2.0 * c0 + tm) / (ht * ht)
                v[5 * w + 4] = (_g(x, l + 1, mp, nt) - _g(x, l + 1, mm, nt) - _g(x, l - 1, mp, nt)
                                + _g(x, l - 1, mm, nt)) / (4.0 * hr * ht)
            o = (v[1] * (v[5] / r + v[8] / (r * r)) + v[6] * (v[0] / r + v[3] / (r * r))
                 - 2.0 * (v[4] / r - v[2] / (r * r)) * (v[9] / r - v[7] / (r * r)))
            out[1 + (l - 1) * nt + m] = o
            if l == 1:
                s += o
    out[0] = s / nt


def shell_bound(kappa, k, hr, ht, alpha=0.0, q=0.0, centre="free"):
    """Stability number of the polar plate / shell scheme (stable when <= 1):
    2 kappa k sqrt(1 - 4 alpha) (1 + c/ht^2) / (hr^2 sqrt(1 - k^2 kappa^2 q^2 / 4)), c = 1 (free centre) or 1/4
    (clamped centre); alpha >= 1/4 is unconditionally stable in the plate term."""
    d = 1.0 - (k * kappa * q) ** 2 / 4.0
    if d <= 0.0:
        return math.inf
    c = 1.0 if centre == "free" else 0.25
    return 2.0 * kappa * k * math.sqrt(max(0.0, 1.0 - 4.0 * alpha)) * (1.0 + c / ht ** 2) / (hr * hr * math.sqrt(d))


@lru_cache(maxsize=8)
def polar_model(fs, kind="plate", nr=16, nt=32, edge="free", centre="free", kappa=50.0, gamma=0.0, q=0.0,
                alpha=0.2495, sig0=0.0, sig1=0.0, nu=0.3):
    """Polar scheme `M (u+ - 2u + u-) = k^2 (-K u + ...)`, M = I + alpha k^2 K.
    kind "membrane": K = -gamma^2 Lap (rim fixed). "plate": K = kappa^2 BIH. "shell": plate + Airy function,
    BIH Phi = -L(u,u) + q Lap u (Phi = Phi_r = 0 at the rim), u gets -kappa^2 (q Lap Phi - L(Phi, u)).
    The ring count is lowered until the measured spectral radius is stable."""
    k = 1.0 / fs
    if kind == "membrane":
        edge = "fixed"
    while True:
        hr, ht = 1.0 / nr, 2.0 * math.pi / nt
        E, rings, cols = _expand(nr, nt, hr, ht, edge, centre, nu)
        lap = _lap(rings, nr + 2, nt, hr, ht) @ E
        K = (-gamma * gamma * lap) if kind == "membrane" else kappa * kappa * _bih(rings, E, nr, nt, hr, ht)
        n = K.shape[0]
        M = (sp.identity(n) + alpha * k * k * K).tocsc()
        keff = K.toarray()
        ph = None
        if kind == "shell":
            Ep, prings, pcols = _expand(nr, nt, hr, ht, "clamped", "free", nu)
            Bp = _bih(prings, Ep, nr, nt, hr, ht)
            lpu = (_lap(rings, nr + 2, nt, hr, ht) @ Ep).tocsr()
            lup = (_lap(prings, nr + 2, nt, hr, ht) @ E).tocsr()
            ph = SimpleNamespace(E=Ep, cols=pcols, lu=splu(Bp.tocsc()), lpu=lpu, lup=lup)
            keff = keff + kappa * kappa * q * q * (lpu.toarray() @ np.linalg.solve(Bp.toarray(), lup.toarray()))
        lam = np.linalg.eigvals(np.linalg.solve(M.toarray(), keff))
        if k * k * float(np.max(lam.real)) < 3.98:
            break
        if nr <= 5:
            raise ValueError("polar scheme unstable at this rate: raise fs or alpha")
        nr -= 1
    w = np.sort(np.sqrt(np.clip(np.linalg.eigvals(keff).real, 0.0, None)))
    return SimpleNamespace(fs=fs, k=k, kind=kind, nr=nr, nt=nt, hr=hr, ht=ht, edge=edge, centre=centre, E=E,
                           cols=cols, rings=rings, n=n, K=K.tocsr(), lap=lap.tocsr(), M=M.tocsr(),
                           A=splu((M + sig0 * k * sp.identity(n)).tocsc()), area=_area(rings, nt, hr, ht, edge, nr),
                           kappa=kappa, gamma=gamma, q=q, alpha=alpha, sig0=sig0, sig1=sig1, phi=ph,
                           freqs=w / (2.0 * math.pi))


def polar_modes(mdl, n=6, skip_rigid=True):
    """Lowest modal frequencies (Hz) of the polar operator; rigid-body modes of a free plate left out."""
    f = mdl.freqs
    if skip_rigid and mdl.kind != "membrane" and mdl.edge == "free" and mdl.centre == "free":
        f = f[3:]                                      # translation and two tilts
    return f[:n]


def _node(mdl, r, theta):
    """Unknown node nearest to (r, theta) (r in 0..1, theta in turns)."""
    best, d0 = 0, math.inf
    for j, c in enumerate(mdl.cols):
        if c == 0:
            x, y = 0.0, 0.0
        else:
            l, m = (c - 1) // mdl.nt + 1, (c - 1) % mdl.nt
            x, y = l * mdl.hr * math.cos(m * mdl.ht), l * mdl.hr * math.sin(m * mdl.ht)
        d = math.hypot(x - r * math.cos(2 * math.pi * theta), y - r * math.sin(2 * math.pi * theta))
        if d < d0:
            best, d0 = j, d
    return best


def polar_simulate(mdl, force, exc=(0.5, 0.0), pickups=((0.8, 0.31),), nonlinear=True, ulim=1e4):
    """Run the polar scheme for len(force) steps; `force` is the scaled force density amplitude at `exc`
    (r, theta in turns). Returns displacement at the pickups, shape (n, P)."""
    force = np.asarray(force, dtype=np.float64)
    k, n = mdl.k, mdl.n
    e = np.zeros(n)
    je = _node(mdl, *exc)
    e[je] = 1.0 / mdl.area[je]
    pk = np.array([_node(mdl, *p) for p in pickups], dtype=np.int64)
    u, u1 = np.zeros(n), np.zeros(n)
    out = np.zeros((force.size, pk.size))
    ph = mdl.phi
    if ph is not None:
        nx = 1 + (mdl.nr + 2) * mdl.nt
        bu, bp = np.zeros(1 + mdl.nr * mdl.nt), np.zeros(1 + mdl.nr * mdl.nt)
        cu = mdl.cols[mdl.cols < bu.size]
        cp = ph.cols
    c2 = (mdl.kappa * k) ** 2
    for s in range(force.size):
        rhs = mdl.M @ (2.0 * u - u1) + mdl.sig0 * k * u1 + k * k * (force[s] * e - mdl.K @ u) \
            + 2.0 * mdl.sig1 * k * (mdl.lap @ (u - u1))
        if ph is not None:
            xu = mdl.E @ u
            src = mdl.q * (ph.lup @ u)
            if nonlinear:
                _pbracket(xu, xu, mdl.nr, mdl.nt, mdl.hr, mdl.ht, bu)
                src = src - bu[cp]
            phi = ph.lu.solve(src)
            rhs = rhs - c2 * mdl.q * (ph.lpu @ phi)
            if nonlinear:
                xp = ph.E @ phi
                _pbracket(xp, xu, mdl.nr, mdl.nt, mdl.hr, mdl.ht, bp)
                nl = np.zeros(n)
                nl[mdl.cols < bp.size] = bp[cu]
                rhs = rhs + c2 * nl
        un = mdl.A.solve(rhs)
        mx = float(np.max(np.abs(un)))
        if not mx <= ulim:
            un = un * (ulim / mx) if np.isfinite(mx) else np.zeros(n)
        u1, u = u, un
        out[s] = u[pk]
    return out


# ---------------------------------------------------------------------------------------------------------
# isotropic Cartesian schemes
# ---------------------------------------------------------------------------------------------------------
def iso_grid(fs, eps=1.0, gamma=0.0, kappa=0.0, alpha=1.0, theta=1.0):
    """(Nx, Ny, h) of the scheme `delta_tt u = -(theta + (1 - theta) mu_t.) K u` with K = -gamma^2 L_alpha
    (membrane) or kappa^2 L_alpha^2 (supported plate), L_alpha = alpha L_5pt + (1 - alpha) L_diagonal, at its
    stability limit: s_max (2 theta - 1) <= 4 with h^2 s_max = gamma^2 k^2 max(8 alpha, 4) (membrane),
    lambda <= min(1, 1/sqrt(2 alpha)) explicitly. theta <= 1/2 is unconditionally stable: the grid of the
    explicit 5-point (membrane) / 13-point (plate) scheme is then kept."""
    k = 1.0 / fs
    lam = max(8.0 * alpha, 4.0)
    w = 2.0 * theta - 1.0
    if gamma:
        h = gamma * k * math.sqrt(lam * w / 4.0) if w > 0 else math.sqrt(2.0) * gamma * k
    else:
        h = (kappa * k * lam * math.sqrt(w) / 2.0) ** 0.5 if w > 0 else 2.0 * math.sqrt(kappa * k)
    nx, ny = max(3, int(math.sqrt(eps) / h)), max(3, int(1.0 / (math.sqrt(eps) * h)))
    return nx, ny, max(math.sqrt(eps) / nx, 1.0 / (math.sqrt(eps) * ny))


def iso_cents(p, q, fs, eps=1.0, gamma=0.0, kappa=0.0, alpha=1.0, theta=1.0):
    """Frequency of mode (p, q) on the scheme's grid (Hz), the exact frequency of the same grid's domain, and the
    error in cents. Sine products are eigenvectors of both stencils with the edges held."""
    k = 1.0 / fs
    nx, ny, h = iso_grid(fs, eps, gamma, kappa, alpha, theta)
    a, b = p * math.pi / nx, q * math.pi / ny
    l5 = (4.0 * math.sin(a / 2) ** 2 + 4.0 * math.sin(b / 2) ** 2) / h ** 2
    lx = (2.0 - 2.0 * math.cos(a) * math.cos(b)) / h ** 2
    lam = alpha * l5 + (1.0 - alpha) * lx
    s = (gamma * k) ** 2 * lam if gamma else (kappa * k * lam) ** 2
    c = (2.0 - s * theta) / (2.0 + s * (1.0 - theta))
    f = math.acos(max(-1.0, min(1.0, c))) / (2.0 * math.pi * k)
    kx, ky = p / (nx * h), q / (ny * h)
    ex = 0.5 * gamma * math.hypot(kx, ky) if gamma else 0.5 * math.pi * kappa * (kx * kx + ky * ky)
    return f, ex, 1200.0 * math.log2(f / ex)


def iso_operator(nx, ny, h, gamma=0.0, kappa=0.0, alpha=1.0):
    """Sparse K on the (nx-1)(ny-1) interior nodes (edges held / simply supported)."""
    def d2(n):
        return sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n - 1, n - 1))

    def t(n):
        return sp.diags([1.0, 1.0], [-1, 1], shape=(n - 1, n - 1))

    ix, iy = sp.identity(nx - 1), sp.identity(ny - 1)
    l5 = (sp.kron(d2(nx), iy) + sp.kron(ix, d2(ny))) / h ** 2
    lx = (sp.kron(t(nx), t(ny)) - 4.0 * sp.identity((nx - 1) * (ny - 1))) / (2.0 * h ** 2)
    L = (alpha * l5 + (1.0 - alpha) * lx).tocsr()
    return (-(gamma ** 2) * L if gamma else kappa ** 2 * (L @ L)).tocsr()


def iso_simulate(fs, force, eps=1.0, gamma=0.0, kappa=0.0, alpha=1.0, theta=1.0, sig0=0.0, exc=(0.31, 0.43),
                 pick=(0.67, 0.29)):
    """Run the theta scheme (implicit part factorised once); returns displacement at `pick`."""
    k = 1.0 / fs
    nx, ny, h = iso_grid(fs, eps, gamma, kappa, alpha, theta)
    K = iso_operator(nx, ny, h, gamma, kappa, alpha)
    n = K.shape[0]
    I = sp.identity(n)
    M = (I + 0.5 * (1.0 - theta) * k * k * K).tocsr()
    lu = splu((M + sig0 * k * I).tocsc())
    node = lambda x, y: (min(nx - 2, max(0, int(round(x * nx)) - 1))) * (ny - 1) + min(ny - 2, max(0, int(round(y * ny)) - 1))
    je, jp = node(*exc), node(*pick)
    u, u1 = np.zeros(n), np.zeros(n)
    out = np.zeros(len(force))
    for s, f in enumerate(np.asarray(force, dtype=np.float64)):
        rhs = 2.0 * u - theta * k * k * (K @ u) - M @ u1 + sig0 * k * u1
        rhs[je] += k * k * f / (h * h)
        u1, u = u, lu.solve(rhs)
        out[s] = u[jp]
    return out


# ---------------------------------------------------------------------------------------------------------
# sine-transform Poisson / biharmonic
# ---------------------------------------------------------------------------------------------------------
def dst_factors(nx, ny, hx, hy):
    """Orthonormal sine basis along y, S[j, b] = sqrt(2/Ny) sin((j+1)(b+1) pi / Ny), and the eigenvalues of the
    y second difference, (2 cos(b pi / Ny) - 2) / hy^2, for `airy_dst`."""
    j = np.arange(1, ny)
    S = math.sqrt(2.0 / ny) * np.sin(np.outer(j, j) * math.pi / ny)
    return np.ascontiguousarray(S), (2.0 * np.cos(j * math.pi / ny) - 2.0) / hy ** 2


@numba.njit(cache=True)
def _poisson(q, S, lam, hx):
    """Solve Lap u = q (5-point, u = 0 on the border), q of shape (Nx-1, Ny-1): sine transform along y, one
    tridiagonal (Thomas) sweep per sine mode along x."""
    nxm, nym = q.shape
    qh = q @ S
    c = 1.0 / (hx * hx)
    cp = np.empty(nxm)
    d = np.empty(nxm)
    for b in range(nym):
        diag = -2.0 * c + lam[b]
        cp[0] = c / diag
        d[0] = qh[0, b] / diag
        for i in range(1, nxm):
            m = diag - c * cp[i - 1]
            cp[i] = c / m
            d[i] = (qh[i, b] - c * d[i - 1]) / m
        qh[nxm - 1, b] = d[nxm - 1]
        for i in range(nxm - 2, -1, -1):
            qh[i, b] = d[i] - cp[i] * qh[i + 1, b]
    return qh @ S


@numba.njit(cache=True)
def airy_dst(q, S, lam, hx):
    """Biharmonic solve BIH Phi = q as two Poisson solves (Phi = Lap Phi = 0 on the border)."""
    return _poisson(_poisson(q, S, lam, hx), S, lam, hx)


def snare_spacings(k, tension, rho, wire_tension, wire_rho, c=343.0, sig1=0.0, sig1_wire=0.0, sig_air=0.0):
    """Smallest stable grid spacings (m) of a membrane + snare wires + 3-D air scheme at time step k:
    membrane h^2 >= 2 k^2 T/rho + 8 sigma1 k; wire h^2 >= k^2 Ts/rho_s + 4 sigma1_s k; air h^2 >= 3 c^2 k^2 + 6 c sigma_a k.
    T in N/m, rho in kg/m^2 (membrane); Ts in N, rho_s in kg/m (wire)."""
    return (math.sqrt(2.0 * k * k * tension / rho + 8.0 * sig1 * k),
            math.sqrt(k * k * wire_tension / wire_rho + 4.0 * sig1_wire * k),
            math.sqrt(3.0 * c * c * k * k + 6.0 * c * sig_air * k))
