"""Modal von Karman plate: thin plates and shallow shells reduced to their linear modes.

The transverse modes come from a Rayleigh-Ritz reduction of the plate's strain energy, sampled on a grid, so the
thickness may vary over the plate (D(x, y), rho h(x, y)) and every edge condition of `plates._operators` works
(free, supported, clamped). `basis="ritz"` projects the energy on the analytic sine modes of the supported
rectangle instead of solving on the grid. The in-plane (Airy) modes solve the clamped stress problem
`div div (B div div F) - (1 + nu) L(B, F)` with B = 1 / (E h), whose energy is the plate's with nu -> -nu.
Coupling tensor H^k_pq = int Psi_k L(Phi_p, Phi_q) dS, L the Monge-Ampere bracket.

An imperfection w0 (`dome`, a spherical cap of rise `dome` metres) adds the quadratic terms: with
z = q + a (a = modal coordinates of w0), the Airy amplitudes are

    kappa_k eta_k = -1/2 (z H^k z - a H^k a),    q_s'' + mu_s q_s' + w_s^2 q_s = sum_k eta_k (H^k z)_s + Phi_s(x0) F

i.e. the potential V = 1/8 sum_k (z H^k z - a H^k a)^2 / kappa_k, quartic for a flat plate (hardening) and with
a cubic part for a shell (softening at small amplitude). Two time schemes, both with the exact linear update
(w~^2 = 2 (1 - cos w k) / k^2, no dispersion, no stability limit from the linear part):

* "sav": scalar auxiliary variable psi = sqrt(2 V + eps), explicit, one Sherman-Morrison inversion per step;
* "conservative": the Airy amplitudes are averaged over n+1, n-1 (s_k = z^n H^k z^(n+-1)), linear in q^(n+1),
  solved with a Woodbury identity of rank N_W per step.

Both conserve a discrete energy without losses (returned with `energy=True`).
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numba
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

# mu = 2 xi w = c w^alpha + d (1/s, w in rad/s); `per_mass`: c and d are per unit area (kg/m^2/s), divided by rho h
DAMPING_LAWS = {
    "crash": (0.007, 0.6, 2.0, False),
    "crash_dt": (0.007, 0.7, 0.0, False),
    "gong": (0.005, 0.6, 0.0, False),
    "rectangle": (0.004, 0.75, 0.08, True),
    "gong_hand": (0.04, 0.75, 0.08, True),
}

# measured cymbal thickness profiles: centre thickness hc and edge thickness he (mm), hc held to r_b / R
PROFILES = {
    "uniform": None,
    "nguyen": (1.0, 0.7, 0.25),
    "nguyen_thin": (1.0, 0.4, 0.25),
    "crash": (1.15, 0.7, 0.25),
    "ride": (1.5, 1.1, 0.25),
    "splash": (0.9, 0.6, 0.25),
}


def damping(omega, law="crash", rho_h=1.0):
    """Modal damping mu = 2 xi w (1/s) of a power law; T60 = 13.82 / mu."""
    c, al, d, per_mass = DAMPING_LAWS[law]
    mu = c * np.asarray(omega, float) ** al + d
    return mu / rho_h if per_mass else mu


def thickness(r, profile="nguyen", h=1e-3):
    """Thickness (m) at normalised radius r (0 centre .. 1 rim) of a measured cymbal profile: hc up to r_b,
    then linear to he at the rim. `profile` = name of PROFILES or (hc_mm, he_mm, rb); "uniform" = h."""
    p = PROFILES.get(profile, profile) if isinstance(profile, str) else profile
    r = np.clip(np.asarray(r, float), 0.0, 1.0)
    if p is None:
        return np.full_like(r, h)
    hc, he, rb = p
    return 1e-3 * np.where(r <= rb, hc, hc + (he - hc) * (r - rb) / (1.0 - rb))


def _energy(Nx, Ny, hx, hy, bc, nu, Dn):
    """Strain energy matrix (all nodes) of D (uxx^2 + uyy^2 + 2 nu uxx uyy + 2 (1 - nu) uxy^2), D a node field."""
    def d2(N):
        return sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(N - 1, N + 1))

    def d1(N):
        return sp.diags([-1.0, 1.0], [0, 1], shape=(N, N + 1))

    wx, wy = np.ones(Nx + 1), np.ones(Ny + 1)
    wx[[0, -1]] = wy[[0, -1]] = 0.5
    edge = lambda w: np.where(w < 1.0, 0.5 * (1.0 - nu * nu), 1.0)
    ix, iy = sp.identity(Nx + 1), sp.identity(Ny + 1)
    Ax = sp.kron(d2(Nx), iy) / hx ** 2
    Ay = sp.kron(ix, d2(Ny)) / hy ** 2
    Axy = sp.kron(d1(Nx), d1(Ny)) / (hx * hy)
    Axi = sp.kron(d2(Nx), sp.eye(Ny - 1, Ny + 1, k=1)) / hx ** 2
    Ayi = sp.kron(sp.eye(Nx - 1, Nx + 1, k=1), d2(Ny)) / hy ** 2
    Dc = 0.25 * (Dn[1:, 1:] + Dn[:-1, 1:] + Dn[1:, :-1] + Dn[:-1, :-1])
    Di = Dn[1:-1, 1:-1].ravel()
    E = (Ax.T @ sp.diags(Dn[1:-1, :].ravel() * np.kron(np.ones(Nx - 1), edge(wy))) @ Ax
         + Ay.T @ sp.diags(Dn[:, 1:-1].ravel() * np.kron(edge(wx), np.ones(Ny - 1))) @ Ay
         + nu * (Axi.T @ sp.diags(Di) @ Ayi + Ayi.T @ sp.diags(Di) @ Axi)
         + 2.0 * (1.0 - nu) * Axy.T @ sp.diags(Dc.ravel()) @ Axy)
    clamp = np.zeros((Nx + 1, Ny + 1))
    if bc[0] == "c":
        clamp[1, :] += Dn[1, :] * wy / hx ** 4
    if bc[1] == "c":
        clamp[Nx - 1, :] += Dn[Nx - 1, :] * wy / hx ** 4
    if bc[2] == "c":
        clamp[:, 1] += Dn[:, 1] * wx / hy ** 4
    if bc[3] == "c":
        clamp[:, Ny - 1] += Dn[:, Ny - 1] * wx / hy ** 4
    E = (E + sp.diags(clamp.ravel())) * hx * hy
    keep = np.ones((Nx + 1, Ny + 1), dtype=bool)
    for side, sl in zip(bc, ((0, slice(None)), (Nx, slice(None)), (slice(None), 0), (slice(None), Ny))):
        if side != "f":
            keep[sl] = False
    return E.tocsr(), keep, hx * hy * np.kron(wx, wy)


def _derivs(P, hx, hy):
    """uxx, uyy at interior nodes and the cell cross derivative of a stack of node fields P (n, Nx+1, Ny+1)."""
    xx = (P[:, 2:, 1:-1] - 2.0 * P[:, 1:-1, 1:-1] + P[:, :-2, 1:-1]) / hx ** 2
    yy = (P[:, 1:-1, 2:] - 2.0 * P[:, 1:-1, 1:-1] + P[:, 1:-1, :-2]) / hy ** 2
    xy = (P[:, 1:, 1:] - P[:, 1:, :-1] - P[:, :-1, 1:] + P[:, :-1, :-1]) / (hx * hy)
    return xx, yy, xy


def bracket(a, b, hx, hy):
    """Discrete L(a, b) = a_xx b_yy + a_yy b_xx - 2 a_xy b_xy at the interior nodes of two node fields."""
    (axx, ayy, axy), (bxx, byy, bxy) = _derivs(a[None], hx, hy), _derivs(b[None], hx, hy)
    c = axy[0] * bxy[0]
    return axx[0] * byy[0] + ayy[0] * bxx[0] - 0.5 * (c[1:, 1:] + c[:-1, 1:] + c[1:, :-1] + c[:-1, :-1])


def build(size=0.4, thickness_mm=1.0, eps=1.0, bc="ssss", material=(8800.0, 1.24e11, 0.33), profile="uniform",
          dome=0.0, modes=48, airy=24, grid=36, basis="grid", law="crash"):
    """Modal model of a plate of area (0.886 size)^2, aspect `eps`, edges `bc`, thickness profile (mm or a
    PROFILES name), spherical-cap imperfection of rise `dome` (m). Returns a namespace with w (rad/s), mu (1/s),
    Phi (modes x nodes, mass-normalised), Psi, kappa, H, a (imperfection coordinates) and the grid."""
    rho, E, nu = material
    L = 0.886 * float(size)
    Lx, Ly = L * math.sqrt(eps), L / math.sqrt(eps)
    Nx, Ny = max(8, int(round(grid * math.sqrt(eps)))), max(8, int(round(grid / math.sqrt(eps))))
    hx, hy = Lx / Nx, Ly / Ny
    X, Y = np.meshgrid(np.arange(Nx + 1) * hx - Lx / 2, np.arange(Ny + 1) * hy - Ly / 2, indexing="ij")
    R = math.hypot(Lx, Ly) / 2 if bc != "ffff" else L / math.sqrt(math.pi)
    r = np.hypot(X, Y) / R
    h = thickness(r, profile, float(thickness_mm) * 1e-3)
    Dn = E * h ** 3 / (12.0 * (1.0 - nu * nu))
    En, keep, wq = _energy(Nx, Ny, hx, hy, bc, nu, Dn)
    sel = np.nonzero(keep.ravel())[0]
    K = En[sel][:, sel]
    mass = wq[sel] * rho * h.ravel()[sel]
    nm = int(modes) + (3 if bc == "ffff" else 0)
    if basis == "ritz":
        p, q = np.meshgrid(np.arange(1, nm + 1), np.arange(1, nm + 1), indexing="ij")
        order = np.argsort(((p / Lx) ** 2 + (q / Ly) ** 2).ravel())[:2 * nm]
        xs, ys = X.ravel()[sel] + Lx / 2, Y.ravel()[sel] + Ly / 2
        S = np.stack([np.sin(math.pi * p.ravel()[o] * xs / Lx) * np.sin(math.pi * q.ravel()[o] * ys / Ly)
                      for o in order], axis=1)
        lam, C = eigh(S.T @ (K @ S), (S * mass[:, None]).T @ S)
        lam, V = lam[:nm], S @ C[:, :nm]
    else:
        lam, V = eigsh(K, k=min(nm, K.shape[0] - 2), M=sp.diags(mass), sigma=-1.0)
    o = np.argsort(lam)
    lam, V = lam[o], V[:, o]
    ok = lam > (2.0 * math.pi) ** 2
    lam, V = lam[ok][:modes], V[:, ok][:, :modes]
    Phi = np.zeros((V.shape[1], (Nx + 1) * (Ny + 1)))
    Phi[:, sel] = V.T
    Phi = Phi.reshape(-1, Nx + 1, Ny + 1)
    Ew, keepw, _ = _energy(Nx, Ny, hx, hy, "cccc", -nu, 1.0 / (E * h))
    selw = np.nonzero(keepw.ravel())[0]
    kap, W = eigsh(Ew[selw][:, selw], k=int(airy), M=sp.diags(wq[selw]), sigma=0.0)
    Psi = np.zeros((W.shape[1], (Nx + 1) * (Ny + 1)))
    Psi[:, selw] = W.T
    Psi = Psi.reshape(-1, Nx + 1, Ny + 1)[:, 1:-1, 1:-1]
    xx, yy, xy = _derivs(Phi, hx, hy)
    cell = hx * hy
    H = np.zeros((Psi.shape[0], Phi.shape[0], Phi.shape[0]))
    Pf = Psi.reshape(Psi.shape[0], -1) * cell
    for i in range(Phi.shape[0]):
        c = xy[i][None] * xy[i:]
        Lb = (xx[i][None] * yy[i:] + yy[i][None] * xx[i:] - 0.5 * (c[:, 1:, 1:] + c[:, :-1, 1:] + c[:, 1:, :-1]
                                                                  + c[:, :-1, :-1]))
        H[:, i, i:] = Pf @ Lb.reshape(Lb.shape[0], -1).T
        H[:, i:, i] = H[:, i, i:]
    a = np.zeros(Phi.shape[0])
    if dome > 0.0:
        w0 = float(dome) * (1.0 - np.clip(r, 0.0, 1.0) ** 2)
        w0 -= np.sum(w0 * wq.reshape(w0.shape)) / np.sum(wq)
        m = (rho * h * wq.reshape(h.shape))
        a = np.array([np.sum(m * w0 * P) for P in Phi])
    rho_h = rho * float(np.mean(h))
    w = np.sqrt(lam)
    return SimpleNamespace(w=w, mu=damping(w, law, rho_h) if law else np.zeros_like(w), Phi=Phi, Psi=Psi,
                           kappa=kap, H=H, a=a, Nx=Nx, Ny=Ny, hx=hx, hy=hy, Lx=Lx, Ly=Ly, h=h, rho=rho, E=E, nu=nu)


def linear_freqs(m):
    """Small-amplitude frequencies (Hz) about the imperfect shape: w^2 + sum_k (H^k a)(H^k a)^T / kappa_k."""
    G = np.einsum("kpq,q->kp", m.H, m.a)
    A = np.diag(m.w ** 2) + (G.T / m.kappa) @ G
    return np.sqrt(np.clip(np.linalg.eigvalsh(0.5 * (A + A.T)), 0.0, None)) / (2.0 * math.pi)


def node(m, x, y):
    """Grid index (i, j) nearest to fractions x, y of the two sides."""
    return int(round(np.clip(x, 0.0, 1.0) * m.Nx)), int(round(np.clip(y, 0.0, 1.0) * m.Ny))


@numba.njit(cache=True)
def _run(H, kap, ca, a, om2, mu, pin, pout, force, k, scheme, out, en, eps):
    NW, NU = H.shape[0], H.shape[1]
    q, qm, qp = np.zeros(NU), np.zeros(NU), np.zeros(NU)
    z, r, g, y, Dv = np.empty(NU), np.empty(NU), np.empty(NU), np.empty(NU), np.empty(NU)
    Hz, s = np.empty((NW, NU)), np.empty(NW)
    for i in range(NU):
        Dv[i] = 1.0 / (k * k) + mu[i] / (2.0 * k)
    psi = -1.0
    for n in range(out.shape[0]):
        for i in range(NU):
            z[i] = q[i] + a[i]
        for kk in range(NW):
            sk = 0.0
            for i in range(NU):
                acc = 0.0
                for j in range(NU):
                    acc += H[kk, i, j] * z[j]
                Hz[kk, i] = acc
                sk += acc * z[i]
            s[kk] = sk - ca[kk]
        for i in range(NU):
            r[i] = (2.0 * q[i] - qm[i]) / (k * k) - om2[i] * q[i] + mu[i] * qm[i] / (2.0 * k) + pin[i] * force[n]
        if scheme == 0:
            V = 0.0
            for kk in range(NW):
                V += s[kk] * s[kk] / kap[kk]
            V *= 0.125
            sq = math.sqrt(2.0 * V + eps)
            if psi < 0.0:
                psi = sq
            gq = 0.0
            for i in range(NU):
                acc = 0.0
                for kk in range(NW):
                    acc += s[kk] / kap[kk] * Hz[kk, i]
                g[i] = 0.5 * acc / sq
                gq += g[i] * qm[i]
            gy, gdg = 0.0, 0.0
            for i in range(NU):
                y[i] = (r[i] - g[i] * psi + 0.25 * g[i] * gq) / Dv[i]
                gy += g[i] * y[i]
                gdg += g[i] * g[i] / Dv[i]
            c = gy / (4.0 + gdg)
            dq = 0.0
            for i in range(NU):
                qp[i] = y[i] - g[i] / Dv[i] * c
                dq += g[i] * (qp[i] - qm[i])
            psi += 0.5 * dq
        elif scheme == 1:
            S = np.zeros((NW, NW))
            b = np.zeros(NW)
            for kk in range(NW):
                acc = 0.0
                for i in range(NU):
                    acc += Hz[kk, i] * (qm[i] + 2.0 * a[i])
                b[kk] = acc - 2.0 * ca[kk]
            for i in range(NU):
                acc = 0.0
                for kk in range(NW):
                    acc += Hz[kk, i] * b[kk] / kap[kk]
                r[i] -= 0.25 * acc
                y[i] = r[i] / Dv[i]
            u = np.zeros(NW)
            for kk in range(NW):
                S[kk, kk] = 4.0 * kap[kk]
                acc = 0.0
                for i in range(NU):
                    acc += Hz[kk, i] * y[i]
                u[kk] = acc
                for ll in range(kk, NW):
                    acc = 0.0
                    for i in range(NU):
                        acc += Hz[kk, i] * Hz[ll, i] / Dv[i]
                    S[kk, ll] += acc
                    if ll != kk:
                        S[ll, kk] += acc
            v = np.linalg.solve(S, u)
            for i in range(NU):
                acc = 0.0
                for kk in range(NW):
                    acc += Hz[kk, i] * v[kk]
                qp[i] = y[i] - acc / Dv[i]
        else:
            for i in range(NU):
                qp[i] = r[i] / Dv[i]
        if en.shape[0] > 0:
            e = 0.0
            for i in range(NU):
                d = (qp[i] - q[i]) / k
                e += 0.5 * d * d + 0.5 * om2[i] * qp[i] * q[i]
            if scheme == 0:
                e += 0.5 * psi * psi
            elif scheme == 1:
                for kk in range(NW):
                    acc = 0.0
                    for i in range(NU):
                        acc += Hz[kk, i] * (qp[i] + a[i])
                    acc -= ca[kk]
                    e += 0.125 * acc * acc / kap[kk]
            en[n] = e
        for i in range(NU):
            qm[i] = q[i]
            q[i] = qp[i]
        for p in range(pout.shape[0]):
            acc = 0.0
            for i in range(NU):
                acc += pout[p, i] * q[i]
            out[n, p] = acc
    return out


def simulate(m, force, fs, exc=(0.4, 0.45), pickups=((0.63, 0.41),), scheme="sav", energy=False, lossless=False):
    """Run the modal model with `force` (N per sample at rate fs) at the strike point. Returns displacement (m)
    at the pickups, shape (n, P), and the discrete energy per step if `energy`."""
    k = 1.0 / fs
    om2 = 2.0 * (1.0 - np.cos(m.w * k)) / (k * k)
    mu = np.zeros_like(m.mu) if lossless else m.mu
    pin = m.Phi[:, node(m, *exc)[0], node(m, *exc)[1]].copy()
    pout = np.array([m.Phi[:, node(m, *p)[0], node(m, *p)[1]] for p in pickups])
    ca = np.einsum("p,kpq,q->k", m.a, m.H, m.a)
    force = np.ascontiguousarray(force, dtype=np.float64)
    out = np.zeros((force.shape[0], len(pickups)))
    en = np.zeros(force.shape[0] if energy else 0)
    sid = {"sav": 0, "conservative": 1}.get(scheme, 2)
    _run(np.ascontiguousarray(m.H), m.kappa.copy(), ca, m.a.copy(), om2, mu.copy(), pin, pout, force, k, sid, out,
         en, 1e-24)
    return (out, en) if energy else out


def instantaneous_freq(x, fs, f0, bw=0.25, hop=0.02):
    """Frequency track (Hz) of the partial near f0: peak of a Hann STFT in [f0 (1 - bw), f0 (1 + bw)]."""
    n = int(0.08 * fs)
    win = np.hanning(n)
    nf = 1 << (4 * n - 1).bit_length()
    fr = np.fft.rfftfreq(nf, 1.0 / fs)
    band = (fr > f0 * (1 - bw)) & (fr < f0 * (1 + bw))
    out = []
    for s in range(0, len(x) - n, int(hop * fs)):
        sp_ = np.abs(np.fft.rfft(x[s:s + n] * win, nf))[band]
        i = int(np.argmax(sp_))
        if 0 < i < len(sp_) - 1:
            a_, b_, c_ = np.log(sp_[i - 1:i + 2] + 1e-30)
            i = i + 0.5 * (a_ - c_) / (a_ - 2 * b_ + c_)
        out.append(fr[band][0] + i * (fr[1] - fr[0]))
    return np.array(out)


def correlation_dimension(x, dim=8, lag=4, n=1500, radii=None):
    """Grassberger-Procaccia correlation dimension of a signal embedded in `dim` dimensions: slope of
    log C(r) against log r over the middle of the radius range."""
    x = np.asarray(x, float)
    x = x / (np.std(x) + 1e-30)
    m = len(x) - (dim - 1) * lag
    idx = np.linspace(0, m - 1, min(n, m)).astype(int)
    P = np.stack([x[idx + j * lag] for j in range(dim)], axis=1)
    d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))[np.triu_indices(len(P), 1)]
    radii = np.geomspace(np.percentile(d, 1), np.percentile(d, 30), 8) if radii is None else radii
    C = np.array([np.mean(d < r) for r in radii])
    return float(np.polyfit(np.log(radii), np.log(C + 1e-30), 1)[0])


def lyapunov(m, force, fs, exc=(0.4, 0.45), d0=1e-9, every=64, scheme="sav"):
    """Largest Lyapunov exponent (1/s) of the lossless modal plate after the strike: growth rate of the distance
    between two runs whose states differ by a tiny kick after the strike, fitted while it stays small."""
    from copy import copy
    n = len(force)
    k = 1.0 / fs
    m2 = copy(m)
    m2.mu = np.zeros_like(m.mu)
    base = simulate(m2, force, fs, exc, pickups=((0.31, 0.27), (0.63, 0.41), (0.77, 0.69)), scheme=scheme,
                    lossless=True)
    f2 = force.copy()
    i0 = int(np.nonzero(force)[0][-1]) + 1 if np.any(force) else 0
    f2[i0:i0 + 1] += d0 / (k * k * float(np.max(np.abs(m.Phi))) + 1e-30)
    pert = simulate(m2, f2, fs, exc, pickups=((0.31, 0.27), (0.63, 0.41), (0.77, 0.69)), scheme=scheme,
                    lossless=True)
    dist = np.linalg.norm(pert - base, axis=1)[i0 + every:]
    if dist.size < 2 * every or not np.all(np.isfinite(dist)):
        return float("nan")
    t = np.arange(dist.size) * k
    grow = dist < 1e-3 * np.max(np.abs(base)) + 1e-30
    if grow.sum() < every:
        grow[:every] = True
    return float(np.polyfit(t[grow], np.log(dist[grow] + 1e-300), 1)[0])
