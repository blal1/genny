"""String-bridge-plate after van Walstijn & Mehes, "An explorative string-bridge-plate model with tunable
parameters", DAFx-17 (2017): a stiff string and a simply supported plate, both modal, joined through a bridge mass
by two springs F_l = kL u_l + k+ [u_l]^a - k- [-u_l]^a (eq. 4), u1 = u_b - u_s(z_c), u2 = u_p(x_c, y_c) - u_b.

Parameters as in their Table 1 (normalised: L_s = L_x L_y = 1, rho_s A_s = 0.001 kg/m; eqs 53-58):
- string modes m_s = rho A / 2, k_l = (EI b^4 + T b^2) / 2, b = l pi; T and EI from f_s and B_s;
- plate modes w_ij = kappa (b_xi^2 + b_yj^2), kappa from f_p (eq. 56), L_x / L_y = R_xy, modal mass R_ps m_s;
- damping zeta = sigma_0 + sigma_1 b + sigma_3 b^3 for every mode (eq. 12); bridge mass R_bs m_s, rate zeta_b;
- kL = (1 - eta) k_b, k+-_l = eta k_b G+-_l 10^(4 (alpha - 1)) (eq. 58); gravity g_b pulls the bridge;
- output: plate momentum at the pickup, rho_p h_p sum g_a,l d u_p,l / dt (eq. 16).
Time stepping differs from theirs: each mode is advanced exactly over a step with the connection forces held
(explicit coupling), the bridge mass by symplectic Euler, at 4x the output rate; they solve an implicit,
energy-conserving scheme with an iterative solver. Excitation peak 0.1 N: the peak of their Fig. 6 force signal.
UNSOURCED: the pulse shape and length (raised cosine, 1 ms; Fig. 6 is a smooth three-lobe wavelet of about 34 ms,
Fig. 4 "a short smooth pulse") and the ``flat`` preset's values (their Fig. 3c names only the springs).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

RHO_A = 0.001
PRESETS = {   # Table 1 columns (Fig. 4 with R_bs = 6 and 0.6, Fig. 5 with eta = 1, Fig. 6) and Fig. 3c
    "body": dict(Bs=1e-5, s0=1.0, s1=1e-3, s3=1e-5, sd=500.0, zc=0.87, zd=0.99, zx=0.07, Rbs=6.0, zb=1.0, kb=1e5,
                 eta=0.0, alpha=1.0, gb=0.0, G=(1, 1, 1, 1), fp=17.7, Rxy=0.89, Rps=10.0, p0=20.0, p1=1e-4, p3=1e-6,
                 xc=0.61, yc=0.50, xa=0.13, ya=0.93),
    "coupled": dict(Bs=1e-5, s0=1.0, s1=1e-3, s3=1e-5, sd=500.0, zc=0.87, zd=0.99, zx=0.07, Rbs=0.6, zb=1.0,
                    kb=1e5, eta=0.0, alpha=1.0, gb=0.0, G=(1, 1, 1, 1), fp=17.7, Rxy=0.89, Rps=10.0, p0=20.0,
                    p1=1e-4, p3=1e-6, xc=0.61, yc=0.50, xa=0.13, ya=0.93),
    "plate": dict(Bs=1e-5, s0=2.0, s1=4e-4, s3=4e-6, sd=0.0, zc=0.93, zd=0.99, zx=0.5, Rbs=1e-4, zb=2.0, kb=1e5,
                  eta=1.0, alpha=3.0, gb=0.0, G=(1, 1, 1, 1), fp=50.0, Rxy=0.98, Rps=1.0, p0=2.0, p1=4e-4, p3=4e-6,
                  xc=0.17, yc=0.11, xa=0.13, ya=0.93),
    "rattle": dict(Bs=1e-5, s0=0.5, s1=1e-2, s3=1e-4, sd=0.0, zc=0.98, zd=0.99, zx=0.5, Rbs=1.0, zb=1e-2, kb=1e6,
                   eta=1.0, alpha=1.1, gb=-0.5, G=(1, 0, 1, 0), fp=30.0, Rxy=0.77, Rps=10.0, p0=4.0, p1=1e-2,
                   p3=1e-4, xc=0.61, yc=0.43, xa=0.13, ya=0.93),
    "flat": dict(Bs=1e-5, s0=1.0, s1=1e-3, s3=1e-5, sd=0.0, zc=0.87, zd=0.99, zx=0.07, Rbs=0.6, zb=1.0, kb=1e5,
                 eta=1.0, alpha=1.1, gb=0.0, G=(1, 0, 1, 1), fp=17.7, Rxy=0.89, Rps=10.0, p0=20.0, p1=1e-4, p3=1e-6,
                 xc=0.61, yc=0.50, xa=0.13, ya=0.93),
}


def _modal_step(w, z, dt):
    """Exact transition of q'' + 2 z q' + w^2 q = a over dt with a held: (A11, A12, A21, A22, B1, B2)."""
    wd = np.sqrt(np.maximum(w * w - z * z, 1e-9))
    e, c, s = np.exp(-z * dt), np.cos(wd * dt), np.sin(wd * dt)
    A11, A12 = e * (c + z / wd * s), e * s / wd
    A21, A22 = -e * w * w / wd * s, e * (c - z / wd * s)
    return A11, A12, A21, A22, (1.0 - A11) / (w * w), A12


@njit(cache=True)
def _spring(u, kL, kp, km, al):
    """F(u) of eq. 4 and dF/du."""
    if u > 0.0:
        return kL * u + kp * u ** al, kL + kp * al * u ** (al - 1.0)
    if u < 0.0:
        return kL * u - km * (-u) ** al, kL + km * al * (-u) ** (al - 1.0)
    return 0.0, kL


@njit(cache=True)
def _run(n, os, Fx, sS, sP, gs, gx, gd, gp, ga, ms, mp, mb, rb, rd, kL, kp1, km1, kp2, km2, al, gb):
    """Forces held over each step and evaluated at its end (implicit): every contact displacement is linear in
    the forces through the exact modal transitions, so the step is a 2-unknown Newton solve in (u1, u2)."""
    out = np.zeros(n)
    Ns, Np = gs.size, gp.size
    qs = np.zeros(Ns)
    vs = np.zeros(Ns)
    qp = np.zeros(Np)
    vp = np.zeros(Np)
    ub = 0.0
    vb = 0.0
    dt = sS[6, 0]
    Cs = 0.0
    Dd = 0.0
    for l in range(Ns):
        Cs += gs[l] * gs[l] * sS[4, l] / ms
        Dd += gd[l] * gd[l] * sS[5, l] / ms
    Cp = 0.0
    for l in range(Np):
        Cp += gp[l] * gp[l] * sP[4, l] / mp
    cb = 1.0 + dt * rb / mb
    Cb = dt * dt / mb / cb
    for i in range(n):
        for o in range(os):
            fx = Fx[i * os + o] if i * os + o < Fx.size else 0.0
            # damper: semi-implicit on the damper point's own response
            Vd = 0.0
            for l in range(Ns):
                Vd += gd[l] * (sS[2, l] * qs[l] + sS[3, l] * vs[l] + sS[5, l] * gx[l] * fx / ms)
            Fd = -rd * Vd / (1.0 + rd * Dd)
            # free (force-less) end-of-step positions at the connection points
            Ss = 0.0
            for l in range(Ns):
                Ss += gs[l] * (sS[0, l] * qs[l] + sS[1, l] * vs[l] + sS[4, l] * (gx[l] * fx + gd[l] * Fd) / ms)
            Sp = 0.0
            for l in range(Np):
                Sp += gp[l] * (sP[0, l] * qp[l] + sP[1, l] * vp[l])
            Sb = ub + dt * (vb + dt * gb) / cb
            u1 = Sb - Ss
            u2 = Sp - Sb
            for _ in range(30):
                F1, d1 = _spring(u1, kL, kp1, km1, al)
                F2, d2 = _spring(u2, kL, kp2, km2, al)
                r1 = u1 - (Sb - Ss) + (Cb + Cs) * F1 - Cb * F2
                r2 = u2 - (Sp - Sb) + (Cp + Cb) * F2 - Cb * F1
                a11 = 1.0 + (Cb + Cs) * d1
                a12 = -Cb * d2
                a21 = -Cb * d1
                a22 = 1.0 + (Cp + Cb) * d2
                det = a11 * a22 - a12 * a21
                du1 = (r1 * a22 - r2 * a12) / det
                du2 = (a11 * r2 - a21 * r1) / det
                u1 -= du1
                u2 -= du2
                if abs(du1) + abs(du2) < 1e-15:
                    break
            F1, _d = _spring(u1, kL, kp1, km1, al)
            F2, _d = _spring(u2, kL, kp2, km2, al)
            for l in range(Ns):
                a = (gs[l] * F1 + gx[l] * fx + gd[l] * Fd) / ms
                q, v = qs[l], vs[l]
                qs[l] = sS[0, l] * q + sS[1, l] * v + sS[4, l] * a
                vs[l] = sS[2, l] * q + sS[3, l] * v + sS[5, l] * a
            for l in range(Np):
                a = -gp[l] * F2 / mp
                q, v = qp[l], vp[l]
                qp[l] = sP[0, l] * q + sP[1, l] * v + sP[4, l] * a
                vp[l] = sP[2, l] * q + sP[3, l] * v + sP[5, l] * a
            vb = (vb + dt * ((-F1 + F2) / mb + gb)) / cb
            ub += dt * vb
        acc = 0.0
        for l in range(Np):
            acc += ga[l] * vp[l]
        out[i] = 4.0 * mp * acc
    return out


def render(f_s: float, dur: float, sr: float, preset: str = "body", *, force: float = 0.1, os: int = 4,
           max_plate_modes: int = 1200) -> np.ndarray:
    """Pluck the string (raised-cosine force pulse at z_x) and return the plate momentum at the pickup."""
    P = PRESETS[preset]
    fs = sr * os
    dt = 1.0 / fs
    B = P["Bs"]
    T = 4.0 * f_s * f_s * RHO_A / (1.0 + B)
    EI = B * T / math.pi ** 2
    ms = RHO_A / 2.0
    l = np.arange(1, int(0.45 * sr / f_s) + 2, dtype=float)
    b = l * math.pi
    ws = np.sqrt((EI * b ** 4 + T * b ** 2) / 2.0 / ms)
    keep = ws < 2 * math.pi * 0.45 * sr
    l, b, ws = l[keep], b[keep], ws[keep]
    zs = P["s0"] + P["s1"] * b + P["s3"] * b ** 3
    Lx, Ly = math.sqrt(P["Rxy"]), 1.0 / math.sqrt(P["Rxy"])
    kap = 2.0 * P["fp"] / (math.pi * (Lx ** -2 + Ly ** -2))          # eq. 56
    mx = int(math.sqrt(0.45 * sr * 2 * math.pi / kap) * Lx / math.pi) + 1
    my = int(math.sqrt(0.45 * sr * 2 * math.pi / kap) * Ly / math.pi) + 1
    I, J = np.meshgrid(np.arange(1, mx + 1), np.arange(1, my + 1), indexing="ij")
    bx, by = I.ravel() * math.pi / Lx, J.ravel() * math.pi / Ly
    wp = kap * (bx * bx + by * by)
    sel = np.argsort(wp)[:max_plate_modes]
    sel = sel[wp[sel] < 2 * math.pi * 0.45 * sr]
    bx, by, wp = bx[sel], by[sel], wp[sel]
    bp = np.sqrt(bx * bx + by * by)
    zp = P["p0"] + P["p1"] * bp + P["p3"] * bp ** 3
    mp = P["Rps"] * ms
    mb = P["Rbs"] * ms
    G1p, G1m, G2p, G2m = P["G"]
    a = P["alpha"]
    kn = P["eta"] * P["kb"] * 10.0 ** (4 * (a - 1))
    sS = np.vstack(_modal_step(ws, zs, dt) + (np.full(ws.size, dt),))
    sP = np.vstack(_modal_step(wp, zp, dt) + (np.full(wp.size, dt),))
    n = int(dur * sr)
    m = max(4, int(1e-3 * fs))
    Fx = force * 0.5 * (1.0 - np.cos(2 * np.pi * np.arange(m) / m))
    y = _run(n, os, Fx, sS, sP, np.sin(b * P["zc"]), np.sin(b * P["zx"]), np.sin(b * P["zd"]),
             np.sin(bx * P["xc"] * Lx) * np.sin(by * P["yc"] * Ly), np.sin(bx * P["xa"] * Lx) * np.sin(by * P["ya"] * Ly),
             ms, mp, mb, 2.0 * P["zb"] * mb, 2.0 * P["sd"] * ms, (1.0 - P["eta"]) * P["kb"],
             kn * G1p, kn * G1m, kn * G2p, kn * G2m, a, P["gb"])
    return y
