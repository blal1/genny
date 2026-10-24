"""Bowed string with elasto-plastic friction after Matusiak & Chatziioannou, "Elasto-plastic friction modeling
toward reconstructing measured bowed-string transients", JASA 156(2), 1135-1147 (2024).

Friction (their eqs 7-13): f = s0 z + s1 dz/dt, dz/dt = v (1 - a(v, z) z / z_ss(v)), v the string velocity
relative to the bow; adhesion map a = 0 below the breakaway z_ba = 0.7 mu_C f_N / s0, 1 beyond |z_ss|,
(1 + sign(z) sin(pi h)) / 2 between (eqs 9-11), a = 0 when sign(v) != sign(z); steady state
S-curve z_ss = f_N / s0 sign(v) (mu_C + (mu_S - mu_C) exp(-(v / v_S)^2)) (eq. 12) or
eps-curve ... exp(-|v| / eps) (eq. 13). Friction values: their Table III (S- and eps-models fitted to a measured
bridge force); string: their cello G string (Table I), stopped to the asked pitch by shortening it.
String: modal, f_l from T and EI, decay r0 + r1 beta^2 (Table I's damping), each mode advanced exactly over a
step; the bow force is solved implicitly with the string's driving-point response (one scalar equation in v per
step). Output: bridge force T du/dx at x = 0. Left out (they include them): the bow width, the bow-hair
compliance and the torsional waves.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

# Table I: cello G string
L0, RHO, R, T, E = 0.7, 10059.0, 5e-4, 149.6, 1.37e10
SIG0, SIG1 = 1.53714, 0.0087
# Table III
CURVES = {"S": dict(s0=3.1860e5, s1=0.0027, vs=0.2280, eps=0.0, muC=0.5071, muS=1.0207),
          "eps": dict(s0=2.4099e5, s1=0.0115, vs=0.0, eps=0.4, muC=0.3382, muS=1.1489)}


@njit(cache=True)
def _zss(v, fN, s0, vs, eps, muC, muS):
    if v == 0.0:
        return 0.0
    m = muC + (muS - muC) * (math.exp(-(v / vs) ** 2) if vs > 0.0 else math.exp(-abs(v) / eps))
    return fN / s0 * (1.0 if v > 0.0 else -1.0) * m


@njit(cache=True)
def _zdot(v, z, fN, s0, vs, eps, muC, muS):
    zs = _zss(v, fN, s0, vs, eps, muC, muS)
    if zs == 0.0:
        return v
    zba = 0.7 * muC * fN / s0
    a = 0.0
    if (v > 0.0) == (z > 0.0) and z != 0.0:
        az, azs = abs(z), abs(zs)
        if az >= azs:
            a = 1.0
        elif az > zba:
            h = (z - math.copysign(0.5 * (azs + zba), z)) / (azs - zba)
            a = 0.5 * (1.0 + math.copysign(1.0, z) * math.sin(math.pi * h))
    return v * (1.0 - a * z / zs)


@njit(cache=True)
def _resid(v, z, c, mob, dt, fN, s0, s1, vs, eps, muC, muS):
    zd = _zdot(v, z, fN, s0, vs, eps, muC, muS)
    return v + mob * (s0 * (z + dt * zd) + s1 * zd) - c


@njit(cache=True)
def _run(n, A, g, gb, m, fN, vB, dt, s0, s1, vs, eps, muC, muS):
    M = g.size
    q = np.zeros(M)
    w = np.zeros(M)
    qf = np.zeros(M)
    wf = np.zeros(M)
    out = np.zeros(n)
    mob = 0.0
    for l in range(M):
        mob += g[l] * g[l] * A[5, l] / m
    z = 0.0
    v = 0.0
    for i in range(n):
        Vf = 0.0
        for l in range(M):
            qf[l] = A[0, l] * q[l] + A[1, l] * w[l]
            wf[l] = A[2, l] * q[l] + A[3, l] * w[l]
            Vf += g[l] * wf[l]
        c = Vf - vB[i]
        fn = fN[i]
        if fn <= 0.0:                                     # bow lifted: no friction, the bristles relax
            z = 0.0
            Fb = 0.0
            for l in range(M):
                q[l] = qf[l]
                w[l] = wf[l]
                Fb += gb[l] * q[l]
            out[i] = Fb
            continue
        # v + mob f(v) = c, f(v) = s0 (z + dt zdot) + s1 zdot. On the Stribeck branch there can be three roots:
        # scan for sign changes, keep the one nearest the last v (the stick-slip hysteresis), refine by bisection
        span = 1.0 + 2.0 * abs(c)
        best_lo, best_hi, best_d = 0.0, 0.0, 1e300
        x0 = c - span
        r0 = _resid(x0, z, c, mob, dt, fn, s0, s1, vs, eps, muC, muS)
        for k in range(1, 129):
            x1 = c - span + 2.0 * span * k / 128.0
            r1 = _resid(x1, z, c, mob, dt, fn, s0, s1, vs, eps, muC, muS)
            if (r0 <= 0.0) != (r1 <= 0.0):
                d = min(abs(x0 - v), abs(x1 - v))
                if d < best_d:
                    best_lo, best_hi, best_d = x0, x1, d
            x0, r0 = x1, r1
        if best_d < 1e300:
            lo, hi = best_lo, best_hi
            rl = _resid(lo, z, c, mob, dt, fn, s0, s1, vs, eps, muC, muS)
            for it in range(60):
                mid = 0.5 * (lo + hi)
                rm = _resid(mid, z, c, mob, dt, fn, s0, s1, vs, eps, muC, muS)
                if (rm <= 0.0) == (rl <= 0.0):
                    lo, rl = mid, rm
                else:
                    hi = mid
            v = 0.5 * (lo + hi)
        zd = _zdot(v, z, fn, s0, vs, eps, muC, muS)
        f = s0 * (z + dt * zd) + s1 * zd
        z += dt * zd
        Fb = 0.0
        for l in range(M):
            a = -f * g[l] / m
            q[l] = qf[l] + A[4, l] * a
            w[l] = wf[l] + A[5, l] * a
            Fb += gb[l] * q[l]
        out[i] = Fb
    return out


def bowed(freq: float, dur: float, sr: float, *, force: float = 1.5, speed: float = 0.2, accel: float = 1.0,
          beta: float = 0.08, curve: str = "S", release: float = 0.05) -> np.ndarray:
    """Bridge force of the cello G string stopped at ``freq`` (>= 98 Hz), bowed at beta L with normal force
    ``force`` N; the bow starts from rest and accelerates at ``accel`` m/s^2 up to ``speed`` m/s (the
    transients of their Guettler diagram: f_N 0.8-3.9 N, a 0.35-2.1 m/s^2)."""
    P = CURVES[curve]
    mu = RHO * math.pi * R * R
    L = math.sqrt(T / mu) / (2.0 * max(freq, 1.0))
    EI = E * math.pi * R ** 4 / 4.0
    m = mu * L / 2.0
    l = np.arange(1, 400, dtype=float)
    b = l * math.pi / L
    wl = np.sqrt((T * b * b + EI * b ** 4) / mu)
    keep = wl < 2 * math.pi * 0.45 * sr
    b, wl = b[keep], wl[keep]
    zt = SIG0 + SIG1 * b * b
    dt = 1.0 / sr
    wd = np.sqrt(np.maximum(wl * wl - zt * zt, 1e-9))
    e, c, s = np.exp(-zt * dt), np.cos(wd * dt), np.sin(wd * dt)
    A11, A12 = e * (c + zt / wd * s), e * s / wd
    A = np.vstack([A11, A12, -e * wl * wl / wd * s, e * (c - zt / wd * s), (1.0 - A11) / (wl * wl), A12])
    n = int((dur + release + 0.6) * sr)
    t = np.arange(n) / sr
    on = t < dur
    vB = np.where(on, np.minimum(accel * t, speed), 0.0)
    fN = np.where(on, force, 0.0)
    g = np.sin(b * beta * L)
    return _run(n, A, g, T * b, m, fN, vB, dt, P["s0"], P["s1"], P["vs"], P["eps"], P["muC"], P["muS"])
