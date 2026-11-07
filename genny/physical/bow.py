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
step). Output: bridge force T du/dx at x = 0. Left out (they include them): the bow width and the bow-hair
compliance.

Torsion (``torsion`` = Y'/Y): McIntyre, Schumacher & Woodhouse 1983, Sec. II-D: the torsional impedance shears the friction
curve, G(q) = F(q + Y' f / 2). Measured Y'/Y 0.26-1, their Fig. 10 uses 0.4, Fig. 14 uses 0.2. Here a second modal series
at the contact, silent at the bridge, with c_R = sqrt(K_T / P_T) and Q_T = 30 (van Walstijn et al. 2026, Table 1).

Thermal friction (``curve="thermal"``): van Walstijn, Chatziioannou, Lampis & Matusiak, Acta Acustica 10, 47 (2026):
z_ss = y(tau) mu_str(v) / sigma0 (eqs 9-10), y and mu_ss eqs 30-32, heat eq. 22 (a Fb dtau/dt + [b sqrt(|v| / Fb) + c] Fb
tau = Fb sigma0 z s, s = v - dz/dt, eqs 20-21), mu_str(v) from the steady balance eq. 28, Table 1 values. Their
contact has N points over Wb; here one point, Fb = force / Wb.
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
# Violin: mu_s ~ 0.4-0.5, mu_d ~ 0.2 (Serafin, Avanzini & Rocchesso, DAFx-02, Sec. 4). the bristle stiffness and
# damping carried over from the cello's S fit, and the whole A string (steel 0.157 mm radius, 50 N, the cello's damping)
CURVES["violin"] = dict(s0=3.186e5, s1=0.0027, vs=0.228, eps=0.0, muC=0.2, muS=0.45)
VIOLIN_A = (7800.0, 1.57e-4, 50.0, 2.0e11, SIG0, SIG1)
# Tables IV-V, the Guettler cases they fitted: (a m/s^2, f_N N, s0, s1, v_S or eps, mu_C, mu_S)
FITS = {"S": ((0.8722, 2.3433, 3.186e5, 0.0027, 0.228, 0.5071, 1.0207), (0.3473, 0.7992, 3.706e5, 0.0082, 0.0513, 0.7727, 1.0902),
              (1.0481, 2.0791, 3.332e5, 0.05, 0.1603, 0.7722, 1.17), (2.0651, 3.9125, 3.0267e5, 0.0026, 0.228, 0.5071, 1.0207)),
        "eps": ((0.8722, 2.3433, 2.409e5, 0.0115, 0.4, 0.3382, 1.1489), (0.3473, 0.7992, 2.597e5, 0.0034, 0.1, 0.6951, 1.0925),
                (1.0481, 2.0791, 1.897e5, 0.00074, 0.4, 0.5308, 1.1725), (2.0651, 3.9125, 3.3147e5, 0.0228, 0.4, 0.3365, 0.9116))}
# van Walstijn et al. 2026, Table 1 (their friction and heat values are "empirically chosen within a physically
# plausible range"); Wb 0.01 m (Sec. 2.2.3)
KT, PT, QT = 2.9e-4, 4.2e-10, 30.0
TH = dict(mus=1.05, mud=0.3, sigma0=1.0e6, taug=25.0, ya=0.4, xi=2.0, chi=0.5, a=1.0e-6, b=0.22, c=1.0e-4, Wb=0.01)


@njit(cache=True)
def _zss2(v, fN, s0, vs, eps, muC, muS, y, vg, mg):
    if v == 0.0:
        return 0.0
    if vg.size:
        m = y * np.interp(abs(v), vg, mg)                   # thermal: y(tau) mu_str(v)
    else:
        m = muC + (muS - muC) * (math.exp(-(v / vs) ** 2) if vs > 0.0 else math.exp(-abs(v) / eps))
    return fN / s0 * (1.0 if v > 0.0 else -1.0) * m


@njit(cache=True)
def _zss(v, fN, s0, vs, eps, muC, muS):
    return _zss2(v, fN, s0, vs, eps, muC, muS, 1.0, np.empty(0), np.empty(0))


@njit(cache=True)
def _zdot2(v, z, fN, s0, vs, eps, muC, muS, y, chi, vg, mg):
    zs = _zss2(v, fN, s0, vs, eps, muC, muS, y, vg, mg)
    if zs == 0.0:
        return v
    zba = chi * y * muC * fN / s0
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
def _zdot(v, z, fN, s0, vs, eps, muC, muS):
    return _zdot2(v, z, fN, s0, vs, eps, muC, muS, 1.0, 0.7, np.empty(0), np.empty(0))


@njit(cache=True)
def _znext(v, z, dt, fN, s0, vs, eps, muC, muS, y, chi, vg, mg):
    """z after one step at velocity v, the adhesion map held: dz/dt = v - (v a / z_ss) z solved exactly, because at
    these stiffnesses one Euler step of sliding can exceed z_ss many times over."""
    zs = _zss2(v, fN, s0, vs, eps, muC, muS, y, vg, mg)
    if zs == 0.0:
        return z
    zd = _zdot2(v, z, fN, s0, vs, eps, muC, muS, y, chi, vg, mg)
    if zd == v:                                             # a = 0
        return z + dt * v
    a = (v - zd) * zs / (v * z)
    return zs / a + (z - zs / a) * math.exp(-abs(v) * a / abs(zs) * dt)


@njit(cache=True)
def _resid(v, z, c, mob, dt, fN, s0, s1, vs, eps, muC, muS, y, chi, vg, mg):
    zn = _znext(v, z, dt, fN, s0, vs, eps, muC, muS, y, chi, vg, mg)
    return v + mob * (s0 * zn + s1 * (zn - z) / dt) - c


@njit(cache=True)
def _run(n, A, g, gb, m, fN, vB, dt, s0, s1, vs, eps, muC, muS, chi, vg, mg, th, Kh=0.0, Dh=0.0):
    """th = (sigma0, taug, ya, xi, a, b, c, Wb), used with a non-empty vg; s0 is then fN * sigma0 at every step.
    Returns the bridge force and the contact temperature above ambient."""
    M = g.size
    q = np.zeros(M)
    w = np.zeros(M)
    qf = np.zeros(M)
    wf = np.zeros(M)
    out = np.zeros(n)
    temp = np.zeros(n)
    mob = 0.0
    for l in range(M):
        mob += g[l] * g[l] * A[5, l] / m
    z = 0.0
    v = 0.0
    tau = 0.0
    thermal = vg.size > 0 and th[0] > 0.0
    sig0, taug, ya, xi, ath, bth, cth, Wb = th
    eta = 0.0
    etad = 0.0
    for i in range(n):
        Vf = 0.0
        for l in range(M):
            qf[l] = A[0, l] * q[l] + A[1, l] * w[l]
            wf[l] = A[2, l] * q[l] + A[3, l] * w[l]
            Vf += g[l] * wf[l]
        c = Vf - (vB[i] - etad)
        fn = fN[i]
        y = 1.0
        if thermal:
            x = (tau / taug) ** xi
            y = (1.0 + ya * x) / (1.0 + x)
            s0 = fn * sig0 if fn > 0.0 else 1.0
        if fn <= 0.0:                                     # bow lifted: no friction, the bristles relax
            z = 0.0
            Fb = 0.0
            for l in range(M):
                q[l] = qf[l]
                w[l] = wf[l]
                Fb += gb[l] * q[l]
            out[i] = Fb
            if thermal:
                tau /= 1.0 + dt * cth / ath
            temp[i] = tau
            continue
        # v + mob f(v) = c, f(v) = s0 (z + dt zdot) + s1 zdot. On the Stribeck branch there can be three roots:
        # scan for sign changes, keep the one nearest the last v (the stick-slip hysteresis), refine by bisection
        span = 1.0 + 2.0 * abs(c)
        best_lo, best_hi, best_d = 0.0, 0.0, 1e300
        x0 = c - span
        r0 = _resid(x0, z, c, mob, dt, fn, s0, s1, vs, eps, muC, muS, y, chi, vg, mg)
        for k in range(1, 129):
            x1 = c - span + 2.0 * span * k / 128.0
            r1 = _resid(x1, z, c, mob, dt, fn, s0, s1, vs, eps, muC, muS, y, chi, vg, mg)
            if (r0 <= 0.0) != (r1 <= 0.0):
                d = min(abs(x0 - v), abs(x1 - v))
                if d < best_d:
                    best_lo, best_hi, best_d = x0, x1, d
            x0, r0 = x1, r1
        if best_d < 1e300:
            lo, hi = best_lo, best_hi
            rl = _resid(lo, z, c, mob, dt, fn, s0, s1, vs, eps, muC, muS, y, chi, vg, mg)
            for it in range(60):
                mid = 0.5 * (lo + hi)
                rm = _resid(mid, z, c, mob, dt, fn, s0, s1, vs, eps, muC, muS, y, chi, vg, mg)
                if (rm <= 0.0) == (rl <= 0.0):
                    lo, rl = mid, rm
                else:
                    hi = mid
            v = 0.5 * (lo + hi)
        zn = _znext(v, z, dt, fn, s0, vs, eps, muC, muS, y, chi, vg, mg)
        zd = (zn - z) / dt
        f = s0 * zn + s1 * zd
        z = zn
        if Dh > 0.0:                                      # the hair ribbon yields: K eta + D eta_t = f
            etad = (f - Kh * eta) / Dh
            eta += dt * etad
        if thermal:
            loss = (bth * math.sqrt(abs(v) * Wb / fn) + cth) / ath   # Fb = fn / Wb
            tau = (tau + dt * sig0 * z * (v - zd) / ath) / (1.0 + dt * loss)
            temp[i] = tau
        Fb = 0.0
        for l in range(M):
            a = -f * g[l] / m
            q[l] = qf[l] + A[4, l] * a
            w[l] = wf[l] + A[5, l] * a
            Fb += gb[l] * q[l]
        out[i] = Fb
    return out, temp


def mu_str_table(Fb: float):
    """(|v| grid, mu_str) of the thermal model at Fb N/m: steady sliding balances heat at tau(v), eq. 28,
    [b sqrt(|v| / Fb) + c] tau = mu_ss(tau) |v|, and mu_str = mu_ss / y (eqs 30-32)."""
    P = TH
    vg = np.geomspace(1e-6, 10.0, 200)
    mg = np.zeros(vg.size)
    for i, v in enumerate(vg):
        k = P["b"] * math.sqrt(v / Fb) + P["c"]
        lo, hi = 0.0, 1e4
        for _ in range(60):
            t = 0.5 * (lo + hi)
            x = (t / P["taug"]) ** P["xi"]
            if k * t < (P["mus"] + P["mud"] * P["ya"] * x) / (1.0 + x) * v:
                lo = t
            else:
                hi = t
        x = (lo / P["taug"]) ** P["xi"]
        mg[i] = (P["mus"] + P["mud"] * P["ya"] * x) / (1.0 + P["ya"] * x)
    return vg, mg


EXP_MU = (0.35, 0.4, 0.01, 0.45, 0.1)       # mu_d + a1 exp(-|v|/v1) + a2 exp(-|v|/v2)
HYP_MU = (0.3, 0.8, 0.1)                     # mu_d, mu_s, v0: mu_d + (mu_s - mu_d) v0 / (v0 + |v|)


def mu_curve_table(curve: str):
    """(|v| grid, mu) of the static friction curves ``exponential`` and ``hyperbolic``."""
    vg = np.geomspace(1e-6, 10.0, 200)
    if curve == "exponential":
        d, a1, v1, a2, v2 = EXP_MU
        return vg, d + a1 * np.exp(-vg / v1) + a2 * np.exp(-vg / v2)
    d, s, v0 = HYP_MU
    return vg, d + (s - d) * v0 / (v0 + vg)


HAIR_K, HAIR_D = 4.8297e6, 5.7674e3       # bow hair stiffness N/m^2 and damping kg/(s m), per metre of ribbon
WOLF_Q = 30.0
WOLF_RATIO = 64.0 / 65.0
BOW_WIDTH = 0.01                            # m of hair ribbon on the string


def _nearest_fit(curve, accel, force):
    """The fitted case nearest (a, f_N), each axis scaled by its range in Tables IV-V."""
    rows = FITS[curve]
    r = min(rows, key=lambda q: ((q[0] - accel) / 1.7) ** 2 + ((q[1] - force) / 3.1) ** 2)
    vs, eps = (r[4], 0.0) if curve == "S" else (0.0, r[4])
    return dict(s0=r[2], s1=r[3], vs=vs, eps=eps, muC=r[5], muS=r[6])


def bowed(freq: float, dur: float, sr: float, *, force: float = 1.5, speed: float = 0.2, accel: float = 1.0,
          beta: float = 0.08, curve: str = "S", release: float = 0.05, torsion: float = 0.0, fit: bool = False,
          string: tuple | None = None, want_temperature: bool = False, width: float = 0.0, hair: bool = False,
          wolf: float = 0.0, wolf_hz: float | None = None, wolf_q: float = WOLF_Q, max_hz: float | None = None):
    """Bridge force of the cello G string stopped at ``freq`` (>= 98 Hz), bowed at beta L with normal force
    ``force`` N; the bow starts from rest and accelerates at ``accel`` m/s^2 up to ``speed`` m/s (the
    transients of their Guettler diagram: f_N 0.8-3.9 N, a 0.35-2.1 m/s^2). ``curve``: S, eps, thermal, or the static curves
    exponential and hyperbolic (bristle stiffness of S).
    ``fit``: S and eps only, take the Table IV-V row nearest (accel, force) instead of the red case. ``torsion``: Y'/Y.
    ``string``: (rho kg/m3, r m, T N, E Pa, sigma0 1/s, sigma1 m2/s) instead of the cello G.
    ``width``: m of bow hair in contact (0 = a point): each mode is driven by its shape averaged over the ribbon,
    sin(k x) sin(k w / 2) / (k w / 2). ``hair``: the ribbon is compliant, f = -K eta - D eta_t with HAIR_K, HAIR_D per
    metre of ribbon times the width (BOW_WIDTH when ``width`` is 0). ``wolf``: a body resonance seen from the bow, its
    mobility `wolf` x that of the string's modes together (0.01 a real wolf), at ``wolf_hz`` (default
    freq x WOLF_RATIO, just below the note) with quality ``wolf_q``: near it the Helmholtz motion stutters.
    ``max_hz``: highest string mode kept (default 0.45 sr), so that an oversampled render (a high note, whose
    stick-slip would otherwise snap to whole samples) costs only its extra samples."""
    if curve == "thermal":
        P = dict(s0=1.0, s1=0.0, vs=0.0, eps=1.0, muC=TH["mud"], muS=TH["mus"])
        chi = TH["chi"]
        vg, mg = mu_str_table(force / TH["Wb"])
    elif curve in ("exponential", "hyperbolic"):
        vg, mg = mu_curve_table(curve)
        P = dict(CURVES["S"], muC=float(mg[-1]), muS=float(mg[0]))
        chi = 0.7
    else:
        P = _nearest_fit(curve, accel, force) if fit else CURVES[curve]
        chi = 0.7
        vg, mg = np.empty(0), np.empty(0)
    th = (TH["sigma0"] if curve == "thermal" else 0.0, TH["taug"], TH["ya"], TH["xi"], TH["a"], TH["b"], TH["c"], TH["Wb"])
    rho, Rs, Ts, Es, sig0, sig1 = string if string is not None else (RHO, R, T, E, SIG0, SIG1)
    mu = rho * math.pi * Rs * Rs
    L = math.sqrt(Ts / mu) / (2.0 * max(freq, 1.0))
    EI = Es * math.pi * Rs ** 4 / 4.0
    m = mu * L / 2.0
    l = np.arange(1, 400, dtype=float)
    b = l * math.pi / L
    wl = np.sqrt((Ts * b * b + EI * b ** 4) / mu)
    keep = wl < 2 * math.pi * min(0.45 * sr, float(max_hz) if max_hz else 0.45 * sr)
    b, wl = b[keep], wl[keep]
    zt = sig0 + sig1 * b * b
    g = np.sin(b * beta * L)
    if width > 0.0:
        g = g * np.sinc(b * float(width) / (2.0 * math.pi))
    gb = Ts * b
    if torsion > 0.0:
        wt = math.pi * math.sqrt(KT / PT) / L * np.arange(1, 400, dtype=float)
        wt = wt[wt < 2 * math.pi * 0.45 * sr]
        lt = wt / (math.pi * math.sqrt(KT / PT) / L)
        wl = np.concatenate((wl, wt))
        zt = np.concatenate((zt, wt / (2.0 * QT)))
        g = np.concatenate((g, math.sqrt(float(torsion)) * np.sin(lt * math.pi * beta)))   # mobility ~ g^2
        gb = np.concatenate((gb, np.zeros(wt.size)))
    if wolf > 0.0:
        fw = float(wolf_hz) if wolf_hz else freq * WOLF_RATIO
        ww = 2.0 * math.pi * fw
        wl = np.concatenate((wl, [ww]))
        zt = np.concatenate((zt, [ww / (2.0 * float(wolf_q))]))
        g = np.concatenate((g, [math.sqrt(float(wolf) * float(np.sum(g * g)))]))
        gb = np.concatenate((gb, [math.sqrt(float(wolf)) * float(gb[0])]))
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
    wr = float(width) if width > 0.0 else BOW_WIDTH
    Kh, Dh = (HAIR_K * wr, HAIR_D * wr) if hair else (0.0, 0.0)
    y, temp = _run(n, A, g, gb, m, fN, vB, dt, P["s0"], P["s1"], P["vs"], P["eps"], P["muC"], P["muS"],
                   chi, vg, mg, th, Kh, Dh)
    return (y, temp) if want_temperature else y
