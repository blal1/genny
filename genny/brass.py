"""Spent brass: shell modes, per-mode damping, air column, bounce kinematics and the ringing case as a parametric engine.

Pure functions, numpy only. A case is a key of ``CASE_GEOM``; a floor is a key of ``FLOORS``. Every constant that is a
choice of ear is a keyword argument of ``render_case``; sources are in research/SOURCES.md (section `brass.py`).
"""
from __future__ import annotations

import math

import numpy as np

from . import presets
from .presets import F

RATE = 48000
G = 9.81
C_AIR = 343.0
BRASS_C = 1090.0        # sqrt(E / (12 rho (1 - nu^2)))
PLASTIC_C = 330.0
BRASS_NU = 0.31
PLASTIC_NU = 0.42
E_BRASS, NU_BRASS = 110.3e9, 0.31
ETA0_BRASS = 3e-4
ETA_PLASTIC = 0.03
ETA_CONTACT = 0.02
ZENER_DELTA = 3.59e-3
D_THERMAL = 3.4e-5
MU_ROLL = 0.015
E_MAX = 0.85
R_EDGE = 1e-3

CASE_GEOM = presets.schema("case", {
    "L": F(float, lo=1.0, hi=500.0, unit="mm"), "db": F(float, lo=1.0, hi=200.0, unit="mm"), "dn": F(float, lo=1.0, hi=200.0, unit="mm"),
    "h": F(float, lo=0.05, hi=5.0, unit="mm"), "cap": F(float, lo=0.1, hi=5000.0, unit="cm3"), "mass": F(float, lo=0.1, hi=5000.0, unit="g"),
    "mat": F(str, choices=("brass", "pe")), "neck": F(float, lo=0.0, hi=300.0, unit="mm"),
})

# Young's modulus Pa, Poisson, reference restitution, soft ground
FLOORS = {"concrete": (30e9, 0.2, 0.45, False), "tile": (70e9, 0.22, 0.50, False), "wood": (9.4e9, 0.3, 0.40, False),
          "metal": (185e9, 0.3, 0.55, False), "dirt": (0.05e9, 0.4, 0.15, True), "grass": (0.02e9, 0.4, 0.10, True)}
# floor thump: (f Hz, d 1/s, gain)
_FLOOR_MODES = {"concrete": [(250, 350, 1.0), (700, 500, 0.6), (1800, 700, 0.4)],
                "tile": [(900, 300, 1.0), (2400, 500, 0.7), (4500, 600, 0.4)],
                "wood": [(140, 120, 1.0), (380, 160, 0.7), (900, 220, 0.5)],
                "metal": [(400, 60, 1.0), (1100, 80, 0.8), (2600, 100, 0.6)]}
_FACE_P = (0.6, 0.2, 0.2)       # flank, neck rim, base
_RAY = (1.0, 0.7, 0.15)
_LOVE = (1.0, 1.0, 0.15)


def _geom(case) -> dict:
    if isinstance(case, dict):
        return case
    if case not in CASE_GEOM:
        raise ValueError(f"case must be {'|'.join(CASE_GEOM)}")
    return CASE_GEOM[case]


def _floor(floor):
    if floor not in FLOORS:
        raise ValueError(f"floor must be {'|'.join(FLOORS)}")
    return FLOORS[floor]


def half_sine_spectrum(f, t_c):
    """Signed spectrum, per unit impulse, of a half-sine force lasting ``t_c`` (phase delay t_c/2 left out)."""
    x = 2.0 * np.asarray(f, float) * t_c
    den = 1.0 - x * x
    near = np.abs(den) < 1e-6
    return np.where(near, math.pi / 4.0, np.cos(math.pi * x / 2.0) / np.where(near, 1.0, den))


def half_sine_force(impulse, t_c, n, rate=RATE):
    """Half-sine force in N, ``n`` samples, whose sum times 1/rate equals ``impulse`` (N.s)."""
    nfft = 1 << int(max(n, 16) - 1).bit_length()
    f = np.fft.rfftfreq(nfft, 1.0 / rate)
    spec = impulse * half_sine_spectrum(f, t_c) * np.exp(-1j * math.pi * f * t_c)
    return (np.fft.irfft(spec, nfft) * rate)[:n]


def case_modes(case, wall_scale=1.0, plastic_c=PLASTIC_C) -> np.ndarray:
    """Frequencies Hz: Rayleigh n = 2, 3, 4, then the same with the free-ends tilting correction, then (50 BMG, 7.62) the beam mode."""
    g = _geom(case)
    c = BRASS_C if g["mat"] == "brass" else plastic_c
    r, h, l = g["db"] * 0.5e-3, g["h"] * 1e-3 * wall_scale, g["L"] * 1e-3
    ns = np.array([2.0, 3.0, 4.0])
    f = h / r ** 2 * c / (2.0 * math.pi) * ns * (ns * ns - 1.0) / np.sqrt(ns * ns + 1.0)
    q = (r / l) ** 2 / (ns * ns)
    love = f * np.sqrt((1.0 + 24.0 * (1.0 - BRASS_NU) * q) / (1.0 + 12.0 * q / (ns * ns + 1.0)))
    out = [f, love]
    if g["mat"] == "brass" and g["L"] > 50:
        rm = 0.25e-3 * (g["db"] + g["dn"])
        out.append(np.array([22.37 / (2.0 * math.pi * l * l) * math.sqrt(E_BRASS / 8530.0) * rm / math.sqrt(2.0)]))
    return np.concatenate(out)


def mode_decay(f, h_mm, material="brass", state="free", ring=1.0) -> np.ndarray:
    """Decay rate d (1/s; T60 = 6.91 / d) of modes at ``f``: internal loss + Zener thermoelastic loss (brass), or a polymer loss (pe);
    ``state`` rest adds the floor contact loss. ``ring`` divides the internal loss."""
    f = np.asarray(f, float)
    if material == "brass":
        h = h_mm * 1e-3
        fte = math.pi * D_THERMAL / (2.0 * h * h)
        x = f / fte
        eta = ETA0_BRASS / ring + ZENER_DELTA * 2.0 * x / (1.0 + x * x)
    else:
        eta = np.full_like(f, ETA_PLASTIC / ring)
    if state == "rest":
        eta = eta + ETA_CONTACT
    return math.pi * f * eta


def air_mode(case, c=C_AIR) -> tuple:
    """(f Hz, d 1/s) of the air column: quarter-wave for a straight case, two-section resonance (body closed, neck open) otherwise."""
    g = _geom(case)
    h = g["h"] * 1e-3
    if g["neck"] <= 0:
        a = g["dn"] * 0.5e-3 - h
        f = c / (4.0 * (g["L"] * 1e-3 - 3e-3 + 0.61 * a))
    else:
        ab, an = g["db"] * 0.5e-3 - h, g["dn"] * 0.5e-3 - h
        ln = g["neck"] * 1e-3
        Ab, An = math.pi * ab * ab, math.pi * an * an
        lb = (g["cap"] * 1e-6 - An * ln) / Ab
        lnp = ln + 0.61 * an
        lo, hi = 1e-6, min(math.pi / (2.0 * lb), math.pi / (2.0 * lnp)) - 1e-6
        for _ in range(80):
            k = 0.5 * (lo + hi)
            if Ab * math.tan(k * lb) < An / math.tan(k * lnp):
                lo = k
            else:
                hi = k
        f = c * k / (2.0 * math.pi)
    return f, math.pi * f / 30.0


def _e_star(floor) -> float:
    ef, nf, _, _ = _floor(floor)
    return 1.0 / ((1.0 - NU_BRASS ** 2) / E_BRASS + (1.0 - nf ** 2) / ef)


def hertz_tc(m, v, floor, edge=True, r_flank=4.8e-3) -> float:
    """Hertz contact time s of a mass ``m`` kg at ``v`` m/s on ``floor``: 2.87 (m^2 / (R E*^2 v))^(1/5); R 1 mm at an edge."""
    r = R_EDGE if edge else r_flank
    return 2.87 * (m * m / (r * _e_star(floor) ** 2 * max(v, 1e-3))) ** 0.2


def restitution(v, floor, lively=0.0) -> float:
    """e = min(0.85, e_ref v^(-1/4)); ``lively`` gives that fraction of (1 - e_ref) back to e_ref."""
    e_ref = _floor(floor)[2]
    e_ref = e_ref + (1.0 - e_ref) * float(np.clip(lively, 0.0, 1.0))
    return float(min(E_MAX, e_ref * max(v, 1e-3) ** -0.25))


def bounce_schedule(case, floor, height, rng, lively=0.0, spin=40.0, ending="auto", wobble=True, wobble_hz=60.0) -> dict:
    """Kinematics of one case. 'hits' (t s, J N.s, tc s, face 0 flank|1 rim|2 base, v, e), 'omega' (rev/s after each bounce, built from
    the off-centre impact), 'rest' (instant it stops flying) and the 'ending': roll (round the cone apex) | wobble (Euler disc) | none."""
    g = _geom(case)
    m, L, soft = g["mass"] * 1e-3, g["L"] * 1e-3, _floor(floor)[3]
    v = math.sqrt(2.0 * G * float(np.clip(height, 0.05, 20.0)))
    t, om = 0.0, float(spin)
    T, J, TC, F, V, E, OM = [], [], [], [], [], [], []
    while True:
        face = int(rng.choice(3, p=_FACE_P))
        s = float(rng.uniform(-0.5, 0.5))
        e = restitution(v, floor, lively)
        meff = m / (1.0 + 12.0 * s * s)
        T.append(t); J.append(meff * v * (1.0 + e)); TC.append(hertz_tc(meff, v, floor, face != 0)); F.append(face); V.append(v); E.append(e)
        om = 0.6 * om + 12.0 * (1.0 + e) * v * s / L / (2.0 * math.pi)
        OM.append(om if spin > 0 else 0.0)
        v *= e * float(rng.uniform(0.85, 1.0))
        dt = 2.0 * v / G
        if v < 0.005 or dt < 10.0 * hertz_tc(m, v, floor):
            break
        t += dt
    hits = dict(t=np.array(T), J=np.array(J), tc=np.array(TC), face=np.array(F), v=np.array(V), e=np.array(E))
    rest = 0.0 if soft else T[-1] + 0.002
    kind = ending
    if kind == "auto":
        kind = "none" if soft else ("wobble" if wobble and rng.random() < 0.2 else "roll")
    out = dict(hits=hits, omega=np.array(OM), rest=rest, ending=kind, roll=None, extra=None)
    r_b, r_n = g["db"] * 0.5, g["dn"] * 0.5
    out["r_roll"] = min(1.0, g["L"] * 1e-3 * r_b / max(r_b - r_n, 1e-3)) if g["neck"] > 0 else 1.0
    t0 = rest
    if kind == "roll":
        vr = float(rng.uniform(0.3, 1.0))
        a = MU_ROLL * G
        n = int(vr * vr / (2.0 * a) / (math.pi * g["db"] * 1e-3) + rng.uniform(0.0, 1.0))
        dist = (np.arange(n) + rng.uniform(0.0, 1.0)) * math.pi * g["db"] * 1e-3
        dist = dist[dist < vr * vr / (2.0 * a)]
        tk = (vr - np.sqrt(np.maximum(vr * vr - 2.0 * a * dist, 0.0))) / a
        vk = np.maximum(vr - a * tk, 0.02)
        out["roll"] = dict(t0=t0, v0=vr, a=a, T=vr / a)
        out["extra"] = dict(t=t0 + tk, J=0.1 * m * vk, tc=np.array([hertz_tc(m, x, floor) for x in vk]), face=np.ones(len(tk), int))
    elif kind == "wobble":
        tw, tau, tt, ts, js, tcs = 0.3, 0.3, 0.0, [], [], []
        while tau > 1e-4:
            ts.append(t0 + tt)
            ju = 0.15 * m * (tau / tw) ** (1.0 / 3.0)
            js.append(ju)
            tcs.append(hertz_tc(m, 0.15 * (tau / tw) ** (1.0 / 3.0), floor))
            nu = min(400.0, wobble_hz * (tau / tw) ** (-1.0 / 3.0))
            tt += 1.0 / nu
            tau -= 1.0 / nu
        out["extra"] = dict(t=np.array(ts), J=np.array(js), tc=np.array(tcs), face=np.full(len(ts), 2))
    return out


def _bank(g, wall_scale, ring, split, air, plastic_c):
    """Mode table: f, d_free, d_rest, kind (0 Rayleigh|1 Love|2 beam|3 air), azimuth order n, orientation 0 cos|1 sin."""
    mat = g["mat"]
    fm = case_modes(g, wall_scale, plastic_c)
    nb = 3 if len(fm) >= 6 else 0
    F, K, N = [], [], []
    for i, f in enumerate(fm):
        kind = 0 if i < 3 else (1 if i < 6 else 2)
        n = (i % 3) + 2 if kind < 2 else 0
        for sgn, ori in ((-1, 0), (1, 1)) if kind < 2 else ((0, 0),):
            F.append(f * (1.0 + sgn * split)); K.append((kind, n, ori))
    F = np.array(F)
    keep = F < 0.45 * RATE
    h = g["h"] * wall_scale
    d_free = mode_decay(F, h, mat, "free", ring)
    d_rest = mode_decay(F, h, mat, "rest", ring)
    fa, da = air_mode(g)
    return F[keep], d_free[keep], d_rest[keep], [k for k, kk in zip(K, keep) if kk], (fa, da)


def render_case(case, floor, height, rng, ring=1.0, spin=40.0, spin_depth=0.5, air=0.25, wobble=True, rattle_law="v",
                rate=RATE, lively=0.0, wall_scale=1.0, split=0.001, roll_mod=1.0, floor_gain=0.15, ending="auto",
                schedule=None, plastic_c=PLASTIC_C) -> np.ndarray:
    """One case falling and bouncing on ``floor`` then rolling or wobbling to rest, peak 1. ``spin`` rev/s at ejection (0 = no tumbling);
    ``ring`` multiplies the internal ringing time; ``air`` the air column gain; ``rattle_law`` v (physical) | sqrt_v."""
    g = _geom(case)
    soft = _floor(floor)[3]
    sc = schedule or bounce_schedule(case, floor, height, rng, lively, spin, ending, wobble)
    h = sc["hits"]
    t, J, tc, face = h["t"], h["J"].copy(), h["tc"], h["face"]
    if rattle_law == "sqrt_v":
        J = J * np.sqrt(h["v"][0] / h["v"])
    if sc["extra"] is not None:
        ex = sc["extra"]
        t, J, tc, face = (np.concatenate([a, b]) for a, b in ((t, ex["t"]), (J, ex["J"]), (tc, ex["tc"]), (face, ex["face"])))
    ts = sc["rest"]
    F, dfr, drs, kinds, (fa, da) = _bank(g, wall_scale, ring, split, air, plastic_c)
    t_end = float(t.max()) + 0.15 + sc.get("tail", 0.0) + (sc["roll"]["T"] if sc["roll"] else 0.0)
    n = int(t_end * rate)
    y = np.zeros(n)
    tt = np.arange(n) / rate
    theta = rng.uniform(0.0, 2.0 * math.pi, len(t))
    rotating = spin > 0 and len(sc["omega"]) > 0
    if rotating:
        w = np.zeros(n)
        tb = h["t"]
        for k in range(len(tb)):
            i0 = int(tb[k] * rate)
            i1 = int(tb[k + 1] * rate) if k + 1 < len(tb) else int(ts * rate)
            w[i0:i1] = sc["omega"][k]
        phi = np.cumsum(w) / rate
        a_t = 1.0 - spin_depth + spin_depth * np.abs(np.sin(2.0 * math.pi * phi + rng.uniform(0.0, 3.14)))
        psi = 0.3 * phi

    def env_len(d_f, d_r, t0):
        tf = max(ts - t0, 0.0)
        return tf + max(9.2 - d_f * tf, 0.0) / d_r if d_f * tf < 9.2 else 9.2 / d_f

    def one(f, d_f, d_r, amp_hit, t_hit, tc_hit):
        out = np.zeros(n)
        for t0, a, c in zip(t_hit, amp_hit, tc_hit):
            i0 = int(math.ceil((t0 + c / 2.0) * rate))
            if i0 >= n or a == 0.0:
                continue
            i1 = min(n, i0 + int(env_len(d_f, d_r, t0) * rate) + 1)
            u = tt[i0:i1] - t0 - c / 2.0
            tf = max(ts - t0 - c / 2.0, 0.0)
            ex = -d_f * np.minimum(u, tf) - d_r * np.maximum(u - tf, 0.0)
            out[i0:i1] += a * np.exp(ex) * np.sin(2.0 * math.pi * f * u)
        return out

    gn = {2: 1.0, 3: 0.5, 4: 0.33}
    for f, d_f, d_r, (kind, nn, ori) in zip(F, dfr, drs, kinds):
        if soft:
            d_f = d_r
        gf = np.array([(_RAY if kind == 0 else _LOVE if kind == 1 else (0.3, 0.3, 0.3))[x] for x in face])
        az = (np.cos(nn * theta) if ori == 0 else np.sin(nn * theta)) if nn else np.ones(len(t))
        amp = J * half_sine_spectrum(f, tc) * gf * az * gn.get(nn, 1.0)
        sig = one(f, d_f, d_r, amp, t, tc)
        if rotating:
            sig = sig * a_t
            if nn and roll_mod > 0:
                sig = sig * ((1.0 - roll_mod) + roll_mod * (np.cos(nn * 2.0 * math.pi * psi) if ori == 0 else np.sin(nn * 2.0 * math.pi * psi)))
        y += sig
    if air > 0:
        y += air * one(fa, da, da, J * half_sine_spectrum(fa, tc), t, tc)
    if floor_gain > 0:
        if soft:
            for t0, a, c in zip(t, J, tc):
                m_ = int(0.012 * rate)
                i0 = int(t0 * rate)
                k = np.exp(-np.arange(int(0.0015 * rate)) / (0.0004 * rate))
                nz = np.convolve(rng.uniform(-1, 1, m_ + len(k)), k)[:m_] * (1.0 - np.arange(m_) / m_) ** 2
                y[i0:i0 + m_] += floor_gain * 0.1 * a * nz[: max(0, min(m_, n - i0))]
        else:
            for ff, dd, gg in _FLOOR_MODES[floor]:
                y += floor_gain * gg * one(ff, dd, dd, J * half_sine_spectrum(ff, tc), t, tc)
    base = face == 2
    if base.any():
        for t0, a, c in zip(t[base], J[base], tc[base]):
            click = np.diff(half_sine_force(a, c, 64, rate), prepend=0.0) / rate * 0.5
            i0 = int(t0 * rate)
            y[i0:i0 + 64] += click[: max(0, min(64, n - i0))]
    if sc["roll"] is not None:
        r = sc["roll"]
        i0, m_ = int(r["t0"] * rate), int(r["T"] * rate)
        i1 = min(n, i0 + m_)
        kern = np.exp(-np.arange(int(0.0004 * rate)) / (0.00005 * rate))
        nz = np.convolve(rng.uniform(-1, 1, i1 - i0 + len(kern)), kern)[: i1 - i0]
        nz = nz / (np.std(nz) + 1e-12)
        vt = np.maximum(r["v0"] - r["a"] * (np.arange(i1 - i0) / rate), 0.0)
        y[i0:i1] += 0.05 * g["mass"] * 1e-3 * vt * nz
    pk = float(np.max(np.abs(y))) or 1.0
    return y / pk
