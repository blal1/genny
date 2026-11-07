"""Combustion engines by architecture and what happens around them: firing layouts, two-stroke, Wankel, diesel, forced
induction, start, stall, shifts, limiter, mechanical noise. Everything is rendered by ``cyber.pipe_engine`` (a
network of digital waveguides) and layers named below, at the physical engine's rate, and resampled at the boundary.

Sources of every constant: research/SOURCES.md, section engines; code taken from other projects: THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import uniform_filter1d

from .core import DEFAULT_SR
from .cyber import _along, _modes, pipe_engine, rpm_curve, turbo
from .matter import _bp, _fin, _fit, _lp, _hp, _n, _rng, _tvbp, _unit, air_leak
from .physical.core import SR
from .sfx import sfx


def _even(c):
    return tuple(i / c for i in range(c))


# Per kind: offsets (fractions of the cycle, firing order), banks by slot, default rpm, header in ms (a number or one per slot).
# Offsets are 720/N unless written out.
LAYOUTS = {
    "engine_single": dict(offsets=(0.0,), rpm=3000.0, header=2.5),                                     # 3000 rpm = 25 Hz
    "vtwin_45": dict(offsets=(0.0, 0.4375), rpm=950.0, header=2.5),                                    # 315/405
    "vtwin_90": dict(offsets=(0.0, 0.375), rpm=1500.0, header=2.5),                                    # 270/450
    "twin_180": dict(offsets=(0.0, 0.25), rpm=3000.0, header=2.0),                                     # 180/540
    "twin_270": dict(offsets=(0.0, 0.375), rpm=3000.0, header=2.0),                                    # 270/450
    "twin_360": dict(offsets=(0.0, 0.5), rpm=3000.0, header=2.0),                                      # 360/360
    # 1-3-2-4 with longer runners on 2 and 4, so short, short, long, long by slot (1.3:1); idle 700 rpm.
    # That pattern repeats over 720 degrees (odd multiples of rpm / 120), not over the spec's calculated 360 degrees / 11.7 Hz.
    "boxer_4": dict(offsets=_even(4), rpm=700.0, header=(1.6, 1.6, 2.08, 2.08)),
    "boxer_6": dict(offsets=_even(6), banks="RLRLRL", rpm=3000.0, header=1.8),                         # R1-L2-R3-L1-R2-L3
    "inline_4": dict(offsets=_even(4), rpm=883.0, header=(0.958, 0.333, 0.708, 0.958)),
    "inline_4_crossplane": dict(offsets=(0.0, 0.125, 0.375, 0.75), rpm=6000.0, header=1.4),            # 90-180-270-180
    "inline_5": dict(offsets=_even(5), rpm=6000.0, header=1.6),                                        # 1-2-4-5-3, 144
    "inline_6": dict(offsets=_even(6), rpm=3000.0, header=1.6),                                        # 1-5-3-6-2-4, 120
    "v6_60": dict(offsets=_even(6), banks="RLRLRL", rpm=3000.0, header=1.6),                           # R1-L1-R2-L2-R3-L3
    "v6_90": dict(offsets=(0.0, 0.125, 1 / 3, 11 / 24, 2 / 3, 19 / 24), banks="RLRLLR", rpm=3000.0, header=1.6),   # R1-L2-R2-L3-L1-R3, 90/150
    "v8_crossplane": dict(offsets=_even(8), banks="LRLLRLRR", rpm=1500.0, header=2.0),                 # per bank 180-90-180-270
    "v8_flatplane": dict(offsets=_even(8), banks="LRLRLRLR", rpm=9000.0, header=1.0),                  # alternating, 9000 rpm = 600 Hz
    "v10": dict(offsets=_even(10), banks="RLRLRLRLRL", rpm=9000.0, header=1.0),                        # R1-L5-R5-L2-R2-L3-R3-L4-R4-L1, 750 Hz
    "v12": dict(offsets=_even(12), banks="LRLRLRLRLRLR", rpm=8000.0, header=1.0),                      # 60, 800 Hz
    # One combustion per shaft turn and per rotor: 2 rotors x 3 chambers of 1080 degrees. 1100-9000 rpm.
    "wankel": dict(offsets=_even(6), cycle="rotary", rpm=3000.0, header=1.2),
    "diesel_i4": dict(offsets=_even(4), rpm=1000.0, header=1.6),                                       # 720-4200 rpm, 1000 rpm
    "diesel_i6_truck": dict(offsets=_even(6), rpm=750.0, header=3.5),                                  # idle 750, nominal 1800-2100
}


def firing_offsets(layout: str) -> list[float]:
    """Firing times of a layout in fractions of its cycle, in firing order, for ``pipe_engine(offsets=)``."""
    if layout not in LAYOUTS:
        raise ValueError(f"unknown engine layout {layout!r}: {', '.join(LAYOUTS)}")
    return list(LAYOUTS[layout]["offsets"])


def firing_banks(layout: str) -> str | None:
    """Exhaust bank by firing slot (``L`` / ``R``), None for one collector."""
    firing_offsets(layout)
    return LAYOUTS[layout].get("banks")


def bank_intervals(layout: str) -> dict[str, list[float]]:
    """Crank degrees between the firings of each bank, from its first firing round the 720 degree cycle."""
    o, out = firing_offsets(layout), {}
    for b in sorted(set(firing_banks(layout) or "L")):
        t = [x * 720.0 for x, s in zip(o, firing_banks(layout) or "L" * len(o)) if s == b]
        out[b] = [round((t[(i + 1) % len(t)] - t[i]) % 720.0 or 720.0, 6) for i in range(len(t))]
    return out


def _params(layout):
    p = LAYOUTS[layout]
    kw = dict(offsets=p["offsets"], banks=p.get("banks"), cycle=p.get("cycle", 4))
    h = p["header"]
    kw.update(headers=h) if isinstance(h, tuple) else kw.update(header=h, unequal=0.0)
    return kw


def _rpm(rpm, default):
    return default if np.ndim(rpm) == 0 and float(rpm) < 0 else rpm


def _run(layout, rpm, dur, load=0.5, seed=0, **kw):
    """A layout through the waveguides, at the engine's rate."""
    return pipe_engine(sr=SR, dur=dur, rpm=rpm, load=load, seed=seed, **{**_params(layout), **kw})


def _events(rpm, offsets, n, per=120.0, shift=0.0):
    """Sample of each event at ``offsets`` of a cycle of ``per`` rpm-per-Hz (120 = 720 degrees), ``shift`` cycles later."""
    ph = np.cumsum(_along(rpm, n)) / (per * SR)
    t = (np.arange(int(ph[-1]) + 2)[:, None] + np.asarray(offsets, float) + shift).ravel()
    t = np.sort(t[(t > ph[0]) & (t < ph[-1])])
    return np.searchsorted(ph, t)


_BLOW = 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(12) + 0.5) / 12.0)   # .25 ms: a blow has a contact time, and a one-sample force would click


def _pulses(idx, n, amp=None):
    x = np.zeros(n)
    np.add.at(x, idx, 1.0 if amp is None else amp)
    return np.convolve(x, _BLOW)[:n]


def _smooth(x, ms=2.0):
    """Takes the corners off a gate, whose steps would click through the ignition."""
    return uniform_filter1d(np.asarray(x, float), max(int(ms * 1e-3 * SR), 2), mode="nearest")


# =============================================================================================
# STRUCTURE: engine_mech
# =============================================================================================
# [f, tau, gain] x sc, sc = 0.086 / B. Head, block, crankcase, diesel panels.
_HEAD = ((2300.0, 0.004, 1.0), (3700.0, 0.003, 0.8), (5400.0, 0.002, 0.55), (7900.0, 0.0015, 0.3))
_BLOCK = ((640.0, 0.012, 1.0), (1150.0, 0.009, 0.8), (1900.0, 0.006, 0.6), (2900.0, 0.004, 0.35))
_CASE = ((210.0, 0.03, 1.0), (390.0, 0.022, 0.8), (760.0, 0.012, 0.4))
_CLATTER = ((1700.0, 0.0025, 1.0), (2600.0, 0.002, 0.7), (4100.0, 0.0012, 0.5), (6200.0, 0.0008, 0.3))
_BLOCK2 = _BLOCK[:3]


def _ring(force, modes, sc=1.0):
    f = np.minimum(np.array([m[0] for m in modes]) * sc, 0.45 * SR)
    return _modes(force, f, 1.0 / np.array([m[1] for m in modes]), [m[2] for m in modes])


def knock_frequency(bore: float) -> float:
    """First transverse gas mode of the chamber, fk = 1.841 c / (pi B), c = 950 m/s in the burned gas: 6.47 kHz at 86 mm."""
    return 1.841 * 950.0 / (math.pi * float(bore))


def engine_mech(bore, rpm, gain=1.0, dur=1.0, offsets=(0.0, 0.25, 0.5, 0.75), valve=0.3, slap=0.3, fire=1.0, rod=0.0, knock=0.0, sr=SR, seed=0):
    """The noise of the engine's structure: its peak is ``gain``. ``bore`` in metres, ``rpm`` a number or a curve.
    Head x 9 (valve seats), block x (2 slap + 1.2 combustion), crankcase x 30 (rod knock), chamber mode x 1.6 (knock).
    Each cylinder has 2 seat events per cycle, as its intake and exhaust valves land (a quarter of a cycle before and half a
    cycle after ignition); slap and rod knock fall at the 2 thrust reversals of each turn;
    ``knock`` is the chance per firing."""
    bore = float(bore)
    if not 0.01 <= bore <= 1.0:
        raise ValueError(f"engine_mech bore is in metres, got {bore}")
    n = _n(dur)
    rng = _rng(seed, 110)
    sc = 0.086 / bore
    r = _along(rpm, n)
    fires = _events(r, offsets, n)
    y = np.zeros(n)
    if valve > 0:
        seats = np.concatenate([_events(r, offsets, n, shift=-0.25), _events(r, offsets, n, shift=0.5)])
        y += _ring(9.0 * valve * _pulses(seats, n, rng.uniform(0.7, 1.0, len(seats))), _HEAD, sc)
    turn = _events(r, (0.0, 0.25, 0.5, 0.75), n)                     # 2 per turn = 4 per 720 degree cycle
    y += _ring(2.0 * slap * _pulses(turn, n, rng.uniform(0.7, 1.0, len(turn))) + 1.2 * fire * _pulses(fires, n), _BLOCK, sc)
    if rod > 0:
        y += _ring(30.0 * rod * _pulses(turn, n, rng.uniform(0.8, 1.0, len(turn))), _CASE, sc)
    if knock > 0:
        hit = fires[rng.random(len(fires)) < knock]
        fk = knock_frequency(bore)
        y += _ring(1.6 * _pulses(hit, n, rng.uniform(0.5, 1.0, len(hit))), ((fk, 0.0045, 1.0), (fk * 1.66, 0.003, 0.5), (fk * 2.1, 0.002, 0.3)))
    return _fin(y, sr, float(gain), 0.002, 0.02)


# =============================================================================================
# ARCHITECTURES
# =============================================================================================

@sfx("engine_layout", "A combustion engine by architecture: the true firing intervals, exhaust banks and runner lengths of a single, twins, boxers, inlines, Vs and a Wankel.",
     kind=("inline_4", "engine_single | vtwin_45 | vtwin_90 | twin_180 | twin_270 | twin_360 | boxer_4 | boxer_6 | inline_4 | inline_4_crossplane | inline_5 | inline_6 | v6_60 | v6_90 | v8_crossplane | v8_flatplane | v10 | v12 | wankel"),
     rpm=(-1.0, "crankshaft, or a curve; -1 = the architecture's usual speed"), load=(0.5, "0 coasting .. 1 full throttle"), dur=(3.0, "s"), seed=(0, "variation"))
def engine_layout(sr=DEFAULT_SR, kind="inline_4", rpm=-1.0, load=0.5, dur=3.0, seed=0):
    if kind not in LAYOUTS or kind.startswith("diesel"):
        raise ValueError(f"engine_layout kind {kind!r}: use one of {', '.join(k for k in LAYOUTS if not k.startswith('diesel'))}")
    y = _run(kind, _rpm(rpm, LAYOUTS[kind]["rpm"]), dur, load, seed)
    if kind == "engine_single":   # a single shakes its frame: the modal bank of the structure under the waveguides
        y = y + engine_mech(0.086, _rpm(rpm, LAYOUTS[kind]["rpm"]), 0.25, dur, (0.0,), seed=seed)
    return _fin(y, sr, 0.7, 0.002, 0.02)


def expansion_chamber(bore=0.05, egt=800.0):
    """Stepped tuned pipe as (delays in samples, junction reflections, section gains): header .2 m at .6 B, diffuser .3 m in 6
    steps up to 1.9 B, belly .1 m, counter-cone .2 m in 5 steps down to 1.15 x stinger, stinger .2 m at .32 B.
    Sound speed sqrt(1.36 x 287 T), T from zones 1, .85, .75, .7, .6 of 300 K..EGT and loss g = exp(-.09 k),
    k = .072 L / sqrt(D / .05) (wall .12 x .6). EGT 800 K, bore 50 mm.
    A wider section reflects a negative wave (the suction that pulls the cylinder clean), a narrower one a positive wave."""
    B = float(bore)
    d_h, d_b, d_s = 0.6 * B, 1.9 * B, 0.32 * B
    sec = [(0.2, d_h, 1.0)]
    sec += [(0.3 / 6, d_h + (d_b - d_h) * (i + 0.5) / 6, 0.85) for i in range(6)]
    sec += [(0.1, d_b, 0.75)]
    sec += [(0.2 / 5, d_b + (1.15 * d_s - d_b) * (i + 0.5) / 5, 0.7) for i in range(5)]
    sec += [(0.2, d_s, 0.6)]
    L, D, Z = (np.array(v, float) for v in zip(*sec))
    c = np.sqrt(1.36 * 287.0 * (300.0 + Z * (float(egt) - 300.0)))
    A = D * D
    return (np.maximum(np.round(L / c * SR).astype(np.int64), 1), (A[:-1] - A[1:]) / (A[:-1] + A[1:]),
            np.exp(-0.09 * 0.072 * L / np.sqrt(D / 0.05)))


@sfx("two_stroke", "Two-stroke engine: one firing per turn through piston-controlled ports and a tuned expansion chamber. Moped, chainsaw, dirt bike, outboard.",
     rpm=(7000.0, "crankshaft, or a curve: 7000 rpm = 116.7 Hz"), cylinders=(1, "1..4, one chamber per bank"), load=(0.6, "0 coasting .. 1 full throttle"),
     bore=(0.05, "m: sets the chamber's diameters"), dur=(3.0, "s"), seed=(0, "variation"))
def two_stroke(sr=DEFAULT_SR, rpm=7000.0, cylinders=1, load=0.6, bore=0.05, dur=3.0, seed=0):
    C = int(np.clip(cylinders, 1, 4))
    y = pipe_engine(sr=SR, dur=dur, rpm=rpm, cylinders=C, load=load, cycle=2, header=0.3, pipe=0.5, unequal=0.0, mic=0.15,
                    banks=[i % 2 for i in range(C)] if C > 1 else None, chamber=expansion_chamber(bore), seed=seed)
    return _fin(y, sr, 0.7, 0.002, 0.02)


# =============================================================================================
# DIESEL
# =============================================================================================

def ignition_delay_ms(cetane=51.0, p_bar=60.0, t_k=900.0):
    """tau = .44 (51 / CN)^.6 p^-1.19 exp(4650 / T) ms, Wolfer's form with a cetane term;
    p in bar and T in K at injection: .59 ms at the defaults."""
    return 0.44 * (51.0 / float(cetane)) ** 0.6 * float(p_bar) ** -1.19 * math.exp(4650.0 / float(t_k))


_DIESEL = {"diesel_i4": dict(bore=0.081, pipe=7.0), "diesel_i6_truck": dict(bore=0.13, pipe=12.0)}    # 130 mm: sc = .66


@sfx("diesel_engine", "Diesel: weak exhaust pulses under the combustion clatter, the premixed spike ringing the head panels and the block once per firing. Car or truck.",
     kind=("diesel_i4", "diesel_i4 (car, 720-4200 rpm) | diesel_i6_truck (big bore, idle 600-750, rated 1800-2100 rpm)"),
     rpm=(-1.0, "crankshaft, or a curve: 1800 rpm on six cylinders = 90 Hz; -1 = 1000 (car), 750 (truck)"), load=(0.3, "0 idle .. 1 full: fuel per stroke"),
     clatter=(1.0, "0..2 level of the combustion knock"), bore=(0.0, "m: 0 = the preset; a bigger bore lowers the clatter"), cylinders=(0, "0 = the preset, else even firing"),
     pipe=(0.0, "ms straight pipe before the muffler (the drone); 0 = the preset: 7 car, 12 truck"), dur=(3.0, "s"), seed=(0, "variation"))
def diesel_engine(sr=DEFAULT_SR, kind="diesel_i4", rpm=-1.0, load=0.3, clatter=1.0, bore=0.0, cylinders=0, pipe=0.0, dur=3.0, seed=0):
    """Clatter x 3 on the panels [1700, 2600, 4100, 6200] sc Hz, tau [2.5, 2, 1.2, .8] ms, gains [1, .7, .5, .3],
    x 2.5 on the block [640, 1150, 1900] sc, sc = .086 / B. Noise 400-2000 Hz peaking near 800 at 1000 rpm, 30 % load, bursts near
    1500 Hz: those modes put it there. A layer above 4 kHz per firing that fades as the rate climbs above 80 Hz of firing rate.
    Ignition at .25 of a petrol engine's, strength .5 + .5 load."""
    if kind not in _DIESEL:
        raise ValueError(f"diesel_engine kind {kind!r}: use one of {', '.join(_DIESEL)}")
    rng = _rng(seed, 111)
    n = _n(dur)
    B = float(bore) if bore > 0 else _DIESEL[kind]["bore"]
    sc = 0.086 / B
    r = _rpm(rpm, LAYOUTS[kind]["rpm"])
    C = int(cylinders)
    extra = dict(offsets=_even(C)) if C > 0 else {}
    off = extra.get("offsets", LAYOUTS[kind]["offsets"])
    ex = _run(kind, r, dur, float(np.clip(load, 0, 1)), seed, fire=0.25, pipe=float(pipe) if pipe > 0 else _DIESEL[kind]["pipe"], mic=0.15, **extra)
    rr = _along(r, n)
    f_fire = rr * len(off) / 120.0
    # Combustion follows top dead centre by the ignition delay.
    idx = _events(rr, off, n) + int(ignition_delay_ms() * 1e-3 * SR)
    idx = idx[idx < n]
    ev = (0.5 + 0.5 * float(np.clip(load, 0, 1))) * _pulses(idx, n, rng.uniform(0.75, 1.0, len(idx)))
    cl = _ring(3.0 * ev, _CLATTER, sc) + _ring(2.5 * ev, _BLOCK2, sc)
    env = _lp(np.abs(ev), 400.0)
    hi = _hp(rng.standard_normal(n), 4000.0, 2) * env / (float(np.max(env)) + 1e-12) * np.clip(80.0 / np.maximum(f_fire, 1.0), 0.0, 1.0)
    y = _unit(ex) + float(np.clip(clatter, 0, 2)) * (0.8 * _unit(cl) + 0.15 * _unit(hi))
    return _fin(y, sr, 0.7, 0.002, 0.02)


# =============================================================================================
# FORCED INDUCTION
# =============================================================================================

def turbo_sim(rpm, throttle, dur, displacement=0.002, boost_target=1.0e5, bov=True):
    """Shaft speed and boost of a turbo at 8 kHz, returned at ``SR``: dict of w (rad/s), boost (Pa), mdot (compressor, kg/s),
    surge (0..1), bov_flow (kg/s). PR = (1 + .72 x .86 U^2 / (cp T))^3.5, U = w D / 2, D = .06 m;
    x = (PRi - pr) / (PRi - 1 + .02); x >= 0: mdot = .36 rho A (U + 8) sqrt(x); x < 0: surge, mdot = -.5 rho A 60 sqrt(-x);
    turbine P = .68 mdot cpE T (1 - pr^.248), cpE 1150; dw/dt = (P / max(w, 300) - Tc - 2.2e-7 w) / I, I = 2.6e-5; wastegate
    over the target; blow-off 3.5e-4 m^2 if throttle < .12 and more than .3 bar.
    Engine volumetric efficiency .1 + .85 throttle, exhaust temperature
    600 + 500 throttle K, back pressure 1e7 mdot^2 Pa, the 4 L charge pipe, the orifice flow as .6 A sqrt(2 rho dp), and the surge
    line: with the throttle shut the wheel stalls above .92 of its ideal pressure ratio and flows backwards down to .78, a
    relaxation cycle around the reversal (x < 0)."""
    n = _n(dur)
    D, Pa, Ta, cp, cpe, kg, ke = 0.06, 101325.0, 298.0, 1005.0, 1150.0, 0.286, 0.248
    A = math.pi / 4.0 * (0.72 * D) ** 2
    rho = Pa / (287.0 * Ta)
    m = max(int(float(dur) * 8000.0), 2)
    dt = float(dur) / m
    rr, th = _along(rpm, m), _along(np.clip(throttle, 0.0, 1.0), m)
    w, p, wg, surge, Tc, stalled = 3000.0, Pa, 0.0, 0.0, 0.0, False
    out = np.zeros((5, m))
    for i in range(m):
        U = w * D * 0.5
        pri = (1.0 + 0.72 * 0.86 * U * U / (cp * Ta)) ** 3.5
        pr = p / Pa
        x = (pri - pr) / (pri - 1.0 + 0.02)
        stalled = not bov and th[i] < 0.12 and pr > (max(0.78 * pri, 1.15) if stalled else max(0.92 * pri, 1.35))
        if x >= 0 and not stalled:
            md = 0.36 * rho * A * (U + 8.0) * math.sqrt(min(x, 1.0))
            surge *= 0.9 ** (dt * 12000.0)
        else:
            md = -0.5 * rho * A * 60.0 * math.sqrt(min(1.0, max(abs(x), 0.3)))
            surge = 1.0
        rise = max(pr, 1.0) ** kg - 1.0
        Tc = (md * cp * Ta * rise / 0.72 if md > 0 else 0.0) / max(w, 50.0)
        me = p / (287.0 * Ta) * displacement * rr[i] / 120.0 * (0.1 + 0.85 * th[i])
        over = (p - Pa - boost_target) / (0.1 * boost_target + 1.0)
        wg += (min(max(over, 0.0), 1.0) - wg) * min(1.0, dt * 25.0)
        mt = me * (1.0 - wg)
        pe = Pa + 1e7 * mt * mt
        P = 0.68 * mt * cpe * (600.0 + 500.0 * th[i]) * (1.0 - (Pa / pe) ** ke)
        dp = p - Pa
        mb = 0.6 * 3.5e-4 * math.sqrt(2.0 * rho * dp) if bov and th[i] < 0.12 and dp > 0.3e5 else 0.0
        p = max(p + dt * (md - me - mb) * 287.0 * Ta / 0.004, 0.3 * Pa)
        w = max(w + dt * (P / max(w, 300.0) - Tc - 2.2e-7 * w) / 2.6e-5, 0.0)
        out[:, i] = w, p - Pa, md, surge, mb
    return dict(zip(("w", "boost", "mdot", "surge", "bov_flow"), (_along(o, n) for o in out)))


def _throttle_lift(n, at=0.6):
    """Full throttle, shut at ``at`` of the sound."""
    return (np.arange(n) < at * n).astype(float)


@sfx("forced_induction", "What a turbo or a blower adds to an engine: the spool, the blow-off valve, compressor surge flutter, the wastegate's pulsed leak, the whine of a Roots blower. One layer, to mix over an engine.",
     kind=("spool", "spool | bov | flutter | wastegate | roots"), rpm=(3500.0, "engine crankshaft, or a curve"), load=(1.0, "0..1 throttle while it is open"),
     cylinders=(4, "wastegate: sets the pulse rate"), bov=("atmo", "bov: atmo (vents out, loud) | recirc (back into the intake, quiet)"), ratio=(2.1, "roots: blower to crank"),
     dur=(3.0, "s"), seed=(0, "variation"))
def forced_induction(sr=DEFAULT_SR, kind="spool", rpm=3500.0, load=1.0, cylinders=4, bov="atmo", ratio=2.1, dur=3.0, seed=0):
    """Spool, blow-off and surge run on ``turbo_sim``. Whine a^2 (.22 sin p + .1 sin 2p + .05 a sin 7p)(.4 + 6 |mdot|),
    a = min(1, w / 9000); hiss band-passed at 3000 + .35 w; blow-off band-passed at 1400 + 5500 min(1, boost / 1.2e5), Q .7, x 3
    recirculated or x 14 vented; surge rnd x surge x .5 x boost / 1e5.
    Roots: phase w_b x 6, saw x .6 + sin 2phi x .4, x b x 1.6 min(1, w_b / 400), b = max(0, p / 101325 - .85), timing gears w_b x 31 at
    .05 min(1, rpm / 2000); ratio 2.1. Wastegate: a 15 mm leak at .6 of the boost, pulsed at the firing rate, 25 % duty.
    Blower pressure 1 + .6 load bar."""
    if kind not in ("spool", "bov", "flutter", "wastegate", "roots"):
        raise ValueError(f"forced_induction kind {kind!r}: spool, bov, flutter, wastegate or roots")
    if bov not in ("atmo", "recirc"):
        raise ValueError(f"forced_induction bov {bov!r}: atmo or recirc")
    rng = _rng(seed, 112)
    n = _n(dur)
    ld = float(np.clip(load, 0, 1))
    if kind == "wastegate":
        leak = air_leak(sr=SR, dur=dur, pressure=0.6, orifice=15.0, seed=seed)
        rr = _along(rpm, n)
        ph = np.cumsum(rr * int(cylinders) / 120.0) / SR % 1.0
        gate = _smooth(ph < 0.25, 3.0)
        return _fin(_fit(leak, n) * gate, sr, 0.5, 0.005, 0.02)
    if kind == "roots":
        rr = _along(rpm, n)
        wb = rr * 2.0 * math.pi / 60.0 * float(ratio)
        ph = np.cumsum(wb * 6.0) / SR
        saw = (ph % (2.0 * math.pi)) / math.pi - 1.0
        b = max(0.0, 1.0 + 0.6 * ld - 0.85)
        y = (0.6 * saw + 0.4 * np.sin(2.0 * ph)) * b * 1.6 * np.minimum(1.0, wb / 400.0)
        tg = np.cumsum(wb * 31.0) / SR
        y = y + 0.05 * np.minimum(1.0, rr / 2000.0) * (np.sin(tg) + 0.6 * np.sin(2.0 * tg) + 0.3 * np.sin(3.0 * tg))
        return _fin(_lp(y, 9000.0, 2), sr, 0.5, 0.01, 0.03)
    thr = ld * (np.ones(n) if kind == "spool" else _throttle_lift(n))
    s = turbo_sim(rpm, thr, dur, bov=kind == "bov")
    w = s["w"]
    a = np.minimum(1.0, w / 9000.0)
    ph = np.cumsum(w) / SR
    y = a * a * (0.22 * np.sin(ph) + 0.1 * np.sin(2.0 * ph) + 0.05 * a * np.sin(7.0 * ph)) * (0.4 + 6.0 * np.abs(s["mdot"]))
    y = y + _tvbp(rng.standard_normal(n), np.minimum(3000.0 + 0.35 * w, 0.45 * SR), 1.5) * np.abs(s["mdot"]) * 1.4 * a
    if kind == "bov":
        fc = 1400.0 + 5500.0 * np.minimum(1.0, np.maximum(s["boost"], 0.0) / 1.2e5)
        k = 3.0 if bov == "recirc" else 14.0
        y = y + _tvbp(rng.standard_normal(n), fc, 0.7) * _lp(s["bov_flow"] / 1.18, 100.0) * k * (1.0 + 0.5 * np.sin(0.012 * np.arange(n) * 2.0))
    if kind == "flutter":
        y = y + rng.uniform(-1.0, 1.0, n) * s["surge"] * 0.5 * np.maximum(s["boost"] / 1e5, 0.0)
    y = _lp(y, 12000.0, 2)
    return _fin(y, sr, 0.5, 0.005, 0.02)


# =============================================================================================
# EVENTS
# =============================================================================================
_GEARS = (3.4, 2.0, 1.4, 1.05, 0.85, 0.7)     # final drive 3.9, wheel .31 m: rpm = 8.556 x 3.9 x g x km/h = 33.4 g km/h
MAX_RPM_S = 3750.0
_FALL_RPM_S = 2500.0


def gear_rpm(gear: int, kmh: float) -> float:
    return 8.556 * 3.9 * _GEARS[int(np.clip(gear, 1, 6)) - 1] * float(kmh)


def _slew(target, up=MAX_RPM_S, down=_FALL_RPM_S):
    """``target`` (one value per sample) followed at a limited rate, computed at 1 kHz."""
    n = len(target)
    step = max(SR // 1000, 1)
    t = target[::step]
    y = np.zeros(len(t))
    v = float(t[0])
    for i, g in enumerate(t):
        v += np.clip(g - v, -down * 1e-3, up * 1e-3)
        y[i] = v
    return np.interp(np.arange(n), np.arange(len(t)) * step, y)


LAUNCH_RPM, LAUNCH_HYST, LAUNCH_CUT = 4000.0, 250.0, 0.3


def limiter_curve(limit, dur, start=None, rise=MAX_RPM_S, fall=_FALL_RPM_S, hyst=120.0, cut=0.0):
    """(rpm, ignition gate) of a revving engine on its limiter: fuel cut above ``limit``, back at ``limit`` - ``hyst``.
    ``cut``: share of cylinders still firing during the cut (0 = full cut; launch control fires some, so it falls slower)."""
    n = _n(dur)
    r, g = np.zeros(n), np.ones(n)
    v, on = float(limit) - 500.0 if start is None else float(start), True
    for i in range(n):
        if v > limit:
            on = False
        elif v < limit - hyst:
            on = True
        v += (rise if on else -fall * (1.0 - cut)) / SR
        r[i], g[i] = v, (1.0 if on else cut)
    return r, g


def launch_curve(dur, launch=LAUNCH_RPM, idle=900.0):
    """Launch control, foot flat and clutch in: the speed climbs from idle and is held at ``launch`` by a partial
    ignition cut (some cylinders keep firing) with a wider band than the rev limiter."""
    return limiter_curve(launch, dur, start=idle, hyst=LAUNCH_HYST, cut=LAUNCH_CUT)


def afterfire(rpm, throttle, n, rng, idle=900.0, egt=850.0):
    """Pops of unburned fuel in a hot collector, as a collector signal. mF falls 14 /s; a
    light-off has a chance per second of 55 hot (.3 + mF / 2e-5), hot = clip((EGT - 620) / 250, 0, 1), if mF > 1.5e-6; it burns mF (.35
    + .6 rnd), E = burn LHV .3, volume source .248 E / 101325 x 90 decaying x .981 per sample at 96 kHz (x .962 here), x (1 + .5 rnd),
    only above idle + 700 rpm and under 3 % throttle. LHV 43.5e6. The rich mixture feeds the pool 2e-4 kg/s; scale to the
    collector 40."""
    r, th = _along(rpm, n), _along(throttle, n)
    out = np.zeros(n)
    u, v = rng.random(n), rng.random(n)
    mF, pop = 0.0, 0.0
    hot = float(np.clip((egt - 620.0) / 250.0, 0.0, 1.0))
    times = []
    for i in range(n):
        if th[i] < 0.03 and r[i] > idle + 700.0:
            mF += 2e-4 / SR
        mF *= 1.0 - 14.0 / SR
        if mF > 1.5e-6 and u[i] < 55.0 * hot * (0.3 + mF / 2e-5) / SR:
            burn = mF * (0.35 + 0.6 * v[i])
            pop += 0.248 * burn * 43.5e6 * 0.3 / 101325.0 * 90.0 * 40.0
            mF -= burn
            times.append(i)
        out[i] = pop
        pop *= 0.962
    # Spread over ~1 ms; per-sample roughness, band-limited at this lower rate.
    out = _smooth(out, 1.0)
    return _lp(out * (1.0 + 0.5 * (2.0 * rng.random(n) - 1.0)), 6000.0, 2), np.array(times, dtype=np.int64)


def _starter(rr, active, rng):
    """Motor at 15 x crank, 138-tooth ring gear, .35 sin 12 phi + .15 sin 24 phi + .25 sin mesh (1 + .5 sin 2 mesh),
    brush noise band-passed at 2800 + .6 wm, Q 1.1, x .35, all under an envelope of .002 per sample, at 96 kHz (5 ms)."""
    n = len(rr)
    w = rr * 2.0 * math.pi / 60.0
    wm = np.maximum(w, 8.0) * 15.0
    phi, mesh = np.cumsum(wm) / SR, np.cumsum(w * 138.0) / SR
    whine = 0.35 * np.sin(12.0 * phi) + 0.15 * np.sin(24.0 * phi) + 0.25 * np.sin(mesh) * (1.0 + 0.5 * np.sin(2.0 * mesh))
    brush = _tvbp(rng.standard_normal(n), np.minimum(2800.0 + 0.6 * wm, 0.45 * SR), 1.1)
    env = _lp(np.asarray(active, float), 1.0 / (2.0 * math.pi * 500.0 / 96000.0))
    return env * (whine * (0.5 + 0.5 * np.minimum(1.0, wm / 1500.0)) + 0.35 * brush)


_EVENTS = ("dfco_overrun", "exhaust_crackle", "engine_start", "engine_start_fail", "engine_stall", "engine_off", "rev_limiter",
           "downshift", "upshift", "free_rev", "engine_knock", "valve_tick", "rod_knock", "engine_bank", "launch_control")
_EVENT_DUR = {"engine_start": 4.0, "engine_start_fail": 4.5, "engine_stall": 2.5, "free_rev": 12.0, "engine_off": 2.5, "downshift": 1.5, "upshift": 1.5}


def _crank(n, rpm=250.0, cylinders=4):
    """Cranking speed with a dip at each compression (4 x rpm / 120 per second on four cylinders), 25 % deep."""
    t = np.arange(n) / SR
    return rpm * (1.0 + 0.25 * np.sin(2.0 * np.pi * cylinders * rpm / 120.0 * t))


def _cranking(rr, cylinders):
    """Cadence of the compressions on the starter's load: the motor drags at each one."""
    return 1.0 + 0.4 * np.sin(np.cumsum(2.0 * np.pi * rr * cylinders / 120.0) / SR)


@sfx("engine_event", "A moment in an engine's life: starting, failing to start, stalling, switching off, overrun, crackle, limiter, shifts, a free rev, knock, valve tick, rod knock, or a loop aligned on the engine cycle.",
     kind=("engine_start", " | ".join(_EVENTS)), layout=("inline_4", "any engine_layout kind"), rpm=(-1.0, "the speed it happens at; -1 = the kind's own"),
     load=(0.5, "0..1"), gear=(3, "shifts: from this gear, 1..6"), speed=(60.0, "shifts: km/h"), limit=(8000.0, "rev_limiter: cut here, back at limit - 120"),
     bore=(0.086, "knock, tick, rod: m (86 mm = knock at 6.47 kHz)"), severity=(0.5, "knock, rod: 0..1"), cycles=(0, "engine_bank: loop length in engine cycles (0 = 3.2 s of them)"),
     dur=(0.0, "s; 0 = the kind's own length"), seed=(0, "variation"))
def engine_event(sr=DEFAULT_SR, kind="engine_start", layout="inline_4", rpm=-1.0, load=0.5, gear=3, speed=60.0, limit=8000.0, bore=0.086, severity=0.5, cycles=0, dur=0.0, seed=0):
    if kind not in _EVENTS:
        raise ValueError(f"engine_event kind {kind!r}: use one of {', '.join(_EVENTS)}")
    firing_offsets(layout)
    dur = float(dur) if dur > 0 else _EVENT_DUR.get(kind, 3.0)
    rng = _rng(seed, 113)
    n = _n(dur)
    C = len(LAYOUTS[layout]["offsets"])
    idle = LAYOUTS[layout]["rpm"]
    t = np.arange(n) / SR
    ld = float(np.clip(load, 0, 1))
    r0 = float(rpm) if np.ndim(rpm) == 0 and rpm > 0 else None
    if kind == "dfco_overrun":
        # g_DFCO = max(-torque, eps), eps = .02: air through the shut throttle while the fuel is off; brake torque rpm / 7500.
        rr = np.linspace(r0 or 4500.0, 1400.0, n)
        y = _unit(_run(layout, rr, dur, 0.0, seed, fire=0.0))
        g = np.maximum(rr / 7500.0, 0.02)
        y = y + 0.5 * g * _unit(_bp(rng.standard_normal(n), 900.0, 0.7))
    elif kind == "exhaust_crackle":
        rr = np.linspace(r0 or 5000.0, 2400.0, n)
        pops, _ = afterfire(rr, np.zeros(n), n, rng, idle=max(idle, 800.0))
        y = _run(layout, rr, dur, 0.0, seed, fire=0.0, pops=pops)
    elif kind in ("engine_start", "engine_start_fail"):
        y = _start(kind == "engine_start", layout, n, dur, C, idle, ld, r0, rng, seed)
    elif kind == "engine_stall":
        # 1100, 700, 420 rpm, rough .9: misfires thicken as it dies, then one shake of the mounts.
        rr = np.interp(t, [0.0, 0.5 * dur, 0.85 * dur, dur], [r0 or 1100.0, 700.0, 420.0, 150.0])
        y = _unit(_run(layout, rr, dur, ld * 0.4, seed, fire=_misfires(rr, n, C, rng, np.linspace(0.3, 0.9, n))))
        y = y * np.clip((dur - t) / (0.2 * dur), 0.0, 1.0)
        from .contact import rumble
        k = int(0.85 * n)
        jolt = rumble(sr=SR, dur=dur - k / SR, fmin=30.0, fmax=70.0, seed=seed)
        y[k:] += 1.5 * _fit(_unit(jolt), n - k) * np.exp(-np.arange(n - k) / (0.09 * SR))
    elif kind == "engine_off":
        # Free run-down, 2 s linear.
        go = np.clip(1.0 - t / min(2.0, dur), 0.0, 1.0)
        rr = np.maximum((r0 or idle) * go, 40.0)
        y = _unit(_run(layout, rr, dur, 0.0, seed, fire=0.0)) * go
    elif kind == "rev_limiter":
        rr, g = limiter_curve(r0 or limit, dur)
        y = _run(layout, rr, dur, ld, seed, fire=_smooth(g), mic=0.2)
    elif kind == "launch_control":
        rr, g = launch_curve(dur, r0 or LAUNCH_RPM, idle)
        y = _run(layout, rr, dur, 1.0, seed, fire=_smooth(g), mic=0.2)
    elif kind in ("downshift", "upshift"):
        y = _shift(kind, layout, n, dur, C, gear, speed, ld, rng, seed)
    elif kind == "free_rev":
        # Throttle .6 at 2 s, 0 at 2.35, 1 at 3.3, 0 at 3.75, 1 at 5, 0 at 8.2. Speed from the torque curve (no limiter here: max_rpm).
        k = np.searchsorted([2.0, 2.35, 3.3, 3.75, 5.0, 8.2], t, side="right")
        thr = np.array([0.0, 0.6, 0.0, 1.0, 0.0, 1.0, 0.0])[k]
        rr = rpm_curve(thr, dur, rpm0=r0 or idle, min_rpm=idle)
        y = _run(layout, _fit(rr, n), dur, np.maximum(_smooth(thr, 20.0), 0.05), seed, fire=_smooth(thr > 0.0, 5.0) * 0.85 + 0.15)
    elif kind in ("engine_knock", "valve_tick", "rod_knock"):
        sev = float(np.clip(severity, 0, 1))
        rr = r0 or 2500.0
        o = LAYOUTS[layout]["offsets"]
        m = dict(engine_knock=dict(knock=sev, valve=0.0, slap=0.0, fire=0.1), valve_tick=dict(valve=1.0, slap=0.0, fire=0.0), rod_knock=dict(rod=0.2 + 0.8 * sev, valve=0.0, slap=0.2, fire=0.1))[kind]
        y = 0.25 * _unit(_run(layout, rr, dur, ld, seed)) + _unit(engine_mech(bore, rr, 1.0, dur, o, seed=seed, **m))
    else:
        return _bank_loop(layout, r0 or idle, ld, cycles, seed, sr)
    return _fin(y, sr, 0.7, 0.003, 0.02)


def _misfires(rr, n, C, rng, chance):
    """Ignition gate that drops a whole firing at a time with probability ``chance``."""
    ph = np.cumsum(rr * C / 120.0) / SR
    k = np.minimum(ph.astype(np.int64), int(ph[-1]))
    drop = rng.random(int(ph[-1]) + 1)
    return _smooth(drop[k] >= chance[np.minimum(np.searchsorted(ph, k), n - 1)], 1.0)


def _start(catches, layout, n, dur, C, idle, ld, r0, rng, seed):
    """Cranking 220-300 rpm, catch at 2300-2600 rpm settling to idle; the delay .25-.75 s and a first second
    of overshoot. Failing: 3 attempts, the battery sagging to .6 of the speed, .35 s apart."""
    t = np.arange(n) / SR
    crank_rpm = rng.uniform(220.0, 300.0)
    if catches:
        tc = 0.5 + rng.uniform(0.25, 0.75)
        peak = rng.uniform(2300.0, 2600.0)
        crank = _crank(n, crank_rpm, C)
        run = np.interp(t, [tc, tc + 0.25, tc + 0.8, tc + 1.6], [crank_rpm, peak, 1.25 * (r0 or idle), r0 or idle])
        on = t >= tc
        rr = _smooth(np.where(on, run, crank), 10.0)
        y = _unit(_run(layout, rr, n / SR, np.where(on, max(ld, 0.3), 0.0), seed, fire=_smooth(on, 3.0)))
        star = _starter(rr, rr < 450.0, rng) * _cranking(rr, C)
        return y * (0.15 + 0.85 * _smooth(on, 40.0)) + 0.2 * _unit(star)
    y = np.zeros(n)
    gaps = 0.35
    each = (dur - 2 * gaps) / 3.0
    for i, sag in enumerate((1.0, 0.8, 0.6)):
        a = int(i * (each + gaps) * SR)
        m = min(int(each * SR), n - a)
        if m < 256:
            break
        rr = _crank(m, crank_rpm * sag, C)
        eng = _unit(pipe_engine(sr=SR, dur=m / SR, rpm=rr, load=0.0, fire=0.0, seed=seed + i, **_params(layout)))
        star = _unit(_starter(rr, np.ones(m), rng) * _cranking(rr, C))
        k = np.minimum(np.minimum(np.arange(m), m - 1 - np.arange(m)) / (0.02 * SR), 1.0)
        y[a:a + m] += (0.6 * _fit(eng, m) + 0.5 * star) * k * sag
    return y


def _shift(kind, layout, n, dur, C, gear, speed, ld, rng, seed):
    """rpm = 33.4 g km/h, slew-limited; a blip into the lower gear on the way down, a cut of .15 s on the way up
    (clutch .25 s). Clutch rattle band-passed at 1900 Hz, Q .6; the box's backlash (`gearbox`); a pop on overrun."""
    from .cyber import gearbox
    t = np.arange(n) / SR
    g = int(np.clip(gear, 1, 6))
    down = kind == "downshift"
    to = max(g - 1, 1) if down else min(g + 1, 6)
    r_a, r_b = gear_rpm(g, speed), gear_rpm(to, speed)
    t0 = 0.3 * dur
    width = 0.25 if down else 0.15
    target = np.where(t < t0, r_a, r_b) if down else np.where(t < t0, r_a, np.where(t < t0 + width, r_a * 0.97, r_b))
    target = np.clip(target, 700.0, 9000.0)
    if down:
        target = np.where((t >= t0) & (t < t0 + width), max(r_a, r_b) * 1.08, target)     # the blip overshoots the new gear
    rr = _slew(target)
    clutch = (t >= t0) & (t < t0 + width)
    thr = np.where(clutch, 0.0 if not down else 0.7, ld)
    y = _unit(_run(layout, rr, dur, _smooth(thr, 10.0), seed, fire=_smooth(~clutch if not down else np.ones(n), 3.0), backfire=0.35 if down else 0.0))
    rattle = _bp(rng.standard_normal(n), 1900.0, 0.6) * _smooth(clutch, 8.0)
    y = y + 0.25 * _unit(rattle)
    box = gearbox(sr=SR, dur=dur, rpm=float(max(r_a, r_b)), load=0.05, backlash=1.0, seed=seed)
    y = y + 0.15 * _unit(_fit(box, n)) * _smooth(clutch, 8.0)
    return y


def _bank_loop(layout, rpm, ld, cycles, seed, sr):
    """Loop aligned on the cycle: wavelength 120 / rpm, length = wavelength ceil(3.2 / wavelength) + crossfade / 2, crossfade 2 wavelengths,
    2 s warm-up thrown away, halves swapped and joined by a linear fade.
    The loop's length is an integer number of cycles, so the seam falls in phase."""
    per = _cycle_per(layout)
    wave = per / float(rpm)
    N = int(cycles) if int(cycles) > 0 else int(math.ceil(3.2 / wave))
    L = int(round(N * wave * SR))
    xf = min(int(round(2.0 * wave * SR)), L // 2)
    warm = 2 * SR
    y = _run(layout, float(rpm), (warm + L + xf + 2000) / SR, ld, seed)[warm:warm + L + xf]
    w = np.linspace(0.0, 1.0, xf)
    out = y[:L].copy()
    out[:xf] = y[L:L + xf] * (1.0 - w) + y[:xf] * w       # the tail fades out over the head: the end runs on into the start
    h = L // 2
    out = np.concatenate([out[h:], out[:h]])
    if int(sr) != SR:
        from scipy.signal import resample
        out = resample(out, int(round(L * int(sr) / SR)))   # periodic resampling keeps the seam
    return out * (0.7 / (float(np.max(np.abs(out))) + 1e-12))


def _cycle_per(layout):
    return {"rotary": 180.0, 2: 60.0, 4: 120.0}[LAYOUTS[layout].get("cycle", 4)]


# =============================================================================================
# TWO FAMILIAR VEHICLES
# =============================================================================================

@sfx("race_car", "A racing car at full song: a high-revving V10 pulling through a gear, with a turbo over it.",
     rpm=(18000.0, "the top of the pull (18000 = F1 V10)"), gear=(4, "1..6: a low gear climbs faster"), load=(1.0, "0..1"), turbo_mix=(0.12, "0..1 turbo whine over the engine"),
     dur=(3.0, "s"), seed=(0, "variation"))
def race_car(sr=DEFAULT_SR, rpm=18000.0, gear=4, load=1.0, turbo_mix=0.12, dur=3.0, seed=0):
    """Rate of climb ``MAX_RPM_S`` scaled by the gear's ratio to first, from 75 % of the top."""
    n = _n(dur)
    top = float(np.clip(rpm, 3000.0, 20000.0))
    rate = MAX_RPM_S * _GEARS[int(np.clip(gear, 1, 6)) - 1] / _GEARS[0]
    rr = np.minimum(0.75 * top + rate * np.arange(n) / SR, top)
    y = _unit(_run("v10", rr, dur, load, seed, mic=0.1))
    y = y + float(np.clip(turbo_mix, 0, 1)) * _unit(_fit(turbo(sr=SR, dur=dur, boost=0.8, blowoff=0, seed=seed), n))
    return _fin(y, sr, 0.7, 0.003, 0.02)


@sfx("jalopy", "A worn-out old car: uneven firing, misfires, backfires on the overrun and a rattling block.",
     rpm=(900.0, "around this speed"), rough=(0.7, "0..1 misfires and wander"), backfire=(0.5, "0..0.99 chance of a pop per cycle as it falls"), load=(0.4, "0..1"),
     dur=(4.0, "s"), seed=(0, "variation"))
def jalopy(sr=DEFAULT_SR, rpm=900.0, rough=0.7, backfire=0.5, load=0.4, dur=4.0, seed=0):
    """`pipe_engine` with backfire, a misfire gate and the rod and slap of ``engine_mech``."""
    rng = _rng(seed, 114)
    n = _n(dur)
    rg = float(np.clip(rough, 0, 1))
    t = np.arange(n) / SR
    rr = float(rpm) * (1.0 + 0.25 * rg * np.sin(2.0 * np.pi * 0.7 * t + rng.uniform(0, 6.28))) * (1.0 - 0.1 * t / dur)
    y = _unit(pipe_engine(sr=SR, dur=dur, rpm=rr, cylinders=4, load=load, lope=0.2 * rg, unequal=0.4, backfire=backfire,
                          fire=_misfires(rr, n, 4, rng, np.full(n, 0.25 * rg)), seed=seed))
    y = y + 0.5 * rg * _unit(engine_mech(0.086, rr, 1.0, dur, slap=1.0, rod=0.3, valve=0.3, seed=seed))
    return _fin(y, sr, 0.7, 0.003, 0.02)


COLD_IDLE = (345.0, 60.0, 0.4, 0.15, 350.0)


def cold_idle(idle, temp_k):
    """Idle speed of a cold engine: idle (1 + 0.4 c^2 + 0.15 c), c = clip((345 - T) / 60, 0, 1), coolant T in K.
    Returns (rpm, fuel-cut recovery rpm = that idle + 350)."""
    t0, span, a, b, rec = COLD_IDLE
    c = float(np.clip((t0 - float(temp_k)) / span, 0.0, 1.0))
    r = float(idle) * (1.0 + a * c * c + b * c)
    return r, r + rec
