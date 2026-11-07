"""Heavy weapons: grenades, mines, launchers, missiles, mortars, guns, bombs, flamethrowers, torpedoes, depth charges.

Spec with every source: reports/sons/armes_lourdes_autres.md. What is sourced: blast levels and durations (Kinney & Graham,
physics.blast_wave), fills and speeds (Wikipedia pages listed there), Farnell Practicals 31 and 32 for shape.
Estimated: every delay inside a sequence, TNT equivalents of muzzles, the rumble's 0.38 s/kg^(1/3). Underwater: Cole's
tables for the peak, the bubble period and the energy each bubble keeps.
"""
from __future__ import annotations

import math

import numpy as np

from . import contact as K
from . import matter as Ma
from .arms import _crack, shell_modes, shock_arrival
from .core import DEFAULT_SR
from .foley import C_AIR, CATCH, ENDSTOP, P, _fin, _mclick, _n, _nm, _put
from .fx import effect
from .klang import _comb3
from .physical import ambience as amb
from .physical import core as pc
from .physical import modal, space
from .physics import blast_wave
from .plates import metal_plate
from .reverbs import doppler
from .sfx import sfx

C_WATER = 1500.0
# Relative effectiveness; from web tables through the deep reading (FM 5-250); Kinney's Table I gives three methods that disagree
RE = {"comp_b": 1.33, "c4": 1.34, "tritonal": 1.05, "torpex": 1.30, "tnt": 1.0, "petn": 1.66, "rdx": 1.60, "semtex": 1.35, "hmx": 1.70, "nitroglycerin": 1.54,
      "anfo": 0.74, "black_powder": 0.55, "tetryl": 1.25, "picric_acid": 1.17, "tatp": 0.80}
# Ground: P up to 5 c on granite. Rock as a Poisson solid (nu 0.25): c_S = c_P / sqrt(3), Rayleigh 0.9194 c_S. Energy of a
# point source at the surface: 7 % P, 26 % S, 67 % Rayleigh; P and S fall as 1/R, Rayleigh as 1/sqrt(R).
V_P = 5.0 * C_AIR
V_S = V_P / math.sqrt(3.0)
V_R = 0.9194 * V_S
GROUND_SHARE = (0.07, 0.26, 0.67)


# TNT products (JWL): A, B (Pa), R1, R2, omega; specific energy J/kg, density kg/m3, detonation speed m/s, CJ pressure Pa
TNT_JWL = dict(A=371.2e9, B=3.231e9, R1=4.15, R2=0.95, omega=0.30, e=4.294e6, rho=1630.0, D=6930.0, p_cj=21e9)


def jwl_pressure(v: float, e: float | None = None) -> float:
    """Pressure (Pa) of TNT detonation products at relative volume v = V / V0 and energy e (J/kg, default the
    charge's): A (1 - w / (R1 v)) exp(-R1 v) + B (1 - w / (R2 v)) exp(-R2 v) + w rho0 e / v."""
    j = TNT_JWL
    e = j["e"] if e is None else float(e)
    v = float(v)
    return (j["A"] * (1 - j["omega"] / (j["R1"] * v)) * math.exp(-j["R1"] * v)
            + j["B"] * (1 - j["omega"] / (j["R2"] * v)) * math.exp(-j["R2"] * v) + j["omega"] * j["rho"] * e / v)


def charge_radius(w_kg: float, rho: float = TNT_JWL["rho"]) -> float:
    """Radius (m) of a bare spherical charge of ``w_kg``: (3 W / (4 pi rho))^(1/3)."""
    return (3.0 * float(w_kg) / (4.0 * math.pi * rho)) ** (1.0 / 3.0)


def rumble_time(w_kg: float) -> float:
    """Tail of the fireball in s: Hopkinson scaling, W^(1/3); 8 s at most (Farnell's fireball)."""
    return min(0.38 * w_kg ** (1.0 / 3.0), 8.0)


def cos_noise_sweep(f: float, rng, dur: float = 0.03) -> np.ndarray:
    """Farnell's blast noise (Fig 54.5, c10 l.26214): cutoff falling from 7000 f^2 Hz, Q 1.5 (bright) .. 5, x 50 and clipped, added
    to the falling line inside a cosine, band 6 kHz Q 1, x line^2. Peak 1. Bounded in amplitude however brutal the sweep."""
    n = _n(dur)
    e = 1.0 - np.arange(n) / n
    x = np.clip(50.0 * Ma._tvbp(rng.standard_normal(n), 20.0 + 7000.0 * (f * e) ** 2, 1.5 + 3.5 * (1.0 - f)), -1.0, 1.0)
    # 0.25 cycle not in the book: at a whole or half cycle the clipped +-1 gives the same cosine and only the line is left
    start = np.minimum(np.arange(n) / _n(0.001), 1.0)
    return _nm(amb.bp(np.cos(2.0 * np.pi * (e + 0.25 * x)) * start, 6000.0, 1.0) * e * e)


def _swept_comb(x: np.ndarray, gain: float = 0.33) -> np.ndarray:
    """Three taps 10, 10 + w, 10 + 2 w ms swept down: the rising fireball's direct and reflected paths (Farnell Fig 54.8; klang.py:971-974)."""
    n = len(x)
    v = 1.0 - np.arange(n) / n
    s = 4.0 * (1.0 - v) + 1e-4
    ms = P * 1e-3
    return _comb3(np.ascontiguousarray(x), (10 + s) * ms, (10 + s + 13.7 * s) * ms, (10 + s + 27.4 * s) * ms, v * gain)


def _fireball(T: float, rng) -> np.ndarray:
    """Farnell's rumble (Fig 54.7; klang.py:952-956): low-passed noise whose cutoff and level ride one line, 40 -> 140 Hz in 100 ms and
    back, quartic amplitude."""
    k = _n(0.1 + T)
    t = np.arange(k) / P
    L = np.where(t < 0.1, t / 0.1, np.clip(1.0 - (t - 0.1) / T, 0.0, 1.0))
    fc = 40.0 + 100.0 * L
    a = np.exp(-2.0 * np.pi * fc / P)
    rum = pc.tv_one_pole_lp(pc.tv_one_pole_lp(Ma._tvbp(rng.uniform(-1, 1, k), fc, 1.0), a), a)
    rum = amb.hip(rum * 10.0, 5.0) * amb.lop(L ** 4, 5.0)
    return _nm(_nm(rum) + 0.7 * _nm(_swept_comb(rum)))


def blast(w_kg: float, r: float, rng, height: float = 1.5, fireball: float = 1.0, shock: float = 1.0, neg: float = 0.1) -> np.ndarray:
    """K1, in pascals, time 0 = the wave arrives. Friedlander shock (x ``shock``), 3 sinusoidal-noise blasts (first after 30 ms, then at
    least 5 ms apart, brightness falling: Farnell Fig 54.6), the fireball, a pressure-wave echo, the "bam-bam" of the ground."""
    ps, td, alpha = blast_wave(w_kg, r)
    T = rumble_time(w_kg)
    y = np.zeros(_n(0.9 + 2.0 * T + 8.0 * td))
    m = max(8, _n(4.0 * td))
    t = np.arange(m) / P / td
    wave = np.where(t < 1.0, (1.0 - t) * np.exp(-alpha * t), 0.0)
    wave += np.where(t >= 1.0, -neg * np.sin(np.pi * (t - 1.0) / 3.0), 0.0)
    _put(y, wave, 0.0, ps * shock)
    if fireball > 0:
        at = 0.03
        for f in sorted(rng.uniform(0.4, 1.0, 3), reverse=True):
            _put(y, cos_noise_sweep(float(f), rng), at, 0.25 * ps * fireball)
            at += max(0.005, float(rng.uniform(0.0, 0.03)))
        _put(y, _fireball(T, rng), 0.0, 0.5 * ps * fireball)
        # Farnell's 500 ms delay whose length shrinks behind the shock (Fig 54.8); down to 250 ms and -10 dB
        tt = np.arange(len(y)) / P
        _put(y, np.interp(tt - np.linspace(0.5, 0.25, len(y)), tt, y, left=0.0), 0.0, 0.3)
    dt = (math.hypot(r, 2.0 * height) - r) / C_AIR
    _put(y, amb.lop(y, 2000.0), dt, 0.6)
    return space.distance(y, r, ref=r)


def ground_wave(w_kg: float, r: float, rng) -> list:
    """Pre-rumble through the ground: (seconds before the blast, Pa) for P, S, Rayleigh. Level 2 % of the blast at 10 m and the 80 Hz thud."""
    p10 = 0.02 * blast_wave(w_kg, 10.0)[0]
    out = []
    for v, share, fall in zip((V_P, V_S, V_R), GROUND_SHARE, (10.0 / r, 10.0 / r, math.sqrt(10.0 / r))):
        k = _n(0.25)
        thud = amb.lop(rng.uniform(-1, 1, k), 80.0) * np.sin(np.pi * np.arange(k) / k) ** 2
        out.append((r / C_AIR - r / v, _nm(thud) * p10 * math.sqrt(share) * min(fall, 1.0)))
    return out


def fragments(count: int, r: float, rng, d_mm: float = 5.0, v_range=(300.0, 1218.0)) -> list:
    """K2: supersonic fragments reach you before the blast by r (1/c - 1/v): (seconds before the blast, crack in Pa)."""
    out = []
    for _ in range(count):
        v = float(rng.uniform(*v_range))
        if v > 1.05 * C_AIR:
            out.append((r * (1.0 / C_AIR - 1.0 / v), _crack(v, d_mm, 2.0 * d_mm, float(rng.uniform(1.0, 8.0)), rng, wake=False)))
    return out


def ball_sheet(r: float, az: float, rng, count: int = 700, arc: float = 30.0) -> list:
    """Claymore: 700 balls of 3.2 mm at 1200-1218 m/s (M18), spread over +-``arc`` degrees. Each passes the listener at its own
    miss distance; the Mach cone (Mach 3.55, 16.4 deg) decides whether its crack gets there: (seconds before the blast, crack in Pa)."""
    out = []
    for _ in range(count):
        v, off = float(rng.uniform(1200.0, 1218.0)), math.radians(az - float(rng.uniform(-arc, arc)))
        x, y = r * math.cos(off), abs(r * math.sin(off))
        t = shock_arrival(v, x, max(y, 0.05))
        if t is not None:
            out.append((r / C_AIR - t, _crack(v, 3.2, 3.2, max(y, 0.5), rng, wake=False)))
    return out


def launch_tube(length: float, closed: bool, burst: np.ndarray, modes: int = 4, f=None) -> np.ndarray:
    """K3: parallel band-passes on the tube's modes: factors 1, 3, 5, 7 of c/4L closed, 2, 3, 4, 5 of c/2L open (Farnell). ``f``: base
    frequency instead, per sample for a moving chamber. c/2L x 2 = 368 Hz for 0.93 m, Farnell's 366."""
    base = (C_AIR / (4.0 * length) if closed else C_AIR / (2.0 * length)) if f is None else f
    ratios = [2 * i + 1 for i in range(modes)] if closed else [i + 2 for i in range(modes)]
    if np.ndim(base):
        return sum(Ma._tvbp(burst, np.minimum(base * h, 0.45 * P), 8.0) / (i + 1) for i, h in enumerate(ratios))
    return sum(amb.bp(burst, min(base * h, 0.45 * P), 8.0) / (i + 1) for i, h in enumerate(ratios))


def rocket_motor(dur: float, rng, u: float = 1000.0, d: float = 0.066, st: float = 0.05, rise: float = 0.09) -> np.ndarray:
    """K4: jet noise peaking at St U/D (St 0.2 lab jet, 0.01-0.02 launchers; 0.05 for a small rocket estimated), through Farnell's rocket
    (Fig 55.8-9): a comb sweeping over the noise, a notch rising as a power curve, the pipe growing as fuel burns (0.2 -> 0.5 m, 51 cm rocket,
    0.2 estimated); louder as it burns, square-root swell (rise 90 ms, c10), exhaust jitter (low-pass 9 Hz)."""
    n = _n(dur)
    t = np.arange(n) / P
    fc = min(st * u / d, 0.4 * P)
    x = amb.bp(rng.uniform(-1, 1, n), fc, 0.7) + 0.25 * amb.hip(rng.uniform(-1, 1, n), fc)
    x = x + _swept_comb(x, 0.5)
    x = x - Ma._tvbp(x, 300.0 + 5000.0 * (t / max(dur, 1e-3)) ** 3, 2.0)
    x = x + 0.5 * launch_tube(0.5, True, x, f=C_AIR / (4.0 * np.linspace(0.2, 0.5, n)))
    jit = 1.0 + 0.3 * amb.lop(rng.uniform(-1, 1, n), 9.0) * 20.0
    env = np.sqrt(np.clip(t / rise, 0.0, 1.0)) * (0.7 + 0.3 * t / max(dur, 1e-3))
    k = min(_n(0.01), n)
    env[-k:] *= np.linspace(1.0, 0.0, k)                                           # burn-out
    return _nm(x * env * jit)


def fly_away(x: np.ndarray, speed: float, closest: float = 3.0, c: float = C_AIR) -> np.ndarray:
    return np.asarray(doppler(x, P, speed=speed, closest=closest, pass_at=0.0, pan=False, c=c))


def moving(x: np.ndarray, v, closest: float, x0: float = 0.0, c: float = C_AIR) -> np.ndarray:
    """Doppler of a source with a speed profile, supersonic allowed while it recedes (reverbs.doppler stops at 0.9 c): ``v`` m/s per sample,
    along a line ``closest`` m off the listener from x0. A sound emitted at t is heard at t + r(t)/c; 1/r, and it dulls with distance
    below 1500 Hz. Output time 0 = emission of the first sample."""
    n = len(x)
    s = x0 + np.cumsum(np.broadcast_to(np.asarray(v, float), (n,))) / P
    r = np.hypot(closest, s)
    te = np.arange(n) / P
    ta = te + r / c
    t = np.arange(int(math.ceil(ta[-1] * P)) + 1) / P
    y = np.interp(t, ta, x, left=0.0, right=0.0)
    ro = np.interp(t, ta, r)
    w = np.minimum(10.0 / ro, 1.0)
    return (w * y + (1.0 - w) * amb.lop(y, 1500.0)) * np.minimum(closest / ro, 2.0)


# TNT in water, SI units (kg, m, s, Pa): peak Pm = UW_PM (W^(1/3)/R)^1.13; bubble period T = UW_T W^(1/3) / (D + UW_D)^(5/6);
# energy left in the bubble after each contraction
UW_PM, UW_T, UW_D = 52.39e6, 2.108, 10.06
UW_KEPT = (0.34, 0.54)


def underwater_blast(w_kg: float, r: float, depth: float, rng, pulses: int = 3, surface: bool = False) -> np.ndarray:
    """Peak 1: exponential shock Pm e^(-t/theta), theta = 96.5 W^(1/3) (W^(1/3)/R)^-0.22 us, then bubble pulses. The first comes T after
    the shock with 0.84 x its impulse; each contraction keeps ``UW_KEPT`` of the energy, so the next period shrinks as its cube root and
    the pulse amplitude as its square root. Pulse width 0.2 T. ``surface``: the free surface's inverted image cuts the shock, 2 D / c later.
    Above the depth's hull, the ear hears it low-passed."""
    a = w_kg ** (1.0 / 3.0)
    pm = UW_PM * (a / r) ** 1.13
    theta = 96.5e-6 * a * (a / r) ** -0.22
    T = UW_T * a / (depth + UW_D) ** (5.0 / 6.0)
    y = np.zeros(_n(3.0 * T + 1.5))
    m = max(16, _n(8.0 * theta))
    _put(y, np.exp(-np.arange(m) / P / theta), 0.0, pm)
    t, gap, g = 0.0, T, 0.84
    for i in range(pulses):
        t += gap
        k = _n(max(0.02, 0.2 * T))
        lump = np.sin(np.pi * np.arange(k) / k) ** 2
        lump = lump * (1.0 + 0.5 * _nm(amb.lop(rng.uniform(-1, 1, k), max(60.0, 4.0 / T))))
        _put(y, lump, t - k / P / 2.0, g * pm * theta / (float(np.sum(lump)) / P))
        kept = UW_KEPT[min(i, len(UW_KEPT) - 1)]
        gap *= kept ** (1.0 / 3.0)
        g *= math.sqrt(kept)
    if surface:
        z = y.copy()
        _put(y, z, 2.0 * depth / C_WATER, -1.0)
    return _nm(amb.lop(y, 4000.0) + 0.1 * y)


def _ring(freqs, decays, dur=0.4, gains=None) -> np.ndarray:
    f = np.zeros(_n(dur))
    f[0] = 1.0
    freqs = np.asarray(freqs, float)
    return _nm(modal.render_modes(f, modal.Modes(freqs, np.asarray(decays, float), np.ones(len(freqs)) if gains is None else np.asarray(gains, float), "vdd")))


def _click(kind, rng, gain=1.0) -> np.ndarray:
    nz = amb.hip(rng.uniform(-1, 1, _n(0.08)), 300.0)
    return _mclick(nz, kind, 0.0, 0.33 * gain)


def _seq(parts, sr, level=0.9, tail=0.05) -> np.ndarray:
    """(signal, seconds, gain) on one timeline. Gain is linear on the signal as given."""
    y = np.zeros(max(_n(t) + len(s) for s, t, _ in parts))
    for s, t, g in parts:
        _put(y, s, t, g)
    return _fin(y, sr, level, fade=0.0005, tail=tail)


def _boom(w_kg, r, rng, frag=0, fireball=1.0, d_mm=5.0, v_range=(300.0, 1218.0), ground=False, ahead=(), **kw) -> list:
    """Blast, fragments, ground wave in Pa, the first arrival at time 0. ``ahead``: more (seconds before the blast, Pa)."""
    ev = (fragments(frag, r, rng, d_mm, v_range) if frag else []) + (ground_wave(w_kg, r, rng) if ground else []) + list(ahead)
    lead = max((a for a, _ in ev), default=0.0)
    return [(blast(w_kg, r, rng, fireball=fireball, **kw), lead, 1.0)] + [(c, lead - a, 1.0) for a, c in ev]


def _shifted(parts, t0):
    return [(s, t + t0, g) for s, t, g in parts]


# shock, negative phase, fireball. "The sound produced is no indicator of destructive force" (Farnell). The ratios estimated, the shape his:
# demolition, a crack of a few ms and nothing after; film, kerosene, "the overpressure wave is very mild"; fuel-air, its negative overpressure.
BLASTS = {"grenade": (1.0, 0.1, 1.0), "demolition": (1.0, 0.1, 0.0), "fuel_air": (1.0, 0.6, 1.5), "film": (0.1, 0.1, 2.0)}


@sfx("blast", "One explosive charge at a distance, in true proportions: the ground's pre-rumble, the shock, three harsh blasts, the fireball. Demolition is a bare crack, film kerosene nearly all fireball, fuel-air has a deep suck behind the shock.",
     kind=("grenade", "|".join(BLASTS)), charge=(1.0, "kg TNT"), distance=(50.0, "m"), ground=(True, "true = the ground wave ahead of the air (up to 5 c)"), seed=(0, "variation"))
def blast_sfx(sr=DEFAULT_SR, kind="grenade", charge=1.0, distance=50.0, ground=True, seed=0):
    kind = str(kind)
    if kind not in BLASTS:
        raise ValueError(f"unknown blast {kind!r}: {'|'.join(BLASTS)}")
    shock, neg, ball = BLASTS[kind]
    rng = np.random.default_rng(int(seed))
    w, r = float(np.clip(charge, 1e-3, 1e5)), float(np.clip(distance, 1.0, 20000.0))
    return _seq(_boom(w, r, rng, fireball=ball, ground=bool(ground), shock=shock, neg=neg), sr, tail=0.3)


@effect("earplug", "Cotton in the ears: -5 dB below 500 Hz, then 5 dB per octave more; special cotton starts at -10 dB. What you hear just after a blast.",
        kind=("cotton", "cotton|special"))
def earplug(x, sr=DEFAULT_SR, kind="cotton"):
    """Farnell, c02 l.15823-15827 (his own 30 dB and 20 dB figures disagree; the curve is what he gives)."""
    x = np.asarray(x, float)
    f = np.fft.rfftfreq(x.shape[0], 1.0 / sr)
    db = -(10.0 if str(kind) == "special" else 5.0) - 5.0 * np.log2(np.maximum(f, 500.0) / 500.0)
    g = 10.0 ** (db / 20.0)
    return np.fft.irfft(np.fft.rfft(x, axis=0) * (g if x.ndim == 1 else g[:, None]), x.shape[0], axis=0)


@sfx("grenade", "Hand and launched grenades: pin, spoon and fuse, then a blast sized by its charge; bouncing, stun, smoke, 40 mm launcher.",
     kind=("frag", "frag|throw_bounce|stun|smoke|gl_pop"), distance=(15.0, "m to the blast"), fuse=(0.0, "s from the spoon to the blast (0 = its own: frag 4.5, range 4-5.5; stun 1.5, range 1-2.3)"),
     dur=(6.0, "s of smoke (50-90 in reality)"), detonate=(False, "throw_bounce: true = the fuse and the blast after the bounces"),
     tinnitus=(0.0, "0..1 stun: the ringing it leaves in your ears, a tone at 4-8 kHz"), seed=(0, "variation"))
def grenade(sr=DEFAULT_SR, kind="frag", distance=15.0, fuse=0.0, dur=6.0, detonate=False, tinnitus=0.0, seed=0):
    rng = np.random.default_rng(int(seed))
    kind, r = str(kind), float(np.clip(distance, 1.0, 5000.0))
    w_frag = 0.18 * RE["comp_b"]                         # M67: 180 g Comp B, W = 0.239 kg
    frag = 8 if r <= 230.0 else 0
    if kind == "throw_bounce":                           # body 64 x 90 mm (M67)
        hop = K.bounce(P if detonate else sr, material="iron", size=0.064, surface="concrete", first_interval=0.25, restitution=0.4, seed=int(seed))
        if not detonate:
            return hop
        t0 = float(fuse) if fuse > 0 else 4.5
        return _seq([(hop, 0.0, 0.05 * blast_wave(w_frag, r)[0])] + _shifted(_boom(w_frag, r, rng, frag=frag, ground=True), t0), sr)
    if kind == "smoke":
        return Ma.air_leak(sr, dur=float(dur), pressure=0.7, orifice=6.0, seed=int(seed))
    if kind == "gl_pop":                                 # M203: 76 m/s, tube 0.305 m (281 Hz); bursts at d / 76 s, heard d / c later; M < 1, no crack
        k = _n(0.03)
        gas = launch_tube(0.305, True, rng.uniform(-1, 1, k) * (1.0 - np.arange(k) / k) ** 2)
        return _seq([(_nm(gas), 0.0, 1.0)] + [(s, t + r / 76.0 + r / C_AIR, g * 0.8) for s, t, g in _boom(0.1, r, rng)], sr)
    pre = 200.0 * (blast_wave(0.239, r)[0] / 2000.0)
    pin = [(_click(CATCH, rng), 0.0, pre), (_ring([3100.0, 5200.0], [40.0, 50.0], 0.2), 0.4, 0.2 * pre)]
    if kind == "stun":                                   # M84: 4.5 g, 170-180 dB, no fragments and no body: the bare shock
        t0 = 0.5 + float(np.clip(fuse if fuse > 0 else 1.5, 0.05, 30.0))
        p = blast_wave(0.0045, r)[0]
        k = _n(0.02)
        holed = launch_tube(0.133, False, rng.uniform(-1, 1, k) * (1.0 - np.arange(k) / k) ** 2)
        parts = pin + _shifted(_boom(0.0045, r, rng, fireball=0.0), t0) + [(_nm(holed), t0, 0.3 * p)]
        if tinnitus > 0:
            k = _n(3.0)
            tone = np.sin(2 * np.pi * rng.uniform(4000.0, 8000.0) * np.arange(k) / P) * np.minimum(np.arange(k) / _n(0.2), 1.0) * (1.0 - np.arange(k) / k) ** 2
            parts.append((tone, t0 + 0.05, 0.1 * p * float(np.clip(tinnitus, 0.0, 1.0))))
        return _seq(parts, sr)
    if kind != "frag":
        raise ValueError(f"unknown grenade {kind!r}")
    t0 = 0.5 + float(np.clip(fuse if fuse > 0 else 4.5, 0.05, 30.0))
    return _seq(pin + _shifted(_boom(w_frag, r, rng, frag=frag, ground=True), t0), sr)


@sfx("molotov", "A bottle breaks and the fuel catches.", distance=(8.0, "m"), dur=(3.0, "s of fire after the catch"), seed=(0, "variation"))
def molotov(sr=DEFAULT_SR, distance=8.0, dur=3.0, seed=0):
    glass = K.smash(P, material="glass", size=0.15, energy=0.6, seed=int(seed))
    ball = Ma.flame(P, kind="fireball", dur=1.2, seed=int(seed))
    fire = Ma.fire(P, dur=float(dur), size=0.5, seed=int(seed))
    y = np.asarray(space.distance(_seq([(glass, 0.0, 1.0), (ball, 0.12, 1.0), (fire, 0.6, 0.5)], P, 1.0), float(distance), ref=1.0))
    return _fin(y, sr, 0.85, fade=0.0005, tail=0.3)


@sfx("mine", "Land mines: anti-tank (a big blast and falling earth), Claymore (clacker, a sheet of 700 balls cracking past if you are in front, then the blast), bounding (pop, jump, blast), arming click.",
     kind=("at", "at|claymore|bounding|arm"), distance=(30.0, "m"), azimuth=(0.0, "claymore: degrees off the direction it faces (cracks only inside its sheet)"), seed=(0, "variation"))
def mine(sr=DEFAULT_SR, kind="at", distance=30.0, azimuth=0.0, seed=0):
    rng = np.random.default_rng(int(seed))
    kind, r = str(kind), float(np.clip(distance, 1.0, 5000.0))
    if kind == "arm":
        return _seq([(_click(ENDSTOP, rng), 0.0, 1.0), (_click(CATCH, rng), 0.03, 0.7)], sr, 0.6)
    if kind == "at":
        boom = _boom(10.0 * RE["comp_b"], r, rng, ground=True)
        dirt = K.hail(P, dur=2.0, rate=50.0, size=3.0, material="concrete", seed=int(seed))
        return _seq(boom + [(dirt, 0.8, 0.05 * blast_wave(13.3, r)[0])], sr, tail=0.5)
    if kind == "claymore":                               # 680 g C-4, M57 clacker (Wikipedia, M18)
        w = 0.68 * RE["c4"]
        p = blast_wave(w, r)[0]
        clack = [(_click(CATCH, rng), 0.0, 0.1 * p), (_click(ENDSTOP, rng), 0.01, 0.1 * p)]
        return _seq(clack + _shifted(_boom(w, r, rng, ahead=ball_sheet(r, float(azimuth), rng)), 0.05), sr)
    if kind == "bounding":
        w = 0.5 * RE["tnt"]
        p = blast_wave(w, r)[0]
        k = _n(0.03)
        pop = amb.lop(rng.uniform(-1, 1, k), 1500.0) * (1.0 - np.arange(k) / k) ** 2
        return _seq([(_nm(pop), 0.0, 0.15 * p)] + _shifted(_boom(w, r, rng, frag=12), 0.5), sr)
    raise ValueError(f"unknown mine {kind!r}")


# length m, rocket length m, motor s beyond the tube, speed m/s, relative back pressure, warhead kg TNT.
# LAW: all burnt in the tube (Wikipedia), 30 ms of it; Farnell lengthens it by choice: `burn`. RPG: motor out to 500 m from 10 m = 1.66 s.
_LAUNCH = {"law": (0.93, 0.51, 0.03, 200.0, 0.6, 0.3), "rpg": (0.95, 0.5, 1.66, 295.0, 1.0, 0.97), "at4": (1.0, 0.5, 0.0, 285.0, 1.5, 0.44)}


def _back_pressure(rng) -> np.ndarray:
    """Farnell Fig 55.7: noise around 100 Hz, up in 100 ms, down in 320 ms on steep curves, severely clipped: "a strong overdriven knocking"."""
    a, b = _n(0.1), _n(0.32)
    env = np.concatenate([(np.arange(a) / a) ** 2, (1.0 - np.arange(b) / b) ** 4])
    return np.clip(4.0 * _nm(amb.bp(rng.uniform(-1, 1, a + b), 100.0, 2.0) * env), -1.0, 1.0)


def _tube_knock(rng=None) -> np.ndarray:
    """M72's two bending modes and its longitudinal one (Farnell, 300, 826, 3200 Hz), direct and ring-modulated: side and end hits."""
    k = _n(0.5)
    t = np.arange(k) / P
    a, b, c = (np.sin(2 * np.pi * f * t) * np.exp(-d * t) for f, d in ((300.0, 20.0), (826.0, 30.0), (3200.0, 60.0)))
    return _nm(a + b + c + 0.5 * a * b + 0.3 * b * c)


@sfx("launcher", "Shoulder rocket launchers: click, the tube closed then opening as the rocket leaves, back blast, gas clearing, a booster and the rocket (M72 LAW, RPG-7 with its second ignition, AT4). An empty tube only clicks.",
     kind=("rpg", "|".join(_LAUNCH)), loaded=(True, "false = an empty tube: the click, nothing else"), burn=(-1.0, "s of motor out of the tube (-1 = the real one: RPG 1.66, LAW and AT4 nothing to speak of)"),
     speed=(0.0, "m/s (0 = its own: LAW 200, 145 on A1-A3)"), target=(0.0, "m: the warhead goes off there, heard back at the launcher (0 = no; RPG self-destructs near 920)"), seed=(0, "variation"))
def launcher(sr=DEFAULT_SR, kind="rpg", loaded=True, burn=-1.0, speed=0.0, target=0.0, seed=0):
    kind = str(kind)
    if kind not in _LAUNCH:
        raise ValueError(f"unknown launcher {kind!r}")
    length, rocket, motor, v, back, w = _LAUNCH[kind]
    rng = np.random.default_rng(int(seed))
    if not loaded:
        return _seq([(_click(CATCH, rng), 0.01, 1.0)], sr, 0.7)      # Farnell: "the firing click after a 10ms delay, but nothing else"
    motor = motor if burn < 0 else float(burn)
    v = float(speed) if speed > 0 else v
    k = _n(0.54)                                                     # exit noise near 1700 Hz, on for 500 ms after the rocket has left (Farnell)
    fire = amb.bp(rng.uniform(-1, 1, k), 1700.0, 1.5) * (1.0 - np.arange(k) / k) ** 4
    ks = _n(0.04)
    closed = launch_tube(length - rocket, True, fire[:ks], f=np.linspace(230.0, 530.0, ks))
    opened = launch_tube(length, False, fire[ks:])
    chamber = np.concatenate([closed, opened]) + 0.2 * fire
    parts = [(_click(CATCH, rng), 0.0, 0.1), (_nm(chamber), 0.01, 1.0), (_back_pressure(rng), 0.01, 0.8 * back), (_tube_knock(), 0.012, 0.25)]
    if kind == "rpg" and motor > 0:                                  # booster 115 m/s, second ignition 10 m out: 87 ms, then 295 m/s
        parts.append((_nm(moving(rocket_motor(0.087, rng, rise=0.01), 115.0, 1.0)), 0.01, 0.5))
        parts.append((_nm(moving(rocket_motor(motor, rng), v, 1.0, x0=10.0)), 0.097, 0.7))
    elif motor > 0:
        parts.append((_nm(moving(rocket_motor(motor, rng), v, 1.0)), 0.01, 0.7))
    if target > 0:
        r = float(target)
        parts.append((_nm(blast(w, r, rng)), 0.01 + r / v + r / C_AIR, 0.5))
    return _seq(parts, sr, tail=0.2)


@sfx("missile", "Guided missiles: a soft eject, then the motor catches at a distance and accelerates away, the Doppler dragging it down until it is gone (Javelin, Stinger).",
     kind=("javelin", "javelin|stinger"), delay=(0.3, "s between eject and ignition (not published)"),
     distance=(0.0, "m ahead of the launcher (0 = you fire it). A Stinger, supersonic, reaches you as a crack first, 157 ms ahead at 100 m"),
     target=(0.0, "m: Javelin's warhead (8.4 kg) goes off there"), seed=(0, "variation"))
def missile(sr=DEFAULT_SR, kind="javelin", delay=0.3, distance=0.0, target=0.0, seed=0):
    kind = str(kind)
    if kind not in ("javelin", "stinger"):
        raise ValueError(f"unknown missile {kind!r}")
    rng = np.random.default_rng(int(seed))
    k = _n(0.08)
    eject = launch_tube(1.1, False, rng.uniform(-1, 1, k) * (1.0 - np.arange(k) / k) ** 2)
    # Javelin: 4750 m in about 19 s. Stinger: Mach 2.2 (2.54 on the same page); Farnell's Mach 4 is not used. Burn under 2 s (Farnell)
    v = 250.0 if kind == "javelin" else 745.0
    dur = 2.0 if kind == "javelin" else 1.5
    src = rocket_motor(dur, rng, st=0.03 if kind == "javelin" else 0.05)
    profile = v * np.arange(len(src)) / len(src)
    parts = [(_nm(eject), 0.0, 0.5), (_nm(moving(src, profile, 3.0)), float(delay), 1.0)]
    if target > 0 and kind == "javelin":
        r = float(target)
        parts.append((_nm(blast(8.4, r, rng)), float(delay) + r / v + r / C_AIR, 0.5))
    d = float(distance)
    if d > 0 and v > C_AIR:
        far = [(s, t + d * (1.0 / C_AIR - 1.0 / v), g * blast_wave(1.0, d)[0]) for s, t, g in parts]
        parts = [(_crack(v, 70.0, 1520.0, 2.0, rng), 0.0, 1.0)] + far
    return _seq(parts, sr, tail=0.3)


@sfx("rocket_salvo", "A battery of Katyusha rockets leaving their rails, one after another, with their howl.",
     count=(16, "rockets (16-24 per launcher)"), rate=(8.0, "departures/s (6.4-13.7 for 4 launchers over 7-10 s)"), seed=(0, "variation"))
def rocket_salvo(sr=DEFAULT_SR, count=16, rate=8.0, seed=0):
    rng = np.random.default_rng(int(seed))
    parts = []
    for i in range(int(np.clip(count, 1, 200))):
        s = rocket_motor(1.2, rng, u=800.0, d=0.132, st=0.15)
        parts.append((_nm(fly_away(s, 250.0, closest=float(rng.uniform(4.0, 20.0)))), i / max(float(rate), 0.5) * rng.uniform(0.9, 1.1), 0.5))
    return _seq(parts, sr, tail=0.4)


@sfx("rocket_flyby", "A rocket passing you: motor, Doppler drop, swell.", speed=(295.0, "m/s (RPG 295)"), closest=(10.0, "m"), dur=(2.0, "s"), seed=(0, "variation"))
def rocket_flyby(sr=DEFAULT_SR, speed=295.0, closest=10.0, dur=2.0, seed=0):
    rng = np.random.default_rng(int(seed))
    y = doppler(rocket_motor(float(dur), rng), P, speed=float(speed), closest=float(closest), pass_at=0.5, pan=False)
    return _fin(np.asarray(y), sr, 0.85, tail=0.2)


@sfx("launcher_handling", "Handling a launcher tube: shoulder knocks, latches, a rocket sliding home (M72 tube modes 300, 826, 3200 Hz); a loaded tube rings differently from an empty one.",
     action=("load", "load|drop|latch"), loaded=(True, "drop: true = with a rocket inside"), seed=(0, "variation"))
def launcher_handling(sr=DEFAULT_SR, action="load", loaded=True, seed=0):
    rng = np.random.default_rng(int(seed))
    action = str(action)
    knock = _tube_knock()
    length, rocket = 0.93, 0.51
    k = _n(0.02)
    latch = _nm(amb.bp(rng.uniform(-1, 1, k), 5000.0, 12.0) * (1.0 - np.arange(k) / k) ** 2)    # 20 ms, Q 12, square-law envelope (Farnell)
    if action == "latch":
        return _seq([(latch, 0.0, 1.0)], sr, 0.7)
    if action == "drop":                                   # the four knocks: 90, 20, 110 ms apart; felt through the tube's air too
        air = _nm(launch_tube(length - rocket, True, knock) if loaded else launch_tube(length, False, knock))
        tube = _nm(knock + 0.5 * air)
        return _seq([(tube, t, g) for t, g in ((0.0, 1.0), (0.09, 0.6), (0.11, 0.4), (0.22, 0.25))], sr, 0.8)
    if action != "load":
        raise ValueError(f"unknown action {action!r}")
    m = _n(0.3)
    # friction: hip 1500, band 6512 Hz Q 45 (Farnell's patch, c10), through the chamber swept the other way as the rocket goes in
    rub = amb.bp(amb.hip(rng.uniform(-1, 1, m), 1500.0), 6512.0, 45.0) * np.sin(np.pi * np.arange(m) / m)
    rub = _nm(rub) + 0.5 * _nm(launch_tube(length - rocket, True, rng.uniform(-1, 1, m) * np.sin(np.pi * np.arange(m) / m) * 0.2, f=np.linspace(530.0, 230.0, m)))
    return _seq([(rub, 0.0, 0.5), (knock, 0.3, 0.4), (latch, 0.31, 0.8), (latch, 0.34, 0.8)], sr, 0.8)


# K5. x along the line of fire, y off it. Crack ahead of the blast when supersonic and inside the cone; none past 75 degrees (Murphy 2019, as firearm).
def shell_pass(w_kg, v, d_mm, l_mm, x, y, rng, fireball=0.4, tube=None, boom=True) -> list:
    r = math.hypot(x, y)
    b = max(y, 0.5)
    t_c = shock_arrival(v, x, b) if math.degrees(math.atan2(y, x)) <= 75.0 else None
    t_b = r / C_AIR
    t0 = min(t_c, t_b) if t_c is not None else t_b
    parts = []
    if boom:
        bl = blast(w_kg, r, rng, fireball=fireball)
        parts.append((bl, t_b - t0, 1.0))
        if tube is not None:
            parts.append((_nm(launch_tube(tube, False, bl[:_n(0.3)])) * float(np.max(np.abs(bl))), t_b - t0, 0.4))
    if t_c is not None:
        parts.append((_crack(v, d_mm, l_mm, b, rng), t_c - t0, 1.0))
    return parts


def _muzzle(w_kg, r, az, v, d_mm, l_mm, length, rng) -> list:
    th = math.radians(az)
    return shell_pass(w_kg, v, d_mm, l_mm, r * math.cos(th), r * math.sin(th), rng, tube=length)


TUBE_81 = shell_modes(40.5, 8.0, 1565.0)


@sfx("mortar", "An 81 mm mortar: the round slides down the tube, the charge kicks it out; or its impact.",
     kind=("fire", "drop|fire|impact"), distance=(30.0, "m (impact)"), seed=(0, "variation"))
def mortar(sr=DEFAULT_SR, kind="fire", distance=30.0, seed=0):
    rng = np.random.default_rng(int(seed))
    kind = str(kind)
    if kind == "impact":                                   # 0.7 kg fill: W = 0.93 kg with Comp B's RE assumed
        return _seq(_boom(0.7 * RE["comp_b"], float(distance), rng, frag=6, ground=True), sr, tail=0.3)
    if kind not in ("drop", "fire"):
        raise ValueError(f"unknown mortar {kind!r}")
    k = _n(0.05)
    thump = launch_tube(1.27, True, rng.uniform(-1, 1, k) * (1.0 - np.arange(k) / k) ** 2, modes=5)      # 67.5 Hz; 241 m/s at least, subsonic: no crack
    gas = amb.lop(rng.uniform(-1, 1, _n(0.3)), 500.0) * np.exp(-np.arange(_n(0.3)) / P / 0.08)
    fire = [(_nm(thump), 0.0, 1.0), (_nm(gas), 0.0, 0.8), (_ring(TUBE_81, [40.0, 60.0, 80.0], 0.4), 0.0, 0.2)]
    if kind == "fire":
        return _seq(fire, sr, tail=0.2)
    slide = K.scrape(P, dur=0.51, speed=1.0, material="steel", size=0.8, seed=int(seed))        # sqrt(2 L / g) = 0.51 s
    n = len(slide)
    # the air column grows as the round falls: L = g t^2 / 2 from the muzzle; 0.15 m at least
    col = np.maximum(0.5 * 9.81 * (np.arange(n) / P) ** 2, 0.15)
    air = launch_tube(1.27, True, slide, modes=5, f=C_AIR / (4.0 * col))
    return _seq([(_nm(slide), 0.0, 0.25), (_nm(air), 0.0, 0.25)] + _shifted(fire, 0.51), sr, tail=0.2)


@sfx("howitzer", "A 155 mm howitzer: muzzle blast rolling through a long barrel, the shell's crack ahead of it when it flies your way.",
     distance=(100.0, "m"), azimuth=(30.0, "degrees off the line of fire: crack up to 75"), seed=(0, "variation"))
def howitzer(sr=DEFAULT_SR, distance=100.0, azimuth=30.0, seed=0):
    rng = np.random.default_rng(int(seed))
    return _seq(_muzzle(6.0, float(distance), float(azimuth), 827.0, 155.0, 700.0, 5.08, rng), sr, tail=0.3)


@sfx("tank_gun", "A 120 mm tank gun: an enormous blast, a Mach 5 crack far ahead, the breech slamming open.",
     distance=(100.0, "m"), azimuth=(30.0, "degrees off the line of fire"), breech=(True, "true = with the breech"), seed=(0, "variation"))
def tank_gun(sr=DEFAULT_SR, distance=100.0, azimuth=30.0, breech=True, seed=0):
    rng = np.random.default_rng(int(seed))
    parts = _muzzle(4.0, float(distance), float(azimuth), 1750.0, 27.0, 500.0, 6.6, rng)
    if breech:
        p = blast_wave(4.0, float(distance))[0]
        parts.append((_nm(metal_plate(P, size=0.6, thickness=20.0, decay=0.6)), 1.2, 0.06 * p))
    return _seq(parts, sr, tail=0.3)


@sfx("shell_incoming", "A shell falling on you: a whistle climbing as it comes closer (an estimate, nothing measured), then the blast. A negative sweep is the film's falling whistle.",
     dur=(2.5, "s of whistle"), f0=(900.0, "Hz at the start"), sweep=(1.0, "octaves over the fall"), distance=(60.0, "m"), charge=(6.0, "kg TNT"), seed=(0, "variation"))
def shell_incoming(sr=DEFAULT_SR, dur=2.5, f0=900.0, sweep=1.0, distance=60.0, charge=6.0, seed=0):
    rng = np.random.default_rng(int(seed))
    n = _n(float(dur))
    t = np.arange(n) / n
    f = float(f0) * 2.0 ** (float(sweep) * t)
    tone = np.sin(2 * np.pi * np.cumsum(f) / P) + 0.3 * Ma._tvbp(rng.uniform(-1, 1, n), f, 3.0)
    p = blast_wave(float(charge), float(distance))[0]
    return _seq([(_nm(tone * t ** 2), 0.0, 0.3 * p)] + _shifted(_boom(float(charge), float(distance), rng, frag=6, ground=True), float(dur)), sr, tail=0.3)


@sfx("shell_crack_overhead", "An artillery shell going over at Mach 2.4: a hard crack, and the gun's boom long after.",
     miss=(50.0, "m it passes overhead"), range=(1000.0, "m from the gun: the boom comes this much later"), boom=(True, "true = with the distant gun"), seed=(0, "variation"))
def shell_crack_overhead(sr=DEFAULT_SR, miss=50.0, range=1000.0, boom=True, seed=0):
    rng = np.random.default_rng(int(seed))
    r = float(np.clip(range, 50.0, 30000.0))
    return _seq(shell_pass(6.0, 827.0, 155.0, 700.0, r, float(miss), rng, fireball=0.5, boom=bool(boom)), sr, tail=0.3)


@sfx("aerial_bomb", "A falling bomb's whistle and a Mk 82 blast (89 kg Tritonal), the rumble scaled to its charge.",
     distance=(200.0, "m"), whistle=(True, "true = the fall first"), charge=(93.5, "kg TNT"), seed=(0, "variation"))
def aerial_bomb(sr=DEFAULT_SR, distance=200.0, whistle=True, charge=93.5, seed=0):
    rng = np.random.default_rng(int(seed))
    w, r = float(charge), float(distance)
    if not whistle:
        return _seq(_boom(w, r, rng, ground=True), sr, tail=0.8)
    n = _n(3.0)
    t = np.arange(n) / n
    tone = np.sin(2 * np.pi * np.cumsum(1200.0 * 2.0 ** (-1.5 * t)) / P)
    return _seq([(tone * t, 0.0, 0.2 * blast_wave(w, r)[0])] + _shifted(_boom(w, r, rng, ground=True), 3.0), sr, tail=0.8)


@sfx("flamethrower", "An M2 flamethrower: a roaring jet (37 m/s through 8 mm, ~920 Hz), its igniter, the nitrogen emptying, the fire left behind.",
     kind=("burst", "burst|ignite|empty|tail"), dur=(7.0, "s (7 s is a full tank)"), seed=(0, "variation"))
def flamethrower(sr=DEFAULT_SR, kind="burst", dur=7.0, seed=0):
    rng = np.random.default_rng(int(seed))
    kind, dur = str(kind), float(np.clip(dur, 0.2, 30.0))
    if kind == "ignite":
        cart = rocket_motor(0.3, rng, d=0.02, rise=0.01)
        return _seq([(_nm(Ma.spark(P, seed=int(seed))), 0.0, 0.7), (cart, 0.01, 0.5), (_nm(Ma.flame(P, kind="fireball", dur=0.6, seed=int(seed))), 0.05, 1.0)], sr)
    if kind == "empty":
        return Ma.air_leak(sr, dur=dur, pressure=0.8, orifice=8.0, burst=dur / 3.0, seed=int(seed))
    if kind == "tail":
        return Ma.fire(sr, dur=dur, size=0.5, seed=int(seed))
    if kind != "burst":
        raise ValueError(f"unknown flamethrower {kind!r}")
    n = _n(dur)
    jet = Ma.jet_noise(n, rng, 37.0, 0.008, u_ref=37.0)
    roar = Ma.combustion_roar(n, rng, area=2.0)
    env = np.minimum(np.arange(n) / _n(0.08), 1.0)
    return _fin(_nm(jet) * env + 0.5 * _nm(roar) * env, sr, 0.85, tail=0.25)


def _wet(y):
    return amb.lop(y, 2500.0)


@sfx("torpedo", "Torpedoes under water: tube launch, steam and piston engines passing, a supercavitating rocket, the hit.",
     kind=("steam", "launch|steam|piston|supercav|hit"), dur=(4.0, "s"), speed=(0.0, "m/s (0 = its own: steam 23.7, or 15.9 low; piston 29; supercav 25.7 -> 103)"), distance=(50.0, "m (hit)"),
     charge=(380.0, "kg TNT (hit: Mk 14 292 kg Torpex, or 230 of TNT)"), depth=(10.0, "m (hit)"), seed=(0, "variation"))
def torpedo(sr=DEFAULT_SR, kind="steam", dur=4.0, speed=0.0, distance=50.0, charge=380.0, depth=10.0, seed=0):
    rng = np.random.default_rng(int(seed))
    kind, dur = str(kind), float(np.clip(dur, 0.2, 60.0))
    if kind == "launch":
        hatch = metal_plate(P, size=0.5, thickness=10.0, decay=0.5)
        air = Ma.air_leak(P, dur=0.8, pressure=0.9, orifice=20.0, burst=0.3, seed=int(seed))
        foam = Ma.bubbles(P, dur=1.5, rate=40.0, size=5.0, underwater=1.0, seed=int(seed))
        return _seq([(_nm(hatch), 0.0, 0.5), (_nm(_wet(air)), 0.05, 1.0), (_nm(foam), 0.3, 0.7)], sr, tail=0.2)
    if kind == "hit":
        return _fin(underwater_blast(float(charge), float(distance), float(depth), rng), sr, 0.9, fade=0.0005, tail=0.3)
    n = _n(dur)
    t = np.arange(n) / P
    if kind == "steam":
        v, blade = float(speed) if speed > 0 else 23.7, 60.0
        src = amb.bp(rng.uniform(-1, 1, n), 2000.0, 0.7) * (0.6 + 0.4 * np.sin(2 * np.pi * blade * t)) + 0.3 * np.sin(2 * np.pi * 720.0 * t)
    elif kind == "piston":
        v = float(speed) if speed > 0 else 29.0
        src = amb.lop(rng.uniform(-1, 1, n), 1500.0) + 0.3 * np.sign(np.sin(2 * np.pi * 45.0 * t)) * amb.lop(rng.uniform(-1, 1, n), 300.0)
    elif kind == "supercav":
        top = float(speed) if speed > 0 else 103.0
        v = 25.7 + (top - 25.7) * np.minimum(t / (0.5 * dur), 1.0)
        src = _nm(Ma.combustion_roar(n, rng, area=3.0)) + 0.5 * _nm(Ma.bubbles(P, dur=dur, rate=200.0, size=3.0, underwater=1.0, seed=int(seed))[:n])
        y = moving(src, v, 10.0, x0=-float(np.sum(v)) / P * 0.5, c=C_WATER)
        return _fin(_wet(y), sr, 0.8, tail=0.3)
    else:
        raise ValueError(f"unknown torpedo {kind!r}")
    y = np.asarray(doppler(src, P, speed=v, closest=10.0, pass_at=0.5, pan=False, air=False, c=C_WATER))
    return _fin(_wet(y), sr, 0.8, tail=0.3)


@sfx("depth_charge", "A depth charge from inside a hull: the distant thump, its bubble pulses, the plates ringing, drops falling.",
     charge=(118.0, "kg TNT (Mk 9: 91 kg Torpex)"), distance=(50.0, "m"), depth=(75.0, "m"), hull=(0.5, "0..1 hull rattling"),
     sink=(False, "true = the splash first, then the fall at 4.4 m/s (17 s to 75 m)"), seed=(0, "variation"))
def depth_charge(sr=DEFAULT_SR, charge=118.0, distance=50.0, depth=75.0, hull=0.5, sink=False, seed=0):
    rng = np.random.default_rng(int(seed))
    shot = underwater_blast(float(charge), float(distance), float(depth), rng)
    h = float(np.clip(hull, 0.0, 1.0))
    ring = _nm(metal_plate(P, size=2.0, thickness=25.0, decay=1.5))
    drops = _nm(Ma.drops(P, dur=2.5, rate=6.0, rate_end=0.5, seed=int(seed)))
    parts = [(shot, 0.0, 1.0), (ring, 0.01, 0.4 * h), (drops, 0.3, 0.1 * h)]
    if sink:
        parts = [(_nm(Ma.splash(P, size=0.7, dur=1.5, seed=int(seed))), 0.0, 0.2)] + _shifted(parts, float(depth) / 4.4)
    return _seq(parts, sr, tail=0.4)


@sfx("sea_mine", "A sea mine going off near the surface, heard from above: a thump cut short by the surface, and the column of water falling back.",
     charge=(100.0, "kg TNT"), distance=(100.0, "m"), seed=(0, "variation"))
def sea_mine(sr=DEFAULT_SR, charge=100.0, distance=100.0, seed=0):
    rng = np.random.default_rng(int(seed))
    shot = underwater_blast(float(charge), float(distance), 3.0, rng, pulses=1, surface=True)
    plume = Ma.splash(P, size=1.0, dur=3.0, droplets=24, seed=int(seed))
    return _seq([(shot, 0.0, 1.0), (_nm(plume), 0.4, 0.6)], sr, tail=0.4)


@sfx("blast_noise", "Farnell's sinusoidal noise: an exploding-shell texture, from noise through a closing resonance, clipped, then used as phase.",
     f=(0.5, "0..1 small and sharp (high) .. 1 short and thin; the length is (1-f) x 2.8 s"), seed=(0, "variation"))
def blast_noise(sr=DEFAULT_SR, f=0.5, seed=0):
    """`pd blast` (Chris Nash's patches): vcf gliding from 20 Hz to 7000 f^2 Hz in 30 ms, Q = 1.5 + 3.5(1-f); x 50 and clipped; y = cos(2 pi x (0.5 + e^2));
    band 6000 Hz, Q 1, x e^2; high-pass 20 Hz x 3, x 0.3; e falls linearly from 1 to 0 over (1-f) 0.397 x 7 s."""
    from . import matter as Ma
    f = float(np.clip(f, 0.05, 0.95))
    n = Ma._n((1.0 - f) * 0.397 * 7.0)
    t = np.arange(n) / Ma.SR
    e = np.clip(1.0 - np.arange(n) / n, 0.0, 1.0)
    rng = Ma._rng(seed, 720)
    fc = 20.0 + (max(7000.0 * f * f, 21.0) - 20.0) * np.minimum(t / 0.03, 1.0)
    x = np.clip(50.0 * Ma._tvbp(rng.standard_normal(n), fc, 1.5 + 3.5 * (1.0 - f)), -1.0, 1.0)
    y = np.cos(2.0 * np.pi * x * (0.5 + e * e))
    return Ma._fin(0.3 * 3.0 * Ma._hp(Ma._bp(y, 6000.0, 1.0) * e * e, 20.0), sr, 0.8)
