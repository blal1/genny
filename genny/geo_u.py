"""Granular and geophysical flows, ice and dunes from measurable quantities (mass, speed, grain diameter, ice thickness, temperature).

Sources of every law and constant: research/SOURCES.md, section FAM_U.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import fftconvolve

from . import geo as G
from . import matter as Ma
from .core import DEFAULT_SR
from .sfx import sfx

SR = Ma.SR
GRAV = 9.81

# E (Pa), Poisson ratio, density (kg/m3) of the two bodies of a contact
MATERIALS = {
    "rock": (50e9, 0.25, 2600.0),
    "sand": (70e9, 0.17, 2650.0),
    "snow": (0.1e9, 0.2, 300.0),
    "ice": (9.33e9, 0.325, 919.5),
}
ICE_SHEAR_SPEED = 1957.0     # m/s
BRUNE_K = 0.37               # corner frequency = K beta / r
DEBRIS_RATE_K = 20.0         # impacts a second per (v / D) at the sensor
DEBRIS_IMPACT_SPEED = 0.3    # share of the flow speed of the impact speed of a grain
BOOM_K = 0.4                 # f = K sqrt(g / d)
CLOUD_SPECTRUM = 5.0 / 3.0   # slope of the power spectrum of the turbulent cloud above U/L


def hertz_contact_time(mass: float, speed: float, diameter: float, material: str = "rock") -> float:
    """s: contact time of a sphere of ``diameter`` m and ``mass`` kg hitting a half-space of the same material at ``speed`` m/s,
    t = 2.87 (m^2 / (R E*^2 v))^(1/5), E* = E / (2 (1 - nu^2)), R = d / 2 (flat target)."""
    e, nu, _ = MATERIALS[material]
    es = e / (2.0 * (1.0 - nu * nu))
    return 2.87 * (mass * mass / (0.5 * diameter * es * es * max(speed, 1e-6))) ** 0.2


def sphere_mass(diameter, material: str = "rock"):
    return MATERIALS[material][2] * math.pi / 6.0 * np.asarray(diameter, float) ** 3


def power_law_sizes(n: int, dmin: float, dmax: float, b: float, rng) -> np.ndarray:
    """m: n diameters with N(>d) proportional to d^-b between dmin and dmax (inverse of the truncated Pareto law)."""
    u = rng.random(n)
    a, c = dmin ** -b, dmax ** -b
    return (a - u * (a - c)) ** (-1.0 / b)


def hill_exponent(d, dmin: float) -> float:
    """Maximum-likelihood exponent b of N(>d) ~ d^-b for the sizes above dmin."""
    d = np.asarray(d, float)
    d = d[d >= dmin]
    return len(d) / float(np.sum(np.log(d / dmin)))


def grain_rate(flow: float, grain_mm: float, material: str = "sand") -> float:
    """grains a second in a flow of ``flow`` kg/s of grains of ``grain_mm`` mm."""
    return flow / float(sphere_mass(grain_mm * 1e-3, material))


def boom_hz(grain_um: float) -> float:
    """Hz: f = 0.4 sqrt(g / d) of a booming avalanche."""
    return BOOM_K * math.sqrt(GRAV / (float(grain_um) * 1e-6))


def squeak_hz(speed: float, grain_um: float) -> float:
    """Hz: stick-slip of a grain layer sliding at ``speed`` m/s, f = speed / d."""
    return float(speed) / (float(grain_um) * 1e-6)


def brune_corner(radius: float) -> float:
    """Hz: corner frequency of a rupture of radius ``radius`` m in ice, 0.37 beta / r."""
    return BRUNE_K * ICE_SHEAR_SPEED / max(float(radius), 1e-3)


def _pulse(m: int) -> np.ndarray:
    if m <= 1:
        return np.ones(1)
    p = np.sin(np.pi * (np.arange(m) + 0.5) / m)
    return p / p.sum()


def shot_noise(n: int, rate, impulse, tc: float, rng, jitter: float = 0.4) -> np.ndarray:
    """Poisson train of impacts of impulse ``impulse`` (arbitrary units, scalar or array) at ``rate`` a second, each a half sine of
    ``tc`` s with its area kept. Variance is rate x impulse^2 x (1 + jitter terms)."""
    lam = np.broadcast_to(np.asarray(rate, float), (n,)) / SR
    cnt = rng.poisson(lam).astype(float)
    amp = rng.lognormal(-0.5 * jitter ** 2, jitter, n)
    y = cnt * amp * np.broadcast_to(np.asarray(impulse, float), (n,)) * SR
    m = int(round(tc * SR))
    return fftconvolve(y, _pulse(m))[:n] if m > 1 else y


def _un(v):
    return v / (np.max(np.abs(v)) + 1e-12)


def rockfall_blocks(dur, blocks, dmin, dmax, b, height, restitution, rng, material="rock", roll=0.5, ground_hz=60.0):
    """Unscaled signal (and the list of diameters) of blocks of power-law sizes falling ``height`` m then bouncing down:
    each impact is a Hertz force pulse, the bounces lose speed by ``restitution``, the block rolls at the end."""
    n = Ma._n(dur)
    y = np.zeros(n)
    sizes = power_law_sizes(int(blocks), dmin, dmax, b, rng)
    for d in sizes:
        m = float(sphere_mass(d, material))
        t = rng.uniform(0.0, 0.6 * dur)
        v = math.sqrt(2.0 * GRAV * max(height, 0.05)) * rng.uniform(0.6, 1.0)
        k = 0
        while t < dur and v > 0.3 and k < 14:
            tc = hertz_contact_time(m, v, d, material)
            j = (1.0 + restitution) * m * v
            mm = max(int(round(tc * SR)), 1)
            s = int(t * SR)
            if s < n:
                p = _pulse(mm) * j * SR
                q = min(len(p), n - s)
                y[s:s + q] += p[:q]
                G._ring(y, t, ground_hz * rng.uniform(0.7, 1.4), 0.06, j * 20.0)
            v *= restitution
            t += 2.0 * v / GRAV * rng.uniform(0.5, 1.0) + 0.004
            k += 1
        if roll > 0.0 and t < dur:
            vr = max(v, 0.3) * 1.5
            span = min(float(roll) * d * 10.0 + 0.2, dur - t)
            s0, s1 = int(t * SR), int((t + span) * SR)
            if s1 - s0 > 8:
                rate = np.full(s1 - s0, 3.0 * vr / d)
                tcr = hertz_contact_time(m, 0.1 * vr, d, material)
                y[s0:s1] += shot_noise(s1 - s0, rate, m * 0.1 * vr, tcr, rng) * np.hanning(s1 - s0) ** 0.5
    return y, sizes


@sfx("rockfall_hertz", "Rockfall from measurable quantities: blocks of power-law sizes N(>d) ~ d^-b, each impact a Hertz contact of its own mass, diameter and speed, rebounds that lose speed, then rolling.",
     dur=(6.0, "s"), blocks=(40, "number of blocks"), dmin=(0.05, "m smallest block"), dmax=(1.0, "m largest block"),
     b=(1.5, "exponent of the size law N(>d) ~ d^-b (volume exponent about 0.5 x ... 1; 1.5 here for the diameter)"),
     height=(15.0, "m of the first fall"), restitution=(0.5, "0..0.9 speed kept at each rebound"), roll=(0.5, "0..2 rolling after the rebounds"),
     material=("rock", "rock|sand|snow|ice (elastic constants of the contact)"), seed=(0, "variation"))
def rockfall_hertz(sr=DEFAULT_SR, dur=6.0, blocks=40, dmin=0.05, dmax=1.0, b=1.5, height=15.0, restitution=0.5, roll=0.5, material="rock", seed=0):
    if material not in MATERIALS:
        raise ValueError("material must be one of: " + ", ".join(MATERIALS))
    rng = Ma._rng(seed, 761)
    lo = float(np.clip(dmin, 0.005, 2.0))
    y, _ = rockfall_blocks(float(np.clip(dur, 0.5, 60.0)), int(np.clip(blocks, 1, 400)), lo, max(float(dmax), lo * 1.5),
                           float(np.clip(b, 0.3, 4.0)), float(height), float(np.clip(restitution, 0.0, 0.9)), rng, material, float(np.clip(roll, 0.0, 2.0)))
    return Ma._fin(y, sr, 0.8)


def debris_impacts(n: int, speed: float, grain: float, rng, material: str = "rock", mud: float = 0.0, rate_k: float = DEBRIS_RATE_K):
    """Impact shot noise of a debris flow at ``speed`` m/s with effective grain ``grain`` m (Hertz impacts at 0.3 of the flow speed,
    rate k v / D); mud lowers the cut of each impact. Power goes as v^3 D^5."""
    vi = DEBRIS_IMPACT_SPEED * speed
    m = float(sphere_mass(grain, material))
    tc = hertz_contact_time(m, vi, grain, material)
    y = shot_noise(n, rate_k * speed / grain, 1.5 * m * vi, tc, rng)
    return y if mud <= 0 else Ma._lp(y, max(30.0, (1.0 - 0.9 * float(np.clip(mud, 0.0, 1.0))) / max(tc, 1e-5)), 2)


@sfx("debris_flow_granular", "Debris flow or lahar from speed, depth and grain size: Hertz impacts of the grains in the flow (power ~ v^3), a 20-50 Hz roar that grows with speed, mud that darkens and water that hisses.",
     dur=(20.0, "s"), speed=(5.0, "m/s of the flow, 1..20"), grain=(0.1, "m effective grain diameter of the snout"), depth=(1.5, "m of the flow"),
     mud=(0.5, "0..1 mud fraction: darkens the impacts"), water=(0.3, "0..1 water on top: hiss"), arrival=(0.4, "0..0.9 share of dur before the front passes"),
     seed=(0, "variation"))
def debris_flow_granular(sr=DEFAULT_SR, dur=20.0, speed=5.0, grain=0.1, depth=1.5, mud=0.5, water=0.3, arrival=0.4, seed=0):
    rng = Ma._rng(seed, 762)
    n = Ma._n(dur)
    u = np.arange(n) / n
    ar = float(np.clip(arrival, 0.0, 0.9))
    sp = float(np.clip(speed, 0.5, 25.0))
    prof = np.where(u < ar, 0.15 + 0.85 * (u / max(ar, 1e-3)) ** 2, np.exp(-(u - ar) / 0.35))
    imp = debris_impacts(n, sp, float(np.clip(grain, 0.005, 1.0)), rng, "rock", float(mud))
    imp = imp * prof ** 1.5
    low = Ma._bp(rng.standard_normal(n), 35.0, 0.8) * prof * (sp / 5.0) ** 1.5 * float(np.clip(depth, 0.1, 10.0)) ** 0.5
    hiss = Ma._hp(Ma._lp(rng.standard_normal(n), 3500.0), 500.0) * prof * float(np.clip(water, 0.0, 1.0))
    return Ma._fin(_un(imp) * 0.6 + _un(low) * 0.8 + 0.3 * _un(hiss), sr, 0.8)


def avalanche_cloud(n: int, speed, size: float, rng, density: float = 1.0) -> np.ndarray:
    """Turbulent powder cloud: noise flat below U / L and falling as f^-5/3 (power) above, amplitude going as density x U^4
    (acoustic intensity of turbulence ~ rho^2 U^8). ``speed`` may be an array of n values."""
    f = np.fft.rfftfreq(n, 1.0 / SR)
    u0 = float(np.mean(speed))
    fc = u0 / max(float(size), 1.0)
    h = (1.0 + (f / fc) ** 2) ** (-CLOUD_SPECTRUM / 4.0)
    z = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * h, n)
    z = z / (math.sqrt(float(np.mean(z * z))) + 1e-30)
    return z * density * np.asarray(speed, float) ** 4


@sfx("avalanche_flow", "Avalanche from speed and size: powder cloud (turbulence, amplitude ~ U^4, flat below U/L and -5/3 above) over a dense flow of snow blocks (Hertz thuds of soft snow) with its 8-40 Hz rumble; the 2-10 Hz infrasound can be added.",
     dur=(14.0, "s"), speed=(30.0, "m/s of the front, 5..80"), size=(100.0, "m of the cloud: lowers its corner U / L"),
     powder=(0.5, "0..1 powder cloud against dense flow"), chunk=(0.25, "m snow block of the dense flow"), infra=(0.0, "0..1 level of the 2-10 Hz infrasound (inaudible: above 0 it is rendered for analysis)"),
     seed=(0, "variation"))
def avalanche_flow(sr=DEFAULT_SR, dur=14.0, speed=30.0, size=100.0, powder=0.5, chunk=0.25, infra=0.0, seed=0):
    rng = Ma._rng(seed, 763)
    n = Ma._n(dur)
    u = np.arange(n) / n
    uf = float(np.clip(speed, 5.0, 80.0)) * np.minimum(u / 0.2, 1.0) * np.clip((1.0 - u) / 0.5, 0.0, 1.0) ** 0.5 + 1.0
    pw = float(np.clip(powder, 0.0, 1.0))
    cloud = avalanche_cloud(n, uf, float(size), rng) * np.sin(np.pi * np.clip(u / 0.6, 0.0, 1.0)) ** 2
    d = float(np.clip(chunk, 0.02, 2.0))
    m = float(sphere_mass(d, "snow"))
    vi = 0.3 * uf
    dense = shot_noise(n, 15.0 * uf / d, 1.5 * m * vi, hertz_contact_time(m, float(np.mean(vi)), d, "snow"), rng)
    rumble = Ma._bp(rng.standard_normal(n), 20.0, 0.6) * (uf / float(np.max(uf))) ** 2
    out = pw * _un(cloud) * 0.6 + (1.0 - 0.5 * pw) * (0.5 * _un(dense) + 0.7 * _un(rumble))
    ia = float(np.clip(infra, 0.0, 1.0))
    if ia > 0.0:
        out = out + ia * _un(Ma._hp(Ma._lp(cloud, 10.0, 2), 2.0, 2))
    return Ma._fin(out, sr, 0.8)


@sfx("icequake_swarm", "Icequakes in a glacier or lake ice: events with power-law rupture sizes, each source low-passed at its Brune corner (0.37 beta / r) and dispersed by the flexural waves of a plate of the given thickness (w^2 = D k^4 / rho h).",
     dur=(8.0, "s"), rate=(1.5, "events a second"), size=(1.0, "m smallest rupture radius"), thickness=(0.3, "m of the ice plate"), distance=(200.0, "m to the events"),
     b=(1.0, "Gutenberg-Richter-like exponent of N(>r) ~ r^-2b"), temp=(-10.0, "C, -50..0: sets the loss"), seed=(0, "variation"))
def icequake_swarm(sr=DEFAULT_SR, dur=8.0, rate=1.5, size=1.0, thickness=0.3, distance=200.0, b=1.0, temp=-10.0, seed=0):
    rng = Ma._rng(seed, 764)
    n = Ma._n(dur)
    y = np.zeros(n)
    k = rng.poisson(max(float(rate), 0.05) * dur) + 1
    r = power_law_sizes(k, float(np.clip(size, 0.1, 50.0)), 50.0 * float(np.clip(size, 0.1, 50.0)), 2.0 * float(np.clip(b, 0.3, 3.0)), rng)
    for t, rr in zip(rng.uniform(0.0, 0.9 * dur, k), r):
        d = float(np.clip(distance, 5.0, 1500.0)) * rng.uniform(0.7, 1.3)
        c = Ma.ice_chirp(d, float(np.clip(thickness, 0.02, 2.0)), temp=float(np.clip(temp, -50.0, 0.0)))
        c = Ma._lp(c, min(brune_corner(rr), 0.4 * SR), 2)
        Ma._place(y, c / (np.max(np.abs(c)) + 1e-12), int(t * SR), rr ** 1.5)
    return Ma._fin(y, sr, 0.8)


@sfx("iceberg_fizz", "A melting iceberg: trapped air bubbles bursting (Minnaert rings, 3.26/r) in a fizz, cracks as it weakens and the low slap of water.",
     dur=(8.0, "s"), melt=(0.5, "0..1 melt rate: bubble rate and crack rate"), bubble=(1.5, "mm mean bubble radius, 0.3..5"), seed=(0, "variation"))
def iceberg_fizz(sr=DEFAULT_SR, dur=8.0, melt=0.5, bubble=1.5, seed=0):
    rng = Ma._rng(seed, 765)
    n = Ma._n(dur)
    m = float(np.clip(melt, 0.0, 1.0))
    fz = np.zeros(n)
    for t in rng.uniform(0.0, dur, int(300 * m * dur) + 1):
        r = float(np.clip(bubble, 0.3, 5.0)) * 1e-3 * rng.lognormal(0.0, 0.4)
        f = 3.26 / r
        G._ring(fz, t, f, 8.0 / f, rng.uniform(0.2, 1.0))
    cr = np.zeros(n)
    for t in rng.uniform(0.0, dur, rng.poisson(1.5 * m * dur)):
        G._grain(cr, t, rng.uniform(300.0, 1500.0), rng.uniform(0.4, 1.0), rng)
    slap = Ma._lp(rng.standard_normal(n), 120.0, 2) * (0.4 + 0.6 * np.abs(Ma._lfn(rng, n, 0.7)))
    return Ma._fin(0.6 * _un(fz) + 0.5 * _un(cr) + 0.4 * _un(slap), sr, 0.8)


@sfx("dune_sound", "Sand that sounds from grain size and speed: boom (avalanche of dry sand, one note 0.4 sqrt(g/d) with its harmonics and a slow beat), squeak (slip of a layer, f = speed / d, about 1 kHz) or silent (the same avalanche without locking: only the hiss of grains).",
     mode=("boom", "boom|squeak|silent"), grain=(170.0, "um grain diameter (170 um: 96 Hz)"), speed=(0.2, "m/s of the slip, used by squeak"),
     hz=(0.0, "Hz: force the note (booming dunes sing 70-105 Hz whatever the grain); 0 = from the grain"), dur=(6.0, "s"), beat=(1.0, "Hz of beating between patches (boom)"), seed=(0, "variation"))
def dune_sound(sr=DEFAULT_SR, mode="boom", grain=170.0, speed=0.2, hz=0.0, dur=6.0, beat=1.0, seed=0):
    if mode not in ("boom", "squeak", "silent"):
        raise ValueError("mode must be boom, squeak or silent")
    rng = Ma._rng(seed, 766)
    n = Ma._n(dur)
    t = np.arange(n) / SR
    u = t / dur
    gr = float(np.clip(grain, 60.0, 1000.0))
    hiss = Ma._bp(rng.standard_normal(n), 3000.0, 0.5) * (0.5 + np.abs(Ma._lfn(rng, n, 20.0)))
    if mode == "silent":
        env = np.minimum(u / 0.1, 1.0) * np.clip((1.0 - u) / 0.3, 0.0, 1.0)
        return Ma._fin(env * hiss, sr, 0.5)
    f = float(hz) if hz > 0 else (boom_hz(gr) if mode == "boom" else squeak_hz(float(np.clip(speed, 0.02, 2.0)), gr))
    f = float(np.clip(f, 20.0, 4000.0))
    fm = np.cumsum(f * (1.0 + 0.004 * Ma._lfn(rng, n, 0.5))) / SR
    if mode == "boom":
        env = np.minimum(u / 0.25, 1.0) ** 2 * np.clip((1.0 - u) / 0.3, 0.0, 1.0)
        body = np.sin(2 * np.pi * fm) + np.sin(2 * np.pi * (fm + float(beat) * t) + 1.0)
        body = body + sum(g * np.sin(2 * np.pi * h * fm) for h, g in G.DUNE_HARMONICS)
        return Ma._fin(env * (body + 0.02 * hiss), sr, 0.7)
    env = np.minimum(u / 0.05, 1.0) * np.clip((1.0 - u) / 0.2, 0.0, 1.0) * (0.6 + 0.4 * np.abs(Ma._lfn(rng, n, 12.0)))
    saw = 2.0 * (fm % 1.0) - 1.0
    return Ma._fin(Ma._lp(env * saw + 0.1 * env * hiss, 6000.0, 2), sr, 0.6)


GRAIN_KINDS = {"sand": ("sand", 0.0), "gravel": ("rock", 0.0), "mud": ("sand", 1.0)}


def pour_signal(n: int, flow: float, grain_mm: float, height: float, kind: str, wet: float, rng) -> np.ndarray:
    """Unscaled impact noise of a pour of ``flow`` kg/s of grains of ``grain_mm`` mm falling ``height`` m (impact speed sqrt(2 g h)):
    Poisson impacts at rate flow / grain mass, Hertz contact time of the grain, level going as sqrt(rate) x grain mass."""
    mat, _ = GRAIN_KINDS[kind]
    d = grain_mm * 1e-3
    m = float(sphere_mass(d, mat))
    v = math.sqrt(2.0 * GRAV * max(height, 0.01))
    rate = grain_rate(flow, grain_mm, mat)
    return shot_noise(n, rate, m * v, hertz_contact_time(m, v, d, mat), rng, jitter=0.2)


@sfx("granular_pour", "Sand, gravel or mud poured, from mass flow, grain size and drop height: Poisson grain impacts (rate = flow / grain mass) with Hertz contacts; wet adds bubble pops and darkens.",
     kind=("sand", "sand|gravel|mud"), flow=(0.2, "kg/s"), grain=(0.2, "mm grain diameter (sand 0.06-2, gravel 2-60)"), height=(0.5, "m of the fall"),
     wet=(0.0, "0..1 water: bubble pops, darker"), dur=(3.0, "s"), seed=(0, "variation"))
def granular_pour(sr=DEFAULT_SR, kind="sand", flow=0.2, grain=0.2, height=0.5, wet=0.0, dur=3.0, seed=0):
    if kind not in GRAIN_KINDS:
        raise ValueError("kind must be one of: " + ", ".join(GRAIN_KINDS))
    rng = Ma._rng(seed, 767)
    n = Ma._n(dur)
    w = float(np.clip(wet, 0.0, 1.0)) if kind != "mud" else 1.0
    y = pour_signal(n, float(np.clip(flow, 1e-4, 50.0)), float(np.clip(grain, 0.02, 60.0)), float(height), kind, w, rng)
    y = Ma._hp(Ma._lp(y, 9000.0 * (1.0 - 0.8 * w), 2), 150.0)
    out = _un(y)
    if w > 0.0:
        pops = np.zeros(n)
        for t in rng.uniform(0.0, dur, int(40 * w * dur) + 1):
            r = rng.uniform(0.004, 0.02)
            G._ring(pops, t, 3.26 / r, 0.015, rng.uniform(0.3, 1.0))
        out = out + 0.5 * w * _un(pops)
    return Ma._fin(out, sr, 0.7)
