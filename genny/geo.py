"""Ground, rock, sand, mud, volcanoes, ice shelves, falling trees and snow: reports/Inventaire genny 2026-10-29.md, sections 8 and 9.1.

The source of every law and constant is in research/SOURCES.md, section geo. Nothing here has been compared with a recording.
"""
from __future__ import annotations

import math

import numpy as np

from . import contact as K
from . import matter as Ma
from .core import DEFAULT_SR
from .physical.core import soft_ends
from .sfx import sfx

SR = Ma.SR
G = 9.81


def _ring(y, t, f, tau, amp):
    """Add a damped sine at t seconds, opening over a quarter period (a contact takes time) and closing over its last tau, so that it does not end on a step at 6 tau."""
    s = int(t * SR)
    if 0 <= s < y.size and f < 0.45 * SR:
        m = min(y.size - s, max(8, int(6.0 * tau * SR)))
        k = np.arange(m) / SR
        y[s:s + m] += amp * np.exp(-k / tau) * np.sin(2.0 * np.pi * f * k) * soft_ends(m, int(0.25 * SR / f), int(tau * SR))


def _grain(y, t, f, amp, rng, n=3):
    """A hit on rough ground: a few inharmonic modes that die in a few periods."""
    for r in [1.0] + list(rng.uniform(1.4, 3.2, n - 1)):
        _ring(y, t, f * r, 3.0 / (f * r), amp / r)


def boulder(mass, v, on="ground"):
    """(energy-like level, centre Hz) of one block hit. Elastic energy goes as m^(5/3) Vz^(11/5) on a plate, m Vz^(13/5) on a
    block; mean frequency as 1/contact time, and Hertz contact as R V^(-1/5); 100 Hz for a 1 m/s, 1 kg rock on ground."""
    e = mass ** (5.0 / 3.0) * v ** (11.0 / 5.0) if on == "plate" else mass * v ** (13.0 / 5.0)
    return math.sqrt(e), 100.0 * mass ** (-1.0 / 3.0) * v ** 0.2


def _cascade(y, t0, t1, blocks, size, height, fragment, rng):
    """Blocks that start between t0 and t1, each bouncing and shrinking. Fragmenting hits are higher: centroid 56.6 against 48.5 Hz (JGR-ES 2018)."""
    v0 = math.sqrt(2.0 * G * max(height, 0.1))
    for _ in range(int(blocks)):
        t = rng.uniform(t0, t1)
        m = 2600.0 * (float(size) * rng.pareto(2.0) + 0.05 * size) ** 3
        m = min(m, 2600.0 * (3.0 * size) ** 3)
        v = v0 * rng.uniform(0.5, 1.0)
        while t < y.size / SR and v > 0.6:
            lev, f = boulder(m, v)
            if rng.random() < fragment:
                f *= 56.6 / 48.5
                m *= rng.uniform(0.3, 0.8)
            _grain(y, t, f, lev, rng)
            t += max(2.0 * v * 0.6 / G * rng.uniform(0.3, 1.0), 0.01)
            v *= rng.uniform(0.45, 0.75)


@sfx("rockfall", "Rocks falling: blocks with a power law of sizes bounce down, breaking up as they go.",
     dur=(5.0, "s"), blocks=(30, "number of blocks"), size=(0.3, "m typical block"), height=(15.0, "m of fall: speed of the hits"),
     fragment=(0.3, "0..1 chance a hit breaks the block"), seed=(0, "variation"))
def rockfall(sr=DEFAULT_SR, dur=5.0, blocks=30, size=0.3, height=15.0, fragment=0.3, seed=0):
    rng = Ma._rng(seed, 701)
    y = np.zeros(Ma._n(dur))
    _cascade(y, 0.0, 0.6 * dur, np.clip(blocks, 1, 400), np.clip(size, 0.02, 3.0), height, np.clip(fragment, 0.0, 1.0), rng)
    return Ma._fin(y, sr, 0.8)


@sfx("landslide", "Landslide: a block cascade over a roaring slide of gravel that grows and dies.",
     dur=(10.0, "s"), blocks=(60, "number of blocks"), size=(0.4, "m"), seed=(0, "variation"))
def landslide(sr=DEFAULT_SR, dur=10.0, blocks=60, size=0.4, seed=0):
    rng = Ma._rng(seed, 702)
    n = Ma._n(dur)
    y = np.zeros(n)
    _cascade(y, 0.05 * dur, 0.8 * dur, np.clip(blocks, 1, 800), np.clip(size, 0.02, 3.0), 20.0, 0.4, rng)
    u = np.arange(n) / n
    env = np.minimum(u / 0.2, 1.0) * np.clip((1.0 - u) / 0.5, 0.0, 1.0) ** 1.5
    roar = Ma._lp(rng.standard_normal(n), 900.0, 2) + 2.0 * Ma._lp(rng.standard_normal(n), 90.0, 2)
    roar *= env * (0.7 + 0.3 * np.abs(Ma._lfn(rng, n, 6.0)))
    return Ma._fin(y / (np.max(np.abs(y)) + 1e-12) + 0.5 * roar / (np.max(np.abs(roar)) + 1e-12), sr, 0.8)


@sfx("debris_flow", "Debris flow or lahar: successive muddy surges, a 20-50 Hz roar with boulders knocking inside and water over them.",
     dur=(20.0, "s"), surges=(4, "waves; on a real volcano 4-5, about 10 minutes apart (Bosa et al. 2021): here squeezed into dur"),
     seed=(0, "variation"))
def debris_flow(sr=DEFAULT_SR, dur=20.0, surges=4, seed=0):
    rng = Ma._rng(seed, 703)
    n = Ma._n(dur)
    t = np.arange(n) / SR
    env = np.zeros(n)
    k = int(np.clip(surges, 1, 8))
    for i in range(k):
        t0 = dur * (i + rng.uniform(0.0, 0.4)) / k
        env += (0.6 ** i) * np.where(t > t0, (1.0 - np.exp(-(t - t0) / 0.5)) * np.exp(-(t - t0) / (0.35 * dur / k)), 0.0)
    low = Ma._bp(rng.standard_normal(n), 35.0, 0.8)             # pic at 20-50 Hz (flume, E3S 2023)
    water = Ma._hp(Ma._lp(rng.standard_normal(n), 3500.0), 500.0)
    y = np.zeros(n)
    for s in rng.uniform(0.0, dur, int(25 * dur)):
        lev = env[min(int(s * SR), n - 1)]
        if rng.random() < lev:
            _grain(y, s, rng.uniform(60.0, 240.0), rng.uniform(0.3, 1.0), rng)
    un = lambda v: v / (np.max(np.abs(v)) + 1e-12)
    return Ma._fin(env * un(low) + 0.25 * env * un(water) + 0.5 * un(y), sr, 0.8)


@sfx("singing_dune", "Singing or booming dune: an avalanche of dry sand gives one sustained, almost pure note near 100 Hz that beats.",
     freq=(100.0, "Hz; set by the grain size (Douady et al. 2006, Vriend et al. 2007: whichever explains it, about 100)"),
     dur=(6.0, "s"), beat=(1.5, "Hz between the patches of grains that do not quite lock"), seed=(0, "variation"))
def singing_dune(sr=DEFAULT_SR, freq=100.0, dur=6.0, beat=1.5, seed=0):
    rng = Ma._rng(seed, 704)
    n = Ma._n(dur)
    t = np.arange(n) / SR
    f = float(np.clip(freq, 40.0, 500.0))
    fm = np.cumsum(f * (1.0 + 0.01 * Ma._lfn(rng, n, 0.5))) / SR
    body = np.sin(2 * np.pi * fm) + np.sin(2 * np.pi * (fm + float(beat) * t) + 1.0)
    body = np.tanh(1.5 * body) + 0.15 * np.sin(4 * np.pi * fm)
    u = t / dur
    env = np.minimum(u / 0.25, 1.0) ** 2 * np.clip((1.0 - u) / 0.3, 0.0, 1.0)
    hiss = Ma._hp(rng.standard_normal(n), 1500.0) * (0.5 + np.abs(Ma._lfn(rng, n, 25.0)))
    return Ma._fin(env * (body + 0.015 * hiss), sr, 0.7)


@sfx("mud_pot", "Boiling mud: hemispheric bubbles swell and burst, a jet of gas, the mud heaving back (Jolly 2016, Vidal 2010).",
     dur=(6.0, "s"), rate=(1.2, "bursts a second"), thick=(0.5, "0..1 viscosity: long film rupture = softer pop, deeper"), seed=(0, "variation"))
def mud_pot(sr=DEFAULT_SR, dur=6.0, rate=1.2, thick=0.5, seed=0):
    rng = Ma._rng(seed, 705)
    n = Ma._n(dur)
    y = np.zeros(n)
    th = float(np.clip(thick, 0.0, 1.0))
    for _ in range(rng.poisson(max(rate, 0.05) * dur) + 1):
        t = rng.uniform(0.0, dur - 0.4)
        r = rng.uniform(0.015, 0.06)                            # m
        f = rng.uniform(0.7, 1.1) * 3.26 / r * (1.0 - 0.5 * th)
        a = r / 0.03
        _ring(y, t, f, (0.015 + 0.03 * th), a)                  # swell, shut by the film breaking
        _ring(y, t + 0.012 + 0.03 * th, 2.5 * f, 0.004, 0.7 * a * (1.0 - 0.6 * th))    # amplitude follows the film rupture time
        s = int((t + 0.02) * SR)                                # jet of gas
        m = min(n - s, int(0.05 * SR))
        if m > 8:
            y[s:s + m] += 0.1 * a * np.exp(-np.arange(m) / (0.012 * SR)) * rng.standard_normal(m)
        _ring(y, t + 0.08 + 0.05 * th, rng.uniform(30.0, 60.0), 0.05, 0.6 * a)         # the heave
    return Ma._fin(Ma._lp(y, 3500.0, 2), sr, 0.8)


@sfx("geyser", "Geyser: a tremor of collapsing steam bubbles about 5 cm across, then the jet and its water raining back (Kedar et al. 1998).",
     dur=(14.0, "s"), rate=(40.0, "bubbles a second in the tremor"), erupt=(0.4, "0..1 when in dur it blows; 1 = tremor only"), seed=(0, "variation"))
def geyser(sr=DEFAULT_SR, dur=14.0, rate=40.0, erupt=0.4, seed=0):
    rng = Ma._rng(seed, 706)
    n = Ma._n(dur)
    t = np.arange(n) / SR
    y = np.zeros(n)
    te = float(np.clip(erupt, 0.05, 1.0)) * dur
    for s in rng.uniform(0.0, min(te * 1.1, dur), rng.poisson(max(rate, 1.0) * te)):
        r = 0.05 * rng.uniform(0.6, 1.3)
        # no column resonance (their result); a collapse is a fast damped Minnaert ring: sharp, two periods
        _ring(y, s, 3.26 / r, 2.0 * r / 3.26, (s / te) ** 2 * rng.uniform(0.2, 1.0))
    jet = np.where(t > te, (1.0 - np.exp(-(t - te) / 0.2)) * np.exp(-(t - te) / (0.25 * dur)), 0.0)
    roar = Ma._hp(Ma._lp(rng.standard_normal(n), 6000.0), 150.0)
    fall = Ma._hp(rng.standard_normal(n), 1800.0) * np.where(t > te + 1.0, (1.0 - np.exp(-(t - te - 1.0) / 1.0)) * np.exp(-(t - te) / (0.4 * dur)), 0.0)
    un = lambda v: v / (np.max(np.abs(v)) + 1e-12)
    return Ma._fin(un(y) + 0.7 * jet * un(roar) + 0.25 * un(fall), sr, 0.8)


@sfx("lava", "Strombolian bursts: big gas slugs breaking a lava film in a conduit, a chest-felt thump, the crack of the film, spatter falling back.",
     dur=(10.0, "s"), bursts=(3, "explosions"), conduit=(25.0, "m depth of the vent, 15..35 (Ripepe, Delle Donne & Ripepe): the thump's pitch"),
     seed=(0, "variation"))
def lava(sr=DEFAULT_SR, dur=10.0, bursts=3, conduit=25.0, seed=0):
    rng = Ma._rng(seed, 707)
    n = Ma._n(dur)
    y = np.zeros(n)
    t = np.arange(n) / SR
    bed = Ma._lp(rng.standard_normal(n), 60.0) * (0.5 + 0.5 * np.abs(Ma._lfn(rng, n, 0.8)))
    for i in range(int(np.clip(bursts, 1, 30))):
        t0 = rng.uniform(0.02, 0.75) * dur
        a = rng.uniform(0.6, 1.0)
        # monopole with a little dipole, quarter-wave conduit; the hearing threshold makes 3.5 Hz a pressure more than a note: its 7th is what is heard
        f0 = 343.0 / (4.0 * float(np.clip(conduit, 5.0, 60.0)))
        for k, g in ((3, 1.0), (5, 0.7), (7, 0.5), (11, 0.25)):
            _ring(y, t0, f0 * k * rng.uniform(0.95, 1.05), 0.25, a * g)
        s = int(t0 * SR)                                        # the film breaking is the audible part
        m = min(n - s, int(0.02 * SR))
        if m > 8:
            y[s:s + m] += 1.2 * a * Ma._hp(rng.standard_normal(m), 400.0) * np.exp(-np.arange(m) / (0.003 * SR))
        for _ in range(rng.poisson(40)):                        # spatter
            _grain(y, t0 + 0.6 + rng.exponential(1.2), rng.uniform(150.0, 900.0), 0.04, rng, 2)
    return Ma._fin(y + 0.15 * bed / (np.max(np.abs(bed)) + 1e-12), sr, 0.8)


@sfx("glacier_calving", "A glacier calving: deep crack and rumble of the iceberg, the huge splash, then the fizz of air bubbles leaving melting ice (Glowacki & Deane).",
     dur=(14.0, "s"), size=(1.0, "0.2..3 mass of the block: lower, longer"), seed=(0, "variation"))
def glacier_calving(sr=DEFAULT_SR, dur=14.0, size=1.0, seed=0):
    rng = Ma._rng(seed, 708)
    n = Ma._n(dur)
    t = np.arange(n) / SR
    s = float(np.clip(size, 0.2, 3.0))
    y = np.zeros(n)
    for c in range(12):
        _ring(y, rng.uniform(0.0, 0.8), rng.uniform(120.0, 700.0) / s, 0.02, rng.uniform(0.3, 1.0))
    rum = np.where(t > 0.4, (1.0 - np.exp(-(t - 0.4) / 0.7)) * np.exp(-(t - 0.4) / (0.2 * dur)), 0.0)
    low = Ma._lp(rng.standard_normal(n), 70.0 / s, 2)
    ts = 0.25 * dur
    sp = np.where(t > ts, (1.0 - np.exp(-(t - ts) / 0.05)) * np.exp(-(t - ts) / (0.1 * dur)), 0.0)
    splash = Ma._hp(Ma._lp(rng.standard_normal(n), 7000.0), 200.0)
    fiz = np.zeros(n)
    w = np.where(t > ts, np.exp(-(t - ts) / (0.5 * dur)), 0.0)
    for b in rng.uniform(ts, dur, int(180 * dur)):
        r = rng.uniform(0.0003, 0.002)
        _ring(fiz, b, 3.26 / r, 8.0 / (3.26 / r), rng.uniform(0.2, 1.0) * w[min(int(b * SR), n - 1)])
    un = lambda v: v / (np.max(np.abs(v)) + 1e-12)
    return Ma._fin(0.5 * un(y) + rum * un(low) + 0.7 * sp * un(splash) + 0.12 * un(fiz), sr, 0.8)


@sfx("sand_storm", "Blown sand: wind, and grain impacts that only start above the threshold and then rise as the cube of its excess (Bagnold; Saltiphone counts).",
     dur=(8.0, "s"), wind=(0.7, "0..1 mean strength; sand moves above 0.35"), hard=(0.5, "0 soft ground .. 1 hard (a helmet, a car)"), seed=(0, "variation"))
def sand_storm(sr=DEFAULT_SR, dur=8.0, wind=0.7, hard=0.5, seed=0):
    rng = Ma._rng(seed, 709)
    n = Ma._n(dur)
    u = np.clip(float(wind) * (1.0 + 0.45 * Ma._lfn(rng, n, 0.4)), 0.0, 1.5)
    ex = np.maximum(u - 0.35, 0.0) ** 3
    pr = np.minimum(ex * 2.5e4 / SR, 0.9)
    imp = (rng.random(n) < pr) * rng.exponential(1.0, n)
    f = 6000.0 + 5000.0 * float(np.clip(hard, 0.0, 1.0))
    grains = Ma._bp(imp, f, 1.2)
    air = Ma._bp(rng.standard_normal(n), 500.0, 0.5) * (u ** 2)
    un = lambda v: v / (np.max(np.abs(v)) + 1e-12)
    return Ma._fin(0.8 * un(air) + 0.6 * un(grains), sr, 0.7)


S_MINUS_P = 1.0 / 3.5 - 1.0 / 6.0        # s per km between the P wave (6 km/s) and the S wave (3.5 km/s)


def s_delay(distance: float, dur: float) -> float:
    """Seconds from the P to the S wave at ``distance`` km, at most half the sound."""
    return min(float(distance) * S_MINUS_P, 0.5 * float(dur))


@sfx("earthquake", "Earthquake heard: the ground reverberating at 40-70 Hz as body waves arrive, sharp detonations over the P wave, things rattling at the S wave.",
     dur=(14.0, "s"), distance=(20.0, "km: the S wave comes distance x 0.119 s after P, the first thing heard (at most half the sound)"),
     rattle=(0.5, "0..1 objects shaking"), seed=(0, "variation"))
def earthquake(sr=DEFAULT_SR, dur=14.0, distance=20.0, rattle=0.5, seed=0):
    rng = Ma._rng(seed, 710)
    n = Ma._n(dur)
    t = np.arange(n) / SR
    ds = s_delay(distance, dur)
    p = np.where(t > 0.1, (1.0 - np.exp(-(t - 0.1) / 0.3)) * np.exp(-(t - 0.1) / (dur * 0.15)), 0.0)
    s = np.where(t > 0.1 + ds, (1.0 - np.exp(-(t - 0.1 - ds) / 0.5)) * np.exp(-(t - 0.1 - ds) / (dur * 0.3)), 0.0)
    y = Ma._bp(rng.standard_normal(n), 55.0, 1.5)
    y /= np.max(np.abs(y)) + 1e-12
    low = Ma._lp(rng.standard_normal(n), 30.0, 2)
    low /= np.max(np.abs(low)) + 1e-12
    bangs = np.zeros(n)
    for b in rng.uniform(0.1, 0.1 + max(ds, 0.5), 5):
        _ring(bangs, b, rng.uniform(40.0, 90.0), 0.04, rng.uniform(0.5, 1.0))
    rat = np.zeros(n)
    k = float(np.clip(rattle, 0.0, 1.0))
    for b in rng.uniform(0.1 + ds, dur, int(60 * dur * k)):
        _grain(rat, b, rng.uniform(600.0, 3000.0), s[min(int(b * SR), n - 1)] * rng.uniform(0.1, 1.0), rng, 2)
    un = lambda v: v / (np.max(np.abs(v)) + 1e-12)
    return Ma._fin((0.3 * p + s) * y + 0.8 * s * low + 0.5 * un(bangs) + 0.3 * un(rat), sr, 0.8)


@sfx("tree_fall", "A tree going down: cracks that come faster and faster, the break, the crown falling through branches, the thud.",
     dur=(7.0, "s"), size=(1.0, "0.3..2 trunk: deeper, slower"), seed=(0, "variation"))
def tree_fall(sr=DEFAULT_SR, dur=7.0, size=1.0, seed=0):
    """Emissions climb before failure (three-point bending, PMC 2024); about 1/6 of the fracture energy goes to sound (Landis & Whittaker)."""
    rng = Ma._rng(seed, 711)
    n = Ma._n(dur)
    y = np.zeros(n)
    s = float(np.clip(size, 0.3, 2.0))
    tb = 0.35 * dur
    t = 0.0
    while t < tb:                                               # gaps shrink towards the break
        x = 1.0 - t / tb
        _grain(y, t, rng.uniform(350.0, 1100.0) / s, (1.0 - x) ** 1.5 * rng.uniform(0.3, 1.0), rng)
        t += 0.25 * x ** 1.5 * rng.uniform(0.3, 1.5) + 0.006
    for _ in range(14):                                         # the break
        _grain(y, tb + rng.uniform(0.0, 0.05), rng.uniform(150.0, 1500.0) / s, rng.uniform(0.7, 1.2), rng)
    tf = tb + 0.45 * dur
    u = (np.arange(n) / SR - tb) / max(tf - tb, 0.1)
    swish = np.where((u > 0.0) & (u < 1.05), np.clip(u, 0.0, 1.0) ** 2, 0.0)
    leaves = Ma._hp(Ma._lp(rng.standard_normal(n), 7000.0), 900.0) * swish
    br = np.zeros(n)
    for b in rng.uniform(tb + 0.5 * (tf - tb), tf + 0.3, 60):
        _grain(br, b, rng.uniform(600.0, 2500.0), rng.uniform(0.1, 0.6), rng, 2)
    _ring(y, tf, 42.0 / s, 0.2, 3.5)
    _ring(y, tf + 0.05, 90.0 / s, 0.12, 1.5)
    un = lambda v: v / (np.max(np.abs(v)) + 1e-12)
    return Ma._fin(un(y) + 0.35 * un(leaves) + 0.35 * un(br), sr, 0.8)


@sfx("snowfall", "Snow settling: the near-silent hiss of flakes and, now and then, the soft whump of snow dropping from a branch or a roof.",
     dur=(10.0, "s"), rate=(0.3, "whumps a second"), hiss=(0.15, "0..1 flakes against clothes or a hood"), seed=(0, "variation"))
def snowfall(sr=DEFAULT_SR, dur=10.0, rate=0.3, hiss=0.15, seed=0):
    """On water each flake traps a bubble, but at 50-200 kHz (Crum et al. 1999): nothing to hear, so it is not modelled."""
    rng = Ma._rng(seed, 712)
    n = Ma._n(dur)
    y = np.zeros(n)
    for b in rng.uniform(0.0, dur - 0.3, rng.poisson(max(rate, 0.0) * dur)):
        s = int(b * SR)
        m = min(n - s, int(0.25 * SR))
        y[s:s + m] += rng.uniform(0.4, 1.0) * Ma._lp(rng.standard_normal(m), 450.0, 2) * np.hanning(2 * m)[m:] ** 2 * 3.0
        _ring(y, b, rng.uniform(50.0, 90.0), 0.05, 0.3)
    f = Ma._hp(rng.standard_normal(n), 3000.0) * (0.6 + 0.4 * np.abs(Ma._lfn(rng, n, 2.0))) * 0.03 * float(np.clip(hiss, 0.0, 1.0))
    return Ma._fin(y + f, sr, 0.5)


@sfx("snow_avalanche", "Powder avalanche: a moving turbulent cloud, blocks of snow thudding inside the flow and the low rumble under it (Kogelnig et al. 2011).",
     dur=(12.0, "s"), powder=(0.6, "0 dense flow .. 1 powder cloud"), blocks=(70, "snow blocks"),
     precursor=(0.0, "s of snowpack cracking before the release: faint crackle at a rising rate, a quiet spell, then the slide"),
     density=(200.0, "kg/m3 of the snowpack, 80..500: light snow damps the highs of the cracking"), seed=(0, "variation"))
def snow_avalanche(sr=DEFAULT_SR, dur=12.0, powder=0.6, blocks=70, precursor=0.0, density=200.0, seed=0):
    y = _snow_avalanche(sr, dur, powder, blocks, seed)
    pre = float(np.clip(precursor, 0.0, 30.0))
    if pre <= 0:
        return y
    pc_ = snow_precursor(pre, float(np.clip(density, 80.0, 500.0)), Ma._rng(seed, 714))
    pc_ = Ma._fin(pc_, sr, PRECURSOR_LEVEL * float(np.max(np.abs(y))), 0.002, 0.002)
    return np.concatenate([pc_, y])


PRECURSOR_QUIET = 0.15          # share of the precursor that is the quiet spell before release
PRECURSOR_LEVEL = 0.05          # peak of the precursor cracking against the avalanche's peak
PRECURSOR_RATE = (2.0, 60.0)    # cracks per second at the start and at the end of the emission


def snow_precursor(dur: float, density: float, rng) -> np.ndarray:
    """Acoustic emission of a loaded snowpack before it fails, at 48 kHz: cracks at a rate growing
    exponentially (PRECURSOR_RATE), then silence for PRECURSOR_QUIET of the time. Each crack is a short
    noise burst low-passed at 10 density Hz (air-dominated light snow absorbs the highs, ice-bonded dense snow less)."""
    n = Ma._n(dur)
    t = np.arange(n) / SR
    act = 1.0 - PRECURSOR_QUIET
    r0, r1 = PRECURSOR_RATE
    rate = np.where(t < act * dur, r0 * (r1 / r0) ** (t / (act * dur)), 0.0)
    y = np.zeros(n)
    for s in Ma.bubble_times(n, rng, rate):
        k = int(rng.uniform(0.001, 0.004) * SR)
        Ma._place(y, rng.standard_normal(k) * np.exp(-np.arange(k) / (0.3 * k)), s, rng.lognormal(0.0, 0.6))
    return Ma._lp(y, 10.0 * density, 2)


def _snow_avalanche(sr, dur, powder, blocks, seed):
    rng = Ma._rng(seed, 713)
    n = Ma._n(dur)
    t = np.arange(n) / SR
    u = t / dur
    env = np.minimum(u / 0.15, 1.0) * np.clip((1.0 - u) / 0.6, 0.0, 1.0) ** 1.2
    pw = float(np.clip(powder, 0.0, 1.0))
    cloud = Ma._tvbp(rng.standard_normal(n), 250.0 + 900.0 * pw * (0.6 + 0.4 * np.abs(Ma._lfn(rng, n, 0.5))), 0.5)
    ru = K.avalanche(SR, dur=dur, seed=int(seed))
    ru = np.resize(np.asarray(ru, float), n) if len(ru) != n else np.asarray(ru, float)
    y = np.zeros(n)
    for b in rng.uniform(0.0, 0.8 * dur, int(np.clip(blocks, 0, 800))):
        m = 400.0 * rng.pareto(2.5) * 0.3 + 1.0                 # kg; snow is soft: dull, short, no ring
        lev, f = boulder(m, rng.uniform(4.0, 25.0))
        _ring(y, b, min(f, 180.0), 0.03, lev * env[min(int(b * SR), n - 1)])
    un = lambda v: v / (np.max(np.abs(v)) + 1e-12)
    return Ma._fin((0.3 + 0.7 * pw) * env * un(cloud) + 0.6 * un(y) + 0.8 * un(ru), sr, 0.8)


def _steps_env(n, rate):
    """Force of one foot after another, 40 ms rise, 250 ms fall; the patches take it from a control."""
    e = np.zeros(n)
    m = int(0.29 * SR)
    one = np.interp(np.arange(m) / SR, [0.0, 0.04, 0.29], [0.0, 1.0, 0.0])
    for s in range(0, n, max(int(SR / max(rate, 0.2)), 1)):
        k = min(m, n - s)
        e[s:s + k] = np.maximum(e[s:s + k], one[:k])
    return e


def _ratio(a, b):
    return a / np.where(np.abs(b) < 1e-9, 1e-9, b)


@sfx("pd_footstep", "Footsteps from formulas: each surface is one short expression of a force curve and filtered noise (Footsteppr, a Pd patch).",
     surface=("gravel", "gravel|snow|wood|dirt|grass"), steps=(4, "steps"), rate=(1.8, "steps a second"), seed=(0, "variation"))
def pd_footstep(sr=DEFAULT_SR, surface="gravel", steps=4, rate=1.8, seed=0):
    """Connections as traced by the deep reading from Bfxr2's Footsteppr; E is the force curve, lopF and hipF one-pole filters, vcf a band-pass at fc with Q."""
    rng = Ma._rng(seed, 721)
    r = max(float(rate), 0.2)
    n = Ma._n(int(np.clip(steps, 1, 64)) / r + 0.3)
    E = _steps_env(n, r)
    N1, N2 = rng.standard_normal(n), rng.standard_normal(n)
    lo, hi = Ma._lp, Ma._hp
    if surface == "gravel":
        g = np.clip(0.01 * hi(_ratio(lo(N1, 300.0), lo(N1, 2000.0)), 400.0) ** 2, -0.9, 0.9)
        y = 2.0 * E * hi(Ma._tvbp(g, np.clip(50000.0 * lo(N2, 50.0) + 1000.0 * E, 500.0, 10000.0), 3.0), 200.0)
    elif surface == "snow":
        g = np.clip(_ratio(lo(N1, 110.0), lo(N1, 900.0)) * _ratio(lo(N2, 50.0), lo(N2, 70.0)) * (0.5 + (17.0 * lo(N2, 10.0)) ** 2), -1.0, 1.0)
        y = 0.2 * E * Ma._tvbp(hi(g, 300.0), 700.0 + 9000.0 * E, 0.5)
    elif surface == "wood":
        a = sum(Ma._bp(N1, f, 90.0) for f in (95.0, 134.0, 139.0, 154.0))
        b = Ma._bp(N2, 123.0, 20.0) + Ma._bp(N2, 156.0, 90.0) + Ma._bp(N2, 189.0, 90.0) + Ma._bp(N2, 201.0, 70.0)
        y = 0.5 * np.sqrt(E) * 6.0 * a + 0.6 * 2.0 * E * E * 8.0 * b
    elif surface == "dirt":
        E4 = E ** 4
        c = np.clip(hi(np.cos(2.0 * np.pi * np.cumsum(70.0 + 70.0 * (E + 0.3) * 70.0 * lo(N1, 80.0)) / SR), 200.0), -1.0, 1.0)
        y = 0.5 * E4 * np.cos(2.0 * np.pi * np.cumsum(40.0 + 500.0 * E4) / SR) + 0.04 * c
    elif surface == "grass":
        E4 = E ** 4
        g = np.clip(1e-5 * hi(_ratio(lo(N1, 300.0), lo(N1, 2000.0)), 2500.0) ** 4, -0.9, 0.9)
        v = Ma._tvbp(g, np.clip(3400.0 + 23800.0 * lo(N2, 16.0), 2000.0, 10000.0), 1.0)
        y = 0.8 * E4 * np.clip(np.cos(2.0 * np.pi * np.cumsum(30.0 + 600.0 * E4) / SR), 0.0, 0.5) + 0.3 * E * hi(v, 900.0)
    else:
        raise ValueError(f"unknown pd_footstep surface {surface!r}")
    return Ma._fin(y, sr, 0.7)


@sfx("snow_squeal", "Very cold dry snow roaring under a load: grains locking into a squeal near 200 Hz, heard kilometres off at -45 C (Patitsas 2013).",
     speed=(0.2, "m/s of slip"), grain=(1.0, "mm; Bagnold's f = speed / grain: 0.2 / 1 = 200 Hz"), dur=(1.5, "s"), seed=(0, "variation"))
def snow_squeal(sr=DEFAULT_SR, speed=0.2, grain=1.0, dur=1.5, seed=0):
    rng = Ma._rng(seed, 722)
    n = Ma._n(dur)
    f = float(np.clip(float(speed) / (max(float(grain), 0.05) * 1e-3), 30.0, 2000.0))
    ph = np.cumsum(f * (1.0 + 0.015 * Ma._lfn(rng, n, 8.0))) / SR
    u = np.arange(n) / n
    env = np.minimum(u / 0.1, 1.0) * np.clip((1.0 - u) / 0.2, 0.0, 1.0) * (0.6 + 0.4 * np.abs(Ma._lfn(rng, n, 15.0)))
    y = (2.0 * (ph % 1.0) - 1.0) * env + 0.1 * env * Ma._hp(rng.standard_normal(n), 1000.0)
    return Ma._fin(Ma._lp(y, 4000.0, 2), sr, 0.7)
