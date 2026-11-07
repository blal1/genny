"""Trucks, buses, two-wheelers, tracked vehicles, site and garden machines. Presets on ``engines`` (the waveguides of
``cyber.pipe_engine``), and two new models: ``tracks`` and ``hydraulic_pump``.

Sources of every constant: research/SOURCES.md, section heavy_vehicles. No airborne measurement of a tracked vehicle was obtained:
the structure of ``tracks`` is sourced, its levels are not. Nothing here has been listened to: tests/test_heavy_vehicles.py says what was measured.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import fftconvolve, resample_poly

from .contact import drop, impact, modal_engine, roll, rumble, scrape
from .core import DEFAULT_SR
from .creatures import footsteps
from .cyber import _HN_BR, _impacts, _modes, _third_octave_noise, gearbox
from .engines import _even, _events, _pulses, _run, _smooth, diesel_engine, two_stroke
from .foley import creak
from .matter import _ar, _fin, _fit, _hp, _lp, _n, _rng, _tvbp, _unit, air_leak
from .physical.core import SR
from .plates import metal_plate
from .reverbs import doppler
from .road import _grains, _kind, air_system, tyre_roll
from .sfx import fan, jet_engine, sfx


def _put(out, y, t, gain=1.0):
    i = int(t * SR)
    m = min(len(y), len(out) - i)
    if m > 0:
        out[i:i + m] += gain * y[:m]


def _rms(x):
    return float(np.sqrt(np.mean(x * x)))


# =============================================================================================
# HARMONOISE, PASS-BY
# =============================================================================================
_CV_AR = np.array([79.5, 79.5, 79.5, 79.5, 81.5, 82.5, 82.5, 85.5, 87.3, 87.7, 87.3, 90.4, 91.2, 95.0, 97.1, 96.8, 97.4, 95.2, 92.6,
                   91.9, 89.5, 86.1, 84.1, 82.2, 80.3, 80.3, 80.3, 80.3, 80.3])
_CV_AP = np.array([97.7, 97.7, 97.3, 98.2, 103.3, 109.5, 105.3, 100.8, 101.2, 99.9, 102.3, 103.5, 104.0, 101.6, 99.2, 99.4, 95.1, 95.8,
                   95.3, 93.8, 93.9, 92.7, 91.6, 90.9, 87.9, 87.9, 81.8, 80.2, 80.2])
_CV2_AP = np.array([97.0, 97.0, 97.7, 98.5, 98.5, 101.5, 101.4, 97.0, 96.5, 95.2, 99.6, 100.7, 101.0, 98.3, 94.2, 92.4, 92.1, 93.8,
                    94.3, 95.2, 94.9, 93.3, 91.2, 89.3, 87.3, 85.3, 84.3, 83.3, 83.3])
_CV_BP = np.r_[np.zeros(11), np.full(18, 8.5)]
# Category 4b (motorcycles), propulsion only, octave bands 63 Hz .. 8 kHz: L = aP + bP (v - 70) / 70.
MOTO_F = 63.0 * 2.0 ** np.arange(8)
MOTO_AP = np.array([99.9, 101.9, 96.7, 94.4, 95.2, 94.7, 92.1, 88.6])
MOTO_BP = np.array([3.2, 5.9, 11.9, 11.6, 11.5, 12.6, 11.1, 12.0])
CV_HEIGHT, LOW_HEIGHT, EAR = 0.75, 0.01, 1.5       # m: the two source heights; the listener's ear, doppler's default 1.8 = .3 + 1.5


def cv_noise_levels(kmh: float, axles: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """(rolling, propulsion) sound power of a commercial vehicle in dB per third octave of ``cyber._HN_F``.
    ``axles`` 2 is the medium-heavy category (its own propulsion); rolling grows 10 lg(axles)."""
    v = max(float(kmh), 1.0)
    ap = _CV2_AP if int(axles) <= 2 else _CV_AP
    return _CV_AR + 10.0 * np.log10(max(int(axles), 2) / 4.0) + _HN_BR * np.log10(v / 70.0), ap + _CV_BP * (v - 70.0) / 70.0


def moto_noise_levels(kmh: float) -> np.ndarray:
    """Propulsion sound power of a motorcycle in dB per octave of ``MOTO_F``."""
    return MOTO_AP + MOTO_BP * (max(float(kmh), 1.0) - 70.0) / 70.0


def moto_propulsion(y, kmh, passes=4):
    """``y`` with its octave bands of ``MOTO_F`` brought to the profile of ``moto_noise_levels(kmh)``, the total kept:
    a gain per octave, interpolated in log f between the centres and held beyond them."""
    target = moto_noise_levels(kmh)
    f = np.fft.rfftfreq(len(y), 1.0 / SR)
    lf = np.log2(np.maximum(f, 1.0))
    Y = np.fft.rfft(y)
    e0 = float(np.sum(np.abs(Y) ** 2))
    for _ in range(int(passes)):
        P = np.abs(Y) ** 2
        cur = np.array([10.0 * np.log10(np.sum(P[(f >= c / math.sqrt(2.0)) & (f < c * math.sqrt(2.0))]) + 1e-30) for c in MOTO_F])
        g = (target - target.mean()) - (cur - cur.mean())
        Y = Y * 10.0 ** (np.interp(lf, np.log2(MOTO_F), g) / 20.0)
    Y *= math.sqrt(e0 / (float(np.sum(np.abs(Y) ** 2)) + 1e-30))
    return np.fft.irfft(Y, len(y))


def cv_sources(kmh, engine, n, rng):
    """(rolling, propulsion), the engine scaled to the band energy of the table as in ``cyber.pass_by``: only their ratio is meaningful."""
    lr, lp = cv_noise_levels(kmh)
    return _unit(_third_octave_noise(n, rng, lr)), math.sqrt(float(np.sum(10 ** (lp / 10)) / np.sum(10 ** (lr / 10)))) * _unit(_fit(engine, n))


def _flyby(y, v, distance, direction, pan, height):
    # Asphalt ground effect, wheel source or high truck source; 2.25 m = CV_HEIGHT + EAR in a shared height.
    return doppler(y, SR, speed=v, closest=float(np.clip(distance, 1.0, 200.0)), direction=direction, pan=bool(pan), surface="asphalt", horn="truck" if height > 2.0 else "low", height=height)


def _finish_stereo(y, sr, peak=0.7):
    """``_fin`` would fold a pan to mono."""
    if int(sr) != SR:
        g = math.gcd(int(sr), SR)
        y = resample_poly(y, int(sr) // g, SR // g, axis=0)
    return y * (peak / (float(np.max(np.abs(y))) + 1e-12))


def _wind(v, n, seed):
    """As in ``pass_by``: low-passed at 250 + 30 v Hz, level (v / 40)^2.5, v in m/s."""
    from . import osc as O
    return 1.5 * (v / 40.0) ** 2.5 * _unit(_lp(O.pink(n, int(seed) + 78), 250.0 + 30.0 * v))


# =============================================================================================
# TWO WHEELS
# =============================================================================================
MOTO_GEARS = (3.34, 2.30, 1.71, 1.41, 1.18, 1.0)
MOTO_FINAL = 2.9
MOTO_WHEEL = 0.31                                   # m: a 150/80-16 rear tyre
CRUISER_IDLE, CRUISER_LIMIT = 900.0, 5600.0
SPORT_RPM = 12000.0                                 # 400 Hz on four cylinders
MOTO_KINDS = ("cruiser", "sport", "passby")
BODY_HZ = 50.0                                      # body of a motorcycle engine (`modal_engine`)


def moto_rpm(gear, kmh):
    return kmh / 3.6 / (2.0 * math.pi * MOTO_WHEEL) * 60.0 * MOTO_GEARS[int(np.clip(gear, 1, 6)) - 1] * MOTO_FINAL


def _moto_engine(kind, rpm, load, crossplane, dur, seed):
    if kind == "sport":
        layout = "inline_4_crossplane" if crossplane else "inline_4"
        return _run(layout, rpm, dur, load, seed, headers=None, header=1.0, unequal=0.0)     # short header, 1 ms
    return _run("vtwin_45", rpm, dur, load, seed)


@sfx("motorcycle", "Motorcycle: cruiser (45 degree V-twin, 315/405 potato-potato), sport (high-revving inline four, or cross-plane), passby (propulsion only, Doppler and ground).",
     kind=("cruiser", "cruiser | sport | passby"), rpm=(-1.0, "crankshaft; -1 = idle 900 (cruiser), 12000 (sport); replaced by speed and gear if speed > 0"),
     speed=(0.0, "km/h: 0 = standing; passby defaults to 80"), gear=(3, "1..6 of 3.34 .. 1.0, final 2.9"), load=(0.6, "0..1"), crossplane=(0, "sport: 1 = fires 90-180-270-180"),
     distance=(6.0, "passby: m"), direction=("lr", "passby: lr | rl"), pan=(1, "passby: 1 stereo, 0 mono"), dur=(4.0, "s"), seed=(0, "variation"))
def motorcycle(sr=DEFAULT_SR, kind="cruiser", rpm=-1.0, speed=0.0, gear=3, load=0.6, crossplane=0, distance=6.0, direction="lr", pan=1, dur=4.0, seed=0):
    """Passby: the engine's spectrum brought to the octave profile of ``moto_noise_levels`` at the speed (`moto_propulsion`)."""
    k = _kind(kind, MOTO_KINDS, "motorcycle")
    kmh = float(speed) if speed > 0 else (80.0 if k == "passby" else 0.0)
    n = _n(dur)
    top = SPORT_RPM if k == "sport" else CRUISER_LIMIT
    r = float(np.clip(moto_rpm(gear, kmh), 800.0, top)) if kmh > 0 else (float(rpm) if rpm > 0 else (SPORT_RPM if k == "sport" else CRUISER_IDLE))
    y = _unit(_moto_engine(k, r, load, crossplane, dur, seed))
    y = y + 0.3 * _unit(_fit(modal_engine(sr=SR, rpm=r, dur=dur, cylinders=4 if k == "sport" else 2, body_freq=BODY_HZ, seed=seed), n))
    if k == "passby":
        v = kmh / 3.6
        y = _ar(n, 0.3, 0.3) * (moto_propulsion(y, kmh) + _wind(v, n, seed))
        return _finish_stereo(_flyby(y, v, distance, direction, pan, 1.8), sr)
    if kmh > 0:
        y = y + 0.3 * _unit(_fit(tyre_roll(sr=SR, speed=kmh, dur=dur, seed=seed), n))
    return _fin(y, sr, 0.7, 0.003, 0.02)


SCOOTER = {"2t": 7000.0, "4t": 6000.0}              # 116.7 Hz, 50 Hz


def cvt_rpm(top, dur, n):
    """A CVT holds the engine nearly still while the speed climbs: 0.4 s from 30 % of the top, then 1 % more."""
    t = np.arange(n) / SR
    return np.interp(t, [0.0, 0.4, max(dur, 0.41)], [0.3 * top, top, 1.01 * top])


@sfx("scooter", "Scooter: a 50 cc single, two-stroke or four-stroke, through a CVT: the engine note holds still while the speed climbs, under a belt noise.",
     kind=("2t", "2t | 4t"), rpm=(-1.0, "where the CVT holds the crank; -1 = 7000 (2t), 6000 (4t)"), load=(0.7, "0..1"), belt=(0.2, "0..1 level of the belt noise"), dur=(4.0, "s"), seed=(0, "variation"))
def scooter(sr=DEFAULT_SR, kind="2t", rpm=-1.0, load=0.7, belt=0.2, dur=4.0, seed=0):
    """Belt: noise 800-2000 Hz, growing with the road speed."""
    k = _kind(kind, SCOOTER, "scooter")
    n, rng = _n(dur), _rng(seed, 700)
    rr = cvt_rpm(float(rpm) if rpm > 0 else SCOOTER[k], float(dur), n)
    eng = two_stroke(sr=SR, rpm=rr, load=load, dur=dur, seed=seed) if k == "2t" else _run("engine_single", rr, dur, load, seed)
    bn = _unit(_hp(_lp(rng.standard_normal(n), 2000.0, 2), 800.0, 2))
    return _fin(_unit(_fit(eng, n)) + float(np.clip(belt, 0, 1)) * bn * np.linspace(0.3, 1.0, n), sr, 0.7, 0.003, 0.02)


# =============================================================================================
# TRUCK AND BUS
# =============================================================================================
BUS_IDLE = 540.0                                    # rpm, a slow diesel idle
JAKE_POP = 0.5 * 0.069                               # of a cycle: half the no-load ignition of `pipe_engine`, as its own backfire pop


def jake_pops(rr, n, cylinders=6):
    """One pop per cylinder per cycle: the exhaust valve opens near compression TDC. ``pipe_engine(pops=)`` scale: 2.5 is one backfire."""
    out = np.zeros(n + 4096)
    for i in _events(rr, _even(cylinders), n):
        m = max(int(JAKE_POP * 120.0 / rr[i] * SR), 4)
        out[i:i + m] += 2.5 * np.sin(np.pi * np.arange(m) / m)
    return out[:n]


def _jake(rpm, rpm_end, n, dur, seed, pops=True):
    rr = np.full(n, float(rpm)) if rpm_end < 0 else np.linspace(float(rpm), float(rpm_end), n)
    return _run("diesel_i6_truck", rr, dur, 0.0, seed, fire=0.0, pipe=4.0, mic=0.05, pops=jake_pops(rr, n) if pops else None)   # a short pipe, 4 ms


TRUCK_KINDS = ("truck_passby", "jake_brake")


@sfx("truck", "Heavy truck: truck_passby (commercial-vehicle Harmonoise levels, source at .75 m, Doppler), jake_brake (compression release: one exhaust bark per cylinder, the machine-gun; worst with no muffler).",
     kind=("truck_passby", "truck_passby | jake_brake"), rpm=(1800.0, "crankshaft: 1800 on six cylinders = 90 Hz"), rpm_end=(-1.0, "jake_brake: slowing to this (-1 = steady)"), speed=(80.0, "truck_passby: km/h"),
     load=(0.7, "truck_passby: 0..1"), muffler=(1, "jake_brake: 1 muffled, 0 straight stack"), distance=(7.5, "truck_passby: m"), direction=("lr", "truck_passby: lr | rl"), pan=(1, "truck_passby: 1 stereo, 0 mono"),
     dur=(5.0, "s"), seed=(0, "variation"))
def truck(sr=DEFAULT_SR, kind="truck_passby", rpm=1800.0, rpm_end=-1.0, speed=80.0, load=0.7, muffler=1, distance=7.5, direction="lr", pan=1, dur=5.0, seed=0):
    """Two sources, 80/20: high (2 prop + roll) / sqrt 5, low (prop + 2 roll) / sqrt 5, without a final weighting.
    7.5 m: the pass-by distance of the standard; the muffler as a 1.5 kHz low-pass and half the peak."""
    k = _kind(kind, TRUCK_KINDS, "truck")
    n = _n(dur)
    if k == "jake_brake":
        y = _jake(rpm, rpm_end, n, dur, seed)
        return _fin(_lp(y, 1500.0, 2) if muffler else y, sr, 0.45 if muffler else 0.9, 0.003, 0.02)
    kmh = float(np.clip(speed, 10.0, 130.0))
    v = kmh / 3.6
    rng = _rng(seed, 701)
    roll_, prop = cv_sources(kmh, diesel_engine(sr=SR, kind="diesel_i6_truck", rpm=rpm, load=load, dur=dur, seed=seed), n, rng)
    fade, s5 = _ar(n, 0.3, 0.3), math.sqrt(5.0)
    w = _wind(v, n, seed)
    hi = _flyby(fade * ((2 * prop + roll_) / s5 + w), v, distance, direction, pan, CV_HEIGHT + EAR)
    lo = _flyby(fade * (prop + 2 * roll_) / s5, v, distance, direction, pan, LOW_HEIGHT + EAR)
    return _finish_stereo(hi + lo, sr)


@sfx("bus", "City bus: a big six at a slow idle, the doors opening on their rams, the air brake's psshh; or rolling, with commercial-vehicle tyres.",
     rpm=(BUS_IDLE, "crankshaft"), load=(0.2, "0..1"), speed=(0.0, "km/h: > 0 adds Harmonoise CV rolling"), doors=(1, "1 = doors at 30 %"), brake=(1, "1 = air brake release at 75 %"), dur=(6.0, "s"), seed=(0, "variation"))
def bus(sr=DEFAULT_SR, rpm=BUS_IDLE, load=0.2, speed=0.0, doors=1, brake=1, dur=6.0, seed=0):
    """Air over the idling engine at 2.5 x `air_system`'s level (brake 105 dB) against 0.35 x the engine."""
    n = _n(dur)
    eng = diesel_engine(sr=SR, kind="diesel_i6_truck", rpm=rpm, load=load, dur=dur, seed=seed)
    y = 0.35 * _fit(eng, n)
    if speed > 0:
        roll_, prop = cv_sources(speed, eng, n, _rng(seed, 702))
        y = y + _rms(y) * roll_ * _rms(roll_) / (_rms(prop) + 1e-12)   # the table's ratio of rolling to propulsion
    if doors:
        _put(y, air_system(sr=SR, kind="bus_door", seed=seed), 0.3 * dur, 2.5)
    if brake:
        _put(y, air_system(sr=SR, kind="air_brake_release", seed=seed + 1), 0.75 * dur, 2.5)
    return _fin(y, sr, 0.7, 0.003, 0.02)


# =============================================================================================
# TRACTORS, MOWERS, SAWS
# =============================================================================================
TRACTOR_RATED = 2100.0                               # rated; idle 800-850: 28.3 Hz, 70 Hz on four
STACK_M, STACK_C = 1.2, 500.0                        # a short vertical stack; m/s at 350 C


def _stack(y):
    """A short open stack: the mouth returns the lows inverted (.6 under 400 Hz), and lets the clatter out."""
    k = int(2.0 * STACK_M / STACK_C * SR)
    out = y.copy()
    out[k:] -= 0.6 * _lp(y[:-k], 400.0)
    return out


@sfx("tractor", "Farm tractor: tractor (slow four-cylinder diesel with a short vertical stack), tractor_vintage (a big slow single, each firing apart).",
     kind=("tractor", "tractor | tractor_vintage"), rpm=(-1.0, "crankshaft; -1 = 2100 (rated) or 600 (vintage)"), load=(0.5, "0..1"), stroke=(4, "vintage: 4 or 2"), dur=(4.0, "s"), seed=(0, "variation"))
def tractor(sr=DEFAULT_SR, kind="tractor", rpm=-1.0, load=0.5, stroke=4, dur=4.0, seed=0):
    """Vintage: 600 rpm = 5 Hz, 2 or 4 stroke by model; the `modal_engine` body at 50 Hz. The modern four: bore .105 m."""
    k = _kind(kind, ("tractor", "tractor_vintage"), "tractor")
    if k == "tractor":
        y = diesel_engine(sr=SR, kind="diesel_i4", rpm=float(rpm) if rpm > 0 else TRACTOR_RATED, load=load, bore=0.105, dur=dur, seed=seed)
        return _fin(_stack(y), sr, 0.7, 0.003, 0.02)
    n, r = _n(dur), float(rpm) if rpm > 0 else 600.0
    two = int(stroke) == 2
    y = two_stroke(sr=SR, rpm=r, load=load, bore=0.1, dur=dur, seed=seed) if two else _run("engine_single", r, dur, load, seed)
    body = modal_engine(sr=SR, rpm=2.0 * r if two else r, dur=dur, body_freq=BODY_HZ, seed=seed)   # `modal_engine` is rpm / 120
    return _fin(_unit(_fit(y, n)) + 0.7 * _unit(_fit(body, n)), sr, 0.7, 0.003, 0.02)


def blade_pass(rpm, blades=2):
    return float(rpm) * int(blades) / 60.0


@sfx("mower", "Lawn mower: a single (four-stroke, or two-stroke) over the blade's pass under the deck, cutting grass.",
     kind=("lawn_mower", "lawn_mower | lawn_mower_2t"), rpm=(3000.0, "crankshaft = blade: 3000 = 25 Hz firing on 4T, 50 Hz on 2T, blade 100 Hz"), blades=(2, "blade tips"), cut=(0.6, "0..1 grass under the deck"), load=(0.6, "0..1"), dur=(3.0, "s"), seed=(0, "variation"))
def mower(sr=DEFAULT_SR, kind="lawn_mower", rpm=3000.0, blades=2, cut=0.6, load=0.6, dur=3.0, seed=0):
    """84-94 dB(A), unused: nothing here is absolute. Cutting dominates under the deck; deck `modal_engine` at 300 Hz, damping 150 1/s.
    Grass as `GROUNDS`' Sekere at 40 grains per blade pass, pulsed by it; levels chosen so the cutting holds most of the power at the default."""
    k = _kind(kind, ("lawn_mower", "lawn_mower_2t"), "mower")
    n, rng = _n(dur), _rng(seed, 703)
    r = float(rpm)
    eng = _run("engine_single", r, dur, load, seed) if k == "lawn_mower" else two_stroke(sr=SR, rpm=r, load=load, dur=dur, seed=seed)
    deck = modal_engine(sr=SR, rpm=r if k == "lawn_mower" else 2.0 * r, dur=dur, body_freq=300.0, damping=150.0, modes=8, seed=seed)
    bpf = blade_pass(r, blades)
    wind = fan(sr=SR, dur=dur, rpm=r, blades=int(blades), seed=seed)
    grass = _lp(_grains("Sekere", 40.0 * bpf, n, rng), 7000.0, 2)
    grass = np.tanh(grass / (2.0 * _rms(grass))) * np.cos(np.pi * bpf * np.arange(n) / SR) ** 2     # the deck stops one blade of grass from standing out
    c = float(np.clip(cut, 0, 1))
    return _fin(_unit(_fit(eng, n)) + 0.4 * _unit(_fit(deck, n)) + 1.2 * _unit(_fit(wind, n)) + 3.0 * c * _unit(grass), sr, 0.7, 0.003, 0.02)


CHAINSAW_RPM, CHAINSAW_IDLE, CHAINSAW_BITE = 9400.0, 2800.0, 0.15
CHAIN_LINKS = 14                                     # links past the sprocket per turn: 7 teeth, two links each


def chain_rate(rpm):
    """Hz of the chain links passing the sprocket."""
    return CHAIN_LINKS * np.asarray(rpm, float) / 60.0


def _two_stroke_tool(rpm, load, dur, seed, body=120.0):
    """`two_stroke` over a light `modal_engine` body at 120 Hz: the 2T force into a modal body, and the exhaust."""
    n = _n(dur)
    y = two_stroke(sr=SR, rpm=rpm, load=load, dur=dur, seed=seed)
    b = modal_engine(sr=SR, rpm=2.0 * np.asarray(rpm, float), dur=dur, body_freq=body, modes=7, seed=seed)
    return _unit(_fit(y, n)) + 0.25 * _unit(_fit(b, n))


@sfx("chainsaw", "Chainsaw: a two-stroke single firing once per turn, the chain's whirr rising with it; biting into wood drags the engine down.",
     rpm=(CHAINSAW_RPM, "crankshaft: 9400 at full throttle, 2800 idle"), load=(1.0, "0..1"), bite=(0.0, "0..1 into the wood, in the middle of the sound: 1 = 15 % off the speed"), dur=(3.0, "s"), seed=(0, "variation"))
def chainsaw(sr=DEFAULT_SR, rpm=CHAINSAW_RPM, load=1.0, bite=0.0, dur=3.0, seed=0):
    """The chain band sits at the link rate, CHAIN_LINKS rpm / 60, and rises with the engine."""
    n, rng = _n(dur), _rng(seed, 704)
    t = np.arange(n) / SR
    bt = float(np.clip(bite, 0, 1))
    gate = _smooth((t > 0.3 * dur) & (t < 0.8 * dur), 80.0) * bt
    rr = float(rpm) * (1.0 - CHAINSAW_BITE * gate)
    y = _two_stroke_tool(rr, load, dur, seed)
    y = y + 0.6 * _unit(_tvbp(rng.standard_normal(n), np.minimum(chain_rate(rr), 0.45 * SR), 0.7)) * (rr / CHAINSAW_RPM)
    if bt > 0:
        cut = scrape(sr=SR, dur=dur, speed=8.0, force=1.0, texture="wood", material="wood", size=0.4, seed=seed)
        y = y + 1.0 * _unit(_fit(cut, n)) * gate
    return _fin(y, sr, 0.7, 0.003, 0.02)


GARDEN = {"souffleur": 7500.0, "taille-haie": 6000.0, "debroussailleuse": 8500.0}
BLOWER_BLADES, HEDGE_RATIO, TRIMMER_LINES = 12, 5.0, 2
TRIMMER_GEAR = 1.43                                                                 # bevel reduction: the head turns slower than the crank


@sfx("garden_machine", "Two-stroke garden machines: souffleur (impeller and air jet), taille-haie (reciprocating blades chattering), debroussailleuse (spinning line).",
     kind=("souffleur", "souffleur | taille-haie | debroussailleuse"), rpm=(-1.0, "crankshaft; -1 = the machine's"), load=(0.8, "0..1"), dur=(3.0, "s"), seed=(0, "variation"))
def garden_machine(sr=DEFAULT_SR, kind="souffleur", rpm=-1.0, load=0.8, dur=3.0, seed=0):
    """Firing as `chainsaw`; the tool on top."""
    k = _kind(kind, GARDEN, "garden_machine")
    r = float(rpm) if rpm > 0 else GARDEN[k]
    n, rng = _n(dur), _rng(seed, 705)
    y = _two_stroke_tool(r, load, dur, seed)
    if k == "souffleur":
        y = y + 0.9 * _unit(_fit(fan(sr=SR, dur=dur, rpm=r, blades=BLOWER_BLADES, seed=seed), n)) \
            + 1.2 * _unit(_fit(air_leak(sr=SR, dur=dur, pressure=0.5, orifice=40.0, seed=seed), n))
    elif k == "taille-haie":
        ph = np.cumsum(np.full(n, 2.0 * r / HEDGE_RATIO / 60.0)) / SR     # two strokes per turn of the output
        hits = _impacts(n, ph, rng, slip=0.02, width=24)
        y = y + 1.2 * _unit(_modes(hits, np.array([1900.0, 3100.0, 4700.0]), np.array([90.0, 120.0, 160.0]), [1.0, 0.7, 0.4]))
    else:
        y = y + 1.0 * _unit(_fit(fan(sr=SR, dur=dur, rpm=r / TRIMMER_GEAR, blades=TRIMMER_LINES, seed=seed), n)) + 0.5 * _unit(_hp(rng.standard_normal(n), 3000.0, 2))
    return _fin(y, sr, 0.7, 0.003, 0.02)


@sfx("atv", "Quad bike: one or two cylinders with a motorcycle's body, and the dirt under it when moving.",
     cylinders=(1, "1 single | 2 parallel twin 270"), rpm=(4000.0, "crankshaft: idle 1000, limit 9000"), load=(0.6, "0..1"), speed=(0.0, "km/h: > 0 adds tyres on dirt"), dur=(3.0, "s"), seed=(0, "variation"))
def atv(sr=DEFAULT_SR, cylinders=1, rpm=4000.0, load=0.6, speed=0.0, dur=3.0, seed=0):
    """One or two cylinders, a motorcycle's body."""
    n, c = _n(dur), int(np.clip(cylinders, 1, 2))
    y = _unit(_run("engine_single" if c == 1 else "twin_270", rpm, dur, load, seed))
    y = y + 0.3 * _unit(_fit(modal_engine(sr=SR, rpm=rpm, dur=dur, body_freq=BODY_HZ, seed=seed), n))
    if speed > 0:
        y = y + 0.5 * _unit(_fit(tyre_roll(sr=SR, kind="tyre_dirt", speed=speed, dur=dur, seed=seed), n))
    return _fin(y, sr, 0.7, 0.003, 0.02)


WAGON_WHEEL, WAGON_TROT = 0.6, 3.0                   # m, m/s


@sfx("wagon", "Horse and wagon: hooves at a trot, iron-shod wooden wheels on the road, a creak per turn of the wheel.",
     speed=(WAGON_TROT, "m/s: scales the trot and the wheels"), dur=(5.0, "s"), seed=(0, "variation"))
def wagon(sr=DEFAULT_SR, speed=WAGON_TROT, dur=5.0, seed=0):
    """Trot: 358 footfalls a minute at the preferred speed of a 500 kg horse (``creatures.GAITS``), which is taken to be ``WAGON_TROT``."""
    n = _n(dur)
    v = float(np.clip(speed, 0.5, 8.0))
    pace = 358.0 * v / WAGON_TROT
    hooves = footsteps(sr=SR, ground="gravel", gait="trot", pace=pace, steps=int(dur * pace / 60.0) + 1, weight=1.5, seed=seed)
    wheels = roll(sr=SR, dur=dur, speed=v, radius=WAGON_WHEEL, roughness=0.7, material="steel", surface="wood", seed=seed)
    y = 1.0 * _unit(_fit(hooves, n)) + 0.5 * _unit(_fit(wheels, n))
    turn = 2.0 * math.pi * WAGON_WHEEL / v
    y = y + 0.5 * _creaks(n, turn, seed)
    return _fin(y, sr, 0.7, 0.003, 0.02)


def _creaks(n, turn, seed):
    c = creak(sr=SR, material="wood", dur=min(0.4, 0.8 * turn), seed=seed)
    c = c / (float(np.max(np.abs(c))) + 1e-12)
    out = np.zeros(n)
    for t in np.arange(0.1, n / SR, turn):
        _put(out, c, float(t))
    return out


# =============================================================================================
# TRACKS
# =============================================================================================
HULL = np.geomspace(80.0, 400.0, 6)                  # Hz, hull plate modes
TRACK_JITTER = 0.05                                  # 5 %
M1_PITCH, CAT_PITCHES = 0.1937, (0.17145, 0.2159)    # m; the two Cat pitches are not tied to one machine


def track_rate(kmh, pitch):
    """f = v / p."""
    return float(kmh) / 3.6 / float(pitch)


def _track(kmh, pitch, wheels, spacing, slack, teeth, weight, n, seed):
    """Sprocket engaging each link, each road wheel crossing each joint (their spacing sets their phase), the load passing from wheel to
    wheel at v / spacing (depth .5); into the shoes and the hull, over the running gear's rumble 35-90 Hz (`rumble`). The sprocket once a turn (.3),
    the slack's slaps at a quarter of the link rate. ``weight`` 0..1 scales what a light vehicle has less of."""
    rng = _rng(seed, 706)
    v = np.maximum(np.asarray(kmh, float), 0.5) / 3.6
    v = np.full(n, float(v)) if v.ndim == 0 else np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(v)), v)
    f = v / float(pitch)
    ph = np.cumsum(f) / SR
    sl = float(np.clip(slack, 0, 1))
    sp = _impacts(n, ph, rng, slip=TRACK_JITTER, width=48)
    if teeth > 0:
        sp = sp * (1.0 + 0.3 * np.cos(2.0 * np.pi * ph / teeth))
    roller = np.zeros(n)
    w = max(int(wheels), 1)
    for k in range(w):
        roller += _impacts(n, ph + (k * float(spacing) / float(pitch)) % 1.0, rng, slip=TRACK_JITTER * (1.0 + sl), width=96)
    am = 1.0 + 0.5 * np.cos(2.0 * np.pi * np.cumsum(v / float(spacing)) / SR)
    force = (sp + roller / w) * am
    if sl > 0:
        idx = np.flatnonzero(rng.random(n) < sl * 0.25 * f / SR)
        force = force + 6.0 * _pulses(idx, n, rng.uniform(0.5, 1.0, len(idx)))
    thump = _lp(sp * am, float(np.mean(f)) * 1.3, 2)     # the link rate is the main peak
    hull = _modes(force, HULL, np.full(6, 30.0), [1.0, 0.9, 0.8, 0.6, 0.5, 0.4])
    plate = metal_plate(sr=SR, size=0.35, thickness=12.0, decay=0.25, hardness=0.9, vel=0.9, quality=0.5)
    shoe = fftconvolve(force, np.asarray(plate, float)[:SR // 2])[:n]
    # Damping 30 /s as the hull's. At rumble's own 3 /s its 35 Hz mode is a line, and beat with the link rate: 57.4 - 35 = 22.4 Hz on the envelope of any M1, whatever the spacing.
    rmb = rumble(sr=SR, dur=n / SR, damping=30.0, seed=seed)
    return 1.5 * _unit(thump) + weight * (0.8 * _unit(hull) + 0.25 * _unit(_hp(shoe, 500.0)) + 0.5 * _unit(_fit(rmb, n)))


# speed km/h, pitch m, wheels, spacing m, sprocket teeth, slack, weight, engine (kind, rpm).
# tank_diesel: V12, 2000 rpm, 150 mm. Snowmobile: a 63.5 mm pitch, at the speed that makes 400 Hz of it.
TRACK_KINDS = {
    "tracked_vehicle": (30.0, 0.15, 6, 0.7, 12, 0.3, 1.0, ("diesel_6", 2000.0)),
    "tank_m1": (40.0, M1_PITCH, 7, 0.7, 11, 0.3, 1.0, ("turbine", 0.0)),
    "tank_diesel": (40.0, 0.15, 6, 0.8, 13, 0.3, 1.0, ("diesel_12", 2000.0)),
    "snowmobile": (91.44, 0.0635, 2, 0.4, 8, 0.1, 0.25, ("two_stroke", 7000.0)),
    "bulldozer": (5.0, CAT_PITCHES[0], 6, 0.5, 21, 0.2, 1.0, ("diesel_6", 1800.0)),
    "excavator_tracks": (5.0, CAT_PITCHES[0], 7, 0.45, 21, 0.3, 1.0, ("diesel_6", 1800.0)),
}
SNOW_HZ, SNOW_SHARE = 400.0, 10 ** (-2.0 / 10.0)     # 2 dB under the total


def _track_engine(spec, rpm, load, dur, seed):
    what, r0 = spec
    r = float(rpm) if rpm > 0 else r0
    if what == "turbine":                            # `jet_engine` for the gas turbine
        return jet_engine(sr=SR, dur=dur, throttle=float(np.clip(load, 0.1, 1.0)), seed=seed)
    if what == "two_stroke":
        return two_stroke(sr=SR, rpm=r, load=load, dur=dur, seed=seed)
    return diesel_engine(sr=SR, kind="diesel_i6_truck", rpm=r, load=load, cylinders=12 if what == "diesel_12" else 0, dur=dur, seed=seed)


def _with_band_share(track, other, f0, share):
    """``track`` + a g ``other``, g such that the third octave of ``f0`` holds ``share`` of the total power: g^2 (ob - s ot) + 2 g (cb - s ct) + tb - s tt = 0, from the bins."""
    T, O = np.fft.rfft(track), np.fft.rfft(other)
    fr = np.fft.rfftfreq(len(track), 1.0 / SR)
    b = (fr >= f0 * 2 ** (-1 / 6)) & (fr < f0 * 2 ** (1 / 6))
    dot = lambda u, v, m=Ellipsis: float(np.sum((u[m] * np.conj(v[m])).real))
    A, B, C = dot(O, O, b) - share * dot(O, O), dot(T, O, b) - share * dot(T, O), dot(T, T, b) - share * dot(T, T)
    return track + max((-B - math.sqrt(max(B * B - A * C, 0.0))) / A, 0.0) * other


@sfx("tracks", "Tracked vehicle: each link engaging the sprocket and crossing the road wheels at v / pitch, into the shoes and the hull, with the engine: tracked_vehicle, tank_m1 (193.7 mm, turbine), tank_diesel (V12), snowmobile (light, 400 Hz at speed), bulldozer (blade on gravel), excavator_tracks.",
     kind=("tracked_vehicle", " | ".join(TRACK_KINDS)), speed=(-1.0, "km/h, or a curve; -1 = the kind's"), pitch=(0.0, "m of a link; 0 = the kind's"), wheels=(0, "road wheels; 0 = the kind's"),
     wheel_spacing=(0.0, "m between road wheels; 0 = the kind's"), slack=(-1.0, "0..1 slapping of a loose track; -1 = the kind's"), engine=(0.6, "level of the engine against the track; 0 = track alone; snowmobile: any > 0, the level comes from the 2 dB"), rpm=(-1.0, "engine; -1 = the kind's"),
     load=(0.6, "engine 0..1"), dur=(4.0, "s"), seed=(0, "variation"))
def tracks(sr=DEFAULT_SR, kind="tracked_vehicle", speed=-1.0, pitch=0.0, wheels=0, wheel_spacing=0.0, slack=-1.0, engine=0.6, rpm=-1.0, load=0.6, dur=4.0, seed=0):
    k = _kind(kind, TRACK_KINDS, "tracks")
    s0, p0, w0, a0, teeth, sl0, weight, eng = TRACK_KINDS[k]
    kmh = s0 if np.ndim(speed) == 0 and float(speed) < 0 else speed
    p = float(pitch) if pitch > 0 else p0
    n = _n(dur)
    y = _track(kmh, p, int(wheels) if wheels > 0 else w0, float(wheel_spacing) if wheel_spacing > 0 else a0, float(slack) if slack >= 0 else sl0, teeth, weight, n, seed)
    if engine > 0:
        e = _unit(_fit(_track_engine(eng, rpm, load, dur, seed), n))
        y = _with_band_share(y, e, track_rate(kmh, p), SNOW_SHARE) if k == "snowmobile" and np.ndim(kmh) == 0 else y + float(engine) * e
    if k == "bulldozer":                             # blade = `scrape` on gravel; grains as `tyre_gravel`, 40 v
        rng = _rng(seed, 707)
        v = float(np.mean(kmh)) / 3.6
        blade = scrape(sr=SR, dur=dur, speed=max(v, 0.5), force=1.0, texture="gritty", material="steel", size=1.2, seed=seed)
        y = y + 0.6 * _unit(_fit(blade, n)) + 0.4 * _lp(_grains("BigRocks", 40.0 * v, n, rng), 9000.0, 2)
    return _fin(y, sr, 0.7, 0.003, 0.02)


# =============================================================================================
# HYDRAULICS AND SITE MACHINES
# =============================================================================================
PUMP_LOW, PUMP_HIGH = 80.0, 200.0                    # bar, working band


def pump_rate(rpm, pistons=9):
    """f = z rpm / 60: 9 pistons at 2000 rpm = 300 Hz."""
    return int(pistons) * float(rpm) / 60.0


def _pump(rpm, pressure, pistons, dur, seed, rpm_end=-1.0):
    """The tonal line of `gearbox` with teeth = z, clean: one tooth per piston, its load law standing for the rise of the ripple with
    pressure. Level .25 to .7 across the band."""
    z = int(pistons)
    if z % 2 == 0 or z < 3:
        raise ValueError(f"hydraulic_pump pistons {pistons}: an odd number, 3 or more")
    x = (float(pressure) - PUMP_LOW) / (PUMP_HIGH - PUMP_LOW)
    y = gearbox(sr=SR, dur=dur, rpm=float(rpm), rpm_end=rpm_end, teeth=z, teeth2=z, load=float(np.clip(x, 0.05, 1.0)), wear=0.0, eccentricity=0.0, backlash=0.0, housing=1300.0, seed=seed)
    return y


def _pump_peak(pressure):
    return 0.25 + 0.45 * float(np.clip((float(pressure) - PUMP_LOW) / (PUMP_HIGH - PUMP_LOW), 0.0, 1.0))


@sfx("hydraulic_pump", "Hydraulic whine of an axial-piston pump, a line and its harmonics at pistons x rpm / 60, rising with the pressure; arm: the pump under load as a ram travels, hits its stop and blows the relief valve.",
     kind=("pump", "pump | arm"), rpm=(2000.0, "shaft"), pressure=(150.0, "bar: 80-200 working"), pistons=(9, "odd"), dur=(3.0, "s"), seed=(0, "variation"))
def hydraulic_pump(sr=DEFAULT_SR, kind="pump", rpm=2000.0, pressure=150.0, pistons=9, dur=3.0, seed=0):
    """Arm: relief as `air_leak`, stop as a metal `impact`, and the end of travel at 65 % of the sound."""
    k = _kind(kind, ("pump", "arm"), "hydraulic_pump")
    if k == "pump":
        return _fin(_pump(rpm, pressure, pistons, dur, seed), sr, _pump_peak(pressure), 0.01, 0.03)
    n = _n(dur)
    t = 0.65 * float(dur)
    y = _unit(_fit(_pump(rpm, pressure, pistons, dur, seed, rpm_end=0.94 * float(rpm)), n))
    y = y * np.interp(np.arange(n) / SR, [0.0, t, t + 0.05, float(dur)], [0.7, 0.8, 1.2, 1.2])
    d = float(dur) - t
    relief = _unit(air_leak(sr=SR, dur=d, pressure=0.5, orifice=6.0, seed=seed)) * np.minimum(np.arange(_n(d)) / (0.01 * SR), 1.0)
    _put(y, _fit(relief, _n(d)), t, 0.6)
    _put(y, impact(sr=SR, material="steel", shape="bar", size=0.5, velocity=3.0, hardness=0.8, modes=12, dur=min(0.6, d), seed=seed), t, 6.0)
    return _fin(y, sr, 0.7, 0.003, 0.02)


EXCAVATOR_RPM = 1650.0                               # 1650-2000
GRAVEL_HZ = 6460.0                                   # large gravel, `particles.GROUNDS`


def _scoop(dur, seed):
    """A bite: the bucket drags in, the pump under load. A swing. Then the load falls out, grains (BigRocks = large gravel, 6460 +- 701 Hz) and stones landing."""
    n, rng = _n(dur), _rng(seed, 709)
    y = 0.5 * _unit(_fit(_pump(EXCAVATOR_RPM, 170.0, 9, dur, seed), n))
    a, b = 0.1 * dur, 0.4 * dur
    dig = _unit(_fit(scrape(sr=SR, dur=b - a, speed=0.6, force=1.0, texture="gritty", material="steel", size=0.8, seed=seed), _n(b - a)))
    _put(y, dig * _ar(_n(b - a), 0.05, 0.1), a, 1.0)
    t = 0.65 * dur
    m = _n(0.3 * dur)
    env = np.exp(-np.arange(m) / (0.08 * dur * SR))
    _put(y, _grains("BigRocks", 1500.0, m, rng) * env, t, 1.5)
    _put(y, _fit(np.asarray(rumble(sr=SR, dur=0.3 * dur, fmin=40.0, fmax=120.0, seed=seed), float), m) * env, t, 1.0)
    for i in range(3):
        _put(y, drop(sr=SR, material="stone", size=0.08, surface="stone", height=1.0 + i, roll=0.2, seed=seed + i), t + 0.03 + 0.06 * i, 0.5)
    return y


@sfx("excavator", "Excavator: idle (big diesel at working speed, the pump over it), tracks (slow crawl on its crawlers), scoop (a bucket bite, a swing, the load tipping out).",
     kind=("idle", "idle | tracks | scoop"), rpm=(EXCAVATOR_RPM, "diesel and pump shaft"), speed=(5.0, "tracks: km/h"), pitch=(CAT_PITCHES[0], "tracks: m (171.45 or 215.9 mm)"), load=(0.5, "0..1"), dur=(4.0, "s"), seed=(0, "variation"))
def excavator(sr=DEFAULT_SR, kind="idle", rpm=EXCAVATOR_RPM, speed=5.0, pitch=CAT_PITCHES[0], load=0.5, dur=4.0, seed=0):
    """A six-cylinder diesel at working speed, the pump over it."""
    k = _kind(kind, ("idle", "tracks", "scoop"), "excavator")
    if k == "tracks":
        return tracks(sr=sr, kind="excavator_tracks", speed=speed, pitch=pitch, rpm=rpm, load=load, dur=dur, seed=seed)
    if k == "scoop":
        return _fin(_scoop(float(dur), seed), sr, 0.7, 0.003, 0.02)
    n = _n(dur)
    y = _unit(_fit(diesel_engine(sr=SR, kind="diesel_i6_truck", rpm=rpm, load=load, dur=dur, seed=seed), n))
    return _fin(y + 0.5 * _unit(_fit(_pump(rpm, 100.0, 9, dur, seed), n)), sr, 0.7, 0.003, 0.02)
