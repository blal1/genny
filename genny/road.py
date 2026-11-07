"""The road: electric drives and AVAS, tyres on every surface, truck air systems, car doors, horns, cabin mechanics.

Sources of every constant: research/SOURCES.md, section road. Nothing here has been listened to: see tests/test_road.py for what was measured.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from . import contact as K
from . import foley as FO
from .core import DEFAULT_SR
from .cyber import gearbox, pipe_engine, road_noise_levels, tyre_squeal, _third_octave_noise
from .dsp import wall
from .foley import C_AIR, CATCH, ENDSTOP, P, RATCHET, _fin, _mclick, _n, _nm, _put
from .friction import squeak
from .fx import _per_channel, effect, muffle
from .matter import air_leak
from .physical import ambience as amb
from .physical import modal
from .physical import particles as pt
from .pinball import solenoid
from .reverbs import occlude
from .sfx import electric_motor, sfx
from .tubes import tube
from .vehicle import car_engine


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    return np.pad(x, (0, n - len(x))) if len(x) < n else x[:n]


def _band(x: np.ndarray, lo: float, hi: float, sr: int = P, order: int = 2) -> np.ndarray:
    return sosfilt(butter(order, [lo, min(hi, 0.45 * sr)], "bandpass", fs=sr, output="sos"), x)


def _kind(kind, table, what) -> str:
    k = str(kind).replace(" ", "_")
    if k not in table:
        raise ValueError(f"{what} kind {kind!r}: use {' | '.join(table)}")
    return k


def _modes(x: np.ndarray, freqs, t60, gains) -> np.ndarray:
    fm = np.asarray(freqs, float)
    return modal.render_modes(x, modal.Modes(fm, np.full(len(fm), 6.91 / float(t60)), np.asarray(gains, float), "vdd"))


# =============================================================================================
# ELECTRIC DRIVE
# =============================================================================================
def whine_orders(poles: int, slots: int) -> list[int]:
    """Shaft orders of the electromagnetic whine: 6p and its multiples, plus the slot order and its double.
    4 pole pairs, 48 slots give the measured H24, H48, H72, H96. 6p: the sixth electrical harmonic of a three-phase machine."""
    p = max(int(poles), 1)
    return sorted({6 * p * k for k in range(1, 5)} | {max(int(slots), 1), 2 * max(int(slots), 1)})


def ev_whine(speed=6000.0, poles=4, slots=48, fs=7500.0, load=0.6, sr=DEFAULT_SR, seed=0, *, dur=1.0, fs_spread=0.0) -> np.ndarray:
    """Inverter-fed motor, no gears or tyres. ``speed``: motor rpm, number or per-sample array (the array sets the length,
    otherwise ``dur``). ``poles``: pole pairs. ``fs``: carrier Hz, number or per-sample array. ``load``: 0..1, number or array.
    ``fs_spread``: Hz, the carrier jumps at random within fs +- spread. Mono, peak below 1, ends faded by 5 ms.
    Orders: `whine_orders`. PWM: f = k1 fs +- k2 f0, f0 = poles rpm / 60, the space-vector set of (k1, k2) pairs."""
    rng = np.random.default_rng(int(seed))
    speed = np.asarray(speed, float)
    n = int(speed.size) if speed.ndim else max(1, int(round(float(dur) * sr)))
    rpm = np.broadcast_to(np.maximum(speed, 0.0), (n,))
    ld = np.broadcast_to(np.clip(np.asarray(load, float), 0.0, 1.0), (n,))
    fc = np.broadcast_to(np.asarray(fs, float), (n,)).astype(float)
    if fs_spread > 0:
        hop = max(1, int(0.005 * sr))
        fc = fc + float(fs_spread) * np.interp(np.arange(n), np.arange(n // hop + 2) * hop, rng.uniform(-1, 1, n // hop + 2))
    shaft = np.cumsum(rpm / 60.0) / sr
    carrier = np.cumsum(fc) / sr
    p = max(int(poles), 1)
    top, y = 0.45 * sr, np.zeros(n)

    def add(phase, f, g):
        np.add(y, g * np.clip((top - f) / (0.03 * sr), 0.0, 1.0) * np.clip(f / 20.0, 0.0, 1.0) * np.sin(2 * np.pi * phase + rng.uniform(0, 6.28)), out=y)

    orders = whine_orders(p, slots)
    for o in orders:
        add(o * shaft, o * rpm / 60.0, (0.25 + 0.75 * ld) * orders[0] / o)
    for k1, k2, g in ((1, 2, 0.16), (1, 4, 0.06), (2, 1, 0.08), (2, 5, 0.03)):
        for sign in (-1, 1):
            add(k1 * carrier + sign * k2 * p * shaft, np.abs(k1 * fc + sign * k2 * p * rpm / 60.0), g * (0.5 + 0.5 * ld))
    k = min(n // 2, max(2, int(0.005 * sr)))
    y[:k] *= np.linspace(0.0, 1.0, k)
    y[n - k:] *= np.linspace(1.0, 0.0, k)
    return y * 0.3


def motor_rpm(kmh, reduction: float = 9.0, wheel: float = 0.33):
    """9:1 single-speed reduction and a 0.33 m tyre radius, as `cyber.pass_by`."""
    return np.asarray(kmh, float) / 3.6 / (2.0 * math.pi * wheel) * 60.0 * reduction


def _speed_track(speed, speed_end, n) -> np.ndarray:
    a = max(float(speed), 1.0)
    return np.full(n, a) if speed_end is None or float(speed_end) < 0 else np.geomspace(a, max(float(speed_end), 1.0), n)


def _drive(kmh: np.ndarray, load, regen, poles, slots, reduction, wheel, seed) -> np.ndarray:
    """Whine + reduction gear mesh, at P; regen loads the gears and rattles their backlash."""
    rpm = motor_rpm(kmh, reduction, wheel)
    rg = float(np.clip(regen, 0.0, 1.0))
    ld = max(float(np.clip(load, 0.0, 1.0)), rg)
    y = ev_whine(rpm, poles, slots, 7500.0, ld, P, seed, fs_spread=250.0)
    mesh = gearbox(sr=P, dur=len(y) / P, rpm=float(rpm[0]), rpm_end=float(rpm[-1]) if rpm[-1] != rpm[0] else -1.0, teeth=17,
                   teeth2=int(round(17 * reduction)), load=ld, wear=0.05, eccentricity=0.15, backlash=0.3 * rg, housing=2600.0, seed=seed)
    return y + 0.25 * _fit(mesh, len(y))


@sfx("ev_drive", "Electric car drive heard alone: slot-order whine (orders 24, 48, 72, 96), reduction gear mesh, PWM carrier at 7.25-7.75 kHz with its sidebands.",
     speed=(60.0, "km/h"), speed_end=(-1.0, "glide to this km/h (-1 = steady)"), load=(0.6, "0 coasting .. 1 full torque"),
     regen=(0.0, "0..1 regenerative braking: torque back through the gears"), poles=(4, "pole pairs"), slots=(48, "stator slots"),
     reduction=(9.0, "gear ratio motor to wheel"), wheel=(0.33, "m wheel radius"), dur=(3.0, "s"), seed=(0, "variation"))
def ev_drive(sr=DEFAULT_SR, speed=60.0, speed_end=-1.0, load=0.6, regen=0.0, poles=4, slots=48, reduction=9.0, wheel=0.33, dur=3.0, seed=0):
    kmh = _speed_track(speed, speed_end, _n(float(np.clip(dur, 0.05, 60.0))))
    return _fin(_drive(kmh, load, regen, poles, slots, reduction, wheel, seed), sr, 0.6, fade=0.02, tail=0.05)


@sfx("ev", "Electric vehicles with their tyres: voiture_electrique (road car), trottinette (kick scooter hub motor), voiturette (golf cart, brushed motor).",
     kind=("voiture_electrique", "voiture_electrique | trottinette | voiturette"), speed=(-1.0, "km/h (-1 = the kind's own)"), load=(0.5, "0..1"),
     dur=(3.0, "s"), seed=(0, "variation"))
def ev(sr=DEFAULT_SR, kind="voiture_electrique", speed=-1.0, load=0.5, dur=3.0, seed=0):
    """`electric_motor`, tyre, `pass_by(power=electric)`: a brushless hub for the scooter, brushed DC for the cart."""
    k = _kind(kind, EV, "ev")
    kmh, wheel, reduction, poles, slots, tyre = EV[k]
    kmh = float(speed) if float(speed) > 0 else kmh
    n, rng = _n(float(np.clip(dur, 0.05, 60.0))), np.random.default_rng(int(seed))
    if k == "voiture_electrique":
        m = _drive(np.full(n, kmh), load, 0.0, poles, slots, reduction, wheel, seed)
    elif k == "trottinette":
        m = ev_whine(float(motor_rpm(kmh, reduction, wheel)), poles, slots, 16000.0, load, P, seed, dur=n / P)
    else:
        m = _fit(electric_motor(sr=P, dur=n / P, rpm=float(motor_rpm(kmh, reduction, wheel)), load=load, seed=seed), n)
    return _fin(_nm(m) + tyre * _nm(_rolling(kmh, n, rng)), sr, 0.7, fade=0.02, tail=0.05)


# speed km/h, wheel m, reduction, pole pairs, slots, tyre gain
EV = {"voiture_electrique": (50.0, 0.33, 9.0, 4, 48, 0.8), "trottinette": (20.0, 0.1, 1.0, 15, 27, 0.3), "voiturette": (20.0, 0.23, 12.0, 2, 24, 0.5)}


# Minimum dB(A) against km/h: 50 at 10, 56 at 20, 47 at 6 in reverse, linear between, clamped outside; FMVSS base 50 dB.
AVAS_RULES = ("unece", "fmvss")
AVAS_F = (400.0, 1600.0)                                    # the two bands, one third-octave at 1600 Hz or less
AVAS_STYLES = {"hum": (1.0, 0.35, 0.0), "chime": (0.5, 1.0, 0.25), "pad": (1.0, 0.7, 0.1)}     # band 1, band 2, FM depth


def avas_level_db(kmh, rule="unece", reverse=False):
    if rule == "fmvss":
        return 50.0 + 3.0 * np.searchsorted([11.0, 21.0, 31.0], kmh, side="right")
    return np.full_like(np.asarray(kmh, float), 47.0) if reverse else np.interp(kmh, [10.0, 20.0], [50.0, 56.0])


def avas_pitch(kmh, rule="unece"):
    """UN R138: pitch +0.8 % per km/h at least; FMVSS 141 asks no glide."""
    return np.ones_like(np.asarray(kmh, float)) if rule == "fmvss" else 1.0 + 0.008 * np.asarray(kmh, float)


@sfx("avas", "Pedestrian warning sound of an electric car: two bands, pitch and level tied to speed. rule=unece: +0.8 %/km/h, 50 dB(A) at 10 km/h, 56 at 20, 47 reversing. rule=fmvss: no glide, +3 dB at 11, 21, 31 km/h.",
     speed=(10.0, "km/h"), speed_end=(-1.0, "glide to this km/h (-1 = steady)"), rule=("unece", "unece | fmvss"), reverse=(0, "1 = reversing"),
     style=("pad", "hum | chime | pad (timbre, free under the rules)"), dur=(2.0, "s"), seed=(0, "variation"))
def avas(sr=DEFAULT_SR, speed=10.0, speed_end=-1.0, rule="unece", reverse=0, style="pad", dur=2.0, seed=0):
    """Peak follows the level: 75 dB(A), the UN maximum, would be full scale, so the game adds its own gain.
    The speed is held at the rule's top (20 km/h, 31 km/h) above which the sound may stop: the game decides."""
    if rule not in AVAS_RULES:
        raise ValueError(f"avas rule {rule!r}: use unece or fmvss")
    if style not in AVAS_STYLES:
        raise ValueError(f"avas style {style!r}: use {' | '.join(AVAS_STYLES)}")
    n = _n(float(np.clip(dur, 0.05, 60.0)))
    a, b = float(speed), float(speed) if float(speed_end) < 0 else float(speed_end)
    v = np.clip(np.linspace(a, b, n), 0.0, 20.0 if rule == "unece" else 31.0)
    g1, g2, fm = AVAS_STYLES[style]
    rng = np.random.default_rng(int(seed))
    pitch = avas_pitch(v, rule)
    x = np.zeros(n)
    for f, g in zip(AVAS_F, (g1, g2)):
        ph = np.cumsum(f * pitch) / P
        x += g * np.sin(2 * np.pi * ph + rng.uniform(0, 6.28) + fm * 4.0 * np.sin(2 * np.pi * 2.0 * ph))
    x /= g1 + g2
    y = x * 10.0 ** ((avas_level_db(v, rule, bool(reverse)) - 75.0) / 20.0)
    return _fin(y, sr, None, fade=0.03, tail=0.05)


@sfx("hybrid_transition", "Hybrid changing over: the electric drive runs, the combustion engine cranks for 0.3 s, catches at 1200 rpm and takes over.",
     speed=(40.0, "km/h"), at=(1.0, "s when the engine starts"), dur=(3.0, "s"), seed=(0, "variation"))
def hybrid_transition(sr=DEFAULT_SR, speed=40.0, at=1.0, dur=3.0, seed=0):
    """The generator spins the engine from 300 rpm to 1200 in 0.3 s, then it fires and the whine fades."""
    d = float(np.clip(dur, 0.6, 60.0))
    at = float(np.clip(at, 0.1, d - 0.4))
    n, m0 = _n(d), _n(at)
    y = _nm(_drive(np.full(n, max(float(speed), 1.0)), 0.6, 0.0, 4, 48, 9.0, 0.33, seed))
    y *= np.interp(np.arange(n), [m0, m0 + _n(0.4)], [1.0, 0.25])
    crank = pipe_engine(sr=P, dur=0.3, rpm=300.0, rpm_end=1200.0, cylinders=4, load=0.3, seed=seed)
    idle = pipe_engine(sr=P, dur=d - at - 0.25, rpm=1200.0, cylinders=4, load=0.3, seed=seed)
    e = np.zeros(n)
    w = np.interp(np.arange(len(idle)), [0, _n(0.05)], [0.0, 1.0])
    _put(e, crank * np.interp(np.arange(len(crank)), [len(crank) - _n(0.05), len(crank)], [1.0, 0.0]), at)
    _put(e, idle * w, at + 0.25)
    return _fin(0.5 * y + _nm(e), sr, 0.7, fade=0.02, tail=0.05)


# =============================================================================================
# TYRES
# =============================================================================================
def _rolling(kmh: float, n: int, rng) -> np.ndarray:
    """Harmonoise rolling noise; RMS follows the source model's total level, 0.15 at 130 km/h (the 0.15 is a mix level)."""
    lr, _ = road_noise_levels(kmh)
    y = _third_octave_noise(n, rng, lr)
    return y / (np.sqrt(np.mean(y * y)) + 1e-12) * 0.15 * 10.0 ** ((_rolling_total(kmh) - _rolling_total(130.0)) / 20.0)


def _rolling_total(kmh: float) -> float:
    return 10.0 * math.log10(float(np.sum(10.0 ** (road_noise_levels(kmh)[0] / 10.0))))


def _tread_line(kmh: float, n: int, rng) -> np.ndarray:
    """Blocks of a tread at a variable pitch of 19-39 mm, drawn uniformly: 0.5-1 kHz at 70 km/h, proportional to speed, a band of Q 3."""
    v = max(float(kmh), 1.0) / 3.6
    k = int(n / P * v / 0.019) + 2
    at = (np.cumsum(rng.uniform(0.019, 0.039, k)) / v * P).astype(int)
    x = np.zeros(n)
    x[at[at < n]] = 1.0
    y = amb.bp(x, min(v / 0.029, 0.4 * P), 3.0)
    return y / (np.sqrt(np.mean(y * y)) + 1e-12)


def _grains(preset: str, rate: float, n: int, rng) -> np.ndarray:
    """PhISEM grains of ``preset`` at ``rate`` per second, Poisson, power-law energy (as `footsteps._texture`).
    A rolling wheel has no loading ramp, so the rate is held, and fscale stays 1. Mono, unit peak."""
    p = pt.preset(preset)
    q = pt._engine_params(p, None, 1.0, None)
    ev = rng.random(n) < min(rate / P, 1.0)
    amps = np.zeros(n)
    k = int(ev.sum())
    if k:
        e = pt._powerlaw(rng, k, pt.GRAIN_E_MIN, 1.0, pt.VISELL_GAMMA)
        amps[ev] = np.sqrt(e / e.sum())
    y = pt._phisem_kernel(p["kind"], False, np.zeros(n), False, True, amps, np.ones(n), q["Ds"], q["Dn"], q["p_event"], 1.0, q["base_f"],
                          q["radii"], q["fgains"], q["vary_on"], q["vary"], q["eq"][0], q["eq"][1], q["eq"][2], q["sweep"],
                          q["ratchet_delta"], q["ratchet_c"], pt.MIN_ENERGY, pt.MAX_SHAKE, 2, int(rng.integers(0, 2**31 - 1)))
    return _nm(np.asarray(y, float))


# PhISEM preset (`particles.GROUNDS`), tyre rolling level left under the grains, high-pass of that bed (loose ground takes the lows away).
SURFACES = {"tyre_gravel": ("BigRocks", 0.6, 120.0), "tyre_dirt": ("Crunch", 0.5, 150.0), "tyre_sand": ("Sandpaper", 0.3, 200.0),
            "tyre_snow": ("Crunch", 0.25, 250.0), "tyre_mud": ("WaterDrops", 0.3, 200.0)}
TYRE_KINDS = ("tyre_road", "tyre_wet", *SURFACES, "tyre_cobble", "tyre_burnout", "tyre_lockup", "road_cruise_interior")
COBBLE_L = 0.12                                             # m: a 10-12 cm sett and its joint
ABS_HZ = 15.0                                               # pressure cycles per second
CONTACT_A = 0.05                                            # m, half contact patch of a car tyre


def _slip_lag(x: np.ndarray, kmh: float) -> np.ndarray:
    """Slip does not follow the brake at once: relaxation length 3a, tau = 3a / v."""
    tau = 3.0 * CONTACT_A / (max(float(kmh), 1.0) / 3.6)
    r = math.exp(-1.0 / (tau * P))
    return lfilter([1.0 - r], [1.0, -r], x)


def _cobble(kmh: float, n: int, rng) -> np.ndarray:
    """Rubber on stone at f = v / L, the stone differing each time, the spacing 12 % uneven."""
    v = max(float(kmh), 1.0) / 3.6
    hit = FO._nm(K.impact(sr=P, material="rubber", shape="rect_plate", size=0.15, velocity=2.0, hardness=0.4, modes=10, dur=0.05, seed=int(rng.integers(1 << 30))))
    k = int(n / P * v / COBBLE_L) + 2
    at = np.cumsum(COBBLE_L / v * (1.0 + 0.12 * rng.uniform(-1, 1, k)))
    x = np.zeros(n)
    x[(at[at * P < n] * P).astype(int)] = rng.uniform(0.6, 1.0, int(np.sum(at * P < n)))
    return _fit(np.convolve(x, hit), n)


@sfx("tyre_roll", "Tyres rolling: tyre_road (Harmonoise bands + tread line), tyre_wet, tyre_gravel, tyre_dirt, tyre_sand, tyre_snow, tyre_mud (PhISEM grains at 40 v), tyre_cobble (v / 0.12 m), tyre_burnout, tyre_lockup (ABS), road_cruise_interior.",
     kind=("tyre_road", " | ".join(TYRE_KINDS)), speed=(70.0, "km/h"), tread=(0.5, "tyre_road, tyre_wet: level of the tread line"), dur=(3.0, "s"), seed=(0, "variation"))
def tyre_roll(sr=DEFAULT_SR, kind="tyre_road", speed=70.0, tread=0.5, dur=3.0, seed=0):
    """Levels between speeds are the source model's: no normalisation (`tyre_road`, `tyre_wet`, the surfaces, cobbles).
    Wet: spray above 2 kHz at 0.8 of the rolling level (as `cyber.pass_by`). Grains at 40 per metre."""
    k = _kind(kind, TYRE_KINDS, "tyre_roll")
    kmh = float(np.clip(speed, 1.0, 250.0))
    n, rng = _n(float(np.clip(dur, 0.05, 60.0))), np.random.default_rng(int(seed))
    bed = _rolling(kmh, n, rng)
    lvl = float(np.sqrt(np.mean(bed * bed)))
    tr = float(np.clip(tread, 0.0, 2.0)) * lvl
    if k == "tyre_road":
        y = bed + tr * _tread_line(kmh, n, rng)
    elif k == "tyre_wet":
        y = bed + tr * _tread_line(kmh, n, rng) + 0.8 * lvl * _unit_rms(amb.hip(rng.standard_normal(n), 2000.0))
    elif k in SURFACES:
        preset, g, hp = SURFACES[k]
        y = g * amb.hip(bed, hp) + lvl * 3.0 * _grains(preset, 40.0 * kmh / 3.6, n, rng)
    elif k == "tyre_cobble":
        y = 0.5 * bed + lvl * 3.0 * _nm(_cobble(kmh, n, rng))
    elif k == "tyre_burnout":
        d = n / P
        y = _fit(tyre_squeal(sr=P, dur=d, slip=1.5, seed=seed), n) + 0.7 * _fit(pipe_engine(sr=P, dur=d, rpm=5500.0, load=1.0, seed=seed), n) \
            + 0.1 * _fit(air_leak(sr=P, dur=d, pressure=0.3, orifice=15.0, seed=seed), n)
        return _fin(y, sr, 0.8, fade=0.05, tail=0.2)
    elif k == "tyre_lockup":
        am = _slip_lag(0.5 + 0.5 * np.tanh(4.0 * np.sin(2 * np.pi * ABS_HZ * np.arange(n) / P)), kmh)
        return _fin(_fit(tyre_squeal(sr=P, dur=n / P, slip=0.8, seed=seed), n) * am + 0.3 * _nm(bed), sr, 0.8, fade=0.02, tail=0.05)
    else:
        road = bed + tr * _tread_line(kmh, n, rng)
        eng = _fit(pipe_engine(sr=P, dur=n / P, rpm=float(np.clip(26.0 * 2.2 * kmh, 700.0, 6000.0)), load=0.3, seed=seed), n)
        cab = wall(_nm(road) + 0.5 * eng, P, material="window")
        return _fin(muffle(np.asarray(cab, float), P, amount=0.6), sr, 0.6, fade=0.05, tail=0.1)
    return _fin(y, sr, None, fade=0.02, tail=0.05)


def _unit_rms(y: np.ndarray) -> np.ndarray:
    return y / (np.sqrt(np.mean(y * y)) + 1e-12)


@sfx("interior", "Heard from the cabin, driving: engine, tyres, floor vibration, all through the shell. kind car | ev | truck | bus.",
     kind=("car", "car | ev | truck | bus"), speed=(90.0, "km/h"), dur=(4.0, "s"), seed=(0, "variation"))
def interior(sr=DEFAULT_SR, kind="car", speed=90.0, dur=4.0, seed=0):
    """Bed + floor vibrations + occlusion (`car_engine`, `occlude`, `rumble`); rpm in top gear by km/h."""
    k = _kind(kind, INTERIORS, "interior")
    rpm_per_kmh, cyl, vib, thick = INTERIORS[k]
    kmh = float(np.clip(speed, 5.0, 200.0))
    d = float(np.clip(dur, 0.1, 60.0))
    n, rng = _n(d), np.random.default_rng(int(seed))
    if k == "ev":
        eng = _drive(np.full(n, kmh), 0.4, 0.0, 4, 48, 9.0, 0.33, seed)
    else:
        eng = car_engine(sr=P, dur=d, rpm=float(np.clip(rpm_per_kmh * kmh, 600.0, 4000.0)), cylinders=cyl, load=0.4, seed=seed)
    rmb = K.rumble(sr=P, dur=d, fmin=25.0, fmax=70.0, seed=seed)
    outside = _nm(_rolling(kmh, n, rng)) + 0.7 * _nm(_fit(np.asarray(eng, float), n))
    return _fin(occlude(outside, P, material="metal", thickness=thick, leak=0.1, edge=500.0) + vib * _nm(_fit(np.asarray(rmb, float), n)), sr, 0.6, fade=0.05, tail=0.1)


# rpm per km/h, cylinders, vibration level, shell m
INTERIORS = {"car": (28.0, 4, 0.3, 0.002), "ev": (0.0, 4, 0.2, 0.002), "truck": (15.0, 6, 0.6, 0.0015), "bus": (15.0, 6, 0.5, 0.001)}


# =============================================================================================
# AIR SYSTEMS
# =============================================================================================
# Peak of each sound: brakes 95-115 dB, midpoint 105; dryer 115-120 dB, midpoint 117.5, full level 0.8.
_DB_FULL = 117.5


def _db_peak(db: float) -> float:
    return 0.8 * 10.0 ** ((db - _DB_FULL) / 20.0)


# burst s, orifice mm: `air_leak pressure=1` is 343 m/s, sonic like 690-830 kPa
AIR_LEAKS = {"bus_kneel": (0.8, 8.0), "air_brake_apply": (0.25, 10.0), "air_brake_release": (0.25, 10.0), "air_dryer_purge": (0.15, 20.0)}
AIR_KINDS = ("bus_door", *AIR_LEAKS, "parking_brake_spring")
MEMBRANE_HZ = 150.0                                         # AM of the valve membrane ("honking")
DOOR_STROKE = 1.2                                           # s


@sfx("air_system", "Truck and bus compressed air: bus_door (ram and stop), bus_kneel (long discharge), air_brake_apply, air_brake_release (psshh with membrane flutter), air_dryer_purge (short, violent), parking_brake_spring (discharge, then the spring lands).",
     kind=("air_brake_release", " | ".join(AIR_KINDS)), seed=(0, "variation"))
def air_system(sr=DEFAULT_SR, kind="air_brake_release", seed=0):
    """Levels between kinds as measured on air brakes; the bus kinds at the brake level."""
    k = _kind(kind, AIR_KINDS, "air_system")
    if k in AIR_LEAKS:
        burst, orifice = AIR_LEAKS[k]
        y = air_leak(sr=P, dur=max(0.4, 3.0 * burst), pressure=1.0, orifice=orifice, burst=burst, seed=seed)
        if k == "air_brake_release":
            y = y * (0.65 + 0.35 * np.sin(2 * np.pi * MEMBRANE_HZ * np.arange(len(y)) / P))
        return _fin(y, sr, _db_peak(117.5 if k == "air_dryer_purge" else 105.0), fade=0.002, tail=0.05)
    if k == "bus_door":                                     # one stroke of `piston`; the stop and the exhaust burst
        n = _n(DOOR_STROKE + 0.5)
        y = _fit(FO.piston(sr=P, cycles=1, rate=0.5 / DOOR_STROKE, pressure=0.8, seed=seed), _n(DOOR_STROKE))
        y[-_n(0.02):] *= np.linspace(1.0, 0.0, _n(0.02))
        out = np.zeros(n)
        out[:len(y)] = y
        _put(out, air_leak(sr=P, dur=0.5, pressure=0.6, orifice=8.0, burst=0.2, seed=seed), DOOR_STROKE, 0.4)
    else:                                                   # spring applied on loss of air
        out = np.zeros(_n(1.2))
        _put(out, air_leak(sr=P, dur=0.8, pressure=1.0, orifice=12.0, burst=0.3, seed=seed), 0.0, 0.6)
        _put(out, K.impact(sr=P, material="steel", shape="bar", size=0.25, velocity=4.0, hardness=0.8, modes=12, dur=0.5, seed=seed), 0.12, 1.0)
    return _fin(out, sr, _db_peak(105.0), fade=0.002, tail=0.05)


# =============================================================================================
# DOORS
# =============================================================================================
# band Hz, quality 0 tinny .. 1 solid, panel m, body Hz, length s, closing force.
# Bands: front 200-2000, rear 315-2500; cheap above 500. Panel 20 ms after the latch, 200 ms long.
DOORS = {
    "car_door_close": ((200.0, 2000.0), 0.8, 0.9, 160.0, 0.2),
    "car_door_rear": ((315.0, 2500.0), 0.7, 0.75, 340.0, 0.2),
    "car_door_cheap": ((500.0, 4000.0), 0.15, 0.8, 300.0, 0.15),
    "trunk_slam": ((80.0, 1200.0), 0.85, 1.2, 110.0, 0.3),
    "hood_slam": ((70.0, 1000.0), 0.7, 1.4, 90.0, 0.35),
}
DOOR_KINDS = (*DOORS, "car_door_open", "van_sliding_door")


def _slam(kind: str, seed: int, quality) -> np.ndarray:
    (lo, hi), q0, size, body, length = DOORS[kind]
    q = q0 if float(quality) < 0 else float(np.clip(quality, 0.0, 1.0))
    rng = np.random.default_rng(int(seed))
    n = _n(length + 0.05)
    nz = amb.hip(rng.uniform(-1, 1, n), 300.0)
    x = np.zeros(n)
    x += 0.5 * _mclick(nz, CATCH, 0.0)                                                      # the latch
    panel = K.impact(sr=P, material="steel", shape="rect_plate", size=size, velocity=2.5, hardness=0.5, damping=25.0, modes=20, dur=length, seed=int(seed))
    _put(x, _nm(np.asarray(panel, float)), 0.02, 1.1 - 0.8 * q)
    f = np.zeros(n)
    f[_n(0.02)] = 1.0
    _put(x, _nm(_modes(f, body * np.array([1.0, 1.52, 2.3]), 0.12, [1.0, 0.6, 0.35])), 0.0, 0.3 + 1.2 * q)   # the shell, B pillar
    y = _band(x, lo, hi) + 0.1 * x
    return y * np.exp(-np.arange(n) / P * 6.91 / length)


@sfx("car_door", "Car doors: car_door_close (latch, panel, body), car_door_rear, car_door_cheap (tinny), car_door_open (handle, seal), trunk_slam, hood_slam, van_sliding_door.",
     kind=("car_door_close", " | ".join(DOOR_KINDS)), quality=(-1.0, "0 tinny .. 1 solid (-1 = the kind's own)"), seed=(0, "variation"))
def car_door(sr=DEFAULT_SR, kind="car_door_close", quality=-1.0, seed=0):
    k = _kind(kind, DOOR_KINDS, "car_door")
    if k in DOORS:
        return _fin(_slam(k, seed, quality), sr, 0.8, fade=0.001, tail=0.02)
    rng = np.random.default_rng(int(seed))
    if k == "car_door_open":                                # handle click, latch release 60 ms later, the seal letting go
        n = _n(0.45)
        nz = amb.hip(rng.uniform(-1, 1, n), 300.0)
        y = 0.6 * _mclick(nz, CATCH, 0.0) + _mclick(nz, ENDSTOP, 0.06, 0.8)
        seal = amb.lop(rng.standard_normal(n), 900.0) * np.interp(np.arange(n), np.array([0.08, 0.13, 0.4]) * P, [0.0, 1.0, 0.0])
        return _fin(y + 0.5 * _nm(seal), sr, 0.7, fade=0.001, tail=0.03)
    # van: rollers on the rail for 0.9 s, then the stop, from `roll` and `impact`.
    run = 0.9
    y = np.zeros(_n(run + 0.5))
    r = K.roll(sr=P, dur=run, speed=0.5, radius=0.02, roughness=0.6, material="steel", surface="steel", surface_size=1.5, joints=0.2, seed=seed)
    _put(y, _nm(np.asarray(r, float)) * np.interp(np.arange(len(r)), [0, _n(0.1), len(r) - _n(0.1), len(r)], [0.0, 1.0, 1.0, 0.5]), 0.0, 0.2)
    _put(y, _slam("car_door_close", seed, 0.6), run, 1.0)
    return _fin(y, sr, 0.8, fade=0.005, tail=0.02)


# =============================================================================================
# HORNS
# =============================================================================================
# Hz, horn tube radius m, buzz drive.
HORNS = {
    "train_horn_k5la": ((311.0, 370.0, 415.0, 494.0, 622.0), 0.06, 2.5),
    "train_horn_euro": ((370.0, 440.0), 0.05, 2.5),
    "truck_air_horn": ((150.0, 180.0), 0.06, 3.0),
    "air_horn_car": ((370.0, 415.0), 0.03, 2.5),
    "can": ((420.0, 490.0), 0.02, 4.0),
    "vuvuzela": ((235.0,), 0.03, 2.0),
}
# Mercedes-style pairs: 335/400; 500 with 405-420; 420-440 with 340-370 (midpoints taken)
CAR_PAIRS = ((335.0, 400.0), (500.0, 412.0), (430.0, 355.0))
HORN_KINDS = (*HORNS, "car_pairs")
VUV_PARTIAL = 465.0                                         # a partial, not the octave of 235


def _parse_freqs(freqs):
    if isinstance(freqs, str):
        return tuple(float(s) for s in freqs.replace(" ", ",").replace("/", ",").split(",") if s)
    return tuple(float(f) for f in np.atleast_1d(freqs))


def _reed(f: float, pressure: np.ndarray, drive: float, rng) -> np.ndarray:
    """A diaphragm reed: the pressure sets the swing and bends the pitch down when it fails."""
    n = len(pressure)
    ph = 2 * np.pi * np.cumsum(f * (0.97 + 0.03 * pressure) * (1.0 + 0.002 * np.sin(2 * np.pi * 5.3 * np.arange(n) / P))) / P + rng.uniform(0, 6.28)
    return np.tanh(drive * pressure * np.sin(ph) + 0.5 * pressure) - np.tanh(0.5 * pressure)


@sfx("air_horn", "Compressed-air horns: train_horn_k5la (311, 370, 415, 494, 622 Hz), train_horn_euro, truck_air_horn (150/180 Hz), air_horn_car (370/415 Hz), can, vuvuzela (235 Hz, partial 465), car_pairs (electric disc pairs).",
     kind=("truck_air_horn", " | ".join(HORN_KINDS)), freqs=(None, "override: Hz list, '285/330'"), pair=(0, "car_pairs: 0 335/400, 1 500/412, 2 430/355"), dur=(1.2, "s"), seed=(0, "variation"))
def air_horn(sr=DEFAULT_SR, kind="truck_air_horn", freqs=None, pair=0, dur=1.2, seed=0):
    """Each tone is a reed into its own half-wave `tube`. Rise 80 ms and the pressure sag at the end."""
    k = _kind(kind, HORN_KINDS, "air_horn")
    d = float(np.clip(dur, 0.2, 20.0))
    if k == "car_pairs":
        a, b = _parse_freqs(freqs) if freqs is not None else CAR_PAIRS[int(pair) % len(CAR_PAIRS)]
        return FO.car_horn(sr=sr, dur=d, freq=a, freq2=b)
    tones, radius, drive = HORNS[k]
    if freqs is not None:
        tones = _parse_freqs(freqs)
    rng = np.random.default_rng(int(seed))
    n = _n(d)
    pr = np.minimum(np.arange(n) / (0.08 * P), 1.0) * np.interp(np.arange(n), [n - _n(0.15), n], [1.0, 0.0])
    y = np.zeros(n)
    for f in tones:
        x = pr * _reed(f, pr, drive, rng)
        if k == "vuvuzela":
            x = x + 0.7 * pr * np.sin(2 * np.pi * VUV_PARTIAL * np.arange(n) / P)
        y += np.asarray(tube(x, P, length_m=C_AIR / (2.0 * f), radius_m=radius, ends="open-open", mix=0.7), float)[:n] / math.sqrt(len(tones))
    return _fin(y, sr, 0.8, fade=0.005, tail=0.03)


CLAMPED = (1.0, 2.081, 3.414, 3.893, 4.995, 5.954)        # clamped circular plate: lambda^2 = 10.22, 21.26, 34.88, 39.77, 51.03, 60.83


def klaxon_rate(n: int, kind: str = "electric", crank: float = 1.0, seed: int = 0) -> np.ndarray:
    """Tooth impacts per second, 100-500 Hz, up then down.
    hand_crank: slower to speed, uneven within each turn of the crank (3 per s)."""
    top = 100.0 + 400.0 * float(np.clip(crank, 0.0, 1.0))
    t = np.linspace(0.0, 1.0, n)
    if kind == "electric":
        return 100.0 + (top - 100.0) * np.interp(t, [0.0, 0.35, 0.65, 1.0], [0.0, 1.0, 1.0, 0.0]) ** 1.5
    shape = np.interp(t, [0.0, 0.55, 0.8, 1.0], [0.0, 1.0, 1.0, 0.25])
    return np.maximum(100.0 + (top - 100.0) * shape * (1.0 + 0.1 * np.sin(2 * np.pi * 3.0 * t * n / P + (seed % 7))), 100.0)


@sfx("klaxon", "A-ooo-ga: a toothed wheel hits a rivet on a steel membrane, the wheel up to speed and down again, through a horn. kind electric (motor) | hand_crank.",
     kind=("electric", "electric | hand_crank"), crank=(1.0, "0..1 top speed: 100..500 impacts/s"), teeth=(8, "wheel teeth, for the shaft rate shown in the wheel run-out"), dur=(1.2, "s"), seed=(0, "variation"))
def klaxon(sr=DEFAULT_SR, kind="electric", crank=1.0, teeth=8, dur=1.2, seed=0):
    """Wheel, rivet and clamped steel membrane, its fundamental at 620 Hz, into a short horn."""
    if kind not in ("electric", "hand_crank"):
        raise ValueError(f"klaxon kind {kind!r}: use electric or hand_crank")
    rng = np.random.default_rng(int(seed))
    n = _n(float(np.clip(dur, 0.3, 10.0)))
    rate = klaxon_rate(n, kind, crank, int(seed))
    ph = np.cumsum(rate) / P
    tooth = np.flatnonzero(np.diff(np.floor(ph), prepend=0.0))
    f = np.zeros(n)
    z = max(int(teeth), 1)
    f[tooth] = 1.0 + 0.15 * np.sin(2 * np.pi * np.arange(len(tooth)) / z + rng.uniform(0, 6.28))     # the wheel's run-out
    m = _modes(f, 620.0 * np.array(CLAMPED), 0.03, [1.0, 0.7, 0.5, 0.4, 0.3, 0.2])
    y = np.asarray(tube(m, P, length_m=0.3, radius_m=0.04, ends="closed-open", mix=0.7), float)[:n]
    return _fin(y * np.interp(np.arange(n), [0, _n(0.03), n - _n(0.08), n], [0.0, 1.0, 1.0, 0.0]), sr, 0.8, fade=0.005, tail=0.03)


@effect("air_speaker", "Compressed-air loudspeaker (Pathé, Auxetophone): the signal opens a valve on an air supply, the flow puffs into a horn. Loud, thin, and the valve shuts on the troughs when driven hard.",
        depth=(1.2, "valve swing: under 1 clean, over 1 the valve shuts on the troughs"), horn=(300.0, "Hz, horn cut-off"), hiss=(0.05, "flow noise, follows the opening"), seed=(0, "variation"))
def air_speaker(x, sr=DEFAULT_SR, depth=1.2, horn=300.0, hiss=0.05, seed=0):
    """Open area 1 + depth x, never below 0; flow at constant supply pressure follows the area; the pressure radiated is its
    derivative (a monopole), into a horn that passes nothing under its cut-off (second-order high-pass). Same peak as the input."""
    rng = np.random.default_rng(int(seed))
    fc = float(np.clip(horn, 50.0, 0.2 * sr))
    hp = butter(2, fc, "highpass", fs=sr, output="sos")

    def one(s):
        pk = float(np.max(np.abs(s))) + 1e-12
        u = np.maximum(1.0 + float(depth) * s / pk, 0.0)
        p = sosfilt(hp, np.diff(u, prepend=u[0]) * sr / (2.0 * np.pi * fc) + float(hiss) * u * rng.standard_normal(len(u)))
        return p * pk / (float(np.max(np.abs(p))) + 1e-12)

    return _per_channel(one, np.asarray(x, float))


@sfx("backup_beeper", "Reversing alarm: a piezo tone with odd harmonics, gated. 1000 Hz is the usual pitch, 1250 Hz dominant in one survey, 700-2800 Hz the SAE J994 range.",
     freq=(1000.0, "Hz, 700..2800"), rate=(1.0, "beeps per second"), duty=(0.5, "on fraction"), dur=(3.0, "s"))
def backup_beeper(sr=DEFAULT_SR, freq=1000.0, rate=1.0, duty=0.5, dur=3.0):
    """~1000 Hz, 97-112 dB; 1250 Hz; 700-2800 Hz (SAE J994). One beep a second, half on, odd harmonics, 5 ms gate edges."""
    f = float(np.clip(freq, 700.0, 2800.0))
    n = _n(float(np.clip(dur, 0.1, 60.0)))
    t = np.arange(n) / P
    r, w = float(np.clip(rate, 0.1, 20.0)), float(np.clip(duty, 0.05, 0.95))
    ph = (t * r) % 1.0
    edge = 0.005 * r
    gate = np.clip(ph / edge, 0, 1) * np.clip((w - ph) / edge, 0, 1)
    y = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(6 * np.pi * f * t) + 0.1 * np.sin(10 * np.pi * f * t)
    return _fin(y * gate, sr, 0.7, fade=0.002)


def _alarm_motif(i: int, n: int) -> np.ndarray:
    """Hz against time, between 1 and 3 kHz: six figures."""
    t = np.arange(n) / P
    saw = lambda r: (t * r) % 1.0
    sq = lambda r: (t * r) % 1.0 < 0.5
    return [1000.0 + 2000.0 * saw(1.5),                     # rising whoop
            np.where(sq(3.0), 1200.0, 2200.0),              # hi-lo
            1000.0 + 2000.0 * saw(8.0),                     # fast yelp
            2000.0 + 1000.0 * np.sin(2 * np.pi * 0.7 * t),  # slow wail
            np.where(sq(10.0), 2800.0, 1500.0),             # warble
            3000.0 - 2000.0 * saw(4.0)][i % 6]              # falling stutter


@sfx("car_alarm", "Car alarm: a piezo siren stepping through six figures, 1-3 kHz, 3-5 s each.",
     seq=("012345", "figures in order, digits 0..5"), motif=(4.0, "s per figure, 3..5"))
def car_alarm(sr=DEFAULT_SR, seq="012345", motif=4.0):
    """Six motifs, 1-3 kHz, 3-5 s."""
    digits = [int(c) % 6 for c in str(seq) if c.isdigit()] or [0]
    m = _n(float(np.clip(motif, 3.0, 5.0)))
    out = np.zeros(m * len(digits))
    for j, d in enumerate(digits):
        f = _alarm_motif(d, m)
        s = np.tanh(2.0 * np.sin(2 * np.pi * np.cumsum(f) / P))
        s[:_n(0.005)] *= np.linspace(0, 1, _n(0.005))
        s[-_n(0.005):] *= np.linspace(1, 0, _n(0.005))
        out[j * m:(j + 1) * m] = s
    return _fin(_band(out, 800.0, 5000.0), sr, 0.7, fade=0.005)


# =============================================================================================
# CABIN MECHANICS AND SUSPENSION
# =============================================================================================
MECH = ("ceinture", "frein_a_main", "boite", "siege", "vitre")


@sfx("vehicle_mech", "Cabin mechanics: ceinture (webbing, retractor ratchet, buckle), frein_a_main (ratchet, then the button), boite (gate and detent), siege (motor on rails), vitre (12 V window motor, seal, stop).",
     kind=("ceinture", " | ".join(MECH)), seed=(0, "variation"))
def vehicle_mech(sr=DEFAULT_SR, kind="ceinture", seed=0):
    """Solenoid, motor and clicks (`servo`, `solenoid`, `click`, `electric_motor`), a 12 V motor for the window."""
    k = _kind(kind, MECH, "vehicle_mech")
    rng = np.random.default_rng(int(seed))
    if k == "ceinture":
        y = np.zeros(_n(1.0))
        web = amb.bp(rng.standard_normal(_n(0.55)), 2200.0, 0.8) * np.interp(np.arange(_n(0.55)), [0, _n(0.1), _n(0.5), _n(0.55)], [0, 1, 0.6, 0])
        _put(y, _nm(web), 0.0, 0.35)
        _put(y, FO.ratchet(sr=P, clicks=10, rate=40.0, size=0.5, mount="plastic", seed=seed), 0.05, 0.3)
        _put(y, solenoid(sr=P, size=0.1, metal=0.7, thump=0.1, bounce=0.2, seed=seed + 1), 0.72, 1.0)     # the buckle
    elif k == "frein_a_main":
        y = np.zeros(_n(1.1))
        _put(y, FO.ratchet(sr=P, clicks=7, rate=13.0, slow=0.5, size=1.6, mount="metal", seed=seed), 0.0, 1.0)
        _put(y, solenoid(sr=P, size=0.2, metal=0.4, thump=0.4, bounce=0.1, seed=seed + 1), 0.85, 0.6)
    elif k == "boite":
        y = np.zeros(_n(0.7))
        nz = amb.hip(rng.uniform(-1, 1, len(y)), 300.0)
        y += _band(0.8 * _mclick(nz, ENDSTOP, 0.02) + 0.5 * _mclick(nz, RATCHET, 0.2), 120.0, 6000.0)
        rub = amb.bp(rng.standard_normal(len(y)), 600.0, 1.0) * np.interp(np.arange(len(y)), [0, _n(0.1), _n(0.4), _n(0.5)], [0, 1, 1, 0])
        y += 0.15 * _nm(rub)
        _put(y, solenoid(sr=P, size=0.5, metal=0.6, thump=0.7, bounce=0.3, seed=seed), 0.42, 1.0)         # into the gate
    elif k == "siege":
        y = np.zeros(_n(1.7))
        _put(y, FO.servo(sr=P, moves=1, dur=1.4, speed=95.0, seed=seed), 0.0, 1.0)
        _put(y, solenoid(sr=P, size=0.25, metal=0.3, thump=0.5, bounce=0.1, seed=seed + 1), 1.42, 0.8)
    else:
        d = 1.8
        n = _n(d + 0.3)
        y = np.zeros(n)
        mot = electric_motor(sr=P, dur=d, rpm=3000.0, load=0.7, startup=0.1, seed=seed)
        _put(y, amb.lop(_nm(np.asarray(mot, float)), 2500.0) * np.interp(np.arange(_n(d)), [_n(d) - _n(0.05), _n(d)], [1.0, 0.0])[:len(mot)], 0.0, 0.8)
        seal = amb.bp(rng.standard_normal(_n(d)), 1200.0, 0.6) * (0.5 + 0.5 * np.abs(np.sin(np.arange(_n(d)) * 2 * np.pi * 1.3 / P)))
        _put(y, _nm(seal), 0.0, 0.1)
        _put(y, solenoid(sr=P, size=0.4, metal=0.2, thump=0.6, bounce=0.1, seed=seed), d - 0.02, 0.9)     # glass into the top seal
    return _fin(y, sr, 0.7, fade=0.002, tail=0.01)


@sfx("suspension", "Car springs and dampers over bumps: bushes creak and squeak, stops knock. load raises the strain.",
     load=(0.5, "0 empty .. 1 full"), bumps=(3, "count"), dur=(1.6, "s"), seed=(0, "variation"))
def suspension(sr=DEFAULT_SR, load=0.5, bumps=3, dur=1.6, seed=0):
    """Creaks and squeaks of the bushes, knocks on the stops (`squeak`, `creak`, `thud`)."""
    ld = float(np.clip(load, 0.0, 1.0))
    d = float(np.clip(dur, 0.3, 30.0))
    rng = np.random.default_rng(int(seed))
    y = np.zeros(_n(d))
    y += 0.5 * _nm(_fit(np.asarray(FO.creak(sr=P, material="iron", force="sway", level=0.5 + 0.5 * ld, dur=d, seed=seed), float), len(y)))
    y += 0.3 * _nm(_fit(np.asarray(squeak(sr=P, surface="hinge", pressure=0.6 + 2.0 * ld, pitch=1.3 - 0.5 * ld, dur=d, seed=seed), float), len(y)))
    for t in np.sort(rng.uniform(0.0, max(d - 0.3, 0.05), int(np.clip(bumps, 1, 40)))):
        _put(y, FO.thud(sr=P, body="box", size=1.0 + ld, force=0.5 + 0.5 * ld, seed=int(rng.integers(1 << 30))), float(t), 0.8)
    return _fin(y, sr, 0.7, fade=0.005, tail=0.03)


CAR_WIND = dict(pillar_d=0.006, st=0.2, pillar_band=(200.0, 6000.0), pillar_q=15.0, lee=0.45, detune=1.03,
                cabin_v=2.7, neck=0.05, gap=0.006, window=0.05, buffet_q=8.0, buffet_gain=2.5,
                cabin_db=-9.0, cabin_lp=900.0, direct=0.6)


def cabin_helmholtz(open_: float, c: float = C_AIR) -> float:
    """Buffeting: the cabin as a Helmholtz resonator, f0 = c / 2 pi sqrt(A / (V l)), A = 0.006 + 0.05 open^2."""
    w = CAR_WIND
    a = w["gap"] + w["window"] * float(open_) ** 2
    return c / (2.0 * math.pi) * math.sqrt(a / (w["cabin_v"] * w["neck"]))


@sfx("car_wind", "Wind on a moving car: boundary-layer roar, the A-pillar vortex tone (f = 0.2 v / 6 mm) and, window open, the cabin booming at its Helmholtz note (17-35 Hz).",
     speed=(110.0, "km/h"), sideslip=(0.0, "rad of crosswind: the pillar tone grows with |beta|"), window=(0.0, "0..1 window opening"),
     view=("cabin", "cabin | outside"), dur=(3.0, "s"), seed=(0, "variation"))
def car_wind(sr=DEFAULT_SR, speed=110.0, sideslip=0.0, window=0.0, view="cabin", dur=3.0, seed=0):
    """Boundary layer: pink noise, low-passed at 250 + 30 v, level (v / 40)^2.5; pillar: band-pass Q 15 at 0.2 v / d,
    level s^3 (0.3 + 3 |beta|) / 4 (s = v / 50 m/s), the leeward pillar 0.45 and 3 % up; buffeting: Q 8 at the cabin
    Helmholtz note, s^2 open 2.5. Cabin: -9 dB and 900 Hz low-pass, plus the direct sound through the gap, open 0.6."""
    if view not in ("cabin", "outside"):
        raise ValueError("car_wind view must be cabin or outside")
    w = CAR_WIND
    rng = np.random.default_rng(int(seed))
    n = _n(float(np.clip(dur, 0.2, 60.0)))
    v = float(np.clip(speed, 5.0, 300.0)) / 3.6
    s, op, beta = v / 50.0, float(np.clip(window, 0.0, 1.0)), abs(float(sideslip))
    bl = amb.lop(amb.pink(n, rng), 250.0 + 30.0 * v) * (v / 40.0) ** 2.5
    fp = float(np.clip(w["st"] * v / w["pillar_d"], *w["pillar_band"]))
    nz = rng.standard_normal(n)
    pil = (amb.bp(nz, fp, w["pillar_q"]) + w["lee"] * amb.bp(rng.standard_normal(n), fp * w["detune"], w["pillar_q"]))
    pil *= s ** 3 * (0.3 + 3.0 * beta) * 0.25
    y = bl + pil
    if op > 0:
        buf = amb.bp(amb.lop(rng.standard_normal(n), 40.0), cabin_helmholtz(op), w["buffet_q"]) * s ** 2 * op * w["buffet_gain"]
        y = y + buf
    if view == "cabin":
        y = 10 ** (w["cabin_db"] / 20) * amb.lop(y, w["cabin_lp"]) + op * w["direct"] * (bl + pil)
    return _fin(y, sr, 0.7, fade=0.2, tail=0.2)
