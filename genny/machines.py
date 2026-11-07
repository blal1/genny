"""Machines and tools: power tools, lifts, rides, pumps, medical beeps, factories, the belt that squeals, Savart's wheel.

Spec with its source tags: reports/sons/taxonomie_universelle.md lines 253-391, Plan sons L15 and the book notes it cites.
- Capstan T1/T2 = e^(mu phi), full slip at mu .5, phi pi: 4.81 and M/2RT0 = 0.656; film speed P = 2.45 (V1+V2) R eta / h0;
  tooth sliding (w1+w2) OP; crater reload 1/R - 1/rho.
- Savart's wheel: the reed follows the imposed rate whatever its own frequency; f = teeth rpm / 60.
- Saw: blade passing = rpm teeth / 60. Mains membrane at twice the line.
The source of every constant is in research/SOURCES.md, section machines. Nothing here has been listened to: see tests/test_machines.py for what was measured.
"""
from __future__ import annotations

import math

import numpy as np

from . import contact as K
from . import filters as F
from . import foley as FO
from .core import DEFAULT_SR
from .creatures import breath, crowd_murmur, footsteps
from .cyber import _impacts, _ramp, bearing, gearbox, transformer
from .dsp import noise_bed
from .foley import CATCH, ENDSTOP, P, RATCHET, _fin, _mclick, _motor as _farnell_motor, _n, _nm, _put
from .friction import FS, _rub_engine
from .klang import ui_chime
from .matter import air_leak, arc, bubbles
from .physical import ambience as amb
from .pinball import solenoid, spinner
from .road import _band, _fit, _kind, _modes
from .sfx import REGISTRY, coin, electric_motor, fan, gears, sfx


def _m(y, n: int | None = None) -> np.ndarray:
    y = np.asarray(y, float)
    y = y.mean(axis=1) if y.ndim == 2 else y
    return y if n is None else _fit(y, n)


def _motor(rpm: float, dur: float, load: float = 0.5, startup: float = 0.0, seed: int = 0) -> np.ndarray:
    return _nm(_m(electric_motor(sr=P, dur=dur, rpm=rpm, load=load, startup=startup, seed=seed), _n(dur)))


def _rumble(d: float, lo: float, hi: float, seed: int) -> np.ndarray:
    """`contact.rumble` held to its band: it has stray clicks above it (measured, 2 in 3 s)."""
    return _nm(amb.lop(_m(K.rumble(sr=P, dur=d, fmin=lo, fmax=hi, seed=seed), _n(d)), 2.5 * hi))


def _clicks(n: int, parts, times, rng, gain: float = 1.0) -> np.ndarray:
    return gain * _mclick(amb.hip(rng.uniform(-1, 1, n), 300.0), parts, [t for t in np.atleast_1d(times) if t * P < n])


def _beats(rate: float, dur: float, at: float = 0.0) -> np.ndarray:
    """Times of a steady event ``rate`` per second."""
    return at + np.arange(int((dur - at) * rate) + 1) / rate


def _dur(d, lo=0.1, hi=60.0) -> float:
    return float(np.clip(d, lo, hi))


# =============================================================================================
# BOOK MECHANICS
# =============================================================================================
def capstan_ratio(mu: float, wrap: float) -> float:
    """T1/T2 at full slip."""
    return math.exp(float(mu) * float(wrap))


def slip_torque(mu: float, wrap: float) -> float:
    """M/2RT0 at which the sliding arc reaches the wrapped arc. mu .5, wrap pi: 0.656."""
    r = capstan_ratio(mu, wrap)
    return (r - 1.0) / (r + 1.0)


def stribeck_speed(eta: float, radius: float, load: float, h0: float = 1e-6) -> float:
    """Speed V1+V2 at which the film of a rigid cylinder pair reaches the roughness h0: V = P h0 / (2.45 R eta).
    eta Pa s, radius m, load N per m. R .1, P 1000, eta .1, 1 um: 0.041 m/s. For `friction.stick_slip`, which still has 0.02 fixed."""
    return float(load) * float(h0) / (2.45 * float(radius) * float(eta))


def sliding_speed(rpm: float, teeth: int, teeth2: int, op):
    """Sliding on a tooth flank: (w1 + w2) OP, signed, zero at the pitch point. ``op``: m from the pitch point."""
    w1 = 2.0 * math.pi * float(rpm) / 60.0
    return (w1 + w1 * teeth / teeth2) * np.asarray(op, float)


def tooth_friction(rpm, teeth: int, n: int, load: float = 0.6, mu: float = 0.1, seed: int = 0, top: float | None = None) -> np.ndarray:
    """Friction force of a mesh, per unit tooth load, at P: a sawtooth at teeth rpm / 60 that is nil at the pitch point and changes sign
    there, amplitude mu load (OP spans -1..1 of the path). Band-limited; ``top`` Hz: heard through a housing,
    the jump at the tooth change rounded off (Fejer taper). `cyber.gearbox` has no such term: see `_geared`."""
    f = np.broadcast_to(np.asarray(rpm, float), (n,)) / 60.0 * int(teeth)
    ph = np.cumsum(f) / P + np.random.default_rng(int(seed)).uniform()
    y = np.zeros(n)
    K_ = int((top or 0.45 * P) / max(float(f.max()), 1.0))
    for k in range(1, K_ + 1):
        y -= (1.0 - k / (K_ + 1.0) if top else 1.0) * np.sin(2 * np.pi * k * ph) / k
    return float(mu) * float(load) * 2.0 / math.pi * y


def _geared(rpm: float, teeth: int, teeth2: int, load: float, dur: float, seed: int, housing: float = 1800.0, wear: float = 0.2) -> np.ndarray:
    """`gearbox` plus the tooth friction it lacks."""
    n = _n(dur)
    g = _m(gearbox(sr=P, dur=dur, rpm=rpm, teeth=teeth, teeth2=teeth2, load=load, wear=wear, backlash=0.5 * wear, housing=housing, seed=seed), n)
    return _nm(g) + 2.0 * tooth_friction(rpm, teeth, n, load, seed=seed, top=3000.0)


def servo_env(n: int) -> np.ndarray:
    """Speed of a proportional servo: sharp start and stop, no overspin, unlike `foley._move`."""
    x = np.arange(n) / max(n, 1)
    return np.minimum(1.0 - np.exp(-x / 0.015), np.clip((1.0 - x) / 0.05, 0.0, 1.0))


def forge_contact_time(mass: float, velocity: float, radius: float, estar: float) -> float:
    """Hertz impact duration, 2.94 alpha / V."""
    return 2.94 * (15.0 * mass * velocity ** 2 / (16.0 * estar * math.sqrt(radius))) ** 0.4 / velocity


def crater_radius(radius: float, rho: float) -> float:
    """Relative radius of an elastic reload in a crater of radius rho: 1/R - 1/rho. Wider, stiffer, so shorter by (R'/R)^(-1/5)."""
    return 1.0 / (1.0 / radius - 1.0 / rho)


# tup 200 kg at 5 m/s, face R .3 m, steel E* 115 GPa, crater .45 m, plastic blow 1.6 times as long, its impulse .7 of an elastic one (no rebound)
FORGE = (200.0, 5.0, 0.3, 1.15e11, 0.45, 1.6, 0.7)


def forge_times() -> tuple[float, float]:
    """Contact time of the first blow (plastic, makes the crater), and of every next one (elastic)."""
    m, v, r, e, rho, slow, _ = FORGE
    return forge_contact_time(m, v, r, e) * slow, forge_contact_time(m, v, crater_radius(r, rho), e)


def _forge_force(n: int, times) -> np.ndarray:
    tc1, tc2 = forge_times()
    f = np.zeros(n)
    for j, t in enumerate(times):
        tc = tc1 if j == 0 else tc2
        k = max(2, int(tc * P))
        i = int(t * P)
        if i + k < n:
            f[i:i + k] += (FORGE[6] if j == 0 else 1.0) * np.sin(np.pi * np.arange(k) / k) * tc2 / tc     # same blow, spread over Tc
    return f


# =============================================================================================
# POWER TOOLS
# =============================================================================================
TOOL_RPM = {"ponceuse": 9000.0, "disqueuse": 11000.0, "scie_circulaire": 5000.0, "tournevis": 3000.0}
TOOL_KINDS = ("ponceuse", "disqueuse", "scie_circulaire", "cloueuse", "tournevis", "soudeur")


def saw_bpf(rpm: float, teeth: int) -> float:
    """Blade passing frequency, Hz."""
    return float(rpm) * int(teeth) / 60.0


def _ponceuse(n, d, rpm, load, rng, seed):
    t = np.arange(n) / P
    sc = _m(K.scrape(sr=P, dur=d, speed=1.2, force=1.0, roughness=0.9, material="wood", size=0.5, seed=seed), n)
    return 0.6 * _motor(rpm, d, load, 0.1, seed) + load * _nm(amb.hip(sc, 500.0)) * (0.75 + 0.25 * np.sin(2 * np.pi * rpm / 60.0 * t))      # the orbit


def _disqueuse(n, d, rpm, load, rng, seed):
    bevel = _geared(rpm, 10, 37, load, d, seed, 2400.0)
    sc = _m(K.scrape(sr=P, dur=d, speed=3.0, force=1.0, roughness=1.0, material="steel", size=0.3, seed=seed), n)
    return 0.5 * _motor(rpm, d, load, 0.15, seed) + 0.6 * _nm(bevel) + load * _nm(amb.hip(sc, 2000.0))


def _scie(n, d, rpm, teeth, load, rng, seed):
    p = np.cumsum(np.full(n, rpm / 60.0)) / P
    bpf = saw_bpf(rpm, teeth)
    tone = sum(np.sin(2 * np.pi * k * teeth * p) / k for k in range(1, 4) if k * bpf < 0.45 * P)
    bite = _impacts(n, teeth * p, rng, slip=0.03)
    sc = _m(K.scrape(sr=P, dur=d, speed=2.0, force=1.0, roughness=0.8, material="wood", size=0.5, seed=seed), n)
    return 0.4 * _motor(rpm, d, load, 0.1, seed) + 0.5 * tone * (0.5 + 0.5 * load) + load * (0.5 * _nm(sc) + 0.3 * _nm(bite))


def _cloueuse(shots, rate, rng, seed):
    y = np.zeros(_n(shots / rate + 0.4))
    for j in range(shots):
        t = j / rate
        _put(y, FO.piston(sr=P, cycles=1, rate=8.0, pressure=0.9, seed=seed + j), t, 0.3)
        _put(y, K.impact(sr=P, material="steel", shape="bar", size=0.08, velocity=8.0, hardness=0.9, modes=10, dur=0.1, seed=seed + j), t + 0.008, 1.0)
        _put(y, air_leak(sr=P, dur=0.3, pressure=0.8, orifice=6.0, burst=0.06, seed=seed + j), t + 0.03, 0.5)
    return y


def _tournevis(n, d, rpm, rng, seed):
    y = 0.6 * _nm(_m(FO.drill(sr=P, dur=d, rpm=rpm, load=0.4, material="none", startup=0.1, seed=seed), n))
    at = 0.65 * d                                           # the clutch lets go at the set torque
    y[_n(at):] *= 0.5
    _put(y, FO.ratchet(sr=P, clicks=int((d - at) * 30.0), rate=30.0, size=0.8, mount="plastic", seed=seed), at, 0.8)
    return y


def _soudeur(n, d, rng, seed):
    a = _m(arc(sr=P, dur=d, mains=50.0, stability=0.7, seed=seed), n)
    at = np.sort(rng.uniform(0.0, d, int(40 * d)))
    return _nm(a) + 0.35 * _nm(_clicks(n, CATCH, at, rng))  # the spatter


@sfx("power_tool", "Power tools, motor plus material: ponceuse (sander), disqueuse (angle grinder), scie circulaire (blade passing at rpm x teeth / 60), cloueuse (nailer), tournevis (clutch), soudeur (arc welder).",
     kind=("scie_circulaire", " | ".join(TOOL_KINDS)), rpm=(-1.0, "motor rpm, 3000..11000 (-1 = the kind's own)"), teeth=(24, "saw blade teeth"),
     load=(0.6, "0 free-running .. 1 biting hard"), shots=(2, "nailer: nails fired"), dur=(2.0, "s"), seed=(0, "variation"))
def power_tool(sr=DEFAULT_SR, kind="scie_circulaire", rpm=-1.0, teeth=24, load=0.6, shots=2, dur=2.0, seed=0):
    k = _kind(kind, TOOL_KINDS, "power_tool")
    d, ld = _dur(dur, 0.3), float(np.clip(load, 0.0, 1.0))
    n, rng, seed = _n(d), np.random.default_rng(int(seed)), int(seed)
    r = float(np.clip(rpm, 3000.0, 11000.0)) if float(rpm) > 0 else TOOL_RPM.get(k, 0.0)
    if k == "ponceuse":
        y = _ponceuse(n, d, r, ld, rng, seed)
    elif k == "disqueuse":
        y = _disqueuse(n, d, r, ld, rng, seed)
    elif k == "scie_circulaire":
        y = _scie(n, d, r, int(np.clip(teeth, 2, 120)), ld, rng, seed)
    elif k == "cloueuse":
        return _fin(_cloueuse(int(np.clip(shots, 1, 20)), 2.0, rng, seed), sr, 0.8, fade=0.001, tail=0.05)
    elif k == "tournevis":
        y = _tournevis(n, d, r, rng, seed)
    else:
        y = _soudeur(n, d, rng, seed)
    return _fin(y, sr, 0.7, fade=0.01, tail=0.05)


@sfx("jackhammer", "Pneumatic breaker: a chisel struck 20-25 times a second with the exhaust puffing at the same rate.",
     rate=(22.5, "blows per second, 20..25 (10..40 with model rod)"), pressure=(0.8, "0..1 air"), dur=(2.0, "s"), seed=(0, "variation"),
     model=("impact", "impact (a struck steel bar and the exhaust) | rod (each blow a short Hann pulse down the tool: a "
                      "longitudinal comb in the chisel and a bending cascade)"))
def jackhammer(sr=DEFAULT_SR, rate=22.5, pressure=0.8, dur=2.0, seed=0, model="impact"):
    """`piston` stops at 16 strokes/s, so the exhaust is `air_leak` gated per blow, as `piston` does."""
    if model == "rod":
        return _jackhammer_rod(sr, float(np.clip(rate, 10.0, 40.0)), float(np.clip(pressure, 0.0, 1.0)), _dur(dur, 0.2), int(seed))
    if model != "impact":
        raise ValueError("jackhammer model must be impact or rod")
    r = float(np.clip(rate, 20.0, 25.0))
    d, rng = _dur(dur, 0.2), np.random.default_rng(int(seed))
    n = _n(d)
    hit = _m(K.impact(sr=P, material="steel", shape="bar", size=0.45, velocity=6.0, hardness=0.9, modes=16, dur=0.035, seed=int(seed)))
    f = np.zeros(n)
    i = (_beats(r, d - 0.05) * P).astype(int)
    f[i] = rng.uniform(0.92, 1.08, len(i))
    x = np.convolve(f, _nm(hit))[:n]
    ph = (np.arange(n) / P * r) % 1.0
    air = _nm(_m(air_leak(sr=P, dur=d, pressure=float(np.clip(pressure, 0.0, 1.0)), orifice=8.0, seed=int(seed)), n)) * np.exp(-ph / (r * 0.012))
    body = _rumble(d, 60.0, 200.0, int(seed))
    return _fin(_nm(x) + 0.4 * air + 0.25 * body * np.exp(-ph / (r * 0.02)), sr, 0.8, fade=0.003, tail=0.02)


JACK_ROD = dict(pulse=(7, 11), length=0.5, c=5100.0, comb=0.8, bend=(820.0, 2300.0, 4500.0, 7400.0), q=12.0)


def _jackhammer_rod(sr, r, pressure, d, seed):
    """Each blow a Hann pulse of 7-11 samples at 44.1 kHz into two branches: the longitudinal wave bouncing in a 0.5 m steel
    chisel (comb of round trip 2 L / c, feedback 0.8) and the bending modes in a cascade of resonators; the air puffs as in
    the impact model."""
    rng = np.random.default_rng(seed)
    n = _n(d)
    f = np.zeros(n)
    for t in _beats(r, d - 0.05):
        w = int(round(rng.integers(JACK_ROD["pulse"][0], JACK_ROD["pulse"][1] + 1) * P / 44100.0))
        i = int(t * P)
        f[i:i + w] += rng.uniform(0.9, 1.1) * np.hanning(w + 2)[1:-1][:max(n - i, 0)]
    m = max(int(2 * JACK_ROD["length"] / JACK_ROD["c"] * P), 1)
    lon = np.copy(f)
    for k in range(m, n):
        lon[k] += JACK_ROD["comb"] * lon[k - m]
    tr = f
    for fb in JACK_ROD["bend"]:
        tr = tr + F.bandpass(tr, fb, P, JACK_ROD["q"])
    ph = (np.arange(n) / P * r) % 1.0
    air = _nm(_m(air_leak(sr=P, dur=d, pressure=pressure, orifice=8.0, seed=seed), n)) * np.exp(-ph / (r * 0.012))
    return _fin(_nm(lon) + 0.7 * _nm(tr) + 0.4 * air, sr, 0.8, fade=0.003, tail=0.02)


@sfx("dental_drill", "Air turbine handpiece: a thin whine at rpm / 60 (5 kHz at 300 000 rpm), the air, and the burr on enamel when load is set.",
     rpm=(300000.0, "turbine rpm"), load=(0.0, "0 free in the air .. 1 on the tooth, the speed sags"), dur=(2.0, "s"), seed=(0, "variation"))
def dental_drill(sr=DEFAULT_SR, rpm=300000.0, load=0.0, dur=2.0, seed=0):
    """`cyber.turbo` clamps at 200 000 rpm under a 5.5 kHz low-pass, `foley.drill` at 30 000, so the
    whine law of `turbo` (0.22 sin p + 0.1 sin 2p) is written out here."""
    d, ld = _dur(dur, 0.2), float(np.clip(load, 0.0, 1.0))
    n, rng, t = _n(d), np.random.default_rng(int(seed)), np.arange(_n(d)) / P
    f = float(np.clip(rpm, 3e4, 5e5)) / 60.0 * (1.0 - 0.3 * ld) * (1.0 - np.exp(-t / 0.15))
    ph = 2 * np.pi * np.cumsum(f) / P
    y = (0.22 * np.sin(ph) + 0.1 * np.where(2 * f < 0.45 * P, np.sin(2 * ph), 0.0)) * np.minimum(1.0, f / f.max())
    air = _nm(_m(air_leak(sr=P, dur=d, pressure=0.25, orifice=1.0, seed=int(seed)), n))
    y = y + 0.05 * air
    if ld > 0:
        y = y + 0.12 * ld * _nm(amb.bp(FO.fractal_noise(n, rng), 3200.0, 1.5))
    return _fin(y, sr, 0.5, fade=0.02, tail=0.1)


# =============================================================================================
# BUILDINGS
# =============================================================================================
ELEVATOR = ("montee", "arret", "porte", "carillon")


def _hoist(d: float, seed: int, up: float, down: float) -> np.ndarray:
    """Hoist motor at speed curve, cabin rumble, a rope creak."""
    n = _n(d)
    x = np.arange(n) / P
    env = np.clip(x / max(up, 1e-3), 0.0, 1.0) if up > 0 else np.ones(n)
    env = np.minimum(env, np.clip((d - x) / max(down, 1e-3), 0.0, 1.0)) if down > 0 else env
    rng = np.random.default_rng(int(seed))
    mot = _motor(1450.0, d, 0.6, up / 3.0, seed) if up > 0 else _motor(1450.0, d, 0.6, down / 3.0, seed)[::-1]     # a slowing motor is a spin-up played back
    rmb = _rumble(d, 30.0, 110.0, seed)
    sway = _nm(amb.bp(rng.standard_normal(n), 1400.0, 3.0)) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * x)) ** 4
    return (0.7 * _nm(mot) + 0.6 * rmb + 0.05 * sway) * np.sqrt(np.maximum(env, 0.0))


@sfx("elevator", "A lift: montee (motor up to speed, cabin hum), arret (slowing, brake landing), porte (servo door, rollers, stop), carillon (the floor bell).",
     kind=("montee", " | ".join(ELEVATOR)), dur=(3.0, "s, montee and arret"), seed=(0, "variation"))
def elevator(sr=DEFAULT_SR, kind="montee", dur=3.0, seed=0):
    """The door uses `servo_env`: a controlled operator, not a free motor."""
    k, seed = _kind(kind, ELEVATOR, "elevator"), int(seed)
    if k == "carillon":
        y = np.zeros(_n(1.4))
        _put(y, _m(ui_chime(sr=P, kind="discovery", pitch=1.0, seed=seed)), 0.0)
        return _fin(y, sr, 0.6, fade=0.001, tail=0.1)
    if k == "porte":
        d = 1.2
        y = np.zeros(_n(d + 0.3))
        rng = np.random.default_rng(seed)
        _put(y, _nm(_farnell_motor(120.0 * np.maximum(servo_env(_n(d)), 1e-3), rng)), 0.0, 0.35)
        _put(y, _nm(_m(K.roll(sr=P, dur=d, speed=0.4, radius=0.02, roughness=0.5, material="steel", surface="steel", surface_size=1.2, seed=seed))) * servo_env(_n(d)), 0.0, 0.3)
        _put(y, _m(solenoid(sr=P, size=0.6, metal=0.3, thump=0.7, bounce=0.2, seed=seed)), d - 0.01, 1.0)
        return _fin(y, sr, 0.7, fade=0.005, tail=0.03)
    d = _dur(dur, 1.0)
    if k == "montee":
        return _fin(_hoist(d, seed, 0.6 * d, 0.0), sr, 0.6, fade=0.02, tail=0.1)
    y = np.zeros(_n(d + 0.4))
    _put(y, _hoist(d, seed, 0.0, d), 0.0, 0.7)
    _put(y, _m(solenoid(sr=P, size=0.9, metal=0.2, thump=1.0, bounce=0.1, seed=seed)), d - 0.05, 1.0)     # the brake
    return _fin(y, sr, 0.7, fade=0.02, tail=0.05)


@sfx("escalator", "Moving stairs: drive motor, the chain, a step passing the comb every 0.4 m (1.25 a second at 0.5 m/s).",
     speed=(0.5, "m/s"), step=(0.4, "m per step"), dur=(4.0, "s"), seed=(0, "variation"))
def escalator(sr=DEFAULT_SR, speed=0.5, step=0.4, dur=4.0, seed=0):
    d, rng, seed = _dur(dur, 0.5), np.random.default_rng(int(seed)), int(seed)
    n = _n(d)
    rate = float(np.clip(speed, 0.1, 2.0)) / max(float(step), 0.05)
    mot = _motor(1450.0, d, 0.6, 0.0, seed)
    ch = _nm(_m(FO.chain(sr=P, dur=d, pushes=4, weight=0.8, density=40.0, settle=0, seed=seed), n))
    y = 0.12 * mot + 0.03 * ch + _nm(_clicks(n, ENDSTOP, _beats(rate, d - 0.1, 0.2), rng))
    return _fin(y, sr, 0.7, fade=0.05, tail=0.1)


HVAC = ("fan", "cool", "rattle", "box_fan")
FARNELL_FAN = dict(speed=22.0, motor_ratio=20.0, motor=0.2, brush=0.05, rotor=0.1, stator=0.3, width=4.0, noise=0.2, level=0.5,
                   blade_bp=700.0, ducts=((0.022, 0.499), (0.070, 0.3)), swing=12.0)


def box_fan(n, rng, speed=FARNELL_FAN["speed"]):
    """A fan in its housing: blade pulses 1 / (1 + k x^2) at the blade rate through a 700 Hz band-pass, noise on each pulse,
    the motor at 20 times the speed with its brushes, rotor and stator, all through two housing combs (22 ms, 0.499;
    70 ms, 0.3), the whole wobbling 12 Hz."""
    c = FARNELL_FAN
    t = np.arange(n) / P
    x = (t * speed) % 1.0 - 0.5
    blade = 1.0 / (1.0 + (c["width"] * 10.0 * x) ** 2)
    blade = F.bandpass(blade * (1.0 + c["noise"] * rng.standard_normal(n)), c["blade_bp"], P, 1.0) * c["level"]
    fm = speed * c["motor_ratio"]
    motor = c["motor"] * np.sin(2 * np.pi * fm * t) + c["rotor"] * np.sin(2 * np.pi * 2 * fm * t) \
        + c["stator"] * np.sin(2 * np.pi * 100.0 * t) + c["brush"] * rng.standard_normal(n)
    y = blade + motor
    for dl, fb in c["ducts"]:
        k = int(dl * P)
        z = np.copy(y)
        for i in range(k, n):
            z[i] += fb * z[i - k]
        y = z
    return y * (1.0 + 0.15 * np.sin(2 * np.pi * c["swing"] * t))


@sfx("hvac", "Air conditioning: fan (blade passing, duct bed), cool (with the compressor hum at twice the mains), rattle (and a loose panel).",
     state=("cool", " | ".join(HVAC)), rpm=(1800.0, "fan rpm"), blades=(5, "fan blades"), mains=(50.0, "Hz"), dur=(4.0, "s"), seed=(0, "variation"))
def hvac(sr=DEFAULT_SR, state="cool", rpm=1800.0, blades=5, mains=50.0, dur=4.0, seed=0):
    s = _kind(state, HVAC, "hvac")
    d, rng, seed = _dur(dur, 0.2), np.random.default_rng(int(seed)), int(seed)
    n = _n(d)
    if s == "box_fan":
        return _fin(_nm(box_fan(n, rng, float(rpm) / 60.0 * 22.0 / 30.0)), sr, 0.7, fade=0.2, tail=0.2)
    y = _nm(_m(fan(sr=P, dur=d, rpm=rpm, blades=blades, seed=seed), n)) + 0.2 * _nm(_m(noise_bed(sr=P, kind="hum", dur=d, freq=mains, seed=seed), n))
    if s != "fan":
        y = y + 0.6 * _nm(_m(transformer(sr=P, dur=d, mains=mains, flux=0.9, load=0.6, seed=seed), n))
    if s == "rattle":
        turn = _beats(float(rpm) / 60.0, d - 0.05)
        y = y + 2.0 * _nm(_clicks(n, CATCH, turn[rng.random(len(turn)) < 0.33], rng))
    return _fin(y, sr, 0.7, fade=0.2, tail=0.2)


CONVEYOR_JOINT = 1.5                                        # m of belt between splices
CONVEYOR_ROLLER = 0.05                                      # m roller radius


@sfx("conveyor", "Conveyor belt: rollers on bearings, a belt splice crossing them, geared drive motor.",
     speed=(1.0, "m/s"), joint=(CONVEYOR_JOINT, "m between belt splices"), dur=(4.0, "s"), seed=(0, "variation"))
def conveyor(sr=DEFAULT_SR, speed=1.0, joint=CONVEYOR_JOINT, dur=4.0, seed=0):
    """Splice rate v / joint, roller rpm from v / 2 pi r."""
    d, seed = _dur(dur, 0.3), int(seed)
    n, v = _n(d), float(np.clip(speed, 0.1, 6.0))
    L = max(float(joint), 0.1)
    roller = v / (2 * math.pi * CONVEYOR_ROLLER) * 60.0
    rl = _nm(amb.lop(_m(K.roll(sr=P, dur=d, speed=v, radius=CONVEYOR_ROLLER, roughness=0.5, material="rubber", surface="steel", surface_size=1.5, joints=L, seed=seed), n), 3000.0))
    splice = _nm(_clicks(n, ENDSTOP[:2], _beats(v / L, d - 0.1, 0.05), np.random.default_rng(seed)))     # `roll(joints=)` alone does not carry v / L in its envelope (measured)
    be = _nm(_m(bearing(sr=P, dur=d, rpm=roller, state="healthy", seed=seed), n))
    drive = _geared(1450.0, 20, 400, 0.6, d, seed, wear=0.0)
    return _fin(0.12 * rl + splice + 0.06 * be + 0.08 * _nm(drive) + 0.08 * _motor(1450.0, d, 0.6, 0.0, seed), sr, 0.7, fade=0.05, tail=0.1)


HDD = ("idle", "seek", "spinup")


def head_click(rng, ms: float = 2.0) -> np.ndarray:
    """One head movement, its energy gone in ``ms``."""
    x = _m(solenoid(sr=P, size=0.0, metal=0.6, thump=0.0, bounce=0.0, seed=int(rng.integers(1 << 30))))
    m = _n(ms * 1e-3)
    return _nm(x[:m] * np.exp(-np.arange(m) / m * 6.91))


@sfx("hard_drive", "Hard disk: spindle at rpm / 60 (120 Hz at 7200 rpm), its bearing whine, and the head ticking 2 ms per seek. state idle | seek | spinup.",
     state=("seek", " | ".join(HDD)), rpm=(7200.0, "spindle"), seeks=(14.0, "head movements per second"), dur=(2.0, "s"), seed=(0, "variation"))
def hard_drive(sr=DEFAULT_SR, state="seek", rpm=7200.0, seeks=14.0, dur=2.0, seed=0):
    s = _kind(state, HDD, "hard_drive")
    d, rng, seed = _dur(dur, 0.2), np.random.default_rng(int(seed)), int(seed)
    n = _n(d)
    y = 0.3 * _motor(float(rpm), d, 0.2, 0.15 * d if s == "spinup" else 0.0, seed)
    if s == "seek":
        k = head_click(rng)
        f = np.zeros(n)
        f[(rng.uniform(0.02, d - 0.02, max(1, int(float(seeks) * d))) * P).astype(int)] = 1.0
        y = y + np.convolve(f, k)[:n]
    return _fin(y, sr, 0.6, fade=0.02, tail=0.05)


# =============================================================================================
# GYM, RIDES, SLOTS
# =============================================================================================
GYM = ("tapis", "velo", "poids")


@sfx("gym", "Gym machines: tapis (treadmill motor, belt, feet), velo (flywheel and pedalling), poids (weight stack lifted, then landing).",
     kind=("tapis", " | ".join(GYM)), rate=(-1.0, "footfalls per minute, cadence rpm or reps per minute (-1 = the kind's own)"), dur=(5.0, "s"), seed=(0, "variation"))
def gym(sr=DEFAULT_SR, kind="tapis", rate=-1.0, dur=5.0, seed=0):
    k = _kind(kind, GYM, "gym")
    d, rng, seed = _dur(dur, 1.0), np.random.default_rng(int(seed)), int(seed)
    n, t = _n(d), np.arange(_n(d)) / P
    if k == "tapis":
        pace = float(rate) if float(rate) > 0 else 170.0
        feet = _nm(_m(footsteps(sr=P, ground="wood", gait="run", pace=pace, steps=int(d * pace / 60.0), variation=0.1, seed=seed), n))
        belt = _nm(_m(K.roll(sr=P, dur=d, speed=2.5, radius=0.03, roughness=0.6, material="rubber", surface="wood", surface_size=1.2, seed=seed), n))
        y = 0.35 * _motor(2400.0, d, 0.5, 0.0, seed) + 0.15 * belt + feet
    elif k == "velo":
        c = (float(rate) if float(rate) > 0 else 70.0) / 60.0
        p = np.cumsum(12.0 * c * (1.0 + 0.04 * np.sin(2 * np.pi * c * t))) / P      # the push of each pedal turn
        whirr = amb.bp(rng.standard_normal(n), 900.0, 1.2) * (0.6 + 0.4 * np.sin(2 * np.pi * c * t)) * (1.0 + 0.1 * np.sin(2 * np.pi * p))
        click = _clicks(n, RATCHET, _beats(c, d - 0.1), rng)
        y = _nm(whirr) + 0.5 * _nm(click) + 0.25 * _nm(_m(gears(sr=P, dur=d, rpm=12.0 * c * 60.0, driver_teeth=11, driven_teeth=44, seed=seed), n))
    else:
        r = (float(rate) if float(rate) > 0 else 15.0) / 60.0
        y = np.zeros(n)
        for j, b in enumerate(_beats(r, d - 0.6)):
            _put(y, _m(FO.clang(sr=P, object="metal_box", size=1.5, force=0.3, dur=0.4, seed=seed + 2 * j)), float(b), 0.3)     # lifted off
            _put(y, _m(FO.clang(sr=P, object="metal_box", size=1.0, force=1.0, dur=0.5, seed=seed + 2 * j + 1)), float(b) + 0.55 / r, 1.0)  # landing
    return _fin(y, sr, 0.7, fade=0.02, tail=0.1)


RIDES = ("montagnes_russes", "manege")


def _coaster(d: float, seed: int, rng) -> np.ndarray:
    lift, crest = 0.3 * d, 0.05 * d
    fast = d - lift - crest
    y = np.zeros(_n(d + 0.5))
    _put(y, FO.ratchet(sr=P, clicks=int(lift * LIFT_CLACKS), rate=LIFT_CLACKS, size=2.2, mount="metal", seed=seed), 0.0, 0.5)
    at = lift + crest
    k = _n(fast)
    run = _nm(_m(FO.train(sr=P, dur=fast, speed=70.0, rail_len=6.0, wheelbase=1.2, bogie_spacing=3.0, seed=seed), k))
    wind = _nm(amb.bp(rng.standard_normal(k), 700.0, 0.6))
    drop = np.interp(np.arange(k), [0, _n(0.4), k - _n(0.5), k], [0.0, 1.0, 1.0, 0.0])
    _put(y, (run + 0.5 * wind) * drop, at, 1.0)
    _put(y, _m(REGISTRY["scream"]["fn"](sr=P, voice="female", seed=seed)), at + 0.2, 0.5)
    _put(y, _m(REGISTRY["scream"]["fn"](sr=P, voice="male", seed=seed + 1)), at + 0.7, 0.4)
    return y


LIFT_CLACKS = 5.0                                           # anti-rollback clacks per second
CAROUSEL_TURN = 8.0                                         # s per turn


def _carousel(d: float, seed: int, rng) -> np.ndarray:
    n = _n(d)
    t = np.arange(n) / P
    turn = 0.7 + 0.3 * np.sin(2 * np.pi * t / CAROUSEL_TURN)
    crowd = _nm(_m(crowd_murmur(sr=P, people=25, mood=0.7, distance=0.4, dur=d, seed=seed), n))
    return 0.5 * _motor(700.0, d, 0.5, 0.0, seed) + 0.4 * _nm(_geared(700.0, 18, 90, 0.5, d, seed, wear=0.0)) + 0.9 * crowd * turn


@sfx("ride", "Fairground: montagnes russes (chain lift clacking, the crest, the drop with wind, wheels and screams), manege (turning carousel, motor, crowd).",
     kind=("montagnes_russes", " | ".join(RIDES)), dur=(10.0, "s"), seed=(0, "variation"))
def ride(sr=DEFAULT_SR, kind="montagnes_russes", dur=10.0, seed=0):
    k = _kind(kind, RIDES, "ride")
    d, rng, seed = _dur(dur, 5.0), np.random.default_rng(int(seed)), int(seed)
    return _fin(_coaster(d, seed, rng) if k == "montagnes_russes" else _carousel(d, seed, rng), sr, 0.8, fade=0.05, tail=0.4)


SLOT = ("play", "pull", "spin", "stop", "win")


def _slot_part(s: str, seed: int, spin: float, gap: float, reels: int) -> np.ndarray:
    if s == "pull":
        y = np.zeros(_n(0.6))
        _put(y, FO.ratchet(sr=P, clicks=8, rate=22.0, size=1.4, mount="metal", seed=seed), 0.0, 0.6)
        _put(y, _m(solenoid(sr=P, size=0.7, metal=0.5, thump=0.7, bounce=0.3, seed=seed)), 0.38, 1.0)
        return y
    if s == "spin":
        k = _n(spin)
        r = _fit(FO.ratchet(sr=P, clicks=int(spin * 40.0), rate=40.0, size=0.8, mount="metal", seed=seed), k)
        w = _nm(_m(spinner(sr=P, spins=int(spin * 18.0), rate=18, seed=seed), k))
        return _nm(r) + 0.6 * w
    if s == "stop":
        y = np.zeros(_n(reels * gap + 0.2))
        for j in range(reels):
            _put(y, _m(solenoid(sr=P, size=0.8, metal=0.4, thump=0.8, bounce=0.2, seed=seed + j)), j * gap)
        return y
    y = np.zeros(_n(1.6))                                   # win: the coins fall in the tray under a bell
    for j in range(10):
        _put(y, _m(coin(sr=P, freq=988.0 + 20.0 * j, up=3)), 0.12 * j, 0.25)
    _put(y, _m(FO.shake(sr=P, seed=seed)), 0.05, 0.3)
    _put(y, _m(ui_chime(sr=P, kind="achievement", seed=seed)), 0.0, 0.7)
    return y


@sfx("slot_machine", "One-armed bandit: pull (lever and ratchet), spin (reels), stop (one thunk per reel), win (coins and bell), play (the whole sequence).",
     state=("play", " | ".join(SLOT)), reels=(3, "reels"), win=(1, "play: ends in a payout"), seed=(0, "variation"))
def slot_machine(sr=DEFAULT_SR, state="play", reels=3, win=1, seed=0):
    """`ball_roll` is for the roulette, which the plan does not list here."""
    s = _kind(state, SLOT, "slot_machine")
    rl, seed = int(np.clip(reels, 1, 8)), int(seed)
    spin, gap = 1.2, 0.35
    if s != "play":
        return _fin(_slot_part(s, seed, spin, gap, rl), sr, 0.8, fade=0.002, tail=0.05)
    y = np.zeros(_n(0.5 + spin + rl * gap + 2.0))
    _put(y, _slot_part("pull", seed, spin, gap, rl), 0.0)
    _put(y, 0.5 * _slot_part("spin", seed, spin + rl * gap, gap, rl), 0.5)
    _put(y, _slot_part("stop", seed, spin, gap, rl), 0.5 + spin)
    if int(win):
        _put(y, _slot_part("win", seed, spin, gap, rl), 0.5 + spin + rl * gap + 0.1)
    return _fin(y, sr, 0.8, fade=0.002, tail=0.05)


# =============================================================================================
# PUMPS, SEWING, MEDICAL
# =============================================================================================
PUMPS = ("aquarium", "puits", "air")


@sfx("pump", "Pumps: aquarium (membrane at twice the mains, bubbles), puits (slow piston, the water), air (compressor piston with its pressure hiss).",
     kind=("aquarium", " | ".join(PUMPS)), mains=(50.0, "Hz, aquarium"), rate=(-1.0, "strokes per second (-1 = the kind's own)"), dur=(4.0, "s"), seed=(0, "variation"))
def pump(sr=DEFAULT_SR, kind="aquarium", mains=50.0, rate=-1.0, dur=4.0, seed=0):
    """Membrane at 2 x mains."""
    k = _kind(kind, PUMPS, "pump")
    d, seed = _dur(dur, 0.5), int(seed)
    n = _n(d)
    if k == "aquarium":
        # Not `buzzer`: its two saws 0.4 Hz apart cancel their fundamental every 2.5 s, and the line must stay at 2 x mains.
        ph = 2 * np.pi * 2.0 * float(mains) * np.arange(n) / P
        y = _modes(np.tanh(2.0 * np.sin(ph)), [320.0, 780.0], 0.05, [0.3, 0.1]) * 0.1 + np.sin(ph) + 0.3 * np.sin(2 * ph) + 0.1 * np.sin(3 * ph)
        y = _nm(y) + 0.5 * _nm(_m(bubbles(sr=P, dur=d, rate=14.0, size=2.5, seed=seed), n))
    else:
        r = float(rate) if float(rate) > 0 else (0.8 if k == "puits" else 4.0)
        st = _fit(FO.piston(sr=P, cycles=max(1, int(d * r)), rate=r, pressure=0.3 if k == "puits" else 0.9, seed=seed), n)
        if k == "puits":
            ph = (np.arange(n) / P * r) % 1.0
            water = _nm(_m(bubbles(sr=P, dur=d, rate=30.0, size=6.0, seed=seed), n)) * (ph < 0.5)
            y = _nm(st) + 0.6 * amb.lop(water, 1500.0)
        else:
            y = _nm(st) + 0.3 * _motor(1450.0, d, 0.8, 0.0, seed) + 0.15 * _nm(_m(air_leak(sr=P, dur=d, pressure=0.4, orifice=1.5, seed=seed), n))
    return _fin(y, sr, 0.7, fade=0.03, tail=0.1)


@sfx("sewing_machine", "Sewing machine: motor and gears, the needle cam ticking 800 times a minute with a lighter tick as it rises.",
     spm=(800.0, "stitches per minute"), dur=(2.0, "s"), seed=(0, "variation"))
def sewing_machine(sr=DEFAULT_SR, spm=800.0, dur=2.0, seed=0):
    d, rng, seed = _dur(dur, 0.3), np.random.default_rng(int(seed)), int(seed)
    n = _n(d)
    s = float(np.clip(spm, 60.0, 5000.0))
    r = s / 60.0
    down = _clicks(n, ENDSTOP, _beats(r, d - 0.05), rng) + 0.5 * _clicks(n, RATCHET, _beats(r, d - 0.05), rng)
    up = _clicks(n, RATCHET, _beats(r, d - 0.05, 0.5 / r), rng, 0.2)
    gear = _nm(_m(gears(sr=P, dur=d, rpm=s, driver_teeth=20, driven_teeth=20, seed=seed), n))
    return _fin(_nm(down + up) + 0.2 * _motor(10.0 * s, d, 0.5, 0.0, seed) + 0.1 * gear, sr, 0.7, fade=0.01, tail=0.05)


MEDICAL = ("EKG", "respirateur", "moniteur", "bip")
BEEP_RANGE = (150.0, 1000.0)
BEEP_HARMONICS = 5
BEEP_EDGE = 0.15                                            # rise and fall, fraction of the pulse
MONITOR = (0.15, 0.15, 2.5)                                 # medium priority: pulse, gap between pulses, pause after the burst (s)


def _pulse(f: float, d: float) -> np.ndarray:
    """An alarm pulse: fundamental in range, 5 harmonics at 1/k, edges `BEEP_EDGE` of its length."""
    f = float(np.clip(f, *BEEP_RANGE))
    n = _n(d)
    t = np.arange(n) / P
    y = sum(np.sin(2 * np.pi * k * f * t) / k for k in range(1, BEEP_HARMONICS + 1))
    e = max(1, min(int(BEEP_EDGE * n), n // 2))
    y[:e] *= np.linspace(0.0, 1.0, e)
    y[-e:] *= np.linspace(1.0, 0.0, e)
    return y


@sfx("medical", "Hospital machines: EKG (one beep per heartbeat), respirateur (piston, breath in and out, 15 a minute), moniteur (three-pulse alarm), bip (one tone).",
     kind=("EKG", " | ".join(MEDICAL)), freq=(880.0, "Hz of the beep, 150..1000"), rate=(-1.0, "beats or breaths per minute (-1 = the kind's own)"), dur=(5.0, "s"), seed=(0, "variation"))
def medical(sr=DEFAULT_SR, kind="EKG", freq=880.0, rate=-1.0, dur=5.0, seed=0):
    k = str(kind).replace(" ", "_")
    k = {"ekg": "EKG"}.get(k.lower(), k)
    k = _kind(k, MEDICAL, "medical")
    d, seed, f = _dur(dur, 0.2), int(seed), float(freq)
    n = _n(d)
    y = np.zeros(n)
    if k == "bip":
        return _fin(_pulse(f, 0.2), sr, 0.6, fade=0.002)
    if k == "EKG":
        for b in _beats((float(rate) if float(rate) > 0 else 72.0) / 60.0, d - 0.1):
            _put(y, _pulse(f, 0.08), float(b))
    elif k == "moniteur":
        td, ts, pause = MONITOR
        burst = 3 * td + 2 * ts
        for b in _beats(1.0 / (burst + pause), d - burst):
            for j in range(3):
                _put(y, _pulse(f, td), float(b) + (td + ts) * j)
    else:
        per = 60.0 / (float(rate) if float(rate) > 0 else 15.0)
        for j, b in enumerate(_beats(1.0 / per, d - 0.1)):
            b = float(b)
            _put(y, _nm(_m(breath(sr=P, kind="in", effort=0.6, dur=0.4 * per, seed=seed + j))), b, 0.6)
            _put(y, FO.piston(sr=P, cycles=1, rate=1.0 / per, pressure=0.4, seed=seed + j), b, 0.6)
            _put(y, _nm(_m(breath(sr=P, kind="out", effort=0.4, dur=0.45 * per, seed=seed + j))), b + 0.5 * per, 0.4)
        return _fin(y, sr, 0.6, fade=0.01, tail=0.1)
    return _fin(y, sr, 0.6, fade=0.002, tail=0.01)


# =============================================================================================
# INDUSTRY
# =============================================================================================
INDUSTRIAL = ("presse", "marteau_pilon", "usine")
FORGE_MODES = (180.0, 420.0, 760.0, 1300.0, 2300.0, 3700.0)  # anvil and work


@sfx("industrial", "Heavy industry: presse (flywheel, one stamp per stroke), marteau-pilon (power hammer: the first blow dull as it makes the crater, the next ones clearer and elastic), usine (factory floor).",
     kind=("presse", " | ".join(INDUSTRIAL)), rate=(-1.0, "blows per second (-1 = the kind's own)"), dur=(5.0, "s"), seed=(0, "variation"))
def industrial(sr=DEFAULT_SR, kind="presse", rate=-1.0, dur=5.0, seed=0):
    """Forge: see `forge_times`."""
    k = _kind(str(kind).replace("-", "_"), INDUSTRIAL, "industrial")
    d, rng, seed = _dur(dur, 1.0), np.random.default_rng(int(seed)), int(seed)
    n = _n(d)
    rmb = _rumble(d, 25.0, 90.0, seed)
    if k == "marteau_pilon":
        r = float(rate) if float(rate) > 0 else 0.8
        at = _beats(r, d - 0.7, 0.1)
        y = _nm(_modes(_forge_force(n, at), FORGE_MODES, 0.5, [1.0, 0.8, 0.7, 0.5, 0.4, 0.3]))
        hiss = _fit(FO.piston(sr=P, cycles=max(1, int(d * r)), rate=max(r, 0.2), pressure=0.6, seed=seed), n)
        return _fin(y + 0.05 * _nm(hiss) + 0.05 * rmb, sr, 0.85, fade=0.01, tail=0.1)
    if k == "presse":
        r = float(rate) if float(rate) > 0 else 1.0
        y = 0.3 * _motor(960.0, d, 0.7, 0.0, seed) + 0.2 * rmb
        for j, b in enumerate(_beats(r, d - 0.7, 0.1)):
            _put(y, _m(FO.clang(sr=P, object="metal_box", size=2.5, force=1.0, dur=0.5, seed=seed + j)), float(b), 1.2)
        return _fin(y, sr, 0.85, fade=0.02, tail=0.1)
    y = 0.12 * _motor(1450.0, d, 0.7, 0.0, seed) + 0.12 * rmb + 0.1 * _nm(_geared(1450.0, 23, 41, 0.7, d, seed))
    y = y + 0.05 * _nm(_m(transformer(sr=P, dur=d, seed=seed), n))
    for j, b in enumerate(_beats(float(rate) if float(rate) > 0 else 0.8, d - 0.7, 0.1)):
        _put(y, _m(FO.clang(sr=P, object="metal_box", size=3.0, force=0.8, dur=0.5, seed=seed + j)), float(b), 1.2)
    return _fin(y, sr, 0.8, fade=0.1, tail=0.2)


# =============================================================================================
# BOOK SOUNDS
# =============================================================================================
PULLEY_MODES = (1.0, 1.59, 2.14)                            # as `friction` body ratios
BELT_REF = 1.5                                              # friction output to full scale
BELT_WIDTH = 0.02                                           # m


def belt_slip(torque, radius: float, preload: float, mu: float, wrap: float) -> np.ndarray:
    """Slip of a belt in 0..1: 0 while part of the arc still grips (M/2RT0 under `slip_torque`), rising linearly once all of it slides."""
    s = np.asarray(torque, float) / (2.0 * float(radius) * float(preload))
    c = slip_torque(mu, wrap)
    return np.clip((s - c) / max(1.0 - c, 1e-6), 0.0, 1.0)


@sfx("belt_squeal", "A slipping belt: silent grip until the torque makes the whole wrapped arc slide (mu .5, half a turn: T1/T2 = 4.81, M/2RT0 = 0.656), then stick-slip squeal on the pulley's modes.",
     torque=(25.0, "N m reached at the end of the rise"), torque_start=(0.0, "N m at the start"), rise=(0.5, "s of torque rise, then held"), radius=(0.06, "m pulley"),
     preload=(150.0, "N belt tension at rest"), mu=(0.5, "belt on pulley"), wrap=(math.pi, "rad wrapped"), speed=(12.0, "m/s belt"), pulley=(2800.0, "Hz first pulley mode"),
     oil=(0.1, "Pa s of the film on the pulley: thicker = the squeal stops at lower sliding speeds"), dur=(2.0, "s"), seed=(0, "variation"))
def belt_squeal(sr=DEFAULT_SR, torque=25.0, torque_start=0.0, rise=0.5, radius=0.06, preload=150.0, mu=0.5, wrap=math.pi, speed=12.0, pulley=2800.0, oil=0.1, dur=2.0, seed=0):
    """The onset comes out of the torque, there is no squeal knob. The Stribeck speed comes from the film (`stribeck_speed`, load 2 T0 / width, 1 um)."""
    d = _dur(dur, 0.2)
    rng = np.random.default_rng(int(seed))
    n = _n(d)
    x = np.arange(n) / P
    M = float(torque_start) + (float(torque) - float(torque_start)) * np.clip(x / max(float(rise), 1e-3), 0.0, 1.0)
    s = belt_slip(M, radius, preload, mu, wrap)
    grip = np.minimum(np.cumsum(s > 0) / (0.01 * P), 1.0)   # 10 ms for the kernel to take hold
    r = np.array(PULLEY_MODES)
    vs = stribeck_speed(oil, radius, 2.0 * float(preload) / BELT_WIDTH)
    y = _rub_engine(float(pulley) * r, 0.004 * np.sqrt(r), 200.0 / r ** 2, np.ones(3), 2.0 * grip * (1.0 + s), 0.1 * float(speed) * np.maximum(s, 0.4) * grip + 1e-4, rng, 0.03, mus=float(mu) * 1.2, mud=float(mu) * 0.4, vs=vs, sigma0=3.0e5)["out"]
    hum = amb.bp(rng.standard_normal(n), 400.0, 0.8)
    return _fin(0.02 * _nm(hum) + y / BELT_REF, sr, None, src=FS, fade=0.003, tail=0.02)


@sfx("savart_wheel", "Savart's wheel: a reed scraped by a toothed wheel sounds teeth x rpm / 60, whatever the reed's own frequency. mount cup adds a cup formant.",
     teeth=(60, "teeth"), rpm=(600.0, "wheel rpm"), rpm_end=(-1.0, "glide to this rpm (-1 = steady)"), reed_f=(1200.0, "Hz of the reed"), mount=("free", "free | cup"),
     dur=(2.0, "s"), seed=(0, "variation"))
def savart_wheel(sr=DEFAULT_SR, teeth=60, rpm=600.0, rpm_end=-1.0, reed_f=1200.0, mount="free", dur=2.0, seed=0):
    if mount not in ("free", "cup"):
        raise ValueError(f"savart_wheel mount {mount!r}: use free or cup")
    d, rng = _dur(dur, 0.1), np.random.default_rng(int(seed))
    n = _n(d)
    f = _ramp(float(np.clip(rpm, 10.0, 12000.0)), float(rpm_end), n) / 60.0
    exc = _impacts(n, np.cumsum(int(np.clip(teeth, 3, 1000)) * f) / P, rng)
    fr = float(np.clip(reed_f, 100.0, 8000.0))
    y = _nm(_modes(exc, [fr, min(6.27 * fr, 0.45 * P)], 6.91 / 150.0, [1.0, 0.2])) + 0.3 * _nm(exc)
    if mount == "cup":
        y = y + 1.2 * _band(y, 600.0, 1050.0)
    return _fin(y, sr, 0.7, fade=0.01, tail=0.03)
