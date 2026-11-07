"""The house, the office, the objects: locks, doors, appliances, plumbing, alarms, tools, things people handle and drop.

Sources of every value: research/SOURCES.md, section household. Nothing here has been listened to: see tests/test_household.py for what was measured.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter

from . import contact as K
from . import foley as FO
from .core import DEFAULT_SR
from .cyber import bearing, gearbox, transformer
from .foley import CATCH, ENDSTOP, P, RATCHET, _drive, _fin, _mclick, _n, _nm, _phisem, _put
from .friction import FS, GLASS, _friction_note, _gate, elastoplastic_friction, squeak
from .matter import MINNAERT_WR, air_leak, bubble_params, drain, gurgle, pour, sizzle, steam
from .physical import ambience as amb
from .pinball import solenoid, spring
from .road import _band, _modes
from .sfx import electric_motor, fan, sfx
from .tubes import pipe_blow


def _kind(kind, table, what) -> str:
    k = str(kind).replace(" ", "_").replace("-", "_")
    if k not in table:
        raise ValueError(f"{what} kind {kind!r}: use {' | '.join(table)}")
    return k


def _z(x) -> np.ndarray:
    """Mono, unit peak."""
    y = np.asarray(x, float)
    return _nm(y.mean(axis=1) if y.ndim == 2 else y)


def _mix(dur: float, *events) -> np.ndarray:
    """Events: (start s, signal, gain); each signal is brought to unit peak first."""
    y = np.zeros(_n(dur))
    for t, x, g in events:
        _put(y, _z(x), t, g)
    return y


def _fitz(x, n: int) -> np.ndarray:
    y = _z(x)
    return np.pad(y, (0, n - len(y))) if len(y) < n else y[:n]


def _noise(rng, n: int) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, n)


def _env(n: int, *points) -> np.ndarray:
    """Piecewise-linear envelope over (time s, value) points."""
    t, v = zip(*points)
    return np.interp(np.arange(n) / P, t, v)


def _ticks(rng, times, parts=CATCH, dur=None, gains=None) -> np.ndarray:
    times = np.asarray(times, float)
    n = _n(float(times.max()) + 0.06 if dur is None else dur)
    nz = amb.hip(_noise(rng, n), 300.0)
    y = np.zeros(n)
    for i, t in enumerate(times):
        y += _mclick(nz, parts, float(t), 1.0 if gains is None else float(gains[i]))
    return y


def _snap(rng, f0: float, t60: float = 0.03, ms: float = 3.0) -> np.ndarray:
    """A spring letting go: `ms` of noise into a small body with the ratios of a free-free bar."""
    n = _n(ms * 1e-3 + 6.0 * t60)
    x = np.zeros(n)
    m = _n(ms * 1e-3)
    x[:m] = _noise(rng, m)
    return _modes(x, f0 * np.array([1.0, 2.76, 5.4]), t60, [1.0, 0.6, 0.3])


def rms_of(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y * y)))


def _a_rms(y: np.ndarray, sr: int) -> float:
    """A-weighted RMS."""
    Y = np.abs(np.fft.rfft(y)) ** 2
    f2 = np.fft.rfftfreq(len(y), 1.0 / sr) ** 2
    a = 12194.0**2 * f2**2 / ((f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194.0**2)) * 1.2589
    return float(np.sqrt(2.0 * np.sum(Y * a * a)) / len(y))


DB_REF = 73.2                                               # dB(A) at A-weighted RMS 0.2


def _to_dba(y: np.ndarray, db: float, sr: int = P) -> np.ndarray:
    return y * (0.2 * 10.0 ** ((db - DB_REF) / 20.0) / (_a_rms(y, sr) + 1e-12))


FAN_POWER = 5.0                                             # sound power as speed^5


def fan_law_db(speed) -> float:
    """Fan law: the sound power grows as the fifth power of the speed, 50 log10(speed) dB (+7.3 dB at +40 %)."""
    return 10.0 * FAN_POWER * math.log10(max(float(speed), 1e-3))


# =============================================================================================
# LOCKS: spring and stop, short impacts on a small body
# =============================================================================================
LOCKS = ("cadenas", "loquet", "clef", "verrou", "combinaison")


def _shackle(rng, seed) -> list:
    """Padlock: the shackle slides, the spring lets go, the hollow body rings, the shackle rattles."""
    return [(0.0, K.scrape(sr=P, dur=0.06, speed=0.3, force=0.5, roughness=0.3, material="steel", body="bar", size=0.05, seed=seed), 0.25),
            (0.05, _snap(rng, 3200.0), 0.7),
            (0.055, K.impact(sr=P, material="steel", shape="rect_plate", size=0.05, hollow=0.6, velocity=2.5, hardness=0.8, modes=12, dur=0.2, seed=seed), 1.0),
            (0.19, K.impact(sr=P, material="steel", shape="bar", size=0.05, velocity=0.8, hardness=0.8, modes=8, dur=0.1, seed=seed + 1), 0.25),
            (0.27, K.impact(sr=P, material="steel", shape="bar", size=0.05, velocity=0.5, hardness=0.8, modes=8, dur=0.1, seed=seed + 2), 0.12)]


@sfx("lock", "Locks: cadenas (shackle snapping into a padlock), loquet (thumb latch falling), clef (key in, pins, turn, bolt), verrou (slide bolt), combinaison (dial clicks, then the shackle).",
     kind=("clef", " | ".join(LOCKS)), seed=(0, "variation"))
def lock(sr=DEFAULT_SR, kind="clef", seed=0):
    """Spring and stop."""
    k = _kind(kind, LOCKS, "lock")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    if k == "cadenas":
        y = _mix(0.5, *_shackle(rng, seed))
    elif k == "loquet":
        y = _mix(0.45, (0.0, FO.switch(sr=P, kind="toggle", mount="metal", seed=seed), 0.4),
                 (0.16, K.impact(sr=P, material="metal", shape="bar", size=0.12, velocity=1.5, hardness=0.8, modes=10, dur=0.15, seed=seed), 1.0),
                 (0.16, FO.thud(sr=P, body="crate", contact_ms=8.0, size=0.6, force=0.5, seed=seed), 0.4))
    elif k == "clef":
        pins = 0.09 + np.cumsum(rng.uniform(0.022, 0.04, 5))
        y = _mix(0.95, (0.0, FO.keys(sr=P, hits=1, seed=seed), 0.12),
                 (0.0, K.scrape(sr=P, dur=0.3, speed=0.15, force=0.5, roughness=0.4, material="steel", body="bar", size=0.06, seed=seed), 0.2),
                 (0.0, _ticks(rng, pins, CATCH, 0.4), 0.5),
                 (0.42, _ticks(rng, [0.0, 0.12], RATCHET, 0.25), 0.5),
                 (0.58, _ticks(rng, [0.0], ENDSTOP, 0.2), 0.9),
                 (0.58, K.impact(sr=P, material="steel", shape="bar", size=0.09, velocity=2.0, hardness=0.8, modes=10, dur=0.2, seed=seed), 0.8))
    elif k == "verrou":
        y = _mix(0.6, (0.0, K.scrape(sr=P, dur=0.25, speed=0.25, force=1.0, roughness=0.6, material="steel", body="bar", size=0.15, seed=seed), 0.5),
                 (0.24, K.impact(sr=P, material="steel", shape="bar", size=0.15, velocity=2.5, hardness=0.9, modes=12, dur=0.25, seed=seed), 1.0),
                 (0.24, solenoid(sr=P, size=0.2, metal=0.7, thump=0.5, bounce=0.2, seed=seed + 1), 0.4))
    else:
        turns = []
        t = 0.0
        for count, step in ((9, 0.035), (6, 0.05), (4, 0.075)):      # three numbers, slowing down
            turns += list(t + np.cumsum(step * rng.uniform(0.85, 1.15, count)))
            t = turns[-1] + 0.22
        y = np.zeros(_n(t + 0.55))
        _put(y, _z(_ticks(rng, turns, RATCHET, t)), 0.0, 0.45)
        for t0, x, g in _shackle(rng, seed):
            _put(y, _z(x), t + t0, g)
    return _fin(y, sr, 0.75, fade=0.001, tail=0.02)


# =============================================================================================
# DOORS
# =============================================================================================
DOOR_FAMILY = ("donjon", "garage", "prison", "coulissante", "hi_tech", "battante", "tournante")
DOOR_WEIGHT = {"donjon": 0.9, "garage": 0.8, "prison": 0.8, "coulissante": 0.4, "hi_tech": 0.5, "battante": 0.3, "tournante": 0.5}
REVOLVE_PASS = 1.0                                          # wings past the seal per second: 4 wings at 15 rpm


@sfx("door_family", "Doors: donjon (bolt and chain, long creak, heavy shut), garage (motor, chain, rolling panels), prison (barred door on a rail, clang, lock), coulissante, hi-tech (pneumatic), battante (swings back and forth), tournante (revolving, brushes passing).",
     kind=("battante", " | ".join(DOOR_FAMILY)), material=("", "'' = the kind's own | wood | metal"), weight=(-1.0, "0 light .. 1 heavy (-1 = the kind's own)"),
     dur=(4.0, "tournante: s"), seed=(0, "variation"))
def door_family(sr=DEFAULT_SR, kind="battante", material="", weight=-1.0, dur=4.0, seed=0):
    """`door`, `creak`, `drawer`, `piston`, `clang`, `chain`, `electric_motor`, with a `kind` and a weight."""
    k = _kind(kind, DOOR_FAMILY, "door_family")
    if material not in ("", "wood", "metal"):
        raise ValueError(f"door_family material {material!r}: use wood or metal")
    w = DOOR_WEIGHT[k] if float(weight) < 0 else float(np.clip(weight, 0.0, 1.0))
    seed = int(seed)
    rng = np.random.default_rng(seed)
    metal = material == "metal" or (material == "" and k in ("garage", "prison", "hi_tech"))
    cr, body = ("iron", "box") if metal else ("wood", "crate")
    shut = FO.thud(sr=P, body=body, contact_ms=25.0 - 15.0 * (1.0 - w), size=0.8 + 1.2 * w, force=0.5 + 0.5 * w, seed=seed)
    if k == "donjon":
        d = 1.6 + 1.2 * w
        y = _mix(d + 1.5, (0.0, FO.chain(sr=P, dur=0.7, pushes=1, weight=w, seed=seed), 0.4),
                 (0.55, FO.clang(sr=P, object="metal_box", size=1.0 + w, force=0.7, damping=6.0, dur=0.5, seed=seed), 0.6),
                 (0.85, FO.creak(sr=P, material=cr, force="pull", level=0.6 + 0.4 * w, dur=d, size=1.0 + w, seed=seed), 0.7),
                 (0.85 + d - 0.1, shut, 1.0),
                 (0.85 + d - 0.1, K.rumble(sr=P, dur=0.5, fmin=30.0, fmax=90.0, seed=seed), 0.4 * w))
    elif k == "garage":
        d = 2.5 + 1.5 * w
        n = _n(d)
        mot = amb.lop(_fitz(electric_motor(sr=P, dur=d, rpm=1500.0, load=0.7, startup=0.3, seed=seed), n), 2500.0)
        y = _mix(d + 0.7, (0.0, mot * _env(n, (0.0, 1.0), (d - 0.1, 1.0), (d, 0.0)), 0.45),
                 (0.0, FO.chain(sr=P, dur=d, pushes=4, weight=0.4, density=25.0, settle=0, seed=seed), 0.3),
                 (0.1, K.roll(sr=P, dur=d - 0.1, speed=0.4, radius=0.03, roughness=0.6, material="steel", surface="steel", surface_size=2.0, joints=0.4, seed=seed), 0.4),
                 (d - 0.05, shut, 1.0),
                 (d, solenoid(sr=P, size=0.4, metal=0.4, thump=0.3, bounce=0.1, seed=seed), 0.4))
    elif k == "prison":
        run = 0.7 + 0.5 * w
        y = _mix(run + 1.3, (0.0, K.roll(sr=P, dur=run, speed=0.8, radius=0.03, roughness=0.7, material="steel", surface="steel", surface_size=1.5, joints=0.3, seed=seed), 0.45),
                 (0.0, FO.chain(sr=P, dur=run, pushes=1, weight=0.3, seed=seed), 0.15),
                 (run, FO.clang(sr=P, object="metal_box", size=1.0 + w, force=1.0, damping=2.0, dur=1.2, seed=seed), 1.0),
                 (run + 0.1, solenoid(sr=P, size=0.7, metal=0.8, thump=0.3, bounce=0.5, seed=seed), 0.6))
    elif k == "coulissante":
        d = 0.7 + 0.6 * w
        n = _n(d)
        run = K.roll(sr=P, dur=d, speed=0.5, radius=0.015, roughness=0.3, material="steel", surface="metal" if metal else "wood", surface_size=1.2, seed=seed)
        y = _mix(d + 0.5, (0.0, _fitz(run, n) * _env(n, (0.0, 0.0), (0.1, 1.0), (d, 0.5)), 0.5),
                 (d - 0.15, FO.drawer(sr=P, direction="close", dur=0.3, size=1.5 + w, force=0.4 + 0.5 * w, seed=seed), 1.0))
    elif k == "hi_tech":
        y = _mix(1.7, (0.0, solenoid(sr=P, size=0.3, metal=0.6, thump=0.2, bounce=0.2, seed=seed), 0.5),
                 (0.06, air_leak(sr=P, dur=0.7, pressure=0.7, orifice=8.0, burst=0.25, seed=seed), 0.55),
                 (0.08, FO.servo(sr=P, moves=1, dur=0.75, speed=180.0, seed=seed), 0.4),
                 (0.08, FO.piston(sr=P, cycles=1, rate=0.6, pressure=0.5, seed=seed), 0.5),
                 (0.85, FO.thud(sr=P, body=body, contact_ms=10.0, size=0.7, force=0.5, seed=seed), 0.6))
    elif k == "battante":
        period = 0.5 + 0.3 * w
        ev = [(0.0, FO.creak(sr=P, material=cr, force="sway", level=0.5, dur=3.0 * period, size=0.8 + w, seed=seed), 0.45)]
        for i in range(4):                                  # swinging back through the frame, each pass weaker
            n = _n(0.25)
            ev.append((i * period, amb.bp(rng.standard_normal(n), 300.0, 0.7) * np.hanning(n), 0.5 * 0.6**i))
            ev.append((i * period + 0.1, FO.thud(sr=P, body=body, contact_ms=20.0, size=1.0 + w, force=0.5, seed=seed + i), 0.8 * 0.5**i))
        y = _mix(4.0 * period + 0.4, *ev)
    else:
        d = float(np.clip(dur, 1.0, 60.0))
        n = _n(d)
        t = np.arange(n) / P
        brush = amb.bp(rng.standard_normal(n), 2200.0, 0.7) * (0.5 - 0.5 * np.cos(2 * np.pi * REVOLVE_PASS * t)) ** 4
        whoosh = amb.bp(rng.standard_normal(n), 350.0, 0.8) * (0.6 + 0.4 * np.cos(2 * np.pi * REVOLVE_PASS * t))
        pivot = bearing(sr=P, dur=d, rpm=900.0, resonance=800.0, seed=seed)
        y = _mix(d, (0.0, brush, 0.6), (0.0, whoosh, 0.3), (0.0, pivot, 0.06))
    return _fin(y, sr, 0.8, fade=0.005, tail=0.03)


# =============================================================================================
# SMALL MECHANISMS
# =============================================================================================
@sfx("lever", "A big lever pulled: bearing creak, optional pawl, the stop at the end.",
     size=(0.5, "0 small .. 1 large (bigger = longer, lower)"), pawl=(1, "1 = ratchet teeth on the way"), seed=(0, "variation"))
def lever(sr=DEFAULT_SR, size=0.5, pawl=1, seed=0):
    """Creak and stop."""
    s = float(np.clip(size, 0.0, 1.0))
    d = 0.3 + 0.6 * s
    ev = [(0.0, FO.creak(sr=P, material="iron", force="pull", level=0.7, dur=d, size=0.7 + 1.3 * s, seed=seed), 0.6),
          (d, K.impact(sr=P, material="steel", shape="bar", size=0.15 + 0.5 * s, velocity=2.5, hardness=0.8, modes=12, dur=0.35, seed=seed), 1.0)]
    if pawl:
        ev.append((0.02, FO.ratchet(sr=P, clicks=int(3 + 5 * s), rate=(3 + 5 * s) / d, size=0.8 + 1.2 * s, seed=seed), 0.35))
    return _fin(_mix(d + 0.45, *ev), sr, 0.75, fade=0.002, tail=0.02)


@sfx("pulley", "A pulley running: bearing rumble, axle squeak, rope or pawl clicks at the sheave's rate.",
     rate=(4.0, "pawl clicks per second"), dur=(2.5, "s"), load=(0.5, "0..1 squeak under load"), seed=(0, "variation"))
def pulley(sr=DEFAULT_SR, rate=4.0, dur=2.5, load=0.5, seed=0):
    """`bearing`, `squeak`, `ratchet`: 15 rpm of bearing per click/s of the sheave."""
    r = float(np.clip(rate, 0.5, 30.0))
    d = float(np.clip(dur, 0.2, 30.0))
    ld = float(np.clip(load, 0.0, 1.0))
    n = _n(d)
    bed = _fitz(bearing(sr=P, dur=d, rpm=300.0 + 150.0 * r, state="outer", severity=0.3, resonance=1400.0, seed=seed), n)
    y = _mix(d, (0.0, bed, 0.3),
             (0.0, squeak(sr=P, surface="hinge", pressure=0.6 + 1.5 * ld, pitch=1.1 - 0.3 * ld, dur=d, seed=seed), 0.3 + 0.3 * ld),
             (0.0, FO.ratchet(sr=P, clicks=max(int(r * d), 1), rate=r, size=0.8, seed=seed), 0.7))
    return _fin(y, sr, 0.7, fade=0.01, tail=0.03)


@sfx("window", "Windows and blinds: battant (casement: latch, frame), coulissant (sash sliding, stop), store_lames (slatted blind rattling down), store_toile (roller blind, cloth, spring).",
     kind=("battant", "battant | coulissant | store_lames | store_toile"), seed=(0, "variation"))
def window(sr=DEFAULT_SR, kind="battant", seed=0):
    """Windows and blinds."""
    k = _kind(kind, ("battant", "coulissant", "store_lames", "store_toile"), "window")
    seed = int(seed)
    if k == "battant":
        y = _mix(0.8, (0.0, FO.switch(sr=P, kind="toggle", mount="metal", seed=seed), 0.5),
                 (0.08, FO.creak(sr=P, material="wood", force="pull", level=0.4, dur=0.45, size=0.7, seed=seed), 0.4),
                 (0.5, FO.thud(sr=P, body="crate", contact_ms=8.0, size=0.5, force=0.5, seed=seed), 0.9),
                 (0.5, K.impact(sr=P, material="glass", shape="rect_plate", size=0.4, velocity=0.8, hardness=0.5, modes=10, dur=0.2, seed=seed), 0.2))
    elif k == "coulissant":
        y = _mix(1.0, (0.0, K.scrape(sr=P, dur=0.6, speed=0.5, force=0.6, roughness=0.4, material="wood", size=0.5, seed=seed), 0.6),
                 (0.55, FO.drawer(sr=P, direction="close", dur=0.25, size=1.0, force=0.5, seed=seed), 1.0),
                 (0.7, K.impact(sr=P, material="glass", shape="rect_plate", size=0.5, velocity=0.6, hardness=0.5, modes=10, dur=0.25, seed=seed), 0.15))
    elif k == "store_lames":
        y = _mix(1.3, (0.0, FO.shake(sr=P, preset="Sticks", energy=0.5, rate=7.0, dur=1.0, seed=seed), 0.6),
                 (0.0, FO.cloth(sr=P, kind="rustle", dur=1.0, rate=2.0, weight=0.2, seed=seed), 0.12),
                 (1.0, FO.thud(sr=P, body="box", contact_ms=6.0, size=0.4, force=0.4, seed=seed), 0.5))
    else:
        y = _mix(1.2, (0.0, FO.cloth(sr=P, kind="rustle", dur=0.8, rate=1.0, weight=0.3, seed=seed), 0.4),
                 (0.0, FO.ratchet(sr=P, clicks=14, rate=25.0, slow=0.6, size=0.5, mount="plastic", seed=seed), 0.6),
                 (0.8, FO.cloth(sr=P, kind="flap", dur=0.25, rate=8.0, weight=0.3, seed=seed), 0.5))
    return _fin(y, sr, 0.7, fade=0.002, tail=0.02)


@sfx("umbrella", "An umbrella: ouvre (runner up the shaft, catch, the canopy snaps taut), ferme (catch, folding cloth), secoue (shaken dry).",
     action=("ouvre", "ouvre | ferme | secoue"), seed=(0, "variation"))
def umbrella(sr=DEFAULT_SR, action="ouvre", seed=0):
    """Spring and cloth."""
    a = _kind(action, ("ouvre", "ferme", "secoue"), "umbrella")
    seed = int(seed)
    if a == "ouvre":
        y = _mix(0.8, (0.0, K.scrape(sr=P, dur=0.14, speed=0.8, force=0.5, roughness=0.3, material="steel", body="bar", size=0.4, seed=seed), 0.4),
                 (0.13, spring(sr=P, action="release", tension=0.3, seed=seed + 5), 0.3),
                 (0.13, _snap(np.random.default_rng(seed), 2400.0), 0.5),
                 (0.14, FO.cloth(sr=P, kind="flap", dur=0.2, rate=12.0, weight=0.7, seed=seed), 1.0))
    elif a == "ferme":
        y = _mix(1.0, (0.0, _snap(np.random.default_rng(seed), 2400.0), 0.5),
                 (0.03, FO.cloth(sr=P, kind="rustle", dur=0.7, rate=3.0, weight=0.5, seed=seed), 0.8),
                 (0.03, K.scrape(sr=P, dur=0.5, speed=0.3, force=0.4, roughness=0.3, material="steel", body="bar", size=0.4, seed=seed), 0.25),
                 (0.7, FO.switch(sr=P, kind="toggle", mount="plastic", seed=seed), 0.5))
    else:
        y = _mix(1.0, (0.0, FO.cloth(sr=P, kind="flap", dur=0.9, rate=6.0, weight=0.6, seed=seed), 1.0),
                 (0.0, FO.shake(sr=P, preset="WaterDrops", energy=0.4, rate=6.0, dur=0.9, seed=seed), 0.15))
    return _fin(y, sr, 0.7, fade=0.002, tail=0.02)


# =============================================================================================
# OFFICE, ROPE, BLADES
# =============================================================================================
@sfx("office", "Desk objects: agrafeuse (stapler: spring, staple through paper, release), stylo (retractable pen clicked twice), tampon (rubber stamp on paper), bouton_de_bureau (desk button), ecriture (pen writing, in strokes).",
     kind=("agrafeuse", "agrafeuse | stylo | tampon | bouton_de_bureau | ecriture"), dur=(2.0, "ecriture: s"), seed=(0, "variation"))
def office(sr=DEFAULT_SR, kind="agrafeuse", dur=2.0, seed=0):
    """Spring + paper; writing as a very light scrape, 7 strokes/s."""
    k = _kind(kind, ("agrafeuse", "stylo", "tampon", "bouton_de_bureau", "ecriture"), "office")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    if k == "agrafeuse":
        y = _mix(0.5, (0.0, solenoid(sr=P, size=0.2, metal=0.7, thump=0.5, bounce=0.2, seed=seed), 1.0),
                 (0.006, FO.paper(sr=P, kind="page", dur=0.06, weight=0.8, seed=seed), 0.25),
                 (0.17, _snap(rng, 3500.0), 0.5),
                 (0.17, FO.thud(sr=P, body="box", contact_ms=4.0, size=0.25, force=0.4, seed=seed), 0.3))
    elif k == "stylo":
        y = _mix(0.42, (0.0, _snap(rng, 5200.0, 0.012), 1.0), (0.012, _ticks(rng, [0.0], CATCH, 0.05), 0.5),
                 (0.22, _snap(rng, 4300.0, 0.012), 0.75), (0.232, _ticks(rng, [0.0], CATCH, 0.05), 0.35))
    elif k == "tampon":
        y = _mix(0.5, (0.0, FO.thud(sr=P, body="crate", contact_ms=7.0, size=0.5, force=0.8, seed=seed), 1.0),
                 (0.001, K.impact(sr=P, material="wood", shape="rect_plate", size=0.5, velocity=1.5, hardness=0.5, modes=10, dur=0.1, seed=seed), 0.35),
                 (0.16, FO.velcro(sr=P, dur=0.07, speed=0.2, seed=seed), 0.12))
    elif k == "bouton_de_bureau":
        y = FO.button(sr=P, release=1, hold=0.12, mount="plastic", seed=seed)
    else:
        d = float(np.clip(dur, 0.2, 30.0))
        n = _n(d)
        t = np.arange(n) / P
        pace = np.interp(t, np.arange(0.0, d + 0.3, 0.3), rng.uniform(0.2, 1.0, len(np.arange(0.0, d + 0.3, 0.3))))
        strokes = (0.5 + 0.5 * np.sin(2 * np.pi * np.cumsum(7.0 * (0.7 + 0.6 * pace)) / P)) ** 2
        lift = np.interp(t, np.arange(0.0, d + 0.9, 0.9), (rng.random(len(np.arange(0.0, d + 0.9, 0.9))) > 0.2).astype(float))
        tip = K.scrape(sr=P, dur=d, speed=0.06, force=0.25, roughness=0.5, texture="", material="wood", size=0.3, bright=0.8, seed=seed)
        y = amb.hip(_fitz(tip, n), 800.0) * strokes * lift
    return _fin(y, sr, 0.6, fade=0.002, tail=0.01)


ROPES = ("tension", "passage", "impact", "frottement")


@sfx("rope", "Rope: tension (fibres creaking under a growing load), passage (running through a block or a hand, whistle for a zip line), impact (slapping a surface), frottement (dragged over wood).",
     kind=("tension", " | ".join(ROPES)), whistle=(0.0, "passage: 0..1 zip-line whistle"), dur=(1.5, "s"), seed=(0, "variation"))
def rope(sr=DEFAULT_SR, kind="tension", whistle=0.0, dur=1.5, seed=0):
    """`creak(material=rope)`, a slap, the whistle as `cavity_tone`."""
    k = _kind(kind, ROPES, "rope")
    seed = int(seed)
    d = float(np.clip(dur, 0.15, 30.0))
    if k == "tension":
        y = _mix(d, (0.0, FO.creak(sr=P, material="rope", force="ramp", level=0.9, dur=d, seed=seed), 1.0),
                 (0.0, FO.cloth(sr=P, kind="rustle", dur=d, rate=2.0, weight=0.4, seed=seed), 0.1))
    elif k == "passage":
        n = _n(d)
        ramp = _env(n, (0.0, 0.0), (0.15 * d, 1.0), (0.8 * d, 1.0), (d, 0.0))
        run = _fitz(K.scrape(sr=P, dur=d, speed=1.2, force=0.7, roughness=0.8, texture="", material="wood", size=0.3, seed=seed), n)
        y = run * ramp + 0.3 * _fitz(FO.cloth(sr=P, kind="rustle", dur=d, rate=8.0, weight=0.3, seed=seed), n) * ramp
        wh = float(np.clip(whistle, 0.0, 1.0))
        if wh > 0:
            y = y + wh * _fitz(FO.cavity_tone(sr=P, speed=25.0, length=0.02, depth=0.02, width=0.01, dur=d, seed=seed), n) * ramp
    elif k == "impact":
        y = _mix(0.5, (0.0, FO.thud(sr=P, body="sack", contact_ms=8.0, size=0.5, force=0.6, seed=seed), 0.8),
                 (0.0, FO.cloth(sr=P, kind="flap", dur=0.12, rate=30.0, weight=0.6, seed=seed), 1.0),
                 (0.08, FO.creak(sr=P, material="rope", force="pull", level=0.4, dur=0.3, seed=seed), 0.15))
    else:
        y = _mix(d, (0.0, K.scrape(sr=P, dur=d, speed=0.3, force=1.0, roughness=0.9, texture="", material="wood", size=0.6, seed=seed), 1.0),
                 (0.0, FO.creak(sr=P, material="rope", force="sway", level=0.5, dur=d, seed=seed), 0.25))
    return _fin(y, sr, 0.7, fade=0.005, tail=0.03)


def _shear(seed: int, size: float = 0.15, snips: int = 1, gap: float = 0.28, target: str = "") -> np.ndarray:
    """Blades sliding on each other for 70 ms, the stop, 150 ms of opening."""
    rng = np.random.default_rng(seed)
    y = np.zeros(_n(0.3 + snips * gap))
    for i in range(int(snips)):
        t = i * gap
        _put(y, _z(K.scrape(sr=P, dur=0.07, speed=0.8, force=0.8, roughness=0.5, texture="", body="bar", material="steel", size=size, seed=seed + i)), t, 0.5)
        _put(y, _z(K.impact(sr=P, material="steel", shape="bar", size=size, velocity=1.5, hardness=0.8, modes=10, dur=0.08, seed=seed + i)), t + 0.07, 1.0)
        _put(y, _z(K.scrape(sr=P, dur=0.15, speed=0.3, force=0.3, roughness=0.3, texture="", body="bar", material="steel", size=size, seed=seed + i + 9)), t + 0.1, 0.15)
        if target == "paper":
            _put(y, _z(FO.paper(sr=P, kind="tear", dur=0.07, weight=0.4, seed=seed + i)), t, 0.3)
        elif target == "wire":
            _put(y, _snap(rng, 2600.0, 0.05, 1.0), t + 0.065, 0.7)
    return y


def _peel(rng, d: float, rate: float, screech: float) -> np.ndarray:
    """Tape peeling: stick-slip fronts at `rate` per second through the tape's sheet, ringing near 1.1 and 3 kHz."""
    n = _n(d)
    x = (rng.random(n) < np.minimum(rate * (0.7 + 0.6 * np.interp(np.arange(n), [0, n / 3, n], [0.2, 1.0, 0.7])) / P, 1.0)).astype(float)
    x *= rng.uniform(0.4, 1.0, n)
    return screech * (amb.bp(x, 3000.0, 2.0) + amb.bp(x, 1100.0, 3.0)) + amb.hip(x, 2000.0) * 0.4


@sfx("scissors", "Blades and tape: lames (a pair of scissors cutting paper), ruban (packing tape pulled off the roll in one long screech), adhesif (a strip of sticky tape: peeled, torn off, pressed down).",
     kind=("lames", "lames | ruban | adhesif"), snips=(3, "lames: cuts"), dur=(1.0, "ruban: s"), seed=(0, "variation"))
def scissors(sr=DEFAULT_SR, kind="lames", snips=3, dur=1.0, seed=0):
    """Scissors and tape."""
    k = _kind(kind, ("lames", "ruban", "adhesif"), "scissors")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    if k == "lames":
        y = _shear(seed, 0.15, int(np.clip(snips, 1, 30)), target="paper")
    elif k == "ruban":
        d = float(np.clip(dur, 0.2, 10.0))
        n = _n(d)
        y = np.zeros(n)
        _put(y, _z(_peel(rng, d, 1800.0, 1.0)) * _env(n, (0.0, 0.0), (0.05, 1.0), (d - 0.08, 1.0), (d, 0.0)), 0.0, 1.0)
        _put(y, _z(FO.velcro(sr=P, dur=d, speed=0.8, seed=seed)), 0.0, 0.2)
    else:
        n = _n(0.35)
        y = _mix(0.95, (0.0, _z(_peel(rng, 0.35, 700.0, 0.7)) * _env(n, (0.0, 0.0), (0.03, 1.0), (0.3, 1.0), (0.35, 0.0)), 0.8),
                 (0.38, FO.paper(sr=P, kind="tear", dur=0.05, weight=0.3, seed=seed), 0.8),
                 (0.55, FO.cloth(sr=P, kind="rustle", dur=0.25, rate=4.0, weight=0.2, seed=seed), 0.3))
    return _fin(y, sr, 0.7, fade=0.002, tail=0.02)


GARDEN = ("rateau", "beche", "secateur")


@sfx("garden_tool", "Garden tools: rateau (tines through leaves and soil, in strokes), beche (spade cutting in, soil falling), secateur (pruner: the blades and a twig).",
     kind=("rateau", " | ".join(GARDEN)), seed=(0, "variation"))
def garden_tool(sr=DEFAULT_SR, kind="rateau", seed=0):
    """Friction + ground."""
    k = _kind(kind, GARDEN, "garden_tool")
    seed = int(seed)
    if k == "rateau":
        ev = []
        for i in range(3):
            d = 0.55
            n = _n(d)
            e = _env(n, (0.0, 0.0), (0.05, 1.0), (0.45, 0.8), (d, 0.0))
            ev.append((i * 0.8, _fitz(K.scrape(sr=P, dur=d, speed=0.7, force=1.0, roughness=0.9, texture="", body="bar", material="metal", size=0.3, seed=seed + i), n) * e, 0.35))
            ev.append((i * 0.8, _fitz(FO.shake(sr=P, preset="Sekere", energy=0.6, rate=6.0, dur=d, seed=seed + i), n) * e, 0.7))
        y = _mix(2.6, *ev)
    elif k == "beche":
        y = _mix(1.3, (0.0, K.impact(sr=P, material="metal", shape="rect_plate", size=0.22, velocity=2.0, hardness=0.8, modes=12, dur=0.15, seed=seed), 0.35),
                 (0.0, FO.shake(sr=P, preset="Crunch", energy=0.9, rate=9.0, dur=0.25, seed=seed), 1.0),
                 (0.45, FO.thud(sr=P, body="sack", contact_ms=15.0, size=1.2, force=0.6, seed=seed), 0.6),
                 (0.47, FO.shake(sr=P, preset="LittleRocks", energy=0.4, rate=7.0, dur=0.6, seed=seed), 0.4))
    else:
        y = _shear(seed, 0.11, 1, target="wire")
        y = np.concatenate((y, _z(_shear(seed + 3, 0.11, 1)) * 0.6))
    return _fin(y, sr, 0.7, fade=0.002, tail=0.02)


TOOLS = ("lime", "ciseaux", "tenaille", "tournevis", "ponceuse_main")


@sfx("tool", "Hand tools: lime (file, push strokes), ciseaux (tin snips), tenaille (pincers: bite, a nail squealing out), tournevis (ratcheting screwdriver, the screw seating), ponceuse_main (sanding block, back and forth).",
     kind=("lime", " | ".join(TOOLS)), strokes=(4, "lime, ponceuse_main: strokes"), rate=(2.0, "lime, ponceuse_main: strokes per second"), seed=(0, "variation"))
def tool(sr=DEFAULT_SR, kind="lime", strokes=4, rate=2.0, seed=0):
    """Periodic friction, small impact. `ciseaux`: tin snips."""
    k = _kind(kind, TOOLS, "tool")
    seed = int(seed)
    if k in ("lime", "ponceuse_main"):
        c = int(np.clip(strokes, 1, 40))
        r = float(np.clip(rate, 0.3, 12.0))
        d = c / r
        n = _n(d)
        t = np.arange(n) / P
        if k == "lime":                                     # cuts on the push only
            e = np.clip(np.sin(2 * np.pi * r * t), 0.0, 1.0) ** 0.5
            x = K.scrape(sr=P, dur=d, speed=0.5, force=1.0, roughness=0.95, texture="", body="bar", material="steel", size=0.25, seed=seed)
        else:                                               # cuts both ways
            e = np.abs(np.sin(2 * np.pi * r * t)) ** 0.7
            x = FO.shake(sr=P, preset="Sandpaper", energy=0.8, rate=6.0, dur=d, seed=seed)
        y = _fitz(x, n) * e
    elif k == "ciseaux":
        y = _shear(seed, 0.2, 2, 0.34, "wire")
    elif k == "tenaille":
        n = _n(0.7)
        y = _mix(1.1, (0.0, _shear(seed, 0.22, 1), 0.7),
                 (0.3, _fitz(squeak(sr=P, surface="hinge", pressure=2.0, pitch=1.6, dur=0.7, seed=seed), n) * _env(n, (0.0, 0.0), (0.1, 1.0), (0.6, 1.0), (0.7, 0.0)), 0.6),
                 (0.95, K.impact(sr=P, material="steel", shape="bar", size=0.05, velocity=1.5, hardness=0.8, modes=8, dur=0.1, seed=seed), 0.3))
    else:
        y = _mix(1.5, (0.0, FO.ratchet_wrench(sr=P, strokes=2, rate=1.6, clicks=7, size=0.6, seed=seed), 1.0),
                 (1.2, _ticks(np.random.default_rng(seed), [0.0], ENDSTOP, 0.2), 0.5),
                 (0.0, K.scrape(sr=P, dur=1.2, speed=0.1, force=0.7, roughness=0.5, texture="", material="wood", size=0.3, seed=seed), 0.1))
    return _fin(y, sr, 0.7, fade=0.003, tail=0.02)


# =============================================================================================
# MONEY, CARDS, GAMES, FURNITURE, LUGGAGE, LEATHER
# =============================================================================================
@sfx("coins", "Coins: chute (one coin bounces and spins out), tas (a pile dumped and settling), comptage (coins counted down on a table, one by one).",
     kind=("chute", "chute | tas | comptage"), count=(8, "comptage: coins"), rate=(4.0, "comptage: coins per second"), seed=(0, "variation"))
def coins(sr=DEFAULT_SR, kind="chute", count=8, rate=4.0, seed=0):
    """PhISEM + bounce."""
    k = _kind(kind, ("chute", "tas", "comptage"), "coins")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    if k == "chute":
        hit = K.bounce(sr=P, material="metal", size=0.012, shape="ball", surface="wood", first_interval=0.12, restitution=0.6, seed=seed)
        y = np.zeros(_n(len(hit) / P + 0.9))
        _put(y, _z(hit), 0.0, 1.0)
        _put(y, _z(FO.coin_spin(sr=P, coin="quarter", dur=0.9, start=14.0, seed=seed)), len(hit) / P * 0.5, 0.5)
    elif k == "tas":
        y = _mix(1.5, (0.0, FO.shake(sr=P, preset="QuarterMug", energy=0.9, rate=9.0, dur=0.8, seed=seed), 1.0),
                 (0.0, FO.shake(sr=P, preset="PennyMug", energy=0.8, rate=11.0, dur=1.1, seed=seed + 1), 0.7),
                 (0.5, FO.casing(sr=P, count=8, size=0.6, spread=0.5, floor="wood", height=0.1, seed=seed), 0.25))
    else:
        c = int(np.clip(count, 1, 60))
        gap = 1.0 / float(np.clip(rate, 0.5, 15.0))
        y = np.zeros(_n(c * gap + 0.4))
        for i in range(c):
            t = i * gap * rng.uniform(0.95, 1.05)
            _put(y, _z(FO.clink(sr=P, object="coin", size=float(rng.uniform(0.9, 1.1)), force=0.5, seed=seed + i)), t, 0.8)
            _put(y, _z(K.impact(sr=P, material="metal", shape="rect_plate", size=0.025, velocity=0.8, hardness=0.8, modes=8, dur=0.08, seed=seed + i)), t, 0.5)
            _put(y, _z(FO.thud(sr=P, body="crate", contact_ms=3.0, size=0.25, force=0.3, seed=seed + i)), t, 0.15)
    return _fin(y, sr, 0.7, fade=0.002, tail=0.02)


CARD_ACTIONS = ("mele", "bridge", "donne", "retourne")


def _card_train(rng, times, gains, slap: float = 0.5) -> np.ndarray:
    """One 3 ms slip and one 2-3 kHz tick per card."""
    y = np.zeros(_n(float(np.max(times)) + 0.05))
    m = _n(0.003)
    burst = amb.hip(_noise(rng, m), 1500.0) * np.hanning(m)
    for t, g in zip(times, gains):
        i = int(t * P)
        y[i:i + m] += g * burst[: len(y[i:i + m])]
    return y + slap * amb.bp(y, 2500.0, 2.0)


@sfx("cards", "Playing cards: mele (overhand shuffle, cards falling at 30-50 per second), bridge (riffle, then the deck squared), donne (dealing: flick and landing), retourne (one card flipped).",
     action=("mele", " | ".join(CARD_ACTIONS)), rate=(40.0, "mele, bridge: cards per second, 30..50"), cards=(52, "cards in the riffle"), seed=(0, "variation"))
def cards(sr=DEFAULT_SR, action="mele", rate=40.0, cards=52, seed=0):
    """Friction and fast slapping, 30-50 cards/s."""
    a = _kind(action, CARD_ACTIONS, "cards")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    r = float(np.clip(rate, 5.0, 80.0))
    c = int(np.clip(cards, 4, 104))
    if a == "mele":
        y = np.zeros(_n(0.6 * 4 + 0.2))
        for i in range(4):                                  # four packets, each let go in a run
            t = np.cumsum(np.full(int(0.4 * r), 1.0 / r))
            _put(y, _z(_card_train(rng, t, rng.uniform(0.5, 1.0, len(t)))), 0.6 * i, 0.8)
            _put(y, _z(FO.paper(sr=P, kind="page", dur=0.15, weight=0.5, seed=seed + i)), 0.6 * i + 0.4, 0.25)
    elif a == "bridge":
        t = np.cumsum(np.full(c, 1.0 / r))
        d = float(t[-1])
        y = np.zeros(_n(d + 0.6))
        _put(y, _z(_card_train(rng, t, rng.uniform(0.5, 1.0, c))), 0.0, 0.9)
        _put(y, _z(FO.paper(sr=P, kind="page", dur=0.3, weight=0.8, seed=seed)), d + 0.05, 0.4)
        _put(y, _z(FO.thud(sr=P, body="crate", contact_ms=5.0, size=0.35, force=0.4, seed=seed)), d + 0.3, 0.4)
    elif a == "donne":
        y = np.zeros(_n(1.7))
        for i in range(4):
            _put(y, _z(FO.paper(sr=P, kind="page", dur=0.07, weight=0.4, seed=seed + i)), 0.4 * i, 0.5)
            _put(y, _z(FO.thud(sr=P, body="crate", contact_ms=3.0, size=0.25, force=0.2, seed=seed + i)), 0.4 * i + 0.14, 0.5)
    else:
        y = _mix(0.3, (0.0, FO.paper(sr=P, kind="page", dur=0.1, weight=0.4, seed=seed), 0.6),
                 (0.11, FO.thud(sr=P, body="crate", contact_ms=3.0, size=0.25, force=0.3, seed=seed), 0.6))
    return _fin(y, sr, 0.7, fade=0.002, tail=0.02)


@sfx("board_game", "Board games: pion (a piece slid and set down), carte (a card flicked on the table), jeton (plastic chips stacked).",
     kind=("pion", "pion | carte | jeton"), seed=(0, "variation"))
def board_game(sr=DEFAULT_SR, kind="pion", seed=0):
    """Small impact + card."""
    k = _kind(kind, ("pion", "carte", "jeton"), "board_game")
    seed = int(seed)
    if k == "pion":
        y = _mix(0.4, (0.0, K.scrape(sr=P, dur=0.1, speed=0.2, force=0.3, roughness=0.3, texture="", material="wood", size=0.4, seed=seed), 0.25),
                 (0.11, K.impact(sr=P, material="wood", shape="bar", size=0.04, velocity=1.0, hardness=0.6, modes=8, dur=0.08, seed=seed), 1.0),
                 (0.11, K.impact(sr=P, material="wood", shape="rect_plate", size=0.4, velocity=0.7, hardness=0.4, modes=10, dur=0.15, seed=seed), 0.5))
    elif k == "carte":
        y = _mix(0.3, (0.0, FO.paper(sr=P, kind="page", dur=0.06, weight=0.3, seed=seed), 0.6),
                 (0.08, FO.thud(sr=P, body="crate", contact_ms=2.0, size=0.25, force=0.3, seed=seed), 0.7))
    else:
        y = np.zeros(_n(0.55))
        for i, t in enumerate((0.0, 0.13, 0.24, 0.32)):
            _put(y, _z(K.impact(sr=P, material="plastic", shape="rect_plate", size=0.035, velocity=1.2, hardness=0.7, modes=8, dur=0.07, seed=seed + i)), t, 1.0 - 0.15 * i)
    return _fin(y, sr, 0.7, fade=0.002, tail=0.01)


FURNITURE = ("chaise_qui_racle", "canape", "armoire")


@sfx("furniture", "Furniture: chaise_qui_racle (chair legs dragged on a floor, chattering), canape (someone sits: springs, fabric, a soft landing), armoire (heavy door creaking, then its thud).",
     kind=("chaise_qui_racle", " | ".join(FURNITURE)), seed=(0, "variation"))
def furniture(sr=DEFAULT_SR, kind="chaise_qui_racle", seed=0):
    """Wood friction + creak."""
    k = _kind(kind, FURNITURE, "furniture")
    seed = int(seed)
    if k == "chaise_qui_racle":
        y = _mix(1.05, (0.0, K.scrape(sr=P, dur=0.8, speed=0.5, force=1.4, roughness=0.8, texture="", material="wood", size=0.5, seed=seed), 0.8),
                 (0.0, squeak(sr=P, surface="shoe", pressure=1.5, pitch=0.7, dur=0.8, seed=seed), 0.4),
                 (0.78, FO.thud(sr=P, body="crate", contact_ms=8.0, size=0.7, force=0.5, seed=seed), 0.6))
    elif k == "canape":
        y = _mix(1.3, (0.0, FO.thud(sr=P, body="sack", contact_ms=35.0, size=1.5, force=0.7, seed=seed), 1.0),
                 (0.02, FO.cloth(sr=P, kind="rustle", dur=0.5, rate=3.0, weight=0.7, seed=seed), 0.3),
                 (0.08, FO.creak(sr=P, material="wood", force="sway", level=0.5, dur=0.9, size=1.2, seed=seed), 0.3),
                 (0.06, K.impact(sr=P, material="steel", shape="bar", size=0.5, velocity=0.4, hardness=0.3, modes=6, dur=0.5, seed=seed), 0.06))
    else:
        y = _mix(2.0, (0.0, FO.creak(sr=P, material="wood", force="pull", level=0.7, dur=1.3, size=1.4, seed=seed), 0.7),
                 (1.25, FO.thud(sr=P, body="crate", contact_ms=12.0, size=1.6, force=0.7, seed=seed), 1.0),
                 (1.25, _ticks(np.random.default_rng(seed), [0.02], ENDSTOP, 0.2), 0.3))
    return _fin(y, sr, 0.7, fade=0.003, tail=0.02)


@sfx("bag", "Bags and luggage: sac_plastique (a plastic bag crumpled), papier (a paper bag), valise (suitcase: zip, latches, set down), roulettes (trolley case rolling over joints).",
     kind=("sac_plastique", "sac_plastique | papier | valise | roulettes"), dur=(1.5, "s, except valise"), speed=(0.9, "roulettes: m/s"), seed=(0, "variation"))
def bag(sr=DEFAULT_SR, kind="sac_plastique", dur=1.5, speed=0.9, seed=0):
    """Crumple, cloth, roll, zipper; 0.6 m floor tiles under the trolley case."""
    k = _kind(kind, ("sac_plastique", "papier", "valise", "roulettes"), "bag")
    seed = int(seed)
    d = float(np.clip(dur, 0.2, 20.0))
    if k == "sac_plastique":
        y = _mix(d + 0.1, (0.0, K.crumple(sr=P, kind="bag", energy=0.6, density=0.6, size=0.5, dur=d, seed=seed), 1.0))
    elif k == "papier":
        y = _mix(d + 0.1, (0.0, K.crumple(sr=P, kind="paper", energy=0.5, density=0.4, size=0.6, dur=d, seed=seed), 1.0),
                 (0.0, FO.paper(sr=P, kind="rustle", dur=d, weight=0.8, seed=seed), 0.3))
    elif k == "valise":
        y = _mix(1.85, (0.0, FO.zipper(sr=P, dur=0.7, seed=seed), 0.7),
                 (0.85, solenoid(sr=P, size=0.25, metal=0.8, thump=0.3, bounce=0.2, seed=seed), 0.8),
                 (1.05, solenoid(sr=P, size=0.25, metal=0.8, thump=0.3, bounce=0.2, seed=seed + 1), 0.8),
                 (1.35, FO.thud(sr=P, body="box", contact_ms=12.0, size=1.2, force=0.6, seed=seed), 1.0))
    else:
        v = float(np.clip(speed, 0.2, 3.0))
        n = _n(d)
        y = np.zeros(n)
        _put(y, _z(K.roll(sr=P, dur=d, speed=v, radius=0.025, roughness=0.6, material="plastic", surface="concrete", surface_size=2.0, seed=seed)), 0.0, 0.6)
        for i, t in enumerate(np.arange(0.2, d, 0.6 / v)):  # the two pairs of wheels over each joint
            _put(y, _z(K.impact(sr=P, material="plastic", shape="rect_plate", size=0.2, velocity=1.5, hardness=0.5, modes=8, dur=0.1, seed=seed + i)), float(t), 0.8)
            _put(y, _z(K.impact(sr=P, material="plastic", shape="rect_plate", size=0.2, velocity=1.2, hardness=0.5, modes=8, dur=0.1, seed=seed + i + 7)), float(t) + 0.04, 0.6)
    return _fin(y, sr, 0.7, fade=0.005, tail=0.02)


CARTS = {"caddie": ("steel", "concrete", 0.05, 0.5, 0.7), "chariot": ("plastic", "wood", 0.04, 0.35, 0.3), "charrette": ("wood", "stone", 0.35, 0.9, 0.0)}    # wheel, ground, radius m, joints, rattle


@sfx("cart", "Wheeled carts: caddie (shopping trolley: wire basket rattling, a wheel squeaking), chariot (light plastic trolley), charrette (wooden cart: big wheels on stone, creaking).",
     kind=("caddie", " | ".join(CARTS)), speed=(1.0, "m/s"), dur=(3.0, "s"), seed=(0, "variation"))
def cart(sr=DEFAULT_SR, kind="caddie", speed=1.0, dur=3.0, seed=0):
    """`roll(joints=)` + rattle + squeak once a turn."""
    k = _kind(kind, CARTS, "cart")
    wheel, ground, radius, joints, rattle = CARTS[k]
    d = float(np.clip(dur, 0.3, 30.0))
    v = float(np.clip(speed, 0.1, 5.0))
    n = _n(d)
    turn = v / (2 * math.pi * radius)
    t = np.arange(n) / P
    ev = [(0.0, K.roll(sr=P, dur=d, speed=v, radius=radius, roughness=0.7, material=wheel, surface=ground, surface_size=2.0, joints=joints, seed=seed), 1.0)]
    if rattle > 0:
        ev.append((0.0, FO.shake(sr=P, preset="Sticks" if k == "chariot" else "Wrench", energy=0.5 + 0.3 * rattle, rate=6.0 * v, dur=d, seed=seed), rattle * 0.6))
    if k == "charrette":
        ev.append((0.0, FO.creak(sr=P, material="wood", force="sway", level=0.6, dur=d, size=1.5, seed=seed), 0.5))
    else:
        gate = np.clip(np.sin(2 * np.pi * turn * t), 0.0, 1.0) ** 3
        ev.append((0.0, _fitz(squeak(sr=P, surface="hinge", pressure=1.0, pitch=1.5, dur=d, seed=seed), n) * gate, 0.25))
    return _fin(_mix(d, *ev), sr, 0.7, fade=0.02, tail=0.05)


@sfx("leather", "Leather: bag (a bag handled), belt (a belt pulled through and tightened), mouvement (a jacket moving), impact (slapped).",
     action=("bag", "bag | belt | mouvement | impact"), seed=(0, "variation"))
def leather(sr=DEFAULT_SR, action="bag", seed=0):
    """Creak + rustle."""
    a = _kind(action, ("bag", "belt", "mouvement", "impact"), "leather")
    seed = int(seed)
    if a == "bag":
        y = _mix(1.2, (0.0, FO.creak(sr=P, material="leather", force="sway", level=0.7, dur=1.1, seed=seed), 0.8),
                 (0.0, K.crumple(sr=P, kind="bag", energy=0.2, density=0.3, size=0.8, dur=1.1, bright=0.2, seed=seed), 0.25),
                 (0.3, FO.keys(sr=P, hits=1, seed=seed), 0.05))
    elif a == "belt":
        y = _mix(1.1, (0.0, K.scrape(sr=P, dur=0.6, speed=0.8, force=0.5, roughness=0.4, texture="", material="wood", size=0.3, seed=seed), 0.4),
                 (0.55, FO.creak(sr=P, material="leather", force="ramp", level=0.9, dur=0.35, seed=seed), 0.8),
                 (0.9, K.impact(sr=P, material="metal", shape="bar", size=0.05, velocity=1.2, hardness=0.8, modes=8, dur=0.1, seed=seed), 0.5))
    elif a == "mouvement":
        y = _mix(1.5, (0.0, FO.creak(sr=P, material="leather", force="sway", level=0.5, dur=1.4, seed=seed), 0.6),
                 (0.0, FO.cloth(sr=P, kind="rustle", dur=1.4, rate=2.5, weight=0.8, seed=seed), 0.6))
    else:
        y = _mix(0.4, (0.0, FO.thud(sr=P, body="sack", contact_ms=5.0, size=0.5, force=0.7, seed=seed), 0.8),
                 (0.0, FO.cloth(sr=P, kind="flap", dur=0.08, rate=30.0, weight=0.7, seed=seed), 1.0))
    return _fin(y, sr, 0.7, fade=0.002, tail=0.02)


# =============================================================================================
# BELLS AND BEEPS
# =============================================================================================
def bell_freq(size) -> float:
    """Gong 800-2000 Hz: size 0 is the small one."""
    return 2000.0 - 1200.0 * float(np.clip(size, 0.0, 1.0))


@sfx("alarm_bell", "School or fire bell: an electric hammer on a large gong, like a telephone ringer at 28.6 strikes per second.",
     size=(0.5, "0 small gong 2000 Hz .. 1 large 800 Hz"), rate=(28.6, "strikes per second"), dur=(2.0, "s"), seed=(0, "variation"))
def alarm_bell(sr=DEFAULT_SR, size=0.5, rate=28.6, dur=2.0, seed=0):
    """A telephone ringer's 28.6 strikes/s on the gong; beating proportional to pitch."""
    f = bell_freq(size)
    y = FO.phone_bell(sr=P, freq=f, detune=f * 3.0 / 650.0, decay=0.6 + 1.2 * float(np.clip(size, 0.0, 1.0)), rate=rate, bursts=1, ring=float(np.clip(dur, 0.1, 30.0)), gap=0.0, bright=1.5, seed=seed)
    return _fin(y, sr, 0.8, fade=0.002, tail=0.02)


SMOKE_TONES = {"piezo": (3100.0, "sine"), "low": (520.0, "square")}
# on-times s, on length s, cycle s. T3: 3 x 0.5 s, 1.5 s pause. T4 (CO): 4 x 0.1 s on, 0.1 s off, 5 s pause.
SMOKE_PATTERNS = {"t3": ((0.0, 1.0, 2.0), 0.5, 4.0), "t4": ((0.0, 0.2, 0.4, 0.6), 0.1, 5.7)}


@sfx("smoke_alarm", "Smoke alarm: narrow-band piezo in the temporal pattern. t3: three 0.5 s beeps, 0.5 s apart, 1.5 s pause, 4 s cycle, at 3.1 kHz or a 520 Hz square; t4: the CO pattern.",
     pattern=("t3", "t3 | t4"), tone=("piezo", "piezo 3100 Hz | low 520 Hz square"), cycles=(2, "cycles"))
def smoke_alarm(sr=DEFAULT_SR, pattern="t3", tone="piezo", cycles=2):
    """Temporal pattern, 3 ms gate edges; the 520 Hz square is an odd series up to a quarter of the rate."""
    p, t = _kind(pattern, SMOKE_PATTERNS, "smoke_alarm pattern"), _kind(tone, SMOKE_TONES, "smoke_alarm tone")
    ons, width, cycle = SMOKE_PATTERNS[p]
    f, wave = SMOKE_TONES[t]
    c = int(np.clip(cycles, 1, 30))
    n = _n(c * cycle)
    tt = np.arange(n) / P
    if wave == "sine":
        x = np.sin(2 * np.pi * f * tt)
    else:
        x = sum(np.sin(2 * np.pi * f * h * tt) / h for h in range(1, int(0.25 * P / f), 2))
    ph = tt % cycle
    edge = 0.003
    gate = sum(np.clip((ph - on) / edge, 0.0, 1.0) * np.clip((on + width - ph) / edge, 0.0, 1.0) for on in ons)
    return _fin(x * gate, sr, 0.7, fade=0.001)


CLOCK_BEEPS = 4                                             # beeps per group
CLOCK_RATE = 8.0                                            # beeps per second


def _piezo(freq: float, times, width: float, dur: float) -> np.ndarray:
    """Square drive through a piezo, gated with 2 ms edges: the third harmonic at 1/9."""
    n = _n(dur)
    t = np.arange(n) / P
    x = np.sin(2 * np.pi * freq * t) + np.sin(6 * np.pi * freq * t) / 9.0 * (3.0 * freq < 0.45 * P)
    gate = np.zeros(n)
    for t0 in times:
        gate += np.clip((t - t0) / 0.002, 0.0, 1.0) * np.clip((t0 + width - t) / 0.002, 0.0, 1.0)
    return x * gate


@sfx("alarm_clock", "Alarm clocks: mecanique (hammer rattling between two bells on top), numerique (digital: 4 beeps at 8 per second, in groups).",
     kind=("mecanique", "mecanique | numerique"), freq=(3000.0, "numerique: Hz, 2000..4000"), groups=(3, "rings or groups"), seed=(0, "variation"))
def alarm_clock(sr=DEFAULT_SR, kind="mecanique", freq=3000.0, groups=3, seed=0):
    """Double bell (1800 and 1980 Hz, 18 strikes/s) or beep (2-4 kHz, 4 beeps at 8/s)."""
    k = _kind(kind, ("mecanique", "numerique"), "alarm_clock")
    g = int(np.clip(groups, 1, 30))
    if k == "mecanique":
        return _fin(FO.phone_bell(sr=P, freq=1800.0, detune=180.0, decay=0.5, rate=18.0, bursts=g, ring=0.7, gap=0.15, bright=1.5, seed=seed), sr, 0.8, fade=0.002, tail=0.02)
    f = float(np.clip(freq, 2000.0, 4000.0))
    period = 1.0
    times = [i * period + b / CLOCK_RATE for i in range(g) for b in range(CLOCK_BEEPS)]
    return _fin(_piezo(f, times, 0.5 / CLOCK_RATE, g * period), sr, 0.7, fade=0.001)


# freq Hz, beeps, on s, off s
APPLIANCE_BEEPS = {"micro_ondes": (2700.0, 3, 0.25, 0.25), "lave_linge": (3200.0, 5, 0.5, 0.5), "four": (2400.0, 4, 0.2, 0.1), "touche": (3500.0, 1, 0.06, 0.0), "telephone": (3800.0, 2, 0.1, 0.1)}


@sfx("appliance_beep", "Piezo beeps of appliances, 2-4 kHz: micro_ondes (end of cycle), lave_linge, four, touche (a key press), telephone.",
     kind=("micro_ondes", " | ".join(APPLIANCE_BEEPS)))
def appliance_beep(sr=DEFAULT_SR, kind="micro_ondes"):
    f, c, on, off = APPLIANCE_BEEPS[_kind(kind, APPLIANCE_BEEPS, "appliance_beep")]
    return _fin(_piezo(f, [i * (on + off) for i in range(c)], on, c * (on + off) + 0.01), sr, 0.6, fade=0.001)


# =============================================================================================
# FANS AND MOTORS
# =============================================================================================
def _fan_pipe(x: np.ndarray) -> np.ndarray:
    """The duct: delays 22 ms and 70 ms, feedback .499 and .3, in series."""
    for ms, fb in ((22.0, 0.499), (70.0, 0.3)):
        d = int(round(ms * 1e-3 * P))
        a = np.zeros(d + 1)
        a[0], a[d] = 1.0, -fb
        x = lfilter([1.0], a, x)
    return x


@sfx("ventilation", "Ventilation unit: motor, fan and duct: speed 22, motor ratio 20, duct delays 22 and 70 ms. shutdown winds it down through a 0.1 Hz lag.",
     speed=(22.0, "main speed, turns per second"), shutdown=(-1.0, "s when the unit is switched off (-1 = never)"), dur=(4.0, "s"), seed=(0, "variation"))
def ventilation(sr=DEFAULT_SR, speed=22.0, shutdown=-1.0, dur=4.0, seed=0):
    """Levels: motor .2, brush .05, rotor .1, stator .3, pulse width 4, fan noise .2, fan .5, noise band 700 Hz Q 1, stop by a 0.1 Hz lag.
    Motor as `foley._motor`, without its housing. Also the fan of `appliance frigo`."""
    d = float(np.clip(dur, 0.2, 60.0))
    n = _n(d)
    s = float(np.clip(speed, 1.0, 200.0))
    rng = np.random.default_rng(int(seed))
    main = np.full(n, s)
    if float(shutdown) >= 0:
        r = math.exp(-2 * math.pi * 0.1 / P)
        off = _n(min(float(shutdown), d))
        main[off:] = s * r ** np.arange(n - off)            # lop~ 0.1 with its target at 0
    ph = np.cumsum(main) / P
    mph = 20.0 * ph
    brush = amb.bp(_noise(rng, n), 4000.0, 1.0)
    motor = 0.2 * (mph % 1.0) ** 4 * (0.05 * brush + 0.1) + 0.2 * 0.3 / ((2.0 * np.cos(2 * np.pi * mph)) ** 2 + 1.0)
    pulse = 1.0 / (1.0 + (4.0 * np.cos(2 * np.pi * ph)) ** 2)
    blades = 0.5 * pulse * (1.0 + 0.2 * amb.bp(_noise(rng, n), 700.0, 1.0))
    x = amb.hip(motor + blades, 25.0)
    return _fin(_fan_pipe(x * np.sqrt(main / s)), sr, 0.7, fade=0.02, tail=0.05)


@sfx("vacuum", "Vacuum cleaner: centrifugal fan (BPF = blades x rpm / 60, 4000 Hz at 12 blades and 20 000 rpm, with 8000 and 12 000), a broadband bed dominant near 600 Hz, suction hiss. clog: the note of a blocked hose, the motor racing.",
     rpm=(20000.0, "rpm"), blades=(12, "count"), clog=(0.0, "0 free .. 1 blocked nozzle"), dur=(3.0, "s"), seed=(0, "variation"))
def vacuum(sr=DEFAULT_SR, rpm=20000.0, blades=12, clog=0.0, dur=3.0, seed=0):
    """BPF and harmonics; a broadband bed dominant at 600 Hz, 73.2 dB(A) at nominal (BPF 2475 Hz at rpm=12375).
    Blocked: the motor unloads and races, +15 % rpm. Level with rpm: the fan law."""
    d = float(np.clip(dur, 0.2, 60.0))
    n = _n(d)
    c = float(np.clip(clog, 0.0, 1.0))
    r = float(np.clip(rpm, 1000.0, 60000.0)) * (1.0 + 0.15 * c)
    z = max(int(blades), 1)
    rng = np.random.default_rng(int(seed))
    t = np.arange(n) / P
    bpf = z * r / 60.0
    tones = sum(g * np.sin(2 * np.pi * h * bpf * t + rng.uniform(0, 6.28)) for h, g in ((1, 1.0), (2, 0.4), (3, 0.2)) if h * bpf < 0.45 * P)
    bed = amb.bp(rng.standard_normal(n), 600.0, 1.5)
    bed = bed / (np.sqrt(np.mean(bed * bed)) + 1e-12)
    hiss = _fitz(air_leak(sr=P, dur=d, pressure=0.6 * (1.0 - 0.6 * c), orifice=30.0, seed=seed), n)
    mot = _fitz(electric_motor(sr=P, dur=d, rpm=r, load=0.6 + 0.3 * c, seed=seed), n)
    y = 2.5 * bed + 0.7 * tones + amb.hip(0.25 * mot + 0.8 * hiss * (1.0 - 0.6 * c), 500.0)
    if c > 0:
        y = y + 1.2 * c * _fitz(FO.cavity_tone(sr=P, speed=40.0, length=0.02, depth=0.02, width=0.03, dur=d, seed=seed), n)
    y = _to_dba(y, DB_REF + fan_law_db(float(rpm) / 20000.0))
    return _fin(y, sr, None, fade=0.03, tail=0.1)


DRYER_RPM = 20000.0
DRYER_BLADES = 7
DRYER_DB = 68.0                                             # dB(A)


@sfx("hair_dryer", "Hair dryer: small fan and hot jet, 500-8000 Hz, mostly 630-6300 Hz. BPF = rpm x blades / 60. 40 % more speed = 7 dB more (6-8 dBA measured).",
     speed=(1.0, "x nominal; 1.4 is the 40 % step"), dur=(3.0, "s"), seed=(0, "variation"))
def hair_dryer(sr=DEFAULT_SR, speed=1.0, dur=3.0, seed=0):
    """Levels as in `vacuum`: 68 dB(A) sits 5.2 dB under it."""
    d = float(np.clip(dur, 0.2, 60.0))
    n = _n(d)
    s = float(np.clip(speed, 0.2, 2.0))
    rng = np.random.default_rng(int(seed))
    t = np.arange(n) / P
    bpf = DRYER_BLADES * DRYER_RPM * s / 60.0
    x = rng.standard_normal(n)
    jet = _band(x, 630.0, 6300.0, order=3) + 0.2 * _band(x, 500.0, 8000.0, order=3)
    jet = jet / (np.sqrt(np.mean(jet * jet)) + 1e-12)
    hiss = _fitz(steam(sr=P, dur=d, pressure=0.4, orifice=30.0, wet=0.0, tone=3000.0, seed=seed), n)
    mot = _fitz(electric_motor(sr=P, dur=d, rpm=DRYER_RPM * s, load=0.5, seed=seed), n)
    y = jet + 0.5 * np.sin(2 * np.pi * bpf * t) + 0.15 * _band(hiss + 0.3 * mot, 500.0, 8000.0, order=3)
    y = _band(y, 500.0, 8000.0, order=4)
    return _fin(_to_dba(y, DRYER_DB + fan_law_db(s)), sr, None, fade=0.03, tail=0.1)


BLENDER_RPM = 20000.0
BLENDER_TEETH = (17, 31)


@sfx("blender", "Kitchen blender: universal motor, gear lines, a dominant broadband roar, the contents thrown about; the motor drags under load.",
     speed=(1.0, "x nominal 20 000 rpm"), load=(0.5, "0 empty .. 1 full: drop of speed, contents louder"), dur=(3.0, "s"), seed=(0, "variation"))
def blender(sr=DEFAULT_SR, speed=1.0, load=0.5, dur=3.0, seed=0):
    """Broadband dominant (1200 Hz bed), with rotation and gear lines; 20 % slower at full load."""
    d = float(np.clip(dur, 0.2, 60.0))
    n = _n(d)
    ld = float(np.clip(load, 0.0, 1.0))
    r = blender_rpm(speed, load)
    rng = np.random.default_rng(int(seed))
    wash = amb.bp(rng.standard_normal(n), 1200.0, 0.6)
    roar = _nm(wash)
    mot = _fitz(electric_motor(sr=P, dur=d, rpm=r, load=0.5 + 0.5 * ld, startup=0.1, seed=seed), n)
    mesh = _fitz(gearbox(sr=P, dur=d, rpm=r, teeth=BLENDER_TEETH[0], teeth2=BLENDER_TEETH[1], load=0.5 + 0.5 * ld, wear=0.2, housing=2200.0, seed=seed), n)
    rot = np.sin(2 * np.pi * r / 60.0 * np.arange(n) / P)
    bits = _fitz(FO.shake(sr=P, preset="BigRocks", energy=0.5 + 0.5 * ld, rate=14.0, dur=d, seed=seed), n)
    y = roar + 0.15 * rot + 0.25 * mot + 0.15 * mesh + (0.1 + 0.6 * ld) * bits
    return _fin(y, sr, 0.8, fade=0.05, tail=0.1)


def blender_rpm(speed, load) -> float:
    return BLENDER_RPM * float(np.clip(speed, 0.1, 1.5)) * (1.0 - 0.2 * float(np.clip(load, 0.0, 1.0)))


SHAVER_HZ = {"rasoir": None, "tondeuse": None, "brosse_a_dents": 260.0}   # None: twice the mains
SHAVER_MODES = {"rasoir": (1800.0, 3200.0), "tondeuse": (900.0, 2400.0), "brosse_a_dents": (1500.0, 3800.0)}   # plastic shell


@sfx("shaver", "Vibrating armature in a plastic shell: rasoir, tondeuse at twice the mains (100 or 120 Hz), brosse_a_dents at 260 Hz.",
     kind=("rasoir", " | ".join(SHAVER_HZ)), mains=(50.0, "Hz: 50 | 60"), dur=(2.0, "s"), seed=(0, "variation"))
def shaver(sr=DEFAULT_SR, kind="rasoir", mains=50.0, dur=2.0, seed=0):
    """`buzzer` rattle + shell, hair crackle for the razor and clipper."""
    k = _kind(kind, SHAVER_HZ, "shaver")
    f = SHAVER_HZ[k] or 2.0 * float(np.clip(mains, 40.0, 70.0))
    d = float(np.clip(dur, 0.1, 30.0))
    n = _n(d)
    rng = np.random.default_rng(int(seed))
    t = np.arange(n) / P
    b = _fitz(FO.buzzer(sr=P, dur=d, freq=f, harsh=0.5 if k == "brosse_a_dents" else 0.8), n)
    b = b + 0.6 * np.sin(2 * np.pi * f * t)
    y = b + sum(0.4 * amb.bp(b, m, 6.0) for m in SHAVER_MODES[k])
    if k != "brosse_a_dents":
        y = y + 0.1 * amb.hip(rng.standard_normal(n), 3000.0) * (0.5 + 0.5 * np.sin(2 * np.pi * f * t))
    return _fin(y, sr, 0.7, fade=0.01, tail=0.02)


# =============================================================================================
# APPLIANCES
# =============================================================================================
DRUM_RPM = 55.0                                             # 50-60 rpm: the middle
COMP_SLIP = 1.0 - 2900.0 / 3000.0                           # two-pole induction motor: 2900 rpm on 50 Hz mains
APPLIANCES = ("micro_ondes", "frigo", "lave_linge", "lave_vaisselle", "seche_linge")


def _lump(x: np.ndarray, rpm: float, sharp: float = 3.0) -> np.ndarray:
    """Once a turn, the load goes over the top."""
    return x * np.clip(0.5 + 0.5 * np.sin(2 * np.pi * rpm / 60.0 * np.arange(len(x)) / P), 0.0, 1.0) ** sharp


@sfx("appliance", "Kitchen and utility appliances: micro_ondes (magnetron hum at twice the mains under a dominant fan, plate, end beeps), frigo (compressor at 50 Hz and its fan, on and off), lave_linge (water and load turning at 55 rpm), lave_vaisselle (pump, spray arms, water), seche_linge (load tumbling at 55 rpm, fan).",
     kind=("micro_ondes", " | ".join(APPLIANCES)), mains=(50.0, "Hz: 50 | 60"), dur=(6.0, "s"), seed=(0, "variation"))
def appliance(sr=DEFAULT_SR, kind="micro_ondes", mains=50.0, dur=6.0, seed=0):
    """Hum and fan (`transformer`, `fan`, `beep`, `pour`, `shake`). The compressor turns at the mains less its slip; spray arms at 0.5 Hz."""
    k = _kind(kind, APPLIANCES, "appliance")
    d = float(np.clip(dur, 1.5, 120.0))
    m = float(np.clip(mains, 40.0, 70.0))
    n = _n(d)
    seed = int(seed)
    t = np.arange(n) / P
    rng = np.random.default_rng(seed)
    if k == "micro_ondes":
        run = d - 0.9
        nr = _n(run)
        hum = _fitz(transformer(sr=P, dur=run, mains=m, flux=0.9, load=0.6, loose=0.3, seed=seed), nr)
        hum = hum + np.sin(2 * np.pi * 2.0 * m * np.arange(nr) / P)
        air = _fitz(fan(sr=P, dur=run, rpm=2800, blades=7, seed=seed), nr) + 0.5 * _z(amb.bp(rng.standard_normal(nr), 900.0, 0.5))
        plate = _fitz(FO.shake(sr=P, preset="Sticks", energy=0.2, rate=3.0, dur=run, seed=seed), nr)
        body = (0.3 * _nm(hum) + 1.0 * _nm(air) + 0.04 * plate) * _env(nr, (0.0, 0.0), (0.2, 1.0), (run - 0.1, 1.0), (run, 0.0))
        y = np.zeros(n)
        _put(y, body, 0.0, 1.0)
        _put(y, _piezo(2700.0, [0.0, 0.32, 0.64], 0.2, 0.9), run, 0.5)
    elif k == "frigo":
        comp = sum(g * np.sin(2 * np.pi * h * m * (1.0 - COMP_SLIP) * t) for h, g in ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.1)))
        vent = _z(ventilation(sr=P, speed=11.0, dur=d, seed=seed))
        e = _env(n, (0.0, 0.0), (0.8, 1.0), (d - 0.8, 1.0), (d, 0.0))
        y = 0.8 * comp * e + 0.3 * _fitz(vent, n) * e
        kick = _z(FO.thud(sr=P, body="box", contact_ms=12.0, size=1.2, force=0.4, seed=seed))
        _put(y, kick, 0.0, 0.6)
        _put(y, kick, d - 0.1, 0.5)
    elif k == "lave_linge":
        water = _fitz(pour(sr=P, dur=d, flow=0.6, fill_start=0.4, fill_end=0.6, height=0.5, radius=0.25, material="metal", seed=seed), n)
        mot = amb.lop(_fitz(electric_motor(sr=P, dur=d, rpm=1400.0, load=0.7, seed=seed), n), 1500.0)
        slosh = _lump(amb.lop(rng.standard_normal(n), 500.0), DRUM_RPM, 2.0)
        y = 0.6 * water + 0.25 * mot + _nm(slosh)
    elif k == "lave_vaisselle":
        pump = _fitz(transformer(sr=P, dur=d, mains=m, flux=0.7, load=0.5, seed=seed), n)
        spray_arm = amb.bp(rng.standard_normal(n), 2800.0, 0.5) * (0.65 + 0.35 * np.sin(2 * np.pi * 0.5 * t))
        y = 0.25 * pump + 0.7 * _nm(spray_arm) + 0.35 * _fitz(FO.shake(sr=P, preset="WaterDrops", energy=0.6, rate=5.0, dur=d, seed=seed), n)
    else:
        air = _fitz(fan(sr=P, dur=d, rpm=2800, blades=30, seed=seed), n)
        y = 0.3 * air + 0.2 * amb.lop(_fitz(electric_motor(sr=P, dur=d, rpm=1400.0, load=0.5, seed=seed), n), 1500.0)
        period = 60.0 / DRUM_RPM
        for i, t0 in enumerate(np.arange(0.2, d - 0.1, period)):   # the load falls once a turn
            _put(y, _z(FO.thud(sr=P, body="sack", contact_ms=25.0, size=1.4, force=0.7, seed=seed + i)), float(t0), 1.0)
            _put(y, _z(FO.cloth(sr=P, kind="rustle", dur=0.25, rate=4.0, weight=0.6, seed=seed + i)), float(t0) - 0.05, 0.25)
    return _fin(y, sr, 0.7, fade=0.03, tail=0.1)


# =============================================================================================
# WATER
# =============================================================================================
FLUSH_KINDS = {"reservoir": (6.0, 0.5), "economique": (4.0, 0.3), "chasse_forte": (8.0, 0.8)}    # total s, drain size


@sfx("toilet_flush", "Toilet flush in 4 phases: the tank lets go, the bowl's vortex, the siphon gulping air, the tank refilling (30 s by default, a hiss rising as it fills).",
     kind=("reservoir", " | ".join(FLUSH_KINDS)), refill=(30.0, "s of refill, 0 to skip"), seed=(0, "variation"))
def toilet_flush(sr=DEFAULT_SR, kind="reservoir", refill=30.0, seed=0):
    """4-8 s, 30 s of refill. Bubbles and vortex of `drain`; the refill's band rises 700 to 1800 Hz as the tank fills, then the valve closes."""
    total, size = FLUSH_KINDS[_kind(kind, FLUSH_KINDS, "toilet_flush")]
    r = float(np.clip(refill, 0.0, 60.0))
    seed = int(seed)
    rng = np.random.default_rng(seed)
    y = np.zeros(_n(total + r + 0.3))
    a = 0.25 * total
    _put(y, _z(solenoid(sr=P, size=0.5, metal=0.2, thump=0.7, bounce=0.1, seed=seed)), 0.0, 0.5)
    na = _n(a + 0.5)
    _put(y, _z(pour(sr=P, dur=a + 0.5, flow=1.6, fill_start=0.5, fill_end=0.8, height=0.3, radius=0.12, material="ceramic", seed=seed))
         * _env(na, (0.0, 0.0), (0.1, 1.0), (a, 1.0), (a + 0.5, 0.0)), 0.03, 0.9)
    nb = _n(0.6 * total)
    _put(y, _fitz(drain(sr=P, dur=0.6 * total, size=size, seed=seed), nb), a, 0.9)
    _put(y, _z(gurgle(sr=P, dur=0.3 * total, rate=4.0, size=size + 0.2, seed=seed)), 0.65 * total, 0.6)
    nc = _n(0.15 * total)
    _put(y, _z(air_leak(sr=P, dur=0.15 * total, pressure=0.3, orifice=25.0, seed=seed)) * np.hanning(nc), 0.85 * total, 0.3)
    if r > 0:
        m = _n(r)
        hiss = rng.standard_normal(m)
        fc = np.linspace(700.0, 1800.0, 12)
        step = m // len(fc) + 1
        out = np.zeros(m)
        for i, f in enumerate(fc):                          # overlapping bands stand in for a swept filter
            out += amb.bp(hiss, f, 1.2) * np.clip(1.0 - np.abs(np.arange(m) - (i + 0.5) * step) / step, 0.0, 1.0)
        out *= np.interp(np.arange(m), [0, _n(0.5), m - _n(0.5), m], [0.0, 1.0, 1.0, 0.0])
        _put(y, _nm(out), total * 0.9, 0.3)
        _put(y, _z(_ticks(rng, [0.0], ENDSTOP, 0.2)), total * 0.9 + r - 0.02, 0.2)
    return _fin(y, sr, 0.8, fade=0.002, tail=0.1)


BUBBLE_HZ = 5000.0                                          # a drop's bubble


def bubble_tau(f: float) -> float:
    """s of decay of a bubble ringing at f Hz: 1 / beta of `matter.bubble_params` at its Minnaert radius."""
    return 1.0 / float(bubble_params(MINNAERT_WR / (2.0 * math.pi * float(f)))[1])
FAUCET_TARGETS = {"evier": ("metal", 0.2, 0.2), "baignoire": ("ceramic", 0.4, 0.35), "douche": ("ceramic", 0.15, 0.4)}   # surface, vessel radius m, bubble rate factor


def _bubble_drops(rng, n: int, rate: float) -> np.ndarray:
    """Drops whose bubble rings near 5 kHz (15 % spread), each decaying at its own radius's damping (`matter.bubble_params`)."""
    y = np.zeros(n)
    m = _n(0.03)
    t = np.arange(m) / P
    for at in np.flatnonzero(rng.random(n) < float(rate) / P):
        f = BUBBLE_HZ * rng.uniform(0.85, 1.15)
        s = np.sin(2 * np.pi * f * t) * np.exp(-t / bubble_tau(f)) * np.sin(0.5 * np.pi * np.minimum(t * f, 1.0)) ** 2 * rng.uniform(0.3, 1.0)   # rises over its first cycle
        y[at:at + m] += s[: len(y[at:at + m])]
    return y


@sfx("faucet", "Tap and shower: flow on a surface. evier (metal sink), baignoire (tub filling), douche (spray on tiles). Below 0.02 L/s, separate drops whose bubble rings near 5 kHz.",
     flow=(0.2, "L/s"), target=("evier", " | ".join(FAUCET_TARGETS)), dur=(3.0, "s"), seed=(0, "variation"))
def faucet(sr=DEFAULT_SR, flow=0.2, target="evier", dur=3.0, seed=0):
    """Flow on a vessel or tiles; below 0.02 L/s, single drops. A building's noise class would be a ceiling: a lone sound is normalised, so none is applied."""
    t = _kind(target, FAUCET_TARGETS, "faucet")
    mat, radius, rate_k = FAUCET_TARGETS[t]
    q = float(np.clip(flow, 0.001, 1.0))
    d = float(np.clip(dur, 0.2, 60.0))
    n = _n(d)
    rng = np.random.default_rng(int(seed))
    if q < 0.02:
        y = _bubble_drops(rng, n, 400.0 * q)
        return _fin(y, sr, 0.5, fade=0.01, tail=0.03)
    drops = _z(_bubble_drops(rng, n, 2000.0 * q * rate_k)) if q * rate_k > 0 else np.zeros(n)
    if t == "douche":
        jet = amb.hip(rng.standard_normal(n), 1500.0) + 0.4 * amb.bp(rng.standard_normal(n), 500.0, 0.6)
        y = _nm(jet) * (0.5 + 0.5 * math.sqrt(q)) + 0.3 * drops + 0.3 * _fitz(FO.shake(sr=P, preset="WaterDrops", energy=0.7, rate=8.0, dur=d, seed=seed), n)
    else:
        vessel = pour(sr=P, dur=d, flow=float(np.clip(5.0 * q, 0.2, 3.0)), fill_start=0.05, fill_end=0.4, height=0.15 if t == "evier" else 0.3, radius=radius, material=mat, seed=seed)
        y = _fitz(vessel, n) + 0.25 * drops
    return _fin(y, sr, 0.7, fade=0.04, tail=0.1)


HAMMER_HZ = (240.0, 410.0, 790.0)                           # pipe and wall modes in 200-800 Hz
HAMMER_T = 0.11                                             # s between bangs: 4 L / c, L 36 m of pipe, c 1300 m/s


@sfx("pipe_water", "Plumbing: coup_de_belier (water hammer: a valve shuts, the pipe bangs and bangs again, 200-800 Hz), tuyau_qui_fuit (a pressurised leak, broadband), radiateur (hot water ticking, knocking, gurgling).",
     kind=("coup_de_belier", "coup_de_belier | tuyau_qui_fuit | radiateur"), dur=(3.0, "tuyau_qui_fuit, radiateur: s"), seed=(0, "variation"))
def pipe_water(sr=DEFAULT_SR, kind="coup_de_belier", dur=3.0, seed=0):
    """A leak is broadband up to 40 kHz: here up to the sampling rate's Nyquist."""
    k = _kind(kind, ("coup_de_belier", "tuyau_qui_fuit", "radiateur"), "pipe_water")
    seed = int(seed)
    d = float(np.clip(dur, 0.2, 60.0))
    rng = np.random.default_rng(seed)
    if k == "coup_de_belier":
        n = _n(1.1)
        x = np.zeros(n)
        for i in range(5):
            at = _n(i * HAMMER_T)
            m = _n(0.002)
            x[at:at + m] += 0.55**i * np.hanning(m)
        y = _modes(x, HAMMER_HZ, 0.35, [1.0, 0.7, 0.5])
        return _fin(y, sr, 0.8, fade=0.001, tail=0.05)
    if k == "tuyau_qui_fuit":
        n = _n(d)
        hiss = _fitz(air_leak(sr=P, dur=d, pressure=0.8, orifice=1.5, tone=14000.0, seed=seed), n)
        white = rng.standard_normal(n)
        y = white * rms_of(hiss) + hiss + 0.3 * _fitz(steam(sr=P, dur=d, pressure=0.6, orifice=2.0, wet=0.8, seed=seed), n)
        return _fin(y, sr, 0.6, fade=0.03, tail=0.05)
    y = np.zeros(_n(d))
    y += 0.15 * _fitz(gurgle(sr=P, dur=d, rate=1.2, size=0.3, seed=seed), len(y))
    for i, t in enumerate(np.sort(rng.uniform(0.0, max(d - 0.3, 0.05), max(int(2.5 * d), 2)))):
        f = float(rng.uniform(900.0, 2200.0))
        _put(y, _z(_snap(rng, f, 0.04, 1.0)), float(t), float(rng.uniform(0.2, 0.8)))
    return _fin(y, sr, 0.7, fade=0.01, tail=0.05)


# =============================================================================================
# KITCHEN OBJECTS AS INSTRUMENTS
# =============================================================================================
@sfx("frying_pan", "A frying pan as a percussion: struck iron, handle rattling. A little oil adds the sizzle.",
     size=(1.0, "bigger = lower"), force=(0.8, "0..1"), oil=(0.0, "0..1 sizzle"), seed=(0, "variation"))
def frying_pan(sr=DEFAULT_SR, size=1.0, force=0.8, oil=0.0, seed=0):
    """`impact` metal, a 0.26 m pan; the handle rattles 40 ms later."""
    s = float(np.clip(size, 0.3, 3.0))
    f = float(np.clip(force, 0.05, 1.0))
    o = float(np.clip(oil, 0.0, 1.0))
    ev = [(0.0, K.impact(sr=P, material="metal", shape="rect_plate", size=0.26 * s, velocity=1.0 + 4.0 * f, hardness=0.85, damping=1.0, modes=24, dur=1.2, seed=seed), 1.0),
          (0.04, K.impact(sr=P, material="steel", shape="bar", size=0.2 * s, velocity=0.6, hardness=0.7, modes=8, dur=0.2, seed=seed), 0.12)]
    if o > 0:
        ev.append((0.0, sizzle(sr=P, dur=1.3, intensity=o, seed=seed), 0.35 * o))
    return _fin(_mix(1.3, *ev), sr, 0.8, fade=0.001, tail=0.1)


@sfx("pepper_grinder", "A pepper mill: the head turned in short strokes, teeth clicking and peppercorns cracking.",
     turns=(3, "strokes"), rate=(2.5, "strokes per second"), seed=(0, "variation"))
def pepper_grinder(sr=DEFAULT_SR, turns=3, rate=2.5, seed=0):
    """`ratchet` + grains, 12 teeth per stroke."""
    c = int(np.clip(turns, 1, 30))
    gap = 1.0 / float(np.clip(rate, 0.5, 8.0))
    y = np.zeros(_n(c * gap + 0.1))
    for i in range(c):
        d = 0.8 * gap
        m = _n(d)
        _put(y, _z(FO.ratchet(sr=P, clicks=12, rate=12.0 / d, size=0.4, mount="plastic", seed=int(seed) + i)), i * gap, 0.6)
        _put(y, _fitz(FO.shake(sr=P, preset="Crunch", energy=0.9, rate=18.0, dur=d, seed=int(seed) + i), m) * np.hanning(m), i * gap, 1.0)
    return _fin(y, sr, 0.7, fade=0.002, tail=0.02)


@sfx("vacuum_hose", "A corrugated vacuum hose as an instrument: blown, or swung so that it sings at its pipe note.",
     length=(1.8, "m"), radius=(0.016, "m"), dur=(2.0, "s"), breath=(0.7, "0..1"), seed=(0, "variation"))
def vacuum_hose(sr=DEFAULT_SR, length=1.8, radius=0.016, dur=2.0, breath=0.7, seed=0):
    """`pipe_blow` open at both ends, a breath noise of 10 %."""
    d = float(np.clip(dur, 0.1, 30.0))
    n = _n(d)
    y = _fitz(pipe_blow(sr=P, length_m=float(np.clip(length, 0.3, 5.0)), radius_m=float(np.clip(radius, 0.005, 0.05)), stopped=False, dur=d, breath=breath, seed=seed), n)
    y = y + 0.1 * amb.bp(np.random.default_rng(int(seed)).standard_normal(n), 1400.0, 0.7)
    return _fin(y, sr, 0.7, fade=0.03, tail=0.1)


# =============================================================================================
# CERAMIC, GLASS
# =============================================================================================
CERAMIC_RATIOS = (1.0, 2.83, 5.42, 8.77)                    # ring modes n(n^2-1)/sqrt(n^2+1) over n = 2


@sfx("bowed_ceramic", "A ceramic bowl sung with a bow or a wet finger: friction locked on the modes of a porcelain ring, tapped first if asked.",
     freq=(600.0, "Hz"), dur=(2.5, "s"), pressure=(0.6, "N"), speed=(0.12, "m/s"), tap=(0.0, "0..1 a strike at the start"), seed=(0, "variation"))
def bowed_ceramic(sr=DEFAULT_SR, freq=600.0, dur=2.5, pressure=0.6, speed=0.12, tap=0.0, seed=0):
    """The `rubbed_glass` friction (elastoplastic), with porcelain's loss: fd 600 in `contact.MATERIALS`, xi = 1 / (2 pi fd).
    Mobility as in `rubbed_glass`, 0.3 s of release."""
    f = float(np.clip(freq, 100.0, 3000.0))
    d = float(np.clip(dur, 0.1, 30.0))
    fn = float(np.clip(pressure, 0.05, 3.0))
    v = float(np.clip(speed, 0.02, 0.25))
    ratios = np.array(CERAMIC_RATIOS)
    xi = np.full(len(ratios), 1.0 / (2.0 * math.pi * K.MATERIALS["ceramic"].fd))
    mob = 50.0 / ratios**2

    def run(scale, tt, n_on):
        n = int(tt * FS)
        g = _gate(n, n_on, 0.01, 0.05)
        noise = 0.02 * fn * g * np.random.default_rng(int(seed)).standard_normal(n)
        return elastoplastic_friction(f * scale * ratios, xi, mob, fn * g, v * (0.5 + 0.5 * g), n / FS, sigma0=1.0e5, noise=noise)["out"]

    y = np.asarray(_friction_note(run, f, d, P, 0.3, 0.1), float)
    tp = float(np.clip(tap, 0.0, 1.0))
    if tp > 0:
        hit = _z(K.impact(sr=P, material="ceramic", shape="rect_plate", size=0.12, freq=f, velocity=1.5, hardness=0.7, modes=10, dur=min(d, 0.6), seed=seed))
        y = _z(y)
        _put(y, hit, 0.0, tp)
    return _fin(y, sr, 0.7, fade=0.01, tail=0.1)


GLASS_ALPHA = 1.25                                          # shape factor of a wine glass
GLASS_R, GLASS_WALL = 0.035, 0.0015                         # m: rim radius, wall thickness
RHO_LIQUID, RHO_GLASS = 1000.0, 2500.0


def fill_coef(radius=GLASS_R, wall=GLASS_WALL, rho_l=RHO_LIQUID, rho_g=RHO_GLASS, alpha=GLASS_ALPHA) -> float:
    """The liquid's added mass on the wall: a = alpha rho_l R / (5 rho_g h)."""
    return alpha * rho_l * radius / (5.0 * rho_g * wall)


def fill_freq(f0, fill, a: float = -1.0) -> float:
    """f = f0 / sqrt(1 + a fill^4), fill the height of liquid over the glass's; a < 0: `fill_coef()`."""
    a = fill_coef() if a < 0 else a
    return float(f0) / math.sqrt(1.0 + a * float(np.clip(fill, 0.0, 1.0)) ** 4)


@sfx("filled_glass", "A drinking glass struck, its pitch falling as it fills: f0 / sqrt(1 + 2.33 fill^4), 10 semitones lower when full.",
     freq=(1200.0, "Hz empty"), fill=(0.0, "0 empty .. 1 full"), force=(0.6, "0..1"), seed=(0, "variation"))
def filled_glass(sr=DEFAULT_SR, freq=1200.0, fill=0.0, force=0.6, seed=0):
    """Ratios `friction.GLASS`, a 0.2 ms strike, T60 1.2 s, more damping with liquid."""
    f = fill_freq(np.clip(freq, 200.0, 4000.0), fill)
    fl = float(np.clip(fill, 0.0, 1.0))
    x = np.zeros(_n(1.8))
    x[:max(_n(0.0002), 2)] = float(np.clip(force, 0.05, 1.0))
    y = _modes(x, f * np.array(GLASS), 1.2 / (1.0 + fl), [1.0, 0.5, 0.3, 0.15, 0.08])
    return _fin(y, sr, 0.25 + 0.5 * float(np.clip(force, 0.05, 1.0)), fade=0.0005, tail=0.1)


# =============================================================================================
# FLAT FACES AND ROLLERS LANDING
# =============================================================================================
def _e_star(e1, nu1, e2, nu2) -> float:
    return 1.0 / ((1.0 - nu1 * nu1) / e1 + (1.0 - nu2 * nu2) / e2)


def flat_contact_s(mass: float, e_star: float, area: float) -> float:
    """Flat punch: P = 2 E* a delta, a = sqrt(area / pi). A linear oscillator: half a period, Tc = pi sqrt(m / k), whatever the speed."""
    return math.pi * math.sqrt(float(mass) / (2.0 * float(e_star) * math.sqrt(float(area) / math.pi)))


def half_sine(tc: float) -> np.ndarray:
    m = max(int(round(tc * P)), 1)
    return np.sin(np.pi * (np.arange(m) + 0.5) / m)


# kg, face m2, effective E Pa (the book's 50 MPa: cover and paper, a soft face), nu, body table, f_scale
FLAT_BODIES = {"livre": (0.5, 0.03, 5e7, 0.3, "body_squareBig", 1.0), "brique": (2.5, 0.0225, 2e10, 0.2, "body_squareBig", 2.2), "caisse": (8.0, 0.16, 1.1e10, 0.35, "body_squareBig", 0.6)}


def _floor(floor: str):
    if floor not in ("wood", "concrete", "stone"):
        raise ValueError(f"floor {floor!r}: use wood | concrete | stone")
    return K.MATERIALS[floor]


@sfx("flat_drop", "Something landing flat on its face: livre, brique, caisse. The contact is a flat punch: its length depends on mass and face, not on speed.",
     kind=("livre", " | ".join(FLAT_BODIES)), velocity=(3.0, "m/s at landing: level only"), floor=("wood", "wood | concrete | stone"), seed=(0, "variation"))
def flat_drop(sr=DEFAULT_SR, kind="livre", velocity=3.0, floor="wood", seed=0):
    """Peak force V sqrt(m k): the level follows the speed, the spectrum does not (`flat_contact_s`). A rebound and the book's pages at fixed proportions. Peak 0.75 at 3 m/s."""
    k = _kind(kind, FLAT_BODIES, "flat_drop")
    m, area, e, nu, table, fs = FLAT_BODIES[k]
    fl = _floor(floor)
    tc = flat_contact_s(m, _e_star(e, nu, fl.young, fl.poisson), area)
    rng = np.random.default_rng(int(seed))
    x = np.zeros(_n(0.6))
    p = half_sine(tc)
    x[:len(p)] = p
    i = _n(0.09)
    x[i:i + len(p)] += 0.2 * p                              # rebound at 90 ms, 20 %
    y = _z(_drive(table, x, fs, 2.0))
    if k == "livre":                                        # as `book_drop`
        q = _n(0.07)
        _put(y, _nm(_phisem("Sandpaper", 0.07, rng, np.linspace(1.0, 0.0, q) ** 2, mode="level", res=0.45)), 0.0, 0.3)
    v = float(np.clip(velocity, 0.05, 6.0))
    return _fin(_nm(y) * 0.25 * v, sr, None, fade=0.0005, tail=0.03)


def roller_a(p: float, l: float, r: float, e_star: float) -> float:
    """Half-width of the Hertz strip under the load per length P / 2l."""
    return math.sqrt(2.0 * p * r / (math.pi * l * e_star))


def roller_delta(p: float, l: float, r: float, e_star: float) -> float:
    """Roller of length 2l: delta = P / (pi l E*) {1.886 + ln(l / a)}."""
    return p / (math.pi * l * e_star) * (1.886 + math.log(l / roller_a(p, l, r, e_star)))


def roller_contact(mass: float, velocity: float, l: float, r: float, e_star: float) -> np.ndarray:
    """Force against time at P, unit peak, integrating m d'' = -P(d) through the inverse of (5.61), elastic both ways. Tc depends a little on V: the logarithm."""
    ps = np.geomspace(1e-3, 1e7, 1500)
    ds = np.array([roller_delta(p, l, r, e_star) for p in ps])
    tc0 = math.pi * math.sqrt(mass * ds[700] / ps[700])
    dt = tc0 / 400.0
    d, v, f, ts, t = 0.0, velocity, [], [], 0.0
    for _ in range(40000):
        a = -np.interp(d, ds, ps) / mass
        v += a * dt
        d += v * dt
        t += dt
        if d <= 0.0:
            break
        f.append(np.interp(d, ds, ps))
        ts.append(t)
    f = np.asarray(f)
    tp = np.arange(max(int(t * P) + 1, 2)) / P
    y = np.interp(tp, ts, f, left=0.0, right=0.0)
    return y / (f.max() + 1e-30) if y.max() > 0 else np.ones(1)


FREE_BEAM = (4.7300, 7.8532, 10.9956)                       # beta L of a free-free beam


def beam_modes(length: float, gyration: float, material: str) -> tuple:
    """Hz of the first three bending modes of a free-free bar: (beta L)^2 / (2 pi L^2) * k * sqrt(E / rho), k its radius of gyration."""
    c = K.MATERIALS[material].c
    return tuple(b * b / (2.0 * math.pi * length * length) * gyration * c for b in FREE_BEAM)


# kg, half length m, radius m, material, modes Hz, T60 s. The bottle rings on its shell modes; the log (solid, k = r / 2) and the tube (thin wall, k = r / sqrt 2) bend.
ROLLERS = {"bouteille": (0.4, 0.1, 0.035, "glass", (1450.0, 2900.0, 4300.0), 0.25),
           "buche": (1.5, 0.2, 0.05, "wood", beam_modes(0.4, 0.05 / 2.0, "wood"), 0.08),
           "tube": (0.8, 0.3, 0.025, "steel", beam_modes(0.6, 0.025 / math.sqrt(2.0), "steel"), 0.6)}


@sfx("roller_drop", "A cylinder landing on its side: bouteille (lying bottle), buche (log), tube. Contact stiffness of a finite roller, almost linear with a logarithmic correction.",
     kind=("bouteille", " | ".join(ROLLERS)), velocity=(2.0, "m/s"), floor=("wood", "wood | concrete | stone"), seed=(0, "variation"))
def roller_drop(sr=DEFAULT_SR, kind="bouteille", velocity=2.0, floor="wood", seed=0):
    """`roller_contact` into three modes, level 0.25 V capped by `_fin`."""
    k = _kind(kind, ROLLERS, "roller_drop")
    m, l, r, mat, modes, t60 = ROLLERS[k]
    ml, fl = K.MATERIALS[mat], _floor(floor)
    v = float(np.clip(velocity, 0.05, 6.0))
    f = roller_contact(m, v, l, r, _e_star(ml.young, ml.poisson, fl.young, fl.poisson))
    x = np.zeros(_n(0.1 + 5.0 * t60))
    x[:len(f)] = f
    y = _modes(x, modes, t60, [1.0, 0.6, 0.35])
    return _fin(_nm(y) * 0.25 * v, sr, None, fade=0.0005, tail=0.03)
