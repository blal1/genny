"""Bows and crossbows, slings, siege engines and hand weapons: edges, points and blunt masses in the air and on a target.

Contact laws: plastic wedge, friction pm = pw (1 + mu cot alpha), pm = c Y with c about 3, ploughing with its three modes,
flat punch P = 2 E* a delta. Tones: f = St u / d, St 0.2 (`modal.aeolian_tone`, `modal.sword_swing`), cavity tone (`modal.cavity_tone`).
Contact times: `foley._TARGET_MS`. Sources: ``research/SOURCES.md``. Nothing has been listened to: see tests/test_melee.py.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from scipy.optimize import brentq
from scipy.signal import butter, lfilter, resample_poly, sosfilt

from . import foley as FO
from .contact import ImpactInteractor, _make_body, _striker_k, get_material, impact, rumble, scrape, smash
from .core import DEFAULT_SR
from .fdstring import twang
from .foley import C_AIR, P, _n, _nm, _put as _fo_put, md, modal, wg
from .matter import _lfn
from .ordnance import moving
from .road import _kind
from .sfx import sfx

G = 9.81
NU_STEEL = 0.3


def _hp(x, hz=30.0):
    """Below every tone of the module: as u falls to 0 the swing's filters leave a near-DC residual (measured at -76 dB, whose smoothness makes any later corner a detect_clicks hit)."""
    return sosfilt(butter(2, hz, "highpass", fs=P, output="sos"), x)


def _put(buf, x, t, g=1.0):
    """foley._put with faded ends, 0.5 ms and 20 ms: a layer that is cut (rumble, a ring past its buffer) would step in the mix (measured: detect_clicks fires on each)."""
    x = np.array(x, float)
    k = min(_n(0.02), len(x) // 2)
    x[-k:] *= np.cos(0.5 * np.pi * np.arange(k) / k) ** 2   # raised cosine: a linear fade's corner is itself a detect_clicks hit on a low rumble
    k = min(_n(0.0005), len(x) // 2)
    x[:k] *= np.sin(0.5 * np.pi * np.arange(k) / k) ** 2
    _fo_put(buf, x, t, g)


def _fin(y, sr, level=0.8, *, fade=0.001, tail=0.0):
    """foley._fin's steps (DC, resample, fade ends, level), with its fades done before the level is set."""
    y = lfilter([1.0, -1.0], [1.0, -math.exp(-2.0 * math.pi * 12.0 / P)], y)
    if sr != P:
        fr = Fraction(int(sr), P)
        y = resample_poly(y, fr.numerator, fr.denominator)
    k = max(2, min(len(y) // 2, int(fade * sr)))
    y[:k] *= np.sin(0.5 * np.pi * np.arange(k) / k) ** 2
    k = max(2, min(len(y) // 2, int(max(fade, tail) * sr)))
    y[-k:] *= np.cos(0.5 * np.pi * (np.arange(k) + 1) / k) ** 2
    pk = float(np.max(np.abs(y))) if len(y) else 0.0
    if pk > 0 and (level is not None or pk > 1.2):
        y = y * ((1.2 if level is None else level) / pk)
    return np.nan_to_num(y)


# Y = H / 3, pm = c Y with c about 3 in full plasticity. Hardness in Pa.
HARDNESS = {"flesh": 3e5, "wood": 3e7, "armour": 1.3e9}
# E in Pa: `contact.get_material` for wood and steel.
YOUNG = {"flesh": (2e5, 0.49), "wood": (11e9, 0.35), "armour": (200e9, 0.30)}
FLAT_FLOOR_MS = 0.3                                         # see contact_ms
BLUNT_MS = {"flesh": 20.0, "wood": 2.0, "armour": 0.3}      # blunt mass, inside foley._TARGET_MS ranges

# base: sword preset. u m/s and d m at the tip. mass kg of what lands. alpha degrees half-angle of the edge (None: blunt), edge m in contact. table, fs: what rings.
WEAPONS = {
    # MLB 35 m/s. thickness 0.6 for a truncheon.
    "baton": dict(base="MLB", u=35.0, hilt=0.028 * 0.6, tip=0.057 * 0.6, mass=0.5, alpha=None, edge=0.05, table="sy_stick", fs=1.0),
    # Head 1.35 kg, handle 52 cm. 30 m/s and 30 mm give 200 Hz.
    "axe": dict(base="MLB", u=30.0, length=0.52, hilt=0.03, tip=0.03, mass=1.35, alpha=15.0, edge=0.08, table="sy_sword2", fs=1.4),
    # speed 0.7, thickness 0.4 of SWD1.
    "knife": dict(base="SWD1", u=36.0 * 0.7, hilt=0.013 * 0.4, tip=0.008 * 0.4, mass=0.15, alpha=12.0, edge=0.02, table="sy_sword1", fs=2.0),
    "mace": dict(base="MLB", u=25.0, length=0.6, hilt=0.03, tip=0.07, mass=2.0, alpha=None, edge=0.03, table="sy_wok", fs=1.0),      # 1-3 kg
    # partials: at so low a speed sword_swing's dB scaling lifts 3 f over the fundamental (measured 143 Hz); the fundamental is wanted.
    "sledge": dict(base="MLB", u=15.0, length=0.9, hilt=0.06, tip=0.06, mass=5.0, alpha=None, edge=0.05, table="sy_wok", fs=1.0, partials=(1, 2)),    # 15 m/s, 60 mm, 3.6-9 kg
    "spear": dict(base="MLB", u=15.0, length=2.5, hilt=0.03, tip=0.03, mass=1.0, alpha=10.0, edge=0.02, table="sy_sword1", fs=2.5),  # 2-3 m
}


# =============================================================================================
# CONTACT LAWS
# =============================================================================================
def yield_stress(hardness):
    """Y = H / 3."""
    return float(hardness) / 3.0


def wedge_pressure(alpha_deg, k=1.0):
    """pm = 2 k (1 + psi), cos(2 alpha - psi) = cos psi / (1 + sin psi): 2.61 k at 30 degrees, 5.14 k at 90."""
    a = math.radians(float(np.clip(alpha_deg, 1.0, 90.0)))
    psi = brentq(lambda s: math.cos(2.0 * a - s) * (1.0 + math.sin(s)) - math.cos(s), 0.0, math.pi / 2.0)
    return 2.0 * k * (1.0 + psi)


def blade_force(alpha_deg, length, depth, Y, mu=0.0):
    """Force in N to hold a wedge ``depth`` m into a plastic solid along ``length`` m: pm 2 a L, a = depth tan alpha, k = Y / 2.
    pm = pw (1 + mu cot alpha); pw kept frictionless. # ponytail: no pile-up on a, the true contact is a little wider."""
    a = math.radians(float(np.clip(alpha_deg, 1.0, 90.0)))
    return wedge_pressure(alpha_deg, Y / 2.0) * (1.0 + mu / math.tan(a)) * 2.0 * depth * math.tan(a) * length


def blade_contact(alpha_deg, mass, speed, length, Y, mu=0.0):
    """(contact ms, peak N, depth m) of a mass stopped by the wedge's force, linear in the depth (calculated: m x'' = -c x, a quarter period, no rebound: it stays stuck)."""
    c = blade_force(alpha_deg, length, 1.0, Y, mu)
    w = math.sqrt(mass / c)
    return 0.5 * math.pi * w * 1e3, speed * math.sqrt(mass * c), speed * w


def flat_contact(mass, a, target="wood"):
    """Flat face on a half-space: alpha = 1, k = 2 E* a; Tc = pi sqrt(m / k) in ms, independent of the speed (calculated)."""
    e, nu = YOUNG[target]
    es = 1.0 / ((1.0 - nu * nu) / e + (1.0 - NU_STEEL ** 2) / 200e9)
    return math.pi * math.sqrt(mass / (2.0 * es * a)) * 1e3


def gouge_mode(alpha_deg, mu):
    """Rubbing (plastic wave) unless the wedge stays stuck, which takes mu > cot alpha: then wear, a prow that shears off.
    Cutting exists but has no plastic condition here: only on request."""
    return "wear" if mu >= 1.0 / math.tan(math.radians(float(np.clip(alpha_deg, 1.0, 89.0)))) else "rubbing"


def gouge_force(normal, alpha_deg, mode="rubbing"):
    """Tangential force. Q / P = cot alpha, Q / P towards 1 for an adhesive wedge; cutting taken at cot alpha, as rubbing."""
    return float(normal) if mode == "wear" else float(normal) / math.tan(math.radians(float(np.clip(alpha_deg, 1.0, 89.0))))


def launch_speed(range_m):
    """v >= sqrt(R g). 80-445 m gives 28-66 m/s."""
    return math.sqrt(float(range_m) * G)


# =============================================================================================
# THE AIR
# =============================================================================================
def whirl(u, d, omega, dur, rng, depth=0.25):
    """Aeolian tone at St u / d, u(t) modulated at the rotation ``omega`` rad/s. Dipole power goes as u^6, so the level as u^3."""
    n = _n(dur)
    v = float(u) * (1.0 + depth * np.sin(float(omega) * np.arange(n) / P))
    return modal.aeolian_tone(v, float(d), n / P, rng, "loq")[:n] * (v / float(u)) ** 3


def _air(name, rng, speed=1.0, arc=1.0):
    """`foley.sword`'s model on this module's table. ``arc`` scales the swept angle."""
    w = WEAPONS[name]
    p = dict(md.SWORD_PRESETS["presets"][w["base"]])
    p.update(TopSpeed=w["u"] * speed, HiltThick=w["hilt"], TipThick=w["tip"], Len=w.get("length", p["Len"]))
    p["PreAzim"], p["PostAzim"] = 180.0 + (p["PreAzim"] - 180.0) * arc, 180.0 + (p["PostAzim"] - 180.0) * arc
    y = _hp(modal.sword_swing(p, rng, model="loq", lead=0.01, tail=0.0, partials=w.get("partials", (1, 2, 3, 4, 5))).mean(axis=1))
    return _nm(y)


# =============================================================================================
# HITS
# =============================================================================================
def _hit(target, table, fs, T, vel, rng):
    """`foley._weapon_hit` with the blade's table and the contact time as arguments (it has sy_sword1 fixed; the plan says copy, not edit)."""
    n = _n(1.2)
    f = np.zeros(n)
    i0 = _n(0.002)
    p = modal.bang(T, vel)
    f[i0:i0 + len(p)] += p
    blade = modal.resolve(table, fs, rng.uniform(0, 1), d_scale=3.0)
    if target == "armour":
        modal.micro_collisions(f, blade, 3, T, vel, rng, start=i0)
    y = _nm(modal.render_modes(f, blade), 1.0)
    pk = lambda z: z / (float(np.max(np.abs(z))) + 1e-12)
    if target == "flesh":
        body = modal.render_modes(f, modal.resolve("sy_stick", 0.1))
        wet = np.zeros(n)
        wet[i0:i0 + len(p)] = p / p.max() * rng.uniform(-1, 1, len(p))
        wet = lfilter([0.25], [1.0, -0.75], wet)
        y = 0.02 * y + pk(body) + 0.45 * pk(wet)
    elif target == "armour":
        y = y + pk(modal.render_modes(f, modal.resolve("sy_wok", 1.0, d_scale=1.5)))
    else:
        y = 0.25 * y + pk(modal.render_modes(f, modal.resolve("sy_stick", 0.6))) + 0.6 * pk(modal.render_modes(f, modal.resolve("body_squareMid", 1.0)))
    drive = (1.0 if target == "flesh" else 2.0) * vel
    return np.tanh(0.6 * drive * pk(y)) / math.tanh(0.6 * drive)


def contact_ms(name, target, force=0.7, flat=False):
    """How long ``name`` stays on ``target``: the flat face, the wedge, or the table for a blunt mass."""
    w = WEAPONS[name]
    speed = w["u"] * (0.4 + 0.6 * force)                    # share of the swing's speed at the target
    if flat:
        # Face edge x 5 cm. A real flat never lands flat all at once: floor 0.3 ms (the top of steel on steel). The flat punch alone gives 0.13 ms on wood against the 1-3 ms measured.
        return max(flat_contact(w["mass"], math.sqrt(w["edge"] * 0.05 / math.pi), target), FLAT_FLOOR_MS)
    if w["alpha"] is None:
        return modal.contact_time(BLUNT_MS[target], 0.3 + 0.9 * force)
    return blade_contact(w["alpha"], w["mass"], speed, w["edge"], yield_stress(HARDNESS[target]))[0]


def _strike(name, target, force, rng, flat=False):
    w = WEAPONS[name]
    return _hit(target, w["table"], w["fs"], contact_ms(name, target, force, flat), 0.3 + 0.9 * force, rng)


def _mass_hit(name, target, force, seed):
    """A heavy mass through `contact.impact`; on flesh, the long contact of `_hit`."""
    w = WEAPONS[name]
    v = w["u"] * (0.4 + 0.6 * force)
    if target == "flesh":
        return _strike(name, "flesh", force, np.random.default_rng(seed))
    material, shape, size = {"armour": ("steel", "circ_plate", 0.4), "metal": ("steel", "bar", 0.5), "wood": ("wood", "rect_plate", 0.6), "stone": ("stone", "rect_plate", 1.0)}[target]
    return _nm(impact(sr=P, material=material, shape=shape, size=size, striker_mass=w["mass"], hardness=1.0, velocity=v, damping=4.0, dur=1.2, seed=seed))


# =============================================================================================
# GOUGE
# =============================================================================================
def _gouge(normal, alpha, mode, dur, rng, table="sy_sword1", fs=1.0):
    """Tangential force Q drawn through a blade's modes, with the texture of the mode: rubbing a smooth wave, wear prows that build and shear off (bursts), cutting a dense chip.
    Level proportional to Q; 1 N of Q gives a fixed gain. Prows 14 per second, chips 400."""
    n = _n(dur)
    q = gouge_force(normal, alpha, mode)
    noise = rng.standard_normal(n)
    if mode == "rubbing":
        drive = lfilter([0.1], [1.0, -0.9], noise) * (0.7 + 0.3 * _lfn(rng, n, 6.0))
    else:
        rate = 14.0 if mode == "wear" else 400.0
        steps = np.cumsum(rng.exponential(1.0 / rate, int(2 * rate * dur) + 8))
        steps = steps[steps < dur]
        tt = np.arange(n) / P
        since = tt - np.concatenate([[0.0], steps])[np.searchsorted(steps, tt)]
        wear = mode == "wear"                               # a prow shears over 6 ms, 3.3 keeping its peak; 2 ms still fired detect_clicks in the silence between prows.
        drive = (3.3 if wear else 1.0) * np.exp(-since / (0.002 if wear else 0.0015)) * np.sin(0.5 * np.pi * np.minimum(since / (0.006 if wear else 0.002), 1.0)) ** 2 * noise
    drive = drive * np.minimum(1.0, np.minimum(np.arange(n), n - 1 - np.arange(n)) / _n(0.01))
    return modal.render_modes(drive, modal.resolve(table, fs, 0.5, d_scale=8.0 if mode == "wear" else 3.0))[:n] * q * GOUGE_GAIN


GOUGE_GAIN = 2e-5                                           # 1 N of Q gives this
GOUGE_MODES = ("auto", "rubbing", "wear", "cutting")


@sfx("gouge", "A hard edge ploughing a softer surface. Tangential force Q = P cot alpha; three textures: rubbing (a plastic wave), wear (a prow that builds and shears off, in bursts), cutting (a continuous chip).",
     alpha=(30.0, "edge half-angle, degrees"), normal=(50.0, "N pressing"), mu=(0.1, "friction on the wedge face: above cot alpha the wedge sticks and the prow wears off"),
     mode=("auto", " | ".join(GOUGE_MODES)), dur=(0.8, "s"), seed=(0, "variation"))
def gouge(sr=DEFAULT_SR, alpha=30.0, normal=50.0, mu=0.1, mode="auto", dur=0.8, seed=0):
    m = _kind(mode, GOUGE_MODES, "gouge")
    m = gouge_mode(alpha, mu) if m == "auto" else m
    return _fin(_gouge(float(normal), float(alpha), m, float(np.clip(dur, 0.05, 20.0)), np.random.default_rng(int(seed))), sr, None, fade=0.005, tail=0.02)


# =============================================================================================
# BOWS
# =============================================================================================
def _string(pitch, rng, sustain=0.5):
    return _nm(wg.pluck(float(np.clip(pitch, 40, 400)), 0.6, rng, position=0.5, brightness=0.5, body="none", velocity=0.9, sustain=sustain))


def _shaft(u, d, m, rng, slow=0.2):
    """A shaft leaving: its Aeolian tone at u slowing down (`foley.bow_shot`: 20 %), only near the shooter."""
    v = float(u) * (1.0 - slow * np.arange(m) / m)
    return _nm(modal.aeolian_tone(v, float(d), m / P, rng, "loq")[:m] * np.sin(np.pi * np.arange(m) / m))


def _tick(table, fs, ms, rng, position=0.5):
    """One small metal or wood contact, ms long."""
    f = np.zeros(_n(0.35))
    p = modal.bang(ms, 1.0)
    f[_n(0.001):_n(0.001) + len(p)] += p
    return _nm(modal.render_modes(f, modal.resolve(table, fs, position, d_scale=4.0)))


def _crank(seconds, seed, rate=6.0):
    """Windlass: pawl clicks at 6 a second, louder as the tension goes from 76 to over 340 kg, level proportional to the tension."""
    y = FO.ratchet(sr=P, clicks=int(seconds * rate), rate=rate, size=1.0, mount="wood", seed=seed)
    return y[:_n(seconds)] * np.linspace(76.0 / 340.0, 1.0, min(len(y), _n(seconds)))


CROSSBOW_KINDS = ("modern", "arbalest", "reload")


@sfx("crossbow", "Crossbow: modern (short stiff string, trigger, ping, a 106 m/s bolt), arbalest (a 30 s windlass, then the release), reload (the windlass alone).",
     kind=("modern", " | ".join(CROSSBOW_KINDS)), speed=(-1.0, "bolt m/s, -1 = the kind's (106 modern, 70 arbalest)"), shaft_d=(0.009, "bolt diameter m"), pitch=(-1.0, "string Hz, -1 = the kind's"),
     crank=(30.0, "s of windlass (2 bolts a minute)"), distance=(20.0, "m to the target (0 = none)"), target=("wood", "wood|flesh|armour"), seed=(0, "variation"))
def crossbow(sr=DEFAULT_SR, kind="modern", speed=-1.0, shaft_d=0.009, pitch=-1.0, crank=30.0, distance=20.0, target="wood", seed=0):
    """349 fps = 106 m/s on a 9 mm bolt gives 2356 Hz. String 250 Hz. Output is normalised: published dB levels are not reproduced."""
    k = _kind(kind, CROSSBOW_KINDS, "crossbow")
    rng = np.random.default_rng(int(seed))
    crank = float(np.clip(crank, 1.0, 60.0))
    if k == "reload":
        return _fin(_crank(crank, seed), sr, 0.8, fade=0.005, tail=0.05)
    u = (106.0 if k == "modern" else 70.0) if speed < 0 else float(speed)   # 70 m/s, 180 Hz for the arbalest
    t0 = crank + 0.2 if k == "arbalest" else 0.0
    d = float(np.clip(distance, 0.0, 150.0))
    fl = d / u
    y = np.zeros(_n(t0 + 0.05 + fl + 1.4))
    if k == "arbalest":
        _put(y, _crank(crank, seed), 0.0, 0.5)
    _put(y, _tick("sy_desklamp", 2.0, 0.4, rng), t0, 0.35)    # trigger
    t1 = t0 + 0.012
    _put(y, _string((250.0 if k == "modern" else 180.0) if pitch < 0 else pitch, rng, 0.35), t1, 0.8)
    if k == "modern":
        _put(y, _tick("sy_wok", 2.5, 0.2, rng), t1 + 0.002, 0.15)     # retaining ping
    _put(y, _shaft(u, shaft_d, _n(min(max(fl, 0.12), 0.45)), rng), t1 + 0.01, 0.5)
    if d > 0:
        _put(y, _hit(_target(target), "sy_sword1", 1.0, FO._TARGET_MS[_target(target)][1], 0.84, rng), t1 + fl, 1.0 / (1.0 + d / 15.0))
    return _fin(y, sr, 0.8, fade=0.005, tail=0.05)


def _target(target):
    return target if target in ("flesh", "wood", "armour") else _kind(target, ("flesh", "wood", "armour"), "target")


ARROW_KINDS = ("vol", "impact", "carquois")


@sfx("arrow", "An arrow by stage: vol (the shaft's tone alone, going away), impact (the point landing), carquois (drawn from the quiver).",
     kind=("vol", "vol | impact | carquois"), speed=(70.0, "m/s"), shaft_d=(0.008, "m"), target=("wood", "impact: wood|flesh|armour"), dur=(0.6, "vol: s"), seed=(0, "variation"))
def arrow(sr=DEFAULT_SR, kind="vol", speed=70.0, shaft_d=0.008, target="wood", dur=0.6, seed=0):
    """Stages of `foley.bow_shot`: 8 mm at 70 m/s slowing 20 %. Receding at the shooter's ear: `ordnance.moving`."""
    rng = np.random.default_rng(int(seed))
    k = _kind({"vol seul": "vol", "sortie carquois": "carquois"}.get(str(kind), kind), ARROW_KINDS, "arrow")
    if k == "vol":
        m = _n(float(np.clip(dur, 0.1, 5.0)))
        u = float(speed)
        v = u * (1.0 - 0.2 * np.arange(m) / m)
        return _fin(moving(_nm(modal.aeolian_tone(v, float(shaft_d), m / P, rng, "loq")[:m]) * np.minimum(np.arange(m) / _n(0.01), 1.0), v, 1.0), sr, 0.7, fade=0.005, tail=0.03)
    if k == "impact":
        return _fin(_hit(_target(target), "sy_sword1", 1.0, FO._TARGET_MS[_target(target)][1], 0.84, rng)[:_n(0.9)], sr, 0.8, tail=0.03)
    y = np.zeros(_n(0.9))                                   # shaft sliding out over 0.35 s, the fletching, the nock on the rim
    _put(y, scrape(sr=P, dur=0.35, speed=[0.3, 1.0, 0.4], force=[0.2, 0.5, 0.1], roughness=0.6, material="wood", size=0.3, seed=int(seed)), 0.0, 0.3)
    _put(y, FO.cloth(sr=P, kind="rustle", dur=0.3, rate=4.0, weight=0.2, seed=int(seed)), 0.12, 0.15)
    _put(y, _tick("sy_stick", 2.0, 0.8, rng), 0.36, 0.9)
    return _fin(y, sr, 0.8, fade=0.005, tail=0.03)


# =============================================================================================
# SLINGS
# =============================================================================================
SLING_U, SLING_D, SLING_R = 40.0, 0.003, 1.0                # 40-50 m/s; 3 mm cord, radius 1 m. So St u / d = 2667 Hz, u / 2 pi r = 6.4 turns/s.
STONE_DENSITY = 11340.0                                     # lead, kg/m^3
SLING_KINDS = ("whirl", "release", "whistling")


def _ball_d(mass_kg, density):
    return (6.0 * mass_kg / (math.pi * density)) ** (1.0 / 3.0)


@sfx("sling", "A sling: whirl (the cord's tone, swelling at each turn), release (a soft flick and the stone leaving, no crack), whistling (a bullet with a 5 mm hole singing away, 20 % of those at Burnswark).",
     kind=("whirl", " | ".join(SLING_KINDS)), speed=(40.0, "m/s, 20-50"), cord=(SLING_D, "m"), radius=(SLING_R, "m"), mass=(0.028, "kg: release, 28 g of lead .. 0.5 kg of stone"), dur=(2.0, "s: whirl, whistling"), seed=(0, "variation"))
def sling(sr=DEFAULT_SR, kind="whirl", speed=40.0, cord=SLING_D, radius=SLING_R, mass=0.028, dur=2.0, seed=0):
    k = _kind(kind, SLING_KINDS, "sling")
    rng = np.random.default_rng(int(seed))
    u = float(np.clip(speed, 5.0, 60.0))
    d = float(np.clip(dur, 0.2, 30.0))
    if k == "whirl":
        y = whirl(u, cord, u / max(float(radius), 0.1), d, rng)
        return _fin(y, sr, 0.7, fade=0.15, tail=0.15)
    if k == "release":
        # `whip` without its N-wave: the thong's end accelerating to the stone's speed, then the stone. A 28 g ball of lead is 16.8 mm (calculated): 476 Hz. 0.15 s and 0.5 s.
        m = _n(0.15)
        y = np.zeros(m + _n(0.6))
        v = u * (np.arange(m) / m) ** 3
        _put(y, _nm(modal.aeolian_tone(v, float(cord), m / P, rng, "loq")[:m] * (v / u) ** 2), 0.0, 0.8)
        _put(y, _shaft(u, _ball_d(float(np.clip(mass, 0.01, 1.0)), STONE_DENSITY), _n(0.5), rng), m / P, 0.6)
        return _fin(y, sr, 0.7, fade=0.005, tail=0.05)
    n = _n(d)                                               # L = D = W = 5 mm, u = 40. Doppler: the bullet going away, 1 m off.
    tone = FO.cavity_tone(sr=P, speed=u, length=0.005, depth=0.005, width=0.005, dur=d, seed=int(seed))[:n]
    return _fin(moving(tone, np.full(len(tone), u), 1.0, 1.0), sr, 0.7, fade=0.01, tail=0.05)


@sfx("slingshot_weapon", "A slingshot, the weapon: the latex drawn, the band's twang, a ball at 28 m/s, its arrival.",
     draw=(0.5, "s of draw (0 = none)"), freq=(140.0, "Hz, the band's settled pitch"), speed=(28.27, "m/s"), distance=(10.0, "m to the target (0 = none)"), target=("wood", "what it lands on: any contact material"), seed=(0, "variation"))
def slingshot_weapon(sr=DEFAULT_SR, draw=0.5, freq=140.0, speed=28.27, distance=10.0, target="wood", seed=0):
    rng = np.random.default_rng(int(seed))
    dr = float(np.clip(draw, 0.0, 3.0))
    u = max(float(speed), 1.0)
    d = float(np.clip(distance, 0.0, 100.0))
    y = np.zeros(_n(dr + d / u + 1.5))
    if dr > 0.05:
        _put(y, FO.creak(sr=P, material="leather", force="ramp", level=0.7, dur=dr, seed=int(seed)), 0.0, 0.3)     # rubber as leather
    _put(y, _nm(twang(sr=P, kind="band", freq=float(freq), dur=0.9)), dr, 0.8)
    _put(y, _shaft(u, 0.01, _n(min(max(d / u, 0.1), 0.4)), rng), dr + 0.005, 0.2)        # 10 mm ball
    if d > 0:
        _put(y, _nm(impact(sr=P, material=target, shape="rect_plate", size=0.3, striker_mass=0.01, hardness=1.0, velocity=u, dur=0.5, seed=int(seed))), dr + d / u, 1.0 / (1.0 + d / 15.0))
    return _fin(y, sr, 0.8, fade=0.005, tail=0.03)


# =============================================================================================
# SIEGE
# =============================================================================================
ARM_D = 0.3                                                 # m: 33 Hz at 50 m/s
CATAPULT_KINDS = ("trebuchet", "mangonel", "ballista")


def _echo(y, delay, gain):
    k = _n(delay)
    z = np.concatenate([y, np.zeros(k)])
    z[k:] += gain * y
    return z


@sfx("catapult", "Siege engines: trebuchet (ropes under load, the latch, the arm's low tone, the counterweight's thud), mangonel (ropes and the arm on its crossbar), ballista (twisted cords and a bolt).",
     kind=("trebuchet", " | ".join(CATAPULT_KINDS)), range=(300.0, "m: the projectile's speed is sqrt(R g), 80-445 m"), arm_speed=(50.0, "m/s at the arm's tip"), counterweight=(10.0, "t, 2-30"), pitch=(100.0, "ballista: cord Hz, 80-120"), seed=(0, "variation"))
def catapult(sr=DEFAULT_SR, kind="trebuchet", range=300.0, arm_speed=50.0, counterweight=10.0, pitch=100.0, seed=0):
    """Arm swing, release, the projectile leaving; the engines differ by arm and timing."""
    k = _kind(kind, CATAPULT_KINDS, "catapult")
    rng = np.random.default_rng(int(seed))
    s = int(seed)
    if k == "trebuchet":
        arm = 1.8
        y = np.zeros(_n(1.6 + arm + 3.0))
        _put(y, FO.creak(sr=P, material="rope", force="ramp", level=0.8, dur=1.5, size=1.5, seed=s), 0.0, 0.35)
        _put(y, _tick("sy_stick", 0.5, 2.0, rng), 1.5, 0.5)
        m = _n(arm)
        v = float(arm_speed) * np.sin(0.5 * np.pi * np.minimum(np.arange(m) / (0.8 * m), 1.0)) ** 2 + 0.1  # the arm up to speed in 80 % of the swing
        tone = modal.aeolian_tone(v, ARM_D, arm, rng, "loq")[:m] * (v / float(arm_speed)) ** 3 * np.sin(0.5 * np.pi * np.minimum(1.0, (m - np.arange(m)) / _n(0.2))) ** 2
        _put(y, _nm(tone), 1.55, 0.9)
        _put(y, _shaft(launch_speed(range), 0.2, _n(0.5), rng), 1.55 + 0.8 * arm, 0.2)       # 20 cm stone
        size = float(np.clip(counterweight, 2.0, 30.0)) / 10.0
        tail = FO.thud(sr=P, body="crate", contact_ms=35.0, size=2.0 * size, force=1.0, seed=s)[:_n(0.3)]
        _put(y, _echo(_nm(tail), 0.28, 0.25), 1.55 + arm, 1.0)
        _put(y, FO.chain(sr=P, dur=0.9, pushes=1, weight=1.0, seed=s), 1.55 + arm, 0.15)
        _put(y, rumble(sr=P, dur=2.4, fmin=35.0, fmax=90.0, seed=s)[:_n(1.6)], 1.55 + arm, 0.25)
    elif k == "mangonel":
        y = np.zeros(_n(3.2))
        _put(y, FO.creak(sr=P, material="rope", force="ramp", level=0.7, dur=0.9, seed=s), 0.0, 0.35)
        _put(y, _nm(twang(sr=P, kind="band", freq=110.0, dur=0.7)), 0.95, 0.5)
        _put(y, _nm(impact(sr=P, material="wood", shape="bar", size=2.0, striker_mass=20.0, hardness=0.8, velocity=15.0, dur=2.0, seed=s))[:_n(1.5)], 1.15, 1.0)
        _put(y, _shaft(launch_speed(range), 0.15, _n(0.5), rng), 1.15, 0.2)
    else:
        y = np.zeros(_n(2.2))
        _put(y, FO.creak(sr=P, material="rope", force="sway", level=0.6, dur=0.5, seed=s), 0.0, 0.25)
        _put(y, _tick("sy_stick", 1.0, 1.0, rng), 0.5, 0.4)
        _put(y, _nm(twang(sr=P, kind="band", freq=float(np.clip(pitch, 80.0, 120.0)), dur=1.0, amount=0.5)), 0.51, 1.0)
        _put(y, _shaft(launch_speed(range), 0.03, _n(0.4), rng), 0.52, 0.3)
    return _fin(y, sr, 0.85, fade=0.01, tail=0.1)


def siege_strike(mass, velocity, seed=0):
    """A stone on masonry: `contact.impact`'s interactor, called directly because `impact` clips its mass at 50 kg and its speed at 30 m/s. 2 m block."""
    mat = get_material("stone")
    body = _make_body("rect_plate", mat, 2.0, 0.3, 0.37, modes=32, damping=4.0, seed=int(seed), fmax=8000.0)
    return ImpactInteractor(body, mass=float(mass), k=_striker_k(mat, 1.0, float(mass)), mu=mat.mu, bright=0.3).strike(float(velocity), 1.8, P).audio


@sfx("siege_impact", "A siege projectile on masonry: 60-250 kg at 34-66 m/s, the wall's blow, stone breaking and the debris.",
     mass=(150.0, "kg, 60-250"), velocity=(50.0, "m/s, 34-66"), debris=(0.5, "0..1 level of the fall of stones"), seed=(0, "variation"))
def siege_impact(sr=DEFAULT_SR, mass=150.0, velocity=50.0, debris=0.5, seed=0):
    s = int(seed)
    y = np.zeros(_n(3.0))
    _put(y, _nm(siege_strike(float(np.clip(mass, 10.0, 500.0)), float(np.clip(velocity, 5.0, 100.0)), s)), 0.003, 1.0)
    g = float(np.clip(debris, 0.0, 1.0))
    if g > 0:
        _put(y, smash(sr=P, material="stone", size=0.8, energy=1.0, fragments=24, dur=2.4, seed=s), 0.02, 0.5 * g)
    _put(y, rumble(sr=P, dur=2.8, fmin=35.0, fmax=90.0, seed=s)[:_n(2.0)], 0.0, 0.3)
    return _fin(y, sr, 0.9, fade=0.001, tail=0.1)


# =============================================================================================
# HAND WEAPONS
# =============================================================================================
def _throw(rng, distance=6.0, speed=10.0, spin=4.0, target="wood"):
    """A spinning blade leaves and lands: 3-5 turns/s, 10 m/s, 6 m. The tip's whirl goes away at the thrower's ear, the point lands d/v + d/c later."""
    fl = distance / speed
    m = _n(fl)
    d = WEAPONS["knife"]["tip"]
    tone = whirl(speed * 1.5, d, 2.0 * math.pi * spin, fl, rng, 0.45) * np.minimum(np.arange(m) / _n(0.02), 1.0) * np.minimum((m - np.arange(m)) / _n(0.02), 1.0)
    away = moving(_nm(tone), speed, 1.0)
    y = np.zeros(max(len(away), _n(fl + distance / C_AIR)) + _n(0.9))
    _put(y, away, 0.0, 0.6)
    _put(y, _strike("knife", target, 0.7, rng), fl + distance / C_AIR, 0.7)
    return y


def _draw(rng, seed):
    """A blade leaving its scabbard: friction speeding up then stopping (`scrape`), the mouth ringing it at the end, in 0.45 s."""
    y = np.zeros(_n(1.0))
    _put(y, scrape(sr=P, dur=0.45, speed=[0.2, 0.9, 1.2], force=[0.2, 0.5, 0.3], roughness=0.5, material="steel", size=0.25, seed=seed), 0.0, 0.4)
    _put(y, _tick("sy_sword1", 2.0, 0.3, rng), 0.45, 0.8)
    return y


def _with_swing(air, hit, level=0.35):
    """The hit lands at the swing's loudest instant, as `foley.slash`."""
    at = int(np.argmax(np.convolve(air ** 2, np.ones(_n(0.02)), "same"))) / P
    y = np.zeros(_n(at) + max(len(hit), len(air)))
    _put(y, air, 0.0, level)
    _put(y, hit, at, 1.0)
    return y


MELEE_KINDS = ("baton_swing", "baton_hit", "baton_extend", "axe_swing", "axe_chop", "mace", "sledgehammer", "knife_slash", "knife_throw", "flail")


@sfx("melee", "Hand weapons: baton_swing, baton_hit, baton_extend (telescopic, 3 clicks), axe_swing, axe_chop (fibres, then the axe pulled free), mace, sledgehammer, knife_slash, knife_throw (spinning, then planted), flail (chain and head).",
     kind=("baton_swing", " | ".join(MELEE_KINDS)), target=("default", "what the blow lands on: flesh|wood|armour, sledgehammer stone|metal; default = the kind's"), force=(0.8, "0..1"),
     flat=(0, "1 = with the flat: alpha = 1, k = 2 E* a (axe_chop, knife_slash)"), arc=(1.0, "knife_slash, swings: scale of the swept angle"), seed=(0, "variation"))
def melee(sr=DEFAULT_SR, kind="baton_swing", target="default", force=0.8, flat=0, arc=1.0, seed=0):
    k = _kind(kind, MELEE_KINDS, "melee")
    rng = np.random.default_rng(int(seed))
    s = int(seed)
    fc = float(np.clip(force, 0.0, 1.0))
    arc = float(np.clip(arc, 0.3, 1.0))                     # beyond the preset the swept angle leaves the model's azimuth range
    tgt = {"baton_hit": "flesh", "axe_chop": "wood", "mace": "armour", "sledgehammer": "stone", "knife_slash": "flesh", "knife_throw": "wood", "flail": "armour"}.get(k, "flesh") if target == "default" else str(target)
    if k in ("baton_swing", "axe_swing"):
        name = k.split("_")[0]
        y = _air(name, rng, 0.6 + 0.5 * fc, arc)
        if name == "axe":                                   # the head: 2 cm across its path, tone 300 Hz
            m = len(y)
            y = y + 0.5 * _nm(modal.aeolian_tone(30.0, 0.02, m / P, rng, "loq")[:m]) * np.sqrt(np.convolve(y ** 2, np.ones(_n(0.03)) / _n(0.03), "same"))
        return _fin(y, sr, 0.7, fade=0.01, tail=0.05)
    if k == "baton_hit":
        return _fin(_strike("baton", _target(tgt), fc, rng)[:_n(0.9)], sr, 0.8, tail=0.03)
    if k == "baton_extend":                                 # 3 telescoping sections, 80-120 ms apart; the last locks harder
        y = np.zeros(_n(0.9))
        t = 0.03
        for i in range(3):
            _put(y, _tick("sy_desklamp", 1.5, 0.3, rng, rng.uniform(0, 1)), t, 0.5 + 0.25 * i)
            _put(y, scrape(sr=P, dur=0.07, speed=1.0, force=0.3, roughness=0.3, material="steel", size=0.2, seed=s + i), t - 0.03 if i else 0.0, 0.04)
            t += rng.uniform(0.08, 0.12)
        return _fin(y, sr, 0.8, fade=0.002, tail=0.05)
    if k == "axe_chop":
        tg = _target(tgt)
        y = _with_swing(_air("axe", rng, 0.6 + 0.5 * fc, arc), _strike("axe", tg, fc, rng, bool(flat)), 0.2)
        at = int(np.argmax(np.abs(y))) / P
        if tg == "wood" and not flat:
            # A fibre tearing: `physics.Destruction` of wood as `smash`. Pulled free: a short gouge in the slot and the blade's tick.
            z = np.zeros(len(y) + _n(1.2))
            z[:len(y)] = y
            _put(z, smash(sr=P, material="wood", size=0.2, energy=0.4, fragments=8, dur=0.5, seed=s), at, 0.25)
            _put(z, _gouge(50.0, 15.0, "wear", 0.25, rng, "sy_stick", 0.8), at + 0.65, 0.3)
            _put(z, _tick("sy_sword2", 1.4, 1.5, rng), at + 0.9, 0.2)
            y = z
        return _fin(y, sr, 0.85, fade=0.005, tail=0.05)
    if k == "mace":
        return _fin(_with_swing(_air("mace", rng, 0.6 + 0.5 * fc), _mass_hit("mace", tgt, fc, s), 0.2), sr, 0.85, fade=0.005, tail=0.05)
    if k == "sledgehammer":
        return _fin(_with_swing(_air("sledge", rng, 1.0), _mass_hit("sledge", tgt, fc, s), 0.5), sr, 0.85, fade=0.005, tail=0.05)
    if k == "knife_slash":
        return _fin(_with_swing(_air("knife", rng, 0.8 + 0.25 * fc, arc), _strike("knife", _target(tgt), fc, rng, bool(flat))[:_n(0.8)]), sr, 0.8, fade=0.005, tail=0.03)
    if k == "knife_throw":
        return _fin(_throw(rng, target=_target(tgt)), sr, 0.8, fade=0.005, tail=0.03)
    y = np.zeros(_n(2.0))                                   # flail: the chain swung, the head landing at .45 s, the chain falling after it
    _put(y, FO.chain(sr=P, dur=0.5, pushes=1, weight=0.8, settle=0, seed=s), 0.0, 0.45)
    _put(y, _mass_hit("mace", tgt, fc, s), 0.45, 1.0)
    _put(y, FO.chain(sr=P, dur=0.5, pushes=1, weight=0.8, settle=1, seed=s + 1), 0.47, 0.35)
    return _fin(y, sr, 0.85, fade=0.005, tail=0.05)


KNIFE_ACTIONS = ("degainer", "couper", "trancher", "lancer")


@sfx("knife", "A knife: degainer (out of its scabbard), couper (the edge drawn through, a continuous chip), trancher (a quick slash and draw), lancer (thrown, spinning, planted).",
     action=("trancher", " | ".join(KNIFE_ACTIONS)), target=("flesh", "trancher: flesh|wood|armour"), seed=(0, "variation"))
def knife(sr=DEFAULT_SR, action="trancher", target="flesh", seed=0):
    a = _kind(action, KNIFE_ACTIONS, "knife")
    rng = np.random.default_rng(int(seed))
    s = int(seed)
    if a == "degainer":
        return _fin(_draw(rng, s), sr, 0.8, fade=0.005, tail=0.05)
    if a == "couper":
        w = WEAPONS["knife"]
        return _fin(_nm(_gouge(30.0, w["alpha"], "cutting", 0.6, rng, w["table"], w["fs"])), sr, 0.7, fade=0.005, tail=0.03)
    if a == "lancer":
        return _fin(_throw(rng), sr, 0.8, fade=0.005, tail=0.03)
    tg = _target(target)
    y = _with_swing(_air("knife", rng, 1.0), _strike("knife", tg, 0.8, rng)[:_n(0.8)])
    at = int(np.argmax(np.abs(y))) / P
    w = WEAPONS["knife"]
    _put(y, _nm(_gouge(30.0, w["alpha"], "cutting", 0.12, rng, w["table"], w["fs"])), at + 0.01, 0.2)       # the draw after the blow: 0.12 s
    return _fin(y, sr, 0.8, fade=0.005, tail=0.03)


ARMOR_KINDS = ("plates", "maille", "cuir")
ARMOR_MOTION = {"walk": 2.0, "run": 3.0}                    # steps per second


@sfx("armor", "Armour moved by the wearer: plates (micro-impacts of steel at each step), maille (a metal PhISEM shake), cuir (leather creak and rustle).",
     kind=("plates", " | ".join(ARMOR_KINDS)), motion=("walk", "walk | run | shift (one movement)"), dur=(2.0, "s"), seed=(0, "variation"))
def armor(sr=DEFAULT_SR, kind="plates", motion="walk", dur=2.0, seed=0):
    k = _kind(kind, ARMOR_KINDS, "armor")
    mo = _kind(motion, (*ARMOR_MOTION, "shift"), "armor motion")
    rng = np.random.default_rng(int(seed))
    s = int(seed)
    d = float(np.clip(dur, 0.3, 30.0))
    rate = ARMOR_MOTION.get(mo, 0.0)
    n = _n(d + 0.6)
    y = np.zeros(n)
    steps = np.arange(0.05, d, 1.0 / rate) if rate else np.array([0.05])
    if k == "plates":
        for t in steps:                                     # 3-6 plates touching in the 60 ms after the foot lands. 0.5 ms, over the steel range of 0.05-0.3: plates strapped on padding (at 0.3 detect_clicks fires on isolated ticks)
            for _ in range(int(rng.integers(3, 7))):
                _put(y, _tick("sy_wok", rng.uniform(1.0, 2.0), 0.5, rng, rng.uniform(0, 1)), t + rng.uniform(0.0, 0.06), rng.uniform(0.15, 0.5))
        _put(y, FO.creak(sr=P, material="leather", force="sway", level=0.5, dur=d, seed=s), 0.0, 0.05)
    elif k == "maille":
        # Coins in a mug approach maille. Rate: the wearer's; a stationary shift is one shake.
        _put(y, FO.shake(sr=P, preset="coins", energy=0.8, rate=rate, dur=d, size=0.8, seed=s), 0.0, 1.0)
        _put(y, FO.chain(sr=P, dur=d, pushes=max(int(rate * d), 1), weight=0.15, density=60.0, seed=s), 0.0, 0.25)
    else:
        _put(y, FO.creak(sr=P, material="leather", force="sway", level=0.6, dur=d, seed=s), 0.0, 0.8)
        _put(y, FO.cloth(sr=P, kind="rustle", dur=d, rate=rate or 1.0, weight=1.0, seed=s), 0.0, 0.4)
    return _fin(y, sr, 0.75, fade=0.005, tail=0.1)


GRAB_KINDS = ("shirt", "arm")
GRAB_MS = 5.0                                               # slap contact


@sfx("grab", "A hand grabbing: the cloth pulled (shirt) or the skin slapped first (arm).",
     kind=("arm", " | ".join(GRAB_KINDS)), force=(0.8, "0..1"), seed=(0, "variation"))
def grab(sr=DEFAULT_SR, kind="arm", force=0.8, seed=0):
    k = _kind(kind, GRAB_KINDS, "grab")
    rng = np.random.default_rng(int(seed))
    fc = float(np.clip(force, 0.05, 1.0))
    y = np.zeros(_n(0.6))
    if k == "arm":                                          # a 5 ms pulse on a body and a skin-like noise, the wet part of `_hit`
        p = modal.bang(GRAB_MS, fc)
        f = np.zeros(_n(0.25))
        f[:len(p)] = p
        body = _nm(modal.render_modes(f, modal.resolve("sy_stick", 0.25)))
        wet = np.zeros(len(f))
        wet[:len(p)] = p / p.max() * rng.uniform(-1, 1, len(p))
        _put(y, 0.8 * body + 0.5 * _nm(lfilter([0.3], [1.0, -0.6], wet)), 0.005, 1.0)
    _put(y, FO.cloth(sr=P, kind="rustle", dur=0.4, rate=4.0, weight=0.6, seed=int(seed)), 0.02, 0.5 if k == "arm" else 1.0)
    return _fin(y, sr, 0.75, fade=0.002, tail=0.05)


SPEAR_KINDS = ("poussee", "parade", "hampe")
SHAFT = 2.5                                                 # m, 2-3


def shaft_f1(length=SHAFT, d=0.03):
    """Free bar, Euler-Bernoulli: f1 = 22.37 / (2 pi L^2) (d / 4) sqrt(E / rho). The diameter stays 30 mm whatever the length, so 1 / L^2. `contact._make_body` scales a bar
    as 1 / L, a bar whose thickness grows with it, and clips f1 at 20 Hz (measured: 2.5 m and 3 m came out the same), so `contact.impact` cannot ring this shaft."""
    m = get_material("wood")
    return 22.37 / (2.0 * math.pi * float(length) ** 2) * (d / 4.0) * math.sqrt(m.young / m.density)


BAR_RATIOS = np.array([1.0, 2.757, 5.404, 8.933, 13.344, 18.637])   # free bar


def shaft_modes(length=SHAFT):
    return shaft_f1(length) * BAR_RATIOS


def _shaft_ring(length, velocity, seed, position=0.28):
    """The shaft's modes on a 2 ms wooden contact: decay 6 + f / 30 1/s, gains 1 / (k + 1), a sine for the mode shape at the strike point."""
    f = shaft_modes(length)
    k = np.arange(len(f))
    g = (0.1 + np.abs(np.sin((k + 1.5) * np.pi * float(position)))) / (k + 1.0)
    x = np.zeros(_n(0.9))
    p = modal.bang(modal.contact_time(2.0, float(velocity)), 1.0)
    x[_n(0.002):_n(0.002) + len(p)] = p
    return _nm(modal.render_modes(x, modal.Modes(f, 6.0 + f / 30.0, g, "zeros"))[:len(x)])


@sfx("spear", "A spear: poussee (a short thrust, the point landing, the shaft shaking), parade (a blow caught on the shaft or a flat shield), hampe (the shaft handled, sliding in the hand).",
     kind=("poussee", " | ".join(SPEAR_KINDS)), target=("flesh", "poussee: flesh|wood|armour"), length=(SHAFT, "m, 2-3"), shield=(0, "parade: 1 = on a wooden shield, a flat contact"), seed=(0, "variation"))
def spear(sr=DEFAULT_SR, kind="poussee", target="flesh", length=SHAFT, shield=0, seed=0):
    k = _kind(kind, SPEAR_KINDS, "spear")
    rng = np.random.default_rng(int(seed))
    s = int(seed)
    L = float(np.clip(length, 1.0, 5.0))
    if k == "poussee":                                      # 15 m/s over 0.2 s, 30 mm shaft: 100 Hz, and the point's small tone
        m = _n(0.2)
        v = 15.0 * np.sin(np.pi * np.arange(m) / m) ** 2 + 0.1
        y = np.zeros(_n(1.2))
        _put(y, _nm(modal.aeolian_tone(v, 0.03, 0.2, rng, "loq")[:m] * (v / 15.0) ** 3), 0.0, 0.35)
        _put(y, _strike("spear", _target(target), 0.8, rng)[:_n(0.9)], 0.12, 1.0)
        _put(y, _shaft_ring(L, 2.0, s), 0.12, 0.3)
        return _fin(y, sr, 0.8, fade=0.005, tail=0.03)
    if k == "parade":
        y = np.zeros(_n(1.3))
        if shield:                                          # 1 kg blade on a 30 cm flat: Tc from k = 2 E* a. A wooden shield
            _put(y, _hit("wood", "sy_sword1", 1.0, max(flat_contact(1.0, 0.15, "wood"), FLAT_FLOOR_MS), 1.0, rng)[:_n(1.0)], 0.0, 1.0)
        else:
            _put(y, _shaft_ring(L, 5.0, s, 0.5), 0.0, 1.0)
        _put(y, _tick("sy_sword1", 1.0, 0.3, rng, rng.uniform(0, 1)), 0.0, 0.5)
        return _fin(y, sr, 0.8, fade=0.002, tail=0.03)
    y = np.zeros(_n(1.4))                                   # hand slipping along it, and the grip closing on the shaft: 0.3 s
    _put(y, scrape(sr=P, dur=0.3, speed=[0.2, 0.8, 0.1], force=[0.3, 0.6, 0.5], roughness=0.7, body="bar", material="wood", size=L, seed=s), 0.0, 0.4)
    _put(y, _shaft_ring(L, 1.5, s, 0.35), 0.3, 1.0)
    return _fin(y, sr, 0.8, fade=0.005, tail=0.03)
