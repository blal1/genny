"""Firearms and everything around them: named weapons, handling, spent brass, bullet impacts, ricochets, passing bullets.

A weapon is a preset (cartridge, barrel, action, muzzle device); the sound is computed in pascals from `ballistics` (interior
ballistics, muzzle blast, N-wave, suppressor, ground, recorder) and `brass` (spent cases). Nothing is normalised to a common peak:
the recorder is a parameter and the file level follows the pressure at the listener. Sources: research/SOURCES.md.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from . import ballistics as B
from . import brass as BR
from . import contact as K
from . import filters as FL
from .core import DEFAULT_SR
from .foley import (C_AIR, CATCH, ENDSTOP, P, RATCHET, RIFLE_BODY, _barrel, _bpar8, _c01, _fin, _mclick, _n, _nm,
                    _noiseburst, _put, _shell, _strike)
from .physical import ambience as amb
from .physical import modal, space
from . import presets
from .presets import F
from .sfx import sfx

P0 = 101325.0

PRESETS = presets.schema("weapon", {
    "cart": F(ref="cartridge"), "barrel": F(float, lo=0.02, hi=2.0, unit="m"), "rate": F(float, lo=0.0, hi=200.0, unit="rounds/s"),
    "action": F(str, choices=("semi_gas", "open_bolt", "manual", "revolver", "lock")), "body": F(float, lo=0.1, hi=5.0),
    "muzzle": F(str, choices=("bare", "brake", "flash_hider", "can")), "wood": F(bool, required=False), "clip": F(int, lo=1, hi=200, required=False),
    "cadence": F(float, lo=0.05, hi=20.0, unit="rounds/s", required=False), "lock": F(tuple, items=(F(float, lo=0.0, hi=500.0, unit="ms"), F(str, choices=("flint", "match", "cap"))), required=False),
    "suppressed": F(float, lo=0.0, hi=1.0, required=False)})


class _View(Mapping):
    """A table that is computed from the live presets at each access, so a cartridge or weapon registered later is in it."""

    def __init__(self, keys, build):
        self._keys, self._build = keys, build

    def __getitem__(self, k):
        if k not in self._keys():
            raise KeyError(k)
        return self._build(k)

    def __iter__(self):
        return iter(self._keys())

    def __len__(self):
        return len(self._keys())

    def __contains__(self, k):
        return k in self._keys()


def _caliber(k: str) -> tuple:
    c, leg = B.CARTRIDGES[k], presets.get("cartridge", k)
    peak = leg["legacy_peak_pa"] if "legacy_peak_pa" in leg else B.blast_at(B.muzzle_state(k, c.l_ref), k, 3.0, 0.0)[0]
    return c.d_mm, c.l_mm, c.v_ref, peak, leg.get("legacy_dir_db", 0.0)


def _charge(k: str) -> tuple:
    c, leg = B.CARTRIDGES[k], presets.get("cartridge", k)
    return (leg["legacy_e_j"] if "legacy_e_j" in leg else 0.5e-3 * c.m_g * c.v_ref * c.v_ref), c.p_max_mpa


# caliber -> (diameter mm, length mm, speed m/s, blast peak Pa at 3 m, directivity dB), from the cartridge presets
CALIBERS = _View(B.CARTRIDGES.keys, _caliber)
_NO_MECH = ("manual", "revolver", "lock")


def _opts(p: dict) -> dict:
    o = {k: p[k] for k in ("wood", "clip", "cadence", "lock", "suppressed") if k in p}
    if p["action"] in _NO_MECH:
        o["manual"] = True
    if p["action"] == "open_bolt":
        o["open_bolt"] = True
    if p["action"] == "revolver":
        o["gap"] = True
    if p["muzzle"] == "brake":
        o["brake"] = True
    if B.CARTRIDGES[p["cart"]].pellets > 1:
        o["pellets"] = B.CARTRIDGES[p["cart"]].pellets
    return o


def _blast_ms(p: dict) -> float:
    st = B.muzzle_state(p["cart"], p["barrel"])
    _, t0, b = B.blast_at(st, p["cart"], 3.0, 0.0)
    return round(1e3 * t0 * (1.0 + 8.0 / b), 2)


# compatibility views of the presets: caliber, rounds/s, body scale, blast span ms (computed), options
WEAPONS = _View(PRESETS.keys, lambda k: (PRESETS[k]["cart"], PRESETS[k]["rate"], PRESETS[k]["body"], _blast_ms(PRESETS[k]), _opts(PRESETS[k])))
BARREL_MM = _View(PRESETS.keys, lambda k: PRESETS[k]["barrel"] * 1e3)
BODY_Q = 10.62       # the gun body's preset Q (pov)
# calibre -> muzzle energy J, peak chamber pressure MPa
CHARGE = _View(B.CARTRIDGES.keys, _charge)
REF_PEAK = 0.7
REF_PEAK_ENV = 0.5
POV_Q = 62.0
_REFC: dict = {}


def _refs() -> tuple:
    """The 5.56 on its 508 mm barrel (chamber pressure, burnt energy) and the .50's on-axis blast at 3 m, from the live cartridges."""
    key = (repr(B.CARTRIDGES["50bmg"]), repr(B.CARTRIDGES["556"]))
    if key not in _REFC:
        _REFC.clear()
        st = B.muzzle_state("556", 0.508)
        _REFC[key] = (B.blast_at(B.muzzle_state("50bmg", 1.143), "50bmg", 3.0, 0.0)[0], st["p_m"], st["e_b"],
                      B.blast_at(st, "556", ENV_REF_M, 30.0)[0])
    return _REFC[key]


def __getattr__(name: str):
    if name in ("REF_PA", "_PM_REF", "_EB_REF"):
        return _refs()[("REF_PA", "_PM_REF", "_EB_REF").index(name)]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


BOLT_PA_M = {"rifle": 20.0, "pistol": 10.0, "open_bolt": 30.0}      # pascals at 1 m, each bolt locking
MOTOR_PA_M, CLIP_PA_M, LOCK_PA_M, BRASS_PA_M = 2.0, 20.0, 5.0, 0.2
IMPACT_REF_J = 18050.0


def n_wave_pa(v: float, d_mm: float, l_mm: float, miss: float, c: float = C_AIR):
    """N-wave of a bullet: ``(peak Pa, duration s)`` at distance ``miss`` m from the trajectory, or ``(0, 0)`` when subsonic.
    Far-field law (miss > 1000 d); `ballistics.n_wave_pa` adds the near-field law."""
    m = v / c
    if m <= 1.0:
        return 0.0, 0.0
    d, l, b = d_mm * 1e-3, l_mm * 1e-3, max(float(miss), 0.05)
    k = m * m - 1.0
    return (0.53 * P0 * k ** 0.125 * d / (b ** 0.75 * l ** 0.25),
            1.82 * m * b ** 0.25 * d / (c * k ** 0.375 * l ** 0.25))


RHO_AIR, CD_BULLET = 1.225, 0.3


def drag_length(cal: str) -> float:
    """Metres over which a bullet loses 1/e of its speed under quadratic drag, 2 m / (rho Cd A); mass from muzzle energy and speed."""
    c = B.CARTRIDGES[cal]
    d, v = c.d_mm, c.v_ref
    m = 2.0 * (CHARGE[cal][0] if cal in CHARGE else 0.5e-3 * c.m_g * v * v) / (v * v)
    return 2.0 * m / (RHO_AIR * CD_BULLET * math.pi * (d * 1e-3) ** 2 / 4.0)


def shock_arrival(v: float, x: float, y: float, c: float = C_AIR) -> float | None:
    """Seconds from the shot until the Mach cone reaches a listener ``x`` m along the trajectory and ``y`` m
    off it; ``None`` when subsonic or when the listener is behind the cone (it never gets there)."""
    if v <= c or x < 0:
        return None
    theta = math.asin(c / v)
    if x < y * math.tan(theta):
        return None
    return x / v + y * math.cos(theta) / c


def _n_wave_shape(dur: float) -> np.ndarray:
    """Rise in one sample (under 2 us in reality), linear to -1, back in one sample."""
    return np.append(np.linspace(1.0, -1.0, max(2, int(round(dur * P)))), 0.0)


def _crack(v, d_mm, l_mm, miss, rng, height=1.5, wake=True, near=False) -> np.ndarray:
    """The passing bullet in Pa: N-wave, ground reflection (dt = path difference / c) and the broadband wake behind it
    (2 kHz up, 30 ms, -20 dB). ``near``: the near-field law inside 1000 diameters."""
    p, t = (B.n_wave_pa if near else n_wave_pa)(v, d_mm, l_mm, miss)
    if p <= 0:
        return np.zeros(1)
    return _cone(p * _n_wave_shape(t), max(miss, 0.05), rng, height, wake)


def _cone(w: np.ndarray, b: float, rng, height=1.5, wake=True) -> np.ndarray:
    """Add the ground reflection and the wake to an N-wave ``w`` (Pa) heard ``b`` m from the path."""
    dt = (math.hypot(b, 2.0 * height) - b) / C_AIR
    y = np.zeros(len(w) + _n(dt) + _n(0.05))
    y[:len(w)] += w
    _put(y, amb.lop(w, 6000.0), dt, 0.6)
    if wake:
        m = _n(0.03)
        _put(y, amb.hip(rng.uniform(-1, 1, m), 2000.0) * (1.0 - np.arange(m) / m) ** 2, len(w) / P, 0.1 * float(np.max(np.abs(w))))
    return y


# Measured muzzle-blast peaks of a rifle every 15 degrees, dB re the axis; 180 continues the last slope
BLAST_SHAPE_DB = B.BLAST_SHAPE_DB


def _blast_gain(a_db: float, azimuth: float) -> float:
    """0 in front, -A behind, the measured shape between, scaled to A."""
    s = float(np.interp(azimuth, np.arange(0.0, 181.0, 15.0), BLAST_SHAPE_DB)) / -BLAST_SHAPE_DB[-1]
    return 10.0 ** (a_db * s / 20.0)


def _friedlander(a: float, n: int) -> np.ndarray:
    """Blast wave at a point: (1 - t/a) e^(-t/a), a = positive phase; decay constant 1."""
    u = np.arange(n) / (P * a)
    return (1.0 - u) * np.exp(-u)


def _gas_tail(t60: float, rng) -> np.ndarray:
    """Gas still leaving and mixing after the front (pov layers): decaying low-passed noise."""
    n = _n(t60)
    return amb.lop(rng.uniform(-1, 1, n), 2500.0) * np.exp(-6.9 * np.arange(n) / n) * (1.0 - np.arange(n) / n)


def _pov_shot(p: dict, st: dict, rng, supp, subsonic, action, cycle, ring, direct, weight, k_t0, L: float) -> np.ndarray:
    """The shooter's ears, in Pa: the gun's body and the breech layers through the body filters, at the level of the blast
    0.35 m behind the muzzle and 165 degrees off the axis."""
    cart, c = p["cart"], B.CARTRIDGES[p["cart"]]
    pk, t0, b = B.blast_at(st, cart, L + 0.35, 165.0, 30.0, k_t0)
    if p["muzzle"] == "brake":
        pk *= 10.0 ** (B.brake_boost_db(165.0) / 20.0)
    if subsonic:
        pk *= 10.0 ** (-4.0 / 20.0)
    t0 += weight * 1e-3 / 2.28
    if supp > 0:
        pk, t0 = B.suppress(pk, t0, cart, (c.supp_db if c.supp_db is not None else B.SUPPRESSOR_DEFAULT_DB) * supp,
                            ejection_port=p["action"] in ("semi_gas", "open_bolt"))
    o = _opts(p)
    body = p["body"]
    bb = max(1.7 * t0, 1e-3)
    v = min(st["v"], 310.0) if subsonic else st["v"]
    delay = L / v
    scale = (st["e_b"] / _refs()[2]) ** (1.0 / 3.0)
    n = _n(0.4 * max(ring, 1.0) + 2.0 * bb)
    x = np.zeros(n)
    mech = np.zeros(n)
    tg = 0.0
    if o.get("gap"):
        _put(x, _barrel(max(0.6 * bb, 1e-3)), 0.0, 0.5)
        tg = 2e-4
    _put(x, _shell(max(bb, 1e-3)), tg, 0.6)
    _put(x, _barrel(max(0.6 * bb, 1e-3)), tg + delay, 0.5)
    _put(x, _friedlander(t0, min(n, int(12 * t0 * P))), tg + delay, 0.5)
    _put(x, _barrel(8.0 / (100.0 * body)), tg + delay, min(0.25, 0.1 * scale))
    _put(x, _gas_tail(max(0.005 * scale, bb), rng), tg + delay, 0.25)
    _put(x, _noiseburst(max(0.6 * bb, 1e-3), rng), tg + delay + 0.35 * bb, 0.8)
    if o.get("brake"):
        _put(x, _noiseburst(2.0 * bb, rng), tg + delay + 1e-3, 0.8)
    if action > 0 and not o.get("manual"):
        nz = amb.hip(rng.uniform(-1, 1, n), 300.0)
        cc = min(cycle, 0.1)
        mech += 0.5 * action * (_mclick(nz, CATCH, 0.45 * cc, 0.33) + _mclick(nz, ENDSTOP, 0.0 if o.get("open_bolt") else 0.7 * cc, 0.33))
    if o.get("wood") and action > 0:
        _put(mech, _nm(_strike("sy_stick", 0.3, 1.5, 0.0, rng, dur=0.12)), 0.09, 0.1 * action)
    centres = [f * body * rng.uniform(0.97, 1.03) for f in RIFLE_BODY]
    q = BODY_Q * ring
    centres.append(C_AIR / (2.0 * L))
    q *= (L * 1e3 / 508.0) ** 0.25
    drive = 4.8 * math.sqrt(st["p_m"] / _refs()[1])

    def voice(u):
        return np.tanh(drive * (ring * _bpar8(u, centres, q=q) + direct * u))
    return _nm(voice(x + mech)) * pk


CAN_HISS_HZ, CAN_HISS_Q, CAN_HISS_RMS, CAN_HISS_S = 1500.0, 0.7, 0.1, 0.012


def _can_hiss(pk: float, rng) -> np.ndarray:
    """Gas leaving the suppressor after the shot: band noise around 1.5 kHz, rms 10 % of the treated peak, dying over 12 ms."""
    n = _n(CAN_HISS_S)
    w = amb.bp(rng.uniform(-1.0, 1.0, n), CAN_HISS_HZ, CAN_HISS_Q) * np.exp(-3.0 * np.arange(n) / n)
    return w * (CAN_HISS_RMS * pk / math.sqrt(float(np.mean(w * w)) + 1e-30))


def _shot(weapon, rng, r, az, miss, suppress, subsonic, action, crack, cycle, ring=1.0, direct=0.35, weight=0.0, pov=False,
          ground="asphalt", flash=0.0, barrel_ring=0.0, barrel=0.0, k_t0=1.0) -> np.ndarray:
    """One round in pascals at the listener (first sample = first arrival): precursor, Friedlander blast of the muzzle gas,
    brake ports, flash, gas jet, suppressor chamber tone, then air absorption and ground effect; the bullet's N-wave and the bolt."""
    p = presets.resolve("weapon", weapon)
    cart, c = p["cart"], B.CARTRIDGES[p["cart"]]
    L = barrel if barrel > 0 else p["barrel"]
    st = B.muzzle_state(cart, L)
    supp = max(suppress, p.get("suppressed", 0.0))
    if pov:
        return _pov_shot(p, st, rng, supp, subsonic, action, cycle, ring, direct, weight, k_t0, L)
    v = min(st["v"], 310.0) if subsonic else st["v"]
    if supp > 0:
        v *= 1.03
    brake = p["muzzle"] == "brake"
    pk, t0, b = B.blast_at(st, cart, r, az, B.BRAKE_A_DB if brake else 30.0, k_t0)
    if brake:
        pk *= 10.0 ** (B.brake_boost_db(az) / 20.0)
    if subsonic:
        pk *= 10.0 ** (-4.0 / 20.0)
    t0 += weight * 1e-3 / 2.28
    pk0 = pk
    if supp > 0:
        pk, t0 = B.suppress(pk, t0, cart, (c.supp_db if c.supp_db is not None else B.SUPPRESSOR_DEFAULT_DB) * supp)
    lead = 1e-3
    parts = [(B.friedlander(pk, t0, b, None), lead)]
    if p["action"] == "revolver":
        parts.append((B.friedlander(0.5 * pk, t0, b, None), lead - 2e-4))
    pre, adv = B.precursor(st, cart, r, az, L, v)
    if adv > 0:
        parts.append((pre * (pk / pk0), lead - adv))
    if brake:
        parts += [(s, lead + d) for s, d in B.brake_pulses(pk, t0, b)]
    if flash > 0:
        s, d = B.flash_pulse(pk, t0, b, flash)
        parts.append((s, lead + d))
    jet = B.gas_jet(st, pk, rng)
    parts.append((jet, lead))
    if barrel_ring > 0:
        f = math.sqrt(B.GAMMA * B.R_GAS * st["t_gas"]) / (4.0 * L)
        parts.append((amb.bp(B.gas_jet(st, pk, rng), f, 8.0) * barrel_ring, lead))
    if supp > 0:
        parts.append((B.can_tone(pk, rng), lead))
        parts.append((_can_hiss(pk, rng), lead))
    x = np.zeros(max(int(t * P) + len(s) for s, t in parts) + 8)
    for s, t in parts:
        _put(x, s, t)
    x = B.ground(B.air_absorb(x, r), r, 1.5, 1.6, ground)
    th = math.radians(az)
    xx, yy = r * math.cos(th), r * math.sin(th)
    if miss > 0:
        yy = float(miss)
    yy = max(yy, 0.05)
    t_b = r / C_AIR
    waves = []
    if crack > 0 and az < 75.0 and v > C_AIR:
        d, l = c.d_mm, c.l_mm
        lam = drag_length(cart)
        for _ in range(c.pellets):
            w, ta = B.n_wave(v, d, l, xx, yy, lam, P)
            if ta is not None:
                w = B.air_absorb(_cone(w, yy, rng), yy, P)
                waves.append((w * crack / math.sqrt(float(c.pellets)), ta + (rng.uniform(-2e-4, 2e-4) if c.pellets > 1 else 0.0)))
    org = min([t_b - lead] + [ta for _, ta in waves])
    mech = None
    if p["action"] not in _NO_MECH and action > 0:
        bp = BOLT_PA_M["open_bolt"] if p["action"] == "open_bolt" else (BOLT_PA_M["pistol"] if L < 0.3 else BOLT_PA_M["rifle"])
        nz = amb.hip(rng.uniform(-1, 1, _n(0.4)), 300.0)
        cc = min(cycle, 0.1)
        tu = st["tau_b"] + 1e-3
        mech = _mclick(nz, CATCH, max(0.45 * cc, tu), 0.33) + _mclick(nz, ENDSTOP, tu if p["action"] == "open_bolt" else 0.7 * cc, 0.33)
        mech = mech * (action * bp / max(r, 1.0) / (float(np.max(np.abs(mech))) + 1e-12))
    out = np.zeros(max([len(x) + int((t_b - lead - org) * P)] + [len(w) + int((ta - org) * P) for w, ta in waves]) + _n(0.05) + (_n(0.4) if mech is not None else 0))
    _put(out, x, t_b - lead - org)
    for w, ta in waves:
        _put(out, w, ta - org)
    if mech is not None:
        _put(out, mech, t_b - org)
    return out


RENDER_HELP = ("heard = as a recording gets it: the place's reflections and the `recorder` chain | "
               "pressure = the bare pressure at the ear: no ground, no place, no recorder")
PLACE_HELP = "render=heard: where it is heard, none | street | forest | canyon | arena | cave | hall ... (room presets)"
RECORDER_HELP = "none (bare Pa) | field (measurement microphone) | camera (clips at clip_db, locks, limiter) | phone"


def _place(y: np.ndarray, place: str) -> np.ndarray:
    if str(place) == "none":
        return y
    from .reverbs import room
    w = room(np.concatenate([y, np.zeros(_n(2.0))]), P, mix=0.35, preset=str(place))
    return w.mean(axis=1) if w.ndim == 2 else w


ENVS = presets.schema("weapon_env", {
    "body_db": F(float, lo=-80.0, hi=30.0, unit="dB"), "body_ms": F(float, lo=1.0, hi=500.0, unit="ms"),
    "body_hz": F(float, lo=50.0, hi=20000.0, unit="Hz"),
    "boom_db": F(float, lo=-80.0, hi=30.0, unit="dB"), "boom_ms": F(float, lo=1.0, hi=2000.0, unit="ms"),
    "boom_hz": F(float, lo=20.0, hi=2000.0, unit="Hz"),
    "tail_db": F(float, lo=-80.0, hi=30.0, unit="dB"), "tail_ms": F(float, lo=0.5, hi=500.0, unit="ms"),
    "t60": F(float, lo=0.05, hi=6.0, unit="s"), "t60_hf": F(float, lo=0.05, hi=6.0, unit="s"),
    "tail_hz": F(float, lo=200.0, hi=20000.0, unit="Hz"),
    "echoes": F(list, of={"ms": F(float, lo=0.5, hi=3000.0, unit="ms"), "db": F(float, lo=-80.0, hi=6.0, unit="dB"),
                          "hz": F(float, lo=100.0, hi=20000.0, unit="Hz")}, max_len=32, required=False)})
ENV_HELP = "open_field | urban | range | indoor | forest | ... (weapon_env presets) or an inline dict; none or anechoic = dry, as before"
ENV_REF_M, ENV_GROW_DB, ENV_SPLIT_HZ, ENV_MAX_S = 10.0, 20.0, 2000.0, 3.0
_DRY = ("none", "anechoic", "None", "")


def _env_params(env):
    if env is None or (isinstance(env, str) and env in _DRY):
        return None
    return presets.resolve("weapon_env", env)


def _lp(x: np.ndarray, hz: float) -> np.ndarray:
    return sosfilt(butter(2, min(hz, 0.45 * P), "low", fs=P, output="sos"), x) if hz < 0.45 * P else x


def _band_energy(x: np.ndarray, g: float, db: float, hz: float) -> np.ndarray:
    """``x`` scaled so that its energy below ``hz`` is ``g`` 10^(db/10) times that of a unit impulse in the same band."""
    return x * math.sqrt(g * 10.0 ** (db / 10.0) * min(1.0, 2.0 * hz / P) / (float(np.sum(x * x)) + 1e-30))


def env_ir(env, r: float, seed: int = 0, azimuth: float = 30.0) -> np.ndarray:
    """Impulse response of a place for a listener ``r`` m from the source, first sample = the direct sound (1). An early dense
    part (``body_*``), a low one (``boom_*``), discrete echoes, then a late tail that builds up over ``tail_ms`` and decays with
    ``t60`` below 2 kHz and ``t60_hf`` above. Energies are given re the direct sound at 10 m and 30 degrees off the bore; they
    grow as r^2 (direct ~ 1/r^2, the field of the place does not) and as the muzzle's directivity takes the direct sound
    away from ``azimuth``, at most +20 dB."""
    e = _env_params(env)
    if e is None:
        return np.ones(1)
    rng = np.random.default_rng(int(seed) + 7919)
    d = B._dir_gain(30.0, 30.0) / B._dir_gain(float(np.clip(azimuth, 0.0, 180.0)), 30.0)
    g = min((max(float(r), 0.1) / ENV_REF_M) ** 2 * d * d, 10.0 ** (ENV_GROW_DB / 10.0))
    echoes = e.get("echoes") or []
    span = max(e["body_ms"] * 1e-3, e["boom_ms"] * 1e-3, e["tail_ms"] * 1e-3 + max(e["t60"], e["t60_hf"]), max([x["ms"] for x in echoes], default=0.0) * 1e-3 + 0.05)
    h = np.zeros(_n(min(span, ENV_MAX_S)))
    h[0] = 1.0
    for k in ("body", "boom"):
        m = min(_n(e[k + "_ms"] * 1e-3), len(h) - 1)
        b = _lp(_lp(rng.standard_normal(m), e[k + "_hz"]), e[k + "_hz"]) * np.exp(-6.9 * np.arange(m) / m)
        _put(h, _band_energy(b, g, e[k + "_db"], e[k + "_hz"]), 1.0 / P)
    t = np.arange(len(h) - 1) / P
    nz = rng.standard_normal(len(t))
    lo = _lp(nz, ENV_SPLIT_HZ)
    w = (1.0 - np.exp(-3.0 * t / (e["tail_ms"] * 1e-3))) * (lo * np.exp(-6.9 * t / e["t60"]) + (nz - lo) * np.exp(-6.9 * t / e["t60_hf"]))
    _put(h, _band_energy(_lp(w, e["tail_hz"]), g, e["tail_db"], e["tail_hz"]), 1.0 / P)
    for x in echoes:
        k = np.zeros(_n(0.01))
        k[0] = 1.0
        k = _lp(k, x["hz"])
        _put(h, k * (math.sqrt(g) * 10.0 ** (x["db"] / 20.0) / (float(np.sum(k)) + 1e-30)), x["ms"] * 1e-3)
    return h


def _env(y: np.ndarray, env, r: float, seed: int = 0, azimuth: float = 30.0) -> np.ndarray:
    if _env_params(env) is None:
        return y
    return fftconvolve(y, env_ir(env, r, seed, azimuth))


def _check_render(render: str) -> bool:
    if render not in ("heard", "pressure"):
        raise ValueError(f"unknown render {render!r}: heard|pressure")
    return render == "heard"


def _finish(y, sr, r, level, fade, tail=0.0, wet=False) -> np.ndarray:
    """``level="peak"``: 0.9 at the crest. ``"relative"``: pascals over a reference, so every weapon keeps its physical level;
    never over 0.89. Dry (no ``env``): the .50's on-axis blast at 3 m sits at 0.7. With an ``env``: the 5.56's blast at 10 m
    and 30 degrees sits at 0.5, so a .22 or a suppressed gun stays audible."""
    if level == "peak":
        return _fin(y, sr, 0.9, fade=fade, tail=tail)
    if level != "relative":
        raise ValueError(f"unknown level {level!r}: relative|peak")
    y = _fin(y * (REF_PEAK_ENV / _refs()[3] if wet else REF_PEAK / _refs()[0]), sr, None, fade=fade, tail=tail)
    return y * min(1.0, 0.89 / (float(np.max(np.abs(y))) + 1e-12))


def _clip_ping(rng) -> np.ndarray:
    """The M1's ping: a light steel clip at 2-3 kHz, three modes dying within 100 ms."""
    f = np.zeros(_n(0.2))
    f[0] = 1.0
    fm = np.array([2300.0, 3100.0, 4650.0]) * rng.uniform(0.98, 1.02)
    return _nm(modal.render_modes(f, modal.Modes(fm, np.array([25.0, 30.0, 40.0]), np.array([1.0, 0.6, 0.3]), "vdd")))


def _lock(kind: str, delay: float, rng) -> np.ndarray:
    """What precedes the charge of an old gun: the lock falling, then the priming flash for ``delay`` ms."""
    n = _n(delay * 1e-3 + 0.05)
    nz = amb.hip(rng.uniform(-1, 1, n), 300.0)
    y = _mclick(nz, ENDSTOP, 0.0, 0.5) + _mclick(nz, CATCH, 0.0, 0.5 if kind == "flint" else 0.2)
    m = _n(delay * 1e-3)
    if kind != "cap":
        _put(y, amb.hip(rng.uniform(-1, 1, m), 2500.0) * np.linspace(0.3, 1.0, m), 0.0, 0.25)
    return y


_CASE_OF = presets.schema("case_of", {"case": F(str)}, shape=lambda d: d["case"])


def _fall(case: str, floor: str, rng, height=1.2) -> np.ndarray:
    return BR.render_case(_case_geom(case), floor, height, rng, rate=P)


@sfx("firearm", "A named firearm in physical units: muzzle blast from the interior ballistics (cartridge, barrel), the bullet's own N-wave and where it reaches you, ground, recorder, bolt cycling, burst rate, suppressor, subsonic rounds.",
     weapon=("ar15", "|".join(PRESETS)), distance=(10.0, "m from the shooter"),
     azimuth=(30.0, "degrees off the line of fire: 0 you are shot at, 180 behind the shooter; no crack past 75"),
     miss=(0.0, "m the bullet passes from you (0 = from distance and azimuth)"), shots=(1, "rounds (up to 2000)"),
     rate=(0.0, "rounds/s (0 = the weapon's own)"), suppressor=(0.0, "0..1: 1 = the cartridge's can reduction at the muzzle (22-32 dB); the crack stays"),
     subsonic=(False, "true = slowed to 310 m/s: no crack, a little quieter"), action=(0.6, "0..1 gain of the bolt cycling, 1 = its physical level (10-30 Pa at 1 m)"),
     crack=(1.0, "0..1 level of the N-wave against its true level"), echo=(0.0, "0..1 street / treeline echoes"),
     first_pop=(3.0, "dB louder on a suppressed gun's first round, half that on the second"),
     pov=(False, "true = the shooter's ears: 0.35 m behind the muzzle, the body ringing 300 ms, the mechanism up, no crack"),
     ring=(1.0, "pov only: how long the gun's body rings, Q = 10.62 x ring"),
     weight=(0.0, "ms added to the blast wave (0-200): a longer shot sounds heavier"),
     barrel=(0.0, "m of barrel (0 = the weapon's own): the speed and the blast follow"),
     ground=("asphalt", "ground between the shooter and you: none | asphalt | grass | ..."),
     treeline=(0.0, "Hz: echoes off a wood scatter as f^4, so the echo is high-passed (2nd order, f^2 in amplitude) and brighter than the direct shot; 0 = street echoes, 1500 is a forest edge"),
     flash=(0.0, "0..1 secondary flash pressure 2 ms after the shot"),
     barrel_ring=(0.0, "0..1 resonance of the bore after the gas leaves"),
     k_t0=(1.0, "scale of the blast's positive phase (0.6 short .. 1.7 long)"),
     brass=(0.0, "0..1 gain of the case falling and ringing after the shot"),
     recorder=("field", RECORDER_HELP), clip_db=(0.0, "dB SPL where the recorder clips (0 = its own)"),
     level=("relative", "relative = pressure over a reference shot (the 5.56 at 10 m sits at 0.5 with an env, the .50 at 3 m at 0.7 without), nothing normalised | peak = every file at 0.9"),
     render=("heard", RENDER_HELP), env=("open_field", "render=heard: " + ENV_HELP), place=("none", PLACE_HELP), seed=(0, "variation"))
def firearm(sr=DEFAULT_SR, weapon="ar15", distance=10.0, azimuth=30.0, miss=0.0, shots=1, rate=0.0, suppressor=0.0,
            subsonic=False, action=0.6, crack=1.0, echo=0.0, first_pop=3.0, pov=False, ring=1.0, weight=0.0, barrel=0.0, ground="asphalt",
            treeline=0.0, flash=0.0, barrel_ring=0.0, k_t0=1.0, brass=0.0, recorder="field", clip_db=0.0, level="relative", render="heard",
            env="open_field", place="none", seed=0):
    heard = _check_render(str(render))
    rng = np.random.default_rng(int(seed))
    pr = presets.resolve("weapon", weapon)
    L = float(barrel) if barrel > 0 else pr["barrel"]
    r = float(np.clip(distance, 1.0, 5000.0))
    az = float(np.clip(abs(float(azimuth)), 0.0, 180.0))
    shots = int(np.clip(shots, 1, 2000))
    ring = float(np.clip(ring, 0.5, 6.0))
    direct = 0.35
    if pov:
        r, az, crack, action, ring, direct = 1.0, 180.0, 0.0, _c01(action) * 3.0, max(ring, POV_Q / BODY_Q), 0.1
    quiet = suppressor > 0 or pr.get("suppressed", 0.0) > 0
    per_s = float(rate) if rate > 0 else (pr["rate"] if pr["rate"] > 0 else pr.get("cadence", 2.0))
    gap = 1.0 / max(per_s, 0.05)
    gr = str(ground) if heard and not pov else "none"
    one = [_shot(weapon, rng, r, az, float(miss), _c01(suppressor), bool(subsonic), float(action) if pov else _c01(action), _c01(crack), gap, ring, direct,
                 float(np.clip(weight, 0.0, 200.0)), bool(pov), gr, _c01(flash), _c01(barrel_ring), float(barrel), float(np.clip(k_t0, 0.3, 3.0)))
           for _ in range(min(shots, 8))]
    lock = pr.get("lock")
    pre = _nm(_lock(lock[1], lock[0], rng)) * LOCK_PA_M / r if lock else None
    start = lock[0] * 1e-3 if lock else (0.0 if pov else 1e-3)
    fall = [_fall(_CASE_OF[pr["cart"]], "concrete", rng) for _ in range(min(shots, 3))] if brass > 0 and pr["cart"] in _CASE_OF else []
    y = np.zeros(_n(start + shots * gap * 1.03) + max(map(len, one)) + (_n(4.5) if fall else 0))
    t = start
    for k in range(shots):
        g = 1.0 + 0.07 * rng.standard_normal()                # +-5-10 % shot to shot
        if quiet and k < 2:
            g *= 10.0 ** (float(first_pop) / (20.0 if k == 0 else 40.0))
        _put(y, one[k % len(one)], t, g)
        if pr.get("clip") and (k + 1) % pr["clip"] == 0:
            _put(y, _clip_ping(rng), t + 0.06, CLIP_PA_M / r)
        if fall:
            _put(y, fall[k % len(fall)], t + 0.35 + rng.uniform(0.0, 0.1), BRASS_PA_M * _c01(brass) / r)
        t += gap * (1.0 + 0.01 * rng.standard_normal())
    if pre is not None:
        _put(y, pre, 0.0)
    if echo > 0:
        e = amb.outdoor_echoes(y) - 0.2 * y
        if treeline > 0:
            e = FL.highpass(e, float(np.clip(treeline, 100.0, 8000.0)), sr, order=2)
        y = y + _c01(echo) * 4.0 * e
    wet = heard and _env_params(env) is not None
    if heard:
        y = B.recorder(_place(_env(y, env, r, int(seed), az), place), str(recorder), float(clip_db) if clip_db > 0 else None)
    return _finish(y, sr, r, str(level), 0.0005, 0.03, wet)


@sfx("minigun", "Rotary machine gun: the motor spins the barrels up, the fire merges into a tearing buzz, then it winds down (fire=false: the spin alone).",
     dur=(3.0, "s of firing"), rate=(50.0, "rounds/s: 33-100 (M134, 2000-6000 rounds/min)"), spin=(0.4, "s to spin up"),
     fire=(True, "false = barrels turning, no rounds"), distance=(10.0, "m"), azimuth=(30.0, "degrees off the line of fire"),
     whine_ratio=(24.0, "motor whine in rotor turns: the gear ratio, 24 = 4 per round"), brass=(1.0, "0..1 gain of the cases raining down"),
     ground=("asphalt", "ground between the gun and you"), recorder=("field", RECORDER_HELP), clip_db=(0.0, "dB SPL where the recorder clips (0 = its own)"),
     level=("relative", "relative|peak, as in firearm"), render=("heard", RENDER_HELP), env=("open_field", "render=heard: " + ENV_HELP),
     place=("none", PLACE_HELP), seed=(0, "variation"))
def minigun(sr=DEFAULT_SR, dur=3.0, rate=50.0, spin=0.4, fire=True, distance=10.0, azimuth=30.0, whine_ratio=24.0, brass=1.0, ground="asphalt",
            recorder="field", clip_db=0.0, level="relative", render="heard", env="open_field", place="none", seed=0):
    """Six barrels: one bolt per round, so the rotor runs at rate/6 (5.6-16.7 Hz). The rounds are the ordinary shot of the
    preset, summed; the motor whine, the bolts and the cases are in pascals at 1 m over the distance."""
    heard = _check_render(str(render))
    rng = np.random.default_rng(int(seed))
    r = float(np.clip(distance, 1.0, 5000.0))
    rate = float(np.clip(rate, 10.0, 120.0))
    spin, dur = float(np.clip(spin, 0.05, 3.0)), float(np.clip(dur, 0.1, 60.0))
    down = 0.8
    n = _n(spin + dur + down)
    t = np.arange(n) / P
    spin_env = np.where(t < spin + dur, np.clip(t / spin, 0.0, 1.0), np.clip((spin + dur + down - t) / down, 0.0, 1.0) ** 2)
    ph = np.cumsum(rate * spin_env) / P
    idx = np.flatnonzero(np.diff(np.floor(ph)) > 0)
    fw = float(whine_ratio) / 6.0 * ph
    whine = (np.sin(2 * np.pi * fw) + 0.5 * np.sin(4 * np.pi * fw) + 0.25 * np.sin(6 * np.pi * fw)) * spin_env
    shots_i = [i for i in idx if spin <= i / P < spin + dur] if fire else []
    rain = (_n(4.0) if (fire and brass > 0) else 0)
    y = np.zeros(n + rain)
    y[:n] = whine * MOTOR_PA_M / r
    nz = amb.hip(rng.uniform(-1, 1, _n(0.05)), 300.0)
    tick = _nm(_mclick(nz, CATCH, 0.0, 0.33) + _mclick(nz, ENDSTOP, 0.0, 0.15))
    for i in idx:
        _put(y, tick, i / P, BOLT_PA_M["rifle"] / r)
    if fire:
        az = abs(float(azimuth))
        pool = [_shot("minigun", rng, r, az, 0.0, 0.0, False, 0.0, 1.0, 1.0 / rate, ground=str(ground) if heard else "none") for _ in range(6)]
        cases = [_fall("762x51", "concrete", rng, 1.0) for _ in range(3)] if brass > 0 else []
        for k, i in enumerate(shots_i):
            _put(y, pool[k % 6], i / P, 1.0 + 0.07 * rng.standard_normal())
            if cases:
                _put(y, cases[k % 3], i / P + rng.uniform(0.3, 0.5), BRASS_PA_M * _c01(brass) / r * rng.uniform(0.5, 1.0))
    wet = heard and _env_params(env) is not None
    if heard:
        y = B.recorder(_place(_env(y, env, r, int(seed), abs(float(azimuth))), place), str(recorder), float(clip_db) if clip_db > 0 else None)
    return _finish(y, sr, r, str(level), 0.001, 0.05, wet)


# ================================================================== handling
# (cluster, ms, gain[, frequency scale]) or ("slide", start ms, length ms, frequency scale). All times and gains are
# except the totals of slide_rack (450 ms) and pump (recording of 1.9 s with the shot): Maher, AES 2024.
_C = {"catch": CATCH, "stop": ENDSTOP, "ratchet": RATCHET}
MECH = {
    "mag_out": [("catch", 0, 1), ("ratchet", 8, 1), ("slide", 15, 75, 0.6)],
    "mag_in": [("slide", 0, 80, 0.7), ("stop", 80, 1), ("catch", 88, 1)],
    "slide_rack": [("catch", 0, 1), ("slide", 0, 120, 1), ("stop", 120, 1), ("slide", 200, 70, 1.1), ("stop", 280, 1.2), ("catch", 290, 1)],
    "slide_lock": [("stop", 0, 1, 1.2)], "slide_release": [("stop", 0, 1.2), ("catch", 2, 1), ("slide", 3, 40, 1.2)],
    "charging_handle": [("catch", 0, 1), ("slide", 0, 200, 1), ("ratchet", 90, 0.7), ("stop", 200, 1), ("slide", 350, 90, 1.1), ("stop", 450, 1.2), ("catch", 460, 1), ("ratchet", 465, 0.5)],
    "bolt_open": [("catch", 0, 1), ("slide", 10, 150, 0.8), ("stop", 160, 1)],
    "bolt_close": [("slide", 0, 130, 0.9), ("stop", 130, 1), ("catch", 150, 1), ("stop", 165, 0.7)],
    "pump": [("catch", 0, 0.7), ("slide", 0, 250, 0.6), ("stop", 250, 1.2), ("slide", 450, 200, 0.7), ("stop", 650, 1.4), ("catch", 660, 0.6)],
    "lever": [("catch", 0, 0.8), ("slide", 0, 200, 0.7), ("stop", 200, 1), ("slide", 300, 200, 0.8), ("stop", 500, 1.2), ("catch", 520, 1)],
    "revolver_open": [("catch", 0, 1), ("slide", 5, 40, 0.9), ("stop", 45, 0.5)],
    "revolver_close": [("slide", 0, 30, 0.9), ("stop", 30, 0.8), ("catch", 35, 1)],
    "revolver_spin": [("ratchet", 25 * k + k * k * 3, 1 - k * 0.03) for k in range(18)],
    "revolver_cock": [("catch", 0, 1), ("ratchet", 60, 1), ("catch", 75, 1.2)],
    "round_insert": [("slide", 0, 30, 0.7), ("stop", 30, 0.6)],
    "shell_insert": [("slide", 0, 40, 0.5), ("stop", 40, 0.7), ("ratchet", 45, 0.5)],
    "clip_strip": [e for k in range(5) for e in (("slide", 30 * k, 20, 0.8), ("stop", 30 * k + 20, 0.4))] + [("catch", 170, 1)],
    "safety": [("catch", 0, 0.5, 1.1)], "selector": [("catch", 0, 0.5, 1.1), ("catch", 40, 0.5, 1.1)],
    "dry_fire": [("catch", 0, 1), ("stop", 2, 1)], "trigger": [("catch", 0, 0.25)], "hammer_drop": [("stop", 0, 1, 0.8), ("catch", 3, 1)],
    "jam": [("slide", 0, 60, 0.9), ("stop", 60, 0.4), ("slide", 62, 80, 0.5)],
    "belt_feed": [("stop", 0, 0.6, 0.8), ("ratchet", 12, 0.7, 0.8)],
    "feed_cover_open": [("catch", 0, 1), ("slide", 0, 200, 0.6), ("stop", 200, 1)],
    "feed_cover_close": [("slide", 0, 200, 0.6), ("stop", 200, 1.3), ("catch", 210, 1)],
    "mag_drop": [("catch", 0, 1), ("ratchet", 8, 1), ("slide", 15, 75, 0.6)],
    "revolver_eject": [("slide", 0, 30, 0.9)],
    "extract": [("catch", 0, 0.6, 0.9), ("slide", 5, 40, 1.2)],
    "link_eject": [], "belt_rattle": [],                     # nothing through the gun's body: see _mech_extras
}
_T = lambda *parts: list(parts)
RELOADS = {
    "reload_pistol": _T(("mag_out", 0.0), ("mag_in", 0.9), ("slide_release", 1.3)),
    "reload_rifle": _T(("mag_out", 0.0), ("mag_in", 1.2), ("charging_handle", 1.8)),
    "reload_shotgun": _T(("shell_insert", 0.0), ("shell_insert", 0.55), ("shell_insert", 1.1), ("pump", 1.7)),
    "reload_revolver": _T(("revolver_open", 0.0), ("revolver_spin", 0.3), *[("round_insert", 1.3 + 0.35 * k) for k in range(6)], ("revolver_close", 3.6)),
    "reload_lmg": _T(("feed_cover_open", 0.0), ("mag_out", 0.5), ("belt_feed", 1.4), ("belt_feed", 1.55), ("feed_cover_close", 2.0), ("charging_handle", 2.6)),
    "reload_bolt": _T(("bolt_open", 0.0), ("clip_strip", 0.4), ("bolt_close", 0.9)),
    "bolt_cycle": _T(("bolt_open", 0.0), ("extract", 0.2), ("bolt_close", 0.53)),   # 700 ms in all
    "jam_clear": _T(("jam", 0.0), ("slide_rack", 0.8)),      # M22: M21, two blows (_mech_extras), M4, the case
    "reload_musket": _T(("safety", 0.0), ("revolver_cock", 20.0)),    # 20-60 s (Wikipedia, arquebus); the steps: _mech_extras
}
MECH_BODY = {"pistol": 1.4, "smg": 1.25, "rifle": 1.0, "shotgun": 0.75, "lmg": 0.9, "hmg": 0.5, "revolver": 1.3, "musket": 0.6}


CLUSTER_WINDOW_MS = (35.0, 14.0, 35.0)
FINISHES = {"steel": (1.0, 1.0), "blued": (0.95, 0.9), "nickel": (1.05, 1.1), "aluminium": (0.8, 0.9), "polymer": (0.35, 0.7), "wood": (0.25, 0.6)}


def _cluster_times(ev, k: float, rng=None, spread: float = 0.0) -> dict:
    """Event ms -> ms at speed 1/k. A faster hand brings the clusters closer, not the clicks inside one: only the start of
    each scales. A gap over 40 ms starts a cluster; with ``spread`` each cluster start is drawn in its own window
    (CLUSTER_WINDOW_MS: 1-36, 87-101, 348-383 ms), the 1st, 2nd, 3rd and later clusters."""
    out, base, prev, i = {}, 0.0, None, -1
    for t in sorted({e[1] for e in ev}):
        if prev is None or t - prev > 40.0:
            base, i = t, i + 1
            jit = float(spread) * CLUSTER_WINDOW_MS[min(i, 2)] * float(rng.uniform(0.0, 1.0)) if rng is not None and spread > 0 and i > 0 else 0.0
        out[t] = (base + jit) * k + (t - base)
        prev = t
    return out


def _mech(action: str, k: float, rng, finish: str = "steel", spread: float = 0.0) -> np.ndarray:
    ev = MECH[action]
    at = _cluster_times(ev, k, rng, spread)
    fs, fg = FINISHES[finish]
    end = max((at[e[1]] + (e[2] * k if e[0] == "slide" else 0) for e in ev), default=0.0) * 1e-3 + 0.2
    n = _n(end)
    x = np.zeros(n)
    nz = amb.hip(rng.uniform(-1, 1, n), 300.0)
    for e in ev:
        if e[0] == "slide":                                  # Farnell's slide: hip 1000 -> bp(4500 + 200 line, Q4) x line^2
            m = _n(e[2] * 1e-3 * k)
            ln = np.arange(m) / m
            _put(x, amb.bp(amb.hip(rng.uniform(-1, 1, m), 1000.0), (4500.0 + 200.0 * ln) * e[3] * fs, 4.0) * ln ** 2, at[e[1]] * 1e-3, 1.2 * fg)
        else:
            f = e[3] if len(e) > 3 else 1.0
            parts = tuple((a * f * fs, q, d) for a, q, d in _C[e[0]])
            x += _mclick(nz, parts, at[e[1]] * 1e-3, 0.33 * e[2] * fg)
    return x


def _floor_hit(material: str, floor: str, seed: int) -> np.ndarray:
    """A magazine landing: `bounce` on the floor's surface and restitution; soft ground muffles it (FLOORS)."""
    surface, _, e, lop = FLOORS[floor]
    y = K.bounce(P, material=material, size=0.15, surface=surface or "wood", first_interval=0.2, restitution=e, seed=seed)
    return amb.lop(y, float(lop)) if lop else y


def _mech_extras(action: str, weapon: str, k: float, seed: int, floor: str, mag: str, rounds: int) -> list:
    """What falls out or is hit, outside the gun's body: (signal, seconds, gain). All timing estimated but the belts'
    (M249 850 rounds/min = 70.6 ms; M2HB 450-600 rounds/min = 100-133 ms)."""
    from . import foley as Fo
    rng = np.random.default_rng(seed + 1)
    scratch = lambda: K.scrape(P, dur=0.08, speed=0.8, material="steel", size=0.2, seed=seed)     # M21: 80 ms
    if action == "mag_drop":                                 # M2: t = 400 ms
        return [(_floor_hit(mag, floor, seed), 0.4 * k, 0.5)]
    if action == "revolver_eject":                           # M13: fall of 0.5 m = sqrt(2 h / g) = 0.32 s; 6 cases 15 ms apart
        return [(casing_physical(P, "9mm", floor, 0.5, 1, 0.0, False, seed + i), 0.03 * k + 0.32 + 0.015 * i, 0.8) for i in range(6)]
    if action == "jam":
        return [(scratch(), 0.062 * k, 0.3)]
    if action == "jam_clear":
        hits = [(K.impact(P, material="steel", size=0.1, seed=seed + i), (0.3 + 0.2 * i) * k, 0.7) for i in range(2)]
        return [(scratch(), 0.062 * k, 0.3)] + hits + [(casing_physical(P, "556", floor, 1.0, 1, 0.0, False, seed), 1.2 * k, 0.5)]
    if action == "link_eject":
        gap = 0.115 if weapon == "hmg" else 0.0706
        return [(amb.tinkle(rng, f_scale=1.3), i * gap * k, 0.5) for i in range(rounds)]
    if action == "belt_rattle":
        return [(Fo.chain(P, dur=1.0 * k, weight=0.4, density=40, seed=seed), 0.0, 0.8)]
    if action == "bolt_cycle":
        return [(casing_physical(P, "762", floor, 1.0, 1, 0.0, False, seed), 0.25 * k, 0.3)]
    if action == "reload_musket":
        m = _n(1.2)
        powder = amb.bp(rng.uniform(-1, 1, m), 5000.0, 0.8) * np.sin(np.pi * np.arange(m) / m)
        wad = amb.lop(rng.uniform(-1, 1, _n(0.5)), 1500.0) * np.sin(np.pi * np.arange(_n(0.5)) / _n(0.5))
        rod = lambda d, i: K.scrape(P, dur=d, speed=0.6, material="steel", size=0.6, seed=seed + i)
        out = [(powder, 1.5 * k, 0.2), (wad, 4.5 * k, 0.15), (Fo.clink(P, object="coin", size=1.0, force=0.6, seed=seed), 6.5 * k, 0.3),
               (rod(1.5, 0), 8.0 * k, 0.5), (rod(1.8, 1), 10.0 * k, 0.6), (rod(1.5, 2), 16.0 * k, 0.4)]
        return out + [(K.impact(P, material="steel", size=0.6, seed=seed + i), (12.5 + 0.8 * i) * k, 0.6) for i in range(3)]
    return []


@sfx("gun_mech", "Weapon handling: magazines, slides, bolts, pumps, levers, revolver cylinders, safeties, dry fire, jams, belts, or a whole reload (reload_*).",
     action=("slide_rack", "|".join((*MECH, *RELOADS))), weapon=("rifle", "|".join(MECH_BODY) + ": the body it rings through"),
     speed=(1.0, "0.5 slow and deliberate .. 2 fast: the clicks inside a cluster keep their gaps"),
     floor=("concrete", "|".join(("concrete", "wood", "metal", "tile", "dirt", "grass")) + ": where magazines and cases land"),
     mag=("plastic", "plastic|steel: the magazine of mag_drop"), rounds=(6, "links of link_eject, one per round"),
     finish=("steel", "|".join(FINISHES) + ": the surface the clicks come from, its peak sits at 1-5 kHz (polymer and wood lowest)"),
     spread=(0.0, "0..1: each cluster of a reload starts at a random time in its own window (up to 35, 14, 35 ms later) instead of the fixed ones"),
     seed=(0, "variation"))
def gun_mech(sr=DEFAULT_SR, action="slide_rack", weapon="rifle", speed=1.0, floor="concrete", mag="plastic", rounds=6, finish="steel", spread=0.0, seed=0):
    """Same recipe as `reload` (R03 §5.3): every click and the slide through the shot's body, "so the shots and
    the reload sound are automatically aligned"."""
    action = str(action)
    if action not in MECH and action not in RELOADS:
        raise ValueError(f"unknown action {action!r}: {'|'.join((*MECH, *RELOADS))}")
    if str(floor) not in FLOORS:
        raise ValueError(f"unknown floor {floor!r}: {'|'.join(FLOORS)}")
    if str(finish) not in FINISHES:
        raise ValueError(f"unknown finish {finish!r}: {'|'.join(FINISHES)}")
    rng = np.random.default_rng(int(seed))
    k = 1.0 / float(np.clip(speed, 0.25, 4.0))
    seq = RELOADS.get(action, [(action, 0.0)])
    parts = [(_mech(a, k, rng, str(finish), float(np.clip(spread, 0.0, 1.0))), t * k) for a, t in seq]
    x = np.zeros(_n(max(t for _, t in parts)) + max(len(p) for p, _ in parts))
    for p, t in parts:
        _put(x, p, t)
    x = amb.hip(x * 2.0, 100.0)
    y = _bpar8(x, [f * MECH_BODY.get(str(weapon), 1.0) for f in RIFLE_BODY], gain=1.0) + 0.5 * x
    extras = _mech_extras(action, str(weapon), k, int(seed), str(floor), str(mag), int(np.clip(rounds, 1, 2000)))
    if extras:
        ref = float(np.max(np.abs(y))) or 1.0                # nothing through the body (links, belt): the extras alone
        y = np.append(y, np.zeros(max(_n(t) + len(s) for s, t, _ in extras)))
        for s, t, g in extras:
            _put(y, _nm(s) * ref, t, g)
    return _fin(y, sr, 0.7)


# ================================================================== spent brass
BRASS_C = 1090.0     # sqrt(E / (12 rho (1 - nu^2))), C26000: E 110.3 GPa, rho 8530 (Concast), nu 0.31 (MakeItFrom)
PLASTIC_C = 505.0    # the same for polyethylene-ish plastic: E 3.2 GPa, rho 1190, nu 0.35 (contact.MATERIALS)
# length, base diameter, wall mm. Lengths and diameters: CIP/Wikipedia; walls estimated. Gives 4.6 kHz for 7.62 against
# Farnell's 4.5-5.5 kHz for rifle brass. Ring: Q 800 at 5 kHz (Farnell) -> d = 20 1/s.
CASES = presets.schema("case_dim", {"L": F(float, lo=1.0, hi=500.0, unit="mm"), "dia": F(float, lo=1.0, hi=200.0, unit="mm"),
                                    "wall": F(float, lo=0.05, hi=5.0, unit="mm")}, shape=lambda d: (d["L"], d["dia"], d["wall"]))
# surface material or None (soft), surface size m, restitution, low-pass Hz
FLOORS = {"concrete": ("concrete", 1.5, 0.5, 0), "wood": ("wood", 0.8, 0.55, 0), "metal": ("steel", 0.6, 0.6, 0),
          "tile": ("ceramic", 0.3, 0.55, 0), "dirt": (None, 0.0, 0.15, 1500), "grass": (None, 0.0, 0.1, 900)}


BRASS_NU = 0.31


def shell_modes(radius_mm: float, wall_mm: float, c: float = BRASS_C, length_mm: float = 0.0, nu: float = BRASS_NU) -> np.ndarray:
    """Inextensional modes n = 2, 3, 4 of a thin cylinder: f_n = (1/2 pi) (h/R^2) c n (n^2 - 1) / sqrt(n^2 + 1), the generators staying
    parallel. With ``length_mm`` (free ends) the tilting set is appended, a little higher:
    f_n sqrt((1 + 24 (1 - nu) R^2 / (n^2 l^2)) / (1 + 12 R^2 / (n^2 (n^2 + 1) l^2)))."""
    ns = np.array([2.0, 3.0, 4.0])
    f = (wall_mm * 1e-3) / (radius_mm * 1e-3) ** 2 * c / (2.0 * math.pi) * ns * (ns * ns - 1.0) / np.sqrt(ns * ns + 1.0)
    if length_mm <= 0:
        return f
    q = (radius_mm / length_mm) ** 2 / (ns * ns)
    return np.concatenate([f, f * np.sqrt((1.0 + 24.0 * (1.0 - nu) * q) / (1.0 + 12.0 * q / (ns * ns + 1.0)))])


_CASE_DIM_KEY = {"9x19": "9mm", "762x51": "762"}


def _case_name(caliber) -> str:
    c = str(caliber)
    if c in CASES:
        return c
    c = _CASE_OF[c] if c in _CASE_OF else c
    return _CASE_DIM_KEY.get(c, c)


def casing_physical(sr, caliber, floor, height, count, spread, roll, seed, size=1.0) -> np.ndarray:
    """Brass modes, struck at each bounce with a new mix of modes (it lands lip, base or flank)
    on a surface that rings too, or on soft ground where it just stops. ``size`` scales the case: pitch follows 1 / size.
    ``caliber`` is a case name or any cartridge name (resolved through ``case_of``)."""
    caliber, floor = _case_name(caliber), str(floor)
    if caliber not in CASES or floor not in FLOORS:
        raise ValueError(f"caliber must be {'|'.join(CASES)}; floor {'|'.join(FLOORS)}")
    rng = np.random.default_rng(int(seed))
    size = float(np.clip(size, 0.2, 5.0))
    length, dia, wall = (x * size for x in CASES[caliber])
    fm = shell_modes(dia / 2.0, wall, length_mm=length)
    fm = fm[fm < 0.45 * P]
    mat, size, e, lop = FLOORS[floor]
    v0 = math.sqrt(2.0 * 9.81 * float(np.clip(height, 0.05, 20.0)))
    body = K._make_body("rect_plate", mat, size, 0.37, 0.41, modes=24, fmax=8000.0) if mat else None
    pulse = np.zeros(_n(0.25))
    pulse[:3] = (1.0, 0.66, 0.33)                            # Hertz contact 40-200 us (spec table C)
    thump = K.modal_bank(pulse, body, P, 0.5)[0] if body is not None else None
    thump = _nm(thump) if thump is not None else None
    tube = shell_modes(10.1, 0.6, PLASTIC_C) if caliber == "12ga" else None
    y = np.zeros(_n(3.0 + float(spread)))
    for c in range(int(np.clip(count, 1, 2000))):
        t = 0.0 if c == 0 else float(rng.uniform(0.0, float(spread)))
        v, g = v0, 1.0
        while g > 0.03 and v > 0.05:
            # lands anywhere along its length: 1/m_eff = 1/M + d^2/I, I = M L^2 / 12 (c17 l.23436). rho V^2 / sigma_s = 3.8e-4
            # at 3 m/s (c20 l.3655): no plastic loss to model, restitution stays FLOORS'
            meff = 1.0 / (1.0 + 12.0 * float(rng.uniform(-0.5, 0.5)) ** 2)
            f = np.zeros(_n(0.4 if lop == 0 else 0.03))
            f[:3] = pulse[:3] * g * meff
            ring = modal.render_modes(f, modal.Modes(fm, np.full(len(fm), 20.0 if lop == 0 else 400.0), rng.uniform(0.2, 1.0, len(fm)), "vdd"))
            ring = ring / (float(np.max(np.abs(modal.render_modes(pulse[:64], modal.Modes(fm, np.full(len(fm), 20.0), np.ones(len(fm)), "vdd"))))) + 1e-12)
            if tube is not None:
                hull = modal.render_modes(f, modal.Modes(tube, np.full(3, 120.0), rng.uniform(0.5, 1.0, 3), "vdd"))
                ring = 0.7 * ring + _nm(hull) * g
            _put(y, ring, t)
            if thump is not None:
                _put(y, thump, t, 0.25 * g)
            else:
                m = _n(0.012)
                _put(y, amb.lop(rng.uniform(-1, 1, m) * (1.0 - np.arange(m) / m) ** 2, float(lop)), t, 0.8 * g)
            v *= e
            gap = 2.0 * v / 9.81
            if gap < 0.012:
                break
            t += gap
            g *= e ** 1.0
        if roll and lop == 0:
            u = float(rng.uniform(0.3, 1.0))
            f_roll = u / (2.0 * math.pi * dia * 0.5e-3)
            for k in range(int(rng.uniform(0.3, 1.2) * f_roll)):
                f = np.zeros(_n(0.05))
                f[0] = 1.0
                gg = 0.05 * g * rng.uniform(0.4, 1.0) * (1.0 - k / (1.2 * f_roll))
                _put(y, modal.render_modes(f, modal.Modes(fm, np.full(len(fm), 60.0), rng.uniform(0.2, 1.0, len(fm)), "vdd")), t + k / f_roll, gg)
    return _fin(y, sr, 0.6)


_CASE_EXTRA = presets.schema("case_extra", {
    "L": F(float, lo=1.0, hi=500.0, unit="mm"), "db": F(float, lo=1.0, hi=200.0, unit="mm"), "dn": F(float, lo=1.0, hi=200.0, unit="mm"),
    "h": F(float, lo=0.05, hi=5.0, unit="mm"), "cap": F(float, lo=0.1, hi=5000.0, unit="cm3"), "mass": F(float, lo=0.1, hi=5000.0, unit="g"),
    "mat": F(str, choices=("brass", "pe")), "neck": F(float, lo=0.0, hi=300.0, unit="mm")})
_CASE_KEY = {"9mm": "9x19", "762": "762x51"}


def _case_geom(case: str) -> dict:
    c = _CASE_KEY.get(str(case), str(case))
    return BR.CASE_GEOM[c] if c in BR.CASE_GEOM else _CASE_EXTRA[c]


@sfx("casing_ricochet", "The spent case of a film or a game, from its physics: modes of the shell wall each with its own decay, the air column, a fall of Hertz bounces that speed up, tumbling in flight, a roll or a wobble to rest.",
     caliber=("556", "|".join(CASES)), floor=("concrete", "|".join(BR.FLOORS)), height=(1.2, "m it falls from (0.1-5)"), count=(1, "cases (1-64)"),
     spread=(0.25, "s between the cases, on average"), ring=(1.0, "how long the brass rings: 1 = the wall's own losses (internal + thermoelastic), 2 = twice as long"),
     spin=(40.0, "tumbling in rev/s as it leaves the floor; 0 = no turning"), lively=(0.0, "0..1 of the gap between the floor's restitution and 1 given back (film choice)"),
     spin_depth=(0.5, "0..1 how much the turning modulates the ring"), rattle_law=("v", "v (physical) | sqrt_v (the louder final rattle of film)"),
     wobble=(True, "true = one case in five ends as a spinning disc instead of rolling"), air=(0.25, "gain of the air column"),
     wall_scale=(1.0, "wall thickness scale: the ring's pitch follows (5.56: 5-7 kHz)"), seed=(0, "variation"))
def casing_ricochet(sr=DEFAULT_SR, caliber="556", floor="concrete", height=1.2, count=1, spread=0.25, ring=1.0, spin=40.0, lively=0.0,
                    spin_depth=0.5, rattle_law="v", wobble=True, air=0.25, wall_scale=1.0, seed=0):
    caliber, floor = _case_name(caliber), str(floor)
    if caliber not in CASES or floor not in BR.FLOORS:
        raise ValueError(f"caliber must be {'|'.join(CASES)}; floor {'|'.join(BR.FLOORS)}")
    if str(rattle_law) not in ("v", "sqrt_v"):
        raise ValueError("rattle_law must be v|sqrt_v")
    rng = np.random.default_rng(int(seed))
    geom = _case_geom(caliber)
    count = int(np.clip(count, 1, 64))
    runs = [(0.0 if c == 0 else float(rng.uniform(0.0, float(spread) * count)),
             BR.render_case(geom, floor, float(np.clip(height, 0.1, 5.0)), rng, ring=float(np.clip(ring, 0.2, 6.0)), spin=float(spin),
                            spin_depth=_c01(spin_depth), air=float(air), wobble=bool(wobble), rattle_law=str(rattle_law), rate=P,
                            lively=_c01(lively), wall_scale=float(np.clip(wall_scale, 0.3, 3.0)))) for c in range(count)]
    y = np.zeros(max(int(t0 * P) + len(u) for t0, u in runs))
    for t0, u in runs:
        _put(y, u, t0)
    return _fin(y, sr, 0.6, fade=0.0005, tail=0.1)


# ================================================================== bullets
# body, plate size m, damping, modes, thickness m (None = stops the bullet), flow stress Pa, restitution of the debris
_TARGET = {"metal": ("steel", 0.5, 0.0, 120, None, 0.0, 0.0), "metal_thin": ("steel", 0.25, 2.0, 80, 0.002, 2.5e8, 0.0),
           "concrete": ("concrete", 0.4, 0.0, 40, None, 0.0, 0.1), "wood": ("wood", 0.3, 30.0, 60, 0.025, 4e7, 0.0),
           "glass": ("glass", 0.4, 0.0, 80, 0.006, 5e7, 0.0)}
_BULLET = {k: v[:4] for k, v in _TARGET.items()}
FLESH_STOP_M = 0.15


def _hit(material, sz, damping, modes, v, cal_mm, rng, impulse=None, t_c=None) -> np.ndarray:
    """Force into the struck body. Without ``impulse``: the ricochet's hit (length 2 calibres / v, at least one sample, peak 1).
    With ``impulse`` (N s) and ``t_c`` (s): a half-sine of that impulse, the modes ring in proportion to it, nothing normalised."""
    body = K._make_body("rect_plate", material, sz, 0.37, 0.41, modes=modes, damping=damping, fmax=16000.0)
    f = np.zeros(_n(max(0.2, min(1.5, body.t60()))))
    if impulse is None:
        m = max(1, _n(2.0 * cal_mm * 1e-3 / max(v, 100.0)))
        f[:m] = np.linspace(1.0, 0.0, m, endpoint=False)
        return _nm(K.modal_bank(f, body, P, 0.7)[0])
    f += B.half_sine_force(impulse, t_c, len(f), P)
    return K.modal_bank(f, body, P, 0.7)[0]


def _transfer(material: str, cal: str) -> tuple[float, float, float]:
    """``(impulse N s, contact time s, energy dissipated J)`` of a bullet on a target. A target that stops the bullet takes
    m v (1 + e); a thin one is perforated (hole work pi/2 sigma d^2 h, residual speed v sqrt(1 - W/E)) and takes m (v - v_r)."""
    d, l, v, _, _ = CALIBERS[cal]
    e = CHARGE[cal][0]
    m = 2.0 * e / (v * v)
    if material in ("flesh", "flesh_armor"):
        return m * v, 2.0 * FLESH_STOP_M / v, e
    h, sig, er = _TARGET[material][4:] if material in _TARGET else (None, 0.0, 0.0)
    if h is not None:
        wp = 0.5 * math.pi * sig * (d * 1e-3) ** 2 * h
        if wp < e:
            return m * v * (1.0 - math.sqrt(1.0 - wp / e)), (l * 1e-3 + h) / v, wp
    return m * v * (1.0 + er), l * 1e-3 / v, e


def _modes(mat, sz, damp, nm, cal, material, law, rng) -> np.ndarray:
    """The target's modes for this calibre, in units where the 5.56 on the same target peaks at 1 (law impulse: in proportion to the
    impulse; law energy: in proportion to sqrt of the muzzle energy)."""
    d, _, v, _, _ = CALIBERS[cal]
    j, tc, _ = _transfer(material, cal)
    j5, tc5, _ = _transfer(material, "556")
    raw = _hit(mat, sz, damp, nm, v, d, rng, j, tc)
    ref = float(np.max(np.abs(raw if cal == "556" else _hit(mat, sz, damp, nm, CALIBERS["556"][2], CALIBERS["556"][0], rng, j5, tc5)))) + 1e-30
    out = raw / ref
    return out if law == "impulse" else out * math.sqrt(CHARGE[cal][0] / CHARGE["556"][0]) * j5 / j


def _debris(kind, rng, gain) -> np.ndarray:
    """Falling fragments as small hail on the surface: ``hail_impulses`` with a thinning rate (estimated)."""
    size, rate, dur, body, cutoff = {"chips": (3e-3, 40.0, 0.5, "stone", 8000.0), "dirt": (1.5e-3, 70.0, 0.6, "concrete", 1500.0),
                                     "splinters": (2e-3, 25.0, 0.3, "wood", 6000.0)}[kind]
    f, _, _ = K.hail_impulses(dur, P, rate, size, 1.0, 1.0, int(rng.integers(1 << 30)))
    b = K._make_body("rect_plate", body, 0.3, 0.3, 0.4, modes=40, fmax=12000.0)
    y = K.modal_bank(f, b, P, 0.6)[0]
    return _nm(amb.lop(y, cutoff)) * gain


MATERIALS = "metal|metal_thin|concrete|wood|flesh|flesh_armor|water|dirt|glass"


def _impact(material: str, cal: str, seed: int, debris: float, law: str, distance: float) -> np.ndarray:
    from . import foley as Fo
    from . import matter as M
    rng = np.random.default_rng(int(seed))
    db = _c01(debris)
    _, _, ed = _transfer(material, cal)
    _, _, e5 = _transfer(material, "556")
    se = math.sqrt(ed / e5)                                  # debris, sparks, splash: in proportion to the energy dissipated
    y = np.zeros(_n(2.5))
    if material in _TARGET:
        mat, sz, damp, nm = _TARGET[material][:4]
        _put(y, _modes(mat, sz, damp, nm, cal, material, law, rng), 0.0)
        if material == "metal_thin":
            _put(y, _nm(K.crumple(P, kind="can", energy=0.7, dur=0.05, seed=int(seed))), 0.0, 0.5 * se)
        elif material == "metal":
            _put(y, _nm(M.spark(P, size=0.2, seed=int(seed))), 0.002, 0.15 * db * se)
        elif material == "concrete":
            m = _n(0.015)
            _put(y, amb.bp(rng.uniform(-1, 1, m), 1500.0, 0.6) * (1.0 - np.arange(m) / m), 0.0, 1.5 * se)
            _put(y, _debris("chips", rng, 0.3 * db * se), 0.03)
        elif material == "wood":
            _put(y, _nm(K.crumple(P, kind="can", energy=0.3, dur=0.04, bright=0.8, seed=int(seed))), 0.0, 0.25 * se)
            _put(y, _debris("splinters", rng, 0.2 * db * se), 0.02)
        elif material == "glass":
            _put(y, _nm(K.smash(P, material="glass", size=0.4, energy=0.6, seed=int(seed))), 0.0, (0.8 if db else 0.4) * se)
    elif material in ("flesh", "flesh_armor"):
        j, _, _ = _transfer("flesh", cal)
        j5, _, _ = _transfer("flesh", "556")
        s = j / j5 if law == "impulse" else math.sqrt(CHARGE[cal][0] / CHARGE["556"][0])
        _put(y, _nm(Fo.thud(P, body="sack", contact_ms=8.0, size=0.6, force=0.9, seed=int(seed))), 0.0, s)
        m = _n(0.01)
        _put(y, amb.lop(rng.uniform(-1, 1, m) * (1.0 - np.arange(m) / m), 1200.0), 0.0, 1.2 * se)
        if material == "flesh_armor":
            _put(y, _modes("steel", 0.2, 30.0, 60, cal, "metal", law, rng), 0.0, 0.5)
    elif material == "water":
        _put(y, _nm(M.splash(P, size=0.1, dur=1.0, droplets=int(6 * db), seed=int(seed))), 0.0, se)
    elif material == "dirt":
        m = _n(0.025)
        _put(y, amb.lop(rng.uniform(-1, 1, m) * (1.0 - np.arange(m) / m) ** 2, 1500.0), 0.0, 1.5 * se)
        _put(y, _debris("dirt", rng, 0.25 * db * se), 0.05)
    else:
        raise ValueError(f"unknown material {material!r}")
    return space.distance(y, float(np.clip(distance, 1.0, 2000.0)), ref=1.0)


_IMPACT_G = []


def _impact_gain() -> float:
    """The constant that brings the .50 on thick steel at 1 m to 0.85, computed once."""
    if not _IMPACT_G:
        _IMPACT_G.append(0.85 / float(np.max(np.abs(_impact("metal", "50bmg", 0, 1.0, "impulse", 1.0)))))
    return _IMPACT_G[0]


@sfx("bullet_impact", "A bullet striking metal, thin metal, concrete, wood, flesh, armour, water, dirt or glass: the target's modes ring in proportion to the impulse the bullet gives it, its fragments in proportion to the energy it dissipates.",
     material=("metal", MATERIALS), caliber=("556", "|".join(CALIBERS)),
     distance=(5.0, "m"), debris=(1.0, "0..1 fragments, chips, drops"),
     level=("relative", "relative = the physical level, the .50 on steel at 1 m at 0.85 | peak = every file at 0.85"),
     gain_db=(0.0, "dB added in relative"), law=("impulse", "impulse (modes follow m v: .50 is 20 dB over the 5.56 on steel) | energy (follow sqrt(E): 10 dB)"),
     env=("open_field", ENV_HELP), seed=(0, "variation"))
def bullet_impact(sr=DEFAULT_SR, material="metal", caliber="556", distance=5.0, debris=1.0, level="relative", gain_db=0.0, law="impulse",
                  env="open_field", seed=0):
    material, law = str(material), str(law)
    if material not in MATERIALS.split("|"):
        raise ValueError(f"unknown material {material!r}")
    if law not in ("impulse", "energy"):
        raise ValueError("law must be impulse|energy")
    if str(level) not in ("relative", "peak"):
        raise ValueError(f"unknown level {level!r}: relative|peak")
    cal = str(caliber) if str(caliber) in CALIBERS else "556"
    y = _impact(material, cal, int(seed), debris, law, distance)
    wet = _env_params(env) is not None
    y = _env(y, env, float(np.clip(distance, 1.0, 2000.0)), int(seed))
    if level == "relative":
        y = _fin(y * _impact_gain() * 10.0 ** (float(gain_db) / 20.0), sr, None, fade=0.0003, tail=0.05)
        return y * min(1.0, 0.89 / (float(np.max(np.abs(y))) + 1e-12)) if wet else y
    return _fin(y, sr, 0.85, fade=0.0003, tail=0.05)


@sfx("ricochet", "Bullet glancing off a surface: a short hit, then a whining tone falling as it flies away.",
     surface=("metal", "metal|stone|water"), variant=("whine", "whine | tumble (fluttering) | zip (short and high) | none (only the hit)"),
     f0=(1343.0, "Hz of the whine (tuned by ear, no measured source)"), speed=(199.0, "m/s after the bounce (tuned by ear)"),
     closest=(5.0, "m"), dur=(0.8, "s"), seed=(0, "variation"))
def ricochet(sr=DEFAULT_SR, surface="metal", variant="whine", f0=1343.0, speed=199.0, closest=5.0, dur=0.8, seed=0):
    from . import matter as M
    from .reverbs import doppler
    rng = np.random.default_rng(int(seed))
    surface, variant = str(surface), str(variant)
    hit = _hit("steel", 0.2, 10.0, 60, 900.0, 7.0, rng) if surface == "metal" else (
        _nm(M._splash(P, 0.08, 0.4, 2, int(seed), soft=False)) if surface == "water" else
        _nm(amb.bp(rng.uniform(-1, 1, _n(0.015)), 1500.0, 0.6)))
    y = np.zeros(_n(0.02) + _n(dur) + _n(0.6))
    _put(y, hit, 0.0)
    if variant != "none":
        f, T = float(f0) * (1.5 if variant == "zip" else 1.0), 0.12 if variant == "zip" else float(np.clip(dur, 0.1, 4.0))
        n = _n(T)
        src = sum(g * amb.bp(rng.uniform(-1, 1, n), f * r, 8.0) for r, g in ((1.0, 1.0), (1.5, 0.4), (2.3, 0.2)))
        if variant == "tumble":
            src = src * (0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(80.0, 300.0) * np.arange(n) / P))
        src = np.asarray(doppler(src, P, speed=float(speed), closest=float(closest), pass_at=0.05, pan=False))
        src = src.mean(axis=1) if src.ndim == 2 else src
        _put(y, _nm(src), 0.015, 0.5)
    return _fin(y, sr, 0.85, fade=0.0003, tail=0.05, soft=False)


@sfx("bullet_pass", "A bullet passing near you: the supersonic crack alone (N-wave, ground reflection, wake); a subsonic round only whizzes; near Mach 1 you get both.",
     caliber=("556", "|".join(CALIBERS)), miss=(3.0, "m from your head"), speed=(0.0, "m/s (0 = the calibre's own)"),
     range=(0.0, "m flown already: it slows down, so far away the crack gives way to a whizz"), spin_tone=(True, "true = the faint line at speed / rifling twist"),
     level=("peak", "peak = the file at 0.9 | relative = pascals over the .50's blast at 3 m (a 5.56 crack at 3 m is 35 dB under)"),
     recorder=("field", RECORDER_HELP), clip_db=(0.0, "dB SPL where the recorder clips (0 = its own)"),
     render=("heard", RENDER_HELP), env=("open_field", "render=heard: " + ENV_HELP), place=("none", PLACE_HELP), seed=(0, "variation"))
def bullet_pass(sr=DEFAULT_SR, caliber="556", miss=3.0, speed=0.0, range=0.0, spin_tone=True, level="peak", recorder="field", clip_db=0.0,
                render="heard", env="open_field", place="none", seed=0):
    from .reverbs import doppler
    heard = _check_render(str(render))
    rng = np.random.default_rng(int(seed))
    d, l, v, _, _ = CALIBERS.get(str(caliber), CALIBERS["556"])
    cal = str(caliber) if str(caliber) in CALIBERS else "556"
    v = (float(speed) if speed > 0 else v) * math.exp(-max(float(range), 0.0) / drag_length(cal))
    b = float(np.clip(miss, 0.1, 500.0))
    y = np.zeros(_n(0.6))
    if v > C_AIR:
        _put(y, _crack(v, d, l, b, rng, near=True), 0.0)
        # the spin line v / twist: 915 m/s / 0.18 m = 5083 Hz (M249) -30 dB, 15 ms
        m = _n(0.015) if spin_tone else 0
        _put(y, np.sin(2 * np.pi * min(v / 0.18, 0.45 * P) * np.arange(m) / P) * (1.0 - np.arange(m) / m), 0.0, 0.03 * B.n_wave_pa(v, d, l, b)[0])
    if v < 1.1 * C_AIR:
        n = _n(0.4)
        src = amb.bp(rng.uniform(-1, 1, n), min(0.2 * v / (d * 1e-3), 0.45 * P), 4.0)
        src = np.asarray(doppler(src, P, speed=min(v, 0.9 * C_AIR), closest=b, pass_at=0.3, pan=False))
        src = src.mean(axis=1) if src.ndim == 2 else src
        _put(y, _nm(src), 0.0, (0.5 if v > C_AIR else 1.0) * max(1.0, float(np.max(np.abs(y)))))
    wet = heard and _env_params(env) is not None
    if heard:
        y = _env(y, env, max(b, ENV_REF_M), int(seed))
    if heard and v > C_AIR:
        y = B.recorder(_place(y, place), str(recorder), float(clip_db) if clip_db > 0 else None)
    return _finish(y, sr, b, str(level), 0.0002, 0.02, wet)


# Specs, no code (armes_feu.md:204, 224). Spread 0.3-1 s and reloads estimated. Indoors: echo 0, the room is the fx (Farnell: not in the sound).
RECIPES = {
    "musket_volley": {"layers": [{"type": "scatter", "dur": 1.0, "count": 12, "seed": 4, "reseed": True, "gain_db": 3.0, "pitch": 0.3,
                                  "layers": [{"type": "sfx", "kind": "firearm", "params": {"weapon": "flintlock", "distance": 5.0, "echo": 0.0}}]}],
                      "fx": [{"type": "distance", "metres": 150.0}], "normalize": -1.0},
    "gunshot_indoor": {"layers": [{"type": "sfx", "kind": "firearm", "params": {"weapon": "pistol_9mm", "distance": 5.0, "echo": 0.0, "env": "none", "place": "none"}}],
                       "fx": [{"type": "room", "preset": "bathroom", "mix": 0.6}], "normalize": -1.0},
}
