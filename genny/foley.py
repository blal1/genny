# Parts of this module are ports of procedural/{weapons,impacts,interface,world,techniques}.py,
# (c) 2025 Chris Nash, Klang Open License 1.0 (see KLANG_LICENSE.txt): the gun layer primitives, the
# Farnell click factory / clock body / switch panel, the material strike recipes, clash and archery.
"""Foley: objects, weapons, machines, signals, sci-fi; plus the finished kernels of
``genny.physical`` that no catalog entry reached (measured modal objects, 25 PhISEM presets, FEM
bells, STK ModalBar / BandedWG, hurdy-gurdy, chain / keys / creak).

The physical kernels run at 48 kHz (``genny.physical.core.SR``); everything is resampled to the
requested rate with ``scipy.signal.resample_poly`` in ``_fin``. Pure-tone signals (DTMF, phone
tones, Shepard, sci-fi) are synthesised directly at the output rate. Sources: ``research/SOURCES.md``."""
from __future__ import annotations

import functools
import math
from fractions import Fraction

import numba
import numpy as np
from scipy.signal import lfilter, resample_poly

from . import filters as F
from . import osc as O
from .core import DEFAULT_SR
from .drums import drum
from .instruments import _fm_cap, instrument
from .notes import to_hz
from .physical import ambience as amb
from .physical import core as pc
from .physical import modal
from .physical import modal_data as md
from .physical import particles
from .physical import space
from .physical import waveguides as wg
from .sfx import sfx

P = pc.SR                 # 48000: rate of the physical kernels
C_AIR = 343.0             # m/s


# ============================================================================== helpers
def _n(d: float) -> int:
    return max(1, int(round(d * P)))


def _nm(x: np.ndarray, peak: float = 1.0) -> np.ndarray:
    return x * (peak / (float(np.max(np.abs(x))) + 1e-12))


def _put(buf: np.ndarray, x: np.ndarray, t: float, g: float = 1.0, rate: int = P) -> None:
    i = int(round(t * rate))
    if i >= len(buf) or i < 0:
        return
    m = min(len(x), len(buf) - i)
    buf[i:i + m] += g * x[:m]


def _fin(y, sr: int, level: float | None = 0.8, *, src: int = P, fade: float = 0.003, tail: float = 0.0, soft: bool = True) -> np.ndarray:
    """Mono, resample ``src`` -> ``sr``, remove DC (one-pole 12 Hz high-pass), fade both ends
    (``tail`` s for the end if longer), and set the peak to ``level`` (None: keep the level, only
    guard against > 1.2). ``soft=False``: linear fades, for the take of `arms.ricochet` validated by ear."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if src != sr:
        fr = Fraction(int(sr), int(src))
        y = resample_poly(y, fr.numerator, fr.denominator)
    r = math.exp(-2 * math.pi * 12.0 / sr)
    y = lfilter([1.0, -1.0], [1.0, -r], y)
    k = max(2, min(len(y) // 2, int(fade * sr)))
    y[:k] *= (np.sin(0.5 * np.pi * np.arange(k) / k) ** 2 if soft else np.linspace(0.0, 1.0, k))      # raised cosines: a linear fade's corner on the DC blocker's tail is a detect_clicks hit
    k = max(2, min(len(y) // 2, int(max(fade, tail) * sr)))
    y[-k:] *= (np.cos(0.5 * np.pi * (np.arange(k) + 1) / k) ** 2 if soft else np.linspace(1.0, 0.0, k))
    pk = float(np.max(np.abs(y))) if len(y) else 0.0
    if level is not None and pk > 0:
        y = y * (level / pk)
    elif pk > 1.2:
        y = y * (1.2 / pk)
    return y


def _c01(v, lo=0.0, hi=1.0) -> float:
    return float(np.clip(float(v), lo, hi))


def _sqdec(n: int, d_ms: float, rise_ms: float = 1.0) -> np.ndarray:
    """``sqdec``: to 1 in ``rise_ms``, to 0 over ``d_ms``, squared."""
    t = np.arange(n) / P * 1e3
    v = np.where(t < rise_ms, t / rise_ms, np.clip(1.0 - (t - rise_ms) / d_ms, 0.0, 1.0))
    return v * v


def _mclick(noise: np.ndarray, parts, at: float = 0.0, gain: float = 1.0) -> np.ndarray:
    """``mclick`` / ``clickfactory``: the shared noise is band-passed, then each band is
    gated by its own squared decay. ``parts`` = [(f Hz, Q, decay ms), ...]."""
    n = len(noise)
    out = np.zeros(n)
    for f, q, d in parts:
        band = amb.bp(noise, min(f, 0.45 * P), q)
        for t0 in np.atleast_1d(at):                 # `at` may be a list of start times (s)
            i0 = int(t0 * P)
            m = min(n - i0, int((d + 1.0) * 1e-3 * P) + 1)
            if m > 0:
                out[i0:i0 + m] += band[i0:i0 + m] * _sqdec(m, d)
    return out * gain


# Click clusters (f Hz, Q, decay ms)
CATCH = ((4564, 3, 50), (4237, 14, 10), (4876, 12, 10))          # reload cluster 1: small catch
ENDSTOP = ((345, 22, 70), (256, 3, 45), (11023, 46, 11))         # reload cluster 2: end-stop clunk
RATCHET = ((5432, 40, 40), (6290, 20, 22), (2470, 20, 8))        # reload cluster 3: ratchet
TICK = ((6543, 30, 30), (3245, 30, 20), (1356, 30, 10))          # clocktick
TOCK = ((7543, 30, 30), (3988, 30, 20), (2765, 30, 10))          # clocktick2
HAND = ((300, 30, 120), (3245, 30, 20), (5356, 30, 10))          # clockhand "clunk"


@numba.njit(cache=True)
def _bpc(f, q, fs):
    w = 2.0 * np.pi * f / fs
    al = np.sin(w) / (2.0 * q)
    a0 = 1.0 + al
    return al / a0, -2.0 * np.cos(w) / a0, (1.0 - al) / a0


@numba.njit(cache=True)
def _body_k(x, da, db, fa, fb_, q, fb, fs):
    """``bodyresonance~``: two delay lines, each read through a band-pass at
    1/delay and fed back x ``fb``; branch A's feedback also feeds branch B's filter."""
    n = x.shape[0]
    size = max(da, db) + 2
    ba = np.zeros(size)
    bb = np.zeros(size)
    a0, a1, a2 = _bpc(fa, q, fs)
    b0, b1, b2 = _bpc(fb_, q, fs)
    xa1 = xa2 = ya1 = ya2 = xb1 = xb2 = yb1 = yb2 = 0.0
    out = np.zeros(n)
    w = 0
    for i in range(n):
        ra = ba[(w - da) % size]
        rb = bb[(w - db) % size]
        ya = a0 * ra - a0 * xa2 - a1 * ya1 - a2 * ya2
        xa2 = xa1
        xa1 = ra
        ya2 = ya1
        ya1 = ya
        inb = rb + fb * ya
        yb = b0 * inb - b0 * xb2 - b1 * yb1 - b2 * yb2
        xb2 = xb1
        xb1 = inb
        yb2 = yb1
        yb1 = yb
        ba[w] = x[i] + fb * ya
        bb[w] = x[i] + fb * yb
        w = (w + 1) % size
        out[i] = ya + yb
    return out


def _body(x: np.ndarray, s: float = 0.09, fb: float = 0.3) -> np.ndarray:
    """Dry + body resonance of scale ``s`` (delay 1/s ms, band-pass s*10 kHz, Q 3; clock = 0.09)."""
    s = float(np.clip(s, 0.012, 1.5))
    fb = float(np.clip(fb, 0.0, 0.6))                      # loop gain < 1: stable
    da, db = max(int(round(P * 1e-3 / s)), 1), max(int(round(P * 1e-3 / (s + 0.02))), 1)
    return x + _body_k(np.ascontiguousarray(x, dtype=np.float64), da, db, s * 1e4, (s + 0.02) * 1e4, 3.0, fb, float(P))


@numba.njit(cache=True)
def _panel_k(x, d, f, q, fb, gin, fs):
    n = x.shape[0]
    size = d + 2
    buf = np.zeros(size)
    c0, c1, c2 = _bpc(f, q, fs)
    x1 = x2 = y1 = y2 = 0.0
    out = np.zeros(n)
    w = 0
    for i in range(n):
        r = buf[(w - d) % size]
        y = c0 * r - c0 * x2 - c1 * y1 - c2 * y2
        x2 = x1
        x1 = r
        y2 = y1
        y1 = y
        buf[w] = gin * x[i] + fb * y
        w = (w + 1) % size
        out[i] = y
    return out


# mount -> (delay ms, band-pass Hz, Q, feedback); input x10. Wood lifts 40-400 Hz; metal: short delay, high band, more feedback.
MOUNTS = {"plastic": (50.0, 700.0, 3.0, 0.1), "wood": (50.0, 300.0, 3.0, 0.1), "metal": (9.0, 1500.0, 8.0, 0.3)}


def _panel(x: np.ndarray, mount: str = "plastic", gin: float = 10.0) -> np.ndarray:
    d, f, q, fb = MOUNTS.get(str(mount), MOUNTS["plastic"])
    return x + _panel_k(np.ascontiguousarray(x, dtype=np.float64), _n(d * 1e-3), f, q, fb, gin, float(P))


def _switchclick(noise: np.ndarray, attack_ms: float, decay_ms: float, f: float, at: float = 0.0) -> np.ndarray:
    """Switch model A: noise -> bp(f, Q12) x (rise, fall)^2."""
    n = len(noise)
    out = np.zeros(n)
    i0 = int(at * P)
    m = min(n - i0, int((attack_ms + decay_ms) * 1e-3 * P) + 1)
    if m > 0:
        out[i0:i0 + m] = amb.bp(noise, f, 12.0)[i0:i0 + m] * _sqdec(m, decay_ms, attack_ms)
    return out


def _shortping(n: int, at: float, gain: float, bright: float = 1.0) -> np.ndarray:
    """Switch model C: (osc 10500 + osc 9453) x (1 -> 0 over 50 ms)^4: the tiny spring."""
    out = np.zeros(n)
    i0 = int(at * P)
    m = min(n - i0, _n(0.05))
    if m > 0:
        t = np.arange(m) / P
        out[i0:i0 + m] = (np.cos(2 * np.pi * 10500 * t) + np.cos(2 * np.pi * 9453 * t)) * (1 - t / 0.05) ** 4 * gain * bright
    return out


def _line(n: int) -> np.ndarray:
    return 1.0 - np.arange(n) / max(n, 1)


def _shock(T: float = 0.010) -> np.ndarray:
    """``pd shockwave``: phi = 5 v^2, 10/T Hz falling to 0: "a nice dull thump"."""
    v = _line(_n(T))
    return amb.hip(np.concatenate([np.cos(2 * np.pi * 5.0 * v * v), np.zeros(_n(0.02))]), 1.0)


def _strike(table: str, f_scale=1.0, contact_ms=1.0, position=0.0, rng=None, **kw) -> np.ndarray:
    return modal.strike(table, float(f_scale), float(np.clip(contact_ms, 0.02, 60.0)), _c01(position), rng, **kw)


def _drive(table: str, force: np.ndarray, f_scale: float = 1.0, d_scale: float = 1.0) -> np.ndarray:
    """Drive a measured body with any force signal (FoleyAutomatic "audio force")."""
    return modal.render_modes(np.ascontiguousarray(force, dtype=np.float64), modal.resolve(table, f_scale, 0.0, d_scale=d_scale))


def _knock(table: str, rng, f_scale=1.0, d_scale=1.0, contact_ms=1.0, vel=1.0, micro=0, felt=False, dur=0.6) -> np.ndarray:
    """One contact on a measured body: (1 - cos) pulse, or the Hertz felt pulse for a hand
    (``felt``); louder = shorter contact; mode gains jittered +-15 % per hit for strike direction.
    Port of procedural/world.knock."""
    m = modal.resolve(table, f_scale, 0.0, d_scale=d_scale)
    m.gains = m.gains * (1.0 + 0.15 * rng.standard_normal(len(m.gains)))
    T = modal.contact_time(contact_ms, vel)
    force = np.zeros(_n(dur))
    p = (pc.felt_pulse(T, 2.5) if felt else modal.bang(T)) * vel
    force[:len(p)] += p[:len(force)]
    if micro:
        modal.micro_collisions(force, m, int(micro), T, vel, rng)
    return modal.render_modes(force, m)


def _phisem(name: str, dur: float, rng, curve: np.ndarray, *, mode="inject", res=1.0, objects=None) -> np.ndarray:
    return particles.shaker(name, dur, rng, energy_curve=curve, n_objects=objects, resonance_scale=res, energy_mode=mode)


@functools.lru_cache(maxsize=None)
def _phisem_ref(name: str) -> float:
    """Peak of one full-energy shake of a preset: the fixed gain that makes `energy` audible."""
    y = _phisem(name, 0.4, np.random.default_rng(1), particles.cook_shakes(0.4, period=10.0))
    return float(np.max(np.abs(y))) or 1.0


def _train(rate: np.ndarray, rng, jitter: float = 0.05) -> np.ndarray:
    """Unit impulses where the integral of ``rate`` (events/s, per sample at P) crosses integers."""
    ph = np.cumsum(np.maximum(rate, 0.0)) / P
    out = np.zeros(len(rate))
    idx = np.nonzero(np.diff(np.floor(ph)) > 0)[0]
    out[idx] = 1.0 + jitter * rng.standard_normal(len(idx))
    return out


def _resons(x: np.ndarray, freqs, radii44, gains=None) -> np.ndarray:
    """STK two-pole resonances (radius given at 44.1 kHz) as a modal bank."""
    f = np.asarray(freqs, float)
    keep = f < 0.45 * P
    d = -np.log(np.asarray(radii44, float)) * 44100.0
    g = np.ones(len(f)) if gains is None else np.asarray(gains, float)
    return modal.render_modes(x, modal.Modes(f[keep], d[keep], g[keep], "zeros", norm=1.0))


# ============================================================================== objects / materials
# name -> (table, f_scale, d_scale). Measured .sy tables;
# crate/box/block/tub: Faust modeInterpRes wooden bodies.
OBJECTS = {
    "stick": ("sy_stick", 1.0, 1.0), "wok": ("sy_wok", 1.0, 1.0), "sword": ("sy_sword1", 1.0, 1.0),
    "sword2": ("sy_sword2", 1.0, 1.0), "bottle": ("sy_calona0", 1.0, 1.0), "vase": ("sy_vase", 1.0, 1.0),
    "metal_box": ("sy_computertower", 1.0, 1.0), "lamp": ("sy_desklamp", 1.0, 1.0),
    "bell": ("sy_bell1", 1.0, 1.0), "church_bell": ("sy_bell4", 1.0, 1.0),
    "crate": ("body_squareBig", 1.0, 1.0), "box": ("body_squareMid", 1.0, 1.0), "block": ("body_squareSmall", 1.0, 1.0),
    "tub": ("body_roundBig", 1.0, 1.0),
    "stone": ("sy_vase", 0.5, 3.0),
}
_MATERIAL_OBJECT = {"steel": "sword", "glass": "bottle", "wood": "stick", "stone": "stone", "aluminum": "metal_box", "rubber": "stick"}
_OBJ_HELP = "|".join(OBJECTS)


@sfx("hit", "Physically resonant impact on a measured object: contact duration is the hardness, the modes are the object.",
     dur=(0.25, "s"), tone=(200, "Hz/body scale (200 = as measured)"), crunch=(0.5, "0..1 hardness: shorter contact, more re-contacts"),
     material=("steel", "steel|glass|wood|stone|aluminum|rubber: picks the object when object=auto"),
     object=("stick", "auto|" + _OBJ_HELP), position=(0.0, "0..1 strike point across the measured points"),
     contact_ms=(0.0, "contact duration ms, 0.05 (steel on steel) .. 50 (dull thud); 0 = from crunch"),
     size=(1.0, "object size: 2 = twice as big, an octave lower"), seed=(0, "variation"))
def hit(sr=DEFAULT_SR, dur=0.25, tone=200, crunch=0.5, material="steel", object="stick", position=0.0, contact_ms=0.0, size=1.0, seed=0):
    """Legacy call (no new params) renders the v0.22 sound exactly: ``sy_stick`` scaled by tone/200.
    Hardness = contact duration, energy = magnitude."""
    c = _c01(crunch)
    name = _MATERIAL_OBJECT.get(str(material), "stick") if object == "auto" else str(object)
    table, fs, ds = OBJECTS.get(name, OBJECTS["stick"])
    m = modal.resolve(table, fs * float(np.clip(float(tone) / 200.0, 0.25, 4.0)) / max(float(size), 0.05), _c01(position),
                      d_scale=ds * (1.35 - 0.65 * c))
    cm = float(contact_ms) if float(contact_ms) > 0 else 1.8 - 1.5 * c
    if material == "rubber" and object == "auto":
        cm = max(cm, 12.0)                                   # a soft striker: long contact, dull
    vel = 0.55 + 0.75 * c
    force = np.zeros(_n(max(0.06, float(dur))))
    p = modal.bang(max(cm / vel, 0.05), vel)
    force[:len(p)] = p[:len(force)]
    modal.micro_collisions(force, m, int(round(1 + 5 * c)), cm, vel, np.random.default_rng(23 + int(seed)))
    y = modal.render_modes(force, m)
    if table != "sy_stick":                                  # other tables have other gain scales
        y = _nm(y, 0.6)
    return _fin(np.tanh(y * (1.2 + 1.6 * c)) * 0.78, sr, None)


@sfx("clang", "Metal struck by metal: measured iron wok, sword, sheet-metal box or lamp with re-contact chatter.",
     object=("wok", "wok|sword|sword2|metal_box|lamp|bell"), size=(1.0, "size (bigger = lower)"), force=(0.8, "0..1"),
     damping=(1.5, "decay-rate scale: 1 free ringing .. 15 hand-muted"), position=(0.0, "0..1 strike point"),
     dur=(2.0, "max s"), seed=(0, "variation"))
def clang(sr=DEFAULT_SR, object="wok", size=1.0, force=0.8, damping=1.5, position=0.0, dur=2.0, seed=0):
    """procedural/impacts.sig_iron: 1 ms contact, 2 micro-collisions, dscale 1.5."""
    rng = np.random.default_rng(int(seed))
    t, fs, ds = OBJECTS.get(str(object), OBJECTS["wok"])
    return _fin(_strike(t, fs / max(float(size), 0.05), 1.0, position, rng, velocity=0.2 + _c01(force), micro=2,
                        d_scale=ds * max(float(damping), 0.1), max_dur=float(np.clip(dur, 0.1, 6.0))), sr)


@sfx("clink", "Glass or small-metal clink: the measured bottle (34 strike points), a vase, or a coin tinkle.",
     object=("bottle", "bottle|vase|coin|bell"), size=(1.25, "size (bigger = lower; 1.25 = a wine glass from the bottle table)"),
     force=(0.7, "0..1"), position=(0.5, "0..1 strike point"), liquid=(0.0, "bottle | vase: fill height 0..1, water lowers the pitch"),
     seed=(0, "variation"))
def clink(sr=DEFAULT_SR, object="bottle", size=1.25, force=0.7, position=0.5, liquid=0.0, seed=0):
    """procedural/impacts.sig_glass: calona0.sy, 0.1 ms contact.
    coin = shell-casing tinkle."""
    rng = np.random.default_rng(int(seed))
    if object == "coin":
        return _fin(amb.tinkle(rng, f_scale=0.8 / max(float(size), 0.05), gain=_c01(force, 0.05)), sr, 0.6)
    t, fs, ds = OBJECTS.get(str(object), OBJECTS["bottle"])
    if liquid > 0 and object in ("bottle", "vase"):
        from .matter import glass_fill_ratio
        fs = fs * float(glass_fill_ratio(liquid))
    return _fin(_strike(t, fs / max(float(size), 0.05), 0.1, position, rng, velocity=0.2 + _c01(force), d_scale=ds, max_dur=2.5), sr, 0.7)


@sfx("thud", "Dull soft impact with no ring: a long contact on a box or a soft mass landing.",
     body=("box", "box (sheet-metal/hollow)|crate (wood)|sack (felt contact on a big body)"), contact_ms=(15.0, "5..50 ms: longer = softer, lower (35 = felt rather than heard)"),
     size=(1.0, "size (bigger = lower)"), force=(0.8, "0..1"), seed=(0, "variation"))
def thud(sr=DEFAULT_SR, body="box", contact_ms=15.0, size=1.0, force=0.8, seed=0):
    """procedural/interface.ui_refuse and world.felt_thud: "30-50 ms: very soft thud; the long pulse
    removes everything above ~100 Hz on its own" plus the clockhand clunk for texture."""
    rng = np.random.default_rng(int(seed))
    n = _n(0.35)
    fs = 1.0 / max(float(size), 0.05)
    if body == "sack":
        y = pc.fit(_knock("body_roundBig", rng, fs, contact_ms=contact_ms * 0.6, felt=True, dur=0.35), n)
    else:
        y = pc.fit(_strike("sy_computertower" if body == "box" else "body_squareBig", fs, contact_ms, 0.0, rng,
                           d_scale=10.0 if body == "box" else 1.0), n)
    clunk = amb.lop(_mclick(amb.hip(rng.uniform(-1, 1, n), 200.0), HAND), 800.0)
    return _fin(_nm(y) + 0.3 * _nm(clunk), sr, 0.25 + 0.6 * _c01(force))


@sfx("knock", "Knocking on a door or panel: a sequence of knuckle contacts on a measured body.",
     count=(3, "knocks"), pace=(4.5, "knocks per second"), material=("wood", "wood|hollow|metal|glass"),
     size=(1.0, "panel size (bigger = lower)"), force=(0.7, "0..1"), seed=(0, "variation"))
def knock(sr=DEFAULT_SR, count=3, pace=4.5, material="wood", size=1.0, force=0.7, seed=0):
    """Knuckle = Hertz felt pulse (soft exciter) on the Faust wooden bodies or the measured
    sheet-metal box / glass."""
    rng = np.random.default_rng(int(seed))
    table, fs, cm, felt = {"wood": ("body_squareMid", 1.0, 2.0, True), "hollow": ("body_roundBig", 0.8, 3.0, True),
                           "metal": ("sy_computertower", 1.5, 1.0, False), "glass": ("sy_calona0", 0.5, 1.5, True)}.get(
        str(material), ("body_squareMid", 1.0, 2.0, True))
    count = int(np.clip(count, 1, 40))
    gap = 1.0 / max(float(pace), 0.5)
    y = np.zeros(_n(count * gap + 0.6))
    for k in range(count):
        v = (0.35 + 0.65 * _c01(force)) * (1.0 if k == 0 else rng.uniform(0.7, 0.9))
        _put(y, _knock(table, rng, fs / max(float(size), 0.05), contact_ms=cm, vel=v, felt=felt), k * gap * (1 + 0.06 * rng.standard_normal()) + 0.005, v)
    return _fin(y, sr, 0.75)


# catalog name -> STK Shakers preset (all 25 of genny.physical.particles)
SHAKE = {
    "maraca": "Maraca", "cabasa": "Cabasa", "sekere": "Sekere", "tambourine": "Tambourine", "sleigh_bells": "SleighBells",
    "bamboo_chimes": "BambooChimes", "sandpaper": "Sandpaper", "soda_can": "CokeCan", "sticks": "Sticks", "crunch": "Crunch",
    "rocks": "BigRocks", "pebbles": "LittleRocks", "mug": "NextMug", "pennies": "PennyMug", "nickels": "NickelMug",
    "dimes": "DimeMug", "coins": "QuarterMug", "francs": "FrancMug", "pesos": "PesoMug", "guiro": "Guiro", "ratchet": "Wrench",
    "water_drops": "WaterDrops", "angklung": "TunedBambooChimes", "gravel": "CookLargeGravel", "fine_gravel": "CookSmallGravel",
}


def _shake(preset: str, dur: float, energy: float, rate: float, seed: int, size: float = 1.0, objects: float = 0.0) -> np.ndarray:
    """PhISEM at 48 kHz with a fixed per-preset gain, so `energy` keeps its meaning (not normalised)."""
    name = SHAKE.get(str(preset), preset if preset in particles.STK_PRESETS or preset in particles.COOK_MEASURED else "Maraca")
    rng = np.random.default_rng(int(seed))
    e = float(np.clip(energy, 0.005, 1.0))
    dur = float(np.clip(dur, 0.05, 30.0))
    if particles.preset(name)["kind"] == particles._RATCHET:
        # the curve is the scrape speed: back-and-forth strokes at `rate` (one sustained stroke at rate 0)
        t = np.arange(_n(dur)) / P
        curve = e * (np.abs(np.sin(np.pi * rate * t)) if rate > 0 else np.ones(len(t)))
    else:
        curve = particles.cook_shakes(dur, period=1.0 / rate if rate > 0 else 1e3, shake_len=min(0.05, 0.5 / max(rate, 1e-3)), amount=e)
    y = _phisem(name, dur, rng, curve, res=1.0 / max(float(size), 0.05), objects=float(objects) if objects else None)
    return y * (0.6 / _phisem_ref(name))


@sfx("shake", "Any of the 25 PhISEM particle presets shaken: coins in a mug, gravel, sleigh bells, bamboo chimes, sandpaper, a soda can, a ratchet...",
     preset=("maraca", "|".join(SHAKE)), energy=(0.7, "0..1 shake energy: louder and longer-lived"),
     rate=(4.0, "shakes per second (0 = one shake that dies away); ratchets: strokes per second"), dur=(1.5, "s"),
     size=(1.0, "container/object size: bigger = lower resonances"), objects=(0, "particle count (0 = the preset's own)"), seed=(0, "variation"))
def shake(sr=DEFAULT_SR, preset="maraca", energy=0.7, rate=4.0, dur=1.5, size=1.0, objects=0, seed=0):
    """PhISEM / STK Shakers: Poisson collisions (p = N/1024 per
    44.1 kHz sample) of level ~ system energy feed the preset's resonances; the energy decays
    exponentially between shakes, so more energy also means more audible collisions."""
    return _fin(_shake(preset, dur, energy, float(rate), seed, size, objects), sr, None)


@sfx("chain", "A chain moved by a body: link contacts fired at a rate that follows the momentum, then settling.",
     dur=(1.5, "s"), pushes=(2, "movements"), weight=(0.5, "0 light jewellery chain .. 1 heavy iron"), density=(40.0, "link contacts per second at full speed"),
     settle=(1, "1 = come to rest with the three-bounce pattern"), seed=(0, "variation"))
def chain(sr=DEFAULT_SR, dur=1.5, pushes=2, weight=0.5, density=40.0, settle=1, seed=0):
    """genny.physical.ambience.chain_shift: rolling-can momentum + shell-casing tinkles + click-factory clunks."""
    y = amb.chain_shift(float(np.clip(dur, 0.3, 20)), np.random.default_rng(int(seed)), pushes=int(np.clip(pushes, 1, 12)),
                        f_scale=1.2 - 0.8 * _c01(weight), contacts_per_s=float(np.clip(density, 2, 200)), settle=bool(settle),
                        clunk_p=0.1 + 0.4 * _c01(weight))
    return _fin(y, sr, 0.7)


@sfx("keys", "A bunch of keys jingling (tambourine jingles: four Q217 bands ring-modulated at 6.3 kHz).",
     hits=(3, "jingles"), decay=(40.0, "ms base decay of the brightest band"), spread=(10.0, "ms between the two contacts of one jingle / 3"),
     seed=(0, "variation"))
def keys(sr=DEFAULT_SR, hits=3, decay=40.0, spread=10.0, seed=0):
    y = amb.keys_jingle(np.random.default_rng(int(seed)), hits=int(np.clip(hits, 1, 30)), sep_ms=float(np.clip(spread, 1, 60)),
                        d_ms=float(np.clip(decay, 5, 200)))
    return _fin(y, sr, 0.6)


def _slips(force: np.ndarray, rng, k: int = 1) -> np.ndarray:
    """k stick-slip pulse trains from one force curve."""
    y = np.zeros(len(force))
    for _ in range(k):
        y += amb.stickslip_pulses(np.clip(force * rng.uniform(0.85, 1.05), 0, 1), rng)
    return y


def _humps(n: int, rng, rate: float, sharp: float = 1.0) -> np.ndarray:
    """Smooth random movement energy in [0, 1]: band-limited noise around ``rate`` Hz, rectified."""
    x = amb.lop(amb.lop(rng.standard_normal(n), rate), rate)
    x = np.abs(x) / (np.max(np.abs(x)) + 1e-12)
    return x ** sharp


CLOTH_TOP = (8000.0, 2000.0)


@sfx("cloth", "Cloth: rustle of a moving garment, a tear, or a flag/cape flapping.",
     kind=("rustle", "rustle|tear|flap"), dur=(1.0, "s"), rate=(3.0, "movements (rustle) or flaps per second"),
     weight=(0.5, "0 silk .. 1 canvas/sacking: lower, rougher"), seed=(0, "variation"))
def cloth(sr=DEFAULT_SR, kind="rustle", dur=1.0, rate=3.0, weight=0.5, seed=0):
    """Port of procedural/world.cloth and body.sacking_tear: the STK Sandpaper brush, kept below 8 kHz
    for a light fabric and 2 kHz for a heavy one, plus a stick-slip band."""
    rng = np.random.default_rng(int(seed))
    dur = float(np.clip(dur, 0.1, 20))
    n = _n(dur)
    w = _c01(weight)
    t = np.arange(n) / P
    if kind == "tear":
        e = np.clip(0.35 + 0.6 * t / dur, 0, 1) * np.minimum(1.0, (dur - t) / 0.03)
        k = 4
    elif kind == "flap":
        e = np.abs(np.sin(np.pi * max(float(rate), 0.2) * t)) ** 12 * (0.7 + 0.3 * _humps(n, rng, 1.0))
        k = 0
    else:
        e = _humps(n, rng, max(float(rate), 0.2), 1.5)
        k = 1
    brush = pc.lowpass(_phisem("Sandpaper", dur, rng, 0.9 * e, mode="level", res=0.55 - 0.3 * w),
                       CLOTH_TOP[0] + w * (CLOTH_TOP[1] - CLOTH_TOP[0]))
    y = _nm(brush)
    if k:
        y += (0.25 if kind == "rustle" else 0.8) * _nm(amb.bp(_slips(0.2 + 0.7 * e, rng, k), 2500.0 - 1200.0 * w, 1.0))
    else:
        y += 0.6 * _nm(amb.lop(np.gradient(e) * rng.uniform(0.6, 1.0), 120.0))
    return _fin(y, sr, 0.6)


CREAKS = {  # material -> (ambience.creak material, size, stick-slip sources)
    "wood": ("wood", 1.0, 1), "door": ("wood", 1.0, 1), "floorboard": ("pine", 1.6, 1),
    "rope": ("pine", 0.5, 3), "leather": ("pine", 0.3, 3), "iron": ("iron", 1.0, 1),
}


def _force_curve(shape: str, n: int, level: float, rng) -> np.ndarray:
    t = np.arange(n) / max(n - 1, 1)
    f = {"ramp": 0.32 + 0.68 * t, "constant": np.ones(n), "sway": 0.6 + 0.4 * np.sin(2 * np.pi * (1.5 * t - 0.25)),
         "pull": np.minimum(1.0, t / 0.35) * np.where(t > 0.8, (1 - t) / 0.2, 1.0)}.get(str(shape), np.ones(n))
    return np.clip(f * level, 0.0, 1.0)


@sfx("creak", "Stick-slip creak of wood, a door, a floorboard, rope, leather or iron under a force curve.",
     material=("wood", "|".join(CREAKS)), force=("ramp", "force curve: ramp|pull|sway|constant"),
     level=(0.8, "0.3..1 peak force: more force = faster slips, higher creak (nothing slips below 0.3)"),
     dur=(1.2, "s"), size=(1.0, "object size (bigger = lower formants)"), seed=(0, "variation"),
     width=(0.0, "m: door leaf width (0 = off); with `height` the leaf rings in its own membrane modes"),
     height=(0.0, "m: door leaf height"), wave_speed=(340.0, "m/s of bending waves in the leaf (100-3300)"))
def creak(sr=DEFAULT_SR, material="wood", force="ramp", level=0.8, dur=1.2, size=1.0, seed=0, width=0.0, height=0.0,
          wave_speed=340.0):
    """Slips every (1-f)60 + 3 ms (+ jitter) while the lagged force f
    exceeds 0.3, through the door formant bank and panel (wood), the same scaled (pine), or the
    measured sheet-metal box (iron). Rope and leather use several stick-slip sources at once."""
    rng = np.random.default_rng(int(seed))
    mat, sz, k = CREAKS.get(str(material), CREAKS["wood"])
    f = _force_curve(force, _n(float(np.clip(dur, 0.15, 20))), _c01(level, 0.31, 1.0), rng)
    y = np.zeros(len(f))
    for _ in range(k):
        y += amb.creak(np.clip(f * (rng.uniform(0.85, 1.05) if k > 1 else 1.0), 0, 1), rng, material=mat,
                       size=sz * max(float(size), 0.05))
    if float(width) > 0.0 and float(height) > 0.0:
        fm = door_modes(width, height, wave_speed)
        y = y + DOOR_MIX * _nm(_resons(y, fm, np.exp(-6.91 / DOOR_T60 / 44100.0) * np.ones(len(fm)),
                                       1.0 / np.arange(1, len(fm) + 1))) * np.max(np.abs(y))
    return _fin(y, sr, 0.7)


DOOR_T60 = 0.25    # s: ring of the leaf modes
DOOR_MIX = 0.6     # level of the leaf modes against the creak itself


def door_modes(width, height, wave_speed=340.0, n=7):
    """Lowest `n` membrane modes of a door leaf held at its edges: f_mn = (c/2) sqrt((m/w)^2 + (n/h)^2)."""
    w, h, c = max(float(width), 0.05), max(float(height), 0.05), float(np.clip(wave_speed, 100.0, 3300.0))
    f = sorted(0.5 * c * math.hypot(m / w, q / h) for m in range(1, 6) for q in range(1, 6))
    return np.array(f[:n])


@sfx("zipper", "Zipper: slider teeth clicking at a rate that follows the pull speed.",
     dur=(0.5, "s"), length=(0.25, "m of zip"), pitch_mm=(2.5, "tooth spacing mm"), direction=("up", "up|down"), seed=(0, "variation"))
def zipper(sr=DEFAULT_SR, dur=0.5, length=0.25, pitch_mm=2.5, direction="up", seed=0):
    """Tooth-passing rate = speed / pitch (a half-sine pull) into the STK Guiro ratchet resonances, x1.3 for small teeth."""
    rng = np.random.default_rng(int(seed))
    n = _n(float(np.clip(dur, 0.08, 5)))
    v = np.sin(np.pi * np.arange(n) / n) * (np.pi / 2) * float(length) / (n / P)      # m/s, integrates to `length`
    rate = np.clip(v / (max(float(pitch_mm), 0.5) * 1e-3), 0, 4000.0)
    p = particles.STK_PRESETS["Guiro"]
    y = _resons(_train(rate, rng, 0.25) * np.sqrt(rate / (rate.max() + 1e-9)), np.array(p["freqs"]) * (1.3 if direction == "up" else 1.1), p["radii"])
    return _fin(y + 0.1 * _nm(amb.bp(rng.standard_normal(n), 3000, 1.0) * rate / (rate.max() + 1e-9)) * np.max(np.abs(y)), sr, 0.6)


@sfx("velcro", "Velcro peeled apart: a dense crackle of hooks letting go.",
     dur=(0.6, "s"), speed=(0.6, "0..1 peel speed"), seed=(0, "variation"))
def velcro(sr=DEFAULT_SR, dur=0.6, speed=0.6, seed=0):
    """PhISEM "Sticks" (brittle snaps, 5500 Hz, r 0.6)
    with 60 objects under a peel-force hump, plus a stick-slip band."""
    rng = np.random.default_rng(int(seed))
    dur = float(np.clip(dur, 0.1, 6))
    n = _n(dur)
    e = (0.25 + 0.75 * _c01(speed)) * np.sin(np.pi * np.arange(n) / n) ** 0.5 * (0.6 + 0.4 * _humps(n, rng, 12.0))
    y = _nm(_phisem("Sticks", dur, rng, e, mode="level", res=0.7, objects=60.0))
    return _fin(y + 0.4 * _nm(amb.bp(_slips(0.5 + 0.5 * e, rng, 3), 3200.0, 0.8)), sr, 0.6)


@sfx("paper", "Paper: rustle, tear, or a page turn.",
     kind=("page", "rustle|tear|page"), dur=(0.5, "s (rustle, tear)"), weight=(0.5, "0 tissue .. 1 heavy paper/card"), seed=(0, "variation"))
def paper(sr=DEFAULT_SR, kind="page", dur=0.5, weight=0.5, seed=0):
    """Port of procedural/interface (ui_screen_change): STK Sandpaper driven by the GRF hump
    as the hand's pressure curve, then the page settling as a soft felt contact at 150 ms, 30 %."""
    rng = np.random.default_rng(int(seed))
    res = 0.7 - 0.35 * _c01(weight)
    hump = lambda m: 1.5 * 3.3333 * (x := np.arange(m) / max(m - 1, 1)) * (1 - x) ** 2 * (1 + x)   # GRF polynomial
    if kind == "page":
        y = np.zeros(_n(0.3))
        _put(y, _nm(_phisem("Sandpaper", 0.18, rng, hump(_n(0.18)), mode="level", res=res)), 0.0)
        land = _drive("body_squareBig", np.pad(pc.felt_pulse(12.0, 2.5), (0, _n(0.12))))
        _put(y, _nm(land), 0.150, 0.3)
        return _fin(y, sr, 0.5)
    dur = float(np.clip(dur, 0.1, 10))
    n = _n(dur)
    if kind == "tear":
        e = np.clip(0.4 + 0.6 * np.arange(n) / n, 0, 1)
        y = _nm(_phisem("Sandpaper", dur, rng, e, mode="level", res=res * 1.3)) + 0.7 * _nm(amb.bp(_slips(e, rng, 3), 3000.0, 1.0))
    else:
        y = _phisem("Sandpaper", dur, rng, _humps(n, rng, 5.0, 2.0), mode="level", res=res)
    return _fin(y, sr, 0.55)


COINS = {"penny": (11000, 5200, 3835), "nickel": (5583, 9255, 9805), "dime": (4450, 4974, 9945), "quarter": (1708, 8863, 9045)}


COIN_WOBBLE_EXP, COIN_TILT_EXP = -2.0 / 9.0, 4.0 / 9.0    # air boundary-layer regime of a spinning disc


@sfx("coin_spin", "A coin spinning down on a table: the wobble accelerates, then stops dead.",
     coin=("quarter", "|".join(COINS)), dur=(1.8, "s until it lies flat"), start=(9.0, "wobble rate at the start, Hz"), seed=(0, "variation"))
def coin_spin(sr=DEFAULT_SR, coin="quarter", dur=1.8, start=9.0, seed=0):
    """Rolling-contact noise gated at the wobble rate into the STK coin resonances (Shakers coin
    rows) and a wooden table.
    Euler's disk losing its energy in the air boundary layer under it: tilt alpha ~ (t0 - t)^(4/9), and
    Omega^2 = 4g/(a alpha) gives a wobble rate ~ (t0 - t)^(-2/9). The rolling (Moffatt) regime would only
    hold for the last ~15 ms. The 40x cap is a design limit."""
    rng = np.random.default_rng(int(seed))
    dur = float(np.clip(dur, 0.3, 10))
    n = _n(dur)
    u = np.clip(1.0 - np.arange(n) / n, 1e-9, 1.0)                 # (t0 - t)/t0
    wob = np.minimum(float(start) * u ** COIN_WOBBLE_EXP, 40.0 * float(start))
    gate = (0.5 + 0.5 * np.sin(2 * np.pi * np.cumsum(wob) / P)) ** 4
    force = np.concatenate([gate * u ** COIN_TILT_EXP * rng.standard_normal(n), np.zeros(_n(0.25))])
    ring = _resons(force, COINS.get(str(coin), COINS["quarter"]), (0.9995,) * 3, (1.0, 0.8, 0.5))
    return _fin(_nm(ring) + 0.5 * _nm(_drive("body_squareSmall", amb.lop(force, 900.0))), sr, 0.6)


def _bounce_times(first: float, restitution: float, t_max: float) -> list[tuple[float, float]]:
    """(time, energy 0..1) of successive bounces: flight time and impact speed both scale by the
    restitution e each bounce."""
    out, t, gap, v = [], 0.0, first, 1.0
    while t < t_max and v > 0.02 and len(out) < 60:
        out.append((t, v))
        t += gap
        gap *= restitution
        v *= restitution
    return out


DICE_EDGE, DICE_CORNER_R, DICE_DROP_V = 0.016, 0.001, 2.4     # 16 mm acrylic die, 1 mm rounded corner, 0.3 m throw


def dice_contact_ms(v: float, surface: str = "wood") -> float:
    """Hertz contact time (ms) of an acrylic die corner (radius DICE_CORNER_R) on ``surface`` at v m/s:
    t = 2.94 (5 m / 4 k)^(2/5) v^(-1/5), k = 4/3 E* sqrt(R)."""
    from .contact import _hertz_pair, get_material
    m = get_material("acrylic").density * DICE_EDGE ** 3
    k = _hertz_pair("acrylic", surface, DICE_CORNER_R)
    return 1e3 * 2.94 * (5.0 * m / (4.0 * k)) ** 0.4 * max(float(v), 1e-3) ** -0.2


@sfx("dice", "Dice thrown on a table: each die bounces with shrinking intervals and rattles to rest.",
     count=(2, "dice"), surface=("wood", "wood|felt"), force=(0.7, "0..1 throw strength"), seed=(0, "variation"))
def dice(sr=DEFAULT_SR, count=2, surface="wood", force=0.7, seed=0):
    """Bounce schedule: one energy value drives interval, level and brightness."""
    rng = np.random.default_rng(int(seed))
    y = np.zeros(_n(1.6))
    felt = surface == "felt"
    for _ in range(int(np.clip(count, 1, 12))):
        t0 = rng.uniform(0.0, 0.12)
        for t, v in _bounce_times(rng.uniform(0.12, 0.22) * (0.6 + 0.6 * _c01(force)), rng.uniform(0.55, 0.7), 1.2):
            v *= 0.3 + 0.7 * _c01(force)
            cms = 1.5 if felt else dice_contact_ms(DICE_DROP_V * v)
            _put(y, _nm(_knock("sy_stick", rng, 2.5 * rng.uniform(0.95, 1.05), contact_ms=cms, vel=v, dur=0.12)), t0 + t, v)
            _put(y, _nm(_knock("body_squareMid", rng, contact_ms=6.0 if felt else 2.0, vel=v, felt=felt, dur=0.3)), t0 + t, 0.7 * v)
    return _fin(y, sr, 0.7)


END_FLANGED, END_UNFLANGED = 0.8216, 0.6133   # end corrections / r: pipe in an infinite flange, open unflanged pipe


def helmholtz_hz(volume_ml: float, neck_d_mm: float, neck_len_mm: float) -> float:
    """f = c/(2 pi) sqrt(A / (V L')), L' = L + (0.8216 + 0.6133) r: flanged end inside the bottle, unflanged end outside."""
    r = neck_d_mm * 0.5e-3
    return C_AIR / (2 * math.pi) * math.sqrt(math.pi * r * r / (volume_ml * 1e-6 * (neck_len_mm * 1e-3 + END_FLANGED * r + END_UNFLANGED * r)))


RHO_AIR = 1.2             # kg/m^3


def guitar_body_modes(top_hz: float = 200.0, hole_hz: float = 100.0, volume: float = 0.014, top_mass: float = 0.2,
                      top_area: float = 0.0814) -> tuple[float, float]:
    """The two low modes (Hz) of a box guitar: the top (piston of ``top_area`` m^2, ``top_mass`` kg, ``top_hz`` with the
    box closed) coupled to the sound-hole air (``hole_hz``, e.g. `helmholtz_hz`) through the air spring
    gamma = rho c^2 / V of ``volume`` m^3: (wp^2 - w^2)(wh^2 - w^2) = wa^2 wh^2, wa^2 = gamma Ap^2 / mp."""
    wp2, wh2 = (2 * math.pi * top_hz) ** 2, (2 * math.pi * hole_hz) ** 2
    wa2 = RHO_AIR * C_AIR ** 2 / volume * top_area ** 2 / top_mass
    s, p = wp2 + wh2, wh2 * max(wp2 - wa2, 0.0)
    r = math.sqrt(s * s - 4.0 * p)
    return math.sqrt((s - r) / 2.0) / (2 * math.pi), math.sqrt((s + r) / 2.0) / (2 * math.pi)


@sfx("cork_pop", "A cork pulled from a bottle: the pressure release rings the bottle's Helmholtz resonance and the neck.",
     volume_ml=(750.0, "air volume of the bottle, ml"), neck_mm=(18.5, "neck inner diameter, mm"), neck_len_mm=(80.0, "neck length, mm"),
     fizz=(0.0, "0..1 sparkling hiss after the pop"), seed=(0, "variation"))
def cork_pop(sr=DEFAULT_SR, volume_ml=750.0, neck_mm=18.5, neck_len_mm=80.0, fizz=0.0, seed=0):
    """Helmholtz resonator plus the neck's quarter-wave c/(4L) while the cork leaves."""
    rng = np.random.default_rng(int(seed))
    fh = float(np.clip(helmholtz_hz(max(volume_ml, 5.0), max(neck_mm, 2.0), max(neck_len_mm, 2.0)), 30.0, 4000.0))
    fq = float(np.clip(C_AIR / (4 * max(neck_len_mm, 2.0) * 1e-3), 100.0, 9000.0))
    n = _n(0.5 + 1.0 * _c01(fizz))
    x = np.zeros(n)
    p = modal.bang(1.5)
    x[:len(p)] = p
    x[:_n(0.006)] += 0.02 * rng.standard_normal(_n(0.006)) * _line(_n(0.006)) ** 2
    y = _nm(amb.bp(x, fh, 12.0)) + 0.5 * _nm(amb.bp(x, fq, 6.0))
    if fizz > 0:
        t = np.arange(n) / P
        y += 0.15 * _c01(fizz) * _nm(amb.hip(rng.standard_normal(n), 3000.0)) * np.minimum(1, t / 0.03) * np.exp(-t / 0.4)
    return _fin(y, sr, 0.8)


CLAMPED = (1.0, 6.267, 17.55, 34.39)                           # clamped-bar mode ratios
FREE_BAR = (1.0, 2.7565, 5.40392, 8.93295, 13.3443, 18.6379)   # free-bar series


@sfx("boing", "Twanged ruler / springy bar: clamped-bar modes with the pitch dropping in the attack.",
     freq=(426, "Hz or note name "), decay=(1.5, "s"), bend=(6.0, "Hz the pitch starts above the final pitch"),
     bright=(1.0, "0.3 dull .. 3 all overtones"), seed=(0, "variation"))
def boing(sr=DEFAULT_SR, freq=426, decay=1.5, bend=6.0, bright=1.0, seed=0):
    """Four sinusoids from one phase at the clamped ratios, amplitudes
    0.5, 0.25, 0.125, 0.0625, quartic envelope (1500 ms); six band-passes at the free-bar series
    driven by low-passed noise; a 16th-power envelope raises the fundamental by 6 Hz at the start;
    the squared fundamental phasor modulates both parts. Overtones roll off above 3 kHz x bright."""
    rng = np.random.default_rng(int(seed))
    f0 = float(np.clip(to_hz(freq), 20.0, 4000.0))
    dec = float(np.clip(decay, 0.05, 10))
    n = _n(dec + 0.05)
    t = np.arange(n) / P
    env = np.clip(1 - t / dec, 0, 1)
    ph = np.cumsum(f0 + float(bend) * env ** 16) / P
    mod = (ph % 1.0) ** 2
    roll = lambda f: 1.0 / (1.0 + (f / (3000.0 * max(float(bright), 0.05))) ** 2)
    y = sum(a * roll(r * f0) * np.sin(2 * np.pi * r * ph) for r, a in zip(CLAMPED, (0.5, 0.25, 0.125, 0.0625)) if r * f0 < 0.45 * P)
    nz = amb.lop(rng.uniform(-1, 1, n), 2.0 * f0) * mod * env ** 4
    free = sum(amb.bp(nz, r * f0, 30.0) * roll(r * f0) for r in FREE_BAR if r * f0 < 0.45 * P)
    return _fin(y * mod * env ** 4 + 0.5 * _nm(free) * float(np.max(np.abs(y))), sr, 0.7)


@sfx("ratchet", "Ratchet / pawl clicks (click-factory ratchet cluster) at a steady or slowing rate.",
     clicks=(10, "count"), rate=(30.0, "clicks per second"), slow=(0.0, "0..1: the rate falls towards the end"),
     size=(1.0, "mechanism size (bigger = lower)"), mount=("metal", "plastic|wood|metal"), seed=(0, "variation"))
def ratchet(sr=DEFAULT_SR, clicks=10, rate=30.0, slow=0.0, size=1.0, mount="metal", seed=0):
    """Click-factory cluster 3: 5432 Hz Q40 d40, 6290 Hz Q20 d22, 2470 Hz Q20 d8 on hip-300 noise."""
    rng = np.random.default_rng(int(seed))
    clicks = int(np.clip(clicks, 1, 400))
    gaps = (1.0 / max(float(rate), 1.0)) * (1.0 + 2.5 * _c01(slow) * np.linspace(0, 1, clicks) ** 2)
    times = np.concatenate([[0.0], np.cumsum(gaps[:-1])])
    y = np.zeros(_n(times[-1] + 0.15))
    parts = [(f / max(float(size), 0.05), q, d * min(max(float(size), 0.05), 3.0)) for f, q, d in RATCHET]
    for tk in times:
        m = _n(0.06 * min(max(float(size), 0.3), 3.0))
        _put(y, _mclick(amb.hip(rng.uniform(-1, 1, m), 300.0), parts), tk, rng.uniform(0.75, 1.0))
    return _fin(_panel(y, mount, gin=1.0), sr, 0.6)


def fractal_noise(n: int, rng, D: float = 1.3) -> np.ndarray:
    """Surface profile with spectrum ~ w^(2(D-2))."""
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / P)
    f[0] = f[1]
    y = np.fft.irfft(X * (f / 1000.0) ** (D - 2.0), n)
    return y / (np.std(y) + 1e-12)


def _scrape(v: np.ndarray, rng, lam: float = 1e-4, q: float = 2.0) -> np.ndarray:
    """Scrape: fractal profile -> reson at v / lambda, x sqrt(v)."""
    v = np.maximum(v, 0.0)
    return pc.tv_bandpass(fractal_noise(len(v), rng), np.clip(v / lam, 60.0, 12000.0), np.full(len(v), q)) * np.sqrt(v)


@sfx("drawer", "A wooden drawer sliding open or shut, ending on its stop.",
     direction=("close", "open|close"), dur=(0.5, "s of travel"), size=(1.0, "drawer size (bigger = lower)"),
     force=(0.6, "0..1: speed of the slide and weight of the stop"), seed=(0, "variation"))
def drawer(sr=DEFAULT_SR, direction="close", dur=0.5, size=1.0, force=0.6, seed=0):
    """FoleyAutomatic scrape force driving the big Faust wooden body, then one
    contact on the same body: a hard stop when closing, a softer catch when opening."""
    rng = np.random.default_rng(int(seed))
    dur = float(np.clip(dur, 0.1, 5))
    n = _n(dur)
    fs = 1.0 / max(float(size), 0.05)
    close = direction != "open"
    v = (0.2 + 0.6 * _c01(force)) * np.sin(np.pi * np.minimum(np.arange(n) / n * (0.62 if close else 1.0), 1.0)) ** 0.7
    y = np.zeros(n + _n(0.5))
    _put(y, _nm(_drive("body_squareBig", np.pad(_scrape(v, rng, lam=4e-4), (0, _n(0.3))), fs)), 0.0, 0.5)
    _put(y, _nm(_knock("body_squareBig", rng, fs, contact_ms=3.0 if close else 8.0, vel=0.4 + 0.6 * _c01(force), felt=not close)), dur, 1.0 if close else 0.45)
    return _fin(y, sr, 0.7)


@sfx("book_drop", "A book dropped flat: the cover's soft slap, the pages, and one small rebound.",
     size=(1.0, "book size (bigger = lower, heavier)"), height=(0.5, "0..1 drop height / force"), seed=(0, "variation"))
def book_drop(sr=DEFAULT_SR, size=1.0, height=0.5, seed=0):
    """Port of procedural/interface._cover_thud: a Hertz felt pulse (p 2.5, 25 ms) into the Faust big
    square body, with a Sandpaper page burst."""
    rng = np.random.default_rng(int(seed))
    y = np.zeros(_n(0.6))
    fs = 1.0 / max(float(size), 0.05)
    h = _c01(height)
    for t, g in ((0.0, 1.0), (0.09 + 0.08 * h, 0.25)):
        _put(y, _nm(_drive("body_squareBig", np.pad(pc.felt_pulse(25.0 - 12.0 * h, 2.5), (0, _n(0.3))), fs)), t, g)
        m = _n(0.07)
        _put(y, _nm(_phisem("Sandpaper", 0.07, rng, _line(m) ** 2, mode="level", res=0.45)), t, 0.35 * g * (0.4 + 0.6 * h))
    return _fin(y, sr, 0.75)


# ============================================================================== weapons
def _barrel(T: float) -> np.ndarray:
    """``pd barrel``: phi = (2v)^2, y = cos(2 pi (phi - 1/4)) v; falls from 8/T Hz to 0."""
    v = _line(_n(T))
    return np.cos(2 * np.pi * ((2 * v) ** 2 - 0.25)) * v


def _shell(T: float) -> np.ndarray:
    """Shell chirp phi = 64^(1.05 v) (the explosive1.pd reading; ~8.8 kHz -> 175 Hz over 39 ms)."""
    v = _line(_n(T))
    y = np.cos(2 * np.pi * (np.mod(64.0 ** (1.05 * v), 1.0) - 0.25))
    k = _n(0.0005)
    y[:k] *= np.linspace(0, 1, k)
    y[-k:] *= np.linspace(1, 0, k)
    return y


def _noiseburst(D: float, rng) -> np.ndarray:
    """``noiseburst``: a triangle oscillator at 100 + 5000 hip40(bp(noise, 3 kHz, Q2)), x v^8."""
    n = _n(D)
    fm = amb.hip(amb.bp(rng.uniform(-1, 1, n), 3000.0, 2.0), 40.0)
    ph = np.cumsum(np.maximum(100.0 + 5000.0 * fm, 0.0)) / P
    return (4.0 * np.abs((ph % 1.0) - 0.5) - 1.0) * _line(n) ** 8


def n_wave(trans_ms: float = 2.0, height_ms: float = 9.133, reflect: float = 1.0, rise_ms: float = 1.0) -> np.ndarray:
    """``pd nwave``: to +1 in ``rise_ms``, to -1 over ``trans_ms``, to 0 in ``rise_ms``; plus
    the ground reflection: a copy delayed by ``height_ms`` through a 100 Hz low-pass."""
    a, b = max(_n(rise_ms * 1e-3), 1), max(_n(trans_ms * 1e-3), 2)
    shape = np.concatenate([np.linspace(0, 1, a, endpoint=False), np.linspace(1, -1, b, endpoint=False), np.linspace(-1, 0, a)])
    d = _n(height_ms * 1e-3)
    y = np.zeros(len(shape) + d + _n(0.04))
    y[:len(shape)] = shape
    refl = np.zeros(len(y))
    refl[d:d + len(shape)] = shape
    return y + reflect * amb.lop(refl, 100.0)


def _bpar8(x: np.ndarray, centres, q: float = 10.62, gain: float = 0.15) -> np.ndarray:
    """``bpar8~``: 8 parallel band-passes sharing one Q (patch preset 10.62), input x 0.15."""
    y = np.zeros(len(x))
    for f in centres:
        y += amb.bp(x * gain, min(float(f), 0.45 * P), q)
    return y


RIFLE_BODY = (230.0, 380.0, 560.0, 760.0, 2200.0, 2500.0, 2800.0, 3050.0)   # "four between 200 and 800 Hz ... another cluster around 2.5 kHz"
# kind -> shell chirp ms, barrel ms, noise burst ms, bullet speed m/s, body formant scale, blast gain, blast low-pass Hz (0 = none),
#         automatic-fire rate (rounds/s), projectile: an `arms.CALIBERS` key or (diameter mm, length mm).
GUNS = {
    "rifle":    (38.97, 6.1, 7.677, 850.0, 1.0, 1.0, 0.0, 10.0, "762x51"),
    "pistol":   (30.0, 4.0, 6.0, 360.0, 1.4, 0.9, 0.0, 6.0, "9x19"),
    "smg":      (30.0, 5.0, 6.0, 400.0, 1.25, 0.9, 0.0, 13.3, "9x19"),
    "shotgun":  (45.0, 20.0, 30.0, 400.0, 0.75, 1.0, 0.0, 1.5, "12_slug"),
    "silenced": (30.0, 4.0, 6.0, 300.0, 1.4, 0.08, 0.0, 6.0, "9x19_sub"),
    "silenced_rifle": (38.97, 6.1, 7.677, 850.0, 1.0, 0.08, 0.0, 10.0, "762x51"),
    "cannon":   (60.0, 60.0, 200.0, 450.0, 0.3, 1.0, 0.0, 0.5, (117.0, 117.0)),
}
SUPPRESSED = {"silenced", "silenced_rifle"}
SUPPRESSOR_CUT = (160.0, 1000.0, 0.8)   # Hz band the can takes out most, depth of that cut
FIRST_POP = 1.6                         # first round through a cold can: the air inside burns too
PORT_LEVEL = 0.25                       # ejection-port gas, which the suppressor does not quiet
EAR_GAIN = {"center": 1.0, "right": 3.16, "left": 0.5}   # shooter's ear on the ejection-port side, +10 dB
CRACK_MISS = 3.0          # m between the bullet's path and the listener


def crack_cone_gain(angle: float, v_bullet: float) -> float:
    """1 inside the Mach cone (angle off the line of fire < 90 - asin(c/v) degrees), fading to 0 over 5 degrees
    outside it; 0 for a subsonic round."""
    if v_bullet <= C_AIR:
        return 0.0
    edge = 90.0 - math.degrees(math.asin(C_AIR / v_bullet))
    return float(np.clip((edge - float(angle)) / 5.0, 0.0, 1.0))


def crack_ms(kind: str, miss: float = CRACK_MISS) -> float:
    """Length of the passing bullet's N-wave in ms (`arms.n_wave_pa`); 0 when subsonic."""
    from .arms import CALIBERS, n_wave_pa
    g = GUNS[kind]
    d, l = CALIBERS[g[8]][:2] if isinstance(g[8], str) else g[8]
    return n_wave_pa(g[3], d, l, miss)[1] * 1e3


def _gun_action(kind: str, rng, n: int, amount: float) -> np.ndarray:
    """Recycling mechanics: a catch and the end-stop clunk; the rifle's "wooden knocking sound
    coming in about 90 ms after the impulse"."""
    y = np.zeros(n)
    if amount <= 0 or kind == "cannon":
        return y
    nz = amb.hip(rng.uniform(-1, 1, n), 300.0)
    y += _mclick(nz, CATCH, 0.045, 0.33) + _mclick(nz, ENDSTOP, 0.07, 0.33)
    if kind == "rifle":
        _put(y, _nm(_strike("sy_stick", 0.3, 1.5, 0.0, rng, dur=0.12)), 0.09, 0.25 * float(np.max(np.abs(y))))
    return y * amount


def _gunshot(kind: str, rng, action: float, crack: float, distance: float, echo: float, ground: str = "none", height: float = 1.6,
             angle=None, ear: str = "center") -> np.ndarray:
    chirp, barrel, burst, v_bullet, bscale, bgain, blop = GUNS[kind][:7]
    miss = CRACK_MISS
    if angle is not None:
        miss = max(0.5, distance * math.sin(math.radians(float(angle))))
        crack = crack * crack_cone_gain(float(angle), v_bullet)
    n = _n(0.45 + burst * 1e-3 + (0.5 if echo > 0 else 0.0))
    x = np.zeros(n)
    _put(x, _shell(chirp * 1e-3), 0.0, 0.6)                      # detonation, heard through the body
    _put(x, _barrel(barrel * 1e-3), 0.006, 1.0)                  # muzzle: "a bursting balloon"
    _put(x, _noiseburst(burst * 1e-3, rng), 0.0192, 0.8)         # gas
    if kind == "cannon":
        _put(x, _shock(0.03), 0.0, 1.5)
    close = float(np.clip(1.0 - distance / 100.0, 0.0, 1.0))     # "over 100 m [the body] is lost"
    x += _gun_action(kind, rng, n, action * close) * 0.5
    body = _bpar8(x, [f * bscale * rng.uniform(0.97, 1.03) for f in RIFLE_BODY])
    blast = np.tanh(0.6 * 8.0 * (body + 0.35 * x)) * bgain      # "natural compression": tanh(0.6 x), driven hard
    if blop:
        blast = amb.lop(amb.lop(blast, blop), blop)
    if kind in SUPPRESSED:
        lo, hi, depth = SUPPRESSOR_CUT
        fc = math.sqrt(lo * hi)
        blast = blast - depth * F.svf(blast, "bandpass", fc, P, q=fc / (hi - lo))     # unit gain, zero phase at fc
    if kind in SUPPRESSED or str(ear) != "center":
        port = np.zeros(len(blast))
        _put(port, _noiseburst(burst * 1e-3, rng), 0.0192, PORT_LEVEL * EAR_GAIN.get(str(ear), 1.0) * close)
        blast = blast + amb.hip(port, 500.0)
    if echo > 0:
        blast = blast + echo * 4.0 * (amb.outdoor_echoes(blast) - 0.2 * blast)   # street/treeline echoes
    blast = space.distance(blast, distance, ref=1.0)             # 1/r and air absorption
    if str(ground) != "none":
        from .reverbs import _ground_sigma
        nf = 1 << int(np.ceil(np.log2(len(blast) + _n(0.05))))
        H = space.ground_effect(np.fft.rfftfreq(nf, 1.0 / P), distance, height, height, _ground_sigma(ground))
        blast = np.fft.irfft(np.fft.rfft(blast, nf) * H, nf)[:len(blast)]
    super_ = v_bullet > C_AIR and crack > 0
    lag = distance * (1.0 / C_AIR - 1.0 / v_bullet) if super_ else 0.0   # the crack outruns the thump
    y = np.zeros(n + _n(lag))
    _put(y, blast, lag)
    if super_:
        _put(y, n_wave(crack_ms(kind, miss), rise_ms=1e3 / P), 0.0, crack * 0.6 * float(np.max(np.abs(blast)) + 1e-12))
    return y


@sfx("gunshot", "Gunshot: detonation chirp, muzzle burst and gas through the weapon body, mechanical action, supersonic crack, distance and echoes.",
     kind=("rifle", "|".join(GUNS)), distance=(10.0, "m to the shooter: 1/r, air absorption, and the blast lags the crack by d(1/c - 1/v)"),
     shots=(1, "rounds"), rate=(0.0, "rounds per second (0 = the weapon's own)"), action=(0.6, "0..1 mechanical recycling (lost beyond 100 m)"),
     crack=(1.0, "0..1 supersonic N-wave of the passing bullet (none for subsonic rounds)"), echo=(0.3, "0..1 street / treeline echoes"),
     ground=("none", "blast reflected off the ground: none | hard | grass | snow | asphalt ... | flow resistivity in cgs (as `distance`)"),
     height=(1.6, "m of the muzzle and the ear above that ground"),
     angle=(None, "degrees between the line of fire and the listener: the crack is heard only inside the Mach cone, its miss distance = distance sin(angle) (default: 3 m off the path)"),
     ear=("center", "center | right | left: the shooter's ear; right = beside the ejection port, its gas +10 dB"),
     seed=(0, "variation"))
def gunshot(sr=DEFAULT_SR, kind="rifle", distance=10.0, shots=1, rate=0.0, action=0.6, crack=1.0, echo=0.3, ground="none", height=1.6, seed=0,
            angle=None, ear="center"):
    """Timeline shell 0 ms, barrel 6 ms, noise burst 19.2 ms; all
    layers through one 8-band body (Q 10.62) and tanh; N-wave 4 ms long with its ground reflection
    9.133 ms later through lop 100. The crack arrives first; the muzzle blast and its echo tail
    follow d (1/c - 1/v_bullet) later (cone geometry), low-passed by the air. The noise burst
    is a gas tail, longer than a measured muzzle blast: ``arms.firearm`` is the measured one.
    ``ground``: image source at the same height, reflected (sqrt(d^2 + 4 h^2) - d) / c after the blast."""
    rng = np.random.default_rng(int(seed))
    kind = kind if kind in GUNS else "rifle"
    d = float(np.clip(distance, 1.0, 2000.0))
    shots = int(np.clip(shots, 1, 60))
    gap = 1.0 / (float(rate) if rate > 0 else GUNS[kind][7])
    ang = None if angle is None else float(np.clip(float(angle), 0.0, 180.0))
    one = [_gunshot(kind, rng, _c01(action), _c01(crack), d, _c01(echo), ground, float(np.clip(height, 0.05, 50.0)), ang, str(ear))
           for _ in range(min(shots, 4))]
    y = np.zeros(_n((shots - 1) * gap * 1.05) + max(len(o) for o in one))
    t = 0.0
    for k in range(shots):
        pop = FIRST_POP if k == 0 and kind in SUPPRESSED else 1.0
        _put(y, one[k % len(one)], t, rng.uniform(0.9, 1.0) * pop)
        t += gap * (1.0 + 0.01 * rng.standard_normal())        # jitter per interval, not growing with the round
    return _fin(y, sr, 0.9, fade=0.0005)


# ms after the ratchet starts -> cluster
RELOAD_CLICKS = ((1, RATCHET), (15, ENDSTOP), (28, CATCH), (36, CATCH), (87, RATCHET), (92, ENDSTOP), (96, CATCH), (101, CATCH),
                 (348, CATCH), (350, RATCHET), (357, CATCH), (362, ENDSTOP), (368, ENDSTOP), (383, RATCHET))


@sfx("reload", "Weapon reload / cocking: metal slide sweep, ratchet click clusters and the end-stop clunk through the weapon body.",
     kind=("rifle", "|".join(GUNS)), speed=(1.0, "0.5 slow and deliberate .. 2 fast"), slide=(1.0, "0..1 slide friction level"), seed=(0, "variation"))
def reload(sr=DEFAULT_SR, kind="rifle", speed=1.0, slide=1.0, seed=0):
    """reload.pd: slide = noise -> hip 1000 -> bp(4500 + 200 line, Q4) x line^2
    over 100 ms; the 14-click schedule over three click-factory clusters; everything through the
    same body as the shot, "so the shots and the reload sound are automatically aligned"."""
    rng = np.random.default_rng(int(seed))
    k = 1.0 / float(np.clip(speed, 0.25, 4.0))
    n = _n(0.62 * k + 0.2)
    x = np.zeros(n)
    m = _n(0.1 * k)
    ln = np.arange(m) / m
    _put(x, amb.bp(amb.hip(rng.uniform(-1, 1, m), 1000.0), 4500.0 + 200.0 * ln, 4.0) * ln ** 2, 0.0, 0.03 * 40 * _c01(slide))
    nz = amb.hip(rng.uniform(-1, 1, n), 300.0)
    for ms, cl in RELOAD_CLICKS:
        x += _mclick(nz, cl, (0.1 + ms * 1e-3) * k, 0.33)
    x = amb.hip(x * 2.0, 100.0)
    return _fin(_bpar8(x, [f * GUNS.get(kind, GUNS["rifle"])[4] for f in RIFLE_BODY], gain=1.0) + 0.5 * x, sr, 0.7)


@sfx("casing", "Spent shell casings (or coins, small brass) bouncing on a hard floor.",
     count=(1, "casings"), size=(1.0, "bigger = lower (1 = rifle brass at 4.5-5.5 kHz); ignored with caliber"), spread=(0.25, "s over which they land"),
     caliber=("", "empty = generic brass | 9mm|45acp|556|762|50bmg|12ga|20mm|105mm: modes of that case"),
     floor=("concrete", "with caliber: concrete|wood|metal|tile|dirt|grass"), height=(1.2, "m it falls, with caliber"),
     roll=(False, "true = it rolls away after the bounces (hard floors, with caliber)"),
     ricochet=(False, "true = the film / game case: a long ring, spinning, bounces speeding up into a rattle (`casing_ricochet`; 556 brass without caliber)"),
     seed=(0, "variation"))
def casing(sr=DEFAULT_SR, count=1, size=1.0, spread=0.25, caliber="", floor="concrete", height=1.2, roll=False, ricochet=False, seed=0):
    """shellcasings.pd: noise x e^4 (0-39 ms) -> 3 bp Q800; bounces at 75 ms, +170-344 ms,
    +75-149 ms with gains 0.416, 0.257, 0.05."""
    if ricochet:
        from .arms import casing_ricochet
        return casing_ricochet(sr, caliber or "556", floor, height, count, spread, seed=seed)
    if caliber:
        from .arms import casing_physical
        return casing_physical(sr, caliber, floor, height, count, spread, roll, seed)
    rng = np.random.default_rng(int(seed))
    count = int(np.clip(count, 1, 40))
    y = np.zeros(_n(1.2 + float(spread)))
    for c in range(count):
        t = 0.075 + (rng.uniform(0, float(spread)) if c else 0.0)
        for g, gap in ((0.416, 0.0), (0.257, rng.uniform(0.170, 0.344)), (0.05, rng.uniform(0.075, 0.149))):
            t += gap
            _put(y, amb.tinkle(rng, f_scale=1.0 / max(float(size), 0.05), gain=g), t)
    return _fin(y, sr, 0.6)


BLADES = {"sword": "SWD1", "sword2": "SWD2", "golf_club": "PGA", "bat": "MLB", "broom": "BROOM", "wood_sword": "WOODSWD"}


def _swing(blade: str, speed: float, rng, model: str = "loq", thick: float = 1.0) -> np.ndarray:
    p = dict(modal.SWING_PRESETS[BLADES.get(str(blade), "SWD1")])
    p["TopSpeed"] *= float(np.clip(speed, 0.2, 3.0))
    p["HiltThick"] *= thick
    p["TipThick"] *= thick
    return modal.sword_swing(p, rng, model="patch" if model == "patch" else "loq").mean(axis=1)


@sfx("sword", "A blade or club swung through the air: Aeolian tones of eight sources along the blade (measured blade presets).",
     blade=("sword", "|".join(BLADES) + " (thin sword, thicker sword, golf club, baseball bat, broom, wooden sword)"), speed=(1.0, "tip-speed scale (1 = the preset, e.g. 36 m/s)"),
     thickness=(1.0, "diameter scale (thicker = lower)"), model=("loq", "loq (St 0.2, Q 10: rated best) | patch (Fey Strouhal, bandwidth Q)"),
     groove=(0.0, "0..1 level of the fuller groove's cavity tones (Rossiter modes of a 3 mm groove: ~3.2, 7.5, 11.8 kHz at 24 m/s)"),
     seed=(0, "variation"))
def sword(sr=DEFAULT_SR, blade="sword", speed=1.0, thickness=1.0, model="loq", groove=0.0, seed=0):
    """f = St u / d per source, dipole gain ~ u^6, Doppler. ``groove``: the flow over the blade's groove
    (``GROOVE``: length along the flow, depth, share of the blade it runs) adds ``modal.cavity_tone`` at the
    groove's mid-point speed."""
    rng = np.random.default_rng(int(seed))
    y = _swing(blade, speed, rng, model, float(np.clip(thickness, 0.2, 5)))
    g = float(np.clip(groove, 0.0, 1.0))
    if g > 0:
        p = dict(modal.SWING_PRESETS[BLADES.get(str(blade), "SWD1")])
        p["TopSpeed"] *= float(np.clip(speed, 0.2, 3.0))
        L, gd, share = GROOVE
        _, _, u_tip, _ = modal._swing_kinematics(p, 0.05, 0.15)
        mid = modal.ARM + p["Len"] * (1.0 - 0.5 * share)
        u = u_tip * mid / (modal.ARM + p["Len"])
        c = modal.cavity_tone(u, L, gd, share * p["Len"], len(u) / P, rng)[:len(y)]
        ry, rc = np.sqrt(np.mean(y * y)), np.sqrt(np.mean(c * c))
        if rc > 0:
            y = y + g * c[:len(y)] * ry / rc
    return _fin(y, sr, 0.7)


GROOVE = (3.07e-3, 1.5e-3, 0.7)     # m along the flow, m deep, share of the blade length


def _speed_curve(v, n: int, gust: float, rng) -> np.ndarray:
    """Number or breakpoint list -> per-sample speed at P, with slow 1/f-ish gusts of relative size ``gust``."""
    from .matter import _lfn
    a = np.atleast_1d(np.asarray(v, float))
    u = np.interp(np.linspace(0, a.size - 1, n), np.arange(a.size), a) if a.size > 1 else np.full(n, float(a[0]))
    if gust > 0:
        # gusts "in the 0.5Hz range" and squalls "in the 3Hz range".
        u = u * (1.0 + gust * (0.35 * _lfn(rng, n, 0.5) + 0.1 * _lfn(rng, n, 3.0)))
    return np.maximum(u, 0.05)


@sfx("cavity_tone", "Wind over an opening (window gap, bottle mouth, wheel well, slot): Rossiter tones + the cavity's depth resonance.",
     speed=(15.0, "m/s air speed; number or list of breakpoints"), length=(0.05, "m along the flow"),
     depth=(0.05, "m: deep (L/d < 1) = the depth resonance rules; very shallow (L/d > 8-11) closes the cavity"),
     width=(0.05, "m across the flow"), elevation=(90.0, "degrees between the flow and the listener (90 = overhead)"),
     gust=(0.0, "0..1 slow gusts"), dur=(2.0, "s"), seed=(0, "variation"))
def cavity_tone(sr=DEFAULT_SR, speed=15.0, length=0.05, depth=0.05, width=0.05, elevation=90.0, gust=0.0, dur=2.0, seed=0):
    rng = np.random.default_rng(int(seed))
    d = _c01(dur, 0.05, 60.0)
    u = _speed_curve(speed, pc.seconds(d), _c01(gust), rng)
    return _fin(modal.cavity_tone(u, _c01(length, 0.002, 5.0), _c01(depth, 0.002, 5.0), _c01(width, 0.002, 5.0), d, rng,
                                  elevation=float(elevation)), sr, 0.7, fade=0.05)


@sfx("edge_tone", "Air jet on a sharp edge (whistle, recorder mouth, slot on a blade): stepped modes with hysteresis.",
     speed=(15.0, "m/s jet speed; number or list of breakpoints (sweep it to hear the mode jumps)"),
     gap=(10.0, "mm nozzle-to-edge distance; number or list of breakpoints"), nozzle=(1.0, "mm nozzle exit width (gap / nozzle 2..61)"),
     profile=("parabolic", "parabolic (long flue) | top_hat (short flue, modes 1-3)"),
     elevation=(0.0, "degrees off the dipole axis (90 = silent)"), gust=(0.0, "0..1 slow gusts"), dur=(2.0, "s"), seed=(0, "variation"))
def edge_tone(sr=DEFAULT_SR, speed=15.0, gap=10.0, nozzle=1.0, profile="parabolic", elevation=0.0, gust=0.0, dur=2.0, seed=0):
    rng = np.random.default_rng(int(seed))
    d = _c01(dur, 0.05, 60.0)
    n = pc.seconds(d)
    u = _speed_curve(speed, n, _c01(gust), rng)
    h = np.clip(_speed_curve(gap, n, 0.0, rng), 0.02, 500.0) * 1e-3
    prof = "top_hat" if str(profile).replace("-", "_") in ("top_hat", "tophat") else "parabolic"
    return _fin(modal.edge_tone(u, h, _c01(nozzle, 0.05, 20.0) * 1e-3, d, rng, profile=prof,
                                elevation=_c01(elevation, 0.0, 89.0)), sr, 0.7, fade=0.03)


@sfx("aeolian_harp", "Strings singing in the wind: Aeolian tones that lock in to a string harmonic and sing.",
     wind=(6.0, "m/s; number or list of breakpoints"), gust=(0.5, "0..1 slow gusts (they sweep the strings in and out of lock-in)"),
     notes=("D3 A3 D4 F#4", "the strings' fundamentals"), diameter=(0.5, "mm string diameter (thinner = higher shedding)"),
     index=(1.0, "FM index of the singing tone (more = richer sidebands)"),
     damping=(0.05, "mass-damping of the fundamental (alpha): lower sings louder and higher up"),
     dur=(6.0, "s"), seed=(0, "variation"))
def aeolian_harp(sr=DEFAULT_SR, wind=6.0, gust=0.5, notes="D3 A3 D4 F#4", diameter=0.5, index=1.0, damping=0.05, dur=6.0, seed=0):
    rng = np.random.default_rng(int(seed))
    d = _c01(dur, 0.2, 120.0)
    u = _speed_curve(wind, pc.seconds(d), _c01(gust), rng)
    f = [to_hz(s) for s in str(notes).replace(",", " ").split()] or [146.8]
    return _fin(modal.aeolian_harp(u, f, d, rng, diameter=_c01(diameter, 0.05, 10.0) * 1e-3, mass_damping=_c01(damping, 0.0, 2.0),
                                   index=_c01(index, 0.0, 10.0)), sr, 0.7, fade=0.2)


@sfx("clash", "Blade on blade: both measured swords ring from one shared steel-on-steel contact.",
     force=(0.8, "0..1"), size=(1.0, "blade size (bigger = lower)"), damping=(2.0, "decay-rate scale (hands damp the blades)"), seed=(0, "variation"))
def clash(sr=DEFAULT_SR, force=0.8, size=1.0, damping=2.0, seed=0):
    """Port of procedural/techniques.clash: sword1.sy and sword2.sy, 0.05 ms contact,
    3 and 2 FoleyAutomatic micro-collisions, random strike points."""
    rng = np.random.default_rng(int(seed))
    fs, v, ds = 1.0 / max(float(size), 0.05), 0.2 + _c01(force), max(float(damping), 0.1)
    a = _strike("sy_sword1", fs * rng.uniform(0.98, 1.02), 0.05, rng.uniform(0, 1), rng, velocity=v, micro=3, d_scale=ds, max_dur=1.2)
    b = _strike("sy_sword2", fs * rng.uniform(0.98, 1.02), 0.05, rng.uniform(0, 1), rng, velocity=v, micro=2, d_scale=ds, max_dur=1.2)
    n = max(len(a), len(b))
    return _fin(pc.fit(_nm(a), n) + 0.7 * pc.fit(_nm(b), n), sr, 0.8)


# target -> contact ms for (slash, stab). Contact ranges: steel on steel ~0.05-0.3 ms, wood
# 1-3 ms, flesh "10-50 ms" thud.
_TARGET_MS = {"flesh": (9.0, 14.0), "armour": (0.3, 0.2), "wood": (1.5, 2.5)}
# weapon -> body table, f scale, d scale, second body (table, f scale, gain) or None, speed,
#           contact ms on (flesh, shell, stone), micro-collisions on (shell, stone)
WEAPON_BODIES = {
    "blade": ("sy_sword1", 1.0, 3.0, None, 1.0, (15.0, 0.6, 0.05), (1, 4)),
    "heavy": ("sy_wok", 0.6, 3.0, ("sy_stick", 0.3, 0.5), 1.5, (35.0, 0.8, 0.1), (1, 3)),
    "spear": ("sy_sword2", 2.0, 4.0, ("sy_stick", 0.35, 0.5), 0.9, (12.0, 0.5, 0.05), (1, 3)),
    "improvised": ("sy_stick", 0.3, 1.0, None, 0.8, (30.0, 1.5, 1.5), (0, 1)),
}
STONE_SEED = 20170515
_TARGETS = ("flesh", "armour", "wood", "stone", "shell")


def _hit_ms(weapon: str, target: str, stab_: bool) -> float:
    w = WEAPON_BODIES[weapon]
    if target in ("stone", "shell"):
        return w[5][2 if target == "stone" else 1]
    if weapon == "blade" or target == "wood":
        return _TARGET_MS[target][1 if stab_ else 0]
    return w[5][0 if target == "flesh" else 1]


def _weapon_hit(target: str, stab_: bool, force: float, rng, weapon: str = "blade") -> np.ndarray:
    """One weapon contact (port of procedural/weapons.hit, simplified): one force pulse drives the
    weapon and the target."""
    target = target if target in _TARGETS else "flesh"
    weapon = weapon if weapon in WEAPON_BODIES else "blade"
    table, wf, wd, second, speed, _, micro = WEAPON_BODIES[weapon]
    vel = (0.3 + 0.9 * force) * speed
    T = modal.contact_time(_hit_ms(weapon, target, stab_), vel)
    n = _n(1.2)
    f = np.zeros(n)
    i0 = _n(0.002)
    p = modal.bang(T, vel)
    f[i0:i0 + len(p)] += p
    blade = modal.resolve(table, wf, rng.uniform(0, 1), d_scale=wd)
    k = 3 if target == "armour" and weapon == "blade" else micro[0] if target in ("armour", "shell") else micro[1] if target == "stone" else 0
    if k:
        modal.micro_collisions(f, blade, k, T, vel, rng, start=i0)
    y = _nm(modal.render_modes(f, blade), 1.0)
    pk = lambda z: z / (float(np.max(np.abs(z))) + 1e-12)
    if second:
        y = y + second[2] * pk(modal.render_modes(f, modal.resolve(second[0], second[1])))
    if target == "stone":
        chips = amb.hip(rng.uniform(-1, 1, n), 3000.0) * np.exp(-np.arange(n) / _n(0.004))
        y = 0.5 * pk(y) + pk(modal.render_modes(f, modal._ground_slab(np.random.default_rng(STONE_SEED)))) + 0.3 * pk(chips)
    elif target == "shell":
        cr = modal.fracture("sy_vase", rng, toughness=200.0 / vel, t_break=0.006, fall_height=0.03, ground=False, velocity=vel, dur=1.2)
        y = 0.5 * pk(y) + pk(pc.fit(cr, n))
    elif target == "flesh":
        # dull thud: stick.sy at fscale 0.1; contact noise "scaled by the pulse"
        # through a one-pole. The long pulse leaves the blade nearly silent on its own.
        body = modal.render_modes(f, modal.resolve("sy_stick", 0.1))
        wet = np.zeros(n)
        wet[i0:i0 + len(p)] = p / p.max() * rng.uniform(-1, 1, len(p))
        wet = lfilter([0.25], [1.0, -0.75], wet)
        y = 0.02 * y + pk(body) + (0.4 if stab_ else 0.5) * pk(wet)
    elif target == "armour":
        y = y + pk(modal.render_modes(f, modal.resolve("sy_wok", 1.0, d_scale=1.5)))
    else:
        y = 0.25 * y + pk(modal.render_modes(f, modal.resolve("sy_stick", 0.6))) + 0.6 * pk(modal.render_modes(f, modal.resolve("body_squareMid", 1.0)))
    drive = (1.0 if target == "flesh" else 2.0) * vel
    return np.tanh(0.6 * drive * pk(y)) / math.tanh(0.6 * drive)


_HIT_PARAMS = dict(target=("flesh", "|".join(_TARGETS) + " (shell = a creature's chitin)"), force=(0.7, "0..1"),
                   weapon=("blade", "|".join(WEAPON_BODIES) + " (heavy = mace / axe head, improvised = a stick or bar)"), seed=(0, "variation"))
_SWING_OF = {"blade": "sword", "heavy": "bat", "spear": "sword2", "improvised": "bat"}


@sfx("stab", "A weapon point driven into flesh, armour, wood, stone or a shell (no swing).", **_HIT_PARAMS)
def stab(sr=DEFAULT_SR, target="flesh", force=0.7, weapon="blade", seed=0):
    return _fin(pc.fit(_weapon_hit(target, True, _c01(force), np.random.default_rng(int(seed)), weapon), _n(0.9)), sr, 0.8)


@sfx("slash", "A weapon swing: the swing through the air, then the blow landing on flesh, armour, wood, stone or a shell.",
     swing=(0.5, "0..1 level of the air swoosh before the cut"), **_HIT_PARAMS)
def slash(sr=DEFAULT_SR, target="flesh", force=0.7, swing=0.5, weapon="blade", seed=0):
    rng = np.random.default_rng(int(seed))
    sw = _nm(_swing(_SWING_OF.get(str(weapon), "sword"), 0.8 + 0.4 * _c01(force), rng))
    at = int(np.argmax(np.convolve(sw ** 2, np.ones(_n(0.02)), "same"))) / P      # the cut lands at the loudest point of the swing
    y = np.zeros(_n(at + 1.0))
    _put(y, sw, 0.0, 0.5 * _c01(swing))
    _put(y, _weapon_hit(target, False, _c01(force), rng, str(weapon)), at)
    return _fin(y, sr, 0.8)


@sfx("bow_shot", "Bow and arrow: the draw creaks, the string twangs, the shaft whistles away and thunks into the target.",
     draw=(0.35, "s of draw creak (0 = none)"), distance=(20.0, "m to the target: the thunk comes distance / 70 m/s later (0 = no impact)"),
     pitch=(120.0, "bowstring Hz (90-160)"), target=("wood", "wood|flesh|armour"), seed=(0, "variation"))
def bow_shot(sr=DEFAULT_SR, draw=0.35, distance=20.0, pitch=120.0, target="wood", seed=0):
    """Port of procedural/techniques.c_archery: creak (pine, force 0.32 -> 0.7), extended
    Karplus-Strong bowstring, LoQ tone of an 8 mm shaft at 70 m/s slowing 20 %, arrival."""
    rng = np.random.default_rng(int(seed))
    draw = float(np.clip(draw, 0.0, 3.0))
    fl = float(np.clip(distance, 0.0, 150.0)) / 70.0
    y = np.zeros(_n(draw + fl + 1.4))
    if draw > 0.05:
        f = np.linspace(0.32, 0.7, _n(draw))
        _put(y, _nm(amb.creak(f, rng, material="pine", size=0.38)), 0.0, 0.35)
    tw = wg.pluck(float(np.clip(pitch, 40, 400)), 0.6, rng, position=0.5, brightness=0.5, body="none", velocity=0.9, sustain=0.5)
    _put(y, _nm(tw), draw, 0.8)
    m = _n(min(max(fl, 0.12), 0.45))                                               # heard while it is near the archer
    u = 70.0 * (1.0 - 0.2 * np.arange(m) / m)
    _put(y, _nm(modal.aeolian_tone(u, 0.008, m / P, rng, "loq") * np.sin(np.pi * np.arange(m) / m) ** 0.5), draw + 0.01, 0.5)
    if distance > 0:
        g = 1.0 / (1.0 + float(distance) / 15.0)                                  # the far target is quieter
        _put(y, _weapon_hit(target, True, 0.6, rng), draw + fl, g)
    return _fin(y, sr, 0.8)


@sfx("whip", "Whip crack: the thong accelerates past the speed of sound and its tip makes a small sonic boom.",
     dur=(0.35, "s of swing before the crack"), size=(1.0, "whip length scale: longer N-wave, lower whoosh"), echo=(0.2, "0..1 outdoor echoes"),
     seed=(0, "variation"))
def whip(sr=DEFAULT_SR, dur=0.35, size=1.0, echo=0.2, seed=0):
    """The Aeolian tone of the 2.5 mm lash: the lash moves at U for ``dur`` minus the loop's run, the loop
    then runs to the tip (``whip_speed``) until Mach 1, the tip goes on linearly to its measured
    peak, and the N-wave with its ground reflection is fired at the peak."""
    rng = np.random.default_rng(int(seed))
    s = float(np.clip(size, 0.3, 4))
    t, u = whip_speed(WHIP_LASH[2] * s)
    lead = max(float(np.clip(dur, 0.08, 2.0)) - t[-1], 0.0)
    m = _n(lead + t[-1])
    u = np.interp(np.arange(m) / P - lead, t, u, left=u[0])
    sw = modal.aeolian_tone(u, 0.0025 * s, m / P, rng, "loq") * (u / C_AIR) ** 2
    y = np.zeros(m + _n(0.9))
    _put(y, _nm(sw), 0.0, 0.25)
    crack = np.pad(n_wave(0.5 * s, 9.133, 0.7), (0, _n(0.6)))
    if echo > 0:
        crack = crack + _c01(echo) * 4.0 * (amb.outdoor_echoes(crack) - 0.2 * crack)
    _put(y, crack, m / P)
    return _fin(y, sr, 0.9, fade=0.0005)


WHIP_LASH = (0.097, 0.37e-3, 3.2)        # lash mass kg, cracker mass kg, lash length m
WHIP_PEAK, WHIP_SUPER = 744.0, 850e-6    # tip peak m/s, time from Mach 1 to it


def whip_speed(length: float = 3.2, n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """(t s, tip speed m/s) from the loop leaving the handle to the peak. Energy
    (M + m) U^2 = (M (L + x)/(2L) + m) V^2, the cracker at x running from L to -L with dx/dt = -V, so
    V(t) = Vmax / ((M/m + 1)^1.5 - 3 (M/m) Vmax t / (4 L))^(1/3), Vmax = U sqrt(M/m + 1), up to Mach 1;
    then linear to the measured peak. U is set so that Vmax is that peak."""
    mu = WHIP_LASH[0] / WHIP_LASH[1]
    L = float(length)
    t1 = 4.0 * L / (3.0 * mu * WHIP_PEAK) * ((mu + 1.0) ** 1.5 - (WHIP_PEAK / C_AIR) ** 3)
    t = np.linspace(0.0, t1, n)
    v = WHIP_PEAK / np.maximum((mu + 1.0) ** 1.5 - 3.0 * mu * WHIP_PEAK * t / (4.0 * L), 1e-12) ** (1.0 / 3.0)
    ts = t1 + np.linspace(0.0, WHIP_SUPER, 64)[1:]
    return np.concatenate([t, ts]), np.concatenate([v, C_AIR + (WHIP_PEAK - C_AIR) * (ts - t1) / WHIP_SUPER])


# slider settings of the explosion patch (0..1): mix levels, delays in s, comb width in ms, sweep in s
_XL = dict(shock=1.0, fire=2900 / 6300, blast=4900 / 6300, comb=4100 / 6300, press=2400 / 6300, fire_fx=1.0, blast_fx=2300 / 6300,
           fire_at=0.5 * 1400 / 6300, press_at=0.1 * 4200 / 6300, press_sweep=8.0 * 4700 / 6300, feedback=0.99 * 4600 / 6300,
           comb_in=0.1 * 3200 / 6300, comb_at=0.2 * 2900 / 6300, width=20.0 * 4300 / 6300, decay=2500 / 6300, spread=2200 / 6300)


def blast_times(spread: float, rng) -> list[tuple[float, float]]:
    """(s, noise value) of the three blasts: the first after 0.1 spread + 5 ms, the next two after random
    waits of 0 .. 0.3 spread s each; values (8..15)/200, (15..24)/200, (25..29)/200."""
    top = max(int(300.0 * spread), 1)
    t0 = (100.0 * spread + 5.0) * 1e-3
    t1 = t0 + int(rng.integers(top)) * 1e-3
    t2 = t1 + int(rng.integers(top)) * 1e-3
    return [(t0, (8 + int(rng.integers(8))) / 200.0), (t1, (15 + int(rng.integers(10))) / 200.0), (t2, (25 + int(rng.integers(5))) / 200.0)]


def _pd_blast(n: int, events, decay: float, rng) -> np.ndarray:
    """Blast noise: one generator retriggered by each (time, value x): vcf glides in 30 ms to 7000 x^2 Hz,
    Q = 1.5 + 3.5 (1 - x); x 50, clipped, x (e^2 + 0.5), cos, band 6 kHz Q 1, x e^2, three 20 Hz high-passes,
    x 0.3; e falls from 1 to 0 over (1 - x) 7 decay s."""
    t = np.arange(n) / P
    e, fc, q = np.zeros(n), np.zeros(n), np.full(n, 2.0)
    prev = 0.0
    for t0, x in events:
        m = t >= t0
        e[m] = np.clip(1.0 - (t[m] - t0) / ((1.0 - x) * decay * 7.0), 0.0, 1.0)
        target = 7000.0 * x * x
        fc[m] = prev + (target - prev) * np.minimum((t[m] - t0) / 0.03, 1.0)
        q[m] = 1.5 + 3.5 * (1.0 - x)
        prev = target
    v = np.clip(50.0 * amb.hip(amb.vcf(rng.uniform(-1.0, 1.0, n), np.maximum(fc, 1.0), q), 20.0), -1.0, 1.0)
    y = amb.bp(np.cos(2.0 * np.pi * v * (e * e + 0.5)), 6000.0, 1.0) * e * e
    return 0.3 * amb.hip(amb.hip(amb.hip(y, 20.0), 20.0), 20.0)


@numba.njit(cache=True)
def _pressure_k(x, g, fb, rate):
    """Variable delay of 500 (1 - g) ms read x g, DC-trapped at 10 Hz, added to the input, low-passed at
    10 kHz; the output is written back x fb."""
    n = x.shape[0]
    buf = np.zeros(n)
    y = np.zeros(n)
    cl = 1.0 - math.exp(-2.0 * math.pi * 10000.0 / rate)
    ch = math.exp(-2.0 * math.pi * 10.0 / rate)
    lo = hx = hy = 0.0
    for i in range(n):
        p = i - 0.5 * rate * (1.0 - g[i])
        r = 0.0
        if p >= 0.0:
            j = int(p)
            fr = p - j
            r = buf[j] * (1.0 - fr) + (buf[j + 1] * fr if j + 1 < i else 0.0)
        w = r * g[i]
        hy = w - hx + ch * hy
        hx = w
        lo += cl * (x[i] + hy - lo)
        y[i] = lo
        buf[i] = fb * lo
    return y


@sfx("explosion_layers", "Explosion built in layers: a chirped shock thump, three harsh blasts, an 8 s rising fireball rumble, a delayed echo the pressure wave squeezes and a comb swept as the fireball rises.",
     spread=(_XL["spread"], "0..1 gaps between the blasts (0.1 spread + 5 ms to the first, up to 0.3 spread s each after)"),
     fireball=(8.0, "s the fireball rumble takes to die away"), seed=(0, "variation"))
def explosion_layers(sr=DEFAULT_SR, spread=_XL["spread"], fireball=8.0, seed=0):
    from .klang import _comb3
    from .physics import fireball as _fireball
    rng = np.random.default_rng(int(seed))
    fall = float(np.clip(fireball, 0.5, 20.0))
    n = _n(_XL["fire_at"] + fall + 0.8)
    shock = np.zeros(n)
    _put(shock, _shock(0.010), 0.0)
    blast = _pd_blast(n, blast_times(_c01(spread), rng), _XL["decay"], rng)
    fire = np.zeros(n)
    _put(fire, _fireball(fall, 0.1, sr=P, seed=int(rng.integers(1 << 30))), _XL["fire_at"])
    fx = _XL["fire_fx"] * fire + _XL["blast_fx"] * blast
    t = np.arange(n) / P
    g = np.where(t < _XL["press_at"], 0.0, np.clip(1.0 - (t - _XL["press_at"]) / _XL["press_sweep"], 0.0, 1.0)) - 1e-4
    press = _pressure_k(fx, g, _XL["feedback"], float(P))
    v = np.where(t < _XL["comb_at"], 0.0, np.clip(1.0 - (t - _XL["comb_at"]) / 5.0, 0.0, 1.0))
    s = (4.0 * (1.0 - v) + 1e-4) * P * 1e-3
    cin = np.zeros(n)
    _put(cin, fx, _XL["comb_in"])
    comb = _comb3(np.ascontiguousarray(cin), s + 10e-3 * P, s + 10e-3 * P + s * _XL["width"], s + 10e-3 * P + 2.0 * s * _XL["width"], v * 0.33)
    y = (_XL["shock"] * shock + _XL["fire"] * fire + _XL["blast"] * blast + _XL["comb"] * comb + _XL["press"] * press)
    return _fin(np.clip(y, -1.0, 1.0), sr, 0.9, tail=0.3)


@sfx("shield_bash", "A shield slammed into something: a heavy long contact on a wooden or iron shield with its boss and rattle.",
     material=("wood", "wood (round shield with an iron boss)|metal"), force=(0.8, "0..1"), size=(1.0, "shield size (bigger = lower)"), seed=(0, "variation"))
def shield_bash(sr=DEFAULT_SR, material="wood", force=0.8, size=1.0, seed=0):
    """One 6 ms contact shared by the shield body, its iron
    parts (wok.sy, 2 micro-collisions) and a shockwave thump."""
    rng = np.random.default_rng(int(seed))
    fs, v = 1.0 / max(float(size), 0.05), 0.3 + 0.9 * _c01(force)
    y = np.zeros(_n(1.2))
    if material == "metal":
        _put(y, _nm(_knock("sy_computertower", rng, fs, d_scale=2.0, contact_ms=4.0, vel=v, micro=2, dur=1.0)), 0, 1.0)
        _put(y, _nm(_knock("sy_wok", rng, fs * 0.8, d_scale=4.0, contact_ms=2.0, vel=v, micro=3, dur=1.0)), 0, 0.7)
    else:
        _put(y, _nm(_knock("body_roundBig", rng, fs * 0.7, contact_ms=6.0, vel=v, dur=0.6)), 0, 1.0)
        _put(y, _nm(_knock("sy_wok", rng, fs * 1.2, d_scale=8.0, contact_ms=3.0, vel=v, micro=2, dur=0.5)), 0.001, 0.35)
    _put(y, _nm(_shock(0.014)), 0.0, 0.6)
    return _fin(np.tanh(1.5 * _nm(y)), sr, 0.85)


BODY_SET = (2225.0, 647.0, 1278.0, 1329.0, 2615.0, 500.0, 1682.0, 2243.0)
SHELL_SET = (5000.0, 6000.0, 7000.0, 8000.0)
BODY_LAW = {"flesh": (0.5, 0.5), "stone": (1.0, 0.3), "shell": (1.0, 0.4), "metal": (1.0, 0.7), "bone": (0.8, 0.5)}   # f x a / size^b


def body_hit_scale(material: str, size: float) -> float:
    """Frequency factor of a creature's body: a / size^b."""
    a, b = BODY_LAW[material]
    return a / float(size) ** b


@sfx("body_hit", "A blow landing on a creature's body: flesh, stone hide, chitin shell, metal plating or bone, scaled by the creature's size.",
     material=("flesh", "|".join(BODY_LAW)), size=(1.0, "creature size (bigger = lower; flesh also longer)"), seed=(0, "variation"))
def body_hit(sr=DEFAULT_SR, material="flesh", size=1.0, seed=0):
    """Flesh: barrel of 20 sqrt(size) ms + thump x 0.7 + noise burst 7.7 ms x 0.3 through the 8 body bands (Q 3), tanh;
    stone: chirp 38.97 ms + the noise burst through the same bands (Q 10.62); shell: 0.2 ms pulse into 5-8 kHz bands
    (Q 30) x 0.7, a 6 kHz band (Q 5) and a 12 ms chirp x 0.15; metal: the measured sword, 2 micro-collisions; bone: the stick."""
    material = str(material)
    if material not in BODY_LAW:
        raise ValueError(f"body_hit: material must be {'|'.join(BODY_LAW)}")
    rng = np.random.default_rng(int(seed))
    k = float(np.clip(size, 0.2, 8.0))
    s = body_hit_scale(material, k)
    x = np.zeros(_n(0.8))
    if material == "flesh":
        _put(x, _barrel(0.02 * math.sqrt(k)), 0.0)
        _put(x, _shock(0.010), 0.0, 0.7)
        _put(x, _noiseburst(0.0077, rng), 0.0, 0.3)
        y = np.tanh(0.6 * _bpar8(x, [f * s for f in BODY_SET], q=3.0))
    elif material == "stone":
        _put(x, _shell(0.03897), 0.0)
        _put(x, _noiseburst(0.0077, rng), 0.0)
        y = _bpar8(x, [f * s for f in BODY_SET])
    elif material == "shell":
        x[:_n(0.0002)] = 1.0
        y = 0.7 * amb.bp(_bpar8(x, [f * s for f in SHELL_SET], q=30.0, gain=0.35), 6000.0, 5.0)
        _put(y, _nm(_shell(0.012)) * float(np.max(np.abs(y))), 0.0, 0.15)
    elif material == "metal":
        y = _knock("sy_sword1", rng, s, contact_ms=0.3, micro=2, dur=0.8)
    else:
        y = _knock("sy_stick", rng, s, contact_ms=1.5, dur=0.8)
    return _fin(_nm(y), sr, 0.8)


# ============================================================================== machines
G_ACC = 9.81
# size -> ticks per second, time scale of the mechanism, body scale s, hand-clunk level.
# mantle = clock-all1 (a tick or tock every 250 ms, body 0.09, hand clunk each second).
# grandfather = a seconds pendulum (T = 2 pi sqrt(L/g) = 2 s for L = 0.994 m: one tick per second).
# watch = 5 ticks/s (18000 beats/h), inside a balance wheel's 4-8 per second; time and body scale with the size.
CLOCKS = {"watch": (5.0, 0.5, 0.3, 0.0), "mantle": (4.0, 1.0, 0.09, 1.0), "grandfather": (1.0, 2.5, 0.03, 0.0)}


def pendulum_ticks(length_m: float) -> float:
    """Ticks per second of an escapement on a pendulum of this length: 2 / T, T = 2 pi sqrt(L / g)."""
    return 1.0 / (math.pi * math.sqrt(max(length_m, 1e-3) / G_ACC))


def _fbell(f0: float, decay: float, tau: np.ndarray, rate: int, rng, bright: float = 1.0) -> np.ndarray:
    """Telephone bell: five groups of three partials, each group x 0.333
    x a squared linear envelope of decay_scale * decay. ``tau`` = seconds since the last strike per
    sample (the Pd envelope restarts on every strike; the oscillators run free). Partials above
    0.45 * rate are dropped and the rest rolled off above 3.5 kHz x ``bright``."""
    t = np.arange(len(tau)) / rate
    out = np.zeros(len(tau))
    for g in md.FARNELL_PHONE_BELL["groups"]:
        s = np.zeros(len(tau))
        for r, a in zip(g["ratios"], g["amps"]):
            f = f0 * r
            if f < 0.45 * rate:
                s += a / (1.0 + (f / (3500.0 * max(bright, 0.05))) ** 2) * np.cos(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        out += s * np.clip(1.0 - tau / (g["decay_scale"] * decay), 0.0, 1.0) ** 2
    return out * md.FARNELL_PHONE_BELL["group_gain"]


def _since(n: int, times, rate: int) -> np.ndarray:
    """Seconds since the most recent of ``times`` for each sample (inf before the first)."""
    tau = np.full(n, np.inf)
    for t0 in sorted(times):
        i = int(round(t0 * rate))
        if i < n:
            tau[i:] = np.arange(n - i) / rate
    return tau


@sfx("clock", "Clockwork tick-tock: escapement micro-clicks in a resonant case, from a wristwatch to a grandfather clock, with an optional chime.",
     size=("mantle", "watch|mantle|grandfather"), rate=(0.0, "ticks per second (0 = the size's own: 5, 4, 1)"),
     dur=(4.0, "s"), pendulum=(0.0, "pendulum length in m; > 0 sets the rate from T = 2 pi sqrt(L/g) (0.994 m = one tick per second)"),
     chime=(0, "bell strokes struck at the start (0 = none)"), seed=(0, "variation"))
def clock(sr=DEFAULT_SR, size="mantle", rate=0.0, dur=4.0, pendulum=0.0, chime=0, seed=0):
    """`clocktick` (6543/3245/1356 Hz) fires its micro-sequence at
    0, 9, 12, 21 ms, `clocktick2` (7543/3988/2765 Hz) at 0, 3, 9, 15, 21 ms on alternate ticks; the
    escapement (10 and 8 kHz pings, 3 ms apart) on every half tick; the hand clunk (300 Hz, 120 ms)
    once a second; all x 0.3 into `bodyresonance~`. Mechanism frequencies scale as
    (time scale)^-0.5 between sizes."""
    rng = np.random.default_rng(int(seed))
    r0, k, s, hand = CLOCKS.get(str(size), CLOCKS["mantle"])
    rate = float(np.clip(pendulum_ticks(float(pendulum)) if pendulum > 0 else (float(rate) if rate > 0 else r0), 0.2, 12.0))
    dur = float(np.clip(dur, 0.2, 120.0))
    n = _n(dur)
    fsc = k ** -0.5
    sc = lambda parts: [(f * fsc, q, d * k) for f, q, d in parts]
    nz = amb.hip(rng.uniform(-1, 1, n), 200.0)
    ticks = np.arange(0.01, dur, 1.0 / rate)
    y = np.zeros(n)
    for parts, sel, cnt in ((TICK, ticks[0::2], (0, 9, 12, 21)), (TOCK, ticks[1::2], (0, 3, 9, 15, 21))):
        y += _mclick(nz, sc(parts), [t + c * 1e-3 * k for t in sel for c in cnt], 0.3)
    m = _n(0.011 * k)
    tt, e = np.arange(m) / P, _sqdec(m, 10.0 * k)
    for b in np.arange(0.01, dur, 0.5 / rate):
        for c, f in enumerate((10000, 8000, 10000, 8000, 8000, 10000, 8000, 10000)):
            _put(y, np.sin(2 * np.pi * min(f * fsc, 0.45 * P) * tt) * e, b + c * 0.003 * k, 0.06)
    if hand:
        y += _mclick(nz, sc(HAND), list(np.arange(0.01, dur, 1.0)), 0.9 * hand)
    y = _nm(_body(y * 0.3, s))
    if chime > 0:
        f0, dec, gap = {"watch": (2000.0, 0.8, 0.5), "mantle": (1100.0, 1.6, 0.9), "grandfather": (440.0, 3.0, 1.6)}.get(str(size), (1100.0, 1.6, 0.9))
        times = [0.05 + j * gap for j in range(int(np.clip(chime, 1, 12)))]
        nb = _n(times[-1] + dec * 1.3)
        y = pc.fit(y, max(n, nb))
        y[:nb] += 0.9 * _nm(_fbell(f0, dec, _since(nb, times, P), P, rng))
    return _fin(y, sr, 0.6)


RELAY_BOUNCE = {"on": 5e-3, "off": 7e-3}
RELAY_OPERATE = {"on": 14e-3, "off": 12e-3}
RELAY_RP3SL = dict(mass_g=18.0, vibration_g=20.0, shock_g=100.0, loaded_per_h=960.0, unloaded_per_h=72000.0)


PNOISE_COLOURS = ((3345.0, 2980.0, 4790.0), (4345.0, 2480.0, 4795.0))


def pnoise(n: int, rng, colours=PNOISE_COLOURS[0]) -> np.ndarray:
    """Click noise from three crossed phasors: each phasor's output is added to the next one's phase, so the result is
    noise-like with the same level on every click."""
    t = np.arange(n) / P
    f1, f2, f3 = colours
    a = (f3 * t + rng.uniform()) % 1.0
    b = (f2 * t + a + rng.uniform()) % 1.0
    return 2.0 * ((f1 * t + b) % 1.0) - 1.0


def _switch(kind: str, state: str, rng, n: int, coil: bool = False, noise: str = "band") -> np.ndarray:
    """Event patterns: each click is one mechanical event."""
    nz = rng.uniform(-1, 1, n)
    if noise == "pnoise":
        nz = pnoise(n, rng, PNOISE_COLOURS[state == "off"])
    cl = lambda dms, f, at: _switchclick(nz, 1.0, dms, f, at)
    lo = amb.hip(nz, 300.0)
    if kind == "rocker":            # "two clicks, then a clunk as the actuator hits the stop"
        return cl(20, 5000, 0.0) + cl(20, 4000, 0.008) + _mclick(lo, ENDSTOP, 0.025, 1.0)
    if kind == "push":              # "double click and latch, sounding slightly different on and off"
        if state == "off":
            return cl(20, 5500, 0.0) + _mclick(lo, [(f * 1.15, q, d) for f, q, d in CATCH], 0.015, 1.0)
        return cl(20, 5000, 0.0) + cl(20, 5200, 0.012) + _mclick(lo, CATCH, 0.030, 1.0)
    if kind == "relay":
        st = "off" if state == "off" else "on"
        k, t0 = RELAY_BOUNCE[st] / 5e-3, RELAY_OPERATE[st] if coil else 0.0
        y = cl(20, 5000, t0) + 0.5 * cl(10, 5000, t0 + 0.0025 * k) + 0.3 * cl(8, 5000, t0 + 0.004 * k) + 0.15 * cl(6, 5000, t0 + 0.005 * k)
        y += _shortping(n, t0, 0.02)
        _put(y, _shock(0.006), t0, 0.3 * float(np.max(np.abs(y))))
        return y
    if kind == "slide":             # model D: 0 -> 0.46 over 100 ms, jump to 1, 0 over 50 ms, all ^4: "ssshh-Tunk"
        t = np.arange(n) / P
        env = np.where(t < 0.1, 0.46 * t / 0.1, np.clip(1 - (t - 0.1) / 0.05, 0, 1)) ** 4
        y = (nz if noise == "pnoise" else sum(amb.bp(nz, f, 8.0) for f in PNOISE_COLOURS[0])) * env
        return y + _shortping(n, 0.1, 0.03 * float(np.max(np.abs(y))))
    if kind == "rotary":            # "multiple clicks plus a rotary slide sound"
        y = sum(cl(20, 5000, 0.045 * j) + cl(20, 4000, 0.045 * j + 0.004) for j in range(4))
        return y + 0.05 * amb.bp(nz, 2200.0, 2.0) * np.sin(np.pi * np.minimum(np.arange(n) / _n(0.18), 1.0))
    # toggle: four clicks, plastic 3 and 4 kHz first, metal 5 and 7 kHz "shifted by ten milliseconds"
    return cl(50, 3000, 0.0) + cl(20, 4000, 0.0) + cl(20, 5000, 0.010) + cl(50, 7000, 0.010)


@sfx("switch", "Mechanical switch: every click is one mechanical event, coloured by what it is mounted on.",
     kind=("toggle", "toggle|rocker|push|relay|slide|rotary"), mount=("plastic", "plastic|wood|metal: the panel it sits on"),
     state=("on", "on|off (latching push button sounds different each way; a relay bounces longer when it releases)"),
     coil=(0, "relay: 1 = start when the coil is switched, the contact moves 14 ms (on) / 12 ms (off) later"), seed=(0, "variation"),
     noise=("band", "band: white noise in each click | pnoise: one shared generator of three crossed phasors, the same level every click"))
def switch(sr=DEFAULT_SR, kind="toggle", mount="plastic", state="on", coil=0, seed=0, noise="band"):
    if noise not in ("band", "pnoise"):
        raise ValueError("switch noise must be band or pnoise")
    return _fin(_panel(_switch(str(kind), str(state), np.random.default_rng(int(seed)), _n(0.3), bool(coil), noise), mount), sr, 0.6)


@sfx("relay_chatter", "A relay switched on and off in a row, each operation a contact click with its bounce; the rate is held to what the relay can do (20 Hz unloaded, 960 an hour under load).",
     rate=(5.0, "operations per second"), loaded=(0, "1 = switching a load: at most 960 per hour"), count=(8, "operations"),
     mount=("plastic", "plastic|wood|metal"), seed=(0, "variation"))
def relay_chatter(sr=DEFAULT_SR, rate=5.0, loaded=0, count=8, mount="plastic", seed=0):
    rng = np.random.default_rng(int(seed))
    top = (RELAY_RP3SL["loaded_per_h"] if loaded else RELAY_RP3SL["unloaded_per_h"]) / 3600.0
    gap = 1.0 / float(np.clip(rate, 0.01, top))
    k = int(min(np.clip(count, 1, 200), 60.0 / gap + 1))
    y = np.zeros(_n((k - 1) * gap + 0.3))
    for j in range(k):
        _put(y, _switch("relay", "off" if j % 2 else "on", rng, _n(0.3)), j * gap)
    return _fin(_panel(y, mount), sr, 0.6)


@sfx("button", "Momentary push button: one sprung click with a tiny metallic ping, and a softer release.",
     release=(1, "1 = add the release click"), hold=(0.12, "s the button is held"), mount=("plastic", "plastic|wood|metal"), seed=(0, "variation"))
def button(sr=DEFAULT_SR, release=1, hold=0.12, mount="plastic", seed=0):
    """Model A click (5 kHz, Q 12, 1 ms / 20 ms) + model C short ping."""
    rng = np.random.default_rng(int(seed))
    hold = float(np.clip(hold, 0.02, 2.0))
    n = _n(hold + 0.2)
    nz = rng.uniform(-1, 1, n)
    y = _switchclick(nz, 1, 20, 5000, 0.0) + _shortping(n, 0.0, 0.02)
    if release:
        y += 0.5 * _switchclick(nz, 1, 15, 5600, hold)
    return _fin(_panel(y, mount), sr, 0.55)


def _key(rng, fs: float, up_at: float) -> np.ndarray:
    nz = rng.uniform(-1, 1, _n(up_at + 0.07))
    return (_switchclick(nz, 1, 50, 3000 * fs, 0.0) + _switchclick(nz, 1, 20, 4000 * fs, 0.0)
            + 0.45 * _switchclick(nz, 1, 20, 5000 * fs, up_at))


@sfx("keyboard_typing", "Typing on a computer keyboard: plastic key clicks with bottom-out and release, human timing, the odd space bar.",
     keys=(12, "key presses"), speed=(6.0, "keys per second"), seed=(0, "variation"))
def keyboard_typing(sr=DEFAULT_SR, keys=12, speed=6.0, seed=0):
    """Each key = a "plastic" click pair (3 and 4 kHz) and a release click, in the panel body; +-timing spread, 1 key in 6 a lower space bar."""
    rng = np.random.default_rng(int(seed))
    keys = int(np.clip(keys, 1, 400))
    gaps = rng.lognormal(0.0, 0.35, keys) / max(float(speed), 0.5)
    times = np.concatenate([[0.0], np.cumsum(gaps[:-1])])
    y = np.zeros(_n(times[-1] + 0.3))
    for t in times:
        space = rng.random() < 1 / 6
        _put(y, _key(rng, (0.7 if space else 1.0) * rng.uniform(0.92, 1.08), rng.uniform(0.05, 0.09)), t, rng.uniform(0.6, 1.0) * (1.3 if space else 1.0))
    return _fin(_panel(y, "plastic", gin=3.0), sr, 0.6)


@sfx("mouse_click", "Computer mouse click: the microswitch going down and coming back up.",
     double=(0, "1 = double click"), hold=(0.09, "s between press and release"), seed=(0, "variation"))
def mouse_click(sr=DEFAULT_SR, double=0, hold=0.09, seed=0):
    rng = np.random.default_rng(int(seed))
    hold = float(np.clip(hold, 0.02, 1.0))
    y = np.zeros(_n(2 * hold + 0.4))
    for j in range(2 if double else 1):
        n = _n(hold + 0.08)
        nz = rng.uniform(-1, 1, n)
        _put(y, _switchclick(nz, 1, 20, 5000, 0.0) + _shortping(n, 0.0, 0.015) + 0.6 * _switchclick(nz, 1, 15, 5600, hold), j * (hold + 0.11))
    return _fin(_body(y, 0.2), sr, 0.5)


@sfx("camera_shutter", "SLR camera shutter: mirror slap, shutter opening and closing after the exposure, optional film wind.",
     exposure=(0.0167, "s the shutter stays open (1/60 = 0.0167)"), wind=(0.0, "s of motor wind afterwards (0 = none)"), seed=(0, "variation"))
def camera_shutter(sr=DEFAULT_SR, exposure=0.0167, wind=0.0, seed=0):
    """Click-factory clusters as the mechanical events in a small case (`bodyresonance~`)."""
    rng = np.random.default_rng(int(seed))
    ex = float(np.clip(exposure, 0.001, 4.0))
    wind = float(np.clip(wind, 0.0, 3.0))
    n = _n(0.07 + ex + 0.25 + wind + 0.1)
    lo = amb.hip(rng.uniform(-1, 1, n), 300.0)
    y = (_mclick(lo, ENDSTOP, 0.0, 1.0) + _mclick(lo, CATCH, 0.0, 0.6) + _mclick(lo, RATCHET, 0.03, 0.8)
         + _mclick(lo, RATCHET, 0.03 + ex, 0.8) + _mclick(lo, ENDSTOP, 0.07 + ex, 0.8))
    y = _nm(_body(y, 0.12))
    if wind > 0:
        _put(y, _nm(_motor(240.0 * _move(_n(wind)), rng)), 0.2 + ex, 0.35)
    return _fin(y, sr, 0.6)


def _move(n: int) -> np.ndarray:
    """Motor speed envelope: attack 1 - (1 - x)^6, linear decay."""
    x = np.arange(n) / max(n, 1)
    return np.minimum(1.0 - (1.0 - np.minimum(x / 0.3, 1.0)) ** 6, np.clip((1.0 - x) / 0.2, 0.0, 1.0))


MOTOR_FM_CARRIER, MOTOR_FM_HIP = 178.0, 180.0


MOTOR_CASE_LP = 4000.0


def _motor(speed: np.ndarray, rng, housing=(1200.0, 2800.0), body: str = "resonators", case: bool = False) -> np.ndarray:
    """Electric motor: rotor = phasor^4 x (4 kHz band noise + DC), unidirectional
    brush spikes; stator = 1 / (x^2 + 1) of 2 cos (four times the frequency at a quarter width);
    then the housing: band-pass resonators, or ``body="fm"``: the motor signal frequency-modulates a 178 Hz
    carrier, through two 180 Hz high-passes; ``case``: shut in its case, a 4 kHz low-pass."""
    if case:
        return F.lowpass(_motor(speed, rng, housing, body), MOTOR_CASE_LP, P)
    # The book does number its FM body: not done here, resonances instead.
    n = len(speed)
    ph = np.cumsum(speed) / P
    rotor = (ph % 1.0) ** 4 * (0.5 * amb.bp(rng.uniform(-1, 1, n), 4000.0, 1.0) + 0.5)
    x = amb.hip(rotor + 0.6 / ((2.0 * np.cos(2 * np.pi * ph)) ** 2 + 1.0), 30.0) * np.sqrt(speed / (float(speed.max()) + 1e-9))
    if body == "fm":
        car = np.sin(2 * np.pi * np.cumsum(MOTOR_FM_CARRIER * (1.0 + x / (np.max(np.abs(x)) + 1e-9))) / P)
        return x + amb.hip(amb.hip(car, MOTOR_FM_HIP), MOTOR_FM_HIP)
    return x + sum(amb.bp(x, f, 8.0) for f in housing)


@sfx("servo", "Servo / stepper moves: a small motor whining up to speed and stopping, in a sequence of moves.",
     moves=(2, "moves"), dur=(0.35, "s per move"), speed=(180.0, "rotor Hz at full speed (pitch)"), gap=(0.12, "s between moves"), seed=(0, "variation"),
     case=(0, "1 = the motor shut in its case: a 4 kHz low-pass"))
def servo(sr=DEFAULT_SR, moves=2, dur=0.35, speed=180.0, gap=0.12, seed=0, case=0):
    """Position-control motor: "sharper start and stop", damped, no overspin: `machines.servo_env`, not `_move`."""
    from .machines import servo_env
    rng = np.random.default_rng(int(seed))
    moves, d = int(np.clip(moves, 1, 60)), float(np.clip(dur, 0.04, 5.0))
    y = np.zeros(_n(moves * (d + gap) + 0.1))
    for j in range(moves):
        dj = d * rng.uniform(0.6, 1.1) if j else d
        _put(y, _motor(float(np.clip(speed, 20, 1500)) * rng.uniform(0.85, 1.1) * servo_env(_n(dj)), rng, case=bool(case)), j * (d + float(gap)))
    return _fin(y, sr, 0.55)


@sfx("printer", "Desktop printer: the carriage stepping back and forth with the head buzzing, and a paper feed after each line.",
     lines=(3, "printed lines"), speed=(1.0, "0.5 slow .. 2 fast"), seed=(0, "variation"))
def printer(sr=DEFAULT_SR, lines=3, speed=1.0, seed=0):
    """Carriage and feed = the motor with its speed envelope; the head is a pulse
    train into a small resonance; the feed ends on a ratchet click."""
    rng = np.random.default_rng(int(seed))
    k = 1.0 / float(np.clip(speed, 0.25, 4.0))
    lines = int(np.clip(lines, 1, 60))
    per = (0.5 + 0.16) * k
    y = np.zeros(_n(lines * per + 0.2))
    for j in range(lines):
        m = _n(0.5 * k)
        mv = _move(m)
        head = amb.bp(_train(600.0 * (mv > 0.6), rng, 0.3), 2200.0, 4.0)
        _put(y, _nm(_motor((150.0 if j % 2 == 0 else 165.0) * mv, rng)) + 0.5 * _nm(head), j * per)
        _put(y, _nm(_motor(320.0 * _move(_n(0.09 * k)), rng)), j * per + 0.52 * k, 0.8)
        _put(y, _nm(_mclick(amb.hip(rng.uniform(-1, 1, _n(0.05)), 300.0), RATCHET)), j * per + 0.61 * k, 0.5)
    return _fin(_body(y, 0.05), sr, 0.55)


@sfx("drill", "Electric drill: motor spin-up, load-dependent speed, and the bit cutting.",
     dur=(2.0, "s"), rpm=(9000.0, "motor RPM unloaded"), load=(0.4, "0..1 pressure on the bit: slower, rougher"),
     material=("wood", "none|wood|metal: what the bit cuts"), startup=(0.25, "s to reach speed"), seed=(0, "variation"))
def drill(sr=DEFAULT_SR, dur=2.0, rpm=9000.0, load=0.4, material="wood", startup=0.25, seed=0):
    """genny.physics.ElectricMotor (rotor, commutation, bearings, housing) plus scrape texture
     for the bit."""
    from .physics import ElectricMotor
    rng = np.random.default_rng(int(seed))
    dur, ld = float(np.clip(dur, 0.2, 30.0)), _c01(load)
    r = float(np.clip(rpm, 600, 30000)) * (1.0 - 0.2 * ld)
    y = _nm(np.asarray(ElectricMotor(rpm=r).render(dur, load=ld, startup=float(np.clip(startup, 0.0, dur)), sr=P, seed=int(seed)), float))
    if material != "none" and ld > 0:
        n = len(y)
        t = np.arange(n) / P
        up = np.minimum(1.0, t / max(float(startup), 1e-3))
        fc, q = (4200.0, 3.0) if material == "metal" else (1800.0, 1.0)
        cut = amb.bp(fractal_noise(n, rng), fc, q) * (0.6 + 0.4 * np.sin(2 * np.pi * r / 60.0 / 8.0 * t)) * up
        y = y + 0.9 * ld * _nm(cut)
    return _fin(y, sr, 0.6, tail=0.03)


@sfx("saw", "Hand saw cutting wood: push and pull strokes, the teeth passing at a rate that follows the stroke speed.",
     strokes=(4, "push-pull cycles"), rate=(1.6, "cycles per second"), length=(0.35, "m of blade travel per stroke"), tpi=(8.0, "teeth per inch"),
     seed=(0, "variation"))
def saw(sr=DEFAULT_SR, strokes=4, rate=1.6, length=0.35, tpi=8.0, seed=0):
    """Tooth-passing frequency = v / pitch (pitch = 25.4 mm / tpi; peak speed pi * length * rate), an
    impulse train plus fractal scrape noise x sqrt(v), driving the measured wooden
    body."""
    rng = np.random.default_rng(int(seed))
    rate = float(np.clip(rate, 0.3, 6.0))
    n = _n(int(np.clip(strokes, 1, 60)) / rate)
    t = np.arange(n) / P
    v = np.pi * float(length) * rate * np.abs(np.sin(2 * np.pi * rate * t))
    push = np.where(np.sin(2 * np.pi * rate * t) > 0, 1.0, 0.55)
    teeth = np.clip(v / (25.4e-3 / max(float(tpi), 1.0)), 0, 6000.0)
    force = (_train(teeth, rng, 0.3) * 6.0 + _scrape(v, rng, lam=3e-4)) * np.sqrt(v / (v.max() + 1e-9)) * push
    force = np.pad(force, (0, _n(0.25)))
    return _fin(_nm(_drive("body_squareMid", force)) + 0.35 * _nm(amb.bp(force, 2600.0, 0.8)), sr, 0.65)


def nail_hz(start: float, sink: float, k: int, strikes: int) -> float:
    """Resonance of the nail's free length after k blows: a clamped bar, f ~ 1/L^2,
    L = 1 - sink k / strikes of the starting length."""
    return start / (1.0 - sink * k / max(strikes, 1)) ** 2


@sfx("hammering", "Hammering a nail into wood: the nail's ring climbs with every blow as it sinks.",
     strikes=(12, "blows"), pace=(2.2, "blows per second"), start_freq=(200.0, "Hz of the nail before the first blow"),
     sink=(0.6, "0..0.9 fraction of the nail driven in by the last blow"), seed=(0, "variation"))
def hammering(sr=DEFAULT_SR, strikes=12, pace=2.2, start_freq=200.0, sink=0.6, seed=0):
    """The nail resonance sweeps upward from ~200 Hz as the nail shortens (12
    strikes)". Each blow is one 0.4 ms contact shared by the nail (clamped-bar modes f and 6.267 f),
    a damped steel hammer head (wok.sy x2) and the board (Faust big square body)."""
    rng = np.random.default_rng(int(seed))
    strikes = int(np.clip(strikes, 1, 100))
    sink = _c01(sink, 0.0, 0.9)
    gap = 1.0 / max(float(pace), 0.3)
    y = np.zeros(_n(strikes * gap + 0.6))
    for k in range(strikes):
        L = 1.0 - sink * k / strikes
        f = min(nail_hz(float(start_freq), sink, k, strikes), 0.4 * P)
        v = rng.uniform(0.75, 1.0)
        force = np.zeros(_n(0.4))
        p = modal.bang(0.4, v)
        force[:len(p)] = p
        nail = modal.render_modes(force, modal.Modes([f, min(6.267 * f, 0.45 * P)], [30.0 / L, 90.0 / L], [1.0, 0.3], "vdd"))
        t = k * gap + 0.08 * gap * float(np.clip(rng.standard_normal(), -2, 2)) * (k > 0)
        _put(y, _nm(nail), t, v)
        _put(y, _nm(_drive("sy_wok", force, 2.0, 12.0)), t, 0.3 * v)
        _put(y, _nm(_drive("body_squareBig", force)), t, (0.35 + 0.5 * (1 - L)) * v)
    return _fin(y, sr, 0.8)


@sfx("ratchet_wrench", "Socket wrench: clicking back on the return swing, quiet on the drive stroke.",
     strokes=(3, "swings"), rate=(1.6, "swings per second"), clicks=(9, "pawl clicks per return swing"), size=(1.0, "tool size (bigger = lower)"),
     seed=(0, "variation"))
def ratchet_wrench(sr=DEFAULT_SR, strokes=3, rate=1.6, clicks=9, size=1.0, seed=0):
    """A pawl-click train (half-sine swing speed) into the STK Wrench resonances (3200 Hz r 0.99,
    8000 Hz r 0.992)."""
    rng = np.random.default_rng(int(seed))
    rate = float(np.clip(rate, 0.3, 8.0))
    n = _n(int(np.clip(strokes, 1, 60)) / rate)
    t = np.arange(n) / P
    ph = (t * rate) % 1.0
    back = ph >= 0.5
    cl = np.where(back, np.pi * float(clicks) * 2 * rate * np.sin(np.pi * (ph - 0.5) * 2), 0.0)
    p = particles.STK_PRESETS["Wrench"]
    y = _resons(np.pad(_train(cl, rng, 0.15), (0, _n(0.3))), np.array(p["freqs"]) / max(float(size), 0.05), p["radii"])
    return _fin(y, sr, 0.6)


@sfx("piston", "Pneumatic piston strokes: a hiss of air, the rod travelling, and the end-stop clunk.",
     cycles=(3, "out-and-back cycles"), rate=(1.2, "cycles per second, up to 12.5 (25 strokes/s, the top of `machines.jackhammer`)"), pressure=(0.6, "0..1 air pressure: louder, brighter hiss"), seed=(0, "variation"))
def piston(sr=DEFAULT_SR, cycles=3, rate=1.2, pressure=0.6, seed=0):
    """Hiss = high-passed noise through a band that opens with pressure; the stops are the end-stop clunk cluster."""
    rng = np.random.default_rng(int(seed))
    rate, pr = float(np.clip(rate, 0.2, 12.5)), _c01(pressure)
    n = _n(int(np.clip(cycles, 1, 60)) / rate + 0.2)
    t = np.arange(n) / P
    ph = (t * rate * 2.0) % 1.0                                  # two strokes per cycle
    env = np.minimum(ph / 0.03, 1.0) * np.exp(-ph / 0.22) * np.where((t * rate * 2.0).astype(int) % 2 == 0, 1.0, 0.6)
    hiss = amb.bp(amb.hip(rng.standard_normal(n), 1500.0), 2500.0 + 2500.0 * pr, 0.8) * env
    stops = [(j + 0.8) / (rate * 2.0) for j in range(int(n / P * rate * 2.0))]
    clunk = _mclick(amb.hip(rng.uniform(-1, 1, n), 300.0), ENDSTOP, stops, 1.0)
    return _fin(_nm(hiss) * (0.4 + 0.6 * pr) + 0.8 * _nm(_body(clunk, 0.04)), sr, 0.65)


@sfx("steam_engine", "Steam locomotive: exhaust chuffs, four to a wheel turn, with the valve gear clanking, speeding up or slowing down.",
     dur=(4.0, "s"), speed=(3.0, "chuffs per second at the start"), accel=(0.0, "chuffs per second gained per second (negative = slowing)"),
     cylinders=(2, "double-acting cylinders: 2 x cylinders chuffs per wheel turn"), seed=(0, "variation"))
def steam_engine(sr=DEFAULT_SR, dur=4.0, speed=3.0, accel=0.0, cylinders=2, seed=0):
    """Two double-acting cylinders quartered give four exhaust beats per driver revolution.
    Chuff = noise through blast-pipe bands 350 / 900 Hz that
    shorten with speed; first beat of each revolution accented; rod clank each half revolution."""
    rng = np.random.default_rng(int(seed))
    dur = float(np.clip(dur, 0.5, 60.0))
    n = _n(dur)
    t = np.arange(n) / P
    rate = np.clip(float(speed) + float(accel) * t, 0.3, 30.0)
    ph = np.cumsum(rate) / P
    beat = np.floor(ph).astype(int)
    per = 2 * int(np.clip(cylinders, 1, 4))
    frac = ph - beat
    env = np.minimum(frac / 0.04, 1.0) * np.exp(-frac / 0.3) * np.array([1.0, 0.75, 0.85, 0.7, 0.9, 0.72, 0.8, 0.7])[beat % per]
    nz = rng.standard_normal(n)
    chuff = (amb.bp(nz, 350.0, 1.2) + 0.6 * amb.bp(nz, 900.0, 1.0)) * env
    idx = np.nonzero(np.diff(beat) > 0)[0]
    clank = _mclick(amb.hip(rng.uniform(-1, 1, n), 300.0), ENDSTOP, [i / P for i in idx[::per // 2]], 1.0)
    hiss = amb.hip(nz, 4000.0) * 0.02
    return _fin(_nm(chuff) + 0.25 * _nm(_body(clank, 0.03)) + hiss, sr, 0.7, tail=0.05)


@sfx("train", "Riding a train: wheels clacking over rail joints in the rhythm set by the bogie spacing and the speed, with an optional horn.",
     dur=(6.0, "s"), speed=(60.0, "km/h"), rail_len=(25.0, "m between rail joints"), wheelbase=(2.5, "m between the two axles of a bogie"),
     bogie_spacing=(16.0, "m between the bogies of a car"), horn=(0.0, "s of horn at the start (0 = none)"),
     creep=(0.0, "0 = none; wheel creep, as a ratio (saturates near 0.002): the contact's tangential force, heard only below V / 2a"),
     corrugation=(0.0, "0 = smooth rail; corrugation amplitude / static compression: a line at V / wavelength, the wheel leaves the rail above 0.5"),
     wavelength=(0.05, "m of the corrugation"), seed=(0, "variation"))
def train(sr=DEFAULT_SR, dur=6.0, speed=60.0, rail_len=25.0, wheelbase=2.5, bogie_spacing=16.0, horn=0.0, creep=0.0, corrugation=0.0, wavelength=0.05, seed=0):
    """Each axle clacks when it crosses a joint: t = (joint position - axle offset) / v. The two
    clacks of a bogie are wheelbase / v apart and the pattern repeats every rail_len / v.
    Clack = shockwave thump + end-stop clunk through a large body.
    Horn: K5LA, 311, 370, 415, 494, 622 Hz. Creep and corrugation: `rail`."""
    rng = np.random.default_rng(int(seed))
    dur = float(np.clip(dur, 0.5, 120.0))
    v = float(np.clip(speed, 3.0, 400.0)) / 3.6
    n = _n(dur)
    wb, bs, L = max(float(wheelbase), 0.3), max(float(bogie_spacing), 1.0), max(float(rail_len), 2.0)
    axles = ((0.0, 1.0), (wb, 0.9), (bs, 0.45), (bs + wb, 0.4), (-6.0, 0.3), (-6.0 - wb, 0.28))   # own bogie, far bogie, next car
    clack = _nm(_shock(0.012)) + 0.6 * _nm(pc.fit(_mclick(amb.hip(rng.uniform(-1, 1, _n(0.1)), 300.0), ENDSTOP), len(_shock(0.012))))
    y = np.zeros(n)
    for j in range(-2, int(dur * v / L) + 3):
        for off, g in axles:
            t = (j * L + off) / v + 0.2
            if 0 <= t < dur:
                _put(y, clack, t, g * rng.uniform(0.85, 1.0))
    y = _nm(_body(y, 0.02))
    roar = amb.lop(amb.lop(rng.standard_normal(n), 60.0 + 6.0 * v), 60.0 + 6.0 * v)
    y = y + 0.3 * _nm(roar) * min(1.0, v / 20.0)
    if creep > 0 or corrugation > 0:
        from . import rail
        if creep > 0:
            y = y + 3.0 * rail.creep_noise(float(creep), v, n, rng)
        if corrugation > 0:
            y = y + 0.3 * rail.corrugation_noise(v * 3.6, float(corrugation), float(wavelength), n)
    if horn > 0:
        m = min(n, _n(float(horn)))
        tt = np.arange(m) / P
        h = sum(np.tanh(2.0 * np.sin(2 * np.pi * f * tt)) for f in (311.0, 370.0, 415.0, 494.0, 622.0))
        y[:m] += 0.5 * _nm(amb.lop(h, 2500.0)) * np.minimum(1, tt / 0.05) * np.minimum(1, (m / P - tt) / 0.08)
    return _fin(y, sr, 0.75, tail=0.05)


@sfx("sonar", "Active sonar ping with its echo coming back from a target.",
     freq=(1500, "Hz or note name"), range_m=(750.0, "m to the target: the echo returns 2 x range / c later (0 = no echo)"),
     c=(1500.0, "sound speed m/s (sea water ~1500)"), ring=(0.6, "s decay of the ping in the water"), seed=(0, "variation"))
def sonar(sr=DEFAULT_SR, freq=1500, range_m=750.0, c=1500.0, ring=0.6, seed=0):
    """Two-way travel time 2 R / c."""
    rng = np.random.default_rng(int(seed))
    f = float(np.clip(to_hz(freq), 100.0, 0.4 * sr))
    delay = 2.0 * max(float(range_m), 0.0) / max(float(c), 100.0)
    ring = float(np.clip(ring, 0.05, 5.0))
    n = int(sr * (delay + 3.0 * ring + 0.1))
    t = np.arange(n) / sr
    ping = np.sin(2 * np.pi * f * t) * np.minimum(1, t / 0.002) * np.exp(-t / (ring / 3.0))
    rev = F.bandpass(rng.standard_normal(n), f, sr, 12.0) * np.exp(-t / ring) * np.minimum(1, t / 0.05)
    y = ping + 0.25 * rev / (np.max(np.abs(rev)) + 1e-12)
    if range_m > 0:
        d = int(delay * sr)
        y[d:] += 0.35 * F.lowpass(ping, 2.0 * f, sr)[:n - d]
    return _fin(y, sr, 0.7, src=sr)


# ============================================================================== signals
DTMF_ROWS, DTMF_COLS = (697.0, 770.0, 852.0, 941.0), (1209.0, 1336.0, 1477.0, 1633.0)
DTMF_KEYS = "123A456B789C*0#D"


def dtmf_pair(key: str) -> tuple[float, float]:
    """(row Hz, column Hz) of a keypad key."""
    i = DTMF_KEYS.index(str(key).upper())
    return DTMF_ROWS[i // 4], DTMF_COLS[i % 4]


def _gate(n: int, sr: int, ramp: float = 0.003) -> np.ndarray:
    k = max(1, min(n // 2, int(ramp * sr)))
    g = np.ones(n)
    g[:k] = np.linspace(0, 1, k, endpoint=False)
    g[n - k:] = np.linspace(1, 0, k)
    return g


@sfx("dtmf", "Telephone keypad tones (DTMF): each key is one row tone plus one column tone.",
     digits=("5551234", "keys to dial: 0-9 * # A-D (anything else is a pause of one tone + gap)"), tone_ms=(80.0, "ms per tone (a real exchange asks for >= 50)"),
     gap_ms=(80.0, "ms of silence between tones (a real exchange asks for 45 to 3000)"), twist=(0.0, "dB the column (high) tone is above the row tone"))
def dtmf(sr=DEFAULT_SR, digits="5551234", tone_ms=80.0, gap_ms=80.0, twist=0.0):
    """Rows 697, 770, 852, 941 Hz x columns 1209, 1336, 1477, 1633 Hz, exact sines. Tone >= 50 ms, interval 45 ms to 3 s. Not enforced: a shorter tone is a choice."""
    nt, ng = max(8, int(sr * float(tone_ms) * 1e-3)), int(sr * max(float(gap_ms), 0.0) * 1e-3)
    t = np.arange(nt) / sr
    hi = 10 ** (float(np.clip(twist, -6, 6)) / 20)
    parts = []
    for ch in str(digits):
        if ch.upper() in DTMF_KEYS:
            lo_f, hi_f = dtmf_pair(ch)
            parts.append((np.sin(2 * np.pi * lo_f * t) + hi * np.sin(2 * np.pi * hi_f * t)) * _gate(nt, sr))
        else:
            parts.append(np.zeros(nt))
        parts.append(np.zeros(ng))
    if not any(np.any(p) for p in parts):
        raise ValueError("dtmf: `digits` has no key from 0-9 * # A-D")
    return _fin(np.concatenate(parts), sr, 0.6, src=sr, fade=0.0005)


# (region, kind) -> (tone Hz, cadence seconds on, off, on, off ... ; () = continuous).
# Tone plans: North America, the UK, continental Europe (425 Hz).
PHONE_TONES = {
    ("us", "dial"): ((350.0, 440.0), ()), ("us", "busy"): ((480.0, 620.0), (0.5, 0.5)), ("us", "ringback"): ((440.0, 480.0), (2.0, 4.0)),
    ("uk", "dial"): ((350.0, 450.0), ()), ("uk", "busy"): ((400.0,), (0.375, 0.375)), ("uk", "ringback"): ((400.0, 450.0), (0.4, 0.2, 0.4, 2.0)),
    ("eu", "dial"): ((425.0,), ()), ("eu", "busy"): ((425.0,), (0.5, 0.5)), ("eu", "ringback"): ((425.0,), (1.0, 4.0)),
}


def phone_line(x: np.ndarray) -> np.ndarray:
    """Line and handset at the kernel rate: clip +-0.9, band-pass 2 kHz Q 12, x 1.2, band-pass 400 Hz Q 7, two
    high-passes at 90 Hz."""
    return amb.hip(amb.hip(amb.bp(1.2 * amb.bp(np.clip(x, -0.9, 0.9), 2000.0, 12.0), 400.0, 7.0), 90.0), 90.0)


@sfx("pulse_dial", "Rotary phone pulse dialling heard down the line: the exchange current switched on and off, one 40 ms pulse every 100 ms per count of the digit.",
     digit=(7, "1..9, or 0 = ten pulses"), line=(True, "through the telephone line and handset"))
def pulse_dial(sr=DEFAULT_SR, digit=7, line=True):
    k = int(digit) % 10 or 10
    y = np.zeros(_n(0.1 * k + 0.2))
    for i in range(k):
        y[_n(0.1 * i):_n(0.1 * i + 0.04)] = 0.8
    return _fin(phone_line(y) if line else y, sr, 0.6, fade=0.0005)


@sfx("phone_tone", "Telephone network tones: dial tone, busy signal or ringback, with each region's frequencies and cadence.",
     kind=("dial", "dial|busy|ringback"), region=("us", "us|uk|eu"), dur=(0.0, "s (0 = 2 s of dial tone, two busy cycles, or one ringback cycle)"),
     line=(False, "through the telephone line and handset (clipped, 2 kHz and 400 Hz band-passes)"))
def phone_tone(sr=DEFAULT_SR, kind="dial", region="us", dur=0.0, line=False):
    key = (str(region).lower(), str(kind).lower())
    if key not in PHONE_TONES:
        raise ValueError(f"phone_tone: unknown region/kind {key}; regions us|uk|eu, kinds dial|busy|ringback")
    freqs, cad = PHONE_TONES[key]
    d = float(dur) if dur > 0 else (2.0 if not cad else sum(cad) * (2 if sum(cad) < 2 else 1))
    n = int(sr * min(d, 300.0))
    t = np.arange(n) / sr
    y = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    if cad:
        env = np.zeros(n)
        t0, j = 0.0, 0
        while t0 < d:
            if j % 2 == 0:
                a, b = int(t0 * sr), min(n, int((t0 + cad[j % len(cad)]) * sr))
                env[a:b] = _gate(b - a, sr)
            t0 += cad[j % len(cad)]
            j += 1
        y = y * env
    if line:
        return _fin(phone_line(_fin(0.125 * y, P, None, src=sr, fade=0.003)), sr, 0.5, fade=0.003)
    return _fin(y, sr, 0.5, src=sr, fade=0.003)


@numba.njit(cache=True)
def _casing_k(x, da, db, fs):
    """Bakelite case: in -> delay A and B; A read at 0.77 ms -> bp 1243, 287, 431 Hz
    (Q 12) -> clip +-0.3 -> into B; B read at 0.88 ms -> out, and x 0.7 back into A."""
    n = x.shape[0]
    size = max(da, db) + 2
    ba = np.zeros(size)
    bb = np.zeros(size)
    fr = np.array([1243.0, 287.0, 431.0])
    c0 = np.zeros(3)
    c1 = np.zeros(3)
    c2 = np.zeros(3)
    for j in range(3):
        c0[j], c1[j], c2[j] = _bpc(fr[j], 12.0, fs)
    x1 = 0.0
    x2 = 0.0
    y1 = np.zeros(3)
    y2 = np.zeros(3)
    out = np.zeros(n)
    w = 0
    for i in range(n):
        ra = ba[(w - da) % size]
        rb = bb[(w - db) % size]
        s = 0.0
        for j in range(3):
            v = c0[j] * ra - c0[j] * x2 - c1[j] * y1[j] - c2[j] * y2[j]
            y2[j] = y1[j]
            y1[j] = v
            s += v
        x2 = x1
        x1 = ra
        s = min(0.3, max(-0.3, s))
        ba[w] = x[i] + 0.7 * rb
        bb[w] = x[i] + s
        w = (w + 1) % size
        out[i] = rb
    return out


@sfx("phone_bell", "Old electromechanical telephone ringer: a hammer rattling between two slightly detuned bells inside a Bakelite case.",
     freq=(650, "Hz or note name of the first bell"), detune=(3.0, "Hz the second bell is higher (beating)"), decay=(2.0, "s base decay of a bell"),
     rate=(28.6, "hammer strikes per second (35 ms apart in the patch)"), bursts=(2, "rings"), ring=(0.875, "s per ring (25 strikes)"),
     gap=(0.325, "s between rings"), bright=(1.0, "0.3 dull .. 3 all hammer partials"), seed=(0, "variation"))
def phone_bell(sr=DEFAULT_SR, freq=650, detune=3.0, decay=2.0, rate=28.6, bursts=2, ring=0.875, gap=0.325, bright=1.0, seed=0):
    """Bells at 650 and 653 Hz, decay 2000 ms, struck
    alternately every 35 ms for 25 strikes, bursts starting 1200 ms apart; striker = 10 ms of noise
    with a quartic decay x 0.1; dry plus the case resonator. Envelopes restart on each strike."""
    rng = np.random.default_rng(int(seed))
    f0 = float(np.clip(to_hz(freq), 60.0, 4000.0))
    rate, ring = float(np.clip(rate, 2.0, 60.0)), float(np.clip(ring, 0.05, 10.0))
    dec = float(np.clip(decay, 0.05, 10.0))
    times = [b * (ring + max(float(gap), 0.0)) + k / rate for b in range(int(np.clip(bursts, 1, 20))) for k in range(max(1, int(round(ring * rate))))]
    n = _n(times[-1] + 1.2 * dec + 0.1)
    y = _fbell(f0, dec, _since(n, times[0::2], P), P, rng, float(bright)) + _fbell(f0 + float(detune), dec, _since(n, times[1::2], P), P, rng, float(bright))
    ns = _n(0.010)
    for t0 in times:
        _put(y, rng.uniform(-1, 1, ns) * _line(ns) ** 4, t0, 0.1 * md.FARNELL_PHONE_BELL["group_gain"] * 0.3)
    return _fin(y + _casing_k(y, _n(0.00077), _n(0.00088), float(P)), sr, 0.7)


def _echo(y: np.ndarray, amount: float) -> np.ndarray:
    """Street echoes: delays 165, 121, 33 ms summed, recirculated x 0.1."""
    return y + amount * 4.0 * (amb.outdoor_echoes(y) - 0.2 * y) if amount > 0 else y


# French decree of 1987 through Bruitparif 2023: Hz, cycles a minute (the middle of its 50-60 and 25-30). Those notes also ask for two harmonics
# in 1-4 kHz above the fundamental: the horn band below gives that. On the street they run 13 s (median), 90-110 dB(A) at 2 m.
_FR_SIRENS = {"police_fr": ((435, 580), 55.0), "gendarmerie": ((435, 732), 55.0), "pompiers": ((435, 488), 27.5), "samu": ((435, 651), 55.0), "ambulance_fr": ((420, 516, 420), 55.0)}


def _rc_wave(u: np.ndarray) -> np.ndarray:
    """Astable-multivibrator shape, -1..1: capacitor charge e^x up over the first half-cycle, its mirror down."""
    a = 2.0 * (u % 1.0)
    r = (math.e - np.exp(1.0 - np.minimum(a, 1.0))) / (math.e - 1.0)
    f = (math.e - np.exp(1.0 - (np.maximum(a, 1.0) - 1.0))) / (math.e - 1.0)
    return 2.0 * np.where(a < 1.0, r, 1.0 - f) - 1.0


@sfx("siren_wail", "Emergency siren heard in a street: police wail, two-tone ambulance or a wind-up air-raid siren, with building echoes.",
     kind=("police", "police|police_rc (both oscillators RC-shaped, overdriven diaphragm)|ambulance|air_raid; French two-tones: police_fr|gendarmerie|pompiers|samu|ambulance_fr"), dur=(5.0, "s"), rate=(0.0, "cycles per second (0 = the kind's own)"),
     echo=(0.5, "0..1 building echoes (delays 165, 121, 33 ms)"), bright=(1.0, "0.5 distant .. 2 close"), seed=(0, "variation"))
def siren_wail(sr=DEFAULT_SR, kind="police", dur=5.0, rate=0.0, echo=0.5, bright=1.0, seed=0):
    """The horn is a rich pulse through a horn band, then the three-delay echo network: much of a siren's
    character comes from the street. Police: the LFO sweeps 300-1100 Hz at 0.3 Hz (sweeps run 0.1-3 Hz),
    rising fast (35 % of the cycle); ambulance two-tone 450 / 600 Hz (a fourth) at 0.9 Hz; air raid: a rotor spinning up to 465 Hz with a second 5:6 port ring, 4 s on / 4 s coasting."""
    dur = float(np.clip(dur, 0.3, 120.0))
    n = _n(dur)
    t = np.arange(n) / P
    if kind in _FR_SIRENS:
        tones, r = _FR_SIRENS[kind]
        r = float(rate) if rate > 0 else r / 60.0
        if kind == "ambulance_fr":                               # 420 / 516 / 420, then 1.5 s of silence
            u = (t % (1.0 / r + 1.5)) * r
            f = np.choose(np.minimum((u * 3.0).astype(int), 2), tones)
        else:
            u = (t * r) % 1.0
            f = np.where(u < 0.5, float(tones[0]), float(tones[1]))
        x = O.saw(amb.lop(np.asarray(f, float), 60.0), n, P)
        if kind == "ambulance_fr":
            x = x * (u < 1.0)
    elif kind == "ambulance":
        r = float(rate) if rate > 0 else 0.9
        f = np.where((t * r) % 1.0 < 0.5, 450.0, 600.0)
        f = amb.lop(f, 60.0)
        x = O.saw(f, n, P)
    elif kind == "air_raid":
        r = float(rate) if rate > 0 else 0.125
        on = ((t * r) % 1.0 < 0.5).astype(float)
        f = np.maximum(465.0 * amb.lop(on, 0.11), 20.0)          # motor inertia: ~1.4 s time constant
        x = np.tanh(3.0 * np.sin(2 * np.pi * np.cumsum(f) / P)) + 0.7 * np.tanh(3.0 * np.sin(2 * np.pi * np.cumsum(f * 1.2) / P))
        x = x * np.clip(f / 465.0, 0, 1) ** 2
    elif kind == "police_rc":
        r = float(rate) if rate > 0 else 0.3
        f = 300.0 + 800.0 * 0.5 * (1.0 + _rc_wave(t * r))
        x = amb.bp(np.clip(_rc_wave(np.cumsum(f) / P), -0.2, 0.2), 1500.0, 4.0)
        return _fin(_echo(np.pad(x, (0, _n(0.4))), _c01(echo)), sr, 0.6, tail=0.05)
    else:
        r = float(rate) if rate > 0 else 0.3
        u = (t * r) % 1.0
        f = 300.0 * (1100.0 / 300.0) ** np.where(u < 0.35, u / 0.35, 1.0 - (u - 0.35) / 0.65)
        x = O.saw(f, n, P)
    b = float(np.clip(bright, 0.2, 4.0))
    x = pc.lowpass(amb.bp(x, 1300.0, 1.5) + 0.3 * x, 3200.0 * b)
    return _fin(_echo(np.pad(x, (0, _n(0.4))), _c01(echo)), sr, 0.6, tail=0.05)


@sfx("car_horn", "Car horn: two diaphragm horns a third apart.",
     dur=(0.7, "s"), freq=(335, "Hz or note name of the low horn"), freq2=(400, "Hz or note name of the high horn"), bright=(1.0, "0.5 muffled .. 2 harsh"))
def car_horn(sr=DEFAULT_SR, dur=0.7, freq=335, freq2=400, bright=1.0):
    """Pair = Hella M26 disc horns, low tone 335 Hz / high tone 400 Hz (hella.co.nz M26 series page).
    2 kHz horn band: the 1800-3550 Hz band must dominate every component above 3550 Hz. Clipped buzz of a vibrating diaphragm."""
    n = int(sr * float(np.clip(dur, 0.05, 20.0)))
    t = np.arange(n) / sr
    x = sum(np.tanh(6.0 * np.sin(2 * np.pi * to_hz(f) * t + 0.3 * np.sin(2 * np.pi * 37.0 * t))) for f in (freq, freq2))
    y = F.lowpass(F.bandpass(x, 2000.0, sr, 1.2) + 0.4 * F.lowpass(x, 1200.0, sr), 3400.0 * float(np.clip(bright, 0.2, 3.0)), sr)
    return _fin(y * np.minimum(1, t / 0.015) * np.minimum(1, (n / sr - t) / 0.04), sr, 0.6, src=sr)


@sfx("bike_bell", "Bicycle bell: ding-ding, the striker catching the dome on the way out and back.",
     freq=(1300, "Hz or note name"), strikes=(2, "dings"), gap=(0.16, "s between dings"), decay=(1.2, "s"), bright=(1.0, "0.3 dull .. 3 bright"),
     seed=(0, "variation"))
def bike_bell(sr=DEFAULT_SR, freq=1300, strikes=2, gap=0.16, decay=1.2, bright=1.0, seed=0):
    """Bell partial groups on one dome, the striker catching it out and back."""
    rng = np.random.default_rng(int(seed))
    dec = float(np.clip(decay, 0.05, 10.0))
    times = [j * max(float(gap), 0.02) for j in range(int(np.clip(strikes, 1, 20)))]
    n = _n(times[-1] + 1.2 * dec + 0.05)
    return _fin(_fbell(float(np.clip(to_hz(freq), 100.0, 6000.0)), dec, _since(n, times, P), P, rng, float(bright)), sr, 0.6)


@sfx("buzzer", "Electromechanical buzzer (door entry, game-show wrong answer): an armature rattling at twice the mains frequency against a metal plate.",
     dur=(0.6, "s"), freq=(100, "Hz or note name (100 = 50 Hz mains, 120 = 60 Hz)"), harsh=(0.6, "0..1 plate rattle"),
     beat=(0.0, "Hz between the two phasors: 0 = an armature locked to the mains; 0.4 = a transformer hum, whose fundamental cancels every 2.5 s"))
def buzzer(sr=DEFAULT_SR, dur=0.6, freq=100, harsh=0.6, beat=0.0):
    """Two sawtooth phasors summed and hard-clipped, into plate resonances 700 / 1400 / 2300 Hz. A 0.4 Hz offset is a drifting beat meant for
    an arc, where it gates the sparks; in a buzzer it swaps the 100 Hz line for 200 Hz every 2.5 s, so it is now an option."""
    n = int(sr * float(np.clip(dur, 0.05, 30.0)))
    f = float(np.clip(to_hz(freq), 20.0, 1000.0))
    b = 0.5 * float(np.clip(beat, 0.0, 20.0))
    x = np.clip(1.5 * (O.saw(f - b, n, sr) + O.saw(f + b, n, sr)), -1.0, 1.0)
    y = F.lowpass(x, 900.0, sr) + _c01(harsh) * sum(F.bandpass(x, fc, sr, 5.0) for fc in (700.0, 1400.0, 2300.0))
    t = np.arange(n) / sr
    return _fin(F.lowpass(y, 3500.0, sr) * np.minimum(1, t / 0.005) * np.minimum(1, (n / sr - t) / 0.01), sr, 0.6, src=sr)


# ============================================================================== sci-fi / game
def _shepard(sr: int, dur: float, rate: float, center: float, voices: int, dropoff: float) -> tuple[np.ndarray, float]:
    """Rising Shepard tone, exactly periodic with period T = 1 / rate (rounded to whole samples).
    Voice j sits at octave position u = j + t / T of ``voices`` octaves centred on ``center``:
    f = center 2^(u - N/2), amplitude exp(-dropoff x^2), x = 2u/N - 1.
    The phases are chosen so voice j at t = T continues as voice j + 1 at t = 0."""
    T = max(1, round(sr / rate)) / sr
    n = int(round(dur * sr))
    t = np.arange(n) / sr
    k = T / math.log(2.0)                                          # integral of 2^(t/T) dt = k (2^(t/T) - 1)
    y = np.zeros(n)
    j_lo = -int(math.ceil(dur / T)) - 1
    theta = 0.0
    for j in range(j_lo, voices):
        f0 = center * 2.0 ** (j - voices / 2.0)
        u = j + t / T
        on = (u >= 0) & (u < voices) & (f0 * 2.0 ** (t / T) < 0.45 * sr)   # no voice above Nyquist (it is at the envelope's foot)
        if np.any(on):
            x = 2.0 * u[on] / voices - 1.0
            y[on] += np.exp(-dropoff * x * x) * np.sin(theta + 2 * np.pi * f0 * k * (2.0 ** (t[on] / T) - 1.0))
        theta = (theta + 2 * np.pi * f0 * k) % (2 * np.pi)       # phase of voice j at t = T = start phase of voice j + 1
    return y, T


@sfx("shepard", "Shepard tone: a pitch that seems to rise (or fall) forever. Loops exactly when dur is a whole number of cycles.",
     direction=("up", "up|down"), rate=(0.125, "octaves per second (one cycle = 1 / rate s)"), dur=(8.0, "s"),
     freq=("C4", "centre of the spectral bell: Hz or note name"), voices=(10, "octave-spaced sines"), dropoff=(10.0, "width of the Gaussian spectral envelope (higher = narrower)"))
def shepard(sr=DEFAULT_SR, direction="up", rate=0.125, dur=8.0, freq="C4", voices=10, dropoff=10.0):
    """Pd D09: sines an octave apart under a Gaussian amplitude exp(-x^2 dropoff)
    over position x in [-1, 1]; `dropoff` 10. The spectrum is the same at t and t + 1/rate."""
    voices = int(np.clip(voices, 2, 12))
    fc = float(np.clip(to_hz(freq), 30.0, 2000.0))
    y, _ = _shepard(sr, float(np.clip(dur, 0.2, 600.0)), float(np.clip(rate, 0.005, 4.0)), fc, voices, float(np.clip(dropoff, 1.0, 60.0)))
    if direction == "down":
        y = y[::-1].copy()
    return _fin(y, sr, 0.5, src=sr, fade=0.004)


@sfx("transporter", "Sci-fi transporter beam: a shimmering whole-tone cluster of narrow noise bands that swells and dissolves upward.",
     dur=(3.0, "s"), freq=(466.0, "lowest band: Hz or note name"), bands=(10, "whole-tone steps stacked"), activate=(0, "1 = start on the activation buzz (775 Hz through three narrow bands)"), seed=(0, "variation"))
def transporter(sr=DEFAULT_SR, dur=3.0, freq=466.0, bands=10, activate=0, seed=0):
    """Source: a two-stage FM of triangles around the base frequency, written into a 120 ms buffer that is
    read in a loop (its repeats are the shimmer), the FM sweeping over 12 s. Ten parallel band-passes, base
    466 Hz, whole-tone ratio 1.12247, resonance 1200 (lowest) to 800 (highest), capped at 4.2 kHz, staggered
    onsets; then a flanger of three 12 ms delays swept over 12 s, and a reverb.
    `activate`: `transporter_activate`, ahead of the beam, at equal peak."""
    rng = np.random.default_rng(int(seed))
    n = int(sr * float(np.clip(dur, 0.3, 30.0)))
    t = np.arange(n) / n
    ts = np.arange(n) / sr
    f0 = float(np.clip(to_hz(freq), 60.0, 2000.0))
    nb = int(TRANSPORTER_BUFFER * sr)
    tri = lambda f: 4.0 * np.abs((np.cumsum(f) / sr) % 1.0 - 0.5) - 1.0     # noqa: E731
    src = np.zeros(n)
    for s in range(0, n, nb):
        sweep = 2.0 ** (((s / sr) % TRANSPORTER_SWEEP) / TRANSPORTER_SWEEP)
        tb = np.full(nb, f0 * sweep)
        m2 = tri(tb * TRANSPORTER_FM[0][0])
        m1 = tri(tb * TRANSPORTER_FM[1][0] + TRANSPORTER_FM[0][1] * tb * m2)
        buf = tri(tb + TRANSPORTER_FM[1][1] * tb * m1)
        src[s:s + nb] = buf[:min(nb, n - s)]
    y = np.zeros(n)
    nbands = int(np.clip(bands, 2, 16))
    for k in range(nbands):
        f = f0 * 2.0 ** (2 * k / 12.0)
        if f > 4200.0:
            break
        b = F.bandpass(src, f, sr, 1200.0 - 400.0 * k / max(nbands - 1, 1))
        y += b / (np.max(np.abs(b)) + 1e-12) * np.clip((t - 0.04 * k) / 0.15, 0, 1) * np.clip((1.0 - t) / (0.2 + 0.05 * k), 0, 1)
    fl = y.copy()
    for i in range(3):
        d = TRANSPORTER_FLANGE * (1.0 + 0.5 * np.sin(2 * np.pi * ts / TRANSPORTER_SWEEP + 2 * np.pi * i / 3)) * sr
        fl += np.interp(np.arange(n) - d, np.arange(n), y, left=0.0)
    from .reverbs import jcrev
    y = pc.fit(_m1(jcrev(fl / 4.0, sr, mix=0.3)), n)
    if activate:
        y = _nm(y) + transporter_activate(n, sr, np.random.default_rng(int(seed) + 1))
    return _fin(y, sr, 0.5, src=sr, fade=0.01)


TRANSPORTER_BUFFER = 0.12           # s of FM written, then looped
TRANSPORTER_SWEEP = 12.0            # s of one sweep of the FM and of the flanger
TRANSPORTER_FLANGE = 0.012          # s, the three flanger delays
TRANSPORTER_FM = ((0.6180, 4.0), (1.4142, 3.0))   # (ratio, index) of the two triangle stages


def _m1(y) -> np.ndarray:
    y = np.asarray(y, float)
    return y.mean(axis=1) if y.ndim == 2 else y


def transporter_activate(n: int, sr: int, rng) -> np.ndarray:
    """Phasor at 775 + 120 e Hz, AM by osc~ 30, noise x 0.1, x 3, clip +-1, bp~ 775, 1550, 2325 Q 13.
    Two envelopes triggered together: "1 0 0, 0 120 10" (e, the pitch blip) and "1 10 0, 0 5000 10" (level): the first bends the pitch, the second carries the level; AM as 0.5 + 0.5 osc so the carrier stays."""
    t = np.arange(n) / sr
    fast = np.clip(1.0 - np.maximum(t - 0.01, 0.0) / 0.12, 0.0, 1.0)
    slow = np.minimum(t / 0.01, 1.0) * np.clip(1.0 - np.maximum(t - 0.01, 0.0) / 5.0, 0.0, 1.0)
    saw = 2.0 * (np.cumsum(775.0 + 120.0 * fast) / sr % 1.0) - 1.0
    x = np.clip(3.0 * (saw * (0.5 + 0.5 * np.cos(2 * np.pi * 30.0 * t)) + 0.1 * rng.uniform(-1, 1, n)), -1.0, 1.0)
    y = sum(F.bandpass(x, f, sr, 13.0) for f in (775.0, 1550.0, 2325.0)) * slow
    return y / (np.max(np.abs(y)) + 1e-12)


@sfx("robot_babble", "Droid chatter: a string of random FM bleeps, whistles and warbles.",
     dur=(1.5, "s"), rate=(11.0, "syllables per second"), low=(300.0, "Hz lowest carrier"), high=(2800.0, "Hz highest carrier"), seed=(0, "variation"))
def robot_babble(sr=DEFAULT_SR, dur=1.5, rate=11.0, low=300.0, high=2800.0, seed=0):
    """Random FM babble: glides on 30 % of the syllables, rests on 15 %."""
    rng = np.random.default_rng(int(seed))
    dur = float(np.clip(dur, 0.1, 60.0))
    y = np.zeros(int(sr * dur))
    t0 = 0.0
    lo, hi = float(np.clip(low, 50, 0.2 * sr)), float(np.clip(high, 100, 0.22 * sr))
    while t0 < dur - 0.02:
        d = float(np.clip(rng.exponential(1.0 / max(float(rate), 0.5)), 0.03, 0.4))
        m = min(int(d * sr), len(y) - int(t0 * sr))
        if rng.random() > 0.15 and m > 8:
            tt = np.arange(m) / sr
            fc = lo * (hi / lo) ** rng.random()
            f = fc * (2.0 ** (rng.uniform(-1, 1) * tt / d) if rng.random() < 0.3 else 1.0)
            idx = rng.uniform(0.0, 3.0) * (fc < 1500.0)
            ph = 2 * np.pi * np.cumsum(np.broadcast_to(f, (m,))) / sr
            ratio = float(rng.choice([0.5, 1.0, 2.0, 0.02, 0.037]))
            idx = min(idx, _fm_cap(float(np.max(f)), ratio, 0.45 * sr) / 0.6)     # Carson band under Nyquist
            seg = np.sin(ph + idx * np.sin(ph * ratio))
            _put(y, seg * _gate(m, sr, 0.004), t0, rng.uniform(0.5, 1.0), rate=sr)
        t0 += d
    return _fin(F.lowpass(y, 5000.0, sr), sr, 0.5, src=sr)


@sfx("red_alert", "Starship red alert: a rising klaxon whoop repeated.",
     whoops=(3, "repeats"), low=(360, "start: Hz or note name"), high=(847, "end: Hz or note name"), rate=(0.78, "whoops per second"))
def red_alert(sr=DEFAULT_SR, whoops=3, low=360, high=847, rate=0.78):
    """The oscillator rises from 360 to 847 Hz in 900 ms; rate 0.78 makes the 70 % sweep 0.9 s.
    Exponential curve, two saws through a 3 kHz low-pass."""
    per = 1.0 / float(np.clip(rate, 0.2, 8.0))
    m = int(sr * per * 0.7)
    f = to_hz(low) * (to_hz(high) / to_hz(low)) ** (np.arange(m) / m)
    one = F.lowpass(O.saw(f, m, sr) + O.saw(f * 1.006, m, sr), 3000.0, sr) * _gate(m, sr, 0.012)
    y = np.zeros(int(sr * per * int(np.clip(whoops, 1, 60))))
    for j in range(int(np.clip(whoops, 1, 60))):
        _put(y, one, j * per, rate=sr)
    return _fin(y, sr, 0.55, src=sr)


@sfx("dark_drone", "Dark resonant drone: pink noise ringing three octave-stacked resonators inside a very long reverb.",
     dur=(6.0, "s"), note=(52.0, "MIDI note of the lowest resonator (52 = E3)"), ring=(0.2, "s ring time of the resonators"),
     reverb_time=(100.0, "s reverb T60"), seed=(0, "variation"))
def dark_drone(sr=DEFAULT_SR, dur=6.0, note=52.0, ring=0.2, reverb_time=100.0, seed=0):
    """genny.physical.ambience.dark_ambience."""
    y = amb.dark_ambience(float(np.clip(dur, 0.5, 120.0)), np.random.default_rng(int(seed)), note=float(np.clip(note, 20, 90)),
                          ring=float(np.clip(ring, 0.02, 3.0)), reverb_time=float(np.clip(reverb_time, 1.0, 200.0)))
    return _fin(y, sr, 0.6, fade=0.05)


@sfx("force_field", "Energy barrier hum: a beating mains-like buzz with a slow comb shimmer.",
     dur=(3.0, "s"), freq=(100, "Hz or note name"), shimmer=(0.5, "0..1 moving comb / high band"), seed=(0, "variation"))
def force_field(sr=DEFAULT_SR, dur=3.0, freq=100, shimmer=0.5, seed=0):
    """Clipped two-phasor hum through a comb swept 4-9 ms at 0.23 Hz, plus a 2.4 kHz band."""
    rng = np.random.default_rng(int(seed))
    n = int(sr * float(np.clip(dur, 0.2, 120.0)))
    f = float(np.clip(to_hz(freq), 20.0, 800.0))
    t = np.arange(n) / sr
    x = F.lowpass(np.clip(O.saw(f - 0.2, n, sr) + O.saw(f + 0.2, n, sr), -1.0, 1.0), 2200.0, sr)
    d = (0.0065 + 0.0025 * np.sin(2 * np.pi * 0.23 * t + rng.uniform(0, 6.28))) * sr
    comb = np.interp(np.arange(n) - d, np.arange(n), x, left=0.0)
    s = _c01(shimmer)
    hi = F.bandpass(rng.standard_normal(n), 2400.0, sr, 6.0) * (0.5 + 0.5 * np.sin(2 * np.pi * 7.0 * t))
    return _fin(x + s * comb + 0.15 * s * hi / (np.max(np.abs(hi)) + 1e-12), sr, 0.5, src=sr, fade=0.02)


@sfx("teleport", "Teleport zap: a fast endless-rise glissando with a noise rush, ending in a thump (or reversed for arriving).",
     dur=(0.8, "s"), direction=("out", "out (dematerialise) | in (arrive)"), seed=(0, "variation"),
     style=("shepard", "shepard | ring (a sine gliding up 4.5-7x under a ring modulator gliding 15-25 -> 80-140 Hz)"))
def teleport(sr=DEFAULT_SR, dur=0.8, direction="out", seed=0, style="shepard"):
    """A Shepard glissando at 4 octaves/s."""
    rng = np.random.default_rng(int(seed))
    dur = float(np.clip(dur, 0.2, 10.0))
    if style == "ring":
        n = int(dur * sr)
        t = np.arange(n) / sr
        x = t / dur
        f0, k, r0, r1 = rng.uniform(180.0, 260.0), rng.uniform(4.5, 7.0), rng.uniform(15.0, 25.0), rng.uniform(80.0, 140.0)
        f = f0 * k ** x * (1.0 + 0.01 * np.sin(2 * np.pi * 7.0 * t))
        rm = 0.25 + 0.75 * (0.5 + 0.5 * np.sin(2 * np.pi * np.cumsum(r0 * (r1 / r0) ** x) / sr))
        y = np.sin(2 * np.pi * np.cumsum(f) / sr) * rm * np.sin(np.pi * x) ** 0.8
        return _fin(y[::-1].copy() if direction == "in" else y, sr, 0.6, src=sr)
    y, _ = _shepard(sr, dur, 4.0, 520.0, 8, 6.0)
    n = len(y)
    t = np.arange(n) / n
    rush = F.biquad(rng.standard_normal(n), "bandpass", 400.0 * 12.0 ** t, sr, q=2.0)
    y = (y / (np.max(np.abs(y)) + 1e-12) + 0.5 * rush / (np.max(np.abs(rush)) + 1e-12)) * t ** 1.5 * np.clip((1 - t) / 0.03, 0, 1)
    m = int(0.12 * sr)
    tm = np.arange(m) / sr
    y = np.concatenate([y, 1.2 * np.sin(2 * np.pi * np.cumsum(50.0 + 90.0 * np.exp(-tm / 0.02)) / sr) * np.exp(-tm / 0.035)])
    return _fin(y[::-1].copy() if direction == "in" else y, sr, 0.6, src=sr)


@sfx("shield_up", "Energy shield raised: two detuned triangles settling in pitch under a slow ring modulator.",
     dur=(0.5, "s"), freq=(0.0, "Hz (0 = random 320-520)"), seed=(0, "variation"))
def shield_up(sr=DEFAULT_SR, dur=0.5, freq=0.0, seed=0):
    rng = np.random.default_rng(int(seed))
    dur = float(np.clip(dur, 0.1, 10.0))
    n = int(dur * sr)
    t = np.arange(n) / sr
    f0 = float(freq) if freq > 0 else rng.uniform(320.0, 520.0)
    ring, depth, det, att = rng.uniform(40.0, 90.0), rng.uniform(0.5, 0.7), rng.uniform(0.004, 0.01), dur * rng.uniform(0.2, 0.35)
    f = f0 * (1.0 + 0.06 * np.exp(-6.0 * t))
    tri = lambda ph: 1.0 - 4.0 * np.abs((ph % 1.0) - 0.5)
    y = 0.5 * (tri(np.cumsum(f) / sr) + tri(np.cumsum(f * (1.0 + det)) / sr))
    y = y * ((1.0 - depth) + depth * (0.5 + 0.5 * np.sin(2 * np.pi * ring * t)))
    y = y * np.minimum(1.0, t / att) * np.clip((dur - t) / (0.3 * dur), 0.0, 1.0)
    return _fin(y, sr, 0.6, src=sr)


def doppler_factor(v_radial: float) -> float:
    """f_heard / f_source for a source approaching at v_radial m/s: c / (c - v)."""
    return C_AIR / (C_AIR - v_radial)


@sfx("energy_blade", "Energy sword: a steady beating hum that swells and bends in pitch as the blade swings past.",
     dur=(2.0, "s"), freq=(92, "hum: Hz or note name"), swings=(1, "swings"), speed=(25.0, "m/s peak blade speed: Doppler shift c / (c - v)"),
     buzz=(0.4, "0..1 upper buzz"), seed=(0, "variation"))
def energy_blade(sr=DEFAULT_SR, dur=2.0, freq=92, swings=1, speed=25.0, buzz=0.4, seed=0):
    """Hum + swing: the radial velocity of the blade past the listener bends the pitch by the
    Doppler factor c / (c - v) and raises the level. Hum: two saws 0.7 % apart through 900 Hz and a
    pulse band at 2.4 kHz; swing 0.35 s, +8 dB at full speed."""
    rng = np.random.default_rng(int(seed))
    dur = float(np.clip(dur, 0.2, 60.0))
    n = int(sr * dur)
    t = np.arange(n) / sr
    v = np.zeros(n)
    for j in range(int(np.clip(swings, 0, 40))):
        c = (j + 0.5 + 0.15 * rng.standard_normal() * (swings > 1)) * dur / max(int(swings), 1)
        v += float(np.clip(speed, 0.0, 150.0)) * np.sin(2 * np.pi * (t - c) / 0.35) * np.exp(-((t - c) / 0.12) ** 2) * -1.0
    f = float(np.clip(to_hz(freq), 20.0, 600.0)) * C_AIR / (C_AIR - v)
    a = np.abs(v) / (float(np.max(np.abs(v))) + 1e-9)
    x = O.saw(f, n, sr) + O.saw(f * 1.007, n, sr)
    y = F.lowpass(x, 900.0, sr) * (1.0 + 1.5 * a) + _c01(buzz) * F.bandpass(O.pulse(f * 2.0, n, sr, width=0.12), 2400.0, sr, 2.0) * (0.3 + a)
    return _fin(y, sr, 0.6, src=sr, fade=0.03)


@sfx("warp", "Warp / hyperspace jump: engines winding up faster and faster, then the jump.",
     dur=(3.0, "s"), start=(40.0, "Hz"), end=(1400.0, "Hz at the jump"), seed=(0, "variation"))
def warp(sr=DEFAULT_SR, dur=3.0, start=40.0, end=1400.0, seed=0):
    """An accelerating (t^3) sweep of three detuned saws with a noise band riding on it; the jump is a falling 120 -> 30 Hz thump."""
    rng = np.random.default_rng(int(seed))
    dur = float(np.clip(dur, 0.5, 30.0))
    n = int(sr * dur * 0.85)
    u = np.arange(n) / n
    f = float(start) * (float(np.clip(end, 100, 0.2 * sr)) / float(start)) ** (u ** 3)
    x = sum(O.saw(f * d, n, sr) for d in (1.0, 1.01, 0.5))
    y = F.biquad(x, "lowpass", np.clip(f * 3.0, 200.0, 3200.0), sr, q=0.9) + 0.3 * F.biquad(rng.standard_normal(n), "bandpass", np.clip(f * 2.0, 100.0, 4000.0), sr, q=1.5) * u
    y = y / (np.max(np.abs(y)) + 1e-12) * (0.15 + 0.85 * u ** 2)
    m = int(sr * dur) - n
    tm = np.arange(m) / sr
    jump = np.sin(2 * np.pi * np.cumsum(30.0 + 90.0 * np.exp(-tm / 0.06)) / sr) * np.exp(-tm / (0.25 * m / sr + 1e-9))
    return _fin(np.concatenate([y, 1.3 * jump]), sr, 0.65, src=sr, fade=0.01)


@sfx("charge_up", "Weapon / ability charging: a tone climbing with a tremolo that speeds up until it is ready.",
     dur=(1.5, "s"), start=(120.0, "Hz"), end=(1600.0, "Hz"), ready=(1, "1 = end on a short ready blip"))
def charge_up(sr=DEFAULT_SR, dur=1.5, start=120.0, end=1600.0, ready=1):
    """Sine + triangle sweep, tremolo 6 -> 40 Hz."""
    n = int(sr * float(np.clip(dur, 0.1, 30.0)))
    u = np.arange(n) / n
    f = float(start) * (float(np.clip(end, 50, 0.2 * sr)) / float(start)) ** u
    trem = 0.6 + 0.4 * np.sin(2 * np.pi * np.cumsum(6.0 + 34.0 * u ** 2) / sr)
    y = (O.sine(f, n, sr) + 0.3 * F.lowpass(O.triangle(f * 2.0, n, sr), 3500.0, sr)) * trem * (0.2 + 0.8 * u) * _gate(n, sr, 0.01)
    if ready:
        m = int(0.12 * sr)
        tm = np.arange(m) / sr
        y = np.concatenate([y, np.sin(2 * np.pi * min(float(end), 0.2 * sr) * tm) * np.exp(-tm / 0.03) * _gate(m, sr, 0.002)])
    return _fin(y, sr, 0.55, src=sr)


@sfx("scanner", "Scanner / tricorder: a repeating swept warble.",
     dur=(2.0, "s"), rate=(2.0, "sweeps per second"), low=(600.0, "Hz"), high=(1800.0, "Hz"), warble=(18.0, "Hz of the fast flutter"))
def scanner(sr=DEFAULT_SR, dur=2.0, rate=2.0, low=600.0, high=1800.0, warble=18.0):
    """A sine swept low -> high each cycle with a square-ish flutter of a fifth."""
    n = int(sr * float(np.clip(dur, 0.1, 60.0)))
    t = np.arange(n) / sr
    u = (t * float(np.clip(rate, 0.1, 20.0))) % 1.0
    f = float(low) * (float(np.clip(high, 100, 0.2 * sr)) / float(low)) ** np.sin(np.pi * u) ** 2
    f = f * np.where(np.sin(2 * np.pi * float(warble) * t) > 0, 1.0, 1.5)
    y = np.sin(2 * np.pi * np.cumsum(F.lowpass(f, 200.0, sr)) / sr) * (0.4 + 0.6 * np.sin(np.pi * u) ** 0.5)
    return _fin(F.lowpass(y, 4500.0, sr), sr, 0.45, src=sr, fade=0.01)


# ============================================================================== drums
def _phisem_drum(name: str, dur: float):
    def fn(sr=DEFAULT_SR, vel=1.0, size=1.0, seed=0):
        return _fin(_shake(name, dur, _c01(vel, 0.02, 1.0), 0.0, int(seed), float(np.clip(size, 0.2, 5.0))), sr, None, tail=0.03)
    return fn


_PH = dict(size=(1.0, "instrument size (bigger = lower)"), seed=(0, "variation"))
cabasa = drum("cabasa", "Cabasa (PhISEM: 512 beads on a 3 kHz gourd): a short dry scratch.", **_PH)(_phisem_drum("Cabasa", 0.35))
sekere = drum("sekere", "Sekere / shekere (PhISEM: 64 beads, 5.5 kHz): a bright net-on-gourd shake.", **_PH)(_phisem_drum("Sekere", 0.5))
maraca = drum("maraca", "Maraca (PhISEM: 25 beans in a 3.2 kHz shell), one shake.", **_PH)(_phisem_drum("Maraca", 0.4))
bamboo_chimes = drum("bamboo_chimes", "Bamboo wind chimes (PhISEM: sparse knocks on three tubes near 2.8 kHz), one stir.", **_PH)(_phisem_drum("BambooChimes", 2.0))
sandpaper = drum("sandpaper", "Sandpaper blocks (PhISEM: 128 grains, 4.5 kHz), one rub.", **_PH)(_phisem_drum("Sandpaper", 0.3))
sleigh_bells = drum("sleigh_bells", "Sleigh bells (PhISEM: 32 pellets ringing five bell modes 2.5-9.8 kHz), one shake that rings on.", **_PH)(_phisem_drum("SleighBells", 1.2))


@drum("agogo", "Agogo bell (STK ModalBar preset: modes 1, 4.08, 6.669 and a fixed 3725 Hz).",
      tune=(660.0, "Hz of the bell (the high bell is about a third above the low one)"), hardness=(0.6, "0 soft beater .. 1 hard"), decay=(1.0, "s"))
def agogo(sr=DEFAULT_SR, vel=1.0, tune=660.0, hardness=0.6, decay=1.0):
    y = modal.stk_modal_bar("Agogo", float(np.clip(tune, 80.0, 3000.0)), float(np.clip(vel, 0.05, 1.0)) * 0.9, hardness=_c01(hardness),
                            dur=float(np.clip(decay, 0.05, 6.0)))
    return _fin(y, sr, 0.8 * _c01(vel, 0.05), tail=0.05)


def _metal_drum(table: str, fs: float, cms: float, ds: float, micro: int):
    def fn(sr=DEFAULT_SR, vel=1.0, tune=1.0, decay=1.0, seed=0):
        v = _c01(vel, 0.05)
        y = _strike(table, fs * float(np.clip(tune, 0.25, 4.0)), cms, 0.0, np.random.default_rng(int(seed)), velocity=0.2 + v, micro=micro,
                    d_scale=ds / float(np.clip(decay, 0.1, 5.0)), max_dur=2.5)
        return _fin(y, sr, 0.85 * v, tail=0.02)
    return fn


_MD = dict(tune=(1.0, "pitch scale (2 = an octave up)"), decay=(1.0, "ring-time scale"), seed=(0, "variation"))
anvil = drum("anvil", "Anvil struck with a hammer: measured steel (sword1.sy) under a 0.15 ms contact with re-contact chatter.", **_MD)(
    _metal_drum("sy_sword1", 0.7, 0.15, 1.5, 3))
brake_drum = drum("brake_drum", "Brake drum: the measured iron wok struck hard, a dry clanging pitch.", **_MD)(
    _metal_drum("sy_wok", 1.4, 0.3, 2.5, 1))


TUBE_SQUARE = (1.0, 2.0 ** (17 / 12), 2.0 ** (28 / 12), 8.0)          # square tubes: fundamental, 4th, 3rd, octave above
CHIME_OBRIEN = (585.8, 656.0, 781.8, 877.5, 982.5, 1167.0)             # measured aluminium chime set, D E G A B D
TUBES = {"bar": FREE_BAR[:4], "square": TUBE_SQUARE}


def chime_decay(f, law: str = "ratio", decay: float = 3.0, f0: float = 1.0):
    """T60 (s) of a mode at ``f`` Hz: ``ratio`` = decay x f0 / f; ``rayleigh`` = aluminium Rayleigh damping
    (alpha, beta) = (0, 1e-7), d = (alpha + beta w^2)/2, T60 = ln(1000)/d."""
    f = np.asarray(f, float)
    if law == "rayleigh":
        from .contact import OBRIEN_RAYLEIGH
        a, b = OBRIEN_RAYLEIGH["aluminium"]
        return 6.9078 / (0.5 * (a + b * (2 * np.pi * f) ** 2))
    return decay * f0 / f


FREE_BAR_BETA = (4.730040745, 7.853204624, 10.995607838, 14.137165491)    # beta_n L of a free-free bar: cos cosh = 1


def tube_amplitudes(strike: float) -> np.ndarray:
    """Mode amplitudes |phi_n(x)| of a free bar struck at ``strike`` (fraction of the length, 0 = end)."""
    from .contact import _bar_shape
    return np.abs(_bar_shape(np.array(FREE_BAR_BETA), float(np.clip(strike, 0.0, 1.0)), "free"))


def _tube(freq: float, n: int, rate: int, decay: float, bright: float, rng=None, tube: str = "bar", law: str = "ratio",
          strike: float = -1.0) -> np.ndarray:
    """A hanging metal tube, modes ``TUBES[tube]`` x freq, T60 by ``chime_decay``; amplitudes 1, 0.5, 0.25, 0.12, or
    with ``strike`` in 0..1 the free-bar mode shapes |phi_n| at the strike point."""
    t = np.arange(n) / rate
    y = np.zeros(n)
    amps = (1.0, 0.5, 0.25, 0.12) if strike < 0 else tube_amplitudes(strike)
    for r, a in zip(TUBES[tube], amps):
        f = freq * r
        if f < 0.45 * rate:
            ph = rng.uniform(0, 2 * np.pi) if rng is not None else 0.0
            y += a / (1.0 + (f / (3500.0 * bright)) ** 2) * np.exp(-6.91 * t / chime_decay(f, law, decay, freq)) * np.sin(2 * np.pi * f * t + ph)
    return y * np.minimum(1.0, t / 0.001)


@drum("wind_chimes", "Metal wind chimes stirred: several tuned tubes struck at random, ringing over each other.",
      tune=(880.0, "Hz of the lowest tube"), tubes=(7, "strikes in the stir"), spread=(1.0, "s over which they are struck"), decay=(3.0, "s ring"),
      tube=("bar", "bar: free-bar overtones | square: measured square tubes 1 : 2.67 : 5.04 : 8"),
      law=("ratio", "ratio: T60 = decay / overtone ratio | rayleigh: aluminium Rayleigh damping, T60 ~ 1/f^2 (10 s at 587 Hz)"),
      scale=("pentatonic", "pentatonic on `tune` | measured: the six tubes of a real aluminium chime set (586-1167 Hz)"),
      strike=(-1.0, "-1: fixed weights 1, 0.5, 0.25, 0.12 | 0..1: struck at this fraction of the length, amplitudes |phi_n| of a free bar"),
      seed=(0, "variation"))
def wind_chimes(sr=DEFAULT_SR, vel=1.0, tune=880.0, tubes=7, spread=1.0, decay=3.0, tube="bar", law="ratio", scale="pentatonic", strike=-1.0, seed=0):
    """Modal tubes (see the `wind_chime` instrument) struck at random over ``spread`` s."""
    if tube not in TUBES or law not in ("ratio", "rayleigh") or scale not in ("pentatonic", "measured"):
        raise ValueError("wind_chimes: tube bar|square, law ratio|rayleigh, scale pentatonic|measured")
    rng = np.random.default_rng(int(seed))
    dec, spread = float(np.clip(decay, 0.2, 20.0)), float(np.clip(spread, 0.0, 10.0))
    ring = dec if law == "ratio" else min(float(chime_decay(min(CHIME_OBRIEN) if scale == "measured" else tune, "rayleigh")), 20.0)
    y = np.zeros(int(sr * (spread + ring)))
    for t0 in np.sort(rng.uniform(0, spread, int(np.clip(tubes, 1, 40)))):
        if scale == "measured":
            f = float(rng.choice(CHIME_OBRIEN))
        else:
            f = float(np.clip(tune, 100, 4000)) * 2.0 ** (rng.choice([0, 2, 4, 7, 9, 12]) / 12.0)
        _put(y, _tube(f, int(sr * ring), sr, dec, 1.0, rng, tube, law, float(strike)), t0, rng.uniform(0.4, 1.0), rate=sr)
    return _fin(y, sr, 0.7 * _c01(vel, 0.05), src=sr, tail=0.05)


# ============================================================================== instruments
# Output peak at vel 1 per instrument, set so the K-weighted level is -12 dB at vel 0.9 (the same
# target as genny._levels.TRIM, which this module may not edit).
_LVL = {"church_bell": 1.01, "handbell": 0.79, "china_bell": 0.79, "bowed_bowl": 0.45, "glass_harmonica": 0.44, "hurdy_gurdy": 0.87, "wind_chime": 0.68}


def _inst(name: str, y: np.ndarray, sr: int, vel: float, src: int = P) -> np.ndarray:
    return _fin(y, sr, _LVL[name] * (0.15 + 0.85 * _c01(vel, 0.02)), src=src, fade=0.001, tail=0.01)


def _roll(f, bright: float, corner: float = 3200.0):
    return 1.0 / (1.0 + (np.asarray(f, float) / (corner * max(float(bright), 0.05))) ** 2)


BELLS = {"church": "churchBell", "english": "englishBell", "french": "frenchBell", "german": "germanBell", "russian": "russianBell",
         "standard": "standardBell", "measured": "sy_bell4"}   # measured: a real true-harmonic bell, T60 as recorded at decay 8


# Church bells.
# Upper partials in cents from the nominal, c = m c_on + b, Table 5-4 (2066 bells): SQ, I-7 .. I-11.
_HIBBERT_UP = ((0.557, -0.175), (1.363, 0.186), (1.668, 0.106), (1.929, -0.203), (2.156, -1.307), (2.362, -9.003))
# Mean levels, whole round, 250 ms: hum, prime, tierce, quint, nominal, SQ, ON, I-7 .. I-11
_HIBBERT_G = (2.79, 5.09, 8.35, 1.00, 10.60, 8.79, 7.10, 4.94, 4.03, 3.30, 2.05, 1.52)
# Measured bells, Hz, lowest first, with the note heard. Ipswich old 9th levels in dB.
_HIBBERT_BELLS = {
    "old_english": ((201.9, 323.7, 424.2, 562.8, 723.4, 1091.4, 1507.3, 1959.3, 2443.2, 2948.3, 3467.6, 4000.8),
                    (-19.3, -5.9, -10.7, -18.7, -7.9, 0.5, -3.2, -8.1, -9.7, -15.9, -27.3, -30.5), 723.4 / 2),
    "medieval": ((135.4, 267.4, 346.4, 365.8, 615.8, 912.0, 1231.6, 1582.4, 1961.8, 2356.0, 2758.0, 3166.0), None, 307.3),   # Freiburg Hosanna, 1258
    "big_ben": ((335.1, 494.3, 676.8, 882.2, 1087.0, 1304.4, 1746.0), None, 327.0),
    "petersglocke": ((252.3, 375.3, 510.4, 658.0, 808.8, 966.6, 1300.3), None, 253.4),
}
_SECOND_STRIKE = {"big_ben": 428.1, "petersglocke": 328.0}


def _hibbert_modes(kind, freq, octave_nominal, tune, dec):
    """(freqs, T60s, gains) with the written note where `tune` says. T60: hum 13.5 s, the rest 2.7 s at decay 8 (sec. 6.9.1, 60 % in
    1 s and 200 ms, extrapolated beyond 250 ms). tune strike: nominal = 2 note 2^(-s/1200), s from sec. 9.7."""
    if kind == "hibbert":
        on = float(np.clip(octave_nominal, 1140.0, 1280.0))
        nom = 2.0 * freq
        if tune == "strike":
            for _ in range(2):                                   # s depends on f
                s = 0.036 * 200.0 + 456.668 * nom ** -1.10012 * (on - 1140.0)   # p -1200, t -900 (true-harmonic)
                nom = 2.0 * freq * 2.0 ** (-s / 1200.0)
        elif tune != "nominal":
            nom = freq * {"hum": 4.0, "prime": 2.0}[tune]
        cents = [-2400.0, -1200.0, -900.0, -498.0, 0.0] + [m * on + b for m, b in _HIBBERT_UP[:1]] + [on] + [m * on + b for m, b in _HIBBERT_UP[1:]]
        f = nom * 2.0 ** (np.array(cents) / 1200.0)
        g = np.array(_HIBBERT_G)
    else:
        f0, db, heard = _HIBBERT_BELLS[kind]
        f = np.array(f0)
        low = kind in _SECOND_STRIKE
        g = 10.0 ** (np.array(db) / 20.0) if db is not None else np.array(_HIBBERT_G[4:4 + len(f0)]) if low else np.array(_HIBBERT_G)
        ref = {"strike": heard, "second": _SECOND_STRIKE.get(kind, heard)} | ({"nominal": f[0]} if low else {"hum": f[0], "prime": f[1], "nominal": f[4]})
        f = f * freq / ref.get(tune, heard)
    t60 = np.full(f.size, 2.7 * dec / 8.0)
    t60[0] = 13.5 * dec / 8.0
    return f, t60, g / np.max(g)


def bell_nominal(diameter: float) -> float:
    """Nominal (Hz) of a cast bronze bell of lip diameter `diameter` (m): ln d = 6.95 - 1.04 ln f."""
    return float(np.exp((6.95 - np.log(max(diameter, 1e-3))) / 1.04))


def bell_partial(table: str, which: str = "hum", position: float = 0.0) -> float:
    """Frequency (Hz, as tabulated) of the partial that carries the written pitch: the hum (the two
    lowest modes, a split doublet) or the next doublet up ("prime": 1.89-1.91 x the hum in the
    church, english, russian and standard bells; 1.38 French, 1.70 German). Gain-weighted mean."""
    t = md.TABLES[table]
    if t["kind"] == "vdd":                                   # measured tables are not sorted, and have no doublets to average
        f = np.sort(t["freqs"])
        return float(f[np.argmin(np.abs(f - 2.0 * f[0]))] if which == "prime" else f[0])
    i = 2 if which == "prime" else 0
    g = modal._interp_rows(t["gains"], position)[i:i + 2]
    return float(np.dot(t["freqs"][i:i + 2], g) / (np.sum(g) + 1e-12))


@instrument("church_bell", "Cast bell from a finite-element model (50 modes with split doublets): church, English, French, German, Russian or standard profile. "
            "The written note is the hum, the lowest and longest partial; the strike note is heard about an octave above.",
            family="mallet", span=("C3", "C5"), type=("church", "|".join(BELLS) + "|hibbert|old_english|medieval|big_ben|petersglocke|ariutti"),
            octave_nominal=(1240.0, "hibbert: cents 1140..1280; low = old and thick, high = thin; moves every upper partial (2066 bells)"), position=(0.0, "0 lip .. 1 crown: where the clapper strikes (7 measured points)"),
            decay=(8.0, "s T60 of the hum"), bright=(1.0, "0.4 soft clapper, distant .. 2.5 hard and close"),
            tune=("hum", "hum|prime: which partial plays the written note; also nominal, and strike (the note heard) for the hibbert types; second: the higher note big_ben and petersglocke also give"),
            diameter=(0.0, "m at the lip; > 0 sets the nominal from the size (above ~0.9 m, nominal under ~900 Hz) and replaces the note; the hibbert types only"),
            swing=(0.0, "degrees each side the bell swings (0..90): Doppler and the mouth turning to and away, 2 s period"))
def church_bell(freq, dur, sr=DEFAULT_SR, vel=1.0, type="church", position=0.0, decay=8.0, bright=1.0, tune="hum", octave_nominal=1240.0,
                diameter=0.0, swing=0.0):
    """Faust physmodels.lib bell models: modeFilter bank, T60 law t60 (1 - f/fmax)^slope,
    struck with a (1 - cos) contact that shortens with velocity."""
    table = BELLS.get(str(type), "churchBell")
    pos, dec = _c01(position), float(np.clip(decay, 0.3, 40.0))
    if str(type) == "ariutti":
        from .physics import TunedPercussion
        y = TunedPercussion("ariutti").note(float(freq) / 2.0, 5.0 * dec / 8.0, velocity=1.0, sr=P)
        return _inst("church_bell", y / (np.max(np.abs(y)) + 1e-12), sr, vel)
    if diameter > 0:
        type = type if str(type) in _HIBBERT_BELLS else "hibbert"
        freq, tune = bell_nominal(float(diameter)) / (2.0 if type == "hibbert" else 1.0), "nominal"
    if str(type) == "hibbert" or str(type) in _HIBBERT_BELLS:
        hf, ht, hg = _hibbert_modes(str(type), float(freq), float(octave_nominal), str(tune), dec)
        keep = hf < 0.45 * P
        m = modal.Modes(hf[keep], 6.9078 / ht[keep], hg[keep], "vdd")
    else:
        m = None
    m = m if m is not None else modal.resolve(table, float(freq) / bell_partial(table, str(tune), pos), pos, t60=dec,
                      d_scale=8.0 / dec if md.TABLES[table]["kind"] == "vdd" else 1.0)
    m.gains = m.gains * _roll(m.freqs, bright)
    force = np.zeros(_n(max(float(dur), 0.05) + min(0.5 * dec, 4.0)))
    p = modal.bang(modal.contact_time(0.5 / max(float(bright), 0.2), max(float(vel), 0.05)))
    force[:len(p)] = p[:len(force)]
    y = modal.render_modes(force, m)
    k = int(0.4 * len(y))
    y[-k:] *= pc.pow_decay(k, 2.0)
    if swing > 0:
        y = swung(y, swing)
    return _inst("church_bell", y, sr, vel)


BELL_SWING = dict(arm=1.0, period=2.0, tilt=0.5)


def swung(y, degrees, arm=BELL_SWING["arm"], period=BELL_SWING["period"]):
    """A bell swinging up to ``degrees`` each side: its mouth travels arm sin(theta), heard through the moving delay
    (Doppler), and it is louder as the mouth turns toward the listener."""
    t = np.arange(len(y)) / P
    th = math.radians(float(np.clip(degrees, 0.0, 90.0))) * np.sin(2 * np.pi * t / period)
    z = np.interp(t - arm * np.sin(th) / C_AIR, t, y, left=0.0)
    return z * (1.0 + BELL_SWING["tilt"] * np.sin(th))


@instrument("handbell", "Small bright handbell (bell partial groups): hammer clang that settles to a pure tone, with a second bell a hair sharp for the beat.",
            family="mallet", span=("G4", "G6"), decay=(2.5, "s base decay"), detune=(0.3, "% the second bell is sharp (0 = one bell)"),
            bright=(1.0, "0.3 dull .. 3 all hammer partials"))
def handbell(freq, dur, sr=DEFAULT_SR, vel=1.0, decay=2.5, detune=0.3, bright=1.0):
    """Partial ratios 0.501, 1, 0.7 | 2.002, 3, 9.6 | 2.49, 11, 2.571 | 3.05, 6.242, 12.49 |
    13, 16, 24 with group decays x 1.2, 0.9, 0.25, 0.14, 0.07; the written note is ratio 1."""
    rng = np.random.default_rng(7)
    dec = float(np.clip(decay, 0.1, 12.0))
    n = int(sr * (max(float(dur), 0.05) + min(1.2 * dec, 4.0)))
    tau = np.arange(n) / sr
    y = _fbell(float(freq), dec, tau, sr, rng, float(bright) * 0.6)
    if detune:
        y = y + 0.6 * _fbell(float(freq) * (1 + float(detune) / 100.0), dec, tau, sr, rng, float(bright) * 0.6)
    return _inst("handbell", y, sr, vel, src=sr)


@instrument("china_bell", "Chinese temple bell: six inharmonic partials with very long quartic decays.",
            family="mallet", span=("C3", "C5"), decay=(1.0, "ring-time scale (1 = 15-22 s partial decays)"), bright=(1.0, "0.4 dark .. 2.5 bright"))
def china_bell(freq, dur, sr=DEFAULT_SR, vel=1.0, decay=1.0, bright=1.0):
    """`pd chinabell`: ratios 1, 1.666, 4.2, 5.89, 7.666, 8.9; amplitudes 0.25 .. 0.07;
    each x (5 ms attack, linear decay 15-22 s)^4. The written note is ratio 1."""
    tb = md.TABLES["chinabell"]
    rng = np.random.default_rng(7)
    k = float(np.clip(decay, 0.05, 3.0))
    n = int(sr * (max(float(dur), 0.05) + min(4.0 * k, 5.0)))
    t = np.arange(n) / sr
    y = np.zeros(n)
    for r, dms, a in zip(tb["ratios"], tb["decay_ms"], tb["amps"]):
        f = float(freq) * r
        if f < 0.45 * sr:
            env = np.where(t < 0.005, t / 0.005, np.clip(1 - (t - 0.005) / (dms * 1e-3 * k), 0, 1)) ** 4
            y += a * _roll(f, bright) * env * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    m = int(0.3 * n)
    y[-m:] *= pc.pow_decay(m, 2.0)
    return _inst("china_bell", y, sr, vel, src=sr)


@functools.lru_cache(maxsize=512)
def _banded_ir(preset: str, f0: float, length: float) -> np.ndarray:
    """Impulse response of the STK BandedWG resonator (struck mode, fractional delays)."""
    return wg.banded(preset, f0, length, None, mode="strike", velocity=1.0, exact_pitch=True)


def _banded_note(name: str, preset: str, freq, dur, sr, vel, mode, pressure, ring: float) -> np.ndarray:
    """BandedWG. mode "strike" is the kernel's own pluck. mode "bow"
    is commuted: the kernel's bow junction needs seconds to speak (measured: peak 0.003 after 1 s of
    bowing), so the band resonator's impulse response is driven by a friction force instead: one
    slip per period of the fundamental (a sawtooth) plus rosin noise. The bands, their
    inharmonic ratios, gains and tuning are the kernel's."""
    from scipy.signal import fftconvolve
    f0 = float(np.clip(freq, 20.0, 1568.0))                     # BandedWG clamps the fundamental at 1568 Hz
    h = _banded_ir(preset, round(f0, 3), ring)
    d = max(float(dur), 0.03)
    if mode == "strike":
        y = h[:_n(d + ring)].copy()
    else:
        n = _n(d + 0.12)
        t = np.arange(n) / P
        env = np.minimum(1.0, t / 0.06) * np.clip((d + 0.1 - t) / 0.1, 0.0, 1.0)
        pr = _c01(pressure)
        e = (O.saw(f0, n, P) * (0.3 + 0.7 * pr) + 0.08 * np.random.default_rng(7).standard_normal(n)) * env
        y = fftconvolve(e, h)[:n + _n(ring)]
    k = int(0.35 * len(y))
    y[-k:] *= pc.pow_decay(k, 2.0)
    return _inst(name, pc.lowpass(y, 5000.0), sr, vel)


@instrument("glass_harmonica", "Glass harmonica: a rubbed glass bowl (banded waveguide, modes 1, 2.32, 4.25, 6.63, 9.38) that swells slowly and rings on.",
            family="bowed", span=("C4", "G6"), mode=("bow", "bow (rubbed rim) | strike (tapped glass)"), pressure=(0.5, "0..1 finger pressure: more upper modes"),
            ring=(1.5, "s of ring kept after the note"))
def glass_harmonica(freq, dur, sr=DEFAULT_SR, vel=1.0, mode="bow", pressure=0.5, ring=1.5):
    return _banded_note("glass_harmonica", "glass_harmonica", freq, dur, sr, vel, str(mode), pressure, float(np.clip(ring, 0.2, 4.0)))


@instrument("bowed_bowl", "Tibetan singing bowl rubbed with a wooden puja (banded waveguide: twelve modes in beating pairs), or struck.",
            family="bowed", span=("C3", "C5"), mode=("bow", "bow (rubbed rim) | strike"), pressure=(0.5, "0..1 puja pressure: more upper modes"),
            ring=(3.0, "s of ring kept after the note"))
def bowed_bowl(freq, dur, sr=DEFAULT_SR, vel=1.0, mode="bow", pressure=0.5, ring=3.0):
    return _banded_note("bowed_bowl", "tibetan_bowl", freq, dur, sr, vel, str(mode), pressure, float(np.clip(ring, 0.2, 6.0)))


def ac_cents(y: np.ndarray, f0: float, rate: int) -> float:
    """Pitch error in cents from the autocorrelation peak next to the expected period (the
    measurement tests/test_instruments.py uses)."""
    y = y[int(0.05 * rate):int(0.05 * rate) + 16384]
    y = y - y.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(y * np.hanning(len(y)), 4 * len(y))) ** 2)
    p = rate / f0 * 4
    lo = int(p * 0.94)
    k = lo + int(np.argmax(ac[lo:int(p * 1.06) + 2]))
    k = k + 0.5 * (ac[k - 1] - ac[k + 1]) / (ac[k - 1] - 2 * ac[k] + ac[k + 1] + 1e-20)
    return float(1200 * np.log2((rate * 4 / k) / f0))


@instrument("hurdy_gurdy", "Hurdy-gurdy: a string bowed by a rosined wheel over a buzzing bridge (the trompette), with an optional drone an octave below.",
            family="bowed", span=("G2", "G5"), buzz=(0.6, "0.2 constant rattle .. 1.5 clean: the buzzing bridge's threshold"),
            pressure=(0.8, "0.3..1 wheel pressure"), drone=(0.0, "0..1 level of the bourdon an octave below"), attack=(0.06, "s for the wheel to reach speed"))
def hurdy_gurdy(freq, dur, sr=DEFAULT_SR, vel=1.0, buzz=0.6, pressure=0.8, drone=0.0, attack=0.06):
    """genny.physical.waveguides.hurdy_gurdy: STK Bowed strings at constant wheel speed with a
    nonlinear chattering bridge; the kernel calibrates its own pitch."""
    f0 = float(np.clip(freq, 30.0, 2000.0))
    bz, pr = float(np.clip(buzz, 0.05, 3.0)), float(np.clip(pressure, 0.2, 1.0))
    d = max(float(dur), 0.12) + 0.08
    kw = dict(amp=0.55 + 0.4 * _c01(vel),      # below ~0.5 the wheel cannot start a string above 1 kHz
               buzz=bz, wheel_attack=float(np.clip(attack, 0.01, 1.0)), pressure=pr)
    # The kernel's own calibration reads 1 s of its control arrays whatever `dur` is, so a shorter
    # render walks off their end (access violation): always render >= 1 s and crop.
    dr = max(d, 1.0)
    y = wg.hurdy_gurdy(f0, dr, np.random.default_rng(7), strings=("trompette",), **kw)
    if drone > 0:
        y = y + _c01(drone) * wg.hurdy_gurdy(f0, dr, np.random.default_rng(8), strings=("bourdon",), **kw)
    y = y[:_n(d)].copy()
    k = _n(0.08)
    y[-k:] *= np.linspace(1.0, 0.0, k)
    return _inst("hurdy_gurdy", pc.lowpass(y, 4000.0), sr, vel)


@instrument("wind_chime", "One tube of a metal wind chime: a struck free-hanging bar with its inharmonic overtones ringing for seconds.",
            family="mallet", span=("C5", "C7"), decay=(3.0, "s T60 of the fundamental (square tubes ring up to 20 s)"), bright=(1.0, "0.4 soft striker .. 2.5 hard"),
            tube=("bar", "bar: free-bar overtones | square: measured square tubes 1 : 2.67 : 5.04 : 8"),
            law=("ratio", "ratio: T60 = decay / overtone ratio | rayleigh: aluminium Rayleigh damping, T60 ~ 1/f^2"),
            dip=(0.0, "0..1 share of the tube dipped in water: added mass lowers it about 3.5 semitones when full, and damps it"))
def wind_chime(freq, dur, sr=DEFAULT_SR, vel=1.0, decay=3.0, bright=1.0, tube="bar", law="ratio", dip=0.0):
    if tube not in TUBES or law not in ("ratio", "rayleigh"):
        raise ValueError("wind_chime: tube bar|square, law ratio|rayleigh")
    dec = float(np.clip(decay, 0.1, 20.0 if tube == "square" else 12.0))
    freq, dec = dipped(float(freq), dec, dip)
    y = _tube(float(freq), int(sr * (max(float(dur), 0.05) + min(dec, 4.0))), sr, dec, float(np.clip(bright, 0.1, 4.0)), None, tube, law)
    k = int(0.3 * len(y))
    y[-k:] *= pc.pow_decay(k, 2.0)
    return _inst("wind_chime", y, sr, vel, src=sr)


DIP_ALPHA = 0.5


def dipped(f: float, t60: float, dip: float):
    """A bar dipped in water: f / sqrt(1 + alpha dip), alpha 0.414 (-3 semitones) .. 0.587 (-4), T60 / (1 + 3 dip)."""
    d = float(np.clip(dip, 0.0, 1.0))
    return f / math.sqrt(1.0 + DIP_ALPHA * d), t60 / (1.0 + 3.0 * d)


@sfx("hummer", "Hummer, the corrugated tube whirled overhead: spin faster and it jumps up the tube's harmonics, wavering as its ends swing.",
     spin=(1.5, "turns a second; may be a ramp from spin to spin2"), spin2=(None, "turns a second at the end"), length=(0.8, "m"), dur=(5.0, "s"), seed=(0, "variation"))
def hummer(sr=DEFAULT_SR, spin=1.5, spin2=None, length=0.8, dur=5.0, seed=0):
    """80 cm, 3 cm across; the corrugations whistle at Strouhal St = f p / U (St 0.35, pitch p 5 mm, flow U = 0.7 of the
    tip speed) and lock on to a harmonic n c / 2L; the vibrato is the two ends interfering plus Doppler."""
    rng = np.random.default_rng(int(seed))
    n = _n(float(np.clip(dur, 0.2, 60.0)))
    t = np.arange(n) / P
    L = float(np.clip(length, 0.2, 3.0))
    w = np.linspace(float(spin), float(spin if spin2 is None else spin2), n)
    tip = 2.0 * np.pi * w * L
    k = np.maximum(np.round(0.35 * 0.7 * tip / 0.005 * 2.0 * L / 343.0), 1.0)
    rev = 2.0 * np.pi * np.cumsum(w) / P
    f = amb.lop(k * 343.0 / (2.0 * L), 20.0) * (1.0 + tip / 343.0 * np.sin(rev))
    ph = 2.0 * np.pi * np.cumsum(f) / P
    y = (np.sin(ph) + 0.12 * np.sin(2.0 * ph)) * (0.65 + 0.35 * np.cos(rev)) + 0.01 * rng.standard_normal(n)
    return _fin(y, sr, 0.6)


@sfx("cuica", "Cuica, the friction drum: a wet cloth rubbing a stick fixed to the skin, stick-slip through the skin's modes; the other hand presses it and bends the note.",
     freq=(450.0, "Hz of the skin at rest"), bend=(0.6, "how far the press raises it, 0..1.5"), strokes=(4, "rubs"), rate=(3.0, "rubs a second"), seed=(0, "variation"))
def cuica(sr=DEFAULT_SR, freq=450.0, bend=0.6, strokes=4, rate=3.0, seed=0):
    """Stick-slip on a stick at the membrane's centre; the finger picks the mode."""
    rng = np.random.default_rng(int(seed))
    r = max(float(rate), 0.3)
    k = int(np.clip(strokes, 1, 32))
    n = _n(k / r + 0.1)
    t = np.arange(n) / P
    u = (t * r) % 1.0
    press = np.sin(np.pi * np.minimum(t * r / k, 1.0))
    f = float(freq) * (1.0 + float(bend) * press) * (1.0 + 0.08 * u)
    env = np.sin(np.pi * u) ** 0.5 * (t * r < k)
    x = O.saw(f, n, P) + 0.05 * rng.standard_normal(n)
    y = amb.bp(x, float(freq) * 1.1, 2.0) + 0.5 * amb.bp(x, float(freq) * 2.3, 3.0)      # membrane 1.59, 2.14, 2.30: the second skin mode with a finger on it
    return _fin(y * env, sr, 0.6)


# ---------------------------------------------------------------- sci-fi FM, Noizer, overblown flue
def _glide(n: int, a: float, b: float) -> np.ndarray:
    return a * (b / a) ** np.linspace(0.0, 1.0, n)


def _pm(fc: np.ndarray, ratio: float, index: float, sr: int) -> np.ndarray:
    ph = 2 * np.pi * np.cumsum(fc) / sr
    return np.sin(ph + index * np.sin(ratio * ph))


SCIFI_FM = {      # kind -> seconds; the rest is in scifi_fm (genre convention, not physics)
    "implode": 0.554,
    "fission": 0.6,
    "thrust": 0.8,
}


@sfx("scifi_fm", "Genre-convention sci-fi FM effects: implode (falling FM, ring-mod and a vent), fission (two splitting FM voices), thrust (opening noise jet over a rising FM).",
     kind=("implode", "|".join(SCIFI_FM)), freq=(440.0, "base frequency f0: Hz or note name"), dur=(None, "s (default: the length of the kind)"), seed=(0, "variation"))
def scifi_fm(sr=DEFAULT_SR, kind="implode", freq=440.0, dur=None, seed=0):
    """Envelope (1 - t)^2. implode: carrier f0 -> 0.18 f0, ratio 1.37, index 3.2, ring-mod at 0.51 f0, triangle
    0.5 f0 -> 0.16 f0, noise vent low-pass 5600 -> 290 Hz. fission: f0 -> 1.32 f0 and 1.005 f0 -> 1.49 f0, ratio 2,
    index 1.8, ring-mods at 0.74 and 0.76 f0. thrust: white noise low-pass 590 -> 16200 Hz over FM 180 -> 1150 Hz,
    ratio 1.98, index 1.4."""
    kind = kind if kind in SCIFI_FM else "implode"
    rng = np.random.default_rng(int(seed))
    f0 = float(np.clip(to_hz(freq), 30.0, 4000.0))
    d = float(np.clip(SCIFI_FM[kind] if dur is None else dur, 0.05, 10.0))
    n = int(d * sr)
    t = np.arange(n) / sr
    env = (1.0 - np.arange(n) / n) ** 2
    if kind == "implode":
        y = _pm(_glide(n, f0, 0.18 * f0), 1.37, 3.2, sr) * np.sin(2 * np.pi * 0.51 * f0 * t)
        tr = O.triangle(_glide(n, 0.5 * f0, 0.16 * f0), n, sr)
        vent = F.lowpass(rng.uniform(-1, 1, n), _glide(n, 5600.0, 290.0), sr)
        y = y + 0.5 * tr + 0.3 * vent
    elif kind == "fission":
        y = (_pm(_glide(n, f0, 1.32 * f0), 2.0, 1.8, sr) * np.sin(2 * np.pi * 0.74 * f0 * t)
             + _pm(_glide(n, 1.005 * f0, 1.49 * f0), 2.0, 1.8, sr) * np.sin(2 * np.pi * 0.76 * f0 * t))
    else:
        jet = F.lowpass(rng.uniform(-1, 1, n), _glide(n, 590.0, min(16200.0, 0.45 * sr)), sr)
        y = jet + 0.5 * _pm(_glide(n, 180.0, 1150.0), 1.98, 1.4, sr)
    return _fin(y * env, sr, 0.8, src=sr, fade=0.002)


@numba.njit(cache=True)
def _fb_sine(f: np.ndarray, fb: float, sr: float) -> np.ndarray:
    y = np.empty(len(f))
    ph, prev = 0.0, 0.0
    for i in range(len(f)):
        ph += 2.0 * np.pi * (f[i] + fb * prev) / sr
        prev = math.sin(ph)
        y[i] = prev
    return y


@numba.njit(cache=True)
def _phaser4(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    s = np.zeros(4)
    y = np.empty(len(x))
    for i in range(len(x)):
        v = x[i]
        for k in range(4):
            w = -c[i] * v + s[k]
            s[k] = v + c[i] * w
            v = w
        y[i] = 0.5 * (x[i] + v)
    return y


def _adsr_ms(n: int, sr: int, a: float, d: float, s: float, r: float) -> np.ndarray:
    t = np.arange(n) / sr * 1000.0
    hold = max(n / sr * 1000.0 - r, a)
    up = np.clip(t / max(a, 1e-3), 0.0, 1.0)
    dec = s + (1.0 - s) * np.exp(-np.maximum(t - a, 0.0) / max(d / 4.6, 1e-3))
    env = np.where(t < a, up, dec)
    return env * np.clip(1.0 - (t - hold) / max(r, 1e-3), 0.0, 1.0)


def _oct_q(bw: float) -> float:
    b = 2.0 ** max(float(bw), 1e-3)
    return math.sqrt(b) / (b - 1.0)


# preset -> voices [(source, value, feedback Hz per unit, bp centre Hz, bp octaves, level, (A, D, S, R) ms)],
#           output filter (Q, centre Hz x (env2 + 0.1), lfo Hz, lfo depth Hz) or None, phaser depth, output gain
NOIZER = {
    "fire2":        ([("pink", 0.315, 0.0, 412.0, 2.0, 0.29, (10, 1000, 0.7, 500)),
                      ("fm", 128.0, 300.0, 128.0, 1.47, 0.13, (10, 1000, 0.7, 500))], None, 0.0, 1.0),
    "thundery":     ([("sh", 79.0, 0.0, 128.0, 1.07, 0.71, (10, 1638, 0.0, 2976)),
                      ("fm", 231.0, 300.0, 231.0, 1.0, 0.3, (10, 1638, 0.0, 2976))], None, 0.0, 1.0),
    "wind_rain":    ([("pink", 0.0, 0.0, 1000.0, 3.0, 0.6, (400, 2000, 0.8, 1500))], (9.95, 3853.0, 0.33, 600.0), 0.0, 1.0),
    "woody":        ([("sh", 5000.0, 0.0, 264.0, 0.02, 0.6, (2, 206, 0.0, 206)),
                      ("sh", 5000.0, 0.0, 165.0, 0.02, 0.6, (2, 260, 0.0, 260)),
                      ("sh", 5000.0, 0.0, 115.0, 0.01, 0.6, (2, 156, 0.0, 156))], None, 0.0, 1.0),
    "helicopter":   ([("sh", 5000.0, 0.0, 791.0, 1.74, 0.449, (10, 1000, 1.0, 300)),
                      ("fm", 148.0, 3300.0, 148.0, 2.51, 0.386, (10, 1000, 1.0, 300))], (2.94, 709.0, 11.93, 1158.0), 0.0, 0.22),
    "steamtrain":   ([("fm", 109.0, 960.0, 109.0, 0.795, 0.291, (110, 2048, 0.598, 1155))], (2.63, 7720.0, 4.14, 3000.0), 0.0, 1.086),
    "flamethrower": ([("sh", 315.0, 0.0, 59.6, 3.79, 0.606, (50, 1000, 0.8, 600)),
                      ("fm", 109.0, 300.0, 109.0, 1.0, 0.465, (50, 1000, 0.8, 600))], (0.58, 160.0, 0.0, 0.0), 0.449, 1.0),
    "rain":         ([("pink", 0.0, 0.0, 3000.0, 3.0, 0.6, (300, 1000, 0.9, 1500)),
                      ("sh", 5000.0, 0.0, 6000.0, 1.0, 0.25, (300, 1000, 0.9, 1500))], None, 0.0, 1.0),
    "surf":         ([("pink", 0.0, 0.0, 600.0, 2.5, 0.7, (1500, 3000, 0.6, 3000))], (1.5, 1500.0, 0.1, 900.0), 0.0, 1.0),
    "engine":       ([("fm", 55.0, 600.0, 110.0, 1.5, 0.5, (100, 1000, 1.0, 500)),
                      ("sh", 220.0, 0.0, 440.0, 1.0, 0.3, (100, 1000, 1.0, 500))], None, 0.3, 1.0),
}


@sfx("noizer", "Three-voice noise synth with named presets: stepped noise, crunched pink noise and a feedback-FM sine, each through its band-pass and envelope, then a phaser and a swept resonant filter.",
     preset=("fire2", "|".join(NOIZER)), dur=(2.0, "s"), seed=(0, "variation"))
def noizer(sr=DEFAULT_SR, preset="fire2", dur=2.0, seed=0):
    """Voice sources: sample-and-hold noise at a rate (Hz); pink noise through a dead zone x - clip(x, -t, t),
    made up by 2 + 36 t^2 and high-passed at 5 Hz; a sine whose frequency is f + fb y (feedback FM). Each voice
    goes through an RBJ band-pass (bandwidth in octaves) and an ADSR. Phaser: 4 first-order all-passes,
    c = 0.97 - env1 x depth. Output filter: resonant band-pass Q, centre (env2 + 0.1) x base + LFO."""
    voices, vcf, phaser, gain = NOIZER.get(str(preset), NOIZER["fire2"])
    rng = np.random.default_rng(int(seed))
    d = float(np.clip(dur, 0.1, 60.0))
    n = int(d * sr)
    y = np.zeros(n)
    envs = []
    for src, val, fb, fc, bw, lvl, adsr_ms in voices:
        if src == "sh":
            x = O.held_noise(n, min(val, sr / 2), sr, seed=int(rng.integers(1 << 30)))
        elif src == "pink":
            x = O.pink(n, int(rng.integers(1 << 30)))
            x = x / (np.max(np.abs(x)) + 1e-12)
            if val > 0:
                x = (x - np.clip(x, -val, val)) * (2.0 + 36.0 * val * val)
                x = F.highpass(x, 5.0, sr)
        else:
            x = _fb_sine(np.full(n, float(val)), float(fb), float(sr))
        env = _adsr_ms(n, sr, *adsr_ms)
        envs.append(env)
        y += lvl * env * F.bandpass(x, min(fc, 0.45 * sr), sr, _oct_q(bw))
    if phaser > 0:
        y = _phaser4(y, np.clip(0.97 - envs[0] * phaser, -0.99, 0.99))
    if vcf is not None:
        q, base, lfo_hz, lfo_amt = vcf
        e2 = envs[1] if len(envs) > 1 else envs[0]
        fc = (e2 + 0.1) * base + lfo_amt * np.sin(2 * np.pi * lfo_hz * np.arange(n) / sr)
        y = F.biquad(y, "bandpass", np.clip(fc, 20.0, 0.45 * sr), sr, q=max(q, 0.3))
    return _fin(y * gain, sr, 0.8, src=sr, fade=0.005)


FLUE_UP, FLUE_DOWN, FLUE_AEOLIAN = 311.5, 259.0, 224.0   # Pa: register 1 -> 2, 2 -> 1, below it the jet only whistles
FLUE_ALPHA = 0.185                                         # sets how far a fast pressure ramp overshoots the thresholds


def flue_register(p: np.ndarray, sr: int, up: float = FLUE_UP, down: float = FLUE_DOWN, aeolian: float = FLUE_AEOLIAN,
                  alpha: float = FLUE_ALPHA) -> np.ndarray:
    """Register (0 aeolian, 1, 2) of a flue pipe blown at pressure ``p`` (Pa): the jump up happens at
    up + Pcorr, the fall back at down - Pcorr, Pcorr = (3 |dP/dt| / (2 alpha))^(2/3), so a fast crescendo
    overshoots and a fast diminuendo undershoots (dynamic hysteresis)."""
    p = np.asarray(p, float)
    rate = np.gradient(F.lowpass(p, 20.0, sr)) * sr
    corr = (3.0 * np.abs(rate) / (2.0 * alpha)) ** (2.0 / 3.0)
    reg = np.empty(len(p), dtype=np.int64)
    r = 1
    for i in range(len(p)):
        if r == 2 and p[i] < down - (corr[i] if rate[i] < 0 else 0.0):
            r = 1
        elif r < 2 and p[i] > up + (corr[i] if rate[i] > 0 else 0.0):
            r = 2
        reg[i] = 0 if (r == 1 and p[i] < aeolian) else r
    return reg


@sfx("overblow", "Recorder-like flue pipe blown through a pressure swell: it whistles weakly at low pressure, sounds its note, jumps to the octave past a threshold and falls back lower down (hysteresis, wider for fast swells).",
     note=("G4", "fingered note"), peak=(400.0, "Pa at the top of the swell"), dur=(3.0, "s"), seed=(0, "variation"))
def overblow(sr=DEFAULT_SR, note="G4", peak=400.0, dur=3.0, seed=0):
    rng = np.random.default_rng(int(seed))
    d = float(np.clip(dur, 0.3, 30.0))
    n = int(d * sr)
    p = float(np.clip(peak, 50.0, 2000.0)) * np.sin(np.pi * np.arange(n) / n)
    reg = flue_register(p, sr)
    f0 = float(to_hz(note))
    f = f0 * np.where(reg == 2, 2.0, 1.0)
    ph = 2 * np.pi * np.cumsum(f) / sr
    lvl = np.sqrt(p / max(float(np.max(p)), 1e-9))
    tone = lvl * (reg > 0) * (np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.1 * np.sin(3 * ph))
    breath = F.bandpass(rng.standard_normal(n), min(4.0 * f0, 0.4 * sr), sr, 2.0)
    whistle = (reg == 0) * lvl * 0.5 * F.bandpass(rng.standard_normal(n), min(6.0 * f0, 0.4 * sr), sr, 30.0)
    y = tone + 0.05 * breath * lvl + whistle
    return _fin(y, sr, 0.7, src=sr, fade=0.02)
