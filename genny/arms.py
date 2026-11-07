"""Firearms and everything around them: named weapons, handling, spent brass, bullet impacts, ricochets, passing bullets.

The shot keeps Farnell's Practical 30 layers (`foley._gunshot`), adds a Friedlander pressure wave and a gas tail (`_body_blast`), and mixes in pascals, on measured levels:
- muzzle blast, peak on axis at 3 m and front/back directivity: Maher 2019 (AES, shot-to-shot), Maher & Shaw 2010 (AES 39);
- N-wave: Whitham's far field as in Flamme & Murphy, Int. J. Audiology 2019, eq. 1 (coefficients 0.53 and 1.82);
- arrival of the shock: t = x/V + y cos(theta_M)/c, Maher 2006 (IEEE DSP), Tables 1-2;
- suppressor: Murphy et al., POMA 33, 2019 (15-30 dB, mostly 160-1000 Hz, shock untouched).
Bullet lengths, the levels of rounds nobody measured, and all handling times except two (see MECH) are
estimates where no published measurement exists. Spec with every source: reports/sons/armes_feu.md, rechargement_douilles.md.
"""
from __future__ import annotations

import math

import numpy as np

from . import contact as K
from .core import DEFAULT_SR
from .foley import (C_AIR, CATCH, ENDSTOP, P, RATCHET, RIFLE_BODY, _barrel, _bpar8, _c01, _fin, _mclick, _n, _nm,
                    _noiseburst, _put, _shell, _strike)
from .physical import ambience as amb
from .physical import modal, space
from .sfx import sfx

P0 = 101325.0

# diameter mm, length mm, speed m/s, blast peak Pa on axis at 3 m, directivity A dB (0 = none, see _blast_gain)
# Peaks: Maher 2019 (9x19, .45, .357, 5.56, .308). A: Maher & Shaw 2010. Speeds: spec table A. Marked rows are estimates:
# peak: .22 (Maher & Shaw: 25 dB under .308), .40, 10 mm, .38, 7.62x39, .300, .50, 12 ga, musket.
CALIBERS = {
    "22lr": (5.7, 11.0, 390.0, 380.0, 30.0), "22lr_sub": (5.7, 11.0, 320.0, 300.0, 30.0),
    "9x19": (9.0, 15.5, 360.0, 3307.0, 15.0), "9x19_sub": (9.0, 18.0, 297.0, 2600.0, 15.0),
    "45acp": (11.5, 17.0, 255.0, 5123.0, 10.0), "40sw": (10.2, 16.0, 340.0, 4000.0, 15.0),
    "10mm": (10.2, 16.0, 400.0, 4000.0, 12.0), "357": (9.1, 16.0, 380.0, 6491.0, 15.0),
    "38spl": (9.1, 16.0, 290.0, 3500.0, 10.0), "556": (5.7, 23.0, 948.0, 8137.0, 15.0),
    "762x39": (7.9, 26.0, 715.0, 6780.0, 15.0), "762x51": (7.8, 29.0, 853.0, 6780.0, 15.0),
    "308": (7.8, 29.0, 802.0, 6780.0, 15.0), "762x54": (7.9, 29.0, 825.0, 6780.0, 15.0),
    "300blk": (7.8, 28.0, 675.0, 6000.0, 15.0), "300blk_sub": (7.8, 35.0, 310.0, 3000.0, 15.0),
    "50bmg": (12.9, 58.0, 928.0, 12000.0, 8.0), "12_buck": (8.4, 8.4, 404.0, 6780.0, 16.0),
    "12_slug": (18.5, 20.0, 396.0, 6780.0, 16.0), "musket": (19.0, 19.0, 300.0, 3000.0, 12.0),
}
# caliber, rounds/s (0 = one at a time), body scale, blast ms (93 % of the energy), options
WEAPONS = {
    "pistol_9mm": ("9x19", 0, 1.4, 0.78, {}), "pistol_9mm_sub": ("9x19_sub", 0, 1.4, 0.78, {}),
    "pistol_45": ("45acp", 0, 1.2, 0.77, {}), "pistol_40": ("40sw", 0, 1.3, 0.75, {}),
    "pistol_10mm": ("10mm", 0, 1.3, 0.75, {}), "pistol_22": ("22lr", 0, 1.6, 0.33, {}),
    "machine_pistol": ("9x19", 22.5, 1.4, 0.78, {"open_bolt": True}),
    "revolver_357": ("357", 0, 1.3, 0.98, {"gap": True, "manual": True}),
    "revolver_38": ("38spl", 0, 1.35, 0.84, {"gap": True, "manual": True}),
    "smg_mp5": ("9x19", 13.3, 1.25, 0.78, {}), "smg_mp5sd": ("9x19_sub", 13.3, 1.25, 0.78, {"suppressed": 1.0}),
    "smg_uzi": ("9x19", 10.0, 1.25, 0.78, {"open_bolt": True}),
    "smg_45": ("45acp", 12.0, 1.15, 0.77, {"open_bolt": True}),
    "ar15": ("556", 13.0, 1.1, 0.72, {}), "m4_short": ("556", 13.0, 1.15, 0.72, {"boost_db": 2.0}),
    "ak47": ("762x39", 10.0, 1.0, 1.93, {"wood": True}), "battle_rifle": ("762x51", 10.0, 1.0, 1.93, {}),
    "m1_garand": ("762x51", 0, 0.95, 1.93, {"wood": True, "clip": 8, "cadence": 0.75}), "blackout": ("300blk", 13.0, 1.05, 1.93, {}),
    "blackout_sub": ("300blk_sub", 13.0, 1.05, 1.93, {}),
    "sniper_308": ("308", 0, 0.95, 1.93, {"manual": True}),
    "sniper_50": ("50bmg", 0, 0.5, 4.0, {"manual": True, "brake": True}),
    "sniper_22": ("22lr_sub", 0, 1.6, 0.33, {"manual": True}),
    "shotgun_buck": ("12_buck", 0, 0.75, 2.56, {"pellets": 9, "manual": True}),
    "shotgun_slug": ("12_slug", 0, 0.75, 2.56, {"manual": True}),
    "shotgun_short": ("12_buck", 0, 0.8, 2.56, {"pellets": 9, "manual": True, "boost_db": 2.0}),
    "lmg_m249": ("556", 14.2, 0.9, 0.72, {"open_bolt": True}), "gpmg_pkm": ("762x54", 11.7, 0.8, 1.93, {"open_bolt": True}),
    "hmg_m2": ("50bmg", 9.0, 0.5, 4.0, {}), "minigun": ("762x51", 50.0, 0.9, 1.93, {"manual": True}),
    "flintlock": ("musket", 0, 0.6, 60.0, {"manual": True, "lock": (35.0, "flint")}),
    "matchlock": ("musket", 0, 0.6, 60.0, {"manual": True, "lock": (60.0, "match")}),
    "percussion": ("musket", 0, 0.6, 50.0, {"manual": True, "lock": (5.0, "cap")}),
}
# Barrel mm, spec section 4 (Wikipedia pages listed there). Detonation to muzzle = barrel / v, 1-2 ms (Farnell, c14 l.24523);
# weapons not listed: 1.5 ms (klang.py:931).
BARREL_MM = {"pistol_9mm": 118.0, "machine_pistol": 117.0, "smg_uzi": 260.0, "ar15": 508.0, "ak47": 415.0, "battle_rifle": 559.0, "m1_garand": 610.0,
             "blackout": 406.0, "sniper_50": 1143.0, "lmg_m249": 465.0, "gpmg_pkm": 605.0, "hmg_m2": 1143.0, "minigun": 559.0,
             # spec table B and section 4: MS10 test guns, W-Bess
             "pistol_22": 508.0, "pistol_45": 127.0, "revolver_357": 76.0, "revolver_38": 51.0, "pistol_10mm": 117.0, "sniper_308": 610.0,
             "shotgun_buck": 457.0, "shotgun_slug": 457.0, "shotgun_short": 356.0, "flintlock": 990.0}
BODY_Q = 10.62       # Farnell's preset
TAIL_GAIN = 0.04
# calibre -> muzzle energy J, peak chamber pressure MPa. Spec section 4: W-9x19, W-.45, W-.357, W-5.56, W-7.62, W-.300, W-.50, W-12, Fed, W-Arq.
# Calculated from the spec's mass and speed: .22 (40 gr), .308 (168 gr), .40 (165 gr), 10 mm (180 gr), .38 (125 gr).
# energy: 7.62x39 (8 g), 7.62x54 (9.6 g), 12 buck (34 g). pressure: .40, 10 mm, .38, 7.62x39, 7.62x54, musket.
CHARGE = {"22lr": (197.0, 165.0), "22lr_sub": (133.0, 165.0), "9x19": (481.0, 235.0), "9x19_sub": (420.0, 235.0), "45acp": (483.0, 135.0),
          "40sw": (618.0, 235.0), "10mm": (933.0, 235.0), "357": (731.0, 270.0), "38spl": (340.0, 150.0), "556": (1797.0, 430.0),
          "762x39": (2045.0, 355.0), "762x51": (3469.0, 415.0), "308": (3505.0, 415.0), "762x54": (3267.0, 390.0), "300blk": (1840.0, 380.0),
          "300blk_sub": (675.0, 380.0), "50bmg": (18050.0, 395.0), "12_buck": (2775.0, 79.0), "12_slug": (2300.0, 79.0), "musket": (1500.0, 30.0)}
REF_PA, REF_PEAK = 12000.0, 0.7
IMPACT_REF_J = 18050.0             # the .50 BMG again: the hardest hit sits at 0.85
# His body rings "several hundred ms" (c14 l.24633): T60 300 ms at 230 Hz. c14 computes Q 31 for it, but amb.bp measures T60 = 1.1 Q / f (Q 31: 148 ms): Q 62.
POV_Q = 62.0


def n_wave_pa(v: float, d_mm: float, l_mm: float, miss: float, c: float = C_AIR):
    """Whitham N-wave of a bullet: ``(peak Pa, duration s)`` at distance ``miss`` m from the trajectory, or
    ``(0, 0)`` when subsonic. Far field only (miss > 1000 d); unreliable close to Mach 1."""
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
    d, _, v, _, _ = CALIBERS[cal]
    m = 2.0 * CHARGE[cal][0] / (v * v)
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


def _crack(v, d_mm, l_mm, miss, rng, height=1.5, wake=True) -> np.ndarray:
    """The passing bullet in Pa: N-wave, ground reflection (M06: dt = path difference / c) and
    the broadband wake behind it (2 kHz up, 30 ms, -20 dB)."""
    p, t = n_wave_pa(v, d_mm, l_mm, miss)
    if p <= 0:
        return np.zeros(1)
    b = max(miss, 0.05)
    dt = (math.hypot(b, 2.0 * height) - b) / C_AIR
    y = np.zeros(_n(dt + t + 0.05))
    w = _n_wave_shape(t)
    y[:len(w)] += w
    _put(y, amb.lop(w, 6000.0), dt, 0.6)
    if wake:
        m = _n(0.03)
        _put(y, amb.hip(rng.uniform(-1, 1, m), 2000.0) * (1.0 - np.arange(m) / m) ** 2, t, 0.1)
    return y * p


# Measured muzzle-blast peaks of a rifle every 15 degrees, dB re the axis; 180 continues the last slope
BLAST_SHAPE_DB = (0.0, 0.1, -0.6, -0.9, -2.6, -4.0, -11.1, -12.2, -16.1, -19.3, -23.6, -26.8, -30.0)


def _blast_gain(a_db: float, azimuth: float) -> float:
    """0 in front, -A behind, the measured shape between, scaled to A."""
    s = float(np.interp(azimuth, np.arange(0.0, 181.0, 15.0), BLAST_SHAPE_DB)) / -BLAST_SHAPE_DB[-1]
    return 10.0 ** (a_db * s / 20.0)


def _friedlander(a: float, n: int) -> np.ndarray:
    """Blast wave at a point: (1 - t/a) e^(-t/a), a = positive phase; the negative phase lasts over twice as long
    ("at least twice", Kinney & Graham, Explosive Shocks in Air, ch. Duration). Decay constant 1."""
    u = np.arange(n) / (P * a)
    return (1.0 - u) * np.exp(-u)


def _gas_tail(t60: float, rng) -> np.ndarray:
    """Gas still leaving and mixing after the front: decaying low-passed noise. Band and shape."""
    n = _n(t60)
    return amb.lop(rng.uniform(-1, 1, n), 2500.0) * np.exp(-6.9 * np.arange(n) / n) * (1.0 - np.arange(n) / n)


def _body_blast(cal, body, blast_ms, opts, rng, action, cycle, close, echo, delay=1.5e-3, ring=1.0, direct=0.35, apart=False, barrel=None, pov=False,
                weight=0.0, texture=0.0):
    """The muzzle blast, peak 1. Heard from outside: the Friedlander wave (``blast_ms`` holds 93 % of its energy) and a short gas
    tail, no body; ``weight`` ms (included in ``blast_ms``) lengthen the wave past its 8 ms cap; ``texture`` adds Farnell's
    layers (chirp, barrel, noise burst) at that share of the peak. ``pov``: Farnell's layers through the gun's body and tanh, ``ring`` scaling the body's Q.
    ``apart``: also returns the mechanics alone, for what a can does not touch."""
    b = blast_ms * 1e-3
    joules, mpa = CHARGE[cal]
    scale = (joules / CHARGE["556"][0]) ** (1.0 / 3.0)
    n = _n(0.4 * max(ring, 1.0) + 2.0 * b + (0.5 if echo > 0 else 0.0))
    x = np.zeros(n)
    mech = np.zeros(n)
    t0 = 0.0
    if opts.get("gap"):                                     # cylinder gap leads the muzzle by ~200 us (Maher & Routh 2017)
        _put(x, _barrel(max(0.6 * b, 1e-3)), 0.0, 0.5)
        t0 = 2e-4
    _put(x, _shell(max(b, 1e-3)), t0, 0.6)
    _put(x, _barrel(max(0.6 * b, 1e-3)), t0 + delay, 0.5)
    a = min(b, 8e-3) / 2.28                                  # 93 % of the energy of (1-u) e^-u lies within u < 2.28: the measure in NIJ Fig. 6
    _put(x, _friedlander(a, min(n, int(12 * a * P))), t0 + delay, 0.5)
    _put(x, _barrel(8.0 / (100.0 * body)), t0 + delay, min(0.25, 0.1 * scale))
    _put(x, _gas_tail(max(0.005 * scale, b), rng), t0 + delay, 0.25)
    _put(x, _noiseburst(max(0.6 * b, 1e-3), rng), t0 + delay + 0.35 * b, 0.8)
    if opts.get("brake"):
        _put(x, _noiseburst(2.0 * b, rng), t0 + delay + 1e-3, 0.8)
    if action > 0 and close > 0 and not opts.get("manual"):
        nz = amb.hip(rng.uniform(-1, 1, n), 300.0)
        c = min(cycle, 0.1)
        mech += 0.5 * action * close * (_mclick(nz, CATCH, 0.45 * c, 0.33) + _mclick(nz, ENDSTOP, 0.0 if opts.get("open_bolt") else 0.7 * c, 0.33))
    if opts.get("wood") and action > 0 and close > 0:       # "wooden knocking sound about 90 ms after the impulse" (Farnell)
        _put(mech, _nm(_strike("sy_stick", 0.3, 1.5, 0.0, rng, dur=0.12)), 0.09, 0.1 * action * close)
    if not pov:
        y = np.zeros(n)
        w = weight * 1e-3
        ad = (min(b - w, 8e-3) + min(w, 0.1)) / 2.28 - (2.5e-4 / 2.28 if opts.get("gap") else 0.0)    # 0.25: the lead and the pre-pulse's own energy
        f = _friedlander(ad, min(n, int(40 * ad * P)))
        for t, g in ((t0 + delay, 1.0),) + (((t0 + delay - 2e-4, 0.5),) if opts.get("gap") else ()) + (((t0 + delay + 1e-3, 0.5),) if opts.get("brake") else ()):
            _put(y, f, t, g)
        _put(y, _gas_tail(max(0.005 * scale, b), rng), t0 + delay, TAIL_GAIN)
        y = y / np.abs(y).max()
        if texture > 0:
            y = y + texture * x / (np.abs(x).max() + 1e-12)
        if not apart:
            y = y + mech
        if echo > 0:
            y = y + echo * 4.0 * (amb.outdoor_echoes(y) - 0.2 * y)
        return (y, mech) if apart else y
    centres = [f * body * rng.uniform(0.97, 1.03) for f in RIFLE_BODY]
    q = BODY_Q * ring
    if barrel:
        centres.append(C_AIR / (2.0 * barrel * 1e-3))
        q *= (barrel / 508.0) ** 0.25
    drive = 4.8 * math.sqrt(mpa / CHARGE["556"][1])

    def voice(u):
        return np.tanh(drive * (ring * _bpar8(u, centres, q=q) + direct * u))
    y = voice(x if apart else x + mech)
    if echo > 0:
        y = y + echo * 4.0 * (amb.outdoor_echoes(y) - 0.2 * y)
    return (_nm(y), voice(mech)) if apart else _nm(y)


def _suppress(y, db):
    """Murphy 2019: the loss sits at 160-1000 Hz; 8 dB of ``db`` as a band dip there."""
    y = y * 10.0 ** (-(db - 8.0) / 20.0)
    return y - 0.6 * amb.bp(y, 400.0, 0.7)


def _shot(weapon, rng, r, az, miss, suppress, subsonic, action, crack, echo, cycle, ring=1.0, direct=0.35, weight=0.0, pov=False,
          texture=0.0) -> np.ndarray:
    cal, _, body, blast_ms, opts = WEAPONS[weapon]
    blast_ms += weight
    d, l, v, p3, a = CALIBERS[cal]
    suppress = max(suppress, opts.get("suppressed", 0.0))
    if subsonic:
        v, p3 = min(v, 310.0), p3 * 10.0 ** (-4.0 / 20.0)
    if suppress > 0:
        v *= 1.03                                            # Murphy 2019: 0-6 % faster
    close = float(np.clip(1.0 - r / 100.0, 0.0, 1.0))        # "over 100 m [the body] is lost" (Farnell)
    acts = action * (2.0 if suppress > 0 else 1.0)
    barrel = BARREL_MM.get(weapon)
    delay = barrel * 1e-3 / v if barrel else 1.5e-3
    if suppress > 0:                                         # the can does not touch the bolt or the ejection port (Mu19): C7
        blast, mech = _body_blast(cal, body, blast_ms, opts, rng, acts, cycle, close, echo, delay, ring, direct, True, barrel, pov, weight, texture)
        blast = _suppress(blast, 25.0 * suppress) + mech
    else:
        blast = _body_blast(cal, body, blast_ms, opts, rng, acts, cycle, close, echo, delay, ring, direct, False, barrel, pov, weight, texture)
    blast = space.distance(blast, r, ref=1.0) * p3 * 3.0 * _blast_gain(a, az) * 10.0 ** (opts.get("boost_db", 0.0) / 20.0)
    th = math.radians(az)
    x, y = r * math.cos(th), r * math.sin(th)
    if miss > 0:
        y = float(miss)
    t_b = r / C_AIR
    t_c = shock_arrival(v, x, max(y, 0.05), C_AIR) if crack > 0 and az < 75.0 else None     # none past 75 deg (Murphy 2019)
    t0 = min(t_b, t_c) if t_c is not None else t_b
    out = np.zeros(len(blast) + _n(0.05))
    _put(out, blast, t_b - t0)
    if t_c is not None:
        bm = max(y, 0.05)
        for _ in range(int(opts.get("pellets", 1))):
            w = space.distance(_crack(v, d, l, bm, rng), bm + 1.0, ref=1.0) * (bm + 1.0)
            gain = crack / math.sqrt(float(opts.get("pellets", 1)))
            _put(out, w, t_c - t0 + rng.uniform(-2e-4, 2e-4) * (opts.get("pellets", 1) > 1), gain)
    return out


HEARD_WEIGHT_MS, HEARD_DRIVE, HEARD_MIX, HEARD_TEXTURE = 3.0, 30.0, 0.35, 0.6
RENDER_HELP = ("heard = as a recording or the ear gets it: the blast's weight, a microphone overloaded at 150 dB, the place's "
               "reflections | pressure = the bare pressure at the ear, a click of a few ms")
PLACE_HELP = "render=heard: where it is heard, none | street | forest | canyon | arena | cave | hall ... (room presets)"


def _heard_weight(cal: str) -> float:
    return HEARD_WEIGHT_MS * (CHARGE[cal][0] / CHARGE["556"][0]) ** (1.0 / 3.0)


def _heard(y: np.ndarray, place: str) -> np.ndarray:
    """Overload then the place: the peak is kept, so relative levels survive."""
    pk = float(np.max(np.abs(y))) + 1e-12
    y = np.tanh(HEARD_DRIVE * y / pk) / math.tanh(HEARD_DRIVE) * pk
    if str(place) != "none":
        from .reverbs import room
        w = room(np.concatenate([y, np.zeros(_n(2.0))]), P, mix=HEARD_MIX, preset=str(place))
        y = w.mean(axis=1) if w.ndim == 2 else w
    return y


def _check_render(render: str) -> bool:
    if render not in ("heard", "pressure"):
        raise ValueError(f"unknown render {render!r}: heard|pressure")
    return render == "heard"


def _finish(y, sr, r, level, fade, tail=0.0) -> np.ndarray:
    """``level="peak"``: 0.9 at the crest. ``"relative"``: pascals at the ear over the .50's on-axis blast at the same distance,
    so calibres keep their measured differences and distance only colours; never over 0.89."""
    if level == "peak":
        return _fin(y, sr, 0.9, fade=fade, tail=tail)
    if level != "relative":
        raise ValueError(f"unknown level {level!r}: relative|peak")
    y = _fin(y * REF_PEAK / (REF_PA * 3.0 / r), sr, None, fade=fade, tail=tail)
    return y * min(1.0, 0.89 / (float(np.max(np.abs(y))) + 1e-12))


def _clip_ping(rng) -> np.ndarray:
    """The M1's ping: a light steel clip at 2-3 kHz, three modes dying within 100 ms."""
    f = np.zeros(_n(0.2))
    f[0] = 1.0
    fm = np.array([2300.0, 3100.0, 4650.0]) * rng.uniform(0.98, 1.02)
    return _nm(modal.render_modes(f, modal.Modes(fm, np.array([25.0, 30.0, 40.0]), np.array([1.0, 0.6, 0.3]), "vdd")))


def _lock(kind: str, delay: float, rng) -> np.ndarray:
    """What precedes the charge of an old gun: the lock falling, then the priming flash for ``delay`` ms
    (20-50 ms flint, US 6374525; match and cap estimated)."""
    n = _n(delay * 1e-3 + 0.05)
    nz = amb.hip(rng.uniform(-1, 1, n), 300.0)
    y = _mclick(nz, ENDSTOP, 0.0, 0.5) + _mclick(nz, CATCH, 0.0, 0.5 if kind == "flint" else 0.2)
    m = _n(delay * 1e-3)
    if kind != "cap":
        _put(y, amb.hip(rng.uniform(-1, 1, m), 2500.0) * np.linspace(0.3, 1.0, m), 0.0, 0.25)
    return y


@sfx("firearm", "A named firearm in physical units: muzzle blast by calibre and direction, the bullet's own N-wave and where it reaches you, bolt cycling, burst rate, suppressor, subsonic rounds.",
     weapon=("ar15", "|".join(WEAPONS)), distance=(10.0, "m from the shooter"),
     azimuth=(30.0, "degrees off the line of fire: 0 you are shot at, 180 behind the shooter (10-30 dB quieter); no crack past 75"),
     miss=(0.0, "m the bullet passes from you (0 = from distance and azimuth)"), shots=(1, "rounds (up to 2000)"),
     rate=(0.0, "rounds/s (0 = the weapon's own)"), suppressor=(0.0, "0..1: 1 = -25 dB at the muzzle, mostly 160-1000 Hz; the crack stays"),
     subsonic=(False, "true = slowed to 310 m/s: no crack, a little quieter"), action=(0.6, "0..1 bolt cycling (lost beyond 100 m)"),
     crack=(1.0, "0..1 level of the N-wave against its true level"), echo=(0.0, "0..1 street / treeline echoes; Farnell keeps them out of the sound, use `room` or a recipe"),
     first_pop=(3.0, "dB louder on a suppressed gun's first round, half that on the second"),
     pov=(False, "true = the shooter's ears: at the breech, the body ringing 300 ms, less blast, the mechanism up, no crack"),
     ring=(1.0, "pov only: how long the gun's body rings, Q = 10.62 x ring (5.8 = T60 300 ms at 230 Hz)"),
     weight=(0.0, "ms added to the blast wave (0-200): a longer shot sounds heavier; from outside it stops at 100, the rest is infrasound"),
     level=("relative", "relative = weapons keep their measured differences (a .22 is 30 dB under a .50; needs normalize off in a spec) | peak = every file at 0.9"),
     render=("heard", RENDER_HELP), place=("street", PLACE_HELP), seed=(0, "variation"))
def firearm(sr=DEFAULT_SR, weapon="ar15", distance=10.0, azimuth=30.0, miss=0.0, shots=1, rate=0.0, suppressor=0.0,
            subsonic=False, action=0.6, crack=1.0, echo=0.0, first_pop=3.0, pov=False, ring=1.0, weight=0.0, level="relative",
            render="heard", place="street", seed=0):
    weapon = str(weapon)
    if weapon not in WEAPONS:
        raise ValueError(f"unknown weapon {weapon!r}: {'|'.join(WEAPONS)}")
    heard = _check_render(str(render))
    rng = np.random.default_rng(int(seed))
    r = float(np.clip(distance, 1.0, 5000.0))
    az = float(np.clip(abs(float(azimuth)), 0.0, 180.0))
    shots = int(np.clip(shots, 1, 2000))
    ring, direct = float(np.clip(ring, 0.5, 6.0)), 0.35
    if pov:
        r, az, crack, action, ring, direct = 1.0, 180.0, 0.0, _c01(action) * 3.0, max(ring, POV_Q / BODY_Q), 0.1
    own = WEAPONS[weapon][1]
    cal, _, _, _, opts = WEAPONS[weapon]
    quiet = suppressor > 0 or opts.get("suppressed", 0.0) > 0
    per_s = float(rate) if rate > 0 else (own if own > 0 else opts.get("cadence", 2.0))
    gap = 1.0 / max(per_s, 0.05)
    w = float(weight) + (_heard_weight(cal) if heard and not pov and not quiet else 0.0)
    tex = HEARD_TEXTURE if heard and not pov else 0.0
    one = [_shot(weapon, rng, r, az, float(miss), _c01(suppressor), bool(subsonic), float(action), _c01(crack), _c01(echo), gap, ring, direct, float(np.clip(w, 0.0, 200.0)), bool(pov), tex)
           for _ in range(min(shots, 8))]
    lock = WEAPONS[weapon][4].get("lock")
    pre = _lock(lock[1], lock[0], rng) * 2000.0 / r if lock else None
    start = lock[0] * 1e-3 if lock else (0.0 if pov else 1e-3)
    y = np.zeros(_n(start + shots * gap * 1.03) + max(map(len, one)))
    t = start
    for k in range(shots):
        g = 1.0 + 0.07 * rng.standard_normal()                # +-5-10 % shot to shot (Maher 2019)
        if quiet and k < 2:
            g *= 10.0 ** (float(first_pop) / (20.0 if k == 0 else 40.0))    # first-round pop: oxygen in the can (Murphy 2019)
        _put(y, one[k % len(one)], t, g)
        if opts.get("clip") and (k + 1) % opts["clip"] == 0:  # en-bloc clip thrown out after the last round (Wikipedia, M1)
            _put(y, _clip_ping(rng), t + 0.06, 0.03 * CALIBERS[cal][3] * 3.0 / r)
        t += gap * (1.0 + 0.01 * rng.standard_normal())
    if pre is not None:
        _put(y, pre, 0.0)
    if heard:
        y = _heard(y, place)
    return _finish(y, sr, r, str(level), 0.0005)


# Pa at 1 m: the motor whine at its rated speed, and each bolt locking
MOTOR_PA_M, BOLT_PA_M = 2.0, 20.0


@sfx("minigun", "Rotary machine gun: the motor spins the barrels up, the fire merges into a tearing buzz, then it winds down (fire=false: the spin alone).",
     dur=(3.0, "s of firing"), rate=(50.0, "rounds/s: 33-100 (M134, 2000-6000 rounds/min)"), spin=(0.4, "s to spin up"),
     fire=(True, "false = barrels turning, no rounds"), distance=(10.0, "m"), azimuth=(30.0, "degrees off the line of fire"),
     level=("relative", "relative|peak, as in firearm"), render=("heard", RENDER_HELP), place=("street", PLACE_HELP), seed=(0, "variation"))
def minigun(sr=DEFAULT_SR, dur=3.0, rate=50.0, spin=0.4, fire=True, distance=10.0, azimuth=30.0, level="relative",
            render="heard", place="street", seed=0):
    """Six barrels: one bolt per round, so the rotor runs at rate/6 (5.6-16.7 Hz). The electric motor whine
    (24 x rotor) and its gain are estimated; both it and the bolts fall with distance like the shots."""
    heard = _check_render(str(render))
    rng = np.random.default_rng(int(seed))
    r = float(np.clip(distance, 1.0, 5000.0))
    rate = float(np.clip(rate, 10.0, 120.0))
    spin, dur = float(np.clip(spin, 0.05, 3.0)), float(np.clip(dur, 0.1, 60.0))
    down = 0.8
    n = _n(spin + dur + down)
    t = np.arange(n) / P
    env = np.where(t < spin + dur, np.clip(t / spin, 0.0, 1.0), np.clip((spin + dur + down - t) / down, 0.0, 1.0) ** 2)
    per_s = rate * env
    ph = np.cumsum(per_s) / P
    idx = np.flatnonzero(np.diff(np.floor(ph)) > 0)
    fw = 4.0 * ph                                            # 24 x rotor = 4 x bolt rate
    whine = (np.sin(2 * np.pi * fw) + 0.5 * np.sin(4 * np.pi * fw) + 0.25 * np.sin(6 * np.pi * fw)) * env
    y = whine * MOTOR_PA_M / r
    nz = amb.hip(rng.uniform(-1, 1, _n(0.05)), 300.0)
    tick = _mclick(nz, CATCH, 0.0, 0.33) + _mclick(nz, ENDSTOP, 0.0, 0.15)
    for i in idx:
        _put(y, tick, i / P, BOLT_PA_M / r)
    if fire:
        w = _heard_weight(WEAPONS["minigun"][0]) if heard else 0.0
        pool = [_shot("minigun", rng, r, abs(float(azimuth)), 0.0, 0.0, False, 0.0, 1.0, 0.0, 1.0 / rate, weight=w, texture=HEARD_TEXTURE if heard else 0.0)
                for _ in range(6)]
        for k, i in enumerate(idx):
            if spin <= i / P < spin + dur:
                _put(y, pool[k % 6], i / P, 1.0 + 0.07 * rng.standard_normal())
    if heard:
        y = _heard(y, place)
    return _finish(y, sr, r, str(level), 0.001, 0.05)


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


def _cluster_times(ev, k: float) -> dict:
    """Event ms -> ms at speed 1/k. A faster hand brings the clusters closer, not the clicks inside one: only the start of
    each scales (clusters 1-36, 87-101, 348-383 ms: Farnell, c14 l.24703). A gap over 40 ms starts a cluster (read off those)."""
    out, base, prev = {}, 0.0, None
    for t in sorted({e[1] for e in ev}):
        if prev is None or t - prev > 40.0:
            base = t
        out[t] = base * k + (t - base)
        prev = t
    return out


def _mech(action: str, k: float, rng) -> np.ndarray:
    ev = MECH[action]
    at = _cluster_times(ev, k)
    end = max((at[e[1]] + (e[2] * k if e[0] == "slide" else 0) for e in ev), default=0.0) * 1e-3 + 0.2
    n = _n(end)
    x = np.zeros(n)
    nz = amb.hip(rng.uniform(-1, 1, n), 300.0)
    for e in ev:
        if e[0] == "slide":                                  # Farnell's slide: hip 1000 -> bp(4500 + 200 line, Q4) x line^2
            m = _n(e[2] * 1e-3 * k)
            ln = np.arange(m) / m
            _put(x, amb.bp(amb.hip(rng.uniform(-1, 1, m), 1000.0), (4500.0 + 200.0 * ln) * e[3], 4.0) * ln ** 2, at[e[1]] * 1e-3, 1.2)
        else:
            f = e[3] if len(e) > 3 else 1.0
            parts = tuple((a * f, q, d) for a, q, d in _C[e[0]])
            x += _mclick(nz, parts, at[e[1]] * 1e-3, 0.33 * e[2])
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
     mag=("plastic", "plastic|steel: the magazine of mag_drop"), rounds=(6, "links of link_eject, one per round"), seed=(0, "variation"))
def gun_mech(sr=DEFAULT_SR, action="slide_rack", weapon="rifle", speed=1.0, floor="concrete", mag="plastic", rounds=6, seed=0):
    """Same recipe as `reload` (R03 §5.3): every click and the slide through the shot's body, "so the shots and
    the reload sound are automatically aligned"."""
    action = str(action)
    if action not in MECH and action not in RELOADS:
        raise ValueError(f"unknown action {action!r}: {'|'.join((*MECH, *RELOADS))}")
    if str(floor) not in FLOORS:
        raise ValueError(f"unknown floor {floor!r}: {'|'.join(FLOORS)}")
    rng = np.random.default_rng(int(seed))
    k = 1.0 / float(np.clip(speed, 0.25, 4.0))
    seq = RELOADS.get(action, [(action, 0.0)])
    parts = [(_mech(a, k, rng), t * k) for a, t in seq]
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
CASES = {"9mm": (19.15, 9.93, 0.35), "45acp": (22.8, 12.1, 0.35), "556": (44.7, 9.58, 0.30), "762": (51.2, 11.9, 0.35),
         "50bmg": (99.31, 20.42, 0.60), "12ga": (16.0, 20.2, 0.50), "20mm": (102.0, 29.0, 1.0), "105mm": (370.0, 120.0, 1.5)}
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


def casing_physical(sr, caliber, floor, height, count, spread, roll, seed) -> np.ndarray:
    """Brass modes (spec table A), struck at each bounce with a new mix of modes (it lands lip, base or flank)
    on a surface that rings too, or on soft ground where it just stops."""
    caliber, floor = str(caliber), str(floor)
    if caliber not in CASES or floor not in FLOORS:
        raise ValueError(f"caliber must be {'|'.join(CASES)}; floor {'|'.join(FLOORS)}")
    rng = np.random.default_rng(int(seed))
    length, dia, wall = CASES[caliber]
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


def _bounce_times(v0: float, e: float, rng, min_gap: float = 1.5e-3) -> tuple:
    """Bounce instants (s) and landing speeds (m/s). Each flight lasts 2 v / g and v falls by the restitution, so the gaps
    shrink geometrically down to ``min_gap``; up to 7 % is lost to the tumble."""
    ts, vs, v, t = [0.0], [v0], v0, 0.0
    while True:
        v *= e * rng.uniform(0.93, 1.0)
        gap = 2.0 * v / 9.81
        if gap < min_gap:
            return np.array(ts), np.array(vs)
        t += gap
        ts.append(t)
        vs.append(v)


@sfx("casing_ricochet", "The film and video-game spent case: a long brass ring bouncing on the floor, turning as it flies, the bounces speeding up into a final rattle.",
     caliber=("556", "|".join(CASES)), floor=("concrete", "|".join(FLOORS)), height=(1.2, "m it falls from (0.1-5)"), count=(1, "cases (1-64)"),
     spread=(0.25, "s between the cases, on average"), ring=(2.0, "how long the brass rings: 1 = Farnell's Q 800, T60 0.35 s at 5 kHz; 2 is a film choice"),
     spin=(40.0, "tumbling in rev/s as it leaves the floor; 0 = no turning"), lively=(0.45, "0..1 of the gap between the floor's restitution and 1 given back (film choice; hard floors only)"),
     seed=(0, "variation"))
def casing_ricochet(sr=DEFAULT_SR, caliber="556", floor="concrete", height=1.2, count=1, spread=0.25, ring=2.0, spin=40.0, lively=0.45, seed=0):
    """`shell_modes` (spec table A) with a constant Q, so d grows with f and the top dies first (Q 800 at 5 kHz, Farnell, kept over the
    band: estimated); a split doublet on each mode, the air column closed at the base (c / 4L), and the mode's lobes passing the
    ear as the case turns, an amplitude modulation at n x spin that runs through flights and rings. Depth 0.5, split 0.2 %,
    amplitude in sqrt(v) so the final rattle stays audible (true law: v): estimated."""
    caliber, floor = str(caliber), str(floor)
    if caliber not in CASES or floor not in FLOORS:
        raise ValueError(f"caliber must be {'|'.join(CASES)}; floor {'|'.join(FLOORS)}")
    rng = np.random.default_rng(int(seed))
    length, dia, wall = CASES[caliber]
    mat, size, e, lop = FLOORS[floor]
    e = e if lop else min(0.9, e + (1.0 - e) * _c01(lively))
    k_ring = float(np.clip(ring, 0.2, 6.0))
    fs = shell_modes(dia / 2.0, wall)
    fs = fs[fs < 0.45 * P]
    # (f, d 1/s, gain, lobes); d at least 3 /s: a shell's low modes would otherwise ring for 20 s
    modes = [(f * (1.0 + s), 400.0 if lop else max(3.0, 20.0 * f / 5000.0 / k_ring), 1.0 / (i + 1.0), i + 2.0) for i, f in enumerate(fs) for s in (-0.001, 0.001)]
    modes.append((C_AIR / (4.0 * length * 1e-3), 300.0, 0.25, 0.0))
    if caliber == "12ga":                                    # the plastic hull dominates, as in casing_physical
        modes += [(f, 120.0, 1.0, 0.0) for f in shell_modes(10.1, 0.6, PLASTIC_C)]
    v0 = math.sqrt(2.0 * 9.81 * float(np.clip(height, 0.1, 5.0)))
    body = K._make_body("rect_plate", mat, size, 0.37, 0.41, modes=24, fmax=8000.0) if mat else None
    pulse = np.zeros(_n(0.25))
    pulse[:3] = (1.0, 0.66, 0.33)                            # Hertz contact 40-200 us (spec table C)
    thump = _nm(K.modal_bank(pulse, body, P, 0.5)[0]) if body is not None else None
    count = int(np.clip(count, 1, 64))
    tail = 7.0 / min(m[1] for m in modes)
    runs = []
    for c in range(count):
        ts, vs = _bounce_times(v0, e, rng)
        runs.append((0.0 if c == 0 else float(rng.uniform(0.0, float(spread) * count)), ts, vs))
    y = np.zeros(_n(max(t0 + ts[-1] for t0, ts, _ in runs) + tail + 0.3))
    unit = [float(np.max(np.abs(modal.render_modes(pulse[:_n(0.01)], modal.Modes(np.array([m[0]]), np.array([m[1]]), np.ones(1), "vdd"))))) + 1e-12 for m in modes]
    for t0, ts, vs in runs:
        idx = np.round((t0 + ts) * P).astype(int)
        g = np.sqrt(vs / v0) * np.minimum(1.0, (2.0 * vs / 9.81 - 1.5e-3) / 6e-3 + 0.15)    # the last few fade out rather than stop
        meff = 1.0 / (1.0 + 12.0 * rng.uniform(-0.5, 0.5, len(ts)) ** 2)   # lands anywhere along its length, c17 l.23436 (casing_physical)
        turn = np.zeros(len(y))
        turn[idx] = float(spin) * np.sqrt(vs / v0) * rng.uniform(0.7, 1.3, len(ts))
        turn = turn[np.maximum.accumulate(np.where(turn > 0, np.arange(len(y)), 0))]   # hold each flight's rate
        turn[idx[-1] + _n(0.02):] = 0.0                      # at rest
        phase = np.cumsum(turn) / P
        for (f, d, gain, lobes), one in zip(modes, unit):
            force = np.zeros(len(y))
            hit = g * meff * gain * rng.uniform(0.2, 1.0, len(idx))
            for j, w in enumerate(pulse[:3]):
                force[idx + j] += w * hit
            u = modal.render_modes(force, modal.Modes(np.array([f]), np.array([d]), np.ones(1), "vdd")) / one
            if lobes > 0 and spin > 0:
                u = u * (1.0 - 0.25 * (1.0 - np.cos(2.0 * np.pi * lobes * phase + rng.uniform(0.0, 6.28))))
            y += u
        for i, gg in zip(idx, g):
            if thump is not None:
                _put(y, thump, i / P, 0.15 * gg)
            else:
                m = _n(0.012)
                _put(y, amb.lop(rng.uniform(-1, 1, m) * (1.0 - np.arange(m) / m) ** 2, float(lop)), i / P, 0.3 * gg)
    return _fin(amb.lop(y, float(lop)) if lop else y, sr, 0.6, fade=0.0005, tail=0.1)


# ================================================================== bullets
_BULLET = {"metal": ("steel", 0.5, 0.0, 120), "metal_thin": ("steel", 0.25, 2.0, 80), "concrete": ("concrete", 0.4, 0.0, 40),
           "wood": ("wood", 0.3, 30.0, 60), "glass": ("glass", 0.4, 0.0, 80)}


def _hit(material, sz, damping, modes, v, cal_mm, rng) -> np.ndarray:
    """Force into the struck body, level from energy in the caller. Above 100 m/s the length follows the penetration, not Hertz
    (Johnson, Table 11.3, c30 L13117): t_p = 2 calibres / v, coefficient estimated (40-110 us; at least one sample)."""
    body = K._make_body("rect_plate", material, sz, 0.37, 0.41, modes=modes, damping=damping, fmax=16000.0)
    f = np.zeros(_n(max(0.2, min(1.5, body.t60()))))
    m = max(1, _n(2.0 * cal_mm * 1e-3 / max(v, 100.0)))
    f[:m] = np.linspace(1.0, 0.0, m, endpoint=False)
    return _nm(K.modal_bank(f, body, P, 0.7)[0])


def _debris(kind, rng, gain) -> np.ndarray:
    """Falling fragments as small hail on the surface: ``hail_impulses`` with a thinning rate (estimated)."""
    size, rate, dur, body, cutoff = {"chips": (3e-3, 40.0, 0.5, "stone", 8000.0), "dirt": (1.5e-3, 70.0, 0.6, "concrete", 1500.0),
                                     "splinters": (2e-3, 25.0, 0.3, "wood", 6000.0)}[kind]
    f, _, _ = K.hail_impulses(dur, P, rate, size, 1.0, 1.0, int(rng.integers(1 << 30)))
    b = K._make_body("rect_plate", body, 0.3, 0.3, 0.4, modes=40, fmax=12000.0)
    y = K.modal_bank(f, b, P, 0.6)[0]
    return _nm(amb.lop(y, cutoff)) * gain


@sfx("bullet_impact", "A bullet striking metal, thin metal, concrete, wood, flesh, armour, water, dirt or glass, with its fragments. Bands and lengths are estimates: no published measurements.",
     material=("metal", "metal|metal_thin|concrete|wood|flesh|flesh_armor|water|dirt|glass"), caliber=("556", "|".join(CALIBERS)),
     distance=(5.0, "m"), debris=(1.0, "0..1 fragments, chips, drops"),
     level=("relative", "relative = amplitude in sqrt(muzzle energy), the .50 at 0.85 | peak = every file at 0.85"), seed=(0, "variation"))
def bullet_impact(sr=DEFAULT_SR, material="metal", caliber="556", distance=5.0, debris=1.0, level="relative", seed=0):
    """Same recipe as before, then what the bullet's size changes: the struck area grows as
    sqrt(calibre), which lowers the modes; the force lasts length / v instead of being one sample; soft targets close down."""
    from . import foley as Fo
    from . import matter as M
    material = str(material)
    rng = np.random.default_rng(int(seed))
    cal = str(caliber) if str(caliber) in CALIBERS else "556"
    d, l, v, _, _ = CALIBERS[cal]
    area = math.sqrt(d / CALIBERS["556"][0])
    db = _c01(debris)
    y = np.zeros(_n(2.5))
    if material in _BULLET:
        mat, sz, damp, modes = _BULLET[material]
        _put(y, _hit(mat, sz * area, damp, modes, v, d, rng), 0.0)
        if material == "metal_thin":
            _put(y, _nm(K.crumple(P, kind="can", energy=0.7, dur=0.05, seed=int(seed))), 0.0, 0.5)
        elif material == "metal":
            s = _nm(M.spark(P, size=0.2, seed=int(seed)))
            _put(y, s, 0.002, 0.15 * db)
        elif material == "concrete":
            m = _n(0.015)
            _put(y, amb.bp(rng.uniform(-1, 1, m), 1500.0, 0.6) * (1.0 - np.arange(m) / m), 0.0, 1.5)
            _put(y, _debris("chips", rng, 0.3 * db), 0.03)
        elif material == "wood":
            _put(y, _nm(K.crumple(P, kind="can", energy=0.3, dur=0.04, bright=0.8, seed=int(seed))), 0.0, 0.25)
            _put(y, _debris("splinters", rng, 0.2 * db), 0.02)
        elif material == "glass":
            _put(y, _nm(K.smash(P, material="glass", size=0.4, energy=0.6, seed=int(seed))), 0.0, 0.8 if db else 0.4)
    elif material in ("flesh", "flesh_armor"):
        _put(y, _nm(Fo.thud(P, body="sack", contact_ms=8.0, size=0.6, force=0.9, seed=int(seed))), 0.0)
        m = _n(0.01)
        _put(y, amb.lop(rng.uniform(-1, 1, m) * (1.0 - np.arange(m) / m), 1200.0 / area), 0.0, 1.2)
        if material == "flesh_armor":
            _put(y, _hit("steel", 0.2 * area, 30.0, 60, v, d, rng), 0.0, 0.5)
    elif material == "water":
        _put(y, _nm(M.splash(P, size=0.1 * area, dur=1.0, droplets=int(6 * db), seed=int(seed))), 0.0)
    elif material == "dirt":
        m = _n(0.025)
        _put(y, amb.lop(rng.uniform(-1, 1, m) * (1.0 - np.arange(m) / m) ** 2, 1500.0 / area), 0.0, 1.5)
        _put(y, _debris("dirt", rng, 0.25 * db), 0.05)
    else:
        raise ValueError(f"unknown material {material!r}")
    a = 0.0327 * (l * 1e-3 / v * P) ** 2 / 2.0               # 3 taps with the variance of a Hann pulse length / v: under 3 samples here
    y = np.convolve(y, (a, 1.0 - 2.0 * a, a))[1:-1]
    y = space.distance(y, float(np.clip(distance, 1.0, 2000.0)), ref=1.0)
    if str(level) not in ("relative", "peak"):
        raise ValueError(f"unknown level {level!r}: relative|peak")
    peak = 0.85 * (math.sqrt(CHARGE[cal][0] / IMPACT_REF_J) if level == "relative" else 1.0)
    return _fin(y, sr, peak, fade=0.0003, tail=0.05)


@sfx("ricochet", "Bullet glancing off a surface: a short hit, then a whining tone falling as it flies away.",
     surface=("metal", "metal|stone|water"), variant=("whine", "whine | tumble (fluttering) | zip (short and high) | none (only the hit)"),
     f0=(1343.0, "Hz of the whine (one forum reading of 3200 -> 850 Hz; UNSOURCED)"), speed=(199.0, "m/s after the bounce (same reading)"),
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
        src = sum(g * amb.bp(rng.uniform(-1, 1, n), f * r, 8.0) for r, g in ((1.0, 1.0), (1.5, 0.4), (2.3, 0.2)))    # UNSOURCED partials
        if variant == "tumble":
            src = src * (0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(80.0, 300.0) * np.arange(n) / P))
        src = np.asarray(doppler(src, P, speed=float(speed), closest=float(closest), pass_at=0.05, pan=False))
        src = src.mean(axis=1) if src.ndim == 2 else src
        _put(y, _nm(src), 0.015, 0.5)
    return _fin(y, sr, 0.85, fade=0.0003, tail=0.05, soft=False)


@sfx("bullet_pass", "A bullet passing near you: the supersonic crack alone (N-wave, ground reflection, wake); a subsonic round only whizzes; near Mach 1 you get both.",
     caliber=("556", "|".join(CALIBERS)), miss=(3.0, "m from your head"), speed=(0.0, "m/s (0 = the calibre's own)"),
     range=(0.0, "m flown already: it slows down, so far away the crack gives way to a whizz"), spin_tone=(True, "true = the faint line at speed / rifling twist"),
     render=("heard", RENDER_HELP), place=("street", PLACE_HELP), seed=(0, "variation"))
def bullet_pass(sr=DEFAULT_SR, caliber="556", miss=3.0, speed=0.0, range=0.0, spin_tone=True, render="heard", place="street", seed=0):
    from .reverbs import doppler
    heard = _check_render(str(render))
    rng = np.random.default_rng(int(seed))
    d, l, v, _, _ = CALIBERS.get(str(caliber), CALIBERS["556"])
    cal = str(caliber) if str(caliber) in CALIBERS else "556"
    v = (float(speed) if speed > 0 else v) * math.exp(-max(float(range), 0.0) / drag_length(cal))
    b = float(np.clip(miss, 0.1, 500.0))
    y = np.zeros(_n(0.6))
    if v > C_AIR:
        _put(y, _crack(v, d, l, b, rng), 0.0)
        # the spin line v / twist: 915 m/s / 0.18 m = 5083 Hz (M249) -30 dB, 15 ms
        m = _n(0.015) if spin_tone else 0
        _put(y, np.sin(2 * np.pi * min(v / 0.18, 0.45 * P) * np.arange(m) / P) * (1.0 - np.arange(m) / m), 0.0, 0.03 * n_wave_pa(v, d, l, b)[0])
    if v < 1.1 * C_AIR:
        n = _n(0.4)
        src = amb.bp(rng.uniform(-1, 1, n), min(0.2 * v / (d * 1e-3), 0.45 * P), 4.0)
        src = np.asarray(doppler(src, P, speed=min(v, 0.9 * C_AIR), closest=b, pass_at=0.3, pan=False))
        src = src.mean(axis=1) if src.ndim == 2 else src
        _put(y, _nm(src), 0.0, (0.5 if v > C_AIR else 1.0) * max(1.0, float(np.max(np.abs(y)))))
    if heard and v > C_AIR:
        y = _heard(y, place)
    return _fin(y, sr, 0.9, fade=0.0002, tail=0.02)


# Specs, no code (armes_feu.md:204, 224). Spread 0.3-1 s and reloads estimated. Indoors: echo 0, the room is the fx (Farnell: not in the sound).
RECIPES = {
    "musket_volley": {"layers": [{"type": "scatter", "dur": 1.0, "count": 12, "seed": 4, "reseed": True, "gain_db": 3.0, "pitch": 0.3,
                                  "layers": [{"type": "sfx", "kind": "firearm", "params": {"weapon": "flintlock", "distance": 5.0, "echo": 0.0}}]}],
                      "fx": [{"type": "distance", "metres": 150.0}], "normalize": -1.0},
    "gunshot_indoor": {"layers": [{"type": "sfx", "kind": "firearm", "params": {"weapon": "pistol_9mm", "distance": 5.0, "echo": 0.0, "place": "none"}}],
                       "fx": [{"type": "room", "preset": "bathroom", "mix": 0.6}], "normalize": -1.0},
}
