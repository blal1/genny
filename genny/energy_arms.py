"""Electric weapons and science fiction: stun guns, arcs, blaster, phaser, beam, laser breakdown, plasma, railgun, coilgun.

Railgun, coilgun and plasma: their kinematics are calculated, their timbre is designed. Sources: ``research/SOURCES.md``. Nothing has been listened to.
"""
from __future__ import annotations

import math

import numba
import numpy as np
from scipy.signal import butter, sosfilt

from . import foley as FO
from . import matter as Ma
from .arms import _crack, _n_wave_shape, shock_arrival
from .core import DEFAULT_SR
from .foley import C_AIR, P, _fin, _n, _nm, _put
from .friction import coil_spring
from .ordnance import _fireball, blast
from .physical import ambience as amb
from .sfx import sfx, whoosh


def _kind(kind, kinds, name):
    kind = str(kind)
    if kind not in kinds:
        raise ValueError(f"unknown {name} {kind!r}: {'|'.join(kinds)}")
    return kind


# =============================================================================================
# ARCS
# =============================================================================================
STUN_PPS = 19.0                                             # pulses per second (20-29 on other models): pass rate=
STUN_TONES = {"contact": 0.6, "arc_gap": 1.0}               # open air 50 kV, under load 1200-2000 V: duller at contact
TASER_V, TASER_RANGE = 55.0, 11.0                           # X26's N2 cartridge, m/s; m


PULSE = 100e-6                                              # s: each discharge


def _shot(y, sr, level, fade=0.0002, tail=0.005):
    """`_fin` with the 12 Hz high-pass's tail let die first (.15 s): a tail fade landing on its residue is a kink in silence that detect_clicks catches."""
    return _fin(np.append(y, np.zeros(_n(0.15))), sr, level, fade=fade, tail=tail)


def arc_train(dur, rate, chaos, tone, rng):
    """(signal, discharge times). `tesla`'s train with the pulse at its measured 100 us, through the spark formant, corona above 3 kHz following it.
    None in the last 50 ms: `tesla` lets one fall into its fade, a burst in near silence. Jitter law and corona level as `tesla`."""
    rate, ch = float(np.clip(rate, 1.0, 600.0)), float(np.clip(chaos, 0.0, 1.0))
    n, m = _n(dur), _n(PULSE)
    times, t = [], 0.0
    while t < dur - 0.05:
        times.append(t)
        t += (1.0 + ch * rng.uniform(-0.5, 0.5)) / rate
    x = np.zeros(n)
    for t in times:
        _put(x, rng.uniform(-1, 1, m) * np.hanning(m + 2)[1:-1], t, rng.lognormal(0.0, 0.1 + 0.5 * ch))
    y = Ma._unit(Ma._formant(x, float(np.clip(tone, 0.5, 1.5))))
    y = y + 0.25 * Ma._unit(amb.hip(rng.uniform(-1, 1, n), 3000.0) * amb.lop(np.abs(x), 80.0))
    return y * np.minimum((n - np.arange(n)) / _n(0.04), 1.0), np.array(times)      # the corona's envelope still rings when the buffer ends


@sfx("stun_gun", "Stun gun: a train of arcs at 19 pulses per second. contact is on a body, duller than the open-air arc_gap.",
     kind=("contact", "contact|arc_gap"), rate=(STUN_PPS, "pulses per second: 19, 20 in some sources, 29 on the X3W"), dur=(1.0, "s"), seed=(0, "variation"))
def stun_gun(sr=DEFAULT_SR, kind="contact", rate=STUN_PPS, dur=1.0, seed=0):
    k = _kind(kind, STUN_TONES, "stun_gun")
    y, _ = arc_train(float(np.clip(dur, 0.2, 30.0)), rate, 0.1, STUN_TONES[k], np.random.default_rng(int(seed)))
    return _shot(y, sr, 0.7)


def taser_flight(distance):
    """Seconds from the cartridge to the probes landing: distance / 55 m/s, 11 m at most. 0.2 s at 11 m (calculated)."""
    return float(np.clip(distance, 0.5, TASER_RANGE)) / TASER_V


@sfx("taser", "Taser fired: the nitrogen cartridge, the wire paying out while the probes fly at 55 m/s, then the 19 pulses per second on the target.",
     kind=("fire", "fire"), distance=(TASER_RANGE, "m to the target, 11 at most: 0.2 s of flight"), rate=(STUN_PPS, "pulses per second"), dur=(1.0, "s on the target"), seed=(0, "variation"))
def taser(sr=DEFAULT_SR, kind="fire", distance=TASER_RANGE, rate=STUN_PPS, dur=1.0, seed=0):
    _kind(kind, ("fire",), "taser")
    rng = np.random.default_rng(int(seed))
    t = taser_flight(distance)
    drive = _nm(arc_train(float(np.clip(dur, 0.2, 30.0)), rate, 0.1, STUN_TONES["contact"], rng)[0])
    y = np.zeros(_n(t) + len(drive))
    # cartridge: `air_leak` burst, tau 15 ms in 80 ms; the wire, a thin band that dies when the probes land.
    _put(y, _nm(Ma.air_leak(sr=P, dur=0.08, pressure=1.0, orifice=3.0, burst=0.015, seed=seed)), 0.0, 1.0)
    m = _n(t)
    _put(y, _nm(amb.bp(rng.standard_normal(m), 5200.0, 3.0)) * np.minimum(np.arange(m) / _n(0.005), 1.0) * np.minimum((m - np.arange(m)) / _n(0.005), 1.0), 0.0, 0.05)
    _put(y, drive, t, 1.0)
    return _shot(y, sr, 0.7)


@sfx("cattle_prod", "Cattle prod: an arc more chopped than a stun gun's, bursts of fast irregular discharges over a mains-less buzz.",
     dur=(1.2, "s"), rate=(60.0, "discharges per second"), chop=(7.0, "bursts per second"), seed=(0, "variation"))
def cattle_prod(sr=DEFAULT_SR, dur=1.2, rate=60.0, chop=7.0, seed=0):
    """Chaos .8, a chopping gate (3:1 on/off, 4 ms edges), the arc under it at .4."""
    rng = np.random.default_rng(int(seed))
    dur = float(np.clip(dur, 0.2, 30.0))
    y = _nm(arc_train(dur, rate, 0.8, 0.85, rng)[0])
    n = len(y)
    z = Ma.arc(sr=P, dur=dur, mains=float(rate) / 2.0, stability=1.0, seed=seed)
    y = y + 0.4 * _nm(np.resize(z, n))
    phase = (np.cumsum(float(np.clip(chop, 1.0, 40.0)) * np.clip(1.0 + 0.3 * Ma._lfn(rng, n, 4.0), 0.3, 1.7)) / P) % 1.0
    gate = amb.lop(amb.lop((phase < 0.75).astype(float), 40.0), 40.0)
    return _shot(y * (0.03 + 0.97 * gate), sr, 0.7)


@sfx("arc_weapon", "Arc projector: a continuous arc buzzing at twice the mains, 100 Hz at 50 Hz; a game weapon has no mains and can take 400 Hz.",
     dur=(2.0, "s"), buzz=(100.0, "Hz of the buzz: 100 on 50 Hz mains, 400 for a game weapon"), tone=(1.0, "formant scale 0.5..1.5"), seed=(0, "variation"))
def arc_weapon(sr=DEFAULT_SR, dur=2.0, buzz=100.0, tone=1.0, seed=0):
    """`arc` holding. Its loudest re-strikes, 6 kHz bursts far above the local level, are rounded so one burst does not stand out (measured: a lone flag at 400 Hz, a 6 kHz oscillation, not a step): tanh at twice the peak."""
    y = Ma.arc(sr=P, dur=float(np.clip(dur, 0.2, 60.0)), mains=float(np.clip(buzz, 40.0, 800.0)) / 2.0, stability=1.0, tone=tone, seed=seed)
    return _shot(np.tanh(2.0 * _nm(y)), sr, 0.65, fade=0.005, tail=0.03)


# =============================================================================================
# ENERGY WEAPONS
# =============================================================================================
PHASER_FM = (((1360.0, 80.0, 32.0), (1390.0, 200.0, 28.0)), ((2446.0, 443.0, 47.0), (2586.0, 346.0, 36.0)))   # carrier, deviation, rate, Hz
PHASER_GROUPS = (0.3, 0.7)
PHASER_POWER = {"stun": 0.2, "kill": 0.4}
ENERGY_KINDS = ("blaster", "phaser", "beam", "real")


def phaser_oscillators(n):
    """The four FM oscillators in their two groups, as the course writes them."""
    t = np.arange(n) / P
    y = np.zeros(n)
    for group, g in zip(PHASER_FM, PHASER_GROUPS):
        for fc, dev, fm in group:
            y += g * np.sin(2.0 * np.pi * np.cumsum(fc + dev * np.sin(2.0 * np.pi * fm * t)) / P)
    return y


def _phaser(n, power, noise, rng):
    """Sum + noise hip 2000 -> bp 2600 Q7 and bp 1300 Q7 in parallel -> clip +-1 -> hip 1200 -> x.2 stun, x.4 kill -> hip 5 -> clip +-1."""
    x = phaser_oscillators(n) + noise * amb.hip(rng.uniform(-1, 1, n), 2000.0)
    y = np.clip(amb.bp(x, 2600.0, 7.0) + amb.bp(x, 1300.0, 7.0), -1.0, 1.0)
    return np.clip(amb.hip(amb.hip(y, 1200.0) * PHASER_POWER[power], 5.0), -1.0, 1.0)


def beam_hum(n, freq, sweep, shimmer, seed):
    """`force_field`, read at a varying speed for ``sweep`` octaves, plus the beat of `energy_blade`'s upper buzz."""
    octv = float(np.clip(sweep, -2.0, 2.0))
    y = np.asarray(FO.force_field(sr=P, dur=(n / P) * max(1.0, 2.0 ** octv), freq=freq, shimmer=shimmer, seed=seed), float)
    blade = np.asarray(FO.energy_blade(sr=P, dur=(n / P) * max(1.0, 2.0 ** octv), freq=freq, swings=0, buzz=0.4, seed=seed), float)
    m = min(len(y), len(blade))
    y = _nm(y[:m]) + 0.4 * _nm(blade[:m])
    # Both are saws: edges flagged every period of the hum, still with three one-poles at 1.5 kHz (measured). A steep 1.5 kHz low-pass rounds them; it takes force_field's 2.4 kHz band away.
    y = sosfilt(butter(8, 1500.0, "lowpass", fs=P, output="sos"), y)
    pos = np.cumsum(2.0 ** (octv * np.arange(n) / n))
    return np.interp(pos - pos[0], np.arange(m), y)


BREAK_AT = 3                                                # samples: the fade must not eat the N-wave's front


def breakdown(rng, length=0.004):
    """Air broken down at the focus. N-wave of half a period at 20 kHz, where the energy lies, then the plasma's hiss for ``length`` s. The beam alone is silent."""
    w = _n_wave_shape(25e-6)
    y = np.zeros(_n(length))
    y[BREAK_AT:BREAK_AT + len(w)] = w
    m = len(y)
    hiss = amb.hip(amb.hip(rng.uniform(-1, 1, m), 8000.0), 8000.0) * (1.0 - np.arange(m) / m) ** 2
    return y + 0.35 * _nm(hiss)


@sfx("energy_weapon", "Sci-fi energy weapons: blaster (a tensioned cable struck: a dispersive chirp falling from the top), phaser (the crossed FM of the TOS phaser), beam (sustained beating hum), real (what a real laser does: silent beam, a 2-5 ms crack of air breakdown).",
     kind=("blaster", "|".join(ENERGY_KINDS)), power=("kill", "phaser: stun (x.2) | kill (x.4)"), noise=(0.15, "phaser: hip 2000 noise in the sum"), freq=(100.0, "beam: Hz of the hum; blaster: the note"),
     sweep=(0.0, "beam: octaves over the duration"), dispersion=(1.0, "blaster: 0..1"), dur=(-1.0, "s, -1 = the kind's: blaster .5, phaser 1, beam 3"), seed=(0, "variation"))
def energy_weapon(sr=DEFAULT_SR, kind="blaster", power="kill", noise=0.15, freq=100.0, sweep=0.0, dispersion=1.0, dur=-1.0, seed=0):
    k = _kind(kind, ENERGY_KINDS, "energy_weapon")
    rng = np.random.default_rng(int(seed))
    if k == "real":
        return _fin(breakdown(rng), sr, 0.8, fade=0.00004, tail=0.0005)
    d = float(np.clip({"blaster": 0.5, "phaser": 1.0, "beam": 3.0}[k] if dur < 0 else dur, 0.05, 60.0))
    if k == "blaster":
        # Hard stick, strong dispersion, ring .5 s.
        y = coil_spring(float(freq) * 1.1, d, sr=P, excite="strike", dispersion=float(dispersion), tone=0.1, decay=0.5, hardness=1.0)
        return _shot(y, sr, 0.8, fade=0.001, tail=0.05)
    if k == "phaser":
        p = _kind(power, PHASER_POWER, "phaser power")
        # The level carries stun against kill: .2 against .4 survive, no normalising.
        return _fin(_phaser(_n(d), p, float(noise), rng), sr, None, fade=0.01, tail=0.04)
    return _fin(beam_hum(_n(d), float(freq), sweep, 0.5, seed), sr, 0.6, fade=0.04, tail=0.1)


# =============================================================================================
# PLASMA
# =============================================================================================
PLASMA_CHARGE, PLASMA_FLIGHT, PLASMA_AM = 0.5, 0.45, 40.0   # s, s, Hz


def _plasma_impact(rng, seed, energy=1.0):
    """Arc and vapour: the snap, .35 s of arc, steam and sizzle."""
    y = np.zeros(_n(0.9))
    _put(y, _nm(Ma.spark(sr=P, size=0.7, tone=0.8, seed=seed)), 0.0, 1.0)
    a = _nm(Ma.arc(sr=P, dur=0.35, mains=50.0, stability=1.0, seed=seed))
    _put(y, a * np.minimum(np.arange(len(a)) / _n(0.002), 1.0), 0.002, 0.5)
    _put(y, _nm(Ma.steam(sr=P, dur=0.8, pressure=0.7, orifice=8.0, wet=0.3, seed=seed)), 0.03, 0.22 * energy)
    _put(y, _nm(Ma.sizzle(sr=P, dur=0.7, intensity=0.7, decay=0.2, seed=seed)), 0.05, 0.22 * energy)
    return y


def _plasma_bolt(rng, seed):
    """Charge .5 s, low whoosh with noise amplitude-modulated at 40 Hz, impact arc with a faint 10 g blast at 15 m."""
    fl = _n(PLASMA_FLIGHT)
    charge = _nm(FO.charge_up(sr=P, dur=PLASMA_CHARGE, start=120.0, end=900.0, ready=0))
    y = np.zeros(_n(PLASMA_CHARGE + PLASMA_FLIGHT) + _n(1.4))
    _put(y, charge, 0.0, 0.6)
    t = np.arange(fl) / P
    env = np.sin(np.pi * np.arange(fl) / fl) ** 0.5
    low = np.resize(np.asarray(whoosh(sr=P, dur=PLASMA_FLIGHT, low=90, high=420, direction="up"), float), fl)
    am = _nm(amb.bp(rng.uniform(-1, 1, fl), 1200.0, 0.7)) * (0.5 + 0.5 * np.sin(2.0 * np.pi * PLASMA_AM * t)) * env
    _put(y, 0.7 * _nm(low) + am, PLASMA_CHARGE, 1.0)
    at = PLASMA_CHARGE + PLASMA_FLIGHT
    _put(y, _plasma_impact(rng, seed), at, 1.0)
    bl = blast(0.01, 15.0, rng, fireball=0.5)
    _put(y, _nm(bl), at, 0.25)
    return y


@sfx("plasma", "Plasma gun: bolt is the charge, the bolt's flight (low whoosh, noise throbbing at 40 Hz) and its hit; impact is the hit alone, an arc and its vapour.",
     kind=("bolt", "bolt|impact"), seed=(0, "variation"))
def plasma(sr=DEFAULT_SR, kind="bolt", seed=0):
    k = _kind(kind, ("bolt", "impact"), "plasma")
    rng = np.random.default_rng(int(seed))
    y = _plasma_bolt(rng, seed) if k == "bolt" else _plasma_impact(rng, seed)
    return _fin(y, sr, 0.8, fade=0.002, tail=0.1)


# =============================================================================================
# RAILGUN
# =============================================================================================
RAIL_SHOTS = {"2008": (2520.0, 3.35), "2010": (3390.0, 3.2)}       # m/s. 3.2 kg is the 18.4 MJ shot; 3.35 kg calculated from 10.64 MJ at 2520
RAIL_D_MM, RAIL_L_MM = 60.0, 300.0                          # projectile diameter and length, mm
RAIL_W = 0.5                                                # kg TNT standing for the muzzle's gas
RAILS = ("charge", "fire", "flyby")
RAIL_FAR = 1000.0 * RAIL_D_MM * 1e-3                        # m
LEAD = 8                                                    # samples of silence: the fade must not eat a front at time 0


def rail_energy(v, mass):
    """Muzzle energy in J: 18.4 MJ for 3.2 kg at 3390 m/s."""
    return 0.5 * mass * v * v


def mach_cone(v, c=C_AIR):
    """(Mach, half-angle in degrees): M 7.35 and 7.8 degrees at 2520 m/s, 9.88 and 5.8 at 3390."""
    m = v / c
    return m, math.degrees(math.asin(1.0 / m))


def _muzzle_flash(rng, seed):
    """Muzzle arc and shear fireball: spark size 1, a 12 ms arc, a 0.8 s fireball (`ordnance._fireball`;
    `flame kind=fireball` has edges of its own every 11.3 ms, measured)."""
    y = np.zeros(_n(1.2))
    _put(y, _nm(Ma.spark(sr=P, size=1.0, tone=0.8, seed=seed)), 0.0, 0.7)
    a = _nm(Ma.arc(sr=P, dur=0.05, mains=400.0, stability=1.0, seed=seed))[:_n(0.012)]
    _put(y, a * np.hanning(len(a)), 0.0, 0.5)
    fb = _nm(_fireball(0.8, rng))
    _put(y, fb * np.minimum((len(fb) - np.arange(len(fb))) / _n(0.05), 1.0), 0.003, 0.5)   # the fireball's rumble stops above zero when its buffer ends: a step, measured at the end of the fireball
    return y


def _rail_parts(v, distance, azimuth, rng, seed):
    """(signal, seconds): the shell_pass of `ordnance` with the arc in place of the charge. x along the line of fire, y off it. Far along the line the crack leads the muzzle by distance / c - distance / v: 252-262 ms at 100 m."""
    th = math.radians(azimuth)
    x, y = distance * math.cos(th), distance * math.sin(th)
    b = max(y, 0.5)
    t_c = shock_arrival(v, x, b) if azimuth <= 75.0 else None     # none past 75 degrees: Murphy 2019, as `ordnance.shell_pass`
    t_b = distance / C_AIR
    t0 = min(t_c, t_b) if t_c is not None else t_b
    mz = _muzzle_flash(rng, seed)
    bl = blast(RAIL_W, distance, rng, fireball=0.0)
    gun = np.zeros(max(len(mz), len(bl)))
    _put(gun, _nm(mz), 0.0, 1.0)
    _put(gun, _nm(bl), 0.0, 0.6)
    parts = [(gun, t_b - t0)]
    if t_c is not None:
        # Arrival on the true geometry; shape at no less than the far field, miss > 1000 d (`arms.n_wave_pa`): nearer has no valid figure.
        parts.append((_nm(_crack(v, RAIL_D_MM, RAIL_L_MM, max(b, RAIL_FAR), rng)), t_c - t0))
    return parts


def _lay(parts, sr, level=0.85, tail=0.05):
    y = np.zeros(LEAD + max(_n(t) + len(s) for s, t, *_ in parts))
    for s, t, *_ in parts:
        _put(y, s, LEAD / P + t)
    return _shot(y, sr, level, fade=0.00004, tail=tail)


@sfx("railgun", "Railgun: charge is the capacitors winding up, 500 Hz to 4 kHz in 1.5 s; fire is the crack of a Mach 7-10 slug ahead of the muzzle's arc and fireball; flyby is the N-wave alone.",
     kind=("fire", "|".join(RAILS)), speed=(3390.0, "m/s: 2520 (10.64 MJ, 2008), 3390 (18.4 MJ, 2010)"), distance=(100.0, "m from the muzzle (fire)"), azimuth=(0.0, "degrees off the line of fire (fire)"),
     miss=(RAIL_FAR, "m from the trajectory (flyby), 60 at least: far field"), dur=(1.5, "s (charge)"), seed=(0, "variation"))
def railgun(sr=DEFAULT_SR, kind="fire", speed=3390.0, distance=100.0, azimuth=0.0, miss=RAIL_FAR, dur=1.5, seed=0):
    k = _kind(kind, RAILS, "railgun")
    rng = np.random.default_rng(int(seed))
    v = float(np.clip(speed, 400.0, 8000.0))
    if k == "charge":                                       # 500 Hz to 4 kHz, 1.5 s
        return _fin(FO.charge_up(sr=P, dur=float(np.clip(dur, 0.2, 10.0)), start=500.0, end=4000.0, ready=1), sr, 0.7, fade=0.003, tail=0.01)
    if k == "flyby":
        return _lay([(_crack(v, RAIL_D_MM, RAIL_L_MM, max(float(miss), RAIL_FAR), rng), 0.0)], sr)
    return _lay(_rail_parts(v, max(float(distance), 1.0), float(azimuth), rng, seed), sr)


# =============================================================================================
# COILGUN
# =============================================================================================
COIL_STAGES, COIL_LEN = 45, 2.1                             # 46.7 mm apart
COIL_PITCH = COIL_LEN / COIL_STAGES
COIL_D_MM = 12.0                                            # slug across
COIL_FAR = 1000.0 * COIL_D_MM * 1e-3                        # m: 1000 d, the far field of `arms.n_wave_pa`
COIL_V = {"emg01a": 45.0, "gr1": 75.0}                      # m/s
COIL_SWITCH = {"spark": (1.0, 0.00025), "igbt": (0.0, 0.00012), "scr": (0.35, 0.00018)}   # noise share and decay of each switch


def stage_times(v_max, stages=COIL_STAGES, pitch=COIL_PITCH):
    """Firing time of each stage. Same force per stage: v(k) = v_max sqrt(k / N), so uniform acceleration, t_k = (2 L / v_max) sqrt(k / N)."""
    k = np.arange(1, int(stages) + 1)
    return 2.0 * int(stages) * float(pitch) / float(v_max) * np.sqrt(k / int(stages))


def _stage_click(switch, rng):
    """A coil firing: its switch's noise tick and the coil's ring."""
    noise, tau = COIL_SWITCH[switch]
    m = max(_n(6.0 * tau), 8)
    t = np.arange(m) / P
    e = np.exp(-t / tau) * (1.0 - np.exp(-t / 6e-5))
    return e * (noise * rng.uniform(-1, 1, m) + (1.0 - noise) * np.sin(2.0 * np.pi * 3200.0 * t))


def _coil_shot(v, stages, pitch, switch, gain, miss, mass, rng, crack_gain=1.0):
    """Train of stage clicks + the slug's N-wave when supersonic, reaching a listener as far down the line as off it. Slug: steel 7850 kg/m3 (handbook), 12 mm across, length from its mass."""
    times = stage_times(v, stages, pitch)
    y = np.zeros(_n(times[-1] + 0.06))
    for t in times:
        _put(y, _stage_click(switch, rng), float(t), gain * rng.uniform(0.85, 1.0))
    parts = [(y, 0.0)]
    d_mm = COIL_D_MM
    l_mm = mass / 7850.0 / (math.pi * (d_mm * 5e-4) ** 2) * 1e3
    arrive = crack_delay(v, miss)
    if arrive is not None:
        parts.append((_nm(_crack(v, d_mm, l_mm, miss, rng)) * crack_gain, float(times[-1]) + arrive))
    return parts, times


def crack_delay(v, miss):
    """Muzzle to N-wave, for a listener ``miss`` m off the line and just ahead of the cone (x = y (tan theta + 1)). None when subsonic."""
    if v <= C_AIR:
        return None
    return shock_arrival(v, miss * (math.tan(math.asin(C_AIR / v)) + 1.0), miss)


@sfx("coilgun", "Coilgun: one click per stage, a rising chirp at v / pitch that ends at 960-1600 Hz. fire has its coils and switch gaps heard; silent is a faint tick train, and a sonic crack when supersonic.",
     kind=("fire", "fire|silent"), speed=(75.0, "m/s at the muzzle: 45 EMG-01A, 75 GR-1"), stages=(COIL_STAGES, "stages"), pitch=(COIL_PITCH * 1e3, "mm between stages"),
     switch=("spark", "|".join(COIL_SWITCH)), miss=(COIL_FAR, "m off the line for the N-wave, far field from 12 m"), seed=(0, "variation"))
def coilgun(sr=DEFAULT_SR, kind="fire", speed=75.0, stages=COIL_STAGES, pitch=COIL_PITCH * 1e3, switch="spark", miss=COIL_FAR, seed=0):
    k = _kind(kind, ("fire", "silent"), "coilgun")
    sw = _kind(switch, COIL_SWITCH, "coilgun switch")
    if k == "silent":
        sw = "igbt"                                          # .06 and no noise: relatively silent
    rng = np.random.default_rng(int(seed))
    g = 1.0 if k == "fire" else 0.06
    parts, _ = _coil_shot(float(np.clip(speed, 5.0, 3000.0)), int(np.clip(stages, 1, 200)), max(float(pitch), 1.0) * 1e-3, sw, g, max(float(miss), COIL_FAR), 0.03, rng)
    return _lay(parts, sr, None)


GAUSS_V = 400.0                                             # above c, so it has an N-wave
GAUSS_MASS, GAUSS_STAGES = 0.03, 5                          # kg, stages


@sfx("gauss_rifle", "Gauss rifle: a coilgun of 5 stages that throws its slug supersonic, so the tick train is followed by a sonic crack.",
     speed=(GAUSS_V, "m/s: below 343 there is no crack"), stages=(GAUSS_STAGES, "stages"), switch=("igbt", "|".join(COIL_SWITCH)), miss=(COIL_FAR, "m off the line"), seed=(0, "variation"))
def gauss_rifle(sr=DEFAULT_SR, speed=GAUSS_V, stages=GAUSS_STAGES, switch="igbt", miss=COIL_FAR, seed=0):
    sw = _kind(switch, COIL_SWITCH, "gauss_rifle switch")
    rng = np.random.default_rng(int(seed))
    parts, _ = _coil_shot(float(np.clip(speed, 5.0, 3000.0)), int(np.clip(stages, 1, 200)), COIL_PITCH, sw, 0.6, max(float(miss), COIL_FAR), GAUSS_MASS, rng)
    return _lay(parts, sr)


# =============================================================================================
# LASER IMPACT
# =============================================================================================
@sfx("laser_impact", "A beam hitting something: a spark, ionised noise, a falling whistle. Pure design: no physics behind it.",
     energy=(0.5, "0..1: size of the spark and length of the rest"), seed=(0, "variation"))
def laser_impact(sr=DEFAULT_SR, energy=0.5, seed=0):
    """Spark + zapped noise + whistle."""
    rng = np.random.default_rng(int(seed))
    e = float(np.clip(energy, 0.0, 1.0))
    L = 0.15 + 0.6 * e
    n = _n(L + 0.05)
    y = np.zeros(n)
    _put(y, _nm(Ma.spark(sr=P, size=0.2 + 0.8 * e, seed=seed)), 0.0, 1.0)
    m = _n(L)
    ion = amb.hip(rng.uniform(-1, 1, m), 3000.0) * (0.6 + 0.4 * (Ma._lfn(rng, m, 90.0) > 0)) * np.exp(-np.arange(m) / (0.25 * m))
    _put(y, _nm(amb.lop(ion, 11000.0)) * np.minimum(np.arange(m) / _n(0.001), 1.0), 0.001, 0.45)
    t = np.arange(m) / P
    f = 3800.0 * (1100.0 / 3800.0) ** (np.arange(m) / m)
    wh = np.sin(2.0 * np.pi * np.cumsum(f) / P) * np.minimum(t / 0.004, 1.0) * np.exp(-np.arange(m) / (0.3 * m))
    _put(y, wh, 0.002, 0.25)
    return _shot(y, sr, 0.8, fade=0.0003, tail=0.03)


# =============================================================================================
# PHOTON TORPEDO
# =============================================================================================
TORPEDO_ENV = ((0.0, 1.0), (0.090, 0.0), (0.091, 0.0), (0.151, 0.89), (0.351, 0.0))   # (s, level): two pings
TORPEDO_F0, TORPEDO_SPAN = 2700.0, 7.0                     # f = F0 SPAN^env
TORPEDO_LOOP = (1.2e-3, 240e-3, 5.0, 23.0, 10000.0, 0.98)  # delay start, sweep, sweep s, high-pass Hz, low-pass Hz, feedback
TORPEDO_FDN = ((79.0, 127.8, 206.8, 48.8), 83.0 / 200.0, 1200.0, 0.3, 0.6)   # ms, matrix gain, send low-pass Hz, dry, wet


def torpedo_env(t):
    """Level of the two pings at ``t`` s."""
    a, b = zip(*TORPEDO_ENV)
    return np.interp(t, a, b, right=0.0)


@numba.njit(cache=True)
def _torpedo_loop(x, d, hp, lp, fb):
    n = x.shape[0]
    y = np.zeros(n)
    h = math.exp(-2.0 * math.pi * hp / 48000.0)
    g = 1.0 - math.exp(-2.0 * math.pi * lp / 48000.0)
    hx = hy = lo = 0.0
    for i in range(n):
        p = i - d[i]
        r = 0.0
        if p >= 0.0:
            j = int(p)
            f = p - j
            r = y[j] * (1.0 - f) + y[j + 1] * f if j + 1 < i else y[j]
        hy = h * (hy + r - hx)
        hx = r
        lo += g * (hy - lo)
        y[i] = x[i] + fb * lo
    return y


@numba.njit(cache=True)
def _fdn4(x, lens, g):
    n = x.shape[0]
    bufs = np.zeros((4, int(lens.max()) + 1))
    idx = np.zeros(4, np.int64)
    y = np.zeros(n)
    for i in range(n):
        o0 = bufs[0, idx[0]]
        o1 = bufs[1, idx[1]]
        o2 = bufs[2, idx[2]]
        o3 = bufs[3, idx[3]]
        y[i] = o0 + o1 + o2 + o3
        m = (o0 + o1 + o2 + o3, o0 - o1 + o2 - o3, o0 + o1 - o2 - o3, o0 - o1 - o2 + o3)
        for k in range(4):
            bufs[k, idx[k]] = x[i] + g * 0.5 * m[k]
            idx[k] = (idx[k] + 1) % lens[k]
    return y


@sfx("photon_torpedo", "Starship photon torpedo: two falling pings fed back through a delay that stretches over seconds, in a small reverb.",
     dur=(3.0, "s"), seed=(0, "variation"))
def photon_torpedo(sr=DEFAULT_SR, dur=3.0, seed=0):
    """An oscillator at F0 SPAN^env with amplitude env (two pings); a feedback delay swept from 1.2 to 241.2 ms over 5 s,
    high-passed and low-passed in the loop; x 0.3 out, and a low-passed send through a four-line Hadamard FDN."""
    n = _n(float(np.clip(dur, 0.5, 10.0)))
    t = np.arange(n) / P
    e = torpedo_env(t)
    x = e * np.sin(2.0 * np.pi * np.cumsum(TORPEDO_F0 * TORPEDO_SPAN ** e) / P)
    d0, sw, ts, hp, lp, fb = TORPEDO_LOOP
    y = _torpedo_loop(x, (d0 + sw * np.minimum(t / ts, 1.0)) * P, hp, lp, fb)
    lens, g, send, dry, wet = TORPEDO_FDN
    rev = _fdn4(amb.lop(dry * y, send), np.array([int(round(m * 1e-3 * P)) for m in lens]), g)
    return _fin(dry * y + wet * rev, sr, 0.8, fade=0.05)
