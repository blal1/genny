"""Machines, vehicles, interfaces and signal damage for industrial and science-fiction scenes:
transformers, gearboxes, bearings, a waveguide engine, turbo, tyres, a car passing, interface cues,
glitch and voice effects, statistical textures.

Rendered at the physical engine's rate (48 kHz) and resampled at the boundary, like ``genny.matter``.
Sources of every constant: research/SOURCES.md, section cyber; code taken from other projects: THIRD_PARTY_NOTICES.md.

Transformer: the core strain is even in the flux and steepens past the knee, the sound follows the strain velocity, a DC bias
brings the odd multiples of the mains. Gear: mesh rate z rpm / 60, its harmonics growing with load and wear, amplitude and phase
modulation at the shaft rates. Bearing: the four kinematic rates, a race defect shocking the housing once per passage, the ball
rates slipping 1-2 % at random, the housing fitted to recordings.
"""
from __future__ import annotations

import math

import numba
import numpy as np
from scipy import fft as sfft
from scipy.ndimage import uniform_filter1d
from scipy.signal import lfilter, resample, resample_poly

from . import filters as F
from . import osc as O
from . import spectral as S
from .calibrate import stats_snr
from .core import DEFAULT_SR, declick, pad_to, samples, to_mono
from .fx import REGISTRY as FX
from .fx import _per_channel, effect
from .matter import _bp, _fin, _fit, _hp, _lfn, _lp, _n, _rng, _tvbp, _unit
from .physical import modal as M
from .physical.core import SR
from .sfx import _round, sfx


def _modes(x, freqs, decays, gains):
    return M.render_modes(x, M.Modes(freqs, decays, gains, "zeros"))


def _ramp(a: float, b: float, n: int) -> np.ndarray:
    return np.full(n, float(a)) if b is None or b < 0 else np.geomspace(max(a, 1e-3), max(b, 1e-3), n)


def _impacts(n: int, phase: np.ndarray, rng, slip: float = 0.0, width: int = 6, walk: bool = False) -> np.ndarray:
    """One raised-cosine pulse of ``width`` samples each time ``phase`` (in cycles) passes an integer.
    ``slip`` is the standard deviation of the timing error as a fraction of the period: an error about
    a fixed position (gear teeth), or with ``walk`` an error of each spacing, which adds up (rolling
    elements slip and keep no memory of it)."""
    out = np.zeros(n + width)
    k = np.arange(1, int(phase[-1]) + 1)
    if len(k) == 0:
        return out[:n]
    err = slip * rng.standard_normal(len(k))
    at = np.interp(k + (np.cumsum(err) if walk else err), phase, np.arange(n))
    w = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(width) + 0.5) / width)
    for i in at.astype(int):
        out[i:i + width] += w
    return out[:n]


# =============================================================================================
# TRANSFORMER
# =============================================================================================

def magnetostriction(b):
    """Core strain (arbitrary units) for a flux density ``b`` in units of the saturation knee
    (about 1.7 T for grain-oriented steel). Even in ``b``: quadratic below the knee, steeper above."""
    b = np.asarray(b, float)
    return b * b * np.exp(0.9 * b * b)


CORE_REGIONS = (0.4, 0.6, 0.5, 1.2, 1.4)                    # rms strain ppm: outer limb, middle limb, yoke, corner, T-joint


CORE_B_NOM = 1.7
CORE_B_TABLE = ((1.71, 1.74, 1.72, 1.65, 1.72), (1.67, 1.62, 1.70, 1.69, 1.70), (1.61, 1.61, 1.69, 1.65, 1.69))


def core_strain(b, joints=0.0, packet=0):
    """Strain of the whole core: with ``joints`` > 0, the mean over five regions whose local induction is b sqrt(e_i / e_mean)
    (strain goes as b^2), blended from a uniform core. ``packet`` 1-3: the regions' measured induction (middle limb, outer
    limb, yoke, corner, T-joint) over the nominal 1.7 T instead."""
    j = float(np.clip(joints, 0.0, 1.0))
    if packet:
        k = np.array(CORE_B_TABLE[int(np.clip(packet, 1, 3)) - 1]) / CORE_B_NOM
        k = 1.0 + (j or 1.0) * (k - 1.0)
        return sum(magnetostriction(np.asarray(b) * ki) for ki in k) / len(k)
    if j <= 0:
        return magnetostriction(b)
    e = np.array(CORE_REGIONS)
    k = 1.0 + j * (np.sqrt(e / e.mean()) - 1.0)
    return sum(magnetostriction(np.asarray(b) * ki) for ki in k) / len(k)


GRID_EPSILON = {"DE": 1.1, "US": 1.35, "EE": 1.4, "RU": 1.5, "GB": 1.7, "SE": 2.6, "ZA": 2.7, "ES-PM": 3.4, "ES-GC": 4.7,
                "IS": 6.5, "FO": 9.8}
GRID_TAU = 10.0


def grid_frequency(n, f0, eps_mhz, rng, tau=GRID_TAU, rate=100.0):
    """Grid frequency as an Ornstein-Uhlenbeck process whose 1 s increments have the standard deviation eps (mHz),
    sampled at ``rate`` and interpolated."""
    m = int(n / SR * rate) + 2
    a = math.exp(-1.0 / (rate * tau))
    sd = float(eps_mhz) * 1e-3 / math.sqrt(2.0 * (1.0 - math.exp(-1.0 / tau)))
    x = np.empty(m)
    x[0] = sd * rng.standard_normal()
    w = rng.standard_normal(m) * sd * math.sqrt(1.0 - a * a)
    for i in range(1, m):
        x[i] = a * x[i - 1] + w[i]
    return f0 + np.interp(np.arange(n) / SR * rate, np.arange(m), x)


TRANSFORMER_LEQ = ((1.3, 65.5), (1.5, 68.4), (1.9, 72.4))     # core induction T, Leq dB of a test core
TRANSFORMER_BSAT = 2.0


def transformer_leq_db(tesla):
    """Noise level of a transformer core against its induction, interpolated in log B between measured points
    and extended by their end slopes."""
    b, l = np.log(np.array([t for t, _ in TRANSFORMER_LEQ])), np.array([v for _, v in TRANSFORMER_LEQ])
    x = np.log(np.asarray(tesla, float))
    lo = l[0] + (x - b[0]) * (l[1] - l[0]) / (b[1] - b[0])
    hi = l[-1] + (x - b[-1]) * (l[-1] - l[-2]) / (b[-1] - b[-2])
    return np.where(x < b[0], lo, np.where(x > b[-1], hi, np.interp(x, b, l)))


@sfx("transformer", "Power transformer / substation hum from the core's magnetostriction: 100 or 120 Hz and its multiples, tank resonances, loose-lamination buzz.",
     dur=(4.0, "s"), mains=(50.0, "Hz supply (50 or 60): the hum is at twice this"),
     flux=(0.8, "0.3..1.3 core induction relative to saturation: higher = more harmonics, harsher"),
     dc_bias=(0.0, "0..0.5 DC magnetisation: brings in the odd multiples (50, 150 Hz), a rougher growl"),
     load=(0.3, "0..1 winding current: adds pure 2f"), loose=(0.0, "0..1 loose laminations / panel: buzz and rattle"),
     size=(1.0, "0.3 small wall unit .. 3 substation: scales the tank resonances down"),
     strip=(0.0, "0..1 resonance of a lamination strip near 1.5 kHz, the 16th harmonic of 100 Hz"),
     joints=(0.0, "0..1 the core's regions differ: corners and T-joints strain 2-3 times the limbs, a harsher hum"),
     stress=(0.0, "0..1 mechanical stress on the sheets (clamping, corrugation): magnetostriction up 11 dB at 1"), seed=(0, "variation"),
     packet=(0, "1..3 the measured induction of one core packet in its five regions (0 = uniform)"),
     grid=("", "grid whose frequency wanders: " + " | ".join(GRID_EPSILON) + " (1 s increments 1.1 to 9.8 mHz); empty = steady"),
     tesla=(0.0, "core induction in T (0.5..2): sets flux = tesla / 2.0 and the level follows the measured noise rise, 0 dB at 1.5 T; 0 = use flux"))
def transformer(sr=DEFAULT_SR, dur=4.0, mains=50.0, flux=0.8, dc_bias=0.0, load=0.3, loose=0.0, size=1.0, strip=0.0, seed=0,
                joints=0.0, stress=0.0, packet=0, grid="", tesla=0.0):
    """The radiated sound is taken as the strain velocity (a large tank radiates as its surface
    velocity) plus the tank's modes. The grid frequency is off by a constant drawn from the seed, sd 3.8e-4
    (18.8 mHz at 50 Hz): over a few seconds a real grid holds still."""
    rng = _rng(seed, 71)
    n = _n(dur)
    gain_t = 1.0
    if tesla and tesla > 0:
        bt = float(np.clip(tesla, 0.5, 2.0))
        flux = bt / TRANSFORMER_BSAT
        gain_t = 10 ** ((float(transformer_leq_db(bt)) - float(transformer_leq_db(1.5))) / 20.0)
    f = float(np.clip(mains, 20.0, 400.0)) * (1.0 + 3.8e-4 * rng.standard_normal())
    if grid:
        if grid not in GRID_EPSILON:
            raise ValueError("transformer grid: " + " | ".join(GRID_EPSILON))
        th = 2 * np.pi * np.cumsum(grid_frequency(n, f, GRID_EPSILON[grid], _rng(seed, 73))) / SR
    else:
        th = 2 * np.pi * f * np.arange(n) / SR
    b = float(np.clip(flux, 0.3, 1.3)) * np.sin(th) + float(np.clip(dc_bias, 0.0, 0.5))
    v = np.gradient(core_strain(b, joints, packet) * 10 ** (11.0 * float(np.clip(stress, 0, 1)) / 20.0)) * SR / (2 * np.pi * 2 * f)
    v = v + float(np.clip(load, 0, 1)) * np.sin(2 * th + 0.7)
    lo = float(np.clip(loose, 0.0, 1.0))
    if lo > 0:
        # A loose sheet leaves its neighbour and lands again once per strain cycle.
        gate = np.maximum(np.abs(v) / (np.max(np.abs(v)) + 1e-12) - (1.0 - 0.7 * lo), 0.0) ** 2
        v = v + lo * 6.0 * np.max(np.abs(v)) * _bp(rng.standard_normal(n), 2200.0, 0.8) * gate
    s = float(np.clip(size, 0.3, 3.0))
    tank = np.array([212.0, 318.0, 437.0, 566.0, 801.0]) / s
    y = _unit(v) + 0.5 * _unit(_modes(v, tank, tank * 0.06, [1.0, 0.8, 0.6, 0.45, 0.3]))
    if strip > 0:                                            # a lamination strip, about 1.5 kHz, Q about 10 (decay 450 /s)
        y = y + float(np.clip(strip, 0, 1)) * _unit(_modes(v, np.array([1500.0]), np.array([450.0]), [1.0]))
    y = y * (1.0 + 0.04 * _lfn(rng, n, 0.3))
    return _fin(_lp(y, 5000.0, 2), sr, 0.5, 0.05, 0.08) * gain_t


# =============================================================================================
# GEARS
# =============================================================================================

def gear_frequencies(rpm: float, teeth: int, teeth2: int) -> dict:
    """Rates in Hz of a gear pair driven at ``rpm`` on the ``teeth`` side."""
    z1, z2 = max(int(teeth), 1), max(int(teeth2), 1)
    f1 = float(rpm) / 60.0
    mesh = z1 * f1
    return {"shaft_in": f1, "shaft_out": f1 * z1 / z2, "mesh": mesh, "hunting": mesh * math.gcd(z1, z2) / (z1 * z2)}


@sfx("gearbox", "Gear whine: the tooth-mesh tone (teeth x rpm / 60) with shaft-rate sidebands, housing ring and backlash rattle.",
     dur=(3.0, "s"), rpm=(1800.0, "input shaft"), rpm_end=(-1.0, "glide to this rpm (-1 = steady)"),
     teeth=(23, "driving gear"), teeth2=(41, "driven gear"), load=(0.6, "0 idle (rattles) .. 1 loaded (whines)"),
     wear=(0.3, "0 ground and quiet .. 1 worn: more mesh harmonics, a damaged tooth knocking once per turn"),
     eccentricity=(0.3, "0..1 run-out of the shafts: strength of the sidebands (the 'singing' roughness)"),
     backlash=(0.3, "0..1 clearance: rattle when the load is low"), housing=(1050.0, "Hz first casing resonance (others at 1.42, 2.05, 2.9 times), or a list of peaks '935,1020,1140', or car"),
     friction=(0.0, "0 = none; flank friction coefficient (about 0.1): a sawtooth at the mesh rate that changes sign at the pitch point, more upper harmonics under load"),
     mount_mass=(0.0, "kg added on the mounts (blocking masses on the dominant paths): the mesh order and everything above drop as a one-pole low-pass at Z / (2 pi m)"),
     seed=(0, "variation"))
def gearbox(sr=DEFAULT_SR, dur=3.0, rpm=1800.0, rpm_end=-1.0, teeth=23, teeth2=41, load=0.6, wear=0.3,
            eccentricity=0.3, backlash=0.3, housing=1050.0, friction=0.0, seed=0, mount_mass=0.0):
    rng = _rng(seed, 72)
    n = _n(dur)
    z1, z2 = int(np.clip(teeth, 3, 400)), int(np.clip(teeth2, 3, 400))
    f1 = _ramp(float(np.clip(rpm, 10.0, 60000.0)), rpm_end, n) / 60.0
    p1 = np.cumsum(f1) / SR                       # input shaft, turns
    p2 = p1 * z1 / z2
    ecc, w, ld = float(np.clip(eccentricity, 0, 1)), float(np.clip(wear, 0, 1)), float(np.clip(load, 0, 1))
    # Run-out changes the tooth load (amplitude) and the speed of each gear (phase): index
    # beta = fluctuation x teeth, here 1 % at full `eccentricity`, capped at 1.5 rad.
    beta = lambda z: min(0.01 * ecc * z, 1.5) / (2 * np.pi)
    pm = z1 * p1 + beta(z1) * np.sin(2 * np.pi * p1) + 0.6 * beta(z2) * np.sin(2 * np.pi * p2 + 1.0)
    am = 1.0 + 0.5 * ecc * (np.cos(2 * np.pi * p1) + 0.6 * np.cos(2 * np.pi * p2 + 2.0))
    # One damaged tooth on the driving gear: the mesh swells for one tooth period per turn, and knocks.
    tooth = _impacts(n, p1, rng, width=max(int(SR / (z1 * float(f1.mean()))), 6))
    am = am + 1.5 * w * tooth
    te = np.zeros(n)
    top = 0.4 * SR / (z1 * float(f1.max()))
    for k in range(1, int(min(top, 12)) + 1):
        # Load: 21 dB on the mesh line and 7 dB on its double between 10 % and full load.
        te += k ** -(2.2 - 1.4 * w) * max(ld, 0.05) ** (k ** -1.6) * np.sin(2 * np.pi * k * pm + rng.uniform(0, 6.28))
    exc = te * am
    exc += w * 4.0 * _impacts(n, p1, rng)
    bl = float(np.clip(backlash, 0, 1)) * (1.0 - ld)
    if bl > 0:
        hits = _impacts(n, z1 * p1, rng, slip=0.25)
        exc += bl * 3.0 * hits * (rng.random(n) < 0.5) * (0.5 + 0.5 * np.cos(2 * np.pi * p1))
    if friction > 0:                                         # flank friction through `machines.tooth_friction`, its 3 kHz top as `machines._geared`
        from .machines import tooth_friction
        exc += 2.0 * tooth_friction(f1 * 60.0, z1, n, ld, float(np.clip(friction, 0, 1)), int(seed), top=3000.0)
    h, w = gear_housing(housing)
    d = np.array(GEAR_HOUSING_CAR_DECAY) if isinstance(housing, str) and housing.strip() == "car" else h * 0.02
    y = 0.5 * _unit(exc) + _unit(_modes(exc, h, d, w))
    y = _lp(_hp(y, 60.0), 9000.0, 2)
    if mount_mass > 0:
        a = math.exp(-2 * math.pi * min(blocking_mass_fc(mount_mass), 0.45 * SR) / SR)
        y = lfilter([1 - a], [1, -a], y)
    return _fin(y, sr, 0.5, 0.01, 0.03)


MOUNT_PLATE_H = 0.005


def blocking_mass_fc(mass, h=MOUNT_PLATE_H, E=2.1e11, rho=7850.0, nu=0.3):
    """Blocking mass on a mount: IL = 20 log10 |1 + j w m / Z|, Z = 8 sqrt(B rho h) the drive-point
    impedance of a steel plate, i.e. a one-pole low-pass at Z / (2 pi m). Returns that corner in Hz."""
    B = E * h ** 3 / (12.0 * (1.0 - nu ** 2))
    return 8.0 * math.sqrt(B * rho * h) / (2.0 * math.pi * max(float(mass), 1e-9))


# =============================================================================================
# BEARINGS
# =============================================================================================

GEAR_HOUSING_CAR = (935.0, 1020.0, 1140.0)
GEAR_HOUSING_CAR_DECAY = (94.0, 298.0, 346.0)                # 1/s, pi x measured half-power bandwidth


def gear_housing(housing):
    """Casing modes and weights: one number = a first mode and its 1.42, 2.05, 2.9 multiples; a list or 'car' =
    those peaks at equal weight."""
    if isinstance(housing, str) and housing.strip() == "car":
        housing = GEAR_HOUSING_CAR
    if isinstance(housing, str):
        housing = [float(v) for v in housing.replace(";", ",").split(",") if v.strip()]
    if np.ndim(housing):
        h = np.clip(np.asarray(housing, float), 200.0, 8000.0)
        return h, [1.0] * len(h)
    return float(np.clip(housing, 200.0, 8000.0)) * np.array([1.0, 1.42, 2.05, 2.9]), [1.0, 0.6, 0.4, 0.25]


def bearing_frequencies(rpm: float, balls: int = 9, ball_mm: float = 7.94, pitch_mm: float = 39.04,
                        angle_deg: float = 0.0) -> dict:
    """Kinematic rates in Hz for a fixed outer race: ball pass outer / inner, ball spin, cage (FTF)."""
    fr = float(rpm) / 60.0
    r = ball_mm / pitch_mm * math.cos(math.radians(angle_deg))
    return {"shaft": fr, "bpfo": balls / 2.0 * fr * (1.0 - r), "bpfi": balls / 2.0 * fr * (1.0 + r),
            "bsf": pitch_mm / (2.0 * ball_mm) * fr * (1.0 - r * r), "ftf": fr / 2.0 * (1.0 - r)}


BEARING_SIZES = {"6004": (9, 6.35, 31.0), "6205": (9, 7.94, 39.04), "6203": (8, 6.747, 28.5), "6206": (9, 9.525, 46.0),
                 "51104": (12, 5.556, 27.5, 90.0), "51104_alt": (6, 5.556, 27.5, 90.0)}   # balls, ball mm, pitch mm[, contact angle]
BEARING_LIMITS = {"6004": (24000.0, 11000.0), "6205": (18000.0, 8500.0), "6203": (24000.0, 12000.0),
                  "6206": (13000.0, 7500.0), "51104": (9500.0, 9500.0)}     # rpm: open, sealed

BEARING_STATES = ("healthy", "worn", "dry", "outer", "inner", "ball", "loose", "misaligned", "pitting", "crack", "spall",
                  "polished", "cage")


def load_zone(angle, load=0.0):
    """Share of the load a rolling element carries at ``angle`` (rad) from the load line. ``load`` 0: cos^2 of the
    old model; above 0, the Stribeck distribution (1 - (1 - cos psi) / (2 eps))^1.5 with eps = 0.2 + 0.8 load
    (0.5 = half the ring loaded)."""
    if load <= 0:
        return np.maximum(np.cos(angle), 0.0) ** 2
    eps = 0.2 + 0.8 * float(np.clip(load, 0.0, 1.0))
    return np.clip(1.0 - (1.0 - np.cos(angle)) / (2.0 * eps), 0.0, None) ** 1.5


MIRROR_D = 0.09                                              # m aperture of the parabolic recording mirror


def mirror_gain(f, c=343.0, d=MIRROR_D, cap_db=30.0):
    """Pressure gain of a parabolic mirror against a bare microphone: (pi D f / c)^2 above c / (pi D), 1 below."""
    g = (np.pi * d * np.asarray(f, float) / c) ** 2
    return np.clip(g, 1.0, 10 ** (cap_db / 20.0))


@sfx("bearing", "Rolling bearing on a spinning shaft, healthy or failing: rumble, dry squeal, the periodic ticking of a race or ball defect, a loose or misaligned shaft.",
     dur=(3.0, "s"), rpm=(1750.0, "shaft"), rpm_end=(-1.0, "glide to this rpm (-1 = steady)"),
     state=("healthy", "|".join(BEARING_STATES)), severity=(0.6, "0..1 how far gone"),
     balls=(9, "rolling elements"), ball_mm=(7.94, "ball diameter"), pitch_mm=(39.04, "pitch diameter"),
     size=("", "designation, replaces balls, ball_mm, pitch_mm: 6004 | 6205 | 6203 | 6206 | 51104 (thrust: angle 90, the balls "
               "pass both races at the same rate) | 51104_alt (rollers set alternately: a defect meets every other one, BPF / 2); "
               "catalogue speed limits open / sealed: 6205 18000 / 8500 rpm"),
     angle=(0.0, "deg contact angle: 0 deep groove, 15-40 angular contact (catalogue suffix), 90 thrust"),
     creep=(0, "1 = the ball rates run a constant 1 to 2 % slow (drawn from seed)"),
     resonance=(1000.0, "Hz main housing resonance (others at 0.7, 2 and 4.1 times it), up to 19000 for an accelerometer's view"),
     load=(0.0, "0 = default load zone; 0..1 radial load: how much of the ring carries the balls (0.5 = half)"),
     rotating=("inner", "inner (shaft) | outer: the outer ring turns (planet gear, wheel hub): an inner-race defect stays in the load zone and ticks evenly, an outer-race defect swells once per turn"),
     defect=(0.0, "deg of race the defect spans: 0 = a point; wider = an entry and an exit shock, merging past the ball spacing"),
     ring=(1.0, "x on the housing ring time: 3-5 for a slow crane or turret bearing"),
     mirror=(0, "1 = recorded through a 90 mm parabolic mirror: highs lifted as (pi D f / c)^2"), seed=(0, "variation"))
def bearing(sr=DEFAULT_SR, dur=3.0, rpm=1750.0, rpm_end=-1.0, state="healthy", severity=0.6, balls=9, ball_mm=7.94,
            pitch_mm=39.04, resonance=1000.0, seed=0, size="", angle=0.0, creep=0, load=0.0, rotating="inner", defect=0.0,
            ring=1.0, mirror=0):
    """States beyond the classic four: pitting (several small pits, uneven ticks at BPFO), crack (sharp inner-race
    clicks), spall (an extended worn spall: no sharp shocks, a level that follows the shaft harmonics and noise
    swelling in the load zone), polished (the same, with a small shock as each ball leaves it), cage (irregular
    knocks at the cage rate)."""
    if size:
        if str(size) not in BEARING_SIZES:
            raise ValueError(f"bearing size {size!r}: use one of {', '.join(BEARING_SIZES)}")
        geo = BEARING_SIZES[str(size)]
        balls, ball_mm, pitch_mm = geo[:3]
        if len(geo) > 3:
            angle = geo[3]
    if state not in BEARING_STATES:
        raise ValueError(f"bearing state {state!r}: use one of {', '.join(BEARING_STATES)}")
    if rotating not in ("inner", "outer"):
        raise ValueError("bearing rotating must be inner or outer")
    rng = _rng(seed, 73)
    n = _n(dur)
    sv = float(np.clip(severity, 0.0, 1.0))
    fr = _ramp(float(np.clip(rpm, 1.0, 60000.0)), rpm_end, n) / 60.0
    zone = lambda a: load_zone(a, float(load))
    k = bearing_frequencies(60.0, int(np.clip(balls, 3, 60)), float(ball_mm), float(pitch_mm),
                            float(np.clip(angle, 0.0, 90.0)))   # per shaft turn
    if creep:
        lag = 1.0 - _rng(seed, 74).uniform(0.01, 0.02)
        k = {key: v if key == "shaft" else v * lag for key, v in k.items()}
    turns = np.cumsum(fr) / SR
    shaft, cage = 2 * np.pi * turns, 2 * np.pi * turns * k["ftf"]
    speed = np.sqrt(fr / float(fr.max()))
    # Housing: four modes at 0.7, 1, 2 and 4.1 times `resonance`, ringing about 7 ms. With the default
    # these are the 700 Hz, 1 kHz, 2 kHz and 4.1 kHz peaks of recordings of an SKF 6004.
    f0 = float(np.clip(resonance, 200.0, 19000.0))
    res = np.minimum(f0 * np.array([0.7, 1.0, 2.0, 4.1]), 0.45 * SR)
    dec = np.array([120.0, 140.0, 180.0, 260.0]) / float(np.clip(ring, 0.2, 10.0))
    housing = lambda force: _modes(force, res, dec, [0.7, 1.0, 1.0, 0.3])
    # Rolling noise: surface roughness under the balls ringing the housing, over the low-mid rumble of
    # the shaft and frame; each ball passing the load modulates it slightly.
    roll = _unit(housing(rng.standard_normal(n))) + 0.5 * _unit(_lp(_hp(rng.standard_normal(n), 150.0), 900.0, 2))
    roll = _unit(roll) * (1.0 + 0.03 * np.cos(shaft * k["bpfo"]))
    y = roll * speed + 0.15 * np.sin(shaft)
    force, level = np.zeros(n), 1.0
    if state == "worn":
        y += sv * (0.9 * roll * speed * (1.0 + 0.5 * _lfn(rng, n, 30.0)) + 0.6 * _unit(_lp(rng.standard_normal(n), 300.0)))
        force = (rng.random(n) < sv * 40.0 * fr / SR) * rng.exponential(1.0, n)
    elif state == "dry":
        # Unlubricated cage and balls stick and slip: a narrow squeal that breaks up once per turn.
        sq = _bp(rng.standard_normal(n), f0 * 2.6, 60.0) * np.clip(0.6 + 0.6 * np.sin(shaft) + 0.5 * _lfn(rng, n, 12.0), 0.0, None)
        y += sv * 2.5 * _unit(sq)
    elif state == "outer":
        # A fixed defect in the load zone: every ball hits it alike, so it is a steady, soft ticking.
        force, level = _impacts(n, turns * k["bpfo"], rng, slip=0.02, width=24, walk=True), 0.45
        if rotating == "outer":
            force = force * zone(shaft)
    elif state == "inner":
        force, level = _impacts(n, turns * k["bpfi"], rng, slip=0.01, walk=True), 1.4
        if rotating == "inner":
            force = force * zone(shaft)
    elif state == "pitting":
        force, level = _impacts(n, turns * k["bpfo"], rng, slip=0.02, width=16, walk=True) * rng.uniform(0.3, 1.0, n), 0.35
        force = force * (zone(shaft) if rotating == "outer" else 1.0)
    elif state == "crack":
        force, level = _impacts(n, turns * k["bpfi"], rng, slip=0.01, width=3, walk=True), 1.6
        force = force * (zone(shaft) if rotating == "inner" else 1.0)
    elif state in ("spall", "polished"):
        # No sharp shock once the spall is long and worn: a deterministic swell on the shaft harmonics plus
        # rolling noise amplitude-modulated while the rough patch crosses the load zone.
        y += sv * (0.4 * sum(np.sin(h * shaft + h) / h for h in range(1, 5)) + 1.2 * roll * speed * zone(shaft))
        if state == "polished":
            force, level = _impacts(n, turns * k["bpfi"], rng, slip=0.01, width=10, walk=True) * zone(shaft), 0.3
    elif state == "cage":
        force, level = _impacts(n, turns * k["ftf"] * 2.0, rng, slip=0.15, width=8, walk=True) * rng.uniform(0.2, 1.0, n), 1.0
        y += sv * 0.4 * roll * speed * (1.0 + np.cos(cage))
    elif state == "ball":
        # The damaged ball strikes a race twice per spin, but only inside the load zone, where the cage
        # brings it once per turn, and while its defect faces a race. Slip is 3 % per strike, above the
        # usual 1 to 2 %: recordings of a ball defect show no line at the strike rate, only the cage
        # rate and its multiples, and that much slip is what smears it.
        gate = np.clip(_lfn(rng, n, float(fr.mean()) * k["ftf"]) + 0.3, 0.0, 1.0)
        force, level = _impacts(n, turns * 2.0 * k["bsf"], rng, slip=0.03, width=12, walk=True) * np.maximum(np.cos(cage), 0.0) ** 2 * gate, 0.5
        y *= 1.0 + 0.25 * sv * np.cos(cage)
    elif state == "loose":
        # Clearance lets the shaft orbit: a low rumble that swells every turn and every other turn.
        rumble = _unit(_lp(_hp(rng.standard_normal(n), 250.0), 900.0, 2)) * (1.0 + 0.6 * np.cos(shaft) + 0.3 * np.cos(shaft / 2.0))
        y += sv * (2.5 * rumble + 0.3 * (np.sin(shaft) + 0.5 * np.sin(2 * shaft) + 0.4 * np.sin(0.5 * shaft)))
    elif state == "misaligned":
        y += sv * (0.8 * np.sin(2 * shaft) + 0.3 * np.sin(shaft) + 0.5 * roll * speed * np.cos(2 * shaft))
    if force.any() and defect > 0 and state in ("outer", "inner", "pitting", "crack"):
        # The ball drops in at the entry edge and strikes the exit edge harder, a race arc later.
        rel = k["ftf"] if state in ("outer", "pitting") else 1.0 - k["ftf"]
        lag = int(float(np.clip(defect, 0.0, 120.0)) / 360.0 / (float(fr.mean()) * rel) * SR)
        force = 0.5 * force + np.concatenate([np.zeros(min(lag, n)), force[:max(n - lag, 0)]])
    if force.any():
        # Impact level against the rolling noise: about equal at severity 0.6 (as recorded), 14 dB over at 1.
        y += level * 5.0 * sv ** 3 * math.sqrt(float(np.mean(y * y))) * _unit(housing(force * speed))
    y = _lp(_hp(y, 25.0), max(6000.0, min(1.5 * f0, 0.45 * SR)), 2)
    if mirror:
        f = np.fft.rfftfreq(n, 1.0 / SR)
        y = np.fft.irfft(np.fft.rfft(y) * mirror_gain(f), n)
    return _fin(y, sr, 0.5, 0.01, 0.03)


# =============================================================================================
# ENGINE AS WAVEGUIDES, TURBO, TYRES, PASS-BY
# =============================================================================================
# Each cylinder, over one 720 degree cycle x in [0, 1):
#   intake valve   sin(4 pi x)              on 0    < x < 0.25
#   exhaust valve  -sin(4 pi x)             on 0.75 < x < 1
#   piston         cos(4 pi x)
#   ignition       sin(2 pi (x - 0.5) / t)  on 0.5  < x < 0.5 + t / 2
# feeds an intake and an exhaust waveguide whose reflection at the cylinder end follows the valve:
# r = r_open v + r_closed (1 - v). The exhausts join in a collector, a straight pipe and a muffler
# of four short parallel guides. Left out: the cylinder itself as a waveguide whose length follows the piston
# (cylinder volume and compression ratio), and the outlet pipe after the muffler. Backfire: while the engine
# revs down, each cycle fires an ignition-shaped pulse into the exhaust with a probability that is
# multiplied by itself after every pop.

@numba.njit(cache=True)
def _port(x, a, b, ramp):
    """Port area fraction: 0 outside [a, b] of the cycle, reaching 1 within ``ramp``."""
    return min(1.0, min(x - a, b - x) / ramp) if a < x < b else 0.0


@numba.njit(cache=True)
def _pipe_engine_k(inc, offs, noise, pops, n_ex, n_in, n_pipe, n_muf, pf, igf, ign_t, bank, mode, ch_len, ch_r, ch_g,
                   dj, t_in, t_ex, n_cyl, n_out, r_out, r_ie=-0.7576, r_xe=0.06):
    n, C, nb, S = len(inc), len(offs), len(n_pipe), len(ch_len)
    m_dj = dj.shape[1]
    cylb = np.zeros((C, max(n_cyl, 1)))
    ob = np.zeros(max(n_out, 1))
    oret = 0.0
    exu, exl = np.zeros((C, n_ex.max())), np.zeros((C, n_ex.max()))
    inu, inl = np.zeros((C, n_in)), np.zeros((C, n_in))
    pu, pl = np.zeros((nb, n_pipe.max())), np.zeros((nb, n_pipe.max()))
    mu, ml = np.zeros((4, n_muf.max())), np.zeros((4, n_muf.max()))
    m_ch = max(ch_len.max(), 1) if S else 1
    fw, bw = np.zeros((nb, max(S, 1), m_ch)), np.zeros((nb, max(S, 1), m_ch))   # expansion chamber sections, per bank
    ca, cb = np.zeros(max(S, 1)), np.zeros(max(S, 1))
    out = np.zeros((3, n))                                   # intake, exhaust, block vibration
    count = np.zeros(nb)
    for c in range(C):
        count[bank[c]] += 1.0
    coll = np.zeros(nb)
    back = np.zeros(nb)                                      # wave returning from each pipe into its headers
    x = 0.0
    for t in range(n):
        x = (x + inc[t]) % 1.0
        coll[:] = 0.0
        intake, vib = 0.0, 0.0
        it = ign_t[t]
        for c in range(C):
            xc = (x + offs[c] + dj[c, t % m_dj]) % 1.0
            if mode == 0:                                    # 4 strokes, 720 degrees, ignition at the middle of the cycle
                xi, xe = (xc + t_in) % 1.0, (xc + t_ex) % 1.0
                iv = math.sin(4 * math.pi * xi) if xi < 0.25 else 0.0
                ev = -math.sin(4 * math.pi * xe) if xe > 0.75 else 0.0
                cyl = pf * math.cos(4 * math.pi * xc)
                if 0.5 < xc < 0.5 + it / 2:
                    cyl += igf[t] * math.sin(2 * math.pi * (xc - 0.5) / it)
            elif mode == 1:                                  # 2 strokes, 360 degrees from TDC: exhaust ports 86, transfer 116
                ev = math.sin(math.pi * (xc - 86.0 / 360.0) / (188.0 / 360.0)) if 86.0 / 360.0 < xc < 274.0 / 360.0 else 0.0
                iv = math.sin(math.pi * (xc - 116.0 / 360.0) / (128.0 / 360.0)) if 116.0 / 360.0 < xc < 244.0 / 360.0 else 0.0
                cyl = pf * math.cos(2 * math.pi * xc)
                if xc < it:                                  # the same crank angle as the 4-stroke pulse: twice its share of a 360 degree cycle
                    cyl += igf[t] * math.sin(math.pi * xc / it)
            else:                                            # Wankel chamber, 1080 degrees of shaft from firing TDC
                ev = _port(xc, 195.0 / 1080.0, 588.0 / 1080.0, 28.0 / 1080.0)
                iv = _port(xc, 572.0 / 1080.0, 860.0 / 1080.0, 28.0 / 1080.0)
                cyl = pf * math.cos(4 * math.pi * xc)        # V = Vmin + Vd (1 - cos(2 psi / 3)) / 2, the same period as a 4-stroke's cos(4 pi x)
                if xc < it / 3:
                    cyl += igf[t] * math.sin(3 * math.pi * xc / it)
            if n_cyl > 0:                                    # the cylinder as a guide: both closed ends reflect 0.75
                kc = t % n_cyl
                cyl += 0.5625 * (1.0 - 0.9 * max(iv, ev)) * cylb[c, kc]
                cylb[c, kc] = cyl
            vib += cyl
            # exhaust guide: left end = valve, right end = collector
            i = t % n_ex[c]
            r = 0.01 * ev + 0.7145 * (1.0 - ev)              # an open valve reflects almost nothing
            u, lo = exu[c, i], exl[c, i]
            b = bank[c]
            exu[c, i] = (1.0 - r) * cyl * 0.5 + r * lo
            exl[c, i] = back[b] / count[b] + r_xe * u
            coll[b] += u * (1.0 - r_xe)
            # intake guide: left end = open air (inverting), right end = valve
            i = t % n_in
            r = 0.006 * iv + 1.0 * (1.0 - iv)
            u, lo = inu[c, i], inl[c, i]
            inu[c, i] = r_ie * lo
            inl[c, i] = (1.0 - r) * cyl * 0.5 + noise[t] * iv + r * u
            intake += lo * (1.0 - r_ie)
        u = 0.0
        for b in range(nb):
            i = t % n_pipe[b]
            ub, lo = pu[b, i], pl[b, i]
            pu[b, i] = coll[b] / count[b] + pops[t] + 0.0617 * lo
            back[b] = lo * (1.0 - 0.0617)
            if S:                                            # stepped chamber: every area step is a scattering junction
                for s in range(S):
                    k = t % ch_len[s]
                    ca[s], cb[s] = fw[b, s, k] * ch_g[s], bw[b, s, k] * ch_g[s]
                fw[b, 0, t % ch_len[0]] = ub
                pl[b, i] = cb[0]
                for s in range(S - 1):
                    r = ch_r[s]
                    fw[b, s + 1, t % ch_len[s + 1]] = (1.0 + r) * ca[s] - r * cb[s + 1]
                    bw[b, s, t % ch_len[s]] = r * ca[s] + (1.0 - r) * cb[s + 1]
                ub = ca[S - 1]
            u += ub
        u /= nb
        u += oret
        ret, ex = 0.0, 0.0
        for j in range(4):
            k = t % n_muf[j]
            a, b2 = mu[j, k], ml[j, k]
            mu[j, k] = u * 0.25
            ml[j, k] = -0.142 * a
            ex += a * 1.142
            ret += b2
        for b in range(nb):
            if S:
                bw[b, S - 1, t % ch_len[S - 1]] = ret
            else:
                pl[b, t % n_pipe[b]] = ret
        if n_out > 0:                                        # outlet pipe: the open end sends r_out back into the muffler
            ko = t % n_out
            a = ob[ko]
            ob[ko] = ex
            oret = r_out * a
            ex = (1.0 - r_out) * a
        out[0, t], out[1, t], out[2, t] = intake, ex, vib
    return out


_CYCLES = {"4": (0, 120.0), "2": (1, 60.0), "rotary": (2, 180.0)}     # kernel mode, rpm per cycle per second: 720, 360, 1080 degrees


def _along(v, n):
    """A value or a curve of any length as ``n`` samples."""
    v = np.asarray(v, float)
    return np.full(n, float(v)) if v.ndim == 0 else np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, len(v)), v)


TRANSIENT_HP = 4000.0
TRANSIENT_F0 = 80.0
TRANSIENT_TAU = 0.003


def firing_transients(inc, C, rng, amount):
    """One exponentially decaying burst of noise above ``TRANSIENT_HP`` (time constant ``TRANSIENT_TAU``) per
    firing; full while the firing rate f = inc sr C is under ``TRANSIENT_F0``, fading linearly to 0 at twice it."""
    n = len(inc)
    fires = np.cumsum(inc) * C
    idx = np.flatnonzero(np.diff(np.floor(fires)) > 0) + 1
    imp = np.zeros(n)
    imp[idx] = 1.0
    m = int(5 * TRANSIENT_TAU * SR)
    burst = np.convolve(imp, np.exp(-np.arange(m) / (TRANSIENT_TAU * SR)))[:n]
    x = _hp(rng.standard_normal(n), TRANSIENT_HP) * burst
    f = inc * SR * C
    return 2.0 * amount * x * np.clip(2.0 - f / TRANSIENT_F0, 0.0, 1.0)


@sfx("pipe_engine", "Engine built from its plumbing: each cylinder's valves or ports gate an intake and an exhaust waveguide, joined by one or two collectors, pipes and a muffler. Four-stroke by default, also two-stroke and Wankel. Rasps, drones and lopes like a real exhaust.",
     dur=(3.0, "s"), rpm=(1800.0, "crankshaft, or a curve of values over the sound"), rpm_end=(-1.0, "glide to this rpm (-1 = steady)"), cylinders=(4, "1..16 (rotary: rotors, 1..5)"),
     load=(0.5, "0 coasting .. 1 full throttle: combustion strength and intake roar; a number or a curve"),
     header=(1.6, "ms exhaust runner length per cylinder: short = raspy, long = deep"),
     pipe=(6.0, "ms straight pipe before the muffler (the drone)"),
     lope=(0.0, "0 even firing .. 1 uneven (odd cylinders fire early: V-twin / cammed idle)"),
     unequal=(0.15, "0..0.5 spread of the header lengths (unequal headers burble)"),
     mic=(0.3, "0 at the tailpipe .. 1 at the air intake"),
     backfire=(0.0, "0..0.99 chance of a pop in the exhaust per engine cycle, only while the rpm falls (rpm_end below rpm, or a falling curve); each pop makes the next less likely"),
     seed=(0, "variation"),
     offsets=(None, "firing times of each cylinder in fractions of a cycle, in firing order; replaces cylinders and lope"),
     banks=(None, "exhaust bank of each firing slot: 'LRLLRLRR' or [0, 1, ...]: two collectors and two pipes, summed before the muffler"),
     headers=(None, "ms runner length per firing slot; replaces header and unequal"),
     cycle=(4, "4 | 2 | rotary"),
     muffler=(None, "ms of the four parallel muffler tubes; default .146 .1875 .208 .25; .23 .34 .45 .57 darker"),
     extractor=(0.0, "ms of the extractor runner each header feeds before the collector (added to the header)"),
     cylinder=(0.0, "ms: the cylinder itself as a short guide between the valves (0 = off; about 0.2), a hollow ring on each firing"),
     timing=(None, "valve timing: 'intake,exhaust' shifts in fractions of a cycle (cam phasing), e.g. '-0.043,-0.004'"),
     jitter=(0.0, "0..1 each cylinder's firing wanders on its own (low-passed at 75 Hz): an engine that is not machined perfect"),
     warp=(0.0, "0..1 the pipe stretches and shrinks with each firing (two delays modulated in antiphase): a cheap growl"),
     block=(1.0, "x on the engine-block vibration heard with the exhaust: 3-5 from inside the car"),
     radiate=(0, "1 = the tailpipe radiates as a monopole: the sound is the rate of change of the outflow (brighter)"),
     units=("ms", "ms | m: header, headers, pipe, extractor, cylinder, outlet, muffler, expansion_ms given as lengths in metres"),
     gas_temp=(600.0, "K exhaust gas temperature when units=m: c = 331.45 sqrt(T / 273.15) (900 K: 601 m/s)"),
     side=(None, "Hz of side-branch resonators on the pipe that cancel their frequency (quarter-wave tubes, Helmholtz bottles), e.g. '110,857'"),
     expansion=(0.0, "area ratio of an expansion chamber on the pipe (0 = none; 4-16 a silencer drum)"),
     expansion_ms=(1.0, "ms length of that chamber"),
     outlet=(0.0, "ms of a tail outlet pipe after the muffler (0 = none)"),
     outlet_r=(0.3, "reflection the outlet's open end sends back into the muffler"),
     alternator=(0.0, "0..1 alternator whine: 36 ripples per alternator turn, pulley 2.8"),
     roots=(0.0, "0..1 Roots supercharger whine: two three-lobe rotors, driven at 2 x crank (6 per rotor turn)"),
     view=("tail", "tail | cabin (heard from the driver's seat: body panels low-pass, 85 Hz boom)"),
     crank=(0.0, "0..1 crank position flutter shared by all cylinders: 0.331 cycle of noise low-passed at 57 Hz at 1"),
     norm4=(0, "1 = scale the collector by 4 / C above four cylinders"),
     intake_gain=(1.0, "gain of the intake bus"), outlet_gain=(1.0, "gain of the exhaust outlet bus"),
     ends=("engine", "free-end reflections: engine (intake -0.7576, collector 0.06) | paper (-0.5, 0.1)"),
     transients=(0.0, "0..1 a short burst above 4 kHz on every firing (the phase transients of the combustion), heard while the firing rate stays under 80 Hz, gone by 160 Hz"))
def pipe_engine(sr=DEFAULT_SR, dur=3.0, rpm=1800.0, rpm_end=-1.0, cylinders=4, load=0.5, header=1.6, pipe=6.0,
                lope=0.0, unequal=0.15, mic=0.3, backfire=0.0, seed=0, offsets=None, banks=None, headers=None, cycle=4,
                fire=None, pops=None, chamber=None, muffler=None, extractor=0.0, cylinder=0.0, timing=None, jitter=0.0,
                warp=0.0, block=1.0, radiate=0, units="ms", gas_temp=600.0, side=None, expansion=0.0, expansion_ms=1.0,
                outlet=0.0, outlet_r=0.3, alternator=0.0, roots=0.0, view="tail", crank=0.0, norm4=0, intake_gain=1.0,
                outlet_gain=1.0, ends="engine", transients=0.0):
    """`lope` is a cycle asymmetry.

    Python callers also have ``fire`` (ignition gain, 0 = fuel cut, a number or a curve), ``pops`` (a signal
    added to the collector, ``_fin`` scale 2.5 = one backfire) and ``chamber`` ((delays in samples, junction
    reflections, section gains) of a stepped expansion chamber between the pipe and the muffler).
    Ports: exhaust 86 and transfer 116 degrees from TDC, 360 degree cycle; Wankel exhaust 195-588, intake 572-860,
    ramp 28 of 1080 degrees of shaft, three chambers per rotor.
    Half-sine port areas on the 2-stroke; the second bank's pipe, 7 % longer: with two identical pipes the banks would sum back to one even train and the half
    orders of a cross-plane V8 would cancel, where in the real car the left and right paths always differ."""
    rng = _rng(seed, 75)
    n = _n(dur)
    if units not in ("ms", "m"):
        raise ValueError("pipe_engine units: ms or m")
    if view not in ("tail", "cabin"):
        raise ValueError("pipe_engine view: tail or cabin")
    if units == "m":
        ms = lambda L: None if L is None else (np.asarray(L, float) if np.ndim(L) else float(L)) / gas_speed(gas_temp) * 1e3
        header, pipe, headers, muffler = ms(header), ms(pipe), ms(headers), ms(muffler)
        extractor, cylinder, outlet, expansion_ms = ms(extractor), ms(cylinder), ms(outlet), ms(expansion_ms)
    key = str(cycle)
    if key not in _CYCLES:
        raise ValueError(f"pipe_engine cycle {cycle!r}: use 4, 2 or rotary")
    mode, per = _CYCLES[key]
    if offsets is not None:
        offs = (-np.asarray(offsets, float)) % 1.0           # slot k fires at offsets[k] of the cycle
        C = len(offs)
        if not 1 <= C <= 16:
            raise ValueError("pipe_engine offsets: 1..16 cylinders")
    else:
        C = 3 * int(np.clip(cylinders, 1, 5)) if mode == 2 else int(np.clip(cylinders, 1, 16))
        offs = (np.arange(C) / C - (np.arange(C) % 2) * 0.5 * float(np.clip(lope, 0, 1)) / C) % 1.0
    ramp = _along(np.clip(rpm, 30.0, 20000.0), n) if np.ndim(rpm) else _ramp(float(np.clip(rpm, 200.0, 20000.0)), rpm_end, n)
    inc = ramp / (per * SR) * (1.0 + 0.01 * _lfn(rng, n, 57.0))
    if isinstance(load, str):
        if load != "auto":
            raise ValueError("pipe_engine load: a number, a curve or auto")
        ld = auto_load(ramp)
    else:
        ld = _along(np.clip(load, 0.0, 1.0), n) if np.ndim(load) else float(np.clip(load, 0.0, 1.0))
    if headers is not None:
        if len(headers) != C:
            raise ValueError(f"pipe_engine headers: one length per cylinder ({C})")
        n_ex = np.maximum((np.clip(np.asarray(headers, float), 0.3, 12.0) * 1e-3 * SR).astype(np.int64), 4)
    else:
        spread = 1.0 + float(np.clip(unequal, 0, 0.5)) * np.linspace(-1.0, 1.0, C) if C > 1 else np.ones(1)
        n_ex = np.maximum((float(np.clip(header, 0.3, 12.0)) * 1e-3 * SR * spread).astype(np.int64), 4)
    if extractor and extractor > 0:
        n_ex = n_ex + int(float(np.clip(extractor, 0.0, 12.0)) * 1e-3 * SR)
    n_muf = np.array([7, 9, 10, 12], dtype=np.int64)        # 0.146 / 0.1875 / 0.208 / 0.25 ms
    if muffler is not None:                                  # [10, 15, 20, 25] samples at 44.1 kHz is .23 .34 .45 .57 ms
        if len(muffler) != 4:
            raise ValueError("pipe_engine muffler: four lengths in ms")
        n_muf = np.maximum((np.clip(np.asarray(muffler, float), 0.05, 5.0) * 1e-3 * SR).astype(np.int64), 2)
    noise = 0.181 * ld * _lp(rng.uniform(-1, 1, n), 10900.0)
    n_pipe = max(int(float(np.clip(pipe, 0.5, 40.0)) * 1e-3 * SR), 4)
    if banks is None:
        bank, pipes = np.zeros(C, dtype=np.int64), np.array([n_pipe], dtype=np.int64)
    else:
        bank = np.array([{"L": 0, "R": 1}[b] if isinstance(b, str) else int(b) for b in banks], dtype=np.int64)
        if len(bank) != C or not set(bank.tolist()) == {0, 1}:
            raise ValueError(f"pipe_engine banks: one of two banks per cylinder ({C}), both in use")
        pipes = np.array([n_pipe, int(n_pipe * 1.07) + 1], dtype=np.int64)
    ign_t = np.broadcast_to(0.069 - 0.053 * ld, (n,))
    pop = np.zeros(n)
    prob = float(np.clip(backfire, 0.0, 0.99))
    falling = np.gradient(ramp) < 0 if np.ndim(rpm) else None
    if prob > 0 and (falling is not None or 0 <= rpm_end < rpm):
        cyc = np.cumsum(inc)
        for start in np.searchsorted(cyc, np.arange(1, int(cyc[-1]) + 1)):
            if falling is not None and not falling[min(start, n - 1)]:
                continue
            if rng.random() < prob:
                m = max(int(0.5 * ign_t[min(start, n - 1)] / inc[min(start, n - 1)]), 4)
                seg = 2.5 * np.sin(np.pi * np.arange(m) / m)
                pop[start:start + m] += seg[:max(n - start, 0)]
                prob *= prob
    if pops is not None:
        pop = pop + _fit(pops, n)
    igf = (2.0 + 6.0 * ld) * (1.0 if fire is None else _along(fire, n))
    if chamber is None and expansion and expansion > 1:
        r1 = (1.0 - float(expansion)) / (1.0 + float(expansion))
        chamber = ([4, max(int(float(np.clip(expansion_ms, 0.05, 20.0)) * 1e-3 * SR), 2), 4], [r1, -r1], [1.0, 1.0, 1.0])
    if chamber is None:
        ch_len, ch_r, ch_g = np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0)
    else:
        ch_len, ch_r, ch_g = np.asarray(chamber[0], dtype=np.int64), np.asarray(chamber[1], float), np.asarray(chamber[2], float)
    r2 = _rng(seed, 79)
    dj = np.zeros((C, 1))
    if jitter > 0:                                           # per cylinder: lp75(0.25 (rand - 0.5)), mean removed
        dj = np.array([_lp(0.25 * r2.uniform(-0.5, 0.5, n), 75.0) for _ in range(C)]) * float(np.clip(jitter, 0, 1))
        dj -= dj.mean(axis=0)
    if crank > 0:
        dj = dj + CRANK_FLUTTER * float(np.clip(crank, 0, 1)) * _lp(_rng(seed, 81).uniform(-1, 1, n), 1.0 / 0.01747)
    t_in, t_ex = (0.0, 0.0) if timing is None else _pair(timing)
    n_cyl = int(float(np.clip(cylinder, 0.0, 5.0)) * 1e-3 * SR) if cylinder else 0
    n_out = int(float(np.clip(outlet, 0.0, 40.0)) * 1e-3 * SR) if outlet else 0
    o = _pipe_engine_k(np.ascontiguousarray(inc), offs, np.ascontiguousarray(noise), pop, n_ex, 9, pipes, n_muf, 2.43,
                       np.array(np.broadcast_to(igf, (n,))), np.array(ign_t), bank, mode, ch_len, ch_r, ch_g,
                       np.ascontiguousarray(dj), float(t_in), float(t_ex), n_cyl, n_out, float(np.clip(outlet_r, -0.99, 0.99)),
                       *PIPE_ENDS[ends])
    m = float(np.clip(mic, 0.0, 1.0))
    ex = np.gradient(o[1]) * SR / (2 * np.pi * 500.0) if radiate else o[1]
    ex = ex * float(outlet_gain) * (4.0 / C if norm4 and C > 4 else 1.0)
    o = (o[0] * float(intake_gain),) + tuple(o[1:])
    if side is not None:
        for f in _floats(side):
            ex = ex - _bp(ex, float(np.clip(f, 20.0, 0.45 * SR)), 5.0)
    if warp > 0:
        ex = warp_delay(ex, np.cumsum(inc) * C, float(np.clip(warp, 0, 1)))
    y = (1.0 - m) * 0.639 * ex + m * 0.325 * o[0] + 0.036 * float(np.clip(block, 0.0, 20.0)) * _lp(o[2], 92.0)
    if alternator > 0 or roots > 0:
        y = y + accessory_whine(ramp, ld, float(np.clip(alternator, 0, 1)), float(np.clip(roots, 0, 1))) * math.sqrt(float(np.mean(y * y)))
    if transients > 0:
        y = y + firing_transients(inc, C, _rng(seed, 83), float(np.clip(transients, 0, 1))) * math.sqrt(float(np.mean(y * y)))
    y = _hp(soft_limit(y), 25.0)
    if view == "cabin":
        y = car_cabin(y, SR)
    return _fin(y, sr, 0.6, 0.01, 0.03)


CRANK_FLUTTER = 0.331
ENGINE_DYN = dict(rpm=(0.0, 1000.0, 2000.0, 3500.0, 7500.0, 8750.0, 10000.0),
                  power=(0.0, 35000.0, 125000.0, 325000.0, 735000.0, 735000.0, 475000.0),
                  brake=100.0, inertia=1.0, friction=0.01, idle=750.0, limit=7500.0, idle_throttle=0.35, starter=200.0,
                  catch=100.0)


def engine_rpm(throttle, rate=1000.0, rpm0=0.0):
    """Crank speed from a throttle curve (one value per 1 / rate s): power P(rpm) a spline through ENGINE_DYN, pedal
    p = throttle^(0.5 + 1.5 (rpm - idle) / (limit - idle)), torque p (P / w + Tb) - Tb, I dw/dt = torque - friction w;
    the starter adds 200 rpm/s below 100 rpm, the idle holds the throttle at 0.35 at least, the limiter cuts at 7500.
    Feed the result to ``pipe_engine(rpm=...)``."""
    from scipy.interpolate import CubicSpline
    d = ENGINE_DYN
    pw = CubicSpline(d["rpm"], d["power"])
    thr = np.clip(np.atleast_1d(np.asarray(throttle, float)), 0.0, 1.0)
    dt = 1.0 / float(rate)
    w = float(rpm0) * 2 * math.pi / 60.0
    out = np.empty(len(thr))
    for i, th in enumerate(thr):
        rpm = w * 60.0 / (2 * math.pi)
        if rpm < d["catch"]:
            rpm += d["starter"] * dt
            w = rpm * 2 * math.pi / 60.0
        else:
            th = max(th, d["idle_throttle"]) if rpm < d["idle"] else th
            th = 0.0 if rpm > d["limit"] else th
            ex = 0.5 + 1.5 * min(max((rpm - d["idle"]) / (d["limit"] - d["idle"]), 0.0), 1.0)
            tq = th ** ex * (float(pw(rpm)) / max(w, 1.0) + d["brake"]) - d["brake"]
            w = max(w + dt * (tq - d["friction"] * w) / d["inertia"], 0.0)
        out[i] = w * 60.0 / (2 * math.pi)
    return out
PIPE_ENDS = {"engine": (-0.7576, 0.06), "paper": (-0.5, 0.1)}


def _floats(v):
    return [float(x) for x in (v.replace(";", ",").split(",") if isinstance(v, str) else np.atleast_1d(v)) if str(x).strip()]


def _pair(v):
    a = _floats(v)
    if len(a) != 2:
        raise ValueError("pipe_engine timing: two shifts, intake and exhaust")
    return float(np.clip(a[0], -0.5, 0.5)), float(np.clip(a[1], -0.5, 0.5))


def gas_speed(temp_k):
    """Speed of sound in air at ``temp_k`` K: 331.45 sqrt(T / 273.15) m/s."""
    return 331.45 * math.sqrt(max(float(temp_k), 50.0) / 273.15)


def auto_load(rpm):
    """Load from the command's rate of change: 0.3 at steady rpm, full at 3000 rpm/s, smoothed over 50 ms."""
    d = np.abs(np.gradient(np.asarray(rpm, float))) * SR
    return np.clip(0.3 + 0.7 * uniform_filter1d(d, max(int(0.05 * SR), 1)) / 3000.0, 0.0, 1.0)


def warp_delay(x, phase, amount, d0=1e-3):
    """Two reads of ``x`` d0 (1 +- 0.8 amount sin 2 pi phase) back, averaged: delays stretched in antiphase by the firing."""
    t = np.arange(len(x), dtype=float)
    m = 0.8 * amount * np.sin(2 * np.pi * phase)
    d = d0 * SR
    return 0.5 * (np.interp(t - d * (1 + m), t, x, left=0.0) + np.interp(t - d * (1 - m), t, x, left=0.0))


ALTERNATOR = (2.8, 36)        # pulley ratio, ripples per alternator turn (12 poles x 3 phases)
ROOTS = (2.0, 6)              # drive ratio, discharges per rotor turn (two 3-lobe rotors)


def accessory_whine(rpm, load, alternator=0.0, roots=0.0):
    """Whines of the belt-driven accessories at the crank speed ``rpm`` (a curve): the alternator at rpm / 60 x 2.8 x 36,
    the Roots blower at rpm / 60 x 2 x 6, each with its 2nd and 3rd harmonics, louder under load."""
    rpm = np.asarray(rpm, float)
    out = np.zeros(len(rpm))
    for g, (ratio, per) in ((alternator, ALTERNATOR), (roots, ROOTS)):
        if g <= 0:
            continue
        f = rpm / 60.0 * ratio * per
        ph = 2 * np.pi * np.cumsum(f) / SR
        w = sum(a * np.sin(h * ph) * (h * f < 0.45 * SR) for h, a in ((1, 1.0), (2, 0.4), (3, 0.2)))
        out += g * (0.5 + 0.5 * np.asarray(load)) * w
    return 0.3 * out


CABIN = (150.0, 85.0, 4.0, 0.8)   # Hz panel low-pass, Hz boom, its Q, its gain


def car_cabin(x, sr=SR):
    """A car body between the source and the driver: panels pass the lows (mass law, -6 dB/oct above 150 Hz)
    and the cabin's first air mode booms at 85 Hz."""
    lp, f, q, g = CABIN
    return F.lowpass(x, lp, sr) + g * F.bandpass(x, f, sr, q)


@effect("car_cabin", "Heard from inside a car: body panels low-pass the outside sound and the cabin booms at 85 Hz.")
def car_cabin_fx(x, sr=DEFAULT_SR):
    return _per_channel(x, lambda c: car_cabin(c, sr))


def expansion_chamber_ir(ratio, k=5):
    """First ``k`` reflections of a closed-end expansion chamber of area ratio m = S2 / S1 seen from the pipe:
    R = (1 - m) / (1 + m), then -(1 + R)(1 - R) R^j (m = 2: -1/3, -8/9, 8/27, -8/81, 8/243)."""
    m = float(ratio)
    r = (1.0 - m) / (1.0 + m)
    return np.array([r] + [-(1.0 + r) * (1.0 - r) * r ** j for j in range(k - 1)])


def plane_wave_cutoff(radius, c=343.0):
    """Frequency (Hz) of the first non-plane mode in a round pipe: k a = 1.8412 (25 mm radius: 4020 Hz)."""
    return 1.8412 * c / (2 * np.pi * float(radius))


def bend_cents(radius, bend_radius, f=0.0, c=343.0):
    """Pitch shift (cents) of a wave through a torus bend of tube radius a and centre-line radius R0, against a straight
    pipe: rho_eff / rho = x^2 / (2 (1 - sqrt(1 - x^2))), x = a / R0, c_eff = c sqrt(rho / rho_eff); fading to 0 at half the
    plane-wave cutoff (a 25, R0 75 mm: +25 cents at low frequency, 0 near 2 kHz)."""
    x = float(radius) / float(bend_radius)
    if not 0 < x < 1:
        return 0.0
    lf = 600.0 * math.log2(2.0 * (1.0 - math.sqrt(1.0 - x * x)) / (x * x))
    return lf * max(0.0, 1.0 - (float(f) / (0.5 * plane_wave_cutoff(radius, c))) ** 2)


def step_end_correction(alpha):
    """End correction (in radii of the narrow pipe) of an area step of radius ratio alpha = a / b <= 0.55:
    0.26164 - 0.353 alpha + 0.0809 alpha^3 (an added mass of air in series)."""
    a = float(np.clip(alpha, 0.0, 0.55))
    return 0.26164 - 0.353 * a + 0.0809 * a ** 3


def side_branch_freq(length, wall=0.0, radius=0.0, c=343.0):
    """Frequency a closed side branch cancels on the main pipe: c / (4 (h + t_w + 0.85 a)) (h 0.1 m: 857 Hz)."""
    return c / (4.0 * (float(length) + float(wall) + 0.85 * float(radius)))


def helmholtz_neck(f0, volume, area, c=343.0):
    """Neck length (m) of a side Helmholtz resonator tuned to ``f0`` with a neck area 0.6 x ``area``: L = A_n c^2 / (V w0^2)."""
    an = 0.6 * float(area)
    return an * c * c / (float(volume) * (2 * np.pi * float(f0)) ** 2)


def twin_power_gain(kd, phase=0.0):
    """Power of two equal pipe outlets d apart, against one: 2 + 2 (sin kd / kd) cos phase (4 = +6 dB in phase and close)."""
    s = np.sinc(np.asarray(kd, float) / np.pi)
    return 2.0 + 2.0 * s * np.cos(phase)


def polack_ir(D, n, sr=SR):
    """Diffusive impulse response of a closed silencer: h(t) = D / (sqrt(pi) t^1.5) exp(-D^2 / t), peak 0.2313 / D^2 at
    t = 2 D^2 / 3 (t and D^2 in s), unit sum."""
    t = (np.arange(int(n)) + 0.5) / sr
    h = float(D) / (math.sqrt(math.pi) * t ** 1.5) * np.exp(-float(D) ** 2 / t)
    return h / (h.sum() + 1e-30)


STEEPEN = dict(gamma=1.36, p0=101325.0, lo=0.62, hi=1.3)


def steepening(p, p0=STEEPEN["p0"], gamma=STEEPEN["gamma"]):
    """Nonlinear steepening of a pressure pulse: its local travel time scales by k = clip(1 - nl p, 0.62, 1.3),
    nl = (gamma + 1) / 2 / (2 gamma p0)."""
    nl = 0.5 * (gamma + 1.0) / (2.0 * gamma * p0)
    return np.clip(1.0 - nl * np.asarray(p, float), STEEPEN["lo"], STEEPEN["hi"])


def soft_limit(s, max_amp=20.0):
    """Identity up to max_amp, then sign(s)(max_amp + 1 - 1/(|s| - max_amp + 1)): continuous, slope 1 at the knee,
    never past max_amp + 1."""
    s = np.asarray(s, float)
    a = np.abs(s)
    return np.where(a > max_amp, np.sign(s) * (max_amp + 1.0 - 1.0 / (a - max_amp + 1.0)), s)


# Power curve (W at rpm), 100 N m engine brake, inertia 1, friction 0.01.
_POWER_RPM = (0.0, 1000.0, 2000.0, 3500.0, 7500.0, 8750.0, 10000.0)
_POWER_W = (0.0, 35000.0, 125000.0, 325000.0, 735000.0, 735000.0, 475000.0)


def rpm_curve(throttle, dur, rpm0=750.0, min_rpm=750.0, max_rpm=7500.0, start=False, inertia=1.0, friction=0.01, brake=100.0, sr=SR):
    """Engine speed for a throttle (a number or a curve of 0..1 over ``dur`` seconds), ``sr`` values per second.
    Integrates torque = p (P(rpm) / omega + brake) - brake - friction, p = throttle^(0.5 + 1.5 (rpm - min) / (max - min)),
    with idle throttle 0.35 below ``min_rpm``, a fuel cut above ``max_rpm`` and, with ``start``, a starter adding
    200 rpm/s until 100 rpm, at 1 kHz. The inertia is what low-passes the pedal."""
    from scipy.interpolate import CubicSpline
    power = CubicSpline(_POWER_RPM, _POWER_W)
    m = max(int(float(dur) * 1000.0), 2)
    thr = _along(np.clip(throttle, 0.0, 1.0), m)
    r = np.zeros(m)
    rpm, starting = (0.0, True) if start else (float(rpm0), False)
    for i in range(m):
        th = float(thr[i])
        if rpm < 100.0 and not starting:
            th = 0.0
            if rpm < 75.0:
                rpm = 0.0
                r[i:] = 0.0
                break
        elif rpm < min_rpm:
            th = 0.35
        if starting:
            if rpm < 100.0:
                rpm += 200.0 * 1e-3
                th = 1.0
            else:
                starting = False
        if rpm > max_rpm:
            th = 0.0
        p = th ** max(0.5 + 1.5 * (rpm - min_rpm) / (max_rpm - min_rpm), 0.0) if th > 0 else 0.0
        omega = max(rpm / 60.0 * 2.0 * math.pi, 1e-3)
        torque = p * (float(power(rpm)) / omega + brake) - brake
        rpm = max((omega * inertia + (torque - friction) * 1e-3) / inertia * 60.0 / (2.0 * math.pi), 0.0)
        r[i] = rpm
    return _along(r, max(int(round(float(dur) * sr)), 2))


@sfx("turbo", "Turbocharger: the compressor whine rising as it spools, intake hiss, and the blow-off valve when the throttle shuts.",
     dur=(2.5, "s"), spool=(90000.0, "peak shaft rpm, 30000..300000 (the pitch of the whine; 300000 = the air turbine of `dental_drill`, 5 kHz)"), boost=(0.7, "0..1 how hard it is pushed: hiss and blow-off level"),
     blowoff=(1, "1 = end on the blow-off valve's pshh"), flutter=(0.0, "0..1 compressor surge chatter in the blow-off"), seed=(0, "variation"))
def turbo(sr=DEFAULT_SR, dur=2.5, spool=90000.0, boost=0.7, blowoff=1, flutter=0.0, seed=0):
    """Whine a^2 (0.22 sin p + 0.1 sin 2p + 0.05 a sin 7p) (0.4 + 6 mdot) with a = min(1, w / 9000 rad/s); hiss band-passed
    at 3000 + 0.35 w Hz; blow-off band-passed at 1400 + 5500 boost Hz, Q 0.7. Spool time constant 0.22 of the sound,
    mass flow 0.1 at full boost, surge flutter 38 Hz."""
    rng = _rng(seed, 76)
    n = _n(dur)
    b = float(np.clip(boost, 0.0, 1.0))
    t = np.arange(n) / SR
    rel = 0.72 * n / SR if blowoff else 10.0 * n / SR
    up = 1.0 - np.exp(-t / (0.22 * n / SR))
    s = float(np.clip(spool, 3e4, 3e5))
    w = s * 2 * np.pi / 60.0 * np.where(t < rel, up, up * np.exp(-(t - rel) / 0.5))
    mdot = 0.1 * b * np.where(t < rel, up, 0.0)
    a = np.minimum(1.0, w / 9000.0)
    ph = np.cumsum(w) / SR
    y = a * a * (0.22 * np.sin(ph) + 0.1 * np.sin(2 * ph) + 0.05 * a * np.sin(7 * ph) * (7.0 * w < np.pi * SR * 0.9)) * (0.4 + 6.0 * mdot)    # sin 7p would fold above 0.45 SR
    y = y + 2.0 * _tvbp(rng.standard_normal(n), 3000.0 + 0.35 * w, 1.5) * mdot * a
    if blowoff:
        k = int(rel * SR)
        m = np.arange(n - k)
        env = np.exp(-m / (0.18 * SR)) * (1.0 - np.exp(-m / (0.004 * SR)))
        env = env * (1.0 + 0.9 * float(np.clip(flutter, 0, 1)) * np.sign(np.sin(2 * np.pi * 38.0 * m / SR)))
        y[k:] += 1.2 * b * _bp(rng.standard_normal(n - k), 1400.0 + 5500.0 * b, 0.7) * env
    return _fin(_lp(y, max(5500.0, 1.2 * s / 60.0), 2), sr, 0.5, 0.01, 0.02)    # 5.5 kHz would take the line away from 275000 rpm


TYRE_A, TYRE_B, TYRE_LOAD = 0.07, 0.05, 4000.0              # patch half-length and half-width m, wheel N: a road car, 2.9 bar over the patch
TYRE_ESTAR, TYRE_KG = 1.0e7, 0.5                            # Pa, kg of the singing tread mode
TYRE_CREEP = 0.1                                            # creep per unit of `slip`: sets saturation (mu P / (8ab/3 1.1 E*)) near slip 0.4
TYRE_MU = (1.0, 0.7)                                        # rubber on dry asphalt: peak, sliding


def tyre_brush(s, f, n, rng, speed=15.0):
    """One wheel through the brush model (`rail._brush`, as `rail.curve_squeal`): creep saturates the patch, falling friction drives a tread
    mode at ``f``. Returns its velocity. Friction falls from TYRE_MU[0] to [1] over 1 m/s, zeta .005, the creep flutters 2 %."""
    from .rail import _brush
    creep = np.ascontiguousarray(TYRE_CREEP * s * (1.0 + 0.02 * _lfn(rng, n, 300.0)))
    w = np.array([2.0 * math.pi * f])
    return _brush(creep, speed / SR, TYRE_A, TYRE_B, 1.1 * TYRE_ESTAR / TYRE_A, TYRE_LOAD, TYRE_MU[0], TYRE_MU[1], 1.0, speed,
                  w, np.array([0.005]), np.array([1.0 / TYRE_KG]), 1)[1]


TYRE_CORNERING = 60e3                                       # N / rad: cornering stiffness of a road car's tyre at TYRE_LOAD


def tyre_lateral_force(alpha, load=TYRE_LOAD, mu=TYRE_MU[0], stiffness=TYRE_CORNERING):
    """Steady side force (N) of the brush tyre with a parabolic pressure, at slip angle ``alpha`` (rad): with
    t = C tan(alpha) / (3 mu F), F_y = 3 mu F t (1 - t + t^2 / 3) up to full sliding at t = 1, then mu F."""
    a = np.asarray(alpha, float)
    t = np.minimum(np.abs(float(stiffness) * np.tan(a)) / (3.0 * float(mu) * float(load)), 1.0)
    return np.sign(a) * 3.0 * float(mu) * float(load) * t * (1.0 - t + t * t / 3.0)


@sfx("tyre_squeal", "Tyres scrubbing past their grip limit (a drift, a hard brake, a getaway).",
     dur=(1.5, "s"), slip=(0.8, "0.2 chirp .. 1.5 full slide: pitch and harshness"), wobble=(0.5, "0..1 unsteadiness of the slide"),
     model=("band", "band = noise through a moving band; brush = the contact's brush model: a tyre that grips (slip 0.05) is 30 times quieter than one that slides"), seed=(0, "variation"))
def tyre_squeal(sr=DEFAULT_SR, dur=1.5, slip=0.8, wobble=0.5, model="band", seed=0):
    """Per wheel, noise through a band-pass at 900 + 700 s + 220 i Hz with Q 6 x (2.2 + 1.5 env).
    ``model="brush"``: the same frequencies as tread modes, on `tyre_brush`."""
    if model not in ("band", "brush"):
        raise ValueError(f"tyre_squeal model {model!r}: use band or brush")
    rng = _rng(seed, 77)
    n = _n(dur)
    y = np.zeros(n)
    s0 = float(np.clip(slip, 0.05, 1.5))
    if model == "brush":
        for i in range(4):
            f = 900.0 + 700.0 * s0 + 220.0 * i
            s = np.clip(s0 * (1.0 + 0.35 * float(np.clip(wobble, 0, 1)) * _lfn(rng, n, 5.0)), 0.02, 1.5)
            y += _bp(tyre_brush(s, f, n, rng), f, 4.0)
        ref = 0.25 * TYRE_CREEP * 15.0                      # m/s: a wheel gripping, far under this, stays quiet instead of being scaled up
        return _fin(y, sr, 0.5, 0.02, 0.08) * min(float(np.max(np.abs(y))) / ref, 1.0)
    for i in range(4):
        s = np.clip(s0 * (1.0 + 0.35 * float(np.clip(wobble, 0, 1)) * _lfn(rng, n, 5.0)), 0.05, 1.5)
        env = np.clip(s / s0, 0.0, 1.5) / 1.5
        y += _tvbp(rng.standard_normal(n), 900.0 + 700.0 * s + 220.0 * i, 6.0 * (2.2 + 1.5 * env)) * env
    return _fin(y, sr, 0.5, 0.02, 0.08)


# Harmonoise source model, category 1 (cars), third-octave bands 20 Hz .. 12.5 kHz; the table runs 25 Hz .. 10 kHz,
# its end values are held at 20 Hz and 12.5 kHz.
# Rolling L = aR + bR log10(v / 70 km/h); propulsion L = aP + bP (v - 70) / 70.
_HN_F = 20.0 * 2.0 ** (np.arange(29) / 3.0)
_HN_AR = np.array([69.9, 69.9, 69.9, 69.9, 74.9, 74.9, 74.9, 79.3, 82.5, 81.3, 80.9, 78.9, 78.8, 80.5, 85.7, 87.7, 89.2, 90.6,
                   89.9, 89.4, 87.6, 85.6, 82.5, 79.6, 76.8, 74.5, 71.9, 69.0, 69.0])
_HN_BR = np.array([33.0, 33.0, 33.0, 33.0, 30.0, 30.0, 30.0, 41.0, 41.2, 42.3, 41.8, 38.6, 35.5, 31.7, 21.5, 21.2, 23.5, 29.1,
                   33.5, 34.1, 35.1, 36.4, 37.4, 38.9, 39.7, 39.7, 39.7, 39.7, 39.7])
_HN_AP = np.array([85.8, 85.8, 87.6, 87.5, 87.5, 96.6, 97.2, 91.5, 86.7, 86.8, 84.9, 86.0, 86.0, 85.9, 80.6, 80.2, 77.8, 78.0,
                   81.4, 82.3, 82.6, 81.5, 80.7, 78.8, 77.0, 76.0, 74.0, 72.0, 72.0])
_HN_BP = np.where(_HN_F < 225.0, 0.0, 8.2)


def road_noise_levels(kmh: float) -> tuple[np.ndarray, np.ndarray]:
    """(rolling, propulsion) sound power levels in dB per third-octave band of ``_HN_F``."""
    v = max(float(kmh), 1.0)
    return _HN_AR + _HN_BR * np.log10(v / 70.0), _HN_AP + _HN_BP * (v - 70.0) / 70.0


def _third_octave_noise(n: int, rng, levels_db: np.ndarray, freqs=None) -> np.ndarray:
    """Noise with the given third-octave band levels (at ``freqs``, default the Harmonoise bands 20 Hz .. 12.5 kHz):
    a band of constant level has a power density falling as 1 / f, so the spectrum is shaped by the interpolated
    level minus 10 log10 f."""
    f = np.maximum(np.fft.rfftfreq(n, 1.0 / SR), 1.0)
    bands = _HN_F if freqs is None else np.asarray(freqs, float)
    g = np.interp(np.log(f), np.log(bands), levels_db, left=levels_db[0] - 30.0, right=levels_db[-1] - 30.0)
    return np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * 10 ** ((g - 10 * np.log10(f)) / 20.0), n)


@sfx("pass_by", "A car driving past: tyre roar that grows with speed, engine or electric drive, wind, the Doppler drop and the ground reflection.",
     dur=(5.0, "s"), speed=(60.0, "km/h, 10..250"), distance=(6.0, "m from the lane at the closest point"),
     power=("combustion", "combustion | electric | none (tyres and wind only)"), cylinders=(4, "combustion: cylinder count"),
     gear=(3.2, "combustion: engine rpm = gear x 26 x km/h (2.2 top gear .. 6 low gear)"),
     wet=(0.0, "0 dry .. 1 wet road: spray hiss above 2 kHz"), direction=("lr", "lr | rl"),
     pan=(1, "1 = stereo, panned across; 0 = mono (when the game positions it itself)"),
     sources=(1, "1 = one source at 0.3 m | 2 = rolling noise mostly at the tyre contact (0.01 m), engine mostly at 0.3 m, each with its own ground path"),
     seed=(0, "variation"))
def pass_by(sr=DEFAULT_SR, dur=5.0, speed=60.0, distance=6.0, power="combustion", cylinders=4, gear=3.2, wet=0.0,
            direction="lr", pan=1, seed=0, sources=1):
    """Source = Harmonoise rolling noise + a propulsion source scaled to the Harmonoise propulsion
    energy + aerodynamic noise (pink noise low-passed at 250 + 30 v Hz, level (v / 40)^2.5, v in m/s, faded in by a
    smoothstep from 15 to 26 m/s). Propagation = the ``doppler`` effect with its ground path.
    Wet road: spray above 2 kHz. Electric drive: a 9:1 reduction gear, 0.33 m wheels and the ripple tone of a
    4-pole-pair motor on a six-step inverter, 12 dB under a combustion engine."""
    from .reverbs import doppler
    if power not in ("combustion", "electric", "none"):
        raise ValueError(f"pass_by power {power!r}: use combustion, electric or none")
    rng = _rng(seed, 78)
    kmh = float(np.clip(speed, 10.0, 250.0))
    v = kmh / 3.6
    n = _n(dur)
    lr, lp = road_noise_levels(kmh)
    y = _unit(_third_octave_noise(n, rng, lr))
    if wet > 0:
        y = y + float(np.clip(wet, 0, 1)) * 0.8 * _unit(_hp(rng.standard_normal(n), 2000.0, 2))
    if power == "combustion":
        eng = pipe_engine(sr=SR, dur=n / SR, rpm=float(np.clip(kmh * 26.0 * float(gear), 700.0, 9000.0)), cylinders=cylinders, seed=seed)
    elif power == "electric":
        mrpm = v / (2 * math.pi * 0.33) * 60.0 * 9.0
        eng = gearbox(sr=SR, dur=n / SR, rpm=mrpm, teeth=17, teeth2=153, load=0.8, wear=0.05, eccentricity=0.15,
                      backlash=0.0, housing=2600.0, seed=seed)
        eng = eng + 0.3 * np.sin(2 * np.pi * min(mrpm / 60.0 * 24.0, 9000.0) * np.arange(len(eng)) / SR)
    roll_, prop = y, np.zeros(n)
    if power != "none":
        ratio = float(np.sum(10 ** (lp / 10)) / np.sum(10 ** (lr / 10)))
        prop = math.sqrt(ratio) * (0.25 if power == "electric" else 1.0) * _unit(_fit(eng, n))
    t = min(max((v - 15.0) / 11.0, 0.0), 1.0)
    wind = 1.5 * t * t * (3.0 - 2.0 * t) * (v / 40.0) ** 2.5 * _unit(_lp(O.pink(n, int(seed) + 78), 250.0 + 30.0 * v))
    pb = lambda x, h: doppler(x, SR, speed=v, closest=float(np.clip(distance, 1.0, 200.0)), direction=direction,
                              pan=bool(pan), surface="asphalt", horn="low", height=h)
    if int(sources) == 2:
        s5 = math.sqrt(5.0)
        hi = (2 * prop + roll_) / s5 + wind
        lo = (prop + 2 * roll_) / s5
        g = 0.9 / (float(np.max(np.abs(hi + lo))) + 1e-12)
        y = pb(g * hi, 0.3 + 1.5) + pb(g * lo, 0.01 + 1.5)
    else:
        y = pb(_fin(roll_ + prop + wind, SR, 0.9, 0.3, 0.3), 1.8)    # ground effect of asphalt, not a flat inverted -0.35
    if int(sr) != SR:
        g = math.gcd(int(sr), SR)
        y = resample_poly(y, int(sr) // g, SR // g, axis=0)
    return y * (0.7 / (float(np.max(np.abs(y))) + 1e-12))


@sfx("traffic", "A road beside you: cars passing at random in both directions, as a Poisson flow with a 2 s minimum gap.",
     dur=(20.0, "s"), rate=(600.0, "vehicles per hour"), speed=(70.0, "km/h speed limit: each car drives 0.85-1.1 of it"),
     distance=(8.0, "m to the near lane (the far lane is 3.5 m further)"), electric=(0.2, "share of electric cars"),
     seed=(0, "variation"))
def traffic(sr=DEFAULT_SR, dur=20.0, rate=600.0, speed=70.0, distance=8.0, electric=0.2, seed=0):
    """Arrivals are exponential at ``rate`` / 3600 per second; a car closer than 2 s behind the one before in its lane
    is dropped. Each is a ``pass_by`` of 6 s centred on its arrival, in stereo."""
    rng = np.random.default_rng(int(seed) + 80)
    n = _n(dur)
    y = np.zeros((n, 2))
    lam = float(np.clip(rate, 10.0, 4000.0)) / 3600.0
    t, last = -3.0, {0: -9.0, 1: -9.0}
    k = 0
    while True:
        t += rng.exponential(1.0 / lam)
        if t > dur + 3.0:
            break
        lane = int(rng.random() < 0.5)
        if t - last[lane] < 2.0:
            continue
        last[lane] = t
        v = float(speed) * rng.uniform(0.85, 1.1)
        car = pass_by(sr=SR, dur=6.0, speed=v, distance=float(distance) + 3.5 * lane, power="electric" if rng.random() < electric else "combustion",
                      gear=rng.uniform(2.2, 3.4), direction="lr" if lane == 0 else "rl", seed=int(seed) * 1000 + k)
        k += 1
        s = int((t - 3.0) * SR)
        a, b = max(s, 0), min(s + len(car), n)
        if b > a:
            y[a:b] += car[a - s:b - s] * 10 ** (-3.0 * lane / 20.0)
    if int(sr) != SR:
        g = math.gcd(int(sr), SR)
        y = resample_poly(y, int(sr) // g, SR // g, axis=0)
    return y * (0.7 / (float(np.max(np.abs(y))) + 1e-12))


@sfx("induction_motor", "Induction motor on the mains (pump, fan, lathe): magnetic hum at twice the mains, the rotor-slot whine, shaft orders; "
     "faults: broken rotor bar, eccentric rotor, worn bearing.",
     dur=(3.0, "s"), mains=(50.0, "Hz supply"), poles=(4, "2 | 4 | 6 | 8"), load=(0.6, "0 idle .. 1 rated: the slip, 0.5 to 5 %"),
     bars=(28, "rotor bars (slot whine at bars x shaft rate)"), fault=("none", "none | broken_bar | eccentric | bearing"),
     severity=(0.5, "0..1 how bad the fault is"), seed=(0, "variation"))
def induction_motor(sr=DEFAULT_SR, dur=3.0, mains=50.0, poles=4, load=0.6, bars=28, fault="none", severity=0.5, seed=0):
    """Slip s = 0.005 + 0.045 load, shaft fr = 2 f (1 - s) / p. Lines: 2 f magnetostriction and its 4 f; slot passing
    bars fr and its +-2 f sidebands; shaft fr, 2 fr. Broken bar: sidebands f (1 +- 2 s) on the supply line and the hum beating
    at 2 s f. Eccentric: slot line +- fr. Bearing: an outer-race ticking at 0.4 balls fr through ``bearing``."""
    if fault not in ("none", "broken_bar", "eccentric", "bearing"):
        raise ValueError("induction_motor fault: none, broken_bar, eccentric or bearing")
    rng = _rng(seed, 81)
    n = _n(dur)
    t = np.arange(n) / SR
    f = float(np.clip(mains, 20.0, 400.0))
    p = int(np.clip(poles, 2, 12)) // 2 * 2
    s = 0.005 + 0.045 * float(np.clip(load, 0, 1))
    fr = 2.0 * f * (1.0 - s) / p
    sv = float(np.clip(severity, 0, 1))
    tone = lambda fq, a: a * np.sin(2 * np.pi * fq * t + rng.uniform(0, 6.28)) * (fq < 0.45 * SR)
    fs = int(np.clip(bars, 8, 80)) * fr
    y = tone(2 * f, 1.0) + tone(4 * f, 0.3) + tone(fs, 0.35) + tone(fs - 2 * f, 0.15) + tone(fs + 2 * f, 0.15)
    y += tone(fr, 0.2) + tone(2 * fr, 0.1)
    if fault == "broken_bar":
        y *= 1.0 + 0.6 * sv * np.cos(2 * np.pi * 2 * s * f * t)
        y += tone(f * (1 - 2 * s), 0.4 * sv) + tone(f * (1 + 2 * s), 0.3 * sv)
    elif fault == "eccentric":
        y += tone(fs - fr, 0.5 * sv) + tone(fs + fr, 0.5 * sv) + tone(fr, 0.4 * sv)
    y = _unit(y) + 0.15 * _unit(_bp(rng.standard_normal(n), 1800.0, 0.8))
    if fault == "bearing":
        y = y + 0.8 * sv * _unit(_fit(bearing(sr=SR, dur=n / SR, rpm=fr * 60.0, state="outer", severity=0.5 + 0.5 * sv, seed=seed), n))
    return _fin(_lp(y, 8000.0, 2), sr, 0.5, 0.05, 0.08)


@sfx("machine_hum", "An auditory-icon machine: a hum whose pitch is the machine's size and whose wobble shows how hard and fast it works.",
     dur=(2.0, "s"), size=(120.0, "Hz: the hum (big machine = low)"), work=(0.5, "0..1 how hard it works: wobble depth"),
     speed=(2.0, "Hz: how fast it runs: wobble rate"), mod_freq=(30.0, "Hz: the rattle of its parts"), seed=(0, "variation"))
def machine_hum(sr=DEFAULT_SR, dur=2.0, size=120.0, work=0.5, speed=2.0, mod_freq=30.0, seed=0):
    """mod_amp = work sin(2 pi speed t); out = sin(2 pi (size t + integral of mod_amp sin(2 pi mod_freq t) size))."""
    n = _n(dur)
    t = np.arange(n) / SR
    f0 = float(np.clip(size, 20.0, 4000.0))
    mod = float(np.clip(work, 0, 1)) * np.sin(2 * np.pi * float(speed) * t) * np.sin(2 * np.pi * float(mod_freq) * t)
    y = np.sin(2 * np.pi * np.cumsum(f0 * (1.0 + 0.5 * mod)) / SR)
    return _fin(y, sr, 0.5, 0.02, 0.05)


UI_CATEGORIES = ("hover", "tap", "transition", "success", "error", "warning", "notification")


def ui_category(dur, attack=0.01, sweep_st=0.0, noise_s=0.0, tonal=True, peak_hz=800.0, rising=False, layers=1, filter_env=False):
    """The interface category a sound's features fit, first match in this order: hover (dur <= .14 s); tap (<= .3 s,
    attack <= .03 s); transition (a sweep >= 4 semitones, or noise >= .08 s, or a filter envelope); success (rising layers,
    tonal, >= .15 s); error (<= .6 s and a fall >= 3 st with the peak <= 500 Hz); warning (<= .4 s, 200-1000 Hz);
    notification (tonal, peak >= 500 Hz, .12-.9 s); else None."""
    if dur <= 0.14:
        return "hover"
    if dur <= 0.3 and attack <= 0.03:
        return "tap"
    if abs(sweep_st) >= 4 or noise_s >= 0.08 or filter_env:
        return "transition"
    if rising and tonal and dur >= 0.15 and layers >= 1:
        return "success"
    if dur <= 0.6 and sweep_st <= -3 and peak_hz <= 500:
        return "error"
    if dur <= 0.4 and 200 <= peak_hz <= 1000:
        return "warning"
    if tonal and peak_hz >= 500 and 0.12 <= dur <= 0.9:
        return "notification"
    return None


# =============================================================================================
# INTERFACE SOUNDS
# =============================================================================================
# A layer is an oscillator with an optional glide and 2:1 FM, a linear attack and an exponential fall of
# time constant decay / 3; success figures rise by consonant intervals, error figures fall or repeat low,
# warnings repeat a pitch. Square and saw layers are rounded by a low-pass, for the ear.

UI_KINDS = ("scan", "confirm", "deny", "hack", "decrypt", "upload", "warning", "target_lock", "implant", "neural_link")
_PENTA = np.array([0, 3, 5, 7, 10, 12, 15, 17, 19, 22, 24])


def _st(semitones) -> float:
    return 2.0 ** (np.asarray(semitones, float) / 12.0)


def _tone(sr, f0, decay, wave="sine", f1=None, attack=0.003, fm=0.0):
    n = samples(attack + decay, sr)
    f = np.geomspace(f0, f1 or f0, n)
    if wave == "sine":
        x = O.fm(f, n, sr, 2.0, fm / (2.0 * f0)) if fm else O.sine(f, n, sr)
    else:
        x = _round(O.osc(wave, f, n, sr), f, sr) * 0.6
    t = np.arange(n) / sr
    return x * np.minimum(t / attack, 1.0) * np.exp(-np.maximum(t - attack, 0.0) * 3.0 / decay)


def _swish(sr, dur, lo, hi, q, seed):
    n = samples(dur, sr)
    return F.lowpass(F.biquad(O.white(n, seed=seed), "bandpass", np.geomspace(lo, hi, n), sr, q=q), 3500.0, sr, order=2) * np.hanning(n)


@sfx("cyber_ui", "Sci-fi interface cue: scan, confirm, deny, hack, decrypt, upload, warning, target_lock, implant or neural_link. Every seed is another cue of the same family.",
     kind=("confirm", "|".join(UI_KINDS)), pitch=(1.0, "0.5..2 transposes the whole cue"), seed=(0, "variation"))
def cyber_ui(sr=DEFAULT_SR, kind="confirm", pitch=1.0, seed=0):
    if kind not in UI_KINDS:
        raise ValueError(f"cyber_ui kind {kind!r}: use one of {', '.join(UI_KINDS)}")
    rng = np.random.default_rng([abs(int(seed)), 74, UI_KINDS.index(kind)])
    u, p, s = rng.uniform, float(np.clip(pitch, 0.25, 4.0)), int(seed)
    ev = []                                                  # (start s, signal, gain)
    crush = False
    if kind == "scan":
        d = u(0.28, 0.4)
        lo = u(600, 900) * p
        ev.append((0.0, _swish(sr, d, lo, lo * u(4, 7), 3.0, s), 1.0))
        beep = _tone(sr, u(900, 1300) * p, d, "square", attack=0.005)
        ev.append((0.0, beep * (np.sin(2 * np.pi * u(18, 30) * np.arange(len(beep)) / sr) > 0), 0.12))
    elif kind == "confirm":
        f, gap = u(350, 600) * p, u(0.07, 0.14)
        for i, semi in enumerate((0, 7, 12)):
            ev.append((i * gap, _tone(sr, f * _st(semi), 0.3 if i == 2 else 0.12, fm=u(100, 300)), 0.22))
    elif kind == "deny":
        f, gap = u(180, 320) * p, u(0.05, 0.08)
        for i in range(3):
            ev.append((i * gap, _tone(sr, f, u(0.03, 0.05), "square"), 0.3))
        ev.append((0.0, _tone(sr, f * _st(1), 3 * gap, "triangle"), 0.1))          # dissonant pair
    elif kind in ("hack", "decrypt"):
        f, gap, t, crush = u(400, 600) * p, 1.0 / u(20, 40), 0.0, True
        for i in range(int(rng.integers(10, 17))):
            if rng.random() > 0.1:
                ev.append((t, _tone(sr, f * _st(rng.choice(_PENTA)), u(0.015, 0.04), "square", attack=0.001), 0.22))
            t += gap
            gap *= 1.15 if kind == "decrypt" else 1.0
        if kind == "decrypt":
            ev.append((t, _tone(sr, f * _st(12), 0.35, fm=120.0), 0.3))
    elif kind == "upload":
        f, gap, step = u(350, 500) * p, u(0.04, 0.07), rng.choice([3, 4, 5])
        k = int(rng.integers(6, 9))
        ev.append((0.0, _swish(sr, k * gap + 0.1, 300 * p, 2400 * p, 1.2, s), 0.5))
        for i in range(k):
            ev.append((i * gap, _tone(sr, f * _st(step * i), 0.06, "triangle"), 0.22))
    elif kind == "warning":
        f, gap = u(400, 600) * p, u(0.09, 0.13)
        for rep in range(2):
            for i, semi in enumerate((rng.choice([3, 4]), 0)):
                ev.append((rep * 2.6 * gap + i * gap, _tone(sr, f * _st(semi), 0.1, "square"), 0.25))
    elif kind == "target_lock":
        f, gap, t = u(700, 1000) * p, u(0.13, 0.17), 0.0
        for _ in range(6):
            ev.append((t, _tone(sr, f, 0.02, attack=0.001), 0.25))
            t += gap
            gap *= 0.7
        hold = _tone(sr, u(1200, 1500) * p, 0.6)
        ev.append((t, hold[:samples(0.17, sr)], 0.3))
    elif kind == "implant":
        ev.append((0.0, F.bandpass(O.white(samples(0.03, sr), seed=s), u(800, 2500), sr, 1.5) * np.hanning(samples(0.03, sr)), 1.2))
        f = u(60, 120)
        ev.append((0.0, _tone(sr, f, 0.2, f1=f / 2), 0.6))
        ev.append((0.012, _tone(sr, u(1600, 2400) * p, 0.09, fm=u(200, 350)), 0.1))
    else:                                                    # neural_link: a detuned pair blooming
        f = u(300, 480) * p
        cents = u(5, 14)
        for g, r in ((0.2, 1.0), (0.12, 1.5)):
            for det in (-cents, cents):
                ev.append((0.0, _tone(sr, f * r * _st(det / 100.0), 0.6, attack=u(0.03, 0.09)), g))
        ev.append((0.0, _tone(sr, f + 40.0, 0.6, attack=0.2), 0.08))                # the same tone shifted 40 Hz
        ev.append((0.0, _swish(sr, 0.5, 500 * p, 3000 * p, 2.0, s), 0.25))
    y = np.zeros(max(int(t0 * sr) + len(x) for t0, x, _ in ev) + 8)
    for t0, x, g in ev:
        i = int(t0 * sr)
        y[i:i + len(x)] += g * x
    if crush:
        y = FX["bitcrush"]["fn"](y, sr, bits=6, rate=11025)
        y = F.lowpass(y, 4000.0, sr, order=2)
    return declick(y * (0.6 / (np.max(np.abs(y)) + 1e-12)), sr, 4.0)


# =============================================================================================
# GLITCH AND VOICE EFFECTS
# =============================================================================================

def _edge(L: int, ramp: int, ndim: int) -> np.ndarray:
    w = np.ones(L)
    if ramp > 0:
        w[:ramp] = np.linspace(0.0, 1.0, ramp)
        w[-ramp:] = w[:ramp][::-1]
    return w[:, None] if ndim == 2 else w


GSM_CHANNEL = {"good": (1 / 750, 0.00016, 0.045), "fair": (15 / 750, 0.0061, 0.083), "poor": (95 / 750, 0.041, 0.13)}
# frame erasure rate, bit error rate of the protected class I bits and of the unprotected class II bits


def bit_errors(y, ber_hi, ber_lo, rng):
    """Flip bits of a 16-bit copy of ``y`` (peak at full scale): the 8 high bits at ``ber_hi`` (protected), the 8 low at ``ber_lo``."""
    peak = float(np.max(np.abs(y))) + 1e-12
    q = np.round(np.clip(y / peak, -1, 1) * 32767).astype(np.int16).view(np.uint16)
    p = np.where(np.arange(16) >= 8, ber_hi, ber_lo)
    flips = (rng.random(q.shape + (16,)) < p).astype(np.uint16)
    mask = (flips << np.arange(16, dtype=np.uint16)).sum(axis=-1).astype(np.uint16)
    return (q ^ mask).view(np.int16).astype(np.float64) / 32767.0 * peak


@effect("packet_loss", "Lossy digital link: packets drop out in bursts and are muted, replaced by the last good one fading away, or by its frozen spectrum (bad radio, failing implant, VoIP).",
        loss=(0.1, "fraction of packets lost, 0..0.95"), burst=(2.0, "average number of packets lost in a row (1 = scattered)"),
        packet_ms=(20.0, "packet length, 10..60 ms"), conceal=("repeat", "mute | repeat (stutter) | noise (the last spectrum with random phase)"),
        channel=("", "GSM radio channel instead of loss: good | fair | poor (C/I 10, 7, 4 dB): frames erased 1, 15, 95 in 750 and bit errors in the samples"),
        seed=(0, "which packets"))
def packet_loss(x, sr=DEFAULT_SR, loss=0.1, burst=2.0, packet_ms=20.0, conceal="repeat", seed=0, channel=""):
    """Gilbert two-state channel: P(bad -> good) = 1 / burst, P(good -> bad) chosen so the long-run
    loss is ``loss``. Concealment as in a waveform-substitution decoder: each lost packet is the
    previous output packet at 0.7 of its level."""
    if conceal not in ("mute", "repeat", "noise"):
        raise ValueError(f"packet_loss conceal {conceal!r}: use mute, repeat or noise")
    rng = np.random.default_rng(int(seed))
    y = np.array(x, dtype=np.float64)
    if channel:
        if channel not in GSM_CHANNEL:
            raise ValueError(f"packet_loss channel {channel!r}: use {', '.join(GSM_CHANNEL)}")
        loss, ber_hi, ber_lo = GSM_CHANNEL[channel]
        y = bit_errors(y, ber_hi, ber_lo, rng)
    L = max(16, int(float(np.clip(packet_ms, 2.0, 500.0)) * 1e-3 * sr))
    p, r = float(np.clip(loss, 0.0, 0.95)), 1.0 / max(float(burst), 1.0)
    g = min(1.0, p * r / (1.0 - p))
    w = _edge(L, min(L // 4, int(0.001 * sr)), y.ndim)
    bad = False
    for i in range(L, len(y) - L + 1, L):
        bad = rng.random() < ((1.0 - r) if bad else g)
        if not bad:
            continue
        if conceal == "mute":
            y[i:i + L] = 0.0
        elif conceal == "repeat":
            y[i:i + L] = 0.7 * y[i - L:i] * w
        else:
            S = np.abs(np.fft.rfft(y[i - L:i], axis=0))
            y[i:i + L] = 0.7 * np.fft.irfft(S * np.exp(1j * rng.uniform(0, 2 * np.pi, S.shape)), L, axis=0) * w
    return y


@effect("beat_repeat", "Buffer repeat: now and then a slice of the sound is caught and played again in place of what follows (stutter, roll, reverse), without changing the length.",
        grid=(0.125, "slice length in seconds (at 120 bpm: 0.125 = a 16th note)"), chance=(0.3, "probability that a slice is caught"),
        repeats=(3, "times the caught slice is played"), divide=(1, "play only the first 1/divide of the slice, that many times (2, 4, 8 = rolls)"),
        decay=(0.85, "level of each repeat relative to the one before"), reverse=(0.2, "probability that a caught slice plays backwards"),
        seed=(0, "which slices"))
def beat_repeat(x, sr=DEFAULT_SR, grid=0.125, chance=0.3, repeats=3, divide=1, decay=0.85, reverse=0.2, seed=0):
    """A slice of one grid step repeated ``rep`` times, its edges crossfaded."""
    rng = np.random.default_rng(int(seed))
    y = np.array(x, dtype=np.float64)
    L = max(32, int(float(grid) * sr))
    d = int(np.clip(divide, 1, 64))
    sub = max(16, L // d)
    w = _edge(sub, min(sub // 4, int(0.002 * sr)), y.ndim)
    i = 0
    while i + L <= len(y):
        if rng.random() >= float(chance):
            i += L
            continue
        unit = np.array(x[i:i + sub], dtype=np.float64)
        if rng.random() < float(reverse):
            unit = unit[::-1]
        unit = np.concatenate([unit * w] * d)[:L]
        for k in range(max(1, int(repeats))):
            j = i + k * L
            if j + len(unit) > len(y):
                break
            y[j:j + len(unit)] = unit * float(decay) ** k
        i += max(1, int(repeats)) * L
    return y


PHI = 1.618033988749895


def ps_drift(tau):
    """Wander of the read point, -1..1: (sin(2 pi phi tau) + sin(2 pi phi^2 tau) + sin(2 pi phi^3 tau)) / 3, tau in s."""
    tau = np.asarray(tau, float)
    return (np.sin(2 * np.pi * PHI * tau) + np.sin(2 * np.pi * PHI ** 2 * tau) + np.sin(2 * np.pi * PHI ** 3 * tau)) / 3.0


def ps_repitch(x, semitones):
    """Pitch by resampling (2^(st/12) faster, shorter): the stretch is multiplied by the same factor to keep the length."""
    if not semitones:
        return x
    from fractions import Fraction
    r = Fraction(2.0 ** (-float(semitones) / 12.0)).limit_denominator(1000)
    return resample_poly(x, r.numerator, r.denominator, axis=0)


_PS_RECIPES = {"subtle": ((0.7, 0.45, 0), (1, 0.75, 0), (1.4, 0.45, 0)), "standard": ((0.5, 0.55, 0), (1, 0.70, 0), (2, 0.50, 0)),
               "lush": ((0.25, 0.40, 0), (0.5, 0.55, 0), (1, 0.70, 0), (2, 0.55, 0), (4, 0.40, 0)),
               "shimmer": ((0.5, 0.55, 0), (1, 0.70, 0), (1, 0.45, 12)), "shimmer_deep": ((0.5, 0.50, -12), (1, 0.70, 0), (1, 0.45, 12))}


@effect("paulstretch", "Extreme time-stretch that turns any sound into a slowly evolving wash (random-phase windows), or holds one instant as an endless pad.",
        stretch=(8.0, "length ratio, 1..1000"), window=(0.25, "seconds: longer = smoother and more blurred in time"),
        hold=(-1.0, "seconds: freeze the sound at this instant instead of stretching (-1 = off)"),
        length=(4.0, "seconds of output when `hold` is used"), smear=(0.0, "0..1 blur across frequency (a held note becomes a noise band)"),
        seed=(0, "phase seed"), phase=(1.0, "1 random phases .. 0 the analysed ones, keeps transients"), continuity=(0.0, "0..1 each frame's phase stays near the last (about 0.3 smooths without pumping)"),
        scan=(0.0, "with hold: the frozen instant travels, by this fraction of the source over the output"), loop=(False, "the source wraps round: for loops, its end does not fade to silence"),
        recipe=(None, "subtle|standard|lush|shimmer|shimmer_deep: several window sizes stacked, some an octave up or down"),
        drift=(0.0, "0..1 the read point wanders round its place (three golden-ratio sines, up to 30 % of the source): a slowly changing wash"))
def paulstretch(x, sr=DEFAULT_SR, stretch=8.0, window=0.25, hold=-1.0, length=4.0, smear=0.0, seed=0, phase=1.0, continuity=0.0, scan=0.0, loop=False, recipe=None, drift=0.0):
    """Paulstretch: windows of the input taken every
    (W / 2) / stretch samples, each resynthesised with its magnitudes and uniformly random phases,
    overlap-added at W / 2, then the amplitude ripple of the overlap is divided out with
    h - (1 - h) cos(2 pi i / (W / 2)), h = (1 + sqrt(0.5)) / 2. The window is
    (1 - x^2)^1.25. ``hold`` and ``smear`` are a snapshot freeze
    (one window, a box blur over max(1, 50 smear) bins, fresh phases every hop); the same port gives
    phase, continuity, scan, loop and the recipes (window scale, gain, semitones; seed + i * 0x9E3779B97F4A7C15)."""
    if recipe:
        if recipe not in _PS_RECIPES:
            raise ValueError(f"unknown paulstretch recipe {recipe!r}; use one of {sorted(_PS_RECIPES)}")
        kw = dict(hold=hold, length=length, smear=smear, phase=phase, continuity=continuity, scan=scan, loop=loop, drift=drift)
        ys = [g * paulstretch(ps_repitch(np.asarray(x, float), st), sr, window=float(window) * w, stretch=float(stretch) * 2.0 ** (st / 12.0),
                              seed=(int(seed) + i * 0x9E3779B97F4A7C15) % 2 ** 63, **kw) for i, (w, g, st) in enumerate(_PS_RECIPES[recipe])]
        m = min(len(y) for y in ys)
        y = sum(y[:m] for y in ys)
        return y * float(np.max(np.abs(x))) / (float(np.max(np.abs(y))) + 1e-12)
    rng = np.random.default_rng(int(seed))
    n = x.shape[0]
    W = max(64, int(float(window) * sr) // 2 * 2)
    half = W // 2
    win = (1.0 - np.linspace(-1.0, 1.0, W) ** 2) ** 1.25
    st = float(np.clip(stretch, 1.0, 1000.0))
    n_out = int(float(length) * sr) if hold >= 0 else int(n * st)
    frames = n_out // half + 1
    starts = np.full(frames, float(hold) * sr) if hold >= 0 else np.arange(frames) * half / st
    if hold >= 0 and scan:
        starts = starts + float(scan) * n * np.arange(frames) / frames
    if drift:
        starts = starts + ps_drift(np.arange(frames) * half / sr) * float(np.clip(drift, 0.0, 1.0)) * 0.30 * n
    starts = (starts % max(n, 1) if loop else np.clip(starts, 0, max(n - 1, 0))).astype(int)
    h = (1.0 + math.sqrt(0.5)) * 0.5
    ripple = np.tile(h - (1.0 - h) * np.cos(2 * np.pi * np.arange(half) / half), frames)
    blur = 2 * max(1, int(float(smear) * 50)) + 1 if smear > 0.01 else 0

    def one(s):
        sp = np.resize(s, n + W) if loop else np.concatenate([s, np.zeros(W)])
        out = np.zeros(frames * half + W)
        prev = None
        for i, a in enumerate(starts):
            spec = np.fft.rfft(sp[a:a + W] * win)
            mag = np.abs(spec)
            if blur:
                mag = uniform_filter1d(mag, blur)
            ang = rng.uniform(0, 2 * np.pi, len(mag)) if phase >= 1.0 else np.angle(spec) + float(phase) * rng.uniform(-np.pi, np.pi, len(mag))
            if continuity and prev is not None:
                ang = prev + (1.0 - float(continuity)) * np.angle(np.exp(1j * (ang - prev)))
            prev = ang
            out[i * half:i * half + W] += np.fft.irfft(mag * np.exp(1j * ang), W) * win
        out[:frames * half] *= ripple
        return out[half:half + n_out]

    y = _per_channel(one, np.asarray(x, float))
    rms = lambda a: math.sqrt(float(np.mean(np.square(a)))) + 1e-12
    return y * min(rms(x) / rms(y), float(np.max(np.abs(x))) / (float(np.max(np.abs(y))) + 1e-12))   # never louder than the input


@effect("waveset", "Waveset repetition: each cycle between zero crossings is played several times and the next ones skipped. Buzzy, pitch drops by octaves, speech turns to insect chatter.",
        repeats=(2, "times each kept cycle is played (2 = an octave down in feel, 4 = two)"), keep_length=(True, "false = keep every cycle, so the sound gets `repeats` times longer"))
def waveset(x, sr=DEFAULT_SR, repeats=2, keep_length=True):
    """Waveset distortion: each cycle between rising zero crossings repeated."""
    m = to_mono(np.asarray(x, float))
    zc = np.flatnonzero((m[:-1] < 0) & (m[1:] >= 0)) + 1
    r = max(1, int(repeats))
    if len(zc) < 3 or r == 1:
        return x
    sets = [x[a:b] for a, b in zip(zc[:-1], zc[1:])]
    y = np.concatenate([x[:zc[0]]] + [s for ws in (sets[::r] if keep_length else sets) for s in [ws] * r] + [x[zc[-1]:]])
    return pad_to(y, len(x))[:len(x)] if keep_length else y


@effect("formant_shift", "Move the formants without changing the pitch: a bigger or smaller speaker, the same notes.",
        semitones=(-3.0, "+ smaller / brighter, - bigger / darker (-12..12)"))
def formant_shift(x, sr=DEFAULT_SR, semitones=-3.0):
    x = np.asarray(x, float)
    return S.transfer_envelope(x, x, sr, shift=float(np.clip(semitones, -24.0, 24.0)))


@effect("cyber_voice", "Synthetic / augmented voice in one step: formant shift, vocoder, ring modulation, a lossy link and a radio band. For speech layers and recorded voices.",
        formant=(-3.0, "semitones the formants move (- = larger, colder)"), pitch=(110.0, "vocoder carrier: Hz or a note name"),
        vocode=(0.6, "0 natural .. 1 fully vocoded"), ring=(70.0, "ring modulator Hz (30..180)"), ring_mix=(0.25, "0..1"),
        loss=(0.06, "fraction of 30 ms packets lost (0 = clean link)"), conceal=("noise", "mute | repeat | noise: what a lost packet becomes"),
        radio=(1.0, "0 full band .. 1 the 300-3400 Hz band of a comms channel"), seed=(0, "variation"))
def cyber_voice(x, sr=DEFAULT_SR, formant=-3.0, pitch=110.0, vocode=0.6, ring=70.0, ring_mix=0.25, loss=0.06,
                conceal="noise", radio=1.0, seed=0):
    rms = lambda a: math.sqrt(float(np.mean(np.square(a)))) + 1e-12
    y = formant_shift(x, sr, formant) if formant else np.asarray(x, float)
    if vocode > 0:
        v = FX["vocoder"]["fn"](y, sr, carrier="saw", freq=pitch, bands=16, hiss=0.12, seed=int(seed))
        y = y * (1.0 - float(vocode)) + v * (float(vocode) * rms(y) / rms(v))
    if ring_mix > 0:
        y = FX["ringmod"]["fn"](y, sr, freq=float(ring), mix=float(ring_mix))
    if loss > 0:
        y = packet_loss(y, sr, loss=loss, burst=2.0, packet_ms=30.0, conceal=conceal, seed=seed)
    if radio > 0:
        pk = float(np.max(np.abs(y))) + 1e-12
        y = y * (1.0 - float(radio)) + float(radio) * pk * FX["telephone"]["fn"](y / pk, sr, drive=2.0)
    return y * min(rms(x) / rms(y), 0.99 / (float(np.max(np.abs(y))) + 1e-12))   # the input's level, never clipping


# =============================================================================================
# STATISTICAL TEXTURES
# =============================================================================================
# 30 half-cosine filters equally spaced on the ERB scale from 52 to 8844 Hz plus a low and a high end filter
# (squared responses sum to one), Hilbert envelopes compressed with the exponent 0.3 and sampled at 400 Hz,
# statistics of those envelopes, imposed on noise by iteration while each band keeps its own fine structure.
# ponytail: a reduced statistic set. Kept: the envelope marginals (as 64 quantiles, which carry the
# mean, variance, skew and kurtosis), the full band-to-band envelope correlation, and the
# modulation power in 9 octave bands from 0.5 Hz, the C1 correlations (each octave modulation band
# 2-128 Hz across all cochlear bands, not only the two nearest neighbours over 3-100 Hz)
# and the C2 correlations (frequency-doubled analytic band against the next octave up, 1-128 Hz:
# the phase that tells an abrupt onset from an offset or a click). Imposed by projection (Cholesky
# recolouring, rotation onto the target complex correlation, band rescaling, rank matching) instead
# of a conjugate-gradient search.

TEX_BANDS, TEX_ENV_SR, TEX_POW, TEX_Q = 30, 400.0, 0.3, 64
_TEX_MOD = 0.5 * 2.0 ** np.arange(10)                        # modulation band edges, Hz
_TEX_C1, _TEX_C2 = range(2, 8), range(1, 7)                  # modulation bands with C1; lower band of each C2 pair


def _erb(f):
    return 21.4 * np.log10(1.0 + 0.00437 * np.asarray(f, float))


def _tex_len(n: int) -> int:
    while sfft.next_fast_len(n) != n:
        n -= 1
    return n


def _tex_bank(n: int, sr: int) -> np.ndarray:
    e = _erb(np.fft.rfftfreq(n, 1.0 / sr))
    c = np.linspace(_erb(52.0), _erb(min(8844.0, 0.45 * sr)), TEX_BANDS + 2)
    H = np.cos(np.clip((e[None, :] - c[:, None]) / (c[1] - c[0]), -1.0, 1.0) * np.pi / 2)
    H[0, e < c[0]] = 1.0
    H[-1, e > c[-1]] = 1.0
    return H


def _tex_envelopes(x: np.ndarray, H: np.ndarray, m: int):
    """Compressed envelopes at the envelope rate (bands, m) and the phase of each band's fine structure."""
    n = len(x)
    X = sfft.rfft(x)
    env, ph = np.empty((len(H), m)), np.empty((len(H), n), np.float32)
    Z = np.zeros(n, complex)
    for k, h in enumerate(H):
        Z[:len(X)] = 2.0 * X * h
        a = sfft.ifft(Z)
        ph[k] = np.angle(a)
        env[k] = resample(np.abs(a) ** TEX_POW, m)
    return env, ph


def _tex_masks(m: int):
    fm = np.fft.rfftfreq(m, 1.0 / TEX_ENV_SR)
    return [(fm >= lo) & (fm < hi) for lo, hi in zip(_TEX_MOD[:-1], _TEX_MOD[1:])]


def _tex_analytic(Zf: np.ndarray, k: np.ndarray, m: int) -> np.ndarray:
    """Analytic signals (bands, m) of the envelopes' modulation band ``k`` (a mask on the rfft bins)."""
    A = np.zeros((len(Zf), m), complex)
    k = np.flatnonzero(k)
    A[:, k] = 2.0 * Zf[:, k]
    return np.fft.ifft(A, axis=1)


def _tex_double(a: np.ndarray) -> np.ndarray:
    """Frequency doubling d = a^2 / |a|: twice the phase, same magnitude."""
    return a * a / (np.abs(a) + 1e-30)


def _tex_unit(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-30)


def texture_stats(x: np.ndarray, sr: int = DEFAULT_SR) -> dict:
    """Envelope statistics of a recording (at least a second or two of a steady texture)."""
    x = to_mono(np.asarray(x, float))
    n = _tex_len(len(x))
    m = int(n * TEX_ENV_SR / sr)
    env, _ = _tex_envelopes(x[:n], _tex_bank(n, sr), m)
    Zf = np.fft.rfft(env - env.mean(axis=1, keepdims=True), axis=1)
    P = np.abs(Zf) ** 2
    masks = _tex_masks(m)
    mod = np.stack([P[:, k].sum(axis=1) for k in masks], axis=1) / (P[:, 1:].sum(axis=1, keepdims=True) + 1e-30)
    A = [_tex_analytic(Zf, k, m) for k in masks]
    c1 = np.nan_to_num(np.stack([np.corrcoef(A[j].real) for j in _TEX_C1]))
    c2 = np.stack([np.sum(np.conj(_tex_unit(_tex_double(A[j]))) * _tex_unit(A[j + 1]), axis=1) for j in _TEX_C2], axis=1)
    return {"sr": int(sr), "quantiles": np.quantile(env, np.linspace(0, 1, TEX_Q), axis=1).T, "modulation": mod,
            "corr": np.corrcoef(env), "c1": c1, "c2": c2}


def texture_synth(stats: dict, dur: float, seed: int = 0, iters: int = 12, target_snr: float | None = None) -> np.ndarray:
    """A new texture of any length with the statistics of ``texture_stats``, at ``stats['sr']``. The
    filters are circular, so the result loops without a seam. ``target_snr`` (dB; 30 is enough for every
    class, within 60 iterations) ends the ``iters`` passes early, at the cost of measuring the statistics each pass.
    # ponytail: 32 float32 phase tracks in memory (6 MB per second at 48 kHz): capped at 60 s; loop it."""
    sr = stats["sr"]
    n = _tex_len(int(float(np.clip(dur, 0.5, 60.0)) * sr))
    m = int(n * TEX_ENV_SR / sr)
    H = _tex_bank(n, sr)
    B = len(H)
    q, grid = stats["quantiles"], np.linspace(0, 1, TEX_Q)
    eye = 1e-3 * np.eye(B)
    Lt = np.linalg.cholesky(stats["corr"] + eye)
    masks = _tex_masks(m)
    Lc1 = [np.linalg.cholesky(c + eye) for c in stats["c1"]]
    c2 = stats["c2"]
    x = np.random.default_rng(int(seed)).standard_normal(n)
    for _ in range(max(1, int(iters))):
        env, ph = _tex_envelopes(x, H, m)
        z = env - env.mean(axis=1, keepdims=True)
        z /= z.std(axis=1, keepdims=True) + 1e-12
        z = Lt @ np.linalg.solve(np.linalg.cholesky(np.corrcoef(z) + eye), z)
        Zf = np.fft.rfft(z, axis=1)
        for i, j in enumerate(_TEX_C1):                        # C1: recolour each modulation band across bands
            k = masks[j]
            if k.sum() < 2 or not np.trace(stats["c1"][i]):        # band empty in the source
                continue
            Zb = np.zeros_like(Zf)
            Zb[:, k] = Zf[:, k]
            b = np.fft.irfft(Zb, m, axis=1)
            s = b.std(axis=1, keepdims=True) + 1e-12
            w = Lc1[i] @ np.linalg.solve(np.linalg.cholesky(np.corrcoef(b / s) + eye), b / s)
            Zf[:, k] = np.fft.rfft(w * s, axis=1)[:, k]
        for i, j in enumerate(_TEX_C2):                        # C2: rotate the upper band onto the target phase
            k = masks[j + 1]
            if k.sum() < 2 or masks[j].sum() < 2:
                continue
            dn = _tex_unit(_tex_double(_tex_analytic(Zf, masks[j], m)))
            a = _tex_analytic(Zf, k, m)
            r = _tex_unit(a - np.sum(np.conj(dn) * a, axis=1, keepdims=True) * dn)
            c = c2[:, i:i + 1]
            a = np.linalg.norm(a, axis=1, keepdims=True) * (c * dn + np.sqrt(np.maximum(1.0 - np.abs(c) ** 2, 0.0)) * r)
            Zf[:, k] = np.fft.rfft(a.real, axis=1)[:, k]
        P = np.abs(Zf) ** 2
        tot = P[:, 1:].sum(axis=1)
        for j, k in enumerate(masks):
            Zf[:, k] *= np.sqrt(stats["modulation"][:, j] * tot / (P[:, k].sum(axis=1) + 1e-30))[:, None]
        z = np.fft.irfft(Zf, m, axis=1)
        X = np.zeros(n // 2 + 1, complex)
        for b in range(B):
            rank = np.argsort(np.argsort(z[b])) / (m - 1.0)
            e = np.maximum(resample(np.interp(rank, grid, q[b]), n), 0.0) ** (1.0 / TEX_POW)
            X += sfft.rfft(e * np.cos(ph[b])) * H[b]
        x = sfft.irfft(X, n)
        if target_snr is not None and min(stats_snr(stats, texture_stats(x, sr)).values()) >= target_snr:
            break
    return x


@effect("texturize", "Turn a recording of a steady texture (rain, a server room, a crowd, static) into a new, endless, seamlessly looping one of any length with the same character.",
        length=(8.0, "seconds of output, up to 60"), iters=(12, "refinement passes (more = closer, slower)"), seed=(0, "a different rendition"))
def texturize(x, sr=DEFAULT_SR, length=8.0, iters=12, seed=0):
    if x.shape[0] < sr // 2:
        return x
    return texture_synth(texture_stats(x, sr), length, seed, iters)



# =============================================================================================
# PSYCHOACOUSTIC AND TEXTURE HELPERS, MODULATION EFFECTS
# =============================================================================================

def fluctuation_rel(fmod):
    """Fluctuation strength against modulation rate, 1 at its 4 Hz maximum: 2 / ((fmod / 4 Hz) + (4 Hz / fmod))."""
    f = np.maximum(np.asarray(fmod, float), 1e-6)
    return 2.0 / (f / 4.0 + 4.0 / f)


def fluctuation_bbn(m, level_db, fmod):
    """Fluctuation strength (vacil) of amplitude-modulated broadband noise: 5.8 (1.25 m - 0.25)(0.05 L - 1) /
    ((fmod / 5 Hz)^2 + (4 Hz / fmod) + 1.5), m the modulation depth, L in dB."""
    f = np.maximum(np.asarray(fmod, float), 1e-6)
    return 5.8 * (1.25 * float(m) - 0.25) * (0.05 * float(level_db) - 1.0) / ((f / 5.0) ** 2 + 4.0 / f + 1.5)


def autocorr_zcr(x, sr=DEFAULT_SR, coeffs=12, window=0.0232, hop=0.0029):
    """Frame descriptors: the first ``coeffs`` normalised autocorrelation coefficients (lags 1..coeffs) and the
    zero-crossing rate (crossings per second), windows of 23.2 ms every 2.9 ms. Returns (ac [frames, coeffs], zcr)."""
    x = to_mono(np.asarray(x, float))
    w, h = max(int(window * sr), coeffs + 2), max(int(hop * sr), 1)
    starts = range(0, max(len(x) - w, 0) + 1, h)
    ac, zc = [], []
    for s in starts:
        fr = x[s:s + w]
        r0 = float(np.dot(fr, fr)) + 1e-20
        ac.append([float(np.dot(fr[:-k], fr[k:])) / r0 for k in range(1, coeffs + 1)])
        zc.append(np.count_nonzero(np.diff(np.signbit(fr))) * sr / w)
    return np.array(ac), np.array(zc)


@effect("adaptive_fm", "Adaptive FM: the sound reads itself through a delay swinging at fm = freq / ratio, which frequency-modulates "
        "everything in it (bright, metallic, the timbre of FM on a real input).",
        freq=(220.0, "Hz the carrier the depth is set for (the note of the input)"), index=(2.0, "modulation index 0..5"),
        ratio=(1.0, "c:m ratio 0.1..5 (an integer = harmonic)"), mix=(1.0, "wet 0..1"))
def adaptive_fm(x, sr=DEFAULT_SR, freq=220.0, index=2.0, ratio=1.0, mix=1.0):
    fc = float(np.clip(freq, 20.0, 5000.0))
    fm = fc / float(np.clip(ratio, 0.1, 5.0))

    def one(s):
        t = np.arange(len(s)) / sr
        d = float(np.clip(index, 0.0, 5.0)) / (np.pi * fc) * (0.5 * np.cos(2 * np.pi * fm * t) + 0.5)
        return np.interp(t - d, t, s, left=0.0)
    y = _per_channel(one, np.asarray(x, float))
    return (1.0 - mix) * x + mix * y


@effect("grain_delay", "Granular delay: raised-cosine grains read from the past at a random delay, size, pan, level and offset "
        "(each a centre plus a random spread), a scattered, glittering echo.",
        delay=(0.25, "s centre delay"), delay_spread=(0.15, "s random spread of the delay"), size=(0.08, "s grain length"),
        size_spread=(0.04, "s"), density=(20.0, "grains per second"), pan_spread=(0.6, "0..1"), level_spread=(0.4, "0..1"),
        mix=(0.5, "wet 0..1"), seed=(0, "variation"))
def grain_delay(x, sr=DEFAULT_SR, delay=0.25, delay_spread=0.15, size=0.08, size_spread=0.04, density=20.0, pan_spread=0.6,
                level_spread=0.4, mix=0.5, seed=0):
    rng = np.random.default_rng(int(seed))
    m = to_mono(np.asarray(x, float))
    n = len(m)
    out = np.zeros((n, 2))
    k = rng.random(int(n / sr * float(np.clip(density, 0.5, 400.0))))
    for t0 in np.sort(rng.integers(0, max(n, 1), len(k))):
        g = max(int(sr * abs(size + size_spread * rng.uniform(-1, 1))), 16)
        src = int(t0 - sr * max(delay + delay_spread * rng.uniform(-1, 1), 0.0))
        if src < 0 or t0 + g > n:
            continue
        grain = m[src:src + g] * (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(g) / g))
        vol = max(1.0 - level_spread * rng.random(), 0.0)
        pan = 0.5 + 0.5 * pan_spread * rng.uniform(-1, 1)
        out[t0:t0 + g, 0] += grain * vol * math.cos(0.5 * np.pi * pan)
        out[t0:t0 + g, 1] += grain * vol * math.sin(0.5 * np.pi * pan)
    out /= max(math.sqrt(float(np.clip(density, 0.5, 400.0)) * float(size)), 1.0)
    dry = np.column_stack([x, x]) if np.ndim(x) == 1 else np.asarray(x, float)[:, :2]
    return (1.0 - mix) * dry + mix * out
