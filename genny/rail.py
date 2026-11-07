"""Rail: rolling on welded or jointed track, diesel and electric traction, curve squeal, pantograph, doors and brakes.

`foley.train` stays what it is, a ride on jointed track. Sources of every constant: research/SOURCES.md, section rail.
Nothing has been listened to: see tests/test_rail.py for what was measured.
"""
from __future__ import annotations

import math

import numba
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfilt

from . import foley as FO
from .core import DEFAULT_SR
from .cyber import _HN_F, _third_octave_noise, pipe_engine
from .dsp import a_weighting_db
from .engines import diesel_engine
from .foley import P, _fin, _n, _put
from .friction import brake_squeal
from .matter import _lfn, arc
from .road import _band, _fit, _kind, _modes, air_system, ev_whine
from .sfx import sfx

E_STAR = 210e9 / (2.0 * (1.0 - 0.3 ** 2))                   # Pa, steel on steel, 210 GPa and nu .3
WHEEL_LOAD = 50e3                                           # N, a 10 t axle
PATCH_A, PATCH_B = 0.005, 0.005                             # m: semi-axes of the wheel-rail contact, about 1 cm^2


# =============================================================================================
# CONTACT: BRUSH MODEL (FASTSIM kernel)
# =============================================================================================
@numba.njit(cache=True)
def _brush(creep, ds, a, b, kq_h, load, mu_s, mu_d, vc, V, w, zeta, im, ny):
    """The brush model rolled in time: bristles enter unstrained at the leading edge, take ds of creep per sample, and saturate at
    mu p (x, y) with the Hertz parabolic pressure. The force acts on modes whose velocity feeds back into the creep. O(patch) per sample.
    Friction falls with slip velocity, mu_d + (mu_s - mu_d) / (1 + v / vc)."""
    n, dt, dy = len(creep), 1.0 / 48000.0, 2.0 * b / ny
    y2 = (-1.0 + (2.0 * np.arange(ny) + 1.0) / ny) ** 2
    M = np.maximum(np.round(2.0 * a * np.sqrt(1.0 - y2) / ds).astype(np.int64), 2)
    u, pf = np.zeros((ny, int(M.max()))), np.zeros((ny, int(M.max())))
    head = np.zeros(ny, dtype=np.int64)
    area, force = 0.0, 0.0
    for j in range(ny):
        for i in range(M[j]):
            x = -1.0 + (2.0 * i + 1.0) / M[j]
            pf[j, i] = max(1.0 - y2[j] - x * x * (1.0 - y2[j]), 0.0)
            area += pf[j, i]
            force += (i + 0.5)
    pf *= load / (area * ds * dy)                           # the discrete patch carries exactly the load
    wn = (8.0 / 3.0 * a * a * b) / (force * ds * ds * dy)   # and has exactly the brush stiffness (8ab/3)(Kq a/h)
    c, s = np.exp(-zeta * w * dt) * np.cos(w * dt), np.exp(-zeta * w * dt) * np.sin(w * dt)
    re, ie = np.zeros(len(w)), np.zeros(len(w))
    Q, out = np.zeros(n), np.zeros(n)
    for t in range(n):
        vel = float(np.sum(re))
        xi = creep[t] + vel / V
        mu = mu_d + (mu_s - mu_d) / (1.0 + abs(xi) * V / vc)
        total = 0.0
        for j in range(ny):
            m = M[j]
            head[j] = (head[j] - 1) % m
            u[j, head[j]] = 0.0
            for i in range(m):
                k = (head[j] + i) % m
                lim = mu * pf[j, i] / kq_h
                v = u[j, k] - xi * ds
                v = lim if v > lim else (-lim if v < -lim else v)
                u[j, k] = v
                total += v
        Q[t] = kq_h * wn * total * ds * dy
        for k in range(len(w)):
            r = c[k] * re[k] - s[k] * ie[k] + Q[t] * dt * im[k]
            ie[k] = s[k] * re[k] + c[k] * ie[k]
            re[k] = r
        out[t] = vel
    return Q, out


def creep_transient(creep, speed, load=WHEEL_LOAD, a=PATCH_A, b=PATCH_B, mu=0.3, ny=16):
    """Tangential force in N, sample by sample at 48 kHz, for a creep (number or curve), starting from rest at ``speed`` m/s."""
    creep = np.asarray(creep, float)
    if not creep.ndim:
        creep = np.full(int(4.0 * a / (speed / P)), float(creep))
    empty = np.zeros(0)
    return _brush(creep, speed / P, a, b, 1.1 * E_STAR / a, float(load), mu, mu, 1.0, float(speed), empty, empty, empty, ny)[0]


def creep_force(xi, speed=5.0, load=WHEEL_LOAD, a=PATCH_A, b=PATCH_B, mu=0.3, ny=16):
    """Steady Qx (N) at creep ``xi``. Small creep: Q = -(8ab/3)(Kq a/h) xi, Kq a/h = 1.1 E*; large: -mu P."""
    return float(creep_transient(xi, speed, load, a, b, mu, ny)[-1])


def creep_cutoff(speed, a=PATCH_A):
    """V / 2a, above which the creep force no longer follows the excitation, and f1 = 2.405 V / (2 pi a), first force frequency free of
    a trailing-edge singularity (J0 zero 2.405)."""
    return speed / (2.0 * a), 2.405 * speed / (2.0 * math.pi * a)


def creep_noise(creep, speed, n, rng, a=PATCH_A, b=PATCH_B, load=WHEEL_LOAD, mu=0.3):
    """What ``foley.train`` hears of a creeping wheel: a creep fluctuating over the rail's roughness (white, 50 %) through the brush of
    `creep_force`, which follows it only below V / 2a and flattens at mu P. The ripple of Q as a fraction of mu P, so it grows with the creep and
    dies once the whole patch slides."""
    xi = float(creep) * (1.0 + 0.5 * rng.standard_normal(n))
    q = creep_transient(np.ascontiguousarray(xi), float(speed), load, a, b, mu, 3)
    m = _n(0.01)                                            # the patch's own start from rest is not part of the ripple
    q[:m] = q[m]
    return (q - float(np.mean(q))) / (mu * load)


# =============================================================================================
# CONTACT: CORRUGATION
# =============================================================================================
def carter_force(xi, load=WHEEL_LOAD, a=PATCH_A, radius=0.46, mu=0.3):
    """Two-dimensional rolling contact, closed form: Q / mu P = 1 - (1 - R |xi| / (mu a))^2 below full slip at
    |xi| = mu a / R, mu P beyond; opposite in sign to the creep, as `creep_force`."""
    s = np.minimum(np.abs(np.asarray(xi, float)) * float(radius) / (float(mu) * float(a)), 1.0)
    return -np.sign(xi) * float(mu) * float(load) * (1.0 - (1.0 - s) ** 2)


def halfspace_resonance(mass, a, shear=81e9, density=7850.0, nu=0.3):
    """Hz and damping ratio of a rigid disc of radius ``a`` carrying ``mass`` on an elastic half-space: stiffness 4 G a / (1 - nu),
    radiation damping 3.4 a^2 sqrt(rho G) / (1 - nu), so zeta = 0.425 / sqrt(B), B = (1 - nu) m / (4 rho a^3)."""
    k = 4.0 * shear * a / (1.0 - nu)
    c = 3.4 * a * a * math.sqrt(density * shear) / (1.0 - nu)
    return math.sqrt(k / mass) / (2.0 * math.pi), c / (2.0 * math.sqrt(k * mass))


def contact_resonance(load=WHEEL_LOAD, mass=30.0, radius=0.3):
    """Hertz: delta0 = (9 W^2 / (16 R E*^2))^(1/3), stiffness 3 W / (2 delta0), on an effective mass of wheel and rail."""
    d0 = (9.0 * load ** 2 / (16.0 * radius * E_STAR ** 2)) ** (1.0 / 3.0)
    return math.sqrt(3.0 * load / (2.0 * d0 * mass)) / (2.0 * math.pi)


@numba.njit(cache=True)
def _hertz(z, w0, zeta):
    """x'' = w0^2 (2/3) ((1 + z - x)+^1.5 - 1) - 2 zeta w0 x', all in units of the static compression. Returns the compression."""
    dt = 1.0 / 48000.0
    x, v = 0.0, 0.0
    d = np.zeros(len(z))
    for i in range(len(z)):
        d[i] = 1.0 + z[i] - x
        c = max(d[i], 0.0)
        v += dt * (w0 * w0 * (2.0 / 3.0) * (c * math.sqrt(c) - 1.0) - 2.0 * zeta * w0 * v)
        x += dt * v
    return d


CONTACT_Q = 2.5                                             # in contact at A/d0 = .30, leaves it at .55: at resonance Q A = d0 puts Q in 1.8-3.3


def corrugated_contact(freq, ratio, n, f0=None):
    """Compression / static compression for corrugations of amplitude ``ratio`` static compressions passed at ``freq`` Hz (number or curve)."""
    f = np.broadcast_to(np.asarray(freq, float), (n,))
    z = float(ratio) * np.sin(2.0 * np.pi * np.cumsum(f) / P)
    return _hertz(np.ascontiguousarray(z), 2.0 * math.pi * (contact_resonance() if f0 is None else f0), 0.5 / CONTACT_Q)


# =============================================================================================
# ROLLING AND JOINTS
# =============================================================================================
def rolling_band(kmh):
    """315-2000 Hz at 60 km/h and under, 100-5000 Hz at 300 km/h and over, log interpolation between."""
    t = float(np.clip(np.log(max(kmh, 1.0) / 60.0) / np.log(5.0), 0.0, 1.0))
    return 315.0 * (100.0 / 315.0) ** t, 2000.0 * 2.5 ** t


def _rolling(kmh, n, rng, band=None, corrugation=0.0, wavelength=0.05):
    """Unit RMS at 60 km/h, +30 log10 v. Flat third-octaves inside the band, -30 dB outside."""
    lo, hi = band or rolling_band(kmh)
    y = _third_octave_noise(n, rng, np.where((_HN_F >= lo * 0.89) & (_HN_F <= hi * 1.12), 0.0, -30.0))
    y = y / (np.std(y) + 1e-12)
    if corrugation > 0:                                     # 0.4: level against the rolling noise
        y = y + 0.4 * corrugation_noise(kmh, corrugation, wavelength, n)
    return y * (max(kmh, 1.0) / 60.0) ** 1.5


def corrugation_noise(kmh, corrugation, wavelength, n):
    """Contact force ripple over a corrugated rail: Hertz (compression)^1.5 at V / wavelength; the wheel lifts above 0.5, so the force goes flat at 0."""
    d = np.maximum(corrugated_contact(kmh / 3.6 / wavelength, corrugation, n), 0.0) ** 1.5 - 1.0
    return _band(d, 60.0, 0.45 * P) / max(float(corrugation), 0.05)


JOINT_MODES = ((340.0, 720.0, 1250.0, 1900.0), (1.0, 0.7, 0.45, 0.25))   # Hz, inside the tram's 250-2000 Hz


def _joints(kmh, rail_len, n, rng, wheelbase=1.8):
    """t = (j L + offset) / v as `foley.train`; a joint is a wheel's blow lasting 1 ms on the bogie modes. Peak near 1."""
    v, x = max(kmh, 3.0) / 3.6, np.zeros(n)
    blow = 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(_n(0.001)) + 0.5) / _n(0.001))
    for j in range(int(n / P * v / rail_len) + 2):
        for off, g in ((0.0, 1.0), (wheelbase, 0.9)):
            _put(x, blow, (j * rail_len + off) / v + 0.05, g * rng.uniform(0.85, 1.0))
    y = _modes(x, JOINT_MODES[0], 0.12, JOINT_MODES[1])
    return y / (np.max(np.abs(y)) + 1e-12)


# =============================================================================================
# TRACTION
# =============================================================================================
BT_RPM, BT_KMH = 627.0, 15.0                                # metro at full tractive effort
PINION = 17                                                 # teeth, as `road._drive`


def traction_rpm(kmh):
    """Motor rpm: 627 at 15 km/h, proportional to speed (fixed gears)."""
    return np.asarray(kmh, float) * BT_RPM / BT_KMH


def _traction(kmh, fs, pwm, poles, slots, load, seed):
    """`road.ev_whine`, its first carrier line placed where the pulse pattern puts its largest harmonic: 2 fs and 4 fs on SV-PWM (the
    lines k1 = 1, 2 of ev_whine at 2 fs), fs and 2 fs on DPWM. Then the stator's structural modes at 2.3-3.3 kHz (narrow enough that
    SV-PWM's 4 kHz falls under DPWM's 2 kHz above fs 1.8 kHz, as measured), the gear mesh."""
    if pwm not in ("sv", "dpwm"):
        raise ValueError(f"traction pwm {pwm!r}: use sv | dpwm")
    rpm = traction_rpm(kmh)
    carrier = np.clip(np.asarray(fs, float), 500.0, 4000.0) * (2.0 if pwm == "sv" else 1.0)
    y = ev_whine(rpm, poles, slots, carrier, load, P, seed)
    y = 0.3 * y + 2.0 * _band(y, 2300.0, 3300.0)
    shaft = np.cumsum(rpm / 60.0) / P
    mesh = sum(g * np.sin(2.0 * np.pi * k * PINION * shaft) for k, g in ((1, 1.0), (2, 0.4), (3, 0.15)))
    return y + 0.03 * (0.4 + 0.6 * float(load)) * mesh * np.clip(rpm / 60.0 * PINION / 20.0, 0.0, 1.0)


@sfx("traction_drive", "Inverter-fed traction motor of a metro or electric train: slot orders, carrier lines k1 fs +- k2 f0, stator modes at 2-4 kHz, gear mesh. 627 rpm at 15 km/h.",
     speed=(15.0, "km/h"), speed_end=(-1.0, "glide to this km/h (-1 = steady)"), fs=(2000.0, "inverter switching frequency, Hz, 500-4000"),
     pwm=("sv", "sv (largest harmonic at 2 fs, loudest near fs 1500) | dpwm (at fs, loudest near 3000)"), poles=(2, "pole pairs"), slots=(36, "stator slots"),
     load=(0.8, "0 coasting .. 1 full torque"), dur=(3.0, "s"), seed=(0, "variation"))
def traction_drive(sr=DEFAULT_SR, speed=15.0, speed_end=-1.0, fs=2000.0, pwm="sv", poles=2, slots=36, load=0.8, dur=3.0, seed=0):
    n = _n(float(np.clip(dur, 0.05, 60.0)))
    kmh = np.linspace(float(speed), float(speed) if speed_end < 0 else float(speed_end), n)
    return _fin(_traction(np.maximum(kmh, 0.0), fs, pwm, poles, slots, load, seed), sr, None, fade=0.02, tail=0.05)


def metro_fs(n, steps=(500.0, 1000.0, 1500.0, 2000.0)):
    """Switching frequency in equal steps: 500 Hz to 2 kHz in a start, 20 ms to change over."""
    k = np.minimum(np.arange(n) * len(steps) // n, len(steps) - 1)
    return uniform_filter1d(np.asarray(steps)[k], _n(0.02), mode="nearest")


# =============================================================================================
# CURVE SQUEAL
# =============================================================================================
# Third-octave A-weighted dB at R 50-54 m, 19 km/h: total 85 against 69 straight, 2 kHz the loudest at about 82, largest gap 25 dB at 1.6 kHz.
# Calculated from these: 1.6 kHz must sit below 82 and leave 80 for 25 dB of gap; the three other bands share what 85 leaves, 73 each.
POMA = {630.0: 73.0, 800.0: 73.0, 1600.0: 80.0, 2000.0: 82.0, 4000.0: 73.0}
UMBRELLA = {412.0: 80.0, 630.0: 82.0, 850.0: 78.0}          # R 100 m, 412-850 Hz
DUTY = {4000.0: 0.5, 2000.0: 0.7, 1600.0: 0.6, 800.0: 0.4, 630.0: 0.4, 412.0: 0.5, 850.0: 0.4}   # share of the time each wheel squeals
STRAIGHT_DBA = 69.0
# 69 in all. Calculated: for the largest gap to be at 1.6 kHz while 2 kHz is loudest, the floor must dip at 1.6 kHz, to 55; 51 at the node, since the bands interpolate between nodes.
# Above 2 kHz, 50 at the nodes: with 73 at 4 kHz, less would open a gap over 25 dB there (measured: 43 dB with no floor above 2 kHz).
FLOOR_DBA = {f: (51.0 if abs(np.log2(f / 1600.0)) < 1 / 6 else 60.0 if f < 2100.0 else 50.0) for f in _HN_F if 300.0 < f < 4500.0}
MODAL_KG, MODAL_ZETA = 50.0, 1e-3                           # wheel modes, lightly damped
REF_DB = 105.0                                              # dB at digital RMS 1: 85 dB(A) is RMS .1


def _rms_of(db):
    return 10.0 ** ((db - REF_DB) / 20.0)


def attack_angle(radius, wheelbase=2.5):
    """Lateral creep of the leading axle: about wheelbase / 2R, wheelbase as `foley.train`."""
    return wheelbase / (2.0 * max(float(radius), 1.0))


def _wheel_gate(n, duty, rng):
    """0.02..1 of the attack angle: gauge and flange wander take a wheel in and out of saturation, entering over 0.3 s: a step of creep from rest is a blow
    that throws the mode into its limit cycle even on a wide curve (measured: 2196 Hz at R 2000 m)."""
    x = _lfn(rng, n, 2.5)
    th = np.quantile(x, 1.0 - duty)
    return (0.02 + 0.98 * np.clip((x - th) / 0.3 + 0.5, 0.0, 1.0)) * np.minimum(np.arange(n) / _n(0.3), 1.0)


def _third(y, f):
    """RMS in the third-octave around f."""
    s = np.fft.rfft(y)
    fr = np.fft.rfftfreq(len(y), 1.0 / P)
    s[(fr < f / 2 ** (1 / 6)) | (fr > f * 2 ** (1 / 6))] = 0.0
    return float(np.std(np.fft.irfft(s, len(y))))


@sfx("curve_squeal", "Wheel squeal in a tight curve: lateral creep saturates the contact (brush model) and falling friction drives a wheel mode. Below about 200 m it sings, above it only rolls. Bands 630, 800, 1600, 2000, 4000 Hz at R 50 m; umbrella mode 412-850 Hz at R 100 m.",
     radius=(50.0, "m"), speed=(19.0, "km/h, 10..80"), dur=(3.0, "s"), seed=(0, "variation"))
def curve_squeal(sr=DEFAULT_SR, radius=50.0, speed=19.0, dur=3.0, seed=0):
    """# ponytail: one mode per wheel, so modes do not compete for one contact: the bands measured together would otherwise be won by one."""
    rng = np.random.default_rng(int(seed))
    n = _n(float(np.clip(dur, 0.2, 30.0)))
    # Below 10 km/h the explicit stick regime is unstable for 50 kg modes.
    kmh = float(np.clip(speed, 10.0, 80.0))
    V = kmh / 3.6
    psi = attack_angle(radius)
    table = POMA if radius < 75.0 else UMBRELLA             # 75 m between the two measured curves
    y = np.zeros(n)
    for f, db in table.items():
        creep = psi * _wheel_gate(n, DUTY[f], rng) * (1.0 + 0.02 * _lfn(rng, n, 300.0))
        w = np.array([2.0 * math.pi * f])
        # mu .5 falling to .25 over vc .05 m/s.
        v = _brush(np.ascontiguousarray(creep), V / P, PATCH_A, PATCH_B, 1.1 * E_STAR / PATCH_A, WHEEL_LOAD, 0.5, 0.25, 0.05, V,
                   w, np.array([MODAL_ZETA]), np.array([1.0 / MODAL_KG]), 3)[1]
        v = _band(v, f / 1.3, min(f * 1.3, 0.45 * P))
        ref = 0.2 * attack_angle(50.0) * 19.0 / 3.6         # a wheel that does not sing is not raised to the band's level
        y += v * _rms_of(db - float(a_weighting_db(f))) / max(_third(v, f), ref)
    gain = (kmh / 19.0) ** 1.5                              # as the rolling
    floor = {f: FLOOR_DBA.get(f, 30.0) - float(a_weighting_db(f)) for f in _HN_F}
    bed = _third_octave_noise(n, rng, np.array([floor[f] for f in _HN_F]))
    bed = bed * sum(_rms_of(FLOOR_DBA[f]) ** 2 for f in FLOOR_DBA) ** 0.5 / max(_a_rms(bed), 1e-12)
    return _fin((y + bed) * gain, sr, None, fade=0.02, tail=0.05)


def _a_rms(y):
    s = np.fft.rfft(y)
    f = np.maximum(np.fft.rfftfreq(len(y), 1.0 / P), 1.0)
    return float(np.std(np.fft.irfft(s * 10.0 ** (a_weighting_db(f) / 20.0), len(y))))


# =============================================================================================
# THE KINDS
# =============================================================================================
EMD_IDLE, EMD_MAX = 200.0, 900.0                            # rpm; 950 on late variants: pass rpm=950
TURBO_GEAR = 18.0                                           # shaft / crank while gear-driven


def notch_rpm(notch):
    """8 notches, evenly from idle (notch 0) to 900 rpm (notch 8)."""
    return EMD_IDLE + (EMD_MAX - EMD_IDLE) * np.clip(notch, 0.0, 8.0) / 8.0


def _v16(rpm, load, dur, seed):
    """16 firings per turn, evenly, into the one turbocharger, one collector. Turbo geared to the crank, whine a^2 (.22 sin p + .1 sin 2p + .05 a sin 7p)
    as `cyber.turbo`."""
    n = _n(dur)
    y = pipe_engine(sr=P, dur=dur, rpm=rpm, cylinders=16, load=load, cycle=2, offsets=[i / 16.0 for i in range(16)],
                    header=6.0, pipe=25.0, unequal=0.0, mic=0.15, seed=seed)
    p = 2.0 * np.pi * np.cumsum(TURBO_GEAR * np.asarray(rpm, float) / 60.0 * np.ones(n)) / P
    a = np.clip(rpm / EMD_MAX, 0.0, 1.0)
    return y + 0.15 * a * a * (0.22 * np.sin(p) + 0.1 * np.sin(2.0 * p) + 0.05 * a * np.sin(7.0 * p)) / 0.37


def _pantograph(dur, seed, rng, reflect=0.7):
    """Sporadic arcs at the contact wire, source at 4 m, with its ground reflection (.7); listener 10 m out and 1.2 m up."""
    n = _n(dur)
    gate = np.clip((_lfn(rng, n, 3.0) - 0.5) / 0.2 + 0.5, 0.0, 1.0)
    y = arc(sr=P, dur=dur, mains=50.0, stability=0.3, seed=seed)[:n] * uniform_filter1d(gate, _n(0.01))
    k = pantograph_delay()
    return y + reflect * np.concatenate([np.zeros(k), y[:n - k]])


def pantograph_delay(height=4.0, listener=1.2, distance=10.0):
    """Samples between the direct and ground-reflected paths."""
    return int(round((math.hypot(distance, height + listener) - math.hypot(distance, height - listener)) / 343.0 * P))


def _chime(rng):
    """Two falling tones, A5 to F5."""
    y = np.zeros(_n(1.1))
    for t, f in ((0.0, 880.0), (0.45, 698.5)):
        m = _n(0.6)
        tt = np.arange(m) / P
        _put(y, (np.sin(2 * np.pi * f * tt) + 0.2 * np.sin(4 * np.pi * f * tt)) * np.minimum(tt / 0.003, 1.0) * np.exp(-tt / 0.2), t)
    return y


SPEEDS = {"train_welded": 300.0, "train_diesel": 60.0, "train_diesel_4t": 80.0, "train_electric": 60.0, "metro_start": 40.0, "tram_roll": 30.0,
          "curve_squeal": 19.0, "train_brake": 40.0}       # km/h
RAIL_KINDS = ("train_welded", "train_diesel", "train_diesel_4t", "train_electric", "metro_start", "tram_roll", "curve_squeal", "pantograph_arc",
              "tram_bell", "train_door", "train_brake")
TRAM_JOINT = 12.0                                           # m between tram rail joints


@sfx("rail", "Rail vehicles: train_welded (rolling alone), train_diesel (EMD V16 two-stroke by notches), train_diesel_4t (railcar), train_electric, metro_start (inverter steps), tram_roll, curve_squeal, pantograph_arc, tram_bell, train_door, train_brake.",
     kind=("train_welded", " | ".join(RAIL_KINDS)), speed=(-1.0, "km/h, -1 = the kind's; metro_start: end of the start"), rail_len=(-1.0, "m between joints: 0 = welded, -1 = the kind's"),
     corrugation=(0.0, "rolling kinds: amplitude / static contact compression, .3 stays in contact, .55 leaves it at resonance"), wavelength=(0.05, "m of the corrugation (V / wavelength Hz)"),
     notch=(8.0, "train_diesel: 0 idle (200 rpm) .. 8 (900 rpm)"), notch_end=(-1.0, "train_diesel: move through the notches to this one"), rpm=(-1.0, "diesel kinds: crankshaft, -1 = by notch, 1800 on the railcar"),
     fs=(2000.0, "electric kinds: switching frequency, Hz"), pwm=("sv", "sv | dpwm"), radius=(50.0, "curve_squeal: m"), dur=(4.0, "s, where the kind has no length of its own"), seed=(0, "variation"))
def rail(sr=DEFAULT_SR, kind="train_welded", speed=-1.0, rail_len=-1.0, corrugation=0.0, wavelength=0.05, notch=8.0, notch_end=-1.0, rpm=-1.0, fs=2000.0, pwm="sv", radius=50.0, dur=4.0, seed=0):
    k = _kind(kind, RAIL_KINDS, "rail")
    rng = np.random.default_rng(int(seed))
    dur = float(np.clip(dur, 0.2, 60.0))
    n = _n(dur)
    kmh = SPEEDS.get(k, 0.0) if speed < 0 else float(speed)
    roll = lambda band=None: _rolling(kmh, n, rng, band, float(corrugation), max(float(wavelength), 0.005))
    if k == "curve_squeal":
        return curve_squeal(sr=sr, radius=radius, speed=kmh, dur=dur, seed=seed)
    if k == "train_welded":
        y = roll()
        if rail_len > 0:
            y = y + 2.0 * _joints(kmh, max(float(rail_len), 2.0), n, rng)
    elif k == "tram_roll":
        L = TRAM_JOINT if rail_len < 0 else float(rail_len)
        y = roll((250.0, 2000.0))                           # the extension to 3000 Hz in the skirt
        if L > 0:
            y = y + 2.5 * _joints(kmh, max(L, 2.0), n, rng)
    elif k == "train_diesel":
        a = notch_rpm(notch)
        curve = np.full(n, float(a))
        if notch_end >= 0:                                  # one notch at a time over the render, .25 s to settle
            steps = notch_rpm(np.round(np.linspace(notch, notch_end, n)))
            curve = uniform_filter1d(steps, _n(0.25), mode="nearest")
        if rpm > 0:
            curve = np.full(n, float(rpm))
        y = _v16(curve, float(np.clip(curve.mean() / EMD_MAX, 0.2, 1.0)), dur, seed) + 0.02 * roll()
    elif k == "train_diesel_4t":
        # Under the floor, behind skirts: clatter .25 and the 2.5 kHz low-pass. At clatter 1 detect_clicks fires at one and the same phase of every firing (R = 1.000), at 0 never:
        # those are diesel_engine's combustion blows, and they would stand out of a railcar heard from outside.
        y = sosfilt(butter(4, 2500.0, "lowpass", fs=P, output="sos"), _fit(diesel_engine(sr=P, kind="diesel_i6_truck", rpm=1800.0 if rpm < 0 else float(rpm), load=0.6, clatter=0.25, dur=dur, seed=seed), n)) + 0.1 * roll()
    elif k == "train_electric":
        y = _traction(np.full(n, kmh), fs, pwm, 2, 36, 0.8, seed) + 0.05 * roll()
    elif k == "metro_start":
        v = np.linspace(0.0, kmh, n)
        y = _traction(v, metro_fs(n) if fs == 2000.0 else fs, pwm, 2, 36, 0.9, seed) + 0.03 * _rolling(kmh, n, rng) * v / max(kmh, 1.0)
    elif k == "pantograph_arc":
        y = _pantograph(dur, seed, rng)
    elif k == "tram_bell":                                  # foley's bell, lower
        return _fin(FO.bike_bell(sr=P, freq=650.0, strikes=2, gap=0.22, decay=1.4, bright=0.8, seed=seed), sr, 0.7, fade=0.001, tail=0.05)
    elif k == "train_door":                                 # chime, then the ram, as a bus door
        y = np.zeros(_n(3.2))
        _put(y, _chime(rng), 0.0, 0.08)
        _put(y, air_system(sr=P, kind="bus_door", seed=seed), 1.0, 1.0)
    else:                                                   # train_brake: shoes squeal to a stop, then the release
        m = max(n, _n(3.0))
        y = np.zeros(m + _n(1.0))
        _put(y, _fit(brake_squeal(sr=P, speed=3.0, pressure=8.0, dur=m / P, seed=seed), m), 0.0, 0.04)
        _put(y, 0.1 * _rolling(kmh, m, rng) * np.linspace(1.0, 0.0, m) ** 2, 0.0, 1.0)
        _put(y, air_system(sr=P, kind="air_brake_release", seed=seed), m / P, 1.0)
    return _fin(y, sr, 0.7, fade=0.005, tail=0.05)
