"""Sport and glide: inflated balls, snow, ice, the pea whistle, skateboard, swimming and the five struck-ball sports.

The inflated ball: omega = sqrt(2 pi P a / M), contact pi / omega; monopole thump -(rho a v^2 / r) cos 2 omega t; ring of the air inside at ka = 2.082,
  (rho c U1 / (k r D1)), D1 = .531, 14 dB under the thump for a 100 cm drop heard at 3 m, "damp in about a single cycle"; breathing mode "about 900 Hz".
shell modes heavily damped. Rolling resistance of plastic and viscoelastic floors, Hertz contact times. The snow squeak is `physical.footsteps`'.
Sources of every value: research/SOURCES.md, section sports. Nothing has been listened to: see tests/test_sports.py.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter

from . import contact as K
from . import foley as FO
from . import matter as M
from .core import DEFAULT_SR
from .foley import P, _fin, _n, _put
from .matter import _lfn
from .physical import footsteps as FS
from .road import _band, _fit, _kind
from .sfx import sfx

G = 9.81
RHO, C_AIR = 1.2, 343.0                                     # air at 20 C
HEAR_AT = 3.0                                               # m
KA = (2.082, 3.342)                                         # l = 1 and l = 2, first roots of j_l'(ka) = 0
D1 = 0.531                                                  # radiation factor at ka = 2.082
PATCH_U1 = 0.0125 / 0.01028                                 # the stated U1 of 1.25 cm/s against the 1.03 its rms formula gives: the ring 14 dB under the thump

# M kg, a m (inside), P gauge Pa, restitution: inside the ranges of each sport's rules.
BALLS = {"basket": dict(M=0.60, a=0.114, P=5.5e4, e=0.80),
         "tennis": dict(M=0.057, a=0.030, P=9.0e4, e=0.73),
         "foot": dict(M=0.43, a=0.106, P=8.0e4, e=0.78),
         "volley": dict(M=0.27, a=0.103, P=3.0e4, e=0.72)}
BALL_MODES = {"basket": ((410.0, 0.3), (564.0, 0.15), (616.0, 0.15), (900.0, 0.2)),   # Hz, level against the thump
              "volley": ((150.0, 1.0),)}                    # breathing mode. Foot and tennis: none.
BALL_MODE_TAU = {410.0: 0.002, 564.0: 0.002, 616.0: 0.002, 900.0: 0.002, 150.0: 0.015}   # s: about a cycle, heavily damped
# surface: restitution factor, floor level against the thump, body, rolling. A soft floor: P/kR = .73, mu_R .076.
SURFACES = {"parquet": dict(e=1.0, floor=0.5, body=("wood", 1.5), soft=0.0), "concrete": dict(e=1.0, floor=0.08, body=("concrete", 3.0), soft=0.0),
            "clay": dict(e=0.85, floor=0.0, body=None, soft=0.45), "turf": dict(e=0.7, floor=0.0, body=None, soft=0.73)}


# =============================================================================================
# LAWS
# =============================================================================================
def ball_omega(mass, a, pressure):
    """The flattened cap is a spring of area 2 pi a x at gauge pressure P."""
    return math.sqrt(2.0 * math.pi * pressure * a / mass)


def cavity_modes(a, c=C_AIR):
    """Hz of the first two air modes in a sphere of inner radius a."""
    return tuple(ka * c / (2.0 * math.pi * a) for ka in KA)


def thump_pa(a, v, r=HEAR_AT):
    """Monopole thump, Pa at r."""
    return RHO * a * v * v / r


def ring_pa(mass, a, pressure, v, r=HEAR_AT):
    """Ring of the air mode, Pa at r, with U1 the air's share of the force's Fourier transform at the mode, ball's recoil U1 = P_w1 / M."""
    w, w1 = ball_omega(mass, a, pressure), 2.0 * math.pi * cavity_modes(a)[0]
    U1 = PATCH_U1 * 2.0 * math.sqrt(2.0) * math.pi * a * pressure * (v / w) * w / (w1 * w1 - w * w) / mass
    return RHO * C_AIR * U1 / (KA[0] / a * r * D1)


def plastic_mu(p_kr):
    """Free rolling on a rigid-plastic floor: M0 = .104 P^2 / k, so mu_R = .104 P / kR."""
    return 0.104 * p_kr


def viscoelastic_mu(contact_a, radius):
    """Viscoelastic rolling at the loss peak: mu_R = .195 a / R. That peak falls where omega = 1.83 V / a."""
    return 0.195 * contact_a / radius


def strain_freq(speed, contact_a):
    """Hz at which a rolling contact strains its rubber: omega = 1.83 V / a."""
    return 1.83 * speed / (2.0 * math.pi * contact_a)


def ball_patch(mass, a, pressure):
    """Half-width of the cap of a ball at rest: Mg = P 2 pi a x, patch sqrt(2 a x) (geometry, calculated)."""
    return math.sqrt(2.0 * a * mass * G / (2.0 * math.pi * a * pressure))


def rolling_time(speed, mu):
    """A rolling solid sphere slows at (5/7) mu_R g (calculated, I = .4 m R^2)."""
    return speed / (5.0 / 7.0 * max(mu, 1e-4) * G)


def hertz_time(m1, m2, radius1, radius2, e_star, speed):
    """Hertz impact: Tc = 2.87 (m^2 / (R E*^2 V))^(1/5) with the relative mass and radius."""
    m, r = m1 * m2 / (m1 + m2), radius1 * radius2 / (radius1 + radius2)
    return 2.87 * (m * m / (r * e_star * e_star * max(speed, 1e-3))) ** 0.2


# =============================================================================================
# BUILDING BLOCKS
# =============================================================================================
def _half_sine(T):
    m = max(int(round(T * P)), 2)
    return np.sin(np.pi * (np.arange(m) + 0.5) / m)


def _hertz_pulse(T):
    """Force of a Hertz contact, sin^1.5 over T (F as delta^1.5, delta about sinusoidal). It starts with no slope, unlike the linear spring of a ball."""
    return _half_sine(T) ** 1.5


def _soft_ends(y, head, tail):
    """Raised-cosine ends, in s: a buffer cut while a mode still rings, or a roll started at speed, would be a step. Zero for the last 2 ms."""
    y = np.array(y, float)
    z = min(_n(0.002), len(y) // 4)
    a, b = min(_n(head), len(y) // 2), min(_n(tail), (len(y) - z) // 2)
    y[:a] *= 0.5 - 0.5 * np.cos(np.pi * (np.arange(a) + 0.5) / a)
    y[len(y) - z - b:len(y) - z] *= 0.5 + 0.5 * np.cos(np.pi * (np.arange(b) + 0.5) / b)
    y[len(y) - z:] = 0.0
    return y


def _finish(y, sr, level, head, tail):
    """`_fin` with its fade at nothing, then a last cosine. `_fin` high-passes at 12 Hz, so a signal already at zero comes out on a slow, smooth residue; the corner of its linear fade
    on that residue fires detect_clicks 1 ms from the end (measured, moved with the fade's length, and stayed when the input was zeroed)."""
    return _soft_ends(_fin(_soft_ends(y, head, tail), sr, level, fade=0.0), 0.0, 0.01)


def _disperse(x, rng, ms=10.0):
    """A slip is a burst of grains, not one sample: an all-pass of random phase over ``ms`` keeps the magnitude of the squeak spectrum and spreads each pulse."""
    m = _n(ms / 1000.0)
    ph = np.exp(2j * np.pi * rng.random(m // 2 + 1))
    ph[0] = ph[-1] = 1.0
    h = np.fft.irfft(ph, m) * np.hanning(m)
    return lfilter(h / np.sqrt(np.sum(h * h)), [1.0], x)


def _free_mode(f, tau, n, T):
    """A mode kept up by the contact of T seconds, then dying in tau, unit peak. Its level is the force's Fourier integral, computed apart, so the slow push of the force must not count
    (a resonator driven by the force is mostly that push, measured: the mode 35 dB under its peak). So only the envelope follows the force: force * exp(-t / tau). A ring at full level from the
    first sample fires detect_clicks at every contact (measured, within .25 ms of it)."""
    h = _half_sine(T)
    env = lfilter([1.0], [1.0, -math.exp(-1.0 / (tau * P))], np.concatenate([h, np.zeros(max(n - len(h), 0))]))
    y = np.sin(2.0 * np.pi * f * np.arange(len(env)) / P) * env
    return y / (np.max(np.abs(y)) + 1e-15)


def _flatten(x):
    """Pulses at 350-1100 Hz have a hole under their rate: without this, 500 Hz lies 7 dB under the squeak curve (measured). Third-octave whitening, so that the spectrum imposed next is the one heard."""
    s = np.fft.rfft(x)
    f = np.maximum(np.fft.rfftfreq(len(x), 1.0 / P), 1.0)
    c = FS.ARABADZHI_BANDS
    p = np.array([np.sqrt(np.mean(np.abs(s[(f >= b / 2 ** (1 / 6)) & (f < b * 2 ** (1 / 6))]) ** 2) + 1e-30) if np.any((f >= b / 2 ** (1 / 6)) & (f < b * 2 ** (1 / 6))) else 1.0 for b in c])
    return np.fft.irfft(s / np.interp(np.log2(f), np.log2(c), p), len(x))


def _thump_shape(w):
    """-cos 2 omega t over the contact. The pulse steps at both ends, where the contact is taken as instantaneous: 1 ms of raised cosine rounds them."""
    m = max(int(round(math.pi / w * P)), 8)
    t = (np.arange(m) + 0.5) / P
    k = min(_n(0.001), m // 4)
    edge = np.ones(m)
    edge[:k] = 0.5 - 0.5 * np.cos(np.pi * (np.arange(k) + 0.5) / k)
    edge[-k:] = edge[:k][::-1]
    return -np.cos(2.0 * w * t) * edge


def _body(material, size, shape="rect_plate", damping=6.0, seed=0, position=0.37):
    return K._make_body(shape, material, size, position, 0.41, modes=24, damping=damping, seed=seed, fmax=12000.0)


def _hit(T, material, size, shape="rect_plate", damping=6.0, seed=0, tail=0.5, position=0.37, bright=0.5):
    """A contact of T seconds on a modal body, unit peak. The half-sine's own spectrum, falling above 1 / T, is what limits the brightness."""
    f = np.zeros(_n(tail))
    h = _hertz_pulse(T)
    f[:len(h)] = h
    y = K.modal_bank(f, _body(material, size, shape, damping, seed, position), P, bright)[0]
    return _soft_ends(y, 0.0001, 0.3 * tail) / (np.max(np.abs(y)) + 1e-15)


# =============================================================================================
# BALL_BOUNCE
# =============================================================================================
def _ball_params(kind, radius):
    b = BALLS[kind]
    return b["M"], (float(radius) if radius > 0 else b["a"]), b["P"], b["e"]


def _ball_train(kind, height, surface, cavity, ring_cycles, radius, seed):
    """Bounce train in Pa at HEAR_AT, with the first impact 10 ms in. Each contact: thump as v^2, ring and shell as v: small bounces ring relatively more."""
    mass, a, pressure, e = _ball_params(kind, radius)
    s = SURFACES[surface]
    w = ball_omega(mass, a, pressure)
    T = math.pi / w
    e = e * s["e"]
    v0 = math.sqrt(2.0 * G * height)
    t, v = K.bounce_schedule(2.0 * e * v0 / G, v0, e, t_min=2.0 * T, seed=seed)
    n = _n(float(t[-1]) + T + 0.5)
    y = np.zeros(n)
    contact = _half_sine(T)
    thump = _thump_shape(w)
    m = _n(T + 0.2)                                         # the slowest mode, 15 ms, has gone by 13 tau
    f1, f2 = cavity_modes(a)
    tau = max(float(ring_cycles), 0.1) / f1                 # damped in about a single cycle
    # l = 2 at half l = 1
    ring = _free_mode(f1, tau, m, T) + 0.5 * _free_mode(f2, tau * f1 / f2, m, T)
    ring = ring / np.max(np.abs(ring))
    shell = sum((g * _free_mode(f, BALL_MODE_TAU[f], m, T) for f, g in BALL_MODES.get(kind, ())), np.zeros(m))
    force = np.zeros(n)
    for ti, vi in zip(t, v):
        at = float(ti) + 0.01
        _put(y, thump, at, thump_pa(a, vi))
        _put(y, ring, at, float(cavity) * ring_pa(mass, a, pressure, vi))
        _put(y, shell, at, float(cavity) * thump_pa(a, v0) * vi / v0)
        _put(force, contact, at, vi)
    if s["body"] is not None and s["floor"] > 0:
        floor = K.modal_bank(force, _body(s["body"][0], s["body"][1], seed=seed), P, 0.4)[0]
        y = y + floor / (np.max(np.abs(floor)) + 1e-15) * s["floor"] * thump_pa(a, v0)
    return y, t + 0.01


@sfx("ball_bounce", "Inflated ball bouncing to rest: a low thump (one cycle over the contact, -cos 2wt) and the ring of the air inside (ka = 2.082). Basket 82 + 997 Hz, 12 ms; tennis, foot, volley (breathing mode 150 Hz).",
     kind=("basket", "basket | tennis | foot | volley"), height=(1.0, "m of the drop"), surface=("parquet", " | ".join(SURFACES)),
     cavity=(1.0, "level of the air and shell modes: 0 = thump alone, 1 = as computed"), ring_cycles=(1.0, "cycles to decay: 1 bouncing, more for a ball held away from the floor"),
     radius=(0.0, "m inside the ball, 0 = the kind's; .116 is a measured basketball"), roll_speed=(0.0, "m/s of rolling after the bounces, slowed by the rolling resistance"), seed=(0, "variation"))
def ball_bounce(sr=DEFAULT_SR, kind="basket", height=1.0, surface="parquet", cavity=1.0, ring_cycles=1.0, radius=0.0, roll_speed=0.0, seed=0):
    k = _kind(kind, tuple(BALLS), "ball_bounce")
    surf = _kind(surface, tuple(SURFACES), "ball_bounce surface")
    y, t = _ball_train(k, float(np.clip(height, 0.02, 10.0)), surf, cavity, ring_cycles, radius, seed)
    if roll_speed > 0:
        mass, a, pressure, _ = _ball_params(k, radius)
        soft = SURFACES[surf]["soft"]
        mu = plastic_mu(soft) if soft else viscoelastic_mu(ball_patch(mass, a, pressure), a)
        d = min(rolling_time(float(roll_speed), mu), 8.0)
        # ponytail: `contact.roll` weighs its ball as solid. A hollow ball would need its own interactor; its level here is set by hand.
        r = K.roll(sr=P, dur=d, speed=[float(roll_speed), 0.0], radius=a, material="rubber", surface="wood" if not soft else "rubber", roughness=0.3, seed=seed)
        z = np.zeros(int(t[-1] * P) + _n(0.05) + len(r))
        z[:min(len(y), len(z))] = y[:len(z)]
        _put(z, _soft_ends(r, 0.05, 0.25 * d), t[-1] + 0.03, 0.1 * np.max(np.abs(y)))   # eased in: it starts at speed; eased out: as v goes to 0 `contact.roll` ends in separate ticks
        y = z
    return _finish(y, sr, 0.85, 0.002, 0.2)


# =============================================================================================
# PEA WHISTLE
# =============================================================================================
PEA_F0 = 3400.0                                             # Hz, middle of 3.3-3.5 kHz
PEA_FM, PEA_AM, PEA_NOISE = 0.07, 2.0, 1.26                 # pea in the jet: pitch -7 %, tone x2 (+6 dB), noise +2 dB
PEA_HARM = (1.0, 0.316, 0.056)                              # 0, -10, -25 dB
PEA_AIR = 0.1                                               # noise -20 dB under harmonic 1


def _pea(rate, n, rng):
    """0..1 as the pea blocks: a rounded square wave at ``rate``, 5 % drift of the orbit."""
    if rate <= 0:
        return np.zeros(n)
    phase = 2.0 * np.pi * np.cumsum(rate * (1.0 + 0.05 * _lfn(rng, n, 3.0))) / P
    return 0.5 + 0.5 * np.tanh(3.0 * np.sin(phase))


@sfx("pea_whistle", "Referee's pea whistle: 3.3-3.5 kHz, the pea crossing the jet drops it 7 % and makes it 6 dB louder, with noise around the harmonics.",
     rate=(20.0, "Hz of the pea's orbit, 0 = no pea"), blasts=(1, "blasts"), dur=(0.5, "s per blast"), gap=(0.25, "s between blasts"), freq=(PEA_F0, "Hz"), seed=(0, "variation"))
def pea_whistle(sr=DEFAULT_SR, rate=20.0, blasts=1, dur=0.5, gap=0.25, freq=PEA_F0, seed=0):
    rng = np.random.default_rng(int(seed))
    blasts = int(np.clip(blasts, 1, 8))
    d = float(np.clip(dur, 0.08, 6.0))
    y = np.zeros(_n(blasts * d + (blasts - 1) * float(gap) + 0.1))
    n = _n(d)
    ramp = np.minimum(np.arange(n) / _n(0.015), 1.0) * np.minimum((n - 1 - np.arange(n)) / _n(0.03), 1.0)   # breath's attack and release
    for b in range(blasts):
        m = _pea(float(rate), n, rng)
        ph = 2.0 * np.pi * np.cumsum(float(freq) * (1.0 - PEA_FM * m)) / P
        tone = sum(g * np.sin(h * ph) for h, g in zip((1, 2, 3), PEA_HARM))
        air = sum(g * _band(rng.standard_normal(n), h * freq * 0.97, min(h * freq * 1.03, 0.45 * P)) for h, g in ((1, 0.25), (2, 0.1), (3, 0.05)))   # narrow noise around the harmonics
        air *= PEA_AIR * np.sqrt(0.5) / (float(np.sqrt(np.mean(air ** 2))) + 1e-12)
        _put(y, (tone * (1.0 + (PEA_AM - 1.0) * m) + air * (1.0 + (PEA_NOISE - 1.0) * m)) * ramp, 0.02 + b * (d + float(gap)))
    return _fin(y, sr, 0.7, fade=0.002)


# =============================================================================================
# SNOW
# =============================================================================================
SNOW_KINDS = ("ski", "luge", "boule", "chute")


def _squeak(fn, temperature, rng):
    """Footsteps' snow squeak out of the step: stick-slip pulses while the load rises, on the measured squeak spectrum, gone at 0 C. Unit RMS at -1.5 C and below."""
    n = len(fn)
    if temperature >= 0.0:
        return np.zeros(n)
    pulses = FS._stick_slip_pulses(n, fn, np.gradient(fn), 0.2, 350.0, 1100.0, 0.30, int(rng.integers(0, 2 ** 31 - 1)))   # rates as footsteps; .2 under load that never leaves the snow
    y = FS.snow_squeak_shape(_flatten(_flatten(_disperse(pulses, rng))), temperature)   # twice: the band's mean hides the hole inside it, one pass leaves 4 dB (measured)
    return y / (np.std(y) + 1e-12) * min(-temperature / 1.5, 1.0)


def _slide(n, rng, lo, hi):
    y = _band(rng.standard_normal(n), lo, hi)
    return y / (np.std(y) + 1e-12)


@sfx("snow", "Snow under load, the squeak of footsteps taken out of the step: ski (carved turns), luge (runners and bumps), boule (packing a snowball), chute (a body into powder). Colder is brighter.",
     kind=("ski", " | ".join(SNOW_KINDS)), temperature=(-8.0, "C: squeak spectrum, none at 0"), speed=(1.0, "pace scale"), dur=(3.0, "s, ski and luge"), seed=(0, "variation"))
def snow(sr=DEFAULT_SR, kind="ski", temperature=-8.0, speed=1.0, dur=3.0, seed=0):
    k = _kind(kind, SNOW_KINDS, "snow")
    rng = np.random.default_rng(int(seed))
    T = float(np.clip(temperature, -40.0, 10.0))
    sp = float(np.clip(speed, 0.2, 4.0))
    d = float(np.clip(dur, 0.3, 30.0))
    if k == "ski":
        n = _n(d)
        t = np.arange(n) / P
        turn = np.sin(np.pi * t * sp / TURN_S) ** 2
        tex = 0.5 + 0.5 * np.sin(2.0 * np.pi * 37.0 * t + 4.0 * _lfn(rng, n, 2.0))
        y = 0.45 * _squeak(turn * (0.55 + 0.45 * tex), T, rng) + turn * _slide(n, rng, 500.0, 7000.0)
    elif k == "luge":
        n = _n(d)
        t = np.arange(n) / P
        bumps = np.clip(_lfn(rng, n, 6.0), 0.0, None)
        tex = 0.5 + 0.5 * np.sin(2.0 * np.pi * 53.0 * t)
        y = 0.3 * _squeak((0.45 + 0.35 * bumps / (bumps.max() + 1e-9)) * (0.7 + 0.3 * tex), T, rng) + 0.6 * _slide(n, rng, 900.0, 5000.0) + (0.5 + 2.0 * bumps) * _slide(n, rng, 40.0, 250.0)
    elif k == "boule":
        n = _n(1.5 / sp + 0.3)
        y = np.zeros(n)
        for i in range(3):                                  # three squeezes, each packed harder
            m = _n(0.35 / sp)
            a = np.sin(0.5 * np.pi * np.arange(m) / m) ** 2 * (0.6 + 0.2 * i)
            press = np.concatenate([a, a[-1] * (1.0 - np.arange(m // 3) / (m // 3))])
            wob = 0.8 + 0.2 * np.sin(2.0 * np.pi * 45.0 * np.arange(len(press)) / P)
            _put(y, _soft_ends(0.8 * _squeak(press * wob, T, rng) + 0.7 * press * _slide(len(press), rng, 600.0, 4000.0), 0.01, 0.02), 0.05 + i * 0.5 / sp)   # the FFT of the squeak wraps round its buffer
    else:
        n = _n(1.6)
        y = np.zeros(n)
        thud = FO.thud(sr=P, body="sack", contact_ms=30.0, size=1.6, force=0.9, seed=seed)
        _put(y, thud, 0.03, 1.6 / (np.max(np.abs(thud)) + 1e-12))
        press = np.exp(-np.arange(n) / _n(0.2)) * np.minimum(np.arange(n) / _n(0.04), 1.0)
        y = y + 0.8 * _squeak(press * (0.8 + 0.2 * np.sin(2.0 * np.pi * 31.0 * np.arange(n) / P)), T, rng) + np.exp(-np.arange(n) / _n(0.35)) * np.minimum(np.arange(n) / _n(0.02), 1.0) * 0.5 * _slide(n, rng, 800.0, 8000.0)
    return _finish(y, sr, 0.75, 0.01, 0.1)


TURN_S = 1.4                                                # s of a ski turn


# =============================================================================================
# ICE SKATE
# =============================================================================================
@sfx("ice_skate", "Skate blade on ice: glide (a push every stroke) or stop (hockey stop, spray of chips). A scrape on an ice body; thin plate cracks optional.",
     stroke=("glide", "glide | stop"), rate=(1.2, "glide: strokes per second"), dur=(3.0, "s, glide"), crack=(0.0, "0..1 level of a crack in the sheet"), seed=(0, "variation"))
def ice_skate(sr=DEFAULT_SR, stroke="glide", rate=1.2, dur=3.0, crack=0.0, seed=0):
    k = _kind(stroke, ("glide", "stop"), "ice_skate")
    rng = np.random.default_rng(int(seed))
    if k == "glide":
        r = float(np.clip(rate, 0.3, 4.0))
        y = np.zeros(_n(float(np.clip(dur, 0.5, 30.0))))
        s = 1.0 / r
        for i in range(int(len(y) / P * r) + 1):
            a = K.scrape(sr=P, dur=0.8 * s, speed=[0.5, 3.0, 2.0], force=[0.05, 1.0, 0.5, 0.0], texture="metal", grain=0.03, material="ice", size=1.2, seed=int(seed) + i)
            _put(y, _soft_ends(a, 0.01, 0.05), 0.02 + i * s, rng.uniform(0.8, 1.0))
    else:
        a = K.scrape(sr=P, dur=0.8, speed=[6.0, 4.0, 0.2], force=[0.1, 1.0, 0.8, 0.0], texture="gritty", grain=0.05, material="ice", size=1.2, seed=seed)
        n = len(a)
        chips = _slide(n, rng, 3000.0, 10000.0) * np.sin(np.pi * (np.arange(n) + 0.5) / n) ** 2
        y = np.max(np.abs(a)) * 0.6 * chips / (np.max(np.abs(chips)) + 1e-12) + a
    if crack > 0:
        c = M.ice_crack(sr=P, dur=1.0, thickness=0.03, cracks=1, seed=seed)
        z = np.zeros(max(len(y), len(c) + _n(0.1)))
        z[:len(y)] = y
        _put(z, c, 0.1, float(crack) * np.max(np.abs(y)))
        y = z
    return _finish(y, sr, 0.7, 0.01, 0.05)


# =============================================================================================
# SKATEBOARD
# =============================================================================================
WHEELBASE = 0.35                                            # m
WHEEL_R = 0.027                                             # m
SKATE_ROLL_RMS = 0.04
SKATE_SURFACES = {"concrete": ("concrete", 0.45, 2.0), "asphalt": ("stone", 0.85, 0.0), "wood": ("wood", 0.25, 1.2)}   # body, roughness, joint spacing m
OLLIE_H = 0.3                                               # m


def ollie_air(height=OLLIE_H):
    """s in the air: 2 sqrt(2 h / g)."""
    return 2.0 * math.sqrt(2.0 * height / G)


@sfx("skateboard", "Skateboard rolling: hard wheels on the ground, both axles clacking over each joint (L / v, then wheelbase / v). ollie = pop of the tail, silence in the air, landing.",
     speed=(4.0, "m/s"), surface=("concrete", " | ".join(SKATE_SURFACES)), joints=(-1.0, "m between joints, -1 = the surface's, 0 = none"), ollie=(0, "1 = an ollie in the middle"), dur=(4.0, "s"), seed=(0, "variation"))
def skateboard(sr=DEFAULT_SR, speed=4.0, surface="concrete", joints=-1.0, ollie=0, dur=4.0, seed=0):
    material, rough, spacing = SKATE_SURFACES[_kind(surface, tuple(SKATE_SURFACES), "skateboard")]
    rng = np.random.default_rng(int(seed))
    V = float(np.clip(speed, 0.3, 15.0))
    d = float(np.clip(dur, 0.5, 30.0))
    r = K.roll(sr=P, dur=d, speed=V, radius=WHEEL_R, roughness=rough, material="plastic", surface=material, surface_size=2.0, asymmetry=0.1, seed=seed)
    n = len(r)
    r = SKATE_ROLL_RMS * r / (np.sqrt(np.mean(r * r)) + 1e-12)
    L = spacing if joints < 0 else float(joints)
    knock = _hit(0.003, "wood", 0.8, damping=30.0, seed=seed, tail=0.1)    # deck, 3 ms
    j = np.zeros(n)
    if L > 0:
        for i in range(int(d * V / L) + 2):
            t = (i * L + 0.1) / V
            _put(j, knock, t, rng.uniform(0.85, 1.0))
            _put(j, knock, t + WHEELBASE / V, 0.9 * rng.uniform(0.85, 1.0))
    y = r + 0.6 * j
    if ollie:
        t0 = 0.45 * d
        air = ollie_air()
        gate = np.ones(n)
        a, b = int(t0 * P), min(int((t0 + air) * P), n)
        gate[a:b] = 0.0
        k = _n(0.015)
        gate = np.convolve(gate, np.hanning(k) / np.hanning(k).sum(), mode="same")   # a boxcar's ramp has corners: detect_clicks fired 7 ms before the landing (seeds 4, 10, 11, 12)
        gate[:k] = 1.0
        gate[-k:] = 1.0
        y = y * gate
        _put(y, _hit(0.002, "wood", 0.8, damping=15.0, seed=seed + 1, tail=0.25), t0, 1.0)              # tail on the ground
        _put(y, _hit(0.006, "wood", 0.8, damping=10.0, seed=seed + 2, tail=0.3), t0 + air, 1.2)         # four wheels land
    return _finish(y, sr, 0.75, 0.01, 0.05)


# =============================================================================================
# SWIM
# =============================================================================================
STROKES = {"crawl": (1.2, 2, 6), "breast": (1.6, 1, 0), "butterfly": (1.4, 1, 2), "back": (1.3, 2, 6)}   # s per arm cycle, arm entries per cycle, kicks per cycle


def swim_rates(stroke, rate=1.0):
    """(arm entries per second, kicks per second): six-beat flutter kick in crawl and back, two kicks in butterfly, per arm cycle."""
    cycle, arms, kicks = STROKES[stroke]
    return arms * rate / cycle, kicks * rate / cycle


@sfx("swim", "Swimming: arm entries as small splashes, kick beats, a breath's gurgle, over water lapping. crawl, breast, butterfly, back.",
     stroke=("crawl", " | ".join(STROKES)), rate=(1.0, "tempo scale"), dur=(6.0, "s"), seed=(0, "variation"))
def swim(sr=DEFAULT_SR, stroke="crawl", rate=1.0, dur=6.0, seed=0):
    k = _kind(stroke, tuple(STROKES), "swim")
    rng = np.random.default_rng(int(seed))
    d = float(np.clip(dur, 1.0, 30.0))
    r = float(np.clip(rate, 0.3, 3.0))
    cycle, arms, kicks = STROKES[k]
    y = np.zeros(_n(d))
    wet = lambda z: _soft_ends(_band(z, 200.0, 6000.0, order=4), 0.002, 0.05)    # `matter.splash`: 300 Hz - 6 kHz by its own table. Unfiltered its bubbles start with clicks
    takes = [wet(M.splash(sr=P, size=0.15 + 0.1 * (k == "butterfly"), dur=0.5, droplets=2, seed=int(seed) + i)) for i in range(4)]
    period = cycle / r / arms
    for i in range(int(d / period) + 1):
        _put(y, takes[i % 4], 0.05 + i * period, rng.uniform(0.8, 1.0))
    kr = swim_rates(k, r)[1]
    if kr:
        kick = wet(M.splash(sr=P, size=0.02, dur=0.2, droplets=0, seed=int(seed) + 7))
        for i in range(int(d * kr) + 1):
            _put(y, kick, 0.03 + i / kr, 0.2 * rng.uniform(0.7, 1.0))
    breath = wet(M.gurgle(sr=P, dur=0.4, rate=8.0, size=0.3, seed=seed))
    for i in range(int(d / (2.0 * cycle / r)) + 1):
        _put(y, breath, 0.5 * cycle / r + 2.0 * i * cycle / r, 0.15)
    bed = _slide(len(y), rng, 150.0, 1500.0) * (0.5 + 0.5 * np.abs(_lfn(rng, len(y), 0.7)))
    return _finish(y + 0.02 * bed, sr, 0.75, 0.02, 0.1)


# =============================================================================================
# SPORT
# =============================================================================================
SPORT_KINDS = ("tennis", "golf", "bowling", "billard", "batte")
SPORT_SPEEDS = {"tennis": 25.0, "golf": 40.0, "bowling": 8.0, "billard": 2.0, "batte": 30.0}   # m/s
LANE = 18.3                                                 # m, foul line to head pin
E_RESIN, NU_RESIN = 3.2e9, 0.35                             # as `contact`'s plastic
E_STAR_RESIN = E_RESIN / (2.0 * (1.0 - NU_RESIN ** 2))


def hertz_patch(mass, radius, e1, nu1, e2, nu2):
    """m: radius of the Hertz contact of a sphere resting on a flat, a = (3 m g R / 4 E*)^(1/3)."""
    e_star = 1.0 / ((1.0 - nu1 * nu1) / e1 + (1.0 - nu2 * nu2) / e2)
    return (3.0 * mass * G * radius / (4.0 * e_star)) ** (1.0 / 3.0)


BOWL = dict(m=6.8, R=0.108)                                 # kg, m
BOWL["patch"] = hertz_patch(BOWL["m"], BOWL["R"], E_RESIN, NU_RESIN, K.MATERIALS["wood"].young, K.MATERIALS["wood"].poisson)
PIN = dict(m=1.5, R=0.06)                                   # kg, m
BILLIARD = dict(m=0.17, R=0.0286)                           # kg, m
BAT = {"wood": ("wood", 0.85, 0.001), "aluminium": ("aluminium", 0.75, 0.001)}   # body, m, contact s. A hollow aluminium bat rings on many modes.


def _swing_with_hit(blade, speed, hit, seed, gain=2.0, top=6000.0):
    """The swing's Aeolian tones (`foley.sword`) and, at its loudest, the hit. `modal.sword_swing` steps where its trapezoid of speed bends (end of the lead, apex, near silence on both sides):
    low-passed at ``top``, above the 5 f of the thickest blade at these speeds, and gated below 1 % of its peak (inaudible). Measured in tests/test_sports.py."""
    s = _band(FO.sword(sr=P, blade=blade, speed=speed, seed=seed), 30.0, top, order=4)
    e = np.sqrt(np.convolve(s * s, np.ones(_n(0.01)) / _n(0.01), mode="same"))
    on = np.flatnonzero(e > 0.01 * e.max())
    g = np.zeros(len(s))
    g[on[0]:on[-1] + 1] = 1.0
    g = np.convolve(g, np.hanning(_n(0.02)) / np.hanning(_n(0.02)).sum(), mode="same")
    s = s * g
    k = _n(0.02)
    t = float(np.argmax(np.convolve(s * s, np.ones(k), mode="same"))) / P
    y = np.zeros(max(len(s), int(t * P) + len(hit)))
    y[:len(s)] = s
    _put(y, hit, t, gain * np.max(np.abs(s)))
    return y, t


def _tennis(speed, seed):
    a = BALLS["tennis"]
    w = ball_omega(a["M"], a["a"], a["P"])
    v = speed
    m = _n(0.15)
    # Ball on the string bed: the ball's own contact and cavity, with the bed's modes for a floor (nylon as plastic), bed 25 cm.
    ball = np.zeros(m)
    _put(ball, _thump_shape(w), 0.0, thump_pa(a["a"], v))
    _put(ball, _free_mode(cavity_modes(a["a"])[0], 1.0 / cavity_modes(a["a"])[0], m, math.pi / w), 0.0, ring_pa(a["M"], a["a"], a["P"], v))
    bed = _hit(math.pi / w, "plastic", 0.25, shape="rect_membrane", damping=25.0, seed=seed, tail=0.15)
    hit = ball / (np.max(np.abs(ball)) + 1e-12) + 0.7 * _fit(bed, m)
    return _swing_with_hit("bat", 0.6, hit, seed, 3.0, top=1000.0)[0]


def _golf(speed, seed):
    # Driver: hollow steel head rings; contact .45 ms, head .09 m.
    hit = _hit(0.00045, "steel", 0.09, shape="circ_plate", damping=40.0, seed=seed, tail=0.2, position=0.5, bright=0.7)
    return _swing_with_hit("golf_club", min(speed / 36.0, 2.0), hit, seed)[0]


def _batte(speed, bat, seed):
    material, size, T = BAT[_kind(bat, tuple(BAT), "sport bat")]
    hit = _hit(T, material, size, shape="bar", damping=25.0 if material == "wood" else 8.0, seed=seed, tail=0.4, position=0.15)
    return _swing_with_hit("bat", min(speed / 36.0, 2.0), hit, seed, top=2000.0)[0]


def bowling_roll(speed):
    """(seconds down the lane, speed at the pins): mu_R = .195 a / R, slowing at (5/7) mu g."""
    mu = viscoelastic_mu(BOWL["patch"], BOWL["R"])
    dv = 5.0 / 7.0 * mu * G
    end = math.sqrt(max(speed * speed - 2.0 * dv * LANE, 0.25))
    return (speed - end) / dv, end


def _bowling(speed, rng, seed):
    t_roll, end = bowling_roll(speed)
    tail = 1.2
    y = np.zeros(_n(t_roll + tail))
    r = K.roll(sr=P, dur=t_roll, speed=[speed, end], radius=BOWL["R"], roughness=0.25, material="plastic", surface="wood", surface_size=1.0, seed=seed)
    k = len(r)
    r = 0.25 * _soft_ends(r, 0.02, 0.02) / (np.max(np.abs(r)) + 1e-12)
    _put(y, r, 0.0)
    release = _hit(0.006, "wood", 1.0, damping=20.0, seed=seed, tail=0.2)
    _put(y, release, 0.01, 0.4)
    # Ball on the head pin (Hertz time), then ten pins in a chain and on the lane.
    T = hertz_time(BOWL["m"], PIN["m"], BOWL["R"], PIN["R"], E_STAR_RESIN, end)
    times = [t_roll]
    for i in range(9):
        times.append(t_roll + rng.exponential(0.09) + 0.015 * (i + 1))
    for i, t in enumerate(sorted(times)):
        pin = _hit(max(T, 0.001), "wood", 0.38, shape="bar", damping=12.0, seed=int(seed) + i, tail=0.35, position=0.2 + 0.05 * i)
        _put(y, pin, t, 1.0 if i == 0 else rng.uniform(0.35, 0.8))
    for i in range(6):                                      # falling on the lane
        fall = _hit(0.004, "wood", 1.0, damping=30.0, seed=int(seed) + 20 + i, tail=0.15)
        _put(y, fall, t_roll + rng.uniform(0.2, 0.75), rng.uniform(0.1, 0.25))
    return y


def _billard(speed, rng, seed):
    y = np.zeros(_n(2.2))
    tip = _hit(0.002, "plastic", 0.06, shape="circ_plate", damping=60.0, seed=seed, tail=0.1, position=0.5)   # leather tip 2 ms
    _put(y, tip, 0.02, 0.5)
    t1 = 0.02 + 1.0 / max(speed, 0.2)                       # 1 m of cloth
    T = hertz_time(BILLIARD["m"], BILLIARD["m"], BILLIARD["R"], BILLIARD["R"], E_STAR_RESIN, speed)
    click = _hit(T, "plastic", 0.057, shape="circ_plate", damping=50.0, seed=seed + 1, tail=0.1, position=0.5, bright=0.8)
    if t1 < 2.0:
        _put(y, click, t1, min(1.0, 0.45 + 0.2 * speed))
    r = K.roll(sr=P, dur=2.0, speed=[speed, speed * 0.7, 0.0], radius=BILLIARD["R"], roughness=0.15, material="plastic", surface="wood", surface_size=1.2, seed=seed)
    _put(y, _soft_ends(r, 0.02, 0.05), 0.03, 0.03 / (np.max(np.abs(r)) + 1e-12))
    cushion = _hit(0.005, "rubber", 0.5, damping=30.0, seed=seed + 2, tail=0.1)
    _put(y, cushion, t1 + 0.6, 0.25)
    return y


@sfx("sport", "Struck balls: tennis (ball on strings, swing), golf (driver), bowling (roll down the lane, then the pins), billard (tip, ball on ball by Hertz time, cushion), batte (bat, wood or aluminium).",
     kind=("tennis", " | ".join(SPORT_KINDS)), speed=(-1.0, "m/s, -1 = the kind's"), bat=("wood", "batte: wood | aluminium"), seed=(0, "variation"))
def sport(sr=DEFAULT_SR, kind="tennis", speed=-1.0, bat="wood", seed=0):
    k = _kind(kind, SPORT_KINDS, "sport")
    rng = np.random.default_rng(int(seed))
    V = SPORT_SPEEDS[k] if speed < 0 else float(np.clip(speed, 0.2, 80.0))
    if k == "tennis":
        y = _tennis(V, seed)
    elif k == "golf":
        y = _golf(V, seed)
    elif k == "bowling":
        y = _bowling(V, rng, seed)
    elif k == "billard":
        y = _billard(V, rng, seed)
    else:
        y = _batte(V, bat, seed)
    return _finish(y, sr, 0.8, 0.005, 0.05)
