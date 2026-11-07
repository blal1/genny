"""Gore, mud, slime, blood, chemistry: everything wet and thick, on one new generator, `viscous_bubble`.

A bubble in mud is assembled from pieces (sources in research/SOURCES.md, section `viscous.py`):
- Radial mode: `matter.bubble_params` with the viscous term 4 mu / (rho w0 r^2) at the liquid's own mu and rho.
- Burst: an N-shaped wave, then a jet wave; fundamental falls linearly with bubble size.
- Film: amplitude follows the film's rupture time.
- Helmholtz: f = (c / 2 pi) sqrt(A / (V L')), L' = h + 1.43 neck, hemispheric cavity; rings with the radiation Q of its opening.
- Shape modes: as `ambience.tube_bubble`, decay (n+2)(2n+1) mu / (rho r^2), at the liquid's mu.
- Sinking: a^3 = (3R / 16 eta) P0 t: the length of the "floc".
- Bone: body is the stick of `foley.OBJECTS`.
See tests/test_viscous.py for what was measured.
"""
from __future__ import annotations

import math

import numpy as np

from . import foley as FO
from . import matter as Ma
from .core import DEFAULT_SR
from .foley import P, _fin, _n, _put
from .matter import _lfn
from .road import _band, _kind
from .sfx import sfx

RHO_MUD, SIGMA_MUD = 1500.0, 0.05                           # kg/m3, N/m
MU_MUD, TAU_MUD = 30.0, 100.0                               # Pa s, Pa, of the liquid round a bubble
ETA_GROUND = 5000.0                                         # Pa s, of the ground under a load (6.66)
RHO_BLOOD, MU_BLOOD, SIGMA_BLOOD = 1060.0, 3.5e-3, 0.058    # handbook values, here
MU_SLIME, MU_FLESH, MU_FAT = 2.0, 8.0, 0.05                 # Pa s
STRAIN = 0.05                                               # wall strain of the oscillation, for the Bingham rate
FILM = 0.001                                                # m, film thickness
C_AIR = 343.0


def _smooth(n, m):
    """1 after a raised-cosine rise of m samples, at least 1.5 ms (a wave starting at full slope is a click), and a raised-cosine end over the last fifth (a tail cut short is another)."""
    e = np.ones(n)
    m = min(max(int(m), _n(0.0015)), n // 2 + 1)
    e[:m] = 0.5 - 0.5 * np.cos(np.pi * np.arange(m) / m)
    k = max(n // 5, 2)
    e[-k:] *= 0.5 + 0.5 * np.cos(np.pi * (np.arange(k) + 1) / k)
    return e


def _end(y, s=0.03):
    """A buffer's end on a raised cosine: a bubble cut by the end of its buffer stops dead, and detect_clicks found exactly that (at each _pull's and _cloud's last sample)."""
    k = min(_n(s), len(y))
    y = y.copy()
    y[-k:] *= 0.5 + 0.5 * np.cos(np.pi * (np.arange(k) + 1) / k)
    return y


def _out(y, sr, level):
    """`_fin`, its linear fade replaced: over a slow tail the kink where that fade starts is itself a click (measured at its first sample, 670 ms of 700)."""
    y = _fin(y, sr, level, fade=0.002)
    m, k = int(0.004 * sr), int(0.05 * sr)                      # after _fin, exact zero over its own 2 ms fade: its kink fired too, on the high-pass's residue, 2 ms from the end
    y[-m:] = 0.0
    y[-m - k:-m] *= 0.5 + 0.5 * np.cos(np.pi * (np.arange(k) + 1) / k)
    return y


# =============================================================================================
# THE GENERATOR
# =============================================================================================
def effective_viscosity(r, mu, tau_y, rho=RHO_MUD, sigma=SIGMA_MUD):
    """Bingham: mu + tau_y / rate, rate = STRAIN w0. A yield stress is what makes the bubble viscoplastic [spec]. Chosen: the Bingham form as used here."""
    w0 = math.sqrt(max(3.0 * Ma.GAMMA * Ma.P_ATM - 2.0 * sigma / r, 1.0)) / (r * math.sqrt(rho))
    return float(mu) + float(tau_y) / (STRAIN * w0)


def bubble_modes(r, mu, rho=RHO_MUD, sigma=SIGMA_MUD):
    """(w0, beta, w_d, delta) of HF eq. 36-38 with this liquid's viscous term. # ponytail: thermal and radiative terms stay water's, as no source gives them for mud."""
    w0 = math.sqrt(max(3.0 * Ma.GAMMA * Ma.P_ATM - 2.0 * sigma / r, 1.0)) / (r * math.sqrt(rho))
    w, _, _, d = Ma.bubble_params(r)
    d = float(d) - 4.0 * Ma.MU_W / (Ma.RHO_W * float(w) * r * r) + 4.0 * float(mu) / (rho * w0 * r * r)
    beta = w0 * d / math.sqrt(d * d + 4.0)
    return w0, beta, math.sqrt(w0 * w0 - beta * beta), d


def burst_freq(r):
    """Fundamental of the burst: falls linearly with size [CPB]. Chosen: 900 Hz and 25 kHz/m, the source's abstract has no figures; r held in 2-30 mm where the line stays positive."""
    return 900.0 - 25000.0 * float(np.clip(r, 0.002, 0.03))


def rupture_time(r, mu):
    """s for the film to open. Chosen: grows with r and mu as a viscous retraction r mu / sigma does, far less steeply (that law gives seconds); 1.5 ms at r 10 mm in a thin liquid."""
    return float(np.clip(0.0015 * (r / 0.01) * math.sqrt(1.0 + mu / 10.0), 0.0005, 0.03))


def helmholtz_freq(r, neck, h=FILM):
    """Air cavity of a hemisphere of radius r opening through a neck."""
    neck = np.asarray(neck, float)
    return C_AIR / (2.0 * np.pi) * np.sqrt(np.pi * neck ** 2 / ((2.0 / 3.0) * np.pi * r ** 3 * (h + 1.43 * neck)))


def _radial(r, mu, rho, sigma):
    """Ringing of the bubble. Radiated pressure goes as wall acceleration, so (w_d / w0)^2 of the inviscid level (textbook, here). r / 10 mm as `tube_bubble`'s r * 1e3."""
    w0, beta, wd, _ = bubble_modes(r, mu, rho, sigma)
    n = _n(min(max(7.0 / beta, 2.0 * math.pi / wd), 0.5))       # at least a period: at mu 400, 7 / beta ends before the first zero
    t = np.arange(n) / P
    return (wd / w0) ** 2 * (r / 0.01) * np.sin(wd * t) * np.exp(-beta * t) * _smooth(n, 0.5 * math.pi / wd * P)


def _shape(r, mu, rho, sigma, rng):
    """Moss zonal modes radiating at 2 f_n, n = 2..5. Viscosity kills them: past critical damping they do not ring at all, which is why thin liquids sound richer. 5 of the radial level, as tube_bubble."""
    n = _n(0.25)
    t = np.arange(n) / P
    y, w_b = np.zeros(n), bubble_modes(r, mu, rho, sigma)[0]
    weights = []
    for k in range(2, 6):
        wn = math.sqrt((k - 1) * (k + 1) * (k + 2) * sigma / (rho * r ** 3))
        b = (k + 2) * (2 * k + 1) * mu / (rho * r * r)
        if b >= wn or 2.0 * wn >= 2.0 * np.pi * 0.45 * P:
            continue
        w = (k - 1) * (k + 2) * (4 * k - 1) / (2 * k + 1) * wn ** 2 / math.sqrt((4 * wn ** 2 - w_b ** 2) ** 2 + (4 * b * wn) ** 2)
        weights.append((w, wn, b))
    top = max((w for w, _, _ in weights), default=1.0)
    for w, wn, b in weights:
        y += w / top * np.sin(2.0 * wn * t + rng.uniform(0.0, 2.0 * np.pi)) * np.exp(-max(b, 8.0) * t)
    return 0.5 * (r / 0.01) * y * _smooth(n, 0.002 * P)


def _thin(r, mu, rho=RHO_MUD, sigma=SIGMA_MUD):
    """w_d / w0. The film's sounds take the radial mode's fall too: [CPL] measures the fall in the bubble's sound as a whole, and without this a thick render came out brighter than a thin one (measured: centroid 942 against 329 Hz), film over a dead ring. Chosen: the same factor for the film."""
    m = bubble_modes(r, mu, rho, sigma)
    return m[2] / m[0]


def _burst(r, mu, rho=RHO_MUD, sigma=SIGMA_MUD):
    """The N-shaped wave, one period at the fundamental, then the jet wave at the same fundamental [CPB]. Amplitude falls as the film takes longer [Vidal]. Chosen: jet's 4 periods and .5 level, the 5 ms."""
    th = _thin(r, mu, rho, sigma)
    f = burst_freq(r) * th
    per = _n(1.0 / f)
    u = np.arange(per) / per
    nw = np.sin(2.0 * np.pi * u) * np.sin(np.pi * u) ** 2
    jn = 3 * per + _n(4.0 / f)
    t = np.arange(jn) / P
    jet = 0.5 * np.sin(2.0 * np.pi * f * t) * np.exp(-t * f / 1.5) * _smooth(jn, per / 4)
    y = np.zeros(per + jn)
    y[:per] = nw
    y[per:] += jet
    return th ** 2 * (r / 0.01) * y / (1.0 + rupture_time(r, mu) / 0.005)


def neck_end(mu):
    """Fraction of r the neck reaches: a thick film does not open wide. """
    return 0.1 + 0.5 / (1.0 + mu / 10.0)


def radiation_q(depth, aperture):
    """Q of a cavity of this depth ringing through an aperture of this diameter, damped by radiation at the open end."""
    return 4.0 * depth ** 2 / (np.pi * aperture ** 2)


def _helmholtz(r, mu, q=None, rho=RHO_MUD, sigma=SIGMA_MUD):
    """The cavity rings through the opening film: the neck grows from .1 r to `neck_end` over the rupture time, so the pitch climbs. Held under 10 kHz.
    Q: radiation at the final opening, the cavity's depth r."""
    tr = rupture_time(r, mu)
    end = neck_end(mu)
    q = radiation_q(r, 2.0 * end * r) if q is None else q
    f1 = min(float(helmholtz_freq(r, end * r)), 1e4)
    n = _n(tr + max(8.0 * q / (np.pi * f1), 0.012))
    t = np.arange(n) / P
    f = np.minimum(helmholtz_freq(r, r * (0.1 + (end - 0.1) * np.minimum(t / tr, 1.0))), 1e4)
    decay = np.exp(-t / np.maximum(q / (np.pi * f), 0.002))    # at least 2 ms: at 10 kHz, Q 3 is gone in .1 ms, inside its own rise, and that blip is a click
    # The slower film excites the cavity less, as it does the burst [Vidal]: without this the Helmholtz line, 1.8 kHz at r 8 mm mu 300, outlives the dead ring and a thick render reads brighter than a thin one.
    return 0.4 * _thin(r, mu, rho, sigma) ** 2 * (r / 0.01) * np.sin(2.0 * np.pi * np.cumsum(f) / P) * decay * _smooth(n, 0.25 * P / f[0]) / (1.0 + tr / 0.005)


def viscous_bubble(r, mu=MU_MUD, rng=None, tau_y=0.0, rho=RHO_MUD, sigma=SIGMA_MUD, burst=1.0):
    """One bubble of radius r m in a liquid of viscosity mu Pa s at 48 kHz: rings, reaches the surface, the film opens (Helmholtz), it bursts (N wave, jet).
    Chosen: the burst two damping times after the ringing starts, at most 120 ms."""
    rng = np.random.default_rng(0) if rng is None else rng
    r = float(np.clip(r, 5e-4, 0.08))
    m = effective_viscosity(r, mu, tau_y, rho, sigma)
    beta = bubble_modes(r, m, rho, sigma)[1]
    tb = min(2.0 / beta, 0.12)
    parts = (_radial(r, m, rho, sigma), _shape(r, m, rho, sigma, rng), _burst(r, m, rho, sigma), _helmholtz(r, m, rho=rho, sigma=sigma))
    y = np.zeros(_n(tb) + max(map(len, parts)))
    _put(y, parts[0], 0.0)
    _put(y, parts[1], 0.0)
    if burst > 0:
        _put(y, parts[3], tb, float(burst))
        _put(y, parts[2], tb, float(burst))
    return y


def _cloud(n, rng, rate, lo, hi, mu, cluster=0.3, **kw):
    """Viscous bubbles placed by `matter.bubble_times`, at the very least one. # ponytail: python loop per bubble, 400 a second at most; a compiled bank if a texture wants thousands."""
    y = np.zeros(n)
    st = Ma.bubble_times(n, rng, rate, cluster)
    if not len(st):
        st = np.array([int(rng.uniform(0.0, 0.5 * n))])
    for s, r in zip(st, Ma._radii(rng, len(st), lo, hi)):
        _put(y, viscous_bubble(r, mu, rng, **kw), s / P, rng.uniform(0.6, 1.0))
    return _end(y)


# =============================================================================================
# SINKING: JOHNSON 6.66
# =============================================================================================
def sink_radius(t, load, R, eta=ETA_GROUND):
    """Contact radius of a sphere R under a step load in a purely viscous material, a^3 = (3R / 16 eta) P0 t (6.66)."""
    return (3.0 * R / (16.0 * eta) * load * np.asarray(t, float)) ** (1.0 / 3.0)


def sink_time(a, load, R, eta=ETA_GROUND):
    """6.66 turned round: seconds to reach contact radius a. The length of the floc."""
    return 16.0 * eta * a ** 3 / (3.0 * R * load)


def _floc(T, rng, lo=40.0, hi=500.0):
    """Mass displaced while the contact grows: the level rises as a(t) ~ t^(1/3) for T seconds (6.66), then lets go. Chosen: band, the release .25 T + 10 ms; 1 ms of rise, as t^(1/3) has an infinite slope at 0."""
    T = float(np.clip(T, 0.003, 3.0))
    tail = 0.25 * T + 0.01
    n = _n(T + 6.0 * tail)
    t = (np.arange(n) + 0.5) / P
    env = np.where(t < T, (t / T) ** (1.0 / 3.0), np.exp(-(t - T) / tail)) * _smooth(n, 0.001 * P)
    x = _band(rng.standard_normal(n + _n(0.05)), lo, hi)[-n:] * env   # the filter settles first
    return x / (np.max(np.abs(x)) + 1e-12)


def _unit(y):
    return y / (np.max(np.abs(y)) + 1e-12)


def _pull(rng, r_hole, T, mu):
    """Suction: air forces into the hole a foot leaves: noise through a Helmholtz cavity whose neck widens, then the hole's bubbles. Chosen: neck .08 to .6 of the hole, levels, 20 bubbles a second."""
    n = _n(T + 0.35)
    u = np.minimum(np.arange(n) / _n(T), 1.0)
    f = helmholtz_freq(r_hole, r_hole * (0.08 + 0.52 * u ** 2))
    env = np.sin(np.pi * u) ** 2
    air = Ma._tvbp(rng.standard_normal(n), np.clip(f, 40.0, 4000.0), 6.0) * env
    rate = np.where(np.arange(n) > _n(0.5 * T), 20.0, 0.0)
    return 0.8 * _unit(air) + 0.5 * _unit(_cloud(n, rng, rate, 0.006, 0.02, mu, tau_y=TAU_MUD))


# =============================================================================================
# THE SOUNDS
# =============================================================================================
MUD_KINDS = ("bulles", "marche", "aspiration", "impact")
FOOT_LOAD, FOOT_R, FOOT_A = 800.0, 0.15, 0.05               # N, m, m, a foot
BODY_LOAD, BODY_R, BODY_A = 2000.0, 0.1, 0.1                # N, m, m, a body falling in


@sfx("mud", "Mud: bulles (viscoplastic bubbles: ring, film, Helmholtz, burst), marche (steps sinking, floc length from viscosity and load, then the suck out), aspiration (a foot pulled free), impact (a mass falling in).",
     kind=("bulles", " | ".join(MUD_KINDS)), dur=(4.0, "s, bulles"), rate=(5.0, "bulles: bubbles a second"), size=(15.0, "bulles: mean bubble radius, mm"),
     mu=(MU_MUD, "Pa s of the liquid round a bubble: higher = lower and quieter"), tau_y=(TAU_MUD, "Pa, yield stress"), eta=(ETA_GROUND, "Pa s of the ground: the floc lasts in proportion"),
     load=(-1.0, "N on the ground, -1 = the kind's"), steps=(4, "marche"), pace=(0.9, "marche: s between steps"), pull=(0.35, "aspiration, marche: s the foot takes to come out"), seed=(0, "variation"))
def mud(sr=DEFAULT_SR, kind="bulles", dur=4.0, rate=5.0, size=15.0, mu=MU_MUD, tau_y=TAU_MUD, eta=ETA_GROUND, load=-1.0, steps=4, pace=0.9, pull=0.35, seed=0):
    k = _kind(kind, MUD_KINDS, "mud")
    rng = np.random.default_rng(int(seed))
    mu, eta, pull = max(float(mu), 1e-3), max(float(eta), 1.0), float(np.clip(pull, 0.05, 2.0))
    if k == "bulles":
        n = _n(float(np.clip(dur, 0.2, 60.0)))
        r = float(np.clip(size, 2.0, 60.0)) * 1e-3
        y = _cloud(n, rng, float(np.clip(rate, 0.1, 200.0)), r / 1.6, r * 1.6, mu, tau_y=max(float(tau_y), 0.0))
    elif k == "marche":
        L = FOOT_LOAD if load <= 0 else float(load)
        y = np.zeros(_n(max(int(steps), 1) * max(float(pace), 0.2) + pull + 1.0))
        for i in range(max(int(np.clip(steps, 1, 40)), 1)):
            t = max(i + rng.uniform(-0.04, 0.04), 0.0) * max(float(pace), 0.2)
            T = sink_time(FOOT_A, L, FOOT_R, eta)
            _put(y, _floc(T, rng), t, rng.uniform(0.85, 1.0))
            _put(y, _pull(rng, FOOT_A, pull, mu), t + T + 0.2, 0.5)        # 0.2 s standing
    elif k == "aspiration":
        y = _pull(rng, FOOT_A, pull, mu)
    else:
        L = BODY_LOAD if load <= 0 else float(load)
        T = sink_time(BODY_A, L, BODY_R, eta)
        y = np.zeros(_n(T + 1.2))
        _put(y, _floc(T, rng, 30.0, 350.0), 0.0)
        _put(y, _cloud(_n(0.9), rng, 25.0, 0.008, 0.03, mu, tau_y=max(float(tau_y), 0.0)), T, 0.3)
    return _out(y, sr, 0.8)


GORE_KINDS = ("splat", "squish", "ooze", "deboitement")


def _soft_hit(contact_ms, size, seed):
    """`foley.thud` on the soft body: a long contact on a big mass."""
    return FO.thud(sr=P, body="sack", contact_ms=contact_ms, size=size, force=1.0, seed=seed)


@sfx("gore", "Wet flesh, on viscous_bubble and soft blows: splat (a mass hits, spatter falls back), squish (pressing out bubbles), ooze (slow, thick seep), deboitement (a joint pops out).",
     kind=("splat", " | ".join(GORE_KINDS)), size=(1.0, "size of the mass, bigger = lower"), mu=(MU_FLESH, "Pa s of the wet part"), dur=(-1.0, "s, -1 = the kind's"), seed=(0, "variation"))
def gore(sr=DEFAULT_SR, kind="splat", size=1.0, mu=MU_FLESH, dur=-1.0, seed=0):
    """Chosen: every number below. The spec gives only the recipe (bubbles and soft blows), its foley reference [MOR] and the 10-10^4 Hz of [SUS]."""
    k = _kind(kind, GORE_KINDS, "gore")
    rng = np.random.default_rng(int(seed))
    s, mu = float(np.clip(size, 0.3, 3.0)), max(float(mu), 1e-3)
    d = lambda v: float(np.clip(v if dur < 0 else dur, 0.1, 30.0))
    if k == "splat":
        n = _n(d(1.0))
        y = np.zeros(n)
        _put(y, _soft_hit(25.0, s, seed), 0.0, 0.9)
        _put(y, _cloud(n, rng, 90.0 * np.exp(-np.arange(n) / _n(0.08)), 0.002 * s, 0.012 * s, mu, cluster=0.5), 0.003, 0.6)
        for _ in range(rng.integers(4, 8)):                              # ballistic, 2 sqrt(2h/g), as `matter.splash`; heights
            _put(y, viscous_bubble(rng.uniform(0.0015, 0.005), mu, rng), 2.0 * math.sqrt(2.0 * rng.uniform(0.02, 0.4) / Ma.G_GRAV), rng.uniform(0.08, 0.2))
    elif k == "squish":
        n = _n(d(1.2))
        u = np.arange(n) / n
        press = np.sin(np.pi * u) ** 2                                  # one squeeze
        y = _cloud(n, rng, 60.0 * press + 0.01, 0.003 * s, 0.014 * s, mu, cluster=0.6)
        y = _unit(y) + 0.5 * press * _unit(_band(rng.standard_normal(n), 80.0, 500.0))
    elif k == "ooze":
        n = _n(d(4.0))
        y = _cloud(n, rng, 2.5, 0.006 * s, 0.02 * s, 3.0 * mu, cluster=0.2, burst=0.6)
        y = _unit(y) + 0.1 * _unit(_band(rng.standard_normal(n), 60.0, 300.0)) * np.clip(0.6 + 0.4 * _lfn(rng, n, 0.7), 0.0, 1.0)
    else:
        # The joint: the head slides under tension (a dull creep up), lets go in one low pop with its blow, soft tissue settles wet.
        n = _n(d(0.9))
        y = np.zeros(n)
        c = _n(0.2)
        _put(y, 0.25 * np.linspace(0.0, 1.0, c) ** 2 * _smooth(c, 2) * _band(rng.standard_normal(c), 100.0, 700.0), 0.0)
        _put(y, viscous_bubble(0.012 * s, 0.5, rng, burst=2.0), 0.2, 1.0)
        _put(y, _soft_hit(15.0, 0.8 * s, seed), 0.205, 0.5)
        _put(y, _cloud(_n(0.5), rng, 18.0, 0.002, 0.008, mu), 0.28, 0.2)
    return _out(y, sr, 0.8)


# ---------------------------------------------------------------------------------------------
BONE_KINDS = ("craquement", "fracture", "os_casse_humide")
BONE_SIZE = 3.0                                             # short body: the stick's modes .. 3 times up


def _lp10k(y):
    """Up to 10 kHz [EGU: rock and wood, not bone; the spec knows no band for bone]. The stick's modes, 3 times up, had 19 % of their energy above."""
    return Ma._lp(y, 10000.0, 4)


def _crack(rng, g=1.0, size=1.0):
    """One brittle rupture: a sub-millisecond blow on the stick, its modes damped 1.5 times. Chosen: both. Measured on the blow alone, 6 seeds: no click at .5 ms for damping 1-3, 18 at .3 ms damping 3.
    No noise over the blow: a 4 ms burst of it was what fired detect_clicks in the runs."""
    table, fs, ds = FO.OBJECTS["stick"]
    y = FO._strike(table, fs * BONE_SIZE / max(size, 0.05), 0.5, rng.random(), rng, velocity=0.5 + 0.5 * g, d_scale=ds * 1.5, max_dur=0.25)
    return g * y / (np.max(np.abs(y)) + 1e-12)


def _train(n, rng, t0, count, gap, size, fall=0.8):
    """Ruptures close together: each next one sooner and weaker, as the crack runs out. Chosen: the law."""
    y, t = np.zeros(n), t0
    g = 1.0
    for _ in range(count):
        _put(y, _crack(rng, g, size), t)
        t += gap * rng.uniform(0.5, 1.3)
        gap *= 0.9
        g *= fall * rng.uniform(0.85, 1.1)
    return y


@sfx("bone", "Bone breaking, brittle pulses close together on a short stiff body: craquement (a run of small cracks), fracture (one snap and its run), os casse humide (a fracture in wet flesh).",
     kind=("fracture", " | ".join(BONE_KINDS)), size=(1.0, "bigger = lower"), seed=(0, "variation"))
def bone(sr=DEFAULT_SR, kind="fracture", size=1.0, seed=0):
    k = _kind(kind, BONE_KINDS, "bone")
    rng = np.random.default_rng(int(seed))
    s = float(np.clip(size, 0.3, 3.0))
    if k == "craquement":
        n = _n(0.8)
        y = _train(n, rng, 0.005, 14, 0.04, s, 0.93)
    elif k == "fracture":
        n = _n(0.7)
        y = _train(n, rng, 0.005, 6, 0.012, s)
    else:
        n = _n(1.2)
        y = _train(n, rng, 0.005, 6, 0.012, s)
        wet = _cloud(n, rng, 60.0 * np.exp(-np.arange(n) / _n(0.12)), 0.002, 0.008, MU_FLESH, cluster=0.5)
        y = y + 0.7 * _unit(wet) + 0.4 * _unit(Ma._fit(_soft_hit(15.0, s, seed), n))
    return _out(_lp10k(y), sr, 0.8)


# ---------------------------------------------------------------------------------------------
STRAND_CENTRE = (900.0, 3200.0)                             # Hz the strand climbs through


def _strand(rng):
    """A string of slime drawn out: thins, its pitch climbs, it snaps into a tiny bubble [spec: "cordes"]. Chosen: all of it."""
    T = rng.uniform(0.12, 0.3)
    n = _n(T)
    u = np.arange(n) / n
    f = STRAND_CENTRE[0] * (STRAND_CENTRE[1] / STRAND_CENTRE[0]) ** u
    y = np.zeros(n + _n(0.1))
    y[:n] = np.sin(2.0 * np.pi * np.cumsum(f) / P) * np.sin(np.pi * u ** 0.6) ** 2 * 0.5
    _put(y, viscous_bubble(rng.uniform(0.001, 0.002), MU_SLIME, rng), T - 0.002, 0.5)
    return _end(y)


@sfx("slime", "Slime: as mud, thinner and higher, with strings of it drawn out and snapping.",
     dur=(3.0, "s"), rate=(14.0, "bubbles a second"), strands=(4.0, "strings a second"), mu=(MU_SLIME, "Pa s"), seed=(0, "variation"))
def slime(sr=DEFAULT_SR, dur=3.0, rate=14.0, strands=4.0, mu=MU_SLIME, seed=0):
    """Model `mud`, higher frequencies, strands (spec): bubbles of 2-8 mm against mud's 9-24 mm."""
    rng = np.random.default_rng(int(seed))
    n = _n(float(np.clip(dur, 0.2, 60.0)))
    y = _unit(_cloud(n, rng, float(np.clip(rate, 0.1, 200.0)), 0.002, 0.008, max(float(mu), 1e-3), rho=1100.0, sigma=0.04))
    s = np.zeros(n)
    for i in Ma.bubble_times(n, rng, float(np.clip(strands, 0.0, 40.0)) + 1e-6):
        _put(s, _strand(rng), i / P)
    return _out(y + 0.3 * s, sr, 0.8)


# ---------------------------------------------------------------------------------------------
BLOOD_KINDS = ("goutte", "flot")
BLOOD_DRIP = 1.0                                            # drops a second, against `drops`' 2 for water [spec: slower drops, ]
PULSE = 1.2                                                 # Hz, a heart at 72 a minute


@sfx("blood", "Blood: goutte (slow thick drops falling into a pool), flot (a stream pulsing with the heart).",
     kind=("goutte", " | ".join(BLOOD_KINDS)), dur=(4.0, "s"), rate=(-1.0, "goutte: drops a second; flot: pulses a second; -1 = the kind's"), seed=(0, "variation"))
def blood(sr=DEFAULT_SR, kind="goutte", dur=4.0, rate=-1.0, seed=0):
    k = _kind(kind, BLOOD_KINDS, "blood")
    rng = np.random.default_rng(int(seed))
    dur = float(np.clip(dur, 0.2, 60.0))
    n = _n(dur)
    kw = dict(rho=RHO_BLOOD, sigma=SIGMA_BLOOD)
    if k == "goutte":
        y = np.zeros(n)
        for t in Ma.drop_times(dur, rng, BLOOD_DRIP if rate <= 0 else float(np.clip(rate, 0.05, 40.0))):
            for j in range(rng.integers(2, 5)):                         # fewer than water's 4-5 [HF]
                _put(y, viscous_bubble(rng.uniform(0.0012, 0.004), MU_BLOOD, rng, **kw), t + 0.012 * j * rng.uniform(0.6, 1.4), rng.uniform(0.5, 1.0))
    else:
        f = PULSE if rate <= 0 else float(np.clip(rate, 0.2, 5.0))
        t = np.arange(n) / P
        beat = np.maximum(np.sin(2.0 * np.pi * f * t - 0.5 * np.pi), 0.0) ** 2   # systole; 0 between
        y = _unit(_cloud(n, rng, 180.0 * beat + 1.0, 0.0012, 0.006, MU_BLOOD, cluster=0.2, **kw))
        y = y + 0.25 * beat * _unit(_band(rng.standard_normal(n), 300.0, 2500.0))
    return _out(y, sr, 0.75)


# ---------------------------------------------------------------------------------------------
CHEM_KINDS = ("acide", "reaction")


def _fry(n, rng, intensity=0.7, env=1.0, hiss=0.5, lp=9000.0):
    """`matter.sizzle`'s bed, same 4000 micro-bursts/s at 1, same 2.5 and 6.5 kHz bands, same hiss, its model there as here. Two changes, both because it fails detect_clicks:
    measured on sizzle itself, 495-549 clicks in 3 s with or without spits and hiss. Its bursts are single samples entering a band-pass, so each starts as a step; outliers of its log-normal sigma .9 stand out of the stream.
    Here each burst is a 15-sample Hann (0.3 ms) and sigma is .5. No spits (`_crackles`, noise starting at full level): bubbles do the spitting."""
    k = int(rng.poisson(4000.0 * intensity * n / P))
    imp = np.zeros(n)
    # Sizes capped at 2: the 1 in 40 outlier still fired (peak 7.8 times the RMS at the click, burn_flesh seed 7; 6 clicks in 16 other seeds).
    np.add.at(imp, rng.integers(0, n, k), np.minimum(rng.lognormal(0.0, 0.5, k), 2.0) * rng.choice([-1.0, 1.0], k))
    imp = np.convolve(imp, np.hanning(17)[1:-1], mode="same")   # 9 points: 12-22 clicks in 3 s, 13: 0-4, 17: none, seeds 0-2. Cost: the Hann's null falls near 6.5 kHz, so that band is weakened
    y = _unit(Ma._bp(imp, 2500.0, 0.8) + Ma._bp(imp, 6500.0, 1.0))
    h = Ma._hp(rng.standard_normal(n), 1000.0) * np.clip(0.6 + 0.3 * _lfn(rng, n, 1.0), 0.05, 1.0)   # sizzle's hiss is a power 4 of slow noise: quiet, then bursts. Here gentler
    y = y * env + hiss * _unit(h) * (0.3 + 0.7 * env)
    s = 3.0 * np.std(y)                                         # bursts that coincide: width and cap above no longer change the count (3 seeds in 40 at any of them, peak 5.8 RMS). A soft ceiling at 3 RMS
    return Ma._lp(s * np.tanh(y / s), lp, 2)


@sfx("chem", "Chemistry: acide (eating into a surface, a hiss with fine fizz that dies away), reaction (a bath that gathers pace and foams up).",
     kind=("acide", " | ".join(CHEM_KINDS)), dur=(4.0, "s"), seed=(0, "variation"))
def chem(sr=DEFAULT_SR, kind="acide", dur=4.0, seed=0):
    """`_fry` for `sizzle`, then presets of `fizz`, `boil` [spec]. Chosen: all their settings."""
    k = _kind(kind, CHEM_KINDS, "chem")
    rng = np.random.default_rng(int(seed))
    dur = float(np.clip(dur, 0.2, 60.0))
    n = _n(dur)
    fit = lambda v: Ma._fit(v, n)
    if k == "acide":
        y = _unit(_fry(n, rng, 0.5, np.exp(-np.arange(n) / (0.5 * n)), 0.7, 11000.0))
        # A steady fizz turned down, not fizz's own half_life: as its bubbles thin out each stands alone and fires detect_clicks (5 in 3 s at half_life .9).
        y = y + 0.5 * 0.5 ** (np.arange(n) / (0.3 * n)) * _unit(fit(Ma.fizz(sr=P, dur=dur, rate=2500.0, half_life=0.0, size=0.45, bright=0.7, seed=seed)))
    else:
        y = _unit(fit(Ma.boil(sr=P, dur=dur, intensity=1.0, ramp=1.0, size=0.2, seed=seed)))
        rise = (np.arange(n) / n) ** 2
        y = y + 0.8 * rise * _unit(fit(Ma.fizz(sr=P, dur=dur, rate=2500.0, half_life=0.0, size=0.6, bright=0.5, seed=seed)))
    return _out(y, sr, 0.75)


@sfx("burn_flesh", "Flesh burning: a sizzle with slow thick bubbles of fat under it.",
     dur=(3.0, "s"), rate=(6.0, "fat bubbles a second"), seed=(0, "variation"))
def burn_flesh(sr=DEFAULT_SR, dur=3.0, rate=6.0, seed=0):
    """`_fry` for `sizzle` + slow bubbles (spec). Chosen: their radii 3-10 mm, viscosity, the balance."""
    rng = np.random.default_rng(int(seed))
    dur = float(np.clip(dur, 0.2, 60.0))
    n = _n(dur)
    y = _unit(_fry(n, rng))
    b = _cloud(n, rng, float(np.clip(rate, 0.1, 80.0)), 0.003, 0.01, MU_FAT, rho=920.0, sigma=0.03)
    return _out(y + 0.65 * _unit(b), sr, 0.75)
