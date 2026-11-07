"""Nature, disasters, fireworks: lot L16 of reports/sons/Plan sons.md, spec in reports/sons/taxonomie_universelle.md.

Fireworks levels measured at 102 m, a tornado's infrasound, singing sand locking into step, rain on a window, a storm crossing the scene.
The source of every constant is in research/SOURCES.md, section nature. Nothing here has been listened to: tests/test_nature.py says what was measured.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter, resample_poly, sosfilt

from . import contact as K
from . import creatures as C
from . import foley as FO
from . import geo as G
from . import matter as Ma
from .core import DEFAULT_SR
from .fauna import _fit, _kind, _put
from .foley import P, _n, _nm
from .machines import _rumble
from .physical import ambience as amb
from .physical import voice as V
from .reverbs import distance as _distance
from .sfx import explosion, sfx

GRAV = 9.81
AIR = 343.0


def _end(y, sr, level=0.8, fade=0.003, tail=0.05) -> np.ndarray:
    """`foley._fin` with its DC filter at 2 Hz instead of 12: the 12 Hz one takes 5 dB off the 8.3 Hz a tornado is sourced at. Raised-cosine ends."""
    y = np.asarray(y, float)
    if int(sr) != P:
        g = math.gcd(int(sr), P)
        y = resample_poly(y, int(sr) // g, P // g)
    y = lfilter([1.0, -1.0], [1.0, -math.exp(-2 * math.pi * 2.0 / int(sr))], y)
    a, b = max(2, int(fade * sr)), max(2, int(tail * sr))
    y[:a] *= np.sin(0.5 * np.pi * np.arange(a) / a) ** 2
    y[-b:] *= np.cos(0.5 * np.pi * np.arange(b) / b) ** 2
    return _nm(y, level)


def _b(fn, n: int, **kw) -> np.ndarray:
    """A brick at P, mono, length n, unit peak, 20 ms out."""
    x = np.asarray(fn(sr=P, **kw), float)
    x = _fit(x.mean(axis=1) if x.ndim == 2 else x, n)
    m = min(n // 2, _n(0.02))
    x[n - m:] *= np.cos(0.5 * np.pi * np.arange(m) / m) ** 2
    return _nm(x)


def _pts(n: int, pts) -> np.ndarray:
    """Envelope through (seconds, value), smooth between points."""
    t, v = zip(*pts)
    u = np.interp(np.arange(n) / P, t, np.arange(len(t)))
    i = np.minimum(u.astype(int), len(t) - 2)
    w = u - i
    v = np.asarray(v, float)
    return v[i] + (v[i + 1] - v[i]) * w * w * (3 - 2 * w)


def _sos(x, kind, f, order=2):
    return sosfilt(butter(order, f, kind, fs=P, output="sos"), x)


def _ring(y, t, f, tau, amp) -> None:
    """`geo._ring` with ends: that one stops dead at exp(-6) of its level, a tick in a quiet bed, and starts on a slope break. 0.25 ms in, 8 tau long, down to zero."""
    s = int(t * P)
    if not 0 <= s < len(y) or f >= 0.45 * P:
        return
    m = min(len(y) - s, max(16, int(8.0 * tau * P)))
    k = np.arange(m) / P
    e = np.exp(-k / tau) * np.minimum(k / 0.00025, 1.0) * np.minimum((m - np.arange(m)) / (0.25 * m), 1.0)
    y[s:s + m] += amp * e * np.sin(2.0 * np.pi * f * k)


def _grain(y, t, f, amp, rng, n=3) -> None:
    for r in [1.0] + list(rng.uniform(1.4, 3.2, n - 1)):
        _ring(y, t, f * r, 3.0 / (f * r), amp / r)


def _glug(n, dur, rate, size, seed) -> np.ndarray:
    """`matter.gurgle` below 2.5 kHz: each of its bubbles starts on a one-sample step from silence (measured: 0.55 of the peak), far above its 165-550 Hz."""
    return _nm(_sos(_b(Ma.gurgle, n, dur=dur, rate=rate, size=size, seed=seed), "lowpass", 2500.0))


# =============================================================================================
# FIREWORKS
# =============================================================================================
def a_weight_db(f):
    """IEC 61672 A curve."""
    f2 = np.maximum(np.asarray(f, float), 1e-3) ** 2
    ra = 12194.0 ** 2 * f2 ** 2 / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194.0 ** 2))
    return 20 * np.log10(ra) + 2.0


def lmax(x, weight=None) -> float:
    """Highest 125 ms (FAST) level in dB re 1, A-weighted if asked."""
    x = np.asarray(x, float)
    if weight == "A":
        X = np.fft.rfft(x)
        x = np.fft.irfft(X * 10 ** (a_weight_db(np.fft.rfftfreq(len(x), 1 / P)) / 20), len(x))
    m = int(0.125 * P)
    p = np.cumsum(np.r_[0.0, x * x])
    e = (p[m:] - p[:-m]) / m if len(x) > m else np.array([p[-1] / len(x)])
    return 10 * math.log10(max(float(e.max()), 1e-30))


def _at(x, db, weight=None) -> np.ndarray:
    return x * 10 ** ((db - lmax(x, weight)) / 20)


def _thump(rng, dur=0.5, tau=0.11) -> np.ndarray:
    """The low-frequency peak at 16-25 Hz: one damped mode."""
    n = _n(dur)
    t = np.arange(n) / P
    return np.sin(2 * np.pi * rng.uniform(18.0, 23.0) * t) * np.exp(-t / tau) * np.minimum(t / 0.004, 1.0) * np.minimum((dur - t) / 0.05, 1.0)


def _launch_noise(rng, dur=0.35) -> np.ndarray:
    """Flat per third octave, with the launch whistle's 2.5 kHz third octave 10 dB up."""
    n = _n(dur)
    S = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f = np.fft.rfftfreq(n, 1 / P)
    S *= np.where(f > 100.0, 1.0 / np.sqrt(np.maximum(f, 1.0)), 0.0) * np.where((f > 2239.0) & (f < 2818.0), 10 ** 0.5, 1.0)
    t = np.arange(n) / P
    return np.fft.irfft(S, n) * np.minimum(t / 0.002, 1.0) * np.exp(-t / 0.09) * np.minimum((dur - t) / 0.03, 1.0)


def _burst(rng, r, a_db, lf_db, seed, crackle=0.0, dur=2.5) -> np.ndarray:
    """Friedlander blast above 80 Hz + low mode, each at its measured level, then the stars crackling."""
    n = _n(dur)
    bl = np.asarray(explosion(sr=P, dur=r / AIR + 0.6, energy=0.3, distance=r, debris=0.0, enclosure=0.0, seed=seed), float)
    bl = _fit(bl[np.argmax(np.abs(bl) > 1e-3 * np.abs(bl).max()):], n)      # `explosion` waits r/c itself: `firework` places the burst
    y = _at(_sos(bl, "highpass", 80.0, 6), a_db, "A") + _fit(_at(_thump(rng), lf_db), n)
    if crackle > 0:
        c = np.zeros(n)
        for s in rng.uniform(0.3, dur - 0.1, int(180 * crackle)):
            _grain(c, s, rng.uniform(1500.0, 6000.0), rng.uniform(0.3, 1.0) * math.exp(-s / 0.9), rng, 2)
        y += _at(c, a_db - 14.0, "A")
    return y


# kind: launch (A, LF), burst (A, LF), altitude, bursts, crackle
FIREWORK = {"mortier": ((85.0, 104.0), (101.0, 101.0), 130.0, 1, 1.0),
            "bouquet": ((82.0, 98.0), (99.0, 100.0), 120.0, 6, 2.0),
            "pop": ((82.0, 98.0), (101.0, 95.0), 120.0, 1, 0.0),
            "sifflet": ((82.0, 98.0), (101.0, 95.0), 120.0, 1, 0.4)}
REF_M = 102.0       # m, where the levels were measured
RISE_S = 3.0
T_LAUNCH = 0.05
WHISTLE = (6000.0, 2000.0)      # Hz at the start and the end of the whistle


def whistle_freq(t, dur: float):
    """A quarter-wave tube whose burning composition leaves it longer at a steady rate: f = c / 4L falls as 1 / (1 + k t)."""
    f0, f1 = WHISTLE
    return f0 / (1.0 + (f0 / f1 - 1.0) * np.clip(np.asarray(t, float) / dur, 0.0, 1.0))


@sfx("firework", "Aerial firework, launch, climb, burst, stars: mortier (size 4 shell, crackle), bouquet (star mine), pop (size 3, dry), sifflet (whistle falling 6 to 2 kHz as its tube burns empty). Levels measured at 102 m.",
     kind=("mortier", " | ".join(FIREWORK)), distance=(102.0, "m from the launch: launch and burst fall 6 dB per doubling of their own range, the burst arrives later"), seed=(0, "variation"))
def firework(sr=DEFAULT_SR, kind="mortier", distance=102.0, seed=0):
    k = _kind(kind, FIREWORK, "firework")
    (la, ll), (ba, bl), h, bursts, crackle = FIREWORK[k]
    rng = np.random.default_rng([abs(int(seed)), 161])
    d = max(float(distance), 1.0)
    r, r0 = math.hypot(d, h), math.hypot(REF_M, h)
    spread_l, spread_b = -20 * math.log10(d / REF_M), -20 * math.log10(r / r0)
    tb = T_LAUNCH + RISE_S + (r - d) / AIR
    out = np.zeros(_n(tb + 0.5 * (bursts - 1) + 3.0))
    FO._put(out, _fit(_at(_launch_noise(rng), la + spread_l, "A"), _n(0.5)) + _at(_thump(rng), ll + spread_l), T_LAUNCH)
    climb = _n(RISE_S)
    if k == "sifflet":
        w = np.sin(2 * np.pi * np.cumsum(whistle_freq(np.arange(climb) / P, RISE_S - 0.15)) / P)
        e = _pts(climb, [(0, 0), (0.1, 1), (RISE_S - 0.3, 1), (RISE_S - 0.15, 0), (RISE_S, 0)])
        FO._put(out, _at(w * e, la - 4.0 + spread_l, "A"), T_LAUNCH + 0.05)
    else:                                                   # the shell's trail
        tr = Ma._hp(Ma._lp(rng.standard_normal(climb), 5000.0), 800.0) * _pts(climb, [(0, 0), (0.15, 1), (RISE_S, 0)])
        FO._put(out, _at(tr, la - 18.0 + spread_l, "A"), T_LAUNCH + 0.05)
    for i in range(bursts):
        t = tb + (0.5 * i + rng.uniform(-0.1, 0.1) if i else 0.0)   # star mine: far enough apart for FAST to read each one
        FO._put(out, _burst(rng, r, ba + spread_b, bl + spread_b, int(seed) + i, crackle), t)
    return _end(out, sr, 0.85)


def _pops(rng, seed) -> list:
    out = []
    for i in range(6):
        x = np.asarray(explosion(sr=P, dur=0.1, energy=0.002, distance=2.0, debris=0.0, enclosure=0.0, seed=seed + i), float)
        x = _sos(x[np.argmax(np.abs(x) > 1e-3 * np.abs(x).max()):], "highpass", 150.0)
        x = _fit(x, _n(0.08)) * np.exp(-np.arange(_n(0.08)) / (0.012 * P))    # a paper tube, not an open-air blast
        x[-_n(0.01):] *= np.linspace(1.0, 0.0, _n(0.01))
        out.append(_nm(x))
    return out


@sfx("firecracker", "A string of firecrackers: small blasts one after another, 30-200 ms apart, fading strength at random.",
     count=(20, "crackers"), rate=(12.0, "crackers a second on average; each gap stays within 30-200 ms"), seed=(0, "variation"))
def firecracker(sr=DEFAULT_SR, count=20, rate=12.0, seed=0):
    rng = np.random.default_rng([abs(int(seed)), 162])
    c = int(np.clip(count, 1, 500))
    gaps = np.clip(rng.lognormal(math.log(1.0 / max(float(rate), 1.0)), 0.4, c), 0.03, 0.2)
    gaps[0] = 0.01
    t = np.cumsum(gaps)
    out = np.zeros(_n(t[-1] + 0.2))
    p = _pops(rng, int(seed))
    for ti in t:
        FO._put(out, p[rng.integers(len(p))], float(ti), rng.uniform(0.5, 1.0))
    return _end(out, sr, 0.85, tail=0.02)


# =============================================================================================
# THE GROUND, HEAT, WATER UNDER PRESSURE
# =============================================================================================
@sfx("lava_flow", "A lava flow, not an eruption (that is `lava`): thermal cracks, slow bubbles, a low mass moving.",
     flow=(0.5, "0..1 cooling crust (0) .. fast fluid flow (1): more bubbles and rumble, fewer cracks"), dur=(8.0, "s"), seed=(0, "variation"))
def lava_flow(sr=DEFAULT_SR, flow=0.5, dur=8.0, seed=0):
    rng = np.random.default_rng([abs(int(seed)), 163])
    n, fl = _n(dur), float(np.clip(flow, 0.0, 1.0))
    cr = np.zeros(n)
    for s in rng.uniform(0.0, dur - 0.1, int(dur * (10.0 - 8.0 * fl))):
        _grain(cr, s, rng.uniform(800.0, 4000.0), rng.uniform(0.2, 1.0), rng, 2)
    glug = _glug(n, dur, 0.5 + 2.0 * fl, 0.9, int(seed))
    low = _rumble(dur, 25.0, 70.0, int(seed))
    return _end(_nm(cr) * (0.7 - 0.4 * fl) + glug * (0.2 + 0.5 * fl) + _fit(low, n) * (0.3 + 0.7 * fl), sr, 0.8)


GEOTHERMAL = ("fumerolle", "geyser", "solfatare", "boue")


@sfx("geothermal", "Geothermal vents: fumerolle (steady steam jet), geyser (tremor, jet, fall), solfatare (hissing vents over sulphur-mud simmer), boue (mud pots).",
     kind=("fumerolle", " | ".join(GEOTHERMAL)), dur=(8.0, "s"), seed=(0, "variation"))
def geothermal(sr=DEFAULT_SR, kind="fumerolle", dur=8.0, seed=0):
    """Geyser and mud carry the laws of `geo.geyser`, `geo.mud_pot`."""
    k = _kind(kind, GEOTHERMAL, "geothermal")
    n, s = _n(dur), int(seed)
    rng = np.random.default_rng([abs(s), 164])
    if k == "geyser":
        y = _b(G.geyser, n, dur=dur, seed=s)
    elif k == "boue":
        y = _b(G.mud_pot, n, dur=dur, thick=0.7, seed=s)
    elif k == "fumerolle":
        jet = _b(Ma.steam, n, dur=dur, pressure=0.55, orifice=30.0, wet=0.2, tone=7000.0, seed=s)
        y = jet * (0.75 + 0.25 * Ma._lfn(rng, n, 0.6)) + 0.25 * _fit(_rumble(dur, 30.0, 90.0, s), n)
    else:
        jet = _b(Ma.steam, n, dur=dur, pressure=0.35, orifice=6.0, wet=0.5, tone=9000.0, seed=s)
        y = 0.45 * jet * (0.7 + 0.3 * Ma._lfn(rng, n, 1.5)) + 0.8 * _b(Ma.boil, n, dur=dur, intensity=0.4, size=0.7, seed=s) + 0.4 * _glug(n, dur, 1.2, 0.7, s)
    return _end(y, sr, 0.8)


TORNADO_MODES = ((8.3, 1.6), (18.0, 3.4), (29.0, 3.8), (36.0, 4.8), (44.0, 4.9))
TORNADO_GAINS = (1.0, 0.7, 0.6, 0.5, 0.45)
TORNADO_BAND = (40.0, 200.0)


def _infrasound(n: int, rng) -> np.ndarray:
    """Noise under the upper envelope of the measured peaks: summed band-passes would move the 8.3 Hz peak up on the skirt of the 18 Hz one."""
    f = np.maximum(np.fft.rfftfreq(n, 1 / P), 1e-3)
    h = np.max([g / np.sqrt(1.0 + (q * (f / f0 - f0 / f)) ** 2) for (f0, q), g in zip(TORNADO_MODES, TORNADO_GAINS)], axis=0)
    return _nm(np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * h, n))


def _tornado_roar(n: int, rng) -> np.ndarray:
    """Band-pass with its -3 dB points on the band: centre sqrt(40 x 200) = 89 Hz, inside the 80-100 Hz centre; Q = 89 / 160."""
    lo, hi = TORNADO_BAND
    c = math.sqrt(lo * hi)
    f = np.maximum(np.fft.rfftfreq(n, 1 / P), 1e-3)
    h = 1.0 / np.sqrt(1.0 + (c / (hi - lo) * (f / c - c / f)) ** 2)
    h *= np.minimum(hi / f, 1.0) ** 1.5 * np.minimum(f / lo, 1.0) ** 1.5    # steeper outside the band: its 2-pole skirts alone leave a third of the power above 200 Hz
    return np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * h, n)


def _unit_rms(x) -> np.ndarray:
    return x / (np.sqrt(np.mean(np.square(x))) + 1e-12)


@sfx("tornado", "Tornado: infrasound peak at 8.3 Hz and its overtones, roar in a 40-200 Hz band, wind above it, debris when close.",
     distance=(500.0, "m: farther = air takes the top off, less debris"), dur=(12.0, "s"), seed=(0, "variation"))
def tornado(sr=DEFAULT_SR, distance=500.0, dur=12.0, seed=0):
    rng = np.random.default_rng([abs(int(seed)), 165])
    n, d = _n(dur), max(float(distance), 10.0)
    roar = _tornado_roar(n, rng) * (0.75 + 0.25 * Ma._lfn(rng, n, 0.3))
    wind = Ma._lp(_sos(rng.standard_normal(n), "highpass", 300.0), 1800.0) * (0.6 + 0.4 * np.abs(Ma._lfn(rng, n, 0.5)))
    y = 0.6 * _unit_rms(_infrasound(n, rng)) + _unit_rms(roar) + 0.15 * _unit_rms(wind)
    near = min(100.0 / d, 1.0)
    if near > 0.05:
        y += near * _b(FO.shake, n, preset="rocks", energy=0.8, rate=6.0, dur=dur, seed=int(seed))
    y = _fit(np.asarray(_distance(y, sr=P, metres=d, ref=1.0), float), n)   # the level goes back to the peak in `_end`: what stays is the air
    return _end(y, sr, 0.8, tail=0.3, fade=0.3)


VOLCANO = ("tremor", "eruption", "retombees", "complet")


def _tremor(n, rng) -> np.ndarray:
    """Below 20 Hz: the top of a volcanic tremor's harmonics."""
    return _nm(_sos(Ma._bp(rng.standard_normal(n), 6.0, 0.7) + 0.5 * Ma._bp(rng.standard_normal(n), 14.0, 2.0), "lowpass", 20.0, 4))


@sfx("volcano", "A volcano by stage: tremor (below 20 Hz, a rumble over it), eruption (blast, roaring column), retombees (ash and stones falling), complet (the three in order).",
     stage=("complet", " | ".join(VOLCANO)), dur=(16.0, "s, whatever the stage"), seed=(0, "variation"))
def volcano(sr=DEFAULT_SR, stage="complet", dur=16.0, seed=0):
    k = _kind(stage, VOLCANO, "volcano")
    rng = np.random.default_rng([abs(int(seed)), 166])
    s, n = int(seed), _n(dur)
    y = np.zeros(n)
    t = (0.0, 0.3 * dur, 0.6 * dur) if k == "complet" else (0.0, 0.0, 0.0)
    span = 0.4 * dur if k == "complet" else dur
    m = _n(span)
    if k in ("tremor", "complet"):
        tr = _nm(_unit_rms(_tremor(m, rng)) + 0.3 * _unit_rms(_fit(_rumble(span, 25.0, 60.0, s), m))) * (0.5 if k == "complet" else 1.0)
        FO._put(y, tr * _pts(m, [(0, 0), (0.4 * span, 1), (span - 0.3, 1), (span, 0)]), t[0])
    if k in ("eruption", "complet"):
        e = _n(max(1.2 * span, 3.0))
        bl = _fit(np.asarray(explosion(sr=P, dur=3.0, energy=50.0, distance=300.0, debris=0.2, enclosure=0.0, seed=s), float), e)
        col = _b(Ma.fire, e, dur=e / P, size=1.0, crackle=30.0, wet=0.0, wind=0.8, seed=s) * _pts(e, [(0, 0), (0.2, 1), (e / P * 0.6, 1), (e / P, 0)])
        FO._put(y, _nm(bl) + 0.5 * col + 0.5 * _fit(_rumble(e / P, 20.0, 60.0, s + 1), e) * _pts(e, [(0, 0), (0.1, 1), (e / P, 0)]), t[1])
    if k in ("retombees", "complet"):
        f = _b(K.hail, m, dur=span, rate=120.0, size=14.0, decay=0.8, hardness=0.8, body="rect_plate", material="stone", surface_size=3.0, bright=0.4, seed=s)
        FO._put(y, 0.6 * f * _pts(m, [(0, 0), (0.1 * span, 1), (span, 1)]), t[2])
    return _end(y, sr, 0.85, tail=0.2)


TSUNAMI = ("retrait", "vague", "debris")


@sfx("tsunami", "A tsunami by stage: retrait (the water drawn back, a low rush), vague (a wall of water, huge surf and rumble), debris (the flood carrying wreckage).",
     stage=("vague", " | ".join(TSUNAMI)), dur=(12.0, "s"), seed=(0, "variation"))
def tsunami(sr=DEFAULT_SR, stage="vague", dur=12.0, seed=0):
    k = _kind(stage, TSUNAMI, "tsunami")
    s, n = int(seed), _n(dur)
    rng = np.random.default_rng([abs(s), 167])
    low = _fit(_rumble(dur, 20.0, 70.0, s), n)
    if k == "retrait":
        wash = _sos(_b(Ma.babble, n, dur=dur, flow=0.5, size=2.0, turbulence=0.8, seed=s), "lowpass", 1500.0)
        y = (_nm(wash) * 0.7 + 0.5 * low) * _pts(n, [(0, 0), (0.3 * dur, 1), (dur, 0.2)])
    else:
        roar = _b(Ma.waterfall, n, dur=dur, height=40.0, distance=25.0, seed=s)
        surge = _b(Ma.surf, n, dur=dur, distance=20.0, period_min=3.0, period_max=5.0, bubble_rate=8000.0, seed=s)
        y = 0.8 * roar + 0.5 * surge + 0.9 * low
        y *= _pts(n, [(0, 0), (0.25 * dur, 1), (dur, 1)]) if k == "vague" else 0.6
        if k == "debris":
            for i, (mat, sz) in enumerate(zip(("wood", "wood", "stone", "glass", "wood", "ceramic"), (0.8, 1.2, 0.6, 0.5, 0.9, 0.4))):
                hit = _b(K.smash, _n(1.6), material=mat, size=sz, energy=1.0, fragments=6, dur=1.5, seed=s + i)
                _put(y, hit, rng.uniform(0.1, dur - 1.7), rng.uniform(0.4, 0.8))
    return _end(y, sr, 0.85, tail=0.2)


def _kuramoto(n, f, cpl, rng, grains=800, spread=0.02, ctrl=2000) -> tuple:
    """Grains as phase oscillators at the shock frequency. cpl: coupling at control rate (rad/s), 0 = out of step.
    Returns the layer's pressure (the sum of one pulse per period per grain, at ctrl) and the order parameter."""
    w = 2 * np.pi * f * (1.0 + spread * rng.standard_normal(grains))
    th = rng.uniform(0.0, 2 * np.pi, grains)
    p, r = np.zeros(n), np.zeros(n)
    dt = 1.0 / ctrl
    for i in range(n):
        z = np.exp(1j * th)
        m = z.mean()
        r[i] = abs(m)
        th = th + dt * (w + cpl[i] * (m * z.conj()).imag)
        p[i] = np.exp(2.0 * (z.real - 1.0)).mean()
    return p - p.mean(), r


SONG_C = 0.92       # m/s sound speed in the flowing layer


def song_freq(grain_um: float) -> float:
    """f = 0.4 sqrt(g/d)."""
    return 0.4 * math.sqrt(GRAV / (float(grain_um) * 1e-6))


@sfx("singing_sand", "Singing sand: an avalanche whose grains lock into step by a resonance in the flowing layer. The note is 0.4 sqrt(g/d); it comes up, holds and stops when the surface slows below c/4.",
     grain=(160.0, "um grain diameter: 160 = 99 Hz, 210 = 87, 270 = 76, 340 = 68"), depth=(6.0, "mm of flowing layer at the start; it drains over dur. Below c/4f (2.3 mm at 160 um) there is no song"),
     dur=(8.0, "s"), seed=(0, "variation"))
def singing_sand(sr=DEFAULT_SR, grain=160.0, depth=6.0, dur=8.0, seed=0):
    rng = np.random.default_rng([abs(int(seed)), 168])
    g = float(np.clip(grain, 60.0, 800.0))
    f = song_freq(g)
    ctrl = 2000
    m = int(dur * ctrl)
    t = np.arange(m) / ctrl
    rise = 0.1 * dur
    h = float(depth) * 1e-3 * np.minimum(t / rise, 1.0) * np.minimum((dur - t) / (dur - rise), 1.0)
    u = f * h
    p, _ = _kuramoto(m, f, 2 * np.pi * f * 0.1 * (u >= SONG_C / 4.0), rng, ctrl=ctrl)
    n = _n(dur)
    song = _fit(resample_poly(p, P // ctrl, 1), n)
    un = _fit(resample_poly(u, P // ctrl, 1), n)
    rustle = Ma._lp(Ma._hp(rng.standard_normal(n), 1500.0), 6000.0) * np.maximum(un, 0.0)        # the grains not in step
    # the song at its own scale: 0.7 = peak of the pulse when all grains are in step, so unlocked grains stay small instead of being brought up
    return _end(song / 0.7 + 0.05 * rustle / max(abs(un).max(), 1e-6), sr, 0.8, tail=0.1, fade=0.01)


# =============================================================================================
# THINGS BREAKING AND FALLING
# =============================================================================================
@sfx("collapse", "A structure coming down: a train of falls, breaking, the mass settling, dust running after.",
     size=(1.0, "0.3..3 of the building part: lower, longer"), material=("stone", "stone|wood|ceramic|glass|concrete"), seed=(0, "variation"))
def collapse(sr=DEFAULT_SR, size=1.0, material="stone", seed=0):
    sz = float(np.clip(size, 0.3, 3.0))
    if material not in ("stone", "wood", "ceramic", "glass", "concrete"):
        raise ValueError(f"collapse material {material!r}: use stone | wood | ceramic | glass | concrete")
    s = int(seed)
    rng = np.random.default_rng([abs(s), 169])
    dur = 3.0 + 3.0 * sz
    n = _n(dur)
    y = np.zeros(n)
    t = 0.05
    for i in range(int(6 + 4 * sz)):
        hit = _b(K.smash, _n(1.6), material="stone" if material == "concrete" else material, size=0.4 * sz * rng.uniform(0.5, 1.2), energy=rng.uniform(0.8, 1.6), fragments=8, dur=1.5, seed=s + i)
        _put(y, hit, t, rng.uniform(0.5, 1.0))
        t += rng.uniform(0.08, 0.45)
        if t > 0.55 * dur:
            break
    low = _rumble(dur, 30.0 / sz, 110.0 / sz, s)
    dust = _b(FO.shake, n, preset="fine_gravel", energy=0.5, rate=8.0, dur=dur, seed=s) * _pts(n, [(0, 0), (0.3 * dur, 0), (0.45 * dur, 1), (dur, 0)])
    return _end(y + 0.8 * _fit(low, n) * _pts(n, [(0, 0), (0.1, 1), (0.5 * dur, 1), (dur, 0)]) + 0.2 * dust, sr, 0.85, tail=0.2)


CRACK = ("branche", "planche", "os", "glace")


@sfx("crack", "Something giving way: branche (green wood, bends then snaps), planche (a board, creak then break), os (short, dry), glace (lake ice, the falling pew).",
     kind=("branche", " | ".join(CRACK)), seed=(0, "variation"))
def crack(sr=DEFAULT_SR, kind="branche", seed=0):
    """The ice is `matter.ice_crack`."""
    k = _kind(kind, CRACK, "crack")
    s = int(seed)
    if k == "glace":
        return _end(_b(Ma.ice_crack, _n(1.5), dur=1.5, thickness=0.05, distance=60.0, cracks=1, seed=s), sr, 0.8)
    if k == "os":
        n = _n(0.25)
        rng = np.random.default_rng([abs(s), 170])
        y = np.zeros(n)
        for t in (0.005, 0.012, 0.03):
            _grain(y, t, rng.uniform(1400.0, 2400.0), rng.uniform(0.6, 1.0), rng, 3)
        return _end(y, sr, 0.8, fade=0.0005, tail=0.02)
    size, cd, cf = (0.12, 0.25, "ramp") if k == "branche" else (0.35, 0.5, "pull")
    n = _n(cd + 1.0)
    y = np.zeros(n)
    _put(y, _b(FO.creak, _n(cd), material="wood", force=cf, level=0.8, dur=cd, size=size * 3.0, seed=s), 0.0, 0.3)
    _put(y, _b(K.smash, _n(0.9), material="wood", size=size, shape="bar", energy=1.2, fragments=3 if k == "branche" else 6, dur=0.8, burst=0.8, seed=s), cd)
    return _end(y, sr, 0.8, tail=0.03)


# =============================================================================================
# RAIN AND STORMS
# =============================================================================================
def raindrops(n: int, f: float, th: float, rng) -> np.ndarray:
    """Low noise moves an oscillator around 40-80 Hz, times 10 x the noise squared, thresholded and re-centred, high-passed twice at 500 Hz."""
    x = V._pd_bp(rng.standard_normal(n), np.full(n, float(f)), np.full(n, 0.4), float(P))
    x = 0.12 * x / (np.std(x) + 1e-12)
    y = np.cos(2 * np.pi * np.cumsum(40.0 + 80.0 * V._pd_lop(np.abs(x), 500.0, float(P))) / P) * 10.0 * x * x
    return amb.hip(amb.hip(0.5 * (np.maximum(y, th) - th), 500.0), 500.0)


def drop_spikes(n: int, rng, f=7.7, q=0.013, th=0.16, a=3.0, hip=2000.0) -> np.ndarray:
    """Noise, bp~ f q, max~ s, -~ s, x a, clip 0..1, fourth power; the instance is high-passed at 2 kHz."""
    x = V._pd_bp(rng.uniform(-1.0, 1.0, n), np.full(n, f), np.full(n, q), float(P))
    return amb.hip(np.clip((np.maximum(x, th) - th) * a, 0.0, 1.0) ** 4, hip)


def _bp_coef(f, q):
    """Pd bp~ as a biquad (`voice._pd_bp`), for block processing with state."""
    w = 2 * math.pi * f / P
    o = min(w / q, 1.0)
    r = 1.0 - o
    return [2.0 * o * (o + r * w)], [1.0, -2.0 * math.cos(w) * r, r * r]


def glass_window(x, bands=(2007.0, 1994.0, 1986.0, 1969.0), rings=(254.0, 669.0, 443.0, 551.0), delays=(3.7, 4.2), gain=0.61, q=2.3) -> np.ndarray:
    """A glass window about 1 m2. Two recirculating delays; each feeds two ring modulators, each into its own band-pass and back into a delay line.
    Which branch feeds which line is read from the order of the objects in the figure's list."""
    x = np.asarray(x, float)
    n = len(x)
    da, db = [int(round(d * 1e-3 * P)) for d in delays]
    size = max(da, db) + 1
    la, lb = np.zeros(size), np.zeros(size)
    co = [_bp_coef(f, q) for f in bands]
    zi = [np.zeros(2) for _ in bands]
    y = np.zeros(n)
    blk = min(da, db) // 2
    for i in range(0, n, blk):
        m = min(blk, n - i)
        idx = np.arange(i, i + m)
        ra, rb = la[(idx - da) % size], lb[(idx - db) % size]
        t = idx / P
        o = []
        for k, src in enumerate((ra, ra, rb, rb)):
            v, zi[k] = lfilter(co[k][0], co[k][1], src * np.cos(2 * np.pi * rings[k] * t) * gain, zi=zi[k])
            o.append(v)
        la[idx % size] = x[i:i + m] + o[1] + o[3]
        lb[idx % size] = o[0] + o[2]
        y[i:i + m] = o[0] + o[1] + o[2] + o[3]
    return y


RAIN = ("vitre", "toit", "bois", "sol", "eau", "feuilles")


def _rain(surface: str, n: int, intensity: float, wetness: float, seed: int) -> np.ndarray:
    """One surface, mono, unit peak. Threshold 0.35 sparse, 0.2 at intensity 0.5, straight line through them."""
    rng = np.random.default_rng([abs(seed), 171])
    iy = float(np.clip(intensity, 0.05, 1.0))
    th = max(0.5 - 0.6 * iy, 0.02)
    dur = n / P
    if surface == "vitre":
        hit = raindrops(n, 70.0, th, rng)
        y = glass_window(amb.hip(hit, 9000.0) * 10.0)
    elif surface in ("toit", "bois"):                       # small-drop hail
        mat = "metal" if surface == "toit" else "wood"
        y = _b(K.hail, n, dur=dur, rate=30.0 + 500.0 * iy, size=2.0 + 2.0 * iy, decay=0.0, hardness=0.25, body="rect_plate", material=mat, surface_size=2.0, bright=0.5, seed=seed)
    elif surface == "eau":
        y = _b(Ma.bubbles, n, dur=dur, rate=20.0 + 400.0 * iy, size=1.5, spread=0.5, cluster=0.2, seed=seed)
    elif surface == "feuilles":                             # a lengthened pulse
        h = np.hanning(_n(0.006))
        y = np.convolve(raindrops(n, 70.0, th, rng), h / h.sum(), "same")
        y = _nm(y) + 0.3 * _nm(Ma._hp(Ma._lp(rng.standard_normal(n), 7000.0), 1500.0)) * iy
    else:                                                   # sol: big low drops on dry ground, then puddles
        dry = raindrops(n, 50.0, th, rng)
        wet = _b(Ma.bubbles, n, dur=dur, rate=30.0 + 300.0 * iy, size=2.0, spread=0.5, cluster=0.2, seed=seed)
        w = float(np.clip(wetness, 0.0, 1.0))
        y = math.cos(0.5 * np.pi * w) * _nm(dry) + math.sin(0.5 * np.pi * w) * wet
    # filler between Poisson drops and noise, used sparingly
    sp = drop_spikes(n, rng)
    return _nm(_nm(y) + 0.12 * iy * _nm(sp))


@sfx("rain_on", "Rain on one surface: vitre (dispersive window), toit (tin), bois, sol (dry ground to puddles with wetness), eau, feuilles.",
     surface=("vitre", " | ".join(RAIN)), intensity=(0.5, "0..1"), wetness=(0.0, "0 dry .. 1 puddles, ground only"), wind=(1.0, "x density: wind drives the rain"),
     rate=(0.0, "mm/h, eau only: > 0 = drop by drop, each drop by its size class (rain_water); 0 = bubble bed from intensity"),
     dur=(4.0, "s"), seed=(0, "variation"))
def rain_on(sr=DEFAULT_SR, surface="vitre", intensity=0.5, wetness=0.0, wind=1.0, rate=0.0, dur=4.0, seed=0):
    k = _kind(surface, RAIN, "rain_on")
    if k == "eau" and rate > 0:
        y = amb.rain_on_water(float(dur), np.random.default_rng([abs(int(seed)), 60]), rate_mm_h=float(np.clip(rate, 0.1, 300.0)))
        return _end(_fit(y, _n(dur)), sr, 0.7, fade=0.02, tail=0.1)
    return _end(_rain(k, _n(dur), float(intensity) * max(float(wind), 0.0), float(wetness), int(seed)), sr, 0.7, fade=0.02, tail=0.1)


@sfx("fire_out", "A fire put out by water: the fire falls, a sizzle and the rising whine of steam take over, then they die too.",
     dur=(5.0, "s"),
     detail=(1.0, "0..1 level of the extinction details: steam whining out of the wood, water bubbling in the embers, cooling creaks, a steam shriek"),
     seed=(0, "variation"))
def fire_out(sr=DEFAULT_SR, dur=5.0, detail=1.0, seed=0):
    n, s = _n(dur), int(seed)
    fi = _b(Ma.fire, n, dur=dur, size=0.5, crackle=14.0, wet=0.4, seed=s) * _pts(n, [(0, 0), (0.1, 1), (0.25 * dur, 1), (0.7 * dur, 0), (dur, 0)])
    si = _b(Ma.sizzle, n, dur=dur, intensity=0.8, spit=6.0, hiss=0.5, seed=s) * _pts(n, [(0, 0), (0.25 * dur, 0), (0.4 * dur, 1), (dur, 0)])
    st = _b(Ma.steam, n, dur=dur, pressure=0.6, orifice=1.5, wet=0.3, tone=9000.0, seed=s) * _pts(n, [(0, 0), (0.3 * dur, 0), (0.5 * dur, 1), (0.85 * dur, 0), (dur, 0)])
    y = fi + 0.7 * si + 0.3 * st
    d = float(np.clip(detail, 0.0, 1.0))
    if d > 0:
        wh = _b(Ma.air_leak, n, dur=dur, pressure=0.3, orifice=1.0, whistle=1.0, seed=s) * _pts(n, [(0, 0), (0.25 * dur, 0), (0.4 * dur, 1), (0.7 * dur, 0), (dur, 0)])
        bu = _b(Ma.boil, n, dur=dur, intensity=0.7, size=0.1, burst=0.4, seed=s) * _pts(n, [(0, 0), (0.3 * dur, 0), (0.45 * dur, 1), (0.8 * dur, 0), (dur, 0)])
        cr = _b(Ma.freeze, n, dur=dur, rate=4.0, creak=0.8, pings=0, seed=s) * _pts(n, [(0, 0), (0.4 * dur, 0), (0.55 * dur, 1), (0.85 * dur, 0), (dur, 0)])
        sh = _b(Ma.kettle, n, dur=dur, onset=0.0, flow=3300.0, breath=0.2, rumble=0.0, seed=s) * _pts(n, [(0, 0), (0.35 * dur, 0), (0.45 * dur, 1), (0.6 * dur, 0), (dur, 0)])
        y = y + d * (0.15 * wh + 0.25 * bu + 0.2 * cr + 0.1 * sh)
    return _end(y, sr, 0.8, fade=0.01, tail=0.1)


GUST_D = 30.0       # m between objects


def gust_delay(metres: float, wind: float) -> float:
    return float(metres) / max(float(wind), 1.0)


def _flat(x) -> np.ndarray:
    """A brick without its own gusts: one wind has to move everything, or the delay between objects means nothing."""
    h = np.hanning(_n(0.25))
    e = np.sqrt(np.maximum(fftconvolve(x * x, h / h.sum(), "same"), 0.0))
    return _nm(x / (e + 0.05 * e.max()))


def _storm_parts(intensity, wind, dur, seed) -> dict:
    """A gust moves the flag, the trees d/v later, the wires d/v after them; the trees' gust pushes the rain. Rain goes from dry ground to puddles over dur."""
    n, s = _n(dur), int(seed)
    rng = np.random.default_rng([abs(s), 172])
    iy = float(np.clip(intensity, 0.05, 1.0))
    d = gust_delay(GUST_D, wind)
    m = int(np.ceil(dur / 2.0)) + 3
    knots = rng.uniform(0.25, 1.0, m)
    tk = np.arange(m) * 2.0
    env = lambda lag: _pts(n, [(float(t + lag), float(v)) for t, v in zip(tk - 4.0, knots)])
    flag = _flat(_rustle(n, rng, 12.0, 300.0, 3000.0, 0.06)) * env(0.0)     # flapping
    trees = _flat(_b(Ma.gust, n, dur=dur, through="leaves", strength=0.6, seed=s + 1)) * env(d)
    wires = _flat(_b(Ma.gust, n, dur=dur, through="wire", strength=0.6, seed=s)) * env(2.0 * d)
    dry, wet = _rain("sol", n, iy, 0.0, s), _rain("sol", n, iy, 1.0, s + 1)
    w = _pts(n, [(0, 0), (0.3 * dur, 0), (0.7 * dur, 1), (dur, 1)])
    rain = (np.cos(0.5 * np.pi * w) * dry + np.sin(0.5 * np.pi * w) * wet) * (0.4 + 0.6 * env(d))
    th = np.zeros(n)
    bolts = max(1, int(round(iy * dur / 8.0)))
    for i in range(bolts):
        km = rng.uniform(0.5, 3.0)
        t0 = rng.uniform(0.0, max(dur - 6.0, 0.1))
        FO._put(th, _b(Ma.lightning, _n(6.0), dur=6.0, distance=km * 1000.0, seed=s + i), t0, 1.0 / km ** 0.5)
    return {"flag": flag, "trees": trees, "wires": wires, "rain": rain, "thunder": th}


@sfx("storm", "A storm: gusts crossing the scene at the wind's speed, rain they push about going from dry ground to puddles, thunder at its distance.",
     intensity=(0.7, "0..1"), wind=(30.0, "m/s: objects 30 m apart get a gust 30/wind s after each other"), dur=(16.0, "s"), seed=(0, "variation"))
def storm(sr=DEFAULT_SR, intensity=0.7, wind=30.0, dur=16.0, seed=0):
    p = _storm_parts(intensity, wind, dur, seed)
    return _end(0.25 * p["flag"] + 0.5 * p["trees"] + 0.5 * p["wires"] + 0.7 * p["rain"] + 0.9 * p["thunder"], sr, 0.85, fade=0.05, tail=0.2)


# =============================================================================================
# GROUND AND PLANTS
# =============================================================================================
FOLIAGE = ("herbe", "feuilles", "branches", "passage")


def _rustle(n, rng, rate, lo, hi, width) -> np.ndarray:
    """Band noise under a stochastic gate of short overlapping swells."""
    g = np.zeros(n)
    h = np.hanning(max(8, _n(width)))
    for s in rng.integers(0, max(1, n - len(h)), int(rate * n / P)):
        g[s:s + len(h)] += h * rng.uniform(0.3, 1.0)
    return Ma._lp(Ma._hp(rng.standard_normal(n), lo), hi) * g


@sfx("foliage", "Plants handled or walked through, no wind: herbe (fine hiss), feuilles (papery rustle), branches (twigs, creak), passage (someone pushing through).",
     kind=("feuilles", " | ".join(FOLIAGE)), dur=(2.0, "s"), seed=(0, "variation"))
def foliage(sr=DEFAULT_SR, kind="feuilles", dur=2.0, seed=0):
    k = _kind(kind, FOLIAGE, "foliage")
    n, s = _n(dur), int(seed)
    rng = np.random.default_rng([abs(s), 173])
    if k == "herbe":
        y = _rustle(n, rng, 30.0, 3000.0, 12000.0, 0.08)
    elif k == "feuilles":
        y = _rustle(n, rng, 18.0, 1200.0, 8000.0, 0.05) + 0.3 * _b(K.crumple, n, kind="paper", energy=0.3, density=0.7, size=0.3, dur=dur, seed=s)
    elif k == "branches":
        y = np.zeros(n)
        for t in rng.uniform(0.0, dur - 0.1, int(6 * dur)):
            _grain(y, t, rng.uniform(700.0, 2500.0), rng.uniform(0.3, 1.0), rng, 2)
        y = _nm(y) + 0.5 * _b(FO.creak, n, material="wood", force="sway", level=0.6, dur=dur, size=0.4, seed=s) + 0.25 * _nm(_rustle(n, rng, 6.0, 800.0, 5000.0, 0.1))
    else:
        steps = _b(C.footsteps, n, ground="leaves", gait="walk", steps=max(2, int(dur * 1.75)), seed=s)
        y = 0.6 * steps + _nm(_rustle(n, rng, 6.0, 900.0, 7000.0, 0.3)) * _pts(n, [(0, 0.2), (0.4 * dur, 1), (dur, 0.4)])
    return _end(y, sr, 0.7, fade=0.01, tail=0.05)


QUAKE = ("premices", "secousse", "repliques")


@sfx("earthquake_stage", "An earthquake by stage: premices (the rumble arriving), secousse (full shaking, objects rattling, debris), repliques (separated aftershocks). `earthquake` is the single heard event.",
     stage=("secousse", " | ".join(QUAKE)), dur=(12.0, "s"), seed=(0, "variation"))
def earthquake_stage(sr=DEFAULT_SR, stage="secousse", dur=12.0, seed=0):
    """`rumble` 35-90 Hz."""
    k = _kind(stage, QUAKE, "earthquake_stage")
    n, s = _n(dur), int(seed)
    rng = np.random.default_rng([abs(s), 174])
    low = _fit(_rumble(dur, 35.0, 90.0, s), n)
    if k == "premices":
        y = low * _pts(n, [(0, 0), (dur * 0.8, 1), (dur, 0.8)])
    elif k == "repliques":
        e = np.zeros(n)
        for t in np.sort(rng.uniform(0.0, dur - 2.5, 3)):
            m = _n(rng.uniform(1.2, 2.4))
            FO._put(e, _pts(m, [(0, 0), (0.15, 1), (m / P, 0)]) * rng.uniform(0.5, 1.0), float(t))
        y = low * e + 0.2 * e * _b(FO.shake, n, preset="pebbles", energy=0.5, rate=10.0, dur=dur, seed=s)
    else:
        e = _pts(n, [(0, 0), (0.15 * dur, 1), (0.7 * dur, 1), (dur, 0)])
        y = low * e + 0.35 * e * _b(FO.shake, n, preset="pebbles", energy=0.8, rate=14.0, dur=dur, seed=s)
        for i in range(4):
            _put(y, _b(K.smash, _n(1.2), material="ceramic", size=0.3, energy=0.9, dur=1.1, seed=s + i), rng.uniform(0.2 * dur, 0.75 * dur), 0.3)
    return _end(y, sr, 0.85, tail=0.2)


DIRT = ("pelle", "poignee", "coulee", "poussiere")


@sfx("dirt", "Soil and sand: pelle (spade in, load thrown down), poignee (a handful dropped), coulee (a steady pour), poussiere (fine dust settling).",
     kind=("pelle", " | ".join(DIRT)), seed=(0, "variation"))
def dirt(sr=DEFAULT_SR, kind="pelle", seed=0):
    """PhISEM `shake`."""
    k = _kind(kind, DIRT, "dirt")
    s = int(seed)
    if k == "poignee":
        n = _n(1.0)
        y = _b(FO.shake, n, preset="fine_gravel", energy=0.7, rate=0.0, dur=1.0, seed=s)
    elif k == "coulee":
        n = _n(3.0)
        y = _b(FO.shake, n, preset="fine_gravel", energy=0.6, rate=24.0, dur=3.0, size=2.5, seed=s) * _pts(n, [(0, 0), (0.2, 1), (2.5, 1), (3.0, 0)])
    elif k == "poussiere":
        n = _n(2.5)
        rng = np.random.default_rng([abs(s), 175])
        y = _rustle(n, rng, 25.0, 4000.0, 14000.0, 0.15) * _pts(n, [(0, 1), (2.5, 0)])
    else:
        n = _n(1.8)
        y = np.zeros(n)
        _put(y, _b(K.scrape, _n(0.35), dur=0.35, speed=0.8, roughness=0.9, seed=s), 0.02, 0.8)
        _put(y, _b(FO.shake, _n(0.7), preset="gravel", energy=0.9, rate=0.0, dur=0.7, seed=s), 0.9, 1.0)
        _ring(y, 0.9, 90.0, 0.05, 0.7)
    return _end(y, sr, 0.7, fade=0.005, tail=0.03)


TREE = ("craquement", "chute")


@sfx("tree", "One part of a tree falling: craquement (fibres going, faster and faster, to the break), chute (crown through the branches, then the ground). `tree_fall` is the whole event.",
     kind=("craquement", " | ".join(TREE)), size=(1.0, "0.3..2 trunk: deeper, slower"), seed=(0, "variation"))
def tree(sr=DEFAULT_SR, kind="craquement", size=1.0, seed=0):
    """Gaps closing before failure as in `geo.tree_fall`."""
    k = _kind(kind, TREE, "tree")
    sz, s = float(np.clip(size, 0.3, 2.0)), int(seed)
    rng = np.random.default_rng([abs(s), 176])
    if k == "craquement":
        dur = 3.0
        n = _n(dur + 1.0)
        y, t = np.zeros(n), 0.0
        while t < dur:
            x = 1.0 - t / dur
            _grain(y, t, rng.uniform(350.0, 1100.0) / sz, (1.0 - x) ** 1.5 * rng.uniform(0.3, 1.0) + 0.02, rng)
            t += 0.25 * x ** 1.5 * rng.uniform(0.5, 1.2) + 0.006
        _put(y, _b(K.smash, _n(1.0), material="wood", size=0.8 * sz, shape="bar", energy=1.8, fragments=8, dur=0.9, burst=0.8, seed=s), dur, 1.0)
        return _end(y, sr, 0.85, fade=0.005, tail=0.1)
    dur = 4.0
    n = _n(dur)
    tf = 2.6
    leaves = _nm(Ma._lp(Ma._hp(rng.standard_normal(n), 900.0), 7000.0)) * _pts(n, [(0, 0), (tf, 1), (tf + 0.4, 0), (dur, 0)])
    br = _b(K.crumple, n, kind="bag", energy=0.7, density=0.8, size=0.8, dur=dur, seed=s) * _pts(n, [(0, 0), (1.2, 0), (tf, 1), (tf + 0.6, 0), (dur, 0)])
    y = np.zeros(n)
    _ring(y, tf, 42.0 / sz, 0.2, 2.0)
    _ring(y, tf + 0.05, 90.0 / sz, 0.12, 1.0)
    return _end(0.4 * leaves + 0.4 * br + y, sr, 0.85, fade=0.01, tail=0.2)


WIND_KINDS = ("interieur", "sifflement", "tempete", "feuillage")


@sfx("wind_kind", "Wind by character: interieur (heard through a doorway, muffled), sifflement (a gap and wires singing), tempete (canyon roar and howl), feuillage (through leaves).",
     kind=("interieur", " | ".join(WIND_KINDS)), dur=(6.0, "s"), seed=(0, "variation"))
def wind_kind(sr=DEFAULT_SR, kind="interieur", dur=6.0, seed=0):
    """Table over `gust`, `cavity_tone`, `aeolian_harp`."""
    k = _kind(kind, WIND_KINDS, "wind_kind")
    n, s = _n(dur), int(seed)
    if k == "interieur":
        y = _sos(_b(Ma.gust, n, dur=dur, through="doorway", strength=0.4, seed=s), "lowpass", 900.0)
    elif k == "sifflement":
        y = _b(FO.cavity_tone, n, speed=[12.0, 22.0, 14.0, 20.0], length=0.03, depth=0.04, width=0.03, gust=0.5, dur=dur, seed=s) + 0.5 * _b(FO.aeolian_harp, n, wind=9.0, notes="A4 E5", diameter=0.4, dur=dur, seed=s)
    elif k == "tempete":
        y = _b(Ma.gust, n, dur=dur, through="canyon", strength=0.55, seed=s) + 0.7 * _b(Ma.gust, n, dur=dur, through="wire", strength=0.9, seed=s + 1)
    else:
        y = _b(Ma.gust, n, dur=dur, through="leaves", strength=0.5, seed=s)
    return _end(y, sr, 0.7, fade=0.05, tail=0.2)


# =============================================================================================
# WATER
# =============================================================================================
@sfx("rapids", "Rapids and whirlpools: dense bubbles over a low roar, surging harder with turbulence.",
     turbulence=(0.5, "0..1: more bubbles, more surge, deeper"), dur=(6.0, "s"), seed=(0, "variation"))
def rapids(sr=DEFAULT_SR, turbulence=0.5, dur=6.0, seed=0):
    tu = float(np.clip(turbulence, 0.0, 1.0))
    n, s = _n(dur), int(seed)
    b = _b(Ma.babble, n, dur=dur, flow=0.8 + 1.8 * tu, size=1.0 + 0.8 * tu, turbulence=tu, seed=s)
    w = _b(Ma.waterfall, n, dur=dur, height=1.0 + 6.0 * tu, distance=8.0, seed=s)
    return _end(b + 0.6 * w + (0.1 + 0.8 * tu) * _fit(_rumble(dur, 30.0, 100.0, s), n), sr, 0.8, fade=0.05, tail=0.2)


def fall_time(height: float) -> float:
    """Up and back down: 2 sqrt(2h/g)."""
    return 2.0 * math.sqrt(2.0 * max(float(height), 0.01) / GRAV)


@sfx("fountain", "A fountain: jets hissing up, then the water reaching the basin after its flight, 2 sqrt(2h/g), as a shower of bubbles.",
     jets=(3, "number of jets"), height=(2.0, "m, 1..5: later and harder fall"), dur=(6.0, "s"), seed=(0, "variation"))
def fountain(sr=DEFAULT_SR, jets=3, height=2.0, dur=6.0, seed=0):
    j, h, s = int(np.clip(jets, 1, 30)), float(np.clip(height, 0.2, 10.0)), int(seed)
    n = _n(dur)
    tf = fall_time(h)
    jet = _sos(_b(Ma.steam, n, dur=dur, pressure=0.3, orifice=12.0, wet=0.8, tone=7000.0, seed=s), "highpass", 1000.0)
    fall = 0.6 * _b(Ma.babble, n, dur=dur, flow=min(0.3 * j * math.sqrt(h), 3.0), size=1.0, turbulence=0.3, seed=s) + 0.5 * _nm(_rain("eau", n, min(0.2 * j, 1.0), 0.0, s))
    gate = _pts(n, [(0, 0), (tf, 0), (tf + 0.25, 1), (dur, 1)])
    return _end(0.25 * _pts(n, [(0, 0), (0.1, 1), (dur, 1)]) * jet + fall * gate, sr, 0.75, fade=0.02, tail=0.2)


# Drops in `cavity`: a buried jar about 0.35 m across. Publication: examples/recipes belongs to L21.
RECIPES = {
    "suikinkutsu": {"layers": [{"type": "sfx", "kind": "drops", "params": {"dur": 8.0, "rate": 1.2, "jitter": 0.8, "size": 0.25, "seed": 3}}],
                    "fx": [{"type": "cavity", "shape": "sphere", "size": 0.35, "wall": 0.2, "mix": 0.9, "tail": 1.5}, {"type": "reverb", "mix": 0.1, "size": 0.2}], "normalize": -3.0},
}
