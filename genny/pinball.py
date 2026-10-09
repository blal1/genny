"""Pinball machine foley: solenoids, the knocker, steel-ball impacts, rolling, flipper,
pop bumper, slingshot, spinner, plunger spring, and a solid-state sound-board sweep.

Registered into the sfx REGISTRY (imported at the bottom of sfx.py), so each is used like
any other sfx: {"type": "sfx", "kind": "pop_bumper", "params": {"tone": 1.2}}.

A real machine is mechanism, not music. Every sound here is built from three things:
  - a noise transient (steel on steel, plastic, wood),
  - short *inharmonic* modes of whatever was struck (a steel bar's 1 : 2.76 : 5.40 : 8.93,
    decaying in tens of milliseconds, so it reads as "clank", never as a bell),
  - a boxy wooden-cabinet thump (pitch-dropping low sine + low-mid body resonances).
Solenoids also bounce: the plunger hits its stop, rebounds and hits again a few ms later,
which is most of what makes a coil sound like a coil ("ka-chak").
"""
from __future__ import annotations

import numpy as np

from . import filters as F
from . import osc as O
from .core import DEFAULT_SR, samples
from .env import lfo
from .sfx import sfx

STEEL = (1.0, 2.756, 5.404, 8.933)  # free-free bar modes: brackets, rails, plungers


def _dec(n, tau, sr):
    return np.exp(-np.arange(n) / (max(tau, 1e-5) * sr))


def _place(buf, x, at, sr):
    s = int(round(at * sr))
    e = min(buf.shape[0], s + x.shape[0])
    if e > s:
        buf[s:e] += x[:e - s]


def _modes(f0, ratios, taus, amps, n, sr, rng):
    t = np.arange(n) / sr
    y = np.zeros(n)
    for r, tau, a in zip(ratios, taus, amps):
        f = f0 * r * (1 + rng.uniform(-0.015, 0.015))
        if f < sr / 2 - 500:
            y += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) * _dec(n, tau, sr)
    return y


def _tick(n, sr, center, q, tau, seed):
    return F.bandpass(O.white(n, seed=seed), center, sr, q) * _dec(n, tau, sr) * 2.0


def _thump(n, sr, f, tau, drop=1.0):
    fr = f * (1 + drop * _dec(n, 0.008, sr))
    return np.sin(2 * np.pi * np.cumsum(fr) / sr) * _dec(n, tau, sr)


def _cabinet(x, sr, amount=1.0):
    """Wooden cabinet: two damped low-mid body resonances."""
    if amount <= 0:
        return x
    y = F.peak(x, 170, sr, 1.8, 5 * amount)
    return F.peak(y, 410, sr, 2.5, 3.5 * amount)


def _finish(y, sr, drive=1.3, level=0.85):
    y = np.tanh(y * drive) / np.tanh(drive)
    a = min(samples(0.0004, sr), y.shape[0])
    y[:a] *= np.linspace(0, 1, a)
    r = min(samples(0.008, sr), y.shape[0] // 2)
    y[y.shape[0] - r:] *= np.linspace(1, 0, r)
    peak = np.max(np.abs(y)) + 1e-9
    return y / peak * level


def _solenoid(sr, size, metal, thump, bounce, seed, n=None):
    size = float(np.clip(size, 0, 1))
    rng = np.random.default_rng(int(seed))
    dur = 0.07 + 0.13 * size
    n = n or samples(dur, sr)
    y = np.zeros(n)
    fm = 2600 - 1500 * size  # plunger / bracket ring
    hits = [(0.0, 1.0)]
    if bounce > 0:
        hits.append((0.0035 + 0.006 * size, 0.5 * bounce))
        hits.append((0.009 + 0.01 * size, 0.18 * bounce))
    for i, (at, a) in enumerate(hits):
        m = samples(0.06 + 0.06 * size, sr)
        strike = _tick(m, sr, 3300 - 1300 * size, 1.1, 0.0012 + 0.0012 * size, seed=int(seed) * 17 + i)
        ring = _modes(fm, STEEL, (0.018, 0.009, 0.005, 0.003), (1.0, 0.55, 0.3, 0.15), m, sr, rng)
        _place(y, (strike + ring * 0.35 * metal) * a, at, sr)
    body = _thump(n, sr, 150 - 75 * size, 0.012 + 0.03 * size) * thump * (0.7 + 0.8 * size)
    knock = F.bandpass(O.brown(n, seed=int(seed) + 3, sr=sr), 320, sr, 0.9) * _dec(n, 0.01 + 0.02 * size, sr) * thump * 1.5
    return _cabinet(y + body + knock, sr, 0.6 * thump)


@sfx("solenoid", "Pinball coil / relay: plunger slams its stop and bounces (ka-chak), steel ring + cabinet thump.",
     size=(0.5, "0 = tiny relay .. 1 = big coil (lower, longer)"), metal=(0.5, "0..1 steel ring of plunger/bracket"),
     thump=(0.5, "0..1 wooden cabinet thud"), bounce=(0.6, "0..1 armature rebound hits"), seed=(1, "variation"))
def solenoid(sr=DEFAULT_SR, size=0.5, metal=0.5, thump=0.5, bounce=0.6, seed=1):
    return _finish(_solenoid(sr, size, metal, thump, bounce, seed), sr)


@sfx("knocker", "Pinball knocker (free game / replay): a big coil banging the inside of the wooden cabinet.",
     size=(1.0, "0.5..1.5 cabinet size (bigger = deeper boom)"), rattle=(0.5, "0..1 backbox glass/parts rattle after the bang"),
     seed=(2, "variation"))
def knocker(sr=DEFAULT_SR, size=1.0, rattle=0.5, seed=2):
    rng = np.random.default_rng(int(seed))
    size = float(np.clip(size, 0.3, 2.0))
    n = samples(0.45, sr)
    y = np.zeros(n)
    y += _tick(n, sr, 1700, 0.8, 0.0025, seed=int(seed) + 1) * 1.4
    y += _thump(n, sr, 78 / size, 0.07 * size, drop=1.4) * 1.6
    for f, tau, a in ((175, 0.06, 0.8), (262, 0.045, 0.55), (405, 0.03, 0.4), (640, 0.018, 0.25)):
        y += np.sin(2 * np.pi * f / size * np.arange(n) / sr + rng.uniform(0, 6.28)) * _dec(n, tau * size, sr) * a
    y += _modes(900, STEEL, (0.012, 0.007, 0.004, 0.002), (0.5, 0.3, 0.15, 0.1), n, sr, rng)
    y += F.bandpass(O.brown(n, seed=int(seed) + 5, sr=sr), 220, sr, 0.8) * _dec(n, 0.05 * size, sr) * 1.5
    if rattle > 0:
        for _ in range(7):
            at = rng.uniform(0.012, 0.09)
            m = samples(0.02, sr)
            click = _tick(m, sr, rng.uniform(2500, 4500), 2.0, 0.0012, seed=int(rng.integers(1 << 30)))
            _place(y, click * rattle * rng.uniform(0.05, 0.16), at, sr)
    return _finish(_cabinet(y, sr, 1.0), sr, drive=1.6)


def _ball(sr, surface, force, seed, n):
    rng = np.random.default_rng(int(seed))
    force = float(np.clip(force, 0, 1))
    y = np.zeros(n)
    if surface == "metal":  # wire guide, rail, post: short clank
        y += _tick(n, sr, 4200, 1.0, 0.0008, seed=int(seed))
        y += _modes(1650, (1.0, 2.31, 3.87, 5.62), (0.022, 0.013, 0.008, 0.005), (0.6, 0.45, 0.3, 0.2), n, sr, rng)
        y += _thump(n, sr, 260, 0.006, drop=0.8) * 0.4
    elif surface == "wood":  # apron, playfield edge: tock
        y += _tick(n, sr, 1500, 0.8, 0.0015, seed=int(seed))
        y += _thump(n, sr, 420, 0.008, drop=0.5) * 0.8
        y += F.bandpass(O.brown(n, seed=int(seed) + 1, sr=sr), 700, sr, 1.2) * _dec(n, 0.006, sr) * 2.0
    elif surface == "rubber":  # rubber ring, post, flipper rubber: dull thup
        y += _thump(n, sr, 180, 0.014, drop=1.5)
        y += F.lowpass(O.white(n, seed=int(seed)), 700, sr) * _dec(n, 0.005, sr) * 1.2
        y += _tick(n, sr, 2600, 1.5, 0.0006, seed=int(seed) + 2) * 0.25
    elif surface == "plastic":  # drop target, ramp, cap: hollow clack
        y += _tick(n, sr, 2300, 1.4, 0.0018, seed=int(seed))
        y += _modes(1150, (1.0, 1.62, 2.34), (0.011, 0.006, 0.004), (0.55, 0.35, 0.2), n, sr, rng)
        y += _thump(n, sr, 320, 0.006, drop=0.6) * 0.5
    else:
        raise ValueError("surface must be metal|wood|rubber|plastic")
    y = F.lowpass(y, 1500 + 10000 * force, sr)
    return y * (0.3 + 0.7 * force)


@sfx("steel_ball", "Steel pinball striking a surface: metal (rail/post clank), wood (tock), rubber (thup), plastic (clack).",
     surface=("metal", "metal|wood|rubber|plastic"), force=(0.7, "0..1 impact speed (louder, brighter)"), seed=(3, "variation"))
def steel_ball(sr=DEFAULT_SR, surface="metal", force=0.7, seed=3):
    y = _ball(sr, surface, force, seed, samples(0.13, sr))
    return _finish(y, sr, level=0.5 + 0.35 * float(np.clip(force, 0, 1)))


@sfx("ball_roll", "Steel ball rolling on a wooden playfield (loopable with \"loop\"): rumble, contact hiss, insert ticks.",
     speed=(0.5, "0..1 ball speed"), dur=(2.0, "s"), seed=(4, "variation"))
def ball_roll(sr=DEFAULT_SR, speed=0.5, dur=2.0, seed=4):
    rng = np.random.default_rng(int(seed))
    speed = float(np.clip(speed, 0, 1))
    n = samples(dur, sr)
    rumble = F.lowpass(O.brown(n, seed=int(seed), sr=sr), 220 + 900 * speed, sr, order=2)
    rumble = F.highpass(rumble, 45, sr)
    rumble = F.peak(F.peak(rumble, 160, sr, 1.5, 6), 330, sr, 2.0, 4)
    rumble /= np.max(np.abs(rumble)) + 1e-9
    hiss = F.bandpass(O.white(n, seed=int(seed) + 1), 2200 + 1800 * speed, sr, 0.9)
    hiss *= 0.02 + 0.05 * speed
    ticks = np.zeros(n)
    rate = 3 + 25 * speed
    t = rng.exponential(1 / rate)
    while t < dur - 0.02:
        m = samples(0.015, sr)
        tk = _tick(m, sr, rng.uniform(900, 2200), 1.2, 0.0012, seed=int(rng.integers(1 << 30)))
        _place(ticks, tk * rng.uniform(0.03, 0.12), t, sr)
        t += rng.exponential(1 / rate)
    flutter = 1 + lfo(dur, 7 + 22 * speed, sr, depth=0.12)[:n]
    y = (rumble * 0.8 + hiss + ticks) * flutter
    return y / (np.max(np.abs(y)) + 1e-9) * 0.8


@sfx("chirp", "Solid-state sound-board sweep (80s pinball/arcade): pitch sweep with warble, stepped pitch, DAC grit.",
     start=(1200, "Hz"), end=(300, "Hz"), dur=(0.15, "s"), wave=("square", "square|pulse|saw|triangle|sine"),
     warble=(0.0, "pitch LFO Hz (0 = off)"), depth=(2.0, "warble depth in semitones"),
     steps=(0, "quantize the sweep into N pitch steps (0 = smooth)"), bits=(0, "DAC bit depth, e.g. 4..6 (0 = clean)"))
def chirp(sr=DEFAULT_SR, start=1200, end=300, dur=0.15, wave="square", warble=0.0, depth=2.0, steps=0, bits=0):
    n = samples(dur, sr)
    k = np.linspace(0, 1, n)
    if steps and steps > 0:
        k = np.floor(k * int(steps)) / max(1, int(steps) - 1)
        k = np.minimum(k, 1.0)
    start, end = max(float(start), 1.0), max(float(end), 1.0)
    f = start * (end / start) ** k
    if warble:
        f = f * 2 ** (depth / 12 * lfo(dur, warble, sr)[:n])
    x = O.osc(wave, f, n, sr, width=0.3)
    if bits and bits > 0:
        q = 2 ** (int(bits) - 1)
        x = np.round(x * q) / q
    x = F.lowpass(x, 6000, sr)
    env = np.ones(n)
    a, r = min(samples(0.002, sr), n), min(samples(0.012, sr), n)
    env[:a] = np.linspace(0, 1, a)
    env[n - r:] *= np.linspace(1, 0, r)
    return x * env * 0.5


@sfx("flipper", "Pinball flipper coil: heavy solenoid kick, bat slapping its stop, brief coil buzz.",
     force=(1.0, "0..1"), buzz=(0.3, "0..1 AC coil hum after the kick"), seed=(8, "variation"))
def flipper(sr=DEFAULT_SR, force=1.0, buzz=0.3, seed=8):
    force = float(np.clip(force, 0, 1))
    n = samples(0.2, sr)
    y = _solenoid(sr, 0.85, 0.35, 0.9, 0.7, seed, n)
    stop = _thump(n, sr, 105, 0.02, drop=1.2) * 0.8 + _ball(sr, "rubber", 0.8, seed + 1, n) * 0.6
    _place(y, stop, 0.006, sr)
    if buzz > 0:
        m = samples(0.09, sr)
        hum = F.bandpass(O.saw(120, m, sr), 480, sr, 0.8) * _dec(m, 0.03, sr) * 0.35 * buzz
        _place(y, hum, 0.01, sr)
    return _finish(F.lowpass(y, 2500 + 7000 * force, sr) * (0.4 + 0.6 * force), sr, level=0.5 + 0.35 * force)


@sfx("pop_bumper", "Pop bumper: coil yanks the ring down (THOCK), rubber ring slaps the ball, hollow cap pock.",
     tone=(1.0, "0.6..1.6 pitch of the cap/body (tell bumpers apart)"), seed=(7, "variation"))
def pop_bumper(sr=DEFAULT_SR, tone=1.0, seed=7):
    rng = np.random.default_rng(int(seed))
    tone = float(np.clip(tone, 0.4, 2.5))
    n = samples(0.22, sr)
    y = _solenoid(sr, 0.75, 0.45, 0.9, 0.6, seed, n)
    y += _thump(n, sr, 210 * tone, 0.02, drop=1.2) * 0.9
    y += _modes(760 * tone, (1.0, 1.58, 2.41), (0.028, 0.014, 0.008), (0.55, 0.35, 0.15), n, sr, rng)
    y += F.bandpass(O.white(n, seed=int(seed) + 9), 950 * tone, sr, 2.0) * _dec(n, 0.012, sr) * 1.2
    _place(y, _ball(sr, "rubber", 0.9, seed + 2, n) * 0.5, 0.003, sr)
    return _finish(y, sr, drive=1.5)


@sfx("slingshot", "Slingshot kicker: coil fires, rubber band snaps the ball away (thwap) and twangs.",
     tone=(1.0, "0.6..1.6 band pitch"), seed=(9, "variation"))
def slingshot(sr=DEFAULT_SR, tone=1.0, seed=9):
    tone = float(np.clip(tone, 0.4, 2.5))
    n = samples(0.2, sr)
    y = _solenoid(sr, 0.6, 0.4, 0.7, 0.5, seed, n)
    snap = F.bandpass(O.white(n, seed=int(seed) + 4), 1100 * tone, sr, 0.9) * _dec(n, 0.006, sr) * 1.6
    snap += _thump(n, sr, 260 * tone, 0.015, drop=2.0)
    band = np.sin(2 * np.pi * 95 * tone * np.arange(n) / sr) * _dec(n, 0.035, sr) * 0.5
    band *= 1 + 0.5 * np.sin(2 * np.pi * 7 * np.arange(n) / sr)
    _place(y, snap + band, 0.002, sr)
    return _finish(y, sr, drive=1.4)


@sfx("spinner", "Spinner flap: thin metal flap ticking past its wire; spins > 1 = a decelerating whirr.",
     spins=(1, "flap passes"), rate=(18, "passes per second at the start"), tone=(1.0, "0.6..1.6 flap pitch"), seed=(6, "variation"))
def spinner(sr=DEFAULT_SR, spins=1, rate=18, tone=1.0, seed=6):
    rng = np.random.default_rng(int(seed))
    spins = max(1, int(spins))
    times, t, gap = [], 0.0, 1.0 / max(float(rate), 1.0)
    for _ in range(spins):
        times.append(t)
        t += gap
        gap *= 1.07
    n = samples(times[-1] + 0.05, sr)
    y = np.zeros(n)
    m = samples(0.04, sr)
    for i, at in enumerate(times):
        tk = _tick(m, sr, 5200 * tone, 1.0, 0.0006, seed=int(seed) + i)
        tk += _modes(2900 * tone, (1.0, 2.42, 4.13), (0.008, 0.005, 0.003), (0.5, 0.3, 0.15), m, sr, rng)
        tk += _thump(m, sr, 700 * tone, 0.003, drop=0.3) * 0.3
        _place(y, tk * (0.97 ** i) * rng.uniform(0.85, 1.0), at, sr)
    return _finish(y, sr, level=0.8)


@sfx("spring", "Plunger spring: 'pull' = stretched-coil creak tick, 'release' = rod slam + coiled-spring twang (boing).",
     action=("release", "pull|release"), tension=(0.7, "0..1 (higher = tighter, higher twang)"), seed=(5, "variation"))
def spring(sr=DEFAULT_SR, action="release", tension=0.7, seed=5):
    rng = np.random.default_rng(int(seed))
    tension = float(np.clip(tension, 0, 1))
    f0 = 70 + 110 * tension
    if action == "pull":
        n = samples(0.09, sr)
        y = _tick(n, sr, 1900, 1.2, 0.0015, seed=int(seed)) * 0.8
        y += F.bandpass(O.white(n, seed=int(seed) + 1), 1200 + 800 * tension, sr, 2.0) * _dec(n, 0.012, sr) * 0.7
        twang_dur, twang_amp = 0.07, 0.35
    elif action == "release":
        n = samples(0.65, sr)
        y = _solenoid(sr, 0.9, 0.5, 1.0, 0.4, seed, n)
        twang_dur, twang_amp = 0.35, 0.8
    else:
        raise ValueError("action must be pull|release")
    # dispersive coil: stretched partials, highs decay first, tiny pitch settle
    t = np.arange(n) / sr
    settle = 1 + 0.06 * np.exp(-t / 0.03)
    twang = np.zeros(n)
    for k in range(1, 9):
        fk = f0 * k * (1 + 0.012 * k * k)
        if fk * 1.1 > sr / 2 - 500:
            break
        ph = 2 * np.pi * np.cumsum(fk * settle) / sr + rng.uniform(0, 6.28)
        twang += np.sin(ph) * _dec(n, twang_dur / (1 + 0.5 * k), sr) / k ** 0.6
    twang = F.bandpass(twang, 600 + 600 * tension, sr, 0.6)
    y += twang / (np.max(np.abs(twang)) + 1e-9) * twang_amp
    return _finish(y, sr)
