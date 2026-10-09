"""Living things: mammals, insects, birds, frogs, body sounds, crowds, footstep sequences, whistling.

Catalog wrappers over the kernels in ``genny.physical.voice`` / ``footsteps`` / ``particles`` /
``modal`` plus the few models those do not have (cricket, wingbeat, FM syrinx with a song grammar,
heartbeat, applause, whistle). Kernels run at 48 kHz (``voice.SR``); every entry resamples at the
boundary (``scipy.signal.resample_poly``), removes DC, fades both ends and peak-normalises.

Sources (research notes in ``synthgen/out/research``)
-----------------------------------------------------
* Farnell, *Designing Sound*: Practical 29 Mammals (flapping source, tract comb, articulation
  envelope, lion arc 30 -> 240 -> 120 Hz, cat ~500 Hz, vowel table; research 03 §3.1), Practical 27
  Insects (field cricket: 7 pulses per 0.12 s burst every 0.7 s, 17 ms pulse period, 4.5 kHz
  falling to 3.9 kHz, wings at 4.4 / 4.6 kHz, 2nd harmonic 1/3; housefly harmonic table fig 50.3,
  "as much as 20 percent" common wing wobble; mosquito / bee 3 % pulse through bp Q12; critters,
  chirper, japanese cricket, random on/off gate; research 03 §3.4), Practical 28 Birds (pulse
  oscillator 1/(1+(w x)^2), two bronchial oscillators, ring-mod crossfade, FM, two tracheal
  band-passes Q 2-5, beak high-pass; research 03 §3.3), frog / rattler patches (§3.2, §3.4),
  Practical 26 Footsteps (GRF, walk / creep / run, quadruped multiphase timebase; §2).
* Song syntax element -> syllable -> phrase -> song, syllables 5-300 ms, oscine f0 "typically
  between 3 and 5 kHz", two-voice syrinx (Fagerlund 2004 via research g15 §5); raven trachea
  resonances 1200 / 2200 Hz and the valve model (Smyth & Smith DAFx-02), tube sizes (Kahrs &
  Avanzini DAFx-01) (research 06 §9).
* Dolbear, "The cricket as a thermometer" (1897): T_F = 50 + (N - 40)/4, N chirps per minute.
* LovelyHeart (N. Ariutti 2016, Pd): lub 63 Hz x 5 cycles, dub 42 Hz x 3 cycles + 2nd harmonic
  0.5, envelope (1 - r)^2, murmur = squared bp 50 Hz Q5 noise (research 09 C2). S1-S2 interval:
  Weissler, Harris & Schoenfeld (1968) LVET = 0.413 - 0.0017 HR.
* Farnell AM07: claps = filtered noise bursts, beyond ~20 claps/s blend granular noise
  (research 05 §summary 6). Cook, *Real Sound Synthesis*: whisper = noise through the vowel
  formants, every footstep differs (research 04 §3, §9).
* Visell et al.: 75-125 steps/min; The Sounding Object ch. 6: walking = legato (overlap), running
  = staccato (flight gap), never loop one step (research 06 §7.1, g05 §10).
* Ported from the sibling ``procedural`` project (creatures.py ``wings``, ``_growl_synth``;
  body.py ``grunt``) - (c) 2025 Chris Nash, Klang Open Licence 1.0, see
  ``genny/physical/KLANG_LICENSE.txt``.

Lines marked ``# UNSOURCED`` are tuning with no number in any source.
"""
from __future__ import annotations

import math

import numba
import numpy as np
from scipy import signal as _sig

from . import filters as F
from .core import DEFAULT_SR
from .instruments import instrument
from .notes import to_hz
from .physical import footsteps as S
from .physical import modal as M
from .physical import particles as P
from .physical import voice as V
from .sfx import sfx

KSR = V.SR                      # kernel rate, 48 kHz
TWO_PI = 2.0 * math.pi
C_AIR = 340.0                   # Farnell's value in F = c/4l


# ============================================================================================
# helpers
# ============================================================================================

def _rng(seed, salt: int = 0) -> np.random.Generator:
    return np.random.default_rng([abs(int(seed)), salt])


def _n(dur: float) -> int:
    return max(64, int(round(float(dur) * KSR)))


def _fin(y, sr: int, peak: float = 0.7, fade_ms: float = 5.0, hp: float = 20.0) -> np.ndarray:
    """48 kHz kernel output -> catalog buffer: DC removed, resampled, faded, peak-normalised."""
    y = np.nan_to_num(np.asarray(y, dtype=np.float64))
    if hp:
        y = _sig.sosfilt(_sig.butter(2, hp, "highpass", fs=KSR, output="sos"), y)
    if int(sr) != KSR:
        g = math.gcd(int(sr), KSR)
        y = _sig.resample_poly(y, int(sr) // g, KSR // g)
    m = min(len(y) // 2, max(2, int(fade_ms * 1e-3 * sr)))
    w = np.sin(0.5 * np.pi * np.arange(m) / m) ** 2
    y[:m] *= w
    y[-m:] *= w[::-1]
    pk = float(np.max(np.abs(y)))
    return y * (peak / pk) if pk > 1e-12 else y


def _lp(x, fc):
    return F.lowpass(x, float(min(fc, 0.45 * KSR)), KSR)


def _hp(x, fc):
    return F.highpass(x, float(fc), KSR)


def _bpf(x, fc, q):
    return F.bandpass(x, float(fc), KSR, q)


def _slow(n: int, rng, hz: float) -> np.ndarray:
    """Twice low-passed noise scaled to +-1 (Farnell's ``lop(lop(noise))`` control signal)."""
    x = rng.uniform(-1.0, 1.0, n + KSR)
    x = V._pd_lop(V._pd_lop(x, hz, float(KSR)), hz, float(KSR))[KSR:]
    return x / (np.max(np.abs(x)) + 1e-12)


def _place(buf: np.ndarray, x: np.ndarray, t: float, g: float = 1.0) -> None:
    i = max(0, int(round(t * KSR)))
    k = min(len(x), len(buf) - i)
    if k > 0:
        buf[i:i + k] += g * x[:k]


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.max(np.abs(x)) + 1e-12)


def _arc(n: int, pts) -> np.ndarray:
    """Contour from ``[(u in 0..1, value), ...]``, lightly smoothed."""
    u = np.array([p[0] for p in pts], float)
    v = np.array([p[1] for p in pts], float)
    y = np.interp(np.linspace(0.0, 1.0, n), u, v)
    k = max(1, int(0.01 * KSR))
    return np.convolve(np.pad(y, (k, k), mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid")


def _choice(value, options, what):
    if value not in options:
        raise ValueError(f"unknown {what} {value!r}; choose from {list(options)}")
    return value


# ============================================================================================
# mammals: Farnell animal generator (flapping source -> tract comb + vowel formants)
# ============================================================================================

def _voc(f0, env, rng, *, tract_m: float, harsh: float = 0.3, vowel=None, vq: float = 9.0, vmix: float = 0.6,
         jitter: float = 0.01, shimmer: float = 0.06, sub: float = 0.0, wrong: float = 0.0) -> np.ndarray:
    """Farnell Practical 29 generator. ``f0``/``env`` per sample; ``harsh`` maps to ripple
    (1 -> 4), cord width (2 -> 12) and noisiness (0.05 -> 0.45) as in ``voice.roar``; tract =
    parallel band-pass comb at (2k-1) c/4l plus, when ``vowel`` is given, three formants from
    Farnell's vowel table scaled by 0.17 m / l. ``wrong`` adds a second shorter tube (procedural
    ``_growl_synth``). UNSOURCED: the harsh mapping, vq (formant Q; the schwa box prints 34 for
    humans, too ringing for animals), vmix."""
    n = len(f0)
    a = env / (float(np.max(env)) + 1e-12)
    f0j, amp = V.roughness(np.maximum(f0, 5.0), rng, jitter=jitter, shimmer=shimmer, subharmonic=sub)
    src = V.flapping_source(f0j, ripple_env=1.0 + 3.0 * harsh * a, width_env=2.0 + 10.0 * harsh * a,
                            noisiness_env=0.05 + 0.4 * harsh * a, rng=rng) * amp
    y = V.vocal_tract_comb(src, tract_m, q=3.0 + 3.0 * a)
    if wrong:
        y = y + 0.6 * V.vocal_tract_comb(src, tract_m * wrong, q=4.0)
    if vowel is not None:
        Fm = np.minimum(V._param_path(vowel, n, "farnell", "freqs") * (0.17 / tract_m), 0.4 * KSR)
        y = (1.0 - vmix) * y + vmix * vq * V.formant_filter(
            src, formants=dict(freqs=Fm, bws=Fm / vq, gains=np.ones(3)))
    return V._pd_hip(np.ascontiguousarray(y), 20.0, float(KSR)) * env


def _vowel_keys(path: str, dur: float):
    keys = [k for k in str(path).split("-") if k]
    for k in keys:
        _choice(k, V.FARNELL_VOWELS, "vowel")
    if len(keys) == 1:
        return keys[0]
    return [(dur * i / (len(keys) - 1), k) for i, k in enumerate(keys)]


def _dark(y, bright: float = 1.0):
    """Nothing here needs energy above ~5 kHz; ``bright`` opens the top."""
    return _lp(y, 4500.0 * max(0.2, float(bright)))


@sfx("roar", "Big-cat roar (Farnell lion arc 30->240->120 Hz, flapping cords into a long tract).",
     size=(1.0, "body scale 0.4..4: f0 = arc/size, tract = 0.4 m x size"), aggression=(0.7, "0 smooth .. 1 snarling"),
     dur=(2.5, "s"), seed=(0, "variation"))
def roar(sr=DEFAULT_SR, size=1.0, aggression=0.7, dur=2.5, seed=0):
    size = float(np.clip(size, 0.4, 4.0))
    rng = _rng(seed, 1)
    n = _n(dur)
    rise = 0.45                                                    # UNSOURCED (voice.roar default)
    a = V.articulation_env(n, rise, offset=0.0)
    nr = int(round(n * rise))
    f0 = np.empty(n)
    f0[:nr] = 30.0 + (240.0 - 30.0) * a[:nr]
    f0[nr:] = 120.0 + (240.0 - 120.0) * a[nr:]
    ag = float(np.clip(aggression, 0.0, 1.0))
    y = _voc(f0 / size, a, rng, tract_m=0.4 * size, harsh=0.15 + 0.85 * ag, jitter=0.005 + 0.03 * ag,
             shimmer=0.3 * ag, sub=0.3 * ag)
    return _fin(_dark(y), sr)


@sfx("growl", "Low threatening growl: slow wobbling pulses, period doubling, noisy cords.",
     size=(1.0, "body scale: f0 = pitch/size, tract 0.25 m x size"), aggression=(0.6, "0..1 harshness and subharmonics"),
     pitch=(70.0, "Hz at size 1"), dur=(1.5, "s"), seed=(0, "variation"))
def growl(sr=DEFAULT_SR, size=1.0, aggression=0.6, pitch=70.0, dur=1.5, seed=0):
    size = float(np.clip(size, 0.3, 4.0))
    rng = _rng(seed, 2)
    n = _n(dur)
    ag = float(np.clip(aggression, 0.0, 1.0))
    env = V.articulation_env(n, 0.3, offset=0.0) ** 0.5
    f0 = to_hz(pitch) / size * (1.0 + 0.12 * _slow(n, rng, 3.0)) * (0.85 + 0.25 * env)   # UNSOURCED wobble
    y = _voc(f0, env * (1.0 + 0.25 * _slow(n, rng, 7.0)), rng, tract_m=0.25 * size, harsh=0.4 + 0.6 * ag,
             jitter=0.03, shimmer=0.25, sub=0.6 * ag, wrong=0.7)
    return _fin(_dark(y), sr)


_DOGS = {  # UNSOURCED: f0 peak Hz, tract m, bark s (sources give no dog numbers)
    "small": (720.0, 0.09, 0.10), "medium": (430.0, 0.15, 0.15), "large": (240.0, 0.22, 0.21)}


@sfx("bark", "Dog barks: short arched pulses bursts, 'wa-u' vowel path.",
     size=("medium", "small|medium|large"), count=(2, "barks"), aggression=(0.6, "0..1"),
     pitch=(0.0, "f0 peak in Hz, 0 = from size"), seed=(0, "variation"))
def bark(sr=DEFAULT_SR, size="medium", count=2, aggression=0.6, pitch=0.0, seed=0):
    fpk, tract, d = _DOGS[_choice(size, _DOGS, "size")]
    fpk = to_hz(pitch) if pitch else fpk
    rng = _rng(seed, 3)
    count = int(np.clip(count, 1, 40))
    out = np.zeros(_n(count * (d + 0.4) + 0.3))
    t = 0.02
    for _ in range(count):
        dd = d * rng.uniform(0.9, 1.15)
        n = _n(dd)
        env = V.articulation_env(n, 0.18, offset=0.0)
        f0 = fpk * rng.uniform(0.94, 1.06) * _arc(n, [(0, 0.6), (0.2, 1.0), (0.6, 0.85), (1, 0.5)])
        y = _voc(f0, env, rng, tract_m=tract, harsh=0.35 + 0.6 * aggression, vowel=_vowel_keys("A-u", dd),
                 shimmer=0.2, jitter=0.02, sub=0.3 * aggression)
        _place(out, _unit(y), t, rng.uniform(0.8, 1.0))
        t += dd + rng.uniform(0.16, 0.32)                          # UNSOURCED gap
    return _fin(_dark(out[:_n(t + 0.1)]), sr)


@sfx("meow", "Cat meow: ~500 Hz smooth cords (Farnell), vowel path i-ae-A-u through a short tract.",
     size=(1.0, "body scale (0.6 kitten .. 2 big cat)"), pitch=(550.0, "f0 peak Hz at size 1"), dur=(0.9, "s"),
     seed=(0, "variation"))
def meow(sr=DEFAULT_SR, size=1.0, pitch=550.0, dur=0.9, seed=0):
    size = float(np.clip(size, 0.4, 4.0))
    rng = _rng(seed, 4)
    n = _n(dur)
    env = V.articulation_env(n, 0.3, offset=0.0) ** 0.7
    f0 = to_hz(pitch) / size * _arc(n, [(0, 0.72), (0.3, 1.0), (0.6, 0.96), (1, 0.68)]) * (1 + 0.01 * _slow(n, rng, 6.0))
    y = _voc(f0, env, rng, tract_m=0.085 * size, harsh=0.12, vowel=_vowel_keys("i:-ae-A-u", dur), vmix=0.75)
    return _fin(_dark(y), sr)


@sfx("purr", "Cat purr: glottal pulses near 25 Hz through the tract, breathing in and out.",
     rate=(25.0, "pulses per second (mean)"), size=(1.0, "body scale"), dur=(3.0, "s"), seed=(0, "variation"))
def purr(sr=DEFAULT_SR, rate=25.0, size=1.0, dur=3.0, seed=0):
    rng = _rng(seed, 5)
    n = _n(dur)
    t = np.arange(n) / KSR
    br = np.sin(TWO_PI * t / 1.7 + rng.uniform(0, TWO_PI))        # UNSOURCED: 1.7 s breath cycle
    f = float(np.clip(rate, 5.0, 80.0)) * (1.0 + 0.04 * np.tanh(4.0 * br)) * (1.0 + 0.01 * _slow(n, rng, 5.0))
    amp = 0.55 + 0.45 * np.abs(np.tanh(4.0 * br)) ** 0.5           # dip at each breath reversal
    src = V.raised_cosine_pulse(f, width_env=5.0)
    src = src + 0.06 * rng.uniform(-1, 1, n) * np.maximum(src, 0.0)
    y = V.vocal_tract_comb(src, 0.085 * float(np.clip(size, 0.4, 4.0)), q=2.5) + 0.15 * _lp(src, 300.0)
    return _fin(_lp(y * amp, 2200.0), sr)


@sfx("moo", "Cow moo: closed-mouth 'mm' opening to 'oo', long tract, slight period roughness.",
     size=(1.0, "body scale (0.6 calf .. 1.5 bull)"), pitch=(115.0, "f0 Hz at size 1"), dur=(1.8, "s"),
     seed=(0, "variation"))
def moo(sr=DEFAULT_SR, size=1.0, pitch=115.0, dur=1.8, seed=0):
    size = float(np.clip(size, 0.4, 4.0))
    rng = _rng(seed, 6)
    n = _n(dur)
    env = V.articulation_env(n, 0.25, offset=0.0) ** 0.6
    f0 = to_hz(pitch) / size * _arc(n, [(0, 0.8), (0.25, 1.0), (0.7, 1.06), (1, 0.74)])
    y = _voc(f0, env, rng, tract_m=0.35 * size, harsh=0.3, vowel=_vowel_keys("@-u-U", dur), shimmer=0.12,
             jitter=0.015, sub=0.15)
    y = F.lowpass(y, _arc(n, [(0, 350.0), (0.22, 3200.0), (1, 2200.0)]) / size ** 0.5, KSR)   # lips opening
    return _fin(y, sr)


@sfx("howl", "Wolf howl: long smooth rising then slowly falling tone, 'a-oo'.",
     size=(1.0, "body scale"), pitch=(420.0, "f0 plateau Hz at size 1"), dur=(3.0, "s"), seed=(0, "variation"))
def howl(sr=DEFAULT_SR, size=1.0, pitch=420.0, dur=3.0, seed=0):
    size = float(np.clip(size, 0.4, 4.0))
    rng = _rng(seed, 7)
    n = _n(dur)
    env = V.articulation_env(n, 0.2, offset=0.0) ** 0.5
    f0 = to_hz(pitch) / size * _arc(n, [(0, 0.55), (0.14, 0.94), (0.3, 1.0), (0.8, 0.96), (1, 0.7)])
    f0 = f0 * (1.0 + 0.006 * np.sin(TWO_PI * 5.0 * np.arange(n) / KSR) * np.linspace(0, 1, n))
    y = _voc(f0, env, rng, tract_m=0.2 * size, harsh=0.06, vowel=_vowel_keys("A-u-u", dur), vmix=0.7,
             jitter=0.004, shimmer=0.03)
    return _fin(_dark(y, 0.8), sr)


@sfx("monster", "Monster voice: multi-pulse cords with subharmonics through a scaled double tract, slow vowel drift.",
     size=(2.5, "body scale: tract 0.17 m x size, f0 = pitch/size"), aggression=(0.8, "0..1"),
     pitch=(220.0, "f0 Hz at size 1"), dur=(2.0, "s"), seed=(0, "variation"))
def monster(sr=DEFAULT_SR, size=2.5, aggression=0.8, pitch=220.0, dur=2.0, seed=0):
    size = float(np.clip(size, 0.5, 8.0))
    rng = _rng(seed, 8)
    n = _n(dur)
    ag = float(np.clip(aggression, 0.0, 1.0))
    env = V.articulation_env(n, 0.35, offset=0.0) ** 0.6
    f0 = to_hz(pitch) / size * _arc(n, [(0, 0.55), (0.35, 1.0), (1, 0.6)]) * (1 + 0.05 * _slow(n, rng, 4.0))
    keys = list(V.FARNELL_VOWELS)
    path = [(dur * i / 3.0, keys[int(rng.integers(len(keys)))]) for i in range(4)]   # Farnell: slow F1/F2 drift
    y = _voc(f0, env, rng, tract_m=0.17 * size, harsh=0.5 + 0.5 * ag, vowel=path, vmix=0.45, jitter=0.04,
             shimmer=0.3, sub=0.7 * ag, wrong=0.7)
    y = y + ag * 0.5 * y * np.cos(TWO_PI * np.cumsum(0.5 * f0) / KSR)   # Puckette octave divider (f0/2 partials)
    return _fin(_dark(y), sr)


_CONTOURS = {
    "flat": [(0, 0), (1, 0)], "rise": [(0, -1), (1, 1)], "fall": [(0, 1), (1, -1)],
    "arch": [(0, -1), (0.4, 1), (1, -0.6)], "dip": [(0, 1), (0.5, -1), (1, 0.6)],
    "wobble": [(0, 0), (0.125, 1), (0.375, -1), (0.625, 1), (0.875, -1), (1, 0)]}


@sfx("animal", "Generic animal call (Farnell generator): size, pitch contour, roughness and a vowel path.",
     size=(1.0, "body scale: f0 = pitch/size, tract 0.17 m x size (formants / size)"),
     pitch=(200.0, "f0 Hz at size 1"), contour=("arch", "flat|rise|fall|arch|dip|wobble"),
     excursion=(0.3, "pitch contour depth as a fraction of f0"), roughness=(0.3, "0 hum .. 1 snarl"),
     vowels=("A-u", "path of Farnell vowels joined by '-': i: I E ae ^ u U @ A"), dur=(0.8, "s"),
     seed=(0, "variation"))
def animal(sr=DEFAULT_SR, size=1.0, pitch=200.0, contour="arch", excursion=0.3, roughness=0.3, vowels="A-u",
           dur=0.8, seed=0):
    size = float(np.clip(size, 0.2, 8.0))
    rng = _rng(seed, 9)
    n = _n(dur)
    r = float(np.clip(roughness, 0.0, 1.0))
    env = V.articulation_env(n, 0.3, offset=0.0) ** 0.6
    c = _arc(n, _CONTOURS[_choice(contour, _CONTOURS, "contour")])
    f0 = to_hz(pitch) / size * (1.0 + float(np.clip(excursion, 0.0, 0.9)) * c)
    y = _voc(f0, env, rng, tract_m=0.17 * size, harsh=r, vowel=_vowel_keys(vowels, dur), jitter=0.003 + 0.03 * r,
             shimmer=0.02 + 0.25 * r, sub=0.4 * r * r)
    return _fin(_dark(y, 1.1), sr)


# ============================================================================================
# insects
# ============================================================================================

def dolbear_rate(temp_c: float) -> float:
    """Chirps per second at ``temp_c`` degC from Dolbear's law T_F = 50 + (N - 40)/4 (N per minute)."""
    return max((7.2 * float(temp_c) - 32.0) / 60.0, 0.1)


_CRICKETS = {
    # carrier Hz, fall, pulses per chirp, pulse period s, wing split, rate factor on Dolbear
    "field": (4500.0, 0.133, 7, 0.017, 0.022, 1.0),          # Farnell Practical 27
    "tree": (2900.0, 0.05, 8, 0.02, 0.01, 1.0),              # UNSOURCED: snowy tree cricket numbers
    "bell": (3000.0, 0.0, 0, 0.0, 0.0, 2.52),                # Farnell "japanese cricket": 3.6 Hz when field = 1.43
}


def _cricket(n: int, rate: float, rng, species: str, bright: float, phase0: float = 0.0) -> np.ndarray:
    fc, fall, pulses, ps, split, _k = _CRICKETS[species]
    t = np.arange(n) / KSR
    cph = np.mod(rate * t + phase0, 1.0)
    if species == "bell":       # osc(3000 + 300 osc(120)) x ((osc(rate) + 1)/2)^4
        car = np.sin(TWO_PI * (fc * t) + (300.0 / 120.0) * np.sin(TWO_PI * 120.0 * t))
        return car * ((np.cos(TWO_PI * cph) + 1.0) / 2.0) ** 4
    ps = min(ps, 0.8 / (rate * pulses))
    tc = cph / rate
    k = np.floor(tc / ps)
    u = tc / ps - k
    on = k < pulses
    # Farnell: amplitude (n + 2)/9 rising over the burst; parabolic attack, tail between exp and linear
    e = np.where(on, (k + 2.0) / (pulses + 2.0) * np.minimum(u / 0.3, 1.0) ** 2 * (1.0 - u) ** 1.5, 0.0)
    ph = TWO_PI * np.cumsum(fc * (1.0 - fall * np.where(on, u, 0.0))) / KSR
    y = np.zeros(n)
    for w in (1.0 - split, 1.0 + split):       # two wings at 4.4 and 4.6 kHz
        y += np.sin(w * ph) + bright / 3.0 * np.sin(2.0 * w * ph) + bright / 30.0 * np.sin(3.0 * w * ph)
    return y * e * (1.0 + 0.15 * _slow(n, rng, 5.0))


@sfx("cricket", "Cricket chirps; chirp rate follows temperature (Dolbear's law).",
     species=("field", "field (7-pulse 4.5 kHz) | tree (lower, Dolbear's snowy tree cricket) | bell (continuous trill)"),
     temp=(18.0, "degC: chirps/min = 7.2 T - 32 (x2.52 for bell)"), rate=(0.0, "chirps per second, 0 = from temp"),
     bright=(0.5, "0..1 level of the 9 kHz 2nd harmonic (1 = Farnell)"), dur=(3.0, "s"), seed=(0, "variation"))
def cricket(sr=DEFAULT_SR, species="field", temp=18.0, rate=0.0, bright=0.5, dur=3.0, seed=0):
    _choice(species, _CRICKETS, "species")
    r = float(rate) if rate else dolbear_rate(temp) * _CRICKETS[species][5]
    y = _cricket(_n(dur), float(np.clip(r, 0.1, 12.0)), _rng(seed, 10), species, float(np.clip(bright, 0, 1)))
    return _fin(y, sr, peak=0.5, hp=300.0)


@sfx("cicada", "Cicada: noise-FM click train through timbal resonances with 12 Hz pulsing (Farnell circada2).",
     rate=(120.0, "click rate Hz"), tone=(0.85, "resonance scale (1 = 5-8 kHz as in the patch; lower = larger insect)"),
     dur=(3.0, "s"), seed=(0, "variation"))
def cicada(sr=DEFAULT_SR, rate=120.0, tone=0.85, dur=3.0, seed=0):
    y = V.chitin_clicks(float(np.clip(rate, 10.0, 600.0)), float(dur), _rng(seed, 11), kind="cicada",
                        res_scale=float(np.clip(tone, 0.1, 1.5)))
    n = len(y)
    swell = 0.6 + 0.4 * np.sin(np.pi * np.arange(n) / n) ** 0.5       # UNSOURCED: call swells and fades
    return _fin(y * swell, sr, peak=0.45, hp=200.0)


_FLY_H = np.array([0.877, 1.000, 0.503, 0.200, 0.216, 0.066, 0.044, 0.039, 0.029, 0.013])   # Farnell fig 50.3
_BUZZ = {  # default wingbeat Hz, band-pass centre of the Farnell patch, duty
    "fly": (220.0, None, None), "mosquito": (600.0, 8000.0, 0.03), "bee": (230.0, 6000.0, 0.03)}


def _buzz(kind: str, sr, freq, wobble, motion, proximity, tone, dur, seed) -> np.ndarray:
    rng = _rng(seed, 12)
    n = _n(dur)
    t = np.arange(n) / KSR
    prox = float(np.clip(proximity, 0.0, 1.0))
    d0 = 0.15 + 3.0 * (1.0 - prox)                                    # UNSOURCED: mean distance m
    mo = float(np.clip(motion, 0.0, 1.0))
    rate = rng.uniform(0.3, 0.6)                                      # UNSOURCED: circling rate Hz
    d = d0 * (1.0 + 0.8 * mo * np.sin(TWO_PI * rate * t + rng.uniform(0, TWO_PI)))
    dop = 1.0 / (1.0 + np.gradient(d) * KSR / 343.0)                  # Doppler from the radial speed
    # Farnell: one common ~4 Hz noise moves both wings ("modulating each wing separately sounds wrong")
    f = float(np.clip(freq, 20.0, 2000.0)) * (1.0 + 0.2 * float(np.clip(wobble, 0, 1)) * _slow(n, rng, 4.0)) * dop
    ph = TWO_PI * np.cumsum(f) / KSR
    _f0, band, duty = _BUZZ[kind]
    kmax = int(min(0.45 * KSR, 12000.0) / float(np.max(f)))
    if kind == "fly":
        amps = _FLY_H[:kmax]
    else:                                                             # 3 % duty pulse (phasor > 0.97)
        k = np.arange(1, min(kmax, 60) + 1)
        amps = np.sinc(k * duty)
    y = np.zeros(n)
    for wing, det in ((0.0, 1.0), (1.3, 1.004)):                      # UNSOURCED: second wing offset
        for i, a in enumerate(amps):
            y += a * np.sin((i + 1) * det * ph + wing)
    if band:
        y = 0.25 * _unit(y) + _unit(_bpf(y, float(np.clip(tone, 300.0, 12000.0)), 12.0))
    else:
        y = _bpf(y, float(np.clip(tone, 300.0, 12000.0)), 0.3)             # fly patch: bp 2000 Q 0.3
    y = _lp(y * (d0 / d), 1500.0 + 6500.0 * prox)
    return _fin(y, sr, peak=0.2 + 0.4 * prox)


def _buzz_params(f0, tone, tone_help):
    return dict(freq=(f0, "wingbeat fundamental Hz"), wobble=(0.5, "0..1 irregular pitch warble (1 = 20 %)"),
                motion=(0.5, "0..1 circling: Doppler and level swings"), proximity=(0.7, "0 far .. 1 at the ear"),
                tone=(tone, tone_help), dur=(2.0, "s"), seed=(0, "variation"))


@sfx("fly", "Housefly buzz (Farnell harmonic table, common wing wobble).",
     **_buzz_params(220.0, 2000.0, "body band-pass centre Hz (Farnell 2000, Q 0.3)"))
def fly(sr=DEFAULT_SR, freq=220.0, wobble=0.5, motion=0.5, proximity=0.7, tone=2000.0, dur=2.0, seed=0):
    return _buzz("fly", sr, freq, wobble, motion, proximity, tone, dur, seed)


@sfx("mosquito", "Mosquito whine: narrow wing pulses through a high band-pass.",
     **_buzz_params(600.0, 4200.0, "whine band centre Hz, Q 12 (Farnell patch: 8000)"))
def mosquito(sr=DEFAULT_SR, freq=600.0, wobble=0.5, motion=0.5, proximity=0.7, tone=4200.0, dur=2.0, seed=0):
    return _buzz("mosquito", sr, freq, wobble, motion, proximity, tone, dur, seed)


@sfx("bee", "Bee / bumblebee buzz: narrow wing pulses through a mid band-pass.",
     **_buzz_params(230.0, 2600.0, "buzz band centre Hz, Q 12 (Farnell patch: 6000)"))
def bee(sr=DEFAULT_SR, freq=230.0, wobble=0.5, motion=0.5, proximity=0.7, tone=2600.0, dur=2.0, seed=0):
    return _buzz("bee", sr, freq, wobble, motion, proximity, tone, dur, seed)


def _gate(n: int, rng, hz: float = 0.2) -> np.ndarray:
    """Farnell's random on/off behaviour gate: clip(1e9 lop(lop(noise, 0.2), 0.2), 0, 1), de-clicked."""
    return V._pd_lop(np.clip(1e9 * _slow(n, rng, hz), 0.0, 1.0), 12.0, float(KSR))


@sfx("night_insects", "Night insect bed: gated crickets, Farnell 'critters' and 'chirper' (loop it with \"loop\": true).",
     density=(0.5, "0..1 number of singers"), temp=(20.0, "degC (sets the cricket chirp rates)"),
     bright=(0.5, "0..1 top end"), dur=(6.0, "s"), seed=(0, "variation"))
def night_insects(sr=DEFAULT_SR, density=0.5, temp=20.0, bright=0.5, dur=6.0, seed=0):
    rng = _rng(seed, 13)
    n = _n(dur)
    t = np.arange(n) / KSR
    out = np.zeros(n)
    for i in range(2 + int(round(8 * float(np.clip(density, 0, 1))))):
        sp = ("field", "tree", "bell", "field")[i % 4]
        r = dolbear_rate(temp + rng.uniform(-1.5, 1.5)) * _CRICKETS[sp][5]
        y = _cricket(n, min(r, 12.0), rng, sp, bright, phase0=rng.random())
        g = _gate(n, rng) if i else 1.0                                # one singer never stops
        out += _unit(y) * g * rng.uniform(0.15, 1.0)
    for fs in ((2012.0, 4.0, 20.0, 2.0), (2134.0, 4.279, 20.4, 15.5)):  # critters: product of four cosines
        c = np.ones(n)
        for f in fs:
            c *= np.cos(TWO_PI * f * t + rng.uniform(0, TWO_PI))
        out += 0.5 * c
    x = np.clip(np.mod(0.8 * t + rng.random(), 1.0) * 44.0, 0.0, 4.0)   # chirper: 4 bumps per 1.25 s
    out += 0.4 * np.clip(-np.sin(TWO_PI * x), 0.0, 1.0) * (np.sin(TWO_PI * 5134.0 * t) + bright * np.sin(TWO_PI * 12342.0 * t)) * _gate(n, rng, 0.1)
    return _fin(_lp(out, 4500.0 + 6000.0 * float(np.clip(bright, 0, 1))), sr, peak=0.4, hp=300.0)


_WINGS = {"bird": (6.0, 1.0), "bat": (11.0, 1.6), "insect": (35.0, 3.0)}   # UNSOURCED: beats/s, band scale


@sfx("wings", "Wing flaps (Farnell fan pulse into band-passed air, swept by the stroke).",
     kind=("bird", "bird|bat|insect"), rate=(0.0, "flaps per second, 0 = from kind (6 / 11 / 35)"),
     size=(1.0, "wing scale: bigger = lower band"), dur=(2.0, "s"), seed=(0, "variation"))
def wings(sr=DEFAULT_SR, kind="bird", rate=0.0, size=1.0, dur=2.0, seed=0):
    r0, sc = _WINGS[_choice(kind, _WINGS, "kind")]
    r = float(np.clip(rate if rate else r0, 0.5, 80.0))
    sc = sc / float(np.clip(size, 0.2, 6.0)) ** 0.3
    rng = _rng(seed, 14)
    n = _n(dur)
    ph = np.pi * r * np.arange(n) / KSR
    pulse = 1.0 / (1.0 + 5.0 * np.sin(ph) ** 2)                        # Farnell fan, k = 5
    nz = V._bp(rng.uniform(-1, 1, n), 700.0 * sc, 1.0)
    x = (nz * pulse * 0.4 + 0.1 * (pulse - pulse.mean())) * 0.2
    y = V._vcf(x, (300.0 + 300.0 * np.cos(2.0 * ph)) * sc, 5.0)
    return _fin(y * (1.0 + 0.2 * _slow(n, rng, 2.0)), sr, peak=0.5)


# ============================================================================================
# birds: Farnell FM syrinx driven by a syllable grammar
# ============================================================================================

@numba.njit(cache=True)
def _fm_syrinx(f, w, fb, nz, sr):
    """Syrinx pulse oscillator y = 1/(1 + (w cos(pi phase))^2), DC removed and scaled to +-~1,
    with feedback FM (Farnell: loose syrinx = "FM with feedback") and noise FM."""
    n = len(f)
    out = np.zeros(n)
    ph = 0.0
    y1 = 0.0
    for i in range(n):
        ph += f[i] * (1.0 + fb[i] * y1 + nz[i]) / sr
        ph -= math.floor(ph)
        wi = w[i]
        c = wi * math.cos(math.pi * ph)
        p = 1.0 / (1.0 + c * c)
        w2 = wi * wi
        y1 = (p - 1.0 / math.sqrt(1.0 + w2)) * (1.0 + w2) / (w2 + 1e-9) * 1.4
        if y1 > 1.5:
            y1 = 1.5
        out[i] = y1
    return out


def _syl(dur, f, gap, rep=(1, 1), **kw):
    return dict(dur=dur, f=f, gap=gap, rep=rep, **kw)


# UNSOURCED: every number in the presets below is tuning by ear / field-guide memory. Sourced
# constraints kept: syllables 5-300 ms for songbirds, oscine f0 mostly 3-5 kHz, tracheal Q 2-5,
# raven-type trachea resonances 1200 / 2200 Hz (Smyth & Smith) for the crow.
BIRDS = {
    "canary": dict(tf=(3000, 4200), q=3.0, hp=1500, w=0.9, pause=0.12, phrases=[
        _syl(0.045, ((0, 3900), (1, 2900)), 0.035, (10, 16)),
        _syl(0.09, ((0, 2100), (0.5, 2500), (1, 2200)), 0.05, (6, 9), w=1.3),
        _syl(0.03, ((0, 4600), (1, 3500)), 0.028, (14, 22))]),
    "robin": dict(tf=(2600, 3800), q=2.5, hp=1200, w=0.9, pause=0.07, phrases=[
        _syl((0.07, 0.2), ("rand", 2000, 4600), (0.035, 0.07), (7, 11), trill=((0, 60), (0, 0.1)))]),
    "blackbird": dict(tf=(1900, 2800), q=2.5, hp=900, w=1.1, pause=0.08, phrases=[
        _syl((0.16, 0.3), ("rand", 1500, 2900), (0.05, 0.1), (4, 6)),
        _syl(0.05, ("rand", 3200, 4800), 0.035, (3, 5), amp=0.5)]),
    "sparrow": dict(tf=(3000, 4300), q=2.0, hp=1500, w=1.8, pause=0.2, phrases=[
        _syl(0.1, ((0, 4300), (0.35, 3100), (1, 3900)), (0.14, 0.3), (5, 8), var=0.05)]),
    "owl": dict(tf=(390, 780), q=4.0, hp=150, w=0.7, pause=0.3, phrases=[
        _syl(0.45, ((0, 395), (0.2, 410), (1, 370)), 0.7),
        _syl(0.16, ((0, 400), (1, 390)), 0.09, (2, 2)),
        _syl(0.6, ((0, 410), (0.15, 420), (1, 360)), 0.0)]),
    "crow": dict(tf=(1200, 2200), q=3.0, hp=300, w=3.0, chaos=0.5, noise=0.2, pause=0.3, phrases=[
        _syl((0.28, 0.36), ((0, 470), (0.15, 560), (1, 430)), (0.16, 0.22), (3, 4))]),
    "gull": dict(tf=(1700, 2900), q=2.5, hp=500, w=2.4, chaos=0.2, noise=0.06, pause=0.1, phrases=[
        _syl(0.5, ((0, 750), (0.3, 1150), (1, 820)), 0.12),
        _syl(0.2, ((0, 900), (0.4, 1250), (1, 950)), 0.07, (4, 7), drift=-0.02)]),
    "duck": dict(tf=(1100, 2600), q=3.0, hp=300, w=3.5, chaos=0.6, noise=0.3, pause=0.2, phrases=[
        _syl(0.17, ((0, 330), (0.2, 300), (1, 230)), 0.09, (4, 6), decay=0.88, stretch=1.06)]),
    "rooster": dict(tf=(1000, 2500), q=2.5, hp=300, w=2.6, chaos=0.12, noise=0.04, pause=0.0, phrases=[
        _syl(0.13, ((0, 520), (1, 640)), 0.05),
        _syl(0.12, ((0, 640), (1, 700)), 0.05),
        _syl(0.26, ((0, 700), (1, 850)), 0.045),
        _syl(0.95, ((0, 860), (0.25, 930), (0.8, 800), (1, 470)), 0.0)]),
    "cuckoo": dict(tf=(600, 1200), q=4.0, hp=250, w=0.7, pause=0.0, phrases=[
        _syl(0.2, ((0, 667), (1, 655)), 0.1),
        _syl(0.34, ((0, 530), (1, 520)), 0.0)]),
    "pigeon": dict(tf=(450, 900), q=4.0, hp=150, w=1.0, pause=0.12, phrases=[
        _syl(0.3, ((0, 430), (0.5, 470), (1, 420)), 0.1, trill=(28, 0.04), am=(28, 0.45)),
        _syl(0.55, ((0, 450), (0.4, 520), (1, 430)), 0.12, trill=(28, 0.04), am=(28, 0.45)),
        _syl(0.3, ((0, 440), (0.5, 470), (1, 410)), 0.3, trill=(28, 0.04), am=(28, 0.45)),
        _syl(0.3, ((0, 430), (0.5, 470), (1, 420)), 0.1, (2, 2), trill=(28, 0.04), am=(28, 0.45))]),
    "woodpecker": dict(drum=True, strikes=(12, 16), ioi=0.055, f0=950.0),
}
SONGBIRDS = ("canary", "robin", "blackbird", "sparrow", "cuckoo", "pigeon")


def _pick(v, rng):
    return float(rng.uniform(v[0], v[1])) if isinstance(v, (tuple, list)) else float(v)


def bird_range(species: str) -> tuple[float, float]:
    """(lowest, highest) carrier frequency in Hz a preset can produce at pitch 1."""
    p = BIRDS[species]
    if p.get("drum"):
        return (p["f0"], p["f0"])
    lo, hi = 1e9, 0.0
    for ph in p["phrases"]:
        f = ph["f"]
        fs = [f[1], f[2]] if f[0] == "rand" else [q[1] for q in f]
        v = ph.get("var", 0.02) + abs(ph.get("drift", 0.0)) * ph["rep"][1] + _trill_max(ph)
        lo, hi = min(lo, min(fs) * (1 - v)), max(hi, max(fs) * (1 + v))
    return lo, hi


def _trill_max(ph) -> float:
    tr = ph.get("trill")
    if not tr:
        return 0.0
    d = tr[1]
    return float(d[1] if isinstance(d, (tuple, list)) else d)


def birdsong_plan(species: str = "robin", seed: int = 0, repeats: int = 1, syllables: int = 0,
                  tempo: float = 1.0, pitch: float = 1.0) -> list[dict]:
    """The song as a list of syllables (element -> syllable -> phrase -> song): each a dict with
    onset ``t`` and ``dur`` in seconds, contour ``f`` [(u, Hz)], and the syrinx parameters."""
    p = BIRDS[_choice(species, BIRDS, "species")]
    rng = _rng(seed, 20)
    tempo = float(np.clip(tempo, 0.25, 4.0))
    out: list[dict] = []
    t = 0.03
    for _ in range(max(1, int(repeats))):
        if p.get("drum"):
            k = int(rng.integers(p["strikes"][0], p["strikes"][1] + 1))
            for j in range(k):
                out.append(dict(t=t, dur=0.004, f=[(0, p["f0"]), (1, p["f0"])], amp=1.0 - 0.5 * j / k))
                t += p["ioi"] * (0.985 ** j) / tempo
            t += 1.2 / tempo
            continue
        for ph in p["phrases"]:
            k = int(rng.integers(ph["rep"][0], ph["rep"][1] + 1))
            for j in range(k):
                f = ph["f"]
                if f[0] == "rand":
                    m = int(rng.integers(2, 4))
                    pts = [(i / (m - 1), float(rng.uniform(f[1], f[2]))) for i in range(m)]
                else:
                    s = (1.0 + rng.uniform(-1, 1) * ph.get("var", 0.02)) * (1.0 + ph.get("drift", 0.0)) ** j
                    pts = [(u, v * s) for u, v in f]
                dur = _pick(ph["dur"], rng) * ph.get("stretch", 1.0) ** j / tempo
                tr = ph.get("trill", (0.0, 0.0))
                out.append(dict(t=t, dur=dur, f=[(u, v * pitch) for u, v in pts],
                                trill=(_pick(tr[0], rng), _pick(tr[1], rng)), am=ph.get("am", (0.0, 0.0)),
                                w=ph.get("w", p["w"]), chaos=ph.get("chaos", p.get("chaos", 0.0)),
                                noise=ph.get("noise", p.get("noise", 0.0)),
                                amp=ph.get("amp", 1.0) * ph.get("decay", 1.0) ** j))
                t += dur + _pick(ph["gap"], rng) / tempo
            t += p["pause"] / tempo
        t += 0.55 / tempo                                              # UNSOURCED: pause between songs
    if syllables and int(syllables) > 0:
        k = int(syllables)
        if k <= len(out):
            out = out[:k]
        else:                                                          # cycle the song to reach the count
            span = out[-1]["t"] + out[-1]["dur"] + 0.3
            base = list(out)
            while len(out) < k:
                s = base[len(out) % len(base)]
                out.append(dict(s, t=s["t"] + span * (len(out) // len(base))))
    return out


def _birdsong(species: str, seed, repeats=1, syllables=0, tempo=1.0, pitch=1.0, two_voice=0.0) -> np.ndarray:
    p = BIRDS[species]
    plan = birdsong_plan(species, seed, repeats, syllables, tempo, pitch)
    rng = _rng(seed, 21)
    n = _n(plan[-1]["t"] + plan[-1]["dur"] + 0.25)
    if p.get("drum"):                      # beak on a resonant trunk: impulses into one persisting wooden body
        force = np.zeros(n)
        pulse = np.sin(np.pi * np.arange(24) / 24) ** 2
        for s in plan:
            _place(force, pulse, s["t"], s["amp"] * rng.uniform(0.85, 1.0))
        return M.render_modes(force, M.resolve("sy_stick", p["f0"] * pitch / 950.0 * 0.5, d_scale=2.0))
    f = np.full(n, plan[0]["f"][0][1])
    amp = np.zeros(n)
    w = np.full(n, p["w"])
    fb = np.zeros(n)
    nzd = np.zeros(n)
    for s in plan:
        i0 = int(s["t"] * KSR)
        m = min(max(16, int(s["dur"] * KSR)), n - i0)
        u = np.arange(m) / m
        tt = np.arange(m) / KSR
        ff = np.interp(u, [q[0] for q in s["f"]], [q[1] for q in s["f"]])
        rate, depth = s["trill"]
        if rate > 0 and depth > 0:         # bronchial pulse oscillator as the FM source (bp1, mod)
            c = np.cos(np.pi * rate * tt)
            ff = ff * (1.0 + depth * (2.0 / (1.0 + c * c) - 1.4))
        na, nr = max(8, int(min(0.25 * m, 0.012 * KSR))), max(8, int(min(0.35 * m, 0.03 * KSR)))
        e = np.ones(m)
        e[:na] = np.sin(0.5 * np.pi * np.arange(na) / na) ** 2
        e[-nr:] *= np.sin(0.5 * np.pi * np.arange(nr, 0, -1) / nr) ** 2
        ar, ad = s["am"]
        if ar > 0:
            e = e * (1.0 - ad * 0.5 * (1.0 + np.cos(TWO_PI * ar * tt)))
        f[i0:i0 + m] = ff
        f[i0 + m:] = ff[-1]
        amp[i0:i0 + m] = e * s["amp"]
        w[i0:i0 + m] = s["w"]
        fb[i0:i0 + m] = s["chaos"]
        nzd[i0:i0 + m] = s["noise"]
    f = np.minimum(f, 0.2 * KSR)
    nz = nzd * V._pd_lop(rng.uniform(-1, 1, n), 2000.0, float(KSR))
    x = _fm_syrinx(f, w, fb, nz, float(KSR))
    if two_voice > 0:                      # second bronchial source at a non-harmonic ratio (great tit)
        x = x + two_voice * _fm_syrinx(f * 1.27, w, fb, nz, float(KSR))   # UNSOURCED ratio
    x = x * amp
    tf1, tf2 = (min(v * pitch, 0.4 * KSR) for v in p["tf"])
    y = _bpf(x, tf1, p["q"]) + _bpf(x, tf2, p["q"])                    # trachea: two parallel band-passes
    return _hp(y, p["hp"] * pitch) * (amp > 0)                         # beak


@sfx("birdsong", "Bird song / call from a syllable grammar driving Farnell's FM syrinx (12 species-like presets).",
     species=("robin", "canary|robin|blackbird|sparrow|owl|crow|gull|duck|rooster|cuckoo|pigeon|woodpecker"),
     repeats=(1, "songs / calls in a row"), syllables=(0, "exact syllable count (0 = the preset's own, random per seed)"),
     pitch=(1.0, "frequency scale (smaller bird > 1)"), tempo=(1.0, "speed scale"),
     two_voice=(0.0, "0..1 level of the second syrinx source"), bright=(1.0, "0.3..2 top-end"),
     seed=(0, "variation"))
def birdsong(sr=DEFAULT_SR, species="robin", repeats=1, syllables=0, pitch=1.0, tempo=1.0, two_voice=0.0,
             bright=1.0, seed=0):
    _choice(species, BIRDS, "species")
    y = _birdsong(species, seed, int(np.clip(repeats, 1, 50)), int(syllables), tempo,
                  float(np.clip(pitch, 0.25, 3.0)), float(np.clip(two_voice, 0.0, 1.0)))
    return _fin(_lp(y, 5500.0 * float(np.clip(bright, 0.3, 2.0))), sr, peak=0.5, fade_ms=2.0, hp=60.0)


_TUBES = {"oil_bird": (0.100, 0.0025), "raven": (0.070, 0.0035), "budgie": (0.050, 0.002)}   # Kahrs & Avanzini


@sfx("bird_call", "Physical syrinx: pressure-driven membrane valves into a trachea waveguide (Smyth & Smith); raven-like.",
     freq=(450.0, "membrane frequency Hz (tension), 150..900"), freq_end=(0.0, "glide target Hz, 0 = none"),
     pressure=(3.0, "air-sac pressure kPa, 1.5..6"), attack=(0.3, "fraction of dur over which pressure rises"),
     two_voice=(5.0, "Hz offset of the second valve (beating); 0 = one valve"),
     species=("raven", "trachea size: oil_bird|raven|budgie"), dur=(0.5, "s"))
def bird_call(sr=DEFAULT_SR, freq=450.0, freq_end=0.0, pressure=3.0, attack=0.3, two_voice=5.0, species="raven",
              dur=0.5):
    length, radius = _TUBES[_choice(species, _TUBES, "species")]
    n = _n(dur)
    f = np.clip(np.linspace(to_hz(freq), to_hz(freq_end) if freq_end else to_hz(freq), n), 150.0, 900.0)
    env = V.articulation_env(n, float(np.clip(attack, 0.05, 0.9)), offset=0.0) ** 0.5
    ps = 1000.0 * float(np.clip(pressure, 1.5, 6.0)) * (0.3 + 0.7 * env)
    y = V.syrinx(ps, (f, f + float(two_voice)), trachea_length_m=length, trachea_radius_m=radius,
                 second_valve=bool(two_voice))
    return _fin(_lp(y * env, 5000.0), sr, peak=0.6)


@sfx("dawn_chorus", "Dawn chorus bed: several songbirds at different distances singing over each other.",
     birds=(6, "number of singers"), dur=(8.0, "s"), bright=(1.0, "0.3..2"), seed=(0, "variation"))
def dawn_chorus(sr=DEFAULT_SR, birds=6, dur=8.0, bright=1.0, seed=0):
    rng = _rng(seed, 22)
    n = _n(dur)
    out = np.zeros(n)
    for b in range(int(np.clip(birds, 1, 40))):
        sp = SONGBIRDS[int(rng.integers(len(SONGBIRDS)))] if b else "robin"
        pitch = rng.uniform(0.85, 1.2)
        dist = rng.random()
        t = rng.uniform(0.0, 0.3 * dur) if b else 0.0
        k = 0
        while t < dur - 0.3:
            y = _birdsong(sp, int(seed) * 977 + b * 31 + k, 1, 0, rng.uniform(0.9, 1.1), pitch)
            _place(out, _lp(_unit(y), 6000.0 - 3500.0 * dist), t, 1.0 - 0.85 * dist)
            t += len(y) / KSR + rng.uniform(0.4, 2.5)
            k += 1
    return _fin(_lp(out, 5500.0 * float(np.clip(bright, 0.3, 2.0))), sr, peak=0.5, hp=100.0)


# ============================================================================================
# amphibians and reptiles
# ============================================================================================

_FROGS = {"a": "a", "croak": "a", "b": "b", "ribbit": "b", "c": "c", "trill": "c"}


@sfx("frog", "Frog calls (Farnell patches): accelerating croak, V-shaped 'brrp', resonant trill.",
     kind=("croak", "croak (a) | ribbit (b) | trill (c)"), size=(1.0, "body scale: bigger = lower and slower"),
     calls=(3, "number of calls"), gap=(0.5, "s between calls"), seed=(0, "variation"))
def frog(sr=DEFAULT_SR, kind="croak", size=1.0, calls=3, gap=0.5, seed=0):
    k = _FROGS[_choice(kind, _FROGS, "kind")]
    rng = _rng(seed, 30)
    size = float(np.clip(size, 0.3, 4.0))
    calls = int(np.clip(calls, 1, 60))
    parts, t = [], 0.01
    for _ in range(calls):
        y = V.frog_croak(k, rng)
        y = S._varispeed(y, 1.0 / (size * rng.uniform(0.97, 1.03)))    # one scale for pitch and time
        parts.append((y, t, rng.uniform(0.75, 1.0)))
        t += len(y) / KSR + max(0.02, float(gap)) * rng.uniform(0.8, 1.25)
    out = np.zeros(_n(t))
    for y, at, g in parts:
        _place(out, _unit(y), at, g)
    return _fin(out[:_n(t - 0.5 * float(gap))], sr, peak=0.6)


@sfx("rattlesnake", "Rattlesnake rattle (Farnell 'rattler': narrow pulses gating band-passed noise).",
     rate=(20.0, "rattle pulses per second (patch: 20)"), tone=(0.85, "band scale (1 = 6-7 kHz as in the patch)"),
     dur=(1.5, "s"), seed=(0, "variation"))
def rattlesnake(sr=DEFAULT_SR, rate=20.0, tone=0.85, dur=1.5, seed=0):
    rng = _rng(seed, 31)
    n = _n(dur)
    t = np.arange(n) / KSR
    ts = float(np.clip(tone, 0.1, 1.5))
    r = float(np.clip(rate, 4.0, 200.0)) * (1.0 + 0.03 * _slow(n, rng, 3.0))
    pulse = 1.0 / (1.0 + (6.0 * np.cos(np.pi * np.cumsum(r) / KSR)) ** 2)
    x = pulse * V._bp(rng.uniform(-1, 1, n), 7000.0 * ts, 13.0) * 0.1
    x = V._bp(x, 6000.0 * ts, 16.0)
    env = np.minimum(t / 0.08, 1.0) * np.minimum((dur - t) / 0.25, 1.0) ** 2      # UNSOURCED shake envelope
    return _fin(x * np.clip(env, 0, 1), sr, peak=0.4, hp=200.0)


_HISS = {"snake": ("shh", 1.0, 0.25, 1.4), "cat": ("xxx", 1.25, 0.08, 0.7)}   # UNSOURCED: phoneme row, scale, rise, s


@sfx("hiss", "Hiss: turbulent noise through fricative formants (STK phoneme rows).",
     kind=("snake", "snake (long, steady) | cat (sharp attack, spitting)"), size=(1.0, "body scale: bigger = lower"),
     dur=(0.0, "s, 0 = from kind"), seed=(0, "variation"))
def hiss(sr=DEFAULT_SR, kind="snake", size=1.0, dur=0.0, seed=0):
    row, sc, rise, d0 = _HISS[_choice(kind, _HISS, "kind")]
    rng = _rng(seed, 32)
    n = _n(dur if dur else d0)
    env = V.articulation_env(n, rise, offset=0.0) ** 0.6 * (1.0 + 0.2 * _slow(n, rng, 9.0))
    y = V.formant_filter(rng.uniform(-1, 1, n) * env, row, "stk", formant_scale=sc / float(np.clip(size, 0.3, 4.0)))
    return _fin(_lp(_hp(y, 700.0), 7500.0), sr, peak=0.4)


# ============================================================================================
# body
# ============================================================================================

def _breath(d: float, direction: str, rng, size: float, effort: float, rise=None) -> np.ndarray:
    y = V.breath(d, direction, rng, formant_scale=(1.0 + 0.1 * effort) / size, rise_frac=rise)
    turb = _hp(rng.uniform(-1, 1, len(y)), 1500.0) * V.articulation_env(len(y), rise or 0.4, offset=0.0)
    return _unit(y) + 0.12 * effort * _lp(turb, 6000.0)               # UNSOURCED: effort adds open turbulence


@sfx("breath", "Breath: noise through whispered-vowel formants (Cook), in / out / pant / gasp / sigh.",
     kind=("out", "in|out|pant|gasp|sigh"), effort=(0.5, "0 resting .. 1 exhausted (brighter, faster)"),
     size=(1.0, "body scale: bigger = lower formants"), dur=(0.0, "s, 0 = from kind"), seed=(0, "variation"))
def breath(sr=DEFAULT_SR, kind="out", effort=0.5, size=1.0, dur=0.0, seed=0):
    _choice(kind, ("in", "out", "pant", "gasp", "sigh"), "kind")
    rng = _rng(seed, 40)
    size = float(np.clip(size, 0.4, 4.0))
    e = float(np.clip(effort, 0.0, 1.0))
    if kind in ("in", "out"):
        y = _breath(dur or 1.3 - 0.5 * e, kind, rng, size, e)
    elif kind == "gasp":
        y = _breath(dur or 0.45, "in", rng, size / 1.15, 1.0, rise=0.2)
    elif kind == "sigh":
        d = dur or 1.7
        y = _breath(d, "out", rng, size, e, rise=0.15)
        n = len(y)
        f0 = 170.0 / size * np.linspace(1.0, 0.7, n)                   # UNSOURCED: voiced part of a sigh
        v = V.formant_filter(V.lf_pulse(f0, 0.25, rng=rng), "hah", "stk", formant_scale=1.0 / size)
        y = y + 0.5 * _unit(v * V.articulation_env(n, 0.12, offset=0.0) ** 2)
    else:                                                               # pant
        d = dur or 2.0
        rate = 2.0 + 1.8 * e                                            # UNSOURCED: pants per second
        y = np.zeros(_n(d))
        t = 0.0
        while t + 0.6 / rate < d:
            _place(y, _breath(0.45 / rate, "in", rng, size, e, rise=0.5), t, 0.7)
            _place(y, _breath(0.55 / rate, "out", rng, size, e, rise=0.25), t + 0.45 / rate, 1.0)
            t += rng.uniform(0.95, 1.05) / rate
    return _fin(y, sr, peak=0.25 + 0.3 * e)


@sfx("grunt", "Short voiced effort: LF glottal pulse through a neutral (schwa) tract, falling pitch.",
     pitch=(110.0, "f0 Hz at size 1"), effort=(0.5, "0 lax .. 1 pressed"), size=(1.0, "body scale"),
     dur=(0.3, "s"), seed=(0, "variation"))
def grunt(sr=DEFAULT_SR, pitch=110.0, effort=0.5, size=1.0, dur=0.3, seed=0):
    rng = _rng(seed, 41)
    size = float(np.clip(size, 0.4, 4.0))
    n = _n(dur)
    env = V.articulation_env(n, 0.2, offset=0.0)
    f = to_hz(pitch) / size * (0.8 + 0.2 * np.linspace(1.0, 0.0, n) ** 1.5)     # UNSOURCED fall (procedural body.grunt)
    src = V.lf_pulse(f, 0.25 + 0.5 * float(np.clip(effort, 0, 1)), rng=rng, voiced_env=np.where(env > 0.02, 1.0, 0.0))
    y = V.formant_filter(src * env, "@", "farnell", formant_scale=1.0 / size)
    return _fin(_hp(y, 60.0), sr, peak=0.5)


@sfx("cough", "Cough: explosive noise burst through the open tract with a short voiced tail.",
     count=(2, "coughs"), effort=(0.7, "0..1"), size=(1.0, "body scale"), seed=(0, "variation"))
def cough(sr=DEFAULT_SR, count=2, effort=0.7, size=1.0, seed=0):
    rng = _rng(seed, 42)
    size = float(np.clip(size, 0.4, 4.0))
    count = int(np.clip(count, 1, 20))
    out = np.zeros(_n(0.45 * count + 0.3))
    t = 0.01
    for k in range(count):                                             # UNSOURCED: all timings and levels
        d = rng.uniform(0.2, 0.28) * (0.85 ** k)
        n = _n(d)
        tt = np.arange(n) / KSR
        e = np.minimum(tt / 0.004, 1.0) * np.exp(-tt / (0.05 + 0.03 * effort))
        burst = V.vocal_tract_comb(rng.uniform(-1, 1, n) * e, 0.17 * size, q=2.5)
        f0 = 150.0 / size * np.linspace(1.1, 0.75, n)
        tail = V.formant_filter(V.lf_pulse(f0, 0.7, rng=rng), "@", "farnell", formant_scale=1.0 / size)
        tail = tail * np.minimum(tt / 0.03, 1.0) * np.exp(-tt / 0.07)
        _place(out, _unit(burst) + 0.35 * _unit(tail), t, 0.8 ** k)
        t += d + rng.uniform(0.08, 0.14)
    return _fin(_lp(out[:_n(t + 0.1)], 6000.0), sr, peak=0.3 + 0.4 * float(np.clip(effort, 0, 1)))


@sfx("sneeze", "Sneeze: rising gasps then an explosive 'tshoo' (fricative burst + falling voiced vowel).",
     effort=(0.8, "0..1"), size=(1.0, "body scale"), seed=(0, "variation"))
def sneeze(sr=DEFAULT_SR, effort=0.8, size=1.0, seed=0):
    rng = _rng(seed, 43)
    size = float(np.clip(size, 0.4, 4.0))
    out = np.zeros(_n(1.6))
    t = 0.01
    for k in range(2):                                                 # UNSOURCED: "ah.. ah.." pre-inhales
        g = _breath(0.22 + 0.08 * k, "in", rng, size / (1.05 + 0.1 * k), 0.8, rise=0.7)
        _place(out, g, t, 0.35 + 0.2 * k)
        t += len(g) / KSR + 0.07
    n = _n(0.42)
    tt = np.arange(n) / KSR
    burst = V.formant_filter(rng.uniform(-1, 1, n) * np.minimum(tt / 0.006, 1.0) * np.exp(-tt / 0.09), "shh", "stk",
                             formant_scale=1.0 / size)
    f0 = 360.0 / size * np.linspace(1.0, 0.5, n)
    vow = V.formant_filter(V.lf_pulse(f0, 0.6, rng=rng), "ooo", "stk", formant_scale=1.0 / size)
    vow = vow * np.clip((tt - 0.04) / 0.04, 0.0, 1.0) * np.exp(-tt / 0.12)
    _place(out, _unit(burst) + 0.6 * _unit(vow), t + 0.03, 1.0)
    return _fin(_lp(out[:_n(t + 0.5)], 6500.0), sr, peak=0.3 + 0.4 * float(np.clip(effort, 0, 1)))


@sfx("snore", "Snore: soft-palate flutter on each inhale, quiet breath out.",
     rate=(14.0, "breaths per minute"), pitch=(45.0, "flutter frequency Hz"), size=(1.0, "body scale"),
     dur=(8.0, "s"), seed=(0, "variation"))
def snore(sr=DEFAULT_SR, rate=14.0, pitch=45.0, size=1.0, dur=8.0, seed=0):
    rng = _rng(seed, 44)
    size = float(np.clip(size, 0.4, 4.0))
    period = 60.0 / float(np.clip(rate, 4.0, 60.0))
    out = np.zeros(_n(dur))
    t = 0.05
    while t < dur - 0.2:                                               # UNSOURCED: 42 % inhale, flutter mapping
        n = _n(0.42 * period * rng.uniform(0.92, 1.08))
        env = V.articulation_env(n, 0.45, offset=0.0)
        f0 = to_hz(pitch) / size * (0.85 + 0.3 * env) * (1.0 + 0.06 * _slow(n, rng, 6.0))
        _place(out, _unit(_voc(f0, env, rng, tract_m=0.17 * size, harsh=0.55, jitter=0.06, shimmer=0.35, sub=0.3)), t)
        _place(out, _breath(0.4 * period, "out", rng, size, 0.2), t + n / KSR + 0.08 * period, 0.18)
        t += period * rng.uniform(0.96, 1.04)
    return _fin(_lp(out, 2500.0), sr, peak=0.5)


@sfx("gulp", "Swallow / gulp: the throat cavity closing and re-opening (two short pitch-swept resonances).",
     size=(1.0, "body scale: bigger = lower"), seed=(0, "variation"))
def gulp(sr=DEFAULT_SR, size=1.0, seed=0):
    rng = _rng(seed, 45)
    size = float(np.clip(size, 0.4, 4.0))
    out = np.zeros(_n(0.4))
    for t0, fa, fb, d, g in ((0.01, 420.0, 170.0, 0.07, 0.7), (0.14, 190.0, 620.0, 0.09, 1.0)):   # UNSOURCED
        n = _n(d)
        u = np.arange(n) / n
        f = (fa + (fb - fa) * u ** 1.5) / size * rng.uniform(0.92, 1.08)
        _place(out, np.sin(TWO_PI * np.cumsum(f) / KSR) * np.sin(np.pi * u) ** 2 * (1.0 - u), t0, g)
    click = _bpf(rng.uniform(-1, 1, _n(0.012)) * np.exp(-np.arange(_n(0.012)) / (0.002 * KSR)), 1800.0 / size, 2.0)
    _place(out, click, 0.135, 0.25)
    return _fin(_lp(out, 2500.0), sr, peak=0.5)


@sfx("eat", "Eating: crunch (brittle bites, PhISEM 'Crunch'), chew (soft, closed mouth) or slurp (liquid suction).",
     kind=("crunch", "crunch|chew|slurp"), bites=(4, "bites / chews / slurps"), rate=(1.7, "per second"),
     seed=(0, "variation"))
def eat(sr=DEFAULT_SR, kind="crunch", bites=4, rate=1.7, seed=0):
    _choice(kind, ("crunch", "chew", "slurp"), "kind")
    rng = _rng(seed, 46)
    bites = int(np.clip(bites, 1, 60))
    rate = float(np.clip(rate, 0.3, 6.0))
    n = _n(bites / rate + 0.4)
    t = np.arange(n) / KSR
    out = np.zeros(n)
    if kind == "slurp":                                                # UNSOURCED recipe
        for k in range(bites):
            m = _n(rng.uniform(0.35, 0.6))
            u = np.arange(m) / m
            bub = 1.0 / (1.0 + (3.0 * np.cos(np.pi * np.cumsum(rng.uniform(28, 45) * (1 + 0.3 * _slow(m, rng, 8.0))) / KSR)) ** 2)
            y = V._vcf(rng.uniform(-1, 1, m) * bub, 500.0 + 2300.0 * u ** 1.3, 6.0) * np.sin(np.pi * u) ** 0.7
            _place(out, _unit(y), 0.02 + k / rate, rng.uniform(0.8, 1.0))
        return _fin(_lp(out, 5000.0), sr, peak=0.4)
    curve = np.zeros(n)
    for k in range(bites):                                             # jaw closings: fast rise, slower release
        tk = 0.03 + k / rate * rng.uniform(0.95, 1.05)
        d = rng.uniform(0.09, 0.16)
        x = np.clip((t - tk) / d, 0.0, 1.0)
        curve += np.where((t >= tk) & (t < tk + d), np.sin(np.pi * x ** 0.5) * (1.0 if k == 0 else rng.uniform(0.45, 0.8)), 0.0)
    y = P.shaker("Crunch", n / KSR, rng, energy_curve=np.clip(curve, 0, 1), energy_mode="level",
                 resonance_scale=1.0 if kind == "crunch" else 0.45)[:n]
    if kind == "chew":                                                 # closed mouth: dull, plus wet smacks
        y = _lp(_unit(y), 900.0) + 0.3 * _lp(curve * rng.uniform(-1, 1, n), 300.0)
        for k in range(bites):
            b = P.surfacing_bubble(rng, dim=0.7, fmin=300.0, fmax=900.0, sweep=1500.0)
            _place(y, _unit(b), 0.03 + (k + 0.55) / rate, 0.35)
        return _fin(y, sr, peak=0.35)
    y = _unit(y) * (0.5 + 0.5 * np.exp(-t * rate / 1.5)) + 0.2 * _bpf(_unit(y), 1200.0, 2.0)   # first bite open-mouthed
    return _fin(_lp(y, 6500.0), sr, peak=0.5)


def heart_s1s2(bpm: float) -> float:
    """S1-S2 interval in s: left-ventricular ejection time (Weissler 1968: 0.413 - 0.0017 HR) plus
    isovolumic contraction. UNSOURCED: the 0.04 s contraction time (gives 0.35 s at 60 bpm, the
    LovelyHeart patch's 35 %)."""
    return float(np.clip(0.453 - 0.0017 * bpm, 0.12, 0.6 * 60.0 / bpm))


def _onebeat(f: float, cycles: int, vol2: float, vol: float) -> np.ndarray:
    """LovelyHeart ``onebeat``: exactly ``cycles`` phase-locked cycles of f (+ 2nd harmonic), (1 - r)^2."""
    n = _n(cycles / f)
    r = np.arange(n) / n
    return (0.5 * np.sin(TWO_PI * cycles * r) + 0.5 * vol2 * np.sin(TWO_PI * 2 * cycles * r)) * (1.0 - r) ** 2 * vol


@sfx("heartbeat", "Heartbeat lub-dub: two phase-locked few-cycle thumps (S1, S2) with physiological timing.",
     rate=(1.0, "beats per second (1.0 = 60 bpm)"), dur=(2.0, "s"), murmur=(0.0, "0..1 turbulent murmur between the sounds"),
     tone=(1.0, "pitch scale of the thumps (1 = 63 / 42 Hz; 1.6 = small chest / stethoscope diaphragm)"),
     dub_delay=(0.0, "S1-S2 interval s, 0 = from heart rate (0.453 - 0.0017 bpm)"), seed=(0, "murmur noise"))
def heartbeat(sr=DEFAULT_SR, rate=1.0, dur=2.0, murmur=0.0, tone=1.0, dub_delay=0.0, seed=0):
    rate = float(np.clip(rate, 0.3, 4.0))
    period = 1.0 / rate
    d = float(np.clip(dub_delay, 0.1, 0.8 * period)) if dub_delay else heart_s1s2(60.0 * rate)
    tone = float(np.clip(tone, 0.5, 4.0))
    rng = _rng(seed, 47)
    n = _n(dur)
    out = np.zeros(n)
    mur = np.zeros(n)
    lub, dub = _onebeat(63.0 * tone, 5, 0.0, 0.65), _onebeat(42.0 * tone, 3, 0.5, 1.0)
    k = 0
    while 0.02 + k * period + d < dur - 0.08:
        t = 0.02 + k * period
        _place(out, lub, t)
        _place(out, dub, t + d)
        for t0, gap in ((t, d), (t + d, period - d)):                  # murmur fired on both sounds
            m = _n(gap)
            u = np.arange(m) / m
            _place(mur, np.where(u < 0.3, u / 0.3, (1.0 - u) / 0.7) ** 2, t0)
        k += 1
    if murmur > 0:
        nz = V._bp(rng.uniform(-1, 1, n), 50.0 * tone, 5.0) ** 2
        out += float(np.clip(murmur, 0, 1)) * 0.5 * _unit(_lp(nz * mur, 500.0))
    return _fin(out, sr, peak=0.8, hp=10.0)


def _clap(rng, fc: float, q: float, decay: float, n: int | None = None) -> np.ndarray:
    m = _n(max(8 * decay, 0.03)) if n is None else n
    t = np.arange(m) / KSR
    return _bpf(rng.uniform(-1, 1, m) * np.minimum(t / 0.0004, 1.0) * np.exp(-t / decay), fc, q)


@sfx("handclap", "One hand clap: a filtered noise burst; cupped hands are lower and hollower.",
     cup=(0.3, "0 flat palms (bright ~2.4 kHz) .. 1 cupped (hollow ~0.8 kHz)"), seed=(0, "variation"))
def handclap(sr=DEFAULT_SR, cup=0.3, seed=0):
    rng = _rng(seed, 48)
    c = float(np.clip(cup, 0.0, 1.0))
    fc = 2400.0 * (800.0 / 2400.0) ** c                                # UNSOURCED resonance range
    y = _clap(rng, fc, 1.5 + 2.5 * c, 0.008 + 0.008 * c, _n(0.12))
    y[_n(0.003):] += 0.5 * _clap(rng, fc * 1.5, 2.0, 0.006, len(y) - _n(0.003))   # UNSOURCED: fingers then palm
    return _fin(y, sr, peak=0.7, fade_ms=1.0, hp=200.0)


CLAP_VOICES = 64        # clappers rendered one by one; the rest of the crowd is a Poisson grain bed


def applause_rate(enthusiasm: float) -> float:
    """Claps per second per person. UNSOURCED: 2 Hz polite .. 4.5 Hz enthusiastic."""
    return 2.0 + 2.5 * float(np.clip(enthusiasm, 0.0, 1.0))


def applause_plan(crowd: int = 30, enthusiasm: float = 0.6, dur: float = 4.0, seed: int = 0) -> list[np.ndarray]:
    """Clap times (s) of each individually rendered clapper: own rate (+-8 %), phase and 4 % jitter."""
    rng = _rng(seed, 49)
    r = applause_rate(enthusiasm)
    plan = []
    for _ in range(int(min(max(crowd, 1), CLAP_VOICES))):
        ri = r * float(np.clip(rng.normal(1.0, 0.08), 0.7, 1.3))
        k = np.arange(int(dur * ri) + 2)
        t = rng.uniform(0.0, 1.0 / ri) + k / ri + rng.normal(0.0, 0.04 / ri, len(k))
        plan.append(np.sort(t[(t >= 0.0) & (t < dur - 0.03)]))
    return plan


@sfx("applause", "Applause: every clapper has a rate, a hand resonance and a place; big crowds blur into Poisson grains.",
     crowd=(30, "number of people"), enthusiasm=(0.6, "0 polite (2 claps/s each) .. 1 wild (4.5 claps/s)"),
     dur=(4.0, "s"), seed=(0, "variation"))
def applause(sr=DEFAULT_SR, crowd=30, enthusiasm=0.6, dur=4.0, seed=0):
    crowd = int(np.clip(crowd, 1, 100000))
    rng = _rng(seed, 50)
    n = _n(dur)
    out = np.zeros(n)
    for times in applause_plan(crowd, enthusiasm, dur, seed):
        fc = float(np.clip(1500.0 * math.exp(rng.normal(0.0, 0.3)), 700.0, 3200.0))    # UNSOURCED spread
        decay = rng.uniform(0.006, 0.014)
        m = _n(8 * decay)
        env = np.exp(-np.arange(m) / (decay * KSR))
        exc = np.zeros(n)
        for t in times:
            _place(exc, rng.uniform(-1, 1, m) * env, t, rng.uniform(0.7, 1.0))
        out += _bpf(exc, fc, rng.uniform(1.5, 4.0)) * rng.uniform(0.35, 1.0)
    extra = crowd - CLAP_VOICES
    if extra > 0:              # Farnell: beyond ~20 claps/s blend granular noise of the same spectrum
        hits = rng.poisson(extra * applause_rate(enthusiasm) / KSR, n) * rng.uniform(0.3, 1.0, n)
        env = _sig.lfilter([1.0], [1.0, -math.exp(-1.0 / (0.01 * KSR))], hits)
        bed = rng.uniform(-1, 1, n) * env
        out += 0.25 * sum(_bpf(bed, fc, 1.5) for fc in (1000.0, 1600.0, 2500.0))
    return _fin(_hp(out, 300.0), sr, peak=0.6, hp=100.0)


@sfx("crowd_murmur", "Crowd murmur / walla: many gated talking-pulse voices through drifting vowel formants.",
     people=(20, "voices (24 rendered at most)"), mood=(0.5, "0 hushed .. 1 lively (more talk, higher pitch)"),
     distance=(0.5, "0 among them .. 1 far (duller)"), dur=(5.0, "s"), seed=(0, "variation"))
def crowd_murmur(sr=DEFAULT_SR, people=20, mood=0.5, distance=0.5, dur=5.0, seed=0):
    rng = _rng(seed, 51)
    n = _n(dur)
    mood = float(np.clip(mood, 0.0, 1.0))
    keys = list(V.FARNELL_VOWELS)
    out = np.zeros(n)
    k = int(np.clip(people, 1, 24))
    for _ in range(k):                                                 # UNSOURCED: talk / pause statistics
        x = V.almost_speech(n / KSR, rng, f0_scale=rng.uniform(0.38, 0.75) * (1.0 + 0.2 * mood))[:n]
        gate = np.zeros(n)
        t = -rng.uniform(0.0, 1.5)
        while t < dur:
            a = rng.uniform(0.6, 2.5) * (0.6 + mood)
            gate[max(0, int(t * KSR)):max(0, int((t + a) * KSR))] = rng.uniform(0.5, 1.0)
            t += a + rng.uniform(0.2, 1.8) * (1.4 - mood)
        gate = V._pd_lop(gate, 8.0, float(KSR))
        path, t = [], 0.0
        while t < dur + 0.3:
            path.append((t, keys[int(rng.integers(len(keys)))]))
            t += rng.uniform(0.12, 0.3)
        out += V.formant_filter(x * gate, path, "farnell", formant_scale=rng.uniform(0.88, 1.2)) * rng.uniform(0.4, 1.0)
    return _fin(_lp(_hp(out, 120.0), 4200.0 - 2800.0 * float(np.clip(distance, 0, 1))), sr, peak=0.45)


def _whistle(f: np.ndarray, amp: np.ndarray, breath: float, rng) -> np.ndarray:
    """Human whistle: a near-sinusoid (mouth Helmholtz resonance locked by the lip jet) with a weak
    2nd harmonic and jet noise in a band around the pitch. UNSOURCED: harmonic and noise levels."""
    n = len(f)
    ph = TWO_PI * np.cumsum(f) / KSR
    y = np.sin(ph) + 0.04 * np.sin(2.0 * ph)
    nz = V._bp(rng.uniform(-1, 1, n), np.minimum(f, 0.4 * KSR), 8.0)
    nz = nz / (np.sqrt(np.mean(nz * nz)) + 1e-12)
    return (y * (1.0 + 0.04 * _slow(n, rng, 6.0)) + 0.12 * breath * nz) * amp


_WHISTLES = {  # UNSOURCED: [(seconds, [(u, frequency ratio), ...]), ...]; a gap is (seconds, None)
    "wolf": [(0.32, [(0, 0.55), (1, 1.3)]), (0.09, None), (0.62, [(0, 0.6), (0.35, 1.45), (1, 0.42)])],
    "up": [(0.4, [(0, 0.6), (1, 1.2)])],
    "down": [(0.45, [(0, 1.2), (1, 0.55)])],
    "note": [(0.7, [(0, 1.0), (1, 1.0)])],
    "taxi": [(0.18, [(0, 0.7), (1, 0.75)]), (0.55, [(0, 0.75), (0.25, 1.5), (1, 1.55)])],
}


@sfx("whistle_human", "Human whistle gestures: wolf-whistle, slides, a held note, a taxi hail.",
     kind=("wolf", "wolf|up|down|note|taxi"), freq=(1800.0, "reference pitch: Hz or note name"),
     breath=(0.3, "0..1 air noise"), speed=(1.0, "tempo scale"), seed=(0, "variation"))
def whistle_human(sr=DEFAULT_SR, kind="wolf", freq=1800.0, breath=0.3, speed=1.0, seed=0):
    rng = _rng(seed, 52)
    f0 = float(np.clip(to_hz(freq), 300.0, 4000.0))
    fs, es = [], []
    for d, pts in _WHISTLES[_choice(kind, _WHISTLES, "kind")]:
        n = _n(d / float(np.clip(speed, 0.25, 4.0)))
        if pts is None:
            fs.append(np.full(n, fs[-1][-1]))
            es.append(np.zeros(n))
            continue
        fs.append(f0 * np.exp(_arc(n, [(u, math.log(r)) for u, r in pts])))
        es.append(np.minimum(np.sin(np.pi * np.arange(n) / n) * 4.0, 1.0))
    f = np.concatenate(fs)
    if kind == "note":
        f = f * (1.0 + 0.012 * np.sin(TWO_PI * 5.5 * np.arange(len(f)) / KSR))
    return _fin(_whistle(f, np.concatenate(es), float(np.clip(breath, 0, 1)), rng), sr, peak=0.5)


# ============================================================================================
# locomotion: footstep sequences over the 24 ground kernels
# ============================================================================================

GROUNDS = tuple(S.GROUND_SPECS)
_HARD = {"stone", "concrete", "cobblestone", "wet_stone", "marble", "hard_floor", "wood", "metal_web", "ice",
         "crystal", "bone", "coral"}
# default footfalls per minute; walk / run inside and above Visell's 75-125 steps/min.
# UNSOURCED: sneak, limp, stairs, trot (78 strides/min) and gallop (110 strides/min) defaults.
GAITS = {"walk": 105.0, "run": 170.0, "sneak": 66.0, "limp": 84.0, "stairs": 96.0, "trot": 312.0, "gallop": 440.0}
# footfall onsets inside one stride. UNSOURCED fractions; the structure is the standard one:
# trot = two beats of diagonal pairs, gallop = four beats then a suspension phase.
_QUAD = {"trot": (0.0, 0.035, 0.5, 0.535), "gallop": (0.0, 0.17, 0.36, 0.52)}
SHOES = ("default", "barefoot", "boot", "heel", "hoof")


def footsteps_plan(gait: str = "walk", pace: float = 0.0, steps: int = 8, variation: float = 0.5,
                   seed: int = 0) -> np.ndarray:
    """Footfall onset times in seconds. Mean rate = ``pace`` footfalls per minute."""
    _choice(gait, GAITS, "gait")
    rng = _rng(seed, 60)
    pace = float(np.clip(pace if pace else GAITS[gait], 20.0, 900.0))
    steps = int(np.clip(steps if steps else (16 if gait in _QUAD else 8), 1, 400))
    v = float(np.clip(variation, 0.0, 1.0))
    ioi = 60.0 / pace
    if gait in _QUAD:
        stride = 4.0 * ioi
        t = np.array([stride * (i // 4 + _QUAD[gait][i % 4]) for i in range(steps)])
        return 0.01 + t + rng.normal(0.0, 0.012 * v * stride, steps).clip(-0.03 * stride, 0.03 * stride)
    gaps = np.full(steps, ioi)
    if gait == "limp":                         # the weak leg is unloaded early: short-long alternation
        gaps *= np.where(np.arange(steps) % 2 == 0, 0.72, 1.28)
    gaps *= 1.0 + np.clip(rng.normal(0.0, 0.03 * v, steps), -0.08, 0.08) * (0.5 if gait == "stairs" else 1.0)
    return 0.01 + np.concatenate(([0.0], np.cumsum(gaps[:-1])))


def _step(ground: str, i: int, gait: str, ioi: float, weight: float, shoe: str, v: float, seed) -> np.ndarray:
    """One freshly synthesised step (never a copy: own random stream, own GRF deviations)."""
    rng = _rng(seed, 1000 + i)
    y = S.render_step(ground, i % 2, rng)                              # 0 = right foot, 1 = left foot
    if abs(weight - 1.0) > 1e-3:               # UNSOURCED: heavier = slower, lower and bassier
        y = S._varispeed(y, weight ** -0.12)
        y = F.lowshelf(y, 200.0, KSR, 6.0 * math.log2(weight))
    pk = float(np.max(np.abs(y))) + 1e-12
    hard = 1.0 if ground in _HARD else 0.25
    if shoe == "barefoot":
        y = _lp(y, 1800.0)
    elif shoe == "boot":
        y = F.highshelf(F.lowshelf(y, 180.0, KSR, 4.0), 3000.0, KSR, -3.0)
    elif shoe == "heel":                       # hard heel tip: a short stiff knock on the strike
        c = M.strike("sy_stick", rng.uniform(1.4, 1.7), 0.6, 0.3, rng, velocity=0.6, dur=0.06)
        _place(y, _unit(c), 0.002, 0.5 * pk * hard)
    elif shoe == "hoof":                       # hollow keratin clop
        c = M.strike("sy_stick", rng.uniform(0.5, 0.62), 1.2, 0.5, rng, velocity=0.8, dur=0.09)
        c = _unit(c) if hard == 1.0 else _lp(_unit(c), 1200.0)
        y = _lp(y, 3500.0)
        _place(y, c, 0.002, 0.9 * pk * max(hard, 0.5))
    if gait in ("run", "limp", "trot", "gallop", "stairs"):            # staccato: flight phase / early unloading
        frac = {"run": 0.55, "stairs": 0.7, "limp": 0.75 if i % 2 == 0 else 1.0}.get(gait, 3.0)
        c = int(min(len(y), max(0.08, frac * ioi) * KSR))
        m = min(c, int(0.04 * KSR))
        y = y[:c].copy()
        y[-m:] *= np.cos(0.5 * np.pi * np.arange(m) / m) ** 2
    g = 10.0 ** (rng.normal(0.0, 1.5 * v) / 20.0) * (0.92 if i % 2 else 1.0)
    if gait == "sneak":                        # creeping minimises the change in GRF: soft onset, dull, quiet
        m = min(len(y), int(0.02 * KSR))
        y[:m] *= np.linspace(0.0, 1.0, m) ** 2
        y = _lp(y, 2200.0)
        g *= 0.35
    elif gait == "limp" and i % 2 == 0:
        g *= 0.6
    elif gait == "stairs":
        y = F.lowshelf(y, 150.0, KSR, 3.0)
    elif gait == "gallop":
        g *= (1.0, 0.85, 0.9, 1.1)[i % 4]
    return y * g


@sfx("footsteps", "Footstep sequence on any of the 24 grounds: every step re-synthesised from a GRF curve; walk, run, sneak, limp, stairs, trot, gallop.",
     ground=("wood", "|".join(GROUNDS)), gait=("walk", "walk|run|sneak|limp|stairs|trot|gallop (trot, gallop: four hooves)"),
     pace=(0.0, "footfalls per minute, 0 = gait default (walk 105, run 170, sneak 66, limp 84, stairs 96, trot 312, gallop 440)"),
     steps=(0, "number of footfalls, 0 = 8 (16 for trot / gallop)"), weight=(1.0, "body weight scale 0.4..3: heavier = lower, longer"),
     shoe=("default", "default|barefoot|boot|heel|hoof (trot / gallop use hoof unless set)"),
     variation=(0.5, "0..1 timing and level irregularity (steps never repeat even at 0)"), seed=(0, "variation"))
def footsteps(sr=DEFAULT_SR, ground="wood", gait="walk", pace=0.0, steps=0, weight=1.0, shoe="default",
              variation=0.5, seed=0):
    _choice(ground, GROUNDS, "ground")
    _choice(shoe, SHOES, "shoe")
    times = footsteps_plan(gait, pace, steps, variation, seed)
    if gait in _QUAD and shoe == "default":
        shoe = "hoof"
    weight = float(np.clip(weight, 0.4, 3.0))
    ioi = 60.0 / float(np.clip(pace if pace else GAITS[gait], 20.0, 900.0))
    out = np.zeros(_n(times[-1] + 1.2))
    end = 0
    for i, t in enumerate(times):
        y = _step(ground, i, gait, ioi, weight, shoe, float(np.clip(variation, 0, 1)), seed)
        _place(out, y, float(t))
        end = max(end, int(t * KSR) + len(y))
    return _fin(out[:min(len(out), end + int(0.02 * KSR))], sr, peak=0.7, fade_ms=2.0, hp=25.0)


# ============================================================================================
# instrument: human whistling (sung voice and choirs live in genny/choir.py)
# ============================================================================================

def _note_env(n: int, dur: float, attack: float, release: float) -> np.ndarray:
    e = np.ones(n)
    na = max(8, int(min(attack, 0.5 * dur) * KSR))
    e[:na] = np.sin(0.5 * np.pi * np.arange(na) / na) ** 2
    off = min(n - 1, int(dur * KSR))
    m = n - off
    e[off:] *= np.cos(0.5 * np.pi * np.arange(m) / m) ** 2
    return e


def _vib(n: int, depth_st: float, rng, rate: float = 5.5) -> np.ndarray:
    """Delayed vibrato (multiplier on f0) plus STK Modulate-style random drift."""
    t = np.arange(n) / KSR
    onset = np.clip((t - 0.12) / 0.25, 0.0, 1.0)                       # UNSOURCED onset delay
    dev = (2.0 ** (depth_st / 12.0) - 1.0) * onset * np.sin(TWO_PI * rate * t + rng.uniform(0, TWO_PI))
    return 1.0 + dev + V.stk_modulate(n, rng, vibrato_gain=0.0, random_gain=0.003)


def _inst_out(y, sr, vel, level, bright=1.0, cut=4800.0):
    y = _sig.sosfilt(_sig.butter(4, min(cut * bright, 0.45 * KSR), "lowpass", fs=KSR, output="sos"), y)
    if int(sr) != KSR:
        g = math.gcd(int(sr), KSR)
        y = _sig.resample_poly(y, int(sr) // g, KSR // g)
    return y * (level * float(np.clip(vel, 0.0, 1.5)) ** 1.2)


@instrument("whistling", "Human whistling: near-pure tone with breath noise around the pitch and a little vibrato.",
            family="winds", span=("C5", "C7"), vibrato=(0.15, "depth in semitones"), breath=(0.3, "0..1 air noise"),
            seed=(0, "variation"))
def whistling(freq, dur, sr=DEFAULT_SR, vel=1.0, vibrato=0.15, breath=0.3, seed=0):
    rng = _rng(seed, 73)
    n = _n(dur + 0.08)
    f = freq * _vib(n, float(vibrato), rng, 5.5)
    y = _whistle(f, _note_env(n, dur, 0.035, 0.07), float(np.clip(breath, 0, 1)), rng)
    return _inst_out(y, sr, vel, _LV_WHISTLE, cut=9000.0)


_LV_WHISTLE = 0.45     # about -12 dB K-weighted at vel 0.9 (examples/level_match.py refines)
