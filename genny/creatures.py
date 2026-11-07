"""Living things: mammals, insects, birds, frogs, body sounds, crowds, footstep sequences, whistling.

Catalog wrappers over the kernels in ``genny.physical.voice`` / ``footsteps`` / ``particles`` /
``modal`` plus the few models those do not have (cricket, wingbeat, FM syrinx with a song grammar,
heartbeat, applause, whistle). Kernels run at 48 kHz (``voice.SR``); every entry resamples at the
boundary (``scipy.signal.resample_poly``), removes DC, fades both ends and peak-normalises.
Sources: ``research/SOURCES.md``.
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
C_AIR = 340.0


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
    """Twice low-passed noise scaled to +-1."""
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
# mammals: flapping source -> tract comb + vowel formants
# ============================================================================================

def _voc(f0, env, rng, *, tract_m: float, harsh: float = 0.3, vowel=None, vq: float = 9.0, vmix: float = 0.6,
         jitter: float = 0.01, shimmer: float = 0.06, sub: float = 0.0, wrong: float = 0.0, biphon: float = 0.0) -> np.ndarray:
    """Mammal voice: ``f0``/``env`` per sample; ``harsh`` maps to ripple
    (1 -> 4), cord width (2 -> 12) and noisiness (0.05 -> 0.45) as in ``voice.roar``; tract =
    parallel band-pass comb at (2k-1) c/4l plus, when ``vowel`` is given, three formants from
    the vowel table scaled by 0.17 m / l. ``wrong`` adds a second shorter tube; ``biphon`` a second
    source at f0 / 0.59 that cross-modulates the first."""
    n = len(f0)
    a = env / (float(np.max(env)) + 1e-12)
    f0j, amp = V.roughness(np.maximum(f0, 5.0), rng, jitter=jitter, shimmer=shimmer, subharmonic=sub)
    src = V.flapping_source(f0j, ripple_env=1.0 + 3.0 * harsh * a, width_env=2.0 + 10.0 * harsh * a,
                            noisiness_env=0.05 + 0.4 * harsh * a, rng=rng) * amp
    if biphon:
        g = V.flapping_source(f0j / 0.59, ripple_env=1.0 + 3.0 * harsh * a, width_env=2.0 + 10.0 * harsh * a, rng=rng)
        src = src * (1.0 + float(biphon) * g / (np.max(np.abs(g)) + 1e-12))
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


_ROAR_TRACT = {"lion": 0.38, "tiger": 0.29, "jaguar": 0.33, "jaguar_female": 0.31, "cub": 0.035}


@sfx("roar", "Big-cat roar (Farnell lion arc 30->240->120 Hz, flapping cords into a long tract).",
     size=(1.0, "body scale 0.4..4: f0 = arc/size, tract = species length x size"), aggression=(0.7, "0 smooth .. 1 snarling"),
     dur=(2.5, "s"), species=("lion", "lion 0.38 m | tiger 0.29 | jaguar 0.33 | jaguar_female 0.31 | cub 0.035 (newborn jaguar): lips-glottis length"),
     seed=(0, "variation"))
def roar(sr=DEFAULT_SR, size=1.0, aggression=0.7, dur=2.5, species="lion", seed=0):
    size = float(np.clip(size, 0.4, 4.0))
    rng = _rng(seed, 1)
    n = _n(dur)
    rise = 0.45
    a = V.articulation_env(n, rise, offset=0.0)
    nr = int(round(n * rise))
    f0 = np.empty(n)
    f0[:nr] = 30.0 + (240.0 - 30.0) * a[:nr]
    f0[nr:] = 120.0 + (240.0 - 120.0) * a[nr:]
    ag = float(np.clip(aggression, 0.0, 1.0))
    y = _voc(f0 / size, a, rng, tract_m=_ROAR_TRACT[_choice(species, _ROAR_TRACT, "species")] * size, harsh=0.15 + 0.85 * ag, jitter=0.005 + 0.03 * ag,
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
    f0 = to_hz(pitch) / size * (1.0 + 0.12 * _slow(n, rng, 3.0)) * (0.85 + 0.25 * env)
    y = _voc(f0, env * (1.0 + 0.25 * _slow(n, rng, 7.0)), rng, tract_m=0.25 * size, harsh=0.4 + 0.6 * ag,
             jitter=0.03, shimmer=0.25, sub=0.6 * ag, wrong=0.7)
    return _fin(_dark(y), sr)


_DOGS = {"small": (720.0, 0.069, 0.10), "medium": (430.0, 0.157, 0.15), "large": (240.0, 0.224, 0.21)}


DOG_VTL_FIT = (3.15, 11.53)


DOMINANT_HZ = (626.0, 1267.0)
DOMINANT_GAIN = 1.0


def dominant_hz(aggression: float) -> float:
    """Dominant frequency of a bark: sqrt(626 x 1267) x (1267/626)^(0.5 - aggression)."""
    lo, hi = DOMINANT_HZ
    return math.sqrt(lo * hi) * (hi / lo) ** (0.5 - float(np.clip(aggression, 0.0, 1.0)))


STRIDE = {"horse": (1.81, 0.019, 5.63, 11.1), "mouse": (6.86, 0.780, 0.0, 2.0), "dog": (2.27, 0.050, 0.0, 15.0)}


def stride_hz(species: str, speed: float) -> float:
    """Gallop stride frequency (Hz) at `speed` m/s: a + b v per species (horse 680 kg, mouse, dog 25 kg)."""
    a, b, lo, hi = STRIDE[_choice(species, STRIDE, "species")]
    return a + b * float(np.clip(speed, lo, hi))


def creature_profile(size: float = 1.0) -> dict:
    """One body scale for every creature sound: f0 / size, tract x size, wing rate / size^0.3,
    impact and footstep pitch / size^(1/3) (mass ~ size^3), loudness +20 log10(size) dB, capped."""
    k = float(np.clip(size, 0.1, 10.0))
    return {"f0_mul": 1.0 / k, "tract_m": 0.17 * k, "wing_mul": k ** -0.3, "impact_mul": k ** (-1.0 / 3.0),
            "mass_mul": k ** 3, "gain_db": float(np.clip(20.0 * math.log10(k), -20.0, 12.0))}


def dog_vtl(mass: float) -> float:
    """Vocal tract length (m) of a dog of `mass` kg: VTL cm = 3.15 + 11.53 log10 m."""
    a, b = DOG_VTL_FIT
    return (a + b * math.log10(float(np.clip(mass, 1.0, 90.0)))) / 100.0


@sfx("bark", "Dog barks: short arched pulses bursts, 'wa-u' vowel path.",
     size=("medium", "small|medium|large"), count=(2, "barks"),
     aggression=(0.6, "0..1: shorter gaps; with pitch 0 also a lower f0 (x1.12 at 0, x0.77 at 1)"),
     pitch=(0.0, "f0 peak in Hz, 0 = from size"),
     mass=(0.0, "dog mass in kg (2.5..50 measured), sets the tract length; 0 = from size"),
     dominant=(False, "spectral peak from aggression: 1267 Hz at 0 .. 626 Hz at 1"), seed=(0, "variation"))
def bark(sr=DEFAULT_SR, size="medium", count=2, aggression=0.6, pitch=0.0, mass=0.0, dominant=False, seed=0):
    fpk, tract, d = _DOGS[_choice(size, _DOGS, "size")]
    fpk = to_hz(pitch) if pitch else fpk * float(np.interp(aggression, [0.0, 0.6, 1.0], [1.116, 1.0, 0.767]))
    tract = dog_vtl(mass) if mass else tract
    rng = _rng(seed, 3)
    count = int(np.clip(count, 1, 40))
    out = np.zeros(_n(count * (1.15 * d + 0.55) + 0.3))
    t = 0.02
    for _ in range(count):
        dd = d * rng.uniform(0.9, 1.15)
        n = _n(dd)
        env = V.articulation_env(n, 0.18, offset=0.0)
        f0 = fpk * rng.uniform(0.94, 1.06) * _arc(n, [(0, 0.6), (0.2, 1.0), (0.6, 0.85), (1, 0.5)])
        y = _voc(f0, env, rng, tract_m=tract, harsh=0.35 + 0.6 * aggression, vowel=_vowel_keys("A-u", dd),
                 shimmer=0.2, jitter=0.02, sub=0.3 * aggression)
        if dominant:
            y = _unit(y) + DOMINANT_GAIN * _unit(_bpf(y, dominant_hz(aggression), 2.0))
        _place(out, _unit(y), t, rng.uniform(0.8, 1.0))
        t += dd + (0.5 - 0.4 * float(np.clip(aggression, 0.0, 1.0))) + rng.uniform(-0.05, 0.05)
    return _fin(_dark(out[:_n(t + 0.1)]), sr)


@sfx("meow", "Cat meow: ~500 Hz smooth cords (Farnell), vowel path i-ae-A-u through a short tract.",
     size=(1.0, "body scale (0.6 kitten, about 1.4 kHz .. 2 big cat)"), pitch=(550.0, "f0 peak Hz at size 1"), dur=(0.9, "s"),
     seed=(0, "variation"))
def meow(sr=DEFAULT_SR, size=1.0, pitch=550.0, dur=0.9, seed=0):
    size = float(np.clip(size, 0.4, 4.0))
    rng = _rng(seed, 4)
    n = _n(dur)
    env = V.articulation_env(n, 0.3, offset=0.0) ** 0.7
    f0 = to_hz(pitch) / size ** (1.8 if size < 1.0 else 1.0) * _arc(n, [(0, 0.72), (0.3, 1.0), (0.6, 0.96), (1, 0.68)]) \
        * (1 + 0.01 * _slow(n, rng, 6.0))
    y = _voc(f0, env, rng, tract_m=0.08 * size, harsh=0.12, vowel=_vowel_keys("i:-ae-A-u", dur), vmix=0.75)
    return _fin(_dark(y), sr)


PURR_VOICE_HZ = 200.0


def purr_breath(t, phase=0.0, period=1.7, expiration=0.4):
    """Breath cycle, +1 at mid-expiration and -1 at mid-inspiration; expiration lasts `expiration` of the period."""
    u = np.mod(np.asarray(t, float) / period + phase, 1.0)
    return np.where(u < expiration, np.sin(np.pi * u / expiration), -np.sin(np.pi * (u - expiration) / (1.0 - expiration)))


@sfx("purr", "Cat purr: glottal pulses near 26 Hz through the tract, breathing in and out.",
     rate=(26.3, "pulses per second (mean)"), size=(1.0, "body scale"), dur=(3.0, "s"), seed=(0, "variation"),
     voice=(0.0, "0..1 voicing during the purr: a ~200 Hz tone (/ size) swelling with each purr cycle, on expiration"))
def purr(sr=DEFAULT_SR, rate=26.3, size=1.0, dur=3.0, seed=0, voice=0.0):
    rng = _rng(seed, 5)
    n = _n(dur)
    t = np.arange(n) / KSR
    br = purr_breath(t, rng.uniform(0, 1.0))
    f = float(np.clip(rate, 5.0, 80.0)) * (1.0 + 0.046 * np.tanh(4.0 * br)) * (1.0 + 0.01 * _slow(n, rng, 5.0))
    amp = 0.55 + 0.45 * np.abs(np.tanh(4.0 * br)) ** 0.5           # dip at each breath reversal
    src = V.raised_cosine_pulse(f, width_env=5.0)
    src = src + 0.06 * rng.uniform(-1, 1, n) * np.maximum(src, 0.0)
    if voice > 0:
        gate = np.maximum(V.raised_cosine_pulse(f, width_env=1.0), 0.0) * np.maximum(br, 0.0)
        src = src + 2.0 * float(np.clip(voice, 0.0, 1.0)) * gate * V.raised_cosine_pulse(
            np.full(n, PURR_VOICE_HZ / float(np.clip(size, 0.4, 4.0))), width_env=1.0)
    y = V.vocal_tract_comb(src, 0.08 * float(np.clip(size, 0.4, 4.0)), q=2.5) + 0.15 * _lp(src, 300.0)   # 80 mm, as meow
    return _fin(_lp(y * amp, 2200.0), sr)


C_TRACT = 350.0
DIANA_TUBES = {                            # (diameter cm, length cm) from the larynx to the lips
    4: ((1.0, 0.88), (0.3, 3.0), (2.4, 5.75), (1.5, 0.25)),
    7: ((1.0, 1.27), (0.3, 3.0), (2.0, 5.75), (0.71, 0.25)),
    8: ((1.0, 1.23), (0.3, 3.0), (2.0, 5.75), (0.76, 0.25)),
}
DIANA_CALL = dict(f0=130.0, dur=0.18, gap=0.17, calls=4)


def tube_formants(tubes, c: float = C_TRACT, fmax: float = 8000.0) -> np.ndarray:
    """Resonances of lossless cylinders (diameter cm, length cm) closed at the source end, p = 0 at the mouth."""
    f = np.arange(20.0, fmax, 0.5)
    k = TWO_PI * f / c
    A = np.ones_like(f)
    B = np.zeros_like(f)
    Cm = np.zeros_like(f)
    D = np.ones_like(f)
    for dia, length in tubes:
        z = 1.0 / (math.pi * (0.5 * dia) ** 2)
        co, si = np.cos(k * length * 0.01), np.sin(k * length * 0.01)
        A, B, Cm, D = A * co - B * si / z, A * z * si + B * co, Cm * co + D * si / z, D * co - Cm * z * si
    s = np.sign(D)
    i = np.nonzero(s[:-1] * s[1:] < 0)[0]
    return f[i] - D[i] * (f[i + 1] - f[i]) / (D[i + 1] - D[i])


def diana_alarm(seed: int = 0, calls: int = 0) -> np.ndarray:
    """Male Diana monkey leopard alarm series at 48 kHz: glottal pulses through the tube model moving
    from configuration 4 (onset) to 7 (middle) to 8 (end) within each call."""
    rng = _rng(seed, 41)
    p = DIANA_CALL
    F = np.array([tube_formants(DIANA_TUBES[i])[:3] for i in (4, 7, 8)])
    m = _n(p["dur"])
    path = np.stack([np.interp(np.linspace(0, 1, m), [0, 0.5, 1], F[:, j]) for j in range(3)], axis=1)
    ring = _n(0.03)                                      # the tract rings on after the source stops
    path = np.concatenate([path, np.repeat(path[-1:], ring, axis=0)])
    calls = int(calls) or p["calls"]
    out = np.zeros(_n(calls * (p["dur"] + p["gap"]) + 0.05))
    for c in range(calls):
        f0 = p["f0"] * rng.uniform(0.95, 1.05) * _arc(m, [(0, 0.9), (0.3, 1.0), (1, 0.85)])
        env = V.articulation_env(m, 0.15, offset=0.0) ** 0.6
        src = V.raised_cosine_pulse(f0, width_env=4.0) + 0.05 * rng.uniform(-1, 1, m)
        y = V.formant_filter(np.concatenate([src * env, np.zeros(ring)]), formants=dict(freqs=path, bws=path / 8.0, gains=np.array([1.0, 0.7, 0.3])))
        y[-ring:] *= np.cos(0.5 * np.pi * np.arange(1, ring + 1) / ring) ** 2
        _place(out, _unit(y), 0.02 + c * (p["dur"] + p["gap"]), rng.uniform(0.8, 1.0))
    return out


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
     size=(1.0, "body scale"), pitch=(420.0, "f0 plateau Hz at size 1"), dur=(3.0, "s"), seed=(0, "variation"),
     biphonation=(0.0, "0..1: a second source at f0/0.59, locked to f0 and its jitter, cross-modulating the first (peaks at m f0 + n g0)"))
def howl(sr=DEFAULT_SR, size=1.0, pitch=420.0, dur=3.0, seed=0, biphonation=0.0):
    size = float(np.clip(size, 0.4, 4.0))
    rng = _rng(seed, 7)
    n = _n(dur)
    env = V.articulation_env(n, 0.2, offset=0.0) ** 0.5
    f0 = to_hz(pitch) / size * _arc(n, [(0, 0.55), (0.14, 0.94), (0.3, 1.0), (0.8, 0.96), (1, 0.7)])
    f0 = f0 * (1.0 + 0.006 * np.sin(TWO_PI * 5.0 * np.arange(n) / KSR) * np.linspace(0, 1, n))
    y = _voc(f0, env, rng, tract_m=0.2 * size, harsh=0.06, vowel=_vowel_keys("A-u-u", dur), vmix=0.7,
             jitter=0.004, shimmer=0.03, biphon=float(np.clip(biphonation, 0.0, 1.0)))
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
    path = [(dur * i / 3.0, keys[int(rng.integers(len(keys)))]) for i in range(4)]
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
    "field": (4500.0, 0.133, 7, 0.017, 0.022, 1.0),
    "tree": (2900.0, 0.05, 8, 1.0 / 52.0, 0.01, 1.0),
    "bell": (3000.0, 0.0, 0, 0.0, 0.0, 2.52),
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


FALL = dict(period=0.3, period_exp=0.4, shrink=0.55, energy=0.5, thump=1.5, impacts=3)
DEBRIS_T = ((0.075, 0.030), (0.170, 0.174), (0.075, 0.074))           # s: base + U(0, spread) after the previous
DEBRIS_G = (0.416, 0.257, 0.05)
DEBRIS_LAW = {"metal": (1.6, 0.5), "bone": (1.0, 0.3), "shell": (0.6, 0.4)}   # pitch factor a / size^b


def fall_schedule(size: float, seed: int = 0) -> list[tuple[float, float]]:
    """(time s, amplitude) of a collapsing body: first period 0.3 size^0.4 x U(0.9, 1.1), each next one x 0.55,
    energy x 0.5 per bounce (amplitude x sqrt 0.5)."""
    rng = _rng(seed, 41)
    p = FALL["period"] * float(size) ** FALL["period_exp"] * rng.uniform(0.9, 1.1)
    t, a, out = 0.0, 1.0, []
    for _ in range(FALL["impacts"]):
        out.append((t, a))
        t += p
        p *= FALL["shrink"]
        a *= math.sqrt(FALL["energy"])
    return out


@sfx("body_fall", "A creature collapsing: the body hits the ground three times, ever closer and softer, with a "
     "heavy thump, then a few pieces of its armour or bones clatter after it.",
     size=(1.0, "creature size 0.2..8 (bigger = slower, lower, longer thump)"),
     material=("flesh", "flesh|stone|shell|metal|bone: the body"),
     debris=("", "none | metal | bone | shell: pieces falling after the body"), seed=(0, "variation"))
def body_fall(sr=DEFAULT_SR, size=1.0, material="flesh", debris="", seed=0):
    from .foley import BODY_LAW, body_hit
    _choice(material, BODY_LAW, "material")
    if debris:
        _choice(debris, DEBRIS_LAW, "debris")
    k = float(np.clip(size, 0.2, 8.0))
    rng = _rng(seed, 42)
    sched = fall_schedule(k, seed)
    y = np.zeros(_n(sched[-1][0] + 1.6))
    thump_s = float(np.clip(0.010 * k ** 1.1, 0.003, 0.12))
    th = np.sin(np.pi * np.arange(_n(thump_s)) / max(1, _n(thump_s))) * _lp(rng.standard_normal(_n(thump_s)), 180.0 / k ** 0.5)
    th = _unit(_lp(np.sin(np.pi * np.arange(_n(thump_s)) / max(1, _n(thump_s))), 120.0) + 0.3 * _unit(th))
    for i, (t, a) in enumerate(sched):
        hit = _unit(np.asarray(body_hit(KSR, material, k, seed + i), float))
        _place(y, hit, t, a)
        _place(y, th, t, a * (FALL["thump"] if i == 0 else 1.0))
    if debris:
        fa, fb = DEBRIS_LAW[debris]
        la, lb = BODY_LAW[debris]
        s = float(np.clip((la * k ** fb / fa) ** (1.0 / lb), 0.2, 8.0))
        t = sched[0][0]
        for j, ((base, spread), g) in enumerate(zip(DEBRIS_T, DEBRIS_G)):
            t += base + rng.uniform(0.0, spread)
            _place(y, _unit(np.asarray(body_hit(KSR, debris, s, seed + 10 + j), float)), t, g / (1.0 + j))
    return _fin(y, sr, peak=0.8, hp=25.0)


STRIDULATION = dict(band=(2000.0, 18000.0), teeth=2500.0, down=0.55, lobster_modes=(900.0, 2300.0, 4700.0),
                    lobster_speed=0.05, lobster_force=1.0, lobster_mass=0.002)


def _grasshopper(n, rate, rng):
    """Hind femur pegs rubbed over a wing vein: per leg stroke a burst of peg strikes (Poisson, STRIDULATION
    teeth/s) exciting band-limited friction noise; the up-stroke loud, the down-stroke quieter."""
    t = np.arange(n) / KSR
    ph = np.mod(rate * t, 1.0)
    env = np.where(ph < 0.5, np.sin(TWO_PI * ph) ** 2, STRIDULATION["down"] * np.sin(TWO_PI * (ph - 0.5)) ** 2)
    hits = (rng.random(n) < STRIDULATION["teeth"] / KSR).astype(float) * rng.uniform(0.5, 1.0, n)
    lo, hi = STRIDULATION["band"]
    grain = _hp(_lp(hits + 0.15 * rng.standard_normal(n), min(hi, 0.45 * KSR)), lo)
    return grain * env


def _lobster(n, rate, rng):
    """Spiny lobster rasp: a soft plectrum sliding over a file, elasto-plastic stick-slip on three modes; no impacts."""
    from .friction import elastoplastic_friction
    p = STRIDULATION
    t = np.arange(n) / KSR
    ph = np.mod(rate * t, 1.0)
    v = p["lobster_speed"] * np.where(ph < 0.6, np.sin(np.pi * ph / 0.6), 0.0) * (1.0 + 0.1 * _slow(n, rng, 3.0))
    r = elastoplastic_friction(np.array(p["lobster_modes"]), 0.02, 1.0 / p["lobster_mass"], p["lobster_force"],
                               v, n / KSR)
    return np.gradient(np.gradient(np.asarray(r["out"], float)[:n]))           # radiated ~ acceleration


@sfx("stridulation", "Stridulation that is not a cricket's: a grasshopper rubbing pegs on a wing vein (friction noise, "
     "two stroke directions) or a spiny lobster's soft stick-slip rasp.",
     kind=("grasshopper", "grasshopper | lobster"), rate=(0.0, "strokes per second (0 = 12 grasshopper, 4 lobster)"),
     dur=(2.0, "s"), seed=(0, "variation"))
def stridulation(sr=DEFAULT_SR, kind="grasshopper", rate=0.0, dur=2.0, seed=0):
    _choice(kind, ("grasshopper", "lobster"), "kind")
    rng = _rng(seed, 31)
    r = float(rate) if rate else (12.0 if kind == "grasshopper" else 4.0)
    n = _n(dur)
    y = _grasshopper(n, r, rng) if kind == "grasshopper" else _lobster(n, r, rng)
    return _fin(y, sr, peak=0.5, hp=300.0)


@sfx("cicada", "Cicada: noise-FM click train through timbal resonances with 12 Hz pulsing (Farnell circada2).",
     rate=(120.0, "click rate Hz"), tone=(0.85, "resonance scale (1 = 5-8 kHz as in the patch; lower = larger insect)"),
     dur=(3.0, "s"), seed=(0, "variation"))
def cicada(sr=DEFAULT_SR, rate=120.0, tone=0.85, dur=3.0, seed=0):
    y = V.chitin_clicks(float(np.clip(rate, 10.0, 600.0)), float(dur), _rng(seed, 11), kind="cicada",
                        res_scale=float(np.clip(tone, 0.1, 1.5)))
    n = len(y)
    swell = 0.6 + 0.4 * np.sin(np.pi * np.arange(n) / n) ** 0.5
    return _fin(y * swell, sr, peak=0.45, hp=200.0)


_FLY_H = np.array([0.877, 1.000, 0.503, 0.200, 0.216, 0.066, 0.044, 0.039, 0.029, 0.013])
_BUZZ = {
    "fly": (220.0, None, None), "mosquito": (600.0, 8000.0, 0.03), "bee": (230.0, 6000.0, 0.03)}


def _buzz(kind: str, sr, freq, wobble, motion, proximity, tone, dur, seed) -> np.ndarray:
    rng = _rng(seed, 12)
    n = _n(dur)
    t = np.arange(n) / KSR
    prox = float(np.clip(proximity, 0.0, 1.0))
    d0 = 0.15 + 3.0 * (1.0 - prox)
    mo = float(np.clip(motion, 0.0, 1.0))
    rate = rng.uniform(0.3, 0.6)
    d = d0 * (1.0 + 0.8 * mo * np.sin(TWO_PI * rate * t + rng.uniform(0, TWO_PI)))
    dop = 1.0 / (1.0 + np.gradient(d) * KSR / 343.0)                  # Doppler from the radial speed
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
    for wing, det in ((0.0, 1.0), (1.3, 1.004)):
        for i, a in enumerate(amps):
            y += a * np.sin((i + 1) * det * ph + wing)
    if band:
        y = 0.25 * _unit(y) + _unit(_bpf(y, float(np.clip(tone, 300.0, 12000.0)), 12.0))
    else:
        y = _bpf(y, float(np.clip(tone, 300.0, 12000.0)), 0.3)
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
    """Random on/off behaviour gate: clip(1e9 lop(lop(noise, 0.2), 0.2), 0, 1), de-clicked."""
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


_WINGS = {"bird": (6.0, 1.0), "bat": (11.0, 1.6), "insect": (50.0, 3.0)}


@sfx("wings", "Wing flaps (Farnell fan pulse into band-passed air, swept by the stroke).",
     kind=("bird", "bird|bat|insect"), rate=(0.0, "flaps per second, 0 = from kind (6 / 11 / 50)"),
     size=(1.0, "wing scale: bigger = lower band"), dur=(2.0, "s"), seed=(0, "variation"))
def wings(sr=DEFAULT_SR, kind="bird", rate=0.0, size=1.0, dur=2.0, seed=0):
    r0, sc = _WINGS[_choice(kind, _WINGS, "kind")]
    r = float(np.clip(rate if rate else r0, 0.5, 80.0))
    sc = sc / float(np.clip(size, 0.2, 6.0)) ** 0.3
    rng = _rng(seed, 14)
    n = _n(dur)
    ph = np.pi * r * np.arange(n) / KSR
    pulse = 1.0 / (1.0 + 5.0 * np.sin(ph) ** 2)
    nz = V._bp(rng.uniform(-1, 1, n), 700.0 * sc, 1.0)
    x = (nz * pulse * 0.4 + 0.1 * (pulse - pulse.mean())) * 0.2
    y = V._vcf(x, (300.0 + 300.0 * np.cos(2.0 * ph)) * sc, 5.0)
    return _fin(y * (1.0 + 0.2 * _slow(n, rng, 2.0)), sr, peak=0.5)



# ============================================================================================
# birds: FM syrinx driven by a syllable grammar
# ============================================================================================

@numba.njit(cache=True)
def _fm_syrinx(f, w, fb, nz, sr):
    """Syrinx pulse oscillator y = 1/(1 + (w cos(pi phase))^2), DC removed and scaled to +-~1,
    with feedback FM (a loose syrinx) and noise FM."""
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


@numba.njit(cache=True)
def _nf_syrinx(alpha, beta, gamma, fs, dl, r):
    """Labial normal form, RK4: x' = y,
    y' = -a g^2 - b g^2 x - g^2 x^3 - g x^2 y + g^2 x^2 - g x y, into the trachea
    Pi(t) = a x(t) - r Pi(t - T) (T = ``dl`` samples); returns Pi."""
    n = alpha.shape[0]
    x, y = 0.01, 0.0
    buf = np.zeros(dl)
    out = np.empty(n)
    h = 1.0 / fs
    for i in range(n):
        a, b, g = alpha[i], beta[i], gamma[i]
        g2 = g * g
        k1x = y
        k1y = -a * g2 - b * g2 * x - g2 * x ** 3 - g * x * x * y + g2 * x * x - g * x * y
        xx, yy = x + 0.5 * h * k1x, y + 0.5 * h * k1y
        k2x = yy
        k2y = -a * g2 - b * g2 * xx - g2 * xx ** 3 - g * xx * xx * yy + g2 * xx * xx - g * xx * yy
        xx, yy = x + 0.5 * h * k2x, y + 0.5 * h * k2y
        k3x = yy
        k3y = -a * g2 - b * g2 * xx - g2 * xx ** 3 - g * xx * xx * yy + g2 * xx * xx - g * xx * yy
        xx, yy = x + h * k3x, y + h * k3y
        k4x = yy
        k4y = -a * g2 - b * g2 * xx - g2 * xx ** 3 - g * xx * xx * yy + g2 * xx * xx - g * xx * yy
        x += h / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        y += h / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        j = i % dl
        p = a * x - r * buf[j]
        buf[j] = p
        out[i] = p
    return out


# f0 / gamma of the normal form at alpha = 0.15 against beta (measured by RK4, f0 scales exactly
# with gamma: all terms carry g^2 once time is g t)
_NF_BETA = np.array([0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7])
_NF_G = np.array([0.0767, 0.089, 0.0993, 0.1147, 0.1267, 0.137, 0.1547])


def _normal_form_source(f, amp, chaos, noise, tf1, rng):
    """Labial source + trachea: alpha (air-sac pressure) follows the syllable
    envelope across the Hopf threshold, beta (labial tension) sets the timbre (small beta: near the
    saddle-node, rich spectrum), gamma scales time so f0 follows the contour. Trachea: quarter-wave
    tube with its first resonance at ``tf1``."""
    os_ = 4
    fs = KSR * os_
    m = len(f) * os_
    up = lambda v: np.repeat(v, os_)                                   # noqa: E731
    alpha = up(-0.05 + 0.2 * np.clip(amp, 0.0, 1.0))
    b0 = np.clip(0.3 - 0.15 * chaos, 0.1, 0.7)
    jitter = _unit(V._pd_lop(rng.uniform(-1, 1, m), 3000.0, float(fs)))
    beta = up(b0) + up(40.0 * noise) / 5.0 * 0.025 * 0.6 * jitter      # Noise 0..40, beta range 0.1..0.7
    gamma = up(f / np.interp(b0, _NF_BETA, _NF_G))
    dl = max(2, int(round(fs / (2.0 * tf1))))                          # T = 2L / c, f1 = c / 4L
    y = _nf_syrinx(alpha, np.clip(beta, 0.05, 0.8), gamma, float(fs), dl, 0.5)
    return _sig.resample_poly(y, 1, os_)[:len(f)]


GARDNER = dict(a0=0.02, b0=0.015, tau=1e-5, zeta=0.05, xs=0.01, r=0.5)   # cm, cm, s, -, cm, -


@numba.njit(cache=True)
def _gardner_kernel(ps, w0, k2, kexp, fs, dl, coupling, a0, b0, tau, zeta, xs, r):
    """Flapping labia (unit mass, RK4): x' = y, y' = -k(x) x - beta(x, y) y + kap p_av,
    k = w0^2 + k2 x^kexp, beta = beta1 (1 + x^2/xs^2 + y^2/(w0 xs)^2), entry a0 + x + tau y, exit b0 + x - tau y,
    p_av = q + (ps - q)(1 - exit / entry) (closed valve: ps), q = coupling x the trachea pressure; kap puts the
    Hopf threshold at ps = 1. Trachea p_i(t) = s(t) - r p_i(t - T), s = (x - tau y)/xs + (y - tau y')/(w0 xs)."""
    n = ps.shape[0]
    h = 1.0 / fs
    x, y = 1e-4 * xs, 0.0
    buf = np.zeros(dl)
    out = np.empty(n)
    q = 0.0
    for i in range(n):
        w = w0[i]
        b1 = 2.0 * zeta * w
        kap = b1 * a0 * a0 / (tau * (a0 + b0))
        p = ps[i]
        kx, ky = 0.0, 0.0
        sx, sy = x, y
        acc_x, acc_y = 0.0, 0.0
        for st in range(4):
            if st == 0:
                xx, yy = x, y
            elif st == 3:
                xx, yy = x + h * kx, y + h * ky
            else:
                xx, yy = x + 0.5 * h * kx, y + 0.5 * h * ky
            en = a0 + xx + tau * yy
            ex = b0 + xx - tau * yy
            pav = p if (en <= 1e-6 or ex <= 1e-6) else q + (p - q) * (1.0 - ex / en)
            kk = max(w * w + k2 * xx ** kexp, 0.25 * w * w)
            bt = b1 * (1.0 + (xx / xs) ** 2 + (yy / (w * xs)) ** 2)
            kx, ky = yy, -kk * xx - bt * yy + kap * pav
            wgt = 1.0 if (st == 0 or st == 3) else 2.0
            acc_x += wgt * kx
            acc_y += wgt * ky
        x = sx + h / 6.0 * acc_x
        yd = acc_y / 6.0
        y = sy + h * yd
        j = i % dl
        s = (x - tau * y) / xs + (y - tau * yd) / (w * xs)
        pi = s - r * buf[j]
        buf[j] = pi
        out[i] = pi
        q = coupling * 0.1 * pi
    return out


def gardner_source(ps, f, tf1, coupling=0.0, k2=0.0, kexp=2, **kw):
    """Labial source + trachea at 4 x KSR: ``ps`` air-sac pressure in units of the phonation threshold (per
    sample at KSR), ``f`` the linear labial frequency (Hz), ``k2`` (1/(s^2 cm^kexp)) the nonlinear restitution
    that lets pressure alone raise the pitch (suboscines: kiskadee, tyrant), ``coupling`` 0..1 how much
    of the trachea pressure acts back on the labia."""
    p = dict(GARDNER, **kw)
    os_ = 4
    fs = KSR * os_
    up = lambda v: np.repeat(np.asarray(v, float), os_)               # noqa: E731
    dl = max(2, int(round(fs / (2.0 * tf1))))
    y = _gardner_kernel(up(ps), TWO_PI * up(np.maximum(f, 50.0)), float(k2), int(kexp), float(fs), dl,
                        float(coupling), p["a0"], p["b0"], p["tau"], p["zeta"], p["xs"], p["r"])
    return _sig.resample_poly(y, 1, os_)[:len(ps)]


SUBOSCINE = {"suboscine": (1e12, 2), "tyrant": (1e14, 3)}             # k2, exponent of the restitution
WC = dict(a=10.0, b=10.0, c=10.0, d=-2.0, k=30.0, amp=5.0, rho_e=-5.0, rho_i=-9.0, fs=2000.0)


def wc_pressure(dur: float, rate: float, **kw) -> np.ndarray:
    """Air-sac pressure gestures from a Wilson-Cowan pair (expiratory E, inspiratory I) whose drive
    rho_e + amp cos(2 pi rate t) stands for the periodic forcing of the HVC-RA-brainstem loop:
    E' = k(-E + S(rho_e(t) + a E - b I)), I' = k(-I + S(rho_i + c E - d I)), S = logistic. Returns E (0..~0.9)
    at WC["fs"]: slow forcing gives separate pulses, fast forcing a sustained pressure with pulses on top."""
    p = dict(WC, **kw)
    fs = p["fs"]
    n = max(2, int(dur * fs))
    e = i = 0.0
    out = np.empty(n)
    for j in range(n):
        r = p["rho_e"] + p["amp"] * math.cos(TWO_PI * rate * j / fs)
        de = p["k"] * (-e + 1.0 / (1.0 + math.exp(-(r + p["a"] * e - p["b"] * i))))
        di = p["k"] * (-i + 1.0 / (1.0 + math.exp(-(p["rho_i"] + p["c"] * e - p["d"] * i))))
        e += de / fs
        i += di / fs
        out[j] = e
    return out


def spectral_content_index(y, sr, f0=None):
    """SCI = mean spectral frequency / fundamental (f0 from the autocorrelation peak, 100 Hz..10 kHz, if not given)."""
    y = np.asarray(y, float) - np.mean(y)
    P = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    fr = np.fft.rfftfreq(len(y), 1.0 / sr)
    msf = float(np.sum(fr * P) / (np.sum(P) + 1e-30))
    if f0 is None:
        ac = np.fft.irfft(np.abs(np.fft.rfft(y, 2 * len(y))) ** 2)[:len(y)]
        lo, hi = int(sr / 10000.0), int(sr / 100.0)
        f0 = sr / (lo + int(np.argmax(ac[lo:hi])))
    return msf / float(f0)


LAJE_C = 1e8                  # nonlinear dissipation, (dyn s)/(cm^3 g)


@numba.njit(cache=True)
def _laje_syrinx(k, b, f0, fs):
    """Labia x' = y, y' = -k x - c x^2 y + b y - f0 (per unit mass), RK4; returns the velocity y."""
    n = k.shape[0]
    x, y = 1e-4, 0.0
    h = 1.0 / fs
    out = np.empty(n)
    c = LAJE_C
    for i in range(n):
        kk, bb, ff = k[i], b[i], f0[i]
        k1x, k1y = y, -kk * x - c * x * x * y + bb * y - ff
        xx, yy = x + 0.5 * h * k1x, y + 0.5 * h * k1y
        k2x, k2y = yy, -kk * xx - c * xx * xx * yy + bb * yy - ff
        xx, yy = x + 0.5 * h * k2x, y + 0.5 * h * k2y
        k3x, k3y = yy, -kk * xx - c * xx * xx * yy + bb * yy - ff
        xx, yy = x + h * k3x, y + h * k3y
        k4x, k4y = yy, -kk * xx - c * xx * xx * yy + bb * yy - ff
        x += h / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        y += h / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        out[i] = y
    return out


def laje_path(f_start: float, f_end: float, n: int, gap: int, level: float = 1.0):
    """One syllable as an ellipse in the (b, k) plane: phonation along the upper arc (b > 0, labia free,
    k swept from (2 pi f_start)^2 to (2 pi f_end)^2), then the mini breath along the lower arc with the
    labia held shut. Returns k, b, f0 over n + gap samples."""
    k0, k1 = (TWO_PI * f_start) ** 2, (TWO_PI * f_end) ** 2
    th = np.concatenate((np.pi * np.arange(n) / max(n, 1), np.pi + np.pi * np.arange(gap) / max(gap, 1)))
    k = 0.5 * (k0 + k1) - 0.5 * (k1 - k0) * np.cos(th)
    bmax = 0.2 * math.sqrt(max(k0, k1)) * level
    b = bmax * np.sin(th)
    f0 = np.where(th < np.pi, 0.0, 2.0 * k * math.sqrt(bmax / LAJE_C))
    return k, b, f0


def _laje_source(plan, n):
    os_ = 4
    k, b, f0 = np.full(n * os_, (TWO_PI * 1000.0) ** 2), np.full(n * os_, -1.0), np.zeros(n * os_)
    for i, s in enumerate(plan):
        i0 = int(s["t"] * KSR) * os_
        m = int(max(16, s["dur"] * KSR)) * os_
        nxt = int(plan[i + 1]["t"] * KSR) * os_ if i + 1 < len(plan) else i0 + m + int(0.03 * KSR) * os_
        g = max(os_, min(nxt, n * os_) - i0 - m)
        kk, bb, ff = laje_path(s["f"][0][1], s["f"][-1][1], m, g, s.get("amp", 1.0))
        e = min(i0 + m + g, len(k))
        k[i0:e], b[i0:e], f0[i0:e] = kk[:e - i0], bb[:e - i0], ff[:e - i0]
    y = _laje_syrinx(k, b, f0, float(KSR * os_))
    return _sig.resample_poly(y, 1, os_)[:n]


def _syl(dur, f, gap, rep=(1, 1), **kw):
    return dict(dur=dur, f=f, gap=gap, rep=rep, **kw)


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
        t += 0.55 / tempo
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


GARDNER_TRACT = (0.03, 0.01, 0.15, 0.4, 0.8)                          # L1 m, L2 m, A1, A2, beak reflection


@numba.njit(cache=True)
def _two_tube_kernel(pi, d1, d2, r, al):
    """Trachea + beak: a = Pi + bb(t-t1), bb = r a(t-t1) + (1-r) cb(t-t2), bf = (1+r) a(t-t1) - r cb(t-t2),
    cb = al bf(t-t2); returns bf(t-t2), the wave reaching the beak."""
    n = pi.shape[0]
    a = np.zeros(n)
    bb = np.zeros(n)
    bf = np.zeros(n)
    cb = np.zeros(n)
    out = np.zeros(n)
    for i in range(n):
        a1 = a[i - d1] if i >= d1 else 0.0
        c2 = cb[i - d2] if i >= d2 else 0.0
        bb[i] = r * a1 + (1.0 - r) * c2
        a[i] = pi[i] + (bb[i - d1] if i >= d1 else 0.0)
        bf[i] = (1.0 + r) * a1 - r * c2
        f2 = bf[i - d2] if i >= d2 else 0.0
        cb[i] = al * f2
        out[i] = f2
    return out


def _two_tube(x):
    l1, l2, a1, a2, al = GARDNER_TRACT
    os_ = 8
    fs = KSR * os_
    y = _two_tube_kernel(_sig.resample_poly(x, os_, 1), max(1, round(l1 / C_AIR * fs)), max(1, round(l2 / C_AIR * fs)), (a1 - a2) / (a1 + a2), al)
    return _sig.resample_poly(y, 1, os_)[:len(x)]


RHO_AIR = 1.2
OEC = dict(lg=20.0, ch=1.5e-8 / 350.0, rh=1000.0, rb=1e5,
           lb=RHO_AIR * GARDNER_TRACT[1] / (GARDNER_TRACT[3] * 1e-4), r=0.5)


def oec_hz(lg: float = OEC["lg"], ch: float = OEC["ch"]) -> float:
    """Helmholtz resonance of the oropharyngeal-esophageal cavity, 1 / (2 pi sqrt(Lg Ch))."""
    return 1.0 / (TWO_PI * math.sqrt(lg * ch))


@numba.njit(cache=True)
def _oec_kernel(v, fs, lg, ch, rh, lb, rb):
    """OEC + beak, three linear ODEs (RK4) driven by Vext; returns the radiated pressure ~ d i3 / dt."""
    n = v.shape[0]
    h = 1.0 / fs
    a1 = 1.0 / (lg * ch)
    a2 = rh * (1.0 / lb + 1.0 / lg)
    a3 = 1.0 / (lg * ch) - rb * rh / (lb * lg)
    i1 = 0.0
    w = 0.0
    i3 = 0.0
    out = np.zeros(n)
    vp = 0.0
    for k in range(n):
        ve = v[k]
        dv = (ve - vp) * fs
        vp = ve
        k1a, k1b, k1c = w, -a1 * i1 - a2 * w + a3 * i3 + dv / lg + rh / (lg * lb) * ve, -lg / lb * w - rb / lb * i3 + ve / lb
        xa, xb, xc = i1 + 0.5 * h * k1a, w + 0.5 * h * k1b, i3 + 0.5 * h * k1c
        k2a, k2b, k2c = xb, -a1 * xa - a2 * xb + a3 * xc + dv / lg + rh / (lg * lb) * ve, -lg / lb * xb - rb / lb * xc + ve / lb
        xa, xb, xc = i1 + 0.5 * h * k2a, w + 0.5 * h * k2b, i3 + 0.5 * h * k2c
        k3a, k3b, k3c = xb, -a1 * xa - a2 * xb + a3 * xc + dv / lg + rh / (lg * lb) * ve, -lg / lb * xb - rb / lb * xc + ve / lb
        xa, xb, xc = i1 + h * k3a, w + h * k3b, i3 + h * k3c
        k4a, k4b, k4c = xb, -a1 * xa - a2 * xb + a3 * xc + dv / lg + rh / (lg * lb) * ve, -lg / lb * xb - rb / lb * xc + ve / lb
        i1 += h / 6.0 * (k1a + 2.0 * k2a + 2.0 * k3a + k4a)
        w += h / 6.0 * (k1b + 2.0 * k2b + 2.0 * k3b + k4b)
        i3 += h / 6.0 * (k1c + 2.0 * k2c + 2.0 * k3c + k4c)
        out[k] = -lg / lb * w - rb / lb * i3 + ve / lb
    return out


def oec_beak(pi, tf1: float, **kw) -> np.ndarray:
    """Trachea input Pi -> Pt = (1 - r) Pi(t - T/2) (T = 1 / (2 tf1), quarter-wave trachea) -> OEC and beak."""
    p = dict(OEC, **kw)
    os_ = 4
    fs = KSR * os_
    x = _sig.resample_poly(np.asarray(pi, float), os_, 1)
    d = max(0, int(round(fs / (4.0 * tf1))))
    pt = (1.0 - p["r"]) * np.concatenate((np.zeros(d), x))[:len(x)]
    y = _oec_kernel(pt, float(fs), p["lg"], p["ch"], p["rh"], p["lb"], p["rb"])
    return _sig.resample_poly(y, 1, os_)[:len(pi)]


def _birdsong(species: str, seed, repeats=1, syllables=0, tempo=1.0, pitch=1.0, two_voice=0.0,
              source="fm", tract="bands", coupling=0.0, gestures=0.0) -> np.ndarray:
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
    tf1, tf2 = (min(v * pitch, 0.4 * KSR) for v in p["tf"])
    if gestures > 0:
        g = wc_pressure(n / KSR, float(gestures))
        amp = np.clip(np.interp(np.arange(n) / KSR, np.arange(len(g)) / WC["fs"], g) / 0.9, 0.0, 1.0)
        amp = np.where(amp > 0.1, amp, 0.0)
    if source == "gardner":
        x = _unit(gardner_source(2.5 * amp, f, tf1, coupling))
    elif source in SUBOSCINE:
        k2, ke = SUBOSCINE[source]
        x = _unit(gardner_source(3.0 * amp, np.full(n, float(np.min(f))), tf1, coupling, k2=k2, kexp=ke))
    elif source == "laje":
        x = _unit(_laje_source(plan, n))
        amp = np.maximum(amp, 0.0) > 0
    elif source == "normal_form":
        x = _normal_form_source(f, amp, fb, nzd, tf1, rng)
        if two_voice > 0:
            x = x + two_voice * _normal_form_source(f * 1.27, amp, fb, nzd, tf1, _rng(seed, 23))
        x = _unit(x)
    else:
        nz = nzd * V._pd_lop(rng.uniform(-1, 1, n), 2000.0, float(KSR))
        x = _fm_syrinx(f, w, fb, nz, float(KSR))
        if two_voice > 0:                  # second bronchial source at a non-harmonic ratio (great tit)
            x = x + two_voice * _fm_syrinx(f * 1.27, w, fb, nz, float(KSR))
    x = x * amp
    if tract == "gardner":
        y = _two_tube(x)
    elif tract == "oec":
        y = oec_beak(x, tf1)
    else:
        y = _bpf(x, tf1, p["q"]) + _bpf(x, tf2, p["q"])
    return _hp(y, p["hp"] * pitch) * (amp > 0)                         # beak


@sfx("birdsong", "Bird song / call from a syllable grammar driving Farnell's FM syrinx (12 species-like presets).",
     species=("robin", "canary|robin|blackbird|sparrow|owl|crow|gull|duck|rooster|cuckoo|pigeon|woodpecker"),
     repeats=(1, "songs / calls in a row"), syllables=(0, "exact syllable count (0 = the preset's own, random per seed)"),
     pitch=(1.0, "frequency scale (smaller bird > 1)"), tempo=(1.0, "speed scale"),
     two_voice=(0.0, "0..1 level of the second syrinx source"), bright=(1.0, "0.3..2 top-end"),
     source=("fm", "fm (Farnell) | normal_form (labial normal form + trachea) | laje (labia driven along one ellipse in pressure-tension per syllable) | gardner (flapping labia, tension follows the song) | suboscine / tyrant (fixed tension, pressure alone raises the pitch through a stiffening restitution)"), seed=(0, "variation"),
     coupling=(0.0, "0..1 gardner/suboscine/tyrant: trachea pressure acting back on the labia (period doubling when strong)"),
     gestures=(0.0, "Hz: > 0 replaces the syllable envelopes by Wilson-Cowan air-sac pressure gestures at this rate (canary: 5 separate pulses .. 25 sustained)"),
     tract=("bands", "bands = the preset's two band-passes | gardner = a 3 cm trachea and a 1 cm beak (a canary's), whatever the species | oec = trachea, oropharyngeal-esophageal cavity (Helmholtz, ~5 kHz) and beak"))
def birdsong(sr=DEFAULT_SR, species="robin", repeats=1, syllables=0, pitch=1.0, tempo=1.0, two_voice=0.0,
             bright=1.0, source="fm", seed=0, tract="bands", coupling=0.0, gestures=0.0):
    _choice(species, BIRDS, "species")
    _choice(source, ("fm", "normal_form", "laje", "gardner", "suboscine", "tyrant"), "source")
    _choice(tract, ("bands", "gardner", "oec"), "tract")
    y = _birdsong(species, seed, int(np.clip(repeats, 1, 50)), int(syllables), tempo,
                  float(np.clip(pitch, 0.25, 3.0)), float(np.clip(two_voice, 0.0, 1.0)), source, tract,
                  float(np.clip(coupling, 0.0, 1.0)), float(np.clip(gestures, 0.0, 40.0)))
    return _fin(_lp(y, 5500.0 * float(np.clip(bright, 0.3, 2.0))), sr, peak=0.5, fade_ms=2.0, hp=60.0)


_TUBES = {"oil_bird": (0.100, 0.0025), "raven": (0.070, 0.0035), "budgie": (0.050, 0.002)}

LATEX = (1.0e6, 930.0, 0.5)                    # Young's modulus Pa, density kg/m3, Poisson ratio
PHONATION_PA = 1000.0
BRONCHUS_M = 0.02


def latex_membrane_hz(radius_m: float, strain: float) -> float:
    """First mode of a clamped circular latex membrane under equibiaxial strain: tension per length
    T = E h strain / (1 - nu), f = 2.405 / (2 pi a) sqrt(T / (rho h)) (thickness cancels)."""
    e, rho, nu = LATEX
    return 2.405 / (TWO_PI * radius_m) * math.sqrt(e * strain / ((1.0 - nu) * rho))


@numba.njit(cache=True)
def _labial_pair(k1, k2, b1, b2, kap, fs, db, dt, yb, yt, rbeak):
    """Two labial oscillators x'' = -k x - c x^2 x' + b x' + kap p_back (per unit mass, RK4), each into
    its bronchus (delay db), joined to the trachea (delay dt) at a 3-port pressure junction
    p_J = 2 sum(Y_i p_i+) / sum(Y_i); open beak reflects -rbeak, transmits the rest."""
    n = k1.shape[0]
    c = LAJE_C
    h = 1.0 / fs
    xs = np.array([1e-4, -1e-4])
    ys = np.zeros(2)
    up = np.zeros((2, db))
    dn = np.zeros((2, db))
    tu = np.zeros(dt)
    td = np.zeros(dt)
    out = np.zeros(n)
    jb = 0
    jt = 0
    ysum = 2.0 * yb + yt
    for i in range(n):
        for v in range(2):
            kk = k1[i] if v == 0 else k2[i]
            bb = b1[i] if v == 0 else b2[i]
            if kk <= 0.0:
                continue
            f = kap[i] * dn[v, jb]
            x, y = xs[v], ys[v]
            k1x, k1y = y, -kk * x - c * x * x * y + bb * y + f
            xx, yy = x + 0.5 * h * k1x, y + 0.5 * h * k1y
            k2x, k2y = yy, -kk * xx - c * xx * xx * yy + bb * yy + f
            xx, yy = x + 0.5 * h * k2x, y + 0.5 * h * k2y
            k3x, k3y = yy, -kk * xx - c * xx * xx * yy + bb * yy + f
            xx, yy = x + h * k3x, y + h * k3y
            k4x, k4y = yy, -kk * xx - c * xx * xx * yy + bb * yy + f
            xs[v] = x + h / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
            ys[v] = y + h / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        a0 = up[0, jb]
        a1 = up[1, jb]
        at = td[jt]
        pj = 2.0 * (yb * (a0 + a1) + yt * at) / ysum
        top = tu[jt]
        tu[jt] = pj - at
        td[jt] = -rbeak * top
        out[i] = (1.0 - rbeak) * top
        up[0, jb] = ys[0] + dn[0, jb]
        up[1, jb] = ys[1] + dn[1, jb]
        dn[0, jb] = pj - a0
        dn[1, jb] = pj - a1
        jb += 1
        if jb >= db:
            jb = 0
        jt += 1
        if jt >= dt:
            jt = 0
    return out


def labial_syrinx(pressure_pa, f_pair, length_m: float = 0.07, radius_m: float = 0.0035,
                  coupling: float = 0.5, beak_r: float = 0.5) -> np.ndarray:
    """Two labial sources (one per bronchus) in physical units: k = (2 pi f)^2, b = 0.1 sqrt(k) (p / p_th - 1),
    so each phonates above p_th = 1 kPa; bronchi of radius a / 2^(1/3) (Murray), 2 cm long, join the trachea
    at a 3-port junction; ``coupling`` scales the back-pressure that lets the two lock or beat. Radiated
    pressure = d/dt of the wave leaving the beak. Arrays at 48 kHz."""
    os_ = 4
    fs = KSR * os_
    ps = np.repeat(np.asarray(pressure_pa, float), os_)
    m = len(ps)
    k1 = (TWO_PI * np.repeat(np.asarray(f_pair[0], float) * np.ones(m // os_), os_)) ** 2
    k2 = (TWO_PI * np.repeat(np.asarray(f_pair[1], float) * np.ones(m // os_), os_)) ** 2 if f_pair[1] is not None \
        else np.zeros(m)
    drive = ps / PHONATION_PA - 1.0
    b1 = 0.1 * np.sqrt(k1) * drive
    b2 = 0.1 * np.sqrt(k2) * drive
    kap = float(coupling) * 0.1 * np.sqrt(np.maximum(k1, k2))
    rb = radius_m / 2.0 ** (1.0 / 3.0)
    db = max(1, int(round(BRONCHUS_M / C_AIR * fs)))
    dt = max(1, int(round(length_m / C_AIR * fs)))
    y = _labial_pair(k1, k2, b1, b2, kap, float(fs), db, dt, rb * rb, radius_m * radius_m, float(beak_r))
    return _sig.resample_poly(np.diff(y, prepend=0.0), 1, os_)[:m // os_]


@sfx("bird_call", "Physical syrinx: pressure-driven membrane valves into a trachea waveguide (Smyth & Smith); raven-like.",
     freq=(450.0, "membrane frequency Hz (tension), 150..900; budgie up to 3000, about 2100 for its contact call"), freq_end=(0.0, "glide target Hz, 0 = none"),
     pressure=(3.0, "air-sac pressure kPa, 1.5..6"), attack=(0.3, "fraction of dur over which pressure rises"),
     two_voice=(5.0, "Hz offset of the second valve (beating); 0 = one valve"),
     species=("raven", "trachea size: oil_bird|raven|budgie"), dur=(0.5, "s"),
     source=("valve", "valve (membrane valves, Smyth & Smith) | labial (two self-oscillating labia, bronchi joined to the trachea, phonation above 1 kPa)"),
     coupling=(0.5, "labial: back-pressure coupling of the two sources (0 = independent, they beat; larger = they lock when close)"),
     latex=(0.0, "membrane strain 0..0.5: if > 0 the frequency is that of a latex membrane across the trachea (artificial syrinx), freq ignored"),
     membrane=("sweep", "valve: sweep (sounds from about 1 kPa) | fletcher (a raven's thin membrane, sounds from 0.3 kPa; pressure may go down to 0.3) | raven (full raven set: valve shut at rest, constant damping, heavy mass law, 1 ml bronchus)"),
     radiation=("lowpass", "valve: lowpass (beak reflection corner 4 kHz) | piston (corner where k b = 2) | fletcher (piston, then a roll-off from 3 kHz)"))
def bird_call(sr=DEFAULT_SR, freq=450.0, freq_end=0.0, pressure=3.0, attack=0.3, two_voice=5.0, species="raven",
              dur=0.5, source="valve", coupling=0.5, latex=0.0, membrane="sweep", radiation="lowpass"):
    length, radius = _TUBES[_choice(species, _TUBES, "species")]
    _choice(source, ("valve", "labial"), "source")
    n = _n(dur)
    budgie = species == "budgie"
    f0 = latex_membrane_hz(radius, float(np.clip(latex, 0.0, 0.5))) if latex > 0 else to_hz(freq)
    f = np.clip(np.linspace(f0, to_hz(freq_end) if freq_end else f0, n), 150.0, 3000.0 if budgie else 900.0)
    env = V.articulation_env(n, float(np.clip(attack, 0.05, 0.9)), offset=0.0) ** 0.5
    _choice(membrane, ("sweep", "fletcher", "raven"), "membrane")
    _choice(radiation, ("lowpass", "piston", "fletcher"), "radiation")
    ps = 1000.0 * float(np.clip(pressure, 1.5 if membrane == "sweep" else 0.3, 6.0)) * (0.3 + 0.7 * env)
    if source == "labial":
        y = labial_syrinx(ps, (f, f + float(two_voice) if two_voice else None), length, radius,
                          float(np.clip(coupling, 0.0, 2.0)))
    else:
        y = V.syrinx(ps, (f, f + float(two_voice)), trachea_length_m=length, trachea_radius_m=radius,
                     second_valve=bool(two_voice), membrane=membrane, radiation=radiation)
    return _fin(_lp(y * env, 12000.0 if budgie else 5000.0), sr, peak=0.6)


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
    env = np.minimum(t / 0.08, 1.0) * np.minimum((dur - t) / 0.25, 1.0) ** 2
    return _fin(x * np.clip(env, 0, 1), sr, peak=0.4, hp=200.0)


_HISS = {"snake": ("shh", 1.0, 0.25, 1.4), "cat": ("xxx", 1.25, 0.08, 0.7)}


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
    return _unit(y) + 0.12 * effort * _lp(turb, 6000.0)


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
        f0 = 170.0 / size * np.linspace(1.0, 0.7, n)
        v = V.formant_filter(V.lf_pulse(f0, 0.25, rng=rng), "hah", "stk", formant_scale=1.0 / size)
        y = y + 0.5 * _unit(v * V.articulation_env(n, 0.12, offset=0.0) ** 2)
    else:                                                               # pant
        d = dur or 2.0
        rate = 2.0 + 1.8 * e
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
    f = to_hz(pitch) / size * (0.8 + 0.2 * np.linspace(1.0, 0.0, n) ** 1.5)
    src = V.lf_pulse(f, 0.25 + 0.5 * float(np.clip(effort, 0, 1)), rng=rng, voiced_env=np.where(env > 0.02, 1.0, 0.0))
    y = V.formant_filter(src * env, "@", "farnell", formant_scale=1.0 / size)
    return _fin(_hp(y, 60.0), sr, peak=0.5)


def _nonverbal(kind: str, what: str):
    def fn(sr=DEFAULT_SR, voice="female", syllables=0, pitch=1.0, seed=0):
        from .physical import nonverbal as NV
        y = NV.vocalize(kind, str(voice), _rng(seed, 60 + NV.KINDS.index(kind)),
                        n_syl=int(syllables) or None, pitch=float(np.clip(pitch, 0.25, 4.0)))
        return _fin(y, sr, peak=0.6, hp=60.0)
    sfx(kind, f"{what} (Anikin's soundgen model with its presets: syllables, jitter, AM, aspiration, formants).",
        voice=("female", "female | male"), syllables=(0, "syllables in the bout, 0 = as the preset"),
        pitch=(1.0, "f0 scale"), seed=(0, "variation"))(fn)


for _k, _w in (("laugh", "Laugh: a bout of short voiced syllables with breathy onsets"),
               ("scream", "Scream: high, rough voice with jitter, shimmer and amplitude modulation"),
               ("cry", "Crying: long falling wail with slow jitter and amplitude modulation"),
               ("moan", "Moan: falling voiced syllable with aspiration")):
    _nonverbal(_k, _w)


@sfx("cough", "Cough: explosive noise burst through the open tract with a short voiced tail.",
     count=(2, "coughs"), effort=(0.7, "0..1"), size=(1.0, "body scale"), seed=(0, "variation"))
def cough(sr=DEFAULT_SR, count=2, effort=0.7, size=1.0, seed=0):
    rng = _rng(seed, 42)
    size = float(np.clip(size, 0.4, 4.0))
    count = int(np.clip(count, 1, 20))
    out = np.zeros(_n(0.45 * count + 0.3))
    t = 0.01
    for k in range(count):
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
    for k in range(2):
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
    while t < dur - 0.2:
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
    for t0, fa, fb, d, g in ((0.01, 420.0, 170.0, 0.07, 0.7), (0.14, 190.0, 620.0, 0.09, 1.0)):
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
    if kind == "slurp":
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
    """S1-S2 interval in s: left-ventricular ejection time 0.413 - 0.0017 HR plus 0.04 s of
    isovolumic contraction."""
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


CLAP_EXP_RISE = 0.0032
CLAP_EXP_Q = 5.5


def _clap(rng, fc: float, q: float, decay: float, n: int | None = None, rise: str = "fast") -> np.ndarray:
    m = _n(max(8 * decay, 0.03)) if n is None else n
    t = np.arange(m) / KSR
    up = 1.0 - np.exp(-t / CLAP_EXP_RISE) if rise == "exp" else np.minimum(t / 0.0004, 1.0)
    return _bpf(rng.uniform(-1, 1, m) * up * np.exp(-t / decay), fc, q)


# hand configuration -> resonator (f, df, B, dB in Hz, gain G, dG)
CLAP_TYPES = {"P1": (1364, 333, 188, 54, 27.2, 9.76), "P2": (774, 448, 139, 7, 13.8, 3.5),
              "P3": (1562, 257, 375, 94, 19.5, 6.31), "A1": (775, 39, 137, 15, 22.2, 3.55),
              "A2": (1355, 57, 222, 53, 24.0, 8.03), "A3": (1452, 12, 183, 40, 13.8, 4.45),
              "A1+": (500, 178, 91, 27, 33.8, 5.57), "A1-": (984, 396, 137, 36, 11.3, 1.42)}


def clap_resonator(kind: str, rng) -> tuple[float, float, float]:
    """(fc, Q, gain) of one clap of hand configuration `kind`, each drawn around its mean with its deviation."""
    f, df, b, db, g, dg = CLAP_TYPES[kind]
    fc = float(np.clip(rng.normal(f, df), 300.0, 2500.0))
    bw = float(np.clip(rng.normal(b, db), 0.25 * b, 3.0 * b))
    return fc, fc / bw, float(max(rng.normal(g, dg), 0.2 * g))


def _clap_burst(rng, kind: str, n: int) -> tuple[np.ndarray, float]:
    fc, q, g = clap_resonator(kind, rng)
    t = np.arange(n) / KSR
    exc = rng.uniform(-1, 1, n) * np.minimum(t / 0.0004, 1.0) * np.exp(-t / 0.005)
    return _bpf(exc, fc, q), g


@sfx("handclap", "One hand clap: a filtered noise burst; cupped hands are lower and hollower.",
     cup=(0.3, "0 flat palms (bright ~1.4 kHz) .. 1 cupped (hollow ~0.5 kHz)"),
     clap_type=("", "hand configurations P1 P2 P3 (palm to palm), A1 A2 A3 (palm to fingers), A1+ (cupped), "
                    "A1- (flat): one resonator drawn around the measured f, B, G; empty = use `cup`"),
     rise=("fast", "fast (0.4 ms linear onset) | exp (exponential onset, 3.2 ms time constant, resonators Q 5.5)"),
     seed=(0, "variation"))
def handclap(sr=DEFAULT_SR, cup=0.3, clap_type="", rise="fast", seed=0):
    rng = _rng(seed, 48)
    if clap_type:
        y, _ = _clap_burst(rng, _choice(clap_type, CLAP_TYPES, "clap_type"), _n(0.12))
        return _fin(y, sr, peak=0.7, fade_ms=1.0, hp=200.0)
    _choice(rise, ("fast", "exp"), "rise")
    c = float(np.clip(cup, 0.0, 1.0))
    fc = 1400.0 * (500.0 / 1400.0) ** c
    q1, q2 = (CLAP_EXP_Q, CLAP_EXP_Q) if rise == "exp" else (1.5 + 2.5 * c, 2.0)
    y = _clap(rng, fc, q1, 0.008 + 0.008 * c, _n(0.12), rise)
    y[_n(0.003):] += 0.5 * _clap(rng, fc * 1.5, q2, 0.006, len(y) - _n(0.003), rise)
    return _fin(y, sr, peak=0.7, fade_ms=1.0, hp=200.0)


CLAP_VOICES = 64        # clappers rendered one by one; the rest of the crowd is a Poisson grain bed


def applause_rate(enthusiasm: float) -> float:
    """Claps per second per person: 2.5 Hz bored .. 4.17 Hz enthusiastic."""
    return 2.5 + 1.67 * float(np.clip(enthusiasm, 0.0, 1.0))


def _kuramoto_plan(n: int, r: float, sd: float, k: float, dur: float, rng) -> list[np.ndarray]:
    """Phase oscillators dphi_k/dt = w_k + (K/N) sum sin(phi_j - phi_k), Euler at 1 ms; a clap at every 2 pi crossing."""
    dt = 0.001
    w = TWO_PI * r * np.clip(rng.normal(1.0, sd, n), 0.7, 1.3)
    ph = rng.uniform(0.0, TWO_PI, n)
    times = [[] for _ in range(n)]
    for step in range(int(dur / dt)):
        z = np.mean(np.exp(1j * ph))
        nxt = ph + dt * (w + k * np.abs(z) * np.sin(np.angle(z) - ph))
        for i in np.nonzero(np.floor(nxt / TWO_PI) > np.floor(ph / TWO_PI))[0]:
            times[i].append((step + 1) * dt)
        ph = nxt
    return [np.array(t) for t in times]


def clap_order(plan: list[np.ndarray], dur: float, r: float) -> float:
    """Kuramoto order parameter q = |mean exp(i phi)| over the second half, phases interpolated between claps."""
    t = np.arange(0.5 * dur, dur - 1.0 / r, 0.005)
    zs = []
    for p in plan:
        if len(p) < 2:
            continue
        k = np.clip(np.searchsorted(p, t) - 1, 0, len(p) - 2)
        zs.append(np.exp(1j * TWO_PI * (t - p[k]) / (p[k + 1] - p[k])))
    return float(np.mean(np.abs(np.mean(zs, axis=0))))


def applause_plan(crowd: int = 30, enthusiasm: float = 0.6, dur: float = 4.0, seed: int = 0,
                  sync: float = 0.0) -> list[np.ndarray]:
    """Clap times (s) of each individually rendered clapper: own rate (sd 10 %), phase and 2.7 % jitter.
    `sync` > 0 is Mode II: period doubled, relative rate spread / 1.3, coupling K = sync * 0.8/(2 pi) * w."""
    rng = _rng(seed, 49)
    r = applause_rate(enthusiasm)
    if sync > 0.0:
        r2 = 0.5 * r
        plan = _kuramoto_plan(int(min(max(crowd, 1), CLAP_VOICES)), r2, 0.10 / 1.3,
                              float(np.clip(sync, 0.0, 1.0)) * 0.8 * r2, dur, rng)
        return [np.sort(np.clip(p + rng.normal(0.0, 0.027 / r2, len(p)), 0.0, dur - 0.03)) for p in plan]
    plan = []
    for _ in range(int(min(max(crowd, 1), CLAP_VOICES))):
        ri = r * float(np.clip(rng.normal(1.0, 0.10), 0.7, 1.3))
        k = np.arange(int(dur * ri) + 2)
        t = rng.uniform(0.0, 1.0 / ri) + k / ri + rng.normal(0.0, 0.027 / ri, len(k))
        plan.append(np.sort(t[(t >= 0.0) & (t < dur - 0.03)]))
    return plan


@sfx("applause", "Applause: every clapper has a rate, a hand resonance and a place; big crowds blur into Poisson grains.",
     crowd=(30, "number of people"), enthusiasm=(0.6, "0 bored (2.5 claps/s each) .. 1 wild (4.2 claps/s)"),
     dur=(4.0, "s"),
     sync=(0.0, "0 free clapping (Mode I) .. 1 rhythmic applause: clappers halve their rate, tighten it and pull "
                "into phase (Mode II, Kuramoto coupling)"),
     clap_type=("", "empty = hand resonances spread log-normally | mix = each clapper one of the 8 hand "
                    "configurations | one type (P1..A1-) for all"),
     seed=(0, "variation"))
def applause(sr=DEFAULT_SR, crowd=30, enthusiasm=0.6, dur=4.0, sync=0.0, clap_type="", seed=0):
    crowd = int(np.clip(crowd, 1, 100000))
    rng = _rng(seed, 50)
    n = _n(dur)
    out = np.zeros(n)
    plan = applause_plan(crowd, enthusiasm, dur, seed, sync)
    kinds = list(CLAP_TYPES)
    if clap_type and clap_type != "mix":
        _choice(clap_type, CLAP_TYPES, "clap_type")
    for times in plan:
        if clap_type:
            kind = kinds[int(rng.integers(len(kinds)))] if clap_type == "mix" else clap_type
            exc = np.zeros(n)
            for t in times:
                y, g = _clap_burst(rng, kind, _n(0.06))
                _place(exc, y * g / 25.0, t)
            out += exc
            continue
        fc = float(np.clip(1100.0 * math.exp(rng.normal(0.0, 0.38)), 500.0, 1600.0))
        decay = rng.uniform(0.006, 0.014)
        m = _n(8 * decay)
        env = np.exp(-np.arange(m) / (decay * KSR))
        exc = np.zeros(n)
        for t in times:
            _place(exc, rng.uniform(-1, 1, m) * env, t, rng.uniform(0.7, 1.0))
        out += _bpf(exc, fc, rng.uniform(1.5, 4.0)) * rng.uniform(0.35, 1.0)
    extra = crowd - CLAP_VOICES
    if extra > 0 and sync > 0.0:      # the rest of the crowd claps with the rendered clappers it is locked to
        hits = np.zeros(n)
        per = extra / max(len(plan), 1)
        for times in plan:
            for t in times:
                k = rng.poisson(per)
                if k:
                    idx = (np.clip(t + rng.normal(0.0, 0.01, k), 0.0, dur - 1e-3) * KSR).astype(int)
                    np.add.at(hits, idx, rng.uniform(0.3, 1.0, k))
        env = _sig.lfilter([1.0], [1.0, -math.exp(-1.0 / (0.01 * KSR))], hits)
        bed = rng.uniform(-1, 1, n) * env
        out += 0.25 * sum(_bpf(bed, fc, 1.5) for fc in (1000.0, 1600.0, 2500.0))
    elif extra > 0:
        hits = rng.poisson(extra * applause_rate(enthusiasm) / KSR, n) * rng.uniform(0.3, 1.0, n)
        env = _sig.lfilter([1.0], [1.0, -math.exp(-1.0 / (0.01 * KSR))], hits)
        bed = rng.uniform(-1, 1, n) * env
        out += 0.25 * sum(_bpf(bed, fc, 1.5) for fc in (1000.0, 1600.0, 2500.0))
    return _fin(_hp(out, 300.0), sr, peak=0.6, hp=100.0)


def _gates(text: str, dur: float) -> list[tuple[float, float]]:
    out = []
    for part in str(text).split(","):
        a, b = (float(v) for v in part.split("-"))
        if not 0.0 <= a < b <= dur:
            raise ValueError(f"gate {part!r} must lie inside 0..{dur} s")
        out.append((a, b))
    return out


@sfx("dread_choir", "Terror: a choir of wrong voices folded into one pressure, silence between (three Pink Trombone tracts + a bass FOF cluster).",
     size=(2.5, "body scale: f0 = 110/size^0.6 Hz (floor 30), tract 0.2 m x size"), dur=(1.9, "s"),
     gates=("0-0.75,1.1-1.85", "voiced phrases a-b in seconds, comma separated"), seed=(0, "variation"))
def dread_choir(sr=DEFAULT_SR, size=2.5, dur=1.9, gates="0-0.75,1.1-1.85", seed=0):
    rng = _rng(seed, 70)
    k = float(np.clip(size, 0.5, 8.0))
    n = _n(dur)
    gate = np.zeros(n)
    for a, b in _gates(gates, dur):
        m = _n(b - a)
        gate[_n(a):_n(a) + m] = V.articulation_env(m, 0.35, 0.0)[: n - _n(a)]
    u = rng.uniform(0.0, 1.0, 5)
    base = 1.2 * max(30.0, 110.0 / k ** 0.6 * (0.9 + 0.2 * u[0]))
    pt = float(np.clip(0.17 / (0.2 * k), 0.25, 1.6))
    t = np.arange(n) / KSR
    parts = []
    for j, sc in enumerate((1.0, 0.66, 0.5)):
        f = base * (1.0 + 0.06 * j - 0.03 * u[1 + j]) * (1 + 0.02 * np.sin(TWO_PI * 0.7 * t + j))
        y = V.pink_trombone(f, 0.35 + 0.15 * j, 17.0 + 3.0 * j, 2.6 + 0.2 * j, (),
                            tick_rate_scale=float(np.clip(pt * sc, 0.25, 1.6)), dur=dur, rng=rng,
                            voiced_env=(gate > 0.02).astype(float), wobble=1.0)
        parts.append(_unit(y[:n]))
    ch = V.choir([base * 2 ** (c / 12) for c in (0.0, 0.8, 5.3)], 2.5, 2, dur, rng, detune_cents=35.0,
                 vibrato_gain=0.02, attack=0.2, release=0.3, voice_type="bass", breath=0.03)
    parts.append(_unit(np.pad(ch, (0, max(0, n - len(ch))))[:n]))
    y = sum(parts) / math.sqrt(len(parts))
    doom = 0.5 + 0.5 * np.sin(TWO_PI * (4.0 + 4.0 * u[4]) * t)
    hi = _hp(y, 800.0)
    y = (y - hi) + hi * (0.35 + 0.65 * doom)
    return _fin(np.tanh(2.0 * _unit(y)) * gate, sr)


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
    for _ in range(k):
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
    2nd harmonic and jet noise in a band around the pitch."""
    n = len(f)
    ph = TWO_PI * np.cumsum(f) / KSR
    y = np.sin(ph) + 0.04 * np.sin(2.0 * ph)
    nz = V._bp(rng.uniform(-1, 1, n), np.minimum(f, 0.4 * KSR), 8.0)
    nz = nz / (np.sqrt(np.mean(nz * nz)) + 1e-12)
    return (y * (1.0 + 0.04 * _slow(n, rng, 6.0)) + 0.12 * breath * nz) * amp


_WHISTLES = {
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
GAITS = {"walk": 105.0, "run": 170.0, "sneak": 66.0, "limp": 84.0, "stairs": 96.0, "trot": 358.0, "gallop": 403.0}
_STRIDE = {"trot": (3.35, -0.130), "gallop": (4.44, -0.156)}     # preferred-speed stride frequency a M^b, strides/s


def stride_pace(gait: str, mass: float) -> float:
    """Footfalls per minute of a quadruped of `mass` kg at its preferred trot / gallop: 4 x 60 x a M^b."""
    a, b = _STRIDE[gait]
    return 240.0 * a * float(np.clip(mass, 0.01, 6000.0)) ** b
_QUAD = {"trot": (0.0, 0.035, 0.5, 0.535), "gallop": (0.0, 0.17, 0.36, 0.52)}
SHOES = ("default", "barefoot", "boot", "heel", "hoof")
WALKERS = {"woman": (120.0, 0.8), "man": (75.0, 1.25)}                 # footfalls / min, weight (resonance) scale
MOODS = {"neutral": (1.0, 0.0, 1.0, 1.0), "happy": (1.15, 3.0, 0.7, 0.95), "sad": (0.75, -6.0, 1.5, 1.0),
         "angry": (1.1, 5.0, 1.6, 1.2), "tender": (0.85, -5.0, 0.6, 0.9)}   # pace x, dB, irregularity x, weight x


def footsteps_plan(gait: str = "walk", pace: float = 0.0, steps: int = 8, variation: float = 0.5, seed: int = 0,
                   stop: int = 0) -> np.ndarray:
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
    gaps *= 1.0 + np.clip(0.03 * v * _fractal(steps, 0.76, rng), -0.08, 0.08) * (0.5 if gait == "stairs" else 1.0)
    if stop and steps > 1:
        from .compose import ritard_map
        f = ritard_map({"beats": min(int(stop), steps - 1), "q": 3.0, "v_end": 0.27}, steps - 1.0)
        if f:
            gaps[:-1] *= np.diff([f(i) for i in range(steps)])
    return 0.01 + np.concatenate(([0.0], np.cumsum(gaps[:-1])))


def _fractal(n: int, alpha: float, rng) -> np.ndarray:
    """Unit-variance gaussian noise whose detrended fluctuation exponent is `alpha` (power ~ f^-(2 alpha - 1))."""
    m = 2 * max(n, 2)
    f = np.fft.rfftfreq(m)
    amp = np.zeros_like(f)
    amp[1:] = f[1:] ** (-(2.0 * alpha - 1.0) / 2.0)
    x = np.fft.irfft(amp * (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)), m)[:n]
    return (x - x.mean()) / (x.std() + 1e-12)


def _step(ground: str, i: int, gait: str, ioi: float, weight: float, shoe: str, v: float, seed,
          temperature: float = -8.0) -> np.ndarray:
    """One freshly synthesised step (never a copy: own random stream, own GRF deviations)."""
    rng = _rng(seed, 1000 + i)
    y = S.render_step(ground, i % 2, rng, temperature=temperature)     # 0 = right foot, 1 = left foot
    if abs(weight - 1.0) > 1e-3:
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
     pace=(0.0, "footfalls per minute, 0 = gait default (walk 105, run 170, sneak 66, limp 84, stairs 96, trot 358, gallop 403)"),
     mass=(0.0, "kg, trot / gallop with pace 0: the preferred stride rate of an animal that heavy (0 = 500 kg horse)"),
     steps=(0, "number of footfalls, 0 = 8 (16 for trot / gallop)"), weight=(1.0, "body weight scale 0.4..3: heavier = lower, longer"),
     shoe=("default", "default|barefoot|boot|heel|hoof (trot / gallop use hoof unless set)"),
     variation=(0.5, "0..1 timing and level irregularity (steps never repeat even at 0)"),
     stop=(0, "the last N footfalls slow down to a halt, like a runner stopping (0 = steady)"),
     temperature=(-8.0, "air temperature in C, snow only: squeak spectrum after Arabadzhi 1963 (-1.5 dull ~300 Hz, -15 bright up to 8 kHz), no squeak at 0"),
     walker=("", "'' | woman (120 steps/min, lighter, higher resonances) | man (75 steps/min, heavier, lower); sets pace and weight left at default"),
     mood=("neutral", "neutral|happy (faster, louder, regular)|sad (slow, soft, uneven)|angry (fast, loud, heavy, uneven)|tender (slow, soft, even)"),
     seed=(0, "variation"))
def footsteps(sr=DEFAULT_SR, ground="wood", gait="walk", pace=0.0, steps=0, weight=1.0, shoe="default",
              variation=0.5, temperature=-8.0, mass=0.0, stop=0, seed=0, walker="", mood="neutral"):
    _choice(ground, GROUNDS, "ground")
    _choice(shoe, SHOES, "shoe")
    _choice(mood, MOODS, "mood")
    if walker:
        _choice(walker, WALKERS, "walker")
        wp, ww = WALKERS[walker]
        pace = pace or wp
        weight = ww if weight == 1.0 else weight
    mp, mdb, mv, mw = MOODS[mood]
    if mood != "neutral" and gait not in _QUAD:
        pace = (pace or GAITS[gait]) * mp
        variation = float(np.clip(variation * mv, 0.0, 1.0))
        weight = weight * mw
    if mass and not pace and gait in _STRIDE:
        pace = stride_pace(gait, mass)
    times = footsteps_plan(gait, pace, steps, variation, seed=seed, stop=stop)
    if gait in _QUAD and shoe == "default":
        shoe = "hoof"
    weight = float(np.clip(weight, 0.4, 3.0))
    ioi = 60.0 / float(np.clip(pace if pace else GAITS[gait], 20.0, 900.0))
    out = np.zeros(_n(times[-1] + 1.2))
    end = 0
    for i, t in enumerate(times):
        y = _step(ground, i, gait, ioi, weight, shoe, float(np.clip(variation, 0, 1)), seed,
                  float(np.clip(temperature, -40.0, 10.0)))
        _place(out, y, float(t))
        end = max(end, int(t * KSR) + len(y))
    peak = 0.7 * 10.0 ** (min(mdb, 3.0) / 20.0) if gait not in _QUAD else 0.7
    return _fin(out[:min(len(out), end + int(0.02 * KSR))], sr, peak=peak, fade_ms=2.0, hp=25.0)


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
    onset = np.clip((t - 0.12) / 0.25, 0.0, 1.0)
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


# ============================================================================================
# mammal vocalisation: glottal wavetable, modulation ladder, Kelly-Lochbaum tract
# ============================================================================================

def glottal_pulse(u, e1: float = 0.4, e2: float = 0.56) -> np.ndarray:
    """Glottal flow over one period, u in [0, 1): opens to u = e1, closes by u = e2, then shut."""
    u = np.asarray(u, float)
    rise = (e2 - e1) / math.pi * (1.0 - np.cos(math.pi * u / e1))
    fall = 2.0 * (e1 - e2) / math.pi * np.cos(math.pi / (2.0 * (e2 - e1)) * (u - 3.0 * e1 + 2.0 * e2))
    return np.where(u < e1, rise, np.where(u <= e2, fall, 0.0))


# complexity ladder: (shimmer depth, [(modulator / f0, phase depth in cycles)])
MAMMAL_LEVELS = {1: (0.0, []), 2: (0.15, []), 3: (0.15, [(0.5, 0.08)]), 4: (0.15, [(0.73, 0.12)]),
                 5: (0.15, [(0.73, 0.12), (0.5, 0.08)]), 6: (0.15, [(0.73, 0.12), (0.5, 0.08), (1.618, 0.2)])}


@numba.njit(cache=True)
def _kl_tract(x, k, d, r_glot, a_lip):
    """Kelly-Lochbaum tube: sections of d samples each way, junction reflections k (len n-1), glottis
    reflection r_glot, lips reflect the low frequencies (one-pole a_lip) and pass the highs."""
    ns = k.shape[0] + 1
    fw = np.zeros((ns, d))
    bw = np.zeros((ns, d))
    out = np.zeros(x.shape[0])
    lp = 0.0
    p = 0
    for i in range(x.shape[0]):
        fo = np.empty(ns)
        bo = np.empty(ns)
        for s in range(ns):
            fo[s] = fw[s, p]
            bo[s] = bw[s, p]
        fin = x[i] + r_glot * bo[0]
        lp = a_lip * lp + (1.0 - a_lip) * fo[ns - 1]
        out[i] = fo[ns - 1] - lp
        bin_last = -lp
        newf = np.empty(ns)
        newb = np.empty(ns)
        newf[0] = fin
        newb[ns - 1] = bin_last
        for j in range(ns - 1):
            kj = k[j]
            newf[j + 1] = (1.0 + kj) * fo[j] - kj * bo[j + 1]
            newb[j] = kj * fo[j] + (1.0 - kj) * bo[j + 1]
        for s in range(ns):
            fw[s, p] = newf[s]
            bw[s, p] = newb[s]
        p = (p + 1) % d
    return out


def mammal_call(f0, dur: float, complexity: int = 3, tract_m: float = 0.38, radii=None, formants: float = 0.8,
                noise: float = 0.05, seed: int = 0) -> np.ndarray:
    """Glottal wavetable read with a modulated phase (levels 1 limit cycle, 2 shimmer, 3 subharmonic,
    4 inharmonic sidebands, 5 two modulators, 6 three: chaos), noise at opening and closing, into a
    15-section lossless tube; `formants` crossfades the dry source with the tract output."""
    rng = _rng(seed, 77)
    n = _n(dur)
    t = np.arange(n) / KSR
    f = np.full(n, float(f0)) if np.isscalar(f0) else np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(f0)), f0)
    shim, mods = MAMMAL_LEVELS[int(np.clip(complexity, 1, 6))]
    ph = np.cumsum(f) / KSR
    for r, a in mods:
        ph = ph + a * np.cos(TWO_PI * np.cumsum(r * f) / KSR)
    u = ph - np.floor(ph)
    g = glottal_pulse(u)
    g = g - g.mean()
    g *= 1.0 + shim * np.sin(TWO_PI * 3.0 * t + rng.uniform(0, 6.3))
    edge = np.exp(-((u - 0.4) / 0.04) ** 2) + np.exp(-((u - 0.56) / 0.04) ** 2)
    g = g + noise * edge * rng.uniform(-1, 1, n)
    rad = np.asarray(radii if radii is not None else np.linspace(1.0, 2.0, 15), float)
    A = rad ** 2
    k = (A[:-1] - A[1:]) / (A[:-1] + A[1:])
    d = max(1, int(round(tract_m / len(A) / 350.0 * KSR)))
    y = _kl_tract(g, k, d, 0.75, 0.6)
    w = float(np.clip(formants, 0.0, 1.0))
    y = w * _unit(y) + (1.0 - w) * _unit(g)
    env = np.minimum(1.0, np.minimum(t / 0.03, (dur - t) / 0.08).clip(0.0, None))
    return y * env


@sfx("mammal", "Mammal call: glottal pulses with a ladder of nonlinear complexity (shimmer, subharmonics, sidebands, chaos) through a tube tract.",
     f0=(120.0, "Hz of the cords"), f0_end=(0.0, "glide target Hz, 0 = none"), dur=(1.5, "s"),
     complexity=(3, "1 limit cycle | 2 shimmer | 3 subharmonic | 4 inharmonic sidebands | 5 two modulators | 6 chaos"),
     tract=(0.38, "vocal tract length m (lion 0.38, wolf about 0.2, dog 0.1-0.2)"), formants=(0.8, "0 dry source .. 1 all tract"),
     seed=(0, "variation"))
def mammal(sr=DEFAULT_SR, f0=120.0, f0_end=0.0, dur=1.5, complexity=3, tract=0.38, formants=0.8, seed=0):
    f = [float(f0), float(f0_end) if f0_end else float(f0)]
    y = mammal_call(np.array(f), float(np.clip(dur, 0.1, 10.0)), int(complexity), float(np.clip(tract, 0.03, 0.6)),
                    formants=formants, seed=seed)
    return _fin(y, sr, peak=0.7, fade_ms=5.0, hp=30.0)
