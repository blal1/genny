"""Envelopes: ADSR, exponential decay, and generic breakpoint envelopes."""
from __future__ import annotations

import numpy as np

from .core import DEFAULT_SR, samples


def adsr(dur: float, attack: float = 0.01, decay: float = 0.1, sustain: float = 0.7, release: float = 0.2,
         sr: int = DEFAULT_SR, curve: float = 2.0) -> np.ndarray:
    """ADSR where `dur` is the note-on length; the release is appended after it.

    Returned length = samples(dur + release). curve > 1 gives exponential-ish (natural) shapes.
    """
    n_on = samples(dur, sr)
    n_rel = samples(release, sr) if release > 0 else 0
    a = min(samples(attack, sr) if attack > 0 else 1, n_on)
    d = min(samples(decay, sr) if decay > 0 else 1, max(1, n_on - a))
    s = max(0, n_on - a - d)
    # concave rise 1 - (1 - x)^curve: same ease-in as x^(1/curve) but with a finite slope at the start, so
    # the first samples of a note do not step (x^(1/2) put 6.7 % of full level on sample 1 of a 5 ms attack)
    env_a = 1.0 - (1.0 - np.linspace(0, 1, a)) ** curve if a > 1 else np.ones(a)
    env_d = sustain + (1 - sustain) * (1 - np.linspace(0, 1, d)) ** curve if d > 1 else np.full(d, sustain)
    env_s = np.full(s, sustain)
    on = np.concatenate([env_a, env_d, env_s])[:n_on]
    if n_rel > 0:
        last = on[-1] if on.size else sustain
        rel = last * (1 - np.linspace(0, 1, n_rel)) ** curve
        return np.concatenate([on, rel])
    return on


def power_adsr(dur: float, attack: float = 0.01, decay: float = 0.1, sustain: float = 0.7, release: float = 0.2,
               sr: int = DEFAULT_SR, curves: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """ADSR with its own power curve per stage: each stage goes from start to target as start + (target - start)
    * x^c, x the linear 0..1 ramp; c < 1 is sharp, c > 1 soft, 1 linear. ``curves`` = (attack, decay, release).
    Same lengths as ``adsr``: samples(dur + release), the release starting at the note-on end."""
    ca, cd, cr = (max(float(c), 1e-6) for c in curves)
    n_on = samples(dur, sr)
    n_rel = samples(release, sr) if release > 0 else 0
    a = min(samples(attack, sr) if attack > 0 else 0, n_on)
    d = min(samples(decay, sr) if decay > 0 else 0, n_on - a)
    ramp = lambda m: np.arange(1, m + 1) / m
    on = np.concatenate([ramp(a) ** ca, 1.0 + (sustain - 1.0) * ramp(d) ** cd, np.full(n_on - a - d, float(sustain))])
    if n_rel == 0:
        return on
    last = on[-1] if on.size else 0.0
    return np.concatenate([on, last * (1.0 - ramp(n_rel) ** cr)])


WINDOWS = ("hann", "blackman", "parzen", "tukey", "welch", "poisson")


def _half_window(window: str, x: np.ndarray, a: float | None) -> np.ndarray:
    """Rising half of a window: x 0 at the edge, 1 at the centre (value 1)."""
    if window == "hann":
        return 0.5 - 0.5 * np.cos(np.pi * x)
    if window == "blackman":
        a = 0.16 if a is None else a
        return (1 - a) / 2 - 0.5 * np.cos(np.pi * x) + a / 2 * np.cos(2 * np.pi * x)
    if window == "parzen":
        u = 1.0 - x
        return np.where(u <= 0.5, 1 - 6 * u ** 2 + 6 * u ** 3, 2 * (1 - u) ** 3)
    if window == "tukey":
        a = 0.5 if a is None else a
        return np.where(x < a, 0.5 - 0.5 * np.cos(np.pi * np.minimum(x / max(a, 1e-9), 1.0)), 1.0)
    if window == "welch":
        return 1.0 - (1.0 - x) ** 2
    if window == "poisson":
        a = np.log(1000.0) if a is None else a
        return np.exp(-a * (1.0 - x))
    raise ValueError(f"unknown window {window!r}; use one of {WINDOWS}")


def win_env(dur: float, attack: float = 0.05, release: float = 0.05, sr: int = DEFAULT_SR, window: str = "hann",
            a: float | None = None) -> np.ndarray:
    """Attack / release envelope shaped by a window function: the attack is the window's rising half, held at 1
    until ``dur``, then the falling half over ``release``. ``a``: the window's coefficient (blackman alpha 0.16,
    tukey taper 0.5, poisson decay ln 1000 = -60 dB at the edge). Length samples(dur + release)."""
    n_on = samples(dur, sr)
    n_rel = samples(release, sr) if release > 0 else 0
    na = min(samples(attack, sr) if attack > 0 else 0, n_on)
    rise = _half_window(window, np.arange(na) / max(na, 1), a)
    fall = _half_window(window, 1.0 - np.arange(1, n_rel + 1) / max(n_rel, 1), a)
    return np.concatenate([rise, np.ones(n_on - na), fall])


def decay(dur: float, tau: float, sr: int = DEFAULT_SR, floor: float = 0.0) -> np.ndarray:
    """Exponential decay exp(-t/tau) over dur seconds."""
    t = np.arange(samples(dur, sr)) / sr
    e = np.exp(-t / max(tau, 1e-4))
    return np.maximum(e, floor)


def perc(dur: float, attack: float = 0.002, sr: int = DEFAULT_SR, curve: float = 3.0) -> np.ndarray:
    """Percussive: fast attack then curved decay to 0 over dur."""
    n = samples(dur, sr)
    a = min(max(1, samples(attack, sr)), n)
    env = np.ones(n)
    env[:a] = np.linspace(0, 1, a)
    rest = n - a
    if rest > 0:
        env[a:] = (1 - np.linspace(0, 1, rest)) ** curve
    return env


def breakpoints(points: list[tuple[float, float]], dur: float, sr: int = DEFAULT_SR) -> np.ndarray:
    """points: [(time_seconds, value), ...] linearly interpolated over dur."""
    n = samples(dur, sr)
    t = np.arange(n) / sr
    ts = [p[0] for p in points]
    vs = [p[1] for p in points]
    return np.interp(t, ts, vs)


def apply(sig: np.ndarray, env: np.ndarray) -> np.ndarray:
    """Multiply, padding whichever is shorter with zeros / last value."""
    n = max(sig.shape[0], env.shape[0])
    if sig.shape[0] < n:
        sig = np.concatenate([sig, np.zeros(n - sig.shape[0])])
    if env.shape[0] < n:
        env = np.concatenate([env, np.zeros(n - env.shape[0])])
    return sig * env


def lfo(dur: float, rate: float, sr: int = DEFAULT_SR, shape: str = "sine", depth: float = 1.0, phase0: float = 0.0) -> np.ndarray:
    t = np.arange(samples(dur, sr)) / sr
    ph = (t * rate + phase0) % 1.0
    if shape == "sine":
        v = np.sin(2 * np.pi * ph)
    elif shape == "triangle":
        v = 4 * np.abs(ph - 0.5) - 1
    elif shape == "square":
        v = np.where(ph < 0.5, 1.0, -1.0)
    elif shape == "saw":
        v = 2 * ph - 1
    elif shape == "random":
        rng = np.random.default_rng(7)
        steps = int(np.ceil(dur * rate)) + 1
        vals = rng.uniform(-1, 1, steps)
        v = vals[np.minimum((t * rate).astype(int), steps - 1)]
    else:
        raise ValueError(f"unknown lfo shape {shape}")
    return v * depth
