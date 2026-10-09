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
    env_a = np.linspace(0, 1, a) ** (1.0 / curve) if a > 1 else np.ones(a)
    env_d = sustain + (1 - sustain) * (1 - np.linspace(0, 1, d)) ** curve if d > 1 else np.full(d, sustain)
    env_s = np.full(s, sustain)
    on = np.concatenate([env_a, env_d, env_s])[:n_on]
    if n_rel > 0:
        last = on[-1] if on.size else sustain
        rel = last * (1 - np.linspace(0, 1, n_rel)) ** curve
        return np.concatenate([on, rel])
    return on


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
