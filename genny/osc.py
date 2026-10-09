"""Oscillators and noise. All accept a scalar frequency or a per-sample frequency array (for sweeps)."""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .core import DEFAULT_SR, samples

WAVES = ["sine", "saw", "square", "pulse", "triangle", "supersaw", "fm", "noise", "pink", "brown", "organ", "wavefold"]


def _freq_array(freq, n: int) -> np.ndarray:
    if np.isscalar(freq):
        return np.full(n, float(freq))
    f = np.asarray(freq, dtype=np.float64)
    if f.shape[0] != n:
        f = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, f.shape[0]), f)
    return f


def phase(freq, n: int, sr: int = DEFAULT_SR, phase0: float = 0.0) -> np.ndarray:
    """Phase in cycles (0..1 wrapping) integrating a possibly time-varying frequency."""
    f = _freq_array(freq, n)
    ph = np.cumsum(f / sr) + phase0
    return ph


def sine(freq, n: int, sr: int = DEFAULT_SR, phase0: float = 0.0) -> np.ndarray:
    return np.sin(2 * np.pi * phase(freq, n, sr, phase0))


def _polyblep(t: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """PolyBLEP residual for discontinuities at t=0 (t in 0..1)."""
    out = np.zeros_like(t)
    m1 = t < dt
    tt = t[m1] / dt[m1]
    out[m1] = tt + tt - tt * tt - 1.0
    m2 = t > 1.0 - dt
    tt = (t[m2] - 1.0) / dt[m2]
    out[m2] = tt * tt + tt + tt + 1.0
    return out


def saw(freq, n: int, sr: int = DEFAULT_SR, phase0: float = 0.0) -> np.ndarray:
    f = _freq_array(freq, n)
    dt = np.maximum(f / sr, 1e-9)
    t = phase(f, n, sr, phase0) % 1.0
    naive = 2.0 * t - 1.0
    return naive - _polyblep(t, dt)


def pulse(freq, n: int, sr: int = DEFAULT_SR, width: float = 0.5, phase0: float = 0.0) -> np.ndarray:
    f = _freq_array(freq, n)
    dt = np.maximum(f / sr, 1e-9)
    t = phase(f, n, sr, phase0) % 1.0
    naive = np.where(t < width, 1.0, -1.0)
    out = naive + _polyblep(t, dt) - _polyblep((t + 1.0 - width) % 1.0, dt)
    return out


def square(freq, n: int, sr: int = DEFAULT_SR, phase0: float = 0.0) -> np.ndarray:
    return pulse(freq, n, sr, 0.5, phase0)


def triangle(freq, n: int, sr: int = DEFAULT_SR, phase0: float = 0.0) -> np.ndarray:
    t = phase(freq, n, sr, phase0) % 1.0
    return 4.0 * np.abs(t - 0.5) - 1.0


def supersaw(freq, n: int, sr: int = DEFAULT_SR, voices: int = 7, detune: float = 0.4, spread: float = 1.0) -> np.ndarray:
    """Detuned stack of saws. detune in semitones total spread."""
    f = _freq_array(freq, n)
    out = np.zeros(n)
    rng = np.random.default_rng(1234)
    for i in range(voices):
        pos = (i / max(1, voices - 1) - 0.5) * 2 if voices > 1 else 0.0
        ratio = 2 ** (pos * detune / 12.0 * spread)
        out += saw(f * ratio, n, sr, phase0=rng.random())
    return out / np.sqrt(voices)


def fm(freq, n: int, sr: int = DEFAULT_SR, ratio: float = 2.0, index=1.0, phase0: float = 0.0) -> np.ndarray:
    """Two-operator FM. index can be a scalar or per-sample envelope array."""
    f = _freq_array(freq, n)
    idx = _freq_array(index, n) if not np.isscalar(index) else index
    mod = np.sin(2 * np.pi * phase(f * ratio, n, sr))
    return np.sin(2 * np.pi * phase(f, n, sr, phase0) + idx * mod)


def organ(freq, n: int, sr: int = DEFAULT_SR, drawbars=(1.0, 0.6, 0.4, 0.3, 0.2, 0.1)) -> np.ndarray:
    f = _freq_array(freq, n)
    out = np.zeros(n)
    harmonics = [1, 2, 3, 4, 6, 8]
    for h, g in zip(harmonics, drawbars):
        out += g * sine(f * h, n, sr)
    return out / sum(drawbars)


def additive(freq, n: int, sr: int = DEFAULT_SR, partials=None, inharmonic: float = 0.0) -> np.ndarray:
    """partials: list of (harmonic_number, amplitude, decay_seconds). Inharmonic stretches upper partials (piano/bell)."""
    f = _freq_array(freq, n)
    t = np.arange(n) / sr
    out = np.zeros(n)
    nyq = sr / 2
    for h, amp, dec in partials or [(1, 1.0, 1.0)]:
        stretch = np.sqrt(1 + inharmonic * h * h)
        fh = f * h * stretch
        if np.max(fh) >= nyq:
            continue
        out += amp * np.exp(-t / max(dec, 1e-3)) * sine(fh, n, sr)
    return out


def wavefold(freq, n: int, sr: int = DEFAULT_SR, amount: float = 2.0) -> np.ndarray:
    x = sine(freq, n, sr) * amount
    return np.abs(((x + 1) % 4) - 2) - 1


def white(n: int, seed: int | None = None) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1, 1, n)


def pink(n: int, seed: int | None = None) -> np.ndarray:
    """Paul Kellet's economical pink filter."""
    w = white(n, seed)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    return lfilter(b, a, w) * 3.5


def brown(n: int, seed: int | None = None, sr: int = DEFAULT_SR) -> np.ndarray:
    w = white(n, seed)
    x = lfilter([1], [1, -0.995], w)
    return x / (np.max(np.abs(x)) + 1e-9)


def noise(n: int, color: str = "white", seed: int | None = None, sr: int = DEFAULT_SR) -> np.ndarray:
    if color == "white":
        return white(n, seed)
    if color == "pink":
        return pink(n, seed)
    if color == "brown":
        return brown(n, seed, sr)
    raise ValueError(f"unknown noise color {color}")


def osc(wave: str, freq, n: int, sr: int = DEFAULT_SR, **kw) -> np.ndarray:
    """Generic dispatcher used by the synth layer and the sfx generators."""
    wave = wave.lower()
    if wave == "sine":
        return sine(freq, n, sr)
    if wave == "saw":
        return saw(freq, n, sr)
    if wave == "square":
        return square(freq, n, sr)
    if wave == "pulse":
        return pulse(freq, n, sr, width=float(kw.get("width", 0.25)))
    if wave in ("triangle", "tri"):
        return triangle(freq, n, sr)
    if wave == "supersaw":
        return supersaw(freq, n, sr, voices=int(kw.get("voices", 7)), detune=float(kw.get("detune", 0.4)))
    if wave == "fm":
        return fm(freq, n, sr, ratio=float(kw.get("ratio", 2.0)), index=kw.get("index", 1.0))
    if wave == "organ":
        return organ(freq, n, sr)
    if wave == "wavefold":
        return wavefold(freq, n, sr, amount=float(kw.get("amount", 2.0)))
    # Noise is seeded (0 unless the caller passes one): the same spec always renders the same file.
    seed = kw.get("seed")
    seed = 0 if seed is None else int(seed)
    if wave in ("noise", "white"):
        return white(n, seed)
    if wave == "pink":
        return pink(n, seed)
    if wave == "brown":
        return brown(n, seed, sr)
    raise ValueError(f"unknown wave {wave!r}; choose from {WAVES}")


def sweep(start_hz: float, end_hz: float, dur: float, sr: int = DEFAULT_SR, curve: str = "exp") -> np.ndarray:
    """Per-sample frequency array from start to end. curve: 'exp' (musical) or 'lin'."""
    n = samples(dur, sr)
    if curve == "lin":
        return np.linspace(start_hz, end_hz, n)
    start_hz = max(start_hz, 1e-3)
    end_hz = max(end_hz, 1e-3)
    return start_hz * (end_hz / start_hz) ** np.linspace(0, 1, n)
