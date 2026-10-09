"""Buffer helpers, mixing, normalization and WAV I/O."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

DEFAULT_SR = 44100


def samples(dur: float, sr: int = DEFAULT_SR) -> int:
    return max(1, int(round(dur * sr)))


def time_axis(dur: float, sr: int = DEFAULT_SR) -> np.ndarray:
    return np.arange(samples(dur, sr)) / sr


def silence(dur: float, sr: int = DEFAULT_SR) -> np.ndarray:
    return np.zeros(samples(dur, sr), dtype=np.float64)


def is_stereo(x: np.ndarray) -> bool:
    return x.ndim == 2


def to_stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x
    return np.stack([x, x], axis=1)


def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return x.mean(axis=1)


def pad_to(x: np.ndarray, n: int) -> np.ndarray:
    """Zero-pad (or leave) so the buffer has at least n frames."""
    if x.shape[0] >= n:
        return x
    if x.ndim == 1:
        return np.concatenate([x, np.zeros(n - x.shape[0])])
    return np.concatenate([x, np.zeros((n - x.shape[0], x.shape[1]))], axis=0)


def mix(parts: list[tuple[np.ndarray, float]], sr: int = DEFAULT_SR) -> np.ndarray:
    """Mix a list of (signal, start_seconds) at their offsets."""
    if not parts:
        return silence(0.1, sr)
    stereo = any(is_stereo(p[0]) for p in parts)
    total = 0
    for sig, at in parts:
        total = max(total, int(round(at * sr)) + sig.shape[0])
    out = np.zeros((total, 2) if stereo else total, dtype=np.float64)
    for sig, at in parts:
        s = int(round(at * sr))
        if stereo:
            sig = to_stereo(sig)
        out[s:s + sig.shape[0]] += sig
    return out


def db_to_gain(db: float) -> float:
    return 10 ** (db / 20.0)


def normalize(x: np.ndarray, level_db: float = -1.0) -> np.ndarray:
    peak = np.max(np.abs(x)) if x.size else 0.0
    if peak < 1e-9:
        return x
    return x * (db_to_gain(level_db) / peak)


def declick(x: np.ndarray, sr: int = DEFAULT_SR, ms: float = 2.0) -> np.ndarray:
    """Short fades so files never start or end on a discontinuity.
    The fade-in is only applied when the first sample is not already near zero, and is kept very short
    (0.5 ms) so percussive attacks survive."""
    n_out = min(int(sr * ms / 1000), x.shape[0] // 2)
    n_in = min(int(sr * 0.0005), x.shape[0] // 2)
    x = x.copy()
    first = np.max(np.abs(x[0])) if x.size else 0.0
    if n_in > 0 and first > 1e-3:
        ramp = np.linspace(0, 1, n_in)
        x[:n_in] *= ramp[:, None] if x.ndim == 2 else ramp
    if n_out > 0:
        ramp = np.linspace(1, 0, n_out)
        x[-n_out:] *= ramp[:, None] if x.ndim == 2 else ramp
    return x


def trim_tail(x: np.ndarray, threshold_db: float = -70.0, keep_ms: float = 30.0, sr: int = DEFAULT_SR) -> np.ndarray:
    """Remove a near-silent tail (e.g. reverb that decayed to nothing)."""
    mono = np.abs(to_mono(x))
    thr = db_to_gain(threshold_db)
    idx = np.nonzero(mono > thr)[0]
    if idx.size == 0:
        return x
    end = min(x.shape[0], idx[-1] + int(sr * keep_ms / 1000))
    return x[:end]


def write_wav(path: str | Path, x: np.ndarray, sr: int = DEFAULT_SR, bits: int = 16) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(x, -1.0, 1.0)
    channels = 1 if x.ndim == 1 else x.shape[1]
    if bits == 16:
        data = (x * 32767).astype("<i2")
    elif bits == 24:
        ints = (x * 8388607).astype("<i4")
        data = np.ascontiguousarray(ints.view("<u1").reshape(-1, 4)[:, :3])
    else:
        raise ValueError("bits must be 16 or 24")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(bits // 8)
        w.setframerate(sr)
        w.writeframes(data.tobytes())
    return path


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v & 0x800000, v - 0x1000000, v)
        x = v.astype(np.float64) / 8388608.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128) / 128.0
    else:
        raise ValueError(f"unsupported sample width {sw}")
    if ch > 1:
        x = x.reshape(-1, ch)
        if ch > 2:
            x = x[:, :2]
    return x, sr


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Band-limited sample-rate conversion (polyphase FIR). Every physical kernel runs at 48 kHz and
    comes through here; plain linear interpolation, which this used to be, aliases and dulls the top."""
    if sr_from == sr_to:
        return x
    from math import gcd
    from scipy.signal import resample_poly
    n_out = int(round(x.shape[0] * sr_to / sr_from))
    g = gcd(int(sr_from), int(sr_to))
    y = resample_poly(x, int(sr_to) // g, int(sr_from) // g, axis=0)
    return pad_to(y[:n_out], n_out)
