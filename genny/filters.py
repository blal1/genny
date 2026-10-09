"""Filters (RBJ biquads through scipy.lfilter). Static or block-wise time-varying cutoff."""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter, lfilter_zi

from .core import DEFAULT_SR


def _rbj(kind: str, fc: float, sr: int, q: float = 0.707, gain_db: float = 0.0):
    fc = float(np.clip(fc, 10.0, sr * 0.49))
    w0 = 2 * np.pi * fc / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2 * max(q, 0.05))
    A = 10 ** (gain_db / 40)
    if kind == "lowpass":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "highpass":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "bandpass":
        b = [alpha, 0, -alpha]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "notch":
        b = [1, -2 * cw, 1]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind == "lowshelf":
        sq = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) - (A - 1) * cw + sq), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sq)]
        a = [(A + 1) + (A - 1) * cw + sq, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sq]
    elif kind == "highshelf":
        sq = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) + (A - 1) * cw + sq), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sq)]
        a = [(A + 1) - (A - 1) * cw + sq, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sq]
    elif kind == "allpass":
        b = [1 - alpha, -2 * cw, 1 + alpha]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    else:
        raise ValueError(f"unknown filter kind {kind}")
    b = np.array(b) / a[0]
    a = np.array(a) / a[0]
    return b, a


def _per_channel(fn, x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return fn(x)
    return np.stack([fn(x[:, c]) for c in range(x.shape[1])], axis=1)


def biquad(x: np.ndarray, kind: str, cutoff, sr: int = DEFAULT_SR, q: float = 0.707, gain_db: float = 0.0,
           order: int = 1, block: int = 64) -> np.ndarray:
    """Apply an RBJ biquad. `cutoff` may be a scalar or a per-sample array (block-wise modulation).

    order = number of cascaded biquads (1 = 12 dB/oct, 2 = 24 dB/oct).
    """
    def one(sig: np.ndarray) -> np.ndarray:
        y = sig
        for _ in range(max(1, order)):
            if np.isscalar(cutoff):
                b, a = _rbj(kind, float(cutoff), sr, q, gain_db)
                y = lfilter(b, a, y)
            else:
                fc = np.asarray(cutoff, dtype=np.float64)
                if fc.shape[0] != y.shape[0]:
                    fc = np.interp(np.linspace(0, 1, y.shape[0]), np.linspace(0, 1, fc.shape[0]), fc)
                out = np.empty_like(y)
                zi = None
                for s in range(0, y.shape[0], block):
                    e = min(s + block, y.shape[0])
                    b, a = _rbj(kind, float(fc[s:e].mean()), sr, q, gain_db)
                    if zi is None:
                        zi = lfilter_zi(b, a) * y[0]
                    out[s:e], zi = lfilter(b, a, y[s:e], zi=zi)
                y = out
        return y

    return _per_channel(one, x)


def lowpass(x, cutoff, sr=DEFAULT_SR, q=0.707, order=1):
    return biquad(x, "lowpass", cutoff, sr, q, order=order)


def highpass(x, cutoff, sr=DEFAULT_SR, q=0.707, order=1):
    return biquad(x, "highpass", cutoff, sr, q, order=order)


def bandpass(x, center, sr=DEFAULT_SR, q=1.0, order=1):
    return biquad(x, "bandpass", center, sr, q, order=order)


def notch(x, center, sr=DEFAULT_SR, q=4.0):
    return biquad(x, "notch", center, sr, q)


def peak(x, center, sr=DEFAULT_SR, q=1.0, gain_db=6.0):
    return biquad(x, "peak", center, sr, q, gain_db)


def lowshelf(x, cutoff, sr=DEFAULT_SR, gain_db=6.0):
    return biquad(x, "lowshelf", cutoff, sr, 0.707, gain_db)


def highshelf(x, cutoff, sr=DEFAULT_SR, gain_db=6.0):
    return biquad(x, "highshelf", cutoff, sr, 0.707, gain_db)


def dc_block(x, sr=DEFAULT_SR):
    return biquad(x, "highpass", 20.0, sr, 0.707)
