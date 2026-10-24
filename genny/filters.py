"""Filters (RBJ biquads through scipy.lfilter). Static cutoff, or a per-sample cutoff: a TPT state-variable
filter for lowpass / highpass / bandpass / notch, block-wise RBJ for the others."""
from __future__ import annotations

import math

import numba
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


_SVF_MODES = {"lowpass": 0, "highpass": 1, "bandpass": 2, "notch": 3}


@numba.njit(cache=True)
def _svf_kernel(x, fc, sr, r, mode):
    """Zavalishin's TPT 2-pole SVF ("The Art of VA Filter Design" 2.1, eq. 4.14 and its code), g = tan(pi fc/fs)
    per sample. Same response as the RBJ biquad at a fixed cutoff, but the states are the integrators, so a
    swept cutoff neither zippers nor blows up."""
    y = np.empty_like(x)
    s1 = 0.0
    s2 = 0.0
    for n in range(x.shape[0]):
        f = min(max(fc[n], 10.0), 0.49 * sr)
        g = math.tan(math.pi * f / sr)
        hp = (x[n] - (2.0 * r + g) * s1 - s2) / (1.0 + 2.0 * r * g + g * g)
        v1 = g * hp
        bp = v1 + s1
        s1 = bp + v1
        v2 = g * bp
        lp = v2 + s2
        s2 = lp + v2
        if mode == 0:
            y[n] = lp
        elif mode == 1:
            y[n] = hp
        elif mode == 2:
            y[n] = 2.0 * r * bp           # unit gain at the centre, as the RBJ band-pass
        else:
            y[n] = x[n] - 2.0 * r * bp
    return y


def svf(x: np.ndarray, kind: str, cutoff, sr: int = DEFAULT_SR, q: float = 0.707) -> np.ndarray:
    """TPT state-variable filter (lowpass, highpass, bandpass, notch) with a scalar or per-sample cutoff."""
    def one(sig):
        fc = np.broadcast_to(np.asarray(cutoff, dtype=np.float64), sig.shape)
        if fc.shape[0] != sig.shape[0]:
            fc = np.interp(np.linspace(0, 1, sig.shape[0]), np.linspace(0, 1, fc.shape[0]), fc)
        return _svf_kernel(np.ascontiguousarray(sig, dtype=np.float64), np.ascontiguousarray(fc), float(sr),
                           0.5 / max(float(q), 0.05), _SVF_MODES[kind])
    return _per_channel(one, np.asarray(x, float))


def biquad(x: np.ndarray, kind: str, cutoff, sr: int = DEFAULT_SR, q: float = 0.707, gain_db: float = 0.0,
           order: int = 1, block: int = 64) -> np.ndarray:
    """Apply an RBJ biquad. `cutoff` may be a scalar or a per-sample array: a modulated lowpass /
    highpass / bandpass / notch runs as a TPT state-variable filter (`svf`), the other kinds block-wise.

    order = number of cascaded biquads (1 = 12 dB/oct, 2 = 24 dB/oct).
    """
    if not np.isscalar(cutoff) and kind in _SVF_MODES:
        y = x
        for _ in range(max(1, order)):
            y = svf(y, kind, cutoff, sr, q)
        return y
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
