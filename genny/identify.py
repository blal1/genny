"""Analysis/resynthesis helpers for fitting procedural models to recordings."""
from __future__ import annotations
import numpy as np
from .core import DEFAULT_SR, resample
from .physical import analysis as _a


def lpc_frames(x: np.ndarray, *, sr: int = DEFAULT_SR, order: int | None = None,
               frame_ms: float = 20.0, hop_ms: float = 10.0):
    """Return time-varying LPC analysis using stable reflection coefficients.

    The research engine works at 48 kHz; arbitrary-rate input is resampled first.
    """
    x = np.asarray(x, float)
    if sr != 48000:
        x = resample(x, sr, 48000)
    return _a.analyse(x, order=order, frame_ms=frame_ms, hop_ms=hop_ms)


def modal_fit(x: np.ndarray, *, sr: int = DEFAULT_SR, **kwargs):
    x = np.asarray(x, float)
    if sr != 48000:
        x = resample(x, sr, 48000)
    return _a.modal_fit(x, **kwargs)
