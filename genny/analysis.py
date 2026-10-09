"""How a sound will be heard, not just how big its waveform is: loudness, sharpness, brightness.

Used by `genny info` (so an agent can check a render is not piercing without ears) and by
examples/level_match.py (which level-matches the instruments).

  loudness   K-weighted (ITU-R BS.1770) level of the loudest window, in LUFS-like dB. Two files with
             the same peak can differ by 15 dB here; this is the number that matches "how loud".
  ear        The same idea weighted like the ear at game volume (A-weighting plus the ear-canal
             bump around 3 kHz) over the loudest 200 ms: bass counts for little, 2-5 kHz for a lot.
             A full-scale sine reads -6 at 500 Hz, -3 at 1 kHz, 0 at 2 kHz, +2 at 3 kHz. This is what
             the spec renderer's `max_loudness` ceiling (-6 by default) limits, because a clean high
             tone normalised to full scale is piercing however pure it is.
  sharpness  Zwicker / DIN 45692 style, in acum: where the loudness sits on the ear's frequency
             scale. A pure tone reads about 0.6 at C5, 1.0 at C6 and 1.5 at C7. For anything with a
             pitch, more than ~0.6 above the pure tone of the same note, or above ~2.0 in absolute
             terms, starts to bite; hats and cymbals are 2.5-3.5 by nature (keep them quiet).
  above 2 kHz / 5 kHz  share of the energy up there. Tonal sounds: over ~3 % above 5 kHz is harsh.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter


def _mono(y: np.ndarray) -> np.ndarray:
    return y if y.ndim == 1 else y.mean(axis=1)


def _k_weight(y: np.ndarray, sr: int) -> np.ndarray:
    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / sr)
    vh = 10 ** (gain_db / 20)
    vb = vh ** 0.4996667741545416
    a0 = 1 + k / q + k * k
    y = lfilter([(vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0],
                [1, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0], y)
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / sr)
    a0 = 1 + k / q + k * k
    return lfilter([1, -2, 1], [1, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0], y)


def loudness(y: np.ndarray, sr: int, window: float = 0.4) -> float:
    """K-weighted level (dB, LUFS-like) of the loudest `window` seconds."""
    p = _k_weight(_mono(y), sr) ** 2
    if p.size == 0:
        return -120.0
    n = max(1, min(p.size, int(window * sr)))
    c = np.cumsum(np.concatenate([[0.0], p]))
    return float(-0.691 + 10 * np.log10((c[n:] - c[:-n]).max() / n + 1e-12))


def _ear_weight(f: np.ndarray) -> np.ndarray:
    """Linear gain of the ear weighting at frequencies f: IEC A-weighting x a +4 dB bump at 3.3 kHz."""
    f2 = np.maximum(f, 1.0) ** 2
    a = (12194.0 ** 2 * f2 ** 2) / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194.0 ** 2)) / 0.7943
    bump = 10 ** (4.0 * np.exp(-0.5 * (np.log2(np.maximum(f, 1.0) / 3300.0) / 0.55) ** 2) / 20)
    return a * bump


def ear_loudness(y: np.ndarray, sr: int, window: float = 0.2) -> float:
    """Ear-weighted level (dB re full scale) of the loudest `window` seconds. Sounds shorter than the
    window are averaged over the whole window, as the ear does: a 50 ms blip is quieter than a held tone."""
    y = _mono(y)
    if y.size == 0:
        return -120.0
    n = max(1, int(window * sr))
    spec = np.fft.rfft(y)
    p = np.fft.irfft(spec * _ear_weight(np.fft.rfftfreq(y.size, 1 / sr)), y.size) ** 2
    if p.size <= n:
        return float(10 * np.log10(p.sum() / n + 1e-12))
    c = np.cumsum(np.concatenate([[0.0], p]))
    return float(10 * np.log10((c[n:] - c[:-n]).max() / n + 1e-12))


def _bark(f):
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def sharpness(y: np.ndarray, sr: int) -> float:
    """Sharpness in acum (energy-weighted mean over 46 ms frames)."""
    y = _mono(y)
    win, hop = 2048, 1024
    if y.size < win:
        y = np.concatenate([y, np.zeros(win - y.size)])
    w = np.hanning(win)
    frames = np.array([np.abs(np.fft.rfft(y[i:i + win] * w)) ** 2 for i in range(0, y.size - win + 1, hop)])
    z = _bark(np.fft.rfftfreq(win, 1 / sr))
    band = np.stack([frames[:, (z >= b) & (z < b + 1)].sum(axis=1) for b in range(24)], axis=1)
    zc = np.arange(24) + 0.5
    fc = 1960 * (zc + 0.53) / (26.28 - zc) / 1000.0
    ear_db = -6.5 * np.exp(-0.6 * (fc - 3.3) ** 2) + 1e-3 * fc ** 4 + 3.64 * fc ** -0.8   # threshold in quiet
    spec = (band * 10 ** (-(ear_db - ear_db.min()) / 10)) ** 0.23                          # specific loudness
    g = np.where(zc < 15.8, 1.0, 0.15 * np.exp(0.42 * (zc - 15.8)) + 0.85)
    s = 0.11 * (spec * g * zc).sum(axis=1) / (spec.sum(axis=1) + 1e-12)
    e = frames.sum(axis=1)
    keep = e > e.max() * 1e-3
    return float((s[keep] * e[keep]).sum() / (e[keep].sum() + 1e-20))


def brightness(y: np.ndarray, sr: int) -> dict:
    """{"centroid": Hz, "above_2k": fraction of energy, "above_5k": fraction of energy}."""
    y = _mono(y)
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size))) ** 2
    f = np.fft.rfftfreq(y.size, 1 / sr)
    tot = spec.sum() + 1e-20
    return {"centroid": float((spec * f).sum() / tot), "above_2k": float(spec[f >= 2000].sum() / tot),
            "above_5k": float(spec[f >= 5000].sum() / tot)}


def describe(y: np.ndarray, sr: int) -> dict:
    """Everything `genny info` prints."""
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    mono = _mono(y)
    rms = float(np.sqrt(np.mean(mono ** 2))) if mono.size else 0.0
    out = {"seconds": y.shape[0] / sr, "channels": 1 if y.ndim == 1 else y.shape[1], "sr": sr,
           "peak_db": 20 * math.log10(peak + 1e-9), "rms_db": 20 * math.log10(rms + 1e-9),
           "loudness": loudness(y, sr), "ear": ear_loudness(y, sr), "sharpness": sharpness(y, sr)}
    out.update(brightness(y, sr))
    return out
