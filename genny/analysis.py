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
    out["crest_db"] = crest_factor_db(y)
    out["log_attack"] = log_attack_time(y, sr)
    out["flux"] = spectral_flux(y, sr)
    out["true_peak_db"] = true_peak_db(y) if y.size else -240.0
    return out


# ------------------------------------------------------------------------------------------
# QA measurements (artefacts rather than perception); used by tests/test_qa.py
# ------------------------------------------------------------------------------------------

def crest_factor_db(y: np.ndarray) -> float:
    """20 log10(peak / RMS). Under ~6 dB a sound is over-compressed or clipped (a sine is 3 dB)."""
    y = _mono(y)
    rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
    return 20 * math.log10((float(np.max(np.abs(y))) + 1e-12) / (rms + 1e-12)) if y.size else 0.0


def spectral_flatness(y: np.ndarray) -> float:
    """Geometric / arithmetic mean of the power spectrum (Hann window): 0 for a pure tone, 1 for white
    noise. A rise on a tonal note points to aliasing or an instability."""
    s = np.abs(np.fft.rfft(_mono(y) * np.hanning(len(y)))) ** 2 + 1e-20
    return float(np.exp(np.mean(np.log(s))) / np.mean(s))


def alias_snr(y: np.ndarray, sr: int, f0: float, guard: int = 4) -> float:
    """Energy on the harmonics k f0 (within ``guard`` bins, 4-term Blackman-Harris window) over the
    energy everywhere else above 20 Hz, in dB: high = clean, low = aliased or noisy."""
    from scipy.signal.windows import blackmanharris
    y = _mono(y)
    S = np.abs(np.fft.rfft(y * blackmanharris(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1.0 / sr)
    df = f[1]
    k = np.round(f / f0)
    harm = (k >= 1) & (np.abs(f - k * f0) <= guard * df)
    rest = ~harm & (f > 20.0)
    return 10 * math.log10((S[harm].sum() + 1e-30) / (S[rest].sum() + 1e-30))


def detect_clicks(y: np.ndarray, k: float = 6.0, window: int = 441) -> np.ndarray:
    """Sample indices where |second difference| exceeds k times its local RMS (over ``window`` samples,
    10 ms at 44.1 kHz): steps and spikes that a smooth signal does not have. Meant for smooth voices
    (pads, bowed, blown); saws and percussive attacks have edges of their own."""
    d = np.abs(np.diff(_mono(y), 2))
    if d.size == 0:
        return np.zeros(0, int)
    local = np.sqrt(np.convolve(d * d, np.ones(window) / window, mode="same"))
    return np.flatnonzero(d > k * local + 1e-9) + 1


def true_peak_db(y: np.ndarray) -> float:
    """4x oversampled peak in dBTP (ITU-R BS.1770 true-peak approximation)."""
    from scipy.signal import resample_poly
    up = resample_poly(np.asarray(y, float), 4, 1, axis=0)
    return 20 * math.log10(float(np.max(np.abs(up))) + 1e-12)


def spectral_flux(y: np.ndarray, sr: int) -> float:
    """Median spectral variation (Timbre Toolbox, Peeters et al. 2011 §II.D, after Krimphoff 1994): 1 minus the
    normalised correlation of successive magnitude spectra, Hamming 23.2 ms, hop 5.8 ms; the median keeps the
    silent frames from counting. 0 = a frozen spectrum, ~1e-6 for a struck bar, 2e-4 to 1e-3 for a bowed, blown or pad note, ~0.1 for white noise."""
    x = _mono(y)
    M, R = int(round(0.0232 * sr)), int(round(0.0058 * sr))
    if x.size < M + R:
        return float("nan")
    fr = np.lib.stride_tricks.sliding_window_view(x, M)[::R] * np.hamming(M)
    a = np.abs(np.fft.rfft(fr, axis=1))
    num = (a[1:] * a[:-1]).sum(axis=1)
    den = np.sqrt((a[1:] ** 2).sum(axis=1) * (a[:-1] ** 2).sum(axis=1))
    return float(np.median(1.0 - num / (den + 1e-30)))


def log_attack_time(y: np.ndarray, sr: int, a: float = 3.0) -> float:
    """log10 of the attack duration in s (Timbre Toolbox, Peeters et al., JASA 130:2902, 2011, eq. 4).

    Energy envelope: |Hilbert| low-passed by a 3rd-order 5 Hz Butterworth (zero-phase here). Attack start and
    end by Peeters' weakest-effort method: times t_i where the envelope first reaches 10 %, 20 % .. 100 % of
    its maximum; efforts w_i = t_{i+1} - t_i; start = first level whose effort is below a * mean effort,
    end = last such level (a = 3)."""
    from scipy.signal import butter, hilbert, sosfiltfilt
    x = _mono(y)
    if x.size < 16 or not np.any(x):
        return float("nan")
    env = sosfiltfilt(butter(3, 5.0, fs=sr, output="sos"), np.abs(hilbert(x)))
    env = env[: int(np.argmax(env)) + 1]
    th = np.arange(1, 11) / 10.0 * env[-1]
    t = np.array([np.argmax(env >= h) for h in th], float) / sr
    w = np.diff(t)
    weak = np.flatnonzero(w < a * np.mean(w)) if w.size else np.zeros(0, int)
    if not weak.size:
        return math.log10(max(t[-1] - t[0], 1.0 / sr))
    return math.log10(max(t[weak[-1] + 1] - t[weak[0]], 1.0 / sr))
