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

import numba
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
    """{"centroid": Hz, "spread": Hz (power-weighted sd around the centroid), "above_2k": fraction of energy,
    "above_5k": fraction of energy}."""
    y = _mono(y)
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size))) ** 2
    f = np.fft.rfftfreq(y.size, 1 / sr)
    tot = spec.sum() + 1e-20
    c = float((spec * f).sum() / tot)
    return {"centroid": c, "spread": float(np.sqrt(((f - c) ** 2 * spec).sum() / tot)),
            "above_2k": float(spec[f >= 2000].sum() / tot), "above_5k": float(spec[f >= 5000].sum() / tot)}


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
    out["effective_duration"] = effective_duration(y, sr)
    out["strong_decay"] = strong_decay(y, sr)
    out["centroid_iqr"] = centroid_iqr(y, sr)
    out["tilt"] = spectral_tilt(y, sr)
    if out["channels"] == 2:
        out["corr"] = interchannel_corr(y, sr)["corr"]
        out["mono_loss_db"] = mono_loss_db(y)
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


# ------------------------------------------------------------------------------------------
# Periodicity and HNR (Boersma 1993, Praat "To Harmonicity (ac)")
# ------------------------------------------------------------------------------------------

def _sinc_peak(r: np.ndarray, i: int, half: int = 16) -> tuple[float, float]:
    """Maximum of r near integer lag i by Hann-windowed sinc interpolation (Boersma 1993 interpolates the
    normalised ACF with a sinc). Returns (lag, value)."""
    from scipy.optimize import minimize_scalar
    j = np.arange(max(0, i - half), min(r.size, i + half + 1))

    def val(t):
        d = t - j
        w = 0.5 + 0.5 * np.cos(np.pi * np.clip(d / (half + 1), -1, 1))
        return float(np.sum(r[j] * np.sinc(d) * w))
    res = minimize_scalar(lambda t: -val(t), bounds=(i - 1.0, i + 1.0), method="bounded", options={"xatol": 1e-4})
    return float(res.x), -float(res.fun)


def periodicity(y: np.ndarray, sr: int, floor: float = 75.0, ceiling: float = 600.0, periods: float = 4.5,
                step: float = 0.01, silence: float = 0.1) -> dict:
    """Boersma's (1993) accurate autocorrelation: Hann frames ``periods`` / ``floor`` long every ``step`` s, mean
    removed, ACF divided by the window's ACF, maximum over lags 1/ceiling .. 1/floor sinc-interpolated. Praat
    defaults for "To Harmonicity (ac)": step 0.01 s, pitch floor 75 Hz, silence threshold 0.1 (frames whose
    peak is under 0.1 of the global peak are silent), 4.5 periods per window (manual page Sound: To Harmonicity
    (ac)). Per frame HNR = 10 log10(r / (1 - r)) (Praat manual, Harmonicity: 99 % periodic = 20 dB).

    Returns {"hnr": mean dB over the non-silent frames with r > 0, "r": median r, "f0": median Hz,
    "frames_hnr": per-frame dB (nan when silent or r <= 0)}. Simplification: Praat's Viterbi path over pitch
    candidates is not ported; the strongest ACF peak in the range is taken."""
    x = _mono(np.asarray(y, float))
    n = int(round(periods / floor * sr))
    if x.size < n or not np.any(x):
        return {"hnr": float("nan"), "r": float("nan"), "f0": float("nan"), "frames_hnr": np.zeros(0)}
    w = np.hanning(n)
    nfft = 1 << int(math.ceil(math.log2(2 * n)))
    rw = np.fft.irfft(np.abs(np.fft.rfft(w, nfft)) ** 2)[:n]
    rw /= rw[0]
    lo, hi = max(2, int(sr / ceiling)), min(int(math.ceil(sr / floor)), n // 2)
    gmax = float(np.max(np.abs(x)))
    hop = max(1, int(round(step * sr)))
    starts = np.arange(0, x.size - n + 1, hop)
    hnr_f, rs, f0 = np.full(starts.size, np.nan), [], []
    for b in range(0, starts.size, 128):                   # 128 frames at a time: bounded memory
        idx = starts[b:b + 128]
        fr = x[idx[:, None] + np.arange(n)]
        loud = np.max(np.abs(fr), axis=1) >= silence * gmax
        fr = (fr - fr.mean(axis=1, keepdims=True)) * w
        ra = np.fft.irfft(np.abs(np.fft.rfft(fr, nfft, axis=1)) ** 2, axis=1)[:, :hi + 20]
        for k in np.flatnonzero(loud & (ra[:, 0] > 0)):
            r = ra[k] / ra[k, 0] / rw[:ra.shape[1]]
            i = lo + int(np.argmax(r[lo:hi + 1]))
            lag, rmax = _sinc_peak(r, i)
            rmax = min(rmax, 1.0 - 1e-9)                    # Praat caps HNR (r -> 1) as well
            if rmax > 0:
                hnr_f[b + k] = 10 * math.log10(rmax / (1.0 - rmax))
                rs.append(rmax)
                f0.append(sr / lag)
    ok = np.isfinite(hnr_f)
    return {"hnr": float(np.mean(hnr_f[ok])) if ok.any() else float("nan"),
            "r": float(np.median(rs)) if rs else float("nan"), "f0": float(np.median(f0)) if f0 else float("nan"),
            "frames_hnr": hnr_f}


def hnr(y: np.ndarray, sr: int, floor: float = 75.0, periods: float = 4.5) -> float:
    """Harmonics-to-noise ratio in dB, Praat "To Harmonicity (ac)" defaults (see ``periodicity``).
    Finger et al. 2009 (women, Praat): 19.3 +- 3.7 dB, range 12.6-26.5."""
    return periodicity(y, sr, floor=floor, periods=periods)["hnr"]


# ------------------------------------------------------------------------------------------
# Clicks on any signal: LPC residual (Essentia ClickDetector, Vaseghi & Rayner 1990)
# ------------------------------------------------------------------------------------------

def detect_clicks_lpc(y: np.ndarray, order: int = 12, frame: int = 512, hop: int = 256,
                      threshold_db: float = 30.0, power_k: float = 10.0, silence_db: float = -50.0) -> np.ndarray:
    """Start samples of impulsive clicks, port of Essentia's ClickDetector (audioproblems/clickdetector.cpp,
    defaults from clickdetector.h: order 12, frame 512, hop 256, detection threshold 30 dB, power estimation
    threshold 10, silence -50 dB). Each frame is inverse filtered by its LPC (stationary part removed), the
    residual is matched-filtered (time-reversed LPC), and a sample is a click where the squared output exceeds
    the residual's robust power (squared residual clipped at ``power_k`` x its median, then averaged) times
    10^(threshold_db/10). Unlike ``detect_clicks`` it predicts the waveform, so struck and plucked attacks pass;
    it finds spikes (impulses), not steps, and a saw's resets are impulses to it too (measured: 118 hits on a
    1 s 110 Hz polyBLEP saw, against 259 for ``detect_clicks``)."""
    from scipy.signal import lfilter as _lf
    from .physical.analysis import _levinson
    x = _mono(np.asarray(y, float))
    start = max(frame // 2 - hop // 2, order)
    end = frame // 2 + hop // 2
    sil = 10 ** (silence_db / 10)
    out = []
    for i0 in range(0, x.size - frame + 1, hop):
        fr = x[i0:i0 + frame]
        if np.mean(fr * fr) < sil:
            continue
        r = np.array([np.dot(fr[:frame - k], fr[k:]) for k in range(order + 1)])
        if r[0] <= 0:
            continue
        a, _, _ = _levinson(r, order)
        a = a / np.max(a)                                   # Essentia normalize(): divide by the maximum
        sub = fr[start - order:end + order]
        e = _lf(a, [1.0], sub)
        emf = _lf(-a, [1.0], e[::-1])[::-1]
        p = e * e
        th = max(float(np.mean(np.minimum(p, np.median(p) * power_k))) * 10 ** (threshold_db / 10), sil)
        det = np.flatnonzero(emf[order:emf.size - order] ** 2 >= th) + order
        if det.size:
            det = det + i0 + start - order
            out.extend(det[np.r_[True, np.diff(det) > 1]].tolist())
    return np.unique(np.asarray(out, int))


# ------------------------------------------------------------------------------------------
# Loudness range (EBU Tech 3342-2023)
# ------------------------------------------------------------------------------------------

def loudness_range(y: np.ndarray, sr: int, window: float = 3.0, hop: float = 0.1) -> float:
    """LRA in LU, EBU Tech 3342: short-term loudness (BS.1770 K-weighting, channel powers summed, 3 s sliding
    window, hop 0.1 s, i.e. 2.9 s overlap, the minimum the Tech allows), absolute gate -70 LUFS, relative gate
    -20 LU under the absolute-gated level, LRA = 95th - 10th percentile (nearest rank as in its Matlab listing).
    nan when the sound is shorter than one window."""
    x = np.asarray(y, float)
    x = x[:, None] if x.ndim == 1 else x
    n, h = int(round(window * sr)), max(1, int(round(hop * sr)))
    if x.shape[0] < n:
        return float("nan")
    p = sum(_k_weight(x[:, c], sr) ** 2 for c in range(x.shape[1]))
    c = np.cumsum(np.concatenate([[0.0], p]))
    st = -0.691 + 10 * np.log10(((c[n:] - c[:-n]) / n)[::h] + 1e-30)
    st = st[st >= -70.0]
    if st.size == 0:
        return float("nan")
    st = np.sort(st[st >= 10 * np.log10(np.mean(10 ** (st / 10))) - 20.0])
    rank = lambda q: st[int(round((st.size - 1) * q / 100.0))]
    return float(rank(95) - rank(10))


# ------------------------------------------------------------------------------------------
# One-shot and harmonic descriptors
# ------------------------------------------------------------------------------------------

def strong_decay(y: np.ndarray, sr: int) -> float:
    """Essentia StrongDecay (sfx/strongdecay.cpp): sqrt(energy / temporal centroid of |x| in s); energy is the
    plain sum of squares, so it scales with sr. High = loud and quickly decaying."""
    x = _mono(np.asarray(y, float))
    a = np.abs(x)
    if x.size < 2 or a.sum() == 0:
        return 0.0
    cen = float((np.arange(x.size) * a).sum() / a.sum()) / sr
    return math.sqrt(float(np.sum(x * x)) / cen) if cen > 0 else 0.0


def effective_duration(y: np.ndarray, sr: int, drop_db: float = 3.0) -> float:
    """Time (s) the RMS envelope (10 ms window, 5 ms hop) stays within ``drop_db`` of its maximum: the
    effective duration of the perceptual timbre study of lot B54 (l. 1214-1228; r = -0.82 with dimension 1)."""
    x = _mono(np.asarray(y, float))
    n, h = max(1, int(0.010 * sr)), max(1, int(0.005 * sr))
    if x.size < n:
        return 0.0
    c = np.cumsum(np.concatenate([[0.0], x * x]))
    e = ((c[n:] - c[:-n]) / n)[::h]
    if e.max() <= 0:
        return 0.0
    return float(np.count_nonzero(e >= e.max() * 10 ** (-drop_db / 10)) * h / sr)


def _harmonic_peaks(x, sr, f0, H):
    """Amplitudes and frequencies of partials 1..H in Blackman frames of 100 ms, hop 25 ms (Peeters et al.
    2011 §II.B.4), each the largest bin within +-f0/2 of h f0, parabolic interpolation on log magnitude.
    Frames under 1e-3 of the loudest frame's energy are dropped."""
    n, hop = int(0.1 * sr), int(0.025 * sr)
    if x.size < n:
        x = np.concatenate([x, np.zeros(n - x.size)])
    fr = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * np.blackman(n)
    S = np.abs(np.fft.rfft(fr, axis=1))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    E = (S ** 2).sum(axis=1)
    keep = E > E.max() * 1e-3
    S, E = S[keep], E[keep]
    rows = np.arange(S.shape[0])
    A, F = np.zeros((S.shape[0], H)), np.zeros((S.shape[0], H))
    for h in range(1, H + 1):
        m = np.flatnonzero(np.abs(f - h * f0) <= 0.5 * f0)
        m = m[(m > 0) & (m < f.size - 1)]
        if m.size == 0:
            break
        i = m[np.argmax(S[:, m], axis=1)]
        a, b, c = (np.log(S[rows, j] + 1e-30) for j in (i - 1, i, i + 1))
        den = a - 2 * b + c
        d = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / np.where(den == 0, 1.0, den), 0.0)
        F[:, h - 1] = (i + d) * f[1]
        A[:, h - 1] = np.exp(b - 0.25 * (a - c) * d)
    return A, F, E, n


def harmonic_descriptors(y: np.ndarray, sr: int, f0: float, H: int = 20) -> dict:
    """Medians over frames of Peeters et al. (JASA 130, 2011) harmonic descriptors, H = 20 partials:
    tristimulus T1, T2, T3 (eq. 19, amplitudes), odd-to-even energy ratio OER (eq. 23), noisiness (eq. 18: energy
    not in the partials / total; partial energy from its peak amplitude and the window's equivalent noise
    bandwidth, exact for steady sines), and Essentia's inharmonicity (tonal/inharmonicity.cpp:
    sum |f_i - round(f_i/f0) f0| a_i^2 / (sum a_i^2 x f0))."""
    x = _mono(np.asarray(y, float))
    A, F, E, n = _harmonic_peaks(x, sr, f0, H)
    s = A.sum(axis=1) + 1e-30
    a2 = A ** 2
    odd, even = a2[:, 0::2].sum(axis=1), a2[:, 1::2].sum(axis=1)
    w = np.blackman(n)
    harm_e = a2.sum(axis=1) * n * (w ** 2).sum() / w.sum() ** 2       # peak |X|^2 x ENBW (bins) = a sine's one-sided energy
    k = np.maximum(np.round(F / f0), 1)
    inh = (np.abs(F - k * f0) * a2).sum(axis=1) / (a2.sum(axis=1) * f0 + 1e-30)
    med = lambda v: float(np.median(v))
    return {"t1": med(A[:, 0] / s), "t2": med(A[:, 1:4].sum(axis=1) / s), "t3": med(A[:, 4:].sum(axis=1) / s),
            "oer": med(odd / (even + 1e-30)), "noisiness": med(np.clip(1.0 - harm_e / (E + 1e-30), 0.0, 1.0)),
            "inharmonicity": med(inh)}


def centroid_iqr(y: np.ndarray, sr: int) -> float:
    """Interquartile range (Hz) of the frame spectral centroid, Hamming 23.2 ms, hop 5.8 ms as ``spectral_flux``:
    the dispersion statistic Peeters et al. 2011 keep next to the median (§V-VI). Frames under 1e-3 of the
    loudest are left out."""
    x = _mono(np.asarray(y, float))
    M, R = int(round(0.0232 * sr)), int(round(0.0058 * sr))
    if x.size < M + R:
        return float("nan")
    a = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(x, M)[::R] * np.hamming(M), axis=1)) ** 2
    f = np.fft.rfftfreq(M, 1.0 / sr)                       # power STFT (one of Peeters' inputs): on magnitudes the
    e = a.sum(axis=1)                                      # Hamming leakage alone spreads a sine's centroid by ~300 Hz
    k = e > e.max() * 1e-3
    if not k.any():
        return float("nan")
    c = (a[k] * f).sum(axis=1) / e[k]
    q1, q3 = np.percentile(c, [25, 75])
    return float(q3 - q1)


# ------------------------------------------------------------------------------------------
# Stereo image
# ------------------------------------------------------------------------------------------

def interchannel_corr(y: np.ndarray, sr: int, max_lag_ms: float = 10.0) -> dict:
    """{"corr": normalised L-R correlation at lag 0, "max": largest |correlation| within +-max_lag_ms, "lag_ms":
    where (positive = R late)}. The maximum of the absolute cross-correlation is the measure of lot B59 (p. 72);
    10 ms covers the precedence range it gives (1 to ~10 ms). Mono input: all 1 / 0."""
    y = np.asarray(y, float)
    if y.ndim == 1 or y.shape[1] == 1:
        return {"corr": 1.0, "max": 1.0, "lag_ms": 0.0}
    from scipy.signal import fftconvolve
    L, R = y[:, 0] - y[:, 0].mean(), y[:, 1] - y[:, 1].mean()
    den = math.sqrt(float(np.dot(L, L) * np.dot(R, R))) + 1e-30
    xc = fftconvolve(R, L[::-1], mode="full") / den
    mid = L.size - 1
    m = min(int(max_lag_ms * 1e-3 * sr), mid)
    seg = xc[mid - m:mid + m + 1]
    i = int(np.argmax(np.abs(seg)))
    return {"corr": float(xc[mid]), "max": float(abs(seg[i])), "lag_ms": (i - m) * 1000.0 / sr}


def mono_loss_db(y: np.ndarray) -> float:
    """Level of the mono fold-down (L+R)/2 against the mean channel power, in dB: 0 for identical channels, about
    -3 for uncorrelated ones, very negative for opposite polarity (what a phone speaker loses)."""
    y = np.asarray(y, float)
    if y.ndim == 1 or y.shape[1] == 1:
        return 0.0
    m = 0.5 * (y[:, 0] + y[:, 1])
    return float(10 * np.log10((np.mean(m * m) + 1e-30) / (0.5 * np.mean(y[:, 0] ** 2 + y[:, 1] ** 2) + 1e-30)))


# ------------------------------------------------------------------------------------------
# Impulse responses: energy decay relief, late-reverb onset
# ------------------------------------------------------------------------------------------

def edr(ir: np.ndarray, sr: int, win: float = 0.035) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Energy decay relief EDR(t_n, f_k) = sum_{m >= n} |H(m, k)|^2 in dB re its start per bin, STFT with a Hann
    window of ``win`` s (30-40 ms in the book of lot chunk_15 §3.2.2), hop win/4. Returns
    (t, f, edr_db[time, freq]). The slope of a column gives that bin's decay: T60 = -60 / slope."""
    from scipy.signal import stft
    h = _mono(np.asarray(ir, float))
    n = int(round(win * sr))
    f, t, Z = stft(h, sr, window="hann", nperseg=n, noverlap=n - n // 4, boundary=None, padded=False)
    P = np.abs(Z.T) ** 2
    E = np.cumsum(P[::-1], axis=0)[::-1]
    return t, f, 10 * np.log10(E / (E[:1] + 1e-30) + 1e-30)


def echo_density(ir: np.ndarray, sr: int, win: float = 0.010) -> tuple[np.ndarray, np.ndarray]:
    """(t, share of samples with |x - mean| above the window's standard deviation) over sliding ``win`` windows
    (10 ms, chunk_15 §3.3 note 5), hop win/4. A Gaussian late field gives 1 - erf(1/sqrt 2) = 0.3173."""
    x = _mono(np.asarray(ir, float))
    n = max(2, int(round(win * sr)))
    hop = max(1, n // 4)
    fr = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    sd = fr.std(axis=1, keepdims=True)
    frac = np.mean(np.abs(fr - fr.mean(axis=1, keepdims=True)) > sd, axis=1)
    return (np.arange(fr.shape[0]) * hop + n / 2) / sr, frac


def late_onset(ir: np.ndarray, sr: int, win: float = 0.010, level: float = 0.30) -> float:
    """Time (s, window centre) of the first 10 ms window with at least ``level`` of its samples over its standard
    deviation: where the IR turns into a Gaussian late field (about 30 %, chunk_15 l. 6132-6145). Sparse early
    echoes give a few %. A Gaussian field scatters by about +-2 % around 31.7 % per 10 ms window at 44.1 kHz, so
    "every later window above 30 %" is not a usable rule. nan if never."""
    t, frac = echo_density(ir, sr, win)
    ok = np.flatnonzero(frac >= level)
    return float(t[ok[0]]) if ok.size else float("nan")


# ------------------------------------------------------------------------------------------
# Critical bands, sensory dissonance and roughness from partials
# ------------------------------------------------------------------------------------------

def hz_to_bark(f):
    """Traunmueller 1990 with Essentia's end corrections (essentiamath.h hz2bark)."""
    f = np.asarray(f, float)
    b = 26.81 * f / (1960.0 + f) - 0.53
    b = np.where(b < 2, b + 0.15 * (2 - b), b)
    return np.where(b > 20.1, b + 0.22 * (b - 20.1), b)


def bark_to_hz(z):
    """Inverse of ``hz_to_bark`` (essentiamath.h bark2hz, corrected: above 20.1 Bark Essentia writes
    (z - 4.422) / 1.22, which does not invert its own hz2bark, z' = 1.22 z - 4.422)."""
    z = np.asarray(z, float)
    z = np.where(z < 2, (z - 0.3) / 0.85, z)
    z = np.where(z > 20.1, (z + 4.422) / 1.22, z)
    return 1960.0 * (z + 0.53) / (26.28 - z)


def critical_bandwidth(f):
    """Zwicker's critical bandwidth in Hz, Fastl & Zwicker eq. 6.2: 25 + 75 (1 + 1.4 (f/kHz)^2)^0.69."""
    return 25.0 + 75.0 * (1.0 + 1.4 * (np.asarray(f, float) / 1000.0) ** 2) ** 0.69


def _plomp_levelt(df):
    """Consonance 0..1 of two partials df critical bandwidths apart: Essentia's polynomial fit to Plomp &
    Levelt 1965 (tonal/dissonance.cpp), 1 outside 0..1.18."""
    df = np.asarray(df, float)
    c = (-6.58977878 * df ** 5 + 28.58224226 * df ** 4 - 47.36739986 * df ** 3 + 35.70679761 * df ** 2
         - 10.36526344 * df + 1.00026609)
    return np.where((df < 0) | (df > 1.18), 1.0, np.clip(c, 0.0, 1.0))


def dissonance(freqs, intensities) -> float:
    """Sensory dissonance 0..1 of a set of partials (Hz, intensities = squared amplitudes), port of Essentia
    tonal/dissonance.cpp: intensities A-weighted twice, every pair within +-1.18 Bark (partials 50 Hz-10 kHz)
    adds (1 - consonance) x their share of loudness, each peak capped at its own share, total / 2. Critical
    bandwidth there: 52548 / (z^2 - 52.56 z + 690.39)."""
    f = np.asarray(freqs, float)
    o = np.argsort(f)
    f, m = f[o], np.asarray(intensities, float)[o]
    aw = 1.25893 * 12200 ** 2 * f ** 4 / ((f * f + 20.6 ** 2) * (f * f + 12200 ** 2) * np.sqrt(f * f + 107.7 ** 2)
                                          * np.sqrt(f * f + 737.9 ** 2))
    loud = m * aw * aw
    tot = loud.sum()
    if tot == 0:
        return 0.0
    cbw = lambda x: 52548.0 / (hz_to_bark(x) ** 2 - 52.56 * hz_to_bark(x) + 690.39)
    total = 0.0
    for i in range(f.size):
        if f[i] <= 50:
            continue
        z = hz_to_bark(f[i])
        sel = (f >= bark_to_hz(z - 1.18)) & (f < bark_to_hz(z + 1.18)) & (f < 10000)
        d = 1.0 - _plomp_levelt(np.abs(f[sel] - f[i]) / np.minimum(cbw(f[i]), cbw(f[sel])))
        peak = float(np.sum(np.where(d > 0, d * (loud[sel] + loud[i]) / tot, 0.0)))
        total += min(peak, loud[i] / tot)
    return total / 2.0


def vassilakis_roughness(freqs, amps) -> float:
    """Vassilakis roughness of sine partials (lot B58, eq. 10-13): sum over pairs of
    0.5 (Am An)^0.1 (2 min / (Am + An))^3.11 (e^(-3.5 s |df|) - e^(-5.75 s |df|)), s = 0.24 / (0.0207 f_min + 18.96).
    A pair is roughest at s |df| = ln(5.75 / 3.5) / 2.25."""
    f, a = np.asarray(freqs, float), np.asarray(amps, float)
    i, j = np.triu_indices(f.size, 1)
    fm, am, an = np.minimum(f[i], f[j]), a[i], a[j]
    s = 0.24 / (0.0207 * fm + 18.96)
    d = np.abs(f[i] - f[j])
    r = 0.5 * (am * an) ** 0.1 * (2 * np.minimum(am, an) / (am + an + 1e-30)) ** 3.11 * (np.exp(-3.5 * s * d) - np.exp(-5.75 * s * d))
    return float(r.sum())


# ------------------------------------------------------------------------------------------
# Psychoacoustic annoyance and sensory pleasantness (Fastl & Zwicker eq. 16.2-16.4 and 9.2)
# ------------------------------------------------------------------------------------------

def annoyance(n5: float, sharpness_acum: float, fluctuation_vacil: float = 0.0, roughness_asper: float = 0.0) -> float:
    """PA = N5 (1 + sqrt(wS^2 + wFR^2)); wS = (S - 1.75) 0.25 lg(N5 + 10) for S > 1.75 acum, else 0;
    wFR = 2.18 / N5^0.4 (0.4 F + 0.6 R), N5 in sone (``loudness_percentiles``), S in acum, F in vacil, R in asper."""
    ws = (sharpness_acum - 1.75) * 0.25 * math.log10(n5 + 10.0) if sharpness_acum > 1.75 else 0.0
    wfr = 2.18 / max(n5, 1e-9) ** 0.4 * (0.4 * fluctuation_vacil + 0.6 * roughness_asper)
    return float(n5 * (1.0 + math.sqrt(ws * ws + wfr * wfr)))


def pleasantness(r_rel: float, s_rel: float, t_rel: float, n_rel: float) -> float:
    """Relative sensory pleasantness P/P0 = e^(-0.7 R) e^(-1.08 S) (1.24 - e^(-2.43 T)) e^(-(0.023 N)^2), each
    argument relative to the reference sound (Fastl & Zwicker eq. 9.2, transcription of lot B08; the OCR of the
    book is garbled there)."""
    return float(math.exp(-0.7 * r_rel) * math.exp(-1.08 * s_rel) * (1.24 - math.exp(-2.43 * t_rel))
                 * math.exp(-(0.023 * n_rel) ** 2))


QC_DURATION = {"coin": (0.20, 0.55), "jump": (0.15, 0.45), "laser": (0.10, 0.40), "explosion": (0.45, 1.10),
               "powerup": (0.20, 0.65), "hit": (0.08, 0.30), "ui_click": (0.03, 0.12), "alarm": (0.35, 1.10),
               "pickup": (0.14, 0.50), "door": (0.30, 0.90), "teleport": (0.40, 1.00), "engine": (0.65, 1.50),
               "blip": (0.03, 0.13), "whoosh": (0.18, 0.60), "land": (0.08, 0.35), "shield": (0.28, 0.80)}


def wav_check(y: np.ndarray, sr: int, category: str | None = None) -> list[str]:
    """Problems of a game sfx file (empty list = passes): peak 0.50-0.95, near-silence (|s| < 4/32767) at most 25 %
    of the file and no gap longer than min(120 ms, 40 % of it), |DC| < 0.01, first and last sample <= 0.02 and the
    three at each end <= 0.05, and the length of its ``category`` (``QC_DURATION``)."""
    a = np.abs(np.asarray(y, float))
    a = a.max(axis=1) if a.ndim == 2 else a
    n, bad = len(a), []
    if n == 0:
        return ["empty"]
    pk = float(a.max())
    if not 0.50 <= pk <= 0.95:
        bad.append(f"peak {pk:.3f} outside 0.50-0.95")
    quiet = a < 4.0 / 32767.0
    if quiet.mean() > 0.25:
        bad.append(f"silence {100 * quiet.mean():.0f}% > 25%")
    edges = np.diff(np.concatenate(([0], quiet.astype(np.int8), [0])))
    run = int((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max()) if quiet.any() else 0
    if run > min(0.120 * sr, 0.40 * n):
        bad.append(f"silent gap {1000 * run / sr:.0f} ms")
    dc = float(np.abs(np.mean(np.asarray(y, float), axis=0)).max())
    if dc >= 0.01:
        bad.append(f"DC {dc:.3f}")
    if max(a[0], a[-1]) > 0.02 or max(a[:3].max(), a[-3:].max()) > 0.05:
        bad.append("clicks at the ends")
    if category:
        lo, hi = QC_DURATION[category]
        if not lo <= n / sr <= hi:
            bad.append(f"{n / sr:.2f} s outside {lo}-{hi} s for {category}")
    return bad


# ------------------------------------------------------------------------------------------
# Third-octave levels, spectral tilt, Zwicker loudness (ISO 532-1 stationary and time-varying)
# ------------------------------------------------------------------------------------------

THIRD_OCTAVES = 1000.0 * 10.0 ** (np.arange(-16, 12) / 10.0)
PESTANA_TILT = -5.0


def third_octave_power(y: np.ndarray, sr: int, centers=THIRD_OCTAVES, order: int = 0) -> np.ndarray:
    """Mean-square value of ``y`` in each third-octave band (base-10 edges fc 10^(+-1/20)), from one FFT.
    ``order`` 0: brick-wall bands; n > 0: Butterworth band-pass of order n per band, |H|^2 = 1/(1 + (Q (f/fc -
    fc/f))^2n), Q = 1/(10^0.05 - 10^-0.05), as the filter bank of a real third-octave analyser (ISO 532-1: n = 3)."""
    x = _mono(np.asarray(y, float))
    p = np.abs(np.fft.rfft(x)) ** 2 / x.size ** 2
    p[1:] *= 2.0
    f = np.fft.rfftfreq(x.size, 1 / sr)
    c = np.asarray(centers, float)
    if order:
        q = 1.0 / (10 ** 0.05 - 10 ** -0.05)
        fp = np.maximum(f, 1e-6)
        return np.array([(p / (1.0 + (q * (fp / fc - fc / fp)) ** (2 * order))).sum() for fc in c])
    edges = np.searchsorted(f, np.concatenate((c * 10 ** -0.05, c[-1:] * 10 ** 0.05)))
    cs = np.concatenate(([0.0], np.cumsum(p)))
    return cs[edges[1:]] - cs[edges[:-1]]


def spectral_tilt(y: np.ndarray, sr: int, lo: float = 100.0, hi: float = 4000.0) -> float:
    """Slope in dB per octave of the power spectral density between ``lo`` and ``hi`` (least squares over
    third-octave bands). White noise reads 0, pink noise -3; commercial mixes average ``PESTANA_TILT``."""
    c = THIRD_OCTAVES[(THIRD_OCTAVES >= lo / 1.01) & (THIRD_OCTAVES <= hi * 1.01)]
    d = third_octave_power(y, sr, c) / (c * (10 ** 0.05 - 10 ** -0.05))
    return float(np.polyfit(np.log2(c), 10 * np.log10(d + 1e-30), 1)[0])


_ZW_RAP = np.array([45, 55, 65, 71, 80, 90, 100, 120], float)
_ZW_DLL = np.array([[-32, -24, -16, -10, -5, 0, -7, -3, 0, -2, 0], [-29, -22, -15, -10, -4, 0, -7, -2, 0, -2, 0],
                    [-27, -19, -14, -9, -4, 0, -6, -2, 0, -2, 0], [-25, -17, -12, -9, -3, 0, -5, -2, 0, -2, 0],
                    [-23, -16, -11, -7, -3, 0, -4, -1, 0, -1, 0], [-20, -14, -10, -6, -3, 0, -4, -1, 0, -1, 0],
                    [-18, -12, -9, -6, -2, 0, -3, -1, 0, -1, 0], [-15, -10, -8, -4, -2, 0, -3, -1, 0, -1, 0]], float)
_ZW_LTQ = np.array([30, 18, 12, 8, 7, 6, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], float)
_ZW_A0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, -1.6, -3.2, -5.4, -5.6, -4, -1.5, 2, 5, 12])
_ZW_DDF = np.array([0, 0, 0.5, 0.9, 1.2, 1.6, 2.3, 2.8, 3, 2, 0, -1.4, -2, -1.9, -1, 0.5, 3, 4, 4.3, 4])
_ZW_DCB = np.array([-0.25, -0.6, -0.8, -0.8, -0.5, 0, 0.5, 1.1, 1.5, 1.7, 1.8, 1.8, 1.7, 1.6, 1.4, 1.2, 0.8, 0.5, 0,
                    -0.5])
_ZW_ZUP = np.array([0.9, 1.8, 2.8, 3.5, 4.4, 5.4, 6.6, 7.9, 9.2, 10.6, 12.3, 13.8, 15.2, 16.7, 18.1, 19.3, 20.6, 21.8,
                    22.7, 23.6, 24.0])
_ZW_RNS = np.array([21.5, 18, 15.1, 11.5, 9, 6.1, 4.4, 3.1, 2.13, 1.36, 0.82, 0.42, 0.30, 0.22, 0.15, 0.10, 0.035, 0])
_ZW_USL = np.array([[13, 8.2, 6.3, 5.5, 5.5, 5.5, 5.5, 5.5], [9, 7.5, 6, 5.1, 4.5, 4.5, 4.5, 4.5],
                    [7.8, 6.7, 5.6, 4.9, 4.4, 3.9, 3.9, 3.9], [6.2, 5.4, 4.6, 4.0, 3.5, 3.2, 3.2, 3.2],
                    [4.5, 3.8, 3.6, 3.2, 2.9, 2.7, 2.7, 2.7], [3.7, 3.0, 2.8, 2.35, 2.2, 2.2, 2.2, 2.2],
                    [2.9, 2.3, 2.1, 1.9, 1.8, 1.7, 1.7, 1.7], [2.4, 1.7, 1.5, 1.35, 1.3, 1.3, 1.3, 1.3],
                    [1.95, 1.45, 1.3, 1.15, 1.1, 1.1, 1.1, 1.1], [1.5, 1.2, 0.94, 0.86, 0.82, 0.82, 0.82, 0.82],
                    [0.72, 0.67, 0.64, 0.63, 0.62, 0.62, 0.62, 0.62], [0.59, 0.53, 0.51, 0.50, 0.42, 0.42, 0.42, 0.42],
                    [0.40, 0.33, 0.26, 0.24, 0.22, 0.22, 0.22, 0.22], [0.27, 0.21, 0.20, 0.18, 0.17, 0.17, 0.17, 0.17],
                    [0.16, 0.15, 0.14, 0.12, 0.11, 0.11, 0.11, 0.11], [0.12, 0.11, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08],
                    [0.09, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06, 0.05], [0.06, 0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02]])


_ZW_FD = np.array([
    [[-6.70260e-4, 6.59453e-4], [-3.75071e-4, 3.61926e-4], [-3.06523e-4, 2.97634e-4]],
    [[-8.47258e-4, 8.30131e-4], [-4.76448e-4, 4.55616e-4], [-3.88773e-4, 3.74685e-4]],
    [[-1.07210e-3, 1.04496e-3], [-6.06567e-4, 5.73553e-4], [-4.94004e-4, 4.71677e-4]],
    [[-1.35836e-3, 1.31535e-3], [-7.74327e-4, 7.22007e-4], [-6.29154e-4, 5.93771e-4]],
    [[-1.72380e-3, 1.65564e-3], [-9.91780e-4, 9.08866e-4], [-8.03529e-4, 7.47455e-4]],
    [[-2.19188e-3, 2.08388e-3], [-1.27545e-3, 1.14406e-3], [-1.02976e-3, 9.40900e-4]],
    [[-2.79386e-3, 2.62274e-3], [-1.64828e-3, 1.44006e-3], [-1.32520e-3, 1.18438e-3]],
    [[-3.57182e-3, 3.30071e-3], [-2.14252e-3, 1.81258e-3], [-1.71397e-3, 1.49082e-3]],
    [[-4.58305e-3, 4.15355e-3], [-2.80413e-3, 2.28135e-3], [-2.23006e-3, 1.87646e-3]],
    [[-5.90655e-3, 5.22622e-3], [-3.69947e-3, 2.87118e-3], [-2.92205e-3, 2.36178e-3]],
    [[-7.65243e-3, 6.57493e-3], [-4.92540e-3, 3.61318e-3], [-3.86007e-3, 2.97240e-3]],
    [[-1.00023e-2, 8.29610e-3], [-6.63788e-3, 4.55999e-3], [-5.15982e-3, 3.75306e-3]],
    [[-1.31230e-2, 1.04220e-2], [-9.02274e-3, 5.73132e-3], [-6.94543e-3, 4.71734e-3]],
    [[-1.73693e-2, 1.30947e-2], [-1.24176e-2, 7.20526e-3], [-9.46002e-3, 5.93145e-3]],
    [[-2.31934e-2, 1.64308e-2], [-1.73009e-2, 9.04761e-3], [-1.30358e-2, 7.44926e-3]],
    [[-3.13292e-2, 2.06370e-2], [-2.44342e-2, 1.13731e-2], [-1.82108e-2, 9.36778e-3]],
    [[-4.28261e-2, 2.59325e-2], [-3.49619e-2, 1.43046e-2], [-2.57855e-2, 1.17912e-2]],
    [[-5.91733e-2, 3.25054e-2], [-5.06072e-2, 1.79513e-2], [-3.69401e-2, 1.48094e-2]],
    [[-8.26348e-2, 4.05894e-2], [-7.40348e-2, 2.24476e-2], [-5.34977e-2, 1.85371e-2]],
    [[-1.17018e-1, 5.08116e-2], [-1.09516e-1, 2.81387e-2], [-7.85097e-2, 2.32872e-2]],
    [[-1.67714e-1, 6.37872e-2], [-1.63378e-1, 3.53729e-2], [-1.16419e-1, 2.93723e-2]],
    [[-2.42528e-1, 7.98576e-2], [-2.45161e-1, 4.43370e-2], [-1.73972e-1, 3.70015e-2]],
    [[-3.53142e-1, 9.96330e-2], [-3.69163e-1, 5.53535e-2], [-2.61399e-1, 4.65428e-2]],
    [[-5.16316e-1, 1.24177e-1], [-5.55473e-1, 6.89403e-2], [-3.93998e-1, 5.86715e-2]],
    [[-7.56635e-1, 1.55023e-1], [-8.34281e-1, 8.58123e-2], [-5.94547e-1, 7.43960e-2]],
    [[-1.10165e0, 1.91713e-1], [-1.23939e0, 1.05243e-1], [-8.91666e-1, 9.40354e-2]],
    [[-1.58477e0, 2.39049e-1], [-1.80505e0, 1.28794e-1], [-1.32500e0, 1.21333e-1]],
    [[-2.50630e0, 1.42308e-1], [-2.19464e0, 2.76470e-1], [-1.90231e0, 1.47304e-1]]])
_ZW_FG = np.array([4.30764e-11, 8.59340e-11, 1.71424e-10, 3.41944e-10, 6.82035e-10, 1.36026e-9, 2.71261e-9,
                   5.40870e-9, 1.07826e-8, 2.14910e-8, 4.28228e-8, 8.54316e-8, 1.70009e-7, 3.38215e-7, 6.71990e-7,
                   1.33531e-6, 2.65172e-6, 5.25477e-6, 1.03780e-5, 2.04870e-5, 4.05198e-5, 7.97914e-5, 1.56511e-4,
                   3.04954e-4, 5.99157e-4, 1.16544e-3, 2.27488e-3, 3.91006e-3])
_ZW_FB = ((1.0, 2.0, 1.0), (1.0, 0.0, -1.0), (1.0, -2.0, 1.0))
ZW_SR = 48000


def _zw_48k(y, sr: int, pa: float) -> np.ndarray:
    x = _mono(np.asarray(y, float)) * pa
    if int(sr) == ZW_SR:
        return x
    from scipy.signal import resample_poly
    g = math.gcd(ZW_SR, int(sr))
    return resample_poly(x, ZW_SR // g, int(sr) // g)


def _zw_band(x: np.ndarray, i: int) -> np.ndarray:
    y = x * _ZW_FG[i]
    for b, (a1, a2) in zip(_ZW_FB, _ZW_FD[i]):
        y = lfilter(b, (1.0, -2.0 - a1, 1.0 - a2), y)
    return y


def _zw_core(lt, field: str) -> np.ndarray:
    lt = np.asarray(lt, float).reshape(28, -1)
    low = lt[:11]
    k = np.zeros(low.shape, int)
    for j in range(7):
        k += (k == j) & (low > (_ZW_RAP[j] - _ZW_DLL[j])[:, None])
    ti = 10 ** ((low + _ZW_DLL[k, np.arange(11)[:, None]]) / 10)
    le = np.vstack((10 * np.log10(np.stack((ti[:6].sum(0), ti[6:9].sum(0), ti[9:].sum(0)))), lt[11:]))
    le = le - _ZW_A0[:, None] + (_ZW_DDF[:, None] if field == "diffuse" else 0.0)
    ltq = _ZW_LTQ[:, None]
    with np.errstate(over="ignore"):
        c = 0.0635 * 10 ** (0.025 * ltq) * ((0.75 + 0.25 * 10 ** ((le - _ZW_DCB[:, None] - ltq) / 10)) ** 0.25 - 1)
    c = np.where(le > ltq, np.maximum(c, 0.0), 0.0)
    c[0] *= np.minimum(1.0, 0.4 + 0.32 * c[0] ** 0.2)
    return np.vstack((c, np.zeros((1, c.shape[1]))))


@numba.njit(cache=True)
def _zw_slopes(core, zup, rns, usl):
    nt = core.shape[1]
    tot = np.zeros(nt)
    ns = np.zeros((240, nt))
    step = 0.10000000149011612
    for t in range(nt):
        n1 = 0.0
        n2 = 0.0
        z = step
        z1 = 0.0
        j = 0
        k = 0
        n = 0.0
        for i in range(21):
            cl = core[i, t]
            zu = zup[i] + 0.0001
            ig = min(max(i - 1, 0), 7)
            done = False
            while not done:
                if n1 > cl:
                    s = usl[j, ig]
                    n2 = max(rns[j], cl)
                    dz = (n1 - n2) / s
                    z2 = z1 + dz
                    if z2 > zu:
                        done = True
                        z2 = zu
                        dz = z2 - z1
                        n2 = n1 - dz * s
                    n += dz * (n1 + n2) / 2
                    zk = z
                    while zk <= z2:
                        if k < 240:
                            ns[k, t] = n1 - (zk - z1) * s
                        k += 1
                        zk += step
                    z = zk
                else:
                    if n1 < cl:
                        j = 0
                        while j < 18 and rns[j] >= cl:
                            j += 1
                    done = True
                    z2 = zu
                    n2 = cl
                    n += n2 * (z2 - z1)
                    zk = z
                    while zk <= z2:
                        if k < 240:
                            ns[k, t] = n2
                        k += 1
                        zk += step
                    z = zk
                while n2 <= rns[j] and j < 17:
                    j += 1
                z1 = z2
                n1 = n2
        tot[t] = max(n, 0.0)
    return tot, ns


@numba.njit(cache=True)
def _zw_nl(core, fs):
    tv, ts, tl = 0.075, 0.005, 0.015
    dt = 1.0 / (fs * 24)
    p = (tv + tl) / (tv * ts)
    q = 1.0 / (ts * tv)
    l1 = -p / 2 + math.sqrt(p * p / 4 - q)
    l2 = -p / 2 - math.sqrt(p * p / 4 - q)
    den = tv * (l1 - l2)
    e1 = math.exp(l1 * dt)
    e2 = math.exp(l2 * dt)
    b0 = (e1 - e2) / den
    b1 = ((tv * l2 + 1) * e1 - (tv * l1 + 1) * e2) / den
    b2 = ((tv * l1 + 1) * e1 - (tv * l2 + 1) * e2) / den
    b3 = (tv * l1 + 1) * (tv * l2 + 1) * (e1 - e2) / den
    b4 = math.exp(-dt / tl)
    b5 = math.exp(-dt / tv)
    out = core.copy()
    nb, nt = core.shape
    for i in range(nb):
        uo = 0.0
        u2 = 0.0
        for t in range(nt):
            ui = core[i, t]
            last = t == nt - 1
            delta = 0.0 if last else (core[i, t + 1] - ui) / 24
            for m in range(1 if last else 24):
                if ui < uo:
                    if uo > u2:
                        n2 = uo * b0 - u2 * b1
                        no = uo * b2 - u2 * b3
                        if no < ui:
                            no = ui
                        if n2 > no:
                            n2 = no
                    else:
                        no = max(uo * b4, ui)
                        n2 = no
                else:
                    no = ui
                    if abs(ui - uo) < 1e-5 and no <= u2:
                        n2 = ui
                    else:
                        n2 = (u2 - ui) * b5 + ui
                uo = no
                u2 = n2
                if m == 0:
                    out[i, t] = uo
                ui += delta
    return out


@numba.njit(cache=True)
def _zw_lowpass(x, tau, fs):
    a1 = math.exp(-1.0 / (fs * 24 * tau))
    b0 = 1.0 - a1
    y1 = 0.0
    out = np.empty(x.size)
    for t in range(x.size):
        x0 = x[t]
        y1 = b0 * x0 + a1 * y1
        out[t] = y1
        if t < x.size - 1:
            xd = (x[t + 1] - x0) / 24
            for _ in range(1, 24):
                x0 += xd
                y1 = b0 * x0 + a1 * y1
    return out


def zwicker_from_bands(lt, field: str = "free") -> tuple[float, np.ndarray]:
    """Stationary loudness (sone) and specific loudness (sone/Bark on 240 steps of 0.1 Bark) from the 28
    third-octave levels 25 Hz - 12.5 kHz in dB SPL, ``field`` "free" or "diffuse"."""
    n, ns = _zw_slopes(_zw_core(lt, field), _ZW_ZUP, _ZW_RNS, _ZW_USL)
    return float(n[0]), ns[:, 0]


def zwicker_band_levels(y: np.ndarray, sr: int, pa: float = 1.0, skip: float | None = None) -> np.ndarray:
    """The 28 third-octave levels (dB SPL, 25 Hz - 12.5 kHz) of the ISO 532-1 filter bank (three 2nd-order
    sections per band at 48 kHz; other rates are resampled), mean square after ``skip`` s (None: 0.2 s when the
    sound lasts at least 0.4 s, else 0). A 1 kHz tone leaks 20 dB down into the 800 Hz and 1.25 kHz bands."""
    x = _zw_48k(y, sr, pa)
    if skip is None:
        skip = 0.2 if x.size >= 0.4 * ZW_SR else 0.0
    s = min(int(skip * ZW_SR), max(x.size - 1, 0))
    return np.array([10 * np.log10((np.mean(_zw_band(x, i)[s:] ** 2) + 1e-12) / 4e-10) for i in range(28)])


def zwicker_loudness(y: np.ndarray, sr: int, pa: float = 1.0, field: str = "free",
                     skip: float | None = None) -> tuple[float, np.ndarray]:
    """ISO 532-1 stationary loudness of ``y`` (samples times ``pa`` = pascals): (sone, specific loudness in
    sone/Bark per 0.1 Bark). A 1 kHz tone at 40 dB SPL reads 1 sone. ``skip``: see ``zwicker_band_levels``."""
    return zwicker_from_bands(zwicker_band_levels(y, sr, pa, skip), field)


def zwicker_loudness_tv(y: np.ndarray, sr: int, pa: float = 1.0, field: str = "free") -> tuple[np.ndarray, np.ndarray]:
    """ISO 532-1 time-varying loudness: (N(t) in sone every 2 ms, N'(z, t) in sone/Bark, shape (240, frames)).
    Band levels smoothed by three first-order low-passes (tau = 2/(3 fc), 2/3 ms above 1 kHz) at a 2 kHz
    frame rate, the diode-capacitor decay of the ear (5 / 15 / 75 ms) on the core loudness, then the total
    weighted 0.47 / 0.53 by 3.5 ms and 70 ms low-passes: a 10 ms burst reads about half a 100 ms one."""
    x = _zw_48k(y, sr, pa)
    n = x.size // 24
    if n < 2:
        return np.zeros(0), np.zeros((240, 0))
    lt = np.empty((28, n))
    for i, fc in enumerate(THIRD_OCTAVES):
        a = math.exp(-1.0 / (ZW_SR * 2.0 / (3.0 * min(fc, 1000.0))))
        v = _zw_band(x, i) ** 2
        for _ in range(3):
            v = lfilter([1.0 - a], [1.0, -a], v)
        lt[i] = 10 * np.log10((v[:n * 24:24] + 1e-12) / 4e-10)
    tot, ns = _zw_slopes(_zw_nl(_zw_core(lt, field), 2000.0), _ZW_ZUP, _ZW_RNS, _ZW_USL)
    tot = 0.47 * _zw_lowpass(tot, 3.5e-3, 2000.0) + 0.53 * _zw_lowpass(tot, 70e-3, 2000.0)
    return tot[::4], ns[:, ::4]


def loudness_level(n_sone) -> np.ndarray:
    """Loudness level in phon of a loudness in sone: 40 + 33.22 lg N from 1 sone, 40 (N + 0.0005)^0.35 below."""
    n = np.asarray(n_sone, float)
    return np.where(n >= 1.0, 40.0 + 33.22 * np.log10(np.maximum(n, 1.0)), 40.0 * (n + 0.0005) ** 0.35)


def zwicker_sharpness(ns: np.ndarray) -> float:
    """DIN 45692 sharpness in acum from a specific loudness pattern (``zwicker_loudness``'s second value)."""
    z = 0.1 * np.arange(1, ns.size + 1)
    g = np.where(z <= 15.8, 1.0, 0.15 * np.exp(0.42 * (z - 15.8)) + 0.85)
    n = ns.sum() * 0.1
    return float(0.11 * (ns * g * z).sum() * 0.1 / n) if n > 0 else 0.0


def loudness_percentiles(y: np.ndarray, sr: int, pa: float = 1.0, field: str = "free") -> dict:
    """Loudness exceeded 5 % and 10 % of the time (N5, N10, sone), the maximum and the mean of the ISO 532-1
    time-varying loudness (``zwicker_loudness_tv``, 2 ms steps)."""
    n = zwicker_loudness_tv(y, sr, pa, field)[0]
    if n.size == 0:
        n = np.zeros(1)
    return {"N5": float(np.percentile(n, 95)), "N10": float(np.percentile(n, 90)), "max": float(n.max()),
            "mean": float(np.mean(n))}


def binaural_loudness(y: np.ndarray, sr: int, pa: float = 1.0, field: str = "free") -> float:
    """Loudness (sone) of a stereo signal heard with both ears. ISO 532-1 loudness is already that of a diotic
    sound (a mono ``y`` reads ``zwicker_loudness``); one ear alone hears it ``r`` times softer, r = 2 at 20 phon
    falling to 1.4 at 80 phon, and the ears add as (NL^k + NR^k)^(1/k) with 2^(1/k) = r."""
    y = np.asarray(y, float)
    if y.ndim == 1:
        return zwicker_loudness(y, sr, pa, field)[0]
    n = np.array([zwicker_loudness(y[:, c], sr, pa, field)[0] for c in range(y.shape[1])])
    if n.max() <= 0:
        return 0.0
    r = float(np.interp(loudness_level(n.max()), BINAURAL_RATIO[0], BINAURAL_RATIO[1]))
    k = math.log(2.0) / math.log(r)
    return float(((n / r) ** k).sum() ** (1.0 / k))


BINAURAL_RATIO = ((20.0, 80.0), (2.0, 1.4))
DURATION_LOUDNESS_KNEE = 0.1


def duration_loudness_db(dur: float) -> float:
    """Loudness level drop (phon) of a short sound against the same sound held: 10 lg(dur / 100 ms) below
    100 ms (-7 at 20 ms, -10 at 10 ms, -20 at 1 ms), 0 from 100 ms."""
    return 10.0 * math.log10(min(max(float(dur), 1e-4), DURATION_LOUDNESS_KNEE) / DURATION_LOUDNESS_KNEE)


def yin(y: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 1000.0, threshold: float = 0.1,
        frame: float = 0.04, hop: float = 0.01) -> np.ndarray:
    """F0 track (Hz, 0 = unvoiced) by YIN: cumulative-mean-normalised difference, first dip under ``threshold``,
    parabolic interpolation of the lag."""
    x = _mono(np.asarray(y, float))
    w, h = int(frame * sr), int(hop * sr)
    lo, hi = max(2, int(sr / fmax)), min(int(sr / fmin), w - 1)
    out = []
    for s in range(0, max(1, x.size - w - hi), h):
        a = x[s:s + w]
        d = np.array([np.sum((a[:w - hi] - a[k:k + w - hi]) ** 2) for k in range(hi + 1)])
        c = d[1:] * np.arange(1, hi + 1) / (np.cumsum(d[1:]) + 1e-30)
        c = np.concatenate(([1.0], c))
        below = np.flatnonzero(c[lo:hi] < threshold)
        if below.size == 0:
            out.append(0.0)
            continue
        k = lo + below[0]
        while k + 1 < hi and c[k + 1] < c[k]:
            k += 1
        den = c[k - 1] - 2 * c[k] + c[k + 1]
        out.append(sr / (k + (0.5 * (c[k - 1] - c[k + 1]) / den if den else 0.0)))
    return np.array(out)


def bark_flux(y: np.ndarray, sr: int, win: float = 0.0232, hop: float = 0.0058) -> np.ndarray:
    """Positive spectral flux on 24 critical bands (Bark), per frame: sum of the rises of the log band energies."""
    x = _mono(np.asarray(y, float))
    m, r = int(round(win * sr)), int(round(hop * sr))
    if x.size < m + r:
        return np.zeros(0)
    a = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(x, m)[::r] * np.hanning(m), axis=1)) ** 2
    z = np.clip(hz_to_bark(np.fft.rfftfreq(m, 1 / sr)).astype(int), 0, 23)
    e = np.stack([a[:, z == b].sum(axis=1) for b in range(24)], axis=1)
    return np.maximum(np.diff(10 * np.log10(e + 1e-12), axis=0), 0.0).sum(axis=1)


def mfcc(y: np.ndarray, sr: int, n: int = 13, n_mels: int = 40) -> np.ndarray:
    """Mel-frequency cepstral coefficients per frame: DCT-II (orthonormal) of the log-mel frames."""
    from scipy.fft import dct
    from .calibrate import logmel_frames
    return dct(logmel_frames(y, sr, n_mels=n_mels), type=2, norm="ortho", axis=1)[:, :n]


def jitter_shimmer(periods, amps) -> dict:
    """Local jitter (%) and shimmer (%, dB) from successive periods and peak amplitudes, as Praat defines them:
    mean |T_i - T_i+1| / mean T, mean |A_i - A_i+1| / mean A, mean |20 log10(A_i+1 / A_i)|."""
    t, a = np.asarray(periods, float), np.asarray(amps, float)
    return {"jitter": 100 * float(np.mean(np.abs(np.diff(t))) / np.mean(t)),
            "shimmer": 100 * float(np.mean(np.abs(np.diff(a))) / np.mean(a)),
            "shimmer_db": float(np.mean(np.abs(20 * np.log10(a[1:] / a[:-1]))))}


def tempo_from_loudness(y: np.ndarray, sr: int, lo: float = 40.0, hi: float = 240.0) -> float:
    """Beats per minute from the autocorrelation of the onset strength (rises of the K-weighted level, 10 ms
    frames), strongest peak between ``lo`` and ``hi`` bpm."""
    x = _k_weight(_mono(np.asarray(y, float)), sr)
    h = int(0.01 * sr)
    e = 10 * np.log10(np.add.reduceat(x * x, np.arange(0, x.size - h, h)) / h + 1e-12)
    o = np.maximum(np.diff(e), 0.0)
    o -= o.mean()
    r = np.correlate(o, o, "full")[o.size - 1:]
    k0, k1 = int(6000 / hi), min(int(6000 / lo), r.size - 1)
    return 6000.0 / (k0 + int(np.argmax(r[k0:k1 + 1])))


# ------------------------------------------------------------------------------------------
# Voice perturbation, file checks, chaos, particle counts, masking, glottal shape, IR measurement
# ------------------------------------------------------------------------------------------

def jitter_measures(periods) -> dict:
    """Praat / MDVP period perturbation: local (%), local absolute (us), RAP (%, each period against the mean of
    itself and its 2 neighbours), PPQ5 (%, 5 neighbours), DDP (%, mean |difference of consecutive differences| /
    mean T = 3 RAP)."""
    t = np.asarray(periods, float)
    mt = float(np.mean(t))
    k3 = np.convolve(t, np.ones(3) / 3.0, "valid")
    k5 = np.convolve(t, np.ones(5) / 5.0, "valid")
    return {"local": 100.0 * float(np.mean(np.abs(np.diff(t)))) / mt,
            "local_abs_us": 1e6 * float(np.mean(np.abs(np.diff(t)))),
            "rap": 100.0 * float(np.mean(np.abs(t[1:-1] - k3))) / mt,
            "ppq5": 100.0 * float(np.mean(np.abs(t[2:-2] - k5))) / mt,
            "ddp": 100.0 * float(np.mean(np.abs(np.diff(t, 2)))) / mt}


def active_bits(y: np.ndarray, bits: int = 16) -> int:
    """Bits really used by a file: log2 of the number of distinct quantised codes, rounded up."""
    q = np.round(np.asarray(y, float).ravel() * (2 ** (bits - 1) - 1)).astype(np.int64)
    return int(math.ceil(math.log2(max(len(np.unique(q)), 1)))) if q.size else 0


def polarity(y: np.ndarray, frac: float = 0.5) -> int:
    """+1 / -1: sign of the first excursion reaching ``frac`` of the peak (a blast or a kick starts positive)."""
    x = _mono(np.asarray(y, float))
    i = int(np.argmax(np.abs(x) >= frac * np.max(np.abs(x))))
    return 1 if x[i] >= 0 else -1


def dc_offset_db(y: np.ndarray) -> float:
    """DC offset in dBFS (mean of the file)."""
    return 20.0 * math.log10(abs(float(np.mean(_mono(np.asarray(y, float))))) + 1e-12)


def null_test_db(a: np.ndarray, b: np.ndarray, sr: int, band=(200.0, 5000.0)) -> float:
    """Level of a - b re a (dB) inside ``band`` Hz: two renders that should match null below about -60 dB."""
    from scipy.signal import butter, sosfiltfilt
    n = min(len(a), len(b))
    sos = butter(4, [band[0], min(band[1], 0.45 * sr)], "bandpass", fs=sr, output="sos")
    x, d = sosfiltfilt(sos, _mono(np.asarray(a, float))[:n]), sosfiltfilt(sos, (_mono(np.asarray(a, float)) [:n] - _mono(np.asarray(b, float))[:n]))
    return 10.0 * math.log10((float(np.mean(d * d)) + 1e-30) / (float(np.mean(x * x)) + 1e-30))


def _embed(x, dim, tau):
    n = x.size - (dim - 1) * tau
    return np.stack([x[i * tau:i * tau + n] for i in range(dim)], axis=1)


def correlation_dimension(x, dim: int = 6, tau: int = 1, n_max: int = 1500, eps=None) -> float:
    """Grassberger-Procaccia: slope of log C(eps) against log eps, C the share of delay-embedded point pairs closer
    than eps, over the scaling range between the 5th and 50th percentile of distances."""
    X = _embed(np.asarray(x, float), dim, tau)[:n_max]
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))[np.triu_indices(len(X), 1)]
    d = d[d > 0]
    eps = np.geomspace(np.percentile(d, 5), np.percentile(d, 50), 12) if eps is None else np.asarray(eps, float)
    C = np.array([np.mean(d < e) for e in eps])
    return float(np.polyfit(np.log(eps), np.log(C), 1)[0])


def lyapunov_exponent(x, dim: int = 6, tau: int = 1, steps: int = 10, n_max: int = 1500, min_sep: int = 10) -> float:
    """Largest Lyapunov exponent per sample (Rosenstein): mean log divergence of each embedded point from its
    nearest neighbour (at least ``min_sep`` samples apart), slope over the first ``steps`` samples."""
    X = _embed(np.asarray(x, float), dim, tau)[:n_max]
    n = len(X) - steps
    D = np.sqrt(((X[:n, None, :] - X[None, :n, :]) ** 2).sum(-1))
    idx = np.arange(n)
    D[np.abs(idx[:, None] - idx[None, :]) < min_sep] = np.inf
    nn = np.argmin(D, axis=1)
    div = [np.mean(np.log(np.linalg.norm(X[idx + k] - X[nn + k], axis=1) + 1e-12)) for k in range(steps)]
    return float(np.polyfit(np.arange(steps), div, 1)[0])


# true particle count -> mean estimate, sd (1050 synthetic walks)
PARTICLE_CALIB = ((2, 2.04, 0.82), (4, 3.06, 1.31), (8, 5.33, None), (16, 13.0, None), (64, 99.1, None),
                  (256, 396.0, None), (1024, 699.0, None))
_D4 = np.array([1 + math.sqrt(3), 3 + math.sqrt(3), 3 - math.sqrt(3), 1 - math.sqrt(3)]) / (4 * math.sqrt(2))


def particle_count(y: np.ndarray, sr: int, block: int = 1024, calibrate: bool = True) -> float:
    """Number of grains per event of a granular sound (gravel, footsteps): LPC-whitened signal at 22.05 kHz,
    Daubechies-4 detail band 5.5-11 kHz rectified, peaks over a threshold between the block's mean and max
    counted in the blocks that have any, Nest = 2^(avgPeaks / blocks / 10); ``calibrate`` maps Nest back through
    ``PARTICLE_CALIB``."""
    from scipy.signal import resample_poly
    from .physical.analysis import _levinson
    x = _mono(np.asarray(y, float))
    x = resample_poly(x, 22050, sr) if sr != 22050 else x
    r = np.correlate(x, x, "full")[x.size - 1:x.size + 13]
    a = np.asarray(_levinson(r, 12)[0], float)
    e = lfilter(a, [1.0], x)
    g = _D4[::-1] * np.array([1, -1, 1, -1])
    d = np.abs(np.convolve(e, g, "same")[::2])
    nb = max(1, d.size // (block // 2))
    counts = []
    for b in np.array_split(d, nb):
        thr = b.mean() + 0.5 * (b.max() - b.mean())
        p = np.flatnonzero((b[1:-1] > thr) & (b[1:-1] >= b[:-2]) & (b[1:-1] > b[2:]))
        if p.size:
            counts.append(p.size)
    if not counts:
        return 0.0
    nest = 2.0 ** (float(np.mean(counts)) / 10.0)
    if not calibrate:
        return nest
    tr, est = np.log([c[0] for c in PARTICLE_CALIB]), np.log([c[1] for c in PARTICLE_CALIB])
    return float(np.exp(np.interp(math.log(nest), est, tr)))


POST_MASK_FAST_S, POST_MASK_SLOW_S, POST_MASK_HOLD_S, POST_MASK_LONG_S = 0.005, 0.040, 0.005, 0.1
PRE_MASK_S = 0.020


def temporal_masking(n_t, dt: float) -> np.ndarray:
    """Loudness after temporal masking: a peak hold that stays flat ``POST_MASK_HOLD_S`` after the masker
    stops, then decays with ``POST_MASK_SLOW_S`` after a masker of ``POST_MASK_LONG_S`` or more (back to
    threshold in ~200 ms) or with the steeper ``POST_MASK_FAST_S`` after a shorter one; pre-masking spreads
    each onset ``PRE_MASK_S`` earlier."""
    n = np.asarray(n_t, float)
    out = np.empty_like(n)
    held, on, off = 0.0, 0.0, 0.0
    for i, v in enumerate(n):
        if v >= held:
            if held <= 0.0 or off > 0.0:
                on = 0.0
            held, off = v, 0.0
            on += dt
        else:
            off += dt
            if off > POST_MASK_HOLD_S:
                tau = POST_MASK_SLOW_S if on >= POST_MASK_LONG_S else POST_MASK_FAST_S
                held = max(v, held * math.exp(-dt / tau))
        out[i] = held
    k = int(round(PRE_MASK_S / dt))
    if k > 0:
        pad = np.concatenate([out, np.zeros(k)])
        out = np.max(np.stack([pad[j:j + n.size] * (1.0 - j / (k + 1)) for j in range(k + 1)]), axis=0)
    return out


def masked_loudness_percentiles(y: np.ndarray, sr: int, pa: float = 1.0, win: float = 0.01,
                                field: str = "free") -> dict:
    """``loudness_percentiles`` on 10 ms frames after ``temporal_masking``."""
    x = _mono(np.asarray(y, float))
    m = max(int(win * sr), 64)
    hop = m // 2
    n = temporal_masking([zwicker_loudness(x[i:i + m], sr, pa, field)[0]
                          for i in range(0, max(1, x.size - m + 1), hop)], hop / sr)
    return {"N5": float(np.percentile(n, 95)), "N10": float(np.percentile(n, 90)), "mean": float(np.mean(n))}


def threshold_in_quiet(f) -> np.ndarray:
    """Absolute threshold (dB SPL): 3.64 f^-0.8 - 6.5 exp(-0.6 (f - 3.3)^2) + 1e-3 f^4, f in kHz."""
    k = np.maximum(np.asarray(f, float), 20.0) / 1000.0
    return 3.64 * k ** -0.8 - 6.5 * np.exp(-0.6 * (k - 3.3) ** 2) + 1e-3 * k ** 4


def audible_partials(freqs, levels_db, ref_db: float = 96.0, offset_db: float = 10.0, spread: str = "terhardt") -> np.ndarray:
    """Mask of the partials heard: levels normalised so the loudest is ``ref_db`` SPL, above the threshold in
    quiet and above every other partial's spreading lowered by ``offset_db``. ``spread`` "terhardt": 27 dB/Bark
    below, -24 - 230/f + 0.2 L dB/Bark above; "nam" (alias audibility): (-27 + 0.37 max(L - 40, 0) theta(dz)) |dz|."""
    f = np.asarray(freqs, float)
    L = np.asarray(levels_db, float)
    L = L - L.max() + ref_db
    z = hz_to_bark(f)
    ok = L > threshold_in_quiet(f)
    for j in range(f.size):
        dz = z - z[j]
        if spread == "nam":
            sp = (-27.0 + 0.37 * max(L[j] - 40.0, 0.0) * (dz > 0)) * np.abs(dz)
        else:
            sp = np.where(dz < 0, 27.0 * dz, (-24.0 - 230.0 / f[j] + 0.2 * L[j]) * dz)
        m = L[j] - offset_db + sp
        m[j] = -np.inf
        ok &= L > m
    return ok


def _harm_phases(y, sr, f0, H):
    n = y.size
    t = np.arange(n) / sr
    w = np.hanning(n)
    return np.array([np.angle(np.sum(w * y * np.exp(-2j * np.pi * k * f0 * t))) for k in range(1, H + 1)])


def estimate_rd(y: np.ndarray, sr: int, f0: float, H: int = 8, grid=None) -> float:
    """Rd of the glottal pulse by minimum mean squared phase difference: the second difference of the harmonic
    phases across k cancels the time shift; its wrapped distance to the LF model's is minimised over Rd."""
    from .physical.voice import lf_waveform
    x = _mono(np.asarray(y, float))
    obs = np.diff(_harm_phases(x, sr, f0, H), 2)
    grid = np.linspace(0.3, 2.7, 49) if grid is None else np.asarray(grid, float)
    err = []
    for rd in grid:
        g = lf_waveform(float(rd), 2048)
        mod = np.diff(np.angle(np.fft.rfft(g)[1:H + 1]), 2)
        err.append(np.mean(np.angle(np.exp(1j * (obs - mod))) ** 2))
    return float(grid[int(np.argmin(err))])


def exp_sweep(f1: float, f2: float, dur: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Exponential sine sweep and its inverse filter (time-reversed, -6 dB/octave), for ``measure_ir``."""
    t = np.arange(int(dur * sr)) / sr
    R = math.log(f2 / f1)
    s = np.sin(2 * np.pi * f1 * dur / R * (np.exp(t * R / dur) - 1.0))
    inv = s[::-1] * np.exp(-t * R / dur)
    return s, inv


def measure_ir(system, f1: float = 20.0, f2: float = 20000.0, dur: float = 2.0, sr: int = 44100,
               length: float = 1.0) -> np.ndarray:
    """Impulse response of ``system`` (callable on a mono array) by exponential sweep and deconvolution."""
    from scipy.signal import fftconvolve
    s, inv = exp_sweep(f1, min(f2, 0.45 * sr), dur, sr)
    ref = fftconvolve(s, inv)
    k = int(np.argmax(np.abs(ref)))
    h = fftconvolve(np.asarray(system(s), float), inv) / np.abs(ref[k])
    return h[k:k + int(length * sr)]


def ddsp_like(n: int = 8, dur: float = 1.0, sr: int = 16000, seed: int = 0) -> list[tuple[np.ndarray, float]]:
    """Synthetic test set: (signal, f0) with 100 harmonics of random decaying amplitudes and 65 bands of filtered
    noise, pitches drawn in MIDI 24-84."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    edges = np.geomspace(20.0, 0.49 * sr, 66)
    out = []
    for _ in range(n):
        f0 = 440.0 * 2 ** ((rng.integers(24, 85) - 69) / 12)
        k = np.arange(1, 101)
        amp = rng.uniform(0, 1, 100) * k ** -rng.uniform(0.8, 2.0) * (k * f0 < 0.5 * sr)
        h = (amp[:, None] * np.sin(2 * np.pi * f0 * k[:, None] * t)).sum(0)
        spec = np.fft.rfft(rng.standard_normal(t.size))
        f = np.fft.rfftfreq(t.size, 1 / sr)
        gains = rng.uniform(0, 1, 65) * 0.02
        spec *= gains[np.clip(np.searchsorted(edges, f) - 1, 0, 64)]
        y = h + np.fft.irfft(spec, t.size)
        out.append((y / (np.max(np.abs(y)) + 1e-12), float(f0)))
    return out


def phase_track(y: np.ndarray, sr: int, f0: float, frame: float = 0.005) -> np.ndarray:
    """Phase of the fundamental per ``frame`` s: lowpass at 1.5 f0, analytic signal, phase relative to a sinusoid
    at f0, one value per frame (circular mean)."""
    from scipy.signal import butter, hilbert, sosfiltfilt
    x = sosfiltfilt(butter(4, 1.5 * f0, fs=sr, output="sos"), _mono(np.asarray(y, float)))
    ph = np.angle(hilbert(x) * np.exp(-2j * np.pi * f0 * np.arange(x.size) / sr))
    m = max(1, int(frame * sr))
    return np.array([np.angle(np.mean(np.exp(1j * ph[i:i + m]))) for i in range(0, x.size - m + 1, m)])


def wichern_features(y: np.ndarray, sr: int, frame: int = 1024) -> dict:
    """Six frame-wise descriptors (medians): loudness (dB RMS), spectral centroid (Hz), spectral sparsity
    (max / sum of the magnitude spectrum), harmonicity (peak of the normalised autocorrelation, 50-1000 Hz),
    temporal sparsity (max / sum of the frame energies over 10 frames), transient index (spectral flux of the
    mel-like log spectrum)."""
    x = _mono(np.asarray(y, float))
    fr = [x[i:i + frame] * np.hanning(frame) for i in range(0, max(1, x.size - frame + 1), frame // 2)]
    f = np.fft.rfftfreq(frame, 1 / sr)
    S = np.array([np.abs(np.fft.rfft(v, frame)) for v in fr])
    E = np.array([float(np.sum(v * v)) for v in fr])
    lo, hi = int(sr / 1000), int(sr / 50)
    harm = []
    for v in fr:
        r = np.correlate(v, v, "full")[frame - 1:]
        harm.append(float(np.max(r[lo:min(hi, frame - 1)]) / (r[0] + 1e-12)) if r[0] > 0 else 0.0)
    L = np.log(S + 1e-9)
    return {"loudness": float(np.median(10 * np.log10(E / frame + 1e-12))),
            "centroid": float(np.median((S * f).sum(1) / (S.sum(1) + 1e-12))),
            "spectral_sparsity": float(np.median(S.max(1) / (S.sum(1) + 1e-12))),
            "harmonicity": float(np.median(harm)),
            "temporal_sparsity": float(np.median([E[i:i + 10].max() / (E[i:i + 10].sum() + 1e-12)
                                                  for i in range(0, max(1, E.size - 9))])),
            "transient_index": float(np.median(np.sqrt((np.maximum(np.diff(L, axis=0), 0) ** 2).sum(1))))
            if len(L) > 1 else 0.0}


def wichern_distance(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    """Distance between two sounds on the six ``wichern_features``, each as a relative (log) difference."""
    fa, fb = wichern_features(a, sr), wichern_features(b, sr)
    d = [(fa[k] - fb[k]) / 20.0 if k == "loudness" else math.log((abs(fa[k]) + 1e-9) / (abs(fb[k]) + 1e-9))
         for k in fa]
    return float(np.sqrt(np.mean(np.square(d))))


def optimize_params(render, target: np.ndarray, sr: int, bounds: dict, metric=None, particles: int = 12,
                    iters: int = 15, seed: int = 0) -> dict:
    """Particle swarm over ``bounds`` {name: (lo, hi)} minimising ``metric(render(**p), target, sr)`` (default
    ``wichern_distance``); renders and target are loudness-matched first."""
    metric = metric or wichern_distance
    rng = np.random.default_rng(seed)
    keys = list(bounds)
    lo, hi = np.array([bounds[k][0] for k in keys], float), np.array([bounds[k][1] for k in keys], float)
    tl = loudness(np.asarray(target, float), sr)

    def cost(v):
        y = np.asarray(render(**dict(zip(keys, v))), float)
        y = y * 10 ** ((tl - loudness(y, sr)) / 20.0)
        return metric(y, target, sr)
    X = rng.uniform(lo, hi, (particles, len(keys)))
    V = np.zeros_like(X)
    P, pc = X.copy(), np.array([cost(v) for v in X])
    g = P[np.argmin(pc)].copy()
    for _ in range(iters):
        r1, r2 = rng.uniform(size=X.shape), rng.uniform(size=X.shape)
        V = 0.72 * V + 1.49 * r1 * (P - X) + 1.49 * r2 * (g - X)
        X = np.clip(X + V, lo, hi)
        c = np.array([cost(v) for v in X])
        better = c < pc
        P[better], pc[better] = X[better], c[better]
        g = P[np.argmin(pc)].copy()
    return dict(zip(keys, (float(v) for v in g)))


QC_LOUDNESS_OFFSET, DEFAULT_QC_EAR = {"impact": 6.0}, -6.0


def wav_check_level(y: np.ndarray, sr: int, category: str | None = None, ceiling: float = DEFAULT_QC_EAR) -> list[str]:
    """Ear-weighted level check: ``ear_loudness`` under ``ceiling`` dB, raised by ``QC_LOUDNESS_OFFSET`` for the
    classes allowed to be louder (impacts +6 dB)."""
    lim = ceiling + QC_LOUDNESS_OFFSET.get(category or "", 0.0)
    e = ear_loudness(y, sr)
    return [] if e <= lim else [f"ear level {e:.1f} dB over {lim:.1f}"]


# ------------------------------------------------------------------------------------------
# Wow and flutter, echo density, pitch, spectral moments
# ------------------------------------------------------------------------------------------

WF_TONE = 3150.0
WF_BAND = (312.9, 5945.0)
WF_LP = 1565.0
WF_WEIGHT = ((0.2, 0.0296), (0.8, 0.5), (4.0, 1.0), (20.0, 0.508))
WF_BANDS = {"drift": (0.0, 0.5), "wow": (0.5, 6.0), "flutter": (6.0, 100.0), "scrape": (100.0, WF_LP)}


def wf_weight(f) -> np.ndarray:
    """Wow-and-flutter weighting, log-log through the tabulated points, the end slopes continued."""
    f = np.maximum(np.asarray(f, float), 1e-6)
    lf, lw = np.log([p[0] for p in WF_WEIGHT]), np.log([p[1] for p in WF_WEIGHT])
    x = np.log(f)
    w = np.interp(x, lf, lw)
    lo, hi = (lw[1] - lw[0]) / (lf[1] - lf[0]), (lw[-1] - lw[-2]) / (lf[-1] - lf[-2])
    w = np.where(x < lf[0], lw[0] + lo * (x - lf[0]), w)
    w = np.where(x > lf[-1], lw[-1] + hi * (x - lf[-1]), w)
    return np.exp(w)


def wow_flutter(y: np.ndarray, sr: int, tone: float = WF_TONE) -> dict:
    """Wow and flutter of a recorded test tone: band-pass, FM demodulation (analytic phase), low-pass, then
    the weighted 2-sigma peak deviation in % of the tone, and the unweighted rms in % per band."""
    from scipy.signal import butter, hilbert, sosfiltfilt
    x = _mono(np.asarray(y, float))
    x = sosfiltfilt(butter(4, [WF_BAND[0], min(WF_BAND[1], 0.45 * sr)], "bandpass", fs=sr, output="sos"), x)
    fi = np.diff(np.unwrap(np.angle(hilbert(x)))) * sr / (2 * np.pi)
    fi = sosfiltfilt(butter(4, min(WF_LP, 0.45 * sr), fs=sr, output="sos"), fi)
    m = int(0.1 * fi.size)
    d = (fi[m:fi.size - m] - tone) / tone
    D = np.fft.rfft(d - d.mean())
    f = np.fft.rfftfreq(d.size, 1 / sr)
    w = np.fft.irfft(D * wf_weight(f), d.size)
    out = {"weighted_peak": 200.0 * float(w.std())}
    hw = np.hanning(d.size)
    Dw = np.fft.rfft((d - d.mean()) * hw)                       # windowed: a wow line does not leak into flutter
    for k, (a, b) in WF_BANDS.items():
        out[k] = 100.0 * float(np.sqrt(2 * np.sum(np.abs(Dw[(f >= a) & (f < b)]) ** 2) / (d.size * np.sum(hw * hw))))
    out["offset"] = 100.0 * float(d.mean())
    return out


def normalized_echo_density(ir: np.ndarray, sr: int, win: float = 0.020) -> tuple[np.ndarray, np.ndarray]:
    """Normalised echo density: the Hann-weighted share of samples beyond one standard deviation in a ``win``
    window, divided by the Gaussian value erfc(1/sqrt 2) = 0.3173. 1 = a diffuse late field."""
    x = _mono(np.asarray(ir, float))
    n = max(3, int(round(win * sr)) | 1)
    w = np.hanning(n)
    w /= w.sum()
    hop = max(1, n // 4)
    if x.size < n:
        x = np.pad(x, (0, n - x.size))
    fr = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    sd = np.sqrt((fr ** 2 * w).sum(axis=1, keepdims=True))
    eta = (w * (np.abs(fr) > sd)).sum(axis=1) / math.erfc(1 / math.sqrt(2))
    return (np.arange(eta.size) * hop + n // 2) / sr, eta


def mixing_time(ir: np.ndarray, sr: int, win: float = 0.020) -> float:
    """First time (s) the normalised echo density reaches 1; nan if never."""
    t, eta = normalized_echo_density(ir, sr, win)
    k = np.flatnonzero(eta >= 1.0)
    return float(t[k[0]]) if k.size else float("nan")


def ratio_pitch_mel(f) -> np.ndarray:
    """Ratio pitch in mel = 100 x critical-band rate (Zwicker's arctangent form): 125 Hz ~ 125 mel,
    1300 Hz ~ 1050 mel, 8 kHz ~ 2100 mel."""
    k = np.asarray(f, float) / 1000.0
    return 100.0 * (13.0 * np.arctan(0.76 * k) + 3.5 * np.arctan((k / 7.5) ** 2))


VP_MAX_HARM = 20
VP_MAX_FREQ = 5000.0
VP_DOMINANCE = 600.0


def virtual_pitch(freqs, levels_db, tol: float = 0.03) -> float:
    """Virtual pitch of a set of partials by subharmonic coincidence: every audible partial below 5 kHz
    votes for its subharmonics f/m (m <= 20), weighted by its level above threshold and by its distance
    (in octaves) from the 600 Hz dominance region; candidates within ``tol`` pool their votes. Hz."""
    f = np.asarray(freqs, float)
    L = np.asarray(levels_db, float)
    ok = (f < VP_MAX_FREQ) & (f > 20.0)
    f, L = f[ok], L[ok]
    if f.size == 0:
        return float("nan")
    L = L - L.max() + 60.0
    wt = np.maximum(L - threshold_in_quiet(f), 0.0) * np.exp(-np.log2(f / VP_DOMINANCE) ** 2 / 2.0)
    cand = np.array([fi / m for fi in f for m in range(1, VP_MAX_HARM + 1) if fi / m >= 20.0])
    best, score = float("nan"), -1.0
    for c in cand:
        h = f / c
        s = float(np.sum(wt * (np.abs(h - np.round(h)) < tol * np.round(h)) * (np.round(h) <= VP_MAX_HARM)))
        if s > score + 1e-9 or (abs(s - score) <= 1e-9 and c > best):
            best, score = float(c), s
    return best


def _frames_spectra(y, sr, win=0.0232, hop=0.0058):
    x = _mono(np.asarray(y, float))
    M, R = int(round(win * sr)), int(round(hop * sr))
    if x.size < M:
        x = np.pad(x, (0, M - x.size))
    fr = np.lib.stride_tricks.sliding_window_view(x, M)[::R] * np.hamming(M)
    return np.fft.rfftfreq(M, 1 / sr), np.abs(np.fft.rfft(fr, axis=1))


def spectral_descriptors(y: np.ndarray, sr: int) -> dict:
    """Median over STFT frames (Hamming 23.2 ms, hop 5.8 ms, the frames of ``spectral_flux``) of the spectral
    centroid, spread (Hz), skewness and kurtosis of the magnitude spectrum; silent frames are left out."""
    f, A = _frames_spectra(y, sr)
    tot = A.sum(axis=1)
    A, tot = A[tot > 1e-9 * (tot.max() + 1e-30)], tot[tot > 1e-9 * (tot.max() + 1e-30)]
    if A.shape[0] == 0:
        return {"centroid": float("nan"), "spread": float("nan"), "skewness": float("nan"), "kurtosis": float("nan")}
    p = A / tot[:, None]
    c = p @ f
    d = f[None, :] - c[:, None]
    s = np.sqrt((p * d ** 2).sum(axis=1))
    sk = (p * d ** 3).sum(axis=1) / (s ** 3 + 1e-30)
    ku = (p * d ** 4).sum(axis=1) / (s ** 4 + 1e-30)
    return {"centroid": float(np.median(c)), "spread": float(np.median(s)), "skewness": float(np.median(sk)),
            "kurtosis": float(np.median(ku))}


def hnr_ma(y: np.ndarray, sr: int, f0: float, H: int = 10) -> float:
    """HNR by spectral moving average: the dB spectrum minus its moving average over one harmonic spacing,
    read at the first ``H`` harmonic peaks, averaged (dB)."""
    x = _mono(np.asarray(y, float))
    S = 20 * np.log10(np.abs(np.fft.rfft(x * np.hanning(x.size))) + 1e-12)
    df = sr / x.size
    k = max(3, int(round(f0 / df)) | 1)
    ma = np.convolve(S, np.ones(k) / k, mode="same")
    vals = []
    for h in range(1, H + 1):
        i = int(round(h * f0 / df))
        if i + 2 >= S.size:
            break
        j = i - 2 + int(np.argmax(S[i - 2:i + 3]))
        vals.append(S[j] - ma[j])
    return float(np.mean(vals)) if vals else float("nan")


TAIL_FLOOR_DB = -60.0


def exposed_tail(y: np.ndarray, sr: int, floor_db: float = TAIL_FLOOR_DB) -> dict:
    """Whether a sound's tail is cut: {"tail_s": time from the peak until the 10 ms rms falls ``floor_db``
    under its maximum (or the end), "end_db": level of the last 10 ms re that maximum, "exposed": the last
    10 ms are still above ``floor_db`` (an audible truncation)}."""
    x = _mono(np.asarray(y, float))
    n = max(1, int(0.01 * sr))
    m = x.size // n
    if m == 0:
        return {"tail_s": 0.0, "end_db": 0.0, "exposed": True}
    e = 10 * np.log10(np.mean(x[:m * n].reshape(m, n) ** 2, axis=1) + 1e-30)
    e -= e.max()
    k = int(np.argmax(e))
    below = np.flatnonzero(e[k:] <= floor_db)
    tail = (below[0] if below.size else m - k) * n / sr
    return {"tail_s": float(tail), "end_db": float(e[-1]), "exposed": bool(e[-1] > floor_db)}


OVER_RUN = 3


def count_overs(y: np.ndarray, run: int = OVER_RUN, level: float = 1.0) -> int:
    """Number of OVERs: runs of at least ``run`` contiguous samples at or above ``level`` (full scale) in magnitude
    (3 by default; 4, 5 or 6 on stricter meters)."""
    x = np.abs(_mono(np.asarray(y, float))) >= level - 1e-9
    if not x.any():
        return 0
    d = np.diff(np.r_[0, x.astype(np.int8), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return int(np.sum(ends - starts >= run))


def temporal_descriptors(y: np.ndarray, sr: int) -> dict:
    """Timbre-toolbox style temporal descriptors of the 10 ms rms energy envelope e: attack slope (mean dB/s
    from 10 % to 90 % of the peak amplitude), decay slope (dB/s of a line fitted from the peak to -40 dB),
    temporal centroid (s, energy-weighted mean time), effective duration (s above 40 % of the peak)."""
    x = _mono(np.asarray(y, float))
    n = max(1, int(0.01 * sr))
    m = x.size // n
    if m < 2:
        return {"attack_slope": float("nan"), "decay_slope": float("nan"), "temporal_centroid": float("nan"),
                "effective_duration": 0.0}
    a = np.sqrt(np.mean(x[:m * n].reshape(m, n) ** 2, axis=1))
    t = (np.arange(m) + 0.5) * n / sr
    k = int(np.argmax(a))
    pk = a[k] + 1e-30
    i10 = int(np.argmax(a[:k + 1] >= 0.1 * pk))
    i90 = int(np.argmax(a[:k + 1] >= 0.9 * pk))
    att = 20 * math.log10(9.0) / max(t[i90] - t[i10], n / sr)
    db = 20 * np.log10(a[k:] / pk + 1e-12)
    j = np.flatnonzero(db <= -40.0)
    stop = j[0] + 1 if j.size else db.size
    dec = float(np.polyfit(t[k:k + stop], db[:stop], 1)[0]) if stop >= 2 else float("nan")
    e = a ** 2
    return {"attack_slope": float(att), "decay_slope": dec, "temporal_centroid": float((t * e).sum() / e.sum()),
            "effective_duration": float(np.sum(a >= 0.4 * pk) * n / sr)}


TWM_WIDTH = 0.1


def harmonic_coherence(freqs, amps, f0: float, width: float = TWM_WIDTH) -> float:
    """Share of the peak amplitude lying on harmonics of f0: sum a_k exp(-(d_k / f0 / width)^2) / sum a_k,
    d_k = distance of peak k to the nearest harmonic."""
    f, a = np.asarray(freqs, float), np.asarray(amps, float)
    d = f - f0 * np.maximum(np.round(f / f0), 1.0)
    return float(np.sum(a * np.exp(-(d / f0 / width) ** 2)) / (np.sum(a) + 1e-30))


def twm_f0(freqs, amps, candidates, width: float = TWM_WIDTH) -> float:
    """Two-way mismatch f0: the candidate maximising measured-to-predicted coherence times the share of its
    predicted harmonics (below the highest peak) that a peak lies near (predicted-to-measured)."""
    f = np.asarray(freqs, float)
    best, score = float("nan"), -1.0
    for c in np.asarray(candidates, float):
        h = c * np.arange(1, max(1, int(f.max() / c)) + 1)
        cover = np.mean(np.exp(-(np.min(np.abs(h[:, None] - f[None, :]), axis=1) / c / width) ** 2))
        s = harmonic_coherence(f, amps, c, width) * cover
        if s > score:
            best, score = float(c), s
    return best


PS_PEAK_HZ = 1500.0
PS_NOISE = 1.0 / 7.0


def pitch_strength(f: float, dur: float = 0.3, level_db: float = 60.0, kind: str = "tone") -> float:
    """Relative pitch strength (1 = a 300 ms, 60 dB pure tone at 1.5 kHz): log-Gaussian in frequency, a third
    at 125 Hz and at 9 kHz; grows with log duration up to 300 ms; 2.5 times stronger at 80 dB than at 20 dB
    (+12.5 % per 10 dB); noise bands 1/7 of it."""
    lf = math.log2(max(f, 1.0) / PS_PEAK_HZ)
    lo, hi = math.log2(125.0 / PS_PEAK_HZ), math.log2(9000.0 / PS_PEAK_HZ)
    s = lo if lf < 0 else hi
    g = math.exp(math.log(1 / 3) * (lf / s) ** 2)
    t = min(1.0, max(0.0, math.log(max(dur, 1e-3) / 1e-3) / math.log(300.0)))
    return g * t * max(0.0, 1.0 + 0.0125 * (level_db - 60.0)) * (PS_NOISE if kind == "noise" else 1.0)


def spectral_content_index(y: np.ndarray, sr: int, f0: float) -> float:
    """SCI = mean spectral frequency / fundamental: sum f E / sum E over the power spectrum, divided by f0."""
    x = _mono(np.asarray(y, float))
    P = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    f = np.fft.rfftfreq(x.size, 1 / sr)
    return float((f * P).sum() / (P.sum() + 1e-30) / f0)


def spectral_irregularity(amps) -> float:
    """log10 of the standard deviation of the harmonic amplitudes from their 3-harmonic running mean."""
    a = np.asarray(amps, float)
    if a.size < 3:
        return float("nan")
    m = np.convolve(a, np.ones(3) / 3, mode="valid")
    return float(np.log10(np.std(a[1:-1] - m) + 1e-30))


def harmonic_centroid(powers) -> float:
    """Spectral centroid in harmonic numbers: sum n P_n / sum P_n (1 = all in the fundamental)."""
    p = np.asarray(powers, float)
    n = np.arange(1, p.size + 1)
    return float((n * p).sum() / (p.sum() + 1e-30))
