"""Non-verbal vocalisations (laugh, scream, cry, moan) after Anikin, "Soundgen", Behav. Res. Methods 51, 778-792
(2019), with the parameter values of soundgen's own presets (``presets.R``, speakers M1 and F1; soundgen is GPL,
only the numbers are taken, the synthesis below is written from the paper and the parameter documentation).

A bout is ``nSyl`` voiced syllables of ``sylLen`` ms separated by ``pauseLen`` ms. Per syllable:
- f0 from the ``pitch`` anchors (spread over the syllable), times 2^(st/12) of the ``pitchGlobal`` anchors (semitones,
  spread over the bout); harmonics k f0 at ``rolloff`` dB/oct, plus the +6 dB/oct of the lips (``lipRad``);
- nonlinear epochs (at least ``shortestEpoch`` ms) cover ``nonlinBalance`` % of the sound; in them the f0 is
  multiplied by 2^(N(0, jitterDep / 12)) drawn every ``jitterLen`` ms (soundgen ``wiggleGC``) and each glottal cycle
  by 1 + N(0, shimmerDep / 100);
- amplitude modulation 1 - amDep/100 (1/2 + 1/2 cos 2 pi amFreq t) (``addAM``, sine form), ``attackLen`` ms fades,
  ``ampl`` dB anchors over the syllable and ``amplGlobal`` over the bout;
- turbulent noise: ``noise`` dB anchors at ms from the syllable onset (may start before it or outlast it), relative
  to the RMS of the voiced part, tilted ``rolloffNoise`` dB/kHz above 1200 Hz, through the same formants;
- the vocal tract: an all-pole envelope (DC-normalised two-pole resonances), applied frame by frame in the STFT as
  soundgen does. Formants from ``formants`` (anchors over the syllable), bandwidths from getBandwidth (TNF-63 above
  500 Hz, Khodai-Joopari & Clermont below), formants above the given ones at the uniform-tube spacing
  F_n = (2n - 1) c / 4 VTL (VTL given, or fitted to the given formants), c = 35400 cm/s; ``mouth`` m shifts every
  formant by (m - 0.5) c / 4 VTL.
UNSOURCED: ``temperature`` here only scatters the syllable lengths and pitch anchors by 2^N(0, temperature) (soundgen
spreads it over many more parameters).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import istft, stft

from .core import SR

C_CM = 35400.0          # speed of sound in warm air, cm/s (Anikin 2019)
LIP_RAD = 6.0           # dB/oct, soundgen default
NOISE_FLAT = 1200.0     # Hz, soundgen noiseFlatSpec default

DEFAULTS = dict(nSyl=1, sylLen=500, pauseLen=200, pitch=(100, 150, 100), pitchGlobal=None, rolloff=-12.0,
                temperature=0.025, nonlinBalance=100.0, jitterDep=0.0, jitterLen=1.0, shimmerDep=0.0, attackLen=50.0,
                amFreq=30.0, amDep=0.0, formants=None, vocalTract=None, mouth=None, noise=None, rolloffNoise=-4.0,
                ampl=None, amplGlobal=None, shortestEpoch=300.0)

# soundgen presets.R (M1 = adult man, F1 = adult woman). Anchors: a tuple of values spread evenly, or (times, values)
# with times in 0..1 (ms for ``noise``).
PRESETS = {
    ("laugh", "male"): dict(nSyl=5, sylLen=120, pauseLen=120, pitch=(180, 166), pitchGlobal=((0, 0.3, 1), (1, 2, 0)),
                            rolloff=-15, temperature=0.1, nonlinBalance=50, jitterDep=0.8, attackLen=10,
                            formants=((860, 550, 550), (1430, 2300, 2300), (2900, 2800, 2800), (4000,)),
                            shortestEpoch=50, noise=((-10, 57, 237), (-35, -20, -30)), rolloffNoise=-6,
                            ampl=((0, 0.32, 1), (0, -10, 0)), amplGlobal=((0, 0.77, 1), (0, -11, -40))),
    ("laugh", "female"): dict(nSyl=3, sylLen=60, pauseLen=90, pitch=(368, 284), temperature=0.075, attackLen=10,
                              formants=((900,), (1300,), (3300,), (4340,)), noise=((0, 67, 86, 186), (-15, -17, -59, -80)),
                              rolloffNoise=-8, ampl=(0, -10)),
    ("scream", "female"): dict(sylLen=1110, pitch=((0, 0.05, 0.95, 1), (900, 1832, 1618, 1200)), temperature=0.1,
                               nonlinBalance=70, jitterDep=1, shimmerDep=10, rolloff=-6, vocalTract=13.5, amFreq=400,
                               amDep=20, rolloffNoise=0, attackLen=(10, 200)),
    ("cry", "female"): dict(sylLen=1600, pitch=(610, 511), temperature=0.2, nonlinBalance=40, vocalTract=13.5,
                            amFreq=125, amDep=10, jitterDep=1.5, mouth=(0,), ampl=(0, -20)),
    ("moan", "male"): dict(sylLen=800, pitch=((0, 0.21, 1), (230, 181, 143)), jitterDep=1, jitterLen=60, attackLen=100,
                           rolloff=-13, formants=((630, 860), (900, 1430), (3000, 2900), (3960, 3960)),
                           noise=((-20, 801, 911), (-30, -20, -60)), rolloffNoise=-8, mouth=(0.54, 0.3), ampl=(0, -9)),
    ("moan", "female"): dict(sylLen=360, pitch=(380, 260), attackLen=100, rolloff=-24,
                             formants=((900,), (1300,), (3300,), (4340,)), vocalTract=13.5,
                             noise=((0, 417, 508), (-38, -22, -80)), mouth=((0, 0.09, 1), (0.12, 0.37, 0.5)),
                             ampl=(0, -40)),
}
KINDS = sorted({k for k, _ in PRESETS})
# UNSOURCED: soundgen has no male scream or cry preset; the female one an octave down (0.5). Tract 17.5 cm,
# "typical adult male" (Story 2005, footnote 2)
MALE_FROM_FEMALE = (0.5, 17.5)


def preset(kind: str, voice: str = "female") -> dict:
    voice = "male" if str(voice).startswith("m") else "female"
    if (kind, voice) in PRESETS:
        return {**DEFAULTS, **PRESETS[(kind, voice)]}
    if kind not in KINDS:
        raise ValueError(f"unknown vocalisation {kind!r}; choose from {KINDS}")
    p = {**DEFAULTS, **PRESETS[(kind, "female")]}
    k, vtl = MALE_FROM_FEMALE
    times, vals = _split(p["pitch"])
    p["pitch"] = (times, tuple(v * k for v in vals)) if times is not None else tuple(v * k for v in vals)
    p["vocalTract"] = vtl
    return p


def _split(a):
    if a is None:
        return None, None
    if len(a) == 2 and isinstance(a[0], (tuple, list)):
        return tuple(a[0]), tuple(a[1])
    return None, tuple(np.atleast_1d(a))


def _curve(a, n: int, default: float = 0.0) -> np.ndarray:
    """Anchor -> n samples over 0..1 (linear between anchors)."""
    times, vals = _split(a)
    if vals is None:
        return np.full(n, float(default))
    if times is None:
        times = np.linspace(0.0, 1.0, len(vals)) if len(vals) > 1 else (0.0,)
    return np.interp(np.linspace(0.0, 1.0, n), times, vals)


def bandwidth(f: np.ndarray) -> np.ndarray:
    """soundgen getBandwidth: TNF-63 above 500 Hz, Khodai-Joopari & Clermont 2002 below."""
    f = np.asarray(f, float)
    return np.where(f < 250, 26.38541 - 0.042 * f + 0.0011 * f * f,
                    np.where(f < 500, 52.08333 * (1 + (f - 500) ** 2 / 1e5), 50 * (1 + f * f / 6e6)))


def _vtl(formants) -> float:
    """Apparent vocal-tract length (cm) fitted to the mean formants: F_n = (n - 1/2) d, VTL = c / 2d."""
    F = np.array([np.mean(_split(f)[1]) for f in formants])
    m = np.arange(1, F.size + 1) - 0.5
    return C_CM / (2.0 * float(m @ F / (m @ m)))


def _envelope(freqs, n_frames: int, p: dict) -> np.ndarray:
    """|H| per (frequency bin, frame): cascade of two-pole resonances normalised at DC, times the lip radiation."""
    given = list(p["formants"] or ())
    vtl = float(p["vocalTract"] or (_vtl(given) if given else 17.5))
    nyq = 0.5 * SR
    tracks = [_curve(f, n_frames) for f in given]
    k = len(tracks) + 1
    while (2 * k - 1) * C_CM / (4 * vtl) < nyq:
        tracks.append(np.full(n_frames, (2 * k - 1) * C_CM / (4 * vtl)))
        k += 1
    shift = (_curve(p["mouth"], n_frames, 0.5) - 0.5) * C_CM / (4 * vtl) if p["mouth"] is not None else 0.0
    f = freqs[:, None]
    H = np.ones((freqs.size, n_frames))
    for tr in tracks:
        F = np.maximum(tr + shift, 60.0)[None, :]
        b = bandwidth(F) / 2.0
        H *= (F * F + b * b) / np.sqrt((b * b + (f - F) ** 2) * (b * b + (f + F) ** 2))
    return H * np.maximum(freqs, 50.0)[:, None] ** (LIP_RAD / 6.0) / 50.0


def _filter(x: np.ndarray, p: dict, nperseg: int = 1024) -> np.ndarray:
    f, _, Z = stft(x, SR, nperseg=nperseg, noverlap=3 * nperseg // 4)
    _, y = istft(Z * _envelope(f, Z.shape[1], p), SR, nperseg=nperseg, noverlap=3 * nperseg // 4)
    return y[:x.size] if y.size >= x.size else np.pad(y, (0, x.size - y.size))


def _epochs(n: int, frac: float, min_len: int, rng) -> np.ndarray:
    """1 inside the nonlinear epochs (about ``frac`` of the samples), 0 elsewhere."""
    if frac >= 1.0:
        return np.ones(n)
    if frac <= 0.0:
        return np.zeros(n)
    g = np.zeros(n)
    i = 0
    while i < n:
        ln = int(min_len * rng.uniform(1.0, 3.0))
        g[i:i + ln] = rng.random() < frac
        i += ln
    return g


def _syllable(p: dict, n: int, f0: np.ndarray, rng) -> np.ndarray:
    t = np.arange(n) / SR
    gate = _epochs(n, float(p["nonlinBalance"]) / 100.0, int(float(p["shortestEpoch"]) * 1e-3 * SR), rng)
    if p["jitterDep"] > 0:
        step = max(1, int(max(float(p["jitterLen"]) * 1e-3, 1.0 / float(np.mean(f0))) * SR))
        pts = np.arange(0, n + step, step)
        jit = rng.normal(0.0, float(p["jitterDep"]) / 12.0, pts.size)
        f0 = f0 * 2.0 ** (np.interp(np.arange(n), pts, jit) * gate)
    ph = np.cumsum(f0) / SR
    cyc = np.floor(ph).astype(np.int64)
    amp = np.ones(n)
    if p["shimmerDep"] > 0:
        amp = np.clip(1.0 + rng.normal(0.0, float(p["shimmerDep"]) / 100.0, cyc[-1] + 1)[cyc] * gate, 0.0, 2.0)
    src = np.zeros(n)
    for k in range(1, int(0.45 * SR / float(np.min(f0))) + 1):
        a = 10.0 ** (float(p["rolloff"]) * np.log2(k) / 20.0)
        src += np.where(k * f0 < 0.45 * SR, a, 0.0) * np.sin(2 * np.pi * k * ph)
    src *= amp
    if p["amDep"] > 0:
        src *= 1.0 - float(p["amDep"]) / 100.0 * (0.5 + 0.5 * np.cos(2 * np.pi * float(p["amFreq"]) * t))
    att = np.atleast_1d(p["attackLen"]).astype(float)
    a_in, a_out = (int(att[0] * 1e-3 * SR), int(att[-1] * 1e-3 * SR))
    env = np.ones(n)
    if a_in > 1:
        env[:min(a_in, n)] = np.sin(0.5 * np.pi * np.arange(min(a_in, n)) / a_in) ** 2
    if a_out > 1:
        env[-min(a_out, n):] *= np.cos(0.5 * np.pi * np.arange(min(a_out, n)) / a_out) ** 2
    return src * env * 10.0 ** (_curve(p["ampl"], n) / 20.0)


def _noise(p: dict, voiced_rms: float, rng) -> tuple[np.ndarray, int]:
    """Aspiration noise of one syllable and its start in samples relative to the syllable onset."""
    times, vals = _split(p["noise"])
    if vals is None:
        return np.zeros(0), 0
    times = np.asarray(times if times is not None else np.linspace(0, p["sylLen"], len(vals)), float)
    i0, i1 = int(times[0] * 1e-3 * SR), int(times[-1] * 1e-3 * SR)
    m = max(i1 - i0, 64)
    w = rng.standard_normal(m)
    W = np.fft.rfft(w)
    fk = np.fft.rfftfreq(m, 1.0 / SR)
    W *= 10.0 ** (float(p["rolloffNoise"]) * np.maximum(fk - NOISE_FLAT, 0.0) / 1000.0 / 20.0)
    w = np.fft.irfft(W, m)
    w = _filter(w, {**p, "mouth": None})
    w /= np.sqrt(np.mean(w * w)) + 1e-12
    env_db = np.interp(np.arange(m) + i0, times * 1e-3 * SR, vals)
    return w * voiced_rms * 10.0 ** (env_db / 20.0), i0


def vocalize(kind: str, voice: str = "female", rng=None, *, n_syl: int | None = None, pitch: float = 1.0) -> np.ndarray:
    """One bout of ``kind`` (laugh, scream, cry, moan) at SR, peak 1. ``pitch`` scales f0, ``n_syl`` the bout."""
    rng = rng or np.random.default_rng(0)
    p = preset(kind, voice)
    ns = int(n_syl or p["nSyl"])
    temp = float(p["temperature"])
    lens = [max(0.02, p["sylLen"] * 2.0 ** rng.normal(0.0, temp)) for _ in range(ns)]
    starts, t = [], 0.0
    for L in lens:
        starts.append(t)
        t += L + p["pauseLen"] * 2.0 ** rng.normal(0.0, temp)
    times, vals = _split(p["noise"])
    pre = max(0.0, -min(times)) if times is not None else 0.0
    post = max(0.0, max(times) - min(lens)) if times is not None else 0.0
    total = int((pre + starts[-1] + lens[-1] + post + 20.0) * 1e-3 * SR)
    out = np.zeros(total)
    glob = _curve(p["pitchGlobal"], ns)
    gamp = _curve(p["amplGlobal"], ns)
    for s, (st, L) in enumerate(zip(starts, lens)):
        n = int(L * 1e-3 * SR)
        f0 = _curve(p["pitch"], n) * 2.0 ** (glob[s] / 12.0 + rng.normal(0.0, temp) / 12.0) * float(pitch)
        v = _filter(_syllable(p, n, f0, rng), p)
        nz, i0 = _noise(p, float(np.sqrt(np.mean(v * v))), rng)
        seg = np.zeros(total)
        a = int((pre + st) * 1e-3 * SR)
        seg[a:a + n] += v[:max(0, min(n, total - a))]
        b = a + i0
        k = min(nz.size, total - b)
        if k > 0:
            seg[b:b + k] += nz[:k]
        out += seg * 10.0 ** (gamp[s] / 20.0)
    return out / (np.max(np.abs(out)) + 1e-12)
