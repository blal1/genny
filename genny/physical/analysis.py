# Adapted for Genny v0.3 (2026): reorganized under genny.physical and integrated with
# Genny high-level APIs. Original procedural/Klang code © 2025 Chris Nash.
# Licensed under Klang Open License 1.0; see KLANG_LICENSE.txt.
"""Analysis / resynthesis of CC0 recordings for the hybrid families (voices, creatures, wind, sea).

Free recordings are *source material* for analysis and procedural transformation, never pasted raw.

* **LPC analysis / resynthesis** - autocorrelation LPC, order about SR/1000 + 4, pre-emphasis 0.97,
  20-30 ms frames (impacts: order 10, 12 ms frames, 6 ms hop). Levinson-Durbin recursion; the filter
  is run as the **ladder / lattice** of reflection coefficients ``k = (R2 - R1)/(R2 + R1)``:
  interpolating k per sample keeps every intermediate filter stable, so the time-varying tract never
  blows up and there are no frame-switch clicks.
* **Residual extraction** - inverse-filter the signal with A(z); the residual is a pulse stream
  (voiced) or white noise (unvoiced). Voicing and f0 come from the residual's normalised
  autocorrelation.
* **Resynthesis with new excitation / cross-synthesis** - keep the time-varying LPC filter and swap
  the source (a creature growl through human vowel filters = almost speech); separating voiced from
  unvoiced parts helps. Timbre stamping is the same idea in STFT form.
* **Formant warping** (vocal-tract size) - F = c/4l: the all-pole envelope is resampled on a
  frequency axis scaled by l_source / l_target and refitted (power spectrum -> autocorrelation ->
  Levinson), so the new filter is again minimum-phase all-pole.
* **Formant morphing** - interpolation of the log power envelopes, refitted to all-pole.
* **Granular engine** - symmetric grain windows (Hann), synchronous two-reader PSOLA for pitch
  shift / time stretch with jitter against the pitched quality at the grain rate, asynchronous
  clouds with pitch variance about 2 % and random start offsets for textures (water, wind, crowds);
  circular placement for seamless loops.
* **Gust statistics** - gusts are band-limited random processes; the measured envelope (modulation)
  spectrum, depth and gust rate of a CC0 wind recording replace a fixed noise shaping, keeping its
  square-law scaling. Circular FFT shaping makes the control exactly periodic (seamless beds).
* **Wave-cycle statistics** - bubble formation follows the crest (Gamma = u^2 kappa); the crest
  timing (period, rise, decay, crest-level spread) is measured on CC0 surf recordings.

Recordings live in ``synthgen/recordings/<family>/`` with ``PROVENANCE.toml``
(``recordings/fetch_recordings.py``). Set ``SYNTHGEN_HYBRID=0`` (or ``analysis.HYBRID = False``)
to render the pure-synthesis versions.
"""

from __future__ import annotations

import math
import os
try:
    import tomllib
except ModuleNotFoundError:                   # Python 3.10
    import tomli as tomllib
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numba
import numpy as np
from scipy.signal import lfilter

SR = 48000
REC = Path(__file__).resolve().parent.parent / "recordings"
PREEMPH = 0.97                       # pre-emphasis 0.97
HYBRID = os.environ.get("SYNTHGEN_HYBRID", "0") != "0"


def enabled(family: str) -> bool:
    """Hybrid rendering on for this family (flag set and at least one recording present)."""
    return HYBRID and bool(bank(family))


# ==========================================================================================
# Recordings
# ==========================================================================================

@lru_cache(maxsize=8)
def bank(family: str) -> tuple:
    """PROVENANCE rows of a family (dicts with an absolute ``path``), excluding refused ones."""
    prov = REC / family / "PROVENANCE.toml"
    if not prov.exists():
        return ()
    with prov.open("rb") as fh:
        rows = tomllib.load(fh).get("recording", [])
    out = []
    for r in rows:
        p = REC / family / r["file"]
        if p.exists() and not r.get("refused"):
            out.append({**r, "path": str(p)})
    return tuple(out)


def rows_matching(family: str, queries) -> list[dict]:
    qs = tuple(queries)
    return [r for r in bank(family) if r.get("query") in qs] or list(bank(family))


@lru_cache(maxsize=16)
def load(path: str) -> np.ndarray:
    """Mono float64 at 48 kHz (the fetcher already stores 48 kHz mono FLAC), DC removed,
    normalised to peak 0.9."""
    import soundfile as sf
    x, sr = sf.read(path, always_2d=True)
    x = x.mean(axis=1).astype(np.float64)
    if sr != SR:
        from scipy.signal import resample_poly
        g = math.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    x = x - np.mean(x)
    return x / (np.max(np.abs(x)) + 1e-12) * 0.9


@lru_cache(maxsize=8)
def rec_level(path: str) -> np.ndarray:
    return level_feature(load(path))


def envelope(x: np.ndarray, win_s: float = 0.01, hop_s: float | None = None) -> np.ndarray:
    """RMS envelope, Hann-weighted, one value per hop (default hop = win / 2)."""
    w = max(8, int(win_s * SR))
    h = max(1, int((hop_s or win_s / 2) * SR))
    k = np.hanning(w)
    k /= k.sum()
    from scipy.signal import fftconvolve
    e = np.sqrt(np.maximum(fftconvolve(x * x, k, mode="same"), 0.0))
    return e[::h]


def level_feature(x: np.ndarray, win_s: float = 0.05) -> np.ndarray:
    """Per-sample level in dB (50 ms RMS, interpolated), for level-matched grain choice."""
    hop = int(0.01 * SR)
    e = envelope(x, win_s, 0.01)
    db = 20 * np.log10(e + 1e-7)
    return np.interp(np.arange(len(x)), np.arange(len(db)) * hop, db)


def quantile_target(control: np.ndarray, feature: np.ndarray, lo: float = 3.0, hi: float = 97.0) -> np.ndarray:
    """Map a control curve (any scale) onto the feature's own distribution by rank: the control's
    minimum asks for the recording's quiet moments, its maximum for the loudest."""
    c = np.asarray(control, float)
    r = (c - c.min()) / (c.max() - c.min() + 1e-12)
    qs = np.percentile(feature[:: max(1, len(feature) // 20000)], np.linspace(lo, hi, 101))
    return np.interp(r, np.linspace(0, 1, 101), qs)


def active_regions(x: np.ndarray, min_s: float = 0.15, floor_db: float = 30.0) -> list[tuple[int, int]]:
    """Sample ranges where the 10 ms envelope sits within ``floor_db`` of the peak, bridged over
    gaps shorter than 60 ms (one vocalisation = one region)."""
    hop = int(0.005 * SR)
    e = envelope(x, 0.01, 0.005)
    db = 20 * np.log10(e + 1e-9)
    on = db > db.max() - floor_db
    idx = np.flatnonzero(on)
    if not len(idx):
        return [(0, len(x))]
    regs = []
    a = b = idx[0]
    for i in idx[1:]:
        if i - b > int(0.06 / 0.005):
            regs.append((a, b))
            a = i
        b = i
    regs.append((a, b))
    out = [(a * hop, min(len(x), (b + 1) * hop)) for a, b in regs if (b - a) * hop >= min_s * SR]
    return out or [(idx[0] * hop, min(len(x), (idx[-1] + 1) * hop))]


# ==========================================================================================
# LPC: Levinson-Durbin, lattice analysis and synthesis
# ==========================================================================================

def lpc_order(sr: float = SR) -> int:
    """SFX_HYBRID / classic rule: order about SR/1000 + 4."""
    return int(round(sr / 1000.0)) + 4


@numba.njit(cache=True)
def _levinson(r, p):
    """Levinson-Durbin on one autocorrelation vector r[0..p]. Returns (a[0..p], k[1..p], err).
    Convention A(z) = 1 + sum a_j z^-j; k_i as in the lattice f_i = f_{i-1} + k_i b_{i-1}(n-1)."""
    a = np.zeros(p + 1)
    a[0] = 1.0
    k = np.zeros(p)
    err = r[0]
    tmp = np.zeros(p + 1)
    for i in range(1, p + 1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i - j]
        ki = -acc / err if err > 1e-30 else 0.0
        if ki > 0.9995:
            ki = 0.9995
        elif ki < -0.9995:
            ki = -0.9995
        k[i - 1] = ki
        for j in range(i + 1):
            tmp[j] = a[j]
        for j in range(1, i):
            a[j] = tmp[j] + ki * tmp[i - j]
        a[i] = ki
        err *= (1.0 - ki * ki)
    return a, k, err


@numba.njit(cache=True)
def _levinson_frames(R, p):
    F = R.shape[0]
    A = np.zeros((F, p + 1))
    K = np.zeros((F, p))
    E = np.zeros(F)
    for f in range(F):
        a, k, e = _levinson(R[f], p)
        A[f] = a
        K[f] = k
        E[f] = e
    return A, K, E


@numba.njit(cache=True)
def _k_at(K, hop, i, k):
    """Reflection coefficients at sample i into k, linearly interpolated between frame centres
    f*hop (a convex mix of stable lattices is stable)."""
    F, p = K.shape
    fp = i / hop
    f0 = int(fp)
    if f0 >= F - 1:
        for j in range(p):
            k[j] = K[F - 1, j]
        return
    w = fp - f0
    for j in range(p):
        k[j] = (1.0 - w) * K[f0, j] + w * K[f0 + 1, j]


@numba.njit(cache=True)
def _lattice_inverse(x, K, hop):
    """FIR lattice (prediction-error filter A(z)) with per-sample interpolated k."""
    n = len(x)
    F, p = K.shape
    b = np.zeros(p + 1)          # b_i(n-1)
    y = np.zeros(n)
    k = np.zeros(p)
    for i in range(n):
        _k_at(K, hop, i, k)
        f = x[i]
        bprev = f                 # b_0(n)
        for j in range(p):
            fn = f + k[j] * b[j]
            bn = b[j] + k[j] * f
            b[j] = bprev
            bprev = bn
            f = fn
        b[p] = bprev
        y[i] = f
    return y


@numba.njit(cache=True)
def _lattice_synth(e, K, hop):
    """All-pole lattice 1/A(z) (ladder / Kelly-Lochbaum) with per-sample k."""
    n = len(e)
    F, p = K.shape
    b = np.zeros(p + 1)          # b_j(n-1), j = 0..p
    y = np.zeros(n)
    k = np.zeros(p)
    for i in range(n):
        _k_at(K, hop, i, k)
        f = e[i]
        for j in range(p - 1, -1, -1):
            f = f - k[j] * b[j]
            b[j + 1] = b[j] + k[j] * f
        b[0] = f
        y[i] = f
    return y


@dataclass
class Track:
    """Frame-wise LPC analysis of one recording segment.

    K: (F, p) reflection coefficients at frame centres f*hop; gain: (F,) residual RMS;
    f0: (F,) Hz (0 where unvoiced); voicing: (F,) normalised autocorrelation peak 0..1;
    n: samples the track spans (at SR)."""
    K: np.ndarray
    gain: np.ndarray
    f0: np.ndarray
    voicing: np.ndarray
    hop: int
    n: int

    @property
    def frames(self) -> int:
        return self.K.shape[0]

    def stretch(self, n_out: int) -> "Track":
        """Time-stretch the filter and control tracks to n_out samples (frames resampled; the
        new excitation is generated at the new length, so no pitch change)."""
        F = max(2, int(math.ceil(n_out / self.hop)) + 1)
        src = np.linspace(0.0, self.frames - 1, F)
        i0 = np.floor(src).astype(int)
        i1 = np.minimum(i0 + 1, self.frames - 1)
        w = (src - i0)[:, None]
        K = (1 - w) * self.K[i0] + w * self.K[i1]
        lin = lambda v: (1 - w[:, 0]) * v[i0] + w[:, 0] * v[i1]
        return replace(self, K=K, gain=lin(self.gain), f0=lin(self.f0), voicing=lin(self.voicing), n=n_out)

    def per_sample(self, v: np.ndarray, n: int | None = None) -> np.ndarray:
        n = self.n if n is None else n
        return np.interp(np.arange(n) / self.hop, np.arange(len(v)), v)


def _frames(x: np.ndarray, N: int, hop: int) -> np.ndarray:
    pad = N // 2
    xp = np.pad(x, (pad, pad + hop), mode="reflect" if len(x) > pad + hop else "constant")
    F = len(x) // hop + 1
    idx = np.arange(N)[None, :] + (np.arange(F) * hop)[:, None]
    return xp[idx]


def autocorr_frames(x: np.ndarray, N: int, hop: int, p: int, lag_bw_hz: float = 60.0) -> np.ndarray:
    """Hann-windowed frame autocorrelations r[0..p] (FFT), with a Gaussian lag window
    (bandwidth expansion) and a 1e-4 white-noise correction for a well-conditioned Levinson."""
    fr = _frames(x, N, hop) * np.hanning(N)[None, :]
    nfft = 1 << int(math.ceil(math.log2(2 * N)))
    S = np.abs(np.fft.rfft(fr, nfft, axis=1)) ** 2
    R = np.fft.irfft(S, nfft, axis=1)[:, : p + 1]
    lag = np.arange(p + 1) / SR
    R *= np.exp(-0.5 * (2 * np.pi * lag_bw_hz * lag) ** 2)[None, :]
    R[:, 0] *= 1.0 + 1e-4
    R[:, 0] += 1e-12
    return R


def preemph(x: np.ndarray, c: float = PREEMPH) -> np.ndarray:
    return lfilter([1.0, -c], [1.0], x)


def deemph(x: np.ndarray, c: float = PREEMPH) -> np.ndarray:
    return lfilter([1.0], [1.0, -c], x)


def pitch_track(e: np.ndarray, hop: int, fmin: float = 40.0, fmax: float = 900.0) -> tuple[np.ndarray, np.ndarray]:
    """f0 and voicing per frame from the residual's normalised autocorrelation (60 ms frames at
    12 kHz after a 1.5 kHz low-pass; the voiced residual is a pulse stream)."""
    from scipy.signal import resample_poly
    y = resample_poly(e, 1, 4)
    sr = SR / 4
    y = lfilter(*_butter_lp(1500.0, sr), y)
    N = int(0.06 * sr)
    h = hop / 4.0
    F = int(len(e) // hop) + 1
    pad = N // 2
    yp = np.pad(y, (pad, pad + N))
    f0 = np.zeros(F)
    v = np.zeros(F)
    lo, hi = int(sr / fmax), min(int(sr / fmin), N - 2)
    win = np.hanning(N)
    for f in range(F):
        s = int(round(f * h))
        fr = yp[s:s + N] * win
        nfft = 1 << int(math.ceil(math.log2(2 * N)))
        r = np.fft.irfft(np.abs(np.fft.rfft(fr, nfft)) ** 2, nfft)[:N]
        if r[0] <= 1e-12:
            continue
        norm = r[: hi + 1] / r[0] / (1.0 - np.arange(hi + 1) / N)   # unbias the lag window
        j = lo + int(np.argmax(norm[lo:hi]))
        if 0 < j < hi:
            a, b, c = norm[j - 1], norm[j], norm[j + 1]
            d = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
        else:
            d = 0.0
        v[f] = float(np.clip(norm[j], 0.0, 1.0))
        f0[f] = sr / (j + d)
    f0[v < 0.35] = 0.0
    return f0, v


def pitch_clean(f0: np.ndarray, hop_s: float, *, L: int = 9, jump: float = 1.05, min_seg: float = 0.04) -> np.ndarray:
    """Post-processing of a frame f0 track (0 = unvoiced): running median over ``L`` frames of the voiced
    values, a new segment wherever two neighbouring frames differ by more than the ratio ``jump``,
    and segments shorter than ``min_seg`` seconds set to 0."""
    from scipy.ndimage import median_filter
    f = np.asarray(f0, float).copy()
    on = f > 0
    if on.any():
        f[on] = median_filter(f[on], size=max(1, int(L)), mode="nearest")
    lo = math.log(max(float(jump), 1.0 + 1e-9))
    brk = np.ones(len(f) + 1, bool)
    brk[1:-1] = ~(on[1:] & on[:-1]) | (np.abs(np.log(np.maximum(f[1:], 1e-9) / np.maximum(f[:-1], 1e-9))) > lo)
    edges = np.flatnonzero(brk)
    need = max(1, int(math.ceil(float(min_seg) / float(hop_s))))
    for a, b in zip(edges[:-1], edges[1:]):
        if on[a] and b - a < need:
            f[a:b] = 0.0
    f[~on] = 0.0
    return f


def _butter_lp(fc: float, sr: float):
    from scipy.signal import butter
    return butter(2, fc / (sr / 2))


def analyse(x: np.ndarray, *, order: int | None = None, frame_ms: float = 25.0, hop_ms: float = 5.0,
            want_pitch: bool = True) -> tuple[Track, np.ndarray]:
    """LPC-analyse a segment: returns (Track, whitened residual at SR). Pre-emphasis
    0.97, Hann frames of 25 ms (SFX_HYBRID 20-30 ms), 5 ms hop, order SR/1000 + 4."""
    p = order or lpc_order()
    N = int(frame_ms * 1e-3 * SR)
    hop = int(hop_ms * 1e-3 * SR)
    xe = preemph(x)
    R = autocorr_frames(xe, N, hop, p)
    _, K, _ = _levinson_frames(R, p)
    e = _lattice_inverse(np.ascontiguousarray(xe), np.ascontiguousarray(K), float(hop))
    g = envelope(e, frame_ms * 1e-3, hop_ms * 1e-3)
    F = K.shape[0]
    g = np.pad(g, (0, max(0, F - len(g))), mode="edge")[:F]
    if want_pitch:
        f0, v = pitch_track(e, hop)
    else:
        f0, v = np.zeros(F), np.zeros(F)
    f0 = np.pad(f0, (0, max(0, F - len(f0))), mode="edge")[:F]
    v = np.pad(v, (0, max(0, F - len(v))), mode="edge")[:F]
    tr = Track(K, g, f0, v, hop, len(x))
    ew = e / (tr.per_sample(g, len(e)) + 1e-6 * (g.max() + 1e-12))
    return tr, ew


def resynth(excitation: np.ndarray, track: Track, *, gain: bool = True) -> np.ndarray:
    """Cross-synthesis: excitation (any source, normalised to unit RMS per frame by the
    caller or here) through the time-varying lattice, times the analysed residual gain track,
    then de-emphasis. The track must span len(excitation) samples (``Track.stretch``)."""
    n = len(excitation)
    tr = track if track.n == n else track.stretch(n)
    ex = excitation * (tr.per_sample(tr.gain, n) if gain else 1.0)
    y = _lattice_synth(np.ascontiguousarray(ex, dtype=np.float64), np.ascontiguousarray(tr.K), float(tr.hop))
    return deemph(y)


# ---- envelopes: warping and morphing (refitted to all-pole) ------------------------------

_NFFT_ENV = 1024


def _k_to_a(K: np.ndarray) -> np.ndarray:
    """Step-up recursion: reflection coefficients (F, p) -> A(z) coefficients (F, p+1)."""
    F, p = K.shape
    A = np.zeros((F, p + 1))
    A[:, 0] = 1.0
    for i in range(1, p + 1):
        ki = K[:, i - 1]
        prev = A[:, :i + 1].copy()
        for j in range(1, i):
            A[:, j] = prev[:, j] + ki * prev[:, i - j]
        A[:, i] = ki
    return A


def power_envelope(K: np.ndarray) -> np.ndarray:
    """|1/A(e^jw)|^2 on an rfft grid of _NFFT_ENV points, one row per frame."""
    A = _k_to_a(K)
    H = np.fft.rfft(A, _NFFT_ENV, axis=1)
    return 1.0 / np.maximum(np.abs(H) ** 2, 1e-12)


def refit(P: np.ndarray, p: int) -> np.ndarray:
    """Power envelope rows -> reflection coefficients of order p (autocorrelation method)."""
    R = np.fft.irfft(P, _NFFT_ENV, axis=1)[:, : p + 1]
    R[:, 0] *= 1.0 + 1e-6
    _, K, _ = _levinson_frames(np.ascontiguousarray(R), p)
    return K


def warp(track: Track, alpha, *, tilt_keep: bool = True) -> Track:
    """Formant warping: every resonance moves by ``alpha`` (alpha < 1 = a longer tract, lower
    formants; F = c/4l so alpha = l_source / l_target). ``alpha`` may be per frame."""
    P = power_envelope(track.K)
    bins = P.shape[1]
    grid = np.arange(bins)
    al = np.broadcast_to(np.asarray(alpha, float), (P.shape[0],))
    out = np.empty_like(P)
    for f in range(P.shape[0]):
        src = np.clip(grid / al[f], 0, bins - 1)
        out[f] = np.exp(np.interp(src, grid, np.log(P[f])))
    return replace(track, K=refit(out, track.K.shape[1]))


def morph(a: Track, b: Track, w) -> Track:
    """Interpolate two envelope tracks (log power), refitted; b is stretched onto a's length.
    ``w`` = weight of b (scalar or per frame of a)."""
    bb = b.stretch(a.n)
    F = min(a.frames, bb.frames)
    Pa, Pb = power_envelope(a.K[:F]), power_envelope(bb.K[:F])
    ww = np.broadcast_to(np.asarray(w, float), (F,))[:, None]
    # equalise overall level so the morph moves formants, not loudness
    Pa /= np.mean(Pa, axis=1, keepdims=True)
    Pb /= np.mean(Pb, axis=1, keepdims=True)
    P = np.exp((1 - ww) * np.log(Pa) + ww * np.log(Pb))
    K = refit(P, a.K.shape[1])
    return replace(a, K=K, gain=a.gain[:F], f0=a.f0[:F], voicing=a.voicing[:F])


def flatten(track: Track, amount: float) -> Track:
    """Pull every frame's envelope toward the track's own mean envelope by ``amount`` (0..1):
    less articulation, no vowel identity (never words)."""
    P = power_envelope(track.K)
    P /= np.mean(P, axis=1, keepdims=True)
    L = np.log(P)
    wts = track.gain[: len(L)] ** 2
    mean = (L * wts[:, None]).sum(0) / (wts.sum() + 1e-12)
    L = (1 - amount) * L + amount * mean[None, :]
    return replace(track, K=refit(np.exp(L), track.K.shape[1]))


def formant_pair(K: np.ndarray, gain: np.ndarray | None = None, fmin: float = 200.0) -> tuple[float, float]:
    """(F1, F2) of the energy-weighted mean all-pole envelope of some frames: the two lowest
    peaks with >= 2 dB prominence above ``fmin`` (LPC fits peaks well). For vowel choice."""
    P = power_envelope(K)
    P /= np.mean(P, axis=1, keepdims=True)
    w = (gain[: len(P)] ** 2) if gain is not None else np.ones(len(P))
    L = 10 * np.log10((P * w[:, None]).sum(0) / (w.sum() + 1e-12))
    f = np.arange(len(L)) * SR / _NFFT_ENV
    from scipy.signal import find_peaks
    pk, _ = find_peaks(L, prominence=2.0)
    pk = [float(f[i]) for i in pk if f[i] >= fmin and f[i] < 4000.0]
    if len(pk) >= 2:
        return pk[0], pk[1]
    return (pk[0], 1500.0) if pk else (500.0, 1500.0)


def first_formant(track: Track, fmin: float = 150.0) -> float:
    """Lowest clear peak (>= 3 dB prominence) of the energy-weighted mean envelope."""
    P = power_envelope(track.K)
    P /= np.mean(P, axis=1, keepdims=True)
    w = track.gain[: len(P)] ** 2
    L = 10 * np.log10((P * w[:, None]).sum(0) / (w.sum() + 1e-12))
    f = np.arange(len(L)) * SR / _NFFT_ENV
    from scipy.signal import find_peaks
    pk, _ = find_peaks(L, prominence=3.0)
    pk = [i for i in pk if f[i] >= fmin]
    return float(f[pk[0]]) if pk else 500.0


# ==========================================================================================
# Granular engine
# ==========================================================================================

@numba.njit(cache=True)
def _ola(src, out_len, t_out, pos, ratio, gain, L, wrap_out, wrap_src):
    """Overlap-add Hann grains. Grain j starts at output sample t_out[j] and reads src from
    pos[j] at speed ratio[j] (linear interpolation). Returns (y, window sum)."""
    y = np.zeros(out_len)
    wsum = np.zeros(out_len)
    ns = len(src)
    for j in range(len(t_out)):
        Lj = L[j]
        for m in range(Lj):
            w = 0.5 - 0.5 * math.cos(2.0 * math.pi * (m + 0.5) / Lj)
            rp = pos[j] + ratio[j] * m
            if wrap_src:
                rp = rp % ns
            elif rp >= ns - 1:
                break
            i0 = int(rp)
            fr = rp - i0
            i1 = i0 + 1
            if i1 >= ns:
                i1 = 0 if wrap_src else ns - 1
            s = (1.0 - fr) * src[i0] + fr * src[i1]
            o = t_out[j] + m
            if wrap_out:
                o = o % out_len
            elif o >= out_len:
                break
            y[o] += gain[j] * w * s
            wsum[o] += w
    return y, wsum


def psola(x: np.ndarray, n_out: int, *, ratio=1.0, read=None, grain_s: float = 0.04, overlap: float = 4.0,
          jitter: float = 0.3, rng: np.random.Generator | None = None, src_f0: float | None = None,
          sync: bool = False) -> np.ndarray:
    """Synchronous granular pitch shift / time stretch (two-phase PSOLA; jitter
    reduces "a pitched quality at the frequency of the grain stream"). ``read`` = source position
    per output sample (default: linear over the whole source); ``ratio`` = pitch ratio (scalar or
    per output sample). With ``src_f0`` the grain length follows 2.5 source periods. ``sync`` (with
    ``src_f0``): each grain's read point is moved by whole source periods so that it continues the phase of a
    resampler running at ``ratio``; the overlapping grains then add in phase at the new pitch instead of
    falling back to the source pitch."""
    rng = rng or np.random.default_rng(0)
    if src_f0:
        grain_s = float(np.clip(2.5 / src_f0, 0.015, 0.08))
    L = max(16, int(grain_s * SR))
    hop = max(1, int(L / overlap))
    starts = np.arange(-L // 2, n_out, hop)
    starts = starts + (rng.uniform(-jitter, jitter, len(starts)) * hop).astype(int)
    rd = np.linspace(0, len(x) - 1, n_out) if read is None else np.asarray(read, float)
    ra = np.broadcast_to(np.asarray(ratio, float), (n_out,))
    c = np.clip(starts + L // 2, 0, n_out - 1)
    pos = rd[c] - ra[c] * L / 2
    if sync and src_f0:
        per = SR / float(src_f0)
        acc = np.cumsum(ra) - ra
        base = acc[np.clip(starts, 0, n_out - 1)] + ra[c] * np.minimum(starts, 0)
        pos = base + per * np.round((pos - base) / per)
        top = np.maximum(0, len(x) - 2 - ra[c] * L)
        pos = np.where(pos < 0, pos + per * np.ceil(-pos / per), pos)
        pos = np.where(pos > top, pos - per * np.ceil((pos - top) / per), pos)
    pos = np.clip(pos, 0, np.maximum(0, len(x) - 2 - ra[c] * L))
    t0 = starts.copy()
    ok = t0 + L > 0
    t0, pos, rr = t0[ok], pos[ok], ra[c][ok]
    # a grain starting before 0 is shifted in (both output and read)
    neg = t0 < 0
    pos[neg] -= rr[neg] * t0[neg]
    t0[neg] = 0
    y, ws = _ola(np.ascontiguousarray(x, float), n_out, t0.astype(np.int64), pos, rr.astype(float),
                 np.ones(len(t0)), np.full(len(t0), L, np.int64), False, False)
    return y / np.maximum(ws, 0.25 * overlap / 2)


def _concat_pick(x, pos, extra, t, L, ratio, min_back, top):
    """Sequential grain choice: candidate 0 is ``pos``, the others ``extra``; each grain keeps the candidate whose
    first samples correlate best with the source just after the previous grain's read point, and never jumps
    back by less than ``min_back`` samples (falls back to the continuation)."""
    out = pos.copy()
    cands = np.stack([pos] + list(extra), axis=1)
    for j in range(1, len(out)):
        cont = min(out[j - 1] + (t[j] - t[j - 1]) * ratio[j - 1], top)
        cont = max(cont, 0.0)
        w = int(min(512, max(16, L[j] // 4)))
        c0 = int(cont)
        ref = x[c0:c0 + w]
        best, score = None, -np.inf
        for c in cands[j]:
            if min_back > 0 and 0 < out[j - 1] - c < min_back:
                continue
            if cands.shape[1] == 1:
                best = c
                break
            s = x[int(c):int(c) + w]
            m = min(len(s), len(ref))
            if m < 8:
                continue
            a, b = s[:m] - s[:m].mean(), ref[:m] - ref[:m].mean()
            r = float(a @ b) / (math.sqrt(float(a @ a) * float(b @ b)) + 1e-12)
            if r > score:
                best, score = c, r
        out[j] = cont if best is None else best
    return out


def cloud(x: np.ndarray, n_out: int, rng: np.random.Generator, *, target=None, feature=None,
          grain_s=(0.06, 0.2), overlap: float = 3.0, pitch_var: float = 0.02, loop: bool = False,
          choose_frac: float = 0.04, src_mask: np.ndarray | None = None, pitch_st: float | None = None,
          positions: str = "random", match: int = 0, min_back: float = 0.0) -> np.ndarray:
    """Asynchronous grain cloud from a recording (random start offsets, pitch
    variance about 2 % "a thick chorusing effect", overlap 1-2+; texture of "wind, water").

    ``target`` (per output sample, any scale) and ``feature`` (per source sample, same scale)
    make it *level-matched*: each grain is taken from a random source position whose feature
    lies within the nearest ``choose_frac`` of the target, so loud crests draw from loud,
    bright moments of the recording and quiet troughs from quiet ones (the procedural control
    decides *when*, the recording supplies *what it sounds like*). ``loop`` wraps grains around
    the output end (circular -> seamless bed).

    ``pitch_st``: each grain transposed by a uniform random amount in +-pitch_st semitones (random
    granulation; replaces ``pitch_var``). ``positions`` (without a target): random | gauss (around a
    head crossing the source once) | walk (random walk) | zigzag (head sweeping the source forth and
    back). ``match`` > 1: concatenative choice, each grain taken among ``match`` candidates as the one
    whose opening best correlates with the natural continuation of the previous grain.
    ``min_back`` (s): no backward jump shorter than this between consecutive grains."""
    ns = len(x)
    mean_L = 0.5 * (grain_s[0] + grain_s[1]) * SR
    hop = mean_L / overlap
    count = int(n_out / hop) + 2
    t = np.sort(rng.uniform(-mean_L if not loop else 0, n_out, count)).astype(np.int64)
    L = rng.uniform(grain_s[0], grain_s[1], count) * SR
    L = L.astype(np.int64)
    if pitch_st:
        ratio = 2.0 ** (rng.uniform(-1.0, 1.0, count) * float(pitch_st) / 12.0)
    else:
        ratio = 2.0 ** (rng.normal(0.0, pitch_var, count) / 1.0) if pitch_var else np.ones(count)
        ratio = np.clip(ratio, 0.8, 1.25)
    valid = np.arange(ns) if src_mask is None else np.flatnonzero(src_mask)
    valid = valid[valid < ns - int(1.3 * L.max()) - 2]
    if not len(valid):
        valid = np.arange(max(1, ns - int(1.3 * L.max()) - 2))
    valid = valid[:: max(1, len(valid) // 200000)]
    if target is not None and feature is not None:
        order = valid[np.argsort(feature[valid])]
        fs = feature[order]
        tc = np.clip(t + L // 2, 0, n_out - 1)
        tv = np.asarray(target, float)[tc]
        q = np.searchsorted(fs, tv)
        span = max(4, int(choose_frac * len(order)))
        lo = np.clip(q - span // 2, 0, len(order) - span)
        pos = order[lo + rng.integers(0, span, count)].astype(float)
        draw = lambda: order[lo + rng.integers(0, span, count)].astype(float)
    elif positions != "random":
        u = np.clip(t, 0, n_out) / max(n_out, 1)
        if positions == "gauss":
            u = u + rng.normal(0.0, 0.05, count)
        elif positions == "walk":
            u = 0.5 + np.cumsum(rng.normal(0.0, 0.03, count))
        elif positions == "zigzag":
            u = 2.0 * u
        else:
            raise ValueError("positions must be random|gauss|walk|zigzag")
        r = np.mod(u, 2.0)
        u = np.where(r > 1.0, 2.0 - r, r)
        pos = valid[(u * (len(valid) - 1)).astype(np.int64)].astype(float)
        draw = lambda: valid[rng.integers(0, len(valid), count)].astype(float)
    else:
        pos = valid[rng.integers(0, len(valid), count)].astype(float)
        draw = lambda: valid[rng.integers(0, len(valid), count)].astype(float)
    if int(match) > 1 or min_back > 0:
        pos = _concat_pick(np.asarray(x, float), pos, [draw() for _ in range(max(int(match), 1) - 1)], t, L,
                           ratio, float(min_back) * SR, float(valid.max()))
    t_in = t.copy()
    if not loop:
        neg = t_in < 0
        pos[neg] -= t_in[neg]
        L[neg] = np.maximum(L[neg] + t_in[neg], 16)
        t_in[neg] = 0
    y, ws = _ola(np.ascontiguousarray(x, float), n_out, t_in, pos, ratio.astype(float), np.ones(count), L,
                 bool(loop), False)
    # constant-power normalisation of random overlaps: divide by sqrt of the window power sum
    return y / np.sqrt(np.maximum(ws, 0.3) * 0.5)


def wavescan(x: np.ndarray, f0, n: int, *, start: float = 0.0, end: float = 1.0, sr: float = SR) -> np.ndarray:
    """Wave scanning: the sound read as a moving wavetable. Two readers half a period apart each read one
    period (sr / f0) of the source under a sin^2 window (overlap 2, the windows sum to 1); the read point
    slides from ``start`` to ``end`` (fractions of the source) over the ``n`` output samples."""
    x = np.asarray(x, float)
    f = np.broadcast_to(np.asarray(f0, float), (n,))
    P = sr / f
    ph = np.cumsum(f / sr)
    head = (start + (end - start) * np.arange(n) / max(n - 1, 1)) * max(len(x) - P.max() - 2, 0)
    y = np.zeros(n)
    for off in (0.0, 0.5):
        q = np.mod(ph + off, 1.0)
        y += np.sin(np.pi * q) ** 2 * np.interp(head + q * P, np.arange(len(x)), x)
    return y


def vector_mix(a, b, c, d, x, y) -> np.ndarray:
    """Vector synthesis: four sources crossfaded by a joystick (x, y) in [0, 1]^2 (bilinear weights,
    a at (0, 0), b at (1, 0), c at (0, 1), d at (1, 1)); x and y may be per-sample arrays."""
    n = min(len(a), len(b), len(c), len(d))
    x = np.clip(np.broadcast_to(np.asarray(x, float), (n,)), 0, 1)
    y = np.clip(np.broadcast_to(np.asarray(y, float), (n,)), 0, 1)
    return ((1 - x) * (1 - y) * a[:n] + x * (1 - y) * b[:n] + (1 - x) * y * c[:n] + x * y * d[:n])


def recorded_texture(family: str, queries, pick: float, control: np.ndarray, rng: np.random.Generator, *,
                     loop: bool = True, grain_s=(0.08, 0.25), overlap: float = 3.0, pitch_var: float = 0.02,
                     used: list | None = None, file: str | None = None) -> np.ndarray:
    """Level-matched grain cloud of one recording of ``family`` under a procedural ``control``
    (len = output length): the control decides when it swells, the recording what a swell
    sounds like (granular texture; control = gusts / crest timing).
    ``file`` forces a recording (e.g. the one whose statistics made the control)."""
    rows = rows_matching(family, queries)
    row = (row_by_file(family, file) if file else None) or rows[int(pick * len(rows)) % len(rows)]
    if used is not None:
        used.append(row["file"])
    x = load(row["path"])
    feat = rec_level(row["path"])
    target = quantile_target(control, feat)
    y = cloud(x, len(control), rng, target=target, feature=feat, grain_s=grain_s, overlap=overlap,
              pitch_var=pitch_var, loop=loop, src_mask=feat > np.percentile(feat, 1.0))
    return y


# ==========================================================================================
# Recording profiles (screening), transients, modal fit of recorded impacts
# ==========================================================================================

PROFILE_CACHE = REC / ".analysis" / "profiles.json"


def _profile_of(path: str) -> dict:
    x = load(path)
    regs = active_regions(x, 0.1)
    active = sum(b - a for a, b in regs) / SR
    hop = int(0.02 * SR)
    f0, v = pitch_track(x, hop, 40.0, 900.0)
    vf = f0[f0 > 0]
    cen, lev = _centroid_track(x) if len(x) > 4096 else (np.array([0.0]), np.array([0.0]))
    ok = lev > lev.max() - 30
    return {"seconds": round(len(x) / SR, 2), "active_s": round(active, 2),
            "voiced_frac": round(float(np.mean(f0 > 0)), 3),
            "f0_median": round(float(np.median(vf)), 1) if len(vf) else 0.0,
            "centroid": round(float(np.median(cen[ok])), 0) if ok.any() else 0.0}


def recording_profile(row: dict) -> dict:
    """Screening numbers of one recording (duration, active seconds, voiced fraction, median f0,
    centroid), cached on disk in recordings/.analysis/profiles.json by file name and size."""
    import json
    key = f"{row['file']}:{Path(row['path']).stat().st_size}"
    cache = _profiles_cache()
    if key not in cache:
        cache[key] = _profile_of(row["path"])
        try:
            PROFILE_CACHE.parent.mkdir(parents=True, exist_ok=True)
            tmp = PROFILE_CACHE.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(cache), encoding="utf-8")
            os.replace(tmp, PROFILE_CACHE)
        except OSError:
            pass
    return cache[key]


@lru_cache(maxsize=1)
def _profiles_cache() -> dict:
    import json
    try:
        return json.loads(PROFILE_CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def rows_where(family: str, queries, pred) -> list[dict]:
    """rows_matching, then kept only where pred(profile) holds (falls back to all matches)."""
    rows = rows_matching(family, queries)
    good = [r for r in rows if pred(recording_profile(r))]
    return good or rows


def transient(x: np.ndarray, rng: np.random.Generator, *, dur: float = 0.5, pre: float = 0.004,
              top: int = 6) -> np.ndarray:
    """One struck event cut from a recording: onsets by positive energy flux (5 ms frames), one
    of the ``top`` strongest chosen by rng (a new take each time), from ``pre`` before the onset
    for ``dur`` seconds, faded out over its last 30 %."""
    hop = int(0.005 * SR)
    e = envelope(x, 0.005, 0.005)
    d = np.diff(np.log(e + 1e-6))
    score = np.maximum(d, 0) * e[1:]
    from scipy.signal import find_peaks
    pk, _ = find_peaks(score, distance=int(0.12 / 0.005))
    if not len(pk):
        pk = np.array([int(np.argmax(e))])
    pk = pk[np.argsort(-score[pk])][:top]
    i = int(pk[int(rng.integers(len(pk)))]) * hop
    a = max(0, i - int(pre * SR))
    y = x[a:a + int(dur * SR)].copy()
    if len(y) < 64:
        y = np.pad(y, (0, 64 - len(y)))
    k = int(0.3 * len(y))
    y[-k:] *= np.linspace(1.0, 0.0, k) ** 2
    return y


@dataclass
class Modes:
    freqs: np.ndarray
    decays: np.ndarray        # 1/s (amplitude e^{-d t})
    gains: np.ndarray


VOTE_FLOOR_DB = 6.0
VOTE_MIN_DECAY = 0.1


def modal_fit(x: np.ndarray, n_modes: int = 16, fmin: float = 60.0, fmax: float = 16000.0, *,
              method: str = "peak", N: int = 4096) -> Modes:
    """Modal analysis of a recorded strike:
    the strongest spectral peaks of the first 60 ms after the onset, each one's decay rate
    from the slope of its STFT magnitude (dB) over the following frames. ``N`` = FFT length.
    method="vote": a peak is kept only if it is found (within one bin) in at least half of the
    frames of those 60 ms, stands ``VOTE_FLOOR_DB`` above the noise floor measured before the
    onset, and decays faster than ``VOTE_MIN_DECAY`` 1/s (steady hum and noise lines are dropped)."""
    from scipy.signal import find_peaks, stft
    N = int(N)
    if len(x) < 2 * N:
        x = np.pad(x, (0, 2 * N - len(x)))
    f, t, S = stft(x, SR, nperseg=N, noverlap=N - 512, boundary=None)
    M = np.abs(S) + 1e-12
    j0 = int(np.argmax(M.sum(0)))
    nf = max(1, int(0.06 * SR / 512))
    spec = 20 * np.log10(M[:, j0:j0 + nf].mean(1))
    pk, _ = find_peaks(spec, prominence=6.0, distance=3)
    pk = pk[(f[pk] >= fmin) & (f[pk] <= fmax)]
    if method == "vote":
        fr = M[:, j0:j0 + nf]
        votes = np.zeros(len(f))
        for k in range(fr.shape[1]):
            p, _ = find_peaks(20 * np.log10(fr[:, k]), prominence=6.0, distance=3)
            hit = np.zeros(len(f), bool)
            hit[p] = True
            hit[np.clip(p - 1, 0, len(f) - 1)] = True
            hit[np.clip(p + 1, 0, len(f) - 1)] = True
            votes += hit
        pk = pk[votes[pk] >= 0.5 * fr.shape[1]]
        if j0 > 0:
            floor = 20 * np.log10(np.median(M[:, :j0], axis=1))
            pk = pk[spec[pk] >= floor[pk] + VOTE_FLOOR_DB]
    elif method != "peak":
        raise ValueError("method must be peak|vote")
    pk = pk[np.argsort(-spec[pk])][:n_modes]
    freqs, decays, gains = [], [], []
    tt = t[j0:] - t[j0]
    for b in pk:
        tr = 20 * np.log10(M[b, j0:])
        m = tr > tr[0] - 40
        stop = int(np.argmin(m)) if not m.all() else len(tr)
        if stop < 3:
            stop = min(len(tr), 3)
        slope = np.polyfit(tt[:stop], tr[:stop], 1)[0] if stop >= 2 else -60.0
        if method == "vote" and -slope / 8.686 < VOTE_MIN_DECAY:
            continue
        decays.append(float(np.clip(-slope / 8.686, 2.0, 400.0)))
        # parabolic interpolation on the dB spectrum: the bin alone is off by up to half a bin, 5.9 Hz here,
        # 23 cents at 400 Hz
        a, c = spec[b - 1], spec[min(b + 1, spec.size - 1)]
        den = a - 2 * spec[b] + c
        freqs.append(float(f[b] + (0.5 * (a - c) / den if den < 0 else 0.0) * (f[1] - f[0])))
        gains.append(float(10 ** (spec[b] / 20)))
    g = np.array(gains) if gains else np.array([1.0])
    return Modes(np.array(freqs or [500.0]), np.array(decays or [30.0]), g / g.max())


def modal_resynth(md: Modes, n: int, rng: np.random.Generator, *, f_scale: float = 1.0,
                  decay_scale: float = 1.0, jitter: float = 0.02) -> np.ndarray:
    """Sum of exponentially decaying sinusoids (modal structure), frequencies scaled
    (size: f ~ 1/L), decays scaled, each mode detuned by up to ``jitter`` so no two takes match."""
    t = np.arange(n) / SR
    y = np.zeros(n)
    for f, d, g in zip(md.freqs, md.decays, md.gains):
        ff = f * f_scale * (1.0 + rng.uniform(-jitter, jitter))
        if ff >= 0.45 * SR or ff < 20:
            continue
        y += g * np.sin(2 * np.pi * ff * t + rng.uniform(0, 2 * np.pi)) * np.exp(-d * decay_scale * t)
    return y


def recorded_event(family: str, queries, pick: float, rng: np.random.Generator, *, dur: float = 0.5,
                   f_scale: float = 1.0, t_scale: float = 1.0, modal: float = 0.0, order: int = 24,
                   used: list | None = None) -> np.ndarray:
    """A recorded impact / step / fall, resynthesised (never pasted): a transient cut from a
    recording is LPC-analysed (impacts resynthesise well from a low order and their
    residual), its residual time-stretched by ``t_scale`` and pitch-shifted by ``f_scale``
    (granular PSOLA), the envelope warped by ``f_scale`` (size), then passed back through the
    lattice. ``modal`` > 0 adds that share of a modal resynthesis of the same strike (
    modes + residual), frequencies x f_scale, decays / t_scale. Peak-normalised."""
    rows = rows_matching(family, queries)
    row = rows[int(pick * len(rows)) % len(rows)]
    if used is not None:
        used.append(row["file"])
    x = transient(load(row["path"]), rng, dur=dur)
    tr, e = analyse(x, order=order, frame_ms=12.0, hop_ms=3.0, want_pitch=False)
    n = int(len(x) * t_scale)
    res = psola(e, n, ratio=f_scale, rng=rng, grain_s=0.012, overlap=4.0, jitter=0.3)
    trn = tr.stretch(n)
    if abs(f_scale - 1.0) > 1e-3:
        trn = warp(trn, f_scale)
    y = resynth(res, trn)
    y = y / (np.max(np.abs(y)) + 1e-12)
    if modal > 0:
        m = modal_resynth(modal_fit(x), n, rng, f_scale=f_scale, decay_scale=1.0 / t_scale)
        m = m / (np.max(np.abs(m)) + 1e-12) * np.minimum(1.0, np.arange(n) / (0.001 * SR))
        y = (1.0 - modal) * y + modal * m
    k = min(n // 3, int(0.02 * SR))
    y[-k:] *= np.linspace(1.0, 0.0, k)
    return y / (np.max(np.abs(y)) + 1e-12)


# ==========================================================================================
# Wind: gust statistics (measured control)
# ==========================================================================================

CTRL_RATE = 100.0                     # control samples per second


@dataclass
class GustStats:
    mod_spectrum: np.ndarray          # power of the log-envelope, freqs CTRL grid (0..5 Hz)
    mod_freqs: np.ndarray
    depth_db: float                   # std of the 50 ms envelope in dB
    gust_rate: float                  # gusts per minute (upward crossings, 0.5 s smoothed, +3 dB)
    tilt_db_oct: float                # long-term spectral slope 100 Hz - 8 kHz
    bands_db: np.ndarray              # long-term 1/3-octave levels (relative)
    band_freqs: np.ndarray
    bright_per_db: float              # centroid change (octaves) per dB of level
    source: str


def third_octaves(lo: float = 50.0, hi: float = 16000.0) -> np.ndarray:
    n = int(math.floor(3 * math.log2(hi / lo))) + 1
    return lo * 2 ** (np.arange(n) / 3)


def long_term_bands(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import welch
    f, P = welch(x, SR, nperseg=8192)
    fc = third_octaves()
    lv = []
    for c in fc:
        m = (f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))
        lv.append(10 * np.log10(np.mean(P[m]) + 1e-20) if np.any(m) else -200.0)
    lv = np.array(lv)
    return fc, lv - lv.max()


def _centroid_track(x: np.ndarray, hop_s: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import stft
    f, t, S = stft(x, SR, nperseg=2048, noverlap=2048 - int(hop_s * SR))
    m = np.abs(S)
    cen = (f[:, None] * m).sum(0) / (m.sum(0) + 1e-12)
    lev = 10 * np.log10((m ** 2).sum(0) + 1e-20)
    return cen, lev


def gust_stats(x: np.ndarray, name: str = "") -> GustStats:
    env = envelope(x, 0.05, 1.0 / CTRL_RATE)
    db = 20 * np.log10(env + 1e-9)
    db = db[db > db.max() - 60]
    from scipy.signal import welch
    f, P = welch(db - db.mean(), CTRL_RATE, nperseg=min(len(db), 1024))
    keep = f <= 5.0
    sm = np.convolve(db, np.ones(50) / 50, mode="same")
    med = np.median(sm)
    ups = np.flatnonzero((sm[1:] > med + 3) & (sm[:-1] <= med + 3))
    rate = len(ups) / (len(db) / CTRL_RATE / 60.0)
    fc, lv = long_term_bands(x)
    m = (fc >= 100) & (fc <= 8000)
    tilt = float(np.polyfit(np.log2(fc[m]), lv[m], 1)[0])
    cen, lev = _centroid_track(x)
    ok = lev > lev.max() - 40
    slope = float(np.polyfit(lev[ok], np.log2(cen[ok] + 1), 1)[0]) if ok.sum() > 10 else 0.0
    return GustStats(P[keep], f[keep], float(np.std(db)), float(rate), tilt, lv, fc, slope, name)


def gust_control(stats: GustStats, dur: float, rng: np.random.Generator, *, loop: bool = True,
                 depth_scale: float = 1.0) -> np.ndarray:
    """Level deviation in dB at SR length: random-phase noise whose spectrum is the measured
    modulation spectrum of the recording's log envelope (circular FFT shaping -> exactly periodic
    when ``loop``), zero mean, std = the measured depth x ``depth_scale``."""
    m = max(64, int(round(dur * CTRL_RATE)))
    spec = np.fft.rfft(rng.standard_normal(m))
    fr = np.fft.rfftfreq(m, 1.0 / CTRL_RATE)
    shape = np.interp(fr, stats.mod_freqs, np.sqrt(np.maximum(stats.mod_spectrum, 1e-12)), right=0.0)
    shape[0] = 0.0
    c = np.fft.irfft(spec * shape, m)
    c = (c - c.mean()) / (c.std() + 1e-12)
    n = int(round(dur * SR))
    tt = np.arange(n) / SR * CTRL_RATE
    if loop:
        c = np.interp(tt, np.arange(m + 1), np.append(c, c[0]))
    else:
        c = np.interp(tt, np.arange(m), c)
    return c * stats.depth_db * depth_scale


@lru_cache(maxsize=16)
def wind_stats(kind: str) -> tuple:
    """Gust statistics of the family's recordings of one kind ("calm" or "gale")."""
    qs = WIND_KINDS[kind]
    out = []
    for r in rows_matching("wind", qs):
        x = load(r["path"])
        out.append((r["file"], gust_stats(x, r["file"])))
    return tuple(out)


WIND_KINDS = {"calm": ("wind", "windy", "wind trees"),
              "gale": ("strong wind", "wind storm", "gale", "wind howling", "blizzard")}
SEA_KINDS = {"calm": ("ocean waves", "sea waves", "waves beach", "sea shore", "gentle waves", "calm sea",
                      "calm waves", "lapping waves", "small waves", "waves pebbles", "shingle beach",
                      "pebble beach waves", "lake waves", "waves sand", "beach waves", "seashore", "coast",
                      "tide", "ocean", "sea"),
             "storm": ("storm sea waves", "rough sea", "waves crashing rocks", "storm waves", "storm sea",
                       "waves rocks", "breaking waves", "surf", "waves crashing", "crashing waves", "wave crash",
                       "surf crash", "big waves", "waves breaking beach", "shore break", "pounding surf",
                       "waves on rocks", "sea cliff waves", "stormy ocean", "heavy surf")}
# wave-cycle screens: storm surf with short crash periods, calm shores with clear cycles
SEA_SCREEN = {"storm": lambda s: 6.0 <= s.period <= 14.0 and s.depth_db >= 4.0,
              "calm": lambda s: 5.0 <= s.period <= 16.0 and s.depth_db >= 5.0}


def eq_to_bands(x: np.ndarray, fc: np.ndarray, target_db: np.ndarray, *, amount: float = 1.0,
                max_db: float = 18.0) -> np.ndarray:
    """Circular (zero-phase FFT) EQ so x's long-term 1/3-octave spectrum follows target_db
    (relative), by ``amount``. Circular: a periodic input stays periodic (seamless beds)."""
    x = np.asarray(x, float)
    _, cur = long_term_bands(x if x.ndim == 1 else x.mean(axis=1))
    d = np.clip((target_db - target_db.max()) - (cur - cur.max()), -max_db, max_db) * amount
    n = x.shape[0]
    fr = np.fft.rfftfreq(n, 1 / SR)
    g = 10 ** (np.interp(np.log2(np.maximum(fr, 1.0)), np.log2(fc), d) / 20)
    X = np.fft.rfft(x, axis=0)
    X = X * (g if x.ndim == 1 else g[:, None])
    return np.fft.irfft(X, n, axis=0)


# ==========================================================================================
# Sea: wave-cycle statistics (measured on CC0 surf)
# ==========================================================================================

@dataclass
class WaveStats:
    period: float                     # s, envelope autocorrelation peak in 3-25 s
    periods: np.ndarray               # crest-to-crest intervals, s
    rises: np.ndarray                 # s, trough -> crest
    decays: np.ndarray                # s, crest -> -8 dB (or next trough)
    crest_db: np.ndarray              # crest levels re the median crest, dB
    depth_db: float                   # median crest - trough, dB
    bands_db: np.ndarray
    band_freqs: np.ndarray
    source: str


def wave_stats(x: np.ndarray, name: str = "") -> WaveStats:
    env = envelope(x, 0.1, 0.05)                           # 20 values / s
    db = 20 * np.log10(env + 1e-9)
    sm = np.convolve(db, np.hanning(15) / np.hanning(15).sum(), mode="same")
    d = sm - sm.mean()
    ac = np.correlate(d, d, "full")[len(d) - 1:]
    ac /= ac[0] + 1e-12
    lo, hi = int(3 * 20), min(int(25 * 20), len(ac) - 1)
    from scipy.signal import find_peaks
    period = 9.0
    if hi > lo + 2:
        # the first clear autocorrelation peak (not the largest lag, which drifts to the window end)
        cand, _ = find_peaks(ac[lo:hi], prominence=0.03)
        if len(cand):
            best = ac[lo:hi][cand].max()
            thr = 0.6 * best if best > 0 else best
            first = [c for c in cand if ac[lo + c] >= thr][0]
            period = (lo + int(first)) / 20.0
        else:
            period = (lo + int(np.argmax(ac[lo:hi]))) / 20.0
    pk, _ = find_peaks(sm, distance=max(1, int(0.45 * period * 20)), prominence=2.0)
    rises, decays, crest = [], [], []
    for i, p in enumerate(pk):
        a = pk[i - 1] if i else 0
        tr = a + int(np.argmin(sm[a:p + 1])) if p > a else p
        rises.append((p - tr) / 20.0)
        b = pk[i + 1] if i + 1 < len(pk) else len(sm) - 1
        seg = sm[p:b + 1]
        below = np.flatnonzero(seg < sm[p] - 8.0)
        decays.append((below[0] if len(below) else len(seg)) / 20.0)
        crest.append(sm[p])
    crest = np.array(crest) if crest else np.array([0.0])
    troughs = [np.min(sm[pk[i]:pk[i + 1]]) for i in range(len(pk) - 1)] or [sm.min()]
    fc, lv = long_term_bands(x)
    return WaveStats(period, np.diff(pk) / 20.0 if len(pk) > 1 else np.array([period]),
                     np.clip(np.array(rises or [1.5]), 0.3, 12.0), np.clip(np.array(decays or [4.0]), 0.5, 20.0),
                     crest - np.median(crest), float(np.median(crest) - np.median(troughs)), lv, fc, name)


@lru_cache(maxsize=8)
def sea_stats(kind: str) -> tuple:
    out = []
    for r in rows_matching("sea", SEA_KINDS[kind]):
        x = load(r["path"])
        if len(x) < 8 * SR:
            continue
        out.append((r["file"], wave_stats(x, r["file"])))
    return tuple(out)


def sea_stats_screened(kind: str) -> tuple:
    """sea_stats(kind) kept to recordings passing SEA_SCREEN[kind] (short crash periods for storm,
    clear cycles for calm); all of them if none passes."""
    import re
    soft = re.compile(r"gentle|lapping|calm|small_waves|zen")
    pool = sea_stats("storm") + sea_stats("calm")
    seen, allst = set(), []
    for name, st in pool:
        if name in seen:
            continue
        seen.add(name)
        is_soft = bool(soft.search(name)) or name in {nm for nm, _ in sea_stats("calm")}
        if (kind == "calm") == is_soft:
            allst.append((name, st))
    ok = tuple(item for item in allst
               if SEA_SCREEN[kind](item[1]) and len(item[1].crest_db) >= 5 and item[1].depth_db <= 30.0)
    return ok or tuple(allst) or sea_stats(kind)


def row_by_file(family: str, file: str) -> dict | None:
    for r in bank(family):
        if r["file"] == file:
            return r
    return None


def wave_control(stats: WaveStats, n: int, rng: np.random.Generator, *, loop: bool = True):
    """Crest envelopes over n samples from the measured statistics (intervals, rise and decay
    times, crest levels drawn from the recording's own distributions with 10 % jitter; square-law
    rise, exponential wash decay reaching -8 dB at the measured decay time). Returns
    (env 0..1 at SR, starts [(i0, rise_samples, amplitude)]) like ambience._wave_env."""
    dur = n / SR
    env = np.zeros(n + int(30 * SR))
    starts = []
    t = rng.uniform(0.0, max(0.5, float(np.median(stats.periods))))
    pick = lambda arr: float(rng.choice(arr)) * rng.uniform(0.9, 1.1)
    while t < dur - (0.0 if loop else 2.0):
        tr, td = pick(stats.rises), pick(stats.decays)
        a = float(np.clip(10 ** (pick(stats.crest_db) / 20.0) * 0.75, 0.2, 1.0))
        nr, nd = max(1, int(tr * SR)), max(1, int(td * 2.5 * SR))
        up = (np.arange(nr) / nr) ** 2
        down = np.exp(-np.arange(nd) / (td * SR) * math.log(10 ** (8 / 20)))
        floor = 10 ** (-stats.depth_db / 20.0)
        e = np.concatenate([up, down]) * a
        i = int(t * SR)
        m = min(len(e), len(env) - i)
        env[i:i + m] = np.maximum(env[i:i + m], e[:m])
        starts.append((i, nr, a))
        t += max(1.5, pick(stats.periods))
    if loop:
        tail = env[n:].copy()
        env = env[:n]
        k = 0
        while k < len(tail):
            seg = tail[k:k + n]
            env[: len(seg)] = np.maximum(env[: len(seg)], seg)
            k += n
    else:
        env = env[:n]
    floor = 10 ** (-min(stats.depth_db, 30.0) / 20.0)
    return np.maximum(env, floor * 0.5), starts


BREAK_FACTOR = 1.25         # a gap between glottal pulses longer than 1.25 / pitch floor is a voice break


def voice_breaks(pulses, duration: float, fmin: float = 75.0) -> tuple[int, float]:
    """(number of voice breaks, degree of voice breaks) from glottal pulse times in s: a break is an interval
    between consecutive pulses longer than ``BREAK_FACTOR`` / ``fmin`` (16.7 ms at 75 Hz); the degree is the
    breaks' summed length over ``duration``."""
    t = np.sort(np.asarray(pulses, float))
    d = np.diff(t)
    gaps = d[d > BREAK_FACTOR / float(fmin)]
    return int(len(gaps)), float(gaps.sum() / max(float(duration), 1e-12))


RESIDUAL_GL_ITERS = 10


def residual_transfer(rec: np.ndarray, md_src: Modes, md_tgt: Modes, rng: np.random.Generator | None = None, *,
                      nfft: int = 1024, iters: int = RESIDUAL_GL_ITERS) -> np.ndarray:
    """Residual of a recording (recording minus its modal resynthesis) carried to another set of modes: for each
    mode pair the spectrogram band between the neighbouring mid-frequencies is moved from f_s to f_t, stretched in
    time by d_s / d_t and scaled by g_t / g_s, then a phase is rebuilt by Griffin-Lim (``iters`` passes)."""
    from scipy.signal import istft, stft
    rng = rng or np.random.default_rng(0)
    x = np.asarray(rec, float)
    res = x - modal_resynth(md_src, x.size, rng, jitter=0.0) * (np.std(x) / (np.std(modal_resynth(md_src, x.size, np.random.default_rng(0), jitter=0.0)) + 1e-12))
    f, t, Z = stft(res, SR, nperseg=nfft)
    M = np.abs(Z)
    out = np.zeros_like(M)
    n = min(len(md_src.freqs), len(md_tgt.freqs))
    fs_, ft_ = np.sort(md_src.freqs[:n]), np.sort(md_tgt.freqs[:n])
    os_, ot_ = np.argsort(md_src.freqs[:n]), np.argsort(md_tgt.freqs[:n])

    def edges(fr):
        mid = 0.5 * (fr[1:] + fr[:-1])
        return np.concatenate([[0.0], mid]), np.concatenate([mid, [0.5 * SR]])
    sl, sh = edges(fs_)
    tl, th = edges(ft_)
    frames = np.arange(M.shape[1], dtype=float)
    for i in range(n):
        ds, dt = md_src.decays[os_[i]], md_tgt.decays[ot_[i]]
        gs, gt = md_src.gains[os_[i]], md_tgt.gains[ot_[i]]
        band = (f >= tl[i]) & (f < th[i])
        if not band.any():
            continue
        src_f = fs_[i] + (f[band] - ft_[i]) * (sh[i] - sl[i]) / max(th[i] - tl[i], 1e-9)
        src_t = frames * (dt / max(ds, 1e-9))
        for j, fj in zip(np.flatnonzero(band), src_f):
            col = np.interp(fj, f, np.arange(f.size))
            k0 = int(np.clip(col, 0, f.size - 2))
            row = (1 - (col - k0)) * M[k0] + (col - k0) * M[k0 + 1]
            out[j] = np.interp(frames, src_t, row, right=0.0) * gt / max(gs, 1e-9) * np.sqrt(ds / max(dt, 1e-9))
    phase = np.exp(2j * np.pi * rng.uniform(size=out.shape))
    y = None
    for _ in range(max(1, iters)):
        _, y = istft(out * phase, SR, nperseg=nfft)
        _, _, Z2 = stft(y, SR, nperseg=nfft)
        Z2 = Z2[:, :out.shape[1]] if Z2.shape[1] >= out.shape[1] else np.pad(Z2, ((0, 0), (0, out.shape[1] - Z2.shape[1])))
        phase = np.exp(1j * np.angle(Z2))
    return y[:x.size]
