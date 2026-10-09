"""STFT-domain effects and analysis -> resynthesis.

Source for everything here: J. O. Smith, *Spectral Audio Signal Processing* (SASP), as read in
``out/research/g12-jos-sasp.md`` (cited below as g12 §n).

* Kernel (g12 §1.1, §2.2, §6.1, §7.2-7.4, §8.3): *periodic* windows, zero-phase framing, root-Hann
  analysis + synthesis windows (WOLA), hop M/4 (the "robust to modification" hop of the §8.3 table).
* Phase vocoder with identity phase locking and transient handling, SOLA (g12 §9.7-9.8).
* Pitch shift = stretch + band-limited resample, cepstral formant correction (g12 §9.3, §9.8).
* Sines + noise model: QIFFT peaks, PARSHL tracking, McAulay-Quatieri cubic phase, Bark/ERB band
  noise (g12 §4.3, §9.4-9.6, §14.1, §17).
* Envelopes: cepstral lifter, LPC (existing Levinson), minimum phase by cepstral folding (g12 §3.2, §9.3).
* Channel vocoder with the Voder band edges, cross-synthesis (g12 §9.2, §16.1, §18 G.5).

Everything takes mono (n,) or stereo (n, 2) unless stated.
"""
from __future__ import annotations

import json
import math
import re
from fractions import Fraction

import numba
import numpy as np
from scipy.fft import next_fast_len
from scipy.ndimage import gaussian_filter, maximum_filter1d, median_filter, uniform_filter
from scipy.signal import correlate, hilbert, resample_poly

from . import osc as O
from .core import DEFAULT_SR, pad_to, read_wav, to_mono
from .fx import _per_channel, effect
from .notes import parse_pitch_list, to_hz
from .spec import layer_type

# g12 §14.1, verbatim Bark band edges (Hz) with JOS's extrapolation for high sample rates
BARK_EDGES = (0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150,
              3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500, 20500, 27000)
# g12 §18 footnote G.5, verbatim Voder / Dudley vocoder band edges (Hz)
VODER_EDGES = (0, 225, 450, 700, 1000, 1400, 2000, 2700, 3800, 5400, 7500)


# ==========================================================================================
# Kernel: windows, COLA check, zero-phase STFT / WOLA
# ==========================================================================================

def _win_len(ms: float, sr: int) -> int:
    """Power-of-two window length nearest to `ms` (>= 64)."""
    return 1 << max(6, int(round(math.log2(max(ms * 1e-3 * sr, 2.0)))))


def window(M: int, kind: str = "hann", root: bool = False) -> np.ndarray:
    """*Periodic* window (denominator M, g12 §2.2): only this form overlap-adds to an exact constant.
    `root=True` gives the square root (root-Hann = MLT sine window) for WOLA (g12 §7.4)."""
    t = 2 * np.pi * np.arange(M) / M
    if kind == "hann":
        w = 0.5 - 0.5 * np.cos(t)
    elif kind == "hamming":
        w = 25 / 46 - 21 / 46 * np.cos(t)
    elif kind == "blackman":
        w = 0.42 - 0.5 * np.cos(t) + 0.08 * np.cos(2 * t)
    elif kind == "rect":
        w = np.ones(M)
    else:
        raise ValueError(f"unknown window {kind!r} (hann|hamming|blackman|rect)")
    return np.sqrt(np.maximum(w, 0.0)) if root else w


def cola_error(w: np.ndarray, R: int) -> float:
    """Relative ripple of sum_m w(n - mR); 0 for a COLA window (g12 §7.2 test code)."""
    s = np.zeros(len(w) + 8 * R)
    for o in range(0, len(s) - len(w) + 1, R):
        s[o:o + len(w)] += w
    mid = s[len(w):-len(w)]
    return float(np.ptp(mid) / mid.mean())


def _fwd(fr: np.ndarray, N: int) -> np.ndarray:
    """Windowed frames (F, M) -> rfft of the zero-phase buffer (frame centre at index 0, zero
    padding in the middle; g12 §1.1), so a peak's phase is the sinusoid's phase at the frame centre."""
    M = fr.shape[1]
    h = M // 2
    buf = np.zeros((fr.shape[0], N))
    buf[:, :M - h] = fr[:, h:]
    buf[:, N - h:] = fr[:, :h]
    return np.fft.rfft(buf, axis=1)


def _inv(X: np.ndarray, M: int) -> np.ndarray:
    N = 2 * (X.shape[1] - 1)
    h = M // 2
    y = np.fft.irfft(X, N, axis=1)
    return np.concatenate([y[:, N - h:], y[:, :M - h]], axis=1)


def _ola(fr: np.ndarray, R: int) -> np.ndarray:
    F, M = fr.shape
    y = np.zeros((F - 1) * R + M)
    for i in range(F):
        y[i * R:i * R + M] += fr[i]
    return y


def stft(x: np.ndarray, M: int = 2048, R: int | None = None, N: int | None = None, w: np.ndarray | None = None) -> np.ndarray:
    """Mono x -> complex (frames, N/2+1). Frame m is centred on sample m*R - M/2 (the signal is
    padded by M at both ends so every sample is under the full overlap). Defaults: root-Hann,
    R = M/4, N = M; pass N = 2M when a per-frame filter is applied (N >= M + L - 1, g12 §7.1).
    # ponytail: all frames in memory (about 0.5 MB per second); process in chunks for minutes-long files."""
    R = R or M // 4
    N = N or M
    w = window(M, root=True) if w is None else w
    F = -(-(len(x) + M - 1) // R) + 1
    xp = np.zeros((F - 1) * R + M)
    xp[M:M + len(x)] = x
    return _fwd(xp[np.arange(M)[None, :] + (np.arange(F) * R)[:, None]] * w, N)


def istft(X: np.ndarray, n: int, M: int = 2048, R: int | None = None, w: np.ndarray | None = None,
          wa: np.ndarray | None = None) -> np.ndarray:
    """Inverse of `stft` (WOLA, g12 §7.4): synthesis window `w` (default root-Hann), gain
    R / sum(wa * w) (the Poisson-sum constant of g12 §7.2; exact when wa*w is COLA at R)."""
    R = R or M // 4
    w = window(M, root=True) if w is None else w
    wa = w if wa is None else wa
    return _ola(_inv(X, M) * w, R)[M:M + n] * (R / np.sum(wa * w))


def _wola(x, fn, M, N=None):
    """Mono/stereo x -> istft(fn(stft(x))) per channel."""
    return _per_channel(lambda s: istft(fn(stft(s, M, N=N)), len(s), M), x)


# ==========================================================================================
# Spectral envelopes
# ==========================================================================================

def cepstral_envelope(mag: np.ndarray, n_c: int) -> np.ndarray:
    """Cepstral-windowed envelope of magnitude spectra on an rfft grid (last axis), g12 §9.3(a):
    lifter 1 for |n| < n_c, 0.5 at n_c, 0 beyond; n_c (samples) must stay below the pitch period."""
    nfft = 2 * (mag.shape[-1] - 1)
    n_c = int(np.clip(n_c, 1, nfft // 2 - 1))
    c = np.fft.irfft(np.log(np.maximum(mag, 1e-5 * mag.max() + 1e-12)), nfft, axis=-1)
    lif = np.zeros(nfft)
    lif[:n_c] = 1
    lif[n_c] = 0.5
    lif[nfft - n_c + 1:] = 1
    lif[nfft - n_c] = 0.5
    return np.exp(np.fft.rfft(c * lif, nfft, axis=-1).real)


def lpc_envelope(mag: np.ndarray, order: int = 24) -> np.ndarray:
    """All-pole (peak-hugging, g12 §9.3(b)) envelope of magnitude spectra (frames, K): autocorrelation
    = IFFT of the power spectrum, then the existing Levinson recursion of `physical/analysis.py`."""
    from .physical.analysis import _levinson_frames
    mag = np.atleast_2d(mag)
    nfft = 2 * (mag.shape[1] - 1)
    r = np.fft.irfft(mag ** 2, nfft, axis=1)[:, :order + 1].copy()
    r[:, 0] = r[:, 0] * (1 + 1e-9) + 1e-30
    A, _, E = _levinson_frames(np.ascontiguousarray(r), order)
    return np.sqrt(np.maximum(E, 0.0))[:, None] / np.maximum(np.abs(np.fft.rfft(A, nfft, axis=1)), 1e-12)


def minimum_phase(G: np.ndarray, nfft: int) -> np.ndarray:
    """Magnitude on the rfft grid of nfft -> complex minimum-phase response (fold the real cepstrum
    onto positive time, g12 §3.2). nfft must be well above the filter length (cepstral aliasing)."""
    c = np.fft.irfft(np.log(np.maximum(G, 1e-9)), nfft)
    c[1:nfft // 2] *= 2.0
    c[nfft // 2 + 1:] = 0.0
    return np.exp(np.fft.rfft(c, nfft))


def _warp(E: np.ndarray, phi: float) -> np.ndarray:
    """Rows E(f) -> E(f / phi): every feature moves up by the ratio phi."""
    K = E.shape[-1]
    k = np.clip(np.arange(K) / phi, 0, K - 1)
    i0 = np.minimum(k.astype(int), K - 2)
    fr = k - i0
    return E[..., i0] * (1 - fr) + E[..., i0 + 1] * fr


def transfer_envelope(src: np.ndarray, dst: np.ndarray, sr: int = DEFAULT_SR, shift: float = 0.0, env_ms: float = 2.0) -> np.ndarray:
    """Give `dst` the spectral envelope of `src` (same length), frame by frame: divide by dst's own
    cepstral envelope, multiply by src's (g12 §9.2-9.3, §17 H.5). `shift` moves the imposed formants
    by semitones. N = 2M so the gain curve's impulse response fits (g12 §7.1).
    # ponytail: fixed 2 ms lifter (envelope detail down to 500 Hz, pitch up to ~450 Hz);
    # make it follow a pitch track if high voices need formant keep."""
    M = _win_len(40.0, sr)
    n_c = max(2, int(env_ms * 1e-3 * sr))

    def one(s, d):
        Xd = stft(d, M, N=2 * M)
        Es = cepstral_envelope(np.abs(stft(pad_to(s, len(d))[:len(d)], M, N=2 * M)), n_c)
        if shift:
            Es = _warp(Es, 2.0 ** (shift / 12.0))
        P = np.abs(Xd) ** 2
        g = np.clip(Es / cepstral_envelope(np.sqrt(P), n_c), 10 ** -1.5, 10 ** 1.5)
        g *= np.sqrt(P.sum(axis=1) / ((P * g * g).sum(axis=1) + 1e-30))[:, None]      # reshape the spectrum, keep each frame's energy
        return istft(Xd * g, len(d), M)

    if dst.ndim == 1:
        return one(to_mono(src), dst)
    return np.stack([one(src[:, c] if src.ndim == 2 else src, dst[:, c]) for c in range(dst.shape[1])], axis=1)


# ==========================================================================================
# Time-scale modification
# ==========================================================================================

def onsets(x: np.ndarray, sr: int = DEFAULT_SR, sensitivity: float = 1.0) -> np.ndarray:
    """Sample positions of attack transients: positive spectral flux of a 12 ms STFT relative to the
    level just before plus a quarter of the local maximum (g12 §9.6 asks for a transient detector and
    gives no numbers).
    # ponytail: full-band flux; a hit under an equally loud sustained sound is missed and a 10 Hz
    # beat is (correctly) not flagged. Per-band flux if drums buried in pads matter."""
    M = _win_len(12.0, sr)
    R = M // 4
    mag = np.abs(stft(to_mono(x), M, R))
    tot = mag.sum(axis=1)
    if tot.max() <= 0:
        return np.zeros(0, int)
    loc = maximum_filter1d(tot, 2 * int(0.3 * sr / R) + 1)
    rel = np.maximum(mag[1:] - mag[:-1], 0.0).sum(axis=1) / (tot[:-1] + 0.25 * loc[1:] + 1e-12)   # UNSOURCED: 0.25, 0.3 s
    thr = 0.5 / max(float(sensitivity), 1e-3)                                                    # UNSOURCED: tuned on clicks vs beats
    pk = (rel > thr) & (rel >= maximum_filter1d(rel, 9)) & (tot[1:] > 1e-3 * tot.max())
    return np.clip((np.flatnonzero(pk) + 1) * R - M // 2, 0, len(x) - 1)


def _time_map(x, sr, factor, M, transients):
    """Piecewise-linear output-time -> input-time map: transient intervals (onset +- M/2) are
    translated at rate 1, everything else is stretched so the total is exactly round(n*factor)
    (g12 §9.7: "translate transient intervals without stretching")."""
    n = x.shape[0]
    n_out = max(1, int(round(n * factor)))
    regs = []
    if transients:
        on = onsets(x, sr, 1.0 if isinstance(transients, bool) else float(transients))
        half = M // 2
        while len(on) and half >= 64:
            regs = []
            for o in on:
                a, b = max(0, int(o) - half), min(n, int(o) + half)
                if regs and a <= regs[-1][1]:
                    regs[-1][1] = b
                else:
                    regs.append([a, b])
            rigid = sum(b - a for a, b in regs)
            if n - rigid > 0 and (n_out - rigid) / (n - rigid) >= 0.5 * min(factor, 1.0):
                break
            regs, half = [], half // 2          # too many transients to keep rigid when compressing: shrink
    rigid = sum(b - a for a, b in regs)
    f_el = (n_out - rigid) / max(n - rigid, 1)
    ibp, obp, prev = [0.0], [0.0], 0
    for a, b in regs:
        ibp += [a, b]
        obp += [obp[-1] + (a - prev) * f_el, obp[-1] + (a - prev) * f_el + (b - a)]
        prev = b
    ibp.append(float(n))
    obp.append(obp[-1] + (n - prev) * f_el)
    return n_out, np.array(ibp), np.array(obp), [a for a, _ in regs]


def _map(t, ibp, obp):
    """Input position for output times t (rate 1 outside the signal)."""
    return np.interp(t, obp, ibp) + np.minimum(t, 0) + np.maximum(t - obp[-1], 0)


def _pv(x: np.ndarray, pos: np.ndarray, M: int, lock: bool = True, reset: np.ndarray | None = None) -> np.ndarray:
    """Phase vocoder core (WOLA, root-Hann, synthesis hop R = M/4; g12 §9.7-9.8, reconstructed there
    and verified in tests/test_spectral.py). Output frame i takes its magnitudes from the input frame
    centred on sample pos[i]; its phase is the previous output phase plus the phase advance measured
    between the input frames at pos[i] - R and pos[i] ("horizontal" coherence, with no unwrapping
    ambiguity because that advance is over exactly one synthesis hop). `lock`: identity phase locking
    (Laroche-Dolson) - bins keep their measured phase relative to the nearest spectral peak
    ("vertical" coherence around peaks). `reset[i]`: frame i takes the original phases (transient).
    Returns the overlap-add buffer; output time t is at index t + M when frame i is meant for time (i-2)R."""
    R = M // 4
    w = window(M, root=True)
    P = 2 * M
    xp = np.concatenate([np.zeros(P), x, np.zeros(P)])
    st = np.clip(pos, -M, len(x) + M) + P - M // 2
    ar = np.arange(M)[None, :]
    A = _fwd(xp[st[:, None] + ar] * w, M)
    B = _fwd(xp[(st - R)[:, None] + ar] * w, M)
    mag, phA = np.abs(A), np.angle(A)
    d = phA - np.angle(B)
    F, K = mag.shape
    bins = np.arange(K)
    if lock:                                       # peaks = maxima over +-2 bins (Laroche-Dolson)
        c = mag[:, 2:-2]
        pkm = np.zeros(mag.shape, bool)
        pkm[:, 2:-2] = (c > mag[:, 1:-3]) & (c > mag[:, :-4]) & (c >= mag[:, 3:-1]) & (c >= mag[:, 4:])
    ph = np.empty_like(phA)
    cur = phA[0].copy()
    ph[0] = cur
    for i in range(1, F):
        if reset is not None and reset[i]:
            cur = phA[i].copy()
        else:
            cur = cur + d[i]
            if lock:
                pk = np.flatnonzero(pkm[i])
                if len(pk):
                    own = pk[np.searchsorted((pk[:-1] + pk[1:]) // 2 + 1, bins, side="right")]
                    cur = cur[own] + phA[i] - phA[i][own]
        ph[i] = cur
    return _ola(_inv(mag * np.exp(1j * ph), M) * w, R) / 2.0          # sum of root-Hann^2 at M/4 = 2 (tested)


def _wsola(x, ibp, obp, n_out, M, sr):
    """Synchronous overlap-add with a similarity search (SOLA-FS / WSOLA family, g12 §9.8): Hann
    frames at a fixed synthesis hop M/2; each input frame is taken where it best continues the
    previous one (maximum normalised cross-correlation) within +-12 ms of the mapped position. Inside
    a rate-1 (transient) interval the natural continuation is in range, so a transient is copied
    once, never repeated or dropped. Stereo shares one offset per frame."""
    x2 = x[:, None] if x.ndim == 1 else x
    n = x2.shape[0]
    H = M // 2
    D = min(H, int(0.012 * sr))                                       # UNSOURCED: +-12 ms search (covers a 42 Hz period)
    P = M + D
    xp = np.concatenate([np.zeros((P, x2.shape[1])), x2, np.zeros((P, x2.shape[1]))])
    mono = xp.mean(axis=1)
    w = window(M)[:, None]
    F = -(-n_out // H) + 2
    y = np.zeros(((F - 1) * H + M, x2.shape[1]))
    tgt = np.clip(np.round(_map((np.arange(F) - 1.0) * H, ibp, obp)).astype(int), -H, n + H) + P - H
    prev = -1
    for k in range(F):
        s = int(tgt[k])
        if prev >= 0:
            seg, tm = mono[s - D:s + D + M], mono[prev + H:prev + H + M]
            if len(tm) == M and len(seg) == M + 2 * D:
                cs = np.concatenate([[0.0], np.cumsum(seg * seg)])
                s = s - D + int(np.argmax(correlate(seg, tm, "valid") / np.sqrt(cs[M:] - cs[:-M] + 1e-12)))
        y[k * H:k * H + M] += xp[s:s + M] * w
        prev = s
    y = y[M:M + n_out]
    return y[:, 0] if x.ndim == 1 else y


def stretch(x: np.ndarray, sr: int = DEFAULT_SR, factor: float = 1.0, mode: str = "vocoder", window_ms: float = 0.0,
            lock: bool = True, transients=True) -> np.ndarray:
    """Time-stretch by `factor` (> 1 longer) at constant pitch; output length is round(n * factor)."""
    factor = float(np.clip(factor, 0.1, 10.0))
    if x.shape[0] < 16 or abs(factor - 1.0) < 1e-9:
        return x.copy()
    if mode not in ("vocoder", "sola"):
        raise ValueError(f"unknown timestretch mode {mode!r} (vocoder|sola)")
    M = _win_len(window_ms or (45.0 if mode == "vocoder" else 25.0), sr)   # 45 ms: g12 §9.8 vocoder settings
    n_out, ibp, obp, starts = _time_map(to_mono(x), sr, factor, M, transients)
    if mode == "sola":
        return _wsola(x, ibp, obp, n_out, M, sr)
    R = M // 4
    F = -(-n_out // R) + 5
    pos = np.round(_map((np.arange(F) - 2.0) * R, ibp, obp)).astype(int)
    reset = np.zeros(F, bool)
    for a in starts:
        i = int(np.searchsorted(pos, a))
        if 0 < i < F:
            reset[i] = True
    return _per_channel(lambda s: _pv(s, pos, M, bool(lock), reset)[M:M + n_out], x)


def shift_pitch(x: np.ndarray, sr: int = DEFAULT_SR, semitones: float = 0.0, formant: str = "shift", mode: str = "vocoder") -> np.ndarray:
    """Pitch shift at constant length: stretch by the ratio, then band-limited resampling by its
    inverse (g12 §9.8). formant="keep" re-imposes the original spectral envelope (no "munchkinization")."""
    r = Fraction(2.0 ** (float(np.clip(semitones, -36, 36)) / 12.0)).limit_denominator(512)
    p, q = r.numerator, r.denominator
    n = x.shape[0]
    if p == q or n < 16:
        return x.copy()
    y = pad_to(resample_poly(stretch(x, sr, p / q, mode), q, p, axis=0), n)[:n]
    return transfer_envelope(x, y, sr) if formant == "keep" else y


# ==========================================================================================
# Sines + noise model
# ==========================================================================================

def band_edges(sr: int, kind: str = "bark") -> np.ndarray:
    """Noise-band edges in Hz up to sr/2: the Bark table, or one band per ERB
    (ERBS(f) = 21.4 log10(0.00437 f + 1), g12 §14.1)."""
    if kind == "erb":
        f = (10 ** (np.arange(0, 21.4 * math.log10(0.00437 * sr / 2 + 1)) / 21.4) - 1) / 0.00437
    elif kind == "bark":
        f = np.array(BARK_EDGES, float)
    else:
        raise ValueError(f"unknown bands {kind!r} (bark|erb)")
    return np.append(f[f < 0.49 * sr], sr / 2.0)


@numba.njit(cache=True)
def _osc_bank(freq, amp, phase, pos, n, sr, cubic):
    """Additive resynthesis of tracks (frames, tracks) sampled at output samples pos[m]. Amplitude is
    linear between frames. cubic: McAulay-Quatieri cubic phase through the measured phases and
    frequencies (g12 §17 H.4, reconstructed there; the end conditions are checked in the tests).
    Otherwise "magnitude-only": linear frequency, running phase."""
    F, T = freq.shape
    y = np.zeros(n)
    two_pi = 2.0 * np.pi
    for t in range(T):
        run = False
        th = 0.0
        for m in range(F - 1):
            a0 = amp[m, t]
            a1 = amp[m + 1, t]
            S = pos[m + 1] - pos[m]
            if (a0 == 0.0 and a1 == 0.0) or S <= 0:
                run = False
                continue
            w0 = two_pi * freq[m, t] / sr
            w1 = two_pi * freq[m + 1, t] / sr
            if cubic:
                th0 = phase[m, t]
                th1 = phase[m + 1, t]
                Mq = np.round(((th0 + w0 * S - th1) + (w1 - w0) * S / 2.0) / two_pi)
                D = th1 - th0 - w0 * S + two_pi * Mq
                ca = 3.0 * D / (S * S) - (w1 - w0) / S
                cb = -2.0 * D / (S * S * S) + (w1 - w0) / (S * S)
            else:
                if not run:
                    th = phase[m, t]
                    run = True
                th0 = th
                ca = (w1 - w0) / (2.0 * S)
                cb = 0.0
                th = th + 0.5 * (w0 + w1) * S
            lo = max(0, -pos[m])
            hi = min(S, n - pos[m])
            for i in range(lo, hi):
                y[pos[m] + i] += (a0 + (a1 - a0) * i / S) * np.cos(th0 + w0 * i + ca * i * i + cb * i * i * i)
    return y


def analyze(x: np.ndarray, sr: int = DEFAULT_SR, window_ms: float = 46.0, thresh_db: float = -50.0, max_peaks: int = 60,
            max_tracks: int = 40, bands: str = "bark", min_frames: int = 4) -> dict:
    """Sines + noise analysis (Serra's SMS with the PARSHL recipe; g12 §9.5, §17) of a mono signal.

    Returns a plain dict of numpy arrays / numbers (see `save_model`):
      sr, n (samples), hop, win; frame m is centred on sample m*hop - win/2
      freq, amp, phase : (frames, tracks) partial frequency in Hz, linear amplitude, cosine phase at
                         the frame centre; amp == 0 where a track is silent
      noise            : (frames, bands) residual level per band, as the standard deviation of white
                         noise with the same spectral density;  edges : (bands + 1,) Hz

    Peaks: periodic Hamming window (the PARSHL default, §17 H.8; main lobe 4 bins, so harmonics are
    resolved when the window holds 4 periods: f0 >= 4000 / window_ms Hz, §4.2 - use a longer window
    for bass), zero padding 2, parabola through the three dB samples around each maximum (QIFFT,
    §4.3), the same parabola on the real and imaginary parts for the phase (§17 H.2). A peak must
    have roughly the window's own curvature (§4.3: "a deviant value flags a non-sinusoidal peak");
    that also rejects the -42 dB side lobes, which are half as wide (PARSHL's MinWid).
    Tracking (§17 H.3, H.8): closest peak within DFmax = MinSep/2 = 2 sr / M Hz per frame, the
    closer track wins a conflict, unmatched tracks ramp to zero over one hop, unmatched peaks are
    born at zero amplitude one frame earlier (the first frame starts at full level), 60 peaks and
    40 tracks. Residual: magnitude spectrum minus that of the resynthesised sines (§9.5), summed
    per Bark or ERB band (§9.6).
    Not modelled: transients (g12 §9.6) - an attack is spread over sines and noise."""
    x = to_mono(np.asarray(x, float))
    n = len(x)
    M = _win_len(window_ms, sr)
    N, R = 2 * M, M // 4
    w = window(M, "hamming")
    X = stft(x, M, R, N, w)
    F, K = X.shape
    db = 20 * np.log10(np.abs(X) + 1e-12)
    c = db[:, 1:-1]
    mask = ((c > db[:, :-2]) & (c >= db[:, 2:]) & (c > db.max(axis=1, keepdims=True) + thresh_db)
            & (c > db.max() - 80.0)                                   # UNSOURCED: absolute floor so silent frames give no tracks
            & (np.arange(1, K - 1) * sr / N > sr / 1000.0)[None, :])  # PARSHL Fc1 = Fs/1000
    fi, ki = np.nonzero(mask)
    ki = ki + 1
    a, b, g = db[fi, ki - 1], db[fi, ki], db[fi, ki + 1]
    curv = 0.5 * (a - 2 * b + g)
    p = np.clip(0.5 * (a - g) / np.minimum(a - 2 * b + g, -1e-12), -0.5, 0.5)
    Wd = 20 * np.log10(np.abs(np.fft.rfft(w, N)[:2]))
    cw = Wd[1] - Wd[0]                                                # the window's own half-curvature (dB / bin^2, < 0)
    ok = (curv < 0.25 * cw) & (curv > 2.5 * cw)                       # UNSOURCED: tolerance factor on the curvature test

    def par(v):
        return v[fi, ki] + 0.5 * (v[fi, ki + 1] - v[fi, ki - 1]) * p + 0.5 * (v[fi, ki - 1] - 2 * v[fi, ki] + v[fi, ki + 1]) * p * p

    pf = ((ki + p) * sr / N)[ok]
    pa = (10 ** ((b - 0.25 * (a - g) * p) / 20) * 2.0 / w.sum())[ok]  # peak height = A sum(w)/2 (g12 §4.1)
    pp = np.arctan2(par(X.imag), par(X.real))[ok]
    bounds = np.searchsorted(fi[ok], np.arange(F + 1))
    dfmax = 2.0 * sr / M
    hopw = 2 * np.pi * R / sr
    GAP = 2                                                           # UNSOURCED: a silent track can be picked up again for 2 frames
    cap = 64
    fr, am, ph = np.zeros((F, cap)), np.zeros((F, cap)), np.zeros((F, cap))
    last, dead = np.zeros(cap), np.full(cap, 10 ** 9)                 # last frequency; frames since last matched
    for m in range(F):
        f_m, a_m, p_m = pf[bounds[m]:bounds[m + 1]], pa[bounds[m]:bounds[m + 1]], pp[bounds[m]:bounds[m + 1]]
        if len(f_m) > max_peaks:
            top = np.sort(np.argsort(-a_m)[:max_peaks])
            f_m, a_m, p_m = f_m[top], a_m[top], p_m[top]
        cand = np.flatnonzero(dead <= GAP)
        used = np.zeros(len(f_m), bool)
        hit = np.zeros(cap, bool)
        if len(cand) and len(f_m):
            Dm = np.abs(last[cand][:, None] - f_m[None, :])
            ti, pj = np.nonzero(Dm < dfmax)
            for q in np.argsort(Dm[ti, pj], kind="stable"):
                t, j = cand[ti[q]], pj[q]
                if hit[t] or used[j]:
                    continue
                hit[t] = used[j] = True
                if m and am[m - 1, t] == 0.0:                         # picked up again: no glide-in
                    fr[m - 1, t], ph[m - 1, t] = f_m[j], p_m[j] - hopw * f_m[j]
                fr[m, t], am[m, t], ph[m, t], last[t] = f_m[j], a_m[j], p_m[j], f_m[j]
        lost = cand[~hit[cand]]
        fr[m, lost] = last[lost]                                       # amp stays 0: ramps down over this hop
        if m:
            ph[m, lost] = ph[m - 1, lost] + hopw * last[lost]
        dead[lost] += 1
        dead[hit] = 0
        live = int(hit.sum())
        for j in np.argsort(-a_m, kind="stable"):
            if live >= max_tracks:
                break
            if used[j]:
                continue
            free = np.flatnonzero(dead > GAP)
            if not len(free):
                fr, am, ph = (np.concatenate([v, np.zeros((F, cap))], axis=1) for v in (fr, am, ph))
                last, dead, hit = np.concatenate([last, np.zeros(cap)]), np.concatenate([dead, np.full(cap, 10 ** 9)]), np.concatenate([hit, np.zeros(cap, bool)])
                free, cap = np.array([cap]), 2 * cap
            t = free[0]
            if m:
                fr[m - 1, t], ph[m - 1, t] = f_m[j], p_m[j] - hopw * f_m[j]
            fr[m, t], am[m, t], ph[m, t], last[t], dead[t] = f_m[j], a_m[j], p_m[j], f_m[j], 0
            live += 1
    on = np.concatenate([np.zeros((1, cap), bool), am > 0, np.zeros((1, cap), bool)]).astype(np.int8)
    for t in range(cap):                                               # drop runs shorter than min_frames (anti-"burbling")
        e = np.flatnonzero(np.diff(on[:, t]))
        for s0, s1 in zip(e[::2], e[1::2]):
            if s1 - s0 < min_frames:
                am[s0:s1, t] = 0.0
    keep = am.any(axis=0)
    fr, am, ph = fr[:, keep], am[:, keep], ph[:, keep]

    pos = (np.arange(F) * R - M // 2).astype(np.int64)
    S = stft(_osc_bank(fr, am, ph, pos, n, float(sr), True), M, R, N, w)
    res = np.maximum(np.abs(X) - np.abs(S), 0.0) ** 2
    edges = band_edges(sr, bands)
    kb = np.round(edges[:-1] * N / sr).astype(int)
    cnt = np.diff(np.append(kb, K))
    noise = np.sqrt(np.add.reduceat(res, kb, axis=1) / cnt / np.sum(w * w))    # E|X|^2 = sigma^2 sum(w^2) for white noise
    return {"sr": int(sr), "n": int(n), "hop": int(R), "win": int(M), "freq": fr, "amp": am, "phase": ph, "noise": noise, "edges": edges}


def resynth(model: dict, sr: int | None = None, stretch: float = 1.0, transpose: float = 0.0, formant: float = 0.0,
            sines_gain: float = 1.0, noise_gain: float = 1.0, seed: int = 0, phases: str = "auto") -> np.ndarray:
    """Render a model from `analyze` (mono), at any sample rate.

    stretch: duration factor (the envelopes are resampled; g12 §17 H.5). transpose: semitones applied
    to the partial frequencies. formant: semitones the spectral envelope moves - 0 with a transpose
    keeps the formants in place (partial amplitudes are re-read from the piecewise-linear envelope
    through the frame's peaks, g12 §9.3), formant == transpose is a plain "tape" transposition;
    the noise bands move with `formant`. phases: "cubic" follows the measured phases (waveform
    preserved), "free" integrates frequency, "auto" = cubic only when stretch == 1 and transpose == 0.
    Noise: random-phase spectra shaped by the band levels, root-Hann at 75 % overlap (g12 §9.5)."""
    msr = int(model["sr"])
    sr = int(sr or msr)
    R, M = int(model["hop"]), int(model["win"])
    stretch = float(np.clip(stretch, 0.05, 50.0))
    fr, am, ph = (np.array(model[k], float) for k in ("freq", "amp", "phase"))
    F = fr.shape[0]
    n_out = max(1, int(round(int(model["n"]) / msr * stretch * sr)))
    rho, phi = 2.0 ** (float(transpose) / 12.0), 2.0 ** (float(formant) / 12.0)
    y = np.zeros(n_out)
    if sines_gain and fr.size:
        if abs(rho - phi) > 1e-9:
            am = am.copy()
            for m in range(F):
                act = np.flatnonzero(am[m] > 0)
                if len(act) >= 2:
                    o = act[np.argsort(fr[m, act])]
                    am[m, act] = 10 ** (np.interp(fr[m, act] * rho / phi, fr[m, o], 20 * np.log10(am[m, o])) / 20)
        fr = fr * rho
        am = np.where(fr < 0.48 * sr, am, 0.0)
        pos = np.round((np.arange(F) * R - M // 2) / msr * stretch * sr).astype(np.int64)
        cubic = phases == "cubic" or (phases == "auto" and stretch == 1.0 and rho == 1.0)
        y += sines_gain * _osc_bank(fr, am, ph, pos, n_out, float(sr), cubic)
    nz, edges = np.array(model["noise"], float), np.array(model["edges"], float)
    if noise_gain and nz.size:
        Mn = _win_len(23.0, sr)
        Rn = Mn // 4
        Fn = -(-n_out // Rn) + 5
        mi = np.clip(((np.arange(Fn) - 2.0) * Rn / (stretch * sr) * msr + M // 2) / R, 0, F - 1)
        i0 = np.minimum(mi.astype(int), max(F - 2, 0))
        i1 = np.minimum(i0 + 1, F - 1)
        Bn = nz[i0] * (1 - (mi - i0))[:, None] + nz[i1] * (mi - i0)[:, None]
        fk = np.arange(Mn // 2 + 1) * sr / Mn / phi
        fc = 0.5 * (edges[:-1] + edges[1:])
        Wm = np.stack([np.interp(fk, fc, e) for e in np.eye(len(fc))], axis=1) * (fk <= edges[-1])[:, None]
        Y = (Bn @ Wm.T) * math.sqrt(Mn / 2.0) * np.exp(1j * np.random.default_rng(seed).uniform(-np.pi, np.pi, (Fn, Mn // 2 + 1)))
        Y[:, 0] = 0.0
        y += noise_gain * _ola(_inv(Y, Mn) * window(Mn, root=True), Rn)[Mn:Mn + n_out]
    return y


def save_model(model: dict, path: str) -> None:
    """Write a model as .npz (compact) or .json."""
    if str(path).lower().endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in model.items()}, f)
    else:
        np.savez_compressed(path, **model)


def load_model(path: str) -> dict:
    if str(path).lower().endswith(".json"):
        with open(path, encoding="utf-8") as f:
            return {k: (np.array(v, float) if isinstance(v, list) else v) for k, v in json.load(f).items()}
    with np.load(path) as z:
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


# ==========================================================================================
# Registered effects
# ==========================================================================================

def _floats(v) -> list[float]:
    if isinstance(v, (int, float)):
        return [float(v)]
    if isinstance(v, str):
        return [float(s) for s in re.split(r"[\s,;]+", v.strip()) if s]
    return [float(s) for s in v]


def _rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


@effect("timestretch", "Change length without changing pitch (phase vocoder, or SOLA for speech and drums).",
        factor=(1.0, "length ratio 0.1..10: 2 = twice as long"), mode=("vocoder", "vocoder (tonal, music) | sola (speech, drums, single voices)"),
        window_ms=(0.0, "analysis window in ms; 0 = auto (45 vocoder, 25 sola). Longer = cleaner bass, more smear"),
        lock=(True, "vocoder: lock bin phases to spectral peaks (less 'phasiness')"),
        transients=(True, "keep attacks unstretched; a number is the detector sensitivity (1 = default, 2 = more attacks)"))
def timestretch(x, sr=DEFAULT_SR, factor=1.0, mode="vocoder", window_ms=0.0, lock=True, transients=True):
    return stretch(x, sr, factor, mode, float(window_ms), lock, transients)


@effect("pitch_shift", "Pitch shift at the same length (phase vocoder + band-limited resampling), optionally keeping the formants.",
        semitones=(0.0, "+/- semitones (-36..36)"), formant=("shift", "shift = formants move with the pitch (chipmunk / giant) | keep = timbre stays (voices, instruments)"),
        fine=(0.0, "extra shift in cents"))
def pitch_shift(x, sr=DEFAULT_SR, semitones=0.0, formant="shift", fine=0.0):
    return shift_pitch(x, sr, float(semitones) + float(fine) / 100.0, formant)


@effect("harmonizer", "Add pitch-shifted copies at musical intervals (formant-preserving).",
        intervals=("4 7", "semitone offsets, e.g. '4 7' (major triad), '-12', [3, 7]"), mix=(0.5, "0 = dry .. 1 = only the added voices"),
        formant=("keep", "keep | shift (see pitch_shift)"))
def harmonizer(x, sr=DEFAULT_SR, intervals="4 7", mix=0.5, formant="keep"):
    iv = _floats(intervals)
    if not iv:
        return x
    return x * (1 - mix) + sum(shift_pitch(x, sr, s, formant) for s in iv) * (mix / len(iv))


def _voc_edges(bands: int, sr: int) -> np.ndarray:
    if int(bands) == 10 and sr / 2 > VODER_EDGES[-1]:
        return np.array(VODER_EDGES, float)
    e = lambda f: 21.4 * math.log10(0.00437 * f + 1)                   # ERB-rate spacing (g12 §14.1)
    return (10 ** (np.linspace(e(60.0), e(min(0.45 * sr, 12000.0)), max(2, int(bands)) + 1) / 21.4) - 1) / 0.00437


def _carrier(kind, n, sr, freq, notes, seed):
    if kind == "noise":
        return O.white(n, seed)
    if kind not in ("saw", "pulse", "chord"):
        raise ValueError(f"unknown vocoder carrier {kind!r} (saw|pulse|noise|chord)")
    fs = parse_pitch_list(notes or "C3 G3 C4 E4") if (notes or kind == "chord") else [to_hz(freq)]
    gen = (lambda f: O.pulse(f, n, sr, width=0.1)) if kind == "pulse" else (lambda f: O.saw(f, n, sr))
    return sum(gen(f) for f in fs) / len(fs)


@effect("vocoder", "Channel vocoder: the sound (modulator, e.g. speech) shapes a synthetic carrier band by band - the robot / talking-synth voice.",
        carrier=("saw", "saw (buzz) | pulse (thin buzz) | noise (whisper) | chord (saws on `notes`)"), freq=(110.0, "carrier pitch: Hz or a note name"),
        notes=("", "carrier chord, e.g. 'C3 G3 C4 E4' (replaces freq)"), bands=(10, "number of bands; 10 = the classic Voder bands 0-7500 Hz, more = clearer"),
        attack=(0.005, "band envelope attack, seconds"), release=(0.03, "band envelope release, seconds (longer = smoother, less articulate)"),
        formant=(0.0, "semitones the band pattern is moved on the carrier (+ smaller / - bigger speaker)"),
        hiss=(0.05, "noise mixed into the carrier 0..1 (consonants need some)"), seed=(0, "noise seed"))
def vocoder(x, sr=DEFAULT_SR, carrier="saw", freq=110.0, notes="", bands=10, attack=0.005, release=0.03, formant=0.0, hiss=0.05, seed=0):
    """Dudley channel vocoder (g12 §16.1) as an FFT filter bank (§9.10): band energies of the
    modulator STFT, smoothed by an attack/release follower, scale the same bands of the carrier
    STFT after flattening the carrier (dividing by its own band level, §9.2)."""
    n = x.shape[0]
    if n < 16:
        return x
    M = _win_len(25.0, sr)
    N, R = 2 * M, M // 4
    edges = _voc_edges(bands, sr)
    B = len(edges) - 1
    f = np.arange(N // 2 + 1) * sr / N
    eye = np.vstack([np.eye(B), np.zeros((1, B))])                     # row B = "no band"
    band = lambda fq: np.where((fq >= edges[0]) & (fq < edges[-1]), np.searchsorted(edges, fq, "right") - 1, B)
    ia, isy = band(f), band(f / 2.0 ** (float(formant) / 12.0))
    car = _carrier(carrier, n, sr, freq, notes, int(seed))
    car = car + float(hiss) * _rms(car) * math.sqrt(3.0) * O.white(n, int(seed) + 1)
    Xc = stft(car, M, N=N)
    Pc = (np.abs(Xc) ** 2) @ eye[isy]
    ca, cr = math.exp(-R / (sr * max(float(attack), 1e-4))), math.exp(-R / (sr * max(float(release), 1e-4)))

    def one(s):
        E = np.sqrt((np.abs(stft(s, M, N=N)) ** 2) @ eye[ia])
        cur = np.zeros(B)
        for m in range(E.shape[0]):
            k = np.where(E[m] > cur, ca, cr)
            cur = k * cur + (1 - k) * E[m]
            E[m] = cur
        g = E / np.sqrt(Pc + 1e-6 * Pc.max() + 1e-30)
        return istft(Xc * np.hstack([g, np.zeros((len(g), 1))])[:, isy], n, M)

    return _per_channel(one, x)


@effect("cross_synth", "Cross-synthesis: put this sound's moving spectral envelope (its 'vowels') on another sound (talking wind, rain, engine...).",
        carrier=(None, "path of a WAV file that is given the envelope (looped to length); omitted = white noise"),
        flatten=(True, "remove the carrier's own envelope first"), smooth=(2.0, "envelope detail: cepstral cut-off in ms, below the pitch period of both sounds"),
        mix=(1.0, "wet amount"), seed=(0, "noise seed when no carrier file is given"))
def cross_synth(x, sr=DEFAULT_SR, carrier=None, flatten=True, smooth=2.0, mix=1.0, seed=0):
    """Y = Xc / env(Xc) * env(Xm), cepstral envelopes, carrier phase kept (g12 §9.2)."""
    n = x.shape[0]
    if n < 16:
        return x
    if carrier:
        c, csr = read_wav(carrier)
        c = to_mono(c)
        if csr != sr:
            r = Fraction(sr, csr)
            c = resample_poly(c, r.numerator, r.denominator)
        c = np.resize(c, n) if len(c) else np.zeros(n)
    else:
        c = O.white(n, int(seed))
    M = _win_len(40.0, sr)
    n_c = max(2, int(float(smooth) * 1e-3 * sr))
    Xc = stft(c, M, N=2 * M)
    if flatten:
        Xc = Xc / np.maximum(cepstral_envelope(np.abs(Xc), n_c), 1e-4 * np.abs(Xc).max() + 1e-12)
    wet = _per_channel(lambda s: istft(Xc * cepstral_envelope(np.abs(stft(s, M, N=2 * M)), n_c), n, M), x)
    if not flatten:
        wet = wet * (_rms(x) / (_rms(wet) + 1e-12))
    return x * (1 - mix) + wet * mix


@effect("spectral_gate", "Spectral gate / denoiser: attenuate the quiet bins of the spectrum; `keep` splits tonal from noisy content.",
        threshold=(-40.0, "dB relative to the loudest spectral bin; bins below it are turned down"), reduction=(30.0, "attenuation in dB"),
        smoothing=(1.0, "gain smoothing over time and frequency 0..4 (0 = hard, 'musical noise')"),
        keep=("all", "all = plain gate | tonal = only spectral peaks above the threshold (removes hiss, breath) | noise = everything but the peaks"))
def spectral_gate(x, sr=DEFAULT_SR, threshold=-40.0, reduction=30.0, smoothing=1.0, keep="all"):
    """Per-bin gain 1 where the signal exceeds the threshold, else `reduction` (g12 §5 denoising
    recipe), N = 2M (§7.1), gains smoothed over about 16 bin x frame cells per unit of `smoothing`
    (§12: band-power variance). Tonal bins = those well above the local median of the frame
    spectrum (a noise-floor estimate in the spirit of `findpeaks`' residual, §15)."""
    if keep not in ("all", "tonal", "noise"):
        raise ValueError(f"unknown keep {keep!r} (all|tonal|noise)")
    M = _win_len(46.0, sr)
    floor = 10 ** (-abs(float(reduction)) / 20)
    sm = max(0, int(round(float(smoothing))))

    def fn(X):
        mag = np.abs(X)
        m = mag > mag.max() * 10 ** (float(threshold) / 20)
        if keep != "all":
            span = 2 * int(round(100.0 * 2 * M / sr)) + 1                 # UNSOURCED: +-100 Hz median span
            ton = maximum_filter1d(m & (mag > 4.0 * median_filter(mag, size=(1, span))), 5, axis=1)   # UNSOURCED: 12 dB over the floor
            m = ton if keep == "tonal" else ~ton
        g = floor + (1 - floor) * m
        return X * (uniform_filter(g, size=(1 + 2 * sm, 1 + 4 * sm)) if sm else g)

    return _wola(x, fn, M, 2 * M)


@effect("sines_only", "Keep only the sinusoidal (pitched) part of the sound: sines + noise analysis, resynthesised partials.",
        thresh=(-50.0, "peak threshold in dB below each frame's loudest partial"))
def sines_only(x, sr=DEFAULT_SR, thresh=-50.0):
    return _per_channel(lambda s: resynth(analyze(s, sr, thresh_db=float(thresh)), noise_gain=0.0), x)


@effect("noise_only", "Keep only the noisy residual of the sound (breath, hiss, scrape) as Bark-band noise.",
        thresh=(-50.0, "peak threshold in dB below each frame's loudest partial"), seed=(0, "noise seed"))
def noise_only(x, sr=DEFAULT_SR, thresh=-50.0, seed=0):
    return _per_channel(lambda s: resynth(analyze(s, sr, thresh_db=float(thresh)), sines_gain=0.0, seed=int(seed)), x)


@effect("freeze", "Hold the sound at one instant for a while (spectral freeze), then let it continue.",
        at=(0.2, "time of the frozen instant, seconds"), length=(2.0, "seconds of frozen sound inserted"),
        jitter=(20.0, "ms of random read-position jitter around `at` (0 = static, glassy; more = alive)"), seed=(0, "jitter seed"))
def freeze(x, sr=DEFAULT_SR, at=0.2, length=2.0, jitter=20.0, seed=0):
    """Phase vocoder whose read position stops at `at` (magnitudes held, each bin's phase advanced
    by its measured instantaneous frequency, peak-locked; g12 §8.1, §9.7)."""
    n = x.shape[0]
    L = int(max(0.0, float(length)) * sr)
    if n < 16 or L == 0:
        return x
    M = _win_len(45.0, sr)
    R = M // 4
    a = int(np.clip(float(at) * sr, 0, n - 1))
    F = -(-(n + L) // R) + 5
    t = (np.arange(F) - 2) * R
    j = np.random.default_rng(int(seed)).uniform(-1, 1, F) * float(jitter) * 1e-3 * sr
    pos = np.where(t < a, t, np.where(t < a + L, np.clip(a + j, 0, n - 1), t - L))
    pos = np.round(pos).astype(int)
    return _per_channel(lambda s: _pv(s, pos, M)[M:M + n + L], x)


@effect("spectral_blur", "Smear the spectrum in time (and optionally frequency): washes attacks into a pad-like sustain.",
        time=(0.08, "blur in seconds (std of a Gaussian over time)"), freq=(0.0, "blur in Hz across frequency"), seed=(0, "phase seed for the smeared-in parts"))
def spectral_blur(x, sr=DEFAULT_SR, time=0.08, freq=0.0, seed=0):
    M = _win_len(46.0, sr)
    rng = np.random.default_rng(int(seed))

    def fn(X):
        mag = np.abs(X)
        bl = gaussian_filter(mag, sigma=(max(0.0, float(time)) * sr / (M // 4), max(0.0, float(freq)) * M / sr))
        ph = np.where(mag > 0.1 * bl, np.angle(X), rng.uniform(-np.pi, np.pi, X.shape))   # no phase to keep where the sound was silent
        return bl * np.exp(1j * ph)

    return _wola(x, fn, M)


@effect("freqshift", "Frequency shifter: every component moves by the same number of Hz (inharmonic, metallic; small values = slow phasing).",
        hz=(100.0, "shift in Hz, negative = down"))
def freqshift(x, sr=DEFAULT_SR, hz=100.0):
    """Single-sideband modulation of the analytic signal (g12 §3.1); the FFT Hilbert transform is
    circular, so the signal is zero-padded first."""
    n = x.shape[0]
    pad = min(n, sr // 10)
    rot = np.exp(2j * np.pi * float(hz) * np.arange(n) / sr)

    def one(s):
        sp = np.concatenate([np.zeros(pad), s, np.zeros(pad)])
        return np.real(hilbert(sp, next_fast_len(len(sp)))[pad:pad + n] * rot)

    return _per_channel(one, x)


@effect("robotize", "Monotone robot voice: every frame is made zero-phase, so the pitch becomes the frame rate.",
        pitch=(110.0, "robot pitch: Hz or a note name"))
def robotize(x, sr=DEFAULT_SR, pitch=110.0):
    R = max(8, int(round(sr / float(np.clip(to_hz(pitch), 20.0, sr / 16)))))
    M = max(512, 1 << int(math.ceil(math.log2(4 * R))))
    w = window(M, root=True)

    def one(s):
        y = istft(np.abs(stft(s, M, R, w=w)).astype(complex), len(s), M, R, w)
        return y * min(_rms(s) / (_rms(y) + 1e-12), 1.5 * np.max(np.abs(s)) / (np.max(np.abs(y)) + 1e-12))

    return _per_channel(one, x)


@effect("whisperize", "Whisper: the phase of every short frame is randomised, which removes the pitch and keeps the formants.",
        window_ms=(12.0, "frame length in ms (shorter = breathier, longer = more of the pitch survives)"), seed=(0, "phase seed"))
def whisperize(x, sr=DEFAULT_SR, window_ms=12.0, seed=0):
    rng = np.random.default_rng(int(seed))
    return _wola(x, lambda X: np.abs(X) * np.exp(1j * rng.uniform(-np.pi, np.pi, X.shape)), _win_len(float(window_ms), sr))


@effect("spectral_tilt", "Tilt the whole spectrum by a constant slope (zero-phase): negative = darker / warmer, positive = brighter.",
        slope=(-3.0, "dB per octave"), pivot=(1000.0, "frequency that keeps its level, Hz"))
def spectral_tilt(x, sr=DEFAULT_SR, slope=-3.0, pivot=1000.0):
    n = x.shape[0]
    N = next_fast_len(n + sr // 20)
    f = np.maximum(np.fft.rfftfreq(N, 1.0 / sr), 20.0)
    g = 10 ** (np.clip(float(slope) * np.log2(f / max(float(pivot), 20.0)), -24, 24) / 20)
    g[0] = 1.0                                                           # dc is not part of the slope
    y = _per_channel(lambda s: np.fft.irfft(np.fft.rfft(s, N) * g, N)[:n], x)
    return y * (_rms(x) / (_rms(y) + 1e-12))                              # a tilt changes colour, not level


@layer_type("resynth")
def _layer_resynth(layer: dict, sr: int, q: float) -> np.ndarray:
    """{"type": "resynth", "path": "in.wav", "stretch": 1.0, "transpose": 0, "formant": 0, "sines": 1, "noise": 1}
    optional: "window_ms" (46; analysis window, at least 4 periods of the lowest pitch), "seed" (noise)."""
    x, fsr = read_wav(layer["path"])
    wms = float(layer.get("window_ms", 46.0))
    kw = dict(stretch=float(layer.get("stretch", 1.0)), transpose=float(layer.get("transpose", 0.0)), formant=float(layer.get("formant", 0.0)),
              sines_gain=float(layer.get("sines", 1.0)), noise_gain=float(layer.get("noise", 1.0)), seed=int(layer.get("seed", 0)))
    return _per_channel(lambda s: resynth(analyze(s, fsr, window_ms=wms), sr=sr, **kw), x)
