"""STFT-domain effects and analysis -> resynthesis.

* Kernel: *periodic* windows, zero-phase framing, root-Hann analysis + synthesis windows (WOLA),
  hop M/4 (the hop robust to modification).
* Phase vocoder with identity phase locking and transient handling, SOLA.
* Pitch shift = stretch + band-limited resample, cepstral formant correction.
* Sines + noise model: QIFFT peaks, peak tracking, cubic phase interpolation, Bark/ERB band noise.
* Envelopes: cepstral lifter, true envelope, LPC (existing Levinson), minimum phase by cepstral folding.
* Channel vocoder with the Voder band edges, cross-synthesis.

Everything takes mono (n,) or stereo (n, 2) unless stated.
"""
from __future__ import annotations

import json
import math
import re
from fractions import Fraction

import numba
import numpy as np
from scipy.fft import dct, idct, next_fast_len
from scipy.ndimage import gaussian_filter, maximum_filter1d, median_filter, uniform_filter
from scipy.signal import correlate, hilbert, resample_poly

from . import osc as O
from .core import DEFAULT_SR, pad_to, read_wav, to_mono
from .fx import _per_channel, effect
from .notes import parse_pitch_list, to_hz
from .spec import layer_type

# Bark band edges (Hz) with an extrapolation for high sample rates
BARK_EDGES = (0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150,
              3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500, 20500, 27000)
# Voder vocoder band edges (Hz)
VODER_EDGES = (0, 225, 450, 700, 1000, 1400, 2000, 2700, 3800, 5400, 7500)


# ==========================================================================================
# Kernel: windows, COLA check, zero-phase STFT / WOLA
# ==========================================================================================

def _win_len(ms: float, sr: int) -> int:
    """Power-of-two window length nearest to `ms` (>= 64)."""
    return 1 << max(6, int(round(math.log2(max(ms * 1e-3 * sr, 2.0)))))


def window(M: int, kind: str = "hann", root: bool = False) -> np.ndarray:
    """*Periodic* window (denominator M): only this form overlap-adds to an exact constant.
    `root=True` gives the square root (root-Hann = MLT sine window) for WOLA."""
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
    """Relative ripple of sum_m w(n - mR); 0 for a COLA window."""
    s = np.zeros(len(w) + 8 * R)
    for o in range(0, len(s) - len(w) + 1, R):
        s[o:o + len(w)] += w
    mid = s[len(w):-len(w)]
    return float(np.ptp(mid) / mid.mean())


def _fwd(fr: np.ndarray, N: int) -> np.ndarray:
    """Windowed frames (F, M) -> rfft of the zero-phase buffer (frame centre at index 0, zero
    padding in the middle), so a peak's phase is the sinusoid's phase at the frame centre."""
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
    R = M/4, N = M; pass N = 2M when a per-frame filter is applied (N >= M + L - 1).
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
    """Inverse of `stft` (WOLA): synthesis window `w` (default root-Hann), gain
    R / sum(wa * w) (the Poisson-sum constant; exact when wa*w is COLA at R)."""
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
    """Cepstral-windowed envelope of magnitude spectra on an rfft grid (last axis):
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


TE_LAMBDA_MAX = 2.5


def true_envelope(mag: np.ndarray, n_c: int, delta_db: float = 2.0, iters: int = 200, accel: bool = False) -> np.ndarray:
    """True envelope: A_0 = log|X|, A_i = max(A_{i-1}, V_{i-1}), V_i the cepstral smoothing of A_i, stop when no frame's
    log|X| rises above V_i by more than delta_db (2 dB there). It rides over the harmonics instead of through them, so a low order does not lose the peaks.
    ``accel``: start from a max filter of log|X| over half the cepstral resolution (nfft / (4 n_c) bins each side) and
    over-relax each step, V_i = V_{i-1} + lambda (V'_i - V_{i-1}) with lambda = (E_I + E_O) / E_I (E_I: squared excess of
    log|X| over V_{i-1}, E_O: squared shortfall), at most 2.5. The number of steps taken is left in ``true_envelope.last_iters``."""
    if accel:
        return _true_envelope_accel(mag, n_c, delta_db, iters)
    nfft = 2 * (mag.shape[-1] - 1)
    n_c = int(np.clip(n_c, 1, nfft // 2 - 1))
    lif = np.zeros(nfft)
    lif[:n_c] = 1
    lif[n_c] = 0.5
    lif[nfft - n_c + 1:] = 1
    lif[nfft - n_c] = 0.5
    A0 = np.log(np.maximum(mag, 1e-5 * mag.max() + 1e-12))
    A = A0.copy()
    lim = float(delta_db) * math.log(10.0) / 20.0
    for _ in range(iters):
        V = np.fft.rfft(np.fft.irfft(A, nfft, axis=-1) * lif, nfft, axis=-1).real
        if float((A0 - V).max()) <= lim:
            break
        A = np.maximum(A, V)
    true_envelope.last_iters = _ + 1
    return np.exp(V)


def _true_envelope_accel(mag, n_c, delta_db, iters):
    nfft = 2 * (mag.shape[-1] - 1)
    n_c = int(np.clip(n_c, 1, nfft // 2 - 1))
    lif = np.zeros(nfft)
    lif[:n_c] = 1
    lif[n_c] = 0.5
    lif[nfft - n_c + 1:] = 1
    lif[nfft - n_c] = 0.5
    A0 = np.log(np.maximum(mag, 1e-5 * mag.max() + 1e-12))
    M = max(1, nfft // (4 * n_c))
    A = maximum_filter1d(A0, 2 * M + 1, axis=-1, mode="nearest")
    lim = float(delta_db) * math.log(10.0) / 20.0
    V = np.fft.rfft(np.fft.irfft(A, nfft, axis=-1) * lif, nfft, axis=-1).real
    it = 1
    for it in range(1, iters + 1):
        d = A0 - V
        if float(d.max()) <= lim:
            break
        A = np.maximum(A0, V)
        Vn = np.fft.rfft(np.fft.irfft(A, nfft, axis=-1) * lif, nfft, axis=-1).real
        ei = float(np.sum(np.maximum(d, 0.0) ** 2))
        eo = float(np.sum(np.minimum(d, 0.0) ** 2))
        lam = min((ei + eo) / max(ei, 1e-30), TE_LAMBDA_MAX)
        V = V + lam * (Vn - V)
    true_envelope.last_iters = it
    return np.exp(V)


true_envelope.last_iters = 0


def max_f0(x: np.ndarray, sr: int = DEFAULT_SR) -> float:
    """Highest steady pitch of a sound: normalised autocorrelation of 46 ms frames, peak over 0.5, 95th percentile (0 when nothing is pitched)."""
    m = to_mono(x)
    N = _win_len(46.0, sr)
    fr = [m[i: i + N] for i in range(0, max(len(m) - N, 1), N // 2)]
    lo, hi = int(sr / 1500.0), N // 2
    out = []
    w = np.hanning(N)
    for f in fr:
        if len(f) < N or np.max(np.abs(f)) < 1e-3 * (np.max(np.abs(m)) + 1e-12):
            continue
        S = np.abs(np.fft.rfft(f * w, 2 * N)) ** 2
        r = np.fft.irfft(S)[:hi]
        r = r / max(r[0], 1e-20)
        pk = np.flatnonzero((r[lo:-1] > 0.5) & (r[lo:-1] >= r[lo - 1:-2]) & (r[lo:-1] >= r[lo + 1:]))
        if len(pk):
            out.append(sr / (pk[0] + lo))
    return float(np.percentile(out, 95)) if out else 0.0


def lpc_envelope(mag: np.ndarray, order: int = 24) -> np.ndarray:
    """All-pole (peak-hugging) envelope of magnitude spectra (frames, K): autocorrelation
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
    onto positive time). nfft must be well above the filter length (cepstral aliasing)."""
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


def transfer_envelope(src: np.ndarray, dst: np.ndarray, sr: int = DEFAULT_SR, shift: float = 0.0, env_ms: float | None = 2.0, method: str = "cepstral", shift_hz: float = 0.0) -> np.ndarray:
    """Give `dst` the spectral envelope of `src` (same length), frame by frame: divide by dst's own
    envelope, multiply by src's. `shift` moves the imposed formants by semitones, `shift_hz` by Hz.
    N = 2M so the gain curve's impulse response fits. env_ms None: the lifter follows the highest pitch of both sounds,
    P_c = 1 / (2 f0 max), so high voices keep their harmonics out of the envelope. method: cepstral | true (`true_envelope`)."""
    M = _win_len(40.0, sr)
    if env_ms is None:
        f0 = max(max_f0(src, sr), max_f0(dst, sr))
        n_c = max(2, int(sr / (2.0 * f0))) if f0 > 0 else max(2, int(2e-3 * sr))
    else:
        n_c = max(2, int(env_ms * 1e-3 * sr))
    env = true_envelope if method == "true" else cepstral_envelope

    def one(s, d):
        Xd = stft(d, M, N=2 * M)
        Es = env(np.abs(stft(pad_to(s, len(d))[:len(d)], M, N=2 * M)), n_c)
        if shift:
            Es = _warp(Es, 2.0 ** (shift / 12.0))
        if shift_hz:
            k = np.arange(Es.shape[-1]) - float(shift_hz) * 2 * M / sr
            k = np.clip(k, 0, Es.shape[-1] - 1)
            i0 = np.minimum(k.astype(int), Es.shape[-1] - 2)
            Es = Es[..., i0] * (1 - (k - i0)) + Es[..., i0 + 1] * (k - i0)
        P = np.abs(Xd) ** 2
        g = np.clip(Es / env(np.sqrt(P), n_c), 10 ** -1.5, 10 ** 1.5)
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
    level just before plus a quarter of the local maximum.
    # ponytail: full-band flux; a hit under an equally loud sustained sound is missed and a 10 Hz
    # beat is (correctly) not flagged. Per-band flux if drums buried in pads matter."""
    M = _win_len(12.0, sr)
    R = M // 4
    mag = np.abs(stft(to_mono(x), M, R))
    tot = mag.sum(axis=1)
    if tot.max() <= 0:
        return np.zeros(0, int)
    loc = maximum_filter1d(tot, 2 * int(0.3 * sr / R) + 1)
    rel = np.maximum(mag[1:] - mag[:-1], 0.0).sum(axis=1) / (tot[:-1] + 0.25 * loc[1:] + 1e-12)
    thr = 0.5 / max(float(sensitivity), 1e-3)
    pk = (rel > thr) & (rel >= maximum_filter1d(rel, 9)) & (tot[1:] > 1e-3 * tot.max())
    return np.clip((np.flatnonzero(pk) + 1) * R - M // 2, 0, len(x) - 1)


def _time_map(x, sr, factor, M, transients):
    """Piecewise-linear output-time -> input-time map: transient intervals (onset +- M/2) are
    translated at rate 1, everything else is stretched so the total is exactly round(n*factor)."""
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
    """Phase vocoder core (WOLA, root-Hann, synthesis hop R = M/4,
    verified in tests/test_spectral.py). Output frame i takes its magnitudes from the input frame
    centred on sample pos[i]; its phase is the previous output phase plus the phase advance measured
    between the input frames at pos[i] - R and pos[i] ("horizontal" coherence, with no unwrapping
    ambiguity because that advance is over exactly one synthesis hop). `lock`: identity phase locking
    (identity phase locking) - bins keep their measured phase relative to the nearest spectral peak
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
    if lock:                                       # peaks = maxima over +-2 bins (identity phase locking)
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


def _wsola(x, ibp, obp, n_out, M, sr, search=True):
    """Synchronous overlap-add with a similarity search (SOLA-FS / WSOLA family): Hann
    frames at a fixed synthesis hop M/2; each input frame is taken where it best continues the
    previous one (maximum normalised cross-correlation) within +-25 ms of the mapped position. Inside
    a rate-1 (transient) interval the natural continuation is in range, so a transient is copied
    once, never repeated or dropped. Stereo shares one offset per frame."""
    x2 = x[:, None] if x.ndim == 1 else x
    n = x2.shape[0]
    H = M // 2
    D = min(H, int(0.025 * sr)) if search else 0   # +-25 ms = half a 20 Hz period; a window M under 50 ms limits it to M/2
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


def hpss(x: np.ndarray, sr: int = DEFAULT_SR, l_h: int = 17, l_p: int = 17) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic and percussive parts by median filtering: Y_h = median along time, Y_p = median along frequency,
    binary masks Y_h > Y_p and Y_p >= Y_h. l_h = l_p = 17, FFT 4096, hop 1024 at 44.1 kHz. Their sum is x."""
    M = _win_len(4096 * 1000.0 / 44100.0, sr)

    def one(s):
        X = stft(s, M, M // 4)
        mag = np.abs(X)
        harm = median_filter(mag, size=(int(l_h), 1))
        perc = median_filter(mag, size=(1, int(l_p)))
        h = istft(X * (harm > perc), len(s), M, M // 4)
        return h, s - h

    if x.ndim == 1:
        return one(x)
    parts = [one(x[:, c]) for c in range(x.shape[1])]
    return np.stack([p[0] for p in parts], axis=1), np.stack([p[1] for p in parts], axis=1)


def _parse_anchors(anchors, n, sr):
    """"t_in:t_out, ..." seconds, or pairs -> strictly increasing breakpoints in samples, from (0, 0); the last segment runs at rate 1 to the end of the input."""
    if isinstance(anchors, str):
        pairs = [tuple(float(v) for v in a.split(":")) for a in re.split(r"[\s,;]+", anchors.strip()) if a]
    else:
        pairs = [tuple(float(v) for v in a) for a in anchors]
    a = sorted(pairs)
    ibp, obp = [0.0], [0.0]
    for ti, to in a:
        i, o = min(ti * sr, float(n)), to * sr
        if i > ibp[-1] and o > obp[-1]:
            ibp.append(i)
            obp.append(o)
    if ibp[-1] < n:
        obp.append(obp[-1] + n - ibp[-1])
        ibp.append(float(n))
    return np.array(ibp), np.array(obp)


def stretch(x: np.ndarray, sr: int = DEFAULT_SR, factor: float = 1.0, mode: str = "vocoder", window_ms: float = 0.0,
            lock: bool = True, transients=True, anchors=None) -> np.ndarray:
    """Time-stretch by `factor` (> 1 longer) at constant pitch; output length is round(n * factor). `anchors`: a time map instead of a factor,
    "t_in:t_out, ..." to land moments of the sound on an animation. mode hps: vocoder on the harmonic part, short OLA (10 ms) on the percussive part. mode ola: overlap-add without the similarity search."""
    factor = float(np.clip(factor, 0.1, 10.0))
    if x.shape[0] < 16 or (abs(factor - 1.0) < 1e-9 and anchors is None):
        return x.copy()
    if mode in ("psola", "auto"):
        if anchors is not None:
            raise ValueError("anchors need mode vocoder|sola|ola|hps")
        p = psola(x, sr, 0.0, factor)
        return p if mode == "psola" else _route(x, sr, p, stretch(x, sr, factor, "vocoder", window_ms, lock, transients))
    if mode not in ("vocoder", "sola", "ola", "hps"):
        raise ValueError(f"unknown timestretch mode {mode!r} (vocoder|sola|ola|hps|psola|auto)")
    if mode == "hps":
        h, p = hpss(x, sr)
        a = stretch(h, sr, factor, "vocoder", window_ms, lock, False, anchors)
        b = stretch(p, sr, factor, "ola", 10.0, lock, False, anchors)
        return a[: b.shape[0]] + b[: a.shape[0]]
    M = _win_len(window_ms or (45.0 if mode == "vocoder" else 50.0), sr)   # 45 ms vocoder; 50 ms for the +-25 ms search
    if anchors is None:
        n_out, ibp, obp, starts = _time_map(to_mono(x), sr, factor, M, transients)
    else:
        ibp, obp = _parse_anchors(anchors, x.shape[0], sr)
        n_out, starts = int(round(obp[-1])), []
    if mode in ("sola", "ola"):
        return _wsola(x, ibp, obp, n_out, M, sr, mode == "sola")
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
    inverse. formant="keep" re-imposes the original spectral envelope (no "munchkinization"), "true" the true envelope with a lifter set from the pitch (high voices)."""
    r = Fraction(2.0 ** (float(np.clip(semitones, -36, 36)) / 12.0)).limit_denominator(512)
    p, q = r.numerator, r.denominator
    n = x.shape[0]
    if p == q or n < 16:
        return x.copy()
    if mode in ("psola", "auto"):
        a = psola(x, sr, 12.0 * math.log2(p / q))
        return a if mode == "psola" else _route(x, sr, a, shift_pitch(x, sr, semitones, formant, "vocoder"))
    y = pad_to(resample_poly(stretch(x, sr, p / q, mode), q, p, axis=0), n)[:n]
    if formant == "true":
        return transfer_envelope(x, y, sr, env_ms=None, method="true")
    return transfer_envelope(x, y, sr) if formant == "keep" else y


# ==========================================================================================
# Sines + noise model
# ==========================================================================================

def band_edges(sr: int, kind: str = "bark") -> np.ndarray:
    """Noise-band edges in Hz up to sr/2: the Bark table, or one band per ERB
    (ERBS(f) = 21.4 log10(0.00437 f + 1))."""
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
    linear between frames. cubic: cubic phase through the measured phases and
    frequencies.
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


TMS_BLOCK_S, TMS_THRESH_DB = 1.0, -20.0


def _tms_render(model: dict, stretch: float, sr: int) -> np.ndarray:
    """The transients of a model, moved to their stretched onsets but not stretched themselves:
    the DCT-domain sines are time-scaled by stretch * sr / model sr, so each block grows and every click moves with it."""
    msr, N, nb = int(model["sr"]), int(model["t_block"]), int(model["t_blocks"])
    sub = {"sr": msr, "n": nb * N, "hop": model["t_hop"], "win": model["t_win"], "freq": model["t_freq"],
           "amp": model["t_amp"], "phase": model["t_phase"], "noise": np.zeros((0, 0)), "edges": np.zeros(0)}
    g = stretch * sr / msr
    C = resynth(sub, sr=msr, stretch=g, noise_gain=0.0)
    Nb = len(C) // nb
    return idct(C[:nb * Nb].reshape(nb, Nb), norm="ortho", axis=1).reshape(-1) / math.sqrt(stretch)


def analyze(x: np.ndarray, sr: int = DEFAULT_SR, window_ms: float = 46.0, thresh_db: float = -50.0, max_peaks: int = 60,
            max_tracks: int = 40, bands: str = "bark", min_frames: int = 4, transients: bool = False) -> dict:
    """Sines + noise analysis (spectral modelling synthesis) of a mono signal.

    Returns a plain dict of numpy arrays / numbers (see `save_model`):
      sr, n (samples), hop, win; frame m is centred on sample m*hop - win/2
      freq, amp, phase : (frames, tracks) partial frequency in Hz, linear amplitude, cosine phase at
                         the frame centre; amp == 0 where a track is silent
      noise            : (frames, bands) residual level per band, as the standard deviation of white
                         noise with the same spectral density;  edges : (bands + 1,) Hz

    Peaks: periodic Hamming window (main lobe 4 bins, so harmonics are
    resolved when the window holds 4 periods: f0 >= 4000 / window_ms Hz - use a longer window
    for bass), zero padding 2, parabola through the three dB samples around each maximum (QIFFT),
    the same parabola on the real and imaginary parts for the phase. A peak must
    have roughly the window's own curvature (a deviant value flags a non-sinusoidal peak);
    that also rejects the -42 dB side lobes, which are half as wide (minimum width).
    Tracking: closest peak within DFmax = MinSep/2 = 2 sr / M Hz per frame, the
    closer track wins a conflict, unmatched tracks ramp to zero over one hop, unmatched peaks are
    born at zero amplitude one frame earlier (the first frame starts at full level), 60 peaks and
    40 tracks. Residual: magnitude spectrum minus that of the resynthesised sines, summed
    per Bark or ERB band.
    transients: sines + transients + noise. The sines' residual is cut in
    non-overlapping blocks, each block's DCT turns a click into a slowly varying sinusoid, and this same
    sinusoidal analysis tracks them (keys t_*); the noise is then measured on what both leave. Off, an
    attack is spread over sines and noise and smears when stretched."""
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
            & (c > db.max() - 80.0)                                   # absolute floor so silent frames give no tracks
            & (np.arange(1, K - 1) * sr / N > sr / 1000.0)[None, :])  # lowest peak frequency Fs/1000
    fi, ki = np.nonzero(mask)
    ki = ki + 1
    a, b, g = db[fi, ki - 1], db[fi, ki], db[fi, ki + 1]
    curv = 0.5 * (a - 2 * b + g)
    p = np.clip(0.5 * (a - g) / np.minimum(a - 2 * b + g, -1e-12), -0.5, 0.5)
    Wd = 20 * np.log10(np.abs(np.fft.rfft(w, N)[:2]))
    cw = Wd[1] - Wd[0]                                                # the window's own half-curvature (dB / bin^2, < 0)
    ok = (curv < 0.25 * cw) & (curv > 2.5 * cw)                       # tolerance on the curvature test

    def par(v):
        return v[fi, ki] + 0.5 * (v[fi, ki + 1] - v[fi, ki - 1]) * p + 0.5 * (v[fi, ki - 1] - 2 * v[fi, ki] + v[fi, ki + 1]) * p * p

    pf = ((ki + p) * sr / N)[ok]
    pa = (10 ** ((b - 0.25 * (a - g) * p) / 20) * 2.0 / w.sum())[ok]  # peak height = A sum(w)/2
    pp = np.arctan2(par(X.imag), par(X.real))[ok]
    bounds = np.searchsorted(fi[ok], np.arange(F + 1))
    dfmax = 2.0 * sr / M
    hopw = 2 * np.pi * R / sr
    GAP = 2
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
    s = _osc_bank(fr, am, ph, pos, n, float(sr), True)
    model = {"sr": int(sr), "n": int(n), "hop": int(R), "win": int(M), "freq": fr, "amp": am, "phase": ph}
    if transients and n:
        Nb = int(TMS_BLOCK_S * sr)
        nb = -(-n // Nb)
        C = dct(np.pad(x - s, (0, nb * Nb - n)).reshape(nb, Nb), norm="ortho", axis=1).reshape(-1)
        t = analyze(C, sr, window_ms, TMS_THRESH_DB, max_peaks, max_tracks, bands, min_frames)
        model.update(t_block=Nb, t_blocks=nb, t_hop=t["hop"], t_win=t["win"], t_freq=t["freq"], t_amp=t["amp"], t_phase=t["phase"])
        s = s + _tms_render(model, 1.0, sr)[:n]
    S = stft(s, M, R, N, w)
    res = np.maximum(np.abs(X) - np.abs(S), 0.0) ** 2
    edges = band_edges(sr, bands)
    kb = np.round(edges[:-1] * N / sr).astype(int)
    cnt = np.diff(np.append(kb, K))
    noise = np.sqrt(np.add.reduceat(res, kb, axis=1) / cnt / np.sum(w * w))    # E|X|^2 = sigma^2 sum(w^2) for white noise
    return {**model, "noise": noise, "edges": edges}


def resynth(model: dict, sr: int | None = None, stretch: float = 1.0, transpose: float = 0.0, formant: float = 0.0,
            sines_gain: float = 1.0, noise_gain: float = 1.0, seed: int = 0, phases: str = "auto",
            transient_gain: float = 1.0) -> np.ndarray:
    """Render a model from `analyze` (mono), at any sample rate.

    stretch: duration factor (the envelopes are resampled). transpose: semitones applied
    to the partial frequencies. formant: semitones the spectral envelope moves - 0 with a transpose
    keeps the formants in place (partial amplitudes are re-read from the piecewise-linear envelope
    through the frame's peaks), formant == transpose is a plain "tape" transposition;
    the noise bands move with `formant`. phases: "cubic" follows the measured phases (waveform
    preserved), "free" integrates frequency, "auto" = cubic only when stretch == 1 and transpose == 0.
    Noise: random-phase spectra shaped by the band levels, root-Hann at 75 % overlap.
    Transients (a model from analyze(transients=True)) move to their stretched onsets at their own
    length and are not transposed."""
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
    if transient_gain and "t_freq" in model:
        y += transient_gain * pad_to(_tms_render(model, stretch, sr), n_out)[:n_out]
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
        factor=(1.0, "length ratio 0.1..10: 2 = twice as long"), mode=("vocoder", "vocoder (tonal, music) | sola (speech, drums, single voices) | ola (sola without the similarity search) | hps (vocoder on the tones, OLA on the hits) | psola (pitch-synchronous, voices) | auto (psola where the voice pulses are sharp, vocoder elsewhere)"),
        window_ms=(0.0, "analysis window in ms; 0 = auto (45 vocoder, 50 sola). Longer = cleaner bass, more smear"),
        anchors=(None, "time map instead of a factor: 't_in:t_out, ...' in seconds, to land moments of the sound on an animation; rate 1 after the last one"),
        lock=(True, "vocoder: lock bin phases to spectral peaks (less 'phasiness')"),
        transients=(True, "keep attacks unstretched; a number is the detector sensitivity (1 = default, 2 = more attacks)"))
def timestretch(x, sr=DEFAULT_SR, factor=1.0, mode="vocoder", window_ms=0.0, anchors=None, lock=True, transients=True):
    return stretch(x, sr, factor, mode, float(window_ms), lock, transients, anchors)


@effect("hpss", "Split a sound by median filtering: keep its tones or its hits.",
        keep=("harmonic", "harmonic | percussive"))
def hpss_fx(x, sr=DEFAULT_SR, keep="harmonic"):
    h, p = hpss(np.asarray(x, float), sr)
    return h if keep == "harmonic" else p


@effect("autotune", "Pull the pitch to the nearest semitone (sines + noise analysis, partials moved frame by frame).",
        amount=(1.0, "0 = untouched .. 1 = on the semitone"), thresh=(-50.0, "peak threshold in dB below each frame's loudest partial"))
def autotune(x, sr=DEFAULT_SR, amount=1.0, thresh=-50.0):
    """nst = round(12 log2(f0 / 55)), f = 55 2^(nst / 12). f0 here: the lowest partial within 12 dB of the loudest; the frames follow the partials."""
    def one(s):
        m = analyze(s, sr, thresh_db=float(thresh))
        fr, am = np.array(m["freq"], float), np.array(m["amp"], float)
        for i in range(fr.shape[0]):
            live = (fr[i] > 20.0) & (am[i] > 0.25 * (am[i].max() + 1e-12))
            if live.any():
                f0 = float(fr[i][live].min())
                fr[i] = fr[i] * (55.0 * 2.0 ** (round(12.0 * math.log2(f0 / 55.0)) / 12.0) / f0) ** float(amount)
        y = resynth({**m, "freq": fr}, phases="free")
        return pad_to(y, len(s))[: len(s)]
    return _per_channel(one, np.asarray(x, float))


@effect("gender", "Voice change by pitch and formants together: male to female or the reverse.",
        to=("female", "female (pitch x 2) | male (pitch / 2)"))
def gender(x, sr=DEFAULT_SR, to="female"):
    """Gender change: pt = 2 or 0.5; spectral shape shift sss = (f0 - 100) / ((500 - 100) / 50) Hz, within 0..50, negative for female to male."""
    up = to == "female"
    f0 = max_f0(x, sr)
    sss = float(np.clip((f0 - 100.0) / (400.0 / 50.0), 0.0, 50.0))
    y = shift_pitch(x, sr, 12.0 if up else -12.0, "keep")
    return transfer_envelope(x, y, sr, shift_hz=sss if up else -sss)


@effect("pitch_shift", "Pitch shift at the same length (phase vocoder + band-limited resampling), optionally keeping the formants.",
        semitones=(0.0, "+/- semitones (-36..36)"), formant=("shift", "shift = formants move with the pitch (chipmunk / giant) | keep = timbre stays (voices, instruments)"),
        fine=(0.0, "extra shift in cents"),
        mode=("vocoder", "vocoder | psola (pitch-synchronous grains, keeps formants; pulsed voices) | auto (psola where the voice pulses are sharp, vocoder elsewhere)"))
def pitch_shift(x, sr=DEFAULT_SR, semitones=0.0, formant="shift", fine=0.0, mode="vocoder"):
    return shift_pitch(x, sr, float(semitones) + float(fine) / 100.0, formant, mode)


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
    e = lambda f: 21.4 * math.log10(0.00437 * f + 1)                   # ERB-rate spacing
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
    """Channel vocoder as an FFT filter bank: band energies of the
    modulator STFT, smoothed by an attack/release follower, scale the same bands of the carrier
    STFT after flattening the carrier (dividing by its own band level)."""
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
        mix=(1.0, "wet amount"), seed=(0, "noise seed when no carrier file is given"),
        method=("envelope", "envelope (cepstral) | stamp (bin by bin: this sound's magnitudes on the carrier, finer and noisier)"),
        squelch=(20.0, "stamp: how far a quiet carrier bin may be raised; low = its noise stays down"), amount=(1.0, "stamp: 0 = carrier only .. 1 = full stamp"))
def cross_synth(x, sr=DEFAULT_SR, carrier=None, flatten=True, smooth=2.0, mix=1.0, seed=0, method="envelope", squelch=20.0, amount=1.0):
    """Y = Xc / env(Xc) * env(Xm), cepstral envelopes, carrier phase kept. stamp: g[k] = c[k] min(1 / f[k], squelch^2 / 100), exponent p."""
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
    if method == "stamp":
        k = np.abs(Xc)
        k = k / (k.max() + 1e-12)
        lim = float(squelch) ** 2 / 100.0

        def stamp(s):                                    # both magnitudes scaled to a peak of 1: the book's units are its own patch's
            f = np.abs(stft(s, M, N=2 * M))
            return istft(Xc * (f / (f.max() + 1e-12) * np.minimum(1.0 / np.maximum(k, 1e-12), lim)) ** float(amount), n, M)

        wet = _per_channel(stamp, x)
        wet = wet * (_rms(x) / (_rms(wet) + 1e-12))
        return x * (1 - mix) + wet * mix
    if flatten:
        Xc = Xc / np.maximum(cepstral_envelope(np.abs(Xc), n_c), 1e-4 * np.abs(Xc).max() + 1e-12)
    wet = _per_channel(lambda s: istft(Xc * cepstral_envelope(np.abs(stft(s, M, N=2 * M)), n_c), n, M), x)
    if not flatten:
        wet = wet * (_rms(x) / (_rms(wet) + 1e-12))
    return x * (1 - mix) + wet * mix


@effect("spectral_gate", "Spectral gate / denoiser: attenuate the quiet bins of the spectrum; `keep` splits tonal from noisy content.",
        threshold=(-40.0, "dB relative to the loudest spectral bin; bins below it are turned down"), reduction=(30.0, "attenuation in dB"),
        smoothing=(1.0, "gain smoothing over time and frequency 0..4 (0 = hard, 'musical noise')"),
        keep=("all", "all = plain gate | tonal = only spectral peaks above the threshold (removes hiss, breath) | noise = everything but the peaks"),
        mode=("gate", "gate | soft (no threshold: g = |X| / (|X| + c), no musical noise) | subtract (the noise is learned from the start of the sound or from `noise`)"),
        c=(0.01, "soft: the knee, on |X| / (N/2)"), learn=(0.2, "subtract: seconds at the start that hold only the noise"), noise=(None, "subtract: WAV file of the noise alone"),
        mask_level=(10.0, "subtract: bins this many times over the noise power start to pass"))
def spectral_gate(x, sr=DEFAULT_SR, threshold=-40.0, reduction=30.0, smoothing=1.0, keep="all", mode="gate", c=0.01, learn=0.2, noise=None, mask_level=10.0):
    """Per-bin gain 1 where the signal exceeds the threshold, else `reduction`, N = 2M, gains smoothed over about 16 bin x frame cells per unit of `smoothing`. Tonal bins = those well above the local median of the frame
    spectrum (a noise-floor estimate in the spirit of `findpeaks`' residual).
    soft: |X| -> |X|^2 / (|X| + c), c on |X| / (N/2), 0.01. subtract: g = sqrt((s - m) / s) if s > m else 0,
    m = mask_level x the mean noise power (typically 10, a bin 20 x over the noise only loses 3 dB)."""
    if keep not in ("all", "tonal", "noise"):
        raise ValueError(f"unknown keep {keep!r} (all|tonal|noise)")
    M = _win_len(46.0, sr)
    floor = 10 ** (-abs(float(reduction)) / 20)
    sm = max(0, int(round(float(smoothing))))
    if mode == "soft":
        return _wola(x, lambda X: X * (np.abs(X) / (np.abs(X) + float(c) * (M / 2.0))), M, 2 * M)
    if mode == "subtract":
        ref = read_wav(noise)[0] if noise else x[: max(int(float(learn) * sr), 2 * M)]
        P = np.mean(np.abs(stft(to_mono(ref), M, N=2 * M)) ** 2, axis=0)

        def sub(X):
            s = np.abs(X) ** 2
            m = float(mask_level) * P
            return X * np.sqrt(np.maximum(s - m, 0.0) / (s + 1e-30))

        return _wola(x, sub, M, 2 * M)
    if mode != "gate":
        raise ValueError("spectral_gate mode must be gate|soft|subtract")

    def fn(X):
        mag = np.abs(X)
        m = mag > mag.max() * 10 ** (float(threshold) / 20)
        if keep != "all":
            span = 2 * int(round(100.0 * 2 * M / sr)) + 1
            ton = maximum_filter1d(m & (mag > 4.0 * median_filter(mag, size=(1, span))), 5, axis=1)
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
    by its measured instantaneous frequency, peak-locked)."""
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


@effect("mutate", "Spectral mutation of this sound with another: multiplied magnitudes, or the phases of one on the magnitudes of the other.",
        other=("", "WAV file, looped to length"), mode=("magmul", "magmul (|X1||X2| with the phase of this sound) | magadd (|X1| + |X2|) | phaseswap (this magnitude, the other's phase)"))
def mutate(x, sr=DEFAULT_SR, other="", mode="magmul"):
    """Spectral denoising helpers (gain per bin from the noise estimate)."""
    o, osr = read_wav(other)
    o = to_mono(o)
    if osr != sr:
        r = Fraction(sr, osr)
        o = resample_poly(o, r.numerator, r.denominator)
    n = x.shape[0]
    o = np.resize(o, n)
    M = _win_len(46.0, sr)
    O2 = stft(o, M, N=2 * M)

    def one(s):
        X = stft(s, M, N=2 * M)
        a, b = np.abs(X), np.abs(O2)
        if mode == "phaseswap":
            Y = a * np.exp(1j * np.angle(O2))
        elif mode == "magadd":
            Y = (a + b) * np.exp(1j * np.angle(X))
        else:
            Y = a * b * np.exp(1j * np.angle(X))
        y = istft(Y, n, M)
        return y * (_rms(s) / (_rms(y) + 1e-12))

    return _per_channel(one, np.asarray(x, float))


@effect("freqshift", "Frequency shifter: every component moves by the same number of Hz (inharmonic, metallic; small values = slow phasing).",
        hz=(100.0, "shift in Hz, negative = down"), fold=(False, "true = what crosses Nyquist or 0 Hz comes back mirrored (the raw ring-modulator artefact); false removes it first"),
        hz_end=(None, "shift at the end: the shift glides linearly from hz to it (barber-pole sweeps)"))
def freqshift(x, sr=DEFAULT_SR, hz=100.0, fold=False, hz_end=None):
    """Single-sideband modulation of the analytic signal; the FFT Hilbert transform is
    circular, so the signal is zero-padded first. Without `fold`, what would wrap is filtered out: 21 kHz + 2 kHz otherwise lands on 21.1 kHz (Zavalishin c28, Addendum)."""
    n = x.shape[0]
    pad = min(n, sr // 10)
    curve = np.linspace(float(hz), float(hz if hz_end is None else hz_end), n)
    rot = np.exp(2j * np.pi * np.cumsum(curve) / sr) if hz_end is not None else np.exp(2j * np.pi * float(hz) * np.arange(n) / sr)
    if not fold:
        from .dsp import steep_filter
        h = float(np.max(np.abs(curve))) if n else 0.0
        hz = curve[np.argmax(np.abs(curve))] if n else hz
        if h > 0:
            x = steep_filter(x, "lowpass", 0.5 * sr - h, sr, 12) if hz > 0 else steep_filter(x, "highpass", min(h, 0.45 * sr), sr, 12)

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
    optional: "window_ms" (46; analysis window, at least 4 periods of the lowest pitch), "seed" (noise),
    "transients" (false; a gain: true or 1 keeps attacks and clicks sharp and unstretched, 0.5 halves them)."""
    x, fsr = read_wav(layer["path"])
    wms = float(layer.get("window_ms", 46.0))
    tg = float(layer.get("transients", 0.0))
    kw = dict(stretch=float(layer.get("stretch", 1.0)), transpose=float(layer.get("transpose", 0.0)), formant=float(layer.get("formant", 0.0)),
              sines_gain=float(layer.get("sines", 1.0)), noise_gain=float(layer.get("noise", 1.0)), seed=int(layer.get("seed", 0)),
              transient_gain=tg)
    return _per_channel(lambda s: resynth(analyze(s, fsr, window_ms=wms, transients=bool(tg)), sr=sr, **kw), x)


def phase_rotate(x, phi):
    """Constant phase shift of every component by ``phi`` radians (Hilbert pair): y = x cos(phi) - H(x) sin(phi).
    Against the original, the correlation is cos(phi): +1 at 0, 0 at pi/2, -1 at pi."""
    def one(v):
        n = len(v)
        return np.real(hilbert(v, next_fast_len(2 * n))[:n] * np.exp(1j * float(phi)))
    return _per_channel(one, np.asarray(x, float))


def twm_f0(freqs, amps_db, f0_candidates=None, rho=0.3, q=1.4, p=0.5, r=0.5, n_peaks=10):
    """Two-way mismatch fundamental: candidates = the 3 strongest peaks divided by 1..10 (unless given); for each,
    ErrPM = sum dF f^-p + MagFactor (q dF f^-p - r), MagFactor = max(0, 1 - max(0, Amax - A + 20)/75), the same from
    measured to predicted (ErrMP), Error = ErrPM/N + rho ErrMP/N over the 10 strongest peaks. Returns (f0, error)."""
    f = np.asarray(freqs, float)
    a = np.asarray(amps_db, float)
    order = np.argsort(-a)[:n_peaks]
    f, a = f[order], a[order]
    if f0_candidates is None:
        f0_candidates = np.unique(np.concatenate([f[:3] / k for k in range(1, 11)]))
    mag = np.maximum(0.0, 1.0 - np.maximum(0.0, a.max() - a + 20.0) / 75.0)
    best, err = 0.0, np.inf
    for f0 in np.asarray(f0_candidates, float):
        if f0 <= 0:
            continue
        harm = f0 * np.arange(1, int(np.ceil(f.max() / f0)) + 1)
        d = np.abs(harm[:, None] - f[None, :])
        dpm = d.min(axis=1)
        jm = d.argmin(axis=1)
        e_pm = np.sum(dpm * harm ** -p + mag[jm] * (q * dpm * harm ** -p - r))
        dmp = d.min(axis=0)
        e_mp = np.sum(dmp * f ** -p + mag * (q * dmp * f ** -p - r))
        e = e_pm / len(harm) + rho * e_mp / len(f)
        if e < err:
            best, err = float(f0), float(e)
    return best, err


def laguerre_map(w, b):
    """Frequency warping of the first-order allpass (z^-1 - b)/(1 - b z^-1): w' = w + 2 atan(b sin w / (1 - b cos w))."""
    w = np.asarray(w, float)
    return w + 2.0 * np.arctan2(b * np.sin(w), 1.0 - b * np.cos(w))


def laguerre_b_for(w0, w0_new):
    """Warping coefficient that moves w0 to w0_new: b = (tan(w0/2) - tan(w0'/2)) / (tan(w0/2) + tan(w0'/2))."""
    t0, t1 = math.tan(w0 / 2.0), math.tan(w0_new / 2.0)
    return (t0 - t1) / (t0 + t1)


def reduce_envelope(t, a, segments=8):
    """Breakpoint envelope cut to at most ``segments`` straight segments: greedily keep the point farthest from the
    current piecewise-linear approximation (Douglas-Peucker order). Returns (t_kept, a_kept)."""
    t, a = np.asarray(t, float), np.asarray(a, float)
    keep = {0, len(t) - 1}
    while len(keep) < min(segments + 1, len(t)):
        k = sorted(keep)
        err = np.abs(a - np.interp(t, t[k], a[k]))
        err[k] = -1.0
        keep.add(int(np.argmax(err)))
    k = sorted(keep)
    return t[k], a[k]


@effect("spectacle", "Spectral delay: the spectrum is cut into bands, each delayed by its own time with its own feedback "
        "(rising, falling or scattered echoes of different colours).",
        delay=(0.5, "s longest band delay"), table=("random", "random | ascending | descending: which bands wait longest"),
        feedback=(0.8, "0..0.95 per-band feedback"), bands=(64, "number of bands"), low=(100.0, "Hz"), high=(4100.0, "Hz"),
        mix=(0.5, "wet 0..1"), seed=(0, "variation"))
def spectacle(x, sr=DEFAULT_SR, delay=0.5, table="random", feedback=0.8, bands=64, low=100.0, high=4100.0, mix=0.5, seed=0):
    nfft, hop = 1024, 256
    fb = float(np.clip(feedback, 0.0, 0.95))
    nb = int(np.clip(bands, 1, 512))
    edges = np.geomspace(max(float(low), 1.0), min(float(high), 0.49 * sr), nb + 1)
    fr = np.fft.rfftfreq(nfft, 1.0 / sr)
    band = np.clip(np.searchsorted(edges, fr) - 1, -1, nb - 1)
    shape = {"ascending": np.linspace(0, 1, nb), "descending": np.linspace(1, 0, nb)}.get(
        table, np.random.default_rng(int(seed)).random(nb))
    lag = np.round(shape * float(delay) * sr / hop).astype(int)
    win = np.hanning(nfft)

    def one(v):
        n = len(v)
        frames = (n + nfft) // hop + 2
        sp = np.concatenate([np.zeros(nfft), v, np.zeros(frames * hop)])
        X = np.array([np.fft.rfft(sp[i * hop:i * hop + nfft] * win) for i in range(frames)])
        Y = np.zeros_like(X)
        for k in range(len(fr)):
            b = band[k]
            if b < 0 or edges[-1] < fr[k]:
                continue
            L = max(int(lag[b]), 1)
            for i in range(L, frames):
                Y[i, k] = X[i - L, k] + fb * Y[i - L, k]
        out = np.zeros(len(sp) + nfft)
        for i in range(frames):
            out[i * hop:i * hop + nfft] += np.fft.irfft(Y[i], nfft) * win
        return out[nfft:nfft + n] / 1.5
    wet = _per_channel(one, np.asarray(x, float))
    return (1.0 - mix) * x + mix * wet


# ==========================================================================================
# Pitch-synchronous OLA (TD-PSOLA) and the PSOLA / vocoder router
# ==========================================================================================

SHARP_PSOLA = 0.9
SHARP_AGREE = 6
SHARP_FADE = 0.020
_MARK_HOP = 0.01


def _f0_frames(m, sr, f0):
    from .analysis import yin
    if f0:
        return np.full(max(1, len(m) // int(_MARK_HOP * sr)), float(f0))
    f = yin(m, sr, hop=_MARK_HOP) if len(m) > int(0.06 * sr) else np.zeros(0)
    return f if len(f) else np.zeros(1)


def _f0_at(f, p, sr):
    return float(f[int(np.clip((p - 0.02 * sr) / (_MARK_HOP * sr), 0, len(f) - 1))])


def pitch_marks(x, sr=DEFAULT_SR, f0=None):
    """One mark per period at the waveform maximum (period from YIN, or the fixed `f0` in Hz),
    one every 10 ms where unvoiced. Returns (marks, voiced) as arrays."""
    m = to_mono(np.asarray(x, float))
    n = len(m)
    f = _f0_frames(m, sr, f0)
    marks, voiced = [], []
    p = 0
    while p < n:
        hz = _f0_at(f, p, sr)
        if hz > 0:
            T = sr / hz
            lo = int(p + 0.8 * T) if voiced and voiced[-1] else p
            hi = int(p + 1.2 * T) if voiced and voiced[-1] else int(p + T)
            if lo >= n:
                break
            q = lo + int(np.argmax(m[lo:min(n, hi + 1)]))
            marks.append(q)
            voiced.append(True)
            p = q
        else:
            q = p + int(_MARK_HOP * sr) if marks else p
            if q >= n:
                break
            marks.append(q)
            voiced.append(False)
            p = q
        if len(marks) > 1 and marks[-1] <= marks[-2]:
            marks[-1] = marks[-2] + 1
            p = marks[-1]
    return np.array(marks, np.int64), np.array(voiced, bool)


def psola(x, sr=DEFAULT_SR, semitones=0.0, factor=1.0, f0=None):
    """TD-PSOLA: two-period Hann grains cut at the pitch marks, re-spaced by period / pitch ratio
    (voiced) and repeated or dropped to follow the time `factor`. Formants stay in place."""
    x = np.asarray(x, float)
    n = x.shape[0]
    alpha = 2.0 ** (float(semitones) / 12.0)
    factor = float(np.clip(factor, 0.1, 10.0))
    n_out = int(round(n * factor))
    marks, voiced = pitch_marks(x, sr, f0)
    if n < 16 or len(marks) < 2:
        return pad_to(x, n_out)[:n_out].copy()
    per = np.diff(marks)
    per = np.concatenate([per, per[-1:]])
    x2 = x[:, None] if x.ndim == 1 else x
    y = np.zeros((n_out + 2 * int(per.max()) + 2, x2.shape[1]))
    ws = np.zeros(y.shape[0])
    pad = int(per.max()) + 1
    xp = np.concatenate([np.zeros((pad, x2.shape[1])), x2, np.zeros((pad, x2.shape[1]))])
    t = float(marks[0]) * factor
    while t < n_out:
        k = int(np.clip(np.searchsorted(marks, t / factor), 0, len(marks) - 1))
        if k > 0 and abs(marks[k - 1] - t / factor) < abs(marks[k] - t / factor):
            k -= 1
        T = int(per[k])
        w = np.hanning(2 * T + 1)
        c = int(marks[k]) + pad
        o = int(round(t)) - T + pad
        if o >= 0:
            y[o:o + 2 * T + 1] += xp[c - T:c + T + 1] * w[:, None]
            ws[o:o + 2 * T + 1] += w
        t += T / alpha if voiced[k] else T
    y = (y / np.maximum(ws, 1.0)[:, None])[pad:pad + n_out]
    y = pad_to(y, n_out)[:n_out]
    return y[:, 0] if x.ndim == 1 else y


def epoch_sharpness(x, sr=DEFAULT_SR, f0=110.0):
    """Pulse sharpness of a periodic segment: (crest factor - 1) x normalised autocorrelation at
    one period, over 8 periods at most. Pulse-like voice > 2, glottal ~1.2, breathy ~0.6, smooth < 0.5."""
    s = to_mono(np.asarray(x, float))
    T = int(round(sr / float(f0)))
    s = s[-min(len(s), 8 * T):]
    if T < 2 or len(s) < 2 * T:
        return 0.0
    s = s - s.mean()
    rms = math.sqrt(float(np.mean(s * s))) + 1e-12
    a, b = s[:-T], s[T:]
    r = float(np.dot(a, b) / math.sqrt(float(np.dot(a, a) * np.dot(b, b)) + 1e-24))
    return float((np.max(np.abs(s)) / rms - 1.0) * max(r, 0.0))


def psola_mask(x, sr=DEFAULT_SR, f0=None):
    """Per-sample weight of the PSOLA path (1) against the vocoder (0): epoch sharpness >= 0.9 on
    voiced 10 ms frames, the state switching after 6 agreeing frames, 20 ms linear crossfades."""
    m = to_mono(np.asarray(x, float))
    f = _f0_frames(m, sr, f0)
    hop = int(_MARK_HOP * sr)
    state, run, cur = np.zeros(len(f)), 0, 0.0
    for i, hz in enumerate(f):
        end = min(len(m), int(0.02 * sr) + i * hop + hop // 2)
        want = 1.0 if hz > 0 and epoch_sharpness(m[:end], sr, hz) >= SHARP_PSOLA else 0.0
        run = run + 1 if want != cur else 0
        if run >= SHARP_AGREE:
            cur, run = want, 0
        state[i] = cur
    t = 0.02 * sr + np.arange(len(f)) * hop
    g = state[np.clip(np.searchsorted(t, np.arange(len(m))), 0, len(f) - 1)]
    L = max(1, int(SHARP_FADE * sr))
    return np.convolve(np.pad(g, (L // 2, L - 1 - L // 2), mode="edge"), np.ones(L) / L, mode="valid")


def _route(x, sr, a, b, f0=None):
    w = psola_mask(x, sr, f0)
    w = np.interp(np.linspace(0, len(w) - 1, a.shape[0]), np.arange(len(w)), w)
    w = w if a.ndim == 1 else w[:, None]
    return w * a + (1.0 - w) * b[: a.shape[0]]


# ==========================================================================================
# Grain engine: ring buffer, freeze, feedback, pan, SOLA crossfades
# ==========================================================================================

GRAIN_RING_S = 6.0
GRAIN_VOICES = 96
GRAIN_TUKEY = 0.15
GRAIN_DC = 0.995


@numba.njit(cache=True)
def _grain_run(x, n_out, cap, hop, L, back, ratio, gl, gr, fb, freeze_n, win, fade, tukey, dcp, V):
    ring = np.zeros(cap)
    yl = np.zeros(n_out)
    yr = np.zeros(n_out)
    age = np.full(V, -1, np.int64)
    gid = np.zeros(V, np.int64)
    pos = np.zeros(V)
    w = 0
    dx = 0.0
    dy = 0.0
    nxt = 0
    G = len(back)
    for n in range(n_out):
        while nxt < G and nxt * hop <= n:
            for s in range(V):
                if age[s] < 0:
                    age[s] = 0
                    gid[s] = nxt
                    pos[s] = w - back[nxt]
                    break
            nxt += 1
        sl = 0.0
        sr_ = 0.0
        for s in range(V):
            a = age[s]
            if a < 0:
                continue
            j = gid[s]
            t = (a + 0.5) / L
            if win == 0:
                e = 0.5 - 0.5 * math.cos(2.0 * math.pi * t)
            elif win == 1:
                e = 1.0
                if t < tukey:
                    e = 0.5 - 0.5 * math.cos(math.pi * t / tukey)
                elif t > 1.0 - tukey:
                    e = 0.5 - 0.5 * math.cos(math.pi * (1.0 - t) / tukey)
            elif win == 2:
                e = math.exp(-5.0 * t) * (1.0 - t)
            else:
                e = min(1.0, (a + 0.5) / fade, (L - a - 0.5) / fade)
            p = pos[s]
            i1 = int(math.floor(p))
            f = p - i1
            y0 = ring[(i1 - 1) % cap]
            y1 = ring[i1 % cap]
            y2 = ring[(i1 + 1) % cap]
            y3 = ring[(i1 + 2) % cap]
            v = y1 + 0.5 * f * (y2 - y0 + f * (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3 + f * (3.0 * (y1 - y2) + y3 - y0)))
            sl += gl[j] * e * v
            sr_ += gr[j] * e * v
            pos[s] = p + ratio
            age[s] = a + 1 if a + 1 < L else -1
        yl[n] = sl
        yr[n] = sr_
        if n < freeze_n:
            mo = 0.5 * (sl + sr_)
            dy = mo - dx + dcp * dy
            dx = mo
            ring[w % cap] = (x[n] if n < len(x) else 0.0) + fb * dy
            w += 1
    return yl, yr


def grains(x, sr=DEFAULT_SR, density=20.0, size=0.08, spray=0.2, position=0.05, spread=0.5, pitch=0.0,
           feedback=0.0, freeze=None, window="hann", overlap="normal", tail=None, seed=0):
    """Grain cloud read from a 6 s ring buffer the input is written into. Each grain starts
    `back` = clip(0.9 position + 0.9 rnd spray, span / ring, 0.98) of the ring behind the write head
    (span = grain x max(ratio, 1) + 64), is read at `pitch` semitones (Catmull-Rom) and panned at
    theta = (pan / 2 + 1 / 2) pi / 2. `freeze` (s) stops recording; `feedback` (<= 0.95, DC-blocked)
    writes the output back into the ring. overlap="sola": each grain fades in over the life left in
    the previous one and out symmetrically (linear ramps, unity sum); otherwise the sum is scaled by
    1 / sqrt(max(density x size, 1)). Returns stereo when spread > 0."""
    m = to_mono(np.asarray(x, float))
    n = len(m)
    density = float(np.clip(density, 0.5, 200.0))
    hop = max(1, int(round(sr / density)))
    L = max(16, int(float(np.clip(size, 0.005, 1.0)) * sr))
    ratio = 2.0 ** (float(np.clip(pitch, -36, 36)) / 12.0)
    cap = int(GRAIN_RING_S * sr)
    span = L * max(ratio, 1.0) + 64.0
    hi = min(0.98, 1.0 - span / cap)
    back_frac = 0.9 * float(position) + 0.9 * float(np.clip(spray, 0, 1))
    if tail is None:
        tail = (min(back_frac, 0.98) * cap + L) / sr
    n_out = n + int(max(float(tail), 0.0) * sr)
    G = n_out // hop + 1
    rng = np.random.default_rng(int(seed))
    fr = np.clip(0.9 * float(position) + 0.9 * rng.random(G) * float(np.clip(spray, 0, 1)), span / cap, hi)
    pan = float(np.clip(spread, 0, 1)) * rng.uniform(-1.0, 1.0, G)
    th = (pan / 2.0 + 0.5) * math.pi / 2.0
    sola = overlap == "sola"
    if window not in ("hann", "tukey", "perc"):
        raise ValueError("window must be hann|tukey|perc")
    win = 3 if sola else ("hann", "tukey", "perc").index(window)
    fade = float(np.clip(L - hop, 1, L // 2))
    freeze_n = n_out if freeze is None else int(max(float(freeze), 0.0) * sr)
    yl, yr = _grain_run(np.ascontiguousarray(m), n_out, cap, hop, L, fr * cap, ratio, np.cos(th), np.sin(th),
                        float(np.clip(feedback, 0.0, 0.95)), freeze_n, win, fade, GRAIN_TUKEY, GRAIN_DC, GRAIN_VOICES)
    g = 1.0 if sola else 1.0 / math.sqrt(max(density * L / sr, 1.0))
    if spread <= 0:
        return (yl + yr) * (g / math.sqrt(2.0))
    return np.stack([yl, yr], axis=1) * g


@effect("granular", "Granular cloud from a ring buffer of the sound: grain size, density, spray, pitch, stereo spread, freeze and feedback.",
        density=(20.0, "grains per second (2..200)"), size=(0.08, "grain length in s (0.005..1)"),
        spray=(0.2, "random reach back into the buffer, 0..1"), position=(0.05, "how far back grains read, 0..1 of the 6 s buffer"),
        spread=(0.5, "stereo spread of the grains 0..1 (0 = mono output)"), pitch=(0.0, "grain transposition in semitones"),
        feedback=(0.0, "output written back into the buffer, 0..0.95"), freeze=(None, "seconds after which the buffer stops recording (hold)"),
        window=("hann", "hann | tukey | perc"), overlap=("normal", "normal | sola (each grain crossfades over the life left in the previous one)"),
        mix=(1.0, "0 = dry .. 1 = grains only"), seed=(0, "random seed"))
def granular(x, sr=DEFAULT_SR, density=20.0, size=0.08, spray=0.2, position=0.05, spread=0.5, pitch=0.0,
             feedback=0.0, freeze=None, window="hann", overlap="normal", mix=1.0, seed=0):
    wet = grains(x, sr, density, size, spray, position, spread, pitch, feedback, freeze, window, overlap, seed=seed)
    dry = np.asarray(x, float)
    if wet.ndim == 2 and dry.ndim == 1:
        dry = np.stack([dry, dry], axis=1)
    elif wet.ndim == 1 and dry.ndim == 2:
        dry = to_mono(dry)
    dry = pad_to(dry, wet.shape[0])
    return (1.0 - float(mix)) * dry + float(mix) * wet


@effect("granulate", "Brassage: the sound rebuilt from short Hann grains drawn near the current time (or anywhere), same length.",
        grain=(0.08, "grain length in s (0.001..0.1)"), window=(0.1, "s around the current time from which grains are drawn; 0 = anywhere in the sound"),
        overlap=(2.0, "grains per grain length (2 = half-grain overlap)"), pitch_var=(0.02, "random transposition per grain, in octaves (sd)"),
        pitch_st=(0.0, "random transposition per grain, uniform in +-semitones (12 = random granulation over two octaves); replaces pitch_var"),
        positions=("random", "with window 0: random | gauss | walk | zigzag read positions"),
        match=(0, "concatenative: candidates per grain, the one that best continues the previous grain wins (0 = off)"),
        min_back=(0.0, "s: no backward jump shorter than this between grains"),
        seed=(0, "random seed"))
def granulate(x, sr=DEFAULT_SR, grain=0.08, window=0.1, overlap=2.0, pitch_var=0.02, pitch_st=0.0, positions="random",
              match=0, min_back=0.0, seed=0):
    from .physical.analysis import SR as PSR, cloud
    x = np.asarray(x, float)
    n = x.shape[0]
    g = float(np.clip(grain, 0.001, 0.1)) * sr / PSR
    kw = {}
    if window and float(window) > 0:
        ramp = np.arange(n, dtype=float)
        kw = {"target": ramp, "feature": ramp, "choose_frac": float(min(1.0, float(window) * sr / max(n, 1)))}
    if pitch_st:
        kw["pitch_st"] = float(np.clip(pitch_st, 0.0, 24.0))
    return _per_channel(lambda s: cloud(s, n, np.random.default_rng(int(seed)), grain_s=(0.75 * g, 1.25 * g),
                                        overlap=float(max(overlap, 1.0)), pitch_var=float(pitch_var),
                                        positions=str(positions), match=int(match),
                                        min_back=float(min_back) * sr / PSR, **kw), x)


FOG_DEFAULTS = dict(dens=80.0, trans=1.0, spd=0.0, band=50.0, ris=0.01, dec=0.07, dur=0.1)


def fof_envelope(n: int, sr: int, band: float, ris: float, dec: float) -> np.ndarray:
    """FOF grain envelope: half-cosine rise over ``ris`` s, exp(-pi band t) body, half-cosine fall over the last ``dec`` s."""
    t = np.arange(n) / sr
    e = np.exp(-math.pi * band * t)
    r = min(int(ris * sr), n)
    if r > 0:
        e[:r] *= 0.5 * (1.0 - np.cos(np.pi * np.arange(r) / r))
    d = min(int(dec * sr), n)
    if d > 0:
        e[n - d:] *= 0.5 * (1.0 + np.cos(np.pi * np.arange(d) / d))
    return e


def fog(x, sr=DEFAULT_SR, dens=80.0, trans=1.0, spd=0.0, band=50.0, ris=0.01, dec=0.07, dur=0.1):
    """FOF granulation of a sample table: ``dens`` grains per second, each reading the table from the pointer
    (``spd`` table passes per second) at speed ``trans`` under a FOF envelope ``dur`` s long."""
    tab = to_mono(np.asarray(x, float))
    L = len(tab)
    if L < 2:
        return tab.copy()
    g = max(2, int(float(dur) * sr))
    env = fof_envelope(g, sr, float(band), float(ris), float(dec))
    idx = np.arange(g) * float(trans)
    y = np.zeros(L + g)
    step = sr / float(np.clip(dens, 1.0, 2000.0))
    t = 0.0
    while t < L:
        s = int(t)
        p0 = (float(spd) * s / sr) % 1.0 * L
        y[s:s + g] += np.interp((p0 + idx) % L, np.arange(L), tab) * env
        t += step
    return y[:L] / math.sqrt(max(float(dens) * float(dur), 1.0))


@effect("fog", "FOG: FOF grains read from the sound as a sample table (granular formant synthesis on a recording).",
        dens=(80.0, "grains per second"), trans=(1.0, "read speed inside each grain (2 = an octave up)"),
        spd=(0.0, "table passes per second of the grain start pointer (0 = every grain starts at the beginning)"),
        band=(50.0, "Hz: grain decay exp(-pi band t)"), ris=(0.01, "s grain rise"), dec=(0.07, "s grain fall at its end"),
        dur=(0.1, "s grain length"))
def fog_fx(x, sr=DEFAULT_SR, dens=80.0, trans=1.0, spd=0.0, band=50.0, ris=0.01, dec=0.07, dur=0.1):
    y = fog(x, sr, dens, trans, spd, band, ris, dec, dur)
    return np.stack([y, y], axis=1) if np.asarray(x).ndim == 2 else y


def _curve_points(curve) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(curve, str):
        pts = [tuple(float(v) for v in p.split(":")) for p in curve.replace(" ", "").split(",") if p]
    else:
        pts = [tuple(map(float, p)) for p in curve]
    pts.sort()
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def tf_pad(x, sr=DEFAULT_SR, curve="0:0,1:0", density=40.0, size=0.08, seed=0):
    """Time-frequency pad: Hann grains scan the sound; each grain is frequency-shifted (single sideband) by the
    drawn curve read at its position (x = position 0..1, y = shift in Hz)."""
    m = to_mono(np.asarray(x, float))
    L = len(m)
    g = max(16, int(float(size) * sr))
    if L <= g:
        return m.copy()
    xs, ys = _curve_points(curve)
    w = np.hanning(g)
    rng = np.random.default_rng(int(seed))
    hop = sr / float(np.clip(density, 1.0, 1000.0))
    y = np.zeros(L + g)
    k = np.arange(g) / sr
    t = 0.0
    while t < L - g:
        s = int(np.clip(t + rng.uniform(-0.25, 0.25) * hop, 0, L - g))
        hz = float(np.interp(s / L, xs, ys))
        a = hilbert(m[s:s + g] * w)
        y[int(t):int(t) + g] += np.real(a * np.exp(2j * np.pi * hz * (k + s / sr)))
        t += hop
    return y[:L] / max(g / hop * 0.5, 1.0)


@effect("tf_pad", "Time-frequency pad: grains scan the sound and a drawn curve shifts each one in frequency by its position.",
        curve=("0:0,1:0", "points 'x:y' (x = read position 0..1, y = shift in Hz), e.g. '0:0,0.5:300,1:-100'"),
        density=(40.0, "grains per second"), size=(0.08, "s grain length"), seed=(0, "random seed"))
def tf_pad_fx(x, sr=DEFAULT_SR, curve="0:0,1:0", density=40.0, size=0.08, seed=0):
    y = tf_pad(x, sr, curve, density, size, seed)
    return np.stack([y, y], axis=1) if np.asarray(x).ndim == 2 else y
