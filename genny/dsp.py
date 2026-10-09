"""Classical DSP toolbox: band-limited resampling, FIR/IIR design, FFT convolution and impulse-response
generators, companding and lo-fi codecs, dither, oversampled waveshaping, noise models and metering.

Sources (findings files in out/research):
  g11 = S. W. Smith, "The Scientist and Engineer's Guide to DSP" (chapter numbers cited as DSPG nn).
  g14 = J. O. Smith, "Introduction to Digital Filters" (FILT) and "Mathematics of the DFT" (MDFT).
Lines marked `# UNSOURCED` carry constants that are not in those sources (hardware lore, voicing choices).

Everything here is offline float64. Registered names: fx `convolve fir_eq steep_lowpass steep_highpass
steep_bandpass overdrive wavefold mulaw codec dither lofi smooth tape wall dc_block`, sfx `noise_bed`.
"""
from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache

import numba
import numpy as np
from scipy import signal as sg
from scipy.fft import dct, idct
from scipy.interpolate import PchipInterpolator
from scipy.linalg import solve_toeplitz

from . import filters as F
from .core import DEFAULT_SR, db_to_gain, read_wav, samples, to_mono
from .fx import _per_channel, effect
from .sfx import sfx

# ======================================================================================================
# 1. Band-limited resampling (DSPG 16 "windowed-sinc", DSPG 3 multirate; g11 top addition 1)
# ======================================================================================================


@lru_cache(maxsize=64)
def _resample_kernel(up: int, down: int, zeros: int, atten: float):
    """Kaiser-windowed sinc low-pass for polyphase conversion by up/down. `zeros` = sinc zero crossings
    each side at the lower of the two rates; the stopband starts exactly at the lower Nyquist."""
    m = max(up, down)
    ntaps = 2 * zeros * m + 1
    width = (atten - 7.95) / (2.285 * (ntaps - 1) * np.pi)  # Kaiser transition width, Nyquist = 1
    cutoff = max(1.0 / m - width / 2, 0.5 / m)
    return sg.firwin(ntaps, cutoff, window=("kaiser", sg.kaiser_beta(atten)))  # resample_poly applies the gain `up`


def resample_sinc(x: np.ndarray, sr_in, sr_out, zeros: int = 32, atten: float = 90.0) -> np.ndarray:
    """Windowed-sinc polyphase sample-rate conversion (axis 0; mono or stereo). No aliasing on
    down-conversion and no imaging on up-conversion, unlike `core.resample` (linear interpolation).
    Source: DSPG ch.16 (windowed-sinc kernel), ch.3 (zero-stuff + low-pass / low-pass + decimate); g11."""
    fr = Fraction(sr_out, sr_in) if float(sr_in).is_integer() and float(sr_out).is_integer() else None
    if fr is None:
        fr = Fraction(float(sr_out) / float(sr_in)).limit_denominator(1000)
    else:
        fr = Fraction(int(sr_out), int(sr_in))
    up, down = fr.numerator, fr.denominator
    if up == down:
        return np.array(x, dtype=np.float64)
    if max(up, down) > 2000:  # keep the kernel bounded for awkward ratios
        fr = fr.limit_denominator(1000)
        up, down = fr.numerator, fr.denominator
    h = _resample_kernel(up, down, int(zeros), float(atten))
    return sg.resample_poly(np.asarray(x, dtype=np.float64), up, down, axis=0, window=h)


@numba.njit(cache=True)
def _vari_read(x, pos, rate, W):
    n = pos.shape[0]
    N = x.shape[0]
    out = np.zeros(n)
    for i in range(n):
        p = pos[i]
        r = abs(rate[i])
        if r < 1.0:
            r = 1.0  # reading faster than 1x: widen the kernel = lower its cutoff (anti-alias)
        half = W * r
        j0 = int(math.ceil(p - half))
        j1 = int(math.floor(p + half))
        if j0 < 0:
            j0 = 0
        if j1 > N - 1:
            j1 = N - 1
        acc = 0.0
        wsum = 0.0
        for j in range(j0, j1 + 1):
            t = (j - p) / r
            s = 1.0 if abs(t) < 1e-9 else math.sin(math.pi * t) / (math.pi * t)
            u = t / W
            w = (0.42 + 0.5 * math.cos(math.pi * u) + 0.08 * math.cos(2 * math.pi * u)) * s
            acc += x[j] * w
            wsum += w
        if abs(wsum) > 1e-6:
            out[i] = acc / wsum
    return out


def varispeed(x: np.ndarray, rate, zeros: int = 16) -> np.ndarray:
    """Variable-rate band-limited read (tape-speed curves). `rate` = playback speed, a scalar or one value
    per OUTPUT sample (1 = normal, 2 = octave up). The read position is the running sum of the rate; each
    output sample is a Blackman-windowed-sinc interpolation whose cutoff drops to 1/rate when rate > 1.
    Source: DSPG ch.16 kernel (Eq 16-2/16-4) used as an interpolator; MDFT sampling theorem (g14 section E)."""
    x = np.asarray(x, dtype=np.float64)
    if np.isscalar(rate):
        n_out = max(1, int(x.shape[0] / max(abs(float(rate)), 1e-3)))
        rate = np.full(n_out, float(rate))
    rate = np.asarray(rate, dtype=np.float64)
    pos = np.concatenate([[0.0], np.cumsum(rate)[:-1]])
    return _per_channel(lambda s: _vari_read(np.ascontiguousarray(s), pos, rate, float(zeros)), x)


# ======================================================================================================
# 2. FIR design (DSPG 14, 16, 17; minimum phase FILT "Creating Minimum Phase Filters", g14 B.2)
# ======================================================================================================

# kernel length M = K / BW so that the stopband edge fc + BW/2 already sits at the window's full
# side-lobe level. Smith's Eq 16-3 (M = 4/BW) measures the 99 %..1 % width; reaching Blackman's -74 dB at
# the band edge needs the whole main lobe (6/M wide), i.e. K = 6 (verified in tests/test_dsp.py).
_WINDOW_K = {"blackman": 6.0, "hamming": 4.0, "hann": 4.0, "rect": 2.0}


def _window(name: str, m1: int) -> np.ndarray:
    i = np.arange(m1)
    M = m1 - 1
    if name == "blackman":
        return 0.42 - 0.5 * np.cos(2 * np.pi * i / M) + 0.08 * np.cos(4 * np.pi * i / M)  # Eq 16-2
    if name == "hamming":
        return 0.54 - 0.46 * np.cos(2 * np.pi * i / M)  # Eq 16-1
    if name == "hann":
        return 0.5 - 0.5 * np.cos(2 * np.pi * i / M)
    if name == "rect":
        return np.ones(m1)
    raise ValueError(f"unknown window {name!r} (blackman|hamming|hann|rect)")


def windowed_sinc(fc: float, bw: float, window: str = "blackman", taps: int | None = None) -> np.ndarray:
    """Low-pass windowed-sinc kernel, DSPG Eq 16-4. fc and bw are fractions of the sample rate (0..0.5);
    fc is the 50 % amplitude (-6 dB) point, bw the transition width. Unity DC gain, M+1 taps, M even,
    linear phase with delay M/2."""
    fc = float(np.clip(fc, 1e-5, 0.5))
    M = int(taps) - 1 if taps else int(math.ceil(_WINDOW_K[window] / max(bw, 1e-5)))
    M += M % 2
    M = max(M, 2)
    i = np.arange(M + 1) - M / 2
    h = 2 * fc * np.sinc(2 * fc * i) * _window(window, M + 1)
    return h / h.sum()


def spectral_invert(h: np.ndarray) -> np.ndarray:
    """LP <-> HP, BP <-> BR: negate the kernel and add 1 at the centre of symmetry (DSPG ch.14)."""
    g = -np.asarray(h, dtype=np.float64)
    g[(g.shape[0] - 1) // 2] += 1.0
    return g


def spectral_reverse(h: np.ndarray) -> np.ndarray:
    """Flip the response left-for-right (cutoff f -> 0.5 - f): multiply by (-1)^n (DSPG ch.14)."""
    g = np.array(h, dtype=np.float64)
    g[1::2] *= -1.0
    return g


def _bw(bw_hz, edge_hz, sr):
    if bw_hz is None:
        bw_hz = max(0.25 * edge_hz, 60.0 * sr / 44100.0)
    return bw_hz / sr


def fir_lowpass(fc, sr=DEFAULT_SR, bw=None, window="blackman"):
    """Linear-phase low-pass, cutoff fc Hz (-6 dB), transition width bw Hz (default fc/4)."""
    return windowed_sinc(fc / sr, _bw(bw, fc, sr), window)


def fir_highpass(fc, sr=DEFAULT_SR, bw=None, window="blackman"):
    return spectral_invert(fir_lowpass(fc, sr, bw, window))


def fir_bandstop(lo, hi, sr=DEFAULT_SR, bw=None, window="blackman"):
    """DSPG Table 16-2: band-reject = low-pass(lo) + high-pass(hi) with equal kernel lengths."""
    b = _bw(bw, lo, sr)
    a = windowed_sinc(lo / sr, b, window)
    return a + spectral_invert(windowed_sinc(hi / sr, b, window, taps=a.shape[0]))


def fir_bandpass(lo, hi, sr=DEFAULT_SR, bw=None, window="blackman"):
    """DSPG Table 16-2: band-pass = spectral inversion of the band-reject kernel."""
    return spectral_invert(fir_bandstop(lo, hi, sr, bw, window))


def min_phase_from_mag(mag: np.ndarray, clip_db: float = -100.0) -> np.ndarray:
    """Minimum-phase impulse response with the given magnitude (sampled at the rfft bins of an even
    length N = 2*(len-1)) by cepstral folding: c = ifft(log|H|); double the causal part, zero the
    anticausal part; h = ifft(exp(fft(c))). Source: FILT mps.m (g14 B.2). N must be large enough that the
    cepstrum does not wrap; the magnitude is floored at clip_db."""
    mag = np.asarray(mag, dtype=np.float64)
    n = 2 * (mag.shape[0] - 1)
    c = np.fft.irfft(np.log(np.maximum(mag, mag.max() * 10 ** (clip_db / 20))), n)
    c[1:n // 2] *= 2.0
    c[n // 2 + 1:] = 0.0
    return np.fft.irfft(np.exp(np.fft.rfft(c)), n)


def min_phase(h: np.ndarray, nfft: int | None = None) -> np.ndarray:
    """Minimum-phase version of a kernel (same magnitude, energy packed at the start, no pre-ringing)."""
    n = nfft or 1 << int(math.ceil(math.log2(8 * h.shape[0])))
    return min_phase_from_mag(np.abs(np.fft.rfft(h, n)))[: h.shape[0]]


def fir_from_curve(freqs, gains_db, sr=DEFAULT_SR, taps: int | None = None, phase: str = "linear",
                   window: str = "blackman") -> np.ndarray:
    """Arbitrary magnitude response by frequency sampling (DSPG ch.17 steps 1-5): sample the curve on a
    dense grid, inverse FFT, rotate by M/2, truncate to M+1 taps, window. `phase="min"` builds the
    minimum-phase response by cepstral folding (g14 B.2) and fades its tail instead. The curve is
    interpolated monotonically (PCHIP) in dB over log-frequency and held flat outside the given points."""
    f = np.asarray(freqs, dtype=np.float64)
    g = np.asarray(gains_db, dtype=np.float64)
    order = np.argsort(f)
    f, g = np.maximum(f[order], 1e-3), g[order]
    if taps is None:
        taps = int(0.093 * sr) | 1  # ~4097 at 44.1 kHz: main lobe 6*sr/M = 65 Hz
    M = (int(taps) - 1) // 2 * 2
    N = 1 << int(math.ceil(math.log2(8 * (M + 1))))
    grid = np.fft.rfftfreq(N, 1.0 / sr)
    lf = np.log(np.clip(grid, f[0], f[-1]))
    db = g[0] * np.ones_like(grid) if f.shape[0] == 1 else PchipInterpolator(np.log(f), g)(lf)
    mag = 10 ** (db / 20)
    if phase == "linear":
        h = np.roll(np.fft.irfft(mag, N), M // 2)[: M + 1] * _window(window, M + 1)
    elif phase in ("min", "minimum"):
        h = min_phase_from_mag(mag)[: M + 1].copy()
        k = (M + 1) // 2
        h[-k:] *= _window(window, 2 * k)[k:]
    else:
        raise ValueError("phase must be linear|min")
    return h


def fft_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Exact linear convolution, output length N + M - 1 (the tail is kept). Direct below 64 taps, FFT
    overlap-add above (DSPG ch.18, crossover ~60 taps; MDFT timings, g14 D). x mono or (n, 2); h mono."""
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    if h.shape[0] < 64:
        return _per_channel(lambda s: np.convolve(s, h), x)
    return _per_channel(lambda s: sg.oaconvolve(s, h, mode="full"), x)


def fir_filter(x: np.ndarray, h: np.ndarray, compensate: bool | None = None) -> np.ndarray:
    """Filter with a FIR kernel and return the same length as x. A symmetric (linear-phase) kernel is
    delay-compensated by (taps-1)/2 samples (DSPG ch.16/29), so the output lines up with the input."""
    if compensate is None:
        compensate = bool(np.allclose(h, h[::-1], atol=1e-12))
    d = (h.shape[0] - 1) // 2 if compensate else 0
    return fft_convolve(x, h)[d: d + x.shape[0]]


# ======================================================================================================
# 3. Recursive filters: Smith's Chebyshev-Butterworth program (DSPG ch.20 Tables 20-4/20-5, ch.33)
# ======================================================================================================


def smith_stage(fc: float, highpass: bool, pr: float, npole: int, p: int):
    """One pole pair of Smith's program (DSPG Table 20-5). fc = cutoff as a fraction of fs (0..0.5),
    pr = percent pass-band ripple 0..29 (0 = Butterworth), npole even, p = 1..npole/2.
    Returns a0, a1, a2, b1, b2 in Smith's convention y = a0 x + a1 x1 + a2 x2 + b1 y1 + b2 y2 (b ADDED),
    before gain normalisation."""
    rp = -math.cos(math.pi / (2 * npole) + (p - 1) * math.pi / npole)
    ip = math.sin(math.pi / (2 * npole) + (p - 1) * math.pi / npole)
    if pr != 0:  # warp the circle into an ellipse
        es = math.sqrt((100.0 / (100.0 - pr)) ** 2 - 1)
        vx = (1.0 / npole) * math.log(1 / es + math.sqrt(1 / es ** 2 + 1))
        kx = (1.0 / npole) * math.log(1 / es + math.sqrt(1 / es ** 2 - 1))
        kx = (math.exp(kx) + math.exp(-kx)) / 2
        rp *= ((math.exp(vx) - math.exp(-vx)) / 2) / kx
        ip *= ((math.exp(vx) + math.exp(-vx)) / 2) / kx
    t = 2 * math.tan(0.5)
    w = 2 * math.pi * fc
    m = rp * rp + ip * ip
    d = 4 - 4 * rp * t + m * t * t
    x0, x1, x2 = t * t / d, 2 * t * t / d, t * t / d
    y1, y2 = (8 - 2 * m * t * t) / d, (-4 - 4 * rp * t - m * t * t) / d
    if highpass:
        k = -math.cos(w / 2 + 0.5) / math.cos(w / 2 - 0.5)
    else:
        k = math.sin(0.5 - w / 2) / math.sin(0.5 + w / 2)
    d = 1 + y1 * k - y2 * k * k
    a0 = (x0 - x1 * k + x2 * k * k) / d
    a1 = (-2 * x0 * k + x1 + x1 * k * k - 2 * x2 * k) / d
    a2 = (x0 * k * k - x1 * k + x2) / d
    b1 = (2 * k + y1 + y1 * k * k - 2 * y2 * k) / d
    b2 = (-(k * k) - y1 * k + y2) / d
    if highpass:
        a1, b1 = -a1, -b1
    return a0, a1, a2, b1, b2


def smith_sos(fc: float, npole: int = 4, kind: str = "lowpass", ripple: float = 0.0) -> np.ndarray:
    """Cascade of npole/2 second-order sections from Smith's program, each stage normalised to unity gain
    at DC (low-pass) or Nyquist (high-pass) (DSPG Eq 33-7/33-8), as scipy sos rows [a0 a1 a2 1 -b1 -b2].
    ripple 0 = true Butterworth (-3 dB at fc, stage Qs 1/(2 sin((2k+1)pi/2N)), not N equal-Q biquads)."""
    npole = int(np.clip(2 * round(npole / 2), 2, 20))
    fc = float(np.clip(fc, 1e-5, 0.4999))
    pr = float(np.clip(ripple, 0.0, 29.0))
    hp = kind == "highpass"
    rows = []
    for p in range(1, npole // 2 + 1):
        a0, a1, a2, b1, b2 = smith_stage(fc, hp, pr, npole, p)
        g = (a0 - a1 + a2) / (1 + b1 - b2) if hp else (a0 + a1 + a2) / (1 - b1 - b2)
        rows.append([a0 / g, a1 / g, a2 / g, 1.0, -b1, -b2])
    return np.array(rows)


def smith_coefficients(sos: np.ndarray):
    """Combined a[] and b[] of a cascade in Smith's convention (for comparison with DSPG Table 20-1).
    Do not run high orders in this form: DSPG Table 20-3 lists where it goes unstable."""
    a, d = np.array([1.0]), np.array([1.0])
    for row in sos:
        a, d = np.convolve(a, row[:3]), np.convolve(d, row[3:])
    return a, -d[1:]


def iir_sos(kind: str, fc, sr=DEFAULT_SR, order: int = 4, family: str = "butter", ripple: float = 0.5) -> np.ndarray:
    """Steep recursive low/high-pass as second-order sections. family butter|cheby (Smith's program;
    `ripple` = percent pass-band ripple, cheby only) | bessel (scipy, magnitude-normalised: -3 dB at fc,
    near-linear phase, no overshoot; not in Smith's program). order = number of poles, even, 2..20."""
    order = int(np.clip(2 * round(order / 2), 2, 20))
    f = float(np.clip(fc, 10.0, 0.49 * sr)) / sr
    if family in ("butter", "butterworth"):
        return smith_sos(f, order, kind, 0.0)
    if family in ("cheby", "chebyshev", "cheby1"):
        return smith_sos(f, order, kind, ripple)
    if family == "bessel":
        return sg.bessel(order, 2 * f, btype=kind, norm="mag", output="sos")
    raise ValueError("family must be butter|cheby|bessel")


def steep_filter(x, kind, fc, sr=DEFAULT_SR, order=4, family="butter", ripple=0.5, fc2=None):
    """Apply iir_sos. kind lowpass|highpass|bandpass (bandpass = high-pass at fc then low-pass at fc2)."""
    if kind == "bandpass":
        lo, hi = sorted((float(fc), float(fc2 if fc2 is not None else fc * 2)))
        sos = np.vstack([iir_sos("highpass", lo, sr, order, family, ripple), iir_sos("lowpass", hi, sr, order, family, ripple)])
    else:
        sos = iir_sos(kind, fc, sr, order, family, ripple)
    return sg.sosfilt(sos, x, axis=0)


def dc_blocker(x, sr=DEFAULT_SR, cutoff: float = 20.0):
    """JOS DC blocker H = g (1 - z^-1)/(1 - R z^-1), g = (1+R)/2 for unity gain at Nyquist; adaptation
    time constant 1/(1-R) samples (FILT "DC Blocker", g14 B.3). R is set from the cutoff."""
    R = float(np.clip(1 - 2 * np.pi * cutoff / sr, 0.5, 0.99999))
    g = (1 + R) / 2
    return sg.lfilter([g, -g], [1, -R], x, axis=0)


def moving_average(x, m: int, passes: int = 1):
    """Symmetric M-point moving average by running sum, repeated `passes` times (1 = box, 2 = triangle,
    4+ ~ Gaussian; response (sin(pi f M)/(M sin(pi f)))^passes, DSPG Eq 15-2), with normalised edges
    (divide by the same filter applied to ones, DSPG ch.24). Zero delay; M is forced odd."""
    m = max(1, int(m)) | 1
    if m == 1:
        return np.array(x, dtype=np.float64)
    p = m // 2

    def box(s):
        c = np.cumsum(np.concatenate([np.zeros(p + 1), s, np.zeros(p)]))
        return c[m:] - c[:-m]

    def one(s):
        y = np.asarray(s, dtype=np.float64)
        norm = box(np.ones(y.shape[0]))
        for _ in range(max(1, int(passes))):
            y = box(y) / norm
        return y
    return _per_channel(one, np.asarray(x, dtype=np.float64))


# ======================================================================================================
# 4. Impulse-response generators (DSPG ch.2, 11, 16, 18; g14 B.2 minimum phase)
# ======================================================================================================

_BAND_EDGES = (177.0, 354.0, 707.0, 1414.0, 2828.0, 5657.0)  # octave-band edges around 125 Hz .. 8 kHz


def ir_noise(sr=DEFAULT_SR, rt60=1.0, rt60_high=None, dur=None, seed=0) -> np.ndarray:
    """Dense reverb tail: Gaussian noise (DSPG ch.2) split into octave bands with windowed-sinc kernels
    whose sum is an exact delayed impulse (DSPG ch.16 "bandsplit"), each band decayed with its own
    exp(-6.91 t / RT60) (t60 = ln(1000) tau, MDFT; g14 D). rt60 = RT60 of the lowest band, rt60_high =
    RT60 of the highest (log-interpolated between); a scalar pair or a list with one value per band (7)."""
    edges = [e for e in _BAND_EDGES if e < 0.45 * sr]
    nb = len(edges) + 1
    if np.ndim(rt60) > 0:
        rts = np.interp(np.linspace(0, 1, nb), np.linspace(0, 1, len(rt60)), np.asarray(rt60, dtype=float))
    else:
        hi = rt60 if rt60_high is None else rt60_high
        rts = np.exp(np.linspace(np.log(max(rt60, 1e-3)), np.log(max(hi, 1e-3)), nb))
    n = samples(dur if dur else 1.25 * float(np.max(rts)) + 0.02, sr)
    w = np.random.default_rng(seed).standard_normal(n)
    taps = windowed_sinc(edges[0] / sr, 0.5 * edges[0] / sr).shape[0]
    lps = [windowed_sinc(e / sr, 0.5 * edges[0] / sr, taps=taps) for e in edges]
    delta = np.zeros(taps)
    delta[taps // 2] = 1.0
    kernels = [lps[0]] + [lps[i] - lps[i - 1] for i in range(1, len(lps))] + [delta - lps[-1]]
    t = np.arange(n) / sr
    out = np.zeros(n)
    for k, rt in zip(kernels, rts):
        out += fir_filter(w, k, compensate=True) * np.exp(-6.9078 * t / rt)
    return out


def ir_early(sr=DEFAULT_SR, gap=0.015, count=8, span=4.0, seed=0) -> np.ndarray:
    """Sparse early reflections: `count` taps between `gap` and `gap*span` seconds, amplitude falling as
    gap/t (spherical spreading) times a wall reflection factor. Spacing guide (DSPG ch.1/22): 10-20 ms =
    modest room, ~100 ms = auditorium."""
    rng = np.random.default_rng(seed)
    t = np.sort(gap * (1 + (span - 1) * rng.random(int(count)) ** 1.5))
    t[0] = gap
    h = np.zeros(samples(gap * span, sr) + 2)
    for i, ti in enumerate(t):
        refl = 0.8 ** (1 + i // 3)  # UNSOURCED: wall reflection coefficient 0.8 per reflection order
        h[int(ti * sr)] += (gap / ti) * refl * (1.0 if rng.random() < 0.8 else -1.0)
    return h


def chirp_allpass(n: int, tau0: float, tau1: float) -> np.ndarray:
    """Smith's "chirp system" (DSPG Eq 11-7): unit magnitude, phase alpha*k + beta*k^2. Here expressed by
    its group delay, rising linearly from tau0 samples at DC to tau1 at Nyquist (tau1 < tau0 = high
    frequencies first, the anti-chirp). tau0 + tau1 is rounded to an even number so the Nyquist phase is a
    multiple of pi (real spectrum end points). Returns an n-sample impulse response (n even)."""
    n = int(n) + int(n) % 2
    s = 2 * round((tau0 + tau1) / 2)
    tau1 = s - tau0
    k = np.arange(n // 2 + 1)
    alpha = 2 * np.pi * tau0 / n
    beta = 2 * np.pi * (tau1 - tau0) / (n * n)
    return np.fft.irfft(np.exp(-1j * (alpha * k + beta * k * k)), n)


def ir_spring(sr=DEFAULT_SR, decay=2.0, size=0.5, bright=0.5, seed=0) -> np.ndarray:
    """Spring-like IR: a dispersive delay loop H = gA/(1 - gA) where A is an allpass whose group delay
    rises with frequency up to a transition frequency (each echo is a rising chirp, and each round trip
    disperses more), built from the chirp-system idea of DSPG ch.11. The loop gain per frequency is
    g = 10^(-3 tau(f)/decay), so every frequency has the requested RT60."""
    rng = np.random.default_rng(seed)
    fcut = min((2500.0 + 3500.0 * bright), 0.45 * sr)  # UNSOURCED: spring transition frequency 2.5-6 kHz
    N = 1 << int(math.ceil(math.log2((1.6 * decay + 0.3) * sr)))
    f = np.fft.rfftfreq(N, 1.0 / sr)
    out = np.zeros(N)
    for j in range(2):  # UNSOURCED: two springs of slightly different length, as in common tanks
        t0 = (0.030 + 0.035 * size) * (1 + 0.23 * j) * (1 + 0.02 * rng.standard_normal())
        tau = t0 * (1 + 1.8 * np.clip(f / fcut, 0, 1) ** 2)  # UNSOURCED: delay ratio 2.8 across the band
        phase = 2 * np.pi * np.cumsum(tau) * (f[1] - f[0])
        g = 10 ** (-3 * tau / max(decay, 0.05)) / np.sqrt(1 + (f / fcut) ** 12)
        A = g * np.exp(-1j * phase)
        out += np.fft.irfft(A / (1 - A), N) * (1.0 if j == 0 else -0.8)
    return out[: samples(1.4 * decay + 0.2, sr)]


def ir_plate(sr=DEFAULT_SR, decay=1.5, size=0.5, bright=0.6, seed=0) -> np.ndarray:
    """Plate-like IR: dense Gaussian tail from t = 0 (no discrete early reflections) passed through an
    anti-chirp allpass, so high frequencies arrive first as bending waves do on a plate (DSPG ch.11)."""
    tail = ir_noise(sr, decay, decay * (0.35 + 0.6 * bright), seed=seed)
    sweep = (0.008 + 0.03 * size) * sr  # UNSOURCED: 8-38 ms dispersion spread
    return fft_convolve(tail, chirp_allpass(int(2 * sweep) + 64, sweep, 2.0))


def ir_room(sr=DEFAULT_SR, decay=0.5, size=0.4, bright=0.5, seed=0, hall=False) -> np.ndarray:
    """Room / hall IR = sparse early reflections + per-band decaying Gaussian tail that fades in over the
    early window. First-reflection gap 4-20 ms (room) or 20-100 ms (hall) with size (DSPG ch.1/22)."""
    gap = (0.020 + 0.080 * size) if hall else (0.004 + 0.016 * size)
    early = ir_early(sr, gap, 10 if hall else 7, 4.0, seed)
    tail = ir_noise(sr, decay * 1.1, decay * (0.25 + 0.6 * bright), seed=seed + 1)  # UNSOURCED: HF decay ratio
    t = np.arange(tail.shape[0]) / sr
    tail *= 1 - np.exp(-t / (1.5 * gap))
    tail *= np.sqrt(np.sum(early ** 2) / max(np.sum(tail ** 2), 1e-12))  # UNSOURCED: early and late energy equal
    out = tail.copy()
    out[: early.shape[0]] += early
    return out


# colour curves, (Hz, dB). UNSOURCED: hand-drawn typical responses, not measurements.
_CURVES = {
    "cabinet": [(20, -30), (60, -12), (90, -2), (120, 3), (250, 0), (1000, 0), (2500, 4), (4000, 1), (5000, -6), (7000, -24), (12000, -50)],
    "phone-speaker": [(100, -40), (300, -18), (500, -6), (800, 0), (1200, 3), (2500, 5), (3500, 0), (5000, -18), (8000, -40)],
}


def ir_curve(points, sr=DEFAULT_SR, shift=1.0, taps=None) -> np.ndarray:
    """Body / cabinet IR: minimum phase from a magnitude curve (preferred over linear phase for
    impulsive sounds: no pre-ringing; FILT listening test, g14 B.2). `shift` scales the frequency axis."""
    f = [p[0] * shift for p in points]
    return fir_from_curve(f, [p[1] for p in points], sr, taps or (int(0.05 * sr) | 1), phase="min")


def ir_tube(sr=DEFAULT_SR, decay=0.15, size=0.3, bright=0.5) -> np.ndarray:
    """Pipe closed at one end: resonances at odd quarter-wave multiples f_n = (2n-1) c / (4 L), each a
    decaying sinusoid (any stable resonant IR is a sum of them, FILT PFE; g14 A). L = 0.1..2 m with size."""
    L = 0.1 + 1.9 * size
    n = samples(1.3 * decay + 0.01, sr)
    t = np.arange(n) / sr
    h = np.zeros(n)
    k = 1
    while (2 * k - 1) * 343.0 / (4 * L) < min(8000.0, 0.45 * sr) and k < 200:
        fk = (2 * k - 1) * 343.0 / (4 * L)
        tk = decay / math.sqrt(k)  # UNSOURCED: higher modes decay faster as 1/sqrt(mode number)
        h += k ** (-(1.6 - bright)) * np.exp(-6.9078 * t / tk) * np.sin(2 * np.pi * fk * t)  # UNSOURCED tilt
        k += 1
    h[0] += 0.3  # some direct sound
    return h


IR_KINDS = ("room", "hall", "plate", "spring", "cabinet", "tube", "car", "phone-speaker")
_IR_DECAY = {"room": 0.45, "hall": 2.2, "plate": 1.6, "spring": 2.0, "tube": 0.15, "car": 0.09}  # UNSOURCED defaults
_SPACE_KINDS = ("room", "hall", "plate", "spring")


def make_ir(kind="room", sr=DEFAULT_SR, decay=0.0, size=0.5, bright=0.5, seed=0) -> np.ndarray:
    """Generated impulse response by name (see IR_KINDS). decay = RT60 seconds (0 = the kind's default).
    Space kinds are normalised to unit energy (a noise-like input keeps its RMS); cabinet/phone-speaker
    keep the 0 dB reference of their curve; tube/car have 0 dB at their strongest resonance."""
    size = float(np.clip(size, 0.0, 1.0))
    bright = float(np.clip(bright, 0.0, 1.0))
    decay = float(np.clip(decay if decay and decay > 0 else _IR_DECAY.get(kind, 0.3), 0.02, 12.0))
    if kind == "room":
        h = ir_room(sr, decay, size, bright, seed)
    elif kind == "hall":
        h = ir_room(sr, decay, size, bright, seed, hall=True)
    elif kind == "plate":
        h = ir_plate(sr, decay, size, bright, seed)
    elif kind == "spring":
        h = ir_spring(sr, decay, size, bright, seed)
    elif kind in _CURVES:
        return ir_curve(_CURVES[kind], sr, shift=(1.6 - 1.2 * size) * (0.8 + 0.4 * bright))
    elif kind == "tube":
        h = ir_tube(sr, decay, size, bright)
    elif kind == "car":
        # small cabin: reflections 1.5-5 ms apart, very short dull tail, direct sound kept
        h = ir_room(sr, decay, 0.1 * size, 0.2 * bright, seed)
        h = np.concatenate([[1.0], 0.6 * h / math.sqrt(np.sum(h ** 2))])
        h = F.lowshelf(h, 120.0, sr, 4.0)  # UNSOURCED: +4 dB cabin gain below 120 Hz
    else:
        raise ValueError(f"unknown ir kind {kind!r}; one of {', '.join(IR_KINDS)}")
    if kind in ("tube", "car"):  # resonant colours: 0 dB at the strongest resonance, so tones are never boosted
        return h / np.max(np.abs(np.fft.rfft(h, 1 << int(math.ceil(math.log2(4 * h.shape[0]))))))
    return h / math.sqrt(max(np.sum(h ** 2), 1e-20))


# ======================================================================================================
# 5. Companding, dither, codecs (DSPG ch.3, 22, 27; MDFT mu-law, g14 E)
# ======================================================================================================

MU = 255.0  # DSPG Eq 22-1
A_LAW = 87.6  # DSPG Eq 22-2


def mulaw_compress(x, mu=MU):
    """y = sign(x) ln(1 + mu|x|)/ln(1 + mu), x in [-1, 1] (DSPG Eq 22-1)."""
    x = np.clip(x, -1.0, 1.0)
    return np.sign(x) * np.log1p(mu * np.abs(x)) / math.log1p(mu)


def mulaw_expand(y, mu=MU):
    """x = sign(y) ((1 + mu)^|y| - 1)/mu (inverse of Eq 22-1)."""
    return np.sign(y) * np.expm1(np.abs(y) * math.log1p(mu)) / mu


def alaw_compress(x, a=A_LAW):
    """DSPG Eq 22-2: linear A|x|/(1+ln A) below 1/A, (1 + ln(A|x|))/(1 + ln A) above."""
    x = np.clip(x, -1.0, 1.0)
    ax = np.abs(x)
    k = 1 + math.log(a)
    return np.sign(x) * np.where(ax < 1 / a, a * ax / k, (1 + np.log(np.maximum(a * ax, 1e-300))) / k)


def alaw_expand(y, a=A_LAW):
    ay = np.abs(y)
    k = 1 + math.log(a)
    return np.sign(y) * np.where(ay < 1 / k, ay * k / a, np.exp(ay * k - 1) / a)


@numba.njit(cache=True)
def _shaped_quantize(v, d, lim):
    out = np.empty(v.shape[0])
    e = 0.0
    for i in range(v.shape[0]):
        u = v[i] - e
        q = np.round(u + d[i])
        if q > lim:
            q = lim
        elif q < -lim:
            q = -lim
        e = q - u
        out[i] = q
    return out


def quantize(x, bits=8, dither="tpdf", seed=0):
    """Round to `bits` (LSB = 2/2^bits, levels +-2^(bits-1)) with dither added before rounding.
    dither: none | tpdf (sum of two uniforms, +-1 LSB; the standard, outside Smith's text) | gauss (sd 2/3
    LSB, DSPG ch.3) | shaped (TPDF + first-order error feedback: quantisation noise shaped by 1 - z^-1,
    i.e. pushed to high frequencies; the concept of DSPG ch.3 delta-sigma / MDFT noise-shaped dither)."""
    x = np.asarray(x, dtype=np.float64)
    S = float(2 ** (int(np.clip(bits, 1, 24)) - 1))
    rng = np.random.default_rng(seed)
    if dither in ("none", "", None, False):
        d = np.zeros(x.shape)
    elif dither == "gauss":
        d = rng.standard_normal(x.shape) * (2.0 / 3.0)
    elif dither in ("tpdf", "shaped", True):
        d = rng.random(x.shape) + rng.random(x.shape) - 1.0
    else:
        raise ValueError("dither must be none|tpdf|gauss|shaped")
    if dither == "shaped":
        if x.ndim == 1:
            return _shaped_quantize(x * S, d, S) / S
        return np.stack([_shaped_quantize(np.ascontiguousarray(x[:, c]) * S, np.ascontiguousarray(d[:, c]), S) for c in range(x.shape[1])], axis=1) / S
    return np.clip(np.round(x * S + d), -S, S) / S


@numba.njit(cache=True)
def _cvsd_core(x, bits, decode, step_min, step_max, syl, leak, run):
    n = bits.shape[0]
    out = np.empty(n)
    est = 0.0
    step = step_min
    last = -1
    cnt = 0
    for i in range(n):
        if decode:
            b = bits[i]
        else:
            b = 1 if x[i] > est else 0
            bits[i] = b
        if b == last:
            cnt += 1
        else:
            cnt = 1
            last = b
        if run > 0:
            target = step_max if cnt >= run else step_min
            step = syl * step + (1.0 - syl) * target
        est = leak * est + (step if b == 1 else -step)
        out[i] = est
    return out


def delta_encode(x, step=0.02):
    """Delta modulation (DSPG ch.3): 1 bit per clock, bit = x > integrator; integrator += +-step.
    Slew limit = step * clock. Returns (bits uint8, reconstruction)."""
    bits = np.zeros(x.shape[0], dtype=np.uint8)
    rec = _cvsd_core(np.ascontiguousarray(x, dtype=np.float64), bits, False, step, step, 0.0, 1.0, 0)
    return bits, rec


def delta_decode(bits, step=0.02):
    return _cvsd_core(np.zeros(1), np.ascontiguousarray(bits, dtype=np.uint8), True, step, step, 0.0, 1.0, 0)


def _cvsd_params(clock, step_min, step_max, syllabic, leak):
    syl = math.exp(-1.0 / (syllabic * clock))
    return step_min, step_max, syl, math.exp(-1.0 / (leak * clock))


def cvsd_encode(x, clock=30000, step_min=0.001, step_max=0.15, syllabic=0.005, leak=0.001, run=4):
    """CVSD (DSPG ch.3): delta modulation whose step grows while the last `run` (4) bits are equal
    (slew-limited) and falls back otherwise, through a "syllabic filter". Clock 30 kHz and 2000 levels
    (step_min = 2/2000) are Smith's numbers. `x` must already be sampled at `clock`.
    Returns (bits, reconstruction); the decoder runs the identical step logic."""
    # UNSOURCED: step_max 0.15, syllabic time constant 5 ms, integrator leak 1 ms (not given in the book)
    bits = np.zeros(x.shape[0], dtype=np.uint8)
    rec = _cvsd_core(np.ascontiguousarray(x, dtype=np.float64), bits, False, *_cvsd_params(clock, step_min, step_max, syllabic, leak), run)
    return bits, rec


def cvsd_decode(bits, clock=30000, step_min=0.001, step_max=0.15, syllabic=0.005, leak=0.001, run=4):
    return _cvsd_core(np.zeros(1), np.ascontiguousarray(bits, dtype=np.uint8), True, *_cvsd_params(clock, step_min, step_max, syllabic, leak), run)


# IMA/DVI ADPCM tables. UNSOURCED in the findings (Smith gives no step-adaptation constants, DSPG ch.27):
# these are the published IMA reference tables.
_IMA_INDEX = np.array([-1, -1, -1, -1, 2, 4, 6, 8], dtype=np.int64)
_IMA_STEP = np.array([
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97, 107,
    118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796, 876, 963,
    1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894,
    6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794,
    32767], dtype=np.int64)


@numba.njit(cache=True)
def _ima(pcm, codes, decode):
    n = codes.shape[0]
    out = np.empty(n, dtype=np.int64)
    pred = 0
    idx = 0
    for i in range(n):
        step = _IMA_STEP[idx]
        if decode:
            code = codes[i]
        else:
            diff = pcm[i] - pred
            code = 0
            if diff < 0:
                code = 8
                diff = -diff
            s = step
            if diff >= s:
                code |= 4
                diff -= s
            s >>= 1
            if diff >= s:
                code |= 2
                diff -= s
            s >>= 1
            if diff >= s:
                code |= 1
            codes[i] = code
        vp = step >> 3
        if code & 4:
            vp += step
        if code & 2:
            vp += step >> 1
        if code & 1:
            vp += step >> 2
        pred = pred - vp if code & 8 else pred + vp
        if pred > 32767:
            pred = 32767
        elif pred < -32768:
            pred = -32768
        idx += _IMA_INDEX[code & 7]
        if idx < 0:
            idx = 0
        elif idx > 88:
            idx = 88
        out[i] = pred
    return out


def adpcm_encode(x):
    """IMA-style 4-bit ADPCM: predict = previous reconstruction, quantise the difference to sign + 3 bits
    of the adaptive step, decoder in the loop so errors do not accumulate (DPCM principle, DSPG ch.27).
    Returns (codes uint8 0..15, reconstruction float)."""
    pcm = np.round(np.clip(x, -1, 1) * 32767).astype(np.int64)
    codes = np.zeros(pcm.shape[0], dtype=np.uint8)
    rec = _ima(pcm, codes, False)
    return codes, rec / 32768.0


def adpcm_decode(codes):
    return _ima(np.zeros(1, dtype=np.int64), np.ascontiguousarray(codes, dtype=np.uint8), True) / 32768.0


def block_dct(x, block=512, qmax=64.0, power=2.0, bits=8):
    """Transform-coding artefact (DSPG ch.27, JPEG principle in 1-D): DCT each block, divide coefficient k
    by a table q[k] = 1 + (k/N)^power * qmax that grows with frequency, round, multiply back, inverse DCT.
    The step of coefficient k of the orthonormal DCT is q[k] / 2^(bits-1) (white noise of sd s has
    coefficients of sd s; a sine of amplitude A puts A*sqrt(N/2) in one coefficient, so tones survive where
    noise is removed). Blocks do not overlap: the seams and the block-rate noise modulation are part of
    the sound."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    block = int(max(8, block))
    pad = (-n) % block
    X = dct(np.concatenate([x, np.zeros(pad)]).reshape(-1, block), norm="ortho", axis=1)
    q = 1.0 + (np.arange(block) / block) ** power * qmax  # UNSOURCED: power-law table (Smith's are 8x8 JPEG tables)
    step = q / 2.0 ** (bits - 1)
    return idct(np.round(X / step) * step, norm="ortho", axis=1).reshape(-1)[:n]


def gsm_like(x, pbits=3, order=8):
    """GSM-full-rate-like coder for a signal ALREADY at 8 kHz: 20 ms frames, LPC (autocorrelation method,
    order 8), the prediction residual is kept on one of three decimate-by-3 grids per 5 ms sub-frame
    (regular pulse excitation), each pulse quantised to `pbits` bits relative to a log-quantised block
    maximum, then re-synthesised through 1/A(z). Not bit-exact GSM 06.10: the long-term predictor, the
    weighting filter and LAR quantisation are omitted. Speech frame model: DSPG ch.22 / 27 (LPC = predict
    from past samples, store the error)."""
    # UNSOURCED: frame/sub-frame sizes and RPE grid follow the published GSM 06.10 layout, not the findings
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    out = np.zeros(n)
    hist = np.zeros(order)
    zi = np.zeros(order)
    levels = 2 ** int(pbits)
    for s in range(0, n, 160):
        fr = x[s: s + 160]
        w = fr * np.hamming(fr.shape[0]) if fr.shape[0] > 1 else fr
        r = np.array([np.dot(w[: w.shape[0] - k], w[k:]) for k in range(order + 1)]) if fr.shape[0] > order else np.zeros(order + 1)
        if r[0] < 1e-10:
            A = np.concatenate([[1.0], np.zeros(order)])
        else:
            r[0] *= 1.0001  # white-noise correction keeps the normal equations well conditioned
            A = np.concatenate([[1.0], -solve_toeplitz(r[:order], r[1: order + 1])])
        e = sg.lfilter(A, [1.0], np.concatenate([hist, fr]))[order:]
        hist = np.concatenate([hist, fr])[-order:]
        eq = np.zeros_like(e)
        for b in range(0, e.shape[0], 40):
            sub = e[b: b + 40]
            m = int(np.argmax([np.sum(sub[k::3] ** 2) for k in range(3)]))
            pulses = sub[m::3]
            xmax = np.max(np.abs(pulses)) if pulses.size else 0.0
            if xmax > 1e-9:
                xmax = 2.0 ** (np.round(np.log2(xmax) * 4) / 4)  # log-quantised block maximum
                p = np.clip(np.floor(pulses / xmax * levels / 2), -levels / 2, levels / 2 - 1) + 0.5
                eq[b + m: b + 40: 3] = math.sqrt(3.0) * p * xmax * 2 / levels  # sqrt(3): one grid of three keeps the residual energy
        y, zi = sg.lfilter([1.0], A, eq, zi=zi)
        out[s: s + fr.shape[0]] = y
    return np.clip(out, -1.0, 1.0)


def zoh_droop(f, rate):
    """Zero-order-hold amplitude response |sin(pi f/rate)/(pi f/rate)| (DSPG Eq 3-1): 0.6366 at rate/2."""
    return np.abs(np.sinc(np.asarray(f, dtype=np.float64) / rate))


def zoh_resample(y, rate, sr):
    """Play samples taken at `rate` through a zero-order-hold DAC and re-sample at sr: every sample is
    held (repeated K times, an exact discrete hold), keeping the spectral images weighted by the sinc
    droop, then band-limited to sr/2 so the images that survive are real ones, not new aliases."""
    rate, sr = int(rate), int(sr)
    K = max(2, int(math.ceil(2.0 * sr / rate)))
    return resample_sinc(np.repeat(np.asarray(y, dtype=np.float64), K, axis=0), rate * K, sr)


def zoh_compensate(x):
    """Smith's 2-pole 1/sinc corrector for the hold droop, valid to 0.45 fs of the DAC rate
    (DSPG Fig 26-13c constants)."""
    return sg.lfilter([1.151692, -0.06794763, -0.07929603], [1.0, 0.1129629, -0.1062142], x, axis=0)


def oversampled(fn, x, factor: int = 4):
    """Run a memoryless nonlinearity at `factor` (1..8) times the sample rate: band-limited up-sampling,
    fn, low-pass and decimation. Harmonics created above the original Nyquist are filtered out instead of
    folding back as an inharmonic floor (harmonic aliasing, DSPG ch.11; fix in g11 top addition 4)."""
    factor = int(np.clip(factor, 1, 8))
    if factor == 1:
        return fn(np.asarray(x, dtype=np.float64))
    n = x.shape[0]
    return resample_sinc(fn(resample_sinc(x, 1, factor)), factor, 1)[:n]


# ======================================================================================================
# 6. Noise toolbox (DSPG ch.2, 9, 25, 34)
# ======================================================================================================


def gaussian(n, seed=0, bounded=False):
    """Gaussian noise, mean 0, sd 1. bounded=True uses Smith's central-limit generator (sum of 12
    uniforms - 6, tails cut at +-6 sigma; DSPG ch.2), else numpy's normal generator."""
    rng = np.random.default_rng(seed)
    if bounded:
        return rng.random((12, int(n))).sum(axis=0) - 6.0
    return rng.standard_normal(int(n))


def poisson_events(n, rate, sr=DEFAULT_SR, seed=0, sigma=0.0):
    """Impulse train of a Poisson process with `rate` events/s (counting statistics, DSPG ch.25: N events
    fluctuate by sqrt(N), so the level of a dense stream grows as sqrt(rate), 3 dB per doubling).
    sigma > 0 gives each event a log-normal amplitude (median 1)."""
    rng = np.random.default_rng(seed)
    ev = rng.poisson(max(rate, 0.0) / sr, int(n)).astype(np.float64)
    if sigma > 0:
        ev *= lognormal(int(n), sigma, seed + 1)
    return ev


def lognormal(n, sigma=0.5, seed=0):
    """Multiplicative random variation, median 1: exp(sigma * Gaussian), the limit of a product of many
    independent factors (DSPG ch.34: multiply, don't add, random factors for impact strengths, drop sizes)."""
    return np.exp(sigma * np.random.default_rng(seed).standard_normal(int(n)))


def one_over_f(n, sr=DEFAULT_SR, knee=100.0, seed=0):
    """Gaussian noise that is white above `knee` Hz and rises as 1/f (in power) below it: the "spectrum
    furniture" of real electronics (DSPG ch.9). Shaped in the frequency domain; sd 1 for the white part."""
    n = int(n)
    X = np.fft.rfft(np.random.default_rng(seed).standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    f[0] = f[1] if n > 1 else 1.0
    return np.fft.irfft(X * np.sqrt(1.0 + knee / np.maximum(f, 1.0)), n)


def mains_hum(n, sr=DEFAULT_SR, freq=50.0, harmonics=(1.0, 0.5, 0.3, 0.2), seed=0):
    """Mains hum: the fundamental (50 or 60 Hz) plus its harmonics (120/180/240 Hz for 60 Hz: the power
    waveform is not a pure sine, DSPG ch.9). Peak about 1."""
    # UNSOURCED: harmonic amplitudes 1, 0.5, 0.3, 0.2 (the book only says harmonics are present)
    rng = np.random.default_rng(seed)
    t = np.arange(int(n)) / sr
    y = np.zeros(int(n))
    for k, a in enumerate(harmonics, start=1):
        if k * freq < 0.45 * sr:
            y += a * np.sin(2 * np.pi * k * freq * t + rng.uniform(0, 2 * np.pi))
    return y / max(sum(harmonics), 1e-9)


# ======================================================================================================
# 7. Metering (MDFT appendix: dBA and VU, g14 E)
# ======================================================================================================

_A_POLES = (129.4, 129.4, 676.7, 4636.0, 76655.0, 76655.0)  # rad/s
_A_GAIN = 7.39705e9


def a_weighting_db(f):
    """Exact analog A-weighting in dB: H(s) = kA s^4 / ((s+129.4)^2 (s+676.7)(s+4636)(s+76655)^2)."""
    w = 2 * np.pi * np.asarray(f, dtype=np.float64)
    den = (w ** 2 + 129.4 ** 2) * np.sqrt(w ** 2 + 676.7 ** 2) * np.sqrt(w ** 2 + 4636.0 ** 2) * (w ** 2 + 76655.0 ** 2)
    return 20 * np.log10(np.maximum(_A_GAIN * w ** 4 / den, 1e-300))


def a_weighting_sos(sr=DEFAULT_SR):
    """Digital A-weighting: bilinear transform of the analog poles, gain set to 0 dB at 1 kHz. Matches the
    analog curve closely below ~sr/6; above that the bilinear warping adds extra attenuation."""
    z, p, k = sg.bilinear_zpk([0.0] * 4, [-p for p in _A_POLES], _A_GAIN, sr)
    sos = sg.zpk2sos(z, p, k)
    _, h = sg.sosfreqz(sos, worN=[1000.0], fs=sr)
    sos[0, :3] /= abs(h[0])
    return sos


def a_weight(x, sr=DEFAULT_SR):
    return sg.sosfilt(a_weighting_sos(sr), x, axis=0)


def vu_meter(x, sr=DEFAULT_SR, tau=0.06514):
    """VU ballistics: power averaged by a one-pole with time constant 65.14 ms (99 % = 40 dB of the step
    reached in t40 ~ 300 ms). Returns the RMS-like level per sample."""
    r = math.exp(-1.0 / (tau * sr))
    return np.sqrt(np.maximum(sg.lfilter([1 - r], [1, -r], np.square(to_mono(np.asarray(x, dtype=np.float64)))), 0.0))


# ======================================================================================================
# 8. Registered effects
# ======================================================================================================


def _fit(y, n):
    if y.shape[0] >= n:
        return y[:n]
    return np.concatenate([y, np.zeros((n - y.shape[0],) + y.shape[1:])], axis=0)


def _points(points):
    """'100:6,1000:0' or [[100, 6], ...] or [{'hz':..,'db':..}] -> (freqs, gains_db)."""
    if isinstance(points, str):
        pairs = [p.replace("=", ":").split(":") for p in points.replace(";", ",").split(",") if p.strip()]
    else:
        pairs = [(p["hz"], p["db"]) if isinstance(p, dict) else p for p in points]
    if not pairs or any(len(p) != 2 for p in pairs):
        raise ValueError("points must look like '100:6,1000:0,8000:-12' (Hz:dB)")
    return [float(p[0]) for p in pairs], [float(np.clip(float(p[1]), -120, 40)) for p in pairs]


@effect("convolve", "Convolution with an impulse response: a wav file or a generated space/body (tail is appended).",
        ir=("", "path to a wav impulse response; empty = generate `kind`"),
        kind=("room", "room|hall|plate|spring|cabinet|tube|car|phone-speaker"),
        decay=(0.0, "RT60 in seconds for room/hall/plate/spring/tube/car (0 = the kind's default)"),
        size=(0.5, "0..1: reflection spacing (room/hall), spring/tube length, cabinet size"),
        bright=(0.5, "0..1: high-frequency decay time / upper roll-off"),
        mix=(None, "wet amount 0..1 (default 0.3 for spaces, 1.0 for cabinet/tube/car/phone-speaker and files)"),
        predelay=(0.0, "seconds before the wet signal starts"), seed=(0, "int: which random room"))
def convolve(x, sr=DEFAULT_SR, ir="", kind="room", decay=0.0, size=0.5, bright=0.5, mix=None, predelay=0.0, seed=0):
    if ir:
        h, hsr = read_wav(ir)
        h = resample_sinc(to_mono(h), hsr, sr)
        h = h / math.sqrt(max(np.sum(h ** 2), 1e-20))
        auto = 1.0
    else:
        h = make_ir(kind, sr, decay, size, bright, int(seed))
        auto = 0.3 if kind in _SPACE_KINDS else 1.0
    mix = auto if mix is None else float(np.clip(mix, 0.0, 1.0))
    pd = int(max(predelay, 0.0) * sr)
    if pd:
        h = np.concatenate([np.zeros(pd), h])
    wet = fft_convolve(x, h)
    dry = _fit(np.asarray(x, dtype=np.float64), wet.shape[0])
    return dry * (1 - mix) + wet * mix


@effect("fir_eq", "Draw-your-own EQ curve as one FIR filter (frequency sampling).",
        points=("100:0,1000:0,8000:-6", "curve as 'hz:db,hz:db,...' (flat outside the end points)"),
        phase=("linear", "linear (no phase smear, delay compensated) | min (no pre-ringing)"),
        taps=(0, "kernel length; 0 = 93 ms (resolves ~65 Hz)"))
def fir_eq(x, sr=DEFAULT_SR, points="100:0,1000:0,8000:-6", phase="linear", taps=0):
    f, g = _points(points)
    h = fir_from_curve(f, g, sr, int(taps) if taps else None, phase)
    return fir_filter(x, h, compensate=(phase == "linear"))


_STEEP = dict(order=(8, "number of poles, even, 2..20 (6 dB/octave each)"), type=("butter", "butter|cheby|bessel"),
              ripple=(0.5, "cheby pass-band ripple in percent, 0..29"))


@effect("steep_lowpass", "Steep low-pass (true Butterworth / Chebyshev / Bessel cascade).", cutoff=(2000, "Hz (-3 dB)"), **_STEEP)
def steep_lowpass(x, sr=DEFAULT_SR, cutoff=2000, order=8, type="butter", ripple=0.5):
    return steep_filter(x, "lowpass", cutoff, sr, order, type, ripple)


@effect("steep_highpass", "Steep high-pass (true Butterworth / Chebyshev / Bessel cascade).", cutoff=(200, "Hz (-3 dB)"), **_STEEP)
def steep_highpass(x, sr=DEFAULT_SR, cutoff=200, order=8, type="butter", ripple=0.5):
    return steep_filter(x, "highpass", cutoff, sr, order, type, ripple)


@effect("steep_bandpass", "Steep band-pass: high-pass at `low` then low-pass at `high`, each of `order` poles.",
        low=(300, "Hz"), high=(3400, "Hz"), **_STEEP)
def steep_bandpass(x, sr=DEFAULT_SR, low=300, high=3400, order=8, type="butter", ripple=0.5):
    return steep_filter(x, "bandpass", low, sr, order, type, ripple, fc2=high)


def _tri_fold(v):
    return (2 / np.pi) * np.arcsin(np.sin(0.5 * np.pi * v))


# UNSOURCED: the curve shapes are voicing choices. The sourced rule (DSPG ch.11) is only: odd-symmetric
# curve -> odd harmonics; any asymmetry (bias, tube, diode) -> even harmonics too.
_SHAPES = {
    "tanh": np.tanh,
    "tube": lambda v: np.where(v >= 0, np.tanh(v), np.tanh(1.5 * v) / 1.5),
    "fuzz": lambda v: np.sign(v) * -np.expm1(-3.0 * np.abs(v)),
    "fold": lambda v: np.sin(0.5 * np.pi * v),
    "diode": lambda v: np.logaddexp(0.0, 4.0 * v) / 4.0,
}


def waveshape(x, shape="tanh", drive=4.0, bias=0.0, oversample=4):
    """Oversampled static waveshaper: f(drive*x + bias) - f(bias), scaled so a full-scale input stays
    within +-1."""
    if shape not in _SHAPES:
        raise ValueError(f"shape must be one of {'|'.join(_SHAPES)}")
    f = _SHAPES[shape]
    drive = float(np.clip(drive, 0.05, 100.0))
    bias = float(np.clip(bias, -2.0, 2.0))
    grid = np.linspace(-drive, drive, 201) + bias
    scale = max(float(np.max(np.abs(f(grid) - f(bias)))), 1e-6)
    return oversampled(lambda v: (f(drive * v + bias) - f(bias)) / scale, x, oversample)


@effect("overdrive", "Alias-free waveshaping overdrive (oversampled; bias adds even harmonics).",
        drive=(4.0, "input gain 0.05..100"), bias=(0.0, "-1..1 DC offset before the curve: asymmetry = even harmonics"),
        shape=("tanh", "tanh|tube|fuzz|fold|diode"), tone=(4000, "post low-pass Hz (0 = off)"),
        mix=(1.0, "wet 0..1"), oversample=(4, "1|2|4|8 (1 = naive, aliases)"))
def overdrive(x, sr=DEFAULT_SR, drive=4.0, bias=0.0, shape="tanh", tone=4000, mix=1.0, oversample=4):
    wet = waveshape(x, shape, drive, bias, oversample)
    if bias != 0 or shape in ("tube", "diode"):
        wet = dc_blocker(wet, sr)
    if tone and tone < 0.49 * sr:
        wet = F.lowpass(wet, tone, sr)
    return x * (1 - mix) + wet * mix


@effect("wavefold", "Wavefolder (oversampled): the wave is reflected back each time it passes full scale.",
        folds=(2.0, "input gain: about how many times a full-scale wave folds, 0.1..20"),
        symmetry=(0.0, "-1..1 offset before folding (even harmonics)"), shape=("sine", "sine|tri"),
        tone=(5000, "post low-pass Hz (0 = off)"), mix=(1.0, "wet 0..1"))
def wavefold(x, sr=DEFAULT_SR, folds=2.0, symmetry=0.0, shape="sine", tone=5000, mix=1.0):
    f = _tri_fold if shape == "tri" else _SHAPES["fold"]
    g = float(np.clip(folds, 0.1, 20.0))
    b = float(np.clip(symmetry, -1.0, 1.0))
    wet = oversampled(lambda v: f(g * v + b) - f(b), x, 8) / (1.0 + abs(f(b)))
    if b != 0:
        wet = dc_blocker(wet, sr)
    if tone and tone < 0.49 * sr:
        wet = F.lowpass(wet, tone, sr)
    return x * (1 - mix) + wet * mix


@effect("mulaw", "Companded quantisation (telephone codec grain: noise follows the signal level).",
        bits=(8, "2..16 bits in the compressed domain"), law=("mu", "mu (mu=255) | a (A=87.6)"))
def mulaw(x, sr=DEFAULT_SR, bits=8, law="mu"):
    comp, exp = (alaw_compress, alaw_expand) if law == "a" else (mulaw_compress, mulaw_expand)
    S = float(2 ** (int(np.clip(bits, 2, 16)) - 1) - 1)
    return exp(np.round(comp(x) * S) / S)


@effect("dither", "Reduce bit depth with dither (and optional noise shaping) instead of bare rounding.",
        bits=(8, "1..24"), shape=("tpdf", "none | tpdf | gauss (sd 2/3 LSB) | shaped (tpdf + first-order noise shaping)"),
        seed=(0, "int"))
def dither(x, sr=DEFAULT_SR, bits=8, shape="tpdf", seed=0):
    return quantize(x, bits, shape, int(seed))


def _at_rate(fn, x, sr, rate):
    """Run a mono coder at its own sample rate and come back band-limited."""
    n = x.shape[0]
    rate = int(rate)

    def one(s):
        y = fn(np.ascontiguousarray(resample_sinc(s, sr, rate)) if rate != sr else np.ascontiguousarray(s))
        return _fit(resample_sinc(y, rate, sr) if rate != sr else y, n)
    return _per_channel(one, np.clip(x, -1.0, 1.0))


def _geo(lo, hi, q):
    return lo * (hi / lo) ** float(np.clip(q, 0.0, 1.0))


@effect("codec", "Speech/streaming codec artefacts.",
        kind=("adpcm", "adpcm (4-bit IMA) | cvsd | delta | gsm (8 kHz LPC, GSM-like) | dct (block transform, streaming artefact)"),
        quality=(0.5, "0..1: adpcm rate 6-32 kHz; cvsd clock 12-64 kHz; delta clock 48-192 kHz; gsm pulse bits 2-4; dct coarse-fine"))
def codec(x, sr=DEFAULT_SR, kind="adpcm", quality=0.5):
    q = float(np.clip(quality, 0.0, 1.0))
    if kind == "adpcm":
        return _at_rate(lambda s: adpcm_encode(s)[1], x, sr, min(_geo(6000, 32000, q), sr))
    if kind == "cvsd":
        clock = int(_geo(12000, 64000, q))
        return _at_rate(lambda s: cvsd_encode(s, clock)[1], x, sr, clock)
    if kind == "delta":
        clock = int(_geo(48000, 192000, q))
        step = 2 * np.pi * 800.0 / clock  # UNSOURCED: slew limit set to a full-scale 800 Hz sine
        return _at_rate(lambda s: delta_encode(s, step)[1], x, sr, clock)
    if kind in ("gsm", "gsm-like"):
        y = steep_filter(x, "bandpass", 200.0, sr, 4, fc2=min(3400.0, 0.45 * sr))
        return _at_rate(lambda s: gsm_like(s, int(round(2 + 2 * q))), y, sr, 8000)
    if kind == "dct":
        block = int(512 * sr / 44100)  # 11.6 ms blocks
        qmax, power, bits = _geo(128, 4, q), 2.0, int(round(5 + 4 * q))  # UNSOURCED quality map
        return _per_channel(lambda s: block_dct(s, block, qmax, power, bits), np.clip(x, -1, 1))
    raise ValueError("codec kind must be adpcm|cvsd|delta|gsm|dct")


# UNSOURCED (all but `telephone`): sample rates, bit depths and bands are commonly quoted hardware figures,
# not from the findings. telephone = 8 kHz, 8-bit companded, 200-3200 Hz (DSPG ch.1/22, Table 22-2).
LOFI_PRESETS = {
    "nes": dict(rate=33144, bits=7, law="linear", band=None, hold=True, adpcm=False),
    "snes": dict(rate=16000, bits=16, law="linear", band=(20, 7000), hold=False, adpcm=True),
    "gameboy": dict(rate=8192, bits=4, law="linear", band=None, hold=True, adpcm=False),
    "amiga": dict(rate=16726, bits=8, law="linear", band=(20, 4400), hold=True, adpcm=False),
    "sp1200": dict(rate=26040, bits=12, law="linear", band=None, hold=True, adpcm=False),
    "telephone": dict(rate=8000, bits=8, law="mu", band=(200, 3200), hold=False, adpcm=False),
    "voicemail": dict(rate=8000, bits=16, law="linear", band=(300, 3000), hold=False, adpcm=True),
    "walkie": dict(rate=8000, bits=6, law="mu", band=(400, 2800), hold=False, adpcm=False),
}


@effect("lofi", "Old-hardware sampler/DAC: band-limited sample-rate reduction, bit depth with dither, zero-order hold.",
        preset=("sp1200", "nes|snes|gameboy|amiga|sp1200|telephone|voicemail|walkie|none"),
        rate=(0, "sample rate Hz (0 = preset; 11025 without a preset)"), bits=(0, "bit depth 1..16 (0 = preset; 8 without)"),
        dither=("tpdf", "none|tpdf|gauss|shaped"), law=("", "linear | mu | a (empty = preset)"),
        hold=(None, "true = zero-order-hold DAC (keeps the metallic images, sinc droop); false = clean reconstruction"),
        seed=(0, "int: dither noise"))
def lofi(x, sr=DEFAULT_SR, preset="sp1200", rate=0, bits=0, dither="tpdf", law="", hold=None, seed=0):
    if preset in ("", "none", None):
        p = dict(rate=11025, bits=8, law="linear", band=None, hold=True, adpcm=False)
    elif preset in LOFI_PRESETS:
        p = dict(LOFI_PRESETS[preset])
    else:
        raise ValueError(f"lofi preset must be one of {'|'.join(LOFI_PRESETS)}|none")
    r = int(np.clip(rate if rate else p["rate"], 500, 192000))
    b = int(np.clip(bits if bits else p["bits"], 1, 16))
    lw = law or p["law"]
    hd = p["hold"] if hold is None else bool(hold)
    n = x.shape[0]
    y = np.asarray(x, dtype=np.float64)
    if p["band"]:
        y = steep_filter(y, "bandpass", p["band"][0], sr, 4, fc2=min(p["band"][1], 0.45 * sr))
    low = r < sr
    if low:
        y = resample_sinc(y, sr, r)  # proper anti-aliased decimation
    y = np.clip(y, -1.0, 1.0)
    if p["adpcm"]:
        y = _per_channel(lambda s: adpcm_encode(s)[1], y)
    if lw == "mu":
        y = mulaw_expand(quantize(mulaw_compress(y), b, dither, int(seed)))
    elif lw == "a":
        y = alaw_expand(quantize(alaw_compress(y), b, dither, int(seed)))
    else:
        y = quantize(y, b, dither, int(seed))
    if low:
        y = zoh_resample(y, r, sr) if hd else resample_sinc(y, r, sr)
    return _fit(y, n)


@effect("smooth", "Moving-average smoothing (no ringing; repeated passes approach a Gaussian).",
        width=(1.0, "window length in ms (first null at 1000/width Hz)"), passes=(1, "1 box, 2 triangle, 4 ~Gaussian"))
def smooth(x, sr=DEFAULT_SR, width=1.0, passes=1):
    return moving_average(x, int(round(max(width, 0.0) * sr / 1000.0)), int(np.clip(passes, 1, 8)))


@effect("tape", "Tape machine: wow and flutter (band-limited variable-speed read), saturation, head bump, hiss.",
        wow=(0.15, "slow speed drift depth in percent (about 0.6 Hz)"), flutter=(0.08, "fast speed wobble depth in percent (about 7 Hz)"),
        drive=(1.5, "saturation input gain 0.1..10"), bump=(2.5, "head-bump boost in dB around 80 Hz"),
        hf=(12000, "high-frequency roll-off Hz"), hiss=(-58, "hiss level in dB RMS (-120 = none)"), seed=(0, "int"))
def tape(x, sr=DEFAULT_SR, wow=0.15, flutter=0.08, drive=1.5, bump=2.5, hf=12000, hiss=-58, seed=0):
    rng = np.random.default_rng(int(seed))
    n = x.shape[0]
    t = np.arange(n) / sr
    # UNSOURCED: wow 0.6 Hz, flutter 7.3 Hz plus a random component below 20 Hz; depths are typical consumer-deck figures
    w = float(np.clip(wow, 0, 5)) / 100.0
    fl = float(np.clip(flutter, 0, 5)) / 100.0
    drift = moving_average(rng.standard_normal(n), int(sr / 20), 2)
    drift /= max(np.max(np.abs(drift)), 1e-9)
    rate = 1 + w * np.sin(2 * np.pi * 0.6 * t + rng.uniform(0, 6.28)) + fl * (0.7 * np.sin(2 * np.pi * 7.3 * t + rng.uniform(0, 6.28)) + 0.3 * drift)
    y = varispeed(x, rate) if (w > 0 or fl > 0) else np.asarray(x, dtype=np.float64)
    d = float(np.clip(drive, 0.1, 10.0))
    y = oversampled(lambda v: np.tanh(d * v) / math.tanh(d), y, 2)
    if bump:
        y = F.peak(y, 80.0, sr, 1.0, float(np.clip(bump, -12, 12)))  # UNSOURCED: bump centre 80 Hz, Q 1
    if hf and hf < 0.49 * sr:
        y = F.lowpass(y, hf, sr)
    if hiss > -119:
        h = F.highshelf(gaussian(n, int(seed) + 7), 3000.0, sr, 6.0) * db_to_gain(hiss) * 0.6  # UNSOURCED hiss tilt
        y = y + (h[:, None] if y.ndim == 2 else h)
    peak = np.max(np.abs(y))
    return y * (1.4 / peak) if peak > 1.4 else y  # a big bump on a saturated signal may exceed full scale


# UNSOURCED: surface densities (kg/m^2), default leak ceilings (dB) and coincidence frequencies (Hz) are
# typical building-acoustics handbook values, not from the findings.
WALL_MATERIALS = {
    "window": dict(mass=10.0, leak=30.0, coincidence=3000.0),   # 4 mm glass
    "door": dict(mass=12.0, leak=22.0, coincidence=0.0),        # hollow interior door, gaps around it
    "solid_door": dict(mass=28.0, leak=30.0, coincidence=0.0),
    "drywall": dict(mass=22.0, leak=42.0, coincidence=2500.0),  # plasterboard stud partition
    "brick": dict(mass=200.0, leak=52.0, coincidence=0.0),      # 110 mm brick
    "concrete": dict(mass=350.0, leak=58.0, coincidence=0.0),   # 150 mm concrete
}


def wall_loss_db(f, mass, leak=40.0, coincidence=0.0):
    """Transmission loss of a single partition: mass law TL = 20 log10(f m) - 47 dB (field incidence,
    f in Hz, m in kg/m^2: +6 dB per octave and per doubling of mass), never below 0, capped at `leak`
    (gaps and flanking paths set the ceiling), with a dip of up to 8 dB one octave wide at the coincidence
    frequency."""
    # UNSOURCED: mass law and the 8 dB dip are textbook building acoustics, outside g11/g14. The findings
    # only establish that `muffle`'s 4 dB level loss is far too small (DSPG ch.22: loudness ~ power^(1/3)).
    f = np.asarray(f, dtype=np.float64)
    tl = np.clip(20 * np.log10(np.maximum(f * mass, 1e-9)) - 47.0, 0.0, leak)
    if coincidence > 0:
        tl = np.maximum(tl - 8.0 * np.exp(-0.5 * (np.log2(np.maximum(f, 1.0) / coincidence) / 0.5) ** 2), 0.0)
    return tl


@effect("wall", "Heard through a wall, door or window: mass-law transmission loss (real level loss, not just dullness).",
        material=("drywall", "window|door|solid_door|drywall|brick|concrete"),
        thickness=(1.0, "mass multiplier 0.25..8 (2 = twice as thick: 6 dB more loss)"),
        leak=(None, "loss ceiling in dB set by gaps and flanking (default per material: door 22 .. concrete 58)"),
        makeup=(0.0, "dB of gain after the loss (the true loss is 20-55 dB)"))
def wall(x, sr=DEFAULT_SR, material="drywall", thickness=1.0, leak=None, makeup=0.0):
    if material not in WALL_MATERIALS:
        raise ValueError(f"wall material must be one of {'|'.join(WALL_MATERIALS)}")
    m = WALL_MATERIALS[material]
    f = np.geomspace(20.0, 0.499 * sr, 48)
    tl = wall_loss_db(f, m["mass"] * float(np.clip(thickness, 0.25, 8.0)),
                      float(np.clip(m["leak"] if leak is None else leak, 0.0, 90.0)), m["coincidence"])
    h = fir_from_curve(f, -tl, sr, int(0.03 * sr) | 1, phase="min")
    return fir_filter(x, h, compensate=False) * db_to_gain(float(np.clip(makeup, -60, 80)))


@effect("dc_block", "Remove DC offset (one-pole/one-zero DC blocker).", cutoff=(20.0, "Hz"))
def dc_block(x, sr=DEFAULT_SR, cutoff=20.0):
    return dc_blocker(x, sr, cutoff)


# ======================================================================================================
# 9. Registered sfx
# ======================================================================================================

_BED_RATE = {"crackle": 25.0, "geiger": 12.0, "static_field": 4000.0}  # UNSOURCED default event rates


@sfx("noise_bed", "Background noise bed from Gaussian and Poisson models: hum, hiss, rumble, crackle, geiger, static_field.",
     kind=("hum", "hum (mains + 1/f floor) | hiss | rumble | crackle | geiger | static_field"),
     dur=(2.0, "s"), rate=(0.0, "events per second for crackle|geiger|static_field (0 = default 25|12|4000); level grows as sqrt(rate)"),
     level=(-12.0, "dB: single-event peak, or about the peak (3.5 sigma) of the Gaussian kinds"),
     freq=(50.0, "mains frequency for hum: 50 or 60 Hz"), bright=(0.4, "0..1 spectrum tilt / click sharpness"),
     seed=(0, "int"))
def noise_bed(sr=DEFAULT_SR, kind="hum", dur=2.0, rate=0.0, level=-12.0, freq=50.0, bright=0.4, seed=0):
    n = samples(float(np.clip(dur, 0.05, 600.0)), sr)
    seed = int(seed)
    bright = float(np.clip(bright, 0.0, 1.0))
    g = db_to_gain(float(np.clip(level, -80.0, 0.0)))
    sigma = g / 3.5  # Gaussian noise peaks at 3-4 sigma (DSPG ch.2 crest-factor note)
    rate = float(np.clip(rate, 0.0, 50000.0)) or _BED_RATE.get(kind, 0.0)
    top = min(0.45 * sr, 20000.0)
    if kind == "hum":
        y = g * mains_hum(n, sr, float(np.clip(freq, 20.0, 400.0)), seed=seed)
        y += 0.15 * sigma * F.lowpass(one_over_f(n, sr, 100.0, seed + 1), 300.0 + 3000.0 * bright, sr)
    elif kind == "hiss":
        y = sigma * F.lowpass(F.highpass(gaussian(n, seed), 200.0, sr), min(1500.0 + 9000.0 * bright, top), sr)
    elif kind == "rumble":
        y = F.lowpass(one_over_f(n, sr, 200.0, seed), 30.0 + 150.0 * bright, sr, order=2)
        y *= sigma / max(np.std(y), 1e-12)
    elif kind in ("crackle", "geiger", "static_field"):
        sg_ev = {"crackle": 0.8, "geiger": 0.15, "static_field": 0.5}[kind]  # UNSOURCED log-normal spreads
        ev = poisson_events(n, rate, sr, seed, sigma=sg_ev)
        if not np.any(ev):
            ev[n // 3] = 1.0  # a Poisson stream may be empty; a bed must not be silent
        if kind == "geiger":
            # each count = a short damped ring of the speaker (UNSOURCED: 2.2-3.4 kHz, 0.35 ms decay)
            m = samples(0.004, sr)
            tt = np.arange(m) / sr
            click = np.sin(2 * np.pi * (2200.0 + 1200.0 * bright) * tt) * np.exp(-tt / 0.00035)
        else:
            # one-sided exponential tick (UNSOURCED: 0.05-0.25 ms)
            m = samples(0.003, sr)
            click = np.exp(-np.arange(m) / (sr * (0.00025 - 0.0002 * bright)))
        click = click / np.max(np.abs(click))
        if kind == "static_field":
            # dense shot noise: variance = rate * E[a^2] * sum(h^2)/sr (counting statistics, DSPG ch.25),
            # so the bed is calibrated to `level` at the default rate and grows as sqrt(rate) from there
            click = F.bandpass(np.concatenate([click, np.zeros(samples(0.02, sr))]), 800.0 + 1700.0 * bright, sr, 0.5)
            ref = math.sqrt(_BED_RATE[kind] * math.exp(2 * sg_ev ** 2) * np.sum(click ** 2) / sr)
            swell = moving_average(lognormal(n, 0.6, seed + 2), int(sr * 0.05), 2)  # slow multiplicative swell
            y = (sigma / ref) * fft_convolve(ev, click)[:n] * swell / np.mean(swell)
        else:
            # `level` = the peak that 98 % of the events stay under (2 sigma of the log-normal)
            y = g * math.exp(-2 * sg_ev) * fft_convolve(ev, click)[:n]
            y = F.lowpass(y, min(2500.0 + 5500.0 * bright, top), sr)
    else:
        raise ValueError("noise_bed kind must be hum|hiss|rumble|crackle|geiger|static_field")
    y = dc_blocker(y - np.mean(y), sr, 10.0)
    k = min(samples(0.005, sr), n // 2)
    w = np.ones(n)
    w[:k] = 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / k)
    w[n - k:] = w[:k][::-1]
    y *= w
    y -= w * (y.sum() / w.sum())  # exact zero mean without lifting the faded ends
    peak = np.max(np.abs(y))
    if peak > 1.4:
        y *= 1.4 / peak
    return y
