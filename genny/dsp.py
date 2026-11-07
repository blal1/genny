"""Classical DSP toolbox: band-limited resampling, FIR/IIR design, FFT convolution and impulse-response
generators, companding and lo-fi codecs, dither, oversampled waveshaping, noise models and metering.

Sources (findings files in out/research):
  g11 = S. W. Smith, "The Scientist and Engineer's Guide to DSP" (chapter numbers cited as DSPG nn).
  g14 = J. O. Smith, "Introduction to Digital Filters" (FILT) and "Mathematics of the DFT" (MDFT).
Constants not in those sources (hardware lore, voicing choices) are justified in research/SOURCES.md.

Everything here is offline float64. Registered names: fx `convolve fir_eq steep_lowpass steep_highpass
steep_bandpass overdrive wavefold mulaw codec dither lofi smooth tape wall dc_block`, sfx `noise_bed complex_osc`.
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

from . import filters as F
from .core import DEFAULT_SR, db_to_gain, read_wav, samples, to_mono
from .fx import _per_channel, compressor as _compressor, effect
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


def butter2_r(R: float, N: int) -> np.ndarray:
    """Dampings of the N 2-pole sections of the Butterworth filter of the 2nd kind of order N built on the 2-pole
    filter of damping R < 1: unit-circle poles at the 1st-kind angles pi/2 + pi(2k+1)/(2N) turned by
    alpha/N, cos alpha = R, so R_k = -cos(pi/2 + pi(2k+1)/(2N) + alpha/N). |H_N(jw)| = |H_2(jw^N)|: the
    resonance peak of the 2-pole filter is kept, the slopes are N times steeper."""
    N = max(1, int(N))
    a = math.acos(float(np.clip(R, 0.0, 1.0)))
    k = np.arange(N)
    return -np.cos(math.pi / 2 + math.pi * (2 * k + 1) / (2 * N) + a / N)


def butter2(x: np.ndarray, cutoff, sr: int = DEFAULT_SR, R: float = 0.3, N: int = 2, kind: str = "lowpass") -> np.ndarray:
    """Butterworth filter of the 2nd kind: N TPT SVF sections in series at the same cutoff, dampings ``butter2_r``."""
    y = np.asarray(x, float)
    for r in butter2_r(R, N):
        y = F.svf(y, kind, cutoff, sr, q=0.5 / max(float(r), 1e-3))
    return y


def iir_sos(kind: str, fc, sr=DEFAULT_SR, order: int = 4, family: str = "butter", ripple: float = 0.5, atten: float = 90.0) -> np.ndarray:
    """Steep recursive low/high-pass as second-order sections. family butter|cheby (Smith's program;
    `ripple` = percent pass-band ripple, cheby only; at fc such a filter is at its last ripple edge, not at -3 dB: order 8 with 29 % is -0.035 dB there,
    Zavalishin c27) | bessel (scipy, magnitude-normalised: -3 dB at fc, near-linear phase, no overshoot; not in Smith's program) |
    cheby2 (flat pass band, `atten` dB stop-band ripple, the -3 dB point at fc: Wn = fc cosh(acosh(1/eps)/N), Zavalishin c27) |
    ellip (ripple in dB here, `atten` in the stop band; ChucK's Elliptic: 0.2 dB and 90 dB). order = number of poles, even, 2..20."""
    order = int(np.clip(2 * round(order / 2), 2, 20))
    f = float(np.clip(fc, 10.0, 0.49 * sr)) / sr
    if family in ("butter", "butterworth"):
        return smith_sos(f, order, kind, 0.0)
    if family in ("cheby", "chebyshev", "cheby1"):
        return smith_sos(f, order, kind, ripple)
    if family == "bessel":
        return sg.bessel(order, 2 * f, btype=kind, norm="mag", output="sos")
    if family == "cheby2":
        c = math.cosh(math.acosh(math.sqrt(10.0 ** (float(atten) / 10.0) - 1.0)) / order)
        t = math.tan(math.pi * f)
        wn = math.atan(t * c if kind == "lowpass" else t / c) / math.pi
        return sg.cheby2(order, float(atten), 2 * min(wn, 0.499), btype=kind, output="sos")
    if family in ("ellip", "elliptic"):
        return sg.ellip(order, float(ripple), float(atten), 2 * f, btype=kind, output="sos")
    raise ValueError("family must be butter|cheby|bessel|cheby2|ellip")


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
        refl = 0.8 ** (1 + i // 3)  # wall reflection coefficient 0.8 per reflection order
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


SPRING_IR_COILS = ((1.0e-4, 2.5e-3, 112), (1.0e-4, 2.5e-3, 126))     # wire r, coil R (m), turns: the two-spring tank
SPRING_IR_EDGE = 0.98                                                   # fraction of fC where the echo time is read last


def ir_spring(sr=DEFAULT_SR, decay=2.0, size=0.5, bright=0.5, seed=0) -> np.ndarray:
    """Spring-like IR: a dispersive delay loop H = gA/(1 - gA) where A is an allpass whose group delay
    rises with frequency up to a transition frequency (each echo is a rising chirp, and each round trip
    disperses more), built from the chirp-system idea of DSPG ch.11. The loop gain per frequency is
    g = 10^(-3 tau(f)/decay), so every frequency has the requested RT60. tau(f) follows the round-trip time 2 L / v_g of
    a real helical spring (`fdstring.spring_echo_time`), its fC mapped onto ``fcut``."""
    from .fdstring import spring_dispersion, spring_echo_time
    rng = np.random.default_rng(seed)
    # transition frequency fC 3.5-5 kHz: "an fC of above 5 kHz is rarely seen in real spring reverberators"
    # (Parker 2011, EURASIP J. Adv. Signal Process. 646134, §3.2 p.3); lower bound: the measured springs span
    # fC 3526-4450 Hz and fC 2000 Hz "would be unusual in a real spring reverberation unit" (Valimaki, Parker & Abel
    # 2010, JAES 58(7/8), Table 1 p.558 and Sec. 3.1 p.557)
    fcut = min((3500.0 + 1500.0 * bright), 0.45 * sr)
    N = 1 << int(math.ceil(math.log2((1.6 * decay + 0.3) * sr)))
    f = np.fft.rfftfreq(N, 1.0 / sr)
    out = np.zeros(N)
    for j in range(2):  # two springs in parallel (Olson X-82 tank, Parker & Bilbao 2009 DAFx-09 App. A)
        t0 = (0.030 + 0.035 * size) * (1 + 0.23 * j) * (1 + 0.02 * rng.standard_normal())
        r, R, turns = SPRING_IR_COILS[j]
        fcs = spring_dispersion(r, R)[3]
        shape = spring_echo_time(np.minimum(f / fcut, SPRING_IR_EDGE) * fcs, r, R, turns)
        tau = t0 * shape / np.min(shape)
        phase = 2 * np.pi * np.cumsum(tau) * (f[1] - f[0])
        g = 10 ** (-3 * tau / max(decay, 0.05)) / np.sqrt(1 + (f / fcut) ** 12)
        A = g * np.exp(-1j * phase)
        out += np.fft.irfft(A / (1 - A), N) * (1.0 if j == 0 else -0.8)
    return out[: samples(1.4 * decay + 0.2, sr)]


def ir_plate(sr=DEFAULT_SR, decay=1.5, size=0.5, bright=0.6, seed=0) -> np.ndarray:
    """Plate-like IR: dense Gaussian tail from t = 0 (no discrete early reflections) passed through an
    anti-chirp allpass, so high frequencies arrive first as bending waves do on a plate (DSPG ch.11)."""
    tail = ir_noise(sr, decay, decay * (0.35 + 0.6 * bright), seed=seed)
    sweep = (0.008 + 0.03 * size) * sr  # 8-38 ms dispersion spread
    return fft_convolve(tail, chirp_allpass(int(2 * sweep) + 64, sweep, 2.0))


def ir_room(sr=DEFAULT_SR, decay=0.5, size=0.4, bright=0.5, seed=0, hall=False) -> np.ndarray:
    """Room / hall IR = sparse early reflections + per-band decaying Gaussian tail that fades in over the
    early window. First-reflection gap 4-20 ms (room) or 20-100 ms (hall) with size (DSPG ch.1/22)."""
    gap = (0.020 + 0.080 * size) if hall else (0.004 + 0.016 * size)
    early = ir_early(sr, gap, 10 if hall else 7, 4.0, seed)
    tail = ir_noise(sr, decay * 1.1, decay * (0.25 + 0.6 * bright), seed=seed + 1)  # HF decay ratio
    t = np.arange(tail.shape[0]) / sr
    tail *= 1 - np.exp(-t / (1.5 * gap))
    tail *= np.sqrt(np.sum(early ** 2) / max(np.sum(tail ** 2), 1e-12))  # early and late energy equal
    out = tail.copy()
    out[: early.shape[0]] += early
    return out


# colour curves, (Hz, dB). hand-drawn typical responses, not measurements.
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
        tk = decay / math.sqrt(k)  # higher modes decay faster as 1/sqrt(mode number)
        h += k ** (-(1.6 - bright)) * np.exp(-6.9078 * t / tk) * np.sin(2 * np.pi * fk * t)  # tilt
        k += 1
    h[0] += 0.3  # some direct sound
    return h


CABIN_F0, CABIN_GAIN_DB = 70.0, 26.0


def cabin_gain(x, sr=DEFAULT_SR, f0=CABIN_F0, gain_db=CABIN_GAIN_DB):
    """Pressure-field gain of a closed cabin below its first mode f0: +12 dB per octave, levelling off at gain_db
    where leakage takes over. Second-order shelf (s^2 + sqrt2 w0 s + w0^2) / (s^2 + sqrt2 w1 s + w1^2), w1 below w0."""
    w0 = 2 * math.pi * f0
    w1 = w0 * 10.0 ** (-gain_db / 40.0)
    b, a = sg.bilinear([1.0, math.sqrt(2) * w0, w0 * w0], [1.0, math.sqrt(2) * w1, w1 * w1], sr)
    return sg.lfilter(b, a, x)


IR_KINDS = ("room", "hall", "plate", "spring", "cabinet", "tube", "car", "phone-speaker")
_IR_DECAY = {"room": 0.3, "hall": 2.1, "plate": 1.6, "spring": 2.0, "tube": 0.15, "car": 0.09}
# room 0.3 s ("living rooms", Kuttruff 2009 §3.6 p.98); hall 2.1 s, top of the 1.6-2.1 s optimum for occupied concert
# halls (Kuttruff 2009 §7.5 p.231-232). plate, spring, tube, car defaults
_SPACE_KINDS = ("room", "hall", "plate", "spring")


def make_ir(kind="room", sr=DEFAULT_SR, decay=0.0, size=0.5, bright=0.5, seed=0) -> np.ndarray:
    """Generated impulse response by name (see IR_KINDS). decay = RT60 seconds (0 = the kind's default).
    Space kinds are normalised to unit energy (a noise-like input keeps its RMS); cabinet/phone-speaker
    keep the 0 dB reference of their curve; tube has 0 dB at its strongest resonance, car 0 dB in the 1 kHz octave (its cabin gain rises below)."""
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
        h = cabin_gain(h, sr)
        n = 1 << int(math.ceil(math.log2(4 * h.shape[0])))
        f = np.fft.rfftfreq(n, 1.0 / sr)
        band = np.abs(np.fft.rfft(h, n))[(f >= 1000 / math.sqrt(2)) & (f <= 1000 * math.sqrt(2))]
        return h / math.sqrt(np.mean(band ** 2))        # 0 dB in the 1 kHz octave: the cabin gain stays a gain
    else:
        raise ValueError(f"unknown ir kind {kind!r}; one of {', '.join(IR_KINDS)}")
    if kind == "tube":  # resonant colour: 0 dB at the strongest resonance, so tones are never boosted
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


@numba.njit(cache=True)
def _fir_shaped_quantize(v, d, lim, c):
    """Error feedback: noise transfer function 1 - sum c_k z^-k."""
    out = np.empty(v.shape[0])
    N = c.shape[0]
    h = np.zeros(N)
    for i in range(v.shape[0]):
        u = v[i]
        for k in range(N):
            u -= c[k] * h[k]
        q = min(max(np.round(u + d[i]), -lim), lim)
        for k in range(N - 1, 0, -1):
            h[k] = h[k - 1]
        h[0] = q - u
        out[i] = q
    return out


# SoX dither.c, via lot B14: fwe44, declared at 46 kHz (SoX tolerance +-5 %, so 44.1 and 48 kHz); lip44 at 44.1 kHz. The sign convention of SoX's CONVOLVE was not read:
# this sign is the one that puts the minimum of the noise near 3-4 kHz, where the ear is most sensitive (tests/test_effets.py); the other one does not.
_SHAPERS = {"fweighted": np.array([2.412, -3.370, 3.937, -4.174, 3.353, -2.205, 1.281, -0.569, 0.0847]),
            "lipshitz": np.array([2.033, -2.165, 1.959, -1.590, 0.6149])}


def quantize(x, bits=8, dither="tpdf", seed=0):
    """Round to `bits` (LSB = 2/2^bits, levels +-2^(bits-1)) with dither added before rounding.
    dither: none | tpdf (sum of two uniforms, +-1 LSB; the standard, outside Smith's text) | gauss (sd 2/3
    LSB, DSPG ch.3) | shaped (TPDF + first-order error feedback: quantisation noise shaped by 1 - z^-1,
    i.e. pushed to high frequencies; the concept of DSPG ch.3 delta-sigma / MDFT noise-shaped dither) |
    fweighted, lipshitz (TPDF + a psychoacoustic error filter, the noise moved away from 3 kHz to 10-22 kHz: Katz, Mastering Audio ch. 4)."""
    x = np.asarray(x, dtype=np.float64)
    S = float(2 ** (int(np.clip(bits, 1, 24)) - 1))
    rng = np.random.default_rng(seed)
    if dither in ("none", "", None, False):
        d = np.zeros(x.shape)
    elif dither == "gauss":
        d = rng.standard_normal(x.shape) * (2.0 / 3.0)
    elif dither in ("tpdf", "shaped", True) or dither in _SHAPERS:
        d = rng.random(x.shape) + rng.random(x.shape) - 1.0
    else:
        raise ValueError("dither must be none|tpdf|gauss|shaped|fweighted|lipshitz")
    if dither in _SHAPERS:
        c = _SHAPERS[dither]
        one = lambda v, e: _fir_shaped_quantize(np.ascontiguousarray(v) * S, np.ascontiguousarray(e), S, c) / S       # noqa: E731
        return one(x, d) if x.ndim == 1 else np.stack([one(x[:, k], d[:, k]) for k in range(x.shape[1])], axis=1)
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
    # syllabic filter 5 ms +-1, reconstruction integrator 1 ms +-0.25 (RCC IRIG 106-99 Telemetry Standards,
    # App. F §5.6, §5.7). step_max 0.15 (IRIG §5.5 gives a 16:1 compression ratio, which with Smith's
    # step_min = 2/2000 would cap the slew at 0.016 * clock; not adopted)
    bits = np.zeros(x.shape[0], dtype=np.uint8)
    rec = _cvsd_core(np.ascontiguousarray(x, dtype=np.float64), bits, False, *_cvsd_params(clock, step_min, step_max, syllabic, leak), run)
    return bits, rec


def cvsd_decode(bits, clock=30000, step_min=0.001, step_max=0.15, syllabic=0.005, leak=0.001, run=4):
    return _cvsd_core(np.zeros(1), np.ascontiguousarray(bits, dtype=np.uint8), True, *_cvsd_params(clock, step_min, step_max, syllabic, leak), run)


# IMA/DVI ADPCM tables: indexTable and the 89-entry stepsizeTable, checked value by value against IMA Digital
# Audio Technical Working Group, "Recommended Practices for Enhancing Digital Audio Compatibility in Multimedia
# Systems", rev. 3.00, 21 Oct 1992, p.28.
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
    q = 1.0 + (np.arange(block) / block) ** power * qmax  # power-law table (Smith's are 8x8 JPEG tables)
    step = q / 2.0 ** (bits - 1)
    return idct(np.round(X / step) * step, norm="ortho", axis=1).reshape(-1)[:n]


_GSM_H = np.array([-134, -374, 0, 2054, 5741, 8192, 5741, 2054, 0, -374, -134]) / 8192.0     # GSM 06.10 Table 3.4
# GSM 06.10 Table 3.5, the upper bounds of xmax: steps of 32 up to 511, then 8 steps per octave up to 32767 (scale of 16-bit samples)
_GSM_XMAX = np.concatenate([np.arange(31, 512, 32)] + [np.arange(hi - 7 * st, hi + 1, st) for hi, st in ((1023, 64), (2047, 128), (4095, 256), (8191, 512), (16383, 1024), (32767, 2048))]) / 32768.0


def gsm_like(x, pbits=3, order=8):
    """GSM-full-rate-like coder for a signal ALREADY at 8 kHz: 20 ms frames, LPC (autocorrelation method,
    order 8), the prediction residual goes through the weighting filter and is kept on one of four grids of 13 pulses
    per 5 ms sub-frame (regular pulse excitation), each pulse quantised to `pbits` bits relative to the block
    maximum of Table 3.5, then re-synthesised through 1/A(z). The reflection coefficients go through log-area ratios,
    quantised and interpolated over four zones of the frame like the real coder. Not bit-exact GSM 06.10: the
    long-term predictor is omitted. Speech frame model: DSPG ch.22 / 27 (LPC = predict from past samples, store the error)."""
    # 160-sample frames, 4 sub-frames of 40, LPC order 8 (LAR 1-8), RPE decimation 3, 3-bit pulses: ETSI ETS 300 961
    # (GSM 06.10 v5.2.1, 2000) §1.5 p.11-12 and Table 1.1. Weighting H, grids: §3.1.18-19, Table 3.4. xmax: Table 3.5.
    from .physical.analysis import _lattice_inverse, _lattice_synth, _levinson
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    levels = 2 ** int(pbits)
    Kf = []
    for s in range(0, n, 160):
        fr = x[s: s + 160]
        w = fr * np.hamming(fr.shape[0]) if fr.shape[0] > 1 else fr
        r = np.array([np.dot(w[: w.shape[0] - k], w[k:]) for k in range(order + 1)]) if fr.shape[0] > order else np.zeros(order + 1)
        if r[0] < 1e-10:
            Kf.append(np.zeros(order))
        else:
            r[0] *= 1.0001  # white-noise correction keeps the normal equations well conditioned
            Kf.append(gsm_lar_quantize(_levinson(r, order)[1]))
    K = np.zeros((n, order))                     # reflection coefficients per sample: the previous frame's blended in
    for f, s in enumerate(range(0, n, 160)):
        prev = Kf[f - 1] if f else Kf[0]
        for lo, hi, wc in _GSM_INTERP:
            K[s + lo: min(s + hi, n)] = lar_to_k((1.0 - wc) * k_to_lar(prev) + wc * k_to_lar(Kf[f]))
    e_all = _lattice_inverse(x, K, 1)
    eq_all = np.zeros(n)
    for s in range(0, n, 160):
        e = e_all[s: s + 160]
        eq = eq_all[s: s + 160]
        for b in range(0, e.shape[0], 40):
            sub = np.convolve(e[b: b + 40], _GSM_H, "same")[:40]
            m = int(np.argmax([np.sum(sub[k: k + 39: 3] ** 2) for k in range(4)]))
            pulses = sub[m: m + 39: 3]
            xmax = np.max(np.abs(pulses)) if pulses.size else 0.0
            if xmax > 1e-9:
                xmax = _GSM_XMAX[min(int(np.searchsorted(_GSM_XMAX, xmax)), 63)]
                p = np.clip(np.floor(pulses / xmax * levels / 2), -levels / 2, levels / 2 - 1) + 0.5
                eq[b + m: b + m + pulses.size * 3: 3] = p * xmax * 2 / levels      # H(0) = 2.779 makes up for keeping one sample in three
    return np.clip(_lattice_synth(eq_all, K, 1), -1.0, 1.0)


_GSM_INTERP = ((0, 13, 0.25), (13, 27, 0.5), (27, 40, 0.75), (40, 160, 1.0))   # samples of the frame, weight of the new frame
_GSM_LAR_A = np.array([20.0, 20.0, 20.0, 20.0, 13.637, 15.0, 8.334, 8.824])
_GSM_LAR_B = np.array([0.0, 0.0, 4.0, -5.0, 0.184, -3.5, -0.666, -2.235])
_GSM_LAR_MIC = np.array([-32, -32, -16, -16, -8, -8, -4, -4])
_GSM_LAR_MAC = np.array([31, 31, 15, 15, 7, 7, 3, 3])


def k_to_lar(k):
    """Piecewise-linear log-area ratio: |r| < 0.675 -> r; < 0.95 -> 2|r| - 0.675; else 8|r| - 6.375 (sign kept)."""
    k = np.asarray(k, dtype=np.float64)
    a = np.abs(k)
    return np.sign(k) * np.where(a < 0.675, a, np.where(a < 0.95, 2 * a - 0.675, 8 * a - 6.375))


def lar_to_k(lar):
    """Inverse of k_to_lar: |L| < 0.675 -> L; < 1.225 -> L/2 + 0.3375; else L/8 + 0.796875."""
    lar = np.asarray(lar, dtype=np.float64)
    a = np.abs(lar)
    return np.sign(lar) * np.minimum(np.where(a < 0.675, a, np.where(a < 1.225, 0.5 * a + 0.3375, 0.125 * a + 0.796875)), 0.999)


def gsm_lar_quantize(k):
    """Reflection coefficients 1-8 through their LARs coded as round(A LAR + B) within [MIC, MAC], decoded back."""
    k = np.asarray(k, dtype=np.float64)
    m = min(len(k), 8)
    lar = k_to_lar(k[:m])
    z = _GSM_LAR_A[:m] * lar + _GSM_LAR_B[:m]
    c = np.clip(np.sign(z) * np.floor(np.abs(z) + 0.5), _GSM_LAR_MIC[:m], _GSM_LAR_MAC[:m])
    out = k.copy()
    out[:m] = lar_to_k((c - _GSM_LAR_B[:m]) / _GSM_LAR_A[:m])
    return out


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
    # harmonic amplitudes 1, 0.5, 0.3, 0.2 (the book only says harmonics are present)
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
        predelay=(0.0, "seconds before the wet signal starts"), seed=(0, "int: which random room"),
        loop=(False, "fold the tail back onto the start: same length as the input, loops seamlessly"))
def convolve(x, sr=DEFAULT_SR, ir="", kind="room", decay=0.0, size=0.5, bright=0.5, mix=None, predelay=0.0, seed=0, loop=False):
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
    if loop:
        n = np.asarray(x).shape[0]
        pad = (-wet.shape[0]) % n
        wet = np.concatenate([wet, np.zeros((pad,) + wet.shape[1:])], axis=0)
        wet = wet.reshape((-1, n) + wet.shape[1:]).sum(axis=0)
    dry = _fit(np.asarray(x, dtype=np.float64), wet.shape[0])
    return dry * (1 - mix) + wet * mix


def factor_body(ir, sr=DEFAULT_SR, modes=3, r=0.9):
    """Body factoring: the ``modes`` strongest resonances of a body IR (pole radius R_m = exp(-pi B / sr) from their
    -3 dB width B) are replaced in the IR by short ones of radius ``r``, so the residual is short.
    Returns (residual, [(f, R_m), ...]); the body is the residual followed, for each mode, by
    (1 - 2 r cos w z^-1 + r^2 z^-2) / (1 - 2 R_m cos w z^-1 + R_m^2 z^-2)."""
    from scipy.signal import find_peaks, lfilter
    h = np.asarray(ir, dtype=np.float64)
    N = 1 << int(math.ceil(math.log2(max(h.shape[0], 2) * 4)))
    H = np.abs(np.fft.rfft(h, N))
    f = np.fft.rfftfreq(N, 1.0 / sr)
    pk, _ = find_peaks(H)
    out = []
    for k in sorted(pk[np.argsort(H[pk])[::-1][:int(modes)]]):
        lo, hi = k, k
        while lo > 0 and H[lo] > H[k] / math.sqrt(2.0):
            lo -= 1
        while hi < H.shape[0] - 1 and H[hi] > H[k] / math.sqrt(2.0):
            hi += 1
        Rm = math.exp(-math.pi * max(f[hi] - f[lo], f[1]) / sr)
        c = math.cos(2.0 * math.pi * f[k] / sr)
        h = lfilter([1.0, -2.0 * Rm * c, Rm * Rm], [1.0, -2.0 * r * c, r * r], h)
        out.append((float(f[k]), Rm))
    return h, out


@effect("fir_eq", "Draw-your-own EQ curve as one FIR filter (frequency sampling).",
        points=("100:0,1000:0,8000:-6", "curve as 'hz:db,hz:db,...' (flat outside the end points)"),
        phase=("linear", "linear (no phase smear, delay compensated) | min (no pre-ringing)"),
        taps=(0, "kernel length; 0 = 93 ms (resolves ~65 Hz)"))
def fir_eq(x, sr=DEFAULT_SR, points="100:0,1000:0,8000:-6", phase="linear", taps=0):
    f, g = _points(points)
    h = fir_from_curve(f, g, sr, int(taps) if taps else None, phase)
    return fir_filter(x, h, compensate=(phase == "linear"))


_STEEP = dict(order=(8, "number of poles, even, 2..20 (6 dB/octave each)"), type=("butter", "butter|cheby|bessel|cheby2|ellip"),
              ripple=(0.5, "cheby pass-band ripple in percent, 0..29; ellip: in dB"))


@effect("steep_lowpass", "Steep low-pass (true Butterworth / Chebyshev / Bessel / inverse Chebyshev / elliptic cascade).", cutoff=(2000, "Hz (-3 dB; cheby and ellip: edge of the ripple)"), **_STEEP)
def steep_lowpass(x, sr=DEFAULT_SR, cutoff=2000, order=8, type="butter", ripple=0.5):
    return steep_filter(x, "lowpass", cutoff, sr, order, type, ripple)


@effect("steep_highpass", "Steep high-pass (true Butterworth / Chebyshev / Bessel / inverse Chebyshev / elliptic cascade).", cutoff=(200, "Hz (-3 dB; cheby and ellip: edge of the ripple)"), **_STEEP)
def steep_highpass(x, sr=DEFAULT_SR, cutoff=200, order=8, type="butter", ripple=0.5):
    return steep_filter(x, "highpass", cutoff, sr, order, type, ripple)


@effect("steep_bandpass", "Steep band-pass: high-pass at `low` then low-pass at `high`, each of `order` poles.",
        low=(300, "Hz"), high=(3400, "Hz"), **_STEEP)
def steep_bandpass(x, sr=DEFAULT_SR, low=300, high=3400, order=8, type="butter", ripple=0.5):
    return steep_filter(x, "bandpass", low, sr, order, type, ripple, fc2=high)


def _tri_fold(v):
    return (2 / np.pi) * np.arcsin(np.sin(0.5 * np.pi * v))


# the curve shapes are voicing choices. The sourced rule (DSPG ch.11) is only: odd-symmetric
# curve -> odd harmonics; any asymmetry (bias, tube, diode) -> even harmonics too.
_SHAPES = {
    "tanh": np.tanh,
    "tube": lambda v: np.where(v >= 0, np.tanh(v), np.tanh(1.5 * v) / 1.5),
    "fuzz": lambda v: np.sign(v) * -np.expm1(-3.0 * np.abs(v)),
    "fold": lambda v: np.sin(0.5 * np.pi * v),
    "diode": lambda v: np.logaddexp(0.0, 4.0 * v) / 4.0,
    "hard": lambda v: np.clip(v, -1.0, 1.0),                  # Parker, Zavalishin & Le Bivic, DAFx-16, eq. 24
    "algebraic": lambda v: v / np.sqrt(1.0 + v * v),          # Zavalishin 6.2a, Vicanek's AADistortion eq. 7
    "hyper": lambda v: v / (1.0 + np.abs(v)),                 # Zavalishin 6.2c; Soundpipe saturator
    "asinh": np.arcsinh,                                      # Zavalishin 6.4-6.5: unbounded, softest
    "asym": lambda v: 1.25 * v + 0.75 * (np.sqrt(v * v + 0.01) - 0.1),
    "tube_bendiksen": lambda v: _tube_bendiksen(v),
}


def _tube_bendiksen(v, q=-0.2, dist=8.0):
    """Asymmetric tube curve (x - Q)/(1 - e^(-dist (x - Q))) + Q/(1 - e^(dist Q)): linear for x >> Q, saturating to
    Q/(1 - e^(dist Q)) on the negative side, f(0) = 0; the 0/0 at x = Q takes its limit 1/dist."""
    u = np.asarray(v, dtype=np.float64) - q
    small = np.abs(u) < 1e-9
    us = np.where(small, 1.0, u)
    return np.where(small, 1.0 / dist, us / -np.expm1(-dist * us)) + q / -np.expm1(dist * q)


def noise_band_for_rate(fe):
    """Bandwidth (Hz) of a narrow band of noise whose envelope fluctuates at ``fe`` per second: fe = 0.64 df."""
    return float(fe) / 0.64


def soft_cf(z):
    """Soft clip by the continued fraction z/(1 + z^2/(3 + 4z^2/(5 + 9z^2/(7 + 16z^2)))) (the expansion of arctan, cut
    after four levels)."""
    z = np.asarray(z, dtype=np.float64)
    z2 = z * z
    return z / (1.0 + z2 / (3.0 + 4.0 * z2 / (5.0 + 9.0 * z2 / (7.0 + 16.0 * z2))))


def wsb(n, f0, sr=DEFAULT_SR, mix=(0.7, 0.15, 0.05), gain=0.2, drive=0.308319):
    """Chebyshev waveshaping bank: a cosine through three stages of T2(x) = 2x^2 - 1 (octaves 2, 4, 8), mixed
    0.7 / 0.15 / 0.05, times 0.2, then the continued-fraction soft clip with its input scaled by 0.308319."""
    c = np.cos(2 * np.pi * float(f0) * np.arange(int(n)) / sr)
    t1 = 2 * c * c - 1
    t2 = 2 * t1 * t1 - 1
    t3 = 2 * t2 * t2 - 1
    return soft_cf(drive * gain * (mix[0] * t1 + mix[1] * t2 + mix[2] * t3))


def _F2_hard(v):
    a = np.abs(v)
    return np.where(a <= 1.0, v ** 3 / 6.0, np.sign(v) * (a * a / 2.0 - a / 2.0 + 1.0 / 6.0))


def _F2_tanh(v):
    """Second antiderivative of tanh, odd about 0, derived from F1 = log cosh = u + log(1 + e^-2u) - log 2: u^2 / 2 + Li2(-e^-2u) / 2 - u log 2.
    The form noted from Parker et al. eq. 22, 1/2 (x (x + 2 log(e^-2x + 1)) - Li2(-e^-2x)), differentiates to x tanh x, not to log cosh: checked by finite differences."""
    from scipy.special import spence
    a = np.abs(v)
    G = lambda u: 0.5 * (u * u + spence(1.0 + np.exp(-2.0 * u))) - u * math.log(2.0)      # noqa: E731
    return np.sign(v) * (G(a) - G(np.zeros_like(a)))


# first and second antiderivatives: shape -> (F1, F2), F2 None when not in closed form here. The published forms of the hard clipper (lot B71: F1 = x^3/3 and (x^2/2 - 1/6) sgn x)
# do not differentiate to x^2/2 and |x| - 1/2: the second antiderivative used here is derived, and checked by finite differences in tests/test_effets.py.
_ANTI = {
    "tanh": (lambda v: np.logaddexp(v, -v) - math.log(2.0), _F2_tanh),
    "hard": (lambda v: np.where(np.abs(v) <= 1.0, v * v / 2.0, np.abs(v) - 0.5), _F2_hard),
    "fold": (lambda v: -(2.0 / np.pi) * np.cos(0.5 * np.pi * v), lambda v: -(2.0 / np.pi) ** 2 * np.sin(0.5 * np.pi * v)),    # lot B49
    "fuzz": (lambda v: np.abs(v) + np.expm1(-3.0 * np.abs(v)) / 3.0, lambda v: np.sign(v) * (v * v / 2.0 - np.expm1(-3.0 * np.abs(v)) / 9.0 - np.abs(v) / 3.0)),    # lot B49
    "algebraic": (lambda v: np.sqrt(1.0 + v * v), lambda v: 0.5 * (v * np.sqrt(1.0 + v * v) + np.arcsinh(v))),
    "hyper": (lambda v: np.abs(v) - np.log1p(np.abs(v)), lambda v: np.sign(v) * (v * v / 2.0 - (1.0 + np.abs(v)) * np.log1p(np.abs(v)) + np.abs(v))),
    "asinh": (lambda v: v * np.arcsinh(v) - np.sqrt(1.0 + v * v), None),
    "diode": (None, None),
    "tube": (None, None),
}


def adaa1(x, f, F1):
    """First-order antiderivative anti-aliasing (Parker, Zavalishin & Le Bivic, DAFx-16): (F1(x[n]) - F1(x[n-1])) / (x[n] - x[n-1]), half a sample of delay."""
    x = np.asarray(x, float)
    xp = np.concatenate((np.zeros((1,) + x.shape[1:]), x[:-1]))
    d = x - xp
    ok = np.abs(d) > 1e-6
    y = np.asarray(f(0.5 * (x + xp)), float).copy()
    y[ok] = (F1(x[ok]) - F1(xp[ok])) / d[ok]
    return y


def adaa2(x, f, F1, F2):
    """Second order, one sample of delay: y = 2 / (x[n] - x[n-2]) (D[n] - D[n-1]), D[n] = (F2(x[n]) - F2(x[n-1])) / (x[n] - x[n-1]).
    When x[n] ~ x[n-2]: xb = (x[n] + x[n-2]) / 2, dl = xb - x[n-1], y = 2 / dl (F1(xb) + (F2(x[n-1]) - F2(xb)) / dl), and f((xb + x[n-1]) / 2) when dl ~ 0 too."""
    x = np.asarray(x, float)
    z = np.zeros((1,) + x.shape[1:])
    xp = np.concatenate((z, x[:-1]))
    d = x - xp
    ok = np.abs(d) > 1e-4
    D = np.asarray(F1(0.5 * (x + xp)), float).copy()
    D[ok] = (F2(x[ok]) - F2(xp[ok])) / d[ok]
    Dp = np.concatenate((D[:1], D[:-1]))
    xpp = np.concatenate((z, xp[:-1]))
    w = x - xpp
    good = np.abs(w) > 1e-4
    xb = 0.5 * (x + xpp)
    dl = xb - xp
    y = np.asarray(f(0.5 * (xb + xp)), float).copy()
    mid = ~good & (np.abs(dl) > 1e-4)
    y[mid] = 2.0 / dl[mid] * (F1(xb[mid]) + (F2(xp[mid]) - F2(xb[mid])) / dl[mid])
    y[good] = 2.0 * (D[good] - Dp[good]) / w[good]
    return y


def _F3_hard(v):
    a = np.abs(v)
    return np.where(a <= 1.0, v ** 4 / 24.0, a ** 3 / 6.0 - a * a / 4.0 + a / 6.0 - 1.0 / 24.0)


_ANTI3 = {
    "hard": _F3_hard,
    "fold": lambda v: (2.0 / np.pi) ** 3 * np.cos(0.5 * np.pi * v),
}


def adaa_n(x, f, FN, order: int = 3, eps: float = 1e-3):
    """Antiderivative anti-aliasing of any order N: N! times the divided difference of the N-th antiderivative FN
    over x[n], ..., x[n-N] (the Lagrange form), (N/2) samples of delay. Where two of those points are closer than
    ``eps`` the divided difference is ill conditioned and f(mean of the points) is used instead."""
    x = np.asarray(x, float)
    N = int(order)
    z = np.zeros((N,) + x.shape[1:])
    xs = np.concatenate((z, x))
    pts = [xs[N - j: len(xs) - j] for j in range(N + 1)]          # pts[j] = x[n - j]
    dd = [FN(p) for p in pts]
    bad = np.zeros(x.shape, dtype=bool)
    for lvl in range(1, N + 1):
        nxt = []
        for j in range(N + 1 - lvl):
            den = pts[j] - pts[j + lvl]
            small = np.abs(den) < eps
            bad |= small
            nxt.append((dd[j] - dd[j + 1]) / np.where(small, 1.0, den))
        dd = nxt
    y = math.factorial(N) * dd[0]
    y[bad] = np.asarray(f(sum(pts) / (N + 1)), float)[bad]
    return y


def vicanek_shaper(x):
    """Vicanek, Note on Alias Suppression in Digital Distortion: algebraic shaper f = x / sqrt(1 + x^2) through the delayed kernel (eq. 10), then the IIR that restores the top octave:
    y = b v - a y[-1], a = 1 / (3 + sqrt 8) = 0.171573, b = 1 + a (-6 dB at Nyquist otherwise)."""
    x = np.asarray(x, float)
    z = np.zeros((1,) + x.shape[1:])
    x1 = np.concatenate((z, x[:-1]))
    x2 = np.concatenate((z, x1[:-1]))
    Fh = np.sqrt(1.0 + (0.5 * (x + x1)) ** 2)
    Fm = np.sqrt(1.0 + x1 * x1)
    Fl = np.sqrt(1.0 + (0.5 * (x1 + x2)) ** 2)
    v = 0.25 * ((x + 3.0 * x1) / (Fh + Fm) + (3.0 * x1 + x2) / (Fm + Fl))
    a = 1.0 / (3.0 + math.sqrt(8.0))
    return sg.lfilter([1.0 + a], [1.0, a], v, axis=0)


def waveshape(x, shape="tanh", drive=4.0, bias=0.0, oversample=4, aa="none"):
    """Oversampled static waveshaper: f(drive*x + bias) - f(bias), scaled so a full-scale input stays
    within +-1. aa: none | adaa1 | adaa2 (antiderivative anti-aliasing, for the shapes that have one) | vicanek (algebraic shape only);
    ADAA + 2x does the work of 8x or more at low cost (lots B49, B71)."""
    if shape not in _SHAPES:
        raise ValueError(f"shape must be one of {'|'.join(_SHAPES)}")
    f = _SHAPES[shape]
    drive = float(np.clip(drive, 0.05, 100.0))
    bias = float(np.clip(bias, -2.0, 2.0))
    grid = np.linspace(-drive, drive, 201) + bias
    scale = max(float(np.max(np.abs(f(grid) - f(bias)))), 1e-6)
    if aa in ("none", "", None):
        g = f
    elif aa == "vicanek":
        if shape != "algebraic":
            raise ValueError("aa=vicanek needs shape=algebraic")
        g = vicanek_shaper
    elif aa == "adaa3":
        if shape not in _ANTI3:
            raise ValueError(f"no adaa3 for shape {shape} (hard | fold)")
        g = lambda v: adaa_n(v, f, _ANTI3[shape], 3)    # noqa: E731
    else:
        if aa not in ("adaa1", "adaa2") or _ANTI.get(shape, (None, None))[0] is None or (aa == "adaa2" and _ANTI[shape][1] is None):
            raise ValueError(f"no {aa} for shape {shape}")
        F1, F2 = _ANTI[shape]
        g = (lambda v: adaa1(v, f, F1)) if aa == "adaa1" else (lambda v: adaa2(v, f, F1, F2))
    return oversampled(lambda v: (g(drive * v + bias) - f(bias)) / scale, x, oversample)


@effect("overdrive", "Alias-free waveshaping overdrive (oversampled; bias adds even harmonics).",
        drive=(4.0, "input gain 0.05..100"), bias=(0.0, "-1..1 DC offset before the curve: asymmetry = even harmonics"),
        shape=("tanh", "tanh|tube|fuzz|fold|diode|hard|algebraic|hyper|asinh|asym (2x above 0, x/2 below, rounded)|tube_bendiksen (asymmetric tube, linear above, clipped below)"), tone=(4000, "post low-pass Hz (0 = off)"),
        mix=(1.0, "wet 0..1"), oversample=(4, "1|2|4|8 (1 = naive, aliases)"),
        aa=("none", "none | adaa1 | adaa2 | adaa3 (shapes hard, fold) | vicanek (shape algebraic): antiderivative anti-aliasing on top of the oversampling"))
def overdrive(x, sr=DEFAULT_SR, drive=4.0, bias=0.0, shape="tanh", tone=4000, mix=1.0, oversample=4, aa="none"):
    wet = waveshape(x, shape, drive, bias, oversample, aa)
    if bias != 0 or shape in ("tube", "diode", "asym", "tube_bendiksen"):
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


@effect("serge_fold", "Serge wavefolder: six diode stages, each fold exact (Lambert W), with its soft knee, oversampled.",
        drive=(1.0, "input gain: full scale = drive x 4 V; each stage takes about 0.6 V off the peak and folds it"), stages=(6, "1..6"), mix=(1.0, "wet 0..1"))
def serge_fold(x, sr=DEFAULT_SR, drive=1.0, stages=6, mix=1.0):
    """Vout = Vin - 2 l eta VT W((R1 Is / (eta VT)) e^(l Vin / (eta VT))), l = sign(Vin), per stage. The input gain is 4 x drive;
    the output gain is replaced by matching the input peak. W(e^u) is the Wright omega of u (the argument overflows near 5 V). Oversampled."""
    from scipy.special import wrightomega
    eta_vt = 1.752 * 25.864e-3
    a = math.log(33e3 * 2.52e-9 / eta_vt)

    def fold(v):
        v = 4.0 * float(np.clip(drive, 0.0, 10.0)) * v
        for _ in range(int(np.clip(stages, 1, 6))):
            lam = np.where(v >= 0.0, 1.0, -1.0)
            v = v - 2.0 * lam * eta_vt * np.real(wrightomega(a + lam * v / eta_vt))
        return v

    wet = oversampled(fold, x, 8)
    wet = wet * float(np.max(np.abs(x))) / (float(np.max(np.abs(wet))) + 1e-12)
    return x * (1 - mix) + wet * mix


def crossover(x, fc, sr=DEFAULT_SR):
    """Linkwitz-Riley 4th order split: (low, high), each a squared 2nd-order Butterworth. low + high is an allpass, flat in level (Zavalishin 10.27)."""
    lp = sg.butter(2, float(fc), "low", fs=sr, output="sos")
    hp = sg.butter(2, float(fc), "high", fs=sr, output="sos")
    return sg.sosfilt(lp, sg.sosfilt(lp, x, axis=0), axis=0), sg.sosfilt(hp, sg.sosfilt(hp, x, axis=0), axis=0)


def split_bands(x, freqs, sr=DEFAULT_SR):
    """Bands of x split at `freqs`, summing to a flat response: each lower band goes through the allpass of the higher crossovers, so the phases
    agree (Zavalishin 10.8; LR4 at 300 and 3000 Hz: -0.11 dB in the sum without that correction)."""
    bands = []
    rest = x
    fs = sorted(float(f) for f in freqs)
    for i, f in enumerate(fs):
        lo, rest = crossover(rest, f, sr)
        for g in fs[i + 1:]:
            lo = sum(crossover(lo, g, sr))
        bands.append(lo)
    return bands + [rest]


@effect("multiband", "Compressor per frequency band: tames one band without pumping the others (phase-corrected Linkwitz-Riley split, sums flat at ratio 1).",
        freqs=("300 3000", "crossover frequencies in Hz"), threshold=(-18.0, "dB, every band"), ratio=(2.0, "n:1"), attack=(0.01, "s"), release=(0.1, "s"),
        band=(-1, "index of the only band to compress (-1 = all), from 0 at the bottom"))
def multiband(x, sr=DEFAULT_SR, freqs="300 3000", threshold=-18.0, ratio=2.0, attack=0.01, release=0.1, band=-1):
    """Katz, Mastering Audio ch. 10-12: two bands if you can; the defaults are voicing choices."""
    fs = [float(s) for s in str(freqs).replace(",", " ").split()]
    out = 0.0
    for i, b in enumerate(split_bands(np.asarray(x, float), fs, sr)):
        out = out + (b if int(band) not in (-1, i) else _compressor(b, sr, threshold, ratio, attack, release, 0.0, 6.0, 1.0, False))
    return out


@effect("deesser", "De-esser: when the sibilant band gets loud against the rest of the sound, that band alone is turned down (voices, `say`).",
        freq=(6000.0, "centre of the sibilant band, Hz"), threshold=(-6.0, "dB of band level against the whole sound"), ratio=(4.0, "n:1"), q=(1.5, "band width"),
        attack=(0.001, "s"), release=(0.05, "s"))
def deesser(x, sr=DEFAULT_SR, freq=6000.0, threshold=-6.0, ratio=4.0, q=1.5, attack=0.001, release=0.05):
    """DAFX 5.2.4: band-pass detector, gain on a peak filter in the direct path, threshold relative to the overall level. Band: DAFX 2-6 kHz, Katz 3-9 kHz;
    attack "very fast", release "medium". threshold, ratio, q and the times in seconds are voicing choices. y = x + (g - 1) BP is the band shelf of Zavalishin 4.6."""
    x = np.asarray(x, float)
    a, r = 1.0 - math.exp(-1.0 / (attack * sr)), 1.0 - math.exp(-1.0 / (release * sr))
    env = lambda s: F.follow(np.ascontiguousarray(np.abs(to_mono(s)), dtype=np.float64), a, r) + 1e-9        # noqa: E731
    bp = F.svf(x, "bandpass", float(freq), sr, float(q))
    over = 20.0 * np.log10(env(bp) / env(x)) - float(threshold)
    g = 10.0 ** (-np.maximum(over, 0.0) * (1.0 - 1.0 / max(float(ratio), 1.0)) / 20.0)
    return x + bp * ((g - 1.0)[:, None] if x.ndim == 2 else g - 1.0)


@effect("exciter", "Exciter (Aphex type): treble is distorted and a little of it added back, for air without boosting the treble.",
        freq=(3000.0, "where the effect starts, Hz"), drive=(4.0, "into the nonlinearity"), mix=(0.15, "level of the added harmonics"))
def exciter(x, sr=DEFAULT_SR, freq=3000.0, drive=4.0, mix=0.15):
    """DAFX 5.4.1: high-passed side path, nonlinearity, compressor (attacks are enriched more than what follows), mixed low into the dry sound.
    The three defaults are voicing choices: DAFX names mix and frequency as the controls and gives no values."""
    h = F.highpass(np.asarray(x, float), freq, sr)
    w = waveshape(h, "tanh", drive, 0.0, 4)
    w = _compressor(w, sr, -18.0, 4.0, 0.005, 0.1, 0.0, 6.0, 1.0, False)
    return x + float(mix) * F.highpass(w, freq, sr)


@effect("tape_stop", "A tape or turntable stopping: speed and level fall to nothing.",
        start=(0.0, "s where the stop begins"), time=(0.5, "s from full speed to standstill"), alpha=(1.0, "curve of the speed (1 - t/time)^alpha"),
        gain_alpha=(1.0, "curve of the level, the same form"))
def tape_stop(x, sr=DEFAULT_SR, start=0.0, time=0.5, alpha=1.0, gain_alpha=1.0):
    """Faust misceffects.lib tapeStop: velocity = (1 - n/T)^alpha, gain the same with gainAlpha; its test values T = 22050 samples at 44.1 kHz, curves 1."""
    s, T = samples(start, sr), max(samples(time, sr), 1)
    u = 1.0 - np.arange(T) / T
    rate = np.concatenate([np.ones(s), u ** float(alpha)])
    y = varispeed(x, rate)
    g = np.concatenate([np.ones(s), u ** float(gain_alpha)])
    return y * (g[:, None] if y.ndim == 2 else g)


@effect("crunch", "Dead zone: only the crests above the threshold survive; on noise, a sparse crackle (fire, Geiger, dust).", threshold=(0.315, "0..1"))
def crunch(x, sr=DEFAULT_SR, threshold=0.315):
    """rjlib noizer: y = x - clip(x, -t, t), makeup 2 + 36 t^2, then a 5 Hz high-pass. Default t: its fire2 preset."""
    t = float(threshold)
    return F.highpass((x - np.clip(x, -t, t)) * (2.0 + 36.0 * t * t), 5.0, sr)


@effect("chebyshev", "Waveshaping by Chebyshev polynomials: a sine at full level comes out with exactly the harmonics you list; quieter input gets fewer of them.",
        harmonics=("1:1,2:0.3,3:0.2", "harmonic:level, ..."), oversample=(4, "1|2|4|8"))
def chebyshev(x, sr=DEFAULT_SR, harmonics="1:1,2:0.3,3:0.2", oversample=4):
    """T_k(cos wn) = cos(k w n); the k-th term falls as a^k with the input level (Puckette, Theory and Technique, E05). Even terms add DC, blocked after."""
    f, g = _points(harmonics)
    c = np.zeros(int(max(f)) + 1)
    c[np.asarray(f, int)] = g
    wet = oversampled(lambda v: np.polynomial.chebyshev.chebval(np.clip(v, -1.0, 1.0), c), np.asarray(x, float), oversample)
    return dc_blocker(wet, sr)


@numba.njit(cache=True)
def _diode_clip_kernel(u, T, R, C, Is, vT):
    """RC low-pass into two antiparallel diodes, y' = (u - y)/(RC) - (2 Is / C) sinh(y / vT), trapezoidal, Newton per sample."""
    y = np.empty_like(u)
    v = 0.0
    up = 0.0
    k = 2.0 * Is / C
    rc = 1.0 / (R * C)
    fp = 0.0
    for n in range(u.shape[0]):
        w = v
        for _ in range(16):
            e = min(max(w / vT, -60.0), 60.0)
            f = (u[n] - w) * rc - k * math.sinh(e)
            d = 1.0 + 0.5 * T * (rc + k * math.cosh(e) / vT)
            s = (w - v - 0.5 * T * (f + fp)) / d
            w -= s
            if abs(s) < 1e-10:
                break
        e = min(max(w / vT, -60.0), 60.0)
        fp = (u[n] - w) * rc - k * math.sinh(e)
        v = w
        up = u[n]
        y[n] = w
    return y


@effect("diode_clipper", "Diode clipper as a circuit: unlike `overdrive`, how much it clips depends on frequency (the RC rolls the treble off before the diodes).",
        drive=(4.0, "full scale = drive volts into the circuit"), mix=(1.0, "wet 0..1"))
def diode_clipper(x, sr=DEFAULT_SR, drive=4.0, mix=1.0):
    """Holters DAFx-19 sec. 4.1: R 1 kOhm, C 33 nF, Is 1 fA, vT 25 mV (lot B77), trapezoidal, run here at 8x instead of a table + ADAA. The output in volts is scaled by 1/0.7 V."""
    def one(s):
        return _diode_clip_kernel(np.ascontiguousarray(s, dtype=np.float64), 1.0 / (8.0 * sr), 1e3, 33e-9, 1e-15, 0.025)
    wet = oversampled(lambda v: _per_channel(one, v * float(drive)), np.asarray(x, float), 8) / 0.7
    return x * (1 - mix) + wet * mix


@effect("lockhart_fold", "Lockhart wavefolder (West Coast transistor circuit, four stages), exact Lambert W at each stage, oversampled.",
        drive=(1.0, "input gain"), stages=(4, "1..4"), load=(7500.0, "load resistor Ohm, 1000..50000"), tone=(1300.0, "post low-pass Hz (0 = off)"), mix=(1.0, "wet 0..1"))
def lockhart_fold(x, sr=DEFAULT_SR, drive=1.0, stages=4, load=7500.0, tone=1300.0, mix=1.0):
    """Vout = a Vin - l eta VT W(D e^(l b Vin)), l = sign(Vin), a = 2 RL / R, b = (2 RL + R) / (eta VT R), D = RL Is / (eta VT), per stage;
    pre-gain 1/3, post-gain 3, one-pole low-pass. Oversampled 8x."""
    from scipy.special import wrightomega
    RL, R, Is, vt = float(load), 15e3, 1e-17, 25.864e-3
    a, b = 2.0 * RL / R, (2.0 * RL + R) / (vt * R)
    ld = math.log(RL * Is / vt)

    def fold(v):
        v = float(drive) * v / 3.0
        for _ in range(int(np.clip(stages, 1, 4))):
            lam = np.where(v >= 0.0, 1.0, -1.0)
            v = a * v - lam * vt * np.real(wrightomega(ld + lam * b * v))
        return 3.0 * v

    wet = oversampled(fold, np.asarray(x, float), 8)
    if tone and tone < 0.49 * sr:
        wet = F.onepole(wet, "lowpass", tone, sr)
    wet = wet * float(np.max(np.abs(x))) / (float(np.max(np.abs(wet))) + 1e-12)
    return x * (1 - mix) + wet * mix


@numba.njit(cache=True)
def _delta_sigma(x, order):
    """One-bit modulator: DAFX 12: q = (s > 0), first order s += x - 2(q - 1/2); second order two integrators in cascade."""
    y = np.empty_like(x)
    s1 = s2 = 0.0
    for n in range(x.shape[0]):
        q = 1.0 if (s2 if order == 2 else s1) > 0.0 else -1.0
        s1 += x[n] - q
        if order == 2:
            s2 += s1 - q
        y[n] = q
    return y


def delta_sigma(x, order=2, osr=64):
    """x (|x| < 1) as a one-bit stream at osr times the rate: the share of +1 follows the level (Smith, DSPG ch.3; ADC16071: decimation by 64)."""
    return _per_channel(lambda s: _delta_sigma(np.ascontiguousarray(s, dtype=np.float64), int(order)), resample_sinc(np.asarray(x, float), 1, int(osr)))


@effect("delta_sigma", "One-bit DAC / PDM: the sound through a sigma-delta stream at a low oversampling ratio, its noise left in (lo-fi hiss that rises with frequency).",
        order=(1, "1|2"), osr=(16, "oversampling ratio 4..128 (lower = dirtier)"))
def delta_sigma_fx(x, sr=DEFAULT_SR, order=1, osr=16):
    r = int(np.clip(osr, 4, 128))
    return resample_sinc(delta_sigma(np.clip(x, -0.99, 0.99) * 0.5, order, r), r, 1)[: x.shape[0]] * 2.0


def dsf(n, f0, sr=DEFAULT_SR, a=0.7, ratio=1.0, terms=None):
    """Discrete summation (Moorer 1976, Designing Sound eq. 17.2 and the N-term form): sum_k a^k sin(theta + k beta), theta = 2 pi f0 t, beta = ratio theta, in closed form.
    Band-limited with `terms`; sqrt(1 - a^2) brings the RMS to 0.7071, as a sine. Peak not bounded (Addendum c12 l.12955-12997)."""
    t = 2.0 * np.pi * f0 * np.arange(int(n)) / sr
    b = ratio * t
    num = np.sin(t) - a * np.sin(t - b)
    if terms is not None:
        N = int(terms)                                    # k = 0 .. N - 1
        num = num - a ** N * (np.sin(t + N * b) - a * np.sin(t + (N - 1) * b))
    den = 1.0 + a * a - 2.0 * a * np.cos(b)
    return num / np.maximum(den, 1e-12) * math.sqrt(max(1.0 - a * a, 0.0))


def k_meter(x, k=14, sr=DEFAULT_SR):
    """K-System reading of a program (Katz, Mastering Audio): RMS over 600 ms, in dB above 0 dBFS + k: a program at K-14 reads 0 at -14 dBFS RMS. Returns (rms_db_k, peak_db_k) for the loudest window. The 83 dB SPL calibration of 0 dB concerns the room, not the file."""
    m = to_mono(np.asarray(x, float))
    w = max(1, min(samples(0.6, sr), m.shape[0]))
    e = sg.fftconvolve(m * m, np.ones(w) / w, mode="valid")
    return 10.0 * math.log10(max(float(e.max()), 1e-20)) + float(k), 20.0 * math.log10(max(float(np.abs(m).max()), 1e-10)) + float(k)


def allpass_for_delay(D, w):
    """First-order allpass coefficient giving group delay D samples at w rad/sample: a = (-D cos w + sqrt(1 - D^2 + D^2 cos^2 w)) / (1 + D) (Parker, B41), the positive root."""
    return (-D * math.cos(w) + math.sqrt(max(1.0 - D * D + (D * math.cos(w)) ** 2, 0.0))) / (1.0 + D)


@effect("mulaw", "Companded quantisation (telephone codec grain: noise follows the signal level).",
        bits=(8, "2..16 bits in the compressed domain"), law=("mu", "mu (mu=255) | a (A=87.6)"))
def mulaw(x, sr=DEFAULT_SR, bits=8, law="mu"):
    comp, exp = (alaw_compress, alaw_expand) if law == "a" else (mulaw_compress, mulaw_expand)
    S = float(2 ** (int(np.clip(bits, 2, 16)) - 1) - 1)
    return exp(np.round(comp(x) * S) / S)


@effect("dither", "Reduce bit depth with dither (and optional noise shaping) instead of bare rounding.",
        bits=(8, "1..24"), shape=("tpdf", "none | tpdf | gauss (sd 2/3 LSB) | shaped (tpdf + first-order noise shaping) | fweighted | lipshitz (psychoacoustic shaping)"),
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
        kind=("adpcm", "adpcm (4-bit IMA) | cvsd | cvsd_irig (military voice: 16 kbps, 32 at quality > 0.5, its band limits) | delta | gsm (8 kHz LPC, GSM-like) | dct (block transform, streaming artefact)"),
        quality=(0.5, "0..1: adpcm rate 6-32 kHz; cvsd clock 12-64 kHz; delta clock 48-192 kHz; gsm pulse bits 2-4; dct coarse-fine"))
def codec(x, sr=DEFAULT_SR, kind="adpcm", quality=0.5):
    q = float(np.clip(quality, 0.0, 1.0))
    if kind == "adpcm":
        return _at_rate(lambda s: adpcm_encode(s)[1], x, sr, min(_geo(6000, 32000, q), sr))
    if kind == "cvsd":
        clock = int(_geo(12000, 64000, q))
        return _at_rate(lambda s: cvsd_encode(s, clock, run=3)[1], x, sr, clock)    # run of three: IRIG 106 App. F §5.4
    if kind == "cvsd_irig":
        # IRIG 106 App. F Tables F-1..F-3: run of three, 16 or 32 kbps, syllabic 5 ms, integrator 1 ms (the defaults of cvsd_encode). Band: high-pass 300 Hz, then a Butterworth 3 kHz
        # of order 10: 29.2 dB at 4200 Hz for the >= 25 dB of Table F-2, 0.24 dB lost at 2600 Hz (lot B26, its own calculation).
        y = steep_filter(x, "highpass", 300.0, sr, 4)
        y = _at_rate(lambda s: cvsd_encode(s, 32000 if q > 0.5 else 16000, run=3)[1], y, sr, 32000 if q > 0.5 else 16000)
        return steep_filter(y, "lowpass", min(3000.0, 0.45 * sr), sr, 10)
    if kind == "delta":
        clock = int(_geo(48000, 192000, q))
        step = 2 * np.pi * 800.0 / clock  # slew limit set to a full-scale 800 Hz sine
        return _at_rate(lambda s: delta_encode(s, step)[1], x, sr, clock)
    if kind in ("gsm", "gsm-like"):
        y = steep_filter(x, "bandpass", 200.0, sr, 4, fc2=min(3400.0, 0.45 * sr))
        return _at_rate(lambda s: gsm_like(s, int(round(2 + 2 * q))), y, sr, 8000)
    if kind == "dct":
        block = int(512 * sr / 44100)  # 11.6 ms blocks
        qmax, power, bits = _geo(128, 4, q), 2.0, int(round(5 + 4 * q))  # quality map
        return _per_channel(lambda s: block_dct(s, block, qmax, power, bits), np.clip(x, -1, 1))
    raise ValueError("codec kind must be adpcm|cvsd|cvsd_irig|delta|gsm|dct")


# telephone = 8 kHz, 8-bit companded, 200-3200 Hz (DSPG ch.1/22, Table 22-2).
# nes: DMC rate $F = 33143.9 Hz, 7-bit output counter (NESdev wiki, APU DMC). snes: S-DSP output 32000 Hz nominal,
# 16-bit stereo DAC (SNESdev wiki, S-DSP). gameboy: 4-bit wave samples (Pan Docs, Audio Registers, CH3).
# amiga: 3579545 / 214 (NTSC clock constant / period, Amiga Hardware Reference Manual, "Limitations on Selection
# of Sampling Period"); low-pass "becomes active around 4 KHz" (HRM, "Low-Pass Filter"). sp1200: 26.04 kHz, 12-bit
# linear (Rossum Electro-Music SP-1200 page, reissue of the original data format). gameboy rate,
# snes band and ADPCM stand-in for BRR, amiga 8 bits, voicemail, walkie.
LOFI_PRESETS = {
    "nes": dict(rate=33144, bits=7, law="linear", band=None, hold=True, adpcm=False),
    "snes": dict(rate=32000, bits=16, law="linear", band=(20, 7000), hold=False, adpcm=True),
    "gameboy": dict(rate=8192, bits=4, law="linear", band=None, hold=True, adpcm=False),
    "amiga": dict(rate=16726, bits=8, law="linear", band=(20, 4000), hold=True, adpcm=False),
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


@numba.njit(cache=True)
def _ja_kernel(x, drive, Ms, a, k, alpha, c, sub):
    """Jiles-Atherton as in Faust hysteresis.lib: dM/dH = (c dMan/dH + d/pin) / (1 - c alpha dMan/dH), Man = tanh((H + alpha M)/a'),
    a' = a/Ms, k' = k/Ms, pin = dir k' - alpha d, d = (Man - M) / (1 + 3 |Man - M|); out x a/(c Ms), unity for small signals.
    # ponytail: H linear between samples, not Faust's Hermite: fine at these sub-steps."""
    ap, kp = a / Ms, k / Ms
    y = np.empty_like(x)
    M = 0.0
    H0 = 0.0
    for n in range(x.shape[0]):
        H1 = drive * x[n]
        dH = (H1 - H0) / sub
        sgn = 1.0 if dH >= 0.0 else -1.0
        H = H0
        for _ in range(sub):
            v = (H + alpha * M) / ap
            Man = math.tanh(v)
            ch = math.cosh(min(abs(v), 300.0))
            dMan = 1.0 / (ap * ch * ch)
            d = (Man - M) / (1.0 + 3.0 * abs(Man - M))
            pin = sgn * kp - alpha * d
            if abs(pin) < 1e-3:                               # sigma 1e-3
                pin = 1e-3 if pin >= 0.0 else -1e-3
            M += dH * (c * dMan + d / pin) / max(1.0 - c * alpha * dMan, 1e-3)
            M = min(max(M, -1.0), 1.0)
            H += dH
        H0 = H1
        y[n] = M * a / (c * Ms) / drive
    return y


def ja_hysteresis(x, sr=DEFAULT_SR, drive=1.0, Ms=380.0, a=720.0, k=380.0, alpha=0.015, c=0.25, sub=4):
    """Magnetic hysteresis of tape, defaults and 4 sub-steps from Faust hysteresis.lib (empirical constants), HP 7 Hz."""
    def one(ch):
        return F.highpass(_ja_kernel(np.ascontiguousarray(ch, dtype=np.float64), float(max(drive, 0.05)) * 3.0, Ms, a, k, alpha, c, int(sub)), 7.0, sr)
    x = np.asarray(x, float)
    return one(x) if x.ndim == 1 else np.stack([one(x[:, i]) for i in range(x.shape[1])], axis=1)


@effect("tape", "Tape machine: wow and flutter (band-limited variable-speed read), saturation, head bump, hiss.",
        wow=(0.15, "slow speed drift depth in percent (about 0.6 Hz)"), flutter=(0.08, "fast speed wobble depth in percent (about 7 Hz)"),
        drive=(1.5, "saturation input gain 0.1..10"), bump=(2.5, "head-bump boost in dB around 80 Hz"),
        hf=(12000, "high-frequency roll-off Hz"), hiss=(-58, "hiss level in dB RMS (-120 = none)"), seed=(0, "int"),
        model=("tanh", "tanh | hysteresis: Jiles-Atherton magnetisation, the loop real tape has"),
        wow_rate=(0.6, "wow frequency Hz (0.556 = a record at 33 1/3 rpm, with wow about 0.5)"))
def tape(x, sr=DEFAULT_SR, wow=0.15, flutter=0.08, drive=1.5, bump=2.5, hf=12000, hiss=-58, seed=0, model="tanh", wow_rate=0.6):
    rng = np.random.default_rng(int(seed))
    n = x.shape[0]
    t = np.arange(n) / sr
    # flutter 7.3 Hz: inside "above 6 Hz" (Zölzer 2002 DAFX glossary p.517). wow 0.6 Hz, the random
    # component below 20 Hz; depths are typical consumer-deck figures
    w = float(np.clip(wow, 0, 5)) / 100.0
    fl = float(np.clip(flutter, 0, 5)) / 100.0
    drift = moving_average(rng.standard_normal(n), int(sr / 20), 2)
    drift /= max(np.max(np.abs(drift)), 1e-9)
    rate = 1 + w * np.sin(2 * np.pi * float(np.clip(wow_rate, 0.05, 6.0)) * t + rng.uniform(0, 6.28)) + fl * (0.7 * np.sin(2 * np.pi * 7.3 * t + rng.uniform(0, 6.28)) + 0.3 * drift)
    y = varispeed(x, rate) if (w > 0 or fl > 0) else np.asarray(x, dtype=np.float64)
    d = float(np.clip(drive, 0.1, 10.0))
    if model == "hysteresis":
        y = ja_hysteresis(y, sr, d)
    else:
        y = oversampled(lambda v: np.tanh(d * v) / math.tanh(d), y, 2)
    if bump:
        y = F.peak(y, 80.0, sr, 1.0, float(np.clip(bump, -12, 12)))  # bump centre 80 Hz, Q 1
    if hf and hf < 0.49 * sr:
        y = F.lowpass(y, hf, sr)
    if hiss > -119:
        h = F.highshelf(gaussian(n, int(seed) + 7), 3000.0, sr, 6.0) * db_to_gain(hiss) * 0.6  # hiss tilt
        y = y + (h[:, None] if y.ndim == 2 else h)
    peak = np.max(np.abs(y))
    return y * (1.4 / peak) if peak > 1.4 else y  # a big bump on a saturated signal may exceed full scale


# Surface mass = (kg/m^2 per mm) x thickness, coincidence fc = A / t (Hz, t in mm).
WALL_MATERIALS = {
    "window": dict(mass=10.0, leak=30.0, coincidence=3800.0),   # 4 mm glass
    "door": dict(mass=12.0, leak=22.0, coincidence=0.0),        # hollow interior door, gaps around it
    "solid_door": dict(mass=28.0, leak=30.0, coincidence=0.0),
    "drywall": dict(mass=10.65, m2=10.65, gap=0.09, stud=0.6, leak=52.0, coincidence=3000.0),  # stud partition, one 13 mm gypsum board each side
    "brick": dict(mass=211.0, leak=52.0, coincidence=0.0),      # 110 mm brick
    "concrete": dict(mass=345.0, leak=58.0, coincidence=125.0), # 150 mm concrete
}


WALL_ETA = 0.03          # total loss factor of an installed panel (internal + edges + radiation)
WALL_BAND = 2 ** (1 / 6) - 2 ** (-1 / 6)   # third-octave bandwidth / centre: the band that holds fc


def wall_loss_db(f, mass, leak=40.0, coincidence=0.0, eta=WALL_ETA, m2=0.0, gap=0.0, stud=0.0):
    """Transmission loss of a single partition: field-incidence mass law TL = 20 log10(f m) - 48 dB (f in Hz, m in kg/m^2:
    +6 dB per octave and per doubling of mass), never below 0, capped at `leak` (gaps and flanking set the ceiling).
    Above the coincidence frequency fc the bending waves match the sound: TL = 20 log10(f m) - 42 + 10 log10(2 eta
    (f / fc - 1) / pi), with f / fc - 1 held at the third-octave width WALL_BAND (0.23) in the band around fc (the dip),
    rising to 9 dB per octave far above; joined to the mass law over the octave below fc.
    With a second leaf `m2` (kg/m^2) across an air gap `gap` (m), the double wall has three regions: below the
    mass-air-mass resonance f0 = sqrt(rho c^2 (m1 + m2)/(d m1 m2))/(2 pi) the mass law of m1 + m2; up to the cavity
    limit fl = c/(2 pi d), R1 + R2 + 20 log10(f d) - 29 (18 dB per octave, with its dip just above f0); above, R1 + R2 + 6.
    `stud` (m, spacing of line studs bridging the leaves, 0 = none) caps it at R(m1 + m2) + 10 log10(b fc)
    + 20 log10(m1/(m1 + m2)) - 18."""
    if m2 > 0 and gap > 0:
        f = np.asarray(f, dtype=np.float64)
        r1 = wall_loss_db(f, mass, 1e3, coincidence, eta)
        r2 = wall_loss_db(f, m2, 1e3, coincidence, eta)
        mt = mass + m2
        rm = 20 * np.log10(np.maximum(f * mt, 1e-9)) - 48.0
        f0 = math.sqrt(1.21 * 343.0 ** 2 * mt / (gap * mass * m2)) / (2 * np.pi)
        fl = 343.0 / (2 * np.pi * gap)
        tl = np.where(f < f0, rm, np.where(f < fl, r1 + r2 + 20 * np.log10(np.maximum(f * gap, 1e-9)) - 29.0, r1 + r2 + 6.0))
        if stud > 0 and coincidence > 0:
            tl = np.minimum(tl, np.maximum(rm, rm + 10 * math.log10(stud * coincidence) + 20 * math.log10(mass / mt) - 18.0))
        return np.clip(tl, 0.0, leak)
    f = np.asarray(f, dtype=np.float64)
    fm = np.maximum(f * mass, 1e-9)
    tl = 20 * np.log10(fm) - 48.0
    if coincidence > 0:
        fc = float(coincidence)
        above = 20 * np.log10(fm) - 42.0 + 10 * np.log10(2 * eta * np.maximum(f / fc - 1.0, WALL_BAND) / np.pi)
        lo = 20 * np.log10(max(0.5 * fc * mass, 1e-9)) - 48.0
        hi = 20 * np.log10(max(fc * mass, 1e-9)) - 42.0 + 10 * np.log10(2 * eta * WALL_BAND / np.pi)
        x = np.log2(np.maximum(f, 1e-9) / (0.5 * fc))
        tl = np.where(f >= fc, np.minimum(tl, above), np.where(f > 0.5 * fc, lo + (hi - lo) * x, tl))
    return np.clip(tl, 0.0, leak)


def leak_loss_db(tl, leak_area):
    """Transmission loss of a wall of loss ``tl`` (dB) with an open fraction r of its area: the weakest path,
    TL' = -10 log10((1 - r) 10^(-TL/10) + r)."""
    r = float(np.clip(leak_area, 0.0, 1.0))
    return -10.0 * np.log10((1.0 - r) * 10.0 ** (-np.asarray(tl, dtype=np.float64) / 10.0) + r)


RW_BANDS = (100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150)
RW_REF = (33, 36, 39, 42, 45, 48, 51, 52, 53, 54, 55, 56, 56, 56, 56, 56)


def weighted_rw(tl):
    """Single-number rating Rw of a third-octave transmission loss on RW_BANDS: the reference curve RW_REF is
    raised in 1 dB steps as far as the unfavourable deviations (reference above tl) sum to at most 32 dB;
    Rw is the shifted curve's value at 500 Hz."""
    tl = np.asarray(tl, dtype=np.float64)
    ref = np.asarray(RW_REF, dtype=np.float64)
    s = math.floor(float(np.min(tl - ref)))
    while np.sum(np.maximum(ref + s + 1 - tl, 0.0)) <= 32.0:
        s += 1
    return int(52 + s)


def wall_rw(material="drywall"):
    """Rw of a WALL_MATERIALS preset, its loss read at the third-octave centres."""
    return weighted_rw(wall_loss_db(np.asarray(RW_BANDS, dtype=np.float64), **WALL_MATERIALS[material]))


@effect("wall", "Heard through a wall, door or window: mass-law transmission loss (real level loss, not just dullness).",
        material=("drywall", "window|door|solid_door|drywall|brick|concrete"),
        thickness=(1.0, "mass multiplier 0.25..8 (2 = twice as thick: 6 dB more loss)"),
        leak=(None, "loss ceiling in dB set by gaps and flanking (default per material: door 22 .. concrete 58)"),
        makeup=(0.0, "dB of gain after the loss (the true loss is 20-55 dB)"),
        leak_area=(None, "open fraction 0..1 of the wall (gaps, keyholes): TL' = -10 log((1 - r) 10^(-TL/10) + r), in place of the leak ceiling"))
def wall(x, sr=DEFAULT_SR, material="drywall", thickness=1.0, leak=None, makeup=0.0, leak_area=None):
    if material not in WALL_MATERIALS:
        raise ValueError(f"wall material must be one of {'|'.join(WALL_MATERIALS)}")
    m = WALL_MATERIALS[material]
    k = float(np.clip(thickness, 0.25, 8.0))
    two = dict(m2=m.get("m2", 0.0) * k, gap=m.get("gap", 0.0), stud=m.get("stud", 0.0))
    f = np.geomspace(20.0, 0.499 * sr, 48)
    if leak_area is None:
        tl = wall_loss_db(f, m["mass"] * k, float(np.clip(m["leak"] if leak is None else leak, 0.0, 90.0)), m["coincidence"], **two)
    else:
        tl = leak_loss_db(wall_loss_db(f, m["mass"] * k, 200.0, m["coincidence"], **two), leak_area)
    h = fir_from_curve(f, -tl, sr, int(0.03 * sr) | 1, phase="min")
    return fir_filter(x, h, compensate=False) * db_to_gain(float(np.clip(makeup, -60, 80)))


@effect("dc_block", "Remove DC offset (one-pole/one-zero DC blocker).", cutoff=(20.0, "Hz"))
def dc_block(x, sr=DEFAULT_SR, cutoff=20.0):
    return dc_blocker(x, sr, cutoff)


# ======================================================================================================
# 9. Registered sfx
# ======================================================================================================

_BED_RATE = {"crackle": 25.0, "geiger": 12.0, "static_field": 4000.0}  # default event rates


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
        sg_ev = {"crackle": 0.8, "geiger": 0.15, "static_field": 0.5}[kind]  # log-normal spreads
        ev = poisson_events(n, rate, sr, seed, sigma=sg_ev)
        if not np.any(ev):
            ev[n // 3] = 1.0  # a Poisson stream may be empty; a bed must not be silent
        if kind == "geiger":
            # each count = a short damped ring of the speaker (2.2-3.4 kHz, 0.35 ms decay)
            m = samples(0.004, sr)
            tt = np.arange(m) / sr
            click = np.sin(2 * np.pi * (2200.0 + 1200.0 * bright) * tt) * np.exp(-tt / 0.00035)
        else:
            # one-sided exponential tick (0.05-0.25 ms)
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


def steepen(x, sr=DEFAULT_SR, length_m=2.0, peak_pa=12000.0, oversample=4):
    """Nonlinear propagation down a brass bore (Chaigne & Kergomard 8.4, Hirschberg 1996): crests travel faster, arrival
    earlier by L (g+1) p / (2 g P0 c), so a wave of slope dp/dt shocks at x_s = 2 g P0 c / ((g+1) dp/dt). ``peak_pa``: what
    the peak of x is at the mouthpiece (trombone ff +-10-15 kPa). Past x_s the wave keeps its front: arrivals forced monotonic."""
    from scipy.signal import resample_poly
    x = np.asarray(x, float)
    u = resample_poly(x, oversample, 1)
    pk = float(np.max(np.abs(u))) + 1e-12
    gamma, P0, c = 1.4, 101325.0, 343.0
    fs = sr * oversample
    shift = float(length_m) * (gamma + 1.0) * (u / pk * float(peak_pa)) / (2.0 * gamma * P0 * c)
    arrive = np.maximum.accumulate(np.arange(u.size) / fs - shift)
    y = np.interp(np.arange(u.size) / fs, arrive + np.arange(u.size) * 1e-12, u)
    return resample_poly(y, 1, oversample)[:x.size]


def pqmf_bank(bands=16, atten=100.0):
    """Pseudo-QMF cosine-modulated bank: Kaiser prototype h (``atten`` dB stop band, transition pi/(2M)), its cutoff
    set by bisection so that |H(pi/(2M))|^2 = 1/2 (near-perfect reconstruction); analysis h_k[n] = 2 h[n] cos((2k+1)
    pi/(2M) (n - (N-1)/2) + (-1)^k pi/4), synthesis the same with - (-1)^k pi/4. Returns (analysis, synthesis), M x N."""
    m = int(bands)
    taps, beta = sg.kaiserord(float(atten), 1.0 / (2 * m))
    taps += taps % 2 == 0
    lo, hi = 0.25 / m, 1.0 / m
    for _ in range(60):
        wc = 0.5 * (lo + hi)
        h = sg.firwin(taps, wc, window=("kaiser", beta))
        g = abs(np.sum(h * np.exp(-1j * np.pi / (2 * m) * np.arange(taps)))) ** 2
        lo, hi = (wc, hi) if g < 0.5 else (lo, wc)
    n = np.arange(taps) - (taps - 1) / 2
    k = np.arange(m)[:, None]
    arg = (2 * k + 1) * np.pi / (2 * m) * n
    sgn = np.where(k % 2 == 0, 1.0, -1.0) * np.pi / 4
    return 2 * h * np.cos(arg + sgn), 2 * h * np.cos(arg - sgn)


def pqmf_analysis(x, bands=16):
    """Split a mono signal into ``bands`` critically sampled sub-bands (each 1/bands of the rate)."""
    ha, _ = pqmf_bank(bands)
    return np.stack([sg.upfirdn(h, np.asarray(x, dtype=np.float64), 1, bands) for h in ha])


def pqmf_synthesis(sub, n=None):
    """Rebuild the signal from pqmf_analysis bands; the bank delay (N - 1) is removed, length ``n``."""
    m = sub.shape[0]
    _, hs = pqmf_bank(m)
    y = m * sum(sg.upfirdn(hs[k], sub[k], m, 1) for k in range(m))
    d = hs.shape[1] - 1
    n = len(y) - 2 * d if n is None else int(n)
    return _fit(y[d:], n)


def ltv_fir(x, mags, hop, taps=257):
    """Linear time-varying FIR by frequency sampling: frame i of ``hop`` input samples goes through the zero-phase
    kernel whose magnitude is mags[i] (points spread from 0 to Nyquist, Hann-windowed), tails overlap-added.
    For noise that follows a control (wind speed, rpm, bow pressure)."""
    x = np.asarray(x, dtype=np.float64)
    mags = np.atleast_2d(np.asarray(mags, dtype=np.float64))
    taps = int(taps) | 1
    w = np.hanning(taps + 2)[1:-1]
    y = np.zeros(len(x) + taps)
    for i, s in enumerate(range(0, len(x), int(hop))):
        mag = np.interp(np.linspace(0, 1, taps // 2 + 1), np.linspace(0, 1, mags.shape[1]), mags[min(i, len(mags) - 1)])
        h = np.roll(np.fft.irfft(mag, taps), taps // 2) * w
        seg = x[s:s + int(hop)]
        y[s:s + len(seg) + taps - 1] += np.convolve(seg, h)
    return y[taps // 2:taps // 2 + len(x)]


def compress_envelope(env, floor_db=-81.0, step_db=1.0, bits=8):
    """Envelope storage of a modal synthesiser: levels below ``floor_db`` (re the peak) are dropped to zero, the rest
    quantised in ``step_db`` steps on ``bits`` bits (8 bits at 1 dB = 256 dB of range)."""
    env = np.asarray(env, dtype=np.float64)
    ref = max(float(np.max(np.abs(env))), 1e-30)
    db = 20 * np.log10(np.maximum(np.abs(env) / ref, 1e-30))
    q = np.clip(np.round(db / step_db), -(2 ** bits - 1), 0) * step_db
    return np.where(db < floor_db, 0.0, ref * 10 ** (q / 20))


def mode_quota(amps, budget):
    """Mode quota of a modal synthesiser sharing a budget of B modes among S sounding ones: keep the loudest
    ``budget`` modes, cutting the fraction (S - B)/S. Returns the kept indices, loudest first, and that fraction."""
    a = np.abs(np.asarray(amps, dtype=np.float64))
    s, b = len(a), int(min(max(budget, 0), len(a)))
    return np.argsort(-a, kind="stable")[:b], ((s - b) / s if s else 0.0)


@numba.njit(cache=True)
def _complex_osc_kernel(n, sr, f1, f2, fm1, fm2, pm1, pm2, ring1, ring2, mul1, mul2):
    out = np.empty(n)
    p1 = 0.0
    p2 = 0.0
    o1 = 0.0
    o2 = 0.0
    for i in range(n):
        m2 = (o2 + 1.0) * 0.5
        m1 = (o1 + 1.0) * 0.5
        a = math.sin(p1 + (-np.pi + 2 * np.pi * m2) * pm1) * (mul1 + o2 * ring1)
        b = math.sin(p2 + (-np.pi + 2 * np.pi * m1) * pm2) * (mul2 + o1 * ring2)
        p1 += 2 * np.pi * (f1 + (1.0 + 999.0 * m2) * fm1) / sr
        p2 += 2 * np.pi * (f2 + (1.0 + 999.0 * m1) * fm2) / sr
        o1 = max(-1.0, min(1.0, a))
        o2 = max(-1.0, min(1.0, b))
        out[i] = a
    return out


@sfx("complex_osc", "Two cross-modulated sines (West-Coast complex oscillator): each one modulates the other's frequency, phase "
     "and amplitude; folded or wrapped output.",
     dur=(2.0, "s"), freq1=(40.0, "Hz, main oscillator"), freq2=(0.5, "Hz, second oscillator"),
     fm1=(0.1, "frequency modulation of osc 1 by osc 2 (x 1..1000 Hz)"), fm2=(0.1, "the reverse"),
     pm1=(0.0, "phase modulation of osc 1 by osc 2 (x -pi..pi)"), pm2=(0.0, "the reverse"),
     ring1=(1.0, "amplitude (ring) modulation of osc 1 by osc 2"), ring2=(1.0, "the reverse"),
     out=("fold", "fold (+-0.8) | wrap (+-0.9)"),
     preset=("none", "none | olarte (fm1 0.143219, fm2 0.2345, pm1 0.013, pm2 0.081, ring 0.1997)"))
def complex_osc(sr=DEFAULT_SR, dur=2.0, freq1=40.0, freq2=0.5, fm1=0.1, fm2=0.1, pm1=0.0, pm2=0.0, ring1=1.0, ring2=1.0,
                out="fold", preset="none"):
    if preset == "olarte":
        fm1, fm2, pm1, pm2, ring1, ring2 = 0.143219, 0.2345, 0.013, 0.081, 0.1997, 0.1997
    n = samples(float(np.clip(dur, 0.05, 60.0)), sr)
    y = _complex_osc_kernel(n, float(sr), float(freq1), float(freq2), float(fm1), float(fm2), float(pm1), float(pm2),
                            float(ring1), float(ring2), 1.0, 1.0)
    if out == "wrap":
        y = (y + 0.9) % 1.8 - 0.9
    else:
        y = 0.8 * _tri_fold(y / 0.8)
    return dc_blocker(y, sr, 10.0)


def fdn_allpass_matrix(A, g):
    """Feedback matrix of an all-pass FDN from a unitary A (N x N) and the gains g: [[-A G, A], [I - G^2, G]]."""
    A = np.asarray(A, float)
    G = np.diag(np.broadcast_to(np.asarray(g, float), (A.shape[0],)))
    return np.block([[-A @ G, A], [np.eye(A.shape[0]) - G @ G, G]])


def fdn_poles(A, m):
    """Poles of an FDN with feedback matrix A and delays m (samples): the eigenvalues of its delay-line state matrix,
    the roots of det(diag(z^m) - A)."""
    A, m = np.asarray(A, float), [int(v) for v in m]
    off = np.concatenate(([0], np.cumsum(m)))
    P = np.zeros((off[-1], off[-1]))
    for i, mi in enumerate(m):
        for j, mj in enumerate(m):
            P[off[i], off[j] + mj - 1] = A[i, j]
        for k in range(1, mi):
            P[off[i] + k, off[i] + k - 1] = 1.0
    return np.linalg.eigvals(P)


def is_unilossless(A, trials: int = 32, max_delay: int = 7, tol: float = 1e-6, seed: int = 0) -> bool:
    """True when every FDN built on A stays lossless: all poles on the unit circle and none repeated, for random
    delay sets 1..max_delay (A = [[3, 2], [-4, -3]] is lossless with m = (1, 1) but gives a triple pole with (1, 2))."""
    A = np.asarray(A, float)
    rng = np.random.default_rng(seed)
    for t in range(trials):
        m = np.ones(A.shape[0], int) if t == 0 else rng.integers(1, max_delay + 1, A.shape[0])
        p = fdn_poles(A, m)
        if np.any(np.abs(np.abs(p) - 1.0) > tol) or np.min(np.abs(p[:, None] - p[None, :]) + 2 * np.eye(p.size)) < 1e-4:
            return False
    return True


@numba.njit(cache=True)
def _apdist_kernel(x, c):
    y = np.empty_like(x)
    x1 = y1 = 0.0
    for n in range(x.shape[0]):
        y[n] = c[n] * x[n] + x1 - c[n] * y1
        x1, y1 = x[n], y[n]
    return y


def apdist(x, sr=DEFAULT_SR, drive=2.0, smooth=500.0):
    """Signal-dependent all-pass (Pd e_apdist): first-order all-pass whose coefficient is the input low-passed at
    ``smooth`` Hz times ``drive``, clipped to +-0.9999. Phase distortion without a change of level."""
    def one(s):
        c = np.clip(F.onepole(s, "lowpass", smooth, sr) * float(drive), -0.9999, 0.9999)
        return _apdist_kernel(np.ascontiguousarray(s, dtype=np.float64), np.ascontiguousarray(c))
    return F._per_channel(one, np.asarray(x, float))


@numba.njit(cache=True)
def _wg_osc_kernel(f, sr, r):
    y = np.empty(f.shape[0])
    x1, x2, gp = 1.0, 0.0, -1.0
    for n in range(f.shape[0]):
        C = math.cos(2.0 * math.pi * min(max(f[n], 0.1), 0.49 * sr) / sr)
        g = math.sqrt((1.0 - C) / (1.0 + C))
        if gp > 0.0:
            x2 *= r[n] * gp / g
        gp = g
        t = C * (x1 + x2)
        x1, x2 = t - x2, t + x1
        y[n] = x1
    return y


def wg_osc(freq, n: int, sr=DEFAULT_SR, tau=0.0):
    """Second-order waveguide oscillator (C = cos(2 pi f T), x1, x2 <- C (x1 + x2) - x2, C (x1 + x2) + x1) whose
    amplitude stays exact under FM: x2 is rescaled by the ratio of g = sqrt((1 - C)/(1 + C)) between samples.
    ``tau`` > 0: exponential decay (s), r = exp(-T/tau) per sample. ``freq`` number or per-sample."""
    f = np.ascontiguousarray(np.broadcast_to(np.asarray(freq, np.float64), (int(n),)))
    r = np.full(int(n), math.exp(-1.0 / (tau * sr)) if tau > 0 else 1.0)
    return _wg_osc_kernel(f, float(sr), r)


def wdf_port(kind: str, value: float, sr=DEFAULT_SR):
    """Port resistance and reflection of a wave-digital element, force-velocity analogy, bilinear c = 2/T: mass m
    (R = 2 m / T, b = -a z^-1), spring k (R = k T / 2, b = a z^-1), damper r (R = r, b = 0)."""
    T = 1.0 / sr
    return {"mass": (2.0 * value / T, -1.0), "spring": (value * T / 2.0, 1.0), "damper": (float(value), 0.0)}[kind]


def wdf_series(R):
    """Series adaptor coefficients beta_i = 2 R_i / sum R (b_i = a_i - beta_i sum a)."""
    R = np.asarray(R, float)
    return 2.0 * R / R.sum()


def wdf_parallel(G):
    """Parallel adaptor coefficients alpha_i = 2 G_i / sum G (b_i = alpha . a - a_i)."""
    G = np.asarray(G, float)
    return 2.0 * G / G.sum()


def wdf_oscillator(force, sr=DEFAULT_SR, m=0.01, k=4000.0, r=0.5):
    """Velocity of a mass-spring-damper driven by ``force`` (N per sample), as a wave digital filter: the three
    elements on one series adaptor, the force entering as the damper's source (b = F). Resonance sqrt(k/m)/(2 pi)."""
    Rm, sm = wdf_port("mass", m, sr)
    Rk, sk = wdf_port("spring", k, sr)
    Rr, _ = wdf_port("damper", r, sr)
    R = np.array([Rm, Rk, Rr])
    beta = wdf_series(R)
    fz = np.asarray(force, float)
    v = np.empty(fz.size)
    bm = bk = 0.0
    for n in range(fz.size):
        a = np.array([sm * bm, sk * bk, fz[n]])
        s = a.sum()
        v[n] = s / R.sum()
        b = a - beta * s
        bm, bk = b[0], b[1]
    return v


def variation_filter(x, sr=DEFAULT_SR, n=10, depth_db=8.0, seed=0):
    """A cascade of ``n`` peaking cuts at random frequencies (log 80 Hz..12 kHz), depths (0..``depth_db``) and Q
    (0.7..4): one random, colourless-on-average variation per seed, for repeated samples."""
    rng = np.random.default_rng(seed)
    y = np.asarray(x, float)
    for _ in range(int(n)):
        y = F.peak(y, float(np.exp(rng.uniform(np.log(80.0), np.log(min(12000.0, 0.45 * sr))))), sr,
                   q=float(rng.uniform(0.7, 4.0)), gain_db=-float(rng.uniform(0.0, depth_db)))
    return y


HEAD_RADIUS = 0.0875
HEAD_ALPHA_MIN, HEAD_THETA_MIN = 0.1, 150.0


def head_shadow(x, sr=DEFAULT_SR, azimuth=0.0, radius=HEAD_RADIUS, c=343.0):
    """Head-shadow filter of one ear, H = (1 + j alpha w / 2w0) / (1 + j w / 2w0), w0 = c / a,
    alpha = (1 + alpha_min/2) + (1 - alpha_min/2) cos(theta / theta_min 180 deg); ``azimuth`` theta in degrees from
    the ear's axis (0 = facing the source, 180 = opposite)."""
    w0 = c / radius
    th = abs(float(azimuth)) % 360.0
    th = 360.0 - th if th > 180.0 else th
    al = (1.0 + HEAD_ALPHA_MIN / 2.0) + (1.0 - HEAD_ALPHA_MIN / 2.0) * math.cos(math.radians(th / HEAD_THETA_MIN * 180.0))
    b, a = sg.bilinear([al / (2.0 * w0), 1.0], [1.0 / (2.0 * w0), 1.0], fs=sr)
    return sg.lfilter(b, a, np.asarray(x, float), axis=0)


DECORR_MAX_S = 0.020


def decorrelation_fir(sr=DEFAULT_SR, length=DECORR_MAX_S, seeds=8, target=0.0, seed=0, bands="erb"):
    """Flat-magnitude random-phase FIR (<= 20 ms) for decorrelation: several seeds are tried and the one whose
    correlation with its input is closest to ``target`` is kept (|rho| < 0.4 is hardly heard). ``bands`` erb: one
    random phase per critical band, interpolated (smoother for diffuse highs); bin: one per FFT bin."""
    n = int(min(length, DECORR_MAX_S) * sr)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    erb = 21.4 * np.log10(1.0 + 0.00437 * f)
    test = np.random.default_rng(seed + 9999).normal(size=sr // 2)
    best, best_d = None, np.inf
    for s in range(int(seeds)):
        rng = np.random.default_rng(seed + s)
        if bands == "erb":
            ph = np.interp(erb, np.arange(int(erb[-1]) + 2), rng.uniform(-np.pi, np.pi, int(erb[-1]) + 2))
        else:
            ph = rng.uniform(-np.pi, np.pi, f.size)
        ph[0] = 0.0
        h = np.fft.irfft(np.exp(1j * ph), n) * np.hanning(n)
        h /= np.sqrt(np.sum(h * h))
        y = sg.fftconvolve(test, h)[:test.size]
        d = abs(float(np.corrcoef(test, y)[0, 1]) - target)
        if d < best_d:
            best, best_d = h, d
    return best


@numba.njit(cache=True)
def _dyn_ap_kernel(x, c0, dep, ph, rate, sr):
    y = x.copy()
    K = c0.shape[0]
    s = np.zeros(K)
    for n in range(x.shape[0]):
        u = x[n]
        t = n / sr
        for k in range(K):
            c = c0[k] + dep[k] * math.sin(2.0 * math.pi * rate * t + ph[k])
            v = u - c * s[k]
            o = c * v + s[k]
            s[k] = v
            u = o
        y[n] = u
    return y


def dynamic_allpass(x, sr=DEFAULT_SR, order=200, rate=0.2, depth=0.1, seed=0):
    """Time-varying all-pass decorrelator: ``order`` first-order sections with random coefficients (|c| < 0.8), each
    swaying slowly (``rate`` Hz, ``depth``) with its own phase."""
    rng = np.random.default_rng(seed)
    c0 = rng.uniform(-0.8, 0.8, int(order))
    dep = np.full(int(order), float(depth))
    ph = rng.uniform(0, 2 * np.pi, int(order))
    return F._per_channel(lambda s: _dyn_ap_kernel(np.ascontiguousarray(s, dtype=np.float64), c0, dep, ph,
                                                   float(rate), float(sr)), np.asarray(x, float))


def median_comb(x, sr=DEFAULT_SR, freq=110.0, k=4, amount=1.0, transient=0.5):
    """Harmonic comb by the median of taps at 0, T, .., (k-1) T (T = 1/freq, k 3..8): out = x - eff (x - median),
    eff = amount (1 - transient onset), so a note's attack is not echoed."""
    x = np.asarray(x, float)
    T = max(int(round(sr / float(freq))), 1)
    k = int(np.clip(k, 3, 8))
    def one(s):
        taps = np.stack([np.concatenate((np.zeros(i * T), s[:s.size - i * T])) if i * T < s.size else np.zeros(s.size)
                         for i in range(k)])
        med = np.median(taps, axis=0)
        fast = F.onepole(np.abs(s), "lowpass", 200.0, sr)
        slow = F.onepole(np.abs(s), "lowpass", 5.0, sr)
        onset = np.clip(fast / (slow + 1e-9) - 1.0, 0.0, 1.0)
        eff = float(amount) * (1.0 - float(transient) * onset)
        return s - eff * (s - med)
    return F._per_channel(one, x)


MODAL_FFT_N = 1024


def modal_fft(freqs, decays, amps, dur, sr=DEFAULT_SR, bins=5, n=MODAL_FFT_N):
    """Modal synthesis in the frequency domain: each frame (FFT ``n``, hop n/2, sqrt-Hann on both sides) gets every
    mode as its ``bins`` nearest bins of the windowed sinusoid's spectrum (energy error 1 bin 53 %, 3 bins 5 %,
    5 bins 1 %), then inverse FFT and overlap-add. ``decays`` T60 s."""
    f, d, a = (np.atleast_1d(np.asarray(v, float)) for v in (freqs, decays, amps))
    hop = n // 2
    total = int(dur * sr)
    out = np.zeros(total + n)
    w = np.sqrt(np.hanning(n + 1)[:n])
    half = int(bins) // 2
    for start in range(0, total, hop):
        S = np.zeros(n // 2 + 1, complex)
        t0 = start / sr
        for fi, di, ai in zip(f, d, a):
            if fi >= sr / 2:
                continue
            amp = ai * 10 ** (-3.0 * t0 / di)
            if amp < 1e-6 * ai:
                continue
            c = fi * n / sr
            k0 = int(round(c))
            for k in range(max(k0 - half, 0), min(k0 + half, n // 2) + 1):
                S[k] += 0.5 * amp * np.exp(2j * np.pi * fi * t0) * _win_dft(w, k - c)
        frame = np.fft.irfft(S, n) * w
        out[start:start + n] += frame
    return out[:total]


def _win_dft(w, delta):
    n = w.size
    return np.sum(w * np.exp(-2j * np.pi * delta * np.arange(n) / n))
