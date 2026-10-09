"""genny.dsp: measurable predictions of the classical DSP toolbox.

Run:  uv run --with pytest pytest tests/test_dsp.py -q      (or: uv run python tests/test_dsp.py)
"""
import os
import tempfile

import numpy as np
from scipy import signal as sg

from genny import dsp as D
from genny import fx as FX
from genny import sfx as SFX
from genny.core import write_wav

SR = 44100
MEASURED = {}


def _db(v):
    return 20 * np.log10(np.maximum(np.abs(v), 1e-300))


def _resp(h, n=1 << 16):
    return np.fft.rfftfreq(n), np.abs(np.fft.rfft(h, n))


def _sine(f, dur=1.0, sr=SR, amp=1.0):
    return amp * np.sin(2 * np.pi * f * np.arange(int(dur * sr)) / sr)


def _snr(ref, y):
    return 10 * np.log10(np.sum(ref ** 2) / np.sum((ref - y) ** 2))


# ---------------------------------------------------------------- FIR
def test_windowed_sinc_stopband():
    fc, bw = 0.1, 0.02
    for window, need in (("blackman", 70.0), ("hamming", 50.0)):
        h = D.windowed_sinc(fc, bw, window)
        f, H = _resp(h)
        stop = _db(H[f >= fc + bw / 2]).max()
        ripple = np.abs(H[f <= fc - bw / 2] - 1).max()
        MEASURED[f"fir {window} taps/stopband dB/passband ripple"] = (h.shape[0], round(stop, 1), float(f"{ripple:.1e}"))
        assert stop <= -need, (window, stop)
        assert ripple < (3e-4 if window == "blackman" else 4e-3)
        assert abs(_db(H[np.argmin(np.abs(f - fc))]) + 6.02) < 0.1      # cutoff = 50 % amplitude
        assert abs(h.sum() - 1) < 1e-12 and np.allclose(h, h[::-1])      # unity DC gain, linear phase


def test_spectral_inversion_reversal_bandpass():
    lp = D.windowed_sinc(0.15, 0.02)
    f, Hlp = _resp(lp)
    _, Hhp = _resp(D.spectral_invert(lp))
    _, Hrev = _resp(D.spectral_reverse(lp))
    assert _db(Hhp[f < 0.14]).max() < -70 and abs(Hhp[f > 0.16] - 1).max() < 3e-4
    assert _db(Hrev[f < 0.34]).max() < -70 and abs(Hrev[f > 0.36] - 1).max() < 3e-4   # 0.15 -> 0.35
    # perfect reconstruction of the band split: LP + (delta - LP) = delayed delta
    s = lp + D.spectral_invert(lp)
    assert abs(s).sum() == 1.0 and s[(s.shape[0] - 1) // 2] == 1.0
    bp = D.fir_bandpass(1960, 2040, 10000, bw=50)      # Smith's Table 16-2 example
    fh, Hbp = _resp(bp)
    fh = fh * 10000
    assert abs(Hbp[np.argmin(np.abs(fh - 2000))] - 1) < 1e-3
    assert _db(Hbp[(fh < 1935) | (fh > 2065)]).max() < -70
    br = D.fir_bandstop(1960, 2040, 10000, bw=50)
    assert _db(np.abs(np.fft.rfft(br, 1 << 16))[np.argmin(np.abs(fh - 2000))]) < -70


def test_fir_from_curve_and_min_phase():
    pts_f = [100, 300, 1000, 4000, 12000]
    pts_g = [0.0, 6.0, 0.0, -12.0, -3.0]
    # resolution = window main lobe = 6 sr / taps (65 Hz for the default 93 ms kernel): a corner at 60 Hz
    # needs a longer kernel (0.51 dB off with the default, inside 0.5 dB with 4x the taps)
    h = D.fir_from_curve([60, 250, 1000], [0.0, 6.0, 0.0], SR, taps=16385)
    H = np.abs(np.fft.rfft(h, 1 << 18))
    assert abs(_db(H[np.argmin(np.abs(np.fft.rfftfreq(1 << 18, 1 / SR) - 60))])) < 0.5
    worst = {}
    for phase in ("linear", "min"):
        h = D.fir_from_curve(pts_f, pts_g, SR, phase=phase)
        H = np.abs(np.fft.rfft(h, 1 << 18))
        f = np.fft.rfftfreq(1 << 18, 1 / SR)
        err = [abs(_db(H[np.argmin(np.abs(f - a))]) - b) for a, b in zip(pts_f, pts_g)]
        worst[phase] = max(err)
        assert max(err) < 0.5, (phase, err)
    MEASURED["fir_from_curve worst error dB (linear, min)"] = (round(worst["linear"], 3), round(worst["min"], 3))
    lin = D.fir_from_curve(pts_f, pts_g, SR)
    mn = D.fir_from_curve(pts_f, pts_g, SR, phase="min")
    assert np.argmax(np.abs(lin)) == (lin.shape[0] - 1) // 2          # linear phase: peak at M/2
    assert np.argmax(np.abs(mn)) < 8                                  # minimum phase: peak at the start
    # minimum phase packs the energy earliest: partial energy sums dominate the linear-phase ones
    k = lin.shape[0] // 8
    assert np.sum(mn[:k] ** 2) / np.sum(mn ** 2) > 0.99 > np.sum(lin[:k] ** 2) / np.sum(lin ** 2)
    # min_phase() of an existing kernel keeps the magnitude
    lp = D.windowed_sinc(0.1, 0.05, "hamming")
    mp = D.min_phase(lp)
    f, a = _resp(lp)
    _, b = _resp(mp)
    assert abs(a[f < 0.07] - b[f < 0.07]).max() < 2e-3
    # delay compensation: a linear-phase filter leaves an in-band sine exactly in place
    x = _sine(1000, 0.2)
    y = D.fir_filter(x, D.fir_lowpass(5000, SR))
    assert y.shape == x.shape and np.abs(y - x)[2000:-2000].max() < 1e-3


# ---------------------------------------------------------------- IIR
def test_butterworth_matches_scipy_and_slope():
    worst = 0.0
    for order in (2, 4, 6, 8, 12):
        for fc in (0.01, 0.1, 0.25, 0.4):
            for kind in ("lowpass", "highpass"):
                mine = D.smith_sos(fc, order, kind)
                ref = sg.butter(order, 2 * fc, btype=kind, output="sos")
                w, h1 = sg.sosfreqz(mine, 4096)
                _, h2 = sg.sosfreqz(ref, 4096)
                worst = max(worst, np.abs(h1 - h2).max())
                a, b = D.smith_coefficients(mine)
                bb, aa = sg.sos2tf(ref)
                if order <= 6 and fc >= 0.1:   # combined polynomials are only trustworthy at low order
                    assert np.allclose(a, bb, atol=1e-6) and np.allclose(-b, aa[1:], atol=1e-6)
    MEASURED["butterworth vs scipy max |dH|"] = float(f"{worst:.1e}")
    assert worst < 1e-6
    for order in (2, 4, 8):
        sos = D.iir_sos("lowpass", 1000, SR, order, "butter")
        _, h = sg.sosfreqz(sos, worN=[1000.0, 8000.0, 16000.0], fs=SR)
        assert abs(_db(h[0]) + 3.0103) < 1e-6                         # -3 dB at fc
        # asymptotic slope, measured on the bilinear-warped frequency axis: -6N dB per octave
        warp = np.tan(np.pi * np.array([8000.0, 16000.0]) / SR)
        slope = (_db(h[2]) - _db(h[1])) / np.log2(warp[1] / warp[0])
        MEASURED[f"butter order {order} slope dB/oct"] = round(float(slope), 2)
        assert abs(slope + 6.02 * order) < 0.1 * order
    # stage Qs of the 4th-order Butterworth: 0.5412 and 1.3066 (not two Q = 0.707 biquads)
    sos = D.smith_sos(0.01, 4)
    # bilinear biquad 1 + a1 z^-1 + a2 z^-2 with K = tan(pi fc):  Q = K (1 - a1 + a2) / (2 (1 - a2))
    qs = sorted(float(np.sqrt((1 + r[4] + r[5]) / (1 - r[4] + r[5])) * (1 - r[4] + r[5]) / (2 * (1 - r[5]))) for r in sos)
    MEASURED["butter 4 stage Q"] = [round(q, 4) for q in qs]
    assert np.allclose(qs, [0.5412, 1.3066], atol=2e-4)


def test_chebyshev_equals_smith_tables():
    # DSPG Table 20-1 / 20-2 rows quoted in g11 (0.5 % ripple) and the Table 20-6 debug data
    def row(fc, kind, npole):
        return D.smith_coefficients(D.smith_sos(fc, npole, kind, 0.5))
    rel = 2e-6
    a, b = row(0.10, "lowpass", 2)
    assert np.allclose(a, [6.372802e-2, 1.274560e-1, 6.372802e-2], rtol=rel) and np.allclose(b, [1.194365, -4.492774e-1], rtol=rel)
    a, b = row(0.25, "lowpass", 2)
    assert np.allclose(a, [2.858110e-1, 5.716221e-1, 2.858110e-1], rtol=rel) and np.allclose(b, [5.423258e-2, -1.974768e-1], rtol=rel)
    a, b = row(0.01, "lowpass", 2)
    assert np.allclose(a, [8.663387e-4, 1.732678e-3, 8.663387e-4], rtol=rel) and np.allclose(b, [1.919129, -9.225943e-1], rtol=rel)
    a, b = row(0.10, "lowpass", 4)
    assert np.allclose(a, [2.780755e-3, 1.112302e-2, 1.668453e-2, 1.112302e-2, 2.780755e-3], rtol=rel)
    assert np.allclose(b, [2.764031, -3.122854, 1.664554, -3.502232e-1], rtol=rel)
    # g11 labels the next row "HP 2 pole fc=0.10" and the value after it "fc=0.4 4-pole"; the program
    # reproduces both numbers to 7 digits at the mirror cutoffs (HP at fc = LP at 0.5 - fc with a1, b1
    # negated), i.e. HP 2-pole fc = 0.40 and 4-pole LP fc = 0.30 = HP fc = 0.20: the labels were slips.
    a, b = row(0.40, "highpass", 2)
    assert np.allclose(a, [6.372801e-2, -1.274560e-1, 6.372801e-2], rtol=rel) and np.allclose(b, [-1.194365, -4.492774e-1], rtol=rel)
    assert abs(row(0.3, "lowpass", 4)[0][0] - 1.335566e-1) < 1e-6 and abs(row(0.2, "highpass", 4)[0][0] - 1.335566e-1) < 1e-6
    # Table 20-6 debug sets (unnormalised stage output of the subroutine)
    s1 = D.smith_stage(0.1, False, 0, 4, 1)
    assert np.allclose(s1, [0.061885, 0.123770, 0.061885, 1.048600, -0.296140], atol=1e-6), s1
    s2 = D.smith_stage(0.1, True, 10, 4, 2)
    assert np.allclose(s2, [0.922919, -1.845840, 0.922919, 1.446913, -0.836653], atol=1e-6), s2
    # all designs stable up to 20 poles in sos form, also where Table 20-3 says single precision fails
    for fam in ("butter", "cheby", "bessel"):
        for order in (2, 8, 20):
            for fc in (30.0, 1000.0, 15000.0):
                for kind in ("lowpass", "highpass"):
                    sos = D.iir_sos(kind, fc, SR, order, fam, 0.5)
                    assert all(np.abs(np.roots(r[3:])).max() < 1 for r in sos)
    # Bessel: -3 dB at fc, step overshoot under 1 % where Butterworth 8 overshoots ~16 % (DSPG Table 3-2)
    step = np.ones(4000)
    ob = sg.sosfilt(D.iir_sos("lowpass", 1000, SR, 8, "bessel"), step).max() - 1
    obw = sg.sosfilt(D.iir_sos("lowpass", 1000, SR, 8, "butter"), step).max() - 1
    MEASURED["8-pole step overshoot bessel / butter"] = (round(float(ob), 4), round(float(obw), 4))
    assert ob < 0.01 and 0.14 < obw < 0.18


# ---------------------------------------------------------------- convolution, impulse responses
def test_convolution():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(5000)
    assert np.array_equal(D.fft_convolve(x, np.array([1.0])), x)                 # delta is the identity
    d = np.zeros(300)
    d[7] = 1.0
    y = D.fft_convolve(x, d)                                                      # FFT path, shifted delta
    assert y.shape[0] == 5000 + 300 - 1 and np.abs(y[7:5007] - x).max() < 1e-12
    h = rng.standard_normal(700)
    assert np.abs(D.fft_convolve(x, h) - np.convolve(x, h)).max() < 1e-9          # equals direct convolution
    st = np.stack([x, -x], axis=1)
    assert D.fft_convolve(st, h).shape == (5699, 2)
    # the fx with a delta wav returns x (plus the zero tail), mono and stereo
    path = os.path.join(tempfile.mkdtemp(), "delta.wav")
    write_wav(path, np.concatenate([[1.0], np.zeros(63)]), SR, bits=24)
    x = 0.5 * x / np.abs(x).max()
    out = D.convolve(x, SR, ir=path)
    assert out.shape[0] == 5000 + 63 and np.abs(out[:5000] - x).max() < 1e-5 and np.abs(out[5000:]).max() < 1e-5
    out = D.convolve(x, SR, kind="room", mix=0.0, predelay=0.01)
    assert np.array_equal(out[:5000], x) and out.shape[0] > 5000 + 441


def _rt60(h, sr, lo, hi):
    """Schroeder backward integration, T30-style fit (-5..-25 dB) extrapolated to 60 dB."""
    b = sg.sosfiltfilt(sg.butter(4, [lo, hi], btype="band", fs=sr, output="sos"), h)
    e = np.cumsum(b[::-1] ** 2)[::-1]
    e = 10 * np.log10(e / e[0])
    i0, i1 = np.argmax(e < -5), np.argmax(e < -25)
    return 60.0 * (i1 - i0) / sr / 20.0


def test_impulse_responses():
    # the lowest octave is only ~80 Hz wide, so one realisation's decay fit scatters: average 8 seeds
    hs = [D.ir_noise(SR, rt60=1.2, rt60_high=0.4, seed=s) for s in range(8)]
    low, high = np.mean([_rt60(h, SR, 90, 170) for h in hs]), np.mean([_rt60(h, SR, 6000, 12000) for h in hs])
    MEASURED["ir_noise RT60 requested 1.2/0.4 s, measured"] = (round(low, 3), round(high, 3))
    assert abs(low - 1.2) < 0.18 and abs(high - 0.4) < 0.06
    # chirp system: unit magnitude, low frequencies first, and chirp * anti-chirp = impulse
    n = 4096
    c = D.chirp_allpass(n, 200, 1800)
    assert np.abs(np.abs(np.fft.rfft(c)) - 1).max() < 1e-9
    early, late = np.fft.rfft(c[150:700] * np.hanning(550), n), np.fft.rfft(c[1300:1850] * np.hanning(550), n)
    f = np.fft.rfftfreq(n)
    cent = [np.sum(f * np.abs(s) ** 2) / np.sum(np.abs(s) ** 2) for s in (early, late)]
    MEASURED["chirp centroid early/late (fs)"] = (round(cent[0], 3), round(cent[1], 3))
    assert cent[1] > 3 * cent[0]
    anti = np.roll(c[::-1], 1)                       # conjugate spectrum = time reversal
    imp = np.fft.irfft(np.fft.rfft(c) * np.fft.rfft(anti))
    assert abs(imp[0] - 1) < 1e-9 and np.abs(imp[1:]).max() < 1e-9
    # spring: RT60 as requested at 1 kHz, and each echo a rising chirp (group delay grows with frequency)
    s = D.ir_spring(SR, decay=1.5, size=0.5, bright=0.5)
    rt = _rt60(s, SR, 700, 1400)
    MEASURED["spring RT60 requested 1.5 s, measured"] = round(rt, 3)
    assert abs(rt - 1.5) < 0.3
    # tube: lowest resonance at c/(4L)
    t = D.ir_tube(SR, decay=0.4, size=0.3)
    L = 0.1 + 1.9 * 0.3
    spec = np.abs(np.fft.rfft(t, 1 << 18))
    fpk = np.fft.rfftfreq(1 << 18, 1 / SR)[np.argmax(spec)]
    MEASURED["tube first mode predicted/measured Hz"] = (round(343 / (4 * L), 2), round(float(fpk), 2))
    assert abs(fpk - 343 / (4 * L)) < 1.0
    # cabinet: minimum phase (energy at the start) and the drawn roll-off
    cab = D.make_ir("cabinet", SR)
    assert np.argmax(np.abs(cab)) < 16
    for kind in D.IR_KINDS:
        for sr in (22050, 48000):
            g = D.make_ir(kind, sr, seed=1)
            assert np.all(np.isfinite(g)) and g.shape[0] < 6 * sr and np.abs(g).max() > 1e-4, kind
            assert np.array_equal(g, D.make_ir(kind, sr, seed=1))


# ---------------------------------------------------------------- resampling
def test_resample_sinc():
    x = _sine(440, 1.0, 44100)
    y = D.resample_sinc(x, 44100, 48000)
    assert y.shape[0] == 48000
    seg = y[4800:-4800] * sg.windows.blackmanharris(38400)
    S = np.abs(np.fft.rfft(seg))
    f = np.fft.rfftfreq(38400, 1 / 48000)
    k = np.argmax(S)
    d = np.log(S[k + 1] / S[k - 1]) / 2 / np.log(S[k] ** 2 / (S[k + 1] * S[k - 1]))   # Gaussian peak fit
    fpk = (k + d) * 48000 / 38400
    spur = _db(S[np.abs(f - 440) > 20].max() / S[k])
    MEASURED["resample 44.1k->48k: peak Hz / worst spur dB"] = (round(float(fpk), 4), round(float(spur), 1))
    assert abs(fpk - 440) < 0.01 and spur < -70
    assert abs(np.sqrt(np.mean(y[4800:-4800] ** 2)) - np.sqrt(0.5)) < 1e-4            # unity gain
    # down-conversion must not alias: 15 kHz is above the 8 kHz Nyquist and has to vanish
    z = D.resample_sinc(_sine(15000, 0.5) + _sine(1000, 0.5), 44100, 16000)
    Z = np.abs(np.fft.rfft(z[1600:-1600] * np.hanning(z.shape[0] - 3200)))
    fz = np.fft.rfftfreq(z.shape[0] - 3200, 1 / 16000)
    alias = _db(Z[np.abs(fz - 1000) > 100].max() / Z.max())
    lin = np.interp(np.arange(8000) / 16000, np.arange(22050) / 44100, _sine(15000, 0.5) + _sine(1000, 0.5))
    L = np.abs(np.fft.rfft(lin[1600:-1600] * np.hanning(4800)))
    fl = np.fft.rfftfreq(4800, 1 / 16000)
    MEASURED["decimation alias dB: sinc / linear interp"] = (round(float(alias), 1), round(float(_db(L[np.abs(fl - 1000) > 100].max() / L.max())), 1))
    assert alias < -70
    # variable-rate read: constant rate 1.5 turns 440 Hz into 660 Hz; a ramp glides upward
    v = D.varispeed(_sine(440, 1.0), 1.5)
    V = np.abs(np.fft.rfft(v[2000:-2000] * np.hanning(v.shape[0] - 4000)))
    fv = np.fft.rfftfreq(v.shape[0] - 4000, 1 / SR)[np.argmax(V)]
    MEASURED["varispeed x1.5 of 440 Hz"] = round(float(fv), 2)
    assert abs(fv - 660) < 1.0 and abs(v.shape[0] - 29400) <= 1
    # rate 2.5 on an 8 kHz tone would land at 20 kHz... and on 10 kHz at 25 kHz: must be filtered, not aliased
    a = D.varispeed(_sine(10000, 0.5), 2.5)
    assert np.sqrt(np.mean(a[500:-500] ** 2)) < 0.02


# ---------------------------------------------------------------- companding / dither / codecs
def test_mulaw_alaw():
    x = np.linspace(-1, 1, 20001)
    assert np.abs(D.mulaw_expand(D.mulaw_compress(x)) - x).max() < 1e-12
    assert np.abs(D.alaw_expand(D.alaw_compress(x)) - x).max() < 1e-12
    assert abs(D.mulaw_compress(np.array([1.0]))[0] - 1) < 1e-12 and abs(D.alaw_compress(np.array([1.0]))[0] - 1) < 1e-12
    y = D.mulaw(x, SR, bits=8)
    bound = 0.5 / 127 * np.log(256) * 256 / 255        # half a step times the steepest expander slope
    MEASURED["mu-law 8-bit max error / bound"] = (round(float(np.abs(y - x).max()), 5), round(bound, 5))
    assert np.abs(y - x).max() <= bound * 1.0001
    expected = 6.02 * 8 + 4.77 - 20 * np.log10(np.log(256))      # ~38 dB, independent of level
    snrs = [_snr(_sine(997, 1, amp=a), D.mulaw(_sine(997, 1, amp=a), SR)) for a in (1.0, 0.1, 0.03)]
    lin = [_snr(_sine(997, 1, amp=a), np.round(_sine(997, 1, amp=a) * 127) / 127) for a in (1.0, 0.03)]
    MEASURED["mu-law SNR dB at 0/-20/-30 dBFS (expected ~%.1f); linear 8-bit at 0/-30" % expected] = ([round(s, 1) for s in snrs], [round(s, 1) for s in lin])
    assert abs(snrs[0] - expected) < 2.5 and snrs[0] - snrs[2] < 6 and snrs[2] > lin[1] + 8
    a = [_snr(_sine(997, 1, amp=v), D.mulaw(_sine(997, 1, amp=v), SR, law="a")) for v in (1.0, 0.1)]
    assert a[0] > 35 and a[1] > 33


def test_dither_and_noise_shaping():
    # Smith's example: a value between two levels is lost by rounding, recovered on average with dither
    x = np.full(200000, 0.3 / 128)
    assert np.all(D.quantize(x, 8, "none") == 0.0)
    for kind in ("tpdf", "gauss", "shaped"):
        m = D.quantize(x, 8, kind, seed=1).mean() * 128
        assert abs(m - 0.3) < 0.01, (kind, m)
    # quantisation noise sd = LSB/sqrt(12); TPDF dither adds 2 more of those -> 3x the power
    s = _sine(997, 2.0, amp=0.5)
    e0 = D.quantize(s, 8, "none") - s
    e1 = D.quantize(s, 8, "tpdf", 2) - s
    lsb = 1 / 128
    MEASURED["quant noise sd / (LSB/sqrt12): none, tpdf"] = (round(float(e0.std() / (lsb / 12 ** 0.5)), 3), round(float(e1.std() / (lsb / 12 ** 0.5)), 3))
    assert abs(e0.std() / (lsb / 12 ** 0.5) - 1) < 0.05 and abs(e1.std() / (lsb / 12 ** 0.5) - 3 ** 0.5) < 0.05
    # first-order shaping moves the noise up: less below fs/8 than plain TPDF, +6 dB/oct tilt
    e2 = D.quantize(s, 8, "shaped", 2) - s
    f, p1 = sg.welch(e1, SR, nperseg=4096)
    _, p2 = sg.welch(e2, SR, nperseg=4096)
    lo = 10 * np.log10(p2[f < SR / 16].mean() / p1[f < SR / 16].mean())
    hi = 10 * np.log10(p2[f > SR * 3 / 8].mean() / p1[f > SR * 3 / 8].mean())
    MEASURED["noise shaping dB vs tpdf: below fs/16, above 3fs/8"] = (round(float(lo), 1), round(float(hi), 1))
    assert lo < -10 and hi > 3
    assert np.array_equal(D.dither(s, SR, 6, "shaped", 5), D.dither(s, SR, 6, "shaped", 5))


def test_delta_cvsd_adpcm_dct_gsm():
    bits, rec = D.delta_encode(np.full(64, 0.0), 0.01)
    assert np.all(np.abs(np.diff(bits.astype(int))) == 1)                # constant input -> 1,0,1,0
    x = _sine(300, 0.5, 32000, 0.5)
    bits, rec = D.cvsd_encode(x, 32000)
    assert np.array_equal(D.cvsd_decode(bits, 32000), rec)                # decoder = identical logic
    b2, r2 = D.delta_encode(x, 0.02)
    assert np.array_equal(D.delta_decode(b2, 0.02), r2)
    lp = sg.butter(6, 3400, fs=32000, output="sos")
    s_c = _snr(sg.sosfiltfilt(lp, x)[2000:], sg.sosfiltfilt(lp, rec)[2000:])
    # slew limit: a delta modulator with step*clock below the signal slope cannot follow
    slow = D.delta_encode(x, 0.5 * 2 * np.pi * 300 * 0.5 / 32000)[1]
    ok = D.delta_encode(x, 2.0 * 2 * np.pi * 300 * 0.5 / 32000)[1]
    s_slow, s_ok = _snr(x[2000:], slow[2000:]), _snr(x[2000:], ok[2000:])
    MEASURED["300 Hz @32k SNR dB: cvsd, delta ok, delta slew-limited"] = (round(s_c, 1), round(s_ok, 1), round(s_slow, 1))
    assert s_c > 15 and s_ok > 12 and s_ok > s_slow + 6
    # 4-bit IMA ADPCM round trip
    x = _sine(440, 1.0, SR, 0.7)
    codes, rec = D.adpcm_encode(x)
    assert codes.max() <= 15 and np.array_equal(D.adpcm_decode(codes), rec)
    s_a = _snr(x[500:], rec[500:])
    MEASURED["ADPCM 4-bit SNR dB, 440 Hz @44.1k"] = round(s_a, 1)
    assert s_a > 15
    # block DCT: coarser table = lower SNR and the top of the spectrum removed
    rng = np.random.default_rng(0)
    w = 0.2 * rng.standard_normal(SR)
    fine, coarse = D.block_dct(w, 512, 4, 2, 9), D.block_dct(w, 512, 128, 2, 5)
    hi = sg.butter(6, 15000, btype="high", fs=SR, output="sos")
    kept = np.std(sg.sosfilt(hi, coarse)) / np.std(sg.sosfilt(hi, w))
    MEASURED["dct SNR dB fine/coarse, >15 kHz kept by coarse"] = (round(_snr(w, fine), 1), round(_snr(w, coarse), 1), round(float(kept), 3))
    assert _snr(w, fine) > _snr(w, coarse) + 10 and kept < 0.2
    # GSM-like: a voiced-speech-like signal keeps its formant peak
    n = 8000
    src = np.zeros(n)
    src[::80] = 1.0                                                       # 100 Hz pulses
    b, a = sg.iirpeak(700, 6, fs=8000)
    v = sg.lfilter(b, a, src)
    v = 0.5 * v / np.abs(v).max()
    g = D.gsm_like(v)
    f, pv = sg.welch(v, 8000, nperseg=1024)
    _, pg = sg.welch(g, 8000, nperseg=1024)
    MEASURED["gsm-like: formant peak in/out Hz, level ratio dB"] = (float(f[np.argmax(pv)]), float(f[np.argmax(pg)]), round(float(10 * np.log10(pg.sum() / pv.sum())), 1))
    assert abs(f[np.argmax(pg)] - f[np.argmax(pv)]) <= 110 and abs(10 * np.log10(pg.sum() / pv.sum())) < 6


def test_zoh():
    assert abs(D.zoh_droop(4000, 8000) - 0.6366) < 1e-4                  # -3.9 dB at half the rate
    # a 3 kHz tone sampled at 8 kHz and held: the image at 5 kHz is there, weighted by the sinc
    y = D.zoh_resample(_sine(3000, 1.0, 8000), 8000, 48000)
    S = np.abs(np.fft.rfft(y[4800:-4800] * sg.windows.blackmanharris(38400)))
    f = np.fft.rfftfreq(38400, 1 / 48000)
    pk = lambda hz: S[np.abs(f - hz) < 30].max()
    r = pk(5000) / pk(3000)
    MEASURED["ZOH image 5k/3k amplitude: measured / sinc ratio"] = (round(float(r), 4), round(float(D.zoh_droop(5000, 8000) / D.zoh_droop(3000, 8000)), 4))
    assert abs(r - D.zoh_droop(5000, 8000) / D.zoh_droop(3000, 8000)) < 0.02
    # Smith's 1/sinc corrector flattens the droop to 0.45 fs
    w, h = sg.freqz([1.151692, -0.06794763, -0.07929603], [1.0, 0.1129629, -0.1062142], worN=np.linspace(0.01, 0.45, 50) * 2 * np.pi)
    assert np.abs(np.abs(h) * D.zoh_droop(w / (2 * np.pi), 1.0) - 1).max() < 0.02


# ---------------------------------------------------------------- nonlinearity
def _nonharmonic_db(y, f0, sr):
    seg = y[sr // 4: sr // 4 + sr // 2] * sg.windows.blackmanharris(sr // 2)
    S = np.abs(np.fft.rfft(seg)) ** 2
    f = np.fft.rfftfreq(sr // 2, 1 / sr)
    harm = np.abs(f - f0 * np.round(f / f0)) < 20
    harm |= f < 40
    return 10 * np.log10(S[~harm].sum() / S[harm].sum())


def test_overdrive_aliasing_and_harmonics():
    x = _sine(5000, 1.0)
    naive = D.overdrive(x, SR, drive=4.0, tone=0, oversample=1)
    clean = D.overdrive(x, SR, drive=4.0, tone=0, oversample=4)
    a, b = _nonharmonic_db(naive, 5000, SR), _nonharmonic_db(clean, 5000, SR)
    MEASURED["overdrive 5 kHz non-harmonic energy dB: naive / 4x"] = (round(float(a), 1), round(float(b), 1))
    assert b < a - 40
    assert np.abs(clean).max() < 1.3
    # symmetric curve -> odd harmonics only; bias -> even harmonics appear (DSPG ch.11)
    x = _sine(200, 1.0, amp=0.8)

    def h2_h3(y):
        S = np.abs(np.fft.rfft(y[SR // 4: SR // 4 + SR // 2] * np.hanning(SR // 2)))
        return _db(S[200] / S[100]), _db(S[300] / S[100])      # bins are 2 Hz apart
    sym, asym = h2_h3(D.overdrive(x, SR, drive=3, tone=0)), h2_h3(D.overdrive(x, SR, drive=3, bias=0.5, tone=0))
    MEASURED["2nd / 3rd harmonic dB: symmetric, biased"] = ([round(float(v), 1) for v in sym], [round(float(v), 1) for v in asym])
    assert sym[0] < -80 and sym[1] > -30 and asym[0] > -30
    assert abs(np.mean(D.overdrive(x, SR, drive=3, bias=0.5)[SR // 2:])) < 1e-3     # DC removed


# ---------------------------------------------------------------- noise, metering
def test_noise_models():
    g = D.gaussian(400000, 1, bounded=True)
    assert abs(g.mean()) < 0.01 and abs(g.std() - 1) < 0.01 and np.abs(g).max() <= 6.0
    assert abs(np.mean(np.abs(g) < 1) - 0.6827) < 0.005 and abs(D.gaussian(400000, 2).std() - 1) < 0.01
    # Poisson: count variance = mean; a dense stream's RMS grows as sqrt(rate) (3 dB per doubling)
    ev = D.poisson_events(SR * 20, 200.0, SR, 3)
    counts = ev.reshape(-1, SR // 10).sum(axis=1)
    MEASURED["poisson 200/s: mean and variance per 0.1 s"] = (round(float(counts.mean()), 2), round(float(counts.var()), 2))
    assert abs(counts.mean() - 20) < 1 and abs(counts.var() / counts.mean() - 1) < 0.3
    r1 = D.noise_bed(SR, "static_field", dur=4, rate=2000, seed=1)
    r2 = D.noise_bed(SR, "static_field", dur=4, rate=8000, seed=1)
    step = 20 * np.log10(np.std(r2) / np.std(r1))
    MEASURED["static_field level change for 4x rate (expect 6.0 dB)"] = round(float(step), 2)
    assert abs(step - 6.02) < 1.5
    ln = D.lognormal(200000, 0.5, 4)
    assert abs(np.median(ln) - 1) < 0.01 and abs(np.log(ln).std() - 0.5) < 0.01 and ln.min() > 0
    # 1/f floor: +10 dB per decade below the knee, flat above
    f, p = sg.welch(D.one_over_f(SR * 20, SR, 100.0, 5), SR, nperseg=1 << 15)
    band = lambda a, b: 10 * np.log10(p[(f >= a) & (f < b)].mean())
    MEASURED["1/f floor dB re white: 5-10 Hz, 5-10 kHz"] = (round(float(band(5, 10) - band(5000, 10000)), 1), 0.0)
    assert abs(band(5, 10) - band(5000, 10000) - 10 * np.log10(1 + 100 / 7.2)) < 1.5
    # hum: the fundamental sits at the mains frequency, harmonics at its multiples
    f, p = sg.welch(D.noise_bed(SR, "hum", dur=4, freq=60), SR, nperseg=1 << 15)
    top = np.sort(f[np.argsort(p)[-12:]])
    assert abs(f[np.argmax(p)] - 60) < 1.5 and any(abs(top - 120) < 2) and any(abs(top - 180) < 2)


def test_metering():
    a1k, a100 = float(D.a_weighting_db(1000.0)), float(D.a_weighting_db(100.0))
    sos = D.a_weighting_sos(SR)
    _, h = sg.sosfreqz(sos, worN=[100.0, 1000.0], fs=SR)
    MEASURED["A-weighting dB at 1 kHz / 100 Hz: analog, digital"] = ((round(a1k, 3), round(a100, 2)), (round(float(_db(h[1])), 3), round(float(_db(h[0])), 2)))
    assert abs(a1k) < 0.05 and abs(a100 + 19.1) < 0.3
    assert abs(_db(h[1])) < 1e-9 and abs(_db(h[0]) + 19.1) < 0.3
    y = D.a_weight(_sine(100, 1.0), SR)
    assert abs(_db(np.sqrt(2 * np.mean(y[SR // 2:] ** 2))) + 19.1) < 0.3
    # VU: 99 % of the power step at 300 ms, 63 % at 65.14 ms
    v = D.vu_meter(np.ones(SR), SR) ** 2
    MEASURED["VU power step at 65.14 ms / 300 ms"] = (round(float(v[int(0.06514 * SR)]), 4), round(float(v[int(0.3 * SR)]), 4))
    assert abs(v[int(0.06514 * SR)] - (1 - np.exp(-1))) < 0.005 and abs(v[int(0.3 * SR)] - 0.99) < 0.002


# ---------------------------------------------------------------- effects: physical checks
def test_wall_tape_smooth_lofi_steep():
    # mass law: +6 dB per octave and per doubling of mass, capped by the leak; much more than muffle's 4 dB
    f = np.array([125.0, 250.0, 500.0])
    tl = D.wall_loss_db(f, 22.0, 90.0)
    assert np.allclose(np.diff(tl), 6.0206, atol=1e-3) and abs(D.wall_loss_db(250.0, 44.0, 90.0) - tl[1] - 6.0206) < 1e-3
    for material in D.WALL_MATERIALS:
        m = D.WALL_MATERIALS[material]
        got = []
        for hz in (250.0, 1000.0):
            y = D.wall(_sine(hz, 0.5), SR, material=material)
            got.append(-_db(np.sqrt(2 * np.mean(y[SR // 8:] ** 2))))
        want = [float(D.wall_loss_db(hz, m["mass"], m["leak"], m["coincidence"])) for hz in (250.0, 1000.0)]
        MEASURED[f"wall {material} loss dB at 250/1000 Hz: measured (model)"] = ([round(float(v), 1) for v in got], [round(v, 1) for v in want])
        assert np.allclose(got, want, atol=1.0), (material, got, want)
    # moving average: first null at fs/M, sqrt(M) noise reduction, zero delay
    m = 21
    w = np.random.default_rng(0).standard_normal(200000)
    assert abs(np.std(D.moving_average(w, m)) - 1 / np.sqrt(m)) < 0.01
    assert np.sqrt(np.mean(D.moving_average(_sine(SR / m, 1.0), m)[100:-100] ** 2)) < 1e-9
    assert np.allclose(D.moving_average(np.ones(500), m, 3), 1.0)                   # normalised edges
    imp = np.zeros(401)
    imp[200] = 1
    tri = D.moving_average(imp, 11, 2)
    assert np.argmax(tri) == 200 and np.allclose(tri, tri[::-1]) and abs(tri.sum() - 1) < 1e-12
    # tape: wow modulates pitch around the original with the requested depth
    y = D.tape(_sine(1000, 2.0, amp=0.3), SR, wow=1.0, flutter=0.0, drive=0.2, bump=0, hiss=-120, hf=0)
    ph = np.unwrap(np.angle(sg.hilbert(y)))
    inst = D.moving_average(np.diff(ph) * SR / (2 * np.pi), 441)[SR // 4: -SR // 4]
    MEASURED["tape wow 1 %: inst. frequency min/mean/max Hz"] = (round(float(inst.min()), 1), round(float(inst.mean()), 1), round(float(inst.max()), 1))
    assert abs(inst.max() - 1010) < 2 and abs(inst.min() - 990) < 2 and y.shape[0] == 2 * SR
    # lofi: telephone band (200-3200 Hz): 1 kHz passes, 6 kHz and 60 Hz do not
    lv = lambda hz: _db(np.sqrt(2 * np.mean(D.lofi(_sine(hz, 0.5, amp=0.5), SR, preset="telephone")[SR // 8:] ** 2)) / 0.5)
    MEASURED["lofi telephone gain dB at 60 / 1000 / 6000 Hz"] = (round(float(lv(60)), 1), round(float(lv(1000)), 1), round(float(lv(6000)), 1))
    assert abs(lv(1000)) < 1 and lv(6000) < -25 and lv(60) < -25
    # lofi hold: the 12 kHz image of a 4 kHz tone at 16 kHz is present with a hold, absent without
    def image(hold):
        y = D.lofi(_sine(4000, 0.5, 48000, 0.5), 48000, preset="none", rate=16000, bits=16, dither="none", hold=hold)
        S = np.abs(np.fft.rfft(y[2400:-2400] * np.hanning(y.shape[0] - 4800)))
        f = np.fft.rfftfreq(y.shape[0] - 4800, 1 / 48000)
        return _db(S[np.abs(f - 12000) < 50].max() / S[np.abs(f - 4000) < 50].max())
    MEASURED["lofi 16 kHz image at 12 kHz re 4 kHz dB: hold / clean (sinc predicts %.1f)" % _db(D.zoh_droop(12000, 16000) / D.zoh_droop(4000, 16000))] = (round(float(image(True)), 1), round(float(image(False)), 1))
    assert abs(image(True) - _db(D.zoh_droop(12000, 16000) / D.zoh_droop(4000, 16000))) < 1 and image(False) < -70
    # steep fx: order 8 Butterworth = -3 dB at cutoff, -48 dB one octave up (within the bilinear warp)
    g = lambda fn, hz, **kw: _db(np.sqrt(2 * np.mean(fn(_sine(hz, 0.5), SR, **kw)[SR // 4:] ** 2)))
    assert abs(g(D.steep_lowpass, 1000, cutoff=1000) + 3.01) < 0.05 and abs(g(D.steep_lowpass, 2000, cutoff=1000) + 48.2) < 1
    assert abs(g(D.steep_highpass, 1000, cutoff=1000) + 3.01) < 0.05 and abs(g(D.steep_highpass, 500, cutoff=1000) + 48.2) < 1
    assert abs(g(D.steep_bandpass, 1000, low=300, high=3400)) < 0.1 and g(D.steep_bandpass, 100, low=300, high=3400) < -70
    # fir_eq hits its points through the registered effect
    assert abs(g(D.fir_eq, 1000, points="100:0,1000:-9,8000:0") + 9) < 0.5
    assert abs(g(D.fir_eq, 1000, points="100:0,1000:-9,8000:0", phase="min") + 9) < 0.5


# ---------------------------------------------------------------- smoke: every registered name
FX_NAMES = ["convolve", "fir_eq", "steep_lowpass", "steep_highpass", "steep_bandpass", "overdrive", "wavefold", "mulaw",
            "codec", "dither", "lofi", "smooth", "tape", "wall", "dc_block"]
VARIANTS = {
    "convolve": [dict(kind=k) for k in D.IR_KINDS] + [dict(kind="hall", decay=6.0, size=1.0, bright=1.0, predelay=0.05)],
    "fir_eq": [dict(), dict(phase="min", points="50:12,500:-30,5000:6")],
    "steep_lowpass": [dict(), dict(type="cheby", order=20, ripple=29, cutoff=30), dict(type="bessel", order=2, cutoff=30000)],
    "steep_highpass": [dict(), dict(type="cheby", order=12, cutoff=8000), dict(type="bessel")],
    "steep_bandpass": [dict(), dict(type="cheby", low=5000, high=100)],
    "overdrive": [dict(shape=s, bias=b) for s in ("tanh", "tube", "fuzz", "fold", "diode") for b in (0.0, 0.6)] + [dict(drive=100, oversample=8)],
    "wavefold": [dict(), dict(folds=20, symmetry=1, shape="tri"), dict(folds=0.1)],
    "mulaw": [dict(), dict(law="a", bits=4), dict(bits=2)],
    "codec": [dict(kind=k, quality=q) for k in ("adpcm", "cvsd", "delta", "gsm", "dct") for q in (0.0, 0.5, 1.0)],
    "dither": [dict(shape=s, bits=b) for s in ("none", "tpdf", "gauss", "shaped") for b in (2, 8, 16)],
    "lofi": [dict(preset=p) for p in list(D.LOFI_PRESETS) + ["none"]] + [dict(rate=4000, bits=3, law="a", hold=False), dict(rate=96000, bits=1)],
    "smooth": [dict(), dict(width=20, passes=4), dict(width=0)],
    "tape": [dict(), dict(wow=5, flutter=5, drive=10, bump=12, hiss=-30, hf=2000)],
    "wall": [dict(material=m) for m in D.WALL_MATERIALS] + [dict(material="door", thickness=8, leak=90, makeup=40)],
    "dc_block": [dict(), dict(cutoff=200)],
}


def _test_signal(sr):
    t = np.arange(int(0.6 * sr)) / sr
    x = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.25 * np.sin(2 * np.pi * 1870 * t) * np.exp(-4 * t)
    x += 0.1 * np.random.default_rng(0).standard_normal(t.shape[0])
    x *= np.minimum(1, np.minimum(t, t[-1] - t) / 0.01)
    return x * 0.7


def test_smoke_all_registered():
    for name in FX_NAMES:
        assert name in FX.REGISTRY, name
        fn, params = FX.REGISTRY[name]["fn"], FX.REGISTRY[name]["params"]
        assert all(isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], str) for v in params.values())
        for sr in (22050, 48000):
            x = _test_signal(sr)
            st = np.stack([x, 0.8 * x[::-1]], axis=1)
            for kw in VARIANTS[name]:
                assert set(kw) <= set(params), (name, kw)
                y = fn(x, sr, **kw)
                ys = fn(st, sr, **kw)
                for out in (y, ys):
                    pk = np.abs(out).max()
                    assert np.all(np.isfinite(out)) and 1e-4 < pk < 1.5, (name, sr, kw, pk)
                assert y.ndim == 1 and ys.ndim == 2 and ys.shape[1] == 2 and ys.shape[0] == y.shape[0]
                assert y.shape[0] >= x.shape[0] if name == "convolve" else y.shape[0] == x.shape[0], (name, kw)
                assert np.array_equal(y, fn(x, sr, **kw)), (name, kw)        # deterministic
    assert "noise_bed" in SFX.REGISTRY
    fn = SFX.REGISTRY["noise_bed"]["fn"]
    for sr in (22050, 44100, 48000):
        for kind in ("hum", "hiss", "rumble", "crackle", "geiger", "static_field"):
            for kw in (dict(), dict(dur=0.05, rate=0.5, level=-80), dict(dur=1.0, rate=50000, level=0, bright=1.0, freq=60, seed=3)):
                y = fn(sr, kind=kind, **kw)
                pk = np.abs(y).max()
                assert y.ndim == 1 and np.all(np.isfinite(y)) and 1e-4 * (0.02 if kw.get("level") == -80 else 1) < pk < 1.5, (kind, kw, pk)
                assert y[0] == 0 and y[-1] == 0 and abs(y.mean()) < 0.02 * max(pk, 1e-9) + 1e-6, (kind, kw, y.mean())
                assert np.array_equal(y, fn(sr, kind=kind, **kw))
    # not piercing by default: the default bed keeps its energy below 5 kHz
    f, p = sg.welch(fn(SR), SR, nperseg=4096)
    assert p[f > 5000].sum() / p.sum() < 0.03


def test_speed():
    import time
    x = _test_signal(SR)
    x = np.tile(x, 2)[:SR]
    slow = {}
    for name in FX_NAMES:
        fn = FX.REGISTRY[name]["fn"]
        fn(x, SR)
        t = time.perf_counter()
        fn(x, SR)
        slow[name] = time.perf_counter() - t
    worst = max(slow, key=slow.get)
    MEASURED["slowest fx for 1 s of audio"] = (worst, round(slow[worst], 3))
    assert slow[worst] < 2.0, slow


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_") and callable(v):
            v()
            print("ok", k)
    for k, v in MEASURED.items():
        print(f"  {k}: {v}")
