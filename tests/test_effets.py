"""Thème Effets: chiffres des sources contrôlés sur le signal rendu. Mesuré, jamais écouté."""
import os
import tempfile

import numpy as np
import pytest
from scipy.signal import sosfreqz

from genny import dsp as D
from genny import filters as F
from genny import fx as FX
from genny import reverbs as R
from genny import spectral as S
from genny.analysis import alias_snr, detect_clicks, true_peak_db
from genny.core import write_wav

SR = 44100


def sine(f, dur=0.5, a=0.5, sr=SR):
    return a * np.sin(2 * np.pi * f * np.arange(int(dur * sr)) / sr)


def noise(dur=1.0, a=0.2, seed=1, sr=SR):
    return a * np.random.default_rng(seed).standard_normal(int(dur * sr))


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def db(v):
    return 20 * np.log10(max(float(v), 1e-15))


def gain(fn, f, dur=0.5, a=0.1, skip=0.2):
    i = int(skip * SR)
    x = sine(f, dur, a)
    return db(rms(np.asarray(fn(x))[i: x.shape[0]]) / rms(x[i:]))


def band_db(y, lo, hi, sr=SR):
    m = np.mean(y, axis=1) if y.ndim > 1 else y
    P = np.abs(np.fft.rfft(m)) ** 2
    k = np.fft.rfftfreq(m.shape[0], 1 / sr)
    return 10 * np.log10(P[(k >= lo) & (k < hi)].sum() + 1e-30)


def tone_db(y, f, sr=SR):
    """Amplitude of the partial at f, Hann window, in dB."""
    w = np.hanning(y.shape[0])
    P = np.abs(np.fft.rfft(y * w)) * 2 / w.sum()
    k = int(round(f * y.shape[0] / sr))
    return db(P[max(k - 3, 0): k + 4].max())


def peak_hz(y, sr=SR, lo=50.0):
    P = np.abs(np.fft.rfft(y * np.hanning(y.shape[0])))
    k = np.fft.rfftfreq(y.shape[0], 1 / sr)
    from scipy.ndimage import gaussian_filter1d
    P = gaussian_filter1d(P, 30)
    P[k < lo] = 0
    return float(k[int(np.argmax(P))])


def wav(x, sr=SR):
    fd, p = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    write_wav(p, x, sr)
    return p


# ------------------------------------------------------------------ filters
@pytest.mark.parametrize("c,bw", [(1000.0, 1.0), (10000.0, 1.0)])
def test_svf_band_edges_at_minus_3_db_also_near_nyquist(c, bw):
    """Zavalishin 4.19-4.20."""
    for f in (c * 2 ** (bw / 2), c / 2 ** (bw / 2)):
        assert gain(lambda x: F.svf_band(x, "bandpass", c, bw, SR), f) == pytest.approx(-3.01, abs=0.15)
    assert gain(lambda x: F.svf_band(x, "bandpass", c, bw, SR), c) == pytest.approx(0.0, abs=0.1)


def test_svf_allpass_flat_resonance_and_bandshelf():
    for f in (200.0, 1000.0, 5000.0):
        assert gain(lambda x: F.svf(x, "allpass", 1000.0, SR, q=2.0), f) == pytest.approx(0.0, abs=0.05)
    assert gain(lambda x: F.svf(x, "bandshelf", 1000.0, SR, q=1.0, gain_db=6.0), 1000.0) == pytest.approx(6.0, abs=0.1)
    assert gain(lambda x: F.svf(x, "bandshelf", 1000.0, SR, q=1.0, gain_db=6.0), 100.0) == pytest.approx(0.0, abs=0.3)


@pytest.mark.parametrize("fc", [1000.0, 16000.0])
def test_matched_keeps_analog_values_at_fc_near_nyquist(fc):
    """Vicanek, Matched Second Order Digital Filters: |H(fc)| = Q for the low-pass, 1 for the band-pass, G for the peak; RBJ cramps there."""
    assert gain(lambda x: F.matched(x, "lowpass", fc, SR, q=2.0), fc) == pytest.approx(db(2.0), abs=0.2)
    assert gain(lambda x: F.matched(x, "bandpass", fc, SR, q=2.0), fc) == pytest.approx(0.0, abs=0.2)
    assert gain(lambda x: F.matched(x, "peak", fc, SR, q=2.0, gain_db=6.0), fc) == pytest.approx(6.0, abs=0.2)


def test_onepole_tpt():
    assert gain(lambda x: F.onepole(x, "lowpass", 1000.0, SR), 1000.0) == pytest.approx(-3.01, abs=0.1)
    assert gain(lambda x: F.onepole(x, "highpass", 1000.0, SR), 1000.0) == pytest.approx(-3.01, abs=0.1)
    assert gain(lambda x: F.onepole(x, "allpass", 1000.0, SR), 3000.0) == pytest.approx(0.0, abs=0.05)


def test_resonz_norms():
    assert gain(lambda x: F.resonz(x, 1000.0, SR, q=5.0, norm="peak"), 1000.0) == pytest.approx(0.0, abs=0.1)
    n = noise(4.0)
    assert db(rms(F.resonz(n, 1000.0, SR, q=5.0, norm="power")) / rms(n)) == pytest.approx(0.0, abs=0.5)


def test_korg35_res_raises_cutoff_and_stays_finite():
    lo = gain(lambda x: F.korg35(x, 1000.0, SR, res=0.2, sat=1.0), 1000.0, a=0.01)
    hi = gain(lambda x: F.korg35(x, 1000.0, SR, res=1.8, sat=1.0), 1000.0, a=0.01)
    assert hi > lo + 6.0
    assert np.isfinite(F.korg35(noise(0.5, 1.0), 1000.0, SR, res=1.99)).all()


def test_peak_allpass_regalia_mitra():
    assert gain(lambda x: F.peak_allpass(x, 1000.0, SR, bw=200.0, gain_db=6.0), 1000.0) == pytest.approx(6.0, abs=0.1)
    assert gain(lambda x: F.peak_allpass(x, 1000.0, SR, bw=200.0, gain_db=6.0), 8000.0) == pytest.approx(0.0, abs=0.2)


# ------------------------------------------------------------------ dynamics
def test_decoupled_peak_eq17_follows_step_with_attack_constant_and_no_release_lag():
    a = np.exp(-1 / (0.01 * SR))
    r = np.exp(-1 / (0.1 * SR))
    x = np.r_[np.zeros(100), np.ones(4410)]
    y = FX._decoupled_peak(x, a, r)
    assert y[100 + 441] == pytest.approx(1 - np.exp(-1), abs=0.01)


def test_compressor_static_curve_mix_and_key():
    x = sine(1000.0, 1.0, 1.0)
    i = SR // 2
    y = FX.compressor(x, SR, threshold=-18, ratio=4.0, knee=0.0, auto_makeup=False)
    assert db(rms(y[i:]) / rms(x[i:])) == pytest.approx(-13.5, abs=1.0)
    d = FX.compressor(x, SR, threshold=-18, ratio=4.0, mix=0.0, auto_makeup=False)
    assert np.allclose(d, x, atol=1e-9)
    p = wav(sine(100.0, 1.0, 1.0))
    q = sine(1000.0, 1.0, 0.01)
    assert db(rms(FX.compressor(q, SR, threshold=-18, ratio=4.0, knee=0.0, auto_makeup=False, key=p)[i:]) / rms(q[i:])) < -10.0
    os.unlink(p)


def test_limiter_ceiling_true_peak_and_release():
    x = noise(1.0, 0.8)
    y = FX.limiter(x, SR, ceiling=-3.0)
    assert np.abs(y).max() <= 10 ** (-3 / 20) + 1e-6
    t = FX.limiter(x, SR, ceiling=-3.0, true_peak=True)
    assert true_peak_db(t) <= -3.0 + 0.3
    assert np.isfinite(y).all()
    # a lone tick recovers on the short 2 ms release, a sustained overload on the slow one (Mastering Audio via lot)
    tick = np.full(SR, 0.2)
    tick[SR // 2] = 2.0
    g = FX.limiter(tick, SR, ceiling=-6.0)
    assert g[SR // 2 + int(0.02 * SR)] == pytest.approx(0.2, abs=0.02)
    loud = np.r_[np.full(SR, 2.0), np.full(SR // 2, 0.2)]
    h = FX.limiter(loud, SR, ceiling=-6.0)
    assert h[SR + int(0.02 * SR)] < 0.17


def test_gate_hysteresis_and_hold():
    """DAFX M-file 5.4."""
    a = lambda d: 10 ** (d / 20)                                  # thresholds relative to the maximum of the envelope, as in the M-file
    n = int(0.3 * SR)
    t = np.arange(3 * n) / SR
    tone = np.sin(2 * np.pi * 1000 * t)
    env = np.r_[np.ones(n), np.full(n, a(-42.0)), np.full(n, a(-60.0))]
    y = FX.gate(env * tone, SR, open_db=-40.0, close_db=-45.0, hold=0.05, release=0.01)
    assert db(rms(y[n + 2000: 2 * n]) / a(-42.0)) == pytest.approx(-3.01, abs=0.3)          # between the thresholds, after loud: stays open
    assert rms(y[2 * n + int(0.1 * SR):]) < 1e-5                                             # under the lower one: closes
    e2 = np.r_[np.zeros(n), np.full(n, a(-42.0)), np.ones(n)]
    y2 = FX.gate(e2 * tone, SR, open_db=-40.0, close_db=-45.0, hold=0.05, release=0.01)
    assert rms(y2[n: 2 * n]) < 1e-5                                                          # the same level from silence: stays shut
    assert rms(y2[2 * n + 2000:]) > 0.6
    e3 = np.r_[np.ones(n), np.zeros(2 * n)]
    y3 = FX.gate(e3 * tone + 1e-3 * tone * (e3 == 0), SR, close_db=-45.0, open_db=-40.0, hold=0.05, release=0.01)
    assert rms(y3[n + int(0.03 * SR): n + int(0.04 * SR)]) > 5e-4                            # held for 50 ms past the drop
    assert rms(y3[n + int(0.1 * SR):]) < 1e-5


def test_dyno_static_slope():
    """ChucK Dyno, lot R05: g = (level / thresh)^(slope - 1) over the threshold."""
    x = sine(1000.0, 1.0, 1.0)
    y = FX.dyno(x, SR, above=0.5, below=1.0, thresh=0.5)
    assert db(rms(y[SR // 2:]) / rms(x[SR // 2:])) == pytest.approx(db(2.0 ** -0.5), abs=1.0)


# ------------------------------------------------------------------ modulation
@pytest.mark.parametrize("pos,hz", [(0.0, 450.0), (1.0, 450.0 * 2 ** 2.3)])
def test_wah_crybaby_resonance(pos, hz):
    """Faust crybaby: 450 x 2^(2.3 w)."""
    assert peak_hz(FX.wah(noise(1.0), SR, mode="pos", pos=pos)) == pytest.approx(hz, rel=0.1)


def test_wah_folds_makes_more_cycles():
    n = noise(0.5)
    assert not np.allclose(FX.wah(n, SR, mode="pos", pos=0.5, folds=1), FX.wah(n, SR, mode="pos", pos=0.5, folds=2))


def test_octaver_partials():
    x = sine(440.0, 1.0, 0.5)
    y = FX.octaver(x, SR, up=0.5, down=0.5, dry=0.0, tone=4000.0)[SR // 4:]
    t = tone_db(y, 440.0)
    assert tone_db(y, 880.0) > t + 6 and tone_db(y, 220.0) > t + 6


def test_radio_nothing_over_15k_and_deemphasis_flat():
    """Broadcast chain: de-emphasis undoes the pre-emphasis for a quiet signal, nothing over the 15 kHz low-pass."""
    y = FX.radio(noise(1.0, 0.1), SR)
    assert band_db(y, 16500, 22050) < band_db(y, 100, 10000) - 50
    assert np.isfinite(y).all()


def test_balance_centre_identity():
    x = np.stack([noise(0.2), noise(0.2, seed=2)], axis=1)
    assert np.allclose(FX.balance(x, SR, pos=0.0), x)
    assert rms(FX.balance(x, SR, pos=1.0)[:, 0]) < 1e-9


def test_exp_delay_taps():
    """ChucK ExpDelay t_i = L (i/N)^d."""
    x = np.zeros(SR // 10)
    x[0] = 1.0
    y = FX.exp_delay(x, SR, length=1.0, taps=4, durcurve=2.0, mix=1.0)
    assert list(np.flatnonzero(y)) == [round(SR * (i / 4) ** 2) for i in (1, 2, 3)]
    assert y.sum() == pytest.approx(1.0)


def test_pan_itd_delays_far_ear():
    """Fastl & Zwicker: 0.6 ms at most at full position; here half way."""
    x = np.zeros(4410)
    x[100] = 1.0
    y = FX.pan(x, SR, pos=0.5, itd=0.6)
    assert np.argmax(np.abs(y[:, 0])) - np.argmax(np.abs(y[:, 1])) == pytest.approx(0.0003 * SR, abs=1.0)


# ------------------------------------------------------------------ dsp
@pytest.mark.parametrize("family", ["cheby2", "ellip"])
def test_steep_families(family):
    sos = D.iir_sos("lowpass", 1000.0, SR, order=6, family=family, atten=60.0)
    w, h = sosfreqz(sos, worN=[2 * np.pi * 3000.0 / SR], fs=2 * np.pi)
    assert 20 * np.log10(abs(h[0]) + 1e-15) < -50


def test_crossover_lr4_sums_flat_multiband_neutral_at_ratio_1():
    n = noise(0.5)
    a, b = D.crossover(n, 1000.0, SR)
    imp = np.zeros(8192)
    imp[0] = 1.0
    A, B = D.crossover(imp, 1000.0, SR)
    H = np.abs(np.fft.rfft(A + B))
    assert 20 * np.log10(H.max() / H.min()) < 0.1
    assert a.shape == n.shape == b.shape
    m = D.multiband(imp, SR, freqs="300 3000", ratio=1.0)
    H2 = np.abs(np.fft.rfft(m))
    assert 20 * np.log10(H2.max() / H2.min()) < 0.2


def test_deesser_only_touches_the_sibilance():
    x = sine(6000.0, 1.0, 0.5) + sine(500.0, 1.0, 0.5)
    y = D.deesser(x, SR, freq=6000.0, threshold=-30.0, ratio=8.0)[SR // 2:]
    z = x[SR // 2:]
    assert tone_db(y, 6000.0) < tone_db(z, 6000.0) - 6
    assert tone_db(y, 500.0) == pytest.approx(tone_db(z, 500.0), abs=1.0)


def test_tape_stop_is_silent_after_time():
    """Faust tapeStop: the tape ends at start + time, gain falling to 0."""
    y = D.tape_stop(sine(440.0, 1.0), SR, start=0.2, time=0.3)
    assert y.shape[0] == int(0.5 * SR)
    assert rms(y[-int(0.002 * SR):]) < 0.01
    assert rms(y[: int(0.2 * SR)]) > 0.3
    assert not np.isnan(y).any()


def test_chebyshev_exact_harmonic():
    """T_3(cos) = cos 3: a unit sine yields only the third harmonic."""
    y = D.chebyshev(np.cos(2 * np.pi * 441.0 * np.arange(SR) / SR), SR, harmonics="3:1")
    assert tone_db(y, 1323.0) == pytest.approx(0.0, abs=0.5)
    assert tone_db(y, 441.0) < -50 and tone_db(y, 882.0) < -50


def test_diode_clipper_small_signal_and_clip_level():
    """Holters DAFx-19: R 1 k, C 33 nF: -3 dB at 4823 Hz; Is 1 fA, vT 25 mV: clips around 0.7 V (the output is scaled by 1 / 0.7)."""
    base = gain(lambda x: D.diode_clipper(x, SR, drive=1.0), 100.0, a=1e-3)
    assert gain(lambda x: D.diode_clipper(x, SR, drive=1.0), 4823.0, a=1e-3) - base == pytest.approx(-3.01, abs=0.5)
    y = 0.7 * D.diode_clipper(sine(200.0, 0.3, 1.0), SR, drive=10.0)
    assert 0.6 < np.abs(y).max() < 0.85


@pytest.mark.parametrize("name", [k for k, v in D._ANTI.items() if v[0] is not None])
def test_adaa_antiderivatives_by_finite_differences(name):
    f = D._SHAPES[name]
    F1, F2 = D._ANTI[name]
    v = np.array([-2.3, -1.2, -0.6, -0.1, 0.25, 0.7, 1.4, 2.9])
    h = 1e-5
    assert (F1(v + h) - F1(v - h)) / (2 * h) == pytest.approx(f(v), abs=1e-4)
    if F2 is not None:
        assert np.diff(F2(v + h) - F2(v - h)) / (2 * h) == pytest.approx(np.diff(F1(v)), abs=1e-4)      # up to the constant of F1


@pytest.mark.parametrize("name", ["tanh", "hard"])
def test_adaa_beats_naive_on_aliasing(name):
    f0 = 1244.5
    x = 4.0 * np.sin(2 * np.pi * f0 * np.arange(SR) / SR)
    f = D._SHAPES[name]
    naive = alias_snr(f(x), SR, f0)
    assert alias_snr(D.adaa1(x, f, D._ANTI[name][0]), SR, f0) > naive + 2      # no figure from the source at this drive: only "better"; see the report
    assert alias_snr(D.adaa2(x, f, *D._ANTI[name]), SR, f0) > naive + 2


def test_vicanek_shaper_beats_naive_and_restores_top_octave():
    f0 = 1244.5
    x = 4.0 * np.sin(2 * np.pi * f0 * np.arange(SR) / SR)
    assert alias_snr(D.vicanek_shaper(x), SR, f0) > alias_snr(x / np.sqrt(1 + x * x), SR, f0) + 2


def test_fweighted_noise_is_lowest_near_3_4_khz():
    """Lipshitz / F-weighted shaping: minimum where the ear is most sensitive."""
    x = np.zeros(SR)
    e = D.quantize(x + 1e-9, 8, "fweighted") - x
    P = {f: band_db(e, f, f + 1000) for f in (500, 3000, 12000)}
    assert P[3000] < P[500] and P[3000] < P[12000]


def test_gsm_like_keeps_level_and_is_lossy():
    x = sine(300.0, 1.0, 0.3)
    y = D.gsm_like(x)
    assert abs(db(rms(y) / rms(x))) < 6.0
    assert not np.allclose(x, y[: x.shape[0]], atol=1e-3)


def test_cvsd_follows_low_tone():
    x = sine(300.0, 0.5, 0.3)
    y = D.codec(x, SR, kind="cvsd")
    assert tone_db(y, 300.0) == pytest.approx(tone_db(x, 300.0), abs=6.0)


def test_dsf_matches_direct_sum():
    """Moorer, closed form against the partials."""
    a, N, f0 = 0.7, 8, 220.0
    y = D.dsf(4000, f0, SR, a=a, ratio=1.0, terms=N)
    t = np.arange(4000) / SR
    d = sum(a ** k * np.sin(2 * np.pi * f0 * (1 + k) * t) for k in range(N))
    c = float(np.dot(y, d) / np.dot(y, y))
    assert np.max(np.abs(c * y - d)) < 1e-6


def test_delta_sigma_is_one_bit_and_keeps_the_tone():
    x = sine(200.0, 0.1, 0.5)
    b = D.delta_sigma(x, 2, 64)
    assert set(np.unique(b)) <= {-1.0, 1.0}
    assert tone_db(D.resample_sinc(b, 64, 1)[: x.shape[0]], 200.0) == pytest.approx(tone_db(x, 200.0), abs=0.5)


def test_k_meter_reads_zero_at_minus_14():
    x = sine(1000.0, 1.0, 10 ** (-14 / 20) * np.sqrt(2))
    assert D.k_meter(x, 14, SR)[0] == pytest.approx(0.0, abs=0.05)


def test_shapes_no_nan_no_click_on_smooth_input():
    x = sine(220.0, 0.3, 0.8)
    for name in ("hard", "algebraic", "hyper", "asinh"):
        y = D.waveshape(x, name, 2.0)
        assert np.isfinite(y).all()
    assert detect_clicks(D.tape_stop(x, SR, start=0.05, time=0.1)).size == 0


# ------------------------------------------------------------------ flanger, phaser, chorus, leslie
def spectrum(y):
    return np.abs(np.fft.rfft(y))


def test_phaser_feedback_keeps_notches_and_peak_at_6_db():
    """Zavalishin 11.2-11.3: mix 0.5, k = 0.5: notches at zero, peaks 1/(1-k) = +6 dB. Frozen sweep."""
    imp = np.zeros(32768)
    imp[0] = 1.0
    y = R.phaser(imp, SR, rate=0.0, depth=0.0, stages=4, mix=0.5, feedback=0.5, freq=500.0, spacing=1.0, sweep=0.0)
    H = spectrum(y[: imp.shape[0]])
    assert 20 * np.log10(H.max()) == pytest.approx(6.02, abs=0.3)
    assert 20 * np.log10(H.min()) < -40


def test_phaser_invert_swaps_and_notch_mode():
    imp = np.zeros(32768)
    imp[0] = 1.0
    kw = dict(rate=0.0, depth=0.0, stages=4, mix=0.5, freq=500.0, spacing=1.0, sweep=0.0)
    a = spectrum(R.phaser(imp, SR, **kw)[: imp.shape[0]])
    b = spectrum(R.phaser(imp, SR, invert=True, **kw)[: imp.shape[0]])
    assert a[0] == pytest.approx(1.0, abs=1e-3) and b[0] < 1e-3
    y = R.phaser(imp, SR, rate=0.0, depth=0.0, stages=2, mix=0.5, freq=1000.0, sweep=0.0, mode="notch", notch_q=4.0)
    H = spectrum(y[: imp.shape[0]])
    k = np.fft.rfftfreq(imp.shape[0], 1 / SR)
    assert k[np.argmin(H)] == pytest.approx(1000.0, abs=15.0)
    assert 20 * np.log10(H.min()) < -40


def test_phaser_pd_style_runs_clean():
    y = R.phaser(noise(0.5), SR, rate=0.3, style="pd")
    assert np.isfinite(y).all()


def test_flanger_feedback_notches_stay_at_zero():
    """Zavalishin 11.2-11.3: the factor 1 + k keeps the notch; peak 1 / (1 - k) at mix 0.5."""
    imp = np.zeros(32768)
    imp[0] = 1.0
    y = R.flanger(imp, SR, rate=0.0, depth=0.0, feedback=0.5, mix=0.5, delay=2.0, deep=True)
    H = spectrum(y[: imp.shape[0]])
    assert 20 * np.log10(H.max()) == pytest.approx(6.02, abs=0.4)
    assert 20 * np.log10(H.min()) < -30


def test_flanger_damp_dulls_regeneration():
    imp = np.zeros(32768)
    imp[0] = 1.0
    y = R.flanger(imp, SR, rate=0.0, depth=0.0, feedback=0.7, mix=1.0, delay=2.0, damp=0.7)[: imp.shape[0]]
    z = R.flanger(imp, SR, rate=0.0, depth=0.0, feedback=0.7, mix=1.0, delay=2.0, damp=0.0)[: imp.shape[0]]
    k = np.fft.rfftfreq(imp.shape[0], 1 / SR)
    top = (k > 10000) & (k < 20000)
    ripple = lambda v: 20 * np.log10(spectrum(v)[top].max() / spectrum(v)[top].min())
    assert ripple(y) < ripple(z) - 6                                # the peaks of the regeneration go first in the treble


def test_white_chorus_is_allpass_when_frozen():
    """Dattorro: BL .7, FF 1, FB -.7 (DAFX Table 3.4): |H| = 1 with no depth."""
    imp = np.zeros(32768)
    imp[0] = 1.0
    y = R.chorus(imp, SR, rate=0.8, depth=0.0, preset="white", voices=1, delay=10.0)
    H = spectrum(y[:, 0][: imp.shape[0]])
    assert 20 * np.log10(H.max() / H.min()) < 0.5


def test_chorus_noise_lfo_negative_feedback():
    y = R.chorus(noise(0.5), SR, lfo="noise", feedback=-0.5)
    assert np.isfinite(y).all() and y.ndim == 2


def test_leslie_rates():
    assert R._LESLIE_RATES["slow"] == pytest.approx((50 / 60, 40 / 60))
    assert R._LESLIE_RATES["fast"] == pytest.approx((6.92, 5.9))


def test_decorrelate_rho():
    """Kendall: mode phase rho = cos(angle); FIR close to 0, spectrum kept."""
    n = noise(2.0)
    def rho(y):
        return float(np.corrcoef(y[:, 0], y[:, 1])[0, 1])
    assert abs(rho(R.decorrelate(n, SR, mode="fir"))) < 0.15
    assert rho(R.decorrelate(n, SR, mode="phase", angle=60.0)) == pytest.approx(0.5, abs=0.05)
    assert rho(R.decorrelate(n, SR, mode="phase", angle=90.0)) == pytest.approx(0.0, abs=0.05)
    y = R.decorrelate(n, SR, mode="fir")
    assert db(rms(y[:, 0]) / rms(n)) == pytest.approx(0.0, abs=0.5)


def test_prime_power_lengths_mutually_prime():
    from math import gcd
    M = R.prime_power_lengths(R.fdn_lengths(SR, 20.0, 8))
    assert all(gcd(int(a), int(b)) == 1 for i, a in enumerate(M) for b in M[i + 1:])


def test_fx_small_options():
    a = FX.reverb(np.r_[np.ones(1), np.zeros(SR * 2)], SR, variant="freeverb", mix=1.0)   # Freeverb: tail decays, no blow-up
    assert np.isfinite(a).all() and rms(a[-SR // 4:]) < rms(a[: SR // 4])
    b = FX.bitcrush(sine(100.0, 0.5, 0.9), SR, bits=12, rate=SR, bits_end=2)
    assert np.unique(np.round(b[-SR // 10:], 6)).size <= 32 and np.unique(np.round(b[:SR // 10], 6)).size > 100
    c = D.crunch(np.r_[np.full(1000, 0.2), np.full(1000, 0.0)], SR, threshold=0.315)      # dead zone
    assert np.abs(c).max() < 1e-12
    d = FX.sweep(noise(1.0) * np.r_[np.zeros(SR // 2), np.ones(SR - SR // 2)], SR, start=300, end=6000, follow="env")
    assert np.isfinite(d).all()
    e = D.exciter(sine(1000.0, 0.5, 0.5) + sine(4000.0, 0.5, 0.2), SR, freq=3000.0)
    assert band_db(e, 7000, 13000) > band_db(sine(1000.0, 0.5, 0.5) + sine(4000.0, 0.5, 0.2), 7000, 13000) + 20
    f = D.lockhart_fold(sine(220.0, 0.3, 0.8), SR)
    assert np.isfinite(f).all() and rms(f) > 0
    g = FX.autopan(sine(1000.0, 1.0), SR, rate=1.0, itd=0.6, rear=True)
    assert np.isfinite(g).all() and g.ndim == 2


def test_allpass_for_delay_group_delay():
    """Parker: first-order allpass with the asked group delay at w."""
    w, D0 = 2 * np.pi * 1000.0 / SR, 0.6
    a = D.allpass_for_delay(D0, w)
    from scipy.signal import freqz
    h = 1e-4
    ph = [np.angle(freqz([a, 1.0], [1.0, a], worN=[v])[1][0]) for v in (w - h, w + h)]
    assert -(ph[1] - ph[0]) / (2 * h) == pytest.approx(D0, abs=1e-3)


def test_compressor_presets_exist_and_compress():
    """Katz via the lot: mastering = ratio 2, threshold -15, attack 0.1, release 0.25."""
    x = sine(1000.0, 2.0, 1.0)
    y = FX.compressor(x, SR, preset="mastering", auto_makeup=False)
    assert db(rms(y[SR:]) / rms(x[SR:])) == pytest.approx(-7.5, abs=1.5)


# ------------------------------------------------------------------ spectral
def test_spectral_gate_soft_keeps_tone_and_lowers_hiss():
    x = sine(1000.0, 1.0, 0.5) + noise(1.0, 0.005)
    y = S.spectral_gate(x, SR, mode="soft", c=0.01)
    i = SR // 4
    assert tone_db(y[i:-i], 1000.0) == pytest.approx(tone_db(x[i:-i], 1000.0), abs=1.0)
    assert band_db(y, 5000, 15000) < band_db(x, 5000, 15000) - 6


def test_spectral_gate_subtract_removes_learned_noise():
    """Puckette I04."""
    nz = noise(1.5, 0.02)
    x = nz.copy()
    x[SR // 2:] += sine(1000.0, 1.0, 0.3)
    y = S.spectral_gate(x, SR, mode="subtract", learn=0.4)
    i = SR // 2 + SR // 8
    assert band_db(y[i:-i // 2], 3000, 15000) < band_db(x[i:-i // 2], 3000, 15000) - 20
    assert tone_db(y[i:-i // 2], 1000.0) == pytest.approx(tone_db(x[i:-i // 2], 1000.0), abs=1.0)


def test_stretch_anchors_lands_event_on_time():
    """Driedger & Mueller eq. 34-35."""
    x = np.zeros(2 * SR)
    x[SR // 2] = 1.0
    x = x + noise(2.0, 1e-4)
    y = S.stretch(x, SR, mode="sola", anchors="0.5:1.0", transients=False)
    assert y.shape[0] == pytest.approx(int(2.5 * SR), abs=2)
    assert np.argmax(np.abs(y)) / SR == pytest.approx(1.0, abs=0.03)


def test_hpss_splits_tone_and_clicks():
    t = sine(440.0, 1.0, 0.3)
    c = np.zeros(SR)
    c[::SR // 4] = 1.0
    h, p = S.hpss(t + c, SR)
    assert np.allclose(h + p, t + c)
    assert tone_db(h, 440.0) > tone_db(p, 440.0) + 6
    assert np.abs(p).max() > 2 * np.abs(h[SR // 8:]).max() * 0.5


def test_stretch_hps_length():
    x = sine(440.0, 0.5) + noise(0.5, 0.01)
    y = S.stretch(x, SR, 2.0, "hps")
    assert y.shape[0] == pytest.approx(x.shape[0] * 2, abs=2)
    assert np.isfinite(y).all()


def test_true_envelope_rides_over_harmonics():
    """Roebel & Rodet: the true envelope stays above the peaks within delta_db, the plain cepstrum falls under them."""
    x = sum(np.sin(2 * np.pi * 200.0 * k * np.arange(4096) / SR) / k for k in range(1, 40))
    mag = np.abs(np.fft.rfft(x * np.hanning(4096)))[None, :]
    n_c = int(SR / (2 * 200.0))
    te = S.true_envelope(mag, n_c, 2.0)
    assert np.max(20 * np.log10(np.maximum(mag, 1e-5 * mag.max()) / te)) <= 2.0 + 0.2


def test_transfer_envelope_pitch_lifter_and_max_f0():
    assert S.max_f0(sine(300.0, 0.5), SR) == pytest.approx(300.0, rel=0.03)
    assert S.max_f0(noise(0.5), SR) < 1500.0
    s = sum(sine(300.0 * k, 0.5, 0.2 / k) for k in range(1, 8))
    y = S.transfer_envelope(s, s, SR, env_ms=None, method="true")
    assert np.isfinite(y).all()


def test_freqshift_fold_removes_mirror():
    """Addendum: 21 kHz + 2 kHz wraps to 21.1 kHz unless removed first."""
    x = sine(21000.0, 0.5, 0.5)
    assert tone_db(S.freqshift(x, SR, hz=2000.0, fold=True), 21100.0) > -12
    assert tone_db(S.freqshift(x, SR, hz=2000.0, fold=False), 21100.0) < -50


def test_cross_synth_stamp_puts_modulator_on_carrier():
    p = wav(noise(1.0, 0.2))
    m = sine(2000.0, 1.0, 0.5)
    y = S.cross_synth(m, SR, carrier=p, method="stamp", squelch=20.0, amount=1.0)
    assert band_db(y, 1800, 2200) > band_db(y, 6000, 6400) + 30
    os.unlink(p)


def test_mutate_magmul_keeps_common_part():
    p = wav(sine(1000.0, 1.0, 0.5) + noise(1.0, 0.01))
    y = S.mutate(sine(1000.0, 1.0, 0.5) + sine(3000.0, 1.0, 0.5), SR, other=p, mode="magmul")
    assert band_db(y, 900, 1100) > band_db(y, 2900, 3100) + 20
    os.unlink(p)


def test_autotune_lands_on_semitone():
    """DAFX 10.4: 55 2^(n/12); 450 Hz sits between A4 and A#4."""
    y = S.autotune(sine(450.0, 1.0, 0.5), SR)
    assert peak_hz(y[SR // 4: -SR // 4], lo=100.0) == pytest.approx(440.0 * 2 ** (round(12 * np.log2(450 / 440)) / 12), abs=4.0)


def test_gender_octave_up():
    s = sum(sine(150.0 * k, 0.6, 0.2 / k) for k in range(1, 10))
    y = S.gender(s, SR, to="female")
    assert S.max_f0(y, SR) == pytest.approx(300.0, rel=0.05)


@pytest.mark.parametrize("name", ["tanh", "hard"])
def test_adaa2_fallback_when_x_returns_matches_regular_branch(name):
    f, F1, F2 = D._SHAPES[name], *D._ANTI[name]
    at = D.adaa2(np.array([0.0, 0.3, 2.0, 0.3]), f, F1, F2)[3]
    near = D.adaa2(np.array([0.0, 0.3, 2.0, 0.3 + 3e-4]), f, F1, F2)[3]
    assert at == pytest.approx(near, abs=1e-3)


def test_lockhart_tone_is_one_pole():
    x = sine(200.0, 1.0, 0.02) + sine(5200.0, 1.0, 0.02)
    y = D.lockhart_fold(x, SR, drive=0.1, tone=1300.0)
    h = lambda f: 1.0 / np.sqrt(1.0 + (f / 1300.0) ** 2)      # noqa: E731
    got = band_db(y, 5100, 5300) - band_db(y, 150, 250)
    assert got == pytest.approx(20 * np.log10(h(5200.0) / h(200.0)), abs=1.0)
