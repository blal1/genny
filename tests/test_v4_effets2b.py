"""V4_EFFETS2B: T_Effets lines outside dsp/cyber/spectral/fx/filters/osc/reverbs/foley/sfx.

Run:  uv run --with pytest pytest tests/test_v4_effets2b.py -q
"""
import numpy as np

from genny import compose as C

SR = 48000
BASE = {"type": "compose", "key": "D", "scale": "dorian", "progression": "i VII VI VII", "bars": 8, "seed": 7}


def _slope(x):
    P = np.abs(np.fft.rfft(x - x.mean())) ** 2
    f = np.arange(P.size)
    m = (f > 2) & (f < P.size // 2)
    return np.polyfit(np.log(f[m]), np.log(P[m] + 1e-30), 1)[0]


def test_gamma_noise_slope():
    rng = np.random.default_rng(0)
    w, b = C.gamma_noise(8192, 0.0, rng), C.gamma_noise(8192, 2.0, rng)
    assert w.min() >= 0 and w.max() <= 1
    assert abs(_slope(w)) < 0.3 and -2.4 < _slope(b) < -1.6


def test_voss_gamma_mode_changes_melody():
    a = C.generate(dict(BASE, role="melody", mode="voss"))
    b = C.generate(dict(BASE, role="melody", mode="voss", gamma=2.0))
    assert [e["pitch"] for e in a] != [e["pitch"] for e in b]


def test_arma_hihat_lag1():
    x = C.arma_hihat(20000, np.random.default_rng(1))
    r1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    # ARMA(4,0,1) with phi1 = -0.5, theta = 0.93: lag-1 autocorrelation of the model
    from scipy.signal import lfilter
    h = lfilter([1, C.ARMA_HIHAT_MA], np.r_[1, -np.array(C.ARMA_HIHAT_AR)], np.r_[1.0, np.zeros(4000)])
    r_th = np.dot(h[:-1], h[1:]) / np.dot(h, h)
    assert abs(r1 - r_th) < 0.03 and abs(x.std() - 1) < 1e-6


def test_humanize_arma_model():
    L = dict(BASE, role="arp", bars=4)
    ref = C.generate(L, 120)
    a = C.generate(dict(L, humanize={"time": 8, "vel": 0, "model": "arma"}), 120)
    b = C.generate(dict(L, humanize={"time": 8, "vel": 0}), 120)
    da = np.array([y["t"] - x["t"] for x, y in zip(ref, a)])
    db = np.array([y["t"] - x["t"] for x, y in zip(ref, b)])
    assert not np.allclose(da, db) and 0.004 < da.std() * 0.5 < 0.012


def test_poisson_grains_density():
    g = C.poisson_grains(200.0, 20.0, np.random.default_rng(2))
    assert abs(len(g) / 200.0 - 20.0) < 1.0
    assert all(0.06 <= d <= 0.12 for _, d in g)


def test_off_echo_min_ioi():
    L = dict(BASE, role="arp", bars=2)
    ref = C.generate(L, 120)
    e = C.generate(dict(L, echo={"n": 2, "time": 1.0, "decay": 0.5}), 120)
    assert len(e) > len(ref)
    o = C.generate(dict(L, off={"time": 0.125, "steps": 12}), 120)
    ps = {round(x["pitch"], 3) for x in ref}
    assert any(round(x["pitch"] - 12, 3) in ps for x in o if round(x["pitch"], 3) not in ps)
    m = C.generate(dict(L, min_ioi=400), 120)                    # 400 ms = 0.8 beat at 120 bpm
    on = sorted({x["t"] for x in m})
    assert len(on) < len({x["t"] for x in ref}) and np.min(np.diff(on)) >= 0.8 - 1e-6


def test_ks_loops():
    p = 100
    for loop, f in (("avg", SR / (p + 0.5)), ("121", SR / p)):
        y = C.ks_drum_raw(SR, p, blend=1.0, loop=loop)
        Y = np.abs(np.fft.rfft(y[SR // 4:] * np.hanning(SR - SR // 4)))
        fr = np.fft.rfftfreq(SR - SR // 4, 1 / SR)
        m = (fr > 0.7 * f) & (fr < 1.3 * f)
        assert abs(fr[m][np.argmax(Y[m])] - f) < 2.0
    s = C.ks_drum_raw(SR // 2, p, blend=1.0, loop="siegel")
    a = C.ks_drum_raw(SR // 2, p, blend=1.0)
    hf = lambda y: np.mean(np.abs(np.diff(y[SR // 8:])))          # noqa: E731
    assert hf(s) < hf(a)
    k = C.ks_drum_raw(SR // 2, 300, blend=1.0, copies=3)
    Y = np.abs(np.fft.rfft(k)) ** 2
    fr = np.fft.rfftfreq(k.size, 1 / SR)
    E = np.array([Y[np.abs(fr - h * SR / 300.5) < 3].sum() for h in range(1, 13)])
    assert E[2::3].sum() > 10 * (E.sum() - E[2::3].sum())          # 3 copies: only every 3rd harmonic of fs/300.5 left


# ---------------------------------------------------------------- analysis
from genny import analysis as A  # noqa: E402


def test_wf_weight_points():
    for f, w in A.WF_WEIGHT:
        assert abs(A.wf_weight(f) - w) < 1e-9
    assert A.wf_weight(4.0) > A.wf_weight(3.0) and A.wf_weight(4.0) > A.wf_weight(6.0)


def test_wow_flutter_measures_fm():
    t = np.arange(4 * SR) / SR
    dev = 0.002                                                   # 0.2 % peak at 4 Hz (weight 1)
    ph = 2 * np.pi * A.WF_TONE * t - A.WF_TONE * dev / 4.0 * np.cos(2 * np.pi * 4.0 * t)
    r = A.wow_flutter(np.sin(ph), SR)
    assert abs(r["weighted_peak"] - 200 * dev / np.sqrt(2) * 100 / 100) < 0.03   # 2 sigma of a sine = sqrt 2 x peak
    assert r["wow"] > 10 * r["flutter"]


def test_normalized_echo_density():
    rng = np.random.default_rng(3)
    t, eta = A.normalized_echo_density(rng.standard_normal(SR), SR)
    assert abs(np.median(eta) - 1.0) < 0.05
    sparse = np.zeros(SR)
    sparse[::2400] = 1.0
    assert np.median(A.normalized_echo_density(sparse, SR)[1]) < 0.3
    ir = np.r_[sparse[:SR // 10], rng.standard_normal(SR)]
    assert 0.05 < A.mixing_time(ir, SR) < 0.13


def test_ratio_pitch_mel():
    for f, m in ((125, 125), (1300, 1050), (8000, 2100)):
        assert abs(A.ratio_pitch_mel(f) - m) / m < 0.03


def test_virtual_pitch_missing_fundamental():
    f0 = 200.0
    fr = f0 * np.arange(3, 9)
    assert abs(A.virtual_pitch(fr, np.zeros(fr.size)) - f0) < 2.0


def test_audible_partials_nam():
    f = np.array([1000.0, 1100.0])
    L = np.array([0.0, -20.0])            # 0.5 Bark apart, offset 10 dB: masked above 86 - 3.2, below 86 - 13.5
    assert A.audible_partials(f, L, spread="nam")[1] == False   # noqa: E712  upward spread -6.3 dB/Bark at 96 dB masks it
    assert A.audible_partials(f, np.array([-20.0, 0.0]), spread="nam")[0] == True   # noqa: E712  downward -27 dB/Bark


def test_spectral_descriptors_and_spread():
    t = np.arange(SR) / SR
    df = SR / round(0.0232 * SR)                  # on the frame's bins: no Hamming side-lobe leakage in the magnitudes
    f1, f2 = 23 * df, 70 * df                     # ~991 and ~3016 Hz
    y = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
    d = A.spectral_descriptors(y, SR)
    assert abs(d["centroid"] - (f1 + f2) / 2) < 30 and abs(d["spread"] - (f2 - f1) / 2) < 40 and abs(d["skewness"]) < 0.1
    b = A.brightness(y, SR)
    assert abs(b["spread"] - (f2 - f1) / 2) < 10


def test_hnr_ma_orders():
    t = np.arange(SR) / SR
    y = sum(np.sin(2 * np.pi * 200 * k * t) / k for k in range(1, 11))
    rng = np.random.default_rng(4)
    clean = A.hnr_ma(y + 0.001 * rng.standard_normal(SR), SR, 200)
    noisy = A.hnr_ma(y + 0.1 * rng.standard_normal(SR), SR, 200)
    assert clean > noisy + 15


def test_exposed_tail():
    t = np.arange(SR) / SR
    y = np.sin(2 * np.pi * 440 * t) * np.exp(-t / 0.05)
    r = A.exposed_tail(y, SR)
    assert not r["exposed"] and abs(r["tail_s"] - 0.05 * np.log(1000)) < 0.03
    assert A.exposed_tail(y[:SR // 10], SR)["exposed"]


# ---------------------------------------------------------------- physical/waveguides
from genny.physical import waveguides as WG  # noqa: E402


def test_nl_allpass_lossless():
    x = np.random.default_rng(5).standard_normal(4000)
    x[2000:] = 0.0
    for mode, g in ((0, 3.0), (1, 15.0)):
        y = WG.nl_allpass(x, g, mode)
        assert abs(np.sum(y ** 2) - np.sum(x ** 2)) / np.sum(x ** 2) < 1e-9


def test_holger_force_formula():
    h, b = 0.5e-3, 1e-3
    t = np.tanh(h / (2 * b))
    phi = (2 / 3) * b * t * (3 - t * t)
    assert abs(WG.holger_force(20.0, 1.0, h, 2e-3, b) - 1.2 * 2e-3 * 1.0 * 400.0 * phi) < 1e-12


def test_recorder_jet_offset_and_reynolds_gate():
    rng = lambda: np.random.default_rng(1)       # noqa: E731
    a = WG.recorder(523.0, None, 0.3, rng(), model="verge", tune=False)
    b = WG.recorder(523.0, None, 0.3, rng(), model="verge", tune=False, jet_offset=0.1)
    c = WG.recorder(523.0, None, 0.3, rng(), model="verge", tune=False, turbulence="reynolds")
    assert not np.allclose(a, b) and not np.allclose(a, c)
    assert WG.RE_TURB_ON == 700.0 and WG.RE_TURB_FULL == 3000.0 and WG.JET_Y0_AUVRAY == 1e-4


# ---------------------------------------------------------------- calibrate
from genny import calibrate as CAL  # noqa: E402


def test_multiscale_spectral_distance():
    rng = np.random.default_rng(6)
    x = rng.standard_normal(SR // 4)
    assert CAL.multiscale_spectral_distance(x, x) == 0.0
    assert CAL.MSS_FFTS == (2048, 1024, 512, 256, 128, 64) and CAL.MSS_OVERLAP == 0.75
    assert CAL.multiscale_spectral_distance(x, 0.5 * x) < CAL.multiscale_spectral_distance(x, 0.1 * x)
    assert CAL.multiscale_spectral_distance(x, 0.5 * x, w_mag=0.0, w_log=1.0) > 0


def test_spectral_flux_error_onsets():
    y = np.zeros(SR)
    y[SR // 4:SR // 4 + 200] = 1.0
    z = np.roll(y, SR // 4)
    assert CAL.spectral_flux_error(y, y, SR) == 0.0 and CAL.spectral_flux_error(y, z, SR) > 0.005


def test_loop_filter_from_modes_and_recording():
    f0, D = 220.0, 1 / 220.0
    fr = f0 * np.arange(1, 9)
    dec = 0.5 * (2.0 + 1e-6 * (2 * np.pi * fr) ** 2)
    lp = CAL.loop_from_modes(fr, dec)
    assert abs(lp["D"] - D) < 1e-12 and np.allclose(lp["gain"], np.exp(-dec * D)) and np.allclose(lp["phase"], 0, atol=1e-9)
    t = np.arange(int(0.4 * SR)) / SR
    x = sum(np.exp(-d * t) * np.sin(2 * np.pi * f * t) for f, d in zip(fr, dec))
    r = CAL.loop_filter_from_recording(x, SR, n_modes=8, dur=0.2)
    assert abs(r["D"] - D) / D < 1e-3 and abs(r["alpha"] - 2.0) < 0.3 and abs(r["beta"] - 1e-6) / 1e-6 < 0.2


# ---------------------------------------------------------------- analysis (2), identify, piano_data, modal_data
def test_count_overs():
    y = np.zeros(100)
    y[10:13] = 1.0
    y[50:52] = -1.0
    y[80:86] = 1.0
    assert A.count_overs(y) == 2 and A.count_overs(y, run=4) == 1 and A.OVER_RUN == 3


def test_temporal_descriptors_decay():
    t = np.arange(SR) / SR
    y = np.sin(2 * np.pi * 440 * t) * np.exp(-t / 0.1)
    d = A.temporal_descriptors(y, SR)
    assert abs(d["decay_slope"] + 20 / np.log(10) / 0.1) < 5            # -86.9 dB/s
    assert abs(d["temporal_centroid"] - 0.05) < 0.01


def test_spectral_kurtosis_impulses():
    from genny import identify as ID
    rng = np.random.default_rng(7)
    g = rng.standard_normal(SR)
    f, K = ID.spectral_kurtosis(g, SR)
    assert np.all(np.abs(K[2:-2]) < 0.3)
    imp = g * 0.1
    imp[::4800] += 30.0
    f, K2 = ID.spectral_kurtosis(imp, SR)
    assert np.median(K2[2:-2]) > 5 and ID.SK_NFFT == 64


def test_bensa_decay_limit():
    from genny.physical import piano_data as PD
    b1, b2 = PD.bensa_losses(261.6)
    assert abs(b1 - (4.4e-3 * 261.6 - 4e-2)) < 1e-12 and abs(b2 - (1e-6 * 261.6 + 1e-5)) < 1e-15
    n = np.arange(1, 6)
    d = PD.bensa_decay(261.6 * n, 261.6, 1e-9, 0.62)
    assert np.allclose(d, b1 + b2 * (np.pi * n / 0.62) ** 2, rtol=1e-4)


def test_tonebar_roundtrip():
    from genny.physical import modal_data as MD
    p = dict(w=0.0111, h=0.0034, E=69e9, rho=2700.0, m_e=0.030, Le=0.0268)
    L = MD.tonebar_length(293.7, **p)
    assert abs(MD.tonebar_f0(L, **p) - 293.7) < 1e-6 and abs(MD.TINE_BEAM_MASS - 3 / 1.8751 ** 4) < 1e-3


def test_master_loudness_headroom_only():
    from genny import spec as SP
    from genny.physical.core import _true_peak
    t = np.arange(SR) / SR
    y = 0.01 * np.sin(2 * np.pi * 1000 * t)
    y[SR // 2] = 0.5                                              # a spike: the true-peak margin is the limit
    out = SP.master_loudness(y, SR, -14.0, -1.0, headroom_only=True)
    assert abs(20 * np.log10(_true_peak(out)) + 1.0) < 0.05
    r = out / y
    assert np.allclose(r[np.abs(y) > 1e-6], r[np.abs(y) > 1e-6][0])     # one plain gain, no knee


def test_twm_and_coherence():
    f0 = 180.0
    fr = f0 * np.arange(1, 9)
    a = 1.0 / np.arange(1, 9)
    assert A.harmonic_coherence(fr, a, f0) > 0.99 and A.harmonic_coherence(fr, a, 1.37 * f0) < 0.5
    assert abs(A.twm_f0(fr, a, [90.0, 120.0, 180.0, 360.0]) - f0) < 1e-9


def test_pitch_strength_shape():
    assert abs(A.pitch_strength(1500.0) - 1.0) < 1e-9
    assert abs(A.pitch_strength(125.0) - 1 / 3) < 1e-9 and abs(A.pitch_strength(9000.0) - 1 / 3) < 1e-9
    assert abs(A.pitch_strength(1500.0, level_db=70.0) - 1.125) < 1e-9
    assert A.pitch_strength(1500.0, dur=0.03) < 0.7 and abs(A.pitch_strength(1500.0, kind="noise") - 1 / 7) < 1e-9


def test_sci_irregularity_centroid():
    t = np.arange(SR) / SR
    y = np.sin(2 * np.pi * 500 * t) + np.sin(2 * np.pi * 1500 * t)
    assert abs(A.spectral_content_index(y, SR, 500.0) - 2.0) < 0.02
    assert A.spectral_irregularity([1, 0.1, 1, 0.1, 1, 0.1]) > A.spectral_irregularity([1, 0.8, 0.6, 0.4, 0.2, 0.1])
    assert abs(A.harmonic_centroid([1.0, 1.0]) - 1.5) < 1e-12


def test_rauhala_tunable_and_strings():
    from genny.physical import piano_data as PD
    p = PD._RAUHALA_T1[4]
    lb = np.log(3e-4)
    ik = 40.0
    f0 = 27.5 * 2 ** ((ik - 1) / 12)
    d = np.exp(np.exp(p["C1"] * lb + p["C2"]) - ik * np.exp(p["k1"] * lb * lb + p["k2"] * lb + p["k3"]))
    assert abs(PD.rauhala_tunable(f0, 3e-4, 4) - (1 - d) / (1 + d)) < 1e-12
    assert PD.CHABASSIER_STRINGS["C1"]["T"] == 905.0 and PD.STEEL_RHO == 7850.0


def test_distortion_grid():
    x = 0.5 * np.sin(2 * np.pi * 440 * np.arange(SR // 10) / SR)
    g = dict(CAL.distortion_grid(x, SR))
    assert len(g) == len(CAL.FAD_NOISE_SD) + len(CAL.FAD_LOWPASS) + len(CAL.FAD_HIGHPASS) + len(CAL.FAD_BITS)
    assert len(np.unique(g["bits_2"])) <= 4 and abs(np.std(g["noise_0.31"] - x) - 0.31) < 0.02


# ---------------------------------------------------------------- graphsolver
def test_thiran_coeffs_order1_and_delay():
    from genny import graphsolver as GS
    from scipy.signal import group_delay
    D = 0.6
    assert np.allclose(GS.thiran_coeffs(D, 1), [1.0, (1 - D) / (1 + D)])
    a = GS.thiran_coeffs(3.3, 3)
    w, gd = group_delay((a[::-1], a), w=[0.01])
    assert abs(gd[0] - 3.3) < 1e-3


def test_thiran_delay_frac_window():
    from genny import graphsolver as GS
    d = GS.ThiranDelay(5.05, 64)
    imp = np.zeros(40)
    imp[0] = 1.0
    out = []
    for x in imp:
        d.write(x)
        out.append(d.read())
    out = np.array(out)
    # integer 4 (floor(5.05 - 0.1)), frac 1.05: first output a = (1 - 1.05)/(1 + 1.05), not 0.95/1.05
    assert abs(out[3] + 0.05 / 2.05) < 1e-12 and GS.THIRAN_MIN_FRAC == 0.1


# ---------------------------------------------------------------- physical/winds
def test_map_limits_and_iterated_map():
    from genny.physical import winds as W
    z, lam = 0.5, 0.93
    chi = (np.sqrt(3 + 1 / z ** 2) - 1 / z) / 9
    A = chi * (z + 1 / (3 * z)) - 1 / 18
    r = W.map_limits(z, lam)
    assert abs(r["A"] - A) < 1e-12 and abs(r["gamma_beat"] - (1 - 2 * lam * A) / (1 + lam)) < 1e-12
    assert r["negative_flow"] == (lam > 1 / (1 + 2 * A))
    quiet = W.iterated_map(0.1, z, lam, 400)
    loud = W.iterated_map(0.6, z, lam, 400)
    assert abs(quiet[-1] - quiet[-2]) < 1e-6 and abs(loud[-1] - loud[-2]) > 0.1      # static vs period-2 square wave


def test_channel_flow_regimes():
    from genny.physical import winds as W
    w, L = 0.01, 0.01
    small = W.channel_flow(1000.0, w, 1e-6, L)
    assert abs(small - w * 1e-18 * 1000.0 / (12 * W.NU * W.RHO * L)) < 1e-20
    big = W.channel_flow(1000.0, w, 1e-3, L)
    assert abs(big - w * 1e-3 * np.sqrt(2000.0 / W.RHO)) < 1e-12
