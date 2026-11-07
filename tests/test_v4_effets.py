"""Effects-theme lines of the deep-reading report (wave 4): foley, fx and dsp additions."""
import math

import numpy as np

from genny import analysis as A
from genny import dsp as D
from genny import foley as FO
from genny import fx as FX
from genny.sfx import REGISTRY as SFX

SR = 48000


def band_db(y, sr, lo, hi):
    X = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / sr)
    return 10 * np.log10(X[(f >= lo) & (f < hi)].sum() + 1e-30)


# ------------------------------------------------------------------ foley
def test_coin_spin_boundary_layer_exponents():
    assert FO.COIN_WOBBLE_EXP == -2.0 / 9.0 and FO.COIN_TILT_EXP == 4.0 / 9.0


def test_crack_only_inside_the_mach_cone():
    edge = 90.0 - math.degrees(math.asin(FO.C_AIR / 850.0))
    assert FO.crack_cone_gain(0.0, 850.0) == 1.0 and FO.crack_cone_gain(edge + 5.0, 850.0) == 0.0
    assert FO.crack_cone_gain(10.0, 300.0) == 0.0
    a = SFX["gunshot"]["fn"](sr=SR, kind="rifle", distance=50.0, angle=85.0, echo=0.0)
    b = SFX["gunshot"]["fn"](sr=SR, kind="rifle", distance=50.0, crack=0.0, echo=0.0)
    assert np.array_equal(a, b)


def test_suppressor_cuts_160_to_1000_hz_and_first_round_pops():
    assert FO.GUNS["silenced"][5] == 0.08 and FO.SUPPRESSOR_CUT[:2] == (160.0, 1000.0)
    s = SFX["gunshot"]["fn"](sr=SR, kind="silenced", distance=5.0, echo=0.0, action=0.0)
    p = SFX["gunshot"]["fn"](sr=SR, kind="pistol", distance=5.0, echo=0.0, action=0.0)
    rel = lambda y: band_db(y, SR, 160, 1000) - band_db(y, SR, 2000, 6000)    # noqa: E731
    assert rel(s) < rel(p) - 3
    two = SFX["gunshot"]["fn"](sr=SR, kind="silenced", distance=5.0, echo=0.0, shots=2, rate=4.0)
    h = int(0.25 * SR)
    assert np.max(np.abs(two[:h])) > 1.3 * np.max(np.abs(two[h:2 * h]))
    r = SFX["gunshot"]["fn"](sr=SR, kind="silenced_rifle", distance=50.0, echo=0.0)
    assert len(r) > len(s)                                            # supersonic: the crack leads, the file is longer


def test_transporter_source_is_deterministic_fm():
    assert FO.TRANSPORTER_BUFFER == 0.12 and FO.TRANSPORTER_SWEEP == 12.0 and FO.TRANSPORTER_FLANGE == 0.012
    assert np.array_equal(FO.transporter(sr=SR, dur=0.5, seed=0), FO.transporter(sr=SR, dur=0.5, seed=3))


def test_scifi_fm_implode_falls():
    y = SFX["scifi_fm"]["fn"](sr=SR, kind="implode", freq=440.0)
    assert abs(len(y) / SR - 0.554) < 0.01
    q = len(y) // 4
    spec = lambda s: np.abs(np.fft.rfft(s * np.hanning(len(s))))     # noqa: E731  windowed: no edge leakage
    cent = lambda s: float(np.sum(np.fft.rfftfreq(len(s), 1 / SR) * spec(s)) / np.sum(spec(s)))  # noqa: E731
    assert cent(y[:q]) > 1.5 * cent(y[2 * q:3 * q])


def test_noizer_woody_rings_at_its_three_bands():
    assert abs(FO._oct_q(1.0) - math.sqrt(2.0)) < 1e-12
    y = SFX["noizer"]["fn"](sr=SR, preset="woody", dur=0.5)
    f = np.fft.rfftfreq(len(y), 1 / SR)
    X = np.abs(np.fft.rfft(y))
    for fc in (264.0, 165.0, 115.0):
        m = (f > fc * 0.9) & (f < fc * 1.1)
        assert abs(f[m][np.argmax(X[m])] - fc) < 4.0


def test_flue_register_thresholds_and_dynamic_overshoot():
    n = 6 * SR                          # 100 Pa/s; Pcorr = 87 Pa, so the fall back is near 172 Pa
    up = np.concatenate([np.linspace(150, 450, n // 2), np.linspace(450, 150, n // 2)])
    reg = FO.flue_register(up, SR)
    i_up = int(np.argmax(reg == 2))
    i_dn = n // 2 + int(np.argmax(reg[n // 2:] < 2))
    corr = (3 * 100.0 / (2 * FO.FLUE_ALPHA)) ** (2 / 3)
    assert abs(up[i_up] - (311.5 + corr)) < 3.0
    assert abs(up[i_dn] - (259.0 - corr)) < 3.0
    assert reg[0] == 0                                                # 200 Pa < 224: aeolian


# ------------------------------------------------------------------ fx
def test_kasfilter_holds_dc():
    y = FX.kasfilter(np.full(SR // 10, 0.5), SR, freq=200.0, res=0.0, accent=1.0)
    assert np.allclose(y[SR // 50:], 0.5)


def test_phase_bash_puts_the_source_on_a_pitch():
    x = np.random.default_rng(0).standard_normal(SR)
    y = FX.phase_bash(x, SR, pitch=200.0)
    X = np.abs(np.fft.rfft(y))
    assert X[600] > 20 * np.median(X[610:790])


def test_envelope_follower_attack_time():
    x = np.concatenate([np.zeros(100), np.ones(SR)])
    e = FX.envelope_follower(x, SR, attack=0.01, release=0.1)
    assert abs(e[100 + 480] - (1 - math.exp(-1))) < 0.01


def test_echo_criterion_and_thresholds():
    ir = np.zeros(SR)
    ir[0] = 1.0
    assert not FX.echo_criterion(ir, SR)[1]
    ir[int(0.12 * SR)] = 0.9
    assert FX.echo_criterion(ir, SR)[1]
    assert FX.echo_threshold_db(10.0) == -0.575 * 10 - 6
    assert FX.haas_delay_ms(3.0) == 93.0 and FX.haas_delay_ms(9.0) == 43.0


def test_tremolo_spread_and_skew():
    x = np.ones(SR)
    y = FX.tremolo(x, SR, rate=2.0, depth=1.0, spread=1.0)
    assert y.shape == (SR, 2) and abs(np.argmin(y[:SR // 2, 0]) - np.argmax(y[:SR // 2, 1])) < SR // 100
    z = FX.tremolo(x, SR, rate=1.0, depth=1.0, skew=0.5)
    assert abs(np.argmin(z) / SR - 0.75) < 0.01                       # crest of the LFO (lowest gain) at 3/4 cycle


def test_distortion_adaa2_aliases_less():
    x = np.sin(2 * np.pi * 3563.2 * np.arange(SR) / SR)
    a1 = A.alias_snr(FX.distortion(x, SR, drive=8.0, tone=0), SR, 3563.2)
    a2 = A.alias_snr(FX.distortion(x, SR, drive=8.0, tone=0, aa="adaa2"), SR, 3563.2)
    assert a2 > a1 + 3


def test_limiter_ceiling():
    x = 3 * np.sin(2 * np.pi * 997 * np.arange(SR) / SR)
    y = FX.limiter(x, SR, ceiling=-1.0)
    assert np.max(np.abs(y)) <= 10 ** (-1 / 20) + 1e-12


# ------------------------------------------------------------------ dsp
def test_adaa3_hard_clip():
    v = np.linspace(-3, 3, 13)
    h = 1e-3
    for F_lo, F_hi in ((D._ANTI["hard"][1], D._F3_hard),):
        d = (F_hi(v + h) - F_hi(v - h)) / (2 * h)
        assert np.allclose(d, F_lo(v), atol=1e-5)
    f0 = 1783.0
    x = 2.0 * np.sin(2 * np.pi * f0 * np.arange(44100) / 44100)
    a2 = A.alias_snr(D.adaa2(x, lambda v: np.clip(v, -1, 1), *D._ANTI["hard"]), 44100, f0)
    a3 = A.alias_snr(D.adaa_n(x, lambda v: np.clip(v, -1, 1), D._F3_hard, 3), 44100, f0)
    assert a3 > a2
    lo = np.sin(2 * np.pi * 50 * np.arange(4800) / 48000) * 2
    y = D.adaa_n(lo, lambda v: np.clip(v, -1, 1), D._F3_hard, 3)
    assert np.max(np.abs(y[10:] - np.clip(lo[8:-2] * 0.5 + lo[9:-1] * 0.5, -1, 1))) < 0.02


def test_gsm_interpolates_reflection_coefficients():
    assert D._GSM_INTERP == ((0, 13, 0.25), (13, 27, 0.5), (27, 40, 0.75), (40, 160, 1.0))
    t = np.arange(8000) / 8000
    x = 0.3 * np.sin(2 * np.pi * 200 * t) * (1 + (t > 0.5)) + 0.01 * np.random.default_rng(0).standard_normal(8000)
    y = D.gsm_like(x)
    assert np.all(np.isfinite(y)) and 0.3 < np.sqrt(np.mean(y ** 2)) / np.sqrt(np.mean(x ** 2)) < 3


# ------------------------------------------------------------------ filters
from genny import filters as FL  # noqa: E402


def _gain_db(y, x, sr, f):
    k = int(round(f * len(x) / sr))
    return 20 * np.log10(np.abs(np.fft.rfft(y))[k] / np.abs(np.fft.rfft(x))[k])


def test_svf_shelves_and_tilt():
    x = np.zeros(SR)
    x[0] = 1.0
    lo = FL.svf(x, "lowshelf", 1000.0, SR, gain_db=12.0)
    hi = FL.svf(x, "highshelf", 1000.0, SR, gain_db=12.0)
    ti = FL.svf(x, "tilt", 1000.0, SR, gain_db=12.0)
    assert abs(_gain_db(lo, x, SR, 20) - 12) < 0.1 and abs(_gain_db(lo, x, SR, 1000) - 6) < 0.1 and abs(_gain_db(lo, x, SR, 20000)) < 0.2
    assert abs(_gain_db(hi, x, SR, 20)) < 0.1 and abs(_gain_db(hi, x, SR, 1000) - 6) < 0.1
    assert abs(_gain_db(ti, x, SR, 20) + 6) < 0.1 and abs(_gain_db(ti, x, SR, 1000)) < 0.1 and abs(_gain_db(ti, x, SR, 20000) - 6) < 0.2


def test_matched_shelf_hits_analog_at_dc_and_nyquist():
    x = np.zeros(4096)
    x[0] = 1.0
    y = FL.matched_shelf(x, "lowshelf", 8000.0, SR, gain_db=12.0)
    H = np.abs(np.fft.rfft(y))
    G, wc, W = 10 ** 0.6, 2 * np.pi * 8000, np.pi * SR
    assert abs(H[0] - G) < 1e-6 and abs(H[-1] - np.sqrt((W ** 2 + G ** 2 * wc ** 2) / (W ** 2 + wc ** 2))) < 1e-6


def test_faust_q_mapping():
    assert FL.faust_q_to_k("moog", 25.0) == 4.0 and FL.faust_q_to_k("korg35", 10.0) == 2.0
    assert abs(FL.faust_q_to_k("diode", 25.0, 1.0) - 7.3) < 1e-12 and FL.faust_q_to_k("moog", 0.7071) == 0.0


def test_ladder_newton_and_adaa_solve_the_loop():
    t = np.arange(SR // 4) / SR
    x = 0.9 * np.sign(np.sin(2 * np.pi * 110 * t))
    a = FL.ladder(x, 2000.0, SR, res=0.9, solver="node", drive=3.0)
    b = FL.ladder(x, 2000.0, SR, res=0.9, solver="newton", drive=3.0)
    c = FL.ladder(x, 2000.0, SR, res=0.9, solver="adaa", drive=3.0)
    assert not np.allclose(a, b) and np.all(np.isfinite(c))
    lin = FL.ladder(1e-4 * x, 2000.0, SR, res=0.9, solver="newton")
    ref = FL.ladder(1e-4 * x, 2000.0, SR, res=0.9, solver="node")
    assert np.allclose(lin, ref, atol=1e-9)                           # small signal: the same linear ladder
    # loop gain k x drive kept under 4 (k 0.4, drive 6): above it the ladder self-oscillates and the measure is moot
    hi = A.alias_snr(FL.ladder(np.sin(2 * np.pi * 3563.2 * np.arange(SR) / SR), 15000.0, SR, res=0.1, drive=6.0, solver="newton"), SR, 3563.2)
    aa = A.alias_snr(FL.ladder(np.sin(2 * np.pi * 3563.2 * np.arange(SR) / SR), 15000.0, SR, res=0.1, drive=6.0, solver="adaa"), SR, 3563.2)
    assert aa > hi


def test_huovilainen_tuning_polynomials():
    x = np.zeros(SR)
    x[0] = 1e-3
    y = FL.ladder(x, 1000.0, SR, res=0.0, kind="huovilainen")
    H = np.abs(np.fft.rfft(y)) / 1e-3
    f = np.fft.rfftfreq(SR, 1 / SR)
    assert abs(H[1] - 1) < 0.02                                       # unity DC through four one-poles
    assert abs(20 * np.log10(H[1000]) + 12) < 1.5                     # four poles at the cutoff: -3 dB each
    assert f[np.argmax(np.abs(np.fft.rfft(FL.ladder(x, 1000.0, SR, res=0.95, kind="huovilainen"))))] > 800


def test_antisat_svf_self_oscillates_bounded():
    x = np.zeros(SR)
    x[0] = 1e-3
    y = FL.svf_antisat(x, "bandpass", 500.0, SR, res=0.6)
    tail = y[SR // 2:]
    assert 0.05 < np.max(np.abs(tail)) < 10
    k = np.argmax(np.abs(np.fft.rfft(tail)))
    assert abs(k * 2 - 500) < 15
    assert np.max(np.abs(FL.svf_antisat(x, "bandpass", 500.0, SR, res=0.3)[SR // 2:])) < 1e-6


def test_onepole_atanh_bounded_and_linear_when_small():
    x = np.full(SR // 10, 5.0)
    assert np.max(np.abs(FL.onepole_atanh(x, 1000.0, SR))) < 1.0
    s = 1e-4 * np.random.default_rng(0).standard_normal(4800)
    assert np.allclose(FL.onepole_atanh(s, 1000.0, SR), FL.onepole(s, "lowpass", 1000.0, SR), atol=1e-8)


def test_stretched_eq_repeats_every_sr_over_k():
    x = np.zeros(SR)
    x[0] = 1.0
    y = FL.stretched_eq(x, SR, gain_db=12.0)
    for f in (183.0, 183.0 + SR / 5):
        assert abs(_gain_db(y, x, SR, f) - 12) < 0.5


def test_prewarp_bounded():
    assert abs(FL.prewarp_bounded(1000.0, SR) - np.tan(np.pi * 1000 / SR)) < 1e-12
    g = FL.prewarp_bounded(20000.0, SR, 16000.0, "min")
    assert abs(g - np.tan(np.pi * 16000 / SR) * 20000 / 16000) < 1e-12
    e = 1e-3
    lo = FL.prewarp_bounded(16000.0 - e, SR, 16000.0, "linear")
    up = FL.prewarp_bounded(16000.0 + e, SR, 16000.0, "linear")
    assert abs(up - lo) < 1e-6


def test_random_allpass_keeps_magnitude():
    x = np.zeros(SR)
    x[0] = 1.0
    y = FL.random_allpass(x, SR, n=8, seed=1)
    assert np.allclose(np.abs(np.fft.rfft(y)), 1.0, atol=1e-6) and not np.allclose(y, x)


# ------------------------------------------------------------------ foley (V4_CONTACT requests)
def test_tube_amplitudes_are_free_bar_shapes():
    a = FO.tube_amplitudes(0.0)
    assert np.allclose(a, 2.0, atol=1e-3)                             # free end: |phi_n| = 2 for every mode (unit mean square)
    m = FO.tube_amplitudes(0.5)
    assert m[1] < 1e-6 and m[3] < 1e-6                                # antisymmetric modes vanish at the middle


def test_dice_contact_is_hertz_acrylic():
    from genny import contact as C
    m = C.get_material("acrylic").density * FO.DICE_EDGE ** 3
    k = C.hertz_k(2.4e9, 0.37, C.get_material("wood").young, C.get_material("wood").poisson, FO.DICE_CORNER_R)
    assert abs(FO.dice_contact_ms(1.0) - 1e3 * 2.94 * (5 * m / (4 * k)) ** 0.4) < 1e-9
    assert abs(FO.dice_contact_ms(32.0) / FO.dice_contact_ms(1.0) - 0.5) < 1e-9          # v^(-1/5)


def test_motor_fm_body_has_the_178_hz_carrier():
    rng = np.random.default_rng(0)
    sp = np.full(FO.P, 40.0)
    y = FO._motor(sp, rng, body="fm") - FO._motor(sp, np.random.default_rng(0))
    assert FO.MOTOR_FM_CARRIER == 178.0 and np.max(np.abs(y)) > 0.01


# ------------------------------------------------------------------ cross requests (V4_ESPACES)
def test_reverb_freeze_hangs():
    x = np.zeros(SR // 10)
    x[0] = 1.0
    y = FX.reverb(x, SR, mix=1.0, width=0.0, freeze=True, tail=2.0)
    z = FX.reverb(x, SR, mix=1.0, width=0.0, tail=2.0)
    late = slice(int(1.8 * SR), int(2.0 * SR))
    assert np.sqrt(np.mean(y[late] ** 2)) > 100 * np.sqrt(np.mean(z[late] ** 2))


def test_wall_leak_area_is_the_weakest_path():
    assert abs(D.leak_loss_db(60.0, 0.01) - (-10 * np.log10(0.99e-6 + 0.01))) < 1e-9
    assert abs(D.leak_loss_db(60.0, 0.01) - 20.0) < 0.01               # a 1 % gap caps a 60 dB wall near 20 dB
    x = np.random.default_rng(0).standard_normal(SR)
    a = D.wall(x, SR, material="concrete", leak_area=0.01)
    assert abs(20 * np.log10(np.std(a) / np.std(x)) + 20) < 3


def test_convolve_loop_folds_the_tail():
    x = np.random.default_rng(0).standard_normal(SR // 2)
    y = D.convolve(x, SR, kind="room", mix=1.0, loop=True)
    full = D.convolve(x, SR, kind="room", mix=1.0)
    n = len(x)
    assert len(y) == n and np.allclose(y[:200], full[:200] + full[n:n + 200])


def test_ir_spring_delay_follows_spring_dispersion():
    from genny import fdstring as FS
    assert D.SPRING_IR_COILS == ((1.0e-4, 2.5e-3, 112), (1.0e-4, 2.5e-3, 126))
    r, R, t = D.SPRING_IR_COILS[0]
    f = np.array([500.0, 3000.0])
    T = FS.spring_echo_time(f, r, R, t)
    assert T[1] > T[0]                                                # high frequencies arrive later (rising chirps)


def test_factor_body_round_trip():
    from scipy.signal import lfilter
    n = np.arange(SR // 2)
    ir = np.exp(-n / (0.3 * SR)) * np.sin(2 * np.pi * 300 * n / SR) + 0.5 * np.exp(-n / (0.2 * SR)) * np.sin(2 * np.pi * 1200 * n / SR)
    res, modes = D.factor_body(ir, SR, modes=2, r=0.9)
    assert [round(m[0] / 50) * 50 for m in modes] == [300, 1200]
    back = res
    for f, Rm in modes:
        c = np.cos(2 * np.pi * f / SR)
        back = lfilter([1, -2 * 0.9 * c, 0.81], [1, -2 * Rm * c, Rm * Rm], back)
    assert np.allclose(back, ir, atol=1e-6)
    tail = slice(SR // 4, SR // 2)
    assert np.sum(res[tail] ** 2) < 0.01 * np.sum(ir[tail] ** 2)    # the residual is short


# ------------------------------------------------------------------ passage 2A : dsp
def test_asym_shape_is_2x_above_and_half_below():
    f = D._SHAPES["asym"]
    assert abs(f(4.0) - f(3.0) - 2.0) < 0.01 and abs(f(-3.0) - f(-4.0) - 0.5) < 0.01 and f(0.0) == 0.0  # slopes 2 and 1/2


def test_tube_bendiksen_curve_and_limit():
    f = D._SHAPES["tube_bendiksen"]
    assert abs(float(f(0.0))) < 1e-12
    assert abs(float(f(-0.2)) - (1 / 8.0 + -0.2 / -np.expm1(8.0 * -0.2))) < 1e-12      # 0/0 at x = Q
    assert abs(float(f(5.0)) - 5.2 - (-0.2 / -np.expm1(-1.6))) < 1e-6                     # linear above
    assert abs(float(f(-5.0)) - (-0.2 / -np.expm1(-1.6))) < 1e-6                          # clipped below


def test_double_wall_three_regions_and_studs():
    m, d = 10.65, 0.09
    f0 = math.sqrt(1.21 * 343.0 ** 2 * 2 * m / (d * m * m)) / (2 * math.pi)
    fl = 343.0 / (2 * math.pi * d)
    single = lambda f, mm: 20 * math.log10(f * mm) - 48                 # noqa: E731
    assert abs(D.wall_loss_db(0.5 * f0, m, 200, 0, m2=m, gap=d) - single(0.5 * f0, 2 * m)) < 1e-9
    mid = D.wall_loss_db(np.array([1.2 * f0, 2.4 * f0]), m, 200, 0, m2=m, gap=d)
    assert abs(mid[1] - mid[0] - 18.06) < 0.01                          # 18 dB per octave
    assert abs(D.wall_loss_db(2 * fl, m, 200, 0, m2=m, gap=d) - (2 * single(2 * fl, m) + 6)) < 1e-9
    bridged = D.wall_loss_db(1000.0, m, 200, 3000.0, m2=m, gap=d, stud=0.6)
    assert abs(bridged - (single(1000.0, 2 * m) + 10 * math.log10(0.6 * 3000) + 20 * math.log10(0.5) - 18)) < 1e-9
    assert D.WALL_MATERIALS["drywall"]["leak"] == 52.0


def test_pqmf_reconstructs():
    x = np.random.default_rng(1).standard_normal(16000)
    sub = D.pqmf_analysis(x, 16)
    assert sub.shape[0] == 16
    y = D.pqmf_synthesis(sub, len(x))
    e = y[2000:-2000] - x[2000:-2000]
    assert 10 * np.log10(np.mean(x ** 2) / np.mean(e ** 2)) > 40


def test_ltv_fir_follows_the_control():
    x = np.random.default_rng(2).standard_normal(SR)
    lp = np.r_[np.ones(16), np.zeros(113)]
    hp = lp[::-1].copy()
    y = D.ltv_fir(x, [lp, hp], SR // 2)
    a, b = y[: SR // 2 - 400], y[SR // 2 + 400:]
    assert band_db(a, SR, 0, 4000) - band_db(a, SR, 12000, 24000) > 20
    assert band_db(b, SR, 12000, 24000) - band_db(b, SR, 0, 4000) > 20


def test_envelope_compression_floor_and_quota():
    env = np.array([1.0, 10 ** (-80.4 / 20), 10 ** (-82 / 20), 10 ** (-6.3 / 20)])
    c = D.compress_envelope(env)
    assert c[2] == 0.0 and abs(20 * np.log10(c[1]) + 80) < 1e-9 and abs(20 * np.log10(c[3]) + 6) < 1e-9
    keep, cut = D.mode_quota([0.1, 0.5, 0.3, 0.05], 3)
    assert list(keep) == [1, 2, 0] and cut == 0.25


def test_noise_band_for_fluctuation_rate():
    assert abs(D.noise_band_for_rate(4.0) - 6.25) < 1e-12 and abs(D.noise_band_for_rate(70.0) - 109.375) < 1e-9


def test_wsb_chebyshev_stages_and_tanh_fraction():
    z = np.linspace(-2, 2, 41)
    assert abs(float(D.soft_cf(1.0)) - 1 / (1 + 1 / (3 + 4 / (5 + 9 / 23)))) < 1e-12 and np.all(np.diff(D.soft_cf(z)) > 0)
    y = D.wsb(4800, 100.0, 48000)
    X = np.abs(np.fft.rfft(y))
    assert X[20] > X[40] > X[80] and X[10] < 1e-6 * X[20]       # 200 Hz > 400 Hz > 800 Hz, no 100 Hz


def test_complex_osc_preset_and_fold():
    y = SFX["complex_osc"]["fn"](sr=SR, dur=0.5, preset="olarte")
    z = SFX["complex_osc"]["fn"](sr=SR, dur=0.5)
    assert not np.allclose(y, z) and np.max(np.abs(z)) < 2 * 0.8      # fold +-0.8, then a first-order DC blocker (<= 2x)


def test_cvsd_syllabic_time_constant():
    clock = 32000
    est = D.cvsd_decode(np.ones(2000, dtype=np.uint8), clock, run=3, leak=1e9)
    step = np.diff(est)
    target = 0.001 + 0.9 * (0.15 - 0.001)
    i90 = int(np.argmax(step >= target))
    assert abs(i90 - (1 + math.log(10) * 0.005 * clock)) < 3


def test_gsm_lar_coding():
    assert abs(D.k_to_lar(0.9) - 1.125) < 1e-12 and abs(D.lar_to_k(1.125) - 0.9) < 1e-12
    assert abs(D.k_to_lar(0.97) - (8 * 0.97 - 6.375)) < 1e-12 and abs(D.lar_to_k(8 * 0.97 - 6.375) - 0.97) < 1e-12
    q = D.gsm_lar_quantize(np.array([0.5, 0.88, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert abs(q[0] - 0.5) < 1e-12                                     # round(20 * 0.5) = 10, exact
    assert abs(q[1] - (0.5 * 22 / 20 + 0.3375)) < 1e-12               # round(21.7) = 22 -> LAR 1.1
    assert q[2] == 0.0 and q[3] == 0.0                                 # offsets B = 4, -5 decode back to 0


def test_tape_wow_rate():
    sr = 8000
    t = np.arange(8 * sr) / sr
    x = 0.5 * np.sin(2 * np.pi * 500 * t)
    for rate in (0.556, 2.0):
        y = D.tape(x, sr, wow=0.5, flutter=0.0, drive=0.1, bump=0, hf=0, hiss=-120, wow_rate=rate)
        from scipy.signal import hilbert
        fi = np.diff(np.unwrap(np.angle(hilbert(y)))) * sr / (2 * np.pi)
        fi = fi[sr:-sr] - np.mean(fi[sr:-sr])
        X = np.abs(np.fft.rfft(fi))
        f = np.fft.rfftfreq(len(fi), 1 / sr)
        assert abs(f[1 + np.argmax(X[1:])] - rate) < 0.12



# ------------------------------------------------------------------ passage 2A : cyber
from genny import cyber as CY  # noqa: E402


def test_transformer_leq_follows_induction():
    assert np.allclose(CY.transformer_leq_db(np.array([1.3, 1.5, 1.9])), [65.5, 68.4, 72.4])
    assert 68.4 < float(CY.transformer_leq_db(1.7)) < 72.4


def test_gear_housing_peaks():
    h, w = CY.gear_housing("car")
    assert list(h) == [935.0, 1020.0, 1140.0] and w == [1.0, 1.0, 1.0]
    h, _ = CY.gear_housing("800, 1600")
    assert list(h) == [800.0, 1600.0]
    h, w = CY.gear_housing(1000.0)
    assert np.allclose(h, [1000, 1420, 2050, 2900]) and w[0] == 1.0


def test_soft_limit_knee():
    assert CY.soft_limit(np.array([5.0]))[0] == 5.0 and CY.soft_limit(np.array([20.0]))[0] == 20.0
    assert abs(CY.soft_limit(np.array([-25.0]))[0] + (21 - 1 / 6)) < 1e-12 and CY.soft_limit(np.array([1e9]))[0] < 21


def test_paulstretch_drift_and_repitch():
    tau = np.array([0.0, 0.25])
    assert CY.ps_drift(tau)[0] == 0.0
    want = (np.sin(2 * np.pi * CY.PHI * 0.25) + np.sin(2 * np.pi * CY.PHI ** 2 * 0.25) + np.sin(2 * np.pi * CY.PHI ** 3 * 0.25)) / 3
    assert abs(CY.ps_drift(tau)[1] - want) < 1e-12
    x = np.sin(2 * np.pi * 300 * np.arange(SR) / SR)
    y = CY.ps_repitch(x, 12)
    assert abs(len(y) - SR / 2) < 2
    X = np.abs(np.fft.rfft(y))
    assert abs(np.argmax(X) * SR / len(y) - 600) < 4
    a = CY.paulstretch(x[: SR // 2], SR, stretch=2.0, window=0.1)
    b = CY.paulstretch(x[: SR // 2], SR, stretch=2.0, window=0.1, drift=1.0)
    assert len(a) == len(b) and not np.allclose(a, b)


def test_fluctuation_strength_laws():
    assert abs(CY.fluctuation_rel(4.0) - 1.0) < 1e-12 and CY.fluctuation_rel(16.0) < 0.5
    assert abs(CY.fluctuation_bbn(1.0, 60.0, 4.0) - 5.8 * 2.0 / ((0.8) ** 2 + 1 + 1.5)) < 1e-12


def test_autocorr_zcr_on_a_sine():
    sr = 44100
    x = np.sin(2 * np.pi * 1000 * np.arange(sr // 4) / sr)
    ac, zc = CY.autocorr_zcr(x, sr)
    assert ac.shape[1] == 12 and abs(np.median(zc) - 2000) < 100
    assert abs(np.median(ac[:, 0]) - np.cos(2 * np.pi * 1000 / sr)) < 0.01


def test_adaptive_fm_index():
    x = np.sin(2 * np.pi * 220 * np.arange(SR) / SR)
    y = CY.adaptive_fm(x, SR, freq=220.0, index=2.0, ratio=1.0)
    X = np.abs(np.fft.rfft(y[SR // 10:]))
    f = np.fft.rfftfreq(len(y) - SR // 10, 1 / SR)
    side = X[np.argmin(np.abs(f - 440))]
    assert side > 0.1 * X[np.argmin(np.abs(f - 220))]
    assert np.allclose(CY.adaptive_fm(x, SR, index=0.0), x)


def test_grain_delay_is_delayed_and_stereo():
    x = np.zeros(SR)
    x[:SR // 20] = np.random.default_rng(3).standard_normal(SR // 20)
    y = CY.grain_delay(x, SR, delay=0.4, delay_spread=0.0, size=0.04, size_spread=0.0, density=200.0, mix=1.0)
    assert y.shape == (SR, 2)
    assert np.max(np.abs(y[: int(0.4 * SR)])) == 0.0 and np.max(np.abs(y[int(0.4 * SR):int(0.5 * SR)])) > 0


def test_sallen_key_is_svf_with_2r_2_minus_k():
    from genny import filters as FLT
    x = np.sin(2 * np.pi * 1000 * np.arange(SR) / SR)
    y = FLT.sallen_key(x, "lowpass", 1000.0, SR, k=1.5)
    g = 20 * np.log10(np.sqrt(np.mean(y[SR // 2:] ** 2)) / np.sqrt(np.mean(x[SR // 2:] ** 2)))
    assert abs(g - 20 * np.log10(2.0)) < 0.5                         # peak 1/(2 - k)
    assert abs(FLT.tsk_allpass_q(0.5) - 1.5) < 1e-12                 # R = 1/3


# ------------------------------------------------------------------ passage 2A : reverbs
def test_fdn_lengths_mode_density():
    from genny import reverbs as RV
    M = RV.fdn_lengths(48000, 3.0, 8, t60=2.0)
    assert M.sum() >= 0.15 * 2.0 * 48000 * 0.98
    assert np.array_equal(RV.fdn_lengths(48000, 3.0, 8), RV.fdn_lengths(48000, 3.0, 8, t60=None))


def test_greedy_filter_and_scattering_helpers():
    from genny import reverbs as RV
    tau = np.array([[1.0, 0.0], [0.0, 1.0]])
    phi = RV.greedy_filter_assign(tau, np.zeros((2, 2)), np.ones((2, 2)))
    assert list(phi) == [0, 1]
    assert RV.scattering_delays(4, 2) == [1, 4, 16] and RV.scattering_delays(4, 2, 2.0) == [1, 2, 8]
    p = RV.room_params(1000.0, 1.5)
    assert abs(p["size"] - 4 * 1000 / 600) < 1e-12 and abs(p["predelay"] - p["size"] / RV.C_AIR) < 1e-12


# ------------------------------------------------------------------ passage 2A : spectral, sfx
from genny import spectral as SP  # noqa: E402


def test_phase_rotate_correlation_is_cos_phi():
    x = np.random.default_rng(4).standard_normal(SR)
    for phi in (0.0, np.pi / 2, np.pi):
        y = SP.phase_rotate(x, phi)
        rho = np.dot(x[1000:-1000], y[1000:-1000]) / np.sqrt(np.dot(x[1000:-1000], x[1000:-1000]) * np.dot(y[1000:-1000], y[1000:-1000]))
        assert abs(rho - np.cos(phi)) < 0.05


def test_twm_picks_the_fundamental_not_an_octave():
    f = np.array([200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0])
    f0, err = SP.twm_f0(f, np.full(6, -10.0))
    assert abs(f0 - 200.0) < 1e-9


def test_laguerre_warp_moves_w0():
    w0, w1 = 0.2, 0.3
    b = SP.laguerre_b_for(w0, w1)
    assert abs(SP.laguerre_map(w0, -b) - w1) < 1e-9 or abs(SP.laguerre_map(w0, b) - w1) < 1e-9
    assert abs(SP.laguerre_map(np.pi, 0.5) - np.pi) < 1e-9 and SP.laguerre_map(0.0, 0.5) == 0.0


def test_reduce_envelope_to_eight_segments():
    t = np.linspace(0, 1, 200)
    a = np.exp(-5 * t) + 0.1 * np.sin(20 * t)
    tk, ak = SP.reduce_envelope(t, a, 8)
    assert len(tk) == 9 and tk[0] == 0.0 and tk[-1] == 1.0
    assert np.max(np.abs(np.interp(t, tk, ak) - a)) < 0.05


def test_freqshift_glide_and_spectacle_delay():
    x = np.sin(2 * np.pi * 1000 * np.arange(SR) / SR)
    y = SP.freqshift(x, SR, hz=0.0, hz_end=200.0)
    early = np.abs(np.fft.rfft(y[: SR // 10]))
    late = np.abs(np.fft.rfft(y[-SR // 10:]))
    fr = np.fft.rfftfreq(SR // 10, 1 / SR)
    assert abs(fr[np.argmax(early)] - 1010) < 30 and abs(fr[np.argmax(late)] - 1190) < 30
    imp = np.zeros(SR)
    imp[1000] = 1.0
    z = SP.spectacle(imp, SR, delay=0.5, table="ascending", feedback=0.0, mix=1.0)
    assert np.max(np.abs(z[SR // 4:])) > 0.01 * np.max(np.abs(z))


def test_riser_pulse_and_vinyl_wear_and_scratch():
    assert abs(SFX["riser"]["fn"](sr=SR, dur=1.0).std() - SFX["riser"]["fn"](sr=SR, dur=1.0, pulse=0.0).std()) == 0
    ph = __import__("genny.sfx", fromlist=["x"]).riser_pulse_phase(SR, SR)
    assert abs(ph[-1] - 14 / np.log(8)) < 0.05                      # integral of 2 * 8^t over 1 s
    v = SFX["vinyl"]["fn"](sr=SR, dur=4.0, wear=1.0)
    turn = int(1.6 * SR)
    assert np.corrcoef(v[1000:turn - 1000], v[turn + 1000:2 * turn - 1000])[0, 1] > 0.2
    b = SFX["scratch"]["fn"](sr=SR, kind="baby", dur=1.0)
    assert np.all(np.isfinite(b)) and np.max(np.abs(b)) > 0.05
