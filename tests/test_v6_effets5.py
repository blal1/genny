"""V6_EFFETS5: remaining T_Effets lines (syrinx raven set, STK Moog, clap envelopes, VA ladders, drift, ...).

Run:  uv run --with pytest pytest tests/test_v6_effets5.py -q
"""
import math

import numpy as np

SR = 48000


def _band_energy(y, lo, hi, sr=SR):
    s = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1.0 / sr)
    return float(s[(f >= lo) & (f < hi)].sum())


# ---------------------------------------------------------------- physical/voice


def test_syrinx_raven_set():
    from genny.physical import voice as V
    r = V.SYRINX_SETS["raven"]
    assert r["kappa"] == 300.0 and r["eps"] == 10.0 and r["vol"] == 1e-6 and r["x0"] == 0.0
    assert abs(r["mass"] - math.pi / 4) < 1e-12 and r["thickness"] == 1e-4
    y = V.syrinx(np.full(int(0.3 * SR), 300.0), (200.0, 200.0), membrane="raven", second_valve=False)
    assert np.all(np.isfinite(y))
    f = V.syrinx(np.full(int(0.3 * SR), 300.0), (200.0, 200.0), membrane="fletcher", second_valve=False)
    assert not np.allclose(y, f)


def test_syrinx_fletcher_radiation():
    from genny.physical import voice as V
    ps = np.full(int(0.4 * SR), 3000.0)
    a = V.syrinx(ps, (500.0, 505.0), radiation="piston")
    b = V.syrinx(ps, (500.0, 505.0), radiation="fletcher")
    ra = _band_energy(a, 6000, 12000) / _band_energy(a, 300, 1500)
    rb = _band_energy(b, 6000, 12000) / _band_energy(b, 300, 1500)
    assert rb < 0.4 * ra
    assert V.SYRINX_RAD_ROLLOFF_HZ == 3000.0


def test_f0_track_public():
    from genny.physical import voice as V
    assert V.f0_track is V._f0_track
    t = np.arange(SR) / SR
    _, f = V.f0_track(np.sin(2 * np.pi * 220.0 * t), 60, 400)
    assert abs(np.median(f) - 220.0) < 2.0


def test_stk_moog():
    from genny.physical import voice as V
    y = V.stk_moog(220.0, 1.0, 0.8)
    assert len(y) == int(1.0 * SR) + int(0.25 * SR)
    _, f = V.f0_track(y[: SR], 60, 400)
    assert abs(np.median(f) - 220.0) < 4.0
    a, b = y[int(0.02 * SR):int(0.1 * SR)], y[int(0.7 * SR):int(0.9 * SR)]
    assert _band_energy(a, 1500, 4000) / _band_energy(a, 100, 600) > _band_energy(b, 1500, 4000) / _band_energy(b, 100, 600)
    assert V.MOOG_ADSR == (0.001, 1.5, 0.6, 0.25) and V.MOOG_VIB_HZ == 6.122


# ---------------------------------------------------------------- creatures, drums


def test_handclap_exp_rise():
    from genny import creatures as C
    assert C.CLAP_EXP_RISE == 0.0032 and C.CLAP_EXP_Q == 5.5
    fast = C.handclap(cup=0.3, seed=1)
    exp = C.handclap(cup=0.3, rise="exp", seed=1)
    pf, pe = np.argmax(np.abs(fast)), np.argmax(np.abs(exp))
    assert pe > pf
    assert np.array_equal(fast, C.handclap(cup=0.3, rise="fast", seed=1))


def test_clap_rj_bursts():
    from genny import drums as D
    e = D.vline(((0.0, 1, 0.0), (0.0, 0, 34.0)), SR, SR)
    assert abs(e[int(0.017 * SR)] - 0.5) < 1e-3
    env = D.vline(D.CLAP_RJ_EVENTS, int(0.6 * SR), SR)
    for t in (0.0, 0.023, 0.031, 0.036):
        assert env[round(t * SR)] > 0.99                         # round: int(0.036 * 48000) is 1727
    assert abs(env[int(0.0105 * SR) + int(0.0005 * SR)] - 1.0) < 0.02
    assert 0.85 < env[int(0.065 * SR)] < 0.9
    y = D.clap(SR, 1.0, model="rj")
    assert np.all(np.isfinite(y)) and len(y) == int(0.6 * SR)


def test_bird_call_raven():
    from genny import creatures as C
    y = C.bird_call(freq=200.0, pressure=0.3, membrane="raven", radiation="fletcher", dur=0.3)
    assert np.all(np.isfinite(y))


# ---------------------------------------------------------------- synths


def test_fm_bass_feedback():
    from genny import synths as S
    a = S.fm_bass(110.0, 0.3, SR)
    b = S.fm_bass(110.0, 0.3, SR, fb=1.0)
    assert len(a) == len(b) and not np.allclose(a, b)
    fc = np.full(SR // 10, 100.0)
    y = S._fb_fm(fc, fc, np.zeros(len(fc)), 0.0, float(SR))
    assert np.allclose(y, np.sin(2 * np.pi * 100.0 * np.arange(len(fc)) / SR), atol=1e-9)


def test_va_ladder_resonances():
    from scipy import signal as sg
    from genny import synths as S
    fc = 1000.0

    def mag(sos, f):
        _, h = sg.sosfreqz(sos, worN=[f], fs=SR)
        return abs(h[0])
    # lp8 near oscillation: peak at tan(pi/8) fc
    sos = S.va_ladder_zpk(fc, SR, 1.8, "lp", 8)
    fs_ = np.linspace(200, 900, 701)
    _, h = sg.sosfreqz(sos, worN=fs_, fs=SR)
    fpk = fs_[np.argmax(np.abs(h))]
    wa = 2 * SR * math.tan(math.pi * fc / SR)
    expect = SR / math.pi * math.atan(0.41421 * wa / (2 * SR))
    assert abs(fpk - expect) < 10.0
    # bp: gain at fc is 1/(2^(N/2) - k): 1/4 and 1/16, oscillation at k = 4 and 16
    for N in (4, 8):
        assert abs(mag(S.va_ladder_zpk(fc, SR, 0.0, "bp", N), fc) - 2.0 ** (-N / 2)) < 1e-3
        assert abs(mag(S.va_ladder_zpk(fc, SR, 2.0 ** (N / 2) / 2, "bp", N), fc) - 1 / (2.0 ** (N / 2) / 2)) < 2e-3
    # lp4: dc gain 1/(1+k); korg35 stable below 2
    assert abs(mag(S.va_ladder_zpk(fc, SR, 3.0, "lp", 4), 1.0) - 0.25) < 1e-3
    p = sg.sos2zpk(S.va_ladder_zpk(fc, SR, 1.9, "korg35"))[1]
    assert np.all(np.abs(p) < 1)
    x = np.random.default_rng(0).standard_normal(SR // 4)
    y = S.va_ladder(x, SR, kind="ellip", cutoff=2000.0)
    assert _band_energy(y, 4000, 20000) < 1e-6 * _band_energy(x, 4000, 20000)


def test_packet_pad_gains():
    from genny import synths as S
    assert np.allclose(S.packet_gains(0.5), S.PACKET_GAINS)
    g = S.packet_gains(0.25)
    assert np.allclose(g, [0.5, 0.333 * 0.5, 0.25 * 0.25, 0.2 * 0.5 ** 4, 0.1666 * 0.5 ** 8])
    y = S.packet_pad(220.0, 0.5, SR)
    assert np.all(np.isfinite(y))


def test_analog_drift_rms():
    from genny import synths as S
    m = S.analog_drift(4 * SR, SR, 5.0, 3)
    c = 1200 * np.log2(m)
    assert abs(np.sqrt(np.mean(c * c)) - 5.0) < 0.05
    assert np.all(S.analog_drift(100, SR, 0.0, 3) == 1.0)
    assert np.array_equal(S.soft_lead(440.0, 0.3, SR), S.soft_lead(440.0, 0.3, SR, drift=0.0))


# ---------------------------------------------------------------- matter


def test_pinch_eps():
    from genny import matter as M
    assert abs(float(M.pinch_eps(1e-3, 0.1)) - 0.1) < 1e-12
    assert float(M.pinch_eps(1e-4, 0.1)) == M.PINCH_EPS_MAX == 0.5
    assert float(M.pinch_eps(20e-3, 0.1)) == 0.01
    assert abs(float(M.excitation_amp(2e-3, "pinch", 0.1)) - 0.05 * 2.0) < 1e-12
    assert np.array_equal(M.excitation_amp(2e-3, "moss", 0.1), M.excitation_amp(2e-3))


def test_impact_energy_franz():
    from genny import matter as M
    e1 = float(M.impact_energy(1e-3, 4.0))
    assert abs(float(M.impact_energy(2e-3, 4.0)) / e1 - 8.0) < 1e-9
    assert abs(float(M.impact_energy(1e-3, 8.0)) / e1 - 32.0) < 1e-9
    K = 2.0 / 3.0 * math.pi * 1000.0 * 1e-9 * 16.0
    assert abs(e1 - 4 * math.pi * K * (4.0 / M.C_WATER) ** 3) < 1e-12 * e1 + 1e-30
    assert M.impact_band(1e-3, 4.0) == (4000.0, M.C_WATER / 1e-3)
    a = M.drip(seed=3)
    b = M.drip(seed=3, impact="franz")
    assert len(a) == len(b) and not np.allclose(a, b)
    assert np.array_equal(a, M.drip(seed=3, impact="parabolic"))


# ---------------------------------------------------------------- spectral


def test_true_envelope_accel():
    from genny import spectral as S
    f0, sr, nfft = 150.0, SR, 4096
    t = np.arange(nfft) / sr
    x = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 40) if k * f0 < 8000)
    mag = np.abs(np.fft.rfft(x * np.hanning(nfft)))
    nc = int(sr / (2 * f0))
    plain = S.true_envelope(mag, nc)
    it_plain = S.true_envelope.last_iters
    fast = S.true_envelope(mag, nc, accel=True)
    it_fast = S.true_envelope.last_iters
    assert it_fast <= it_plain
    db = 20 * np.log10(fast / mag.clip(1e-5 * mag.max()))
    assert db.min() > -2.0 - 1e-6
    band = slice(int(200 / sr * nfft), int(6000 / sr * nfft))
    assert np.median(np.abs(20 * np.log10(fast[band] / plain[band]))) < 3.0


def test_fog_grains():
    from genny import spectral as S
    env = S.fof_envelope(4800, SR, 50.0, 0.01, 0.07)
    assert env[0] == 0.0 and abs(env[480] - math.exp(-math.pi * 50.0 * 0.01)) < 1e-9 and env[-1] < 1e-3
    tab = np.sin(2 * np.pi * 440.0 * np.arange(SR) / SR)
    y = S.fog(tab, SR, dens=10.0, dur=0.1)
    assert len(y) == SR and np.all(np.isfinite(y))
    early = np.max(np.abs(y[int(0.108 * SR):int(0.112 * SR)]))
    late = np.max(np.abs(y[int(0.130 * SR):int(0.134 * SR)]))
    assert abs(late / early - math.exp(-math.pi * 50.0 * 0.022)) < 0.03
    assert S.FOG_DEFAULTS == dict(dens=80.0, trans=1.0, spd=0.0, band=50.0, ris=0.01, dec=0.07, dur=0.1)


def test_tf_pad_shift():
    from genny import spectral as S
    x = np.sin(2 * np.pi * 1000.0 * np.arange(SR) / SR)
    y = S.tf_pad(x, SR, "0:300,1:300", density=40.0, size=0.08)
    s = np.abs(np.fft.rfft(y[SR // 4: 3 * SR // 4]))
    f = np.fft.rfftfreq(SR // 2, 1 / SR)
    assert abs(f[np.argmax(s)] - 1300.0) < 10.0


# ---------------------------------------------------------------- reverbs


def test_vector_allpass_and_weights():
    from genny import reverbs as R
    A = R.weighted_matrix([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(A @ A.T, np.eye(4))
    assert np.allclose(R.feedback_matrix("weighted", 4, "1,1"), R.weighted_matrix(np.ones(4)))
    X = np.zeros((SR, 4))
    X[:2000] = np.random.default_rng(1).standard_normal((2000, 4))
    M = np.array([200, 290, 380, 470], np.int64)
    Y = R._vector_allpass(X, np.ascontiguousarray(R.hadamard_matrix(4)), M, 0.6)
    assert abs(np.sum(Y ** 2) / np.sum(X ** 2) - 1.0) < 1e-6
    x = np.zeros(SR // 2)
    x[0] = 1.0
    a = R.fdn_reverb(x, SR, diffuser="vector", tail=0.5)
    assert np.all(np.isfinite(a)) and not np.allclose(a, R.fdn_reverb(x, SR, tail=0.5))


def test_lucier_converges():
    from genny import reverbs as R
    x = np.random.default_rng(0).standard_normal(SR // 2)
    y1 = R.lucier(x, SR, 1)
    y8 = R.lucier(x, SR, 8)
    flat = lambda v: np.exp(np.mean(np.log(np.abs(np.fft.rfft(v)) + 1e-9))) / np.mean(np.abs(np.fft.rfft(v)))
    assert flat(y8) < flat(y1)
    assert len(R.lucier(x, SR, 3, chain=True)) == 3 * len(x)


# ---------------------------------------------------------------- marine, engines


def test_lap_plume():
    from genny import marine as Mr
    f0 = Mr.plume_f0(0.1)
    ring = Mr.plume_tone(SR, [0.0], 0.1)
    s = np.abs(np.fft.rfft(ring))
    assert abs(np.fft.rfftfreq(SR, 1 / SR)[np.argmax(s)] - f0) < 2.0
    assert np.array_equal(Mr.lap(dur=2.0, seed=1), Mr.lap(dur=2.0, seed=1, plume=0.0))
    assert not np.allclose(Mr.lap(dur=2.0, seed=1), Mr.lap(dur=2.0, seed=1, plume=0.1))


def test_launch_curve():
    from genny import engines as E
    r, g = E.launch_curve(3.0)
    tail = r[int(1.5 * SR):]
    assert tail.max() < E.LAUNCH_RPM + 50 and tail.min() > E.LAUNCH_RPM - E.LAUNCH_HYST - 50
    assert set(np.unique(g)) <= {1.0, E.LAUNCH_CUT}
    r0, g0 = E.limiter_curve(8000.0, 1.0)
    assert set(np.unique(g0)) <= {0.0, 1.0}
    assert "launch_control" in E._EVENTS


# ---------------------------------------------------------------- acoustic


def test_regalia_mitra_peak_and_clav_beat():
    from genny import acoustic as A
    x = np.random.default_rng(0).standard_normal(SR)
    y = A.regalia_mitra(x, SR, 1000.0, 10 ** (12 / 20), 100.0)
    f = np.fft.rfftfreq(SR, 1 / SR)
    r = np.abs(np.fft.rfft(y)) / np.abs(np.fft.rfft(x))
    assert abs(20 * np.log10(np.median(r[(f > 995) & (f < 1005)])) - 12.0) < 0.5
    assert abs(20 * np.log10(np.median(r[(f > 290) & (f < 310)]))) < 0.5
    c0 = A.clav(110.0, 1.0, SR, 0.8, model="dwg")
    assert np.array_equal(c0, A.clav(110.0, 1.0, SR, 0.8, model="dwg", beat=0.0))
    assert not np.allclose(c0, A.clav(110.0, 1.0, SR, 0.8, model="dwg", beat=12.0))
    assert np.array_equal(A.clav(440.0, 0.5, SR, 0.8, beat=12.0), A.clav(440.0, 0.5, SR, 0.8))


def test_ripple_rrate_and_harpsichord_partial():
    from genny import acoustic as A
    assert A.ripple_rrate(6, -0.0015) == 1.0 / 12.0 and A.ripple_rrate(6, 0.001) == 1.0 / 6.0
    h0 = A.harpsichord(196.0, 0.5, SR, 0.8)
    assert np.array_equal(h0, A.harpsichord(196.0, 0.5, SR, 0.8, ripple_partial=6))
    assert not np.allclose(h0, A.harpsichord(196.0, 0.5, SR, 0.8, ripple_partial=3))


def test_dist_guitar_loop():
    from genny import acoustic as A
    y = A.dist_loop(110.0, SR, SR, 6.0)
    assert np.isfinite(y).all() and np.std(y[-2000:]) < 0.2 * np.max(np.abs(y))
    s = np.abs(np.fft.rfft(y[: SR // 2]))
    f = np.fft.rfftfreq(SR // 2, 1 / SR)
    assert abs(f[np.argmax(s * (f < 200))] - 110.0) < 4.0
    d0 = A.dist_guitar(110.0, 0.5, SR, 0.8)
    assert np.array_equal(d0, A.dist_guitar(110.0, 0.5, SR, 0.8, loop=0))
    d1 = A.dist_guitar(110.0, 0.5, SR, 0.8, loop=1)
    assert d1.shape == d0.shape and np.isfinite(d1).all()


def test_loss_onepole_closed_form():
    from genny import acoustic as A
    f0, tau0, fh = 220.0, 4.0, 3000.0
    g, a1 = A.loss_onepole(f0, SR, tau0, fh)
    for f, tau in ((f0, tau0), (fh, tau0 / 2)):
        H = abs(g * (1 + a1) / (1 + a1 * np.exp(-2j * np.pi * f / SR)))
        assert abs(-1 / (f0 * math.log(H)) / tau - 1) < 0.05


def test_dispersion_allpass_from_partials():
    from genny import acoustic as A
    for f0, B, tol in ((220.0, 2e-5, 1.0), (110.0, 1e-4, 16.0)):
        k = np.arange(1, 31 if B > 5e-5 else 16)
        fk = k * f0 * np.sqrt(1 + B * k * k)
        N, a = A.dispersion_allpass_from_partials(fk, SR, 5)
        th = 2 * np.pi * fk / SR
        ph = -5 * th + 2 * np.sum(np.arctan2(np.outer(a, np.sin(th)), 1 + np.outer(a, np.cos(th))), axis=0)
        fit = k * SR / ((N * th - ph) / th)
        assert np.max(np.abs(1200 * np.log2(fit / fk))) < tol


def test_tension_glide_cents():
    from genny import acoustic as A
    assert abs(A.tension_glide_cents(2e-3, 1 / 3, 0.65) - 7.0) < 0.2
    assert abs(A.tension_glide_cents(4e-3, 1 / 3, 0.65) - 28.0) < 0.5
    assert np.array_equal(A.guitar(147.0, 0.4, SR, 1.0), A.guitar(147.0, 0.4, SR, 1.0, glide_mm=0.0))


def test_flute_novice_and_body_am():
    from genny import acoustic as A
    assert A.novice_reynolds(261.63) == 3000.0 and A.novice_reynolds(2093.0) == 7000.0
    assert A.flute_noise_factor(2000.0) == 1.0 and abs(A.flute_noise_factor(5000.0) - 2.0) < 1e-12
    c0 = A.cello(220.0, 0.6, SR, 0.8)
    assert np.array_equal(c0, A.cello(220.0, 0.6, SR, 0.8, body_am=0))
    assert not np.allclose(c0, A.cello(220.0, 0.6, SR, 0.8, body_am=1))


# ---------------------------------------------------------------- fdstring, plates, klang, bank_piano


def test_rattle_gravity():
    from genny import fdstring as FD
    quiet = FD.StiffString(110.0, SR).rattle(0.4, eps=1e-4, mass_ratio=10.0).run(4000)
    assert np.max(np.abs(quiet)) == 0.0
    fall = FD.StiffString(110.0, SR).rattle(0.4, eps=1e-4, mass_ratio=10.0, gravity=9.81 / 0.65).run(4000)
    assert np.isfinite(fall).all() and np.max(np.abs(fall)) > 0.0


def test_apnl_lossless_and_wgr_quadrature():
    from genny import plates as Pl
    x = np.random.default_rng(1).standard_normal(4000)
    xx = np.concatenate([x, np.zeros(4000)])
    assert abs(np.sum(Pl.apnl(xx, 0.6, 0.6) ** 2) / np.sum(x ** 2) - 1.0) < 1e-6
    y = Pl.apnl(xx, 0.6, -0.3)
    assert np.isfinite(y).all() and abs(np.sum(y ** 2) / np.sum(x ** 2) - 1.0) < 0.1 and not np.allclose(y, Pl.apnl(xx, 0.6, 0.6))
    c, s = Pl.wgr(440.0, SR, SR)
    assert abs(np.max(np.abs(c)) - 1.0) < 1e-3 and abs(np.max(np.abs(s)) - 1.0) < 1e-3
    assert abs(np.mean(c * s)) < 1e-3


def test_cauchy_waveshaper():
    from genny import klang as K
    G, H = K.cauchy_gh(1.0)
    assert abs(20 * math.log10(H) + 15.31) < 0.01 and abs(G - 2 / math.sqrt(2)) < 1e-12
    assert abs(20 * math.log10(K.cauchy_gh(4.0)[1]) + 4.30) < 0.01
    f0 = 375.0
    y = K.cauchy_wave(f0, SR, SR, 2.0)
    s = np.abs(np.fft.rfft(y)) / (SR / 2)
    a1, a2 = s[int(f0)], s[int(2 * f0)]
    assert abs(a2 / a1 - K.cauchy_gh(2.0)[1]) < 1e-3 and abs(np.mean(y)) < 1e-6


def test_bank_piano_air_vc_and_guard():
    from genny.physical import bank_piano as BP
    s = BP.air_rate_vc(np.array([100.0, 1000.0]), 1e-3, 7850.0)
    assert np.all(s > 0) and s[1] > s[0]
    assert BP.BANK_COND_MAX == 1e6
