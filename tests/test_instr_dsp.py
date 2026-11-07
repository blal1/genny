"""Instruments and DSP, values from the documents downloaded 2026-11-05 (sources: research/SOURCES.md).

Run:  uv run --with pytest pytest tests/test_instr_dsp.py -q
"""
import math

import numpy as np

from genny import filters as F

SR = 44100


def test_phasor_reson_impulse_response_is_r_pow_n_sin():
    f, tau = 1000.0, 0.05
    y = F.phasor_reson(np.r_[1.0, np.zeros(2000)], f, tau, SR)
    n = np.arange(y.size)
    r, th = math.exp(-1 / (tau * SR)), 2 * math.pi * f / SR
    assert np.max(np.abs(y - r ** n * np.sin(n * th))) < 1e-9


def test_phasor_reson_retune_keeps_amplitude():
    n = 8000
    f = np.where(np.arange(n) < n // 2, 300.0, 1700.0)
    y = F.phasor_reson(np.r_[1.0, np.zeros(n - 1)], f, np.inf, SR)
    a, b = np.max(np.abs(y[n // 2 - 600:n // 2])), np.max(np.abs(y[n // 2 + 10:n // 2 + 600]))
    assert abs(a - 1.0) < 1e-3 and abs(b - 1.0) < 1e-3, (a, b)


def test_plectrum_release_follows_perng_table_2():
    from genny.acoustic import plectrum_release
    table = ((1 / 2, 2.2236), (1 / 3, 1.8708), (1 / 4, 1.6858), (1 / 5, 1.5285), (1 / 6, 1.3719), (1 / 7, 1.2636))
    amp = [math.hypot(*plectrum_release(b)[1:]) * 1e3 for b, _ in table]
    assert all(a > b for a, b in zip(amp, amp[1:])), amp
    for a, (_, mm) in zip(amp, table):
        assert abs(a / mm - 1) < 0.2, (a, mm)
    assert abs((amp[0] / amp[-1]) / (2.2236 / 1.2636) - 1) < 0.12   # quasi-static 1.56 against 1.76 at 1 m/s


def test_harpsichord_pluck_near_nut_is_softer():
    from genny.acoustic import harpsichord
    a = np.max(np.abs(harpsichord(220.0, 0.3, SR, soundboard=0.0, pluck=0.5)))
    b = np.max(np.abs(harpsichord(220.0, 0.3, SR, soundboard=0.0, pluck=1 / 7)))
    assert 1.4 < a / b < 1.9, a / b


def _poles(fs, qs):
    fs, qs = np.asarray(fs, float), np.asarray(qs, float)
    return np.exp(-np.pi * fs / (qs * SR)) * np.exp(2j * np.pi * fs / SR)


def test_admittance_reflectance_is_y_minus_y0_over_y_plus_y0():
    from genny.strings import admittance_reflect
    p, w = _poles([110.0, 250.0, 900.0], [30.0, 40.0, 50.0]), np.array([0.3, 1.0, 0.2])
    h = admittance_reflect(np.r_[1.0, np.zeros(4000)][:, None], p, w[:, None, None], [[0.5]])[:, 0]
    from numpy.polynomial import polynomial as P
    from scipy.signal import lfilter
    dens = [np.array([1.0, -2 * pr.real, abs(pr) ** 2]) for pr in p]
    D = np.array([1.0])
    for d in dens:
        D = P.polymul(D, d)
    N = np.zeros(1)
    for k, wr in enumerate(w):
        t = wr * np.array([1.0, 0.0, -1.0])
        for j, d in enumerate(dens):
            if j != k:
                t = P.polymul(t, d)
        N = P.polyadd(N, t)
    ref = lfilter(P.polysub(N, 0.5 * D), P.polyadd(N, 0.5 * D), np.r_[1.0, np.zeros(h.size - 1)])
    assert np.max(np.abs(h - ref)) < 1e-9


def test_admittance_matrix_bridge_is_passive():
    from genny.strings import admittance_reflect
    rng = np.random.default_rng(3)
    p = _poles(rng.uniform(80, 3000, 30), rng.uniform(20, 80, 30))
    phi = rng.standard_normal((30, 2, 1))
    Yr = phi @ phi.transpose(0, 2, 1) * 0.05
    vp = rng.standard_normal((20000, 2))
    vm = admittance_reflect(vp, p, Yr, np.diag([1.0, 1.0]))
    assert np.sum(vm ** 2) < np.sum(vp ** 2)
    x = admittance_reflect(np.c_[np.r_[1.0, np.zeros(3000)], np.zeros(3001)], p, Yr, np.eye(2))
    assert np.max(np.abs(x[1:, 1])) > 1e-3                   # a z pluck reaches y through the off-diagonal terms


def test_pluck_scattering_is_structurally_passive():
    from scipy.signal import freqz
    from genny.strings import pluck_scattering
    rng = np.random.default_rng(1)
    for _ in range(50):
        M, R, K = rng.uniform(0, 0.02), rng.uniform(0, 5.0), rng.uniform(0, 5e4)
        b, a = pluck_scattering(M, R, K, 0.6, SR)
        _, h = freqz(b, a, 4096)
        assert np.max(np.abs(h)) <= 1 + 1e-9 and np.all(np.abs(np.roots(a)) < 1)
    _, h = freqz(*pluck_scattering(0.005, 0.0, 2e4, 0.6, SR), 512)
    assert np.allclose(np.abs(h), 1.0)                       # no damper: lossless junction
    b, a = pluck_scattering(0.0, 0.0, 0.0, 0.6, SR)
    assert np.allclose(b, a)                                 # finger away: Q = 1, S = identity


def test_tr808_bridged_t_resonators():
    from genny.drums import bridged_t
    fc, q, _ = bridged_t("bass_drum")
    assert abs(fc - 45.74) < 0.05 and abs(q - 2.156) < 0.005
    assert abs(bridged_t("snare_low")[0] - 249.6) < 0.2 and abs(bridged_t("snare_high")[0] - 499.0) < 0.5
    assert abs(bridged_t("clave")[0] - 2526) < 3 and abs(bridged_t("rimshot")[0] - 452.5) < 0.5


def test_snare808_tones_sit_on_the_bridged_t_pair():
    from genny.percussion import snare808
    y = snare808(SR, snappy=0.0)[: SR // 10]
    s = np.abs(np.fft.rfft(y * np.hanning(y.size), 1 << 16))
    f = np.fft.rfftfreq(1 << 16, 1 / SR)
    lo = f[np.argmax(s * (f < 350))]
    hi = f[np.argmax(s * ((f > 400) & (f < 650)))]
    assert abs(lo - 249.6) < 3 and abs(hi - 499.0) < 5, (lo, hi)


def test_daxophone_strip_and_pitch():
    from genny.friction import DAX_STRIP, dax_free_length, daxophone
    assert DAX_STRIP == (0.330, 0.030, 0.005)
    L = dax_free_length(220.0)
    assert 0.05 < L < DAX_STRIP[0]
    assert abs(dax_free_length(55.0) / L - 2.0) < 1e-9          # f ~ 1 / L^2
    y = daxophone(220.0, 0.6, SR)
    seg = y[int(0.25 * SR):int(0.55 * SR)]
    s = np.abs(np.fft.rfft(seg * np.hanning(seg.size), 1 << 16))
    f = np.fft.rfftfreq(1 << 16, 1 / SR)
    assert abs(f[np.argmax(s)] / 220.0 - 1) < 0.02
    y = daxophone(220.0, 0.6, SR, bow=0.3)                      # bowed near the dax: the tone toggles to the next mode
    seg = y[int(0.25 * SR):int(0.55 * SR)]
    s = np.abs(np.fft.rfft(seg * np.hanning(seg.size), 1 << 16))
    assert abs(f[np.argmax(s)] / 220.0 / 6.267 - 1) < 0.05


def test_ren_appendix_b_match_score():
    from genny.physical.modal import ren_features, ren_match_ratio, ren_match_score
    assert np.allclose(ren_match_score([0.0, 0.5, 1.0, 2.0, 5.0, 5.01]), [1.0, 0.75, 0.5, 0.25, 0.1, 0.0])
    assert np.allclose(ren_features(600.0, 50.0), [5.0 * 6.0 * np.arcsinh(1.0), 2.0])
    modes = [(440.0, 20.0), (1200.0, 60.0)]
    assert ren_match_ratio(modes, modes) == 1.0
    assert ren_match_ratio(modes, [(5000.0, 400.0)]) == 0.0


def test_thoret_additive_friction_source():
    from genny.friction import squeak, thoret_friction
    e = thoret_friction(200.0, 1.0, SR, eps=0.0)                  # self-sustained: harmonics 1/k on 200 Hz
    s = np.abs(np.fft.rfft(e)) / (SR / 2)
    assert abs(s[200] - 1.0) < 1e-3 and abs(s[600] - 1 / 3) < 1e-3 and abs(s[1000] - 0.2) < 1e-3
    a = thoret_friction(200.0, 1.0, SR, eps=1.0, wander=0.1, seed=2)   # squeaking: the fundamental wanders
    assert np.max(np.abs(np.fft.rfft(a))) < 0.7 * np.max(np.abs(np.fft.rfft(e)))
    g = thoret_friction(200.0, 1.0, SR, eps=0.0, diameter=0.08, speed=0.2)   # singing glass: beats at v / D = 2.5 Hz
    env = np.abs(np.fft.rfft(np.abs(g) - np.mean(np.abs(g))))
    assert abs(np.argmax(env[1:20]) + 1 - 5) <= 1                # |sin| of 2.5 Hz -> 5 Hz envelope
    y = squeak(SR, model="additive")
    assert y.size == int(0.6 * SR) and np.max(np.abs(y)) > 0.3


def test_car_cabin_gain_rises_12_db_per_octave_below_first_mode():
    from scipy.signal import freqz
    from genny import dsp as D
    h = D.cabin_gain(np.r_[1.0, np.zeros(1 << 17)], SR)
    f, H = freqz(h, 1, [5.0, 20.0, 40.0, 1000.0], fs=SR)
    db = 20 * np.log10(np.abs(H))
    assert abs(db[0] - D.CABIN_GAIN_DB) < 0.5 and abs(db[3]) < 0.1
    up = D.cabin_gain(np.r_[1.0, np.zeros(1 << 17)], SR, gain_db=40.0)   # plateau far below: the slope
    _, H = freqz(up, 1, [15.0, 30.0], fs=SR)
    assert abs(20 * np.log10(abs(H[0]) / abs(H[1])) - 12.0) < 0.5


def test_wall_coincidence_follows_cremer_law():
    from genny import dsp as D
    m, fc = 10.0, 3800.0
    field = lambda f: 20 * np.log10(f * m) - 48.0                  # noqa: E731
    dip = field(fc) - D.wall_loss_db(fc, m, 200.0, fc)
    assert abs(dip - (-6.0 - 10 * np.log10(2 * D.WALL_ETA * 0.2316 / np.pi))) < 1e-3   # 17.5 dB at eta 0.03
    up = D.wall_loss_db(4 * fc, m, 200.0, fc) - D.wall_loss_db(2 * fc, m, 200.0, fc)
    assert abs(up - (20 * np.log10(2) + 10 * np.log10(3))) < 1e-9  # (f / fc - 1): 10.8 dB from 2 fc to 4 fc
    r0 = lambda f: 20 * np.log10(f * m) - 42.0                     # noqa: E731
    assert abs(D.wall_loss_db(3 * fc, m, 200.0, fc) - (r0(3 * fc) + 10 * np.log10(2 * D.WALL_ETA / np.pi * 2))) < 1e-9
    assert abs(D.wall_loss_db(0.5 * fc, m, 200.0, fc) - field(0.5 * fc)) < 1e-9
