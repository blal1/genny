import numpy as np

from genny import spectral as S

SR = 44100


def test_bend_inertance_and_sign_change():
    from genny import tubes as T
    assert abs(T.BEND_BETA - 0.5) < 0.01
    assert T.bend_inertance(0.0) == 1.0
    assert abs(T.bend_inertance(0.5) - 0.9330) < 1e-3
    assert T.bend_inertance(1.0) < T.bend_inertance(0.5) < 1.0
    a, lb = 0.0175, 4.0 * np.pi / 3.0 * 0.0175
    assert T.bend_length_correction(lb, 0.5) < 0.0
    fs = T.bend_sign_change_frequency(a, 7.0 / 9.0)
    fc = 1.8412 * T.C_AIR / (2.0 * np.pi * a)
    assert 0.47 * fc < fs < 0.5 * fc
    assert abs(fc - 5760.0) < 60.0


def test_bubble_peak_pressure_one_mm_at_one_metre():
    from genny import matter as M
    p = float(M.bubble_peak_pressure(1e-3))
    assert 0.1 < p < 0.2
    assert abs(float(M.bubble_peak_pressure(1e-3, r=2.0)) - p / 2.0) < 1e-12
    assert float(M.bubble_peak_pressure(2e-3)) > p


def test_periodicity_gauss_window_tracks_snr():
    from genny import analysis as A
    t = np.arange(SR) / SR
    x = np.sin(2 * np.pi * 150 * t) + 0.5 * np.sin(2 * np.pi * 300 * t)
    rng = np.random.default_rng(0)
    for snr in (10.0, 20.0):
        noise = rng.standard_normal(SR) * np.sqrt(np.mean(x ** 2)) * 10 ** (-snr / 20)
        p = A.periodicity(x + noise, SR, window="gauss")
        assert abs(p["hnr"] - snr) < 1.5, (snr, p["hnr"])
        assert abs(p["f0"] - 150.0) < 5.0


def test_hann_default_six_periods_gauss_four_and_half():
    from genny import analysis as A
    t = np.arange(SR) / SR
    x = np.sin(2 * np.pi * 150 * t) + 0.5 * np.sin(2 * np.pi * 300 * t)
    a =A.periodicity(x, SR)["frames_hnr"]
    b = A.periodicity(x, SR, periods=6.0)["frames_hnr"]
    assert a.shape == b.shape and np.allclose(a, b, equal_nan=True)
    assert A.periodicity(x, SR, window="gauss")["frames_hnr"].shape != a.shape


def test_bubble_surface_tension_term_haris_eq5():
    from genny import matter as M
    r = 50e-6
    w0 = float(M.bubble_params(r)[0])
    ref = np.sqrt(3 * M.GAMMA * (M.P_ATM + 2 * M.SIGMA_W / r) - 2 * M.SIGMA_W / r) / (r * np.sqrt(M.RHO_W))
    assert abs(w0 - ref) < 1e-9 * ref
    old = np.sqrt(3 * M.GAMMA * M.P_ATM - 2 * M.SIGMA_W / r) / (r * np.sqrt(M.RHO_W))
    assert 0.005 < w0 / old - 1 < 0.03


def test_true_envelope_accel_cepstral_energy_ratio():
    f0, nfft = 150.0, 4096
    t = np.arange(nfft) / SR
    x = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 40) if k * f0 < 8000)
    mag = np.abs(np.fft.rfft(x * np.hanning(nfft)))
    nc = int(SR / (2 * f0))
    plain = S.true_envelope(mag, nc)
    fast = S.true_envelope(mag, nc, accel=True)
    assert S.true_envelope.last_iters <= 200
    db = 20 * np.log10(fast / mag.clip(1e-5 * mag.max()))
    assert db.min() > -2.0 - 1e-6
    band = slice(int(200 / SR * nfft), int(6000 / SR * nfft))
    assert np.median(np.abs(20 * np.log10(fast[band] / plain[band]))) < 3.0
