import numpy as np

from genny import analysis, choir, designed, foley, heavy_vehicles, ordnance, percussion, spectral


def _ok(y):
    return np.all(np.isfinite(y)) and len(y) > 0


def test_underwater_blast_bottom_option():
    a = ordnance.underwater_blast(50.0, 200.0, 30.0, np.random.default_rng(1))
    b = ordnance.underwater_blast(50.0, 200.0, 30.0, np.random.default_rng(1), bottom=False)
    c = ordnance.underwater_blast(50.0, 200.0, 30.0, np.random.default_rng(1), bottom=True)
    assert np.array_equal(a, b) and _ok(c) and not np.array_equal(a, c)


def test_gongs_amp_over_h_hardens_free_disc():
    a = percussion.gong(vel=0.8, decay=1.0)
    b = percussion.gong(vel=0.8, decay=1.0, amp_over_h=0.0)
    c = percussion.gong(vel=0.8, decay=1.0, amp_over_h=1.0)
    assert np.array_equal(a, b) and _ok(c) and not np.array_equal(a, c)


def test_gunshot_temp_moves_the_blast_lag():
    cold = foley.gunshot(kind="rifle", distance=300.0, temp=-20.0, seed=0)
    hot = foley.gunshot(kind="rifle", distance=300.0, temp=40.0, seed=0)
    assert _ok(cold) and _ok(hot) and len(cold) != len(hot)
    assert np.array_equal(foley.gunshot(seed=0), foley.gunshot(seed=0, temp=None))


def test_morse_prosign():
    assert designed.morse_units("<AR>") == [(True, 1), (False, 1), (True, 3), (False, 1), (True, 1), (False, 1), (True, 3), (False, 1), (True, 1)]
    assert designed.morse_units("SOS") == designed.morse_units("sos")


def test_yin_local_best_estimate():
    sr = 8000
    t = np.arange(int(0.5 * sr)) / sr
    y = np.sin(2 * np.pi * 200.0 * t) + 0.3 * np.sin(2 * np.pi * 400.0 * t)
    a, b = analysis.yin(y, sr), analysis.yin(y, sr, local=True)
    assert a.shape == b.shape and abs(np.median(b) - 200.0) < 3.0


def test_voice_ra_bandwidth_option():
    a = choir.voice(220.0, 0.3, 22050, seed=1)
    b = choir.voice(220.0, 0.3, 22050, seed=1, ra_bw=True)
    assert _ok(b) and a.shape == b.shape and not np.array_equal(a, b)


def test_tracks_irregular_and_stagger():
    a = heavy_vehicles.tracks(kind="snowmobile", dur=0.5, seed=0)
    b = heavy_vehicles.tracks(kind="snowmobile", dur=0.5, seed=0, irregular=False, stagger=False)
    c = heavy_vehicles.tracks(kind="snowmobile", dur=0.5, seed=0, irregular=True)
    d = heavy_vehicles.tracks(kind="snowmobile", dur=0.5, seed=0, stagger=True)
    assert np.array_equal(a, b) and _ok(c) and _ok(d) and not np.array_equal(a, c) and not np.array_equal(a, d)


def test_analyze_kaiser_and_time_residual():
    sr = 8000
    t = np.arange(sr // 2) / sr
    x = np.sin(2 * np.pi * 440.0 * t) + 0.01 * np.random.default_rng(0).standard_normal(t.size)
    m0 = spectral.analyze(x, sr)
    m1 = spectral.analyze(x, sr, kaiser_beta=2.8, residual="time")
    assert m1["freq"].shape[0] == m0["freq"].shape[0] and np.all(np.isfinite(m1["noise"]))
    assert abs(spectral.window(64, "kaiser", beta=0.0)[5] - 1.0) < 1e-12


def test_cross_synth_balance():
    sr = 8000
    x = np.random.default_rng(0).standard_normal(sr // 2) * np.linspace(0.1, 1.0, sr // 2)
    a = spectral.cross_synth(x, sr)
    b = spectral.cross_synth(x, sr, balance=0.0)
    c = spectral.cross_synth(x, sr, balance=1.0)
    assert np.array_equal(a, b) and np.all(np.isfinite(c)) and not np.array_equal(a, c)
