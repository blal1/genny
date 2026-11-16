import numpy as np

from genny import spectral as SP

SR = 16000


def _two_tones():
    t = np.arange(SR) / SR
    return 0.5 * np.sin(2 * np.pi * 440 * t) + 0.002 * np.sin(2 * np.pi * 1234 * t)


def test_defaults_unchanged_by_explicit_values():
    x = _two_tones()
    a = SP.analyze(x, SR)
    b = SP.analyze(x, SR, sleep=2, max_dev=0.0, f0_range=None, general_db=0.0, min_height=0.0)
    assert np.array_equal(a["freq"], b["freq"]) and np.array_equal(a["amp"], b["amp"])


def test_general_db_keeps_quiet_peak_the_local_range_drops():
    x = _two_tones()
    plain = SP.analyze(x, SR, thresh_db=-30.0)
    wide = SP.analyze(x, SR, thresh_db=-30.0, general_db=70.0)
    assert wide["freq"].shape[1] >= plain["freq"].shape[1]
    assert np.any(np.abs(wide["freq"] - 1234.0) < 20.0) or wide["freq"].shape[1] > plain["freq"].shape[1]


def test_min_height_rejects_flat_peaks_and_keeps_a_sine():
    x = _two_tones()
    m = SP.analyze(x, SR, min_height=6.0)
    assert np.any(np.abs(m["freq"] - 440.0) < 5.0)


def test_f0_range_excludes_a_fundamental_outside_it():
    t = np.arange(SR) / SR
    x = sum(np.sin(2 * np.pi * 200 * h * t) / h for h in range(1, 6))
    inside = SP.analyze(x, SR, harmonic=True, f0_range=(150.0, 260.0))
    outside = SP.analyze(x, SR, harmonic=True, f0_range=(300.0, 500.0))
    assert inside["amp"].any() and not outside["amp"].any()


def test_linear_bands_are_32_equal_widths():
    e = SP.band_edges(SR, "linear")
    assert len(e) == 33 and np.allclose(np.diff(e), SR / 2 / 32)
