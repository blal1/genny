import numpy as np

from genny import designed, road


def test_morse_accented_e_and_multiplication_sign():
    assert designed.MORSE["É"] == "..-.."
    assert designed.MORSE["×"] == designed.MORSE["X"]
    assert designed.morse_units("é")[0] == (True, 1)


def test_ev_whine_carrier_pattern_is_periodic_and_default_unchanged():
    a = road.ev_whine(3000.0, fs=7500.0, dur=0.2, seed=1)
    b = road.ev_whine(3000.0, fs=7500.0, dur=0.2, seed=1, fs_pattern=None)
    assert np.array_equal(a, b)
    p = road.ev_whine(3000.0, dur=0.2, seed=1, fs_pattern=(3448.0, 6667.0), pattern_len=200)
    assert p.shape == a.shape and np.all(np.isfinite(p)) and np.max(np.abs(p)) < 1.0
