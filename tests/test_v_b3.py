import numpy as np

from genny import psychoacoustics_p2 as P

SR = 48000


def _tone(f, level, dur=1.0):
    t = np.arange(int(dur * SR)) / SR
    return np.sqrt(2.0) * 2e-5 * 10.0 ** (level / 20.0) * np.sin(2 * np.pi * f * t)


def test_mg_one_sone_at_1k_40db():
    assert abs(P.moore_glasberg_loudness(_tone(1000.0, 40.0), SR)[0] - 1.0) < 0.02


def test_mg_low_frequency_equal_loudness():
    n100 = P.moore_glasberg_loudness(_tone(100.0, 64.1), SR)[0]
    n63 = P.moore_glasberg_loudness(_tone(63.0, 70.0), SR)[0]
    assert 0.9 < n100 < 1.25
    assert 0.6 < n63 < 1.0


def test_mg_tables_published_values():
    assert abs(P._ethrq_db(np.array([100.0, 1000.0]))[0] - 15.68) < 1e-6
    assert abs(P._ethrq_db(np.array([1000.0]))[0] - 3.63) < 1e-6
    assert abs(np.interp(-24.54531, P.MG_GA, P.MG_A_TAB) - 8.852) < 1e-6
    assert abs(np.interp(-25.0, P.MG_ALPHA_G, P.MG_ALPHA_V) - 0.26692) < 1e-6


def test_two_stroke_chamber_diffuser_area_ratio():
    from genny import cyber as c
    d, r, g = c.tuned_chamber(11500, 196.0, 0.038)
    area = np.concatenate([[1.0], np.cumprod((1.0 - r) / (1.0 + r))])
    assert abs(area.max() - 6.25) < 0.05
    assert abs(c.tuned_length(8000, 190.0, 800.0) * 39.37 - 190.0 * 1700.0 / 8000.0) < 2.0
