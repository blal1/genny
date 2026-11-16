import numpy as np

from genny import drums, foley, percussion
from genny.physics import MembraneDrum


def test_bass_drum_series_follows_rossing_table_4_4():
    f = np.array(percussion.BASS_DRUM_SERIES) * 39.0
    assert np.allclose(f, [39, 80, 121, 162, 204, 248])


def test_djembe_and_conga_use_measured_ratios():
    assert abs(percussion.DJEMBE_RATIOS[4] * 351 - 970) < 5
    assert abs(percussion.CONGA_RATIOS[1] * 240 - 420) < 1
    for fn in (percussion.djembe, percussion.conga, percussion.bass_drum):
        assert np.isfinite(np.asarray(fn())).all()


def test_crotales_ratios_follow_rossing_table_9_1():
    r = percussion.crotale_ratios(1480.0)
    assert np.allclose(r, [1.0, 1.42, 1.98, 2.85, 3.41, 5.19])
    assert abs(percussion.crotale_ratios(2093.0)[2] - 1.92) < 1e-6
    y = np.asarray(percussion.crotales(tune=1318.5))
    assert np.isfinite(y).all() and np.abs(y).max() > 0.1


def test_membrane_ratios_option_replaces_series():
    m = MembraneDrum(100.0, ratios=(1.0, 2.16, 3.14))._modes(0.3, "tone")
    assert np.allclose(m.freqs, [100.0, 216.0, 314.0])


def test_tom_air_and_handbell_and_crash_options_render():
    for y in (drums.tom(head="air"), foley.handbell(1046.5, 0.5, tuning="english"), foley.handbell(523.0, 0.5, tuning="dutch"),
              drums.crash(), drums.crash(series="reduced")):
        assert np.isfinite(np.asarray(y)).all() and np.abs(y).max() > 0.01
