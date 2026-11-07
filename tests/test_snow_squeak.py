"""Snow squeak spectrum after Arabadzhi & Rudik (Priroda 1963 no. 10 p. 106)."""
import numpy as np

from genny.physical import footsteps as S
from genny.physical.core import SR


def _band_db(y, lo, hi):
    p = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1.0 / SR)
    return 10 * np.log10(p[(f >= lo) & (f < hi)].sum() + 1e-30)


def test_shape_follows_temperature():
    x = np.random.default_rng(0).standard_normal(SR)
    warm, cold = S.snow_squeak_shape(x, -1.5), S.snow_squeak_shape(x, -15.0)
    # -1.5 C: maximum near 300 Hz, little above 1 kHz
    assert _band_db(warm, 280, 355) - _band_db(warm, 2800, 3550) > 10   # table: 8.2 vs -5.2 dB
    # -15 C: the 3 kHz band is close to the 300 Hz one (new weak maximum)
    assert abs(_band_db(cold, 280, 355) - _band_db(cold, 2800, 3550)) < 6
    # interpolation is monotonic in brightness
    mid = S.snow_squeak_shape(x, -8.0)
    hf = [_band_db(y, 2000, 8000) - _band_db(y, 200, 500) for y in (warm, mid, cold)]
    assert hf[0] < hf[1] < hf[2]


def test_step_squeak_layer():
    _, info = S.render_step("snow", 0, np.random.default_rng(1), return_info=True, temperature=-12.0)
    assert "squeak" in info["layers"]
    _, info = S.render_step("snow", 0, np.random.default_rng(1), return_info=True, temperature=1.0)
    assert "squeak" not in info["layers"]


def test_snow_ground_dip_is_lower_than_grass():
    from genny.physical import space
    f = np.geomspace(40.0, 2000.0, 400)
    dip = {s: f[np.argmin(np.abs(space.ground_effect(f, 50.0, sigma=s)))] for s in (10.0, 300.0)}
    # softer ground (Albert 1990: snow 10-35 kN s m^-4) moves the ground dip to low frequencies
    assert dip[10.0] < 200.0 < 300.0 < dip[300.0]


def test_miki_matches_delany_bazley_in_its_range():
    from genny.physical import space
    f = np.geomspace(10.0, 1000.0, 50) * 300.0                     # x = f/sigma over Delany-Bazley's fitted range
    m, d = space.ground_impedance(f, 300.0), space.ground_impedance(f, 300.0, "delany")
    assert np.all(np.abs(m - d) / np.abs(d) < 0.25)


def test_piano_stretch_curve():
    from genny.fdstring import _stretch_cents
    assert _stretch_cents(440.0) == 0.0
    assert abs(_stretch_cents(55.0) - 5.0) < 1e-9 and abs(_stretch_cents(3520.0) - 20.0) < 1e-9


def test_air_absorption_matches_iso_9613_2_table():
    # ISO 9613-2 Table 2 (recalled from memory), dB/km at 15 C / 50 %: octave values agree with the pure-tone formula
    from genny.physical.space import air_db_per_m
    got = air_db_per_m(np.array([250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]), 50, 15.0) * 1000
    assert np.all(np.abs(got / np.array([1.2, 2.2, 4.2, 10.8, 36.2, 129.0]) - 1) < 0.03)


def test_ground_presets_render():
    from genny.reverbs import distance
    x = np.random.default_rng(0).standard_normal(4800)
    forest = distance(x, 48000, metres=50.0, ground="forest")
    asphalt = distance(x, 48000, metres=50.0, ground="asphalt")
    assert np.isfinite(forest).all() and np.isfinite(asphalt).all() and not np.allclose(forest, asphalt)
