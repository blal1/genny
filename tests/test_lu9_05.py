import numpy as np
from genny import fdstring, contact, geo, analysis, matter, heavy_vehicles
from genny.physical import piano_data

def test_marimba_tuning():
    assert fdstring._TUNING["marimba"][1] == 9.8

def test_wood_moisture():
    e0, l0 = contact.wood_moisture_factors(6.0)
    e1, l1 = contact.wood_moisture_factors(12.0)
    assert abs(e0 - 1) < 0.05 and e1 < e0 and l1 > l0

def test_plume():
    assert 90 < geo.bubble_plume_frequency(0.2) < 105

def test_kurtosis_gaussian():
    sr = 8000
    x = np.random.default_rng(0).standard_normal(sr)
    _, k = analysis.kurtosis_echo_density(x, sr)
    assert abs(np.median(k[len(k)//2:]) - 1) < 0.2

def test_slinky_echo_period():
    f = fdstring.slinky_modes(22000, 0.06)
    assert f is not None

def test_rauhala_sections():
    a1, a2 = piano_data.rauhala_tunable_sections(200.0, 1e-4, 4)
    assert np.isfinite(a1) and np.isfinite(a2)

def test_acfw_ice_stiffer():
    assert matter.acfw_frequency(0.3, e=9.7e9, nu=0.31) > 0
    assert abs(matter.air_sound_speed(0) - 331.3) < 1e-6

def test_leopard1():
    assert heavy_vehicles.TRACK_KINDS["tank_leopard1"][1] == 0.169
