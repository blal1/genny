import numpy as np

from genny import compose, contact, reverbs


def test_moorer_table_2_and_3():
    assert reverbs.MOORER_COMBS_MS == (50.0, 56.0, 61.0, 68.0, 72.0, 78.0)
    assert np.allclose(reverbs.moorer_loop_g1(25000), (0.24, 0.26, 0.28, 0.29, 0.30, 0.32))
    assert np.allclose(reverbs.moorer_loop_g1(50000), (0.46, 0.48, 0.50, 0.52, 0.53, 0.55))
    mid = reverbs.moorer_loop_g1(37500)
    assert np.allclose(mid, (0.35, 0.37, 0.39, 0.405, 0.415, 0.435))
    assert len(reverbs.MOORER_EARLY_7) == 7 and len(reverbs.MOORER_EARLY_19) == 19
    assert reverbs.MOORER_EARLY_19[-1] == (0.0797, 0.134)


def test_moorer_t60_rule():
    assert abs(reverbs.moorer_g_for_t60(2.0) - 0.817) < 1e-9
    assert abs(reverbs.moorer_g_for_t60(1.0) - 0.634) < 1e-9


def test_moorer_early_taps():
    sr = 25000
    x = np.zeros(4000)
    x[0] = 1.0
    y = reverbs.moorer_early(x, sr, 7)
    assert abs(y[int(round(0.0199 * sr))] - 1.02) < 1e-12
    assert abs(y[0] - 1.0) < 1e-12


def test_moorer_noise_ir_direct_ratio():
    h = reverbs.moorer_noise_ir(25000, 1.5, seed=3)
    assert h[0] > 0.9 and np.std(h[1:200]) < 0.12


def test_moorer79_late_runs():
    x = np.zeros(8000)
    x[0] = 1.0
    wl, wr = reverbs.moorer_late(x, 25000, 1.5, 0.8, table=True)
    assert np.isfinite(wl).all() and np.max(np.abs(wl)) > 0


def test_ride_pickup_ratio():
    assert abs(compose.ride_ratio_pickup(60) - 4.0) < 1e-9
    assert abs(compose.ride_ratio_pickup(120) - 3.0) < 1e-9
    assert abs(compose.ride_ratio_pickup(240) - 67.0 / 33.0) < 1e-9
    assert compose.ride_ratio_pickup(300) == compose.ride_ratio_pickup(240)


def test_wood_from_density():
    s = contact.wood_from_density(450.0, "softwood")
    assert abs(s["young"] - 56.3e6 * 450.0 ** 0.84) < 1.0
    assert 4400 < s["c"] < 4800
    h = contact.wood_from_density(650.0, "hardwood")
    assert abs(h["hardness"] - 6.48e-3 * 650.0 ** 2.09) < 1e-6
    assert abs(h["impedance"] - h["c"] * 650.0) < 1e-6
