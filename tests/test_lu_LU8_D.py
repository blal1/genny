import numpy as np

from genny import ordnance as O

LB, FT, PSI = 0.45359237, 0.3048, 6894.757293


def test_bubble_pulse_peak_is_coles_300_lb_example():
    w, r, d = 300 * LB, 60 * FT, 50 * FT
    assert abs(O.UW_PULSE_PM - 2590 * PSI * FT / LB ** (1 / 3)) < 0.01e6
    assert abs(O.bubble_pulse_peak(w, r) / PSI - 80.0) < 2.0
    y = O.underwater_blast(w, r, d, np.random.default_rng(0))
    T = O.UW_T * w ** (1 / 3) / (d + O.UW_D) ** (5 / 6)
    i, q = int(T * O.P), int(0.1 * T * O.P)
    ratio = np.max(y[i - q:i + q]) / np.max(y[:int(0.01 * O.P)])
    assert 0.03 < ratio < 0.08


def test_surface_cut_stops_at_zero_absolute_pressure():
    a = O.underwater_blast(100.0, 100.0, 3.0, np.random.default_rng(0), pulses=1, surface=False)
    b = O.underwater_blast(100.0, 100.0, 3.0, np.random.default_rng(0), pulses=1, surface=True)
    floor = (O.UW_P_ATM + O.UW_RHO_G * 3.0) / (O.UW_PM * (100.0 ** (1 / 3) / 100.0) ** 1.13)
    assert floor < 0.1 and a.min() >= -1e-9
    assert -0.3 < b.min() / b.max() < 0


def test_calving_fizz_rings_at_1_to_3_khz():
    from genny import geo as G
    f = [3.26 / r for r in G.CALVING_FIZZ_R]
    assert 2900 < f[0] < 3000 and 980 < f[1] < 1000


def test_ice_sing_uses_cold_air_sound_speed():
    import inspect
    from genny import matter as M
    assert inspect.signature(M.ice_sing).parameters["c"].default == 329.0
    assert M.acfw_frequency(0.05) < M.acfw_frequency(0.05, 343.0)


def test_corr_length_and_wake_constants():
    from genny.physical import modal as Md
    assert abs(Md.corr_length(1e4, 1.0) - 10 ** 1.536 * 1e4 ** -0.245) < 1e-12
    assert 3.4 < Md.corr_length(1e4, 1.0) < 3.8 and Md.corr_length(2e5, 1.0) < 2.0
    assert Md.WAKE_GAMMA == 1e-4 and Md.WAKE_B == 0.7


def test_cavitation_blade_lines_fall_with_harmonic():
    import inspect
    from genny import marine as Mr
    src = inspect.getsource(Mr._cavitation)
    assert "/ k" in src and "50.0 <= f" not in src


def test_slosh_hz_blevins_bucket():
    from genny.matter import slosh_hz
    assert abs(slosh_hz(0.1, 0.15) - 2.13) < 0.02
    assert slosh_hz(0.1, 0.15, 1) > slosh_hz(0.1, 0.15)
    assert slosh_hz(0.1, 0.01) < slosh_hz(0.1, 0.15)


def test_snow_avalanche_cloud_is_an_early_spindle():
    from genny import geo as G
    assert G.AVALANCHE_CLOUD == 0.5
    y = np.asarray(G.snow_avalanche(sr=48000, dur=4.0, powder=1.0, blocks=0, seed=0))
    assert np.all(np.isfinite(y))


def test_hail_sizes_follow_measured_law():
    from genny.contact import hail_impulses
    _, d, _ = hail_impulses(30.0, 8000, rate=400.0, size=6.7e-3, decay=0.0, seed=1, measured=True)
    assert abs(np.median(d) - 6.7e-3) < 0.3e-3
    assert np.mean(d >= 20e-3) < 0.02
    assert np.min(d) > 4.5e-3


def test_geyser_collapse_rebounds_50_ms_apart():
    from genny import geo as G
    assert abs(G.geyser_tau(0.055) - 0.05) < 0.01
    assert G.GEYSER_REBOUNDS == (1.0, 0.6, 0.35)


def test_singing_dune_has_harmonics_2_to_5():
    from genny import geo as G
    y = np.asarray(G.singing_dune(sr=48000, dur=4.0))
    s = np.abs(np.fft.rfft(y[48000:3 * 48000]))
    f = np.fft.rfftfreq(2 * 48000, 1 / 48000)
    for h in (2, 3, 4):
        m = (f > 100 * h - 5) & (f < 100 * h + 5)
        assert 20 * np.log10(s[m].max() / s.max()) > -30


def test_lava_end_correction():
    from genny import geo as G
    assert abs(G.lava_f0(25.0) - 343 / (4 * 26.2)) < 0.01 * G.lava_f0(25.0)


def test_debris_flow_approach_climbs_in_pitch():
    from genny import geo as G
    sr = 48000
    y = np.asarray(G.debris_flow(sr=sr, dur=8.0, approach=20.0, seed=0))
    assert len(y) == 28 * sr
    pre = y[:20 * sr]

    def centroid(x):
        s = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(len(x), 1 / sr)
        return (s * f).sum() / s.sum()
    assert centroid(pre[:5 * sr]) < centroid(pre[-5 * sr:])
    assert np.sqrt(np.mean(pre[:5 * sr] ** 2)) < np.sqrt(np.mean(pre[-5 * sr:] ** 2))

def test_drip_tap_is_one_8_khz_bubble():
    from genny import matter as M
    y = np.asarray(M.drip(sr=48000, regime="tap", seed=1))
    f = np.fft.rfftfreq(len(y), 1 / 48000)
    peak = f[np.argmax(np.abs(np.fft.rfft(y)))]
    assert 7000 < peak < 9500
    assert 0.020 < np.argmax(np.abs(y)) / 48000 < 0.030
