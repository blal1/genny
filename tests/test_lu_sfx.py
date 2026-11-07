"""SFX group, deep reading 2026-11-07 (sources: research/SOURCES.md).

Run:  uv run --with pytest pytest tests/test_lu_sfx.py -q
"""
import numpy as np

SR = 48000


def _peak_hz(y, smooth):
    s = np.abs(np.fft.rfft(np.asarray(y, float))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return f[np.argmax(np.convolve(s, np.ones(smooth) / smooth, mode="same"))]


def test_paper_crumple_exponent_spans_houle_sethna_range():
    from genny import contact as K
    _, _, _, _, _, _, _, g0, gs, e_min = K._CRUMPLE["paper"]
    assert abs(g0 + 1.30) < 1e-9 and abs(g0 + gs + 1.59) < 1e-9
    assert e_min == 1e-3


def test_paper_crumple_rings_at_1_khz_for_1_ms_whatever_the_facet():
    from genny import contact as K
    fmin, fmax, tmin, tmax = K._CRUMPLE["paper"][:4]
    assert fmin == fmax == 1000.0 and tmin == tmax == 0.001
    assert 800.0 < _peak_hz(K.crumple(SR, kind="paper", seed=3), 200) < 1250.0


def test_bag_crumple_exponent_is_kramer_lobkovsky_mylar():
    from genny import contact as K
    _, _, _, _, _, _, _, g0, gs, e_min = K._CRUMPLE["bag"]
    assert -1.01 >= g0 + gs >= -1.2 and g0 > -1.05
    assert e_min == 1e-4


def test_sheet_crumple_keeps_event_count_with_wider_energy_span():
    from genny import contact as K
    g = -1.445
    m = K.crumple_energy_floor(g)
    r = K._power_mean(g, m) / K._power_mean(g, 1e-3)
    assert 10.0 < r < 30.0
    ev = K.crumple_events(1e9, g, 20.0, e_min=1e-3, n_max=20000, seed=2)
    assert ev["energy"].min() < 0.01 < 0.4 < ev["energy"].max()


def test_cloth_light_fabric_keeps_energy_above_4_khz():
    from genny import foley as F
    assert F.CLOTH_TOP == (8000.0, 2000.0)
    y = np.asarray(F.cloth(SR, weight=0.0), float)
    s = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    assert s[f > 4000].sum() / s.sum() > 0.4


def test_cabin_gain_follows_rieder_coupe_and_stratman_plateau():
    from scipy.signal import freqz
    from genny import dsp as D
    assert (D.CABIN_F0, D.CABIN_GAIN_DB) == (70.0, 26.0)
    h = D.cabin_gain(np.r_[1.0, np.zeros(1 << 17)], SR)
    _, H = freqz(h, 1, [5.0, 20.0, 1000.0], fs=SR)
    db = 20 * np.log10(np.abs(H))
    assert abs(db[0] - 26.0) < 0.5 and 18.0 < db[1] < 23.0 and abs(db[2]) < 0.1


def test_rotor_broadband_peaks_at_wright_strouhal_frequency():
    from genny import physics
    r = physics.Rotor()
    assert abs(r.broadband_hz - 1.0 * 30.0 / 0.05) < 1e-9
    assert physics.HELI_BLADE == {"thickness": 0.01, "strouhal": 0.8}
    assert abs(physics.Rotor(300.0, **physics.HELI_BLADE).broadband_hz - 400.0) < 1e-9
    assert 450.0 < _peak_hz(physics.Rotor(tonal=0.0).render(4.0, sr=SR), 400) < 800.0


def test_bat_search_sweep_is_eptesicus():
    from genny import fauna
    assert fauna.BAT_SWEEP == (28000.0, 22000.0)
    assert 5.0 <= fauna.BAT_RATE <= 10.0


def test_dolphin_sonar_click_train():
    from genny import fauna
    assert fauna.SONAR_CLICK == (117100.0, 37100.0)
    y = np.asarray(fauna.whale(SR, kind="dauphin_sonar"), float)
    assert abs(_peak_hz(y, 400) - 11710.0) < 1500.0
    env = np.abs(y)
    hits = np.flatnonzero((env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]) & (env[1:-1] > 0.05 * env.max()))
    starts = hits[np.r_[True, np.diff(hits) > SR * 0.01]]
    gaps = np.diff(starts) / SR
    assert 0.025 < np.median(gaps) < 0.045
    assert len(starts) >= 25
