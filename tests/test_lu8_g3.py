import numpy as np

from genny.physical import bank_piano as bp

SR = 44100


def test_rib_constants_from_measured_soundboard():
    assert bp.RIB_HZ == 1150.0
    assert bp.RIB_DENSITY == 0.4
    assert bp.RIB_MOBILITY == 3.5
    assert bp.RIB_BRIDGE_HZ == 2500.0


def test_ribs_sparser_stronger_poles_above_rib_limit():
    a1, a2, d0, _ = bp.soundboard_filter(SR, ribs=True)
    r = np.sqrt(a2)
    f = np.arccos(np.clip(-a1 / (2 * r), -1, 1)) * SR / (2 * np.pi)
    plain = bp.soundboard_filter(SR)
    fp = np.arccos(np.clip(-plain[0] / (2 * np.sqrt(plain[1])), -1, 1)) * SR / (2 * np.pi)
    assert np.sum(f > bp.RIB_HZ) < 0.5 * np.sum(fp > bp.RIB_HZ)
    sigma = bp.SB_A * f * f + bp.SB_B * f
    g = np.abs(d0) / (sigma / sigma.max())
    mid = (f > bp.RIB_HZ + 0.01) & (f < bp.RIB_BRIDGE_HZ - 0.01)
    top = f > bp.RIB_BRIDGE_HZ + 0.01
    assert np.allclose(g[f < bp.RIB_HZ - 0.01], 1.0)
    assert np.allclose(g[mid], 1.0 / bp.RIB_DENSITY)
    assert np.allclose(g[top], bp.RIB_MOBILITY / bp.RIB_DENSITY)


def test_horizontal_polarisation_tuning_below_detection():
    assert bp.HORIZ_CENTS <= 0.1
    assert bp.HORIZ_Y == 0.1


def test_blown_strings_and_brass_end_vibrato_follows_prame():
    import inspect
    from genny import acoustic as A
    assert inspect.signature(A._blown).parameters["end_boost"].default == -1.0
    for name in ("oboe", "sax", "harmonica"):
        assert "end_boost=2.0" in inspect.getsource(getattr(A, name))


def test_bass_drum_felt_contact_in_mallet_range():
    from genny import percussion as P
    felt, rute = P.BASS_DRUM_CONTACT_MS
    assert 2.0 <= felt <= 4.0 and rute < felt


def test_bridge_profile_bass_and_top_couple_less():
    assert bp.bridge_mobility_db(30.0) == -9.0
    assert bp.bridge_mobility_db(60.0) == 0.0
    assert bp.bridge_mobility_db(103.0) == -10.0


def test_violin_sordino_band_shape():
    from genny import acoustic as A
    x = np.random.default_rng(0).standard_normal(2 * SR)
    X = np.abs(np.fft.rfft(x)) ** 2
    Y = np.abs(np.fft.rfft(A.violin_sordino(x, SR))) ** 2
    f = np.fft.rfftfreq(x.size, 1 / SR)

    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return 10 * np.log10(Y[m].sum() / X[m].sum())
    assert band(200, 800) > -1.5
    assert -13.0 < band(1000, 2500) < -5.0
    assert -5.0 < band(2500, 4000) < -2.0


def test_bridge_profile_acts_on_bass_only():
    assert not np.allclose(bp.note(61.735, 0.5, SR, 0.8, tail=0.2),
                           bp.note(61.735, 0.5, SR, 0.8, tail=0.2, bridge_profile=True))
    assert np.allclose(bp.note(261.6, 0.3, SR, 0.8, tail=0.1),
                       bp.note(261.6, 0.3, SR, 0.8, tail=0.1, bridge_profile=True))


def _peak(y, f0):
    X = np.abs(np.fft.rfft(y[:SR] * np.hanning(SR)))
    fr = np.fft.rfftfreq(SR, 1 / SR)
    m = (fr > 0.98 * f0) & (fr < 1.02 * f0)
    return fr[m][np.argmax(X[m])]


def test_tubular_bell_loaded_tube_ratios():
    from genny import acoustic as A
    assert A.TUBULAR_BELL_RATIOS == (0.63, 1.22, 2.0, 2.93, 4.01, 5.21)
    y = A.tubular_bell(220.0, 1.0, SR)
    for r in (1.22, 2.93, 4.01):
        assert abs(_peak(y, 220.0 * r) / (220.0 * r) - 1) < 0.003


def test_glockenspiel_bar_ratios():
    from genny import acoustic as A
    assert A.GLOCKENSPIEL_RATIOS == (1.0, 2.71, 5.15, 8.43, 12.21)
    y = A.glockenspiel(784.0, 1.0, SR)
    for r in (1.0, 2.71, 5.15):
        assert abs(_peak(y, 784.0 * r) / (784.0 * r) - 1) < 0.003


def test_singing_bowl_qing_ratios():
    from genny import acoustic as A
    assert A.QING_RATIOS == (1.0, 2.75, 5.06, 7.78, 10.83)
    y = A.singing_bowl(200.0, 1.0, SR)
    for r in (2.75, 5.06, 7.78):
        assert abs(_peak(y, 200.0 * r) / (200.0 * r) - 1) < 0.003


def test_accordion_bend_drops_pitch_per_kpa():
    from genny import acoustic as A
    assert np.allclose(A.accordion(264.0, 0.5, SR), A.accordion(264.0, 0.5, SR, bend=0.0))
    for b, f in ((2.5, 259.0), (5.1, 253.8)):
        y = A.accordion(264.0, 1.0, SR, musette=0, bend=b, pressure=2.0)
        assert abs(_peak(y, f) - f) <= 1.0


def test_free_reed_harmonium_preset_speaks_just_below_its_reed():
    from genny import tubes as T
    y, tip = T.free_reed(dur=0.8, kind="-+", reed="harmonium", full=True)
    z = tip[tip.size // 2:] - tip[tip.size // 2:].mean()
    f = np.sum((z[:-1] < 0) & (z[1:] >= 0)) / (z.size / (tip.size / 0.8))
    assert 950.0 < f < 990.0


def test_harmonium_partials_from_the_measured_source():
    from genny import acoustic as A
    assert len(A.HARMONIUM_PARTIALS) == 37 and A.HARMONIUM_PARTIALS[:3] == (1.0, 0.2121, 0.3841)
    y = A.harmonium(349.0, 1.0, SR)
    X = np.abs(np.fft.rfft(y[:SR] * np.hanning(SR)))
    a = [X[int(round(349.0 * k))] for k in (1, 2, 3)]
    assert a[2] > a[1] and abs(a[2] / a[0] - 0.3841) < 0.05


def test_gamelan_jegogan_option_keeps_saron_default():
    from genny import acoustic as A
    assert A.GAMELAN_BARS["jegogan"][0] == (1.0, 2.7, 5.2, 8.4, 12.2)
    assert np.allclose(A.gamelan(200.0, 2.0, SR), A.gamelan(200.0, 2.0, SR, bar="saron"))
    y = A.gamelan(200.0, 2.0, SR, bar="jegogan", ombak=0.0)
    for r in (2.7, 5.2, 8.4):
        assert abs(_peak(y, 200.0 * r) / (200.0 * r) - 1) < 0.003
