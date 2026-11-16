import numpy as np

SR = 48000


def _peak(y, lo, hi, sr=SR):
    s = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / sr)
    m = (f >= lo) & (f <= hi)
    return float(f[m][np.argmax(s[m])])


def test_sethares_dissonance_curve_minima_of_a_harmonic_tone():
    from genny.analysis import sethares_dissonance, SETHARES_DISS
    assert SETHARES_DISS == (3.5, 5.75, 0.24, 0.021, 19.0)
    f = 500.0 * np.arange(1, 7)
    a = np.ones(6)
    al = np.arange(1.0, 2.2, 0.002)
    d = np.array([sethares_dissonance(f, a, x) for x in al])
    mins = al[1:-1][(d[1:-1] < d[:-2]) & (d[1:-1] < d[2:])]
    for r in (1.25, 4 / 3, 1.5, 5 / 3, 2.0):
        assert np.min(np.abs(mins - r)) < 0.005, r
    peak = np.log(5.75 / 3.5) / 2.25
    s = 0.24 / (0.021 * 500.0 + 19.0)
    df = np.linspace(1, 200, 2000)
    dd = [sethares_dissonance([500.0, 500.0 + x], [1.0, 1.0]) for x in df]
    assert abs(df[int(np.argmax(dd))] * s - peak) < 0.002


def test_gamelan_third_partial_on_the_saron_measure():
    from genny.acoustic import gamelan
    y = gamelan(200.0, 2.0, sr=SR)
    assert abs(_peak(y, 900.0, 1100.0) / 200.0 - 4.84) < 0.03


def test_pea_whistle_constants_are_cooks():
    from genny import sports as S
    assert S.PEA_FM == 0.07
    assert abs(20 * np.log10(S.PEA_AM) - 6.0) < 0.05 and abs(20 * np.log10(S.PEA_NOISE) - 2.0) < 0.05
    assert np.allclose(20 * np.log10(S.PEA_HARM), (0.0, -10.0, -25.0), atol=0.1)
    assert abs(20 * np.log10(S.PEA_AIR) + 20.0) < 0.01
