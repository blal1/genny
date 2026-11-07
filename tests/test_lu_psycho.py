import numpy as np

from genny import analysis as A

SR = 48000
T = np.arange(2 * SR) / SR


def _tone(db, f=1000.0, t=T):
    return np.sqrt(2) * 2e-5 * 10 ** (db / 20) * np.sin(2 * np.pi * f * t)


def test_usl_table_a9_corrected_cell():
    assert A._ZW_USL[12, 4] == 0.22


def test_iso532_reference_tone_is_one_sone():
    assert abs(A.zwicker_loudness(_tone(40), SR)[0] - 1.0) < 0.005
    assert abs(A.zwicker_loudness(_tone(40, t=np.arange(2 * 44100) / 44100), 44100)[0] - 1.0) < 0.005


def test_iso532_annex_e_tone_and_pink_noise():
    assert abs(A.zwicker_loudness(_tone(60), SR)[0] / 4.019 - 1) < 0.01
    assert abs(A.zwicker_loudness(_tone(60.7), SR)[0] / 4.228 - 1) < 0.01
    w = np.random.default_rng(0).standard_normal(SR * 3)
    f = np.fft.rfftfreq(w.size, 1 / SR)
    s = np.fft.rfft(w) / np.sqrt(np.maximum(f, 1.0))
    s[(f < 20) | (f > 20000)] = 0
    pink = np.fft.irfft(s, w.size)
    pink *= 2e-2 / np.sqrt(np.mean(pink ** 2))
    assert abs(A.zwicker_loudness(pink, SR)[0] / 12.46 - 1) < 0.02


def test_iso532_figure_examples():
    assert abs(A.zwicker_from_bands([78.0] * 28)[0] / 95.0 - 1) < 0.01
    n70 = A.zwicker_loudness(_tone(70), SR)[0]
    assert abs(n70 / 8.0 - 1) < 0.02 and abs(float(A.loudness_level(n70)) - 70.0) < 0.3


def test_iso532_filter_bank_leaks_20_db():
    lv = A.zwicker_band_levels(_tone(70), SR)
    assert abs(lv[16] - 70) < 0.1 and abs(lv[15] - 50) < 0.3 and abs(lv[17] - 50) < 0.3


def test_loudness_level_formulae():
    assert float(A.loudness_level(1.0)) == 40.0
    assert abs(float(A.loudness_level(2.0)) - 50.0) < 0.01
    assert abs(float(A.loudness_level(0.5)) - 40 * 0.5005 ** 0.35) < 1e-12


def test_time_varying_steady_and_duration():
    n = A.zwicker_loudness_tv(_tone(40), SR)[0]
    assert n.size == 1000 and abs(n[-100:].mean() - 1.0) < 0.01
    peaks = []
    for d in (0.01, 0.1):
        y = np.zeros(SR)
        y[:int(d * SR)] = _tone(70)[:int(d * SR)]
        peaks.append(A.zwicker_loudness_tv(y, SR)[0].max())
    assert peaks[0] < 0.8 * peaks[1]


def test_nl_decay_steeper_after_short_signals():
    core = np.zeros((21, 400))
    core[:, :200] = 1.0
    long = A._zw_nl(core, 2000.0)[0, 230]
    core = np.zeros((21, 400))
    core[:, 196:200] = 1.0
    short = A._zw_nl(core, 2000.0)[0, 230]
    assert short < 0.5 * long


def test_percentiles_from_time_varying():
    y = np.concatenate([np.zeros(SR // 2), _tone(40)[:SR // 2]])
    p = A.loudness_percentiles(y, SR)
    assert abs(p["N5"] - 1.0) < 0.02 and p["mean"] < p["N10"] <= p["N5"] <= p["max"]


def test_air_absorption_iso_9613_1_table_1_20_and_25_c():
    from genny.physical.space import air_db_per_m
    fm = lambda k: 1000 * 10 ** (k / 10)  # noqa: E731
    for t, rh, k, ref in ((20, 50, 0, 4.66), (20, 50, 6, 29.4), (20, 10, 10, 193.0), (20, 70, -12, 0.0897),
                          (20, 20, 3, 21.5), (20, 100, 3, 9.21), (25, 80, 9, 59.6), (25, 10, -13, 0.262)):
        assert abs(float(air_db_per_m([fm(k)], rh, t)[0]) * 1000 / ref - 1) < 0.006


def test_binaural_ratio_falls_with_level():
    for db, r in ((25, 1.95), (80, 1.4)):
        x = _tone(db)[:SR]
        one = A.binaural_loudness(np.stack([x, 0 * x], 1), SR)
        two = A.binaural_loudness(np.stack([x, x], 1), SR)
        assert abs(two / one - r) < 0.03 and abs(two - A.zwicker_loudness(x, SR)[0]) < 1e-9


def test_duration_loudness_ten_phon_per_decade():
    assert A.duration_loudness_db(0.01) == -10.0 and abs(A.duration_loudness_db(0.001) + 20.0) < 1e-9
    assert A.duration_loudness_db(0.1) == 0.0 and A.duration_loudness_db(0.5) == 0.0


def test_pitch_strength_factor_2_5_from_20_to_80_db():
    assert abs(A.pitch_strength(1500.0, level_db=80.0) / A.pitch_strength(1500.0, level_db=20.0) - 2.5) < 1e-9


def test_one_acum_reference_noise():
    w = np.random.default_rng(1).standard_normal(SR * 2)
    f = np.fft.rfftfreq(w.size, 1 / SR)
    s = np.fft.rfft(w)
    s[(f < 920) | (f > 1080)] = 0
    nb = np.fft.irfft(s, w.size)
    nb *= 2e-2 / np.sqrt(np.mean(nb ** 2))
    assert abs(A.zwicker_sharpness(A.zwicker_loudness(nb, SR)[1]) - 1.0) < 0.05
