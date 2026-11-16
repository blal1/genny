import numpy as np

from genny import choir, creatures, foley, fx, reverbs, sfx, spec
from genny.core import DEFAULT_SR


def _peak_hz(y, sr, lo=100.0):
    s = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1.0 / sr)
    m = f > lo
    return float(f[m][np.argmax(s[m])])


def _tone(f, sr, dur=1.0):
    return np.sin(2 * np.pi * f * np.arange(int(sr * dur)) / sr)


def test_ui_level_closes_three_quarters_of_the_distance_and_hover_lowers_the_target():
    sr = 44100
    y = 0.01 * _tone(800.0, sr, 0.2)
    win_db = 20 * np.log10(0.01 / np.sqrt(2))
    g = spec.ui_level(y, sr)
    assert abs(g - 0.75 * (-25.5 - win_db)) < 0.05
    assert abs((g - spec.ui_level(y, sr, offset=-6.0)) - 0.75 * 6.0) < 1e-6
    loud = 0.5 * _tone(800.0, sr, 0.2)
    assert 20 * np.log10(0.5) + spec.ui_level(loud, sr) <= -1.0 + 1e-9
    out, _ = spec.render_spec({"layers": [{"type": "sfx", "kind": "beep"}], "ui_level": "hover"})
    assert np.isfinite(out).all() and np.abs(out).max() <= 1.0


def test_crossing_beeper_is_a_100_ms_gated_2_5_khz_sine_with_weak_5_and_7_5_khz():
    sr = 44100
    y = sfx.crossing_beeper(sr)
    assert abs(_peak_hz(y, sr) - 2500.0) < 20.0
    s = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(y), 1.0 / sr)
    at = lambda hz: s[np.argmin(np.abs(f - hz))]       # noqa: E731
    assert abs(at(5000.0) / at(2500.0) - 0.1) < 0.02 and abs(at(7500.0) / at(2500.0) - 0.05) < 0.02
    assert np.abs(y[: int(0.1 * sr)]).max() > 0.5 and np.abs(y[int(0.125 * sr): int(0.19 * sr)]).max() < 1e-6
    d = sfx.proximity(sr)
    assert np.isfinite(d).all()


def test_two_tap_shift_moves_a_tone_by_the_ratio_and_octave_up_is_unchanged():
    sr = 44100
    x = _tone(400.0, sr)
    for t in (2.0, 1.5, 0.75):
        y = reverbs._two_tap_shift(x, sr, t)
        assert abs(_peak_hz(y[sr // 4:], sr) - 400.0 * t) < 8.0
    assert np.array_equal(reverbs._octave_up(x, sr), reverbs._two_tap_shift(x, sr, 2.0, 0.06))
    y = fx.pitch(x, sr, semitones=7.0, mode="delay")
    assert abs(_peak_hz(y[sr // 4:], sr) - 400.0 * 2 ** (7 / 12)) < 8.0


def test_footsteps_speed_follows_the_stride_laws():
    p = creatures.speed_pace
    assert abs(p("walk", 1.31) - creatures.GAITS["walk"]) < 1e-9
    assert abs(p("walk", 4 * 1.31) / p("walk", 1.31) - 2.0) < 1e-9
    pref = 1.09 * 500.0 ** 0.222
    assert abs(p("trot", pref) - creatures.stride_pace("trot", 500.0)) < 1e-6
    pref_g = 2.78 * 500.0 ** 0.176
    assert abs(p("gallop", pref_g) - creatures.stride_pace("gallop", 500.0)) < 1e-6
    assert p("trot", pref + 1.0) > p("trot", pref) > p("trot", pref - 1.0)
    assert abs(p("gallop", pref_g + 1.0) - p("gallop", pref_g)) < 240 * 0.019 * 1.0 + 1e-6
    y = creatures.footsteps(DEFAULT_SR, ground="wood", speed=1.31, steps=6, seed=1)
    z = creatures.footsteps(DEFAULT_SR, ground="wood", steps=6, seed=1)
    assert np.isfinite(y).all() and abs(len(y) - len(z)) < 0.1 * DEFAULT_SR


def test_snow_depth_adds_low_energy_and_zero_changes_nothing():
    def low(y):
        s = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(len(y), 1.0 / DEFAULT_SR)
        return s[f < 500].sum() / s.sum()
    a = creatures.footsteps(DEFAULT_SR, ground="snow", steps=4, seed=2)
    b = creatures.footsteps(DEFAULT_SR, ground="snow", steps=4, seed=2, snow_depth=0.0)
    deep = creatures.footsteps(DEFAULT_SR, ground="snow", steps=4, seed=2, snow_depth=20.0)
    thin = creatures.footsteps(DEFAULT_SR, ground="snow", steps=4, seed=2, snow_depth=2.0)
    assert np.array_equal(a, b)
    assert low(deep) > low(thin)


def test_purr_inhale_is_off_by_default_and_changes_the_breath_in():
    a = creatures.purr(DEFAULT_SR, seed=3)
    b = creatures.purr(DEFAULT_SR, seed=3, inhale=0.0)
    c = creatures.purr(DEFAULT_SR, seed=3, inhale=1.0)
    assert np.array_equal(a, b) and np.isfinite(c).all() and not np.array_equal(a, c)


def test_choir_rd_slope_is_the_measured_oq_against_level():
    assert abs(choir.RD_PER_DB - 0.0275) < 1e-9
    assert abs(choir.RD_PER_EFFORT - 0.0275 * 22.0 / 0.66) < 1e-9


def test_partial_wander_walks_the_upper_two_thirds_by_15_and_30_hz_and_is_bounded():
    from genny import spectral
    sr = 44100
    t = np.arange(int(1.5 * sr)) / sr
    x = sum(np.sin(2 * np.pi * 440 * k * t) / k for k in range(1, 21))
    m = spectral.analyze(x, sr)
    fr = np.array(m["freq"], float)
    off = spectral._partial_walk(fr, 15.0, 0)
    live = np.flatnonzero((fr > 0).sum(0))
    order = live[np.argsort(np.where(fr > 0, fr, 0).sum(0)[live] / (fr > 0).sum(0)[live])]
    k = len(order)
    rms = lambda sl: np.sqrt((off[:, sl] ** 2).sum() / max((fr[:, sl] > 0).sum(), 1))      # noqa: E731
    assert rms(order[: k // 3]) == 0.0
    assert rms(order[2 * k // 3:]) > 1.5 * rms(order[k // 3: 2 * k // 3]) > 0.0
    assert (np.abs(off) <= spectral.PARTIAL_WANDER_BOUND * fr + 1e-9).all()
    assert np.array_equal(spectral.resynth(m, sr), spectral.resynth(m, sr, partial_wander=0.0))
    assert not np.allclose(spectral.resynth(m, sr), spectral.resynth(m, sr, partial_wander=15.0))


def test_robot_babble_pd_is_bright_deterministic_and_the_fm_default_stays():
    a = foley.robot_babble(DEFAULT_SR, dur=2.0, seed=4, style="pd")
    b = foley.robot_babble(DEFAULT_SR, dur=2.0, seed=4, style="pd")
    f0 = foley.robot_babble(DEFAULT_SR, dur=2.0, seed=4)
    assert np.array_equal(a, b) and np.isfinite(a).all()

    def centroid(y):
        s = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
        return float((s * np.fft.rfftfreq(len(y), 1.0 / DEFAULT_SR)).sum() / s.sum())
    assert centroid(a) > 2.0 * centroid(f0)
    assert np.array_equal(f0, foley.robot_babble(DEFAULT_SR, dur=2.0, seed=4, style="fm"))
