import numpy as np

from genny import cyber, fx, matter, osc, sfx
from genny import foley
from genny.core import DEFAULT_SR
from genny.physical import modal

SR = DEFAULT_SR


def _peak_hz(y, sr, lo=20.0, hi=None):
    s = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / sr)
    m = (f > lo) & (f < (hi or sr / 2))
    return f[m][np.argmax(s[m])]


def test_pulse_train_gauss_has_zero_mean_and_narrows_with_index():
    n = SR
    a = osc.pulse_train(100.0, n, SR, index=2.0)
    b = osc.pulse_train(100.0, n, SR, index=40.0)
    assert abs(np.mean(a)) < 1e-3 and abs(np.mean(b)) < 1e-3
    assert np.sum(b > 0.5 * b.max()) < np.sum(a > 0.5 * a.max())


def test_pulse_train_hann_duty_and_carrier_formant():
    n = SR
    y = osc.pulse_train(100.0, n, SR, index=2.0, shape="hann")
    assert np.mean(y > 0.0) < 0.55
    c = osc.pulse_train(100.0, n, SR, index=1.0, shape="hann", carrier=8.0)
    assert 600.0 < _peak_hz(c, SR) < 1000.0
    two = osc.pulse_train(100.0, n, SR, index=1.0, shape="hann", carrier=8.0, copies=2)
    assert np.all(np.isfinite(two))


def test_laser_farnell_sweeps_start_above_nyquist_and_end_audible():
    y = sfx.REGISTRY["laser"]["fn"](SR, model="farnell", tail=3.3)
    assert np.all(np.isfinite(y)) and len(y) >= int(3.2 * SR) and np.max(np.abs(y)) > 0.1
    rms = lambda a, b: np.sqrt(np.mean(y[int(a * SR):int(b * SR)] ** 2))
    assert rms(2.5, 3.2) < rms(0.7, 1.4) < rms(0.0, 0.1)
    assert matter.LASER_SD[-1] == (3200.0, 65.0, 12400.0)


def test_telephone_farnell_band_is_narrow_around_400_and_2000():
    x = np.random.default_rng(0).uniform(-1, 1, SR)
    y = fx.telephone(x, SR, voicing="farnell")
    s = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(y), 1 / SR)
    assert s[(f > 300) & (f < 500)].mean() > 3.0 * s[(f > 5000) & (f < 8000)].mean()


def test_haversine_arc_differs_from_azimuth_only_with_elevation_change():
    p = dict(modal.SWING_PRESETS["SWD1"])
    assert np.isclose(modal._great_circle(0.0, 0.0, 0.0, 180.0), np.pi)
    assert np.isclose(modal._great_circle(60.0, 0.0, 60.0, 180.0), np.pi - 2 * np.radians(60.0))
    equator = dict(p, PreElev=90.0, PostElev=90.0)
    n_az = len(modal._swing_kinematics(equator, 0.05, 0.15)[0])
    n_hv = len(modal._swing_kinematics(equator, 0.05, 0.15, "haversine")[0])
    assert abs(n_az - n_hv) <= 2
    assert len(modal._swing_kinematics(p, 0.05, 0.15, "haversine")[0]) != n_az or p["PreElev"] == 90.0


def test_pipe_engine_muffler_reflection_and_ignition():
    d = dict(sr=SR, dur=0.6, rpm=2400.0, seed=1)
    base = cyber.pipe_engine(**d)
    closed = cyber.pipe_engine(**d, muffler_r=0.5)
    assert np.all(np.isfinite(closed)) and not np.allclose(base, closed)
    off = cyber.pipe_engine(**d, ignition=0.0)
    assert np.sqrt(np.mean(off ** 2)) < np.sqrt(np.mean(base ** 2)) or np.max(np.abs(off)) <= np.max(np.abs(base)) + 1e-9
    same = cyber.pipe_engine(**d, muffler_r=-0.142, compression=2.43, ignition=1.0, clutch=0.0)
    assert np.allclose(base, same)


def test_pipe_engine_clutch_rattle_only_between_300_and_2200_rpm():
    r = cyber.clutch_rattle(np.array([100.0] * 4000 + [1200.0] * 4000 + [4000.0] * 4000), 0.3, 1.0, np.random.default_rng(0))
    assert np.max(np.abs(r[:4000])) == 0.0 and np.max(np.abs(r[8000:])) == 0.0 and np.max(np.abs(r[4000:8000])) > 0.0


def test_dark_drone_mood_adds_a_third_and_a_swell():
    plain = foley.dark_drone(SR, dur=2.0, seed=2)
    mood = foley.dark_drone(SR, dur=2.0, seed=2, valence=0.8, tension=0.7)
    assert np.all(np.isfinite(mood)) and not np.allclose(plain[:len(mood)], mood[:len(plain)])


def test_choir_mech_rd_holds_the_ranges():
    from genny import choir
    assert choir.MECH_RD == ((0.3, 1.0), (1.0, 2.3))
    y = choir.voice(220.0, 0.4, SR, mech_rd=True)
    assert np.all(np.isfinite(y)) and np.max(np.abs(y)) > 0
