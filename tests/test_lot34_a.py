import numpy as np

from genny import contact as C
from genny import fdstring, foley, friction, geo
from genny.physical import analysis as A
from genny.physical import footsteps as FS

SR = 44100


def _peak_hz(y, sr, lo=100.0, hi=8000.0):
    X = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1.0 / sr)
    m = (f > lo) & (f < hi)
    return float(f[m][np.argmax(X[m])])


def _centroid(y, sr):
    X = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(y), 1.0 / sr)
    return float((f * X).sum() / X.sum())


def _clean(y):
    assert np.all(np.isfinite(y)) and np.max(np.abs(y)) > 1e-3


def test_fd_bar_resonator_adds_the_tube_under_the_bar():
    a = fdstring.fd_bar(440.0, 0.4, sr=SR)
    b = fdstring.fd_bar(440.0, 0.4, sr=SR, resonator=1.0)
    _clean(b)
    assert len(a) == len(b) and not np.allclose(a, b)
    assert np.array_equal(a, fdstring.fd_bar(440.0, 0.4, sr=SR, resonator=0.0))


def test_bistable_relay_bounces_15_ms_on_release_not_7():
    mono = foley._switch("relay", "off", np.random.default_rng(1), foley._n(0.3))
    bi = foley._switch("relay", "off", np.random.default_rng(1), foley._n(0.3), False, "band", True)
    assert foley.RELAY_BOUNCE["off"] == 7e-3 and foley.RELAY_BOUNCE_BISTABLE["off"] == 15e-3
    i = int(0.013 * foley.P)
    assert np.sum(bi[i:i + int(0.008 * foley.P)] ** 2) > 3.0 * np.sum(mono[i:i + int(0.008 * foley.P)] ** 2)
    on_m = foley._switch("relay", "on", np.random.default_rng(1), foley._n(0.3))
    on_b = foley._switch("relay", "on", np.random.default_rng(1), foley._n(0.3), False, "band", True)
    assert np.allclose(on_m, on_b)


def test_snow_rub_rings_the_first_torsional_mode_and_is_silent_when_warm():
    y = geo.snow_rub(sr=SR, length=0.3, temperature=-15.0, dur=0.6)
    _clean(y)
    f1 = float(FS.stick_modes(0.3)[0])
    assert abs(_peak_hz(y, SR, 0.5 * f1, 1.5 * f1) - f1) < 0.04 * f1
    assert np.max(np.abs(geo.snow_rub(sr=SR, temperature=0.0, dur=0.3))) == 0.0


def test_squeak_snow_shoe_sits_below_the_dry_shoe():
    a = friction.squeak(sr=SR, surface="shoe", dur=0.5, seed=3)
    b = friction.squeak(sr=SR, surface="snow_shoe", dur=0.5, seed=3)
    _clean(b)
    assert _centroid(b, SR) < _centroid(a, SR)


def test_modal_fit_residual_is_what_the_modes_do_not_explain():
    sr = A.SR
    t = np.arange(sr) / sr
    rng = np.random.default_rng(0)
    x = 0.3 * np.exp(-9.0 * t) * np.sin(2 * np.pi * 523.0 * t + 0.4) + 0.2 * np.exp(-14.0 * t) * np.sin(2 * np.pi * 1310.0 * t + 1.1)
    x = np.concatenate([np.zeros(sr // 20), x]) + 1e-3 * rng.standard_normal(sr + sr // 20)
    md = A.modal_fit(x, 4)
    md2, res = A.modal_fit(x, 4, residual=True)
    assert np.allclose(md.freqs, md2.freqs) and len(res) == len(x)
    assert np.sqrt(np.mean(res ** 2)) < 0.3 * np.sqrt(np.mean(x ** 2))
    assert np.sqrt(np.mean(res[:sr // 20] ** 2)) <= np.sqrt(np.mean(x[:sr // 20] ** 2)) + 1e-12


def test_roll_slip_adds_the_scraper_model_and_slip_zero_is_unchanged():
    base = C.roll(sr=SR, dur=0.5, speed=0.5, seed=2)
    assert np.array_equal(base, C.roll(sr=SR, dur=0.5, speed=0.5, seed=2, slip=0.0))
    s = C.roll(sr=SR, dur=0.5, speed=0.5, seed=2, slip=0.6)
    _clean(s)
    assert len(s) == len(base) and not np.allclose(s, base)


def test_rest_force_replaces_the_flat_damping_by_the_measured_gamma():
    sp = [0.3, 0.3, 0.0, 0.0]
    a = C.scrape(sr=SR, dur=0.8, speed=sp, seed=1)
    b = C.scrape(sr=SR, dur=0.8, speed=sp, seed=1, rest_force=1.0)
    _clean(b)
    tail = slice(int(0.6 * SR), None)
    assert np.sqrt(np.mean(b[tail] ** 2)) > np.sqrt(np.mean(a[tail] ** 2))
    r = C.roll(sr=SR, dur=0.4, speed=0.5, rest_force=1.0, seed=0)
    _clean(r)
