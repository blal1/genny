import numpy as np

from genny import arms, cyber


def _band(x, sr, lo, hi):
    s = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    return float(s[(f >= lo) & (f < hi)].sum())


def test_twin_outlet_lifts_the_low_band_against_the_high_band():
    sr = 48000
    one = cyber.pipe_engine(sr=sr, dur=1.0, seed=3)
    two = cyber.pipe_engine(sr=sr, dur=1.0, seed=3, twin=0.3)
    r1 = _band(one, sr, 40, 300) / _band(one, sr, 2000, 6000)
    r2 = _band(two, sr, 40, 300) / _band(two, sr, 2000, 6000)
    assert r2 > 1.5 * r1


def test_bend_shortens_the_pipe():
    assert 20.0 < cyber.bend_cents(0.025, 0.075) < 28.0
    a = cyber.pipe_engine(dur=0.5, seed=1)
    b = cyber.pipe_engine(dur=0.5, seed=1, bend="25,75")
    assert not np.allclose(a, b) and np.isfinite(b).all()
    try:
        cyber.pipe_engine(dur=0.2, bend="25")
    except ValueError:
        return
    raise AssertionError("bend needs two radii")


def test_step_radius_renders():
    y = cyber.pipe_engine(dur=0.5, seed=1, expansion=4.0, step_radius=25.0)
    assert np.isfinite(y).all()


def test_treeline_echo_is_brighter():
    sr = 48000
    dry = arms.firearm(sr=sr, weapon="ar15", distance=50.0, echo=0.0, seed=2)
    street = arms.firearm(sr=sr, weapon="ar15", distance=50.0, echo=1.0, seed=2)
    wood = arms.firearm(sr=sr, weapon="ar15", distance=50.0, echo=1.0, treeline=2000.0, seed=2)
    n = min(len(dry), len(street), len(wood))
    e_street, e_wood = street[:n] - dry[:n], wood[:n] - dry[:n]
    low = lambda e: _band(e, sr, 20, 500) / _band(e, sr, 500, 20000)
    assert low(e_wood) < 0.5 * low(e_street)
