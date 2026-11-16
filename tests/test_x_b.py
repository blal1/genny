"""Lot X-B: the two-stroke presets over the generic core, and their ignition cuts."""
import numpy as np
import pytest

from genny import engines as e, heavy_vehicles as h, marine as m
from genny.cyber import piston_x
from genny.physical.core import SR

FIRING = {"chainsaw": (9400.0, 1, 156.67), "scooter": (7000.0, 1, 116.67), "mx125": (11500.0, 1, 191.67), "mx250": (8500.0, 1, 141.67),
          "outboard": (5500.0, 3, 275.0), "snowmobile": (7000.0, 1, 116.67), "mower": (3000.0, 1, 50.0), "default": (7000.0, 1, 116.67)}


def _spec(y):
    return np.abs(np.fft.rfft(y * np.hanning(len(y)))), np.fft.rfftfreq(len(y), 1.0 / SR)


def _line(y, f0):
    sp, fr = _spec(y)
    return 20.0 * np.log10(sp[(fr > f0 * 0.98) & (fr < f0 * 1.02)].max() + 1e-12)


@pytest.mark.parametrize("kind", list(FIRING))
def test_each_preset_fires_at_its_rate(kind):
    rpm, C, f0 = FIRING[kind]
    y = e.two_stroke(sr=SR, kind=kind, rpm=rpm, cylinders=C, dur=3.0)
    assert np.isfinite(y).all()
    sp, fr = _spec(y)
    band = (fr > 0.5 * f0) & (fr < 1.6 * f0)
    if C == 1:
        assert abs(fr[band][np.argmax(sp[band])] - f0) < 0.03 * f0
    else:                                                   # three unequal pipes: the per-cylinder lines stay, the firing rate is among the strongest
        assert _line(y, f0) > max(_line(y, f0 * k / C) for k in range(1, 2 * C)) - 6.0


def test_working_speed_is_the_kinds():
    y = e.two_stroke(sr=SR, kind="mx125", dur=2.0)
    assert _line(y, 191.67) > _line(y, 95.8) + 15


def test_ports_of_the_125_follow_the_note():
    eo, to, bex, btr, S, l, B = (float(v) for v in e.TWO_STROKE_KINDS["mx125"]["ports"].split(","))
    assert abs(float(piston_x(eo, S, l)) - 26.8) < 0.3 and abs(float(piston_x(to, S, l)) - 40.8) < 0.3


def test_unknown_kind_is_refused():
    with pytest.raises(ValueError):
        e.two_stroke(sr=SR, kind="steam", dur=0.5)
    with pytest.raises(ValueError):
        e.two_stroke(sr=SR, kind="vintage", cut="machine", dur=0.5)


@pytest.mark.parametrize("kind", ["chainsaw", "scooter", "mx125", "mx250", "outboard"])
def test_the_limiter_cuts_the_ignition(kind):
    C = 3 if kind == "outboard" else 1
    y = e.two_stroke(sr=SR, kind=kind, cylinders=C, cut="machine", dur=3.0)
    env = np.convolve(np.abs(y), np.ones(int(0.01 * SR)) / int(0.01 * SR), "same")[SR // 2:-SR // 4]
    assert np.isfinite(y).all()
    quiet = float(np.mean(env < 0.3 * env.max()))
    assert 0.05 < quiet < 0.8


def test_a_cut_is_much_quieter_than_a_firing_engine():
    on = e.two_stroke(sr=SR, kind="mx125", dur=2.0, load=0.8)
    # the exhaust alone, raw: a spark that never comes leaves the motored pressure
    from genny.cyber import pipe_engine
    kw = dict(sr=SR, dur=2.0, rpm=11500.0, load=0.8, cycle=2, ports=e.TWO_STROKE_KINDS["mx125"]["ports"], tuned_rpm=11500.0, header=0.3, pipe=0.5,
              unequal=0.0, mic=0.0, block=0.0, raw=1, seed=0)
    a, b = pipe_engine(fire=1.0, **kw), pipe_engine(fire=0.0, **kw)
    assert 20.0 * np.log10((np.std(b) + 1e-12) / np.std(a)) < -10.0
    assert np.isfinite(on).all()


def test_the_other_callers_render():
    for y in (h.chainsaw(sr=SR, dur=1.5, limiter=1), h.scooter(sr=SR, dur=1.5, limiter=1), h.mower(sr=SR, kind="lawn_mower_2t", dur=1.5),
              h.garden_machine(sr=SR, dur=1.5), h.tractor(sr=SR, kind="tractor_vintage", stroke=2, dur=1.5), h.tracks(sr=SR, kind="snowmobile", dur=1.5),
              h.motorcycle(sr=SR, kind="cross_125", dur=1.5), h.motorcycle(sr=SR, kind="cross_250", limiter=1, dur=1.5),
              m.outboard(sr=SR, strokes=2, limiter=1, dur=1.5)):
        assert np.isfinite(y).all() and np.abs(y).max() > 0.1
