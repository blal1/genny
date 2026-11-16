"""Two-stroke core of cyber.pipe_engine: port areas from the piston, blowdown exhaust, crankcase, scavenging, misfires, tuned chamber,
steepening, and the ignition-cut sequences. Four-stroke defaults stay as they were."""
import numpy as np
import pytest

import genny.cyber as c

SR = c.SR
P125 = (82, 113, 37.8, 30, 54.5, 110, 54)
ENG = dict(cycle=2, cylinders=1, header=0.3, pipe=0.5, unequal=0.0, ports=P125, tuned_rpm=11500, raw=1, seed=1)


def _rms(y):
    return float(np.sqrt(np.mean(np.asarray(y) ** 2)))


def _lines(y, f0, k=3):
    s = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1.0 / SR)
    return [float(s[(f > j * f0 * 0.97) & (f < j * f0 * 1.03)].sum()) for j in range(1, k + 1)]


def _exhaust(**kw):
    return c.pipe_engine(**{**dict(dur=1.5, mic=0.0, block=0.0), **ENG, **kw})


def test_port_area_opens_at_the_angle_and_is_symmetric_about_bdc():
    th = np.arange(0.0, 360.0, 0.25)
    a = c.port_area(th, 82.0, 37.8, 54.5, 110.0)
    assert a[th < 81.5].max() == 0.0 and a[(th > 82.5) & (th < 277.5)].min() > 0.0
    assert th[a.argmax()] == pytest.approx(180.0, abs=0.5)
    d = np.arange(1.0, 80.0, 1.0)
    lo, hi = c.port_area(180.0 - d, 82.0, 37.8, 54.5, 110.0), c.port_area(180.0 + d, 82.0, 37.8, 54.5, 110.0)
    assert np.abs(lo - hi).max() < 0.01 * a.max()


def test_cross_125_geometry_matches_the_hand_numbers():
    assert float(c.piston_x(82.0, 54.5, 110.0)) == pytest.approx(26.8, abs=0.3)
    assert float(c.piston_x(113.0, 54.5, 110.0)) == pytest.approx(40.8, abs=0.3)
    tab, info = c.two_stroke_tables(P125)
    assert info["a_ex"] * 1e6 == pytest.approx(1040.0, rel=0.06)
    assert info["exhaust_deg"] == 196.0 and tab.shape[0] == 5


@pytest.mark.parametrize("ports, rpm", [(P125, 9000.0), (c.TWO_STROKE_PORTS, 6000.0), ((95, 120, 30, 25, 39.2, 85, 40), 7000.0)])
def test_ignition_line_is_one_per_turn(ports, rpm):
    y = _exhaust(rpm=rpm, load=0.8, ports=ports, dur=2.0)
    s = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1.0 / SR)
    f0 = rpm / 60.0
    m = (f > 0.97 * f0) & (f < 1.03 * f0)
    ref = np.median(s[(f > 0.3 * f0) & (f < 0.9 * f0)])
    assert f[m][s[m].argmax()] == pytest.approx(f0, rel=0.02) and s[m].max() > 20 * ref
    k = f[(f > 40) & (f < 1500)][s[(f > 40) & (f < 1500)].argmax()] / f0
    assert abs(k - round(k)) < 0.03 * round(k)


def test_fuel_cut_leaves_less_than_18_db_on_the_ignition_lines_and_10_db_overall():
    for rpm, load in ((8000.0, 0.8), (11500.0, 0.8), (6000.0, 0.5)):
        a, b = _exhaust(rpm=rpm, load=load), _exhaust(rpm=rpm, load=load, fire=0.0)
        assert 10 * np.log10(np.mean(b * b) / np.mean(a * a)) <= -10.0
        la, lb = _lines(a, rpm / 60.0), _lines(b, rpm / 60.0)
        assert all(10 * np.log10(x / y) <= -18.0 for x, y in zip(lb, la))
        assert 10 * np.log10(np.mean(b * b) / np.mean(a * a)) < -4.7


def test_four_stroke_default_is_unchanged_and_blowdown_cut_is_12_db():
    y = c.pipe_engine(dur=1.0, rpm=3000, cylinders=4, load=0.6, seed=1)
    assert _rms(y) == pytest.approx(0.31668061344303267, abs=1e-12)
    assert float(np.abs(y).sum()) == pytest.approx(float(np.abs(c.pipe_engine(dur=1.0, rpm=3000, cylinders=4, load=0.6, seed=1)).sum()))
    kw = dict(dur=1.5, rpm=3000, cylinders=4, load=0.8, raw=1, mic=0.0, block=0.0, seed=1, blowdown=1)
    a, b = c.pipe_engine(**kw), c.pipe_engine(fire=0.0, **kw)
    assert 10 * np.log10(np.mean(b * b) / np.mean(a * a)) <= -12.0
    assert all(10 * np.log10(x / y) <= -12.0 for x, y in zip(_lines(b, 100.0), _lines(a, 100.0)))


def test_tuned_chamber_length_and_reflection_delay():
    assert c.tuned_length(11500, 196.0, 800.0) == pytest.approx(0.74, rel=0.03)
    d, r, g = c.tuned_chamber(11500, 196.0, 0.038)
    assert len(d) == len(r) + 1 == len(g) and d.min() >= 1
    a = (1.36 * 287.0 * 800.0) ** 0.5
    assert 2 * d[0] == pytest.approx(2 * 0.3 * c.tuned_length(11500, 196.0, 800.0) / a * SR, rel=0.2, abs=2)
    assert (r[1:6] < 0).all() and (r[-8:] > 0).any()


def test_level_peaks_near_the_tuned_speed():
    rpms = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]) * 11500.0
    lv = [_rms(_exhaust(rpm=float(r), load=0.9, dur=1.0)) for r in rpms]
    assert abs(rpms[int(np.argmax(lv))] / 11500.0 - 1.0) <= 0.15


def test_limiter_period_and_skipped_fraction():
    ramp, st = c._limiter_run(2 * SR, 11000.0, 12500.0, 8000.0, 6000.0, 150.0, False)
    cut = st == 2
    edges = np.flatnonzero(np.diff(cut.astype(int)) == 1)
    assert 0.030 <= float(np.diff(edges).mean()) / SR <= 0.060
    assert 0.2 <= cut[edges[0]:].mean() <= 0.6
    assert ramp[edges[0]:].max() <= 12500.0 + 10.0 and ramp[edges[0]:].min() >= 12500.0 - 150.0 - 10.0
    y = c.pipe_engine(cycle=2, cylinders=1, dur=1.0, rpm=11000, load=1.0, ports=P125, tuned_rpm=11500, cut="limiter", rev_limit=12500.0, raw=1, seed=2, mic=0.0, block=0.0)
    assert np.isfinite(y).all() and _rms(y) > 0.01


def test_soft_cut_skips_one_in_four_before_the_hard_cut():
    n = 3 * SR
    ramp, st = c._limiter_run(n, 7000.0, 8000.0, 5000.0, 3000.0, 150.0, True)
    assert (st == 1).any() and (st == 2).any()
    offs = np.arange(4) / 4.0
    gate, _ = c.firing_gates(0, offs, np.cumsum(ramp / (120.0 * SR)), n, np.random.default_rng(0), state=st)
    assert (gate == 0).any() and (gate == 1).any()


def test_four_stroking_at_idle_and_none_at_charge():
    m = 30 * SR
    rp = np.full(m, 3000.0)
    cx = np.cumsum(rp / (60.0 * SR))
    for ld, lo, hi in ((0.1, 0.2, 0.5), (0.8, 0.0, 0.02)):
        dr, se = c.scavenge(ld, rp)
        g, _ = c.firing_gates(1, np.array([0.0]), cx, m, np.random.default_rng(1), pmf=c.misfire_probability(se), stick=0.3)
        miss = (g[0] == 0).astype(float)
        assert lo <= miss.mean() <= hi
        if ld < 0.5:
            assert np.corrcoef(miss[:-1], miss[1:])[0, 1] < 0.0


def test_steepening_adds_highs_that_grow_with_charge():
    def share(y):
        s = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
        return float(s[np.fft.rfftfreq(len(y), 1.0 / SR) > 4000.0].sum() / s.sum())
    gain = []
    for ld in (0.2, 0.9):
        a = _exhaust(rpm=9000.0, load=ld, steepen=0.0, dur=1.0)
        b = _exhaust(rpm=9000.0, load=ld, steepen=1.0, dur=1.0)
        gain.append(share(b) - share(a))
    assert gain[1] > gain[0] and gain[1] > 0.0


def test_deceleration_cut_drops_the_exhaust_12_to_16_db():
    rp = np.linspace(4000.0, 2000.0, 10)
    y = c.pipe_engine(cylinders=4, dur=4.0, rpm=rp, load=0.05, raw=1, mic=0.0, block=0.0, seed=1, cut="dfco")
    on, off = y[int(0.3 * SR):int(1.3 * SR)], y[int(2.2 * SR):int(3.5 * SR)]
    assert -17.0 <= 10 * np.log10(np.mean(off ** 2) / np.mean(on ** 2)) <= -11.0


def test_defaults_render_finite_and_documented():
    for kw in (dict(), dict(cycle=2), dict(cycle=2, induct="port", ccr=1.6), dict(cycle=2, misfire=0.0, steepen=0.0)):
        y = c.pipe_engine(sr=SR, dur=0.5, rpm=5000, seed=3, **kw)
        assert np.isfinite(y).all() and len(y) == int(0.5 * SR)
    with pytest.raises(ValueError):
        c.pipe_engine(dur=0.1, cut="limiter")
    with pytest.raises(ValueError):
        c.pipe_engine(dur=0.1, cycle=2, ports="1,2,3")
