import math

import numpy as np

from genny import brass as B

RATE = B.RATE
TABLE_F2 = {"9x19": (5665, 6279), "556": (6086, 6208), "762x51": (4602, 4710), "50bmg": (2685, 2734)}


def _single(case, tail=1.6, spin=0.0, omega=0.0, seed=1, **kw):
    sc = dict(hits=dict(t=np.array([0.0]), J=np.array([0.01]), tc=np.array([8e-5]), face=np.array([0]), v=np.array([4.0]), e=np.array([0.5])),
              omega=np.array([omega]), rest=1e9, ending="none", roll=None, extra=None, tail=tail)
    return B.render_case(case, "concrete", 1.0, np.random.default_rng(seed), spin=spin, air=0.0, floor_gain=0.0, schedule=sc, **kw)


def _peak_near(y, f0, width=0.03):
    s = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    fr = np.fft.rfftfreq(len(y), 1.0 / RATE)
    m = (fr > f0 * (1 - width)) & (fr < f0 * (1 + width))
    return float(fr[m][np.argmax(s[m])])


def _t60(y, f, band=0.01, win=0.01):
    s = np.fft.rfft(y)
    fr = np.fft.rfftfreq(len(y), 1.0 / RATE)
    s[(fr < f * (1 - band)) | (fr > f * (1 + band))] = 0.0
    x = np.fft.irfft(s, len(y))
    k = int(win * RATE)
    r = np.sqrt(np.convolve(x * x, np.ones(k) / k, "valid")[::k])
    t = (np.arange(len(r)) + 0.5) * win
    ok = (t > 0.03) & (r > 1e-2 * r.max())
    slope = np.polyfit(t[ok & (t < 0.8)], np.log(r[ok & (t < 0.8)]), 1)[0]
    return 6.91 / -slope


def test_half_sine_force_area_and_zero():
    f = B.half_sine_force(0.02, 80e-6, 2048)
    assert abs(f.sum() / RATE - 0.02) < 1e-4 * 0.02
    assert abs(float(B.half_sine_spectrum(1.5 / 80e-6, 80e-6))) < 1e-9


def test_case_modes_match_table():
    for c, (r, love) in TABLE_F2.items():
        f = B.case_modes(c)
        assert abs(f[0] / r - 1) < 0.01 and abs(f[3] / love - 1) < 0.01
    f = B.case_modes("12ga", plastic_c=505.0)
    assert abs(f[0] / 1268 - 1) < 0.01 and abs(f[3] / 1314 - 1) < 0.01


def test_rendered_peak_on_modes():
    for c, (r, love) in TABLE_F2.items():
        y = _single(c, tail=0.8)
        p = _peak_near(y, r * 1.01, 0.03)
        assert min(abs(p / r - 1), abs(p / love - 1)) < 0.01


def test_mode_decay_table():
    d = B.mode_decay([6208.0, 17383.0], 0.30)
    assert abs(d[0] - 18.9) / 18.9 < 0.10 and abs(d[1] - 29.6) / 29.6 < 0.10
    d50 = B.mode_decay([2734.0], 0.60)[0]
    assert 6.91 / d50 > 1.0
    assert 400 < B.mode_decay([6208.0], 0.30, state="rest")[0] < 800


def test_t60_of_rendered_modes():
    assert abs(_t60(_single("556", tail=1.0), 6208.0) / 0.37 - 1) < 0.10
    assert _t60(_single("50bmg", tail=2.2), 2734.0) > 1.0


def test_ring_scales_internal_loss():
    assert B.mode_decay([6208.0], 0.30, ring=4.0)[0] < B.mode_decay([6208.0], 0.30)[0]


def test_air_column():
    assert abs(B.air_mode("556")[0] - 1870) < 60
    assert abs(B.air_mode("9x19")[0] - 4514) < 50
    assert abs(B.air_mode("12ga")[0] - 1178) < 30
    lo, hi = {"762x51": (1141, 1733), "50bmg": (511, 871)}["762x51"]
    assert lo < B.air_mode("762x51")[0] < hi
    assert 511 < B.air_mode("50bmg")[0] < 871


def test_hertz_time_depends_on_speed():
    m = 6.2e-3
    assert abs(B.hertz_tc(m, 4.0, "concrete") - 79e-6) < 3e-6
    assert B.hertz_tc(m, 0.2, "concrete") > B.hertz_tc(m, 1.0, "concrete") > B.hertz_tc(m, 4.0, "concrete")


def test_restitution_grows_as_speed_falls_and_is_capped():
    assert B.restitution(0.1, "concrete") > B.restitution(1.0, "concrete") > B.restitution(5.0, "concrete")
    assert B.restitution(1e-3, "concrete") <= B.E_MAX
    assert B.restitution(2.0, "concrete", lively=0.5) > B.restitution(2.0, "concrete")


def test_bounce_gaps_speed_up_into_a_rattle():
    first, last, tail = [], [], []
    for s in range(8):
        t = B.bounce_schedule("556", "concrete", 1.2, np.random.default_rng(s), ending="none")["hits"]["t"]
        g = np.diff(t)
        assert len(g) >= 6
        first.append(np.mean(g[1:4] / g[:3]))
        last.append(np.mean(g[-3:] / g[-4:-1]))
        tail.append(g[-1])
        assert g[0] > 0.2 and np.all(g > 0.0005)
    assert np.mean(last) > np.mean(first)
    assert max(tail) < 0.004


def test_last_impacts_are_duller_than_first():
    a, b = [], []
    for s in range(8):
        h = B.bounce_schedule("556", "concrete", 1.2, np.random.default_rng(s), ending="none")["hits"]
        a.append(h["tc"][:3].mean())
        b.append(h["tc"][-3:].mean())
    assert np.mean(b) > np.mean(a)


def test_kinematics_gives_the_turning():
    s = B.bounce_schedule("556", "concrete", 1.2, np.random.default_rng(0), spin=40.0, ending="none")
    assert len(s["omega"]) == len(s["hits"]["t"]) and np.any(np.abs(s["omega"]) > 5)
    s0 = B.bounce_schedule("556", "concrete", 1.2, np.random.default_rng(0), spin=0.0, ending="none")
    assert np.all(s0["omega"] == 0)


def test_spin_sidebands_only_when_turning():
    a = _single("556", tail=0.6, spin=40.0, omega=40.0)
    b = _single("556", tail=0.6, spin=0.0, omega=0.0)

    def side(y):
        s = np.abs(np.fft.rfft(y[: int(0.6 * RATE)] * np.hanning(int(0.6 * RATE)))) ** 2
        fr = np.fft.rfftfreq(int(0.6 * RATE), 1.0 / RATE)
        m = ((np.abs(fr - 6208) > 15) & (np.abs(fr - 6208) < 120)) | ((np.abs(fr - 6086) > 15) & (np.abs(fr - 6086) < 120) & False)
        return s[m].sum() / s.sum()

    assert side(a) > 5.0 * side(b)


def test_renders_are_clean_and_seeded():
    for c in B.CASE_GEOM:
        for fl in ("concrete", "wood", "grass"):
            y = B.render_case(c, fl, 1.2, np.random.default_rng(4))
            assert np.isfinite(y).all() and abs(np.max(np.abs(y)) - 1.0) < 1e-9 and len(y) < 14 * RATE
    a = B.render_case("556", "tile", 1.2, np.random.default_rng(9))
    assert np.array_equal(a, B.render_case("556", "tile", 1.2, np.random.default_rng(9)))


def test_endings_roll_and_wobble():
    for e in ("roll", "wobble", "none"):
        s = B.bounce_schedule("556", "concrete", 1.2, np.random.default_rng(3), ending=e)
        assert s["ending"] == e and (e == "none" or len(s["extra"]["t"]) > 3)
        y = B.render_case("556", "concrete", 1.2, np.random.default_rng(3), ending=e)
        assert np.isfinite(y).all()
    w = B.bounce_schedule("556", "concrete", 1.2, np.random.default_rng(3), ending="wobble")["extra"]
    assert np.all(np.diff(np.diff(w["t"])) <= 1e-9) and np.all(np.diff(w["J"]) < 0)
