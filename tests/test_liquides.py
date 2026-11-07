"""Liquides theme (lecture profonde): sourced numbers checked on the rendered signal or the law."""
import math

import numpy as np
from scipy.signal import hilbert

import genny.matter as m
from genny.analysis import detect_clicks
from genny.physical import ambience as A
from genny.physical import particles as P

SR = m.SR


def _decay_rate(y, lo=0.2, hi=0.7):
    """beta (1/s) from the slope of log |y| at its cycle peaks, between -6 and -40 dB."""
    from scipy.signal import find_peaks
    a = np.abs(y)
    k, _ = find_peaks(a)
    k = k[(a[k] < 0.5 * a.max()) & (a[k] > 0.01 * a.max())]
    return -np.polyfit(k / SR, np.log(a[k]), 1)[0]


# ---------------------------------------------------------------- bubble damping, depth

def test_thermal_damping_is_xue_eq5():
    # delta_th, default / legacy, r = 0.5, 1, 3, 10 mm (see research/SOURCES.md)
    for r_mm, xue, hf in ((0.5, 0.0365, 0.0296), (1.0, 0.0262, 0.0212), (3.0, 0.0154, 0.0124), (10.0, 0.0085, 0.0068)):
        r = r_mm * 1e-3
        for legacy, ref in ((False, xue), (True, hf)):
            w0, _, _, d = m.bubble_params(r, legacy=legacy)
            th = float(d) - float(w0) * r / m.C_WATER - 4 * m.MU_W / (m.RHO_W * float(w0) * r * r)
            assert abs(th / ref - 1) < 0.02, (r_mm, legacy, th)
        assert abs(m._beta_of(float(w0)) / float(m.bubble_params(r)[1]) - 1) < 0.01


def test_bubble_field_damping_follows_radius():
    # rendered decay = the radius law of bubble_params; the former fixed 0.0139 + 0.02 was 2.0x short at 15 kHz
    n = int(0.05 * SR)
    rate = np.zeros(n)
    rate[10] = SR
    for r in (0.22e-3, 1e-3):
        y, info = P.bubble_field(0.05, np.random.default_rng(0), rate=rate, r_range=(r, r * 1.0001), return_info=True)
        f = float(info["f0"][0])
        assert info["n"] == 1
        beta = float(m.bubble_params(r)[1])
        assert abs(_decay_rate(y) / beta - 1) < 0.03
        if r < 0.5e-3:
            assert abs(beta / (math.pi * f * (P.DELTA_RAD + P.DELTA_TH)) - 2.0) < 0.05


def test_minnaert_rises_with_depth():
    # p0 = P_atm + rho g z, f r x sqrt(p0 / P_atm) = 1.40 at 10 m
    assert abs(m.minnaert(1e-3, 10.0) / m.minnaert(1e-3) - math.sqrt(1 + 1000 * 9.8 * 10 / 101325)) < 1e-9
    assert abs(float(m.bubble_params(1e-3, depth=10.0)[0]) / float(m.bubble_params(1e-3)[0]) - 1.40) < 0.01
    n = int(0.3 * SR)
    y = m._render_bubbles(n, np.random.default_rng(0), np.array([100]), np.array([3e-3]), depth=(1.0, 1.0),
                          water_depth=10.0)
    sp = np.abs(np.fft.rfft(y, 1 << 18))
    f = np.argmax(sp) * SR / (1 << 18)
    assert abs(f / float(m.minnaert(3e-3, 10.0)) - 1) < 0.02


def test_bank_damping_at_depth():
    # the bank takes p0 into the radius and into psi: rendered decay at 10 m / at the surface = bubble_params' ratio
    n = int(0.3 * SR)
    for r in (1e-3, 3e-3):
        b = [_decay_rate(m._render_bubbles(n, np.random.default_rng(0), np.array([100]), np.array([r]), depth=(1.0, 1.0),
                                           water_depth=z)) for z in (0.0, 10.0)]
        want = float(m.bubble_params(r, depth=10.0)[1]) / float(m.bubble_params(r)[1])
        assert abs(b[1] / b[0] / want - 1) < 0.03, (r, b, want)


def test_xue_rise_speed():
    assert abs(m.XUE_RISE - 1.633) < 0.001                  # 1.63 sqrt(g r)


# ---------------------------------------------------------------- drop chronology

def test_first_bubble_delay_follows_drop_size():
    # terminal speed: 9.35 ms at D = 1 mm, 65 +- 5 ms above 2.2 mm (default size 0.4: D = 2.44 mm); impact width w <= 12.1 ms
    for size, lo, hi in ((0.0, 9.35, 9.35 + 12.1), (0.4, 65.0 - 15.0, 65.0 + 15.0 + 12.1),
                         (1.0, 65.0 - 15.0, 65.0 + 15.0 + 12.1)):
        for seed in range(4):
            y = m._drop(np.random.default_rng(seed), size, click=0.0, secondary=0.0)
            t = np.flatnonzero(np.abs(y) > 1e-3 * np.abs(y).max())[0] / SR * 1e3
            assert lo - 0.5 < t < hi, (size, seed, t)


def test_rebound_delay_is_ballistic():
    # 2 sqrt(2h/g), h 2.5-15 cm: 143-350 ms; on the render, the rebound drop (size 0: first bubble 9.35 ms after
    # an impact of up to 12.1 ms) starts in [143 + 9.35, 350 + 9.35 + 12.1] ms
    assert abs(m.DROP_REBOUND[0] - 0.143) < 0.001 and abs(m.DROP_REBOUND[1] - 0.350) < 0.001
    t = []
    for seed in range(20):
        a, b = (m._drop(np.random.default_rng(seed), 0.0, click=0.0, secondary=s) for s in (0.0, 1.0))
        d = np.abs(b - a)
        t.append(np.flatnonzero(d > 1e-9 * d.max())[0] / SR * 1e3)
    assert 143.0 + 9.35 - 0.5 < min(t) and max(t) < 350.0 + 9.35 + 12.1 + 0.5, t
    assert max(t) - min(t) > 100.0                                     # spread over the range


# ---------------------------------------------------------------- rain on water by class

def test_rain_classes_and_wind():
    _, a = A.rain_on_water(2.0, np.random.default_rng(0), rate_mm_h=10.0, area_m2=2.0, return_info=True)
    assert a["D"].min() >= A.RAIN_D_SILENT                  # < 0.8 mm: silent
    assert a["bubbles1"] == a["type1"] > 50                  # type I: a bubble at every normal impact
    assert 0.5 - 0.1 < a["bubbles2"] / a["type2"] < 0.65 + 0.1   # type II: 0.5-0.65
    _, w = A.rain_on_water(2.0, np.random.default_rng(0), rate_mm_h=10.0, area_m2=2.0, wind=10.0, return_info=True)
    assert w["bubbles1"] == 0                                # drizzle peak gone above 8-10 m/s


def test_rain_type1_energy_ratio():
    # bubble 0.5-5 pJ against impact 0.008-0.016 pJ, ratio 30-600
    lo = A.RAIN_T1_E_BUB[0] / A.RAIN_T1_E_IMP[1]
    hi = A.RAIN_T1_E_BUB[1] / A.RAIN_T1_E_IMP[0]
    assert 30 < lo and hi < 630
    assert abs(A.RAIN_IMP_EFF * 3.61e-6 / 0.016e-12 - 1) < 0.5     # efficiency inside the measured span


def test_rain_type1_energy_ratio_on_the_render(monkeypatch):
    # one 0.9 mm drop: rendered bubble energy / impact energy inside 30-600 (0.5-5 pJ against 0.008-0.016 pJ)
    monkeypatch.setattr(A, "rain_drops", lambda *a, **k: (np.array([0.01]), np.array([0.9])))
    s, cut = int(0.01 * SR), int(0.0105 * SR) + 64
    for seed in range(6):
        y = A.rain_on_water(0.2, np.random.default_rng(seed))
        assert np.abs(y[:s]).max() == 0.0 and np.abs(y[cut:s + int(0.009 * SR)]).max() < 1e-9 * np.abs(y).max()
        ratio = np.sum(y[cut:] ** 2) / np.sum(y[:cut] ** 2)
        assert 30 < ratio < 630, (seed, ratio)


def test_rain_level_grows_as_10_log_n():
    # SL = 10 log N + F(f); 4x the area = +6 dB
    r = [float(np.sqrt(np.mean(A.rain_on_water(3.0, np.random.default_rng(s), rate_mm_h=5.0, area_m2=a) ** 2)))
         for a in (0.5, 2.0) for s in range(3)]
    db = 20 * np.log10(np.mean(r[3:]) / np.mean(r[:3]))
    assert abs(db - 6.0) < 2.0, db


def test_rain_water_sfx_and_rate():
    from genny.sfx import render_sfx
    y = render_sfx("rain_water", sr=44100, dur=1.0, rate=5.0)
    assert np.isfinite(y).all() and abs(len(y) - 44100) < 10
    assert A.rain_rate(1.0, 3.0) == 600.0                    # 200 /s/m2 x 3 m2


def test_rain_spikes_layer_is_farnells_drops():
    # same numbers as nature.drop_spikes
    from genny import nature
    from genny.foley import P as NP
    assert NP == SR
    noise = np.random.default_rng(3).uniform(-1, 1, SR)
    a = A.drop_spikes(noise)
    b = nature.drop_spikes(SR, np.random.default_rng(3))
    assert np.count_nonzero(a) > 100
    assert np.max(np.abs(a - b)) < 1e-6 * np.max(np.abs(b))


# ---------------------------------------------------------------- leftovers

def test_surf_crest_inside_the_first_seconds():
    for seed in range(5):
        y = A.surf(6.0, np.random.default_rng(seed), loop=True)
        assert np.sqrt(np.mean(y ** 2)) > 0.02, seed           # was 0.0 for seed 0 (no wave in a 6 s loop)
    y = m.surf(dur=14.0, seed=0)
    assert np.abs(y[: 6 * 44100]).max() > 0.5 * np.abs(y).max()


def test_mains_hum_starts_at_its_level():
    y = m.mains_hum(sr=44100, dur=3.0)
    lvl = lambda a, b: 20 * np.log10(np.abs(y[int(a * 44100):int(b * 44100)]).max())
    assert abs(lvl(0.04, 0.3) - lvl(1.0, 2.0)) < 1.0         # was 2.4 dB low in the first 0.3 s


def test_surfacing_bubble_tail_closes():
    for seed in range(5):
        b = P.surfacing_bubble(np.random.default_rng(seed))
        assert np.abs(b[-50:]).max() < 1e-3 * np.abs(b).max()
        assert len(detect_clicks(b)) == 0


def test_tube_bubble_no_click():
    bad = 0
    for r in np.geomspace(5e-3, 40e-3, 12):
        for seed in range(6):
            bad += len(detect_clicks(A.tube_bubble(r, np.random.default_rng(seed)))) > 0
    assert bad == 0


def test_render_bed_without_charter():
    y = A.render_bed({"id": "amb.lib.sea_shore", "prompt": "Place: sea shore"}, np.random.default_rng(0), dur=8.0)
    assert np.isfinite(y).all() and np.sqrt(np.mean(y ** 2)) > 1e-3


def test_cave_drip_regimes():
    # fast > 100 drops / 15 min, medium 10-100, slow < 10
    # periods 900 s / count: fast < 9 s, medium 9-90 s, slow > 90 s; jitter +-12 %
    assert A.CAVE_DRIP_PERIODS["fast"][1] == 900 / 100 and A.CAVE_DRIP_PERIODS["medium"] == (900 / 100, 900 / 10)
    assert A.CAVE_DRIP_PERIODS["slow"][0] == 900 / 10
    for reg, dur in (("fast", 40.0), ("medium", 120.0)):
        lo, hi = A.CAVE_DRIP_PERIODS[reg]
        y = A.cave_drips(dur, np.random.default_rng(3), sources=1, reverb=None, regime=reg)[:, 0]
        on = np.flatnonzero(np.abs(y) > 0.1 * np.abs(y).max())
        starts = on[np.r_[True, np.diff(on) > SR]] / SR
        gaps = np.diff(starts)
        assert len(gaps) >= 5, (reg, gaps)
        assert gaps.min() > lo * 0.88 - 0.05 and gaps.max() < hi * 1.12 + 0.05, (reg, gaps)
        assert gaps.max() / gaps.min() < 1.12 / 0.88 + 0.02                 # one source: one period, jittered


def test_glass_fill_ratio_jundt():
    # n = 5.5; full glass 416 Hz against 897 Hz empty (Jundt Fig. 3); Fig. 2 at 5.5 of 10 cm: 842/887
    assert abs(m.glass_fill_ratio(1.0) - 416.0 / 897.0) < 1e-9
    assert abs(m.glass_fill_ratio(5.5 / 10.1) - 842.0 / 887.0) < 0.02
    assert m.glass_fill_ratio(0.3) > 0.99
    # on the render: the empty glass's strongest partial reappears at 416/897 of it when full
    fr = np.fft.rfftfreq(1 << 19, 1 / 48000)
    def peak(y, lo, hi):
        sp = np.abs(np.fft.rfft(y, 1 << 19))
        k = (fr > lo) & (fr < hi)
        return fr[k][np.argmax(sp[k])]
    for seed in range(3):
        f0 = peak(m.ice_cubes(sr=48000, liquid=0.0, seed=seed), 200.0, 2500.0)
        r = peak(m.ice_cubes(sr=48000, liquid=1.0, seed=seed), 0.4 * f0, 0.55 * f0) / f0
        assert abs(r / m.glass_fill_ratio(1.0) - 1) < 0.005, (seed, r)


def test_kettle_two_regimes():
    # in air, the paper's rig: Helmholtz line of whistle 2 near 1500 Hz; vortex tone 1716 Hz at U = 12.6 m/s
    nu_air = 0.051 * 10.0 / 3.5e4
    assert abs(m.kettle_helmholtz(10.2, 25.0, c=340.0) / 1500.0 - 1) < 0.05
    assert abs(float(m.kettle_tone(12.6 * 3e-3 / nu_air, 10.2, 25.0, nu=nu_air, c=340.0)) / 1716.0 - 1) < 0.06
    # rendered: constant Helmholtz note below Re 2000, St U / d above, its harmonic 15 dB down
    fr = np.fft.rfftfreq(1 << 18, 1 / 48000)
    def spec(**kw):
        y = m.kettle(sr=48000, dur=3.0, onset=0.05, breath=0.0, rumble=0.0, **kw)[48000:]
        return np.abs(np.fft.rfft(y, 1 << 18))
    lo = spec(flow=1800.0)
    assert abs(fr[np.argmax(lo)] / m.kettle_helmholtz() - 1) < 0.01
    hi = spec(flow=3000.0)
    f2 = float(m.kettle_tone(3000.0))
    assert abs(fr[np.argmax(hi)] / f2 - 1) < 0.01
    band = lambda s, f: float(np.sum(s[np.abs(fr - f) < 0.02 * f] ** 2))
    assert abs(10 * math.log10(band(hi, 2 * f2) / band(hi, f2)) + 15.0) < 1.0
    h1 = m.kettle_helmholtz()
    assert band(lo, 2 * h1) < 1e-4 * band(lo, h1)


def test_cap_burst_deane():
    # 2.49 mm bubble draining 7 ms: film ~23 um, Taylor-Culick 2.5 m/s
    h = P.film_thickness(2.49e-3, 0.007)
    assert abs(h / 23e-6 - 1) < 0.05
    assert abs(P.retraction_speed(h) / 2.5 - 1) < 0.05
    # rendered chirp: rises over pi R / (2 u), ends at c/2pi sqrt(3 pi^2 R / 16 V)
    rb, R, V = 2.49e-3, 3.82e-3, 6.47e-8
    y = P.cap_burst(rb, 0.007)
    tr = math.pi * R / (2 * P.retraction_speed(h))
    fmax = 343.0 / (2 * math.pi) * math.sqrt(3 * math.pi ** 2 * R / (16 * V))
    ph = np.unwrap(np.angle(hilbert(y)))
    fi = np.diff(ph) * 48000 / (2 * math.pi)
    u = P.retraction_speed(h)
    for x in (0.4, 0.7, 0.95):
        j = int(x * tr * 48000)
        law = fmax * math.sqrt(math.sin(u * x * tr / R))
        assert abs(np.median(fi[j - 6:j + 6]) / law - 1) < 0.05, (x, np.median(fi[j - 6:j + 6]), law)
    # longer drainage: thinner film, faster opening, shorter chirp
    assert len(P.cap_burst(rb, 0.5)) < len(P.cap_burst(rb, 0.007))


def test_hinze_radius_surf():
    assert abs(P.hinze_radius(12.0) / 1.0e-3 - 1) < 0.03
    assert abs(P.hinze_radius(A.SURF_EPS) - P.hinze_radius(12.0)) < 1e-12
    assert P.hinze_radius(1.0) > P.hinze_radius(10.0)


def test_boil_cap_bursts():
    def hf(y):
        sp = np.abs(np.fft.rfft(y)) ** 2
        fr = np.fft.rfftfreq(len(y), 1 / 48000)
        return sp[fr > 6000].sum() / sp.sum()
    a, b = (m.boil(sr=48000, dur=2.0, intensity=1.0, burst=v) for v in (0.0, 1.0))
    assert hf(b) > 3 * hf(a)
