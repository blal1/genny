"""Electric and sci-fi weapons: what the spec sources, measured on the signal. Timbre is UNSOURCED, only checked by measure.

Clicks: every variant is scanned whole. Where a shock front exists by physics (N-wave, blast), each flagged sample must lie on a front computed from the model; anything else fails.
"""
import math

import numpy as np
import pytest

from scipy.signal import butter, sosfiltfilt

from genny import energy_arms as E
from genny.analysis import detect_clicks
from genny.arms import n_wave_pa, shock_arrival
from genny.foley import C_AIR
from genny.physics import blast_wave
from genny.sfx import REGISTRY

SR = 48000
LEAD = E.LEAD


def _env(y, ms=2.0):
    k = int(ms * 1e-3 * SR)
    return np.sqrt(np.convolve(y * y, np.ones(k) / k, mode="same"))


def _peak(y, lo, hi):
    s = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / SR)
    m = (f >= lo) & (f <= hi)
    return float(f[m][np.argmax(s[m])])


def _centroid(y):
    s = np.abs(np.fft.rfft(y)) ** 2
    return float(np.sum(np.fft.rfftfreq(len(y), 1 / SR) * s) / np.sum(s))


def _share(y, lo, hi):
    s = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return float(s[(f >= lo) & (f <= hi)].sum() / s.sum())


def _period(e, lo, hi):
    """Lag in s of the strongest autocorrelation of an envelope between lo and hi s."""
    e = e - e.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(e, 2 * len(e))) ** 2)
    a, b = int(lo * SR), int(hi * SR)
    return (a + int(np.argmax(ac[a:b]))) / SR


def _clean(y):
    assert np.isfinite(y).all() and np.abs(y).max() > 0.01
    assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3


SMOOTH = {
    "arc100": lambda s=0: E.arc_weapon(sr=SR, seed=s),
    "arc400": lambda s=0: E.arc_weapon(sr=SR, buzz=400.0, seed=s),
    "phaser_kill": lambda s=0: E.energy_weapon(sr=SR, kind="phaser", seed=s),
    "phaser_stun": lambda s=0: E.energy_weapon(sr=SR, kind="phaser", power="stun", seed=s),
    "beam": lambda s=0: E.energy_weapon(sr=SR, kind="beam", seed=s),
    "bolt": lambda s=0: E.plasma(sr=SR, seed=s),
    "impact": lambda s=0: E.plasma(sr=SR, kind="impact", seed=s),
    "charge": lambda s=0: E.railgun(sr=SR, kind="charge", seed=s),
    "laser_impact": lambda s=0: E.laser_impact(sr=SR, seed=s),
}


@pytest.mark.parametrize("name", SMOOTH)
@pytest.mark.parametrize("seed", [0, 1])
def test_no_click_on_the_whole_signal(name, seed):
    y = SMOOTH[name](seed)
    _clean(y)
    assert len(detect_clicks(y)) == 0, (name, (detect_clicks(y)[:8] / SR).tolist())


def _crack_edges(v, d, l, b, at):
    """Front and back shock of the N-wave and of its ground reflection, in s from the first sample."""
    _, T = n_wave_pa(v, d, l, b)
    dt = (math.hypot(b, 3.0) - b) / C_AIR
    return [at, at + T, at + dt, at + dt + T]


def _on_edges(y, edges, slack=0.0006):
    c = detect_clicks(y, window=min(441, len(y) - 2)) / SR
    stray = [float(t) for t in c if min(abs(t - e) for e in edges) > slack]
    assert not stray, stray[:8]


@pytest.mark.parametrize("kind,tone", [("contact", 0.6), ("arc_gap", 1.0)])
@pytest.mark.parametrize("seed", [0, 1])
def test_stun_clicks_are_only_its_discharges(kind, tone, seed):
    y = E.stun_gun(sr=SR, kind=kind, seed=seed)
    _clean(y)
    _, times = E.arc_train(1.0, E.STUN_PPS, 0.1, tone, np.random.default_rng(seed))
    _on_edges(y, list(times), slack=0.002)                  # the formant rings for about 2 ms


@pytest.mark.parametrize("seed", [0, 1])
def test_taser_clicks_are_only_cartridge_and_discharges(seed):
    y = E.taser(sr=SR, seed=seed)
    _clean(y)
    _, times = E.arc_train(1.0, E.STUN_PPS, 0.1, 0.6, np.random.default_rng(seed))
    _on_edges(y, [0.0] + list(times + 0.2), slack=0.002)


@pytest.mark.parametrize("seed", [0, 1])
def test_prod_clicks_are_only_its_discharges(seed):
    y = E.cattle_prod(sr=SR, seed=seed)
    _clean(y)
    _, times = E.arc_train(1.2, 60.0, 0.8, 0.85, np.random.default_rng(seed))
    _on_edges(y, list(times), slack=0.002)


def test_blaster_clicks_are_only_its_strike():
    y = E.energy_weapon(sr=SR)
    _clean(y)
    _on_edges(y, [0.0], slack=0.0012)                       # the wrench on the cable, inside the 1 ms attack


def test_railgun_flyby_clicks_are_only_its_n_wave():
    y = E.railgun(sr=SR, kind="flyby")
    _clean(y)
    _on_edges(y, _crack_edges(3390.0, E.RAIL_D_MM, E.RAIL_L_MM, E.RAIL_FAR, LEAD / SR))


def test_railgun_fire_clicks_are_only_shock_fronts():
    y = E.railgun(sr=SR)
    _clean(y)
    v, r = 3390.0, 100.0
    t_c = shock_arrival(v, r, 0.5)
    t0 = LEAD / SR + r / C_AIR - t_c
    td = blast_wave(E.RAIL_W, r)[1]
    dt = (math.hypot(r, 3.0) - r) / C_AIR
    edges = _crack_edges(v, E.RAIL_D_MM, E.RAIL_L_MM, E.RAIL_FAR, LEAD / SR) + [t0 + a + b for a in (0.0, td, 4 * td) for b in (0.0, dt)]
    _on_edges(y, edges)


@pytest.mark.parametrize("kw", [dict(), dict(kind="silent"), dict(switch="igbt"), dict(switch="scr")])
def test_coilgun_clicks_are_only_its_stages(kw):
    y = E.coilgun(sr=SR, **kw)
    _clean(y)
    _on_edges(y, list(LEAD / SR + E.stage_times(75.0)))


def test_gauss_clicks_are_only_stages_and_crack():
    y = E.gauss_rifle(sr=SR)
    _clean(y)
    t = E.stage_times(400.0, 5)
    d, l = E.COIL_D_MM, E.GAUSS_MASS / 7850.0 / (math.pi * (E.COIL_D_MM * 5e-4) ** 2) * 1e3
    at = LEAD / SR + t[-1] + E.crack_delay(400.0, E.COIL_FAR)
    _on_edges(y, list(LEAD / SR + t) + _crack_edges(400.0, d, l, E.COIL_FAR, at))


def test_real_laser_click_is_its_front():
    y = E.energy_weapon(sr=SR, kind="real")
    _clean(y)
    _on_edges(y, [E.BREAK_AT / SR, (E.BREAK_AT + 2) / SR], slack=0.0001)


def test_names_are_registered_and_unknown_kinds_refused():
    assert {"stun_gun", "taser", "cattle_prod", "arc_weapon", "energy_weapon", "plasma", "railgun", "coilgun", "gauss_rifle", "laser_impact"} <= set(REGISTRY)
    for fn, kw in ((E.stun_gun, dict(kind="x")), (E.taser, dict(kind="x")), (E.energy_weapon, dict(kind="pew")), (E.plasma, dict(kind="x")), (E.railgun, dict(kind="x")), (E.coilgun, dict(kind="x")),
                   (E.coilgun, dict(switch="x")), (E.energy_weapon, dict(kind="phaser", power="x"))):
        with pytest.raises(ValueError):
            fn(sr=SR, **kw)


# Stun guns (W)
@pytest.mark.parametrize("rate", [19.0, 20.0, 29.0])
def test_stun_rate_on_the_signal(rate):
    y = E.stun_gun(sr=SR, rate=rate, dur=3.0)
    assert abs(_period(_env(y), 0.6 / rate, 1.5 / rate) - 1.0 / rate) < 0.05 / rate


def test_stun_discharge_excitation_is_100_us_and_none_in_the_fade():
    assert E._n(E.PULSE) == round(1e-4 * SR)
    _, times = E.arc_train(2.0, 19.0, 0.1, 1.0, np.random.default_rng(0))
    assert times.max() < 2.0 - 0.05 and abs(len(times) / 1.95 - 19.0) < 1.0


def test_contact_is_duller_than_open_air():
    a, b = E.stun_gun(sr=SR, dur=2.0), E.stun_gun(sr=SR, kind="arc_gap", dur=2.0)
    assert _centroid(a) < 0.8 * _centroid(b)


def test_taser_flight_is_distance_over_55_ms():
    assert abs(E.taser_flight(11.0) - 0.2) < 1e-9 and abs(E.taser_flight(50.0) - 0.2) < 1e-9     # 11 m at most
    for d in (11.0, 5.5):
        y = E.taser(sr=SR, distance=d, dur=1.0, seed=2)
        e = _env(y, 1.0)
        t = d / 55.0
        i = int(0.05 * SR)                                 # past the cartridge
        first = (i + np.argmax(e[i:] > 0.25 * e[i:].max())) / SR
        assert abs(first - t) < 0.012, first
        assert e[int(0.3 * t * SR):int(0.9 * t * SR)].max() < 0.3 * e[i:].max()   # only the wire in flight, thin
        assert abs(_period(e[int(t * SR):], 0.6 / 19, 1.5 / 19) - 1 / 19) < 0.005


def test_cattle_prod_is_more_chopped_than_the_stun_gun():
    y = E.cattle_prod(sr=SR, dur=3.0, seed=1)
    e = _env(y, 20.0)[SR // 10:-SR // 10]
    assert np.quantile(e, 0.9) > 5 * np.quantile(e, 0.1)
    z = _env(E.arc_weapon(sr=SR, dur=3.0, seed=1), 20.0)[SR // 10:-SR // 10]
    assert np.quantile(z, 0.9) < 2 * np.quantile(z, 0.1)


@pytest.mark.parametrize("buzz", [100.0, 400.0])
def test_arc_buzz(buzz):
    y = E.arc_weapon(sr=SR, buzz=buzz, dur=2.0)
    assert abs(_period(_env(y, 0.5), 0.7 / buzz, 1.3 / buzz) - 1 / buzz) < 0.03 / buzz


# Energy weapons
def test_blaster_chirps_downward():
    """Highs first, a falling chirp (Bu): each band arrives later going down, t about 1 / sqrt(f) for a bar-like dispersion (coil_spring's law)."""
    y = E.energy_weapon(sr=SR)
    def arrival(lo, hi):
        b = sosfiltfilt(butter(4, [lo, hi], "bandpass", fs=SR, output="sos"), y)
        return np.argmax(_env(b, 2.0)) / SR
    t = [arrival(*b) for b in ((2500, 4000), (1200, 2000), (600, 1000), (300, 500))]
    assert t[0] < t[1] < t[2] < t[3] and t[3] < 0.05
    ratio = t[3] / t[0]
    assert abs(ratio - math.sqrt(3162 / 387)) < 0.7             # geometric band centres


def test_phaser_lines_and_levels():
    n = SR
    t = (np.arange(n) + 1) / SR                                # cumsum's phase after one step
    x = E.phaser_oscillators(n)
    expect = sum(g * np.sin(2 * np.pi * f * t + d / r * (1 - np.cos(2 * np.pi * r * t))) for grp, g in zip(E.PHASER_FM, E.PHASER_GROUPS) for f, d, r in grp)
    assert np.corrcoef(x, expect)[0, 1] > 0.99                 # f + dev sin(2 pi r t), the course's form
    y = E.energy_weapon(sr=SR, kind="phaser", noise=0.0, dur=2.0)
    s = np.abs(np.fft.rfft(y[SR // 2:SR // 2 + SR])) ** 2      # 1 Hz bins
    on = sum(s[2446 + 47 * k] for k in range(-3, 4))
    off = sum(s[2446 + 47 * k + 23] for k in range(-3, 4))
    assert on > 3 * off                                        # sidebands at 47 Hz around 2446 Hz
    assert 1280 <= _peak(y, 900, 1800) <= 1600 and 2000 <= _peak(y, 1900, 3400) <= 3030
    assert _share(y, 0, 1000) < 0.02                           # hip 1200


def test_phaser_stun_is_half_kill():
    kill = E.energy_weapon(sr=SR, kind="phaser", noise=0.0, dur=1.0)
    stun = E.energy_weapon(sr=SR, kind="phaser", noise=0.0, power="stun", dur=1.0)
    assert abs(np.std(kill) / np.std(stun) - 2.0) < 0.02


def test_beam_is_a_steady_beating_hum():
    y = E.energy_weapon(sr=SR, kind="beam", dur=12.0)
    mid = y[SR:-SR]
    s = np.abs(np.fft.rfft(mid * np.hanning(len(mid))))
    f = np.fft.rfftfreq(len(mid), 1 / SR)                      # 0.1 Hz bins
    m = (f > 98) & (f < 102)
    top = sorted(f[m][np.argsort(s[m])[-2:]])
    assert abs(top[0] - 99.8) < 0.11 and abs(top[1] - 100.2) < 0.11     # force_field's two phasors .4 Hz apart, on the signal
    e = _env(y, 100.0)[SR:-SR]
    assert e.max() < 4 * e.min()                               # beats, never dies
    z = _peak(E.energy_weapon(sr=SR, kind="beam", dur=4.0, sweep=1.0)[-SR // 2:], 100, 400)
    assert 170 < z < 210                                       # an octave up at the end


def test_real_laser_is_short_and_up_near_20_khz():
    y = E.energy_weapon(sr=SR, kind="real")
    i = np.flatnonzero(np.abs(y) > 1e-3 * np.abs(y).max())
    assert 0.002 <= (i[-1] - i[0]) / SR <= 0.005
    assert _share(y, 10000, 24000) > 0.5 and _centroid(y) > 12000


# Plasma, all UNSOURCED: the spec's structure
def test_plasma_bolt_structure():
    y = E.plasma(sr=SR, seed=1)
    a, b = int(E.PLASMA_CHARGE * SR), int((E.PLASMA_CHARGE + E.PLASMA_FLIGHT) * SR)
    flight = y[a + SR // 20:b - SR // 20]
    env = _env(flight, 2.0)
    assert abs(_peak(env - env.mean(), 20, 90) - 40.0) < 2.0     # noise modulated at 40 Hz
    assert _centroid(y[a - 2000:a]) > _centroid(y[1000:3000])    # the charge climbs
    assert np.abs(y[b:b + SR // 10]).max() > 2 * np.std(flight)  # the hit
    assert np.std(y[b + SR // 5:b + SR // 2]) > 0.002            # its vapour


def test_plasma_impact_has_arc_then_vapour():
    y = E.plasma(sr=SR, kind="impact", seed=1)
    e = _env(y, 5.0)
    assert np.argmax(e) < 0.05 * SR and e[int(0.6 * SR)] > 0.003 and e[int(0.6 * SR)] < e.max() / 3


# Railgun
def test_rail_figures():
    assert abs(E.rail_energy(3390.0, 3.2) - 18.4e6) < 0.05e6 and abs(E.rail_energy(2520.0, E.RAIL_SHOTS["2008"][1]) - 10.64e6) < 0.05e6
    m, a = E.mach_cone(2520.0)
    assert abs(m - 7.35) < 0.02 and abs(a - 7.8) < 0.05
    m, a = E.mach_cone(3390.0)
    assert abs(m - 9.88) < 0.02 and abs(a - 5.8) < 0.05


def test_rail_charge_climbs_500_to_4000():
    y = E.railgun(sr=SR, kind="charge")
    w = SR // 12
    assert abs(_peak(y[SR // 40:SR // 40 + w], 200, 1200) - 500.0 * 8 ** ((SR // 40 + w / 2) / (1.5 * SR))) < 60
    assert abs(_peak(y[int(1.5 * SR) - w:int(1.5 * SR)], 2000, 6000) - 4000.0) < 300
    assert abs(len(y) / SR - 1.62) < 0.02                      # 1.5 s and the ready blip


@pytest.mark.parametrize("v,lag", [(2520.0, 0.252), (3390.0, 0.262)])
def test_rail_crack_leads_the_muzzle_by_252_262_ms_at_100_m(v, lag):
    y = E.railgun(sr=SR, speed=v, seed=1)
    e = _env(y, 0.5)
    crack = np.argmax(e > 0.5 * e[:int(0.1 * SR)].max())
    muzzle = int(0.2 * SR) + np.argmax(e[int(0.2 * SR):] > 0.2 * e[int(0.2 * SR):].max())
    assert abs((muzzle - crack) / SR - lag) < 0.003, (muzzle - crack) / SR


def test_rail_off_the_cone_has_no_crack():
    y, z = E.railgun(sr=SR, azimuth=120.0, seed=1), E.railgun(sr=SR, azimuth=0.0, seed=1)
    w = lambda a: np.std(a[int(0.02 * SR):int(0.2 * SR)])
    assert w(z) < 0.01 < w(y)                                  # on the line, silence between crack and muzzle; behind, the muzzle opens at once


@pytest.mark.parametrize("miss", [60.0, 240.0])
def test_flyby_n_wave_duration_is_whitham(miss):
    y = E.railgun(sr=SR, kind="flyby", miss=miss)
    _, T = n_wave_pa(3390.0, E.RAIL_D_MM, E.RAIL_L_MM, miss)
    a = int(np.argmax(np.abs(y) > 0.5 * np.abs(y).max()))
    b = a + 3 + int(np.argmax(np.diff(y[a + 3:a + int(2 * T * SR)])))     # the back shock is the largest jump up
    assert abs((b - a) / SR - T) < 2.0 / SR, ((b - a) / SR, T)


# Coilgun
@pytest.mark.parametrize("v", [45.0, 75.0])
def test_chirp_ends_at_v_over_p(v):
    f_end = v / E.COIL_PITCH
    assert 960 <= f_end <= 1610                                 # spec: 960-1600 Hz, on its rounded 46.7 mm
    y = E.coilgun(sr=SR, speed=v, switch="igbt")
    t = E.stage_times(v)
    assert abs(t[-1] - 2 * E.COIL_LEN / v) < 1e-9
    # On the signal: the envelope of the last 20 ms repeats at the local rate, which runs at v / p to within the last ten stages' acceleration.
    a, b = int((LEAD / SR + t[-11]) * SR), int((LEAD / SR + t[-1]) * SR) + 30
    e = _env(y[a:b], 0.1)
    rate = 1 / _period(e, 0.7 / f_end, 1.4 / f_end)
    assert abs(rate - 10 / (t[-1] - t[-11])) < 0.04 * rate
    assert abs(10 / (t[-1] - t[-11]) - f_end) < 0.08 * f_end


def test_chirp_rises_as_sqrt_k():
    t = E.stage_times(75.0)
    assert abs(t[0] / t[-1] - math.sqrt(1 / 45)) < 1e-12
    gaps = np.diff(t)
    assert (np.diff(gaps) < 0).all()                            # always speeding up
    assert abs(E.COIL_PITCH * 1e3 - 46.67) < 0.05
    v_end = E.COIL_PITCH / (t[-1] - t[-2])
    assert abs(v_end - 75.0) < 1.0


def test_silent_is_silent_unless_supersonic():
    fire, quiet = E.coilgun(sr=SR), E.coilgun(sr=SR, kind="silent")
    assert np.abs(quiet).max() < 0.15 * np.abs(fire).max()
    sonic = E.coilgun(sr=SR, kind="silent", speed=500.0, stages=45)
    assert np.abs(sonic).max() > 5 * np.abs(quiet).max()


def test_gauss_has_a_crack_only_above_c():
    assert E.GAUSS_V > C_AIR
    a = E.gauss_rifle(sr=SR)
    b = E.gauss_rifle(sr=SR, speed=300.0)
    t = E.stage_times(400.0, 5)
    at = LEAD / SR + t[-1] + E.crack_delay(400.0, E.COIL_FAR)
    w = a[int((at - 0.0005) * SR):int((at + 0.002) * SR)]
    assert np.abs(w).max() > 0.5
    assert E.crack_delay(300.0, E.COIL_FAR) is None             # the spec's 300 m/s is subsonic
    # What is left is the 12 Hz high-pass's residue, measured decaying at its 13 ms: no front in it. An N-wave's back shock jumps about 1 in one sample.
    tail = b[LEAD + int((E.stage_times(300.0, 5)[-1] + 0.005) * SR):]
    assert np.abs(np.diff(tail)).max() < 0.001
    assert np.abs(np.diff(a[int((at - 0.0005) * SR):int((at + 0.01) * SR)])).max() > 0.5


@pytest.mark.parametrize("sr", [44100, 22050])
def test_other_sample_rates_keep_time(sr):
    for fn, kw, dur in ((E.stun_gun, dict(), 1.15), (E.taser, dict(), 1.35), (E.energy_weapon, dict(kind="phaser"), 1.0), (E.railgun, dict(kind="charge"), 1.62), (E.plasma, dict(kind="impact"), 0.9)):
        y = fn(sr=sr, **kw)
        assert np.isfinite(y).all() and abs(len(y) / sr - dur) < 0.02, fn.__name__


def test_laser_impact_scales_with_energy():
    lo, hi = E.laser_impact(sr=SR, energy=0.0), E.laser_impact(sr=SR, energy=1.0)
    def last(y):
        e = _env(y, 5.0)
        return np.flatnonzero(e > 0.02 * e.max())[-1] / SR
    assert last(hi) > 2 * last(lo)
    assert _centroid(hi[:int(0.05 * SR)]) > 2000
    assert np.argmax(_env(hi, 1.0)) < 0.02 * SR                 # the spark is first


def test_photon_torpedo_recipe():
    from genny import energy_arms as EA
    e = EA.torpedo_env(np.array([0.0, 0.045, 0.09, 0.121, 0.151, 0.251, 0.351, 1.0]))
    assert np.allclose(e, [1.0, 0.5, 0.0, 0.445, 0.89, 0.445, 0.0, 0.0])
    assert EA.TORPEDO_F0 * EA.TORPEDO_SPAN ** 1.0 == 18900.0                  # 2700 x 7 at the top of the ping
    lens = EA.TORPEDO_FDN[0]
    assert np.allclose(np.array(lens) / 79.0, [1.0, 1.618, 2.618, 1 / 1.618], atol=2e-3)
    y = EA.photon_torpedo(48000, dur=1.0)
    w = y[:int(0.02 * 48000)]
    X = np.abs(np.fft.rfft(w * np.hanning(len(w))))
    f = np.fft.rfftfreq(len(w), 1 / 48000)
    assert f[np.argmax(X)] > 8000.0                                             # the first 20 ms sit high (f > 2700 x 7^0.78)
    assert np.isfinite(y).all() and np.abs(y[int(0.5 * 48000):]).max() > 0.01   # the loop and the FDN keep it ringing
