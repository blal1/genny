"""Weather and fire: thunder channel (Ribner & Roy), Fineberg's thunder layers, tree sway (Moore & Maguire),
ice loss (Kuroiwa), lightning strokes and precursor, fire components (Farnell).

Run:  uv run --with pytest pytest tests/test_meteo_feu.py -q
"""
import math

import numpy as np

from genny import aero
from genny import matter as M
from genny.physical import ambience as amb
from genny.physical.core import SR


def _spec(y):
    return np.fft.rfftfreq(len(y), 1 / SR), np.abs(np.fft.rfft(y)) ** 2


def test_wright_medendorp_wave_broadside_and_oblique():
    T, half = 0.0055, 5.0
    ct = 343.0 * T
    side = amb.wm_wave(T, half, 0.0)
    assert abs(np.max(side) - 2 * half) < 0.02 * 2 * half                    # broadside: the N-wave times 2l
    p = amb.wm_wave(T, half, 1.0)                                              # l sin psi = 5 m > cT = 1.9 m
    assert abs(np.sum(p)) < 1e-5 * np.sum(np.abs(p))                            # the N-waves cancel overall
    quiet = np.flatnonzero(np.abs(p) < 1e-12)
    gap = (2 * half - 2 * ct) / 343.0 * SR                                      # flat between p+ and p-
    assert abs(len(quiet) - gap) < 3, (len(quiet), gap)
    i = np.argmax(p), np.argmin(p)
    assert i[0] < i[1] and abs(p[i[0]] + p[i[1]]) < 1e-4 * abs(p[i[0]])


def test_nwave_stretch_ribner_twenty_percent():
    assert abs(amb.nwave_stretch(3600.0) / amb.nwave_stretch(600.0) - 1.20) < 0.005
    assert abs(float(amb.nwave_stretch(1000.0)) - 1.0) < 1e-12


def test_ribner_deflection_and_segment_statistics():
    rng = np.random.default_rng(0)
    z = np.array([0.0, 0.0, 1.0])
    th = np.degrees(np.arccos(np.clip([amb._deflect(rng, z)[2] for _ in range(20000)], -1, 1)))
    # density sin T exp(-T^2 / 25deg^2): mode at 25 / sqrt 2 = 17.7 deg
    h, e = np.histogram(th, bins=np.arange(0, 90, 2.0))
    assert abs(e[np.argmax(h)] + 1.0 - 17.7) < 2.5
    v = amb.ribner_channel(np.random.default_rng(1))
    lens = np.linalg.norm(v, axis=1)
    assert abs(lens.mean() - 3.0) < 0.4 and abs(lens.sum() - 2000.0) < 30.0


def test_strokes_share_the_channel_and_start_60_ms_apart():
    y1 = amb.thunder_channel(np.random.default_rng(4), dur=4.0, distance_m=800.0)
    y3 = amb.thunder_channel(np.random.default_rng(4), dur=4.0, distance_m=800.0, strokes=3)
    k = int(0.05 * SR)
    a = np.dot(y3[:k], y1[:k]) / np.dot(y1[:k], y1[:k])
    assert np.max(np.abs(y3[:k] - a * y1[:k])) < 1e-3 * np.max(np.abs(y3[:k]))
    j = slice(int(0.07 * SR), int(0.3 * SR))
    assert np.max(np.abs(y3[j] - a * y1[j])) > 0.05 * np.max(np.abs(y3))


def test_channel_walk_ribner_renders():
    y = amb.thunder_channel(np.random.default_rng(2), dur=6.0, distance_m=1500.0, walk="ribner")
    assert np.all(np.isfinite(y)) and abs(np.max(np.abs(y)) - 1.0) < 1e-9


def test_fineberg_afterimage_and_deepener():
    n = 20 * SR
    a = amb.fb_afterimage(n, np.random.default_rng(1))
    f, S = _spec(a[: 4 * SR])
    k = int(10 / f[1])
    Sm = np.convolve(S, np.ones(k) / k, "same")
    assert abs(f[np.argmax(Sm)] - 333.0) < 25.0
    assert np.max(np.abs(a[int(14.0 * SR):])) < 1e-9                         # gone after 14 s
    d = amb.fb_deepener(n, np.random.default_rng(1))
    assert np.max(np.abs(d[int(18.5 * SR) + 1:])) < 1e-9                      # gone after 18.5 s
    f, S = _spec(d[: 8 * SR])
    lo, mid = S[(f > 8) & (f < 15)].mean(), S[(f > 40) & (f < 60)].mean()
    assert lo < mid / 10                                                       # high-pass at 30 Hz, not 15
    y = amb.thunder(np.random.default_rng(1), dur=20.0, fineberg=True)
    assert y.shape == (n, 2) and np.all(np.isfinite(y))


def test_tree_sway_frequency_and_drag():
    assert abs(amb.tree_sway_hz(20.0, 15.0, pine=False) - 0.40) < 0.005      # eq. 9, spruce
    assert abs(amb.tree_sway_hz(20.0, 15.0) - (0.0766 + 3.1219 * 20 / 225)) < 1e-12
    assert amb.tree_sway_hz(20.0, 15.0, pine=True) < amb.tree_sway_hz(20.0, 15.0, pine=False)
    assert abs(amb.tree_sway_hz(20.0, 15.0, snow=True) / amb.tree_sway_hz(20.0, 15.0) - 0.7) < 1e-12
    w = np.full(20 * SR, 0.3)
    lin = amb.sway_force(w, np.random.default_rng(1))[10 * SR:]
    pine = amb.sway_force(w, np.random.default_rng(1), species="scots_pine")[10 * SR:]
    assert abs(pine.mean() / lin.mean() - 0.3 ** (1.377 - 1.0)) < 0.02


def test_ice_loss_debye_peaks_and_default():
    f = np.geomspace(20.0, 40000.0, 4000)
    for temp, top, tol in ((-10.0, 3000.0, 150.0), (-31.5, 330.0, 20.0), (-40.0, 120.0, 8.0)):
        assert abs(f[np.argmax(M.ice_eta(f, temp))] - top) < tol, temp
    assert abs(f[np.argmax(M.ice_eta(f, -10.0, salt=True))] - 9600.0) < 500.0
    assert abs(float(M.ice_eta(3040.0, -10.0)) - (0.009 + 0.002)) < 1e-4
    y, fr = M.ice_chirp(60.0, 0.05, return_freq=True)
    y0 = M.ice_chirp(60.0, 0.05, loss=0.0)
    i = len(y) // 3
    t = np.arange(len(y)) / SR
    assert abs(y[i] / y0[i] - math.exp(-0.0045 * 2 * math.pi * fr[i] * t[i])) < 1e-6
    for kw in (dict(temp=-10.0), dict(temp=-30.0, salt=1)):
        z = M.ice_crack(SR, seed=1, **kw)
        assert np.all(np.isfinite(z))


def test_lightning_strokes_reuse_the_first_and_precursor_leads():
    one = M.lightning(SR, dur=3.0, distance=150.0, seed=2)
    three = M.lightning(SR, dur=3.0, distance=150.0, seed=2, strokes=3)
    k = int(0.055 * SR)
    a = np.dot(three[:k], one[:k]) / np.dot(one[:k], one[:k])
    assert np.max(np.abs(three[:k] - a * one[:k])) < 1e-6 * np.max(np.abs(three))
    pre = M.lightning(SR, dur=3.0, distance=150.0, seed=2, precursor=1.0)
    assert abs((np.argmax(np.abs(pre)) - np.argmax(np.abs(one))) / SR - M.PRECURSOR_S) < 0.005
    assert np.max(np.abs(pre[: int(0.25 * SR)])) > 0.05 * np.max(np.abs(pre))
    ch = M.lightning(SR, dur=3.0, distance=900.0, seed=2, model=1, strokes=2)
    assert np.all(np.isfinite(ch))


def test_fire_components():
    base = M.fire(SR, dur=4.0, seed=1)
    f, S0 = _spec(base)
    woof = M._fire_woof(4 * SR, np.random.default_rng(0), np.ones(4 * SR), 0.4)
    f, Sw = _spec(woof)
    assert Sw[(f >= 3) & (f <= 160)].sum() > 0.98 * Sw.sum()                # 3-80 Hz air induction, 2nd-order skirts
    y = M.fire(SR, dur=4.0, seed=1, woof=1.0)
    _, S1 = _spec(y)
    lo = lambda S: S[(f >= 3) & (f < 80)].sum() / S.sum()  # noqa: E731
    assert lo(S1) > lo(S0) + 0.03
    wh = M._fire_whine(4 * SR, np.random.default_rng(0), None, 0.4)
    seg = wh[np.flatnonzero(np.abs(wh) > 0.3 * np.max(np.abs(wh)))[0]:][: int(0.1 * SR)]
    ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
    lag = np.argmax(ac[int(SR / 2300):int(SR / 400)]) + int(SR / 2300)
    assert ac[lag] / ac[0] > 0.8 and 400 < SR / lag < 2300                     # strongly pitched
    cl = M.fire(SR, dur=4.0, seed=1, clatter=1.0)
    crest = lambda v: np.max(np.abs(v)) / np.sqrt(np.mean(v * v))  # noqa: E731
    assert crest(cl) > crest(base)
    med = M.fire(SR, dur=4.0, seed=1, detail="medium")
    assert np.array_equal(med, M.fire(SR, dur=4.0, seed=1, detail="medium", whine=1.0, woof=1.0, clatter=1.0))
    assert not np.array_equal(med, base)


def test_rain_wind_incidence_kills_type1_bubbles():
    calm = amb.rain_on_water(4.0, np.random.default_rng(3), rate_mm_h=20.0, return_info=True)[1]
    assert calm["bubbles1"] == calm["type1"] > 20
    windy = amb.rain_on_water(4.0, np.random.default_rng(3), rate_mm_h=20.0, wind=1.33, return_info=True)[1]
    assert windy["bubbles1"] < 0.1 * windy["type1"]                            # under 5 % at 20 degrees (Kurgan)


def test_scintillation_saturates_under_6_db_and_covariance():
    sat = float(aero.scintillation_sigma(1e4, 1e9))
    assert abs(sat ** 2 - math.sqrt(math.pi) / 4) < 1e-3
    assert 20 * math.log10(math.e) * sat < 6.0
    s, fs = aero.turbulence_sequence(2200.0, 0.55, np.random.default_rng(0))
    assert fs == 5 / 0.55
    rho = float(np.mean(s[:-5] * s[5:]))
    assert abs(rho - 0.5 * math.sqrt(math.pi) * math.erf(1.0)) < 0.03          # C(tau0)


def test_scintillation_keeps_the_mean_power():
    # <chi> = -sigma^2: mean power kept in a saturated band (exp(sigma s) alone gave exp(2 sigma^2) = +3.8 dB)
    from scipy.signal import butter, sosfiltfilt
    b = sosfiltfilt(butter(4, (5400.0, 6600.0), "bandpass", fs=SR, output="sos"),
                    np.random.default_rng(0).standard_normal(30 * SR))
    assert abs(float(aero.scintillation_sigma(6000.0, 5000.0)) ** 2 - math.sqrt(math.pi) / 4) < 0.01
    db = [10 * math.log10(np.mean(aero.scintillate(b, 5000.0, 20.0, np.random.default_rng(s)) ** 2) / np.mean(b ** 2))
          for s in range(3)]
    assert abs(np.mean(db)) < 1.5, db


def test_scintillation_log_amplitude_grows_with_frequency():
    from scipy.signal import hilbert
    t = np.arange(40 * SR) / SR
    for f0, fc in ((100.0, 125.0), (8000.0, math.sqrt(4000.0 * 24000.0))):
        z = aero.scintillate(np.sin(2 * np.pi * f0 * t), 1000.0, 2.0, np.random.default_rng(1))
        got = np.std(np.log(np.abs(hilbert(z))[SR:-SR]))
        assert abs(got / float(aero.scintillation_sigma(fc, 1000.0)) - 1) < 0.12
    r0 = aero.space_rocket(dur=2.0)
    assert np.array_equal(r0, aero.space_rocket(dur=2.0, turbulence=0.0))
    assert np.all(np.isfinite(aero.space_rocket(dur=2.0, turbulence=3.0)))


def test_arc_chirps_peak_near_5_khz_and_default_unchanged():
    c = M.arc_chirps(4 * SR, 99.8, np.random.default_rng(0))
    f, S = _spec(c)
    assert 4000.0 < np.sum(f * S) / np.sum(S) < 6000.0
    a = M.arc(seed=1)
    assert np.array_equal(a, M.arc(seed=1, click=0.0))
    assert np.max(np.abs(M.arc(seed=1, click=0.8) - a)) > 0.05


def test_ice_c_axis_parallel_one_fifth_of_the_peak():
    tau = 6.9e-16 * math.exp(54810.0 / (8.314 * 263.15))
    fp = 1 / (2 * math.pi * tau)
    full = M.ice_eta(fp, -10.0) - M.ICE_ETA_FLOOR
    par = M.ice_eta(fp, -10.0, c_parallel=True) - M.ICE_ETA_FLOOR
    assert abs(full - 0.009) < 1e-9 and abs(par / full - 0.2) < 1e-9
    assert np.array_equal(M.ice_crack(seed=2), M.ice_crack(seed=2, c_axis=0))


def test_sfx_thunder_exposes_strokes_walk_fineberg():
    from genny import sfx
    from genny.physics import _to_sr
    base = sfx.thunder(sr=SR, dur=3.0, seed=1)
    assert np.array_equal(base, sfx.thunder(sr=SR, dur=3.0, seed=1, strokes=1, walk="farnell", fineberg=0))
    y = sfx.thunder(sr=SR, dur=3.0, channel=1.0, strokes=2, walk="ribner", seed=1)
    ch = _to_sr(amb.thunder_channel(np.random.default_rng(33), dur=3.0, distance_m=80.0, strokes=2, walk="ribner"), SR)
    k = int(2.9 * SR)
    assert np.corrcoef(y[:k], ch[:k])[0, 1] > 0.999
    y = sfx.thunder(sr=SR, dur=3.0, channel=0.0, fineberg=1, seed=1)
    ref = amb.thunder(np.random.default_rng(33), dur=3.0, distance=80.0 / 3000.0, fineberg=True).mean(axis=1)
    assert np.corrcoef(y[:k], ref[:k])[0, 1] > 0.999


def test_reverbs_scintillation_keeps_the_mean_power():
    from scipy.signal import butter, sosfiltfilt
    from genny import reverbs
    b = sosfiltfilt(butter(4, (5400.0, 6600.0), "bandpass", fs=SR, output="sos"),
                    np.random.default_rng(0).standard_normal(20 * SR))
    db = [10 * math.log10(np.mean(reverbs._scintillate(b, SR, 5000.0, 20.0, 343.0, np.random.default_rng(s)) ** 2)
                          / np.mean(b ** 2)) for s in range(3)]
    assert abs(np.mean(db)) < 1.5, db
