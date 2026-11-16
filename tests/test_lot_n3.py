"""Lot 34 N3: the closing points of N2, K2, X-A, X-B and X-C, each measured on a rendered signal or on its closed form."""
import math

import numpy as np

SR = 48000


def _peak(y, lo, hi, sr=SR):
    s = np.abs(np.fft.rfft(np.asarray(y) * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1.0 / sr)
    m = (f >= lo) & (f <= hi)
    return float(f[m][np.argmax(s[m])])


def test_cricket_falls_over_the_whole_chirp():
    from genny import creatures as C
    y = C._cricket(C.KSR, 2.0, np.random.default_rng(0), "field", 0.0)
    w = int(0.006 * C.KSR)

    def centroid(t0):
        i = int(t0 * C.KSR)
        s = np.abs(np.fft.rfft(y[i:i + w] * np.hanning(w), 8192)) ** 2
        f = np.fft.rfftfreq(8192, 1.0 / C.KSR)
        m = (f > 3000) & (f < 6000)
        return float((f[m] * s[m]).sum() / s[m].sum())

    first, last = centroid(0.004), centroid(0.105)
    assert 0.86 < last / first < 0.92, (first, last)          # 4.5 -> 3.9 kHz over the chirp, not a drop inside every pulse


def test_fly_second_wing_wanders():
    from genny import creatures as C
    a = C.fly(sr=SR, dur=1.0, seed=3)
    orig = C._slow
    C._slow = lambda n, rng, hz: np.zeros(n) if hz == 5.0 else orig(n, rng, hz)
    try:
        b = C.fly(sr=SR, dur=1.0, seed=3)
    finally:
        C._slow = orig
    assert np.all(np.isfinite(a)) and not np.allclose(a, b)


def test_bird_call_farnell_source_follows_freq():
    from genny import creatures as C
    lo = C.bird_call(SR, freq=300.0, dur=0.6, source="farnell")
    hi = C.bird_call(SR, freq=700.0, dur=0.6, source="farnell")
    assert np.all(np.isfinite(lo)) and len(lo) > 1000

    def centroid(y):
        s = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(len(y), 1.0 / SR)
        return float((f * s).sum() / s.sum())

    assert centroid(hi) > centroid(lo)
    assert not np.allclose(lo[:2000], C.bird_call(SR, freq=300.0, dur=0.6)[:2000])


def test_cossack_choir_is_looser_than_classical():
    from genny import choir
    st = choir.STYLES
    assert abs(2.2 * st["cossack"]["scatter"] - 60.0) < 1.0 and st["cossack"]["fscat"] == 0.06
    rc = choir._roster(("bass",), 60, st["cossack"], 1, False, 0.5)
    rk = choir._roster(("bass",), 60, st["classical"], 1, False, 0.5)
    sd = lambda r: float(np.std([v for s in r for v in s["fdev"]]))      # noqa: E731
    assert max(abs(s["detune"]) for s in rc) <= 62.0 and np.std([s["detune"] for s in rc]) > 2.5 * np.std([s["detune"] for s in rk])
    assert 0.045 < sd(rc) < 0.075 and sd(rk) < 0.03
    y = choir.voice(220.0, 0.5, 16000, 1.0, voice="bass", vowel="ah", style="cossack", seed=1)
    assert np.all(np.isfinite(y)) and np.max(np.abs(y)) > 0


def test_roughness_numpy_model_peaks_at_70_hz_and_grows_with_depth():
    from genny import calibrate
    t = np.arange(SR) / SR
    am = lambda fm, d=1.0: np.sin(2 * np.pi * 1000 * t) * (1 + d * np.sin(2 * np.pi * fm * t))      # noqa: E731
    r = {fm: calibrate.roughness(am(fm), SR) for fm in (4, 30, 70, 150, 400)}
    assert abs(r[70] - 1.0) < 0.05 and r[70] > 3 * r[30] and r[70] > 3 * r[150] and r[4] < 0.05 and r[400] < 0.05
    d = [calibrate.roughness(am(70, k), SR) for k in (0.0, 0.25, 0.5, 1.0)]
    assert d[0] < 0.01 and d[0] < d[1] < d[2] < d[3]


def test_scream_roughness_rose_with_the_new_preset():
    from genny import calibrate
    from genny.physical import nonverbal as nv
    key = ("scream", "female")
    new = nv.vocalize("scream", "female", np.random.default_rng(0))
    old = dict(nv.PRESETS[key])
    nv.PRESETS[key] = dict(old, amFreq=400, amDep=20)
    try:
        was = nv.vocalize("scream", "female", np.random.default_rng(0))
    finally:
        nv.PRESETS[key] = old
    assert calibrate.roughness(new, nv.SR) > 1.5 * calibrate.roughness(was, nv.SR)


def test_limiter_holes_hold_the_note_numbers_at_the_rpm_where_they_agree():
    from genny import cyber
    per, frac, skip = cyber.limiter_skips(12500.0)
    assert 0.030 <= per <= 0.060 and 0.2 <= frac <= 0.6 and abs(skip - 5.2) < 0.1
    assert 1.0 <= cyber.limiter_skips(6000.0)[2] <= 3.0
    ramp, st = cyber._limiter_run(2 * SR, 11000.0, 12500.0, 8000.0, 6000.0, 150.0, False)
    edges = np.flatnonzero(np.diff((st == 2).astype(int)) == 1)
    assert abs(float(np.diff(edges).mean()) / SR - per) < 0.002


def test_exhaust_level_peaks_between_tuned_speed_and_1_2_times_it():
    from genny import cyber
    eng = dict(cycle=2, cylinders=1, header=0.3, pipe=0.5, unequal=0.0, ports=(82, 113, 37.8, 30, 54.5, 110, 54),
               tuned_rpm=11500, raw=1, seed=1, dur=1.0, mic=0.0, block=0.0, load=0.9)
    rpms = np.arange(0.6, 1.301, 0.05) * 11500.0
    lv = np.array([float(np.sqrt(np.mean(np.asarray(cyber.pipe_engine(**eng, rpm=float(r))) ** 2))) for r in rpms])
    assert 1.0 <= rpms[int(np.argmax(lv))] / 11500.0 <= 1.2 + 1e-9


def test_calliope_jumps_to_the_third_harmonic_when_overblown():
    from genny import instruments_extra as X
    sr = 44100
    f = 261.63
    base = X.calliope(f, 0.8, sr, 1.0, pressure=1.6)
    over = X.calliope(f, 0.8, sr, 1.0, pressure=1.6, overblow=1.5)
    f0, _ = X.calliope_pitch(f, 1.0, 1.6, 0)
    seg = lambda y: y[int(0.35 * sr):int(0.75 * sr)]      # noqa: E731
    assert abs(_peak(seg(base), 0.7 * f0, 4 * f0, sr) / f0 - 1.0) < 0.05
    assert abs(_peak(seg(over), 0.7 * f0, 4 * f0, sr) / (3 * f0) - 1.0) < 0.06
    assert np.array_equal(X.calliope(f, 0.8, sr, 1.0, pressure=1.0), X.calliope(f, 0.8, sr, 1.0, pressure=1.0, overblow=1.5))


def test_scooter_keeps_the_cvt_until_the_belt_slips_and_idles_at_1900():
    from genny import engines, heavy_vehicles as H
    assert engines.TWO_STROKE_KINDS["scooter"]["idle"] == 1900.0 and engines.TWO_STROKE_KINDS["chainsaw"]["idle"] == 2800.0
    a = H.scooter(SR, limiter=1, dur=3.0, seed=1)
    b = H.scooter(SR, limiter=1, dur=3.0, seed=1, slip=2.0)
    assert not np.array_equal(a, b)
    from genny import cyber
    r0, s0 = cyber._limiter_run(3 * SR, 7000.0, 9000.0, 8000.0, 6000.0, 150.0, False)
    y = engines.two_stroke(SR, rpm=np.full(3 * SR, 7070.0), kind="scooter", cut="machine", cut_at=2.0, dur=3.0, seed=1)
    assert np.all(np.isfinite(y)) and len(y) > SR
    idle = engines.two_stroke(SR, kind="scooter", idle=1, dur=2.0, seed=1, load=0.2)
    assert np.all(np.isfinite(idle)) and not np.array_equal(idle, engines.two_stroke(SR, kind="scooter", dur=2.0, seed=1, load=0.2))
    assert np.all(np.isfinite(H.scooter(SR, idle=1, dur=2.0))) and np.all(np.isfinite(H.motorcycle(SR, kind="cross_125", idle=1, dur=2.0)))


def test_engine_bank_autopitch_ratio_and_crossfade():
    from genny import vehicle
    r, st = vehicle.bank_ratio(20.0, 26.0)
    assert abs(float(r) - 0.769) < 0.001 and abs(float(st) + 4.54) < 0.01
    y = vehicle.engine_bank(np.full(2 * SR, 2600.0), (2000.0, 3000.0), SR, loop=2.0, cylinders=4, load=0.5)
    assert abs(_peak(y[SR // 2:], 60, 130) / (2600.0 / 60.0 * 2.0) - 1.0) < 0.04


def test_moving_source_doppler_at_100_m_s_is_1411_hz():
    from genny.acoustics import MovingPropagation
    t = np.arange(SR) / SR
    x = np.sin(2 * np.pi * 1000.0 * t)
    d = 120.0 - 100.0 * t
    y = MovingPropagation().apply_trajectory(x, d, sr=SR, block=0)
    k = np.flatnonzero(y != 0)
    seg = y[k[0] + 2000:k[0] + 2000 + 16000]
    assert abs(_peak(seg, 800, 2000) / 1411.5 - 1.0) < 0.01


def test_values_checked_against_the_books():
    from genny import acoustic, tubes, machines, nature, reverbs
    assert tubes.END_REFLECTION["flanged"] == (0.182, 1.825, 0.649) and tubes.END_REFLECTION["unflanged"] == (0.167, 1.393, 0.457)
    for end, d in (("flanged", 0.8216), ("unflanged", 0.6133)):
        n1, d1, d2 = tubes.END_REFLECTION[end]
        assert abs((d1 - n1) / 2 - d) < 0.005 and abs(0.5 * (d1 * d1 - n1 * n1) - 2 * (0.5 if end == "flanged" else 0.25) - d2) < 0.003
    assert abs(machines.capstan_ratio(0.5, math.pi) - 4.81) < 0.005 and abs(machines.slip_torque(0.5, math.pi) - 0.656) < 0.001
    g = nature.glass_window.__defaults__
    assert g[0] == (2007.0, 1994.0, 1986.0, 1969.0) and g[1] == (254.0, 669.0, 443.0, 551.0) and g[2] == (3.7, 4.2) and g[3] == 0.61 and g[4] == 2.3
    prem = {(m, n): (f, t) for m, n, f, t in acoustic._PREMIER_2500}
    assert [prem[(m, 1)] for m in (1, 2, 3, 4, 5)] == [(147.0, 2.8), (245.0, 4.2), (288.0, 8.4), (355.0, 4.6), (419.0, 5.2)]
    assert [prem[(0, n)][1] for n in (1, 2, 3)] == [0.4, 0.5, 1.0]
    from genny import creatures
    assert abs(creatures.stride_pace("trot", 500.0) - 358.0) < 1.0 and abs(creatures.stride_pace("gallop", 500.0) - 403.0) < 1.5
    assert reverbs.huggins.__defaults__ is not None


def test_footsteps_incline_shifts_the_weight():
    from genny import creatures
    up = creatures.footsteps(SR, ground="concrete", incline=0.15, steps=6, seed=2)
    down = creatures.footsteps(SR, ground="concrete", incline=-0.15, steps=6, seed=2)
    flat = creatures.footsteps(SR, ground="concrete", steps=6, seed=2)
    assert not np.array_equal(up, down) and not np.array_equal(up, flat)
    low = lambda y: float(np.sum(np.abs(np.fft.rfft(y[:SR]))[: SR // 5] ** 2) / np.sum(np.abs(np.fft.rfft(y[:SR])) ** 2))      # noqa: E731
    assert low(down) > low(up)


def test_plate_cascade_with_the_articles_loss_law():
    import math as m
    from scipy.signal import spectrogram
    from genny import plates
    size = m.sqrt(0.4 * 0.6) / 0.886
    y = plates.nl_plate(SR, 1.0, size, 1.2, 300.0, 14.0, (0.31, 0.43), model="vk", eps=1.5, material=(7860.0, 2e11, 0.3), bright=1.0,
                        quality=1.5, dur=1.5, drive=lambda n, fs: plates.pulse(n, fs, 300.0, 14.0), loss=(173.0, 2000.0, 2.86), accel=False, hp=20.0)
    f, t, S = spectrogram(y, SR, nperseg=2048, noverlap=1536)
    d = 10 * np.log10(S + 1e-30)
    front = max(float(f[np.flatnonzero(d[:, j] > d.max() - 80)[-1]]) for j in range(d.shape[1]))
    assert 5500.0 < front < 12000.0, front
