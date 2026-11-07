"""Aircraft and rockets: the sourced numbers measured on the rendered signal, and every variant of lot L09 renders clean."""
import math

import numpy as np
import pytest
from scipy.signal import welch
from scipy.stats import skew

from genny import aero as A
from genny import analysis

SR = 48000


def line_db(y, f, sr=SR, half=None):
    """Height in dB of the strongest FFT line within 1.5 % of f (at least 1 Hz), over the median spectrum within 15 %."""
    s = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    fr = np.fft.rfftfreq(len(y), 1.0 / sr)
    w = half or max(0.015 * f, 1.0)
    near = (fr > f - w) & (fr < f + w)
    around = (fr > 0.85 * f) & (fr < 1.15 * f) & ~near
    return 20.0 * np.log10(s[near].max() / (np.median(s[around]) + 1e-12))


def line_abs(y, f, sr=SR, half=1.0):
    s = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    fr = np.fft.rfftfreq(len(y), 1.0 / sr)
    return 20.0 * np.log10(s[(fr > f - half) & (fr < f + half)].max() + 1e-12)


def psd_peak(y, lo=2.0, hi=4000.0, nper=1 << 14, sr=SR):
    f, p = welch(y, sr, nperseg=min(nper, len(y)))
    p = np.convolve(p, np.ones(5) / 5, mode="same")
    k = (f >= lo) & (f <= hi)
    return f[k][np.argmax(p[k])]


def band_rms(y, lo, hi, sr=SR):
    f, p = welch(y, sr, nperseg=1 << 13)
    return math.sqrt(p[(f >= lo) & (f < hi)].sum())


def db(a, b):
    return 20.0 * math.log10(a / b)


def rms(y):
    return float(np.sqrt(np.mean(y * y)))


# ------------------------------------------------------------------- every line of the lot
CALLS = [("sonic_boom", {}), ("sonic_boom", {"interior": 1}),
         ("prop_engine", {"aircraft": "r985"}), ("prop_engine", {"aircraft": "p51"}), ("prop_engine", {"aircraft": "cessna172"}), ("prop_engine", {"aircraft": "atr72"}),
         ("turbofan", {}), ("jet_cycle", {"dur": 6.0}), ("jet_afterburner", {"crackle_gain": 0.0}), ("flyby_jet", {"dur": 4.0}), ("glider", {"dur": 2.0}),
         ("airframe", {"flaps": 1.0, "gear": 1, "dur": 2.0}), ("airship", {"dur": 2.0}),
         ("space_rocket", {"size": "heavy", "dur": 4.0}), ("space_rocket", {"size": "small", "dur": 3.0}), ("space_rocket", {"phase": "liftoff", "dur": 12.0}),
         ("rotorcraft", {"blades": 2}), ("rotorcraft", {"blades": 4}), ("rotorcraft", {"kind": "tandem"}), ("rotorcraft", {"kind": "coaxial"}),
         ("rotorcraft", {"tail_rpm": 1600, "tail_blades": 5}), ("rotorcraft", {"gearbox": 1}), ("rotorcraft", {"interior": 1}),
         ("rotorcraft", {"state": "descent"}), ("rotorcraft", {"state": "fast", "craft": "uh60"}), ("rotorcraft", {"angle": 120}), ("rotorcraft", {"engine": "piston"}),
         ("drone", {"n": 4}), ("drone", {"n": 6}), ("drone", {"n": 8}), ("drone", {"maneuver": 1}), ("cabin", {"dur": 2.0})]


@pytest.mark.parametrize("name,kw", CALLS, ids=[f"{n}-{i}" for i, (n, _) in enumerate(CALLS)])
def test_every_variant_renders_clean(name, kw):
    y = getattr(A, name)(sr=44100, **kw)
    assert np.isfinite(y).all() and 0.02 < np.abs(y).max() <= 1.0
    assert len(analysis.detect_clicks(y)) == 0
    assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3


def test_unknown_names_are_refused():
    for fn, kw in ((A.prop_engine, {"aircraft": "x"}), (A.rotorcraft, {"craft": "x"}), (A.rotorcraft, {"state": "x"}), (A.rotorcraft, {"kind": "x"}),
                   (A.rotorcraft, {"engine": "x"}), (A.turbofan, {"engine": "x"}), (A.space_rocket, {"size": "x"}), (A.space_rocket, {"phase": "x"}),
                   (A.jet_cycle, {"phase": "x"}), (A.cabin, {"kind": "x"})):
        with pytest.raises(ValueError):
            fn(sr=SR, dur=0.1, **kw)


# ------------------------------------------------------------------- sonic boom
def test_carlson_worked_problem():
    dp, dt = A.carlson_boom(1.99, 14480.0, 32.7)
    assert abs(dp - 74.0) < 1.0 and abs(dt - 0.17) < 0.002


def test_boom_rendered_is_the_n_wave():
    y = A.sonic_boom(sr=SR)
    dp, dt = A.carlson_boom(1.99, 14480.0, 32.7)
    assert abs(y.max() - dp / A.PA_FULL) / (dp / A.PA_FULL) < 0.05
    assert abs((np.argmin(y) - np.argmax(y)) / SR - dt) / dt < 0.05
    rise = np.argmax(y > 0.9 * y.max()) - np.argmax(y > 0.1 * y.max())
    assert rise / SR < 0.004                                            # rise_ms 3 of a Hann: 10-90 % shorter than the window
    assert band_rms(y, 3, 20) > 0.1 * band_rms(y, 20, 200)              # infrasound kept


def test_boom_duration_and_level_follow_the_laws():
    a, b = A.carlson_boom(2.0, 15000.0, 30.0), A.carlson_boom(2.0, 15000.0 * 16, 30.0)
    assert a[0] > b[0]
    assert A.carlson_boom(0.9, 10000.0, 30.0) == (0.0, 0.0)


def test_boom_indoors_is_duller_and_rings():
    out, ins = A.sonic_boom(sr=SR), A.sonic_boom(sr=SR, interior=1)
    assert np.abs(ins).max() < np.abs(out).max()
    assert band_rms(ins, 300, 3000) / band_rms(ins, 3, 100) > band_rms(out, 300, 3000) / band_rms(out, 3, 100)   # panes
    assert band_rms(ins, 3, 100) < band_rms(out, 3, 100)


# ------------------------------------------------------------------- propellers
@pytest.mark.parametrize("aircraft,bpf,firing", [("r985", 76.67, 172.5), ("p51", 85.33, 300.0), ("cessna172", 90.0, 90.0)])
def test_blade_rate_and_firing_lines(aircraft, bpf, firing):
    pr, z, _, _, er, c, _, _ = A.PROP_AIRCRAFT[aircraft]
    assert abs(pr * z / 60.0 - bpf) < 0.01 and abs(er * c / 120.0 - firing) < 0.01
    assert line_db(A.prop_engine(sr=SR, aircraft=aircraft, engine=0.0), bpf) > 10.0
    assert line_db(A.prop_engine(sr=SR, aircraft=aircraft, engine=1.0), firing) > 10.0


def test_p51_reduction_is_0427():
    assert abs(A.PROP_AIRCRAFT["p51"][0] / A.PROP_AIRCRAFT["p51"][4] - 0.427) < 0.001


def test_engine_rpm_moves_the_propeller_through_the_reduction():
    y = A.prop_engine(sr=SR, aircraft="p51", rpm=1500.0, engine=0.0)
    assert line_db(y, 85.33 / 2.0) > 10.0


def test_turboprop_blade_rate_and_quiet_turbine():
    y = A.prop_engine(sr=SR, aircraft="atr72", engine=0.0)
    assert line_db(y, 120.0) > 10.0
    t = A.prop_engine(sr=SR, aircraft="atr72", engine=1.0)
    assert line_db(t, 5588.0 * A.TURBOPROP_SPEED, half=3.0) > 20.0


def test_airship_blade_and_firing_at_90_hz_and_engines_apart():
    y = A.airship(sr=SR, dur=6.0, seed=1)
    assert line_db(y, 90.0) > 10.0
    s = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1.0 / SR)
    k = (f > 88.0) & (f < 92.0)
    top = f[k][s[k] > 0.3 * s[k].max()]
    assert top.max() - top.min() > 0.5           # 3 engines 0.5 % apart: 89.55, 90, 90.45 Hz


# ------------------------------------------------------------------- turbofan
def test_cfm56_numbers():
    assert abs(A.tip_mach(5175.0, 1.55) - 1.22) < 0.01 and abs(A.tip_mach(5000.0, 1.84) - 1.40) < 0.01
    assert line_db(A.turbofan(sr=SR), 2070.0) > 20.0
    assert line_db(A.turbofan(sr=SR, engine="cfm56_5b"), 3000.0) > 20.0


def test_buzz_saw_only_above_mach_1():
    assert line_db(A.turbofan(sr=SR, seed=2), 86.25 * 5, half=1.0) > 15.0
    assert line_db(A.turbofan(sr=SR, rpm=3000.0, seed=2), 50.0 * 5, half=1.0) < 12.0     # Mach 0.71


# ------------------------------------------------------------------- start and afterburner
def test_jet_cycle_slides_the_turbine_linearly():
    y = A.jet_cycle(sr=SR, phase="start", speed=0.2, dur=8.0, ignite=0)
    tail = y[-SR:]
    # the end of the ramp sits near 0.2 x 739 Hz: the strongest partial, shifted, and not at full speed
    assert 125.0 < psd_peak(tail, 90.0, 400.0, 1 << 13) < 160.0
    head = y[int(1.0 * SR):int(2.0 * SR)]
    assert psd_peak(head, 40.0, 400.0, 1 << 13) < 110.0


def test_jet_exhaust_brightens_with_speed():
    lo, hi = A.jet_cycle(sr=SR, phase="start", speed=0.25, dur=4.0, ignite=0), A.jet_cycle(sr=SR, phase="start", speed=1.0, dur=4.0, ignite=0)
    centroid = lambda y: (lambda f, p: (f * p).sum() / p.sum())(*welch(y[-SR:], SR, nperseg=4096))
    assert centroid(hi) > 2.0 * centroid(lo)


def test_igniter_ticks_only_with_ignite():
    on, off = A.jet_cycle(sr=SR, phase="start", dur=5.0, seed=3), A.jet_cycle(sr=SR, phase="start", dur=5.0, ignite=0, seed=3)
    w = lambda y: rms(y[SR // 2:SR // 2 + SR // 30])
    assert w(on) > 1.25 * w(off)


def test_jet_power_laws():
    assert abs(float(A.jet_power(200.0) / A.jet_power(100.0)) - 2 ** 8) < 1e-6
    assert abs(float(A.jet_power(2000.0) / A.jet_power(1000.0)) - 2 ** 3) < 1e-6
    assert abs(float(A.jet_power(A.C)) - 1.0) < 1e-9


@pytest.mark.parametrize("u1,u2,expect", [(200.0, 300.0, 80 * math.log10(1.5)), (450.0, 900.0, 30 * math.log10(2.0))])
def test_jet_level_on_the_signal(u1, u2, expect):
    a = A.jet_afterburner(sr=SR, velocity=u1, ignite=-1.0, crackle_gain=0.0, dur=2.0)
    b = A.jet_afterburner(sr=SR, velocity=u2, ignite=-1.0, crackle_gain=0.0, dur=2.0)
    assert abs(db(rms(b), rms(a)) - expect) < 2.0


def test_jet_peak_at_the_measured_strouhal():
    y = A.jet_afterburner(sr=SR, ignite=-1.0, crackle_gain=0.0, angle=0.0, dur=4.0)
    assert 160.0 < psd_peak(y, 50.0, 2000.0) < 215.0        # St 0.17 x 1100 / 1 = 187 Hz


def test_tam_similarity_spectra_are_continuous_and_peak_at_one():
    big, fine = A.tam_spectra(np.array([1.0, 0.5, 2.5, 0.05, 0.15, 10.0, 30.0]))
    assert abs(big[0]) < 1e-6 and abs(fine[0]) < 1e-6
    for x in (0.5, 2.5):
        a, b = A.tam_spectra(np.array([x * (1 - 1e-9), x * (1 + 1e-9)]))[0]
        assert abs(a - b) < 0.01
    for x in (0.05, 0.15, 10.0, 30.0):
        a, b = A.tam_spectra(np.array([x * (1 - 1e-9), x * (1 + 1e-9)]))[1]
        assert abs(a - b) < 0.01
    assert abs(big[1] + 3.0) < 0.01 and abs(fine[3] + 9.5) < 0.01 and abs(fine[6] + 26.6) < 0.01


def test_jet_large_scale_falls_at_the_tam_slope():
    y = A._shaped(np.random.default_rng(1).standard_normal(4 * SR), 200.0, 0)
    f, p = welch(y, SR, nperseg=1 << 14)
    lvl = lambda fc: 10 * np.log10(np.mean(p[(f > fc / 1.1) & (f < fc * 1.1)]))
    assert abs((lvl(800.0) - lvl(1600.0)) - 27.7472 * math.log10(2.0)) < 1.0


def test_jet_peak_strouhal_by_angle():
    assert abs(A.jet_peak_strouhal(30.0)[0] - 0.17) < 1e-9
    assert abs(A.jet_peak_strouhal(90.0)[1] - (0.18 + 40.0 / 70.0 * 0.10)) < 1e-9
    assert abs(A.jet_peak_strouhal(60.0)[0] - 0.29) < 1e-9


def test_rocket_spectrum_shape():
    assert abs(float(A.rocket_spectrum_db(0.02)) - 11.2) < 1e-9
    assert float(A.rocket_spectrum_db(0.02)) == max(p[1] for p in A.ROCKET_SPECTRUM)
    y = A.rocket_spectrum_noise(8 * SR, 40.0, np.random.default_rng(2))
    f, p = welch(y, SR, nperseg=1 << 15)
    lvl = lambda fc: 10 * np.log10(np.mean(p[(f > fc / 1.15) & (f < fc * 1.15)]))
    assert abs((lvl(40.0) - lvl(1000.0)) - (11.2 - float(A.rocket_spectrum_db(0.5)))) < 1.5     # St 0.02 -> 0.5: 18.4 dB


def test_large_scale_downstream_fine_scale_to_the_side():
    ratio = lambda y: band_rms(y, 50, 400) / band_rms(y, 800, 6000)
    down = A.jet_afterburner(sr=SR, ignite=-1.0, crackle_gain=0.0, angle=20.0, dur=2.0)
    side = A.jet_afterburner(sr=SR, ignite=-1.0, crackle_gain=0.0, angle=90.0, dur=2.0)
    assert ratio(down) > ratio(side)


def test_crackle_meets_the_skewness_criteria():
    c = A.crackle(2 * SR, np.random.default_rng(0))
    assert skew(c) > 0.4 and skew(np.diff(c)) > 5.0
    g = np.random.default_rng(0).standard_normal(2 * SR)
    assert abs(skew(g)) < 0.1


def test_crackle_is_audible_in_the_jet():
    a = A.jet_afterburner(sr=SR, ignite=-1.0, crackle_gain=1.0, dur=2.0)
    b = A.jet_afterburner(sr=SR, ignite=-1.0, crackle_gain=0.0, dur=2.0)
    assert skew(np.diff(a)) > skew(np.diff(b)) + 0.05


def test_reheat_lights_at_ignite():
    y = A.jet_afterburner(sr=SR, dur=4.0, ignite=0.5)
    before, after = rms(y[SR // 2:int(1.8 * SR)]), rms(y[int(2.6 * SR):int(3.8 * SR)])
    assert after > 2.0 * before


def test_flyby_pitch_falls_by_the_doppler_ratio():
    y = A.flyby_jet(sr=SR, rpm=1500.0, dur=6.0, closest=80.0)
    n = len(y)
    early = psd_peak(y[:n // 5], 1500.0, 3200.0, 1 << 13)
    late = psd_peak(y[-n // 4:], 250.0, 600.0, 1 << 13)
    assert 1700.0 < early < 2300.0 and 300.0 < late < 400.0  # 600 Hz x 343/(343 -+ 250): 2213 and 347 at the far limit


# ------------------------------------------------------------------- airframe
def test_airframe_peak_518_hz():
    assert abs(A.airframe_peak(30.0, 0.005) - 518.0) < 3.0
    assert 400.0 < psd_peak(A.airframe(sr=SR, speed=30.0, dur=4.0), 100.0, 3000.0) < 650.0
    assert 380.0 < psd_peak(A.glider(sr=SR, dur=4.0), 200.0, 3000.0) < 680.0


def test_airframe_u5():
    a, b = A.airframe(sr=SR, speed=40.0, dur=2.0), A.airframe(sr=SR, speed=80.0, dur=2.0)
    assert abs(db(rms(b), rms(a)) - 50.0 * math.log10(2.0)) < 0.5


def test_flaps_and_gear_add_noise():
    base = rms(A.airframe(sr=SR, dur=1.0))
    assert rms(A.airframe(sr=SR, dur=1.0, flaps=1.0)) > 1.2 * base
    assert rms(A.airframe(sr=SR, dur=1.0, gear=1)) > 1.5 * base


# ------------------------------------------------------------------- rockets
def test_sls_power_level():
    assert abs(A.rocket_pwl(39.1e6, 2600.0, 0.0033) - 202.4) < 1.0     # thrust and Ue UNSOURCED, result re Kellison & Gee
    assert abs(A.rocket_pwl(1e6, 2000.0, 0.01) - A.rocket_pwl(1e6, 2000.0, 0.005) - 3.01) < 0.01


def test_rocket_peaks():
    assert abs(A.rocket_peak(0.3, 2000.0) - 133.3) < 0.1
    assert 10.0 <= psd_peak(A.space_rocket(sr=SR, size="heavy", distance=200.0, dur=16.0), 3.0, 300.0, 1 << 13) <= 25.0   # flat top: St 0.01-0.05 within 2 dB
    assert 90.0 <= psd_peak(A.space_rocket(sr=SR, size="small", distance=200.0, dur=4.0), 30.0, 800.0) <= 180.0
    assert psd_peak(A.space_rocket(sr=SR, diameter=0.6, size="small", distance=200.0, dur=4.0), 20.0, 800.0) < 90.0   # f ~ 1 / De


def test_rocket_loses_its_top_with_distance():
    ratio = lambda y: band_rms(y, 2000, 8000) / band_rms(y, 3, 300)
    assert ratio(A.space_rocket(sr=SR, distance=5000.0, dur=3.0)) < 0.5 * ratio(A.space_rocket(sr=SR, distance=100.0, dur=3.0))


def test_rocket_is_loudest_at_50_degrees():
    at = lambda a: rms(A.space_rocket(sr=SR, angle=a, dur=2.0))
    assert at(50.0) > 1.5 * at(120.0) and at(50.0) > 1.1 * at(0.0)


def test_liftoff_envelope():
    y = A.space_rocket(sr=SR, phase="liftoff", dur=26.0)
    seg = lambda a, b: rms(y[int(a * SR):int(b * SR)])
    assert seg(0.2, 1.0) < 0.4 * seg(9.0, 17.0)     # ramp
    assert seg(9.0, 17.0) > seg(4.0, 6.0)           # maximum T+8..18 s
    assert seg(22.0, 25.0) < 0.8 * seg(9.0, 17.0)   # going away


# ------------------------------------------------------------------- rotors
def test_rotor_pulse_apex_follows_symmetry():
    for s in (0.1, 0.5, 0.9):
        p = A.rotor_pulse(10.0, SR // 10, s, duty=1.0)
        assert abs(np.argmax(p) / len(p) - s) < 0.02
        assert p.min() >= -1e-9 and abs(p.max() - 1.0) < 1e-3


def test_rotor_cadences():
    for kw, bpf in (({"blades": 2}, 10.0), ({"blades": 4}, 17.2), ({"craft": "ch47"}, 12.15), ({"craft": "r44"}, 13.33), ({"craft": "oh58"}, 11.58)):
        y = A.rotorcraft(sr=SR, angle=0.0, dur=6.0, **kw)
        assert abs(psd_peak(np.abs(y), bpf * 0.6, bpf * 1.5, 1 << 18) - bpf) < 0.5, kw


def test_uh60_tip_mach():
    assert abs(A.tip_mach_rotor(258.0, 8.18) - 0.64) < 0.01


def test_tail_law_and_line():
    assert A.tail_rpm_law(320.0, 0.5) == 320.0 * 4.5 and A.tail_rpm_law(600.0, 1.0) == 2000.0
    y = A.rotorcraft(sr=SR, angle=180.0, tail_rpm=1600.0, tail_blades=4, dur=4.0)
    assert line_db(y, 1600.0 * 4 / 60.0) > 8.0


def test_no_tail_on_tandem_and_coaxial():
    tail = 1600.0 * 4 / 60.0
    single = line_db(A.rotorcraft(sr=SR, angle=180.0, tail_rpm=1600.0, dur=4.0, rpm=243.0, blades=3), tail)
    assert single > line_db(A.rotorcraft(sr=SR, angle=180.0, tail_rpm=1600.0, kind="coaxial", dur=4.0, rpm=243.0, blades=3), tail) + 3.0


def test_tandem_second_rotor_52_ms_later():
    assert abs(A.TANDEM_DELAY - 0.052) < 0.001
    def ac(**kw):
        y = A.rotorcraft(sr=SR, craft="ch47", angle=30.0, state="descent", tail_blades=2, dur=4.0, symmetry=0.5, **kw)
        e = np.abs(y - np.convolve(y, np.ones(96) / 96, mode="same"))
        e = np.convolve(e, np.ones(48) / 48, mode="same")
        e = e - e.mean()
        c = np.fft.irfft(np.abs(np.fft.rfft(e, 2 * len(e))) ** 2)
        lag = int(((A.TANDEM_DELAY + 0.5 / 12.15) % (1 / 12.15)) * SR)     # second rotor: half a blade off, then the 52 ms
        return c[lag - 48:lag + 48].max() / c[0]
    assert ac() > ac(kind="single") + 0.02


def test_coaxial_doubles_the_blade_rate():
    harm = lambda y, f: line_abs(np.abs(y), f)
    single = A.rotorcraft(sr=SR, rpm=300.0, blades=2, angle=0.0, dur=6.0)
    coax = A.rotorcraft(sr=SR, rpm=300.0, blades=2, kind="coaxial", angle=0.0, dur=6.0)
    assert harm(coax, 20.0) - harm(coax, 10.0) > harm(single, 20.0) - harm(single, 10.0) + 3.0


def test_gearbox_lines_and_rotor_fm():
    e = 20900.0 / 60.0
    y = A.rotorcraft(sr=SR, craft="uh60", gearbox=1, angle=90.0, dur=4.0)
    for k, _ in A.GEAR_RATIOS:
        assert line_db(y, k * e) > 12.0, k
    assert line_db(y, 1.45 * e + 17.2, half=2.0) > 6.0     # a side band at the blade rate


def test_cabin_levels_from_tm83726():
    a = A.rotorcraft(sr=SR, craft="uh60", interior=1, dur=1.0)
    b = A.rotorcraft(sr=SR, craft="ch47", interior=1, dur=1.0)
    assert abs(db(rms(b), rms(a)) - 3.0) < 1.0


def test_interior_is_filtered_by_the_sheet():
    out = A.rotorcraft(sr=SR, dur=2.0, seed=1)
    ins = A.rotorcraft(sr=SR, interior=1, dur=2.0, seed=1)
    top = lambda y: band_rms(y, 4000, 16000) / rms(y)
    assert top(ins) < 0.7 * top(out) or top(ins) > 1.4 * top(out)   # the plate reshapes the spectrum


def test_descent_slaps_in_the_mid_band():
    mid = lambda y: band_rms(y, 170, 700) / band_rms(y, 5, 170)
    assert mid(A.rotorcraft(sr=SR, craft="uh60", state="descent", dur=3.0)) > 1.3 * mid(A.rotorcraft(sr=SR, craft="uh60", state="hover", dur=3.0))


def test_fast_steepens_only_past_mach_085():
    ds = lambda y: abs(skew(np.diff(y)))
    assert A.tip_mach_rotor(258.0, 8.18, 80.0) > 0.85
    assert ds(A.rotorcraft(sr=SR, craft="uh60", state="fast", dur=3.0)) > ds(A.rotorcraft(sr=SR, craft="uh60", state="cruise", dur=3.0))
    assert A.tip_mach_rotor(150.0, 5.0, 80.0) < 0.85


def test_angle_slap_peaks_30_below_the_plane():
    mid = lambda a: (lambda y: band_rms(y, 170, 1000) / band_rms(y, 5, 170))(A.rotorcraft(sr=SR, craft="uh60", state="descent", angle=a, dur=2.0))
    assert mid(30.0) > mid(0.0)


def test_chopping_gone_under_the_rotor():
    mod = lambda y: (lambda e: np.std(e) / np.mean(e))(np.abs(y)[::10])
    assert mod(A.rotorcraft(sr=SR, angle=90.0, dur=3.0)) < mod(A.rotorcraft(sr=SR, angle=0.0, dur=3.0))


def test_tail_louder_behind():
    tail = 1600.0 * 4 / 60.0
    assert line_db(A.rotorcraft(sr=SR, angle=180.0, tail_rpm=1600.0, dur=4.0), tail) > line_db(A.rotorcraft(sr=SR, angle=0.0, tail_rpm=1600.0, dur=4.0), tail)


def test_far_leaves_the_throb():
    ratio = lambda y: band_rms(y, 1000, 8000) / band_rms(y, 5, 200)
    assert ratio(A.rotorcraft(sr=SR, far=1.0, dur=2.0)) < 0.2 * ratio(A.rotorcraft(sr=SR, dur=2.0))


def test_piston_engine_fires_at_its_rate():
    y = A._piston(90.0, SR * 2, np.random.default_rng(0))
    f, p = welch(np.abs(y), SR, nperseg=1 << 15)
    k = (f > 60) & (f < 130)
    assert abs(f[k][np.argmax(p[k])] - 90.0) < 3.0


# ------------------------------------------------------------------- drones
def test_phantom_ii_levels():
    y = A.drone(sr=SR, n=1, detune=0.0, dur=4.0, seed=1)
    a = lambda f: line_abs(y, f, half=1.5)
    assert abs((a(128.0) - a(64.0)) - (45.0 - 19.0)) < 2.0
    assert abs((a(128.0) - a(832.0)) - (45.0 - 34.0)) < 2.0
    assert abs(a(256.0) - a(128.0)) < 1.5 and abs(a(512.0) - a(128.0)) < 1.5
    assert 7.0 < a(128.0) - a(640.0) < 13.0                  # above harmonic 4: 32-38 dB against 45


def test_hump_10_db_600_to_6000():
    y = A.drone(sr=SR, n=1, detune=0.0, dur=4.0)
    f, p = welch(y, SR, nperseg=1 << 12)
    d = 10.0 * np.log10(np.median(p[(f > 3000) & (f < 5000)]) / np.median(p[(f > 12000) & (f < 18000)]))
    assert abs(d - 10.0) < 2.0


def test_rotors_are_apart_and_cadence_follows_blades():
    y = A.drone(sr=SR, n=4, dur=5.0, seed=4)
    s = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1.0 / SR)
    k = (f > 124.0) & (f < 132.0)
    top = f[k][s[k] > 0.2 * s[k].max()]
    assert top.max() - top.min() > 0.5
    assert line_db(A.drone(sr=SR, n=6, rpm=4500.0, dur=3.0), 150.0, half=3.0) > 15.0
    assert line_db(A.drone(sr=SR, n=8, rpm=3000.0, blades=3, dur=3.0), 150.0, half=3.0) > 15.0


def test_maneuver_swings_30_percent_in_50_ms():
    steps = A._maneuver_rpm(3840.0, SR * 3, np.random.default_rng(5), 1)
    assert steps.min() >= 3840.0 * 0.69 and steps.max() <= 3840.0 * 1.31 and steps.max() - steps.min() > 500.0
    assert np.abs(np.diff(steps)).max() * 0.05 * SR <= 2.5 * (steps.max() - steps.min())
    assert np.ptp(A._maneuver_rpm(3840.0, SR, np.random.default_rng(5), 0)) == 0.0


# ------------------------------------------------------------------- cabin
def test_cabin_wall_takes_30_db_above_500_hz():
    t = np.arange(SR) / SR
    for f, loss in ((3000.0, 30.0), (60.0, 0.0)):
        x = np.sin(2 * np.pi * f * t)
        assert abs(db(rms(A.cabin_wall(x)[SR // 4:]), rms(x[SR // 4:])) + loss) < 2.5
