import numpy as np
import pytest

SR = 44100


def test_sampler_hold_extends_at_stable_rms():
    from genny.klang import _extend
    t = np.arange(int(0.6 * SR)) / SR
    env = np.minimum(1.0, t / 0.02) * np.where(t < 0.5, 1.0, np.exp(-(t - 0.5) / 0.02))
    y = 0.5 * env * np.sin(2 * np.pi * 220.0 * t)
    need = int(2.0 * SR)
    z = _extend(y, SR, need)
    assert len(z) == need
    seg = z[int(1.0 * SR):int(1.9 * SR)]
    rms = np.sqrt(np.mean(seg ** 2))
    assert rms == pytest.approx(0.5 / np.sqrt(2), rel=0.1)
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    f = np.fft.rfftfreq(len(seg), 1 / SR)
    assert f[np.argmax(spec)] == pytest.approx(220.0, abs=3.0)
    assert np.max(np.abs(np.diff(z))) < 0.05
    assert _extend(y, SR, len(y) - 10) is y


def test_collision_step_conserves_energy_equal_masses():
    from genny.plates import collision_step
    k, kap, al = 1e-6, 1e8, 1.3
    x1, x1p = 0.0, 0.0
    x2, x2p = -1e-4 + k, -1e-4
    g = 2.0 * k * k
    for _ in range(4000):
        t1, t2 = 2 * x1 - x1p, 2 * x2 - x2p
        a0 = x2p - x1p
        r, F = collision_step(a0, a0 - (t2 - t1), g, kap, al)
        assert abs(r + g * F + a0 - (t2 - t1)) < 1e-12
        assert F >= 0.0
        x1p, x1 = x1, t1 + k * k * F
        x2p, x2 = x2, t2 - k * k * F
    assert (x1 - x1p) / k == pytest.approx(1.0, abs=1e-6)
    assert (x2 - x2p) / k == pytest.approx(0.0, abs=1e-6)


def test_snare_wires_far_away_leave_the_head_linear_and_buzz_when_touching():
    from genny import plates as P
    mdl = P._model(8000, 1.0, "ssss", 0.0, 120.0, 1.0, 0.0, 0.3, None, True, 0.0)
    f = P.pulse(1600, 8000, 1.0, 2.0) * (200.0 / (P.HEAD_DENSITY * 0.33 ** 3))
    exc, pts = (0.6, 0.55, 0.1), ((0.71, 0.33),)
    lin = P.simulate(mdl, f, exc, pts)
    far, w_far = P.simulate_snare(mdl, f, exc, pts, 0.33, gap=1.0)
    assert np.max(np.abs(far - lin)) < 1e-9 * np.max(np.abs(lin))
    touch, w = P.simulate_snare(mdl, f, exc, pts, 0.33, gap=0.0)
    assert np.max(np.abs(w)) > 0.0
    assert np.max(np.abs(touch - lin)) > 0.01 * np.max(np.abs(lin))

    def hf(y):
        s = np.abs(np.fft.rfft(np.diff(y[:, 0])))
        fr = np.fft.rfftfreq(len(y) - 1, 1 / 8000)
        return np.sum(s[fr > 1500] ** 2) / np.sum(s ** 2)
    assert hf(touch) > hf(lin)


def test_sfx_helicopter_exposes_the_new_fields_with_unchanged_defaults():
    from genny.physics import Helicopter
    from genny.sfx import helicopter
    a = helicopter(dur=0.5, seed=1)
    b = Helicopter(320.0, 4, 1600, 4, .55).render(0.5, sr=SR, seed=1)
    assert np.array_equal(a, b)
    c = helicopter(dur=0.5, seed=1, partials=1.0)

    def band(y, f):
        s = np.abs(np.fft.rfft(y))
        fr = np.fft.rfftfreq(len(y), 1 / SR)
        return np.sum(s[(fr > f - 15) & (fr < f + 15)] ** 2) / np.sum(s ** 2)
    shaft = (4500 + 2500 * .55) / 60.0
    assert band(c, 9.41 * shaft) > 3 * band(a, 9.41 * shaft)
    d = helicopter(dur=0.5, seed=1, rotors=3, spread=.05, fuselage=.5, box=.5)
    assert not np.allclose(a, d)


def test_tube_steepening_advances_the_compression_peak():
    from genny.tubes import steepen_through
    fs = 44100
    t = np.arange(4096) / fs
    y = np.exp(-0.5 * ((t - 0.04) / 0.002) ** 2)
    z = steepen_through(y, fs, 0.01, 20000.0)
    nl = 0.5 * 2.36 / (2 * 1.36 * 101325.0)
    shift = 0.01 * nl * 20000.0 * fs
    assert np.argmax(y) - np.argmax(z) == pytest.approx(shift, abs=2)
    assert np.max(np.diff(z)) > 1.2 * np.max(np.diff(y))
    assert np.array_equal(steepen_through(np.zeros(10), fs, 0.01, 1e4), np.zeros(10))


def test_tube_silencer_is_the_polack_diffusion():
    from genny.tubes import silence_through, tube
    fs = 44100
    x = np.zeros(fs)
    x[0] = 1.0
    h = silence_through(x, fs, 0.004)
    assert np.argmax(h) == pytest.approx(0.004 * fs, abs=2)
    assert np.sum(h) == pytest.approx(1.0, rel=0.01)
    n = np.random.default_rng(0).standard_normal(4410)
    assert np.array_equal(tube(n, fs), tube(n, fs, steepen=0.0, silencer=0.0))
    assert not np.allclose(tube(n, fs), tube(n, fs, silencer=5.0))


def _hammer(note, speed, mass_factor=1.0, agraffe=False):
    import math
    from genny.piano import PianoHammerAction
    h = PianoHammerAction(note, sr=SR)
    h = PianoHammerAction(note, sr=SR, hammer_mass_kg=mass_factor * h.hammer_mass_kg, agraffe=agraffe)
    h.strike(math.log(speed / 0.5) / math.log(8.0))
    return h.last_contact_time_s


def test_agraffe_reflection_brings_the_bass_double_mass_to_sixty_percent():
    from genny.piano import PianoHammerAction
    assert PianoHammerAction(60).agraffe_delay_s() == pytest.approx((0.145 - 0.06 * 39 / 87) / 261.6256, rel=1e-4)
    r_old = _hammer(36, 2.5, 2.0) / _hammer(36, 2.5)
    r_new = _hammer(36, 2.5, 2.0, True) / _hammer(36, 2.5, 1.0, True)
    assert r_old > 2.0
    assert 1.45 < r_new < 1.75
    slope_old = _hammer(60, 0.74) / _hammer(60, 1.31)
    slope_new = _hammer(60, 0.74, 1.0, True) / _hammer(60, 1.31, 1.0, True)
    assert slope_new > 1.08 > slope_old


def test_agung_partial_626_fades_and_returns_at_3_s():
    from genny.percussion import gong
    y = gong(SR, vel=1.0, tune=90, decay=8.0, glide=0.0, preset="agung")
    t = np.arange(len(y)) / SR

    def amp(f, t0):
        s = slice(int((t0 - 0.1) * SR), int((t0 + 0.1) * SR))
        return abs(np.sum(y[s] * np.exp(-2j * np.pi * f * t[s])))
    a = {t0: amp(626.0, t0) for t0 in (0.5, 3.0, 5.0)}
    assert a[3.0] < 0.2 * a[0.5]
    assert a[5.0] > 3 * a[3.0]
    env = np.abs(y[:SR])
    hull = np.array([env[i:i + 441].max() for i in range(0, SR - 441, 441)])
    spec = np.abs(np.fft.rfft(hull - hull.mean()))
    assert np.fft.rfftfreq(len(hull), 0.01)[np.argmax(spec)] == pytest.approx(4.0, abs=0.6)


def test_weighted_rw_and_wall_presets():
    from genny.dsp import RW_REF, wall_rw, weighted_rw
    ref = np.asarray(RW_REF, float)
    assert weighted_rw(ref + 10) == 64                    # shifted 12 dB: 16 bands x 2 dB = 32 dB of deviations
    dip = ref + 10
    dip[4] -= 20
    assert weighted_rw(dip) == 62
    dip[5] -= 20
    assert weighted_rw(dip) == 58
    assert weighted_rw(ref - 0.5) == 53
    assert 39 <= wall_rw("drywall") <= 48
    assert 50 <= wall_rw("concrete") <= 58
    assert 26 <= wall_rw("window") <= 32


def test_tsa_drs_sanc_separate_a_line_from_noise():
    from genny.identify import drs, sanc, tsa
    rng = np.random.default_rng(3)
    fs = 8000
    t = np.arange(8 * fs) / fs
    line = np.sin(2 * np.pi * 80.0 * t)
    noise = rng.standard_normal(len(t))
    x = line + noise
    avg = tsa(x, 100.0, 400)
    assert np.sqrt(np.mean((avg - line[:100]) ** 2)) < 0.15
    det, rnd = drs(x, 2000, 1024)
    s = slice(4000, len(t) - 4000)
    assert np.corrcoef(det[s], line[s])[0, 1] > 0.9
    assert np.std(rnd[s]) == pytest.approx(1.0, rel=0.15)
    pred, err = sanc(x[:3 * fs], 50, 200, 0.05)
    s = slice(2 * fs, 3 * fs)
    assert np.corrcoef(pred[s], line[s])[0, 1] > 0.8


def test_med_and_ar_residual_raise_kurtosis_of_smeared_impulses():
    from scipy.signal import lfilter
    from genny.identify import ar_residual, kurtosis4, med
    rng = np.random.default_rng(4)
    imp = np.zeros(20000)
    imp[::997] = 1.0
    r, w = 0.995, 2 * np.pi * 0.11
    x = lfilter([1.0], [1, -2 * r * np.cos(w), r * r], imp) + 0.05 * rng.standard_normal(20000)
    x += 0.5 * np.sin(2 * np.pi * 0.013 * np.arange(20000))
    k0 = kurtosis4(x)
    y, f = med(x, 40, 30)
    assert kurtosis4(y) > 2 * k0
    assert kurtosis4(ar_residual(x, 30)) > 2 * k0
    assert kurtosis4(rng.standard_normal(200000)) == pytest.approx(3.0, abs=0.1)


def test_filtered_kurtosis_of_an_inner_race_fault_exceeds_healthy():
    from scipy.signal import butter, sosfiltfilt
    from genny.cyber import bearing
    from genny.identify import kurtosis4
    sos = butter(4, 1500.0, "highpass", fs=SR, output="sos")
    k = {s: kurtosis4(sosfiltfilt(sos, np.asarray(bearing(SR, dur=1.0, state=s, severity=1.0), float)))
         for s in ("healthy", "inner")}
    assert k["inner"] > k["healthy"] + 1.0


def test_material_from_modal_damping_distribution():
    from genny.identify import material_from_modes
    f = np.array([500.0, 1200.0, 2600.0, 4100.0])
    assert material_from_modes(f, f / 477.0)[0][0] == "metal"
    assert material_from_modes(f, f / 3.2)[0][0] == "rubber"
    assert material_from_modes(f, f / 150.0)[0][0] == "glass"


def test_register_break_hysteresis_tenor():
    from genny.choir import VOICES, break_states
    brk = VOICES["tenor"]["brk"]
    up = [62, 64, 66, 69, 71]
    down = [71, 69, 67, 65, 60]
    assert break_states(up, brk) == [0, 0, 0, 1, 1]
    assert break_states(down, brk) == [1, 1, 1, 1, 0]
    assert break_states(up, None) == [0] * 5
    assert brk - 3 == 69


def _peak_hz(y, fs):
    s = np.asarray(y, float) - np.mean(y)
    S = np.abs(np.fft.rfft(s * np.hanning(len(s)), 8 * len(s)))
    return np.fft.rfftfreq(8 * len(s), 1 / fs)[np.argmax(S)]


def test_gardner_labia_oscillate_above_threshold_at_the_labial_frequency():
    from genny import creatures as C
    n = int(0.3 * C.KSR)
    quiet = C.gardner_source(np.full(n, 0.8), np.full(n, 3000.0), 3500.0)
    loud = C.gardner_source(np.full(n, 2.0), np.full(n, 3000.0), 3500.0)
    s = slice(int(0.2 * C.KSR), n)
    assert np.std(loud[s]) > 10 * np.std(quiet[s])
    assert _peak_hz(loud[s], C.KSR) == pytest.approx(3000.0, rel=0.08)


def test_suboscine_pitch_rises_with_pressure_alone():
    from genny import creatures as C
    n = int(0.3 * C.KSR)
    s = slice(int(0.2 * C.KSR), n)
    for k2, ke in C.SUBOSCINE.values():
        f = [_peak_hz(C.gardner_source(np.full(n, p), np.full(n, 2000.0), 3500.0, k2=k2, kexp=ke)[s], C.KSR)
             for p in (1.2, 2.0, 3.0)]
        assert f[0] < f[1] < f[2]
        assert f[2] > 1.1 * f[0]
    y = C.birdsong(SR, species="sparrow", source="suboscine")
    assert np.isfinite(y).all() and np.max(np.abs(y)) > 0.1


def test_spectral_content_index_of_a_pure_tone_and_a_rich_wave():
    from genny.creatures import spectral_content_index
    t = np.arange(4800) / 48000
    assert spectral_content_index(np.sin(2 * np.pi * 1000 * t), 48000) == pytest.approx(1.0, abs=0.05)
    saw = sum(np.sin(2 * np.pi * 1000 * k * t) / k for k in range(1, 8))
    assert spectral_content_index(saw, 48000) > 1.5


def test_wilson_cowan_gestures_follow_the_forcing_rate():
    from genny.creatures import WC, wc_pressure
    for rate in (5.0, 15.0):
        g = wc_pressure(2.0, rate)[int(WC["fs"]):]
        on = np.sum(np.diff((g > 0.5).astype(int)) == 1)
        assert on == pytest.approx(rate, abs=1)
    slow, fast = wc_pressure(2.0, 5.0)[2000:], wc_pressure(2.0, 25.0)[2000:]
    assert slow.min() < 0.1 < 0.25 < fast.min()


def test_sparrow_preset_within_chingolo_ranges():
    from genny.creatures import birdsong_plan
    for seed in range(4):
        for s in birdsong_plan("sparrow", seed):
            assert 0.010 <= s["dur"] <= 0.300
            assert all(2000.0 <= hz <= 7000.0 for _, hz in s["f"])


def test_stridulation_grasshopper_broadband_lobster_modal():
    from genny.creatures import stridulation

    def centroid(y):
        S = np.abs(np.fft.rfft(y)) ** 2
        return np.sum(np.fft.rfftfreq(len(y), 1 / SR) * S) / np.sum(S)
    g = stridulation(SR, kind="grasshopper", dur=1.0)
    lob = stridulation(SR, kind="lobster", dur=1.0)
    assert centroid(g) > 5000.0
    assert 300.0 < centroid(lob) < 3000.0


def test_body_fall_schedule_and_debris():
    from genny.creatures import body_fall, fall_schedule
    s = fall_schedule(1.0, seed=0)
    assert len(s) == 3
    p0, p1 = s[1][0] - s[0][0], s[2][0] - s[1][0]
    assert 0.27 <= p0 <= 0.33
    assert p1 / p0 == pytest.approx(0.55)
    assert s[1][1] ** 2 == pytest.approx(0.5)
    assert fall_schedule(8.0, 0)[1][0] == pytest.approx(8.0 ** 0.4 * p0, rel=1e-9)
    y = body_fall(SR, size=1.0, debris="metal")
    assert np.isfinite(y).all() and np.max(np.abs(y)) == pytest.approx(0.8, rel=0.01)


def test_footsteps_walker_and_mood():
    from genny import creatures as C
    assert C.WALKERS["woman"][0] == 120.0 and C.WALKERS["man"][0] == 75.0
    a = C.footsteps(SR, steps=4, seed=1)
    b = C.footsteps(SR, steps=4, seed=1, walker="", mood="neutral")
    assert np.array_equal(a, b)
    w = C.footsteps(SR, steps=4, seed=1, walker="woman")
    m = C.footsteps(SR, steps=4, seed=1, walker="man")
    assert len(w) < len(m)
    sad = C.footsteps(SR, steps=4, seed=1, mood="sad")
    assert np.max(np.abs(sad)) < np.max(np.abs(a))


def test_ruler_slide_raises_pitch_as_inverse_square_length():
    from genny.fdstring import length_slide, twang
    fs, f0, s = 8000, 200.0, 0.3
    n = 8000
    y = np.sin(2 * np.pi * f0 * np.arange(int(n / (1 - s)) + 2) / fs)
    z = length_slide(y, s, n)

    def rate(seg):
        return np.sum(np.diff(np.signbit(seg).astype(int)) != 0) / 2 / (len(seg) / fs)
    assert rate(z[:400]) == pytest.approx(f0 / (1 - s * 0.025) ** 2, rel=0.05)
    assert rate(z[-400:]) == pytest.approx(f0 / (1 - s * 0.975) ** 2, rel=0.05)
    assert np.array_equal(twang(SR, kind="ruler", dur=0.3), twang(SR, kind="ruler", dur=0.3, slide=0.0))
    assert len(twang(SR, kind="ruler", dur=0.3, slide=0.4)) == len(twang(SR, kind="ruler", dur=0.3))
