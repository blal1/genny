"""Rail: the numbers the module claims from its sources, measured on the signal, and every kind renders without a click."""
import numpy as np
import pytest

from genny import rail as R
from genny.analysis import detect_clicks
from genny.dsp import a_weighting_db
from genny.sfx import REGISTRY

SR = 48000


def _third(y, f):
    s = np.fft.rfft(y)
    fr = np.fft.rfftfreq(len(y), 1 / SR)
    s[(fr < f / 2 ** (1 / 6)) | (fr > f * 2 ** (1 / 6))] = 0
    return float(np.std(np.fft.irfft(s, len(y))))


def _dba(y, f):
    return 20 * np.log10(_third(y, f) + 1e-15) + float(a_weighting_db(f))


def _a_total(y):
    s = np.fft.rfft(y)
    f = np.maximum(np.fft.rfftfreq(len(y), 1 / SR), 1.0)
    return 20 * np.log10(np.std(np.fft.irfft(s * 10 ** (a_weighting_db(f) / 20), len(y))) + 1e-15)


def _peak(y, lo, hi):
    """Frequency of the strongest line between lo and hi, Hann window."""
    s = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / SR)
    m = (f >= lo) & (f <= hi)
    return float(f[m][np.argmax(s[m])])


def _centroid(u):
    s = np.abs(np.fft.rfft(u)) ** 2
    return np.sum(np.fft.rfftfreq(len(u), 1 / SR) * s) / np.sum(s)


def _share(y, lo, hi):
    s = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return float(s[(f >= lo) & (f <= hi)].sum() / s.sum())


@pytest.mark.parametrize("kind", R.RAIL_KINDS)
def test_every_kind_renders_without_a_click(kind):
    y = R.rail(sr=SR, kind=kind, dur=2.0)
    assert np.isfinite(y).all() and np.abs(y).max() > 0.01
    assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3
    assert len(detect_clicks(y)) == 0, kind


@pytest.mark.parametrize("speed", [0.0, 20.0, 80.0])
def test_railcar_stays_clean_with_no_rolling_to_mask_it(speed):
    for seed in (0, 1, 2):
        assert len(detect_clicks(R.rail(sr=SR, kind="train_diesel_4t", speed=speed, dur=2.0, seed=seed))) == 0


@pytest.mark.parametrize("kw", [dict(kind="train_welded", rail_len=25.0, speed=100.0), dict(kind="train_welded", speed=170.0, corrugation=0.55), dict(kind="metro_start", dur=8.0)])
def test_options_stay_clean(kw):
    assert len(detect_clicks(R.rail(sr=SR, **{"dur": 3.0, **kw}))) == 0


def test_names_are_registered_and_unknown_kinds_refused():
    assert {"rail", "traction_drive", "curve_squeal"} <= set(REGISTRY)
    with pytest.raises(ValueError):
        R.rail(sr=SR, kind="monorail")
    with pytest.raises(ValueError):
        R.traction_drive(sr=SR, pwm="foc")


# brush model, Johnson 8.64-8.71
def test_brush_small_creep_is_8_70_and_large_creep_saturates():
    xi = 1e-4
    q = (8 * R.PATCH_A * R.PATCH_B / 3) * 1.1 * R.E_STAR * xi
    assert abs(-R.creep_force(xi) - q) / q < 0.03
    assert abs(-R.creep_force(0.05) - 0.3 * R.WHEEL_LOAD) / (0.3 * R.WHEEL_LOAD) < 0.02


def test_brush_rises_over_distance_rolled_not_time():
    """Johnson 8.6: steady at 2a without slip, at 1.6a from rest to Q = .75 mu P."""
    a = R.PATCH_A
    def settle(xi, V):
        q = R.creep_transient(xi, V)
        return (int(np.argmax(np.abs(q) >= (1 - 1e-6) * np.abs(q[-1]))) + 1) * V / R.P / a
    assert abs(settle(1e-4, 5.0) - 2.0) < 0.1 and abs(settle(1e-4, 15.0) - 2.0) < 0.1
    slope = (8 * a * R.PATCH_B / 3) * 1.1 * R.E_STAR
    lo, hi = 1e-4, 0.05
    for _ in range(25):       # the creep giving Q = .75 mu P
        xi = (lo + hi) / 2
        lo, hi = (xi, hi) if -R.creep_force(xi) < 0.75 * 0.3 * R.WHEEL_LOAD else (lo, xi)
    assert xi > 0.75 * 0.3 * R.WHEEL_LOAD / slope
    # Book, half-space: 1.6a. This brush, on the parabolic pressure of 4.58: 1.35a. A gap in the model, reported in suivi/L08.md: the bound only pins the slip shortening the rise.
    assert 1.2 < settle(xi, 5.0) < 1.6


def test_creep_force_stops_following_above_v_over_2a():
    V = 30.0
    cut, f1 = R.creep_cutoff(V, 0.005)
    assert abs(cut - 3000.0) < 1 and abs(f1 - 2300.0) < 10      # the book's example
    def gain(f):
        n = 9600
        t = np.arange(n) / R.P
        q = R.creep_transient(1e-5 * np.sin(2 * np.pi * f * t), V)[n // 2:]
        return np.std(q)
    assert gain(cut) < 0.5 * gain(cut / 30)


# corrugation, Johnson 11.3 and fig. 11.4
def test_corrugation_30_stays_in_contact_55_bounces_at_resonance():
    f0 = R.contact_resonance()
    n = 19200
    least = lambda r: min(R.corrugated_contact(f, r, n)[n // 2:].min() for f in f0 * np.linspace(0.6, 1.2, 25))
    assert least(0.30) > 0.0 and least(0.55) < 0.0


def test_large_corrugation_softens_and_jumps():
    """Book: the frequency at maximum amplitude is below the natural one; the larger corrugation's curve has the jump."""
    f0, n, k = R.contact_resonance(), 480000, 960
    sweep = np.linspace(1.2, 0.5, n) * f0
    def curve(r):
        d = R.corrugated_contact(sweep, r, n)
        amp = np.array([np.ptp(d[i:i + k]) for i in range(0, n - k, k)])
        return sweep[np.argmax(amp) * k] / f0, np.abs(np.diff(amp)).max()
    (fa, ja), (fb, jb) = curve(0.30), curve(0.55)
    assert fb < 0.97 and fb < fa and jb > 3 * ja


def test_corrugation_is_a_line_at_v_over_wavelength():
    kmh, L = 150.0, 0.05
    y = R.rail(sr=SR, kind="train_welded", speed=kmh, corrugation=0.3, wavelength=L, dur=2.0)
    z = R.rail(sr=SR, kind="train_welded", speed=kmh, dur=2.0)
    f = kmh / 3.6 / L
    assert abs(_peak(y, 300, 3000) - f) < 3
    assert _third(y, f) > 2 * _third(z, f)


def test_contact_resonance_is_hertz():
    d0 = (9 * 5e4 ** 2 / (16 * 0.3 * R.E_STAR ** 2)) ** (1 / 3)
    assert abs((2 * np.pi * R.contact_resonance(mass=30.0)) ** 2 - 3 * 5e4 / (2 * d0 * 30.0)) < 1.0


# rolling
def test_welded_has_no_joints_and_fills_its_band():
    y = R.rail(sr=SR, kind="train_welded", speed=300.0, dur=2.0)
    assert _share(y, 100, 5000) > 0.97
    k = int(0.002 * SR)
    e = np.sqrt(np.convolve(y * y, np.ones(k) / k, mode="valid"))[SR // 4:-SR // 4]
    assert e.max() < 3 * np.median(e)      # nothing stands out
    z = R.rail(sr=SR, kind="train_welded", speed=60.0, dur=2.0)
    assert _share(z, 315, 2000) > 0.9 and _share(z, 2500, 5000) < _share(y, 2500, 5000)


def test_tram_band_and_joint_cadence():
    y = R.rail(sr=SR, kind="tram_roll", dur=4.0)
    assert _share(y, 250, 3000) > 0.95 and _share(y, 250, 2000) > 0.85
    k = int(0.005 * SR)
    e = np.sqrt(np.convolve(y * y, np.ones(k) / k, mode="same"))
    e = e - e.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(e, 2 * len(e))) ** 2)
    T = R.TRAM_JOINT / (30.0 / 3.6)
    a, b = int(0.7 * T * SR), int(1.3 * T * SR)
    assert abs((a + np.argmax(ac[a:b])) / SR - T) < 0.02


# engines
@pytest.mark.parametrize("notch,f", [(0, 53.33), (8, 240.0)])
def test_v16_two_stroke_firing_rate(notch, f):
    y = R.rail(sr=SR, kind="train_diesel", notch=notch, speed=0.0, dur=3.0)
    assert abs(_peak(y[SR // 2:], f * 0.92, f * 1.08) - f) < 0.02 * f      # 270 Hz is the turbo
    # the firing rate is the first thing there: nothing at half of it
    assert _third(y, f) > 3 * _third(y, f / 2)


def test_v16_notches_move_up():
    y = R.rail(sr=SR, kind="train_diesel", notch=0, notch_end=8, speed=0.0, dur=6.0)
    k = int(0.3 * SR)                      # each of the 9 notches lasts 6 / 16 s at the ends
    assert abs(_peak(y[:k], 40, 70) - 53.3) < 3 and abs(_peak(y[-k:], 221, 259) - 240.0) < 6


def test_turbo_is_geared_to_the_crank():
    y = R.rail(sr=SR, kind="train_diesel", notch=8, speed=0.0, dur=3.0)
    f = 18 * 900 / 60
    assert abs(_peak(y, f * 0.98, f * 1.02) - f) < 2 and abs(R.notch_rpm(8) - 900) < 1e-9 and abs(R.notch_rpm(0) - 200) < 1e-9


def test_railcar_six_cylinders_at_1800_is_90_hz():
    y = R.rail(sr=SR, kind="train_diesel_4t", speed=0.0, dur=3.0)
    assert abs(_peak(y[SR // 2:], 70, 110) - 90.0) < 2.5


# traction [BT], [CF]
def test_627_rpm_at_15_kmh():
    assert abs(R.traction_rpm(15.0) - 627.0) < 1e-9 and abs(R.traction_rpm(30.0) - 1254.0) < 1e-9
    y = R.traction_drive(sr=SR, speed=15.0, dur=3.0)
    f = 627.0 / 60.0 * 17
    assert abs(_peak(y, f * 0.9, f * 1.1) - f) < 2                      # mesh, on the real shaft speed


def test_sv_pwm_lines_at_2fs_plus_minus_k2f0_and_dpwm_at_fs():
    f0 = 2 * 627.0 / 60.0
    y = R.traction_drive(sr=SR, speed=15.0, fs=1500.0, dur=3.0)
    assert abs(_peak(y, 2800, 3200) - (3000.0 - 2 * f0)) < 3 or abs(_peak(y, 2800, 3200) - (3000.0 + 2 * f0)) < 3
    z = R.traction_drive(sr=SR, speed=15.0, fs=3000.0, pwm="dpwm", dur=3.0)
    assert abs(abs(_peak(z, 2800, 3200) - 3000.0) - 2 * f0) < 3


def test_largest_amplitudes_are_at_2_to_4_khz():
    y = R.traction_drive(sr=SR, speed=15.0, fs=1500.0, dur=3.0)
    bands = [125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000]
    assert 2000 <= max(bands, key=lambda f: _third(y, f)) <= 4000


def test_noise_peaks_near_1500_on_sv_and_3000_on_dpwm_and_sv_is_low_at_2000():
    noise = lambda fs, p: _a_total(R.traction_drive(sr=SR, speed=15.0, fs=fs, pwm=p, dur=1.5))
    sv = {f: noise(f, "sv") for f in (500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0)}
    dp = {f: noise(f, "dpwm") for f in (1000.0, 2000.0, 3000.0, 4000.0)}
    assert max(sv, key=sv.get) == 1500.0 and max(dp, key=dp.get) == 3000.0
    assert sv[2000.0] < sv[1500.0] - 2.0 and sv[2000.0] < dp[2000.0]    # SV-PWM lowest above 1.8 kHz


def test_metro_start_steps_its_inverter_and_climbs():
    dur = 8.0
    y = R.rail(sr=SR, kind="metro_start", dur=dur)
    for i, fs in enumerate((500.0, 1000.0, 1500.0, 2000.0)):
        a = int((i + 0.3) * dur / 4 * SR)
        w = y[a:a + SR // 2]
        f = _peak(w, 2 * fs * 0.8, 2 * fs * 1.2)
        kmh = 40.0 * (a + SR / 4) / (dur * SR)
        assert abs(abs(f - 2 * fs) - 2 * 2 * kmh * 627 / 15 / 60) < 12, (fs, f)


# squeal [POMA], [PAR]
def _squeal():
    return R.curve_squeal(sr=SR, radius=50.0, speed=19.0, dur=4.0, seed=1), R.curve_squeal(sr=SR, radius=2000.0, speed=19.0, dur=4.0, seed=1)


def test_squeal_levels():
    y, s = _squeal()
    assert abs(_a_total(y) - _a_total(s) - 16.0) < 2.0                   # 85 against 69 dB(A)
    lv = {f: _dba(y, f) for f in (630.0, 800.0, 1600.0, 2000.0, 4000.0)}
    assert max(lv, key=lv.get) == 2000.0
    assert all(v > _dba(s, f) + 6.0 for f, v in lv.items())              # all five bands stand out
    gap = {f: _dba(y, f) - _dba(s, f) for f in lv}
    assert abs(gap[1600.0] - 25.0) < 3.0 and max(gap, key=gap.get) == 1600.0


def test_squeal_lines_sit_on_the_wheel_modes():
    y, _ = _squeal()
    for f in (630.0, 800.0, 1600.0, 2000.0, 4000.0):
        # [POMA] gives third-octaves. The line sits above the free mode: stuck phases add the contact's stiffness (1000 Hz on 50 kg).
        assert f * 0.999 <= _peak(y, f * 0.92, f * 2 ** (1 / 6)) < f * 2 ** (1 / 6)


def test_umbrella_mode_at_100_m():
    y = R.curve_squeal(sr=SR, radius=100.0, speed=19.0, dur=3.0, seed=1)
    assert 412 * 0.98 <= _peak(y, 200, 6000) <= 850 * 1.02


def test_squeal_is_the_contact_saturating_not_a_switch():
    sat = 0.3 * R.WHEEL_LOAD / ((8 * R.PATCH_A * R.PATCH_B / 3) * 1.1 * R.E_STAR)
    assert R.attack_angle(50.0) > 3 * sat > 3 * R.attack_angle(5000.0)
    tone = lambda r: _third(R.curve_squeal(sr=SR, radius=r, dur=2.0, seed=1), 2000.0)
    assert tone(50.0) > 10 * tone(2000.0)


def test_squeal_is_intermittent():
    y = R.curve_squeal(sr=SR, radius=50.0, dur=6.0, seed=2)
    k = SR // 10
    e = np.sqrt(np.convolve(y * y, np.ones(k) / k, mode="valid"))[SR // 4:-SR // 4]
    assert e.max() > 1.5 * np.quantile(e, 0.1)


# the rest
def test_pantograph_source_at_4_m():
    k = R.pantograph_delay()
    assert abs(k / SR - (np.hypot(10, 5.2) - np.hypot(10, 2.8)) / 343) < 1e-4
    ac = lambda r: (lambda y: (lambda c: c[k] / c[0])(np.fft.irfft(np.abs(np.fft.rfft(y, 2 * len(y))) ** 2)))(R._pantograph(3.0, 1, np.random.default_rng(1), r))
    assert ac(0.7) - ac(0.0) > 0.3      # 2 r / (1 + r^2) = .94 for a white source: the reflection, not the arc's colour


def test_pantograph_is_sporadic():
    y = R.rail(sr=SR, kind="pantograph_arc", dur=4.0, seed=3)
    h = SR // 20
    e = np.array([np.std(y[i:i + h]) for i in range(0, len(y) - h, h)])
    assert e.max() > 5 * np.quantile(e, 0.2)


def test_tram_bell_is_low_with_two_strikes():
    y = R.rail(sr=SR, kind="tram_bell")
    assert _peak(y, 100, 6000) < 1300 * 0.9
    k = int(0.002 * SR)
    e = np.sqrt(np.convolve(y * y, np.ones(k) / k, mode="same"))
    assert e[int(0.23 * SR):int(0.3 * SR)].max() > 1.05 * e[int(0.2 * SR):int(0.22 * SR)].max()


def test_door_has_its_chime_then_the_ram():
    y = R.rail(sr=SR, kind="train_door")
    a, b = y[:int(0.9 * SR)], y[int(1.0 * SR):int(2.7 * SR)]
    assert abs(_peak(a, 500, 1200) - 880.0) < 4 or abs(_peak(a, 500, 1200) - 698.5) < 4
    assert np.std(b) > 0.3 * np.std(a) and _share(b, 3000, 20000) > _share(a, 3000, 20000)


def test_brake_ends_on_the_release():
    y = R.rail(sr=SR, kind="train_brake", dur=3.0)
    assert len(y) >= 3.5 * SR
    assert _share(y[int(3.0 * SR):], 2000, 20000) > _share(y[:int(1.0 * SR)], 2000, 20000) * 0.5
    assert np.std(y[int(3.05 * SR):int(3.5 * SR)]) > 0.5 * np.std(y[int(2.8 * SR):int(2.95 * SR)])      # the release is not lost under the squeal
    assert _centroid(y[:int(0.5 * SR)]) < _centroid(y[int(3.02 * SR):int(3.4 * SR)])                    # squeal and roll, then broad air


def test_carter_force_closed_form():
    P, a, Rw, mu = 50e3, 0.005, 0.46, 0.3
    xi_s = mu * a / Rw
    small = 1e-3 * xi_s
    assert abs(-float(R.carter_force(small, P, a, Rw, mu)) / small - 2 * P * Rw / a) < 2e-3 * 2 * P * Rw / a
    assert abs(float(R.carter_force(xi_s, P, a, Rw, mu)) + mu * P) < 1e-6
    assert abs(float(R.carter_force(3 * xi_s, P, a, Rw, mu)) + mu * P) < 1e-6
    assert abs(float(R.carter_force(-0.5 * xi_s, P, a, Rw, mu)) - 0.75 * mu * P) < 1e-6


def test_halfspace_resonance_lysmer():
    m, a, G, rho, nu = 30.0, 0.005, 81e9, 7850.0, 0.3
    f, z = R.halfspace_resonance(m, a, G, rho, nu)
    assert abs(f - np.sqrt(4 * G * a / (1 - nu) / m) / (2 * np.pi)) < 1e-6
    B = (1 - nu) * m / (4 * rho * a ** 3)
    assert abs(z - 0.425 / np.sqrt(B)) < 1e-3 * z
