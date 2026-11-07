"""Theme Mechanics of the deep reading: each number below is sourced or measured on the render. Nothing has been listened to."""
import numpy as np
import pytest
from scipy.signal import hilbert

from genny import cyber as C
from genny import engines as E
from genny import foley as FO
from genny import friction as Fr
from genny import heavy_vehicles as H
from genny import machines as M
from genny import rail as R
from genny.analysis import detect_clicks
from genny.reverbs import doppler

SR = 48000


def spec(y):
    y = np.asarray(y, float)
    y = y.mean(axis=1) if y.ndim > 1 else y
    return np.fft.rfftfreq(len(y), 1 / SR), np.abs(np.fft.rfft(y * np.hanning(len(y))))


def peak(y, lo, hi):
    f, X = spec(y)
    m = (f > lo) & (f < hi)
    return float(f[m][np.argmax(X[m])])


def db(y, f0, w=5.0):
    f, X = spec(y)
    return float(20 * np.log10(X[(f > f0 - w) & (f < f0 + w)].max() + 1e-12))


def rms(y):
    return float(np.sqrt(np.mean(np.asarray(y) ** 2)))


def env_line(y, f, bw=0.5):
    e = np.abs(hilbert(y))
    fr, X = spec(e - e.mean())
    return float(20 * np.log10(X[(fr > f - bw) & (fr < f + bw)].max() / np.median(X[(fr > f / 2) & (fr < 2 * f)])))


# ---------------------------------------------------------------------------------- train
def test_train_horn_is_the_k5la():
    y = FO.train(sr=SR, dur=2.0, horn=1.5)[:SR]
    for f in (311.0, 370.0, 415.0, 494.0, 622.0):                  # vehicules_terre.md:231 [K5]
        assert abs(peak(y, f - 12, f + 12) - f) <= 1.0
    assert db(y, 470.0) < db(y, 494.0) - 10                        # the old UNSOURCED 470 Hz is gone


def test_train_without_options_does_not_change():
    a = FO.train(sr=SR, dur=1.0, seed=3)
    assert np.array_equal(a, FO.train(sr=SR, dur=1.0, seed=3, creep=0.0, corrugation=0.0))


def test_creep_noise_follows_the_brush():
    V = 60 / 3.6
    f1 = R.creep_cutoff(V)[1]
    assert f1 == pytest.approx(1276, abs=1)                        # 2.405 V / (2 pi a), a = 5 mm: 2300 Hz at 30 m/s in Johnson's example
    assert R.creep_cutoff(30.0)[1] == pytest.approx(2297, abs=3)
    q = R.creep_noise(0.0005, V, SR, np.random.default_rng(0))
    f, X = spec(q)
    assert 20 * np.log10(X[(f > 200) & (f < 0.5 * f1)].mean() / X[(f > 3 * f1) & (f < 6 * f1)].mean()) > 12    # measured 17: no force above V / 2a
    lev = [rms(R.creep_noise(c, V, SR, np.random.default_rng(0))) for c in (0.0005, 0.001, 0.002, 0.01)]
    assert lev[1] > 1.4 * lev[0] and lev[3] < 0.6 * lev[2]         # grows with the creep, dies when the whole patch slides
    force = [abs(R.creep_force(c, V)) for c in (0.0005, 0.01)]
    assert force[1] == pytest.approx(0.3 * R.WHEEL_LOAD, rel=0.02) # mu P


def test_train_creep_and_corrugation_are_heard():
    base = FO.train(sr=SR, dur=2.0, speed=72.0)
    y = FO.train(sr=SR, dur=2.0, speed=72.0, creep=0.001)
    f, X0 = spec(base)
    _, X1 = spec(y)
    m = (f > 500) & (f < 1500)                                     # below V / 2a = 2000 Hz, above the roar
    assert np.mean(X1[m] ** 2) > 2.0 * np.mean(X0[m] ** 2)
    yc = FO.train(sr=SR, dur=4.0, speed=72.0, corrugation=0.3, wavelength=0.05)
    assert peak(yc, 300, 500) == pytest.approx(400.0, abs=1.0)     # V / wavelength
    assert db(yc, 400) > db(FO.train(sr=SR, dur=4.0, speed=72.0), 400) + 10


# ---------------------------------------------------------------------------------- tyre
def test_tyre_brush_grips_then_sings_at_its_mode():
    s_grip, s_slide = 0.05, 0.8
    grip = C.tyre_squeal(sr=SR, slip=s_grip, wobble=0.0, model="brush", dur=1.0)
    slide = C.tyre_squeal(sr=SR, slip=s_slide, wobble=0.0, model="brush", dur=1.0)
    full = C.tyre_squeal(sr=SR, slip=1.5, wobble=0.0, model="brush", dur=1.0)
    assert rms(slide) > 30 * rms(grip) and rms(full) > 30 * rms(grip)   # below saturation the patch grips
    assert peak(slide, 800, 3000) == pytest.approx(900 + 700 * s_slide, rel=0.03)   # first wheel: 900 + 700 s (ice-simulator), the modal line
    assert len(detect_clicks(slide)) == 0
    with pytest.raises(ValueError):
        C.tyre_squeal(sr=SR, model="x")


def test_tyre_default_is_still_the_band():
    assert rms(C.tyre_squeal(sr=SR, dur=0.5)) > 0.1


# ---------------------------------------------------------------------------------- gears, servo, friction
def test_gearbox_tooth_friction_scales_with_load():
    kw = dict(sr=SR, dur=3.0, rpm=600.0, teeth=23, teeth2=41, wear=0.0, eccentricity=0.0, backlash=0.0)
    g = lambda fr, ld: np.mean([db(C.gearbox(friction=fr, load=ld, **kw), 230.0 * k, 3) for k in (6, 7, 9)])
    assert g(0.3, 0.8) > g(0.0, 0.8) + 4                           # measured about +7: sawtooth 1 / k against the mesh's k^-2.2
    assert g(0.3, 0.8) - g(0.0, 0.8) > g(0.3, 0.2) - g(0.0, 0.2) + 1.5   # mu load (chunk_29 l.24)
    a = C.gearbox(sr=SR, dur=0.5)
    assert np.array_equal(a, C.gearbox(sr=SR, dur=0.5, friction=0.0))


def test_servo_starts_and_stops_sharply():
    n = FO._n(0.5)
    make = lambda env: FO._fin(FO._motor(180.0 * env(n), np.random.default_rng(1)), SR, 0.55)
    span = lambda y: np.flatnonzero(np.abs(y[:n // 480 * 480]).reshape(-1, 480).max(axis=1) > 0.5 * np.abs(y).max())[[0, -1]] * 10
    a, b = span(make(FO._move)), span(make(M.servo_env))
    assert b[0] <= a[0] - 10 and b[1] >= a[1] + 30                 # measured 0 against 20 ms, 480 against 430 ms of 500
    y = FO.servo(sr=SR, moves=1, dur=0.5, seed=1)
    assert np.abs(y).max() > 0.3


def test_stick_slip_stribeck_from_the_film():
    assert Fr.oil_stribeck(0.1, 0.1, 10.0) == pytest.approx(0.041, abs=0.0005)   # Johnson 10.33: R .1, P 1000 N/m, eta .1, 1 um
    assert np.array_equal(Fr.stick_slip(sr=SR, dur=0.3), Fr.stick_slip(sr=SR, dur=0.3, oil=0.0))
    thin = Fr.stick_slip(sr=SR, oil=0.05, pressure=10.0, dur=1.0)       # vs .082 m/s, flat at 0.01 m/s: no creak (the constant's reason in the docstring)
    thick = Fr.stick_slip(sr=SR, oil=1.0, pressure=10.0, dur=1.0)       # vs .004 m/s
    assert rms(thick) > 4 * rms(thin)


# ---------------------------------------------------------------------------------- buzzer, piston, turbo
def test_buzzer_keeps_its_fundamental():
    seg = lambda y, t: y[int(t * SR):int((t + 0.2) * SR)]
    y = FO.buzzer(sr=SR, dur=3.0)
    for t in (0.1, 1.15, 2.4):
        assert db(seg(y, t), 100, 8) > db(seg(y, t), 200, 8) + 10
    beat = FO.buzzer(sr=SR, dur=3.0, beat=0.4)                     # Farnell's hum as an option: at 1.25 s the 100 Hz line has gone and 200 Hz rules
    assert db(seg(beat, 1.15), 200, 8) > db(seg(beat, 1.15), 100, 8) + 10


def test_piston_and_turbo_reach_the_rates_of_their_kin():
    p = FO.piston(sr=SR, cycles=30, rate=12.5)
    e = np.abs(p)
    assert peak(e - e.mean(), 10, 60) == pytest.approx(25.0, abs=0.5)   # top of `machines.jackhammer`
    y = C.turbo(sr=SR, dur=2.0, spool=300000.0, blowoff=0)
    assert peak(y[SR:], 3000, 7000) == pytest.approx(5000.0, rel=0.04)   # 300000 rpm = 5 kHz at the end of the spool
    assert np.array_equal(C.turbo(sr=SR, dur=0.6), C.turbo(sr=SR, dur=0.6, spool=90000.0))


# ---------------------------------------------------------------------------------- phone, transporter
def test_dtmf_at_the_shortest_legal_timing():
    y = FO.dtmf(sr=SR, digits="55", tone_ms=50.0, gap_ms=45.0)     # Farnell l.15316-15318
    assert len(y) / SR * 1000 == pytest.approx(190.0, abs=0.5)
    on = np.flatnonzero(np.abs(y) > 0.02)
    assert (on[-1] - on[0]) / SR * 1000 == pytest.approx(145.0, abs=2.0)    # tone + gap + tone


def test_transporter_activation_bands():
    a = FO.transporter_activate(SR, SR, np.random.default_rng(0))[SR // 5:]    # after the 120 ms blip
    for f in (775.0, 1550.0, 2325.0):                              # fig. 22.3
        assert peak(a, 0.92 * f, 1.08 * f) == pytest.approx(f, abs=4.0)
    assert db(a, 1000.0) < db(a, 775.0) - 12                       # Q 13
    assert np.array_equal(FO.transporter(sr=SR, dur=0.5), FO.transporter(sr=SR, dur=0.5, activate=0))
    y = FO.transporter(sr=SR, dur=1.0, activate=1)
    assert db(y[:SR // 2], 775.0) > db(FO.transporter(sr=SR, dur=1.0)[:SR // 2], 775.0) + 6
    assert len(detect_clicks(y)) == 0


# ---------------------------------------------------------------------------------- vehicles
def test_diesel_pipe_is_a_parameter():
    a = E.diesel_engine(sr=SR, rpm=1500.0, load=0.5, clatter=0.0, dur=2.0)
    assert np.array_equal(a, E.diesel_engine(sr=SR, rpm=1500.0, load=0.5, clatter=0.0, dur=2.0, pipe=7.0))   # the car's preset
    b = E.diesel_engine(sr=SR, rpm=1500.0, load=0.5, clatter=0.0, dur=2.0, pipe=35.0)
    assert db(b, 100, 10) > db(a, 100, 10) + 3                     # measured +8: a long pipe drones lower


def test_tracks_have_no_parasitic_line_at_22_hz():
    """57.36 - 35 = 22.36 Hz: the link rate of an M1 at 40 km/h against the 35 Hz mode of `rumble`, was 15.3 dB at any wheel spacing."""
    for seed in (0, 1):
        y = H.tracks(sr=SR, kind="tank_m1", wheel_spacing=0.7, engine=0.0, dur=6.0, seed=seed)
        assert env_line(y, 22.36) < 10.5
        assert env_line(y, 40 / 3.6 / 0.7) > 15                    # v / spacing stays
        assert env_line(y, 40 / 3.6 / 0.1937) > 15                 # v / pitch stays


def test_pass_by_ground_is_asphalt_with_tyre_horn():
    kmh, r0, h = 60.0, 6.0, 1.8
    y = C._unit(C._third_octave_noise(4 * SR, np.random.default_rng(0), C.road_noise_levels(kmh)[0]))
    flat = doppler(y, SR, speed=0.0, closest=r0, pan=False, air=False, surface="asphalt", height=h)
    bare = doppler(y, SR, speed=0.0, closest=r0, pan=False, air=False)
    f = np.fft.rfftfreq(SR, 1 / SR)
    tf = np.abs(np.fft.rfft(flat[SR:2 * SR])) ** 2 / (np.abs(np.fft.rfft(bare[SR:2 * SR])) ** 2 + 1e-12)
    smooth = np.convolve(tf, np.ones(41) / 41, "same")
    notch = 343.0 / (2 * (np.hypot(r0, h) - r0))                   # in phase on a hard surface: first dip at c / (2 extra path), 642 Hz
    m = (f > 0.5 * notch) & (f < 1.5 * notch)
    assert f[m][np.argmin(smooth[m])] == pytest.approx(notch, rel=0.12)
    z = C.pass_by(sr=SR, dur=5.0, speed=kmh, distance=r0, pan=0, power="none")
    e = np.abs(z[:len(z) // 2400 * 2400]).reshape(-1, 2400).max(axis=1)
    assert 2.0 < np.argmax(e) * 0.05 < 3.0                         # still peaks at the pass
    assert len(detect_clicks(z)) == 0
    assert peak(C.pass_by(sr=SR, dur=5.0, speed=120.0, distance=6.0, power="electric", pan=0)[: SR], 100, 2000) > 0


# ---------------------------------------------------------------------------------- cyber small items
def test_bearing_6004_defect_rate():
    k = C.bearing_frequencies(1750.0, 9, 6.35, 31.0)
    assert k["bpfo"] == pytest.approx(4.5 * 1750.0 / 60.0 * (1.0 - 6.35 / 31.0), rel=1e-9)
    y = C.bearing(sr=SR, dur=4.0, rpm=1750.0, state="outer", severity=0.8, size="6004")
    e = np.abs(hilbert(y))
    assert peak(e - e.mean(), 60, 160) == pytest.approx(k["bpfo"], abs=2.0)    # 104.4 Hz, not the 106.6 of the default 6205 geometry
    with pytest.raises(ValueError):
        C.bearing(sr=SR, dur=0.2, size="x")


def test_transformer_strip_near_1500_hz():
    band = lambda y: np.sum(spec(y)[1][(spec(y)[0] > 1300) & (spec(y)[0] < 1700)] ** 2) / np.sum(spec(y)[1] ** 2)
    a, b = C.transformer(sr=SR, dur=2.0), C.transformer(sr=SR, dur=2.0, strip=1.0)
    assert 10 * np.log10(band(b) / band(a)) > 6
    assert np.array_equal(a, C.transformer(sr=SR, dur=2.0, strip=0.0))


def test_muffler_lengths_darken_the_pipe():
    centroid = lambda y: float(np.sum(spec(y)[0] * spec(y)[1] ** 2) / np.sum(spec(y)[1] ** 2))
    a = C.pipe_engine(sr=SR, dur=2.0, rpm=3000.0)
    assert np.array_equal(a, C.pipe_engine(sr=SR, dur=2.0, rpm=3000.0, muffler=None))
    b = C.pipe_engine(sr=SR, dur=2.0, rpm=3000.0, muffler=[0.23, 0.34, 0.45, 0.57])     # DasEtwas .cpp:172-193
    assert centroid(b) < centroid(a)
    with pytest.raises(ValueError):
        C.pipe_engine(sr=SR, dur=0.2, muffler=[1.0])
