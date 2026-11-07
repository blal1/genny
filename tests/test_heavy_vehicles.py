"""Heavy vehicles, two wheels, tracks, site machines: every variant renders, and the numbers of the spec are measured on the signal."""
import numpy as np
import pytest
from scipy.signal import hilbert

from genny import heavy_vehicles as H
from genny.analysis import detect_clicks
from genny.cyber import _HN_F
from genny.sfx import REGISTRY
from genny.spec import render_spec

SR = H.SR


def _spec(y):
    y = y.mean(axis=1) if y.ndim == 2 else y
    N = 1 << int(np.ceil(np.log2(len(y))) + 1)
    return np.abs(np.fft.rfft(y * np.hanning(len(y)), N)), np.fft.rfftfreq(N, 1.0 / SR)


def _db(sp, fr, f, bw=1.5):
    return 20 * np.log10(sp[(fr > f - bw) & (fr < f + bw)].max() + 1e-12)


def _peak(y, lo, hi):
    sp, fr = _spec(y)
    m = (fr >= lo) & (fr <= hi)
    return fr[m][np.argmax(sp[m])]


def _rms(y):
    return float(np.sqrt(np.mean(y * y)))


def _centroid(y):
    sp, fr = _spec(y)
    return float(np.sum(fr * sp ** 2) / np.sum(sp ** 2))


def _bandpower(y, lo, hi):
    sp, fr = _spec(y)
    return 10 * np.log10(np.sum(sp[(fr >= lo) & (fr < hi)] ** 2) + 1e-12)


def _env_line(y, f, bw=0.5):
    """A line of the envelope against the envelope's median spectrum around it, dB."""
    e = np.abs(hilbert(y))
    sp, fr = _spec(e - e.mean())
    return _db(sp, fr, f, bw) - 20 * np.log10(np.median(sp[(fr > f / 2) & (fr < 2 * f)]) + 1e-12)


def _period(y, lo, hi):
    """Lag in s of the strongest repetition of the 10 ms envelope between lo and hi."""
    k = int(0.01 * SR)
    e = np.sqrt(np.convolve(y * y, np.ones(k) / k, mode="same"))
    e = e - e.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(e, 2 * len(e))) ** 2)
    a, b = int(lo * SR), int(hi * SR)
    return (a + int(np.argmax(ac[a:b]))) / SR


# Whole variants of Plan sons L07, in its order.
VARIANTS = [(H.motorcycle, dict(kind=k)) for k in ("cruiser", "sport", "passby")]
VARIANTS += [(H.scooter, dict(kind=k)) for k in ("2t", "4t")]
VARIANTS += [(H.bus, {})]
VARIANTS += [(H.truck, dict(kind=k)) for k in ("truck_passby", "jake_brake")]
VARIANTS += [(H.tractor, dict(kind=k)) for k in ("tractor", "tractor_vintage")]
VARIANTS += [(H.mower, dict(kind=k)) for k in ("lawn_mower", "lawn_mower_2t")]
VARIANTS += [(H.chainsaw, {})]
VARIANTS += [(H.tracks, dict(kind=k)) for k in ("tracked_vehicle", "tank_m1", "tank_diesel", "snowmobile")]
VARIANTS += [(H.excavator, dict(kind="idle")), (H.hydraulic_pump, {}), (H.hydraulic_pump, dict(kind="arm")),
             (H.excavator, dict(kind="tracks")), (H.excavator, dict(kind="scoop")), (H.tracks, dict(kind="bulldozer"))]
VARIANTS += [(H.wagon, {}), (H.atv, {})]
VARIANTS += [(H.garden_machine, dict(kind=k)) for k in ("souffleur", "taille-haie", "debroussailleuse")]
# Attacks of their own, which detect_clicks takes for steps (it is meant for smooth voices, see test_engines, test_road): grass grains, sparse firings, diesel
# clatter, stones, stops. These get the same criterion as test_engines._ends_clean: clean ends, no jump across the range.
PERCUSSIVE = {("motorcycle", "cruiser"), ("motorcycle", "sport"), ("motorcycle", "passby"), ("truck", "truck_passby"), ("scooter", "2t"), ("scooter", "4t"), ("bus", ""), ("tractor", "tractor"), ("tractor", "tractor_vintage"), ("mower", "lawn_mower"), ("mower", "lawn_mower_2t"),
              ("tracks", "bulldozer"), ("tracks", "tank_diesel"), ("tracks", "tracked_vehicle"), ("tracks", "tank_m1"), ("tracks", "snowmobile"), ("excavator", "idle"), ("excavator", "tracks"), ("excavator", "scoop"),
              ("hydraulic_pump", "arm"), ("wagon", ""), ("atv", ""), ("garden_machine", "taille-haie")}


def _id(c):
    return f"{c[0].__name__}-{c[1].get('kind', '')}"


def test_the_list_has_28_lines():
    assert len(VARIANTS) == 28


@pytest.mark.parametrize("case", VARIANTS, ids=_id)
def test_every_variant_renders_clean(case):
    f, kw = case
    y = np.asarray(f(sr=22050, dur=2.0, **kw))
    assert np.isfinite(y).all() and np.abs(y).max() > 0.1
    m = y.mean(axis=1) if y.ndim == 2 else y
    assert abs(m[0]) < 1e-3 and abs(m[-1]) < 1e-3
    # Loud high-frequency noise jumps from sample to sample without being a click: the jump criterion looks at what lies under it.
    if f is H.bus:                  # the air is checked in test_road
        m = H.bus(sr=22050, doors=0, brake=0, dur=2.0)
    if f is H.mower:                # the grass: no single grain may stand out (raw PhISEM grains have a crest of 14)
        assert np.abs(m).max() / _rms(m) < 8
        m = H.mower(sr=22050, cut=0.0, dur=2.0, **kw)
    if (f.__name__, kw.get("kind", "")) in PERCUSSIVE:
        assert np.abs(np.diff(m)).max() < np.abs(m).max()
    else:
        assert len(detect_clicks(m)) == 0


def test_names_in_the_catalog_and_kinds_refused():
    for name in ("motorcycle", "scooter", "bus", "truck", "tractor", "mower", "chainsaw", "tracks", "hydraulic_pump", "excavator", "wagon", "atv", "garden_machine"):
        assert name in REGISTRY
    for f in (H.motorcycle, H.scooter, H.truck, H.tractor, H.mower, H.tracks, H.hydraulic_pump, H.excavator, H.garden_machine):
        with pytest.raises(ValueError):
            f(sr=SR, kind="sneeze")


def test_renders_through_a_spec():
    y, _ = render_spec({"sr": 22050, "layers": [{"type": "sfx", "kind": "tracks", "params": {"kind": "tank_m1", "dur": 0.5}}]})
    assert np.abs(y).max() > 0.1


# ---------------------------------------------------------------- tracks
def test_m1_at_40_kmh_is_57_4_hz():
    assert abs(H.track_rate(40.0, H.M1_PITCH) - 57.4) < 0.05


def _track_lines(y, f):
    sp, fr = _spec(y)
    return [_db(sp, fr, k * f) - _db(sp, fr, (k + 0.5) * f) for k in (1, 2, 3)], [_db(sp, fr, k * f) for k in (1, 2, 3)]


def test_m1_main_peak_and_harmonics_on_the_signal():
    f = H.track_rate(40.0, H.M1_PITCH)
    over, lv = _track_lines(H.tracks(sr=SR, kind="tank_m1", engine=0.0, dur=4.0), f)
    assert min(over) > 6                    # harmonics stand out of the rest [TR]
    assert lv[0] == max(lv)                 # the main peak is the link rate [TR]


def test_m1_line_survives_its_turbine():
    y = H.tracks(sr=SR, kind="tank_m1", dur=4.0)
    assert abs(_peak(y, 45.0, 70.0) - 57.36) < 1.0


def test_speed_can_be_a_curve():
    y = H.tracks(sr=SR, kind="tank_m1", speed=[10.0, 50.0], engine=0.0, dur=3.0)
    assert np.isfinite(y).all() and _peak(y[:SR // 2], 10.0, 40.0) < _peak(y[-SR // 2:], 40.0, 90.0)


def test_link_rate_follows_speed_and_pitch():
    for kmh, p in ((20.0, 0.1937), (40.0, 0.15), (30.0, 0.25)):
        y = H.tracks(sr=SR, kind="tank_m1", speed=kmh, pitch=p, engine=0.0, dur=4.0)
        f = H.track_rate(kmh, p)
        assert abs(_peak(y, 0.8 * f, 1.2 * f) - f) < 1.0


def test_road_wheel_spacing_is_a_line_of_the_envelope():
    y = H.tracks(sr=SR, kind="tank_m1", wheel_spacing=0.7, engine=0.0, dur=6.0)
    y2 = H.tracks(sr=SR, kind="tank_m1", wheel_spacing=0.9, engine=0.0, dur=6.0)
    a, b = 40.0 / 3.6 / 0.7, 40.0 / 3.6 / 0.9
    assert _env_line(y, a) > _env_line(y2, a) + 3 and _env_line(y2, b) > _env_line(y, b) + 3


def test_slack_adds_slaps():
    a = H.tracks(sr=SR, kind="tank_m1", slack=0.0, engine=0.0, dur=3.0)
    b = H.tracks(sr=SR, kind="tank_m1", slack=1.0, engine=0.0, dur=3.0)
    crest = lambda y: np.abs(y).max() / _rms(y)
    assert crest(b) > 1.1 * crest(a)
    assert _bandpower(b, 1000, 8000) - _bandpower(b, 100, 400) > _bandpower(a, 1000, 8000) - _bandpower(a, 100, 400) + 3


@pytest.mark.parametrize("p, hz", [(0.17145, 8.10), (0.2159, 6.43)])
def test_excavator_crawls_at_its_link_rate(p, hz):
    assert abs(H.track_rate(5.0, p) - hz) < 0.01
    y = H.excavator(sr=SR, kind="tracks", pitch=p, load=0.1, rpm=1650.0, dur=6.0)
    assert abs(1.0 / _period(y, 0.8 / hz, 1.2 / hz) - hz) < 0.3


def test_bulldozer_has_a_link_rate_and_a_blade():
    y = H.tracks(sr=SR, kind="bulldozer", engine=0.0, dur=6.0)
    assert abs(1.0 / _period(y, 0.8 / 8.1, 1.2 / 8.1) - 8.10) < 0.3
    bare = H.tracks(sr=SR, kind="excavator_tracks", engine=0.0, dur=6.0)
    assert _bandpower(y, 2000, 10000) > _bandpower(bare, 2000, 10000)


def test_tank_diesel_v12_at_2000_rpm_is_200_hz():
    y = H.tracks(sr=SR, kind="tank_diesel", engine=2.0, dur=4.0)
    sp, fr = _spec(y)
    assert _db(sp, fr, 200.0) > _db(sp, fr, 230.0) + 10


def test_snowmobile_400_hz_is_2_db_under_the_total():
    assert abs(H.track_rate(91.44, 0.0635) - 400.0) < 0.1
    y = H.tracks(sr=SR, kind="snowmobile", dur=3.0)
    assert abs(_peak(y, 300.0, 500.0) - 400.0) < 2.0
    sp, fr = _spec(y)
    band = np.sum(sp[(fr >= 400 * 2 ** (-1 / 6)) & (fr < 400 * 2 ** (1 / 6))] ** 2)
    assert abs(10 * np.log10(band / np.sum(sp ** 2)) + 2.0) < 0.5


# ---------------------------------------------------------------- hydraulics
def test_pump_is_300_hz_at_2000_rpm_up_to_harmonic_10():
    assert H.pump_rate(2000.0, 9) == 300.0
    y = H.hydraulic_pump(sr=SR, dur=3.0)
    assert abs(_peak(y, 200.0, 400.0) - 300.0) < 1.0
    sp, fr = _spec(y)
    for k in range(1, 11):
        assert _db(sp, fr, 300.0 * k) > _db(sp, fr, 300.0 * (k + 0.5)) + 6, k


def test_pump_pistons_must_be_odd_and_set_the_rate():
    with pytest.raises(ValueError):
        H.hydraulic_pump(sr=SR, pistons=8, dur=0.5)
    assert abs(_peak(H.hydraulic_pump(sr=SR, pistons=7, rpm=1500.0, dur=3.0), 100.0, 250.0) - 175.0) < 1.0


def test_pump_rises_with_pressure():
    lo, hi = (H.hydraulic_pump(sr=SR, pressure=p, dur=2.0) for p in (80.0, 200.0))
    assert _rms(hi) > 1.4 * _rms(lo)
    sp, fr = _spec(lo)
    sq, fq = _spec(hi)
    # The fall from line 1 to line 2 is smaller under load, as in Randall's mesh lines (the law borrowed by the pump).
    assert _db(sq, fq, 600.0) - _db(sq, fq, 300.0) < _db(sp, fr, 600.0) - _db(sp, fr, 300.0)


def test_arm_stops_and_blows_off():
    y = H.hydraulic_pump(sr=SR, kind="arm", dur=3.0)
    t = int(0.65 * 3.0 * SR)
    assert _bandpower(y[t + 2000:], 3000, 12000) > _bandpower(y[:t - 100], 3000, 12000) + 6
    hit = np.abs(y[t - 200:t + 2000]).max()
    assert hit > 2 * np.abs(y[t - 8000:t - 200]).max()


def test_excavator_idle_has_engine_and_pump():
    y = H.excavator(sr=SR, kind="idle", dur=4.0)
    sp, fr = _spec(y)
    assert _db(sp, fr, 82.5) > _db(sp, fr, 110.0) + 6                      # 6 cylinders at 1650
    assert _db(sp, fr, 247.5) > _db(sp, fr, 280.0) + 6                     # 9 pistons at 1650


def test_scoop_tips_out_gravel_at_cooks_frequency():
    y = H.excavator(sr=SR, kind="scoop", dur=4.0)
    t = int(0.65 * 4.0 * SR)
    pile = y[t:t + int(0.6 * SR)]
    assert _bandpower(pile, 5000, 8000) > _bandpower(y[int(0.45 * 4.0 * SR):t - 100], 5000, 8000) + 10
    assert abs(_peak(pile, 3000.0, 12000.0) - H.GRAVEL_HZ) < 1500.0


# ---------------------------------------------------------------- trucks
def test_cv_table_is_harmonoise_table_6_3_6_4():
    r, p = H.cv_noise_levels(70.0)
    band = lambda f: int(np.argmin(np.abs(_HN_F - f)))
    assert (r[band(25)], r[band(100)], r[band(400)], r[band(10000)]) == (79.5, 85.5, 95.0, 80.3)
    assert (p[band(25)], p[band(63)], p[band(10000)]) == (97.7, 109.5, 80.2)
    r2, p2 = H.cv_noise_levels(140.0)
    assert np.allclose(p2[:band(200) + 1], p[:band(200) + 1]) and abs(p2[band(250)] - p[band(250)] - 8.5) < 1e-9
    r6, p6 = H.cv_noise_levels(70.0, axles=2)
    assert abs(r6[band(25)] - 76.5) < 0.02 and (p6[band(63)], p6[band(10000)]) == (101.5, 83.3)
    assert len(r) == len(_HN_F) == 29


def test_moto_4b_levels():
    assert H.moto_noise_levels(70.0)[1] == 101.9
    assert H.moto_noise_levels(140.0)[5] - H.moto_noise_levels(70.0)[5] == pytest.approx(12.6)


def test_cv_rolling_on_the_signal_follows_the_table():
    rng = np.random.default_rng(0)
    roll_, prop = H.cv_sources(70.0, rng.standard_normal(2 * SR), 2 * SR, rng)
    lr, lp = H.cv_noise_levels(70.0)
    F = np.abs(np.fft.rfft(roll_)) ** 2
    fr = np.fft.rfftfreq(len(roll_), 1.0 / SR)
    i = np.arange(8, 24)
    lev = np.array([10 * np.log10(np.sum(F[(fr >= _HN_F[k] * 2 ** (-1 / 6)) & (fr < _HN_F[k] * 2 ** (1 / 6))])) for k in i])
    assert np.abs((lev - lev.mean()) - (lr[i] - lr[i].mean())).max() < 3.0
    want = np.sum(10 ** (lp / 10)) / np.sum(10 ** (lr / 10))
    assert abs(np.sum(prop ** 2) / np.sum(roll_ ** 2) / want - 1.0) < 0.01


def test_truck_passby_uses_the_75_cm_source(monkeypatch):
    heights = []
    real = H.doppler

    def spy(x, *a, **k):
        heights.append(k["height"])
        return real(x, *a, **k)
    monkeypatch.setattr(H, "doppler", spy)
    H.truck(sr=SR, dur=1.0)
    assert heights == [0.75 + H.EAR, 0.01 + H.EAR]


def _doppler_ratio(y, f, half=0.15):
    """(line before the pass, line after), from the first and last 30 % of the sound."""
    k = int(0.3 * len(y))
    return _peak(y[:k], f * (1 - half), f * (1 + half)), _peak(y[-k:], f * (1 - half), f * (1 + half))


def test_truck_passby_engine_line_drops_by_doppler():
    y = H.truck(sr=SR, speed=80.0, rpm=1800.0, load=1.0, dur=8.0, distance=5.0, pan=0)
    a, b = _doppler_ratio(y, 90.0)
    v = 80.0 / 3.6
    assert abs(a / b - (343.0 + v) / (343.0 - v)) < 0.02
    assert a > 90.0 > b


def test_passby_is_stereo_and_mono_on_request():
    assert H.truck(sr=22050, dur=1.0).ndim == 2 and H.truck(sr=22050, dur=1.0, pan=0).ndim == 1
    assert H.motorcycle(sr=22050, kind="passby", dur=1.0).ndim == 2


def test_jake_brake_barks_at_90_hz_at_1800_rpm():
    y = H.truck(sr=SR, kind="jake_brake", dur=3.0)
    sp, fr = _spec(y)
    assert abs(_peak(y, 60.0, 120.0) - 90.0) < 1.5
    assert _db(sp, fr, 90.0) > _db(sp, fr, 135.0) + 10


def test_jake_brake_is_one_pop_per_cylinder_and_louder_than_the_engine():
    n = H._n(1.0)
    pops = H.jake_pops(np.full(n, 1800.0), n)
    assert np.sum((pops[1:] > 0.1) & (pops[:-1] <= 0.1)) in (89, 90)     # 90 a second, the rate of `_events`
    with_pops, without = (H._jake(1800.0, -1.0, n, 1.0, 0, pops=p) for p in (True, False))
    line = lambda y: _db(*_spec(y), 90.0) - _db(*_spec(y), 135.0)
    assert line(with_pops) > line(without) + 6                           # `pipe_engine` normalises: the barks, not the absolute level
    assert np.abs(H.truck(sr=SR, kind="jake_brake", muffler=0, dur=1.0)).max() > np.abs(H.diesel_engine(sr=SR, kind="diesel_i6_truck", rpm=1800.0, dur=1.0)).max()


def test_jake_brake_is_worse_with_no_muffler():
    m, o = (H.truck(sr=SR, kind="jake_brake", muffler=k, dur=1.5) for k in (1, 0))
    assert _rms(o) > 1.4 * _rms(m) and _bandpower(o, 1500, 8000) > _bandpower(m, 1500, 8000) + 6


def test_bus_idles_and_the_air_is_on_top():
    y = H.bus(sr=SR, dur=6.0)
    i = _bandpower(y[int(0.5 * SR):int(1.5 * SR)], 3000, 12000)
    assert _bandpower(y[int(2.5 * SR):int(3.5 * SR)], 3000, 12000) > i + 6        # door, its stroke ending at 3 s
    assert _bandpower(y[int(4.3 * SR):int(5.3 * SR)], 3000, 12000) > i + 6        # brake from 4.5 s
    z = H.bus(sr=SR, doors=0, brake=0, dur=6.0)
    assert abs(1.0 / _period(z, 0.02, 0.06) - 6 * 540.0 / 120.0) < 1.5


def test_bus_rolling_adds_tyres():
    a, b = (H.bus(sr=SR, doors=0, brake=0, speed=s, dur=2.0) for s in (0.0, 40.0))
    assert _bandpower(b, 500, 2000) - _bandpower(b, 10, 100) > _bandpower(a, 500, 2000) - _bandpower(a, 10, 100)


# ---------------------------------------------------------------- two wheels
def test_cruiser_fires_315_405():
    y = H.motorcycle(sr=SR, kind="cruiser", rpm=2400.0, dur=4.0)
    sp, fr = _spec(y)
    assert _db(sp, fr, 40.0) > _db(sp, fr, 160.0) + 10         # 1 + exp(2 pi i k .4375) vanishes at k = 8 (test_engines)


def test_cruiser_gears_are_the_source_ratios():
    assert H.MOTO_GEARS[0] == 3.34 and H.MOTO_GEARS[-1] == 1.0 and H.MOTO_FINAL == 2.9
    assert abs(H.moto_rpm(3, 60.0) - 60.0 / 3.6 / (2 * np.pi * 0.31) * 60.0 * 1.71 * 2.9) < 1e-9
    y = H.motorcycle(sr=SR, kind="cruiser", speed=60.0, gear=3, dur=4.0)
    assert abs(_peak(y, 30.0, 60.0) - H.moto_rpm(3, 60.0) * 2 / 120.0) < 2.0


def test_cruiser_idles_at_900_and_stays_under_its_limit():
    assert abs(_peak(H.motorcycle(sr=SR, kind="cruiser", dur=4.0), 25.0, 40.0) - 2 * 2 * 900.0 / 120.0) < 1.0
    y = H.motorcycle(sr=SR, kind="cruiser", speed=300.0, gear=1, dur=3.0)
    assert abs(_peak(y, 80.0, 110.0) - 2 * 5600.0 / 120.0) < 1.5


def test_sport_is_400_hz_at_12000_rpm():
    y = H.motorcycle(sr=SR, kind="sport", dur=4.0)
    assert abs(_peak(y, 300.0, 500.0) - 400.0) < 2.0


def test_crossplane_sport_brings_the_half_orders():
    def low(y):
        sp, fr = _spec(y)
        return max(_db(sp, fr, k * 100.0) for k in (1, 3, 5)) - _db(sp, fr, 400.0)
    assert low(H.motorcycle(sr=SR, kind="sport", crossplane=1, dur=3.0)) > low(H.motorcycle(sr=SR, kind="sport", dur=3.0)) + 10


def test_motorcycle_passby_has_no_rolling_and_drops_by_doppler(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a motorcycle passby has propulsion only")
    monkeypatch.setattr(H, "tyre_roll", boom)
    monkeypatch.setattr(H, "_third_octave_noise", boom)
    y = H.motorcycle(sr=SR, kind="passby", speed=80.0, gear=3, dur=8.0, distance=5.0, pan=0)
    f = H.moto_rpm(3, 80.0) * 2 / 120.0
    a, b = _doppler_ratio(y, 2 * f)
    v = 80.0 / 3.6
    assert abs(a / b - (343.0 + v) / (343.0 - v)) < 0.03


@pytest.mark.parametrize("kind, hz", [("2t", 116.667), ("4t", 50.0)])
def test_scooter_note_holds_while_the_speed_climbs(kind, hz):
    y = H.scooter(sr=SR, kind=kind, belt=0.0, dur=6.0)
    mid, late = y[int(1.0 * SR):int(3.0 * SR)], y[int(4.0 * SR):]
    a, b = _peak(mid, 0.8 * hz, 1.2 * hz), _peak(late, 0.8 * hz, 1.2 * hz)
    assert abs(a / hz - 1.0) < 0.012 and abs(b / hz - 1.0) < 0.012
    assert H.cvt_rpm(7000.0, 6.0, 48000)[0] == 0.3 * 7000.0        # a start below the note; under the 25 Hz high-pass on a 4T, so not measured there
    if kind == "2t":
        assert _peak(y[:int(0.2 * SR)], 0.2 * hz, 1.2 * hz) < 0.85 * hz


def test_scooter_belt_rises_with_speed():
    y = H.scooter(sr=SR, kind="4t", belt=1.0, dur=4.0)
    z = H.scooter(sr=SR, kind="4t", belt=0.0, dur=4.0)
    assert _bandpower(y, 800, 2000) > _bandpower(z, 800, 2000) + 3
    assert _bandpower(y[int(3 * SR):], 800, 2000) > _bandpower(y[int(0.5 * SR):int(1.5 * SR)], 800, 2000) + 3


# ---------------------------------------------------------------- tractors, saws, garden
@pytest.mark.parametrize("rpm, hz", [(2100.0, 70.0), (850.0, 28.33)])
def test_tractor_firing_rate(rpm, hz):
    y = H.tractor(sr=SR, rpm=rpm, dur=4.0)
    assert abs(1.0 / _period(y, 0.5 / hz, 1.5 / hz) - hz) < 0.08 * hz


@pytest.mark.parametrize("stroke, period", [(4, 0.2), (2, 0.1)])
def test_vintage_fires_each_stroke_apart(stroke, period):
    y = H.tractor(sr=SR, kind="tractor_vintage", stroke=stroke, dur=4.0)
    assert abs(_period(y, 0.7 * period, 1.3 * period) - period) < 0.01


def test_mower_blade_at_100_hz_over_firing_at_25():
    y = H.mower(sr=SR, dur=4.0)
    sp, fr = _spec(y)
    assert _db(sp, fr, 100.0) > max(_db(sp, fr, 75.0), _db(sp, fr, 125.0)) + 6      # 100 is also the 4th harmonic of 25 [spec]: the blade must stand over its neighbours
    assert _db(sp, fr, 100.0) > _db(sp, fr, 112.5) + 10
    assert _db(sp, fr, 50.0) > _db(sp, fr, 62.5) + 6


def test_mower_blades_set_the_line_and_cutting_dominates():
    three = H.mower(sr=SR, blades=3, dur=4.0)
    sp, fr = _spec(three)
    assert _db(sp, fr, 150.0) > _db(sp, fr, 137.5) + 10
    dry, cutting = (H.mower(sr=SR, cut=c, dur=2.0) for c in (0.0, 1.0))
    tilt = lambda u: _bandpower(u, 1000, 8000) - _bandpower(u, 40, 400)
    assert tilt(cutting) > tilt(dry) + 6
    y = H.mower(sr=SR, dur=2.0)
    assert _bandpower(y, 1000, 24000) > _bandpower(y, 0, 1000)                  # the cutting dominates under the deck [MOW]


def test_mower_2t_fires_at_50_hz():
    y = H.mower(sr=SR, kind="lawn_mower_2t", blades=3, dur=4.0)
    sp, fr = _spec(y)
    assert _db(sp, fr, 50.0) > _db(sp, fr, 37.5) + 10


def test_chainsaw_fires_once_per_turn():
    y = H.chainsaw(sr=SR, dur=4.0)
    assert abs(_peak(y, 120.0, 200.0) - 9400.0 / 60.0) < 1.5


def test_chainsaw_bite_takes_15_percent():
    y = H.chainsaw(sr=SR, bite=1.0, dur=8.0)
    mid = y[int(0.45 * 8 * SR):int(0.7 * 8 * SR)]
    assert abs(_peak(mid, 110.0, 170.0) - 0.85 * 9400.0 / 60.0) < 2.0


def test_chainsaw_third_octaves_rise_with_speed():
    assert _centroid(H.chainsaw(sr=SR, rpm=9400.0, dur=2.0)) > 1.3 * _centroid(H.chainsaw(sr=SR, rpm=2800.0, dur=2.0))


@pytest.mark.parametrize("kind, line", [("souffleur", 7500.0 * 12 / 60.0), ("debroussailleuse", 8500.0 / 1.43 * 2 / 60.0)])
def test_garden_two_stroke_and_its_tool(kind, line):
    y = H.garden_machine(sr=SR, kind=kind, dur=3.0)
    rate = H.GARDEN[kind] / 60.0
    assert abs(_peak(y, 0.8 * rate, 1.2 * rate) - rate) < 2.0
    assert abs(_peak(y, 0.97 * line, 1.03 * line) - line) < 2.0


def test_hedge_trimmer_chatters():
    y = H.garden_machine(sr=SR, kind="taille-haie", dur=3.0)
    rate = 6000.0 / 60.0
    assert abs(_peak(y, 80.0, 120.0) - rate) < 2.0
    assert _env_line(_hp_(y), 2 * 6000.0 / 5.0 / 60.0, 1.0) > 6


def _hp_(y):
    s = np.fft.rfft(y)
    s[np.fft.rfftfreq(len(y), 1.0 / SR) < 1500.0] = 0
    return np.fft.irfft(s, len(y))


@pytest.mark.parametrize("cyl, strong, weak", [(1, 66.67, 83.3), (2, 100.0, 133.33)])
def test_atv_firing(cyl, strong, weak):
    # 33.3 Hz harmonics of 4000 rpm; a 270 degree twin nulls harmonic 4 (4 x .375 = 1.5).
    sp, fr = _spec(H.atv(sr=SR, cylinders=cyl, dur=4.0))
    assert _db(sp, fr, strong) > _db(sp, fr, weak) + 6


def test_wagon_trots_at_the_horse_rate():
    y = H.wagon(sr=SR, dur=8.0)
    # Two beats a stride of 1.49 /s (Heglund & Taylor via creatures.GAITS), 358 footfalls a minute in diagonal pairs.
    assert abs(_period(y, 0.25, 0.45) - 60.0 / 358.0 * 2) < 0.03


def test_wagon_speed_moves_the_trot():
    y = H.wagon(sr=SR, speed=1.5, dur=8.0)
    assert abs(_period(y, 0.5, 0.9) - 2 * 60.0 / 179.0) < 0.05


def _octaves(y):
    P = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return np.array([10 * np.log10(P[(f >= c / np.sqrt(2)) & (f < c * np.sqrt(2))].sum()) for c in H.MOTO_F])


@pytest.mark.parametrize("kmh", [50.0, 110.0])
def test_moto_propulsion_follows_the_4b_octave_profile(kmh):
    x = np.random.default_rng(3).standard_normal(SR * 2)
    y = H.moto_propulsion(x, kmh)
    got, want = _octaves(y), H.moto_noise_levels(kmh)
    assert np.max(np.abs((got - got.mean()) - (want - want.mean()))) < 1.0
    assert abs(_rms(y) / _rms(x) - 1.0) < 1e-4


def test_motorcycle_passby_is_shaped_by_moto_levels(monkeypatch):
    seen = []
    real = H.moto_propulsion
    monkeypatch.setattr(H, "moto_propulsion", lambda y, kmh: seen.append(kmh) or real(y, kmh))
    H.motorcycle(sr=SR, kind="passby", speed=90.0, dur=1.0)
    assert seen == [90.0]


def test_chain_band_at_the_link_rate(monkeypatch):
    assert abs(float(H.chain_rate(9400.0)) - 14 * 9400.0 / 60.0) < 1e-9
    seen = []
    real = H._tvbp
    monkeypatch.setattr(H, "_tvbp", lambda x, fc, q: seen.append(np.asarray(fc)) or real(x, fc, q))
    H.chainsaw(sr=SR, dur=1.0)
    assert abs(float(np.median(seen[0])) - 14 * 9400.0 / 60.0) < 1.0


def test_brushcutter_head_turns_slower_than_the_crank(monkeypatch):
    seen = []
    real = H.fan
    monkeypatch.setattr(H, "fan", lambda **kw: seen.append(kw["rpm"]) or real(**kw))
    H.garden_machine(sr=SR, kind="debroussailleuse", dur=1.0)
    assert abs(seen[0] - 8500.0 / 1.43) < 1e-6
