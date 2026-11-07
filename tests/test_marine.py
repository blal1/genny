"""Ships: the numbers the module claims from its sources, measured on the signal, and every variant renders without a click."""
import numpy as np
import pytest
from scipy.signal import welch

from genny import marine as M
from genny.analysis import detect_clicks
from genny.sfx import REGISTRY

SR = 48000


def _peak(y, lo, hi):
    s = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / SR)
    m = (f >= lo) & (f <= hi)
    return float(f[m][np.argmax(s[m])])


def _psd(y):
    return welch(y, SR, nperseg=2 ** 16)


def _level(y, f0, ref, width=0.08):
    """Continuum in dB at f0: median over the band, which lines cannot move, in a Welch estimate 0.73 Hz wide."""
    f, p = _psd(y)
    return 10 * np.log10(np.median(p[(f > f0 * (1 - width)) & (f < f0 * (1 + width))])) + ref


def _db(x):
    return 20 * np.log10(np.std(x) + 1e-15)


def _active(y, thr=0.02, k=480):
    """Spans (s) above thr of the peak."""
    e = np.sqrt(np.convolve(y * y, np.ones(k) / k, mode="same"))
    on = e > thr * e.max()
    d = np.diff(on.astype(int))
    a, b = np.flatnonzero(d == 1) / SR, np.flatnonzero(d == -1) / SR
    return [(float(x), float(z)) for x, z in zip(a, b) if z - x > 0.2]


VARIANTS = [("ship", dict(kind="cargo", listen="under")), ("ship", dict(kind="tanker", speed=20.0)), ("ship", dict(cavitation=1, speed=16.0)), ("ship", dict(listen="air")),
            ("ship_engine", dict()), ("ship_engine", dict(strokes=4)), ("ship_horn", dict()), ("ship_horn", dict(length=30.0, signal="sail", distance=100.0, coast=300.0)),
            ("fog_signal", dict()), ("fog_signal", dict(signal="stopped")), ("fog_signal", dict(signal="sail")), ("fog_signal", dict(signal="anchored")), ("ship_bell", dict()),
            ("outboard", dict()), ("outboard", dict(strokes=2)), ("outboard", dict(planing=1)), ("outboard", dict(listen="under")),
            ("jet_ski", dict()), ("jet_ski", dict(jump=1)), ("jet_ski", dict(listen="under")), ("sailboat", dict()),
            ("submarine", dict(view="out")), ("submarine", dict(view="in")), ("foghorn", dict()),
            ("lap", dict(kind="clapot")), ("lap", dict(kind="quai")), ("lap", dict(kind="baignade"))]


@pytest.mark.parametrize("name,kw", VARIANTS, ids=[f"{n}-{i}" for i, (n, _) in enumerate(VARIANTS)])
def test_every_variant_renders_without_a_click(name, kw):
    for seed in (0, 1, 2):
        y = getattr(M, name)(sr=SR, seed=seed, **kw)
        assert np.isfinite(y).all() and 1e-3 < np.abs(y).max() < 1.2      # absolute levels, some small; the 1.2 guard must not have fired
        assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3
        assert len(detect_clicks(y)) == 0, (name, kw, seed)


def test_names_are_registered_and_bad_values_refused():
    assert {"ship", "ship_engine", "ship_horn", "fog_signal", "ship_bell", "outboard", "jet_ski", "sailboat", "submarine", "foghorn", "lap"} <= set(REGISTRY)
    for call in (lambda: M.ship(sr=SR, kind="yacht"), lambda: M.ship(sr=SR, listen="deck"), lambda: M.ship(sr=SR, model="bogus"), lambda: M.ship_engine(sr=SR, strokes=3), lambda: M.outboard(sr=SR, strokes=3),
                 lambda: M.fog_signal(sr=SR, signal="tug"), lambda: M.submarine(sr=SR, view="up"), lambda: M.lap(sr=SR, kind="river"), lambda: M.foghorn(sr=SR, kind="tyfon"),
                 lambda: M.ship_bell(sr=SR, mouth=200.0), lambda: M.ship_bell(sr=SR, hammer=0.01)):
        with pytest.raises(ValueError):
            call()


# Wales-Heitmeyer [CH 3.9-3.11], [JE]
def test_wales_heitmeyer_values_of_the_spec():
    assert abs(M.wales_heitmeyer(30.0) - 177) < 0.5 and abs(M.wales_heitmeyer(100.0) - 158) < 0.5 and abs(M.wales_heitmeyer(1000.0) - 131) < 0.5
    assert M.wales_heitmeyer(10.0) == M.wales_heitmeyer(30.0)      # flat under 30 Hz


@pytest.mark.parametrize("f,db", [(30.0, 177.0), (100.0, 158.0), (1000.0, 131.0)])
def test_wales_heitmeyer_measured_on_the_render(f, db):
    y = M.ship(sr=SR, kind="cargo", listen="under", dur=40.0)
    assert abs(_level(y, f, M.SHIP_REF_DB) - db) < 1.0


def test_speed_length_and_sonic_laws_on_the_render():
    lv = lambda **kw: _level(M.ship(sr=SR, dur=40.0, seed=3, **kw), 1000.0, M.SHIP_REF_DB)
    base = lv(speed=10.0)
    assert abs(lv(speed=20.0) - base - 60 * np.log10(2)) < 0.5
    assert abs(lv(speed=10.0, length=2 * M.LENGTH_REF) - base - 20 * np.log10(2)) < 0.5
    assert abs(lv(speed=10.0, model="sonic") - base - 8.0) < 0.5


def test_blade_rate_of_the_cargo_comb():
    y = M.ship(sr=SR, dur=30.0)
    assert abs(_peak(y, 5, 12) - 100 * 5 / 60) < 0.1
    f, p = _psd(y)
    i = np.argmin(np.abs(f - 100 * 5 / 60))
    assert 10 * np.log10(p[i]) + M.SHIP_REF_DB > M.wales_heitmeyer(8.3) + 6


# Ross [CH 3.1-3.4]
def test_ross_172_4_db_and_measured():
    assert abs(M.ross(100.0, 20.0, 30000.0) - 172.4) < 0.1
    assert M.ross(100.0, 20.0, 90000.0) == M.ross(100.0, 20.0, 30000.0)      # DT < 30000 t
    y = M.ship(sr=SR, kind="tanker", speed=20.0, dur=40.0)
    assert abs(_level(y, 300.0, M.SHIP_REF_DB, 0.03) - M.ross(300.0, 20.0, 30000.0)) < 1.0
    assert abs(_level(y, 200.0, M.SHIP_REF_DB, 0.04) - _level(y, 400.0, M.SHIP_REF_DB, 0.04) - 20 * np.log10(2)) < 1.0
    z = M.ship(sr=SR, kind="tanker", speed=20.0, tons=3000.0, dur=40.0)
    assert abs(_level(y, 300.0, M.SHIP_REF_DB, 0.03) - _level(z, 300.0, M.SHIP_REF_DB, 0.03) - 9.0) < 0.7


# Wittekind, not verified: SL1's reading of table 3.2, onset, the extract's 50-150 Hz peak
def test_cavitation_only_above_vc():
    a, b = M.ship(sr=SR, speed=8.0, cavitation=1, dur=10.0), M.ship(sr=SR, speed=8.0, cavitation=0, dur=10.0)
    assert np.array_equal(a, b)
    c, d = M.ship(sr=SR, speed=16.0, cavitation=1, dur=10.0), M.ship(sr=SR, speed=16.0, cavitation=0, dur=10.0)
    band = lambda y: _db(M._band(y, 50.0, 150.0))
    assert band(c) > band(d) + 6                                # where the extract puts it; its level (A) is UNSOURCED
    crackle = lambda y: _db(M._band(y, 500.0, 1500.0))
    assert crackle(c) > crackle(d) + 3


def test_sl1_polynomial_as_read():
    assert abs(M.wittekind_sl1(100.0) - 122.2) < 0.01          # 125 + 35 - 80 + 60 - 20 + 2.2
    assert M.wittekind_sl1(1000.0) == M.wittekind_sl1(400.0) == 40.0   # the rounded coefficients go negative there: floored


def test_cavitation_peaks_in_50_150_on_blade_harmonics_and_modulates_at_blade_rate():
    y = M.ship(sr=SR, speed=16.0, cavitation=1, dur=20.0)
    bpf = 100 / 14 * 16 * 5 / 60
    f = _peak(y, 20, 400)
    assert 50 <= f <= 150 and abs(f / bpf - round(f / bpf)) < 0.03
    band = M._band(y, 150.0, 350.0)
    e = np.abs(band) - np.abs(band).mean()
    assert abs(_peak(e, 5, 15) - bpf) < 0.2


# air-water, chunk_07
def test_air_water_is_29_5_db_on_the_render():
    assert abs(M.AIR_WATER_DB + 29.5) < 0.05
    a, b = M.ship(sr=SR, listen="under", dur=8.0), M.ship(sr=SR, listen="air", engine=0.0, dur=8.0)
    assert abs(_db(a) - _db(b) - 29.5) < 0.1


# slow diesel
@pytest.mark.parametrize("kw,f", [(dict(), 10.0), (dict(strokes=4), 5.0), (dict(rpm=150.0), 15.0)])
def test_ship_engine_firing_rate(kw, f):
    y = M.ship_engine(sr=SR, dur=8.0, **kw)
    assert abs(_peak(y, f * 0.7, f * 1.3) - f) < 0.2
    s = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    fr = np.fft.rfftfreq(len(y), 1 / SR)
    assert s[np.argmin(np.abs(fr - f))] > 10 * np.median(s[(fr > f * 1.1) & (fr < f * 1.9)])


def test_hull_pulse_is_the_low_end():
    a, b = M.ship_engine(sr=SR, hull=1.0, dur=6.0), M.ship_engine(sr=SR, hull=0.0, dur=6.0)
    low = lambda y: _db(M._band(y, 6.0, 25.0))
    assert low(a) > low(b) + 6


# horn [COL]
@pytest.mark.parametrize("length,lo,hi,db,nm", [(250.0, 70, 200, 143, 2.0), (200.0, 70, 200, 143, 2.0), (100.0, 130, 350, 138, 1.5), (50.0, 250, 700, 130, 1.0), (10.0, 250, 700, 120, 0.5)])
def test_horn_table_measured(length, lo, hi, db, nm):
    assert M.horn_spec(length) == ((float(lo), float(hi)), float(db), nm)
    y = M.ship_horn(sr=SR, length=length)
    w = y[SR:4 * SR]
    f = float(np.sqrt(lo * hi))
    assert abs(_peak(w, lo, hi) - f) < 0.03 * f                    # the fundamental, in its band, strongest there
    assert abs(_peak(w, f * 1.05, f * 1.2) - f * M.SECOND_TONE) < 0.03 * f     # the second tone
    assert abs(20 * np.log10(M.horn_band_rms(w)) + M.HORN_REF_DB - db) < 0.5       # the annex's level, in the band 180-700 Hz


def test_horn_is_rich_and_fundamental_leads():
    w = M.ship_horn(sr=SR, length=250.0)[SR:4 * SR]
    f = float(np.sqrt(70 * 200))
    s = np.abs(np.fft.rfft(w * np.hanning(len(w))))
    fr = np.fft.rfftfreq(len(w), 1 / SR)
    h = [s[(fr > k * f * 0.98) & (fr < k * f * 1.02)].max() for k in (1, 2, 3, 4)]
    assert h[0] == max(h) and all(x > 0.03 * h[0] for x in h)


def test_horn_distance_and_coast():
    a, b = M.ship_horn(sr=SR), M.ship_horn(sr=SR, distance=100.0)
    assert 40.0 < _db(a) - _db(b) < 50.0                           # 1 / r, and air on top
    c = M.ship_horn(sr=SR, signal="short", coast=343.0, length=100.0)
    assert len(c) > len(M.ship_horn(sr=SR, signal="short", length=100.0)) + 1.9 * SR
    e = np.abs(c)
    assert np.max(e[int(2.2 * SR):int(2.9 * SR)]) > 0.0003 * np.max(e[:SR])      # 2 x 343 m later, 1 / 686 x .5


# rule 35 [R35, extract]
@pytest.mark.parametrize("signal,blasts", [("underway", [5.0]), ("stopped", [5.0, 5.0]), ("sail", [5.0, 1.0, 1.0])])
def test_fog_signal_sequences(signal, blasts):
    spans = _active(M.fog_signal(sr=SR, signal=signal))
    assert len(spans) == len(blasts)
    for (a, b), d in zip(spans, blasts):
        assert abs(b - a - d) < 0.35 and 4 * (d > 1) + 0.8 * (d <= 1) <= b - a <= 6


def test_short_and_prolonged_limits():
    assert 4.0 <= M.PROLONGED <= 6.0 and M.SHORT == 1.0
    assert M.FOG_INTERVAL == 120.0 and M.BELL_INTERVAL == 60.0


def test_interval_never_passes_the_rule():
    one = M.fog_signal(sr=SR, signal="anchored")
    two = M.fog_signal(sr=SR, signal="anchored", repeats=2, interval=200.0)
    assert abs(len(two) - (60.0 * SR + len(one))) < 2
    spaced = M.fog_signal(sr=SR, repeats=2, interval=10.0)
    assert abs(len(spaced) - (10.0 * SR + len(M.fog_signal(sr=SR)))) < 2


def test_anchored_is_the_bell():
    a = M.fog_signal(sr=SR, signal="anchored")
    b = M.ship_bell(sr=SR)
    k = SR // 10                                                # past the two renders' different fades
    assert np.allclose(a[k:len(b) - k] / np.max(np.abs(a)), b[k:-k] / np.max(np.abs(b)), atol=0.05)


# bell [COL], [R35]
def test_bell_strokes_for_5_s_at_110_db():
    y = M.ship_bell(sr=SR)
    assert abs(20 * np.log10(M._bell_rms(y)) + M.HORN_REF_DB - 110.0) < 0.5
    k = int(0.005 * SR)
    e = np.sqrt(np.convolve(y * y, np.ones(k) / k, mode="same"))
    hop = e[int(0.1 * SR):int(5.0 * SR)]
    hop = hop - np.mean(hop)
    ac = np.fft.irfft(np.abs(np.fft.rfft(hop, 2 * len(hop))) ** 2)
    T = 1 / M.BELL_RATE
    a, b = int(0.7 * T * SR), int(1.3 * T * SR)
    assert abs((a + np.argmax(ac[a:b])) / SR - T) < 0.03
    last = 5.0 - 1 / M.BELL_RATE
    assert np.max(e[int((last - 0.05) * SR):int((last + 0.05) * SR)]) > np.max(e[int(5.3 * SR):int(5.4 * SR)])


def test_bell_is_33_db_under_the_big_horn():
    h = M.ship_horn(sr=SR, length=250.0)[SR:4 * SR]
    assert abs(20 * np.log10(M.horn_band_rms(h)) - 20 * np.log10(M._bell_rms(M.ship_bell(sr=SR))) - 33.0) < 0.5


# outboard
@pytest.mark.parametrize("strokes", [4, 2])
def test_outboard_275_hz(strokes):
    y = M.outboard(sr=SR, strokes=strokes, rpm=5500.0, speed=3.0)
    assert abs(_peak(y, 230, 320) - 275.0) < 4.0


def test_outboard_exhaust_is_low_passed_by_the_water():
    from genny.cyber import pipe_engine
    y = M.outboard(sr=SR, speed=1.0)
    z = pipe_engine(sr=SR, dur=4.0, rpm=np.full(4 * SR, 5500.0), cylinders=6, load=0.7, cycle=4, unequal=0.0)
    hi = lambda x: np.std(M._band(x, 3000.0, 12000.0)) / np.std(x)
    assert hi(y) < 0.5 * hi(z)


def test_outboard_under_water_34_to_71_law_and_band():
    assert abs(M.boat_level(3.0) - 134.0) < 1e-9 and abs(M.boat_level(16.0) - 171.0) < 1e-9
    a, b = M.outboard(sr=SR, speed=3.0, listen="under"), M.outboard(sr=SR, speed=16.0, listen="under")
    assert abs(_db(b) - _db(a) - 37.0) < 0.1
    f, p = _psd(b)
    assert p[(f >= 70) & (f <= 400)].sum() / p.sum() > 0.4      # the plateau holds the most; with 20 dB skirts 44 % in theory
    assert 70 <= _peak(b, 20, 3000) <= 400


def test_outboard_under_water_skirts_20_db_per_decade():
    y = M.outboard(sr=SR, speed=10.0, listen="under", dur=4.0)
    assert abs(_level(y, 4000.0, 0.0) - _level(y, 400.0 * 1.15, 0.0) + 20.0 * np.log10(4000.0 / 460.0)) < 2.5
    assert abs(_level(y, 30.0, 0.0, 0.2) - _level(y, 70.0, 0.0, 0.2) + 20.0 * np.log10(70.0 / 30.0)) < 2.5


def test_planing_blows_at_1_to_3_per_second():
    a = M.outboard(sr=SR, planing=1, dur=8.0, speed=10.0)
    b = M.outboard(sr=SR, planing=0, dur=8.0, speed=10.0)
    r = a - b * (a @ b) / (b @ b)                               # the same seed: what planing added, taken out of the render
    k = SR // 100
    e = np.sqrt(np.convolve(r * r, np.ones(k) / k, mode="same"))
    th = 0.25 * e.max()
    rises = np.flatnonzero((e[1:] > th) & (e[:-1] <= th))
    n = 1 + np.count_nonzero(np.diff(rises) > 0.15 * SR) if len(rises) else 0
    assert 1 * 8 <= n <= 3 * 8
    assert np.max(np.abs(r)) > 0.5 * np.max(np.abs(a))          # sparse blows: judged by their peak, they stand out of the engine


# jet ski
def test_jet_ski_175_hz_and_impeller_350():
    y = M.jet_ski(sr=SR)
    assert abs(_peak(y, 150, 200) - 175.0) < 3 and abs(_peak(y, 320, 380) - 350.0) < 3


def test_jump_free_revs_stops_the_jet_and_lands():
    dur = 4.0
    y = M.jet_ski(sr=SR, jump=1, dur=dur)
    a, b = int(0.42 * dur * SR), int(0.58 * dur * SR)
    assert abs(_peak(y[a:b], 150, 260) - 175 * M.JET_FREE / M.JET_RPM) < 5
    jet = lambda w: np.std(M._band(w, 2000.0, 8000.0)) / np.std(w)
    assert jet(y[a:b]) < 0.7 * jet(y[:int(0.3 * dur * SR)])
    land = int(0.65 * dur * SR)
    assert np.max(np.abs(y[land:land + SR // 4])) > 1.2 * np.max(np.abs(y[:int(0.3 * dur * SR)]))


def test_jet_ski_under_water_percentiles_and_band():
    y = M.jet_ski(sr=SR, listen="under", dur=20.0)
    h = SR // 8
    e = np.array([_db(y[i:i + h]) for i in range(0, len(y) - h, h)]) + M.BOAT_REF_DB
    lo, mid, hi = np.percentile(e, [5, 50, 95])
    assert abs(lo - 122) < 2.0 and abs(mid - 137) < 1.5 and abs(hi - 149) < 1.5
    f, p = _psd(y)
    assert p[(f >= 100) & (f <= 10000)].sum() / p.sum() > 0.95
    assert abs(_peak(y, 320, 380) - 350) < 3


# sailboat
def test_shroud_sings_at_400_hz():
    y = M.sailboat(sr=SR)
    assert abs(_peak(y, 250, 600) - 400.0) < 0.05 * 400
    z = M.sailboat(sr=SR, wind=5.0)
    assert abs(_peak(z, 120, 300) - 200.0) < 0.1 * 200


def test_sailboat_has_flap_and_halyard_layers():
    a, b = M.sailboat(sr=SR, flap=2.0, halyard=1.0, dur=6.0), M.sailboat(sr=SR, flap=4.0, halyard=5.0, dur=6.0)
    hi = lambda y: np.max(np.abs(M._band(y, 2500.0, 12000.0))) / np.std(y)
    k = SR // 50
    e = lambda y: np.sqrt(np.convolve(M._band(y, 2500.0, 12000.0) ** 2, np.ones(k) / k, mode="same"))
    th = lambda y: np.count_nonzero(np.diff((e(y) > 0.3 * e(y).max()).astype(int)) == 1)
    assert th(b) > th(a)


# submarine
@pytest.mark.parametrize("kind,db", [("electric", 100.0), ("nuclear", 110.0)])
def test_submarine_level_at_1_khz_4_kn(kind, db):
    y = M.submarine(sr=SR, kind=kind, speed=4.0, dur=30.0)
    assert abs(_level(y, 1000.0, M.SUB_REF_DB) - db) < 1.0


def test_machines_below_8_kn_flow_above():
    level = lambda v: [_db(x) for x in M._sub_outside("electric", v, 4 * SR, np.random.default_rng(0))]
    for v, machines_lead in ((4.0, True), (6.0, True), (10.0, False), (16.0, False)):
        a, b = level(v)
        assert (a > b) == machines_lead
    a, b = level(8.0)
    assert abs(a - b) < 0.1


def test_submarine_blade_rate_and_no_cavitation():
    y = M.submarine(sr=SR, speed=16.0, dur=20.0)
    assert abs(_peak(y, 10, 18) - 60 * 2 * 7 / 60) < 0.2
    x = M._band(y, 800.0, 3000.0)
    e = np.abs(x) - np.abs(x).mean()
    f, p = welch(e, SR, nperseg=2 ** 16)
    m = (f > 5) & (f < 60)
    assert np.max(p[m]) < 8 * np.median(p[m])     # the noise is not chopped at the blade rate, unlike `ship cavitation=1`


def test_submarine_inside_has_its_events():
    dur = 10.0
    y = M.submarine(sr=SR, view="in", dur=dur)
    assert abs(_peak(y[int(0.1 * dur * SR):int(0.1 * dur * SR) + SR // 2], 1000, 2200) - 1500.0) < 8    # the ping
    assert np.std(y[int(0.66 * dur * SR):int(0.8 * dur * SR)]) > 1.2 * np.std(y[int(0.25 * dur * SR):int(0.35 * dur * SR)])     # the ballast over the machines
    assert abs(_peak(y[int(0.3 * dur * SR):int(0.38 * dur * SR)], 30, 70) - 50.0) < 2                  # the machines alone


# diaphone
def test_diaphone_note_and_final_grunt_an_octave_down():
    y = M.foghorn(sr=SR, freq=240.0, dur=4.0, grunt=0.8)
    assert abs(_peak(y[SR:2 * SR], 180, 300) - 240.0) < 3
    assert abs(_peak(y[-int(0.3 * SR):], 100, 260) - 120.0) < 25
    assert np.std(y[-int(0.3 * SR):]) < 0.7 * np.std(y[SR:2 * SR])          # the pressure has fallen


# lapping
@pytest.mark.parametrize("kind", ["clapot", "quai", "baignade"])
def test_lap_beats_in_1_to_3_s(kind):
    y = M.lap(sr=SR, kind=kind, dur=30.0, seed=4)
    k = SR // 10                                                # the random spacing has no autocorrelation peak of its own: read the intervals
    e = np.sqrt(np.convolve(y * y, np.ones(k) / k, mode="same"))
    th = 0.35 * np.quantile(e, 0.98)
    rises = np.flatnonzero((e[1:] > th) & (e[:-1] <= th)) / SR
    gaps = np.diff(rises[np.r_[True, np.diff(rises) > 0.5]])
    assert len(gaps) >= 8 and 1.0 <= np.median(gaps) <= 3.0


def test_lap_kinds_differ_as_described():
    c = {k: M.lap(sr=SR, kind=k, dur=10.0) for k in ("clapot", "quai", "baignade")}
    low = lambda y: np.std(M._band(y, 60.0, 350.0)) / np.std(y)
    assert low(c["clapot"]) > low(c["quai"]) and low(c["clapot"]) > low(c["baignade"])      # the hull


def test_wittekind_reference_ship_fig5():
    # 33000 t, cB .78, Vc 9 kn, 14 kn: low-frequency peak ~188 dB at 40 Hz, high-frequency cavitation 163.5 dB at 1 kHz
    sl1 = M.wittekind_sl1(40.0) + M.wittekind_a(14.0, 9.0, 0.78) + M.wittekind_b(33000.0)
    assert abs(sl1 - 188.0) < 1.0
    assert abs(float(M.wittekind_sl2(1000.0, 14.0, 9.0, 33000.0)) - 163.5) < 0.01
    assert abs(float(M.wittekind_sl2(1000.0, 28.0, 9.0, 33000.0)) - 163.5 - 60 * np.log10(2)) < 0.01
    # density: third-octave level over 0.23 f
    d = float(M.wittekind_density(1000.0, 14.0, 9.0, 0.78, 33000.0))
    assert abs(d - (163.5 - 10 * np.log10(231.6))) < 0.5


def test_randi_table2_merchant():
    # average merchant: 337.5 ft, 12.5 kn -> 153.5 dB at 100 Hz, 137.1 at 300 Hz (RANDI 3.1 Table 2)
    L = 337.5 * 0.3048
    assert abs(float(M.randi(100.0, 12.5, L)) - 153.5) < 0.2
    assert abs(float(M.randi(300.0, 12.5, L)) - 137.1) < 0.2
    assert abs(float(M.ship_level(100.0, "randi", 12.5, length=L)) - 153.5) < 0.2
