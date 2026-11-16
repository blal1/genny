"""Lot BR3: formulas wired to their consumers (opt-in, defaults unchanged) and public utilities checked by call.

Run:  uv run python tests/test_lot_br3.py
"""
import math

import numpy as np

from genny import acoustics, creatures, cyber, engines, heavy_vehicles, tubes
from genny.physical import voice, winds

SR = 44100


def _same(a, b):
    return a.shape == b.shape and np.array_equal(a, b)


def _centroid(y, sr=SR):
    s = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(y), 1 / sr)
    return float((s * f).sum() / s.sum())


def _band_share(y, lo, hi, sr=SR):
    s = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / sr)
    return float(s[(f >= lo) & (f < hi)].sum() / s.sum())


def _peak_hz(y, lo, hi, sr=SR):
    s = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / sr)
    m = (f >= lo) & (f < hi)
    return float(f[m][np.argmax(s[m])])


def _ok(y):
    assert np.all(np.isfinite(y)) and np.max(np.abs(y)) > 0


def test_engine_cold_idle():
    for kind in ("dfco_overrun", "engine_off", "free_rev"):
        a = engines.engine_event(sr=SR, kind=kind, seed=1, dur=1.0)
        assert _same(a, engines.engine_event(sr=SR, kind=kind, seed=1, dur=1.0, temperature=0.0))
        c = engines.engine_event(sr=SR, kind=kind, seed=1, dur=1.0, temperature=265.0)
        _ok(c)
        assert not _same(a, c)
    assert abs(engines.cold_idle(800, 285)[0] - 1240) < 1e-6
    assert engines.cold_idle(800, 285)[1] == engines.cold_idle(800, 285)[0] + 350.0


def test_pipe_engine_exhaust_formulas():
    kw = dict(sr=SR, dur=0.6, rpm=3000.0, seed=2)
    base = cyber.pipe_engine(**kw)
    assert _same(base, cyber.pipe_engine(**kw, branch=None, bottle=None, expansion_ir=0))
    assert abs(cyber.side_branch_freq(0.1) - 857.5) < 1.0
    br = cyber.pipe_engine(**kw, branch="100,0,0", gas_temp=293.15)
    base20 = cyber.pipe_engine(**kw, gas_temp=293.15)
    assert _band_share(br, 800, 920) < 0.8 * _band_share(base20, 800, 920)
    c = cyber.gas_speed(600.0)
    neck = cyber.helmholtz_neck(110.0, 3e-3, 1e-3, c=c)
    bo = cyber.pipe_engine(**kw, bottle=f"3,{neck * 1e3},10")
    assert _band_share(bo, 95, 125) < 0.95 * _band_share(base, 95, 125)
    e0 = cyber.pipe_engine(**kw, expansion=4.0, expansion_ms=1.0)
    e1 = cyber.pipe_engine(**kw, expansion=4.0, expansion_ms=1.0, expansion_ir=5)
    _ok(e1)
    assert not _same(e0, e1)
    assert _same(base, cyber.pipe_engine(**kw, expansion_ir=5))
    h = cyber.expansion_chamber_ir(2.0, 5)
    assert abs(h[0] + 1.0 / 3.0) < 1e-12 and abs(h[1] + 8.0 / 9.0) < 1e-12


def test_creatures_species_stride_formants_range():
    a = creatures.footsteps(sr=SR, gait="gallop", steps=16, speed=10.0, seed=1)
    assert _same(a, creatures.footsteps(sr=SR, gait="gallop", steps=16, speed=10.0, seed=1, species=""))
    b = creatures.footsteps(sr=SR, gait="gallop", steps=16, speed=10.0, seed=1, species="horse")
    ref = creatures.footsteps(sr=SR, gait="gallop", steps=16, pace=240.0 * creatures.stride_hz("horse", 10.0), seed=1)
    assert _same(b, ref) and not _same(a, b)
    base = creatures.roar(sr=SR, seed=3, dur=1.0)
    assert _same(base, creatures.roar(sr=SR, seed=3, dur=1.0, formants=""))
    df = 350.0 / (2.0 * 0.29)
    fm = ",".join(str((2 * n - 1) / 2.0 * df) for n in (1, 2, 3))
    assert abs(creatures.formant_spacing([float(x) for x in fm.split(",")]) - df) < 1e-6
    tiger = creatures.roar(sr=SR, seed=3, dur=1.0, species="tiger")
    assert np.allclose(creatures.roar(sr=SR, seed=3, dur=1.0, formants=fm), tiger, atol=1e-6)
    s0 = creatures.birdsong(sr=SR, species="robin", seed=1)
    assert _same(s0, creatures.birdsong(sr=SR, species="robin", seed=1, centre=0.0))
    lo, hi = creatures.bird_range("robin")
    s1 = creatures.birdsong(sr=SR, species="robin", seed=1, centre=1.4 * math.sqrt(lo * hi))
    _ok(s1)
    assert _centroid(s1) > 1.15 * _centroid(s0)


def test_heavy_vehicles_pump_rate_and_gravel():
    a = heavy_vehicles.hydraulic_pump(sr=SR, seed=1, dur=0.6)
    assert _same(a, heavy_vehicles.hydraulic_pump(sr=SR, seed=1, dur=0.6, rate=0.0))
    r = heavy_vehicles.pump_rate(2000.0, 9)
    assert r == 300.0 and _same(a, heavy_vehicles.hydraulic_pump(sr=SR, seed=1, dur=0.6, rate=r))
    t0 = heavy_vehicles.tracks(sr=SR, dur=1.0, seed=1)
    assert _same(t0, heavy_vehicles.tracks(sr=SR, dur=1.0, seed=1, gravel=0.0))
    t1 = heavy_vehicles.tracks(sr=SR, dur=1.0, seed=1, gravel=1.0)
    _ok(t1)
    assert _band_share(t1, 5500, 7500) > 1.5 * _band_share(t0, 5500, 7500)


def test_tubes_bend_and_impedance():
    k = 0.5
    corr = tubes.bend_length_correction(0.5, k)
    assert corr < 0.0 and abs(corr - 0.5 * (math.sqrt(tubes.bend_inertance(k)) - 1.0)) < 1e-12
    a = tubes.reed_tube(220.0, 0.5, sr=SR, seed=1)
    assert _same(a, tubes.reed_tube(220.0, 0.5, sr=SR, seed=1, bend=0.0))
    b = tubes.reed_tube(220.0, 0.5, sr=SR, seed=1, bend=k, bend_share=0.5)
    _ok(b)
    assert _peak_hz(b, 150, 300) > _peak_hz(a, 150, 300)
    x = np.random.default_rng(0).standard_normal(SR)
    t0 = tubes.tube(x, sr=SR, length_m=1.0)
    assert _same(t0, tubes.tube(x, sr=SR, length_m=1.0, bend=0.0, bend_m=0.0, model="fd"))
    assert not _same(t0, tubes.tube(x, sr=SR, length_m=1.0, bend=k, bend_m=0.5))
    z = tubes.tube(x, sr=SR, length_m=0.5, radius_m=0.01, model="impedance", mix=1.0)
    _ok(z)
    f = _peak_hz(z, 100, 300)
    assert abs(f - 343.0 / (4.0 * (0.5 + 0.6133 * 0.01))) < 0.06 * f
    zr = tubes.causse_zr(np.array([0.05, 0.5]))
    assert zr[0].real < zr[1].real and zr[0].imag > 0.0
    assert tubes.reed_tube(220.0, 0.4, sr=SR, seed=1, reed="sax_alto").shape[0] > 0
    assert _same(tubes.reed_tube(220.0, 0.4, sr=SR, seed=1), tubes.reed_tube(220.0, 0.4, sr=SR, seed=1, rest=-1.0))
    assert not _same(tubes.reed_tube(220.0, 0.4, sr=SR, seed=1), tubes.reed_tube(220.0, 0.4, sr=SR, seed=1, rest=0.0))


def _vowel(n, f0=130.0):
    from scipy.signal import lfilter
    t = np.arange(n)
    e = (np.diff(np.floor(t * f0 / voice.SR), prepend=0) > 0).astype(float)
    y = e
    for fc, bw in ((700.0, 90.0), (1200.0, 110.0), (2600.0, 160.0)):
        r = math.exp(-math.pi * bw / voice.SR)
        th = 2 * math.pi * fc / voice.SR
        y = lfilter([1.0 - r], [1.0, -2.0 * r * math.cos(th), r * r], y)
    return y / np.max(np.abs(y))


def test_hybrid_voice_segment_entry():
    n = int(0.8 * voice.SR)
    seg = _vowel(n)
    m = int(0.5 * voice.SR)
    y = voice.hybrid_voice(m, np.random.default_rng(0), family="none", queries=(), pick=0.0, f0=np.full(m, 200.0), segment=seg)
    assert y.shape == (m,) and np.all(np.isfinite(y)) and abs(np.max(np.abs(y)) - 1.0) < 1e-9
    from genny.physical import analysis
    pair = analysis.analyse(seg)
    y2 = voice.hybrid_voice(m, np.random.default_rng(0), family="none", queries=(), pick=0.0, f0=np.full(m, 200.0), segment=pair)
    assert np.allclose(y, y2)
    used = []
    voice.hybrid_voice(m, np.random.default_rng(0), family="none", queries=(), pick=0.0, segment=pair, used=used)
    assert used == ["segment"]


def test_tmm_bore_and_hole_models():
    f = np.arange(100.0, 1500.0, 2.0)
    holes = [(0.3, 0.004, 0.003)]
    k0 = winds.tmm_impedance(0.5, 0.01, holes, f)
    assert np.array_equal(k0, winds.tmm_impedance(0.5, 0.01, holes, f, bore="keefe", hole_model="keefe", v0=0.0))
    fr = lambda z: float(f[np.argmax(np.abs(z))])    # noqa: E731
    v = winds.tmm_impedance(0.5, 0.01, holes, f, bore="visco")
    assert abs(fr(v) - fr(k0)) < 0.03 * fr(k0)
    d = winds.tmm_impedance(0.5, 0.01, holes, f, hole_model="dubos")
    assert abs(fr(d) - fr(k0)) < 0.05 * fr(k0) and not np.allclose(d, k0)
    n = winds.tmm_impedance(0.5, 0.01, holes, f, v0=20.0)
    assert np.max(np.abs(n)) < np.max(np.abs(k0))
    assert winds.end_nonlinear_impedance(20.0, 1e6) > 0.0
    ra, rh = winds.hole_nonlinear_resistance(20.0, 0.004, 0.01)
    assert ra > 0.0 and rh > 0.0
    assert 330.0 < float(winds.phase_velocity(500.0, 0.007)) < 343.0 and float(winds.visco_alpha(1000.0, 0.007)) > 0.0


def test_brass_shock_and_lips():
    t = np.arange(SR // 2) / SR
    y = np.sin(2 * np.pi * 440.0 * t)
    s = winds.brass_shock(y, "trumpet", 20000.0, SR)
    assert np.max(np.abs(np.diff(s))) > 1.05 * np.max(np.abs(np.diff(y)))
    assert np.array_equal(winds.brass_shock(y, "trumpet", 0.0, SR), y)
    assert winds.shock_delay(1000.0, 0.6, 1.5) < 0.0 and winds.local_speed(1.0) > 343.0
    p = winds.lip_params(0.5, "tight")
    assert p["fL1"] == 222.0 and p["b"] == 11e-3 and p["H0"] == 0.44e-3 and p["fL2"] == 259.5
    assert winds.lip_params(0.5, "medium")["fL2"] == winds.lip_params(0.5)["fL2"]
    assert winds.lip_params(0.3) == winds.lip_params(0.3, "")
    base = winds.trombone_lip(233.0, 0.4)
    assert np.array_equal(base, winds.trombone_lip(233.0, 0.4, lips="", lip_q="", hysteresis=False, shock=0.0))
    for kw in (dict(lips="tight"), dict(lip_q="glycerine"), dict(hysteresis=True), dict(shock=20000.0)):
        z = winds.trombone_lip(233.0, 0.4, **kw)
        _ok(z)
        assert z.shape == base.shape
    assert not np.array_equal(base, winds.trombone_lip(233.0, 0.4, lip_q="glycerine"))
    assert not np.array_equal(base, winds.trombone_lip(233.0, 0.4, shock=20000.0))
    tb = winds.trumpet_lip2d(466.0, 0.3)
    assert np.array_equal(tb, winds.trumpet_lip2d(466.0, 0.3, shock=0.0))
    assert not np.array_equal(tb, winds.trumpet_lip2d(466.0, 0.3, shock=20000.0))
    on = winds.lip_sounding(np.array([0.0, 120.0, 100.0, 96.0, 90.0]), 100.0, 0.95)
    assert on.tolist() == [False, True, True, True, False]


def test_registers_recorder_modal():
    sr = 1000
    ramp = 200.0 + 100.0 * np.arange(3 * sr) / sr
    reg = winds.register_track(ramp, sr)
    i = int(np.argmax(reg == 2))
    thr = winds.dynamic_threshold(100.0, winds.REGISTER_G4["p12"])
    assert abs(ramp[i] - thr) < 1.5
    g = winds.register_track(ramp, sr, fingering="G4")
    assert np.array_equal(g, reg)
    f4 = winds.register_track(ramp, sr, fingering="F4")
    assert np.array_equal(f4, winds.register_track(ramp, sr, winds.REGISTER_F4["p12"], winds.REGISTER_F4["p21"], 0.85))
    assert int(np.argmax(f4 == 2)) < i
    a = winds.auvray_recorder(440.0, 0.15, seed=1)
    assert np.array_equal(a, winds.auvray_recorder(440.0, 0.15, seed=1, recorder="zenon", trajectory=0.0))
    for kw in (dict(recorder="yrt304"), dict(trajectory=2000.0)):
        b = winds.auvray_recorder(440.0, 0.15, seed=1, **kw)
        _ok(b)
        assert b.shape == a.shape and not np.array_equal(a, b)
    base = winds.modal_clarinet(207.0, 0.4)
    assert np.array_equal(base, winds.modal_clarinet(207.0, 0.4, peaks=None))
    sx = winds.modal_clarinet(207.0, 0.4, peaks=winds.SAX_SOPRANO_PEAKS[:4], n_modes=4)
    _ok(sx)
    assert not np.array_equal(base, sx)
    pk = _peak_hz(sx[int(0.1 * SR):], 100, 1500)
    assert min(abs(pk / q - 1.0) for q in winds.SAX_SOPRANO_PEAKS[:4]) < 0.1
    al = winds.SAX_ALTO_REED
    assert abs(al["omega0"] - 2 * math.pi * winds.SAX_ALTO["fr"]) < 1e-9 and al["Mr"] == 0.0475 * 1.77e-4


def test_winds_exposed_utilities():
    assert winds.series_length_ls(3e-3, 7e-3, 3e-3) < 0.0 and winds.series_length_ls(3e-3, 7e-3, 3e-3, open_=False) < 0.0
    assert winds.key_correction(5e-3, 2e-3, 3e-3, 8e-3) > 0.0
    assert winds.key_correction(5e-3, 1e-4, 3e-3, 8e-3) > winds.key_correction(5e-3, 4e-3, 3e-3, 8e-3)
    assert winds.closed_hole_length(3e-3, 7e-3, 3e-3) > winds.closed_hole_length(3e-3, 7e-3, 3e-3, finger_radius=8e-3)
    z = winds.open_hole_shunt(1j * 2 * math.pi * 500.0, 0.5, 0.5, 1e-4, 0.5, 1.0, 0.1)
    assert np.isfinite(z)
    assert winds.cents(5.78e-4) == 1.0
    ih = [winds.ih_register_hole(1e-5, 1e-4, 0.1, 1e-2, 5.0), winds.ih_open_hole(3e-3, 7e-3, 5e-3, 0.2, 5.0)[1],
          winds.ih_closed_hole(1e-6, 1e-4, 1e-3, 5.0, 0.3), winds.ih_section(1.2, 5.0, 0.3), winds.ih_cone(0.1, 0.05, 5.0, 0.3),
          winds.ih_dispersion(0.9, 0.8), winds.ih_temperature(300.0, 295.0)]
    assert all(math.isfinite(v) for v in ih) and winds.ih_temperature(300.0, 295.0) < 0.0
    assert math.isfinite(winds.bell_length_correction(5.0, 0.05)) and winds.bell_length_correction(5.0, 0.05) != 0.0
    hs, cost = winds.optimise_holes(0.5, 0.01, [(0.3, 0.004, 0.003)], [400.0], fingerings=[[True]])
    assert 0.05 * 0.5 <= hs[0][0] <= 0.98 * 0.5 and cost >= 0.0
    f = np.array([100.0, 200.0, 300.0])
    zb = winds.bore_mode_impedance(f, 200.0, 20.0, 5e6)
    assert abs(zb[1] - 5e6) < 1e-6 and np.all(np.abs(zb) <= 5e6 + 1e-6)
    assert abs(winds.lip_response(0.0, 200.0, 10.0, 0.05) - 0.05 / (2 * math.pi * 200.0) ** 2) < 1e-12
    assert set(winds.LIP_RANGES) == {"cullen", "elliot_bowsher", "saneyoshi"} and set(winds.LIP_TABLE_1) == {"slack", "medium", "tight"}
    assert winds.LIP_Q_HUMAN[0] < winds.LIP_Q_GLYCERINE[0] < winds.LIP_Q_LATEX[0]
    fr = np.linspace(500.0, 3000.0, 2501)
    d = np.abs(winds.reed_response(fr, 0.0475, 1500.0, 0.05 * 2 * math.pi * 1500.0))
    assert abs(fr[np.argmax(d)] - 1500.0) < 40.0
    assert abs(float(winds.reed_recovery(0.0)) - 0.93) < 1e-12 and float(winds.reed_recovery(1e6)) > 0.999 and winds.REED_HYSTERESIS == 0.04
    rng = np.random.default_rng(0)
    q, dp, dpdt = rng.uniform(0.5, 1.0, 400), rng.uniform(-200.0, 400.0, 400), rng.standard_normal(400) * 1e3
    u = 1e-6 * q * np.abs(dp) ** 1.5 + 2e-4 * q * np.abs(dp) ** 0.5 - 3e-9 * dpdt
    c = winds.reed_flow_fit(q, dp, dpdt, u)
    assert np.allclose(c, (1e-6, 2e-4, -3e-9), rtol=1e-6)
    assert len(winds.SAX_ALTO_HARMONICS) == 4 and len(winds.SAX_SOPRANO_PEAKS) == 9 and winds.SAX_GEOMETRY["tenor"]["L"] == 1.49
    assert winds.MOUTHPIECE_FACING["length"] == 19.0e-3 and winds.SOPRANO_CONE_ANGLE == 0.07
    assert winds.REED_ALONE["dry"][0] > winds.REED_ALONE["wet"][0] and winds.RECORDER_YRT304["W"] == 4.83e-3
    assert winds.JET_STROUHAL["W_over_h_opt"] == 8.0 and winds.CONVECTION_RANGE == (0.4, 0.92)
    assert abs(winds.strouhal_h(0.2) - 0.03) < 1e-12 and abs(winds.jet_delay(4e-3, 10.0) - 1e-3) < 1e-12
    jt = winds.jet_transfer(2 * math.pi * np.array([200.0, 400.0]), 20.0, 4e-3)
    assert jt.shape == (2,) and np.all(np.isfinite(jt))
    assert abs(winds.bickley_flux(1e-3, 20.0, 0.012) - 2.0 / 3.0 * 1e-3 * 20.0 * 0.012) < 1e-15
    assert winds.even_odd_db(0.0) == -1.0 and winds.even_odd_db(1.2) == 28.0
    p = np.sin(np.linspace(0.0, 200.0, 4000))
    assert abs(winds.radiated_power(p, 1.0) - 0.5 * 4 * math.pi / (1.2 * 343.0)) < 0.01
    assert abs(winds.source_power(1e-4, 5.0) - 1e-8 * 1.2 * 343.0 * 25.0 / (8 * math.pi)) < 1e-12
    assert winds.source_power(1e-4, 5.0, "dipole", 0.01) < winds.source_power(1e-4, 5.0)
    assert winds.pipe_tone(500.0, [200.0, 520.0, 800.0]) == 520.0
    assert winds.REGISTER_F4["p12"] == 294.3 and winds.REGISTER_EFFECTS["flutist_hysteresis"] == (1.69, 3.80)
    assert abs(winds.cubic_flow(0.35, 0.39)[0] - 0.35 * (3 * 0.39 - 1) / (2 * math.sqrt(0.39))) < 1e-12
    assert winds.vibrato_am("trombone", 1000.0) == 9.0 and winds.vibrato_am("flute", 100.0) == 0.0
    assert abs(winds.lip_volume_freq(9.0) - 500.0) < 1e-9 and winds.lip_volume_freq(12.0) < 470.0
    lo, hi = winds.lip_two_resonance(370.0)
    assert 96.0 < lo < 97.0 and 111.0 < hi < 117.0
    assert winds.channel_flow(100.0, 1e-2, 1e-6, 1e-2) < winds.channel_flow(100.0, 1e-2, 1e-3, 1e-2)
    assert winds.phase_velocity(1000.0, 0.007) > 0.0


def test_room_absorber_specs():
    from genny.acoustics import Room, absorber
    r0 = Room(volume=200.0).t60_bands()
    for spec in (absorber("layer", 30.0, sigma=20000.0, thickness=0.05), absorber("helmholtz", f0=200.0, n=20),
                 absorber("panel", 20.0, mass=5.0, depth_cm=10.0)):
        t = Room(volume=200.0, absorbers=[spec]).t60_bands()
        assert np.all(t <= r0 + 1e-12) and np.any(t < r0 - 1e-6)
    assert acoustics.Room(volume=200.0, absorbers=[]).t60_bands().tolist() == r0.tolist()
    try:
        absorber("none")
    except ValueError:
        pass
    else:
        raise AssertionError


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
