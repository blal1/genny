"""Lot BR2: orphan formulas wired to their consumers (opt-in, defaults unchanged) and public utilities checked by call.

Run:  uv run python tests/test_lot_br2.py
"""
import math

import numpy as np

from genny import acoustics, machines, matter, nature, rail, sports
from genny.physical import space

SR = 44100


def _centroid(y, sr=SR):
    s = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(y), 1 / sr)
    return float((s * f).sum() / s.sum())


def test_rail_creep_noise_follow():
    v = 10.0
    base = rail.creep_noise(0.002, v, 24000, np.random.default_rng(0))
    same = rail.creep_noise(0.002, v, 24000, np.random.default_rng(0), follow=False)
    assert np.array_equal(base, same)
    fol = rail.creep_noise(0.002, v, 24000, np.random.default_rng(0), follow=True)
    assert fol.shape == base.shape and not np.array_equal(fol, base)
    assert rail.creep_cutoff(v)[0] == v / (2 * rail.PATCH_A)


def test_rail_halfspace_support():
    f_h, zeta = rail.halfspace_resonance(30.0, rail.PATCH_A)
    assert abs(rail.contact_resonance(model="halfspace") - f_h) < 1e-9
    a = rail.corrugated_contact(300.0, 0.2, 4800)
    b = rail.corrugated_contact(300.0, 0.2, 4800, support="halfspace")
    assert a.shape == b.shape and not np.allclose(a, b)
    assert np.array_equal(a, rail.corrugated_contact(300.0, 0.2, 4800, support="hertz"))


def test_rail_straight_bed():
    a = rail.curve_squeal(sr=SR, dur=0.6, seed=1)
    b = rail.curve_squeal(sr=SR, dur=0.6, seed=1, straight=0.0)
    c = rail.curve_squeal(sr=SR, dur=0.6, seed=1, straight=1.0)
    assert np.array_equal(a, b) and a.shape == c.shape and not np.array_equal(a, c)
    assert rail.STRAIGHT_DBA == 69.0


def test_tornado_roughness_and_approach():
    base = nature.tornado(sr=SR, dur=1.5, seed=2, distance=2000.0)
    assert np.array_equal(base, nature.tornado(sr=SR, dur=1.5, seed=2, distance=2000.0, roughness=0.0, approach=0.0))
    rough = nature.tornado(sr=SR, dur=1.5, seed=2, distance=2000.0, roughness=0.03)
    assert _centroid(rough) > _centroid(base)
    assert _centroid(nature.tornado(sr=SR, dur=1.5, seed=2, distance=2000.0, approach=40.0)) > _centroid(base)
    assert math.isclose(nature.tornado_peak_hz(67.5, 0.03), 150.0) and nature.doppler_shift(100.0, 34.3) > 110.0


def test_ice_cubes_slosh():
    a = matter.ice_cubes(sr=SR, seed=3)
    assert np.array_equal(a, matter.ice_cubes(sr=SR, seed=3, slosh=0.0))
    assert not np.array_equal(a, matter.ice_cubes(sr=SR, seed=3, slosh=1.0))
    assert 3.0 < matter.slosh_hz(0.0175, 0.042) < 8.0


def test_ball_bounce_modes():
    a = sports.ball_bounce(sr=SR, seed=1)
    assert np.array_equal(a, sports.ball_bounce(sr=SR, seed=1, modes=2))
    assert not np.array_equal(a, sports.ball_bounce(sr=SR, seed=1, modes=6))
    m = sports.sphere_cavity_modes(0.12, 4)
    assert abs(m[0][2] - 2.082) < 0.01 and abs(m[1][2] - 3.342) < 0.01


def test_conveyor_oiled():
    a = machines.conveyor(sr=SR, dur=0.6, seed=1)
    assert np.array_equal(a, machines.conveyor(sr=SR, dur=0.6, seed=1, oiled=0.0))
    assert not np.array_equal(a, machines.conveyor(sr=SR, dur=0.6, seed=1, oiled=1.0))
    assert 0.005 < float(machines.gear_friction_coefficient(0.3, 1.0, 5.0, 30.0)) < 0.06


def test_room_absorbers_shorten_t60():
    r = acoustics.Room()
    t0 = r.t60_bands()
    assert np.array_equal(t0, acoustics.Room(absorbers=[]).t60_bands())
    r.absorbers.append((10.0, lambda f: space.rayleigh_absorber(f, 0.95, 20000.0, 0.05)))
    r.absorbers.append((5.0, lambda f: space.fabric_absorption(f, 1.0, 0.1)))
    assert np.all(r.t60_bands() < t0)


def test_recorder_regime():
    from genny.physical import winds
    a = winds.auvray_recorder(440.0, 0.1, sr=22050)
    assert np.array_equal(a, winds.auvray_recorder(440.0, 0.1, sr=22050, regime=0))
    b = winds.auvray_recorder(440.0, 0.1, sr=22050, regime=2)
    assert b.shape == a.shape and not np.array_equal(a, b)
    assert winds.optimal_jet_velocity(440.0, 4e-3, 1) < winds.optimal_jet_velocity(440.0, 4e-3, 0)
    assert winds.jet_regime(440.0, 4e-3, 10.0)["mode"] in ("jet-drive", "transition", "vortex")
    assert -2.0 < winds.even_odd_db(0.0) < 0.0 and winds.pipe_tone(430.0, [260.0, 440.0]) == 440.0


def test_locked_files_utilities_exist():
    from genny import creatures, cyber, engines, heavy_vehicles, tubes
    from genny.physical import voice
    for mod, names in ((engines, ["cold_idle"]), (cyber, ["side_branch_freq", "helmholtz_neck", "expansion_chamber_ir"]),
                        (creatures, ["stride_hz", "formant_spacing", "bird_range"]), (heavy_vehicles, ["pump_rate", "GRAVEL_HZ"]),
                        (tubes, ["bend_length_correction", "causse_zr"]), (voice, ["hybrid_voice", "farnell_syrinx"])):
        for nm in names:
            assert hasattr(mod, nm)
    assert len(engines.cold_idle(800.0, 253.0)) == 2


def test_public_utilities_import_and_call():
    from genny import marine
    assert abs(matter.minnaert(1e-3) - 3280.0) < 30.0
    assert abs(matter.plume_frequency(0.75) - marine.plume_f0(0.75)) < 1e-6
    assert 5.0 < sports.strain_freq(10.0, 0.005) < 3000.0
    assert acoustics.air_absorption_db_per_m([1000.0, 8000.0])[1] > acoustics.air_absorption_db_per_m([1000.0, 8000.0])[0]
    assert 0.0 < space.absorption_8k("plaster") <= 1.0
    assert space.perforated_mass(0.005, 0.002, 0.1) > 0.0
    assert len(space.qrd_lobes(7, 0.03, 2000.0)) >= 1
    assert space.scale_model([1000.0], 10.0)["f_model"][0] == 10000.0
    assert space.ambi_decode(np.ones((4, 4)), [(0, 0), (90, 0)]).shape == (4, 2)
    assert space.transaural_gain(30.0)["gain"] < 1.0
    assert 0.0 < machines.gear_mesh_efficiency([1.0], [1.0], [1.0], [1.0], 10.0, 100.0) < 1.0
    assert machines.gear_rolling_force(10.0, 1e5, 0.02, 2.3e11, 2e-8, 0.05) > 0.0


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
