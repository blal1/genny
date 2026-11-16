import math

import numpy as np

from genny import acoustic, contact, graphsolver, physics, sfx
from genny.physical import modal

SR = 44100
TON = 907.18
FT = 0.3048


def _clean(y):
    assert np.all(np.isfinite(y)) and np.max(np.abs(y)) > 1e-3


def _rms(y):
    return float(np.sqrt(np.mean(np.asarray(y, float) ** 2)))


def test_point_source_blast_reads_the_reference_table_and_tnt_is_unchanged():
    ps, td, al = physics.blast_wave(TON, 100 * FT, source="point")
    assert abs(ps / 101325.0 - 0.43) < 0.02
    assert abs(td * 1e3 - 18.8) < 0.5 and abs(al - 0.9) < 1e-9
    ps, td, al = physics.blast_wave(TON, 250 * FT, source="point")
    assert abs(td * 1e3 - 24.7) < 0.5 and abs(al - 1.4) < 1e-9 and abs(ps / 101325.0 - 0.110) < 0.005
    assert physics.blast_wave(1.0, 10.0) == physics.blast_wave(1.0, 10.0, source="tnt")
    assert physics.blast_wave(TON, 100 * FT, source="point")[0] < physics.blast_wave(TON, 100 * FT)[0]


def test_point_source_duration_scales_with_the_cube_root_of_the_charge():
    _, t1, _ = physics.blast_wave(1.0, 20.0, source="point")
    _, t8, _ = physics.blast_wave(8.0, 40.0, source="point")
    assert abs(t8 / t1 - 2.0) < 1e-9


def test_secondary_shock_only_inside_z_0_4_and_delay_follows_the_18_t_reading():
    assert physics.blast_secondary(1.0, 0.4) == (0.0, 0.0)
    assert physics.blast_secondary(1.0, 5.0) == (0.0, 0.0)
    w = 18000.0
    lag, ratio = physics.blast_secondary(w, 3.29)
    assert abs(lag - 0.027) < 0.002 and 0.1 < ratio < 0.3
    assert physics.blast_secondary(w, 0.0001)[1] > ratio


def test_explosion_secondary_pulse_is_off_by_default_and_adds_a_delayed_copy():
    a = physics.Explosion(1000.0, 3.0, 0.0, 0.0).render(1.0, sr=48000)
    b = physics.Explosion(1000.0, 3.0, 0.0, 0.0, secondary=1.0).render(1.0, sr=48000)
    assert np.array_equal(a, physics.Explosion(1000.0, 3.0, 0.0, 0.0, secondary=0.0).render(1.0, sr=48000))
    _clean(b)
    assert len(a) == len(b) and not np.allclose(a, b)
    far = physics.Explosion(1.0, 30.0, 0.0, 0.0).render(1.0, sr=48000)
    assert np.array_equal(far, physics.Explosion(1.0, 30.0, 0.0, 0.0, secondary=1.0).render(1.0, sr=48000))


def test_explosion_model_farnell_is_the_layered_patch():
    f = sfx.explosion(sr=SR, dur=2.0, model="farnell", seed=1)
    _clean(f)
    assert len(f) > 2.0 * SR
    p = sfx.explosion(sr=SR, dur=1.0, seed=1)
    assert not (len(p) == len(f) and np.allclose(p, f))
    assert np.array_equal(p, sfx.explosion(sr=SR, dur=1.0, seed=1, model="physical", secondary=0.0))


def test_assembly_multipliers_are_the_geometric_means_of_the_damping_table():
    bare = (0.01 * 0.05) ** 0.5
    for name, lo, hi in (("welded", 0.1, 0.5), ("bolted", 0.5, 2.0), ("reinforced", 2.0, 5.0), ("elastomer", 5.0, 20.0)):
        assert abs(contact.ASSEMBLY_DAMPING[name] / ((lo * hi) ** 0.5 / bare) - 1.0) < 0.02
    assert contact.ASSEMBLY_DAMPING["bare"] == 1.0


def test_impact_assembly_shortens_the_ring_and_default_is_unchanged():
    base = contact.impact(sr=SR, material="steel", shape="bar", seed=3, dur=0.8)
    assert np.array_equal(base, contact.impact(sr=SR, material="steel", shape="bar", seed=3, dur=0.8, assembly=""))
    bolted = contact.impact(sr=SR, material="steel", shape="bar", seed=3, dur=0.8, assembly="bolted")
    _clean(bolted)
    tail = slice(int(0.3 * SR), int(0.8 * SR))
    assert _rms(bolted[tail]) < 0.2 * _rms(base[tail])
    try:
        contact.impact(sr=SR, assembly="glued")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown assembly accepted")


def test_jet_bore_reports_strouhal_region_and_bounds_the_jet_speed():
    r = graphsolver.jet_bore(440.0, 0.1, jet_velocity=18.0, sr=24000)
    st = 440.0 * 0.009 / 18.0
    assert abs(r.metadata["strouhal"] - st) < 1e-12
    assert r.metadata["gain_region"] == graphsolver.jet_gain_region(st) == 1
    assert graphsolver.jet_gain_region(0.4) == 0 and graphsolver.jet_gain_region(0.6) == 2
    slow = graphsolver.jet_bore(440.0, 0.1, jet_velocity=3.0, sr=24000, bound=True)
    assert abs(slow.metadata["strouhal"] - 0.2871) < 1e-9 and slow.metadata["gain_region"] == 1
    fast = graphsolver.jet_bore(440.0, 0.1, jet_velocity=90.0, sr=24000, bound=True)
    assert abs(fast.metadata["strouhal"] - 0.0705) < 1e-9
    p = graphsolver.jet_bore(440.0, 0.1, pressure=200.0, sr=24000)
    assert abs(p.metadata["jet_velocity"] - math.sqrt(2 * 200.0 / 1.2041)) < 1e-9
    _clean(p.audio)


def test_jet_peak_hz_follows_st_falling_from_0_08_to_0_03_with_pressure():
    lo, hi = acoustic.jet_peak_hz(0.1), acoustic.jet_peak_hz(1.0)
    assert abs(lo - 0.08 * math.sqrt(2 * 40.0 / 1.2) / 1e-3 * (1 - 0.05 / 0.08 * 40.0 / 750.0)) < 1.0
    assert hi > lo > 0.0


def test_flute_jet_peak_adds_noise_at_the_peak_and_zero_is_unchanged():
    a = acoustic.flute(440.0, 0.5, sr=SR, vel=0.8)
    assert np.array_equal(a, acoustic.flute(440.0, 0.5, sr=SR, vel=0.8, jet_peak=0.0))
    b = acoustic.flute(440.0, 0.5, sr=SR, vel=0.8, jet_peak=1.0)
    _clean(b)
    fj = acoustic.jet_peak_hz(0.25 + 0.75 * 0.8)
    f = np.fft.rfftfreq(len(a), 1.0 / SR)
    band = (f > 0.85 * fj) & (f < 1.15 * fj)
    ea = np.sum(np.abs(np.fft.rfft(a * np.hanning(len(a))))[band] ** 2)
    eb = np.sum(np.abs(np.fft.rfft(b * np.hanning(len(b))))[band] ** 2)
    assert eb > 1.3 * ea


def test_aeolian_harp_density_and_damping_set_the_mass_damping_parameter():
    u = np.full(modal.seconds(1.0), 3.0)
    notes = [293.66, 440.0]
    base = modal.aeolian_harp(u, notes, 1.0, np.random.default_rng(1))
    same = modal.aeolian_harp(u, notes, 1.0, np.random.default_rng(1), density=modal.RHO_AIR, zeta=0.05)
    assert np.allclose(base, same)
    heavy = modal.aeolian_harp(u, notes, 1.0, np.random.default_rng(1), density=2432.2, zeta=0.006)
    _clean(heavy)
    assert _rms(heavy) < _rms(base)
