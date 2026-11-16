"""Interior ballistics, muzzle blast, N-wave, suppressor, ground, air and recording chain: the design numbers, measured on the signals."""
import math

import numpy as np

from genny import ballistics as B

R = B.RATE
BECK_T0 = {"9x19": 0.29e-3, "556": 0.42e-3, "308": 0.47e-3}
MAHER_PEAK = {"9x19": 3307.0, "45acp": 5123.0, "357": 6491.0, "556": 8137.0, "308": 6780.0}
BARREL = {"22lr": 0.508, "9x19": 0.118, "45acp": 0.127, "357": 0.102, "556": 0.508, "308": 0.61, "50bmg": 1.143}


def _db(a, b):
    return 20.0 * math.log10(a / b)


def _axis(cart, r=3.0, az=0.0, trim=False, a_db=30.0):
    return B.blast_at(B.muzzle_state(cart, BARREL[cart]), cart, r, az, a_db, trim=trim)


def _shot(cart, r, rate=R, dur=0.5):
    st = B.muzzle_state(cart, BARREL[cart])
    pk, t0, b = B.blast_at(st, cart, r, 0.0)
    y = np.zeros(int(dur * rate))
    w = B.friedlander(pk, t0, b, None, rate)
    y[:len(w)] += w
    j = B.gas_jet(st, pk, np.random.default_rng(2), rate)
    y[:len(j)] += j
    return y


def _crest(y):
    return float(np.max(np.abs(y)) / math.sqrt(np.mean(y[:R // 10] ** 2)))


def test_velocity_556_by_barrel():
    assert abs(B.muzzle_velocity("556", 0.368) - 897.0) < 10.0
    assert abs(B.muzzle_velocity("556", 0.267) - 842.0) < 8.0
    assert abs(B.muzzle_velocity("556", 0.61) - 975.0) < 10.0


def test_velocity_9mm_by_barrel():
    assert abs(B.muzzle_velocity("9x19", 0.20) - 383.0) < 5.0
    assert abs(B.muzzle_velocity("9x19", 0.40) - 407.0) < 5.0


def test_velocity_reference_and_monotone():
    for k, c in B.CARTRIDGES.items():
        assert abs(B.muzzle_velocity(k, c.l_ref) - c.v_ref) < 1e-6
        v = [B.muzzle_velocity(k, c.l_ref * s) for s in (0.4, 0.7, 1.0, 1.5)]
        assert all(b > a for a, b in zip(v, v[1:])), k


def test_muzzle_state_556():
    s = B.muzzle_state("556", 0.508)
    assert abs(s["p_m"] / 1e6 - 50.4) < 0.5
    assert abs(s["e_b"] - 3226.0) < 20.0
    assert abs(s["w_tnt"] * 1e3 - 0.771) < 0.01
    assert abs(s["r0"] - 0.317) < 0.003
    assert abs(s["f_jet"] - 3220.0) < 40.0
    assert abs(s["t_gas"] - 1665.0) < 20.0


def test_muzzle_state_table():
    table = {("22lr", 0.508): (2.2, 122.0), ("9x19", 0.118): (16.7, 551.0), ("45acp", 0.127): (9.5, 547.0), ("357", 0.102): (46.2, 1560.0),
             ("762x39", 0.415): (31.2, 2964.0), ("308", 0.61): (39.1, 5416.0), ("300blk", 0.406): (23.9, 2075.0),
             ("300blk_sub", 0.406): (7.8, 649.0), ("50bmg", 1.143): (41.9, 30550.0), ("12_buck", 0.457): (4.1, 2103.0),
             ("musket", 1.0): (0.8, 970.0)}
    for (k, length), (pm, eb) in table.items():
        s = B.muzzle_state(k, length)
        assert abs(s["p_m"] / 1e6 / pm - 1.0) < 0.05, k
        assert abs(s["e_b"] / eb - 1.0) < 0.02, k


def test_energy_balance_and_efficiency():
    for k, c in B.CARTRIDGES.items():
        s = B.muzzle_state(k, c.l_ref)
        ek = 0.5 * c.m_g * 1e-3 * s["v"] ** 2
        eg = c.m_c_g * 1e-3 * s["v"] ** 2 / 6.0
        total = ek + eg + s["u_m"]
        assert abs(s["e_gas"] - eg) < 1e-9
        assert 0.25 < ek / total < 0.70, k
        assert s["p_m"] < c.p_max_mpa * 1e6, k


def test_shorter_barrel_louder_blast():
    long_, short = B.muzzle_state("556", 0.61), B.muzzle_state("556", 0.267)
    assert short["e_b"] > long_["e_b"] and short["p_m"] > long_["p_m"] and short["tau_b"] < long_["tau_b"]
    assert _db(B.blast_at(short, "556", 3.0, 0.0)[0], B.blast_at(long_, "556", 3.0, 0.0)[0]) > 0.3


def test_axis_peaks_3m_without_trim():
    expected = {"22lr": 1536.0, "9x19": 3445.0, "45acp": 3433.0, "357": 6014.0, "556": 8414.0, "308": 10137.0, "50bmg": 22804.0}
    for k, p in expected.items():
        assert abs(_axis(k)[0] / p - 1.0) < 0.01, k
    for k, p in MAHER_PEAK.items():
        assert abs(_db(_axis(k)[0], p)) < 3.6, k


def test_axis_peaks_3m_with_trim():
    for k, p in MAHER_PEAK.items():
        assert abs(_db(_axis(k, trim=True)[0], p)) < 0.35, k
    assert abs(_db(_axis("308", trim=True)[0], MAHER_PEAK["308"])) < 0.35


def test_t0_3m_against_published_durations():
    for k, t in BECK_T0.items():
        assert abs(_axis(k)[1] / t - 1.0) < 0.2, k
    assert abs(_axis("556")[1] * 1e3 - 0.425) < 0.005
    assert _axis("308", az=0.0)[1] == _axis("308", az=150.0)[1]
    assert abs(_axis("308")[1] * 1.7 - 0.85e-3) < 0.05e-3


def test_friedlander_shape_and_spectrum_peak():
    pk, t0, b = _axis("308")
    y = B.friedlander(pk, t0, b, None)
    assert y[0] == pk
    zero = int(np.argmax(y < 0))
    assert abs(zero / R - t0) < 1.5 / R
    assert y.min() < 0 and y.min() > -0.2 * pk
    f = np.fft.rfftfreq(1 << 16, 1 / R)
    assert 200.0 < f[np.argmax(np.abs(np.fft.rfft(y, 1 << 16)))] < 450.0
    pk5, t5, b5 = _axis("556")
    y5 = B.friedlander(pk5, t5, b5, None)
    assert 200.0 < f[np.argmax(np.abs(np.fft.rfft(y5, 1 << 16)))] < 450.0


def test_distance_law_weak_shock():
    st = B.muzzle_state("556", 0.508)
    r0 = st["r0"]
    p3, t3, _ = B.blast_at(st, "556", 3.0, 0.0)
    for r in (10.0, 30.0, 100.0):
        p, t, _ = B.blast_at(st, "556", r, 0.0)
        want = p3 * 3.0 / r * math.sqrt(math.log(3.0 / r0) / math.log(r / r0))
        assert abs(_db(p, want)) < 0.05
        assert abs(t / (t3 * math.sqrt(math.log(r / r0) / math.log(3.0 / r0))) - 1.0) < 1e-9
    assert abs(B.blast_at(st, "556", 10.0, 0.0)[1] * 1e3 - 0.527) < 0.005
    assert abs(B.blast_at(st, "556", 100.0, 0.0)[1] * 1e3 - 0.680) < 0.005
    assert abs(B.blast_at(st, "556", 3.0 + 1e-9, 0.0)[0] / p3 - 1.0) < 1e-6
    ts = [B.blast_at(st, "556", r, 0.0)[1] for r in (3, 10, 30, 100, 300)]
    assert all(b > a for a, b in zip(ts, ts[1:]))


def test_air_absorption_beyond_three_metres_only():
    x = np.zeros(4096)
    x[0] = 1.0
    f = np.fft.rfftfreq(4096, 1 / R)
    h = np.abs(np.fft.rfft(B.air_absorb(x, 100.0)))
    at = lambda hz: -_db(h[np.argmin(abs(f - hz))], 1.0)       # noqa: E731
    assert abs(at(4000.0) - 97.0 * 0.0297) < 0.4
    assert at(100.0) < 0.1
    assert np.array_equal(B.air_absorb(x, 2.0), x)
    y = B.air_absorb(x, 100.0)
    assert int(np.argmax(np.abs(y))) < 20 and abs(y.sum() - 1.0) < 0.03


def test_ten_metre_level_with_air_matches_design():
    st = B.muzzle_state("556", 0.508)
    p3, t3, b = B.blast_at(st, "556", 3.0, 0.0, trim=False)
    p10, t10, _ = B.blast_at(st, "556", 10.0, 0.0, trim=False)
    w = B.air_absorb(B.friedlander(p10, t10, b, None), 10.0)
    assert abs(_db(p10, p3) - (-12.3)) < 0.15
    assert -1.5 < _db(np.max(np.abs(w)), p10) < 0.0


def test_directivity():
    p0 = _axis("556", az=0.0)[0]
    assert abs(_db(_axis("556", az=90.0)[0], p0) + 11.1) < 0.3
    assert abs(_db(_axis("556", az=165.0)[0], p0) + 26.8) < 0.3
    assert abs(_db(_axis("556", az=165.0, a_db=10.0)[0], p0) + 26.8 / 3.0) < 0.3
    assert abs(_db(_axis("556", az=90.0, a_db=0.0)[0], p0)) < 1e-9


def test_small_calibre_against_large():
    d = _db(_axis("22lr", trim=True)[0], _axis("308", trim=True)[0])
    assert abs(d + 25.0) < 3.0
    p50 = B.blast_at(B.muzzle_state("50bmg", 1.143), "50bmg", 10.0, 30.0)[0]
    p22 = B.blast_at(B.muzzle_state("22lr", 0.508), "22lr", 10.0, 30.0)[0]
    assert _db(p50, p22) > 30.0
    assert abs(_axis("50bmg")[0] / 22804.0 - 1.0) < 0.01


def test_precursor():
    st = B.muzzle_state("556", 0.508)
    y, lead = B.precursor(st, "556", 10.0, 30.0, 0.508, 948.0)
    pk = B.blast_at(st, "556", 10.0, 0.0)[0]
    assert abs(lead * 1e6 - 89.0) < 1.0
    assert abs(y[0] / (0.12 * pk * math.cos(math.radians(30.0)) ** 2) - 1.0) < 1e-9
    t0 = B.blast_at(st, "556", 10.0, 0.0)[1]
    assert abs(len(y) / R - 0.3 * t0 * (1 + 8 / 0.88)) < 2 / R
    z, lz = B.precursor(st, "556", 10.0, 120.0, 0.508, 948.0)
    assert not z.any() and lz == 0.0
    y9, l9 = B.precursor(B.muzzle_state("9x19", 0.118), "9x19", 10.0, 0.0, 0.118, 360.0)
    assert abs(l9 * 1e6 - 55.6) < 1.0


def test_gas_jet():
    st = B.muzzle_state("556", 0.508)
    j = B.gas_jet(st, 2000.0, np.random.default_rng(3))
    assert abs(math.sqrt(np.mean(j * j)) / 60.0 - 1.0) < 1e-9
    assert abs(len(j) / R - 5.0 * st["tau_b"]) < 2 / R
    h = len(j) // 4
    assert np.sqrt(np.mean(j[:h] ** 2)) > 3.0 * np.sqrt(np.mean(j[-h:] ** 2))
    spec = np.abs(np.fft.rfft(j, 1 << 14)) ** 2
    f = np.fft.rfftfreq(1 << 14, 1 / R)
    assert 0.3 * st["f_jet"] < np.sum(f * spec) / np.sum(spec) < 2.5 * st["f_jet"]
    j2 = B.gas_jet(st, 2000.0, np.random.default_rng(3))
    assert np.array_equal(j, j2)
    assert B.gas_jet(B.muzzle_state("22lr", 0.508), 100.0, np.random.default_rng(1)).max() > 0


def test_brake_and_flash():
    p, t0, b = _axis("556", trim=True)
    (a1, d1), (a2, d2) = B.brake_pulses(p, t0, b)
    assert (d1, d2) == (0.15e-3, 0.30e-3)
    assert abs(a1[0] / p - 0.4) < 1e-12 and abs(a2[0] / p - 0.25) < 1e-12
    f, d = B.flash_pulse(p, t0, b, 0.8)
    assert d == 2e-3 and abs(f[0] / p - 0.4) < 1e-12
    assert abs(len(f) / len(B.friedlander(p, t0, b, None)) - 2.0) < 0.02
    assert not B.flash_pulse(p, t0, b, 0.0)[0].any()
    assert B.brake_boost_db(120.0) == 3.0 and B.brake_boost_db(30.0) == 0.0 and B.BRAKE_A_DB == 10.0


def test_suppressor():
    p, t0, b = _axis("556", trim=True)
    p2, t2 = B.suppress(p, t0, "556")
    assert abs(_db(p, p2) - 26.0) < 1e-9 and abs(t2 / t0 - 10 ** (26 / 40)) < 1e-9
    assert abs(_db(p, B.suppress(p, t0, "556", 10.0)[0]) - 10.0) < 1e-9
    assert B.suppress(p, 5e-3, "50bmg")[1] == 10e-3
    pe, _ = B.suppress(p, t0, "556", ejection_port=True)
    assert abs(_db(p, pe) - 14.0) < 1e-9
    assert abs(_db(p, B.suppress(p, t0, "357")[0]) - 25.0) < 1e-9
    assert (B.can_warmup_db(0), B.can_warmup_db(1), B.can_warmup_db(2)) == (3.0, 1.5, 0.0)
    assert abs(_db(B.suppress(1.0, 0.4e-3, "22lr")[0], 1.0) + 22.0) < 1e-9


def test_can_tone_and_helmholtz():
    t = B.can_tone(100.0, np.random.default_rng(4))
    assert abs(_db(np.max(np.abs(t)), 100.0) + 12.0) < 1e-6
    f = np.fft.rfftfreq(1 << 15, 1 / R)
    assert abs(f[np.argmax(np.abs(np.fft.rfft(t, 1 << 15)))] / 1100.0 - 1.0) < 0.08
    assert 700.0 < B.helmholtz_freq(math.pi * 0.008 ** 2 / 4, 30e-6, 0.01) < 1600.0
    assert B.helmholtz_freq(1e-4, 1e-4, 0.01) > B.helmholtz_freq(1e-4, 4e-4, 0.01)


def test_ground_reflection():
    x = np.zeros(4096)
    x[0] = 1.0
    y = B.ground(x, 10.0, 1.5, 1.6, "hard")
    assert len(y) == 4096 + int(0.05 * R)
    h = np.abs(np.fft.rfft(y, 1 << 15))
    f = np.fft.rfftfreq(1 << 15, 1 / R)
    dt = (math.hypot(10.0, 3.1) - math.hypot(10.0, 0.1)) / B.C_AIR
    band = (f > 800.0) & (f < 1500.0)
    assert abs(f[band][np.argmin(h[band])] * 2 * dt / 3.0 - 1.0) < 0.04
    assert abs(h[(f > 20) & (f < 60)].mean() / 2.0 - 1.0) < 0.1
    assert B.ground(x, 10.0, 1.5, 1.6, "none") is x
    soft = B.ground(x, 30.0, 1.5, 1.6, "grass")
    assert np.all(np.isfinite(soft))


def test_recorder_none_and_linear_field():
    y = _shot("22lr", 50.0)
    assert np.array_equal(B.recorder(y, "none"), y) and B.recorder(y, "none") is not y
    f1, f2 = B.recorder(y, "field"), B.recorder(2.0 * y, "field")
    assert abs(np.max(np.abs(f2)) / np.max(np.abs(f1)) - 2.0) < 1e-3
    assert abs(_db(np.max(np.abs(f1)), np.max(np.abs(y)))) < 0.1
    assert np.max(np.abs(B.recorder(y * 0.0, "field"))) == 0.0


def test_recorder_camera_quiet_and_loud():
    quiet = _shot("22lr", 50.0)
    assert abs(_db(np.max(np.abs(B.recorder(quiet, "camera"))), np.max(np.abs(quiet)))) < 0.5
    loud = _shot("50bmg", 10.0)
    cam = B.recorder(loud, "camera")
    assert _db(_crest(loud), _crest(cam)) >= 6.0
    assert np.max(np.abs(cam)) < 20e-6 * 10 ** (135 / 20)
    assert np.all(np.isfinite(cam))


def test_recorder_cuts_differ_by_weapon_not_by_normalisation():
    quiet, loud = B.recorder(_shot("22lr", 50.0), "camera"), B.recorder(_shot("50bmg", 10.0), "camera")
    assert _db(np.max(np.abs(loud)), np.max(np.abs(quiet))) > 6.0
    assert _crest(quiet) > 2.0 * _crest(loud)
    raw = _shot("22lr", 50.0)
    ph = B.recorder(raw, "phone")
    f = np.fft.rfftfreq(1 << 14, 1 / R)

    def low_ratio(x):
        s = np.abs(np.fft.rfft(x, 1 << 14))
        return s[(f > 20) & (f < 100)].mean() / s[(f > 300) & (f < 1500)].mean()
    assert low_ratio(ph) < 0.3 * low_ratio(raw)
    try:
        B.recorder(ph, "tape")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown recorder accepted")


def test_half_sine_force():
    for tc in (20e-6, 62e-6, 400e-6):
        f = B.half_sine_force(2.5, tc, 4800)
        assert abs(f.sum() / (2.5 * R) - 1.0) < 1e-6
        assert f.min() >= 0.0
    tc = 400e-6
    f = B.half_sine_force(1.0, tc, 4096)
    spec = np.abs(np.fft.rfft(f, 1 << 16))
    fq = np.fft.rfftfreq(1 << 16, 1 / R)
    assert spec[np.argmin(abs(fq - 1.5 / tc))] / spec[0] < 1e-3
    assert abs(spec[np.argmin(abs(fq - 0.3 / tc))] / spec[0] - math.cos(math.pi * 0.3) / (1 - 0.36)) < 0.02
    assert abs(f.max() / (math.pi / 2.0 * R / (tc * R)) - 1.0) < 0.05
    assert abs(B.half_sine_force(2.0, tc, 4096).sum() / f.sum() - 2.0) < 1e-9


def test_n_wave_556():
    p, t = B.n_wave_pa(948.0, 5.7, 23.0, 3.0, near=False)
    assert abs(p / 437.0 - 1.0) < 0.01 and abs(t - 139e-6) < 1 / R
    for v, d, l, want in ((853.0, 7.8, 29.0, 547.0), (928.0, 12.9, 58.0, 780.0), (390.0, 5.7, 11.0, 356.0)):
        assert abs(B.n_wave_pa(v, d, l, 3.0, near=False)[0] / want - 1.0) < 0.01
    assert B.n_wave_pa(300.0, 9.0, 15.5, 3.0) == (0.0, 0.0)
    assert B.n_wave_pa(B.C_AIR, 5.7, 23.0, 3.0) == (0.0, 0.0)
    near, far = B.n_wave_pa(948.0, 5.7, 23.0, 1.0)[0], B.n_wave_pa(948.0, 5.7, 23.0, 1.0, near=False)[0]
    assert abs(_db(near, far) - 1.8) < 0.1
    assert B.n_wave_pa(948.0, 5.7, 23.0, 10.0)[0] == B.n_wave_pa(948.0, 5.7, 23.0, 10.0, near=False)[0]


def test_n_wave_emission_and_cone():
    lam = 700.0
    ve, xe = B.emission_speed(948.0, 100.0, 5.0, lam)
    assert B.C_AIR < ve < 948.0
    assert abs(ve - 948.0 * math.exp(-xe / lam)) < 0.5
    assert abs(xe - (100.0 - 5.0 * math.tan(math.asin(B.C_AIR / ve)))) < 1e-2
    assert B.emission_speed(948.0, 1.0, 5.0, lam) == (0.0, 0.0)
    assert B.emission_speed(300.0, 100.0, 5.0, lam) == (0.0, 0.0)
    assert B.emission_speed(400.0, 600.0, 2.0, 100.0) == (0.0, 0.0)
    assert B.emission_speed(948.0, 100.0, 5.0, 1e9)[0] > 947.9
    sig, t = B.n_wave(948.0, 5.7, 23.0, 100.0, 5.0, lam)
    assert t is not None and sig[0] > 0 and sig.min() < 0
    p_slow, _ = B.n_wave_pa(ve, 5.7, 23.0, 5.0)
    assert abs(sig[0] / p_slow - 1.0) < 1e-9
    assert t > 100.0 / 948.0
    miss, tn = B.n_wave(948.0, 5.7, 23.0, 1.0, 5.0, lam)
    assert tn is None and not miss.any()


def test_cartridge_table_consistent():
    assert set(B.CARTRIDGES) >= {"22lr", "9x19", "45acp", "357", "556", "762x39", "308", "300blk", "50bmg", "12_buck", "musket"}
    for k, c in B.CARTRIDGES.items():
        e = 0.5 * c.m_g * 1e-3 * c.v_ref ** 2
        assert 100.0 < e < 20000.0, k
        assert c.v_c_cm3 > 0 and c.m_c_g > 0 and 0.8 < c.b < 0.9
    assert B.CARTRIDGES["12_buck"].pellets == 9 and B.CARTRIDGES["12_buck"].bore_mm == 18.5
    assert abs(0.5 * 0.0040 * 948.0 ** 2 - 1797.0) < 2.0
    assert B.CARTRIDGES["musket"].t_flame == 2000.0
    assert B.CARTRIDGES["50bmg"].supp_db == 32.0 and B.CARTRIDGES["22lr"].trim_db == -12.0


def test_full_chain_hygiene():
    rng = np.random.default_rng(5)
    for k, c in B.CARTRIDGES.items():
        st = B.muzzle_state(k, c.l_ref)
        pk, t0, b = B.blast_at(st, k, 25.0, 40.0)
        parts = [B.friedlander(pk, t0, b, None), B.gas_jet(st, pk, rng), B.precursor(st, k, 25.0, 40.0, c.l_ref, st["v"])[0]]
        ps, ts = B.suppress(pk, t0, k)
        parts += [B.friedlander(ps, ts, b, None), B.can_tone(ps, rng)]
        y = np.zeros(int(0.3 * R))
        for w in parts:
            y[:len(w)] += w
        y = B.air_absorb(y, 25.0)
        y = B.ground(y, 25.0, 1.5, 1.6, "asphalt")
        for rec in ("none", "field", "camera", "phone"):
            z = B.recorder(y, rec)
            assert np.all(np.isfinite(z)) and len(z) == len(y), (k, rec)
        assert len(y) / R < 0.5
