"""Group V3A (2026-11-06): micro-collisions, car cabin, percussion and contact leftovers (sources: research/SOURCES.md).

Run:  uv run --with pytest pytest tests/test_v3a.py -q
"""
import math

import numpy as np

SR = 44100


def test_micro_collisions_within_15_ms_on_the_four_lowest_periods():
    from genny.physical import modal as M
    modes = M.Modes([100.0, 300.0, 700.0, 1200.0, 5000.0], [5.0] * 5, [0.1, 0.1, 0.1, 0.1, 1.0], "vdd")
    allowed = {int(round(M.SR / f)) for f in (100.0, 300.0, 700.0, 1200.0)}
    onsets_all = []
    for seed in range(20):
        force = np.zeros(M.SR)
        M.micro_collisions(force, modes, 12, 1.0, 1.0, np.random.default_rng(seed))
        nz = force > 0
        on = np.nonzero(nz & ~np.r_[False, nz[:-1]])[0]
        assert np.nonzero(nz)[0][-1] < M.MICRO_WINDOW_MS * 1e-3 * M.SR
        gaps = np.diff(np.r_[len(M.bang(1.0)) + 1, on])
        assert set(gaps.tolist()) <= allowed
        onsets_all += gaps.tolist()
    assert len(set(onsets_all)) >= 3


def test_car_ir_is_0_db_in_the_1_khz_octave():
    from genny import dsp as D
    h = D.make_ir("car", SR)
    n = 1 << 16
    f = np.fft.rfftfreq(n, 1 / SR)
    H = np.abs(np.fft.rfft(h, n))
    band = H[(f >= 1000 / math.sqrt(2)) & (f <= 1000 * math.sqrt(2))]
    assert abs(10 * math.log10(np.mean(band ** 2))) < 0.2
    low = H[(f >= 15) & (f <= 25)]
    assert 20 * math.log10(np.sqrt(np.mean(low ** 2))) > 6.0


def test_free_disc_frequency_equation_matches_ducceschi_table_at_nu_038():
    from genny import contact as C
    for mode, w in C.FREE_DISC_OMEGA.items():
        assert abs(C.free_disc_omega(mode, 0.38) / w - 1.0) < 2e-3
    assert abs(C.free_disc_omega((2, 0), 0.3) - 5.358) < 0.005          # Leissa / Itao-Crandall, nu = 0.3
    assert abs(C.free_disc_omega((0, 1), 0.3) - 9.003) < 0.005


def test_circ_plate_free_shape_ratios_and_free_rim():
    from genny import contact as C
    b = C.ShapeBody("circ_plate_free", 100.0, n_modes=6, position=1.0)
    assert np.allclose(b.freqs[:3] / 100.0, [1.0, 9.003 / 5.358, 12.439 / 5.358], rtol=2e-3)
    assert np.all(np.abs(b.phi[:3]) > 0.5)                              # the free rim moves (clamped: node)
    assert abs(C.fundamental("circ_plate_free", 0.8, "steel", thickness=1e-3) - C.free_disc_freq(0.8, 1e-3, "steel")) < 1e-9


def test_cut_length_inverts_the_bar_fundamental():
    from genny import contact as C
    tube = (0.0125, 0.0115)
    L = C.cut_length(585.8, "aluminium", tube=tube)
    assert abs(C.fundamental("bar", L, "aluminium", tube=tube) - 585.8) < 1e-6
    assert abs(L - 0.505) < 0.01


def test_airy_disc_roots_solve_b9_and_coupling_rules():
    from scipy import special as S
    from genny import contact as C
    for l in (0, 1, 4):
        x = C.airy_disc_roots(l, 4)
        assert len(x) == 4
        res = S.jv(l - 1, x) * S.iv(l, x) - S.iv(l - 1, x) * S.jv(l, x)
        assert np.max(np.abs(res / S.iv(l, x))) < 1e-8
    assert abs(C.airy_disc_roots(0, 1)[0] - 3.19622) < 1e-4
    assert C.vk_disc_couplings(0) == (0,) and C.vk_disc_couplings(3) == (0, 6)


def test_residual_compliance_restores_the_static_point_compliance():
    from genny import contact as C
    few = C.ShapeBody("bar", 200.0, boundary="supported", position=0.3, n_modes=4, mass=0.1)
    many = C.ShapeBody("bar", 200.0, boundary="supported", position=0.3, n_modes=200, fmax=1e9, mass=0.1)
    stat = lambda b: float(np.sum(b.phi ** 2 / (b.mass * (2 * np.pi * b.freqs) ** 2)))   # noqa: E731
    assert abs((stat(few) + few.residual_compliance(200)) / stat(many) - 1.0) < 1e-6
    assert few.residual_compliance(200) > 0.001 * stat(many)
    f = np.r_[np.hanning(48), np.zeros(200)]
    _, u0, _ = C.modal_bank(f, few, 48000)
    _, u1, _ = C.modal_bank(f, few, 48000, residual=True)
    assert np.allclose(u1 - u0, few.residual_compliance() * f)


def test_modal_engine_rank_pick_reaches_the_thesis_span():
    from genny import contact as C
    sr = 22050
    lo = C.modal_engine(sr=sr, dur=0.6, fan=0.0)
    hi = C.modal_engine(sr=sr, dur=0.6, fan=0.0, pick="rank")
    f = np.fft.rfftfreq(lo.size, 1 / sr)
    share = lambda y: np.sum(np.abs(np.fft.rfft(y))[f > 400] ** 2) / np.sum(np.abs(np.fft.rfft(y)) ** 2)  # noqa: E731
    assert share(lo) < 0.05 < share(hi)


def test_sine_noise_band_and_decay():
    from genny.physical import ambience as A
    y = A.sine_noise(A.SR, np.random.default_rng(0))
    f = np.fft.rfftfreq(y.size, 1 / A.SR)
    P = np.abs(np.fft.rfft(y)) ** 2
    lo, hi = A.TEAR_BAND[0] * A.SR, A.TEAR_BAND[1] * A.SR
    assert np.sum(P[(f > 0.8 * lo) & (f < 1.2 * hi)]) > 0.8 * np.sum(P)
    q = A.SR // 4
    r = np.sqrt(np.mean(y[q:2 * q] ** 2) / np.mean(y[:q] ** 2))
    assert abs(r - math.exp(-0.25 / A.TEAR_DECAY)) < 0.05


def test_stone_scrape_bank_scales_as_one_over_sqrt_size(monkeypatch):
    from genny.physical import ambience as A
    e = np.full(A.SR, 0.8)
    def centroid(y):
        f = np.fft.rfftfreq(y.size, 1 / A.SR)
        P = np.abs(np.fft.rfft(y)) ** 2
        return np.sum(f * P) / np.sum(P)
    y = A.stone_scrape(e, np.random.default_rng(1), size=4.0)
    assert y.size == A.SR and np.isfinite(y).all()
    # the stick-slip pulses peak at 200-1000 Hz whatever the size: drive the bank with white noise to see it alone
    monkeypatch.setattr(A, "stickslip_pulses", lambda f, rng, **k: rng.standard_normal(len(f)))
    c1 = centroid(A.stone_scrape(e, np.random.default_rng(1), size=1.0))
    c4 = centroid(A.stone_scrape(e, np.random.default_rng(1), size=4.0))
    assert 0.35 < c4 / c1 < 0.65                                          # bank x 1/sqrt(4)
    monkeypatch.undo()
    y = A.tear(e, np.random.default_rng(2))
    assert y.size == A.SR and np.isfinite(y).all()


def test_sway_params_under_load():
    from genny.physical import ambience as A
    f, q = A.sway_params(np.array([0.0, 1.0]), 0.4, 3.0)
    assert np.allclose(f, [0.4, 0.4 * (1 - A.SWAY_F_DROP)]) and np.allclose(q, [3.0, 3.0 / (1 + A.SWAY_DAMP_RISE)])
    w = np.full(A.SR * 2, 0.8)
    a = A.sway_force(w, np.random.default_rng(0))
    b = A.sway_force(w, np.random.default_rng(0), amplitude=1.0)
    assert not np.allclose(a, b)


def test_fineberg_strike_length_and_count():
    from genny.physical import ambience as A
    v = 0.5
    T = max(2, A.seconds(A.FB_STRIKE_MS * (1.4 - v) ** 5 / 1000.0))
    s = A.fineberg_strike(v, np.random.default_rng(3))
    assert s.size == 2 * T and np.max(np.abs(s)) > 0
    assert abs(T / A.SR * 1000 - 240 * 0.9 ** 5) < 0.1


def test_guitar_low_pair_is_the_coupled_top_and_hole():
    from genny import acoustic as AC
    from genny import foley as FO
    top, hole = AC.guitar_uncoupled()
    assert abs(top - 179.95) < 0.2 and abs(hole - 112.97) < 0.2                # typical top and Helmholtz notes
    assert np.allclose(FO.guitar_body_modes(top, hole, AC.GUITAR_VOLUME), (91.0, 192.0), rtol=1e-9)
    f0 = AC.guitar_body_modes(0)[0]
    fv = AC.guitar_body_modes(0, volume=AC.GUITAR_VOLUME)[0]
    assert np.allclose(f0, fv, rtol=1e-9)
    big = AC.guitar_body_modes(0, volume=2 * AC.GUITAR_VOLUME)[0]
    assert big[0] < 0.85 * f0[0] and big[1] < f0[1] and np.allclose(big[2:], f0[2:])


def test_tr808_pulse_shaper_follows_its_transfer_and_diode():
    from genny import drums as D
    R162, R163, C40, _, _ = D.TR808_PULSE_SHAPER
    sr = 48000
    lin = D.pulse_shaper(14.0, sr, diode=False)
    tau = R162 * R163 / (R162 + R163) * C40
    i = int(0.0009 * sr)                                                   # end of the 1 ms pulse: H(0) = R162 / (R162 + R163)
    assert abs(lin[i] / 14.0 - R162 / (R162 + R163)) < 0.01
    assert lin[0] > 14.0 * 0.6 and lin[0] > lin[3]                          # H(inf) = 1, decaying with tau = 67 us
    assert lin.min() < -10.0                                               # the falling edge without D53
    v = D.pulse_shaper(14.0, sr)
    assert -0.8 < v.min() < -0.2 and abs(v[i] - lin[i]) < 0.02            # D53 clamps it near a diode drop
    assert abs(tau - 67.1e-6) < 0.5e-6


def test_kick808_circuit_attack_glide_and_resonance():
    from genny import drums as D
    sr = 48000
    y = D.kick808(sr, 1.0, model="circuit")
    def hz(seg):
        z = np.nonzero(np.diff(np.signbit(seg)))[0]
        return (len(z) - 1) / 2 / ((z[-1] - z[0]) / sr)
    assert abs(hz(y[int(0.4 * sr):int(0.9 * sr)]) / D.TR808_BD_HZ - 1.0) < 0.03
    assert hz(y[:int(0.012 * sr)]) > 1.4 * D.TR808_BD_HZ
    assert abs(np.sqrt(D.TR808_BD_ATTACK) - 2.813) < 0.01


def test_thermoelastic_plate_loss_zener_peak():
    from genny import plates as PL
    rho, E, nu, C, a, k = PL.THERMO["aluminium"]
    h = 1e-3
    tau = rho * C * h * h / (k * np.pi ** 2)
    assert abs(tau - 1.368e-3) < 0.01e-3
    f = np.logspace(0, 4, 2001)
    eta, d = PL.thermoelastic(f, h, "aluminium")
    assert abs(f[np.argmax(eta)] * 2 * np.pi * tau - 1.0) < 0.01
    delta = E * a * a * PL.ROOM_K / (rho * C) * (1 + nu) / (1 - nu)
    assert abs(eta.max() - delta / 2) < 1e-3 * delta and abs(delta - 8.1e-3) < 0.2e-3
    eta2, _ = PL.thermoelastic(f, 2 * h, "aluminium")
    assert abs(f[np.argmax(eta2)] / f[np.argmax(eta)] - 0.25) < 0.01             # peak goes as 1 / h^2
    assert np.allclose(d, eta * np.pi * f)
