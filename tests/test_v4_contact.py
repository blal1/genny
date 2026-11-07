import math

import numpy as np

from genny import contact as C

SR = 44100


def _band_power(x, sr, f0, half=15.0):
    X = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    return float(X[(f > f0 - half) & (f < f0 + half)].mean())


def test_resonant_prefilter_lifts_the_force_at_the_modes():
    body = C.ModalObject(np.array([440.0, 1300.0]), np.array([20.0, 30.0]), np.ones(2))
    x = np.random.default_rng(0).standard_normal(SR)
    y = C.resonant_prefilter(x, body, SR, 1.0)
    assert math.isclose(float(np.sqrt(np.mean(y * y))), float(np.sqrt(np.mean(x * x))), rel_tol=1e-6)
    lift = _band_power(y, SR, 440.0) / _band_power(x, SR, 440.0)
    gap = _band_power(y, SR, 850.0) / _band_power(x, SR, 850.0)
    assert lift > 20.0 * gap
    assert np.array_equal(C.resonant_prefilter(x, body, SR, 0.0), x)


def test_plastic_texture_is_smooth_plastic_fractal_dimension():
    beta = C._TEXTURES["plastic"][0]
    assert math.isclose(2.0 - beta / 2.0, 0.88, abs_tol=1e-9)


def test_hertz_elliptic_reduces_to_hertz_and_crossed_cylinders():
    es, p, r = 100e9, 10.0, 0.01
    a, b, d, p0 = C.hertz_elliptic(p, r, r, es)
    ah = (3 * p * r / (4 * es)) ** (1 / 3)
    assert math.isclose(a, ah, rel_tol=0.04) and math.isclose(b, ah, rel_tol=0.04)
    assert math.isclose(d, ah * ah / r, rel_tol=0.03)
    a2, b2, _, _ = C.hertz_elliptic(p, r, 5.83 * r, es)
    assert math.isclose(b2 / a2, 5.83 ** (-2 / 3), rel_tol=0.08)
    assert math.isclose(C.hertz_k_elliptic(es, r, r), C.hertz_k(2 * es, 0.0, 2 * es, 0.0, r), rel_tol=0.05)


def test_line_and_cylinder_and_edge():
    tau, z = C.line_contact_shear(1000.0, 0.01, 100e9)
    a, p0 = C.line_contact(1000.0, 0.01, 100e9)
    assert math.isclose(tau, 0.30 * p0) and math.isclose(z, 0.78 * a)
    d1 = C.cylinder_between_plates(1000.0, 0.01, 200e9, 0.3)
    d2 = C.cylinder_between_plates(2000.0, 0.01, 200e9, 0.3)
    assert 1.8 < d2 / d1 < 2.0
    assert math.isclose(C.hertz_edge_tension(3.0, 0.3), 0.4)


def test_jkr_pulloff_on_the_load_curve():
    g, r, es = 0.05, 0.01, 1e6
    pc, ac = C.jkr_pulloff(g, r, es)
    assert math.isclose(C.jkr_load(ac, g, r, es), -pc, rel_tol=1e-9)


def test_elastoplastic_loading_and_crater():
    r, y, es = 0.005, 1e9, 115e9
    a_y = 1.07 * 3 * math.pi * r * y / (4 * es)
    assert math.isclose(C.elastoplastic_mean_pressure(0.5 * a_y, r, y, es), 4 * es * 0.5 * a_y / (3 * math.pi * r))
    assert math.isclose(C.elastoplastic_mean_pressure(1.0, r, y, es), 3 * y)
    dmax = 20 * a_y ** 2 / r
    pm, res = C.elastoplastic_contact(dmax, dmax, r, y, es)
    assert 0 < res < dmax
    assert C.elastoplastic_contact(res, dmax, r, y, es)[0] == 0.0
    assert C.elastoplastic_contact(0.5 * (res + dmax), dmax, r, y, es)[0] < pm


def test_matthews_and_layer_exponents():
    assert C.matthews_alpha(1) == 1.5 and C.matthews_alpha(1e9) < 1.0001
    d, _ = C.matthews_indentation(1e-3, 0.01, 1e6, 1e8, 1e-3)
    assert math.isclose(d, 0.368 * 1e-4, rel_tol=0.01)
    k, al = C.layer_k(1e6, 0.3, 1e-3, 0.01)
    kb, _ = C.layer_k(1e6, 0.3, 1e-3, 0.01, bonded=True)
    assert al == 1.5 and math.isclose(kb / k, 0.49 / 0.4)
    assert C.layer_k(1e6, 0.5, 1e-3, 0.01, bonded=True)[1] == 2.5


def test_wavy_gw_and_rough_sphere():
    assert math.isclose(C.wavy_contact(0.5, 1e9, 1e-6, 1e-3), 2 / math.pi * math.asin(math.sqrt(0.5 / (math.pi * 1e9 * 1e-3))))
    assert C.wavy_contact(1e9, 1e9, 1e-6, 1e-3, two_d=True) == 1.0
    frac, pr = C.gw_contact(1e5, 1e11, 1e-6, 1e5)
    assert math.isclose(pr, 0.56 * 1e11 * math.sqrt(0.1))
    assert math.isclose(frac, math.sqrt(math.pi) * 1e5 / (1e11 * math.sqrt(0.1)))
    al = C.rough_sphere_alpha(1e-6, 0.01, 1.0, 1e11)
    assert math.isclose(al, 1e-6 * (16 * 0.01 * 1e22 / 9) ** (1 / 3))
    assert C.rough_hertz_k(5.0, 0.01) == 5.0 and C.rough_hertz_k(5.0, 0.5) < 5.0


def test_mindlin_stiffness_loss_and_torque():
    g, nu = 80e9, 0.3
    c, d, kt = C.mindlin(0.0, 100.0, 0.5, 1e-4, g, nu, g, nu)
    assert c == 1.0 and d == 0.0 and math.isclose(kt, 8e-4 / (2 * 1.7 / g))
    assert C.mindlin(50.0, 100.0, 0.5, 1e-4, g, nu, g, nu)[0] == 0.0
    small = C.mindlin_loss(1.0, 100.0, 0.5, 1e-4, g, nu, g, nu)
    assert math.isclose(small, 2 * 1.7 / g * 1.0 / (36 * 1e-4 * 0.5 * 100.0), rel_tol=0.02)
    assert math.isclose(C.mindlin_torque(100.0, 0.5, 1e-4), 3 * math.pi * 0.5 * 100 * 1e-4 / 16)
    assert C.sliding_traction(0.35, 1.0) == (0.7, True) and not C.sliding_traction(0.2, 1.0)[1]


def test_oblique_impact_regimes_and_gross_slip():
    r = C.oblique_impact(10.0, 1.0, 0.1)
    assert math.isclose(r["kappa"], 0.824, abs_tol=1e-3) and math.isclose(r["chi"], 1.44, abs_tol=0.01)
    assert r["regime"] == "gross_slip"
    assert math.isclose(r["vt_out"], 10.0 - 0.1 * 2.0 * 3.5, rel_tol=0.03)
    assert C.oblique_impact(0.05, 1.0, 0.5)["regime"] == "stick"


def test_rayleigh_speed_and_halfspace_power():
    assert math.isclose(C.rayleigh_speed(1.0, 0.25), 0.919, abs_tol=2e-3)
    assert math.isclose(C.rayleigh_speed(1.0, 0.5), 0.955, abs_tol=2e-3)
    assert math.isclose(C.halfspace_radiated_power(1.0, 2.0, 1.0, 1.0), 0.296)


def test_regimes_moving_load_viscoelastic():
    assert C.impact_regime(7850, 0.01, 1e9, 115e9) == "elastic"
    assert C.impact_regime(7850, 5.0, 1e9, 115e9) == "plastic"
    assert C.impact_regime(7850, 1000.0, 1e9, 115e9) == "hypervelocity"
    assert C.moving_contact_ratio(1.0, 6000.0, 3200.0, 0.3) > 0.99
    assert C.moving_contact_ratio(2900.0, 6000.0, 3200.0, 0.3) > 1.0
    a = C.viscoelastic_contact_radius([0.0, 1e9], 1.0, 0.01, 2e6, 1e6, 1e3)
    assert math.isclose(a[0] ** 3, 3 / 8 * 0.01 / 2e6) and math.isclose(a[1] ** 3, 3 / 8 * 0.01 * 1.5e-6)
    assert math.isclose(C.hunter_restitution(1e-4, 1e-2), 1 - 4 / 9 * 0.01)
    assert math.isclose(C.rolling_microslip(1e-4, 0.01, beta=0.21), 1.5e-4 * 0.21 * 0.01)


def test_contact_gate_loses_contact_only_above_the_curvature_limit():
    dx, lam, amp, mass, N = 1e-4, 64, 1e-5, 0.02, 0.5
    z = amp * np.sin(2 * np.pi * np.arange(lam * 16) / lam)
    g_eff = N / mass + 9.81
    v_lim = lam * dx / (2 * np.pi) * math.sqrt(g_eff / amp)
    slow = C.contact_gate(z, np.full(SR // 4, 0.7 * v_lim), N, dx, SR, mass)
    fast = C.contact_gate(z, np.full(SR // 4, 2.0 * v_lim), N, dx, SR, mass)
    assert slow.min() == 1.0
    assert 1.0 - fast.mean() > 0.05 and fast[len(fast) // 2:].max() == 1.0      # flies, lands again


def test_wood_species_and_ono_norimoto():
    m = C.get_material("wood:padauk")
    assert math.isclose(m.density, 770.0) and math.isclose(m.young, 0.77 * 18.3e9)
    assert math.isclose(m.tan_phi, 5.2e-3) and m.name == "wood"
    assert math.isclose(C.wood_eta(20.0), 10 ** -1.23 * 20.0 ** -0.68)
    assert math.isclose(C.wood_eta(20.0, "softwood", "compression"), 1.3 * 10 ** -1.222 * 20.0 ** -0.639)
    a = C.impact(SR, material="wood:padauk", size=0.3)
    b = C.impact(SR, material="wood:sycamore", size=0.3)
    tail = lambda x: float(np.sum(x[SR // 4:] ** 2) / np.sum(x ** 2))
    assert tail(a) > tail(b)


def test_gammatone_bank_envelope_peaks_at_one_over_d_with_the_modal_amplitude():
    body = C.ModalObject(np.array([500.0]), np.array([50.0]), np.ones(1), mass=1.0)
    imp = np.zeros(SR // 2)
    imp[0] = SR                                     # 1 N s
    y = C.gammatone_bank(imp, body, SR, 0.0)
    ref = C.modal_bank(imp, body, SR, 0.0)[0]
    t_peak = np.argmax(np.abs(y)) / SR
    assert abs(t_peak - 1 / 50.0) < 2e-3
    assert math.isclose(np.max(np.abs(y)), np.max(np.abs(ref)), rel_tol=0.05)
    assert np.argmax(np.abs(ref)) / SR < 2e-3


def test_interaction_roll_wobble_follows_the_ball_radius():
    w = C.conan_weights(240.0, 1.0)
    f0 = C.conan_force(SR, SR, w, 0.5, 0.5, 0.3, 0, 1.0, 0.0)
    f1 = C.conan_force(SR, SR, w, 0.5, 0.5, 0.3, 0, 1.0, 0.03)
    ph = np.arange(1, SR + 1) / SR
    m0 = 1 + 0.3 * np.sin(2 * np.pi * 3 * 0.3 / 0.5 * ph)
    m1 = 1 + 0.3 * np.sin(2 * np.pi * 0.3 / (2 * np.pi * 0.03) * ph)
    assert np.allclose(f1 * m0, f0 * m1, atol=1e-9)


def test_measured_objects_materials_directivity_effective_mass():
    b = C._make_body("ri_iron_pan", "metal", 0.3)
    assert list(b.freqs) == [473.0, 1103.0, 1246.0]
    cu, ac = C.get_material("copper"), C.get_material("acrylic")
    assert (cu.young, cu.poisson) == (123.4e9, 0.34) and (ac.young, ac.poisson, ac.density) == (2.4e9, 0.37, 1200)
    g = C.directivity_gains(400, 37.0, 3)
    assert math.isclose(float(np.mean(g ** 2)), 1.0, rel_tol=0.1)
    assert g.min() >= 0.5 - 1e-12 and g.max() <= math.sqrt(1.75) + 1e-12
    a = C.impact(SR, azimuth=0.0, seed=1)
    c = C.impact(SR, azimuth=90.0, seed=1)
    assert not np.allclose(a, c)
    I = np.diag([2.0, 2.0, 2.0])
    assert C.effective_mass(3.0, I, [0, 0, 0], [0, 0, 1]) == 3.0
    assert math.isclose(C.effective_mass(3.0, I, [1.0, 0, 0], [0, 0, 1]), 1.0 / (1 / 3 + 0.5))


def test_eccentric_ball_kinematics():
    n, fs, R, m = 20000, 88200, 0.03, 0.1
    v = np.full(n, 0.5)
    off0, load0 = C._roll_stage(n, fs, v, R, m, 0.5, 0.0, 0.0, 0.0, 0)
    off1, load1 = C._roll_stage(n, fs, v, R, m, 0.5, 0.0, 0.0, 1.0, 0)
    assert np.allclose(load0, m * 9.81)
    om = 0.5 / R
    assert np.allclose(load1, m * 9.81)                  # the ball's inertia on the offset gives m r w^2 cos: not in the load too
    s0 = C._roll_stage(n, fs, v, R, m, 0.0, 0.0, 0.0, 0.0, 0)[0]
    s1 = C._roll_stage(n, fs, v, R, m, 0.0, 0.0, 0.0, 1.0, 0)[0]
    acc = np.diff(s1 - s0, 2) * fs * fs
    assert math.isclose(acc[0], 0.1 * R * om * om, rel_tol=1e-2)      # theta = 0: r w^2
    assert 0.0 < float(np.max(off1 - off0)) <= 2 * 0.1 * R + 1e-4


def test_moving_bank_follows_the_mode_shape_and_reduces_to_the_fixed_bank():
    body = C._make_body("bar", "steel", 0.3, 0.5, modes=6)
    x = np.random.default_rng(1).standard_normal(SR // 4)
    fixed = C.modal_bank(x, body, SR, 0.5)[0]
    same = C.moving_bank(x, body, SR, np.full(len(x), 0.5), 0.5)
    assert np.allclose(fixed, same, rtol=1e-6, atol=1e-9 * np.max(np.abs(fixed)))
    moved = C.moving_bank(x, body, SR, np.linspace(0.5, 0.05, len(x)), 0.5)
    assert not np.allclose(fixed, moved)
    varied = C.moving_bank(x, body, SR, None, 0.5, 0.2, 3)
    assert not np.allclose(fixed, varied)


def test_sample_modes_keeps_the_mean_spacing_and_varies_with_seed():
    ref = C.ShapeBody("rect_plate", 300.0, material="wood", n_modes=24)
    a = C.sample_modes("wood", 24, 300.0, seed=1)
    b = C.sample_modes("wood", 24, 300.0, seed=2)
    target = np.mean(np.diff(ref.freqs))
    for m in (a, b):
        assert abs(np.mean(np.diff(m.freqs)) / target - 1) <= 0.10
        assert np.all(np.diff(m.freqs) > 0) and m.freqs[0] == ref.freqs[0]
    assert not np.allclose(a.freqs, b.freqs)


def test_masked_modes_spreading_and_pruned_masking():
    m = C.masked_modes(np.array([1000.0, 1050.0, 4000.0]), np.array([1.0, 1e-4, 1e-4]))
    assert list(m) == [False, True, False]
    z = float(C.bark(1000.0))
    assert math.isclose(z, 13.0 * math.atan(0.76) + 3.5 * math.atan((1000.0 / 7500.0) ** 2))
    body = C.ModalObject(np.array([1000.0, 1050.0]), np.array([10.0, 10.0]), np.array([1.0, 1e-3]))
    assert len(body.pruned().freqs) == 2
    assert list(body.pruned(masking=True).freqs) == [1000.0]


def test_cristal_bend_glides_while_singing():
    from genny import friction as X
    y = X.cristal(277.2, 1.0, SR, bend=1.0, radiator="none")
    early = X._spectral_peak(y[int(0.05 * SR):int(0.2 * SR)], 277.2, fs=SR, span=0.1)
    late = X._spectral_peak(y[int(0.85 * SR):int(1.0 * SR)], 277.2 * 2 ** (1 / 12), fs=SR, span=0.1)
    assert 2 ** (0.65 / 12) < late / early < 2 ** (0.95 / 12)
    flat = X.cristal(277.2, 1.0, SR, radiator="none")
    assert np.allclose(flat, X.cristal(277.2, 1.0, SR, bend=0.0, radiator="none"))


def test_coil_spring_round_trip_from_geometry():
    from genny import friction as X
    from genny.fdstring import spring_geometry
    td = spring_geometry(1e-4, 2.5e-3, 112)["td"]
    c = math.sqrt(2.0e11 / 7850.0)
    assert 0.025 < td < 0.045 and math.isclose(td, 4 * (2 * math.pi * 2.5e-3 * 112) * 2.5e-3 / (1e-4 * c), rel_tol=0.02)
    a = X.coil_spring(220.0, 0.3, SR)
    assert np.allclose(a, X.coil_spring(220.0, 0.3, SR, wire=0.1, coil=2.5, turns=112))
    assert not np.allclose(a, X.coil_spring(220.0, 0.3, SR, turns=400))


def test_baschet_tables_from_the_thesis():
    from genny import friction as X
    assert X.ALU_BLADE_B == (1.0, 4.32, 5.45, 9.05, 13.53)
    assert X.IRON_BLADE == (1.0, 2.83, 6.35, 7.55) and math.isclose(X.IRON_BLADE[1], 1661.2 / 587.33, rel_tol=2e-3)
    assert X.SQUARE_TUBE == (1.0, 2.67, 5.04, 8.0)
    assert X.PIANO_ROD == (1.0, 2.0, 3.36, 6.73, 8.98)
    assert math.isclose(X.LAMINES_PAIR, (180 / 160) ** 2) and math.isclose(X.LAMINES_BLADE[1], 204 / 78)
    assert math.isclose(X.BRONTO_BAR[3], 500 / 55) and math.isclose(X.SAD_O[1], 84 / 44)
    assert X._BAR_MATERIAL["comb"][0][:4] == (1, 2, 3, 4)


def test_rod_bank_new_sets_strike_and_pair():
    from genny import friction as X
    f = 220.0
    sq = X.rod_bank(f, 0.6, SR, rod="square", sympathetic=0.0, radiator="none")
    assert _band_power(sq, SR, 2.67 * f) > 30 * _band_power(sq, SR, 2.756 * f + 40)
    tan = X.rod_bank(f, 0.6, SR, rod="square", sympathetic=0.0, radiator="none", strike="tangential")
    assert _band_power(tan, SR, 2.67 * f) < 1e-3 * _band_power(sq, SR, 2.67 * f) * (np.sum(sq ** 2) / np.sum(tan ** 2))
    pr = X.rod_bank(f, 0.6, SR, rod="paired", sympathetic=0.0, radiator="none")
    assert _band_power(pr, SR, X.LAMINES_PAIR * f) > 30 * _band_power(pr, SR, 1.13 * f)
    fx = X.rod_bank(f, 1.0, SR, rod="fixed", sympathetic=0.0, radiator="none")
    assert np.abs(fx[int(0.4 * SR):]).max() < 1e-3 * np.abs(fx).max()


def test_sheet_metal_raises_the_mode_ceiling():
    from genny import friction as X
    x = np.random.default_rng(2).standard_normal(SR // 2) * 0.1
    hi = lambda y: float(np.sum(np.abs(np.fft.rfft(y))[np.fft.rfftfreq(len(y), 1 / SR) > 8000] ** 2))
    a = X.sheet_radiator(x, SR, mix=1.0)
    b = X.sheet_radiator(x, SR, mix=1.0, metal="stainless")
    assert hi(b) > 10 * hi(a)


def test_cristal_harmonics_grow_with_pressure():
    from genny import friction as X
    def h2(p):
        y = X.cristal(277.2, 0.8, SR, pressure=p, radiator="none")[int(0.4 * SR):int(0.8 * SR)]
        return _band_power(y, SR, 2 * 277.2, 20) / _band_power(y, SR, 277.2, 20)
    assert h2(3.0) > 2.0 * h2(0.5)


def test_bowed_sheet_glide_and_musical_saw_settle_on_the_note():
    from genny import friction as X
    y = X.bowed_sheet(220.0, 1.0, SR, glide=-5.0, glide_time=0.4, ring=0.0)
    early = X._spectral_peak(y[int(0.03 * SR):int(0.13 * SR)], 220.0 * 2 ** (-4 / 12), fs=SR, span=0.2)
    late = X._spectral_peak(y[int(0.6 * SR):int(0.95 * SR)], 220.0, fs=SR, span=0.05)
    assert abs(late / 220.0 - 1) < 0.01 and early < 220.0 * 2 ** (-2 / 12)
    s = X.musical_saw(440.0, 1.2, SR, vibrato=0.0, radiator="none")
    e = X._spectral_peak(s[int(0.02 * SR):int(0.1 * SR)], 440.0 * 2 ** (-1.5 / 12), fs=SR, span=0.1)
    l = X._spectral_peak(s[int(0.6 * SR):int(1.1 * SR)], 440.0, fs=SR, span=0.05)
    assert abs(l / 440.0 - 1) < 0.01 and e < 440.0 * 2 ** (-0.7 / 12)


def test_struck_rod_sustained_partials_separate_from_attack():
    from genny import friction as X
    f1 = X.rod_frequency(0.3)
    y = X.struck_rod(0.3, 1.5, SR, sustain="c7")
    late = y[int(0.8 * SR):]
    late = late * np.hanning(len(late))                                  # a bare cut leaks the 65 Hz partial over 411 Hz
    assert _band_power(late, SR, X.ANGELS_TUBES["c7"][1] * f1) > 100 * _band_power(late, SR, 6.267 * f1)
    assert math.isclose(X.ANGELS_TUBES["a6"][1], 4698.6 / 1760.0, rel_tol=1e-3)


def test_collector_mass_and_sheet_mute():
    from genny import friction as X
    assert X.collector_loss(4.0) == 1.0 and X.collector_loss(3.0) == 1.0
    assert math.isclose(X.collector_loss(1.5), 0.25)
    a = X.rod_bank(220.0, 1.0, SR, sympathetic=0.0, radiator="none")
    assert np.allclose(a, X.rod_bank(220.0, 1.0, SR, sympathetic=0.0, radiator="none", collector=4.0))
    b = X.rod_bank(220.0, 1.0, SR, sympathetic=0.0, radiator="none", collector=1.0)
    assert np.sum(b[int(0.5 * SR):] ** 2) < 0.2 * np.sum(a[int(0.5 * SR):] ** 2)
    x = np.zeros(SR)
    x[100] = 1.0
    ring = X.sheet_radiator(x, SR, mix=1.0)
    muted = X.sheet_radiator(x, SR, mix=1.0, mute=1.0)
    assert np.sum(muted[SR // 4:] ** 2) < 0.1 * np.sum(ring[SR // 4:] ** 2)


def test_friction_helpers_b48():
    from genny import friction as X
    assert math.isclose(X.wineglass_f0(1.5e-3, 0.035, 0.08), 770.0, rel_tol=0.01)
    mu = X.akay_friction(np.array([-1.0, 0.0, 1e-6, 1.0]), 0.4, 0.25)
    assert mu[1] == 0.0 and abs(mu[2]) < 0.01 and math.isclose(mu[3], 0.25, rel_tol=1e-6) and mu[0] == -mu[3]
    assert math.isclose(X.stick_slip_number(2.0, 1000.0, 0.01, 0.1), (2.0 / 1000.0) / (0.01 / 100.0))
    env = X.coulomb_envelope(1000, 1000, 1.0, 5.0, 100.0, 10.0)
    assert math.isclose(env[100], 0.8) and env[600] == 0.0
    assert math.isclose(X.pulley_rod_freq(0.3, 0.1), math.sqrt(0.3 * 9.81 / 0.1) / (2 * math.pi))


def test_micro_impact_energy_counter():
    f, hits = C.micro_impact_force(np.zeros(100), 0.5, 0.1)
    assert hits == 0 and np.allclose(f, -0.1 * 9.81)
    z = np.zeros(100)
    z[10:] = 1e-4
    f, hits = C.micro_impact_force(z, 0.5, 0.1, depth=2.0, decay=4.0)
    assert hits == 1 and math.isclose(f[10], -0.1 * 9.81 + 1e-4 * 2.0 * 0.5 * 0.1 * 0.25 / 2.0)


def test_aramaki_boundaries_classify_the_reference_materials():
    assert C.classify_material(0.6, 2e-4) == "metal"
    assert C.classify_material(3.0, 4e-4) == "wood"
    assert C.classify_material(2.5, 1.5e-4) == "glass"
    assert math.isclose(float(C.critical_bandwidth(1000.0)), 25 + 75 * 2.4 ** 0.69)


def test_impact_rough_adds_sidebands_at_a_quarter_critical_band():
    body = C.ModalObject(np.array([1000.0]), np.array([5.0]), np.array([1.0]))
    am = C.am_sidebands(body, 0.8)
    fn = 0.25 * (25 + 75 * 2.4 ** 0.69)
    assert np.allclose(np.sort(am.freqs), [1000.0 - fn, 1000.0, 1000.0 + fn])
    assert np.allclose(am.phi[1:], 0.4)
    a = C.impact(SR, material="metal", shape="bar", seed=1)
    b = C.impact(SR, material="metal", shape="bar", seed=1, rough=0.8)
    assert not np.allclose(a[:len(b)], b[:len(a)])


def test_cristal_resonator_modes_and_fe_materials():
    from genny import friction as X
    assert math.isclose(X.CRISTAL_MODES[3], 293.3 / 277.7) and math.isclose(X.CRISTAL_MODES[0], 0.072, rel_tol=0.01)
    assert X.GLASS_FE == (7.0e10, 2300.0, 0.25) and X.ALUMINIUM_FE == (5.5e10, 2500.0, 0.25)
    y = X.cristal(277.2, 0.8, SR, modes=4, radiator="none")[int(0.3 * SR):]
    assert _band_power(y, SR, 277.2 * X.CRISTAL_MODES[3], 5) > 1e-6 * _band_power(y, SR, 277.2, 5)


def test_fracture_energy_budget_and_ellipsoid_proxy():
    from genny.physical import modal as M
    n, dek = M.fracture_budget(0.3, 6.0, 0.1, 80.0)
    es = 0.5 * 0.5 * 0.3 * 36.0
    af = 0.8 * es / 80.0
    assert n == round((1 + af / 0.03) ** 3) and math.isclose(dek, es - 80.0 * af)
    assert math.isclose(dek, 0.2 * es)
    assert M.fracture_budget(0.3, 12.0, 0.1, 80.0)[0] > n > M.fracture_budget(0.3, 6.0, 0.1, 200.0)[0]
    m, a, b, c = 2.0, 0.3, 0.2, 0.1
    inertia = (m * (b * b + c * c) / 5, m * (a * a + c * c) / 5, m * (a * a + b * b) / 5)
    assert np.allclose(M.ellipsoid_axes(inertia, m), [a, b, c])
    assert math.isclose(M.ellipsoid_gamma([a / 2, b / 2, c / 2], [a, b, c]), 0.5)
    y0 = M.fracture("sy_vase", np.random.default_rng(1), velocity=3.0, dur=0.6)
    y1 = M.fracture("sy_vase", np.random.default_rng(1), velocity=3.0, dur=0.6, mass=0.3, size=0.1)
    assert y0.shape == y1.shape and not np.allclose(y0, y1)


def test_hc_closed_form_rebound_close_to_exact():
    for x in (0.2, 1.0, 1.5, 6.0):
        exact = C.hc_rebound(0.5, x / 0.5)
        assert abs(C.hc_rebound_closed(0.5, x / 0.5) / exact - 1) < 0.01
    assert abs(C.hc_rebound_closed(0.5, 6.0) / C.hc_rebound(0.5, 6.0) - 1) < 0.025


def test_meso_impulses_from_slope_changes():
    z = np.array([0.0, 1e-4, 1e-4, 0.0])
    dx, sr, m = 1e-3, 1000, 0.02
    f = C.meso_force(z, np.full(8, 1.0), m, dx, sr)
    s = np.diff(np.concatenate([z, z[:1]])) / dx
    expected = [m * 1.0 * (s[i] - s[i - 1]) * sr for i in range(4)]
    hits = f[f != 0]
    assert 4 <= np.count_nonzero(np.diff(np.floor(np.cumsum(np.full(8, 1.0)) / sr / dx)) != 0) + 1
    assert hits.size >= 3 and all(any(math.isclose(h, e) for e in expected) for h in hits)


def test_directional_gains_and_material_restitution():
    a = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
    assert np.allclose(C.directional_gains(a, [0, 0, 2]), [3.0, 3.0])
    assert np.allclose(C.directional_gains(a, [1, 1, 0]), np.array([1.0, 2.0]) / math.sqrt(2))
    assert C.material_restitution("steel") == 0.75 and C.material_restitution("rubber") == 0.85
    assert C.material_restitution("wood:maple") == 0.5 and C.material_restitution("unknown") == 0.6
    a1 = C.drop(SR, material="rubber", restitution=0.0, roll=0.0)
    a2 = C.drop(SR, material="rubber", restitution=0.85, roll=0.0)
    assert np.allclose(a1, a2)


def test_impact_freq_and_decay_factors():
    a = C.impact(SR, material="metal", seed=2)
    b = C.impact(SR, material="metal", seed=2, decay_factor=50.0)
    assert len(b) < len(a)
    c = C.impact(SR, material="metal", seed=2, freq_factor=1.0, decay_factor=1.0)
    assert np.allclose(a, c)


def test_addendum_contact_helpers():
    b, d = C.winkler_roller(1000.0, 0.05, 1e8)
    assert math.isclose(2 * 1e8 * b ** 3 / (3 * 0.05), 1000.0) and math.isclose(d, b * b / 0.1)
    assert math.isclose(C.tangential_stiffness(80e9, 1e-3, 0.3), 8 * 80e9 * 1e-3 / 1.7)
    assert math.isclose(C.torsional_stiffness(80e9, 1e-3), 16 * 80e9 * 1e-9 / 3)
    assert math.isclose(C.relative_radius(0.01, -0.02), 0.02) and C.relative_radius(0.01, 0.01) == 0.005
    rho = C.crater_radius(1e-3, 0.01, 100.0, 1e11)
    assert math.isclose(4e-9 * (1 / 0.01 - 1 / rho), 3 * 100.0 / 1e11)
    a = C.mud_contact_radius(0.02, 50.0, 10.0, np.array([1.0, 8.0]))
    assert math.isclose(a[1] / a[0], 2.0) and math.isclose(a[0] ** 3, 3 * 0.02 * 10.0 / (16 * 50.0))


def test_hollow_bar_and_brass():
    from genny import physics as P
    solid = P.BarBody(radius_m=0.01).modes().freqs[0]
    tube = P.BarBody(radius_m=0.01, inner_radius_m=0.009).modes().freqs[0]
    assert math.isclose(tube / solid, math.sqrt(0.01 ** 2 + 0.009 ** 2) / 0.01, rel_tol=1e-6)
    assert math.isclose(P.BarBody(radius_m=None, thickness_m=0.01).modes().freqs[0],
                        P.BarBody(radius_m=None, width_m=0.01).modes().freqs[0])
    assert P.MATERIALS["brass"].density == 8500 and P.MATERIALS["brass"].young_modulus == 110e9


def test_johnson_damage_number_regimes():
    assert C.johnson_regime(7850.0, 0.01, 250e6) == "elastic"
    assert C.johnson_regime(7850.0, 1.0, 250e6) == "elastoplastic"
    assert C.johnson_regime(7850.0, 50.0, 250e6) == "plastic"
    assert C.johnson_regime(7850.0, 1000.0, 250e6) == "hypervelocity"


def test_swing_measured_sources_and_new_presets():
    from genny.physical import modal as pm
    for name in ("BROOM", "WOODSWD"):
        y = pm.sword_swing(name)
        assert y.ndim == 2 and np.all(np.isfinite(y)) and np.max(np.abs(y)) > 0
    assert pm.SWING_PRESETS["BROOM"]["Source"] == "broom"
    assert "Source" not in pm.SWING_PRESETS["SWD1"]
    assert abs(pm.swing_peak_tone("BROOM") - pm.swing_peak_tone("WOODSWD")) > 50


def test_avalanche_rise_is_a_fifth_of_the_length():
    from genny import contact
    y = contact.avalanche(dur=5.0)
    env = np.abs(y)
    k = len(env) // 20
    assert np.mean(env[:k]) < np.mean(env[4 * k:6 * k])
