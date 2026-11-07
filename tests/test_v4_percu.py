"""V4_PERCU: modal von Karman plates, cymbal damping laws, tam-tam, gong glide, triangle bank, Berger membrane,
TR-808 decay loop, output filter, toms and congas."""
import math

import numpy as np

SR = 44100


def _peak_hz(y, sr, lo, hi):
    n = 1 << (len(y) - 1).bit_length()
    sp = np.abs(np.fft.rfft(y * np.hanning(len(y)), n))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    b = (f > lo) & (f < hi)
    return f[b][np.argmax(sp[b])]


def test_vk_damping_laws_and_t60():
    from genny.physical import vkmodal as V
    w = 2 * math.pi * np.array([100.0, 1000.0, 10000.0])
    t60 = 13.82 / V.damping(w, "crash")
    assert np.allclose(t60, [5.93, 4.15, 1.89], atol=0.03)
    assert abs(13.82 / V.damping(2 * math.pi * 100.0, "gong") - 57.8) < 0.5
    m = 7850.0 * 1.2e-3
    assert abs(V.damping(1000.0, "rectangle", m) - (0.004 * 1000.0 ** 0.75 + 0.08) / m) < 1e-12


def test_vk_thickness_profiles():
    from genny.physical import vkmodal as V
    h = V.thickness(np.array([0.0, 0.25, 1.0]), "nguyen")
    assert np.allclose(h, [1e-3, 1e-3, 0.7e-3])
    assert abs(V.thickness(1.0, "nguyen_thin") - 0.4e-3) < 1e-12
    assert abs(V.thickness(0.5, "uniform", 2e-3) - 2e-3) < 1e-12


def test_vk_ritz_basis_matches_supported_plate_modes():
    from genny.physical import vkmodal as V
    rho, E, nu = 7850.0, 2e11, 0.3
    m = V.build(0.553, 1.2, 1.5, "ssss", (rho, E, nu), modes=6, airy=4, grid=30, basis="ritz", law=None)
    Lx, Ly = m.Lx, m.Ly
    D = E * 1.2e-3 ** 3 / (12 * (1 - nu * nu))
    p, q = np.meshgrid(np.arange(1, 5), np.arange(1, 5), indexing="ij")
    w = np.sort((np.sqrt(D / (rho * 1.2e-3)) * ((p * math.pi / Lx) ** 2 + (q * math.pi / Ly) ** 2)).ravel())[:4]
    assert np.allclose(m.w[:4], w, rtol=0.03)


def test_vk_variable_thickness_lowers_the_modes():
    from genny.physical import vkmodal as V
    a = V.build(0.4, 1.0, 1.0, "ssss", profile="uniform", modes=6, airy=4, grid=24, law=None)
    b = V.build(0.4, 1.0, 1.0, "ssss", profile="nguyen_thin", modes=6, airy=4, grid=24, law=None)
    assert b.w[0] < 0.95 * a.w[0]


def test_vk_dome_stiffens_the_linear_modes():
    from genny.physical import vkmodal as V
    flat = V.build(0.3, 1.2, 1.0, "ssss", modes=8, airy=8, grid=24, law=None)
    shell = V.build(0.3, 1.2, 1.0, "ssss", dome=0.01, modes=8, airy=8, grid=24, law=None)
    assert np.allclose(V.linear_freqs(flat), flat.w / (2 * math.pi), rtol=1e-6)
    assert V.linear_freqs(shell)[0] > 1.2 * flat.w[0] / (2 * math.pi)


def test_vk_schemes_conserve_energy_without_loss():
    from genny.physical import vkmodal as V
    m = V.build(0.3, 1.0, 1.0, "ssss", dome=0.005, modes=10, airy=6, grid=20, law=None)
    fs = 20000.0
    f = np.zeros(4000)
    f[:20] = 200.0
    for scheme in ("sav", "conservative"):
        _, e = V.simulate(m, f, fs, scheme=scheme, energy=True, lossless=True)
        tail = e[100:]
        assert np.max(np.abs(tail - tail[0])) < 1e-8 * abs(tail[0])


def test_vk_flat_gong_glides_down_curved_glides_up():
    from genny.physical import vkmodal as V
    fs = 16000.0
    out = {}
    for dome in (0.0, 0.02):
        m = V.build(0.2, 1.0, 1.0, "ssss", dome=dome, modes=12, airy=8, grid=20, law="gong")
        f = np.zeros(int(1.5 * fs))
        f[:40] = 3000.0 if dome == 0 else 1500.0
        y = V.simulate(m, f, fs)[:, 0]
        f0 = V.linear_freqs(m)[0]
        tr = V.instantaneous_freq(y, fs, f0)
        out[dome] = tr[-1] - tr[1]
    assert out[0.0] < 0.0 < out[0.02]


def test_vk_chaos_measures():
    from genny.physical import vkmodal as V
    t = np.arange(4000) / 8000.0
    assert abs(V.correlation_dimension(np.sin(2 * math.pi * 50 * t), dim=4) - 1.0) < 0.3
    x = np.random.default_rng(0).standard_normal(4000)
    assert V.correlation_dimension(x, dim=4) > 3.0


def test_cymbal_presets_follow_the_crash_damping_law():
    from genny import plates as PL
    d, fc, t = PL._law(6.9, PL.CYMBAL_LOSS)
    assert (d, fc) == (6.9, 1000.0) and abs(t - 4.15) < 1e-9
    assert PL.TAMTAM_LOSS == (75.0, 8000.0, 4.0)
    from genny.drums import REGISTRY
    assert REGISTRY["plate_crash"]["params"]["decay"][0] == 6.9
    assert REGISTRY["tam_tam"]["params"]["decay"][0] == 75.0 and REGISTRY["tam_tam"]["params"]["dur"][0] == 8.0


def test_vk_cymbal_wash_moves_up_after_the_hit():
    from genny import plates as PL
    y = PL.vk_cymbal(SR, 1.0, "crash", dur=0.3, modes=32)

    def hi_share(a, b):
        s = np.abs(np.fft.rfft(y[int(a * SR):int(b * SR)])) ** 2
        f = np.fft.rfftfreq(int(b * SR) - int(a * SR), 1.0 / SR)
        return s[f > 3000].sum() / s.sum()
    assert hi_share(0.05, 0.1) > hi_share(0.01, 0.02)


def test_gong_pitch_rises_80_cents_at_full_velocity():
    from genny import percussion as P
    t = np.arange(SR * 6) / SR
    x = np.sin(2 * math.pi * 200.0 * t)
    y = P.pitch_glide(x, SR, 80.0, 4.0)
    early = _peak_hz(y[:int(0.2 * SR)], SR, 150, 250)
    late = _peak_hz(y[int(4.5 * SR):int(5.5 * SR)], SR, 150, 250)
    assert abs(1200 * math.log2(late / early) - 80.0) < 15.0
    assert abs(late - 200.0) < 1.0


def test_shell_freqs_add_the_ring_frequency():
    from genny import percussion as P
    rho, E, nu = P.BRONZE_SHELL
    fr = math.sqrt(E / (rho * (1 - nu * nu))) / (2 * math.pi * 1.0)
    assert np.allclose(P.shell_freqs([0.0, 300.0], 1.0), [fr, math.hypot(300.0, fr)])


def test_triangle_dense_bank_peaks_near_6_khz():
    from genny import percussion as P
    fr, am = P.triangle_modes(400.0, 20000.0)
    assert len(fr) == 30 and abs(fr[0] - 400.0) < 1e-9                    # 10 bar modes under 20 kHz, three each
    assert 4500.0 < fr[np.argmax(am)] < 8000.0
    y = P.triangle(SR, 1.0, dur=1.0)
    assert 4500.0 < _peak_hz(y[:int(0.3 * SR)], SR, 300, 16000) < 8000.0


def test_berger_membrane_starts_sharp_and_detune_splits_pairs():
    from genny import plates as PL
    lin = PL.membrane(SR, 1.0, tension=1000.0, size=0.33, decay=1.0, dur=0.6)
    nl = PL.membrane(SR, 1.0, tension=1000.0, size=0.33, decay=1.0, dur=0.6, berger=True, berger_force=2000.0)
    f0 = _peak_hz(lin[int(0.2 * SR):], SR, 50, 400)
    assert _peak_hz(nl[:int(0.08 * SR)], SR, 50, 400) > 1.01 * f0
    assert abs(math.sqrt((1 + 0.07) / (1 - 0.07)) - 1.0726) < 1e-3


def test_tr808_output_filter_is_the_table_b1_cascade():
    from genny import drums as D
    x = np.zeros(SR)
    x[0] = 1.0
    y = D.bd_output_filter(x, SR, tone=1.0, level=1.0)
    h = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(SR, 1.0 / SR)
    p = D.TR808_BD_OUTPUT
    mid = h[np.argmin(np.abs(f - 30.0))]                      # passband: highpasses under 4 Hz, tone lowpass at 305 Hz
    assert abs(mid - p["R176"] / (p["R176"] + p["R177"])) < 0.03
    fc_lp = 1 / (2 * math.pi * (p["R171"] + p["R172"] * p["VR5"] / (p["R172"] + p["VR5"])) * p["C45"])
    assert abs(h[np.argmin(np.abs(f - fc_lp))] / mid - 1 / math.sqrt(2)) < 0.05


def test_tr808_decay_knob_lengthens_the_ring():
    from genny import drums as D
    x = np.zeros(SR)
    x[:44] = 4.0
    short = D.bd_resonator(x, SR, knob=0.05)
    long = D.bd_resonator(x, SR, knob=1.0)
    e = lambda y: np.sqrt(np.mean(y[int(0.5 * SR):int(0.6 * SR)] ** 2)) / np.max(np.abs(y))
    assert e(long) > 2.0 * e(short)


def test_tr808_tom_knob_and_conga_tuning():
    from genny import drums as D
    assert abs(D.tom_r1(0.0) - 500.0) < 1e-9 and abs(D.tom_r1(1.0) - 333.33) < 0.01
    fc_lo = D.bridged_t("mid_tom", D.tom_r1(0.0))[0]
    fc_hi = D.bridged_t("mid_tom", D.tom_r1(1.0))[0]
    assert abs(fc_hi / fc_lo - math.sqrt(500.0 / 333.333)) < 1e-3
    y = D.conga808(SR, 1.0, "high", 1.0)
    assert abs(_peak_hz(y, SR, 100, 2000) / D.bridged_t("high_conga", D.tom_r1(1.0))[0] - 1.0) < 0.02


def test_plate_acoustics_helpers():
    from genny import plates as PL
    steel = (7850.0, 2e11, 0.3)
    cl = math.sqrt(2e11 / (7850.0 * 0.91))
    assert abs(PL.modal_density(1.0, 1e-3, steel) - math.sqrt(3.0) / (cl * 1e-3)) < 1e-9
    base = PL.prestressed_modes(0.4, 0.6, 1e-3, steel, n=1)[0]
    tens = PL.prestressed_modes(0.4, 0.6, 1e-3, steel, tx=1e4, n=1)[0]
    kx2 = (math.pi / 0.4) ** 2
    assert abs((2 * math.pi * tens) ** 2 - (2 * math.pi * base) ** 2 - 1e4 * kx2 / (7850.0 * 1e-3)) < 1e-6 * tens ** 2
    assert abs(PL.rayleigh_t60([1000.0], "abs")[0] - 6.91 / (0.5 * 0.02 * 2 * math.pi * 1000.0)) < 1e-6
    fc = PL.coincidence(1e-3, steel)
    assert 11000.0 < fc < 13500.0
    assert PL.radiation_efficiency(2 * fc, 1e-3, 1.0, 4.0, steel) == 1.0
    assert PL.radiation_efficiency(0.1 * fc, 1e-3, 1.0, 4.0, steel) < 0.2
    x = np.zeros(SR)
    x[0] = 1.0
    h = np.abs(np.fft.rfft(PL.point_force_pressure(x, SR, 1.0)))
    f = np.fft.rfftfreq(SR, 1.0 / SR)
    hi = h[np.argmin(np.abs(f - 2000.0))]
    assert abs(hi - PL.RHO_AIR / (2 * math.pi)) < 0.01 * hi
    assert abs(h[np.argmin(np.abs(f - PL.RHO_AIR * PL.C_AIR / (2 * math.pi)))] / hi - 1 / math.sqrt(2)) < 0.05
    nf = np.abs(np.fft.rfft(PL.near_field(x, SR, 0.5)))
    assert nf[np.argmin(np.abs(f - 40.0))] > 2.0 * nf[np.argmin(np.abs(f - 2000.0))]


def test_bodhran_hand_damping_shortens_it():
    from genny import plates as PL
    o = PL.bodhran(SR, 1.0, damp=0.0)
    d = PL.bodhran(SR, 1.0, damp=1.0)
    tail = lambda y: np.sqrt(np.mean(y[int(0.4 * SR):int(0.5 * SR)] ** 2)) / np.max(np.abs(y))
    assert tail(o) > 3.0 * tail(d)
    assert 50.0 < _peak_hz(o[int(0.1 * SR):], SR, 30, 200) < 90.0


def test_immersed_tube_goes_flat_by_its_added_mass():
    from genny import percussion as P
    r, load = P.immersed_ratio(2, 1.0)
    ratio = 1000.0 * (0.019 ** 2 + 0.0175 ** 2) / (2700.0 * (0.019 ** 2 - 0.0175 ** 2))
    assert abs(load - ratio) < 1e-6 and abs(r - 1 / math.sqrt(1 + ratio)) < 1e-9
    assert P.immersed_ratio(2, 0.0) == (1.0, 0.0)
    assert P.immersed_ratio(2, 0.3)[0] < P.immersed_ratio(2, 0.1)[0]
    y = P.water_tube(SR, 1.0, 440.0, 0.0, 0.0, 1.0)
    assert abs(_peak_hz(y, SR, 300, 600) - 440.0) < 3.0


# ---------------------------------------------------------------------------------------------------------
# V4_PERCU2 (passage 2): polar grids, isotropic schemes, DST Airy, double miking, Perrin bell, tubes, door
# ---------------------------------------------------------------------------------------------------------
def test_gong_is_registered_once():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "genny" / "percussion.py").read_text(encoding="utf-8")
    assert src.count('@drum("gong"') == 1


def test_polar_membrane_fundamental_is_the_first_bessel_zero():
    from genny.physical import fdgrid as G
    m = G.polar_model(48000, "membrane", 16, 32, gamma=100.0, alpha=0.0)
    f = G.polar_modes(m, 1)[0]
    assert abs(f / (100.0 * 2.4048 / (2 * math.pi)) - 1.0) < 0.03


def test_polar_clamped_plate_fundamental():
    from genny.physical import fdgrid as G
    m = G.polar_model(48000, "plate", 16, 32, "clamped", "free", kappa=1.0, alpha=0.0)
    assert abs(G.polar_modes(m, 1)[0] / (10.2158 / (2 * math.pi)) - 1.0) < 0.05


def test_polar_free_plate_first_elastic_mode():
    from genny.physical import fdgrid as G
    m = G.polar_model(48000, "plate", 16, 32, "free", "free", kappa=1.0, alpha=0.0, nu=0.3)
    assert np.all(m.freqs[:3] < 0.2 * m.freqs[3])
    assert abs(G.polar_modes(m, 1)[0] / (5.26 / (2 * math.pi)) - 1.0) < 0.08


def test_shell_curvature_empties_the_bottom_of_the_spectrum():
    from genny.physical import fdgrid as G
    flat = G.polar_model(32000, "shell", 12, 24, "free", "clamped", kappa=5.0, q=0.0)
    deep = G.polar_model(32000, "shell", 12, 24, "free", "clamped", kappa=5.0, q=60.0)
    thr = flat.freqs[9]
    assert np.sum(deep.freqs <= thr) < 10
    assert deep.freqs[0] > flat.freqs[0]


def test_shell_stability_number():
    from genny.physical import fdgrid as G
    k = 1.0 / 32000
    assert abs(G.shell_bound(50.0, k, 0.1, 0.2) - 2 * 50 * k * 26.0 / 0.01) < 1e-9
    assert abs(G.shell_bound(50.0, k, 0.1, 0.2, centre="clamped") - 2 * 50 * k * 7.25 / 0.01) < 1e-9
    assert G.shell_bound(50.0, k, 0.1, 0.2, alpha=0.25) == 0.0
    assert G.shell_bound(50.0, k, 0.1, 0.2, q=2.0 / (k * 50.0)) == math.inf


def test_polar_shell_crashes_brighter_when_hit_hard():
    from genny import percussion as P

    def centroid(y):
        s = np.abs(np.fft.rfft(y))
        f = np.fft.rfftfreq(len(y), 1.0 / SR)
        return float(np.sum(f * s) / np.sum(s))
    soft = P.polar_shell(SR, 0.2, "cymbal", dur=0.5, rings=8)
    hard = P.polar_shell(SR, 1.0, "cymbal", dur=0.5, rings=8)
    assert centroid(hard) > 1.05 * centroid(soft)


def test_isotropic_membrane_schemes_table_11_1():
    from genny.physical import fdgrid as G
    c5 = G.iso_cents(1, 4, 16000, gamma=1000.0)[2]
    c9 = G.iso_cents(1, 4, 16000, gamma=1000.0, alpha=2 / math.pi)[2]
    ci = G.iso_cents(1, 4, 16000, gamma=1000.0, alpha=2 / math.pi, theta=1.2)[2]
    assert abs(c5 + 44.5) < 3.0
    assert abs(ci) < abs(c9) < abs(c5) and abs(c9) < 20.0 and abs(ci) < 5.0


def test_explicit_plate_dispersion_table_12_1():
    from genny.physical import fdgrid as G
    assert abs(G.iso_cents(2, 2, 44100, kappa=100.0)[2] + 54.9) < 3.0
    assert G.iso_cents(2, 5, 44100, kappa=100.0)[2] < G.iso_cents(2, 2, 44100, kappa=100.0)[2]


def test_implicit_iso_membrane_rings_at_its_predicted_frequency():
    from genny.physical import fdgrid as G
    fs = 8000
    f = np.zeros(fs)
    f[0] = 1.0
    y = G.iso_simulate(fs, f, gamma=200.0, alpha=2 / math.pi, theta=1.2, exc=(0.5, 0.5), pick=(0.5, 0.5))
    pred = G.iso_cents(1, 1, fs, gamma=200.0, alpha=2 / math.pi, theta=1.2)[0]
    assert abs(_peak_hz(y, fs, 80, 250) - pred) < 2.0


def test_dst_airy_inverts_the_supported_biharmonic():
    import scipy.sparse as sp
    from genny.physical import fdgrid as G
    nx, ny, hx, hy = 9, 7, 0.1, 0.15

    def d2(n, h):
        return sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n - 1, n - 1)) / h ** 2
    L = sp.kron(d2(nx, hx), sp.identity(ny - 1)) + sp.kron(sp.identity(nx - 1), d2(ny, hy))
    u = np.random.default_rng(3).standard_normal((nx - 1) * (ny - 1))
    q = (L @ (L @ u)).reshape(nx - 1, ny - 1)
    S, lam = G.dst_factors(nx, ny, hx, hy)
    assert np.allclose(G.airy_dst(np.ascontiguousarray(q), S, lam, hx).ravel(), u, atol=1e-8)


def test_snare_wire_and_air_grid_spacings():
    from genny.physical import fdgrid as G
    k = 1.0 / 44100
    hm, hs, ha = G.snare_spacings(k, 3000.0, 0.262, 100.0, 1e-3, sig1=0.001, sig_air=1e-6)
    assert abs(hm - math.sqrt(2 * k * k * 3000.0 / 0.262 + 8 * 0.001 * k)) < 1e-12
    assert abs(hs - k * math.sqrt(100.0 / 1e-3)) < 1e-12
    assert abs(ha - math.sqrt(3 * 343.0 ** 2 * k * k + 6 * 343.0 * 1e-6 * k)) < 1e-12


def test_double_mic_cancels_lows_unless_flipped():
    from genny import percussion as P
    t = np.arange(SR) / SR
    y = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 5000 * t)

    def ratio(x):
        return abs(np.dot(x, np.sin(2 * np.pi * 50 * t))) / abs(np.dot(x, np.sin(2 * np.pi * 5000 * t)))
    r0 = ratio(y)
    assert ratio(P.double_mic(y, SR, 1.0, False)) < 0.45 * r0
    assert ratio(P.double_mic(y, SR, 1.0, True)) > 1.3 * r0
    assert P.double_mic(y, SR, 0.0) is y


def test_perrin_bell_measured_modes_and_doublets():
    from genny import percussion as P
    parts = P.perrin_partials(292.72)
    f = [p[0] for p in parts]
    assert len(P.PERRIN_MODES) == 63 and abs(f[0] - 292.72) < 1e-9
    assert any(abs(x - 1171.4) < 1e-6 for x in f) and any(abs(x - 1172.0) < 1e-6 for x in f)
    assert any(abs(x - 4401.0) < 1e-6 for x in f) and max(f) == 9254.0
    y = P.perrin_bell(SR, 1.0, 292.72, 4.0)
    assert abs(_peak_hz(y, SR, 200, 400) - 292.72) < 1.0


def test_sphere_on_bar_end_loses_energy_into_the_bar():
    from genny import percussion as P
    t, f = P.bar_end_impact(1.0)
    m = 7850.0 * 4 / 3 * math.pi * 0.005 ** 3
    J = float(np.sum(f) * (t[1] - t[0]))
    assert m * 1.0 < J < 2.0 * m * 1.0
    assert 5e-6 < t[-1] < 2e-4


def test_rational_radiation_impedance_and_end_reflection():
    from genny import tubes as T
    ka = 1e-3
    assert abs(T.radiation_impedance(ka, "rayleigh1").imag / ka - 0.8488) < 1e-3
    assert abs(T.radiation_impedance(ka, "levine2").imag / ka - 0.6133) < 1e-3
    assert abs(T.radiation_impedance(1e3, "rayleigh2") - 1.0) < 1e-2
    for end, d in (("flanged", 0.8215), ("unflanged", 0.613)):
        r = T.end_reflection(ka, end)
        assert abs(np.angle(-r) / (-2 * ka) - d) < 2e-3
    assert abs(T.end_reflection(1e3, "unflanged")) < 0.01
    b, a = T.end_reflection_filter(0.01, SR)
    assert abs(np.sum(b) / np.sum(a) + 1.0) < 1e-9


def test_2d_mesh_quarter_wave_tube():
    from genny import tubes as T
    mask = np.zeros((42, 5), dtype=np.bool_)
    mask[:, 1:4] = True
    fs = 343.0 * math.sqrt(2.0) / T.MESH2D_DX
    src = np.zeros(int(0.2 * fs))
    src[0] = 1.0
    y = T._mesh2d(src, mask, 1.0)
    assert abs(_peak_hz(y, fs, 250, 900) / (343.0 / (4 * 42 * T.MESH2D_DX)) - 1.0) < 0.06
    w = T.mesh2d_shape(np.full(10, math.pi * 1.0), 0.16)
    assert w.shape[0] == 40 and w.sum(axis=1).max() == 5


def test_box_resonator_tuned_drains_the_bar():
    from genny import tubes as T
    tuned = T.bar_resonator(440.0, 440.0, SR, 1.0)[0]
    off = T.bar_resonator(440.0, 440.0 * 2 ** (200 / 1200), SR, 1.0)[0]
    tail = slice(SR // 2, SR)
    assert np.sqrt(np.mean(tuned[tail] ** 2)) < 0.5 * np.sqrt(np.mean(off[tail] ** 2))
    assert abs(T.box_length(440.0) - (343.0 / 1760.0 - 0.6 * 0.078)) < 1e-12


def test_door_leaf_membrane_modes():
    from genny import foley as Fo
    f = Fo.door_modes(0.9, 3.0, 340.0)
    assert abs(f[0] - 170.0 * math.hypot(1 / 0.9, 1 / 3.0)) < 1e-9
    assert len(f) == 7 and np.all(np.diff(f) >= 0)
    a = Fo.creak(SR, seed=1)
    b = Fo.creak(SR, seed=1, width=0.9, height=3.0)
    assert a.shape == b.shape and not np.allclose(a, b)
