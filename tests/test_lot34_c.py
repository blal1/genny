"""Lot 34 C: Cordes (Lecture profonde 3.4, l.1661-1763)."""
import math

import numpy as np

from genny import acoustic, analysis, dsp, fdstring, networks, spec
from genny.physical import modal, modal_data as md, piano_data

SR = 22050


def test_table_i_rows_are_consistent():
    for key, f1 in (("C2", 65.4), ("C4", 261.6), ("C7", 2093.0)):
        r = piano_data.BENSA_TABLE_I[key]
        assert abs(r["c"] / (2.0 * r["L"]) - f1) / f1 < 2e-3
        f, s = piano_data.stiff_string_mode(key, 1)
        assert abs(f - f1) / f1 < 0.01
        assert abs(s - (r["b1"] + r["b2"] * (math.pi / r["L"]) ** 2)) < 1e-9


def test_scheme_matches_the_pde_of_table_i():
    for key, pmax, cents, ratio in (("C2", 8, 1.0, 0.01), ("C4", 5, 2.0, 0.01), ("C7", 2, 6.0, 0.03)):
        for p, fp, fs, dc, tp, ts in fdstring.table_i_check(key):
            if p <= pmax:
                assert abs(dc) < cents, (key, p, dc)
                assert abs(ts / tp - 1.0) < ratio, (key, p, tp, ts)


def test_body_interp_corners_and_middle():
    assert np.allclose(modal.body_interp(0, 0)["freqs"], md.BODY_SQUARESMALL["freqs"])
    assert np.allclose(modal.body_interp(2, 2)["t60"], md.BODY_ROUNDBIG["t60"])
    assert np.allclose(modal.body_interp(1, 1)["gains"][0], md.BODY_SEMMID["gains"][0])
    mid = modal.body_interp(0.5, 0)["freqs"]
    assert np.allclose(mid, 0.5 * (np.asarray(md.BODY_SQUARESMALL["freqs"]) + np.asarray(md.BODY_SEMSMALL["freqs"])))
    m = modal.resolve(modal.body_interp(1.5, 0.5))
    assert len(m.freqs) > 20 and np.all(np.isfinite(m.decays))


def test_passive_bridge_is_passive():
    z = np.exp(1j * np.linspace(0, math.pi, 4000))
    F, B = [96.52, 189.33, 219.95, 800.0], [50.0] * 4
    for method in (1, 2):
        b, a = networks.passive_bridge(F, B, 0.9, method, sr=8000)
        rho = np.polyval(b[::-1], 1 / z) / np.polyval(a[::-1], 1 / z)
        assert np.max(np.abs(rho)) < 1.0, method
        assert np.min(((1 - rho) / (1 + rho)).real) > 0.0, method
        if method == 1:
            assert np.max(np.abs(rho)) <= 0.9 + 1e-6
            assert abs(np.min(((1 - rho) / (1 + rho)).real) - 0.1 / 1.9) < 1e-6


def test_loaded_junction_limits():
    v = np.array([0.3, -0.2, 0.5])
    R = np.full(3, 0.65)
    rigid, vj = networks.loaded_junction(v, R, 1e9)
    assert abs(vj) < 1e-8 and np.allclose(rigid, -v, atol=1e-8)
    free, vj = networks.loaded_junction(np.array([0.4]), np.array([0.65]), 0.0)
    assert np.allclose(free, 0.4) and abs(vj - 0.8) < 1e-12
    out, _ = networks.loaded_junction(np.random.default_rng(1).standard_normal((50, 3)), R, 2.0)
    assert out.shape == (50, 3)


def test_loaded_sympathetic_piano_renders():
    p = networks.SympatheticPiano(struck_frequency=261.6, sr=SR, junction="loaded")
    y = p.render(0.3)
    assert np.all(np.isfinite(y)) and np.max(np.abs(y)) > 0.1
    mix = networks.SympatheticPiano(struck_frequency=261.6, sr=SR).render(0.3)
    assert np.max(np.abs(y - mix)) > 0.05


def test_weinreich_coupling_matches_eta_at_c4():
    c = networks.CoupledStringBank.weinreich_coupling(261.63, 3)
    eta = -261.63 * math.log(1.0 - c) / 3.0
    assert abs(eta - 1.05) / 1.05 < 0.2
    bank = networks.CoupledStringBank.from_bridge(261.63, sr=SR)
    assert abs(bank.coupling - c) < 1e-12
    y = bank.render(0.2)
    assert np.all(np.isfinite(y))


def test_ppm_meter_ballistics():
    sr = 8000
    for kind, t_int, db_s in (("I", 0.010, 20.0 / 1.7), ("IIa", 0.005, 24.0 / 1.5), ("IIb", 0.005, 24.0 / 2.8)):
        x = np.concatenate((np.ones(sr * 4), np.zeros(sr * 3)))
        y = dsp.ppm_meter(x, sr, kind)
        assert abs(y[int(t_int * sr)] - 0.8) < 0.04
        i0 = sr * 4
        drop = -20 * math.log10(y[i0 + sr] / y[i0 - 1])
        assert abs(drop - db_s) / db_s < 0.02


def test_koto_and_sitar_air_loss():
    for name in ("koto", "sitar"):
        fn = getattr(acoustic, name)
        a = fn(220.0, 0.5, SR, 0.8)
        b = fn(220.0, 0.5, SR, 0.8, loss="air")
        assert np.all(np.isfinite(b)) and len(analysis.detect_clicks(b, SR)) == 0
        assert np.max(np.abs(a[:min(len(a), len(b))] - b[:min(len(a), len(b))])) > 1e-3


def test_piano_action_noise_decays_in_the_attack():
    base = []
    for lvl in (0.0, 1.0):
        p = __import__("genny.piano", fromlist=["x"]).PolyphonicPiano(sr=SR, action_level=lvl)
        p.note_on(60, 0.9)
        base.append(p.render(0.2))
    d = base[1] - base[0]
    n = SR // 50
    assert np.sqrt(np.mean(d[:n] ** 2)) > 5.0 * np.sqrt(np.mean(d[-n:] ** 2)) > 0.0
    assert np.sqrt(np.mean(d[:n] ** 2)) < 0.5 * np.sqrt(np.mean(base[0][:n] ** 2)) * 4


def test_seq_pedal_rings_past_the_note():
    st = [{"note": "C3", "dur": 0.2}]
    off = spec.render_sequence({"inst": "piano", "steps": st}, SR)
    on = spec.render_sequence({"inst": "piano", "steps": st, "pedal": 1}, SR)
    k = int(0.6 * SR)
    assert np.sqrt(np.mean(on[k:] ** 2)) > 1.2 * np.sqrt(np.mean(off[k:len(on)] ** 2) + 1e-30)


def test_rhodes_hysteresis_lengthens_the_contact():
    t0 = acoustic.rhodes_contact_time(2.0)
    assert abs(acoustic.rhodes_contact_time(2.0, 1e-3) - t0) / t0 < 0.1
    assert acoustic.rhodes_contact_time(2.0, 1.0) > t0
    y = acoustic.rhodes(261.6, 0.4, SR, 0.8, hysteresis=1.0)
    assert np.all(np.isfinite(y))


def test_clav_flux_polynomial_tangent():
    p = acoustic.CLAV_FLUX_POLY
    assert abs(np.polyval(np.polyder(p), 0.0) - (-1.544)) < 1e-12
    x = np.linspace(-1e-3, 1e-3, 201)
    lin = p[-1] + p[-2] * x
    assert np.max(np.abs(np.polyval(p, x) - lin)) < 0.15 * 1.544e-3
