import math

import numpy as np

from genny import contact as C
from genny import creatures as CR
from genny import geo as G
from genny import identify as I
from genny import nature as N
from genny import spectral as SP
from genny import tubes as T
from genny.physical import story as S


def test_levine_fit_matches_rational_end_reflection():
    for ka in (0.05, 0.3, 0.5, 0.8, 1.0, 1.4):
        a, b = T.end_reflection_levine(ka), T.end_reflection(ka, "unflanged")
        assert abs(abs(a) - abs(b)) < 0.03
        assert abs(np.angle(a) - np.angle(b)) < 0.08
    assert abs(abs(T.end_reflection_levine(1.0)) - 0.6940) < 1e-3
    assert abs(T.end_reflection_levine(0.5 - 1e-9) - T.end_reflection_levine(0.5 + 1e-9)) < 1e-6


def test_oblique_impact_angle_of_reflexion():
    mu, vz, kap = 0.5, 1.0, 2 * 0.7 / 1.7
    out = {}
    for psi in (1.2, 2.0, 6.0, 10.0):
        out[psi] = C.oblique_impact(psi * mu * vz / kap, vz, mu)
    chi = out[1.2]["chi"]
    assert out[2.0]["psi_out"] < 0 and out[2.0]["psi_out_rigid"] == 0.0
    for psi in (6.0, 10.0):
        assert math.isclose(out[psi]["psi_out"], psi - 4 * chi, rel_tol=0.03)
        assert math.isclose(out[psi]["psi_out_rigid"], psi - 4 * chi, rel_tol=1e-9)


def test_drm_regions_move_formants_with_their_sign():
    base = S.formants(S.drm_area([5.0] * 8))[:3]
    assert math.isclose(sum(S.DRM_LENGTHS), 1.0)
    for n in range(8):
        a = [5.0] * 8
        a[n] = 6.0
        f = S.formants(S.drm_area(a))[:3]
        for j in range(3):
            assert np.sign(f[j] - base[j]) == S.DRM_SIGNS[j][n], (n + 1, j + 1)


def test_diana_alarm_leopard_and_eagle():
    p = CR.DIANA_CALL
    assert p["f0"] < 70.0 and 0.09 < p["gap"] < 0.13
    a, b = CR.diana_alarm(0), CR.diana_alarm(0, predator="eagle")
    assert len(a) == CR._n(p["calls"] * (p["dur"] + p["gap"]) + 0.05)
    assert len(b) == CR._n(CR.DIANA_EAGLE_CALLS * (p["dur"] + p["gap"]) + 0.05)
    assert len(b) > 2.5 * len(a)


def test_double_pulse_tremor_is_harmonic_at_inverse_gap():
    rng = np.random.default_rng(1)
    n = int(N.P * 600)
    y = N.double_pulse_tremor(n, rng, gap=(1.67, 1.67))
    f = np.fft.rfftfreq(n, 1.0 / N.P)
    s = np.abs(np.fft.rfft(y)) ** 2
    lo = (f > 0.4) & (f < 0.8)
    assert abs(f[lo][np.argmax(s[lo])] - 1 / 1.67) < 0.03
    assert math.isclose(N.doppler_shift(1.75, 70.0, 340.0), 2.2, abs_tol=0.01)


def test_glacier_calving_modes_and_bubble_pulse():
    assert G.CALVING_MODES["submarine"][0] > G.CALVING_MODES["typical"][0]
    assert [G.CALVING_MODES[k][1] for k in ("typical", "sliding", "submarine")] == [17.0, 24.0, 32.0]
    base = G.glacier_calving(dur=3.0, seed=3)
    assert np.array_equal(base, G.glacier_calving(dur=3.0, seed=3, mode="", pulse=0.0))
    assert not np.array_equal(base, G.glacier_calving(dur=3.0, seed=3, pulse=0.8))
    for m in G.CALVING_MODES:
        assert np.isfinite(G.glacier_calving(dur=3.0, seed=3, mode=m)).all()


def test_psola_reverse_repeats_only_changes_unvoiced_repeats():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(24000)
    a = SP.psola(x, 48000, factor=2.0)
    b = SP.psola(x, 48000, factor=2.0, reverse_repeats=True)
    assert a.shape == b.shape and not np.array_equal(a, b)
    t = np.arange(24000) / 48000
    v = np.sin(2 * np.pi * 200 * t)
    assert np.array_equal(SP.psola(v, 48000, factor=2.0, f0=200.0), SP.psola(v, 48000, factor=2.0, f0=200.0, reverse_repeats=True))


def test_sanc_rules_drs_window_and_med_damping():
    assert I.sanc_order(10.0, 48000) == 4800
    assert I.sanc_delay(100.0, 48000) == 480
    assert math.isclose(I.sanc_static_gain(1.0, 99), 0.99)
    rng = np.random.default_rng(0)
    n = 60000
    x = np.sin(2 * np.pi * 0.01 * np.arange(n)) + 0.5 * rng.standard_normal(n)
    d, r = I.drs(x, 2000, 1024, window="parzen")
    assert np.var(d) > 0.4 and np.var(r) < np.var(x)
    imp = np.zeros(4000)
    imp[::400] = 1.0
    y, f = I.med(np.convolve(imp, [1.0, 0.7, 0.4], "same"), 20, 6, damp=0.003)
    assert I.kurtosis4(y) > I.kurtosis4(np.convolve(imp, [1.0, 0.7, 0.4], "same"))


def test_formant_spacing_and_kuramoto_helpers():
    L = 0.17
    F = [(2 * i - 1) * 350.0 / (4 * L) for i in (1, 2, 3)]
    df = CR.formant_spacing(F)
    assert math.isclose(df, 350.0 / (2 * L), rel_tol=1e-12)
    assert math.isclose(CR.apparent_vocal_tract_length(df), L, rel_tol=1e-12)
    assert math.isclose(CR.kuramoto_threshold(1.0), math.sqrt(8 / math.pi))
    assert CR.kuramoto_threshold(0.5, "lorentz") == 1.0
    assert CR.kuramoto_order(0.9, 1.0) == 0.0 and math.isclose(CR.kuramoto_order(4.0, 1.0), math.sqrt(0.75))


def test_speech_vot_tables_are_lisker_abramson_values():
    from genny import speech as SPK
    assert SPK.ASPIRATED_VOT == {"lab": 0.058, "alv": 0.070, "vel": 0.080}
    assert SPK.SPANISH_VOT == {"lab": 0.004, "alv": 0.009, "vel": 0.029}
