"""Group V4_VOIX (reports/lecture_profonde/suivi/IMPL_V4_VOIX.md): every check is a sourced or derived number.

Run:  uv run --with pytest pytest tests/test_v4_voix.py -q
"""
import math

import numpy as np

from genny.physical import story as ST

SP = 44100


def _f0(x, sr=SP, lo=60.0, hi=400.0):
    x = x - x.mean()
    r = np.correlate(x, x, "full")[x.size - 1:]
    a, b = int(sr / hi), int(sr / lo)
    return sr / (a + int(np.argmax(r[a:b])))


# ---------------------------------------------------------------- story: tables, lengths, branches
def test_measured_fricatives_table_v():
    a, d = ST.fricative_area("s")
    assert a.size == 44 and d == 0.353 and a[0] == 0.47 and a[-1] == 0.06
    a, d = ST.fricative_area("sh")
    assert d == 0.377 and np.all(a[35:38] == 0.0)
    assert ST.fricative_area("v")[1] == 0.361


def test_section_lengths_tier_iii_add_pg_pm():
    L = ST.section_lengths(pg=-1.0, pm=2.0)
    assert abs(L.sum() - (44 * ST.DX_CM + 1.0)) < 1e-9
    assert L[0] < ST.DX_CM < L[-1]
    A, dx = ST.uniform(ST.vowel_area("ɑ"), L, ST.DX_CM)
    assert abs(A.size * dx - L.sum()) < 1e-9


def test_piriform_branch_lowers_formants():
    V = ST.vowel_area("ʌ")
    F0 = ST.formants(V)[:3]
    F1 = ST.formants(V, branches=[ST.piriform_branch()])[:3]
    d = F0 - F1
    assert np.all(d > 0) and d.max() < 150.0


def test_nasal_branch_closed_port_is_transparent_open_port_shifts():
    V = ST.vowel_area("ɑ")
    F = ST.formants(V)[:2]
    shut = ST.formants(V, branches=[ST.nasal_branch(1e-4)])[:2]
    open_ = ST.formants(V, branches=[ST.nasal_branch(1.04)])[:2]
    assert np.all(abs(shut - F) / F < 0.01)
    assert abs(open_[0] - F[0]) / F[0] > 0.03


def test_maxillary_sinus_helmholtz_peak():
    s = ST.SINUSES[0]
    ln = s["neck_len"] + 1.7 * math.sqrt(s["neck_area"] / math.pi)
    f0 = ST.C_CMS / (2 * math.pi) * math.sqrt(s["neck_area"] / (s["volume"] * ln))
    f = np.arange(200.0, 900.0, 1.0)
    Y, _ = ST.branch_load(s, 2j * np.pi * f)
    assert abs(f[np.argmax(np.abs(Y))] - f0) / f0 < 0.02 and 400 < f0 < 550


def test_subglottal_first_resonance_quarter_wave():
    z = ST.source_impedance(subglottal=True)
    zg = ST.source_impedance(subglottal=False)
    f = np.arange(200.0, 1200.0, 2.0)
    w = 2j * np.pi * f
    zs = np.abs(z(w) - zg(w))
    fp = f[np.argmax(zs)]
    assert abs(fp - ST.C_CMS / (4 * ST.TRACHEA_LEN_CM)) / fp < 0.15


def test_glottal_resistance_van_den_berg():
    R, k, L = ST.glottal_resistance(0.08)
    assert math.isclose(R, 12 * 1.86e-4 * 1.4 ** 2 * 0.3 / 0.08 ** 3)
    assert math.isclose(k, 1.38 * 0.00114 / (2 * 0.08 ** 2))
    assert math.isclose(L, 0.00114 * 0.3 / 0.08)


def test_jet_recovery_hypotheses():
    assert ST.jet_recovery(0.1, 300.0, 3.0, "uniform") == 0.0
    r = (0.1 / 1.4) / (0.1 / 1.4 + 2 * math.tan(math.radians(4)) * 0.3)
    assert math.isclose(ST.jet_recovery(0.1, 300.0, 3.0, "turbulent"), 2 * r * (1 - r))
    assert 0.0 < ST.jet_recovery(0.1, 300.0, 3.0, "laminar") <= 0.5


def test_sensitivity_uniform_tube_kinetic_at_lips():
    A = np.full(44, 3.0)
    S = ST.sensitivity(A, ST.formants(A)[:3])
    assert np.all(np.abs(S.sum(1)) < 0.05)
    assert S[0, -1] > 0 > S[0, 0]


def test_deflection_closure_places():
    V = ST.vowel_area("neutral")
    for c, x in (("b", 17.5), ("d", 14.7), ("g", 12.7)):
        C = ST.deflection_constriction(V, c, mu=1.0)
        assert C.min() == 0.0
        assert abs((np.argmin(C) + 0.5) * ST.DX_CM - x) < 2.0


def test_q_from_formants_round_trip():
    V = ST.substrate(1.0, 1.0)
    F = ST.formants(V, fmax=3200.0, df=5.0)[:2]
    q1, q2 = ST.q_from_formants(F[0], F[1])
    Fq = ST.formants(ST.substrate(float(q1), float(q2)), fmax=3200.0, df=5.0)[:2]
    assert np.all(np.abs(np.log(Fq / F)) < 0.05)


def test_transfer_maps_extremes():
    om, f1, f2, pa, pb, L = ST.speaker_modes("female")
    _, _, _, qa, qb, _ = ST.speaker_modes("male")
    V, Lf = ST.transfer(qa[0], qa[1])
    assert np.allclose(V, np.pi / 4 * np.maximum(om + pa[0] * f1 + pa[1] * f2, 0) ** 2) and Lf == L


def test_liquids_sublingual_and_lateral():
    A, br = ST.liquid("ɹb")
    assert br[0]["at"] == round(13.1 / ST.DX_CM) and tuple(br[0]["area"]) == (1.5, 0.5)
    Al, _ = ST.liquid("lb")
    assert math.isclose(Al[21] - ST.liquid_area("lb")[21], 0.04) and math.isclose(Al[31] - ST.liquid_area("lb")[31], 0.28)


# ---------------------------------------------------------------- story: time-domain tract
def test_glottal_area_open_quotient_and_peak():
    fs = 88200.0
    a = ST.glottal_area(np.full(int(fs), 100.0), fs)
    assert abs(np.mean(a > 0) - 0.6) < 0.01 and abs(a.max() - 0.08) < 1e-4
    per = a[:882]
    assert abs(np.argmax(per) / 882 - 0.6 * 2.4 / 3.4) < 0.01


def test_protocol_f0_falls_25_percent():
    y = ST.protocol("ɑ", "male")
    env = lambda x: np.convolve(np.abs(x), np.ones(44) / 44, "same")    # pulse envelope: F1 ringing hides the period
    head, tail = _f0(env(y[int(0.03 * SP):int(0.08 * SP)])), _f0(env(y[-int(0.05 * SP):]))
    assert 1.25 < head / tail < 1.75 and 110 < head < 150


def test_kl_fricative_noisier_than_vowel():
    def hi(x):
        X = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(x.size, 1 / SP)
        return X[f > 4000].sum() / X.sum()
    s = ST.kl_voice("s", 120.0, 0.3, amax=0.0, leak=0.2)                 # voiceless: glottis abducted, 0.2 cm^2
    assert hi(s) > 2 * hi(ST.kl_voice("ɑ", 120.0, 0.3))


def test_kl_nasal_port_changes_spectrum():
    a = ST.kl_voice("ɑ", 120.0, 0.3, port=0.0, noise=False)
    b = ST.kl_voice("ɑ", 120.0, 0.3, port=1.04, noise=False)
    A, B = np.abs(np.fft.rfft(a)), np.abs(np.fft.rfft(b))
    assert np.sum((A - B) ** 2) / np.sum(A ** 2) > 0.05


def test_female_validation_formants_within_story_errors():
    # the measured tract already misses F1 by about -21 % (/i/, /ɑ/) with the piriform branch: allow 30 %
    for v, nat in ST.FEMALE_NATURAL.items():
        F = ST.formants(ST.vowel_area(v, "female"), branches=[ST.piriform_branch()])[:2]
        assert np.all(np.abs(F - np.array(nat[:2])) / np.array(nat[:2]) < 0.30), (v, F)


# ---------------------------------------------------------------- analysis
from genny import analysis as AN  # noqa: E402


def test_jitter_measures_definitions():
    t = np.array([0.010, 0.0102, 0.0099, 0.0101, 0.0100, 0.0103, 0.0098])
    m = AN.jitter_measures(t)
    assert math.isclose(m["ddp"], 100 * np.mean(np.abs(np.diff(t, 2))) / t.mean())
    assert math.isclose(m["local_abs_us"], 1e6 * np.mean(np.abs(np.diff(t))))
    rap = np.mean(np.abs(t[1:-1] - (t[:-2] + t[1:-1] + t[2:]) / 3)) / t.mean() * 100
    assert math.isclose(m["rap"], rap)


def test_active_bits_polarity_dc_null():
    x = np.round(np.sin(np.linspace(0, 20, 4000)) * 7) / 32767
    assert AN.active_bits(x) == 4
    assert AN.polarity(np.concatenate([np.zeros(10), [0.9, -0.5]])) == 1
    assert abs(AN.dc_offset_db(np.full(100, 0.001)) + 60.0) < 0.01
    y = np.random.default_rng(0).standard_normal(44100)
    assert AN.null_test_db(y, y * (1 + 1e-4), 44100) < -70


def test_correlation_dimension_and_lyapunov():
    x = np.sin(2 * np.pi * np.arange(3000) / 37.3)
    assert abs(AN.correlation_dimension(x, dim=3, tau=9) - 1.0) < 0.25
    z = np.empty(2000)
    z[0] = 0.3
    for i in range(1, z.size):
        z[i] = 4.0 * z[i - 1] * (1 - z[i - 1])
    assert abs(AN.lyapunov_exponent(z, dim=1, steps=4, min_sep=1) - math.log(2)) < 0.25


def test_temporal_masking_plateau_then_slow():
    dt = 0.001
    n = np.concatenate([np.ones(50), np.zeros(300)])
    m = AN.temporal_masking(n, dt)
    assert m[49] == 1.0 and m[54] == 1.0 and math.isclose(m[55], math.exp(-dt / 0.005), rel_tol=1e-9)
    assert m[349] < 0.01 and m[30] == 1.0 and m[0] > 0.0
    long = AN.temporal_masking(np.concatenate([np.ones(200), np.zeros(300)]), dt)
    assert long[204] == 1.0 and math.isclose(long[205], math.exp(-dt / 0.040), rel_tol=1e-9) and long[399] < 0.01


def test_threshold_and_partial_masking():
    assert abs(AN.threshold_in_quiet(1000.0) - 3.37) < 0.1
    ok = AN.audible_partials([1000.0, 1100.0, 4000.0], [0.0, -40.0, -20.0])
    assert ok[0] and not ok[1] and ok[2]


def test_estimate_rd_recovers_lf():
    from genny.physical.voice import lf_waveform
    y = np.tile(lf_waveform(1.2, 400), 20)
    assert abs(AN.estimate_rd(y, 40000, 100.0) - 1.2) < 0.3


def test_measure_ir_recovers_delay_and_gain():
    h = AN.measure_ir(lambda s: 0.5 * np.concatenate([np.zeros(100), s])[:s.size], dur=1.0, length=0.01)
    assert int(np.argmax(np.abs(h))) == 100 and abs(h.max() - 0.5) < 0.05


def test_ddsp_like_pitch_range():
    for y, f0 in AN.ddsp_like(6):
        assert 32.7 - 0.1 <= f0 <= 1046.6 and np.max(np.abs(y)) <= 1.0


def test_phase_track_constant_for_steady_tone():
    sr = 16000
    y = np.sin(2 * np.pi * 200 * np.arange(sr) / sr + 0.7)
    p = AN.phase_track(y, sr, 200.0)[20:-20]
    assert np.std(np.angle(np.exp(1j * (p - p.mean())))) < 0.05


def test_optimize_params_finds_frequency():
    sr = 16000
    t = np.arange(sr // 2) / sr
    target = np.sin(2 * np.pi * 440 * t)
    best = AN.optimize_params(lambda f: np.sin(2 * np.pi * f * t), target, sr, {"f": (200.0, 900.0)})
    assert abs(best["f"] - 440) < 40


def test_particle_count_grows_with_grains():
    rng = np.random.default_rng(1)
    def grains(k):
        y = np.zeros(22050)
        for p in rng.integers(0, 22000, k):
            y[p:p + 20] += rng.standard_normal(20)
        return y
    assert AN.particle_count(grains(400), 22050) > AN.particle_count(grains(20), 22050)


def test_hot_master_true_peak():
    from genny.spec import HOT_TRUE_PEAK, LOUDNESS_PRESETS
    assert HOT_TRUE_PEAK == -2.0 and LOUDNESS_PRESETS["apple_tv"] == -24.0 and LOUDNESS_PRESETS["kids"] == -28.0


def test_impact_level_allowance():
    y = 0.9 * np.sin(2 * np.pi * 3000 * np.arange(44100) / 44100)
    assert AN.wav_check_level(y, 44100) and len(AN.wav_check_level(y, 44100, "impact")) <= len(AN.wav_check_level(y, 44100))


# ---------------------------------------------------------------- speech: place-indexed loci
def test_speech_locus_place_labial_and_velar():
    from genny import speech as SPc
    v = "IY" if "IY" in SPc.PH else "IY1"
    u = "UW" if "UW" in SPc.PH else "UW1"
    f2v = SPc._targets([{"ph": v}])[0][0][1]
    lab_c = SPc._targets([{"ph": "B"}, {"ph": v}])[0][0][1]
    lab_p = SPc._targets([{"ph": "B"}, {"ph": v}], "place")[0][0][1]
    assert abs(lab_p - f2v) < abs(lab_c - f2v)
    vel_c = SPc._targets([{"ph": "G"}, {"ph": u}])[0][0][1]
    vel_p = SPc._targets([{"ph": "G"}, {"ph": u}], "place")[0][0][1]
    assert vel_c >= 1350 and vel_p >= SPc.VELAR_FLOOR and vel_p <= vel_c


# ---------------------------------------------------------------- voice / chiptune2 / choir / creatures
def test_lf_variant2_rk_offset():
    from genny.physical import voice as Vv
    a = Vv._lf_coefs(3.0)
    b = Vv._lf_coefs(3.0, 2)
    assert abs(a[3] - b[3]) > 1e-3          # Te moves with Rk = 2 Rg OQupp - 0.9572
    assert Vv._lf_coefs(1.0, 2) == Vv._lf_coefs(1.0)


def test_octave_doubler_removes_fundamental():
    from genny.physical import voice as Vv
    sr, f0 = Vv.SR, 200.0
    t = np.arange(sr // 2) / sr
    x = np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
    y = Vv.octave_doubler(x, f0)[sr // 10:]
    sp = np.abs(np.fft.rfft(y))
    fr = np.fft.rfftfreq(len(y), 1 / sr)
    assert sp[np.argmin(abs(fr - f0))] < 0.05 * sp[np.argmin(abs(fr - 2 * f0))]


def test_beak_filter_open_adds_10db_band():
    from genny.physical import voice as Vv
    rng = np.random.default_rng(0)
    x = rng.normal(size=Vv.SR)
    def band(y):
        sp = np.abs(np.fft.rfft(y)) ** 2
        fr = np.fft.rfftfreq(len(y), 1 / Vv.SR)
        return sp[(fr > 5000) & (fr < 7000)].sum()
    assert abs(Vv.beak_filter(x, 0.0) - x).max() < 1e-12
    g = 10 * np.log10(band(Vv.beak_filter(x, 1.0)) / band(x))
    assert 7.0 < g < 11.0


def test_soprano_second_carrier_near_2100():
    from genny import chiptune2 as C2
    y = C2.chowning(300.0, 1.0, voice="soprano")
    sp = np.abs(np.fft.rfft(y))
    fr = np.fft.rfftfreq(len(y), 1 / 48000)
    hi = (fr > 1500) & (fr < 3000)
    assert abs(fr[hi][np.argmax(sp[hi])] - 2100) < 200


def test_stk_fmvoice_formant_carriers():
    from genny import chiptune2 as C2
    y = C2.stk_fmvoice(110.0, 0.6, vowel="ee")
    sp = np.abs(np.fft.rfft(y))
    fr = np.fft.rfftfreq(len(y), 1 / 48000)
    hi = (fr > 1500) & (fr < 2700)
    assert abs(fr[hi][np.argmax(sp[hi])] - 2310) < 120   # 21 x 110 Hz, nearest harmonic to 2290


def test_choir_vib_am_key():
    from genny import choir as Ch
    assert "vib_am" in Ch._SING_KEYS


def test_bark_dominant_and_profiles():
    from genny import creatures as Cr
    assert abs(Cr.dominant_hz(0.0) - 1267) < 1 and abs(Cr.dominant_hz(1.0) - 626) < 1
    assert abs(Cr.stride_hz("horse", 10.0) - (1.81 + 0.19)) < 1e-9
    assert abs(Cr.stride_hz("mouse", 1.0) - 7.64) < 1e-9
    p = Cr.creature_profile(2.0)
    assert abs(p["f0_mul"] - 0.5) < 1e-12 and abs(p["tract_m"] - 0.34) < 1e-12 and abs(p["mass_mul"] - 8) < 1e-9
    a = Cr.bark(count=1, seed=1)
    b = Cr.bark(count=1, seed=1, dominant=True)
    assert len(a) == len(b) and np.abs(a - b).max() > 1e-3
