"""V5_EFFETS4: remaining T_Effets lines, compose feel, drums clean-up, cli overs.

Run:  uv run --with pytest pytest tests/test_v5_effets4.py -q
"""
import numpy as np

from genny import compose as C

SR = 48000
BASE = {"type": "compose", "key": "D", "scale": "dorian", "progression": "i VII VI VII", "bars": 4, "seed": 7}


def test_ellis_delay_and_ride_ratio():
    assert C.ellis_delay(90) == 40.0 and C.ellis_delay(210) == 63.0 and abs(C.ellis_delay(135) - 50.0) < 1e-9
    assert abs(C.ride_ratio(132) - 2.4) < 1e-9 and C.ride_ratio(300) == 1.0 and abs(C.ride_ratio(30) - 3.25) < 1e-9


def test_feel_soloist_bass_ride():
    L = dict(BASE, role="arp")
    ref = C.generate(L, 120)
    on = {round(e["t"], 6) for e in ref if abs(e["t"] - round(e["t"])) < 1e-6}
    s = C.generate(dict(L, feel="soloist"), 120)
    late = 47.0 / 1000.0 / 0.5
    assert all(any(abs(e["t"] - (t + late)) < 1e-5 for e in s) for t in on)
    b = C.generate(dict(L, feel="bass"), 120)
    assert abs(min(e["t"] for e in b) - min(e["t"] for e in ref) - 0.02) < 1e-5
    r = C.generate(dict(L, feel="ride"), 132)
    ref132 = C.generate(L, 132)
    half = [e["t"] for e in ref132 if abs(e["t"] % 1 - 0.5) < 1e-6]
    if half:
        x = 2.4 / 3.4
        assert any(abs(e["t"] % 1 - x) < 1e-5 for e in r)


def test_drums_kick_tom_are_physical():
    from genny import drums as D
    assert D.REGISTRY["kick"]["fn"] is D.kick and "model" in D.REGISTRY["kick"]["params"]
    assert D.REGISTRY["tom"]["fn"] is D.tom and "under" in D.REGISTRY["tom"]["params"]
    assert sum(1 for ln in open(D.__file__, encoding="utf-8") if ln.startswith("def kick(")) == 1
    assert sum(1 for ln in open(D.__file__, encoding="utf-8") if ln.startswith("def tom(")) == 1


def test_cli_info_json_overs(tmp_path, capsys):
    import json
    from genny.cli import main
    from genny.core import write_wav
    y = np.zeros(1000)
    y[100:105] = 1.0
    y[500:502] = -1.0
    p = tmp_path / "c.wav"
    write_wav(p, y, SR)
    main(["info", str(p), "--json"])
    d = json.loads(capsys.readouterr().out)
    assert d[0]["overs"] == 1
    main(["info", str(p)])
    assert "overs=" not in capsys.readouterr().out                # the tmp path itself contains "overs"


def test_cymbal_taper_and_zone_drive():
    from genny import percussion as P
    assert P.cymbal_thickness(0.1) == 1.0 and abs(P.cymbal_thickness(1.0) - 0.7) < 1e-12
    assert abs(P.cymbal_thickness(0.625) - 0.85) < 1e-12
    assert P.zone_drive(0.12) == 1.0 and abs(P.zone_drive(1.0) - 0.7 ** -3) < 1e-9
    assert P.zone_drive(P.SHELL_ZONES["edge"]) > P.zone_drive(P.SHELL_ZONES["bow"]) > 1.0


def test_glide_couple_refills_plucked_out_harmonic():
    from genny.strings import DispersiveString, StringPhysicalProperties, GLIDE_COUPLE_POLE
    assert GLIDE_COUPLE_POLE == 0.97
    sr = 48000
    p = StringPhysicalProperties.from_frequency(110.0, material="steel")
    base = DispersiveString(p, sr=sr).render_pluck(1.0, velocity=.9, position=1 / 3, glide_cents=40)
    same = DispersiveString(p, sr=sr).render_pluck(1.0, velocity=.9, position=1 / 3, glide_cents=40, glide_couple=0.0)
    assert np.array_equal(base, same)
    cpl = DispersiveString(p, sr=sr).render_pluck(1.0, velocity=.9, position=1 / 3, glide_cents=40, glide_couple=1.0)

    def h3(y):
        Y = np.abs(np.fft.rfft(y[2400:] * np.hanning(y.size - 2400)))
        fr = np.fft.rfftfreq(y.size - 2400, 1 / sr)
        band = lambda f: Y[(fr > f * 0.97) & (fr < f * 1.03)].max()   # noqa: E731
        return band(330.0) / band(110.0)
    assert h3(cpl) > h3(base)


def test_damped_calls_rhythm():
    from genny.physical import nonverbal as NV
    st, ln, g = NV.damped_calls(5)
    assert st[0] == 0.0 and abs(ln[1] - 1000 / 4.6 / 2) < 15.0
    assert np.all(np.diff(st) > 0) and np.all(np.diff(g) < 0)
    period = np.diff(st)
    assert period[-1] > period[0]                       # the calls slow down: 4.6 -> 3.6 Hz
    centre = (st[2] + ln[2] / 2) / 1000.0
    assert abs(g[2] - 20 * np.log10(np.exp(-1.2 * centre))) < 0.5
    y = NV.vocalize("laugh", "female", n_syl=5, rhythm="oscillator")
    assert np.isfinite(y).all() and y.size > 0


def test_moan_sova_preset_and_vibrato():
    from genny.physical import nonverbal as NV
    p = NV.preset("moan_sova")
    assert p["nSyl"] == 3 and p["sylLen"] == 520 and p["pauseLen"] == 400 and p["vibratoDep"] == 0.5
    assert p["formants"][0] == (900,) and p["noise"][1][1] == -40
    assert "moan_sova" not in NV.KINDS and NV.preset("moan")["vibratoDep"] == 0.0
    y = NV.vocalize("moan_sova")
    assert np.isfinite(y).all() and y.size > int(1.5 * NV.SR)


def test_tx81z_dac_matches_source_coefficients_at_44k1():
    from scipy.signal import butter
    from genny import chiptune2 as C2
    b, a = butter(1, C2.TX81Z_DAC_LP, "low", fs=44100)
    assert abs(a[1] + 0.51005) < 2e-3 and abs(b[0] - 0.24498) < 2e-3
    b, a = butter(1, C2.TX81Z_DAC_HP, "high", fs=44100)
    assert abs(a[1] + 0.998695) < 2e-5 and abs(b[0] - 0.999347) < 2e-5
    on = C2.stk_fm(220.0, 0.3, sr=44100, dac=1)
    off = C2.stk_fm(220.0, 0.3, sr=44100)
    assert on.shape == off.shape and not np.allclose(on, off)


def test_minblep_is_causal_and_cleaner_than_naive():
    from genny import osc as O
    from genny.analysis import alias_snr
    t = O.minblep_table()
    assert abs(t[0] + 1.0) < 0.05 and abs(t[-1]) < 1e-9
    assert np.argmax(t > -0.5) < 3 * 64                           # the step is half done within 3 samples
    sr, f0, n = 44100, 2093.0, 44100
    y = O.minblep_saw(f0, n, sr)
    naive = 2.0 * (O.phase(f0, n, sr) % 1.0) - 1.0
    assert alias_snr(y, sr, f0) > alias_snr(naive, sr, f0) + 10.0


def test_ensemble_sigma_and_nma():
    assert C.ensemble_sigma_ms(500) == 30.0 and C.ensemble_sigma_ms(1500) == 45.0 and C.ensemble_sigma_ms(3000) == 50.0
    L = dict(BASE, role="arp")
    ref = C.generate(L, 120)
    n = C.generate(dict(L, nma=True), 120)
    rt, nt = sorted(e["t"] for e in ref), sorted(e["t"] for e in n)
    assert all(abs((x - y) - 0.06) < 1e-4 for x, y in zip(rt, nt) if x > 0.1)
    e = C.generate(dict(L, ensemble=True), 120)
    assert len(e) == len(ref) and any(abs(a["t"] - b["t"]) > 1e-4 for a, b in zip(sorted(e, key=lambda x: x["pitch"]), sorted(ref, key=lambda x: x["pitch"])))


def test_emotion_palettes():
    assert C.EMOTIONS["sad"]["tempo"] == 0.85 and C.EMOTIONS["angry"]["level_db"] == 3.0
    L = dict(BASE, role="melody")
    ref = C.generate(L, 120)
    sad = C.generate(dict(L, emotion="sad"), 120)
    assert np.mean([x["vel"] for x in sad]) < np.mean([x["vel"] for x in ref])
    import pytest
    with pytest.raises(Exception):
        C.generate(dict(L, emotion="bored"), 120)


def test_phrase_lengthen_and_long_note_contrast():
    assert C.PHRASE_LENGTHEN_LAST == 0.20 and not hasattr(C, "DDC_LONG_MS")
    L = dict(BASE, role="melody")
    ref = C.generate(L, 120)
    pl = C.generate(dict(L, phrase_lengthen=1.0), 120)
    assert len(pl) == len(ref) and sorted(x["t"] for x in pl) != sorted(x["t"] for x in ref)


def test_accent_dip_drums():
    L = dict(BASE, role="drums", kit={"snare": "x--Xx--X"}, bars=1)
    ref = C.generate(L, 120)
    dip = C.generate(dict(L, accent_dip=True), 120)
    r = sorted(x["vel"] for x in ref)
    d = sorted(x["vel"] for x in dip)
    assert min(d) < min(r) and abs(min(d) / min(r) - 10 ** (-2.0 / 20)) < 0.05


def test_hole_corrections_and_jet_receptivity():
    import math
    from genny import tubes as T
    tm, ti, tr, tot = T.hole_length_corrections(0.003, 0.006, "unflanged")
    assert abs(tm - 0.003 * 0.5 * (1 + 0.207 * 0.125) / 8) < 1e-12 and abs(ti - 0.003 * 0.565375) < 1e-9
    assert abs(tr - 0.6133 * 0.003) < 1e-12 and abs(tot - tm - ti - tr) < 1e-15
    assert abs(T.hole_length_corrections(0.003, 0.006, "flanged")[2] - 0.8216 * 0.003) < 1e-12
    holes = [(0.8, 0.003, 0.005)]
    a = T.WebsterTube(0.3, holes=holes).hpar
    b = T.WebsterTube(0.3, holes=holes, hole_kind="unflanged").hpar
    assert a[0, 0] == b[0, 0] and a[0, 2] != b[0, 2]
    U0, l = 20.0, 0.004
    f = U0 / (2 * math.pi * l)
    g, ph, rounded = T.jet_receptivity(f, l, U0, R=0.0)
    assert abs(g - math.exp(0.4)) < 1e-9 and abs(ph - (0.5 * math.pi + 2.0)) < 1e-9 and not rounded
    assert T.jet_receptivity(f, l, U0, R=U0 / f)[2]


def test_vowel_tube_minjerk_and_kl():
    from genny import tubes as T
    a = T.vowel_tube(220.0, 0.3, vowel="a", vowel2="i")
    b = T.vowel_tube(220.0, 0.3, vowel="a", vowel2="i", glide="minjerk")
    assert a.shape == b.shape and not np.allclose(a, b)
    k = T.vowel_tube(220.0, 0.3, vowel="a", model="kl")
    n = T.vowel_tube(220.0, 0.3, vowel="a", nasal=0.5)
    assert np.isfinite(k).all() and np.isfinite(n).all() and not np.allclose(k[:n.size], n[:k.size])


def test_perceptual_alias_limits():
    from genny import osc as O
    assert O.DPW_ALIAS_FREE[44100] == (600.0, 2037.0, 4593.0, 7851.0, 12221.0)
    assert O.dpw_order_for(500) == 2 and O.dpw_order_for(2000) == 3 and O.dpw_order_for(5000) == 5
    assert O.dpw_order_for(13000) == 0 and O.dpw_order_for(1000, 88200) == 2 and O.dpw_order_for(1100, 88200) == 3
    assert O.SAW_ALIAS_FREE["blep"] == 5135.0 and O.FRAC_DELAY_ALIAS_FREE["lagrange3"] == 4143.0
    # the measured order 3 limit holds for genny's DPW: cleaner than order 2 at 2 kHz
    from genny.analysis import alias_snr
    assert alias_snr(O.dpw(2000.0, 44100, 44100, order=3), 44100, 2000.0) > alias_snr(O.dpw(2000.0, 44100, 44100, order=2), 44100, 2000.0)


def test_wavetable_from_spectrum():
    from genny import osc as O
    t = O.wavetable_from_spectrum([1.0, 0.5])
    x = np.arange(O.WAVETABLE_N) / O.WAVETABLE_N
    assert np.allclose(t, np.cos(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * x), atol=1e-9)
    y = O.wavetable_osc(10000.0, 4800, 48000, harmonics=(1.0, 1.0, 1.0))
    Y = np.abs(np.fft.rfft(y))
    assert Y[2000] > 0.3 * Y.max() and Y[1800] < 1e-3 * Y.max()       # 20 kHz kept, 30 kHz (alias at 18 kHz) dropped


def test_choir_vibrato_options():
    from genny import choir as CH
    assert CH.VIB_DYN[2:4] == (0.37, 0.48) and CH.STYLES["amateur"]["rate"] == 5.3 and CH.STYLES["amateur"]["vib"] == 45.0
    notes = [(0.0, 60.0, 0.6, 1.0), (0.6, 67.0, 0.6, 1.0)]
    ref = CH.sing_phrase(notes, 22050, stereo=False, voice="tenor", singers=1, dynamics="ff")
    for kw in ({"peak_stretch": 40.0}, {"vib_dyn": True}, {"vib_cycle": True}):     # vib_dyn is neutral at mf
        y = CH.sing_phrase(notes, 22050, stereo=False, voice="tenor", singers=1, dynamics="ff", **kw)
        assert y.shape == ref.shape and not np.allclose(y, ref)


def test_nonlinear_onepoles_and_ladder_mix():
    from genny.filters import ladder_mix, onepole_nl, onepole4, tbvcf
    sr = 44100
    x = np.random.default_rng(0).normal(0, 0.3, sr // 4)
    for k in ("transistor", "ota", "diode"):
        assert np.isfinite(onepole_nl(x, 1000, sr, k)).all()
    imp = np.zeros(8192)
    imp[0] = 1.0
    fr = np.fft.rfftfreq(8192, 1 / sr)
    H = np.abs(np.fft.rfft(ladder_mix(imp * 1e-3, 1000, sr, 0.0, "hp4") / 1e-3))
    assert H[np.argmin(abs(fr - 100))] < 0.01 < 0.9 < H[np.argmin(abs(fr - 10000))]
    N = np.abs(np.fft.rfft(ladder_mix(imp * 1e-3, 1000, sr, 0.0, "notch") / 1e-3))
    assert N[np.argmin(abs(fr - 1000))] < 0.05
    L = np.abs(np.fft.rfft(onepole4(imp, 1000, sr)))
    assert abs(fr[np.argmin(abs(L - 0.7071))] - 1000) < 60
    saw = 2 * ((110 * np.arange(sr // 4) / sr) % 1) - 1
    assert 0.05 < np.abs(tbvcf(saw)).max() < 2.0


def test_velvet_fit_roundtrip():
    from genny.identify import velvet_fit, velvet_render
    sr = 44100
    t = np.arange(sr) / sr
    rir = np.random.default_rng(1).normal(0, 1, sr) * np.exp(-6.9 * t / 0.8)
    fit = velvet_fit(rir, sr)
    assert fit["a"].shape[1] == 11
    y = velvet_render(fit)
    late, early = np.sqrt(np.mean(y[35280:39690] ** 2)), np.sqrt(np.mean(y[:4410] ** 2))
    assert np.isfinite(y).all() and late < 0.01 * early


def test_dsp_v5_effets4_helpers():
    from genny import dsp
    sr = 44100
    assert dsp.is_unilossless([[0, 1], [1, 0]]) and not dsp.is_unilossless([[3, 2], [-4, -3]])
    assert dsp.fdn_allpass_matrix(np.eye(2), 0.5).shape == (4, 4)
    o = dsp.wg_osc(np.linspace(200, 3000, sr), sr, sr)
    assert abs(np.abs(o[-2000:]).max() - 1.0) < 0.01
    imp = np.zeros(sr)
    imp[0] = 1.0
    V = np.abs(np.fft.rfft(dsp.wdf_oscillator(imp, sr)))
    assert abs(np.argmax(V) - np.sqrt(4000 / 0.01) / (2 * np.pi)) < 3
    near = np.abs(np.fft.rfft(dsp.head_shadow(imp[:4096], sr, 0.0)))[929]
    far = np.abs(np.fft.rfft(dsp.head_shadow(imp[:4096], sr, 180.0)))[929]
    assert near > 1.5 and far < 0.4
    assert dsp.decorrelation_fir(sr).size == int(0.02 * sr)
    x = np.random.default_rng(0).normal(0, 0.3, 8000)
    assert abs(dsp.apdist(x, sr).std() - x.std()) < 0.05
    assert np.isfinite(dsp.dynamic_allpass(x, sr, order=40)).all()
    assert np.isfinite(dsp.variation_filter(x, sr)).all()
    s = np.sin(2 * np.pi * 110.25 * np.arange(sr) / sr)
    assert np.abs(dsp.median_comb(s, sr, 110.25) - s)[sr // 2:].max() < 0.01
    m = dsp.modal_fft([440.0], [2.0], [1.0], 0.5, sr)
    assert 0.6 < np.abs(m[2048:4096]).max() < 1.05


def test_plume_and_cold_idle():
    from genny.engines import cold_idle
    from genny.marine import plume_f0
    assert abs(plume_f0(0.75) - 6.3) < 0.1 and abs(plume_f0(0.1) - 47) < 1
    assert cold_idle(800, 400)[0] == 800 and abs(cold_idle(800, 285)[0] - 1240) < 1e-6
