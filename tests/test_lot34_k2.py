import math

import numpy as np
import pytest
from scipy.signal import fftconvolve, find_peaks, welch

from genny import creatures, fx, physics, plates, tubes
from genny.physical import bank_piano, modal, winds
from genny.physical.waveguides import measure_f0


def test_lipping_both_ways_on_the_rendered_lip():
    lp = winds.lip_params(0.5)
    fa, qa, za = 229.5, 24.0, 20e6
    sr = winds.SR
    t = np.arange(sr) / sr
    for fl, ql, im, above in ((fa * 0.95, lp["QL1"], lp["inv_mu1"], True), (fa * lp["fL2"] / lp["fL1"], lp["QL2"], lp["inv_mu2"], False)):
        pt, _ = winds.lip_threshold(fl, ql, im, lp["b"], lp["H0"], fa, qa, za)
        p = winds.lip_one_mass(pt * 1.5 * np.clip(t / 0.03, 0, 1), fl, ql, im, lp["b"], lp["H0"], [(fa, za, qa)], sr)
        f = measure_f0(p[int(0.4 * sr):], fa)
        assert (f > fa and f > fl) if above else (f < fa and f < fl), (above, f, fa, fl)


def test_hammer_contact_lasts_about_five_ms_on_a_loud_c2():
    cap = {}
    orig = bank_piano._hammer

    def spy(*a):
        cap["F"], cap["sr"] = orig(*a), a[1]
        return cap["F"]

    bank_piano._hammer = spy
    try:
        bank_piano.note(65.406, 0.3, 44100, 1.0, tail=0.0)
    finally:
        bank_piano._hammer = orig
    nz = np.flatnonzero(cap["F"] > 0)
    assert 4.0e-3 < (nz[-1] - nz[0] + 1) / cap["sr"] < 6.0e-3


def test_bird_call_peaks_sit_on_the_quarter_wave_resonances_within_15_percent():
    y = creatures.bird_call(48000, freq=200.0, freq_end=800.0, dur=1.5, two_voice=0.0)
    f, p = welch(y, 48000, nperseg=8192)
    k = np.convolve(10 * np.log10(p + 1e-20), np.ones(9) / 9, "same")
    m = (f > 500) & (f < 8000)
    pk = f[m][find_peaks(k[m], prominence=1.5)[0]]
    for ref in (343.0 / (4 * 0.07) * n for n in (1, 3, 5)):
        assert np.min(np.abs(pk / ref - 1)) < 0.15, (ref, pk)
    assert np.min(np.abs(pk / 1200.0 - 1)) < 0.15


def test_blast_ps_and_impulse_within_20_percent_of_the_epx_runs():
    w = 3629.0
    for z, ps_ref, is_ref in ((1.199, 591.19, 1625.33), (1.599, 264.78, 1299.18), (1.989, 153.20, 1122.71)):
        ps, td, a = physics.blast_wave(w, z * w ** (1 / 3))
        imp = ps * td * (1 / a - (1 - math.exp(-a)) / a ** 2)
        assert abs(ps / 1e3 / ps_ref - 1) < 0.20 and abs(imp / is_ref - 1) < 0.20, (z, ps / 1e3, imp)


def test_steel_rectangle_300n_7ms_cascades_up_the_spectrum():
    h = 1.2e-3
    mat = (7860.0, 2e11, 0.3)
    size = math.sqrt(0.4 * 0.6) / 0.886
    top = {}
    for model in ("vk", "linear"):
        info = {}
        y = plates.nl_plate(48000, 1.0, size, 1.2, 300.0, 14.0, (0.31, 0.43), 10.0, 0.0005, model, 1.5, mat, 1.0, 1.5, dur=1.5,
                            drive=lambda n, fs: plates.pulse(n, fs, 300.0, 14.0), info=info, accel=False, hp=20.0)
        f, p = welch(y[int(0.25 * 48000):int(0.75 * 48000)], 48000, nperseg=4096)
        pd = 10 * np.log10(p + 1e-30)
        top[model] = (f[np.flatnonzero(pd > pd.max() - 60)[-1]], info["peak_u"] * h / math.sqrt(6 * 0.91) / h)
    assert 3 < top["vk"][1] < 8                                    # the plate swings to several thicknesses (almost 6 h)
    assert top["vk"][0] > 3000.0 and top["vk"][0] > 5 * top["linear"][0], top


def test_reed_closes_half_the_cycle_for_loud_notes_and_not_at_threshold():
    sr = tubes.FS

    def closed(pm):
        t = tubes.WebsterTube(0.664, 1.0, radius_m=math.sqrt(tubes.REED_S0 / math.pi), left="reed", right="unflanged")
        y = t.run(pm=np.full(2 * sr, pm) * tubes._edge(2 * sr, 200, 0), reed=tubes.reed_params())["y"][sr:]
        return float(np.mean(y <= -1.0))

    assert closed(0.0134) == 0.0
    assert closed(0.018) == pytest.approx(0.454, abs=0.03) and closed(0.028) == pytest.approx(0.494, abs=0.03)


def test_tube_with_radiation_lands_on_the_corrected_pitch():
    u = np.zeros(2 * tubes.FS)
    u[0] = 1.0
    for right, want in (("open", 444.08), ("unflanged", 419.83)):
        t = tubes.WebsterTube(0.193, 1.0, radius_m=0.0185, left="closed", right=right, loss=0.5)
        p = t.run(u)["p_in"]
        Y = np.abs(np.fft.rfft(p * np.hanning(len(p)), 4 * len(p)))
        f = np.fft.rfftfreq(4 * len(p), 1 / tubes.FS)
        k = int(np.argmax(Y[:int(700 / f[1])]))
        assert f[k] == pytest.approx(want, rel=1.5e-3), (right, f[k])


def test_cavity_cases_of_table_4_6():
    for u, length, width, depth, fd_want, fm_want in (
            (40.0, 0.06, 0.06, 0.35, None, 213.0), (31.0, 0.15, 0.15, 0.15, (84, 196, 308, 420), 303.0)):
        fd, fm = modal.cavity_frequencies(u, length, depth, width)
        assert fm == pytest.approx(fm_want, rel=0.01)
        if fd_want:
            assert np.allclose(fd, fd_want, rtol=0.003, atol=1.0)
    assert modal.cavity_frequencies(40.0, 0.06, 0.35, 0.06)[1] == pytest.approx(225.0, rel=0.064)   # the measured 225 Hz


def test_spring_reverb_delay_time_by_autocorrelation_is_56_ms():
    sr = 44100
    click = np.zeros(2 * sr)
    click[0] = 1.0
    w = fx.REGISTRY["spring_reverb"]["fn"](click, sr, mix=1.0, preset="leem1")
    ac = fftconvolve(w, w[::-1])[len(w) - 1:]
    lo = int(0.01 * sr)
    td = (lo + int(np.argmax(np.abs(ac[lo:int(0.2 * sr)])))) / sr
    assert abs(td - 0.056) < 0.002


def test_chirped_impact_atom_follows_its_law():
    sr = modal.SR
    y = modal.chirped_impact(1000.0, 2000.0, 30.0, 0.3)
    for t0, want in ((0.05, 1100.0), (0.2, 1400.0)):
        seg = y[int((t0 - 0.01) * sr):int((t0 + 0.01) * sr)]
        z = np.flatnonzero(np.diff(np.signbit(seg)))
        assert abs(sr * (len(z) - 1) / 2.0 / (z[-1] - z[0]) - want) < 0.03 * want
    env = lambda a: np.max(np.abs(y[int(a * sr):int((a + 0.002) * sr)]))                    # noqa: E731
    assert env(0.1) / env(0.0) == pytest.approx(math.exp(-3.0), rel=0.08)


def test_roar_onset_starts_eleven_semitones_down_and_lands_in_a_tenth_of_a_second():
    from genny import choir
    sr, f0 = 16000, 523.25

    def cents(style):
        y = choir.voice(f0, 0.8, sr, 1.0, voice="soprano", vowel="ah", style=style, vibrato=0.0, seed=1)
        w = int(0.04 * sr)
        out = []
        for i in range(0, int(0.4 * sr), int(0.01 * sr)):
            S = np.abs(np.fft.rfft(y[i:i + w] * np.hanning(w), 1 << 15))
            fr = np.fft.rfftfreq(1 << 15, 1 / sr)
            m = (fr > f0 * 0.4) & (fr < f0 * 1.4)
            out.append(1200 * math.log2(fr[m][np.argmax(S[m])] / f0))
        return np.array(out)

    r, c = cents("roar"), cents("classical")
    assert choir.STYLES["roar"]["scoop"] == 1100.0 and r[0] < -200.0 and np.all(np.diff(r[:12]) > -5.0)
    assert abs(r[25]) < 25.0 and abs(c[0]) < 30.0


def test_erb_noise_bank_is_on_above_the_edge_and_off_below():
    from genny import vehicle
    sr, n = 44100, 1 << 17
    for high, edge in ((True, 1000.0), (False, 2000.0)):
        x = vehicle._erb_noise(np.random.default_rng(0), n, sr, edge, high)
        p, f = np.abs(np.fft.rfft(x)) ** 2, np.fft.rfftfreq(n, 1 / sr)
        on, off = (f > 1.5 * edge, f < 0.5 * edge) if high else (f < 0.6 * edge, f > 2.0 * edge)
        assert 10 * np.log10(p[on].mean() / p[off].mean()) > 30.0
    y = vehicle.car_engine(sr, rpm=2400, cylinders=8, model="ptr", noise=1.0, bank="erb", dur=0.5)
    assert np.all(np.isfinite(y)) and not np.allclose(y, vehicle.car_engine(sr, rpm=2400, cylinders=8, model="ptr", noise=1.0, dur=0.5))


def test_scream_is_amplitude_modulated_inside_the_roughness_window():
    from genny.physical import nonverbal as nv
    assert nv.PRESETS[("scream", "female")]["amFreq"] == 100 and nv.PRESETS[("scream", "female")]["amDep"] == 40
    y = nv.vocalize("scream", "female", np.random.default_rng(0))
    x = y[int(0.3 * nv.SR):int(1.0 * nv.SR)]
    k = int(0.001 * nv.SR)
    env = np.convolve(np.abs(x), np.ones(k) / k, "same")
    E = np.abs(np.fft.rfft(env - env.mean()))
    f = np.fft.rfftfreq(len(env), 1 / nv.SR)
    band = (f > 30) & (f < 150)
    assert 70.0 <= f[band][np.argmax(E[band])] <= 110.0


def test_clav_knock_adds_a_burst_under_1200_hz_in_the_first_30_ms():
    from genny import instruments as I
    a = I.render_note("clav", 220.0, 0.5, 44100, 1.0)
    b = I.render_note("clav", 220.0, 0.5, 44100, 1.0, knock=1.0)
    sr, k = 44100, int(0.03 * 44100)
    g = np.dot(a[2 * k:], b[2 * k:]) / np.dot(a[2 * k:], a[2 * k:])                      # the output is peak-normalised
    d = b - g * a
    S, f = np.abs(np.fft.rfft(d[:k], 8192)) ** 2, np.fft.rfftfreq(8192, 1 / sr)
    assert np.sum(d[2 * k:] ** 2) < 1e-3 * np.sum(d[:k] ** 2) and np.sum(d[:k] ** 2) > 0.01 * np.sum(a[:k] ** 2)
    assert S[f < 1400].sum() > 0.8 * S.sum()
    assert np.array_equal(a, I.render_note("clav", 220.0, 0.5, 44100, 1.0, knock=0.0))


def test_danger_sits_where_a_minor_second_beats_in_the_roughness_window():
    from genny import compose as C
    from genny.notes import midi_to_freq
    g = sorted(C.generate({"type": "stinger", "mood": "danger", "length": 4, "seed": 0}), key=lambda e: e["t"])
    p0 = [e["pitch"] for e in g if e["t"] == g[0]["t"]][0]
    p1 = [e["pitch"] for e in g if e["t"] > g[0]["t"]][0]
    f0, f1 = midi_to_freq(p0), midi_to_freq(p1)
    assert p1 - p0 == 1 and 30.0 <= f1 - f0 <= 80.0 and 500.0 <= f0 and f1 <= 1350.0, (f0, f1)


def test_strum_var_spreads_onsets_and_t60_and_is_off_by_default(monkeypatch):
    from genny import instruments as I
    seen = []
    orig = I.render_note

    def spy(name, f, dur, sr, vel, **kw):
        seen.append(kw.get("decay"))
        return orig(name, f, dur, sr, vel, **kw)

    monkeypatch.setattr(I, "render_note", spy)
    fr = [110.0, 165.0, 220.0, 330.0]
    a = I.render_chord("ks_string", fr, 0.4, 22050, 0.8, strum=0.02)
    assert seen == [None] * 4
    seen.clear()
    b = I.render_chord("ks_string", fr, 0.4, 22050, 0.8, strum=0.02, strum_var=0.3)
    assert len(set(seen)) == 4 and all(v > 0 for v in seen) and not (len(a) == len(b) and np.allclose(a, b))
    assert np.array_equal(a, I.render_chord("ks_string", fr, 0.4, 22050, 0.8, strum=0.02, strum_var=0.0))


def test_broadleaf_sway_q_narrows_the_band():
    from genny.physical import ambience as amb
    n = 120 * amb.SR
    w = np.full(n, 0.5)
    share = []
    for q in (3.0, 5.0):
        f = amb.sway_force(w, np.random.default_rng(3), sway_hz=0.13, q=q)
        P, fr = np.abs(np.fft.rfft(f - f.mean())) ** 2, np.fft.rfftfreq(n, 1 / amb.SR)
        share.append(P[(fr > 0.08) & (fr < 0.18)].sum() / P[fr < 2.0].sum())
    assert share[1] > share[0]


def test_salient_event_in_a_bed_is_found_and_stationary_noise_has_none():
    from genny.physical import ambience as amb
    rng = np.random.default_rng(0)
    x = 0.1 * rng.standard_normal(20 * amb.SR)
    assert amb.salient_events(x).size == 0
    x[7 * amb.SR:7 * amb.SR + 4800] += 3.0 * rng.standard_normal(4800)
    ev = amb.salient_events(x)
    assert ev.size == 1 and abs(ev[0] - 7.1) < 0.2


def test_creature_durations_against_the_measured_ranges():
    gasp = creatures.breath(48000, kind="gasp")
    grunt = creatures.grunt(48000)
    assert 1.1 <= len(gasp) / 48000 <= 2.6 and 0.3 <= len(grunt) / 48000 <= 0.8


def test_pyin_follows_a_glide_and_calls_noise_unvoiced():
    from genny import analysis
    sr = 16000
    rng = np.random.default_rng(0)
    t = np.arange(int(1.6 * sr)) / sr
    f = 220 * 2 ** (2.0 * np.sin(2 * np.pi * 0.7 * t) / 12)
    ph = 2 * np.pi * np.cumsum(f) / sr
    y = sum(np.sin(k * ph) / k for k in range(1, 6))
    y[int(0.6 * sr):int(1.0 * sr)] = 0.5 * rng.standard_normal(int(0.4 * sr))
    y += 0.05 * rng.standard_normal(y.size)
    f0 = analysis.pyin(y, sr)
    fr = np.arange(f0.size) * 0.01 + 0.02
    ok = (fr < 0.55) | (fr > 1.05)
    err = 1200 * np.abs(np.log2(f0[ok] / np.interp(fr[ok], t, f)))
    assert np.all(f0[ok] > 0) and np.median(err) < 10.0
    assert np.mean(f0[(fr > 0.65) & (fr < 0.95)] > 0) < 0.1


def test_webmushra_one_page_per_reference(tmp_path):
    from genny import calibrate
    from genny.core import write_wav
    for n in ("r1", "r2", "a", "b"):
        write_wav(tmp_path / f"{n}.wav", np.zeros(100), 22050)
    cfg = calibrate.webmushra({"bell": str(tmp_path / "r1.wav"), "gong": str(tmp_path / "r2.wav")},
                              {"bell": {"A": str(tmp_path / "a.wav")}, "gong": {"A": str(tmp_path / "b.wav")}}, str(tmp_path / "t"))
    text = cfg.read_text()
    assert text.count("type: mushra") == 2 and "id: trial2" in text and "bell_reference.wav" in text and "gong_A.wav" in text


def test_equalized_noise_levels_follow_the_requested_rate():
    from genny import cyber
    lv = np.zeros(32)
    lv[10] = 15.0
    x = cyber.equalized_noise(1 << 17, np.random.default_rng(0), lv, sr=24000)
    f = np.fft.rfftfreq(len(x), 1 / 24000)
    p = np.abs(np.fft.rfft(x)) ** 2
    cs = cyber.erb_centres(32)
    band = lambda g: p[np.abs(f - g) < 24.7 * (0.00437 * g + 1.0) / 2].mean()               # noqa: E731
    assert 9.0 < 10 * np.log10(band(cs[10]) / np.mean([band(cs[7]), band(cs[13])])) < 15.5          # the +15 dB, interpolated across the band
