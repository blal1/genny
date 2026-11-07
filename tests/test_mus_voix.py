"""Performance rules, FM voices, animal and human sounds checked against the values of the papers read on 2026-11-05."""
import math
from types import SimpleNamespace as NS

import numpy as np

from genny import compose as C


def _ev(ts, ps, dur=2.0):
    return [{"t": t, "dur": dur, "pitch": p, "vel": 0.5} for t, p in zip(ts, ps)]


def test_leap_articulation_micropause():
    # Friberg 1991 GMI 1A': DRO = 8 dN (0.3 (DR - 100)/600 + 0.3) k ms, dN capped at 9, not below 100 ms
    c = NS(q=0.1, meter=4.0, sections=[])
    a = _ev((0.0, 2.0, 4.0), (60.0, 67.0, 67.0))
    C._kth(a, c, {"leap_articulation": 1.0})
    assert math.isclose(a[0]["dur"], 2.0 - 8 * 7 * (0.3 * 100 / 600 + 0.3) / 100) and a[1]["dur"] == 2.0
    a = _ev((0.0, 2.0, 4.0), (60.0, 80.0, 80.0))
    C._kth(a, c, {"leap_articulation": 1.0})
    assert math.isclose(a[0]["dur"], 2.0 - 8 * 9 * 0.35 / 100)
    a = _ev((0.0, 2.0, 4.0), (60.0, 67.0, 67.0))
    C._kth(a, NS(q=0.01, meter=4.0, sections=[]), {"leap_articulation": 1.0})
    assert a[0]["dur"] == 2.0


def test_harmonic_charge():
    # Friberg 1991 GMA 2A: Charm = 2 (Cmel I/2 + Cmel III/3 + Cmel V/6) - 3 (G in C = 2); dL = 1.5 sqrt(Charm) k dB;
    # DR x sqrt(1 + 0.018 dL); +2 sqrt(Charm) k ms on every tone, +8 sqrt(Charm) k on the first over the chord
    c = NS(q=0.1, kpc=0.0, meter=4.0, chords=[{"root": 0.0, "pcs": [0.0, 4.0, 7.0]}, {"root": 7.0, "pcs": [7.0, 11.0, 2.0]}])
    a = _ev((0.0, 2.0, 4.0, 6.0, 8.0), (60.0,) * 5)
    C._kth(a, c, {"harmonic_charge": 1.0})
    dl = 1.5 * math.sqrt(2.0)
    assert a[0]["vel"] == 0.5 and math.isclose(a[1]["vel"], 0.5 * 10 ** (dl / 2 / 20))   # crescendo toward the G chord
    assert math.isclose(a[2]["vel"], 0.5 * 10 ** (dl / 20))
    ioi = 200 * math.sqrt(1 + 0.018 * dl) + 10 * math.sqrt(2.0)
    assert math.isclose(a[3]["t"] - a[2]["t"], ioi / 100) and a[1]["t"] == 2.0 and math.isclose(a[2]["t"] - a[1]["t"], 2.0 * math.sqrt(1 + 0.018 * dl / 2))


def test_kth_final_ritard():
    # Friberg 1991 GMA 3: dDR = 100 / sqrt(1.1 - Tn/Ttot) %, about 330 % on the last tone
    f = C.ritard_map({"beats": 4, "model": "kth"}, 16.0)
    assert f(12.0) == 12.0 and math.isclose(f(16.0) - f(12.0), 8 * (math.sqrt(1.1) - math.sqrt(0.1)))
    assert math.isclose(f(17.0) - f(16.0), 1 / math.sqrt(0.1)) and math.isclose((f(12.001) - f(12.0)) / 0.001, 1 / math.sqrt(1.1), rel_tol=1e-3)


def test_jazz_soloist_downbeat_delay():
    # Friberg & Sundstrom 2002 Fig. 6: downbeat lag falls with tempo (about 70 ms at 120 bpm, 15 at 300); off-beats together
    assert abs(C.soloist_delay(120) - 70.5) < 0.5 and abs(C.soloist_delay(300) - 15.4) < 0.5 and C.soloist_delay(360) == 0.0
    base = {"type": "compose", "role": "melody", "bars": 2, "seed": 3, "bpm": 120, "grid": 0.5}
    x, y = C.generate(base), C.generate({**base, "downbeat_delay_ms": 50.0})
    for e, g in zip(x, y):
        on = abs(e["t"] - round(e["t"])) < 1e-6
        assert math.isclose(g["t"] - e["t"], 0.1 if on else 0.0, abs_tol=1e-6)
    arr = C.arrange({"style": "jazz_swing", "bars": 2, "seed": 1, "bpm": 150, "melody_lead": True})
    lead = [p for p in arr["layers"] if p["name"] == "lead"][0]
    assert math.isclose(lead["downbeat_delay_ms"], C.soloist_delay(150)) and not any("offset_ms" in p for p in arr["layers"])


def test_phrase_arch_keeps_length_slow_ends():
    c = NS(q=0.1, meter=4.0, sections=[])
    a = _ev([float(i) for i in range(11)], (60.0,) * 11, 1.0)
    C._kth(a, c, {"phrase_arch": 1.0})
    t = np.array([e["t"] for e in a])
    d = np.diff(t)
    assert math.isclose(t[-1], 10.0) and d[0] > d[7] and d[-1] > d[7] and d[7] == d.min()
    assert math.isclose(d[0] - d[7], 0.1) and a[7]["vel"] > a[0]["vel"]


def test_ga_rhythm():
    # Ariza 2002: n 20, crossover 0.8, mutation 0.18, elitism 0.16; weights 1.5 1.5 2.33 1 0.66; (8,2,0) = (4,1,0)
    fit = list(C.GA_FIT)
    assert C.ga_fitness(fit, fit) == 0.0
    assert C.ga_fitness(fit[:4] + [(8, 2, 0)] + fit[5:], fit) == 0.0
    flip = fit[:3] + [(2, 1, 0)] + fit[4:]
    assert math.isclose(C.ga_fitness(flip, fit), 1.5 * 0.5 + 1.5 * 0.5 + 1.0 * 0.5)
    best = C.ga_rhythm(fit, 40, np.random.default_rng(1))
    f = [C.ga_fitness(b, fit) for b in best]
    assert all(b <= a + 1e-12 for a, b in zip(f, f[1:])) and f[-1] < f[0]
    ev = C.generate({"type": "compose", "role": "melody", "bars": 2, "seed": 2, "mode": "ga"})
    assert ev and all(e["pitch"] is not None for e in ev)


def test_mode_two_relative_spread():
    # Neda et al. 2000 eq. 6-7: D_I / D_II = 2.5 for half the rate, so the relative spread drops by 1.3
    from genny import creatures as K
    plan = K.applause_plan(64, 0.6, 12.0, seed=2, sync=1e-9)
    rates = np.array([1.0 / np.median(np.diff(p)) for p in plan])
    assert abs(rates.mean() / (0.5 * K.applause_rate(0.6)) - 1) < 0.03
    assert 0.06 < rates.std() / rates.mean() < 0.095


def test_footsteps_stop_and_fractal_gait():
    # Friberg & Sundberg 1999: stopping runners, v(x) = [1 + (v_end^q - 1) x]^(1/q), q 3, v_end 0.27 (Table V);
    # Hausdorff 1995: stride intervals with DFA alpha about 0.76
    from genny import creatures as K
    t = K.footsteps_plan("walk", 120.0, 9, 0.0, stop=4)
    f = C.ritard_map({"beats": 4, "q": 3.0, "v_end": 0.27}, 8.0)
    assert np.allclose(np.diff(t), 0.5 * np.diff([f(i) for i in range(9)]))
    assert math.isclose(t[-1] - t[-2], 0.5 * (f(8) - f(7))) and np.allclose(np.diff(t)[:4], 0.5)
    x = K._fractal(4096, 0.76, np.random.default_rng(0))
    y = np.cumsum(x - x.mean())
    ns = np.array([16, 32, 64, 128, 256])
    F = [np.sqrt(np.mean([np.mean((s - np.polyval(np.polyfit(np.arange(n), s, 1), np.arange(n))) ** 2)
                          for s in y[:len(y) // n * n].reshape(-1, n)])) for n in ns]
    assert abs(np.polyfit(np.log(ns), np.log(F), 1)[0] - 0.76) < 0.1


def test_chowning_envelopes():
    # Chowning 1973: brass Fig. 11 (0, 1 at P3/6, 0.75 at P3/3, 0.62 at 5P3/6, 0 at P3); drum Fig. 15 (0.8, 1 at 0.05 s,
    # 0.5 at 0.08, 0.15 at 0.1, 0.05 at 0.13, 0 at 0.2); wood drum index Fig. 16 falls linearly to 0 in 25 ms; bell exp(-3t/P3)
    from genny.chiptune2 import chowning
    sr = 48000
    pk = lambda y, t, w=0.006: np.max(np.abs(y[int((t - w) * sr):int((t + w) * sr)]))   # noqa: E731
    y = chowning(200.0, 1.2, sr, voice="brass")
    assert abs(pk(y, 0.2) - 0.5) < 0.01 and abs(pk(y, 0.4) / pk(y, 0.2) - 0.75) < 0.02 and abs(pk(y, 1.0) / pk(y, 0.2) - 0.62) < 0.02
    y = chowning(1000.0, 0.2, sr, voice="drum")
    for t, a in ((0.05, 1.0), (0.08, 0.5), (0.1, 0.15), (0.13, 0.05)):
        assert abs(pk(y, t, 0.0006) / 0.5 - a) < 0.03, t
    y = chowning(80.0, 0.2, sr, voice="wood_drum")
    t = np.arange(len(y)) / sr
    i = slice(int(0.005 * sr), int(0.02 * sr))
    amp = np.interp(t[i] / 0.2, [0.0, 0.25, 0.4, 0.5, 0.65, 1.0], [0.8, 1.0, 0.5, 0.15, 0.05, 0.0])
    want = 0.5 * amp * np.sin(2 * np.pi * 80 * t[i] + 25 * (1 - t[i] / 0.025) * np.sin(2 * np.pi * 55 * t[i]))
    assert np.allclose(y[i], want, atol=1e-6)
    y = chowning(200.0, 1.0, sr, voice="bell")
    assert abs(pk(y, 7.5) / pk(y, 0.3) - math.exp(-3 * 7.2 / 15)) < 0.03


def test_laje_labia_and_ellipse():
    # Laje, Gardner & Mindlin 2002: x' = y, y' = -k x - c x^2 y + b y - f0, c = 1e8; one ellipse in (b, k) per syllable,
    # phonation on the arc with b > 0; a large f0 (labia pushed together) stops the oscillation
    from genny import creatures as K
    k, b, f0 = K.laje_path(3000.0, 5000.0, 400, 200)
    kc, ka = 0.5 * ((2 * np.pi * 3000) ** 2 + (2 * np.pi * 5000) ** 2), 0.5 * ((2 * np.pi * 5000) ** 2 - (2 * np.pi * 3000) ** 2)
    assert np.allclose((b / b.max()) ** 2 + ((k - kc) / ka) ** 2, 1.0, atol=1e-3)
    assert np.all(f0[:400] == 0) and np.all(f0[401:] > 0) and k[0] < k[399]
    fs, n = 192000.0, 96000
    kk = np.full(n, (2 * np.pi * 4000.0) ** 2)
    y = K._laje_syrinx(kk, 0.2 * np.sqrt(kk), np.zeros(n), fs)[n // 2:]
    Y = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    assert abs(np.argmax(Y) * fs / y.size / 4000.0 - 1) < 0.02
    shut = K._laje_syrinx(kk, 0.2 * np.sqrt(kk), np.full(n, 2 * kk[0] * np.sqrt(0.2 * np.sqrt(kk[0]) / K.LAJE_C)), fs)
    assert np.abs(shut[n // 2:]).max() < 1e-3 * np.abs(y).max()
    s = K.birdsong(48000, species="canary", source="laje", seed=1)
    assert np.isfinite(s).all() and np.abs(s).max() > 0.1


def test_mammal_glottis_ladder_tract():
    # Wilkinson & Reiss 2016: glottal flow eq. 2 (continuous, shut after e2), complexity ladder by phase modulation
    # (level 3 = subharmonic), 15-section Kelly-Lochbaum tract
    from genny import creatures as K
    e1, e2 = 0.4, 0.56
    g = K.glottal_pulse(np.array([e1 - 1e-9, e1, e2, e2 + 1e-6, 0.9]))
    assert np.allclose(g[:2], 2 * (e2 - e1) / np.pi) and abs(g[2]) < 1e-9 and g[3] == 0 and g[4] == 0
    sr = K.KSR
    spec = lambda y, f: np.abs(np.fft.rfft(y * np.hanning(y.size)))[int(round(f * y.size / sr))]   # noqa: E731
    lo, hi = (K.mammal_call(200.0, 2.0, c, formants=0.0, noise=0.0)[sr // 2:sr * 3 // 2] for c in (1, 3))
    assert spec(hi, 100.0) > 0.05 * spec(hi, 200.0) and spec(lo, 100.0) < 1e-3 * spec(lo, 200.0)
    d = 50
    x = np.zeros(sr)
    x[0] = 1.0
    y = K._kl_tract(x, np.zeros(14), d, 0.75, 0.6)
    Y = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(y.size, 1 / sr)
    f1 = sr / (4 * 15 * d)
    band = (f > 0.5 * f1) & (f < 1.5 * f1)
    assert abs(f[band][np.argmax(Y[band])] / f1 - 1) < 0.05


def test_zwicker_iso532_reference_and_tilt():
    from genny import analysis as A
    sr = 48000
    t = np.arange(2 * sr) / sr
    tone = lambda db: np.sqrt(2) * 2e-5 * 10 ** (db / 20) * np.sin(2 * np.pi * 1000 * t)   # noqa: E731
    n40, ns = A.zwicker_loudness(tone(40), sr)
    assert abs(n40 - 1.0) < 0.02 and ns.shape == (240,) and 78 <= int(np.argmax(ns)) <= 91
    assert abs(A.zwicker_loudness(tone(50), sr)[0] / n40 - 2.0) < 0.15
    assert A.zwicker_loudness(tone(0), sr)[0] == 0.0
    w = np.random.default_rng(0).standard_normal(sr * 4)
    pink = np.fft.irfft(np.fft.rfft(w) / np.sqrt(np.maximum(np.fft.rfftfreq(w.size, 1 / sr), 1.0)), w.size)
    assert abs(A.spectral_tilt(w, sr)) < 0.3 and abs(A.spectral_tilt(pink, sr) + 3.0) < 0.3
    assert A.PESTANA_TILT == -5.0 and "tilt" in A.describe(w[:sr], sr)


def test_zwicker_sharpness_and_percentiles():
    from genny import analysis as A
    sr = 48000
    t = np.arange(sr) / sr
    s1k = A.zwicker_sharpness(A.zwicker_loudness(0.02 * np.sin(2 * np.pi * 1000 * t), sr)[1])
    s4k = A.zwicker_sharpness(A.zwicker_loudness(0.02 * np.sin(2 * np.pi * 4000 * t), sr)[1])
    assert 0.1 < s1k < s4k
    y = np.concatenate([0.0 * t[:sr // 2], np.sqrt(2) * 2e-5 * 100 * np.sin(2 * np.pi * 1000 * t[:sr // 2])])
    p = A.loudness_percentiles(y, sr)
    assert abs(p["N5"] - A.zwicker_loudness(y[sr // 2:], sr)[0]) < 0.1 and p["mean"] < p["N10"] <= p["N5"]


def test_zwicker_iso532_annex_b():
    # ISO 532-1 Annex B.2 test signal 1 (third-octave levels in, N = 83.296 sone) and B.3 test signal 3
    # (1 kHz 60 dB, N = 4.019 sone, compliance +-5 %)
    from genny import analysis as A
    lt = [-60, -60, 78, 79, 89, 72, 80, 89, 75, 87, 85, 79, 86, 80, 71, 70, 72, 71, 72, 74, 69, 65, 67, 77, 68, 58, 45, 30]
    assert abs(A.zwicker_from_bands(lt)[0] - 83.296) < 0.01
    sr = 48000
    t = np.arange(2 * sr) / sr
    n60 = A.zwicker_loudness(np.sqrt(2) * 2e-2 * np.sin(2 * np.pi * 1000 * t), sr)[0]
    assert abs(n60 / 4.019 - 1) < 0.05


def test_voiced_vot_by_place():
    # velar voiced VOT about twice the labial (Lisker & Abramson 1964; Cho & Ladefoged 1999)
    from genny import speech as S
    segs = S._build_segments(S.text_to_phrases("bah dah gah"), 1.0, 0.1)
    S._durations(segs, 1.0)
    heads = [sg.get("head", 0.0) for sg in segs if sg["ph"] != "_" and S.PH[sg["ph"]]["k"] in S.SONORANT]
    heads = [h for h in heads if h > 0]
    assert np.allclose(heads[:3], [0.006, 0.008, 0.012])


def test_belt_style_presses_and_raises_f1():
    from genny import choir as CH
    sr = 48000
    def centroid(style):
        y = CH.sing_phrase([(0.0, 69, 0.8, 1.0)], sr, stereo=False, lyrics="ah", style=style, singers=1)
        a = np.abs(np.fft.rfft(y[sr // 4:3 * sr // 4]))
        f = np.fft.rfftfreq(sr // 2, 1 / sr)
        return (a * f).sum() / a.sum()
    assert CH.STYLES["belt"]["f1"] > 1.1 and CH.STYLES["belt"]["rd"] < 0 and centroid("belt") > centroid("classical")


def test_analysis_helpers_yin_flux_mfcc_binaural_duration_jitter_tempo():
    from genny import analysis as A
    sr = 16000
    t = np.arange(sr) / sr
    assert abs(np.median(A.yin(np.sin(2 * np.pi * 220 * t), sr)) - 220) < 1.0
    y = np.where(t > 0.5, np.sin(2 * np.pi * 1000 * t), 0.0)
    fl = A.bark_flux(y, sr)
    assert abs(np.argmax(fl) * 0.0058 - 0.5) < 0.03
    assert A.mfcc(y, sr).shape[1] == 13
    m = 0.02 * np.sin(2 * np.pi * 1000 * t)
    assert abs(A.binaural_loudness(np.stack([m, m], 1), sr) / A.zwicker_loudness(m, sr)[0] - 1) < 1e-9
    assert A.duration_loudness_db(0.1) == 0.0 and abs(A.duration_loudness_db(0.02) + 6.99) < 0.01 and A.duration_loudness_db(1.0) == 0
    js = A.jitter_shimmer([0.01, 0.0101, 0.01, 0.0101], [1.0, 1.1, 1.0, 1.1])
    assert abs(js["jitter"] - 100 * 0.0001 / 0.01005) < 1e-9 and abs(js["shimmer_db"] - 20 * np.log10(1.1)) < 1e-9
    clicks = np.zeros(8 * sr)
    clicks[(np.arange(16) * 0.5 * sr).astype(int)] = 1.0
    assert abs(A.tempo_from_loudness(clicks, sr) - 120) < 3
