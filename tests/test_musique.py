"""Theme Musique: performance rules in compose, stinger intervals, adaptive one-shot states, extra tunings."""
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from genny import compose as C
from genny.core import read_wav


def test_swing_auto_follows_tempo():
    # Friberg & Sundstrom 2002: short note about 100 ms, ratio 2:1 near 200 bpm, 1:1 near 300, at most 3.5:1
    ratio = lambda bpm: (1 + C.swing_auto(60.0 / bpm)) / (1 - C.swing_auto(60.0 / bpm))   # noqa: E731
    assert abs(ratio(200) - 2.0) < 1e-9 and abs(ratio(300) - 1.0) < 1e-9 and abs(ratio(320) - 1.0) < 1e-9
    assert abs(ratio(40) - 3.5) < 1e-9
    for bpm in (160, 200, 240, 280):
        b = 60.0 / bpm
        short = b * (1 - (1 + C.swing_auto(b)) / 2)
        assert abs(short - 0.1) < 1e-9, bpm
    ev = C.generate({"type": "compose", "role": "drums", "style": "rock", "bars": 1, "seed": 1, "grid": 0.5,
                     "swing": "auto", "swing_grid": 0.5, "bpm": 200})
    offs = sorted({round(e["t"] % 1.0, 6) for e in ev})
    assert offs[0] == 0.0 and all(abs(o - 2 / 3) < 1e-6 for o in offs[1:]), offs


def test_final_ritard_formula():
    # Friberg & Sundberg 1999, eq. 2: v(x) = [1 + (v_end^q - 1) x]^(1/q); q = 3, v_end = 0.3 (listening test)
    f = C.ritard_map({"beats": 4, "v_end": 0.3, "q": 3}, 16.0)
    assert f(12.0) == 12.0 and f(8.0) == 8.0
    h = 1e-6
    for x in (0.0, 0.25, 0.5, 0.9):
        b = 12.0 + 4 * x
        tempo = h / (f(b + h) - f(b))
        assert abs(tempo - (1 + (0.3 ** 3 - 1) * x) ** (1 / 3)) < 1e-4, x
    assert abs(h / (f(16.0 + 2 * h) - f(16.0 + h)) - 0.3) < 1e-6
    base = {"type": "compose", "role": "melody", "bars": 4, "seed": 3, "bpm": 100}
    a, b = C.generate(base), C.generate({**base, "ritard": 4})
    assert [e["pitch"] for e in a] == [e["pitch"] for e in b]
    s0 = max(e["t"] for e in a) - 4
    assert all(abs(x["t"] - y["t"]) < 1e-9 for x, y in zip(a, b) if x["t"] < s0)
    late = [(x, y) for x, y in zip(a, b) if x["t"] >= s0]
    assert late and all(y["t"] > x["t"] for x, y in late if x["t"] > s0)
    arr = C.arrange({"style": "pop", "bars": 4, "seed": 1, "ritard": {"beats": 4}})
    r = arr["layers"][0]["ritard"]
    assert arr["beats"] > 16 and abs(arr["beats"] - C.ritard_map(r, 16.0)(16.0)) < 1e-9


def test_ritard_reaches_v_end_at_last_onset():
    # Friberg & Sundberg 1999: x = 1 at the onset of the last note, which is then played at v_end
    f = C.ritard_map({"beats": 4, "v_end": 0.3, "q": 3}, 16.0, 4.0, end=15.0)
    h = 1e-6
    assert f(11.0) == 11.0 and abs(h / (f(15.0) - f(15.0 - h)) - 0.3) < 1e-4
    assert abs((f(16.0) - f(15.0)) - 1 / 0.3) < 1e-9
    base = {"type": "compose", "role": "melody", "bars": 4, "seed": 3, "bpm": 100}
    a, b = C.generate(base), C.generate({**base, "ritard": 4})
    la, lb = max(a, key=lambda e: e["t"]), max(b, key=lambda e: e["t"])
    assert abs(lb["dur"] - la["dur"] / 0.3) < 1e-5, (la, lb)
    arr = C.arrange({"style": "pop", "bars": 4, "seed": 1, "ritard": 4})
    ends = {p["ritard"]["end"] for p in arr["layers"]}
    last = max(e["t"] for p in arr["layers"] for e in C.generate({**p, "ritard": None, "offset_ms": 0.0}))
    assert ends == {last}


def test_melody_lead_20_ms():
    # Friberg, Bresin & Sundberg 2006 (melodic lead): melody about 20 ms before the accompaniment
    arr = C.arrange({"style": "pop", "bars": 2, "seed": 1, "melody_lead": True}, 120)
    ms = {p["name"]: p.get("offset_ms", 0.0) for p in arr["layers"]}
    assert ms["lead"] == 0.0 and all(v == 20.0 for k, v in ms.items() if k != "lead")
    bass = next(p for p in arr["layers"] if p["name"] == "bass")
    plain = C.generate({k: v for k, v in bass.items() if k != "offset_ms"}, 120)
    moved = C.generate(bass, 120)
    d = [y["t"] - x["t"] for x, y in zip(plain, moved)]
    assert all(abs(v - 0.020 * 120 / 60) < 1e-6 for v in d)


def test_stinger_intervals():
    # Curtis & Bharucha 2010 via Phillips: rising minor second = threat, falling minor third = sadness
    for seed in range(4):
        for n in (2, 3, 5):
            d = [e["pitch"] for e in C.generate({"type": "stinger", "mood": "defeat", "length": n, "seed": seed})]
            assert d[-2] - d[-1] == 3, (seed, n, d)
            g = sorted(C.generate({"type": "stinger", "mood": "danger", "length": max(n, 3), "seed": seed}), key=lambda e: e["t"])
            first = [e["pitch"] for e in g if e["t"] == g[0]["t"]][0]
            second = [e["pitch"] for e in g if e["t"] > g[0]["t"]][0]
            assert second - first == 1, (seed, n)


def test_extra_tunings():
    ik = C.scale_offsets("ikeda_18")
    assert len(ik) == 18 and all(abs(b - a - 12 / 18) < 1e-6 for a, b in zip(ik, ik[1:]))
    assert abs(2 ** (1 / 18) - 1.0393) < 1e-4                         # +3.93 % per plate (Baschet Ikedaphone)
    py = sorted(C.RATIO_SCALES["pythagorean"])
    assert len(py) == 12 and abs(1.5 ** 12 / 2 ** 7 - 1.01364) < 1e-5 and 3 / 2 in py and 9 / 8 in py


def test_adaptive_oneshot_state():
    with tempfile.TemporaryDirectory() as d:
        L = {"type": "adaptive", "style": "ambient", "bpm": 120, "bars": 1, "seed": 2, "out_stems": d, "master": False,
             "states": {"intro": {"oneshot": True, "next": "loop", "tail": 0.5}, "loop": {}}, "order": ["intro", "loop"]}
        from genny import spec as SP
        y, _ = SP.render_layer(L, 22050)
        g = json.loads((Path(d) / "graph.json").read_text())
        assert g["states"]["intro"]["loop"] is False and g["states"]["intro"]["next"] == "loop"
        assert g["states"]["loop"]["loop"] is True
        assert np.isfinite(y).all()
        intro, loop = read_wav(Path(d) / "intro.wav")[0], read_wav(Path(d) / "loop.wav")[0]
        assert intro.shape[0] == int(2.5 * 22050) and loop.shape[0] == 2 * 22050   # one bar at 120 bpm, + the kept 0.5 s tail
        assert y.shape[0] == 4 * 22050                                             # the loop starts on the bar, over the tail


def _partials(y, f0, K, B, sr=44100):
    S = np.abs(np.fft.rfft(y * np.hanning(len(y)), 8 * len(y)))
    fr = np.fft.rfftfreq(8 * len(y), 1 / sr)
    out = []
    for k in range(1, K + 1):
        g = k * f0 * np.sqrt(1 + B * k * k)
        m = (fr > g - 0.3 * f0) & (fr < g + 0.3 * f0)
        out.append(fr[m][np.argmax(S[m])])
    return np.array(out)


def test_clav_dwg_dispersion_per_key():
    # Gabrielli et al. 2013, Table 1: B = 2e-4 at A#1, 9e-5 at D3, linearly interpolated between keys
    from genny import acoustic as A
    assert abs(A.clav_b(58.27) - 2e-4) < 1e-7 and abs(A.clav_b(146.83) - 9e-5) < 1e-7 and abs(A.clav_b(155.56) - 1e-4) < 1e-7
    for f0 in (58.27, 146.83):
        B = A.clav_b(f0)
        y = A._harpsi_string(f0, 66150, 66150, 44100, A.clav_pulse(0.8, 44100), 20.0, B, a=0.0, r=-0.006, rrate=0.4)[2205:]
        p, k = _partials(y, f0, 10, B), np.arange(1, 11)
        cents = 1200 * np.log2(p / (k * f0 * np.sqrt(1 + B * k * k)))
        assert np.max(np.abs(cents)) < 6.0, cents                      # stretch at k = 10: 17 and 8 cents
    stretch = 1200 * np.log2(_partials(A._harpsi_string(58.27, 66150, 66150, 44100, A.clav_pulse(0.8, 44100), 20.0, 2e-4, a=0.0, r=0.0)[2205:], 58.27, 10, 2e-4)[-1] / (10 * 58.27))
    assert stretch > 12.0


def test_clav_dwg_pulse_and_knock():
    # eq. 7: N = fs d / v, v from 1 m/s (soft) to 4 m/s (hard): the soft pulse is 4 times longer
    from genny import acoustic as A
    from genny.analysis import detect_clicks
    soft, hard = len(A.clav_pulse(0.0, 44100)), len(A.clav_pulse(1.0, 44100))
    assert abs(soft / hard - 4.0) < 0.1
    # knock: energy below 1200 Hz at the attack of a high key (E6, 1319 Hz), gone 100 ms later
    y = A.clav(1318.5, 0.4, 44100, 0.9, model="dwg")
    assert np.isfinite(y).all() and len(detect_clicks(y, 44100)) == 0

    def low(seg):
        S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        fr = np.fft.rfftfreq(len(seg), 1 / 44100)
        return np.sum(S[(fr > 150) & (fr < 1200)])
    assert low(y[:1323]) > 10 * low(y[4410:5733])


def test_sieve_and_tone_row():
    # Ariza, MIT 21M.380: 11@1 | 13@2 | 23@5 | 25@6 has period 82225 (lcm of the moduli)
    steps, period = C.sieve("11@1 | 13@2 | 23@5 | 25@6")
    assert period == 82225 and steps[:4] == [1, 2, 5, 6]
    assert C.sieve("(3@0 | 4@0) - 12@0", 12)[0] == [3, 4, 6, 8, 9]
    assert C.sieve("2@0 & 3@0", 12)[0] == [0, 6]
    ev = C.generate({"type": "euclid", "kind": "rim", "sieve": "3@0 | 4@1", "n": 12, "step": 0.25})
    assert sorted(round(e["t"] / 0.25) for e in ev) == [0, 1, 3, 5, 6, 9]
    # I = (12 - n) % 12 against the first note
    row = [0, 11, 7, 8, 3, 1, 2, 10, 6, 5, 4, 9]
    M = C.tone_row_matrix(row)
    assert M[0] == row and [r[0] for r in M] == [(12 - p) % 12 for p in row]
    assert all(sorted(r) == list(range(12)) for r in M) and all(sorted(col) == list(range(12)) for col in zip(*M))
    mel = C.generate({"type": "compose", "role": "melody", "mode": "row", "row": row, "scale": "chromatic", "bars": 2, "seed": 1,
                      "snap": False, "range": "C4-C6"})
    assert [int(e["pitch"]) % 12 for e in mel][:9] == row[:9]                   # P0 until the phrase cadence


def test_markov_orders_and_logistic():
    base = {"type": "compose", "role": "melody", "bars": 4, "seed": 5, "motif": "C4 D4 E4 G4 E4 D4 C4 A3"}
    for order in (0, 1, 2, 3, 4):
        ev = C.generate({**base, "mode": "markov", "order": order})
        assert len(ev) > 8, order
    # logistic map x(n+1) = r x (1 - x), r from 2.75 to 4 (Ariza): r = 2.75 settles on the fixed point 1 - 1/r
    ev = C.generate({"type": "compose", "role": "melody", "mode": "logistic", "r": 2.75, "x0": 0.3, "bars": 4, "seed": 1, "snap": False})
    P = C._pitches(0, C.scale_offsets("major"), 60, 84)
    fixed = P[round((1 - 1 / 2.75) * (len(P) - 1))]
    assert sum(e["pitch"] == fixed for e in ev[len(ev) // 2:]) >= 0.8 * (len(ev) - len(ev) // 2)


def test_risset_arpeggio_csound_values():
    # Csound Catalog version (Bain, Csound Journal 17): 9 oscillators f + k 0.03 Hz, GEN10 1 0 0 0 .7 x6, rise .02, decay .5
    from genny import instruments as I
    import genny.instruments_extra  # noqa: F401
    sr, f = 8000, 96.0
    y = I.render_note("risset_arpeggio", f, 4.0, sr, 1.0, delta=0.0)[sr // 2: 4 * sr]
    S = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    fr = np.fft.rfftfreq(len(y), 1 / sr)
    band = lambda h: S[(fr > h * f - 20) & (fr < h * f + 20)].max()   # noqa: E731
    assert all(band(h) < 1e-3 * band(1) for h in (2, 3, 4))
    assert all(abs(band(h) / band(1) - 0.7) < 0.01 for h in (5, 6, 7, 8, 9, 10))
    y = I.render_note("risset_arpeggio", f, 40.0, sr, 1.0)[: 40 * sr]
    S = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    fr = np.fft.rfftfreq(len(y), 1 / sr)
    m = (fr > 10 * f - 1.5) & (fr < 10 * f + 1.5)
    line, fl = S[m], fr[m]
    peaks = fl[[i for i in range(1, len(line) - 1) if line[i] > line[i - 1] and line[i] > line[i + 1] and line[i] > 0.3 * line.max()]]
    assert len(peaks) == 9 and np.allclose(np.diff(peaks), 10 * 0.03, atol=0.01)      # copies k 0.03 Hz apart, k = -4..4
    on = I.render_note("risset_arpeggio", f, 1.0, sr, 1.0)
    assert abs(len(on) / sr - 1.5) < 0.01 and abs(on[-1]) < 1e-3                       # 0.5 s decay after the note


def test_chowning_percussion_and_brass():
    # Chowning 1973: bell 15 s c 200 m 280 I 0-10; drum 0.2 s I 0-2; wood drum c 80 m 55 I 0-25; brass c = m = 440 I 0-5
    from genny.chiptune2 import chowning
    sr = 48000
    line = lambda y, f: np.abs(np.fft.rfft(y))[int(round(f * len(y) / sr))]   # noqa: E731
    bell = chowning(200.0, 1.0, sr, voice="bell")
    assert len(bell) == 15 * sr and np.isfinite(bell).all()
    head, tail = bell[:sr], bell[-2 * sr:-sr]
    for f in (200.0, 480.0, 80.0, 760.0):                                       # c, c + m, |c - m|, c + 2m
        assert line(head, f) > 0.01 * line(head, 200.0)
    assert line(tail, 480.0) / line(tail, 200.0) < line(head, 480.0) / line(head, 200.0)   # index falls with amplitude
    for v in ("drum", "wood_drum"):
        y = chowning(200.0, 0.05, sr, voice=v)
        assert len(y) == int(0.2 * sr) and np.max(np.abs(y[-200:])) < 0.01 * np.max(np.abs(y))
    brass = chowning(440.0, 0.6, sr, voice="brass")
    seg = brass[int(0.2 * sr):int(0.6 * sr)]
    assert line(seg, 880.0) > 0.05 * line(seg, 440.0) and line(seg, 660.0) < 0.01 * line(seg, 440.0)   # harmonic


def test_kth_high_loud_duration_contrast_rmd():
    # Friberg 1991 DPC1B: dL = (N - 60)/4 k dB; DDC1: 200 ms -> -16.5 ms, -0.825 dB; Repp 1995: IOI' = e^a IOI^b
    c = type("C", (), {"q": 0.1})()
    ev = lambda: [{"t": 2.0 * i, "dur": 2.0, "pitch": 60.0 + 12 * (i % 2), "vel": 0.5} for i in range(4)]   # noqa: E731
    a = ev()
    C._kth(a, c, {"high_loud": 1.0})
    assert math.isclose(a[1]["vel"] / a[0]["vel"], 10 ** (3 / 20)) and a[0]["vel"] == 0.5
    a = ev()
    C._kth(a, c, {"duration_contrast": 1.0})                  # IOI 2 beats x 0.1 s = 200 ms
    assert all(math.isclose(x["t"], i * (0.2 - 0.0165) / 0.1, abs_tol=1e-9) for i, x in enumerate(a))
    assert math.isclose(a[0]["vel"], 0.5 * 10 ** (-0.825 / 20)) and a[-1]["vel"] == 0.5
    a = [{"t": t, "dur": 1.0, "pitch": 60.0, "vel": 0.5} for t in (0.0, 1.0, 3.0, 7.0, 8.0)]
    C._rmd(a, c, {"rmd": 0.6, "rmd_unit": 100.0})
    d = np.diff([x["t"] for x in a])
    assert math.isclose(d.sum(), 8.0) and math.isclose(d[2] / d[1], 2 ** 0.6) and math.isclose(d[1] / d[0], 2 ** 0.6)
    b = [{"t": t, "dur": 0.5, "pitch": 60.0, "vel": 0.5} for t in (0.0, 1.0, 3.0, 7.0, 8.0)]
    C._rmd(b, c, {"rmd": 0.6})                                # Repp: IOIs cut in eighth notes first, all equal here
    assert all(math.isclose(x["t"], y, abs_tol=1e-9) for x, y in zip(b, (0.0, 1.0, 3.0, 7.0, 8.0)))
    base = {"type": "compose", "role": "melody", "bars": 2, "seed": 3, "bpm": 100}
    assert C.generate({**base, "rmd": 1.0}) == C.generate(base)


def test_kth_double_duration_floor_uphill_repetition_sharp():
    # Friberg 1991: DDC2B 12 % of the short tone moved from the long one; GMI 1C 2 ms; GMI 4' DRO 35 ms;
    # DPC1A 4 cents per octave. Friberg et al. 2006: duration contrast floor about 100 ms
    c = type("C", (), {"q": 0.1})()
    mk = lambda ts, ps, dur=1.0: [{"t": t, "dur": dur, "pitch": p, "vel": 0.5} for t, p in zip(ts, ps)]   # noqa: E731
    a = mk((0.0, 4.0, 6.0, 10.0), (60.0, 62.0, 64.0, 65.0))
    C._kth(a, c, {"double_duration": 1.0})
    assert [round(x["t"], 9) for x in a] == [0.0, round(4.0 - 0.24, 9), 6.0, 10.0]
    a = mk((0.0, 2.0, 4.0), (60.0, 60.0, 60.0))
    C._kth(a, c, {"duration_contrast": 10.0})
    assert math.isclose(a[1]["t"], 1.0) and math.isclose(a[2]["t"], 2.0)
    a = mk((0.0, 2.0, 4.0, 6.0), (60.0, 62.0, 64.0, 60.0))
    C._kth(a, c, {"faster_uphill": 1.0})
    assert np.allclose(np.diff([x["t"] for x in a]), [1.98, 1.98, 2.0])
    a = mk((0.0, 2.0, 4.0), (60.0, 60.0, 62.0), dur=2.0)
    C._kth(a, c, {"repetition": 1.0})
    assert math.isclose(a[0]["dur"], 2.0 - 0.35) and a[1]["dur"] == 2.0
    a = mk((0.0, 2.0), (60.0, 72.0))
    C._kth(a, c, {"high_sharp": 1.0})
    assert a[0]["pitch"] == 60.0 and math.isclose(a[1]["pitch"], 72.04)


def test_tension_glide_kirchhoff_carrier():
    # Bilbao 2009 (8.38): u_tt = gamma^2 G u_xx, G = 1 + alpha^2/2 ||u_x||^2; one mode: f = f0 sqrt(1 + alpha^2 pi^2 A^2 / 8)
    from genny.physical import modal as M
    sr, f0, d, alpha = M.SR, 440.0, 2.0, 1.0
    y = M.tension_glide(M.Modes([f0], [d], [1.0], "vdd"), 2 * sr, alpha)
    for t0 in (0.05, 0.5, 1.5):
        seg = y[int(t0 * sr):int((t0 + 0.1) * sr)]
        Y = np.abs(np.fft.rfft(seg * np.hanning(seg.size), 64 * seg.size))
        i = int(np.argmax(Y))
        p = 0.5 * (Y[i - 1] - Y[i + 1]) / (Y[i - 1] - 2 * Y[i] + Y[i + 1])
        f = (i + p) * sr / (64 * seg.size)
        want = f0 * math.sqrt(1 + alpha ** 2 * math.pi ** 2 * math.exp(-2 * d * (t0 + 0.05)) / 8)
        assert abs(f / want - 1) < 2e-3, (t0, f, want)
    assert M.tension_glide(M.Modes([f0, 2 * f0], [d, d], [0.0, 0.0], "vdd"), 100, alpha).max() == 0.0


def test_kth_inegales_leap_charge_phrase_rmd_auto():
    # Friberg 1991: GMI 3 22 %; GMI 1B (thesis erratum) up -4.2 / +4.2 sqrt(dN) ms, down +2.4 / -4.2; DPC 2A 0.2 Cmel dB and max 4.3 % at Cmel 6.5;
    # GMA 1 phrase end +40 ms and DRO 80 ms, last tone +80 ms. Repp 1995: b 0.8 at the fast tempo (durations x 0.75)
    from types import SimpleNamespace as NS
    c = NS(q=0.1, kpc=0.0, meter=4.0, chords=[{"root": 0.0}, {"root": 0.0}],
           sections=[NS(bar0=0, nbars=1), NS(bar0=1, nbars=1)])
    mk = lambda ts, ps: [{"t": t, "dur": 1.0, "pitch": p, "vel": 0.5} for t, p in zip(ts, ps)]   # noqa: E731
    a = mk((0.0, 2.0, 4.0, 6.0, 8.0), (60.0,) * 5)
    C._kth(a, c, {"inegales": 1.0})
    assert np.allclose([x["t"] for x in a], [0.0, 2.44, 4.0, 6.44, 8.0])
    a = mk((0.0, 2.0, 4.0, 6.0, 8.0), (60.0, 62.0, 69.0, 67.0, 65.0))
    C._kth(a, c, {"leap_duration": 1.0})
    assert np.allclose([x["t"] for x in a], [0.0, 2.0, 4.0 - 0.042 * math.sqrt(7), 6.0, 8.0])
    a = mk((0.0, 2.0, 4.0), (60.0, 66.0, 60.0))
    C._kth(a, c, {"melodic_charge": 1.0})
    assert math.isclose(a[1]["vel"], 0.5 * 10 ** (1.2 / 20)) and a[0]["vel"] == 0.5
    assert math.isclose(a[2]["t"] - a[1]["t"], 2.0 * 1.04)                   # F#: Cmel 6, 2/3 x 6 = 4 %
    a = mk((0.0, 2.0, 4.0, 6.0), (60.0,) * 4)
    for x in a:
        x["dur"] = 2.0
    C._kth(a, c, {"kth_phrase": 1.0})
    assert math.isclose(a[2]["t"], 4.4) and math.isclose(a[1]["dur"], 2.4 - 0.8) and math.isclose(a[3]["dur"], 2.8)
    a = [{"t": t, "dur": 0.5, "pitch": 60.0, "vel": 0.5} for t in (0.0, 1.0, 3.0, 7.0)]
    C._rmd(a, c, {"rmd": "auto", "rmd_ref_bpm": 450.0, "rmd_unit": 100.0})
    d = np.diff([x["t"] for x in a])
    assert math.isclose(d[1] / d[0], 2 ** 0.8)
    keys = dict(inegales=1, leap_duration=1, melodic_charge=1, kth_phrase=1, rmd="auto", high_sharp=1, double_duration=1,
                duration_contrast=1, faster_uphill=1, repetition=1, high_loud=1)
    ev = C.generate({"type": "compose", "role": "melody", "bars": 4, "seed": 3, "bpm": 100, "style": "pop", **keys})
    assert ev and C.generate({"type": "compose", "role": "melody", "bars": 4, "seed": 3, "bpm": 100, "style": "pop"}) != ev


def test_koenig_selection_tendency_urn():
    rng = np.random.default_rng(0)
    u = C.urn(range(7), 21, rng)
    assert all(sorted(u[i:i + 7]) == list(range(7)) for i in (0, 7, 14))       # nothing returns before all the others
    g = C.select("abc", 30, rng, "group", group=3)
    assert all(len(set(g[i:i + 3])) == 1 for i in range(0, 30, 3))
    assert C.select([1, 2], 5, rng, "sequence") == [1, 2, 1, 2, 1]
    assert set(C.select([1, 2], 50, rng, "ratio", weights=[1, 0])) == {1}
    x = C.tendency([[0, 60], [1, 72]], [[0, 64], [1, 84]], 200, rng)
    lo, hi = np.linspace(60, 72, 200), np.linspace(64, 84, 200)
    assert np.all(x >= lo - 1e-9) and np.all(x <= hi + 1e-9)


def test_hiller_monte_carlo_ca_lsystem():
    # Ariza, monteCarlo.py: 14 diatonic pitches, 36 notes, steps kept only between min and max
    idx = C.monte_carlo(36, 14, np.random.default_rng(3), 1, 3)
    assert len(idx) == 36 and all(1 <= abs(b - a) <= 3 for a, b in zip(idx, idx[1:]))
    ca = C.elementary_ca(90, 21, 3)
    assert list(np.flatnonzero(ca[1])) == [9, 11] and list(np.flatnonzero(ca[2])) == [8, 12]   # rule 90 = XOR of neighbours
    # the algae grammar a -> ab, b -> a: lengths follow Fibonacci
    assert [len(C.lsystem("a{3}b{-2} @ a{ab} b{a} @ a", k)) for k in range(6)] == [1, 2, 3, 5, 8, 13]
    for mode, extra in (("montecarlo", {}), ("ca", {"rule": 30}), ("lsystem", {}), ("koenig", {"select": "series"}),
                        ("tendency", {"mask_lo": [[0, 60], [1, 72]], "mask_hi": [[0, 67], [1, 84]]}), ("markov", {"order": 1.5})):
        ev = C.generate({"type": "compose", "role": "melody", "bars": 2, "seed": 4, "mode": mode, **extra})
        assert ev and all(e["pitch"] is not None for e in ev), mode


def test_mood_map_and_numeric_stinger():
    m = C.mood_params(*C.MOOD_POINTS["calm"])
    assert abs(m["bpm"] - 43.2) < 1e-9 and abs(m["root"] - 48.8) < 1e-9 and m["scale"] == "major"
    assert abs(m["brightness"] - (600 + 2600 * (0.5 + 0.21 - 0.15))) < 1e-9 and abs(m["detune"] - 0.0475) < 1e-9
    assert C.mood_params(*C.MOOD_POINTS["threat"])["scale"] == "chromatic" and C.mood_params(-0.5, 0, 0)["scale"] == "phrygian"
    tri = C.mood_params(0.9, 0.7, 0.15)
    ev = C.generate({"type": "stinger", "mood": [0.9, 0.7, 0.15]})
    on_scale = lambda x: min(abs(x % 12 - o) for o in (0, 2, 4, 5, 7, 9, 11, 12)) < 1e-6   # noqa: E731
    assert all(on_scale(e["pitch"] - tri["root"]) or on_scale(e["pitch"] - tri["root"] - tri["detune"]) for e in ev)
    assert any(not on_scale(e["pitch"] - tri["root"]) for e in ev)                      # the detuned voice
    spec = C.to_spec({"type": "stinger", "mood": "sad"})
    assert spec["fx"][-1]["type"] == "lowpass"
    for mood in ("alarm", "settle"):
        ev = sorted(C.generate({"type": "stinger", "mood": mood, "seed": 2}), key=lambda e: e["t"])
        p = [e["pitch"] for e in ev]
        assert sorted(p) == [60, 61, 62, 63, 64] and (p == sorted(p, reverse=True)) == (mood == "settle")
        assert abs(max(e["t"] + e["dur"] for e in ev) - 2.0) < 1e-6


def test_oaml_tension_conditions_and_fades():
    assert [C.tension_step(t) for t in (100, 88, 50, 2, 1, 0)] == [88, 78, 43, 0, 0, 0]
    assert C.condition_met({"type": "range", "value": [10, 20]}, 20) and not C.condition_met({"type": "less", "value": 5}, 5)
    st = {"a": {"xfade_out": 0.3}, "b": {}}
    assert C.xfade_times({"xfade_in": 0.2}, st, "a", "b") == (0.2, 0.3)
    st["b"] = {"xfade_in": 0.1, "xfade_out": 0.05}
    assert C.xfade_times({"xfade_in": 0.2}, st, "a", "b") == (0.1, 0.05)
    st["b"] = {"xfade_in": 0, "xfade_out": 0}                                   # an explicit 0 is a dry cut
    assert C.xfade_times({"xfade_in": 0.2}, st, "a", "b") == (0.0, 0.0)


def test_kth_phrase_key_is_not_the_phrase_length():
    from types import SimpleNamespace as NS
    c = NS(q=0.1, kpc=0.0, meter=4.0, sections=[NS(bar0=0, nbars=1), NS(bar0=1, nbars=1)])
    a = [{"t": t, "dur": 2.0, "pitch": 60.0, "vel": 0.5} for t in (0.0, 2.0, 4.0, 6.0)]
    ref = [dict(e) for e in a]
    C._kth(a, c, {"phrase": 4})                                                  # phrase length in bars: no GMA 1
    assert a == ref
    base = {"type": "compose", "role": "melody", "bars": 4, "seed": 3, "bpm": 100, "sections": "AB"}
    x, y = C.generate(base), C.generate({**base, "kth_phrase": 0.5})
    assert [e["pitch"] for e in x] == [e["pitch"] for e in y]                     # same phrases (2 bars), not 1-bar ones
    assert [(e["t"], e["dur"]) for e in x] != [(e["t"], e["dur"]) for e in y]


def test_high_sharp_keeps_leap_steps():
    from types import SimpleNamespace as NS
    c = NS(q=0.1, kpc=0.0, meter=4.0)
    a = [{"t": t, "dur": 1.0, "pitch": p, "vel": 0.5} for t, p in zip((0.0, 2.0, 4.0, 6.0, 8.0), (60.0, 62.0, 69.0, 67.0, 65.0))]
    C._kth(a, c, {"leap_duration": 1.0, "high_sharp": 1.0})
    assert np.allclose([x["t"] for x in a], [0.0, 2.0, 4.0 - 0.042 * math.sqrt(7), 6.0, 8.0])
    assert math.isclose(a[1]["pitch"], 62.0 + 0.04 * 2 / 12)


def test_monte_carlo_without_admissible_degree_fails_fast():
    import threading
    from genny.spec import SpecError
    res = []

    def run():
        try:
            C.monte_carlo(5, 14, np.random.default_rng(0), 8, 9, start=7)
        except SpecError as e:
            res.append(e)
    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(5.0)
    assert not th.is_alive() and res
    idx = C.monte_carlo(40, 14, np.random.default_rng(1), 7, 9, start=0)
    assert all(7 <= abs(b - a) <= 9 for a, b in zip(idx, idx[1:]))


def test_adaptive_markers_tails_zone_and_layer_stinger():
    from genny import spec as SP
    with tempfile.TemporaryDirectory() as d:
        L = {"type": "adaptive", "style": "ambient", "bpm": 120, "bars": 2, "seed": 2, "out_stems": d, "master": False, "tails": True,
             "states": {"calm": {"condition": {"type": "less", "value": 30}}, "zone": {"xfade_in": 0.1}},
             "transitions": [{"from": "calm", "to": "zone", "layer": {"type": "drum", "kind": "kick"}}],
             "order": ["calm", {"state": "zone", "bars": 1}]}
        y, _ = SP.render_layer(L, 22050)
        g = json.loads((Path(d) / "graph.json").read_text())
        assert g["states"]["calm"]["markers"] == [0, 44100] and g["states"]["calm"]["meter"] == 4
        assert g["states"]["calm"]["condition"]["type"] == "less" and (Path(d) / "calm__tail.wav").exists()
        assert g["transitions"][0]["stinger"] == "stinger_calm_to_zone.wav"
        assert y.shape[0] >= 3 * 22050 and np.isfinite(y).all()


def test_sync_phase_drift_trend_expression_accent_group_silence():
    # Repp 2005 eq. 1-2 with beta 0: a(n+1) = (1 - alpha) a(n) + noise, lag-1 correlation 1 - alpha
    a = C.sync_asynchrony(np.full(20000, 0.5), np.random.default_rng(1), 0.01, alpha=0.6)
    assert abs(np.corrcoef(a[1:], a[:-1])[0, 1] - 0.4) < 0.03
    base = {"type": "compose", "role": "drums", "style": "rock", "bars": 4, "seed": 1, "grid": 0.5, "swing": 0.0}
    plain = C.generate(base)
    ph = C.generate({**base, "phase": {"cycle": 4, "shift": 0.25}})
    kept = [e for e in plain if e["t"] + 0.25 * (e["t"] // 4) < 16 - 1e-9]
    assert sorted(round(e["t"] + 0.25 * (e["t"] // 4), 6) for e in kept) == sorted(e["t"] for e in ph)
    tp = sorted({e["t"] for e in plain})
    td = sorted({e["t"] for e in C.generate({**base, "drift_trend": 0.12})})
    slope = lambda i: (td[i + 1] - td[i]) / (tp[i + 1] - tp[i])                        # noqa: E731
    assert len(tp) == len(td) and abs(slope(0) / slope(len(tp) - 2) - 1.12) < 0.01 and td[-1] < 16
    hum = {**base, "humanize": 3}
    assert [e["t"] for e in C.generate({**hum, "expression": 0})] == [e["t"] for e in plain]
    assert [e["t"] for e in C.generate({**hum, "expression": 1})] == [e["t"] for e in C.generate(hum)]
    acc = C.generate({**base, "kit": {"snare": "X-x-"}, "accent": 5, "fill_every": 0})
    v = sorted({e["vel"] for e in acc})
    assert abs(v[-1] / v[0] - 5.0) < 0.02
    assert C.generate({**base, "intensity": 0}) == []
    m = {"type": "compose", "role": "melody", "bars": 8, "seed": 5, "group": "lead", "group_size": 2}
    bars = [{int(e["t"] // 4) for e in C.generate({**m, "group_index": i})} for i in (0, 1)]
    assert not bars[0] & bars[1] and len(bars[0]) == len(bars[1]) == 4


def test_chromatic_charge_and_cents():
    from genny.notes import to_freq
    assert abs(to_freq("A4+100c") - to_freq("A#4")) < 1e-9 and abs(to_freq("440hz-1200c") - 220.0) < 1e-9
    # Friberg 1991 GMA 2B: C = [(32 - dN) mod 12] - 6; a line of semitones has C = 1 everywhere: +1.35 dB
    base = {"type": "compose", "role": "melody", "bars": 1, "mode": "motif", "motif": "C4 C#4 D4 D#4 E4 F4"}
    a, b = C.generate(base), C.generate({**base, "chromatic_charge": 0.1})
    pt = [e["pitch"] for e in a]
    cc = [((32 - abs(y - x)) % 12) - 6 for x, y in zip(pt, pt[1:])] + [np.nan]
    for m, (x, y) in enumerate(zip(a, b)):
        cm = np.nanmean(cc[max(0, m - 2):m + 3])
        assert abs(y["vel"] / x["vel"] - 10 ** (1.35 * cm * 0.1 / 20)) < 2e-3, m
