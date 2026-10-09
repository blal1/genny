"""genny.compose: generators are deterministic, stay in key and range, share harmony across parts,
export to ordinary layers sample for sample; the Karplus-Strong voices obey their decay and tuning laws.

Run:  uv run --with pytest pytest tests/test_compose.py -q     (or: uv run python tests/test_compose.py)
"""
import copy
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from genny import compose as C
from genny import drums as D
from genny import filters as F
from genny import instruments as I
from genny import spec as SP
from genny.analysis import brightness, loudness, sharpness
from genny.notes import midi_to_freq, note_to_midi

SR = 44100
ROLES = ("melody", "bass", "chords", "arp", "drums", "pad", "counter")
BASE = {"type": "compose", "key": "D", "scale": "dorian", "progression": "i VII VI VII", "bars": 8, "seed": 7}


def cents_off(y, f0, sr=SR):
    """Pitch error from the autocorrelation peak next to the expected period (as tests/test_instruments.py)."""
    y = y[int(0.05 * sr):int(0.05 * sr) + 16384]
    y = y - y.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(y * np.hanning(len(y)), 4 * len(y))) ** 2)
    p = sr / f0 * 4
    lo = int(p * 0.94)
    k = lo + int(np.argmax(ac[lo:int(p * 1.06) + 2]))
    k = k + 0.5 * (ac[k - 1] - ac[k + 1]) / (ac[k - 1] - 2 * ac[k] + ac[k + 1] + 1e-20)
    return 1200 * np.log2((sr * 4 / k) / f0)


def in_scale(pitch, key_pc, offs):
    return any(abs((pitch - key_pc - o + 6) % 12 - 6) < 1e-4 for o in offs)


# ------------------------------------------------------------------ rhythm and theory
def test_euclid():
    assert C.steps_str(C.euclid(3, 8)) == "x--x--x-"
    assert C.steps_str(C.euclid(5, 8)) == "x-xx-xx-"
    assert C.steps_str(C.euclid(3, 8, 1)) == "-x--x--x"
    assert C.steps_str(C.euclid(4, 16)) == "x---x---x---x---"
    for n in range(1, 20):
        for k in range(0, n + 1):
            pat = C.euclid(k, n)
            assert len(pat) == n and sum(pat) == k
            if k:                                                   # maximally even: gaps differ by at most one step
                on = [i for i, p in enumerate(pat) if p]
                gaps = [(on[(i + 1) % k] - on[i]) % n or n for i in range(k)]
                assert max(gaps) - min(gaps) <= 1, (k, n, gaps)
    assert C.steps_str(C.euclid(3, 8)) == C.CLAVES["tresillo"] and C.steps_str(C.euclid(5, 8)) == C.CLAVES["cinquillo"]


def test_scales_and_tunings():
    for name in ("harmonic_minor", "melodic_minor", "hijaz", "in", "hirajoshi", "pelog", "slendro", "whole_tone",
                 "octatonic", "blues", "bebop", "ptolemy", "seven_eleven", "7-11", "slendro_equal"):
        offs = C.scale_offsets(name)
        assert offs[0] == 0 and offs == sorted(offs) and all(0 <= o < 12 for o in offs), name
    assert C.scale_offsets("hijaz") == [0, 1, 4, 5, 7, 8, 10] and C.scale_offsets("in") == [0, 1, 5, 7, 8]
    # Ptolemy's syntonic diatonic: 1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8; the "7-11": 6:7:8:9:11
    pt = [2 ** (o / 12) for o in C.scale_offsets("ptolemy")]
    assert np.allclose(pt, [1, 9 / 8, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 15 / 8], atol=1e-6)
    assert np.allclose([2 ** (o / 12) for o in C.scale_offsets("7-11")], [1, 7 / 6, 4 / 3, 3 / 2, 11 / 6], atol=1e-6)
    assert np.allclose([2 ** (o / 12) for o in C.scale_offsets([1, "9/8", 1.25, "3/2"])], [1, 1.125, 1.25, 1.5], atol=1e-6)
    assert C.scale_offsets({"cents": [0, 204, 702]}) == [0, 2.04, 7.02]
    assert abs(2 ** (C.scale_offsets("major", "just")[4] / 12) - 1.5) < 1e-6
    # a just fifth reaches the exported note as 3/2 of the tonic, in Hz
    L = dict(BASE, role="pad", key="A", scale="ptolemy", progression="I", bars=1, range="A3-A4")
    hz = sorted(float(n[:-2]) if n.endswith("hz") else midi_to_freq(note_to_midi(n)) for n in C.to_spec(L)["layers"][0]["steps"][0]["notes"])
    assert abs(hz[0] - 220.0) < 1e-3 and any(abs(h / hz[0] - 1.5) < 1e-4 for h in hz) and any(abs(h / hz[0] - 1.25) < 1e-4 for h in hz), hz
    assert C.quantize(61, "C", "major") in (60, 62) and C.quantize(66.2, "C", "major") == 67


def test_chords_and_voice_leading():
    offs = C.scale_offsets("major")
    prog = C.parse_progression("I IV V7 vi", offs)
    assert [c["pcs"] for c in prog] == [[0, 4, 7], [5, 9, 0], [7, 11, 2, 5], [9, 0, 4]]
    assert C.chord_pcs(0, C.scale_offsets("octatonic"), 4) == [0, 3, 6, 9]              # a diminished seventh
    assert C.voice_lead([60, 64, 67], [5, 9, 0]) == [60, 65, 69]                         # C -> F: common tone held
    assert C.voice_lead([60, 64, 67], [7, 11, 2]) == [59, 62, 67]                        # C -> G
    moves = []
    prev = None
    for ch in C.parse_progression("I vi IV V I vi ii V", offs):
        v = C.voice_lead(prev, ch["pcs"])
        if prev:
            moves.append(max(min(abs(a - b) for b in prev) for a in v))
        prev = v
    assert max(moves) <= 4, moves                                                        # no voice jumps more than a third
    for s in range(5):
        toks = C.functional_progression(np.random.default_rng(s), 4).split()
        assert toks[0] == "I" and toks[-1] in ("V", "vii")


def test_motif_operators():
    m = [(0, 1.0), (2, 0.5), (4, 0.5)]
    assert C.transpose(m, 2) == [(2, 1.0), (4, 0.5), (6, 0.5)]
    assert C.invert(m) == [(0, 1.0), (-2, 0.5), (-4, 0.5)]
    assert C.retrograde(m) == m[::-1] and C.augment(m, 2) == [(0, 2.0), (2, 1.0), (4, 1.0)]
    assert C.retrograde(C.retrograde(m)) == m and C.invert(C.invert(m)) == m


# ------------------------------------------------------------------ generators
def test_deterministic():
    for role in ROLES:
        L = dict(BASE, role=role, style="funk", humanize=0.5, swing=0.1)
        a, b = C.generate(L, 104), C.generate(copy.deepcopy(L), 104)
        assert a == b and len(a) > 0, role
        assert C.generate(dict(L, seed=8), 104) != a, role
    for L in ({"type": "euclid", "k": 5, "n": 16, "prob": 0.6, "seed": 1}, {"type": "stinger", "mood": "mystery", "seed": 1},
              {"type": "arp", "chords": "C4:maj", "mode": "random", "seed": 1}):
        assert C.generate(L) == C.generate(dict(L)) and C.generate(dict(L, seed=2)) != C.generate(L), L["type"]
    sc = {"type": "scatter", "layers": [{"type": "drum", "kind": "rim"}], "count": 9, "dur": 5, "pitch": 3, "seed": 3}
    assert C.to_spec(sc) == C.to_spec(dict(sc)) and C.to_spec(dict(sc, seed=4)) != C.to_spec(sc)


def test_in_scale_and_range():
    n_notes = n_approach = 0
    for scale in ("dorian", "hijaz", "in", "octatonic", "ptolemy", "harmonic_minor", "blues"):
        offs = C.scale_offsets(scale)
        combos = [(r, {}) for r in ("melody", "bass", "chords", "arp", "pad", "counter")]
        combos += [("melody", {"mode": m}) for m in ("markov", "voss", "collatz", "motif", "call_response")]
        combos += [("bass", {"pattern": p}) for p in ("walking", "acid", "syncopated", "octave", "tumbao")]
        combos += [("arp", {"mode": m}) for m in ("pool", "updown", "random")] + [("chords", {"pattern": "broken"})]
        for role, extra in combos:
            L = dict(BASE, role=role, scale=scale, progression="i iv v i", seed=3, **extra)
            ev = C.generate(L, 110)
            assert ev, (scale, role, extra)
            for e in ev:
                n_notes += 1
                if e.get("approach"):
                    n_approach += 1
                    continue
                assert in_scale(e["pitch"], 2, offs), (scale, role, extra, e)
    assert n_approach > 0                                           # walking lines do use declared chromatic approach tones
    print(f"  in-scale: {n_notes} notes checked, {n_approach} declared chromatic approach tones")
    for mode in ("walk", "markov", "voss", "collatz", "motif", "call_response"):
        for rg, lo, hi in (("C4-C6", 60, 84), ("G3-D5", 55, 74), ("E5-E6", 76, 88)):
            ev = C.generate(dict(BASE, role="melody", mode=mode, range=rg, density=0.8, seed=11), 100)
            ps = [e["pitch"] for e in ev]
            assert min(ps) >= lo and max(ps) <= hi, (mode, rg, min(ps), max(ps))


def test_chord_tones_on_strong_beats():
    hit = tot = 0
    for seed in range(12):
        L = dict(BASE, role="melody", seed=seed, density=0.7)
        har = C.harmony(L)
        for e in C.generate(L):
            bar, pos = divmod(e["t"], 4.0)
            if pos in (0.0, 2.0):                                   # beats 1 and 3
                tot += 1
                hit += any(abs((e["pitch"] - pc + 6) % 12 - 6) < 1e-6 for pc in har[int(bar)]["pcs"])
    print(f"  chord tones on strong beats: {hit}/{tot} = {100 * hit / tot:.0f} %")
    assert tot > 100 and hit / tot >= 0.7


def test_walking_bass():
    for style, key in (("jazz_swing", "F"), ("blues_shuffle", "E"), (None, "D")):
        L = {"type": "compose", "role": "bass", "pattern": "walking", "key": key, "bars": 12, "seed": 5}
        if style:
            L["style"] = style
        har, ev = C.harmony(L), C.generate(L)
        for b in range(12):
            first = [e for e in ev if abs(e["t"] - 4.0 * b) < 1e-9]
            assert len(first) == 1 and abs((first[0]["pitch"] - har[b]["root"] + 6) % 12 - 6) < 1e-6, (style, b)
            assert len([e for e in ev if 4.0 * b <= e["t"] < 4.0 * b + 4]) == 4                 # four to the bar
        steps = [abs(b["pitch"] - a["pitch"]) for a, b in zip(ev, ev[1:])]
        assert max(steps) <= 12 and np.mean(steps) < 5, (max(steps), np.mean(steps))             # a line, not leaps
        for a, b in zip(ev, ev[1:]):                                                             # approach = a half step from the target
            if a.get("approach"):
                assert abs(b["pitch"] - a["pitch"]) == 1


def test_sources_markov_voss():
    # order 2 reproduces only transitions that exist in the motif
    motif = [0, 2, 4, 2, 0, -1, 0, 4]
    ev = C.generate(dict(BASE, role="melody", mode="markov", order=2, motif=motif, snap=False, scale="major", key="C",
                         progression="I", range="C3-C7", density=1.0, bars=8))
    allp = C._pitches(0, C.scale_offsets("major"), 0, 127)
    idx = [allp.index(e["pitch"]) for e in ev]
    rel = [i - idx[0] for i in idx]
    pairs = {(motif[i], motif[(i + 1) % len(motif)]) for i in range(len(motif))}
    seen = set(zip(rel, rel[1:]))
    ends = {i for i, e in enumerate(ev[:-1]) if ev[i + 1]["t"] - e["t"] > 1.01}           # phrase cadences are snapped
    inner = {(rel[i], rel[i + 1]) for i in range(len(rel) - 2) if i not in ends and i + 1 not in ends and i + 2 < len(rel) - 1}
    assert len(inner & pairs) >= 0.8 * len(inner), (inner - pairs)
    assert len(seen) > 3
    # 1/f: neighbouring values are correlated (white noise: about 0), yet it is not a slow walk
    v = C.voss(4096, np.random.default_rng(0))
    r1 = np.corrcoef(v[:-1], v[1:])[0, 1]
    w = np.random.default_rng(0).random(4096)
    assert 0.3 < r1 < 0.95 and abs(np.corrcoef(w[:-1], w[1:])[0, 1]) < 0.1, r1
    spec = np.abs(np.fft.rfft(v - v.mean())) ** 2
    lo, hi = spec[1:64].mean(), spec[512:2048].mean()
    print(f"  voss: lag-1 autocorrelation {r1:.2f}, low/high band power ratio {10 * np.log10(lo / hi):.1f} dB")
    assert lo > 4 * hi


def test_form_and_intensity():
    # "A A B A": the A sections come back note for note, B differs
    L = dict(BASE, role="melody", sections="A A B A", progression={"A": "i VII", "B": "iv v"}, bars=8)
    ev = C.generate(L)
    sec = lambda k: [(round(e["t"] - 8 * k, 6), e["pitch"], e["dur"]) for e in ev if 8 * k <= e["t"] < 8 * k + 8]   # noqa: E731
    assert sec(0) == sec(1) and sec(0) != sec(2)
    assert [(a, b) for a, b, _ in sec(0)][:-1] == [(a, b) for a, b, _ in sec(3)][:-1]      # last A ends on the root
    last = max(ev, key=lambda e: e["t"])
    assert abs((last["pitch"] - 2) % 12 - C.harmony(L)[-1]["root"] + 2) % 12 < 1e-6 or (last["pitch"] - C.harmony(L)[-1]["root"]) % 12 == 0
    har = C.harmony(L)
    assert [h["roman"] for h in har] == ["i", "VII", "i", "VII", "iv", "v", "i", "VII"]
    # intensity mutes parts under their threshold: vertical layering from one score
    a = C.arrange({"type": "arrangement", "style": "synthwave", "bars": 8, "seed": 2, "sections": "A B", "intensity": [0.3, 1.0]})
    for p in a["layers"]:
        ev = C.generate(p, a["bpm"])
        early = [e for e in ev if e["t"] < 16]
        assert (len(early) > 0) == (p["threshold"] <= 0.3), (p["name"], len(early))
        assert any(e["t"] >= 16 for e in ev), p["name"]
        full = C.generate(dict(p, intensity=[1.0]), a["bpm"])                           # the quiet version is a subset of the full one
        assert {(e["t"], e["pitch"], e.get("drum")) for e in ev} <= {(e["t"], e["pitch"], e.get("drum")) for e in full}


def test_performance():
    L = dict(BASE, role="arp", bars=2)
    straight = C.generate(L)
    swung = C.generate(dict(L, swing=0.3))
    d = [round(b["t"] - a["t"], 6) for a, b in zip(straight, swung)]
    assert set(d[0::2]) == {0.0} and set(d[1::2]) == {0.075}                            # odd sixteenths late by 0.3 of a step
    hum = C.generate(dict(L, humanize={"time": 8, "vel": 0.13}), 120)
    dt = np.array([b["t"] - a["t"] for a, b in zip(straight, hum)])[1:] * 0.5 * 1000    # beats -> ms at 120 bpm
    vr = np.array([b["vel"] / a["vel"] for a, b in zip(straight, hum)])
    print(f"  humanize: onset jitter sd {dt.std():.1f} ms (asked 8), velocity ratio {vr.min():.2f}..{vr.max():.2f}")
    assert 5 < dt.std() < 11 and 0.86 <= vr.min() and vr.max() <= 1.0 + 1e-6
    # tempo drift bends time but keeps the length: +-4 % (MIT 21M.380), ends on the bar line
    dr = C.generate(dict(BASE, role="pad", bars=16, drift=0.04), 100)
    st = C.generate(dict(BASE, role="pad", bars=16), 100)
    off = np.array([a["t"] - b["t"] for a, b in zip(dr, st)])
    assert np.max(np.abs(off)) > 0.05 and max(e["t"] + e["dur"] for e in dr) <= 64 + 1e-6
    local = np.diff(sorted({e["t"] for e in dr})) / 4.0
    assert 0.955 < local.min() and local.max() < 1.045, (local.min(), local.max())


def test_drums_and_fills():
    ev = C.generate({"type": "compose", "role": "drums", "style": "funk", "bars": 8, "seed": 1})
    kick = sorted(e["t"] for e in ev if e["drum"] == "kick" and e["t"] < 4)
    assert kick == [0.0, 0.75, 2.5, 3.25]                                               # X--x------X--x--
    fills = [e for e in ev if e.get("fill")]
    assert fills and all(int(e["t"] // 4) in (3, 7) and e["t"] % 4 >= 2 for e in fills)
    assert not any(e.get("fill") for e in C.generate({"type": "compose", "role": "drums", "style": "funk", "bars": 8, "fill_every": 0}))
    tri = C.generate({"type": "compose", "role": "drums", "style": "jazz_swing", "bars": 1, "fill_every": 0})
    assert sorted({round(e["t"] % 1, 3) for e in tri}) == [0.0, 0.667]                  # swung: beats and last triplets only


# ------------------------------------------------------------------ arrangement
def test_styles_table():
    assert len(C.STYLES) == 35
    lib = Path(__file__).resolve().parents[1] / "examples" / "library" / "styles.json"
    if lib.exists():
        names = {s["out"][:-4] for s in json.loads(lib.read_text(encoding="utf-8"))["sounds"]}
        assert names == set(C.STYLES)
    for name, st in C.STYLES.items():
        a = C.arrange({"type": "arrangement", "style": name, "bars": 4, "seed": 1})
        assert st.bpm[0] <= a["bpm"] <= st.bpm[1] and a["bpm"] == C.style_bpm(name, 1)
        offs, kpc = C.scale_offsets(st.scale), 0
        for p in a["layers"]:
            if p["role"] == "drums":
                for e in C.generate(p, a["bpm"]):
                    assert e["drum"].split(":")[0] in D.REGISTRY, (name, e["drum"])
            else:
                assert p["inst"] in I.REGISTRY, (name, p["inst"])
                ev = C.generate(p, a["bpm"])
                assert ev and all(e.get("approach") or in_scale(e["pitch"], kpc, offs) for e in ev), (name, p["name"])
                assert max(e["t"] + e["dur"] for e in ev) <= a["beats"] + 1e-6


def test_arrangement_length_and_harmony():
    L = {"type": "arrangement", "style": "synthwave", "key": "A", "bars": 4, "seed": 3, "sections": "A B"}
    a = C.arrange(L, 100)
    assert a["beats"] == 4 * 4 and len(a["layers"]) >= 4
    hs = [C.harmony(p, a["bpm"]) for p in a["layers"]]
    assert all(h == hs[0] for h in hs) and C.harmony(L, 100) == hs[0]                    # every part: the same progression
    assert len({h["roman"] for h in hs[0]}) > 1
    bass = C.generate(next(p for p in a["layers"] if p["name"] == "bass"), a["bpm"])
    for b in range(4):                                                                   # ...and the bass really plays it
        first = min((e for e in bass if 4 * b <= e["t"] < 4 * b + 4), key=lambda e: e["t"])
        assert (first["pitch"] - hs[0][b]["root"]) % 12 == 0
    chord_pcs = {b: set() for b in range(4)}
    for p in a["layers"]:
        if p["name"] in ("chords", "arp"):
            for e in C.generate(p, a["bpm"]):
                chord_pcs[int(e["t"] // 4)].add(e["pitch"] % 12)
    assert all(chord_pcs[b] <= set(hs[0][b]["pcs"]) for b in range(4))
    for sr, bpm in ((22050, 100), (44100, 93)):
        n = int(round(4 * 4 * 60.0 / bpm * sr))
        y, _ = SP.render_layer(dict(L, fold=True), sr, bpm)
        assert y.shape[0] == n and np.all(np.isfinite(y)) and 1e-3 < np.max(np.abs(y)) <= 1.0, (y.shape, n)
    y0, _ = SP.render_layer(dict(L, tail=0), 22050, 100)
    assert y0.shape[0] == int(round(16 * 0.6 * 22050)) and np.max(np.abs(y0[-1])) < 1e-6
    full, _ = SP.render_layer(L, 22050, 100)
    assert full.shape[0] > int(round(16 * 0.6 * 22050))                                  # default keeps the ring-out
    own, _ = SP.render_layer(dict(L, fold=True), 22050, None)                            # no tempo given: the style's own
    assert own.shape[0] == int(round(16 * 60.0 / C.style_bpm("synthwave", 3) * 22050))
    spec = C.to_spec(dict(L, fold=True), 100, 22050)
    assert spec["type"] == "group" and all(l["type"] == "group" and all(k["type"] in ("seq", "pattern") for k in l["layers"]) for l in spec["layers"])


def test_automix_panner():
    assert [C.pan_steps(r) for r in (1, 2, 3, 5, 6, 8)] == [[64], [0, 127], [64, 0, 127], [64, 32, 95, 0, 127],
                                                              [51, 76, 25, 102, 0, 127], [54, 73, 36, 91, 18, 109, 0, 127]]
    assert all(abs(a - b) <= 1 for a, b in zip(C.pan_steps(4), [43, 84, 0, 127]))       # Table 1 prints 43/84
    sr, t = 48000, np.arange(48000) / 48000.0
    tone = lambda f: np.sin(2 * np.pi * f * t)                                            # noqa: E731
    # the paper's test idea: equal tones share a band and are pushed apart, lows stay in the middle
    sig = [tone(125), tone(125), tone(5000), tone(5000), tone(5000), tone(1000), tone(1000), tone(1000), tone(1000), tone(10000)]
    assert [C.band_class(s, sr) for s in sig] == [2, 2, 7, 7, 7, 5, 5, 5, 5, 8]
    pos = C.automix_pan(sig, sr)
    midi = [int(round((p + 1) / 2 * 127)) for p in pos]
    print("  automix pan (MIDI 0..127):", midi)
    assert midi[:2] == [64, 64] and midi[2:5] == [64, 0, 127] and midi[5:9] == [42, 85, 0, 127] and midi[9] == 64
    assert C.band_class(np.zeros(1000), sr) is None and C.automix_pan([np.zeros(100)], sr) == [0.0]
    gated = np.concatenate([tone(3000)[:9600], 1e-4 * tone(100)])                        # the quiet tail is under the -60 dB gate
    assert C.band_class(gated, sr) == 7
    out = C.automix([{"type": "synth", "inst": "sine", "notes": "C6", "dur": 0.4}, {"type": "synth", "inst": "sine", "notes": "D6", "dur": 0.4},
                     {"type": "drum", "kind": "kick"}], 22050)
    assert [l.get("fx") for l in out] == [[{"type": "pan", "pos": -1.0}], [{"type": "pan", "pos": 1.0}], None]


# ------------------------------------------------------------------ export
def test_to_spec_renders_identically():
    worst = 0.0
    cases = [dict(BASE, role=r, style="funk", bars=2, humanize=0.7, swing=0.12, drift=0.03, gain=0.8,
                  inst={"melody": "marimba", "bass": "bass", "chords": "epiano", "arp": "synth_pluck", "pad": "strings", "counter": "sine"}.get(r))
             for r in ROLES]
    cases += [dict(BASE, role="melody", inst="sine", scale="ptolemy", bars=2, at=1.0, fx=[{"type": "lowpass", "cutoff": 3000}]),
              {"type": "euclid", "kind": "hihat", "k": 5, "n": 16, "bars": 2, "accent": "X--x", "swing": 0.2},
              {"type": "arp", "inst": "marimba", "chords": "C4:maj7 A3:min", "mode": "updown", "octaves": 2},
              {"type": "stinger", "mood": "unlock", "seed": 2, "inst": "marimba"},
              {"type": "scatter", "layers": [{"type": "drum", "kind": "woodblock"}, {"type": "sfx", "kind": "click"}], "count": 10, "dur": 4,
               "pitch": [-3, 3], "gain_db": [-9, 0], "pan": 0.7, "seed": 4, "repeat": 2, "every": 4}]
    for L in cases:
        for sr, bpm in ((44100, 104), (22050, 97.3), (48000, None)):
            a, at_a = SP.render_layer(L, sr, bpm)
            sp = C.to_spec(L, bpm)
            assert json.loads(json.dumps(sp)) == sp                                      # plain JSON
            kinds = {k["type"] for k in sp["layers"]}
            assert kinds <= ({"group"} if L["type"] == "scatter" else {"seq", "pattern"}), kinds
            b, at_b = SP.render_layer(sp, sr, bpm)
            assert a.shape == b.shape and at_a == at_b, (L.get("role", L["type"]), a.shape, b.shape)
            worst = max(worst, float(np.max(np.abs(a - b))))
            assert np.all(np.isfinite(a)) and np.max(np.abs(a)) > 1e-4
    print(f"  to_spec vs generator layer: max abs difference {worst:.2e}")
    assert worst < 1e-6
    full = SP.render_spec({"bpm": 104, "beats": 8, "layers": [cases[0], cases[4]]})[0]   # and through the whole spec renderer
    assert np.all(np.isfinite(full)) and full.shape[0] > 0


def test_scatter_and_layers():
    kid = {"type": "drum", "kind": "rim"}
    sp = C.to_spec({"type": "scatter", "layers": [kid], "count": 15, "dur": 6, "seed": 1})
    assert len(sp["layers"]) == 15 and all(0 <= l["at"] < 6 for l in sp["layers"])
    iv = [l["at"] for l in C.to_spec({"type": "scatter", "layers": [kid], "interval": [0.5, 1.0], "dur": 30, "seed": 1})["layers"]]
    gaps = np.diff(iv)
    assert gaps.min() >= 0.5 - 1e-6 and gaps.max() <= 1.0 + 1e-6 and 28 < len(iv) < 62           # FMOD scatterer window
    po = [l["at"] for l in C.to_spec({"type": "scatter", "layers": [kid], "rate": 8, "dur": 60, "seed": 2})["layers"]]
    g = np.diff(po)
    print(f"  scatter rate 8/s: {len(po) / 60:.2f} events/s, gap cv {g.std() / g.mean():.2f} (Poisson: 1)")
    assert abs(len(po) / 60 - 8) < 1.0 and 0.85 < g.std() / g.mean() < 1.15
    half = C.to_spec({"type": "scatter", "layers": [kid], "count": 400, "dur": 6, "seed": 1, "silence_p": 0.5})["layers"]
    assert 150 < len(half) < 250
    rs = C.to_spec({"type": "scatter", "layers": [{"type": "drum", "kind": "ks_drum"}], "count": 4, "seed": 1, "reseed": True})["layers"]
    assert len({l["layers"][0]["params"]["seed"] for l in rs}) == 4
    # stingers: shape by mood, `length` in notes
    for mood in C.MOODS:
        for n in (3, 5, 7):
            ev = C.generate({"type": "stinger", "mood": mood, "length": n, "key": "D"})
            assert len({e["t"] for e in ev}) == n, (mood, n)
    line = lambda mood: [e["pitch"] for e in C.generate({"type": "stinger", "mood": mood, "length": 5})]   # noqa: E731
    v, d = line("victory"), line("defeat")
    assert all(b > a for a, b in zip(v[:4], v[1:5])) and {p % 12 for p in v} == {0, 4, 7} and v[4] % 12 == 0
    assert all(b < a for a, b in zip(d, d[1:])) and d[-1] % 12 == 0 and {p % 12 for p in d} <= {0, 2, 3, 5, 7, 8, 10}
    # arp modes
    arp = lambda mode, **k: [e["pitch"] for e in C.generate({"type": "arp", "chords": "C4:maj", "mode": mode, "rate": 0.5, "beats": 3, **k})]   # noqa: E731
    assert arp("up") == [60, 64, 67, 60, 64, 67] and arp("down") == [67, 64, 60, 67, 64, 60]
    assert arp("updown") == [60, 64, 67, 64, 60, 64] and arp("pattern", pattern=[0, 2, 1, 2]) == [60, 67, 64, 67, 60, 67]
    assert set(arp("random", seed=3)) <= {60, 64, 67} and arp("up", octaves=2)[:6] == [60, 64, 67, 72, 76, 79]
    # Farnell slot sequencer over the unresolved chord lists: bass alternates slots 1 and 2, half-chance gate
    sl = C.generate({"type": "compose", "role": "bass", "mode": "slots", "slots": [1, 2], "mask": "x---x---x---?---", "flip": 16,
                     "progression": "pokesdown", "key": "D", "scale": "mixolydian", "bars": 24, "seed": 1})
    assert {e["pitch"] for e in sl if e["t"] < 4} == {50.0} and {e["pitch"] for e in sl if 4 <= e["t"] < 8} == {52.0}
    assert {e["pitch"] for e in sl if 64 <= e["t"] < 68} == {52.0}                       # third list: 52 48 ...
    gated = sum(1 for e in sl if e["t"] % 4 == 3.0)
    assert 5 <= gated <= 19                                                              # about half of 24 bars
    assert all(in_scale(p, 2, C.scale_offsets("mixolydian")) for ch in C.POKESDOWN for p in ch)
    pool = C.generate({"type": "compose", "role": "arp", "mode": "pool", "progression": "pokesdown", "key": "D", "scale": "mixolydian", "bars": 16, "seed": 2})
    rests = 1 - len(pool) / (16 * 16)
    print(f"  Farnell arp pool: rest fraction {rests:.3f} (1/12 = 0.083)")
    assert 0.04 < rests < 0.13 and {e["pitch"] for e in pool if e["t"] < 32} <= set(C.POKESDOWN[0]) | {p + 12 for p in C.POKESDOWN[0][5:8]}
    acid = C.generate({"type": "compose", "role": "bass", "pattern": "acid", "style": "acid_techno", "bars": 2, "seed": 4})
    assert any(e["slide"] for e in acid) and any(e["accent"] for e in acid) and all(len(e["tuple303"]) == 4 for e in acid)
    assert C.tempo_delay(120, 3)["time"] == 0.75                                         # three eighths at 120


def test_adaptive_stems():
    with tempfile.TemporaryDirectory() as d:
        L = {"type": "adaptive", "style": "synthwave", "key": "A", "bars": 2, "seed": 2, "out_stems": d,
             "states": {"explore": {"intensity": [0.3]}, "combat": {"intensity": [1.0]}},
             "transitions": [{"from": "explore", "to": "combat", "stinger": "danger", "inst": "marimba"}, {"from": "combat", "to": "explore"}],
             "order": ["explore", "combat", "explore"]}
        sr, bpm = 22050, 100
        y, _ = SP.render_layer(L, sr, bpm)
        n = int(round(8 * 0.6 * sr))
        assert y.shape[0] >= 3 * n and np.all(np.isfinite(y)) and 1e-3 < np.max(np.abs(y)) <= 1.0
        files = sorted(p.name for p in Path(d).iterdir())
        graph = json.loads((Path(d) / "graph.json").read_text())
        assert set(graph["states"]) == {"explore", "combat"} and graph["transitions"][0]["stinger"] == "stinger_explore_to_combat.wav"
        from genny.core import read_wav
        ex, cb = read_wav(Path(d) / "explore.wav")[0], read_wav(Path(d) / "combat.wav")[0]
        assert ex.shape[0] == cb.shape[0] == n                                           # synchronised loops of equal length
        rms = lambda x: float(np.sqrt(np.mean(x ** 2)))                                  # noqa: E731
        print(f"  adaptive: {len(files)} files, explore rms {20 * np.log10(rms(ex)):.1f} dB, combat rms {20 * np.log10(rms(cb)):.1f} dB")
        assert rms(cb) > 1.1 * rms(ex)                                                   # one gain: intensity survives
        assert rms(read_wav(Path(d) / "explore__drums.wav")[0]) < 1e-6 < rms(read_wav(Path(d) / "combat__drums.wav")[0])
        a, b = read_wav(Path(d) / "explore__chords.wav")[0], read_wav(Path(d) / "combat__chords.wav")[0]
        corr = float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))            # the same score underneath, played softer
        assert corr > 0.95 and rms(a) < rms(b), corr
        bad = SP.render_layer(dict(L, states={"../x y": {}}, transitions=[], order=None), sr, bpm)
        assert sorted(p.name for p in Path(d).iterdir() if p.name.startswith(".._x")) == [".._x_y.wav"] or (Path(d) / "___x_y.wav").exists()
        assert not (Path(d).parent / "x y.wav").exists() and bad[0].shape[0] == n


# ------------------------------------------------------------------ Karplus-Strong voices
def test_ks_string_tuning_shape_level():
    lo, hi = (note_to_midi(n) for n in I.REGISTRY["ks_string"]["range"].split("-"))
    worst = 0.0
    for kw in ({}, {"stretch": 4.0}, {"pick": 0.5, "decay": 1.0}, {"sympathetic": 0.3}, {"loss": 0.97}):
        for m in (lo, (lo + hi) // 2, hi):
            f = midi_to_freq(m)
            c = cents_off(I.render_note("ks_string", f, 0.6, SR, 0.9, **kw), f)
            worst = max(worst, abs(float(c)))
            assert abs(c) < 10, (kw, m, round(float(c), 2))
    for sr in (22050, 48000):
        for m in (lo, hi):
            f = midi_to_freq(m)
            assert abs(cents_off(I.render_note("ks_string", f, 0.6, sr, 0.9), f, sr)) < 10, (sr, m)
    print(f"  ks_string: worst tuning error {worst:.2f} cents")
    for m in (lo - 5, lo, (lo + hi) // 2, hi, hi + 7):
        for dur, vel, sr in ((0.03, 1.0, 44100), (0.1234, 0.3, 44100), (0.3333, 0.7, 22050), (1.0, 1.0, 48000)):
            y = I.render_note("ks_string", midi_to_freq(m), dur, sr, vel, sympathetic=0.2 if dur > 0.3 else 0.0)
            assert y.ndim == 1 and np.all(np.isfinite(y)) and 1e-4 < np.max(np.abs(y)) < 1.5 and abs(y[-1]) < 1e-3 and abs(y[0]) < 0.05, (m, dur)
            assert abs(np.mean(y)) < 2e-3
    assert np.array_equal(I.render_note("ks_string", 220.0, 0.3, SR, 0.8, seed=3), I.render_note("ks_string", 220.0, 0.3, SR, 0.8, seed=3))
    assert not np.array_equal(I.render_note("ks_string", 220.0, 0.3, SR, 0.8, seed=3), I.render_note("ks_string", 220.0, 0.3, SR, 0.8, seed=4))
    levels = []
    for m in (lo + (hi - lo) // 4, (lo + hi) // 2, hi - (hi - lo) // 4, min(hi, 96)):
        f = midi_to_freq(m)
        y = I.render_note("ks_string", f, 0.5, SR, 0.9)
        levels.append(loudness(y, SR, window=0.15))
        s, b = sharpness(y, SR), brightness(y, SR)
        assert s < max(sharpness(I.render_note("sine", f, 0.5, SR, 0.9), SR) + 1.0, 1.5) and s < 2.4 and b["above_5k"] < 0.03, (m, s, b)
    print("  ks_string: level", [round(v, 1) for v in levels], "dB K-weighted at vel 0.9")
    assert abs(sum(levels[:3]) / 3 + 12.0) < 3.0
    soft, hard = (brightness(I.render_note("ks_string", 196.0, 0.5, SR, v), SR)["centroid"] for v in (0.2, 1.0))
    assert hard > 1.1 * soft                                                             # the dynamics filter: harder = brighter


def _t60(y, sr, f0, a=0.15, b=0.9, q=8.0):
    env = np.abs(F.bandpass(y, f0, sr, q=q))
    w = int(0.03 * sr)
    r = np.array([np.sqrt(np.mean(env[i:i + w] ** 2)) for i in range(int(a * sr), int(b * sr), w)])
    slope = np.polyfit(np.arange(len(r)) * w / sr, 20 * np.log10(r + 1e-12), 1)[0]
    return -60.0 / slope


def test_ks_string_physics():
    out = []
    for f, t60 in ((110.0, 1.0), (110.0, 3.0), (440.0, 0.8), (880.0, 2.0), (1318.5, 0.6), (1318.5, 3.0), (2093.0, 2.0)):
        Nd, Cc, S, rho = C.js_design(f, SR, t60)                    # the first five are shortened (rho), the last two stretched (S)
        assert abs(Cc) < 1 and 0 < S <= 0.5 and rho < 1 and (S < 0.5) == (f > 1000 and t60 > 1)
        got = _t60(C.js_string(f, 2 * SR, SR, t60), SR, f, 0.15, 1.5)
        out.append((f, t60, round(got, 2), round(S, 3), round(rho, 5)))
        assert abs(got / t60 - 1) < 0.12, out[-1]
        # loop phase delay: N + P_s + P_c = fs/f1 with the allpass delay from its phase response
        w = 2 * np.pi * f / SR
        z = np.exp(-1j * w)
        pc = -np.angle((Cc + z) / (1 + Cc * z)) / w
        ps = -np.angle((1 - S) + S * z) / w
        assert abs(Nd + ps + pc - SR / f) < 1e-6, (f, Nd + ps + pc, SR / f)
    print("  ks_string T60 (f, asked, measured, S, rho):", out)
    # eq. 21 and decay stretching: with the plain string (S = 1/2) partial k dies as cos(pi k f T) per period
    f = 220.0
    y1 = C.js_string(f, SR, SR, 5000.0, stretch=1.0, pick=0.0, bright=9000, level_hz=9000)
    y8 = C.js_string(f, 2 * SR, SR, 5000.0, stretch=8.0, pick=0.0, bright=9000, level_hz=9000)
    k = 12
    pred = math.log(1e-3) / (f * math.log(math.cos(math.pi * k * f / SR)))               # seconds for -60 dB
    got1, got8 = _t60(y1, SR, k * f, 0.05, 0.5, 40.0), _t60(y8, SR, k * f, 0.05, 1.9, 40.0)
    print(f"  ks_string partial {k}: T60 predicted {pred:.2f} s, measured {got1:.2f} s; with stretch 8: {got8:.2f} s (x{got8 / got1:.1f})")
    assert abs(got1 / pred - 1) < 0.15 and 6.0 < got8 / got1 < 10.0
    # dynamics filter, pitch-equalised: the gain at every fundamental equals the gain at the reference pitch
    L = 800.0
    rl = math.exp(-2 * math.pi * L / SR)
    gain = lambda R, fr: abs((1 - R) / (1 - R * np.exp(-2j * np.pi * fr / SR)))          # noqa: E731
    gm = gain(rl, C._KS_FM)
    for fr in (82.4, 164.8, 329.6, 659.3, 1318.5):
        R = C.dynamics_R(fr, SR, L)
        assert 0 <= R < 1 and abs(gain(R, fr) - gm) < 1e-9, (fr, R)
    assert abs(C.dynamics_R(C._KS_FM, SR, L) - rl) < 1e-9
    # pick position 1/2 removes the even harmonics
    y = C.js_string(110.0, SR, SR, 4.0, pick=0.5, bright=6000, level_hz=6000)[2000:2000 + 16384]
    sp = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    h = lambda k: sp[int(round(k * 110.0 * len(y) / SR)) - 2:int(round(k * 110.0 * len(y) / SR)) + 3].max()   # noqa: E731
    odd, even = np.mean([h(k) for k in (1, 3, 5, 7)]), np.mean([h(k) for k in (2, 4, 6, 8)])
    print(f"  ks_string pick 0.5: even harmonics {20 * np.log10(even / odd):.0f} dB below the odd ones")
    assert even < 0.1 * odd
    # a sympathetic string an octave below rings on after the note is damped
    dry = I.render_note("ks_string", 220.0, 0.15, SR, 0.9)
    wet = I.render_note("ks_string", 220.0, 0.15, SR, 0.9, sympathetic=0.4)
    tail = slice(int(0.5 * SR), int(0.9 * SR))
    assert np.sqrt(np.mean(wet[tail] ** 2)) > 3 * np.sqrt(np.mean(dry[tail] ** 2))


def test_ks_drum():
    out = []
    for p in (20, 50, 200, 400):                                                        # rms halves every 2p+1 samples
        w = 2 * p + 1
        slopes = []
        for seed in range(6):
            y = C.ks_drum_raw(24 * w, p, 0.5, 1.0, seed)
            r = np.array([np.sqrt(np.mean(y[i * w:(i + 1) * w] ** 2)) for i in range(1, 24)])
            slopes.append(np.polyfit(np.arange(len(r)), 20 * np.log10(r + 1e-300), 1)[0])
        out.append((p, round(float(np.mean(slopes)), 2)))
        assert abs(np.mean(slopes) + 6.02) < 0.9, out[-1]
    print("  ks_drum dB per 2p+1 samples (theory -6.02):", out)
    # blend 1 is the plucked string at fs/(p+1/2); blend 0 an octave lower with odd harmonics only
    p, fi = 100, C._KS_FI
    s1, s0 = C.ks_drum_raw(fi, p, 1.0, 1.0, 1), C.ks_drum_raw(fi, p, 0.0, 1.0, 1)
    assert abs(cents_off(s1, fi / (p + 0.5), fi)) < 3 and abs(cents_off(s0, fi / (2 * p + 1), fi)) < 3
    sp = np.abs(np.fft.rfft(s0[2000:2000 + 16384] * np.hanning(16384)))
    f0 = fi / (2 * p + 1)
    h = lambda k: sp[int(round(k * f0 * 16384 / fi)) - 2:int(round(k * f0 * 16384 / fi)) + 3].max()   # noqa: E731
    assert np.mean([h(2), h(4), h(6)]) < 0.05 * np.mean([h(1), h(3), h(5)])
    # the registered drum: decays in about `decay` seconds at any rate, stretch lengthens it
    for sr in (22050, 44100, 48000):
        for decay in (0.1, 0.25, 0.5):
            y = D.render_drum("ks_drum", sr, 1.0, decay=decay, tone=9000)
            w = int(0.01 * sr)
            r = np.array([np.sqrt(np.mean(y[i:i + w] ** 2)) for i in range(int(0.01 * sr), int(decay * 0.8 * sr), w)])
            t60 = -60.0 / np.polyfit(np.arange(len(r)) * w / sr, 20 * np.log10(r), 1)[0]
            assert abs(t60 / decay - 1) < 0.3, (sr, decay, round(t60, 3))
            assert np.sqrt(np.mean(y[-len(y) // 8:] ** 2)) < 0.05 * np.sqrt(np.mean(y[:len(y) // 8] ** 2))
    a, b = (C.ks_drum_raw(8000, 60, 0.5, s, 2) for s in (1.0, 4.0))
    assert np.sqrt(np.mean(b[4000:] ** 2)) > 10 * np.sqrt(np.mean(a[4000:] ** 2))
    assert np.array_equal(D.render_drum("ks_drum", SR, 0.8, seed=5), D.render_drum("ks_drum", SR, 0.8, seed=5))
    assert not np.array_equal(D.render_drum("ks_drum", SR, 0.8, seed=5), D.render_drum("ks_drum", SR, 0.8, seed=6))


# ------------------------------------------------------------------ smoke: every registered name
def test_smoke_everything_registered():
    assert {"compose", "arrangement", "euclid", "scatter", "arp", "stinger", "adaptive"} <= set(SP.LAYER_TYPES)
    assert "song" not in C.__dict__ and "automix_pan" not in __import__("genny.fx", fromlist=["REGISTRY"]).REGISTRY
    layers = [dict(BASE, role=r, bars=2, style="synthwave", inst=None if r == "drums" else "synth_pluck") for r in ROLES]
    layers += [{"type": "arrangement", "style": "chiptune", "bars": 2, "seed": 1},
               {"type": "arrangement", "style": "celtic_jig", "bars": 2, "seed": 1, "parts": {"chords": {"inst": "pluck"}, "lead": {"inst": "whistle"}, "counter": False}, "fold": True},
               {"type": "euclid", "kind": "ks_drum", "k": 3, "n": 8, "params": {"decay": 0.12}},
               {"type": "euclid", "inst": "ks_string", "note": "E2", "k": 5, "n": 8},
               {"type": "scatter", "layers": [{"type": "drum", "kind": "ks_drum"}, {"type": "sfx", "kind": "click"}], "rate": 3, "dur": 3, "pan": 0.6, "seed": 1},
               {"type": "arp", "inst": "ks_string", "chords": ["E3 G3 B3", "A3:min"], "mode": "updown"},
               {"type": "adaptive", "style": "ambient", "bars": 2, "states": {"calm": {}, "tense": {"scale": "phrygian"}},
                "transitions": [{"from": "calm", "to": "tense", "stinger": "mystery"}]}]
    layers += [{"type": "stinger", "mood": m, "seed": 1} for m in C.MOODS]
    for L in layers:
        L = {k: v for k, v in L.items() if v is not None}
        for sr, bpm in ((22050, 120), (48000, 96)):
            y, _ = SP.render_layer(L, sr, bpm)
            name = L.get("role") or L.get("mood") or L.get("style") or L["type"]
            assert np.all(np.isfinite(y)) and 1e-4 < np.max(np.abs(y)) < 1.5, (L["type"], name, float(np.max(np.abs(y))))
            # (no DC check here: a layer only places notes, and several existing instruments carry an offset)
            if not L.get("fold") and L["type"] != "adaptive":
                assert np.max(np.abs(y[0])) < 0.2 and np.max(np.abs(y[-1])) < 2e-3, (L["type"], name, np.max(np.abs(y[-1])))
    for sr in (22050, 44100, 48000):
        for kw in ({}, {"blend": 1.0, "decay": 0.4}, {"blend": 0.0, "stretch": 3.0}, {"decay": 0.03}):
            for vel in (1.0, 0.2):
                y = D.render_drum("ks_drum", sr, vel, **kw)
                assert y.ndim == 1 and np.all(np.isfinite(y)) and 1e-4 < np.max(np.abs(y)) < 1.5 and abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3
                assert abs(np.mean(y)) < 5e-3
        for kw in ({}, {"stretch": 6.0, "decay": 5.0}, {"loss": 0.9, "pick": 0.0, "dyn": 6000, "bright": 9000}, {"sympathetic": 0.5, "symp_interval": 7}):
            y = I.render_note("ks_string", 196.0, 0.4, sr, 0.9, **kw)
            assert np.all(np.isfinite(y)) and 1e-4 < np.max(np.abs(y)) < 1.5 and abs(y[-1]) < 1e-3
    import time
    I.render_note("ks_string", 82.4, 1.0, 48000)
    t = time.time()
    I.render_note("ks_string", 82.4, 1.0, 48000, sympathetic=0.3)
    D.render_drum("ks_drum", 48000, decay=1.0)
    assert time.time() - t < 2.0


if __name__ == "__main__":
    import sys
    for name, fn in list(globals().items()):
        if name.startswith("test_") and (len(sys.argv) < 2 or sys.argv[1] in name):
            fn()
            print("ok", name, flush=True)
