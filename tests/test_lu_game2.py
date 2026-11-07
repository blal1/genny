import math

import numpy as np

from genny import chiptune as CT
from genny import chiptune2 as C


def test_tia_two_channels_and_five_bit_divider():
    assert CT.SYSTEMS["tia"] == ("ch1", "ch2")
    assert C.tia_audf(20.0, 4) == 31 and C.tia_audf(20000.0, 4) == 0
    assert math.isclose(C.tia_hz(20.0, 4), 3579545.0 / 114 / 64, rel_tol=1e-9)


def test_tia_lead_a4_is_flat_ntsc_and_flatter_pal():
    ntsc = 1200 * math.log2(C.tia_hz(440.0, 12, "ntsc") / 440.0)
    pal = 1200 * math.log2(C.tia_hz(440.0, 12, "pal") / 440.0)
    assert -17 < ntsc < -13
    assert -33 < pal < -26


def test_tia_waves_have_their_own_grids():
    assert C.tia_hz(440.0, C.TIA_WAVES["bass"]) != C.tia_hz(440.0, C.TIA_WAVES["lead"])
    assert math.isclose(C.tia_hz(440.0, 6), 3579545.0 / 114 / 31 / 2, rel_tol=1e-9)


def test_tia_hardware_pitch_on_rendered_signal():
    sr = 44100
    y = C.tia(440.0, 0.5, sr, wave="lead", mode="hardware_accurate")
    zc = np.where(np.diff(np.sign(y[2000:20000])) > 0)[0]
    assert abs(sr / np.median(np.diff(zc)) - C.tia_hz(440.0, 12)) < 3.0
    z = C.tia(440.0, 0.5, sr, wave="lead")
    zc = np.where(np.diff(np.sign(z[2000:20000])) > 0)[0]
    assert abs(sr / np.median(np.diff(zc)) - 440.0) < 3.0


def _amp(x, f, sr):
    n = x.shape[0]
    t = np.arange(n) / sr
    return 2 * abs(np.dot(x, np.exp(-2j * np.pi * f * t))) / n


def test_priority_duck_band_only_turns_down_the_band():
    from genny import spec as SP
    sr = 44100
    t = np.arange(sr) / sr
    low = 0.3 * np.sin(2 * np.pi * 100 * t) + 0.3 * np.sin(2 * np.pi * 2000 * t)
    high = 0.8 * np.sin(2 * np.pi * 1000 * t)
    for band, keep_100 in ((None, False), ([500, 4000], True)):
        lay = {"priority": 0} if band is None else {"priority": 0, "duck_band": band}
        out = SP._duck([(low, 0.0), (high, 0.0)], [lay, {"priority": 4}], sr)[0][0]
        seg = slice(sr // 2, sr)
        db100 = 20 * np.log10(_amp(out[seg], 100, sr) / _amp(low[seg], 100, sr))
        db2k = 20 * np.log10(_amp(out[seg], 2000, sr) / _amp(low[seg], 2000, sr))
        assert db2k < -9
        assert (abs(db100) < 1.5) if keep_100 else (db100 < -10)


def test_adaptive_first_pass_is_the_loop_without_its_folded_tail(tmp_path):
    import json
    from genny import spec as SP
    from genny.core import read_wav
    lay = {"type": "adaptive", "style": "ambient", "key": "C", "bars": 2, "seed": 1, "out_stems": str(tmp_path),
           "first": True, "tails": True, "states": {"calm": {"intensity": [0.5]}}}
    SP.render_layer(lay, 22050, None)
    g = json.loads((tmp_path / "graph.json").read_text())
    assert g["states"]["calm"]["first"] == "calm__first.wav"
    loop, _ = read_wav(tmp_path / "calm.wav")
    first, _ = read_wav(tmp_path / "calm__first.wav")
    tail, _ = read_wav(tmp_path / "calm__tail.wav")
    assert first.shape == loop.shape
    k = min(tail.shape[0], loop.shape[0])
    assert np.max(np.abs(loop[:k] - first[:k] - tail[:k])) < 2e-3
    assert np.max(np.abs(first[:k] - loop[:k])) > 1e-3


def test_tia_grids_match_the_published_tuning_sets_ntsc():
    table = {1: {"C2": 3, "D#2": -5, "B2": -3, "A#3": -2, "F4": 0, "C6": 2},
             12: {"E3": -11, "G3": -17, "A3": -14, "C4": 3, "D4": -16, "A4": -13, "D5": -16},
             6: {"C1": 0, "D#1": 4, "F1": 16, "F#2": -6},
             14: {"G#0": 0, "B0": -11}}
    from genny import notes as N
    for audc, notes in table.items():
        for name, cents in notes.items():
            hz = N.midi_to_freq(N.note_to_midi(name))
            got = 1200 * math.log2(C.tia_hz(hz, audc, "ntsc") / hz)
            assert abs(got - cents) <= 4, (audc, name, got, cents)


def test_variants_expand_with_ranges_and_new_seeds():
    from genny import spec as SP
    spec = {"out": "sfx/step.wav", "variants": 4, "vary": {"pitch": [0.8, 1.1], "gain": [0.5, 0.7]},
            "layers": [{"type": "sfx", "kind": "footstep", "params": {"seed": 3}}]}
    vs = SP.load_specs(spec)
    assert [v["out"].replace("\\", "/") for v in vs] == [f"sfx/step_0{i}.wav" for i in range(1, 5)]
    assert [v["layers"][0]["params"]["seed"] for v in vs] == [3, 4, 5, 6]
    for v in vs:
        assert 0.5 <= v["gain"] <= 0.7
        assert 0.8 <= v["fx"][-1]["factor"] <= 1.1 and v["fx"][-1]["type"] == "speed"
        assert "variants" not in v and "vary" not in v
    assert vs == SP.load_specs(spec)
    assert SP.load_specs({"out": "a.wav", "layers": []}) == [{"out": "a.wav", "layers": []}]


def test_sid_fc_max_moves_the_cutoff():
    sr = 44100

    def centroid(y):
        s = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(y.shape[0], 1 / sr)
        return float((s * f).sum() / s.sum())
    a = C.sid(220.0, 0.5, sr, wave="saw", cutoff=400, res=0, mode="hardware_accurate")
    b = C.sid(220.0, 0.5, sr, wave="saw", cutoff=400, res=0, mode="hardware_accurate", fc_max=6250.0)
    assert np.allclose(a, C.sid(220.0, 0.5, sr, wave="saw", cutoff=400, res=0, mode="hardware_accurate", fc_max=12500.0))
    assert centroid(b) < 0.85 * centroid(a)
