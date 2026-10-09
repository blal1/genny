"""Tempo, pattern, loop and loudness-ceiling behaviour of the spec renderer.

Run:  uv run python tests/test_spec.py     (plain asserts, no test framework needed)
"""
import numpy as np
from genny import spec as SP

sr = 44100
# 1. seconds vs beats give identical audio
a, _ = SP.render_spec({"layers": [{"type": "seq", "inst": "piano", "steps": "C4:0.5 E4:0.25 -:0.25 [G4,C5]:1.0", "at": 0.5}], "normalize": None, "trim": False})
b, _ = SP.render_spec({"bpm": 120, "layers": [{"type": "seq", "inst": "piano", "steps": "C4:1 | E4:.5 -:.5 | [G4,C5]:2", "at": 1}], "normalize": None, "trim": False})
assert a.shape == b.shape and np.allclose(a, b), (a.shape, b.shape)
# 2. old pattern semantics unchanged (seconds, x and X... x only), default step 0.125 s
old, _ = SP.render_spec({"layers": [{"type": "pattern", "kind": "hihat", "steps": "x-x-xxx-", "step": 0.125}], "normalize": None, "trim": False})
hits, _ = SP.render_spec({"layers": [{"type": "pattern", "kind": "hihat", "hits": [0, 0.25, 0.5, 0.625, 0.75]}], "normalize": None, "trim": False})
assert np.allclose(old, hits)
dflt, _ = SP.render_spec({"layers": [{"type": "pattern", "kind": "hihat", "steps": "x-x-xxx-"}], "normalize": None, "trim": False})
assert np.allclose(old, dflt)
# 3. with bpm the pattern step defaults to a 16th; bars repeats; accents/ghosts differ in level
p, _ = SP.render_spec({"bpm": 120, "layers": [{"type": "pattern", "kind": "kick", "steps": "X---o---|x---x---", "bars": 2}], "normalize": None, "trim": False})
on = [int(i * 0.125 * sr) for i in (0, 4, 8, 12, 16, 20, 24, 28)]
pk = [float(np.max(np.abs(p[s:s + 2000]))) for s in on]
assert pk[0] > pk[2] > pk[1] and abs(pk[0] - pk[4]) < 1e-6, pk
# 4. swing delays the off steps
s0, _ = SP.render_spec({"bpm": 120, "layers": [{"type": "pattern", "kind": "rim", "steps": "xx", "swing": 0.5}], "normalize": None, "trim": False})
second = np.argmax(np.abs(s0[3000:]) > 0.05) + 3000
assert abs(second - int(1.5 * 0.125 * sr)) < 30, second
# 5. hits, dur, every, group in beats
g, _ = SP.render_spec({"bpm": 60, "layers": [{"type": "group", "layers": [{"type": "pattern", "kind": "rim", "hits": [0, 1]}, {"type": "synth", "inst": "sine", "notes": "A4", "at": 2}]}],
                       "normalize": None, "trim": False})
assert abs(g.shape[0] / sr - (2 + 1 + 0.05)) < 0.02, g.shape[0] / sr     # sine: 1 beat (default dur) + 50 ms release
# 6. beats + fold: exact length, tail wrapped onto the head, and looping it has no jump at the seam
spec = {"bpm": 120, "beats": 8, "loop": "fold",
        "layers": [{"type": "seq", "inst": "pad", "steps": "[C4,E4,G4]:4 [A3,C4,E4]:4"}, {"type": "pattern", "kind": "kick", "steps": "x---", "bars": 8}],
        "fx": [{"type": "reverb", "mix": 0.3, "size": 0.8}]}
f, _ = SP.render_spec(spec)
assert f.shape[0] == 4 * sr, f.shape
dry, _ = SP.render_spec({**spec, "loop": None, "beats": None, "normalize": None, "max_loudness": None, "trim": False})
assert dry.shape[0] > 4 * sr                                                     # there was a tail to fold
mono = f.mean(axis=1) if f.ndim == 2 else f
jump = abs(mono[0] - mono[-1])
assert jump < 4 * np.max(np.abs(np.diff(mono))), jump
# 7. beats without fold: padded/cut to length (+ tail seconds)
t, _ = SP.render_spec({"bpm": 120, "beats": 4, "tail": 0.5, "trim": False, "layers": [{"type": "seq", "inst": "sine", "steps": "C4:1"}]})
assert t.shape[0] == int(2.5 * sr), t.shape
try:
    SP.render_spec({"loop": "fold", "layers": [{"type": "sfx", "kind": "beep"}]})
    raise AssertionError("fold without a length should fail")
except SP.SpecError:
    pass
# 8. ceiling: a pure 3 kHz tone is turned down, a kick is not; null disables
tone = {"layers": [{"type": "sfx", "kind": "tone", "params": {"freq": 3000, "dur": 1.0}}]}
y1, _ = SP.render_spec(tone)
y2, _ = SP.render_spec({**tone, "max_loudness": None})
k, _ = SP.render_spec({"layers": [{"type": "drum", "kind": "kick"}]})
assert np.max(np.abs(y1)) < 0.5 and abs(np.max(np.abs(y2)) - 10 ** (-1 / 20)) < 1e-3 and abs(np.max(np.abs(k)) - 10 ** (-1 / 20)) < 1e-3
print("all ok")
