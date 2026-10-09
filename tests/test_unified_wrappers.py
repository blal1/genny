"""Regression tests for the v0.5 high-level physical-engine compatibility wrappers."""
import numpy as np
from genny import instruments as I
from genny import sfx as S


def _finite_mono(x):
    x = np.asarray(x)
    assert x.ndim == 1
    assert x.size > 8
    assert np.all(np.isfinite(x))
    assert np.max(np.abs(x)) > 1e-6
    assert abs(float(x[-1])) < 2e-3


def test_physical_instrument_wrappers():
    cases = {
        "piano": (261.63, {"decay": 1.2, "bright": 0.8}),
        "violin": (440.0, {"attack": 0.04, "vibrato": 0.8}),
        "cello": (220.0, {"attack": 0.07, "vibrato": 0.8}),
        "clarinet": (261.63, {}),
        "flute": (440.0, {"breath": 0.12}),
        "trumpet": (329.63, {"mute": 0.2}),
        "french_horn": (220.0, {}),
        "trombone": (164.81, {}),
        "tuba": (82.41, {}),
    }
    for sr in (22050, 44100, 48000):
        for name, (f, params) in cases.items():
            _finite_mono(I.render_note(name, f, 0.10, sr, 0.75, **params))


def test_physical_sfx_wrappers():
    for sr in (22050, 44100, 48000):
        _finite_mono(S.render_sfx("footstep", sr, ground="wood", variation=1))
        _finite_mono(S.render_sfx("wind", sr, dur=0.12, scale=1.0, seed=2))
        _finite_mono(S.render_sfx("thunder", sr, dur=0.20, distance=80, seed=2))
        _finite_mono(S.render_sfx("hit", sr, dur=0.12, tone=180, crunch=0.7, material="steel"))
