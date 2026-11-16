"""The shared sfx level mechanism: `out_level` in render_sfx, `level` on an sfx layer, `--level` on `genny sfx`."""
import inspect

import numpy as np

from genny import _sfx_levels as L
from genny import sfx as S
from genny import spec as SP
from genny.cli import build_parser

SR = 44100
SAMPLE = ("beep blip click pop coin powerup powerdown laser zap hit punch jump whoosh alarm siren error bubble static footstep "
          "door").split()
LEVELLED = ("beep", "click", "coin", "whoosh", "hit", "laser", "pop")


def test_default_unchanged_bit_for_bit():
    assert len(SAMPLE) == 20
    for name in SAMPLE:
        a = S.render_sfx(name, SR)
        b = S.REGISTRY[name]["fn"](SR)
        assert np.array_equal(a, b), name
        assert np.array_equal(a, S.render_sfx(name, SR, out_level=None)), name


def test_no_name_collision():
    for name, e in S.REGISTRY.items():
        assert "out_level" not in e["params"], name
        assert "out_level" not in inspect.signature(e["fn"]).parameters, name


def test_rms_level_reached():
    for name in LEVELLED:
        y = S.render_sfx(name, SR, out_level=-18)
        assert abs(S.rms_db(y, SR) + 18.0) < 0.3, name
        assert np.max(np.abs(y)) <= 1.0


def test_lufs_level_reached():
    from genny.analysis import loudness
    y = S.render_sfx("whoosh", SR, out_level="-23lufs")
    assert abs(loudness(y, SR, window=0.4) + 23.0) < 0.3


def test_match_table_and_level():
    assert set(S.REGISTRY) - set(L.GAIN_DB) == set(L.UNMEASURED)
    for name in LEVELLED:
        y = S.render_sfx(name, SR, out_level="match")
        want = L.LEVELS[name][1]
        assert np.max(np.abs(y)) <= 1.0
        assert S.rms_db(y, SR) < want + 0.3, (name, S.rms_db(y, SR), want)       # a limited transient may stop short of it
        if name != "hit":
            assert abs(S.rms_db(y, SR) - want) < 0.3, (name, S.rms_db(y, SR), want)
    assert abs(L.LEVELS["rain"][1] - L.LEVELS["hit"][1]) > 3.0          # families differ


def test_physical_default_untouched_by_table():
    for name in ("firearm", "minigun"):
        assert L.LEVELS[name][1] == -14.0


def test_stereo_keeps_balance():
    rng = np.random.default_rng(1)
    y = np.stack([rng.standard_normal(SR), 0.25 * rng.standard_normal(SR)], axis=1) * 0.1
    z = S.apply_level(y, SR, "x", -20)
    assert z.shape == y.shape
    assert abs(S.rms_db(z, SR) + 20.0) < 0.05
    ratio = np.sqrt(np.mean(z[:, 1] ** 2) / np.mean(z[:, 0] ** 2))
    assert abs(ratio - 0.25) < 0.01


def test_silence_no_nan():
    for z in (np.zeros(SR), np.zeros((SR, 2)), np.zeros(0)):
        out = S.apply_level(z, SR, "x", -18)
        assert np.array_equal(out, z) and np.all(np.isfinite(out))
    assert np.all(np.isfinite(S.apply_level(np.full(SR, 1e-9), SR, "x", "-23lufs")))


def test_bad_level_rejected():
    try:
        S.apply_level(np.ones(100) * 0.1, SR, "beep", "loud")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_spec_layer_level_replaces_peak_normalising():
    base, _ = SP.render_spec({"layers": [{"type": "sfx", "kind": "whoosh"}]})
    assert abs(20 * np.log10(np.max(np.abs(base))) + 1.0) < 0.05          # default: peak -1 dB as before
    y, _ = SP.render_spec({"layers": [{"type": "sfx", "kind": "whoosh", "level": -26}]})
    assert abs(S.rms_db(y, SR) + 26.0) < 0.3
    z, _ = SP.render_spec({"layers": [{"type": "sfx", "kind": "whoosh", "params": {"out_level": -26}}]})
    assert np.array_equal(y, z)
    m, _ = SP.render_spec({"layers": [{"type": "sfx", "kind": "whoosh", "level": "match"}]})
    assert abs(S.rms_db(m, SR) - L.LEVELS["whoosh"][1]) < 0.3


def test_cli_level_option():
    p = build_parser()
    assert p.parse_args(["sfx", "beep", "--level", "-18"]).level == -18.0
    assert p.parse_args(["sfx", "beep", "--level", "match"]).level == "match"
    assert p.parse_args(["sfx", "beep", "--level", "-23lufs"]).level == "-23lufs"
    assert p.parse_args(["sfx", "beep"]).level is None


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_") and callable(v):
            v()
            print("ok", k)
