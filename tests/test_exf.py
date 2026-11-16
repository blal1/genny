"""car_cabin et alias de cartouches pour casing (hors pytest) : uv run python tests/test_exf.py"""
import os
import tempfile
from pathlib import Path

import numpy as np

os.environ["GENNY_PRESETS_USER"] = ""
os.environ.pop("GENNY_PRESETS", None)

from genny import arms, foley, presets  # noqa: E402
from genny.fx import REGISTRY  # noqa: E402


def test_car_cabin_runs_mono_and_stereo():
    x = np.random.default_rng(0).uniform(-0.5, 0.5, 4410)
    fn = REGISTRY["car_cabin"]["fn"]
    y = fn(x, 44100)
    assert y.shape == x.shape and np.isfinite(y).all()
    s = fn(np.stack([x, x], axis=1), 44100)
    assert s.shape == (4410, 2) and np.array_equal(s[:, 0], y)


def test_cartridge_names_resolve_to_case_geometry():
    for alias, name in (("9x19", "9mm"), ("357", "9mm"), ("22lr", "9mm"), ("762x51", "762"), ("762x39", "762"), ("308", "762"), ("12_buck", "12ga")):
        a = foley.casing(count=3, caliber=alias)
        b = foley.casing(count=3, caliber=name)
        assert np.array_equal(a, b) and np.abs(a).max() > 0, alias
    assert len(foley.casing(count=1, caliber="357", ricochet=True)) > 0


def test_accepted_names_unchanged():
    for c in arms.CASES:
        assert arms._case_name(c) == c
    try:
        foley.casing(caliber="musket")
    except ValueError:
        pass
    else:
        raise AssertionError


def test_user_cartridge():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "u.toml"
        p.write_text('schema = "genny.preset/1"\nfamily = "case_of"\n\n[ma_balle]\ncase = "9x19"\n', encoding="utf-8")
        presets.register(p)
        try:
            assert np.array_equal(foley.casing(count=2, caliber="ma_balle"), foley.casing(count=2, caliber="9mm"))
        finally:
            presets.reload(clear=True)


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
