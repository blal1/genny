"""Armes / balistique en presets TOML (hors pytest) : uv run python tests/test_y2.py"""
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

os.environ["GENNY_PRESETS_USER"] = ""
os.environ.pop("GENNY_PRESETS", None)

from genny import arms, ballistics as B, brass, presets  # noqa: E402

TOML = Path(presets.__file__).with_name("presets")


def test_builtin_files_and_counts():
    for f, n in (("cartridge", 20), ("case", 5), ("weapon", 33)):
        assert (TOML / f"{f}.toml").is_file()
        assert len(presets.list(f)) == n and all(presets.origin(f, k) == "builtin" for k in presets.list(f))
    assert len(arms.PRESETS) == 33 and len(B.CARTRIDGES) == 20 and len(brass.CASE_GEOM) == 5
    assert isinstance(B.CARTRIDGES["556"], B.Cartridge) and arms.PRESETS["ar15"]["cart"] == "556"


def test_preset_by_name_matches_view():
    assert presets.get("weapon", "ar15") == arms.PRESETS["ar15"]
    assert list(presets.get("weapon", "flintlock")["lock"]) == [35.0, "flint"]


def test_inline_dict_equals_barrel_override():
    a = arms.firearm(weapon={"extends": "ar15", "barrel": 0.4})
    b = arms.firearm(weapon="ar15", barrel=0.4)
    assert np.array_equal(a, b)
    a = arms.firearm(weapon={"extends": "pistol_9mm", "barrel": 0.2}, pov=True)
    b = arms.firearm(weapon="pistol_9mm", barrel=0.2, pov=True)
    assert np.array_equal(a, b)


def test_unknown_weapon_and_bad_field():
    for bad in ("nope", {"extends": "ar15", "barrel": 4.0}, {"extends": "ar15", "cart": "nope"}, {"extends": "ar15", "barel": 0.4}):
        try:
            arms.firearm(weapon=bad)
        except ValueError as e:
            assert str(e)
        else:
            raise AssertionError(bad)


def test_user_toml_weapon_and_cartridge():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "w.toml").write_text('schema = "genny.preset/1"\nfamily = "weapon"\n\n[ma_carabine]\nextends = "ar15"\nbarrel = 0.3\nbody = 1.2\n', encoding="utf-8")
        presets.register(d / "w.toml")
        try:
            assert "ma_carabine" in arms.PRESETS
            y = arms.firearm(weapon="ma_carabine")
            assert np.array_equal(y, arms.firearm(weapon={"extends": "ar15", "barrel": 0.3, "body": 1.2}))
            assert not np.array_equal(y, arms.firearm(weapon="ar15"))
        finally:
            presets.reload(clear=True)
        assert "ma_carabine" not in arms.PRESETS
        (d / "c.toml").write_text('schema = "genny.preset/1"\nfamily = "cartridge"\n\n[ma_balle]\nextends = "556"\nv_ref = 900.0\n', encoding="utf-8")
        (d / "w2.toml").write_text('schema = "genny.preset/1"\nfamily = "weapon"\n\n[carabine_x]\nextends = "ar15"\ncart = "ma_balle"\n', encoding="utf-8")
        presets.register(d / "c.toml")
        presets.register(d / "w2.toml")
        try:
            assert abs(B.muzzle_velocity("ma_balle", 0.508) - 900.0) < 1e-9
            assert arms.firearm(weapon="carabine_x").size > 0
        finally:
            presets.reload(clear=True)
        assert "ma_balle" not in B.CARTRIDGES


def test_env_var_weapon_renders():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "w.toml").write_text('schema = "genny.preset/1"\nfamily = "weapon"\n\n[env_gun]\nextends = "revolver_38"\nbody = 1.5\n', encoding="utf-8")
        os.environ["GENNY_PRESETS"] = d
        try:
            presets.reload()
            assert arms.firearm(weapon="env_gun").size > 0
        finally:
            os.environ.pop("GENNY_PRESETS")
            presets.reload(clear=True)


def test_bit_identity_when_frozen_files_exist():
    ref = Path(__file__).resolve().parent.parent / "sounds" / "preset_ref"
    if not (ref / "weapon_table.json").exists():
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
    import preset_freeze as pf
    for f in ("cartridge", "case", "weapon"):
        assert pf.check(f, ref) == 0, f


if __name__ == "__main__":
    bad = 0
    for k, v in sorted(globals().items()):
        if k.startswith("test_") and callable(v):
            try:
                v()
                print("ok  ", k)
            except Exception:
                bad += 1
                print("FAIL", k)
                traceback.print_exc()
    sys.exit(1 if bad else 0)
