"""Tables d'armes dynamiques et cartouche utilisateur de bout en bout (hors pytest) : uv run python tests/test_y2b.py"""
import os
import sys
import tempfile
import traceback
from collections.abc import Mapping
from pathlib import Path

import numpy as np

os.environ["GENNY_PRESETS_USER"] = ""
os.environ.pop("GENNY_PRESETS", None)

from genny import arms, foley, presets  # noqa: E402

TOML = Path(presets.__file__).with_name("presets")
USER = '''schema = "genny.preset/1"
family = "{fam}"

{body}
'''
FILES = {
    "cartridge": '[ma_balle]\nextends = "556"\nd_mm = 6.5\nl_mm = 30.0\nv_ref = 1000.0\nm_g = 9.0\n',
    "case": '[mon_etui]\nL = 40.0\ndb = 11.0\ndn = 7.0\nh = 0.4\ncap = 3.0\nmass = 9.0\nmat = "brass"\nneck = 8.0\n',
    "case_dim": '[mon_etui]\nL = 40.0\ndia = 11.0\nwall = 0.4\n',
    "case_of": '[ma_balle]\ncase = "mon_etui"\n',
    "weapon": '[carabine_x]\nextends = "ar15"\ncart = "ma_balle"\nbarrel = 0.55\n',
    "gun_legacy": '[mon_fusil]\nchirp = 38.97\nbarrel = 6.1\nburst = 7.677\nv = 1000.0\nbscale = 1.0\nbgain = 1.0\nblop = 0.0\nrate = 8.0\nproj = "ma_balle"\n',
}


class User:
    def __enter__(self):
        self.d = tempfile.TemporaryDirectory()
        for fam, body in FILES.items():
            p = Path(self.d.name) / f"{fam}.toml"
            p.write_text(USER.format(fam=fam, body=body), encoding="utf-8")
            presets.register(p)
        return self

    def __exit__(self, *a):
        presets.reload(clear=True)
        self.d.cleanup()


def test_tables_are_live_views():
    for t in (arms.CALIBERS, arms.CHARGE, arms.WEAPONS, arms.BARREL_MM):
        assert isinstance(t, Mapping) and len(t) == len(list(t)) and ("556" in t or "ar15" in t)
    assert len(arms.CALIBERS) == 20 and len(arms.CHARGE) == 20 and len(arms.WEAPONS) == 33 and len(arms.BARREL_MM) == 33
    assert len(arms.CASES) == 8 and len(arms._CASE_EXTRA) == 3 and len(arms._CASE_OF) == 19 and len(foley.GUNS) == 9
    assert arms.CALIBERS.get("nope") is None and "nope" not in arms.CALIBERS
    try:
        arms.CHARGE["nope"]
    except KeyError:
        pass
    else:
        raise AssertionError
    assert arms.REF_PA > 0 and arms._PM_REF > 0 and arms._EB_REF > 0
    assert len(arms.CALIBERS["556"]) == 5 and abs(arms.BARREL_MM["ar15"] - 508.0) < 1e-9


def test_user_cartridge_in_tables():
    with User():
        assert "ma_balle" in arms.CALIBERS and "ma_balle" in arms.CHARGE and "carabine_x" in arms.WEAPONS
        d, l, v, peak, _ = arms.CALIBERS["ma_balle"]
        assert (d, l, v) == (6.5, 30.0, 1000.0) and peak > 0
        e, pm = arms.CHARGE["ma_balle"]
        assert e > 0 and pm == presets.get("cartridge", "556")["p_max_mpa"]
        assert abs(arms.BARREL_MM["carabine_x"] - 550.0) < 1e-9
        assert arms.WEAPONS["carabine_x"][0] == "ma_balle"
        assert "mon_etui" in arms.CASES and arms._CASE_OF["ma_balle"] == "mon_etui" and "mon_fusil" in foley.GUNS
    assert "ma_balle" not in arms.CALIBERS and "ma_balle" not in arms.CHARGE and "carabine_x" not in arms.WEAPONS
    assert "mon_etui" not in arms.CASES and "mon_fusil" not in foley.GUNS


def test_user_cartridge_end_to_end():
    ref_pass, ref_imp = arms.bullet_pass(caliber="556"), arms.bullet_impact(caliber="556")
    with User():
        yp = arms.bullet_pass(caliber="ma_balle")
        assert yp.size and np.all(np.isfinite(yp)) and not np.array_equal(yp, ref_pass)
        assert np.array_equal(yp, arms.bullet_pass(caliber="ma_balle"))
        for m in ("metal", "wood", "flesh", "glass"):
            yi = arms.bullet_impact(material=m, caliber="ma_balle")
            assert yi.size and np.all(np.isfinite(yi)) and not np.array_equal(yi, arms.bullet_impact(material=m, caliber="556"))
        yf = arms.firearm(weapon="carabine_x", brass=1.0)
        assert yf.size and np.all(np.isfinite(yf)) and not np.array_equal(yf, arms.firearm(weapon="ar15", brass=1.0))
        assert arms.drag_length("ma_balle") > 0
        c1, c2 = foley.casing(caliber="mon_etui"), foley.casing(caliber="mon_etui", ricochet=True)
        for c in (c1, c2, arms.casing_ricochet(caliber="mon_etui", floor="wood")):
            assert c.size and np.all(np.isfinite(c))
        g = foley.gunshot(kind="mon_fusil")
        assert g.size and foley.crack_ms("mon_fusil") > 0
    assert np.array_equal(arms.bullet_pass(caliber="ma_balle"), ref_pass)
    assert np.array_equal(arms.bullet_impact(caliber="ma_balle"), ref_imp)


def test_override_builtin_cartridge_changes_refs():
    before = arms._refs()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.toml"
        p.write_text(USER.format(fam="cartridge", body='[556]\nextends = "762x51"\n'), encoding="utf-8")
        presets.register(p)
        try:
            assert arms._refs() != before and arms.REF_PA == arms._refs()[0]
        finally:
            presets.reload(clear=True)
    assert arms._refs() == before


def test_files_and_identity_with_frozen():
    for f, n in (("case_dim", 8), ("case_extra", 3), ("case_of", 19), ("gun_legacy", 9)):
        assert (TOML / f"{f}.toml").is_file() and len(presets.list(f)) == n
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "examples"))
    import preset_freeze as pf
    for ref, fams in (("preset_ref", ("cartridge", "case", "weapon")), ("preset_ref_y2b", ("case_dim", "case_extra", "case_of", "gun_legacy"))):
        for f in fams:
            if (root / "sounds" / ref / f"{f}_table.json").exists():
                assert pf.check(f, root / "sounds" / ref) == 0, f


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
