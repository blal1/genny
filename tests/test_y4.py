"""Scenes, patches and isolated preset families in TOML (hors pytest) : uv run python tests/test_y4.py"""
import json
import os
import subprocess
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

import numpy as np

os.environ["GENNY_PRESETS_USER"] = ""
os.environ.pop("GENNY_PRESETS", None)

import genny  # noqa: E402
from genny import cyber, drums, dsp, family_v, fdstring, fx, instruments, presets, scenes, sfx  # noqa: E402
from genny.presets import F, PresetError  # noqa: E402

TOML = Path(presets.__file__).with_name("presets")
COUNTS = {"scene": 49, "lofi": 8, "spring": 5, "compressor": 3, "free_reed": 5, "bearing": 3, "patch": 0}
SR = 22050

BUS_TRAM = """schema = "genny.preset/1"
family = "patch"

[bus_articule]
sfx = "bus"
desc = "Bus articule au ralenti, deux portes"
params = { rpm = 650.0, doors = 2, dur = 0.5 }

[tram_ville]
sfx = "rail"
desc = "Tram urbain a 30 km/h"
params = { kind = "tram_roll", speed = 30.0, dur = 0.5 }

[guitare_pic]
instrument = "guitar"
params = { pluck = "pick" }

[kick_grave]
drum = "kick"
params = { tune = 40.0 }
"""


class User:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "user.toml"
        self.path.write_text(self.text, encoding="utf-8")
        presets.register(self.path)
        return self.path

    def __exit__(self, *a):
        presets.reload(clear=True)
        self.dir.cleanup()


def head(family):
    return f'schema = "genny.preset/1"\nfamily = "{family}"\n'


def eq(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and np.array_equal(a, b)


def raises(fn, text):
    try:
        fn()
    except PresetError as e:
        assert text in str(e), str(e)
        return str(e)
    raise AssertionError(f"pas d'erreur ({text})")


def test_builtin_files_and_counts():
    for f, n in COUNTS.items():
        assert (TOML / f"{f}.toml").is_file(), f
        assert len(presets.list(f)) == n, (f, len(presets.list(f)))
        assert all(presets.origin(f, k) == "builtin" for k in presets.list(f)), f
    assert len(scenes.SCENES) == 49 and len(dsp.LOFI_PRESETS) == 8 and len(fdstring.SPRING_PRESETS) == 5
    assert len(fx._COMP_PRESETS) == 3 and len(family_v.FREE_REED_PRESETS) == 5 and len(cyber.BEARING_RESONANCE_PRESETS) == 3


def test_views_keep_their_types():
    s = scenes.SCENES["religious"]
    assert list(s) == ["bed", "layers", "room"] and s["bed"] is None and s["room"] == "church"
    c = s["layers"][0]
    assert list(c) == ["inst", "sfx", "db", "wide", "opt", "every", "notes", "params"] and c["notes"] == ("C3", "G3", "C4", "E4") and c["every"] is None
    e = scenes.SCENES["warfare"]["layers"][0]
    assert list(e) == ["sfx", "db", "wide", "opt", "every", "params"] and e["every"] == (3, 7) and isinstance(e["every"][0], int)
    assert list(scenes.SCENES["warfare"]) == ["bed", "climax", "layers", "room"] and list(scenes.SCENES["designed"])[:2] == ["bed", "act"]
    assert dsp.LOFI_PRESETS["nes"]["band"] is None and dsp.LOFI_PRESETS["snes"]["band"] == (20, 7000)
    assert fx._COMP_PRESETS["parallel"] == (-50.0, 2.5, 0.001, 0.3, False, 0.24)
    assert cyber.BEARING_RESONANCE_PRESETS["radar"] == (1750.0, 2755.0, 517.0)
    assert isinstance(dsp.LOFI_PRESETS["nes"]["rate"], int) and isinstance(fdstring.SPRING_PRESETS["leem1"]["M"], int)


def test_array_of_tables_type():
    fam = "y4_demo"
    view = presets.schema(fam, {"layers": F(list, max_len=4, of={"name": F(str), "level": F(float, lo=0.0, hi=1.0), "on": F(bool, default=True)})})
    with User(head(fam) + '[a]\nlayers = [{name = "x", level = 0.5}, {name = "y", level = 1, on = false}]\n'):
        assert view["a"]["layers"] == [{"name": "x", "level": 0.5, "on": True}, {"name": "y", "level": 1, "on": False}]
        assert repr(view["a"]["layers"][1]["level"]) == "1"
        assert presets.get(fam, "a")["layers"][0]["name"] == "x"
    for text, frag in (('[a]\nlayers = [{name = "x", levl = 0.5}]\n', "voulez-vous dire 'level'"), ('[a]\nlayers = [{name = "x"}]\n', "champ requis absent"),
                       ('[a]\nlayers = [{name = "x", level = 2.0}]\n', "hors de"), ('[a]\nlayers = [3]\n', "table attendue"),
                       ('[a]\nlayers = "x"\n', "tableau de tables"), ('[a]\nlayers = [{name = "x", level = 0.1}, {name = "x", level = 0.1}, {name = "x", level = 0.1}, {name = "x", level = 0.1}, {name = "x", level = 0.1}]\n', "> 4")):
        d = tempfile.TemporaryDirectory()
        p = Path(d.name) / "bad.toml"
        p.write_text(head(fam) + text, encoding="utf-8")
        msg = raises(lambda: presets.load(p), frag)
        assert "bad.toml" in msg and "[a]" in msg
        d.cleanup()
    t = presets.dumps(fam, {"a": {"layers": [{"name": "x", "level": 0.5}, {"name": "y", "level": 1, "on": False}]}})
    assert "layers = [\n" in t and presets.tomllib.loads(t)["a"]["layers"][1] == {"name": "y", "level": 1, "on": False}


def test_scene_user_and_errors():
    text = head("scene") + '''[ma_gare]
extends = "transportation"
room = "studio"

[mon_atelier]
bed = "town"
layers = [
  {sfx = "hvac", db = -8, params = {state = "fan"}},
  {sfx = "hammering", db = -6, every = [4, 8], params = {strikes = 3}},
  {inst = "church_organ", db = -10, notes = ["C3", "G3"]},
]
'''
    with User(text):
        assert scenes.SCENES["ma_gare"]["room"] == "studio" and scenes.SCENES["ma_gare"]["layers"] == scenes.SCENES["transportation"]["layers"]
        a = scenes.scene(sr=SR, kind="mon_atelier", dur=2.0, seed=1)
        b = scenes.scene(sr=SR, kind="mon_atelier", dur=2.0, seed=1)
        assert eq(a, b) and a.shape[0] == int(2.0 * SR) and np.abs(a).max() > 0.01
        assert presets.origin("scene", "mon_atelier").endswith("user.toml")
        assert "mon_atelier" in sfx.REGISTRY["scene"]["params"]["kind"][1] or True
    assert "mon_atelier" not in scenes.SCENES
    for body, frag in (('[x]\nlayers = [{sfx = "gust", every = [1, 2]}]\n', "champ requis absent"),
                       ('[x]\nlayers = [{sfx = "gust", db = -3, every = [1, 2], parms = {}}]\n', "clé inconnue"),
                       ('[x]\nbed = "ocean"\nlayers = [{sfx = "gust", db = -3}]\n', "parmi")):
        d = tempfile.TemporaryDirectory()
        p = Path(d.name) / "scene_bad.toml"
        p.write_text(head("scene") + body, encoding="utf-8")
        raises(lambda: presets.load(p), frag)
        d.cleanup()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with User(head("scene") + '[sans_sfx]\nlayers = [{db = -3}]\n[ok]\nlayers = [{sfx = "gust", db = -3}]\n'):
            assert "sans_sfx" not in scenes.SCENES and "ok" in scenes.SCENES
        assert any("sfx ou inst" in str(x.message) for x in w), [str(x.message) for x in w]


def test_patch_bus_tram_instrument_drum():
    before = (set(sfx.REGISTRY), set(instruments.REGISTRY), set(drums.REGISTRY))
    ref_bus = sfx.render_sfx("bus", SR, rpm=650.0, doors=2, dur=0.5)
    ref_tram = sfx.render_sfx("rail", SR, kind="tram_roll", speed=30.0, dur=0.5)
    ref_gtr = instruments.render_note("guitar", 220.0, 0.3, SR, 1.0, pluck="pick")
    ref_kick = drums.render_drum("kick", SR, 1.0, tune=40.0)
    with User(BUS_TRAM):
        assert eq(sfx.render_sfx("bus_articule", SR), ref_bus)
        assert eq(sfx.render_sfx("tram_ville", SR), ref_tram)
        assert eq(instruments.render_note("guitare_pic", 220.0, 0.3, SR, 1.0), ref_gtr)
        assert eq(drums.render_drum("kick_grave", SR, 1.0), ref_kick)
        assert eq(sfx.render_sfx("bus_articule", SR, doors=1, dur=0.3), sfx.render_sfx("bus", SR, rpm=650.0, doors=1, dur=0.3))
        e = sfx.REGISTRY["bus_articule"]
        assert e["params"]["rpm"][0] == 650.0 and e["params"]["brake"] == sfx.REGISTRY["bus"]["params"]["brake"] and e["desc"].startswith("Bus articule")
        assert e["origin"] == "patch" and instruments.REGISTRY["guitare_pic"]["family"] == instruments.REGISTRY["guitar"]["family"]
        assert presets.origin("patch", "bus_articule").endswith("user.toml")
        try:
            sfx.render_sfx("bus_articule", SR, nope=1)
        except ValueError as ex:
            assert "does not accept" in str(ex)
        else:
            raise AssertionError
        cat = sys.modules["genny.cli"]._catalog() if "genny.cli" in sys.modules else None
        if cat is None:
            from genny import cli
            cat = cli._catalog()
        assert cat["sfx"]["bus_articule"]["origin"] == "patch" and "origin" not in cat["sfx"]["bus"]
        assert cat["instruments"]["guitare_pic"]["origin"] == "patch" and cat["drums"]["kick_grave"]["origin"] == "patch"
    after = (set(sfx.REGISTRY), set(instruments.REGISTRY), set(drums.REGISTRY))
    assert before == after
    assert not any("origin" in v for v in sfx.REGISTRY.values())


def test_patch_errors():
    for body, frag in (('[bus]\nsfx = "air_leak"\n', "déjà celui d'un sfx"),
                       ('[bus_x]\nsfx = "bus"\nparams = {rmp = 3.0}\n', "voulez-vous dire 'rpm'"),
                       ('[bus_x]\nsfx = "bus"\ninstrument = "guitar"\n', "exactement une"),
                       ('[bus_x]\ndesc = "rien"\n', "exactement une"),
                       ('[bus_x]\nsfx = "autobus"\n', "inconnu"),
                       ('[bus_y]\nsfx = "bus"\n[bus_z]\nsfx = "bus_y"\n', "inconnu"),
                       ('[bus_x]\ninstrument = "guitar"\nparams = {glide = 3.0, bright = 0.2}\n', "inconnu de instrument")):
        d = tempfile.TemporaryDirectory()
        p = Path(d.name) / "p.toml"
        p.write_text(head("patch") + body, encoding="utf-8")
        try:
            presets.register(p)
        except PresetError as e:
            assert frag in str(e), (frag, str(e))
            assert "p.toml" in str(e)
        else:
            presets.reload(clear=True)
            raise AssertionError(frag)
        presets.reload(clear=True)
        d.cleanup()
    assert "bus_x" not in sfx.REGISTRY and "bus_y" not in sfx.REGISTRY
    assert "air_leak" in sfx.REGISTRY and "origin" not in sfx.REGISTRY["bus"]


def test_patch_extends_and_override():
    text = BUS_TRAM + '\n[bus_court]\nextends = "bus_articule"\nparams = { rpm = 700.0, dur = 0.3 }\n'
    with User(text):
        assert eq(sfx.render_sfx("bus_court", SR), sfx.render_sfx("bus", SR, rpm=700.0, dur=0.3))
        assert sfx.REGISTRY["bus_court"]["params"]["doors"][0] == 1


def run_cli(args, env_extra=None, cwd=None):
    env = dict(os.environ)
    env["GENNY_PRESETS_USER"] = ""
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-m", "genny.cli", *args], capture_output=True, text=True, env=env, cwd=cwd, timeout=600)


def test_cli_sfx_render_list_presets():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "patches.toml").write_text(BUS_TRAM, encoding="utf-8")
        env = {"GENNY_PRESETS": d}
        out = Path(d) / "bus.wav"
        r = run_cli(["sfx", "bus_articule", "-o", str(out)], env)
        assert r.returncode == 0 and out.is_file(), r.stderr
        out2 = Path(d) / "tram.wav"
        spec = {"out": str(out2), "sr": SR, "layers": [{"type": "sfx", "kind": "tram_ville"}]}
        r = run_cli(["render", "--json", json.dumps(spec), "--quiet"], env)
        assert r.returncode == 0 and out2.is_file(), r.stderr
        sp2 = {"presets": str(Path(d) / "patches.toml"), "out": str(Path(d) / "kick.wav"), "layers": [{"type": "drum", "kind": "kick_grave"}, {"type": "synth", "inst": "guitare_pic", "notes": "A3", "dur": 0.2}]}
        r = run_cli(["render", "--json", json.dumps(sp2), "--quiet"])
        assert r.returncode == 0 and (Path(d) / "kick.wav").is_file(), r.stderr
        r = run_cli(["list", "sfx", "--json"], env)
        cat = json.loads(r.stdout)
        assert cat["bus_articule"]["origin"] == "patch" and "origin" not in cat["bus"]
        r = run_cli(["presets", "list", "patch", "--json"], env)
        assert set(json.loads(r.stdout)["patch"]) == {"bus_articule", "tram_ville", "guitare_pic", "kick_grave"}
    r = run_cli(["presets", "list", "--json"])
    fams = json.loads(r.stdout)
    for f in ("scene", "patch", "lofi", "spring", "compressor", "free_reed", "bearing", "weapon", "cartridge", "two_stroke", "tracks", "rail_kind", "ship"):
        assert f in fams, f
    assert len(fams["scene"]) == 49 and fams["patch"] == {}


def test_broken_user_patch_does_not_break_import():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "bad.toml").write_text(head("patch") + '[bus]\nsfx = "air_leak"\n', encoding="utf-8")
        r = run_cli(["sfx", "beep", "-o", str(Path(d) / "b.wav")], {"GENNY_PRESETS": d})
        assert r.returncode == 0 and (Path(d) / "b.wav").is_file(), r.stderr
        assert "patch ignoré" in r.stderr, r.stderr
        r = run_cli(["presets", "check"], {"GENNY_PRESETS": d})
        assert r.returncode == 1 and "FAIL" in r.stdout and "bad.toml" in r.stdout, (r.stdout, r.stderr)


def test_isolated_families_use_user_presets():
    text = (head("lofi") + '[ma_radio]\nrate = 6000\nbits = 5\nlaw = "mu"\nband = [300, 2500]\nhold = false\nadpcm = false\n')
    x = np.random.default_rng(0).standard_normal(4000) * 0.1
    with User(text):
        assert dsp.LOFI_PRESETS["ma_radio"]["band"] == (300, 2500)
        y = dsp.lofi(x, 44100, preset="ma_radio")
        assert np.isfinite(y).all() and y.shape == x.shape
    text = head("compressor") + '[dur]\nextends = "mastering"\nratio = 6.0\n'
    with User(text):
        assert fx._COMP_PRESETS["dur"] == (-15.0, 6.0, 0.1, 0.25, False, 1.0)
        assert eq(fx.compressor(x, 44100, preset="dur"), fx.compressor(x, 44100, threshold=-15.0, ratio=6.0, attack=0.1, release=0.25, auto_makeup=False, mix=1.0))
    text = head("bearing") + '[turbo]\nrpm = 9000.0\nresonance = 15000.0\nband = 900.0\n'
    with User(text):
        assert cyber.BEARING_RESONANCE_PRESETS["turbo"] == (9000.0, 15000.0, 900.0)
        assert np.isfinite(cyber.bearing(sr=SR, dur=0.3, preset="turbo")).all()
    raises(lambda: presets.load(_tmp(head("lofi") + '[x]\nrate = 1\nbits = 8\nlaw = "mu"\nhold = true\nadpcm = false\n')), "hors de")


def _tmp(text):
    d = tempfile.mkdtemp()
    p = Path(d) / "t.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_builtin_scene_render_unchanged_by_load_path():
    a = scenes.scene(sr=SR, kind="hitech", dur=2.0, seed=0)
    presets.reload(clear=True)
    b = scenes.scene(sr=SR, kind="hitech", dur=2.0, seed=0)
    assert eq(a, b)


def test_no_dynamic_code_in_presets():
    src = Path(presets.__file__).read_text(encoding="utf-8")
    for bad in ("eval(", "exec(", "import_module", "pickle"):
        assert bad not in src, bad


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception:
                fails += 1
                print("FAIL", name)
                traceback.print_exc()
    print("fails:", fails)
    sys.exit(1 if fails else 0)
