"""Preset loader: validation, inheritance, precedence, cache, inline dicts. Runs under pytest or alone: python tests/test_presets.py"""
import contextlib
import os
import random
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

from genny import presets as P

FIELDS = {
    "size": P.F(float, lo=0.5, hi=10.0, unit="m"),
    "n": P.F(int, lo=0, hi=9, default=1),
    "kind": P.F(str, choices=("a", "b"), default="a"),
    "pair": P.F(tuple, items=(float, str), required=False),
    "link": P.F(ref="tp_ref", required=False),
}
HEAD = 'schema = "genny.preset/1"\nfamily = "{}"\n'


@contextlib.contextmanager
def sandbox(fam):
    with tempfile.TemporaryDirectory() as d:
        keep = {k: os.environ.get(k) for k in ("GENNY_PRESETS", "GENNY_PRESETS_USER")}
        old = P._BUILTIN
        P._BUILTIN = Path(d) / "builtin"
        P._BUILTIN.mkdir()
        os.environ["GENNY_PRESETS_USER"] = ""
        os.environ.pop("GENNY_PRESETS", None)
        P.reload(clear=True)
        try:
            yield Path(d)
        finally:
            for k, v in keep.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v
            P._BUILTIN = old
            P.reload(clear=True)


def put(dirpath, fname, fam, body):
    p = Path(dirpath) / fname
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(HEAD.format(fam) + body, encoding="utf-8")
    return p


def fails(fn, *needles):
    try:
        fn()
    except P.PresetError as e:
        msg = str(e)
        for n in needles:
            assert n in msg, (n, msg)
        return msg
    raise AssertionError("PresetError attendue")


def test_valid_and_types_kept():
    with sandbox("t1") as d:
        put(d / "builtin", "f.toml", "t1", '[a]\nsize = 2\n[b]\nextends = "a"\nsize = 3.5\npair = [1.5, "x"]\n')
        v = P.schema("t1", FIELDS)
        assert repr(v["a"]) == "{'size': 2, 'n': 1, 'kind': 'a'}"
        assert type(v["a"]["size"]) is int and v["b"]["pair"] == (1.5, "x")
        assert P.origin("t1", "a") == "builtin"
        assert P.list("t1") == ["a", "b"] and "t1" in P.list()


def test_errors_cite_file_table_key():
    with sandbox("t2") as d:
        P.schema("t2", FIELDS)
        cases = [
            ('[x]\nsise = 1.0\n', ("x.toml", "[x].sise", "size")),
            ('[x]\nsize = 99.0\n', ("x.toml", "[x].size", "[0.5, 10.0]", "m")),
            ('[x]\nsize = "a"\n', ("[x].size", "float")),
            ('[x]\nsize = 1.0\nn = 2.0\n', ("[x].n", "int")),
            ('[x]\nsize = 1.0\nkind = "c"\n', ("[x].kind",)),
            ('[x]\nsize = 1.0\npair = [1.0]\n', ("[x].pair",)),
            ('[x]\nn = 1\n', ("[x].size", "requis")),
            ('[x]\nextends = "nope"\n', ("[x].extends", "nope")),
            ('[x]\nextends = "x"\n', ("[x].extends", "cycle")),
            ('[x]\nextends = "y"\nsize=1.0\n[y]\nextends = "x"\n', ("cycle",)),
            ('["Bad Name"]\nsize = 1.0\n', ("Bad Name", "invalide")),
            ('[x]\nsize = inf\n', ("[x]", "non finie")),
        ]
        for body, needles in cases:
            p = put(d, "x.toml", "t2", body)
            fails(lambda: P.load(p), *needles)


def test_depth_limit():
    with sandbox("t3") as d:
        P.schema("t3", FIELDS)
        body = "[p0]\nsize = 1.0\n" + "".join(f'[p{i}]\nextends = "p{i - 1}"\n' for i in range(1, 10))
        p = put(d, "deep.toml", "t3", body)
        fails(lambda: P.load(p), "profondeur", "deep.toml", "[p9].extends")
        ok = put(d, "ok.toml", "t3", "[p0]\nsize = 1.0\n" + "".join(f'[p{i}]\nextends = "p{i - 1}"\n' for i in range(1, 9)))
        assert P.load(ok)["t3"]["p8"]["size"] == 1.0


def test_schema_version_and_size():
    with sandbox("t4") as d:
        P.schema("t4", FIELDS)
        p = Path(d) / "v2.toml"
        p.write_text('schema = "genny.preset/2"\nfamily = "t4"\n[x]\nsize = 1.0\n', encoding="utf-8")
        fails(lambda: P.load(p), "v2.toml", ".schema", "mettez genny à jour")
        big = Path(d) / "big.toml"
        big.write_text(HEAD.format("t4") + "# " + "x" * (1 << 20) + "\n[x]\nsize = 1.0\n", encoding="utf-8")
        fails(lambda: P.load(big), "big.toml", "1 MiB")
        s = Path(d) / "s.toml"
        s.write_text(HEAD.format("t4") + '[x]\nsize = 1.0\nkind = "' + "a" * 300 + '"\n', encoding="utf-8")
        fails(lambda: P.load(s), "s.toml", "[x]")


def test_inheritance_and_inline_dict():
    with sandbox("t5") as d:
        put(d / "builtin", "f.toml", "t5", '[a]\nsize = 2.0\nn = 3\n[b]\nextends = "a"\nsize = 4.0\n')
        P.schema("t5", FIELDS)
        assert P.get("t5", "b") == {"size": 4.0, "n": 3, "kind": "a"}
        inline = P.resolve("t5", {"extends": "a", "size": 5.0})
        assert inline == {"size": 5.0, "n": 3, "kind": "a"} and repr(inline) == repr({"size": 5.0, "n": 3, "kind": "a"})
        fails(lambda: P.resolve("t5", {"size": 99.0}), "<inline>", "[0.5, 10.0]")
        fails(lambda: P.resolve("t5", {"n": 1}), "requis")
        fails(lambda: P.resolve("t5", {"extends": "zz", "size": 1.0}), "zz")
        fails(lambda: P.resolve("t5", "zz"), "zz")
        fails(lambda: P.resolve("t5", 3.0), "float")
        g = P.get("t5", "a")
        g["size"] = 9.0
        assert P.get("t5", "a")["size"] == 2.0


def test_ref():
    with sandbox("tp") as d:
        put(d / "builtin", "r.toml", "tp_ref", "[one]\nx = 1\n")
        put(d / "builtin", "f.toml", "tp", '[a]\nsize = 1.0\nlink = "one"\n[b]\nsize = 1.0\nlink = "ona"\n')
        P.schema("tp_ref", {"x": P.F(int)})
        user = put(d, "u.toml", "tp", '[c]\nsize = 1.0\nlink = "ona"\n')
        P._BUILTIN.joinpath("f.toml").write_text(HEAD.format("tp") + '[a]\nsize = 1.0\nlink = "one"\n', encoding="utf-8")
        P.reload()
        v = P.schema("tp", FIELDS)
        assert v["a"]["link"] == "one"
        fails(lambda: P.load(user), "[c].link", "one")


def test_precedence():
    with sandbox("t6") as d:
        put(d / "builtin", "f.toml", "t6", '[a]\nsize = 1.0\n[only_builtin]\nsize = 1.0\n')
        udir = d / "user"
        put(udir, "u.toml", "t6", '[a]\nsize = 2.0\n[only_user]\nsize = 2.0\n')
        env_f = put(d / "env", "e.toml", "t6", '[a]\nsize = 3.0\n')
        reg_f = put(d / "reg", "r.toml", "t6", '[a]\nsize = 4.0\n')
        v = P.schema("t6", FIELDS)
        assert v["a"]["size"] == 1.0
        os.environ["GENNY_PRESETS_USER"] = str(udir)
        P.reload()
        assert v["a"]["size"] == 2.0 and set(v) == {"a", "only_builtin", "only_user"}
        assert P.origin("t6", "only_builtin") == "builtin" and P.origin("t6", "only_user").endswith("u.toml")
        os.environ["GENNY_PRESETS"] = str(env_f)
        P.reload()
        assert v["a"]["size"] == 3.0
        P.register(d / "reg")
        assert v["a"]["size"] == 4.0 and P.origin("t6", "a") == str(reg_f.resolve())
        os.environ["GENNY_PRESETS"] = ""
        P.reload(clear=True)
        assert v["a"]["size"] == 2.0


def test_register_refuses_bad_file_and_user_file_only_warns():
    with sandbox("t7") as d:
        v = P.schema("t7", FIELDS)
        bad = put(d / "u", "bad.toml", "t7", "[x]\nsize = 100.0\n")
        fails(lambda: P.register(bad), "bad.toml", "[x].size")
        assert "x" not in v and not P._REG
        os.environ["GENNY_PRESETS"] = str(bad)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            P._WARNED.clear()
            P.reload()
        assert any("bad.toml" in str(x.message) for x in w) and "x" not in v and P.problems()
        os.environ["GENNY_PRESETS"] = ""
        P.reload()
        put(d / "builtin", "b.toml", "t7", "[y]\nsize = 100.0\n")
        fails(P.reload, "b.toml", "[y].size")
        (d / "builtin" / "b.toml").unlink()


def test_cache_and_live_view():
    with sandbox("t8") as d:
        f = put(d / "builtin", "f.toml", "t8", "[a]\nsize = 1.0\n")
        v = P.schema("t8", FIELDS)
        n = P.parse_count()
        P.reload()
        P.reload()
        assert P.parse_count() == n
        f.write_text(HEAD.format("t8") + "[a]\nsize = 2.0\n[b]\nsize = 3.0\n", encoding="utf-8")
        P.reload()
        assert P.parse_count() == n + 1 and set(v) == {"a", "b"} and v["a"]["size"] == 2.0
        other = put(d / "extra", "o.toml", "t8", "[c]\nsize = 4.0\n")
        alias = v
        P.register(other)
        assert alias is v and "c" in v and P.get("t8", "c")["size"] == 4.0


def test_pending_family_is_validated_on_declaration():
    with sandbox("t9") as d:
        put(d / "builtin", "f.toml", "t9", "[a]\nsize = 1.0\n")
        P.reload()
        assert "t9" in P.list() and P.list("t9") == ["a"]
        fails(lambda: P.get("t9", "a"), "non déclarée")
        assert P.schema("t9", FIELDS)["a"]["size"] == 1.0


def test_shape_view_and_defaults_order():
    with sandbox("t10") as d:
        put(d / "builtin", "f.toml", "t10", "[a]\nsize = 1.0\n")
        v = P.schema("t10", FIELDS, shape=lambda dd: tuple(dd.values()))
        assert v["a"] == (1.0, 1, "a")


def test_save_load_roundtrip_repr():
    with sandbox("t11") as d:
        P.schema("t11", {"x": P.F(float, lo=-1e300, hi=1e300), "i": P.F(int), "t": P.F(tuple, of=float), "s": P.F(str)})
        path = d / "out" / "s.toml"
        rnd = random.Random(7)
        for i in range(10000):
            x = rnd.choice([rnd.uniform(-1e3, 1e3), rnd.random() * 10 ** rnd.randint(-300, 300), float(rnd.randint(-5, 5))])
            data = {"x": x, "i": rnd.randint(-2**62, 2**62), "t": [rnd.random(), -rnd.random() * 1e-12], "s": "é\"\\\x7f'" + str(i)}
            if i % 500 == 0:
                P.save("t11", "k", data, path)
                got = P.load(path)["t11"]["k"]
                want = P.resolve("t11", data)
                assert repr(got) == repr(want), (got, want)
            else:
                from genny.presets import _val
                import tomllib
                back = tomllib.loads(f"x = {_val(x)}")["x"]
                assert repr(back) == repr(x)
        P.save("t11", "k2", {"extends": "k", "x": 1}, path)
        assert type(P.load(path)["t11"]["k2"]["x"]) is int and set(P.load(path)["t11"]) == {"k", "k2"}


def test_no_code_execution_tokens():
    src = Path(P.__file__).read_text(encoding="utf-8")
    for tok in ("eval", "exec", "import_module", "pickle"):
        assert tok not in src, tok


def test_bad_user_file_does_not_break_import():
    with tempfile.TemporaryDirectory() as d:
        bad = Path(d) / "bad.toml"
        bad.write_text("this is = = not toml\n", encoding="utf-8")
        env = dict(os.environ, GENNY_PRESETS=str(bad), GENNY_PRESETS_USER="")
        code = "import warnings\nwarnings.simplefilter('always')\nimport genny, genny.presets as p\nprint('OK', len(p.problems()))\n"
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        assert r.returncode == 0 and "OK 1" in r.stdout and "bad.toml" in r.stderr, (r.stdout, r.stderr)


def test_freeze_export_check_detects_one_ulp():
    import importlib.util
    import io
    import numpy as np
    spec = importlib.util.spec_from_file_location("preset_freeze", Path(__file__).resolve().parents[1] / "examples" / "preset_freeze.py")
    pf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pf)
    with sandbox("tfz") as d:
        table = {"a": {"f": 440.0, "g": 0.25, "n": 3}, "b": {"f": 523.25, "g": 0.5, "n": 4}}
        live = {k: dict(v) for k, v in table.items()}
        tick = {"n": 0}

        def sine(k):
            return lambda: np.sin(2 * np.pi * live[k]["f"] * np.arange(2000) / 44100) * live[k]["g"] * live[k]["n"]

        def flaky():
            tick["n"] += 1
            return np.full(4, float(tick["n"]))

        pf.register_family("tfz", lambda: live, lambda: {"sine_a": sine("a"), "sine_b": sine("b"), "flaky": flaky})
        ref = d / "ref"
        assert pf.freeze("tfz", ref) == 0
        assert pf.check("tfz", ref) == 0
        meta = __import__("json").loads((ref / "tfz_table.json").read_text(encoding="utf-8"))
        assert meta["nondeterministic"] == ["flaky"]
        toml = d / "out" / "tfz.toml"
        assert pf.export("tfz", toml) == 0
        P.schema("tfz", {"f": P.F(float), "g": P.F(float), "n": P.F(int)})
        assert [repr(v) for v in P.load(toml)["tfz"].values()] == [repr(v) for v in table.values()]
        live["b"]["f"] = float(np.nextafter(523.25, 1e9))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = pf.check("tfz", ref)
        out = buf.getvalue()
        assert code == 1 and "entry b" in out and "render sine_b" in out and "sine_a" not in out, out
        live["b"]["n"] = 4.0
        live["b"]["f"] = 523.25
        assert pf.check("tfz", ref) == 1


def test_example_file_loads_without_real_family():
    assert "example" in P.list() and P.list("example") == ["base", "wide"]


if __name__ == "__main__":
    fails_n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:
                fails_n += 1
                print("FAIL", name, type(e).__name__, str(e)[:300])
    sys.exit(1 if fails_n else 0)
