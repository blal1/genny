"""Moteurs et vehicules en presets TOML (hors pytest) : uv run python tests/test_y3.py"""
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

os.environ["GENNY_PRESETS_USER"] = ""
os.environ.pop("GENNY_PRESETS", None)

from genny import engines as E, heavy_vehicles as H, marine as M, presets, rail, road  # noqa: E402

TOML = Path(presets.__file__).with_name("presets")
COUNTS = {"two_stroke": 9, "engine_layout": 21, "diesel": 2, "tracks": 7, "moto_2t": 2, "scooter": 2, "garden": 3, "ev": 3, "avas_style": 3,
          "interior": 4, "air_leak": 4, "door": 5, "horn": 6, "tyre_surface": 5, "rail_kind": 8, "ship": 2, "lap": 3}


class User:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        self.dir = tempfile.TemporaryDirectory()
        p = Path(self.dir.name) / "user.toml"
        p.write_text(self.text, encoding="utf-8")
        presets.register(p)
        return p

    def __exit__(self, *a):
        presets.reload(clear=True)
        self.dir.cleanup()


def head(family):
    return f'schema = "genny.preset/1"\nfamily = "{family}"\n'


def test_builtin_files_and_counts():
    for f, n in COUNTS.items():
        assert (TOML / f"{f}.toml").is_file(), f
        assert len(presets.list(f)) == n, f
        assert all(presets.origin(f, k) == "builtin" for k in presets.list(f)), f
    assert len(E.TWO_STROKE_KINDS) == 9 and len(E.LAYOUTS) == 21 and len(H.TRACK_KINDS) == 7 and len(road.DOORS) == 5 and len(rail.SPEEDS) == 8


def test_views_keep_their_types():
    assert E.TWO_STROKE_KINDS["chainsaw"]["tuned_rpm"] is None and "no_pipe" not in E.TWO_STROKE_KINDS["chainsaw"]
    assert isinstance(E.LAYOUTS["inline_4"]["header"], tuple) and isinstance(E.LAYOUTS["wankel"]["header"], float)
    assert isinstance(H.TRACK_KINDS["tank_m1"], tuple) and H.TRACK_KINDS["tank_m1"][7] == ("turbine", 0.0)
    assert H.TRACK_KINDS["tank_leopard1"][1] == 0.169 and isinstance(H.TRACK_KINDS["tank_leopard1"][2], int)
    assert H.SCOOTER["2t"] == 7000.0 and H.MOTO_2T["cross_125"] == "mx125" and rail.SPEEDS["train_welded"] == 300.0
    assert road.EV["trottinette"][3:5] == (15, 27) and road.DOORS["trunk_slam"][0] == (80.0, 1200.0) and road.HORNS["vuvuzela"][0] == (235.0,)
    assert E._DIESEL["diesel_i4"] == {"bore": 0.081, "pipe": 7.0} and M.LAPS["quai"]["modes"] == ()


def test_two_stroke_inline_and_user():
    a = E.two_stroke(kind={"extends": "mx125", "rpm": 12000.0}, dur=0.3)
    b = E.two_stroke(kind="mx125", rpm=12000.0, dur=0.3)
    assert np.array_equal(a, b)
    with User(head("two_stroke") + '[mx125_fast]\nextends = "mx125"\nrpm = 12000.0\n'):
        assert "mx125_fast" in E.TWO_STROKE_KINDS and E.TWO_STROKE_KINDS["mx125_fast"]["limit"] == 12500.0
        assert np.array_equal(E.two_stroke(kind="mx125_fast", dur=0.3), b)
    assert "mx125_fast" not in E.TWO_STROKE_KINDS


def test_user_no_pipe_and_moto():
    with User(head("two_stroke") + '[saw2]\nextends = "chainsaw"\nrpm = 8000.0\n[pipe2]\nextends = "chainsaw"\nno_pipe = false\ntuned_rpm = 9000.0\n'):
        assert E.TWO_STROKE_KINDS["saw2"]["tuned_rpm"] is None and E.TWO_STROKE_KINDS["pipe2"]["tuned_rpm"] == 9000.0
        assert np.isfinite(E.two_stroke(kind="saw2", dur=0.3)).all()
    with User(head("moto_2t") + '[cross_80]\npreset = "mx125"\n'):
        assert np.array_equal(H.motorcycle(kind="cross_80", dur=0.3), H.motorcycle(kind="cross_125", dur=0.3))
        assert "cross_80" in H.MOTO_2T


def test_user_layout_equals_builtin():
    text = head("engine_layout") + '[my_inline4]\noffsets = [0.0, 0.25, 0.5, 0.75]\nrpm = 883.0\nheaders = [0.958, 0.333, 0.708, 0.958]\n[my_twin]\noffsets = [0.0, 0.25]\nrpm = 2000.0\nheader = 2.0\n'
    with User(text):
        assert E.LAYOUTS["my_inline4"] == E.LAYOUTS["inline_4"]
        assert np.array_equal(E.engine_layout(kind="my_inline4", dur=0.3), E.engine_layout(kind="inline_4", dur=0.3))
        assert np.isfinite(E.engine_layout(kind="my_twin", dur=0.3)).all()


def test_user_diesel_needs_layout():
    with User(head("diesel") + '[solo]\nbore = 0.1\npipe = 5.0\n'):
        try:
            E.diesel_engine(kind="solo", dur=0.2)
        except ValueError as e:
            assert "engine_layout" in str(e)
        else:
            raise AssertionError("diesel kind without a layout")


def test_tracks_inline_equals_params():
    a = H.tracks(kind={"extends": "tank_diesel", "speed_kmh": 20.0}, dur=0.3)
    b = H.tracks(kind="tank_diesel", speed=20.0, dur=0.3)
    assert np.array_equal(a, b)
    with User(head("tracks") + '[crawler]\nextends = "bulldozer"\nweight = 0.5\n'):
        assert np.isfinite(H.tracks(kind="crawler", dur=0.3)).all()


def test_user_scooter_garden_ev_interior():
    with User(head("scooter") + '[3t]\nrpm = 6500.0\n'):
        assert np.array_equal(H.scooter(kind="3t", dur=0.3), H.scooter(kind="4t", rpm=6500.0, dur=0.3))
    with User(head("garden") + '[mini]\nrpm = 7000.0\n'):
        assert np.isfinite(H.garden_machine(kind="mini", dur=0.3)).all()
    with User(head("ev") + '[kart]\nextends = "voiturette"\nspeed_kmh = 30.0\n'):
        assert road.EV["kart"][0] == 30.0 and np.isfinite(road.ev(kind="kart", dur=0.3)).all()
    with User(head("interior") + '[van]\nextends = "bus"\nvibration = 0.3\n'):
        assert np.isfinite(road.interior(kind="van", dur=0.3)).all()
    with User(head("avas_style") + '[soft]\nband1 = 0.5\nband2 = 0.5\nfm = 0.0\n'):
        assert np.isfinite(road.avas(style="soft", dur=0.3)).all()


def test_user_door_horn_air_surface_rail():
    with User(head("door") + '[my_door]\nextends = "car_door_close"\nquality = 0.2\n'):
        assert "my_door" in road.DOORS
        assert np.array_equal(road.car_door(kind="my_door"), road.car_door(kind="car_door_close", quality=0.2))
    with User(head("horn") + '[my_horn]\nextends = "truck_air_horn"\ntones = [200.0, 240.0]\n'):
        assert np.array_equal(road.air_horn(kind="my_horn", dur=0.3), road.air_horn(kind="truck_air_horn", freqs="200/240", dur=0.3))
    with User(head("air_leak") + '[hiss]\nburst_s = 0.3\norifice_mm = 6.0\n'):
        assert np.isfinite(road.air_system(kind="hiss")).all()
    with User(head("tyre_surface") + '[tyre_ash]\ngrains = "Sandpaper"\ngain = 0.3\nhp_hz = 200.0\n'):
        assert np.array_equal(road.tyre_roll(kind="tyre_ash", dur=0.3), road.tyre_roll(kind="tyre_sand", dur=0.3))
    with User(head("rail_kind") + '[train_welded]\nspeed_kmh = 100.0\n'):
        assert np.array_equal(rail.rail(kind="train_welded", dur=0.3), rail.rail(kind="train_welded", speed=100.0, dur=0.3))
    assert rail.SPEEDS["train_welded"] == 300.0


def test_marine_user():
    with User(head("ship") + '[ferry]\nextends = "cargo"\nrpm = 150.0\nblades = 4\n'):
        assert np.isfinite(M.ship(kind="ferry", dur=0.8)).all()
    with User(head("lap") + '[pond]\nextends = "baignade"\nbubbles = 10.0\n'):
        assert np.isfinite(M.lap(kind="pond", dur=2.0)).all()


def test_errors_name_file_table_key():
    for fam, body, frag in (("two_stroke", "[a]\nrpm = 1.0e9\n", "hors de"), ("two_stroke", "[a]\nrpm = 7000.0\nrmp = 1.0\n", "clé inconnue"),
                            ("tracks", '[a]\nextends = "tank_m1"\nengine = "steam"\n', "parmi"), ("moto_2t", '[a]\npreset = "nope"\n', "inconnu"),
                            ("door", '[a]\nextends = "trunk_slam"\nband = [1.0]\n', "éléments attendus"), ("engine_layout", "[a]\noffsets = [0.0, 2.0]\nrpm = 1.0\n", "hors de")):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.toml"
            p.write_text(head(fam) + body, encoding="utf-8")
            try:
                presets.register(p)
            except presets.PresetError as e:
                assert "bad.toml" in str(e) and "[a]" in str(e) and frag in str(e), str(e)
            else:
                raise AssertionError(fam)
            presets.reload(clear=True)
    for fn, arg in ((E.two_stroke, "nope"), (H.tracks, "nope"), (road.car_door, "nope"), (rail.rail, "nope")):
        try:
            fn(kind=arg, dur=0.2) if fn in (E.two_stroke, H.tracks, rail.rail) else fn(kind=arg)
        except ValueError:
            pass
        else:
            raise AssertionError(fn)


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
