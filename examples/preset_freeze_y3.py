"""Vehicle and engine families for ``preset_freeze``: two_stroke, engine_layout, diesel, tracks, moto_2t, scooter, garden, ev,
avas_style, interior, air_leak, door, horn, tyre_surface, rail_kind, ship, lap.

    uv run python examples/preset_freeze_y3.py --family two_stroke --out sounds/preset_ref_y3
    uv run python examples/preset_freeze_y3.py --export two_stroke
    uv run python examples/preset_freeze_y3.py --check two_stroke --out sounds/preset_ref_y3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preset_freeze import main, register_family  # noqa: E402


def _tab(mod, name):
    def f():
        import importlib
        return dict(getattr(importlib.import_module(f"genny.{mod}"), name))
    return f


def _same(**fields):
    def f(k, v):
        return {n: (v[i] if isinstance(i, int) else v) for n, i in fields.items()}
    return f


def _tuple(*names):
    return lambda k, v: dict(zip(names, v))


def _two_stroke_dict(k, v):
    out = {}
    for key, val in v.items():
        if key == "tuned_rpm" and val is None:
            out["no_pipe"] = True
        else:
            out[key] = val
    return out


def _layout_dict(k, v):
    return {("headers" if key == "header" and isinstance(val, tuple) else key): val for key, val in v.items()}


def _tracks_dict(k, v):
    d = dict(zip(("speed_kmh", "pitch_m", "wheels", "spacing_m", "teeth", "slack", "weight"), v[:7]))
    d["engine"], d["engine_rpm"] = v[7]
    return d


# ---- engines
def _two_stroke_cases():
    from genny import engines as E, heavy_vehicles as H
    out = {}
    for k in E.TWO_STROKE_KINDS:
        out[f"two_stroke.{k}"] = lambda k=k: E.two_stroke(kind=k, dur=0.5)
        out[f"two_stroke_idle.{k}"] = lambda k=k: E.two_stroke(kind=k, idle=1, dur=0.5)
        out[f"two_stroke_twin.{k}"] = lambda k=k: E.two_stroke(kind=k, cylinders=2, dur=0.4)
        if E.TWO_STROKE_KINDS[k].get("cut"):
            out[f"two_stroke_cut.{k}"] = lambda k=k: E.two_stroke(kind=k, cut="machine", dur=0.5)
    out["chainsaw"] = lambda: H.chainsaw(dur=0.5)
    out["chainsaw_limiter"] = lambda: H.chainsaw(dur=0.5, limiter=1)
    out["mower_2t"] = lambda: H.mower(kind="lawn_mower_2t", dur=0.5)
    out["mower_2t_limiter"] = lambda: H.mower(kind="lawn_mower_2t", dur=0.5, limiter=1)
    out["tractor_vintage_2t"] = lambda: H.tractor(kind="tractor_vintage", stroke=2, dur=0.5)
    return out


def _layout_cases():
    from genny import engines as E
    out = {}
    for k in E.LAYOUTS:
        if not k.startswith("diesel"):
            out[f"engine_layout.{k}"] = lambda k=k: E.engine_layout(kind=k, dur=0.5)
        out[f"engine_bank.{k}"] = lambda k=k: E.engine_event(kind="engine_bank", layout=k, cycles=2)
        out[f"rod_knock.{k}"] = lambda k=k: E.engine_event(kind="rod_knock", layout=k, dur=0.5)
    return out


def _layout_extras():
    from genny import engines as E
    out = {}
    for k in E.LAYOUTS:
        out[f"offsets.{k}"] = E.firing_offsets(k)
        out[f"banks.{k}"] = E.firing_banks(k)
        out[f"intervals.{k}"] = E.bank_intervals(k)
        out[f"cycle_per.{k}"] = E._cycle_per(k)
        out[f"params.{k}"] = E._params(k)
    return out


def _diesel_cases():
    from genny import engines as E, heavy_vehicles as H
    out = {}
    for k in E._DIESEL:
        out[f"diesel.{k}"] = lambda k=k: E.diesel_engine(kind=k, dur=0.5)
        out[f"diesel_load.{k}"] = lambda k=k: E.diesel_engine(kind=k, load=0.8, clatter=1.5, dur=0.5)
    out["tractor"] = lambda: H.tractor(dur=0.5)
    out["excavator_idle"] = lambda: H.excavator(dur=0.5)
    return out


# ---- heavy vehicles
def _tracks_cases():
    from genny import heavy_vehicles as H
    out = {}
    for k in H.TRACK_KINDS:
        out[f"tracks.{k}"] = lambda k=k: H.tracks(kind=k, dur=0.5)
        out[f"tracks_alone.{k}"] = lambda k=k: H.tracks(kind=k, engine=0.0, dur=0.5)
        out[f"tracks_stagger.{k}"] = lambda k=k: H.tracks(kind=k, engine=0.0, stagger=True, gravel=0.3, dur=0.4)
    out["tracks_snow_irregular"] = lambda: H.tracks(kind="snowmobile", irregular=True, dur=0.5)
    out["excavator_tracks"] = lambda: H.excavator(kind="tracks", dur=0.5)
    return out


def _moto_cases():
    from genny import heavy_vehicles as H
    out = {}
    for k in H.MOTO_KINDS:
        out[f"moto.{k}"] = lambda k=k: H.motorcycle(kind=k, dur=0.5)
        out[f"moto_speed.{k}"] = lambda k=k: H.motorcycle(kind=k, speed=60.0, dur=0.5)
    for k in H.MOTO_2T:
        out[f"moto_limiter.{k}"] = lambda k=k: H.motorcycle(kind=k, limiter=1, dur=0.5)
        out[f"moto_idle.{k}"] = lambda k=k: H.motorcycle(kind=k, idle=1, dur=0.5)
    return out


def _moto_extras():
    from genny import heavy_vehicles as H
    return {"MOTO_KINDS": H.MOTO_KINDS}


def _scooter_cases():
    from genny import heavy_vehicles as H
    out = {}
    for k in H.SCOOTER:
        out[f"scooter.{k}"] = lambda k=k: H.scooter(kind=k, dur=0.5)
        out[f"scooter_limiter.{k}"] = lambda k=k: H.scooter(kind=k, limiter=1, slip=0.2, dur=0.5)
        out[f"scooter_idle.{k}"] = lambda k=k: H.scooter(kind=k, idle=1, dur=0.5)
    return out


def _garden_cases():
    from genny import heavy_vehicles as H
    out = {}
    for k in H.GARDEN:
        out[f"garden.{k}"] = lambda k=k: H.garden_machine(kind=k, dur=0.5)
        out[f"garden_rpm.{k}"] = lambda k=k: H.garden_machine(kind=k, rpm=5000.0, load=0.4, dur=0.5)
    return out


# ---- road
def _ev_cases():
    from genny import road as R
    out = {}
    for k in R.EV:
        out[f"ev.{k}"] = lambda k=k: R.ev(kind=k, dur=0.5)
        out[f"ev_speed.{k}"] = lambda k=k: R.ev(kind=k, speed=35.0, load=0.9, dur=0.5)
    return out


def _avas_cases():
    from genny import road as R
    out = {}
    for k in R.AVAS_STYLES:
        out[f"avas.{k}"] = lambda k=k: R.avas(style=k, dur=0.5)
        out[f"avas_fmvss.{k}"] = lambda k=k: R.avas(style=k, rule="fmvss", speed=15.0, speed_end=30.0, dur=0.5)
    return out


def _interior_cases():
    from genny import road as R
    out = {}
    for k in R.INTERIORS:
        out[f"interior.{k}"] = lambda k=k: R.interior(kind=k, dur=0.5)
        out[f"interior_speed.{k}"] = lambda k=k: R.interior(kind=k, speed=50.0, dur=0.5)
    return out


def _air_cases():
    from genny import road as R, heavy_vehicles as H
    out = {f"air_system.{k}": (lambda k=k: R.air_system(kind=k)) for k in R.AIR_KINDS}
    out["bus"] = lambda: H.bus(dur=0.5)
    out["bus_doors"] = lambda: H.bus(doors=2, brake=1, dur=1.0)
    return out


def _air_extras():
    from genny import road as R
    return {"AIR_KINDS": R.AIR_KINDS}


def _door_cases():
    from genny import road as R
    out = {f"car_door.{k}": (lambda k=k: R.car_door(kind=k)) for k in R.DOOR_KINDS}
    out["car_door_quality"] = lambda: R.car_door(kind="car_door_cheap", quality=0.9)
    return out


def _door_extras():
    from genny import road as R
    return {"DOOR_KINDS": R.DOOR_KINDS}


def _horn_cases():
    from genny import road as R
    out = {f"air_horn.{k}": (lambda k=k: R.air_horn(kind=k, dur=0.4)) for k in R.HORN_KINDS}
    out["air_horn_freqs"] = lambda: R.air_horn(kind="truck_air_horn", freqs="285/330", dur=0.4)
    out["klaxon"] = lambda: R.klaxon(dur=0.4)
    return out


def _horn_extras():
    from genny import road as R
    return {"HORN_KINDS": R.HORN_KINDS}


def _tyre_cases():
    from genny import road as R
    out = {f"tyre_roll.{k}": (lambda k=k: R.tyre_roll(kind=k, dur=0.5)) for k in R.TYRE_KINDS}
    out["tyre_roll_fast"] = lambda: R.tyre_roll(kind="tyre_gravel", speed=100.0, tread=1.0, dur=0.5)
    return out


def _tyre_extras():
    from genny import road as R
    return {"TYRE_KINDS": R.TYRE_KINDS}


# ---- rail
def _rail_cases():
    from genny import rail as R
    out = {f"rail.{k}": (lambda k=k: R.rail(kind=k, dur=0.8)) for k in R.RAIL_KINDS}
    out["rail_speed"] = lambda: R.rail(kind="train_welded", speed=120.0, dur=0.5)
    return out


def _rail_extras():
    from genny import rail as R
    return {"RAIL_KINDS": R.RAIL_KINDS}


# ---- marine
def _ship_cases():
    from genny import marine as M
    out = {}
    for k in M.SHIPS:
        out[f"ship_under.{k}"] = lambda k=k: M.ship(kind=k, dur=0.8)
        out[f"ship_air.{k}"] = lambda k=k: M.ship(kind=k, listen="air", dur=0.8)
        out[f"ship_fast.{k}"] = lambda k=k: M.ship(kind=k, speed=20.0, dur=0.8)
    return out


def _lap_cases():
    from genny import marine as M
    out = {}
    for k in M.LAPS:
        out[f"lap.{k}"] = lambda k=k: M.lap(kind=k, dur=2.0)
        out[f"lap_plume.{k}"] = lambda k=k: M.lap(kind=k, dur=2.0, plume=0.1)
    return out


register_family("two_stroke", _tab("engines", "TWO_STROKE_KINDS"), _two_stroke_cases, to_dict=_two_stroke_dict)
register_family("engine_layout", _tab("engines", "LAYOUTS"), _layout_cases, _layout_extras, _layout_dict)
register_family("diesel", _tab("engines", "_DIESEL"), _diesel_cases)
register_family("tracks", _tab("heavy_vehicles", "TRACK_KINDS"), _tracks_cases, to_dict=_tracks_dict)
register_family("moto_2t", _tab("heavy_vehicles", "MOTO_2T"), _moto_cases, _moto_extras, _same(preset=None))
register_family("scooter", _tab("heavy_vehicles", "SCOOTER"), _scooter_cases, to_dict=_same(rpm=None))
register_family("garden", _tab("heavy_vehicles", "GARDEN"), _garden_cases, to_dict=_same(rpm=None))
register_family("ev", _tab("road", "EV"), _ev_cases, to_dict=_tuple("speed_kmh", "wheel_m", "reduction", "poles", "slots", "tyre"))
register_family("avas_style", _tab("road", "AVAS_STYLES"), _avas_cases, to_dict=_tuple("band1", "band2", "fm"))
register_family("interior", _tab("road", "INTERIORS"), _interior_cases, to_dict=_tuple("rpm_per_kmh", "cylinders", "vibration", "shell_m"))
register_family("air_leak", _tab("road", "AIR_LEAKS"), _air_cases, _air_extras, _tuple("burst_s", "orifice_mm"))
register_family("door", _tab("road", "DOORS"), _door_cases, _door_extras, _tuple("band", "quality", "panel_m", "body_hz", "length_s"))
register_family("horn", _tab("road", "HORNS"), _horn_cases, _horn_extras, _tuple("tones", "radius_m", "drive"))
register_family("tyre_surface", _tab("road", "SURFACES"), _tyre_cases, _tyre_extras, _tuple("grains", "gain", "hp_hz"))
register_family("rail_kind", _tab("rail", "SPEEDS"), _rail_cases, _rail_extras, _same(speed_kmh=None))
register_family("ship", _tab("marine", "SHIPS"), _ship_cases)
register_family("lap", _tab("marine", "LAPS"), _lap_cases)


if __name__ == "__main__":
    sys.exit(main())
