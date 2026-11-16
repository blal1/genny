"""Scene and isolated preset families for ``preset_freeze``: scene, lofi, spring, compressor, free_reed, bearing.

    uv run python examples/preset_freeze_y4.py --family scene --out sounds/preset_ref_y4
    uv run python examples/preset_freeze_y4.py --export scene
    uv run python examples/preset_freeze_y4.py --check scene --out sounds/preset_ref_y4
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preset_freeze import main, register_family  # noqa: E402

SR = 22050


def _tab(mod, name):
    def f():
        import importlib
        return dict(getattr(importlib.import_module(f"genny.{mod}"), name))
    return f


def _signal(n=8000, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 44100.0
    return 0.4 * np.sin(2 * np.pi * 330.0 * t) + 0.1 * rng.standard_normal(n)


def _layer_dict(k, v):
    out = {}
    for key, val in v.items():
        if key == "layers":
            out[key] = [{kk: vv for kk, vv in layer.items() if not (kk == "every" and vv is None) and not (kk == "sfx" and "inst" in layer)
                         and not (kk in ("wide", "opt") and not vv) and not (kk == "params" and not vv)} for layer in val]
        elif val is not None:
            out[key] = val
    return out


def _scene_cases():
    from genny import scenes as S
    out = {}
    for k in S.SCENES:
        out[f"scene.{k}"] = lambda k=k: S.scene(sr=SR, kind=k, dur=3.0, seed=0)
    for k in ("forest", "hitech", "religious"):
        out[f"scene_loop.{k}"] = lambda k=k: S.scene(sr=SR, kind=k, dur=3.0, seed=1, loop=1)
    out["scene_climax"] = lambda: S.scene(sr=SR, kind="warfare", dur=6.0, seed=2, climax=3.0)
    return out


def _scene_extras():
    from genny import scenes as S
    return {"UCS": S.UCS, "TEXT": S.TEXT, "ROOM_TONES": S.ROOM_TONES}


def _lofi_cases():
    from genny import dsp as D
    x = _signal()
    out = {f"lofi.{k}": (lambda k=k: D.lofi(x, 44100, preset=k)) for k in D.LOFI_PRESETS}
    out["lofi.none"] = lambda: D.lofi(x, 44100, preset="none")
    out["lofi.hold_off"] = lambda: D.lofi(x, 44100, preset="nes", hold=False, seed=2)
    return out


def _spring_cases():
    from genny import fdstring as S
    x = _signal(6000)
    out = {f"spring.{k}": (lambda k=k: S.spring_reverb(x, 44100, preset=k)) for k in S.SPRING_PRESETS}
    out["spring.two"] = lambda: S.spring_reverb(x, 44100, preset="sansui", tank="two_spring", efficient=True)
    return out


def _comp_cases():
    from genny import fx as X
    x = _signal(12000)
    out = {f"compressor.{k}": (lambda k=k: X.compressor(x, 44100, preset=k)) for k in X._COMP_PRESETS}
    out["compressor.none"] = lambda: X.compressor(x, 44100)
    return out


def _reed_cases():
    from genny import family_v as V
    out = {}
    for k in V.FREE_REED_PRESETS:
        out[f"reed.{k}"] = lambda k=k: V.free_reed_voice(220.0, 0.15, 22050, k)
        out[f"reed_over.{k}"] = lambda k=k: V.free_reed_voice(330.0, 0.1, 22050, k, vel=0.5, Q=80.0)
    from genny.instruments import render_note
    for name in ("accordion_reed", "harmonica_reed", "melodica_reed", "sho_reed", "khaen_reed"):
        try:
            out[f"inst.{name}"] = lambda name=name: render_note(name, 261.63, 0.15, 22050)
        except Exception:
            pass
    return out


def _bearing_cases():
    from genny import cyber as C
    out = {}
    for k in C.BEARING_RESONANCE_PRESETS:
        out[f"bearing.{k}"] = lambda k=k: C.bearing(sr=22050, dur=0.6, preset=k)
        out[f"bearing_rpm.{k}"] = lambda k=k: C.bearing(sr=22050, dur=0.6, preset=k, rpm=900.0, state="spall")
    return out


register_family("scene", _tab("scenes", "SCENES"), _scene_cases, _scene_extras, _layer_dict)
register_family("lofi", _tab("dsp", "LOFI_PRESETS"), _lofi_cases, to_dict=lambda k, v: {key: val for key, val in v.items() if not (key == "band" and val is None)})
register_family("spring", _tab("fdstring", "SPRING_PRESETS"), _spring_cases)
register_family("compressor", _tab("fx", "_COMP_PRESETS"), _comp_cases, to_dict=lambda k, v: dict(zip(("threshold", "ratio", "attack", "release", "auto_makeup", "mix"), v)))
register_family("free_reed", _tab("family_v", "FREE_REED_PRESETS"), _reed_cases)
register_family("bearing", _tab("cyber", "BEARING_RESONANCE_PRESETS"), _bearing_cases, to_dict=lambda k, v: dict(zip(("rpm", "resonance", "band"), v)))

if __name__ == "__main__":
    sys.exit(main())
