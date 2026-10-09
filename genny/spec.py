"""JSON sound spec renderer. A spec describes layers (synth notes, sequences, drums, sfx, files)
placed on a timeline, each with its own fx chain, plus a master fx chain.

Minimal spec:   {"out": "beep.wav", "layers": [{"type": "sfx", "kind": "beep"}]}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import drums as D
from . import fx as FX
from . import instruments as I
from . import sfx as S
from . import speech as SPK
from .core import DEFAULT_SR, declick, mix, normalize, read_wav, resample, samples, silence, trim_tail, write_wav
from .analysis import ear_loudness
from .notes import parse_pitch_list, parse_sequence

DEFAULT_MAX_LOUDNESS = -6.0   # ear-weighted dB (see analysis.ear_loudness); "max_loudness": null disables


class SpecError(ValueError):
    pass


LAYER_TYPES: dict = {}   # extra layer types registered by other modules: name -> fn(layer, sr, q) -> ndarray


LAYER_HELP: dict = {}    # name -> one-line description (defaults to the function's docstring)


def layer_type(name: str, help: str | None = None):
    def deco(fn):
        LAYER_TYPES[name] = fn
        LAYER_HELP[name] = help or ((fn.__doc__ or "").strip().splitlines() or [""])[0]
        return fn
    return deco


def _params(layer: dict) -> dict:
    p = dict(layer.get("params") or {})
    return p


def _fx(x: np.ndarray, chain, sr: int) -> np.ndarray:
    if not chain:
        return x
    return FX.apply_chain(x, chain, sr)


def render_layer(layer: dict, sr: int, bpm: float | None = None) -> tuple[np.ndarray, float]:
    """Returns (signal, start_seconds).

    With a tempo in effect (`"bpm"` on the spec, a group or the layer itself) every musical time in
    the layer is in beats instead of seconds: `at`, `dur`, step durations, `step`, `gap`, `hits`,
    `every`. Performance and sound-design times (`strum`, instrument/sfx params, fx) stay in seconds."""
    kind = layer.get("type")
    bpm = layer.get("bpm", bpm)
    q = 60.0 / float(bpm) if bpm else 1.0      # seconds per time unit
    at = float(layer.get("at", 0.0)) * q
    gain = float(layer.get("gain", 1.0))
    vel = float(layer.get("vel", 1.0))

    if kind in ("synth", "note", "notes"):
        inst = layer.get("inst", "synth")
        notes = layer.get("notes", layer.get("note", "C4"))
        freqs = parse_pitch_list(notes)
        dur = float(layer.get("dur", 1.0 if bpm else 0.5)) * q
        strum = float(layer.get("strum", 0.0))
        y = I.render_chord(inst, freqs, dur, sr, vel, strum=strum, **_params(layer))

    elif kind in ("seq", "sequence", "melody"):
        y = render_sequence(layer, sr, q)

    elif kind == "drum":
        y = D.render_drum(layer.get("kind", "kick"), sr, vel, **_params(layer))

    elif kind == "pattern":
        # drum pattern: {"type":"pattern","kind":"hihat","hits":[0,0.25,0.5],"params":{}} or "steps":"x-x-x-x-" with "step":0.125
        # In "steps": x = hit, X = accent, o = ghost note, anything else = rest; spaces and | are ignored.
        name, params = layer.get("kind", "kick"), _params(layer)
        if layer.get("hits") is not None or "steps" not in layer:
            one = D.render_drum(name, sr, vel, **params)
            y = mix([(one, float(t) * q) for t in (layer.get("hits") or [0.0])], sr)
        else:
            step = float(layer.get("step", 0.25 if bpm else 0.125)) * q
            swing = float(layer.get("swing", 0.0))            # delays every other step by this fraction of a step
            levels = {"x": 1.0, "X": float(layer.get("accent", 1.4)), "o": float(layer.get("ghost", 0.4))}
            steps = [ch for ch in str(layer["steps"]) if ch not in " |"]
            bars = max(1, int(layer.get("bars", 1)))          # play the written pattern this many times
            parts, cache = [], {}
            for i, ch in enumerate(steps * bars):
                if ch in levels:
                    if ch not in cache:
                        cache[ch] = D.render_drum(name, sr, min(1.0, vel * levels[ch]), **params) * levels[ch]
                    parts.append((cache[ch], (i + (swing if i % 2 else 0.0)) * step))
            y = mix(parts, sr) if parts else silence(len(steps) * bars * step, sr)

    elif kind == "sfx":
        y = S.render_sfx(layer.get("kind", "beep"), sr, **_params(layer))

    elif kind in ("speech", "say", "voice"):
        y = SPK.render_speech(layer.get("text"), sr, phonemes=layer.get("phonemes"), **_params(layer))

    elif kind in ("file", "sample"):
        y, fsr = read_wav(layer["path"])
        y = resample(y, fsr, sr)

    elif kind == "silence":
        y = silence(float(layer.get("dur", 0.5)) * q, sr)

    elif kind == "group":
        y = render_layers(layer.get("layers", []), sr, bpm)

    elif kind in LAYER_TYPES:
        y = LAYER_TYPES[kind](layer, sr, q)

    else:
        raise SpecError(f"unknown layer type {kind!r} (synth|seq|drum|pattern|sfx|speech|file|silence|group|{'|'.join(LAYER_TYPES)})")

    if "repeat" in layer:
        reps = int(layer["repeat"])
        every = float(layer["every"]) * q if "every" in layer else y.shape[0] / sr
        y = mix([(y, i * every) for i in range(reps)], sr)

    y = _fx(y, layer.get("fx"), sr)
    return y * gain, at


def render_sequence(layer: dict, sr: int, q: float = 1.0) -> np.ndarray:
    """`q` = seconds per time unit (60/bpm when a tempo is in effect, else 1)."""
    inst = layer.get("inst", "pluck")
    steps = layer.get("steps", layer.get("notes", "C4 E4 G4"))
    default_dur = float(layer.get("step", 0.25))
    if isinstance(steps, str):
        steps = parse_sequence(steps, default_dur)
    else:
        norm = []
        for s in steps:
            if isinstance(s, str):
                norm.extend(parse_sequence(s, default_dur))
            else:
                pitches = s.get("notes", s.get("note", s.get("pitches", [])))
                pitches = parse_pitch_list(pitches) if pitches not in ([], None, "-") else []
                norm.append({"pitches": pitches, "dur": float(s.get("dur", default_dur)), "vel": float(s.get("vel", 1.0))})
        steps = norm
    legato = float(layer.get("legato", 1.0))
    gap = float(layer.get("gap", 0.0)) * q
    strum = float(layer.get("strum", 0.0))
    vel = float(layer.get("vel", 1.0))
    params = _params(layer)
    transpose = float(layer.get("transpose", 0))
    parts = []
    t = 0.0
    for st in steps:
        if st["pitches"]:
            freqs = [f * 2 ** (transpose / 12) for f in st["pitches"]]
            y = I.render_chord(inst, freqs, st["dur"] * q * legato, sr, vel * st["vel"], strum=strum, **params)
            parts.append((y, t))
        t += st["dur"] * q + gap
    if not parts:
        return silence(t or 0.1, sr)
    return mix(parts, sr)


def render_layers(layers: list[dict], sr: int, bpm: float | None = None) -> np.ndarray:
    parts = [render_layer(l, sr, bpm) for l in layers]
    return mix(parts, sr)


def render_spec(spec: dict, sr: int | None = None) -> tuple[np.ndarray, int]:
    sr = int(sr or spec.get("sr", DEFAULT_SR))
    layers = spec.get("layers")
    if layers is None:
        # allow a bare single layer spec: {"type": "sfx", "kind": "beep", "out": ...}
        if "type" in spec:
            layers = [spec]
        else:
            raise SpecError("spec needs a 'layers' list (or be a single layer with 'type')")
    bpm = spec.get("bpm")
    y = render_layers(layers, sr, bpm)
    y = _fx(y, spec.get("fx"), sr)
    length = spec.get("duration")
    if spec.get("beats") and bpm:          # hard length in beats (4 bars of 4/4 = 16), plus `tail` seconds of ring-out
        length = float(spec["beats"]) * 60.0 / float(bpm) + float(spec.get("tail", 0.0))
    loop = spec.get("loop")
    fold = loop == "fold"
    if fold and not length:
        raise SpecError('"loop": "fold" needs a loop length: set "beats" (with "bpm") or "duration"')
    if length:
        n = samples(float(length) - (float(spec.get("tail", 0.0)) if fold and spec.get("beats") and bpm else 0.0), sr)
        if fold:
            # Musical loop: whatever rings past the loop point (release tails, reverb) is laid back over
            # the start, which is exactly what the next repetition would sound like. Sample-accurate length.
            from .core import pad_to
            out = pad_to(y[:n].copy(), n)
            for s in range(n, y.shape[0], n):
                seg = y[s:s + n]
                out[:seg.shape[0]] += seg
            y = out
        elif y.shape[0] > n:
            y = y[:n]
        else:
            from .core import pad_to
            y = pad_to(y, n)
    if fold:
        pass
    elif loop:
        # Loopable output: crossfade tail into head; trim/declick would break the seam.
        from .fx import loop as _loop
        y = _loop(y, sr, crossfade=float(loop) if not isinstance(loop, bool) else 0.05)
    else:
        if spec.get("trim", True):
            y = trim_tail(y, sr=sr)
        if spec.get("declick", True):
            y = declick(y, sr)
    lvl = spec.get("normalize", -1.0)
    if lvl is not None and lvl is not False:
        y = normalize(y, float(lvl))
    # Peak level says little about how loud a sound is heard: a clean tone at 2-4 kHz normalised to
    # full scale is 10+ dB louder to the ear than a drum hit with the same peak, and piercing.
    # Anything still above the ceiling after normalising is turned down to it.
    cap = spec.get("max_loudness", DEFAULT_MAX_LOUDNESS if lvl is not None and lvl is not False else None)
    if cap is not None and cap is not False:
        over = ear_loudness(y, sr) - float(cap)
        if over > 0:
            y = y * 10 ** (-over / 20)
    y = y * float(spec.get("gain", 1.0))
    return np.clip(y, -1, 1), sr


def render_to_file(spec: dict, out: str | Path | None = None, out_dir: str | Path | None = None, sr: int | None = None) -> Path:
    y, sr = render_spec(spec, sr)
    path = Path(out or spec.get("out") or "out.wav")
    if out_dir and not path.is_absolute():
        path = Path(out_dir) / path
    return write_wav(path, y, sr, bits=int(spec.get("bits", 16)))


def load_specs(source: str | Path | dict | list) -> list[dict]:
    """Accepts a path, a JSON string, a dict, or a list. Returns a list of sound specs.
    Batch files may be {"sounds": [...], "out_dir": "..."} or a plain list."""
    if isinstance(source, (str, Path)):
        s = str(source)
        if s.strip().startswith(("{", "[")):
            data = json.loads(s)
        else:
            data = json.loads(Path(s).read_text(encoding="utf-8"))
    else:
        data = source
    if isinstance(data, list):
        return data
    if "sounds" in data:
        out_dir = data.get("out_dir")
        specs = []
        for sp in data["sounds"]:
            sp = dict(sp)
            for k in ("sr", "fx", "normalize", "max_loudness", "bpm"):
                if k in data and k not in sp:
                    sp[k] = data[k]
            if out_dir and "out" in sp and not Path(sp["out"]).is_absolute():
                sp["out"] = str(Path(out_dir) / sp["out"])
            specs.append(sp)
        return specs
    return [data]


EXAMPLE = {
    "out": "sounds/example_stinger.wav",
    "sr": 44100,
    "layers": [
        {"type": "seq", "inst": "bell", "steps": "C5:0.12 E5:0.12 G5:0.12 C6:0.5", "gain": 0.8,
         "fx": [{"type": "delay", "time": 0.18, "feedback": 0.3, "mix": 0.25}]},
        {"type": "synth", "inst": "pad", "notes": "C4:maj", "dur": 0.9, "at": 0.0, "gain": 0.5},
        {"type": "drum", "kind": "kick", "at": 0.0, "gain": 0.7},
        {"type": "sfx", "kind": "whoosh", "at": 0.3, "params": {"dur": 0.4}, "gain": 0.4},
    ],
    "fx": [{"type": "reverb", "mix": 0.25, "size": 0.6}, {"type": "compressor"}],
    "normalize": -1.0,
}
