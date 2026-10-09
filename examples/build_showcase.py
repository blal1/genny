"""Writes and renders a short tune for every instrument, a groove for every drum and a generated
piece for every music style, all in genny's own notation (bpm, beats, step strings).

    uv run python examples/build_showcase.py            # write the JSON batches and render them
    uv run python examples/build_showcase.py --no-render

Batches (edit and re-render with `genny render <file>`):
  examples/library/instrument_tunes.json   -> sounds/instrument_tunes/<instrument>.wav
  examples/library/drum_grooves.json       -> sounds/drum_grooves/<drum>.wav
  examples/library/style_pieces.json       -> sounds/style_pieces/<style>.wav   (the `arrangement` layer)
The hand-written loop of each style is in examples/library/styles.json (examples/build_styles.py).

Every tune is written from the instrument's own register (`range` in the catalog), so new instruments
are picked up without touching this file.
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import genny  # noqa: E402,F401  (registers every module)
from genny import compose as C  # noqa: E402
from genny import drums as D  # noqa: E402
from genny import instruments as I  # noqa: E402
from genny import spec as SP  # noqa: E402
from genny.notes import midi_to_name, note_to_midi  # noqa: E402

LIB = ROOT / "examples" / "library"
SCALES = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10], "dorian": [0, 2, 3, 5, 7, 9, 10]}
# four-bar melodies as (scale degree, beats); degree 1 = tonic, 8 = octave, 0 = the note under the tonic
MELODIES = [
    [(1, 1), (3, .5), (5, .5), (8, 1), (5, 1), (6, .5), (5, .5), (3, 1), (2, .5), (3, .5), (1, 1),
     (4, .5), (6, .5), (8, 1), (7, .5), (5, .5), (3, 1), (2, .5), (3, .5), (2, .5), (0, .5), (1, 2)],
    [(5, .5), (6, .5), (8, 1), (5, 1), (3, 1), (4, .5), (3, .5), (2, 1), (5, 2),
     (3, .5), (4, .5), (5, 1), (8, 1), (6, 1), (5, .5), (4, .5), (3, .5), (2, .5), (1, 2)],
    [(1, 1.5), (2, .5), (3, 1), (5, 1), (4, 1.5), (3, .5), (2, 2),
     (3, 1.5), (5, .5), (8, 1), (6, 1), (5, 1), (2, 1), (1, 2)],
]
BASS = [(1, 1), (1, .5), (5, .5), (8, 1), (5, 1), (6, 1), (6, .5), (3, .5), (4, 1), (5, 1),
        (1, 1), (3, .5), (5, .5), (6, 1), (5, 1), (4, .5), (5, .5), (4, .5), (2, .5), (1, 2)]
CHORDS = [((1, 3, 5), 4), ((6, 8, 10), 4), ((4, 6, 8), 4), ((5, 7, 9), 2), ((1, 3, 5), 2)]   # I vi IV V I
FAMILY = {  # family -> (bpm, legato, note-length scale)
    "bass": (96, 0.9, 1.0), "pad": (66, 1.0, 1.0), "bowed": (76, 0.98, 1.0), "wind": (88, 0.95, 1.0),
    "brass": (92, 0.92, 1.0), "voice": (72, 1.0, 1.0), "mallet": (104, 1.0, 1.0), "plucked": (96, 1.0, 1.0),
    "keys": (100, 0.95, 1.0), "retro": (132, 0.85, 1.0), "strings": (84, 0.98, 1.0),
}


def _semi(degree: int, scale: list[int]) -> int:
    d = degree - 1
    return 12 * (d // 7) + scale[d % 7]


def _fit(m: int, lo: int, hi: int) -> int:
    while m > hi:
        m -= 12
    while m < lo:
        m += 12
    return m


def tune(name: str) -> dict:
    e = I.REGISTRY[name]
    lo, hi = (note_to_midi(n) for n in e["range"].split("-"))
    h = zlib.crc32(name.encode())
    family = e.get("family", "synth")
    bpm, legato, _ = FAMILY.get(family, (104, 0.95, 1.0))
    scale = SCALES[("major", "minor", "dorian")[h % 3]]
    tonic = int(min(max(round((lo + hi) / 2 - 6), lo + 1), max(lo + 1, hi - 12))) if hi - lo >= 14 else lo
    note = lambda deg: midi_to_name(_fit(tonic + _semi(deg, scale), lo, hi))
    if family == "pad":
        steps = " ".join("[" + ",".join(note(d) for d in degs) + f"]:{beats}" for degs, beats in CHORDS)
    else:
        line = BASS if family == "bass" else MELODIES[(h >> 3) % len(MELODIES)]
        steps = " ".join(f"{note(d)}:{b:g}" for d, b in line)
    layer = {"type": "seq", "inst": name, "step": 1, "steps": steps, "vel": 0.85, "legato": legato}
    return {"out": f"{name}.wav", "bpm": bpm, "tail": 2.0, "normalize": -3.0, "layers": [layer],
            "fx": [{"type": "reverb", "mix": 0.14, "size": 0.5}]}


def groove(name: str) -> dict:
    """Two bars: the drum on its own first, then against a soft kick and hat so its role is clear."""
    pat = ("x--x--x-x--x-x--", "x-x-x-xxx-x-x-xX", "x---x---x---x-x-")[zlib.crc32(name.encode()) % 3]
    layers = [{"type": "pattern", "kind": name, "steps": pat, "bars": 2, "gain": 0.9}]
    if name not in ("kick", "hihat"):
        layers += [{"type": "pattern", "kind": "kick", "steps": "x-------x-------", "at": 4, "gain": 0.6},
                   {"type": "pattern", "kind": "hihat", "steps": "--x---x---x---x-", "at": 4, "gain": 0.2}]
    return {"out": f"{name}.wav", "bpm": 100, "beats": 8, "tail": 2.5, "normalize": -3.0, "layers": layers}


def style_piece(style: str, i: int) -> dict:
    return {"out": f"{style}.wav", "normalize": -2.0, "tail": 2.0,
            "layers": [{"type": "arrangement", "style": style, "bars": 8, "seed": 100 + i}]}


def main():
    batches = {
        "instrument_tunes": [tune(n) for n in I.REGISTRY],
        "drum_grooves": [groove(n) for n in D.REGISTRY],
        "style_pieces": [style_piece(s, i) for i, s in enumerate(C.STYLES)],
    }
    for name, sounds in batches.items():
        path = LIB / f"{name}.json"
        path.write_text(json.dumps({"out_dir": f"sounds/{name}", "sounds": sounds}, indent=1), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)} ({len(sounds)} sounds)")
    if "--no-render" in sys.argv:
        return
    failed = 0
    for name in batches:
        for sp in SP.load_specs(LIB / f"{name}.json"):
            try:
                SP.render_to_file(sp, out_dir=ROOT)
            except Exception as e:   # keep going, report at the end
                failed += 1
                print(f"ERROR {sp.get('out')}: {e}", file=sys.stderr)
    print(f"rendered {sum(map(len, batches.values())) - failed}, failed {failed}")


if __name__ == "__main__":
    main()
