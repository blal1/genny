"""Renders one short audition phrase per instrument (and one file per drum) so the whole
palette can be listened through:  sounds/audition/instruments/<name>.wav, sounds/audition/drums/<name>.wav

Run:  uv run python examples/audition.py [--out DIR] [--only NAME ...]

Each phrase climbs through the instrument's usual register and ends on a chord (or a long
note for basses), dry, so what you hear is the instrument itself. The phrases deliberately
reach the top of the range: that is where a voice turns piercing if it is going to.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from genny import drums as D  # noqa: E402
from genny import instruments as I  # noqa: E402
from genny import spec as SP  # noqa: E402

# register -> (run of notes, final chord / long note)
RUNS = {
    "bass": ("C2 G2 C3 E3 G3 C4", "C2"),
    "low": ("C3 E3 G3 C4 E4 G4 C5", "[C3,G3,C4,E4]"),
    "mid": ("C4 E4 G4 C5 E5 G5 C6", "[C4,E4,G4,C5]"),
    "high": ("C5 E5 G5 C6 E6 G6 C7", "[C5,E5,G5,C6]"),
}
# anything not listed plays in the "mid" register with short notes
REGISTER = {
    "bass": "bass", "sub": "bass", "wobble": "bass", "upright_bass": "bass", "finger_bass": "bass", "slap_bass": "bass",
    "acid": "bass", "reese": "bass", "fm_bass": "bass", "bass808": "bass", "tuba": "bass", "didgeridoo": "bass",
    "timpani": "bass", "chiptri": "low",
    "cello": "low", "trombone": "low", "french_horn": "low", "dist_guitar": "low", "muted_guitar": "low",
    "guitar": "low", "steel_guitar": "low", "electric_guitar": "low", "handpan": "low", "tubular_bell": "low",
    "singing_bowl": "low", "warm_pad": "low", "pad": "low", "church_organ": "low", "sax": "low", "sitar": "low",
    "bell": "high", "glass": "high", "music_box": "high", "glockenspiel": "high", "celesta": "high", "kalimba": "high",
    "xylophone": "high", "whistle": "high", "ocarina": "high", "chip": "high", "square": "high", "sine": "high",
    "flute": "high", "pan_flute": "high", "violin": "high", "marimba": "high", "vibraphone": "high", "harp": "high",
    "mandolin": "high", "koto": "high", "glass_pad": "high", "steel_drum": "high", "gamelan": "high", "toy_piano": "high",
}
# sustained voices get longer notes so the attack and vibrato can be heard
SUSTAINED = {"pad", "warm_pad", "glass_pad", "strings", "string_machine", "choir", "violin", "cello", "flute", "pan_flute",
             "clarinet", "oboe", "sax", "trumpet", "french_horn", "trombone", "tuba", "harmonica", "accordion", "organ",
             "church_organ", "whistle", "ocarina", "theremin", "didgeridoo", "brass", "lead", "soft_lead", "square_lead",
             "pwm", "singing_bowl", "reese", "wobble", "dist_guitar"}


def phrase(name: str) -> dict:
    reg = REGISTER.get(name, "mid")
    run, last = RUNS[reg]
    step = 0.34 if name in SUSTAINED else 0.22
    hold = 1.6 if name in SUSTAINED else 1.1
    steps = " ".join(f"{n}:{step}" for n in run.split()) + f" -:0.12 {last}:{hold}"
    return {"out": f"instruments/{name}.wav", "normalize": -3.0,
            "layers": [{"type": "seq", "inst": name, "steps": steps, "vel": 0.85, "legato": 0.92}]}


def drum_hit(name: str) -> dict:
    return {"out": f"drums/{name}.wav", "normalize": -3.0,
            "layers": [{"type": "pattern", "kind": name, "hits": [0.0, 0.45, 0.675, 0.9]}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "sounds" / "audition"))
    ap.add_argument("--only", nargs="*", help="instrument / drum names to render (default: all)")
    args = ap.parse_args()
    specs = [phrase(n) for n in I.REGISTRY] + [drum_hit(n) for n in D.REGISTRY]
    if args.only:
        specs = [s for s in specs if Path(s["out"]).stem in args.only]
    for s in specs:
        SP.render_to_file(s, out_dir=args.out)
    print(f"{len(specs)} files in {args.out}")


if __name__ == "__main__":
    main()
