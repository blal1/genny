"""genny command line interface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from . import drums as D
from . import fx as FX
from . import instruments as I
from . import osc as O
from . import sfx as S
from . import spec as SP
from . import speech as SPK
from .core import DEFAULT_SR, read_wav, write_wav
from .notes import CHORDS, SCALES


def _kv(items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        for kv in it.split(","):
            if not kv.strip():
                continue
            k, _, v = kv.partition("=")
            out[k.strip()] = FX._coerce(v.strip())
    return out


def _fx_chain(items: list[str] | None) -> list[dict]:
    return [FX.parse_fx_arg(s) for s in (items or [])]


def _finish(args, layer: dict) -> Path:
    spec = {"layers": [layer], "fx": _fx_chain(args.fx), "sr": args.sr, "normalize": args.normalize, "out": args.out}
    if getattr(args, "no_trim", False):
        spec["trim"] = False
    path = SP.render_to_file(spec)
    _report(path)
    return path


def _report(path: Path):
    y, sr = read_wav(path)
    print(f"wrote {path}  ({y.shape[0] / sr:.2f}s, {'stereo' if y.ndim == 2 else 'mono'}, {sr} Hz)")


def cmd_synth(args):
    layer = {"type": "synth", "inst": args.inst, "notes": args.notes, "dur": args.dur, "params": _kv(args.p),
             "vel": args.vel, "strum": args.strum}
    _finish(args, layer)


def cmd_seq(args):
    layer = {"type": "seq", "inst": args.inst, "steps": args.steps, "params": _kv(args.p), "vel": args.vel,
             "legato": args.legato, "gap": args.gap, "step": args.step, "transpose": args.transpose}
    if args.bpm:
        layer["bpm"] = args.bpm
    _finish(args, layer)


def cmd_drum(args):
    layer = {"type": "drum", "kind": args.kind, "params": _kv(args.p), "vel": args.vel}
    _finish(args, layer)


def cmd_sfx(args):
    layer = {"type": "sfx", "kind": args.kind, "params": _kv(args.p)}
    _finish(args, layer)


def cmd_say(args):
    params = _kv(args.p)
    if args.show:
        lang = str(params.get("lang", "en"))
        print(SPK.to_phonemes(None if args.phonemes else args.text, lang, phonemes=args.text if args.phonemes else None))
    layer = {"type": "speech", "params": params}
    layer["phonemes" if args.phonemes else "text"] = args.text
    _finish(args, layer)


def cmd_fx(args):
    y, sr = read_wav(args.input)
    y = FX.apply_chain(y, _fx_chain(args.fx), sr)
    if args.normalize is not None:
        from .core import normalize
        y = normalize(y, args.normalize)
    write_wav(args.out, np.clip(y, -1, 1), sr)
    _report(Path(args.out))


def cmd_render(args):
    specs: list[dict] = []
    if args.json:
        specs += SP.load_specs(args.json)
    for src in args.specs:
        if src == "-":
            specs += SP.load_specs(sys.stdin.read())
        else:
            specs += SP.load_specs(src)
    if not specs:
        print("no specs given (pass files, '-' for stdin, or --json)", file=sys.stderr)
        sys.exit(2)
    ok, failed = 0, 0
    for i, sp in enumerate(specs):
        try:
            path = SP.render_to_file(sp, out_dir=args.out_dir, sr=args.sr)
            if not args.quiet:
                _report(path)
            ok += 1
        except Exception as e:  # keep going in batches, report at the end
            failed += 1
            print(f"ERROR in spec #{i} ({sp.get('out', '?')}): {e}", file=sys.stderr)
            if args.strict:
                raise
    if len(specs) > 1:
        print(f"{ok} rendered, {failed} failed")
    if failed:
        sys.exit(1)


def _catalog() -> dict:
    def table(reg):
        out = {}
        for k, v in reg.items():
            out[k] = {"desc": v["desc"], "params": {p: {"default": d, "help": h} for p, (d, h) in v["params"].items()}}
            for extra in ("family", "range"):   # instruments: what kind of voice, and the register it is voiced for
                if extra in v:
                    out[k][extra] = v[extra]
        return out
    return {
        "instruments": table(I.REGISTRY),
        "drums": table(D.REGISTRY),
        "sfx": table(S.REGISTRY),
        "fx": table(FX.REGISTRY),
        "speech": table({"speech": {"desc": SPK.SPEECH_DESC, "params": SPK.PARAMS}}),
        "layers": dict(sorted(SP.LAYER_HELP.items())),
        "phonemes": SPK.PHONEMES,
        "waves": O.WAVES,
        "scales": list(SCALES),
        "chords": list(CHORDS),
    }


def cmd_list(args):
    cat = _catalog()
    cat["words"] = SPK.WORDS
    what = args.what
    if what == "all":  # the word list is long; only on request
        cat.pop("words")
    if args.json:
        print(json.dumps(cat if what == "all" else cat[what], indent=2))
        return
    sections = list(cat) if what == "all" else [what]
    for sec in sections:
        print(f"== {sec} ==")
        entries = cat[sec]
        if isinstance(entries, list):
            print("  " + ", ".join(entries))
        else:
            for name, e in entries.items():
                if isinstance(e, str):          # layer types: name -> one-line description
                    print(f"  {name:<12} {e}")
                    continue
                ps = ", ".join(f"{p}={v['default']!r}" for p, v in e["params"].items())
                tag = f"[{e['family']}, {e['range']}] " if "range" in e else ""
                print(f"  {name:<12} {tag}{e['desc']}")
                if ps:
                    print(f"  {'':<12}   params: {ps}")
        print()


def cmd_info(args):
    from .analysis import describe
    out = []
    for f in args.files:
        y, sr = read_wav(f)
        d = describe(y, sr)
        d["file"] = str(f)
        out.append(d)
        if not args.json:
            print(f"{f}: {d['seconds']:.3f}s {'stereo' if d['channels'] == 2 else 'mono'} {sr}Hz peak={d['peak_db']:.1f}dB rms={d['rms_db']:.1f}dB "
                  f"loud={d['loudness']:.1f} ear={d['ear']:.1f} sharp={d['sharpness']:.2f} above5k={100 * d['above_5k']:.1f}%")
    if args.json:
        print(json.dumps(out, indent=2))


def cmd_identify(args):
    from .identify import identify_machine
    out = []
    for f in args.files:
        y, sr = read_wav(f)
        out.append({"file": str(f), **identify_machine(y, sr, rpm=args.rpm)})
    print(json.dumps(out, indent=2))


def cmd_play(args):
    import platform
    import shutil
    import subprocess
    for f in args.files:
        if platform.system() == "Windows":
            import winsound
            winsound.PlaySound(str(f), winsound.SND_FILENAME)
        elif shutil.which("afplay"):
            subprocess.run(["afplay", str(f)])
        elif shutil.which("ffplay"):
            subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(f)])
        elif shutil.which("aplay"):
            subprocess.run(["aplay", "-q", str(f)])
        else:
            print("no player found (need winsound/afplay/ffplay/aplay)", file=sys.stderr)
            sys.exit(1)


def cmd_example(args):
    print(json.dumps(SP.EXAMPLE, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="genny", description="Procedural sound generator for games. See AGENTS.md for the full spec format.")
    p.add_argument("--version", action="version", version=f"genny {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, default_out):
        sp.add_argument("-o", "--out", default=default_out, help="output .wav path")
        sp.add_argument("--fx", action="append", metavar="NAME[:k=v,k=v]", help="effect to apply after the sound (repeatable, in order)")
        sp.add_argument("--sr", type=int, default=DEFAULT_SR, help="sample rate (default 44100)")
        sp.add_argument("--normalize", type=float, default=-1.0, help="peak dB target (default -1)")
        sp.add_argument("--no-trim", action="store_true", help="keep silent tail")

    s = sub.add_parser("synth", help="play one or more notes (a chord) with an instrument")
    s.add_argument("inst", help="instrument name (genny list instruments)")
    s.add_argument("notes", help="'C4' | 'C4 E4 G4' | 'C4:maj' | '440hz' | MIDI number")
    s.add_argument("--dur", type=float, default=0.5, help="note length seconds (release is added after)")
    s.add_argument("--vel", type=float, default=1.0, help="velocity 0..1")
    s.add_argument("--strum", type=float, default=0.0, help="seconds between chord notes")
    s.add_argument("-p", action="append", metavar="k=v", help="instrument params, e.g. -p decay=2 -p ratio=3.5")
    common(s, "synth.wav")
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("seq", help="play a melody/stinger: 'C4:0.1 E4:0.1 G4:0.3 [C5,E5]:0.5 -:0.1'")
    s.add_argument("inst")
    s.add_argument("steps", help="step notation (see AGENTS.md)")
    s.add_argument("--step", type=float, default=0.25, help="default step duration when a token has none")
    s.add_argument("--legato", type=float, default=1.0, help="note length as fraction of step (0.5 = staccato)")
    s.add_argument("--gap", type=float, default=0.0, help="extra silence between steps")
    s.add_argument("--transpose", type=float, default=0.0, help="semitones")
    s.add_argument("--bpm", type=float, default=None, help="tempo: step durations, --step and --gap are then in beats")
    s.add_argument("--vel", type=float, default=1.0)
    s.add_argument("-p", action="append", metavar="k=v")
    common(s, "seq.wav")
    s.set_defaults(fn=cmd_seq)

    s = sub.add_parser("drum", help="one drum hit")
    s.add_argument("kind")
    s.add_argument("--vel", type=float, default=1.0)
    s.add_argument("-p", action="append", metavar="k=v")
    common(s, "drum.wav")
    s.set_defaults(fn=cmd_drum)

    s = sub.add_parser("sfx", help="one sound effect")
    s.add_argument("kind")
    s.add_argument("-p", action="append", metavar="k=v", help="e.g. -p freq=880 -p dur=0.2")
    common(s, "sfx.wav")
    s.set_defaults(fn=cmd_sfx)

    s = sub.add_parser("say", help="speak text: letters, numbers, short words ('/JH IY1/' = raw phonemes)")
    s.add_argument("text", help="text to speak, e.g. 'G', 'fuel low', 'level up!'; with --phonemes: ARPAbet")
    s.add_argument("--phonemes", action="store_true", help="treat TEXT as raw ARPAbet phonemes")
    s.add_argument("--show", action="store_true", help="print the phoneme transcription")
    s.add_argument("-p", action="append", metavar="k=v", help="e.g. -p voice=female -p rate=1.2 -p lang=es")
    common(s, "say.wav")
    s.set_defaults(fn=cmd_say)

    s = sub.add_parser("fx", help="apply effects to an existing wav")
    s.add_argument("input")
    s.add_argument("-o", "--out", required=True)
    s.add_argument("--fx", action="append", required=True, metavar="NAME[:k=v]")
    s.add_argument("--normalize", type=float, default=None)
    s.set_defaults(fn=cmd_fx)

    s = sub.add_parser("render", help="render JSON spec file(s) or batches (the main AI interface)")
    s.add_argument("specs", nargs="*", help="spec .json files, or '-' for stdin")
    s.add_argument("--json", help="inline JSON spec string")
    s.add_argument("--out-dir", help="prefix for relative 'out' paths")
    s.add_argument("--sr", type=int, default=None)
    s.add_argument("--quiet", "-q", action="store_true")
    s.add_argument("--strict", action="store_true", help="stop at first error")
    s.set_defaults(fn=cmd_render)

    s = sub.add_parser("list", help="list instruments, drums, sfx, fx, extra layer types and their params")
    s.add_argument("what", nargs="?", default="all", choices=["all", "instruments", "drums", "sfx", "fx", "layers", "speech", "phonemes", "words", "waves", "scales", "chords"])
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("info", help="print duration, peak/rms, loudness and brightness (sharpness, energy above 5 kHz) of wav files")
    s.add_argument("files", nargs="+")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("identify", help="name the machine in a steady recording (transformer, gearbox, bearing fault, engine) and the sfx params that reproduce it")
    s.add_argument("files", nargs="+")
    s.add_argument("--rpm", type=float, default=None, help="shaft speed, if known")
    s.set_defaults(fn=cmd_identify)

    s = sub.add_parser("play", help="play wav files through the system player")
    s.add_argument("files", nargs="+")
    s.set_defaults(fn=cmd_play)

    s = sub.add_parser("example", help="print an example JSON spec")
    s.set_defaults(fn=cmd_example)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.fn(args)
    except (ValueError, KeyError, FileNotFoundError, SP.SpecError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
