"""Bench: genny against recorded notes (TinySOL, CC BY 4.0, Zenodo 3685367) on three spectral descriptors.

    uv run python examples/bench_datasets.py                  # streams the first 250 MB of the archive, keeps 3 notes per instrument
    uv run python examples/bench_datasets.py --max-mb 1100    # the whole archive (1 GB), all instruments
    uv run python examples/bench_datasets.py --offline        # compares the notes already in research/datasets_bench/tinysol

Writes research/datasets_bench/report.md. Other sets to bench the same way (not downloaded here, see research/datasets_bench/README.md):
NSynth (https://magenta.tensorflow.org/datasets/nsynth, 30 GB), MP-RIR (https://zenodo.org/records/6857203), VocalSet
(https://zenodo.org/records/1193957, 2.7 GB), InsectSet32 (https://zenodo.org/records/7072196), Dresden VT (https://www.vocaltractlab.de/),
Mores guitars (https://zenodo.org/search?q=Mores%20guitar).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from genny import instruments as I  # noqa: E402
from genny.notes import note_to_midi  # noqa: E402

URL = "https://zenodo.org/api/records/3685367/files/TinySOL.tar.gz/content"
META = "https://zenodo.org/api/records/3685367/files/TinySOL_metadata.csv/content"
OUT = ROOT / "research" / "datasets_bench"
SR = 44100


class _Limit(Exception):
    pass
GENNY = {"Fl": "flute", "Ob": "oboe", "ClBb": "clarinet", "Bn": "bassoon", "ASax": "sax", "BTb": "tuba", "Hn": "french_horn",
         "TpC": "trumpet", "Tbn": "trombone", "Acc": "accordion", "Cb": "upright_bass", "Vn": "violin", "Va": "viola", "Vc": "cello"}


def fetch_meta():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "TinySOL_metadata.csv"
    if not p.exists():
        urllib.request.urlretrieve(META, p)
    with open(p, newline="", encoding="utf-8") as f:
        return {Path(r["Path"]).name: r for r in csv.DictReader(f)}


def download(meta, max_mb, per_inst):
    dest = OUT / "tinysol"
    dest.mkdir(exist_ok=True)
    kept, got = {}, 0
    req = urllib.request.Request(URL, headers={"User-Agent": "genny-bench"})
    with urllib.request.urlopen(req) as resp:
        class Counter:
            def read(self, n):
                nonlocal got
                b = resp.read(n)
                got += len(b)
                if got > max_mb * 2 ** 20:
                    raise _Limit
                return b
        try:
            with tarfile.open(fileobj=Counter(), mode="r|gz") as tf:
                for m in tf:
                    r = meta.get(Path(m.name).name)
                    if not m.isfile() or r is None or r["Technique (abbr.)"] != "ord" or r["Dynamics"] != "mf":
                        continue
                    ab = r["Instrument (abbr.)"]
                    if kept.get(ab, 0) >= per_inst:
                        continue
                    (dest / Path(m.name).name).write_bytes(tf.extractfile(m).read())
                    kept[ab] = kept.get(ab, 0) + 1
        except (_Limit, RuntimeError, EOFError, tarfile.ReadError):
            pass
    return kept, got / 2 ** 20


def descriptors(y, sr, f_nom):
    y = np.asarray(y, float)
    y = y.mean(1) if y.ndim == 2 else y
    pk = np.argmax(np.abs(y) > 0.1 * np.abs(y).max())
    y = y[pk + int(0.05 * sr): pk + int(0.65 * sr)]
    if len(y) < 2048:
        return None
    s = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1.0 / sr)
    cen = float(np.sum(f * s) / np.sum(s))
    m = (f >= 100) & (f <= 4000)
    sl = float(np.polyfit(np.log2(f[m]), 10 * np.log10(s[m] + 1e-20), 1)[0])
    lo, hi = f_nom * 0.8, f_nom * 1.25
    w = (f >= lo) & (f <= hi)
    f0 = float(f[w][np.argmax(s[w])])
    return cen, sl, 1200.0 * np.log2(f0 / f_nom)


def compare(meta):
    rows = []
    for p in sorted((OUT / "tinysol").glob("*.wav")):
        r = meta.get(p.name)
        if r is None or r["Instrument (abbr.)"] not in GENNY:
            continue
        name = GENNY[r["Instrument (abbr.)"]]
        midi = note_to_midi(r["Pitch"])
        f_nom = 440.0 * 2 ** ((midi - 69) / 12.0)
        y, sr = sf.read(p)
        ref = descriptors(y, sr, f_nom)
        try:
            g = I.REGISTRY[name]["fn"](f_nom, 1.0, SR, 0.7)
        except Exception as e:
            rows.append((p.name, name, "error: " + str(e)[:40]))
            continue
        mine = descriptors(g, SR, f_nom)
        if ref and mine:
            rows.append((p.name, name, ref, mine))
    return rows


def report(rows, kept, mb):
    L = ["# Bench genny vs TinySOL", "", f"Downloaded: {sum(kept.values())} notes (ordinario, mf), {mb:.0f} MB streamed of 1026 MB. "
         "Descriptors on 0.6 s after the attack: spectral centroid (Hz), tilt (dB per octave, 100 Hz-4 kHz), f0 error (cents).", "",
         "| note | genny | centroid ref / genny | tilt ref / genny | f0 error ref / genny |", "|---|---|---|---|---|"]
    for r in rows:
        if len(r) == 3:
            L.append(f"| {r[0]} | {r[1]} | {r[2]} | | |")
            continue
        a, b = r[2], r[3]
        L.append(f"| {r[0]} | {r[1]} | {a[0]:.0f} / {b[0]:.0f} | {a[1]:.1f} / {b[1]:.1f} | {a[2]:+.0f} / {b[2]:+.0f} |")
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-mb", type=float, default=250.0)
    ap.add_argument("--per-instrument", type=int, default=3)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    meta = fetch_meta()
    kept, mb = ({}, 0.0) if a.offline else download(meta, a.max_mb, a.per_instrument)
    if a.offline:
        kept = {k: 1 for k in (meta[p.name]["Instrument (abbr.)"] for p in (OUT / "tinysol").glob("*.wav") if p.name in meta)}
    report(compare(meta), kept, mb)


if __name__ == "__main__":
    main()
