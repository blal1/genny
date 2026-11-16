"""Measures the arms sounds (firearm, minigun, bullet_pass, bullet_impact, casings, ricochet) and writes the numbers and WAVs.

    uv run python examples/arms_measure.py --label avant --json out.json --md out.md --wav-dir DIR
    uv run python examples/arms_measure.py --label apres --json after.json --before before.json --md after.md
    uv run python examples/arms_measure.py --only impact,casing --quiet

Per sound: peak and rms (dBFS), loudest 100 ms rms, length above -40 dB re peak, spectral centroid and peak, T0 (main peak to the
next zero crossing), and the levels of the 1/3-octave bands (dB re full scale, mean square, so the bands add up to the rms).
Groups: firearm (33 weapons, 10 m / 30 deg), axis3 (bare pressure on axis at 3 m), minigun, pass, impact, casing, ricochet.
With --rico-out the 24 ricochets are also saved as an npz, and --rico-ref compares them bit for bit with an earlier npz.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from genny import arms as A  # noqa: E402
from genny.core import DEFAULT_SR, write_wav  # noqa: E402
from genny.sfx import render_sfx  # noqa: E402

BANDS = [1000.0 * 10.0 ** (0.1 * n) for n in range(-16, 13)]     # 25 Hz .. 16 kHz
IMPACT_MATERIALS = ["metal", "metal_thin", "concrete", "wood", "flesh", "flesh_armor", "water", "dirt", "glass"]
IMPACT_CALIBERS = ["22lr", "9x19", "556", "762x51", "50bmg"]
FLOORS = ["concrete", "wood"]
GROUPS = ["firearm", "axis3", "minigun", "pass", "impact", "casing", "ricochet"]
REFERENCE = {"firearm": "firearm/ar15", "axis3": "axis3/ar15", "pass": "pass/556", "impact": "impact/metal_556",
             "casing": "casing/ricochet_556_concrete", "ricochet": "ricochet/metal_whine_0"}


def cases(only: set[str]) -> list[tuple[str, str, str, dict]]:
    """(group, name, sfx kind, params)."""
    out = []
    if "firearm" in only:
        out += [("firearm", w, "firearm", {"weapon": w}) for w in A.WEAPONS]
    if "axis3" in only:
        out += [("axis3", w, "firearm", {"weapon": w, "distance": 3.0, "azimuth": 0.0, "render": "pressure"}) for w in A.WEAPONS]
    if "minigun" in only:
        out += [("minigun", "default", "minigun", {}), ("minigun", "rate100", "minigun", {"rate": 100.0, "dur": 2.0}),
                ("minigun", "motor_only", "minigun", {"fire": False, "dur": 2.0})]
    if "pass" in only:
        out += [("pass", c, "bullet_pass", {"caliber": c}) for c in A.CALIBERS]
    if "impact" in only:
        out += [("impact", f"{m}_{c}", "bullet_impact", {"material": m, "caliber": c}) for m in IMPACT_MATERIALS for c in IMPACT_CALIBERS]
    if "casing" in only:
        out += [("casing", f"ricochet_{c}_{f}", "casing_ricochet", {"caliber": c, "floor": f}) for c in A.CASES for f in FLOORS]
        out += [("casing", f"physical_{c}_concrete", "casing", {"caliber": c}) for c in A.CASES]
        out += [("casing", "generic", "casing", {}), ("casing", "generic_ricochet", "casing", {"ricochet": True})]
    if "ricochet" in only:
        out += [("ricochet", f"{s}_{v}_{k}", "ricochet", {"surface": s, "variant": v, "seed": k})
                for s in ("metal", "stone", "water") for v in ("whine", "short") for k in range(4)]
    return out


def mono(y) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    return y.mean(axis=1) if y.ndim == 2 else y


def db(x: float) -> float:
    return float(20.0 * np.log10(max(x, 1e-12)))


def measure(y: np.ndarray, sr: int) -> dict:
    y = mono(y)
    n = len(y)
    pk = float(np.max(np.abs(y))) + 1e-20
    c2 = np.concatenate([[0.0], np.cumsum(y * y)])
    w = max(1, int(0.1 * sr))
    seg = (c2[w:] - c2[:-w]) / w if n > w else np.array([c2[-1] / max(n, 1)])
    above = np.flatnonzero(np.abs(y) > pk * 0.01)
    spec = np.fft.rfft(y)
    p = np.abs(spec) ** 2 / n ** 2
    p[1:-1 if n % 2 == 0 else None] *= 2.0
    f = np.fft.rfftfreq(n, 1.0 / sr)
    tot = float(np.sum(p)) + 1e-30
    k = int(np.argmax(np.abs(y)))
    s = np.sign(y[k])
    z = np.flatnonzero(np.sign(y[k + 1:]) != s) if k + 1 < n else np.array([])
    third = []
    for fc in BANDS:
        sel = (f >= fc / 2 ** (1 / 6)) & (f < fc * 2 ** (1 / 6))
        third.append(round(float(10.0 * np.log10(np.sum(p[sel]) + 1e-20)), 1))
    return {"n": n, "dur_s": round(n / sr, 4), "peak": round(pk, 6), "peak_db": round(db(pk), 2),
            "rms_db": round(10.0 * np.log10(c2[-1] / n + 1e-20), 2),
            "rms100_db": round(10.0 * np.log10(float(np.max(seg)) + 1e-20), 2),
            "dur40_ms": round(1e3 * (above[-1] - above[0]) / sr, 1) if len(above) else 0.0,
            "centroid_hz": round(float(np.sum(f * p) / tot), 1),
            "fpeak_hz": round(float(f[1 + np.argmax(p[1:])]), 1) if len(p) > 1 else 0.0,
            "t0_ms": round(1e3 * (z[0] + 1) / sr, 3) if len(z) else None,
            "nan": bool(not np.all(np.isfinite(y))), "third_db": third}


def run(only: set[str], wav_dir: Path | None, quiet: bool) -> dict:
    res = {}
    for g, name, kind, params in cases(only):
        t = time.time()
        key = f"{g}/{name}"
        try:
            y = render_sfx(kind, DEFAULT_SR, **params)
            m = measure(y, DEFAULT_SR)
            m.update(kind=kind, params=params)
            if wav_dir is not None:
                d = wav_dir / g
                d.mkdir(parents=True, exist_ok=True)
                write_wav(d / f"{name}.wav", np.asarray(y), DEFAULT_SR)
        except Exception as e:   # one bad case must not hide the others
            m = {"error": repr(e)[:200], "kind": kind, "params": params}
        m["render_s"] = round(time.time() - t, 2)
        res[key] = m
        if not quiet:
            print(key, m.get("peak_db", m.get("error")), f"{m['render_s']}s", flush=True)
    return res


def table(res: dict, before: dict | None) -> str:
    L = []
    for g in GROUPS:
        rows = [(k, v) for k, v in res.items() if k.startswith(g + "/")]
        if not rows:
            continue
        ref = res.get(REFERENCE.get(g, ""))
        L.append(f"\n## {g}\n")
        head = "| sound | peak dBFS | rms dBFS | rms100 dBFS | >-40 dB ms | centroid Hz | f peak Hz | T0 ms | rms vs ref dB |"
        sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        if before:
            head += " d peak | d rms | centroid x | dur x |"
            sep += " ---: | ---: | ---: | ---: |"
        L += [head, sep]
        for k, v in rows:
            if "error" in v:
                L.append(f"| {k.split('/', 1)[1]} | ERROR {v['error']} |")
                continue
            rel = f"{v['rms_db'] - ref['rms_db']:+.1f}" if ref and "rms_db" in ref else ""
            line = (f"| {k.split('/', 1)[1]} | {v['peak_db']:.1f} | {v['rms_db']:.1f} | {v['rms100_db']:.1f} | {v['dur40_ms']:.0f} | "
                    f"{v['centroid_hz']:.0f} | {v['fpeak_hz']:.0f} | {'' if v['t0_ms'] is None else v['t0_ms']} | {rel} |")
            if before:
                b = before.get(k)
                if b and "rms_db" in b:
                    line += (f" {v['peak_db'] - b['peak_db']:+.1f} | {v['rms_db'] - b['rms_db']:+.1f} | "
                             f"{v['centroid_hz'] / max(b['centroid_hz'], 1e-9):.2f} | {v['dur40_ms'] / max(b['dur40_ms'], 1e-9):.2f} |")
                else:
                    line += " | | | |"
            L.append(line)
    L.append("\n## 1/3-octave bands (dB re full scale, mean square; bands add up to the rms)\n")
    L.append("Centres in Hz: " + " ".join(f"{b:.0f}" for b in BANDS) + "\n")
    for g in GROUPS:
        rows = [(k, v) for k, v in res.items() if k.startswith(g + "/") and "third_db" in v]
        if not rows:
            continue
        L.append(f"\n### {g}\n\n```")
        w = max(len(k) for k, _ in rows)
        for k, v in rows:
            L.append(f"{k:<{w}} " + " ".join(f"{x:4.0f}" for x in v["third_db"]))
        L.append("```")
    return "\n".join(L) + "\n"


def rico_npz(path: Path) -> None:
    out = {}
    for s in ("metal", "stone", "water"):
        for v in ("whine", "short"):
            for seed in range(4):
                out[f"{s}_{v}_{seed}"] = np.asarray(A.ricochet(surface=s, variant=v, seed=seed))
    np.savez(path, **out)


def rico_compare(a: Path, b: Path) -> str:
    x, y = np.load(a), np.load(b)
    same = sum(k in y.files and x[k].shape == y[k].shape and np.array_equal(x[k], y[k]) for k in x.files)
    return f"ricochet identical to the bit: {same}/{len(x.files)}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", default="")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--md", type=Path)
    ap.add_argument("--wav-dir", type=Path)
    ap.add_argument("--before", type=Path, help="earlier --json to add the change columns")
    ap.add_argument("--only", default=",".join(GROUPS), help="comma list of: " + ",".join(GROUPS))
    ap.add_argument("--rico-out", type=Path)
    ap.add_argument("--rico-ref", type=Path)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    only = {g for g in a.only.split(",") if g}
    bad = only - set(GROUPS)
    if bad:
        sys.exit(f"unknown group {sorted(bad)}")
    res = run(only, a.wav_dir, a.quiet)
    if a.json:
        a.json.write_text(json.dumps({"label": a.label, "sr": DEFAULT_SR, "results": res}, indent=1), encoding="utf-8")
    before = json.loads(a.before.read_text(encoding="utf-8"))["results"] if a.before else None
    md = table(res, before)
    if a.md:
        a.md.write_text(f"# arms_measure {a.label}\n" + md, encoding="utf-8")
    else:
        print(md)
    if a.rico_out:
        rico_npz(a.rico_out)
        if a.rico_ref:
            print(rico_compare(a.rico_out, a.rico_ref))
    err = [k for k, v in res.items() if "error" in v]
    print(f"{len(res)} sounds, {len(err)} errors {err[:5]}")


if __name__ == "__main__":
    main()
