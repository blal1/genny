"""Freeze a preset table and its renders before a migration to TOML, export it, check it after.

    uv run python examples/preset_freeze.py --family weapon --out sounds/preset_ref     freeze (repr of every entry + float64 renders, twice each)
    uv run python examples/preset_freeze.py --export weapon --toml genny/presets/weapon.toml
    uv run python examples/preset_freeze.py --check weapon --out sounds/preset_ref       repr equality, then np.array_equal on every render

A family is declared in FAMILIES (or with register_family): table() gives {name: value}, cases() gives {label: callable returning the
render}, extras() the derived views to compare by repr. A render that differs between its two runs is listed as non deterministic and
left out of the guard. Any difference makes the exit code 1.
"""
import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from genny import presets
from genny.presets import tomllib


@dataclass
class Family:
    table: Callable
    cases: Callable
    extras: Callable | None = None
    to_dict: Callable | None = None
    preset_family: str | None = None


FAMILIES: dict[str, Family] = {}


def register_family(name, table, cases, extras=None, to_dict=None, preset_family=None):
    FAMILIES[name] = Family(table, cases, extras, to_dict, preset_family)


def _fam(name):
    if name not in FAMILIES:
        raise SystemExit(f"famille inconnue {name!r}; connues : {sorted(FAMILIES)}")
    return FAMILIES[name]


def _render(fn):
    return np.asarray(fn(), dtype=np.float64)


def snapshot(name):
    fam = _fam(name)
    table = {str(k): repr(v) for k, v in fam.table().items()}
    extras = {str(k): repr(v) for k, v in (fam.extras() if fam.extras else {}).items()}
    renders, nondet = {}, []
    for label, fn in fam.cases().items():
        a, b = _render(fn), _render(fn)
        if a.shape == b.shape and np.array_equal(a, b, equal_nan=True):
            renders[label] = a
        else:
            nondet.append(label)
    return table, extras, renders, nondet


def freeze(name, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table, extras, renders, nondet = snapshot(name)
    keys = {label: f"r{i:05d}" for i, label in enumerate(renders)}
    np.savez(out / f"{name}.npz", **{keys[k]: v for k, v in renders.items()})
    meta = {"table": table, "extras": extras, "renders": keys, "nondeterministic": nondet, "numpy": np.__version__}
    (out / f"{name}_table.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{name}: {len(table)} entries, {len(extras)} extras, {len(renders)} renders frozen, {len(nondet)} non deterministic {nondet}")
    return 0


def check(name, ref):
    ref = Path(ref)
    meta = json.loads((ref / f"{name}_table.json").read_text(encoding="utf-8"))
    table, extras, renders, nondet = snapshot(name)
    bad = []
    for kind, want, got in (("entry", meta["table"], table), ("extra", meta["extras"], extras)):
        for k in sorted(set(want) | set(got)):
            if want.get(k) != got.get(k):
                bad.append(f"{kind} {k}: gelé {want.get(k)!r} / maintenant {got.get(k)!r}")
    with np.load(ref / f"{name}.npz") as z:
        for label, key in meta["renders"].items():
            if label not in renders:
                bad.append(f"render {label}: " + ("non déterministe maintenant" if label in nondet else "absent"))
                continue
            a, b = z[key], renders[label]
            if a.shape != b.shape:
                bad.append(f"render {label}: forme {a.shape} -> {b.shape}")
            elif not np.array_equal(a, b):
                d = np.abs(a - b)
                bad.append(f"render {label}: {int((a != b).sum())} échantillons différents, écart max {d.max():.3e}")
    if meta["nondeterministic"]:
        print(f"non gardés (non déterministes au gel) : {meta['nondeterministic']}")
    for line in bad:
        print("ECART", line)
    print(f"{name}: {'OK' if not bad else 'REFUSÉ'} ({len(meta['table'])} entrées, {len(meta['renders'])} rendus comparés)")
    return 1 if bad else 0


def _norm(v):
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def _as_dict(fam, k, v):
    if fam.to_dict:
        return dict(fam.to_dict(k, v))
    if isinstance(v, dict):
        return dict(v)
    if hasattr(v, "_asdict"):
        return v._asdict()
    if dataclasses.is_dataclass(v):
        return dataclasses.asdict(v)
    raise SystemExit(f"{k}: pas de to_dict pour {type(v).__name__}")


def export(name, path):
    fam = _fam(name)
    tables = {str(k): _as_dict(fam, k, v) for k, v in fam.table().items()}
    text = presets.dumps(fam.preset_family or name, tables)
    back = tomllib.loads(text)
    back.pop("schema"), back.pop("family")
    diff = [k for k in tables if repr(_norm(tables[k])) != repr(back.get(k))]
    if diff:
        print("ECART à la relecture du TOML :", diff)
        return 1
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8", newline="\n")
    print(f"{name}: {len(tables)} presets écrits dans {path}, relecture identique en repr")
    return 0


def _cartridge_table():
    from genny import ballistics
    return dict(ballistics.CARTRIDGES)


def _cartridge_to_dict(k, v):
    from genny import arms
    d = {f.name: getattr(v, f.name) for f in dataclasses.fields(v) if getattr(v, f.name) is not None}
    d["legacy_peak_pa"], d["legacy_dir_db"] = arms.CALIBERS[k][3:5]
    d["legacy_e_j"] = arms.CHARGE[k][0]
    return d


def _cartridge_extras():
    from genny import arms, ballistics as B
    out = {f"CALIBERS.{k}": v for k, v in arms.CALIBERS.items()}
    out.update({f"CHARGE.{k}": v for k, v in arms.CHARGE.items()})
    out.update({f"state.{k}": B.muzzle_state(k, c.l_ref) for k, c in B.CARTRIDGES.items()})
    return out


def _cartridge_cases():
    from genny import arms
    out = {f"bullet_pass.{k}": (lambda k=k: arms.bullet_pass(caliber=k)) for k in arms.CALIBERS}
    for m in arms.MATERIALS.split("|"):
        for k in ("22lr", "9x19", "556", "762x51", "50bmg"):
            out[f"bullet_impact.{m}.{k}"] = lambda m=m, k=k: arms.bullet_impact(material=m, caliber=k)
    return out


def _case_table():
    from genny import brass
    return dict(brass.CASE_GEOM)


def _case_extras():
    from genny import arms
    out = {f"CASES.{k}": v for k, v in arms.CASES.items()}
    out.update({f"EXTRA.{k}": v for k, v in arms._CASE_EXTRA.items()})
    out.update({f"CASE_OF.{k}": v for k, v in arms._CASE_OF.items()})
    return out


def _case_cases():
    from genny import arms, foley
    out = {f"casing.{k}": (lambda k=k: foley.casing(caliber=k)) for k in arms.CASES}
    out.update({f"casing_ricochet.{k}": (lambda k=k: arms.casing_ricochet(caliber=k)) for k in ("9mm", "556", "762", "50bmg", "12ga")})
    out.update({f"casing_physical.{f}": (lambda f=f: arms.casing_physical(48000, "556", f, 1.2, 2, 0.25, True, 0)) for f in ("concrete", "grass")})
    return out


def _weapon_table():
    from genny import arms
    return dict(arms.PRESETS)


def _weapon_extras():
    from genny import arms
    out = {f"WEAPONS.{k}": v for k, v in arms.WEAPONS.items()}
    out.update({f"BARREL_MM.{k}": v for k, v in arms.BARREL_MM.items()})
    out.update({"REF_PA": arms.REF_PA, "_PM_REF": arms._PM_REF, "_EB_REF": arms._EB_REF})
    return out


def _weapon_cases():
    from genny import arms
    out = {}
    for w in arms.PRESETS:
        out[f"firearm.{w}"] = lambda w=w: arms.firearm(weapon=w)
        out[f"firearm_pressure.{w}"] = lambda w=w: arms.firearm(weapon=w, render="pressure")
        out[f"firearm_pov.{w}"] = lambda w=w: arms.firearm(weapon=w, pov=True)
    out["minigun"] = lambda: arms.minigun()
    out["minigun_short"] = lambda: arms.minigun(dur=1.0, rate=33.0)
    out["minigun_spin"] = lambda: arms.minigun(dur=1.0, fire=False)
    for s in ("metal", "stone", "water"):
        for v in ("whine", "short"):
            for seed in range(4):
                out[f"ricochet.{s}_{v}_{seed}"] = lambda s=s, v=v, seed=seed: arms.ricochet(surface=s, variant=v, seed=seed)
    return out


register_family("cartridge", _cartridge_table, _cartridge_cases, _cartridge_extras, _cartridge_to_dict)
register_family("case", _case_table, _case_cases, _case_extras)
register_family("weapon", _weapon_table, _weapon_cases, _weapon_extras)


def _case_dim_table():
    from genny import arms
    return dict(arms.CASES)


def _case_dim_to_dict(k, v):
    return dict(zip(("L", "dia", "wall"), v))


def _case_extra_table():
    from genny import arms
    return dict(arms._CASE_EXTRA)


def _case_of_table():
    from genny import arms
    return dict(arms._CASE_OF)


def _case_of_to_dict(k, v):
    return {"case": v}


def _gun_table():
    from genny import foley
    return dict(foley.GUNS)


def _gun_to_dict(k, v):
    d = dict(zip(("chirp", "barrel", "burst", "v", "bscale", "bgain", "blop", "rate"), v[:8]))
    d["proj" if isinstance(v[8], str) else "proj_dim"] = v[8]
    return d


def _gun_extras():
    from genny import foley
    return {f"crack_ms.{k}": foley.crack_ms(k) for k in foley.GUNS}


def _gun_cases():
    from genny import foley
    out = {f"gunshot.{k}": (lambda k=k: foley.gunshot(kind=k)) for k in foley.GUNS}
    out.update({f"gunshot_ground.{k}": (lambda k=k: foley.gunshot(kind=k, distance=40.0, ground="grass")) for k in foley.GUNS})
    out.update({f"reload.{k}": (lambda k=k: foley.reload(kind=k)) for k in foley.GUNS})
    return out


register_family("case_dim", _case_dim_table, lambda: {}, None, _case_dim_to_dict)
register_family("case_extra", _case_extra_table, lambda: {})
register_family("case_of", _case_of_table, lambda: {}, None, _case_of_to_dict)
register_family("gun_legacy", _gun_table, _gun_cases, _gun_extras, _gun_to_dict)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--family", metavar="F", help="freeze")
    g.add_argument("--export", metavar="F", help="write the TOML from the in-memory table")
    g.add_argument("--check", metavar="F", help="compare with the frozen files")
    ap.add_argument("--out", default="sounds/preset_ref", help="frozen files folder")
    ap.add_argument("--toml", help="TOML to write (default genny/presets/F.toml)")
    a = ap.parse_args(argv)
    if a.family:
        return freeze(a.family, a.out)
    if a.export:
        return export(a.export, a.toml or Path(presets.__file__).with_name("presets") / f"{a.export}.toml")
    return check(a.check, a.out)


if __name__ == "__main__":
    sys.exit(main())
