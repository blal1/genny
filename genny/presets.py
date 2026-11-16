"""Presets: named parameter sets read from TOML files, validated against the schema each family declares."""
from __future__ import annotations

import builtins
import difflib
import json
import math
import numbers
import os
import re
import warnings
from pathlib import Path
from typing import Any, NamedTuple

try:
    import tomllib
except ImportError:
    import tomli as tomllib

FORMAT = 1
MAX_FILE = 1 << 20
MAX_PRESETS = 2000
MAX_STR = 256
MAX_ARR = 64
MAX_DEPTH = 8

_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_FAMILY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SCHEMA = re.compile(r"^genny\.preset/(\d+)$")
_BARE = re.compile(r"^[A-Za-z0-9_-]+$")
_MISSING = object()
_list = builtins.list


class PresetError(ValueError):
    pass


class PresetWarning(UserWarning):
    pass


class F(NamedTuple):
    type: Any = float
    lo: float | None = None
    hi: float | None = None
    unit: str = ""
    required: bool | None = None
    default: Any = _MISSING
    choices: Any = None
    ref: str | None = None
    items: Any = None
    of: Any = None
    max_len: int = MAX_ARR


_BUILTIN = Path(__file__).with_name("presets")
_SCHEMAS: dict = {}
_VIEWS: dict = {}
_REG: _list = []
_FILES: dict = {}
_INFO: dict = {}
_BAD: dict = {}
_RAW: dict = {}
_RES: dict = {}
_WARNED: set = set()
_PARSES = 0
_PATCH_ON = False
_PATCHED: dict = {}
_PATCH_BAD: dict = {}


def parse_count() -> int:
    return _PARSES


def _warn_once(msg):
    if msg not in _WARNED:
        _WARNED.add(msg)
        warnings.warn(msg, PresetWarning, stacklevel=4)


def _suggest(word, choices):
    near = difflib.get_close_matches(str(word), _list(choices), n=1)
    return f" (voulez-vous dire {near[0]!r} ?)" if near else ""


def _struct(v, loc, depth=0):
    if isinstance(v, bool) or isinstance(v, int):
        return
    if isinstance(v, float):
        if not math.isfinite(v):
            raise PresetError(f"{loc} = {v!r} : valeur non finie refusée")
    elif isinstance(v, str):
        if len(v) > MAX_STR:
            raise PresetError(f"{loc} : chaîne de {len(v)} caractères > {MAX_STR}")
    elif isinstance(v, (_list, tuple)):
        if len(v) > MAX_ARR:
            raise PresetError(f"{loc} : tableau de {len(v)} éléments > {MAX_ARR}")
        for i, x in enumerate(v):
            _struct(x, f"{loc}[{i}]", depth + 1)
    elif isinstance(v, dict):
        if depth > 3 or len(v) > MAX_ARR:
            raise PresetError(f"{loc} : table trop profonde ou de plus de {MAX_ARR} clés")
        for k, x in v.items():
            if not isinstance(k, str) or len(k) > MAX_STR:
                raise PresetError(f"{loc} : clé invalide {k!r}")
            _struct(x, f"{loc}.{k}", depth + 1)
    else:
        raise PresetError(f"{loc} : type {type(v).__name__} non autorisé")


def _read(p, k, size):
    if size > MAX_FILE:
        raise PresetError(f"{k} : fichier de {size} octets > 1 MiB, refusé avant lecture")
    try:
        with open(p, "rb") as fh:
            data = tomllib.load(fh)
    except OSError as e:
        raise PresetError(f"{k} : lecture impossible ({e})") from None
    except ValueError as e:
        raise PresetError(f"{k} : TOML invalide : {e}") from None
    sch = data.pop("schema", None)
    m = _SCHEMA.match(sch) if isinstance(sch, str) else None
    if not m:
        raise PresetError(f'{k} .schema = {sch!r} : attendu "genny.preset/{FORMAT}"')
    if int(m.group(1)) > FORMAT:
        raise PresetError(f"{k} .schema = {sch!r} : version {m.group(1)} non supportée (max {FORMAT}), mettez genny à jour")
    fam = data.pop("family", None)
    if not isinstance(fam, str) or not _FAMILY.match(fam):
        raise PresetError(f"{k} .family = {fam!r} : nom de famille invalide ou absent")
    if len(data) > MAX_PRESETS:
        raise PresetError(f"{k} : {len(data)} presets > {MAX_PRESETS}")
    for name, body in data.items():
        if not _NAME.match(name):
            raise PresetError(f"{k} [{name}] : nom de preset invalide (attendu {_NAME.pattern})")
        if not isinstance(body, dict):
            raise PresetError(f"{k} .{name} : une table [{name}] est attendue")
        _struct(body, f"{k} [{name}]")
    return fam, data


def _parse(p, k):
    global _PARSES
    try:
        st = p.stat()
    except OSError as e:
        raise PresetError(f"{k} : introuvable ({e})") from None
    c = _FILES.get(k)
    if c is None or c[0] != st.st_mtime_ns or c[1] != st.st_size:
        _PARSES += 1
        try:
            val = _read(p, k, st.st_size)
        except PresetError as e:
            val = e
        c = _FILES[k] = (st.st_mtime_ns, st.st_size, val)
    if isinstance(c[2], PresetError):
        raise c[2]
    return c[2]


def _user_dir():
    v = os.environ.get("GENNY_PRESETS_USER")
    if v is not None:
        return Path(v) if v else None
    try:
        if os.name == "nt":
            base = os.environ.get("APPDATA")
            return Path(base) / "genny" / "presets" if base else None
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
        return Path(base) / "genny" / "presets"
    except RuntimeError:
        return None


def _sources():
    out = [(_BUILTIN, "builtin")]
    u = _user_dir()
    if u:
        out.append((u, "user"))
    for s in os.environ.get("GENNY_PRESETS", "").split(os.pathsep):
        if s.strip():
            out.append((Path(s.strip()), "env"))
    return out + _list(_REG)


def sources():
    return _sources()


def declared():
    return sorted(_SCHEMAS)


def _expand(base, org):
    try:
        if base.is_dir():
            return [p for p in sorted(base.iterdir()) if p.suffix == ".toml" and p.is_file()]
        if base.is_file():
            return [base]
    except OSError:
        pass
    if org not in ("builtin", "user"):
        _warn_once(f"{base} : dossier ou fichier de presets introuvable")
    return []


def _flatten(fam, raws, name, raw, src, trail):
    loc = f"{src} [{name}]"
    vals, prov = {}, {}
    parent = raw.get("extends")
    if parent is not None:
        if not isinstance(parent, str):
            raise PresetError(f"{loc}.extends = {parent!r} : un nom de preset est attendu")
        if parent == name or parent in trail:
            raise PresetError(f"{loc}.extends : cycle {' -> '.join(trail + (name, parent))}")
        if len(trail) + 1 > MAX_DEPTH:
            raise PresetError(f"{src} [{trail[0]}].extends : profondeur d'héritage > {MAX_DEPTH} (atteint en [{name}])")
        if parent not in raws:
            raise PresetError(f"{loc}.extends = {parent!r} : preset inconnu dans la famille {fam}{_suggest(parent, raws)}")
        praw, psrc = raws[parent]
        vals, prov = _flatten(fam, raws, parent, praw, psrc, trail + (name,))
    for key, v in raw.items():
        if key != "extends":
            vals[key] = v
            prov[key] = (name, src)
    return vals, prov


def _F(x):
    return x if isinstance(x, F) else F(x)


def _check(v, f, loc, known):
    t = str if f.ref else f.type
    if t is bool:
        if not isinstance(v, bool):
            raise PresetError(f"{loc} = {v!r} : booléen attendu")
        return v
    if t in (int, float):
        if isinstance(v, bool) or not isinstance(v, numbers.Real) or (t is int and not isinstance(v, numbers.Integral)):
            raise PresetError(f"{loc} = {v!r} : {t.__name__} attendu")
        v = int(v) if isinstance(v, numbers.Integral) else float(v)
        if not math.isfinite(v):
            raise PresetError(f"{loc} = {v!r} : valeur non finie")
        if (f.lo is not None and v < f.lo) or (f.hi is not None and v > f.hi):
            u = f" {f.unit}" if f.unit else ""
            raise PresetError(f"{loc} = {v!r} hors de [{'-inf' if f.lo is None else f.lo}, {'+inf' if f.hi is None else f.hi}]{u}")
        return v
    if t is str:
        if not isinstance(v, str) or len(v) > MAX_STR:
            raise PresetError(f"{loc} = {v!r} : chaîne (≤ {MAX_STR}) attendue")
        if f.choices is not None and v not in f.choices:
            raise PresetError(f"{loc} = {v!r} : parmi {_list(f.choices)}{_suggest(v, f.choices)}")
        if f.ref and v not in known.get(f.ref, ()):
            raise PresetError(f"{loc} = {v!r} : preset inconnu dans la famille {f.ref}{_suggest(v, known.get(f.ref, ()))}")
        return v
    if t is tuple:
        if not isinstance(v, (_list, tuple)):
            raise PresetError(f"{loc} = {v!r} : tableau attendu")
        if f.items is not None:
            if len(v) != len(f.items):
                raise PresetError(f"{loc} : {len(f.items)} éléments attendus, reçu {len(v)}")
            return tuple(_check(x, _F(it), f"{loc}[{i}]", known) for i, (x, it) in enumerate(zip(v, f.items)))
        if len(v) > f.max_len:
            raise PresetError(f"{loc} : {len(v)} éléments > {f.max_len}")
        it = _F(f.of or float)
        return tuple(_check(x, it, f"{loc}[{i}]", known) for i, x in enumerate(v))
    if t is dict:
        if not isinstance(v, dict):
            raise PresetError(f"{loc} = {v!r} : table attendue")
        return dict(v)
    if t is _list:
        if not isinstance(v, (_list, tuple)):
            raise PresetError(f"{loc} = {v!r} : tableau de tables attendu")
        if len(v) > f.max_len:
            raise PresetError(f"{loc} : {len(v)} éléments > {f.max_len}")
        sub = {k: _F(x) for k, x in dict(f.of).items()}
        out = []
        for i, item in enumerate(v):
            l2 = f"{loc}[{i}]"
            if not isinstance(item, dict):
                raise PresetError(f"{l2} = {item!r} : table attendue")
            o = {}
            for key, x in item.items():
                if key not in sub:
                    raise PresetError(f"{l2}.{key} : clé inconnue{_suggest(key, sub)}")
                o[key] = _check(x, sub[key], f"{l2}.{key}", known)
            for key, sf in sub.items():
                if key in o:
                    continue
                if sf.default is not _MISSING:
                    o[key] = sf.default
                elif sf.required is not False:
                    raise PresetError(f"{l2}.{key} : champ requis absent")
            out.append(o)
        return out
    raise PresetError(f"{loc} : type de champ {t!r} non géré par le schéma")


def _validate(fam, vals, prov, name, src, known):
    fields, _version, _shape, renamed = _SCHEMAS[fam]
    out = {}
    for key, v in vals.items():
        k2 = renamed.get(key, key)
        n, s = prov.get(key, (name, src))
        loc = f"{s} [{n}].{key}"
        if k2 not in fields:
            raise PresetError(f"{loc} : clé inconnue{_suggest(key, fields)}")
        out[k2] = _check(v, fields[k2], loc, known)
    for key, f in fields.items():
        if key in out:
            continue
        if f.default is not _MISSING:
            out[key] = f.default
        elif f.required is not False:
            raise PresetError(f"{src} [{name}].{key} : champ requis absent")
    return out


def _rebuild(quiet=()):
    order = {}
    for base, org in _sources():
        for p in _expand(base, org):
            k = str(p.resolve())
            order.pop(k, None)
            order[k] = (p, org)
    parsed, bad = {}, {}
    for k, (p, org) in order.items():
        try:
            parsed[k] = _parse(p, k)
        except PresetError as e:
            if org == "builtin":
                raise
            bad[k] = e
    while True:
        merged = {}
        for k in order:
            if k in parsed and k not in bad:
                fam, tables = parsed[k]
                m = merged.setdefault(fam, {})
                for n, raw in tables.items():
                    m[n] = (raw, k)
        res, again = {}, False
        for fam in _SCHEMAS:
            res[fam] = {}
            for n, (raw, k) in merged.get(fam, {}).items():
                try:
                    vals, prov = _flatten(fam, merged[fam], n, raw, k, ())
                    res[fam][n] = _validate(fam, vals, prov, n, k, merged)
                except PresetError as e:
                    if order[k][1] == "builtin":
                        raise
                    bad[k], again = e, True
                    break
            if again:
                break
        if not again:
            break
    _INFO.clear()
    _INFO.update({k: v[1] for k, v in order.items()})
    _RAW.clear()
    _RAW.update(merged)
    _RES.clear()
    _RES.update(res)
    _BAD.clear()
    _BAD.update(bad)
    for fam, (_f, _v, shape, _r) in _SCHEMAS.items():
        view = _VIEWS[fam]
        view.clear()
        for n, d in res[fam].items():
            try:
                view[n] = shape(d) if shape else d
            except (PresetError, KeyError, TypeError, ValueError) as e:
                k = _RAW[fam][n][1]
                if _INFO.get(k) == "builtin":
                    raise
                _warn_once(f"fichier de presets ignoré : {k} [{n}] : {e}")
    for k, e in bad.items():
        if k not in quiet:
            _warn_once(f"fichier de presets ignoré : {e}")
    if _PATCH_ON:
        _sync_patches(quiet)


def _patch_fn(kind, pfn, over):
    if kind == "sfx":
        return lambda sr, **kw: pfn(sr, **{**over, **kw})
    if kind == "drum":
        return lambda sr, vel, **kw: pfn(sr, vel, **{**over, **kw})
    return lambda freq, dur, sr, vel, **kw: pfn(freq, dur, sr, vel, **{**over, **kw})


def _sync_patches(quiet=()):
    from . import drums as D
    from . import instruments as I
    from . import sfx as S
    regs = {"sfx": S.REGISTRY, "instrument": I.REGISTRY, "drum": D.REGISTRY}
    for (kind, name), ent in _list(_PATCHED.items()):
        if regs[kind].get(name) is ent:
            del regs[kind][name]
        if kind == "instrument":
            I.TRIM.pop(name, None)
    _PATCHED.clear()
    _PATCH_BAD.clear()
    for name, d in _RES.get("patch", {}).items():
        src = _RAW["patch"][name][1]
        loc = f"{src} [{name}]"
        try:
            kinds = [k for k in ("sfx", "instrument", "drum") if k in d]
            if len(kinds) != 1:
                raise PresetError(f"{loc} : exactement une des clés sfx, instrument ou drum est attendue")
            kind = kinds[0]
            reg, parent = regs[kind], d[kind]
            if name in reg:
                raise PresetError(f"{loc} : le nom {name!r} est déjà celui d'un {kind} ; un patch ne remplace pas un son existant")
            if parent not in reg or (kind, parent) in _PATCHED:
                raise PresetError(f"{loc}.{kind} = {parent!r} : {kind} inconnu{_suggest(parent, [k for k in reg if (kind, k) not in _PATCHED])}")
            pe = reg[parent]
            over = dict(d.get("params", {}))
            for key in over:
                if key not in pe["params"] or key in pe.get("generic", ()):
                    raise PresetError(f"{loc}.params.{key} : paramètre inconnu de {kind} {parent!r}{_suggest(key, pe['params'])}")
            ent = dict(pe)
            ent["fn"] = _patch_fn(kind, pe["fn"], over)
            ent["desc"] = d.get("desc") or f"{pe['desc']} (patch {name})"
            ent["params"] = {k: ((over[k] if k in over else v[0]), v[1]) for k, v in pe["params"].items()}
            ent["origin"] = "patch"
            reg[name] = ent
            _PATCHED[(kind, name)] = ent
            if kind == "instrument" and parent in I.TRIM:
                I.TRIM[name] = I.TRIM[parent]
        except PresetError as e:
            _PATCH_BAD[src] = e
            if _INFO.get(src) == "builtin":
                raise
            if src not in quiet:
                _warn_once(f"patch ignoré : {e}")


def enable_patches():
    global _PATCH_ON
    _PATCH_ON = True
    _sync_patches()


def schema(family, fields, version=1, shape=None, renamed=None):
    if not isinstance(family, str) or not _FAMILY.match(family):
        raise PresetError(f"schema : nom de famille invalide {family!r}")
    fs = {k: _F(v) for k, v in dict(fields).items()}
    for k, f in fs.items():
        if f.default is not _MISSING:
            _check(f.default, f, f"schema {family}.{k} (défaut)", {})
    _SCHEMAS[family] = (fs, version, shape, dict(renamed or {}))
    view = _VIEWS.setdefault(family, {})
    _rebuild()
    return view


def reload(clear=False):
    if clear:
        _REG.clear()
    _rebuild()


def register(path, origin="user"):
    p = Path(path)
    if not p.exists():
        raise PresetError(f"{p} : introuvable")
    ent = (p, origin)
    if ent not in _REG:
        _REG.append(ent)
    keys = {str(q.resolve()) for q in _expand(p, origin)}
    _rebuild(quiet=keys)
    errs = [_BAD[k] for k in keys if k in _BAD] + [_PATCH_BAD[k] for k in keys if k in _PATCH_BAD]
    if errs:
        _REG.remove(ent)
        _rebuild()
        raise errs[0]


def problems():
    return [str(e) for e in _BAD.values()] + [str(e) for e in _PATCH_BAD.values()]


def _fam_known(family):
    if family not in _SCHEMAS:
        raise PresetError(f"famille de presets non déclarée : {family!r} (déclarées : {sorted(_SCHEMAS)})")


def _copy(d):
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in d.items()}


def get(family, name):
    _fam_known(family)
    res = _RES[family]
    if name not in res:
        raise PresetError(f"{family} : preset inconnu {name!r}{_suggest(name, res)}; disponibles : {_list(res)}")
    return _copy(res[name])


def resolve(family, value):
    if isinstance(value, str):
        return get(family, value)
    if not isinstance(value, dict):
        raise PresetError(f"{family} : un nom ou un dict est attendu, reçu {type(value).__name__}")
    _fam_known(family)
    _struct(value, f"<inline> {family}")
    vals, prov = _flatten(family, _RAW.get(family, {}), "<inline>", value, "<inline>", ())
    return _validate(family, vals, prov, "<inline>", "<inline>", _RAW)


def list(family=None):
    if family is None:
        return sorted(set(_RAW) | set(_SCHEMAS))
    if family not in _RAW and family not in _SCHEMAS:
        raise PresetError(f"famille de presets inconnue : {family!r}")
    return _list(_RAW.get(family, {}))


def origin(family, name):
    if name not in _RAW.get(family, {}):
        raise PresetError(f"{family} : preset inconnu {name!r}")
    k = _RAW[family][name][1]
    return "builtin" if _INFO.get(k) == "builtin" else k


def load(path):
    p = Path(path)
    k = str(p.resolve())
    fam, tables = _parse(p, k)
    if fam not in _SCHEMAS:
        return {fam: {n: dict(t) for n, t in tables.items()}}
    raws = dict(_RAW.get(fam, {}))
    raws.update({n: (t, k) for n, t in tables.items()})
    known = dict(_RAW)
    known[fam] = raws
    out = {}
    for n, t in tables.items():
        vals, prov = _flatten(fam, raws, n, t, k, ())
        out[n] = _validate(fam, vals, prov, n, k, known)
    return {fam: out}


def _key(k):
    return k if _BARE.match(k) else _val(k)


def _val(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, numbers.Integral):
        return str(int(v))
    if isinstance(v, numbers.Real):
        x = float(v)
        if not math.isfinite(x):
            raise PresetError(f"valeur non finie {v!r}")
        return repr(x)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False).replace("\x7f", "\\u007f")
    if isinstance(v, (_list, tuple)):
        return "[" + ", ".join(_val(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_key(k)} = {_val(x)}" for k, x in v.items()) + "}"
    raise PresetError(f"type {type(v).__name__} non écrivable en TOML")


def dumps(family, tables):
    out = [f'schema = "genny.preset/{FORMAT}"', f"family = {_val(family)}", ""]
    for name, d in tables.items():
        out.append(f"[{_key(name)}]")
        for k, v in d.items():
            if isinstance(v, (_list, tuple)) and v and all(isinstance(x, dict) for x in v):
                out += [f"{_key(k)} = ["] + [f"  {_val(x)}," for x in v] + ["]"]
            else:
                out.append(f"{_key(k)} = {_val(v)}")
        out.append("")
    return "\n".join(out)


def save(family, name, data, path):
    _fam_known(family)
    if not _NAME.match(name):
        raise PresetError(f"{path} [{name}] : nom de preset invalide (attendu {_NAME.pattern})")
    p = Path(path)
    k = str(p)
    tables = {}
    if p.exists():
        fam, tables = _read(p, k, p.stat().st_size)
        if fam != family:
            raise PresetError(f"{k} : le fichier est de la famille {fam!r}, pas {family!r}")
    data = dict(data)
    _struct(data, f"{k} [{name}]")
    raws = dict(_RAW.get(family, {}))
    raws.update({n: (t, k) for n, t in tables.items()})
    raws[name] = (data, k)
    vals, prov = _flatten(family, raws, name, data, k, ())
    known = dict(_RAW)
    known[family] = raws
    _validate(family, vals, prov, name, k, known)
    tables[name] = data
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(dumps(family, tables), encoding="utf-8", newline="\n")
    os.replace(tmp, p)
    _rebuild()


_rebuild()


schema("patch", {"sfx": F(str, required=False), "instrument": F(str, required=False), "drum": F(str, required=False),
                 "desc": F(str, required=False), "params": F(dict, required=False)})
