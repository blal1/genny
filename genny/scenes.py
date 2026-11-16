"""Ambience scenes: a bed under sustained layers and scattered events, all taken from the catalogue.

``scene(kind=)`` reaches ``physical.ambience`` (``render_bed``, ``ACT_BEDS``, ``bed_generic``) and lays existing sfx over it,
by name, through the registry. Before a climax the quiet passage is lowered, then a silence, four small hits and the big one;
after it the ears are dulled (``ordnance.earplug``). Sources of every value: research/SOURCES.md, section scenes.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample as _fft_resample

from . import fx as Fx
from . import presets
from . import reverbs as Rv
from .core import DEFAULT_SR
from .instruments import render_chord
from .notes import to_hz
from .ordnance import earplug
from .physical import ambience as Am
from .physical.core import pan_gains
from .presets import F
from .sfx import REGISTRY, render_sfx, sfx

REF = 0.1                # RMS of a layer at 0 dB, room to sum about ten of them under full scale
EDGE = 0.5               # s of fade at both ends of a scene that is not a loop
LOOP_XF = 1.0            # s of crossfade closing a sustained layer on itself
JITTER_DB = 3.0          # spread of the level of one event layer
VARIANTS = 3             # takes of an event before they repeat, with their own level and pan

DIP_DB = -1.5            # the quiet passage lowered over DIP_S
DIP_S = 12.0
SMALL_HITS = 4           # small hits, then the big one
GAP_S = 0.6              # s of silence before the hit
GAP_RAMP_S = 0.05
HIT_GAP_S = 0.7          # between the small hits
EAR_HOLD_S = 1.0         # s of full cotton after the blast
EAR_S = 3.0              # s for the ears to clear after that



TEXT = {    # text bed_generic reads: its keywords pick the emitters
    "open": "open air", "alpine": "mountain peak snow pass pine", "meadow": "grass meadow", "forest": "forest tree",
    "desert": "dry plain", "city": "city street stone wall", "sea": "sea shore", "harbour": "harbour sea",
    "jungle": "jungle tree", "tundra": "snow glacier pass", "cave": "cave tunnel underground", "swamp": "swamp",
    "war": "battle war burn stone", "lake": "lake grass", "town": "town stone", "burn": "burn forest tree wind",
}


def _layer(d):
    """A layer as the code reads it. every = (lo, hi) s for events, none for a layer sounding all along; inst + notes = a held chord."""
    every = tuple(d["every"]) if "every" in d else None
    params = dict(d.get("params", {}))
    if ("inst" in d) == ("sfx" in d):
        raise presets.PresetError(f"layer {d!r} : exactement une des clés sfx ou inst est attendue")
    if every is not None and not (len(every) == 2 and 0 < every[0] <= every[1]):
        raise presets.PresetError(f"layer {d!r} : every = [lo, hi] avec 0 < lo <= hi (s)")
    if "inst" in d:
        return dict(inst=d["inst"], sfx=d["inst"], db=d["db"], wide=d["wide"], opt=d["opt"], every=every, notes=tuple(d.get("notes", ())), params=params)
    return dict(sfx=d["sfx"], db=d["db"], wide=d["wide"], opt=d["opt"], every=every, params=params)


def _scene(d):
    out = dict(bed=d.get("bed"))
    for key in ("act", "climax"):
        if key in d:
            out[key] = d[key]
    out["layers"] = [_layer(x) for x in d["layers"]]
    if "room" in d:
        out["room"] = d["room"]
    return out


# One scene = a bed (a key of TEXT, or none for the quiet air of ambience.room_tone) and its layers; room = reverbs.room preset for everything but the bed.
SCENES = presets.schema("scene", {
    "bed": F(str, choices=tuple(TEXT), required=False),
    "act": F(str, required=False),
    "climax": F(float, lo=0.0, hi=1.0, required=False),
    "layers": F(list, max_len=32, of={
        "sfx": F(str, required=False), "inst": F(str, required=False),
        "db": F(float, lo=-60.0, hi=24.0, unit="dB"),
        "wide": F(bool, default=False), "opt": F(bool, default=False),
        "every": F(tuple, of=float, required=False), "notes": F(tuple, of=str, required=False),
        "params": F(dict, required=False)}),
    "room": F(str, required=False),
}, shape=_scene)

UCS = {
    "AMBAir": "air", "AMBAlpn": "alpine", "AMBAmus": "amusement", "AMBBird": "birdsong", "AMBCele": "celebration", "AMBCnst": "construction", "AMBDsrt": "desert", "AMBDsgn": "designed",
    "AMBEmrg": "emergency", "AMBFant": "fantasy", "AMBFarm": "farm", "AMBForst": "forest", "AMBGras": "grassland", "AMBHist": "historical", "AMBTech": "hitech", "AMBHosp": "hospital",
    "AMBInd": "industrial", "AMBInsc": "insect", "AMBLake": "lakeside", "AMBMrkt": "market", "AMBNaut": "nautical", "AMBOffc": "office", "AMBPark": "park", "AMBPrisn": "prison", "AMBPrtst": "protest",
    "AMBPubl": "public", "AMBRlgn": "religious", "AMBHome": "residential", "AMBRest": "restaurant", "AMBRoom": "room", "AMBRurl": "rural", "AMBSchl": "school", "AMBSci": "scifi", "AMBSea": "seaside",
    "AMBSprt": "sport", "AMBSubn": "suburban", "AMBSwmp": "swamp", "AMBTown": "town", "AMBTraf": "traffic", "AMBTran": "transportation", "AMBTrop": "tropical", "AMBTndra": "tundra",
    "AMBUndr": "underground", "AMBUndwtr": "underwater", "AMBUrbn": "urban", "AMBWar": "warfare",
}

ROOM_TONES = {          # kind: (room preset, dB of the mains hum under the air, or None: no mains there)
    "salon": ("bedroom", -14.0), "bureau": ("studio", -9.0), "hall": ("hall", -16.0), "eglise": ("church", None), "cave": ("cave", None), "cabine": ("closet", -6.0),
}


def _stereo(y):
    y = np.asarray(y, float)
    return np.stack([y, y], axis=1) if y.ndim == 1 else y


def _to_rate(y, n):
    """48 kHz kernel to the scene's rate. FFT resampling treats the buffer as one period, so a seamless bed stays so."""
    return _fft_resample(y, n, axis=0) if y.shape[0] != n else y


def _base(kind, s, dur, sr, seed, modifier, place):
    """The bed, RMS REF: an act, a place described in words, or the room's air."""
    rng = np.random.default_rng([int(seed), 7])
    if place or s.get("bed"):
        text = place or TEXT[s["bed"]]
        y = Am.render_bed({"id": f"amb.library.{kind}", "prompt": text}, rng, dur)
    elif s.get("act"):
        y = Am.render_bed({"id": f"amb.act.{s['act']}", "prompt": ""}, rng, dur)
    else:
        y = Am.room_tone(dur, rng, loop=True)
    if modifier:
        if modifier not in Am.BED_MODIFIERS:
            raise ValueError(f"unknown modifier {modifier!r}; choose from {list(Am.BED_MODIFIERS)}")
        y = Am.bed_modifier(y, modifier, rng, dur)
    y = _to_rate(_stereo(y), int(round(dur * sr)))
    return y / (np.sqrt(np.mean(y ** 2)) + 1e-12) * REF


def _take(layer, dur, seed, sr):
    if "inst" in layer:
        return render_chord(layer["inst"], [to_hz(n) for n in layer["notes"]], dur, sr)
    p = dict(layer["params"])
    known = REGISTRY[layer["sfx"]]["params"]
    if "dur" in known and layer["every"] is None:
        p["dur"] = dur
    if "seed" in known:
        p["seed"] = int(seed)
    return render_sfx(layer["sfx"], sr, **p)


def _sustained(layer, n, sr, seed, loop):
    dur = n / sr + (LOOP_XF if loop else 0.0)
    m = int(round(dur * sr))

    def one(k):
        y = np.asarray(_take(layer, dur, seed + 101 * k, sr), float)
        y = y.mean(axis=1) if y.ndim == 2 and layer["wide"] else y
        y = np.pad(y, [(0, max(0, m - y.shape[0]))] + [(0, 0)] * (y.ndim - 1))[:m]
        return Fx.loop(y, sr, crossfade=LOOP_XF)[:n] if loop else y[:n]

    y = np.stack([one(0), one(1)], axis=1) if layer["wide"] else _stereo(one(0))
    return y / (np.sqrt(np.mean(y ** 2)) + 1e-12) * REF * 10 ** (layer["db"] / 20)


def _shape_event(y, sr):
    """Level from the loudest 100 ms, so that a blip and a long pass-by are set alike. Ends faded: an sfx may stop on a non-zero sample."""
    y = np.array(y, float)
    a, b = min(int(0.001 * sr), y.shape[0] // 2), min(int(0.02 * sr), y.shape[0] // 2)
    fade = np.ones(y.shape[0])
    fade[:a] = np.linspace(0, 1, a, endpoint=False)
    fade[y.shape[0] - b:] = np.linspace(1, 0, b)
    y = y * (fade if y.ndim == 1 else fade[:, None])
    w = max(1, int(0.1 * sr))
    e = np.convolve(np.sum(_stereo(y) ** 2, axis=1) / 2, np.ones(w) / w, "same")
    return y / (np.sqrt(e.max()) + 1e-12)


def _add(buf, y, i, loop):
    """Lay y into buf at i. In a loop the part past the end comes back at the start."""
    n = buf.shape[0]
    k = min(y.shape[0], n)
    y = y[:k]
    end = min(k, n - i)
    buf[i:i + end] += y[:end]
    if loop and end < k:
        buf[: k - end] += y[end:]


def _events(layer, n, sr, seed, loop, density):
    """(buffer, start samples). Always at least one event."""
    rng = np.random.default_rng([int(seed), 13])
    lo, hi = (x / max(density, 1e-3) for x in layer["every"])
    dur = n / sr
    t = rng.uniform(0.0, min(lo, 0.5 * dur))
    takes = []
    known = "seed" in REGISTRY[layer["sfx"]]["params"]
    for k in range(VARIANTS if known else 1):
        takes.append(_shape_event(_take(layer, 0.0, seed + 211 * k, sr), sr))
    buf = np.zeros((n, 2))
    at = []
    while t < dur:
        y = takes[int(rng.integers(len(takes)))]
        g = REF * 10 ** ((layer["db"] + rng.uniform(-JITTER_DB, JITTER_DB)) / 20)
        i = int(t * sr)
        if y.ndim == 1:
            gl, gr = np.array(pan_gains(rng.uniform(-0.7, 0.7))) * np.sqrt(2.0)    # equal power per channel-pair: same level as the centred take it was set on
            y = np.stack([y * gl, y * gr], axis=1)
        _add(buf, y * g, i, loop)
        at.append(i)
        t += rng.uniform(lo, hi)
    return buf, at


def _in_room(x, sr, preset, mix, loop):
    """reverbs.room, same length. Its early reflections repeat whatever edge goes in, as clicks 15-100 ms later (measured), so the input has none:
    faded, or in a loop run over two periods with the second kept (exact once the tail is shorter than a period)."""
    n = x.shape[0]
    if loop:
        return _stereo(Rv.room(np.concatenate([x, x]), sr, preset=preset, mix=mix))[n:2 * n]
    k = min(int(0.05 * sr), n // 2)
    x = np.array(x, float)
    x[:k] *= (np.arange(k) / k)[:, None]
    x[n - k:] *= (1 - np.arange(k) / k)[:, None]
    return _stereo(Rv.room(x, sr, preset=preset, mix=mix))[:n]


def contrast_gain(n, sr, t0):
    """Gain on all that sounds before the big hit at t0 s: the quiet passage lowered, then silence. 1 again from t0 on."""
    g = np.ones(n)
    i0 = min(n, int(t0 * sr))
    gap = int(GAP_S * sr)
    ramp = max(1, int(GAP_RAMP_S * sr))
    fall = min(int(DIP_S * sr), max(i0 - gap - ramp, 0))
    start = i0 - gap - ramp - fall
    dip = 10 ** (DIP_DB / 20)
    if fall > 0:
        g[start:start + fall] = 1 + (dip - 1) * (1 - np.cos(np.pi * np.arange(fall) / fall)) / 2
    g[max(start + fall, 0):i0] = dip
    r0 = max(i0 - gap - ramp, 0)
    g[r0:r0 + ramp] *= np.cos(np.pi / 2 * np.arange(ramp) / ramp)
    g[r0 + ramp:i0] = 0.0
    k = min(ramp, n - i0)                # the place comes back under the blast, without a step of its own
    g[i0:i0 + k] = np.sin(np.pi / 2 * np.arange(k) / ramp)
    return g


def after_blast(y, sr, t0):
    """From t0 on the ears are cotton (ordnance.earplug): held EAR_HOLD_S, then clearing over EAR_S. Linear: both signals are the same sound, in phase."""
    i0 = min(y.shape[0], int(t0 * sr))
    m = y.shape[0] - i0
    if m <= 0:
        return y
    u = np.clip((np.arange(m) / sr - EAR_HOLD_S) / EAR_S, 0.0, 1.0)[:, None]
    out = y.copy()
    out[i0:] = earplug(y, sr, "cotton")[i0:] * (1 - u) + y[i0:] * u
    return out


def climax_hits(sr, seed, n, t0):
    """Four firecrackers, then the big one at t0, as blasts: small and far, then big and near."""
    buf = np.zeros((n, 2))
    i0 = int(t0 * sr)
    for k in range(SMALL_HITS):
        y = _shape_event(render_sfx("blast", sr, kind="grenade", charge=0.05, distance=120.0, ground=False, seed=int(seed) + k), sr)
        at = i0 - int((GAP_S + GAP_RAMP_S + (SMALL_HITS - k - 0.5) * HIT_GAP_S) * sr)
        if at >= 0:
            _add(buf, _stereo(y) * REF * 10 ** (-3.0 / 20), at, False)
    y = _shape_event(render_sfx("blast", sr, kind="grenade", charge=5.0, distance=40.0, seed=int(seed) + 9), sr)
    _add(buf, _stereo(y) * REF * 10 ** (12.0 / 20), i0, False)          # -3 and +12 dB re the bed
    return buf


def build(kind, sr=DEFAULT_SR, dur=20.0, seed=0, density=1.0, modifier="", place="", climax=-1.0, loop=0):
    """Scene in parts, for the sfx and for the measurements: dict(bed, layers=[(name, buffer, starts or None)], room, climax_t, out)."""
    kind = UCS.get(kind, kind)
    if kind == "place":
        if not place:
            raise ValueError("scene kind=place needs place=<words>")
        s = dict(layers=[])
    elif kind in Am.ACT_BEDS:
        s = dict(act=kind, layers=[])
    elif kind in SCENES:
        s = SCENES[kind]
    else:
        raise ValueError(f"unknown scene {kind!r}; choose from {sorted(SCENES) + sorted(Am.ACT_BEDS) + ['place'] + sorted(UCS)}")
    dur, density = float(dur), float(density)
    n, loop = int(round(dur * sr)), bool(int(loop))
    bed = _base(kind, s, dur, sr, seed, modifier, place)
    over, parts = np.zeros((n, 2)), []
    for k, layer in enumerate(s["layers"]):
        if layer["opt"] and layer["sfx"] not in REGISTRY:
            continue
        sd = int(seed) * 1009 + k
        if layer["every"] is None:
            y, at = _sustained(layer, n, sr, sd, loop), None
        else:
            y, at = _events(layer, n, sr, sd, loop, density)
        parts.append((layer["sfx"], y, at))
        over += y
    room = s.get("room")
    if room:
        over = _in_room(over, sr, room, 0.6, loop)
    y = bed + over
    t0 = (s.get("climax", 0.0) * dur if climax < 0 else float(climax)) if not loop else 0.0
    if t0 > 0:
        y = y * contrast_gain(n, sr, t0)[:, None]
        hits = climax_hits(sr, seed, n, t0)
        parts.append(("blast", hits, None))
        y = after_blast(y, sr, t0) + hits
    if not loop:
        e = min(int(EDGE * sr), n // 2)
        y[:e] *= (np.arange(e) / e)[:, None]
        y[n - e:] *= (1 - np.arange(e) / e)[:, None]
    return dict(bed=bed, layers=parts, room=room, climax_t=t0, out=y / max(1.0, np.abs(y).max() / 0.9))


@sfx("scene", "An ambience: a bed (wind, sea, fire, rooms) with layers and events of the catalogue over it. Every UCS AMB* place by name or CatID, the five act beds, or a place described in words.",
     kind=("forest", "|".join(sorted(SCENES)) + "|" + "|".join(sorted(Am.ACT_BEDS)) + "|place | a UCS CatID: AMBForst, AMBWar..."),
     dur=(20.0, "s"), density=(1.0, "x how often events come"), modifier=("", "night | dawn | high_tide | dark: the bed's variant"),
     place=("", "words for bed_generic: its keywords pick the emitters ('frozen river under a stone wall'); replaces the kind's bed"),
     climax=(-1.0, "s of a big blast with a quieter lead-in, four small hits, a silence, then dulled ears; -1 = the kind's own (warfare 70 % of dur), 0 = none"),
     loop=(0, "1 = seamless: no end fades, event tails folded onto the start, no climax"), seed=(0, "variation"))
def scene(sr=DEFAULT_SR, kind="forest", dur=20.0, density=1.0, modifier="", place="", climax=-1.0, loop=0, seed=0):
    return build(str(kind), sr, dur, seed, density, str(modifier), str(place), climax, loop)["out"]


@sfx("room_tone", "What an empty room sounds like: its air (pink noise low-passed at 300 Hz, slowly breathing) through the room's own decay, with the mains hum where there are mains.",
     kind=("salon", "|".join(ROOM_TONES)), mains=(50.0, "Hz of the hum: 50 or 60"), dur=(6.0, "s"), seed=(0, "variation"))
def room_tone(sr=DEFAULT_SR, kind="salon", mains=50.0, dur=6.0, seed=0):
    if str(kind) not in ROOM_TONES:
        raise ValueError(f"unknown room_tone {kind!r}; choose from {list(ROOM_TONES)}")
    preset, hum_db = ROOM_TONES[str(kind)]
    n = int(round(float(dur) * sr))
    air = _to_rate(Am.room_tone(float(dur), np.random.default_rng([int(seed), 3]), loop=True), n)
    air = _in_room(air, sr, preset, 0.5, False)
    y = air / (np.sqrt(np.mean(air ** 2)) + 1e-12)
    if hum_db is not None:
        # mains_hum starts on a step that its 1/f0 comb repeats as clicks at 20, 40 ms (measured): that start is left out
        h = render_sfx("mains_hum", sr, dur=float(dur) + 0.2, mains=float(mains), seed=int(seed))[int(0.2 * sr):int(0.2 * sr) + n]
        y = y + _stereo(h) / (np.sqrt(np.mean(h ** 2)) + 1e-12) * 10 ** (hum_db / 20)
    e = min(int(0.1 * sr), n // 2)
    y[:e] *= (np.arange(e) / e)[:, None]
    y[n - e:] *= (1 - np.arange(e) / e)[:, None]
    return y / np.abs(y).max() * 0.5


presets.enable_patches()
