"""Animals: birds by group, farm and wild mammals, whales, swarms, creatures on the move.

Sources: ``research/SOURCES.md``. Nothing here has been listened to: see tests/test_fauna.py for what was measured.
"""
from __future__ import annotations

import numpy as np

from . import contact as K
from . import creatures as C
from . import foley as FO
from .core import DEFAULT_SR
from .creatures import _arc, _dark, _slow, _unit, _voc, _vowel_keys
from .foley import P, _fin, _n, _nm
from .friction import squeak
from .fx import underwater
from .physical import ambience as amb
from .physical import voice as V
from .pinball import chirp
from .sfx import REGISTRY, sfx


def _kind(kind, table, what) -> str:
    k = str(kind).replace(" ", "_")
    if k not in table:
        raise ValueError(f"{what} kind {kind!r}: use {' | '.join(table)}")
    return k


def _fit(x, n: int) -> np.ndarray:
    x = np.asarray(x, float)
    return np.pad(x, (0, n - len(x))) if len(x) < n else x[:n]


def _put(buf: np.ndarray, x: np.ndarray, t: float, g: float = 1.0) -> None:
    """`foley._put` with 2 ms off the end of the layer: a brick that stops above zero leaves a step in the silence after it."""
    x = np.array(x, float)
    m = min(len(x) // 2, _n(0.002))
    x[len(x) - m:] *= np.cos(0.5 * np.pi * np.arange(m) / m) ** 2
    FO._put(buf, x, t, g)


def _voice(pts, dur, *, size=1.0, rough=0.3, vowels="A-u", seed=0, rise=0.3, trill=(0.0, 0.0), am=(0.0, 0.0), sub=0.0, bright=1.1) -> np.ndarray:
    """The engine of `creatures.animal` with a free Hz contour [(u, Hz)], a vibrato and an amplitude trill. Mono at P, unit peak."""
    rng = np.random.default_rng([abs(int(seed)), 121])
    n = _n(dur)
    r = float(np.clip(rough, 0.0, 1.0))
    t = np.arange(n) / P
    f0 = _arc(n, pts) * (1.0 + trill[1] * np.sin(2 * np.pi * trill[0] * t))
    # articulation_env ** 0.6 leaves zero with an infinite slope at both ends (in `animal` too): 30 ms edges take that step away, and let the first low-pitched pulses in gradually
    k = min(n // 2, _n(0.03))
    edge = np.minimum(np.minimum(np.arange(n), n - 1 - np.arange(n)) / k, 1.0)
    env = np.sin(0.5 * np.pi * edge) ** 2 * V.articulation_env(n, rise, offset=0.0) ** 0.6 * (1.0 - am[1] * 0.5 * (1.0 + np.cos(2 * np.pi * am[0] * t)))
    y = _voc(f0, env, rng, tract_m=0.17 * size, harsh=r, vowel=_vowel_keys(vowels, dur), jitter=0.003 + 0.03 * r, shimmer=0.02 + 0.25 * r, sub=0.4 * r * r)
    return _nm(_dark(y, bright))


def _hz_pts(hz):
    return [(0, hz), (1, hz)]


def _am(y: np.ndarray, rate: float, depth: float) -> np.ndarray:
    return y * (1.0 - depth * 0.5 * (1.0 + np.cos(2 * np.pi * rate * np.arange(len(y)) / P)))


# =============================================================================================
# BIRDS: second bronchus and ring modulator
# =============================================================================================
def _pulse(f, w) -> np.ndarray:
    """1/(1 + x^2) shaping of a cosine, one pulse per cycle of ``f``."""
    return 1.0 / (1.0 + (w * np.cos(np.pi * np.cumsum(f) / P)) ** 2)


def bronchi(n: int, bp1, bp2, bw1, bw2, rm) -> np.ndarray:
    """rm 0 = difference only (the product), 1 = sum only, 0.5 = both."""
    a, b = _pulse(np.full(n, float(bp1)), bw1), _pulse(np.full(n, float(bp2)), bw2)
    return rm * (a + b) + (1.0 - rm) * a * b


def _syrinx(syl, seed: int) -> np.ndarray:
    """One syllable: base frequency + mod x ring into the pessulus pulse, then the two parallel trachea filters and the beak.
    syl: dur, [(u, Hz)], gain, bp1, bp2, rm, mod Hz, pw, tf1, tf2, q (2-5), hp, chaos (FM feedback for loose syrinxes), noise."""
    dur, pts, g, bp1, bp2, rm, mod, pw, tf1, tf2, q, hp, chaos, noise = syl
    n = _n(dur)
    rng = np.random.default_rng([abs(int(seed)), 122])
    f = _arc(n, pts) + mod * (bronchi(n, bp1, bp2, 1.0, 1.0, rm) - rm)
    nz = noise * V._pd_lop(rng.uniform(-1, 1, n), 2000.0, float(P))
    x = C._fm_syrinx(np.maximum(f, 20.0), np.full(n, float(pw)), np.full(n, float(chaos)), nz, float(P))
    y = C._bpf(x, tf1, q) + C._bpf(x, tf2, q)
    m = min(n // 2, _n(0.012))
    e = np.ones(n)
    e[:m] = np.sin(0.5 * np.pi * np.arange(m) / m) ** 2
    e[n - m:] = e[:m][::-1]
    return g * C._hp(y, hp) * e


def _song(syls, seed: int) -> np.ndarray:
    """syls: (syllable, gap after it). Mono at P, unit peak."""
    out = np.zeros(_n(sum(s[0] + gap for s, gap in syls) + 0.05))
    t = 0.01
    for i, (s, gap) in enumerate(syls):
        _put(out, _syrinx(s, seed + i), t)
        t += s[0] + gap
    return _nm(out)


#          dur  contour                                   g    bp1   bp2  rm   mod  pw  tf1   tf2   q   hp   chaos noise
def _eagle(seed):                                         # thin descending scream
    rng = np.random.default_rng(seed)
    s = [((rng.uniform(0.18, 0.26), [(0, 2300), (0.25, 2600), (1, 1800)], 1.0 - 0.1 * i, 7.0, 9.0, 0.7, 120.0, 1.5, 2400, 2900, 3.0, 800, 0.05, 0.02), 0.07) for i in range(5)]
    return _song(s, seed)


def _parrot(seed):                                        # harsh squawks: loose syrinx, strong ring
    rng = np.random.default_rng(seed)
    s = [((rng.uniform(0.12, 0.28), [(0, rng.uniform(900, 1400)), (0.5, rng.uniform(1500, 2200)), (1, rng.uniform(800, 1300))], 1.0, 9.0, 6.0, 0.3, 200.0, 2.5, 1800, 2900, 2.5, 500, 0.5, 0.15), rng.uniform(0.05, 0.2)) for _ in range(5)]
    return _song(s, seed)


def _heron(seed):                                         # one or two low rasping croaks
    s = [((0.6, [(0, 330), (0.2, 400), (1, 240)], 1.0, 10.0, 8.0, 0.5, 90.0, 3.0, 900, 1800, 3.0, 150, 0.6, 0.3), 0.18),
         ((0.35, [(0, 300), (1, 220)], 0.8, 10.0, 8.0, 0.5, 90.0, 3.0, 900, 1800, 3.0, 150, 0.6, 0.3), 0.0)]
    return _song(s, seed)


BIRD_KINDS = ("rapaces", "tropicaux", "echassiers", "marins")


@sfx("bird_kind", "Birds by group: rapaces (eagle scream), tropicaux (parrot squawks), echassiers (heron croak), marins (gull). Farnell's ring-modulated syrinx.",
     kind=("rapaces", " | ".join(BIRD_KINDS)), seed=(0, "variation"))
def bird_kind(sr=DEFAULT_SR, kind="rapaces", seed=0):
    k = _kind(kind, BIRD_KINDS, "bird_kind")
    if k == "marins":
        return _fin(C.birdsong(sr=P, species="gull", seed=seed), sr, 0.7, fade=0.005, tail=0.03)
    return _fin({"rapaces": _eagle, "tropicaux": _parrot, "echassiers": _heron}[k](int(seed)), sr, 0.7, fade=0.005, tail=0.03)


CHICKEN_KINDS = ("caquet", "poule", "poussin")


@sfx("chicken", "Chickens: caquet (buk-buk-buk-bukAAK), poule (low contented croon), poussin (rising peeps). The rooster is birdsong species=rooster.",
     kind=("caquet", " | ".join(CHICKEN_KINDS)), seed=(0, "variation"))
def chicken(sr=DEFAULT_SR, kind="caquet", seed=0):
    k = _kind(kind, CHICKEN_KINDS, "chicken")
    rng = np.random.default_rng(int(seed))
    if k == "caquet":
        buk = lambda d, a, b, g: (d, [(0, a), (1, b)], g, 8.0, 5.0, 0.4, 80.0, 2.5, 1100, 2400, 3.0, 300, 0.45, 0.2)
        s = [(buk(0.07, 520, 400, 0.6), rng.uniform(0.11, 0.17)) for _ in range(4)] + [(buk(0.42, 640, 860, 1.0), 0.0)]
    elif k == "poule":
        s = [((rng.uniform(0.25, 0.4), [(0, 340), (0.4, 400), (1, 300)], 0.8, 6.0, 4.0, 0.5, 40.0, 1.8, 800, 1600, 3.5, 150, 0.1, 0.05), rng.uniform(0.12, 0.3)) for _ in range(4)]
    else:
        s = [((rng.uniform(0.09, 0.14), [(0, 2900), (0.7, 3900), (1, 3500)], 1.0, 0.0, 0.0, 0.5, 0.0, 1.0, 3000, 2800, 4.0, 1000, 0.0, 0.0), rng.uniform(0.1, 0.22)) for _ in range(6)]
    return _fin(_song(s, int(seed)), sr, 0.7, fade=0.005, tail=0.03)


FARM_BIRDS = ("oie", "dindon", "canard")


@sfx("farm_bird", "Farm birds: oie (goose honks), dindon (turkey gobble), canard (duck, birdsong species=duck).",
     kind=("oie", " | ".join(FARM_BIRDS)), seed=(0, "variation"))
def farm_bird(sr=DEFAULT_SR, kind="oie", seed=0):
    k = _kind(kind, FARM_BIRDS, "farm_bird")
    if k == "canard":
        return _fin(C.birdsong(sr=P, species="duck", seed=seed), sr, 0.7, fade=0.005, tail=0.03)
    if k == "oie":
        s = [((0.28, [(0, 420), (0.3, 620), (1, 480)], 1.0, 8.0, 6.0, 0.4, 100.0, 2.5, 1000, 2300, 3.0, 300, 0.5, 0.15), 0.14)] * 3
    else:
        s = [((0.055, [(0, 700), (1, 520)], 1.0 - 0.05 * i, 10.0, 7.0, 0.5, 120.0, 2.5, 1200, 2500, 2.5, 300, 0.35, 0.1), 0.025) for i in range(12)]
    return _fin(_song(s, int(seed)), sr, 0.7, fade=0.005, tail=0.03)


# =============================================================================================
# CATTLE AND HORSES
# =============================================================================================
PIG = {
    "grognement": ([(0, 95), (0.5, 120), (1, 80)], 0.94, 0.6, 0.75, "U-@-U", 0.3),
    "couinement": ([(0, 1100), (0.3, 1700), (1, 1200)], 1.0, 0.45, 0.8, "ae-E", 2.8),
    "aboiement": ([(0, 380), (0.25, 520), (1, 300)], 0.21, 0.55, 0.65, "A-@", 1.6),
    "trompe": ([(0, 260), (0.6, 640), (1, 520)], 0.7, 0.5, 0.45, "@-A-ae", 1.3),
}


@sfx("pig", "Pig calls: grognement (0.94 s), couinement, aboiement (0.21 s), trompe.", kind=("grognement", " | ".join(PIG)), seed=(0, "variation"))
def pig(sr=DEFAULT_SR, kind="grognement", seed=0):
    pts, d, size, r, vw, br = PIG[_kind(kind, PIG, "pig")]
    return _fin(_voice(pts, d, size=size, rough=r, vowels=vw, seed=seed, bright=br), sr, 0.7, fade=0.005, tail=0.01)


SHEEP = {"brebis": (150.0, 1.1, 1.0), "agneau": (330.0, 0.6, 0.6)}
BLEAT_HZ, BLEAT_DEPTH = 7.0, 0.7


@sfx("sheep", "Sheep bleat with its fast amplitude trill and vibrato: brebis (ewe), agneau (lamb).",
     kind=("brebis", " | ".join(SHEEP)), seed=(0, "variation"))
def sheep(sr=DEFAULT_SR, kind="brebis", seed=0):
    hz, size, d = SHEEP[_kind(kind, SHEEP, "sheep")]
    y = _voice([(0, hz * 0.92), (0.2, hz), (1, hz * 0.9)], d, size=size, rough=0.3, vowels="E-ae-@", seed=seed, trill=(BLEAT_HZ, 0.02), am=(BLEAT_HZ, BLEAT_DEPTH))
    return _fin(y, sr, 0.7, fade=0.005, tail=0.02)


WHINNY_F0, WHINNY_G0 = 399.22, 1543.26
INTRO = 1.0 - 0.794                                                   # G0 alone: the overlap is 79.4 % of the call
WHISTLE_G = 0.5
WHINNY_S = 1.4


def whinny(f0=WHINNY_F0, g0=WHINNY_G0, dur=WHINNY_S, valence=0.0, seed=0) -> np.ndarray:
    """Two sources: cords through the tract (F0), a sine whistle in the larynx (G0), not harmonically related.
    Negative whinnies start higher and last longer."""
    v = float(np.clip(valence, -1.0, 1.0))
    d = float(dur) * (1.0 - 0.2 * v)
    n = _n(d)
    rng = np.random.default_rng([abs(int(seed)), 123])
    t = np.arange(n) / P
    u = t / d
    g = g0 * np.interp(u, [0.0, INTRO, 0.8, 1.0], [1.12 - 0.1 * v, 1.04, 0.96, 0.88]) * (1.0 + 0.012 * np.sin(2 * np.pi * 6.0 * t))
    whistle = np.sin(2 * np.pi * np.cumsum(g) / P + rng.uniform(0, 6.28)) * np.interp(u, [0.0, 0.03, 0.8, 1.0], [0.0, 1.0, 0.8, 0.0])
    m = _n(d * (1.0 - INTRO))
    cords = _voice([(0, f0 * 1.06), (0.7, f0), (1, f0 * 0.8)], d * (1.0 - INTRO), size=2.0, rough=0.12, vowels="ae-A-@", seed=seed, rise=0.15, trill=(6.0, 0.02), am=(6.0, 0.25))
    end = m - _n(0.2 * d)
    cords[end:] *= 1.0 - np.interp(np.arange(m - end), [0, _n(0.05)], [0.0, 0.85]) * (0.5 + 0.5 * np.cos(2 * np.pi * 28.0 * np.arange(m - end) / P))
    y = WHISTLE_G * whistle
    _put(y, cords, INTRO * d)
    return y


HORSE = ("hennissement", "ebrouement", "soufflement", "nicker")


@sfx("horse", "Horse calls: hennissement (two sources: cords near 399 Hz, laryngeal whistle near 1543 Hz, not harmonic), ebrouement (snort), soufflement (blow), nicker (low pulsed). Hooves: footsteps gait=gallop.",
     kind=("hennissement", " | ".join(HORSE)), f0=(WHINNY_F0, "hennissement: Hz of the cords"), g0=(WHINNY_G0, "hennissement: Hz of the whistle"),
     valence=(0.0, "hennissement: -1 distress (longer, starts higher) .. 1 glad"), seed=(0, "variation"))
def horse(sr=DEFAULT_SR, kind="hennissement", f0=WHINNY_F0, g0=WHINNY_G0, valence=0.0, seed=0):
    k = _kind(kind, HORSE, "horse")
    if k == "hennissement":
        return _fin(whinny(float(f0), float(g0), WHINNY_S, valence, seed), sr, 0.7, fade=0.005, tail=0.02)
    if k == "ebrouement":                                    # lips and nostrils flutter on one burst of air
        b = _nm(_fit(C.breath(sr=P, kind="out", effort=1.0, size=2.0, dur=0.7, seed=seed), _n(0.7)))
        return _fin(_am(b, 42.0, 0.85), sr, 0.7, fade=0.005, tail=0.05)
    if k == "soufflement":
        return _fin(_nm(C.breath(sr=P, kind="out", effort=0.7, size=2.2, dur=1.1, seed=seed)), sr, 0.6, fade=0.01, tail=0.08)
    y = _voice([(0, 90), (0.3, 110), (1, 75)], 0.9, size=2.2, rough=0.1, vowels="U-u", seed=seed, am=(13.0, 0.9), bright=0.5)
    return _fin(y, sr, 0.7, fade=0.01, tail=0.03)


HEE, HAW = (500.0, 320.0), (180.0, 100.0)
BRAY_S = (0.55, 0.7)


@sfx("donkey", "Donkey bray: hee on the breath in, haw on the breath out, one source only (no second voice in a donkey).",
     cycles=(3, "hee-haws"), seed=(0, "variation"))
def donkey(sr=DEFAULT_SR, cycles=3, seed=0):
    c = int(np.clip(cycles, 1, 12))
    out = np.zeros(_n(c * sum(BRAY_S) + 0.05))
    for i in range(c):
        t0 = i * sum(BRAY_S)
        _put(out, _voice([(0, HEE[0]), (1, HEE[1])], BRAY_S[0], size=1.8, rough=0.55, vowels="i:-E", seed=seed + 2 * i, rise=0.15), t0, 0.55)
        _put(out, _voice([(0, HAW[0]), (1, HAW[1])], BRAY_S[1], size=1.8, rough=0.5, vowels="A-u", seed=seed + 2 * i + 1, rise=0.12), t0 + BRAY_S[0], 1.0)
    return _fin(out, sr, 0.7, fade=0.005, tail=0.03)


KID_F0 = (573.0, 467.0)
FARM = {"ane": dict(size=1.8, pitch=500.0, contour="arch", excursion=0.5, roughness=0.55, vowels="i:-A-u", dur=1.0),
        "mouton": dict(size=1.0, pitch=180.0, contour="flat", excursion=0.0, roughness=0.3, vowels="E-@", dur=0.9),
        "chevre": dict(size=1.0, pitch=KID_F0[0], contour="flat", excursion=0.0, roughness=0.35, vowels="E-ae", dur=0.8)}
FARM_KINDS = (*FARM, "table")


@sfx("farm_animal", "Table of `animal` presets: ane, mouton, chevre (age in weeks: kid 573 Hz at 0, 467 Hz at 7), and table (the three in a row).",
     kind=("ane", " | ".join(FARM_KINDS)), age=(0.0, "chevre: weeks, 0..7"), seed=(0, "variation"))
def farm_animal(sr=DEFAULT_SR, kind="ane", age=0.0, seed=0):
    """``table`` plays the whole preset table."""
    k = _kind(kind, FARM_KINDS, "farm_animal")
    if k == "table":
        out = np.zeros(_n(sum(p["dur"] for p in FARM.values()) + 0.5))
        t = 0.0
        for name, p in FARM.items():
            _put(out, _farm_call(name, age, seed), t)
            t += p["dur"] + 0.25
        return _fin(out, sr, 0.7, fade=0.005, tail=0.02)
    return _fin(_farm_call(k, age, seed), sr, 0.7, fade=0.005, tail=0.02)


def _farm_call(k, age, seed) -> np.ndarray:
    p = dict(FARM[k])
    if k == "chevre":
        p["pitch"] = float(np.interp(age, [0.0, 7.0], KID_F0))
    y = _nm(C.animal(sr=P, seed=seed, **p))
    return y if k == "ane" else _am(y, BLEAT_HZ, BLEAT_DEPTH)


# =============================================================================================
# WILD ANIMALS
# =============================================================================================
WILD = ("ours", "cerf", "renard", "loup", "hyene")


@sfx("wild_animal", "Wild animals: ours (bear growl), cerf (stag bell), renard (fox scream), loup (wolf howl), hyene (rising whoops).",
     kind=("ours", " | ".join(WILD)), seed=(0, "variation"))
def wild_animal(sr=DEFAULT_SR, kind="ours", seed=0):
    k = _kind(kind, WILD, "wild_animal")
    if k == "ours":
        y = C.growl(sr=P, size=2.2, aggression=0.75, pitch=65.0, dur=1.6, seed=seed)
    elif k == "cerf":
        y = _voice([(0, 90), (0.2, 300), (0.7, 240), (1, 90)], 2.4, size=2.2, rough=0.55, vowels="@-A-u", seed=seed, sub=0.2)
    elif k == "renard":
        y = _voice([(0, 900), (0.2, 1350), (1, 700)], 0.8, size=0.5, rough=0.8, vowels="ae-E", seed=seed, bright=2.0)
    elif k == "loup":
        y = C.howl(sr=P, size=1.0, pitch=420.0, dur=3.0, seed=seed)
    else:
        y = np.zeros(_n(2.6))
        for i in range(3):
            _put(y, _voice([(0, 320), (0.8, 900 + 90 * i), (1, 1000 + 90 * i)], 0.62, size=0.8, rough=0.2, vowels="u-A", seed=seed + i, rise=0.7), 0.05 + 0.85 * i)
    return _fin(_nm(y), sr, 0.7, fade=0.005, tail=0.03)


PRIMATES = ("singe", "gorille", "diane")


@sfx("primate", "Primates: singe (hoo-hoo rising to chatter), gorille (roar and chest beats), diane (male Diana monkey leopard alarm calls, formants from a four-tube tract).", kind=("singe", " | ".join(PRIMATES)), seed=(0, "variation"))
def primate(sr=DEFAULT_SR, kind="singe", seed=0):
    k = _kind(kind, PRIMATES, "primate")
    if k == "diane":
        y = C.diana_alarm(seed)
    elif k == "singe":
        y = np.zeros(_n(2.4))
        t = 0.02
        for i in range(7):
            d = 0.34 - 0.03 * i
            _put(y, _voice([(0, 350 + 90 * i), (0.6, 620 + 120 * i), (1, 450 + 100 * i)], d, size=0.55, rough=0.2 + 0.08 * i, vowels="u-A", seed=seed + i, rise=0.5), t, 0.5 + 0.07 * i)
            t += d + 0.03
        _put(y, _nm(REGISTRY["laugh"]["fn"](sr=P, voice="female", syllables=4, pitch=1.8, seed=seed)), t, 0.4)
    else:
        y = np.zeros(_n(3.2))
        _put(y, _nm(C.roar(sr=P, size=1.5, aggression=0.8, dur=1.6, seed=seed)), 0.0, 1.0)
        for i in range(9):                                   # chest beats
            _put(y, _nm(FO.thud(sr=P, body="sack", contact_ms=18.0, size=1.6, force=0.8, seed=seed + i)), 1.5 + 0.13 * i, 0.9 if i % 2 else 0.7)
    return _fin(y, sr, 0.75, fade=0.005, tail=0.03)


ELEPHANT = ("barrissement", "rumble", "combine")
RUMBLE_HZ = 20.0
ROAR_HZ = 470.0
TRUMPET_HZ = (420.0, 640.0, 480.0)


@sfx("elephant", "Elephant: barrissement (trumpet), rumble (20 Hz with its harmonics), combine (rumble grading into a roar, 20 to 470 Hz, over four octaves).",
     kind=("barrissement", " | ".join(ELEPHANT)), seed=(0, "variation"))
def elephant(sr=DEFAULT_SR, kind="barrissement", seed=0):
    k = _kind(kind, ELEPHANT, "elephant")
    if k == "barrissement":
        y = _voice([(0, TRUMPET_HZ[0]), (0.25, TRUMPET_HZ[1]), (1, TRUMPET_HZ[2])], 1.2, size=3.0, rough=0.55, vowels="ae-A", seed=seed, rise=0.12, bright=1.6)
    elif k == "rumble":
        y = _voice(_hz_pts(RUMBLE_HZ), 3.0, size=5.0, rough=0.1, vowels="u", seed=seed, rise=0.3, bright=0.3)
    else:
        y = _voice([(0, RUMBLE_HZ), (0.35, RUMBLE_HZ * 1.1), (0.9, ROAR_HZ), (1, ROAR_HZ)], 3.0, size=4.0, rough=0.45, vowels="u-@-A", seed=seed, rise=0.75, bright=1.2)
    return _fin(y, sr, 0.75, fade=0.01, tail=0.08)


# =============================================================================================
# WHALES
# =============================================================================================
WHALES = ("bleue", "bosse", "dauphin", "dauphin_sonar")
SONAR_CLICK = (117100.0, 37100.0)
SONAR_LAG = 0.02
SONAR_RANGE = 10.0
SONAR_SWING_DB = 20.0
SONAR_DOWN = 10.0
BLUE_CALL = [(0.0, 27.0), (9.0, 27.0), (10.0, 19.0), (18.0, 18.0)]
HUMPBACK = ((0.9, 70, 180), (0.7, 420, 240), (1.1, 95, 60), (0.5, 1500, 2400), (0.6, 3600, 2800), (1.2, 120, 300), (0.8, 900, 500))
DOLPHIN = ((0.35, 6000, 14000), (0.2, 14000, 9000), (0.5, 8000, 15000), (0.3, 12000, 5000))


def _sweep(segments, harmonics=(1.0, 0.3, 0.1), limit=4000.0) -> np.ndarray:
    """Tones along (s, Hz from, Hz to), each faded in and out; harmonics stop at ``limit``."""
    out = []
    for d, a, b in segments:
        n = _n(d)
        f = np.geomspace(a, b, n)
        ph = 2 * np.pi * np.cumsum(f) / P
        x = sum(g * np.sin((k + 1) * ph) * np.clip((limit - (k + 1) * f) / (0.1 * limit), 0.0, 1.0) for k, g in enumerate(harmonics))
        m = min(n // 2, _n(0.08))
        w = np.ones(n)
        w[:m] = np.sin(0.5 * np.pi * np.arange(m) / m) ** 2
        w[n - m:] = w[:m][::-1]
        out.append(x * w)
    return np.concatenate(out)


def _sonar_train(rng, dur: float) -> np.ndarray:
    """Echolocation clicks brought down by SONAR_DOWN in frequency and stretched by it in time: Gabor clicks,
    one per two-way travel to SONAR_RANGE plus SONAR_LAG, the interval swaying cyclically; the level rises
    from SONAR_SWING_DB below, holds, then falls."""
    f0, bw = (v / SONAR_DOWN for v in SONAR_CLICK)
    sig = np.sqrt(np.log(2.0)) / (np.pi * bw)
    tc = np.arange(-4 * sig, 4 * sig, 1.0 / P)
    click = np.exp(-tc ** 2 / (2 * sig ** 2)) * np.cos(2 * np.pi * f0 * tc + 0.75 * np.pi)
    gap = 2 * SONAR_RANGE / 1500.0 + SONAR_LAG
    times = []
    t = 0.01
    while t < dur - 0.05:
        times.append(t)
        t += gap * (1.0 + 0.15 * np.sin(2 * np.pi * len(times) / 9.0))
    y = np.zeros(_n(dur))
    m = max(len(times) - 1, 1)
    for i, t in enumerate(times):
        db = -SONAR_SWING_DB * (1.0 - np.sin(np.pi * i / m) ** 0.5) + rng.uniform(-2.5, 2.5)
        _put(y, click, t, 10 ** (db / 20.0))
    return y


def _in_water(y: np.ndarray, wet: float) -> np.ndarray:
    """Parallel: `fx.underwater` alone is a low-pass at most at 1800 Hz, which would remove the sourced top of the humpback and the whole dolphin."""
    w = np.asarray(underwater(y, P, amount=0.8), float)
    return (1.0 - wet) * y + wet * _fit(w, len(y))


@sfx("whale", "Whales: bleue (Southern Ocean blue whale: 27 Hz for 9 s, down to 19 Hz, then 18 Hz, 18 s, mostly felt), bosse (humpback song units, 40-4000 Hz), dauphin (FM whistles), dauphin_sonar (echolocation click train brought down from 117 kHz by 10).",
     kind=("bosse", " | ".join(WHALES)), wet=(0.3, "0..1 share through the underwater effect"), seed=(0, "variation"))
def whale(sr=DEFAULT_SR, kind="bosse", wet=0.3, seed=0):
    k = _kind(kind, WHALES, "whale")
    rng = np.random.default_rng(int(seed))
    wt = float(np.clip(wet, 0.0, 1.0))
    if k == "bleue":
        n = _n(BLUE_CALL[-1][0])
        f = np.interp(np.arange(n) / P, *zip(*BLUE_CALL))
        ph = 2 * np.pi * np.cumsum(f) / P
        y = (np.sin(ph) + 0.2 * np.sin(2 * ph) + 0.06 * np.sin(3 * ph)) * (1.0 + 0.1 * _slow(n, rng, 0.5))
        return _fin(_in_water(y, wt), sr, 0.8, fade=0.3, tail=0.8)
    if k == "dauphin_sonar":
        return _fin(_sonar_train(rng, 1.2), sr, 0.7, fade=0.002, tail=0.02)
    seg = list(HUMPBACK if k == "bosse" else DOLPHIN)
    order = rng.permutation(len(seg))
    if k == "bosse":
        y = amb.hip(_sweep([seg[i] for i in order]), 40.0)
        return _fin(_in_water(y, wt), sr, 0.8, fade=0.05, tail=0.1)
    y = _sweep([seg[i] for i in order], (1.0, 0.1), limit=0.45 * P)
    return _fin((1.0 - 0.5 * wt) * y + 0.5 * wt * np.roll(y, _n(0.03)) * 0.5, sr, 0.7, fade=0.02, tail=0.05)


# =============================================================================================
# SMALL THINGS: rodents, bats, reptiles, what slithers
# =============================================================================================
RODENTS = ("souris", "rat", "chauve_souris")
BAT_SWEEP = (28000.0, 22000.0)
BAT_DOWN = 10.0
BAT_RATE = 10.0


@sfx("rodent", "Small squeakers: souris (high chirps), rat (rougher squeaks), chauve_souris (bat searching: 28-22 kHz sweeps brought down from ultrasound by 10, and wings).",
     kind=("souris", " | ".join(RODENTS)), seed=(0, "variation"))
def rodent(sr=DEFAULT_SR, kind="souris", seed=0):
    """The ultrasound is brought down so it can be heard."""
    k = _kind(kind, RODENTS, "rodent")
    rng = np.random.default_rng(int(seed))
    if k == "chauve_souris":
        y = np.zeros(_n(1.3))
        a, b = BAT_SWEEP[0] / BAT_DOWN, BAT_SWEEP[1] / BAT_DOWN
        for i in range(int(1.2 * BAT_RATE)):
            _put(y, _sweep([(0.03, a, b)], (1.0,), limit=0.45 * P), 0.02 + i / BAT_RATE, rng.uniform(0.6, 1.0))
        _put(y, _nm(C.wings(sr=P, kind="bat", dur=1.2, seed=seed)), 0.0, 0.5)
        return _fin(y, sr, 0.7, fade=0.005, tail=0.02)
    if k == "rat":
        y = np.zeros(_n(1.2))
        t = 0.02
        for i in range(4):
            d = rng.uniform(0.09, 0.2)
            _put(y, _nm(_fit(squeak(sr=P, surface="hinge", pressure=1.5, speed=2.0, pitch=2.0, dur=d, seed=seed + i), _n(d))) * np.hanning(_n(d)), t)
            t += d + rng.uniform(0.05, 0.14)
        return _fin(y, sr, 0.7, fade=0.005, tail=0.02)
    y = np.zeros(_n(1.1))
    t = 0.02
    for i in range(6):
        d = rng.uniform(0.04, 0.09)
        c = _fit(chirp(sr=P, start=rng.uniform(4500, 6000), end=rng.uniform(3200, 4400), dur=d, wave="sine", warble=40.0, depth=1.0), _n(d))
        _put(y, c * np.hanning(_n(d)), t)
        t += d + rng.uniform(0.05, 0.11)
    return _fin(y, sr, 0.7, fade=0.005, tail=0.02)


REPTILES = ("serpent", "crocodile", "tortue")


@sfx("reptile", "Reptiles: serpent (hiss then rattle), crocodile (hiss over a low bellow), tortue (slow breathy hiss with a low grunt).",
     kind=("serpent", " | ".join(REPTILES)), seed=(0, "variation"))
def reptile(sr=DEFAULT_SR, kind="serpent", seed=0):
    """``hiss(kind=snake)`` plus a low layer."""
    k = _kind(kind, REPTILES, "reptile")
    if k == "serpent":
        y = np.zeros(_n(2.6))
        _put(y, _nm(C.hiss(sr=P, kind="snake", dur=1.2, seed=seed)), 0.0, 0.8)
        _put(y, _nm(C.rattlesnake(sr=P, dur=1.4, seed=seed)), 1.15, 1.0)
    elif k == "crocodile":
        y = np.zeros(_n(2.2))
        _put(y, _nm(C.growl(sr=P, size=3.0, aggression=0.7, pitch=38.0, dur=2.0, seed=seed)), 0.0, 1.0)
        _put(y, _nm(C.hiss(sr=P, kind="snake", size=2.5, dur=1.0, seed=seed)), 0.5, 0.5)
    else:
        y = np.zeros(_n(2.0))
        _put(y, _nm(C.hiss(sr=P, kind="snake", size=3.0, dur=1.8, seed=seed)), 0.0, 0.7)
        _put(y, _voice([(0, 90), (1, 70)], 0.5, size=1.6, rough=0.4, vowels="@-U", seed=seed), 0.4, 0.5)
    return _fin(y, sr, 0.7, fade=0.01, tail=0.05)


SLITHER = ("sable", "feuilles", "pierre")
SLITHER_HZ = 1.5


@sfx("slither", "A long body sliding: sable (fine grit), feuilles (leaves and cloth-like rustle), pierre (dry scales on rock). The force swells with each body wave.",
     kind=("sable", " | ".join(SLITHER)), rate=(SLITHER_HZ, "body waves per second"), dur=(3.0, "s"), seed=(0, "variation"))
def slither(sr=DEFAULT_SR, kind="sable", rate=SLITHER_HZ, dur=3.0, seed=0):
    """Friction with ``scrape``, the leaves with ``cloth`` and ``shake``."""
    k = _kind(kind, SLITHER, "slither")
    d = float(np.clip(dur, 0.3, 30.0))
    r = float(np.clip(rate, 0.2, 6.0))
    n = _n(d)
    wave = 0.55 + 0.45 * np.cos(2 * np.pi * r * np.arange(n) / P)
    tx, mat, sz, gr = {"sable": ("sandpaper", "wood", 0.5, 0.05), "feuilles": ("gritty", "wood", 0.3, 0.4), "pierre": ("gritty", "stone", 0.6, 0.15)}[k]
    y = _nm(_fit(K.scrape(sr=P, dur=d, speed=0.25, force=0.8, texture=tx, grain=gr, material=mat, size=sz, seed=seed), n))
    if k == "feuilles":
        y = 0.5 * y + _nm(_fit(FO.cloth(sr=P, kind="rustle", dur=d, rate=2.0 * r, weight=0.3, seed=seed), n))
    return _fin(amb.hip(y, 120.0) * wave, sr, 0.6, fade=0.05, tail=0.15)


# =============================================================================================
# SWARMS: crickets, flies and cicadas, without touching `creatures`
# =============================================================================================
def wing_det(n: int, rng, turn: bool = False) -> np.ndarray:
    """The second wing: a second slow noise (lop 5) moves the offset, size 1.004 as ``creatures._buzz``.
    ``turn``: a mosquito turning, one wing slows while the other speeds up."""
    det = 1.004 + 0.004 * _slow(n, rng, 5.0)
    if turn:
        c = int(rng.uniform(0.2, 0.8) * n)
        det = det + 0.03 * np.exp(-0.5 * ((np.arange(n) - c) / (0.08 * P)) ** 2)
    return det


def _wings(kind: str, n: int, rng, f0: float, wobble: float = 0.5) -> np.ndarray:
    """One insect, unit peak: table of `creatures._buzz`. No wandering in distance: the swarm's voices stand at their own place."""
    f = f0 * (1.0 + 0.2 * wobble * _slow(n, rng, 4.0))
    ph = 2 * np.pi * np.cumsum(f) / P
    det = wing_det(n, rng, kind == "mosquito")
    _f, band, duty = C._BUZZ[kind]
    kmax = int(min(0.45 * P, 12000.0) / float(np.max(f)))
    amps = C._FLY_H[:kmax] if kind == "fly" else np.sinc(np.arange(1, min(kmax, 40) + 1) * duty)
    y = np.zeros(n)
    for w, dt in ((0.0, 1.0), (1.3, det)):
        for i, a in enumerate(amps):
            y += a * np.sin((i + 1) * dt * ph + w)
    if band:
        y = 0.25 * _unit(y) + _unit(C._bpf(y, {"mosquito": 4200.0, "bee": 2600.0}[kind], 12.0))
    else:
        y = C._bpf(y, 2000.0, 0.3)
    return _unit(y)


def cricket_voice(n: int, rate: float, rng, bright: float = 0.5, phase0: float = 0.0) -> np.ndarray:
    """``creatures._cricket`` "field" with the fall: pulses towards the end of the chirp drop from 4.5 to 3.9 kHz."""
    fc, fall, pulses, ps, split, _k = C._CRICKETS["field"]
    t = np.arange(n) / P
    ps = min(ps, 0.8 / (rate * pulses))
    tc = np.mod(rate * t + phase0, 1.0) / rate
    k = np.floor(tc / ps)
    u = tc / ps - k
    on = k < pulses
    e = np.where(on, (k + 2.0) / (pulses + 2.0) * np.minimum(u / 0.3, 1.0) ** 2 * (1.0 - u) ** 1.5, 0.0)
    ph = 2 * np.pi * np.cumsum(fc * (1.0 - fall * np.where(on, (k + u) / pulses, 0.0))) / P
    y = sum(np.sin(w * ph) + bright / 3.0 * np.sin(2.0 * w * ph) for w in (1.0 - split, 1.0 + split))
    return y * e


CICADA_CALLS = {"chirp": (0.5, 128.0), "rattle": (8.0, 6.0), "whirr": (30.0, 3.0), "song": (25.0, 3.0)}
CICADA_EVERY = 5.0                                            # metro 5000: a new call every 5 s


def cicada_call_plan(n: int, call: str, rng) -> np.ndarray:
    """Index per 5 s block into the three patch calls; ``auto`` is the patch's random 3, a named call stays."""
    blocks = int(n / P / CICADA_EVERY) + 1
    if call == "auto":
        return rng.integers(0, 3, blocks)
    return np.full(blocks, list(CICADA_CALLS).index(call))


def cicada_voice(n: int, call: str, rng) -> np.ndarray:
    """Noise hip 5000 x2, lop 8000 x2 -> bp 5500 and 7500 Q40 x 0.7 -> AM 500 Hz -> pulse 1/(1 + x^2). `creatures.cicada` stays as it is."""
    x = rng.uniform(-1, 1, n)
    for _ in range(2):
        x = amb.lop(amb.hip(x, 5000.0), 8000.0)
    y = (C._bpf(x, 5500.0, 40.0) + 0.35 * C._bpf(x, 7500.0, 40.0)) * (1.0 + np.sin(2 * np.pi * 500.0 * np.arange(n) / P))
    return _unit(y * cicada_pulse(n, cicada_call_plan(n, call, rng)))


def cicada_pulse(n: int, plan) -> np.ndarray:
    """The 1/(1 + x^2) gate, per block of the plan."""
    names = list(CICADA_CALLS)
    p = np.zeros(n)
    block = int(CICADA_EVERY * P)
    for i, c in enumerate(plan):
        hz, w = CICADA_CALLS[names[c]]
        a = i * block
        m = min(block, n - a)
        p[a:a + m] = (1.0 / (1.0 + (w * np.cos(np.pi * hz * np.arange(a, a + m) / P)) ** 2))
    k = _n(0.01)
    return np.convolve(np.pad(p, k // 2, mode="edge"), np.ones(k) / k, "valid")[:n]


SWARM = {"abeilles": "bee", "mouches": "fly", "moustiques": "mosquito", "cigales": None, "grillons": None}
VOICES_MAX = 64


def swarm_voices(kind: str, count: int, n: int, seed: int, call: str = "auto") -> np.ndarray:
    """Sum of N decorrelated insects, unnormalised, each at its own wingbeat within +-6 % and level within -6 dB."""
    y = np.zeros(n)
    for i in range(int(np.clip(count, 1, VOICES_MAX))):
        rng = np.random.default_rng([abs(int(seed)), 124, i])
        if kind == "cigales":
            v = cicada_voice(n, call, rng)
        elif kind == "grillons":
            v = _unit(cricket_voice(n, C.dolbear_rate(18.0) * rng.uniform(0.9, 1.1), rng, phase0=rng.uniform()))
        else:
            b = SWARM[kind]
            f0 = C._BUZZ[b][0] * rng.uniform(0.94, 1.06)
            v = _wings(b, n, rng, f0)
        y += v * rng.uniform(0.5, 1.0)
    return y


@sfx("swarm", "Swarm of N decorrelated insects: abeilles (230 Hz wings), mouches (220 Hz), moustiques (600 Hz), cigales (Farnell's 3 calls, bands at 5.5 and 7.5 kHz), grillons (7-pulse chirps falling 4.5 to 3.9 kHz).",
     kind=("abeilles", " | ".join(SWARM)), count=(8, "insects, 1..64"), call=("auto", "cigales: auto (new call every 5 s) | chirp | rattle | whirr | song (25 Hz)"), dur=(3.0, "s"), seed=(0, "variation"))
def swarm(sr=DEFAULT_SR, kind="abeilles", count=8, call="auto", dur=3.0, seed=0):
    k = _kind(kind, SWARM, "swarm")
    if call != "auto" and call not in CICADA_CALLS:
        raise ValueError(f"swarm call {call!r}: use auto | {' | '.join(CICADA_CALLS)}")
    n = _n(float(np.clip(dur, 0.1, 60.0)))
    return _fin(swarm_voices(k, count, n, seed, call), sr, 0.6, fade=0.02, tail=0.1)


# =============================================================================================
# CREATURE STEPS AND THE HERD
# =============================================================================================
def stride_pace(weight: float, base: float = C.GAITS["walk"]) -> float:
    """Footfalls per minute: a heavier creature steps slower, as weight^-0.13."""
    return base * float(weight) ** -0.13


def _claw(seed: int) -> np.ndarray:
    """Short hard tick plus a scratch."""
    rng = np.random.default_rng([abs(int(seed)), 125])
    m = _n(0.06)
    s = amb.bp(rng.uniform(-1, 1, m), 3500.0, 1.5) * np.exp(-np.arange(m) / (0.012 * P))
    return s * np.sin(0.5 * np.pi * np.minimum(np.arange(m) / _n(0.001), 1.0)) ** 2


def creature_track(weight, steps, claws, ground, seed) -> tuple[np.ndarray, np.ndarray]:
    """`creatures._step` is not capped at 3 as `footsteps` is. Mono at P, and the footfall times."""
    w = float(np.clip(weight, 1.0, 20.0))
    cl = float(np.clip(claws, 0.0, 1.0))
    ioi = 60.0 / stride_pace(w)
    k = int(np.clip(steps, 1, 40))
    rng = np.random.default_rng([abs(int(seed)), 126])
    times = 0.02 + ioi * np.arange(k) + rng.normal(0.0, 0.015 * ioi, k).clip(-0.04 * ioi, 0.04 * ioi)
    out = np.zeros(_n(times[-1] + 1.5))
    body = 0.5 * np.log2(w) / np.log2(20.0)
    for i, t in enumerate(times):
        _put(out, _nm(C._step(ground, i, "walk", ioi, w, "barefoot", 0.5, seed)), float(t), 1.0)
        _put(out, _nm(FO.thud(sr=P, body="sack", contact_ms=30.0, size=1.0 + 0.2 * w, force=0.9, seed=seed + i)), float(t), body)
        if cl > 0:
            _put(out, _claw(seed + i), float(t) + 0.03, 0.5 * cl)
    return out, times


FOCUS = 2
HERD_HZ = 1.68


def herd_texture_times(count: int, dur: float, rng) -> np.ndarray:
    """Poisson footfalls of the rest of the herd: 4 per stride per animal."""
    rate = max(int(count) - FOCUS, 0) * 4.0 * HERD_HZ
    return np.sort(rng.uniform(0.0, dur, rng.poisson(rate * dur)))


def herd(count, dur, ground, seed) -> np.ndarray:
    """Focus + texture ("one, two, many"): two galloping animals rendered in full, the rest grains over a floor rumble. Mono at P."""
    n = _n(dur)
    y = np.zeros(n)
    pace = 4.0 * HERD_HZ * 60.0
    for i in range(FOCUS):
        steps = int(dur * 4.0 * HERD_HZ) + 1
        _put(y, _nm(C.footsteps(sr=P, ground=ground, gait="gallop", pace=pace, steps=steps, seed=seed + i)), 0.09 * i, 1.0 if i == 0 else 0.7)
    rng = np.random.default_rng([abs(int(seed)), 127])
    grains = [amb.lop(_nm(C._step(ground, i, "gallop", 60.0 / pace, 1.0, "hoof", 0.5, seed + i)), 1500.0) for i in range(6)]
    tx = np.zeros(n)
    for t in herd_texture_times(count, dur, rng):
        _put(tx, grains[int(rng.integers(len(grains)))], float(t), float(rng.uniform(0.2, 0.6)))
    bed = _fit(K.rumble(sr=P, dur=dur, fmin=30.0, fmax=90.0, seed=seed), n)
    g = np.sqrt(max(int(count) - FOCUS, 0) / 20.0)
    return y + min(g, 1.0) * (0.5 * _nm(tx) + 0.35 * _nm(bed))


CREATURE_KINDS = ("creature", "troupeau")


@sfx("creature_steps", "Heavy creature with claws (weight up to 20, slower and lower), or a galloping herd: troupeau, two animals in focus and the rest a texture.",
     kind=("creature", " | ".join(CREATURE_KINDS)), weight=(6.0, "creature: 1..20"), claws=(0.5, "creature: 0..1"), steps=(6, "creature: footfalls"),
     count=(30, "troupeau: animals"), dur=(4.0, "troupeau: s"), ground=("forest_floor", "any ground of `footsteps`"), seed=(0, "variation"))
def creature_steps(sr=DEFAULT_SR, kind="creature", weight=6.0, claws=0.5, steps=6, count=30, dur=4.0, ground="forest_floor", seed=0):
    k = _kind(kind, CREATURE_KINDS, "creature_steps")
    C._choice(ground, C.GROUNDS, "ground")
    if k == "creature":
        return _fin(creature_track(weight, steps, claws, ground, seed)[0], sr, 0.75, fade=0.002, tail=0.05)
    return _fin(herd(count, float(np.clip(dur, 0.5, 60.0)), ground, seed), sr, 0.75, fade=0.02, tail=0.2)
