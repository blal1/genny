"""Bodies and crowds: mouth and body noises, shouts, babies, crowd reactions, whispers, beatbox, yodel, rap.

The voices extend the presets of `physical.nonverbal` (`NV.PRESETS`). Screams carry an amplitude modulation inside the 30-150 Hz
window of rough vocalisations. A baby's cry is the male cry preset at twice its f0 (610 Hz, phonated range) with formants x1.6.
Sources of every value: research/SOURCES.md, section human. Nothing here has been listened to: see tests/test_human.py for what was measured.
"""
from __future__ import annotations

import math

import numpy as np

from . import choir as CH
from . import contact as K
from . import creatures as C
from . import foley as FO
from . import matter as M
from .core import DEFAULT_SR
from .creatures import _arc
from .drums import kick
from .fauna import _fit, _kind, _put, _voice
from .foley import P, _fin, _n, _nm
from .fx import speed
from .klang import strain, weight_shift
from .notes import to_hz
from .percussion import snap
from .physical import ambience as amb
from .physical import nonverbal as NV
from .physical import particles as PT
from .physical import voice as V
from .sfx import punch, sfx
from .speech import render_speech

ADULT_M, ADULT_F = 17.5, 13.5       # cm of vocal tract
INFANT_F = ADULT_F / 1.6            # formants x1.6
CHILD_F = ADULT_F / 1.28            # speech.VOICES["child"]
WOMAN = 1.7                         # f0 of the women in a crowd against the men


def _fmt(*vowels, k=1.0):
    """Formant anchors from `voice.FARNELL_VOWELS`, one vowel per anchor."""
    return tuple(tuple(V.FARNELL_VOWELS[v][i] * k for v in vowels) for i in range(3))


def _scaled(a, k):
    times, vals = NV._split(a)
    vals = tuple(v * k for v in vals)
    return (times, vals) if times is not None else vals


def _both(name, tract=ADULT_M, **p):
    """A preset for men, and the same for women (name_f)."""
    NV.PRESETS[(name, "female")] = dict(p, vocalTract=tract)
    NV.PRESETS[(name + "_f", "female")] = dict(p, vocalTract=ADULT_F, pitch=_scaled(p["pitch"], WOMAN))


_ROUGH = dict(amFreq=70, nonlinBalance=70, jitterDep=1, shimmerDep=10)         # 70 Hz: inside the 30-150 Hz roughness window; the rest as the scream preset
# Pitch contours and lengths of shouts and reactions; rolloff -6 = the scream preset, the pressed voice.
_both("h_war", sylLen=1300, pitch=((0, 0.15, 0.8, 1), (220, 420, 400, 260)), rolloff=-8, amDep=30, attackLen=(90, 200), formants=_fmt("A"), **_ROUGH)
_both("h_battle", sylLen=450, pitch=((0, 0.3, 1), (240, 360, 240)), rolloff=-8, amDep=25, attackLen=(90, 150), formants=_fmt("A"), **_ROUGH)
_both("h_cheer", sylLen=240, pauseLen=110, nSyl=3, pitch=(300, 380), pitchGlobal=((0, 1), (0, 3)), rolloff=-9, attackLen=(60, 90), ampl=(0, -6), formants=_fmt("A"))
NV.PRESETS[("h_howl", "female")] = {**NV.PRESETS[("scream", "female")], "sylLen": 1800, "amFreq": 70, "amDep": 50}
NV.PRESETS[("h_panic", "female")] = {**NV.PRESETS[("scream", "female")], "sylLen": 800, "amFreq": 70, "amDep": 50, "attackLen": (50, 150)}
_both("h_oh", sylLen=1100, pitch=((0, 0.3, 1), (150, 215, 120)), rolloff=-12, jitterDep=0.4, attackLen=(70, 300), formants=_fmt("U", "u"), temperature=0.06)
_both("h_ah", sylLen=1100, pitch=((0, 0.45, 1), (140, 240, 190)), rolloff=-12, jitterDep=0.4, attackLen=(90, 300), formants=_fmt("^", "A"), temperature=0.06)
_both("h_boo", sylLen=2400, pitch=((0, 0.2, 1), (135, 125, 100)), rolloff=-14, jitterDep=0.6, jitterLen=40, attackLen=(120, 350), formants=_fmt("u"), temperature=0.06)
_both("h_roar", sylLen=1900, pitch=((0, 0.3, 1), (230, 340, 260)), rolloff=-8, amDep=15, attackLen=(120, 500), formants=_fmt("A"), temperature=0.08, **_ROUGH)

_LAUGH_F = {**NV.PRESETS[("laugh", "female")], "attackLen": (25, 40), "noise": ((0, 15, 50, 70, 110), (-50, -15, -17, -59, -80))}
# the laugh preset starts the noise at -15 dB, a step on every syllable, and ends it at 186 ms, into the next syllable where the gate of `_vo` would stay open: 15 ms up, gone by 110 ms
NV.PRESETS[("h_baby_cry", "female")] = {**NV.PRESETS[("cry", "female")], "nSyl": 3, "sylLen": 950, "pauseLen": 330, "attackLen": (110, 350), "vocalTract": INFANT_F}     # f0 610 -> 511 = voice=male x 2
NV.PRESETS[("h_baby_laugh", "female")] = {**_LAUGH_F, "nSyl": 5, "pitch": (360, 332), "formants": tuple((f[0] * 1.6,) for f in _LAUGH_F["formants"])}   # laugh male x 2, female formants x 1.6
NV.PRESETS[("h_baby_coo", "female")] = dict(nSyl=5, sylLen=230, pauseLen=170, pitch=((0, 0.4, 1), (380, 520, 400)), rolloff=-15, attackLen=(40, 70), ampl=(0, -6), vocalTract=INFANT_F, temperature=0.12)
NV.PRESETS[("h_kid_laugh", "female")] = {**_LAUGH_F, "nSyl": 4, "sylLen": 90, "pitch": (430, 340), "formants": None, "vocalTract": CHILD_F}
NV.PRESETS[("h_kid_shout", "female")] = dict(sylLen=650, pitch=((0, 0.3, 1), (480, 720, 520)), rolloff=-9, attackLen=(50, 200), vocalTract=CHILD_F, temperature=0.08)
NV.PRESETS[("h_yawn", "female")] = dict(sylLen=2100, pitch=((0, 0.2, 1), (165, 235, 95)), rolloff=-16, jitterDep=0.4, jitterLen=50, attackLen=(250, 700), vocalTract=ADULT_M,
                                        formants=_fmt("A", "^", "U"), noise=((0, 1000, 2100), (-18, -24, -12)), rolloffNoise=-6, temperature=0.03)


def _vo(kind, rng, pitch=1.0):
    """`vocalize` cuts each syllable at its length after the STFT filter, a step at both ends where pre-echo and tail have nonzero level (`laugh` has them too).
    A gate closes the pauses: open above -60 dB, narrowed by all its 16 ms detector spreads, both ways, so the steps stay outside, 16 ms Hann transitions."""
    y = NV.vocalize(kind, "female", np.random.default_rng(int(rng.integers(2 ** 31))), pitch=pitch)
    k = _n(0.008)
    e = np.convolve(np.convolve(np.abs(y), np.ones(k) / k, "same"), np.ones(k) / k, "same")
    on = e > 1e-3 * e.max()
    on = on & np.roll(on, 2 * k) & np.roll(on, -2 * k)
    return y * np.convolve(on.astype(float), np.hanning(2 * k) / np.hanning(2 * k).sum(), "same")


def _voices(kinds, count, rng, spread=0.12):
    """Decorrelated people: each with its own random stream and pitch scale. kinds: preset names, drawn in turn with ~ one woman in three when `_f` exists."""
    out = []
    for i in range(int(count)):
        k = kinds[i % len(kinds)]
        if (k + "_f", "female") in NV.PRESETS and rng.random() < 1 / 3:
            k += "_f"
        out.append(_vo(k, rng, float(rng.uniform(1.0 - spread, 1.0 + spread))))
    return out


def _scatter(buf, ys, a, b, rng, g=(0.5, 1.0)):
    for y in ys:
        _put(buf, y, float(rng.uniform(a, b)), float(rng.uniform(*g)))


def _seed(rng) -> int:
    return int(rng.integers(2 ** 31))


# =============================================================================================
# Mouth and body
# =============================================================================================
@sfx("burp", "Burp: low pulses of the opening oesophagus through the vocal tract (40-120 Hz).",
     size=(1.0, "0.7..2.1: bigger = lower, longer (f0 85/size, kept in 40-120 Hz)"), seed=(0, "variation"))
def burp(sr=DEFAULT_SR, size=1.0, seed=0):
    s = float(np.clip(size, 0.7, 2.1))         # the contour stays inside 44-110: the source's jitter has to stay inside 40-120
    f = 85.0 / s
    pts = [(u, float(np.clip(f * k, 44.0, 110.0))) for u, k in ((0, 0.8), (0.2, 1.2), (0.7, 1.0), (1, 0.7))]
    return _fin(_voice(pts, 0.35 + 0.25 * s, size=1.0, rough=0.55, vowels="U-A-@", seed=seed, rise=0.2, bright=0.7), sr, 0.7, tail=0.03)


def _flap(dur, pts, jet_v, cav, seed, sputter=0.0, size=1.0, jet_g=0.25):
    """`matter.balloon`'s neck (relaxation valve gating a jet, into a cavity) lower down, with a smooth opening instead of its sawtooth, whose wrap would click."""
    rng = np.random.default_rng([abs(int(seed)), 131])
    n = _n(dur)
    f = np.clip(_arc(n, pts) / size * (1.0 + 0.06 * M._lfn(rng, n, 12.0)), 30.0, 120.0)
    ph = np.cumsum(f) / P % 1.0
    pulse = np.sin(np.pi * np.minimum(ph / 0.45, 1.0)) ** 2                     # open 45 % of the cycle, as `balloon`
    gate = 1.0 - sputter * (0.5 - 0.5 * np.tanh(2.0 * M._lfn(rng, n, 5.0)))     # the valve starves now and then; no clip: a corner in the gate shows
    jet = C._lp(M.jet_noise(n, rng, jet_v, 0.006), 1200.0) * pulse             # low-passed: noise gated at these rates is all peaks
    low = pulse - pulse.mean()
    y = jet_g * C._unit(jet) + C._unit(M._tvbp(low, cav, 2.5)) + 0.7 * C._unit(low)
    k = min(n // 2, _n(0.02))
    edge = np.minimum(np.minimum(np.arange(n), n - 1 - np.arange(n)) / k, 1.0)
    return y * gate * np.sin(0.5 * np.pi * edge) ** 2


_FARTS = {"reel": (1.1, [(0, 85), (0.3, 60), (1, 35)], 28.0, 240.0, 0.9),       # s, flutter Hz contour inside 30-120 Hz, jet, cavity Hz, sputter
          "whoopee": (1.4, [(0, 70), (0.6, 95), (1, 118)], 55.0, 480.0, 0.2)}


@sfx("fart", "Fart: lip valve like `balloon` but lower, flutter 30-120 Hz. reel (sputtering, falling), whoopee (rubber cushion, rising).",
     kind=("reel", " | ".join(_FARTS)), dur=(0.0, "s, 0 = from kind"), size=(1.0, "0.7..1.4: bigger = lower flutter"), seed=(0, "variation"))
def fart(sr=DEFAULT_SR, kind="reel", dur=0.0, size=1.0, seed=0):
    d, pts, jet, cav, sput = _FARTS[_kind(kind, _FARTS, "fart")]
    return _fin(_flap(dur or d, pts, jet, cav, seed, sput, float(np.clip(size, 0.7, 1.4))), sr, 0.7, fade=0.0005, tail=0.03)      # `_flap` has its opening edge


def _seal(ms, rng):
    """The rupture of a lip seal: a Hann pressure pulse of `ms` (1-5 ms) ringing in the mouth, 1.1-2.4 kHz."""
    n = _n(0.04)
    m = _n(ms * 1e-3)
    exc = np.zeros(n)
    exc[:m] = np.hanning(m) * rng.uniform(-1, 1, m)
    return C._unit(M._tvbp(exc, np.linspace(1100.0, 2400.0, n), 1.2))


KISSES = {"bise": (2.0, 0.08, 1), "smack": (4.0, 0.2, 1), "double": (2.0, 0.08, 2)}        # seal ms, suction level, count


@sfx("kiss", "Kiss: suction through the lips, then the seal breaking in 1-5 ms. bise, smack, double (both cheeks).",
     kind=("bise", " | ".join(KISSES)), seal=(0.0, "seal rupture in ms, 1..5; 0 = from kind"), seed=(0, "variation"))
def kiss(sr=DEFAULT_SR, kind="bise", seal=0.0, seed=0):
    ms, suck, cnt = KISSES[_kind(kind, KISSES, "kiss")]
    ms = float(np.clip(seal or ms, 1.0, 5.0))
    rng = np.random.default_rng([abs(int(seed)), 132])
    out = np.zeros(_n(0.2 + 0.42 * cnt))
    t = 0.1
    for _ in range(cnt):
        m = _n(0.07)
        s = C._bpf(rng.uniform(-1, 1, m), 2800.0, 1.5) * np.sin(np.pi * np.arange(m) / m) ** 2
        _put(out, C._unit(s), t - 0.07, suck)
        _put(out, _seal(ms, rng), t)
        t += 0.42
    return _fin(out, sr, 0.7, fade=0.002, tail=0.01)


@sfx("spit", "Spit: fricative 'ptt', then the drops landing after the time of the fall.",
     height=(1.0, "m the drops fall"), seed=(0, "variation"))
def spit(sr=DEFAULT_SR, height=1.0, seed=0):
    h = float(np.clip(height, 0.1, 5.0))
    fall = math.sqrt(2.0 * h / 9.81)
    rng = np.random.default_rng([abs(int(seed)), 133])
    out = np.zeros(_n(0.2 + fall + 0.5))
    y = C.hiss(sr=P, kind="cat", dur=0.2, seed=seed)
    _put(out, y, 0.01)
    t0 = 0.01 + 0.1 + fall                                                      # from the middle of the jet
    _put(out, M.drip(sr=P, size=0.15, bubbles=2.0, secondary=0.0, click=0.5, seed=seed), t0, 0.5)
    _put(out, M.drip(sr=P, size=0.08, bubbles=1.0, secondary=0.0, click=0.5, seed=seed + 1), t0 + rng.uniform(0.02, 0.06), 0.3)
    return _fin(out, sr, 0.7, fade=0.002, tail=0.03)


@sfx("vomit", "Vomit: two retching efforts (`strain`), then voice over a splash and gurgles.",
     size=(0.5, "0..1 how much comes out"), seed=(0, "variation"))
def vomit(sr=DEFAULT_SR, size=0.5, seed=0):
    s = float(np.clip(size, 0.0, 1.0))
    out = np.zeros(_n(3.0))
    retch = lambda d, f, s: strain(sr=P, dur=d, freq=f, seed=s)
    _put(out, retch(0.4, 110.0, seed), 0.02, 0.55)
    _put(out, retch(0.55, 120.0, seed + 1), 0.6, 0.8)
    t = 1.2
    _put(out, _voice([(0, 130), (1, 85)], 0.9, rough=0.7, vowels="A-U", seed=seed, rise=0.15, bright=0.6), t, 0.65)
    _put(out, M.splash(sr=P, size=0.2 + 0.4 * s, dur=1.3, droplets=6, seed=seed), t + 0.05, 0.9)
    _put(out, M.gurgle(sr=P, dur=1.0, rate=6.0, size=0.2, seed=seed), t + 0.1, 0.45)
    return _fin(out, sr, 0.75, tail=0.05)


@sfx("pee", "Pee: thin stream into water (`pour` at low flow, level hardly changing), then the last spurts.",
     dur=(4.0, "s"), flow=(0.25, "0.1..0.4 flow rate of `pour`"), seed=(0, "variation"))
def pee(sr=DEFAULT_SR, dur=4.0, flow=0.25, seed=0):
    d = float(np.clip(dur, 1.0, 30.0))
    y = M.pour(sr=P, dur=d, flow=float(np.clip(flow, 0.1, 0.4)), fill_start=0.5, fill_end=0.55, height=0.18, radius=0.1, material="ceramic", resonance=0.25, seed=seed)
    y = np.asarray(y, float)
    t = np.arange(len(y)) / P
    # the stream fails over the last 0.9 s in three spurts
    end = np.clip((d - t) / 0.9, 0.0, 1.0)
    spurts = end + (1.0 - end) * end * np.sin(np.pi * np.minimum(d - t, 0.9) * 3.0 / 0.9) ** 2
    return _fin(y * spurts, sr, 0.6, tail=0.05)


@sfx("yawn", "Yawn: long inward pull, a held voiced vowel rising then falling, breath out.",
     size=(1.0, "0.7..1.5 body scale"), seed=(0, "variation"))
def yawn(sr=DEFAULT_SR, size=1.0, seed=0):
    s = float(np.clip(size, 0.7, 1.5))
    rng = np.random.default_rng([abs(int(seed)), 134])
    out = np.zeros(_n(3.9))
    _put(out, C._breath(1.0, "in", rng, s, 0.5, rise=0.75), 0.02, 0.5)
    _put(out, _vo("h_yawn", rng, 1.0 / s), 0.85, 1.0)
    _put(out, C._breath(0.8, "out", rng, s, 0.3, rise=0.1), 2.95, 0.35)
    return _fin(out, sr, 0.6, tail=0.05)


@sfx("gargle", "Gargling: voice through liquid in the mouth, trilling and bubbling at 10-20 Hz.",
     rate=(15.0, "Hz, 10..20"), dur=(2.0, "s"), seed=(0, "variation"))
def gargle(sr=DEFAULT_SR, rate=15.0, dur=2.0, seed=0):
    r = float(np.clip(rate, 10.0, 20.0))
    d = float(np.clip(dur, 0.5, 20.0))
    rng = np.random.default_rng([abs(int(seed)), 135])
    out = np.zeros(_n(d))
    _put(out, _voice([(0, 125), (1, 110)], d, rough=0.4, vowels="^", seed=seed, rise=0.1, am=(r, 0.9), bright=0.7), 0.0)
    t = 0.03
    while t < d - 0.1:
        b = C._unit(amb.tube_bubble(rng.uniform(3e-3, 6e-3), rng))             # 3-6 mm: 0.5-1.1 kHz
        b[:_n(0.002)] *= np.sin(0.5 * np.pi * np.arange(_n(0.002)) / _n(0.002)) ** 2                       # the mouth is already moving: no impact starts them
        _put(out, b, t, 0.35 * rng.uniform(0.7, 1.0))
        t += 1.0 / r
    return _fin(C._lp(out, 3500.0), sr, 0.7, tail=0.05)     # the mouth takes the shape modes of a 3-6 mm bubble (6 to 12 detections at -30 dB without it, each a real onset)


HICCUP = 0.05       # s


def _hic(size, rng):
    """50 ms of voice (f0 340 Hz) and intake, cut by the glottis closing in 3 ms. The closure comes after the filters: nothing rings past it."""
    n = _n(HICCUP)
    nn = _n(HICCUP) + _n(0.02)
    f0 = 340.0 / size * np.linspace(0.85, 1.15, nn)
    v = V.formant_filter(V.lf_pulse(f0, 0.8, rng=rng), "@", "farnell", formant_scale=1.0 / size)
    a = C._hp(rng.uniform(-1, 1, nn), 1500.0)
    y = (C._unit(v) + 0.3 * C._unit(a))[-n:]
    e = np.ones(n)
    m, c = _n(0.006), _n(0.003)
    e[:m] = np.sin(0.5 * np.pi * np.arange(m) / m) ** 2
    e[n - c:] = np.cos(0.5 * np.pi * np.arange(c) / c) ** 2
    return y * e


@sfx("hiccup", "Hiccup: a 50 ms voiced intake cut by the glottis slamming shut.",
     count=(3, "hiccups"), gap=(1.4, "s between them"), size=(1.0, "0.7..1.5 body scale"), seed=(0, "variation"))
def hiccup(sr=DEFAULT_SR, count=3, gap=1.4, size=1.0, seed=0):
    rng = np.random.default_rng([abs(int(seed)), 136])
    c, g = int(np.clip(count, 1, 30)), float(np.clip(gap, 0.2, 10.0))
    s = float(np.clip(size, 0.7, 1.5))
    out = np.zeros(_n(0.04 + c * g))
    for i in range(c):
        FO._put(out, _hic(s, rng), 0.02 + i * g * rng.uniform(0.97, 1.03), rng.uniform(0.8, 1.0))
    return _fin(out[:_n(0.02 + (c - 1) * g * 1.03 + 0.1)], sr, 0.7, fade=0.002)


REBOUND = 0.5       # on top of force 0.3 against 0.9


@sfx("bodyfall", "A body falling: cloth, a long dull contact (`thud` 15 ms) on a ground of `footsteps`, a weak rebound, settling.",
     weight=(1.0, "0.5..2"), ground=("wood", "|".join(C.GROUNDS)), seed=(0, "variation"))
def bodyfall(sr=DEFAULT_SR, weight=1.0, ground="wood", seed=0):
    if ground not in C.GROUNDS:
        raise ValueError(f"bodyfall ground {ground!r}: {'|'.join(C.GROUNDS)}")
    w = float(np.clip(weight, 0.5, 2.0))
    out = np.zeros(_n(1.3))
    t = 0.2
    _put(out, FO.cloth(sr=P, kind="rustle", dur=0.3, rate=8.0, weight=0.7, seed=seed), 0.0, 0.2)
    hit = lambda s, f: FO.thud(sr=P, body="sack", contact_ms=15.0, size=0.8 * w ** 0.5, force=f, seed=s)
    _put(out, hit(seed, 0.9), t)
    _put(out, C._step(ground, 0, "walk", 0.6, w, "barefoot", 0.5, seed), t, 0.5)
    _put(out, hit(seed + 1, 0.3), t + 0.13, REBOUND)
    _put(out, FO.cloth(sr=P, kind="rustle", dur=0.35, rate=5.0, weight=0.7, seed=seed + 1), t + 0.2, 0.16)
    if ground in PT.GROUNDS:
        _put(out, weight_shift(sr=P, ground=ground, dur=0.45, weight=w, seed=seed), t + 0.25, 0.25)
    return _fin(out, sr, 0.8, tail=0.05)


# =============================================================================================
# Voices
# =============================================================================================
SHOUTS = {"cri_de_guerre": "h_war", "hurlement": "h_howl", "encouragement": "h_cheer"}


@sfx("shout", "Shout (the nonverbal scream engine): cri de guerre, hurlement, encouragement. people > 1: a group, each voice its own.",
     kind=("cri de guerre", "cri de guerre | hurlement | encouragement"), people=(1, "voices, 1..24"), pitch=(1.0, "f0 scale"), seed=(0, "variation"))
def shout(sr=DEFAULT_SR, kind="cri de guerre", people=1, pitch=1.0, seed=0):
    k = SHOUTS[_kind(kind, SHOUTS, "shout")]
    rng = np.random.default_rng([abs(int(seed)), 137])
    p = int(np.clip(people, 1, 24))
    if p == 1:
        return _fin(_vo(k, rng, float(np.clip(pitch, 0.5, 2.0))), sr, 0.7, tail=0.02)
    ys = _voices([k], p, rng)
    out = np.zeros(max(map(len, ys)) + _n(0.2))
    _scatter(out, ys, 0.0, 0.15, rng)
    return _fin(out, sr, 0.7, tail=0.02)


BABY = {"pleurs": "h_baby_cry", "rire": "h_baby_laugh", "gazouillis": "h_baby_coo"}


@sfx("baby", "Baby (the nonverbal engine, short tract): pleurs, rire, gazouillis. F0 x2, formants x1.6 of the adult preset.",
     kind=("pleurs", " | ".join(BABY)), pitch=(1.0, "f0 scale"), seed=(0, "variation"))
def baby(sr=DEFAULT_SR, kind="pleurs", pitch=1.0, seed=0):
    k = BABY[_kind(kind, BABY, "baby")]
    return _fin(_vo(k, np.random.default_rng([abs(int(seed)), 138]), float(np.clip(pitch, 0.7, 1.3))), sr, 0.7, tail=0.02)


BLOWS = ("bougie", "ballon", "souffle")
BALLOON_PUFFS = 3


@sfx("blow", "Blowing: bougie (short puff), ballon (puffs into a neck, the cavity falling as it fills), souffle (long steady breath).",
     kind=("bougie", " | ".join(BLOWS)), size=(1.0, "0.7..1.5 body scale"), seed=(0, "variation"))
def blow(sr=DEFAULT_SR, kind="bougie", size=1.0, seed=0):
    k = _kind(kind, BLOWS, "blow")
    s = float(np.clip(size, 0.7, 1.5))
    rng = np.random.default_rng([abs(int(seed)), 139])
    if k == "bougie":
        return _fin(C._breath(0.35, "out", rng, s, 1.0, rise=0.1), sr, 0.6, tail=0.03)
    if k == "souffle":
        return _fin(C._breath(1.8, "out", rng, s, 0.7, rise=0.25), sr, 0.5, tail=0.05)
    out = np.zeros(_n(3.2))
    for i in range(BALLOON_PUFFS):
        # `matter.balloon`'s cavity law at size 0.5, read the other way: x, how empty, goes down as it fills
        x = 1.0 - (i + 1) / (BALLOON_PUFFS + 1)
        y = C._breath(0.6, "out", rng, s, 0.8, rise=0.2)
        cav = M._tvbp(y, 350.0 + 900.0 * x ** 2, 4.0)
        _put(out, 0.35 * y + C._unit(cav), 0.02 + i * 0.95, 0.8 + 0.1 * i)
    return _fin(out, sr, 0.6, tail=0.05)

CARTOONS = {"aigu": 1.5, "ecureuil": 2.0, "souris": 2.5}        # tape speeds


@sfx("cartoon_voice", "Cartoon voice: speech played fast, pitch and tempo together. aigu x1.5, ecureuil x2, souris x2.5.",
     kind=("ecureuil", " | ".join(CARTOONS)), text=("hello there", "what it says"), seed=(0, "variation"))
def cartoon_voice(sr=DEFAULT_SR, kind="ecureuil", text="hello there", seed=0):
    f = CARTOONS[_kind(kind, CARTOONS, "cartoon_voice")]
    y = render_speech(str(text), sr=P, voice="male", seed=int(seed), intonation="flat")
    # `speed` interpolates linearly: band-limit first, or what passes 24 kHz folds back
    return _fin(speed(C._lp(np.asarray(y, float), 0.45 * P / f), P, factor=f), sr, 0.7, fade=0.002, tail=0.01)


@sfx("finger_snap", "Finger snap, the `snap` drum as a sound effect.", vel=(1.0, "0..1"))
def finger_snap(sr=DEFAULT_SR, vel=1.0):
    return snap(sr=sr, vel=float(np.clip(vel, 0.0, 1.0)))


SKIN = ("grattage", "tape", "poignee")


@sfx("skin", "Skin: grattage (nails, strokes of `scrape`), tape (pats, cupped `handclap`), poignee (handshake: slap, rub, cloth).",
     kind=("grattage", " | ".join(SKIN)), rate=(4.0, "scratches or pats per second"), count=(4, "strokes or pats"), seed=(0, "variation"))
def skin(sr=DEFAULT_SR, kind="grattage", rate=4.0, count=4, seed=0):
    k = _kind(kind, SKIN, "skin")
    r, c = float(np.clip(rate, 1.0, 8.0)), int(np.clip(count, 1, 40))
    rng = np.random.default_rng([abs(int(seed)), 140])
    if k == "grattage":
        d = c / r
        speeds = [0.02] + [v for _ in range(c) for v in (0.5, 0.02)]               # 0.5 m/s, soft body
        y = K.scrape(sr=P, dur=d, speed=speeds, force=[0.0] + [v for _ in range(c) for v in (0.6, 0.0)], texture="pink", grain=0.05, body="circ_membrane",
                     material="rubber", size=0.3, bright=0.7, seed=seed)
        y = C._lp(C._hp(y, 1000.0), 6000.0)
        m = _n(0.03)
        y[:m] *= np.sin(0.5 * np.pi * np.arange(m) / m) ** 2
        return _fin(y, sr, 0.5, tail=0.02)      # nails on skin, no body under them
    if k == "tape":
        out = np.zeros(_n(c / r + 0.3))
        for i in range(c):
            _put(out, C.handclap(sr=P, cup=0.95, seed=seed + i), 0.01 + i / r, rng.uniform(0.7, 1.0))
        return _fin(out, sr, 0.6, fade=0.002, tail=0.03)
    out = np.zeros(_n(1.1))
    _put(out, C.handclap(sr=P, cup=0.7, seed=seed), 0.01, 0.8)
    y = K.scrape(sr=P, dur=0.5, speed=[0.02, 0.15, 0.02, 0.15, 0.02], force=[0.0, 0.5, 0.2, 0.5, 0.0], texture="pink", body="circ_membrane", material="rubber", size=0.3, bright=0.3, seed=seed)
    m = _n(0.03)
    rub = _fit(y, _n(0.8)) * 0.3 + 0.25 * _fit(FO.cloth(sr=P, kind="rustle", dur=0.8, rate=3.0, weight=0.3, seed=seed), _n(0.8))
    rub[:m] *= np.sin(0.5 * np.pi * np.arange(m) / m) ** 2                      # under the slap's decay: no edge of their own
    _put(out, rub, 0.1)
    return _fin(out, sr, 0.6, fade=0.002, tail=0.03)


WHISPERS = ("are you ready", "do not go there", "they are coming", "over here", "what was that", "keep it down", "the time is over", "stay low", "come back", "who is there")   # words of the speech dictionary


@sfx("whisper_bed", "Bed of whispers: several unvoiced speakers, each their own phrases, tract and place.",
     voices=(6, "speakers, 1..16"), dur=(4.0, "s"), seed=(0, "variation"))
def whisper_bed(sr=DEFAULT_SR, voices=6, dur=4.0, seed=0):
    rng = np.random.default_rng([abs(int(seed)), 141])
    d = float(np.clip(dur, 1.0, 30.0))
    out = np.zeros(_n(d))
    for _ in range(int(np.clip(voices, 1, 16))):
        t = rng.uniform(-0.8, 0.8)
        while t < d - 0.4:
            y = np.asarray(render_speech(str(rng.choice(WHISPERS)), sr=P, voice="whisper", seed=_seed(rng), formant_shift=float(rng.uniform(0.9, 1.2)), rate=float(rng.uniform(0.9, 1.15))), float)
            a = max(0.0, -t)
            _put(out, y[int(a * P):], max(t, 0.0), float(rng.uniform(0.5, 1.0)))
            t += len(y) / P + rng.uniform(0.3, 1.6)
    return _fin(C._hp(out, 300.0), sr, 0.5, tail=0.3)


# =============================================================================================
# Crowds
# =============================================================================================
CROWD = ("oh", "ah", "huee", "ovation", "panique", "chants", "enfants", "combat")
CHANT = ((0.0, 62, 0.5), (0.5, 64, 0.5), (1.0, 66, 1.0))        # three notes of "o-le-o", MIDI; one cycle per 2 s


def _chant(voices, d, seed) -> np.ndarray:
    k = int(d // 2) + 1
    notes = [(2.0 * i + t, m, l, 1.0) for i in range(k) for t, m, l in CHANT]
    return CH.sing_phrase(notes, P, stereo=False, lyrics=" ".join(["o-le-o"] * k), section="men", singers=min(voices, 12), style="pop", mode="shout", dynamics=0.8, seed=seed)


def crowd_track(k, v, d, seed):
    """The mix at P, and (start, seconds) of what is meant to strike: blades, blows, whistles."""
    rng = np.random.default_rng([abs(int(seed)), 142])
    hits = []

    def hit(y, g):
        t = float(rng.uniform(0.0, max(0.1, d - len(y) / P)))
        _put(out, y, t, g)
        hits.append((t, len(y) / P))

    if k in ("oh", "ah"):
        ys = _voices(["h_" + k], v, rng)
        out = np.zeros(max(map(len, ys)) + _n(0.3))
        _scatter(out, ys, 0.0, 0.2, rng)                    # 200 ms to react together
        return C._lp(out, 5000.0), hits                     # a crowd is far: without this, 4 detections at -57 dB on a first glottal pulse in near-silence (ah, seed 1)
    out = np.zeros(_n(d))
    if k == "huee":
        for _ in range(max(1, int(round(d / 2.6)))):
            _scatter(out, _voices(["h_boo"], v, rng), 0.0, max(0.1, d - 2.2), rng)
        for i in range(max(1, v // 6)):
            hit(C.whistle_human(sr=P, kind=("down", "note")[i % 2], freq=float(rng.uniform(1600, 2400)), seed=_seed(rng)), 0.25)
    elif k == "ovation":
        _put(out, _fit(C.applause(sr=P, crowd=v * 5, enthusiasm=0.9, dur=d, seed=seed), _n(d)), 0.0, 0.9)
        _scatter(out, _voices(["h_roar"], v // 2 + 1, rng), 0.0, max(0.1, d - 1.9), rng, (0.2, 0.5))
        for i in range(max(1, v // 8)):
            hit(C.whistle_human(sr=P, kind="wolf", freq=float(rng.uniform(1500, 2200)), seed=_seed(rng)), 0.15)
    elif k == "panique":
        _scatter(out, _voices(["h_panic"], int(np.ceil(v * d / 3.0)), rng, spread=0.2), 0.0, max(0.1, d - 0.8), rng)     # each person screams every 3 s
        for _ in range(3):
            y = C.footsteps(sr=P, ground="concrete", gait="run", steps=max(2, int(d * 2.8)), seed=_seed(rng))
            t = float(rng.uniform(0.0, 0.3))
            _put(out, _fit(y, _n(d)), t, 0.25)
            hits.append((t, d))
    elif k == "chants":
        out = _fit(_chant(v, d, seed), _n(d))
    elif k == "enfants":
        _scatter(out, _voices(["h_kid_laugh", "h_kid_shout"], int(np.ceil(v * d / 3.0)), rng, spread=0.15), 0.0, max(0.1, d - 0.7), rng)
    else:
        c = int(np.ceil(v * d / 3.0))
        _scatter(out, _voices(["h_battle"], c, rng), 0.0, max(0.1, d - 0.5), rng, (0.4, 0.8))
        for _ in range(c):
            _put(out, C.grunt(sr=P, pitch=float(rng.uniform(90, 140)), effort=0.6, dur=0.25, seed=_seed(rng)), float(rng.uniform(0.0, max(0.1, d - 0.3))), 0.5)
        for _ in range(max(2, v // 3)):
            hit(FO.clash(sr=P, force=float(rng.uniform(0.5, 1.0)), seed=_seed(rng)), 0.35)
            hit(punch(sr=P), 0.4)
    return out, hits


@sfx("crowd", "Crowd reaction, N decorrelated voices: oh, ah (one reaction, about 1.3 s), huee (boos and whistles), ovation (cheers on applause), panique (screams, running), chants (unison shouting), enfants (playground), combat (shouts, grunts, blades, blows).",
     kind=("oh", " | ".join(CROWD)), voices=(20, "people, 1..24 (chants: 12 at most)"), dur=(4.0, "s, for all but oh and ah"), seed=(0, "variation"))
def crowd(sr=DEFAULT_SR, kind="oh", voices=20, dur=4.0, seed=0):
    k = _kind(kind, CROWD, "crowd")
    y, _ = crowd_track(k, int(np.clip(voices, 1, 24)), float(np.clip(dur, 1.0, 30.0)), seed)
    return _fin(y, sr, 0.7, tail=0.03 if k in ("oh", "ah") else 0.1)


# =============================================================================================
# Vocal performance
# =============================================================================================
def _stop(ph, seed) -> np.ndarray:
    """One speech consonant, its silence cut off."""
    y = np.asarray(render_speech(phonemes=ph, sr=P, seed=seed), float)
    e = np.abs(y)
    on = np.flatnonzero(e > 0.03 * e.max())
    y = y[max(0, on[0] - _n(0.001)):on[-1] + _n(0.004)]
    m = min(len(y) // 2, _n(0.004))
    y[-m:] *= np.cos(0.5 * np.pi * np.arange(m) / m) ** 2
    y[:_n(0.0005)] *= np.linspace(0.0, 1.0, _n(0.0005))
    return C._unit(y)


@sfx("beatbox", "Beatbox: stops in a pattern. B = lip kick (speech P + `kick`), t = hat (T S), K = snare (K SH), - = rest.",
     pattern=("B t K t B B K t", "tokens B t K -, one per eighth"), bpm=(100.0, "beats per minute"), bars=(1, "times through the pattern"), seed=(0, "variation"))
def beatbox(sr=DEFAULT_SR, pattern="B t K t B B K t", bpm=100.0, bars=1, seed=0):
    toks = str(pattern).split()
    bad = sorted(set(toks) - set("BtK-"))
    if bad or not toks:
        raise ValueError(f"beatbox pattern: tokens B t K -, got {bad or 'nothing'}")
    step = 30.0 / float(np.clip(bpm, 40.0, 240.0))
    b = int(np.clip(bars, 1, 32))
    rng = np.random.default_rng([abs(int(seed)), 143])
    out = np.zeros(_n(0.02 + len(toks) * b * step + 0.4))
    for i, tok in enumerate(toks * b):
        s = _seed(rng) % 1000
        if tok == "-":
            continue
        t = 0.02 + i * step
        if tok == "B":
            k = np.array(kick(sr=P, tune=70, decay=0.14, punch=0.7), float)         # the drum stops at its own length, above zero
            m = _n(0.03)
            k[-m:] *= np.cos(0.5 * np.pi * np.arange(m) / m) ** 2
            _put(out, k, t, 1.0)
            _put(out, _stop("P", s), t, 0.2)
        elif tok == "t":
            _put(out, C._hp(_stop("T S", s), 2500.0), t, 0.4)
        else:
            _put(out, _stop("K", s), t, 0.5)
            _put(out, C._hp(_stop("SH", s), 800.0), t + 0.005, 0.7)
    return _fin(out, sr, 0.8, fade=0.0005, tail=0.03)


YODEL = (("G3", 0), ("D4", 1), ("B3", 0), ("G4", 1), ("D4", 0), ("B4", 1), ("G4", 1), ("D4", 0), ("G3", 0))    # chest 0, falsetto 1


@sfx("yodel", "Yodel: leaps across the chest-falsetto break, chest on 'oh', falsetto on 'ee', with the short drop of the break between them.",
     note=(0.28, "s per note"), transpose=(0, "semitones"), seed=(0, "variation"))
def yodel(sr=DEFAULT_SR, note=0.28, transpose=0, seed=0):
    d = float(np.clip(note, 0.12, 1.0))
    xf = 0.03                                               # s of the break
    k = 2.0 ** (float(transpose) / 12.0)
    out = np.zeros(_n(len(YODEL) * d + 0.3))
    n = _n(d + xf)
    m = _n(xf)
    w = np.ones(n)
    w[:m] = np.sin(0.5 * np.pi * np.arange(m) / m) ** 2
    w[-m:] = w[:m][::-1]
    for i, (nt, reg) in enumerate(YODEL):
        y = CH._note(to_hz(nt) * k, d + xf + 0.1, P, 0.8, vowel="ee" if reg else "oh", voice="tenor", mode="falsetto" if reg else "sing", vibrato=0.0, style="pop", exact=True, seed=int(seed) + i)
        FO._put(out, _fit(y, n) * w, 0.02 + i * d)
    return _fin(out, sr, 0.75, tail=0.04)


@sfx("rap", "Rap: speech with its words set on a grid of eighth notes, level pitch.",
     text=("time to go over the level now", "words"), bpm=(95.0, "beats per minute"), voice=("male", "any speech voice"), seed=(0, "variation"))
def rap(sr=DEFAULT_SR, text="time to go over the level now", bpm=95.0, voice="male", seed=0):
    words = str(text).split()
    if not words:
        raise ValueError("rap needs text")
    step = 30.0 / float(np.clip(bpm, 50.0, 200.0))
    out = np.zeros(_n(0.02 + len(words) * step + 1.0))
    for i, w in enumerate(words):
        y = np.asarray(render_speech(w, sr=P, voice=voice, seed=int(seed) + i, intonation="flat", rate=1.3), float)
        on = np.flatnonzero(np.abs(y) > 0.05 * np.abs(y).max())[0]         # the consonant lands on the beat, not the padding
        m = min(on, _n(0.004))
        y = y[on - m:].copy()
        y[:m] *= np.linspace(0.0, 1.0, m)
        _put(out, y, 0.02 + i * step - m / P, 1.0 if i % 2 == 0 else 0.8)
    return _fin(out, sr, 0.75, fade=0.001, tail=0.04)
