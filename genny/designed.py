"""Designed sounds, signals, toys: cartoon, magic, creatures, radio, modem, morse, alarms, CRT, toys and the books' oddities.

- Morse: t_dit = 60 / (50 wpm), 1:3:1:3:7, "PARIS " = 50 units.
- Modem: Bell 103 1070/1270, 2025/2225 Hz, 300 bit/s; ANS 2100 Hz; V.21 980/1180, 1650/1850 Hz;
  fax CNG 1100 Hz 0.5 s on 3 s off, CED 2100 Hz for 2.6-4 s.
- Alarms: four-phase generator, cos(cos wt), sin(cos wt), DC trap.
- Phone: line clip 0.9, BP 2000 Q12, HP 90 x2, BP 400 Q3 beside clip 0.4; pulse dial 100 ms, 40 ms contact.
- Can: modes 359, 426, 1748, 3150 Hz; rolling (k, a).
- Stingers: rising minor second = threat, falling minor third = sadness.
- Pendulum Music gate |A exp(-t/tau) cos(2 pi t/T)| < width.
Sources: ``research/SOURCES.md``. Nothing here has been listened to: see tests/test_designed.py for what was measured.
"""
from __future__ import annotations

import math
import os
import tempfile

import numpy as np

from .contact import impact
from .core import DEFAULT_SR, write_wav
from .choir import voice
from .creatures import growl, hiss, monster, roar, wings
from .dsp import noise_bed
from .fdstring import twang
from .foley import (C_AIR, P, _fin, _n, _nm, _put, boing, camera_shutter, charge_up, dark_drone, edge_tone, ratchet, servo, teleport,
                    thud)
from .friction import rod_bank, squeak
from .fx import distortion
from .instruments import brass, music_box
from .klang import ui_tick
from .matter import balloon, bubbles, fire, gurgle, ice_crack
from .cyber import transformer
from .physical import ambience as amb
from .pinball import chirp, solenoid, spring
from .reverbs import feedback_guitar, reverse_reverb, room, shimmer
from .road import _band, _fit, _kind, _modes
from .sfx import REGISTRY, click, electric_motor, explosion, magic, pop, render_sfx, riser, sfx, static, vinyl, whoosh, wind
from .spectral import cross_synth
from .speech import render_speech
from .tubes import bottle, formant_tract, reed_tube


def _m(y) -> np.ndarray:
    """Mono float."""
    y = np.asarray(y, float)
    return y.mean(axis=1) if y.ndim == 2 else y


def _sine(f, n: int) -> np.ndarray:
    return np.sin(2 * np.pi * np.cumsum(np.broadcast_to(np.asarray(f, float), (n,))) / P)


def _ramp(n: int, xs, ys) -> np.ndarray:
    return np.interp(np.arange(n) / P, xs, ys)


def _noise(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(int(seed)).standard_normal(n)


def _dur(d, lo=0.05, hi=60.0) -> float:
    return float(np.clip(d, lo, hi))


# =============================================================================================
# CARTOON
# =============================================================================================
CARTOON = ("splat", "stretch", "pop", "warble", "zip", "bonk")


@sfx("cartoon", "Cartoon effects: splat (wet landing), stretch (rubber pulled, pitch climbing), pop (cork), warble (wobbling slide), zip (fast rise), bonk (hollow knock and spring).",
     kind=("bonk", " | ".join(CARTOON)), seed=(0, "variation"))
def cartoon(sr=DEFAULT_SR, kind="bonk", seed=0):
    """A table over `boing`, `pop`, `twang`, `chirp`."""
    k = _kind(kind, CARTOON, "cartoon")
    if k == "splat":
        n = _n(0.45)
        y = np.zeros(n)
        _put(y, _nm(_m(thud(sr=P, body="sack", contact_ms=25.0, size=1.2, seed=seed))), 0.0, 0.8)
        wet = amb.bp(_noise(n, seed), 900.0, 0.7) * np.exp(-np.arange(n) / P / 0.07)
        y += 0.6 * _nm(wet)
        _put(y, _nm(_m(chirp(sr=P, start=500, end=120, dur=0.14, wave="sine"))), 0.04, 0.5)
    elif k == "stretch":
        n = _n(0.8)
        y = 0.8 * _nm(_sine(np.geomspace(180.0, 900.0, n) * (1.0 + 0.012 * _sine(24.0, n)), n)) * _ramp(n, [0, 0.05, 0.7, 0.8], [0, 1, 1, 0])
        y += 0.3 * _nm(_fit(_m(twang(sr=P, kind="band", freq=180.0, dur=0.8, amount=0.5)), n))
    elif k == "pop":
        n = _n(0.2)
        y = np.zeros(n)
        _put(y, _nm(_m(pop(sr=P, freq=650))), 0.0, 1.0)
    elif k == "warble":
        n = _n(0.9)
        y = _sine(np.interp(np.arange(n), [0, n], [700.0, 500.0]) * (1.0 + 0.2 * _sine(9.0, n)), n) * _ramp(n, [0, 0.02, 0.8, 0.9], [0, 1, 1, 0])
    elif k == "zip":
        y = _m(chirp(sr=P, start=300, end=3200, dur=0.13, wave="sine"))
        y = y * np.sin(np.pi * np.minimum(np.arange(len(y)) / _n(0.012), 1.0) / 2.0) ** 2
    else:
        n = _n(0.7)
        y = np.zeros(n)
        _put(y, _nm(_m(impact(sr=P, material="wood", shape="bar", size=0.25, hollow=0.8, velocity=3.0, dur=0.2, seed=seed))), 0.0, 1.0)
        _put(y, _nm(_m(boing(sr=P, freq=320, decay=0.6, seed=seed))), 0.01, 0.5)
    return _fin(y, sr, 0.8, fade=0.002, tail=0.01)


# =============================================================================================
# STINGERS, BRAAM, MORPH, MAGIC
# =============================================================================================
STINGERS = ("boom", "bass_dive", "danger", "defeat")
MINOR_2ND, MINOR_3RD = 2.0 ** (1 / 12), 2.0 ** (3 / 12)


def _note(fn, f, dur, at, y, g=1.0, **kw):
    _put(y, _nm(_m(fn(f, dur, sr=P, **kw))), at, g)


@sfx("stinger_sfx", "Designed stingers: boom (reversed rise into a hit), bass_dive (sub falling away), danger (rising minor second), defeat (falling minor third).",
     kind=("boom", " | ".join(STINGERS)), freq=(165.0, "danger, defeat: Hz of the first note"), seed=(0, "variation"))
def stinger_sfx(sr=DEFAULT_SR, kind="boom", freq=165.0, seed=0):
    """Intervals carry the mood; then the boom and the dive."""
    k = _kind(kind, STINGERS, "stinger_sfx")
    f = float(np.clip(freq, 40.0, 2000.0))
    if k == "boom":
        y = np.zeros(_n(3.0))
        r = _nm(_m(reverse_reverb(_m(riser(sr=P, dur=0.9, start=100, end=2000)), P, mix=0.5, t60=0.8)))
        _put(y, r[:_n(1.0)] * _ramp(min(len(r), _n(1.0)), [0, 0.8, 1.0], [1, 1, 0]), 0.0, 0.4)
        _put(y, _nm(_m(explosion(sr=P, dur=1.9, seed=seed))), 1.0, 1.0)
    elif k == "bass_dive":
        n = _n(1.8)
        y = _sine(np.geomspace(140.0, 28.0, n), n) * _ramp(n, [0, 0.03, 1.0, 1.8], [0, 1, 0.8, 0])
        _put(y, _nm(_m(thud(sr=P, body="sack", contact_ms=35.0, size=2.0, seed=seed))), 0.0, 0.6)
    else:
        y = np.zeros(_n(2.2))
        second = f * (MINOR_2ND if k == "danger" else 1.0 / MINOR_3RD)
        _note(brass, f, 0.45, 0.0, y, 0.8, vel=0.8)
        _note(brass, second, 1.1, 0.55, y, 1.0, vel=0.9)
    return _fin(y, sr, 0.8, fade=0.003, tail=0.05)


@sfx("braam", "Trailer braam: brass chord pushed through distortion, over a sub-bass an octave down.",
     pitch=(55.0, "Hz of the root"), dur=(3.5, "s"), seed=(0, "variation"))
def braam(sr=DEFAULT_SR, pitch=55.0, dur=3.5, seed=0):
    """Voicing root, octave, fifth; drive; a sub and the drone under it."""
    f, d = float(np.clip(pitch, 25.0, 220.0)), _dur(dur, 0.5, 20.0)
    n = _n(d)
    b = np.zeros(n)
    for m, g in ((2.0, 1.0), (3.0, 0.7), (4.0, 0.6)):
        b += g * _fit(_nm(_m(brass(f * m, d, sr=P, vel=1.0, attack=0.12))), n)
    b = _nm(_m(distortion(b, P, drive=6.0, tone=2800)))
    sub = _sine(f, n) * _ramp(n, [0, 0.08, d], [0, 1, 0.7])
    drone = _nm(_fit(_m(dark_drone(sr=P, dur=d, note=f, seed=seed)), n))
    y = 0.7 * b + 0.8 * sub + 0.15 * drone
    return _fin(y * _ramp(n, [0, 0.03, d - 0.5, d], [0, 1, 1, 0]), sr, 0.85, fade=0.005, tail=0.05)


def _named(name: str, d: float) -> np.ndarray:
    kw = {"dur": d} if "dur" in REGISTRY.get(name, {}).get("params", {}) else {}
    return _fit(_m(render_sfx(name, sr=P, **kw)), _n(d))


@sfx("morph", "One sound turning into another: a, the cross-synthesis of a's envelope on b, then b.",
     a=("roar", "sfx name"), b=("wind", "sfx name"), dur=(2.0, "s"), seed=(0, "variation"))
def morph(sr=DEFAULT_SR, a="roar", b="wind", dur=2.0, seed=0):
    """`cross_synth` reads its carrier from a file; a three-part crossfade."""
    d = _dur(dur, 0.2, 20.0)
    x, c = _nm(_named(str(a), d)), _nm(_named(str(b), d))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "carrier.wav")
        write_wav(path, c, P)
        mid = _fit(_m(cross_synth(x, P, carrier=path, seed=seed)), len(x))
    n = len(x)
    y = _ramp(n, [0, 0.5 * d], [1, 0]) * x + _ramp(n, [0, 0.5 * d, d], [0, 1, 0]) * _nm(mid) + _ramp(n, [0.5 * d, d], [0, 1]) * c
    return _fin(y, sr, 0.8, fade=0.01, tail=0.05)


MAGICS = ("ange", "mal", "element", "sort", "poof", "shimmer")


@sfx("magic_kind", "Spell families: ange (bright rising glitter), mal (dark drone and falling sweep), element (fire and a charge), sort (charge then release), poof (puff of smoke), shimmer (sparkle in a rising reverb).",
     kind=("sort", " | ".join(MAGICS)), seed=(0, "variation"))
def magic_kind(sr=DEFAULT_SR, kind="sort", seed=0):
    """A table over `magic`, `charge_up`, `teleport`, `shimmer`, `fire`."""
    k = _kind(kind, MAGICS, "magic_kind")
    if k == "ange":
        x = _m(magic(sr=P, dur=1.4, density=20, freq="C7"))
        y = _m(shimmer(x, P, mix=0.5, shimmer=0.6, t60=1.5, tail=0.5))
    elif k == "mal":
        y = np.zeros(_n(2.0))
        _put(y, _nm(_m(dark_drone(sr=P, dur=2.0, note=45.0, ring=0.5, seed=seed))), 0.0, 1.0)
        n = _n(1.2)
        _put(y, _nm(_sine(np.geomspace(400.0, 60.0, n), n) * _ramp(n, [0, 0.1, 1.2], [0, 1, 0])), 0.2, 0.5)
    elif k == "element":
        y = np.zeros(_n(2.0))
        _put(y, _nm(_m(fire(sr=P, dur=2.0, size=0.7, crackle=20.0, seed=seed))), 0.0, 0.8)
        _put(y, _nm(_m(charge_up(sr=P, dur=1.2, start=100.0, end=700.0, ready=0))), 0.3, 0.5)
    elif k == "sort":
        y = np.zeros(_n(1.6))
        _put(y, _nm(_m(charge_up(sr=P, dur=0.8, start=150.0, end=1500.0, ready=0))), 0.0, 0.35)
        _put(y, _nm(_m(teleport(sr=P, dur=0.7, direction="out", seed=seed))), 0.8, 1.0)
    elif k == "poof":
        n = _n(0.45)
        y = amb.lop(amb.lop(amb.lop(_noise(n, seed), 1800.0), 1800.0), 1800.0) * _ramp(n, [0, 0.01, 0.45], [0, 1, 0]) ** 2
        y = _nm(y)
        _put(y, _nm(_m(thud(sr=P, body="sack", contact_ms=30.0, size=1.0, seed=seed))), 0.0, 0.5)
    else:
        y = _m(shimmer(_m(magic(sr=P, dur=0.6, density=30, freq="C6")), P, mix=0.7, shimmer=0.9, t60=2.0, tail=1.0))
    return _fin(y, sr, 0.8, fade=0.003, tail=0.1)


@sfx("ui_slide", "Panel sliding in or out: a short whoosh that lands on a tick.",
     direction=("in", "in (rising) | out (falling)"), dur=(0.18, "s"), seed=(0, "variation"))
def ui_slide(sr=DEFAULT_SR, direction="in", dur=0.18, seed=0):
    """`whoosh` + `ui_tick`."""
    if direction not in ("in", "out"):
        raise ValueError(f"ui_slide direction {direction!r}: use in or out")
    d = _dur(dur, 0.06, 1.0)
    y = np.zeros(_n(d + 0.1))
    _put(y, _nm(_m(whoosh(sr=P, dur=d, low=400, high=5000, direction="up" if direction == "in" else "down"))), 0.0, 0.5)
    _put(y, _nm(_m(ui_tick(sr=P, kind="focus", seed=seed))), d - 0.005, 1.0)
    return _fin(y, sr, 0.8, fade=0.002, tail=0.01)


# =============================================================================================
# RADIO, PHONE LINE
# =============================================================================================
RADIOS = ("squelch", "emetteur", "recherche")


def _voiceband(x: np.ndarray) -> np.ndarray:
    return _band(x, 300.0, 3400.0, P, 4)


@sfx("radio", "Walkie-talkie: squelch (the tail of static cut by a click), emetteur (keyed carrier: click, hiss, click), recherche (scanning: bursts of static and heterodyne whistles).",
     kind=("squelch", " | ".join(RADIOS)), dur=(1.5, "emetteur, recherche: s"), seed=(0, "variation"))
def radio(sr=DEFAULT_SR, kind="squelch", dur=1.5, seed=0):
    """Band 300-3400 Hz is the `telephone` effect's."""
    k = _kind(kind, RADIOS, "radio")
    rng = np.random.default_rng(int(seed))
    tick = _nm(_m(click(sr=P, tone=2500)))
    if k == "squelch":
        n = _n(0.32)
        y = _nm(_fit(_m(static(sr=P, dur=0.3)), n)) * _ramp(n, [0, 0.005, 0.26, 0.28], [0, 1, 0.8, 0])
        y = _nm(_voiceband(y))
        _put(y, tick, 0.28, 0.9)
    elif k == "emetteur":
        d = _dur(dur, 0.3, 30.0)
        n = _n(d)
        y = 0.25 * _nm(_voiceband(_fit(_m(static(sr=P, dur=d)), n))) * _ramp(n, [0, 0.03, d - 0.03, d], [0, 1, 1, 0])
        y = y + 0.04 * _sine(1000.0, n) * _ramp(n, [0, 0.03, d - 0.03, d], [0, 1, 1, 0])
        _put(y, tick, 0.0, 0.9)
        _put(y, tick, d - 0.02, 0.7)
    else:
        d = _dur(dur, 0.3, 30.0)
        n = _n(d)
        hop = _n(0.12)
        f, g = np.zeros(n), np.zeros(n)
        for i in range(0, n, hop):
            f[i:i + hop] = rng.uniform(500.0, 3000.0)
            g[i:i + hop] = rng.uniform(0.1, 1.0)
        kern = np.hanning(_n(0.01))
        g = np.convolve(g, kern / kern.sum(), "same")
        whistle = _sine(np.convolve(f, kern / kern.sum(), "same"), n)
        y = _voiceband(0.3 * amb.hip(_noise(n, seed), 100.0) * g + 0.15 * whistle * (1.0 - g))
    return _fin(y, sr, 0.8, fade=0.002, tail=0.01)


def phone_line(x: np.ndarray) -> np.ndarray:
    """Line at rate P: clip 0.9, BP 2000 Q12, HP 90 x2, then BP 400 Q3 beside clip 0.4, gains 0.5 and 0.15. `fx.telephone` lacks it."""
    y = amb.bp(np.clip(x, -0.9, 0.9), 2000.0, 12.0)
    y = amb.hip(amb.hip(y, 90.0), 90.0)
    return 0.5 * amb.bp(y, 400.0, 3.0) + 0.15 * np.clip(y, -0.4, 0.4)


PULSE_PERIOD, PULSE_CONTACT = 0.1, 0.04     # s: one pulse per 100 ms, 40 ms of it


def pulse_count(digit: str) -> int:
    """Digit 0 sends 10 pulses."""
    return int(digit) or 10


@sfx("rotary", "Rotary phone dialling: the dial wound round by hand (ratchet), then its return: 10 line pulses per second, each digit its own count, 0 = 10.",
     digits=("0471", "digits to dial"), seed=(0, "variation"))
def rotary(sr=DEFAULT_SR, digits="0471", seed=0):
    """Pulses are steps down a line through `phone_line`; the dial winds back, 0.3 s between digits."""
    ds = [c for c in str(digits) if c.isdigit()] or ["0"]
    wind, pause = 0.05, 0.3
    total = sum(wind * pulse_count(c) + 0.05 + PULSE_PERIOD * pulse_count(c) + pause for c in ds) + 0.2
    n = _n(total)
    mech, line = np.zeros(n), np.zeros(n)
    t = 0.05
    for c in ds:
        p = pulse_count(c)
        _put(mech, _nm(_m(ratchet(sr=P, clicks=p, rate=1.0 / wind, size=0.4, mount="plastic", seed=seed))), t, 1.0)
        t += wind * p + 0.05
        for i in range(p):
            a = _n(t + i * PULSE_PERIOD)
            line[a:a + _n(PULSE_CONTACT)] = 1.0
        t += PULSE_PERIOD * p + pause
    y = 0.25 * mech + _nm(phone_line(line))
    return _fin(y, sr, 0.8, fade=0.002, tail=0.02)


# =============================================================================================
# GEIGER, MODEM, MORSE
# =============================================================================================
@sfx("geiger", "Geiger counter: Poisson clicks of the tube.",
     rate=(12.0, "clicks per second"), dur=(3.0, "s"), seed=(0, "variation"))
def geiger(sr=DEFAULT_SR, rate=12.0, dur=3.0, seed=0):
    """Alias of `noise_bed(kind=geiger)`."""
    return noise_bed(sr=sr, kind="geiger", dur=_dur(dur, 0.05, 60.0), rate=max(float(rate), 0.01), level=-3.0, seed=seed)


BELL103_ORIG, BELL103_ANS = (1070.0, 1270.0), (2025.0, 2225.0)      # space, mark [BEL]
V21_CH1, V21_CH2 = (980.0, 1180.0), (1850.0, 1650.0)                # 0, 1
BAUD = 300.0
ANS_HZ, CNG_HZ, CED_HZ = 2100.0, 1100.0, 2100.0
MODEMS = ("poignee_de_main", "fax")


def fsk(bits, f01, baud: float = BAUD) -> np.ndarray:
    """Continuous-phase FSK at rate P: ``f01[bit]`` Hz for 1/baud s each."""
    per = P / float(baud)
    edges = np.round(np.arange(len(bits) + 1) * per).astype(int)
    f = np.repeat(np.asarray(f01, float)[np.asarray(bits, int)], np.diff(edges))
    return _sine(f, len(f))


def _seg(y: np.ndarray, x: np.ndarray, t: float, g: float = 1.0, edge: float = 0.003) -> None:
    k = max(2, min(len(x) // 2, _n(edge)))
    w = np.ones(len(x))
    w[:k], w[-k:] = np.linspace(0, 1, k), np.linspace(1, 0, k)
    _put(y, x * w, t, g)


@sfx("modem", "Dial-up signals: poignee_de_main (ANS 2100 Hz, then Bell 103 FSK both ways at 300 bit/s), fax (CNG 1100 Hz, CED 2100 Hz, V.21 flags and frame).",
     kind=("poignee_de_main", " | ".join(MODEMS)), seed=(0, "variation"))
def modem(sr=DEFAULT_SR, kind="poignee_de_main", seed=0):
    """Frequencies: module docstring. Heard on a line at this rate, sr under 12 kHz loses the top. Bell 103 exchange: 1 s ANS, 0.5 s carriers,
    2 s of data; fax: 75 ms silence and 1 s of 0x7E flags before the frame."""
    k = _kind(kind, MODEMS, "modem")
    rng = np.random.default_rng(int(seed))
    if k == "poignee_de_main":
        y = np.zeros(_n(4.0))
        _seg(y, _sine(ANS_HZ, _n(1.0)), 0.0)
        _seg(y, _sine(BELL103_ANS[1], _n(0.5)), 1.0, 0.7)
        _seg(y, _sine(BELL103_ORIG[1], _n(0.25)), 1.25, 0.7)
        nb = int(2.5 * BAUD)
        _seg(y, fsk(rng.integers(0, 2, nb), BELL103_ANS), 1.5, 0.7)
        _seg(y, fsk(rng.integers(0, 2, nb), BELL103_ORIG), 1.5, 0.7)
    else:
        y = np.zeros(_n(8.2))
        _seg(y, _sine(CNG_HZ, _n(0.5)), 0.0)
        _seg(y, _sine(CED_HZ, _n(3.0)), 3.5)
        flags = np.tile([0, 1, 1, 1, 1, 1, 1, 0], int(BAUD / 8))
        frame = np.concatenate([flags, rng.integers(0, 2, int(0.5 * BAUD))])
        _seg(y, fsk(frame, V21_CH2), 6.575)
    return _fin(y, sr, 0.7, fade=0.003, tail=0.01)


MORSE = {"A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.", "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-",
         "L": ".-..", "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.", "S": "...", "T": "-", "U": "..-", "V": "...-",
         "W": ".--", "X": "-..-", "Y": "-.--", "Z": "--..", "1": ".----", "2": "..---", "3": "...--", "4": "....-", "5": ".....", "6": "-....",
         "7": "--...", "8": "---..", "9": "----.", "0": "-----", ".": ".-.-.-", ",": "--..--", "?": "..--..", "/": "-..-.", "=": "-...-",
         "+": ".-.-.", "-": "-....-", "@": ".--.-.", "'": ".----.", "(": "-.--.", ")": "-.--.-", ":": "---...", '"': ".-..-."}


def morse_units(text: str) -> list[tuple[bool, int]]:
    """(key down, units) for ``text``. Dit 1, dah 3, within a letter 1, between letters 3, between words 7.
    A trailing space keeps its word gap, so "PARIS " is 50 units."""
    out: list[tuple[bool, int]] = []
    upper = str(text).upper()
    words = upper.split()
    for w, word in enumerate(words):
        for l, ch in enumerate(word):
            if ch not in MORSE:
                raise ValueError(f"morse: no code for {ch!r}")
            if l:
                out.append((False, 3))
            for e, sym in enumerate(MORSE[ch]):
                if e:
                    out.append((False, 1))
                out.append((True, 1 if sym == "." else 3))
        if w < len(words) - 1 or upper[-1:].isspace():
            out.append((False, 7))
    return out


@sfx("morse", "Morse from text: gated tone, or a telegraph sounder clacking on key down and up. Dit = 1200/wpm ms; 1:3:1:3:7.",
     text=("SOS", "letters, digits, punctuation"), wpm=(20.0, "words per minute (PARIS)"), freq=(700.0, "Hz, tone only"),
     mode=("tone", "tone | sounder"), seed=(0, "variation"))
def morse(sr=DEFAULT_SR, text="SOS", wpm=20.0, freq=700.0, mode="tone", seed=0):
    """PARIS timing; a 700 Hz tone, 5 ms key edges, or the sounder's two strokes."""
    if mode not in ("tone", "sounder"):
        raise ValueError(f"morse mode {mode!r}: use tone or sounder")
    units = morse_units(text) or [(False, 7)]
    dit = 60.0 / (50.0 * max(float(wpm), 1.0))
    n = _n(sum(u for _, u in units) * dit + 0.02)
    if mode == "tone":
        key = np.zeros(n)
        t = 0.0
        for down, u in units:
            if down:
                key[_n(t):_n(t + u * dit)] = 1.0
            t += u * dit
        k = np.hanning(2 * _n(0.005) + 1)
        return _fin(_sine(float(freq), n) * np.convolve(key, k / k.sum(), "same"), sr, 0.7, fade=0.002, tail=0.015)
    y = np.zeros(n + _n(0.1))
    down_hit = _nm(_m(solenoid(sr=P, size=0.3, metal=0.5, thump=0.4, bounce=0.1, seed=seed)))
    up_hit = _nm(_m(solenoid(sr=P, size=0.15, metal=0.6, thump=0.1, bounce=0.0, seed=seed + 1)))
    t = 0.0
    for down, u in units:
        if down:
            _put(y, down_hit, t, 1.0)
            _put(y, up_hit, t + u * dit, 0.5)
        t += u * dit
    return _fin(y, sr, 0.8, fade=0.001, tail=0.01)


# =============================================================================================
# ALARMS
# =============================================================================================
# duration ms, scale, four frequencies, timbre. Wrong buzzer is two messages.
ALERTS = {
    "happy_blips": ((380, 2, (349, 0, 0, 0), 1),),
    "affirmative": ((238, 1, (317, 0, 0, 476), 0),),
    "activate": ((317, 7, (300, 125, 0, 0), 1),),
    "invaders": ((1031, 9, (360, 238, 174, 158), 1),),
    "information": ((900, 4, (2000, 2010, 2000, 2010), 1),),
    "message_alert": ((1428, 3, (619, 571, 365, 206), 1),),
    "finished": ((450, 1, (365, 571, 619, 206), 0.5),),
    "error_code": ((714, 74, (1000, 0, 1000, 0), 1),),
    "wrong_buzzer": ((200, 30, (1000, 476, 159, 0), 1), (634, 61, (1000, 476, 159, 0), 1)),
    "three_tone": ((1500, 1, (723, 932, 1012), 1),),      # three tones, metro 500
}


def cosc(f: float, n: int, timbre: float) -> np.ndarray:
    """`pd cosc`: cos(cos wt) has the even harmonics, sin(cos wt) the odd ones; offset between them mixes."""
    return np.cos(np.cos(2 * np.pi * float(f) * np.arange(n) / P) - float(timbre) * np.pi / 2.0)


def alert_segment(ms: float, scale: float, freqs, timbre: float) -> np.ndarray:
    """Per-sample so the audio-rate gates (Error, Buzzer) work: pulse = ms / (len(freqs) scale), half-cosine each, DC trap at 50 Hz,
    x 0.25. The book wants 0 Hz to be silence, but its one-pole trap after the gate leaves a thump under each pulse (measured: 17 Hz
    the top line of Affirmative), so each oscillator's DC goes before the gate; the trap stays after it."""
    n = _n(ms / 1000.0)
    ph = np.arange(n) / n * float(scale) * len(freqs)
    win = np.sin(np.pi * (ph % 1.0))
    which = np.floor(ph).astype(int) % len(freqs)
    y = np.zeros(n)
    for i, f in enumerate(freqs):
        c = cosc(f, n, timbre)
        y += (which == i) * win * (c - c.mean())
    return 0.25 * amb.hip(y, 50.0)


@sfx("alert_tone", "Four-phase alarm generator: happy_blips, affirmative, activate, invaders, information, message_alert, finished, error_code, wrong_buzzer, three_tone (723, 932, 1012 Hz). Or your own via f1..f4.",
     preset=("message_alert", " | ".join(ALERTS)), ms=(0.0, "own message: duration, ms (0 = use the preset)"), scale=(1.0, "own message: cycles of the four tones"),
     f1=(0.0, "Hz"), f2=(0.0, "Hz"), f3=(0.0, "Hz"), f4=(0.0, "Hz"), timbre=(1.0, "0 even harmonics (an octave up, 11.7 dB softer) .. 1 odd"), level=(0.7, "peak: level is matched per preset"),
     rough=(0.0, "0..1 depth of an amplitude modulation in the roughness band: alarms read as urgent"), rough_hz=(70.0, "modulation rate, 30..150 Hz"))
def alert_tone(sr=DEFAULT_SR, preset="message_alert", ms=0.0, scale=1.0, f1=0.0, f2=0.0, f3=0.0, f4=0.0, timbre=1.0, level=0.7,
               rough=0.0, rough_hz=70.0):
    if float(ms) > 0:
        msgs = ((float(ms), float(scale), (f1, f2, f3, f4), float(timbre)),)
    else:
        msgs = ALERTS[_kind(preset, ALERTS, "alert_tone")]
    y = np.concatenate([alert_segment(*m) for m in msgs])
    return _fin(roughen(y, rough, rough_hz), sr, float(level), fade=0.002)


ROUGH_BAND = (30.0, 150.0)


def roughen(y, depth, rate=70.0):
    """Amplitude modulation at 30-150 Hz: the band heard as roughness."""
    d = float(np.clip(depth, 0.0, 1.0))
    if d == 0:
        return y
    fm = float(np.clip(rate, *ROUGH_BAND))
    return y * (1.0 - 0.5 * d + 0.5 * d * np.cos(2 * np.pi * fm * np.arange(len(y)) / P))


# =============================================================================================
# CANS, TOYS, TV AND AUDIO GEAR
# =============================================================================================
CAN_MODES = (359.0, 426.0, 1748.0, 3150.0)
CAN_LEVELS = (1.0, 0.513, 0.192, 0.1)                       # 1748 Hz and below measured; 3150 Hz at 0.1
CAN_GAIN = 0.3                                              # mix level
ROLL_PATTERN = ((2.0, 0.1), (3.0, 0.05), (4.2, 0.02), (5.6, 0.012))   # (k, a)
ROLL_PATTERNS = (ROLL_PATTERN, ((1.3, 0.1), (1.7, 0.05), (2.4, 0.02), (8.1, 0.012)))
ROLL_BUMPS = dict(lop=20.0, gain=3340.0)


def roll_pattern(rev: np.ndarray, pattern: int = 1) -> np.ndarray:
    """Rotation phase in turns -> force: sum of a * wrap(k phase), rzero 0.9, an impulse at each wrap."""
    y = sum(a * ((k * rev) % 1.0) for k, a in ROLL_PATTERNS[int(np.clip(pattern, 1, 2)) - 1])
    return y - 0.9 * np.concatenate(([y[0]], y[:-1]))


def roll_bumps(n: int, rng) -> np.ndarray:
    """Ground bumps: noise low-passed at 20 Hz, times 3340, wrapped, each period shaped 1 - 4 (x - 0.5)^2."""
    x = (amb.lop(rng.uniform(-1, 1, n), ROLL_BUMPS["lop"]) * ROLL_BUMPS["gain"]) % 1.0
    return 1.0 - 4.0 * (x - 0.5) ** 2


@sfx("can_roll", "Empty can rolling: its four measured modes struck by a pattern repeated each turn. Stands in for `roll body=can`.",
     speed=(2.0, "turns per second"), dur=(2.0, "s"), pattern=(1, "1 or 2: the two dent patterns of the can"),
     bumps=(0.0, "0..1 ground bumps modulating the drive"), seed=(0, "variation (bumps)"))
def can_roll(sr=DEFAULT_SR, speed=2.0, dur=2.0, pattern=1, bumps=0.0, seed=0):
    """Amplitude in sqrt(speed), clip 0.6; T60 80 ms, a 150 Hz high-pass keeps the ramps out of the modes."""
    d = _dur(dur, 0.1, 30.0)
    n = _n(d)
    rate = max(float(speed), 0.05)
    drive = amb.hip(roll_pattern(np.arange(n) * rate / P, pattern), 150.0) * math.sqrt(rate)
    b = float(np.clip(bumps, 0.0, 1.0))
    if b > 0:
        drive = drive * (1.0 - b + b * roll_bumps(n, np.random.default_rng(int(seed))))
    y = _modes(np.clip(drive, -0.6, 0.6), CAN_MODES, 0.08, CAN_LEVELS)
    return _fin(y * CAN_GAIN, sr, None, fade=0.005, tail=0.02)      # no normalising: the speed law is in the level


TOYS = ("mecanique", "telecommandee", "jouet_parlant")


@sfx("toy", "Toys: mecanique (wind-up: ratchet, spring, clockwork), telecommandee (RC: servo and a small buzzing motor), jouet_parlant (pull string, tiny speaker voice, a few music-box notes).",
     kind=("mecanique", " | ".join(TOYS)), seed=(0, "variation"))
def toy(sr=DEFAULT_SR, kind="mecanique", seed=0):
    """`servo`, `ratchet`, `spring`, `music_box`."""
    k = _kind(kind, TOYS, "toy")
    if k == "mecanique":
        y = np.zeros(_n(2.6))
        _put(y, _nm(_m(ratchet(sr=P, clicks=8, rate=9.0, size=0.5, mount="plastic", seed=seed))), 0.0, 0.8)
        _put(y, _nm(_m(spring(sr=P, action="release", tension=0.3, seed=seed))), 0.95, 0.5)
        _put(y, _nm(_m(ratchet(sr=P, clicks=24, rate=16.0, slow=0.4, size=0.25, mount="plastic", seed=seed + 1))), 1.05, 0.5)
    elif k == "telecommandee":
        y = np.zeros(_n(2.2))
        _put(y, _nm(_m(servo(sr=P, moves=2, dur=0.25, speed=220.0, gap=0.2, seed=seed))), 0.0, 0.6)
        n = _n(1.5)
        mot = _fit(_m(electric_motor(sr=P, dur=1.5, rpm=14000, load=0.7, startup=0.1, seed=seed)), n)
        _put(y, _nm(mot) * _ramp(n, [0, 0.1, 1.3, 1.5], [0, 1, 1, 0]), 0.6, 0.7)
    else:
        y = np.zeros(_n(2.6))
        _put(y, _nm(_m(ratchet(sr=P, clicks=9, rate=30.0, size=0.3, mount="plastic", seed=seed))), 0.0, 0.5)
        v = _m(render_speech("hello friend", sr=P, voice="child", seed=seed))
        small = np.tanh(3.0 * amb.lop(amb.hip(_nm(v), 500.0), 3000.0))
        _put(y, _nm(small), 0.4, 1.0)
        at = 0.4 + len(small) / P + 0.05
        for i, f in enumerate((1047.0, 1319.0, 1568.0)):
            _put(y, _nm(_m(music_box(f, 0.15, sr=P))), at + 0.14 * i, 0.3)
    return _fin(y, sr, 0.8, fade=0.002, tail=0.03)


TURNTABLE_HZ = 100.0 / 3.0 / 60.0                           # 33.3 rpm = 0.56 Hz, calculated
TT_STATES = ("lecture", "demarrage", "arret", "pose")


@sfx("turntable", "Record player: lecture (needle in the groove, ticking once a turn), demarrage (motor spins up), arret (spins down), pose (the stylus lands).",
     state=("lecture", " | ".join(TT_STATES)), rpm=(100.0 / 3.0, "platter rpm"), dur=(6.0, "s"), seed=(0, "variation"))
def turntable(sr=DEFAULT_SR, state="lecture", rpm=100.0 / 3.0, dur=6.0, seed=0):
    """1.2 s of run-up and run-down, the tick of a warped record."""
    st = _kind(state, TT_STATES, "turntable")
    d = _dur(dur, 0.5, 60.0)
    n = _n(d)
    f = float(rpm) / 60.0
    speed = {"lecture": np.ones(n), "pose": np.ones(n), "demarrage": np.minimum(np.arange(n) / P / 1.2, 1.0),
             "arret": np.maximum(1.0 - np.arange(n) / P / 1.2, 0.0)}[st]
    rev = np.cumsum(speed * f) / P
    motor = _nm(_fit(_m(electric_motor(sr=P, dur=d, rpm=300, load=0.3, seed=seed)), n)) * speed
    groove = _nm(_fit(_m(vinyl(sr=P, dur=d, crackle=8.0, hiss=0.3, seed=seed)), n)) * (0.75 + 0.25 * np.cos(2 * np.pi * rev)) * speed
    y = 0.15 * motor + 0.5 * groove
    tick = amb.bp(_noise(_n(0.02), seed), 1500.0, 1.0) * np.hanning(_n(0.02))
    for t in np.flatnonzero(np.diff(np.floor(rev))) / P:
        _put(y, _nm(tick), float(t), 0.9)
    if st == "pose":
        y *= _ramp(n, [0, 0.12, 0.2], [0, 0, 1])
        _put(y, _nm(_m(thud(sr=P, body="crate", contact_ms=8.0, size=0.3, seed=seed))), 0.1, 0.5)
        _put(y, _nm(_m(click(sr=P, tone=4000))), 0.1, 0.6)
    return _fin(y, sr, 0.8, fade=0.005, tail=0.03)


# NTSC 4.5 MHz / 286, PAL 625 x 25.
LINE_HZ = {"pal": 15625.0, "ntsc": 4.5e6 / 286.0}
MAINS_HZ = {"pal": 50.0, "ntsc": 60.0}
FILM_FPS = 24.0                                             # sound film
CRT_STATES = ("allumage", "souffle_15_kHz", "projecteur", "cassette")


@sfx("crt", "TV and AV gear: allumage (relay, degauss hum dying away, line whine coming up), souffle_15_kHz (line whine), projecteur (film claw at 24 fps, motor), cassette (keys, motor, tape hiss). Needs sr above 32 kHz for the line whine.",
     state=("allumage", " | ".join(CRT_STATES)), system=("pal", "pal 15625 Hz, 50 Hz mains | ntsc 15734 Hz, 60 Hz"), dur=(3.0, "souffle, projecteur, cassette: s"), seed=(0, "variation"))
def crt(sr=DEFAULT_SR, state="allumage", system="pal", dur=3.0, seed=0):
    """Line whine, the degauss with its 2 s fall, the mechanisms."""
    st = _kind(state, CRT_STATES, "crt")
    if system not in LINE_HZ:
        raise ValueError(f"crt system {system!r}: use pal or ntsc")
    line = LINE_HZ[system]
    d = _dur(dur, 0.3, 60.0)
    if st == "allumage":
        n = _n(3.0)
        y = np.zeros(n)
        _put(y, _nm(_m(solenoid(sr=P, size=0.4, metal=0.4, thump=0.5, bounce=0.2, seed=seed))), 0.0, 0.7)
        hum = _fit(_m(transformer(sr=P, dur=3.0, mains=MAINS_HZ[system], flux=1.3, load=0.6, loose=0.3, seed=seed)), n)
        y += _nm(hum) * _ramp(n, [0, 0.03, 0.5, 2.5], [0, 1, 1, 0])
        y += 0.08 * _sine(line, n) * _ramp(n, [0, 0.8, 1.6], [0, 0, 1])
    elif st == "souffle_15_kHz":
        n = _n(d)
        y = _sine(line, n) + 0.05 * amb.hip(_noise(n, seed), 8000.0)
    elif st == "projecteur":
        n = _n(d)
        shutter = _nm(_m(camera_shutter(sr=P, exposure=0.005, seed=seed)))[:_n(0.03)]
        clicks = np.zeros(n)
        clicks[(np.arange(int(d * FILM_FPS)) / FILM_FPS * P).astype(int)] = 1.0
        y = np.convolve(clicks, shutter)[:n] + 0.5 * _nm(_fit(_m(electric_motor(sr=P, dur=d, rpm=1440, load=0.5, seed=seed)), n))
    else:
        n = _n(d + 0.3)
        y = np.zeros(n)
        _put(y, _nm(_m(solenoid(sr=P, size=0.45, metal=0.3, thump=0.5, bounce=0.2, seed=seed))), 0.0, 1.0)
        run = _n(d)
        body = 0.3 * _nm(_fit(_m(electric_motor(sr=P, dur=d, rpm=2400, load=0.2, seed=seed)), run)) + 0.25 * _nm(amb.hip(_noise(run, seed), 3000.0))
        _put(y, body * _ramp(run, [0, 0.15, d - 0.05, d], [0, 1, 1, 0]), 0.05, 1.0)
        _put(y, _nm(_m(solenoid(sr=P, size=0.3, metal=0.3, thump=0.4, bounce=0.1, seed=seed + 1))), d + 0.05, 0.8)
    return _fin(y, sr, 0.8, fade=0.003, tail=0.02)


# =============================================================================================
# SMALL THINGS
# =============================================================================================
@sfx("tinnitus", "Ringing in the ears: a thin high tone over a little noise around it, as after a blast.",
     freq=(6000.0, "Hz, 4000..8000"), dur=(3.0, "s"), seed=(0, "variation"))
def tinnitus(sr=DEFAULT_SR, freq=6000.0, dur=3.0, seed=0):
    """A 4-8 kHz tone over faint noise, 0.3 s fade-in, 1 s fade-out."""
    f, d = float(np.clip(freq, 4000.0, 8000.0)), _dur(dur, 0.5, 60.0)
    n = _n(d)
    y = _sine(f, n) + 0.12 * _nm(amb.bp(_noise(n, seed), f, 8.0))
    return _fin(y * _ramp(n, [0, 0.3, d - 1.0 if d > 1.3 else d * 0.7, d], [0, 1, 1, 0]), sr, 0.4, fade=0.01)


@sfx("party_horn", "Party blower: langue (reed and paper tongue unrolling, pitch climbing then flopping) | trompette (cone: plain steady reed).",
     kind=("langue", "langue | trompette"), freq=(480.0, "Hz of the reed"), seed=(0, "variation"))
def party_horn(sr=DEFAULT_SR, kind="langue", freq=480.0, seed=0):
    """`reed_tube` + `balloon`; tongue flutter 22 Hz, the pitch rising 6 %."""
    k = _kind(kind, ("langue", "trompette"), "party_horn")
    f = float(np.clip(freq, 120.0, 1500.0))
    d = 0.9
    n = _n(d)
    x = _nm(_fit(_m(reed_tube(f, d, sr=P, pressure=1.2, breath=0.1, seed=seed)), n))
    if k == "trompette":
        return _fin(x * _ramp(n, [0, 0.02, d - 0.1, d], [0, 1, 1, 0]), sr, 0.8, fade=0.003, tail=0.03)
    ratio = 1.0 + 0.06 * np.minimum(np.arange(n) / P / 0.5, 1.0)
    y = np.interp(np.cumsum(ratio) - 1.0, np.arange(n), x, right=0.0)
    y *= 0.75 + 0.25 * _sine(22.0, n)
    y *= _ramp(n, [0, 0.02, 0.6, 0.75, d], [0, 1, 1, 0.4, 0])
    y += 0.12 * _nm(_fit(_m(balloon(sr=P, dur=d, size=0.2, seed=seed)), n)) * _ramp(n, [0, 0.55, 0.65, d], [0, 0, 1, 0])
    return _fin(y, sr, 0.8, fade=0.003, tail=0.03)


@sfx("cap_gun", "Toy cap gun: a bare snap with no barrel, a little room after it.",
     seed=(0, "variation"))
def cap_gun(sr=DEFAULT_SR, seed=0):
    """5 ms, 3 kHz, Q 1.5, in a small room."""
    n = _n(0.25)
    m = _n(0.005)
    x = np.zeros(n)
    x[:m] = amb.bp(_noise(m, seed), 3000.0, 1.5) * np.hanning(m)
    x = _nm(x)
    _put(x, _nm(_m(click(sr=P, tone=3000))), 0.0, 0.3)
    return _fin(_fit(_m(room(x, P, mix=0.08, preset="closet")), n), sr, 0.9, fade=0.0005, tail=0.05)


@sfx("mic_noise", "A live microphone handled: bump (a knock on the body) | souffle (breath or wind on the capsule).",
     kind=("bump", "bump | souffle"), dur=(1.0, "souffle: s"), seed=(0, "variation"))
def mic_noise(sr=DEFAULT_SR, kind="bump", dur=1.0, seed=0):
    """`thud`, and `wind` through an 800 Hz low-pass for the plosives."""
    k = _kind(kind, ("bump", "souffle"), "mic_noise")
    if k == "bump":
        y = _nm(_m(thud(sr=P, body="sack", contact_ms=20.0, size=1.5, seed=seed)))
        return _fin(y, sr, 0.8, fade=0.001, tail=0.03)
    d = _dur(dur, 0.1, 20.0)
    n = _n(d)
    y = amb.hip(amb.lop(_nm(_fit(_m(wind(sr=P, dur=d, seed=seed)), n)), 800.0), 30.0)
    return _fin(y * _ramp(n, [0, 0.05, d - 0.1, d], [0, 1, 1, 0]), sr, 0.8, fade=0.005, tail=0.03)


@sfx("mic_feedback", "PA howl: the mic bumped, then the feedback loop grabs one pitch and swells.",
     freq=(2500.0, "Hz of the howl"), dur=(2.5, "s"), seed=(0, "variation"))
def mic_feedback(sr=DEFAULT_SR, freq=2500.0, dur=2.5, seed=0):
    """Loop: `feedback_guitar`, swelling 0.05..1."""
    f, d = float(np.clip(freq, 200.0, 8000.0)), _dur(dur, 0.3, 20.0)
    n = _n(d)
    loop = _nm(_fit(_m(feedback_guitar(f, d, sr=P, drive=6.0, feedback=0.8, seed=seed)), n))
    y = (0.5 * loop + 0.5 * _sine(f, n)) * np.geomspace(0.05, 1.0, n) * _ramp(n, [0, d - 0.03, d], [1, 1, 0])
    _put(y, _nm(_m(thud(sr=P, body="sack", contact_ms=12.0, size=1.0, seed=seed))), 0.0, 0.4)
    return _fin(y, sr, 0.8, fade=0.002, tail=0.01)


@sfx("ice_ring", "A sheet of lake ice singing: one crack whose dispersed whistle rings, with a struck ice mode under it.",
     thickness=(0.05, "m"), distance=(120.0, "m: the further, the longer the falling whistle"), seed=(0, "variation"))
def ice_ring(sr=DEFAULT_SR, thickness=0.05, distance=120.0, seed=0):
    """`ice_crack(sing=1)` with a struck layer."""
    a = _nm(_m(ice_crack(sr=P, dur=2.0, thickness=thickness, distance=distance, cracks=1, sing=1.0, seed=seed)))
    b = _nm(_m(impact(sr=P, material="ice", shape="rect_plate", size=0.6, velocity=1.0, dur=1.0, seed=seed)))
    y = np.zeros(len(a))
    y += a
    _put(y, b, 0.0, 0.25)
    return _fin(y, sr, 0.8, fade=0.002, tail=0.1)


@sfx("kazoo", "Kazoo: a hummed voice buzzing through a membrane.",
     freq=(220.0, "Hz of the hum"), dur=(1.0, "s"), seed=(0, "variation"))
def kazoo(sr=DEFAULT_SR, freq=220.0, dur=1.0, seed=0):
    """`voice` through `formant_tract`, a tanh drive 6 and a 300-5000 Hz membrane band."""
    f, d = float(np.clip(freq, 60.0, 1200.0)), _dur(dur, 0.1, 20.0)
    n = _n(d)
    v = _nm(_fit(_m(voice(f, d, sr=P, vowel="oo", vibrato=0.0, seed=seed)), n))
    y = amb.lop(amb.hip(np.tanh(6.0 * v), 300.0), 5000.0)
    y = _fit(_m(formant_tract(y, P, vowel="u", mix=0.5)), n)
    return _fin(y * _ramp(n, [0, 0.03, d - 0.05, d], [0, 1, 1, 0]), sr, 0.8, fade=0.003, tail=0.02)


@sfx("rubber_duck", "Rubber duck squeezed: air whining through the squeaker into its small body, up on the squeeze, down on the release.",
     squeezes=(1, "count"), pitch=(1.0, "0.5..2"), seed=(0, "variation"))
def rubber_duck(sr=DEFAULT_SR, squeezes=1, pitch=1.0, seed=0):
    """`squeak` + `bottle`: 1.1 kHz, a +20 % glide, 0.28 s."""
    p = float(np.clip(pitch, 0.5, 2.0))
    count = int(np.clip(squeezes, 1, 20))
    each = 0.34
    y = np.zeros(_n(each * count + 0.1))
    for i in range(count):
        n = _n(0.28)
        glide = np.interp(np.arange(n), [0, 0.5 * n, n], [0.85, 1.0, 0.82])
        air = _sine(1100.0 * p * glide, n)
        body = _nm(_fit(_m(bottle(1100.0 * p, 0.28, sr=P, noise=0.3, seed=seed + i)), n))
        rub = _nm(_fit(_m(squeak(sr=P, surface="shoe", pitch=p, dur=0.28, seed=seed + i)), n))
        _put(y, (air + 0.25 * body + 0.2 * rub) * np.sin(np.pi * np.arange(n) / n) ** 2, each * i, 1.0)
    return _fin(y, sr, 0.8, fade=0.003, tail=0.02)


def water_whistle_pitch(n: int, length: float, warble: float) -> np.ndarray:
    """Hz: a pipe stopped by the water, f = c / 4 (L - depth), depth bobbing at ``warble`` Hz, depth 0.1..0.4 L."""
    depth = float(length) * (0.25 + 0.15 * np.sin(2 * np.pi * float(warble) * np.arange(n) / P))
    return C_AIR / (4.0 * (float(length) - depth))


@sfx("water_whistle", "Bird whistle dipped in water: an edge tone whose pitch warbles with the immersion, and bubbles. After the 1972 Mexico fountain (\"similar to nightingale song\").",
     length=(0.04, "m of pipe"), warble=(9.0, "Hz of the bobbing"), bubble=(0.4, "level of the bubbles"), dur=(3.0, "s"), seed=(0, "variation"))
def water_whistle(sr=DEFAULT_SR, length=0.04, warble=9.0, bubble=0.4, dur=3.0, seed=0):
    """`edge_tone` + `bubbles`; pitch moved by varispeed to `water_whistle_pitch` (40 mm, 9 Hz)."""
    d = _dur(dur, 0.2, 30.0)
    n = _n(d)
    target = water_whistle_pitch(n, float(np.clip(length, 0.01, 0.3)), float(warble))
    base = _nm(_m(edge_tone(sr=P, speed=20.0, gap=8.0, dur=d + 2.0, seed=seed)))
    base = amb.hip(base, 200.0)
    F = np.abs(np.fft.rfft(base * np.hanning(len(base))))
    f0 = float(np.fft.rfftfreq(len(base), 1 / P)[np.argmax(F)])
    pos = np.cumsum(target / f0)
    if pos[-1] >= len(base) - 1:                            # the jet tone is long enough for a ratio of ~4; beyond that it loops
        base = np.resize(base, int(pos[-1]) + 2)
    y = np.interp(pos, np.arange(len(base)), base)
    y = _nm(y) + float(bubble) * _nm(_fit(_m(bubbles(sr=P, dur=d, rate=25.0, size=1.0, seed=seed)), n))
    return _fin(y * _ramp(n, [0, 0.05, d - 0.1, d], [0, 1, 1, 0]), sr, 0.8, fade=0.005, tail=0.02)


def pendulum_gate(t: np.ndarray, tau: float, period: float, width: float, amp: float = 1.0) -> np.ndarray:
    """True while the swinging mic is close enough to howl: |A exp(-t/tau) cos(2 pi t/T)| < width."""
    return np.abs(float(amp) * np.exp(-t / float(tau)) * np.cos(2 * np.pi * t / float(period))) < float(width)


def room_modes(lx: float, ly: float, lz: float, lo: float = 800.0, hi: float = 3000.0) -> np.ndarray:
    """f = c/2 sqrt(sum (n/l)^2), rigid box, sorted, within lo..hi."""
    k = np.arange(0, 40)
    a, b, c = np.meshgrid(k / lx, k[:20] / ly, k[:12] / lz, indexing="ij")
    f = 0.5 * C_AIR * np.sqrt(a * a + b * b + c * c)
    return np.unique(np.round(f[(f >= lo) & (f <= hi)], 1))


@sfx("pendulum_feedback", "Pendulum Music (Reich): a mic swings over its speaker, howling at each pass; the swing dies, the whistles lengthen, until the howl never stops. The room sets the pitches.",
     tau=(12.0, "s, decay of the swing"), period=(3.0, "s, full swing"), width=(0.3, "mic-speaker gate, 0..1 of the first swing"), lx=(6.0, "m room"), ly=(5.0, "m"), lz=(3.0, "m"),
     dur=(20.0, "s"), seed=(0, "variation"))
def pendulum_feedback(sr=DEFAULT_SR, tau=12.0, period=3.0, width=0.3, lx=6.0, ly=5.0, lz=3.0, dur=20.0, seed=0):
    """The gate above; three modes per pass drawn from `room_modes`, 8 ms edge, tanh."""
    d = _dur(dur, 0.5, 120.0)
    n = _n(d)
    rng = np.random.default_rng(int(seed))
    fm = room_modes(float(lx), float(ly), float(lz))
    picks = rng.choice(fm, size=min(3, len(fm)), replace=False)
    x = sum(g * _sine(f, n) for f, g in zip(picks, (1.0, 0.5, 0.3)))
    gate = pendulum_gate(np.arange(n) / P, tau, period, width).astype(float)
    k = np.hanning(2 * _n(0.008) + 1)
    gate = np.convolve(gate, k / k.sum(), "same")
    return _fin(np.tanh(1.5 * x) * gate, sr, 0.8, fade=0.01, tail=0.05)


# Pitch of the rods, rod_bank kit.
ROD_HZ = (262.0, 311.0, 370.0, 440.0, 523.0, 622.0)
STRIKES_PER_S = 1.5                                         # per pendulum
BALL_RATE = 3.0                                             # per s


def cage_hits(pendulums: int, dur: float, rng) -> np.ndarray:
    """Strike times: each pendulum passes at its own steady rate; the number of pendulums sets the density."""
    out = [np.arange(rng.uniform(0.0, 1.0 / STRIKES_PER_S), dur, 1.0 / (STRIKES_PER_S * rng.uniform(0.9, 1.1))) for _ in range(int(pendulums))]
    return np.sort(np.concatenate(out)) if out else np.zeros(0)


@sfx("rod_cage", "Sound sculpture left to itself: pendulums strike a cage of metal rods, a rod drawn at random each time, with glass or wooden balls rattling free inside.",
     pendulums=(2, "their number sets the density"), ball=("glass", "glass | wood | none"), dur=(6.0, "s"), seed=(0, "variation"))
def rod_cage(sr=DEFAULT_SR, pendulums=2, ball="glass", dur=6.0, seed=0):
    """`rod_bank` kernel, random rod."""
    if ball not in ("glass", "wood", "none"):
        raise ValueError(f"rod_cage ball {ball!r}: use glass, wood or none")
    d = _dur(dur, 0.5, 60.0)
    rng = np.random.default_rng(int(seed))
    p = int(np.clip(pendulums, 1, 8))
    y = np.zeros(_n(d + 1.0))
    bank = [_nm(_m(rod_bank(f, 0.5, sr=P, decay=1.0, seed=seed))) for f in ROD_HZ]
    for t in cage_hits(p, d, rng):
        _put(y, bank[int(rng.integers(len(bank)))], float(t), float(rng.uniform(0.6, 1.0)))
    if ball != "none":
        hit = [_nm(_m(impact(sr=P, material=ball, shape="bar", size=0.03, velocity=1.0, dur=0.08, seed=seed + i))) for i in range(3)]
        for t in np.cumsum(rng.exponential(1.0 / BALL_RATE, int(BALL_RATE * d * 2) + 1)):
            if t < d:
                _put(y, hit[int(rng.integers(3))], float(t), float(rng.uniform(0.04, 0.12)))
    return _fin(y, sr, 0.8, fade=0.002, tail=0.3)


# =============================================================================================
# CREATURES
# =============================================================================================
CREATURES = ("dragon", "dino", "zombie", "kraken", "elementaire", "fantome", "insectoid", "blob")


@sfx("creature", "Creature families from layers: dragon (roar, fire, wings), dino (huge low roar), zombie (wet groan), kraken (roar under gurgling water), elementaire (fire and a growl), fantome (airy gliding moan), insectoid (hiss, wings), blob (gurgle, bubbles).",
     kind=("dragon", " | ".join(CREATURES)), dur=(2.5, "s"), seed=(0, "variation"))
def creature(sr=DEFAULT_SR, kind="dragon", dur=2.5, seed=0):
    """`monster` tract and layers."""
    k = _kind(kind, CREATURES, "creature")
    d = _dur(dur, 0.5, 15.0)
    n = _n(d)

    def L(x):
        return _nm(_fit(_m(x), n))

    if k == "dragon":
        y = L(roar(sr=P, size=2.2, aggression=0.9, dur=d, seed=seed)) + 0.35 * L(fire(sr=P, dur=d, size=0.8, seed=seed)) + 0.3 * L(wings(sr=P, kind="bird", rate=3.0, size=3.0, dur=d, seed=seed))
    elif k == "dino":
        y = L(roar(sr=P, size=3.5, aggression=0.8, dur=d, seed=seed)) + 0.5 * L(growl(sr=P, size=3.0, pitch=45.0, dur=d, seed=seed))
    elif k == "zombie":
        y = L(growl(sr=P, size=0.9, aggression=0.3, pitch=85.0, dur=d, seed=seed)) + 0.35 * L(gurgle(sr=P, dur=d, rate=5.0, size=0.3, seed=seed))
    elif k == "kraken":
        y = L(monster(sr=P, size=4.0, aggression=0.7, pitch=90.0, dur=d, seed=seed)) + 0.6 * L(gurgle(sr=P, dur=d, rate=2.0, size=1.0, seed=seed)) + 0.3 * L(bubbles(sr=P, dur=d, rate=10.0, size=4.0, seed=seed))
    elif k == "elementaire":
        y = L(fire(sr=P, dur=d, size=1.0, crackle=25.0, wind=0.6, seed=seed)) + 0.7 * L(growl(sr=P, size=1.8, aggression=0.8, pitch=55.0, dur=d, seed=seed))
    elif k == "fantome":
        f = 330.0 * (1.0 + 0.25 * np.sin(2 * np.pi * 0.7 * np.arange(n) / P)) * (1.0 + 0.012 * _sine(5.0, n))
        moan = _m(reverse_reverb(_sine(f, n) + 0.3 * _sine(2.0 * f, n), P, mix=0.5, t60=1.5))
        y = _nm(_fit(moan, n)) + 0.5 * L(hiss(sr=P, kind="snake", size=1.5, dur=d, seed=seed))
        y *= _ramp(n, [0, 0.4 * d, d], [0, 1, 0.2])
    elif k == "insectoid":
        y = L(hiss(sr=P, kind="cat", size=0.4, dur=d, seed=seed)) * (0.6 + 0.4 * _sine(35.0, n)) + 0.7 * L(wings(sr=P, kind="insect", rate=60.0, size=0.5, dur=d, seed=seed))
    else:
        y = L(gurgle(sr=P, dur=d, rate=3.0, size=1.2, seed=seed)) + 0.5 * L(bubbles(sr=P, dur=d, rate=8.0, size=5.0, seed=seed))
        y = amb.lop(y, 1200.0)
    return _fin(y, sr, 0.85, fade=0.01, tail=0.1)
