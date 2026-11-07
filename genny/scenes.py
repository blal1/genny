"""Ambience scenes: a bed under sustained layers and scattered events, all taken from the catalogue.

``scene(kind=)`` reaches ``physical.ambience`` (``render_bed``, ``ACT_BEDS``, ``bed_generic``) and lays existing sfx over it,
by name, through the registry. Before a climax the quiet passage is lowered, then a silence, four small hits and the big one;
after it the ears are dulled (``ordnance.earplug``). Sources of every value: research/SOURCES.md, section scenes.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample as _fft_resample

from . import fx as Fx
from . import reverbs as Rv
from .core import DEFAULT_SR
from .instruments import render_chord
from .notes import to_hz
from .ordnance import earplug
from .physical import ambience as Am
from .physical.core import pan_gains
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


def L(name, db, wide=True, opt=False, **params):
    """A layer sounding all along. wide = one take per channel (independent, so mono-safe); else one centred take."""
    return dict(sfx=name, db=db, wide=wide, opt=opt, every=None, params=params)


def E(name, db, every, opt=False, **params):
    """Events every (lo, hi) s. opt = skipped when this sfx is not registered."""
    return dict(sfx=name, db=db, wide=False, opt=opt, every=every, params=params)


def C(inst, db, notes):
    """A held chord from an instrument."""
    return dict(inst=inst, sfx=inst, db=db, wide=False, opt=False, every=None, notes=notes, params={})


TEXT = {    # text bed_generic reads: its keywords pick the emitters
    "open": "open air", "alpine": "mountain peak snow pass pine", "meadow": "grass meadow", "forest": "forest tree",
    "desert": "dry plain", "city": "city street stone wall", "sea": "sea shore", "harbour": "harbour sea",
    "jungle": "jungle tree", "tundra": "snow glacier pass", "cave": "cave tunnel underground", "swamp": "swamp",
    "war": "battle war burn stone", "lake": "lake grass", "town": "town stone", "burn": "burn forest tree wind",
}

CROWD = dict(people=12)
TALK = dict(mood=0.3, distance=0.6, **CROWD)
BUSY = dict(mood=0.8, distance=0.4, **CROWD)

# Scene: bed = key of TEXT, or None for the quiet air of ambience.room_tone. room = reverbs.room preset for everything but the bed.
SCENES = {
    "air": dict(bed="open", layers=[L("noise_bed", -12, kind="rumble"), E("gust", -4, (6, 12), through="gap")]),
    "alpine": dict(bed="alpine", layers=[E("gust", -3, (6, 12), through="canyon"), E("bird_kind", -14, (10, 22), kind="rapaces"),
                                         E("distant_bell", -20, (14, 28), distance=800.0)]),
    "amusement": dict(bed=None, layers=[L("crowd_murmur", -3, mood=0.9, distance=0.3, **CROWD), E("ride", -6, (9, 16), kind="manege", dur=8.0),
                                        E("slot_machine", -12, (5, 11), state="spin"), E("crowd", -6, (7, 14), kind="ah"), E("scream", -14, (9, 20), syllables=2)]),
    "birdsong": dict(bed="meadow", layers=[L("dawn_chorus", 3)]),
    "celebration": dict(bed=None, layers=[L("crowd_murmur", -2, mood=1.0, distance=0.3, **CROWD), E("applause", -6, (9, 18), dur=3.0), E("cork_pop", -10, (6, 14)),
                                          E("party_horn", -13, (5, 11)), E("firework", -14, (8, 16), kind="pop", distance=400.0)]),
    "construction": dict(bed="town", layers=[E("jackhammer", -3, (6, 12), dur=2.5), E("power_tool", -6, (5, 10), kind="disqueuse", dur=2.0), E("hammering", -6, (4, 8), strikes=5),
                                             E("saw", -10, (7, 13), strokes=5), E("backup_beeper", -16, (10, 20), dur=3.0), E("excavator", -8, (10, 18), dur=4.0)]),
    "desert": dict(bed="desert", layers=[L("sand_storm", -6, wind=0.42, hard=0.3), L("night_insects", -20, density=0.15, temp=30.0), E("gust", -5, (7, 14), through="gap")]),
    "designed": dict(bed=None, act="awakening", layers=[L("dark_drone", -2, wide=False), L("shepard", -16, wide=False, direction="down", rate=0.05)]),
    "emergency": dict(bed="city", layers=[E("siren_wail", -3, (7, 13), kind="ambulance", dur=5.0, echo=0.8), L("crowd", -8, kind="panique", voices=12), E("shout", -10, (5, 10), kind="hurlement"),
                                          E("alarm_bell", -16, (8, 16), dur=2.0)]),
    "fantasy": dict(bed="forest", layers=[L("dark_drone", -8, wide=False, note=45.0), E("magic_kind", -8, (5, 11), kind="shimmer"), E("wings", -14, (8, 15), dur=2.0),
                                          E("creature", -16, (12, 24), kind="fantome", dur=2.5)]),
    "farm": dict(bed="meadow", layers=[E("moo", -6, (7, 15)), E("chicken", -10, (3, 7), kind="caquet"), E("farm_animal", -10, (6, 13), kind="mouton"), E("pig", -12, (8, 16)),
                                       E("horse", -12, (12, 24)), E("birdsong", -12, (12, 24), species="rooster"), E("tractor", -18, (15, 30), dur=4.0)]),
    "forest": dict(bed="forest", layers=[L("dawn_chorus", -6, birds=6), E("foliage", -8, (6, 12), kind="branches"), E("wild_animal", -18, (14, 28), kind="cerf"), E("crack", -18, (11, 22), kind="branche")]),
    "grassland": dict(bed="meadow", layers=[L("night_insects", -7, density=0.4), E("foliage", -8, (4, 9), kind="herbe"), E("bird_kind", -18, (12, 24), kind="rapaces")]),
    "historical": dict(bed="town", layers=[L("crowd_murmur", -5, **TALK), E("footsteps", -9, (5, 10), ground="cobblestone", steps=6), E("door_family", -12, (10, 20), kind="donjon"),
                                           E("wagon", -8, (8, 16), dur=4.0), E("distant_bell", -15, (12, 24), distance=300.0)]),
    "hitech": dict(bed=None, layers=[L("hvac", -5, wide=False, state="fan"), L("room_tone", -14, wide=False, kind="bureau"), E("cyber_ui", -12, (4, 9), kind="scan"), E("hard_drive", -14, (6, 12), dur=1.5), E("servo", -14, (7, 14))], room="studio"),
    "hospital": dict(bed=None, layers=[L("hvac", -8, wide=False, state="cool"), L("medical", -10, wide=False, kind="moniteur"), L("medical", -14, wide=False, kind="respirateur"),
                                       E("crowd_murmur", -14, (8, 16), people=4, mood=0.15, dur=4.0), E("cart", -12, (9, 18), kind="chariot", dur=3.0), E("elevator", -14, (15, 28), kind="carillon"),
                                       E("door_family", -12, (10, 20), kind="coulissante")], room="hall"),
    "industrial": dict(bed=None, layers=[L("industrial", -2, wide=False, kind="usine"), L("conveyor", -6, wide=False), L("transformer", -12, wide=False, loose=0.3), E("industrial", -3, (5, 10), kind="presse", dur=2.0),
                                         E("air_system", -10, (7, 14), kind="air_dryer_purge")], room="hall"),
    "insect": dict(bed="meadow", layers=[L("night_insects", 2, density=0.8), L("swarm", -6, kind="cigales", count=8)]),
    "lakeside": dict(bed="lake", layers=[L("lap", -2, kind="clapot"), L("dawn_chorus", -8, birds=5), E("frog", -13, (7, 14), kind="ribbit"), E("farm_bird", -14, (10, 20), kind="canard")]),
    "market": dict(bed=None, layers=[L("crowd_murmur", -1, **BUSY), E("coins", -11, (4, 8), kind="comptage"), E("shout", -9, (5, 11), kind="encouragement"), E("cart", -9, (7, 14), kind="charrette"),
                                     E("chicken", -15, (9, 17), kind="caquet"), E("clink", -15, (5, 11))]),
    "nautical": dict(bed="harbour", layers=[L("lap", -3, kind="quai"), L("sailboat", -6), E("creak", -11, (6, 12), material="wood"), E("ship_bell", -14, (12, 24), ring=1.5), E("foghorn", -12, (15, 30), distance=800.0),
                                            E("bird_kind", -15, (8, 16), kind="marins")]),
    "office": dict(bed=None, layers=[L("hvac", -8, wide=False, state="fan"), L("crowd_murmur", -14, people=5, mood=0.3, distance=0.7), E("keyboard_typing", -9, (5, 9), keys=20), E("printer", -13, (12, 22), lines=2),
                                     E("office", -13, (7, 15), kind="agrafeuse"), E("phone_bell", -15, (14, 26))], room="studio"),
    "park": dict(bed="meadow", layers=[L("dawn_chorus", -4, birds=6), L("crowd_murmur", -12, people=8, mood=0.5, distance=0.9), L("fountain", -13), E("bicycle", -11, (10, 18), dur=4.0),
                                       E("bark", -16, (11, 20), size="small", count=2), E("ball_bounce", -16, (8, 15), opt=True, kind="foot", surface="turf")]),
    "prison": dict(bed=None, layers=[L("crowd_murmur", -10, people=8, mood=0.25, distance=0.8), E("chain", -8, (6, 13), dur=2.0), E("door_family", -5, (10, 20), kind="prison"), E("lock", -9, (9, 18), kind="verrou"),
                                     E("footsteps", -11, (6, 11), ground="concrete", shoe="boot", steps=6), E("shout", -13, (8, 16), kind="hurlement")], room="hall"),
    "protest": dict(bed="city", layers=[L("crowd", -1, kind="chants", voices=12), E("shout", -8, (4, 9), kind="cri de guerre", people=4), E("crowd", -6, (9, 16), kind="huee", dur=3.0), E("applause", -10, (12, 22), dur=3.0),
                                        E("siren_wail", -18, (15, 28), kind="police_fr", dur=4.0, bright=0.5)]),
    "public": dict(bed=None, layers=[L("crowd_murmur", -2, **TALK), E("footsteps", -9, (4, 8), ground="hard_floor", steps=6), E("alert_tone", -15, (14, 26), preset="information"), E("door_family", -14, (10, 20), kind="tournante", dur=3.0)], room="hall"),
    "religious": dict(bed=None, layers=[C("church_organ", -6, ("C3", "G3", "C4", "E4")), L("crowd_murmur", -15, people=8, mood=0.1, distance=0.8), E("distant_bell", -12, (9, 18), distance=100.0), E("cough", -18, (9, 18))], room="church"),
    "residential": dict(bed=None, layers=[L("room_tone", 0, kind="salon"), L("clock", -14, wide=False, size="mantle"), L("appliance", -12, wide=False, kind="frigo"), E("door_family", -10, (12, 24), kind="battante"),
                                          E("faucet", -13, (10, 22), dur=2.0), E("kettle", -18, (20, 40), dur=4.0)], room="bedroom"),
    "restaurant": dict(bed=None, layers=[L("crowd_murmur", -1, **BUSY), E("clink", -10, (2, 5)), E("eat", -15, (6, 12)), E("cork_pop", -14, (12, 24)), E("frying_pan", -14, (9, 18)),
                                         E("footsteps", -14, (7, 13), ground="wood", steps=5)], room="studio"),
    "room": dict(bed=None, layers=[L("room_tone", 0, kind="salon")]),
    "rural": dict(bed="meadow", layers=[L("dawn_chorus", -6, birds=5), E("moo", -14, (9, 18)), E("tractor", -18, (15, 30), dur=4.0), E("bark", -20, (11, 22), count=2)]),
    "school": dict(bed=None, layers=[L("crowd", -2, kind="enfants", voices=12), E("alarm_bell", -12, (16, 30), dur=2.0), E("ball_bounce", -14, (6, 12), opt=True, kind="basket", surface="concrete"),
                                     E("door_family", -12, (9, 18), kind="battante"), E("footsteps", -11, (5, 9), ground="hard_floor", gait="run", steps=6)], room="hall"),
    "scifi": dict(bed=None, layers=[L("dark_drone", -3, wide=False, note=40.0), L("force_field", -12, wide=False, freq=80), L("hvac", -12, wide=False, state="fan"), E("cyber_ui", -12, (4, 9), kind="scan"), E("servo", -13, (7, 14)),
                                    E("door_family", -9, (10, 19), kind="hi_tech")]),
    "seaside": dict(bed="sea", layers=[L("surf", -2, distance=60.0), E("bird_kind", -10, (5, 11), kind="marins"), E("birdsong", -14, (7, 15), species="gull"), L("crowd_murmur", -19, people=8, mood=0.6, distance=1.0)]),
    "sport": dict(bed=None, layers=[L("crowd", -1, kind="ovation", voices=16), E("crowd", -4, (7, 13), kind="oh"), E("applause", -8, (9, 16), dur=3.0), E("pea_whistle", -8, (9, 16), opt=True, blasts=2),
                                    E("ball_bounce", -14, (4, 8), opt=True, kind="basket")]),
    "suburban": dict(bed="meadow", layers=[L("dawn_chorus", -7, birds=5), E("pass_by", -6, (8, 15), dur=5.0, speed=45.0), E("bark", -16, (10, 20), count=2), E("mower", -17, (16, 30), dur=4.0), E("car_door", -14, (11, 20))]),
    "swamp": dict(bed="swamp", layers=[L("night_insects", -3, density=0.7, temp=26.0), E("frog", -5, (4, 9), kind="croak", calls=4), E("frog", -9, (5, 10), kind="trill", calls=2), E("bubbles", -14, (8, 15), dur=2.0),
                                       E("mud", -14, (9, 18), opt=True, kind="bulles", dur=2.0)]),
    "town": dict(bed="town", layers=[L("crowd_murmur", -5, **TALK), E("pass_by", -5, (7, 14), dur=5.0, speed=40.0), E("distant_bell", -14, (12, 24), distance=300.0), E("bike_bell", -14, (10, 20)),
                                     E("footsteps", -12, (6, 11), ground="stone", steps=6)]),
    "traffic": dict(bed="city", layers=[L("tyre_roll", -7, speed=60.0), E("pass_by", 0, (2, 5), dur=4.0), E("truck", -1, (8, 15), kind="truck_passby", dur=5.0), E("motorcycle", -4, (9, 18), kind="passby", dur=4.0),
                                        E("car_horn", -12, (8, 16), dur=0.5)]),
    "transportation": dict(bed=None, layers=[L("crowd_murmur", -3, **TALK), L("escalator", -13, wide=False), E("rail", -2, (14, 24), kind="metro_start"), E("alert_tone", -14, (12, 22), preset="information"),
                                             E("footsteps", -10, (4, 8), ground="hard_floor", steps=6)], room="hall"),
    "tropical": dict(bed="jungle", layers=[L("night_insects", -4, density=0.7, temp=28.0), L("rain_on", -9, surface="feuilles", intensity=0.3), E("bird_kind", -8, (4, 9), kind="tropicaux"), E("primate", -13, (11, 20), kind="singe"),
                                           E("birdsong", -14, (8, 15), species="cuckoo")]),
    "tundra": dict(bed="tundra", layers=[L("snowfall", -10, hiss=0.3), E("gust", -3, (7, 14), through="canyon"), E("wild_animal", -14, (15, 28), kind="loup"), E("ice_crack", -14, (11, 22), distance=80.0)]),
    "underground": dict(bed="cave", layers=[L("cave_drips", -3), L("noise_bed", -10, kind="rumble"), E("rail", -9, (15, 25), kind="metro_start")]),
    "underwater": dict(bed=None, layers=[L("underwater_ambience", 3), E("bubbles", -8, (6, 12), dur=2.0, underwater=1), E("whale", -15, (12, 24)), E("submarine", -14, (16, 28), dur=6.0)]),
    "urban": dict(bed="city", layers=[L("crowd_murmur", -6, **TALK), E("pass_by", -2, (4, 8), dur=4.0, speed=50.0), E("bus", -8, (14, 26), dur=5.0), E("siren_wail", -18, (15, 28), kind="police_fr", dur=4.0, bright=0.5), E("car_horn", -14, (8, 16), dur=0.5),
                                      E("air_system", -14, (12, 22), kind="air_dryer_purge")]),
    "warfare": dict(bed="war", climax=0.7, layers=[E("firearm", -6, (3, 7), distance=120.0, shots=4, echo=0.3, render="pressure"), E("firearm", -9, (4, 9), weapon="ak47", distance=200.0, shots=3, echo=0.3, render="pressure"), E("howitzer", -5, (7, 13), distance=900.0),
                                                   E("shell_incoming", -7, (8, 15), distance=300.0), E("crowd", -11, (9, 17), kind="combat", voices=8, dur=3.0), E("shout", -13, (7, 13), kind="cri de guerre", people=3)], room="street"),
    "wildfire": dict(bed="burn", layers=[L("fire", 2, size=1.0, wind=0.6), E("crack", -6, (3, 7), kind="branche"), E("tree", -9, (11, 22), kind="chute"), E("gust", -5, (7, 13), through="leaves")]),
    "cacophony": dict(bed="city", layers=[L("crowd", -2, kind="panique", voices=12), E("car_alarm", -4, (5, 9), seq="01", motif=3.0), E("air_horn", -4, (3, 6), dur=1.2), E("siren_wail", -4, (5, 9), dur=4.0), E("jackhammer", -5, (4, 8), dur=2.0), E("alarm_bell", -7, (4, 8), dur=2.0)]),
    "environmental": dict(bed="open", layers=[L("noise_bed", -9, kind="hum"), E("pass_by", -9, (7, 14), dur=4.0), E("flyby_jet", -13, (18, 30), dur=8.0, closest=800.0)]),
}

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
