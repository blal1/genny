# genny documentation

| If you want to… | Read |
|---|---|
| Make sounds from JSON specs or the command line | [`../AGENTS.md`](../AGENTS.md): spec format, notation, tempo and loops, recipes |
| Look up a name and its parameters | [`catalog.md`](catalog.md) (generated; same data as `genny list --json`) |
| Know which physical model is behind a sound, and its source | [`physics.md`](physics.md); every constant's source: [`../research/SOURCES.md`](../research/SOURCES.md) |
| Call genny from Python | [`api.md`](api.md) (generated: every public class and function) |
| Add an instrument, sound, effect or layer type | "Extending genny" below |

## Module pages

Each page lists the names the module registers, their parameters with physical meaning, JSON recipes, the
numbers its tests measure and the constants that have no source.

| Page | Contents |
|---|---|
| [`contact.md`](contact.md) | Solids: impact, bounce, drop, smash, crumple, roll, scrape, modal bodies of any shape and material |
| [`friction.md`](friction.md) | Stick-slip: squeaks, brakes, rubbed glass, bowed bars; Cristal Baschet and the Baschet structures |
| [`matter.md`](matter.md) | Liquids, gases, fire, electricity, ice: bubbles, pouring, steam, flames, sparks, cracking ice |
| [`foley.md`](foley.md) | Objects, weapons, machines, signals (DTMF, phones, sirens), sci-fi, shakers, bells |
| [`creatures.md`](creatures.md) | Animals, insects, birds, body sounds, crowds, footsteps |
| [`choir.md`](choir.md) | Sung voice and choir: `voice`, `vocal_choir`, `sing`, `satb` |
| [`acoustic.md`](acoustic.md) | Physical strings, keys and winds: newer string instruments (kantele, tanpura, zanza...), guitar / Rhodes / clav / winds options, string and piano helpers |
| [`fdstring.md`](fdstring.md) | Finite-difference strings and bars: stiff, hammered, prepared, bowed, tension-modulated |
| [`plates.md`](plates.md) | Membranes and plates: drums, cymbals, gongs, struck panels, plate reverb |
| [`tubes.md`](tubes.md) | Acoustic tubes: reeds, bottles, pipes, vocal tract, duct effect |
| [`reverbs.md`](reverbs.md) | Rooms, distance, Doppler, rotary speaker, flanger, phaser, chorus, tape delay |
| [`spectral.md`](spectral.md) | Time-stretch, pitch shift, vocoder, spectral gate, freeze, analysis and resynthesis |
| [`dsp.md`](dsp.md) | Convolution, FIR and steep filters, overdrive, codecs, lo-fi, tape, wall; newer core effects and `filters` |
| [`osc.md`](osc.md) | Oscillators and noise: band-limited classic waves, supersaw / JP-8000, FM, CZ, DPW, BLIT, hard sync, pitched noise |
| [`retro.md`](retro.md) | sfxr and ZzFX engines, AY-3-8910 voice, tracker `song` layer |
| [`chiptune.md`](chiptune.md) | NES, Game Boy, YM2612, YM2151, YM2610, SN76489, SID and POKEY sound chips: `chip` layer (hardware-accurate) and per-channel voices |
| [`compose.md`](compose.md) | Generated music: `compose`, `arrangement`, `euclid`, `arp`, `stinger`, `scatter`, `adaptive` |
| [`cyber.md`](cyber.md) | Machines (transformer, gearbox, bearing), waveguide engine, turbo, car pass-by, sci-fi interface cues, glitch and voice effects, texture synthesis, machine identification |
| [`klang.md`](klang.md) | Ports of the Klang procedural library (vehicles), UI ticks, `sampler` and `articulate` layers, QA harness |
| [`household.md`](household.md) | Locks, doors, windows, tools, appliances, alarms, plumbing, games, furniture (models of the less obvious entries) |
| [`sports.md`](sports.md) | Balls, whistles, skates, swimming (models of the less obvious entries) |
| [`arms.md`](arms.md) | Firearms and heavy weapons: `firearm`, `minigun`, `gun_mech`, impacts, ricochets, blasts, launchers, shells, underwater charges |

Modules without a page of their own (names and parameters in [`catalog.md`](catalog.md), "Index by module"):

| Module | Contents |
|---|---|
| `melee` | Blades, spears, armour, grabs, crossbows, arrows, slings, catapults, siege impacts (Johnson's wedge contact) |
| `energy_arms` | Stun guns, tasers, cattle prods, arcs, blasters, plasma, railgun, coilgun, gauss rifle, laser impacts, photon torpedo |
| `human` | Body sounds (burp, hiccup, spit, gargle...), shouts, babies, crowds, beatbox, yodel, rap |
| `fauna` | Farm and wild animals, birds by kind, whales, elephants, rodents, reptiles, swarms, animal steps |
| `machines` | Power tools, jackhammer, dental drill, lifts, escalators, HVAC, pumps, gym and fairground rides, industrial presses |
| `engines` | Engine layouts and firing orders, diesel, two-stroke, turbo and supercharger, start/stall/shift events, race car, jalopy |
| `heavy_vehicles` | Motorcycles, scooters, trucks, buses, tractors, mowers, chainsaws, tracked vehicles, hydraulics, excavators |
| `road` | Electric drive and AVAS, tyre noise, car doors, horns, air brakes, alarms, suspension, interior |
| `rail` | Trains and metros, traction inverters, curve squeal |
| `marine` | Ships above and under water, ship horns and fog signals, outboards, jet skis, sailboats, submarines, lapping water |
| `aero` | Sonic boom, propeller and jet engines, helicopters, airships, gliders, cabins, rockets, drones |
| `nature` | Fireworks, volcanoes, tornadoes, tsunamis, singing sand, rain on surfaces, storms, foliage, rapids, fountains |
| `viscous` | Mud, slime, gore, blood, bone, chemistry |
| `designed` | Cartoon, UI and magic cues, braams, Morse, modems, alert tones, rotary dials, toys, microphones, Geiger counter |
| `scenes` | Ambience beds by category (`scene kind=`), room tones |
| `instruments_extra` | `ukulele`, `zither`, `shamisen`, `bagpipes`, `shofar`, `mellotron`, `calliope`, `waterphone`, `berimbau`, `talking_drum`, `gendyn`, `risset_arpeggio`... |
## Using genny from Python

```python
import numpy as np
import genny
from genny import spec, instruments, drums, sfx, fx
from genny.core import write_wav

# 1. Anything a JSON spec can do
y, sr = spec.render_spec({"bpm": 100, "layers": [
    {"type": "seq", "inst": "guitar", "step": 1, "steps": "E3:1 G3:1 B3:1 E4:2"},
    {"type": "sfx", "kind": "drip", "at": 4}], "fx": [{"type": "room", "preset": "hall", "mix": 0.2}]})
write_wav("out.wav", y, sr)

# 2. One note, one hit, one effect (the registries behind the catalog)
note = instruments.render_note("cristal", 440.0, 1.5, sr=44100, vel=0.8)
hit = drums.render_drum("plate_gong", 44100, 0.9)
drop = sfx.render_sfx("impact", 44100, material="glass", size=0.2)
wet = fx.apply_chain(note, [{"type": "plate_reverb", "mix": 0.3}], 44100)

# 3. The physical models themselves
from genny.contact import ShapeBody
from genny.strings import DispersiveString, StringPhysicalProperties
string = DispersiveString(StringPhysicalProperties.from_frequency(196.0, material="steel", t60_s=4.0), sr=48000)
audio = string.render_pluck(3.0, velocity=0.8, position=0.15)
```

`genny.instruments.REGISTRY`, `genny.drums.REGISTRY`, `genny.sfx.REGISTRY`, `genny.fx.REGISTRY` and
`genny.spec.LAYER_TYPES` hold every catalog entry: `{"fn", "desc", "params"}` (instruments also `"family"`
and `"range"`).

## Extending genny

A sound exists for users only once it is registered: JSON specs, the CLI and `genny list` read the
registries.

```python
from .sfx import sfx                  # one-shot or textural sound
@sfx("name", "One-line description.", size=(0.1, "m"), seed=(0, "variation"))
def name(sr=DEFAULT_SR, size=0.1, seed=0): ...            # -> mono float64 array

from .instruments import instrument   # pitched voice, used by synth / seq layers
@instrument("name", "Description.", family="plucked", span=("C2", "C6"), bright=(1.0, "0.5 .. 2"))
def name(freq, dur, sr=DEFAULT_SR, vel=1.0, bright=1.0): ...   # -> the note plus its release

from .drums import drum
@drum("name", "Description.", tune=(120, "Hz"))
def name(sr=DEFAULT_SR, vel=1.0, tune=120): ...

from .fx import effect                # x is mono (n,) or stereo (n, 2)
@effect("name", "Description.", mix=(0.3, "wet 0..1"))
def name(x, sr=DEFAULT_SR, mix=0.3): ...

from .spec import layer_type          # a new "type" for JSON layers
@layer_type("name", "One-line description.")
def render(layer: dict, sr: int, q: float): ...            # q = seconds per time unit (60 / bpm, else 1)
```

Put the code in a module of `genny/` and add the module's name to the list at the end of
`genny/__init__.py`. Every parameter is declared as `name=(default, "help")`; undeclared parameters are
rejected at render time.

What every entry must satisfy (the tests enforce it):

1. Works at 22050, 44100 and 48000 Hz. Run a kernel at its own rate and resample at the boundary if the
   scheme needs it (`genny.core.resample` is band-limited).
2. Finite, bounded (peak below about 1.5), no DC, starts and ends at zero.
3. Deterministic: randomness comes from a `seed` parameter through `np.random.default_rng(seed)`.
4. Per-sample loops are compiled with `@numba.njit(cache=True)`.
5. Instruments are within 10 cents of the requested pitch at the bottom, middle and top of their `span`,
   measure about -12 dB K-weighted at velocity 0.9 (`uv run python examples/level_match.py` rewrites the
   trims in `genny/_levels.py`) and do not carry much energy above 5 kHz by default
   (`tests/test_instruments.py` iterates over every registered instrument).
6. A test checks a prediction of the model, not only that it runs: a mode frequency against its formula, a
   decay time against the request, an event rate, energy conservation of a lossless scheme.
7. No source citation in the code (no paper, author, table or figure in comments or docstrings): the source of
   each constant, or its UNSOURCED status, is a line of `research/SOURCES.md` under its module.

After adding or changing entries:

```
uv run python examples/level_match.py       # instrument level trims
uv run python examples/build_docs.py        # docs/catalog.md and docs/api.md
uv run --with pytest pytest -q              # whole suite
uv run python -m genny.klang --qa           # renders every catalog entry and flags level, DC, clicks
```
