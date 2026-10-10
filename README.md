# genny

Offline sound and music synthesis in Python: physical models of instruments, objects, liquids, fire, voices
and rooms, rendered to WAV from JSON specs, a command line or Python. Built to be driven by AI agents
(see `AGENTS.md`), and deterministic: the same spec always gives the same file.

Version 1.1.0: **136 instruments, 48 drums, 234 sound effects, 96 effects, 14 generative and vocal layer
types**, plus a speech synthesiser (English, Spanish). Python only (numpy, scipy, numba).

On PyPI: [pypi.org/project/genny](https://pypi.org/project/genny/). `pip install genny` gives the library and
the `genny` command.

```
python install.py --tool          # installs the `genny` command (needs uv) and the Claude Code skill
uv tool install --editable .      # or: just the command
genny sfx impact -p material=glass -p size=0.2 -o glass_tap.wav
genny sfx pour -p dur=4 -o pour.wav
genny seq guitar "E3:1 G3:1 B3:1 E4:2" --bpm 100 --fx room:preset=hall,mix=0.2 -o tune.wav
genny seq cristal "A4:2 C5:2 E5:4" --bpm 60 -o cristal.wav
genny say "level up!" -p voice=female -o level_up.wav
genny render examples/library/instrument_tunes.json    # a short tune for each of the 129 instruments
genny render examples/library/style_pieces.json        # a generated piece for each of the 35 styles
genny render examples/library/cyber.json               # 47 machines, vehicles, interface cues and damaged voices (new in 1.0.1)
genny list                                             # every name and parameter (--json for agents)
genny info tune.wav                                    # loudness, sharpness, energy above 5 kHz
```

In a program, a sound is one call that returns a numpy array (`pip install genny`):

```python
import genny
from genny.sfx import render_sfx
from genny.core import write_wav

tap = render_sfx("impact", 44100, material="glass", size=0.2)          # float array, -1..1
ok = render_sfx("cyber_ui", 44100, kind="confirm", seed=3)              # another seed, another cue
engine = render_sfx("pipe_engine", 44100, rpm=3200.0, cylinders=6, dur=2.0)
write_wav("engine.wav", engine, 44100)
```

`examples/in_your_code.py` shows the patterns a project needs: a cache keyed by the arguments, an event
table for an interface, variations by seed, an engine bank by rpm, sounds driven by program state, effects
on an array, a scene built from a dict, and WAV bytes in memory.

| Read | For |
|---|---|
| `AGENTS.md` | the spec format, note and step notation, tempo and loops, recipes |
| `docs/README.md` | the documentation index, the Python quick start and how to add a sound |
| `docs/catalog.md` | every instrument, drum, sound effect, effect and layer type with its parameters |
| `docs/physics.md` | which physical model is behind each sound, its source, and the known limits |
| `docs/api.md` | the Python API |
| `examples/in_your_code.py` | genny called from a program: caching, variations, parameter banks, scenes |
| `examples/library/*.json` | demo batches; renders go to `sounds/` |

What is modelled: plucked, hammered, bowed and prepared strings (waveguide and finite difference); bars,
membranes, linear and nonlinear plates (cymbals, gongs); reeds, lips, jets and tubes; the singing voice and
choirs with lyrics; impacts, bouncing, rolling, scraping, breaking and crumpling of solids by material and
size; stick-slip friction and the Baschet sound sculptures; bubbles, pouring, steam, flames, sparks and ice;
animals, insects, birds, footsteps and crowds; engines, machines, weapons and signals; rooms, distance,
Doppler and occlusion; spectral processing, convolution, codecs; sfxr / ZzFX retro engines; generated music
in 35 styles. `docs/physics.md` has the full table.

Checks: `uv run --with pytest pytest -q` (the tests compare each model with its own physics: mode
frequencies, decay times, tuning, event rates); `uv run python -m genny.klang --qa` renders every catalog
entry and flags level, DC and clicks. Nothing was validated by listening tests: audition
`sounds/instrument_tunes/`, `sounds/style_pieces/` and `sounds/choir_audition/` and tune to taste.

Licensing: parts of `genny/physical/`, `genny/klang.py` and a few functions elsewhere are adapted from the
Klang procedural library, Copyright (c) 2025 Chris Nash, Klang Open License 1.0 (Apache 2.0 plus an
attribution requirement for interactive products). See `THIRD_PARTY_NOTICES.md` before shipping a game.

The sections below document the Python physical API as it grew, version by version.

## Physical synthesis API (v0.3)

Genny now includes a physically informed engine alongside the original synth/drum/SFX API. Existing
calls remain available; new code can compose reusable physical models:

```python
import genny

# Extended Karplus-Strong + commuted body
string = genny.StringInstrument(body="guitar", position=0.18, brightness=0.6)
audio = string.pluck(220.0, 2.0, velocity=0.8)

# Nonlinear felt hammer + stiff, dispersive piano strings
piano = genny.Piano(felt=0.2, inharmonicity=4e-4)
audio = piano.note(261.63, 3.0, velocity=0.75)

# PhISEM / stochastic particles
shaker = genny.ParticleMaterial("Maraca")
audio = shaker.shake(1.5, energy=0.8)

# Minnaert bubble population
water = genny.BubblePopulation(rate=80)
audio = water.render(3.0)

# Articulatory voice and avian syrinx
voice = genny.VocalTract(tract_scale=1.1)
audio = voice.phonate(120.0, 1.0)

bird = genny.BirdSyrinx(pressure_pa=3000)
audio = bird.call((650, 657), 1.0)

# Environment and physical room/propagation
rain = genny.Environment(seed=7).rain(5.0, intensity=0.8)
room = genny.Room(volume=120, surfaces={"plaster": 80, "wood": 35, "glass": 15})
rain_in_room = room.reverberate(rain, mix=0.25)
```

Low-level research engines are exposed under `genny.physical`:

- `modal`: modal banks, strike-position coupling, micro-collisions, fracture, aeroacoustic swooshes;
- `particles`: PhISEM shakers, granular footsteps, Minnaert bubble fields and drips;
- `waveguides`: plucked/bowed strings, mandolin, brass, clarinet, recorder and felt piano;
- `voice`: LF glottis, Pink Trombone tract, formant/FOF tools and syrinx;
- `space`: Zita/Puckette/Farnell reverbs, Sabine T60, distance and air absorption;
- `analysis`: LPC/lattice analysis-resynthesis, PSOLA/granular tools and modal fitting;
- `ambience`: wind, fire, water, rain, thunder, surf, creaks and object-detail textures;
- `footsteps`: physically/statistically informed surface interactions.

The physical kernels run internally at 48 kHz and the high-level API resamples to the requested Genny
rate, preserving the original API default of 44.1 kHz.

### Licensing note

Parts of `genny/physical` are adapted from the supplied Klang/procedural source. See
`THIRD_PARTY_NOTICES.md` and `THIRD_PARTY_KLANG_LICENSE.txt`. The Klang license contains an additional
attribution requirement for interactive audio-visual products.

## Physical engine v0.4

Version 0.4 extends the physically informed API introduced in v0.3 with additional
causal models derived from the supplied research corpus and the adapted Klang kernels:

```python
import genny

# Geometry + material -> bending modes
bar = genny.BarBody(genny.MATERIALS["steel"], length_m=0.35, radius_m=0.006)
y = bar.strike(duration=1.5, velocity=1.2)

# Bidirectional bowed-string waveguide
violin = genny.BowedString(body="violin", bow_pressure=0.9)
y = violin.note(440.0, 2.0)

# Jet-driven flute
flute = genny.Flute(noise=0.18)
y = flute.note(523.25, 1.5)

# GRF + modal/PhISEM footstep
step = genny.Footstep("wood")
y = step.render(seed=12)

# Reynolds/Strouhal aeroacoustics for a swept object
y = genny.AeroacousticSwing("SWD1").render(seed=3)

# Moving propagation: Doppler emerges from time-varying delay
prop = genny.MovingPropagation()
distance = np.linspace(20.0, 2.0, len(y))
y_moving = prop.apply_trajectory(y[:, 0] if y.ndim == 2 else y, distance, sr=44100)
```

The legacy `genny.instruments`, `genny.acoustic`, `genny.sfx`, `genny.vehicle`, etc.
remain available; the physical API is additive so existing code does not need to migrate all at once.

## v0.5 unified legacy names

Genny 0.5 begins routing familiar high-level names through the new physical engine. Existing calls such as `render_note("piano", ...)`, `render_note("violin", ...)`, `render_note("clarinet", ...)`, and `render_sfx("footstep", ...)` keep their API but now use physical waveguide/modal/contact models underneath. See `PHYSICAL_ENGINE_CHANGELOG_V05.md` for the migrated list and compatibility notes.

## 0.6 physical catalogue migration

Genny 0.6 moves more of the historical public catalogue onto the common physical engines. New reusable facades include:

```python
import genny

# Modal / struck bodies
bar = genny.TunedPercussion("marimba", hardness=0.7)
y = bar.note(440.0, 1.0)

# Physical/commuted plucked strings
mandolin = genny.PluckedString("mandolin", position=0.35)
y = mandolin.note(330.0, 1.0)

# Four-stroke engine; firing rate = cylinders * RPM / 120
engine = genny.CombustionEngine(cylinders=4, rpm=1800, load=0.7)
y = engine.render(2.0)

# Pouring combines stream texture, bubble resonances and a rising air-cavity mode
y = genny.Environment(seed=4).pour(2.0, flow=0.8, fill=0.1)
```

The legacy `xylophone`, `glockenspiel`, `tubular_bell`, `mandolin`, `steel_guitar`, `banjo`, `shaker`, `tambourine`, `woodblock`, `gong`, `rain`, `fire`, `stream`, `pour`, `engine` and `swoosh` catalogue names now route through these physical/statistical engines.

## Genny 0.7 physical families

Version 0.7 adds shared membrane, plate, surface-contact and rotor/aeroacoustic engines. The important point is that these are reusable physical models rather than one generator per named sound.

```python
import genny

# Circular membrane / hand drum physics
mem = genny.MembraneDrum(frequency=180, radius_m=0.16, air_coupling=0.4)
y = mem.strike(0.8, velocity=0.9, position=0.2, stroke="tone")

# Nonlinear cymbal / thin plate
y = genny.CymbalPlate(profile="crash").strike(2.0, velocity=0.9)

# A surface can be scraped or rolled over using the same spatial roughness model
surface = genny.SurfaceContact(roughness=0.7, characteristic_mm=1.2)
scrape = surface.scrape(1.0, speed=0.3)
roll = surface.roll(1.0, speed=0.5, radius_m=0.03)

# Rotor formulae are physical controls, not post pitch shifting
fan = genny.Rotor(rpm=1800, blades=5, radius_m=0.15)
y = fan.render(2.0)
print(fan.blade_passing_hz)  # 150 Hz

jet = genny.JetEngine(rpm=9000, throttle=0.8).render(3.0)
heli = genny.Helicopter(main_rpm=320, main_blades=4, load=0.7).render(3.0)
```

Existing names such as `conga`, `tabla`, `timpani`, `crash`, `ride`, `hihat`, `koto`, `sitar`, `fan`, `propeller`, and `helicopter` now map to these shared engines while retaining their public parameters.

## Genny 0.8: physical ports, machine sounds and physical kick/snare

Genny 0.8 introduces typed physical ports so models can exchange conjugate SI
variables rather than only normalized audio arrays:

```python
import numpy as np
import genny

force = np.ones(256)
velocity = np.ones(256) * 0.1
port = genny.PhysicalPort.mechanical(force, velocity, name="hammer")
print(port.effort_unit, port.flow_unit)  # N, m/s
```

The new physical drum models are directly available:

```python
kick = genny.KickDrum(frequency=52, beater_hardness=0.75)
a = kick.strike(0.6, velocity=0.9, sr=48000)

snare = genny.SnareDrum(frequency=185, snare_tension=0.8, wire_count=24)
b = snare.strike(0.45, velocity=0.85, snappy=0.8, sr=48000)
```

Machine models derive their important frequencies from geometry and speed:

```python
motor = genny.ElectricMotor(rpm=6000, commutations_per_rev=12)
print(motor.rotation_hz, motor.commutation_hz)
y = motor.render(2.0)

gears = genny.GearTrain(rpm=1800, driver_teeth=20, driven_teeth=40)
print(gears.mesh_hz, gears.output_rpm)
z = gears.render(2.0)
```

`AdvancedRollingContact` closes a weak contact↔resonator feedback loop rather
than treating texture and body resonance as unrelated layers. `Explosion`
combines a reduced N-wave shock, low-frequency expansion, debris and distance
propagation.

## Unified physical interactions (0.9)

Genny 0.9 standardizes instrument excitation/resonator interfaces using typed
physical ports while retaining the proven nonlinear waveguide kernels as the
internal high-rate solvers.

```python
import genny

interaction = genny.HammerStringInteraction(
    felt=0.18,
    brightness=0.65,
)
result = interaction.solve(440.0, 1.0, velocity=0.8)

# Audio renderer output
samples = result.audio

# Mechanical SI-domain interface traces
force = result.drive_port.effort       # newtons
velocity = result.drive_port.flow      # metres / second
power = result.drive_port.power        # watts
```

Acoustic interactions use pressure and volume flow:

```python
reed = genny.ReedBoreInteraction(breath=0.7)
result = reed.solve(220.0, 1.0)

pressure = result.drive_port.effort    # pascals
flow = result.drive_port.flow          # m^3 / s
```

The high-level models expose the same layer:

```python
piano = genny.Piano()
physical = piano.interaction()
audio = piano.note(440.0, 1.0)

violin = genny.BowedString()
physical = violin.interaction()

clarinet = genny.Clarinet()
physical = clarinet.interaction()
```

This is intentionally a staged refactor: in 0.9 the shared ports, units,
power accounting and public interaction contracts are unified, while the
specialized nonlinear waveguide loops remain their existing calibrated
solvers. Future versions can move the inner scattering loops onto the same
physical graph without another public-API change.

## Common physical interaction solver (0.10)

Genny now contains a shared sample-by-sample interaction core in
`genny.graphsolver`. It is the first step from model-specific waveguide loops to
one common causal physical graph.

```python
import genny

solver = genny.CommonPhysicalSolver

piano = solver.hammer_string(440.0, 1.0, velocity=0.8, sr=48000)
violin = solver.bow_string(220.0, 1.0, pressure=0.85, sr=48000)
clarinet = solver.reed_bore(220.0, 1.0, breath=0.7, sr=48000)

# SI-domain interface traces
force = piano.drive.effort       # N
hammer_velocity = piano.drive.flow  # m/s
pressure = clarinet.drive.effort    # Pa
volume_flow = clarinet.drive.flow   # m^3/s
```

The public `HammerStringInteraction`, `BowStringInteraction`,
`ReedBoreInteraction`, `LipBoreInteraction`, and `SyrinxTracheaInteraction`
currently use a hybrid renderer: the shared solver supplies the causal physical
interaction/ports while the older calibrated renderer remains mixed in for
sound quality during migration.

## v0.11 shared waveguide core

Genny 0.11 adds the next layer of the common physical solver. Fractional delays can now use linear or cubic Lagrange interpolation, while `ThiranDelay` supplies a magnitude-preserving allpass option for feedback loops. The common solver also exposes frequency-dependent passive losses, stiff-string dispersion, stateful stick/slip friction, multi-port scattering, tone holes, bell radiation and hydrodynamic jet delay.

```python
import genny

# Higher-accuracy fractional delay
line = genny.FractionalDelay(37.4, 256, method="lagrange3")

# Magnitude-preserving fractional delay for waveguide loops
thiran = genny.ThiranDelay(37.4, 256)

# Physical flute/edge-tone model now participates in the shared graph
flute = genny.Flute(pressure=0.8, jet_ratio=0.32)
audio = flute.note(440.0, 1.0, sr=48000)
```

The v0.11 common reed/bore and brass/bore solvers now include reduced tone-hole/bell termination behavior, and bowed strings use a stateful stick/slip model rather than a memoryless friction curve.

## 0.12 — passive multiport waveguide networks

Genny 0.12 adds `WaveguideBranch` and `WaveguideNetwork`.  A network joins any number of delayed acoustic branches through the common lossless impedance-weighted scattering junction.  This is the foundation for multi-hole bores, branched pipes, coupled strings and cavity networks.

```python
import genny

branches = [
    genny.WaveguideBranch(1.0, 17.3, 128),
    genny.WaveguideBranch(1.5, 23.7, 128),
    genny.WaveguideBranch(2.0, 31.2, 128),
]
network = genny.WaveguideNetwork(branches)
ir = network.render_impulse(4096)
```

The legacy `timpani`, `upright_bass` and `finger_bass` wrappers were also corrected for tuning/very-short-note compatibility while keeping their physical renderers underneath.

## Genny 0.13: multi-hole wind networks and coupled resonators

Genny 0.13 adds reduced real-time acoustic networks built from the passive waveguide primitives introduced in 0.11 and 0.12.

### Dynamic tone holes

```python
import numpy as np
import genny

clarinet = genny.MultiHoleClarinet(220.0, sr=48000)
openings = np.zeros((48000, len(clarinet.bore.holes)))
openings[24000:, -2:] = 1.0   # open two holes during the note

audio, reed_port, bore_port = clarinet.note(1.0, fingering=openings)
```

A tone hole may be anywhere from closed (`0`) to open (`1`). Internally each hole is a third acoustic branch attached to the two neighbouring bore segments through an impedance-weighted scattering junction.

### Multi-hole flute

```python
flute = genny.MultiHoleFlute(440.0, jet_velocity=18.0, sr=48000)
fingering = np.zeros(len(flute.bore.holes))
fingering[-1] = 0.65

audio, jet_port, bore_port = flute.note(1.0, fingering=fingering)
```

The excitation retains a hydrodynamic jet delay while the resonator is now a chain of bore segments, side holes and bell radiation rather than one monolithic delay.

### Coupled strings

```python
strings = genny.CoupledStringBank(
    frequencies=(440.0, 440.7, 439.3),
    coupling=0.08,
    sr=48000,
)

audio = strings.render(2.0, velocity=0.8)
```

This is intended for piano unisons, mandolin courses and sympathetic-string experiments. The strings exchange energy through a shared bridge state instead of being mixed only after synthesis.

### Multi-port acoustic cavity

```python
cavity = genny.AcousticCavityNetwork(
    volume_m3=0.025,
    port_radii_m=(0.01, 0.007),
    port_lengths_m=(0.03, 0.02),
    sr=48000,
)

impulse_response = cavity.render_impulse(4096)
```

The cavity is a reduced compliance connected to inertive neck/port branches. It is suitable for bottles, enclosures, coupled drum cavities and other low-order resonant systems.

These models are reduced real-time approximations. They are designed to preserve topology, coupling and causal interaction at low cost; they are not substitutes for full FEM/BEM simulation when exact geometry-dependent radiation is required.

## Genny 0.14: shared piano bridge, soundboard and sympathetic strings

Genny 0.14 adds a reduced real-time piano network in which unison and unstruck strings share one mechanical bridge and one modal soundboard. Energy can therefore move from a struck course into sympathetic strings instead of those strings being rendered independently.

```python
import genny

piano = genny.SympatheticPiano(
    struck_frequency=220.0,
    coupling=0.07,
    sr=48000,
)

audio, stems = piano.render(
    2.0,
    velocity=0.9,
    return_stems=True,
)
```

The existing high-level piano exposes the same network without changing the legacy-calibrated `note()` method:

```python
piano = genny.Piano()
audio = piano.sympathetic_note(
    220.0,
    2.0,
    coupling=0.07,
)
```

The shared bridge is a reduced mass-spring-damper system and keeps mechanical force/velocity state. The soundboard is a single damped modal bank driven by bridge force. This is a reduced-order real-time model, not a full finite-element piano simulation.

## v0.15 — Stateful polyphonic piano

Genny now includes a continuous shared-state piano engine.  Unlike rendering
separate notes and mixing them, `PolyphonicPiano` keeps one bridge, one modal
soundboard, per-note string courses, dampers, pedal state, a keyboard-wide
sympathetic bank, and passive aliquot modes alive across render calls.

```python
import genny

p = genny.Piano().polyphonic(sr=48000)
p.note_on(60, velocity=0.85)
p.note_on(64, velocity=0.75)
p.note_on(67, velocity=0.80)
attack = p.render(0.5)

p.sustain_pedal(True)
p.note_off(60)
p.note_off(64)
p.note_off(67)
tail = p.render(2.0)
```

For sample-accurate offline sequencing:

```python
p = genny.PolyphonicPiano(sr=48000)
audio = p.render_events([
    {"time": 0.00, "type": "note_on",  "note": 60, "velocity": 0.85},
    {"time": 0.00, "type": "note_on",  "note": 64, "velocity": 0.75},
    {"time": 0.30, "type": "sustain",  "down": True},
    {"time": 0.50, "type": "note_off", "note": 60},
    {"time": 0.50, "type": "note_off", "note": 64},
    {"time": 1.40, "type": "sustain",  "down": False},
], dur=2.0)
```

`Piano.note()` remains unchanged for backward compatibility, while
`Piano.sympathetic_note()` remains the lightweight single-note shared-network
renderer introduced in v0.14.


## v0.16 - Piano pedals, dampers and action geometry

The stateful `PolyphonicPiano` now models the three grand-piano pedal behaviours separately:

```python
p = genny.Piano().polyphonic(sr=48000)
p.note_on(60, 0.85)
p.sostenuto_pedal(True)       # captures only keys already held
p.note_off(60)
p.una_corda(True)             # subsequent strikes use fewer strings/softer action
p.set_damper(72, 0.45)        # partial per-note damper contact, 0=open, 1=closed
audio = p.render(2.0)
```

`render_events()` also accepts `sostenuto`, `una_corda`/`soft`, and `damper` events. Una corda changes excitation before the string network: multi-string courses strike one fewer string and use a softer hammer impulse. Sostenuto captures only the notes held at pedal-down, unlike sustain which opens all dampers. Per-note damper contact continuously changes string loss instead of gating the final signal.

Hammer strike position now varies across the keyboard and is represented by a weak delayed counter-wave in the excitation. Treble duplex/aliquot resonators receive a small note-dependent stretch rather than sitting on perfectly coincident integer harmonics. These are reduced-order realtime models, not a geometric FEM action.

## Genny 0.17 — piano polarization, longitudinal coupling, and hammer rebound

The stateful piano now models two transverse polarization planes per string, a weak longitudinal mode, note-position-dependent bridge mobility, and a nonlinear felt hammer contact with rebound diagnostics.

```python
import genny

p = genny.Piano().polyphonic(sr=48000)
p.note_on(60, 0.85)
audio = p.render(1.0)

course = p.courses[60]
print(course.polarization_energy())
print(course.longitudinal_energy())
print(course.hammer_action.last_contact_time_s)
print(course.hammer_action.last_rebound_velocity_m_s)
```

The hammer contact follows a reduced nonlinear felt law:

`F = K * x**p`

while each course contains a primary and orthogonal transverse delay loop with a small frequency split. A weak longitudinal resonance exchanges energy with the transverse motion at the bridge. `FrequencyDependentBridge.step_note_forces()` keeps one shared bridge state while returning slightly different local mobilities for different keyboard regions.

These are realtime reduced-order models: they preserve the coupling topology and physically meaningful state, but they are not substitutes for full 3D finite-element analysis.

## v0.18 — shared physical string core

Genny now has one explicit reduced-order string model that can be reused by piano,
guitar, harp and related instruments.

```python
import genny

props = genny.StringPhysicalProperties(
    length_m=0.65,
    radius_m=0.00035,
    material="steel",
    frequency_hz=440.0,
)

print(props.tension_n)
print(props.characteristic_impedance_n_s_m)
print(props.flexural_rigidity_n_m2)
print(props.inharmonicity_B)

string = genny.DispersiveString(props, sr=48000)
audio = string.render_pluck(2.0, velocity=0.8, position=0.22)
```

The ideal fundamental is

`f0 = (1/(2L))*sqrt(T/mu)`

and the string-side characteristic impedance is

`Z = sqrt(T*mu)`.

For a circular string, `I = pi*r^4/4` and flexural rigidity is `E*I`.  Stiff-string
partials use the reduced approximation

`f_n ~= n*f0*sqrt(1 + B*n^2)`

with `B = pi^2*E*I/(T*L^2)`.

The longitudinal-wave estimate is `c_L=sqrt(E/rho)` and
`f_L=c_L/(2L)`.

Existing plucked-string facades remain compatible, while the common core is available
explicitly:

```python
inst = genny.PluckedString(kind="guitar")
audio = inst.physical_note(196.0, 1.5, sr=48000)
```

`PianoStringCourse` now also creates one `StringPhysicalProperties` object for every
unison string and derives dispersion from its physical inharmonicity.  The model is a
realtime reduced-order waveguide, not a full 3-D FEM string/bridge solver.

## v0.19 — shared string migration and wound strings

Genny now uses the same reduced-order physical string core for the main plucked and bowed families. `WoundStringPhysicalProperties` represents a load-bearing core plus a helical winding and exposes its mass per unit length, tension, characteristic impedance, effective bending rigidity and inharmonicity.

```python
import genny

bass_string = genny.WoundStringPhysicalProperties.from_frequency(
    55.0,
    length_m=0.864,
    core_material="steel",
    winding_material="nickel",
)

audio = genny.DispersiveString(bass_string, sr=48000).render_pluck(
    2.0, velocity=0.8, position=0.28
)
```

The winding mass follows the helical path rather than treating the string as a single solid cylinder. Bending stiffness is a reduced partially-coupled core/winding approximation; this is not a contact-resolved helical FEM model.

`DispersiveString` also supports bowed excitation with stateful stick/slip hysteresis:

```python
string = genny.DispersiveString(
    genny.StringPhysicalProperties.from_frequency(440.0, material="steel"),
    sr=48000,
)
violin_like = string.render_bow(1.5, bow_velocity=0.18, bow_pressure=0.75)
```

The public guitar, harp, mandolin, steel guitar, koto, sitar, bass, violin and cello wrappers source their string motion from this shared core. Instrument bodies, pickups and jawari layers remain specialised radiation/post-processing stages.

## Genny 0.20 — local string interactions

The shared physical string core now models common local playing interactions rather than treating them as unrelated effects.

```python
import genny

props = genny.WoundStringPhysicalProperties.from_frequency(
    82.41,
    length_m=0.648,
)
string = genny.DispersiveString(props, sr=48000)

# Pick/finger/nail hardness changes physical contact duration.
y = string.render_exciter(
    1.0,
    genny.StringExciter("pick", hardness=0.9, position=0.18),
    velocity=0.8,
)
```

Fretting preserves construction/tension and shortens the vibrating length according to equal temperament:

```python
fb = genny.FretboardGeometry(scale_length_m=0.648, action_m=0.0018)
y = genny.DispersiveString(props, sr=48000).render_fret_buzz(
    5, 1.0, fretboard=fb, velocity=0.9, buzz=0.75
)
```

A slide changes the waveguide delay continuously:

```python
y = genny.DispersiveString(props, sr=48000).render_glissando(
    1.2, 82.41, 123.47
)
```

Additional shared interactions include `render_natural_harmonic()`, `render_slap()`, bridge-local `palm_mute`, `PluckedString.local_note()`, `PluckedString.glissando()`, and `BowedString.glissando()`.

These are realtime reduced-order contact models. They preserve causal local excitation/contact behaviour but are not a full geometric string/fret FEM solver.

## v0.21 — expressive physical string gestures

The shared string core now supports expressive techniques inside the stateful waveguide rather than as post-render effects.

```python
import genny

guitar = genny.PluckedString("guitar", position=0.18)

bend = guitar.expressive_note(196, 1.0, technique="bend", cents=200)
vibrato = guitar.expressive_note(196, 1.0, technique="vibrato",
                                 depth_cents=20, rate_hz=5.2)
slide = guitar.expressive_note(196, 1.2, technique="slide",
                               end_frequency=293.66,
                               roughness=0.5, pressure=0.6)
buzz = guitar.expressive_note(196, 1.0,
                              technique="fret_buzz_feedback", fret=3)
tap = guitar.expressive_note(196, 0.8, technique="tapping", fret=12)
pinch = guitar.expressive_note(196, 1.0,
                               technique="pinch_harmonic", harmonic=4)
```

For a bend, fixed vibrating length and linear density imply

`T / T0 = (f / f0)^2`.

`render_bend()` and `render_vibrato()` therefore change the effective waveguide delay continuously. `render_fret_buzz_feedback()` detects a reduced string/fret penetration state and reinjects the collision impulse into the travelling wave, rather than adding fret noise only after the string has been rendered.

These remain realtime reduced-order interaction models, not contact FEM.

## v0.22 — continuous combined string performance

Genny 0.22 adds a stateful performance layer that combines expressive controls in one physical string loop instead of rendering one technique per call.

```python
import genny

guitar = genny.PluckedString("guitar")

audio = guitar.performance(
    196.0,
    1.5,
    sr=48000,
    gestures=[
        genny.StringGesture("bend", 0.1, 0.8, 0, 180),
        genny.StringGesture("vibrato", 0.35, 1.2, 12, rate_hz=5.4),
        genny.StringGesture("slide", 0.7, 1.15, 0, 3),
        genny.StringGesture("fret_contact", 0.5, 1.3, 0.3, 0.9),
        genny.StringGesture("palm_mute", 1.05, 1.5, 0, 0.8),
    ],
)
```

The same sample loop now combines effective vibrating length, tension, vibrato, loop damping, and nonlinear fret contact. `HertzFretContact` uses a reduced `F = k delta^(3/2)` law. `StringPerformance.last_diagnostics` reports the resulting frequency, length, tension ratio, inharmonicity and contact-force extrema.

This is still a realtime reduced-order model rather than a contact-resolved FEM string/fret simulation.
