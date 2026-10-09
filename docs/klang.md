# genny.klang — the Klang `procedural` library in genny

`genny/klang.py` holds the ports of the `procedural` project (`synthgen/procedural`: C++ sounds
written for the Klang framework, and the Python families built on the same kernels), the parts of
it no other genny module covers, and a QA harness for the whole catalog.

**Licence.** The `procedural` code is Copyright (c) 2025 Chris Nash, **Klang Open License 1.0**
(Apache-2.0 plus one condition: an interactive product such as a game that ships these sounds must
show the Klang logo and "Powered by Klang" in its credits or about page; artwork at
https://nash.audio/art/). Full text: `genny/physical/KLANG_LICENSE.txt`. Every name below marked
*port* is a derivative work under that licence.

## Catalog

### Sound effects

| name | what | origin |
|---|---|---|
| `toy_boat` | putt-putt toy boat engine | port, `vehicles/Motors.h` `ToyBoatEngine` |
| `klang_car` | car engine, `kind` `basic` or `mini` | port, `Motors.h` `Car`/`FourStrokeEngine`, `Mini` |
| `harrier` | jet engine: turbine whine, burn, wind | port, `vehicles/Harrier.h` |
| `bicycle` | freewheel whirr, tick and chain | port, `vehicles/Bicycle.h` |
| `klang_rain` | drop ticks over a noise bed | **not a port**: `nature/Nature.h` is empty (see below) |
| `ui_tick` | 11 physical interface cues | port, `interface.py` `ui_*` |
| `ui_chime` | 4 bell notifications | port, `interface.py` `channel_*`, `urgency_alert` |
| `damage` | physical / soul / mental hit on a character | port, `impacts.py` `impact_*` |
| `sniff`, `strain` | nasal sniffing; pressed vocal effort | port, `body.py` |
| `weight_shift` | feet shifting on granular ground | port, `body.py` |
| `handle` | picking up / equipping objects, 11 kinds | port, `world.py` `item_pickup`, `item_equip` |
| `tide` | tide rising / falling | port, `world.py` `tide_*` |
| `distant_bell` | tower bell at a distance, with street echoes | port, `world.py` `week_turn` |

**`toy_boat`** — `rate` (9): valve pulses per second, the header's constant; noise bursts come at
2 × rate because the valve signal has a bump on each edge. `broken` (0/1): the pulse is replaced
by 9 Hz band-passed noise (sputtering). `dur`, `seed`.

**`klang_car`** — `kind` `basic` | `mini`; `rpm` 0..7000; `rpm_end` (−1 = steady) for a linear rev
sweep; `dur`, `seed`.
- `basic`: pulse rate = rpm × 16 / 700 Hz (68.6 Hz at 3000 rpm). The level grows with rpm
  (× speed × min(speed, 0.25), speed = rpm/7000) and it is silent near 0.
- `mini`: 15 partials measured on a Mini at `partials × rate`, where
  rate = rpm/900 − 0.05 (rpm/900)² + min(throttle, 0.707); the two loudest sit at 86.1 × rate and
  64.6 × rate Hz (82 and 61 Hz at the 900 rpm idle). `throttle` 0..1 raises the rate up to 0.707;
  **exactly 1.0** floors it (over-rev: more exhaust rasp, resonant noise). `ignition` 1 = the
  engine is switched on at t = 0 (starter at half power, a rev blip of 2–5 × idle, settled after
  0.75–1.25 s); 0 = already running. `off_at` = second at which the ignition is cut (2 s run-down).

**`harrier`** — `speed` 0..1 (engine speed: turbine partials at speed × 3097 / 4495 / 5588 / 7471 /
11000 Hz, burn rumble at speed² × 150 Hz, overdrive 30..45); `speed_end` for spool-up or -down;
`altitude`: 0 = on the ground with no burn (turbine only — the C++ default), ≥ 2 = full burn,
then wind above speed 0.6 with gain min(200, altitude/2) × (0.5 + 3 speed); `gain` (0.5);
`echo` 0 = as the C++ behaves, 1 = an echo of period `speed` seconds (see quirks); `dur`, `seed`.

**`bicycle`** — `pedalling` 0 coasting .. 1 hard; `wheel_speed` 0..50 turns per second (the whirr
and the tick repeat at this rate); `pedal_speed` 0..4 strokes per second (the chain is modulated
at 22 × pedal_speed Hz); `dur`, `seed`. The output sits between 5 and 11 kHz by construction (the
header ends on a band-pass at 11 kHz): put a `lowpass` on the layer for a distant bike.

**`klang_rain`** — `intensity` 0..1 (2 + 600 × intensity² audible drops per second, and the bed
level), `hiss` (level of the bed, 0 = drops only), `tone` 0.5 soft ground .. 2 hard surface,
`dur`, `seed`. For a physically sourced rain use genny's `rain`.

**`ui_tick`** — `kind`: `focus` (soft wooden tick, 65 ms), `back` (double tick), `activate` (iron
click with a faint ring), `refuse` (dull thud, felt), `toggle` (four-click panel switch), `page`
(heavy page turn), `save` (book closing), `load` (book opening), `dialog_open` (catch, then lid),
`dialog_close` (lid, catch, end-stop), `hint` (felt on wood). `seed`: the same seed always gives
the same cue; `focus`, `back` and `hint` have no random part at all.

**`ui_chime`** — `kind`: `discovery` (Risset bell, 523 Hz), `achievement` (two telephone bells
650/653 Hz, 60 ms apart), `alert` (muted English bell, 195 Hz, dies in 0.6 s), `spell` (fingertip
on a glass rim, 880 Hz). `pitch`: ratio applied to those frequencies. `seed`.

**`damage`** — `kind`: `physical` (shockwave thump + whump + noise burst + N-wave through one body,
0.45 s), `soul` (one heartbeat lub-dub at 63 and 42 Hz with murmur, over a rumble, 1.7 s),
`mental` (a 200 Hz boom through a swept comb over three resonances 6 Hz apart at 90 Hz, low-passed
at 800 Hz, 1.6 s). `seed` = the take.

**`sniff`** — `pulls`, `pull` and `gap` (s), `size` (bigger body = lower). **`strain`** — `dur`,
`freq` (voice pitch), `tenseness` 0.5 lax .. 0.95 pressed, `size`. **`weight_shift`** — `ground`
(any footstep ground: sand, gravel, snow, leaves...), `dur`, `weight`.

**`handle`** — `kind`: pick-ups `weapon`, `armor`, `tool`, `food`, `glass`, `trinket`, `bell`;
equips `draw` (blade from a scabbard), `shield`, `don` (armour on), `amulet`. `seed` = the take.

**`tide`** — `kind` `rising` | `falling`, `dur`, `seed`. **`distant_bell`** — `distance` (m: air
absorption takes the highs; the level is normalised, set it with the layer `gain`), `size`
(1 = a 452 Hz bell, bigger = lower), `dur`, `seed`.

### Instruments

| name | span | params | built on |
|---|---|---|---|
| `viola` | C3–E6 | `attack`, `vibrato` | bowed voice with viola body formants 230 / 370 / 900 / 2000 Hz |
| `bassoon` | Bb1–E5 | – | reed voice, formants 480 and 1200 Hz |
| `cor_anglais` | E3–A5 | – | reed voice, formants 950 and 2300 Hz |
| `bass_clarinet` | D2–F5 | – | odd-harmonic reed voice, darker than `clarinet` |
| `piccolo` | D5–C8 | `breath` | near-sine fipple voice |
| `recorder` | C4–D7 | `breath`, `chiff` | fipple voice with an attack chiff |
| `hand_chime` | G3–C7 | `decay` | port of `charter.hand_chime`: modes 1, 3, 5.404 + clapper click |

These are the orchestral voices listed in `procedural/instrument_map.py` (the sampler's map of
CC0 recordings) that genny had no instrument for. The map itself only names recordings, so the
voices are built on genny's pitch-exact helpers (`_blown`, `_modes`); the formant centres are
typical textbook values, marked UNSOURCED in the code.

### Layer types

**`articulate`** — a line played with one of the sampler's eight articulations on any instrument:

```json
{"type": "articulate", "inst": "viola", "artic": "tremolo", "rate": 14, "steps": "G3:1 A3:1 Bb3:2"}
```
`artic`: `sustain`, `legato` (8 % overlap), `staccato` (35 % of the length), `hit` (60 ms),
`pizz` (the `pizzicato` instrument), `harmonic` (an octave up, softer), `tremolo` (re-struck
`rate` times a second), `roll` (the same with a crescendo over each note). Other keys as in a
`seq` layer: `steps`, `step`, `vel`, `transpose`, `params`.

**`sampler`** — a folder of recordings played as an instrument:

```json
{"type": "sampler", "dir": "samples/kalimba", "steps": "C4 E4 G4:0.5 [C4,G4]:1", "release": 0.15}
```
The mapping comes from the file names: a note name (`C4`, `F#3`, `As2`, `Bb1`; C4 = MIDI 60), a
dynamic token for the velocity layer (`v1`, `v2`.. / `pp`..`fff` / `soft`, `loud`), and files with
the same note and layer are round-robin takes used in turn. Each note takes the layer for its
velocity, the nearest recorded pitch, and is repitched by resampling. Keys: `dir`, `pattern`
(glob, default `*`; wav, flac, aiff, ogg), `root` (note of files that name none, default C4),
`steps`, `step`, `legato`, `gap`, `transpose`, `vel`, `release` (s; 0 = let every sample ring out),
`tune` (cents), `max_shift` (semitones; farther notes raise an error). A note longer than its
recording ends with the recording (no looping).

## Recipes

```json
{"out": "mini_start.wav", "layers": [
  {"type": "sfx", "kind": "klang_car", "params": {"kind": "mini", "rpm": 900, "ignition": 1, "dur": 5}}]}
```
```json
{"out": "mini_bank_3000.wav", "loop": 0.3, "duration": 3.3, "layers": [
  {"type": "sfx", "kind": "klang_car", "params": {"kind": "mini", "rpm": 3000, "throttle": 0.4, "dur": 3.6}}]}
```
```json
{"out": "jet_takeoff.wav", "layers": [
  {"type": "sfx", "kind": "harrier", "params": {"speed": 0.15, "speed_end": 0.95, "altitude": 3, "dur": 6},
   "fx": [{"type": "lowpass", "cutoff": 6000}]}]}
```
```json
{"out": "bike_pass.wav", "layers": [
  {"type": "sfx", "kind": "bicycle", "params": {"pedalling": 0.7, "wheel_speed": 8, "pedal_speed": 1.5, "dur": 4},
   "fx": [{"type": "lowpass", "cutoff": 7000}]}]}
```
```json
{"out": "menu_confirm.wav", "layers": [
  {"type": "sfx", "kind": "ui_tick", "params": {"kind": "activate"}},
  {"type": "sfx", "kind": "ui_chime", "params": {"kind": "discovery", "pitch": 1.5}, "at": 0.05, "gain": 0.5}]}
```

## What was checked (tests/test_klang.py)

| sound | prediction from the header | measured |
|---|---|---|
| `toy_boat` | envelope at 2 × rate: 18 Hz (rate 9), 28 Hz (rate 14) | 17.96 Hz, 27.94 Hz |
| `klang_car` basic | pulse rate rpm × 16/700: 34.29, 68.57, 114.29 Hz | 34.35, 68.58 (autocorrelation), 114.31 Hz; all strong partials are its harmonics |
| basic, sample 0 | 0.25 tanh(3 b0 (2 + 2/(s²+1)) speed · 0.25)/tanh 3, by hand | equal within 0.5 % |
| `klang_car` mini | 86.1 × rate and 64.6 × rate: 81.8/61.4 (900 rpm), 195.9/147.0, 323.5/242.7 Hz | 81.76/61.32, 195.90/146.95, 323.50/242.58 Hz |
| `harrier`, altitude 0 | loudest turbine partial 5588 × speed: 1676.4, 2794.0, 5029.2 Hz | 1676.39, 2794.02, 5029.18 Hz |
| `bicycle` wheel | pattern repeats at wheel_speed: 4, 6, 12 Hz | 3.99, 5.99, 11.97 Hz |
| `bicycle` chain | envelope at 22 × pedal_speed: 22, 44 Hz | 22.95, 45.9 Hz (frequency-modulated ±10 % by the pedal) |
| `klang_rain` | 2 + 600 i² drops/s: 520, 3040, 12040 in 20 s at i = 0.2, 0.5, 1 | 521, 3042, 12044 placed; at i = 0.1 without the bed, 71 ticks detected in the audio of the 73 placed in 10 s |

Monotonic controls: basic car −59 / −33 / −23 dB at 800 / 3000 / 6500 rpm; `harrier` −36.8 / −26.5
/ −24.5 dB at speed 0.2 / 0.5 / 1.0; `bicycle` −40.8 / −38.8 / −36.1 dB at pedalling 0 / 0.5 / 1;
Mini floored is louder and brighter than at throttle 0. The seven instruments are within 1 cent
at the bottom, middle and top of their span and at −12.0 dB. Three seconds of any C++ port render
in 0.03–0.35 s after JIT warm-up.

## Klang quirks kept, and deviations

The C++ runs at `klang::fs = 44100` and several constants are per sample, so the ports run at
44100 Hz and are resampled to the requested rate.

Kept on purpose (they are what the C++ sounds like):
- klang's `min`/`max` return the type of their **first** argument. `max(0, x)` therefore truncates
  `x` to an integer. In `Mini`: `gas` is 0 unless throttle is exactly 1; the comb feedback is on
  only while rate ≤ 1; the low-pass Q is an integer 5..10; one throttle term is all-or-nothing.
  In `Harrier`: the wind band-pass frequency and the wind gain are whole numbers, and the echo
  feedback `max(0, speed × 0.75)` is always 0, so the C++ echo never sounds. `echo=1` restores
  what the line evidently meant.
- `Mini` jitters every partial's frequency by ±20 % on every sample (`random(0.8, 1.2)`).
- `Burn`'s Pd-style band-pass at 8 kHz, Q 0.5 degenerates to a gain of 2 at 44.1 kHz, and its
  second `vcf` is a plain gain above speed 0.35.
- `fast::sin` is replaced by the exact sine and C `rand()` by a seeded xorshift (deterministic).

Changed, because the contract needs bounded output:
- `Harrier`'s wind band-pass centre is ≤ 0 Hz above altitude 5000 + 2000 × speed; the C++ filter
  is then unstable and the output becomes NaN. The port mutes the wind there.
- The control smoothers start settled instead of rising from 0 (the C++ start is a 100 ms chirp).
- Where the C++ exceeds full scale (`Harrier` wind at height, `Bicycle` above about 15 wheel turns
  per second, where the tick term grows with the cube of the speed) `render.cpp` hard-clips; the
  port scales the whole sound down to a peak of 0.98 instead.
- Added controls the headers do not expose: `rate` (toy boat), `rpm_end` / `speed_end` ramps,
  `ignition` 0 (skip the start), `off_at`.
- Not ported: `Mini`'s unused `Gear` control; the commented-out frame resonances in `Bicycle.h`.

**Rain.** `procedural/nature/Nature.h` contains only `#pragma once`; the README lists "Rain
(work-in-progress)". There was nothing to translate. `klang_rain` is an original small model
made of klang primitives (white noise, one-pole high-pass, biquad low-pass); every constant in it
is marked UNSOURCED.

**Tide.** `tide` passes the same dynamics curve to Reshetnikov's stream as `world.py` does. As the
dynamics rise (0.05 → 0.55) that kernel gets darker (share of energy above 2.5 kHz 0.7 → 0.02)
and quieter: a bright trickle becoming a deep flow. The falling tide runs the other way and ends
loud and bright over the shingle.

## QA harness

```
uv run python -m genny.klang --qa                       # every sfx, instrument, drum and effect
uv run python -m genny.klang --qa --only "klang_*" harrier --kind sfx --deep --json qa.json
```
```python
from genny.klang import coverage, report
print(report(coverage(only="plate_*", kinds=("drum",))))
```
Each entry is rendered with its defaults (instruments: the middle of their range, 0.5 s, vel 0.9;
effects: a 220 Hz tone with a click) and gets a row: `peak`, `loud` (K-weighted dB of the loudest
0.4 s), `sharp` (acum), `above5k`, `dur`, `seconds` (render time), and `flags`: `error` (with the
exception), `empty`, `non-finite`, `silent` (peak < 1e-4), `hot` (peak > 1.5), `dc`,
`ends-on-step`, `piercing` (an instrument sharper than 2.4 acum). `--deep` adds for sfx:
`non-deterministic`, `seed-ignored` (a `seed` param that changes nothing — `ui_tick` is flagged
truthfully: its default wooden tick has no random part) and `fails-22050`. One failing entry
never stops the run; the exit code is 1 if anything is flagged. `python -m` prints a harmless
runpy warning because the package imports the module itself.

First full run (2026-10-09, 44100 Hz): 492 entries (226 sfx, 129 instruments, 48 drums, 89 effects),
23 flagged, none from this module. `hot`: sfx `thunder` (peak 24.7), `swoosh` (15.8), `rain` (8.9),
instrument `muted_guitar` (1.54). `ends-on-step`: the loop-style sfx `ball_roll`, `car_engine`,
`electric_motor`, `engine`, `fan`, `gears`, `helicopter`, `jet_engine`, `powerdown`, `propeller`,
`rain`, `riser`, `stream`. `dc`: instruments `clarinet`, `flute`, `french_horn`, `trombone`,
`trumpet`, `tuba` and sfx `glitch`. No errors, no NaN, nothing silent.

This is the idea of `procedural/catalogue.py`, `coverage.py` and `qa_catalogue.py` (render every
id, log a record per id, measure loudness / peaks / NaN / near-duplicate takes) applied to genny's
registries instead of one game's manifest.

## Inventory of the `procedural` project

Status: **same** = `genny/physical/<file>` is that module (see the diff at the end);
**klang** = ported in `genny/klang.py` under the name given; other module names = where genny
has it; **no** = not ported, with the reason.

### C++ (`klang.h` framework)

| file | class | what it does | status |
|---|---|---|---|
| `vehicles/Motors.h` | `ToyBoatEngine` | Farnell toy boat: 9 Hz valve gating formant noise into 3 body band-passes | klang `toy_boat` |
| | `FourStrokeEngine`, `Car` | four cylinders as 1/(1+x²) pulses on a phasor, delayed noise jitter, tanh | klang `klang_car` kind `basic` |
| | `Mini`, `Mini::Engine`, `partials[15]` | Mini resynthesised: 15 partials, soft clip, noise EQ, comb, resonant LPF, starter/rev envelopes | klang `klang_car` kind `mini` |
| | `clip_0_1`, `softclip` | helpers | inlined |
| `vehicles/Harrier.h` | `Harrier`, `Turbine`, `Burn`, `Additive<N>` | jet: 5-partial turbine, overdriven noise burn, wind, echo | klang `harrier` |
| | `pd::lop`, `pd::bpf`, `pd::vcf`, `pd::fastcos` | Pure Data style filters | inlined in `_harrier_k` |
| | `pd::noise` | Pd's LCG noise; declared, never used | no (unused) |
| `vehicles/Bicycle.h` | `Bicycle`, `Pedal`, `Chain`, `Wheel`, `partials[8]` | freewheel tone⁴ × noise, tick pulse, chain noise² × pressure sine | klang `bicycle` |
| `nature/Nature.h` | – | empty (`#pragma once`) | nothing to port; klang `klang_rain` is original |
| `render/render.cpp` | `main`, `write_wav`, `FMath`, `fastsin` | offline renderer: sets up to 3 controls, 512-sample blocks, 16-bit WAV, hard clip | no; its rate (44100) and clip rule are documented above |
| `klang/klang.h` | `Biquad::LPF/HPF/BPF`, `OnePole::LPF/HPF`, `DCF`, `Delay<N>`, `Fast::Sine/Phasor/Pulse/Noise`, `Envelope`, `Control::smooth`, `min`/`max`, `random` | the framework the headers use | the subset used is in `_bq`, `_bt`, `_tap`, `_pulse_avg`, `_u`, `_mini_env` |

### Python: shared kernels

| file | functions | what | status |
|---|---|---|---|
| `core.py` | `rate_*`, `seconds`, `pow_decay/attack`, `ar_env`, `exp_decay`, `contact_pulse`, `felt_pulse`, `biquad`, `reson`, `tv_bandpass`, `tv_lowpass`, `one_pole_lp`, `dc_block`, `lowpass/highpass/bandpass`, `soft_clip`, `place`, `fit`, `master`, `write_wav`, `spectrum_peaks`, `decay_rate` | 48 kHz DSP core | same |
| `modal.py`, `modal_data.py` | `Modes`, `resolve`, `render_modes`, `strike`, `faust_strike`, `micro_collisions`, `stk_modal_bar`, `additive_bell`, `farnell_bell`, `fracture`, `aeolian_tone`, `sword_swing`; 48 mode tables | modal synthesis and measured objects | same |
| `particles.py` | `shaker` (23 PhISEM presets), `granular_footstep_texture`, `bubble_field`, `surfacing_bubble`, `drip` | particles | same |
| `space.py` | `zita_rev1`, `puckette_rev`, `farnell_room`, `sabine_t60`, `distance`, `air_db_per_m` | reverbs, distance | same |
| `waveguides.py` | `pluck`, `mandolin`, `bowed_string`, `hurdy_gurdy`, `banded`, `brass`, `clarinet`, `recorder`, `felt_piano`, `measure_f0` | waveguide instruments | same |
| `voice.py` | glottal sources, formant tables, `pink_trombone`, `breath`, `roar`, `almost_speech`, `fof_voice`, `choir`, `syrinx`, `hybrid_voice` | voices | same |
| `ambience.py` | `wind_scene`, `creak`, `stickslip_pulses`, `fire`, `stream`, `rain`, `thunder`, `surf`, `cave_drips`, `tinkle`, `clunk`, `chain_shift`, `keys_jingle`, `outdoor_echoes`, `bed_*`, `d_*`, `render_bed`, `render_detail` | ambience beds and details | same |
| `footsteps.py` | `grf_poly`, `foot_grf`, `render_step`, `charter_step` | footsteps on 24 grounds | same |
| `analysis.py` | LPC `analyse`/`resynth`, `morph`, `psola`, `modal_fit`, `recorded_event`, `gust_stats`, `wave_stats` | resynthesis of recordings (needs a recordings bank) | same, except one default (see diff) |

### Python: families

| file | functions | what | status |
|---|---|---|---|
| `interface.py` | `ui_focus`, `ui_back`, `ui_activate`, `ui_refuse`, `ui_toggle`, `ui_screen_change`, `ui_save`, `ui_load`, `ui_dialog_open`, `ui_dialog_close`, `ui_hint` | physical UI cues | klang `ui_tick` (kinds focus, back, activate, refuse, toggle, page, save, load, dialog_open, dialog_close, hint) |
| | `channel_spell`, `channel_discovery`, `channel_achievement`, `urgency_alert` | glass rim, Risset bell, phone bells, English bell | klang `ui_chime` |
| | `_mclick`, `_switchclick`, `_shortping`, `_clock_body`, `_panel_body` | Farnell click and body models | foley (`_mclick`, `_switchclick`, `_shortping`, `_body`, `_panel`), reused by klang |
| | `c_*` (17 charter cues), `CHARTER_RECIPES` | one game's tuned bell motifs on D5 A5 E6 | no: game-specific pitches |
| | `render`, `report`, `selftest` | manifest dispatch and self-test | no: manifest-bound; klang `coverage` |
| `body.py` | `sniff`, `strain`, `weight_shift` | nasal pulls, pressed effort, feet shifting | klang `sniff`, `strain`, `weight_shift` |
| | `breath`, `grunt` | breath and grunt | creatures `breath`, `grunt` (synthesis branch) |
| | `sacking_tear`, `_stick_slip` | cloth tear | foley `cloth` kind `tear` |
| | `_second_wind`, `_rally`, `_brace`, `_bandage`, `_root`, `_meditation`, `_enduring`, `_fatigue`, `_measured`, `_battle_madness`, `_condition_black` | 11 timelines of breath + grunt + strain | no: each is three or four `sfx` layers placed with `at` |
| | hybrid branches (`voice.hybrid_voice`) | LPC tract of recorded breaths | no: needs the recordings bank |
| `impacts.py` | `impact_physical`, `impact_soul`, `impact_mental` | damage cues | klang `damage` |
| | `_shockwave`, `_barrel`, `_noiseburst`, `_nwave`, `_bpar8` | Farnell explosion layers | foley (`gunshot`); klang has its own small copies for `damage` |
| | `_onebeat`, `_murmur`, `_fireball`, `_black_noise`, `_boomer`, `_combsweep`, `_doom_pipe` | heartbeat, rumble, swept comb | klang (inside `damage`); heartbeat also creatures `heartbeat` |
| | `sig_handbell`, `sig_blade`, `sig_dead_blade`, `sig_cracked`, `sig_iron`, `sig_glass`, `sig_stone`, `sig_wood`, `sig_small_metal`, `sig_cloth` | one tuned recipe per material | foley `hit`, `clang`, `clink`, `thud`, `casing`, `cloth`; instruments `handbell`, `church_bell` |
| | `relic_kind`, `RELIC_KINDS`, `_varied_signature`, `_charter_impact` | keyword → material for one game's relics, take variation | no: game-specific |
| `weapons.py` | `swing_preset`, `swing` | Selfridge swing presets | foley `sword` |
| | `hit`, `_flesh_body`, `_stone_slab`, `_chitin` | weapon on flesh / shell / stone | foley `stab`, `slash`, `shield_bash` |
| | `charter_hit`, `_describe`, `render` | recorded-layer hit, self-test | no: needs recordings / manifest |
| `world.py` | `item_pickup`, `item_equip` | pick up and equip by kind | klang `handle` |
| | `tide_rising`, `tide_falling` | tide turning | klang `tide` |
| | `week_turn` | distant bell | klang `distant_bell` |
| | `knock`, `drive`, `fractal_noise`, `scrape` | contact, driven body, Cook scrape | foley (`_knock`, `_drive`, `fractal_noise`, `_scrape`), contact `scrape` |
| | `creak_sources`, `rub`, `cloth`, `felt_thud`, `click_cluster`, `casing_bounces`, `bowed`, `struck_glass`, `sparks` | thin wrappers | foley `creak`, `shake`, `cloth`, `thud`, `reload`, `casing`; instruments `bowed_bowl`, `glass_harmonica` |
| | `storm_start`, `camp_fire`, `camp_cold` | rain arriving, fire catching, wind over a camp | matter `fire`, `gust`; sfx `rain`: compose them as layers |
| | `notice_rise`, `notice_fall`, `hunting_party` | crowd murmur joining / fading, walkers with mail | creatures `crowd_murmur`, `footsteps`; foley `chain` |
| | `search`, `rest`, `hide`, `detected`, `eat`, `bandage` | short scenes | creatures `eat`, `breath`; klang `weight_shift`; foley `cloth`, `shake` |
| | `craft_generic`, `craft_rope`, `craft_food`, `craft_lure`, `craft_blade`, `craft_container`, `craft_fire_payload`, `craft_flask`, `craft_saddlebags` | crafting scenes | no: each is 3–5 existing sfx on a timeline (`scrape`, `creak`, `shake`, `smash`, `pour`, `fizz`, `knock`, `thud`) |
| | `memory_*`, `shadow_*`, `fragments_gain`, `keys_fuse`, `mask_on`, `mask_off`, `_charter_world` | one game's magic and story cues | no: game-specific |
| `techniques.py` | `swing`, `hit`, `clash`, `feet`, `steps`, `breath`, `cloth`, `friction`, `air`, `fire`, `glass_rub`, `bowl`, `bell`, `glass_break`, `stone_break`, `heartbeat`, `black`, `pressure`, `blast`, `pluck`, `aeolian`, `creak`, `chain`, `knock`, `rub_body`, `scrape_body`, `rocks`, `shaker`, `bubbles`, `drip`, `wings`, `roar`, `sing`, `snap` | 34 thin wrappers with tuned defaults over the shared kernels | foley (`sword`, `clash`, `hit`, `cloth`, `creak`, `chain`, `knock`, `shake`), contact (`smash`, `scrape`), matter (`fire`, `bubbles`, `drip`, `spark`, `gust`), creatures (`heartbeat`, `wings`, `roar`), choir (`voice`), klang (`damage` for `black` / `pressure`) |
| | `c_cut`, `c_feint`, `c_unarmed`, `c_throw`, `c_archery`, `c_flame`, `c_lightning`, `c_explosion`, `c_shadow`, `c_shadow_solid`, `c_glass`, `c_threads`, `c_forge`, `c_bell`, `c_soul`, `c_mind`, `c_gravity`, `c_transform`, `c_elements`, `c_voice`, `c_blood`, `c_tactics`, `c_lure` | 23 combat / magic "technique" scenes for one game (not instrument playing techniques) | foley `bow_shot` (= `c_archery`), `slash`, `stab`; matter `lightning`; sfx `explosion`; the rest no: game-specific scenes |
| | `_charter_technique`, `_band_profile`, `render`, `_selftest` | charter overlay, dispatch | no |
| `instrument_map.py` | `Src`, `Artic`, `Spec`, `SPECS` (113 keys), `ARTICULATIONS` | which CC0 recordings (VSCO 2 CE, VCSL, Philharmonia, Iowa) play each instrument and articulation | data only, needs the libraries. Used as the list of voices: klang `viola`, `bassoon`, `cor_anglais`, `bass_clarinet`, `piccolo`, `recorder`, `hand_chime`; `ARTICULATIONS` → klang `articulate` layer |
| `sampler.py` | `_name_midi`, `_vel_rank`, `_repitch`, `_release_curve`, `_vel_gain_db`, round robin in `_pick` | file-name zones, repitch, release, velocity curve | klang `sampler` layer (`name_midi`, `vel_rank`, `sampler_zones`) |
| | `play`, `phrase`, `instruments`, `_build_artic`, `_octave_for`, `_yin`, `_analyse_file`, index cache | per-library octave detection, fine tuning, level matching, timpani pitch | no: tied to the downloaded libraries and their folder map |
| | `_extend`, `_bow_strokes`, `_steady_region`, `_best_align`, `_xfade_gains`, `_level_match` | holding a note longer than its recording | no (the layer ends with the recording); port `_extend` if sustained libraries need holding |
| `rooms.py` | `reverb`, `rooms`, `room_info`, `rt60`, `_ir`, `_decode_bformat` | convolution reverb with 14 measured OpenAIR rooms | dsp `convolve` (user IR); the IR files (CC BY 4.0) are not shipped, so the room list is not ported |
| `charter.py` | `hand_chime` | tuned tube chime | klang `hand_chime` |
| | `glock`, `tubular_bell`, `wine_glass`, `felt_tick`, `wood_knock` | small bell set and ticks | instruments `glockenspiel`, `tubular_bell`, `glass_harmonica`; klang `ui_tick` |
| | `hz`, `collides`, `avoid_reserved`, `clear_centre`, `glide`, `glass_tone`, `bell_note`, `sequence`, `rune_call`, `silver_bell`, `mirror_tick`, `mirror_hum`, `rune_examined`, `rune_circle`, `thread_tone`, `weave_threads`, `nexus_lock`, `memory_*`, `fragment_absorbed`, `soul_sea_drone`, `shadow_air`, `echo_voice` | one game's sound grammar: reserved pitches and motifs | no: game-specific |
| `moments.py` | `render`, `_crisis_start/won/lost`, `_moment_awakening/death/true_name/act_change`, `_Score`, `_play`, `_motif`, `_membrane_drum`, `_heartbeat` | stinger and narrative cue composer for one game | compose (`stinger`, `arrangement` layers) |
| `music.py` | `inst_*` (26), `TempoMap`, `euclid`, `Ctx`, `gen_*` (10), `compose`, `render_voice`, `render_layer`, `set_gain`, `seam_report`, `render_act` | generative composition engine and note renderers | compose (`compose`, `arrangement`, `euclid`, `scatter`, `arp`, `adaptive`); instruments in foley (`hurdy_gurdy`, `bowed_bowl`), friction, tubes |
| `score.py` | `Key`, `parse_chord`, `parse_prog`, `parse_theme`, `op_var`, `voice_lead`, `counter_line`, `Score`, `perform`, `write_midi`, `render_stems`, `gentle_limit`, `file_metrics`, `build_*`, `export_*` | notated composition engine on the sampler, one game's themes | compose (`voice_lead`, harmony, arrangement); the rest no: tied to the sampler and one game |
| `choir.py` | `Singer`, `roster`, `play`, `phrase`, `syllabify`, `parse_syllable`, `lead_time` | choir of individual singers, sung syllables | choir (`voice`, `vocal_choir`, `sing`, `satb`) |
| `creatures.py` | `Profile`, `growl`, `frog_clicks`, `chitin`, `almost_speech`, `impact`, `footfall`, `gait`, `wings`, `tear`, `scrape`, `debris`, `collapse`, `presence`, `hit`, `death`, `intent` ... | creature voices and bodies | creatures, contact, foley |
| `catalogue.py` | `render_id`, `planned_files`, `rng_for`, `_master`, `_write`, `main` | parallel resumable rendering of a game manifest, WAV/OGG | klang `coverage` (the "never stop on one id, log a record" idea); the rest no: genny `render` batches specs |
| `coverage.py` | `audio_rows`, `claims`, `owner`, `main` | which family claims each manifest id | klang `coverage` (per registry instead of per manifest) |

### Driver scripts and notes

| file | what | status |
|---|---|---|
| `render_samples.py` | the first renderer (naive modal, Karplus-Strong, FM); judged unrealistic and replaced by `procedural/` | no |
| `render_variants.py`, `run_*.sh` | manifest variants and memory-safe batch drivers | no: tied to one machine and manifest |
| `render_hybrid_pilot.py`, `render_charter_pilot.py` | A/B pilots: recordings on/off, charter on/off | no: need the recordings |
| `render_sampler_check.py` | per-note checks of the sampler: NaN, clipping, silent edges, click detector | idea in klang `coverage` (non-finite, hot, ends-on-step) |
| `qa_catalogue.py`, `qa_beds.py` | loudness vs class target, clipping, loop seams, near-duplicate takes | klang `coverage` (`--deep`: seed-ignored = duplicate takes); loop-seam metrics not ported (genny loops are made by `"loop"`) |
| `listen.py` | spectrogram and loudness plots (matplotlib) | no: new dependency |
| `AUDIO_CHARTER.md`, `proposed_manifest_ids.md` | one game's sound grammar and ids | no: game-specific |
| `PLAN.md`, `WAVE2.md` | the rules the Python modules were written under (port, cite, 48 kHz, numba, UNSOURCED) | same rules as genny's contract |
| `SFX_HYBRID.md` | recordings resynthesised by LPC; later switched off by the author | no: no outside material |
| `MUSIC_V3.md` | sampler + convolution rooms + notated scores | klang `sampler` layer (generic folder); the rest no |

### `genny/physical` against `procedural` (diff)

The ten shared modules are identical to `procedural/` apart from a three-line header and a
relative import, with one difference:

`procedural/analysis.py` lines 63–65
```python
# Off since 2026-10-08 (author's ruling: no outside material, the recordings folder is deleted);
# every family renders its pure-synthesis version.
HYBRID = os.environ.get("SYNTHGEN_HYBRID", "0") != "0"
```
`genny/physical/analysis.py` line 66 still has the old default:
```python
HYBRID = os.environ.get("SYNTHGEN_HYBRID", "1") != "0"
```
genny should take the `"0"` default with its comment. No function exists in one copy and not
the other.
