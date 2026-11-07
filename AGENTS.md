# genny — agent guide

`genny` is a CLI that synthesizes game sounds (instruments, drums, sound effects, effects chains) into `.wav` files. It is deterministic, has no external audio dependencies, and is meant to be driven by an AI agent that writes JSON specs.

**Read this file top to bottom once, then use `genny list --json` for the exact, current catalog of names and parameters. The catalog is the source of truth; this file explains how to combine things.**

## Setup / invocation

```
python install.py --tool                     # installs the `genny` command (uv) and the Claude Code skill
uv tool install --editable <path-to-genny>   # just the command
# or, from the repo:
uv run genny ...
```

Every command writes a WAV and prints `wrote <path> (<seconds>, mono|stereo, <sr> Hz)`. Non-zero exit on error with `error: ...` on stderr. Unknown instrument/effect/param names are rejected with the allowed list, so failures are cheap to fix.

## Two ways to talk to genny

1. **Quick one-shots** on the command line (`synth`, `seq`, `drum`, `sfx`, `say`, `fx`).
2. **JSON specs** via `genny render` — the main interface for anything layered or for batches. Prefer this when making more than one sound.

### One-shots

```
genny synth <inst> "<notes>" [--dur S] [--vel 0..1] [--strum S] [-p k=v ...] [--fx NAME:k=v,k=v ...] -o out.wav
genny seq   <inst> "<steps>" [--step S] [--legato F] [--gap S] [--transpose N] [--bpm N] [-p ...] [--fx ...] -o out.wav
genny drum  <kind> [-p k=v ...] [--fx ...] -o out.wav
genny sfx   <kind> [-p k=v ...] [--fx ...] -o out.wav
genny say   "<text>" [--phonemes] [--show] [-p k=v ...] [--fx ...] -o out.wav   # speech (see "Speech")
genny fx    in.wav --fx NAME[:k=v,...] [--fx ...] -o out.wav      # process an existing file
genny render spec.json [more.json ...] [--out-dir DIR] [--quiet]   # or:  genny render --json '{...}'   or:  ... | genny render -
genny list [instruments|drums|sfx|fx|speech|phonemes|words|waves|scales|chords] [--json]   # instruments: family + register
genny info file.wav ...      # duration, peak/rms, loudness, sharpness, energy above 5 kHz (--json, with `overs`)
genny play file.wav ...      # audition through the system player
genny eval ref.wav a.wav b.wav -o test   # webMUSHRA listening test (config + audio) rating a, b against ref
genny example                # prints a full example spec
```

Examples:

```
genny sfx beep -p freq=880 -p dur=0.1 -o ui/beep.wav
genny sfx proximity -p rate=8 -p dur=1 -o ui/wall_close.wav
genny synth bell "C6" --dur 0.3 -p ratio=5.04 --fx reverb:mix=0.3 -o notif/ding.wav
genny seq brass "C4:0.15 E4:0.15 G4:0.15 C5:0.6" --fx reverb:mix=0.25,size=0.7 -o win/fanfare.wav
genny seq violin "D5:.5 E5:.5 F#5:1 A5:2" --bpm 100 --fx reverb:mix=0.2 -o tune.wav
genny drum kick -p tune=48 -p decay=0.6 -o drums/kick.wav
genny fx voice.wav --fx telephone --fx reverb:mix=0.1 -o voice_radio.wav
```

## Pitch notation

Two kinds of pitch input:

- **Frequency parameters** (`freq`, `freq2` on sfx such as `tone`/`beep`/`alarm`, speech `pitch`): **a number is always Hz** — `"freq": 55` is a 55 Hz hum. `"110hz"`, `"110"` and a note name (`"A2"`) also work.
- **Notes** (`notes` in synth layers, `steps` in seq layers, `genny synth`/`seq` arguments): `C4`, `F#3`, `Bb2`, a MIDI number `60`, or Hz `440hz` / `440.0`. Here a bare integer 0–127 is a MIDI note (musical context); write Hz with the suffix.

Multiple notes: `"C4 E4 G4"` or `"C4,E4,G4"`. Named chord: `"C4:maj"` (chords: `genny list chords` → maj min dim aug sus2 sus4 maj7 min7 dom7 add9 power maj9 min9 oct).

## Step notation (melodies / stingers)

Whitespace-separated tokens for `seq`:

| token | meaning |
|---|---|
| `C5` | note with default step duration (`--step`, default 0.25 s) |
| `C5:0.12` | note lasting 0.12 s |
| `[C5,E5,G5]:0.5` | chord for 0.5 s |
| `C5:maj:0.5` | named chord for 0.5 s |
| `-:0.1` | rest for 0.1 s |
| `C5:0.2@0.6` | velocity 0.6 (softer, usually darker) |
| `\|` | bar line, ignored (readability) |

Instruments add their own release/ring after each note, so notes overlap naturally. `legato` < 1 shortens the sounding part of each step (staccato); `gap` adds silence between steps. With a tempo (`"bpm"`, see "Music") the durations are beats instead of seconds.

## JSON spec format

```jsonc
{
  "out": "sounds/win/fanfare_01.wav",   // output path (relative to --out-dir if given)
  "sr": 44100,                          // optional
  "duration": 1.5,                      // optional hard length (pads or cuts)
  "layers": [ ...layer objects... ],    // mixed together on one timeline
  "fx": [ {"type": "reverb", "mix": 0.2}, {"type": "compressor"} ],   // master chain, in order
  "bpm": 120,                           // optional tempo: musical times in the layers are then beats (see "Music")
  "normalize": -1.0,                    // peak dB after fx; null to skip
  "max_loudness": -6.0,                 // ear-weighted ceiling applied after normalize; null to disable
  "loudness": "streaming",              // optional: master to integrated LUFS instead (replaces normalize + max_loudness):
                                        //   "streaming" -14, "broadcast" -23 (EBU R128), "console" -23, "mobile" -18, "apple_tv" -24, "itunes" -16.2, "kids" -28, or a number
  "true_peak": -1.0,                    // dBTP ceiling used with "loudness"; absent = -1, or -2 for a target above -14 LUFS
  "headroom_only": false,               // with "loudness": one plain gain, min(target - loudness, ceiling - true peak), never a knee
  "dither": null,                       // export dither: "tpdf" (or gauss, shaped, fweighted, lipshitz) at the output word length; null = plain truncation
  "dither_seed": 0,                     // fixed dither seed: the same spec gives the same file
  "gain": 1.0,
  "trim": true,                         // drop silent tail
  "declick": true,
  "loop": false,                        // true or crossfade seconds: seamless loop; "fold": musical loop (see "Music")
  "variants": 6, "vary": {"pitch": [0.95, 1.05], "gain": [0.8, 1.0]}   // N files out_01.wav ... for random playback; every "seed" in the spec + 0, 1, 2 ... (pitch = playback rate, length follows; gain linear)
}
```

### Layer objects

Common keys for every layer: `"at"` (start seconds, default 0), `"gain"` (linear, default 1), `"fx"` (effects chain applied to just this layer), `"repeat"` + `"every"` (repeat the layer N times every X seconds), `"priority"` (number: while a higher-priority layer sounds, this one drops 3 dB per step, 12 at most), `"duck_band"` ([lo, hi] Hz or true = 500-4000: the priority duck only cuts that band, a notch instead of a level drop).

```jsonc
{"type": "synth", "inst": "bell", "notes": "C5 E5 G5", "dur": 0.4, "vel": 0.8, "strum": 0.03, "params": {"ratio": 3.5}}
{"type": "seq",   "inst": "pluck", "steps": "C4:0.1 E4:0.1 G4:0.3", "step": 0.25, "legato": 1.0, "gap": 0, "transpose": 0, "vel": 1, "params": {}}
{"type": "seq",   "inst": "piano", "steps": [{"note": "C4", "dur": 0.2}, {"notes": ["E4", "G4"], "dur": 0.4, "vel": 0.7}, {"note": "-", "dur": 0.1}]}
{"type": "drum",  "kind": "kick", "vel": 1.0, "params": {"tune": 50}}
{"type": "pattern", "kind": "hihat", "steps": "x-x-xXx-", "step": 0.125}      // x hit, X accent, o ghost; or "hits": [0, 0.5, 0.75]
{"type": "sfx",   "kind": "whoosh", "params": {"dur": 0.4, "direction": "down"}}
{"type": "speech", "text": "fuel low", "params": {"voice": "female"}}          // spoken words (see "Speech")
{"type": "file",  "path": "existing.wav"}                                      // import a sample
{"type": "silence", "dur": 0.5}
{"type": "group", "layers": [ ... ], "fx": [ ... ]}                              // sub-mix with its own fx
```

`params` must only contain keys the instrument/drum/sfx/speech declares (see `genny list --json`). Anything else is an error.

### Batches

```jsonc
{
  "out_dir": "sounds/notifications",     // prefixed to each relative "out"
  "fx": [{"type": "compressor"}],         // defaults inherited by sounds that don't set their own (also sr, normalize, max_loudness, bpm)
  "sounds": [ { ...spec... }, { ...spec... } ]
}
```
A top-level JSON array of specs also works. `genny render batch.json` renders all, reports `N rendered, M failed`, and continues past errors unless `--strict`.

## Catalog overview (run `genny list` for parameters)

**Instruments** (`synth`/`seq` layers). Each note = `inst(freq, dur)`, returns note plus release. `genny list instruments` shows every one with its family, the register it is voiced for (`range`) and its params.
- Keys: `piano` (`pedal`, `soft` una corda, `lid` open|half|closed), `felt_piano` (soft, lo-fi), `epiano` (Rhodes, FM), `rhodes` (physical tine + pickup: `voicing`, bark when hit hard; `sustain`, `onset` hammer|swell, `tremolo`, `tonebar` fork|2dof, `tonebar_cents`, `tuning_mass`), `wurli` (capacitive reed pickup: pure soft, bark hard), `clav` (`tone` S M T B rockers, `beat` dB), `harpsichord` (Valimaki physical model: partials decaying unevenly, a beating partial, ringing soundboard; `octave` 4-foot rank, `soundboard`, `pluck` point as a fraction of the string, 0 = by register 12-50 %; `phantom`; `ripple_partial`), `organ` (drawbars; `upper`), `church_organ` (`chiff` classic|grey), `accordion` (`musette`), `celesta`, `toy_piano`
- Mallets and bells: `marimba`, `xylophone`, `vibraphone`, `glockenspiel`, `kalimba`, `music_box`, `steel_drum`, `handpan`, `bell` (`bright`, `ratio`), `glass`, `tubular_bell`, `gamelan`, `singing_bowl`, `timpani`
- Plucked: `pluck`, `harp` (Cb1-C7, stretched low register), `guitar` (nylon), `steel_guitar`, `electric_guitar` (clean; `pickup` single|humbucker|rail = measured magnetic nonlinearity), `muted_guitar`, `dist_guitar`, `banjo`, `mandolin` (`tremolo`), `koto`, `sitar` (`model` fd = string striking its curved bridge, buzz grows with velocity), `pizzicato`. All of these and the two plucked basses take `glide` (cents a hard pluck starts sharp and settles: tension modulation, 10-40 audible); `guitar` and `steel_guitar` take `body` (a number = a different instrument, with the spread real guitars have) and `body_size` (0 small box .. 1 big box), `radiation`, `air_q`; `guitar` `glide_mm` (tension glide from the pluck amplitude); `dist_guitar` `loop` 1 (clip inside the string loop); `guitar` `pluck` nail|finger|pick, `rattle`, `cloth`; `steel_guitar` `worn`; `electric_guitar` `pickup_pos2`, `mix` in|out, `stiff`, `amp` bassman (`treble`, `middle`, `bass`), `horizontal`; `clav` `pickup` A|B|C|D, `magnetic`. Also (`docs/acoustic.md`): `kantele`, `string_scrape`, `guitar_chord` (`chord`, `strum`, `coupling`), `tanpura` (`jawari`, `cycle`, `first`), `zanza` (`buzz`, `cone`), `baschet` (`bend`, `sheet`, `screw`), `resonator_guitar`, `whirl_string` (`ellipse`, `slack`)
- Bass: `bass` (analog synth), `sub`, `wobble`, `upright_bass`, `finger_bass`, `slap_bass`, `acid` (303), `reese`, `fm_bass` (`ratio`, `model` classic|mex, `index`, `fb` feedback), `bass808`
- Bowed: `strings` (ensemble), `violin`, `cello` (`model` elastoplastic = measured string and bow friction, a real Helmholtz attack; `body_am` 1 = vibrato as body AM; `violin` `model` commuted, `direction`), `tromba_marina` (`gap`), `bowed_psaltery`
- Winds: `flute` (`meyer`, `expert`, `novice`), `pan_flute`, `clarinet` (`model` impedance = reed and bore physics, thicker and brighter with velocity; `embouchure`; `meyer`, `precursor`), `oboe` (`even`, `precursor`), `sax` (`lattice` soprano|tenor), `harmonica`, `whistle`, `ocarina`, `didgeridoo`
- Brass: `brass` (synth section), `trumpet` (`mute`), `french_horn`, `trombone`, `tuba` (all four take `model`: `formant` default, `physical` = lip-reed waveguide, in tune, speaks with a real attack; `trumpet model=lip2d`, `trombone model=lip`; `meyer` = measured attack, spectral ceiling, dynamic timbre, band vibrato)
- Synth leads and pads: `lead` (supersaw), `soft_lead`, `square_lead`, `synth_pluck`, `theremin`, `pad`, `warm_pad`, `packet_pad` (Chebyshev packet pad: `cutoff`, `res`, `attack`, `release`), `glass_pad`, `string_machine`, `choir`, `pwm`; `soft_lead`, `square_lead`, `warm_pad`, `reese` take `drift` (cents of analog pitch wander)
- Retro: `chip` (pulse; `width` 0.125/0.25/0.5, `bright`), `chiptri`, `board` (80s pinball/arcade sound-board voice: gritty pulse with an onset pitch blip)
- Raw: `sine`, `square`, `saw`, `synth` (generic and untamed: `wave`, ADSR, `cutoff`, `res`, `fenv`, fm `ratio`/`index`; `wave` also jp8000, cz, gbuzz, dpw_saw, dpw_triangle, dpw_square, blit, blit_saw, blit_square, sync, pitched_noise, held_noise: `docs/osc.md`)

**Drums**: `kick` (`model` farnell = techno click), `kick808` (`model` classic|circuit, `knob`, `tone`), `tom808`, `conga808`, `bass_drum` (orchestral), `snare` (`under`, `flip`: bottom mic), `snare808`, `clap` (`model` classic|rj), `snap`, `brush` (jazz; `swish`), `hihat`, `openhat`, `tom` (`under`, `flip`), `rim`, `cowbell`, `crash`, `ride`, `shaker`, `tambourine`, `woodblock`, `taiko`, `zap_kick`; hand and world percussion: `conga` (`slap`), `bongo`, `djembe` (`stroke` bass|tone|slap), `tabla` (`stroke` na|ge|ke), `cajon` (`stroke` bass|snare), `timbale`, `clave`, `guiro`, `triangle`, `gong` (`glide`, `curvature`, `preset` agung: bossed gong, beating doublets), `water_tube`, `sleigh`. (`timpani` is an instrument: it takes notes.)

**SFX**: `beep`, `blip`, `click`, `pop`, `coin`, `powerup`, `powerdown`, `laser`, `zap`, `hit`, `punch`, `explosion`, `jump`, `whoosh`, `swoosh`, `alarm`, `siren`, `error`, `success`, `proximity`, `radar`, `riser` (`pulse`), `sweep_up`, `sweep_down`, `bubble`, `glitch`, `static`, `wind`, `thunder`, `footstep`, `door`, `engine`, `car_engine`, `magic`, `heartbeat`, `tone`, `noise`, `vinyl` (record crackle; `wear`), `scratch` (DJ: `kind` baby|scribble|hydroplane, `pitch`), `typewriter`, `countdown`, `fm_machine` (auditory-icon machine); pinball/mechanical: `solenoid`, `flipper`, `pop_bumper`, `slingshot`, `knocker`, `steel_ball`, `ball_roll`, `spinner`, `spring`, `chirp` (see "Pinball / mechanical foley").

**Speech**: `speech` layer / `genny say` — formant voice for letters, numbers and short game words; voices `male`, `female`, `child`, `whisper`, plus character voices `robot`, `android`, `synth`, `bad_robot`, `evil_robot`, `monster`, `giant`, `alien`, `ghost`; English + Spanish (`lang=es`). See "Speech" below.

**Effects** (`"fx"` chains; also `--fx name:k=v,k=v`):
- Space: `reverb` (mix,size,damp,predelay,width), `delay` (time,feedback,mix,pingpong), `chorus`, `flanger`, `phaser`, `width`, `pan`, `autopan`
- Tone: `lowpass`, `highpass`, `bandpass`, `notch`, `eq`, `lowshelf`, `highshelf`, `sweep` (moving filter), `muffle` (through a wall), `telephone`, `underwater`
- Character: `distortion`, `bitcrush`, `ringmod` (`leak`), `body_morph` (through a guitar body, morphing between two bodies), `vibrato`, `tremolo`, `stutter`, `pitch` (semitones, same length), `speed` (tape speed), `reverse`
- Dynamics/utility: `compressor` (`preset`, `key`, `mix`), `limiter` (`lookahead`, `true_peak`), `gate`, `expander`, `duck`, `dyno`, `gain`, `normalize`, `fade` (in,out), `trim`, `mono`
- More: `ladder` (`kind` moog|diode|lpf18|huovilainen, `solver` node|newton|adaa), `wah`, `octaver`, `radio`, `balance`, `exp_delay`, `shaker_comb`, `kasfilter`, `phase_bash`, `antisat_svf`, `random_allpass`; `reverb freeze`, `tremolo` `skew`/`duty`/`phase`/`spread`

`reverb`, `chorus`, `pan`, `autopan`, `width`, `delay pingpong` and `granular` (when `spread` > 0) produce stereo files; everything else keeps the channel count.

## Extended catalog (physical models)

The names above are the core set. The modules below add the rest: TOTALS_TBD in all. `genny list` (or `genny list --json`, `genny list layers`) prints every name with
its parameters; `docs/catalog.md` is the same list as a page, `docs/physics.md` says which physical model is behind
each sound, and each module has a page with recipes.

- **contact** (`docs/contact.md`) — sfx: `roll` (`asymmetry`, `patch`), `scrape` (`gate`, moving `position`, `variation`, `model` meso), `impact` (`law`: measured decay tables; `material_angle`: continuous glass-metal-wood blend; `azimuth`; `shape` circ_plate_free = free-edged disc; `rough`, `freq_factor`, `decay_factor`), `interaction` (rub / scratch / roll and every blend between, `action` 0 / 120 / 240 degrees; `prefilter`, `radius`, `gamma`), `bounce`, `drop` (both `restitution` 0 = the material's value), `smash`, `crumple`, `hail`, `avalanche`, `rumble`, `modal_engine` (`pick` lowest|rank, `misfire`); fx: `resonate`, `cavity`; materials `wood:<species>` (32 woods), `copper`, `acrylic`; shapes `ri_<object>` (measured objects)
- **friction** (`docs/friction.md`) — instruments: `cristal` (`bend`, `modes`), `rubbed_glass`, `bowed_bar` (`material` aluminium_b|iron|comb), `daxophone` (bowed wooden tongue; `bow` point from the dax), `bowed_sheet` (`glide`), `rod_bank` (`rod` square|piano|bronto|sad|fixed|paired, `strike`, `collector`), `whistling_blades` (`sumtone`), `nail_violin` (bowed clamped steel nails on a wooden box), `marloye_harp` (rubbed rods 1:3:5:7), `tuning_fork`, `coil_spring` (`wire`, `coil`, `turns`), `musical_saw` (`glide`, `vibrato`, `rate`); sfx: `stick_slip`, `squeak` (`model` physical|additive), `brake_squeal` (`kind` rub|squeal_hf|squeal_lf|squeak|groan|judder|wire_brush|squelch, `hot_spots`), `rub`; fx: `sheet_radiator` (`metal`, `mute`), `cone_radiator`, `sympathetic` (`tuning` random), `force_pitch` (snaps a voice or other voiced sound onto one note)
- **matter** (`docs/matter.md`) — sfx: `fire`, `pour` (`temperature`, `collective`), `drip` (`harmonic`, `crater`, `impact` parabolic|franz), `drops`, `splash` (`entrain`), `babble`, `bubbles` (`flow_lpm`, `forcing`, `amp_law` moss|pinch|xue, `transfer`, `surface`, `vessel`), `boil`, `sizzle`, `fizz` (`pops`), `gurgle`, `drain`, `surf` (`sizes`, `plume`, `puffs`, `active`), `cave_drips` (`drop_ml`), `waterfall`, `underwater_ambience` (`surf`, `tide`, `wind`, `ships`, `biophony`), `steam`, `air_leak`, `spray`, `kettle`, `balloon`, `gust`, `flame` (`length`), `match`, `spark`, `arc`, `mains_hum`, `tesla`, `neon`, `lightning`, `ice_crack`, `ice_cubes`, `freeze`, `rain_water`
- **foley** (`docs/foley.md`) — instruments: `church_bell` (`type` measured = a recorded bell), `handbell`, `china_bell`, `glass_harmonica`, `bowed_bowl`, `hurdy_gurdy`, `wind_chime`; drums: `cabasa`, `sekere`, `maraca`, `bamboo_chimes`, `sandpaper`, `sleigh_bells`, `agogo`, `anvil`, `brake_drum`, `wind_chimes`; sfx: `hit`, `clang`, `clink`, `thud`, `knock`, `shake`, `chain`, `keys`, `cloth`, `creak` (`width`, `height`, `wave_speed`: door-leaf modes), `zipper`, `velcro`, `paper`, `coin_spin`, `dice`, `cork_pop`, `boing`, `ratchet`, `drawer`, `book_drop`, `gunshot` (`angle`, `ear`, kind `silenced_rifle`), `scifi_fm`, `noizer`, `overblow`, `reload`, `casing` (`ricochet` = the film case), `sword` (`groove`), `cavity_tone` (wind over an opening), `aeolian_harp` (strings singing in the wind), `clash`, `stab`, `slash` (both `weapon` blade|heavy|spear|improvised, `target` flesh|armour|wood|stone|shell), `body_hit` (`material` flesh|stone|shell|metal|bone, `size`), `bow_shot`, `whip`, `shield_bash`, `clock`, `switch`, `button`, `keyboard_typing`, `mouse_click`, `camera_shutter`, `servo`, `printer`, `drill`, `saw`, `hammering`, `ratchet_wrench`, `piston`, `steam_engine`, `train`, `sonar`, `dtmf`, `phone_tone`, `phone_bell`, `siren_wail`, `car_horn`, `bike_bell`, `buzzer`, `shepard`, `transporter`, `robot_babble`, `red_alert`, `dark_drone`, `force_field`, `teleport`, `energy_blade`, `warp`, `charge_up`, `scanner`, `explosion_layers`, `pulse_dial`, `shield_up`, `relay_chatter` (relay switched on and off at a rate the part allows)
- **arms** (`docs/arms.md`) — sfx: `firearm` (33 weapons: distance, azimuth, miss, suppressor, subsonic, pov; `level` relative by default = measured differences between weapons, `peak` = every file at 0.9; `render` heard by default = through an overloaded mic and the `place` (`street`; `none` when your spec adds a room), `pressure` = the bare click at the ear; `echo` 0 by default), `minigun` (`render`, `place`), `gun_mech` (handling, `reload_*`, `reload_musket`, `jam_clear`, `revolver_eject`), `bullet_impact` (`level`), `ricochet`, `bullet_pass` (`render`, `place`), `casing_ricochet`; `casing caliber=` for real cases
- **ordnance** (`docs/arms.md`) — sfx: `blast` (`kind`), `grenade`, `molotov`, `mine`, `launcher`, `missile`, `rocket_salvo`, `rocket_flyby`, `launcher_handling`, `mortar`, `howitzer`, `tank_gun`, `shell_incoming`, `shell_crack_overhead`, `aerial_bomb`, `flamethrower`, `torpedo`, `depth_charge`, `sea_mine`; fx: `earplug`
- **melee** — sfx: `melee` (10 kinds), `knife`, `spear`, `armor`, `grab`, `gouge`, `crossbow`, `arrow`, `sling`, `slingshot_weapon`, `catapult`, `siege_impact`
- **energy_arms** — sfx: `stun_gun`, `taser`, `cattle_prod`, `arc_weapon`, `energy_weapon`, `plasma`, `railgun`, `coilgun`, `gauss_rifle`, `laser_impact`, `photon_torpedo`
- **creatures** (`docs/creatures.md`) — instruments: `whistling`; sfx: `heartbeat`, `roar`, `growl`, `bark` (`dominant`), `meow`, `purr` (`voice`), `moo`, `howl`, `monster`, `animal`, `cricket`, `cicada`, `fly`, `mosquito`, `bee`, `stridulation` (`kind` grasshopper|lobster), `body_fall` (`material`, `debris`), `night_insects`, `wings`, `birdsong` (`source` fm|normal_form|laje|gardner|suboscine|tyrant, `coupling`, `gestures`, `tract` bands|gardner|oec), `bird_call` (`source` valve|labial, `coupling`, `latex`, `membrane` sweep|fletcher|raven, `radiation` lowpass|piston|fletcher), `dawn_chorus`, `frog`, `rattlesnake`, `hiss`, `breath`, `grunt`, `cough`, `sneeze`, `snore`, `laugh`, `scream`, `cry`, `moan` (`voice` female|male, `syllables`), `gulp`, `eat`, `handclap` (`rise` fast|exp), `applause` (`clap_type`, `sync`), `dread_choir`, `crowd_murmur`, `whistle_human`, `footsteps` (`mass`; `stop` = the last N steps slow to a halt; `walker` woman|man, `mood` neutral|happy|sad|angry|tender), `mammal` (mammal call: glottal pulses with a ladder of nonlinear `complexity` 1-6 through a tube `tract`)
- **human** — sfx: `burp`, `fart`, `kiss`, `spit`, `vomit`, `pee`, `yawn`, `gargle`, `hiccup`, `bodyfall`, `crowd` (up to 24 voices), `shout`, `baby`, `blow`, `skin`, `finger_snap`, `cartoon_voice`, `whisper_bed`, `beatbox`, `yodel`, `rap`
- **fauna** — sfx: `bird_kind`, `chicken`, `farm_bird`, `pig`, `sheep`, `horse`, `donkey`, `farm_animal`, `wild_animal`, `primate` (`kind` singe|gorille|diane), `elephant`, `whale`, `rodent`, `reptile`, `slither`, `swarm`, `creature_steps`
- **choir** (`docs/choir.md`) — instruments: `voice`, `vocal_choir`, `hum`, `throat_singing`, `falsetto`, `boys_choir`, `chant`; layers: `sing`, `satb`; options `peak_stretch`, `vib_dyn`, `vib_cycle`, `vib_arch`; style `amateur`
- **fdstring** (`docs/fdstring.md`) — instruments: `fd_string` (`bead`, `bead_pos`), `fd_piano` (`scheme` explicit|theta4|stable|bensa, `hammer_model` semi|conservative, `hammer_width`), `prepared_piano`, `slack_string`, `fd_bar` (`glide`, `scan`, `scheme` explicit|implicit; `tuning` xylophone = 1:3:6), `bowed_string_fd`, `interceptor_string` (string buzzing on a screw, or two screws on arms: `kind` screw|tuba, `gap`, `cloth`), `string_bridge_plate` (`config` body|coupled|plate|rattle|flat: from a soundboard to buzz, rattle and cymbal-like plates); sfx: `twang`; fx: `spring_reverb` (`preset` leem1|leem2|leem3|sansui|leem_auto, `tank`, `efficient`, `clip`, `model` chain|fd, `wander` smooth|uniform)
- **synths and filters** — instrument: `padsynth` (`bw`: wide, chorused pads that loop); drum: `hat808`; fx: `ladder` (`kind` moog|diode), `va_ladder` (`kind` lp|hp|bp|korg35|ellip, `poles` 4|8, `cutoff`, `k`), `dattorro` (plate reverb), `gverb`, `binaural` (`azimuth`, `elevation`: headphones); `tape model=hysteresis`; `acid model=diode`; `sax model=stk`; `timpani model=premier`; `marimba`, `vibraphone` take `resonator`; `shell_crash` (drum); `engine_knock` (sfx); `church_bell type=hibbert|old_english|medieval|big_ben`
- **geo** — sfx: `rockfall`, `landslide`, `debris_flow`, `singing_dune`, `mud_pot`, `geyser`, `lava`, `glacier_calving`, `sand_storm`, `earthquake`, `tree_fall`, `snowfall`, `snow_avalanche` (`precursor` s of snowpack cracking before the release, `density`), `snow_squeal`, `pd_footstep` (`surface`)
- **more synths and sfx** — instruments: `stk_fm` (`voice` rhodey|wurley|tube_bell|b3|heavy_metal|perc_flute; `dac` 1 = TX81Z output filters, `chiptune2.tx81z_dac`), `chowning` (`voice` clarinet|woodwind|bassoon|brass|bell|drum|wood_drum|soprano), `stk_fmvoice` (STK FMVoices: three carriers on the harmonics nearest the vowel's formants, one modulator; `vowel` ah|eh|ee|oh|oo, `tilt`, `vib`), `jaw_harp` (`path` circle|sine|line|figure8|triangle), `soft_sync`, `psg3`; sfx: `bytebeat` (`formula` 0..6), `hummer`, `cuica`, `blast_noise`; fx: `serge_fold`; `paulstretch recipe=subtle|standard|lush|shimmer|shimmer_deep`; `sfxr link=<jsfxr link>`; `zzfx mutate=`; `lightning strokes=`; `siren_wail kind=police_fr|gendarmerie|pompiers|samu|ambulance_fr|police_rc`
- **instruments_extra** — instruments: `ukulele`, `zither`, `shamisen`, `bagpipes`, `shofar`, `traditional_horn`, `mellotron`, `slide_whistle`, `calliope`, `waterphone`, `berimbau`, `talking_drum`, `flexatone`, `risset_arpeggio`, `gendyn`; also `clavichord`, `double_bass` (acoustic) and the drum `bass_drum` (orchestral; `mallet`, `pos`)
- **nature** — sfx: `firework`, `firecracker`, `lava_flow`, `geothermal`, `tornado`, `volcano`, `tsunami`, `singing_sand`, `collapse`, `crack`, `rain_on`, `fire_out` (`detail` 0..1: steam whine, bubbling embers, cooling creaks, a steam shriek), `storm`, `foliage`, `earthquake_stage`, `dirt`, `tree`, `wind_kind`, `rapids`, `fountain`
- **scenes** — sfx: `scene` (`kind`: ambience beds by category), `room_tone`
- **sports** (`docs/sports.md`) — sfx: `ball_bounce`, `pea_whistle`, `snow`, `sport`, `skateboard`, `ice_skate`, `swim`
- **viscous** — sfx: `mud`, `gore`, `slime`, `bone`, `blood`, `chem`, `burn_flesh`
- **household** (`docs/household.md`) — sfx: `lock`, `door_family`, `lever`, `pulley`, `window`, `umbrella`, `office`, `rope`, `scissors`, `garden_tool`, `tool`, `coins`, `cards`, `board_game`, `furniture`, `bag`, `cart`, `leather`, `alarm_bell`, `smoke_alarm`, `alarm_clock`, `appliance_beep`, `ventilation`, `vacuum`, `hair_dryer`, `blender`, `shaver`, `appliance`, `toilet_flush`, `faucet`, `pipe_water`, `frying_pan`, `pepper_grinder`, `vacuum_hose`, `bowed_ceramic`, `filled_glass`, `flat_drop`, `roller_drop`
- **machines** — sfx: `power_tool`, `jackhammer` (`model` impact|rod), `dental_drill`, `elevator`, `escalator`, `hvac` (`state` fan|cool|rattle|box_fan), `conveyor`, `hard_drive`, `gym`, `ride`, `slot_machine`, `pump`, `sewing_machine`, `medical`, `industrial`, `belt_squeal`, `savart_wheel`
- **engines** — sfx: `engine_layout` (19 architectures), `diesel_engine`, `two_stroke`, `forced_induction`, `engine_event` (kind `launch_control` among others), `race_car`, `jalopy`; Python `launch_curve`, `limiter_curve`, `cold_idle(idle, temp_k)` (cold idle and fuel-cut recovery rpm)
- **heavy_vehicles** — sfx: `motorcycle`, `scooter`, `truck`, `bus`, `tractor`, `mower`, `chainsaw`, `garden_machine`, `atv`, `wagon`, `tracks`, `hydraulic_pump`, `excavator`
- **road** — sfx: `ev_drive`, `ev`, `avas`, `hybrid_transition`, `tyre_roll`, `interior`, `air_system`, `car_door`, `air_horn`, `klaxon`, `backup_beeper`, `car_alarm`, `vehicle_mech`, `suspension`, `car_wind` (wind on a moving car: A-pillar tone, window buffeting; `speed`, `sideslip`, `window`, `view`); fx: `air_speaker` (compressed-air horn speaker: `depth`, `horn`, `hiss`)
- **rail** — sfx: `rail`, `traction_drive`, `curve_squeal`
- **marine** — sfx: `ship`, `ship_engine`, `ship_horn`, `ship_bell`, `fog_signal`, `foghorn`, `outboard`, `jet_ski`, `sailboat`, `submarine`, `lap` (`plume` m: bubble-plume thump per wave); Python `plume_f0(radius)` (bubble-plume collective frequency)
- **aero** — sfx: `sonic_boom`, `prop_engine`, `airship`, `glider`, `airframe`, `rotorcraft`, `turbofan`, `jet_cycle`, `flyby_jet`, `cabin`, `jet_afterburner`, `space_rocket`, `drone`
- **designed** — sfx: `cartoon`, `stinger_sfx`, `braam`, `morph`, `magic_kind`, `ui_slide`, `radio`, `rotary`, `geiger`, `modem`, `morse`, `alert_tone` (`rough`, `rough_hz`), `can_roll` (`pattern` 1|2, `bumps`), `toy`, `turntable`, `crt`, `tinnitus`, `party_horn`, `cap_gun`, `mic_noise`, `mic_feedback`, `ice_ring`, `kazoo`, `rubber_duck`, `water_whistle`, `pendulum_feedback`, `rod_cage`, `creature`
- **plates** (`docs/plates.md`) — drums: `fd_drum` (`berger`, `detune`, `under`, `flip`, `snare`, `snare_gap`: snare wires by energy-conserving collision), `vk_cymbal` (`kind` crash|ride|splash|china, `dome`), `vk_gong` (`kind` opera|curved|tam_tam|rectangle), `bodhran`, `polar_shell` (`kind` cymbal|flat|deep|gong, `q`, `zone` bell|bow|edge on a tapered cymbal: `percussion.cymbal_thickness`, `zone_drive`), `perrin_bell` (`tune`), `plate_crash`, `plate_ride`, `plate_china`, `plate_gong`, `tam_tam`, `steel_cymbal`, `steel_gong` (DAFx-23 calibrated steel plates; `scan` = swinging gong), `thunder_sheet`; sfx: `metal_plate`, `glass_pane`, `wood_panel`, `plate_bow`; fx: `plate_reverb`, `room2d`
- **tubes** (`docs/tubes.md`) — instruments: `reed_tube`, `reed_cone` (both `reed` default|played|sax, `contact`, `mouthpiece`), `vowel_tube` (`source` bilbao|rosenberg, `consonant`, `glide` smooth|minjerk, `model` webster|kl, `nasal`), `piston_pipe`, `pvc_pipe`, `bottle`, `mesh_vowel` (`vowel`, `vibrato`), `box_bar` (`detune`, `decay`); sfx: `pipe_blow`, `wah_pipe` (long fan-blown pipe: partials come in with the pressure, open or stopped); fx: `formant_tract`, `tube`, `tube_mesh` (3-D duct: `distance`, `absorption`; stereo); Python `WebsterTube(hole_kind=lumped|unflanged|flanged)`, `hole_length_corrections`, `jet_receptivity`
- **reverbs** (`docs/reverbs.md`) — instruments: `feedback_guitar` (`aperture`, pickup width in m or single|humbucker); fx: `chorus`, `flanger`, `phaser`, `jcrev`, `satrev`, `fdn_reverb` (`matrix` householder|hadamard|weighted, `weights`, `diffuser` serial|vector), `lucier` (`generations`, `t60`, `size`, `chain`: Lucier room re-recording), `velvet_reverb` (`kind` velvet|dark|edvn|sparse, `preset`; pori = measured octave-band decay), `early_reflections`, `room` (`model` fdn|zita|puckette|farnell|moorer|sdn|raytrace, `scattering`, `mic`), `zita`, `distance` (`gradient`, `rolling`, `across_water`, `pipe`), `doppler`, `occlude`, `leslie` (`leak`), `tape_delay`, `shimmer`, `gated_reverb`, `reverse_reverb`, `nrev`, `prcrev`, `costello`, `coupled_room`, `directivity` (french_horn, `wall`, `podium`, `split`), `decorrelate`, `jpverb`, `greyhole`, `room_feedback`, `reflection`, `huggins`, `bell_swing`, `orchestra_seat`, `pv_reverb`
- **spectral** (`docs/spectral.md`) — fx: `timestretch` (`mode` vocoder|sola|ola|hps|psola|auto), `pitch_shift` (`mode` vocoder|psola|auto), `harmonizer`, `vocoder`, `cross_synth`, `spectral_gate`, `sines_only`, `noise_only`, `freeze`, `spectral_blur`, `freqshift` (`hz_end`), `robotize`, `whisperize`, `spectral_tilt`, `hpss`, `autotune`, `gender`, `mutate`, `spectacle` (spectral delay), `granular` (ring-buffer grain cloud: `density`, `size`, `spray`, `position`, `spread`, `pitch`, `feedback`, `freeze`, `window`, `overlap` sola), `fog` (FOF granular: `dens`, `trans`, `spd`, `band`, `ris`, `dec`, `dur`), `tf_pad` (`curve`, `density`, `size`), `granulate` (brassage: `grain`, `window`); layers: `resynth` (`"transients": true` keeps clicks and attacks sharp when stretching)
- **dsp** (`docs/dsp.md`) — sfx: `noise_bed`, `complex_osc` (two cross-modulated sines; `preset` olarte); fx: `convolve`, `fir_eq`, `steep_lowpass`, `steep_highpass`, `steep_bandpass`, `overdrive` (`shape` asym, tube_bendiksen), `wavefold`, `mulaw`, `dither`, `codec`, `lofi`, `smooth`, `tape` (`wow_rate`), `wall` (drywall = double stud wall), `dc_block`, `multiband`, `deesser`, `exciter`, `tape_stop`, `crunch`, `chebyshev`, `diode_clipper`, `lockhart_fold`, `delta_sigma`; Python: `filters.onepole_nl`, `filters.ladder_mix` (+ `LADDER_MIX`), `filters.tbvcf`, `filters.onepole4`, `dsp.fdn_allpass_matrix`, `dsp.fdn_poles`, `dsp.is_unilossless`, `dsp.apdist`, `dsp.wg_osc`, `dsp.wdf_*`, `dsp.variation_filter`, `dsp.head_shadow`, `dsp.decorrelation_fir`, `dsp.dynamic_allpass`, `dsp.median_comb`, `dsp.modal_fft`, `identify.velvet_fit` / `velvet_render`; `osc.minblep_saw`, `osc.minblep_table`, `osc.wavetable_from_spectrum`, `osc.wavetable_osc`, `osc.dpw_order_for` (`docs/osc.md`)
- **retro** (`docs/retro.md`) — instruments: `sfxr_voice`, `zzfx_voice` (`engine` zzfx|zzfxg), `psg`; sfx: `sfxr` (`strict` with bfxr), `zzfx`; layers: `song` (`format` zzfxm (`engine` zzfxg) | mod (`path`, `instruments`), `ticks` + `frame` zx48|zx128|pal); chips (`docs/chiptune.md`): instrument `glitch_saw` (comparator-and-hysteresis saw, two pitches, glitch circuit), `tia` (`wave` saw|engine|square|bass|pitfall|noise|lead|low|buzz or AUDC 0-15, `region` ntsc|pal), layer `chip` `system` `tia` (channels `ch1`, `ch2`), `sid` `fc_max`
- **compose** (`docs/compose.md`) — instruments: `ks_string`; drums: `ks_drum`; layers: `compose` (`humanize` model arma, `gamma` in voss mode, `off`, `echo`, `min_ioi`; perform keys `feel` soloist|bass|ride, `emotion` angry|happy|sad|tender, `phrase_lengthen`, `nma`, `ensemble`, `accent_dip`, KTH intonation `melodic_intonation`, `harmonic_intonation`; helpers `ellis_delay(bpm)`, `ride_ratio(bpm)`, `ensemble_sigma_ms(ioi_ms)`), `euclid`, `arp`, `stinger`, `scatter`, `arrangement`, `adaptive` (`first`)
- **cyber** (`docs/cyber.md`) — sfx: `transformer`, `gearbox`, `bearing`, `pipe_engine`, `turbo`, `tyre_squeal`, `pass_by` (`sources`), `traffic` (road traffic flow: Poisson pass-bys at the speed limit), `induction_motor` (2× mains, slot passing, slip sidebands, `fault`), `machine_hum` (Gaver FM machine), `cyber_ui` (kind: scan, confirm, deny, hack, decrypt, upload, warning, target_lock, implant, neural_link); `bearing` (`size`, `load`, `rotating`, `defect`, `ring`, `mirror`), `transformer` (`joints`, `stress`, `packet`, `grid`), `gearbox` (`mount_mass`), `pipe_engine` (duct options, `engine_rpm`); effects: `car_cabin` (heard from the driver's seat), `packet_loss` (`channel`), `beat_repeat`, `paulstretch` (`drift`), `waveset`, `formant_shift`, `cyber_voice`, `texturize`, `adaptive_fm`, `grain_delay`; CLI: `genny identify file.wav`
- **klang** (`docs/klang.md`) — instruments: `viola`, `bassoon` (`meyer`, `precursor`), `cor_anglais`, `bass_clarinet`, `a_clarinet`, `eb_clarinet`, `contrabassoon`, `heckelphone`, `bass_trombone`, `natural_horn`, `clarino`, `piccolo`, `recorder`, `hand_chime`; sfx: `toy_boat`, `klang_car`, `harrier`, `bicycle`, `klang_rain`, `ui_tick`, `ui_chime`, `damage`, `sniff`, `strain`, `weight_shift`, `handle`, `tide`, `distant_bell` (`diameter`); layers: `articulate`, `sampler`
- **physical/winds** (`docs/physics.md`, Python only) — wind acoustics and self-oscillating models: `auvray_recorder`, `modal_clarinet`, `trombone_lip`, `trumpet_lip2d`, `lip_buzz`, `buffet_clarinet`, `tmm_impedance`, `lip_threshold`, `register_track`. Other Python options (`docs/acoustic.md`, `docs/physics.md`): `physical.bank_piano.note` (`polarisations`, `bidirectional`, `bridge_res`, `bridge_angle`, `pedal`, `sympathetic_strings`, `lid_state`, `ribs`, `action_kind`; no damper above MIDI 89), `physical.bow.bowed(wolf=)`, `physical.waveguides.modular_string` / `bridge_filter`, `strings.render_pluck(glide_couple=)`, `physical.nonverbal.vocalize(rhythm=preset|oscillator)`, `damped_calls(n, rate, decay)`, preset `moan_sova`

Extra layer types are used like the built-in ones (`"type": "compose"`, `"sing"`, ...) and take the common layer keys
(`at`, `gain`, `fx`, `repeat`/`every`). The ones you will reach for most:

```jsonc
{"type": "sing", "section": "sopranos", "singers": 8, "steps": "C5:1 D5:1 E5:2", "lyrics": "A-ve Ma-ri-a", "style": "cathedral"}
{"type": "satb", "chords": "C4:maj:2 F4:maj:2 G4:dom7:2 C4:maj:2", "lyrics": "A-men A-men", "singers": 4}
{"type": "compose", "role": "melody", "inst": "flute", "key": "D", "scale": "dorian", "bars": 8, "seed": 7}
{"type": "arrangement", "style": "funk", "key": "E", "bars": 8, "seed": 3}
{"type": "euclid", "kind": "conga", "k": 5, "n": 8, "bars": 4}
{"type": "stinger", "mood": "victory", "inst": "trumpet", "seed": 1}
{"type": "song", "rows": 4, "instruments": {"lead": {"inst": "psg"}, "kick": {"drum": "kick"}}, "patterns": {"A": {"lead": "C-4 . E-4 . G-4 . C-5 .", "kick": "x . . . x . . ."}}, "order": ["A", "A"]}
```

Choosing by physics rather than by name: a struck, dropped, rolled, scraped or broken object is `impact`, `bounce`,
`drop`, `roll`, `scrape`, `smash` with `material` and `size`; anything rubbed or squeaking is in `friction`; water,
steam, fire, sparks and ice are in `matter`; a room is the `room` effect with a `preset` or its dimensions; a sound
heard from far away, moving past, or through a wall is `distance`, `doppler`, `occlude` / `wall`.

Changed from earlier versions: `pour` takes `fill_start` / `fill_end` (was `fill`); `heartbeat` is a lub-dub model
with `rate` and `dur`; `flanger`, `phaser`, `chorus`, `roll`, `scrape`, `hit`, `fire` are new models that still accept
their old parameters; plucked strings (`guitar`, `harp`, `banjo`, basses...) now ring for their natural decay after
the note ends, so write shorter notes or lower `gain` if a line smears.

## Instruments: register, brightness, loudness

The instruments are voiced so they do not pierce. Brightness is set in Hz, not in harmonics: upper partials roll off above about 3 kHz, FM clang is capped by the note and filters key-track with a ceiling. A voice is rich in its low register and gets purer as it climbs, like a real instrument, so a melodic layer rarely needs a `lowpass` to be bearable. What is left to you:

- **Write in the instrument's register.** The catalog gives a `range` for each (`piano` A0-C7, `upright_bass` E1-G3, `glockenspiel` G5-C8). Bass lines E1-G3, chords and pads C3-C5, melodies C4-C6. Above C6 use the voices made for it (`glockenspiel`, `celesta`, `music_box`, `glass`, `xylophone`, `whistle`) and keep them soft (`vel` 0.5-0.7, low `gain`): a clean tone at 2-4 kHz is where the ear is most sensitive.
- **Brightness controls**: `vel` (softer = darker on most voices), `bright` (`piano`, `bell`, `brass`, `chip`, `clav`, `synth_pluck` have their own; every other instrument takes a generic `bright` 0..1 and `warmth` 0..1, 0.5 = as voiced, a +-9 dB shelf at 2.5 kHz / 300 Hz; also `onset_lag` in ms: the upper partials enter late, 5-20 for a brassy or bowed speech), `cutoff` (`lead`, `pad`, `soft_lead`, `warm_pad`, `bass`, `acid`, `reese`), `tone` (`electric_guitar`, `finger_bass`, `dist_guitar`, `hihat`). Anything else: a `highshelf` (+3 dB at 3000 for air) or `lowpass` in the layer `fx`.
- **Raw building blocks are not tamed**: the `synth` instrument and the `tone` sfx give exactly the wave and cutoff you ask for. Keep `cutoff` at or below about 4x the note (and under 3500) unless harshness is the point. The tonal sfx (`beep`, `blip`, `coin`, `powerup`, `laser`, `jump`, `alarm`...) round their square/saw waves themselves.
- **Levels are matched**: at the same `gain` and `vel` every instrument is about as loud as a kick or snare, so `gain` is the mix (see "Music"). High notes are trimmed 1.5 dB per octave above C5.
- **Loudness ceiling**: after peak normalisation, a sound that is still louder *to the ear* than `"max_loudness"` (default -6, ear-weighted dB) is turned down to it. Only held tones in the 1-5 kHz range ever reach it (a full-scale 3 kHz tone is 8 dB over); drums, noise, bass and short blips never do. `"max_loudness": null` on a spec or batch disables it.
- **Check without ears**: `genny info file.wav` prints `loud` (K-weighted level of the loudest 0.4 s), `ear` (ear-weighted level of the loudest 0.2 s: what the ceiling limits), `sharp` (sharpness in acum) and `above5k` (share of energy above 5 kHz), then `crest`, `tp` (true peak), `lat` (log10 attack time in s) and `flux` (median spectral variation: ~1e-6 struck, 2e-4 to 1e-3 bowed/blown/pad, 0.1 noise); `--json` also has `tilt` (spectral slope in dB/octave, 100 Hz-4 kHz; commercial mixes average -5). For anything with a pitch: `sharp` up to about 1.6 is soft, 1.6-2.0 bright, above 2.0 it bites (a pure tone already reads 1.0 at C6 and 1.5 at C7); `above5k` over 3 % is harsh. Hats, cymbals and shakers are 2.5-3.5 by nature: keep them quiet instead.

## Music (tempo, loops, styles)

Set `"bpm"` on a spec (or on a batch, a group or one layer) and every musical time in it is in **beats** instead of seconds: `at`, `dur`, the durations in `steps`, `step`, `gap`, `hits`, `every`. Performance and sound-design times (`strum`, instrument/sfx `params`, fx times, `tail`, `duration`) stay in seconds.

```jsonc
{
  "out": "funk_loop.wav", "bpm": 104,
  "beats": 16,                // hard length: 4 bars of 4/4
  "tail": 1.5,                // seconds of ring-out kept after the last beat (ignored with "loop": "fold")
  "layers": [
    {"type": "seq", "inst": "slap_bass", "step": 1, "steps": "E2:.5 -:.25 E2:.25 -:.25 G2:.25 A2:.5 -:.5 E3:.25 -:.25 D3:.5 E3:.5 | E2:1 -:1 G2:1 A2:1", "gain": 0.9},
    {"type": "seq", "inst": "clav", "step": 1, "steps": "-:.5 [G3,B3,D4]:.25 -:1.25 [A3,C#4,E4]:.25 -:1.75", "gain": 0.45, "at": 8},
    {"type": "pattern", "kind": "kick",  "steps": "X--x------X--x--", "bars": 4, "gain": 0.7},
    {"type": "pattern", "kind": "snare", "steps": "----X--o-o--X--o", "bars": 4, "gain": 0.55},
    {"type": "pattern", "kind": "hihat", "steps": "x-x-x-x-x-x-x-xx", "bars": 4, "gain": 0.25, "swing": 0.15}
  ],
  "fx": [{"type": "reverb", "mix": 0.14, "size": 0.45}, {"type": "compressor", "threshold": -16, "ratio": 2.5}]
}
```

- **`steps` for notes**: durations are beats (`C4:1` a quarter, `:.5` an eighth, `:.25` a sixteenth, `:.3333` a triplet eighth); a token without a duration lasts `step` beats (default 0.25, so set `"step": 1` for quarter-note lines). `|` is a bar line and is ignored. One-shot: `genny seq sax "C4:1 E4:.5 G4:1.5" --bpm 96`.
- **`pattern` for drums**: `steps` is a grid, by default sixteenths with a tempo (`"step": 0.25` beats; `0.3333` for triplets, `0.125` for 32nd rolls). `x` = hit, `X` = accent (1.4x), `o` = ghost note (0.4x), anything else = rest; spaces and `|` are ignored. `"bars": N` plays the written pattern N times. `"swing": 0..0.6` delays every other step by that fraction of a step (0.15 lazy, 0.33 shuffle; or write true triplets with `"step": 0.3333`).
- **Length and loops**: `"beats"` fixes the length. `"loop": "fold"` makes a sample-exact musical loop: whatever rings past the last beat (release tails, reverb) is laid back over the start, which is what the next repetition would sound like, so the file loops with no gap, click or missing tail. (`"loop": true` is the crossfade loop for drones and engines, not for music.)
- **Mix by `gain`** (instruments are level-matched): kick 1.0, snare/clap 0.6-0.85, hats/shaker/ride 0.25-0.4, crash 0.3-0.5, bass 0.8-0.9, chords and pads 0.35-0.5, arpeggios 0.3-0.4, lead 0.6-0.8, counter-melody 0.4-0.5. A `distortion` on a layer brings it to full scale: drop that layer to 0.25-0.35. Master: a small `reverb` (mix 0.12-0.25) and a gentle `compressor` (threshold -16, ratio 2.5), `normalize` -2.
- **Per-note samples for a game's own sequencer**: render one file per reference pitch every 6-12 semitones (the engine re-pitches by a few semitones), all with the same spec apart from the note. Do not bake LFOs, tremolo or detuned pairs into them (re-pitching changes their speed: a wobble that drifts against the beat); add movement at runtime or keep it very slow.

**Style palette.** `styles.json` (next to the skill's `SKILL.md`; in the repo `examples/library/styles.json`, written by `examples/build_styles.py`) holds a complete 4-8 bar loop for each of these, a dozen lines each: find `"out": "<style>.wav"` and read that entry before writing your own.

| style | bpm | drums | bass | harmony | lead and colour |
|---|---|---|---|---|---|
| `lofi` | 70-85 | soft `kick`, `snap` + `brush` backbeat, dark `hihat`, swing 0.18 | `upright_bass` | `epiano` 9th chords | `felt_piano`, `vinyl` sfx bed, master `lowpass` 5000 |
| `jazz_swing` | 120-180 | `ride` in triplets, `hihat` on 2 and 4, `brush` | walking `upright_bass` | sparse `piano` stabs | `trumpet` (`mute`), `sax`, `vibraphone` |
| `bossa_nova` | 120-140 | `rim` bossa clave, `shaker` 16ths, soft `kick` | `upright_bass` root-fifth | `guitar` syncopated chords | `flute` |
| `blues_shuffle` | 90-120 | triplet grid: `kick`, `snare`, `hihat` | walking `finger_bass` | `electric_guitar` boogie 5th-6th | `harmonica` |
| `soul_ballad` | 60-80 | `kick`, `snap`, quiet `hihat` | `finger_bass` | `wurli`, `strings` pad | `sax` |
| `funk` | 95-115 | syncopated `kick`, ghosted `snare`, 16th `hihat` | `slap_bass` | `clav`, `muted_guitar` scratch | `brass` stabs |
| `disco` | 115-125 | four-on-the-floor, off-beat `openhat`, `clap` | octave `finger_bass` | `string_machine`, `clav` chick | `violin` line |
| `reggae` | 70-80 | one drop: `kick` + `rim` on beat 3, swung `hihat` | spacious `finger_bass` (`tone` 1100) | `organ` + `muted_guitar` off-beat skank | `trombone` |
| `latin_salsa` | 90-100 | `clave` 2-3, `conga` tumbao, `cowbell`, `timbale`, `guiro` | anticipated `upright_bass` | `piano` montuno | `trumpet` |
| `reggaeton` | 90-98 | dembow: `kick` every beat, `snare808` on "a" of 1 and "and" of 2 | `bass808` | `warm_pad` | `marimba` hook |
| `rock` | 110-150 | `kick`/`snare` backbeat, 8th `hihat`, `crash` | `finger_bass` eighths | `electric_guitar` power chords + `distortion` fx | `dist_guitar` |
| `metal` | 150-190 | 16th double `kick`, half-time `snare` | `finger_bass` + light `distortion` | `electric_guitar` gallops + `distortion` drive 16 | |
| `pop` | 100-120 | `kick`, `clap`, 8th `hihat` | `finger_bass` | `piano` chords | `soft_lead`, `synth_pluck` arp |
| `synthwave` | 90-110 | four-on-the-floor, `snare808` with reverb, `tom` fills | `bass` eighths | `string_machine` | `lead` + chorus, `synth_pluck` arp with delay |
| `house` | 120-128 | four-on-the-floor, `clap`, off-beat `openhat` | off-beat `fm_bass` | `piano` stabs, `warm_pad` | |
| `acid_techno` | 128-136 | four-on-the-floor, off-beat `hihat`, `clap`, `rim` | `acid` 16ths, accents by `@vel` | | |
| `drum_and_bass` | 165-175 | two-step `kick`, `snare` on 2 and 4 with ghosts | long `reese` notes | `glass_pad` | `bell` stabs with delay |
| `trap` | 130-150 | half-time `clap` on 3, `hihat` 16ths with 32nd rolls | long `bass808` | `warm_pad` | dark sparse `bell` (`bright` 0.6) |
| `chiptune` | 130-160 | bitcrushed `kick`/`snare`, short `hihat` | `chiptri` | fast `chip` arpeggio (`width` 0.125) | `chip` (`width` 0.25) |
| `atari_vcs` | 120-150 | `tia` `noise` (AUDC 8) hits | `tia` `bass` | — | `tia` `lead` (keep both voices on notes their grids share) |
| `orchestral_heroic` | 90-120 | `timpani`, `taiko`, `crash` accents | `cello` | `strings` chords | `french_horn` theme, `trumpet` answer, `harp` gliss |
| `baroque` | 90-110 | none | `cello` walking line | `harpsichord` broken chords | `oboe`, `violin` |
| `ragtime` | 90-105 | none | `piano` left-hand stride | | `piano` syncopated right hand |
| `lullaby` | 70-90 (3/4) | none | | `celesta` broken chords, `glass_pad` | `music_box` |
| `spooky` | 80-100 | `timpani` | | `church_organ` diminished chords, `celesta` ostinato | `theremin`, `tubular_bell` |
| `christmas` | 100-120 | `sleigh` eighths | `upright_bass` | `strings`, `french_horn` | `glockenspiel`, `tubular_bell` |
| `celtic_jig` | 110-125 (6/8) | `djembe` as bodhran | | `harp` chords | `violin` fiddle (`attack` 0.025), `whistle` |
| `bluegrass` | 110-130 | none (the `mandolin` chop is the snare) | `upright_bass` on 1 and 3 | `steel_guitar` boom-chick | `banjo` roll, `violin` |
| `polka_oompah` | 110-125 | `kick`, `snare` off-beats, `crash` | `tuba` | `accordion` off-beats | `clarinet` |
| `tango` | 110-125 | none | `upright_bass` | staccato `accordion`, `piano` | `violin` (`vibrato` 1.4) |
| `andean` | 90-110 | low `taiko` as bombo, `shaker` | `guitar` bass notes | `guitar`, `mandolin` (`tremolo` 12) | `pan_flute` |
| `middle_eastern` | 90-105 | `djembe` bass/slap (dum-tek), `tambourine` | | `strings` drone, `pluck` riff | `oboe` in Hijaz (D Eb F# G A Bb C) |
| `indian_raga` | 80-100 | `tabla` ge/na/ke | | `sitar` drone (root + fifth) | `sitar` |
| `japanese` | 70-90 | sparse `taiko`, `woodblock` | | `koto` arpeggios in the In scale (D Eb G A Bb) | `pan_flute` (`breath` 0.5) |
| `gamelan` | 90-100 | `gong` on the cycle | `handpan` | interlocking `gamelan` patterns | `gamelan` core melody |
| `ambient` | 50-70 | none | | `warm_pad`, `glass_pad` | `handpan`, `singing_bowl`, reverb mix 0.4 |

## Speech (spoken letters, numbers, short words)

A built-in Klatt-style formant synthesizer (no models, no downloads) for short spoken cues: letter names, numbers, and short game words/phrases ("fuel low", "level up", "game over"). English by default; Spanish with `lang=es`.

```
genny say "G" -o say/g.wav
genny say "fuel low" -p voice=female -o say/fuel_low.wav
genny say "G" -p lang=es -o say/g_es.wav                      # "ge" /xe/
genny say "level up!" --show -o say/level_up.wav               # --show prints the phonemes it used
genny say "JH IY1" --phonemes -o say/g_raw.wav                 # raw ARPAbet
```

Layer (all common layer keys work: `at`, `gain`, `fx`, `repeat`/`every`):

```jsonc
{"type": "speech", "text": "fuel low", "params": {"voice": "female", "rate": 1.1}}
{"type": "speech", "text": "S", "params": {"lang": "es"}}          // "ese"
{"type": "speech", "phonemes": "F Y UW1 AH0 L | L OW1"}            // raw ARPAbet, "|" = word break, "_" = short pause
{"type": "speech", "text": "go /G OW1/ now"}                       // /slashes/ inside text = raw phonemes
```

Params (`genny list speech`):

| param | default | meaning |
|---|---|---|
| `voice` | `male` | `male`, `female`, `child`, `whisper` (unvoiced); robots/creatures: `robot` (monotone + ring-mod), `android` (clean, snapped pitch, faint ring + chorus), `synth` (vocoder-style saw carrier, semitone-snapped), `bad_robot` (bit-crushed, stutters/dropouts/pitch hiccups), `evil_robot` (very low, octave-down sub, slow ring, overdrive), `monster` (huge tract, growl, distortion), `giant` (big slow natural), `alien` (tiny tract, high, warbling ring), `ghost` (swimming whisper) |
| `pitch` | from voice | base F0 in Hz or a note name (male 115, female 205, child 280, robot 100, evil_robot 62, monster 58, alien 240); every preset still takes `pitch`/`formant_shift`/`rate` overrides |
| `rate` | 1.0 | speed multiplier |
| `formant_shift` | from voice | vocal-tract size: 0.85 = big/deep, 1.16 female, 1.28 child |
| `breath` | from voice | breathiness 0..1 |
| `vibrato` | 0 | depth in semitones |
| `intonation` | `auto` | `statement` (falling), `question` (rising), `exclaim`, `flat`; `auto` follows a final `?`/`!` |
| `emphasis` | 1.0 | pitch-accent strength |
| `lang` | `en` | `en` or `es` |
| `gap` | 0 | extra seconds between words (countdowns, spelling) |
| `locus` | `class` | `class` or `place`: labial stops follow the vowel's F2 almost fully (slope 0.95) and velar onsets may fall to 750 Hz before back vowels |
| `seed` | 0 | tiny jitter/noise variation between takes |

Text handling: a lone letter (or an all-caps token of 2–5 letters not in the dictionary, e.g. `GPS`) is spoken as letter names; digits become number words (`21` → "twenty one"; Spanish "veintiuno"); `,` `.` `!` `?` split phrases with a pause. English words come from a built-in dictionary (`genny list words`: numbers, letters and ~250 common game words — go, ready, level, up, fuel, low, bomb, shield, magnet, turbo, time, game, over, perfect, combo, bonus, score, win, lose, danger, warning, coin, lap, left, right...) with a rough letter-to-sound fallback for anything else — check odd words with `--show` and pass phonemes if it guesses wrong. Spanish uses spelling rules (Castilian: `z`/`ce`/`ci` = TH, `j`/`ge` = X, `r` tap / `rr` trill, stress from accents and the penultimate rule), so any Spanish word works.

Phonemes (`genny list phonemes`), ARPAbet with stress digit on vowels (1 primary, 2 secondary, 0 unstressed):
- vowels `IY IH EH AE AA AO UH UW AH AX IX ER AXR`, diphthongs `EY AY OY AW OW`, Spanish pure vowels `A E I O U`
- stops `P B T D K G`, fricatives `F V TH DH S Z SH ZH HH X` (X = Spanish jota), affricates `CH JH`
- nasals `M N NG`, liquids/glides `L R W Y`, flap/tap `DX`, trill `RR`

Tips:
- Output is dry mono; `normalize: -3` is a good level. A touch of `reverb` (mix 0.08–0.15, size 0.3) makes it sit in a mix; `telephone` (+ `distortion` light) = radio/announcer; `voice: robot`/`android`/`synth` = computer, `bad_robot` = broken machine, `evil_robot`/`monster` = villain/boss, `alien`/`ghost` = otherworldly (character voices are mono and dry — add fx on top); `pitch` + `formant_shift` 0.85 = giant/boss voice.
- Keep utterances short (1–4 words). It is intelligible but clearly synthetic; single letters and numbers are the strongest, long sentences the weakest.
- Weakest sounds: `TH`/`DH`/`F`/`V` (quiet, easily confused), `NG`, and the letter-to-sound fallback for unknown English words.

## Pinball / mechanical foley

Real machines are mechanism, not music: build them from these instead of `bell`/`glass`/`marimba` (tonal mallets make a table sound like a bell tree). Each is a short noise transient + inharmonic steel modes that die in tens of ms + a wooden-cabinet thump.

- `solenoid` — any coil or relay: `size` 0 (relay/switch tick) .. 1 (big coil), `metal` ring, `thump` cabinet thud, `bounce` armature rebound (the "ka-chak"). Use size 0 for menu ticks and switch hits.
- `flipper` (`force`, `buzz` AC coil hum), `pop_bumper` (`tone` 0.6..1.6 tells bumpers apart), `slingshot` (`tone`), `knocker` (the replay/extra-ball BANG; `size`, `rattle`).
- `steel_ball` — ball impact, `surface` metal (rail/post clank) | wood (tock) | rubber (thup) | plastic (target/ramp clack), `force` 0..1.
- `ball_roll` — steel ball on a wooden playfield, `speed` 0..1; loop it with `"loop": 0.3`.
- `spinner` — one flap tick per pass (`spins` > 1 = a decelerating whirr); `spring` — plunger, `action` pull (creak tick) | release (rod slam + boing), `tension`.
- Solid-state sound board (80s pinball/arcade speaker): `chirp` sfx (pitch sweep with `warble` LFO, `steps` for stepped bip-bips, `bits` DAC grit) and the `board` instrument (gritty pulse with a pitch blip on every note) for score blips and jingles; `chiptri` makes the bass. Recipe: foley layer + one short `chirp`/`board` blip = a scoring hit (e.g. `pop_bumper` + square `chirp` 980→330 Hz 0.07 s `bits` 5 at gain 0.3).

## Design recipes (what tends to sound right)

- **UI / notification**: short (0.1–0.6 s), C5–C6 (up to C7 only with `glass`/`celesta`/`glockenspiel`/`music_box` at `vel` ≤ 0.7), `bell`/`glass`/`marimba`/`kalimba`/`celesta`, a small `reverb` (mix 0.15–0.3, size 0.3–0.5). Two rising notes = positive, two falling = negative. Keep `normalize` at -1 to -3 dB. Build a UI set from small variations of one or two sources, in the key of the menu music, one tone per action; keep loud UI out of the 2-4 kHz piercing zone and away from phone ringtones.
- **Wall proximity / radar**: `sfx proximity` with `rate` 2 (far) → 12 (close); or a `beep` layer with `repeat`/`every`.
- **Stingers**: `seq` 3–6 notes, 0.08–0.15 s each, ending on a longer note or chord; layer a `pad`/`strings` chord underneath at `gain` 0.4–0.6; master `reverb` mix 0.2–0.35. For a flavour, borrow the lead and harmony instruments of a style from the palette in "Music" (`koto` + `pan_flute`, `banjo` + `violin`, `harpsichord` + `oboe`...).
- **Win / fanfare**: `brass`, `trumpet` or `lead` arpeggio up a major triad ending on the octave chord, `french_horn`/`strings` underneath; add `crash`, `timpani` or `taiko` at `at: 0`; `chip`+`bitcrush` for retro.
- **Game over**: descending minor line (`piano`, `felt_piano`, `cello`, `chip`, `organ`), slow `pad`/`warm_pad` minor chord, `powerdown` sfx, `taiko`/`kick808`/`gong`; darken with `lowpass` 1500–3000 or `muffle`; long `reverb` (size 0.8).
- **Impacts**: `hit`/`punch`/`explosion` + `kick` layer for weight; `distortion` for crunch; `lowpass` sweep for distance. For physics impacts render soft / medium / hard sets (velocity bands, e.g. under 15 %, 15-40 %, over 40 % of the maximum) rather than one file scaled by the engine; more than 4 hits within 0.3 s → one "many" file.
- **Layering rule of thumb**: 2–4 layers; one body (low), one transient (click/noise), one tail (reverb/pad). Sum gains ≈ 1–1.5, normalize handles the rest.
- **Dialogue over a mix**: give the voice layer a higher `priority` than music and sfx (each step ducks lower layers by 3 dB, 12 dB at most; engines typically drop SFX and music to 0.2, -14 dB, while dialogue plays), or add `"duck_band": [500, 4000]` on the music and effects layers to notch the speech band instead of lowering everything. High-pass voices at 100-150 Hz (male) or 150-200 Hz (female).
- **Ambience beds**: a stereo bed, never true silence (room tone at least), plus mono one-shots of the recognisable details (an owl, a dog, a door) for the engine to scatter (e.g. every 3-8 s); layer far / mid / close. Never one long loop with a recognisable event in it: render about 6 `loop` files of 2-8 s (~30 s in all) for random playback. Layers of co-prime lengths (7, 9, 11 s) repeat only after their least common multiple; never stack identical copies 1-35 ms apart (comb filtering). Distant rumbles can go at 11-22 kHz (`sr`), crickets and other high sounds need 44.1 kHz.
- **Variations**: footsteps, impacts, creatures, debris: render a set with `"variants"` (4-8 files) for the engine to shuffle without repeating the last one. Keep `vary.pitch` subtle (0.95-1.05, at most 0.8-1.1) and vary gain more than pitch for creatures and footsteps; weapons tiny or none; dialogue never. UI, pickups and other "information" sounds stay identical. Render explosion body, debris and tail as separate sets (N sets recombine into 2^N − 1 mixes); fragments under 0.4 s chained at random (radio crackle, fire, robot beeps) make long beds from a few files. Without `"variants"`, change `vel`, `transpose`, `params.decay`, effect `mix` or `seed` by hand and name files by intent (`ui_confirm_soft.wav`).
- **Weapons**: firing interval = 60 / RPM (600-1,200 RPM: 0.05-0.1 s). Held fire: a short dry shot retriggered by the engine plus a separate release tail; above ~1,500 RPM a seamless `loop` plus a tail. Sharper crack on the first shot of a burst. Render a distance set (dry, medium, distant, echo) with `distance` rather than low-passing one file.
- **Character and surface sets**: a character needs dodge, double jump, light / heavy land, footstep, death, pain small / medium / large, gib; footsteps and impacts per surface (stone, dirt, foliage, glass, water deep / shallow, metal, snow, wood, sand, energy, flesh).
- **Adaptive music**: escalate fast (transition `quantize: "beat"`, a stinger masked by the action), calm down slowly (`"bar"`, through an outro state; crossfades 1-2 s in, up to 5 s out). 4-8 bar loops with a few varying parts last for minutes. Add a "bored" state that fades to silence after a few minutes without events. `first: true` for loops that fold their tail. Keep stems short (a long stem delays a remix until it restarts). Pitched stingers fit only their chord; percussive ones fit anything.
- **Chip music**: lead on one pulse, harmony on the other, bass on triangle (NES) or wave (Game Boy), drums on noise; chords as fast `arp`; echo = the lead repeated a few frames later, quieter, on the second pulse; boss loops 1-2 sections of 8-10 bars, overworld longer; transitions by a 1-2 bar rising scale. On `tia` keep both voices on notes their grids share.

## Typical agent workflow

1. `genny list --json` once, cache it.
2. Write one batch JSON per folder/category with descriptive `out` names.
3. `genny render batch.json --quiet`; read the `N rendered, M failed` line and fix any `ERROR in spec #i` lines.
4. `genny info` on outputs if you need to verify durations/levels. `genny play` to audition (blocks until done).

## Vehicle engines (`car_engine`)

`{"type": "sfx", "kind": "car_engine", "params": {"rpm": 3000, "load": 0.6}}` is a physically-modelled combustion engine: per-cylinder firing pulses (4-stroke, fixed per-cylinder strength = the lope), exhaust pipe comb + body resonances + muffler lowpass that opens with `load`, a low exhaust burble and intake breath modulated by the firing. Params: `rpm` (crank speed), `cylinders` (4), `load` 0..1 (throttle: brighter, more intake), `rough` 0..1 (uneven firing, jitter, misfires), `pipe` (exhaust delay s, longer = throatier), `body` 0..1, `bright` 0..1 (0 = road car from the cabin, nothing above ~2.5 kHz; raise for an open sporty exhaust with combustion rasp), `seed` (engine character), `dur`. `model`: `classic` (default), `ptr` (pulse train and resonators: pulse exponent ν 0.9, envelope α 7.6-11.5 and β 1.15-2.3 /rad with `load`, per-cycle jitter sd 2/720 of a cycle, `inharm` 0..1 a Gaussian bump of detuned partials up to ±8 %), `pd003` (the four-cylinder Pd engine003 patch: 1/(1+x²) pulses, two cross-fed paths, four 50 ms guides). Keep `bright` at 0 for the player's own car: sharp/raspy engine timbres read as piercing and "too loud" long before their level is.

- **Engine bank for a player car:** render 4–6 loops at fixed rpm (e.g. 1100, 2000, 3000, 4200, 5600; `"loop": 0.3`, `duration` 3.3) and at runtime equal-power crossfade the two nearest, each played at `rate = rpm / renderedRpm`. One sample pitched across the whole range sounds like a tape speeding up. Export batch: a 300 rpm/s run-up from max(1000, 1.3 × idle) to the limiter − 100, free revs, then 5 s loops (50 ms crossfade) every 500 rpm with throttle min(1, 0.25 + 0.75 rpm / limit), e.g. `car_engine(rpm=r, load=min(1, .25 + .75 * r / 7500), dur=5)` for r = 1000, 1500 ... 7000.
- **Ignition / stall:** chain short `car_engine` layers — cranking is `rpm` 220–300 with `rough` 0.9 and a starter-motor `synth` saw with tremolo; a catch is a burst at 2300–2600 then settling to idle; a stall is 1100 → 700 → 420 with fades.
- The generic `engine` sfx is a simple saw hum; prefer `car_engine` for anything the player drives.

## Loops (for runtime engines that loop a sample)

Set `"loop": true` (or `"loop": 0.08` for a custom crossfade in seconds) at the top level of a spec. The last `crossfade` seconds are blended into the start with an equal-power crossfade and cut off, so playing the file with `loop=true` in a game engine has no click or gap. `trim`/`declick` are skipped for loop specs (they would break the seam). Use a `duration` long enough to hide the repetition (2–6 s for drones, wind, engines). The same thing is available as an effect: `{"type": "loop", "crossfade": 0.05}`. For music use `"loop": "fold"` with `"bpm"` + `"beats"` instead (see "Music"): the ring-out is folded onto the start and the length is exact.

## Changing genny's code

- No source citation in the code: no paper, author, table or figure number in comments or docstrings. The source of every constant goes in `research/SOURCES.md`, one line per value under its module. Some modules still carry citations and UNSOURCED markers in the code: when you touch one, settle the value (see the next point), move its source to `SOURCES.md` and delete the marker. Do not add new ones.
- How to add an entry, and the rules the tests enforce: `docs/README.md`, "Extending genny". Afterwards: `uv run python examples/level_match.py`, `uv run python examples/build_docs.py`, and a line in `CHANGELOG.md` (say it when a default sound changes).
- A value with no readable source is still set: take it from physics or from the published literature you know, and write it in `research/SOURCES.md` as `connaissance: <short reason or presumed reference>`. Never present it as read, and never leave a gap only because a paper is missing.

## Keeping the documents up to date

Every change to the code updates its documents in the same piece of work, not at release. A change is not finished until all of these hold:

- **`research/SOURCES.md`**: one line per changed or new constant, under its module's section. Whether the value was read (document, page, table or equation) or comes from knowledge (`connaissance`), say which. When a value changes, edit its existing line; when code is deleted, delete its lines.
- **`docs/<module>.md`**: what each new or changed sound, parameter or model does, written for a user. Remove what no longer exists. `docs/physics.md` names the physical model behind each entry: update it when the model changes.
- **`AGENTS.md`**: the catalog line of the module (above) lists every new sfx, instrument, drum, effect or layer and the parameters an agent needs. Rename or remove entries there when the code does.
- **`CHANGELOG.md`**, section "Unreleased": one line per user-visible change, and an explicit note when a default sound changes.
- **Generated files**: run `uv run python examples/build_docs.py` (rewrites `docs/catalog.md` and `docs/api.md`) after any change to a signature or a docstring; never edit those two files by hand. Run `uv run python examples/level_match.py` when a default level changes.
- **`THIRD_PARTY_NOTICES.md`**: add the licence of any data, table or code taken from a third party.
- **Tracking files**: when the work comes from a report in `../reports/` (for example `../reports/lecture_profonde/suivi/*.md`), update the status of the line you handled (`fait`, `fait (connaissance)`, or the precise reason it is not done), in the same session.

Before declaring the work done, check these documents with grep for the names you added, renamed or removed. `AGENTS.md`, `CHANGELOG.md` and `genny/cli.py` use CRLF line endings: edit them with an exact-string edit tool, or run `unix2dos -q` after a tool that rewrites line endings. Agents working in parallel on separate modules do not edit `AGENTS.md`, `CHANGELOG.md`, `docs/*` or `_levels.py`: they list the needed doc changes in their report, and the coordinator applies them all and then runs `build_docs.py` and `level_match.py` once.
