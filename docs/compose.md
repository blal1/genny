# compose — music from a seed

`genny/compose.py` generates notes instead of asking you to type them. Every generator is deterministic:
the same layer (same `seed`) gives the same notes and the same samples; another seed gives another take.
With a tempo in effect (`"bpm"`), all musical times are beats, as everywhere else in genny.

| name | type | what it is |
|---|---|---|
| `compose` | layer | one generated part: melody, bass, chords, arp, drums, pad or counter-line |
| `arrangement` | layer | a whole piece from a style name: drums + bass + harmony + lead with one shared harmony |
| `euclid` | layer | Euclidean rhythm E(k, n) on a drum (or a pitched note) |
| `scatter` | layer | random placement of child layers: ambiences, debris, bird groups |
| `arp` | layer | arpeggiate the chords you give |
| `stinger` | layer | a short jingle by mood: victory, defeat, mystery, danger, pickup, unlock |
| `adaptive` | layer | game-music states rendered as synchronised loops, stems and transition stingers |
| `ks_string` | instrument | extended Karplus-Strong string (Jaffe-Smith) |
| `ks_drum` | drum | Karplus-Strong drum |

Python helpers (no CLI needed): `compose.generate(layer)` → note events, `compose.to_spec(layer)` →
ordinary `seq` / `pattern` layers, `compose.harmony(layer)`, `compose.arrange(layer)`,
`compose.automix(layers)`, `compose.euclid(k, n, rotate)`, `compose.voice_lead(prev, pcs)`,
`compose.scale_offsets(scale)`, `compose.quantize(pitch, key, scale)`, `compose.tempo_delay(bpm, k)`,
`compose.style_bpm(style, seed)`, motif operators `transpose` / `invert` / `retrograde` / `augment`.

All the common layer keys (`at`, `gain`, `fx`, `repeat` / `every`, `bpm`) work on every layer here.

## `compose` — one generated part

```jsonc
{"type": "compose", "role": "melody", "inst": "sax", "key": "D", "scale": "dorian",
 "progression": "i VII VI VII", "bars": 8, "seed": 7, "density": 0.5, "range": "C4-C6",
 "swing": 0.15, "humanize": 0.5}
```

| key | default | meaning |
|---|---|---|
| `role` | `melody` | `melody` (or `lead`), `bass`, `chords`, `arp`, `drums`, `pad`, `counter` |
| `inst` | by role | any instrument (`params`, `strum` are passed on); drums use `kit` instead |
| `key` | `C` | tonic, a note name without octave |
| `scale` | style's, else `major` | a scale name, or a custom tuning (see "Scales and tunings") |
| `tuning` | `equal` | `just` retunes a 12-TET scale to 5-limit ratios |
| `progression` | `auto` | roman numerals, one chord per bar, cycled: `"i VII VI VII"`; `{"A": "...", "B": "..."}` per section; `auto` = the style's progressions (or a tonic → subdominant → dominant walk); `pokesdown` = Farnell's three unresolved 8-note chords |
| `chords` | — | explicit chords instead: `[["D3","E3","F#4",...], ...]` (note names or MIDI) |
| `harmonic_rhythm` | 1 | bars per chord |
| `chord_size` | style's, else 3 | notes per chord (2 = root + fifth, 4 = sevenths) |
| `bars` | 4 | length; the part never sounds a note past `bars × meter` beats |
| `sections` | `A` | form: `"intro A A B A outro"`; equal bars each; a repeated letter repeats its material note for note; `intro` / `outro` use the first letter's harmony at low intensity |
| `intensity`, `threshold` | all 1, 0 | intensity per section (or any list spread over the bars); the part is silent in bars where intensity < `threshold` (vertical layering) |
| `style` | — | one of the 35 styles: sets scale, progressions, groove, patterns, swing, meter |
| `meter`, `grid` | 4, 0.25 | beats per bar, beats per step (1/3 = triplet grid; `meter` 2 + `grid` 1/3 = 6/8) |
| `density` | 0.5 | 0 sparse .. 1 busy |
| `range` | by role | `"C4-C6"`; melody C4–C6, bass E1–G3, chords and pad C3–C5, counter G3–G5 |
| `seed` | 0 | the take |
| `swing` | style's | delays every other grid step by this fraction of a step (as in `pattern`); `swing_grid` picks the step |
| `humanize` | 0 | 0..1, or `{"time": ms, "vel": 0..1}`: onset deviations that are half 1/f, half white (long-range drift over phrases, hit-to-hit anticorrelation, as measured on real drummers), 1 → 0.98 % of a beat (6 ms at 96 bpm); 1/f velocity spread (1 → 13 %). Notes of a chord move together. `{"model": "arma"}`: timing deviations from the ARIMA(4,0,1) fitted to a drummer's 16th-note hi-hat |
| `drift` | 0 | tempo drift ±fraction (0.04 = ±4 %) in slow ~20 s waves; the total length does not change |
| `drift_trend` | 0 | monotone tempo change: the end is (1 + p) × faster (negative = slower, > -1); length kept |
| `swing` | | also `"auto"`: the swing ratio measured on jazz drummers for this tempo |
| `ritard` | — | final ritardando over that many beats (default one bar), or `{"beats", "v_end", "q"}` (q 3, v_end 0.3); `{"beats", "model": "kth"}` uses Friberg's 1991 final-ritard curve instead |
| `offset_ms` | 0 | this part plays that many ms late; `arrangement` `melody_lead` (true = 20 ms, or a number) delays every part but the lead; `melody_lead` `"jazz"` (or true with a `jazz*` style) makes the soloist play the beats late by 107.2 − 0.306 × bpm ms, the off-beats on time |
| `sync` | — | with `humanize`: onsets that correct each other like a player keeping time (`{"alpha", "beta"}`) |
| `expression` | — | scales every timing deviation from the score: 0 deadpan, 1 as performed |
| `off` | — | `{"time": beats, "steps": semitones, "vel"}` (defaults 0.25, 0, 0.8): adds a shifted, transposed copy of every event |
| `echo` | — | `{"n", "time": beats, "decay"}` (defaults 2, 0.5, 0.5): adds n delayed repeats, each `decay` times softer |
| `min_ioi` | 0 | ms: per drum (or for the pitched part), drops onsets closer than that to the last kept one |
| `phase` | — | Reich phasing: each repetition of `cycle` beats lands `shift` beats later (`{"cycle", "shift"}` or a cycle) |
| `high_loud`, `high_sharp`, `double_duration`, `duration_contrast`, `faster_uphill`, `repetition`, `inegales`, `leap_duration`, `melodic_charge`, `chromatic_charge`, `leap_articulation`, `harmonic_charge`, `phrase_arch` | 0 | KTH performance rules, each a strength k (0 = off). `leap_duration` (Friberg's thesis erratum): before an upward leap the note shortens by 4.2 √ΔN·k ms and the landing note lengthens by as much; before a downward leap the note lengthens by 2.4 √ΔN·k ms and the landing note shortens by 4.2 √ΔN·k ms (ΔN in semitones); `melodic_charge` also smooths the level of a neighbouring step (0.75 / 0.55 of the charged note); `leap_articulation` puts a micropause after a leap of 3 semitones or more; `harmonic_charge` swells and stretches toward chords far from the key; `phrase_arch` slows and softens each phrase's ends around a peak at `phrase_turn` (0.7 of the phrase); `duration_contrast` only touches notes of 30-600 ms |
| `melodic_intonation`, `harmonic_intonation` | 0 | KTH intonation, strength k (0 = off). `melodic_intonation` tunes the top voice by its interval to the chord root (cents: 0 −7 3 −4 4 2 10 1 −6 4 −4 9, leading notes up instead); `harmonic_intonation` tunes every note toward just intonation (0 5 4 16 −14 −2 −10 2 14 −16 18 −12 cents) |
| `phrase_lengthen` | 0 | k: the last 3 notes of each phrase lengthen, up to +20 %·k on the last |
| `feel` | — | `soloist`: plays the beats late (off-beats on time) by `downbeat_delay_ms`, default `ellis_delay(bpm)` (40 ms at 90 bpm → 63 ms at 210); `bass`: +10 ms; `ride`: off-beat eighths placed at r/(1+r) of the beat, r = `ride_ratio(bpm)` (2.4 at 132 bpm, straight at 300, held within 1..3.5) |
| `emotion` | — | `angry`, `happy`, `sad`, `tender`: a palette of KTH rules plus tempo (×1.15 or ×0.85), level (±3 dB) and articulation (0.75 or 1.0); explicit keys override the palette |
| `nma` | — | negative mean asynchrony: `true` / `"musician"` (hits 30 ms early), `"novice"` (50 ms), or ms |
| `ensemble` | false | true: per-note jitter N(0, σ), σ = `ensemble_sigma_ms(ioi)` (3 % of the IOI, held within 30-50 ms) |
| `accent_dip` | — | drums: the hit just before an accent drops 2 dB (`true`, or dB) |
| `rmd`, `rmd_unit` | —, 0.5 | Repp's power law on the beat lengths (`"auto"` from the tempo); long notes are cut in `rmd_unit` beats first |
| `accent` | — | drums: accent level as a ratio of a normal hit (ghosts keep theirs) |

Notes may carry cents: `C4+12c`. A section at intensity 0 is silent.

Per role:

- **melody** `mode`: `walk` (default: constrained random walk — mostly steps, chord tones on beats 1 and 3,
  an arch per phrase, each phrase ends on a chord tone and rests, the piece ends on the root),
  `markov` (order 1–2 chain learnt from `motif`), `voss` (1/f pitch sequence; `gamma` sets 1/f^γ: 0 white, 1 pink, 2 brown), `collatz` (hailstone numbers
  folded into the range; `start`), `motif` (a motif developed phrase by phrase; `develop`:
  `state|transpose|invert|retrograde|augment|diminish`), `call_response`.
  `motif`: `"D4 F4 G4"`, scale steps `[0, 2, 3]`, or `[[step, beats], ...]`. `phrase` (bars, 2),
  `contour` `arch|rise|fall|flat`, `snap` (chord tones on strong beats), `p_near`, `p_chord`, `gate`.
  Also: `markov` takes `order` 0–4; `montecarlo` (`min_step`, `max_step`), `ca` (elementary cellular automaton:
  `rule` 90, `cells`, `init`), `lsystem` (`grammar`), `koenig` (`select`, `weights`, `group`), `tendency` (masks
  `mask_lo`, `mask_hi`), `logistic` (`r` 3.9, `x0`), `row` (a twelve-tone `row` and its matrix), `ga` (rhythm
  evolved by a genetic algorithm toward `fit`, a list of `[divisor, multiple, sounding]` durations, beats =
  multiple / divisor; the best rhythm of each of the `generations` (25) is played in turn).
- **bass** `pattern`: `root_fifth`, `walking` (root on beat 1, approach tone on the last beat; `approach` =
  probability that it is chromatic), `octave`, `eighths`, `sixteenths`, `offbeat`, `long`, `tumbao`
  (anticipates the next chord), `syncopated` (funk: Euclidean mask + ghost notes), `acid` (303 line with
  slides and accents).
- **chords** `pattern`: `sustain`, `half`, `backbeat`, `offbeat`, `eighths`, `synco`, `montuno`, `gallop`,
  `stabs` (sparse jazz comping), `broken`. Each chord takes the inversion nearest the previous one.
- **arp** `mode`: `up`, `down`, `updown`, `random`, `pattern` (+ `pattern`: `[0, 2, 1, 2]`), `pool`
  (Farnell: uniform over 11 chord notes, one draw in twelve is a rest); `rate` (beats), `octaves`, `gate`.
- **drums** `kit`: `{"kick": "x---x---x---x---", "snare": "----x-------x---", "djembe:stroke=slap": "..."}`
  (`x` hit, `X` accent, `o` ghost, `?` hit with probability ½; a pattern longer than a bar runs across
  bars) or a style name; `fill_every` (bars, 4; 0 = none); `kit_params`: `{"snare": {"decay": 0.16}}`.
- **pad** holds each chord; **counter** plays slow guide tones entering a beat late.
- Any pitched role with `"mode": "slots"`: Farnell's sequencer — a step `mask` whose hits read fixed
  `slots` (1–8) of the current chord list, changing slot every `flip` steps; `gate_steps`
  `{"3": 0.5}`, `rest_p`, `sub_octave_p`.

Every pitch belongs to the scale. The one exception is the walking bass's chromatic approach tone, which
is marked `"approach": true` in the events.

### Events and export

```python
from genny import compose
events = compose.generate(layer, bpm=104)   # [{"t": beats, "dur": beats, "pitch": MIDI, "vel": 0..1}, ...]
editable = compose.to_spec(layer, bpm=104)  # {"type": "group", "layers": [{"type": "seq", ...}, {"type": "pattern", ...}]}
```

`pitch` is a MIDI number (fractional in a non-12-TET tuning; `None` for drums, which carry `"drum"`).
Acid notes add `slide`, `accent` and `tuple303` (note, steps, slide, level). `to_spec` writes notes as
`seq` steps and drum hits as `pattern` layers (one per drum and velocity level) and keeps the layer's
`at` / `gain` / `fx`. It renders to exactly the same samples as the generator layer, so you can paste it
into a spec and edit single notes. `bpm` only matters for `humanize` (milliseconds) and `drift`.

## `arrangement` — a whole piece

```jsonc
{"bpm": 100, "beats": 64, "loop": "fold",
 "layers": [{"type": "arrangement", "style": "synthwave", "key": "A", "bars": 16, "seed": 3,
             "sections": "A A B A", "intensity": [0.4, 0.7, 1.0, 0.8]}]}
```

Takes `style` (required), and the shared keys of `compose`: `key`, `scale`, `tuning`, `bars` (8),
`sections`, `progression`, `chords`, `harmonic_rhythm`, `seed`, `meter`, `grid`, `swing`, `humanize`,
`drift`, `intensity`, `density`, `chord_size`. All parts get the same harmony.

- **Parts** come from the style: `drums`, `bass`, `chords`, `pad`, `lead`, `arp`, `counter` (those the
  style has). `"parts": {"lead": {"inst": "sax", "mode": "markov"}, "arp": false, "counter": {"inst": "flute"}}`
  overrides, removes or adds one.
- **Tempo**: the spec's `bpm` if there is one, else the layer's own `"bpm"`, else a seeded pick in the
  style's range (`compose.style_bpm(style, seed)`). Length = `bars × meter` beats.
- **Mix**: gains are the "Mix by gain" values of AGENTS.md (drums 1.0, bass 0.85, chords 0.45, pad 0.3,
  lead 0.7, arp 0.35, counter 0.45). `"automix": true` (default) pans the parts automatically
  (`width` 0.6 scales the spread; 1 = the published rule). `"master": false` drops the built-in
  small reverb + compressor.
- **Intensity** thresholds: chords and pad always play, bass from 0.2, drums 0.4, lead 0.5, arp 0.7,
  counter 0.85. `intro` / `outro` sections default to 0.35.
- **Length**: by default the ring-out is kept. `"tail": 0.5` keeps that many seconds after the last beat
  (0 = cut on the beat). `"fold": true` lays the ring-out back over the start: an exact, seamless loop
  of `bars × meter × 60 / bpm` seconds (it does not end on silence, by design).
- The output is scaled down by one gain if it would exceed full scale.

Styles: `lofi jazz_swing bossa_nova blues_shuffle soul_ballad funk disco reggae latin_salsa reggaeton
rock metal pop synthwave house acid_techno drum_and_bass trap chiptune orchestral_heroic baroque ragtime
lullaby spooky christmas celtic_jig bluegrass polka_oompah tango andean middle_eastern indian_raga
japanese gamelan ambient` (the palette of AGENTS.md "Music").

`compose.arrange(layer, bpm)` returns the plan (`bpm`, `beats`, the `compose` parts, master fx);
`compose.to_spec(arrangement_layer, bpm)` expands it all the way to `seq` / `pattern` layers with the pans.

### `compose.automix(layers, sr, bpm, width)`

Automatic panner of Perez-Gonzalez & Reiss (DAFx-07). It needs each source on its own, so it is a
function and not an fx. For every layer it finds the dominant octave band (100 ms frames, frames 60 dB
under the peak ignored, most votes wins). Sources whose band ends under 200 Hz stay in the centre.
Sources sharing a band are spread over equidistant positions, the first in the list nearest the centre,
then alternating sides. It returns the layers with a constant-power `pan` fx added. Gains are not
touched: the papers give no numbers for automatic gain or unmasking EQ.

## `euclid`

```jsonc
{"type": "euclid", "kind": "hihat", "k": 5, "n": 16, "rotate": 2, "bars": 4, "step": 0.25, "accent": "X--x", "swing": 0.15}
{"type": "euclid", "inst": "ks_string", "note": "E2", "k": 3, "n": 8, "step": 0.5}
```

`k` hits spread as evenly as possible over `n` steps (E(3,8) = `x--x--x-`, E(5,8) = `x-xx-xx-`),
`rotate` steps later, repeated `bars` times; `step` in beats. `accent` cycles a level string over the
steps; `prob` drops hits at random (`seed`); `params` go to the drum. With `inst` + `note` it plays a pitch.
`swing`, `humanize` as in `compose`. `compose.CLAVES` holds son / rumba / bossa claves, tresillo,
cinquillo and the 6/8 bell as step strings for `pattern` layers.

## `scatter`

```jsonc
{"type": "scatter", "dur": 20, "interval": [0.5, 1.0], "seed": 4,
 "pitch": [-3, 3], "gain_db": [-12, 0], "pan": 0.8, "reseed": true,
 "layers": [{"type": "sfx", "kind": "bubble"}, {"type": "drum", "kind": "woodblock"}]}
```

Each event picks one child at random. Placement: `interval` `[min, max]` between onsets (the FMOD
Scatterer; its default window is 0.5–1 s), or `rate` (events per time unit, random Poisson arrivals),
or `count` (that many, anywhere in `dur`). Jitter windows per event: `pitch` (semitones, by resampling:
the event gets shorter or longer), `gain_db`, `pan` (−1..1), `time_jitter`; a single number means ±.
`silence_p` leaves that share of slots empty. `reseed` gives every event a new `seed` when the child sfx
or drum has one.

## `arp`

```jsonc
{"type": "arp", "inst": "harp", "chords": "C4:maj7 A3:min F3:maj G3:dom7", "mode": "updown", "rate": 0.25, "beats": 4, "octaves": 2, "gate": 0.8}
```

`chords`: a string of named chords or a list of note lists; `beats` per chord; `mode` `up|down|updown|random|pattern`.

## `stinger`

```jsonc
{"type": "stinger", "mood": "victory", "key": "D", "length": 5, "seed": 1, "inst": "trumpet", "fx": [{"type": "reverb", "mix": 0.25}]}
```

| mood | shape | default voice |
|---|---|---|
| `victory` | up the major triad to a held octave chord | `brass` |
| `defeat` | minor line stepping down, then a falling minor third to the tonic | `piano` |
| `mystery` | wandering whole tones, unresolved augmented chord | `vibraphone` |
| `danger` | root against the flat second, tritone cluster, low | `brass` |
| `pickup` | quick rising pentatonic figure, high | `marimba` |
| `unlock` | lydian turn opening to the octave | `bell` |

`length` = number of notes (the last one is the held note or chord), `step` (0.25 beat with a tempo,
about 0.11 s without), `hold` (final length in steps, 4), `octave` (shift), `scale`, `seed` (variant).

## `adaptive` — states, stems, stingers

```jsonc
{"type": "adaptive", "style": "orchestral_heroic", "key": "D", "bars": 8, "seed": 5, "out_stems": "music/act1",
 "states": {"explore": {"intensity": [0.3]}, "tension": {"intensity": [0.6]}, "combat": {"intensity": [1.0]}},
 "transitions": [{"from": "explore", "to": "combat", "stinger": "danger"}, {"from": "combat", "to": "explore", "stinger": "victory"}],
 "order": ["explore", "combat", "explore"]}
```

Each state is an `arrangement` built from the shared keys plus its own overrides. States that only
differ in `intensity` are the same score with parts muted (vertical layering); a state with its own
`sections` / `progression` is a different segment (horizontal re-sequencing). Every state is a folded
loop of exact length. With `out_stems` it writes, into that folder: `<state>.wav`, one
`<state>__<part>.wav` per part (dry, before the master fx), `stinger_<from>_to_<to>.wav`, and
`graph.json` (states, loop length in beats, bpm, transitions with `quantize: "bar"`). All files share
one gain, so a quiet state stays quiet. The layer's own audio is a demo: the states in `order`, each
transition stinger laid over its boundary. Switching states at run time is the game engine's job; a
WAV renderer can only hand over the pieces.

Per state, also: `oneshot` (plays once instead of looping; `next` names the state that follows), `condition`
(written to `graph.json` for the engine), `xfade_in` / `xfade_out` (seconds). `tails` on the layer writes each
loop's ring-out as `<state>__tail.wav`; `first: true` also writes `<state>__first.wav`, the loop's first pass without
its tail folded onto the start (the engine plays it on entering the state, then the loop; `graph.json` names it under
`"first"`); a transition's `quantize` is free (`"bar"` by default, `"beat"` to escalate quickly); `graph.json` gives bar `markers` in samples. A transition can carry any
`layer` instead of a mood stinger; stingers are in the key of the state they lead to. Moods `alarm` and `settle`
are chromatic clusters (rising, falling). A transition's `level` (dB, or `[from, to]`) is written to `graph.json` as
`level_db` `{"from", "to", "curve": "linear_db"}`, and its `pause` (beats, `true` = 1) as `pause_s` in seconds.

## Scales and tunings

Added to `genny list scales`: `harmonic_minor`, `melodic_minor`, `locrian`, `hijaz`, `in`, `hirajoshi`,
`pelog`, `slendro` (12-TET approximations), `octatonic`, `octatonic_hw`, `bebop`, `bebop_major`.
Ratio scales (usable in `scale` here, not 12-TET): `ptolemy` (syntonic diatonic 1, 9/8, 5/4, 4/3, 3/2, 5/3,
15/8), `seven_eleven` or `"7-11"` (harmonics 6:7:8:9:11), `slendro_equal` (five equal steps), `ikeda_18`
(18 equal steps), `pythagorean` (twelve stacked fifths). `euclid` takes a Xenakis `sieve` in place of `k`.
Custom: a list starting with 0 is semitones (`[0, 2, 3.5, 7]`), a list starting with 1 or holding
`"a/b"` strings is frequency ratios (`[1, "9/8", "5/4", "3/2"]`), or `{"cents": [...]}`.
`"tuning": "just"` plays any 12-TET scale in 5-limit just intonation. Chords are always built from the
scale itself (every other degree), so a numeral means "the chord on that degree": use the mode that
contains the chord you want (`mixolydian` for a flat VII, `harmonic_minor` for a major V in minor).

## `ks_string` (instrument, E2–E6)

The Jaffe-Smith extended Karplus-Strong string: a one-period noise burst circulating in a delay loop.

| param | default | physical meaning |
|---|---|---|
| `decay` | 3.0 | seconds for the fundamental to fall 60 dB at the bottom of the range (shorter up the neck) |
| `stretch` | 0 | how many times longer the upper partials ring than on the plain string (1 = plain: highs die first; 8 = wiry steel); 0 = chosen automatically so `decay` is met |
| `loss` | 1.0 | loop gain per period, 0.9–1: below 1 every partial dies sooner (palm mute) |
| `pick` | 0.2 | where the string is plucked, as a fraction of its length: 0.5 (middle) removes the even harmonics, small = near the bridge, thin |
| `dyn` | 2000 | Hz: bandwidth of the pluck at full velocity; softer notes are darker. Equalised across pitch, so a run sounds even |
| `bright` | 3500 | Hz: ceiling on the pluck spectrum |
| `sympathetic` | 0 | 0–0.5: drive of an undamped second string that keeps ringing after the note is stopped |
| `symp_interval` | −12 | semitones from the note to that second string |
| `seed` | 0 | the noise burst |

## `ks_drum` (drum)

A noise loop where every recirculated sample keeps its sign with probability `blend`.

| param | default | meaning |
|---|---|---|
| `decay` | 0.25 | seconds to fall 60 dB; it sets the loop length (0.2–0.4 snare, 0.03 brushed tom) |
| `blend` | 0.5 | 0.5 = drum (no pitch), 1 = a plucked string (pitch 22050 Hz / loop length), 0 = an octave lower with odd harmonics only |
| `stretch` | 1 | ≥ 1: longer, more "snare" |
| `tone` | 6000 | Hz lowpass |
| `seed` | 0 | the sign sequence |

Python: `ks_drum_raw(n, p, blend, stretch, seed, loop="avg"|"121"|"siegel", copies=k)`: `121` averages three
samples (Y[t−p−1] + 2Y[t−p] + Y[t−p+1])/4 (pitch fs/p), `siegel` (Y[t−p] + Y[t−1])/2 (softer, nylon-like); `copies` k fills the table with k copies of one noise burst
(k times higher, same decay). Also
`ellis_delay(bpm)`, `ride_ratio(bpm)`, `ensemble_sigma_ms(ioi_ms)`, `gamma_noise(n, gamma, rng)`, `arma_hihat(n, rng)` and `poisson_grains(dur, density, rng, grain=(0.06, 0.12))`
(a stochastic grain cloud with exponential inter-onset times).

## Recipes

A funk rhythm section you can edit afterwards:

```jsonc
{"out": "funk.wav", "bpm": 104, "beats": 32, "loop": "fold", "layers": [
  {"type": "compose", "role": "drums", "style": "funk", "bars": 8, "seed": 2, "humanize": 0.4},
  {"type": "compose", "role": "bass", "style": "funk", "inst": "slap_bass", "key": "E", "bars": 8, "seed": 2, "gain": 0.85},
  {"type": "compose", "role": "chords", "style": "funk", "inst": "clav", "key": "E", "bars": 8, "seed": 2, "gain": 0.45},
  {"type": "compose", "role": "melody", "style": "funk", "inst": "sax", "key": "E", "bars": 8, "seed": 2, "gain": 0.7, "humanize": 0.5}]}
```

A never-resolving bed (Farnell's chords, slot bass and arp pool with a tempo-locked echo of three eighths):

```jsonc
{"bpm": 120, "layers": [
  {"type": "compose", "role": "pad", "inst": "warm_pad", "progression": "pokesdown", "key": "D", "scale": "mixolydian", "bars": 24, "gain": 0.4},
  {"type": "compose", "role": "bass", "mode": "slots", "slots": [1, 2], "mask": "x-----x-?-------", "sub_octave_p": 0.5, "inst": "bass",
   "progression": "pokesdown", "key": "D", "scale": "mixolydian", "bars": 24, "gain": 0.8},
  {"type": "compose", "role": "arp", "mode": "pool", "inst": "synth_pluck", "progression": "pokesdown", "key": "D", "scale": "mixolydian",
   "bars": 24, "gain": 0.35, "fx": [{"type": "delay", "time": 0.75, "feedback": 0.3, "mix": 0.25}]}]}
```

A koto melody in the In scale over a Euclidean wood block, and a just-intoned drone:

```jsonc
{"bpm": 80, "layers": [
  {"type": "compose", "role": "melody", "inst": "koto", "key": "D", "scale": "in", "progression": "i iv i v", "bars": 8, "density": 0.35, "seed": 9},
  {"type": "euclid", "kind": "woodblock", "k": 3, "n": 8, "step": 0.5, "bars": 8, "gain": 0.25},
  {"type": "compose", "role": "pad", "inst": "strings", "key": "D", "scale": "ptolemy", "progression": "I", "bars": 8, "gain": 0.3}]}
```

Debris after an explosion, then a pickup jingle:

```jsonc
{"layers": [
  {"type": "sfx", "kind": "explosion"},
  {"type": "scatter", "at": 0.3, "dur": 2.5, "rate": 9, "seed": 2, "pitch": [-5, 5], "gain_db": [-18, -4], "pan": 0.9, "reseed": true,
   "layers": [{"type": "drum", "kind": "ks_drum", "params": {"decay": 0.06}}, {"type": "sfx", "kind": "click"}]},
  {"type": "stinger", "at": 3.2, "mood": "pickup", "key": "E", "gain": 0.7}]}
```

A steel-string riff on `ks_string` with a sympathetic string:

```jsonc
{"bpm": 96, "layers": [{"type": "compose", "role": "arp", "inst": "ks_string", "key": "E", "scale": "minor_pentatonic",
  "progression": "i VII", "mode": "updown", "rate": 0.5, "range": "E2-E4", "bars": 4,
  "params": {"stretch": 6, "pick": 0.12, "sympathetic": 0.25}}]}
```

## Notes and limits

- Acid `slide` is exported as a flag and played legato; the `acid` instrument has no pitch glide to drive.
- `scatter` renders each event separately (no caching): a few hundred events of a slow sfx take a while.
- Without a tempo, `scatter` children keep seconds; at exactly 60 bpm a generator layer treats its
  children's times as seconds too (the numbers are the same).
- Like `group`, a `scatter` has a `layers` key of its own, so put it inside a spec's `layers` list, not at the top level.
- A generated layer starts and ends as its instruments do; several existing instruments carry a DC
  offset, which this module does not remove.
