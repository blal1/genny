# Sung voice and choir (`genny/choir.py`)

People singing, not a pad. Every singer is rendered on his or her own: one glottal source and one
moving vocal tract, continuously across the whole phrase, with its own tract length, pitch, vibrato,
timing, breathiness and place on the stage. A section is the sum of those singers; nothing is a
chorus effect on one voice. Output is dry and close: add the room yourself (see "Space").

The old catalog instrument `choir` (a supersaw pad) is untouched. For a real choir use what is below.

## What is registered

| type | name | what it is |
|---|---|---|
| layer | `sing` | a phrase with lyrics sung by a soloist or a section; stereo |
| layer | `satb` | four `sing` parts (sopranos, altos, tenors, basses) from a chord progression or four lines |
| instrument | `voice` | solo singer, one note per call (use in `synth` / `seq` layers) |
| instrument | `vocal_choir` | a section of singers on one note |
| instrument | `hum` | closed-mouth humming, solo or section |
| instrument | `throat_singing` | overtone singing: low drone, whistle on one harmonic |
| instrument | `falsetto` | male head voice |
| instrument | `boys_choir` | treble voices |
| instrument | `chant` | unison men, straight tone |

No `choir_space` effect was added: the existing `room` and `reverb` cover it.

**Use `sing` / `satb` for anything melodic.** The instruments render each note separately (the
`seq` contract), so they cannot glide between notes or carry a syllable across them. They are for
held chords, pads of real voices and single stabs.

## `sing` layer

```jsonc
{"type": "sing", "section": "tenors", "singers": 8,
 "steps": "C4:1 D4:1 E4:2", "lyrics": "A-ve Ma-ri-a",
 "legato": true, "dynamics": "mp<f>p", "style": "cathedral", "seed": 1}
```

| key | default | meaning |
|---|---|---|
| `steps` | required | the same step notation as `seq` (`C4:1 -:0.5 [..]`, beats when `bpm` is set). A chord sings its first note |
| `voice` | by pitch | one voice type: `bass` `baritone` `tenor` `countertenor` `alto` `mezzo` `soprano` `treble`. Solo unless `singers` > 1 |
| `section` | none | `sopranos` `altos` `tenors` `basses` `baritones` `mezzos` `boys` `children` `men` `women` `mixed` (men an octave under the women), or `auto` |
| `singers` | 1 with `voice`, 6 with `section` | number of people, 1..48 |
| `lyrics` | `"ah"` | one syllable per note, see "Lyrics" |
| `lang` | `la` | how words are read: `la` Latin (pure vowels), `en`, `es` |
| `legato` | `true` | slur adjacent notes: continuous voicing with a portamento. `false` = re-articulated |
| `articulation` | `normal` | `normal` `legato` `staccato` (half length, separated) `marcato` (pressed accent on every note) |
| `dynamics` | `mf` | `ppp pp p mp mf f ff fff`; several marks are spread evenly over the phrase; `<` `>` between two marks is a hairpin, a space a sudden change: `"mp<f>p"`, `"p ff"`, `"pp<"` |
| `style` | `classical` | see "Styles" |
| `mode` | `sing` | `sing` `whisper` (no voicing: noise through the vowels) `shout` `falsetto` |
| `vibrato` | from style | extent in cents, +- peak (0 = straight tone) |
| `vib_rate` | from style | Hz |
| `breath` | from style | 0 clean .. 1 airy |
| `ring` | from style | singer's formant: 0 untrained .. 1 operatic .. 1.5 |
| `effort` | 1 | multiplies the dynamics (0.5 softer and darker, 1.3 pressed) |
| `bright` | 1 | 0.5 dark .. 2 bright (shelf above 3 kHz) |
| `spread` | 0.6 | stereo width of the section 0..1 |
| `pan` | 0 | centre of the section, -1..1 |
| `breaths` | 0 | 0..1: audible breath intake before each phrase (needs 0.3 s of rest or `lead` before it) |
| `portamento` | 1 | glide time factor between slurred notes |
| `timing` | from style | onset spread between singers in seconds |
| `lead` | 0 | seconds of pre-roll before beat 1 (room for the first consonant and breath) |
| `tail` | 0 | seconds after the last beat. 0 = the singers release inside the last note |
| `overtones`, `sub` | none | throat singing: harmonic numbers spread over the phrase (`"8 9 10 12 10"`), kargyraa depth 0..0.6 |
| `transpose`, `step`, `vel`, `seed` | 0, 0.25, 1, 0 | as in `seq`; `seed` picks the people |

**Length is exact.** The buffer is `lead` + the steps' length + `tail`, to the sample (tested at
60, 90 and 132 bpm). With `tail` 0 the last note is released inside its written length, so a phrase
tiles and loops (`"loop": "fold"`).

**Timing of words.** Consonants are sung before the beat and the vowel starts on it (measured: the
voicing of "ta", "sa", "ka", "pa" starts 2-7 ms after the beat, the noise peaks 26-66 ms before it). The
first note of a layer has nothing before it: start with a rest (`"-:1 ..."`) or give `"lead": 0.15`
and move the layer's `at` earlier by the same time, otherwise its first vowel is late by the length
of its consonants.

### Lyrics

- One syllable per note. Words are split at their vowels: `"Ave Maria"` and `"A-ve Ma-ri-a"` both
  give five syllables. Hyphens only matter where your split differs from the automatic one.
- `ah eh ee oh oo` are plain vowels, `mm` `nn` `ng` closed-mouth hums, `uh` a schwa.
- `_` or `~` holds the previous vowel on the next note (melisma); a final consonant moves to the end.
- One syllable for many notes (`"lyrics": "ah"`) is sung on every note; with `legato` it is one
  unbroken vowel.
- Fewer syllables than notes: the rest is a melisma on the last one.
- `/T AA/` is one syllable of raw phonemes (`genny list phonemes`).
- `lang: "la"` is ecclesiastical Latin: a e i o u pure, `ae`/`oe` = e, `c`/`g` soft before e and i
  (pa-cem = "pa-chem"), `gn` = ny, `ti` + vowel = tsi, `h` silent, `qu` = kw, `x` = ks, `xc` = ksh.
  Use it for any text you want sung with pure vowels. `en` uses genny's speech dictionary and
  letter-to-sound rules (check odd words with `genny say --show`); `es` uses the Spanish rules.

## `satb` layer

```jsonc
{"type": "satb", "chords": "D4:min:2 G3:min:2 A3:maj:2 D4:min:4", "lyrics": "A-men a-men",
 "singers": 8, "dynamics": "p<ff>pp", "style": "cathedral", "lead": 0.4, "tail": 0.8}

{"type": "satb", "soprano": "E5:2 F5:2", "alto": "C5:2 C5:2", "tenor": "G4:2 A4:2", "bass": "C3:2 F3:2",
 "lyrics": "A-men", "singers": 6}
```

`chords` (step notation with chord names, root first) is voiced automatically: the bass takes the
root, the upper voices the nearest chord tones that complete the chord without crossing. Or give
the four lines yourself (`soprano` `alto` `tenor` `bass`, or `"parts": {...}`; you may leave parts
out). `singers` is per section. `"balance": [s, a, t, b]` sets the part gains. All other `sing` keys
apply to every part. The sections stand left to right S T B A.

## Instruments

All take `vel` (soft = breathier and darker, not just quieter) and `seed`.

| name | range | params |
|---|---|---|
| `voice` | C2-C6 | `voice` (auto by pitch), `vowel` (`ah`, `oo`, `mm`, a syllable like `la`), `vibrato` (cents, -1 = style), `vib_rate`, `breath`, `effort`, `ring`, `mode`, `style`, `lang`, `bright` |
| `vocal_choir` | C2-C6 | `section` (auto), `singers` 8, `vowel`, `spread` (1 = normal scatter of pitch and timing, 0 tight, 2 loose), `vibrato`, `breath`, `style`, `lang`, `bright` |
| `hum` | C2-C5 | `singers` 6, `section`, `consonant` m or n, `vibrato`, `breath`, `style` |
| `throat_singing` | C2-C3 | `harmonic` 8 (which partial whistles, 5..14), `harmonic2` (glide to it), `sub` (kargyraa 0..0.6), `singers` |
| `falsetto` | A3-E5 | `voice` tenor, `vowel` oo, `singers`, `vibrato`, `breath`, `style` |
| `boys_choir` | C4-G5 | `singers` 8, `vowel`, `vibrato`, `breath` |
| `chant` | A2-D4 | `singers` 8, `vowel` oh, `vibrato`, `breath` |

They are mono, level-matched (-12 dB K-weighted at vel 0.9) and in tune as an ensemble (measured
worst case 7 cents).

## Styles

Presets of the same parameters; every one can be overridden per layer.

| style | vibrato solo / section | ring | character |
|---|---|---|---|
| `classical` = `cathedral` | +-60 / +-30 cents, 5.7 Hz, starts after 0.3 s | 0.9 | trained voices, blended |
| `gregorian` = `chant` | +-7 cents | 0.45 | straight tone, soft onsets |
| `gospel` | +-75 / +-49 cents | 0.6 | pressed, scooped onsets, open vowels |
| `pop` | +-32 / +-16 cents, late | 0.15 | breathy, speech-like, no singer's formant |
| `children` | +-12 cents | 0.2 | straight, a little airy, looser tuning and timing |
| `epic` = `trailer` | +-45 / +-27 cents | 1.15 | pressed, loud, tight; use Latin syllables and `marcato` |
| `throat` = `overtone` | none | 0.5 | pressed drone |
| `bulgarian` | none | 0.55 | straight, pressed, bright, raised first formant; write close seconds |

Sections get 65 % of the style's `ring`: choir singers hold the singer's formant back.

## Recipes

Solo soprano with a hairpin, in a church:

```json
{"out": "ave.wav", "bpm": 72, "layers": [
  {"type": "sing", "voice": "soprano", "steps": "D5:2 D5:1 E5:1 | F5:2 E5:1 D5:1 | E5:3",
   "lyrics": "A-ve Ma-ri-a gra-ti-a", "dynamics": "mp<f>p", "lead": 0.4, "tail": 0.6, "breaths": 1}],
 "fx": [{"type": "room", "preset": "church", "mix": 0.45}]}
```

Gregorian chant with melismas:

```json
{"out": "kyrie.wav", "bpm": 84, "layers": [
  {"type": "sing", "section": "men", "singers": 9, "style": "gregorian",
   "steps": "D3:1 F3:.5 G3:.5 A3:1.5 G3:.5 A3:1 C4:.5 A3:.5 G3:1 F3:.5 G3:.5 A3:2",
   "lyrics": "Ky-ri-e _ _ e _ _ le _ i-son", "lead": 0.4, "tail": 0.6, "breaths": 0.8}],
 "fx": [{"type": "room", "preset": "church", "mix": 0.45}]}
```

Epic trailer choir:

```json
{"out": "dies_irae.wav", "bpm": 96, "layers": [
  {"type": "satb", "chords": "D4:min:1 D4:min:1 Bb3:maj:2 A3:maj:1 D4:min:3",
   "lyrics": "Di-es i-rae di-es", "singers": 10, "style": "epic", "articulation": "marcato",
   "legato": false, "dynamics": "f<fff", "spread": 0.9, "lead": 0.3, "tail": 0.8}],
 "fx": [{"type": "reverb", "mix": 0.32, "size": 0.92, "damp": 0.45}]}
```

Pop backing vocals, and a hummed pad from the instrument:

```json
{"out": "oohs.wav", "bpm": 92, "layers": [
  {"type": "satb", "chords": "C4:maj7:4 A3:min7:4 F3:maj7:4 G3:dom7:4", "lyrics": "oo oo ah ah",
   "singers": 4, "style": "pop", "dynamics": "mp", "lead": 0.3, "tail": 0.6},
  {"type": "seq", "inst": "hum", "steps": "[C3,G3]:8 [F2,C3]:8", "params": {"singers": 4}, "gain": 0.4}],
 "fx": [{"type": "reverb", "mix": 0.18, "size": 0.6}]}
```

Throat singing with an overtone melody:

```json
{"out": "khoomei.wav", "bpm": 60, "layers": [
  {"type": "sing", "voice": "bass", "steps": "C2:12", "overtones": "8 9 10 12 10 9 8 6 8 10 12",
   "sub": 0.25, "lead": 0.2, "tail": 0.5}]}
```

Eighteen more complete specs, with their renders, are in `sounds/choir_audition/specs.json`.

## Space

The voices are dry. A choir needs a room:

- cathedral, chant: `{"type": "room", "preset": "church", "mix": 0.4-0.5}`
- concert hall: `{"type": "room", "preset": "hall", "mix": 0.35}` or `{"type": "reverb", "mix": 0.22, "size": 0.75, "damp": 0.5}`
- epic: `{"type": "reverb", "mix": 0.3, "size": 0.92, "damp": 0.45}`
- pop, close: `{"type": "reverb", "mix": 0.1-0.18, "size": 0.5}`

Give the layer `lead` and `tail` so breaths, first consonants and the release have room.

## What the main parameters are, physically

- **effort / dynamics / vel** - how hard the vocal folds close. It sets the shape of the glottal
  pulse (Rd: 0.5 pressed, 0.8 trained mf, above 2 breathy) and the spectral tilt of the source, so
  loud is brighter, not only louder. Measured pp to ff on a tenor: +22.5 dB, and at equal level the
  2-4 kHz band rises 10 dB and the centroid from 536 to 676 Hz.
- **breath** - how much the folds leak: more noise between the harmonics, a steeper tilt, a wider
  first formant. The noise pulses with the open phase of each cycle.
- **ring** - the singer's formant. A trained man lowers F4 and F5 onto F3, which makes a strong
  peak at 2.4-3.3 kHz that carries over an orchestra. 0 puts F4 and F5 where an untrained tract has
  them. Measured on a bass, the 2.4-3.4 kHz band goes -48, -40, -24 dB at ring 0, 0.5, 1. Women's
  tables have no such cluster, so `ring` does little for them.
- **vibrato** - extent in cents, +- peak. It starts after a delay and fades in. Measured: asked
  5.0 Hz +-40 cents and 6.2 Hz +-80 cents, got 5.00 Hz +-40.2 and 6.19 Hz +-80.2.
- **voice type** - the published formant row (F1-F5 and bandwidths for a e i o u) and the tract
  length that goes with it. Above the top of his full voice (bass E4, baritone G4, tenor C5) a man
  goes into falsetto by himself.
- **formant tuning** - above about G4 a soprano's pitch passes the first formant of the vowel; she
  opens the jaw so F1 follows the pitch. Without it the note is thin: measured on /i/ at 700 Hz,
  the fundamental is 36 dB over the third harmonic with tuning and 0.2 dB without.
- **singers / spread** - more people = more beating and a wider line. Measured: the envelope of
  the third harmonic varies 1.6 % for a soloist and 47 % for eight singers; no two singers correlate
  above 0.32; the mean pitch of the section stays within 6 cents.

## Model and sources

| element | source |
|---|---|
| Glottal pulse: Liljencrants-Fant flow derivative, one parameter Rd | Fant, Liljencrants & Lin 1985; Fant 1995; coefficients `genny.physical.voice._lf_coefs` (Pink Trombone, research 08 §5.4) |
| Band-limited pulse tables normalised on the first harmonic; kernel layout | ported from the sibling `procedural/choir.py` (Klang Open Licence) |
| Spectral tilt with effort (TL, dB down at 3 kHz) | Klatt 1980; Klatt & Klatt 1990 |
| Breath noise through the tract, gated by the glottal flow | Cook, *Real Sound Synthesis* p.67-69 (research 04 §9.1); Klatt & Klatt 1990 |
| Jitter, shimmer; flutter 12.7 + 7.1 + 4.7 Hz | Baken & Orlikoff 2000; Klatt & Klatt 1990 |
| Vibrato delayed and faded in, 5-6.5 Hz | research 12 §8; Prame 1994, 1997; STK SingWave / Modulate (research 07 §8) |
| Tract: cascade of two-pole resonators, unity gain at DC | Klatt 1980; Cook Fig 8.4; JOS (research g14 item 4: the tract is all-pole in series) |
| F1-F5 and bandwidths per vowel and voice type | Csound manual table = Faust `physmodels.lib formantValues` = `genny.physical.voice.formant_table` (research 08, 11 C8) |
| Poles above F5: the neutral tube's (2k-1)c/4L | Fant 1960 |
| Low formants broadened by the yielding wall | Bilbao, *Numerical Sound Synthesis* §9.2 (research g02 §16) |
| Formant tuning (F1 follows f0, F2 bent) | Faust `autobendFreq` = `genny.physical.voice._autobend`; Sundberg 1987; Joliveau, Smith & Wolfe 2004 |
| Singer's formant, weaker in choirs | Sundberg 1974; Rossing, Sundberg & Ternstrom 1986 |
| Nasals: pole-zero pair mixed in by a nasality weight; hum formants | Klatt 1980 via `genny.speech`; STK `Phonemes.cpp` mmm / nnn (research 07 §8) |
| Phonemes, loci, coarticulation, frication and burst spectra and levels; English and Spanish text | `genny.speech` (reused, not rewritten) |
| Latin pronunciation | Liber Usualis rules (small rule set in `choir.py`) |
| Ensemble: tract length, pitch scatter, sqrt(N) sum | Fitch & Giedd 1999; Ternstrom & Sundberg 1988; Farnell (research 01 §7.5) |

Baritone, mezzo and treble rows are blends or scalings of the five published rows, and the style
table, the effort-to-Rd and effort-to-tilt slopes, the blend weights for English vowels and the
level constants are mine: each is marked `# UNSOURCED` in the code with the reason.

Three things differ from the sibling project's choir, each because the measurement said so:

- Its breath-noise level (0.33) measures -3 dB harmonics-to-noise at 1-4 kHz on this kernel, a
  hoarse whisper. It is 0.015 here: 18 dB at mf, 6-8 dB at pp, 21-24 dB at ff.
- Its +12..28 dB correction filter above 5 kHz put 6 % of an alto's energy above 12 kHz. It is not
  ported. The voices are therefore dark above 5 kHz (under 0.1 % of the energy); `bright` opens them.
- The cascade runs from the highest pole to the lowest. In the usual order a gliding formant came
  out as a burst of noise three times the level of the voice, because the fixed upper poles amplify
  every small step of the moving ones.

## Limits

- It is a formant synthesizer. Vowels, hums and legato lines are its strength; fast consonant-heavy
  English is its weakness, as in `genny say`. Latin, Italian and Spanish text works best.
- A soloist is clearly a synthetic singer; a section of six or more hides most of that.
- `satb` chord voicing is a nearest-tone search, not counterpoint: write the four lines yourself
  when the voice leading matters.
- Cost: about 0.02 s per singer per second of sound.
