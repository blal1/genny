# Spectral effects and analysis → resynthesis (`genny/spectral.py`)

STFT-domain processing after J. O. Smith, *Spectral Audio Signal Processing* (findings:
`out/research/g12-jos-sasp.md`). Everything works on mono or stereo and at any sample rate
(window lengths are given in milliseconds and rounded to a power of two).

- 14 effects, usable in any `"fx"` chain or as `genny fx in.wav --fx NAME:k=v,...`
- 1 layer type, `resynth`: analyse a WAV file into sines + noise and render it transformed
- a Python API (`stft`, `istft`, `stretch`, `shift_pitch`, `analyze`, `resynth`, envelopes)

## Which one do I want?

| I want to… | use |
|---|---|
| make a sound longer/shorter at the same pitch | `timestretch` (`mode: sola` for speech, drums, single voices) |
| change the pitch at the same length | `pitch_shift` (`formant: keep` for voices and instruments that must not turn chipmunk/giant) |
| add a third and a fifth above a voice | `harmonizer` |
| a robot / talking-synth voice | `vocoder` (pitched carrier), `robotize` (monotone), `whisperize` (no pitch) |
| make wind, rain or an engine "speak" | `cross_synth` with `carrier: path.wav` |
| remove hiss, or split pitched from noisy content | `spectral_gate`, `sines_only`, `noise_only` |
| sustain one instant of a sound | `freeze` |
| turn attacks into a wash | `spectral_blur` |
| metallic / inharmonic detune, slow barber-pole phasing | `freqshift` |
| darker or brighter overall, no resonance | `spectral_tilt` |
| stretch ×8, transpose, or re-balance tone vs noise of a recording | layer `resynth` |

Added since 1.2.1 (parameters: `catalog.md`):

| I want to… | use |
|---|---|
| keep only the tones, or only the hits, of a sound | `hpss` (median filtering) |
| snap a voice to the nearest semitone | `autotune` |
| turn a male voice female or the reverse | `gender` (pitch and formants together) |
| mix two sounds' spectra (magnitudes multiplied, or phases of one on the other) | `mutate` |
| stretch a mix with both tones and hits | `timestretch mode=hps` (vocoder on the tones, overlap-add on the hits) |
| land moments of a sound on an animation | `timestretch anchors="t_in:t_out, ..."` |
| finer, noisier cross-synthesis | `cross_synth method=stamp` |
| denoise without musical noise, or from a noise sample | `spectral_gate mode=soft` / `mode=subtract` |

`freqshift fold=false` (the default) now removes what would cross Nyquist or 0 Hz; `fold=true` keeps the mirrored
ring-modulator artefact.

## Effects

Length: `timestretch` returns `factor ×` the input length and `freeze` adds `length` seconds;
every other effect keeps the length, except `granular` (longer by the buffer delay plus one grain). Channel count is always kept, except `granular`, which returns stereo when `spread` > 0.

### `timestretch`
| param | default | meaning |
|---|---|---|
| `factor` | 1.0 | length ratio, 0.1–10. 2 = twice as long, same pitch |
| `mode` | `vocoder` | `vocoder`: phase vocoder, best for music, chords, pads, ambience. `sola`: time-domain overlap-add with a similarity search, best for speech, drums and single voices (no "phasiness", but chords may flutter). `ola`: sola without the similarity search. `hps`: vocoder on the tones, OLA on the hits. `psola`: pitch-synchronous overlap-add, cuts two-period grains at each pitch period (best for a single pulsed voice: no phasiness, formants kept). `auto`: psola where the voice pulses are sharp (epoch sharpness ≥ 0.9 over 6 frames of 10 ms), vocoder elsewhere, 20 ms crossfades. `anchors` need vocoder, sola, ola or hps |
| `window_ms` | 0 | analysis window; 0 = auto (45 ms vocoder, 50 ms sola). Longer resolves closer/lower partials but smears more |
| `lock` | true | vocoder: keep the phase relations around each spectral peak (less reverberant "phasiness") |
| `transients` | true | attacks are moved, not stretched (a click stays a click). A number is the detector sensitivity (2 = also softer attacks); false stretches everything |

### `pitch_shift`
| param | default | meaning |
|---|---|---|
| `semitones` | 0 | −36…+36 |
| `formant` | `shift` | `shift`: the whole spectrum moves, like tape speed (resonances move too: smaller/larger body). `keep`: the spectral envelope (vowel, body resonances) stays where it was, only the pitch moves |
| `fine` | 0 | additional cents |
| `mode` | `vocoder` | `vocoder`; `psola`: pitch-synchronous grains, the formants stay in place whatever `formant` says; `auto`: psola on sharp voice pulses, vocoder elsewhere |

The older granular `pitch` effect is untouched; `pitch_shift` is the clean one for tonal material.

### `harmonizer`
| param | default | meaning |
|---|---|---|
| `intervals` | `"4 7"` | semitone offsets of the added voices: a string (`"4 7"`, `"-12"`, `"3;7;10"` on the command line) or a list `[4, 7]` |
| `mix` | 0.5 | 0 = dry, 1 = only the added voices |
| `formant` | `keep` | as in `pitch_shift` |

### `vocoder`
The layer is the **modulator** (speech, a drum loop, anything with a moving spectrum); its energy in
each frequency band opens the same band of a synthetic **carrier**. Physically: a bank of band-pass
filters with envelope followers (Dudley's channel vocoder).

| param | default | meaning |
|---|---|---|
| `carrier` | `saw` | `saw` (buzz), `pulse` (thin buzz), `noise` (whisper), `chord` (saws on `notes`) |
| `freq` | 110 | carrier pitch, Hz or note name — this is the pitch you hear |
| `notes` | `""` | chord for the carrier, e.g. `"C3 G3 C4 E4"`; replaces `freq` |
| `bands` | 10 | 10 = the classic Voder bands (0-225-450-700-1000-1400-2000-2700-3800-5400-7500 Hz). 16–32 = clearer speech (ERB-spaced, 60 Hz–12 kHz) |
| `attack` | 0.005 | band envelope rise time, s |
| `release` | 0.03 | band envelope fall time, s; longer = smoother and more "synthetic", shorter = more articulate |
| `formant` | 0 | semitones the band pattern is moved up on the carrier (+ = smaller speaker, − = giant) |
| `hiss` | 0.05 | noise mixed into the carrier so consonants (s, t, f) have something to shape |
| `seed` | 0 | noise seed |

### `cross_synth`
The layer's spectral envelope (its "vowels") is imposed on a second sound.
| param | default | meaning |
|---|---|---|
| `carrier` | none | path of a WAV file; it is resampled, made mono and looped to the layer's length. Without it, white noise (a whisper) |
| `flatten` | true | remove the carrier's own envelope first (stronger effect) |
| `smooth` | 2.0 | envelope detail as a cepstral cut-off in ms; keep it below the pitch period of both sounds (2 ms ↔ features ≥ 500 Hz wide) |
| `mix` | 1.0 | wet amount |
| `seed` | 0 | noise seed when there is no carrier file |

### `spectral_gate`
| param | default | meaning |
|---|---|---|
| `threshold` | −40 | dB relative to the loudest spectral bin of the sound; quieter bins are turned down |
| `reduction` | 30 | how much, in dB |
| `smoothing` | 1.0 | 0–4: averages the gain over neighbouring frames and bins. 0 = hard switching ("musical noise"); higher = smoother but softens attacks |
| `keep` | `all` | `all`: plain gate. `tonal`: only spectral peaks that stand ≥ 12 dB above their surroundings and above the threshold (removes hiss, breath, room). `noise`: the opposite — the peaks are removed |

### `sines_only`, `noise_only`
The sound is analysed into sinusoidal partials plus a noise residual (see "The model") and only one
part is resynthesised. `thresh` (−50): how far below a frame's loudest partial a peak may be and still
count as a partial. `noise_only` also takes `seed`.

### `freeze`
| param | default | meaning |
|---|---|---|
| `at` | 0.2 | the instant that is held, seconds |
| `length` | 2.0 | seconds of frozen sound inserted there; the sound then continues |
| `jitter` | 20 | ms of random wander of the read position: 0 = perfectly static (glassy, a little buzzy), 20–60 = alive |
| `seed` | 0 | jitter seed |

### `spectral_blur`
`time` (0.08 s): Gaussian smoothing of every bin's level over time — attacks soften into a swell and
sounds ring on. `freq` (0 Hz): smoothing across frequency — partials widen into noise bands. `seed`.

### `freqshift`
`hz` (100): every component moves by the same number of Hz (single-sideband modulation), so
harmonic sounds become inharmonic: 440/880/1320 → 540/980/1420. ±0.1–2 Hz gives slow beating
against the dry signal; ±20–300 Hz is metallic; negative shifts go down. `hz_end`: the shift glides
linearly from `hz` to it over the sound (barber-pole sweeps).

### `robotize`
`pitch` (110, Hz or note): each analysis frame is turned into a symmetric pulse, so the output has
the constant pitch `sr / round(sr / pitch)` whatever the input did; vowels stay.

### `whisperize`
`window_ms` (12): frame length; the phase of each frame is randomised, which destroys the pitch and
keeps the formants. Longer frames (30+) let some pitch through. `seed`.

### `spectral_tilt`
`slope` (−3 dB/octave), `pivot` (1000 Hz): a straight-line change of the spectrum in dB per octave
(limited to ±24 dB), linear phase, level-compensated. −3 turns white noise pink; +3 adds air.

### `spectacle`
Spectral delay: the spectrum is cut into `bands` (geometric, `low` to `high` Hz), each delayed by its own time with
its own feedback: rising, falling or scattered echoes of different colours.

| param | default | meaning |
|---|---|---|
| `delay` | 0.5 | s, longest band delay |
| `table` | `random` | `random`, `ascending`, `descending`: which bands wait longest |
| `feedback` | 0.8 | 0..0.95 per band |
| `bands` | 64 | number of bands |
| `low`, `high` | 100, 4100 | Hz |
| `mix`, `seed` | 0.5, 0 | |

### `granular`
Granular cloud read from a 6 s ring buffer the sound is written into (96 grains at most). Each grain reads
`position` + random `spray` back into the buffer, at `pitch`, panned at random within `spread`. `freeze` stops
recording after that many seconds, so the grains keep scanning the held moment; `feedback` writes the output back
into the buffer (DC-blocked, 0.95 at most) for swelling, self-feeding clouds. Output is stereo when `spread` > 0,
and longer than the input by the buffer delay plus one grain.

| param | default | meaning |
|---|---|---|
| `density` | 20 | grains per second, 2-200 |
| `size` | 0.08 | grain length in s, 0.005-1 |
| `spray` | 0.2 | random reach back into the buffer, 0-1 |
| `position` | 0.05 | how far back grains read, fraction of the 6 s buffer |
| `spread` | 0.5 | stereo spread, 0 = mono output |
| `pitch` | 0 | grain transposition in semitones |
| `feedback` | 0 | 0-0.95 |
| `freeze` | none | seconds after which the buffer stops recording |
| `window` | `hann` | `hann`, `tukey` (flat with 15 % edges), `perc` (exp(−5t)(1−t): plucked grains) |
| `overlap` | `normal` | `normal`: sum scaled by 1/sqrt(density × size); `sola`: each grain fades in over the life left in the previous one and out the same way, linear ramps that sum to one (no level pumping) |
| `mix` | 1 | 0 = dry, 1 = grains only |
| `seed` | 0 | |

### `granulate`
Brassage: the sound is rebuilt, at the same length, from Hann grains of `grain` s (0.08; 0.001-0.1) drawn within
`window` s (0.1) of the current time (0 = anywhere in the sound: a shuffled texture), `overlap` grains per grain
length (2 = half-grain overlap) and a random transposition of `pitch_var` octaves (sd, 0.02) per grain; `seed`.
`pitch_st` (0: ± semitones per grain, uniform, replaces `pitch_var`), `positions` (with `window` 0: `random`, `gauss`,
`walk`, `zigzag`), `match` (0: concatenative, candidates per grain, the one that best continues the previous grain
wins), `min_back` (s: no backward jump shorter than this). Python `physical.analysis.cloud(..., pitch_st=,
positions=, match=, min_back=)` is the engine behind it.

### `fog`
FOF grains read from the sound as a sample table (granular formant synthesis on a recording): `dens` grains per
second (80), `trans` read speed inside each grain (1; 2 = an octave up), `spd` table passes per second of the grain
start pointer (0 = every grain starts at the beginning), `band` Hz (50: grain decay exp(−π band t)), `ris` s rise
(0.01), `dec` s fall at the end (0.07), `dur` s grain length (0.1).

### `tf_pad`
Time-frequency pad: grains scan the sound and a drawn `curve` shifts each one in frequency by its read position
(points `"x:y"`, x 0..1, y Hz, e.g. `"0:0,0.5:300,1:-100"`; Hann grains, single-sideband shift); `density` grains per
second (40), `size` s (0.08), `seed`.

### PSOLA (Python)
`spectral.pitch_marks(x, sr, f0=None)` (one mark per period at the waveform peak; YIN when `f0` is None),
`spectral.psola(x, sr, semitones, factor, f0=None)`, `spectral.epoch_sharpness(x, sr, f0)` ((crest − 1) ×
autocorrelation at one period: pulsed voice > 2, glottal about 1.2, breathy about 0.6, smooth < 0.5),
`spectral.psola_mask(x, sr)`. `spectral.true_envelope(mag, n_c, accel=True)`: the true envelope started from a max
filter of log|X| and over-relaxed at each step (factor at most 2.5), so it needs far fewer iterations.

## Layer type `resynth`

```json
{"type": "resynth", "path": "in.wav", "stretch": 1.0, "transpose": 0, "formant": 0, "sines": 1, "noise": 1}
```

| key | default | meaning |
|---|---|---|
| `path` | — | WAV file to analyse (any rate; rendered directly at the spec's rate) |
| `stretch` | 1.0 | duration factor (0.05–50); the partial and noise envelopes are stretched, not the waveform, so extreme factors do not smear |
| `transpose` | 0 | semitones for the partials |
| `formant` | 0 | semitones the spectral envelope moves. With `transpose` 7 and `formant` 0 the timbre stays and the pitch rises; `formant` = `transpose` is a tape-style transposition; `transpose` 0 with `formant` ±4 changes the apparent size only |
| `sines` | 1 | gain of the pitched part |
| `noise` | 1 | gain of the noise part (breath, bow, hiss); 0 = clean, 2 = airy |
| `window_ms` | 46 | analysis window; must hold 4 periods of the lowest pitch (46 ms ↔ 87 Hz; use 90 for bass) |
| `seed` | 0 | noise seed |

The usual layer keys (`at`, `gain`, `fx`, `repeat`) apply.

Limits: attacks are not modelled separately, so drum hits and consonants lose their snap (use
`timestretch` for those); the model follows at most 40 partials at a time.

## Recipes

Robot announcer (speech through a vocoder with a chord carrier):
```json
{"out": "announcer.wav",
 "layers": [{"type": "speech", "text": "shield low", "params": {"rate": 0.9},
             "fx": [{"type": "vocoder", "carrier": "chord", "notes": "A2 E3 A3 C#4", "bands": 20, "release": 0.04, "hiss": 0.1}]}],
 "fx": [{"type": "reverb", "mix": 0.12, "size": 0.4}]}
```

Slow-motion impact with a clean attack, an octave down:
```json
{"out": "slowmo_hit.wav",
 "layers": [{"type": "sfx", "kind": "explosion",
             "fx": [{"type": "timestretch", "factor": 3, "transients": true}, {"type": "pitch_shift", "semitones": -12}]}]}
```

Choir pad from one frozen instant of a voice, harmonised:
```json
{"out": "freeze_pad.wav",
 "layers": [{"type": "speech", "text": "ah", "params": {"voice": "female"},
             "fx": [{"type": "freeze", "at": 0.15, "length": 4, "jitter": 40},
                    {"type": "harmonizer", "intervals": [-12, 7], "mix": 0.6},
                    {"type": "spectral_blur", "time": 0.15}]}],
 "fx": [{"type": "reverb", "mix": 0.35, "size": 0.8}, {"type": "fade", "in": 0.3, "out": 1.0}]}
```

Ghost whisper and a talking wind:
```json
{"sounds": [
  {"out": "ghost.wav", "layers": [{"type": "speech", "text": "get out",
     "fx": [{"type": "whisperize"}, {"type": "freqshift", "hz": -40}, {"type": "reverb", "mix": 0.4, "size": 0.9}]}]},
  {"out": "talking_wind.wav", "layers": [{"type": "speech", "text": "danger",
     "fx": [{"type": "cross_synth", "carrier": "wind_loop.wav", "smooth": 2.5}]}]}
]}
```

Re-render a recording: twice as long, a fifth up with the timbre kept, breath reduced:
```json
{"out": "flute_long.wav",
 "layers": [{"type": "resynth", "path": "flute_note.wav", "stretch": 2.0, "transpose": 7, "formant": 0, "noise": 0.5}]}
```

## Python API

```python
from genny import spectral as S

X = S.stft(x, M=2048)                 # mono -> (frames, M/2+1); root-Hann, hop M/4, zero-phase frames
y = S.istft(X, len(x), M=2048)        # identity to 1e-15
S.cola_error(S.window(2048), 1024)    # 0 for a window that overlap-adds to a constant at that hop

y = S.stretch(x, sr, 2.0, mode="vocoder")           # or "sola"
y = S.shift_pitch(x, sr, 7, formant="keep")

m = S.analyze(x, sr)                  # dict: sr, n, hop, win, freq/amp/phase (frames, tracks), noise (frames, bands), edges
y = S.resynth(m, sr=48000, stretch=4, transpose=-5, formant=0, sines_gain=1, noise_gain=0.5, seed=0)
S.save_model(m, "voice.npz"); m = S.load_model("voice.npz")        # or .json

E = S.cepstral_envelope(np.abs(X), n_c=int(0.002 * sr))   # smooth spectral envelope per frame
E = S.lpc_envelope(np.abs(X), order=24)                    # all-pole envelope (hugs the peaks)
H = S.minimum_phase(G, nfft)          # complex response with magnitude G and no pre-ringing
y = S.phase_rotate(x, phi)            # every component shifted by phi rad (Hilbert pair)
f0 = S.twm_f0(freqs, amps_db)         # two-way mismatch fundamental of a peak list
w2 = S.laguerre_map(w, b); b = S.laguerre_b_for(w0, w0_new)   # first-order allpass frequency warping
t2, a2 = S.reduce_envelope(t, a, segments=8)                  # breakpoint envelope cut to a few straight segments
```

### The model (`analyze` → `resynth`)

- **Partials**: frame by frame (Hamming window, 46 ms, hop 11.5 ms) the spectral peaks are located
  to a fraction of a bin by a parabola through three dB values, and linked from frame to frame into
  tracks when they move less than `2·sr/M` Hz (43 Hz) per frame. `freq` is in Hz, `amp` the
  sinusoid's amplitude, `phase` its phase at the frame centre (sample `m·hop − win/2`).
- **Noise**: what remains of the magnitude spectrum after the partials are subtracted, as one level
  per Bark band (`bands="erb"` for finer bands) and frame; `noise[m, b]` is the standard deviation
  white noise would need to have that spectral density in band `b`.
- **Resynthesis**: an oscillator bank with linearly interpolated amplitude; with `stretch == 1` and
  `transpose == 0` the measured phases are followed (cubic phase), so the waveform itself is
  reproduced; otherwise the phase runs free. Noise is random-phase spectra shaped by the band levels.

### Design constants (and where they come from)

| what | value | source |
|---|---|---|
| windows | periodic (denominator M), root-Hann analysis + synthesis | SASP ch. 3, 8 (g12 §2.2, §7.4) |
| hop | M/4 | "robust to modification" hop for the Hann family (g12 §8.3) |
| FFT size when a gain curve is applied | 2M | N ≥ M + L − 1 (g12 §7.1) |
| vocoder frame | 45 ms | SASP TSM comparison settings (g12 §9.8) |
| peak threshold, tracks | −50 dB re frame maximum (book: −30), 60 peaks, 40 tracks, DFmax = 2·sr/M | PARSHL listing (g12 §17 H.8) |
| noise bands | Bark edges 0…15500 (+20500, 27000) Hz, or 1 ERB | g12 §14.1 |
| vocoder bands | Voder edges | g12 §18 G.5 |
| unsourced (justified in `research/SOURCES.md`) | transient detector constants, ±12 ms SOLA search, peak-curvature tolerance, 2-frame track gap, tonal test of `spectral_gate` (12 dB over a ±100 Hz median) | tuned on the test signals |
