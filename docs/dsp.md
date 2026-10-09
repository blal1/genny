# genny.dsp — classical DSP toolbox

Band-limited resampling, FIR and steep IIR filters, convolution with generated impulse responses,
companding and lo-fi codecs, dither, alias-free waveshaping, noise models and metering.
Sources: S. W. Smith, *The Scientist and Engineer's Guide to DSP* and J. O. Smith, *Introduction to
Digital Filters* / *Mathematics of the DFT*. Constants that come from neither book are marked
`# UNSOURCED` in `genny/dsp.py` (hardware presets, voicing curves, wall materials).

All effects keep the channel count and the length of the input, except `convolve`, which appends the
tail of the impulse response. Effects with randomness take `seed`.

## Effects

### `convolve` — convolution reverb / body
| param | default | meaning |
|---|---|---|
| `ir` | `""` | path to a wav impulse response (resampled to the render rate, energy-normalised); empty = generate `kind` |
| `kind` | `room` | `room`, `hall`, `plate`, `spring`, `cabinet`, `tube`, `car`, `phone-speaker` |
| `decay` | 0 | RT60 in seconds: time for the tail to fall 60 dB (0 = default: room 0.45, hall 2.2, plate 1.6, spring 2.0, tube 0.15, car 0.09) |
| `size` | 0.5 | 0..1. room: first reflection after 4-20 ms; hall: 20-100 ms; spring: loop delay 30-65 ms; tube: length 0.1-2 m; cabinet/phone: larger = lower resonances |
| `bright` | 0.5 | 0..1. High-frequency RT60 relative to the low one (0.25x..0.85x); spring transition frequency 2.5-6 kHz |
| `mix` | auto | wet 0..1; default 0.3 for the four spaces, 1.0 for cabinet/tube/car/phone-speaker and files |
| `predelay` | 0 | seconds before the wet signal |
| `seed` | 0 | which random room |

What the kinds are: `room`/`hall` = sparse early reflections (amplitude falls as 1/distance) plus Gaussian
noise decaying per octave band; `plate` = dense tail from t = 0, high frequencies first; `spring` = a
dispersive delay loop, every echo a rising chirp; `cabinet`/`phone-speaker` = minimum-phase filter from a
magnitude curve (no pre-ringing); `tube` = pipe closed at one end, resonances at (2n-1)·343/(4L) Hz;
`car` = direct sound + reflections 1.5-5 ms apart + cabin bass lift.

### `fir_eq` — draw an EQ curve
`points` `"hz:db,hz:db,..."` (flat outside the end points, smooth in between), `phase` `linear` (no phase
smear, delay compensated; rings before and after transients) or `min` (rings only after), `taps` (0 = 93 ms,
which resolves features about 65 Hz wide; raise it for corners below ~100 Hz).

### `steep_lowpass`, `steep_highpass`, `steep_bandpass`
`cutoff` Hz (-3 dB) — or `low`/`high` for the band-pass — `order` (number of poles, even, 2..20; the slope
is 6 dB/octave per pole), `type` `butter` (flat pass band), `cheby` (steeper, with `ripple` percent of
pass-band ripple, 0..29; 0.5 is the usual choice) or `bessel` (no overshoot: use it on clicks and drums).
These are true cascades with the correct per-stage Q, unlike `lowpass order=2`, which repeats one biquad.

### `overdrive` — alias-free waveshaper
`drive` (input gain, 0.05..100), `bias` (-1..1: offset before the curve; asymmetry adds even harmonics,
the octave "warmth"; a symmetric curve gives odd harmonics only), `shape` `tanh|tube|fuzz|fold|diode`
(`tube` and `diode` are asymmetric by themselves), `tone` (post low-pass Hz, 0 = off), `mix`,
`oversample` (1|2|4|8; 1 is the naive version that aliases).

### `wavefold`
`folds` (input gain: how many times a full-scale wave is reflected, 0.1..20), `symmetry` (-1..1 offset),
`shape` `sine|tri`, `tone`, `mix`. Always 8x oversampled.

### `mulaw` — companded quantisation
`bits` (2..16), `law` `mu` (μ = 255) or `a` (A = 87.6). The quantisation noise follows the signal level
(about 38 dB below it at 8 bits) instead of sitting at a fixed floor: the telephone-codec grain.

### `dither`
`bits` (1..24), `shape` `none` (bare rounding: quiet tails turn into gritty distortion, then silence),
`tpdf` (±1 LSB triangular noise: the error becomes steady hiss), `gauss` (sd 2/3 LSB), `shaped` (tpdf with
the noise pushed toward high frequencies, 13 dB quieter below fs/16), `seed`.

### `codec`
| `kind` | what it is | `quality` 0..1 |
|---|---|---|
| `adpcm` | 4-bit IMA ADPCM: hiss that follows the signal | sample rate 6-32 kHz |
| `cvsd` | 1-bit, step grows after four equal bits: grainy military-radio speech | clock 12-64 kHz |
| `delta` | 1-bit fixed step: slewing smear on loud highs, idle buzz | clock 48-192 kHz |
| `gsm` | 8 kHz, LPC + pulse excitation (GSM-like, not bit-exact): buzzy mobile-phone voice | pulse bits 2-4 |
| `dct` | block transform with coarse high-frequency quantisation: watery/bubbly streaming artefact | coarse..fine |

### `lofi` — old sampler / DAC
`preset` `nes|snes|gameboy|amiga|sp1200|telephone|voicemail|walkie|none`, then optional overrides:
`rate` (Hz), `bits`, `dither` (`none|tpdf|gauss|shaped`), `law` (`linear|mu|a`), `hold` (true = zero-order
hold DAC: keeps the mirror images above rate/2, each weighted by sin(x)/x — the metallic ring of old
samplers; false = clean reconstruction), `seed`. The input is properly low-passed before the rate is
reduced, so what you hear is the machine, not accidental aliasing.

| preset | rate | bits | extra |
|---|---|---|---|
| `nes` | 33144 | 7 | hold |
| `snes` | 16000 | 16 | ADPCM, 7 kHz band |
| `gameboy` | 8192 | 4 | hold |
| `amiga` | 16726 | 8 | 4.4 kHz band, hold |
| `sp1200` | 26040 | 12 | hold |
| `telephone` | 8000 | 8 μ-law | 200-3200 Hz |
| `voicemail` | 8000 | ADPCM | 300-3000 Hz |
| `walkie` | 8000 | 6 μ-law | 400-2800 Hz |

### `smooth`
`width` (ms; first null at 1000/width Hz), `passes` (1 box, 2 triangle, 4 ≈ Gaussian). No ringing, no delay.
The gentlest way to round a click or a stepped control signal.

### `tape`
`wow` (slow speed drift, percent, ~0.6 Hz), `flutter` (fast wobble, percent, ~7 Hz), `drive` (saturation),
`bump` (dB boost around 80 Hz from the playback head), `hf` (roll-off Hz), `hiss` (dB RMS, -120 = none),
`seed`. Pitch moves by exactly the depth: `wow` 1 swings a 1000 Hz tone between 990 and 1010 Hz.

### `wall` — through a wall, door or window
`material` `window|door|solid_door|drywall|brick|concrete`, `thickness` (mass multiplier; doubling adds
6 dB of loss), `leak` (dB: the most the partition can remove, set by gaps and flanking paths; default per
material from 22 for a door to 58 for concrete), `makeup` (dB of gain afterwards).
The loss follows the mass law, 20·log10(f·m) − 47 dB with m in kg/m²: 6 dB more per octave, so bass gets
through and treble does not. Unlike `muffle` (4 dB quieter at most) the level really drops, by 20-55 dB;
either mix the layer at that level or add `makeup`.

| material | loss at 250 Hz | at 1 kHz |
|---|---|---|
| `window` | 21 dB | 30 dB |
| `door` | 22 dB | 22 dB |
| `solid_door` | 30 dB | 30 dB |
| `drywall` | 28 dB | 40 dB |
| `brick` | 47 dB | 52 dB |
| `concrete` | 52 dB | 58 dB |

### `dc_block`
`cutoff` (Hz, default 20).

## SFX

### `noise_bed`
| param | default | meaning |
|---|---|---|
| `kind` | `hum` | `hum` (mains + harmonics + 1/f floor), `hiss`, `rumble`, `crackle` (vinyl/fire ticks), `geiger` (counter clicks), `static_field` (dense electrical static) |
| `dur` | 2.0 | seconds |
| `rate` | 0 | events per second for crackle/geiger/static_field (0 = 25 / 12 / 4000). Events are a Poisson process: random, clustered, and a dense stream gets 3 dB louder per doubling of the rate |
| `level` | -12 | dB: peak of a typical single event, or about the peak of the continuous kinds |
| `freq` | 50 | mains frequency for `hum`: 50 or 60 |
| `bright` | 0.4 | 0..1 spectrum tilt / click sharpness |
| `seed` | 0 | |

Loop a bed with `"loop": 0.3` on the spec.

## Recipes

```jsonc
// Voice from the next room, then the door opens
{"out": "next_room.wav", "normalize": null, "layers": [
  {"type": "speech", "text": "game over", "fx": [{"type": "wall", "material": "door", "makeup": 10}]},
  {"type": "speech", "text": "game over", "at": 1.5}]}

// Guitar through an amp: overdrive into a speaker cabinet, a little spring reverb
{"out": "amp.wav", "layers": [{"type": "seq", "inst": "electric_guitar", "steps": "E2:.4 G2:.4 A2:.8"}],
 "fx": [{"type": "overdrive", "drive": 8, "shape": "tube", "tone": 0},
        {"type": "convolve", "kind": "cabinet"},
        {"type": "convolve", "kind": "spring", "decay": 1.8, "mix": 0.2}]}

// 90s sampler drums on worn tape
{"out": "dusty.wav", "bpm": 90, "layers": [
  {"type": "pattern", "kind": "kick", "steps": "x---x---"}, {"type": "pattern", "kind": "snare", "steps": "--x---x-"},
  {"type": "sfx", "kind": "noise_bed", "params": {"kind": "crackle", "dur": 2.7}, "gain": 0.3}],
 "fx": [{"type": "lofi", "preset": "sp1200"}, {"type": "tape", "wow": 0.3, "flutter": 0.15, "drive": 2.5}]}

// Radio call over a bad link
{"out": "radio.wav", "layers": [{"type": "speech", "text": "fuel low"}],
 "fx": [{"type": "lofi", "preset": "walkie"}, {"type": "codec", "kind": "cvsd", "quality": 0.2}]}

// Old amplifier left on: hum bed, loopable
{"out": "amp_hum.wav", "loop": 0.3, "duration": 4,
 "layers": [{"type": "sfx", "kind": "noise_bed", "params": {"kind": "hum", "dur": 4.3, "freq": 60, "level": -18}}]}
```

## Python helpers (not in the catalog)

`resample_sinc(x, sr_in, sr_out)`, `varispeed(x, rate)`; `windowed_sinc`, `fir_lowpass/highpass/bandpass/
bandstop`, `spectral_invert`, `spectral_reverse`, `fir_from_curve`, `min_phase`, `min_phase_from_mag`,
`fir_filter`, `fft_convolve`; `smith_stage`, `smith_sos`, `iir_sos`, `steep_filter`, `dc_blocker`,
`moving_average`; `make_ir`, `ir_noise`, `ir_early`, `ir_room`, `ir_plate`, `ir_spring`, `ir_tube`,
`ir_curve`, `chirp_allpass`; `mulaw_compress/expand`, `alaw_compress/expand`, `quantize`,
`delta_encode/decode`, `cvsd_encode/decode`, `adpcm_encode/decode`, `block_dct`, `gsm_like`, `zoh_droop`,
`zoh_resample`, `zoh_compensate`, `oversampled`, `waveshape`; `gaussian`, `poisson_events`, `lognormal`,
`one_over_f`, `mains_hum`; `a_weighting_db`, `a_weighting_sos`, `a_weight`, `vu_meter`, `wall_loss_db`.
