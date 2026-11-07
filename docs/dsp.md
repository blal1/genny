# genny.dsp — classical DSP toolbox

Band-limited resampling, FIR and steep IIR filters, convolution with generated impulse responses,
companding and lo-fi codecs, dither, alias-free waveshaping, noise models and metering.
Sources: S. W. Smith, *The Scientist and Engineer's Guide to DSP* and J. O. Smith, *Introduction to
Digital Filters* / *Mathematics of the DFT*. Constants that come from neither book (hardware presets,
voicing curves, wall materials) are justified in `research/SOURCES.md`.

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
`car` = direct sound + reflections 1.5-5 ms apart + cabin bass lift: `cabin_gain`, +12 dB per octave below the
first cabin mode (60 Hz), levelling off at 12 dB where leakage takes over (was +4 dB below 120 Hz). The IR sits at
0 dB in the 1 kHz octave, so only the lows below 60 Hz are lifted.

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
(`tube`, `diode`, `asym` (2x above 0, x/2 below, rounded) and `tube_bendiksen` (asymmetric tube: linear above, clipped below) are asymmetric by themselves), `tone` (post low-pass Hz, 0 = off), `mix`,
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
| `gsm` | 8 kHz, LPC + pulse excitation (GSM-like, not bit-exact): log-area ratios quantised as GSM 06.10 Table 3.1 and interpolated between frames; buzzy mobile-phone voice | pulse bits 2-4 |
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
`wow_rate` (wow frequency, Hz, default 0.6; 0.556 = a record at 33 1/3 rpm), `seed`. Pitch moves by exactly the depth: `wow` 1 swings a 1000 Hz tone between 990 and 1010 Hz.

### `wall` — through a wall, door or window
`material` `window|door|solid_door|drywall|brick|concrete`, `thickness` (mass multiplier; doubling adds
6 dB of loss), `leak` (dB: the most the partition can remove, set by gaps and flanking paths; default per
material from 22 for a door to 58 for concrete), `makeup` (dB of gain afterwards).
The loss follows the field-incidence mass law, 20·log10(f·m) − 48 dB with m in kg/m²: 6 dB more per octave, so
bass gets through and treble does not. Above the coincidence frequency fc (window 3.8 kHz, drywall 3 kHz,
concrete 125 Hz) Cremer / Josse-Lamure takes over, 20·log10(f·m) − 42 + 10·log10[(2η/π)(f/fc − 1)] with η = 0.03,
f/fc − 1 held at a third-octave width (0.23) in the band around fc: a 17.5 dB dip at fc, rising to 9 dB more per
octave far above; the two are blended over fc/2..fc, and `leak` caps the result. Rw of the presets (`wall_rw`):
window 28, door 22, solid_door 30, drywall 43, brick 52, concrete 50. Unlike `muffle` (4 dB quieter at most) the level really drops, by 20-55 dB;
either mix the layer at that level or add `makeup`.

`drywall` is a double-leaf stud wall: two 13 mm gypsum boards (10.65 kg/m² each) on a 90 mm air gap, studs every
0.6 m, flanking ceiling 52 dB (`wall_loss_db(f, mass, m2=, gap=, stud=)`). Below the mass-air-mass resonance it
follows the mass law of both leaves together; above it the loss rises 18 dB per octave up to the cavity limit
c/(2πd), then R1 + R2 + 6; the studs bridging the leaves cap it. Before, drywall was a single leaf (27 / 39 dB).

| material | loss at 250 Hz | at 1 kHz |
|---|---|---|
| `window` | 20 dB | 30 dB |
| `door` | 22 dB | 22 dB |
| `solid_door` | 29 dB | 30 dB |
| `drywall` | 35 dB | 47 dB |
| `brick` | 46 dB | 52 dB |
| `concrete` | 40 dB | 58 dB |

### `dc_block`
`cutoff` (Hz, default 20).

### Added since 1.2.1

Parameters: `catalog.md`.

- Distortion and folding: `chebyshev` (a full-level sine comes out with exactly the harmonics listed, quieter input
  gets fewer), `diode_clipper` (clipping that depends on frequency, as in the circuit), `crunch` (dead zone: only
  crests survive; on noise a sparse crackle), `serge_fold` and `lockhart_fold` (exact Lambert W folds, oversampled;
  `lockhart_fold` has a one-pole tone filter), `exciter` (Aphex type: distorted treble added back).
- Dynamics: `multiband` (a compressor per band, Linkwitz-Riley split that sums flat at ratio 1), `deesser`.
- Lo-fi: `delta_sigma` (one-bit PDM stream, its rising noise left in), `tape_stop` (speed and level fall to
  nothing).
- In `fx` (core effects): `ringmod` `leak` (0 = pure ring; carrier and input leak through as an unbalanced diode
  ring, wet = x c + leak (x + c); 0.5 = the classic ab + (a + b)/2), `gate` (hysteresis and hold: `open_db`, `close_db`, `hold`), `dyno` (ChucK's Dyno: limit,
  compress, gate, expand, duck), `expander`, `duck` (side-chain under another sound), `compressor` `mix` (parallel),
  `key` (side-chain WAV), `preset` mastering | invisible | parallel; `limiter` `lookahead`, `true_peak` and a double
  release; `ladder` (Moog or diode ladder, self-oscillating at `res` 1), `wah` (pedal, LFO or auto-wah), `octaver`,
  `radio` (walkie-talkie squelch, keyed carrier, scanning), `balance`, `exp_delay` (echoes bunching like a bouncing
  ball), `shaker_comb` (Puckette's wandering four-tap comb), `pan itd` (interaural delay), `width mode=ms`.
- Every `compressor` and `limiter` sounds different from 1.2.1 (new detector); so do `gsm_like` and the CVSD codec.
- More core effects (`fx`): `kasfilter` (undersampling filter: two held values per cycle of `freq` cross-faded on a
  cosine, `res` feedback, `accent`), `phase_bash` (each window becomes one cycle of a zero-phase wavetable played at
  `pitch`), `antisat_svf` (SVF with a sinh antisaturator on its damping: self-oscillates past `res` 0.5 at a held
  level), `random_allpass` (second-order allpasses at random log-spaced frequencies: same spectrum, scrambled
  phase). `ladder` `kind` huovilainen (2x oversampled, tanh per stage) and `solver` node | newton (delay-free loop)
  | adaa (antialiased tanh in the loop); `tremolo` `skew`, `duty`, `phase`, `spread` (stereo); `distortion
  aa=adaa2`; `overdrive aa=adaa3` (shapes hard, fold); `limiter`: what is left to clip is clipped 4x oversampled;
  `reverb freeze=true` (Freeverb freeze: the sound hangs for the whole tail); `width`: the Haas copy is at most
  12 ms late (the precedence holds for 1-30 ms, beyond it the copy splits off). Python: `fx.envelope_follower`,
  `echo_threshold_db`, `haas_delay_ms`, `echo_criterion`. Clean low end: `highpass cutoff=30 order=2`.
- Filters (`genny.filters`, Python): `svf` modes lowshelf, highshelf, tilt; `matched_shelf` (first-order shelf
  matched to its analog prototype); `svf_antisat`; `onepole_atanh` (output kept inside ±1); `ladder(solver=, q=)`
  with `faust_q_to_k` (Faust's Q to loop-gain mapping) and kind huovilainen; `stretched_eq`; `prewarp_bounded`
  (bounded cutoff prewarping); `random_allpass`; `sallen_key(x, kind, cutoff, sr, k)` (linear MS-20-type low-pass or
  band-pass, the SVF with 2R = 2 − k; k → 2 self-oscillates, held below 1.99) and `tsk_allpass_q(k)` (Q of the TSK
  all-pass built on it: R = (1 − k)/(1 + k), Q = 1/(2R)). `onepole_nl(x, cutoff, sr, kind)`: nonlinear one-pole
  low-pass, `transistor` (tanh at input and state), `ota` (tanh of the difference) or `diode` (four-stage diode
  cascade), `zdf_hard` (zero-delay-feedback pole, hard clip of the state) or `zdf_ota` (tanh solved by bisection,
  12 iterations); `butter2(x, cutoff, sr, R=0.3, N=2, kind)`: Butterworth of the second kind, N SVF sections, the
  resonance bump (4.85 dB at R 0.3) independent of N; `ladder_mix(x, cutoff, sr, res, mode="hp4")`: modal mixes of the four ladder stages, `mode` a key of
  `LADDER_MIX` (lp1, lp2, lp4, hp1, hp2, hp4, bp2, bp4, notch, ap3) or 5 weights; `tbvcf(x, cutoff=500, res=0.8,
  dist=2, asym=0.5)`: Pinkston's TB-303 VCF (Csound tbvcf), fixed step so it does not depend on sr; `onepole4(x,
  cutoff, sr)`: four one-pole low-passes as one recursion, −3 dB at the cutoff.
- Effect `va_ladder` (`genny.synths`): virtual-analog filters as exact linear models (prewarped bilinear transform).
  `kind` lp | hp | bp (LP-HP stages, positive feedback; bp 8 with k < 0 = two peaks) | korg35 (MS-20 low-pass) |
  ellip (steep elliptic low-pass); `poles` 4 | 8 (lp, hp, bp); `cutoff` 1000 Hz; `k` feedback (lp 0..4, 8 poles
  0..1.88; hp > −1; bp 0..2^N, 8 poles also down to −64; korg35 0..2; ignored by ellip).
- In this module: `adaa_n` (antiderivative anti-aliasing of any order N; `waveshape aa="adaa3"`), `gsm_like` with
  interpolated reflection coefficients (analysis and synthesis lattices: `codec` sounds different), `wall
  leak_area` (open fraction: TL' = −10 log((1 − r) 10^(−TL/10) + r), `leak_loss_db`), `convolve loop=true` (the
  tail folded back onto the start: same length, loops seamlessly), `ir_spring` with the echo time of a real
  two-spring tank (helical-spring dispersion; `convolve kind=spring` changes), `factor_body` (the strongest body
  resonances replaced by short ones, so the residual IR is short).

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

### `complex_osc`
Two cross-modulated sines (West-Coast complex oscillator): each modulates the other's frequency, phase and
amplitude, and the output is folded or wrapped.

| param | default | meaning |
|---|---|---|
| `dur` | 2.0 | seconds |
| `freq1`, `freq2` | 40, 0.5 | Hz of the main and the second oscillator |
| `fm1`, `fm2` | 0.1, 0.1 | frequency modulation of each by the other (× 1..1000 Hz) |
| `pm1`, `pm2` | 0, 0 | phase modulation of each by the other (× −π..π) |
| `ring1`, `ring2` | 1, 1 | amplitude (ring) modulation of each by the other |
| `out` | `fold` | `fold` (±0.8) or `wrap` (±0.9) |
| `preset` | `none` | `olarte`: fm1 0.143219, fm2 0.2345, pm1 0.013, pm2 0.081, ring 0.1997 |

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
`one_over_f`, `mains_hum`; `a_weighting_db`, `a_weighting_sos`, `a_weight`, `vu_meter`, `wall_loss_db` (`m2`, `gap`, `stud`: double wall), `weighted_rw(tl)` (single-number Rw of a third-octave loss: the
reference curve raised in 1 dB steps while the unfavourable deviations sum to 32 dB at most, read at 500 Hz),
`wall_rw(material="drywall")` (Rw of a `wall` preset), `cabin_gain`;
`pqmf_bank`, `pqmf_analysis`, `pqmf_synthesis` (pseudo-QMF cosine-modulated sub-band bank), `ltv_fir` (linear
time-varying FIR by frequency sampling), `compress_envelope` and `mode_quota` (envelope storage and mode budget of a
modal synthesiser), `noise_band_for_rate` (bandwidth of a noise band whose envelope fluctuates at fe: fe = 0.64 Δf),
`wsb` (Chebyshev waveshaping bank, octaves 2, 4, 8), `soft_cf` (soft clip by a continued fraction of arctan),
`k_to_lar`, `lar_to_k`, `gsm_lar_quantize` (GSM 06.10 log-area ratios).

FDN and decorrelation: `fdn_allpass_matrix(A, g)` (feedback matrix of an all-pass FDN from a unitary A and the gains),
`fdn_poles(A, m)` (poles for delays m), `is_unilossless(A)` (lossless for every delay set: all poles on the unit circle,
none repeated); `decorrelation_fir(sr, length, seeds, target, seed, bands="erb")` (flat-magnitude random-phase FIR
of 20 ms or less, the best of several seeds), `dynamic_allpass(x, sr, order=200, rate=0.2, depth=0.1, seed)`
(time-varying all-pass decorrelator). Oscillators and wave digital filters: `wg_osc(freq, n, sr, tau)` (second-order
waveguide oscillator, `tau` s exponential decay); `wdf_port(kind, value, sr)` (mass, spring or damper: port
resistance and reflection), `wdf_series(R)`, `wdf_parallel(G)` (adaptor coefficients), `wdf_oscillator(force, sr, m,
k, r)` (velocity of a driven mass-spring-damper). Colour: `apdist(x, sr, drive, smooth)` (signal-dependent all-pass:
phase distortion at constant level), `variation_filter(x, sr, n=10, depth_db=8, seed)` (random peaking cuts, a
different colour per seed for repeated samples), `head_shadow(x, sr, azimuth, radius)` (one ear's head-shadow
filter), `median_comb(x, sr, freq, k=4, amount, transient)` (harmonic comb by a median of taps, the attack left
dry). `modal_fft(freqs, decays, amps, dur, sr, bins=5, n)`: modal synthesis frame by frame in the frequency domain.
