# osc — oscillators and noise (`genny/osc.py`)

The waves behind the `synth` instrument (`wave`), the tonal sfx and many models. Every function takes a scalar
frequency or a per-sample frequency array (sweeps). `genny list waves` prints the names `osc.osc(wave, freq, n,
sr, **kw)` accepts.

## Waves

| wave | what it is |
|---|---|
| `sine`, `saw`, `square`, `pulse` (`width` 0.25), `triangle` | the classic shapes, band-limited. Steps (saw, pulse, square) are corrected with a 2-point polyBLEP, and with the 4-point residual (integrated cubic B-spline) above 2 kHz; the triangle's corners use the 4-point polyBLAMP by default (`triangle(points=2)` for the old one) |
| `supersaw` | detuned stack of saws (`voices` 7, `detune` 0.4 semitones total) |
| `jp8000` | the Roland JP-8000 supersaw: seven fixed, asymmetric offsets −21, −14, −7, 0, 4, 8, 12 cents (`supersaw(mode="jp8000")`, scaled by `detune` / 0.4) |
| `fm` | sine carrier and modulator (`ratio`, `index`); `fm(ratio2=, index2=)` adds a second modulator in parallel |
| `cz` | Casio CZ phase distortion: the phase is bent in two straight pieces before a cosine is read (`amount`) |
| `gbuzz` | closed-form buzz: `nharm` cosine harmonics, each `mul` times the one below |
| `dpw_saw`, `dpw_triangle`, `dpw_square` | differentiated polynomial waveforms, `order` 1-6 (default 3) |
| `blit`, `blit_saw`, `blit_square` | band-limited impulse train and the shapes integrated from it; `kernel` sinc (closed form), bspline or thiran (fractional delay by a first-order allpass) |
| `sync` | hard-synced saw: a slave saw at `ratio` 2.5 x the note restarted every period; every wrap and restart corrected by a windowed-sinc band-limited step |
| `organ`, `wavefold`, `gendyn` | drawbar sum, folded sine, Xenakis's dynamic stochastic synthesis (`gendyn(curve=)` bends the segments between breakpoints) |
| `pitched_noise` | noise clocked by the pitch: two new values per period, joined by lines or held (`color` white, pink, brown) |
| `held_noise` | stepped noise: a new value at the given rate (Hz), held in between |
| `noise`, `pink`, `brown` | seeded noise |

## Python building blocks

- `residual(t, deg=3, kind="step"|"ramp")`: band-limited step or ramp minus the trivial one, from the integrated
  B-spline of degree `deg` (1: 2-point polyBLEP / polyBLAMP, 3: 4-point).
- `dpw(freq, n, sr, order=3, wave="saw")`, `blit`, `hard_sync`, `phase_distortion`, `gbuzz`, `classic(jumps=,
  corners=)` (any waveform from its breaks).
- `harmonic(f0, amp, dist, n)`: harmonic bank with factored amplitudes, `amp` the overall amplitude and `dist` the
  share of each harmonic (frames x harmonics); harmonics at or above Nyquist are dropped and the rest renormalised.
- `corner_clip(x, limit, mode="clip"|"half"|"full")`: hard clipper or half- / full-wave rectifier with its corners
  band-limited (each crossing found by a cubic fit on 4 samples, then a polyBLAMP; Esqueda's method).
- `minblep_saw(freq, n, sr, phase0)`: saw whose wraps are corrected by a minimum-phase BLEP, causal, so it also
  suits resets that cannot be foreseen (hard sync, retrigger); `minblep_table(zc, over)` builds that residual
  (windowed sinc made minimum phase through the real cepstrum, integrated, minus the ideal step).
- `wavetable_from_spectrum(harmonics, n=2048)`: one period from harmonic amplitudes; `wavetable_osc(freq, n, sr,
  harmonics)` reads it with linear interpolation and drops harmonics above 0.45 sr at the highest frequency.
- Perceptual alias-free limits (Hz up to which the aliasing stays inaudible): `DPW_ALIAS_FREE` per DPW order at
  44.1 and 88.2 kHz, `SAW_ALIAS_FREE` per saw method, `FRAC_DELAY_ALIAS_FREE` per fractional-delay interpolator;
  `dpw_order_for(f0, sr)` gives the lowest DPW order (2-6) that stays clean up to f0, 0 if none does.

Default sounds changed in this version: everything built on `triangle`, and `saw`, `pulse`, `square` above 2 kHz.
