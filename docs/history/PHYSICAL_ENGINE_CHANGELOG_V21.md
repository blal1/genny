# Genny 0.21 — Expressive string gestures

## Shared-string interaction layer

v0.21 extends the v0.18–0.20 common physical string core with expressive gestures that operate on the same stateful travelling-wave loop rather than as post-render pitch/effect layers.

### Added to `DispersiveString`

- `render_bend(...)`
  - continuous frequency trajectory produced by changing waveguide round-trip delay;
  - diagnostic `last_tension_ratio` follows `T/T0 = (f/f0)^2` for fixed length and linear density.
- `render_vibrato(...)`
  - finger vibrato as periodic tension/delay modulation inside the string loop.
- `render_slide(...)`
  - continuous delay/length motion plus reduced friction excitation proportional to slide speed, pressure and roughness.
- `render_fret_buzz_feedback(...)`
  - thresholded string/fret collision impulses are reinjected into the travelling-wave loop instead of being mixed only after rendering.
- `render_hammer_on(...)`
- `render_pull_off(...)`
- `render_tapping(...)`
- `render_dead_note(...)`
- `render_pinch_harmonic(...)`

### High-level facades

`PluckedString.expressive_note(...)` exposes:

- `bend`
- `vibrato`
- `slide`
- `fret_buzz_feedback`
- `hammer_on`
- `pull_off`
- `tapping`
- `dead_note`
- `pinch_harmonic`

`BowedString.vibrato_note(...)` uses the same dynamic-delay string core for reduced finger vibrato.

## Notes on modelling scope

These are realtime reduced-order contact models. Fret collision uses a bounded displacement/clearance proxy rather than full geometrical contact FEM, while slide friction is a stochastic reduced texture coupled to the continuous string-length trajectory.

## Validation

Targeted regression suite:

- 88 passed
- versions v0.6 through v0.21
- unified wrappers
- physical models
- spec renderer

The new expressive-string tests include 22.05, 44.1 and 48 kHz rendering.
