# Genny 0.20 — Local string interactions

## New shared string-contact primitives

- `StringExciter`: reduced compliant pick/finger/nail contact. Hardness changes contact duration before the string waveguide.
- `FretboardGeometry`: equal-tempered vibrating length, fret position and clearance/action metadata.
- `DispersiveString.render_exciter()`
- `DispersiveString.render_fretted()`
- `DispersiveString.render_fret_buzz()`
- `DispersiveString.render_glissando()`
- `DispersiveString.render_natural_harmonic()`
- `DispersiveString.render_slap()`

## Physical/reduced modelling changes

- fretting shortens the vibrating length while preserving string construction and tension;
- glissando changes the waveguide delay sample-by-sample rather than pitch-shifting rendered audio;
- fret buzz is generated from thresholded string/fret collision events which excite short metal resonances;
- palm muting changes loop damping locally at the bridge instead of applying a post-render amplitude envelope;
- natural harmonics combine a touched-node partial with a weak residual fundamental;
- slap/pop use the wound-string core plus fret-contact impulses;
- `PluckedString.local_note()` and `.glissando()` expose the shared interactions at the high-level API;
- `BowedString.glissando()` exposes continuous-length bowed gestures;
- public `pizzicato` and `slap_bass` wrappers now use the shared string/contact engine.

These remain realtime reduced-order contact models, not geometric FEM/contact solvers.

## Validation

Targeted compatibility/physical suite: **83 passed**.

The package compiles cleanly for `genny/`, `tests/`, and `examples/`.
