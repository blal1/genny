# Genny 0.5.0 — unified high-level physical wrappers

This pass starts routing the long-standing public instrument/SFX names through the physical engine while preserving their established call signatures.

## Migrated instrument names

- `piano` → nonlinear felt hammer + stiff/dispersive string + commuted soundboard
- `violin`, `cello` → bowed-string waveguides with body coupling and rosin noise
- `clarinet` → nonlinear reed + bore waveguide
- `flute` → jet-driven recorder/flute waveguide
- `trumpet`, `french_horn`, `trombone`, `tuba` → lip-reed brass waveguide family with instrument-specific radiation filtering

Historical controls such as `bright`, `decay`, `attack`, `vibrato`, `breath`, and `mute` remain accepted and are mapped onto physical or perceptual controls.

## Migrated SFX names

- `footstep` → GRF/contact/ground model from `physical.footsteps`
- `wind` → causal procedural wind scene
- `thunder` → segmented procedural thunder with distance parameter
- `hit` → modal-body impact with material/hardness-driven micro-collisions

`footstep`, `wind`, and `thunder` are downmixed to mono at the legacy SFX surface for backward compatibility; the high-level physical APIs can still expose richer spatial output.

## Robustness fixes

- Short thunder renders no longer fail when the underlying research patch's one-second fade is longer than the requested buffer. Genny renders a valid causal minimum window then crops/pads.
- Short wind and cave-drip safeguards from v0.4 remain in place.
- New regression tests exercise migrated wrappers at 22.05, 44.1 and 48 kHz.

## Migration policy

The older synthesis recipes remain in source as documented fallbacks/reference implementations during the transition, but the registry entries for the migrated names point at the physical wrappers. This makes the refactor reversible while avoiding API breakage.

## Pitch-stable compatibility layer

Nonlinear lip/jet/bow waveguides can legitimately settle into neighbouring oscillation regimes at the extremes of a legacy instrument's advertised register. For the established `render_note(...)` names, v0.5 therefore blends the physical model with a quiet deterministic acoustic anchor at the requested fundamental. This keeps legacy tuning predictable while preserving the physical transient/noise/nonlinear texture. The direct `genny.Brass`, `genny.Flute`, and `genny.BowedString` APIs remain unanchored physical models.
