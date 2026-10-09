# Genny 0.4 physical-engine continuation

This pass continues the v0.3 integration without breaking the legacy module APIs.

## Added

- `BarBody`: analytical Euler-Bernoulli free/cantilever bar model using material density,
  Young's modulus, dimensions and Rayleigh damping, rendered by the shared modal bank.
- `BowedString`: high-level wrapper around the bidirectional bowed-string waveguide.
- `Flute`: high-level jet/waveguide flute and experimental recorder model.
- `Footstep`: GRF-driven footsteps with hard-floor modal impacts, PhISEM grains,
  stick-slip/crushing layers and shoe/floor coupling.
- `AeroacousticSwing`: Reynolds/Strouhal distributed swept-object aeroacoustics.
- `Environment.wind`, `Environment.creak`, `Environment.cave_drips`.
- `MovingPropagation`: time-varying physical propagation delay; Doppler emerges from the
  varying delay instead of a separate pitch shift, with geometric loss and atmospheric filtering.

## Design consolidation

The public physical API now covers the major reusable causal families found in the supplied
research corpus: modal solids, fracture, impacts, bars, particles, liquids, plucked/bowed strings,
piano, reeds, brass, jets/flutes, vocal tracts, syrinx, footsteps, environments, static and moving
propagation, and room acoustics.
