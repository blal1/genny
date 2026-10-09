# Genny 0.3 physical-engine integration

This integration keeps the original Genny API and adds a second, physically informed synthesis layer.

## Added low-level engines

- `genny.physical.core`: rate-correct DSP utilities, contact/felt exciters, filters, loudness helpers.
- `genny.physical.modal`: modal resonator banks, measured/analytic bodies, micro-collisions, fracture,
  aeroacoustic aeolian/sword models.
- `genny.physical.particles`: PhISEM/STK shakers, granular footsteps, Minnaert bubbles and drips.
- `genny.physical.waveguides`: extended Karplus-Strong, mandolin, bowed strings, banded waveguides,
  brass, clarinet, recorder/flute, stiff strings and nonlinear felt piano.
- `genny.physical.voice`: LF glottal source, dynamic Pink Trombone tract, formant/FOF tools and syrinx.
- `genny.physical.analysis`: LPC/lattice analysis-resynthesis, pitch tracking, PSOLA/granular tools,
  modal fitting and environmental statistics.
- `genny.physical.space`: Zita/Puckette/Farnell reverbs, Sabine T60, air absorption and propagation.
- `genny.physical.ambience`: wind, stick-slip/creaks, fire, water, rain, thunder, surf and details.
- `genny.physical.footsteps`: step-force/material models connected to modal and particle engines.

## Added stable high-level API

- `genny.physics`: `Material`, `ModalBody`, `ParticleMaterial`, `BubblePopulation`,
  `StringInstrument`, `Piano`, `Clarinet`, `Brass`, `VocalTract`, `BirdSyrinx`, `Environment`.
- `genny.acoustics`: `Room`, `Propagation`, Sabine and air-absorption helpers.
- `genny.identify`: stable entry points for LPC and modal identification.

All high-level models accept normal Genny sample rates; the research kernels are evaluated at 48 kHz
and resampled at their boundary.

## Compatibility

The original `genny.instruments`, `genny.drums`, `genny.sfx`, `genny.vehicle`, `genny.spec`, filters,
effects and CLI remain intact. Existing instrument/spec regression tests pass.

## Tests performed

- original spec renderer test: pass
- original instrument shape/level test: pass
- original tuning test: pass
- original high-frequency/piercing test: pass
- original level-matching test: pass
- new physical-model smoke tests: pass
- `python -m compileall genny`: pass

## Packaging

Version changed to 0.3.0. Added runtime dependencies `numba`, `soundfile`, `pyloudnorm` in addition to
NumPy/SciPy. The stale 0.2 lockfile was removed because the integration environment had no PyPI network
access to solve the new dependency graph; run `uv lock` on a connected machine.

## Third-party licensing

The adapted physical kernels originate from the user-supplied Klang/procedural project and retain the
Klang Open License 1.0. See `THIRD_PARTY_NOTICES.md`, `THIRD_PARTY_KLANG_LICENSE.txt`, and
`genny/physical/KLANG_LICENSE.txt`.
