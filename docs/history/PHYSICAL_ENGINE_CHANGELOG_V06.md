# Genny 0.6.0 — Physical catalogue migration

This release continues the migration of Genny's public catalogue onto the shared physical engines introduced in 0.3–0.5.

## New high-level physical models

- `TunedPercussion`: modal/additive struck bars, bells, wood blocks and gong-like bodies.
- `PluckedString`: physical/commuted plucked-string facade, including STK-style mandolin.
- `CombustionEngine`: four-stroke engine facade; firing rate is `cylinders * RPM / 120`.
- `Environment.pour()`: running-water + Minnaert bubble population + rising container-cavity resonance.

## Public catalogue migrated

### Instruments
- `xylophone` -> physical modal bar
- `glockenspiel` -> physical bell model
- `tubular_bell` -> long-decay physical/additive bell
- `mandolin` -> double-string commuted waveguide
- `steel_guitar` -> physical plucked string + body resonances
- `banjo` -> bright physical plucked string + compact body

### Drums / percussion
- `shaker` -> PhISEM Maraca
- `tambourine` -> PhISEM Tambourine
- `woodblock` -> STK modal wooden bar
- `gong` -> Farnell-derived inharmonic gong groups with time-varying shimmer

### SFX / environments
- `rain` -> procedural physical/statistical rain field
- `fire` -> coupled hiss/crackle/roar combustion texture
- `stream` -> stochastic running-water model
- `pour` -> stream + bubbles + filling-cavity resonance
- `engine` -> four-stroke combustion engine model
- `swoosh` -> distributed Reynolds/Strouhal aeroacoustic swing

## Validation

The focused regression suite passes at release time:

- `tests/test_spec.py`
- `tests/test_physics.py`
- `tests/test_unified_wrappers.py`
- `tests/test_v06_models.py`
- `tests/test_v06_migration.py`

Result: **11 passed**.

`compileall` also succeeds for `genny/`, `tests/` and `examples/`.
