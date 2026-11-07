# household — objects, appliances and plumbing (`genny/household.py`)

Locks, doors, windows, office and hand tools, appliances, alarms, plumbing, games and furniture. Every name and
parameter is in [`catalog.md`](catalog.md) ("Index by module"); this page gives the physics behind the entries
whose model is not obvious from their parameters.

## Models

- **`appliance kind=frigo`**: the compressor turns at the mains frequency minus the slip of an induction motor
  (2900 rpm on 50 Hz, `COMP_SLIP`): 48.3 Hz on 50 Hz mains, 58 Hz with `mains=60`.
- **`faucet`**: each drop rings a bubble near 5 kHz that dies in about 1.4 ms, the damping of a bubble of that
  radius (thermal, viscous and radiation, `bubble_tau(f)`); it rises over its first period, so it starts without a
  click.
- **`filled_glass`**: the liquid adds mass to the wall, so the pitch falls as f0 / √(1 + a·fill⁴),
  a = α ρ_liquid R / (5 ρ_glass h) = 2.33 for a wine glass of 35 mm rim radius and 1.5 mm wall (`fill_coef()`):
  full, about 10 semitones lower.
- **`roller_drop`**: `buche` (log) and `tube` ring on the bending modes of a free bar (`beam_modes(length,
  gyration, material)`, `FREE_BEAM`): a 40 cm log near 2.3 kHz, a 60 cm steel tube near 880 Hz. The lying bottle
  (`bouteille`) rings on its shell modes.
- **`hair_dryer`, `vacuum`**: the level follows the fan law, sound power ∝ speed⁵ (`FAN_POWER`): +50·log10(speed)
  dB, so +7.3 dB at +40 %.

## Python

`bubble_tau(f)`, `fill_coef(radius, wall, rho_l, rho_g, alpha)`, `fill_freq(f0, fill)`, `beam_modes(length,
gyration, material)`; constants `FAN_POWER`, `COMP_SLIP`, `GLASS_ALPHA`, `GLASS_R`, `GLASS_WALL`, `FREE_BEAM`.
Sources of every constant: `research/SOURCES.md`.
