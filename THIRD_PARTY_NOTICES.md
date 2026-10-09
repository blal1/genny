# Third-party notices

## Klang / procedural physical-model engines

Parts of `genny/physical/` are adapted from the user-supplied **procedural** project by Chris Nash,
Copyright © 2025 Chris Nash, under the **Klang Open License 1.0** (based on Apache 2.0).
The complete license is included as `THIRD_PARTY_KLANG_LICENSE.txt` and
`genny/physical/KLANG_LICENSE.txt`.

Since genny 1.0, `genny/klang.py` (vehicle, interface, body and world sounds translated to Python),
and functions marked as ports in `genny/creatures.py`, `genny/choir.py`, `genny/compose.py` and
`genny/foley.py`, are also derived from that project and fall under the same licence.

The adapted files were reorganized under `genny.physical`, package imports were changed for Genny,
and a stable high-level API was added in `genny.physics`, `genny.acoustics`, and `genny.identify`.

Important: the Klang Open License adds an attribution requirement for interactive audio-visual
products. Consult the bundled license before shipping a game, simulation, or multimedia installation.

## genny/cyber.py: models and constants read in other projects

`genny/cyber.py` contains no code copied from another project. It re-implements published models, and
some of its numeric constants were read in these sources:

- `pipe_engine`: the engine model of Baldan, Lachambre, Delle Monache & Boussard (2015) as implemented in
  **DasEtwas/enginesound** and **Antonio-R1/engine-sound-generator** (both MIT); reflection coefficients,
  mix levels and muffler lengths are those of DasEtwas' `default.esc`.
- `cyber_ui`: the layer grammar and the note, gap and envelope ranges of **m1ckc3s/procedural-sounds** (MIT).
- `bearing` and `identify_machine`: constants and thresholds were set by measuring three sets of recordings,
  none of which is redistributed: the FSTF sound dataset (Mendeley Data n9y9c7xrz3), the AHU parabolic
  acoustic mirror bearing dataset (Lab-of-AMFD, no licence stated) and SUBF v2.0 (Kaggle, CC BY-NC-SA 4.0).
- `paulstretch`, `beat_repeat`, `waveset`: the algorithms of Paul Nasca's paulstretch and of
  **Soundpipe** (MIT) and **arraypress/paulstretch** (MIT).
- `pass_by`: the Harmonoise car coefficients (Jonasson et al., "Source modelling of road vehicles", 2004),
  transcribed from the table in **boschresearch/acoustic-traffic-simulation-counting** (GPL-3; the numbers
  are those of the public report).
- `turbo`, `tyre_squeal`, the wind term of `pass_by` and the default `ground` coefficient of `doppler`:
  formulas and constants read in **Agent00PED/ice-simulator**, which carries **no licence file**. Ask its
  author, or replace these constants, before a release that must be licence-clean.
- `texturize`: the model of McDermott & Simoncelli (Neuron, 2011), reduced.
