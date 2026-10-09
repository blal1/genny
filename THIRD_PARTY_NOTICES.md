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
