"""Physically informed synthesis engines integrated into Genny.

This subpackage incorporates and adapts generic DSP/physical-model code from the
user-supplied ``procedural`` project (Klang Open License 1.0). See
``THIRD_PARTY_KLANG_LICENSE.txt`` at the project root for license and attribution
requirements, including the interactive-product attribution clause.

The legacy Genny API remains available.  New projects should compose the low-level
modules here through :mod:`genny.physics` and :mod:`genny.acoustics`.
"""
from .core import SR
from .modal import Modes, resolve, render_modes, strike, fracture, aeolian_tone, sword_swing
from .particles import shaker, granular_footstep_texture, bubble_field, surfacing_bubble, drip
from .space import zita_rev1, puckette_rev, farnell_room, sabine_t60, distance, air_db_per_m

__all__ = [
    "SR", "Modes", "resolve", "render_modes", "strike", "fracture", "aeolian_tone", "sword_swing",
    "shaker", "granular_footstep_texture", "bubble_field", "surfacing_bubble", "drip",
    "zita_rev1", "puckette_rev", "farnell_room", "sabine_t60", "distance", "air_db_per_m",
]
