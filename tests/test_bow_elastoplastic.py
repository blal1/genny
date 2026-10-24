"""Elasto-plastic bowed string (genny.physical.bow; `cello` with model=elastoplastic).

Run:  uv run --with pytest pytest tests/test_bow_elastoplastic.py -q
"""
import math

import numpy as np

from genny.physical import bow
from genny.physical.waveguides import measure_f0

SR = 48000.0


def test_adhesion_map_regimes():
    P = bow.CURVES["S"]
    args = (2.0, P["s0"], P["vs"], P["eps"], P["muC"], P["muS"])
    zba = 0.7 * P["muC"] * 2.0 / P["s0"]
    v = 0.05
    assert bow._zdot(v, 0.5 * zba, *args) == v                       # elastic pre-sliding: dz/dt = v
    zs = bow._zss(v, *args)
    assert abs(bow._zdot(v, zs, *args)) < 1e-12                     # plastic steady state: z = z_ss
    assert bow._zdot(v, -0.5 * zs, *args) == v                       # sign(z) != sign(v): a = 0


def test_helmholtz_pitch_and_stability_over_the_guettler_range():
    for curve in ("S", "eps"):
        for f in (98.0, 196.0, 392.0):
            for fN, a in ((0.8, 0.35), (2.0, 1.0), (3.9, 2.0)):
                y = bow.bowed(f, 0.8, SR, force=fN, speed=0.1 + 0.05 * fN, accel=a, curve=curve)
                assert np.all(np.isfinite(y)), (curve, f, fN)
                m = measure_f0(y[int(0.5 * SR):int(0.78 * SR)], f)
                assert abs(1200 * math.log2(m / f)) < 35, (curve, f, fN, m)
