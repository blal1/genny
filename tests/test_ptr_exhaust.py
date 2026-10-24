"""PTR exhausts learned by Doerfler & Wyse (genny.vehicle.PTR_EXHAUST) in the `car_engine` sfx.

Run:  uv run --with pytest pytest tests/test_ptr_exhaust.py -q
"""
import numpy as np

from genny import vehicle as V
from genny.sfx import render_sfx


def test_learned_loops_are_stable_and_the_exhausts_differ():
    for res in V.PTR_EXHAUST.values():
        for L, a1, a2 in res:
            assert 2 < L < 130 and abs(a1) + abs(a2) < 0.999          # stable for any delay (their bound)
    ys = [render_sfx("car_engine", 16000, model="ptr", exhaust=e, cylinders=8, rpm=2400, dur=1.0)
          for e in V.PTR_EXHAUST]
    for y in ys:
        assert np.all(np.isfinite(y)) and np.max(np.abs(y)) > 0.01
    spec = [np.abs(np.fft.rfft(y))[:4000] for y in ys]
    spec = [s / np.linalg.norm(s) for s in spec]
    assert max(float(spec[i] @ spec[j]) for i in range(3) for j in range(i + 1, 3)) < 0.98
