"""Chaigne & Touze 1:1:2 internal-resonance triad behind `gong cascade=` (genny.percussion.triad).

Run:  uv run --with pytest pytest tests/test_gong_triad.py -q
"""
import numpy as np

from genny.percussion import triad
from genny.drums import render_drum

SR = 48000


def test_twins_bloom_only_above_the_threshold_and_take_energy_from_the_parent():
    peak = {}
    for a3 in (0.2, 1.0):
        tw, par = triad(113.0, 2 * SR, SR, a3, 6.0)
        peak[a3] = np.max(np.abs(tw[int(0.4 * SR):int(1.0 * SR)])) / a3
    assert peak[0.2] < 0.01 and peak[1.0] > 0.3, peak
    tw, par = triad(113.0, 2 * SR, SR, 1.0, 6.0)
    early = np.max(np.abs(tw[:int(0.1 * SR)]))
    assert early < 0.02                                             # delayed: not there at the hit
    # the parent gives energy away (and takes some back): its envelope dips far below its own decay
    ratio = [np.max(np.abs(par[int(t * SR):int((t + 0.05) * SR)])) / np.exp(-6.91 / 6.0 * t)
             for t in np.arange(0.4, 1.2, 0.05)]
    assert min(ratio) < 0.4, ratio


def test_gong_cascade_param_renders():
    y0 = render_drum("gong", SR, 1.0, tune=90, decay=3.0)
    y1 = render_drum("gong", SR, 1.0, tune=90, decay=3.0, cascade=1.0)
    assert np.all(np.isfinite(y1)) and not np.allclose(y0, y1)
