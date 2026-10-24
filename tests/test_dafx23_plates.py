"""Steel plates calibrated on Bilbao et al., "Real-time gong synthesis", DAFx-23 (genny.plates steel_cymbal, steel_gong).

Run:  uv run --with pytest pytest tests/test_dafx23_plates.py -q
"""
import numpy as np

from genny.plates import STEEL, nl_plate

SR = 44100


def _hf(y, t0, t1):
    s = y[int(t0 * SR):int(t1 * SR)]
    S = np.abs(np.fft.rfft(s)) ** 2
    return S[np.fft.rfftfreq(s.size, 1 / SR) > 3000].sum() / S.sum()


def _plate(size, ms, force, **kw):
    return nl_plate(SR, 1.0, size, 0.5, force, ms, (0.38, 0.45), 20.0, 0.0, "vk", eps=1.4, material=STEEL,
                    loss=(20.0, 1000.0, 10.0), dur=1.2, hp=30.0, **kw)


def test_small_plate_rings_clean_then_crashes():
    # their Fig. 3: distinct modes at low amplitude, wideband noise (the crash) at high amplitude
    assert _hf(_plate(0.113, 2.0, 2.0), 0.3, 0.4) < 0.01
    assert _hf(_plate(0.113, 2.0, 80.0), 0.3, 0.4) > 0.2


def test_large_plate_swells_and_the_scanned_output_moves():
    # their Fig. 4: energy migrates to the high range over the first few hundred ms
    y = _plate(0.451, 4.0, 60.0)
    assert _hf(y, 0.3, 0.4) > 3 * _hf(y, 0.0, 0.1)
    fixed = _plate(0.113, 2.0, 2.0)
    moving = _plate(0.113, 2.0, 2.0, scan=(0.4, 1.0))
    assert np.all(np.isfinite(moving)) and not np.allclose(fixed, moving)
