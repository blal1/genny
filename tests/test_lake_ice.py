"""Air-coupled flexural tone of floating ice (Romeyn et al. 2021, The Cryosphere 15:2939)."""
import numpy as np

from genny import matter as M
from genny.physical.core import SR


def test_acfw_matches_skating_recording():
    # Romeyn sect. 6: 725 Hz in the National Geographic skating film -> 4.3 cm at c = 329 m/s
    assert abs(M.acfw_frequency(0.043, 329.0) - 725.0) < 15.0
    assert M.acfw_frequency(0.02) > M.acfw_frequency(0.1) > M.acfw_frequency(0.5)


def test_ice_sing_is_a_precursor_at_the_coupled_frequency():
    y = M.ice_sing(60.0, 0.05)
    f = np.fft.rfftfreq(len(y), 1.0 / SR)
    assert abs(f[np.argmax(np.abs(np.fft.rfft(y)))] / M.acfw_frequency(0.05, M.C_AIR) - 1.0) < 0.05
    air = int(0.3 * SR)
    assert np.abs(y[:air]).max() > 10 * np.abs(y[air + int(0.02 * SR):]).max()    # rises before the air wave, not after


def test_ice_crack_renders():
    y = M.ice_crack(dur=1.0, sing=1.0)
    assert np.isfinite(y).all() and np.abs(y).max() > 0.1
