import numpy as np

from genny import friction as FR


def test_thoret_singing_glass_beat_is_v_over_pi_d():
    sr, d, v = 8000, 0.08, 0.5
    plain = FR.thoret_friction(440.0, 2.0, sr, eps=0.0)
    beat = FR.thoret_friction(440.0, 2.0, sr, eps=0.0, diameter=d, speed=v)
    m = np.abs(plain) > 0.5
    t = (np.arange(plain.size) + 1) / sr
    assert np.allclose(beat[m] / plain[m], np.sin(2 * np.pi * v * t[m] / (np.pi * d)), atol=1e-6)
