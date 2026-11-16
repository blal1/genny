import numpy as np

from genny import creatures as C


def test_clap_exp_rise_peltola_eq5():
    """Attack 0.99^(140 - n) at 44.1 kHz: 0.245 -> 1 over 3.2 ms, time constant 2.26 ms."""
    assert abs(C.CLAP_EXP_TAU * 1e3 - 2.26) < 0.01
    t = np.array([0.0, C.CLAP_EXP_RISE, C.CLAP_EXP_RISE + 0.01])
    g = C.clap_rise(t, "exp")
    assert abs(g[0] - 0.99 ** (C.CLAP_EXP_RISE * 44100.0)) < 1e-6
    assert g[1] == 1.0 and g[2] == 1.0


def test_handclap_rise_exp_renders():
    y = C.handclap(rise="exp")
    assert np.isfinite(y).all() and np.abs(y).max() > 0.1
    fast = C.handclap(rise="fast")
    assert len(fast) != len(y) or not np.allclose(fast, y)
