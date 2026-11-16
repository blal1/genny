import numpy as np

from genny import fx, reverbs
from genny.reverbs import feedback_guitar, phaser

SR = 44100


def _noise(n=SR, seed=0):
    return np.random.default_rng(seed).standard_normal(n) * 0.1


def test_phaser_barber_finite_and_differs():
    x = _noise()
    a = phaser(x, SR, rate=1.0, barber=False)
    b = phaser(x, SR, rate=1.0, barber=True)
    assert np.all(np.isfinite(b)) and b.shape == a.shape
    assert np.max(np.abs(a - b)) > 1e-3
    c = phaser(x, SR, rate=-1.0, barber=True)
    assert np.max(np.abs(b - c)) > 1e-3


def test_wah4_steeper_than_wah():
    x = _noise()
    y2 = fx.wah(x, SR, mode="pos", pos=0.3, order=2)
    y4 = fx.wah(x, SR, mode="pos", pos=0.3, order=4)
    assert np.all(np.isfinite(y4))
    f = np.fft.rfftfreq(SR, 1 / SR)
    s2, s4 = np.abs(np.fft.rfft(y2[:SR])), np.abs(np.fft.rfft(y4[:SR]))
    hi = f > 6000
    assert s4[hi].mean() / s4.max() < s2[hi].mean() / s2.max()


def test_feedback_guitar_cabinet():
    a = feedback_guitar(110.0, 0.4, SR)
    b = feedback_guitar(110.0, 0.4, SR, cabinet="cabinet")
    assert a.shape == b.shape and np.all(np.isfinite(b)) and np.max(np.abs(b)) > 0
    assert not np.allclose(a, b)
