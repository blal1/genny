import numpy as np

from genny import analysis, fx, friction

SR = 16000


def test_virtual_bass_adds_harmonics_and_cut_removes_fundamental():
    t = np.arange(SR) / SR
    x = 0.5 * np.sin(2 * np.pi * 60 * t)
    y = fx.virtual_bass(x, SR, fc=120.0, amount=0.6, cut=True)
    X = np.abs(np.fft.rfft(y))
    assert np.all(np.isfinite(y))
    assert X[180] > 10 * X[60]


def test_estimate_f0_sub_half_bin():
    t = np.arange(4096) / SR
    f = 400.0
    x = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 9))
    assert abs(analysis.estimate_f0(x, SR) - f) < 2.0
    y = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(2, 9))      # missing fundamental
    assert abs(analysis.estimate_f0(y, SR) - f) < 2.0


def test_rod_bank_radiator_size_cuts_low_fundamental():
    big = friction.rod_bank(60.0, 0.5, SR, radiator="cone", radiator_size=2.0)
    small = friction.rod_bank(60.0, 0.5, SR, radiator="cone", radiator_size=0.1)
    lo = lambda y: (lambda X, h: X[int(50 * h):int(70 * h)].sum() / X.sum())(np.abs(np.fft.rfft(y)), len(y) / SR)
    assert lo(small) < 0.5 * lo(big)
