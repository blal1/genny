"""Guillemain digital-impedance clarinet (genny.physical.impedance; `clarinet` with model=impedance).

Run:  uv run --with pytest pytest tests/test_impedance_clarinet.py -q
"""
import math

import numpy as np

from genny.physical import impedance as I

SR = 48000.0


def test_loss_filter_is_exact_at_the_first_two_peaks():
    L = 0.5
    a1, b0 = I.loss_filter(L, SR)
    alpha = 2.0 / (I.BORE_R * I.C ** 1.5) * (math.sqrt(I.LV) + (I.CPCV - 1.0) * math.sqrt(I.LT))
    for w in (math.pi * I.C / (2 * L), 3 * math.pi * I.C / (2 * L)):
        exact = math.exp(-2.0 * alpha * I.C * math.sqrt(w / 2.0) * L)          # |F(w)^2|^2
        approx = b0 * b0 / (1 + a1 * a1 - 2 * a1 * math.cos(w / SR))
        assert abs(approx / exact - 1) < 1e-9


def _spec(y, f0):
    S = np.abs(np.fft.rfft(y * np.hanning(y.size), 1 << 17))
    f = np.fft.rfftfreq(1 << 17, 1 / SR)
    return [S[np.argmin(np.abs(f - k * f0))] for k in (1, 2, 3, 4, 5)]


def test_in_tune_odd_harmonics_and_brighter_when_blown_harder():
    for freq in (146.83, 261.63, 392.0, 523.25, 554.37, 659.25, 1046.5):   # C5-C#5: the twelfth trap
        soft = I.clarinet(freq, 0.7, SR, blow=0.0)[int(0.4 * SR):int(0.7 * SR)]
        hard = I.clarinet(freq, 0.7, SR, blow=1.0)[int(0.4 * SR):int(0.7 * SR)]
        for y in (soft, hard):
            f0 = I._f0(y, SR, freq)
            assert abs(1200 * math.log2(f0 / freq)) < 10, freq
        if freq < 400:                                                       # chalumeau: even harmonics weak
            h = _spec(hard, freq)
            assert h[1] < 0.2 * h[0] and h[1] < 0.2 * h[2], freq
        assert _centroid(hard) > 1.05 * _centroid(soft), freq


def _centroid(y):
    S = np.abs(np.fft.rfft(y * np.hanning(y.size))) ** 2
    return float(np.sum(np.fft.rfftfreq(y.size, 1 / SR) * S) / np.sum(S))
