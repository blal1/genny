"""Few's string-of-pearls thunder (genny.physical.ambience.thunder_channel) and the `thunder` sfx.

Run:  uv run --with pytest pytest tests/test_thunder_channel.py -q
"""
import numpy as np

from genny.physical import ambience as amb
from genny.physical.core import SR


def test_channel_peaks_in_the_measured_band_and_lasts_the_range_spread():
    y = amb.thunder_channel(np.random.default_rng(1), dur=14.0, distance_m=1000.0)
    S = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    # smoothed over 10 Hz: the ripples from the tortuosity are not the peak
    Sm = np.convolve(S, np.ones(int(10 / f[1])) / int(10 / f[1]), "same")
    fp = f[np.argmax(Sm[f > 5]) + np.count_nonzero(f <= 5)]
    assert 30 < fp < 90, fp                                           # 40-80 Hz for ground flashes
    # a 4 km channel 1 km away: arrivals spread over less than (sqrt(1^2 + 4^2) - 1) km / c = 9 s (it winds)
    last = np.flatnonzero(np.abs(y) > 10 ** (-50 / 20))[-1] / SR
    assert 4.0 < last < 9.5, last
    # close by, the near segments dominate: the crack comes first
    near = amb.thunder_channel(np.random.default_rng(1), dur=14.0, distance_m=50.0)
    assert np.argmax(np.abs(near)) < 0.5 * SR


def test_sfx_distance_is_metres():
    from genny.sfx import render_sfx
    for d in (80, 1000, 5000):
        y = render_sfx("thunder", 22050, dur=1.0, distance=d, seed=1)
        assert np.all(np.isfinite(y)) and 0 < np.max(np.abs(y)) <= 1.0
