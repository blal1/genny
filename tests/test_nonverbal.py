"""Soundgen-style non-verbal vocalisations (genny.physical.nonverbal; sfx laugh, scream, cry, moan).

Run:  uv run --with pytest pytest tests/test_nonverbal.py -q
"""
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from genny.physical import nonverbal as NV
from genny.physical.core import SR


def _env(y, lp):
    return sosfiltfilt(butter(2, lp, fs=SR, output="sos"), np.abs(hilbert(y)))


def test_laugh_is_a_bout_of_syllables():
    for voice, n in (("female", 3), ("female", 6)):   # (the male preset fades its bout to -40 dB: last laugh inaudible)
        y = NV.vocalize("laugh", voice, np.random.default_rng(1), n_syl=n)
        fr = int(0.02 * SR)
        db = 20 * np.log10([np.sqrt(np.mean(y[i:i + fr] ** 2)) + 1e-9 for i in range(0, y.size - fr, fr)])
        on = db > db.max() - 28.0                                   # the pauses sit 25-35 dB down
        assert np.count_nonzero(np.diff(on.astype(int)) == 1) + on[0] == n, (voice, np.round(db))


def test_cry_carries_its_amplitude_modulation_and_scream_its_pitch():
    y = NV.vocalize("cry", "female", np.random.default_rng(2))
    e = _env(y, 400.0)[int(0.2 * SR):int(1.0 * SR)]
    S = np.abs(np.fft.rfft((e - e.mean()) * np.hanning(e.size)))
    f = np.fft.rfftfreq(e.size, 1 / SR)
    band = (f > 60) & (f < 300)
    assert abs(f[band][np.argmax(S[band])] - 125.0) < 8.0              # amFreq 125 Hz (soundgen F1 cry)
    s = NV.vocalize("scream", "female", np.random.default_rng(3))
    mid = s[int(0.3 * SR):int(0.6 * SR)]
    S = np.abs(np.fft.rfft(mid * np.hanning(mid.size)))
    f = np.fft.rfftfreq(mid.size, 1 / SR)
    lo = (f > 700) & (f < 2600)
    assert 1400 < f[lo][np.argmax(S[lo])] < 2100                         # f0 plateau 1832 -> 1618 Hz


def test_bandwidth_formula():
    b = NV.bandwidth(np.array([200.0, 499.9, 500.0, 3000.0]))
    assert abs(b[1] - b[2]) < 0.1 and abs(b[3] - 50 * 2.5) < 1e-9 and b[0] > 30
