import numpy as np

from genny import arms as A
from genny import dsp as D


def test_rifle_crack_spectrum_in_measured_band():
    # supersonic rifle bullet passing 3-10 m away: spectral peak of the N-wave inside 2.5-7.5 kHz (the measurement
    # distance is not given; the peak falls with the miss distance, 2.4 kHz at 50 m)
    sr = 1_000_000
    peaks = []
    for miss in (3.0, 10.0, 50.0):
        _, T = A.n_wave_pa(948.0, 5.7, 23.0, miss)
        x = np.zeros(1 << 16)
        x[:int(T * sr)] = np.linspace(1.0, -1.0, int(T * sr))
        peaks.append(np.fft.rfftfreq(x.size, 1 / sr)[np.argmax(np.abs(np.fft.rfft(x)))])
    assert all(2500.0 < f < 7500.0 for f in peaks[:2]), peaks
    assert peaks[0] > peaks[1] > peaks[2]


def test_coincidence_band_dip_and_rw_presets():
    m, fc = 10.0, 3800.0
    r0 = 20 * np.log10(fc * m) - 42.0
    assert abs(D.wall_loss_db(fc, m, 200.0, fc) - (r0 + 10 * np.log10(2 * 0.03 * 0.2316 / np.pi))) < 1e-3
    assert abs(D.wall_loss_db(1.1 * fc, m, 200.0, fc) - (20 * np.log10(1.1 * fc * m) - 42.0 + 10 * np.log10(2 * 0.03 * 0.2316 / np.pi))) < 1e-3
    assert D.wall_rw("concrete") == 50 and D.wall_rw("drywall") == 43
