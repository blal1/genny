import numpy as np

from genny import synths as S


def test_fat_pulse_mex_ratio_of_cosines():
    sr, f = 48000, 100.0
    t = np.arange(sr) / sr
    raw = 0.005 * (0.49 * np.cos(2 * np.pi * f * t) + 0.5) / (0.49 * np.cos(2 * np.pi * 2.01 * f * t) + 0.5)
    assert raw.max() > 0.45
    w = S.fat_pulse_wave(f, sr, sr, 2.0)
    seg = w[sr // 4: sr // 4 + sr // 2] * np.hanning(sr // 2)
    sp = np.abs(np.fft.rfft(seg))
    fr = np.fft.rfftfreq(sr // 2, 1 / sr)
    peaks = fr[np.argsort(sp)[-6:]]
    assert np.all(np.abs(peaks / 200.0 - np.round(peaks / 200.0)) < 0.02)
    assert S.FAT_PULSE == (0.49, 0.5, 2.01, 0.005, 500.0)


def test_farnell_hat_mex_fm_lengths_and_spectrum():
    import numpy as np
    from genny import drums
    sr = 48000
    h = drums.hihat(sr=sr, model="farnell")
    o = drums.openhat(sr=sr, model="farnell")
    assert h.size == round(0.05 * sr) and o.size == round(0.2 * sr)
    assert 0.2 < np.abs(h).max() <= 0.231
    S = np.abs(np.fft.rfft(o, 1 << 16))
    f = np.fft.rfftfreq(1 << 16, 1 / sr)
    assert 5000 < (S * f).sum() / S.sum() < 12000
    assert drums.hihat(sr=sr).size != h.size
