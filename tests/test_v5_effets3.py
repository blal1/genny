import math
import wave

import numpy as np

from genny import spectral as SP
from genny.analysis import yin
from genny.core import write_wav

SR = 16000


def _read_ints(path):
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(int)


def test_write_wav_default_is_plain_truncation(tmp_path):
    x = np.linspace(-0.9, 0.9, 1001)
    a = _read_ints(write_wav(tmp_path / "a.wav", x, SR))
    assert np.array_equal(a, (x * 32767).astype("<i2").astype(int))


def test_write_wav_tpdf_dither_linearises_below_one_lsb(tmp_path):
    x = np.full(40000, 0.3 / 32768)
    plain = _read_ints(write_wav(tmp_path / "p.wav", x, SR))
    d1 = _read_ints(write_wav(tmp_path / "d1.wav", x, SR, dither="tpdf", seed=3))
    d2 = _read_ints(write_wav(tmp_path / "d2.wav", x, SR, dither="tpdf", seed=3))
    assert not plain.any()
    assert np.array_equal(d1, d2)
    assert abs(d1.mean() - 0.3) < 0.03
    assert set(np.unique(d1)) <= {-1, 0, 1, 2}


def _pulses(f0, dur, sr=SR):
    n = int(dur * sr)
    t = np.arange(n) / sr
    return sum(np.cos(2 * np.pi * k * f0 * t) for k in range(1, int(0.45 * sr / f0))) * 0.05


def _f0(y, sr=SR):
    f = yin(y, sr, fmin=80, fmax=900)
    return float(np.median(f[f > 0]))


def test_pitch_marks_one_per_period():
    m, v = SP.pitch_marks(_pulses(200, 0.5), SR, f0=200)
    assert v.all()
    assert np.median(np.diff(m)) == SR // 200


def test_psola_shifts_pitch_and_keeps_length():
    x = _pulses(200, 0.6)
    y = SP.psola(x, SR, 12.0, f0=200)
    assert len(y) == len(x)
    assert abs(_f0(y[2000:-2000]) / 400.0 - 1) < 0.03


def test_psola_stretches_time_and_keeps_pitch():
    x = _pulses(200, 0.5)
    y = SP.psola(x, SR, 0.0, 2.0, f0=200)
    assert len(y) == 2 * len(x)
    assert abs(_f0(y[2000:-2000]) / 200.0 - 1) < 0.03


def test_epoch_sharpness_threshold_separates_pulses_from_sine():
    t = np.arange(4000) / SR
    sine = np.sin(2 * np.pi * 200 * t)
    assert abs(SP.epoch_sharpness(sine, SR, 200) - (math.sqrt(2) - 1)) < 0.02
    assert SP.epoch_sharpness(_pulses(200, 0.25), SR, 200) > SP.SHARP_PSOLA


def test_psola_mask_follows_sharpness_with_hysteresis():
    t = np.arange(int(0.5 * SR)) / SR
    assert SP.psola_mask(np.sin(2 * np.pi * 200 * t), SR, f0=200).max() == 0.0
    w = SP.psola_mask(_pulses(200, 0.5), SR, f0=200)
    assert w[: int(0.02 * SR)].max() < 0.5 and w[-int(0.1 * SR):].min() > 0.99


def test_timestretch_and_pitch_shift_modes_default_unchanged():
    from genny.fx import REGISTRY
    assert REGISTRY["timestretch"]["params"]["mode"][0] == "vocoder"
    assert REGISTRY["pitch_shift"]["params"]["mode"][0] == "vocoder"
    x = _pulses(200, 0.4)
    assert np.array_equal(SP.pitch_shift(x, SR, 3.0), SP.shift_pitch(x, SR, 3.0, "shift", "vocoder"))
    assert len(SP.stretch(x, SR, 1.5, "psola")) == round(1.5 * len(x))


def test_grains_sola_crossfades_sum_to_unity():
    x = np.ones(SR)
    y = SP.grains(x, SR, density=20, size=0.08, spray=0.0, position=0.05, spread=0.0, overlap="sola")
    assert y.ndim == 1
    assert np.allclose(y[int(0.5 * SR):int(0.9 * SR)], 1.0, atol=1e-9)


def test_grains_normal_overlap_scaled_by_density_times_size():
    x = np.ones(SR)
    y = SP.grains(x, SR, density=20, size=0.08, spray=0.0, position=0.05, spread=0.0)
    mean_hann_sum = 0.08 * 20 * 0.5
    assert abs(y[int(0.5 * SR):int(0.9 * SR)].mean() - mean_hann_sum / math.sqrt(1.6)) < 0.01


def test_grains_freeze_holds_the_buffer():
    t = np.arange(int(1.5 * SR)) / SR
    x = np.sin(2 * np.pi * 440 * t) * (t < 0.5)
    late = slice(int(1.2 * SR), int(1.5 * SR))
    held = SP.grains(x, SR, spray=0.0, spread=0.0, freeze=0.5)
    free = SP.grains(x, SR, spray=0.0, spread=0.0)
    assert np.sqrt(np.mean(held[late] ** 2)) > 0.1
    assert np.abs(free[late]).max() < 1e-9


def test_grains_pan_law():
    y = SP.grains(np.ones(SR), SR, spray=0.0, spread=1.0, overlap="sola", seed=1)
    assert y.ndim == 2
    seg = y[int(0.5 * SR):int(0.9 * SR)]
    power = seg[:, 0] ** 2 + seg[:, 1] ** 2
    assert abs(power.max() - 1.0) < 1e-6 and power.min() <= 1.0 + 1e-6
    assert np.abs(seg[:, 0] - seg[:, 1]).max() > 0.1


def test_granulate_draws_grains_near_current_time():
    t = np.arange(SR) / SR
    x = np.sin(2 * np.pi * 300 * t) * (t < 0.5)
    near = SP.granulate(x, SR, grain=0.02, window=0.05)
    anywhere = SP.granulate(x, SR, grain=0.02, window=0.0)
    late = slice(int(0.7 * SR), int(0.95 * SR))
    early = slice(int(0.1 * SR), int(0.4 * SR))
    assert len(near) == len(x)
    assert np.sqrt(np.mean(near[late] ** 2)) < 0.05 * np.sqrt(np.mean(near[early] ** 2))
    assert np.sqrt(np.mean(anywhere[late] ** 2)) > 0.2 * np.sqrt(np.mean(anywhere[early] ** 2))


def test_feedback_guitar_declares_aperture():
    import genny.reverbs  # noqa: F401
    from genny.instruments import REGISTRY
    p = REGISTRY["feedback_guitar"]["params"]["aperture"]
    assert p[0] == 0.0
