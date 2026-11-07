import math

import numpy as np
import pytest

from genny import env


def test_power_adsr_stage_curves():
    sr = 1000
    e = env.power_adsr(1.0, attack=0.1, decay=0.2, sustain=0.5, release=0.4, sr=sr, curves=(1.75, 1.25, 1.5))
    assert e.size == 1400
    assert e[49] == pytest.approx(0.5 ** 1.75, abs=1e-12)
    assert e[99] == pytest.approx(1.0)
    assert e[199] == pytest.approx(1.0 - 0.5 * 0.5 ** 1.25, abs=1e-12)
    assert e[299] == pytest.approx(0.5)
    assert e[999 + 200] == pytest.approx(0.5 * (1.0 - 0.5 ** 1.5), abs=1e-12)
    assert e[-1] == pytest.approx(0.0, abs=1e-12)
    lin = env.power_adsr(1.0, 0.1, 0.2, 0.5, 0.4, sr=sr)
    assert lin[49] == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("window, a, x, want", [
    ("hann", None, 0.5, 0.5),
    ("blackman", None, 0.5, 0.42 - 0.08),
    ("welch", None, 0.5, 0.75),
    ("parzen", None, 0.5, 0.25),
    ("tukey", 0.5, 0.25, 0.5),
    ("poisson", None, 0.0, 1e-3),
])
def test_win_env_half_windows(window, a, x, want):
    sr = 1000
    e = env.win_env(1.0, attack=0.2, release=0.2, sr=sr, window=window, a=a)
    assert e.size == 1200
    assert e[int(round(x * 200))] == pytest.approx(want, abs=1e-9)
    assert e[500] == pytest.approx(1.0)


def test_tube_mesh_grid_and_distance_cues():
    from genny.tubes import mesh_grid, tube_mesh
    nx, ny, dw = mesh_grid(9.5, 0.45)
    assert (nx, ny) == (127, 5)
    assert dw == pytest.approx(0.0743, abs=1e-4)
    sr = 48000
    x = np.zeros(int(0.2 * sr))
    x[100] = 1.0
    near = tube_mesh(x, sr, distance=0.42, tail=0.2)
    far = tube_mesh(x, sr, distance=6.71, tail=0.2)
    assert near.shape == far.shape == (x.size + int(0.2 * sr), 2)

    def onset(y):
        return int(np.argmax(np.abs(y[:, 0]) > 0.1 * np.max(np.abs(y[:, 0]))))

    assert (onset(far) - onset(near)) / sr == pytest.approx((6.71 - 0.42) / 343.0, abs=0.003)
    drop = 10 * math.log10(np.mean(near ** 2) / np.mean(far ** 2))
    assert 0.5 < drop < 20 * math.log10(6.71 / 0.42)


def test_bank_piano_soundboard_fdn(monkeypatch):
    from genny import reverbs
    from genny.physical import bank_piano as bp
    calls = []
    real = reverbs.soundboard_fdn

    def spy(x, sr, **kw):
        calls.append(kw)
        return real(x, sr, **kw)

    monkeypatch.setattr(reverbs, "soundboard_fdn", spy)
    sr = 22050
    a = bp.note(220.0, 0.4, sr, 0.7, max_modes=30)
    b = bp.note(220.0, 0.4, sr, 0.7, max_modes=30, soundboard="fdn")
    assert len(calls) == 1
    assert np.sqrt(np.mean(b ** 2)) == pytest.approx(np.sqrt(np.mean(a ** 2)), rel=1e-6)
    assert not np.allclose(a, b)
    with pytest.raises(ValueError):
        bp.note(220.0, 0.1, sr, soundboard="plate")
