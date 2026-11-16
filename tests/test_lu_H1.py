import numpy as np

from genny import filters as F


def _gain_at(y, x):
    n = len(x) // 2
    return np.sqrt(np.mean(y[n:] ** 2) / np.mean(x[n:] ** 2))


def test_levien_linear_ladder_is_minus_12_db_at_cutoff_without_resonance():
    sr, fc = 48000, 1000.0
    t = np.arange(sr) / sr
    x = 1e-3 * np.sin(2 * np.pi * fc * t)
    g = _gain_at(F.ladder(x, fc, sr, res=0.0, kind="levien_linear"), x)
    assert abs(20 * np.log10(g) + 12.04) < 0.2


def test_levien_ladder_is_the_exact_sample_and_hold_ladder_near_nyquist():
    from scipy.signal import cont2discrete, dlti, dfreqresp
    sr, fc, k = 48000, 12000.0, 2.0
    w = 2 * np.pi * fc
    A = np.array([[-w, 0, 0, -k * w], [w, -w, 0, 0], [0, w, -w, 0], [0, 0, w, -w]])
    B = np.array([[w], [0], [0], [0]])
    C = np.array([[0, 0, 0, 1.0]])
    Ad, Bd, Cd, Dd, _ = cont2discrete((A, B, C, np.zeros((1, 1))), 1 / sr, method="zoh")
    want = abs(dfreqresp(dlti(Ad, Bd, Cd, Dd, dt=1 / sr), w=[2 * np.pi * fc / sr])[1][0])
    t = np.arange(sr) / sr
    x = 1e-3 * np.sin(2 * np.pi * fc * t)
    g = _gain_at(F.ladder(x, fc, sr, res=k / 4, kind="levien_linear"), x)
    assert abs(20 * np.log10(g / want)) < 0.1


def test_levien_nonlinear_matches_linear_at_small_signal():
    sr = 48000
    x = 1e-4 * np.random.default_rng(0).standard_normal(sr // 4)
    a = F.ladder(x, 2000.0, sr, res=0.6, kind="levien_linear")
    b = F.ladder(x, 2000.0, sr, res=0.6, kind="levien")
    assert np.max(np.abs(a - b)) < 1e-3 * np.max(np.abs(a))


def test_playhead_loss_is_spacing_thickness_and_gap_losses():
    from genny import dsp as D
    k = 2 * np.pi * 1000.0 / (7.5 * 0.0254)
    want = 20 * np.log10(np.exp(-k * 20e-6) * (1 - np.exp(-k * 35e-6)) / (k * 35e-6) * np.sinc(k * 5e-6 / (2 * np.pi)))
    assert abs(D.playhead_loss_db(1000.0, 7.5) - want) < 1e-9
    assert abs(want + 10.28) < 0.05
    assert D.playhead_loss_db(1000.0, 15.0) > D.playhead_loss_db(1000.0, 3.75) + 10


def test_tape_speed_applies_the_playhead_loss_on_a_sine():
    from genny import dsp as D
    sr = 48000
    t = np.arange(sr) / sr
    x = 0.01 * np.sin(2 * np.pi * 1000.0 * t)
    y = D.playhead_loss(x, sr, 7.5)
    assert abs(20 * np.log10(_gain_at(y, x)) - D.playhead_loss_db(1000.0, 7.5)) < 0.5


def test_esr_is_error_energy_over_target_energy():
    from genny import calibrate as C
    y = np.sin(np.linspace(0, 20, 1000))
    assert C.esr(y, y) == 0.0
    assert abs(C.esr(np.zeros(1000), y) - 1.0) < 1e-12
    assert abs(C.esr(0.9 * y, y) - 0.01) < 1e-12
