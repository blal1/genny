import numpy as np

from genny import compose, dsp
from genny.fx import REGISTRY as FX


def _voiced(sr=8000, f0=140.0, n=8000):
    t = np.arange(n) / sr
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    return 0.5 * x / np.max(np.abs(x))


def _snr(x, y):
    return 10 * np.log10(np.sum(x ** 2) / np.sum((x - y) ** 2))


def test_gsm_ltp_improves_periodic_voice_and_default_unchanged():
    x = _voiced()
    a, b = dsp.gsm_like(x), dsp.gsm_like(x, ltp=True)
    assert np.all(np.isfinite(b))
    assert _snr(x, b) > _snr(x, a) + 3.0
    assert np.array_equal(a, dsp.gsm_like(x, ltp=False))


def test_gsm_bitrate_is_13_kbit():
    assert dsp.gsm_bitrate(3) == 13000.0
    assert dsp.gsm_bitrate(2) < dsp.gsm_bitrate(4)


def test_codec_gsm_ltp_option_and_gsm_call():
    x = _voiced(8000, 140.0, 16000)
    y = FX["codec"]["fn"](x, 8000, kind="gsm", ltp=True)
    assert y.shape == x.shape
    z = FX["gsm_call"]["fn"](x, 8000, channel="poor", seed=1)
    assert np.all(np.isfinite(z)) and z.shape[0] == x.shape[0] + int(round(0.024 * 8000))
    assert np.all(z[: int(0.024 * 8000)] == 0.0)


def test_harmonizer_series_intervals_and_draw():
    x = _voiced(16000, 200.0, 16000)
    y = FX["harmonizer"]["fn"](x, 16000, series="4 5 6", mix=1.0)
    s = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / 16000)
    peak = lambda c: s[(f > c * 0.97) & (f < c * 1.03)].max()
    assert peak(250.0) > 3 * peak(225.0) and peak(300.0) > 3 * peak(275.0)
    z = FX["harmonizer"]["fn"](x, 16000, series="4 5 6", draw=2, seed=3)
    assert np.all(np.isfinite(z))


def _ev(role, **kw):
    return compose.generate({"type": "compose", "role": role, "bars": 2, "seed": 4, "bpm": 120, **kw}, 120)


def test_vel_law_beta_is_bimodal_and_in_range():
    ev = compose.generate({"type": "euclid", "kind": "kick", "k": 16, "n": 16, "bars": 8, "seed": 2, "vel_law": "beta",
                           "vel_range": {"kick": [110, 127]}}, 120)
    v = np.array([e["vel"] for e in ev]) * 127
    assert v.min() >= 109.9 and v.max() <= 127.01
    assert np.mean((v < 114) | (v > 123)) > 0.6


def test_gate_scales_chord_durations():
    a, b = _ev("chords", gate=0.92), _ev("chords", gate=0.46)
    da, db = np.mean([e["dur"] for e in a]), np.mean([e["dur"] for e in b])
    assert abs(db / da - 0.5) < 0.05


def test_figures_onset_spacing_in_range():
    for name, (_, (lo, hi), acc) in compose.FIGURES.items():
        ev = compose.generate({"type": "figure", "name": name, "seed": 1}, None)
        t = np.array([e["t"] for e in ev]) * 1000.0
        gap0 = t[1] - t[0]
        assert lo - 0.5 <= gap0 <= hi + 0.5, name
        if acc:
            assert np.all(np.diff(np.diff(t)) < 0)
    ev = compose.generate({"type": "figure", "name": "rise-three", "note": "C5", "step": 4}, None)
    assert [round(e["pitch"] - ev[0]["pitch"]) for e in ev] == [0, 4, 8]
