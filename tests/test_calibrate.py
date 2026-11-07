"""Offline calibration tools (genny.calibrate).

Run:  uv run --with pytest pytest tests/test_calibrate.py -q
"""
import math

import numpy as np
import pytest

from genny import calibrate as C

SR = 44100


def _string(f0, B, n=SR * 2, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for k in range(1, 51):
        fk = k * f0 * math.sqrt(1 + B * k * k)
        if fk > 0.45 * SR:
            break
        y += rng.uniform(0.5, 1.0) / k * np.sin(2 * np.pi * fk * t + rng.uniform(0, 6.3)) * np.exp(-t * (1 + 0.1 * k))
    return y + 1e-3 * rng.standard_normal(n)


def test_pfd_finds_the_inharmonicity():
    for f0, B in ((55.0, 2.5e-4), (110.0, 1e-4), (220.0, 4e-4)):
        Be, f1 = C.estimate_B(_string(f0, B), SR, f0)
        assert abs(Be / B - 1) < 0.1, (f0, B, Be)
        assert abs(f1 / (f0 * math.sqrt(1 + B)) - 1) < 1e-3


def test_esprit_and_rayleigh_recover_a_strike():
    t = np.arange(int(0.3 * SR)) / SR
    modes = ((440.0, 0.5 * (2.0 + 1e-6 * (2 * np.pi * 440) ** 2)), (1210.0, 0.5 * (2.0 + 1e-6 * (2 * np.pi * 1210) ** 2)),
             (2330.0, 0.5 * (2.0 + 1e-6 * (2 * np.pi * 2330) ** 2)))
    x = sum(np.sin(2 * np.pi * f * t) * np.exp(-d * t) for f, d in modes)
    r = C.esprit(x, SR, n_modes=3)
    for (f, d) in modes:
        i = np.argmin(np.abs(r["freqs"] - f))
        assert abs(r["freqs"][i] - f) < 0.5 and abs(r["decays"][i] / d - 1) < 0.02, (f, r["freqs"][i], r["decays"][i])
    a, b = C.rayleigh_fit(r["freqs"][:3], r["decays"][:3])
    assert abs(a - 2.0) < 0.1 and abs(b / 1e-6 - 1) < 0.05, (a, b)
    assert max(r["residual"].values()) < 1e-3


def test_texture_snr_prefers_the_same_texture():
    rng = np.random.default_rng(3)
    a = rng.standard_normal(2 * SR)
    b = rng.standard_normal(2 * SR)
    t = np.arange(2 * SR) / SR
    c = rng.standard_normal(2 * SR) * (1 + 0.9 * np.sin(2 * np.pi * 4 * t))       # a modulated noise
    assert C.texture_snr(a, b, SR)["mean"] > C.texture_snr(a, c, SR)["mean"] + 3


def test_webmushra_config(tmp_path):
    from genny.core import write_wav
    for n in ("ref", "a", "b"):
        write_wav(tmp_path / f"{n}.wav", np.zeros(100), SR)
    cfg = C.webmushra(str(tmp_path / "ref.wav"), {"model A": str(tmp_path / "a.wav"), "B": str(tmp_path / "b.wav")},
                      str(tmp_path / "test"), name="Piano check")
    text = cfg.read_text()
    assert "type: mushra" in text and "reference: configs/resources/audio/reference.wav" in text
    assert "model_A: configs/resources/audio/model_A.wav" in text and "type: finish" in text
    assert (tmp_path / "test/configs/resources/audio/B.wav").exists()


def test_roughness_of_the_calibration_tone():
    pytest.importorskip("mosqito")
    t = np.arange(0, 0.6, 1 / 48000)
    x = 0.5 * (1 + np.sin(2 * np.pi * 70 * t)) * np.sin(2 * np.pi * 1000 * t)
    assert abs(C.roughness(x, 48000) - 1.0) < 0.15                  # 1 asper by definition


def test_texture_snr_sees_c2_phase():
    from genny import cyber
    x = np.random.default_rng(1).standard_normal(4 * 48000)
    x = np.cumsum(x) * 0.01 + x * np.exp(-((np.arange(x.size) % 4800) / 500.0))   # one-sided events: C2 has a phase
    s = cyber.texture_stats(x, 48000)
    flipped = dict(s, c2=np.conj(s["c2"]))
    assert C.stats_snr(s, flipped)["c2"] < C.stats_snr(s, s)["c2"] - 40


def test_target_snr_stops_early():
    from genny import cyber
    s = cyber.texture_stats(np.random.default_rng(2).standard_normal(48000 * 2), 48000)
    calls = []
    real = cyber.stats_snr
    cyber.stats_snr = lambda a, b: calls.append(1) or real(a, b)
    try:
        cyber.texture_synth(s, 1.0, iters=5, target_snr=-1000.0)
    finally:
        cyber.stats_snr = real
    assert len(calls) == 1


def test_ir_params_read_back_a_known_decay_and_direct_ratio():
    sr = 16000
    t = np.arange(int(2.0 * sr)) / sr
    rng = np.random.default_rng(0)
    tail = rng.standard_normal(len(t)) * np.exp(-3 * math.log(10) * t / 0.8)     # T60 0.8 s in energy
    ir = 0.05 * tail
    ir[0] = 1.0
    p = C.ir_params(ir, sr)
    assert p["t30"] == pytest.approx(0.8, rel=0.05) and p["t20"] == pytest.approx(0.8, rel=0.05)
    rest = float((ir[41:] ** 2).sum())
    assert p["drr"] == pytest.approx(10 * math.log10(float((ir[:41] ** 2).sum()) / rest), abs=0.01)
