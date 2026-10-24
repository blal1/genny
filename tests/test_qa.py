"""Artefact checks across genny: aliasing of the oscillators and saturators, clicks on smooth voices,
true peak after normalising, crest factor and spectral flatness sanity.

Run:  uv run --with pytest pytest tests/test_qa.py -q     (plain asserts)
"""
import numpy as np

from genny import analysis as A
from genny import fx as FX
from genny import osc
from genny import spec as SP

SR = 48000
# f0 not commensurate with SR, so aliases do not fall back on the harmonics
F0S = (1046.5, 2093.0, 3563.2)


def test_alias_snr_of_the_oscillators():
    snr = {w: [A.alias_snr(getattr(osc, w)(f, SR, SR), SR, f) for f in F0S] for w in ("sine", "saw", "square", "triangle")}
    print({w: [round(v, 1) for v in s] for w, s in snr.items()})
    assert min(snr["sine"]) > 80
    # 2-point polyBLEP: 25-31 dB; the polyBLAMP triangle 42-60 dB (naive: 31-48)
    assert min(snr["saw"]) > 23 and min(snr["square"]) > 23
    assert snr["triangle"][0] > 57 and snr["triangle"][1] > 52 and snr["triangle"][2] > 40


def test_triangle_keeps_its_shape():
    y = osc.triangle(110.0, SR, SR)
    t = osc.phase(110.0, SR, SR) % 1.0
    err = np.abs(y - (4 * np.abs(t - 0.5) - 1))                     # the correction only touches corners:
    assert err.max() < 8 * 110.0 / SR / 6 + 1e-9                    # at most 8 dt / 6, on the sample next to one
    assert np.count_nonzero(err > 1e-12) <= 2 * 2 * 110 + 2
    h = np.abs(np.fft.rfft(osc.triangle(1000.0, SR, SR)))             # odd harmonics at 1/k^2
    assert abs(h[3000] / h[1000] - 1 / 9) < 0.01 and h[2000] < 1e-3 * h[1000]


def test_adaa_distortion_aliases_less_than_plain_tanh():
    x = np.sin(2 * np.pi * 3563.2 * np.arange(SR) / SR)
    for drive in (8.0, 30.0):
        plain = A.alias_snr(np.tanh(drive * x), SR, 3563.2)
        adaa = A.alias_snr(FX.REGISTRY["distortion"]["fn"](x, SR, drive=drive, tone=0), SR, 3563.2)
        assert adaa > plain + 4, (drive, plain, adaa)
    # same transfer curve at low frequency: the ADAA output follows tanh to within a few %
    lo = np.sin(2 * np.pi * 50 * np.arange(SR) / SR)
    y = FX.REGISTRY["distortion"]["fn"](lo, SR, drive=4.0, tone=0)
    assert np.max(np.abs(y[1:] - np.tanh(4 * lo[:-1] * 0.5 + 4 * lo[1:] * 0.5) / np.tanh(4))) < 0.02


def test_click_detector():
    x = 0.5 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)
    assert len(A.detect_clicks(x)) == 0
    x[SR // 2:] += 0.05                                                 # a 0.05 step: one click
    hits = A.detect_clicks(x)
    assert len(hits) and np.all(np.abs(hits - SR // 2) < 3)


def test_smooth_voices_have_no_clicks_and_no_intersample_overs():
    # flute is left out: its breath noise onset is spiky by design
    for inst in ("violin", "warm_pad", "glass", "cello", "pad"):
        y, sr = SP.render_spec({"layers": [{"type": "seq", "inst": inst, "steps": "C4:0.4 E4:0.4 G4:0.8"}]})
        assert len(A.detect_clicks(y)) == 0, inst
        assert A.true_peak_db(y) < 0.0, inst                            # normalize -1 dBFS never clips between samples


def test_crest_factor_and_flatness():
    t = np.arange(SR) / SR
    assert abs(A.crest_factor_db(np.sin(2 * np.pi * 440 * t)) - 3.01) < 0.05
    assert A.crest_factor_db(np.sign(np.sin(2 * np.pi * 440 * t))) < 0.1
    noise = np.random.default_rng(0).standard_normal(SR)
    assert A.spectral_flatness(noise) > 0.5 and A.spectral_flatness(np.sin(2 * np.pi * 440 * t)) < 1e-3


def test_log_attack_time_orders_attacks():
    t = np.arange(2 * SR) / SR
    lat = {a: A.log_attack_time(np.sin(2 * np.pi * 440 * t) * np.clip(t / a, 0, 1), SR) for a in (0.05, 0.5)}
    assert abs(10 ** lat[0.5] - 0.5) < 0.1 and lat[0.05] < lat[0.5] - 0.6
    got = {}
    for inst in ("marimba", "violin", "warm_pad"):
        y, sr = SP.render_spec({"layers": [{"type": "synth", "inst": inst, "notes": "C4", "dur": 1.5}]})
        got[inst] = A.log_attack_time(y, sr)
    assert got["marimba"] < got["violin"] < got["warm_pad"], got


def test_spectral_flux_separates_frozen_evolving_and_noisy_spectra():
    t = np.arange(SR) / SR
    still = A.spectral_flux(np.sin(2 * np.pi * 440 * t), SR)
    glide = A.spectral_flux(np.sin(2 * np.pi * (300 * t + 900 * t * t)), SR)
    noise = A.spectral_flux(np.random.default_rng(0).standard_normal(SR), SR)
    assert still < 1e-3 < glide < noise


def test_lf_pulse_has_the_asked_rd():
    """Wang & Gobl 2023: the transformed LF pulse must measure Up/(0.11 Ee T0) = Rd, with zero net flow."""
    from genny.physical.voice import lf_waveform
    for rd in (0.3, 0.8, 1.5, 1.71, 2.0, 2.7):
        x = lf_waveform(rd, 8192)
        flow = np.cumsum(x) / len(x)
        assert abs(flow.max() / 0.11 - rd) < 2e-3 * rd, rd
        assert abs(flow[-1]) < 1e-3, rd


def test_roughness_jitter_is_praat_local_jitter():
    """Praat local jitter (mean |T_i - T_(i-1)| / mean T) of the cycles drawn must equal the asked jitter."""
    from genny.physical.voice import SR, roughness
    f0 = np.full(SR * 20, 100.0)
    fj, _ = roughness(f0, np.random.default_rng(1), jitter=0.01, shimmer=0.0)
    ph = np.cumsum(fj) / SR
    T = np.diff(np.flatnonzero(np.diff(np.floor(ph)) > 0)).astype(float)
    assert abs(np.mean(np.abs(np.diff(T))) / T.mean() - 0.01) < 0.0015
