"""Analysis, QA and evaluation measures: each one checked against a sourced figure or an analytic value.

Run:  uv run --with pytest pytest tests/test_analyse.py -q
"""
import math
from pathlib import Path

import numpy as np
import pytest

from genny import analysis as A
from genny import calibrate as C
from genny import identify as I
from genny import spec as SP

REC = Path(__file__).resolve().parents[2] / "recordings" / "voice_sung"
VOWEL = REC / "vowel_a__ll_q1860_eng_flame_not_lame_natural_vowe.flac"


def _render(layer, sr=44100):
    y, sr = SP.render_spec({"sr": sr, "layers": [layer]})
    return (y if y.ndim == 1 else y.mean(axis=1)), sr


# ------------------------------------------------------------------ HNR
def test_hnr_reads_the_signal_to_noise_ratio_of_a_periodic_tone():
    """Praat manual (Voice 4, Harmonicity): for a periodic signal plus noise the HNR is the SNR."""
    sr = 16000
    t = np.arange(sr) / sr
    saw = sum(np.sin(2 * np.pi * 200 * k * t) / k for k in range(1, 30))
    saw /= np.sqrt(np.mean(saw ** 2))
    rng = np.random.default_rng(0)
    for snr in (0.0, 10.0, 20.0, 30.0):
        p = A.periodicity(saw + rng.standard_normal(sr) * 10 ** (-snr / 20), sr)
        assert abs(p["hnr"] - snr) < 1.0, (snr, p["hnr"])
        assert abs(p["f0"] - 200.0) < 2.0
    assert A.hnr(saw, sr) > 60.0                                  # Praat: measurable to 90 dB on clean signals
    assert A.hnr(rng.standard_normal(sr), sr) < 0.0


def test_genny_voice_hnr_against_finger_and_a_recording():
    """Measured, not tuned here: genny's solo voice reads about 28.5 dB at A3 and A4, above Finger et al. 2009
    (women, Praat: 19.3 +- 3.7 dB, max 26.5). Reported to the Voix theme; the test pins the measurement."""
    for note in ("A3", "A4"):
        y, sr = _render({"type": "synth", "inst": "voice", "notes": note, "dur": 2.0})
        assert 26.5 < A.hnr(y, sr) < 30.5, note
    if VOWEL.exists():
        from genny.core import read_wav
        x, sr = read_wav(VOWEL)
        x = x if x.ndim == 1 else x.mean(axis=1)
        h = A.hnr(x, sr)
        assert 12.6 < h < 26.5                                    # a real sung /a/: inside Finger's range (15.0)


# ------------------------------------------------------------------ clicks
def test_lpc_click_detector_ignores_struck_attacks_and_finds_spikes():
    sr = 44100
    x = 0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
    assert len(A.detect_clicks_lpc(x)) == 0
    x[sr // 2] += 0.05
    assert list(A.detect_clicks_lpc(x)) == [sr // 2]
    for inst in ("piano", "marimba", "guitar"):
        y, sr = _render({"type": "synth", "inst": inst, "notes": "C4 E4", "dur": 0.8})
        assert len(A.detect_clicks_lpc(y)) == 0, inst
        y[20000] += 0.1
        hits = A.detect_clicks_lpc(y)
        assert len(hits) == 1 and abs(hits[0] - 20000) <= 1, (inst, hits)


# ------------------------------------------------------------------ loudness range
@pytest.mark.parametrize("levels,lra", [((-20, -30), 10), ((-20, -15), 5), ((-40, -20), 20),
                                        ((-50, -35, -20, -35, -50), 15)])
def test_loudness_range_passes_ebu_tech_3342_minimum_requirements(levels, lra):
    """Tech 3342-2023 Table 1, cases 1-4: stereo 1 kHz tones, 20 s each, LRA within +-1 LU."""
    sr = 16000
    s = np.sin(2 * np.pi * 1000 * np.arange(20 * sr) / sr)
    m = np.concatenate([10 ** (l / 20) * s for l in levels])
    assert abs(A.loudness_range(np.stack([m, m], axis=1), sr) - lra) < 1.0
    assert math.isnan(A.loudness_range(m[: 2 * sr], sr))


# ------------------------------------------------------------------ descriptors
def test_harmonic_descriptors_on_analytic_spectra():
    sr, f0 = 44100, 220.0
    t = np.arange(sr) / sr
    saw = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 21))
    d = A.harmonic_descriptors(saw, sr, f0)
    s = sum(1 / k for k in range(1, 21))
    assert abs(d["t1"] - 1 / s) < 1e-3 and abs(d["t2"] - (1 / 2 + 1 / 3 + 1 / 4) / s) < 1e-3
    oer = sum(1 / k ** 2 for k in range(1, 21, 2)) / sum(1 / k ** 2 for k in range(2, 21, 2))
    assert abs(d["oer"] - oer) < 0.01 and d["noisiness"] < 0.01 and d["inharmonicity"] < 1e-4
    sq = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 21, 2))
    assert A.harmonic_descriptors(sq, sr, f0)["oer"] > 1e3       # clarinet-like: odd only
    B = 1e-3                                                      # stiff string f_k = k f0 sqrt(1 + B k^2)
    fk = [f0 * k * math.sqrt(1 + B * k * k) for k in range(1, 11)]
    stiff = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate(fk, 1))
    a2 = np.array([1 / k ** 2 for k in range(1, 11)])
    want = sum(abs(f - round(f / f0) * f0) * a for f, a in zip(fk, a2)) / (a2.sum() * f0)
    assert abs(A.harmonic_descriptors(stiff, sr, f0, H=10)["inharmonicity"] - want) < 0.05 * want
    noisy = saw / np.sqrt(np.mean(saw ** 2)) + np.random.default_rng(0).standard_normal(sr)
    assert abs(A.harmonic_descriptors(noisy, sr, f0)["noisiness"] - 0.5) < 0.03    # 0 dB SNR: half the energy


def test_effective_duration_strong_decay_and_centroid_spread():
    sr = 44100
    t = np.arange(sr) / sr
    tau = 0.2
    want = tau * math.log(10 ** (3 / 20))                          # power envelope 3 dB down at tau ln 10^(3/20)
    assert abs(A.effective_duration(np.exp(-t / tau) * np.sin(2 * np.pi * 440 * t), sr) - want) < 0.005
    assert abs(A.effective_duration(np.where(t < 0.5, np.sin(2 * np.pi * 440 * t), 0), sr) - 0.5) < 0.01
    e = np.exp(-np.arange(10 * sr) / sr / tau)                    # energy sr tau / 2, |x| centroid tau
    assert abs(A.strong_decay(e, sr) - math.sqrt(sr / 2)) < 0.01 * math.sqrt(sr / 2)
    steady = A.centroid_iqr(np.sin(2 * np.pi * 440 * t), sr)
    glide = A.centroid_iqr(np.sin(2 * np.pi * (200 * t + 2000 * t * t)), sr)
    assert steady < 20 and 1500 < glide < 2300                    # the glide's centroid quartiles: ~1200 .. ~3200 Hz


def test_describe_carries_the_new_descriptors():
    sr = 44100
    n = np.random.default_rng(1).standard_normal(sr)
    d = A.describe(np.stack([n, -0.5 * n], axis=1) * 0.1, sr)
    assert d["corr"] == pytest.approx(-1.0)                    # fold-down 0.25 n against mean power 0.625 n^2:
    assert d["mono_loss_db"] == pytest.approx(10 * math.log10(0.0625 / 0.625))     # -10 dB
    assert math.isnan(A.centroid_iqr(np.zeros(sr), sr)) and A.effective_duration(np.zeros(sr), sr) == 0.0
    assert abs(d["effective_duration"] - 1.0) < 0.02 and d["centroid_iqr"] > 0
    assert "corr" not in A.describe(n, sr)


def test_modal_fit_frequencies_are_interpolated_between_bins():
    """Half a bin is 5.9 Hz at 48 kHz, N 4096 (23 cents at 400 Hz, Addendum §2): the fit must do much better."""
    from genny.physical.analysis import SR, modal_fit
    t = np.arange(SR) / SR
    x = sum(np.exp(-8 * t) * np.sin(2 * np.pi * f * t) for f in (403.0, 1131.0, 2717.0))
    got = np.sort(modal_fit(x, n_modes=3).freqs)
    assert np.all(np.abs(got - [403.0, 1131.0, 2717.0]) < 1.0), got


# ------------------------------------------------------------------ stereo
def test_interchannel_corr_and_mono_loss_of_a_haas_delay():
    """Addendum §2: a 6 ms delay between channels folds to mono as a comb at 83.3, 250, 416.7 Hz, ~24 dB deep."""
    sr = 44100
    rng = np.random.default_rng(0)
    n = rng.standard_normal(sr)
    d = int(0.006 * sr)
    st = np.stack([n, np.r_[np.zeros(d), n[:-d]]], axis=1)
    c = A.interchannel_corr(st, sr)
    assert abs(c["corr"]) < 0.05 and c["max"] > 0.99 and abs(c["lag_ms"] - 6.0) < 0.05
    assert abs(A.mono_loss_db(st) + 3.0) < 0.2
    assert A.mono_loss_db(np.stack([n, n], 1)) == 0.0 and A.mono_loss_db(np.stack([n, -n], 1)) < -100
    from scipy.signal import welch
    f, P = welch(st.mean(axis=1), sr, nperseg=8192)
    _, P0 = welch(n, sr, nperseg=8192)
    depth = lambda fn: 10 * np.log10(P[np.argmin(abs(f - fn))] / P0[np.argmin(abs(f - fn))])
    assert all(depth(fn) < -20 for fn in (83.33, 250.0, 416.7)) and abs(depth(166.7)) < 1.0


# ------------------------------------------------------------------ impulse responses
def test_edr_slope_and_late_onset():
    sr = 44100
    rng = np.random.default_rng(0)
    t = np.arange(int(1.5 * sr)) / sr
    ir = rng.standard_normal(t.size) * 10 ** (-3 * t)             # T60 = 1 s
    tE, fE, E = A.edr(ir, sr)
    band = 10 * np.log10(np.mean(10 ** (E[:, (fE > 500) & (fE < 2000)] / 10), axis=1))   # one bin scatters ~10 %
    m = (band > -40) & (band < -5)
    assert abs(-60 / np.polyfit(tE[m], band[m], 1)[0] - 1.0) < 0.05
    assert abs(np.mean(A.echo_density(rng.standard_normal(sr), sr)[1]) - 0.3173) < 0.005
    early = np.zeros(sr)
    early[[0, 300, 900, 1500, 2600]] = 1.0
    k0 = int(0.1 * sr)
    early[k0:] = rng.standard_normal(sr - k0) * 10 ** (-3 * t[:sr - k0])
    assert abs(A.late_onset(early, sr) - 0.1) < 0.01


# ------------------------------------------------------------------ bands, dissonance, roughness, annoyance
def test_critical_bandwidth_matches_zwicker_table():
    """F&Z Table 6.1 band widths: 100 Hz at 50 Hz, 160 at 1 kHz (920-1080), 700 at 4 kHz (3700-4400), 2500 at 10.5 kHz."""
    for f, bw in ((50, 100), (1000, 160), (4000, 700), (10500, 2500)):
        assert abs(A.critical_bandwidth(f) / bw - 1) < 0.03, f
    z = np.linspace(1, 23, 50)
    assert np.allclose(A.hz_to_bark(A.bark_to_hz(z)), z, atol=1e-6)


def test_dissonance_and_roughness_peak_where_their_sources_say():
    grid = np.linspace(0.01, 1.2, 500)
    peak = grid[np.argmax(1 - A._plomp_levelt(grid))]
    assert 0.2 < peak < 0.27                                      # Plomp & Levelt: most dissonant near 1/4 critical band
    assert A.dissonance([440, 880], [1, 1]) == 0.0 and A.dissonance([440, 466.16], [1, 1]) == pytest.approx(0.5)
    s = 0.24 / (0.0207 * 440 + 18.96)
    dd = np.linspace(1, 100, 991)
    r = [A.vassilakis_roughness([440, 440 + x], [1, 1]) for x in dd]
    assert abs(dd[int(np.argmax(r))] - math.log(5.75 / 3.5) / 2.25 / s) < 0.1
    assert A.vassilakis_roughness([440, 466], [1, 0.1]) < A.vassilakis_roughness([440, 466], [1, 1])


def test_annoyance_and_pleasantness_formulas():
    assert A.annoyance(10.0, 1.5) == 10.0                          # S <= 1.75 acum, no F or R: PA = N5
    ws, wfr = 0.25 * math.log10(20.0), 2.18 / 10 ** 0.4
    assert A.annoyance(10.0, 2.75, 1.0, 1.0) == pytest.approx(10 * (1 + math.hypot(ws, wfr)))
    assert A.pleasantness(0, 0, 1, 0) == pytest.approx(1.24 - math.exp(-2.43))
    assert A.pleasantness(1, 0, 1, 0) / A.pleasantness(0, 0, 1, 0) == pytest.approx(math.exp(-0.7))


# ------------------------------------------------------------------ listening tests
def test_sdt_reproduces_lindau_table_1():
    """Lindau & Weinzierl 2012 Table I: 2AFC rate Pc and the sensitivity d' it implies."""
    from statistics import NormalDist
    N = NormalDist()
    for pc, d in ((0.5, 0.0), (0.55, 0.1777), (0.60, 0.3583), (0.65, 0.5449), (0.70, 0.7416), (0.75, 0.9539)):
        r = C.sdt(N.cdf(d / 2), N.cdf(-d / 2))                    # fair observer with that d'
        assert abs(r["d"] - d) < 1e-9 and abs(r["pc"] - pc) < 5e-4, (pc, r)
    r = C.sdt(0.8, 0.2)
    assert r["beta"] == pytest.approx(1.0) and r["lambda"] == pytest.approx(r["d"] / 2)   # fair observer


def test_mushra_post_screening_follows_bs1534():
    def page(ref, anc, a=50):
        return {"reference": ref, "anchor70": anc, "A": a}
    items = [f"i{k}" for k in range(10)]
    s = {"good": {i: page(100, 60) for i in items},
         "ref_deaf": {i: page(80 if k < 2 else 100, 60) for k, i in enumerate(items)},     # 20 % > 15 %
         "one_slip": {i: page(80 if k < 1 else 100, 60) for k, i in enumerate(items)},     # 10 %: kept
         "anchor_fan": {i: page(100, 95 if k < 2 else 60) for k, i in enumerate(items)}}   # 20 % anchors > 90
    r = C.mushra_screen(s)
    assert sorted(r["excluded"]) == ["anchor_fan", "ref_deaf"]
    s["x1"] = {i: page(100, 95 if i in ("i0", "i1") else 60) for i in items}
    s["x2"] = {i: page(100, 95 if i in ("i0", "i1") else 60) for i in items}
    r = C.mushra_screen(s)                                        # 3 of 6 > 25 % on i0, i1: those items do not count
    assert "anchor_fan" not in r["excluded"] and "ref_deaf" in r["excluded"]
    assert C._quartiles([1, 2, 3, 4, 5, 6, 7]) == (2.5, 4.0, 5.5)
    s["good"]["i3"]["A"] = 100                                    # far out of the others' 50
    assert ("good", "i3", "A", 100) in C.mushra_screen(s)["outliers"]


def test_align_stimuli_levels_and_limit(tmp_path):
    from genny.core import read_wav, write_wav
    import pyloudnorm
    sr = 48000
    t = np.arange(sr) / sr
    paths = {f"s{k}": str(write_wav(tmp_path / f"in{k}.wav", (0.05 + 0.2 * k) * np.sin(2 * np.pi * 440 * t), sr))
             for k in range(3)}
    out = C.align_stimuli(paths, tmp_path / "out", lufs=-23.0)
    for p in out.values():
        y, r = read_wav(p)
        assert abs(pyloudnorm.Meter(r).integrated_loudness(y) + 23.0) < 0.2       # BS.1534 §8: equal loudness
    with pytest.warns(UserWarning):
        C.align_stimuli({f"s{k}": paths["s0"] for k in range(10)}, tmp_path / "many")


# ------------------------------------------------------------------ render against recording
def test_fad_on_gaussians():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((20000, 4))
    assert abs(C.fad(a, a + np.array([1.0, 0, 0, 2.0])) - 5.0) < 1e-6                # |mu|^2, same covariance
    assert abs(C.fad(a, 2.0 * rng.standard_normal((20000, 4))) - 4.0) < 0.2           # tr(I + 4I - 2 * 2I) = d


def test_compare_render_to_a_recording():
    """The first render-against-recording check: a sung /a/ (Openverse, CC0, recordings/voice_sung) against genny's
    voice at the recording's own f0, a saw, white noise and a flute. Measured: FAD on log-mel frames picks the voice
    (1037 against 1154-1830); the Welch PSD RMSE does not (voice 20.3 dB, flute 13.5): genny's sung /a/ spectrum is
    off this speaker's by 20 dB RMS, reported to the Voix theme."""
    if not VOWEL.exists():
        pytest.skip("recordings/ not present")
    from genny.core import read_wav
    x, sr = read_wav(VOWEL)
    x = x if x.ndim == 1 else x.mean(axis=1)
    f0 = A.periodicity(x, sr)["f0"]
    assert abs(f0 - 199.3) < 2.0
    dur = x.size / sr
    got = {}
    for name, layer in (("voice", {"type": "synth", "inst": "voice", "notes": f"{f0:.2f}hz", "dur": dur, "params": {"vowel": "ah"}}),
                        ("saw", {"type": "synth", "inst": "saw", "notes": f"{f0:.2f}hz", "dur": dur}),
                        ("noise", {"type": "sfx", "kind": "noise", "params": {"dur": dur}}),
                        ("flute", {"type": "synth", "inst": "flute", "notes": f"{f0:.2f}hz", "dur": dur})):
        y, _ = _render(layer, sr)
        got[name] = C.compare(y, x, sr)
    assert min(got, key=lambda k: got[k]["fad"]) == "voice"
    assert 17.0 < got["voice"]["psd_rmse"] < 23.5 and got["flute"]["psd_rmse"] < got["voice"]["psd_rmse"]
    assert C.compare(x, x, sr)["psd_rmse"] < 1e-6 and C.compare(x, x, sr)["loudness_l1"] < 1e-6


def test_pcen_steady_state_and_onset():
    """librosa.pcen defaults: a constant input settles at (2 + S / (1e-6 + S)^0.98)^0.5 - 2^0.5; an onset stands out."""
    sr, hop = 16000, 256
    S = np.full((4, 400), 0.3)
    out = I.pcen(S, sr, hop)
    assert abs(out[0, -1] - ((2 + 0.3 / (1e-6 + 0.3) ** 0.98) ** 0.5 - 2 ** 0.5)) < 1e-4
    S[:, 300:] = 3.0
    out = I.pcen(S, sr, hop)
    assert out[0, 300] > 3 * out[0, -1]


def test_voice_breaks_count_gaps_over_1_25_of_the_floor_period():
    from genny.physical.analysis import voice_breaks
    p = np.r_[np.arange(0, 0.5, 0.01), np.arange(0.52, 1.0, 0.01)]     # 20 ms gap > 16.7 ms at 75 Hz, 10 ms intervals are not
    n, deg = voice_breaks(p, 1.0, 75.0)
    assert n == 1 and deg == pytest.approx(0.03, abs=1e-6)
    assert voice_breaks(p, 1.0, 40.0)[0] == 0                          # 31 ms floor at 40 Hz
