"""Theme Voix (reports/lecture_profonde/suivi/T_Voix.md): every check is a sourced or measured number.

Run:  uv run --with pytest pytest tests/test_voix.py -q
"""
import math

import numpy as np
import pytest
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.special import gammainc

from genny import choir as C, speech as S
from genny.analysis import detect_clicks
from genny.creatures import crowd_murmur
from genny.physical import nonverbal as NV, story as ST, voice as V
from genny.physical.core import SR

SP = 44100
# Story 2008, Table II: calculated formants of his 11 vowels, in the table's order
ORDER = ["i", "ɪ", "e", "ɛ", "æ", "ʌ", "ɑ", "ɔ", "o", "ʊ", "u"]
F1C = (325, 413, 484, 565, 810, 593, 694, 557, 499, 477, 314)
F2C = (2139, 2103, 1887, 1595, 1655, 836, 942, 709, 768, 789, 702)
F3C = (2976, 2626, 2404, 2200, 2310, 3099, 2948, 3176, 2421, 2499, 2298)


def _band_db(y, f, sr=SP):
    """Level (dB) of the strongest component inside a frequency band."""
    S_ = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    k = np.fft.rfftfreq(y.size, 1 / sr)
    return 20 * np.log10(S_[(k >= f[0]) & (k <= f[1])].max() + 1e-12)


def _vowel(y, sr=SP):
    a = int(0.35 * y.size)
    return y[a:a + int(0.3 * y.size)]


def _cep_f0(y, sr, lo, hi):
    """f0 of a held vowel by autocorrelation."""
    r = np.correlate(y, y, "full")[y.size - 1:]
    a, b = int(sr / hi), int(sr / lo)
    return sr / (a + int(np.argmax(r[a:b])))


def _inst_f0(y, sr, lo, hi):
    """Instantaneous frequency of the fundamental."""
    z = hilbert(sosfiltfilt(butter(3, (lo, hi), "bandpass", fs=sr, output="sos"), y))
    return np.diff(np.unwrap(np.angle(z))) * sr / (2 * np.pi), np.abs(z)[1:]


def test_almost_speech_voices_no_longer_share_their_syllable_phase():
    # 4 Hz line of the envelope over its neighbours, seeds 0, 1: 15.0 and 19.1 dB with every phasor at phase 0,
    # 6.1 and 13.0 dB measured with random phases (about 9 and 6 dB less; a line remains, each voice still has it)
    for seed, top in ((0, 9.0), (1, 14.0)):
        y = crowd_murmur(people=20, dur=4.0, seed=seed)
        e = np.abs(y) - np.abs(y).mean()
        Y = np.abs(np.fft.rfft(e * np.hanning(e.size)))
        f = np.fft.rfftfreq(e.size, 1 / SP)
        band = (f > 3.5) & (f < 4.5)
        rest = (f > 1) & (f < 12) & ~band
        assert 20 * np.log10(Y[band].max() / np.median(Y[rest])) < top
    a, b = (V.almost_speech(0.5, np.random.default_rng(s)) for s in (1, 2))
    assert not np.allclose(a, b)


def test_helium_moves_the_formants_not_the_pitch():
    a = S.render_speech(None, SP, phonemes="AA1", voice="male", intonation="flat")
    b = S.render_speech(None, SP, phonemes="AA1", voice="helium", intonation="flat")
    assert abs(_cep_f0(_vowel(b), SP, 70, 300) / _cep_f0(_vowel(a), SP, 70, 300) - 1) < 0.02
    # Brittan-Powell 1997: 485 / 345.5 = 1.40, carried on the F1 of the same phoneme
    assert S.VOICES["helium"]["formant_shift"] == pytest.approx(485 / 345.5, abs=0.01)
    k = np.fft.rfftfreq(_vowel(a).size, 1 / SP)
    centre = lambda y: k[(k < 2200)] @ np.abs(np.fft.rfft(_vowel(y) * np.hanning(_vowel(y).size)))[k < 2200] ** 2 \
        / np.sum(np.abs(np.fft.rfft(_vowel(y) * np.hanning(_vowel(y).size)))[k < 2200] ** 2)
    assert 1.2 < centre(b) / centre(a) < 1.6


def test_effort_defaults_leave_the_voice_alone_and_soft_loud_follow_brockmann():
    _, vs = S._resolve({"voice": "male"})
    assert vs["pitch"] == 115.0 and vs["jitter"] == 0.004 and "effort_db" not in vs
    _, f = S._resolve({"voice": "female", "effort": 0.9})
    _, m = S._resolve({"voice": "male", "effort": 0.9})
    assert f["pitch"] / 205.0 == pytest.approx(301 / 211, rel=1e-6)          # +43 %
    assert m["pitch"] / 115.0 == pytest.approx(178 / 127, rel=1e-6)          # +40 %
    _, s = S._resolve({"voice": "male", "effort": 0.3})
    assert s["jitter"] / 0.004 == pytest.approx(0.94 / 0.31, rel=1e-6) and s["shimmer"] / 0.035 == pytest.approx(1.00 / 0.28, rel=1e-6)
    assert m["effort_db"] == pytest.approx(98.8 - 79.2)


def test_effort_flattens_a3_re_a1_by_the_lienard_slope():
    def a31(**kw):
        y = _vowel(S.render_speech(None, SP, phonemes="AA1", voice="male", intonation="flat", **kw))
        return _band_db(y, (2200, 2900)) - _band_db(y, (500, 1000))
    d = a31(effort=0.9, pitch=115.0 * 127 / 178) - a31()                     # same F0: harmonics land alike
    assert 0.20 * 19.6 - 0.3 < d < 0.31 * 19.6, d                            # Lienard: 0.20 (regression) to 0.31 dB/dB; measured 4.15 dB


def test_rendered_f0_follows_effort():
    y = _vowel(S.render_speech(None, SP, phonemes="AA1", voice="female", intonation="flat", effort=0.9))
    assert _cep_f0(y, SP, 150, 500) == pytest.approx(205 * 301 / 211, rel=0.03)


def test_klglott88_constants_of_doval_annex():
    x = np.linspace(0, 1, 200001)[:-1]
    d, f, J = S.glottal_pulse(x, 1.0, source="klglott88")
    assert x[np.argmax(f)] == pytest.approx(2 / 3, abs=1e-3)                  # alpha_m
    assert J == pytest.approx(27 / 4) and f.max() == pytest.approx(1.0, abs=1e-6)
    # normalised to A = 4/27: in = 1/12, E = 1, fg = sqrt(3)/pi
    in_ = np.mean(f) * 4 / 27
    assert in_ == pytest.approx(1 / 12, rel=1e-3)
    assert math.sqrt(1 / in_) / (2 * math.pi) == pytest.approx(math.sqrt(3) / math.pi, rel=1e-3)
    # H1-H2 at Oq: two Oq for one value between 7.94 and 9.82 dB, single-valued up to Oq 0.74
    oq = np.arange(0.3, 1.0, 0.005)
    h = []
    x = np.arange(4096) / 4096
    for q in oq:
        U = np.abs(np.fft.rfft(S.glottal_pulse(x, float(q), source="klglott88")[0]))     # H1-H2 of the radiated flow derivative
        h.append(20 * np.log10(U[1] / U[2]))
    h = np.array(h)
    assert h.max() == pytest.approx(9.82, abs=0.05)
    assert h[np.argmin(np.abs(oq - 0.74))] == pytest.approx(7.94, abs=0.1) and h[-1] == pytest.approx(7.94, abs=0.1)
    y = S.render_speech(None, SP, phonemes="AA1", source="klglott88")
    assert np.isfinite(y).all() and detect_clicks(y).size == 0


def test_skew_places_the_flow_peak():
    x = np.linspace(0, 1, 100001)[:-1]
    for skew in (0.6, 0.8):
        assert x[np.argmax(S.glottal_pulse(x, 0.6, skew)[1])] == pytest.approx(0.6 * skew, abs=1e-3)


def test_h1h2_inverts_fant():
    assert S.oq_for_h1h2(-6 + 0.27 * math.exp(5.5 * 0.6)) == pytest.approx(0.6, abs=1e-9)
    assert S.oq_for_h1h2(30.0) == 0.74                                       # the bound of Doval p. 1041
    _, vs = S._resolve({"h1h2": 3.4})
    assert vs["oq"] == pytest.approx(0.6455, abs=1e-3)                       # Hanson's mean H1-H2 of 3.4 dB


def test_tilt_db_drops_three_khz():
    def a3(t):
        y = _vowel(S.render_speech(None, SP, phonemes="AA1", intonation="flat", tilt_db=t))
        return _band_db(y, (2200, 3400)) - _band_db(y, (100, 300))
    assert 8.0 < a3(1.0) - a3(13.0) < 14.0


def test_order_6_step_response():
    fs = 20000.0
    tau = 0.015
    x = np.zeros(8000)
    x[3000:] = 1.0
    y = S.target_approach(x, tau, 6, fs)
    t0 = 3000 / fs - 6 * tau                                                 # the step read 6 tau ahead
    t = np.arange(x.size) / fs - t0
    for level, k in ((0.1, 3.15), (0.5, 5.67), (0.9, 9.27)):                 # Birkholz 2011, as t10 t50 t90 in tau
        assert t[np.argmax(y >= level)] / tau == pytest.approx(k, abs=0.15)
    assert t[np.argmax(y >= gammainc(6, 7.0))] / tau == pytest.approx(7.0, abs=0.15)


def test_gesture_keeps_the_vowel_targets():
    a = S.render_speech(None, SP, phonemes="M AA1", intonation="flat", seed=1)
    b = S.render_speech(None, SP, phonemes="M AA1", intonation="flat", seed=1, trajectory="gesture")
    assert not np.allclose(a[:min(a.size, b.size)], b[:min(a.size, b.size)])
    assert _cep_f0(_vowel(b), SP, 70, 300) == pytest.approx(_cep_f0(_vowel(a), SP, 70, 300), rel=0.02)
    assert np.isfinite(b).all()


def test_rd_up_to_six_huber_roebel():
    for rd in (2.0, 2.7, 3.0, 4.0, 5.0, 6.0):
        y = V.lf_waveform(rd, 16384)
        u = np.cumsum(y) / 16384
        assert u.max() / 0.11 == pytest.approx(rd, rel=2e-3), rd              # Up / (Ee T0) = 0.11 Rd
        assert abs(u[-1]) < 1e-3
    te = lambda r: V._lf_coefs(r)[3]
    assert abs(te(2.7001) / te(2.6999) - 1) < 0.015                            # "saut de Te de 1 %", lot_B51
    for rd in (3.0, 6.0):
        assert te(rd) < 1 - 1 / (2.17 * rd)                                    # Te <= OQupp
    assert te(6.5) == te(6.0)


def test_pink_trombone_speaks_an_external_excitation():
    rng = np.random.default_rng(3)
    y = V.pink_trombone(120.0, 0.6, dur=0.5, excitation=rng.normal(0, 0.1, int(0.5 * SR)), voiced_env=1.0)
    g = V.pink_trombone(120.0, 0.6, dur=0.5)
    assert np.isfinite(y).all() and y.size == g.size
    # noise excitation: no 120 Hz comb left, and the peaks are those of the tract's own impulse response (5 %)
    r = np.correlate(g, g, "full")[g.size - 1:]
    q = np.correlate(y, y, "full")[y.size - 1:]
    lag = int(SR / 120)
    assert q[lag - 8:lag + 8].max() / q[0] < 0.5 * r[lag - 8:lag + 8].max() / r[0]
    ir = np.array(V._resp_peaks(V._pt_ir(12.9, 2.43)[0], fs=96000.0)[:3])
    assert np.array(V._smoothed_peaks(y)[:3]) == pytest.approx(ir, rel=0.05)


def test_f1_margin_is_the_four_semitones_of_sundberg():
    F = np.tile(C.voice_formants("soprano", "a")["freqs"], (1, 1))
    f0 = np.array([900.0])
    assert C.tuned_formants(F, f0, "soprano")[0, 0] == pytest.approx(900.0)
    assert C.tuned_formants(F, f0, "soprano", 4.0)[0, 0] == pytest.approx(900.0 * 2 ** (4 / 12))
    assert 2 ** (4 / 12) == pytest.approx(1.26, abs=0.001)


def test_coloratura_rule_targets():
    tb = np.arange(0.0, 0.6, 0.001)
    c = C.coloratura_cents(tb, [0.0, 0.2, 0.4], [0.2, 0.2, 0.2], [60.0, 62.0, 60.0])
    at = lambda t: c[int(round(t * 1000))]
    assert at(0.0) == pytest.approx(6100.0)                                     # +1 st at the onset
    assert at(0.1) == pytest.approx(5900.0)                                     # -1 st in the middle
    assert at(0.2) == pytest.approx(6400.0)                                     # peak note: +2
    assert at(0.3) == pytest.approx(6100.0)
    assert at(0.05) == pytest.approx(6000.0, abs=1.0)                           # cosine, halfway


def test_coloratura_widens_a_rendered_run():
    notes = [(i * 0.12, 57.0 + (i % 2) * 2, 0.12, 0.8) for i in range(10)]
    span = []
    for col in (False, True):
        y = C.sing_phrase(notes, SP, voice="tenor", stereo=False, coloratura=col, vibrato=0.0, seed=2)
        f, a = _inst_f0(y, SP, 150, 400)
        m = (np.arange(f.size) > 0.25 * SP) & (np.arange(f.size) < 1.0 * SP)
        st = 12 * np.log2(np.maximum(f[m], 1.0))
        span.append(np.percentile(st, 97) - np.percentile(st, 3))
    assert span[1] > span[0] + 1.0, span


def test_diphthong_holds_its_first_vowel_35_percent(monkeypatch):
    seen = []
    add = C._Tracks.add

    def spy(self, name, t, v):
        if name == "FB":
            seen.append((t, np.array(v, float).copy()))
        add(self, name, t, v)
    monkeypatch.setattr(C._Tracks, "add", spy)
    C.sing_phrase([(0.1, 60.0, 1.0, 0.8)], 22050, voice="tenor", lyrics="ay", lang="en", stereo=False, seed=1)
    ts = [t for t, _ in seen]
    held = [t for t, v in seen if np.allclose(v, seen[0][1]) and t > ts[0]]
    assert held and held[-1] - 0.1 == pytest.approx(0.35, abs=0.03)


def test_end_boost_speeds_the_last_half_second_of_vibrato(monkeypatch):
    f0s = []
    kernel = C._kernel

    def spy(f0, *a):
        f0s.append(np.array(f0))
        return kernel(f0, *a)
    monkeypatch.setattr(C, "_kernel", spy)
    n = [(0.0, 57.0, 3.0, 0.9)]
    C.sing_phrase(n, SP, voice="tenor", stereo=False, seed=1, end_boost=0.0)
    C.sing_phrase(n, SP, voice="tenor", stereo=False, seed=1, end_boost=2.0)
    d = 1200 * np.log2(f0s[1] / f0s[0])
    win = lambda a, b: np.abs(d[int(a * SP):int(b * SP)]).max()
    assert win(0.0, 2.45) < 1e-6
    # extra phase 2 pi x 2 Hz x 0.5 s / 2 = pi by the end: sin(p + pi) - sin(p) peaks at twice the 60 cents extent
    assert win(2.5, 2.6) < 30.0
    assert 90.0 < win(2.9, 3.0) < 150.0


def test_default_end_boost_is_prame(monkeypatch):
    # Prame 1994: rate +15 % over the last ~5 cycles; extra phase 2 pi 0.15 r (5 / r) / 2 = 0.75 pi whatever the rate
    f0s = []
    kernel = C._kernel

    def spy(f0, *a):
        f0s.append(np.array(f0))
        return kernel(f0, *a)
    monkeypatch.setattr(C, "_kernel", spy)
    n = [(0.0, 57.0, 3.0, 0.9)]
    C.sing_phrase(n, SP, voice="tenor", stereo=False, seed=1, end_boost=0.0)
    C.sing_phrase(n, SP, voice="tenor", stereo=False, seed=1)
    d = 1200 * np.log2(f0s[1] / f0s[0])
    win = lambda a, b: np.abs(d[int(a * SP):int(b * SP)]).max()
    assert win(0.0, 1.85) < 1e-6
    assert 85.0 < win(2.9, 3.0) < 125.0


def test_closed_mouth_nasalises_f1(monkeypatch):
    f = np.arange(100.0, 1500.0, 2.0)
    base = {**NV.DEFAULTS, "formants": ((600.0,), (1500.0,)), "vocalTract": 17.5}
    op = NV._envelope(f, 1, {**base, "mouth": (0.5,)})[:, 0]
    cl = NV._envelope(f, 1, {**base, "mouth": (0.0,)})[:, 0]
    def w(H):
        low = H[f < 1000]                                                       # the F1 peak, not F2
        return np.count_nonzero(low >= low.max() / math.sqrt(2)) * 2.0
    assert w(op) < 100                                                          # 50-ish Hz: the given bandwidth
    assert w(cl) > 2 * w(op)
    monkeypatch.setattr(NV, "NASAL_PAIR", (1.0, 1.0, 175.0))                    # pole on zero: the pair cancels
    monkeypatch.setattr(NV, "LIP_RAD", 0.0)                                     # the +6 dB/oct skews the peak
    assert w(NV._envelope(f, 1, {**base, "mouth": (0.0,)})[:, 0]) == pytest.approx(175.0, rel=0.2)   # B1 alone
    ch = NV.vocalize("cry", "female", np.random.default_rng(2))
    assert np.isfinite(ch).all()


def test_subharmonic_walk_puts_energy_at_half_f0():
    p = {**NV.DEFAULTS, "rolloff": -12.0, "subDep": 0.8}
    n = int(0.8 * SR)
    f0 = np.full(n, 300.0)
    with_sub = None
    for seed in range(8):
        reg = NV.regimes(n, np.random.default_rng(seed))
        assert set(np.unique(reg)) <= {0, 1, 2}
        y = NV._syllable(p, n, f0, np.random.default_rng(seed))
        d = _band_db(y, (140, 160), SR) - _band_db(y, (290, 310), SR)
        with_sub = d if with_sub is None else max(with_sub, d)
    y0 = NV._syllable({**p, "subDep": 0.0}, n, f0, np.random.default_rng(0))
    assert with_sub > _band_db(y0, (140, 160), SR) - _band_db(y0, (290, 310), SR) + 20


def test_lossy_line_gives_story_table_ii():
    for k, a, b, c in zip(ORDER, F1C, F2C, F3C):
        Fs = ST.formants(ST.vowel_area(k), ST.MEASURED[k][0])
        err = ST.formant_error(Fs, (a, b, c))
        assert np.abs(err).max() < 2.0, (k, Fs[:3], err)                        # his F1C, F2C, F3C


def test_neutral_tract_and_liquids():
    assert np.abs(ST.formant_error(ST.formants(ST.substrate(0, 0)), (600, 1504, 2492))).max() < 2.5   # Story & Bunton 2017
    assert ST.vowel_area("neutral").shape == (44,)
    assert ST.vowel_area("neutral")[0] == pytest.approx(np.mean([a[0] for _, a in ST.MEASURED.values()]))
    # Table II: F3 falls to 1.75 kHz for /ɹ/b, rises to 3 kHz for /l/b
    assert ST.formants(ST.liquid_area("ɹb"))[2] == pytest.approx(1750, rel=0.07)
    assert ST.formants(ST.liquid_area("lb"))[2] == pytest.approx(3000, rel=0.07)


def test_min_jerk_and_corner_vowels():
    t = np.array([0.0, 0.5, 1.0])
    assert ST.min_jerk(t, 1.0, 2.0, 4.0).tolist() == [2.0, 3.0, 4.0]
    assert ST.min_jerk(0.25, 1.0) == pytest.approx(10 / 64 - 15 / 256 + 6 / 1024)
    # Birkholz /y/: x_y = 0.25 x_a + 0.69 x_i + 0.06 x_u comes back as a1 = 0.69, a2 = 0.06
    xa, xi, xu = (ST.vowel_area(k) for k in ("ɑ", "i", "u"))
    a1, a2 = ST._corner_weights(0.25 * xa + 0.69 * xi + 0.06 * xu)
    assert (a1, a2) == pytest.approx((0.69, 0.06), abs=1e-6)
    assert ST.coarticulated_area("t", "ɑ") == pytest.approx(ST.consonant_area("t", xa))
    assert ST.corner_weights("i") == pytest.approx((1.0, 0.0), abs=1e-6)


_ST98 = {"i": (386, 2615, 36), "ɪ": (721, 2197, 34), "ɛ": (863, 2219, 32), "æ": (933, 2338, 30), "ʌ": (884, 1528, 34),
         "ɑ": (792, 1585, 34), "ɔ": (727, 1327, 36), "o": (730, 1546, 36), "ʊ": (883, 1548, 36), "u": (463, 1152, 38)}


def test_female_area_functions_story_1998():
    for k, (f1, f2, n) in _ST98.items():
        A = ST.vowel_area(k, "female")
        assert A.size == n and ST.vowel_length(k, "female") == pytest.approx(n * 0.396825e-2)
        err = ST.formant_error(ST.formants(A), (f1, f2))
        assert -8.0 < err[0] < 0.0 and abs(err[1]) < 2.0, (k, err)
    with pytest.raises(ValueError):
        ST.vowel_area("e", "female")
