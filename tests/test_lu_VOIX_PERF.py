import numpy as np

from genny import speech as SP


def _tracks(phonemes, **params):
    p, vs = SP._resolve(params)
    segs = SP._build_segments([{"words": SP._parse_raw(phonemes), "punct": ""}], 1.0, 0.0, "en")
    return segs, SP._build_tracks(segs, vs, "en")


def test_klatt1980_diphthongs_table_ii():
    assert SP.PH["AY"]["f"] == (660, 1200, 2550) and SP.PH["AY"]["f2"] == (400, 1880, 2500)
    assert SP.PH["OY"]["f2"] == (360, 1820, 2450)
    assert SP.PH["AW"]["f"] == (640, 1230, 2550) and SP.PH["AW"]["f2"] == (420, 940, 2350)


def test_klatt1980_sonorants_table_iii():
    assert SP.PH["L"]["f"] == (310, 1050, 2880) and SP.PH["L"]["b"] == (50, 100, 280)
    assert SP.PH["R"]["f"] == (310, 1060, 1380) and SP.PH["W"]["f"] == (290, 610, 2150)
    tgt = SP._targets([{"ph": "Y"}])
    assert tgt[0][0][5:8] == [40, 250, 500]


def test_klatt1980_h_first_bandwidth_300():
    segs, T = _tracks("HH AA1")
    s, e = segs[0]["s"], segs[0]["e"]
    t = np.array([s + min(0.012, (e - s) / 3)])
    b1 = T.curve("FM5", t, 0.0) + T.curve("B1ADD", t, 0.0)
    assert abs(b1[0] - 300.0) < 1.0


def test_klatt1980_prevocalic_sonorant_10db_down():
    segs, T = _tracks("L AA1")
    t = np.array([0.5 * (segs[0]["s"] + segs[0]["e"])])
    assert abs(20 * np.log10(T.curve("AV", t, 0.0)[0]) + 10.0) < 0.2


def test_klatt1980_nasalized_vowel_zero():
    segs, T = _tracks("AA1 M", nasal_vowels=True)
    e = np.array([segs[0]["e"]])
    f1 = SP.PH["AA"]["f2"][0]
    assert abs(T.curve("FNZ", e, 0.0)[0] - 0.5 * (f1 + 100 + 270)) < 1.0
    assert abs(T.curve("F1N", e, 0.0)[0] - 100.0) < 1e-6
    _, T0 = _tracks("AA1 M")
    assert not T0.p.get("F1N")


def test_klatt1990_open_quotient_defaults():
    assert SP.VOICES["male"]["oq"] == 0.50 and SP.VOICES["female"]["oq"] == 0.60


def test_klatt1990_flutter_eq1():
    assert SP.FLUTTER == 25.0 and SP.FLUTTER_HZ == (12.7, 7.1, 4.7)
    a = SP.FLUTTER / 50 * 100.0 / 100
    assert abs(a - 0.5) < 1e-12


def test_langston_riffology_table():
    from genny import compose as C
    assert len(C.RIFFS) == 40 and all(len(r) == 8 for r in C.RIFFS)
    assert C.RIFFS[0][:3] == [63, 62, 69] and C.RIFFS[2][2] is None
    assert C.RIFF_BIV == (28, 0, 7, 0, 14, 0, 7, 4) and C.RIFF_TRIES == 3
    assert C.riff_energy(0, 12) == 100 and C.riff_energy(6, 12) == 70.0 and C.riff_energy(11, 12) > 100


def test_langston_riffology_events():
    from genny import compose as C
    ev = C.riffology(8, np.random.default_rng(1))
    assert ev and all(0 <= e["t"] < 32 and e["dur"] > 0 for e in ev)
    assert all(abs(e["t"] * 8 - round(e["t"] * 8)) < 1e-9 for e in ev)


def test_huron_parallel_perfects():
    from genny import choir as CH
    assert CH.parallel_perfects((72, 67, 64, 60), (74, 69, 65, 62)) >= 1
    assert CH.parallel_perfects((72, 67, 64, 48), (71, 67, 62, 55)) == 0
    assert CH.PARALLEL_COST > 0


def test_hennig_gamma_noise_beta():
    from genny import compose as C
    assert C.HENNIG_BETA == 1.0
    g = C.gamma_noise(4096, 1.0, np.random.default_rng(0))
    P = np.abs(np.fft.rfft(g - g.mean()))[1:] ** 2
    f = np.arange(1, len(P) + 1)
    k = f < len(P) // 4
    assert -1.5 < np.polyfit(np.log(f[k]), np.log(P[k]), 1)[0] < -0.5


def test_smf_writer_header_and_vlq():
    from genny import compose as C
    assert C._vlq(0) == b"\x00" and C._vlq(0x40) == b"\x40" and C._vlq(0x80) == b"\x81\x00"
    assert C._vlq(0x0FFFFFFF) == b"\xff\xff\xff\x7f"
    d = C.write_midi([{"t": 0.0, "dur": 1.0, "pitch": 60, "vel": 1.0}], None, bpm=120, ppq=96)
    assert d[:14] == b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00\x60"
    assert b"\xff\x51\x03\x07\xa1\x20" in d and d.endswith(b"\x00\xff\x2f\x00")
    assert b"\x00\x90\x3c\x7f" in d and b"\x60\x80\x3c\x00" in d


def test_ddsp_mss_log_term_on():
    from genny import calibrate as CAL
    x = np.random.default_rng(0).standard_normal(4096)
    assert CAL.multiscale_spectral_distance(x, 0.5 * x) > CAL.multiscale_spectral_distance(x, 0.5 * x, w_log=0.0)


def test_klatt1990_diplophonia_halves_f0():
    sr = 16000
    x = SP.render_speech(phonemes="AA1", sr=sr, voice="male", pitch=120.0, intonation="flat", emphasis=0.0, diplophonia=1.0)
    x = np.asarray(x, float).reshape(len(x), -1)[:, 0]
    seg = x[len(x) // 3: 2 * len(x) // 3]
    X = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    f = np.fft.rfftfreq(len(seg), 1 / sr)
    band = lambda c: X[(f > c - 8) & (f < c + 8)].max()
    assert band(180.0) > 0.3 * band(240.0)
    y = SP.render_speech(phonemes="AA1", sr=sr, voice="male", pitch=120.0, intonation="flat", emphasis=0.0)
    y = np.asarray(y, float).reshape(len(y), -1)[:, 0][len(y) // 3: 2 * len(y) // 3]
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    assert Y[(f > 172) & (f < 188)].max() < 0.1 * Y[(f > 232) & (f < 248)].max()


def test_hennig_default_sigma_15ms():
    from genny import compose as C
    assert C.HENNIG_MS == 15.0


def test_huron_hidden_octave_needs_two_leaps():
    from genny import choir as CH
    assert CH.parallel_perfects((64, 60, 55, 48), (72, 67, 64, 60)) >= 1
    assert CH.parallel_perfects((71, 67, 62, 43), (72, 67, 64, 48)) == 0
