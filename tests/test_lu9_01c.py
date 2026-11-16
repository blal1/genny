import numpy as np

from genny import compose, nature, speech


def test_prevoice_lengthens_spanish_initial_voiced_stop_only():
    segs = speech._build_segments(speech.text_to_phrases("ba", "es"), 1.0, 0.0, "es", True)
    assert abs(segs[0]["lead"] - 0.138) < 1e-9 and segs[0]["dur"] >= 0.138
    off = speech._build_segments(speech.text_to_phrases("ba", "es"), 1.0, 0.0, "es")
    assert "lead" not in off[0] and off[0]["dur"] < 0.1
    en = speech._build_segments(speech.text_to_phrases("ba", "en"), 1.0, 0.0, "en", True)
    assert "lead" not in en[0]
    p = speech._build_segments(speech.text_to_phrases("pa", "es"), 1.0, 0.0, "es", True)
    assert "lead" not in p[0]


def test_prevoice_default_unchanged_and_audible():
    a = speech.render_speech("da", lang="es")
    assert np.array_equal(a, speech.render_speech("da", lang="es", prevoice=False))
    b = speech.render_speech("da", lang="es", prevoice=True)
    assert len(b) > len(a)
    head = int(0.08 * 44100)
    assert np.abs(b[:head]).max() > np.abs(a[:head]).max()


def test_sentence_vot_tables():
    assert speech.VOT_SENTENCES["spanish_pr"]["voiced"] == (-110, -109, -92)
    assert speech.VOT_SENTENCES["korean"]["aspirated"][2] == 125


def test_interval_affect_table():
    assert compose.interval_affect(-300)["sadness"] == 4.47
    assert compose.interval_affect(100)["anger"] == 4.72
    assert compose.interval_affect(1210)["happiness"] == 4.96
    assert compose.interval_affect(1900)["pleasantness"] == compose.interval_affect(700)["pleasantness"]


def test_double_pulse_measured_cadence():
    n = int(600 * nature.P)
    y = nature.double_pulse_tremor(n, np.random.default_rng(3), **nature.DOUBLE_PULSE_MEASURED)
    assert y.shape == (n,) and np.abs(y).max() > 0
    d = nature.volcano(dur=8.0, stage="tremor", seed=1, tremor="double_pulse_measured")
    assert np.isfinite(d).all()


def test_miller_pursey_directivity_integrals():
    from genny import contact
    t = np.linspace(0, np.pi / 2, 2001)
    p, s = contact.surface_source_directivity(t, 0.25)
    assert abs(np.trapezoid(p ** 2 * np.sin(t), t) - 0.333) < 0.002
    assert abs(np.trapezoid(s ** 2 * np.sin(t), t) - 1.246) < 0.005
    assert abs(contact.surface_source_directivity(np.radians(40.0), 0.25)[1] ** 2 - 3.122) < 0.01
    assert abs(sum(contact.GROUND_SHARE_POISSON[0.25]) - 1.0) < 0.002


def test_tornado_psd_peak_and_tail():
    f = np.array([1.0, 14.8, 100.0, 400.0])
    s = nature.tornado_psd(f)
    assert s[1] > s[2] > s[3] > 0
