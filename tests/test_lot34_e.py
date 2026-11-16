import numpy as np

from genny import foley, klang, spec
from genny.core import DEFAULT_SR


def _spectrum(y, sr):
    s = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return np.fft.rfftfreq(len(y), 1 / sr), s


def test_red_alert_patch_renders_with_formants():
    sr = DEFAULT_SR
    y = foley.red_alert(sr, whoops=2, model="patch")
    assert np.all(np.isfinite(y)) and np.max(np.abs(y)) > 0.1
    f, s = _spectrum(y, sr)
    band = lambda lo, hi: s[(f > lo) & (f < hi)].max()
    assert band(1350, 1650) > 2.0 * band(2600, 3200)
    assert band(700, 780) > band(250, 330)


def test_red_alert_whoop_unchanged_signature():
    y = foley.red_alert(DEFAULT_SR, whoops=1)
    assert y.shape[0] > 0 and np.all(np.isfinite(y))


def test_red_alert_rejects_unknown_model():
    try:
        foley.red_alert(DEFAULT_SR, model="x")
    except ValueError:
        return
    raise AssertionError


def test_klang_wind_formant_tables():
    from genny import instruments as I
    assert [p[0] for p in klang.HORN_BODY] == [340, 750, 1225, 2000, 3500]
    for name in ("bassoon", "contrabassoon", "bass_trombone", "natural_horn"):
        y = I.REGISTRY[name]["fn"](110.0, 0.3, DEFAULT_SR, 0.8)
        assert np.all(np.isfinite(y)) and np.max(np.abs(y)) > 0


def test_seq_articulation_shortens_only_notes_over_100_ms():
    base = {"type": "seq", "inst": "sine", "steps": "A4:0.05 A4:0.4", "step": 0.5}
    full = spec.render_sequence(dict(base), DEFAULT_SR, 1.0)
    art = spec.render_sequence(dict(base, articulation=0.5), DEFAULT_SR, 1.0)
    n = int(0.1 * DEFAULT_SR)
    assert np.allclose(art[:n], full[:n])
    assert np.sum(np.abs(art)) < np.sum(np.abs(full))
