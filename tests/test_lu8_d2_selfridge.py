import numpy as np

from genny import sfx as S
from genny.physical import modal


def _chord_used(monkeypatch, **kw):
    seen = {}

    def fake(rpm, blades, radius, power, dur, rng, **k):
        seen.update(k)
        return np.ones(int(dur * 44100))

    monkeypatch.setattr(modal, "propeller_tone", fake)
    S.propeller(dur=0.1, **kw)
    return seen["chord"]


def test_c130_blade_chord_is_047(monkeypatch):
    assert _chord_used(monkeypatch, aircraft="c130") == 0.47
    assert _chord_used(monkeypatch, aircraft="c130", chord=0.3) == 0.3
    assert _chord_used(monkeypatch, aircraft="p51") == 0.15


def test_aeolian_tone_gamma_scales_wake():
    def wake_only(g):
        y = modal.aeolian_tone(30.0, 0.01, 0.3, np.random.default_rng(1), physical=True, wake=True, gamma=g)
        return y - modal.aeolian_tone(30.0, 0.01, 0.3, np.random.default_rng(1), physical=True)

    r = np.std(wake_only(0.2)) / np.std(wake_only(1e-4))
    assert 40 < r < 50  # sqrt(0.2 / 1e-4) = 44.7


def test_thesis_table_values_hold():
    t = modal.AIRCRAFT
    assert t["c130"][:4] == (164.0, 1020.0, 4590.0, 2.06) and t["cessna340"][:4] == (125.0, 2500.0, 310.0, 0.96)
    assert abs(modal.corr_length(1e4, 1.0) - 10 ** 1.536 * 1e4 ** -0.245) < 1e-12
