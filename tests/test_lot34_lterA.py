import numpy as np

from genny import cyber, friction, spectral


def test_grid_tau():
    assert cyber.GRID_TAU == 300.0


def test_baschet_spiral():
    assert friction.BASCHET_SPIRAL["turns"] == 150


def test_onsets_bands():
    sr = 22050
    x = np.zeros(sr)
    x[sr // 4:sr // 4 + 200] = np.random.default_rng(0).standard_normal(200)
    assert len(spectral.onsets(x, sr, bands=6)) >= 1


def test_guns_and_blast_angle():
    import math

    from genny import foley

    assert foley.GUNS["smg_compact"][3] == 374.9 and "smg_sd" in foley.SUPPRESSED
    r = lambda g: float(np.abs(foley._gunshot("pistol", np.random.default_rng(1), 0.6, 0.0, 3.0, 0.0, angle=g)).max())
    assert r(180) < 0.3 * r(0)
    assert np.isfinite(foley.robot_babble(dur=0.5, style="farnell")).all()
