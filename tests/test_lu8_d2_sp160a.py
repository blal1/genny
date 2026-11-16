import math

import numpy as np

from genny import contact as C
from genny import plates as P


def _lam(bc, kappa=20.0, n=6):
    mdl = P._model(44100, 1.0, bc, kappa, 0.0, 0.0, 0.0, 0.3)
    f = np.sort(P.scheme_modes(mdl, n + 3) * 2.0 * math.pi / kappa)
    return f[f > 0.5][:n]


def test_clamped_square_plate_frequencies_match_the_published_values():
    lam = _lam("cccc")
    assert abs(lam[0] / 35.998965 - 1.0) < 0.015
    assert abs(lam[1] / 73.405 - 1.0) < 0.025


def test_cantilever_square_plate_first_mode():
    assert abs(_lam("cfff")[0] / 3.494 - 1.0) < 0.015


def test_free_square_plate_modes_stay_within_one_percent():
    lam = _lam("ffff")
    for got, ref in zip(lam[:3], (13.4728, 19.5961, 24.2702)):
        assert abs(got / ref - 1.0) < 0.01


def test_free_disc_ratios_follow_the_material_poisson_ratio():
    for mat in ("steel", "bronze", "glass"):
        nu = C.get_material(mat).poisson
        b = C.ShapeBody("circ_plate_free", 100.0, material=mat, n_modes=4)
        ref = C.free_disc_omega((0, 1), nu) / C.free_disc_omega((2, 0), nu)
        assert abs(b.freqs[1] / b.freqs[0] - ref) < 1e-6
    b = C.ShapeBody("circ_plate_free", 100.0, material="bronze", n_modes=4)
    assert abs(b.freqs[1] / b.freqs[0] / 1.70 - 1.0) < 0.03
