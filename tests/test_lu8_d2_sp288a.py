import math

from genny import arms


def _omega(n, r_mm=20.0, h_mm=1.0, nu=0.3):
    c = arms.BRASS_C                       # sqrt(E / (12 rho (1 - nu^2)))
    f = arms.shell_modes(r_mm, h_mm, c)[n - 2]
    return 2.0 * math.pi * f * r_mm * 1e-3 / (c * math.sqrt(12.0))   # w R sqrt(rho (1 - nu^2) / E)


def test_shell_modes_match_infinite_shell_love_table():
    # R/h = 20, nu = 0.3: lowest radial frequency parameter of an infinite shell
    for n, ref in ((2, 0.0387307), (3, 0.109548), (4, 0.210077)):
        assert math.isclose(_omega(n), ref, rel_tol=5e-4)
