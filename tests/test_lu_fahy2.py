import math

from genny import cyber
from genny import plates as PL

STEEL = (7850.0, 2.1e11, 0.3)


def test_blocking_mass_corner_is_z_over_2_pi_m():
    B = 2.1e11 * 0.005 ** 3 / (12.0 * (1.0 - 0.09))
    z = 8.0 * math.sqrt(B * 7850.0 * 0.005)
    assert abs(cyber.blocking_mass_fc(1.0) - z / (2.0 * math.pi)) < 1e-6
    assert abs(cyber.blocking_mass_fc(2.0) - 195.5) < 1.0


def test_stiffeners_raise_subcritical_radiation_by_maidanik_factor():
    fc = PL.coincidence(1e-3, STEEL)
    plain = float(PL.radiation_efficiency(0.05 * fc, 1e-3, 1.0, 4.0, STEEL))
    ribbed = float(PL.radiation_efficiency(0.05 * fc, 1e-3, 1.0, 4.0, STEEL, stiffeners=2.0))
    assert abs(ribbed / plain - 2.0) < 1e-9
    assert float(PL.radiation_efficiency(0.9 * fc, 1e-3, 1.0, 4.0, STEEL, stiffeners=1000.0)) == 1.0   # sigma 0.011 x 501
    assert float(PL.radiation_efficiency(2.0 * fc, 1e-3, 1.0, 4.0, STEEL, stiffeners=2.0)) == 1.0
    assert abs(float(PL.radiation_damping(0.05 * fc, 1e-3, 1.0, 4.0, STEEL, stiffeners=2.0))
               / float(PL.radiation_damping(0.05 * fc, 1e-3, 1.0, 4.0, STEEL)) - 2.0) < 1e-9
