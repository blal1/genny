import math

import numpy as np

from genny import contact as C
from genny import foley as Fo


def test_layer_k_sphere_is_quadratic_and_cubic():
    k, al = C.layer_k(1e6, 0.3, 1e-3, 0.01)
    assert al == 2.0 and math.isclose(k, math.pi * 1e6 * 0.01 / 1e-3)
    k, al = C.layer_k(1e6, 0.5, 1e-3, 0.01, bonded=True)
    assert al == 3.0 and math.isclose(k, 2 * math.pi / 3 * 1e6 * 1e-4 / 1e-9)


def test_cavity_mean_pressure_meets_hertz_and_caps():
    r, y, es = 0.005, 1e9, 115e9
    a4 = 4.0 * y * r / es
    lo = C.elastoplastic_mean_pressure(a4 * 0.999, r, y, es)
    hi = C.elastoplastic_mean_pressure(a4 * 1.001, r, y, es)
    assert abs(hi - lo) < 0.02 * y
    assert C.elastoplastic_mean_pressure(40 * y * r / es, r, y, es) == 3 * y


def test_plastic_radius_goes_from_hertz_to_twice():
    r, y, es = 0.005, 1e9, 115e9
    d = 1e-9
    assert math.isclose(C._plastic_radius(d, r, y, es) ** 2, r * d)
    d = 0.1 * r
    assert math.isclose(C._plastic_radius(d, r, y, es) ** 2, 2 * r * d)


def test_wavy_2d_follows_numerical_curve():
    ps = 2 * math.pi * 1e9 * 1e-6 / 1e-3
    assert math.isclose(C.wavy_contact(0.4 * ps, 1e9, 1e-6, 1e-3, two_d=True), 0.62)


def test_cloth_fabric_default_unchanged_and_polyester_tone():
    a = Fo.cloth(dur=0.5, seed=1)
    b = Fo.cloth(dur=0.5, seed=1, fabric="none")
    assert np.array_equal(a, b)
    c = Fo.cloth(dur=0.5, seed=1, fabric="polyester")
    assert c.shape == a.shape and not np.array_equal(a, c)
    e = np.ones(4800)
    y = Fo._fabric_tone(e, "corduroy", np.random.default_rng(0))
    f = np.argmax(np.abs(np.fft.rfft(y))) * Fo.P / y.size
    assert abs(f - 1.3 / 0.00175) < 30
