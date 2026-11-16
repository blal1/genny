import inspect

import numpy as np

from genny import analysis as A
from genny import creatures as C
from genny import rail as R


def test_yin_sine_subsample():
    sr = 16000
    t = np.arange(int(0.5 * sr)) / sr
    f = A.yin(np.sin(2 * np.pi * 441.3 * t), sr)
    assert abs(float(np.median(f)) - 441.3) < 1.5


def test_growl_mass_sets_the_dog_tract():
    assert "mass" in inspect.signature(C.growl).parameters
    assert abs(C.dog_vtl(15.0) - (3.15 + 11.53 * np.log10(15.0)) / 100) < 1e-9


def test_curve_squeal_harmonics_option():
    assert "harmonics" in inspect.signature(R.curve_squeal).parameters
    assert R.HARMONIC_DB == {2: -33.0, 3: -40.0}
