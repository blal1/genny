import numpy as np
from genny import tubes
from genny.physical import space
from genny import creatures


def test_mesh2d_wall_mullen():
    assert tubes.MESH2D_WALL == 0.7


def test_foliage_attenuation_eq7():
    f, F, a, L = 2000.0, 2.0, 0.024, 10.0
    ka = 2 * np.pi * f / space.C_GROUND * a
    assert abs(space.foliage_attenuation(np.array([f]), F, a, L)[0] - 0.1 * np.sqrt(F * L) * (ka + 0.9 * np.sqrt(ka))) < 1e-12


def test_variable_porosity_impedance():
    f = np.array([500.0])
    z = space.ground_impedance(f, 75.3, "variable_porosity", alpha=46.8)[0]
    assert z.real > 0 and z.imag > z.real
    assert abs(space.ground_impedance(f, 75.3, "variable_porosity", alpha=0.0)[0].real - z.real) < 1e-12


def test_laje_coupling_default_unchanged_and_active():
    plan = [dict(t=0.0, dur=0.06, f=((0, 3500), (1, 3000)))]
    n = int(0.1 * creatures.KSR)
    a, b = creatures._laje_source(plan, n), creatures._laje_source(plan, n, 0.0)
    c = creatures._laje_source(plan, n, 1.0)
    assert np.array_equal(a, b) and np.all(np.isfinite(c)) and not np.allclose(a, c)


def test_laje_coupling_subharmonics_fig54a():
    L = creatures.LAJE_COUPLING
    fs = 1.0e6

    def sub_db(eps, cpl):
        n = int(0.1 * fs)
        y = creatures._laje_syrinx_coupled(np.full(n, eps), np.full(n, 1e4), np.zeros(n), fs, cpl, L["alpha"], L["beta"],
                                           L["tau"], L["D"], L["gamma"], int(round(L["T"] * fs)))[n // 2:]
        s = np.abs(np.fft.rfft(y * np.hanning(y.size))) ** 2
        fr = np.fft.rfftfreq(y.size, 1 / fs)
        pk = fr[np.argmax(s)]
        m = lambda f: s[(fr > 0.96 * f) & (fr < 1.04 * f)].max()
        return 10 * np.log10(m(pk / 2) / m(pk))
    assert sub_db(3.6e8, 0.0) < -80 and sub_db(2.6e8, 1.0) < -80
    assert -40 < sub_db(3.6e8, 1.0) < -5
