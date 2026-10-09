import numpy as np
import genny
from genny.graphsolver import WaveguideBranch, WaveguideNetwork

def test_network_finite_and_passive_decay():
    bs=[WaveguideBranch(1.0,17.3,128,.995,.98), WaveguideBranch(1.5,23.7,128,.994,.975), WaveguideBranch(2.0,31.2,128,.993,.97)]
    net=WaveguideNetwork(bs)
    y=net.render_impulse(4000)
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) < 2.0
    assert np.max(np.abs(y[-500:])) < np.max(np.abs(y[:500]))

def test_exports_and_version():
    assert tuple(map(int, genny.__version__.split('.'))) >= (0, 12, 0)
    assert genny.WaveguideNetwork is WaveguideNetwork
