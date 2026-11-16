import math

import numpy as np

import genny
from genny import dsp, graphsolver as GS
from genny.fx import REGISTRY
from genny.physical import waveguides as W


def _string_mode_energy(signs):
    s = genny.CoupledStringBank((220.0, 220.0, 220.0), sr=22050, coupling=.1)
    n = int(22050 / 220)
    for i in range(n):
        for (dl, _), a in zip(s.lines, signs):
            dl.write(a * math.sin(2 * math.pi * i / n))
    e = []
    for _ in range(22050):
        inc = np.array([dl.read() for dl, _ in s.lines])
        s.bridge = .92 * s.bridge + .08 * float(np.mean(inc))
        for (dl, loss), x in zip(s.lines, inc):
            dl.write(loss.process(x - s.coupling * s.bridge))
        e.append(float(np.sum(inc ** 2)))
    e = np.array(e)
    return e[-2000:].mean() / e[:2000].mean()


def test_coupled_strings_in_phase_mode_decays_antiphase_rings():
    assert _string_mode_energy((1, 1, 1)) < 0.01 * _string_mode_energy((1, -1, 0))


def _piano_mode_energy(signs, coupling):
    sr = 22050
    p = genny.SympatheticPiano(220.0, unison_detune_cents=(0.0, 0.0, 0.0), sympathetic_ratios=(), sr=sr,
                               coupling=coupling)
    n = int(sr / 220)
    for i in range(n):
        for (dl, _), a in zip(p.lines, signs):
            dl.write(a * math.sin(2 * math.pi * i / n))
    e = []
    for _ in range(sr):
        inc = np.array([dl.read() for dl, _ in p.lines])
        p._scatter(inc, p.string_impedance_n_s_m, coupling)
        e.append(float(np.sum(inc ** 2)))
    e = np.array(e)
    return e[-1000:].mean() / e[:1000].mean()


def test_sympathetic_piano_two_stage_decay():
    free = _piano_mode_energy((1, -1, 0), 0.0)
    anti = _piano_mode_energy((1, -1, 0), 0.055)
    inph = _piano_mode_energy((1, 1, 1), 0.055)
    assert abs(anti - free) < 1e-3 * free      # antiphase: bridge still, string losses only
    assert inph < 0.5 * anti                   # in-phase: energy leaves through the bridge


def test_fractional_delays_stay_in_valid_range():
    d = GS.FractionalDelay(1.2, 32, 'lagrange3')
    for i in range(10):
        d.write(float(i))
    d.buf[d.w % len(d.buf)] = 1e9          # next write slot: never read
    assert abs(d.read()) < 1e6
    t = GS.ThiranDelay(1.02, 32)
    for _ in range(4):
        t.write(0.0)
    t.write(1.0)
    assert 0.0 <= t.read() < 1.0           # frac clipped to 0.1: coefficient (1-f)/(1+f) < 0.82


def test_wdf_oscillator_prewarped():
    sr = 48000
    imp = np.zeros(sr)
    imp[0] = 1.0
    for k, m in ((4000.0, 0.01), (4.0e6, 0.001)):
        f0 = math.sqrt(k / m) / (2 * math.pi)
        V = np.abs(np.fft.rfft(dsp.wdf_oscillator(imp, sr, m=m, k=k, r=0.05)))
        assert abs(np.argmax(V) - f0) < 2


def test_chorus_flanger_phaser_come_from_reverbs():
    for name in ('chorus', 'flanger', 'phaser'):
        assert REGISTRY[name]['fn'].__module__ == 'genny.reverbs'


def test_pasp_tables():
    assert W._pasp_reedtable(0.5, 0.3, 1.0) == 1.0
    assert abs(W._pasp_reedtable(0.0, 0.3, 1.0) - (1 - 0.3 / 1.3)) < 1e-12
    assert W._pasp_reedtable(-1.0, 0.3, 1.0) == 0.0
    assert W._pasp_bowtable(0.05, 0.1) == 1.0 and W._pasp_bowtable(1.2, 0.1) == 0.0
    assert abs(W._pasp_bowtable(0.55, 0.1) - 0.5) < 1e-12


def test_cubic_and_atan_shapes():
    v = np.array([-3.0, -1.0, 0.0, 0.5, 1.0, 3.0])
    c = dsp._SHAPES["cubic"](v)
    assert np.allclose(c, [-2 / 3, -2 / 3, 0.0, 0.5 - 0.125 / 3, 2 / 3, 2 / 3])
    assert np.allclose(dsp._SHAPES["atan"](np.array([0.0, 1.0])), [0.0, 0.5])
    y = dsp.overdrive(np.sin(np.linspace(0, 200, 4800)) * 0.9, 48000, shape="cubic") if hasattr(dsp, "overdrive") else None
    assert y is None or np.isfinite(y).all()


def test_pasp_reed_and_bow_play_in_tune():
    rng = np.random.default_rng(0)
    y = W.clarinet(220.0, 0.9, 0.6, rng, reed_table='pasp')
    assert np.sqrt(np.mean(y[-5000:] ** 2)) > 0.1
    assert abs(W.measure_f0(y[int(0.3 * W.SR):], 220.0) - 220.0) < 1.0
    y = W.bowed_string(220.0, 0.8, 0.5, 0.8, rng, body='none', vibrato=False, bow_table='pasp')
    assert np.sqrt(np.mean(y[-5000:] ** 2)) > 0.01
    assert abs(W.measure_f0(y[int(0.4 * W.SR):], 220.0) - 220.0) < 1.0
