import numpy as np

from genny import chiptune as CT
from genny import chiptune2 as C2
from genny import chiptune_q as Q
from genny import spec as SP

SR = 44100


def _peak_hz(y, sr=SR):
    w = np.hanning(len(y))
    sp = np.abs(np.fft.rfft(y * w))
    k = int(np.argmax(sp))
    a, b, c = np.log(sp[k - 1] + 1e-12), np.log(sp[k] + 1e-12), np.log(sp[k + 1] + 1e-12)
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) * sr / len(y)


def _rms(y):
    return float(np.sqrt(np.mean(y ** 2)))


def _note(name, hz, patch, dur=1.0, fn=None, **kw):
    return fn({name: [(0.0, dur, hz, 1.0, patch)]}, dur, SR, **kw)


def test_tia_polynomial_periods():
    def run(audc, audf, n):
        seg = np.array([[0, audf, 15, audc, 0]], np.int64)
        return C2._tia_core(n, seg, np.array([0, 1, 1], np.int64), C2.TIA_BIT4, C2.TIA_BIT5, C2.POKEY_P9, C2.TIA_DIV31)

    def period(x):
        for p in range(1, 3000):
            if np.array_equal(x[200:200 + 2 * 1100], x[200 + p:200 + p + 2 * 1100]):
                return p
        return None
    assert len(C2.TIA_BIT4) == 15 and len(C2.TIA_BIT5) == 31 and len(C2.POKEY_P9) == 511
    assert period(run(1, 0, 6000)) == 15
    assert period(run(6, 0, 6000)) == 31 * 2 or period(run(6, 0, 6000)) == 31
    assert period(run(8, 0, 6000)) == 511
    assert period(run(2, 0, 6000)) == 465
    assert abs(C2.TIA_CLOCK["ntsc"] - 3579545 / 114) < 1e-9
    assert abs(C2.TIA_CLOCK["ntsc"] - 31399.5) < 0.1


def test_tia_pitch_and_volume_linear():
    p = dict(C2.TIA_PATCH, wave=4)
    y = _note("ch1", 220.0, p, fn=C2.render_tia)
    assert abs(_peak_hz(y[2000:]) - C2.tia_hz(220.0, 4)) < 1.0
    lo = _note("ch1", 220.0, dict(p, volume=7), fn=C2.render_tia)
    assert abs(_rms(lo) / _rms(y) - 7 / 15) < 0.03


def test_scc_pitch_volume_and_period_stop():
    assert Q.scc_period(440.0) == round(Q.SCC_CLOCK / (32 * 440.0)) - 1
    p = dict(Q.SCC_PATCH, wave="sine")
    y = _note("ch1", 440.0, p, fn=Q.render_scc)
    assert abs(_peak_hz(y[2000:]) - Q.scc_hz(440.0)) < 0.6
    half = _note("ch1", 440.0, dict(p, volume=7), fn=Q.render_scc)
    assert abs(_rms(half) / _rms(y) - 7 / 15) < 0.03
    assert Q.scc_hz(Q.SCC_CLOCK / 32 / 5) == 0.0                    # register <= 8: stopped
    assert _rms(_note("ch1", Q.SCC_CLOCK / 32 / 5, p, fn=Q.render_scc)) < 1e-6
    assert Q.SCC_WAVES["saw"].shape == (32,) and Q.SCC_WAVES["saw"].min() >= -128 and Q.SCC_WAVES["saw"].max() <= 127


def test_scc_channel5_shares_channel4_wave_unless_plus():
    lanes = {"ch4": [(0.0, 0.5, 220.0, 1.0, dict(Q.SCC_PATCH, wave="sine"))],
             "ch5": [(0.0, 0.5, 220.0, 1.0, dict(Q.SCC_PATCH, wave="square"))]}
    solo = Q.render_scc({"ch5": lanes["ch5"]}, 0.5, SR)
    shared = Q.render_scc({"ch4": lanes["ch4"], "ch5": lanes["ch5"]}, 0.5, SR)
    four = Q.render_scc({"ch4": lanes["ch4"]}, 0.5, SR)
    plus = Q.render_scc(lanes, 0.5, SR, plus=True)
    assert np.allclose(shared, 2 * four, atol=1e-6)               # ch5 plays ch4's sine, so it doubles it
    assert np.allclose(plus, four + solo, atol=1e-6)


def test_wsg_frequency_word_and_voices():
    f = Q.wsg_period(440.0)
    assert f == round(440.0 * 2 ** 20 / 96000)
    assert Q.wsg_period(440.0, "v2") % 16 == 0 and Q.wsg_period(440.0, "v1") != Q.wsg_period(440.0, "v2") or f % 16 == 0
    y = _note("v1", 440.0, dict(Q.WSG_PATCH, wave="sine"), fn=Q.render_wsg)
    assert abs(_peak_hz(y[2000:]) - Q.wsg_hz(440.0)) < 0.6
    half = _note("v1", 440.0, dict(Q.WSG_PATCH, wave="sine", volume=5), fn=Q.render_wsg)
    assert abs(_rms(half) / _rms(y) - 5 / 15) < 0.03
    assert all(len(w) == 32 and w.min() >= 0 and w.max() <= 15 for w in Q.WSG_WAVES.values()) and len(Q.WSG_WAVES) == 8
    rom = np.zeros(256)
    rom[32:64] = Q.WSG_WAVES["square"]
    z = Q.render_wsg({"v1": [(0.0, 0.3, 220.0, 1.0, dict(Q.WSG_PATCH, wave=1))]}, 0.3, SR, prom=rom)
    assert _rms(z) > 0.05


def test_huc6280_pitch_log_volume_and_noise():
    assert Q.hu_period(440.0) == round(Q.HU_CLOCK / (32 * 440.0))
    y = _note("ch1", 440.0, dict(Q.HU_PATCH, wave="sine"), fn=Q.render_hu)[:, 0]
    assert abs(_peak_hz(y[2000:]) - Q.hu_hz(440.0)) < 0.6
    a = Q._hu_amp(np.array([15, 14]))
    assert abs(20 * np.log10(a[0] / a[1]) - 3.0) < 1e-9             # two 1.5 dB steps per driver step
    pan = _note("ch1", 440.0, dict(Q.HU_PATCH, pan="L"), fn=Q.render_hu)
    assert _rms(pan[:, 1]) < 1e-9 < _rms(pan[:, 0])
    bits = Q.hu_noise_bits()
    assert len(bits) > 1000 and set(np.unique(bits)) == {0, 1}
    n = _note("ch5", 220.0, dict(Q.HU_PATCH, noise=True, noise_freq=16), dur=1.0, fn=Q.render_hu)[:, 0]
    assert _rms(n) > 0.02
    try:
        _note("ch1", 220.0, dict(Q.HU_PATCH, noise=True), dur=0.1, fn=Q.render_hu)
        raise AssertionError("noise on ch1 must be refused")
    except SP.SpecError:
        pass


def test_huc6280_lfo_modulates_channel1():
    p1 = dict(Q.HU_PATCH, wave="sine")
    lanes = {"ch1": [(0.0, 1.0, 440.0, 1.0, p1)], "ch2": [(0.0, 1.0, 16 * 6.0, 1.0, dict(Q.HU_PATCH, wave="sine", volume=0))]}
    off = Q.render_hu(lanes, 1.0, SR, lfo_mode=0)[:, 0]
    on = Q.render_hu(lanes, 1.0, SR, lfo_mode=2, lfo_freq=16)[:, 0]

    def spread(y):
        z = np.where(np.diff(np.sign(y[2000:])) > 0)[0]
        return float(np.std(np.diff(z)))
    assert spread(on) > 1.8 * spread(off)


def test_chip_layers_and_instruments_render():
    for system, ch, extra in (("scc", "ch1", {}), ("wsg", "v1", {}), ("huc6280", "ch1", {"lfo_mode": 0})):
        assert system in CT.SYSTEMS
        y = CT.chip_layer(dict({"system": system, "channels": {ch: {"steps": "C4:.2 E4:.2 G4:.2"}}}, **extra), SR, 1.0)
        assert np.isfinite(y).all() and np.abs(y).max() > 0.01
    for fn in (Q.scc, Q.namco_wsg, Q.huc6280):
        y = fn(330.0, 0.3, SR)
        assert np.isfinite(y).all() and np.abs(y).max() > 0.01
    y = Q.huc6280(330.0, 0.3, SR, noise=True)
    assert np.isfinite(y).all() and np.abs(y).max() > 0.01
