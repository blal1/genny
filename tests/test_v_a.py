import numpy as np


from genny import chiptune_q as Q
from genny import chiptune_r as R

SR = 44100


def _rms(y):
    return float(np.sqrt(np.mean(np.square(y))))


def _peak_hz(y, sr=SR):
    sp = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    k = int(np.argmax(sp))
    a, b, c = np.log(sp[k - 1] + 1e-12), np.log(sp[k] + 1e-12), np.log(sp[k + 1] + 1e-12)
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) * sr / len(y)


def test_huc6280_balance_steps_of_3_db_and_silence():
    full = float(Q._hu_amp(np.array([15]))[0])
    assert abs(float(Q._hu_amp(np.array([15]), 15, 15)[0]) - full) < 1e-12
    a9, a10 = float(Q._hu_amp(np.array([15]), 9, 15)[0]), float(Q._hu_amp(np.array([15]), 10, 15)[0])
    assert abs(20 * np.log10(a10 / a9) - 3.0) < 0.01
    assert Q._hu_amp(np.array([15]), 0, 15)[0] == 0.0
    p = dict(Q.HU_PATCH, bal=(15, 7))
    y = Q.render_hu({"ch1": [(0.0, 0.5, 440.0, 1.0, p)]}, 0.5, SR)
    assert 20 * np.log10(_rms(y[:, 0]) / _rms(y[:, 1])) > 20.0
    y2 = Q.render_hu({"ch1": [(0.0, 0.5, 440.0, 1.0, dict(Q.HU_PATCH))]}, 0.5, SR, master=(15, 0))
    assert _rms(y2[:, 1]) < 1e-9 < _rms(y2[:, 0])


def test_huc6280_dda_plays_the_written_values():
    d = [31, 0] * 8
    p = dict(Q.HU_PATCH, dda=d, dda_hz=2000.0)
    y = Q.render_hu({"ch1": [(0.0, 0.5, 440.0, 1.0, p)]}, 0.5, SR)
    assert _rms(y[:, 0]) > 0.01


def test_vrc6_frequency_shift_register():
    assert abs(R.vrc6_hz(440.0, "p1", 4) - 440.0) / 440.0 < 0.002
    assert R.vrc6_period(440.0, "p1", 4) > 16 * R.vrc6_period(440.0, "p1", 0) - 20
    p = dict(R.VRC6_PATCH, volume=15)
    y = R.render_vrc6({"p1": [(0.0, 0.5, 440.0, 1.0, p)]}, 0.5, SR, shift=4)
    assert abs(_peak_hz(y[3000:]) - R.vrc6_hz(440.0, "p1", 4)) < 3.0


def test_fds_table_entry_applied_twice_per_cycle():
    tab = [1] + [0] * 31
    f0 = 220.0
    reg = R.fds_freq_reg(f0)
    kw = dict(mod_gain=63, mod_hz=2.0, mod_table=tab, mod_counter=0)
    lanes = {"fds": [(0.0, 1.0, f0, 1.0, dict(R.FDS_PATCH, **kw))]}
    y = R.render_fds(lanes, 1.0, SR)[int(0.55 * SR):int(0.95 * SR)]
    got = _peak_hz(y)
    want = R.CPU_NTSC * R.fds_mod_pitch(reg, 2, 63) / 2 ** 22
    other = R.CPU_NTSC * R.fds_mod_pitch(reg, 1, 63) / 2 ** 22
    assert abs(got - want) < abs(got - other)


def test_snes_decay_time_constant_is_a_tenth_per_table_time():
    y = R.render_snes(440.0, 1.5, SR, wave="sine", blocks=2, attack=15, decay=0, sustain=0, release_s=0.05)
    w = int(0.05 * SR)
    a = _rms(y[int(0.10 * SR):int(0.10 * SR) + w])
    b = _rms(y[int(0.40 * SR):int(0.40 * SR) + w])
    assert abs(b / a - np.exp(-0.30 / (1.2 / 2.3))) < 0.06


def test_opl_ksl_field_order_is_0_3_1p5_6_db_per_octave():
    def lvl(k):
        pt = dict(ops=[dict(tl=63), dict(ksl=k, ar=15, dr=0, sl=0, sus=1)], fb=0, cnt=0)
        p = dict(R.OPL_PATCH, patch=pt, tail=0.0)
        return _rms(R.render_opl({"ch1": [(0.0, 0.3, 2093.0, 1.0, p)]}, 0.3, SR))
    l0, l1, l2, l3 = (lvl(k) for k in range(4))
    assert l0 > l2 > l1 > l3


def test_saa1099_noise_is_a_deterministic_18_bit_register():
    a = R.saa1099(220.0, 0.2, SR, noise=1.0)
    b = R.saa1099(220.0, 0.2, SR, noise=1.0)
    assert np.array_equal(a, b) and _rms(a) > 0.01


def test_wonderswan_noise_sequence_lengths():
    from genny import instruments
    for taps, length in ((7, 28), (5, 63), (4, 73)):
        y = instruments.REGISTRY["wonderswan"]["fn"](1000.0, 0.5, 32000, channel="noise", taps=taps, mode="hardware_accurate")
        y = y[2000:8000] - np.mean(y[2000:8000])
        ac = lambda lag: float(np.dot(y[:-lag], y[lag:]) / np.dot(y, y))
        assert ac(length) > 0.8 and ac(length // 2 + 1) < ac(length)


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
