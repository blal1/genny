"""Values read in the velvet-noise, loss-filter, Leslie and surf documents (LU_DSP)."""
import numpy as np
from scipy.signal import butter, freqz, sosfilt

from genny import reverbs as R
from genny import strings as S

SR = 44100


def _t60(y, sr):
    e = np.cumsum((y ** 2)[::-1])[::-1]
    db = 10 * np.log10(e / e[0] + 1e-30)
    i5, i25 = np.argmax(db < -5), np.argmax(db < -25)
    return 3.0 * (i25 - i5) / sr


def test_pori_octave_t60_table():
    assert dict(R.HALL_T60["pori"]) == {125.0: 2.7, 250.0: 2.5, 500.0: 2.4, 1000.0: 2.3, 2000.0: 2.1, 4000.0: 1.7, 8000.0: 1.1}
    ir = R.velvet_ir(SR, 2.5, 1.7, seed=1, t60_bands=R.HALL_T60["pori"])
    for fc, want in ((1000.0, 2.3), (8000.0, 1.1)):
        b = sosfilt(butter(4, (fc / 1.26, fc * 1.26), "band", fs=SR, output="sos"), ir)
        assert abs(_t60(b, SR) / want - 1) < 0.15, (fc, _t60(b, SR))


def test_pori_preset_uses_bands():
    x = np.zeros(SR // 10)
    x[0] = 1.0
    y = R.velvet_reverb(x, SR, preset="pori", mix=1.0, width=0.0)[:, 0]
    b = sosfilt(butter(4, (8000 / 1.26, 8000 * 1.26), "band", fs=SR, output="sos"), y)
    assert _t60(b, SR) < 1.4                                    # held 4 kHz value would give ~1.7 s


def test_leslie_herrera_speeds_and_cabinet():
    assert R._LESLIE_RATES["fast"] == (6.92, 5.9)
    assert R._LESLIE_TAU[0] == 1.0
    assert R._CAB == 0.265


def test_leslie_crossover_12db_per_octave():
    imp = np.zeros(8192)
    imp[0] = 1.0
    lo, hi = R._leslie_split(imp, 800.0, SR)
    f = np.array([200.0, 3200.0])
    _, Hh = freqz(hi, worN=f, fs=SR)
    _, Hl = freqz(lo, worN=f, fs=SR)
    assert abs(20 * np.log10(abs(Hh[0])) + 24.6) < 1.0      # two octaves below: -24.6 dB, not -48
    assert abs(20 * np.log10(abs(Hl[1])) + 24.6) < 1.0
    _, Hs = freqz(lo + hi, worN=np.array([100.0, 800.0, 5000.0]), fs=SR)
    assert np.allclose(np.abs(Hs), 1.0, atol=0.02)


def test_loss_fir_decay_time_weights():
    f0 = 330.0
    k = np.arange(1, 40)
    tau = 4.0 / (1 + 0.0004 * (k * f0) ** 1.3)
    h = S.loss_fir(f0, tau, SR)
    fk = k * f0
    m = fk < 0.45 * SR
    _, H = freqz(h, worN=fk[m], fs=SR)
    that = -1.0 / (f0 * np.log(np.minimum(np.abs(H), 0.999999)))
    err = np.abs(that / tau[m] - 1)[:10].mean()
    assert np.abs(H).max() < 1.0 and err < 0.14, err          # 1/f weights: 0.19
