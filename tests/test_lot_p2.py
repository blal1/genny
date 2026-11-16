import numpy as np

from genny import chiptune_r as R
from genny import instruments, psychoacoustics_p2 as P

SR = 48000


def _tone(f, level, dur=1.0):
    t = np.arange(int(dur * SR)) / SR
    return np.sqrt(2.0) * 2e-5 * 10.0 ** (level / 20.0) * np.sin(2 * np.pi * f * t)


def test_moore_glasberg_one_sone_at_1khz_40db():
    n = P.moore_glasberg_loudness(_tone(1000.0, 40.0), SR)[0]
    assert abs(n - 1.0) < 0.05
    assert P.moore_glasberg_loudness(_tone(1000.0, 50.0), SR)[0] > 1.5 * n
    assert P.moore_glasberg_loudness(_tone(100.0, 40.0), SR)[0] < 0.3 * n


def test_moore_glasberg_tv_rises_then_settles():
    s, dt = P.moore_glasberg_loudness_tv(_tone(1000.0, 60.0, 0.5), SR)
    assert s[0] < s[len(s) // 2] and abs(s[-1] - s[-2]) < 0.05 * s[-1]


def test_hms_calibrations():
    assert abs(P.hms_loudness(_tone(1000.0, 40.0), SR)[0] - 1.0) < 0.01
    assert P.hms_loudness(_tone(1000.0, 60.0), SR)[0] > 3.0
    t = np.arange(SR) / SR
    am = lambda fm: _tone(1000.0, 60.0) * (1 + np.sin(2 * np.pi * fm * t)) / np.sqrt(1.5)
    assert abs(P.hms_roughness(am(70.0), SR)[0] - 1.0) < 0.02
    assert P.hms_roughness(am(70.0), SR)[0] > 2.0 * P.hms_roughness(am(10.0), SR)[0]
    assert P.hms_roughness(am(70.0), SR)[0] > 2.0 * P.hms_roughness(am(300.0), SR)[0]
    assert P.hms_roughness(_tone(1000.0, 60.0), SR)[0] < 0.01


def test_snes_gauss_table_is_the_documented_one():
    g = R.SNES_GAUSS
    assert g.size == 512 and g[0] == 0 and g[255] == 0x172 and g[511] == 0x519 and g[256] == 0x176
    for i in (0, 1, 100, 255):
        assert 0x7FF <= g[i] + g[255 - i] + g[256 + i] + g[511 - i] <= 0x801


def test_brr_filters_round_like_the_hardware():
    assert R.brr_predict(1, 100, 0) == 100 + ((-100) >> 4)
    assert R.brr_predict(2, 100, 50) == 200 + ((-300) >> 5) - 50 + (50 >> 4)
    assert R.brr_predict(3, 100, 50) == 200 + ((-1300) >> 6) - 50 + ((150) >> 4)


def test_snes_echo_delays_by_edl_steps_and_fir_filters():
    x = np.zeros(8000)
    x[0] = 1.0
    y = R.snes_echo(x, 1, 0.0, 1.0, "echo")
    assert abs(y[512 + 7] - 127.0 / 128.0) < 1e-9 and abs(y[0] - 1.0) < 1e-9
    y2 = R.snes_echo(x, 1, 0.5, 1.0, "echo")
    assert abs(y2[1038] - 0.5 * (127.0 / 128.0) ** 2) < 1e-6
    lp = R.snes_echo(x, 1, 0.0, 1.0, "lowpass")
    assert np.count_nonzero(np.abs(lp[512:1024]) > 1e-9) == 8


def test_opll_rom_matches_the_open_table():
    d = dict(R.OPLL_ROM)
    assert d["clarinet"] == "32211E06E0760028" and d["harpsichord"] == "13018311FAE41004" and d["electric_guitar"] == "61418903F1E44013"


def test_new_voices_render_and_behave():
    f = instruments.REGISTRY
    for n, kw in (("wonderswan", {}), ("wonderswan", {"channel": "noise", "taps": 3}), ("wonderswan", {"channel": "voice"}),
                  ("ppg_wave", {}), ("optigan", {}), ("chamberlin", {}), ("chamberlin", {"sound": "strings"})):
        y = f[n]["fn"](220.0, 0.6, 22050, 1.0, **kw)
        assert np.isfinite(y).all() and np.abs(y).max() > 0.05, (n, kw)
    assert abs(R.chiptune_p2.ws_hz(440.0) - 440.0) < 2.0
    y = f["wonderswan"]["fn"](440.0, 0.5, 44100, 1.0, mode="hardware_accurate")
    z = y[2000:2000 + 16384]; k = np.argmax(np.abs(np.fft.rfft(z * np.hanning(len(z)))))
    assert abs(k * 44100 / len(z) - 440.0) < 5.0
    tab = R.chiptune_p2.ppg_table("harmonics")
    assert tab.shape == (64, 64) and np.allclose(tab * 127, np.round(tab * 127))
    assert np.abs(np.fft.rfft(tab[40])[10:30]).max() > 5 * np.abs(np.fft.rfft(tab[3])[10:30]).max()
    a = f["chamberlin"]["fn"](330.0, 12.0, 8000, 1.0)
    assert len(a) < 8.2 * 8000


def test_church_organ_light_voicing_changes_only_when_asked():
    f = instruments.REGISTRY["church_organ"]["fn"]
    d0 = f(220.0, 0.8, 22050, 1.0)
    assert np.allclose(d0, f(220.0, 0.8, 22050, 1.0, voicing="default"))
    assert not np.allclose(d0, f(220.0, 0.8, 22050, 1.0, voicing="light"))
    lo, hi = f(220.0, 0.8, 22050, 1.0, voicing="light", mouth=0.0), f(220.0, 0.8, 22050, 1.0, voicing="light", mouth=1.0)
    h = lambda y: np.abs(np.fft.rfft(y[2000:2000 + 16384]))[[round(220.0 * k * 16384 / 22050) for k in (1, 2)]]
    assert h(lo)[1] / h(lo)[0] > h(hi)[1] / h(hi)[0]


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            try:
                v()
                print("ok  ", k)
            except Exception as e:
                import traceback
                print("FAIL", k, repr(e))
                traceback.print_exc()
