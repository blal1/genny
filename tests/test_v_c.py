"""V-C checks (Paula period floor, DX7 key scaling, SNES GAIN, SID 6581 cutoff, PPG table size). Run as a script: `uv run python tests/test_v_c.py`."""
import numpy as np

from genny import chiptune2 as C2
from genny import chiptune_p2 as P2
from genny import chiptune_r as R
from genny import synth_s as S

SR = 44100


def test_paula_period_floors():
    assert R.paula_period(1e6, 32) == 113
    assert R.paula_period(1e6, 32, "pal", dma=True) == 123 and R.paula_period(1e6, 32, "ntsc", dma=True) == 124
    assert R.paula_period(440.0, 32, "pal", dma=True) == R.paula_period(440.0, 32, "pal")
    assert abs(R.paula_hz(1e6, 32, "pal", dma=True) - 3546895.0 / (123 * 32)) < 1e-6
    y = R.render_paula({"ch1": [(0.0, 0.2, 5000.0, 1.0, dict(R.PAULA_PATCH, dma=True))]}, 0.2, SR)
    assert np.isfinite(y).all()


def test_dx7_key_scale_matches_the_published_indexing():
    f = S.dx7_key_scale
    assert [f(n, 39, 99, 99, 0, 0) for n in (54, 55, 56, 57, 58, 59)] == [-7, 0, 0, 0, -7, -7]
    assert f(39 + 17 + 59, 39, 0, 99, 3, 3) == (20 * 99 * 329) >> 12
    assert f(39 + 17 + 59, 39, 0, 99, 2, 2) == (66 * 99 * 329) >> 15
    assert f(39 + 17 + 59, 39, 0, 99, 0, 0) == -f(39 + 17 + 59, 39, 0, 99, 3, 3)
    assert f(39 + 17 + 59, 39, 0, 99, 1, 1) == -f(39 + 17 + 59, 39, 0, 99, 2, 2)
    assert f(39 + 17 + 97, 39, 0, 99, 2, 2) == (250 * 99 * 329) >> 15


def test_saa1099_frequency_is_clock_over_512():
    # MAME: a half period lasts (511 - n) << (8 - oct) counts, the counter loses 256 per sample, 8 MHz / 256 samples per second
    for o, n in ((0, 0), (4, 120), (7, 255)):
        half_samples = ((511 - n) << (8 - o)) / 256.0
        assert abs(R.SAA_CLOCK / 256.0 / (2 * half_samples) - R.SAA_CLOCK / 512.0 * 2.0 ** o / (511 - n)) < 1e-9
    assert abs(R.SAA_CLOCK / 512.0 / 511 - 30.6) < 0.1 and abs(R.SAA_CLOCK / 512.0 * 128 / 256 - 7812.5) < 1e-9


def test_snes_gain_modes():
    e = R.snes_gain_env
    d = e(0x40 // 2, 100)                       # direct, level 16 N
    assert abs(d[0] - 32 * 16 / 2048.0) < 1e-12 and np.all(d == d[0])
    up = e(128 + 64 + 31, 200)                  # linear increase, rate 31: +32 every sample
    assert abs(up[0] - 32 / 2048.0) < 1e-12 and abs(up[10] - 11 * 32 / 2048.0) < 1e-12
    bent = e(128 + 96 + 31, 400)                # bent: +32 below 0x600, then +8
    k = int(np.argmax(bent * 2048 >= 0x600))
    assert abs(bent[k] * 2048 - 0x600) < 1e-9 and abs((bent[k + 1] - bent[k]) * 2048 - 8) < 1e-9
    top = e(128 + 96 + 31, 2000)
    assert abs(top[-1] * 2048 - 0x7FF) < 1e-9
    lin = e(128 + 0 + 31, 50)                   # linear decrease from level 0 stays at 0
    assert np.all(lin == 0)
    ex = R.SNES_RATE_PERIOD
    assert ex[0] == 0 and ex[1] == 2048 and ex[16] == 64 and ex[31] == 1 and len(ex) == 32
    y = R.render_snes(440.0, 0.3, SR, wave="sine", gain=128 + 64 + 28)
    assert np.isfinite(y).all() and np.sqrt(np.mean(y[-3000:] ** 2)) > np.sqrt(np.mean(y[:300] ** 2))


def test_sid_cutoff_curves():
    a = C2.sid_cutoff_hz(np.array([0, 128, 768, 1023, 1024, 2047]), "6581")
    assert list(a) == [220, 230, 1600, 6000, 4600, 18000]
    b = C2.sid_cutoff_hz(np.array([0, 2047]), "8580")
    assert abs(b[0] - 12500 / 2048) < 1e-9 and abs(b[1] - 12500) < 1e-9
    f = lambda chip: C2.render_sid({"voice1": [(0.0, 0.5, 220.0, 1.0, dict(C2.SID_PATCH, wave="saw"))]}, 0.6, SR, cutoff=300,
                                   res=0, mode="lp", routed=(1,), chip=chip)
    lo, hi = f("6581"), f("8580")      # register 300: 6581 about 270 Hz, 8580 about 1.8 kHz
    assert np.sqrt(np.mean(lo ** 2)) < 0.8 * np.sqrt(np.mean(hi ** 2))


def test_ppg_table_is_64_by_64():
    t = P2.ppg_table("resonant")
    assert t.shape == (64, 64) and P2.PPG_LEN == 64


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
