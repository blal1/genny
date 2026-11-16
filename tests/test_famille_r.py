import numpy as np
import pytest

from genny import chiptune as CT
from genny import chiptune_r as R
from genny import spec as SP

SR = 44100


def _peak_hz(y, sr=SR):
    w = np.hanning(len(y))
    sp = np.abs(np.fft.rfft(y * w))
    k = int(np.argmax(sp))
    a, b, c = np.log(sp[k - 1] + 1e-12), np.log(sp[k] + 1e-12), np.log(sp[k + 1] + 1e-12)
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) * sr / len(y)


def _rms(y):
    return float(np.sqrt(np.mean(np.asarray(y) ** 2)))


def _one(patch0, name, hz, dur=1.0, fn=None, **kw):
    p = dict(patch0)
    p.update({k: v for k, v in kw.items() if k in patch0})
    extra = {k: v for k, v in kw.items() if k not in patch0}
    return fn({name: [(0.0, dur, hz, 1.0, p)]}, dur, SR, **extra)


def test_vrc6_pitch_duty_and_saw_levels():
    for hz in (220.0, 440.0):
        y = _one(R.VRC6_PATCH, "p1", hz, fn=R.render_vrc6)
        assert abs(_peak_hz(y[2000:]) - R.vrc6_hz(hz)) < 0.6
    ys = _one(R.VRC6_PATCH, "saw", 220.0, fn=R.render_vrc6)
    assert abs(_peak_hz(ys[2000:]) - R.vrc6_hz(220.0, "saw")) < 0.6
    assert R.vrc6_period(220.0, "saw") != R.vrc6_period(220.0, "p1")
    for duty in range(8):
        y = _one(R.VRC6_PATCH, "p2", 100.0, fn=R.render_vrc6, duty=duty)[3000:]
        mid = 0.5 * (y.max() + y.min())
        frac = float(np.mean(y > mid))
        assert abs(frac - (duty + 1) / 16.0) < 0.03, (duty, frac)
    # the saw has 7 distinct levels (6 adds, the rate 42 at full volume): 0 .. 6 * 42 >> 3 = 31
    lanes = {"saw": [(0.0, 1.0, 220.0, 1.0, dict(R.VRC6_PATCH))]}
    raw = R.render_vrc6(lanes, 1.0, SR)[3000:]
    lev = np.unique(np.round(raw / raw.max() * 31 / 0.3 * 0.3, 0))
    assert len(np.unique(np.round(raw * 20, 1))) >= 7


def test_n163_pitch_depends_on_channel_count():
    f = R.n163_freq_reg(220.0, 1, 32)
    hz1 = R.n163_hz(220.0, 1, 32)
    y1 = _one(R.N163_PATCH, "ch1", 220.0, fn=R.render_n163, n=1)
    assert abs(_peak_hz(y1[2000:]) - hz1) < 0.5
    # the same register on a chip running 4 channels plays four times slower
    p = dict(R.N163_PATCH)
    lanes = {"ch1": [(0.0, 1.0, 55.0, 1.0, p)]}
    y4 = R.render_n163(lanes, 1.0, SR, n=4)
    assert abs(_peak_hz(y4[2000:]) - R.n163_hz(55.0, 4, 32)) < 0.5
    assert R.n163_freq_reg(55.0, 4, 32) == R.n163_freq_reg(220.0, 1, 32) or abs(R.n163_freq_reg(55.0, 4, 32) - f) <= 1
    # level falls as 1 / n for the same note
    a = _rms(R.render_n163({"ch1": [(0.0, 1.0, 220.0, 1.0, p)]}, 1.0, SR, n=1))
    b = _rms(R.render_n163({"ch1": [(0.0, 1.0, 220.0, 1.0, p)]}, 1.0, SR, n=4))
    assert abs(b / a - 0.25) < 0.03
    with pytest.raises(SP.SpecError):
        R.render_n163({"ch1": [(0.0, 0.2, 220.0, 1.0, dict(p, length=30))]}, 0.2, SR)


def test_fds_pitch_and_modulator():
    y = _one(R.FDS_PATCH, "fds", 220.0, fn=R.render_fds)
    assert abs(_peak_hz(y[2000:]) - R.fds_hz(220.0)) < 0.6
    assert R.fds_mod_pitch(1000, 16, 32) == 1500
    assert R.fds_mod_pitch(1000, -16, 32) == 500
    assert R.fds_mod_pitch(1000, 0, 63) == 1000
    # counter set to +16 with the modulator halted: the note plays at 1.5 x its pitch (and 0.5 x with -16)
    ym = _one(R.FDS_PATCH, "fds", 220.0, fn=R.render_fds, mod_gain=32, mod_counter=16, mod_hz=0.0)
    assert abs(_peak_hz(ym[3000:]) - 1.5 * R.fds_hz(220.0)) < 1.5
    yl = _one(R.FDS_PATCH, "fds", 220.0, fn=R.render_fds, mod_gain=32, mod_counter=-16, mod_hz=0.0)
    assert abs(_peak_hz(yl[3000:]) - 0.5 * R.fds_hz(220.0)) < 1.5
    # vibrato (tri table) spreads the spectrum: more zero-crossing jitter than without it
    def zc_sd(x):
        z = np.where(np.diff(np.signbit(x)))[0]
        return float(np.std(np.diff(z)))
    yv = _one(R.FDS_PATCH, "fds", 220.0, fn=R.render_fds, mod_gain=20, mod_table="tri", mod_hz=6.0, wave="triangle")
    y0 = _one(R.FDS_PATCH, "fds", 220.0, fn=R.render_fds, wave="triangle")
    assert zc_sd(yv[2000:]) > 2.0 * zc_sd(y0[2000:])
    # volume 32 / 15 steps: linear
    lo = _rms(_one(R.FDS_PATCH, "fds", 220.0, fn=R.render_fds, volume=7))
    hi = _rms(_one(R.FDS_PATCH, "fds", 220.0, fn=R.render_fds, volume=15))
    assert abs(lo / hi - 14 / 30) < 0.03
    m1 = _rms(_one(R.FDS_PATCH, "fds", 220.0, fn=R.render_fds, master=2))
    assert abs(m1 / hi - 0.5) < 0.02


def test_paula_period_pitch_filters_and_modulation():
    assert abs(R.PAULA_CLOCK["pal"] / 428 - 8287.1) < 0.1
    assert R.paula_period(440.0, 32) == round(3546895 / (440.0 * 32))
    assert R.paula_period(1e6, 32) == 113
    for sysn in ("pal", "ntsc"):
        y = _one(R.PAULA_PATCH, "ch1", 220.0, fn=R.render_paula, wave="sine", filter="none", system=sysn)
        assert abs(_peak_hz(y[2000:, 0] + y[2000:, 1]) - R.paula_hz(220.0, 32, sysn)) < 0.6
    # channels 1 and 4 are left, 2 and 3 right
    y = R.render_paula({"ch2": [(0.0, 0.3, 220.0, 1.0, dict(R.PAULA_PATCH))]}, 0.3, SR)
    assert _rms(y[:, 0]) < 1e-6 < _rms(y[:, 1])
    # filters: A500 cuts 8 kHz by about 6 dB per octave from 4.9 kHz, the LED filter much more
    sq = dict(R.PAULA_PATCH, wave="square")
    def band(**kw):
        y = _one(sq, "ch1", 1000.0, fn=R.render_paula, **kw)[:, 0]
        sp = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        f = np.fft.rfftfreq(len(y), 1 / SR)
        return float(sp[(f > 6800) & (f < 7200)].max()), float(sp[(f > 900) & (f < 1100)].max())
    h0, f0 = band(filter="none")
    h1, f1 = band(filter="a500")
    h2, f2 = band(filter="a500", led=True)
    assert 0.3 < (h1 / f1) / (h0 / f0) < 0.75
    assert (h2 / f2) < 0.4 * (h1 / f1)
    # hard sample-and-hold: with 64 as the volume of a constant sample, level is 64 / 64 of full scale
    full = _rms(_one(R.PAULA_PATCH, "ch1", 220.0, fn=R.render_paula, wave="square", volume=64, filter="none"))
    half = _rms(_one(R.PAULA_PATCH, "ch1", 220.0, fn=R.render_paula, wave="square", volume=32, filter="none"))
    assert abs(half / full - 0.5) < 0.02
    # AM: channel 1's words are channel 2's volume (constant 32 = half)
    lanes = {"ch1": [(0.0, 1.0, 220.0, 1.0, dict(R.PAULA_PATCH, wave=[32, 32], modulates="am"))],
             "ch2": [(0.0, 1.0, 220.0, 1.0, dict(R.PAULA_PATCH, wave="square", volume=64))]}
    am = R.render_paula(lanes, 1.0, SR, filter="none")
    ref = R.render_paula({"ch2": [(0.0, 1.0, 220.0, 1.0, dict(R.PAULA_PATCH, wave="square", volume=64))]}, 1.0, SR, filter="none")
    assert _rms(am[:, 0]) < 1e-6 and abs(_rms(am[:, 1]) / _rms(ref[:, 1]) - 0.5) < 0.03
    # FM: channel 1's words replace channel 2's period (428 -> 3546895 / 428 / 2 for a 2-byte wave)
    lanes = {"ch1": [(0.0, 1.0, 220.0, 1.0, dict(R.PAULA_PATCH, wave=[428, 428], modulates="fm"))],
             "ch2": [(0.0, 1.0, 220.0, 1.0, dict(R.PAULA_PATCH, wave=[127, -128]))]}
    fm = R.render_paula(lanes, 1.0, SR, filter="none")
    assert abs(_peak_hz(fm[2000:, 1]) - 3546895 / 428 / 2) < 5.0


def _make_mod(rows, nch=4, sample=None, loop=True, vol=64, ft=0):
    """rows: list of [(sample, period, effect, param)] per channel for pattern 0; one pattern, order [0]."""
    smp = np.round(127 * np.sin(2 * np.pi * np.arange(32) / 32)).astype(np.int8) if sample is None else np.asarray(sample, np.int8)
    head = bytearray(20) + bytearray(31 * 30)
    h = bytearray(30)
    h[22:24] = (len(smp) // 2).to_bytes(2, "big")
    h[24], h[25] = ft, vol
    h[26:28] = (0).to_bytes(2, "big")
    h[28:30] = ((len(smp) // 2) if loop else 1).to_bytes(2, "big")
    head[20:50] = h
    blob = bytearray(head)
    blob += bytes([1, 0]) + bytes(128)
    blob += b"M.K." if nch == 4 else (b"6CHN" if nch == 6 else b"8CHN")
    pat = bytearray(64 * nch * 4)
    for r, row in enumerate(rows):
        for c, (s, per, fx, pa) in enumerate(row):
            o = (r * nch + c) * 4
            pat[o:o + 4] = bytes([(s & 0xF0) | (per >> 8), per & 255, ((s & 15) << 4) | fx, pa])
    blob += pat + smp.tobytes()
    return bytes(blob)


E = (0, 0, 0, 0)


def test_mod_parse_pitch_volume_and_timing():
    mod = _make_mod([[(1, 428, 0, 0), E, E, E], [E, E, E, E], [E, E, E, (0, 0, 13, 0)]])
    d = R.mod_parse(mod)
    assert d["nch"] == 4 and d["order"] == [0] and d["samples"][0]["volume"] == 64 and len(d["samples"][0]["data"]) == 32
    y = R.render_mod(mod, SR, filter="none")
    # three rows of 6 ticks of 2.5 / 125 s, ended by the pattern break
    assert abs(len(y) / SR - 3 * 6 * 0.02) < 0.002
    mod2 = _make_mod([[(1, 428, 0, 0), E, E, E]] + [[E] * 4] * 63)
    y = R.render_mod(mod2, SR, filter="none")
    left = y[1000:, 0]
    assert abs(_peak_hz(left) - R.PAULA_CLOCK["pal"] / 428 / 32) < 0.3
    assert _rms(y[:, 1]) < 1e-9
    # set volume C20 halves the level
    half = R.render_mod(_make_mod([[(1, 428, 12, 32), E, E, E]] + [[E] * 4] * 63), SR, filter="none")
    assert abs(_rms(half[1000:, 0]) / _rms(y[1000:, 0]) - 0.5) < 0.02
    # tempo F 150 shortens the 64 rows from 7.68 s to 6.4 s... F with speed 3 halves it
    fast = R.render_mod(_make_mod([[(1, 428, 15, 3), E, E, E]] + [[E] * 4] * 63), SR, filter="none")
    assert abs(len(fast) / SR - 64 * 3 * 0.02) < 0.01
    # NTSC clock lifts the pitch by 3579545 / 3546895
    yn = R.render_mod(mod2, SR, system="ntsc", filter="none")
    assert abs(_peak_hz(yn[1000:, 0]) - R.PAULA_CLOCK["ntsc"] / 428 / 32) < 0.3


def test_mod_effects_slide_arpeggio_vibrato():
    base = [[(1, 428, 0, 0), E, E, E]]
    def seg(row_fx, secs):
        mod = _make_mod(base + [[(0, 0, *row_fx), E, E, E]] * 10 + [[(0, 0, 13, 0), E, E, E]])
        return R.render_mod(mod, SR, filter="none")[:, 0]
    up = seg((1, 8), 0)          # period falls: the pitch rises
    n = len(up)
    a = _peak_hz(up[int(0.15 * SR):int(0.3 * SR)])
    b = _peak_hz(up[int(1.1 * SR):int(1.3 * SR)])
    assert b > 1.5 * a
    dn = seg((2, 8), 0)
    assert _peak_hz(dn[int(1.1 * SR):int(1.3 * SR)]) < 0.8 * _peak_hz(dn[int(0.15 * SR):int(0.3 * SR)])
    # arpeggio 047: three pitches C, E, G (period 428 -> 340 -> 285 about) cycling every tick
    arp = seg((0, 0x47), 0)
    def zp_peak(x):
        sp = np.abs(np.fft.rfft(x * np.hanning(len(x)), 1 << 17))
        return float(np.argmax(sp[:int(1500 / SR * (1 << 17))])) * SR / (1 << 17)
    ticks = [zp_peak(arp[int((0.12 + k * 0.02) * SR) + 40:int((0.12 + (k + 1) * 0.02) * SR) - 40]) for k in range(3)]
    assert abs(ticks[0] - R.PAULA_CLOCK["pal"] / 428 / 32) < 25
    assert abs(ticks[1] / ticks[0] - 2 ** (4 / 12)) < 0.05 and abs(ticks[2] / ticks[0] - 2 ** (7 / 12)) < 0.05
    # vibrato 4xy: the pitch wanders: zero-crossing intervals spread out
    vb = seg((4, 0x4A), 0)[int(0.2 * SR):int(1.0 * SR)]
    st = seg((0, 0), 0)[int(0.2 * SR):int(1.0 * SR)]
    def sd(x):
        z = np.where(np.diff(np.signbit(x)))[0]
        return float(np.std(np.diff(z)))
    assert sd(vb) > 1.5 * sd(st)


def test_opl_tables_pitch_and_waveforms():
    assert R.OPL_LOGSIN[0] == 2137 and R.OPL_LOGSIN[255] == 0
    assert R.OPL_EXP[0] == 1018 and R.OPL_EXP[255] == 0
    assert np.all(np.diff(R.OPL_LOGSIN) <= 0) and np.all(np.diff(R.OPL_EXP) <= 0)
    fn, b = R.opl_note(440.0)
    assert (fn, b) == (580, 4) or abs(R.opl_hz(fn, b) - 440.0) < 0.5
    assert abs(R.OPL_RATE - 49716.0) < 1.0
    t0 = R.opl_wave_table(0)
    assert t0.max() == 4084 and t0.min() == -4084 and np.array_equal(t0[:512], -t0[512:])
    assert abs(t0[256] - 4084) < 4
    t1 = R.opl_wave_table(1)
    assert np.all(t1[512:] == 0) and np.array_equal(t1[:512], t0[:512])
    t2 = R.opl_wave_table(2)
    assert np.array_equal(t2[512:], t2[:512]) and t2.min() >= 0
    t3 = R.opl_wave_table(3)
    assert np.all(t3[256:512] == 0) and np.array_equal(t3[:256], t0[:256])
    t6 = R.opl_wave_table(6)
    assert set(np.unique(t6)) == {-4084, 4084}
    t4 = R.opl_wave_table(4)
    assert np.all(t4[512:] == 0) and abs(int(t4[128]) - 4084) < 4
    t7 = R.opl_wave_table(7)
    assert t7[0] == 4084 and t7[300] < t7[100] and t7[600] <= 0
    # a pure sine: tl 63 modulator silent, carrier alone: the pitch is fnum * 49716 / 2**(20 - block)
    for hz in (220.0, 440.0, 880.0):
        y = _one(R.OPL_PATCH, "ch1", hz, fn=R.render_opl, patch="sine", dur=1.0) if False else \
            R.render_opl({"ch1": [(0.0, 1.0, hz, 1.0, dict(R.OPL_PATCH, patch="sine"))]}, 1.0, SR)
        fnum, blk = R.opl_note(hz)
        assert abs(_peak_hz(y[3000:]) - R.opl_hz(fnum, blk)) < 0.8
    # distortion of the sine is small (log-sine and exponent tables at 8-bit resolution)
    y = R.render_opl({"ch1": [(0.0, 1.0, 440.0, 1.0, dict(R.OPL_PATCH, patch="sine"))]}, 1.0, SR)[3000:30000]
    sp = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / SR)
    fund = np.sqrt(np.sum(sp[(f > 425) & (f < 455)] ** 2))
    for h in (2, 3, 4, 5):
        assert np.sqrt(np.sum(sp[(f > 440 * h - 15) & (f < 440 * h + 15)] ** 2)) / fund < 0.04


def test_opl_envelope_rates_feedback_and_opl3():
    def t20(dr):
        p = dict(R.OPL_PATCH, patch=dict(ops=[dict(tl=63), dict(ar=15, dr=dr, sl=15, sus=0, rr=dr)], fb=0, cnt=0))
        y = R.render_opl({"ch1": [(0.0, 6.0, 440.0, 1.0, p)]}, 6.0, SR)
        w = 110
        pk = np.abs(y[:len(y) // w * w]).reshape(-1, w).max(axis=1)
        return float(np.argmax(pk < 0.1 * pk[:3].max())) * w / SR
    # one more step on the 4-bit rate is one more octave on the envelope speed: the time to fall 20 dB halves
    for dr in (5, 6, 7):
        r = t20(dr) / t20(dr + 1)
        assert 1.6 < r < 2.5, (dr, r)
    # feedback adds harmonics: fb 7 sine carrier-only chain: the third harmonic grows
    def h3(fb):
        p = dict(R.OPL_PATCH, patch=dict(ops=[dict(tl=0), dict(tl=0)], fb=fb, cnt=1))
        y = R.render_opl({"ch1": [(0.0, 1.0, 220.0, 1.0, p)]}, 1.0, SR)[4000:44100]
        sp = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        f = np.fft.rfftfreq(len(y), 1 / SR)
        return float(sp[(f > 640) & (f < 680)].max() / sp[(f > 210) & (f < 230)].max())
    assert h3(7) > 5 * h3(0) + 0.01
    # opl2 refuses a waveform above 3, opl3 plays it; opl3 pans
    with pytest.raises(SP.SpecError):
        R.render_opl({"ch1": [(0.0, 0.2, 220.0, 1.0, dict(R.OPL_PATCH, wave=6))]}, 0.2, SR, chip="opl2")
    y = R.render_opl({"ch1": [(0.0, 0.3, 220.0, 1.0, dict(R.OPL_PATCH, wave=6, pan="L"))]}, 0.3, SR, chip="opl3")
    assert y.shape[1] == 2 and _rms(y[:, 1]) < 1e-9 < _rms(y[:, 0])
    # 4 operators: only the last carrier sounds when the chain's other operators are silent; its pitch follows mult
    p4 = dict(R.OPL_PATCH, patch=dict(ops=[dict(tl=63), dict(tl=63), dict(tl=63), dict(mult=2)], fb=0, cnt=(0, 0)))
    y = R.render_opl({"ch1": [(0.0, 1.0, 220.0, 1.0, p4)]}, 1.0, SR, chip="opl3")
    fnum, blk = R.opl_note(220.0)
    assert abs(_peak_hz((y[3000:, 0] + y[3000:, 1])) - 2 * R.opl_hz(fnum, blk)) < 1.5
    with pytest.raises(SP.SpecError):
        R.render_opl({"ch1": [(0.0, 0.2, 220.0, 1.0, p4)]}, 0.2, SR, chip="opl2")
    # tremolo: the level moves by about 4.8 dB at 3.7 Hz
    pt = dict(R.OPL_PATCH, patch="sine", trem=True)
    y = R.render_opl({"ch1": [(0.0, 2.0, 440.0, 1.0, pt)]}, 2.0, SR)
    env = np.array([_rms(y[i:i + 441]) for i in range(2000, 80000, 441)])
    assert 3.0 < 20 * np.log10(env.max() / env.min()) < 6.0
    # rhythm: noise-like and finite
    r = R.render_opl({"hh": [(0.0, 0.2, 440.0, 1.0, dict(R.OPL_PATCH))], "sd": [(0.3, 0.5, 220.0, 1.0, dict(R.OPL_PATCH))],
                      "bd": [(0.6, 0.8, 110.0, 1.0, dict(R.OPL_PATCH))]}, 1.0, SR, chip="opl2", rhythm=True)
    assert np.isfinite(r).all() and _rms(r[:int(0.2 * SR)]) > 1e-3 and _rms(r[int(0.6 * SR):int(0.8 * SR)]) > 1e-3


def test_registered_instruments_and_layers_render_clean():
    from genny.instruments import REGISTRY
    for name, kw in (("vrc6", {}), ("vrc6", dict(wave="saw")), ("n163", dict(channels=4)), ("fds", dict(mod_gain=12)),
                     ("paula", dict(led=True)), ("opl2", dict(patch="bell")), ("opl3", dict(patch="wow4", wave=1))):
        y = REGISTRY[name]["fn"](220.0, 0.3, SR, **kw)
        assert np.isfinite(y).all() and _rms(y) > 1e-3 and abs(y[0]) < 0.05 and abs(y[-1]) < 0.05, name
    for system, ch in (("vrc6", "p1"), ("n163", "ch1"), ("fds", "fds"), ("paula", "ch1"), ("opl2", "ch1"), ("opl3", "ch1")):
        y = CT.chip_layer({"system": system, "channels": {ch: [{"steps": "C4:0.2 E4:0.2"}]}}, SR, 1.0)
        assert np.isfinite(y).all() and _rms(y) > 1e-3, system
