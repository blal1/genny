"""F3 checks (DX7 topology and constants, SysEx, ProTracker effects). Run as a script: `python tests/test_f3.py` (no pytest needed)."""
import hashlib
import os
import sys

import numpy as np

from genny import chiptune_r as R
from genny import synth_s as S
from genny.spec import SpecError

SR = 44100
HERE = os.path.dirname(os.path.abspath(__file__))
ROM1A = os.path.join(HERE, "..", "..", "research", "dx7_syx", "rom1a.syx")

# --------------------------------------------------------------------------- DX7: published algorithm flag table (6 bytes per algorithm,
# first byte = operator 6 ... last = operator 1; bits: 1/2 output bus, 4 add to bus, 16/32 input bus, 64 feedback in, 128 feedback out)
FLAGS = (
    (0xc1, 0x11, 0x11, 0x14, 0x01, 0x14), (0x01, 0x11, 0x11, 0x14, 0xc1, 0x14), (0xc1, 0x11, 0x14, 0x01, 0x11, 0x14),
    (0x41, 0x11, 0x94, 0x01, 0x11, 0x14), (0xc1, 0x14, 0x01, 0x14, 0x01, 0x14), (0x41, 0x94, 0x01, 0x14, 0x01, 0x14),
    (0xc1, 0x11, 0x05, 0x14, 0x01, 0x14), (0x01, 0x11, 0xc5, 0x14, 0x01, 0x14), (0x01, 0x11, 0x05, 0x14, 0xc1, 0x14),
    (0x01, 0x05, 0x14, 0xc1, 0x11, 0x14), (0xc1, 0x05, 0x14, 0x01, 0x11, 0x14), (0x01, 0x05, 0x05, 0x14, 0xc1, 0x14),
    (0xc1, 0x05, 0x05, 0x14, 0x01, 0x14), (0xc1, 0x05, 0x11, 0x14, 0x01, 0x14), (0x01, 0x05, 0x11, 0x14, 0xc1, 0x14),
    (0xc1, 0x11, 0x02, 0x25, 0x05, 0x14), (0x01, 0x11, 0x02, 0x25, 0xc5, 0x14), (0x01, 0x11, 0x11, 0xc5, 0x05, 0x14),
    (0xc1, 0x14, 0x14, 0x01, 0x11, 0x14), (0x01, 0x05, 0x14, 0xc1, 0x14, 0x14), (0x01, 0x14, 0x14, 0xc1, 0x14, 0x14),
    (0xc1, 0x14, 0x14, 0x14, 0x01, 0x14), (0xc1, 0x14, 0x14, 0x01, 0x14, 0x04), (0xc1, 0x14, 0x14, 0x14, 0x04, 0x04),
    (0xc1, 0x14, 0x14, 0x04, 0x04, 0x04), (0xc1, 0x05, 0x14, 0x01, 0x14, 0x04), (0x01, 0x05, 0x14, 0xc1, 0x14, 0x04),
    (0x04, 0xc1, 0x11, 0x14, 0x01, 0x14), (0xc1, 0x14, 0x01, 0x14, 0x04, 0x04), (0x04, 0xc1, 0x11, 0x14, 0x04, 0x04),
    (0xc1, 0x14, 0x04, 0x04, 0x04, 0x04), (0xc4, 0x04, 0x04, 0x04, 0x04, 0x04),
)


def _decode(flags):
    """Carriers, {receiver: set of modulators}, feedback receiver, feedback source (operators 1..6) by running the two buses."""
    bus = {1: set(), 2: set()}
    car, mods, fbr, fbs = set(), {}, None, None
    for i, f in enumerate(flags):
        op = 6 - i
        ib, ob = (f >> 4) & 3, f & 3
        if ib:
            mods[op] = set(bus[ib])
        if f & 0x40:
            fbr = op
        if f & 0x80:
            fbs = op
        if ob == 0:
            car.add(op)
        else:
            bus[ob] = (bus[ob] | {op}) if f & 4 else {op}
    return car, mods, fbr, fbs


def test_dx7_topology_of_the_32_algorithms():
    assert len(FLAGS) == 32
    for n, fl in enumerate(FLAGS, 1):
        car, mods, fbr, fbs = _decode(fl)
        c, m, fr, fs = S.dx7_algorithm(n)
        assert {i + 1 for i in range(6) if c[i]} == car, n
        assert {i + 1: {j + 1 for j in range(6) if m[i, j]} for i in range(6) if m[i].any()} == mods, n
        assert (fr + 1, fs + 1) == (fbr, fbs), n


def test_dx7_presets_render_unchanged():
    got = {k: hashlib.md5(S.render_dx7(v, 261.63, 0.6, 22050).tobytes()).hexdigest() for k, v in S.DX7_PRESETS.items()}
    for line in open(os.path.join(HERE, "f3_preset_md5.txt")):
        k, h = line.split()
        assert got[k] == h, k
    assert len(got) == 4


def _peak(y, sr=SR):
    sp = np.abs(np.fft.rfft(y * np.hanning(len(y)), 1 << 18))
    k = int(np.argmax(sp))
    a, b, c = np.log(sp[k - 1] + 1e-12), np.log(sp[k] + 1e-12), np.log(sp[k + 1] + 1e-12)
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) * sr / (1 << 18)


def _solo(**kw):
    ops = [S._op(l=(99, 99, 99, 0), ol=99, **kw)] + [S._OFF] * 5
    return ops


def test_dx7_detune_step():
    assert abs(S.DET_CENTS - 12606 * 1200 / 2 ** 24) < 1e-12 and abs(S.DET_CENTS - 0.9017) < 1e-3
    lo = S.render_dx7(S.dx7_voice("A", _solo(det=7), alg=32), 441.0, 1.0, SR)[SR // 4:SR // 4 + 32768]
    hi = S.render_dx7(S.dx7_voice("B", _solo(det=14), alg=32), 441.0, 1.0, SR)[SR // 4:SR // 4 + 32768]
    assert abs(_peak(hi) / _peak(lo) - 2 ** (7 * 12606 / 2 ** 24)) < 2e-4


def test_dx7_lfo_speed_table():
    def hz(speed, sec):
        w = S._lfo_wave(4, int(sec * SR), SR, speed)
        s = np.signbit(w)
        return np.count_nonzero(s[1:] != s[:-1]) / 2.0 / sec
    for speed, want in ((35, 25190424 * 990 / 2 ** 32), (99, 25190424 * 4080 / 2 ** 32), (60, 25190424 * 1694 / 2 ** 32)):
        assert abs(hz(speed, 6.0) - want) < 0.25, speed
    assert abs(1.0 / (25190424 * 11 / 2 ** 32) - 15.5) < 0.05                         # speed 0: one cycle in 15.5 s
    w = S._lfo_wave(4, 16 * SR, SR, 0)
    assert w[0] == 0 and w[int(3.9 * SR)] > 0.99 and w[int(11.6 * SR)] < -0.99


def test_dx7_lfo_delay():
    assert np.all(S._lfo_delay(1000, SR, 0) == 1.0)
    f = S._lfo_delay(6 * SR, SR, 50)                       # silent 0.31 s, then a ramp of 0.33 s
    assert f[int(0.30 * SR)] == 0.0 and abs(f[int(0.31 * SR) + int(0.1666 * SR)] - 0.5) < 0.03 and f[int(0.7 * SR)] == 1.0
    f = S._lfo_delay(8 * SR, SR, 98)                   # a = 1: silent 2.5 s, ramp 0.67 s
    assert f[int(2.4 * SR)] == 0.0 and f[int(3.3 * SR)] == 1.0


def test_dx7_pms_depth_and_pitch_eg():
    v = S.dx7_voice("V", _solo(), alg=32, lfo_speed=0, lfo_wave=3, pmd=99, pms=7, lfo_delay=0)
    y = S.render_dx7(v, 220.0, 0.8, SR)[2000:2000 + 22050]
    assert abs(_peak(y) / 220.0 - 2 ** (255 * 255 / 65536)) < 2e-3
    v = S.dx7_voice("V", _solo(), alg=32, lfo_speed=0, lfo_wave=3, pmd=50, pms=3, lfo_delay=0)
    y = S.render_dx7(v, 220.0, 0.8, SR)[2000:2000 + 22050]
    assert abs(_peak(y) / 220.0 - 2 ** (((50 * 165 >> 6) * 33) / 65536)) < 1e-3
    assert S.PITCH_EG_LEVEL[0] == -128 and S.PITCH_EG_LEVEL[50] == 0 and S.PITCH_EG_LEVEL[99] == 127 and len(S.PITCH_EG_LEVEL) == 100
    assert S.PITCH_EG_RATE[0] == 1 and S.PITCH_EG_RATE[99] == 255 and len(S.PITCH_EG_RATE) == 100
    n = SR
    y = S._pitch_eg(n, SR, (99, 99, 99, 99), (99, 99, 99, 50), n)
    assert abs(y[0]) < 1e-9 and abs(y[int(0.1 * SR)] - 0.1 * 255 / 21.3) < 2e-3 and abs(y[int(0.9 * SR)] - 127 / 32) < 1e-9


# --------------------------------------------------------------------------- DX7 SysEx
def _voice_ext(k):
    rng = np.random.default_rng(k)
    ops = [S._op(r=tuple(int(x) for x in rng.integers(0, 100, 4)), l=tuple(int(x) for x in rng.integers(0, 100, 4)), ol=int(rng.integers(0, 100)),
                 coarse=int(rng.integers(0, 32)), fine=int(rng.integers(0, 100)), det=int(rng.integers(0, 15)), mode=int(rng.integers(0, 2)),
                 bp=int(rng.integers(0, 100)), ld=int(rng.integers(0, 100)), rd=int(rng.integers(0, 100)), lc=int(rng.integers(0, 4)),
                 rc=int(rng.integers(0, 4)), rs=int(rng.integers(0, 8)), ams=int(rng.integers(0, 4)), kvs=int(rng.integers(0, 8))) for _ in range(6)]
    return S.dx7_voice("VOICE %02d" % k, ops, alg=int(rng.integers(1, 33)), fb=int(rng.integers(0, 8)), oks=int(rng.integers(0, 2)),
                       pr=tuple(int(x) for x in rng.integers(0, 100, 4)), pl=tuple(int(x) for x in rng.integers(0, 100, 4)),
                       lfo_speed=int(rng.integers(0, 100)), lfo_delay=int(rng.integers(0, 100)), pmd=int(rng.integers(0, 100)),
                       amd=int(rng.integers(0, 100)), lfo_sync=int(rng.integers(0, 2)), lfo_wave=int(rng.integers(0, 6)),
                       pms=int(rng.integers(0, 8)), transpose=int(rng.integers(0, 49)))


def test_dx7_sysex_built_dump_roundtrip_and_checksum():
    voices = [_voice_ext(k) for k in range(32)]
    raw = S.dx7_make_sysex(voices)
    assert len(raw) == 4104 and raw[:6] == bytes([0xF0, 0x43, 0, 9, 0x20, 0]) and raw[-1] == 0xF7
    assert (sum(raw[6:-2]) + raw[-2]) & 0x7F == 0
    assert S.dx7_parse_sysex(raw) == voices
    one = S.dx7_make_single(voices[5])
    assert len(one) == 163 and S.dx7_parse_sysex(one) == [voices[5]]
    bad = bytearray(raw)
    bad[1000] ^= 1
    try:
        S.dx7_parse_sysex(bytes(bad))
        raise AssertionError("checksum not checked")
    except SpecError:
        pass


def test_dx7_sysex_public_factory_dump():
    if not os.path.isfile(ROM1A):
        print("  (rom1a.syx absent: public dump check skipped)")
        return
    raw = open(ROM1A, "rb").read()
    vs = S.dx7_parse_sysex(raw)
    names = [v["name"] for v in vs]
    assert names[:4] == ["BRASS   1", "BRASS   2", "BRASS   3", "STRINGS 1"] and names[25] == "TUB BELLS" and names[31] == "TAKE OFF"
    assert vs[0]["alg"] == 22 and vs[0]["fb"] == 7 and vs[14]["name"] == "BASS    1"
    assert all(1 <= v["alg"] <= 32 and all(0 <= o["ol"] <= 99 and 0 <= o["coarse"] <= 31 for o in v["ops"]) for v in vs)
    assert S.dx7_make_sysex(vs) == raw
    for v in vs[:4]:
        y = S.render_dx7(v, 220.0, 0.3, 22050)
        assert np.isfinite(y).all() and np.abs(y).max() > 0.01


# --------------------------------------------------------------------------- MOD
def _make_mod(rows, sample=None, loop=True, vol=64, ft=0, nch=4):
    smp = np.round(127 * np.sin(2 * np.pi * np.arange(8) / 8)).astype(np.int8) if sample is None else np.asarray(sample, np.int8)
    head = bytearray(20) + bytearray(31 * 30)
    h = bytearray(30)
    h[22:24] = (len(smp) // 2).to_bytes(2, "big")
    h[24], h[25] = ft, vol
    h[26:28] = (0).to_bytes(2, "big")
    h[28:30] = ((len(smp) // 2) if loop else 1).to_bytes(2, "big")
    head[20:50] = h
    blob = bytearray(head) + bytes([1, 0]) + bytes(128) + b"M.K."
    pat = bytearray(64 * nch * 4)
    for r, row in enumerate(rows):
        for c, (s, per, fx, pa) in enumerate(row):
            o = (r * nch + c) * 4
            pat[o:o + 4] = bytes([(s & 0xF0) | (per >> 8), per & 255, ((s & 15) << 4) | fx, pa])
    return bytes(blob + pat + smp.tobytes())


E = (0, 0, 0, 0)
END = [E, E, E, (0, 0, 13, 0)]
CLK = R.PAULA_CLOCK["pal"]
TICK = SR // 50                                                    # 0.02 s at speed 6, tempo 125


def _song(ch0, **kw):
    rows = [[c] + [E, E, E] for c in ch0] + [END]
    return _make_mod(rows, **kw)


def _tick(y, k):
    return y[k * TICK:(k + 1) * TICK]


def _tick_period(y, k, bytes_per_cycle=8):
    f = _peak(_tick(y, k)[40:])
    return CLK / (f * bytes_per_cycle)


def _tick_rms(y, k):
    return float(np.sqrt(np.mean(_tick(y, k) ** 2)))


def test_mod_period_tables():
    t = R.PT_PERIODS
    assert len(t) == 16 and all(len(r) == 36 for r in t)
    assert t[0][0] == 856 and t[0][12] == 428 and t[0][35] == 113 and t[1][12] == 425 and t[8][12] == 453
    assert t[8][0] == 907 and t[15][0] == 862 and t[1][0] == 850
    for r in t:
        assert all(a > b for a, b in zip(r, r[1:]))
    for f in range(-8, 8):
        for i in (0, 12, 24):
            assert abs(t[f & 15][i] - t[0][i] * 2.0 ** (-f / 96)) < 1.6


def test_mod_finetune_uses_table():
    for ft, want in ((1, 425), (8, 453), (15, 431)):
        y = R.render_mod(_song([(1, 428, 0, 0)] * 3, ft=ft), SR, filter="none")[:, 0]
        assert abs(_tick_period(y, 2) - want) < 1.5, ft


def test_mod_hw_limit():
    rows = [(1, 113, 0, 0)] * 3
    a = R.render_mod(_song(rows), SR, filter="none")[:, 0]
    b = R.render_mod(_song(rows), SR, filter="none", hw_limit=True)[:, 0]
    assert abs(_tick_period(a, 2) - 113) < 1.0 and abs(_tick_period(b, 2) - 123) < 1.0
    sl = [(1, 428, 0, 0), (0, 0, 1, 0xFF)] + [(0, 0, 1, 0xFF)] * 6
    a = R.render_mod(_song(sl), SR, filter="none")[:, 0]
    b = R.render_mod(_song(sl), SR, filter="none", hw_limit=True)[:, 0]
    k = 7 * 6 + 2
    assert abs(_tick_period(a, k) - 113) < 1.0 and abs(_tick_period(b, k) - 123) < 1.0


def _vib_dev(y, rows, first_k):
    return [_tick_period(y, first_k + i) - 428 for i in range(rows)]


def test_mod_vibrato_waveforms():
    for kind in (0, 1, 2):
        cells = [(1, 428, 14, 0x40 | kind), (0, 0, 4, 0x4F)] + [(0, 0, 4, 0x00)] * 12
        # E4x in row 0 comes with the note; vibrato from row 1 on
        y = R.render_mod(_song(cells), SR, filter="none")[:, 0]
        devs, exp = [], []
        pos = 0
        for row in range(1, 14):
            for k in range(1, 6):
                devs.append(_tick_period(y, row * 6 + k) - 428)
                d = (R._pt_wave(kind, pos) * 15) >> 7
                exp.append(-d if pos & 128 else d)
                pos = (pos + 16) & 255
        err = np.abs(np.array(devs) - np.array(exp))
        assert err.max() < 3.5, (kind, err.max())
        if kind == 2:
            assert set(np.round(np.abs(exp)).astype(int)) == {29}
        if kind == 0:
            assert len(set(exp)) > 6


def test_mod_tremolo_waveforms():
    for kind in (0, 2):
        cells = [(1, 428, 12, 32), (0, 0, 14, 0x70 | kind)] + [(0, 0, 7, 0x44)] * 14
        cells[2] = (0, 0, 7, 0x44)
        y = R.render_mod(_song(cells), SR, filter="none")[:, 0]
        base = _tick_rms(y, 1)
        pos = 0
        for row in range(2, 15):
            for k in range(0, 6):
                if k == 0:
                    assert abs(_tick_rms(y, row * 6) / base - 1.0) < 0.02
                    continue
                d = (R._pt_wave(kind, pos) * 4) >> 6
                v = min(64, max(0, 32 + (-d if pos & 128 else d)))
                assert abs(_tick_rms(y, row * 6 + k) / base - v / 32.0) < 0.03, (kind, row, k)
                pos = (pos + 16) & 255
    assert max(_tick_rms(y, k) for k in range(12, 60)) / min(_tick_rms(y, k) for k in range(12, 60) if k % 6) > 2


def test_mod_retrigger_E9():
    smp = np.array([(-1) ** i * 100 for i in range(160)], np.int8)
    for x, hits in ((2, (0, 2, 4)), (3, (0, 3)), (0, (0,))):
        y = R.render_mod(_song([(1, 428, 14, 0x90 | x)], sample=smp, loop=False), SR, filter="none")[:, 0]
        for k in range(6):
            r = _tick_rms(y, k)
            assert (r > 0.002) == (k in hits), (x, k, r)


def test_mod_pattern_loop_E6_and_delay_EE():
    cells = [(1, 428, 0, 0), (0, 0, 14, 0x60), E, (0, 0, 14, 0x62), E]
    y = R.render_mod(_song(cells), SR, filter="none")
    assert abs(len(y) / SR - (4 + 3 + 3 + 1 + 1) * 0.12) < 0.004                 # rows 0-3, 1-3, 1-3, 4 and the end row
    cells = [(1, 428, 14, 0xE2), E]
    y = R.render_mod(_song(cells), SR, filter="none")
    assert abs(len(y) / SR - (3 + 1 + 1) * 0.12) < 0.004
    cells = [(1, 428, 0, 0), (0, 0, 14, 0x60), E, (0, 0, 14, 0x61), E]
    y = R.render_mod(_song(cells), SR, filter="none")
    assert abs(len(y) / SR - (4 + 3 + 1 + 1) * 0.12) < 0.004


def test_mod_pan_8xx_and_E8():
    def lr(cell, ch):
        rows = [[E] * 4, [E] * 4]
        rows[0][ch] = (1, 428, 8, cell[0]) if cell[1] is None else (1, 428, 14, cell[1])
        rows[1][ch] = (0, 0, 0, 0)
        y = R.render_mod(_make_mod(rows + [END]), SR, filter="none")
        return float(np.sqrt(np.mean(y[2000:, 0] ** 2))), float(np.sqrt(np.mean(y[2000:, 1] ** 2)))
    l, r = lr((0x00, None), 1)
    assert l > 0.01 and r < 1e-9
    l, r = lr((0xFF, None), 0)
    assert r > 0.01 and l < 1e-9
    l, r = lr((0x80, None), 0)
    assert abs(l / r - 1.0) < 0.02
    l, r = lr((0, 0x8F), 0)
    assert r > 0.01 and l < 1e-9
    l, r = lr((0, 0x80), 1)
    assert l > 0.01 and r < 1e-9
    l, r = lr((0, 0x87), 0)
    assert abs(l / r - 1.0) < 0.1


def test_mod_glissando_E3():
    tab = np.array(R.PT_PERIODS[0])
    def dists(gl):
        cells = [(1, 428, 14, 0x30 | gl), (1, 285, 3, 6)] + [(0, 0, 3, 0)] * 5
        y = R.render_mod(_song(cells), SR, filter="none")[:, 0]
        return [np.min(np.abs(tab - _tick_period(y, 6 + k))) for k in range(1, 30) if k % 6]
    assert max(dists(1)) < 2.5
    assert max(dists(0)) > 5


def test_mod_existing_effects():
    y = R.render_mod(_song([(1, 428, 14, 0x14)] * 3), SR, filter="none")[:, 0]
    assert abs(_tick_period(y, 2) - 424) < 1.5                                        # E1 fine slide
    y = R.render_mod(_song([(1, 428, 14, 0x24)] * 3), SR, filter="none")[:, 0]
    assert abs(_tick_period(y, 2) - 432) < 1.5                                        # E2
    a = _tick_rms(R.render_mod(_song([(1, 428, 12, 32)] * 2), SR, filter="none")[:, 0], 2)
    b = _tick_rms(R.render_mod(_song([(1, 428, 12, 32), (0, 0, 14, 0xA8)]), SR, filter="none")[:, 0], 8)
    c = _tick_rms(R.render_mod(_song([(1, 428, 12, 32), (0, 0, 14, 0xB8)]), SR, filter="none")[:, 0], 8)
    assert abs(b / a - 40 / 32) < 0.03 and abs(c / a - 24 / 32) < 0.03                 # EA / EB
    y = R.render_mod(_song([(1, 428, 14, 0xC3)]), SR, filter="none")[:, 0]
    assert _tick_rms(y, 2) > 0.01 and _tick_rms(y, 4) < 1e-3 * _tick_rms(y, 2)      # EC
    y = R.render_mod(_song([(1, 428, 14, 0xD3)]), SR, filter="none")[:, 0]
    assert _tick_rms(y, 2) < 1e-9 and _tick_rms(y, 3) > 0.01                          # ED
    y = R.render_mod(_song([(1, 428, 0, 0), (0, 0, 11, 0)]), SR, filter="none")      # B jump onto a played row ends the render
    assert len(y) / SR < 0.3


if __name__ == "__main__":
    bad = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:
                bad += 1
                import traceback
                traceback.print_exc()
                print("FAIL", name, repr(e))
    sys.exit(1 if bad else 0)
