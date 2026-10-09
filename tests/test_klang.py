"""genny.klang: the C++ ports against the formulas of their headers, the Python ports, the sampler
and articulation layers, the new instruments and the QA harness.

Run:  uv run --with pytest pytest tests/test_klang.py -q      (or: uv run python tests/test_klang.py)
"""
import math
import tempfile
from pathlib import Path

import numpy as np

import genny.klang as K
from genny import instruments as I
from genny import sfx as SX
from genny.analysis import brightness, loudness, sharpness
from genny.core import write_wav
from genny.notes import midi_to_freq, note_to_midi

SR = 44100
SFX = ("toy_boat", "klang_car", "harrier", "bicycle", "klang_rain", "ui_tick", "ui_chime", "damage",
       "sniff", "strain", "weight_shift", "handle", "tide", "distant_bell")
INSTRUMENTS = ("viola", "bassoon", "cor_anglais", "bass_clarinet", "piccolo", "recorder", "hand_chime")
KINDS = {"klang_car": ("basic", "mini"), "ui_tick": K.UI_KINDS, "ui_chime": ("discovery", "achievement", "alert", "spell"),
         "damage": ("physical", "soul", "mental"), "handle": K.HANDLE_KINDS, "tide": ("rising", "falling")}


# ------------------------------------------------------------------ measuring helpers
def peaks(y, lo, hi, k=4, sr=SR):
    """The k strongest separate spectral peaks in [lo, hi] Hz."""
    y = y[int(0.3 * sr):]
    s = np.abs(np.fft.rfft(y * np.hanning(len(y)), 1 << 19))
    f = np.fft.rfftfreq(1 << 19, 1 / sr)
    m = (f > lo) & (f < hi)
    fm, sm, out = f[m], s[m], []
    for i in np.argsort(sm)[::-1]:
        if all(abs(fm[i] - p) > max(1.0, 0.03 * fm[i]) for p in out):
            out.append(float(fm[i]))
        if len(out) >= k:
            break
    return out


def env_peaks(y, lo, hi, k=3):
    """Strongest frequencies of the amplitude envelope (|y| averaged to 2 kHz)."""
    d = SR // 2000
    e = np.abs(y)[:len(y) // d * d].reshape(-1, d).mean(1)
    return peaks(e - e.mean(), lo, hi, k, 2000)


def acf_f0(y, lo, hi):
    y = y[int(0.3 * SR):]
    y = y - y.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(y, 2 * len(y))) ** 2)
    a, b = int(SR / hi), int(SR / lo)
    return SR / (a + int(np.argmax(ac[a:b])))


def centroid(y):
    s = np.abs(np.fft.rfft(y)) ** 2
    return float(np.sum(np.fft.rfftfreq(len(y), 1 / SR) * s) / np.sum(s))


def near(a, b, tol):
    return abs(a - b) <= tol * abs(b)


def cents_off(y, f0):
    """As tests/test_instruments.py: pitch error from the autocorrelation peak near the period."""
    y = y[int(0.05 * SR):int(0.05 * SR) + 16384]
    y = y - y.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(y * np.hanning(len(y)), 4 * len(y))) ** 2)
    p = SR / f0 * 4
    lo = int(p * 0.94)
    k = lo + int(np.argmax(ac[lo:int(p * 1.06) + 2]))
    k = k + 0.5 * (ac[k - 1] - ac[k + 1]) / (ac[k - 1] - 2 * ac[k] + ac[k + 1] + 1e-20)
    return 1200 * np.log2((SR * 4 / k) / f0)


# ------------------------------------------------------------------ klang primitives, by hand
def test_klang_primitives():
    # Biquad BPF 590 Hz Q 4 at fs 44100 (klang.h init_peak): b0 = a / (1 + a), a = sin(w) / (2 Q)
    w = 2 * math.pi * 590 / 44100
    a = math.sin(w) / 8
    b0, b1, b2, a1, a2 = K._bq(2, 590.0, 4.0)
    assert near(b0, a / (1 + a), 1e-12) and b1 == 0 and near(b2, -a / (1 + a), 1e-12)
    assert near(a1, -2 * math.cos(w) / (1 + a), 1e-12) and near(a2, (1 - a) / (1 + a), 1e-12)
    # Q below 0.5 is clamped for alpha (klang.h l.6224)
    assert K._bq(2, 400.0, 0.1) == K._bq(2, 400.0, 0.5)
    # Delay::tap(float): d samples behind the last written sample, linear interpolation
    buf = np.arange(9.0)                       # SIZE 8, buf[8] is never written by klang
    assert K._tap(buf, 5, 8, 0.0) == 4.0 and K._tap(buf, 5, 8, 1.5) == 2.5
    assert K._tap(buf, 0, 8, 0.0) == 7.0       # wraps to SIZE - 1
    # Fast::Pulse: first sample is the low level; mean over a cycle = 2 duty - 1
    f = 6.0 / 44100
    assert K._pulse_avg(0.0, f, 0.1) == -1.0
    assert near(np.mean([K._pulse_avg(k * f % 1.0, f, 0.1) for k in range(7350)]), -0.8, 1e-3)
    # Turbine gain curve (Harrier.h l.232-239)
    assert K.turbine_gain(0.1) == 0.8 and K.turbine_gain(0.2) == 1.0
    assert near(K.turbine_gain(0.5), 0.5, 1e-12) and near(K.turbine_gain(0.3), 0.9, 1e-12) and near(K.turbine_gain(1.0), 0.5, 1e-12)


def test_car_first_sample_by_hand():
    """FourStrokeEngine, sample 0 at 3000 rpm: the delay lines are still empty and the phasor is 0, so
    cylinder d sees cos(-2 pi (0.75 - 0.25 d)) (22 - 15 speed): 0, -s, 0, +s -> o = 2 + 2 / (s^2 + 1)."""
    sp = 3000 / 7000
    s = 22 - 15 * sp
    o = 2 + 2 / (s * s + 1)
    b0 = K._bq(1, 100.0, 1 / math.sqrt(2))[0]                    # hpf(100), then DCF passes y[0]
    want = 0.25 * math.tanh(3 * b0 * o * sp * min(sp, 0.25)) / math.tanh(3)
    got = K._car_k(np.full(8, sp), 12345)[0]
    assert near(got, want, 5e-3), (got, want)
    assert K._toy_boat_k(4, 9.0, False, 1)[0] == 0.0             # sin(0) = 0 -> valve closed


def test_mini_envelopes():
    rng = np.random.default_rng(3)
    delay = np.random.default_rng(3).uniform(0.25, 0.75)
    n = int(9 * K.KFS)
    power, rev, done = K._mini_env(n, rng, off_at=7.0)
    at = lambda t: int(round(t * K.KFS))                                       # noqa: E731
    assert power[0] == 0.5 and near(power[at(delay)], 1.0, 1e-3) and near(power[at(delay + 0.25)], 2.0, 1e-3)
    assert near(power[at(delay + 0.5)], 1.0, 1e-3) and near(power[at(6.9)], 1.0, 1e-9)
    assert rev[0] == 0 and near(rev[at(delay)], 1.0, 1e-3) and 2 <= rev[at(delay + 0.125)] <= 5 and rev[at(delay + 0.3)] == 0
    assert not done[at(delay + 4.9)] and done[at(delay + 5.01)]
    assert near(power[at(8.0)], 0.5, 1e-3) and power[-1] < 1e-4                 # starter.release(2)


# ------------------------------------------------------------------ dominant frequencies vs the headers
def test_toy_boat_pulse_rate():
    for rate in (9.0, 14.0):                   # header: osc.set(9); a burst on each edge of the valve
        got = env_peaks(SX.render_sfx("toy_boat", SR, rate=rate, dur=6.0), 3, 80, 1)[0]
        assert near(got, 2 * rate, 0.02), (rate, got)
    a = SX.render_sfx("toy_boat", SR, dur=4.0)
    b = SX.render_sfx("toy_boat", SR, broken=1, dur=4.0)
    ea, eb = (np.abs(x)[:len(x) // 2205 * 2205].reshape(-1, 2205).mean(1) for x in (a, b))
    assert np.std(np.diff(ea[4:-4])) < np.std(np.diff(eb[4:-4]))                # broken = irregular


def test_car_basic_fundamental():
    for rpm in (1500, 3000, 5000):
        f0 = K.car_fundamental(rpm)            # rpm * 16 / 700
        y = SX.render_sfx("klang_car", SR, rpm=rpm, dur=4.0)
        for p in peaks(y, 15, 1500, 3):        # every strong partial is a harmonic of the pulse rate
            assert abs(p / f0 - round(p / f0)) < 0.02 and round(p / f0) >= 1, (rpm, p, f0)
    assert near(acf_f0(SX.render_sfx("klang_car", SR, rpm=3000, dur=4.0), 15, 400), K.car_fundamental(3000), 0.01)


def test_mini_partials():
    for rpm, thr in ((900, 0.0), (2000, 0.3), (4000, 0.3)):
        r = K.mini_rate(rpm, thr)
        got = sorted(peaks(SX.render_sfx("klang_car", SR, kind="mini", rpm=rpm, throttle=thr, dur=6.0), 10, 2000, 2))
        assert near(got[0], 64.6 * r, 0.01) and near(got[1], 86.1 * r, 0.01), (rpm, thr, got, 86.1 * r)
    assert near(K.mini_rate(900), 0.95, 1e-12)                                  # idle: 1 - 0.05


def test_harrier_turbine_and_burn():
    for sp in (0.3, 0.5, 0.9):                 # altitude 0: overdrive x min(altitude, 2) / 2 = 0 -> turbine only
        got = peaks(SX.render_sfx("harrier", SR, speed=sp, altitude=0.0), 200, 20000, 1)[0]
        assert near(got, 5588.0 * sp, 0.005), (sp, got)
    lv = [loudness(SX.render_sfx("harrier", SR, speed=s, altitude=2.0), SR) for s in (0.2, 0.5, 1.0)]
    assert lv[0] < lv[1] < lv[2], lv           # more speed -> more overdrive -> louder
    assert loudness(SX.render_sfx("harrier", SR, speed=0.5, altitude=0.0), SR) < lv[1] - 10    # the burn dominates
    c = [centroid(SX.render_sfx("harrier", SR, speed=s, altitude=0.0)) for s in (0.3, 0.6)]
    assert c[1] > 1.5 * c[0]                   # the whine climbs with the speed
    for alt in (400.0, 20000.0, 100000.0):     # wind at height; the header's filter is unstable up there
        y = SX.render_sfx("harrier", SR, speed=0.9, altitude=alt)
        assert np.all(np.isfinite(y)) and 1e-4 < np.max(np.abs(y)) <= 0.981
    e = SX.render_sfx("harrier", SR, speed=0.8, echo=1.0)
    assert np.all(np.isfinite(e)) and loudness(e, SR) > loudness(SX.render_sfx("harrier", SR, speed=0.8), SR)


def test_harrier_spool_up():
    y = SX.render_sfx("harrier", SR, speed=0.2, speed_end=0.8, altitude=0.0, dur=4.0)
    a, b = centroid(y[:SR]), centroid(y[-SR:])
    assert b > 2 * a, (a, b)


def test_bicycle_rates():
    for ws in (4.0, 6.0, 12.0):                # the wheel pattern repeats once per wheel_speed
        got = env_peaks(SX.render_sfx("bicycle", SR, pedalling=0.0, wheel_speed=ws, dur=8.0), 1, 100, 1)[0]
        assert near(got, ws, 0.02), (ws, got)
    for ps in (1.0, 2.0):                      # chain: |sin| at 22 * (pedal_speed / 2) Hz -> 22 * pedal_speed
        got = env_peaks(SX.render_sfx("bicycle", SR, pedalling=1.0, wheel_speed=0.0, pedal_speed=ps, dur=8.0), 2, 200, 1)[0]
        assert near(got, 22.0 * ps, 0.08), (ps, got)
    lv = [loudness(SX.render_sfx("bicycle", SR, pedalling=p), SR) for p in (0.0, 0.5, 1.0)]
    assert lv[0] < lv[1] < lv[2], lv


def test_monotonic_controls():
    lv = [loudness(SX.render_sfx("klang_car", SR, rpm=r), SR) for r in (1000, 2500, 4500, 6500)]
    assert all(a < b for a, b in zip(lv, lv[1:])), lv
    a = SX.render_sfx("klang_car", SR, kind="mini", rpm=3000, throttle=0.0)
    b = SX.render_sfx("klang_car", SR, kind="mini", rpm=3000, throttle=1.0)     # floored: gas = 1
    assert loudness(b, SR) > loudness(a, SR) + 1.0 and centroid(b) > centroid(a)
    # throttle below 1 never opens the gas (klang's integer max), it only raises the rate
    c = SX.render_sfx("klang_car", SR, kind="mini", rpm=3000, throttle=0.6)
    assert near(sorted(peaks(c, 10, 2000, 2))[1], 86.1 * K.mini_rate(3000, 0.6), 0.01)
    rise = SX.render_sfx("klang_car", SR, rpm=1500, rpm_end=5000, dur=4.0)
    assert loudness(rise[-SR:], SR) > loudness(rise[:SR], SR) + 6


def test_mini_ignition_and_stop():
    y = SX.render_sfx("klang_car", SR, kind="mini", rpm=900, ignition=1, off_at=2.0, dur=4.5, seed=4)
    rms = lambda a, b: float(np.sqrt(np.mean(y[int(a * SR):int(b * SR)] ** 2)))   # noqa: E731
    assert rms(4.1, 4.5) < 1e-5 < rms(1.5, 2.0)                                    # run-down takes 2 s
    assert rms(3.0, 3.5) < rms(1.5, 2.0)


# ------------------------------------------------------------------ rain
def test_rain_event_rate():
    assert K.rain_rate(0.0) == 2.0 and near(K.rain_rate(1.0), 602.0, 1e-12)
    n = [len(K.rain_events(20.0, i, 1)) for i in (0.0, 0.2, 0.5, 1.0)]
    assert all(a < b for a, b in zip(n, n[1:])), n
    for i, c in zip((0.2, 0.5, 1.0), n[1:]):
        assert near(c, K.rain_rate(i) * 20.0, 0.1), (i, c)                          # Poisson: sqrt(N) / N < 5 %
    # heard: count the ticks of a sparse rain without the bed
    y = SX.render_sfx("klang_rain", SR, intensity=0.1, hiss=0.0, dur=10.0, seed=2)
    e = np.abs(y)[:len(y) // 44 * 44].reshape(-1, 44).max(1)                        # 1 ms frames
    on = np.flatnonzero((e[1:] > 0.02) & (e[:-1] <= 0.02))
    heard = 1 + int(np.sum(np.diff(on) > 8)) if len(on) else 0
    assert near(heard, K.rain_rate(0.1) * 10.0, 0.3), heard
    lv = [loudness(SX.render_sfx("klang_rain", SR, intensity=i), SR) for i in (0.1, 0.5, 1.0)]
    assert lv[0] < lv[1] < lv[2], lv


# ------------------------------------------------------------------ Python ports
def test_damage_is_low():
    for kind, top in (("soul", 0.05), ("mental", 0.02), ("physical", 0.6)):
        y = SX.render_sfx("damage", SR, kind=kind)
        s = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(len(y), 1 / SR)
        assert np.sum(s[f > 1000]) / np.sum(s) < top, (kind, np.sum(s[f > 1000]) / np.sum(s))
    # soul: lub 63 Hz x 5 cycles and dub 42 Hz x 3 cycles (lovelyheart onebeat)
    assert len(K._onebeat(63.0, 5, 0.0, 0.65)) == K.P.seconds(5 / 63.0)
    assert 30 < peaks(np.concatenate([np.zeros(int(0.3 * 48000)), K._onebeat(42.0, 3, 0.5, 1.0), np.zeros(48000)]), 10, 200, 1, 48000)[0] < 100
    assert len(SX.render_sfx("damage", SR, kind="physical")) < 0.5 * SR


def test_ui_cues():
    a = SX.render_sfx("ui_tick", SR, kind="activate", seed=3)
    assert np.array_equal(a, SX.render_sfx("ui_tick", SR, kind="activate", seed=3))      # same action, same sound
    assert not np.array_equal(a, SX.render_sfx("ui_tick", SR, kind="activate", seed=4))
    assert centroid(SX.render_sfx("ui_tick", SR, kind="refuse")) < centroid(SX.render_sfx("ui_tick", SR, kind="focus")) < centroid(a)
    for k in K.UI_KINDS:
        assert len(SX.render_sfx("ui_tick", SR, kind=k)) < 0.45 * SR
    # telephone bell pair 650 / 653 Hz (x pitch), English bell x0.75: mode 0 at 259.4 x 0.75
    y = SX.render_sfx("ui_chime", SR, kind="achievement")
    assert any(abs(p - 651.5) < 6 for p in peaks(np.concatenate([np.zeros(int(0.3 * SR)), y]), 300, 1000, 2))
    y = SX.render_sfx("ui_chime", SR, kind="alert")
    assert any(near(p, 259.429 * 0.75, 0.01) for p in peaks(np.concatenate([np.zeros(int(0.3 * SR)), y]), 100, 1000, 4))
    y = SX.render_sfx("ui_chime", SR, kind="spell", pitch=0.5)
    assert near(peaks(np.concatenate([np.zeros(int(0.3 * SR)), y]), 200, 3000, 1)[0], 440.0, 0.02)


def test_world_ports():
    # Reshetnikov's stream gets darker as its dynamics d rise (0.05 -> 0.55): a trickle becomes a deep
    # flow; the backwash runs the other way and ends as a bright trickle over shingle
    def hf(x):
        s = np.abs(np.fft.rfft(x)) ** 2
        return float(np.sum(s[np.fft.rfftfreq(len(x), 1 / SR) > 2500]) / np.sum(s))
    r, f = SX.render_sfx("tide", SR, kind="rising", dur=4.0), SX.render_sfx("tide", SR, kind="falling", dur=4.0)
    q = len(r) // 4
    assert hf(r[:q]) > 3 * hf(r[-q:]) and hf(f[-q:]) > 5 * hf(r[-q:]), (hf(r[:q]), hf(r[-q:]), hf(f[-q:]))
    near_, far = SX.render_sfx("distant_bell", SR, distance=20.0), SX.render_sfx("distant_bell", SR, distance=800.0)
    assert centroid(far) < 0.8 * centroid(near_)                                      # air absorption
    big, small = SX.render_sfx("distant_bell", SR, size=2.0, distance=20.0), SX.render_sfx("distant_bell", SR, size=1.0, distance=20.0)
    assert centroid(big) < centroid(small)
    assert not np.array_equal(SX.render_sfx("handle", SR, kind="armor", seed=0), SX.render_sfx("handle", SR, kind="armor", seed=1))
    g = [np.sqrt(np.mean(SX.render_sfx("weight_shift", SR, ground=x) ** 2)) for x in ("sand", "gravel")]
    assert all(v > 1e-3 for v in g)


# ------------------------------------------------------------------ layers
def _tone(path, f, amp=0.5, sr=22050, stereo=False, dur=0.6):
    t = np.arange(int(dur * sr)) / sr
    x = amp * np.sin(2 * np.pi * f * t) * np.exp(-2.0 * t)
    write_wav(path, np.stack([x, 0.5 * x], axis=1) if stereo else x, sr)


def test_sampler_names():
    assert K.name_midi("Piano_C4_v1") == 60 and K.name_midi("vln_A#3") == 58 and K.name_midi("harp_As3_ff") == 58
    assert K.name_midi("Bb2-soft") == 46 and K.name_midi("kick_01") is None and K.name_midi("x_c5") == 72
    assert K.vel_rank("Piano_C4_v3") == 3 and K.vel_rank("cello_F2_pp") == 3 and K.vel_rank("cello_F2_ff") == 8
    assert K.vel_rank("tom_loud_2") == 8 and K.vel_rank("flute_F4") == 0 and K.vel_rank("x_vl2_C3") == 2


def test_sampler_layer():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _tone(d / "tone_C4_v1.wav", 261.63, 0.25)
        _tone(d / "tone_C5_v1.wav", 523.25, 0.25)
        _tone(d / "tone_C4_v2_rr1.wav", 261.63, 0.5)
        _tone(d / "tone_C4_v2_rr2.wav", 261.63 * 1.5, 0.5)          # take 2 is recognisable: a fifth up
        (d / "notes.txt").write_text("not audio")
        z = K.sampler_zones(str(d))
        assert [r for r, _ in z] == [1, 2] and [m for m, _ in z[0][1]] == [60, 72] and len(z[1][1][0][1]) == 2
        f0 = lambda y, f: peaks(np.concatenate([np.zeros(int(0.3 * SR)), y]), f * 0.7, f * 1.4, 1)[0]      # noqa: E731
        # pitch: D4 comes from the C4 sample two semitones up; B4 from C5 one down; 22050 -> 44100 Hz
        for note, f in (("D4", 293.66), ("B4", 493.88), ("C4", 261.63)):
            y = K.sampler_layer({"dir": str(d), "steps": f"{note}:0.5@0.2"}, SR, 1.0)
            assert y.ndim == 1 and abs(1200 * math.log2(f0(y, f) / f)) < 5, (note, f0(y, f))
        # velocity picks the layer (loud layer only has C4), and the level follows the velocity curve
        soft = K.sampler_layer({"dir": str(d), "steps": "C4:0.5@0.2"}, SR, 1.0)
        loud = K.sampler_layer({"dir": str(d), "steps": "C4:0.5@1.0"}, SR, 1.0)
        assert near(np.max(np.abs(loud)), 0.5, 0.05)
        assert near(np.max(np.abs(soft)), 0.25 * 10 ** (-28 * 0.8 ** 1.15 / 20), 0.08)
        # round robin: the two loud takes alternate, and restart with each render
        y = K.sampler_layer({"dir": str(d), "steps": "C4:0.5 C4:0.5 C4:0.5", "release": 0.02}, SR, 1.0)
        got = [f0(y[int(k * 0.5 * SR):int((k + 1) * 0.5 * SR)], 320.0) for k in range(3)]
        assert near(got[0], 261.63, 0.01) and near(got[1], 392.44, 0.01) and near(got[2], 261.63, 0.01), got
        assert np.array_equal(y, K.sampler_layer({"dir": str(d), "steps": "C4:0.5 C4:0.5 C4:0.5", "release": 0.02}, SR, 1.0))
        # note length: cut with a release that ends on exactly zero; release 0 lets the sample ring
        short = K.sampler_layer({"dir": str(d), "steps": "C4:0.1", "release": 0.05}, SR, 1.0)
        assert abs(len(short) - 0.15 * SR) <= 2 and short[-1] == 0.0 and short[0] == 0.0
        assert len(K.sampler_layer({"dir": str(d), "steps": "C4:0.1", "release": 0}, SR, 1.0)) > 0.55 * SR
        # transpose / tune, max_shift, bpm scaling, other sample rates
        up = K.sampler_layer({"dir": str(d), "steps": "C4:0.5@0.2", "transpose": 2, "tune": 50}, SR, 1.0)
        assert abs(1200 * math.log2(f0(up, 300.0) / 261.63) - 250) < 5
        try:
            K.sampler_layer({"dir": str(d), "steps": "G4:0.5@0.2", "max_shift": 2}, SR, 1.0)
            raise AssertionError("max_shift not enforced")
        except ValueError:
            pass
        assert abs(len(K.sampler_layer({"dir": str(d), "steps": "C4:1 C4:1", "release": 0.05}, SR, 0.5)) - 1.05 * SR) <= 2
        y48 = K.sampler_layer({"dir": str(d), "steps": "D4:0.5@0.2"}, 48000, 1.0)
        s = np.abs(np.fft.rfft(y48 * np.hanning(len(y48)), 1 << 18))
        assert near(np.fft.rfftfreq(1 << 18, 1 / 48000)[int(np.argmax(s))], 293.66, 0.005)
        # through the spec renderer; stereo files give a stereo layer; unpitched files sit on `root`
        from genny.spec import render_spec
        out, _ = render_spec({"layers": [{"type": "sampler", "dir": str(d), "steps": "C4:0.3 E4:0.3"}]}, SR)
        assert np.all(np.isfinite(out)) and np.max(np.abs(out)) > 0.1
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _tone(d / "hit_a.wav", 200.0, stereo=True)
        _tone(d / "hit_b.wav", 300.0, stereo=True)
        y = K.sampler_layer({"dir": str(d), "steps": "C3:0.3 C3:0.3", "root": "C3"}, SR, 1.0)
        assert y.ndim == 2 and y.shape[1] == 2 and np.all(np.isfinite(y))
        for bad in ({"dir": str(d / "nope"), "steps": "C4"}, {"dir": str(d), "pattern": "*.flac", "steps": "C4"}):
            try:
                K.sampler_layer(bad, SR, 1.0)
                raise AssertionError("bad sampler folder accepted")
            except ValueError:
                pass


def test_articulate_layer():
    base = {"inst": "viola", "steps": "G4:0.6 A4:0.6"}
    r = {a: K.articulate_layer(dict(base, artic=a), SR, 1.0) for a in K.ARTICULATIONS}
    assert all(np.all(np.isfinite(y)) and np.max(np.abs(y)) > 1e-3 for y in r.values())
    assert len(r["staccato"]) < len(r["sustain"]) < len(r["legato"]) and len(r["hit"]) < len(r["staccato"])
    seg = lambda y: y[int(0.15 * SR):int(0.55 * SR)]                                   # noqa: E731
    f = lambda y: peaks(np.concatenate([np.zeros(int(0.3 * SR)), seg(y)]), 200, 1200, 1)[0]   # noqa: E731
    assert near(f(r["sustain"]), 392.0, 0.02) and near(f(r["harmonic"]), 784.0, 0.02)
    env = lambda y: np.abs(y)[:len(y) // 441 * 441].reshape(-1, 441).max(1)            # noqa: E731
    assert np.std(np.diff(env(r["tremolo"])[20:50])) > 2 * np.std(np.diff(env(r["sustain"])[20:50]))
    e = env(r["roll"])
    assert e[45:58].max() > 1.15 * e[2:12].max()                                         # crescendo over the note
    assert K.articulate_layer(dict(base, artic="pizz"), SR, 1.0).shape[0] > 0
    try:
        K.articulate_layer(dict(base, artic="sul_tasto"), SR, 1.0)
        raise AssertionError("unknown articulation accepted")
    except ValueError:
        pass


# ------------------------------------------------------------------ instruments (as test_instruments.py)
def test_instruments():
    for name in INSTRUMENTS:
        e = I.REGISTRY[name]
        lo, hi = (note_to_midi(n) for n in e["range"].split("-"))
        for m in (lo - 5, lo, (lo + hi) // 2, hi, hi + 7):
            for dur, vel, sr in ((0.03, 1.0, 44100), (0.1234, 0.3, 44100), (0.3333, 0.7, 22050), (1.0, 1.0, 48000)):
                y = I.render_note(name, midi_to_freq(m), dur, sr, vel)
                assert y.ndim == 1 and np.all(np.isfinite(y)), (name, m, dur)
                assert 1e-4 < np.max(np.abs(y)) < 1.5, (name, m, dur, float(np.max(np.abs(y))))
                assert abs(y[-1]) < 1e-3, (name, "ends on a step")
        for m in (lo, (lo + hi) // 2, hi):
            c = cents_off(I.render_note(name, midi_to_freq(m), 0.6, SR, 0.9), midi_to_freq(m))
            assert abs(c) < 10, (name, m, round(float(c), 1))
        for m in sorted({min((lo + hi) // 2, 96), min(hi, 96)}):
            f = midi_to_freq(m)
            y = I.render_note(name, f, 0.5, SR, 0.9)
            pure = sharpness(I.render_note("sine", f, 0.5, SR, 0.9), SR)
            s, b = sharpness(y, SR), brightness(y, SR)
            assert s < max(pure + 1.0, 1.5) and s < 2.4, (name, m, round(s, 2), round(pure, 2))
            assert b["above_5k"] < 0.03, (name, m, round(b["above_5k"], 3))
        lv = [loudness(I.render_note(name, midi_to_freq(m), 0.5, SR, 0.9), SR, window=0.15)
              for m in (lo + (hi - lo) // 4, (lo + hi) // 2, hi - (hi - lo) // 4)]
        assert abs(sum(lv) / 3 + 12.0) < 6.0, (name, [round(v, 1) for v in lv])
    # hand chime: fundamental and its twelfth (charter.hand_chime modes 1 and 3)
    y = I.render_note("hand_chime", 440.0, 1.0, SR, 0.9)
    p = peaks(np.concatenate([np.zeros(int(0.3 * SR)), y]), 200, 3000, 2)
    assert near(min(p), 440.0, 0.005) and near(max(p), 1320.0, 0.005), p
    # the reeds differ where they should: a bassoon's main formant sits below a cor anglais's
    assert centroid(I.render_note("bassoon", 220.0, 0.5, SR, 0.9)) < centroid(I.render_note("cor_anglais", 220.0, 0.5, SR, 0.9))


# ------------------------------------------------------------------ smoke: every name, two rates
def test_smoke_all_names():
    for sr in (22050, 48000):
        for name in SFX:
            for kind in KINDS.get(name, (None,)):
                p = {} if kind is None else {"kind": kind}
                y = SX.render_sfx(name, sr, **p)
                tag = (name, kind, sr)
                assert y.ndim == 1 and np.all(np.isfinite(y)), tag
                assert 1e-4 < np.max(np.abs(y)) < 1.5, (tag, float(np.max(np.abs(y))))
                assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3, (tag, "starts or ends on a step")
                assert abs(np.mean(y)) < 0.03 * np.max(np.abs(y)), (tag, "dc")
                e = SX.REGISTRY[name]
                assert abs(len(y) / sr - float(e["params"]["dur"][0])) < 0.01 if "dur" in e["params"] else True, tag
    for name in SFX:                           # deterministic, and every declared param has a help text
        assert np.array_equal(SX.render_sfx(name, SR), SX.render_sfx(name, SR)), name
        for k, v in SX.REGISTRY[name]["params"].items():
            assert isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], str) and v[1], (name, k)
    for name, kw in (("klang_car", {"kind": "v8"}), ("ui_tick", {"kind": "x"}), ("damage", {"kind": "x"}),
                     ("handle", {"kind": "x"}), ("tide", {"kind": "x"}), ("weight_shift", {"ground": "lava"})):
        try:
            SX.render_sfx(name, SR, **kw)
            raise AssertionError(f"{name} accepted {kw}")
        except ValueError:
            pass
    # out-of-range controls are clamped, not fatal
    for name, kw in (("klang_car", {"rpm": 99999}), ("klang_car", {"kind": "mini", "rpm": 0, "throttle": 5}),
                     ("harrier", {"speed": 7, "altitude": -3, "gain": 9}), ("bicycle", {"wheel_speed": 500, "pedal_speed": 99}),
                     ("toy_boat", {"rate": 0}), ("klang_rain", {"intensity": 9, "tone": 99})):
        y = SX.render_sfx(name, SR, dur=1.0, **kw)
        assert np.all(np.isfinite(y)) and np.max(np.abs(y)) < 1.5, (name, kw)


def test_speed():
    import time
    for name, kw in (("toy_boat", {}), ("klang_car", {}), ("klang_car", {"kind": "mini"}), ("harrier", {}), ("bicycle", {}), ("klang_rain", {})):
        SX.render_sfx(name, SR, dur=0.2, **kw)                                           # JIT warm-up
        t = time.perf_counter()
        SX.render_sfx(name, SR, dur=1.0, **kw)
        assert time.perf_counter() - t < 2.0, (name, kw, time.perf_counter() - t)


# ------------------------------------------------------------------ QA harness
def test_coverage_harness():
    rows = K.coverage(list(SFX) + list(INSTRUMENTS))
    assert {r["name"] for r in rows} == set(SFX) | set(INSTRUMENTS)
    assert all(r["ok"] for r in rows), [(r["name"], r["flags"]) for r in rows if not r["ok"]]
    for r in rows:
        assert 1e-4 < r["peak"] < 1.5 and -70 < r["loud"] < 0 and r["sharp"] > 0 and 0 <= r["above5k"] <= 1 and r["seconds"] >= 0
    deep = {r["name"]: r["flags"] for r in K.coverage(["toy_boat", "klang_*", "ui_tick", "damage"], kinds=("sfx",), deep=True)}
    assert deep["toy_boat"] == [] and deep["klang_car"] == [] and deep["klang_rain"] == [] and deep["damage"] == []
    assert deep["ui_tick"] == ["seed-ignored"]            # true: the default wooden tick has no random part
    # it must flag what is broken and keep going
    SX.REGISTRY["_qa_boom"] = {"fn": lambda sr: 1 / 0, "desc": "", "params": {}}
    SX.REGISTRY["_qa_nan"] = {"fn": lambda sr: np.full(100, np.nan), "desc": "", "params": {}}
    SX.REGISTRY["_qa_mute"] = {"fn": lambda sr: np.zeros(100), "desc": "", "params": {}}
    SX.REGISTRY["_qa_hot"] = {"fn": lambda sr: np.sin(np.arange(4410) / 7.0) * 3.0 * np.hanning(4410), "desc": "", "params": {}}
    SX.REGISTRY["_qa_step"] = {"fn": lambda sr: np.ones(4410) * 0.5, "desc": "", "params": {}}
    try:
        bad = {r["name"]: r for r in K.coverage("_qa_*", kinds=("sfx",))}
    finally:
        for k in [k for k in SX.REGISTRY if k.startswith("_qa_")]:
            del SX.REGISTRY[k]
    assert bad["_qa_boom"]["flags"] == ["error"] and "ZeroDivisionError" in bad["_qa_boom"]["error"]
    assert bad["_qa_nan"]["flags"] == ["non-finite"] and bad["_qa_mute"]["flags"] == ["silent"]
    assert "hot" in bad["_qa_hot"]["flags"] and {"dc", "ends-on-step"} <= set(bad["_qa_step"]["flags"])
    txt = K.report(list(bad.values()) + rows[:2])
    assert "5 flagged" in txt and txt.splitlines()[1].split()[1].startswith("_qa_")       # failures first
    assert [r["kind"] for r in K.coverage("gain", kinds=("effect",))] == ["effect"]
    assert K.main(["--only", "toy_boat", "--kind", "sfx"]) == 0


if __name__ == "__main__":
    for fn in [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]:
        fn()
        print("ok", fn.__name__)
