"""genny.creatures: measurable predictions of each model, plus a smoke loop over everything it registers.

Run:  uv run --with pytest pytest tests/test_creatures.py -q      (or: uv run python tests/test_creatures.py)
"""
import math

import numpy as np

import genny.creatures as C
from genny import instruments as I
from genny import sfx as X
from genny.analysis import brightness, loudness, sharpness
from genny.notes import midi_to_freq, note_to_midi

SR = 44100
SFX = [k for k, v in X.REGISTRY.items() if v["fn"].__module__ == "genny.creatures"]
INST = [k for k, v in I.REGISTRY.items() if v["fn"].__module__ == "genny.creatures"]
REPORT: dict[str, object] = {}


# ------------------------------------------------------------------ measurement helpers
def envelope(y, sr=SR, ms=4.0):
    k = max(1, int(sr * ms / 1000))
    return np.sqrt(np.convolve(y * y, np.ones(k) / k, "same"))


def bursts(y, sr=SR, ms=4.0, thresh=0.12, min_gap=0.012):
    """Onset times of the bursts separated by at least ``min_gap`` seconds below the threshold."""
    e = envelope(y, sr, ms)
    on = e > thresh * e.max()
    idx = np.flatnonzero(on)
    if not len(idx):
        return np.array([])
    starts = idx[np.r_[True, np.diff(idx) > min_gap * sr]]
    return starts / sr


def f0_of(y, sr, lo, hi):
    """Fundamental from the autocorrelation peak between lo and hi Hz."""
    y = y - y.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(y * np.hanning(len(y)), 2 * len(y))) ** 2)
    a, b = int(sr / hi), int(sr / lo)
    k = a + int(np.argmax(ac[a:b]))
    k = k + 0.5 * (ac[k - 1] - ac[k + 1]) / (ac[k - 1] - 2 * ac[k] + ac[k + 1] + 1e-20)
    return sr / k


def cents_off(y, f0, sr=SR):
    """As tests/test_instruments.py: autocorrelation peak next to the expected period."""
    y = y[int(0.05 * sr):int(0.05 * sr) + 16384]
    y = y - y.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(y * np.hanning(len(y)), 4 * len(y))) ** 2)
    p = sr / f0 * 4
    lo = int(p * 0.94)
    k = lo + int(np.argmax(ac[lo:int(p * 1.06) + 2]))
    k = k + 0.5 * (ac[k - 1] - ac[k + 1]) / (ac[k - 1] - 2 * ac[k] + ac[k + 1] + 1e-20)
    return 1200 * np.log2((sr * 4 / k) / f0)


def harmonic_env(y, sr, f0, fmax=5000.0):
    """(frequencies, amplitudes) of the harmonics of f0: samples of the spectral envelope."""
    n = len(y)
    sp = np.abs(np.fft.rfft(y * np.hanning(n)))
    hz = sr / n
    fs, am = [], []
    for k in range(1, int(fmax / f0) + 1):
        c = int(round(k * f0 / hz))
        w = max(2, int(0.3 * f0 / hz))
        fs.append(k * f0)
        am.append(sp[c - w:c + w + 1].max())
    return np.array(fs), np.array(am)


def formant_near(fs, am, target, span=0.3):
    """Envelope peak (parabolic over log amplitude) of the harmonics within +-span of target."""
    sel = np.flatnonzero((fs > target * (1 - span)) & (fs < target * (1 + span)))
    i = sel[np.argmax(am[sel])]
    if 0 < i < len(fs) - 1:
        a, b, c = np.log(am[i - 1:i + 2] + 1e-20)
        if b >= a and b >= c:
            return fs[i] + 0.5 * (a - c) / (a - 2 * b + c - 1e-20) * (fs[1] - fs[0])
    return fs[i]


def centroid(y, sr):
    sp = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / sr)
    return float((sp * f).sum() / sp.sum())


# ------------------------------------------------------------------ smoke: contract requirement 2 and 3
def test_smoke_sfx():
    for name in SFX:
        for sr in (22050, 48000):
            y = X.render_sfx(name, sr)
            assert y.ndim == 1 and np.all(np.isfinite(y)), name
            assert 1e-4 < np.max(np.abs(y)) < 1.5, (name, float(np.max(np.abs(y))))
            assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3, (name, "edges", y[0], y[-1])
            assert abs(y.mean()) < 2e-3, (name, "dc", float(y.mean()))
        assert np.array_equal(X.render_sfx(name, 22050), X.render_sfx(name, 22050)), (name, "not deterministic")
    REPORT["sfx"] = len(SFX)


def test_smoke_variants():
    """Every enumerated option renders."""
    cases = [("bark", dict(size=s)) for s in ("small", "medium", "large")]
    cases += [("cricket", dict(species=s)) for s in C._CRICKETS]
    cases += [("birdsong", dict(species=s)) for s in C.BIRDS]
    cases += [("bird_call", dict(species=s)) for s in C._TUBES] + [("bird_call", dict(two_voice=0, freq=300, freq_end=700))]
    cases += [("frog", dict(kind=k)) for k in ("croak", "ribbit", "trill")]
    cases += [("hiss", dict(kind=k)) for k in C._HISS] + [("wings", dict(kind=k)) for k in C._WINGS]
    cases += [("breath", dict(kind=k)) for k in ("in", "out", "pant", "gasp", "sigh")]
    cases += [("eat", dict(kind=k)) for k in ("crunch", "chew", "slurp")]
    cases += [("whistle_human", dict(kind=k)) for k in C._WHISTLES]
    cases += [("animal", dict(contour=k, vowels="i:-@-A")) for k in C._CONTOURS]
    cases += [("footsteps", dict(gait=g, steps=4)) for g in C.GAITS]
    cases += [("footsteps", dict(shoe=s, steps=3, ground="concrete")) for s in C.SHOES]
    cases += [("footsteps", dict(ground=g, steps=2)) for g in C.GROUNDS]
    cases += [("applause", dict(crowd=400, dur=1.5)), ("heartbeat", dict(rate=2.5, murmur=1.0)),
              ("monster", dict(size=6.0)), ("roar", dict(size=0.4, aggression=1.0))]
    for name, p in cases:
        y = X.render_sfx(name, SR, **p)
        assert np.all(np.isfinite(y)) and 1e-4 < np.max(np.abs(y)) < 1.5, (name, p)
        assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3, (name, p)
    REPORT["variants"] = len(cases)


# ------------------------------------------------------------------ insects
def test_cricket_follows_dolbear():
    """Chirps per minute = 7.2 T_C - 32 (Dolbear: T_F = 50 + (N - 40)/4)."""
    assert math.isclose(C.dolbear_rate((70 - 32) / 1.8) * 60, 4 * (70 - 50) + 40, rel_tol=1e-9)   # 70 F -> 120 / min
    out = {}
    for temp in (14.0, 20.0, 26.0):
        dur = 12.0
        y = C.cricket(SR, species="field", temp=temp, dur=dur)
        n = len(bursts(y, ms=8.0, thresh=0.1, min_gap=0.06))
        want = (7.2 * temp - 32.0) / 60.0 * dur
        out[temp] = (n, round(want, 1))
        assert abs(n - want) <= 1.5, (temp, n, want)
    y = C.cricket(SR, species="field", temp=16.4, dur=4.0)      # Farnell's recording: 7 pulses per chirp, 17 ms apart
    from scipy.signal import find_peaks
    e = envelope(y, SR, 2.0)
    pk, _ = find_peaks(e, height=0.1 * e.max(), distance=int(0.008 * SR), prominence=0.05 * e.max())
    first = pk[pk < pk[0] + 0.2 * SR] / SR
    assert len(first) == 7 and abs(np.median(np.diff(first)) - 0.017) < 0.002, (len(first), np.diff(first))
    assert e[pk[6]] > 2.0 * e[pk[0]]                               # amplitude rises over the burst, (n + 2)/9
    REPORT["cricket chirps in 12 s (measured, Dolbear)"] = out


def test_wingbeat_fundamental():
    out = {}
    for name, f in (("fly", 220.0), ("fly", 190.0), ("mosquito", 600.0), ("mosquito", 450.0), ("bee", 230.0)):
        y = X.render_sfx(name, SR, freq=f, wobble=0.0, motion=0.0, dur=1.0)
        got = f0_of(y[SR // 4:3 * SR // 4], SR, 0.7 * f, 1.4 * f)
        out[(name, f)] = round(got, 2)
        assert abs(got / f - 1) < 0.01, (name, f, got)
    for kind, rate in (("bird", 6.0), ("bat", 11.0), ("insect", 35.0)):
        y = C.wings(SR, kind=kind, rate=rate, dur=4.0)
        e = envelope(y, SR, 3.0)
        got = f0_of(e - e.mean(), SR, 0.6 * rate, 1.6 * rate)
        out[("wings", rate)] = round(got, 2)
        assert abs(got / rate - 1) < 0.04, (kind, rate, got)
    y = C.purr(SR, rate=25.0, dur=4.0)
    got = f0_of(y, SR, 15.0, 40.0)
    out["purr 25"] = round(got, 2)
    assert abs(got / 25.0 - 1) < 0.06, got
    REPORT["wingbeat Hz"] = out


# ------------------------------------------------------------------ body
def test_heartbeat_timing():
    out = {}
    for rate in (1.0, 1.5, 2.0):
        y = C.heartbeat(SR, rate=rate, dur=8.0)
        t = bursts(y, ms=6.0, thresh=0.08, min_gap=0.03)
        s1, s2 = t[0::2], t[1::2]
        m = min(len(s1), len(s2))
        period = float(np.median(np.diff(s1)))
        gap = float(np.median(s2[:m] - s1[:m]))
        want = 0.453 - 0.0017 * 60 * rate
        out[rate] = (round(period, 4), round(gap, 4), round(want, 4))
        assert abs(period - 1.0 / rate) < 0.004, (rate, period)
        assert abs(gap - want) < 0.006 and math.isclose(want, C.heart_s1s2(60 * rate)), (rate, gap, want)
    y = C.heartbeat(SR, rate=1.0, dur=3.0, dub_delay=0.5)
    t = bursts(y, ms=6.0, thresh=0.08, min_gap=0.03)
    assert abs((t[1] - t[0]) - 0.5) < 0.006
    lub = y[int((t[0] + 0.005) * SR):int((t[0] + 0.07) * SR)]       # 63 Hz x 5 cycles
    assert abs(f0_of(np.r_[lub, np.zeros(8 * len(lub))], SR, 40, 100) - 63.0) < 3.0
    REPORT["heartbeat (period, S1-S2, predicted)"] = out


def test_applause_count():
    out = {}
    for crowd, enth, dur in ((10, 0.2, 5.0), (40, 0.6, 4.0), (64, 1.0, 3.0)):
        n = sum(len(t) for t in C.applause_plan(crowd, enth, dur, seed=3))
        want = crowd * C.applause_rate(enth) * dur
        out[(crowd, enth, dur)] = (n, round(want))
        assert abs(n / want - 1) < 0.2, (crowd, n, want)
    y = C.applause(SR, crowd=1, enthusiasm=0.4, dur=6.0, seed=1)     # one clapper: count the claps in the audio
    n = len(bursts(y, ms=3.0, thresh=0.15, min_gap=0.05))
    want = C.applause_rate(0.4) * 6.0
    out["1 person, audio"] = (n, round(want, 1))
    assert abs(n / want - 1) < 0.2, (n, want)
    lo = loudness(C.applause(SR, crowd=5, dur=2.0), SR)              # peak-normalised: denser = higher mean level
    hi = loudness(C.applause(SR, crowd=60, dur=2.0), SR)
    assert hi > lo
    REPORT["applause claps (counted, crowd x rate x dur)"] = out


# ------------------------------------------------------------------ mammals
def test_size_scales_pitch_and_formants():
    out = {}
    # f0 ~ 1/size
    f = [f0_of(C.animal(48000, size=s, pitch=200.0, contour="flat", roughness=0.0, vowels="A", dur=1.0)[12000:36000],
               48000, 60, 600) for s in (0.5, 1.0, 2.0)]
    out["animal f0 at size .5/1/2"] = [round(v, 1) for v in f]
    assert abs(f[0] / 400 - 1) < 0.03 and abs(f[1] / 200 - 1) < 0.03 and abs(f[2] / 100 - 1) < 0.03, f
    # resonances ~ 1/size at the same f0 (pitch doubled with size): the tract's quarter-wave mode
    # modes c/4l, 5c/4l = 500, 2500 Hz / size (Farnell, l = 0.17 m x size), which the schwa formants share
    got = {}
    for s in (1.0, 2.0):
        y = C.animal(48000, size=s, pitch=80.0 * s, contour="flat", roughness=0.3, vowels="@", dur=1.5)[16000:56000]
        fs, am = harmonic_env(y, 48000, f0_of(y, 48000, 60, 110), 4000.0)
        got[s] = (formant_near(fs, am, 500 / s, 0.2), formant_near(fs, am, 2500 / s, 0.2))
        assert abs(got[s][0] / (500 / s) - 1) < 0.15 and abs(got[s][1] / (2500 / s) - 1) < 0.15, (s, got[s])
    out["animal tract modes 1 and 3 at size 1/2"] = {k: [round(x) for x in v] for k, v in got.items()}
    assert got[2.0][0] < 0.65 * got[1.0][0] and got[2.0][1] < 0.65 * got[1.0][1]
    # roar: peak f0 = 240 / size (Farnell arc), spectrum moves down with size
    pk, cen = [], []
    for s in (1.0, 2.0):
        y = C.roar(48000, size=s, aggression=0.0, dur=3.0)
        i = int(0.45 * len(y))
        pk.append(f0_of(y[i - 4800:i + 4800], 48000, 40, 400))
        cen.append(centroid(y, 48000))
    out["roar peak f0, centroid at size 1/2"] = ([round(v, 1) for v in pk], [round(v) for v in cen])
    assert abs(pk[0] / 240 - 1) < 0.08 and abs(pk[1] / 120 - 1) < 0.08, pk
    assert cen[1] < 0.7 * cen[0], cen
    y = C.roar(48000, size=1.0, aggression=0.0, dur=3.0)             # the arc starts low and ends near 120 Hz
    assert f0_of(y[int(0.9 * len(y)) - 4800:int(0.9 * len(y)) + 4800], 48000, 60, 300) < 0.75 * pk[0]
    REPORT["size scaling"] = out


# ------------------------------------------------------------------ birds
def test_birdsong_matches_preset():
    out = {}
    for sp in ("cuckoo", "owl", "crow", "canary", "rooster", "sparrow", "duck"):
        plan = C.birdsong_plan(sp, seed=2)
        y = C.birdsong(SR, species=sp, seed=2)
        n = len(bursts(y, ms=2.0, thresh=0.06, min_gap=0.012))
        assert n == len(plan), (sp, n, len(plan))
        lo, hi = C.bird_range(sp)
        fs = []
        for s in plan:                                              # dominant frequency inside each syllable
            seg = y[int((s["t"] + 0.2 * s["dur"]) * SR):int((s["t"] + 0.8 * s["dur"]) * SR)]
            sp_ = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 8192))
            fs.append(np.argmax(sp_) * SR / 8192)
        ok = [lo * 0.93 <= v <= hi * 1.07 or lo * 0.93 <= v / 2 <= hi * 1.07 or lo * 0.93 <= v / 3 <= hi * 1.07
              for v in fs]                                          # pulse-rich calls peak on a harmonic
        assert np.mean(ok) > 0.9, (sp, lo, hi, fs)
        if sp in ("cuckoo", "owl", "canary"):                       # whistles: the carrier itself is the peak
            assert all(lo * 0.93 <= v <= hi * 1.07 for v in fs), (sp, lo, hi, fs)
        out[sp] = (n, (round(lo), round(hi)), (round(min(fs)), round(max(fs))))
    y = C.birdsong(SR, species="cuckoo", repeats=3)
    assert len(bursts(y, ms=2.0, thresh=0.06, min_gap=0.012)) == 6
    assert len(bursts(C.birdsong(SR, species="robin", syllables=5), ms=2.0, thresh=0.06, min_gap=0.012)) == 5
    seg = C.birdsong(SR, species="cuckoo")                          # a falling third: 667 -> 530 Hz
    t = bursts(seg, ms=2.0, thresh=0.06, min_gap=0.012)
    a = f0_of(seg[int((t[0] + 0.05) * SR):int((t[0] + 0.17) * SR)], SR, 400, 900)
    b = f0_of(seg[int((t[1] + 0.05) * SR):int((t[1] + 0.25) * SR)], SR, 400, 900)
    assert abs(a / 661 - 1) < 0.02 and abs(b / 525 - 1) < 0.02, (a, b)
    plan = C.birdsong_plan("woodpecker", seed=1)                    # drum: one strike per plan entry
    y = C.birdsong(SR, species="woodpecker", seed=1)
    e = envelope(y, SR, 1.0)
    from scipy.signal import find_peaks
    pk, _ = find_peaks(e, height=0.2 * e.max(), distance=int(0.03 * SR), prominence=0.1 * e.max())
    assert len(pk) == len(plan), (len(pk), len(plan))
    out["woodpecker strikes"] = len(pk)
    REPORT["birdsong (syllables, preset range Hz, measured range Hz)"] = out


def test_syrinx_follows_tension():
    """Smyth-Smith valve: the call's f0 follows the membrane frequency (the kernel's stated working range)."""
    out = {}
    for f in (300.0, 450.0, 700.0):
        y = C.bird_call(SR, freq=f, two_voice=0.0, dur=0.6)
        got = f0_of(y[int(0.25 * SR):int(0.5 * SR)], SR, 0.6 * f, 1.6 * f)
        out[f] = round(got, 1)
        assert abs(got / f - 1) < 0.12, (f, got)
    REPORT["bird_call f0 vs membrane frequency"] = out


# ------------------------------------------------------------------ footsteps
def test_footsteps_count_pace_and_variation():
    out = {}
    for ground, pace, steps in (("concrete", 100.0, 8), ("wood", 120.0, 10), ("gravel", 80.0, 6)):
        y = C.footsteps(SR, ground=ground, gait="walk", pace=pace, steps=steps, variation=0.3, seed=4)
        plan = C.footsteps_plan("walk", pace, steps, 0.3, 4)
        e = envelope(y, SR, 20.0)
        from scipy.signal import find_peaks
        pk, _ = find_peaks(e, height=0.25 * e.max(), distance=int(0.6 * 60 / pace * SR))
        ioi = float(np.median(np.diff(pk))) / SR
        out[(ground, pace)] = (len(pk), round(ioi, 3), round(60 / pace, 3))
        assert len(pk) == steps == len(plan), (ground, len(pk), steps)
        assert abs(ioi / (60 / pace) - 1) < 0.06, (ground, ioi, 60 / pace)
        assert abs(np.mean(np.diff(plan)) / (60 / pace) - 1) < 0.04
    # never the same step twice: consecutive steps are different samples even with variation 0
    y = C.footsteps(48000, ground="concrete", pace=100.0, steps=6, variation=0.0)
    t = C.footsteps_plan("walk", 100.0, 6, 0.0, 0)
    assert np.allclose(np.diff(t), 0.6)
    k = int(0.3 * 48000)
    segs = [y[int(round(v * 48000)):int(round(v * 48000)) + k] for v in t]
    worst = 0.0
    for a, b in zip(segs, segs[1:] + segs[:1] + segs[2:]):
        assert not np.array_equal(a, b)
        c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        worst = max(worst, c)
    assert worst < 0.99, worst
    out["max correlation between steps"] = round(worst, 3)
    # gait structure: running is staccato (silence between steps), walking is legato (Sounding Object ch. 6)
    def quiet(g, pace):
        yy = C.footsteps(SR, ground="concrete", gait=g, pace=pace, steps=8, variation=0.0)
        e = envelope(yy, SR, 10.0)
        tt = C.footsteps_plan(g, pace, 8, 0.0, 0)
        mid = e[int(tt[1] * SR):int(tt[-1] * SR)]
        return float(np.mean(mid < 0.02 * e.max()))
    qw, qr = quiet("walk", 110.0), quiet("run", 110.0)
    out["silent fraction walk / run at 110"] = (round(qw, 3), round(qr, 3))
    assert qr > qw + 0.1, (qw, qr)
    # limp: short-long alternation; gallop: four beats then a suspension longer than any beat gap
    d = np.diff(C.footsteps_plan("limp", 90.0, 9, 0.0, 0))
    assert np.allclose(d[0::2] / d[1::2], 0.72 / 1.28)
    g = np.diff(C.footsteps_plan("gallop", 440.0, 12, 0.0, 0))
    assert np.all(g[3::4] > 2.0 * g.reshape(-1)[[0, 1, 2]].max()) and math.isclose(g[:4].sum(), 4 * 60 / 440.0)
    tr = np.diff(C.footsteps_plan("trot", 312.0, 8, 0.0, 0))
    assert np.allclose(tr[0::2], tr[0]) and tr[0] < 0.1 * tr[1]     # diagonal pairs land together: two beats
    REPORT["footsteps (count, IOI, 60/pace)"] = out


def test_frog_size():
    """One scale for the whole body: a frog twice the size calls an octave lower and twice as long."""
    a = C.frog(48000, kind="croak", size=1.0, calls=1)
    b = C.frog(48000, kind="croak", size=2.0, calls=1)
    assert abs(centroid(a, 48000) / centroid(b, 48000) - 2.0) < 0.3
    REPORT["frog centroid size 1 / 2"] = (round(centroid(a, 48000)), round(centroid(b, 48000)))


# ------------------------------------------------------------------ instruments
def test_instruments_shapes_and_levels():
    for name in INST:
        e = I.REGISTRY[name]
        lo, hi = (note_to_midi(n) for n in e["range"].split("-"))
        for m in (lo - 5, lo, (lo + hi) // 2, hi, hi + 7):
            for dur, vel, sr in ((0.03, 1.0, 44100), (0.1234, 0.3, 44100), (0.3333, 0.7, 22050), (1.0, 1.0, 48000)):
                y = I.render_note(name, midi_to_freq(m), dur, sr, vel)
                assert y.ndim == 1 and np.all(np.isfinite(y)), (name, m, dur)
                assert 1e-4 < np.max(np.abs(y)) < 1.5, (name, m, dur, float(np.max(np.abs(y))))
                assert abs(y[-1]) < 1e-3 and abs(y[0]) < 1e-2, (name, "edges")
        a = I.render_note(name, 220.0, 0.2, 44100, 0.8)
        assert np.array_equal(a, I.render_note(name, 220.0, 0.2, 44100, 0.8)), (name, "not deterministic")


def test_instruments_tuning():
    out = {}
    for name in INST:
        e = I.REGISTRY[name]
        lo, hi = (note_to_midi(n) for n in e["range"].split("-"))
        cs = []
        for m in (lo, (lo + hi) // 2, hi):
            for dur in (0.6, 1.0):
                c = cents_off(I.render_note(name, midi_to_freq(m), dur, SR, 0.9), midi_to_freq(m))
                assert abs(c) < 10, (name, m, dur, round(float(c), 1))
                cs.append(round(float(c), 1))
            for dur in (0.03, 0.15):                                # short notes: count the zero crossings of the fundamental
                f = midi_to_freq(m)
                y = I.render_note(name, f, dur, 48000, 0.9)
                got = f0_of(np.r_[y, np.zeros(4 * len(y))], 48000, f * 0.9, f * 1.1)
                assert abs(1200 * np.log2(got / f)) < (10 if dur > 0.1 else 25), (name, m, dur, got, f)
        out[name] = cs
    REPORT["instrument cents (lo, mid, hi x 0.6 s, 1 s)"] = out


def test_instruments_not_piercing_and_level():
    out = {}
    for name in INST:
        e = I.REGISTRY[name]
        lo, hi = (note_to_midi(n) for n in e["range"].split("-"))
        for m in sorted({min((lo + hi) // 2, 96), min(hi, 96)}):
            f = midi_to_freq(m)
            y = I.render_note(name, f, 0.5, SR, 0.9)
            pure = sharpness(I.render_note("sine", f, 0.5, SR, 0.9), SR)
            s, b = sharpness(y, SR), brightness(y, SR)
            assert s < max(pure + 1.0, 1.5) and s < 2.4, (name, m, round(s, 2), round(pure, 2))
            assert b["above_5k"] < 0.03, (name, m, round(b["above_5k"], 3))
        lv = [loudness(I.render_note(name, midi_to_freq(m), 0.5, SR, 0.9), SR, window=0.15)
              for m in (lo + (hi - lo) // 4, (lo + hi) // 2, hi - (hi - lo) // 4)]
        out[name] = [round(v, 1) for v in lv]
        assert abs(sum(lv) / 3 + 12.0) < 6.0, (name, lv)
    REPORT["instrument loudness dB (target -12)"] = out


if __name__ == "__main__":
    import time
    for fn in [v for k, v in list(globals().items()) if k.startswith("test_")]:
        t = time.time()
        fn()
        print(f"ok {fn.__name__} ({time.time() - t:.1f} s)")
    for k, v in REPORT.items():
        print(f"{k}: {v}")
