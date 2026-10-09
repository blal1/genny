"""Finite-difference strings and bars (genny/fdstring.py): physical checks and a smoke loop.

Run:  uv run --with pytest pytest tests/test_fdstring.py -q      (or: uv run python tests/test_fdstring.py)
"""
import numpy as np

from genny import fdstring as M
from genny import fx as FX
from genny import instruments as I
from genny import sfx as S
from genny.analysis import brightness, loudness, sharpness
from genny.notes import midi_to_freq, note_to_midi

SR = 44100
NAMES = ("fd_string", "fd_piano", "prepared_piano", "slack_string", "fd_bar", "bowed_string_fd")


def peak_freq(y, sr, f, w=0.03):
    """Frequency of the spectral peak within +-w of f (parabolic interpolation on a zero-padded FFT)."""
    n = len(y)
    Y = np.abs(np.fft.rfft(y * np.hanning(n), 8 * n))
    fr = np.fft.rfftfreq(8 * n, 1 / sr)
    m = np.flatnonzero((fr > f * (1 - w)) & (fr < f * (1 + w)))
    i = m[np.argmax(Y[m])]
    a, b, c = np.log(Y[i - 1:i + 2] + 1e-300)
    return fr[i] + 0.5 * (a - c) / (a - 2 * b + c) * (fr[1] - fr[0])


def cents_off(y, f0, sr=SR):
    """Same estimator as tests/test_instruments.py."""
    y = y[int(0.05 * sr):int(0.05 * sr) + 16384]
    y = y - y.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(y * np.hanning(len(y)), 4 * len(y))) ** 2)
    p = sr / f0 * 4
    lo = int(p * 0.94)
    k = lo + int(np.argmax(ac[lo:int(p * 1.06) + 2]))
    k = k + 0.5 * (ac[k - 1] - ac[k + 1]) / (ac[k - 1] - 2 * ac[k] + ac[k + 1] + 1e-20)
    return 1200 * np.log2((sr * 4 / k) / f0)


def t60_of(y, sr, f, t0=0.1, t1=1.5):
    """Decay time of the partial at f: slope of its short-time amplitude."""
    n = 4096
    t = np.arange(n) / sr
    probe = np.hanning(n) * np.exp(-2j * np.pi * f * t)
    ts = np.arange(int(t0 * sr), int(t1 * sr) - n, 1024)
    db = np.array([20 * np.log10(abs(np.dot(y[i:i + n], probe)) + 1e-300) for i in ts])
    return -60.0 / np.polyfit(ts / sr, db, 1)[0]


def test_lossless_energy_conserved():
    worst = 0.0
    cases = [dict(), dict(bc=("clamped", "clamped")), dict(tension=3000.0, backoff=0.75), dict(tension=500.0, backoff=0.8)]
    for kw in cases:
        s = M.StiffString(110.0, SR, B=1e-3, t60=None, **kw).pluck(0.3, 0.08, 0.02)
        e = s.run(SR, energy=True)[1][2:]
        worst = max(worst, (e.max() - e.min()) / e.mean())
    s = M.StiffString(110.0, SR, B=1e-3, t60=None).pluck(0.3, 0.08, 0.02).spring(0.5, w0=2000.0, w1=300.0)
    e = s.run(SR, energy=True)[1][2:]
    worst = max(worst, (e.max() - e.min()) / e.mean())
    for bc in (("free", "free"), ("clamped", "free"), ("ss", "ss"), ("clamped", "clamped")):
        b = M.Bar(440.0, SR, bc, None, 32).pluck(0.4, 0.2, 1.0)
        e = b.run(200000, energy=True)[1][2:]
        worst = max(worst, (e.max() - e.min()) / e.mean())
    print("lossless energy drift (relative), worst of 9 schemes:", worst)
    assert worst < 1e-10, worst


def test_long_nonlinear_energy():
    """g01 3.1 warning: a nonlinear scheme can run for 400k steps before blowing up. 1e6 steps, hard pluck."""
    s = M.StiffString(220.0, SR, B=1e-4, t60=None, tension=20000.0, backoff=0.75).pluck(0.5, 0.2, 0.05)
    out, e = s.run(1_000_000, energy=True)
    e = e[2:]
    drift = (e.max() - e.min()) / e.mean()
    print("Kirchhoff-Carrier, 1e6 steps: energy drift", drift, "peak", np.abs(out).max())
    assert drift < 1e-9 and np.abs(out).max() < 0.5


def test_hammer_energy_passes_to_string():
    """Hammer + string: total energy after the hammer has left equals the hammer's initial kinetic energy."""
    for alpha, tol in ((1.0, 5e-3), (2.5, 5e-3), (3.0, 5e-3)):
        s = M.StiffString(262.0, SR, B=1e-4, t60=None, backoff=0.95).hammer(0.12, 0.75, 3000.0, alpha, 1.5)
        e = s.run(int(0.05 * SR), energy=True)[1]
        e0 = 0.5 * 0.75 * 1.5 ** 2
        print(f"hammer alpha={alpha}: energy before {e[0]:.6f} (analytic {e0:.6f}) after release {e[-1]:.6f} ratio {e[-1] / e[0]:.5f}")
        assert abs(e[0] - e0) < 1e-9 and s.force[-1] == 0.0 and abs(e[-1] / e[0] - 1.0) < tol


def test_stiff_string_partials():
    f1, B = 110.0, 1e-3
    s = M.StiffString(f1, SR, B=B, t60=(f1, 8.0, 4000.0, 4.0)).pluck(0.137, 0.03, 1.0)
    y = s.render(2.0, (0.221, 0.6))[:, 0]
    f0 = f1 / np.sqrt(1 + B)
    errs = []
    for n in (1, 2, 3, 4, 5, 6, 8):
        want = n * f0 * np.sqrt(1 + B * n * n)
        errs.append(1200 * np.log2(peak_freq(y, SR, want, 0.02) / want))
        assert abs(1200 * np.log2(s.partial(n)[0] / peak_freq(y, SR, want, 0.02))) < 0.3   # closed form == simulation
    print("partials 1,2,3,4,5,6,8 vs n f0 sqrt(1+B n^2), cents:", np.round(errs, 2))
    assert abs(errs[0]) < 0.1 and max(abs(e) for e in errs) < 5.0
    # clamped ends: the tuned fundamental is still on pitch (eigenvalue route)
    c = M.StiffString(196.0, SR, B=2e-3, t60=(196.0, 8.0, 4000.0, 4.0), bc=("clamped", "clamped")).pluck(0.2, 0.05, 1.0)
    fc = peak_freq(c.render(1.5)[:, 0], SR, 196.0)
    print("clamped stiff string fundamental:", fc, "Hz (wanted 196)")
    assert abs(1200 * np.log2(fc / 196.0)) < 0.5


def test_bar_table_7_1():
    """Book Table 7.1: explicit scheme, simply-supported bar, fundamental 100 Hz at 44.1 kHz, mu at the bound."""
    k = 1 / 44100.0
    kappa = 200.0 / np.pi                      # f_p = pi kappa p^2 / 2
    N = int(1 / np.sqrt(2 * kappa * k))
    got = [M.ss_mode(p, N, 0.0, kappa, 0.0, 0.0, k)[0] for p in range(1, 7)]
    print("Table 7.1 explicit column:", np.round(got, 1))
    assert np.allclose(got, [99.7, 396.0, 880.2, 1539.1, 2356.4, 3313.5], atol=0.15)


def test_t60_two_frequencies():
    f1 = 220.0
    for bc in (("ss", "ss"),):
        s = M.StiffString(f1, SR, B=2e-4, t60=(f1, 3.0, 2200.0, 1.0), bc=bc).pluck(0.137, 0.03, 1.0)
        y = s.render(2.0, (0.221, 0.6))[:, 0]
        f10 = s.partial(10)[0]
        lo, hi = t60_of(y, SR, f1), t60_of(y, SR, f10)
        want_hi = M._LN1000 / (s.sig0 + s.sig1 * (2 * np.pi * f10) ** 2 * 2 / (s.gamma ** 2 + np.sqrt(s.gamma ** 4 + 4 * s.kappa ** 2 * (2 * np.pi * f10) ** 2)))
        print(f"string T60: {lo:.3f} s at {f1} Hz (asked 3.0), {hi:.3f} s at {f10:.0f} Hz (law gives {want_hi:.3f}; asked 1.0 at 2200)")
        assert abs(lo / 3.0 - 1) < 0.05 and abs(hi / want_hi - 1) < 0.05 and abs(hi / 1.0 - 1) < 0.08
    b = M.Bar(440.0, SR, ("free", "free"), (440.0, 1.2, 4000.0, 0.3), 32).hammer(0.37, 0.3, 6000.0, 2.0, 1.5)
    y = b.render(1.4, (0.42, 0.85))[:, 0]
    lo = t60_of(y, SR, 440.0, 0.1, 1.2)
    print(f"bar T60 at 440 Hz: {lo:.3f} s (asked 1.2)")
    assert abs(lo / 1.2 - 1) < 0.05


def _track(y, sr, f, t0, t1):
    seg = y[int(t0 * sr):int(t1 * sr)]
    return peak_freq(seg, sr, f, 0.25)


def test_tension_modulation_glide():
    """Kirchhoff-Carrier: hard pluck starts sharp and glides down to the linear pitch; soft pluck does not."""
    f = 110.0
    res = {}
    for vel in (1.0, 0.05):
        s, y = M._slack(f, 1.5, SR, 3.0, vel, 0.8, 1.5, 0.3)
        y = s.audio(y)[:, 0]
        res[vel] = [1200 * np.log2(_track(y, SR, f, a, a + 0.12) / f) for a in (0.0, 0.15, 0.4, 1.0)]
    print("glide (cents vs note) at t=0, 0.15, 0.4, 1.0 s: hard", np.round(res[1.0], 1), "soft", np.round(res[0.05], 1))
    h = res[1.0]
    assert h[0] > 60 and h[0] > h[1] > h[2] > h[3] - 0.5 and abs(h[3]) < 5
    assert max(abs(c) for c in res[0.05]) < 6
    # the default instrument still reads in tune (glide over in the first tens of ms)
    assert abs(cents_off(I.render_note("slack_string", f, 0.6, SR, 0.9), f)) < 10


def test_hammer_velocity_brightens():
    """g01 3.3 / g02 section 6: stiffening felt -> harder strike = shorter contact = brighter (Fig 7.11)."""
    rows = []
    for v in (0.5, 1.5, 5.0):
        s = M.StiffString(262.0, SR, B=1e-4, t60=(262.0, 5.0, 4000.0, 1.0), backoff=0.95).hammer(0.12, 0.75, 3000.0, 2.5, v)
        y = s.run(int(0.5 * SR), (0.31, 0.77))[:, 0]
        contact = np.count_nonzero(s.force > 0) / SR * 1e3
        Y = np.abs(np.fft.rfft(np.diff(y) * np.hanning(len(y) - 1)))
        fr = np.fft.rfftfreq(len(y) - 1, 1 / SR)
        rows.append((v, contact, float(np.sum(fr * Y) / np.sum(Y)), float(s.force.max())))
    for r in rows:
        print("hammer v=%.1f: contact %.2f ms, centroid %.0f Hz, peak force %.0f" % r)
    assert rows[0][1] > rows[1][1] > rows[2][1] and 1.0 < rows[2][1] and rows[0][1] < 4.0
    assert rows[0][2] < rows[1][2] < rows[2][2]
    assert rows[0][3] < rows[1][3] < rows[2][3]
    # a linear spring (alpha = 1) gives the same contact time at every speed: no timbre change
    lin = []
    for v in (0.5, 5.0):
        s = M.StiffString(262.0, SR, B=1e-4, t60=None, backoff=0.95).hammer(0.12, 0.75, 3000.0, 1.0, v)
        s.run(int(0.05 * SR))
        lin.append(np.count_nonzero(s.force > 0))
    print("linear felt contact samples at v=0.5 / 5:", lin)
    assert abs(lin[0] - lin[1]) <= 1


def test_course_two_stage_decay():
    """g02 section 7: three strings a few cents apart cancel as they drift -> fast early decay, slow aftersound."""
    def env(det):
        s = M.StiffString(110.0, SR, B=0.0, t60=(110.0, 12.0, 4000.0, 12.0), n_strings=3, detune=det, backoff=0.95)
        s.hammer(0.1, 1.0 / 3, 800.0, 2.5, 1.0)
        y = s.run(int(3.0 * SR), (0.3, 0.3))[:, 0]
        r = [np.sqrt(np.mean(y[int(a * SR):int((a + 0.25) * SR)] ** 2)) for a in (0.05, 0.6, 1.2)]
        return 20 * np.log10(r[1] / r[0]), 20 * np.log10(r[2] / r[1])
    a, b = env(0.0), env(8.0)
    print("course dB change (0.05->0.6 s, 0.6->1.2 s): unison", np.round(a, 2), "detuned 8 cents", np.round(b, 2))
    assert b[0] < a[0] - 1.5        # the detuned course drops faster at first than loss alone


def test_prepared_elements():
    f = 110.0
    def partial1(**kw):
        s = M.StiffString(f, SR, B=0.0, t60=(f, 6.0, 4000.0, 2.0)).pluck(0.65, 0.1, 1e-3)
        if kw:
            s.spring(0.5, **kw)
        return s, s.render(1.5, (0.8, 0.3))[:, 0]
    _, y0 = partial1()
    _, y1 = partial1(w0=np.pi * 220.0 * 2)
    f_plain, f_spring = peak_freq(y0, SR, f), peak_freq(y1, SR, f * 1.2, 0.25)
    _, yd = partial1(sigma=1.0)
    td, want = t60_of(yd, SR, f, 0.1, 1.2), M._LN1000 / (M._LN1000 / 6.0 + 2.0 * 1.0)   # added rate 2 sigma_P U1(x_P)^2
    print(f"mid-string spring: partial 1 {f_plain:.2f} -> {f_spring:.2f} Hz; damper sigma_P=1: T60 6.00 -> {td:.3f} s (perturbation theory {want:.3f})")
    assert f_spring > f_plain * 1.03 and abs(td / want - 1) < 0.03
    # heavy damper pins the midpoint: the odd partials die and the note jumps to the octave (Fig 7.16b)
    _, yh = partial1(sigma=200.0)
    def amp(y, fq):
        Y = np.abs(np.fft.rfft(y[4410:4410 + 32768] * np.hanning(32768)))
        fr = np.fft.rfftfreq(32768, 1 / SR)
        return Y[(fr > fq - 10) & (fr < fq + 10)].max()
    print(f"heavy damper: 110 Hz partial x{amp(yh, 110) / amp(y0, 110):.1e}, 220 Hz partial x{amp(yh, 220) / amp(y0, 220):.4f}")
    assert amp(yh, 110) < 1e-3 * amp(y0, 110) and abs(amp(yh, 220) / amp(y0, 220) - 1) < 0.01
    # rattle: only a large excursion reaches it; it adds high-frequency energy and stays bounded
    def hi_share(amp, rattle=True):
        s = M.StiffString(f, SR, B=0.0, t60=(f, 6.0, 4000.0, 2.0)).pluck(0.65, 0.1, amp)
        if rattle:
            s.rattle(0.2, w=8000.0, eps=1e-3, mass_ratio=100.0, alpha=2.0)
        y = s.render(1.0, (0.8, 0.3))[:, 0]
        Y = np.abs(np.fft.rfft(y[2000:])) ** 2
        fr = np.fft.rfftfreq(len(y) - 2000, 1 / SR)
        assert np.all(np.isfinite(y)) and np.abs(y).max() < 3 * amp
        return Y[fr > 1500].sum() / Y.sum()
    quiet, loud, plain = hi_share(2e-4), hi_share(5e-3), hi_share(5e-3, False)
    print(f"rattle: share of energy above 1.5 kHz: excursion inside the gap {quiet:.2e}, beyond it {loud:.2e}, same pluck without rattle {plain:.2e}")
    assert quiet == hi_share(2e-4, False) and loud > 2 * plain


def test_bar_modes():
    b = M.Bar(440.0, SR, ("free", "free"), (440.0, 3.0, 4000.0, 2.0), 32).hammer(0.13, 0.3, 20000.0, 2.0, 1.5)
    y = b.render(1.0, (0.07, 0.93))[:, 0]
    got = [peak_freq(y, SR, 440.0 * r, 0.04) / 440.0 for r in (1.0, 2.756, 5.404)]
    print("free-free bar partial ratios:", np.round(got, 4), "(theory 1 : 2.756 : 5.404); scheme prediction", np.round(b.modes[:3] / 440.0, 4))
    assert abs(got[0] - 1) < 2e-4 and abs(got[1] / 2.756 - 1) < 0.01 and abs(got[2] / 5.404 - 1) < 0.01
    assert np.allclose(got, b.modes[:3] / 440.0, rtol=2e-3)
    c = M.Bar(220.0, SR, ("clamped", "free"), None, 32)
    print("clamped-free ratios:", np.round(c.modes[:3] / 220.0, 3), "(theory 1 : 6.267 : 17.55)")
    assert abs(c.modes[1] / 220.0 / 6.267 - 1) < 0.01 and abs(c.modes[2] / 220.0 / 17.55 - 1) < 0.01
    for name, ratio in (("xylophone", 3.0), ("marimba", 4.0)):
        s, y = M._bar_note(330.0, 0.6, SR, 0.9, "metal", "free", 0.9, name, 0.3, 0.1, 1)
        y = s.audio(y)[:, 0]
        r = peak_freq(y, SR, 330.0 * ratio, 0.04) / peak_freq(y, SR, 330.0)
        print(f"undercut bar '{name}': arch depth {s.phi.min():.3f}, partial 2 / partial 1 = {r:.4f}")
        assert abs(r / ratio - 1) < 0.003


def test_bow():
    s = M.StiffString(196.0, SR, B=1e-5, t60=(196.0, 3.0, 4000.0, 1.0))
    n = int(0.8 * SR)
    still = s.bow(np.full(n, 0.1 * s.gamma), np.zeros(n), np.full(n, 0.12))
    y = s.bow(np.full(n, 0.1 * s.gamma), np.full(n, 0.2), np.linspace(0.12, 0.2, n))[:, 0]
    f = peak_freq(y[n // 2:], SR, 196.0)
    print(f"bowed string: steady pitch {f:.2f} Hz (string 196), peak displacement {np.abs(y).max():.2e}")
    assert np.abs(still).max() == 0.0                 # passive: no bow motion, no sound
    assert np.all(np.isfinite(y)) and np.abs(y[n // 2:]).max() > 1e-5 and abs(1200 * np.log2(f / 196.0)) < 15


def test_instruments_contract():
    """The three checks of tests/test_instruments.py for the instruments registered here."""
    for name in NAMES:
        e = I.REGISTRY[name]
        lo, hi = (note_to_midi(n) for n in e["range"].split("-"))
        for m in (lo - 5, lo, (lo + hi) // 2, hi, hi + 7):
            for dur, vel, sr in ((0.03, 1.0, 44100), (0.1234, 0.3, 44100), (0.3333, 0.7, 22050), (1.0, 1.0, 48000)):
                y = I.render_note(name, midi_to_freq(m), dur, sr, vel)
                assert y.ndim == 1 and np.all(np.isfinite(y)), (name, m, dur)
                assert 1e-4 < np.max(np.abs(y)) < 1.5, (name, m, dur, float(np.max(np.abs(y))))
                assert abs(y[-1]) < 1e-3 and abs(y[0]) < 1e-3 and abs(y.mean()) < 0.02, (name, m, dur)
                assert np.array_equal(y, I.render_note(name, midi_to_freq(m), dur, sr, vel)) or dur > 0.05, name
        cents, lev = [], []
        for m in (lo, (lo + hi) // 2, hi):
            f = midi_to_freq(m)
            cents.append(float(cents_off(I.render_note(name, f, 0.6, SR, 0.9), f)))
            assert abs(cents[-1]) < 10, (name, m, cents[-1])
        for m in sorted({min((lo + hi) // 2, 96), min(hi, 96)}):
            f = midi_to_freq(m)
            y = I.render_note(name, f, 0.5, SR, 0.9)
            pure = sharpness(I.render_note("sine", f, 0.5, SR, 0.9), SR)
            s, b = sharpness(y, SR), brightness(y, SR)
            assert s < max(pure + 1.0, 1.5) and s < 2.4 and b["above_5k"] < 0.03, (name, m, s, pure, b["above_5k"])
        for m in (lo + (hi - lo) // 4, (lo + hi) // 2, hi - (hi - lo) // 4):
            lev.append(loudness(I.render_note(name, midi_to_freq(m), 0.5, SR, 0.9), SR, window=0.15))
        print(f"{name}: cents at bottom/middle/top {np.round(cents, 1)}, level {np.mean(lev):.1f} dB (K-weighted)")
        assert abs(np.mean(lev) + 12.0) < 6.0, (name, lev)


def test_variants_smoke():
    """Every option of every registered name, two sample rates: finite, bounded, no DC, zero at both ends."""
    variants = [("fd_string", dict(excitation="strike")), ("fd_string", dict(excitation="bow")), ("fd_string", dict(B=2e-3, position=0.5)),
                ("fd_piano", dict(strings=3, detune=12.0, hardness=4.0)), ("fd_piano", dict(hammer_mass=3.0, hardness=0.2)),
                ("slack_string", dict(glide=6.0)), ("bowed_string_fd", dict(force=4.0, travel=0.3)), ("bowed_string_fd", dict(force=0.05, speed=4.0))]
    variants += [("prepared_piano", dict(preparation=p, amount=a)) for p in M._PREP for a in (1.0, 3.0)]
    variants += [("fd_bar", dict(material=mt, boundary=b)) for mt in M._MATERIAL for b in M._BOUNDARY]
    variants += [("fd_bar", dict(tuning=t, mallet=h, quality=q)) for t, h, q in (("marimba", 0.0, 0), ("xylophone", 1.0, 2))]
    for sr in (22050, 48000):
        for name, kw in variants:
            for note in ("C2", "A3", "C6"):
                y = I.render_note(name, midi_to_freq(note_to_midi(note)), 0.25, sr, 0.8, **kw)
                assert np.all(np.isfinite(y)) and 1e-4 < np.max(np.abs(y)) < 1.5, (name, kw, note, sr, float(np.max(np.abs(y))))
                assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3 and abs(y.mean()) < 0.02, (name, kw, note)
        for kw in (dict(), dict(kind="ruler"), dict(kind="ruler", amount=2.0, freq=60.0), dict(kind="band", amount=2.0, freq=400.0, dur=0.3)):
            y = S.REGISTRY["twang"]["fn"](sr, **kw)
            assert y.ndim == 1 and np.all(np.isfinite(y)) and 1e-4 < np.max(np.abs(y)) < 1.5, (kw, sr)
            assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3 and abs(y.mean()) < 0.02, kw
            assert np.array_equal(y, S.REGISTRY["twang"]["fn"](sr, **kw))
        click = np.zeros(sr)
        click[10] = 1.0
        for x in (click, np.stack([click, 0.5 * click], axis=1)):
            for kw in (dict(), dict(decay=6.0, density=200, cutoff=8000.0, delay=0.02), dict(mix=1.0, density=10)):
                w = FX.REGISTRY["spring_reverb"]["fn"](x, sr, **kw)
                assert w.shape == x.shape and np.all(np.isfinite(w)) and np.max(np.abs(w)) < 1.5


def test_spring_reverb_chirps_and_decays():
    click = np.zeros(3 * SR)
    click[0] = 1.0
    w = FX.REGISTRY["spring_reverb"]["fn"](click, SR, mix=1.0, decay=2.0)
    r = [20 * np.log10(np.sqrt(np.mean(w[int(a * SR):int((a + 0.3) * SR)] ** 2)) + 1e-30) for a in (0.2, 1.2, 2.2)]
    # dispersion: in the first echo low frequencies arrive before high ones (allpass group delay rises with f)
    seg = w[:int(0.2 * SR)]
    def arrival(lo, hi):
        Y = np.fft.rfft(seg)
        fr = np.fft.rfftfreq(len(seg), 1 / SR)
        Y[(fr < lo) | (fr > hi)] = 0
        return np.argmax(np.abs(np.fft.irfft(Y))) / SR
    print(f"spring reverb: level per second {np.round(np.diff(r), 1)} dB (asked -30), arrival 300-800 Hz {arrival(300, 800) * 1e3:.1f} ms, 2-3.5 kHz {arrival(2000, 3500) * 1e3:.1f} ms")
    assert -40 < r[1] - r[0] < -20 and -40 < r[2] - r[1] < -20
    assert arrival(2000, 3500) > arrival(300, 800) + 0.002


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
