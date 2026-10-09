"""genny.tubes: the Webster tube, the reed, the vocal tract and everything registered from them.

Run:  uv run --with pytest pytest tests/test_tubes.py -q     (or: uv run python tests/test_tubes.py)
"""
import math
import time

import numpy as np
from scipy.optimize import brentq

from genny import fx as FX
from genny import instruments as I
from genny import sfx as SFX
from genny import tubes as T
from genny.analysis import brightness, loudness, sharpness
from genny.notes import midi_to_freq, note_to_midi

SR = T.FS
C = T.C_AIR
INSTRUMENTS = ("reed_tube", "reed_cone", "bottle", "vowel_tube", "piston_pipe", "pvc_pipe")


def _peaks(y, fmax, sr=SR, floor=0.03):
    """Spectral peak frequencies of a decayed response (zero-padded FFT, parabolic refinement)."""
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y)), 4 * len(y)))
    f = np.fft.rfftfreq(4 * len(y), 1 / sr)
    m = Y[:int(fmax / f[1])]
    i = np.nonzero((m[1:-1] > m[:-2]) & (m[1:-1] >= m[2:]) & (m[1:-1] > floor * m.max()))[0] + 1
    d = 0.5 * (m[i - 1] - m[i + 1]) / (m[i - 1] - 2 * m[i] + m[i + 1])
    return (i + d) * f[1]


def _harmonics(y, f0, n=6, sr=SR):
    y = y[int(0.2 * sr):int(0.2 * sr) + 16384]
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y)), 8 * len(y)))
    f = np.fft.rfftfreq(8 * len(y), 1 / sr)
    h = np.array([Y[(f > k * f0 * 0.97) & (f < k * f0 * 1.03)].max() for k in range(1, n + 1)])
    return 20 * np.log10(h / h[0] + 1e-12)


def _impulse(n):
    u = np.zeros(n)
    u[0] = 1.0
    return u


# ------------------------------------------------------------------------------------------ the tube
def test_cylinder_closed_open():
    """Ideal closed-open cylinder: odd multiples of c/4L, in the time-domain kernel and in the response."""
    L = 0.5
    t = T.WebsterTube(L, 1.0, right="open", loss=0.5)
    want = C / (4 * L) * (2 * np.arange(6) + 1)
    got = _peaks(t.run(_impulse(2 * SR))["p_in"], 2000.0)[:6]
    assert np.max(np.abs(got / want - 1)) < 2e-3, (got, want)
    assert np.max(np.abs(t.resonances(2000.0)[:6] / want - 1)) < 2e-3
    print("closed-open  ", np.round(got, 1), "analytic", np.round(want, 1))


def test_cylinder_open_open():
    """Both ends ideally open: all multiples of c/2L."""
    L = 0.5
    t = T.WebsterTube(L, 1.0, left="open", right="open", loss=0.5)
    want = C / (2 * L) * np.arange(1, 6)
    got = _peaks(t.run(_impulse(2 * SR), src_pos=0.13, pick_pos=0.29)["p_pick"], 1850.0)[:5]
    assert got.size == 5 and np.max(np.abs(got / want - 1)) < 2e-3, (got, want)
    print("open-open    ", np.round(got, 1), "analytic", np.round(want, 1))


def test_cone_harmonic_series():
    """Cone closed at its small end (x0 from the apex), open at the wide end: tan(kL) = -k x0, which tends to
    the full harmonic series n c/2L as x0 -> 0 (not the odd series of the cylinder)."""
    L, x0 = 1.0, 0.04
    t = T.WebsterTube(L, lambda x: (1 + x * L / x0) ** 2, right="open", radius_m=0.002, sr=4 * SR)
    want = np.array([brentq(lambda k: math.tan(k * L) + k * x0, (n - 0.5) * math.pi / L + 1e-6, n * math.pi / L)
                     for n in range(1, 6)]) * C / (2 * math.pi)
    got = t.resonances(1000.0)[:5]
    assert np.max(np.abs(got / want - 1)) < 0.01, (got, want)
    assert np.max(np.abs(got / (C / (2 * L) * np.arange(1, 6)) - 1)) < 0.04       # nearly harmonic 1:2:3:4:5
    print("cone         ", np.round(got, 1), "analytic", np.round(want, 1), "c/2L =", C / (2 * L))


def test_radiation_end_correction():
    """The passive radiating end (9.9)-(9.11) acts as the textbook end correction: 0.6133 a unflanged,
    0.8216 a flanged, and it is lossy (finite peak)."""
    L, a = 0.6, 0.008
    for kind, k in (("unflanged", T.UNFLANGED), ("flanged", T.FLANGED)):
        t = T.WebsterTube(L, 1.0, radius_m=a, right=kind)
        f1 = t.resonances(400.0)[0]
        assert abs(f1 / (C / (4 * (L + k * a))) - 1) < 1.5e-3, (kind, f1)
        assert np.isfinite(np.abs(t.response([f1])[0][0]))
        print(f"{kind:13s} f1 = {f1:.2f} Hz, c/4(L + {k} a) = {C / (4 * (L + k * a)):.2f}")


def test_kernel_matches_response():
    """Parity: FFT of the simulated impulse response == the closed-form response of the scheme, with loss, wall,
    a flared bore and a half-open side hole."""
    t = T.WebsterTube(0.5, lambda x: (1 + 2 * x) ** 2, radius_m=0.009, right="flanged", loss=2.0, wall=1.0,
                      holes=[(0.6, 0.003, 0.004)])
    r = t.run(_impulse(4 * SR), hole_state=[0.5])
    f = np.fft.rfftfreq(4 * SR, 1 / SR)[4:12000]
    Z, Hu, Hp = t.response(f, hole_state=[0.5])
    for sim, ref in ((r["p_in"], Z), (r["u_out"], Hu), (r["p_out"], Hp)):
        err = np.max(np.abs(np.fft.rfft(sim)[4:12000] - ref) / np.abs(ref))
        assert err < 1e-8, err


def test_stable_for_any_bore():
    """lambda <= 1 is the whole stability condition: a wildly jagged bore, lossless ends, must not grow; the
    same bore with a radiating end and losses must decay."""
    rng = np.random.default_rng(3)
    S = np.exp(rng.uniform(-2.0, 2.0, 40))
    u = rng.standard_normal(200)
    y = T.WebsterTube(0.7, S, right="open").run(u, n=3 * SR)["p_in"]
    assert np.all(np.isfinite(y))
    e = [float(np.sum(y[i * SR:(i + 1) * SR] ** 2)) for i in range(3)]
    assert abs(e[2] / e[1] - 1) < 0.05, e                    # lossless: energy neither grows nor leaks
    y = T.WebsterTube(0.7, S, right="unflanged", loss=1.0, wall=1.0).run(u, n=3 * SR)["p_out"]
    assert np.all(np.isfinite(y)) and np.sum(y[2 * SR:] ** 2) < 1e-6 * np.sum(y[:SR] ** 2)
    print("jagged bore: energy per second (lossless)", ["%.4g" % v for v in e])


def test_wall_broadens_f1():
    """Yielding walls (9.18) with sigma0 = 8125, omega0 = 0, eps = 2718: the first formant gets wider
    (Bilbao Fig 9.7); a lossless wall could not do that."""
    bw = []
    for wall in (0.0, 1.0):
        t = T._tract("a", wall=wall)
        f = np.arange(300.0, 1000.0, 0.5)
        m = np.abs(t.response(f)[1])
        above = f[m > m.max() / math.sqrt(2)]
        bw.append(above[-1] - above[0])
    assert bw[1] > 1.2 * bw[0], bw
    print("F1 bandwidth of /a/: rigid %.0f Hz, yielding wall %.0f Hz" % tuple(bw))


def test_tonehole():
    """Side hole (9.40): closed it barely matters, fully open it shortens the tube; the state is continuous."""
    L, a = 0.5, 0.0074
    f = [T.WebsterTube(L, 1.0, radius_m=a, holes=[(0.5, 0.006, 0.004)]).resonances(700.0, hole_state=[p])[0]
         for p in (0.0, 0.25, 0.5, 1.0)]
    f0 = T.WebsterTube(L, 1.0, radius_m=a).resonances(700.0)[0]
    assert abs(f[0] / f0 - 1) < 0.01
    assert f[0] < f[1] < f[2] < f[3] and 1.5 < f[3] / f0 < 2.0, (f0, f)
    print("tonehole at L/2: no hole %.1f Hz, closed %.1f, half %.1f, open %.1f (half-length tube %.1f)"
          % (f0, f[0], f[2], f[3], 2 * f0))


# ------------------------------------------------------------------------------------------ reed
def _blow(pm, n=2 * SR, **kw):
    t = T.WebsterTube(0.664, 1.0, radius_m=math.sqrt(T.REED_S0 / math.pi), left="reed", right="unflanged", **kw)
    p = np.full(n, pm) * T._edge(n, 200, 0)
    return t, t.run(pm=p, reed=T.reed_params())


def test_reed_threshold_and_beating():
    """Bilbao Fig 9.12/9.13 with his reed and bore: below the threshold (pm ~ 0.0121) the note dies, above it
    grows into a steady tone; the reed starts hitting the lay (y < -1) between 0.0134 and 0.0201."""
    rms = lambda y, a, b: float(np.sqrt(np.mean(y[int(a * SR):int(b * SR)] ** 2)))   # noqa: E731
    _, lo = _blow(0.0105)
    _, hi = _blow(0.0134)
    _, beat = _blow(0.0201)
    assert rms(lo["p_out"], 1.8, 2.0) < 0.01 * rms(lo["p_out"], 0.0, 0.1)
    assert rms(hi["p_out"], 1.8, 2.0) > 3 * rms(hi["p_out"], 0.0, 0.1) and rms(hi["p_out"], 1.8, 2.0) > 1e-4
    assert hi["y"][SR:].min() > -1.0 and beat["y"][SR:].min() < -1.05
    print("reed: pm 0.0105 -> rms %.1e (dies), 0.0134 -> %.1e, 0.0201 -> %.1e; reed minimum %.2f / %.2f (beats)"
          % (rms(lo["p_out"], 1.8, 2.0), rms(hi["p_out"], 1.8, 2.0), rms(beat["p_out"], 1.8, 2.0),
             hi["y"][SR:].min(), beat["y"][SR:].min()))


def test_reed_loading_is_a_length_correction():
    """The reed's own flow is a compliance worth c S Q / omega0^2 = 9.6 mm of tube: the played pitch sits on
    the resonance of the bore loaded with it, not on the bare bore's."""
    rp = T.reed_params()
    dl = C * rp[2] * rp[0] / rp[3] ** 2
    assert abs(dl - 0.0096) < 2e-4
    t, r = _blow(0.027, n=SR, loss=1.0)
    y = r["p_out"][SR // 2:]
    k = np.argmax(np.abs(np.fft.rfft(y * np.hanning(y.size), 16 * y.size))[:16 * 300 * y.size // SR])
    played = k * SR / (16 * y.size)
    bare, loaded = t.resonances(300.0)[0], t.resonances(300.0, comp=dl / C)[0]
    assert abs(1200 * math.log2(played / loaded)) < 6 < abs(1200 * math.log2(played / bare))
    print("reed: plays %.2f Hz; bore %.2f Hz (%.0f cents off), bore + reed compliance %.2f Hz (%.1f cents)"
          % (played, bare, 1200 * math.log2(played / bare), loaded, 1200 * math.log2(played / loaded)))


def test_odd_and_even_harmonics():
    """Cylinder + reed = odd harmonics (clarinet low register); the cone brings the even ones in."""
    for m in (50, 55, 60):
        f = midi_to_freq(m)
        cyl = _harmonics(I.render_note("reed_tube", f, 0.6, SR, 0.9), f)
        con = _harmonics(I.render_note("reed_cone", f, 0.6, SR, 0.9), f)
        assert cyl[1] < -25 and cyl[3] < -20 and cyl[2] > -8, (m, cyl)
        assert con[1] > cyl[1] + 8 and con[1] > -26, (m, con)
        print("midi %d  H2..H5 re H1 (dB): cylinder %s   cone %s" % (m, np.round(cyl[1:5]), np.round(con[1:5])))


def test_register_hole_overblows():
    """Opening the register hole takes the cylinder off its fundamental onto the twelfth (3 f0)."""
    f = midi_to_freq(53)
    for reg, want in ((0.0, 1.0), (1.0, 3.0)):
        y = I.render_note("reed_tube", f, 0.6, SR, 0.9, register=reg)
        y = y[int(0.3 * SR):int(0.3 * SR) + 8192]
        k = np.argmax(np.abs(np.fft.rfft(y * np.hanning(y.size), 8 * y.size)))
        assert abs(k * SR / (8 * y.size) / f / want - 1) < 0.06, (reg, k * SR / (8 * y.size) / f)


# ------------------------------------------------------------------------------------------ voice
def test_vowel_formants():
    """F1/F2 of the tract shapes vs Peterson & Barney (1952) adult-male averages, +-15 %. /a/ is Bilbao's table
    (Fant's Russian /a/, usually quoted near 650 / 1080 Hz)."""
    pb = {"a": (730, 1090), "i": (270, 2290), "u": (300, 870)}
    for v, want in pb.items():
        tr = T._tract(v)
        got = tr.resonances(4000.0, kind="u")[:2]
        sim = _peaks(tr.run(_impulse(SR))["u_out"], 4000.0)[:2]            # the kernel says the same
        assert np.max(np.abs(got / np.array(want) - 1)) < 0.15, (v, got)
        assert np.max(np.abs(sim / got - 1)) < 0.01, (v, sim, got)
        print("/%s/  F1 %.0f  F2 %.0f   (Peterson-Barney %d, %d)" % (v, got[0], got[1], want[0], want[1]))
    fa = T._tract("a").resonances(4000.0, kind="u")[:2]
    assert abs(fa[0] / 650 - 1) < 0.05 and abs(fa[1] / 1080 - 1) < 0.05


def test_vowel_morph():
    """Time-varying tract (9.23): gliding /a/ -> /i/ moves F2 up by about an octave and stays finite."""
    y = I.render_note("vowel_tube", 110.0, 1.2, SR, 0.9, vowel="a", vowel2="i", vibrato=0.0, breath=0.0)
    assert np.all(np.isfinite(y))

    def f2_band(seg):
        Y = np.abs(np.fft.rfft(seg * np.hanning(seg.size))) ** 2
        f = np.fft.rfftfreq(seg.size, 1 / SR)
        return Y[(f > 1800) & (f < 2600)].sum() / Y[(f > 900) & (f < 1300)].sum()

    a, b = f2_band(y[int(0.05 * SR):int(0.25 * SR)]), f2_band(y[int(1.0 * SR):int(1.2 * SR)])
    assert b > 10 * a, (a, b)
    print("morph a->i: energy(1.8-2.6 kHz)/energy(0.9-1.3 kHz) %.3f -> %.2f" % (a, b))


# ------------------------------------------------------------------------------------------ fx / pipes
def test_tube_fx_resonances():
    """Noise through the duct comes out with the pipe's series: odd quarter-wave (closed-open) or half-wave
    (open-open), each with its end correction(s)."""
    x = np.random.default_rng(0).standard_normal(4 * SR)
    L, a = 1.0, 0.03
    f = np.fft.rfftfreq(x.size, 1 / SR)
    for ends, want in (("closed-open", C / (4 * (L + 0.6133 * a)) * np.array([1, 3, 5])),
                       ("open-open", C / (2 * (L + 2 * 0.6133 * a)) * np.array([1, 2, 3]))):
        Y = np.abs(np.fft.rfft(FX.REGISTRY["tube"]["fn"](x, SR, length_m=L, radius_m=a, ends=ends, loss=1.0))) ** 2
        Y = np.convolve(Y, np.ones(9) / 9, mode="same")
        for w in want:
            band = (f > w * 0.9) & (f < w * 1.1)
            got = f[band][np.argmax(Y[band])]
            assert abs(got / w - 1) < 0.012, (ends, got, w)
            assert Y[band].max() > 8 * np.median(Y[(f > w * 1.25) & (f < w * 1.6)]), (ends, w)
        print("fx tube %-12s peaks near %s Hz" % (ends, np.round(want, 1)))


def test_piston_pipe_needs_a_matching_pipe():
    """Baschet: the air column amplifies only when the driven note matches the pipe."""
    f = midi_to_freq(43)
    on = loudness(T.piston_pipe(f, 0.8, SR, 0.9, rod=0.0), SR)
    # same drive, pipe a fourth away: compare the un-normalised radiated pressure
    tb0, tb1 = T._pipe(round(f, 4), 0.1, 1.0), T._pipe(round(f * 2 ** (5 / 12), 4), 0.1, 1.0)
    u = np.sin(2 * np.pi * f * np.arange(SR) / SR)
    p0, p1 = tb0.run(u)["p_out"], tb1.run(u)["p_out"]
    gain = 20 * math.log10(np.std(p0[SR // 2:]) / np.std(p1[SR // 2:]))
    assert gain > 12 and np.isfinite(on), gain
    print("piston pipe: tuned pipe radiates %.1f dB more than one a fourth off" % gain)


def test_formant_tract_and_stereo():
    x = np.random.default_rng(1).standard_normal((48000, 2)) * 0.2
    for name, kw in (("formant_tract", dict(vowel="i")), ("tube", dict(length_m=0.4, ends="closed-closed")),
                     ("tube", dict(length_m=3.0, radius_m=0.2, ends="open-open", mix=0.5))):
        for sr in (22050, 48000):
            xs = x[:sr]
            y = FX.REGISTRY[name]["fn"](xs, sr, **kw)
            assert y.shape == xs.shape and np.all(np.isfinite(y)) and 1e-4 < np.max(np.abs(y)) < 4.0, (name, kw)
            m = FX.REGISTRY[name]["fn"](xs[:, 0], sr, **kw)
            assert m.shape == (sr,) and np.allclose(m, y[:, 0])
    a = FX.REGISTRY["formant_tract"]["fn"](x[:SR, 0], SR, vowel="a")
    Y = np.convolve(np.abs(np.fft.rfft(a)) ** 2, np.ones(41) / 41, mode="same")
    f = np.fft.rfftfreq(SR, 1 / SR)
    band = (f > 400) & (f < 850)
    assert abs(f[band][np.argmax(Y[band])] / 649 - 1) < 0.12            # /a/ F1 imposed on white noise


# ------------------------------------------------------------------------------------------ catalog contract
def _span(name):
    return tuple(note_to_midi(n) for n in I.REGISTRY[name]["range"].split("-"))


def test_registered():
    for name in INSTRUMENTS:
        assert name in I.REGISTRY, name
    assert "pipe_blow" in SFX.REGISTRY and "tube" in FX.REGISTRY and "formant_tract" in FX.REGISTRY


def test_shapes_and_levels():
    for name in INSTRUMENTS:
        lo, hi = _span(name)
        for m in (lo - 5, lo, (lo + hi) // 2, hi, hi + 7):
            for dur, vel, sr in ((0.03, 1.0, 44100), (0.1234, 0.3, 44100), (0.3333, 0.7, 22050), (1.0, 1.0, 48000)):
                y = I.render_note(name, midi_to_freq(m), dur, sr, vel)
                assert y.ndim == 1 and np.all(np.isfinite(y)), (name, m, dur)
                assert 1e-4 < np.max(np.abs(y)) < 1.5, (name, m, dur, float(np.max(np.abs(y))))
                assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3, (name, "starts/ends on a step")
                assert abs(np.mean(y)) < 0.02 * np.max(np.abs(y)) + 1e-4, (name, m, dur, "DC")
                assert y.size >= int(dur * sr)


def test_tuning():
    worst = {}
    for name in INSTRUMENTS:
        lo, hi = _span(name)
        for m in (lo, (lo + hi) // 2, hi):
            c = T._cents_off(I.render_note(name, midi_to_freq(m), 0.6, SR, 0.9), midi_to_freq(m))
            worst[name] = max(worst.get(name, 0.0), abs(float(c)))
            assert abs(c) < 10, (name, m, round(float(c), 1))
    print("tuning, worst of bottom/middle/top (cents):", {k: round(v, 1) for k, v in worst.items()})


def test_reeds_in_tune_on_every_semitone_and_dynamic():
    """The measured compensation holds between its calibration points and across mouth pressure."""
    worst = 0.0
    for name in ("reed_tube", "reed_cone"):
        lo, hi = _span(name)
        for m in range(lo, hi + 1):
            for vel in (0.3, 0.6, 1.0):
                c = T._cents_off(I.render_note(name, midi_to_freq(m), 0.6, SR, vel), midi_to_freq(m))
                worst = max(worst, abs(float(c)))
                assert abs(c) < 10, (name, m, vel, round(float(c), 1))
    print("reeds, every semitone x vel 0.3/0.6/1.0: worst %.1f cents" % worst)


def test_not_piercing():
    for name in INSTRUMENTS:
        lo, hi = _span(name)
        for m in sorted({min((lo + hi) // 2, 96), min(hi, 96)}):
            f = midi_to_freq(m)
            y = I.render_note(name, f, 0.5, SR, 0.9)
            pure = sharpness(I.render_note("sine", f, 0.5, SR, 0.9), SR)
            s, b = sharpness(y, SR), brightness(y, SR)
            assert s < max(pure + 1.0, 1.5) and s < 2.4, (name, m, round(s, 2), round(pure, 2))
            assert b["above_5k"] < 0.03, (name, m, round(b["above_5k"], 3))


def test_level():
    out = {}
    for name in INSTRUMENTS:
        lo, hi = _span(name)
        lv = [loudness(I.render_note(name, midi_to_freq(m), 0.5, SR, 0.9), SR, window=0.15)
              for m in (lo + (hi - lo) // 4, (lo + hi) // 2, hi - (hi - lo) // 4)]
        out[name] = round(sum(lv) / 3, 1)
        assert abs(sum(lv) / 3 + 12.0) < 6.0, (name, lv)
    print("K-weighted level at vel 0.9 (dB):", out)


def test_smoke_everything_two_rates():
    for sr in (22050, 48000):
        for name in INSTRUMENTS:
            lo, hi = _span(name)
            y = I.render_note(name, midi_to_freq((lo + hi) // 2), 0.4, sr, 0.8)
            assert np.all(np.isfinite(y)) and 1e-4 < np.max(np.abs(y)) < 1.5, (name, sr)
            assert np.array_equal(y, I.render_note(name, midi_to_freq((lo + hi) // 2), 0.4, sr, 0.8)), (name, "not deterministic")
        for kw in (dict(), dict(stopped=False, length_m=1.5, radius_m=0.03, seed=4)):
            y = SFX.REGISTRY["pipe_blow"]["fn"](sr, **kw)
            assert np.all(np.isfinite(y)) and 1e-4 < np.max(np.abs(y)) < 1.5 and abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3
            assert abs(np.mean(y)) < 1e-3 and np.array_equal(y, SFX.REGISTRY["pipe_blow"]["fn"](sr, **kw))


def test_params_in_range_stay_finite():
    f = midi_to_freq(57)
    cases = [("reed_tube", dict(pressure=0.5)), ("reed_tube", dict(pressure=1.3, embouchure=0.5, breath=0.5)),
             ("reed_tube", dict(embouchure=1.5, hole=1.0, vibrato=1.0, tongue=0.0)), ("reed_cone", dict(cone=0.0)),
             ("reed_cone", dict(cone=0.5, register=1.0, bright=4.0)), ("bottle", dict(noise=1.5, vibrato=1.0)),
             ("vowel_tube", dict(vowel="u", vowel2="e", morph=0.5, size=0.6, tense=0.8, breath=1.0)),
             ("vowel_tube", dict(vowel="o", size=2.0, vibrato=2.0)), ("piston_pipe", dict(diameter=0.36, rod=1.0, detune=200)),
             ("pvc_pipe", dict(radius=0.08, hardness=1.0, loss=10.0)), ("pvc_pipe", dict(radius=0.004, hardness=0.0, loss=0.5))]
    for name, kw in cases:
        y = I.render_note(name, f, 0.4, SR, 1.0, **kw)
        assert np.all(np.isfinite(y)) and np.max(np.abs(y)) < 2.0, (name, kw, float(np.max(np.abs(y))))


def test_speed():
    I.render_note("reed_cone", 233.0, 0.2, SR, 0.9)             # JIT + tuning cache warm
    for name, f in (("reed_cone", 233.0), ("reed_tube", 147.0), ("vowel_tube", 110.0), ("piston_pipe", 65.4), ("pvc_pipe", 65.4)):
        t0 = time.perf_counter()
        I.render_note(name, f, 1.0, SR, 0.9)
        dt = time.perf_counter() - t0
        assert dt < 2.0, (name, dt)
    x = np.random.default_rng(0).standard_normal(SR)
    t0 = time.perf_counter()
    FX.REGISTRY["tube"]["fn"](x, SR, length_m=10.0)
    assert time.perf_counter() - t0 < 5.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
