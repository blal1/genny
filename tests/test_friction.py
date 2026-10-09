"""Friction-driven sound and the Baschet family: measurable predictions of the models, then the catalog.

Run:  uv run --with pytest pytest tests/test_friction.py -q      (or: uv run python tests/test_friction.py)
"""
import math

import numpy as np

from genny import friction as X
from genny import fx as FX
from genny import instruments as I
from genny import sfx as S
from genny.analysis import brightness, loudness, sharpness
from genny.notes import midi_to_freq, note_to_midi

FS = X.FS
SR = 44100
F0 = 277.2            # the C#4 resonator measured in the cristal paper
INSTRUMENTS = ("cristal", "rubbed_glass", "bowed_bar", "bowed_sheet", "rod_bank", "whistling_blades", "tuning_fork", "coil_spring")
SFX = ("squeak", "brake_squeal", "rub", "stick_slip")
EFFECTS = ("sheet_radiator", "cone_radiator", "sympathetic")


def zc_freq(v, fs=FS):
    """Mean frequency from interpolated upward zero crossings."""
    v = v - v.mean()
    i = np.where((v[:-1] < 0) & (v[1:] >= 0))[0]
    t = i + (-v[i]) / (v[i + 1] - v[i])
    return fs * (len(t) - 1) / (t[-1] - t[0])


def peak_freq(y, sr, lo, hi):
    n = y.shape[0]
    sp = np.abs(np.fft.rfft(y * np.hanning(n), 4 * n))
    f = np.fft.rfftfreq(4 * n, 1 / sr)
    m = (f >= lo) & (f <= hi)
    return float(f[m][np.argmax(sp[m])])


def cents_off(y, f0, sr=SR):
    """Same measure as tests/test_instruments.py: autocorrelation peak next to the expected period."""
    y = y[int(0.05 * sr):int(0.05 * sr) + 16384]
    y = y - y.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(y * np.hanning(len(y)), 4 * len(y))) ** 2)
    p = sr / f0 * 4
    lo = int(p * 0.94)
    k = lo + int(np.argmax(ac[lo:int(p * 1.06) + 2]))
    k = k + 0.5 * (ac[k - 1] - ac[k + 1]) / (ac[k - 1] - 2 * ac[k] + ac[k + 1] + 1e-20)
    return 1200 * np.log2((sr * 4 / k) / f0)


# ---------------------------------------------------------------------------- modal recurrence
def test_modal_recurrence_is_exact():
    """One step of the recurrence has the eigenvalues exp((-xi w +- j wd) dt) and the static gain a/w^2."""
    f, xi, a, dt = 440.0, 0.01, 7.0, 1.0 / FS
    co = X._coefs([f], xi, a, dt)[:, 0]
    w = 2 * np.pi * f
    lam = np.linalg.eigvals(np.array([[co[0], co[1]], [co[2], co[3]]]))
    assert np.allclose(np.abs(lam), math.exp(-xi * w * dt), rtol=1e-12)
    assert np.allclose(np.abs(np.angle(lam)), w * math.sqrt(1 - xi * xi) * dt, rtol=1e-12)
    y, yd = 0.0, 0.0
    for _ in range(FS):                                   # 1 s under a unit force: settles on a/w^2
        y, yd = co[0] * y + co[1] * yd + co[4], co[2] * y + co[3] * yd + co[5]
    assert abs(y / (a / w ** 2) - 1) < 1e-6


# ---------------------------------------------------------------------------- cristal Baschet minimal model
def test_cristal_minimum_normal_force():
    """Paper: minimum normal force ~0.06 N at 0.1 m/s. No oscillation below it, oscillation above it."""
    fmin = X.fn_min(F0)
    assert 0.055 < fmin < 0.07, fmin
    growth = {}
    for k in (0.8, 1.3):
        v = X.friction_modal_resonator(F0, k * fmin, 0.1, 3.0)["vel"]
        growth[k] = np.sqrt(np.mean(v[-FS // 2:] ** 2)) / np.sqrt(np.mean(v[FS // 10:FS // 2] ** 2))
    assert growth[0.8] < 0.8 and growth[1.3] > 1.5, growth
    assert X.friction_modal_resonator(F0, 0.0, 0.1, 0.2)["stick"].sum() == 0      # no contact, no stick, no force
    assert np.abs(X.friction_modal_resonator(F0, 0.0, 0.1, 0.2)["force"]).max() < 1e-9
    # the model's explanation of the hard high register: the threshold rises with pitch
    assert X.fn_min(2 * F0) > 1.9 * fmin


def test_cristal_rise_time():
    """Paper: periodic after ~0.12 s for F_N = 1 N, finger 0.1 m/s (time to the first stick phase)."""
    st = X.friction_modal_resonator(F0, 1.0, 0.1, 1.0)["stick"]
    t_first = (np.argmax(st[10:]) + 10) / FS
    assert 0.08 < t_first < 0.16, t_first
    assert abs(X.rise_time(F0) - t_first) < 0.03, (X.rise_time(F0), t_first)       # LSA estimate (eq. 23)


def test_cristal_pitch_locked_to_mode():
    """Stick-slip frequency = the chosen mode within 1 %."""
    for f, fn in ((F0, 1.0), (130.8, 8 * X.fn_min(130.8)), (523.3, 8 * X.fn_min(523.3)), (1046.5, 8 * X.fn_min(1046.5))):
        v = X.friction_modal_resonator(f, fn, 0.1, 2.0)["vel"]
        assert abs(zc_freq(v[-FS // 2:]) / f - 1) < 0.01, (f, zc_freq(v[-FS // 2:]))
    # a second, weaker mode does not steal the pitch
    v = X.friction_modal_resonator([F0, 2.6 * F0], 1.0, 0.1, 1.5, mobility=np.array([21.0, 2.0]))["vel"]
    assert abs(peak_freq(v[-FS // 2:], FS, 50, 3000) / F0 - 1) < 0.01


def test_cristal_waveform_has_stick_and_slip_phases():
    """Helmholtz-like cycle: a stick phase at the finger speed, a slip phase swinging negative, a short
    forward slip (rod slightly faster than the finger) and a friction force bounded by mu_s F_N."""
    r = X.friction_modal_resonator(F0, 1.0, 0.1, 1.0)
    v, st, f = r["vel"][-FS // 5:], r["stick"][-FS // 5:], r["force"][-FS // 5:]
    periods = len(v) * F0 / FS
    frac = st.mean()
    assert 0.01 < frac < 0.6, frac                                  # both phases every cycle
    # every period sticks (the stick phase may be split in two by a one-sample slip at the force limit)
    assert 0.9 < np.sum(np.diff(st.astype(int)) == 1) / periods < 2.1
    assert np.allclose(v[st == 1], 0.1, atol=1e-9)                  # sticking: rod moves with the finger
    assert v.min() < -0.05                                          # slip: rod swings back
    assert 0.1 < v.max() < 0.12                                     # forward slip, slightly above the finger speed
    assert np.abs(f).max() <= X.MU_S * 1.0 + 1e-9
    # low mobility -> forward slip; a 5x lighter resonator (paper appendix B) sticks for longer
    st2 = X.friction_modal_resonator(200.4, 0.3, 0.1, 1.0, mobility=5 * 21.0)["stick"][-FS // 5:]
    st1 = X.friction_modal_resonator(200.4, 0.3, 0.1, 1.0)["stick"][-FS // 5:]
    assert st2.mean() > st1.mean()


def test_amplitude_grows_with_finger_speed():
    amp = [np.abs(X.friction_modal_resonator(F0, 1.0, u, 1.5)["vel"][-FS // 5:]).max() for u in (0.05, 0.1, 0.2)]
    assert amp[0] < amp[1] < amp[2], amp
    assert all(abs(a / u - 1) < 0.3 for a, u in zip(amp, (0.05, 0.1, 0.2))), amp     # peak velocity ~ finger speed
    lv = [loudness(I.render_note("cristal", 261.63, 0.8, SR, 0.9, finger_speed=u), SR) for u in (0.06, 0.12, 0.24)]
    assert lv[0] < lv[1] < lv[2], lv


def test_cristal_instrument_needs_pressure():
    quiet = I.render_note("cristal", 277.2, 0.8, SR, 0.9, pressure=0.02, radiator="cone")
    loud = I.render_note("cristal", 277.2, 0.8, SR, 0.9, pressure=1.0, radiator="cone")
    assert np.abs(quiet).max() < 0.1 * np.abs(loud).max()


# ---------------------------------------------------------------------------- Bilbao bow
def test_bow_pitch_flattening_and_attack():
    """Bowed mass, f0 = 200 Hz, v_B = 0.2: light bow plays the mode; more force -> longer period and
    shorter attack (Bilbao §4.3.1)."""
    out = {}
    for fb in (10.0, 50.0, 200.0):
        v = X.bowed_modal(200.0, 0.0, 1.0, fb, 0.2, 1.0)["vel"]
        out[fb] = (zc_freq(v[-FS // 4:]), np.argmax(np.abs(v) > 0.5 * np.abs(v).max()) / FS, np.abs(v[-FS // 10:]).max())
    assert abs(out[10.0][0] / 200.0 - 1) < 0.01, out
    assert out[10.0][0] > out[50.0][0] > out[200.0][0], out
    assert out[10.0][1] > out[50.0][1] > out[200.0][1], out
    assert 0.2 < out[10.0][2] < 0.3                                   # stick phase rides at the bow speed
    huge = X.bowed_modal(200.0, 0.0, 1.0, 1e9, 0.2, 0.2)["vel"]       # clamped to the uniqueness bound
    assert np.all(np.isfinite(huge)) and np.abs(huge).max() < 50


def test_bow_characteristic_is_passive():
    """q + g phi(q) - b = 0 is solved exactly: the residual vanishes and force * relative velocity <= 0."""
    r = X.bowed_modal([220.0, 606.0], 0.001, np.array([1.0, 0.5]), 60.0, 0.2, 0.5)
    q = r["vel"] - 0.2
    assert np.all(r["force"] * q <= 1e-12)                            # friction never pushes along the slip


# ---------------------------------------------------------------------------- elasto-plastic friction
def test_elastoplastic_steady_state_is_the_stribeck_curve():
    """Against an (almost) immovable body the steady friction force is f_c + (f_s - f_c) exp(-(v/v_s)^2)."""
    for v in (0.02, 0.1, 0.3):
        r = X.elastoplastic_friction(300.0, 0.05, 1e-6, 0.3, v, 0.3, sigma0=1e5)
        want = 0.3 * (0.197 + (0.975 - 0.197) * math.exp(-(v / 0.1) ** 2))
        assert abs(abs(r["force"][-1]) / want - 1) < 0.02, (v, r["force"][-1], want)


def test_elastoplastic_presliding_is_elastic():
    """Below breakaway the bristles act as a spring sigma0: no drift under a small constant pull."""
    s0 = 1e5
    r = X.elastoplastic_friction(300.0, 0.05, 1e-6, 1.0, 1e-4 * np.r_[np.ones(48), np.zeros(4752)], 0.1, sigma0=s0, sigma1=0.0)
    z = 1e-4 * 48 / FS                                     # displacement of the exciter while it moved
    assert z < 0.7 * 0.197 * 1.0 / s0                      # inside the elastic zone z_ba
    assert abs(abs(r["force"][-1]) / (s0 * z) - 1) < 0.02  # held force = sigma0 * z, and it stays


def test_rubbed_glass_locks_in_and_has_a_playable_window():
    f = 880.0
    ratios = np.array(X.GLASS)
    sing = X.elastoplastic_friction(f * ratios, 0.0006, 50.0 / ratios ** 2, 0.3, 0.1, 1.0, sigma0=1e5)["vel"]
    assert abs(peak_freq(sing[-FS // 2:], FS, 100, 6000) / f - 1) < 0.02
    assert np.abs(sing[-FS // 10:]).max() > 0.05
    fast = X.elastoplastic_friction(f * ratios, 0.0006, 50.0 / ratios ** 2, 0.3, 0.6, 1.0, sigma0=1e5)["vel"]
    assert np.abs(fast[-FS // 10:]).max() < 0.1 * np.abs(sing[-FS // 10:]).max()    # too fast: it stops singing


def test_creak_rate_follows_speed():
    """Slip events of the stick-slip driver come faster when the drag is faster."""
    def events(v):
        fr = np.array([math.sqrt(2e4 / 0.5) / (2 * np.pi), 420.0])
        r = X.elastoplastic_friction(fr, np.array([0.01, 0.02]), np.array([2.0, 3.0]), 8.0, v, 2.0, v_s=0.02, sigma0=2e6)
        f = np.abs(r["force"][FS // 2:])
        hi = f > 0.8 * f.max()
        return int(np.sum(hi[:-1] & ~hi[1:])) / 1.5
    slow, fast = events(0.005), events(0.02)
    assert 0 < slow < fast, (slow, fast)


def test_brake_squeals_at_the_end():
    y = S.REGISTRY["brake_squeal"]["fn"](SR)
    q = [np.sqrt(np.mean(c ** 2)) for c in np.array_split(y, 4)]
    assert q[3] > 3 * max(q[0], q[1]), q


# ---------------------------------------------------------------------------- Baschet rods and radiators
def test_rod_pitch_follows_inverse_square_length():
    assert abs(X.rod_frequency(0.5 * 0.70710678) / X.rod_frequency(0.5) - 2.0) < 1e-6      # x0.7071 per octave
    assert abs(X.rod_length(X.rod_frequency(0.31)) - 0.31) < 1e-9
    f = []
    for length in (0.20, 0.20 * 0.70710678):
        y = X.struck_rod(length, 1.0, SR)
        f.append(peak_freq(y, SR, 50, 600))
        assert abs(f[-1] / X.rod_frequency(length) - 1) < 0.005, (length, f[-1])
    assert abs(f[1] / f[0] - 2.0) < 0.01, f
    y = X.struck_rod(0.25, 1.0, SR)
    f1 = X.rod_frequency(0.25)
    assert abs(peak_freq(y, SR, 4 * f1, 9 * f1) / f1 - 6.267) < 0.03                       # cantilever second partial


def test_beam_shapes():
    assert abs(X.beam_shape("clamped", 0.0)).max() < 1e-9                 # no motion at the clamp
    assert np.allclose(np.abs(X.beam_shape("free", 0.0)), 2.0, atol=1e-3)  # free end: every mode at its maximum
    assert abs(X.beam_shape("free", 0.224)[0]) < 0.02                     # 22.4 % node of the first free mode


def test_whistling_blade_intervals():
    assert abs(X._whistle_ratio(1484.0) - 2 ** 0.5) < 1e-9                # augmented fourth on the longest blade
    assert abs(X._whistle_ratio(math.sqrt(1484.0 * 5350.0)) - 1.0) < 1e-9  # crossing at the middle blade
    assert abs(X._whistle_ratio(5350.0) - 2 ** (-1 / 3)) < 1e-9           # major third, roles swapped
    y = I.render_note("whistling_blades", math.sqrt(1484.0 * 5350.0), 1.5, SR, 0.9, second=1.0, decay=6.0)
    env = np.abs(y[int(0.1 * SR):int(1.4 * SR)])
    env = np.convolve(env, np.ones(441) / 441, "same")[441:-441]
    beat = peak_freq(env - env.mean(), SR, 1.0, 30.0)
    assert 4.0 <= beat <= 6.0, beat                                        # ombak-like beating of the middle blade


def test_sheet_radiator_is_level_dependent():
    t = np.arange(SR) / SR
    x = np.sin(2 * np.pi * 220 * t) * np.exp(-3 * t)
    def gain(amp, **kw):
        y = FX.REGISTRY["sheet_radiator"]["fn"](amp * x, SR, **kw)
        return np.sqrt(np.mean(y ** 2)) / np.sqrt(np.mean((amp * x) ** 2))
    assert gain(1.0, thickness=0.3) < 0.8 * gain(0.05, thickness=0.3)      # saturates when driven hard
    assert gain(1.0, thickness=0.3) < gain(1.0, thickness=0.8)             # thinner sheet saturates earlier
    def thunder(amp):                                                      # share of the low "thunder" modes
        y = FX.REGISTRY["sheet_radiator"]["fn"](amp * x, SR, thickness=0.5, mix=1.0)
        sp = np.abs(np.fft.rfft(y * np.hanning(y.size))) ** 2
        f = np.fft.rfftfreq(y.size, 1 / SR)
        return sp[f < 160].sum() / sp.sum()
    assert thunder(1.0) > 3 * thunder(0.05), (thunder(1.0), thunder(0.05))  # "from a pure sound to frequency packets"
    lo = FX.REGISTRY["sheet_radiator"]["fn"](x, SR, mix=1.0)
    assert np.sqrt(np.mean(lo[-SR // 4:] ** 2)) > 1e-4                     # it rings after the input has died


def test_sympathetic_strings_ring_on_their_tuning():
    t = np.arange(2 * SR) / SR
    x = np.sin(2 * np.pi * 220 * t) * (t < 0.3)
    hit = FX.REGISTRY["sympathetic"]["fn"](x, SR, tuning="A3 E4", mix=1.0)[int(0.5 * SR):]
    miss = FX.REGISTRY["sympathetic"]["fn"](x, SR, tuning="Bb3 F4", mix=1.0)[int(0.5 * SR):]
    assert np.sqrt(np.mean(hit ** 2)) > 5 * np.sqrt(np.mean(miss ** 2))
    assert abs(peak_freq(hit, SR, 100, 2000) - 220) < 2
    st = FX.REGISTRY["sympathetic"]["fn"](np.stack([x, x], axis=1), SR)
    assert st.shape == (x.size, 2)


def test_coil_spring_is_dispersive():
    """High partials travel faster: after the strike the echoes arrive as downward chirps, so in the gap
    before the first full echo the high band leads the low band."""
    y = I.render_note("coil_spring", 130.81, 0.5, SR, 0.9, dispersion=1.0, tone=0.0, hardness=1.0)
    seg = y[int(0.004 * SR):int(0.034 * SR)]
    def centroid(lo, hi):
        from genny import filters as F
        e = F.bandpass(seg, math.sqrt(lo * hi), SR, q=1.0) ** 2
        return float((e * np.arange(e.size)).sum() / e.sum()) / SR
    assert centroid(1500, 3000) < centroid(150, 400)


# ---------------------------------------------------------------------------- catalog
def test_registered():
    assert all(n in I.REGISTRY for n in INSTRUMENTS)
    assert all(n in S.REGISTRY for n in SFX)
    assert all(n in FX.REGISTRY for n in EFFECTS)


def test_instruments_shapes_and_levels():
    for name in INSTRUMENTS:
        lo, hi = (note_to_midi(n) for n in I.REGISTRY[name]["range"].split("-"))
        for m in (lo - 5, lo, (lo + hi) // 2, hi, hi + 7):
            for dur, vel, sr in ((0.03, 1.0, 44100), (0.1234, 0.3, 44100), (0.3333, 0.7, 22050), (1.0, 1.0, 48000)):
                y = I.render_note(name, midi_to_freq(m), dur, sr, vel)
                assert y.ndim == 1 and np.all(np.isfinite(y)), (name, m, dur)
                assert 1e-4 < np.max(np.abs(y)) < 1.5, (name, m, dur, float(np.max(np.abs(y))))
                assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3, (name, "starts or ends on a step")
                assert abs(y.mean()) < 0.02, (name, m, "dc")
        lv = [loudness(I.render_note(name, midi_to_freq(m), 0.5, SR, 0.9), SR, window=0.15)
              for m in (lo + (hi - lo) // 4, (lo + hi) // 2, hi - (hi - lo) // 4)]
        assert abs(sum(lv) / 3 + 12.0) < 6.0, (name, [round(v, 1) for v in lv])


def test_instruments_in_tune():
    for name in INSTRUMENTS:
        lo, hi = (note_to_midi(n) for n in I.REGISTRY[name]["range"].split("-"))
        for m in (lo, (lo + hi) // 2, hi):
            c = cents_off(I.render_note(name, midi_to_freq(m), 0.6, SR, 0.9), midi_to_freq(m))
            assert abs(c) < 10, (name, m, round(float(c), 1))


def test_instruments_not_piercing():
    for name in INSTRUMENTS:
        lo, hi = (note_to_midi(n) for n in I.REGISTRY[name]["range"].split("-"))
        for m in sorted({min((lo + hi) // 2, 96), min(hi, 96)}):
            f = midi_to_freq(m)
            y = I.render_note(name, f, 0.5, SR, 0.9)
            pure = sharpness(I.render_note("sine", f, 0.5, SR, 0.9), SR)
            s, b = sharpness(y, SR), brightness(y, SR)
            assert s < max(pure + 1.0, 1.5) and s < 2.4, (name, m, round(s, 2), round(pure, 2))
            assert b["above_5k"] < 0.03, (name, m, round(b["above_5k"], 3))


def _check(y, what):
    assert np.all(np.isfinite(y)), what
    assert 1e-4 < np.max(np.abs(y)) < 1.5, (what, float(np.max(np.abs(y))))
    assert np.max(np.abs(y[0])) < 1e-3 and np.max(np.abs(y[-1])) < 1e-3, (what, "edges")
    assert abs(float(np.mean(y))) < 0.02, (what, "dc")


def test_smoke_every_name_two_rates():
    variants = {"cristal": [{"radiator": "cone"}, {"radiator": "balloon", "wetness": 0.3, "bright": 2.0}],
                "rubbed_glass": [{"roughness": 1.0, "pressure": 2.0}], "bowed_bar": [{"material": "wood", "position": 0.15, "force": 2.5}],
                "bowed_sheet": [{"thickness": 0.3, "ring": 1.0}],
                "rod_bank": [{"rod": "clamped", "radiator": "cone"}, {"rod": "weighted", "radiator": "balloon"}, {"rod": "mushroom", "radiator": "metal"}],
                "whistling_blades": [{"spin": 3.0, "second": 1.0}], "tuning_fork": [{"radiator": "cone", "swell": 0.3, "hardness": 1.0}, {"radiator": "none"}],
                "coil_spring": [{"excite": "scrape"}, {"dispersion": 1.0, "tone": 0.0}],
                "squeak": [{"surface": "shoe"}, {"surface": "chalk", "pressure": 3.0}], "rub": [{"surface": "glass"}, {"surface": "wood"}],
                "stick_slip": [{"speed": 0.003}, {"speed": 0.2, "stiffness": 1e5}], "brake_squeal": [{"speed": 4.0, "dur": 0.8}],
                "sheet_radiator": [{"thickness": 0.3, "drive": 6.0, "mix": 1.0}], "cone_radiator": [{"material": "metal"}, {"material": "balloon"}],
                "sympathetic": [{"kind": "rod", "tuning": ["C3", 196.0]}]}
    for sr in (22050, 48000):
        for name in INSTRUMENTS:
            lo, hi = (note_to_midi(n) for n in I.REGISTRY[name]["range"].split("-"))
            for kw in [{}] + variants.get(name, []):
                y = I.render_note(name, midi_to_freq((lo + hi) // 2), 0.4, sr, 0.8, **kw)
                _check(y, (name, sr, kw))
                assert np.array_equal(y, I.render_note(name, midi_to_freq((lo + hi) // 2), 0.4, sr, 0.8, **kw)), (name, "not deterministic")
        for name in SFX:
            for kw in [{}] + variants.get(name, []):
                y = S.REGISTRY[name]["fn"](sr, **kw)
                _check(y, (name, sr, kw))
                assert np.array_equal(y, S.REGISTRY[name]["fn"](sr, **kw)), (name, "not deterministic")
        t = np.arange(sr) / sr
        x = 0.6 * np.sin(2 * np.pi * 196 * t) * np.sin(np.pi * t) ** 2
        for name in EFFECTS:
            for kw in [{}] + variants.get(name, []):
                for sig in (x, np.stack([x, 0.5 * x], axis=1)):
                    y = FX.REGISTRY[name]["fn"](sig, sr, **kw)
                    assert y.shape == sig.shape and np.all(np.isfinite(y)), (name, sr, kw)
                    assert 1e-4 < np.max(np.abs(y)) < 1.5, (name, sr, kw, float(np.max(np.abs(y))))


def test_fast_enough():
    import time
    for name in INSTRUMENTS:
        lo, hi = (note_to_midi(n) for n in I.REGISTRY[name]["range"].split("-"))
        I.render_note(name, midi_to_freq((lo + hi) // 2), 0.2, SR, 0.8)             # JIT warm-up
        t = time.perf_counter()
        I.render_note(name, midi_to_freq((lo + hi) // 2), 1.0, SR, 0.8)
        assert time.perf_counter() - t < 2.0, name


if __name__ == "__main__":
    for k, fn in sorted(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
