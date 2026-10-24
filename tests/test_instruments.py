"""Every instrument and drum renders cleanly, in tune, and without piercing highs.

Run:  uv run python tests/test_instruments.py     (plain asserts, no test framework needed)
"""
import numpy as np

from genny import drums as D
from genny import instruments as I
from genny.analysis import brightness, sharpness
from genny.notes import midi_to_freq, note_to_midi

SR = 44100


def cents_off(y, f0):
    """Pitch error of a rendered note, from the autocorrelation peak next to the expected period."""
    y = y[int(0.05 * SR):int(0.05 * SR) + 16384]
    y = y - y.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(y * np.hanning(len(y)), 4 * len(y))) ** 2)
    p = SR / f0 * 4
    lo = int(p * 0.94)
    k = lo + int(np.argmax(ac[lo:int(p * 1.06) + 2]))
    k = k + 0.5 * (ac[k - 1] - ac[k + 1]) / (ac[k - 1] - 2 * ac[k] + ac[k + 1] + 1e-20)
    return 1200 * np.log2((SR * 4 / k) / f0)


def test_shapes_and_levels():
    for name, e in I.REGISTRY.items():
        lo, hi = (note_to_midi(n) for n in e["range"].split("-"))
        for m in (lo - 5, lo, (lo + hi) // 2, hi, hi + 7):
            for dur, vel, sr in ((0.03, 1.0, 44100), (0.1234, 0.3, 44100), (0.3333, 0.7, 22050), (1.0, 1.0, 48000)):
                y = I.render_note(name, midi_to_freq(m), dur, sr, vel)
                assert y.ndim == 1 and np.all(np.isfinite(y)), (name, m, dur)
                assert 1e-4 < np.max(np.abs(y)) < 4.0, (name, m, dur, float(np.max(np.abs(y))))
                assert abs(y[-1]) < 1e-3, (name, "ends on a step")
    for name in D.REGISTRY:
        for vel, sr in ((1.0, 44100), (0.2, 22050)):
            y = D.render_drum(name, sr, vel)
            assert y.ndim == 1 and np.all(np.isfinite(y)) and 1e-4 < np.max(np.abs(y)) < 2.0, name


def test_tuning():
    # ensembles and beating / inharmonic voices are detuned on purpose; everything else must be within 10 cents
    loose = {"choir", "pad", "strings", "tubular_bell", "singing_bowl", "gamelan", "mandolin", "acid", "wobble", "didgeridoo", "piano"}
    # wound bass strings: the fundamental is exact but the stiff-string stretch (Rauhala dispersion) puts
    # partials 3-4 4-10 cents sharp on E1, and the autocorrelation pitch follows them over a weak fundamental
    loose |= {"upright_bass", "finger_bass", "slap_bass"}
    # church and china bells are tuned on one stated partial of a strongly inharmonic spectrum, which an
    # autocorrelation pitch cannot follow; tests/test_foley.py checks that partial against the mode table
    inharmonic = {"church_bell", "china_bell"}
    # the chips' noise channels take a note only to choose a noise rate (tests/test_chiptune.py checks those rates)
    inharmonic |= {"nes_noise", "gb_noise", "sn_noise"}
    # the bridge spring of string_bridge_plate stretches its partials by up to 4 %: tuned on its fundamental,
    # which tests/test_string_bridge_plate.py checks
    inharmonic |= {"string_bridge_plate"}
    for name, e in I.REGISTRY.items():
        if name in inharmonic:
            continue
        lo, hi = (note_to_midi(n) for n in e["range"].split("-"))
        for m in (lo, (lo + hi) // 2, hi):
            c = cents_off(I.render_note(name, midi_to_freq(m), 0.6, SR, 0.9), midi_to_freq(m))
            assert abs(c) < (25 if name in loose else 10), (name, m, round(float(c), 1))


def test_not_piercing():
    """In its own register no instrument may be much sharper than a pure tone of the same note, nor carry
    real energy above 5 kHz. (`synth` is the raw building block: what you set is what you get.)"""
    for name, e in I.REGISTRY.items():
        if name == "synth":
            continue
        lo, hi = (note_to_midi(n) for n in e["range"].split("-"))
        for m in sorted({min((lo + hi) // 2, 96), min(hi, 96)}):   # up to C7; above that even a sine is sharp
            f = midi_to_freq(m)
            y = I.render_note(name, f, 0.5, SR, 0.9)
            pure = sharpness(I.render_note("sine", f, 0.5, SR, 0.9), SR)
            s, b = sharpness(y, SR), brightness(y, SR)
            assert s < max(pure + 1.0, 1.5) and s < 2.4, (name, m, round(s, 2), round(pure, 2))
            assert b["above_5k"] < 0.03, (name, m, round(b["above_5k"], 3))


def test_level_matched():
    import math
    from genny.analysis import loudness
    for name, e in I.REGISTRY.items():
        lo, hi = (note_to_midi(n) for n in e["range"].split("-"))
        levels = [loudness(I.render_note(name, midi_to_freq(m), 0.5, SR, 0.9), SR, window=0.15)
                  for m in (lo + (hi - lo) // 4, (lo + hi) // 2, hi - (hi - lo) // 4)]
        assert abs(sum(levels) / 3 + 12.0) < 1.0, (name, [round(v, 1) for v in levels], "run examples/level_match.py")
    assert math.isclose(I.key_gain(261.63), 1.0)


if __name__ == "__main__":
    for fn in (test_shapes_and_levels, test_tuning, test_not_piercing, test_level_matched):
        fn()
        print("ok", fn.__name__)


def test_vibrato_speeds_up_at_the_end_of_a_held_note():
    from genny.instruments import _vibrato
    sr, n = 8000, 8000 * 3
    semis = 12 * np.log2(_vibrato(440.0, n, sr, rate=5.8, depth=0.25, delay=0.1, end_boost=2.0) / 440.0)
    def rate(seg):          # from the last full period: upward zero crossings, interpolated
        i = np.flatnonzero((seg[:-1] < 0) & (seg[1:] >= 0))
        t = (i + seg[i] / (seg[i] - seg[i + 1])) / sr
        return 1.0 / (t[-1] - t[-2])
    mid, end = rate(semis[sr:2 * sr]), rate(semis)
    assert abs(mid - 5.8) < 0.1 and end > mid + 1.2, (mid, end)
    flat = 12 * np.log2(_vibrato(440.0, n, sr, rate=5.8, depth=0.25, delay=0.1) / 440.0)
    assert abs(rate(flat) - 5.8) < 0.1


def test_generic_bright_and_warmth_variants():
    from genny import instruments as I
    from genny.analysis import brightness
    sr = 44100
    own = [n for n, e in I.REGISTRY.items() if "bright" not in e["generic"]]
    assert "piano" in own and "bell" in own                        # their own bright control wins
    assert all({"bright", "warmth"} <= set(e["params"]) for e in I.REGISTRY.values())
    c = [brightness(I.render_note("flute", 523.25, 1.0, sr, 0.8, bright=b), sr)["centroid"] for b in (0.0, 0.5, 1.0)]
    assert c[0] < c[1] < c[2], c
    w = [brightness(I.render_note("cello", 130.8, 1.0, sr, 0.8, warmth=v), sr)["centroid"] for v in (0.0, 1.0)]
    assert w[1] < w[0], w
    a, b = I.render_note("flute", 523.25, 1.0, sr, 0.8), I.render_note("flute", 523.25, 1.0, sr, 0.8, bright=0.5, warmth=0.5)
    assert np.array_equal(a, b)                                    # 0.5 = as voiced, untouched


def test_onset_lag_delays_only_the_upper_partials():
    from genny import instruments as I
    from genny.analysis import brightness
    sr = 44100
    a, b = I.render_note("organ", 261.63, 0.5, sr, 0.8), I.render_note("organ", 261.63, 0.5, sr, 0.8, onset_lag=20.0)
    head, k = slice(0, int(0.008 * sr)), int(0.03 * sr)
    assert brightness(b[head], sr)["centroid"] < 0.8 * brightness(a[head], sr)["centroid"]
    assert np.allclose(a[k:], b[k:], atol=1e-4)                     # past the lag: the note as voiced (DC blocker memory aside)


def test_rhodes_pickup_voicing_and_bark():
    from genny import instruments as I
    sr = 44100

    def harm(y, f0):
        seg = y[4410:30000] * np.hanning(25590)
        S = np.abs(np.fft.rfft(seg))
        f = np.fft.rfftfreq(len(seg), 1 / sr)
        return [20 * np.log10(S[np.argmin(np.abs(f - f0 * k))] + 1e-12) for k in (1, 2, 3, 4, 5)]
    centred = harm(I.render_note("rhodes", 196.0, 1.0, sr, 0.6, voicing=0.0), 196.0)
    off = harm(I.render_note("rhodes", 196.0, 1.0, sr, 0.6, voicing=1.0), 196.0)
    assert centred[1] > centred[0] + 20 and off[0] > off[1] + 3          # centred: odd partials cancel (Shear 2.2.1)
    soft = harm(I.render_note("rhodes", 196.0, 1.0, sr, 0.2), 196.0)
    hard = harm(I.render_note("rhodes", 196.0, 1.0, sr, 1.0), 196.0)
    assert hard[2] - hard[0] > soft[2] - soft[0] + 6                     # harder = more distortion (bark)
    # measured Q: a low tine rings far longer than a high one
    def drop(y):                                                       # dB lost between 0.05 s and 1 s
        r = lambda a: np.sqrt(np.mean(y[int(a * sr):int(a * sr) + 2205] ** 2))   # noqa: E731
        return 20 * np.log10(r(0.05) / r(1.0))
    lo, hi = I.render_note("rhodes", 77.8, 1.5, sr, 0.8), I.render_note("rhodes", 1244.5, 1.5, sr, 0.8)
    assert drop(hi) > drop(lo) + 3, (drop(lo), drop(hi))

def test_physical_brass_holds_the_written_pitch():
    from genny.physical import waveguides as wg
    sr = wg.SR                                                         # measure_f0 assumes the kernel rate
    for name, f in (("trumpet", 233.1), ("trumpet", 698.5), ("french_horn", 130.8), ("trombone", 98.0), ("tuba", 55.0)):
        y = I.render_note(name, f, 1.5, sr, 0.8, model="physical")
        seg = y[int(0.4 * sr):int(1.3 * sr)]
        mf = wg.measure_f0(seg, f)
        assert abs(1200 * np.log2(mf / f)) < 3, (name, f, mf)
        w = max(2205, int(8 * sr / f))
        r = np.sqrt(np.convolve(seg ** 2, np.ones(w) / w, "valid"))
        assert (r.max() - r.min()) / r.mean() < 0.1, name               # no beating: one pitch source
        ref = I.render_note(name, f, 1.5, sr, 0.8)[int(0.4 * sr):int(1.3 * sr)]
        assert abs(20 * np.log10(np.std(seg) / np.std(ref))) < 2.5, name


def test_wurli_capacitive_pickup_gets_richer_with_velocity():
    sr, f = 44100, 261.6
    def h2(v):
        y = I.render_note("wurli", f, 1.0, sr, v)[2205:13230]
        S = np.abs(np.fft.rfft(y * np.hanning(len(y)), 1 << 16))
        fr = np.fft.rfftfreq(1 << 16, 1 / sr)
        return S[np.argmin(abs(fr - 2 * f))] / S[np.argmin(abs(fr - f))]
    assert h2(1.0) > 2 * h2(0.3) and h2(0.3) < 0.3                 # pure when soft, lopsided when hit hard


def test_magnetic_pickup_adds_harmonics_that_grow_with_the_swing():
    from genny import acoustic as AC
    sr, f = 44100, 110.0
    x = np.sin(2 * np.pi * f * np.arange(sr) / sr)
    def thd(y):
        S = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        k = int(round(f * len(y) / sr))
        return np.sqrt(sum(S[n * k] ** 2 for n in range(2, 8))) / S[k]
    for kind in AC.PICKUPS:
        assert thd(AC.magnetic_pickup(x, sr, kind, swing_mm=2.0)) > 3 * thd(AC.magnetic_pickup(x, sr, kind, swing_mm=0.5))
    # Novak: close to the pole the humbucker bends harder than the single coil
    assert thd(AC.magnetic_pickup(x, sr, "humbucker")) > thd(AC.magnetic_pickup(x, sr, "single"))


def test_harpsichord_ripple_loss_filter_and_tuning():
    # Valimaki et al. 2004 Fig. 8: g 0.996, a -0.0296, r -0.0015, rrate 1/12 at 197 Hz -> partial 6 rings
    # longest, partials 1 and 12 shortest
    from genny import acoustic as A
    from genny import filters as F
    from genny.instruments import render_note
    sr, f = 44100, 197.0
    exc = np.zeros(64)
    exc[0] = 1.0
    y = A._harpsi_string(f, 3 * sr, 3 * sr, sr, exc, 6.0, 1e-6)
    def t60(k):
        b = F.bandpass(y, k * f, sr, 40.0)
        a1, a2 = (np.sqrt(np.mean(b[int(t * sr):int((t + 0.3) * sr)] ** 2)) for t in (0.3, 2.3))
        return 2.0 * 60 / (20 * np.log10(a1 / a2))
    assert t60(6) > 1.3 * t60(1) and t60(6) > 1.5 * t60(12)
    for note in (65.4, 523.3):
        z = render_note("harpsichord", note, 1.0, sr, 0.9)
        seg = z[int(.1 * sr):int(.9 * sr)]
        Y = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 8 * len(seg)))
        fr = np.fft.rfftfreq(8 * len(seg), 1 / sr)
        m = (fr > note * .9) & (fr < note * 1.1)
        assert abs(1200 * np.log2(fr[m][np.argmax(Y[m])] / note)) < 3


def test_guitar_body_modes_spread_and_size():
    from genny.acoustic import guitar_body_modes
    f0, q0, _ = guitar_body_modes(0)
    assert 75 <= f0[0] <= 100 and 150 <= f0[1] <= 250                     # Woodhouse's two bands
    shifts = np.array([guitar_body_modes(s)[0][:2] / f0[:2] for s in range(1, 200)])
    assert 0.04 < np.std(np.log(shifts)) < 0.08                            # ~6 % between guitars
    small, big = guitar_body_modes(0, 0.0)[0], guitar_body_modes(0, 1.0)[0]
    assert np.all(small[:2] > f0[:2]) and np.all(big[:2] < f0[:2])          # Penttinen warp: size moves every mode
    assert np.allclose(guitar_body_modes(0, 0.5, 44100)[0], guitar_body_modes(0, 0.5, 48000)[0])
