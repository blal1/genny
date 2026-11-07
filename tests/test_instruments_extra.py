"""L19 instruments: the same register, tuning, level and brightness contract as tests/test_instruments.py, measured on every
semitone of each `span`, and each voice's sourced or stated behaviour measured on the rendered signal."""
import numpy as np
import pytest

from genny import instruments as I
from genny import instruments_extra as X
from genny.analysis import brightness, detect_clicks, loudness, sharpness
from genny.notes import midi_to_freq, note_to_midi, to_freq

SR = 44100
NAMES = ["ukulele", "bagpipes", "mellotron", "zither", "shofar", "traditional_horn", "slide_whistle", "waterphone",
         "talking_drum", "berimbau", "shamisen", "calliope", "flexatone", "risset_arpeggio", "gendyn"]
VARIANTS = [("traditional_horn", {"kind": k}) for k in X.TRAD_HORNS] + [
    ("waterphone", {"action": "struck"}), ("waterphone", {"slosh": 1.0}), ("slide_whistle", {"sweep": 12.0}), ("slide_whistle", {"sweep": -12.0, "slide": 0.2}),
    ("talking_drum", {"squeeze": 1.0}), ("talking_drum", {"squeeze": 0.6, "curve": "wave"}), ("talking_drum", {"squeeze": 0.6, "curve": "fall"}),
    ("flexatone", {"squeeze": 0.5, "curve": "rise"}), ("flexatone", {"rate": 20.0}), ("berimbau", {"gourd": 0.0}), ("berimbau", {"wah": 3.0}),
    ("berimbau", {"press": 2.0}), ("shamisen", {"sawari": 0.0}), ("shamisen", {"sawari": 1.0, "skin": 1.0}), ("bagpipes", {"drones": 0.0}),
    ("bagpipes", {"chanter_a": 441.0}), ("mellotron", {"flutter": 1.0, "wow": 0.5}), ("mellotron", {"tape_s": 0.2}), ("zither", {"ring": 1.0}), ("calliope", {"steam": 1.0}),
    ("calliope", {"waver": 0.3})]


def span(name):
    lo, hi = (note_to_midi(n) for n in I.REGISTRY[name]["range"].split("-"))
    return lo, hi


def cents_off(y, f0):
    """tests/test_instruments.py's measure: the autocorrelation peak next to the expected period."""
    y = y[int(0.05 * SR):int(0.05 * SR) + 16384]
    y = y - y.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(y * np.hanning(len(y)), 4 * len(y))) ** 2)
    p = SR / f0 * 4
    lo = int(p * 0.94)
    k = lo + int(np.argmax(ac[lo:int(p * 1.06) + 2]))
    k = k + 0.5 * (ac[k - 1] - ac[k + 1]) / (ac[k - 1] - 2 * ac[k] + ac[k + 1] + 1e-20)
    return 1200 * np.log2((SR * 4 / k) / f0)


def peak_hz(y, f, width=0.06):
    """Strongest spectral line within +-width of f, on a zero-padded Hann transform."""
    S = np.abs(np.fft.rfft(y * np.hanning(len(y)), 1 << 19))
    fr = np.fft.rfftfreq(1 << 19, 1 / SR)
    m = (fr > f * (1 - width)) & (fr < f * (1 + width))
    return float(fr[m][np.argmax(S[m])])


def line(y, f):
    """Level of the spectrum at f."""
    S = np.abs(np.fft.rfft(y * np.hanning(len(y)), 1 << 19))
    return float(S[int(round(f * (1 << 19) / SR))])


def cents(a, b):
    return 1200 * np.log2(a / b)


def centroids(y, frame=4096):
    fr = np.fft.rfftfreq(frame, 1 / SR)
    out = []
    for i in range(0, len(y) - frame, frame // 2):
        S = np.abs(np.fft.rfft(y[i:i + frame] * np.hanning(frame))) ** 2
        out.append(np.sum(fr * S) / (np.sum(S) + 1e-20))
    return np.array(out)


def test_all_registered_with_their_own_span():
    assert sorted(NAMES) == sorted(X._LEVEL)
    for n in NAMES:
        e = I.REGISTRY[n]
        assert e["fn"].__module__ == "genny.instruments_extra" and e["range"] and e["family"], n
        assert span(n)[1] > span(n)[0]


@pytest.mark.parametrize("name", NAMES)
def test_in_tune_on_every_semitone_of_the_span(name):
    lo, hi = span(name)
    bad = {m: round(float(cents_off(I.render_note(name, midi_to_freq(m), 0.6, SR, 0.9), midi_to_freq(m))), 1) for m in range(lo, hi + 1)}
    assert all(abs(c) < 10 for c in bad.values()), bad


@pytest.mark.parametrize("name", NAMES)
def test_contract_of_test_instruments(name):
    """Registry-wide tests/test_instruments.py criteria: shapes outside the span, level, sharpness."""
    lo, hi = span(name)
    for m in (lo - 5, lo, (lo + hi) // 2, hi, hi + 7):
        for dur, vel, sr in ((0.03, 1.0, 44100), (0.1234, 0.3, 44100), (0.3333, 0.7, 22050), (1.0, 1.0, 48000)):
            y = I.render_note(name, midi_to_freq(m), dur, sr, vel)
            assert y.ndim == 1 and np.all(np.isfinite(y)), (name, m, dur)
            assert 1e-4 < np.max(np.abs(y)) < 4.0, (name, m, dur, float(np.max(np.abs(y))))
            assert abs(y[-1]) < 1e-3 and abs(y[0]) < 1e-2, (name, m, dur)
    levels = [loudness(I.render_note(name, midi_to_freq(m), 0.5, SR, 0.9), SR, window=0.15) for m in (lo + (hi - lo) // 4, (lo + hi) // 2, hi - (hi - lo) // 4)]
    assert abs(sum(levels) / 3 + 12.0) < 1.0, levels
    for m in sorted({min((lo + hi) // 2, 96), min(hi, 96)}):
        f = midi_to_freq(m)
        y = I.render_note(name, f, 0.5, SR, 0.9)
        pure = sharpness(I.render_note("sine", f, 0.5, SR, 0.9), SR)
        s, b = sharpness(y, SR), brightness(y, SR)
        assert s < max(pure + 1.0, 1.5) and s < 2.4, (m, s, pure)
        assert b["above_5k"] < 0.03, (m, b["above_5k"])


@pytest.mark.parametrize("name,params", [(n, {}) for n in NAMES] + VARIANTS, ids=lambda v: v if isinstance(v, str) else ",".join(f"{k}={x}" for k, x in v.items()) or "default")
def test_no_clicks(name, params):
    lo, hi = span(name)
    for m in (lo, (lo + hi) // 2, hi):
        for vel in (0.3, 1.0):
            y = I.render_note(name, midi_to_freq(m), 1.0, SR, vel, **params)
            assert np.all(np.isfinite(y))
            c = detect_clicks(y)
            assert len(c) == 0, (m, vel, len(c), (c[:5] / SR).round(4).tolist())


def test_ukulele_gcea_open_strings():
    assert X.UKULELE_OPEN == ("G4", "C4", "E4", "A4")
    assert to_freq("G4") > to_freq("E4") > to_freq("C4")            # re-entrant: the 4th string sits above the 3rd
    for n in X.UKULELE_OPEN:
        f = to_freq(n)
        assert abs(cents_off(I.render_note("ukulele", f, 0.6, SR, 0.9), f)) < 5, n


def test_bagpipes_chanter_and_drones_on_a():
    assert 476 <= X.BAG_CHANTER_A <= 486                            # BAG: today's chanter
    for a in (X.BAG_CHANTER_A, 441.0):                              # BAG: 441 in 1885
        y = I.render_note("bagpipes", midi_to_freq(76), 1.2, SR, 0.9, chanter_a=a)[int(0.2 * SR):int(1.1 * SR)]
        assert abs(cents(peak_hz(y, a / 2, 0.03), a / 2)) < 3
        assert abs(cents(peak_hz(y, a / 4, 0.03), a / 4)) < 3
        assert abs(cents(peak_hz(y, midi_to_freq(76), 0.03), midi_to_freq(76))) < 3
    bare = I.render_note("bagpipes", midi_to_freq(76), 1.2, SR, 0.9, drones=0.0)[int(0.2 * SR):int(1.1 * SR)]
    full = I.render_note("bagpipes", midi_to_freq(76), 1.2, SR, 0.9)[int(0.2 * SR):int(1.1 * SR)]
    assert line(full, X.BAG_CHANTER_A / 4) > 20 * line(bare, X.BAG_CHANTER_A / 4)


def test_mellotron_flutter_and_tape_end():
    f = midi_to_freq(60)

    def spread(fl):
        from scipy.signal import hilbert
        y = I.render_note("mellotron", f, 2.0, SR, 0.9, flutter=fl, wow=0.0)[int(0.3 * SR):int(1.9 * SR)]
        z = hilbert(I.F.bandpass(y, f, SR, 12.0))
        inst = np.diff(np.unwrap(np.angle(z))) * SR / (2 * np.pi)
        inst = I.F.lowpass(inst, 30.0, SR, order=2)                 # harmonics leaking through the band ripple it at audio rate
        return float(np.std(inst[8000:-8000]))
    assert spread(1.5) > 2 * spread(0.0)
    short = I.render_note("mellotron", f, 4.0, SR, 0.9, tape_s=1.0)
    assert len(short) < 1.3 * SR


def test_slide_whistle_follows_the_stopped_pipe_law():
    from genny.tubes import _pipe
    f = 880.0
    L = float(X.slide_length(f))
    assert abs(_pipe(f, X.SLIDE_RADIUS_M, 1.0).L / L - 1) < 0.06    # the Webster pipe of the same note, grid included
    y = I.render_note("slide_whistle", f, 1.0, SR, 0.9, sweep=12.0, breath=0.0)
    L0 = float(X.slide_length(2 * f))
    for x in (0.2, 0.5, 0.8):                                       # steady piston: f(t) = c / 4 (L(t) + 0.6133 a)
        want = X.C_AIR / (4.0 * (L0 + (L - L0) * x + X.UNFLANGED * X.SLIDE_RADIUS_M))
        k = int(x * SR)
        seg = y[k - 400:k + 400]
        assert abs(cents(peak_hz(seg, want, 0.1), want)) < 15, x
    # steady in pitch when the piston does not move
    assert abs(cents_off(I.render_note("slide_whistle", f, 0.6, SR, 0.9), f)) < 1


@pytest.mark.parametrize("name,k", [("talking_drum", X.TALKING_K), ("flexatone", X.FLEXATONE_K)])
def test_squeeze_law(name, k):
    assert X.tension_ratio(1.0, 3.0) == pytest.approx(2.0)          # an octave at full squeeze
    f0 = midi_to_freq(55 if name == "talking_drum" else 72)
    for s in (0.25, 0.5, 1.0):
        want = f0 * np.sqrt(1 + k * s)
        c = cents_off(I.render_note(name, f0, 0.6, SR, 0.9, squeeze=s), want)
        assert abs(c) < 6, (s, c)
    y = I.render_note(name, f0, 1.0, SR, 0.9, squeeze=1.0, curve="rise")
    head, tail = peak_hz(y[:int(0.1 * SR)], f0, 0.12), peak_hz(y[int(0.85 * SR):int(1.0 * SR)], 2 * f0, 0.12)
    assert abs(cents(head, f0)) < 60 and abs(cents(tail, 2 * f0)) < 60, (head, tail)
    with pytest.raises(ValueError):
        I.render_note(name, f0, 0.3, SR, 0.9, curve="nope")


def test_talking_drum_membrane_partials():
    from genny.physics import MembraneDrum
    f = midi_to_freq(55)
    y = I.render_note("talking_drum", f, 0.8, SR, 0.9)[int(0.01 * SR):int(0.3 * SR)]
    r = MembraneDrum._RATIOS[1]
    assert abs(cents(peak_hz(y, f * r, 0.03), f * r)) < 15          # circular membrane, not harmonic


def test_flexatone_strike_rate():
    f = midi_to_freq(72)
    y = I.render_note("flexatone", f, 2.0, SR, 0.9, rate=12.0)[:2 * SR]
    e = I.F.lowpass(np.abs(y), 60.0, SR)
    e = e[int(0.1 * SR):] - e[int(0.1 * SR):].mean()
    S = np.abs(np.fft.rfft(e * np.hanning(len(e)), 1 << 18))
    fr = np.fft.rfftfreq(1 << 18, 1 / SR)
    m = (fr > 3) & (fr < 40)
    # the two balls strike in turn at different strengths: lines at the strike rate and at half of it
    assert abs(fr[m][np.argmax(S[m])] - 12.0) < 1.0 or abs(fr[m][np.argmax(S[m])] - 6.0) < 0.5


def test_waterphone_rods_and_formants():
    f = midi_to_freq(60)
    assert X.vessel_formants(SR, SR, 1.0).max() == pytest.approx(np.sqrt(1 + X.WATERPHONE_K), abs=1e-3)   # f = f0 sqrt(1 + k slosh)
    assert X.vessel_formants(SR, SR, 0.0).max() == 1.0

    sway = [float(np.std(centroids(I.render_note("waterphone", f, 3.0, SR, 0.9, slosh=s, neighbours=0.0)[int(0.3 * SR):int(2.9 * SR)]))) for s in (0.0, 1.0)]
    assert sway[1] > 1.3 * sway[0], sway                            # moving water re-equalises the formants
    # unequal rods: a neighbour at f / L^2 appears only with the others ringing
    fn = f / X.WATERPHONE_RODS[1] ** 2
    a = I.render_note("waterphone", f, 1.5, SR, 0.9, neighbours=1.0, slosh=0.0)[int(0.1 * SR):int(1.0 * SR)]
    b = I.render_note("waterphone", f, 1.5, SR, 0.9, neighbours=0.0, slosh=0.0)[int(0.1 * SR):int(1.0 * SR)]
    assert line(a, fn) > 3 * line(b, fn)
    # cantilever partial 6.267 when struck
    s = I.render_note("waterphone", 174.6, 1.5, SR, 1.0, action="struck", neighbours=0.0, slosh=0.0)[:int(0.8 * SR)]
    assert abs(cents(peak_hz(s, 174.6 * X.CLAMPED[1], 0.02), 174.6 * X.CLAMPED[1])) < 15
    # struck: at once; bowed: the bow takes time to bring the rod up
    def rise(y):
        e = np.abs(y)
        return np.argmax(e > 0.5 * e.max()) / SR
    assert rise(I.render_note("waterphone", f, 1.5, SR, 0.9, action="struck")) < rise(I.render_note("waterphone", f, 1.5, SR, 0.9))
    with pytest.raises(ValueError):
        I.render_note("waterphone", f, 0.3, SR, 0.9, action="plucked")


def test_berimbau_gourd_and_stone():
    f = midi_to_freq(52)
    assert X.gourd_freq(0.25) == pytest.approx(0.5 * X.BERIMBAU_GOURD_HZ)   # Helmholtz, f ~ sqrt(A)
    def rms(g):
        y = I.render_note("berimbau", f, 0.8, SR, 0.9, gourd=g)
        return float(np.sqrt(np.mean(y ** 2)))
    assert rms(1.0) > 1.5 * rms(0.0)                                # closed against the body: less air
    # wah: the level swings at the rate of the gesture
    y = I.render_note("berimbau", f, 2.0, SR, 0.9, wah=4.0)[int(0.2 * SR):int(1.8 * SR)]
    dry = I.render_note("berimbau", f, 2.0, SR, 0.9)[int(0.2 * SR):int(1.8 * SR)]
    def wobble(x):
        e = I.F.lowpass(np.abs(x), 25.0, SR)
        e = e / (I.F.lowpass(e, 1.5, SR) + 1e-6)
        s = np.abs(np.fft.rfft((e - e.mean()) * np.hanning(len(e)), 1 << 18))
        fr = np.fft.rfftfreq(1 << 18, 1 / SR)
        return float(s[np.argmin(np.abs(fr - 4.0))])
    assert wobble(y) > 3 * wobble(dry)
    p = I.render_note("berimbau", f, 0.6, SR, 0.9, press=2.0)       # the stone on the string
    assert abs(cents_off(p, f * 2 ** (2 / 12))) < 10


def test_shamisen_sawari_only_on_the_low_string():
    low, high = to_freq("D3"), to_freq("G4")
    assert low < to_freq(X.SHAMISEN_SAWARI_BELOW) <= high
    def upper(fr, s):
        y = I.render_note("shamisen", fr, 0.8, SR, 0.9, sawari=s, skin=0.0)[int(0.02 * SR):int(0.6 * SR)]
        hi = I.F.highpass(y, 8 * fr, SR, order=2)
        return float(np.sqrt(np.mean(hi ** 2)) / np.sqrt(np.mean(y ** 2)))
    assert upper(low, 1.0) > 1.5 * upper(low, 0.05)
    assert np.array_equal(I.render_note("shamisen", high, 0.4, SR, 0.9, sawari=0.0), I.render_note("shamisen", high, 0.4, SR, 0.9, sawari=1.0))


def test_shofar_bore_is_irregular():
    t = X._shofar_tube()
    u = np.zeros(1 << 15)
    u[0] = 1.0
    S = np.abs(np.fft.rfft(t.run(u)["u_out"]))
    fr = np.fft.rfftfreq(1 << 15, 1 / X.FS)
    i = np.flatnonzero((S[1:-1] > S[:-2]) & (S[1:-1] > S[2:]) & (S[1:-1] > 0.1 * S.max())) + 1
    i = i[fr[i] > 100]
    r = fr[i[1:]] / fr[i[0]]
    assert len(r) >= 2 and all(min(abs(x - round(x)), abs(2 * x - round(2 * x)) / 2) > 0.02 for x in r[:2]), r   # neither cone nor cylinder series


def test_traditional_horns_differ():
    f = midi_to_freq(55)
    c = {k: brightness(I.render_note("traditional_horn", f, 1.0, SR, 0.9, kind=k), SR)["centroid"] for k in X.TRAD_HORNS}
    assert c["war_horn"] > 1.1 * c["alphorn"], c
    assert abs(c["conch"] - c["alphorn"]) > 20, c
    with pytest.raises(ValueError):
        I.render_note("traditional_horn", f, 0.3, SR, 0.9, kind="tuba")


def test_zither_open_strings_ring_along():
    played, open_ = midi_to_freq(62), to_freq("G3")                 # D4 on a course whose G3 is open
    y = [I.render_note("zither", played, 1.0, SR, 0.9, ring=r)[int(0.05 * SR):int(0.8 * SR)] for r in (0.0, 1.0)]
    assert line(y[1], open_) > 2 * line(y[0], open_)


def test_calliope_steam():
    f = midi_to_freq(67)
    def between(steam):
        y = I.render_note("calliope", f, 1.0, SR, 0.9, steam=steam, waver=0.0)[int(0.2 * SR):int(0.9 * SR)]
        return line(y, 1.5 * f) / (line(y, f) + 1e-12)
    assert between(1.0) > 3 * between(0.0)                          # hiss between the partials


def test_gendyn_walks_stay_inside_their_walls():
    from genny import osc as O
    assert np.allclose(O.reflect(np.array([1.3, -1.25, 0.2, 3.5]), -1.0, 1.0), [0.7, -0.75, 0.2, -0.5])
    x = O.gendyn(220.0, SR * 2, SR, dur_spread=0.1, seed=3)
    assert x.max() <= 1.0 and x.min() >= -1.0 and np.ptp(x) > 1.5
    # no walk: a fixed cycle of exactly sr / f samples
    f = SR / 200.0
    y = O.gendyn(f, SR, SR, points=10, amp_step=0.0, dur_spread=0.0, seed=1)
    assert np.allclose(y[400:600], y[600:800], atol=1e-9)
    # walking lengths: the spectral peak stays within +-10 % of the note
    X = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    fr = np.fft.rfftfreq(x.size, 1 / SR)
    pk = fr[np.argmax(np.where(fr > 50, X, 0))]
    assert abs(pk / 220.0 - 1) < 0.1
