"""genny.choir: measured checks of the sung-voice model (not just "it renders").

Run:  uv run --with pytest pytest tests/test_choir.py -q      or      uv run python tests/test_choir.py
Every test prints the numbers it measured (pytest -s, or the __main__ runner).
"""
import time

import numpy as np
from scipy.signal import hilbert, welch

from genny import choir as C
from genny import instruments as I
from genny import spec as S
from genny.analysis import brightness, loudness, sharpness
from genny.notes import midi_to_freq, note_to_midi

SR = 44100
NAMES = ("voice", "vocal_choir", "hum", "throat_singing", "falsetto", "boys_choir", "chant")


# ------------------------------------------------------------------------------- measuring tools
def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def mid(y, a=0.5, b=None, sr=SR):
    return y[int(a * sr): (int(b * sr) if b else len(y) - int(0.4 * sr))]


def spectral_peaks(y, sr=SR, smooth_hz=45.0, fmax=4500.0):
    """Peaks (Hz) of the smoothed long-term spectrum."""
    f, p = welch(y, sr, nperseg=4096, noverlap=3072)
    d = 10 * np.log10(p + 1e-20)
    k = np.exp(-0.5 * (np.arange(-40, 41) * (f[1] - f[0]) / smooth_hz) ** 2)
    d = np.convolve(d, k / k.sum(), mode="same")
    i = np.flatnonzero((d[1:-1] > d[:-2]) & (d[1:-1] >= d[2:]) & (f[1:-1] > 120) & (f[1:-1] < fmax)) + 1
    return f[i], d[i]


def f0_track(y, f0, sr=SR):
    """Instantaneous frequency (Hz) of the first harmonic: band-limit around f0, analytic signal."""
    Y = np.fft.rfft(y)
    f = np.fft.rfftfreq(len(y), 1 / sr)
    Y[(f < f0 * 0.7) | (f > f0 * 1.4)] = 0
    ph = np.unwrap(np.angle(hilbert(np.fft.irfft(Y, len(y)))))
    inst = np.diff(ph) * sr / (2 * np.pi)
    k = int(sr / f0 * 2)
    return np.convolve(inst, np.ones(k) / k, mode="same")


def cents_off(y, f0, sr=SR):
    """tests/test_instruments.py: pitch error from the autocorrelation peak next to the period."""
    y = y[int(0.05 * sr):int(0.05 * sr) + 16384]
    y = y - y.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(y * np.hanning(len(y)), 4 * len(y))) ** 2)
    p = sr / f0 * 4
    lo = int(p * 0.94)
    k = lo + int(np.argmax(ac[lo:int(p * 1.06) + 2]))
    k = k + 0.5 * (ac[k - 1] - ac[k + 1]) / (ac[k - 1] - 2 * ac[k] + ac[k + 1] + 1e-20)
    return 1200 * np.log2((sr * 4 / k) / f0)


def harmonic_db(y, f0, k, sr=SR):
    s = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / sr)
    return 20 * np.log10(s[(f > k * f0 * 0.97) & (f < k * f0 * 1.03)].max() + 1e-12)


def band_db(y, lo, hi, sr=SR):
    s = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / sr)
    return 10 * np.log10(s[(f >= lo) & (f < hi)].sum() / s.sum() + 1e-20)


def rms_env(y, sr=SR, ms=10.0):
    k = int(sr * ms / 1000)
    return np.sqrt(np.mean(y[:len(y) // k * k].reshape(-1, k) ** 2, axis=1))


def solo(midi, dur=2.5, **kw):
    kw.setdefault("stereo", False)
    return C.sing_phrase([(0.0, midi, dur, 1.0)], SR, **kw)


# ------------------------------------------------------------------------------------- formants
def test_formant_peaks_match_tables():
    """Sustained /a/ /i/ /u/, bass and soprano: F1 and F2 (and F3 of /a/, /i/) of the tract within
    12 % of the published table. Measured on the whispered vowel (noise through the same tract:
    Cook - whispered vowels keep the formant peaks), so the harmonics do not sample the envelope."""
    worst = 0.0
    for voice, m in (("bass", 43), ("soprano", 55)):
        for vw, lyr in (("a", "ah"), ("i", "ee"), ("u", "oo")):
            tab = C.voice_formants(voice, vw)["freqs"]
            y = mid(solo(m, 3.0, voice=voice, lyrics=lyr, mode="whisper", ring=1.0))
            pk, _ = spectral_peaks(y)
            got = []
            for k in range(3 if vw != "u" else 2):
                e = pk[np.argmin(np.abs(pk - tab[k]))]
                got.append(e)
                err = abs(e / tab[k] - 1)
                worst = max(worst, err)
                assert err < 0.12, (voice, vw, k + 1, round(float(e)), tab[k])
            print(f"  {voice} /{vw}/ table {np.round(tab[:3])} measured {np.round(got)}")
    print(f"  worst formant error {100 * worst:.1f} %")


def test_voiced_vowel_has_the_same_formants():
    """The sung (voiced) bass /a/ has its strongest partials at F1 and in the F3-F5 cluster."""
    tab = C.voice_formants("bass", "a")["freqs"]
    f0 = hz(40)
    y = mid(solo(40, 3.0, voice="bass", lyrics="ah", vibrato=0, ring=1.0))
    lv = np.array([harmonic_db(y, f0, k) for k in range(1, 40)])
    fk = f0 * np.arange(1, 40)
    f1 = fk[(fk > 350) & (fk < 900)][np.argmax(lv[(fk > 350) & (fk < 900)])]
    fc = fk[(fk > 1800) & (fk < 3300)][np.argmax(lv[(fk > 1800) & (fk < 3300)])]
    print(f"  bass /a/ strongest partial near F1: {f1:.0f} Hz (table {tab[0]:.0f}); in the cluster: {fc:.0f} Hz (table {tab[2]:.0f}-{tab[4]:.0f})")
    assert abs(f1 - tab[0]) < f0 and tab[2] - f0 < fc < tab[4] + f0


def test_soprano_f1_tracks_f0():
    f = C.tuned_formants(np.tile(C.voice_formants("soprano", "i")["freqs"], (3, 1)), np.array([200.0, 600.0, 900.0]), "soprano")
    assert f[0, 0] == 270.0 and f[1, 0] == 600.0 and f[2, 0] == 900.0
    # acoustically: /i/ (F1 270 Hz) sung at 700 Hz - with tuning the fundamental sits on F1
    m = 69 + 12 * np.log2(700 / 440)
    on = mid(solo(m, 2.0, voice="soprano", lyrics="ee", vibrato=0), 0.5)
    off = mid(solo(m, 2.0, voice="soprano", lyrics="ee", vibrato=0, tuning=False), 0.5)
    d_on = harmonic_db(on, 700, 1) - harmonic_db(on, 700, 3)
    d_off = harmonic_db(off, 700, 1) - harmonic_db(off, 700, 3)
    print(f"  soprano /i/ at 700 Hz: H1-H3 {d_on:.1f} dB tuned, {d_off:.1f} dB untuned")
    assert d_on > d_off + 6.0


def test_singers_formant_follows_ring():
    out = []
    for voice, m in (("bass", 45), ("tenor", 55)):
        lv = [band_db(mid(solo(m, 2.5, voice=voice, vibrato=0, ring=r)), 2400, 3400) for r in (0.0, 0.5, 1.0)]
        out.append((voice, np.round(lv, 1)))
        assert lv[0] + 4 < lv[1] < lv[2] - 4, (voice, lv)
    print(f"  2.4-3.4 kHz share (dB re total) at ring 0 / 0.5 / 1: {out}")


# ----------------------------------------------------------------------------------- the source
def test_vibrato_rate_and_extent():
    for rate, ext in ((5.0, 40.0), (6.2, 80.0)):
        f0 = hz(57)
        y = solo(57, 4.0, voice="tenor", vibrato=ext, vib_rate=rate)
        c = 1200 * np.log2(f0_track(y, f0) / f0)[int(1.4 * SR):int(3.6 * SR)]
        c = c - c.mean()
        sp = np.abs(np.fft.rfft(c * np.hanning(len(c)), 8 * len(c)))
        fr = np.fft.rfftfreq(8 * len(c), 1 / SR)
        band = (fr > 3) & (fr < 9)
        got_rate = fr[band][np.argmax(sp[band])]
        got_ext = np.sqrt(2) * np.std(np.convolve(c, np.ones(400) / 400, mode="same"))
        print(f"  vibrato asked {rate} Hz +-{ext} c: measured {got_rate:.2f} Hz +-{got_ext:.1f} c")
        assert abs(got_rate - rate) < 0.25 and abs(got_ext / ext - 1) < 0.2
    # delayed onset: the first 150 ms are straight
    c = 1200 * np.log2(f0_track(y, f0) / f0)
    early = np.ptp(c[int(0.12 * SR):int(0.25 * SR)])
    print(f"  pitch range in 120-250 ms: {early:.1f} cents (vibrato not started)")
    assert early < 0.6 * ext


def test_breathy_vs_pressed_h1_h2():
    f0 = hz(55)
    d = []
    for breath, effort in ((0.0, 1.35), (0.1, 1.0), (1.0, 0.75)):
        y = mid(solo(55, 2.0, voice="tenor", vibrato=0, breath=breath, effort=effort))
        d.append(harmonic_db(y, f0, 1) - harmonic_db(y, f0, 2))
    print(f"  H1-H2 pressed / modal / breathy: {np.round(d, 1)} dB")
    assert d[0] + 2 < d[1] < d[2] - 2


def test_dynamics_change_the_spectrum():
    pp = mid(solo(57, 2.0, voice="tenor", vibrato=0, dynamics="pp"))
    ff = mid(solo(57, 2.0, voice="tenor", vibrato=0, dynamics="ff"))
    lv = 20 * np.log10(np.sqrt(np.mean(ff ** 2)) / np.sqrt(np.mean(pp ** 2)))
    pp = pp / np.sqrt(np.mean(pp ** 2))
    ff = ff / np.sqrt(np.mean(ff ** 2))
    c_pp, c_ff = brightness(pp, SR)["centroid"], brightness(ff, SR)["centroid"]
    u_pp, u_ff = band_db(pp, 2000, 4000), band_db(ff, 2000, 4000)
    print(f"  pp -> ff: level +{lv:.1f} dB; at equal level the centroid goes {c_pp:.0f} -> {c_ff:.0f} Hz and the "
          f"2-4 kHz share {u_pp:.1f} -> {u_ff:.1f} dB")
    assert lv > 8 and c_ff > 1.12 * c_pp and u_ff > u_pp + 6


def test_aspiration_is_gated_by_the_open_phase():
    """Breath noise is modulated at the pitch period (not a steady hiss added on top)."""
    f0 = hz(45)
    y = mid(solo(45, 2.0, voice="bass", vibrato=0, breath=1.0, dynamics="p"))
    Y = np.fft.rfft(y)
    f = np.fft.rfftfreq(len(y), 1 / SR)
    Y[(f < 4000) | (f > 9000)] = 0                       # above the harmonics: only noise is left
    e = np.abs(hilbert(np.fft.irfft(Y, len(y)))) ** 2
    sp = np.abs(np.fft.rfft((e - e.mean()) * np.hanning(len(e))))
    fe = np.fft.rfftfreq(len(e), 1 / SR)
    peak = sp[(fe > f0 * 0.97) & (fe < f0 * 1.03)].max()
    floor = np.median(sp[(fe > f0 * 1.2) & (fe < f0 * 1.8)])
    print(f"  noise envelope line at f0: {20 * np.log10(peak / floor):.1f} dB over the floor")
    assert peak > 4 * floor


# --------------------------------------------------------------------------------- the ensemble
def _beating(y, f0, k=3):
    """Coefficient of variation of the envelope of harmonic k: solo ~ steady, ensemble fades."""
    Y = np.fft.rfft(y)
    f = np.fft.rfftfreq(len(y), 1 / SR)
    Y[(f < k * f0 * 0.94) | (f > k * f0 * 1.06)] = 0
    e = np.abs(hilbert(np.fft.irfft(Y, len(y))))[int(0.8 * SR):-int(0.6 * SR)]
    return float(e.std() / e.mean())


def _line_width(y, f0, k=3):
    y = y[int(0.8 * SR):-int(0.6 * SR)]
    s = np.abs(np.fft.rfft(y * np.hanning(len(y)), 4 * len(y))) ** 2
    f = np.fft.rfftfreq(4 * len(y), 1 / SR)
    m = (f > k * f0 * 0.94) & (f < k * f0 * 1.06)
    p, ff = s[m] / s[m].sum(), f[m]
    c = (p * ff).sum()
    return float(np.sqrt((p * (ff - c) ** 2).sum())), float(1200 * np.log2(c / (k * f0)))


def test_ensemble_beats_and_stays_in_tune():
    f0 = hz(57)
    one = C.sing_phrase([(0, 57, 5.0, 1)], SR, section="tenors", singers=1, style="gregorian", vibrato=0, stereo=False)
    eight = C.sing_phrase([(0, 57, 5.0, 1)], SR, section="tenors", singers=8, style="gregorian", vibrato=0, stereo=False)
    b1, b8 = _beating(one, f0), _beating(eight, f0)
    (w1, c1), (w8, c8) = _line_width(one, f0), _line_width(eight, f0)
    print(f"  harmonic-3 envelope variation: solo {b1:.3f}, 8 singers {b8:.3f}; line width {w1:.2f} -> {w8:.2f} Hz; "
          f"mean pitch {c1:+.1f} / {c8:+.1f} cents")
    assert b8 > 3 * b1 and b8 > 0.2 and w8 > 1.5 * w1
    assert abs(c8) < 10 and abs(c1) < 10
    for style in ("classical", "epic", "children"):
        y = C.sing_phrase([(0, 60, 4.0, 1)], SR, section="mixed" if style == "epic" else "altos", singers=8, style=style, stereo=False)
        c = 1200 * np.log2(np.mean(f0_track(y, hz(60))[int(1.0 * SR):int(3.4 * SR)]) / hz(60))
        print(f"  {style} section of 8, mean pitch {c:+.1f} cents")
        assert abs(c) < 10


def test_singers_are_different_people():
    """No two streams are identical or delayed copies: peak of the normalised cross-correlation
    over +-50 ms stays under 0.9 (a chorus effect on one voice would give ~1)."""
    streams, n = C.render_singers([(0, 55, 2.5, 1), (2.5, 57, 1.5, 1)], SR, section="tenors", singers=8, lyrics="ah")
    xs = [s[int(0.3 * SR):int(3.6 * SR)] for s, _ in streams]
    lag = int(0.05 * SR)
    worst = 0.0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            a, b = xs[i] - xs[i].mean(), xs[j] - xs[j].mean()
            cc = np.fft.irfft(np.fft.rfft(a, 2 * len(a)) * np.conj(np.fft.rfft(b, 2 * len(a))))
            cc = np.concatenate([cc[-lag:], cc[:lag + 1]]) / np.sqrt((a * a).sum() * (b * b).sum())
            worst = max(worst, float(np.abs(cc).max()))
    pans = sorted(round(p, 2) for _, p in streams)
    print(f"  8 tenors: highest pairwise cross-correlation peak {worst:.3f}; pans {pans}")
    assert worst < 0.9 and len(set(pans)) == 8
    st = C.sing_phrase([(0, 55, 2.0, 1)], SR, section="tenors", singers=8)
    assert st.ndim == 2 and st.shape[1] == 2
    lr = np.corrcoef(st[SR // 2:SR * 3 // 2, 0], st[SR // 2:SR * 3 // 2, 1])[0, 1]
    print(f"  stereo: L/R correlation {lr:.2f}")
    assert lr < 0.97


# ---------------------------------------------------------------------------------------- text
def test_lyrics_parse():
    syl = C.parse_lyrics("A-ve Ma-ri-a", "la")
    got = [(s["onset"], s["nucl"], s["coda"]) for s in syl]
    assert got == [([], ["A"], []), (["V"], ["E"], []), (["M"], ["A"], []), (["DX"], ["I"], []), ([], ["A"], [])], got
    assert [s["nucl"] for s in C.parse_lyrics("Ave Maria", "la")] == [["A"], ["E"], ["A"], ["I"], ["A"]]   # no hyphens
    s = C.parse_lyrics("San-ctus Do-mi-nus De-us", "la")
    assert len(s) == 7 and s[0]["coda"] == ["N", "K"] and s[1]["onset"] == ["T"] and s[1]["coda"] == ["S"]
    ex = C.parse_lyrics("glo-ri-a in ex-cel-sis", "la")
    assert len(ex) == 7 and ex[4]["coda"] == ["K"] and ex[5]["onset"] == ["SH"] and ex[5]["coda"] == ["L"]   # ek-shel-sis
    pa = C.parse_lyrics("pa-cem", "la")
    assert pa[0]["coda"] == ["T"] and pa[1]["onset"] == ["SH"] and pa[1]["coda"] == ["M"]    # soft c = tch
    assert C.parse_lyrics("mm", "la")[0]["hum"] == "M" and C.parse_lyrics("oo", "en")[0]["nucl"] == ["U"]
    assert C.parse_lyrics("/T AA/ _", "en")[0]["onset"] == ["T"] and C.parse_lyrics("/T AA/ _", "en")[1] == "_"
    en = C.parse_lyrics("hal-le-lu-jah glo-ry", "en")
    assert len(en) == 6 and all(x["nucl"] for x in en)
    assert len(C.parse_lyrics("a-le-gri-a", "es")) == 4
    t, v = C.parse_dynamics("mp<f>p", 8.0)
    assert np.allclose(np.interp([0, 2, 4, 8], t, v), [0.45, 0.585, 0.72, 0.33])
    t, v = C.parse_dynamics("p f", 4.0)
    assert np.interp(3.9, t, v) < 0.34 and np.interp(4.0, t, v) == 0.72                      # subito


def test_consonant_before_the_beat_vowel_on_it():
    """ "ta" on beat 2 at 120 bpm (0.5 s): a noise burst just before the beat, and voicing (the
    first harmonic) starts on the beat (+-15 ms). A nasal hums before the beat and opens on it."""
    def bands(lyr):
        y = S.render_layer({"type": "sing", "voice": "tenor", "steps": "-:1 A3:2", "lyrics": lyr, "bpm": 120,
                            "vibrato": 0}, SR)[0].mean(axis=1)
        Y = np.fft.rfft(y)
        f = np.fft.rfftfreq(len(y), 1 / SR)
        env = lambda lo, hi: rms_env(np.fft.irfft(np.where((f > lo) & (f < hi), Y, 0), len(y)), ms=2.0)
        return env(120, 330), env(500, 1500), env(3500, 20000)       # H1 (voicing), open vowel, burst / frication

    def voicing_onset(lyr, f0=220.0):
        """First moment the signal repeats at the pitch period (normalised autocorrelation > 0.7)."""
        y = S.render_layer({"type": "sing", "voice": "tenor", "steps": "-:1 A3:2", "lyrics": lyr, "bpm": 120,
                            "vibrato": 0}, SR)[0].mean(axis=1)
        P = int(round(SR / f0))
        ref = np.sqrt(np.mean(y[int(0.8 * SR):int(1.2 * SR)] ** 2))
        for s0 in range(int(0.3 * SR), int(0.7 * SR), 44):
            a, b = y[s0:s0 + 2 * P], y[s0 + P:s0 + 3 * P]
            if np.sqrt(np.mean(a * a)) > 0.03 * ref and (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-20) > 0.7:
                return s0 / SR
        return 9.9

    for lyr in ("ta", "sa", "ka", "pa"):
        e_h1, _, e_hi = bands(lyr)
        onset = voicing_onset(lyr)
        pre = e_hi[int(0.40 * 500):int(0.497 * 500)].max()
        vow = np.median(e_hi[400:600])
        quiet = e_hi[50:150].max()
        t_pre = (int(0.40 * 500) + int(np.argmax(e_hi[int(0.40 * 500):int(0.497 * 500)]))) * 2
        print(f"  '{lyr}': voicing onset at {1000 * onset:.0f} ms (beat 500); noise peak at {t_pre} ms, "
              f"{20 * np.log10(pre / vow):+.1f} dB re the vowel's high band; nothing before ({20 * np.log10(quiet / pre + 1e-9):.0f} dB)")
        assert abs(onset - 0.5) < 0.015, (lyr, onset)
        assert pre > 2.0 * vow and quiet < 0.1 * pre, (lyr, pre, vow, quiet)
    e_h1, e_f1, _ = bands("ma")
    murmur = np.median(e_h1[int(0.455 * 500):int(0.49 * 500)]) / np.median(e_h1[400:600])
    jump = 20 * np.log10(np.median(e_f1[int(0.53 * 500):int(0.56 * 500)]) / np.median(e_f1[int(0.455 * 500):int(0.49 * 500)]))
    print(f"  'ma': murmur before the beat at {20 * np.log10(murmur):.1f} dB (first harmonic), mouth opens +{jump:.1f} dB in 500-1500 Hz across the beat")
    assert murmur > 0.1 and jump > 8


def test_phrase_length_is_exact_in_beats():
    for bpm, sr in ((90, 44100), (132, 48000), (60, 22050)):
        y, _ = S.render_layer({"type": "sing", "section": "altos", "singers": 3, "steps": "C4:1 D4:.5 E4:1.5 -:1 G4:2",
                               "lyrics": "A-ve Ma-ri", "bpm": bpm}, sr)
        want = int(round(6 * 60 / bpm * sr))
        assert y.shape == (want, 2), (y.shape, want)
        assert np.abs(y[-int(0.01 * sr):]).max() < 0.02 * np.abs(y).max()        # released before the end
    y, _ = S.render_layer({"type": "sing", "voice": "bass", "steps": "C3:1", "lead": 0.2, "tail": 0.5}, 44100)
    assert y.shape[0] == int(round(1.7 * 44100))
    print("  sing layer length == beats * 60 / bpm * sr at 3 tempi / sample rates")


def test_legato_glides_without_a_gap():
    notes = [(0.0, 55, 1.2, 1.0), (1.2, 62, 1.2, 1.0)]
    f_a, f_b = hz(55), hz(62)
    res = {}
    for name, kw in (("legato", dict(legato=True)), ("detached", dict(legato=False)), ("staccato", dict(legato=False, articulation="staccato"))):
        y = C.sing_phrase(notes, SR, voice="tenor", lyrics="ah", vibrato=0, stereo=False, **kw)
        e = rms_env(y, ms=10.0)
        res[name] = (float(e[105:135].min() / np.median(e[60:100])), y)
    print(f"  level dip at the note change (min / sustained): legato {res['legato'][0]:.2f}, detached {res['detached'][0]:.2f}, "
          f"staccato {res['staccato'][0]:.3f}")
    assert res["legato"][0] > 0.6 and res["detached"][0] < 0.5 and res["staccato"][0] < 0.05
    y = res["legato"][1]
    Y = np.fft.rfft(y)
    f = np.fft.rfftfreq(len(y), 1 / SR)
    Y[(f < f_a * 0.8) | (f > f_b * 1.25)] = 0
    ph = np.unwrap(np.angle(hilbert(np.fft.irfft(Y, len(y)))))
    inst = np.convolve(np.diff(ph) * SR / (2 * np.pi), np.ones(441) / 441, mode="same")
    seg = inst[int(1.0 * SR):int(1.45 * SR)]
    inside = np.mean((seg > f_a * 1.02) & (seg < f_b * 0.98))            # time spent between the two pitches
    t10 = np.flatnonzero(seg > f_a * (f_b / f_a) ** 0.1)[0] / SR
    t90 = np.flatnonzero(seg > f_a * (f_b / f_a) ** 0.9)[0] / SR
    print(f"  portamento 10-90 % in {1000 * (t90 - t10):.0f} ms, centred {1000 * (0.5 * (t10 + t90) + 1.0 - 1.2):+.0f} ms from the beat")
    assert 0.03 < t90 - t10 < 0.3 and inside * 0.45 > 0.03
    assert abs(np.median(inst[int(0.5 * SR):int(0.9 * SR)]) / f_a - 1) < 0.01 and abs(np.median(inst[int(1.7 * SR):int(2.1 * SR)]) / f_b - 1) < 0.01


def test_hum_and_throat_and_falsetto():
    f0 = hz(48)
    a = mid(solo(48, 2.0, voice="tenor", lyrics="ah", vibrato=0))
    m = mid(solo(48, 2.0, voice="tenor", lyrics="mm", vibrato=0))
    print(f"  energy above 1 kHz: 'ah' {band_db(a, 1000, 8000):.1f} dB, hum {band_db(m, 1000, 8000):.1f} dB")
    assert band_db(m, 1000, 8000) < band_db(a, 1000, 8000) - 10
    # overtone singing: the strongest partial above 800 Hz is the requested harmonic
    for h in (8, 10, 12):
        f0 = hz(38)
        y = mid(I.render_note("throat_singing", f0, 2.5, SR, 0.9, harmonic=h))
        lv = [harmonic_db(y, f0, k) for k in range(1, 20)]
        top = 6 + int(np.argmax(lv[5:]))
        prom = lv[top - 1] - max(lv[top - 3], lv[top + 1])
        print(f"  throat singing harmonic {h}: strongest upper partial {top}, {prom:.1f} dB over its neighbours two away")
        assert top == h and prom > 6
    # falsetto: a man above his break has far weaker upper partials than in chest voice
    f0 = hz(74)
    chest = mid(solo(74, 2.0, voice="tenor", vibrato=0, falsetto=False))
    fal = mid(solo(74, 2.0, voice="tenor", vibrato=0, mode="falsetto"))
    auto = mid(solo(74, 2.0, voice="tenor", vibrato=0))
    d = [harmonic_db(x, f0, 1) - max(harmonic_db(x, f0, k) for k in (3, 4, 5, 6)) for x in (chest, auto, fal)]
    print(f"  tenor D5, H1 over the strongest of H3-H6: chest {d[0]:.1f} dB, automatic register {d[1]:.1f} dB, falsetto {d[2]:.1f} dB")
    assert d[2] > d[0] + 6 and d[1] > d[0] + 3


def test_satb_layer_and_voice_leading():
    lines = C.voice_lead([[60, 64, 67], [65, 69, 72], [67, 71, 74, 77], [60, 64, 67]])
    for chord, col in zip(([0, 4, 7], [5, 9, 0], [7, 11, 2, 5], [0, 4, 7]), zip(*lines)):
        s, a, t, b = col
        assert s > a > t > b and b % 12 == chord[0] and {int(x) % 12 for x in col} <= set(chord), col
        assert len({int(x) % 12 for x in col}) >= 3
    leap = max(abs(a - b) for ln in lines[:3] for a, b in zip(ln, ln[1:]))
    print(f"  I-IV-V7-I voiced: S {lines[0]} A {lines[1]} T {lines[2]} B {lines[3]} (largest upper-voice step {leap:.0f} semitones)")
    assert leap <= 5
    y, _ = S.render_layer({"type": "satb", "chords": "C4:maj:2 F4:maj:2 G4:dom7:2 C4:maj:2", "lyrics": "A-ve Ma-ri",
                           "singers": 3, "bpm": 120, "seed": 2}, SR)
    assert y.shape == (4 * SR, 2) and np.all(np.isfinite(y)) and 0.01 < np.abs(y).max() <= 1.0
    seg = y[int(0.4 * SR):int(0.9 * SR)].mean(axis=1)
    for mnote in (lines[0][0], lines[3][0]):                        # soprano and bass notes are both present
        assert harmonic_db(seg, hz(mnote), 1) > harmonic_db(seg, hz(mnote) * 1.06, 1) + 6
    y2, _ = S.render_layer({"type": "satb", "soprano": "E5:2 F5:2", "alto": "C5:2 C5:2", "tenor": "G4:2 A4:2",
                            "bass": "C3:2 F3:2", "lyrics": "A-men", "singers": 2, "bpm": 120}, SR)
    assert y2.shape == (2 * SR, 2)


# ------------------------------------------------------------------ contract: every name, any rate
def test_deterministic_by_seed():
    a = I.render_note("vocal_choir", 220.0, 0.5, SR, 0.8, seed=3)
    b = I.render_note("vocal_choir", 220.0, 0.5, SR, 0.8, seed=3)
    c = I.render_note("vocal_choir", 220.0, 0.5, SR, 0.8, seed=4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    lay = {"type": "sing", "section": "sopranos", "singers": 4, "steps": "C5:1 D5:1", "lyrics": "glo-ri", "seed": 5}
    assert np.array_equal(S.render_layer(lay, SR)[0], S.render_layer(dict(lay), SR)[0])
    assert not np.array_equal(S.render_layer(lay, SR)[0], S.render_layer(dict(lay, seed=6), SR)[0])


def test_every_name_any_rate_finite_bounded():
    for sr in (22050, 44100, 48000):
        for name in NAMES:
            e = I.REGISTRY[name]
            lo, hi = (note_to_midi(n) for n in e["range"].split("-"))
            for m in (lo - 5, lo, (lo + hi) // 2, hi, hi + 7):
                for dur, vel in ((0.03, 1.0), (0.1234, 0.3), (0.5, 0.9)):
                    y = I.render_note(name, midi_to_freq(m), dur, sr, vel)
                    assert y.ndim == 1 and np.all(np.isfinite(y)), (name, m, dur, sr)
                    assert 1e-4 < np.max(np.abs(y)) < 1.5, (name, m, dur, sr, float(np.max(np.abs(y))))
                    assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3, (name, "starts or ends on a step")
                    assert abs(y.mean()) < 0.01, (name, "dc")
        for layer in ({"type": "sing", "voice": "tenor", "steps": "C3:.5 E3:.5 G3:1", "lyrics": "Ky-ri-e", "dynamics": "p<ff"},
                      {"type": "sing", "section": "mixed", "singers": 4, "steps": "C4:1 -:.5 C4:.5", "lyrics": "sanc-tus", "style": "epic", "breaths": 1, "lead": 0.3},
                      {"type": "sing", "voice": "soprano", "steps": "A4:.5 C5:.5", "lyrics": "ah", "mode": "whisper"},
                      {"type": "sing", "voice": "bass", "steps": "C2:2", "overtones": "8 10 9 12", "sub": 0.3},
                      {"type": "sing", "section": "women", "singers": 3, "steps": "D4:.5 D#4:.5", "style": "bulgarian", "articulation": "marcato", "lyrics": "e"},
                      {"type": "sing", "section": "sopranos", "singers": 2, "steps": "C5:.25 C5:.25", "articulation": "staccato", "lyrics": "la la", "lang": "en", "style": "pop"},
                      {"type": "satb", "chords": "A3:min:1 E3:maj:1", "lyrics": "a-men", "singers": 2, "style": "gospel"}):
            y, _ = S.render_layer(dict(layer, bpm=100), sr)
            assert y.ndim == 2 and np.all(np.isfinite(y)) and 1e-4 < np.abs(y).max() < 1.5, (layer, sr)
            assert np.abs(y[0]).max() < 1e-3 and np.abs(y[-1]).max() < 1e-3 and np.abs(y.mean(axis=0)).max() < 0.01, layer
    try:
        S.render_layer({"type": "sing", "steps": "C4:1", "vowle": "ah"}, SR)
        raise AssertionError("unknown key accepted")
    except S.SpecError:
        pass


def test_instruments_in_tune_level_not_piercing():
    """tests/test_instruments.py's three checks for the instruments registered here."""
    for name in NAMES:
        e = I.REGISTRY[name]
        lo, hi = (note_to_midi(n) for n in e["range"].split("-"))
        cents = [float(cents_off(I.render_note(name, midi_to_freq(m), 0.6, SR, 0.9), midi_to_freq(m))) for m in (lo, (lo + hi) // 2, hi)]
        lev = [loudness(I.render_note(name, midi_to_freq(m), 0.5, SR, 0.9), SR, window=0.15)
               for m in (lo + (hi - lo) // 4, (lo + hi) // 2, hi - (hi - lo) // 4)]
        sh = []
        for m in sorted({min((lo + hi) // 2, 96), min(hi, 96)}):
            f = midi_to_freq(m)
            y = I.render_note(name, f, 0.5, SR, 0.9)
            pure = sharpness(I.render_note("sine", f, 0.5, SR, 0.9), SR)
            s, b = sharpness(y, SR), brightness(y, SR)
            sh.append((round(s, 2), round(pure, 2), round(b["above_5k"], 4)))
            assert s < max(pure + 1.0, 1.5) and s < 2.4, (name, m, round(s, 2), round(pure, 2))
            assert b["above_5k"] < 0.03, (name, m, round(b["above_5k"], 3))
        print(f"  {name:15s} cents at lo/mid/hi {np.round(cents, 1)}; loudness {np.round(lev, 1)} dB; (sharpness, pure tone, share > 5 kHz) {sh}")
        assert max(abs(c) for c in cents) < 10, (name, cents)
        assert abs(sum(lev) / 3 + 12.0) < 6.0, (name, lev)
        for dur in (0.03, 0.2, 1.0):                                  # in tune for short notes as well
            f = midi_to_freq((lo + hi) // 2)
            y = I.render_note(name, f, dur, SR, 0.9)
            if dur >= 0.2:
                assert abs(cents_off(np.concatenate([np.zeros(int(0.05 * SR)), y[int(0.06 * SR):int(0.06 * SR) + int(0.12 * SR)], np.zeros(16384)]), f)) < 25, (name, dur)


def test_speed():
    solo(57, 1.0, voice="tenor")
    t = time.time()
    solo(57, 1.0, voice="tenor")
    t1 = time.time() - t
    t = time.time()
    C.sing_phrase([(0, 57, 1.0, 1)], SR, section="tenors", singers=8)
    t8 = time.time() - t
    print(f"  1 s of solo voice in {t1:.3f} s, 1 s of 8 singers in {t8:.3f} s")
    assert t1 < 2.0 and t8 < 5.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            print(name)
            fn()
            print(f"ok {name} ({time.time() - t0:.1f} s)")
