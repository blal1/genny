"""Bodies and crowds: every variant renders without a click; the figures of the spec and the sourced numbers are measured on the rendered signal."""
import numpy as np
import pytest
from scipy.signal import butter, hilbert, sosfilt

from genny import creatures as C
from genny import human as H
from genny import matter as M
from genny.analysis import detect_clicks
from genny.percussion import snap
from genny.sfx import REGISTRY

SR = 44100
HI = 48000


def spec(y, sr):
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    return np.fft.rfftfreq(len(y), 1 / sr), Y


def top(y, sr, lo, hi):
    f, Y = spec(y, sr)
    m = (f >= lo) & (f <= hi)
    return f[m][np.argmax(Y[m])]


def share(y, sr, lo, hi):
    f, Y = spec(y, sr)
    return Y[(f >= lo) & (f < hi)].sum() / Y.sum()


def centroid(y, sr):
    f, Y = spec(y, sr)
    return (f * Y).sum() / Y.sum()


def env_top(y, sr, lo, hi):
    return top(np.abs(y) - np.mean(np.abs(y)), sr, lo, hi)


def f0_est(y, sr, lo, hi):
    y = y - y.mean()
    c = np.fft.irfft(np.abs(np.fft.rfft(y, 2 * len(y))) ** 2)[:len(y)]
    a, b = int(sr / hi), int(sr / lo)
    return sr / (a + np.argmax(c[a:b]))


def periodicity(y, sr, lo, hi):
    y = y - y.mean()
    c = np.fft.irfft(np.abs(np.fft.rfft(y, 2 * len(y))) ** 2)[:len(y)]
    return c[int(sr / hi):int(sr / lo)].max() / c[0]


def envelope(y, sr, ms=5.0):
    k = max(1, int(ms * 1e-3 * sr))
    return np.convolve(np.abs(y), np.ones(k) / k, "same")


def span(y, sr, db=-20.0, ms=2.0):
    e = envelope(y, sr, ms)
    on = np.flatnonzero(e > e.max() * 10 ** (db / 20))
    return (on[-1] - on[0]) / sr


def clean(y):
    assert np.isfinite(y).all() and 0.01 < np.abs(y).max() <= 1.0
    assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3


NO_CLICK = [(H.burp, {}), (H.yawn, {}), (H.gargle, {}), (H.hiccup, {}), (H.bodyfall, {})]
NO_CLICK += [(H.shout, dict(kind=k)) for k in ("cri de guerre", "hurlement", "encouragement")] + [(H.baby, dict(kind=k)) for k in H.BABY]
NO_CLICK += [(H.blow, dict(kind=k)) for k in H.BLOWS] + [(H.cartoon_voice, dict(kind="ecureuil")), (H.finger_snap, {})]
NO_CLICK += [(H.skin, dict(kind=k)) for k in H.SKIN] + [(H.whisper_bed, dict(voices=2, dur=2.0)), (H.yodel, {})]
NO_CLICK += [(H.crowd, dict(kind=k, voices=4, dur=2.5)) for k in ("oh", "ah", "huee", "chants", "enfants")]


def _id(c):
    return f"{c[0].__name__}-{'-'.join(map(str, c[1].values()))}"


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("case", NO_CLICK, ids=_id)
def test_variant_renders_without_click(case, seed):
    kw = dict(case[1]) if case[0] is H.finger_snap else dict(case[1], seed=seed)
    y = np.asarray(case[0](sr=SR, **kw))
    clean(y)
    assert len(detect_clicks(y)) == 0


def test_names_in_the_catalogue():
    for n in "burp fart kiss spit vomit pee crowd yawn gargle hiccup beatbox yodel rap bodyfall shout baby blow cartoon_voice finger_snap skin whisper_bed".split():
        assert n in REGISTRY


# Sounds with attacks of their own: every detection must be explained by where it falls, and its cause checked.
@pytest.mark.parametrize("kind", list(H._FARTS))
def test_fart_detections_come_from_the_gated_jet_and_not_from_the_valve(kind):
    """The valve alone (jet off) has no detection: the opening is smooth. What is left are noise peaks inside each short opening of the jet: in sound, never in the closed phase."""
    d, pts, jet, cav, sput = H._FARTS[kind]
    y = H.fart(sr=SR, kind=kind)
    clean(y)
    valve = H._fin(H._flap(d, pts, jet, cav, 0, sput, 1.0, jet_g=0.0), SR, 0.7, fade=0.0005, tail=0.03)
    assert len(detect_clicks(valve)) == 0
    e = envelope(y, SR, 8.0)
    c = detect_clicks(y)
    assert np.all(e[c] > 0.03 * e.max())


@pytest.mark.parametrize("kind", list(H.KISSES))
def test_kiss_detections_are_the_rupture(kind):
    """A 1-5 ms burst is alone in the detector's 20 ms window: only there, nowhere in the suction or the silence."""
    y = H.kiss(sr=HI, kind=kind)
    clean(y)
    ts = 0.1 + 0.42 * np.arange(H.KISSES[kind][2])
    c = detect_clicks(y) / HI
    assert all(np.any((u >= ts - 0.001) & (u <= ts + 0.02)) for u in c)


def test_spit_detections_are_the_drop_impacts():
    h = 1.0
    y = H.spit(sr=SR, height=h)
    clean(y)
    t = 0.11 + np.sqrt(2 * h / 9.81)
    c = detect_clicks(y) / SR
    assert np.all((c > t - 0.005) & (c < t + 0.08))     # the drip's impact pulse, click=0.5 of its own


def test_beatbox_detections_lie_on_the_grid():
    y = H.beatbox(sr=SR, bpm=100)
    clean(y)
    step = 0.3
    u = (detect_clicks(y) / SR - 0.02) % step
    assert np.all((u < 0.04) | (u > step - 0.002))      # the stops themselves


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("f,kw", [(H.cartoon_voice, dict(kind="aigu")), (H.cartoon_voice, dict(kind="souris")), (H.rap, {})], ids=["aigu", "souris", "rap"])
def test_speech_detections_are_glottal_pulses(f, kw, seed):
    """Speech alone (`render_speech("level")`) on its L: narrow pulses of a low first formant. Spaced by the pitch period (integer multiples), so a voice, not a defect in time."""
    y = np.asarray(f(sr=SR, seed=seed, **kw))
    clean(y)
    c = detect_clicks(y)
    assert np.all(envelope(y, SR, 10.0)[c] > 0.03 * np.abs(y).max())      # never in the gaps
    near = np.diff(np.unique(c))
    near = near[(near > 8) & (near < 0.03 * SR)]
    if len(near) >= 3:
        period = np.median(near)
        assert np.all(np.abs(near / period - np.round(near / period)) < 0.25)


@pytest.mark.parametrize("seed", [0, 1])
def test_vomit_detections_lie_in_the_retching_and_the_splash(seed):
    """`strain` (Pink Trombone) has pulses the detector flags on some seeds alone, the splash its slap. None in the two silences between."""
    y = H.vomit(sr=SR, seed=seed)
    clean(y)
    assert any(len(detect_clicks(H.strain(sr=SR, dur=0.55, freq=120.0, seed=s))) for s in range(4))
    c = detect_clicks(y) / SR
    assert not np.any((c > 0.47) & (c < 0.6))
    assert not np.any(c > 2.8)


@pytest.mark.parametrize("kind", ["ovation", "combat", "panique"])
def test_crowd_detections_fall_on_hits_or_on_applause(kind):
    y, hits = H.crowd_track(kind, 5, 2.5, 0)
    r = H.crowd(sr=SR, kind=kind, voices=5, dur=2.5)
    clean(r)
    c = detect_clicks(r) / SR
    claps = detect_clicks(C.applause(sr=SR, crowd=25, enthusiasm=0.9, dur=2.5)) / SR if kind == "ovation" else np.zeros(0)     # same seed, same mix place
    for u in c:
        on_hit = any(t - 0.002 <= u <= t + d for t, d in hits)
        assert on_hit or np.any(np.abs(claps - u) < 0.005), u


def test_pee_detections_are_the_bubbles_of_pour():
    """A bubble in the bed of a thin stream, sharp at 22 kHz only: none at the rate it is synthesised, and at 22 kHz the same times come from `pour` alone."""
    assert len(detect_clicks(H.pee(sr=HI, dur=4.0))) == 0
    y = H.pee(sr=SR, dur=4.0)
    clean(y)
    pour = detect_clicks(M.pour(sr=SR, dur=4.0, flow=0.25, fill_start=0.5, fill_end=0.55, height=0.18, radius=0.1, material="ceramic", resonance=0.25)) / SR
    assert all(np.any(np.abs(pour - u) < 0.005) for u in detect_clicks(y) / SR)


def test_bodyfall_on_snow_detections_are_the_grains_of_its_footfall():
    """The crunch of a granular ground: `_step` on snow alone has them, and they stay within the impact and the settling."""
    assert len(detect_clicks(C._step("snow", 0, "walk", 0.6, 1.0, "barefoot", 0.5, 0))) > 0
    y = H.bodyfall(sr=SR, ground="snow")
    clean(y)
    c = detect_clicks(y) / SR
    assert np.all((c >= 0.2 - 0.002) & (c <= 1.0))


def test_gurgle_brick_no_longer_starts_its_bubbles_on_a_step():
    """`vomit` no longer low-passes it."""
    assert len(detect_clicks(M.gurgle(sr=SR, dur=1.0, rate=6.0, size=0.2))) == 0


# Spec figures, measured
@pytest.mark.parametrize("size", [0.7, 1.0, 2.1])
def test_burp_pulses_40_120_hz(size):
    f = f0_est(H.burp(sr=HI, size=size), HI, 30.0, 160.0)
    assert 40.0 <= f <= 120.0
    assert f0_est(H.burp(sr=HI, size=0.7), HI, 30, 160) > f0_est(H.burp(sr=HI, size=2.1), HI, 30, 160)


@pytest.mark.parametrize("kind", list(H._FARTS))
def test_fart_flutter_30_120_hz(kind):
    y = H.fart(sr=SR, kind=kind)
    f = f0_est(y, SR, 20.0, 200.0)
    assert 30.0 <= f <= 120.0
    assert 30.0 <= env_top(y, SR, 20.0, 200.0) <= 120.0


def test_fart_is_lower_than_the_balloon():
    assert f0_est(H.fart(sr=SR, kind="reel"), SR, 20, 300) < f0_est(M.balloon(sr=SR), SR, 20, 300)


def test_bodyfall_contact_15_ms():
    """`thud` gets 15; its sack body turns that into 9 ms of contact (contact_ms * 0.6, foley.thud). The rise of the rendered hit is what is measured."""
    y = H.FO.thud(sr=HI, body="sack", contact_ms=15.0, size=0.8, force=0.9)
    e = envelope(y, HI, 1.0)
    assert (np.argmax(e) - np.flatnonzero(e > 0.1 * e.max())[0]) / HI <= 0.015


@pytest.mark.parametrize("ms", [1.0, 3.0, 5.0])
def test_kiss_seal_breaks_in_1_to_5_ms(ms):
    y = H.kiss(sr=HI, kind="bise", seal=ms)
    i = np.argmax(np.abs(y))
    e = np.abs(hilbert(y[i - int(0.01 * HI):i + int(0.02 * HI)]))
    on = np.flatnonzero(e > 0.5 * e.max())
    rise = (np.argmax(e) - np.flatnonzero(e > 0.1 * e.max())[0]) / HI * 1e3
    assert rise <= 5.0 and (on[-1] - on[0]) / HI * 1e3 <= 2.5 * ms + 2.0


def test_kiss_seal_longer_is_longer():
    w = [span(H.kiss(sr=HI, kind="bise", seal=m), HI, -12.0, 0.5) for m in (1.0, 5.0)]
    assert w[1] > w[0]


@pytest.mark.parametrize("rate", [10.0, 15.0, 20.0])
def test_gargle_rate(rate):
    y = H.gargle(sr=SR, rate=rate)
    assert abs(env_top(y, SR, 5.0, 40.0) - rate) < 1.0


def test_hiccup_lasts_50_ms():
    y = H.hiccup(sr=HI, count=1)
    assert abs(span(y, HI, -30.0, 0.5) - H.HICCUP) < 0.006


def test_hiccup_closure_is_sudden():
    y = H.hiccup(sr=HI, count=1)
    e = envelope(y, HI, 1.0)
    on = np.flatnonzero(e > 0.03 * e.max())
    assert (on[-1] - np.flatnonzero(e > 0.3 * e.max())[-1]) / HI < 0.005


def test_bodyfall_contact_rebound_ground():
    y = H.bodyfall(sr=SR, ground="wood")
    e = envelope(y, SR, 5.0)
    t = int(0.2 * SR)
    prim = e[t:t + int(0.08 * SR)].max()
    reb = e[t + int(0.125 * SR):t + int(0.17 * SR)].max()
    assert prim == e.max() and reb < 0.5 * prim            # rebond faible
    assert centroid(y, SR) != centroid(H.bodyfall(sr=SR, ground="snow"), SR)
    light, heavy = H.bodyfall(sr=SR, weight=0.5), H.bodyfall(sr=SR, weight=2.0)
    assert centroid(heavy, SR) < centroid(light, SR)
    with pytest.raises(ValueError):
        H.bodyfall(sr=SR, ground="lava")


def test_baby_f0_x2_and_tract_x1_6():
    y = H.baby(sr=SR, kind="pleurs")
    m = y[int(0.15 * SR):int(0.6 * SR)]
    f = f0_est(m, SR, 200.0, 1000.0)
    assert 250.0 <= f <= 700.0                              # phonated cry, arXiv 1009.2796
    a = np.asarray(REGISTRY["cry"]["fn"](sr=SR, voice="male"))
    fa = f0_est(a[int(0.15 * SR):int(0.6 * SR)], SR, 150.0, 500.0)
    assert abs(f / fa - 2.0) < 0.2
    assert H.INFANT_F == pytest.approx(13.5 / 1.6)
    assert centroid(y, SR) > 1.3 * centroid(a, SR)


@pytest.mark.parametrize("kind,lo,hi", [("rire", 280, 420), ("gazouillis", 330, 560)])
def test_baby_other_kinds_f0(kind, lo, hi):
    assert lo <= f0_est(H.baby(sr=SR, kind=kind), SR, 200.0, 800.0) <= hi


@pytest.mark.parametrize("kind", ["cri de guerre", "hurlement"])
def test_shout_roughness_in_arnal_window(kind):
    y = H.shout(sr=SR, kind=kind)
    m = y[int(0.2 * SR):int(0.9 * SR)]
    assert 30.0 <= env_top(m, SR, 25.0, 200.0) <= 150.0


def test_shout_kinds():
    war = H.shout(sr=SR, kind="cri de guerre")
    assert abs(f0_est(war[int(0.4 * SR):int(0.8 * SR)], SR, 250.0, 600.0) - 410.0) < 45.0
    howl = H.shout(sr=SR, kind="hurlement")
    assert abs(f0_est(howl[int(0.4 * SR):int(1.2 * SR)], SR, 1000.0, 2500.0) - 1700.0) < 250.0
    assert len(H.shout(sr=SR, kind="encouragement")) < len(war)
    moan = np.asarray(REGISTRY["moan"]["fn"](sr=SR, voice="male"))
    assert share(war, SR, 2000, 8000) > share(moan, SR, 2000, 8000)            # pressed


def test_shout_group_is_many():
    one = H.shout(sr=SR, kind="encouragement")
    assert periodicity(H.shout(sr=SR, kind="encouragement", people=4), SR, 150, 600) < periodicity(one, SR, 150, 600)


def test_blow_kinds():
    can, ball, slow = (H.blow(sr=SR, kind=k) for k in H.BLOWS)
    assert span(can, SR) < 0.5 < 1.2 < span(slow, SR, -30.0)
    b = [top(ball[int((0.02 + i * 0.95) * SR):int((0.62 + i * 0.95) * SR)], SR, 200.0, 2000.0) for i in range(3)]
    assert np.mean(b[:2]) > b[2]                 # the cavity falls as it fills


@pytest.mark.parametrize("kind", list(H.CARTOONS))
def test_cartoon_voice_is_speech_at_speed(kind):
    f = H.CARTOONS[kind]
    base = H.render_speech("hello there", sr=SR, voice="male", intonation="flat")
    y = H.cartoon_voice(sr=SR, kind=kind)
    assert len(y) / len(base) == pytest.approx(1 / f, abs=0.02)
    b = f0_est(base, SR, 70.0, 200.0)
    assert abs(f0_est(y, SR, 70.0 * f, 200.0 * f) / b - f) < 0.2 * f


def test_finger_snap_is_the_drum():
    assert np.array_equal(H.finger_snap(sr=SR), snap(sr=SR, vel=1.0))


def test_skin_kinds():
    g = H.skin(sr=SR, kind="grattage", rate=4.0, count=4)
    assert abs(env_top(g, SR, 1.0, 12.0) - 4.0) < 1.5
    t = H.skin(sr=SR, kind="tape", rate=4.0, count=4)
    e = envelope(t, SR, 5.0)
    p = np.flatnonzero((e[1:-1] > e[:-2]) & (e[1:-1] >= e[2:]) & (e[1:-1] > 0.3 * e.max()))
    assert 3 <= len(np.unique(np.round(p / SR / 0.25))) <= 5
    assert centroid(t, SR) < centroid(g, SR)


def test_whisper_bed_has_no_pitch_and_voices_differ():
    y = H.whisper_bed(sr=SR, voices=2, dur=2.0)
    assert periodicity(y, SR, 80.0, 300.0) < 0.35
    assert abs(np.corrcoef(y, H.whisper_bed(sr=SR, voices=2, dur=2.0, seed=1))[0, 1]) < 0.15


def test_crowd_voices_are_decorrelated():
    rng = np.random.default_rng(3)
    ys = H._voices(["h_oh"], 4, rng)
    n = min(map(len, ys))
    c = np.abs(np.corrcoef([y[:n] for y in ys]) - np.eye(4))
    assert c.max() < 0.2
    assert len({round(f0_est(y[:n], HI, 80, 400), 0) for y in ys}) >= 3


@pytest.mark.parametrize("kind", ["oh", "ah"])
def test_crowd_blurs_the_pitch_of_one(kind):
    one = H.crowd(sr=SR, kind=kind, voices=1)
    many = H.crowd(sr=SR, kind=kind, voices=6)
    assert periodicity(many, SR, 80, 400) < periodicity(one, SR, 80, 400)
    assert 1.0 < len(many) / SR < 2.0


def test_crowd_kinds():
    assert centroid(H.crowd(sr=SR, kind="huee", voices=4, dur=2.5), SR) < centroid(H.crowd(sr=SR, kind="ah", voices=4, dur=2.5), SR)
    p = H.crowd(sr=SR, kind="panique", voices=4, dur=2.5)
    assert 30.0 <= env_top(p, SR, 25.0, 200.0) <= 150.0     # Arnal, again
    ch = H.crowd(sr=SR, kind="chants", voices=3, dur=4.0)
    e = envelope(ch, SR, 40.0)
    e = e - e.mean()
    c = np.fft.irfft(np.abs(np.fft.rfft(e, 2 * len(e))) ** 2)[:len(e)]
    assert abs(np.argmax(c[int(1.5 * SR):int(2.5 * SR)]) + int(1.5 * SR) - 2 * SR) / SR < 0.15   # one "o-le-o" per 2 s
    with pytest.raises(ValueError):
        H.crowd(sr=SR, kind="nope")


def test_ovation_has_applause_band():
    y = H.crowd(sr=SR, kind="ovation", voices=4, dur=2.5)
    assert share(y, SR, 500, 4000) > 0.4


def test_spit_drops_land_after_the_fall():
    for h in (0.5, 2.0):
        y = H.spit(sr=SR, height=h)
        t = 0.11 + np.sqrt(2 * h / 9.81)
        j = y[:int(0.18 * SR)]
        assert centroid(j, SR) > 2000.0
        e = envelope(y, SR, 2.0)
        after = e[int(0.25 * SR):]
        on = 0.25 + np.flatnonzero(after > 0.1 * after.max())[0] / SR
        assert abs(on - t) < 0.03


def test_vomit_order():
    y = H.vomit(sr=SR)
    assert abs(f0_est(y[int(0.65 * SR):int(1.05 * SR)], SR, 80.0, 180.0) - 116.0) < 15.0   # strain, 120 Hz falling 6 %
    assert np.sqrt(np.mean(y[int(1.3 * SR):int(2.0 * SR)] ** 2)) > np.sqrt(np.mean(y[int(0.02 * SR):int(0.4 * SR)] ** 2))
    assert share(y[int(1.3 * SR):int(2.0 * SR)], SR, 1000, 8000) > share(y[int(0.65 * SR):int(1.05 * SR)], SR, 1000, 8000)


def test_pee_is_pour_range():
    y = H.pee(sr=SR, dur=3.0)
    assert share(y, SR, 300, 6000) > 0.9                    # HF Table 1 pouring, in matter.pour
    assert 2.9 < len(y) / SR < 3.1


def test_yawn_starts_unvoiced_then_falls():
    y = H.yawn(sr=SR)
    a = periodicity(y[int(0.1 * SR):int(0.6 * SR)], SR, 80, 300)
    b = periodicity(y[int(1.1 * SR):int(1.6 * SR)], SR, 80, 300)
    assert b > a
    assert f0_est(y[int(1.1 * SR):int(1.5 * SR)], SR, 80, 300) > f0_est(y[int(2.3 * SR):int(2.7 * SR)], SR, 80, 300)


def test_beatbox_kinds_are_on_tempo_and_differ():
    r = {}
    for tok in "BtK":
        y = H.beatbox(sr=SR, pattern=f"{tok} - - -", bpm=100)
        r[tok] = centroid(y, SR)
        e = envelope(y, SR, 1.0)
        assert np.flatnonzero(e > 0.1 * e.max())[0] / SR < 0.03
    assert r["B"] < r["K"] < r["t"]
    y = H.beatbox(sr=SR, pattern="B - - - B - - -", bpm=100)
    e = envelope(y, SR, 3.0)
    a = np.flatnonzero(e > 0.1 * e.max())[0]
    g = int(0.9 * SR)
    b = g + np.flatnonzero(e[g:] > 0.1 * e.max())[0]
    assert abs((b - a) / SR - 1.2) < 0.02
    with pytest.raises(ValueError):
        H.beatbox(sr=SR, pattern="B x")


def test_yodel_notes_and_registers():
    y = H.yodel(sr=HI, note=0.3)
    h = {0: [], 1: []}
    for i, (nt, reg) in enumerate(H.YODEL):
        f = H.to_hz(nt)
        m = y[int((0.02 + i * 0.3 + 0.08) * HI):int((0.02 + i * 0.3 + 0.27) * HI)]
        assert abs(1200 * np.log2(f0_est(m, HI, f * 0.8, f * 1.25) / f)) < 60.0     # the note, within 60 cents
        s_f, S = spec(m, HI)
        a = lambda k: S[np.abs(s_f - k * f) < 30].max()
        h[reg].append(a(2) / a(1))
    assert np.median(h[1]) < np.median(h[0])                # falsetto: weaker second harmonic


def test_rap_words_on_the_grid():
    y = H.rap(sr=SR, text="go time go time", bpm=100)
    e = envelope(y, SR, 4.0)
    step = 0.3
    on = [np.flatnonzero(e[int((0.02 + i * step - 0.02) * SR):] > 0.05 * e.max())[0] / SR - 0.02 for i in range(4)]
    assert np.all(np.abs(on) < 0.03)
