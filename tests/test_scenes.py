"""Scenes: every kind renders and holds its layers; the book's contrast and cotton are measured; every click found in a mix is traced to its cause."""
import numpy as np
import pytest

import genny  # noqa: F401
from genny import scenes as Sc
from genny.analysis import detect_clicks
SR = 22050                # as test_household; the full rate costs twice the time, on a machine short of RAM
from genny.physical import ambience as Am
from genny.sfx import REGISTRY, render_sfx

DUR = 4.0


@pytest.fixture(autouse=True)
def one_take(monkeypatch):
    monkeypatch.setattr(Sc, "VARIANTS", 1)      # three takes of each event differ only by seed: the time budget goes to more kinds
KINDS = sorted(Sc.SCENES) + sorted(Am.ACT_BEDS)

# Each cause below was read sample by sample on the render (L20.md). A mix click traced to anything else, or to nothing, fails: the assembly makes none of its own.
ATTACKS = {"firearm", "howitzer", "shell_incoming", "blast", "sand_storm", "fire", "eat", "truck", "slot_machine", "excavator", "saw", "conveyor", "applause", "swarm", "force_field", "foliage", "servo"}    # hits, grains, saw edges of the sound itself; servo: the rotor edge (ph % 1)**4 once per turn at full speed, not servo_env
KNOWN_DEFECTS = {"crowd_murmur"}     # NOT normal: almost_speech's 4 Hz phasor wraps on the same sample in every voice (L20.md, reported to the owner of creatures.py / voice.py)
# Beds: rain (parabolic drops), wind_scene's leaves (thresholded noise), stream grains (1 ms bursts through vcf~).
BEDS_WITH_GRAINS = {"seaside", "nautical", "shore", "fantasy", "forest", "wildfire", "swamp", "tropical", "park", "suburban", "farm", "grassland", "lakeside", "birdsong", "insect", "rural"}


def mono(y):
    return y.mean(axis=1) if y.ndim == 2 else y


def rms(y):
    return float(np.sqrt(np.mean(np.square(y))))


def band_db(y, sr, lo, hi):
    Y = np.abs(np.fft.rfft(mono(y) * np.hanning(y.shape[0]))) ** 2
    f = np.fft.rfftfreq(y.shape[0], 1 / sr)
    return 10 * np.log10(Y[(f >= lo) & (f < hi)].sum() + 1e-30)


def clusters(c, sr):
    return [g[0] for g in np.split(c, np.flatnonzero(np.diff(c) > int(0.002 * sr)) + 1)] if len(c) else []


def stray(y, sr):
    """A cut (a jump over half the level so far, the sound gone after it) or a one-sample spike, from test_nature."""
    y = mono(y)
    c = detect_clicks(y)
    w, bad = int(0.005 * sr), []
    for i in c[(c > 10) & (c < len(y) - 10)]:
        before = y[max(0, i - w):i - 1]
        if len(before) and abs(y[i + 1] - y[i]) > 0.5 * np.abs(before).max() > 0 and rms(y[i + 2:i + 2 + w]) < 0.25 * rms(before):
            bad.append(round(i / sr, 3))
        if abs(y[i]) > 4 * max(np.abs(y[i - 8:i]).max(), np.abs(y[i + 1:i + 9]).max()):
            bad.append(round(i / sr, 3))
    return bad


def test_stray_catches_a_cut():
    t = np.arange(SR) / SR
    assert stray(0.5 * np.sin(2 * np.pi * 200 * t) * (t < 0.5012), SR)


def test_catalogue_covers_every_ambience_cat():
    """All of taxonomie_universelle.md:505-551 but AMBMisc, sans objet there."""
    assert len(Sc.UCS) == 46 and set(Sc.UCS.values()) <= set(Sc.SCENES)
    for s in Sc.SCENES.values():
        for layer in s["layers"]:
            if layer["opt"] or "inst" in layer:
                continue
            assert layer["sfx"] in REGISTRY
            assert set(layer["params"]) <= set(REGISTRY[layer["sfx"]]["params"])


@pytest.mark.parametrize("kind", KINDS)
def test_kind_renders_with_its_layers_and_no_fault(kind):
    b = Sc.build(kind, SR, dur=DUR, seed=0)
    y = b["out"]
    assert y.shape == (int(DUR * SR), 2) and np.isfinite(y).all()
    assert 0.02 < np.abs(y).max() <= 0.9 + 1e-9
    assert abs(y[0]).max() < 1e-3 and abs(y[-1]).max() < 1e-2
    assert stray(y, SR) == []
    for name, part, at in b["layers"]:
        assert rms(part) > 0, name
        if at is not None:
            assert len(at) >= 1, name
    # each click of the mix has a cause among the parts: the assembly (seams, fades, resampling, climax) makes none of its own.
    y_m = mono(y)
    w = int(EDGE_N)
    flagged = [i for i in clusters(detect_clicks(y_m), SR) if w < i < len(y_m) - w]
    if not flagged:
        return
    parts = [(n, p) for n, p, _ in b["layers"]] + [("bed", b["bed"])]     # a sfx may hold several layers (warfare: two firearms)
    own = [(n, detect_clicks(mono(p))) for n, p in parts]
    for i in flagged:
        src = {n for n, c in own if c.size and np.min(np.abs(c - i)) <= 3}
        if not src:      # not over its own part's threshold: the part that makes at least 80 % of the second difference there
            d = [(n, np.abs(np.diff(mono(p)[i - 3:i + 4], 2)).max()) for n, p in parts]
            top = max(d, key=lambda e: e[1])
            if top[1] >= 0.8 * sum(v for _, v in d):
                src = {top[0]}
        assert src & (ATTACKS | KNOWN_DEFECTS) or ("bed" in src and kind in BEDS_WITH_GRAINS), (kind, round(i / SR, 3), src)


EDGE_N = 0.01 * SR


def test_seamless_loop_has_no_seam():
    y = Sc.build("market", SR, dur=DUR, seed=1, loop=1)["out"]
    n = y.shape[0]
    assert rms(y[:2000]) > 0.1 * rms(y)                  # no end fade
    r = mono(np.roll(y, n // 2, axis=0))
    assert not np.any(np.abs(detect_clicks(r) - n // 2) < 200)   # the same detector, across the join


def test_act_scene_is_the_act_bed():
    n = int(DUR * SR)
    y = Sc.build("caravan", SR, dur=DUR, seed=3)["bed"]
    b = Am.ACT_BEDS["caravan"](np.random.default_rng([3, 7]), DUR)
    z = Sc._to_rate(b, n)
    z = z / np.sqrt(np.mean(z ** 2)) * Sc.REF
    assert np.allclose(y, z, atol=1e-9)


def test_place_words_reach_bed_generic():
    y = Sc.build("place", SR, dur=DUR, seed=3, place="sea shore harbour")["bed"]
    z = Am.bed_generic(np.random.default_rng([3, 7]), "sea shore harbour", DUR)
    z = Sc._to_rate(z, int(DUR * SR))
    assert np.allclose(y, z / np.sqrt(np.mean(z ** 2)) * Sc.REF, atol=1e-9)
    with pytest.raises(ValueError):
        Sc.build("place", SR, dur=DUR)
    with pytest.raises(ValueError):
        Sc.build("nowhere", SR, dur=DUR)
    with pytest.raises(ValueError):
        Sc.build("forest", SR, dur=DUR, modifier="noon")


def test_modifier_night_lowpass_at_4k():
    """AUDIO_CHARTER 8.7 through ambience.bed_modifier: 4 kHz, 4th order, on a bed with a wind layer added (-10 dB)."""
    in_scene = Sc.build("place", SR, dur=DUR, seed=2, place="storm rain", modifier="night")["bed"]
    assert np.isfinite(in_scene).all() and rms(in_scene) == pytest.approx(Sc.REF, rel=1e-6)
    x = np.random.default_rng(5).standard_normal((int(DUR * 48000), 2)) * 0.1
    a = Am.bed_modifier(x, "night", np.random.default_rng(2), DUR)
    b = Am.bed_modifier(-x, "night", np.random.default_rng(2), DUR)
    lp, wind = (a - b) / 2, (a + b) / 2                      # same wind in both: it cancels in one, the bed in the other
    def gain(lo, hi):
        return band_db(lp, 48000, lo, hi) - band_db(x, 48000, lo, hi)
    assert gain(3950, 4050) == pytest.approx(-3.0, abs=1.0)  # 1 / sqrt(1 + (f/4000)^8)
    assert gain(7900, 8100) == pytest.approx(-24.1, abs=1.0)
    assert 20 * np.log10(rms(wind) / rms(lp)) == pytest.approx(-10.0, abs=0.1)


def test_ucs_id_equals_name():
    a = Sc.build("AMBHosp", SR, dur=3.0, seed=4)["out"]
    b = Sc.build("hospital", SR, dur=3.0, seed=4)["out"]
    assert np.array_equal(a, b)


def test_sustained_level_is_the_table():
    b = Sc.build("insect", SR, dur=DUR, seed=0)
    for (name, part, _), layer in zip(b["layers"], Sc.SCENES["insect"]["layers"]):
        assert 20 * np.log10(rms(part) / Sc.REF) == pytest.approx(layer["db"], abs=0.01), name
    assert 20 * np.log10(rms(b["bed"]) / Sc.REF) == pytest.approx(0.0, abs=0.01)


def test_density_scales_the_events():
    few = Sc.build("farm", SR, dur=8.0, seed=0, density=0.5)["layers"]
    many = Sc.build("farm", SR, dur=8.0, seed=0, density=3.0)["layers"]
    assert sum(len(a) for _, _, a in many) > 2 * sum(len(a) for _, _, a in few)


# c32 line 26: the quiet passage lowered, 12 s at -1.5 dB, a silence before the hit, four small then one big.
def test_contrast_gain_numbers():
    sr, t0 = 1000, 20.0
    g = Sc.contrast_gain(30 * sr, sr, t0)
    i0 = int(t0 * sr)
    gap = int(Sc.GAP_S * sr)
    ramp = int(Sc.GAP_RAMP_S * sr)
    assert g[i0 - gap - ramp - int(12.0 * sr) - 1] == 1.0          # untouched before the 12 s
    assert 20 * np.log10(g[i0 - gap - ramp - 2]) == pytest.approx(-1.5, abs=0.02)
    assert np.all(g[i0 - gap:i0] == 0.0)
    assert np.all(g[i0 + ramp:] == 1.0)                            # the loud part is not raised nor lowered
    assert np.abs(np.diff(g)).max() < 2 / ramp


def test_warfare_climax_on_the_render():
    dur = 14.0
    b = Sc.build("warfare", SR, dur=dur, seed=0)
    w = int(0.1 * SR)
    for (name, part, at), layer in zip(b["layers"], Sc.SCENES["warfare"]["layers"]):     # by loudest 100 ms: whatever level the sfx comes out at
        loud = 10 * np.log10(np.convolve(np.sum(part ** 2, axis=1) / 2, np.ones(w) / w, "same").max()) - 20 * np.log10(Sc.REF)
        assert layer["db"] - Sc.JITTER_DB - 0.5 < loud < layer["db"] + Sc.JITTER_DB + 6.5, name    # +6: two events may overlap
    y, t0 = mono(b["out"]), b["climax_t"]
    assert t0 == pytest.approx(0.7 * dur)
    i0 = int(t0 * SR)
    gap = y[i0 - int(Sc.GAP_S * SR) + 50:i0 - 5]
    hits = mono(dict((n, p) for n, p, _ in b["layers"])["blast"])
    # only the tails of the small hits may sound in the gap: the place is gone
    bed_gain = Sc.contrast_gain(y.shape[0], SR, t0)
    assert np.all(bed_gain[i0 - int(Sc.GAP_S * SR):i0] == 0.0)
    assert rms(gap) < 0.05 * rms(y[i0:i0 + SR // 2])
    assert np.abs(y).argmax() >= i0                                # the blast is the loudest thing
    w = int(0.1 * SR)
    small = [np.abs(hits[i0 - int((Sc.GAP_S + Sc.GAP_RAMP_S + (Sc.SMALL_HITS - k - 0.5) * Sc.HIT_GAP_S) * SR):][:w]).max() for k in range(Sc.SMALL_HITS)]
    assert all(s > 0 for s in small) and max(small) < 0.5 * np.abs(hits[i0:i0 + w]).max()


@pytest.mark.parametrize("f,db", [(250.0, -5.0), (500.0, -5.0), (1000.0, -10.0), (2000.0, -15.0), (4000.0, -20.0)])
def test_cotton_after_the_blast(f, db):
    """c02: -5 dB below 500 Hz, then 5 dB per octave, measured in the hold, on the signal."""
    sr, t0 = SR, 0.5
    t = np.arange(int(6 * sr)) / sr            # t0 + hold + clearing = 4.5 s
    x = np.sin(2 * np.pi * f * t)
    y = Sc.after_blast(np.stack([x, x], axis=1), sr, t0)
    a, b = int(0.7 * sr), int(1.3 * sr)        # inside the 1 s hold after t0
    assert 20 * np.log10(rms(y[a:b, 0]) / rms(x[a:b])) == pytest.approx(db, abs=0.3)
    assert rms(y[-sr // 2:, 0]) == pytest.approx(rms(x[-sr // 2:]), rel=0.02)        # cleared


@pytest.mark.parametrize("kind", list(Sc.ROOM_TONES))
def test_room_tone_kinds(kind):
    y = render_sfx("room_tone", SR, kind=kind, dur=4.0)
    assert y.shape == (4 * SR, 2) and np.isfinite(y).all()
    assert len(detect_clicks(y)) == 0
    Y = np.abs(np.fft.rfft(mono(y) * np.hanning(y.shape[0]))) ** 2
    f = np.fft.rfftfreq(y.shape[0], 1 / SR)
    assert (f * Y).sum() / Y.sum() < 1000                        # pink air under two lop~ 300: a low room


@pytest.mark.parametrize("mains", [50.0, 60.0])
def test_room_tone_hum_at_the_mains(mains):
    """mains_hum: lines at the supply; hum lives where mains do, none in a cave."""
    y = mono(render_sfx("room_tone", SR, kind="bureau", mains=mains, dur=4.0))
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    m = (f > 30) & (f < 4.5 * mains)
    peak = f[m][np.argmax(Y[m])]
    assert abs(peak / mains - round(peak / mains)) < 0.02
    near = (f > 1.9 * mains) & (f < 2.1 * mains)
    off = (f > 1.2 * mains) & (f < 1.7 * mains)
    assert 10 * np.log10(Y[near].max() / np.median(Y[off])) > 12         # a line standing out of the air
    assert np.array_equal(render_sfx("room_tone", SR, kind="cave", mains=mains, dur=1.0), render_sfx("room_tone", SR, kind="cave", mains=0.0, dur=1.0))    # no mains in a cave


def test_warfare_gunfire_keeps_a_tail():
    """firearm now comes dry and at its relative level: the scene asks for echo, and the street gives it a tail; its level is set by the scene."""
    assert all(l["params"].get("echo") == 0.3 for l in Sc.SCENES["warfare"]["layers"] if l["sfx"] == "firearm")
    assert Sc.SCENES["warfare"]["room"] == "street"
    z = render_sfx("firearm", SR, distance=120.0, shots=1, echo=0.3, render="pressure", seed=0)
    dry = np.zeros((2 * SR, 2))
    Sc._add(dry, Sc._stereo(Sc._shape_event(z, SR)), SR // 4, False)
    wet = Sc._in_room(dry, SR, "street", 0.6, False)
    def tail(y):
        i = int(np.abs(mono(y)).argmax())
        return 20 * np.log10(rms(mono(y)[i + int(0.1 * SR):i + int(0.6 * SR)]) / np.abs(mono(y)).max())
    assert tail(wet) > tail(dry) + 3
