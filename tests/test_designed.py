"""Designed sounds, signals, toys: every variant renders; sourced numbers are measured on the rendered signal."""
import numpy as np
import pytest

from genny import designed as D
from genny.analysis import detect_clicks
from genny.matter import ice_crack
from genny.sfx import REGISTRY

SR = 22050
HI = 48000


def spec(y, sr):
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    return np.fft.rfftfreq(len(y), 1 / sr), Y


def top(y, sr, lo=20.0, hi=None):
    f, Y = spec(y, sr)
    m = (f >= lo) & (f <= (hi or sr / 2))
    return f[m][np.argmax(Y[m])]


def share(y, sr, lo, hi):
    f, Y = spec(y, sr)
    return Y[(f >= lo) & (f < hi)].sum() / Y.sum()


def centroid(y, sr):
    f, Y = spec(y, sr)
    return (f * Y).sum() / Y.sum()


def rms(y):
    return float(np.sqrt(np.mean(np.square(y))))


def seg(y, sr, a, b):
    return y[int(a * sr):int(b * sr)]


def env(y, sr, ms=2.0):
    w = max(1, int(ms * sr / 1000))
    return np.convolve(np.abs(y), np.ones(w) / w, "same")


def line_db(y, sr, f0, width=0.01):
    f, Y = spec(y, sr)
    near = (f >= f0 * (1 - width)) & (f <= f0 * (1 + width))
    around = (f >= f0 * 0.7) & (f <= f0 * 1.4)
    return 10 * np.log10(Y[near].max() / np.median(Y[around]))


def runs(mask):
    """(start, length) of each run of True."""
    d = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    a, b = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return a, b - a


# ---------------------------------------------------------------------------------- every variant
SMOOTH = [(D.cartoon, dict(kind=k)) for k in ("stretch", "pop", "warble", "zip")]
SMOOTH += [(D.tinnitus, {}), (D.party_horn, dict(kind="trompette")), (D.mic_noise, dict(kind="souffle")), (D.stinger_sfx, dict(kind="bass_dive"))]
SMOOTH += [(D.magic_kind, dict(kind=k)) for k in ("ange", "mal", "sort", "poof", "shimmer")]
SMOOTH += [(D.radio, dict(kind=k)) for k in D.RADIOS]
SMOOTH += [(D.creature, dict(kind=k)) for k in ("elementaire", "fantome", "insectoid")]
SMOOTH += [(D.modem, dict(kind=k)) for k in ("poignee de main", "fax")]
SMOOTH += [(D.morse, {}), (D.toy, dict(kind="jouet parlant")), (D.mic_feedback, {})]
SMOOTH += [(D.crt, dict(state=k)) for k in ("souffle 15 kHz", "projecteur")]
SMOOTH += [(D.kazoo, {}), (D.rubber_duck, {}), (D.water_whistle, {}), (D.pendulum_feedback, dict(dur=6.0))]
SMOOTH += [(D.alert_tone, dict(preset=k)) for k in ("information", "message_alert", "finished", "error_code", "three_tone")]
# Edges of their own (detect_clicks is meant for smooth voices): brass and distortion saws, crackle, glottal pulses, relays, grains, the book's audio-rate gates.
EDGY = [(D.cartoon, dict(kind=k)) for k in ("splat", "bonk")]
EDGY += [(D.party_horn, {}), (D.cap_gun, {}), (D.mic_noise, {}), (D.ui_slide, {}), (D.braam, {}), (D.geiger, {}), (D.morse, dict(mode="sounder"))]
EDGY += [(D.turntable, dict(state=k)) for k in D.TT_STATES]
EDGY += [(D.stinger_sfx, dict(kind=k)) for k in ("boom", "danger", "defeat")]
EDGY += [(D.magic_kind, dict(kind="element")), (D.rotary, {}), (D.morph, {}), (D.ice_ring, {}), (D.rod_cage, dict(dur=2.0)), (D.can_roll, {})]
EDGY += [(D.creature, dict(kind=k)) for k in ("dragon", "dino", "zombie", "kraken", "blob")]
EDGY += [(D.toy, dict(kind=k)) for k in ("mecanique", "telecommandee")]
EDGY += [(D.crt, dict(state=k)) for k in ("allumage", "cassette")]
EDGY += [(D.alert_tone, dict(preset=k)) for k in ("happy_blips", "affirmative", "activate", "invaders", "wrong_buzzer")]


def _id(c):
    return f"{c[0].__name__}-{'-'.join(map(str, c[1].values()))}"


@pytest.mark.parametrize("case", SMOOTH, ids=_id)
def test_smooth_variant_renders_without_click(case):
    y = np.asarray(case[0](sr=44100, **case[1]))
    assert np.isfinite(y).all() and np.abs(y).max() > 0.01
    assert len(detect_clicks(y)) == 0


@pytest.mark.parametrize("case", EDGY, ids=_id)
def test_edgy_variant_renders_with_clean_ends(case):
    y = np.asarray(case[0](sr=SR, **case[1]))
    assert np.isfinite(y).all() and np.abs(y).max() > 0.01
    assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3


def test_every_plan_call_is_in_the_catalog_and_kinds_are_checked():
    for name in ("cartoon", "tinnitus", "party_horn", "cap_gun", "mic_noise", "turntable", "stinger_sfx", "magic_kind", "ui_slide", "radio", "creature", "braam",
                 "geiger", "modem", "morse", "toy", "mic_feedback", "rotary", "crt", "morph", "ice_ring", "kazoo", "rubber_duck", "water_whistle",
                 "pendulum_feedback", "rod_cage"):
        assert name in REGISTRY
    for f in (D.cartoon, D.stinger_sfx, D.magic_kind, D.radio, D.creature, D.modem, D.toy, D.party_horn, D.mic_noise):
        with pytest.raises(ValueError):
            f(sr=SR, kind="nope")
    with pytest.raises(ValueError):
        D.morse(sr=SR, text="a~")
    with pytest.raises(ValueError):
        D.morph(sr=SR, a="nope")


# ---------------------------------------------------------------------------------- morse
def test_paris_is_50_units():
    assert sum(u for _, u in D.morse_units("PARIS ")) == 50


def test_morse_timing_on_the_signal():
    wpm = 20.0
    dit = 1.2 / wpm                                            # 60 ms
    y = D.morse(sr=HI, text="SOS", wpm=wpm, freq=700.0)
    assert abs(top(y, HI) - 700.0) < 2.0
    a, ln = runs(env(y, HI) > 0.5 * env(y, HI).max())
    assert len(a) == 9
    assert np.allclose(ln[:3] / HI, dit, atol=0.008) and np.allclose(ln[3:6] / HI, 3 * dit, atol=0.008) and np.allclose(ln[6:] / HI, dit, atol=0.008)
    gaps = (a[1:] - (a[:-1] + ln[:-1])) / HI
    assert np.allclose(gaps[[0, 1, 3, 4, 6, 7]], dit, atol=0.008)   # within a letter
    assert np.allclose(gaps[[2, 5]], 3 * dit, atol=0.008)           # between letters


def test_morse_word_gap_is_7_units_and_length_follows_wpm():
    dit = 1.2 / 30.0
    y = D.morse(sr=HI, text="E E", wpm=30.0)
    a, ln = runs(env(y, HI) > 0.5 * env(y, HI).max())
    assert abs((a[1] - a[0] - ln[0]) / HI - 7 * dit) < 0.008
    assert len(D.morse(sr=HI, text="PARIS PARIS", wpm=12.0)) / HI == pytest.approx(93 * 1.2 / 12.0 + 0.02, abs=0.01)


def test_sounder_clacks_at_the_key():
    y = D.morse(sr=HI, text="E", mode="sounder")
    assert np.argmax(env(y, HI)) / HI < 0.02


# ---------------------------------------------------------------------------------- modem
def test_bell103_lines_on_the_signal():
    y = D.modem(sr=HI, kind="poignee de main")
    assert abs(top(seg(y, HI, 0.1, 0.9), HI) - 2100.0) < 3.0            # ANS [V32]
    assert abs(top(seg(y, HI, 1.05, 1.2), HI) - 2225.0) < 8.0           # answer mark, alone
    data = seg(y, HI, 1.6, 3.9)
    a, b = share(data, HI, 770, 1570), share(data, HI, 1725, 2525)      # Carson: mean +- (100 Hz shift + 300 bit/s)
    assert a > 0.3 and b > 0.3 and a + b > 0.97
    assert line_db(data, HI, 1270.0, 0.03) > 3 and line_db(data, HI, 2225.0, 0.03) > 3


def test_fsk_rate_and_tones():
    bits = [0, 1] * 150
    y = D.fsk(bits, D.BELL103_ORIG)
    assert len(y) == pytest.approx(HI, abs=2)                           # 300 bit/s
    assert line_db(y, HI, 1070.0, 0.05) > 10 and line_db(y, HI, 1270.0, 0.05) > 10
    one = D.fsk([1] * 300, D.BELL103_ORIG)
    assert abs(top(one, HI) - 1270.0) < 2.0


def test_fax_cng_ced_v21():
    y = D.modem(sr=HI, kind="fax")
    assert abs(top(seg(y, HI, 0.05, 0.45), HI) - 1100.0) < 3.0
    assert rms(seg(y, HI, 0.6, 3.4)) < 1e-3                             # 3 s off
    assert abs(top(seg(y, HI, 3.6, 6.4), HI) - 2100.0) < 2.0
    assert share(seg(y, HI, 6.6, 8.0), HI, 1350, 2150) > 0.97       # Carson, V.21 channel 2


# ---------------------------------------------------------------------------------- alarms
def test_timbre_0_is_even_harmonics_11_7_db_down():
    n = HI
    even, odd = D.cosc(1000.0, n, 0.0), D.cosc(1000.0, n, 1.0)
    fe, Ye = spec(even - even.mean(), HI)
    _, Yo = spec(odd, HI)
    assert fe[np.argmax(Ye)] == pytest.approx(2000.0, abs=2)            # an octave up
    assert fe[np.argmax(Yo)] == pytest.approx(1000.0, abs=2)
    assert 10 * np.log10(Yo.max() / Ye.max()) == pytest.approx(11.7, abs=0.2)


def test_affirmative_sounds_an_octave_up():
    y = D.alert_tone(sr=HI, preset="affirmative")                       # 238 ms, 317 then 476 Hz, timbre 0, pulses 59.5 ms
    assert len(y) / HI == pytest.approx(0.238, abs=0.002)
    assert abs(top(seg(y, HI, 0.0, 0.0595), HI) - 634.0) < 40.0
    assert abs(top(seg(y, HI, 0.1785, 0.238), HI) - 952.0) < 40.0


def test_zero_hz_is_silence():
    y = D.alert_segment(380.0, 2, (349, 0, 0, 0), 1)
    e = env(y, HI, 1.0)
    assert rms(y[int(0.0475 * HI * 1.15):int(0.0475 * HI * 3.9)]) < 0.02 * rms(y[:int(0.0475 * HI)])
    assert e[int(0.5 * 0.0475 * HI)] > 0.02 and e[int(4.5 * 0.0475 * HI)] > 0.02   # two blips, at pulses 1 and 5


@pytest.mark.parametrize("preset,pulse_ms", [("happy_blips", 47.5), ("message_alert", 119.0), ("invaders", 28.64)])
def test_pulse_is_duration_over_four_scale(preset, pulse_ms):
    (ms, scale, _, _), = D.ALERTS[preset]
    assert ms / (4 * scale) == pytest.approx(pulse_ms, abs=0.02)


def test_message_alert_descends_three_times():
    y = D.alert_tone(sr=HI, preset="message_alert")
    p = 0.119
    for cycle in range(3):
        got = [top(seg(y, HI, (4 * cycle + i) * p + 0.01, (4 * cycle + i + 1) * p - 0.01), HI, 100, 1500) for i in range(4)]
        assert got == pytest.approx([619, 571, 365, 206], abs=25)


def test_three_tone_723_932_1012():
    y = D.alert_tone(sr=HI, preset="three_tone")
    for i, f in enumerate((723, 932, 1012)):
        assert abs(top(seg(y, HI, i * 0.5 + 0.05, (i + 1) * 0.5 - 0.05), HI) - f) < 3.0


def test_error_code_is_gated_at_207_hz():
    y = D.alert_tone(sr=HI, preset="error_code")
    assert 714.0 / (74 * 4) == pytest.approx(2.41, abs=0.005)
    assert line_db(y, HI, 1000.0 + 207.0, 0.01) > 10 and line_db(y, HI, 1000.0 - 207.0, 0.01) > 10


def test_wrong_buzzer_is_two_messages():
    assert len(D.alert_tone(sr=HI, preset="wrong_buzzer")) == pytest.approx(0.834 * HI, abs=3)


# ---------------------------------------------------------------------------------- phone
def test_line_resonates_at_2k_and_rejects_deep_lows():
    n = HI
    imp = np.zeros(n)
    imp[100] = 1.0
    y = D.phone_line(imp)
    Y = np.abs(np.fft.rfft(y)) ** 2                                     # no window: the impulse sits at the start
    f = np.fft.rfftfreq(len(y), 1 / HI)
    level = lambda a, b: Y[(f >= a) & (f < b)].mean()
    assert level(1950, 2050) > 10 * level(1000, 1300)                   # the 2 kHz line resonance, Q 12
    assert level(350, 450) > 10 * level(40, 70)                         # the 400 Hz band beside it


@pytest.mark.parametrize("digit,count", [("3", 3), ("7", 7), ("0", 10)])
def test_rotary_pulses_at_10_per_second(digit, count):
    y = D.rotary(sr=HI, digits=digit)
    assert D.pulse_count(digit) == count
    wind = 0.05 + 0.05 * count + 0.05
    win = seg(y, HI, wind, wind + 0.1 * count + 0.08)
    e = env(win, HI, 3.0)
    a, _ = runs(e > 0.35 * e.max())
    onsets = a[np.concatenate(([True], np.diff(a) > 0.05 * HI))]        # 40 ms contact, 60 ms open: only the wider gap starts a pulse
    assert len(onsets) == count
    assert np.median(np.diff(onsets)) / HI == pytest.approx(0.1, abs=0.005)
    assert (a[1] - a[0]) / HI == pytest.approx(0.04, abs=0.005)         # make and break of the 40 ms contact


# ---------------------------------------------------------------------------------- can
def test_can_modes_and_turn_period():
    y = D.can_roll(sr=HI, speed=2.0, dur=4.0)
    for f in (359.0, 426.0, 1748.0):
        assert line_db(y, HI, f, 0.02) > 10, f
    a = np.abs(y) - np.mean(np.abs(y))
    c = np.correlate(a[::24], a[::24], "full")[len(a[::24]) - 1:]
    assert np.argmax(c[int(0.3 * 2000):int(0.7 * 2000)]) / 2000 + 0.3 == pytest.approx(0.5, abs=0.01)   # one turn


def test_can_gets_louder_with_speed_as_sqrt():
    a, b = np.abs(D.can_roll(sr=HI, speed=0.25, dur=18.0)).max(), np.abs(D.can_roll(sr=HI, speed=1.0, dur=4.5)).max()
    assert b / a == pytest.approx(2.0, abs=0.1)                         # sqrt law, lot_B03 l.40; the same 4.5 turns: the pattern's wraps coincide at turn 5
    assert np.abs(D.can_roll(sr=HI, speed=8.0, dur=2.0)).max() < 1.2    # never reaches _fin's guard, which would hide the law
    assert D.CAN_LEVELS[:3] == (1.0, 0.513, 0.192) and D.ROLL_PATTERN[3] == (5.6, 0.012)


# ---------------------------------------------------------------------------------- stingers
def test_danger_rises_a_minor_second():
    y = D.stinger_sfx(sr=HI, kind="danger", freq=165.0)
    a, b = top(seg(y, HI, 0.1, 0.4), HI, 100, 400), top(seg(y, HI, 0.7, 1.5), HI, 100, 400)
    assert a == pytest.approx(165.0, abs=4) and b / a == pytest.approx(2 ** (1 / 12), abs=0.015)


def test_defeat_falls_a_minor_third():
    y = D.stinger_sfx(sr=HI, kind="defeat", freq=165.0)
    a, b = top(seg(y, HI, 0.1, 0.4), HI, 100, 400), top(seg(y, HI, 0.7, 1.5), HI, 100, 400)
    assert a / b == pytest.approx(2 ** (3 / 12), abs=0.02)


def test_boom_hits_after_the_rise_and_dive_falls():
    y = D.stinger_sfx(sr=SR, kind="boom")
    assert np.argmax(env(y, SR, 10)) / SR > 0.9
    d = D.stinger_sfx(sr=HI, kind="bass_dive")
    assert top(seg(d, HI, 0.05, 0.4), HI) > 2 * top(seg(d, HI, 1.2, 1.7), HI)


def test_braam_has_the_sub_at_pitch():
    y = D.braam(sr=HI, pitch=55.0)
    assert top(seg(y, HI, 0.5, 3.0), HI) == pytest.approx(55.0, abs=1.0)
    assert share(y, HI, 100, 3000) > 0.05                               # the distorted brass


# ---------------------------------------------------------------------------------- cartoons, design
def test_cartoon_gestures():
    def p(y, a, b):
        return top(seg(y, SR, a, b), SR, 100, 6000)
    s = D.cartoon(sr=SR, kind="stretch")
    assert p(s, 0.5, 0.75) > 2 * p(s, 0.02, 0.2)                        # pitch climbs
    z = D.cartoon(sr=HI, kind="zip")
    assert centroid(z[len(z) // 2:], HI) > 2 * centroid(z[:len(z) // 2], HI)
    assert len(z) / HI < 0.15
    b = D.cartoon(sr=SR, kind="bonk")
    assert np.argmax(env(b, SR)) / SR < 0.05
    assert D.cartoon(sr=SR, kind="pop").size < D.cartoon(sr=SR, kind="bonk").size


def test_warble_wobbles_at_9_hz():
    y = D.cartoon(sr=HI, kind="warble")
    hop = HI // 40
    fr = [top(y[i:i + 2 * hop], HI, 200, 1500) for i in range(0, len(y) - 2 * hop, hop // 2)]
    fr = np.array(fr) - np.mean(fr)
    F = np.abs(np.fft.rfft(fr * np.hanning(len(fr))))
    fm = np.fft.rfftfreq(len(fr), (hop / 2) / HI)
    assert abs(fm[np.argmax(F[2:]) + 2] - 9.0) < 3.0


def test_magic_families_differ_as_described():
    ange, mal = D.magic_kind(sr=SR, kind="ange"), D.magic_kind(sr=SR, kind="mal")
    assert centroid(ange, SR) > 5 * centroid(mal, SR)
    poof = D.magic_kind(sr=SR, kind="poof")
    assert len(poof) / SR < 0.5 and centroid(poof, SR) < 1500
    sort = D.magic_kind(sr=SR, kind="sort")
    assert rms(seg(sort, SR, 0.8, 1.2)) > 0.5 * rms(seg(sort, SR, 0.3, 0.7))
    s = D.magic_kind(sr=SR, kind="shimmer")
    assert len(s) / SR > 1.5


@pytest.mark.parametrize("direction", ["in", "out"])
def test_ui_slide_direction_and_tick(direction):
    y = D.ui_slide(sr=HI, direction=direction, dur=0.18)
    a, b = centroid(seg(y, HI, 0.0, 0.08), HI), centroid(seg(y, HI, 0.09, 0.17), HI)
    assert (b > a) if direction == "in" else (a > b)
    assert np.argmax(env(y, HI)) / HI == pytest.approx(0.18, abs=0.03)


def test_radio_is_in_the_voice_band():
    for k in ("emetteur", "recherche"):
        assert share(D.radio(sr=HI, kind=k), HI, 250, 4000) > 0.8, k
    q = D.radio(sr=HI, kind="squelch")
    assert np.argmax(env(q, HI)) / HI == pytest.approx(0.28, abs=0.02)


def test_morph_goes_from_a_to_b():
    y = D.morph(sr=SR, a="riser", b="wind", dur=2.0)
    assert np.isfinite(y).all() and not np.array_equal(seg(y, SR, 0.0, 0.3), seg(y, SR, 1.7, 2.0))
    assert centroid(seg(D.morph(sr=SR, a="wind", b="tinnitus", dur=2.0), SR, 1.7, 1.95), SR) > 3 * centroid(seg(D.morph(sr=SR, a="wind", b="tinnitus", dur=2.0), SR, 0.05, 0.3), SR)


def test_creature_sizes_and_layers():
    dino, insect, blob = (D.creature(sr=SR, kind=k) for k in ("dino", "insectoid", "blob"))
    assert centroid(insect, SR) > 5 * centroid(dino, SR)
    assert share(blob, SR, 0, 1200) > 0.9
    fantome = D.creature(sr=SR, kind="fantome")
    assert rms(seg(fantome, SR, 0.0, 0.2)) < 0.6 * rms(seg(fantome, SR, 0.9, 1.2))      # creeps in


# ---------------------------------------------------------------------------------- small things
@pytest.mark.parametrize("freq", [4000.0, 6000.0, 8000.0])
def test_tinnitus_band(freq):
    assert top(D.tinnitus(sr=HI, freq=freq), HI) == pytest.approx(freq, abs=2.0)


def test_tinnitus_clamped_to_4_8_k():
    assert top(D.tinnitus(sr=HI, freq=12000.0), HI) == pytest.approx(8000.0, abs=2.0)


def test_cap_gun_5_ms_around_3_khz():
    y = D.cap_gun(sr=HI)
    e = np.cumsum(y * y)
    assert np.searchsorted(e, 0.8 * e[-1]) / HI < 0.012
    assert share(y, HI, 1000, 8000) > 0.7


def test_party_horn_reed_pitch_and_rise():
    t = D.party_horn(sr=HI, kind="trompette", freq=480.0)
    assert top(seg(t, HI, 0.1, 0.7), HI, 200, 2000) == pytest.approx(480.0, abs=6.0)
    y = D.party_horn(sr=HI, freq=480.0)
    assert top(seg(y, HI, 0.5, 0.6), HI, 200, 2000) > top(seg(y, HI, 0.02, 0.1), HI, 200, 2000)


def test_mic_noise_is_low():
    assert centroid(D.mic_noise(sr=SR, kind="bump"), SR) < 600
    assert share(D.mic_noise(sr=SR, kind="souffle"), SR, 0, 1500) > 0.8


def test_mic_feedback_locks_on_freq_and_swells():
    y = D.mic_feedback(sr=HI, freq=2500.0)
    assert top(seg(y, HI, 1.5, 2.4), HI) == pytest.approx(2500.0, abs=10.0)
    assert rms(seg(y, HI, 1.8, 2.3)) > 3 * rms(seg(y, HI, 0.5, 1.0))


def test_turntable_ticks_once_a_turn_at_0_56_hz():
    assert D.TURNTABLE_HZ == pytest.approx(0.5556, abs=0.001)
    y = D.turntable(sr=HI, state="lecture", dur=9.0)
    e = env(y, HI, 3.0)
    a, _ = runs(e > 0.6 * e.max())
    assert len(a) >= 3
    d = np.diff(a[np.concatenate(([True], np.diff(a) > 0.5 * HI))]) / HI
    assert np.median(d) == pytest.approx(1.0 / 0.5556, abs=0.03)


def test_turntable_runs_up_and_down():
    up, down = D.turntable(sr=SR, state="demarrage", dur=2.4), D.turntable(sr=SR, state="arret", dur=2.4)
    assert rms(seg(up, SR, 0.0, 0.25)) < rms(seg(up, SR, 1.4, 2.2))
    assert rms(seg(down, SR, 1.4, 2.2)) < 0.1 * rms(seg(down, SR, 0.0, 0.6))


@pytest.mark.parametrize("system,f", [("pal", 15625.0), ("ntsc", 15734.27)])
def test_crt_line_whine(system, f):
    assert top(D.crt(sr=HI, state="souffle 15 kHz", system=system, dur=1.0), HI) == pytest.approx(f, abs=2.0)
    y = D.crt(sr=HI, state="allumage", system=system)
    assert rms(seg(y, HI, 0.1, 0.5)) > 3 * rms(seg(y, HI, 2.6, 2.95))    # the degauss dies away
    assert line_db(seg(y, HI, 1.7, 2.9), HI, f, 0.002) > 20


def test_crt_mains_follows_system():
    a = D.crt(sr=HI, state="allumage", system="pal")
    assert top(seg(a, HI, 0.1, 1.0), HI, 40, 400) == pytest.approx(100.0, abs=3.0)


def test_projector_runs_at_24_fps():
    y = D.crt(sr=HI, state="projecteur", dur=3.0)
    a = D.crt(sr=HI, state="projecteur", dur=3.0)
    e = np.abs(a) - np.mean(np.abs(a))
    assert top(e, HI, 10, 40) == pytest.approx(24.0, abs=0.5) and np.array_equal(y, a)


def test_cassette_has_hiss_between_clunks():
    y = D.crt(sr=HI, state="cassette", dur=2.0)
    assert np.argmax(env(y, HI)) / HI < 0.1 or np.argmax(env(y, HI)) / HI > 2.0
    assert share(seg(y, HI, 0.5, 1.8), HI, 3000, 24000) > 0.05


def test_ice_ring_is_tonal_next_to_a_dry_crack():
    y = D.ice_ring(sr=HI)
    c = np.asarray(ice_crack(sr=HI, dur=2.0, distance=120.0, cracks=1, sing=0.0), float)
    f, Y = spec(y, HI)
    _, Z = spec(c, HI)
    m = (f > 200) & (f < 5000)
    assert Y[m].max() / np.median(Y[m]) > Z[m].max() / np.median(Z[m])


def test_kazoo_pitch_and_buzz():
    y = D.kazoo(sr=HI, freq=220.0)
    f, Y = spec(seg(y, HI, 0.2, 0.8), HI)
    assert line_db(seg(y, HI, 0.2, 0.8), HI, 220.0, 0.02) > 15
    assert Y[(f > 1500) & (f < 5000)].sum() / Y.sum() > 0.01            # the membrane's buzz


def test_rubber_duck_squeezes_and_glides():
    y = D.rubber_duck(sr=HI, squeezes=3)
    assert len(y) / HI == pytest.approx(3 * 0.34 + 0.1, abs=0.01)
    one = D.rubber_duck(sr=HI)
    assert 900 < top(seg(one, HI, 0.11, 0.17), HI, 500, 3000) < 1250
    assert top(seg(one, HI, 0.11, 0.17), HI, 500, 3000) > top(seg(one, HI, 0.24, 0.27), HI, 500, 3000)


def test_water_whistle_follows_the_immersion_law():
    y = D.water_whistle(sr=HI, length=0.04, warble=4.0, bubble=0.0, dur=1.0)
    law = D.water_whistle_pitch(HI, 0.04, 4.0)
    assert law.min() == pytest.approx(343 / (4 * 0.04 * 0.9), rel=0.01) and law.max() == pytest.approx(343 / (4 * 0.04 * 0.6), rel=0.01)
    hi_i, lo_i = int(np.argmax(law[:HI // 2])), int(np.argmin(law[:HI // 2]))
    w = int(0.015 * HI)
    assert top(y[hi_i - w:hi_i + w], HI, 500, 8000) == pytest.approx(law[hi_i], rel=0.05)
    assert top(y[lo_i - w:lo_i + w], HI, 500, 8000) == pytest.approx(law[lo_i], rel=0.05)


def test_bubbles_join_the_whistle():
    a, b = D.water_whistle(sr=SR, bubble=0.0, dur=1.0), D.water_whistle(sr=SR, bubble=0.8, dur=1.0)
    assert share(b, SR, 200, 1500) > share(a, SR, 200, 1500)


def test_pendulum_gate_on_the_signal():
    tau, T, w = 12.0, 3.0, 0.3
    y = D.pendulum_feedback(sr=SR, tau=tau, period=T, width=w, dur=20.0)
    t = np.arange(len(y)) / SR
    gate = D.pendulum_gate(t, tau, T, w)
    e = env(y, SR, 5.0)
    on = e > 0.05 * e.max()
    assert np.mean(on == gate) > 0.97                                  # the signal is on exactly when the formula says
    a, ln = runs(gate)
    assert ln[1] < ln[-2] and len(a) >= 5                              # whistles lengthen
    assert gate[t > tau * np.log(1.0 / w) + 0.1].all()                 # at last it never stops


def test_room_sets_the_howl():
    assert not np.array_equal(D.pendulum_feedback(sr=SR, dur=2.0, lx=3.0), D.pendulum_feedback(sr=SR, dur=2.0, lx=9.0))
    f = D.room_modes(6.0, 5.0, 3.0)
    assert (f >= 800).all() and (f <= 3000).all() and np.min(np.abs(f - 343 / 2 * 30 / 6.0)) < 0.2       # mode (30, 0, 0)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_pendulums_set_the_density(p):
    hits = D.cage_hits(p, 20.0, np.random.default_rng(0))
    assert len(hits) == pytest.approx(p * D.STRIKES_PER_S * 20.0, rel=0.15)


def test_rod_cage_rendered_density_and_ball():
    def onsets(y):
        e = env(y, SR, 4.0)
        a, _ = runs(e > 0.25 * e.max())
        return len(a[np.concatenate(([True], np.diff(a) > 0.05 * SR))])
    assert onsets(D.rod_cage(sr=SR, pendulums=3, ball="none", dur=6.0, seed=1)) > onsets(D.rod_cage(sr=SR, pendulums=1, ball="none", dur=6.0, seed=1))
    assert not np.array_equal(D.rod_cage(sr=SR, ball="glass", dur=2.0), D.rod_cage(sr=SR, ball="wood", dur=2.0))


def test_geiger_is_poisson_at_the_rate():
    y = D.geiger(sr=HI, rate=40.0, dur=10.0, seed=2)
    e = env(y, HI, 1.0)
    a, _ = runs(e > 0.1 * e.max())
    k = len(a[np.concatenate(([True], np.diff(a) > 0.002 * HI))])
    assert k == pytest.approx(400, abs=3 * 20 + 40)                    # sqrt(400) = 20; a few fall under the threshold
    gaps = np.diff(a) / HI
    assert np.std(gaps) / np.mean(gaps) > 0.6                          # exponential gaps have a CV of 1


def test_toys_contain_their_mechanisms():
    m = D.toy(sr=SR, kind="mecanique")
    assert rms(seg(m, SR, 0.05, 0.9)) > 0.01 and rms(seg(m, SR, 1.1, 2.3)) > 0.003
    r = D.toy(sr=HI, kind="telecommandee")
    assert rms(seg(r, HI, 0.9, 1.8)) > rms(seg(r, HI, 2.15, 2.19))
    t = D.toy(sr=SR, kind="jouet parlant")
    assert share(seg(t, SR, 0.4, 1.5), SR, 400, 5000) > 0.7            # small speaker
