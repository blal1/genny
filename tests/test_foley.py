"""genny.foley: physical checks and a smoke loop over every name it registers.

Run:  uv run --with pytest pytest tests/test_foley.py -q     (or: uv run python tests/test_foley.py)
"""
import math

import numpy as np

from genny import drums as D
from genny import foley as fo
from genny import instruments as I
from genny import sfx as S
from genny.analysis import brightness, loudness, sharpness
from genny.notes import midi_to_freq, note_to_midi
from genny.physical import modal_data as md

SR = 44100


def MINE(reg):
    return [k for k, e in reg.items() if e["fn"].__module__ == "genny.foley"]


def _ok(y, name):
    assert y.ndim == 1 and np.all(np.isfinite(y)), name
    assert 1e-4 < np.max(np.abs(y)) < 1.5, (name, float(np.max(np.abs(y))))
    assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3, (name, "does not start/end at zero")
    assert abs(np.mean(y)) < 5e-3, (name, "DC", float(np.mean(y)))


def _peak_hz(y, sr, lo, hi):
    """Frequency of the strongest spectral peak in [lo, hi] (Hann, zero-padded, parabolic)."""
    n = 1 << int(np.ceil(np.log2(len(y) * 8)))
    X = np.abs(np.fft.rfft(y * np.hanning(len(y)), n))
    a, b = int(lo * n / sr), int(hi * n / sr)
    k = a + int(np.argmax(X[a:b]))
    k = k + 0.5 * (X[k - 1] - X[k + 1]) / (X[k - 1] - 2 * X[k] + X[k + 1] + 1e-20)
    return k * sr / n


def _onsets(env, thr, refractory):
    out, last = [], -10 ** 9
    for i in np.nonzero((env[1:] >= thr) & (env[:-1] < thr))[0]:
        if i - last > refractory:
            out.append(i)
            last = i
    return np.array(out)


def test_smoke_every_name_two_rates():
    assert len(MINE(S.REGISTRY)) >= 60 and len(MINE(D.REGISTRY)) == 10 and len(MINE(I.REGISTRY)) == 7
    for sr in (22050, 48000):
        for name in MINE(S.REGISTRY):
            _ok(S.render_sfx(name, sr), name)
        for name in MINE(D.REGISTRY):
            for vel in (1.0, 0.2):
                _ok(D.render_drum(name, sr, vel), name)
    for name in ("gunshot", "shake", "chain", "keyboard_typing", "robot_babble", "creak"):     # deterministic
        assert np.array_equal(S.render_sfx(name, SR, seed=3), S.render_sfx(name, SR, seed=3)), name
        assert not np.array_equal(S.render_sfx(name, SR, seed=3), S.render_sfx(name, SR, seed=4)), name


def test_variants_render():
    for name, key, values in (("hit", "object", ["auto"] + list(fo.OBJECTS)), ("gunshot", "kind", list(fo.GUNS)), ("creak", "material", list(fo.CREAKS)),
                              ("creak", "force", ["ramp", "pull", "sway", "constant"]), ("cloth", "kind", ["rustle", "tear", "flap"]),
                              ("paper", "kind", ["rustle", "tear", "page"]), ("switch", "kind", ["toggle", "rocker", "push", "relay", "slide", "rotary"]),
                              ("clock", "size", list(fo.CLOCKS)), ("siren_wail", "kind", ["police", "ambulance", "air_raid"]),
                              ("sword", "blade", list(fo.BLADES)), ("stab", "target", ["flesh", "armour", "wood"]), ("slash", "target", ["flesh", "armour", "wood"]),
                              ("knock", "material", ["wood", "hollow", "metal", "glass"]), ("clink", "object", ["bottle", "vase", "coin", "bell"]),
                              ("thud", "body", ["box", "crate", "sack"]), ("shepard", "direction", ["up", "down"])):
        for v in values:
            _ok(S.render_sfx(name, SR, **{key: v}), (name, v))
    for kind in ("dial", "busy", "ringback"):
        for region in ("us", "uk", "eu"):
            _ok(S.render_sfx("phone_tone", SR, kind=kind, region=region), (kind, region))
    _ok(S.render_sfx("hit", SR, material="glass", object="auto", contact_ms=0.1, size=2.0, position=0.7), "hit auto")
    _ok(S.render_sfx("clock", SR, chime=3, dur=1.0), "clock chime")
    _ok(S.render_sfx("gunshot", SR, kind="smg", shots=5), "smg burst")
    for t in fo.BELLS:
        _ok(I.render_note("church_bell", 220.0, 0.3, SR, 0.8, type=t), t)
    for name in ("glass_harmonica", "bowed_bowl"):
        _ok(I.render_note(name, 440.0, 0.3, SR, 0.8, mode="strike"), name)


def test_hit_legacy_params_unchanged():
    """`hit` with only its v0.22 params is the v0.22 sound (same modal body, contact and re-contacts)."""
    from genny.physics import ModalBody
    for tone, crunch in ((200, 0.5), (320, 0.9)):
        ref = ModalBody("sy_stick", frequency_scale=tone / 200.0, decay_scale=1.35 - 0.65 * crunch).strike(
            duration=0.25, contact_ms=1.8 - 1.5 * crunch, velocity=0.55 + 0.75 * crunch, sr=48000,
            micro_collisions=int(round(1 + 5 * crunch)), seed=23)
        ref = np.tanh(ref * (1.2 + 1.6 * crunch)) * 0.78
        from scipy.signal import lfilter                          # the new version only adds the 12 Hz DC blocker and edge fades
        ref = lfilter([1.0, -1.0], [1.0, -math.exp(-2 * math.pi * 12.0 / 48000)], ref)
        new = S.render_sfx("hit", 48000, tone=tone, crunch=crunch)
        k = 480                                                    # skip the 3 ms edge fades
        c = np.corrcoef(ref[k:-k], new[k:-k])[0, 1]
        assert c > 0.999, (tone, crunch, c)


def test_dtmf_pairs_exact():
    keys = "123A456B789C*0#D"
    rows, cols = (697, 770, 852, 941), (1209, 1336, 1477, 1633)
    worst = 0.0
    for sr in (22050, 44100, 48000):
        y = S.render_sfx("dtmf", sr, digits=keys, tone_ms=400, gap_ms=100)
        for i, k in enumerate(keys):
            seg = y[int(i * 0.5 * sr) + int(0.02 * sr):int(i * 0.5 * sr) + int(0.38 * sr)]
            lo, hi = _peak_hz(seg, sr, 600, 1050), _peak_hz(seg, sr, 1100, 1800)
            worst = max(worst, abs(lo - rows[i // 4]), abs(hi - cols[i % 4]))
            assert abs(lo - rows[i // 4]) < 1.0 and abs(hi - cols[i % 4]) < 1.0, (k, lo, hi)
            assert fo.dtmf_pair(k) == (rows[i // 4], cols[i % 4])
    print("dtmf worst error Hz", round(worst, 3))


def test_phone_cadences():
    want = {("us", "busy"): (0.5, 0.5), ("us", "ringback"): (2.0, 4.0), ("uk", "busy"): (0.375, 0.375), ("uk", "ringback"): (0.4, 0.2, 0.4, 2.0),
            ("eu", "busy"): (0.5, 0.5), ("eu", "ringback"): (1.0, 4.0)}
    tones = {("us", "dial"): (350, 440), ("us", "busy"): (480, 620), ("us", "ringback"): (440, 480), ("uk", "dial"): (350, 450),
             ("uk", "busy"): (400,), ("uk", "ringback"): (400, 450), ("eu", "dial"): (425,), ("eu", "busy"): (425,), ("eu", "ringback"): (425,)}
    worst = 0.0
    for (region, kind), fr in tones.items():
        cad = want.get((region, kind))
        dur = 2.0 if cad is None else 2 * sum(cad)
        y = S.render_sfx("phone_tone", SR, kind=kind, region=region, dur=dur)
        on = np.convolve(np.abs(y), np.ones(220) / 220, "same") > 0.05 * np.max(np.abs(y))
        if cad is None:
            assert on[int(0.05 * SR):int(-0.05 * SR)].all(), (region, kind, "dial tone must be continuous")
            seg = y[int(0.2 * SR):int(1.8 * SR)]
        else:
            edges = np.nonzero(np.diff(on.astype(int)))[0] / SR
            runs = np.diff(np.concatenate([[0.0], edges]))[:len(cad)]
            worst = max(worst, float(np.max(np.abs(runs - cad))))
            assert np.allclose(runs, cad, atol=0.012), (region, kind, runs)
            seg = y[int(0.02 * SR):int((cad[0] - 0.02) * SR)]
        for f in fr:
            got = _peak_hz(seg, SR, f - 12, f + 12)
            assert abs(got - f) < 1.0, (region, kind, f, got)
    print("phone cadence worst on/off error s", round(worst, 4))


def _tick_rate(y, sr):
    """Ticks per second from the 0.7-3 kHz envelope (the low bands of the tick/tock clusters at every size; the
    escapement pings, which fire twice per tick, sit above 5 kHz)."""
    from scipy.signal import butter, sosfiltfilt
    b = sosfiltfilt(butter(4, [700, 3000], "band", fs=sr, output="sos"), y)
    w = int(0.004 * sr)
    env = np.convolve(np.abs(b), np.ones(w) / w, "same")
    env = env[::20] - env[::20].mean()                           # the tick period = first strong peak of the envelope's autocorrelation
    ac = np.correlate(env, env, "full")[len(env) - 1:]
    fs = sr / 20
    a = int(0.1 * fs)
    big = np.nonzero(ac[a:int(3.0 * fs)] > 0.5 * ac[a:int(3.0 * fs)].max())[0][0] + a
    k = big + int(np.argmax(ac[big:big + int(0.05 * fs)]))
    return fs / k, k


def test_clock_tick_rate():
    for kw, want in ((dict(size="mantle"), 4.0), (dict(size="watch"), 5.0), (dict(size="grandfather"), 1.0), (dict(rate=2.5), 2.5),
                     (dict(size="grandfather", pendulum=0.994), 1.0), (dict(pendulum=0.25), 1.0 / (math.pi * math.sqrt(0.25 / 9.81)))):
        y = S.render_sfx("clock", SR, dur=8.0, **kw)
        rate, n = _tick_rate(y, SR)
        print("clock", kw, "ticks/s", round(rate, 3), "want", round(want, 3))
        assert abs(rate - want) / want < 0.02, (kw, rate, want)
    assert abs(fo.pendulum_ticks(0.994) - 1.0) < 2e-3            # the seconds pendulum


def test_gunshot_nwave_and_distance():
    # the N-wave alone: +1 to -1 over the transition (2 ms), 4 ms in all
    w = fo.n_wave(2.0, 9.133, reflect=0.0)
    i_hi, i_lo = int(np.argmax(w)), int(np.argmin(w))
    nz = np.nonzero(np.abs(w) > 1e-6)[0]
    dur_ms = (nz[-1] - nz[0]) / fo.P * 1e3
    print("n-wave peak-to-trough ms", (i_lo - i_hi) / fo.P * 1e3, "length ms", round(dur_ms, 2))
    assert abs((i_lo - i_hi) / fo.P * 1e3 - 2.0) < 0.05 and abs(dur_ms - 4.0) < 0.1
    r = fo.n_wave(2.0, 9.133) - w                                 # the ground reflection: 9.133 ms later, low-passed
    assert 9.0 < np.nonzero(np.abs(r) > 1e-6)[0][0] / fo.P * 1e3 < 9.3
    # through the catalog: the crack leads; the muzzle blast lags by d (1/c - 1/v) and is duller further away
    sr, v = 48000, fo.GUNS["rifle"][3]
    cen, lag = [], []
    for d in (20.0, 150.0, 400.0):
        y = S.render_sfx("gunshot", sr, kind="rifle", distance=d, echo=0.0, action=0.0)
        crack = y[:int(0.008 * sr)]
        hi, lo = int(np.argmax(crack)), int(np.argmin(crack))
        if d >= 150:                                               # the crack stands clear of the blast
            assert abs((lo - hi) / sr * 1e3 - 2.0) < 0.3, (d, (lo - hi) / sr * 1e3)
        env = np.convolve(np.abs(y), np.ones(96) / 96, "same")
        want = d * (1 / fo.C_AIR - 1 / v)
        a = int((want + 0.004) * sr)                               # the blast window: after its predicted arrival
        b0 = a + int(np.argmax(env[a:a + int(0.06 * sr)]))
        blast = y[a:a + int(0.25 * sr)]
        X = np.abs(np.fft.rfft(blast * np.hanning(len(blast)))) ** 2
        f = np.fft.rfftfreq(len(blast), 1 / sr)
        cen.append(float(X[f > 2000].sum() / X.sum()))
        quiet = env[int(0.06 * sr):int((want - 0.004) * sr)] if want > 0.1 else np.zeros(1)
        lag.append((want, b0 / sr, float(quiet.max() / env[a:].max())))
    print("gunshot blast energy share above 2 kHz at 20/150/400 m", [round(c, 5) for c in cen], "(predicted lag s, blast peak s, level in between)", [tuple(round(x, 3) for x in g) for g in lag])
    assert cen[0] > cen[1] > cen[2] and cen[2] < 0.5 * cen[0]
    for want, got, quiet in lag[1:]:
        assert want < got < want + 0.06 and quiet < 0.2
    # a subsonic round has no crack, so nothing is delayed: the file is shorter by the rifle's lag
    y = S.render_sfx("gunshot", sr, kind="silenced", distance=300.0, echo=0.0)
    assert len(y) < len(S.render_sfx("gunshot", sr, kind="rifle", distance=300.0, echo=0.0)) - 0.4 * sr


def test_hammering_pitch_rises():
    strikes, pace, start, sink = 12, 2.0, 200.0, 0.6
    y = S.render_sfx("hammering", SR, strikes=strikes, pace=pace, start_freq=start, sink=sink, seed=1)
    env = np.convolve(np.abs(y), np.ones(200) / 200, "same")
    on = _onsets(env, 0.2 * env.max(), int(0.25 * SR))
    assert len(on) == strikes, len(on)
    got = []
    for k, i in enumerate(on):
        want = fo.nail_hz(start, sink, k, strikes)
        seg = y[i + int(0.004 * SR):i + int(0.1 * SR)]
        got.append(_peak_hz(seg, SR, want * 0.85, want * 1.15))
        assert abs(got[-1] - want) / want < 0.03, (k, got[-1], want)
    print("nail Hz per blow", [round(g) for g in got])
    assert np.all(np.diff(got) > 0) and got[-1] > 4 * got[0]


def _kw_bands(y, sr):
    """K-weighted energy in octave bands 30 Hz - 7.7 kHz (dB): each band always holds exactly one Shepard voice."""
    from genny.analysis import _k_weight
    X = np.abs(np.fft.rfft(_k_weight(y, sr) * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / sr)
    edges = 30.0 * 2.0 ** np.arange(0, 9)
    return 10 * np.log10(np.array([X[(f >= a) & (f < b)].sum() for a, b in zip(edges[:-1], edges[1:])]) + 1e-12)


def test_shepard_stationary_across_the_loop_point():
    for sr in (44100, 22050):
        rate = 0.5                                                # one cycle = 2 s
        y = S.render_sfx("shepard", sr, rate=rate, dur=6.0)
        T = int(round(sr / rate))
        a, b = y[T // 2:T // 2 + T], y[T // 2 + T:T // 2 + 2 * T]
        err = float(np.max(np.abs(a - b)) / np.max(np.abs(a)))    # sample-exact periodicity
        w = int(0.1 * sr)
        before, after = _kw_bands(y[2 * T - w:2 * T], sr), _kw_bands(y[2 * T:2 * T + w], sr)
        cycle = _kw_bands(y[T:T + w], sr)
        loud = before > before.max() - 30
        d1, d2 = float(np.max(np.abs(before - after)[loud])), float(np.max(np.abs(after - cycle)[loud]))
        l0, l1 = loudness(y[T - w:T], sr), loudness(y[T:T + w], sr)
        print("shepard", sr, "periodicity error", round(err, 5), "| K-weighted octave-band change across the loop point dB", round(d1, 2),
              "| same phase one cycle apart dB", round(d2, 3), "| loudness before/after", round(l0, 2), round(l1, 2))
        assert err < 1e-3 and d2 < 0.1 and abs(l0 - l1) < 0.3
        assert d1 < 2.0                                           # adjacent 0.1 s windows differ only by 0.05 octave of glide
    up = S.render_sfx("shepard", SR, rate=0.25, dur=2.0, voices=2, dropoff=1.0)
    down = S.render_sfx("shepard", SR, rate=0.25, dur=2.0, voices=2, dropoff=1.0, direction="down")
    assert _peak_hz(up[-12000:-2000], SR, 100, 400) > _peak_hz(up[2000:12000], SR, 100, 400)
    assert _peak_hz(down[2000:12000], SR, 100, 400) > _peak_hz(down[-12000:-2000], SR, 100, 400)


def test_bell_partials_match_the_table():
    """Every FEM bell: the rendered partials are the table's, scaled so the hum is the written note."""
    f_note = 220.0
    for typ, table in fo.BELLS.items():
        t = md.TABLES[table]
        y = I.render_note("church_bell", f_note, 2.0, SR, 0.9, type=typ, bright=2.5)
        k = f_note / fo.bell_partial(table, "hum")
        g = np.array(t["gains"][0])
        # every strong spectral peak of the note is a table mode (or sits inside an unresolved doublet of the table)
        seg = y[int(0.02 * SR):int(1.5 * SR)]
        nfft = 1 << 20
        X = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), nfft))
        tab = np.array(t["freqs"]) * k
        worst, n = 0.0, 0
        for i in np.nonzero((X[1:-1] > X[:-2]) & (X[1:-1] >= X[2:]) & (X[1:-1] > 0.03 * X.max()))[0] + 1:
            f = (i + 0.5 * (X[i - 1] - X[i + 1]) / (X[i - 1] - 2 * X[i] + X[i + 1])) * SR / nfft
            j = int(np.argmin(np.abs(tab - f)))
            nb = tab[max(j - 1, 0):j + 2]
            inside = any(a <= f <= b and b / a < 1.012 for a, b in zip(nb[:-1], nb[1:]))
            if not inside:
                worst = max(worst, abs(1200 * math.log2(f / tab[j])))
            n += 1
        hum = _peak_hz(y[int(0.05 * SR):], SR, f_note * 0.97, f_note * 1.03)
        lo, hi = t["freqs"][0] * k, t["freqs"][1] * k
        print(f"{typ:9s} hum doublet {lo:.2f}/{hi:.2f} Hz for a written {f_note} Hz, measured hum peak {hum:.2f}; {n} spectral peaks, all within {worst:.2f} cents"
              f" of a table mode; next partials / hum {[round(x / t['freqs'][0], 3) for x in t['freqs'][2:9:2]]}")
        assert n >= 6 and worst < 3.0, (typ, n, worst)
        assert lo - 0.5 < hum < hi + 0.5 and abs(1200 * math.log2(hum / f_note)) < 10, (typ, hum)
        assert abs(1200 * math.log2((lo * g[0] + hi * g[1]) / (g[0] + g[1]) / f_note)) < 0.01
    # prime tuning moves the same table so that the second doublet is the written note
    y = I.render_note("church_bell", 440.0, 1.5, SR, 0.9, type="english", tune="prime")
    assert abs(1200 * math.log2(_peak_hz(y[2000:], SR, 430, 450) / 440.0)) < 8


def test_other_table_ratios():
    y = I.render_note("handbell", 500.0, 1.0, SR, 0.9, detune=0.0, bright=3.0)
    for r in (1.0, 2.002, 3.0, 2.49, 6.242):                       # Farnell ratios.pd
        got = _peak_hz(y[:int(0.4 * SR)], SR, 500 * r * 0.985, 500 * r * 1.015)
        assert abs(got / 500.0 - r) < 0.004 * r, (r, got / 500.0)
    y = I.render_note("china_bell", 200.0, 2.0, SR, 0.9, bright=2.5)
    for r in md.TABLES["chinabell"]["ratios"]:
        got = _peak_hz(y[:int(1.5 * SR)], SR, 200 * r * 0.98, 200 * r * 1.02)
        assert abs(got / 200.0 - r) < 0.003 * r, (r, got / 200.0)
    y = I.render_note("wind_chime", 600.0, 1.0, SR, 0.9, bright=2.5)
    for r in fo.FREE_BAR[:4]:
        assert abs(_peak_hz(y[:int(0.5 * SR)], SR, 600 * r * 0.98, 600 * r * 1.02) / 600.0 - r) < 0.003 * r
    y = S.render_sfx("boing", SR, freq=300, bend=20.0, bright=0.3)  # settles on freq after the pitch drop
    assert abs(_peak_hz(y[int(0.5 * SR):int(1.2 * SR)], SR, 280, 330) - 300.0) < 1.5
    y = S.render_sfx("phone_bell", SR, bursts=1, bright=0.3)        # bells at 650 and 653 Hz
    assert 648 < _peak_hz(y, SR, 600, 700) < 655


def test_phisem_presets_bounded_and_energy_raises_the_event_rate():
    assert len(fo.SHAKE) == 25 and len(set(fo.SHAKE.values())) == 25
    for name in fo.SHAKE:
        counts, rms = [], []
        for e in (0.05, 0.3, 1.0):
            y = S.render_sfx("shake", 48000, preset=name, energy=e, rate=3.0, dur=1.2, seed=5)
            assert np.all(np.isfinite(y)) and 1e-5 < np.max(np.abs(y)) < 1.5, (name, e, float(np.max(np.abs(y))))
            env = np.convolve(np.abs(y), np.ones(96) / 96, "same")    # audible collisions: time the 2 ms envelope is above -50 dBFS
            counts.append(float(np.mean(env > 0.003)))
            rms.append(float(np.sqrt(np.mean(y ** 2))))
        print(f"{name:14s} share of time with audible collisions at energy .05/.3/1: {[round(c, 3) for c in counts]}  rms {[round(r, 4) for r in rms]}")
        assert rms[0] < rms[1] < rms[2], (name, rms)
        if name != "water_drops":       # STK's water engine retunes its three drop voices; at full energy they merge into fewer, louder drops
            assert counts[0] < counts[1] < counts[2], (name, counts)


def test_cork_pop_is_the_helmholtz_resonance():
    for vol, d, L in ((750.0, 18.5, 80.0), (330.0, 18.5, 60.0), (1500.0, 20.0, 90.0)):
        want = fo.helmholtz_hz(vol, d, L)
        got = _peak_hz(S.render_sfx("cork_pop", SR, volume_ml=vol, neck_mm=d, neck_len_mm=L), SR, 40, 400)
        print("cork_pop", vol, "ml: Helmholtz", round(want, 1), "Hz, measured", round(got, 1))
        assert abs(got - want) / want < 0.03


def test_creak_slips_faster_with_force_and_train_rhythm():
    from genny.physical import ambience as amb
    def slips(level):                                              # the stick-slip source behind `creak`, 4 s of constant force
        return len(amb.stickslip_pulses(np.full(4 * 48000, level), np.random.default_rng(2), return_times=True)[1]) / 4.0
    def centroid(level):
        y = S.render_sfx("creak", 48000, material="wood", force="constant", level=level, dur=2.0, seed=2)
        X = np.abs(np.fft.rfft(y)) ** 2
        return float(np.sum(np.fft.rfftfreq(len(y), 1 / 48000) * X) / np.sum(X))
    lo, hi = slips(0.5), slips(0.95)
    print("creak slips/s at force 0.5 and 0.95:", lo, hi, "(Farnell: one slip every (1-f)60 + 3 + U(0,6(1-f)) ms -> 29 and 162 /s);"
          " spectral centroid Hz", round(centroid(0.5)), round(centroid(0.95)))
    assert abs(lo - 1000 / 34.5) < 2 and abs(hi - 1000 / 6.15) < 8 and centroid(0.95) > centroid(0.5)
    v, L = 72.0 / 3.6, 20.0                                        # train: the pattern repeats every rail_len / v
    y = S.render_sfx("train", 22050, speed=72.0, rail_len=L, dur=8.0)
    env = np.convolve(np.abs(y), np.ones(220) / 220, "same")[::10]
    env = env - env.mean()
    ac = np.correlate(env, env, "full")[len(env) - 1:]
    a = int(0.6 * 2205)
    lag = (np.argmax(ac[a:int(1.5 * 2205)]) + a) / 2205.0
    print("train joint period s", round(lag, 3), "want", L / v)
    assert abs(lag - L / v) < 0.02


# ---------------------------------------------------------------- instruments (mirrors tests/test_instruments.py)
INHARMONIC = {"church_bell", "china_bell"}     # tuned on a stated partial; an autocorrelation pitch is not defined for them


def test_instrument_shapes_and_levels():
    for name in MINE(I.REGISTRY):
        lo, hi = (note_to_midi(n) for n in I.REGISTRY[name]["range"].split("-"))
        for m in (lo - 5, lo, (lo + hi) // 2, hi, hi + 7):
            for dur, vel, sr in ((0.03, 1.0, 44100), (0.1234, 0.3, 44100), (0.3333, 0.7, 22050), (1.0, 1.0, 48000)):
                y = I.render_note(name, midi_to_freq(m), dur, sr, vel)
                assert y.ndim == 1 and np.all(np.isfinite(y)), (name, m, dur)
                assert 1e-4 < np.max(np.abs(y)) < 1.5, (name, m, dur, float(np.max(np.abs(y))))
                assert abs(y[-1]) < 1e-3 and abs(y[0]) < 1e-3, (name, "ends on a step")
        levels = [loudness(I.render_note(name, midi_to_freq(m), 0.5, SR, 0.9), SR, window=0.15)
                  for m in (lo + (hi - lo) // 4, (lo + hi) // 2, hi - (hi - lo) // 4)]
        print(f"{name:16s} level {[round(v, 1) for v in levels]} dB")
        assert abs(sum(levels) / 3 + 12.0) < 1.0, (name, levels)


def test_instrument_tuning():
    for name in MINE(I.REGISTRY):
        lo, hi = (note_to_midi(n) for n in I.REGISTRY[name]["range"].split("-"))
        out = []
        for m in (lo, (lo + hi) // 2, hi):
            f = midi_to_freq(m)
            for dur in (0.03, 0.6, 1.0):
                y = I.render_note(name, f, dur, SR, 0.9)
                if name in INHARMONIC:      # the stated partial (hum / ratio 1) is the strongest peak next to the written note
                    c = 1200 * math.log2(_peak_hz(y[int(0.05 * SR):int(0.05 * SR) + 32768], SR, f * 0.96, f * 1.04) / f)
                elif dur < 0.6 and name == "hurdy_gurdy":
                    continue                # a wheel-bowed string has no settled pitch 30 ms in
                else:
                    c = fo.ac_cents(np.pad(y, (0, 20000)), f, SR)
                out.append(round(float(c), 1))
                assert abs(c) < 10, (name, m, dur, c)
        print(f"{name:16s} cents (lo/mid/hi x dur .03/.6/1) {out}")


def test_instrument_not_piercing():
    for name in MINE(I.REGISTRY):
        lo, hi = (note_to_midi(n) for n in I.REGISTRY[name]["range"].split("-"))
        for m in sorted({min((lo + hi) // 2, 96), min(hi, 96)}):
            f = midi_to_freq(m)
            y = I.render_note(name, f, 0.5, SR, 0.9)
            pure = sharpness(I.render_note("sine", f, 0.5, SR, 0.9), SR)
            s, b = sharpness(y, SR), brightness(y, SR)
            assert s < max(pure + 1.0, 1.5) and s < 2.4, (name, m, round(s, 2), round(pure, 2))
            assert b["above_5k"] < 0.03, (name, m, round(b["above_5k"], 3))


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
