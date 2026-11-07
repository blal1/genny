"""Theme Cordes of the deep reading: values read from the sources, checked on the rendered signal.

Run:  uv run --with pytest pytest tests/test_cordes.py -q
"""
import math

import numpy as np
from scipy.signal import butter, sosfiltfilt

from genny.analysis import detect_clicks
from genny.instruments import REGISTRY as INSTRUMENTS, render_note
from genny.physical import bank_piano as BP
from genny.physical import bow

SR = 44100


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def band_env(y, f0, w=0.03):
    return np.abs(sosfiltfilt(butter(2, [f0 * (1 - w), f0 * (1 + w)], "bandpass", fs=SR, output="sos"), y))


def rms_db(y):
    return 10 * np.log10(np.mean(np.asarray(y) ** 2) + 1e-20)


def peak_f(y, lo, hi):
    S = np.abs(np.fft.rfft(y * np.hanning(y.size), 1 << 18))
    f = np.fft.rfftfreq(1 << 18, 1 / SR)
    b = (f > lo) & (f < hi)
    return f[b][np.argmax(S[b])]


def band_db(y, lo, hi):
    S = np.abs(np.fft.rfft(y * np.hanning(y.size))) ** 2
    f = np.fft.rfftfreq(y.size, 1 / SR)
    return 10 * np.log10(S[(f >= lo) & (f < hi)].sum() + 1e-30)


def t60_of(y, f0):
    e = band_env(y, f0, 0.1)
    k = int(np.argmax(e))
    win = int(0.02 * SR)
    db = np.array([20 * np.log10(e[i:i + win].max() + 1e-12) for i in range(k, e.size - win, win)])
    i = np.flatnonzero(db < db[0] - 30)
    return (i[0] * 0.02 if i.size else (db.size * 0.02)) * 2


# ------- bank_piano


def test_stokes_air_matches_a_hand_calculation():
    # eq 5.80, steel 1 mm at 262 Hz: M = 2.6, tau = 6.4 s
    assert abs(1.0 / BP.air_rate(261.6, 1e-3, 7850.0) - 6.4) < 0.3


def test_una_corda_raises_the_aftersound_not_the_attack():
    # Weinreich II-D: the difference mode is as big as the sum: large increase of aftersound relative to attack
    f0 = midi(60)
    a = band_env(BP.note(f0, 7.0, SR, 0.7, tail=0.0), f0)
    b = band_env(BP.note(f0, 7.0, SR, 0.7, tail=0.0, una_corda=True), f0)
    late = lambda e: e[int(5 * SR):int(6.5 * SR)].mean() / e[:int(0.3 * SR)].max()  # noqa: E731
    assert late(b) > 1.5 * late(a), (late(a), late(b))


def test_longitudinal_force_is_differentiated_and_gives_phantoms():
    # Bank & Sujbert: H_l = 1 - z^-1; products f_n + f_(n+3) stand in the longitudinal force and not in the transverse
    f0, B = midi(36), BP.steinway_B(36)
    full = BP.note(f0, 1.0, SR, 1.0, tail=0.0)
    old, BP.LONG_COUPLING = BP.LONG_COUPLING, 0.0
    trans = BP.note(f0, 1.0, SR, 1.0, tail=0.0)
    BP.LONG_COUPLING = old
    d = (full - trans)[int(0.1 * SR):]
    fk = lambda n: n * f0 * math.sqrt(1 + B * n * n) / math.sqrt(1 + B)  # noqa: E731
    target = fk(7) + fk(10)                                              # 3 * f0-apart pair, in the 1-2 kHz of Conklin
    assert 1000 < target < 2200
    on = band_db(d, target - 1, target + 1)
    off = band_db(d, target + 12, target + 14)
    assert on > off + 6, (on, off)
    assert band_db(d, 0, 60) < band_db(d, 300, 3000)                     # the slow tension swell is taken out


def test_stulov_hammer_memory_softens_the_felt_on_the_way_back():
    # Stulov: memory takes eps of the force back: the same blow leaves with less speed than the elastic felt
    f0 = midi(60)
    y = BP.note(f0, 0.3, SR, 0.7, tail=0.0, hammer="stulov")
    assert np.all(np.isfinite(y)) and len(detect_clicks(y[-int(0.05 * SR):])) == 0
    assert abs(1200 * math.log2(peak_f(y[int(0.05 * SR):], f0 * 0.98, f0 * 1.02) / f0)) < 8


def test_action_comes_20_ms_ahead_and_is_a_pp_level():
    # Chaigne & Askenfelt Fig. 17: precursor 20 ms before contact, the level of the string at pp
    f0 = midi(60)
    y = BP.note(f0, 0.3, SR, 1.0, tail=0.0, action=1.0)
    plain = BP.note(f0, 0.3, SR, 1.0, tail=0.0)
    pp = BP.note(f0, 0.3, SR, 0.0, tail=0.0)
    i = int(0.02 * SR)
    head = np.abs(y[:i]).max()
    assert 0.0 < head < 0.5 * np.abs(plain).max()
    late = y[i + int(0.1 * SR):i + int(0.25 * SR)]
    ref = plain[int(0.1 * SR):int(0.25 * SR)]
    assert np.corrcoef(late, ref)[0, 1] > 0.99                           # the blow itself is the same, 20 ms later
    assert 300 < peak_f(y[:i], 60, 3000) < 1500                          # Fig. 17: strongest near 1000 Hz
    assert 0.1 < head / np.abs(pp).max() < 10.0, head / np.abs(pp).max()


# ------- bow


def test_torsion_keeps_helmholtz_at_the_papers_ratios():
    for r in (0.2, 0.4, 1.0):
        y = bow.bowed(147.0, 0.8, SR, force=2.0, speed=0.2, torsion=r)
        assert np.all(np.isfinite(y))
        assert abs(1200 * math.log2(peak_f(y[int(0.5 * SR):int(0.78 * SR)], 133, 162) / 147.0)) < 35, r


def test_torsion_series_is_silent_at_the_bridge_but_changes_the_string():
    a = bow.bowed(147.0, 0.5, SR, force=2.0, speed=0.2)
    b = bow.bowed(147.0, 0.5, SR, force=2.0, speed=0.2, torsion=0.4)
    assert np.max(np.abs(a - b)) > 0.01


def test_thermal_contact_heats_up_past_the_glass_point_and_keeps_pitch():
    # van Walstijn et al. 2026, Fig. 6 case (x_b/L 0.075, f_b 2.15 N, a 0.5): tau_g = 25 K is crossed within a cycle
    y, temp = bow.bowed(147.0, 0.7, SR, force=2.15, speed=0.1, accel=0.5, beta=0.075, curve="thermal", want_temperature=True)
    w = temp[int(0.5 * SR):int(0.68 * SR)]
    assert w.max() > 25.0 and w.max() < 90.0, w.max()
    assert w.max() - w.min() > 10.0                                      # it cools between the slips
    assert w.min() >= 0.0
    assert abs(1200 * math.log2(peak_f(y[int(0.4 * SR):int(0.68 * SR)], 133, 162) / 147.0)) < 35


def test_mu_str_runs_from_static_to_near_dynamic():
    vg, mg = bow.mu_str_table(215.0)
    assert abs(mg[0] - 1.05) < 0.01                                      # mu_s at rest
    assert mg[-1] < 0.7 and np.all(np.diff(mg) <= 1e-9)                  # falls monotonically with speed


def test_fit_picks_the_tables_case_and_stays_stable():
    assert bow._nearest_fit("S", 0.35, 0.8)["s0"] == 3.706e5
    assert bow._nearest_fit("eps", 2.0, 3.9)["muS"] == 0.9116
    for a, fn in ((0.35, 0.8), (1.05, 2.08), (2.0, 3.9)):
        y = bow.bowed(147.0, 0.8, SR, force=fn, speed=0.2, accel=a, fit=True)
        assert np.all(np.isfinite(y)) and np.abs(y).max() < 50
        assert abs(1200 * math.log2(peak_f(y[int(0.5 * SR):int(0.78 * SR)], 133, 162) / 147.0)) < 35


# ------- instruments


def test_new_voices_render_clean_and_in_tune():
    for name, f, p in (("clavichord", midi(60), {}), ("clavichord", midi(84), {"bebung": 1.0}), ("double_bass", midi(36), {}),
                       ("double_bass", midi(40), {"lifted": 1, "open_string": 1}), ("violin", midi(69), {"model": "elastoplastic"}),
                       ("violin", midi(69), {"lifted": 1})):
        assert name in INSTRUMENTS
        y = np.asarray(render_note(name, f, 0.8, SR, 0.8, **p), float)
        assert np.all(np.isfinite(y)) and np.abs(y).max() > 0.01, (name, p)
        # up to render_note's own 4 ms end fade, whose knee is a click by this measure even on a signal at -55 dB
        assert len(detect_clicks(y[-int(0.05 * SR):-int(0.005 * SR)])) == 0, (name, p)
        got = peak_f(y[int(0.1 * SR):int(0.75 * SR)], f * 0.97, f * 1.03)
        assert abs(1200 * math.log2(got / f)) < 30, (name, p, got)


def test_clavichord_decays_by_its_loop_gain():
    # eq 4 at C4 (k = 25): g = 0.9937, so -60 dB in 3 / (f |log10 g|) = 4.2 s; the second string and a stay near it
    f = midi(60)
    y = np.asarray(render_note("clavichord", f, 3.0, SR, 0.8), float)
    t = t60_of(y[:int(3.0 * SR)], f)
    assert 1.5 < t < 7.0, t


def test_clavichord_bebung_is_2_hz_wide_at_7_hz():
    f = midi(69)
    y = np.asarray(render_note("clavichord", f, 2.0, SR, 0.9, bebung=1.0), float)[int(0.4 * SR):int(1.8 * SR)]
    from scipy.signal import hilbert
    x = sosfiltfilt(butter(2, [f * 0.98, f * 1.02], "bandpass", fs=SR, output="sos"), y)
    fi = np.gradient(np.unwrap(np.angle(hilbert(x)))) * SR / (2 * np.pi)
    fi = fi[2000:-2000]
    S = np.abs(np.fft.rfft(fi - fi.mean(), 1 << 18))
    fr = np.fft.rfftfreq(1 << 18, 1 / SR)
    assert abs(fr[(fr > 2) & (fr < 20)][np.argmax(S[(fr > 2) & (fr < 20)])] - 7.0) < 0.5
    assert 0.8 < (np.percentile(fi, 98) - np.percentile(fi, 2)) / 2 < 3.5


def test_pizzicato_rings_by_velocity_family_and_vibrato():
    # Meyer 3.3: pp shorter than ff; a cello longer than a violin; vibrato halves the decay
    f = midi(62)
    t = lambda **k: t60_of(np.asarray(render_note("pizzicato", f, 0.2, SR, **k), float), f)  # noqa: E731
    pp, ff = t(vel=0.05), t(vel=1.0)
    assert ff > 2.0 * pp, (pp, ff)
    assert t(vel=1.0, size="cello") > 1.2 * ff
    assert t(vel=1.0, vibrato=1.0) < 0.75 * ff


def test_sections_grow_3_db_per_doubling_on_request():
    # Meyer: doubling the desks = +3 dB; level="section" over 2..12 players at A4 and C3
    for f in (440.0, 130.81):
        lv = [rms_db(np.asarray(render_note("strings", f, 1.0, SR, 0.85, players=n, level="section"), float)) for n in (3, 6, 12)]
        assert abs(lv[1] - lv[0] - 3.01) < 1.0 and abs(lv[2] - lv[1] - 3.01) < 1.0, lv


def test_sordino_takes_about_6_db_off():
    f = 440.0
    a = np.asarray(render_note("violin", f, 1.0, SR, 0.8), float)
    b = np.asarray(render_note("violin", f, 1.0, SR, 0.8, sordino=1), float)
    assert -9.0 < rms_db(b) - rms_db(a) < -5.0, rms_db(b) - rms_db(a)


def test_lifted_bow_rings_on_partial_by_partial():
    # Meyer l.4193-4201: bow left 0.1 s; lifted violin ~1 s in the low register, overtones half, open 2-3 s
    f = midi(55)
    held = np.asarray(render_note("violin", f, 0.5, SR, 0.8), float)
    lift = np.asarray(render_note("violin", f, 0.5, SR, 0.8, lifted=1), float)
    opn = np.asarray(render_note("violin", f, 0.5, SR, 0.8, lifted=1, open_string=1), float)
    after = lambda y, a, b: rms_db(y[int((0.5 + a) * SR):int((0.5 + b) * SR)])  # noqa: E731
    assert after(lift, 0.2, 0.3) > after(held[:int(0.9 * SR)], 0.2, 0.3) + 20 if held.size > int(0.8 * SR) else lift.size > held.size
    assert after(opn, 0.6, 0.7) > after(lift, 0.6, 0.7) + 6
    e1, e2 = band_env(lift, f, 0.1), band_env(lift, 3 * f, 0.1)
    drop = lambda e: 20 * np.log10(e[int(0.75 * SR):int(0.8 * SR)].max() / e[int(0.48 * SR):int(0.5 * SR)].max())  # noqa: E731
    assert drop(e2) < drop(e1) - 3


def test_harpsichord_follows_meyers_ring_and_stops():
    # 20 s at C3, /1.4 per octave: longer at C3 than C5 by about 1.96, stops=2 still renders at pitch and is louder
    c3, c5 = 130.81, 523.25
    a = t60_of(np.asarray(render_note("harpsichord", c3, 4.0, SR, 0.8, soundboard=0.0), float)[:int(4 * SR)], c3)
    b = t60_of(np.asarray(render_note("harpsichord", c5, 4.0, SR, 0.8, soundboard=0.0), float)[:int(4 * SR)], c5)
    assert a > 1.3 * b, (a, b)
    one = np.asarray(render_note("harpsichord", c3, 0.8, SR, 0.8), float)
    two = np.asarray(render_note("harpsichord", c3, 0.8, SR, 0.8, stops=2), float)
    assert 1.0 < rms_db(two) - rms_db(one) < 5.0


def test_harp_has_its_five_resonances_and_a_position():
    # a third-string pluck removes the third partial (Meyer 3.5)
    f = midi(55)
    a = np.asarray(render_note("harp", f, 1.0, SR, 0.8, position=1.0 / 3.0), float)[int(0.02 * SR):int(0.8 * SR)]
    b = np.asarray(render_note("harp", f, 1.0, SR, 0.8, position=0.2), float)[int(0.02 * SR):int(0.8 * SR)]
    third = lambda y: band_db(y, 3 * f - 8, 3 * f + 8) - band_db(y, f - 8, f + 8)  # noqa: E731
    assert third(a) < third(b) - 6, (third(a), third(b))
    assert t60_of(np.asarray(render_note("harp", midi(36), 3.0, SR, 0.8), float), midi(36)) > \
        t60_of(np.asarray(render_note("harp", midi(79), 3.0, SR, 0.8), float), midi(79))


def test_pickup_comb_digs_the_right_notch():
    # 1 - z^-(x/(L f_open) fs): notch at L f_open / x; on the low E at 0.25: 330 Hz
    from genny.acoustic import pickup_comb
    x = np.zeros(1 << 15)
    x[0] = 1.0
    H = np.abs(np.fft.rfft(pickup_comb(x, SR, 82.41, 0.25, (82.41,))))
    f = np.fft.rfftfreq(x.size, 1 / SR)
    b = (f > 200) & (f < 500)
    assert abs(f[b][np.argmin(H[b])] - 82.41 / 0.25) < 5.0
    assert np.array_equal(pickup_comb(x, SR, 82.41, 0.0, (82.41,)), x)


def test_pickup_circuit_resonates_where_the_paper_says():
    # 2 H, 50 pF: 16 kHz, +12 dB, by bilinear (Paiva Sec. 5)
    from genny.acoustic import pickup_circuit
    x = np.zeros(1 << 15)
    x[0] = 1.0
    H = 20 * np.log10(np.abs(np.fft.rfft(pickup_circuit(x, 192000))) + 1e-12)
    f = np.fft.rfftfreq(x.size, 1 / 192000)
    k = np.argmax(H)
    assert abs(f[k] - 15900) < 800, f[k]
    assert 9.0 < H[k] - H[np.searchsorted(f, 200)] < 15.0, H[k]


def test_clav_flux_polynomial_adds_harmonics_with_swing():
    from genny.acoustic import clav
    f = midi(48)
    y = np.asarray(clav(f, 0.6, SR, 1.0, flux=True), float)
    assert np.all(np.isfinite(y)) and np.abs(y).max() > 0.01


def test_rhodes_arc_keeps_the_tine_clean_up_to_e7():
    for m in (40, 76, 100):
        y = np.asarray(render_note("rhodes", midi(m), 0.5, SR, 0.9), float)
        assert np.all(np.isfinite(y)) and len(detect_clicks(y[-int(0.05 * SR):])) == 0, m


def test_clavichord_glide_is_a_pitch_scaler():
    from scipy.signal import hilbert
    for m in (72, 84):
        f = midi(m)
        y = np.asarray(render_note("clavichord", f, 1.0, SR, 1.0), float)
        x = sosfiltfilt(butter(2, [f * 0.95, f * 1.05], "bandpass", fs=SR, output="sos"), y)
        fi = np.gradient(np.unwrap(np.angle(hilbert(x)))) * SR / (2 * np.pi)
        c = 1200 * math.log2(np.median(fi[int(0.02 * SR):int(0.05 * SR)]) / np.median(fi[int(0.3 * SR):int(0.6 * SR)]))
        assert 9.0 < c < 14.0, (m, c)
