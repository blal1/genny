import numpy as np

from genny import instruments as I
from genny import tubes as T


def test_organ_wheel_click_is_sixth_harmonic_gone_in_70ms():
    sr, f = 44100, 220.0
    base = I.organ(f, 0.3, sr, key_click=0.0)
    click = I.organ(f, 0.3, sr, key_click=1.0, click="wheel") - base
    m = int(I.ORGAN_CLICK_DECAY * sr)
    assert I.ORGAN_CLICK_DECAY == 0.07
    assert np.max(np.abs(click[m + 1:])) < 1e-9
    seg = click[:m] * np.hanning(m)
    spec = np.abs(np.fft.rfft(seg, 1 << 16))
    peak = np.argmax(spec) * sr / (1 << 16)
    assert abs(peak - 6 * f) < 10.0


def test_double_reed_staple_moves_max_flow_to_a_quarter():
    d = T.DOUBLE_REED["oboe"]
    dp, q = T.reed_characteristic(d["pM"], d["ks"], 0.0, d["Sc"], n=4001)
    assert abs(dp[np.argmax(q)] / d["pM"] - 1.0 / 3.0) < 1e-3
    assert q[-1] == 0.0
    dp, q = T.reed_characteristic(d["pM"], d["ks"], d["Cp"], d["Sc"], n=4001)
    assert 0.2 <= dp[np.argmax(q)] / d["pM"] <= 0.26


def _tip_f0(tip):
    tp = tip[int(0.25 * T.FS):]
    sp = np.abs(np.fft.rfft((tp - tp.mean()) * np.hanning(tp.size), 1 << 18))
    return (np.argmax(sp[50:]) + 50) * T.FS / (1 << 18)


def test_free_reed_section_flat_is_the_clearance_ring():
    flat = T.FREE_REED["er"] / 2
    su = T.free_reed_section(flat, "+-")[0]
    r = T.FREE_REED
    assert abs(su - ((r["Wr"] + r["hmin"]) * r["hmin"] + 2 * r["Lr"] * r["hmin"])) < 2e-9
    assert 5e-6 < T.free_reed_section(flat + 500e-6, "+-")[0] < 8e-6


def test_free_reed_blow_open_plays_above_blow_closed_below():
    y, tip = T.free_reed(dur=0.4, kind="+-", full=True)
    assert np.isfinite(y).all()
    assert 444.0 < _tip_f0(tip) < 470.0
    y, tip = T.free_reed(dur=0.4, kind="-+", full=True)
    assert 420.0 < _tip_f0(tip) < 444.0


def test_pipe_end_correction_matches_levine_schwinger():
    from genny.physical import space as S
    for ka, r_exact, la_exact in ((1.0, 0.6951, 0.5274), (2.5, 0.2332, 0.3636), (3.82, 0.0764, 0.1581)):
        r = S.pipe_reflection(ka)
        assert abs(abs(r) - r_exact) < 0.005
        assert abs(np.angle(-r) + 2 * ka * la_exact) % (2 * np.pi) < 1e-3
    assert abs(np.angle(-S.pipe_reflection(1e-3)) + 2e-3 * 0.6133) < 1e-6


def test_loaded_membrane_groups_nine_modes_on_five_tones():
    from genny.physics import MembraneDrum
    m = MembraneDrum(200.0, harmonic_loading=1.0)._modes(0.5, "tone")
    assert np.allclose(np.asarray(m.freqs) / 200.0, [1, 2, 3, 3, 4, 4, 5, 5, 5])


def test_pea_whistle_trills_at_the_spin_rate_and_dips_in_pitch():
    sr = 44100
    y = I.pea_whistle(2500.0, 1.0, sr, 1.0, spin=25.0)
    assert np.isfinite(y).all() and np.max(np.abs(y)) < 1.0
    seg = y[int(0.1 * sr):int(0.9 * sr)]
    w = int(0.004 * sr)
    e = np.convolve(np.abs(seg), np.ones(w) / w, "same")
    em = np.abs(np.fft.rfft(e - e.mean()))
    fe = np.fft.rfftfreq(e.size, 1 / sr)
    assert abs(fe[np.argmax(em[5:200]) + 5] - 25.0) < 1.5
    assert I.PEA_WHISTLE["drop"] == 0.07


def test_clarinet_bend_follows_almeida_2013_fig3():
    from genny.physical import winds as W
    ref = 337.25
    for force, p, hz in ((0.3, 5.0, 331.3), (1.0, 6.5, 340.5), (2.0, 4.5, 344.2)):
        assert abs(W.clarinet_bend_cents(p, force) - 1200 * np.log2(hz / ref)) < 6.0
    assert W.clarinet_bend_cents(5.0, 0.3) < W.clarinet_bend_cents(4.0, 0.3)

def test_vibrato_end_follows_prame_1994_tail():
    sr, n = 8000, 8000 * 4
    for eb, lo, hi in ((0.0, 0.99, 1.01), (-1.0, 1.07, 1.13)):
        s = np.log2(I._vibrato(440.0, n, sr, rate=6.0, depth=0.3, delay=0.1, ramp=0.2, end_boost=eb) / 440.0)
        p = np.diff(np.where((s[:-1] < 0) & (s[1:] >= 0))[0] / sr)
        assert abs(1 / np.median(p[:10]) - 6.0) < 0.05
        assert lo < p[-6] / p[-1] < hi
    assert I.PRAME_END[:2] == (0.15, 1.7)
def test_vibraphone_and_glass_partials_follow_rossing():
    assert I.VIBES_R == ((4.0, 4.0, 3.8), (9.7, 8.9, 7.6)) and I.VIBES_AM == 0.5
    sr = 44100
    y = I.glass(500.0, 1.0, sr)[:sr // 2]
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size), 1 << 17))
    fr = np.fft.rfftfreq(1 << 17, 1 / sr)
    for r, floor in ((1.89, 0.05), (3.21, 0.01)):                     # the 3.21 partial is rolled off (corner 2.4 kHz) and dies in ring/14
        band = (fr > 500.0 * r - 20) & (fr < 500.0 * r + 20)
        assert spec[band].max() > floor * spec.max()