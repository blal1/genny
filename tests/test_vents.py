"""Winds and brass: the reed regimes, the lip Q, the reed presets, Ka(dp), Meyer's attacks and playing options.

Run:  uv run --with pytest pytest tests/test_vents.py -q
"""
import math

import numpy as np

from genny import acoustic as A
from genny import filters as F
from genny import instruments as I
from genny import tubes as T
from genny.analysis import detect_clicks
from genny.physical import impedance as Z
from genny.physical import waveguides as wg

SR = 44100


def _bands(y, f0, ks, t0=0.3, n=16384, sr=SR):
    y = y[int(t0 * sr):int(t0 * sr) + n]
    Y = np.abs(np.fft.rfft(y * np.hanning(y.size), 8 * y.size))
    fr = np.fft.rfftfreq(8 * y.size, 1 / sr)
    return [Y[(fr > k * f0 * 0.97) & (fr < k * f0 * 1.03)].max() for k in ks]


def _peak(y, sr=SR, t0=0.3, t1=0.75):
    s = y[int(t0 * sr):int(t1 * sr)]
    S = np.abs(np.fft.rfft(s * np.hanning(s.size), 1 << 16))
    return np.fft.rfftfreq(1 << 16, 1 / sr)[np.argmax(S)]


def _db(a, b):
    return 20 * math.log10((a + 1e-15) / (b + 1e-15))


# ------------------------------------------------------------------------------------------ reed_tube regimes
def test_reed_tube_slow_attack_keeps_the_fundamental():
    """Before the fix, attack 0.2 s: H1 14-22 dB under H3 (98-418 Hz), 69 dB at 600 Hz (the L10 defect)."""
    for f in (98.0, 220.0, 418.0, 600.0):
        for vel in (0.3, 0.9):
            y = I.render_note("reed_tube", f, 1.2, SR, vel, attack=0.2)
            h1, h3 = _bands(y, f, (1, 3), 0.7)
            assert _db(h1, h3) > -6.0, (f, vel, _db(h1, h3))
            assert np.std(y[int(0.7 * SR):int(1.1 * SR)]) > 0.1, (f, vel)


def test_reed_tube_slow_attack_still_rises_slowly():
    y = I.render_note("reed_tube", 220.0, 1.2, SR, 0.9, attack=0.2)
    rms = lambda a, b: np.std(y[int(a * SR):int(b * SR)])  # noqa: E731
    assert rms(0.01, 0.03) < 0.1 * rms(0.5, 0.6)


def test_default_attack_does_not_take_the_new_branch():
    """The default keeps its sound: H3 stays within 6 dB of H1 on the held note, across the span."""
    for f in (147.0, 330.0, 600.0, 880.0):
        y = I.render_note("reed_tube", f, 0.7, SR, 0.9)
        assert not T._on_twelfth(y[int(0.2 * SR):int(0.6 * SR)], f), f


# ------------------------------------------------------------------------------------------ lip Q
def test_lip_q_is_the_pole_and_default_unchanged():
    f = 220.0
    r, _ = wg._brass_constants(f, 3.0)
    assert abs(r - math.exp(-math.pi * f / (3.0 * wg.SR))) < 1e-12
    r0, _ = wg._brass_constants(f)
    Q = -math.pi * f / (wg.SR * math.log(r0))
    assert abs(Q - 5.2) < 0.15, Q                                    # the Q the report measured at 110-440 Hz
    a = I.render_note("trumpet", 233.1, 0.4, wg.SR, 0.8, model="physical")
    b = I.render_note("trumpet", 233.1, 0.4, wg.SR, 0.8, model="physical", lip_q=0.0)
    assert np.array_equal(a, b)


def test_low_lip_q_raises_the_speaking_threshold():
    """Measured, in the direction Newton's broad human lips predict for this model: p 0.3 speaks at Q 5.2, not at Q 3."""
    n = int(1.0 * wg.SR)
    t = np.arange(n) / wg.SR
    env = 0.3 * np.clip(t / 0.03, 0, 1)
    rms = lambda q: np.std(wg.brass(466.0, 0.5, env, 1.0, lip_q=q)[int(0.4 * wg.SR):])  # noqa: E731
    assert rms(None) > 0.02 > 0.005 > rms(3.0)


def test_lip_q_voices_stay_in_tune():
    for Q in (2.0, 3.0):
        for name, f in (("trumpet", 233.1), ("trumpet", 698.5), ("trombone", 98.0), ("tuba", 55.0)):
            for vel in (0.3, 0.9):
                y = I.render_note(name, f, 1.5, wg.SR, vel, model="physical", lip_q=Q)
                s = y[int(0.4 * wg.SR):int(1.3 * wg.SR)]
                assert abs(1200 * math.log2(wg.measure_f0(s, f) / f)) < 3, (Q, name, f, vel)
                assert np.std(s) > 0.1, (Q, name, f, vel)


def test_physical_brass_has_no_click():
    """L21a: 2-3 detections at -103 dB on this model before the knee. The knee stays; detect_clicks now finds none."""
    n = wg.SR
    t = np.arange(n) / wg.SR
    env = 0.5 * np.clip(t / 0.03, 0, 1) * np.clip((1.0 - t) / 0.1, 0, 1)
    for f in (130.8, 164.8, 261.6, 523.3):
        y = wg.brass(f, 0.5, env, 1.0)
        assert len(detect_clicks(y / np.max(np.abs(y)), wg.SR)) == 0, f


# ------------------------------------------------------------------------------------------ clarinet reeds
def test_reed_presets_values_and_squeak():
    assert Z.REEDS["typical"][:2] == (2500.0, 0.2)                   # Guillemain 2005, Sec. III
    assert Z.REEDS["squeak"][:2] == (1850.0, 0.2)                    # Sec. IV: "to facilitate the raising of squeaks"
    fr, q = Z.REEDS["van_walstijn"][:2]
    assert fr == 1900.0 and abs(q * 2 * math.pi * fr - 2900.0) < 1e-6
    for f in (494.0, 523.0):
        assert abs(_peak(Z.clarinet(f, 0.8, 44100.0, reed="squeak", blow=0.2)) / (3 * f) - 1) < 0.02, f
        assert abs(_peak(Z.clarinet(f, 0.8, 44100.0, blow=0.2)) / f - 1) < 0.02, f
    assert abs(_peak(Z.clarinet(330.0, 0.8, 44100.0, reed="van_walstijn")) / 330.0 - 1) < 0.02


def test_ka_of_dp_against_figure_3():
    ka = lambda dp: float(np.interp(dp, Z.KA_DP, Z.KA_FIG))       # noqa: E731
    for dp, fig in ((0.0, 7.27e6), (1550.0, 7.5e6), (3000.0, 10.1e6), (5000.0, 13.2e6), (8000.0, 20.8e6)):
        assert abs(ka(dp) / fig - 1) < 1e-9, dp
    assert abs(ka(2500.0) - 9.3e6) < 1.0 and Z.KA0 == 0.75e7


def test_stiffening_kernel_ramp():
    """1100-4000 Pa over 4.5 s (their Fig. 8 ramp). The level grows. MEASURED pitch rise here
    2.9 cents with Ka(dp), 11 without: the paper reports about 10-20 with it, not reproduced by this bore and flow."""
    sr = 44100.0
    n, L = int(4.5 * sr), 0.4
    fr, q, H, PM = Z.REEDS["van_walstijn"]
    g = (1100.0 + 2900.0 * np.arange(n) / n) / PM
    z = np.full(n, Z.mouthpiece(0.0, H, PM)[0])
    a1, b0 = Z.loss_filter(L, sr)
    ac = math.exp(-2 * math.pi * Z.TONEHOLE_FC / sr)
    D = 2 * sr * L / Z.C
    p, _ = Z._kernel_ka(g, z, D, a1, b0, ac, PM, fr, q * 2 * math.pi * fr, sr, 1.0, 0.0)
    pc, _ = Z._kernel(g, z, D, a1, b0, ac, *Z.reed_coefs(sr, fr, q))
    kf = (Z.RHO * Z.C / (math.pi * Z.BORE_R ** 2)) * H / PM
    pf, _ = Z._kernel_ka(g, z, D, a1, b0, ac, PM, fr, q * 2 * math.pi * fr, sr, 1.0, kf)
    seg = lambda x, t: x[int(t * sr):int(t * sr) + 8192]            # noqa: E731
    rms = [np.std(seg(p, t)) for t in (0.6, 1.5, 2.5, 3.5, 4.2)]
    assert all(b > a for a, b in zip(rms, rms[1:])), rms
    rise = lambda x: 1200 * math.log2(Z._f0(seg(x, 4.2), sr, Z.C / (4 * L)) / Z._f0(seg(x, 0.6), sr, Z.C / (4 * L)))  # noqa: E731
    assert 1.0 < rise(p) < 6.0 and 8.0 < rise(pc) < 15.0, (rise(p), rise(pc))
    # the flow the reed sweeps, Sr(dp) H dz/dt: Sr 0.78e-4 at rest, 0.27e-4 at 5100 Pa; it moves the ramp's pitch
    assert abs(np.interp(5100.0, Z.SR_DP, Z.SR_FIG) - 0.27e-4) < 1e-12
    assert abs(rise(pf) - rise(p)) > 0.5, (rise(pf), rise(p))
    # Ka(dp) on another reed: its knee moved to P_M / 2; the note still sounds
    y = Z.clarinet(220.0, 0.5, 44100.0, stiffening=True)
    assert np.std(y[int(0.25 * 44100):]) > 1e-4


def test_lip_force_moves_the_closing_pressure():
    """Almeida 2013: dF/dP about -3 cm^2, so 0.6 N = 20 mbar."""
    assert abs(Z.mouthpiece(0.5)[1] - Z.mouthpiece(0.5, lip_force=0.6)[1] - 2000.0) < 1e-6
    a = Z.clarinet(220.0, 0.5, 44100.0, blow=0.3)
    b = Z.clarinet(220.0, 0.5, 44100.0, blow=0.3, lip_force=0.6)
    assert abs(_peak(b, t1=0.5) / 220.0 - 1) < 0.02 and not np.allclose(a, b)


# ------------------------------------------------------------------------------------------ Meyer
def test_flutter_is_25_hz_and_15_db():
    y = I.render_note("flute", 880.0, 1.2, SR, 0.8, flutter=1.0, breath=0.0)
    w = int(0.004 * SR)
    e = np.sqrt(np.convolve(y ** 2, np.ones(w) / w, "valid"))[int(0.2 * SR):int(1.0 * SR)]
    assert 12.0 < _db(np.percentile(e, 99), np.percentile(e, 1)) < 17.0
    S = np.abs(np.fft.rfft((e - e.mean()) * np.hanning(e.size), 1 << 16))
    assert abs(np.fft.rfftfreq(1 << 16, 1 / SR)[np.argmax(S)] - 25.0) < 1.0


def test_noise_peaks_follow_the_register():
    """The option is additive: y1 - y0 is the noise alone, measured on its own. Meyer: f0/2 and odd multiples E5-D6,
    f0/3 and multiples not of 3 from Eb6."""
    def added(f, at):
        d = I.render_note("flute", f, 0.8, SR, 0.8, noise_peaks=1.0) - I.render_note("flute", f, 0.8, SR, 0.8)
        return _bands(d, at, (1,), 0.2)[0]
    assert _db(added(659.3, 659.3 / 2), added(659.3, 659.3 * 2 / 3)) > 15.0
    assert _db(added(1244.5, 1244.5 / 3), added(1244.5, 1244.5 / 2)) > 15.0
    assert np.array_equal(I.render_note("flute", 440.0, 0.3, SR, 0.8), I.render_note("flute", 440.0, 0.3, SR, 0.8, noise_peaks=1.0))


def test_flute_precursor_is_ten_db_for_50_ms():
    f = 523.3
    y = I.render_note("flute", f, 0.8, SR, 0.8, precursor=1.0, breath=0.0)
    b = F.bandpass(y, 8 * f, SR, 8.0)
    head = np.sqrt(np.max(np.convolve(b[:int(0.06 * SR)] ** 2, np.ones(441) / 441, "valid")))
    d = _db(head, np.std(b[y.size // 2:]))
    assert 6.0 < d < 14.0, d
    assert np.std(b[int(0.07 * SR):int(0.12 * SR)]) < head / 2


def test_bell_up_raises_the_top_by_10_db():
    f = 466.2
    a = I.render_note("trumpet", f, 0.8, SR, 0.9)
    b = I.render_note("trumpet", f, 0.8, SR, 0.9, bell_up=1.0)
    hi = lambda y: np.std(F.highpass(y[int(0.2 * SR):], 3000.0, SR, order=4))  # noqa: E731
    mid = lambda y: np.std(F.bandpass(y[int(0.2 * SR):], 932.4, SR, 8.0))  # noqa: E731
    assert 8.0 < _db(hi(b), hi(a)) < 12.0, _db(hi(b), hi(a))         # "more than 10 dB above 2000 Hz"
    assert _db(mid(b), mid(a)) < -0.5, _db(mid(b), mid(a))           # the loss at 400-1200 Hz


def test_attack_table_and_rendered_rise():
    assert A.attack_ms("trumpet", 165.0, 0.0) == 180.0 and A.attack_ms("trumpet", 1047.0, 0.0) == 40.0    # Meyer soft
    assert A.attack_ms("trumpet", 1047.0, 1.0) == 20.0 and A.attack_ms("tuba", 32.7, 1.0) == 60.0          # Meyer dry
    assert A.attack_ms("trombone", 523.0, 1.0) == 20.0 and A.attack_ms("oboe", 233.0, 0.0) == 100.0

    def t90(vel):
        y = I.render_note("trumpet", 165.0, 0.8, SR, vel, meyer=1)
        w = int(0.005 * SR)
        e = np.sqrt(np.convolve(y ** 2, np.ones(w) / w, "valid"))
        return np.argmax(e > 0.9 * np.max(e[:int(0.5 * SR)])) / SR
    assert t90(0.1) > 2.5 * t90(1.0), (t90(0.1), t90(1.0))
    assert np.array_equal(I.render_note("french_horn", 220.0, 0.3, SR, 0.8), I.render_note("french_horn", 220.0, 0.3, SR, 0.8, meyer=0))


def test_brass_precursor_sits_where_meyer_puts_it():
    for name, (lo, hi) in (("trumpet", (0.010, 0.015)), ("french_horn", (0.010, 0.030))):
        a = I.render_note(name, 220.0, 0.3, SR, 0.9)
        b = I.render_note(name, 220.0, 0.3, SR, 0.9, precursor=1.0)
        d = np.abs(b - a)
        assert lo <= np.argmax(d) / SR <= hi, (name, np.argmax(d) / SR)
        assert d[int(0.04 * SR):].max() < 0.05 * d.max()


def test_clarinet_odd_over_even_d3_to_d4():
    """Meyer 2009, 3.2: odd partials over even by more than 25 dB, up to 40, from D3 to D4. reed_tube only:
    clarinet(model=impedance) measures 23.6 dB at A3, short of it, and is left open in suivi/T_Vents.md."""
    for f in (146.8, 220.0, 293.7):
        h1, h2 = _bands(I.render_note("reed_tube", f, 0.7, SR, 0.9), f, (1, 2))
        assert _db(h1, h2) > 25.0, (f, _db(h1, h2))


# ------------------------------------------------------------------------------------------ Bessel
def test_bessel_bore_exponent_and_modes():
    """S ~ (x0 - x)^(-1.26) (C&K, nu 0.63). The flare pulls the low modes down from a cylinder's 1:3:5: measured."""
    S = T.bessel_bore(1.1)
    assert abs(math.log(S(0.9) / S(0.0)) / math.log(1.1 / 0.2) - 1.26) < 1e-9
    r = T.WebsterTube(1.4, S, radius_m=0.006, loss=1.0).resonances(fmax=1200.0, count=4)
    c = T.WebsterTube(1.4, 1.0, radius_m=0.006, loss=1.0).resonances(fmax=1200.0, count=4)
    assert r[1] / r[0] < 2.4 < 2.9 < c[1] / c[0], (r, c)
