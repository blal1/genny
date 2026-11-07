"""Percussion theme (Lecture profonde 2026-10-29): sourced numbers checked on the rendered signal."""
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from genny import percussion as P
from genny.analysis import detect_clicks
from genny.drums import render_drum

SR = 44100


def _level(y, f, t, win=0.2):
    i, m = int(t * SR), int(win * SR)
    X = np.abs(np.fft.rfft(y[i:i + m] * np.hanning(m)))
    fr = np.fft.rfftfreq(m, 1 / SR)
    return 20 * np.log10(X[(fr > 0.98 * f) & (fr < 1.02 * f)].max() + 1e-15)


def _t60(y, f, t1, t2):
    return 60 * (t2 - t1) / (_level(y, f, t1) - _level(y, f, t2))


def _clean(y):
    assert np.isfinite(y).all() and np.abs(y).max() > 0.05
    assert len(detect_clicks(y[-int(0.05 * SR):])) == 0                 # the tail fades, no step


def test_farnell_kick_phase_synchronised_sweep():
    """70 ms, (2.413 line)^2 = 5.82 cycles (mean 83 Hz), every hit identical."""
    y = render_drum("kick", SR, 1.0, model="farnell")
    assert len(y) == int(round(0.07 * SR))
    zc = np.flatnonzero(np.diff(np.sign(y)) != 0)
    assert 11 <= len(zc) <= 13                                          # 2 x 5.82 half-cycles
    assert np.array_equal(y, render_drum("kick", SR, 1.0, model="farnell"))
    assert abs(y[-2]) < 1e-3                                            # the line ends at 0
    half = np.diff(zc[:3]).mean() / SR                                  # first half-periods: ~166 Hz at the start
    assert 120 < 0.5 / half < 200


def test_triangle_meyer_t60_law():
    """30 s at the 400 Hz partial, /2 per octave: 30 * 400 / 2100 = 5.71 s; 4 ms attack."""
    y = render_drum("triangle", SR, 1.0)
    _clean(y)
    assert abs(_t60(y, 2100, 0.2, 3.0) / 5.714 - 1) < 0.1
    assert abs(_t60(y, 400, 0.2, 3.0) / 30.0 - 1) < 0.15
    assert np.argmax(np.abs(y[:int(0.02 * SR)])) >= int(0.003 * SR)
    p = render_drum("triangle", SR, 1.0, axis="parallel")
    X = np.abs(np.fft.rfft(p[:SR] * np.hanning(SR)))
    fr = np.fft.rfftfreq(SR, 1 / SR)
    assert abs(fr[np.argmax(X)] - 400) < 3
    for f in (1600, 2000):                                              # the two near-harmonic partials
        assert X[(fr > f - 5) & (fr < f + 5)].max() > 0.3 * X.max()
    assert len(render_drum("triangle", SR, 1.0, open=0)) == int(0.3 * SR)


def test_bass_drum_meyer():
    """<50 Hz ring 15 s; 4 s at 200-400 Hz, /2 per octave above; a drop up to 140 cents in the
    first second; centred hit favours ring modes (duller), edge harsher; the rute is harder than the felt."""
    assert abs(float(P.meyer_bass_drum_t60(100.0)) - 8.0) < 0.5        # median of ~8 s
    assert float(P.meyer_bass_drum_t60(300.0)) == 4.0 and float(P.meyer_bass_drum_t60(800.0)) == 2.0
    y = render_drum("bass_drum", SR, 1.0)
    _clean(y)
    assert abs(_t60(y, 50.0, 0.5, 3.0) / 15.0 - 1) < 0.15
    f02 = 50.0 * 5.5201 / 2.4048                                        # (0,2) ring mode, 114.8 Hz
    assert abs(_t60(y, f02, 1.5, 3.0) / float(P.meyer_bass_drum_t60(f02)) - 1) < 0.15
    sos = butter(4, [40, 62], "bandpass", fs=SR, output="sos")

    def f01(v, t):
        b = render_drum("bass_drum", SR, v, pos=0.0)
        fi = np.diff(np.unwrap(np.angle(hilbert(sosfiltfilt(sos, b))))) * SR / (2 * np.pi)
        return np.median(fi[int((t - 0.01) * SR):int((t + 0.01) * SR)])
    drop = 1 + (2 ** (140 / 1200) - 1) * np.exp(-0.4 / 0.25)            # 0.4 s: clear of the filter onset
    assert abs(f01(1.0, 0.4) / (50 * drop) - 1) < 0.002
    assert f01(1.0, 0.05) > 50 * 2 ** (100 / 1200)                      # most of the 140 cents at the onset
    assert abs(f01(1.0, 1.5) / 50 - 1) < 0.005                          # settled after the first second
    assert abs((f01(0.5, 0.4) / 50 - 1) / (drop - 1) - 0.25) < 0.05    # vel^2
    fr = np.fft.rfftfreq(SR, 1 / SR)
    lim = fr < 2000

    def cen(b):
        X = np.abs(np.fft.rfft(b[:SR] * np.hanning(SR)))[lim] ** 2
        return (fr[lim] * X).sum() / X.sum()
    centre, edge = (render_drum("bass_drum", SR, 1.0, pos=p) for p in (0.0, 0.8))
    assert cen(edge) > 1.1 * cen(centre)
    X = np.abs(np.fft.rfft(centre[:SR] * np.hanning(SR)))
    f11 = 50.0 * 3.8317 / 2.4048                                        # (1,1): zero amplitude at the centre
    assert X[(fr > f11 - 1) & (fr < f11 + 1)].max() < 0.01 * X.max()
    assert cen(render_drum("bass_drum", SR, 1.0, mallet="rute")) > 1.1 * cen(y)


def test_head_bend_scales_with_velocity_squared():
    """Cubic hardening: the onset pitch excess goes as amplitude^2, here vel^2."""
    flat = P._head(SR // 5, SR, 100.0, [(1.0, 1.0, 1.0)], bend=0.0)

    def lead(v):                                                        # extra phase, in samples, at the 8th crossing
        h = P._head(SR // 5, SR, 100.0, [(1.0, 1.0, 1.0)], bend=0.06, bend_tau=0.02, vel=v)
        z = [np.flatnonzero(np.diff(np.sign(s)) != 0)[8] for s in (flat, h)]
        return z[0] - z[1], z[0] / SR
    full, t = lead(1.0)
    assert abs(full - 0.06 * 0.02 * (1 - np.exp(-t / 0.02)) * SR) <= 3   # f0 bend tau (1 - e^-t/tau) cycles ahead
    assert abs(lead(0.5)[0] / full - 0.25) < 0.1


def test_membrane_drum_bend_scales_with_velocity_squared():
    """MembraneDrum (conga, bongo, djembe, tabla, tom, snare): the same cubic-spring pitch drop as _head."""
    from genny.physics import MembraneDrum

    def lead(v):                                                        # lag of the bent render behind the flat one
        a, b = (np.asarray(MembraneDrum(150.0, air_coupling=0.0, bend=bd).strike(0.2, velocity=v, sr=SR))
                for bd in (0.0, 0.06))
        i, j = int(0.06 * SR), int(0.12 * SR)
        lags = np.arange(-80, 81)
        c = [np.dot(a[i:j], b[i + k:j + k]) for k in lags]
        return -lags[int(np.argmax(c))]
    full = lead(1.0)
    assert abs(full - 0.06 * 0.02 * SR) <= 6                            # f0 bend tau cycles ahead once settled
    assert abs(lead(0.5) / full - 0.25) < 0.1


def test_snare_meyer():
    """Strongest partials T60 ~ 1 s, spectral maximum 300-1000 Hz; the wires first fall back together ~9 ms after the hit."""
    y = render_drum("snare", SR, 1.0)
    _clean(y)
    edc = np.cumsum((y ** 2)[::-1])[::-1]
    edc = 10 * np.log10(edc / edc[0] + 1e-30)
    i, j = np.argmax(edc < -5), np.argmax(edc < -35)
    assert 0.8 < 60 * (j - i) / SR / 30 < 1.2
    X = np.abs(np.fft.rfft(y, 4 * SR))
    fr = np.fft.rfftfreq(4 * SR, 1 / SR)
    assert 300 <= fr[np.argmax(X)] <= 1000
    hi = sosfiltfilt(butter(4, 1500, "highpass", fs=SR, output="sos"), y) ** 2
    assert hi[int(0.009 * SR):int(0.012 * SR)].mean() > 10 * hi[int(0.002 * SR):int(0.008 * SR)].mean()


def test_membrane_radiation_follows_premier_table():
    """(0,n) modes radiate as monopoles: T60 ratios to the neighbouring (m,1) modes as the Premier 2500 table."""
    from genny.physics import MembraneDrum
    plain, rad = (MembraneDrum(150.0, radiation=r)._modes(0.3, "tone").decays for r in (0.0, 1.0))
    assert np.allclose((rad / plain)[[0, 3, 8]], [2.8 / 0.4, 4.2 / 0.5, 4.6 / 1.0])
    assert np.allclose((rad / plain)[[1, 2, 4, 5, 6, 7]], 1.0)


def test_808_cowbell_and_hat_oscillators():
    """TR-808 circuit: cowbell on the 540 and 800 Hz oscillators; the square waves run at 47.98 % duty."""
    from genny import osc as O
    y = render_drum("cowbell", SR, 1.0, decay=1.0)
    X = np.abs(np.fft.rfft(y))
    fr = np.fft.rfftfreq(len(y), 1 / SR)
    pk = [fr[(fr > lo) & (fr < hi)][np.argmax(X[(fr > lo) & (fr < hi)])] for lo, hi in ((450, 650), (700, 950))]
    assert abs(pk[0] - 540) < 3 and abs(pk[1] - 800) < 3
    p = O.pulse(100.0, SR, SR, 0.4798)
    assert abs((p > 0).mean() - 0.4798) < 0.002
    h = render_drum("hat808", SR, 1.0)
    assert np.isfinite(h).all() and np.abs(h).max() > 0.05
    # on the hat: the six oscillators, and second harmonics that only a duty off 50 % gives:
    # sin(2 pi D) / (2 sin(pi D)) = -24 dB, +12 dB from the two 7 kHz high-passes over an octave
    h = render_drum("hat808", SR, 1.0, open=1, decay=4.0)
    X = 20 * np.log10(np.abs(np.fft.rfft(h * np.hanning(len(h)))) + 1e-12)
    fr = np.fft.rfftfreq(len(h), 1 / SR)
    pk = lambda f, bw: (fr[(fr > f - bw) & (fr < f + bw)][np.argmax(X[(fr > f - bw) & (fr < f + bw)])],
                        X[(fr > f - bw) & (fr < f + bw)].max())
    for f in (205.3, 304.4, 369.6, 522.7, 540.0, 800.0):
        assert abs(pk(f, 8.0)[0] - f) < 0.5, f
    for f in (205.3, 800.0):
        assert -16.0 < pk(2 * f, 2.0)[1] - pk(f, 2.0)[1] < -8.0, f
    assert pk(250.0, 2.0)[1] < pk(205.3, 2.0)[1] - 60.0                 # nothing between the partials


def test_low_bells_second_strike_note():
    """Nominal under ~500 Hz: a second note on I-7 / 2, about a fourth above (Petersglocke 253.4 and 328.0 Hz)."""
    from genny.foley import _hibbert_modes, church_bell, _HIBBERT_G
    f, _, g = _hibbert_modes("petersglocke", 328.0, 1240.0, "second", 8.0)
    assert abs(f[3] / 2 / 328.0 - 1) < 0.01
    assert abs(f[0] - 252.3) < 0.01 and np.allclose(g, np.array(_HIBBERT_G[4:11]) / max(_HIBBERT_G[4:11]))
    f, _, _ = _hibbert_modes("big_ben", 327.0, 1240.0, "strike", 8.0)
    assert abs(f[0] - 335.1) < 0.01
    y = church_bell(253.4, 0.5, SR, type="petersglocke", tune="strike")
    assert np.isfinite(y).all() and np.max(np.abs(y)) > 0.01


def test_bell_size_law():
    """ln d = 6.95 - 1.04 ln f_n: 700 Hz for 1.147 m; diameter replaces the note."""
    from genny.foley import bell_nominal, _hibbert_modes, church_bell
    assert abs(bell_nominal(1.147) / 700.0 - 1) < 0.005
    y = church_bell(220.0, 0.5, SR, diameter=1.147)
    X = np.abs(np.fft.rfft(y[:SR // 2] * np.hanning(len(y[:SR // 2]))))
    fr = np.fft.rfftfreq(len(y[:SR // 2]), 1 / SR)
    band = (fr > 650) & (fr < 750)
    assert abs(fr[band][np.argmax(X[band])] / bell_nominal(1.147) - 1) < 0.01


def test_ariutti_additive_bell():
    """(osc(r f) line)^2 amp: partials at 2 r f; the three loudest at r = 1, 2.803, 5.074, level 1 : 0.891 : 0.794."""
    from genny.foley import church_bell
    y = church_bell(440.0, 1.0, SR, type="ariutti")
    _clean(y)
    m = SR // 2
    X = np.abs(np.fft.rfft(y[:m] * np.hanning(m)))
    fr = np.fft.rfftfreq(m, 1 / SR)
    lv = [X[(fr > 0.99 * f) & (fr < 1.01 * f)].max() for f in (440.0, 2.803 * 440, 5.074 * 440)]
    assert abs(lv[1] / lv[0] - 0.891) < 0.08
    assert fr[np.argmax(X)] == fr[(fr > 435) & (fr < 445)][np.argmax(X[(fr > 435) & (fr < 445)])]


def test_timpani_measured_table():
    """Default timpani = the measured kettledrum: 11 on the note, 21 at 1.50 and 31 at 1.98; T60 3.7 s (21), 4.6 s (31);
    the hand brings the principal down to 0.2 s high in the range."""
    from genny.acoustic import timpani
    y = timpani(145.0, 4.0, SR, decay=1.6)
    _clean(y)
    assert abs(_t60(y, 1.50 * 145, 0.5, 2.5) / 3.7 - 1) < 0.15
    assert abs(_t60(y, 1.98 * 145, 0.5, 2.5) / 4.6 - 1) < 0.15
    assert abs(_t60(y, 145, 0.5, 2.5) / 2.3 - 1) < 0.15
    for note in (65.41, 130.81):                                        # mode 11 on the written note
        z = timpani(note, 2.0, SR)[int(0.5 * SR):int(1.5 * SR)]
        X = np.abs(np.fft.rfft(z * np.hanning(len(z))))
        fr = np.fft.rfftfreq(len(z), 1 / SR)
        band = (fr > 0.95 * note) & (fr < 1.05 * note)
        assert abs(fr[band][np.argmax(X[band])] - note) <= 1.0
    h = timpani(196.0, 2.0, SR, damp=1.0)
    assert abs(_t60(h, 196, 0.1, 0.3, ) / 0.2 - 1) < 0.15


def test_tubular_bell_modes_81_121_169():
    from genny.acoustic import tubular_bell
    y = tubular_bell(220.0, 1.0, SR)
    X = np.abs(np.fft.rfft(y[:SR] * np.hanning(SR)))
    fr = np.fft.rfftfreq(SR, 1 / SR)
    pk = lambda f: fr[(fr > 0.97 * f) & (fr < 1.03 * f)][np.argmax(X[(fr > 0.97 * f) & (fr < 1.03 * f)])]
    for k in (81, 121, 169):
        assert abs(pk(220.0 * k / 40.5) / (220.0 * k / 40.5) - 1) < 0.003
    assert fr[np.argmax(X)] == pk(440.0)                                  # mode 4, an octave above the note


def test_ola_has_no_search():
    """OLA (WSOLA with no tolerance) for the percussive part of HPS: frames are taken at the mapped positions."""
    from genny import spectral as Sp
    x = np.random.default_rng(1).standard_normal(SR)
    M = int(0.01 * SR) // 2 * 2
    y = Sp._wsola(x, np.array([0.0, SR]), np.array([0.0, SR]), SR, M, SR, search=False)
    assert np.allclose(y[M:-M], x[M:-M], atol=1e-6)
    z = Sp.stretch(x, SR, 1.5, mode="hps")
    assert np.isfinite(z).all() and abs(len(z) / (1.5 * SR) - 1) < 0.01


def test_marimba_bar_tuned_on_two_ratios():
    """Marimba bars: the arch puts partials 2 and 3 at 4 and 10 times the first."""
    from genny.fdstring import Bar, fd_bar
    s = Bar(440.0, SR, ("free", "free"), (440.0, 1.0, 4000.0, 0.2), 32, (4.0, 10.0))
    assert abs(s.modes[1] / s.modes[0] / 4.0 - 1) < 0.01 and abs(s.modes[2] / s.modes[0] / 10.0 - 1) < 0.01
    y = fd_bar(220.0, 0.5, SR, tuning="marimba", material="metal", position=0.3)      # off the centre node of mode 2
    m = len(y)
    X = np.abs(np.fft.rfft(y * np.hanning(m)))
    fr = np.fft.rfftfreq(m, 1 / SR)
    for k in (4, 10):
        b = (fr > 0.97 * k * 220) & (fr < 1.03 * k * 220)
        assert abs(fr[b][np.argmax(X[b])] / (k * 220) - 1) < 0.01


def test_rosewood_bar_viscous_loss():
    """E 2.13e10 Pa, eta 5.0e3 N s/m^2 (Kelvin-Voigt): T60 = 2 ln 1000 E / (omega^2 eta), 7.7 s at 440 Hz, 0.093 s at 4 kHz."""
    from genny.fdstring import _MATERIAL
    lo, hi = _MATERIAL["rosewood"]
    assert abs(lo - 7.70) < 0.01 and abs(hi - 0.0932) < 0.001


def test_l_membrane():
    """Three unit squares: lambda 9.6397, 15.1973, 2 pi^2, 29.5215 ...; f ~ sqrt(lambda). The grid confirms 9.6397."""
    from genny import contact as C
    lam, sh = C._l_modes()
    assert abs(lam[0] - 9.6397238) < 1e-6 and abs(lam[2] - 2 * np.pi ** 2) < 1e-9
    b = C.ShapeBody("l_membrane", 100.0, position=(0.25, 0.25), n_modes=12)
    assert np.allclose(b.freqs, 100.0 * np.sqrt(np.array(C._L_LAMBDA) / C._L_LAMBDA[0]))
    assert np.abs(b.row((0.75, 0.75))).max() < 1e-9 and np.abs(b.row((0.25, 0.25))).max() > 0.1
    from scipy.sparse import diags, identity, kron
    from scipy.sparse.linalg import eigsh
    n = 64
    x = -1.0 + np.arange(1, 2 * n) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    k = np.flatnonzero(~((X > -1e-12) & (Y > -1e-12)).ravel())
    D = diags([-np.ones(2 * n - 2), 2.0 * np.ones(2 * n - 1), -np.ones(2 * n - 2)], [-1, 0, 1]) * n * n
    A = (kron(D, identity(2 * n - 1)) + kron(identity(2 * n - 1), D)).tocsr()[k][:, k]
    assert abs(eigsh(A, k=1, sigma=0.0)[0][0] / 9.6397238 - 1) < 0.001


def test_l_membrane_renders():
    from genny.sfx import REGISTRY as S
    y = S["impact"]["fn"](SR, shape="l_membrane", material="metal")
    assert np.isfinite(y).all() and np.abs(y).max() > 0.01


def test_mode_radiation_monopole_share():
    """Rectangle, simple support: |int phi| / int |phi| = 1/(m n) for odd-odd modes, 0 otherwise; circle m > 0: 0.
    radiation=modes keeps the breathing mode and lowers the others by ka / sqrt 3."""
    from genny import contact as C
    from genny.sfx import REGISTRY as S
    b = C.ShapeBody("rect_membrane", 100.0, aspect=1.0, n_modes=10)
    q = b.monopole_share()
    for m, n, qq in zip(b._desc["m"], b._desc["n"], q):
        want = 1.0 / (m * n) if (m % 2 and n % 2) else 0.0
        assert abs(qq - want) < 0.01, (m, n, qq)
    c = C.ShapeBody("circ_membrane", 100.0, n_modes=6)
    assert np.all(c.monopole_share()[c._desc["m"] > 0] < 0.01) and c.monopole_share()[0] > 0.9
    a, z = (S["impact"]["fn"](SR, shape="rect_plate", material="metal", size=0.1, freq=150.0, radiation=r) for r in ("sphere", "modes"))
    assert np.isfinite(z).all() and not np.allclose(a, z)
