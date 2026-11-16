import numpy as np

from genny import instruments as I
from genny import instruments_extra as X
from genny import viscous as V
from genny.tubes import C_AIR, _pipe

SR = 44100


def _f0(y, lo, hi):
    y = y[int(0.15 * SR):int(0.9 * SR)]
    S = np.abs(np.fft.rfft(y * np.hanning(len(y)), 1 << 18))
    f = np.fft.rfftfreq(1 << 18, 1 / SR)
    m = (f > lo) & (f < hi)
    return f[m][np.argmax(S[m])]


def _cents(a, b):
    return 1200 * np.log2(a / b)


def test_steam_speed():
    assert abs(X.steam_speed(1.0) - 498.0) < 3.0
    assert abs(X.steam_speed(1.0) / C_AIR - 1.45) < 0.02
    assert abs(_cents(X.steam_speed(0.0), X.steam_speed(1.0)) / 100) > 1.0 and abs(_cents(X.steam_speed(0.0), X.steam_speed(1.0)) / 100) < 1.4


def test_pipe_length_follows_c():
    L_air, L_steam = _pipe(220.0, 0.03, 1.0).L, _pipe(220.0, 0.03, 1.0, c=498.0).L
    assert abs(L_steam / L_air - 498.0 / C_AIR) < 0.08


def test_calliope_in_tune_and_cold_flat():
    f = 261.63
    assert abs(_cents(_f0(I.render_note("calliope", f, 1.0, SR, 0.9, waver=0.0), 200, 330), f)) < 8
    cold = _f0(I.render_note("calliope", f, 1.0, SR, 0.9, waver=0.0, warm=0.0), 180, 330)
    assert -140 < _cents(cold, f) < -100


def test_calliope_chord_sags_and_low_pressure_is_duller():
    f = 261.63
    one = _f0(I.render_note("calliope", f, 1.0, SR, 0.9, waver=0.0), 200, 330)
    four = _f0(I.render_note("calliope", f, 1.0, SR, 0.9, waver=0.0, sag=3), 200, 330)
    assert 5 < _cents(one, four) < 30
    half = _f0(I.render_note("calliope", f, 1.0, SR, 0.9, waver=0.0, pressure=0.5), 200, 330)
    assert half < one
    assert X.calliope_pitch(f, pressure=0.5)[1] ** 2 < 0.26          # hiss against the note falls with the pressure squared


def test_flexatone_two_octaves_and_linear():
    f = 261.63
    lo = _f0(I.render_note("flexatone", f, 1.0, SR, 0.9, vibrato=0.0), 200, 1200)
    hi = _f0(I.render_note("flexatone", f, 1.0, SR, 0.9, squeeze=1.0, vibrato=0.0), 600, 1600)
    assert abs(np.log2(hi / lo) - 2.0) < 0.2
    mid = _f0(I.render_note("flexatone", f, 1.0, SR, 0.9, squeeze=0.5, vibrato=0.0), 400, 1400)
    assert abs(mid / lo - 2.5) < 0.2


def test_flexatone_strikes_jitter_naturally():
    rng = np.random.default_rng(0)
    turns, hits = X._shake(2 * SR, SR, 11.0, rng)
    t = np.sort([k for k, _ in hits]) / SR
    assert abs(len(t) / 2.0 - 11.0) < 3.0
    d = np.diff(t)
    d = d[d > 0.03]
    assert np.std(d) / np.mean(d) > 0.03


def test_burst_freq_calibration_and_water():
    assert abs(V.burst_freq(0.0055, mu=30.0) - 42.0) < 2.0 and abs(V.burst_freq(0.0205, mu=30.0) - 22.0) < 1.5
    assert abs(V.burst_freq(0.01) - 30.0) < 2.0
    assert abs(V.burst_freq(0.01, calibration="liu") - 140.0) < 5.0
    assert 1500.0 < V.burst_freq(0.002, mu=1e-3, rho=1000.0, sigma=0.072) < 4000.0
    assert V.burst_freq(0.01, mu=10.0) > V.burst_freq(0.01, mu=30.0) > V.burst_freq(0.01, mu=100.0)
    rs = np.linspace(0.003, 0.03, 12)
    assert np.all(np.diff([V.burst_freq(r) for r in rs]) < 0)
    mus = np.logspace(-3, 3, 60)
    f = np.array([V.burst_freq(0.005, mu=m, rho=1000.0, sigma=0.072) for m in mus])
    assert np.all(np.diff(f) <= 1e-9) and np.max(np.abs(np.diff(np.log(f)))) < 0.25
