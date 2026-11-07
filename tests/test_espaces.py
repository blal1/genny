"""Theme Espaces: propagation and rooms against the numbers of their sources (Kuttruff, Meyer, Chowning, Damiano, STK)."""
import math

import numpy as np
import pytest
from scipy.signal import butter, freqz, sosfiltfilt

from genny import reverbs as R
from genny.physical import space as S

SR = 44100


def noise(sec=1.0, seed=1):
    return np.random.default_rng(seed).standard_normal(int(sec * SR))


def tone(f, sec=1.0):
    return np.sin(2 * np.pi * f * np.arange(int(sec * SR)) / SR)


def freq_of(y):
    z = y[np.abs(y) > 0]
    c = np.flatnonzero(np.diff(np.sign(z)) > 0)
    return SR * (len(c) - 1) / (c[-1] - c[0])


def band_db(y, f0, frac=1.1):
    Y = np.abs(np.fft.rfft(y)) ** 2
    fr = np.fft.rfftfreq(y.shape[0], 1.0 / SR)
    return 10 * np.log10(Y[(fr > f0 / frac) & (fr < f0 * frac)].sum())


def rms_db(y):
    return 10 * np.log10(np.mean(y ** 2))


def t60_of(y, lo=-5.0, hi=-30.0):
    e = np.cumsum((y ** 2)[::-1])[::-1]
    edc = 10 * np.log10(e / e[0] + 1e-30)
    i0, i1 = np.nonzero(edc <= lo)[0][0], np.nonzero(edc <= hi)[0][0]
    return -60.0 / np.polyfit(np.arange(i0, i1) / SR, edc[i0:i1], 1)[0]


# ---------------------------------------------------------------- doppler: path

def test_path_gives_the_moving_source_ratio_in_each_segment():
    c = 343.0
    # head on: 100 m/s for 1 s, then stopped: c/(c-v) = 1411.4 Hz (DAFX 5.11; Addendum c16), then 1000 Hz
    path = [[0, 0, 300, 0], [1, 0, 200, 0], [2, 0, 200, 0]]
    y = R.doppler(tone(1000.0, 2.0), SR, path=path, c=c, air=False, pan=False)
    n1 = int(1000 / 1411.5 * SR * 1.0)                                         # 1 s of emission arrives in 243/343 s
    assert abs(freq_of(y[2000:n1 - 2000]) - 343000 / 243) < 3
    assert abs(freq_of(y[n1 + 2000:-2000]) - 1000.0) < 2


def test_path_circle_has_no_doppler_and_constant_level():
    th = np.linspace(0, 2 * np.pi, 200)
    path = np.column_stack([np.linspace(0, 1, 200), 10 * np.cos(th), 10 * np.sin(th), np.zeros(200)]).tolist()
    y = R.doppler(tone(1000.0), SR, path=path, air=False, pan=False)
    assert abs(freq_of(y[2000:-2000]) - 1000.0) < 0.5
    assert abs(rms_db(y[2000:10000]) - rms_db(y[20000:28000])) < 0.1


def test_path_matches_the_straight_flyby():
    v, d = 40.0, 6.0
    n = SR
    straight = R.doppler(tone(800.0), SR, speed=v, closest=d, pan=False, air=False)
    path = [[0, -0.5 * v, d, 0], [1, 0.5 * v, d, 0]]
    y = R.doppler(tone(800.0), SR, path=path, pan=False, air=False)
    m = min(len(y), len(straight)) - 400
    assert np.sqrt(np.mean((y[400:m] - straight[400:m]) ** 2)) < 0.02


def test_path_rejects_supersonic_approach():
    with pytest.raises(ValueError):
        R.doppler(tone(100.0), SR, path=[[0, 0, 900, 0], [1, 0, 100, 0]])


# ---------------------------------------------------------------- doppler: real ground

def test_surface_follows_ground_effect_at_rest():
    x = noise(3.0)
    h = 3.0
    dry = R.doppler(x, SR, speed=0.0, closest=100.0, pan=False, air=False)
    wet = R.doppler(x, SR, speed=0.0, closest=100.0, pan=False, air=False, surface="grass", height=h)
    for f in (100.0, 250.0, 500.0, 1000.0):
        H = S.ground_effect(np.array([f]), 100.0, h / 2, h / 2, 300.0, 20.0)
        assert abs(band_db(wet, f, 1.03) - band_db(dry, f, 1.03) - 20 * np.log10(abs(H[0]))) < 1.5, f


def test_surface_dip_glides_along_the_pass():
    # asphalt, 20000 cgs (Damiano 2024 sec. 4): the first dip sits at c/(2 dr), from the extra path dr of the reflection
    x = noise(2.0)
    h = 3.0
    y = R.doppler(x, SR, speed=30.0, closest=4.0, pan=False, air=False, surface="asphalt", height=h)
    d = R.doppler(x, SR, speed=30.0, closest=4.0, pan=False, air=False)
    n = len(y)
    _, r, _, _ = R.flyby_path(len(x), SR, 30.0, 4.0)
    for at in (n // 2, n // 8):
        seg = slice(at - 1500, at + 1500)
        f_dip = R.C_AIR / (2.0 * (math.hypot(r[at], h) - r[at]))
        if f_dip < 0.4 * SR:
            assert band_db(y[seg], f_dip, 1.08) - band_db(d[seg], f_dip, 1.08) < -3.0, (at, f_dip)
    assert R._GROUND_SIGMA["asphalt"] == 20000.0


# ---------------------------------------------------------------- doppler: Chowning

def _wet_dry(D, x):
    kw = dict(speed=0.0, closest=D, ref=1.0, air=False, pan=False)
    return R.doppler(x, SR, **kw), R.doppler(x, SR, reverb=1.0, reverb_time=1.0, **kw)


def test_chowning_direct_1_over_d_reverb_1_over_sqrt_d():
    x = noise(1.0)
    d2, w2 = _wet_dry(2.0, x)
    d8, w8 = _wet_dry(8.0, x)
    # ref=1: gain capped 2 for closer only; direct 1/D
    assert abs(rms_db(d8) - rms_db(d2) + 20 * np.log10(4)) < 0.05
    e2 = w2[: len(d2)] - d2
    e8 = w8[: len(d8)] - d8
    assert abs(rms_db(e8) - rms_db(e2) + 10 * np.log10(4)) < 0.2


def test_chowning_far_reverb_comes_from_the_source_side():
    x = noise(1.0)
    wet = lambda D: R.doppler(x, SR, path=[[0, D, 0, 0], [1, D, 0, 0]], ref=1.0, air=False, reverb=1.0, reverb_time=0.5)[SR + 2000:]
    far = wet(8.0)                                                             # global .125/2.83, local right (7/8) sqrt2 /2.83
    expected = 20 * np.log10((0.125 + 0.875 * math.sqrt(2)) / 0.125)
    assert abs(rms_db(far[:, 1]) - rms_db(far[:, 0]) - expected) < 1.5
    near = wet(1.0)
    assert abs(rms_db(near[:, 1]) - rms_db(near[:, 0])) < 1.5


def test_shadow_dulls_the_far_ear():
    x = noise(1.5)
    path = [[0, -3, 4, 0], [1.5, -3, 4, 0]]                                    # s = -0.6: right ear is far
    a, b = R.doppler(x, SR, path=path, air=False), R.doppler(x, SR, path=path, air=False, shadow=True)
    pole = math.exp(-2 * np.pi * 1500.0 / SR)
    _, H = freqz([1 - pole], [1, -pole], worN=[6000.0, 200.0], fs=SR)
    m = 0.48
    for f, h in zip((6000.0, 200.0), H):
        want = 20 * np.log10(abs(1 - m + m * h))
        assert abs(band_db(b[:, 1], f) - band_db(a[:, 1], f) - want) < 0.5, f
        assert abs(band_db(b[:, 0], f) - band_db(a[:, 0], f)) < 0.01


def test_horn_low_eq15_and_truck_eq16():
    x = tone(2000.0, 2.0)
    y0 = R.doppler(x, SR, speed=20.0, closest=2.0, pan=False, air=False)
    y1 = R.doppler(x, SR, speed=20.0, closest=2.0, pan=False, air=False, horn="low")
    db = lambda y, i: rms_db(y[i - 200:i + 200])
    mid = int(np.argmin(R.flyby_path(len(x), SR, 20.0, 2.0)[1]))               # the closest point arrives before input time 1 s
    assert abs(db(y1, mid) - db(y0, mid) + 2.5) < 0.5                          # beside it: -2.5 dB
    assert abs(db(y1, 3000) - db(y0, 3000) - 1.5) < 0.2                        # far off axis: +1.5 dB
    bass = tone(100.0, 2.0)
    b0 = R.doppler(bass, SR, speed=20.0, closest=2.0, pan=False, air=False)
    b1 = R.doppler(bass, SR, speed=20.0, closest=2.0, pan=False, air=False, horn="low")
    assert abs(rms_db(b1) - rms_db(b0)) < 0.15                                 # 0 outside 800-6300 Hz
    t = R.doppler(x, SR, speed=20.0, closest=2.0, pan=False, air=False, horn="truck")
    assert abs(db(t, mid) - db(y0, mid) - 0.6) < 0.1                           # eq. 16 at phi = pi/2
    r0 = R.flyby_path(len(x), SR, 20.0, 2.0)[1][3000]
    u = math.asin(math.sqrt(r0 * r0 - 4.0) / r0)                               # pi/2 - phi at that sample
    assert abs(db(t, 3000) - db(y0, 3000) - (1.546 * u ** 3 - 1.425 * u ** 2 + 0.22 * u + 0.6)) < 0.1


# ---------------------------------------------------------------- decay formulas

def test_exponents_against_kuttruff():
    a = np.array([[0.5], [0.5]])
    ar = [1.0, 1.0]
    d = (5.0, 5.0, 5.0)
    assert abs(S.absorption_exponent(a, "sabine", ar, d)[0] - 0.5) < 1e-12
    assert abs(S.absorption_exponent(a, "eyring", ar, d)[0] - math.log(2)) < 1e-12
    # (5.33) with gamma^2 = 0.342 for a cube (Table 5.1)
    assert abs(S.absorption_exponent(a, "kuttruff", ar, d)[0] - math.log(2) * (1 - 0.171 * math.log(2))) < 1e-12
    m = S.absorption_exponent(np.array([[0.1], [0.9]]), "millington", ar, d)[0]
    assert abs(m + 0.5 * (math.log(0.9) + math.log(0.1))) < 1e-12
    assert S.path_variance((3, 3, 3)) == 0.342 and S.path_variance((2, 10, 20)) == 0.510 and S.path_variance((1, 10, 10)) == 0.613


def test_sabine_runs_long_past_alpha_02(monkeypatch):
    monkeypatch.setitem(R.ABSORPTION, "half", (0.5,) * 6)
    monkeypatch.setitem(R.ABSORPTION, "tenth", (0.1,) * 6)
    kw = dict(lx=10.0, ly=10.0, lz=10.0, walls="half", floor="half", ceiling="half")
    sab = R.room_info("hall", **kw)["t60"][2]
    eyr = R.room_info("hall", formula="eyring", **kw)["t60"][2]
    assert abs(sab / eyr - math.log(2) / 0.5) < 0.02                          # (5.24) against (5.25)
    kut = R.room_info("hall", formula="kuttruff", **kw)["t60"][2]
    assert eyr < kut < sab                                                     # "the reverberation time is longer than ... Eyring"
    k2 = dict(lx=10.0, ly=10.0, lz=10.0, walls="tenth", floor="tenth", ceiling="tenth")
    assert abs(R.room_info("hall", formula="eyring", **k2)["t60"][2] / R.room_info("hall", **k2)["t60"][2] - 0.1 / -math.log(0.9)) < 0.01


def test_diffuse_room_with_air_b32(monkeypatch):
    # Kinney/SPPS validation case of lot B32: 5 x 4 x 3 m, alpha 0.10, 20 C, 50 %: T(4 kHz)/T(500 Hz) = 0.865
    monkeypatch.setitem(R.ABSORPTION, "tenth", (0.1,) * 6)
    t = R.room_info("hall", lx=5.0, ly=4.0, lz=3.0, walls="tenth", floor="tenth", ceiling="tenth")["t60"]
    assert abs(t[5] / t[2] - 0.865) < 0.01


def test_schroeder_frequency_and_density():
    assert abs(S.schroeder_frequency(2.0, 2000.0) - 63.25) < 0.01               # 2000 sqrt(T/V), Kuttruff 3.32
    assert R.room_info("concert_hall")["schroeder_hz"] < 50.0                  # "below 50 Hz in large halls"
    i = R.room_info("bedroom")
    assert abs(i["reflections_per_s"] * i["mfp"] - R.C_AIR) < 1e-9


def test_room_modes_table_3_1():
    f, order = S.room_modes(4.7, 4.1, 3.1, 60.0, c=340.0)
    assert np.allclose(f[:4], [36.17, 41.46, 54.84, 55.02], atol=0.01)         # Kuttruff Table 3.1
    assert tuple(order[0]) == (1, 0, 0)


def test_modal_low_end_rings_at_a_mode():
    y = R.room(np.r_[1.0, np.zeros(SR // 10)], SR, preset="bedroom", mix=1.0, modal=True).mean(axis=1)
    info = R.room_info("bedroom")
    f, _ = S.room_modes(*info["dims"], 120.0)
    Y = np.abs(np.fft.rfft(y[1000:], 1 << 17))
    fr = np.fft.rfftfreq(1 << 17, 1.0 / SR)
    lo = (fr > 30) & (fr < 120)
    peak = fr[lo][np.argmax(Y[lo])]
    assert np.min(np.abs(f - peak)) < 1.5, (peak, f)
    plain = R.room(np.r_[1.0, np.zeros(SR // 10)], SR, preset="bedroom", mix=1.0).mean(axis=1)
    assert not np.allclose(y[:len(plain)], plain)
    kw = dict(lx=2.0, ly=3.0, lz=4.7, walls="plaster", floor="plaster", ceiling="plaster", furnish=4.0)
    i2 = R.room_info("hall", **kw)
    f1 = S.room_modes(*i2["dims"], 60.0)[0][0]                                 # 36.5 Hz, 57 Hz next: the lowest mode stands alone
    z = R.room(np.r_[1.0, np.zeros(SR // 10)], SR, mix=1.0, modal=True, **kw).mean(axis=1)
    band = sosfiltfilt(butter(2, [28.0, 46.0], "band", fs=SR, output="sos"), z)
    assert abs(t60_of(band[int(0.3 * SR):], -3, -25) / i2["t60"][0] - 1) < 0.15, f1     # delta = 6.91/T, T of the 125 Hz band held below it
    # the low end keeps its energy within 1 dB of the plain room, the band above the crossover is untouched
    P = np.abs(np.fft.rfft(plain[1000:], 1 << 17))
    assert abs(10 * np.log10(np.sum(Y[lo] ** 2) / np.sum(P[lo] ** 2))) < 1.0


def test_piston_gain_and_null():
    assert abs(S.piston_gain(0.01) - 1.0) < 1e-3
    assert abs(S.piston_gain(5.0) - 12.5 / (1.0 - 0.04347 / 5.0)) < 0.01        # J1(10) = 0.04347 in (10.3)
    assert S.piston_directivity(3.8317, math.pi / 2) < 1e-3                     # first zero of J1


def _tail_vs_direct(**kw):
    y = R.room(np.r_[1.0, np.zeros(SR // 2)], SR, mix=1.0, lx=20.0, ly=40.0, lz=15.0, walls="plaster", floor="wood", ceiling="plaster", furnish=3000.0, tail=1.5, **kw)
    return 10 * np.log10(np.sum(y[int(0.3 * SR):] ** 2) / np.sum(y[0, :] ** 2))


def test_critical_distance_takes_g_and_late_field_loses_with_distance():
    base = _tail_vs_direct(distance=10.0)
    assert abs(_tail_vs_direct(distance=10.0, source_gain=4.0) - base + 6.02) < 0.05     # r_c x sqrt g, Kuttruff 5.41
    a = _tail_vs_direct(distance=5.0)
    info = R.room_info("hall", lx=20.0, ly=40.0, lz=15.0, walls="plaster", floor="wood", ceiling="plaster", furnish=3000.0)
    d = [np.linalg.norm(np.subtract(*R._room_positions(info, m))) for m in (5.0, 10.0)]
    assert abs(base - a - (20 * np.log10(d[1] / d[0]) - 0.85 * (d[1] - d[0]) / 10)) < 0.05   # (r/rc)^2 less 0.85 dB per 10 m, Barron & Lee


# ---------------------------------------------------------------- presets and materials

@pytest.mark.parametrize("name,mid,low", [("church", 5.0, None), ("practice_room", 0.35, None), ("orchestra_studio", 1.2, 1.2),
                                          ("opera", 1.2, 1.5), ("concert_hall", 2.05, 2.4), ("arena", 0.85, None)])
def test_preset_decay_against_meyer(name, mid, low):
    t = R.room_info(name)["t60"]
    assert abs(np.mean(t[2:4]) - mid) < max(0.05 * mid, 0.05), t               # about the 4 % just noticeable difference (Kuttruff 7.5)
    if low:
        assert abs(np.mean(t[:2]) - low) < 0.05 * low, t


def test_preset_volumes_and_rc():
    assert abs(R.room_info("concert_hall")["volume"] - 14600) < 50             # Musikverein
    assert abs(R.room_info("concert_hall")["critical_distance"] - 4.81) < 0.2   # Meyer l.9352 prints 4.75, recalculated 4.81
    assert 30 <= R.room_info("practice_room")["volume"] <= 40
    y = R.room(np.r_[1.0, np.zeros(1000)], SR, preset="amphitheater", mix=1.0)
    # no diffuse field: nothing at all once the image sources stop (at most 0.6 s)
    assert np.sum(y[int(0.65 * SR):] ** 2) < 1e-20
    late = np.sum(R.room(np.r_[1.0, np.zeros(1000)], SR, preset="arena", mix=1.0)[int(0.65 * SR):] ** 2)
    assert late > 0


def test_kuttruff_materials_reach_room_info():
    assert R.ABSORPTION["curtain"] == (0.15, 0.45, 0.90, 0.92, 0.95, 0.95)      # Table 9.1
    assert R.ABSORPTION["musician"][2] == 1.06                                 # Table 6.2
    t = S.sabine_t60(100.0, {"foam": 130.0})
    assert abs(t[3] - 0.161 * 100.0 / (130.0 * 0.70)) < 1e-9


# ---------------------------------------------------------------- distance and barrier

def test_piston_off_axis_and_wall_proximity():
    x = noise(2.0)
    a = 0.1
    f0 = 3.8317 * R.C_AIR / (2 * np.pi * a)
    on = R.distance(x, SR, metres=1.0, air=False)
    off = R.distance(x, SR, metres=1.0, air=False, source_radius=a, angle=90.0)
    assert band_db(off, f0, 1.02) - band_db(on, f0, 1.02) < -25
    assert abs(band_db(off, 50.0) - band_db(on, 50.0)) < 0.1
    wall = R.distance(x, SR, metres=1.0, air=False, wall_distance=0.75)
    assert abs(10 ** ((band_db(wall, 30.0, 1.05) - band_db(on, 30.0, 1.05)) / 10) - 1.89) < 0.08   # 1 + sin(2kd)/(2kd)


def test_scatter_and_flat_room_levels():
    x = noise(0.5)
    plain = R.distance(x, SR, metres=41.0, air=False)
    sc = R.distance(x, SR, metres=41.0, air=False, scatter_mfp=20.0)
    assert abs(rms_db(sc) - rms_db(plain) - 20 * np.log10(math.exp(-1.0))) < 0.01    # Kuttruff 4.3: exp(-r/rs) in energy
    flat = R.distance(x, SR, metres=16.0, air=False, ref=2.0, flat_height=4.0, flat_alpha=0.2)
    w = lambda r: S.flat_room_energy(r, 4.0, 0.2)
    assert abs(rms_db(flat) - rms_db(x) - 10 * np.log10(w(16.0) / w(2.0))) < 0.05
    # Table 9.2 value of b inside (9.13), alpha 0.5, r = h: rho 0.5, b 1.903
    v = (2.0 ** -1.5 + 1.903 * 1.0 * (1.903 ** 2 + 1.0) ** -1.5) / (1.0 + 2.0)
    assert abs(float(S.flat_room_energy(1.0, 1.0, 0.5)) - v) < 1e-9
    # reverberation makes spreading slower than 1/r^2 in a low room with hard surfaces
    assert w(16.0) / w(2.0) > (2.0 / 16.0) ** 2 * 0.5 * (1 + 16.0 ** 2 / 16.0) ** -1.5


def test_barrier_follows_maekawa_fit():
    a = tone(343.2, 1.0)
    y = R.occlude(a, SR, material="metal", thickness=1.0, leak=1.0, path_diff=0.5)
    # N = 2 delta f / c = 1: IL = 10 log10 23 = 13.6 dB; transmission of 7.9 t/m2 steel is far below that
    assert abs(rms_db(y[2000:-2000]) - rms_db(a) + 10 * np.log10(23)) < 0.5
    b = R.occlude(tone(1.0 * 343.2, 1.0), SR, material="metal", thickness=1.0, leak=1.0, path_diff=0.0)
    assert abs(rms_db(b[2000:-2000]) - rms_db(a) + 10 * np.log10(3)) < 0.5


# ---------------------------------------------------------------- reverberators

@pytest.mark.parametrize("fn,kw", [(R.nrev, {}), (R.prcrev, {}), (R.jcrev, {"variant": "stk"})])
def test_stk_reverbs_hit_their_decay(fn, kw):
    y = fn(np.r_[1.0, np.zeros(1000)], SR, mix=1.0, t60=1.0, tail=2.5, **kw)
    assert y.ndim == 2 and not np.allclose(y[:, 0], y[:, 1])
    t = t60_of(y.mean(axis=1), -10, -40)
    assert abs(t - 1.0) < 0.25, t


def test_stk_lengths_are_scaled_primes():
    assert R._stk_len((341, 613, 1557, 2137), 44100, 44100.0).tolist() == [341, 613, 1557, 2137]
    assert R._stk_len((341, 613, 1557, 2137), 48000, 44100.0).tolist() == [373, 673, 1697, 2333]
    assert R._stk_len((1433, 347), 25641, 25641.0, always=True).tolist() == [1433, 347]


def test_coupled_room_has_kuttruffs_bend():
    tn, tf, vn, vf, ap = 0.5, 3.0, 500.0, 5000.0, 20.0
    d2, d1, C, bend = R.coupled_decay(tn, tf, vn, vf, ap)
    assert abs(bend - (10 / (1 - d1 / d2) * math.log10(1 / C) - 3)) < 1e-9 and 20 < bend < 40
    y = R.coupled_room(np.r_[1.0, np.zeros(100)], SR, mix=1.0, t60_near=tn, t60_far=tf, v_near=vn, v_far=vf, aperture=ap).mean(axis=1)
    y = sosfiltfilt(butter(2, [500.0, 4000.0], "band", fs=SR, output="sos"), y)
    hop = int(0.02 * SR)
    e = np.array([np.mean(y[i:i + hop] ** 2) for i in range(0, len(y) - hop, hop)])
    db = 10 * np.log10(e / e[0] + 1e-30)
    t = np.arange(len(db)) * 0.02
    knee_t = math.log(1 / C) / (2 * (d2 - d1))
    early = t < 0.5 * knee_t
    late = (t > 1.3 * knee_t) & (t < 2.0)
    assert abs(-60 / np.polyfit(t[early], db[early], 1)[0] / tn - 1) < 0.25
    assert abs(-60 / np.polyfit(t[late], db[late], 1)[0] / tf - 1) < 0.25
    # the two fitted lines meet at the bend, -(bend + 3) dB below the start of the fast line
    p1, p2 = np.polyfit(t[early], db[early], 1), np.polyfit(t[late], db[late], 1)
    meet = np.polyval(p1, (p2[1] - p1[1]) / (p1[0] - p2[0])) - p1[1]
    assert abs(-meet - (bend + 3.0)) < 3.0


def test_velvet_decorrelator():
    x = noise(1.0)
    y = R.decorrelate(x, SR, mode="velvet")
    rho = np.corrcoef(y[:, 0], y[:, 1])[0, 1]
    assert abs(rho) < 0.15
    assert abs(rms_db(y[:, 0]) - rms_db(x)) < 0.6                              # unit-energy FIR
    # EVN30: at most 30 impulses in 30 ms, envelope down 60 dB at the end
    imp = R.decorrelate(np.r_[1.0, np.zeros(4000)], SR, mode="velvet")
    n = int(0.03 * SR)
    nz = np.abs(imp[:n, 0]) > 1e-9
    assert 20 <= nz.sum() <= 31 and np.all(np.abs(imp[n + 1:, 0]) < 1e-12)
    late, early = np.abs(imp[int(0.025 * SR):n, 0]), np.abs(imp[:int(0.005 * SR), 0])
    assert 20 * np.log10(late.max() / early.max()) < -40


def test_defaults_unchanged_without_the_new_parameters():
    x = noise(0.5)
    assert np.allclose(R.doppler(x, SR, speed=30.0), R.doppler(x, SR, speed=30.0, path=None, surface=None, reverb=0.0, horn="none"))
    # Sabine stays the default decay
    assert np.array_equal(R.room_info("hall")["t60"], R.room_info("hall", formula="sabine")["t60"])


# ---------------------------------------------------------------- second pass

def test_costello_decays_and_matches_revsc_constants():
    y = R.costello(np.r_[1.0, np.zeros(100)], SR, mix=1.0, feedback=0.8, tail=6.0)
    assert np.isfinite(y).all() and y.shape[1] == 2 and not np.allclose(y[:, 0], y[:, 1])
    assert y.max() < 2.0
    # a louder feedback rings longer, each pass of about 75 ms loses 20 log10(fb) dB
    long = R.costello(np.r_[1.0, np.zeros(100)], SR, mix=1.0, feedback=0.9, tail=8.0)
    ta, tb = t60_of(y.mean(axis=1), -10, -40), t60_of(long.mean(axis=1), -10, -40)
    assert 1.5 < tb / ta < 3.0                                                 # ln(.8)/ln(.9) = 2.1
    assert [q[0] * 44100 for q in R._REVSC][:2] == pytest.approx([2473, 2767])


def test_room_metrics_on_an_exponential_decay():
    T = 2.0
    t = np.arange(4 * SR) / SR
    h = np.random.default_rng(3).standard_normal(t.size) * np.exp(-6.91 * t / T)
    h[0] = 8.0
    m = R.room_metrics(h, SR)
    assert abs(m["t20"] / T - 1) < 0.05
    pure = np.exp(-6.91 * t / T)
    want = 10 * np.log10(1 / np.exp(-13.82 * 0.08 / T) - 1)                    # C80 of an exponential, analytically
    assert abs(R.room_metrics(pure, SR)["c80"] - want) < 0.05
    ir = R.room(np.r_[1.0, np.zeros(1000)], SR, preset="concert_hall", mix=1.0)
    c80 = R.room_metrics(ir, SR)["c80"]
    assert -2.0 <= c80 <= 6.0, c80                                             # Meyer: -2..+4 dB in halls; measured here, the upper bound is ours


def test_audience_dip_depth_with_distance():
    x = noise(3.0)
    kw = dict(lx=30.0, ly=60.0, lz=15.0, walls="plaster", floor="wood", ceiling="plaster", furnish=2000.0, mix=1.0, tail=0.1)
    far0, far1 = R.room(x, SR, distance=22.5, **kw), R.room(x, SR, distance=22.5, audience=True, **kw)
    near0, near1 = R.room(x, SR, distance=2.0, **kw), R.room(x, SR, distance=2.0, audience=True, **kw)
    # the difference is what left the direct sound: at 150 Hz, G (1 - 10^(-depth/20)) of the input, G = direct gain of the impulse response
    G = R.room(np.r_[1.0, np.zeros(1000)], SR, distance=22.5, **kw)[0, 0]
    got = band_db(far1[:, 0] - far0[:, 0], 150.0, 1.01) - band_db(x, 150.0, 1.01)
    assert abs(got - 20 * np.log10(abs(G) * (1 - 10 ** (-15.0 / 20)))) < 1.0   # 15 dB at 22.5 m, middle of Meyer's 10-20 dB at 20-25 m
    assert band_db(near1[:, 0] - near0[:, 0], 150.0, 1.01) - band_db(far1[:, 0] - far0[:, 0], 150.0, 1.01) < 6.0   # the direct is louder near, the dip far shallower
    assert abs(band_db(far1[:, 0], 1000.0) - band_db(far0[:, 0], 1000.0)) < 0.1


def test_bell_rear_and_flute_and_drumhead():
    x = noise(1.0)
    for name, f, want in (("trumpet", 2000.0, -10.0), ("trumpet", 5000.0, -17.0), ("trumpet", 10000.0, -22.0), ("trombone", 4000.0, -18.0)):
        y = R.directivity(x, SR, instrument=name, angle=180.0)
        assert abs(band_db(y, f, 1.02) - band_db(x, f, 1.02) - want) < 0.3, (name, f)
    assert np.allclose(R.directivity(x, SR, instrument="trumpet", angle=0.0), x, atol=1e-9)
    t = tone(440.0)
    # sin(phi) = 0.5: fundamental 0.707, second harmonic 1
    assert abs(rms_db(R.directivity(t, SR, instrument="flute", angle=30.0)[1000:-1000]) - rms_db(t[1000:-1000]) + 3.01) < 0.1
    t2 = tone(880.0)
    assert abs(rms_db(R.directivity(t2, SR, instrument="flute", angle=30.0)[1000:-1000]) - rms_db(t2[1000:-1000])) < 0.1
    assert abs(20 * math.log10(S.membrane_directivity(1, math.radians(45.0), 0.0)) + 3.01) < 0.01
    assert abs(20 * math.log10(S.membrane_directivity(1, math.radians(71.5), 0.0)) + 10.0) < 0.05


def test_new_renders_have_no_clicks():
    from genny.analysis import detect_clicks
    x = tone(440.0) * np.hanning(SR)
    th = np.linspace(0, 2 * np.pi, 50)
    circle = np.column_stack([np.linspace(0, 1, 50), 8 * np.cos(th), 8 * np.sin(th), np.zeros(50)]).tolist()
    for y in (R.doppler(x, SR, path=circle), R.doppler(x, SR, speed=30.0, surface="grass"), R.doppler(x, SR, speed=30.0, reverb=0.5),
              R.doppler(x, SR, speed=30.0, horn="low", shadow=True), R.costello(x, SR, feedback=0.8), R.nrev(x, SR), R.prcrev(x, SR),
              R.jcrev(x, SR, variant="stk"), R.coupled_room(x, SR), R.decorrelate(x, SR, mode="velvet"), R.directivity(x, SR),
              R.room(x, SR, preset="concert_hall", modal=True, audience=True), R.distance(x, SR, wall_distance=0.5, source_radius=0.1, angle=45.0)):
        assert np.isfinite(y).all()
        assert np.abs(y).max() < 4.0 and detect_clicks(y).size == 0


_ISO_TABLE1 = [(-20, 20, -9, 1.49), (-20, 50, 0, 9.14), (-20, 10, 6, 3.86), (-20, 10, 8, 7.42), (-20, 80, 3, 19.0),
               (-20, 40, -4, 4.60), (0, 10, -13, 0.302), (0, 10, 0, 14.0), (0, 10, 6, 19.0), (0, 10, 10, 32.2),
               (5, 40, 1, 9.81), (5, 70, 1, 5.43), (10, 10, 0, 21.6), (10, 10, 6, 57.3), (10, 10, 10, 75.9)]


@pytest.mark.parametrize("temp, rh, k, table", _ISO_TABLE1)
def test_air_absorption_matches_iso_9613_1_table_1(temp, rh, k, table):
    f = 1000.0 * 10 ** (k / 10)
    assert S.air_db_per_m(np.array([f]), rh, temp)[0] * 1000 == pytest.approx(table, rel=0.006)


def test_gravel_and_old_snow_inside_measured_flow_resistivity():
    assert 1.5 <= R._GROUND_SIGMA["gravel"] <= 59.0
    assert R._GROUND_SIGMA["gravel"] == 53.0
    assert R._GROUND_SIGMA["snow"] == 10.0 and R._GROUND_SIGMA["old_snow"] == 30.0      # Albert 1990 Table 1: fresh 10, dense 25-35


def test_attenborough3_matches_hess_eq4_and_table3():
    f = np.array([100.0, 500.0, 2000.0])
    sig, T, W = S.HESS_SOILS["sand"]
    assert (sig, T, W) == (42.0, 2.15, 0.36) and S.HESS_SOILS["snow"] == (3.47, 2.70, 0.88)
    s = 1000.0 * sig
    kb = 0.0079 * np.sqrt(f) * np.sqrt(8.14 * T * f + 4j * s)
    k0 = 2 * np.pi * f / 343.0
    want = k0 * (4 / 3 * T + 2j * s / (np.pi * 1.2 * f)) / (W * kb)
    z = S.ground_impedance(f, sig, "attenborough3", tortuosity=T, porosity=W)
    assert np.allclose(z, want, rtol=0.03)
    assert np.allclose(S.ground_impedance(f, sig, "attenborough3", T, 0.5) * 0.5, S.ground_impedance(f, sig, "attenborough3", T, 1.0))


def test_forest_floor_presets_attenborough_1980():
    assert S.FOREST_FLOORS["red_pine"] == (49.63, 0.51, 0.020, 1350.0, 0.28)
    z = S.layered_impedance(np.array([500.0]), *S.FOREST_FLOORS["red_pine"])
    assert z.real[0] > 1.0 and z.imag[0] > 0.0
