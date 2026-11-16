"""Loudness and roughness models, lot P2: Moore-Glasberg loudness (the ISO 532-3 procedure) and the Sottek hearing model (the ECMA-418-2 procedure).

Both take a signal in pascals (samples times `pa`) and are calibrated on their reference tone: 1 kHz at 40 dB SPL is 1 sone for the
Moore-Glasberg model (binaural) and 1 sone_HMS for the Sottek model; 1 kHz at 60 dB SPL with 100 % modulation at 70 Hz is 1 asper_HMS.

Functions: `moore_glasberg_loudness`, `moore_glasberg_loudness_tv`, `hms_loudness`, `hms_roughness`, `procedural_grid` (coverage of the procedural-audio sound taxonomy).
"""
from __future__ import annotations

import numpy as np

from .analysis import ISO226_F, ISO226_TF

P_REF2 = 4e-10
TH_1K = 2.4
ERB_STEP = 0.1
_ERBN = np.arange(1.8, 39.0 + 1e-9, ERB_STEP)
_FC = (10.0 ** (_ERBN / 21.366) - 1.0) / 4.368 * 1000.0
_ERB = 24.673 * (4.368 * _FC / 1000.0 + 1.0)
_P51 = 4.0 * _FC / _ERB
_P1K = 4.0 * 1000.0 / (24.673 * 5.368)

MG_C = 0.0631
MG_ETHRQ_F = np.array([50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500.0])
MG_ETHRQ_DB = np.array([28.18, 23.90, 19.20, 15.68, 12.67, 10.09, 8.08, 6.30, 5.30, 4.50, 3.63])
MG_ETHRQ_HI = 3.63
MG_A_TAB = np.array([8.852, 7.165, 6.196, 5.679, 5.396, 5.187, 5.013, 4.913, 4.863, 4.813, 4.769, 4.723, 4.72096])
MG_GA = np.array([-24.54531, -17.11816, -11.75255, -8.35715, -6.14589, -4.39555, -2.78894, -1.84973,
                  -1.37466, -0.89518, -0.46783, -0.02470, 0.0])
MG_ALPHA_G = np.array([-25.0, -20.0, -15.0, -10.0, -5.0, 0.0])
MG_ALPHA_V = np.array([0.26692, 0.25016, 0.23679, 0.22228, 0.21055, 0.20])


def _ethrq_db(f):
    return np.where(f < 500.0, np.interp(f, MG_ETHRQ_F, MG_ETHRQ_DB), MG_ETHRQ_HI)


_E_DB = _ethrq_db(_FC)
MG_THR = 10.0 ** ((_E_DB - MG_ETHRQ_HI + TH_1K) / 10.0)
MG_GCH = 10.0 ** ((MG_ETHRQ_HI - _E_DB) / 10.0)
_GDB = 10.0 * np.log10(MG_GCH)
MG_ACH = np.interp(_GDB, MG_GA, MG_A_TAB)
MG_ALPHACH = np.interp(_GDB, MG_ALPHA_G, MG_ALPHA_V)


def _mono(y):
    y = np.asarray(y, float)
    return y.mean(axis=1) if y.ndim == 2 else y


def _transmission_db(f):
    """dB between the level in the free field and the level reaching the cochlea: flat at the threshold of hearing, so a tone at
    threshold excites every channel as much as the 1 kHz threshold tone (the outer and middle ear folded into one filter)."""
    return TH_1K - np.interp(np.log(np.maximum(f, 1.0)), np.log(ISO226_F), ISO226_TF)


def _mg_transmission_db(f):
    return _ethrq_db(f) - MG_ETHRQ_HI + TH_1K - np.interp(np.log(np.maximum(f, 1.0)), np.log(ISO226_F), ISO226_TF)


def _spectrum(x, sr):
    n = len(x)
    X = np.fft.rfft(x * np.hanning(n))
    w = np.hanning(n)
    p = np.abs(X) ** 2 / (np.sum(w ** 2) * n) * 2.0
    p[0] *= 0.5
    return np.fft.rfftfreq(n, 1.0 / sr), p


def _excitation(f, p):
    keep = (p > p.max() * 1e-11) & (f > 20.0) & (f < 16000.0)
    f, p = f[keep], p[keep] * 10.0 ** (_mg_transmission_db(f[keep]) / 10.0)
    g = (f[None, :] - _FC[:, None]) / _FC[:, None]
    near = np.abs(f[None, :] - _FC[:, None]) < 0.5 * _ERB[:, None]
    lev = 10.0 * np.log10((p[None, :] * near).sum(1) / P_REF2 + 1e-12)
    pl = np.maximum(_P51 - 0.35 * (_P51 / _P1K) * (lev - 51.0), 1.0)
    pp = np.where(g < 0, pl[:, None], _P51[:, None])
    ag = np.abs(g)
    w = (1.0 + pp * ag) * np.exp(-pp * ag)
    return (w * (p[None, :] / P_REF2)).sum(1)


def _specific(E):
    a, al, c, g = MG_ACH, MG_ALPHACH, MG_C, MG_GCH
    mid = c * ((g * E + a) ** al - a ** al)
    low = c * (2.0 * E / (E + MG_THR)) ** 1.5 * ((g * E + a) ** al - a ** al)
    high = c * (E / 1.04e6) ** 0.5
    return np.where(E < MG_THR, low, np.where(E > 1e10, high, mid))


def moore_glasberg_loudness(y, sr, pa=1.0):
    """Stationary loudness of a diotic signal (the same at both ears), in sone, and the specific loudness per ERB number (372 values,
    sone per ERB, step 0.1): (N, N'). Excitation from rounded-exponential filters, lower slope level dependent, Moore-Glasberg compression."""
    x = _mono(y) * float(pa)
    f, p = _spectrum(x, sr)
    n_ = _specific(_excitation(f, p))
    return float(2.0 * n_.sum() * ERB_STEP), n_


def moore_glasberg_loudness_tv(y, sr, pa=1.0, win=0.064, hop=0.001):
    """Short-term loudness series (sone every `hop` s): the stationary model on a Hann window of `win` s, then a one-pole smoothing with
    attack 45 ms and release 20 ms (the loudness meter's), as an approximation of the three-window procedure."""
    x = _mono(y) * float(pa)
    nw, nh = int(win * sr), max(1, int(hop * sr))
    if len(x) < nw:
        x = np.pad(x, (0, nw - len(x)))
    stp = max(nh, int(0.004 * sr))
    n = np.array([moore_glasberg_loudness(x[i:i + nw], sr)[0] for i in range(0, len(x) - nw + 1, stp)])
    out = np.empty_like(n)
    s = 0.0
    dt = stp / sr
    for i, v in enumerate(n):
        tau = 0.045 if v > s else 0.020
        s += (v - s) * (1.0 - np.exp(-dt / tau))
        out[i] = s
    return out, stp / sr


# ---------------------------------------------------------------------------------------- Sottek hearing model
HMS_Z = np.arange(0.5, 26.5 + 1e-9, 0.5)
HMS_FC = 1960.0 * (HMS_Z + 0.53) / (26.28 - HMS_Z)
HMS_BW = 25.0 + 75.0 * (1.0 + 1.4 * (HMS_FC / 1000.0) ** 2) ** 0.69
HMS_PT = 2e-5 * 10.0 ** (3.0 / 20.0)
HMS_FMOD, HMS_QMOD = 70.0, 0.9


def _bands(x, sr):
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    tr = 10.0 ** (_transmission_db(f) / 20.0)
    out = []
    for fc, bw in zip(HMS_FC, HMS_BW):
        if fc > 0.45 * sr:
            out.append(None)
            continue
        h = (1.0 + ((f - fc) / (0.5 * bw)) ** 2) ** -2
        out.append(np.fft.ifft(2.0 * X * h * tr, n))
    return out


def _hms_raw(x, sr):
    z = _bands(x, sr)
    sl = np.zeros(len(HMS_Z))
    for i, b in enumerate(z):
        if b is None:
            continue
        p2 = np.mean(np.abs(b) ** 2) / 2.0
        sl[i] = (p2 + HMS_PT ** 2) ** 0.3 - HMS_PT ** 0.6
    return sl.sum() * 0.5, sl


def _ref_tone(sr, level, dur=1.0, am=0.0):
    t = np.arange(int(dur * sr)) / sr
    x = np.sqrt(2.0) * 2e-5 * 10.0 ** (level / 20.0) * np.sin(2 * np.pi * 1000.0 * t)
    return x * (1.0 + am * np.sin(2 * np.pi * HMS_FMOD * t)) / (np.sqrt(1.0 + 0.5 * am ** 2) if am else 1.0)


_CAL = {}


def _cal_loud(sr):
    if ("n", sr) not in _CAL:
        _CAL[("n", sr)] = 1.0 / _hms_raw(_ref_tone(sr, 40.0), sr)[0]
    return _CAL[("n", sr)]


def hms_loudness(y, sr, pa=1.0):
    """Sottek-model loudness in sone_HMS of a signal in pascal: 53 half-Bark auditory bands (order-4 gammatone-like filters of the Bark
    bandwidth, outer and middle ear folded in as for the other model), band power compressed with exponent 0.3 above a hearing threshold
    floor, summed over the bands. Returns (N, specific loudness per half-Bark band). 1 kHz at 40 dB SPL = 1 sone_HMS."""
    n, sl = _hms_raw(_mono(y) * float(pa), sr)
    c = _cal_loud(sr)
    return float(n * c), sl * c


def _rough_raw(x, sr):
    z = _bands(x, sr)
    r = np.zeros(len(HMS_Z))
    for i, b in enumerate(z):
        if b is None:
            continue
        e = np.abs(b)
        dc = e.mean()
        if dc <= 1e-12:
            continue
        spec = np.abs(np.fft.rfft((e - dc) * np.hanning(len(e)))) / (np.sum(np.hanning(len(e))) / 2.0)
        fm = np.fft.rfftfreq(len(e), 1.0 / sr)
        wgt = np.exp(-np.log(np.maximum(fm, 1e-3) / HMS_FMOD) ** 2 / (2.0 * HMS_QMOD ** 2))
        m = np.sqrt(np.sum((spec * wgt) ** 2)) / dc
        lev = np.sqrt(np.mean(e ** 2) / 2.0)
        r[i] = min(m, 1.0) ** 2 * ((lev ** 2 + HMS_PT ** 2) ** 0.3 - HMS_PT ** 0.6)
    return r.sum() * 0.5, r


def hms_roughness(y, sr, pa=1.0):
    """Sottek-model roughness in asper_HMS: in each band the modulation spectrum of the envelope is weighted around 70 Hz (log-Gaussian,
    Q 0.9), the weighted modulation depth is scaled by the band's compressed loudness, and the bands are summed. 1 kHz at 60 dB SPL with
    100 % modulation at 70 Hz = 1 asper_HMS. Returns (R, per-band roughness)."""
    if ("r", sr) not in _CAL:
        _CAL[("r", sr)] = 1.0 / _rough_raw(_ref_tone(sr, 60.0, am=1.0), sr)[0]
    r, rb = _rough_raw(_mono(y) * float(pa), sr)
    c = _CAL[("r", sr)]
    return float(r * c), rb * c


PROCEDURAL_GRID = {
    "aerodynamic": {
        "aeolian_harp": ("aeolian_harp",), "aircraft": ("prop_engine", "flyby_jet"), "airflow": ("gust", "wind"),
        "cavity_tone": ("cavity_tone",), "edge_tone": ("whistle", "pipe_blow"), "lightsaber": ("energy_blade",),
        "propeller": ("propeller",), "rotating_air_fan": ("fan",), "swinging_objects": ("sword", "whoosh"),
    },
    "biotic": {
        "applause": ("applause",), "bird_call": ("bird_call", "birdsong"), "drinking_through_a_straw": ("eat",),
        "flocks_hordes_swarms": ("swarm", "crowd"), "footsteps": ("footsteps",), "heartbeat": ("heartbeat",),
        "insects": ("cricket", "cicada", "bee"), "laughter": ("laugh",), "mammalian_vocalisations": ("mammal",),
        "yelling": ("shout",),
    },
    "contact": {
        "bell": ("bell", "church_bell"), "bouncing": ("bounce", "ball_bounce"), "collision": ("impact", "smash"),
        "creaking_door": ("creak",), "crumpling": ("crumple",), "friction": ("scrape", "squeak"),
        "golf_putt": ("impact",), "jackhammer": ("jackhammer",), "near_rigid_thin_shell_vibrations": ("metal_plate",),
        "non_rigid_bodies": ("cloth",), "pouring_liquid": ("pour",), "rolling": ("roll",), "ruler_twang": ("twang",),
        "shuffling_cards": ("cards",), "soda_bottle_fizz": ("fizz",), "sword_clang": ("sword", "impact"),
    },
    "machine": {
        "alarm": ("alarm",), "electric_motor": ("electric_motor", "induction_motor"), "geiger_counter": ("geiger",),
        "helicopter": ("rotorcraft", "helicopter"), "industrial_machinery": ("industrial", "machine_hum"),
        "pedestrian_beeping_tone": ("crossing_beeper",), "phone_receiver_and_dtmf_tones": ("dtmf", "phone_tone"),
        "r2d2": ("robot_babble",), "siren": ("siren",), "switch_click": ("switch",), "telephone_ringing": ("phone_bell",),
        "ticking_clock": ("clock",), "traffic_noise": ("traffic",), "transporter": ("transporter",),
        "vehicle_engine": ("car_engine",), "weapons": ("firearm",),
    },
    "nature": {
        "avalanche": ("avalanche", "snow_avalanche"), "bubbles": ("bubbles",), "electricity": ("arc", "mains_hum"),
        "explosion": ("explosion",), "fire": ("fire",), "fluid_events": ("drip", "splash"), "rain": ("rain",),
        "room_tone": ("room_tone",), "storm": ("storm",), "thunder": ("thunder",), "turbulence": ("gust",),
        "volcanic_activity": ("volcano", "strombolian"), "wave": ("surf",), "wind": ("wind",),
    },
}

PROCEDURAL_TECHNIQUES = {
    "additive": ("church_organ", "organ", "bell"), "modal": ("impact", "modal_engine", "metal_plate"),
    "modulation": ("fm_bass", "stk_fm", "dx7"), "physical_modelling": ("fd_string", "reed_tube", "vowel_tube"),
    "physically_informed": ("roll", "scrape", "drip", "footsteps"), "source_filter": ("vowel_tube", "mammal", "speech"),
    "subtractive": ("wind", "rain", "fire", "ladder"), "waveguide": ("harp", "reed_tube", "tube"),
    "other": ("granular", "bytebeat", "gendyn"),
}


def procedural_grid(catalog=None):
    """Coverage of the five-group sound taxonomy of the procedural-audio review: for each sound class the genny names that cover it
    and those not found in the catalog. Returns {group: {class: {"genny": [...], "missing": [...]}}} plus a "summary" entry."""
    if catalog is None:
        from . import cli
        catalog = cli._catalog()
    known = set()
    for k in ("instruments", "drums", "sfx", "fx", "layers", "speech"):
        known |= set(catalog.get(k, {}))
    out, total, hit = {}, 0, 0
    for group, classes in PROCEDURAL_GRID.items():
        out[group] = {}
        for cls, names in classes.items():
            ok = [n for n in names if n in known]
            out[group][cls] = {"genny": ok, "missing": [n for n in names if n not in known]}
            total += 1
            hit += bool(ok)
    out["summary"] = {"classes": total, "covered": hit,
                      "techniques": {t: [n for n in v if n in known] for t, v in PROCEDURAL_TECHNIQUES.items()}}
    return out
