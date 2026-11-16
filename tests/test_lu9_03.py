import math

import numpy as np

from genny import compose as C
from genny.physical import story as ST
from genny.piano import PianoHammerAction


def test_hammer_hysteresis_default_unchanged_and_loses_energy():
    a = PianoHammerAction(60)
    a.strike(0.53)
    b = PianoHammerAction(60, hysteresis=1.5)
    b.strike(0.53)
    assert a.hysteresis == 0.0
    assert b.last_contact_time_s < a.last_contact_time_s
    assert abs(b.last_rebound_velocity_m_s) < abs(a.last_rebound_velocity_m_s)


def test_nasal_volume_rescale_to_measured_mean():
    sec = ST.NASAL_LEN_CM
    a = ST.nasal_area(1.04, volume_cm3=ST.NASAL_VOLUME_MRI_CM3)
    assert abs(a.sum() * sec / len(a) - 25.5) < 0.05
    assert ST.nasal_area(1.04).sum() * sec / len(ST.nasal_area(1.04)) > 28.0


def test_mri_sinuses_helmholtz_range():
    for s in ST.SINUSES_MRI[:2]:
        ln = s["neck_len"] + 1.7 * math.sqrt(s["neck_area"] / math.pi)
        f0 = ST.C_CMS / (2 * math.pi) * math.sqrt(s["neck_area"] / (s["volume"] * ln))
        assert 400 < f0 < 450
    assert len(ST.nasal_branch(1.04, "mri")["branches"]) == 6


def test_emotion_palettes_bresin_friberg_2000():
    lv = {k: v["level_db"] for k, v in C.EMOTIONS.items()}
    assert lv == {"angry": 8.0, "happy": 3.0, "sad": -6.0, "tender": -6.0, "fear": -6.0, "solemn": 3.0}
    ioi = {k: 1.0 / v["tempo"] for k, v in C.EMOTIONS.items()}
    assert np.allclose([ioi[k] for k in ("angry", "happy", "sad", "tender", "solemn", "fear")],
                       [0.85, 0.80, 1.3, 1.3, 1.3, 1.8])
