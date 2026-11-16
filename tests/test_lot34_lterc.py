import math
import pytest
from genny import dsp, foley, creatures, reverbs
import inspect


def test_silenced_gains():
    assert foley.GUNS["silenced"][5] == pytest.approx(0.058)
    assert foley.GUNS["silenced_rifle"][5] == pytest.approx(0.067)


def test_oec_lb():
    assert creatures.OEC["lb"] == pytest.approx(1000.0)


def test_soundboard_t60_dc():
    assert inspect.signature(reverbs.soundboard_fdn).parameters["t60_dc"].default == 6.0


def test_plexiglass_wall():
    assert dsp.WALL_MATERIALS["plexiglass"]["mass"] == pytest.approx(11.5)
    assert dsp.WALL_MATERIALS["plexiglass"]["coincidence"] == pytest.approx(3080.0)
