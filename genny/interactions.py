"""Unified bidirectional interaction facade for Genny physical models.

This module standardizes the *interfaces* of the existing specialized physical
solvers.  Each interaction exposes conjugate SI-domain ports and diagnostics,
while delegating the high-rate nonlinear waveguide solve to the mature kernels
in :mod:`genny.physical`.  This separation lets Genny evolve the numerical
solver without changing public physical model APIs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import numpy as np

from .core import DEFAULT_SR, resample
from .ports import PhysicalPort, PassiveCoupler
from .physical import waveguides as _wg
from .physical import voice as _voice
from . import graphsolver as _common

_PHYS_SR = 48000
_RHO_AIR = 1.2041
_C_AIR = 343.0

def _hybrid(common, specialized, mix=.22):
    a=np.asarray(common,float); b=np.asarray(specialized,float)
    n=min(len(a),len(b)); a=a[:n]; b=b[:n]
    y=(1.0-mix)*b + mix*a
    m=float(np.max(np.abs(y))) if len(y) else 0.0
    return y if m<=1.0 or m<=1e-12 else y/m


def _to_phys(x: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x if sr == _PHYS_SR else resample(x, sr, _PHYS_SR)


def _from_phys(x: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x if sr == _PHYS_SR else resample(x, _PHYS_SR, sr)


def _env(value: float | np.ndarray, dur: float, attack: float = .03) -> np.ndarray:
    n = max(1, int(round(dur * _PHYS_SR)))
    if np.ndim(value) == 0:
        out = np.full(n, float(value), dtype=float)
    else:
        a = np.asarray(value, dtype=float)
        if len(a) != n:
            # Parameter envelopes are control signals, so linear interpolation is
            # preferable to audio resampling here.
            xp = np.linspace(0.0, 1.0, len(a), endpoint=True)
            xq = np.linspace(0.0, 1.0, n, endpoint=True)
            out = np.interp(xq, xp, a)
        else:
            out = a.copy()
    na = min(n, max(1, int(round(attack * _PHYS_SR))))
    out[:na] *= np.linspace(0.0, 1.0, na)
    return out


@dataclass
class InteractionResult:
    """Audio plus physical interface traces for one solved interaction."""
    audio: np.ndarray
    drive_port: PhysicalPort
    response_port: PhysicalPort
    metadata: dict[str, Any] = field(default_factory=dict)

    def net_abs_energy(self, sr: int) -> float:
        return float((np.sum(np.abs(self.drive_port.power)) +
                      np.sum(np.abs(self.response_port.power))) / max(float(sr), 1.0))


@dataclass
class HammerStringInteraction:
    """Nonlinear felt hammer coupled to a stiff-string piano solver."""
    felt: float = .15
    brightness: float = .55
    inharmonicity: float = 4e-4
    decay: float = .55
    soundboard: float = .25
    hammer_mass_kg: float = .008
    coupling: float = .92

    def solve(self, frequency: float, dur: float, *, velocity: float=.8,
              sr: int=DEFAULT_SR, seed: int=0) -> InteractionResult:
        # Felt compression force F=K*x^p.  This is a reduced diagnostic contact
        # trace; the source kernel performs the nonlinear high-rate solve.
        n = max(1, int(round(dur * _PHYS_SR)))
        contact_s = min(.012, max(.0004, .006 / max(float(velocity), .08)))
        nc = max(2, min(n, int(round(contact_s * _PHYS_SR))))
        t = np.linspace(0.0, 1.0, nc)
        compression = np.sin(np.pi*t)
        key = float(np.clip(49.0 + 12.0*math.log2(max(frequency,1e-6)/440.0), 1.0, 88.0))
        p = 3.7 + .015*key
        q0 = 183.0*math.exp(.045*key)
        force = np.zeros(n)
        force[:nc] = q0 * np.power(np.maximum(compression,0.0), p) * float(velocity) * 1e-4
        hammer_v = np.zeros(n)
        hammer_v[:nc] = max(float(velocity),0.0) * np.cos(.5*np.pi*t)
        # Reduced string characteristic impedance. It only provides diagnostics
        # and port coupling; the waveguide kernel retains its calibrated model.
        z_string = max(.03, .08 * math.sqrt(max(frequency, 20.0)/440.0))
        string_v = force / z_string
        a = PhysicalPort.mechanical(force, hammer_v, "hammer")
        b = PhysicalPort.mechanical(np.zeros(n), string_v, "string")
        pa, pb = PassiveCoupler(self.coupling, polarity=-1.0).connect(a,b)
        common = _common.hammer_string(frequency,dur,velocity,sr=_PHYS_SR,felt=self.felt,coupling=self.coupling)
        audio = _wg.felt_piano(frequency, velocity, dur, np.random.default_rng(seed),
                               felt=self.felt, brightness=self.brightness,
                               inharmonicity=self.inharmonicity, decay=self.decay,
                               soundboard=self.soundboard)
        hybrid=_hybrid(common.audio,audio)
        return InteractionResult(_from_phys(hybrid,sr), common.drive, common.response,
                                 {"contact_s": contact_s, "felt_exponent": p,
                                  "felt_stiffness": q0, "solver": "hybrid-common/stiff-string-waveguide"})


@dataclass
class BowStringInteraction:
    """Stateful bow/string friction interface over the bowed waveguide kernel."""
    body: str = "violin"
    position: float = .127236
    pressure: float = .85
    velocity: float = .20
    vibrato: bool = True
    bow_noise: float = .06
    coupling: float = .88

    def solve(self, frequency: float, dur: float, *, sr: int=DEFAULT_SR,
              seed: int=0) -> InteractionResult:
        p = _env(self.pressure, dur, attack=.05)
        vbow = _env(self.velocity, dur, attack=.05)
        # Reduced stick/slip curve: high force near zero relative velocity and
        # lower kinetic force after release.  It is used for the shared physical
        # port diagnostics; the specialized kernel keeps its table-based solver.
        normal_n = 1.5 * np.clip(p,0,1.5)
        rel = vbow.copy()
        mu = .18 + .62*np.exp(-np.abs(rel)/.08)
        f = normal_n * mu
        string_z = max(.04, .11*math.sqrt(max(frequency,20.0)/440.0))
        vstring = np.tanh(f/string_z) * .04
        a = PhysicalPort.mechanical(f, vbow, "bow")
        b = PhysicalPort.mechanical(-f, vstring, "string")
        pa,pb = PassiveCoupler(self.coupling, polarity=-1.0).connect(a,b)
        audio = _wg.bowed_string(frequency, p, vbow, dur, np.random.default_rng(seed),
                                 body=self.body, position=self.position,
                                 vibrato=self.vibrato, bow_noise=self.bow_noise)
        common=_common.bow_string(frequency,dur,self.pressure,self.velocity,sr=_PHYS_SR,coupling=self.coupling)
        hybrid=_hybrid(common.audio,audio)
        return InteractionResult(_from_phys(hybrid,sr), common.drive,common.response,
                                 {"solver":"hybrid-common/bowed-string-waveguide",
                                  "friction":"stateful-table-kernel"})


@dataclass
class ReedBoreInteraction:
    """Pressure/volume-flow coupling for a single-reed bore."""
    breath: float = .6
    noise: float = .15
    bore_radius_m: float = .007
    coupling: float = .95

    @property
    def bore_impedance(self) -> float:
        area = math.pi*max(self.bore_radius_m,1e-5)**2
        return _RHO_AIR*_C_AIR/area

    def solve(self, frequency: float, dur: float, *, sr: int=DEFAULT_SR,
              seed: int=0) -> InteractionResult:
        ctrl = _env(self.breath,dur,.03)
        # Map normalized musical breath to a plausible mouth-pressure range.
        pm = 4500.0*np.clip(ctrl,0,1.5)
        z = self.bore_impedance
        # Saturating reed flow; ensures bounded flow at large pressure.
        u = (pm/z) * np.tanh(pm/3500.0)
        bore_p = z*u
        a = PhysicalPort.acoustic(pm, u, "reed")
        b = PhysicalPort.acoustic(bore_p, -u, "bore")
        pa,pb = PassiveCoupler(self.coupling, polarity=-1.0).connect(a,b)
        audio = _wg.clarinet(frequency, ctrl, dur, np.random.default_rng(seed),
                             noise_gain=self.noise)
        common=_common.reed_bore(frequency,dur,self.breath,self.noise*.15,sr=_PHYS_SR,bore_radius=self.bore_radius_m,seed=seed)
        hybrid=_hybrid(common.audio,audio)
        return InteractionResult(_from_phys(hybrid,sr),common.drive,common.response,
                                 {"bore_impedance":z,"solver":"hybrid-common/reed-bore-waveguide"})


@dataclass
class LipBoreInteraction:
    """Lip-valve pressure/flow coupling for brass instruments."""
    pressure: float=.55
    lip_tension: float=.5
    bore_radius_m: float=.006
    coupling: float=.94

    @property
    def bore_impedance(self) -> float:
        return _RHO_AIR*_C_AIR/(math.pi*max(self.bore_radius_m,1e-5)**2)

    def solve(self, frequency: float, dur: float, *, sr: int=DEFAULT_SR,
              seed: int=0) -> InteractionResult:
        ctrl = _env(self.pressure,dur,.04)
        pm = 7000.0*np.clip(ctrl,0,1.5)
        z = self.bore_impedance
        opening = np.clip(.2 + .75*self.lip_tension, .05, .95)
        dp = np.maximum(pm,0.0)
        # Bernoulli-inspired volume flow through a reduced lip opening.
        area = 2e-5*opening
        u = area*np.sqrt(2.0*dp/_RHO_AIR)
        bore_p = np.minimum(z*u, pm)
        a = PhysicalPort.acoustic(pm,u,"lips")
        b = PhysicalPort.acoustic(bore_p,-u,"brass_bore")
        pa,pb = PassiveCoupler(self.coupling,polarity=-1.0).connect(a,b)
        audio = _wg.brass(frequency,self.lip_tension,ctrl,dur,np.random.default_rng(seed))
        common=_common.lip_bore(frequency,dur,self.pressure,self.lip_tension,sr=_PHYS_SR,bore_radius=self.bore_radius_m,seed=seed)
        hybrid=_hybrid(common.audio,audio)
        return InteractionResult(_from_phys(hybrid,sr),common.drive,common.response,
                                 {"bore_impedance":z,"solver":"hybrid-common/lip-bore-waveguide"})


@dataclass
class SyrinxTracheaInteraction:
    """Dual-source syrinx coupled to a tracheal acoustic port."""
    pressure_pa: float=3000.0
    trachea_length_m: float=.07
    trachea_radius_m: float=.0035
    coupling: float=.93

    @property
    def trachea_impedance(self) -> float:
        area=math.pi*max(self.trachea_radius_m,1e-5)**2
        return _RHO_AIR*_C_AIR/area

    def solve(self, frequency: float|tuple[float,float], dur: float, *,
              sr: int=DEFAULT_SR) -> InteractionResult:
        pair = frequency if isinstance(frequency,tuple) else (frequency,frequency*1.01)
        n=max(1,int(round(dur*_PHYS_SR)))
        pm=np.full(n,float(self.pressure_pa))
        na=min(n,max(1,int(round(.02*_PHYS_SR))))
        pm[:na]*=np.linspace(0,1,na)
        z=self.trachea_impedance
        # Reduced pressure-driven flow with saturating membrane opening.
        u=(pm/z)*np.tanh(pm/max(self.pressure_pa,.1))
        ptract=z*u
        a=PhysicalPort.acoustic(pm,u,"syrinx")
        b=PhysicalPort.acoustic(ptract,-u,"trachea")
        pa,pb=PassiveCoupler(self.coupling,polarity=-1.0).connect(a,b)
        audio=_voice.syrinx(self.pressure_pa,pair,dur,
                            trachea_length_m=self.trachea_length_m,
                            trachea_radius_m=self.trachea_radius_m)
        common=_common.syrinx_trachea(pair,dur,self.pressure_pa,sr=_PHYS_SR,trachea_length=self.trachea_length_m,trachea_radius=self.trachea_radius_m)
        hybrid=_hybrid(common.audio,audio)
        return InteractionResult(_from_phys(hybrid,sr),common.drive,common.response,
                                 {"trachea_impedance":z,"solver":"hybrid-common/syrinx-trachea"})
