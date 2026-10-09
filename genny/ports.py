"""Typed physical ports and lightweight passive coupling graph.

The graph is intentionally small: it does not try to replace a full mechanics
solver.  It gives Genny models a common language for exchanging conjugate
physical variables (force/velocity and pressure/volume-flow) while keeping
energy accounting explicit.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Literal
import numpy as np

Domain = Literal["mechanical", "acoustic"]

@dataclass
class PhysicalPort:
    """A conjugate effort/flow signal pair.

    Mechanical ports use effort=N and flow=m/s. Acoustic ports use effort=Pa
    and flow=m^3/s. Instantaneous power is always effort*flow (watts).
    """
    domain: Domain
    effort: np.ndarray
    flow: np.ndarray
    effort_unit: str
    flow_unit: str
    name: str = ""

    def __post_init__(self):
        self.effort = np.asarray(self.effort, dtype=float)
        self.flow = np.asarray(self.flow, dtype=float)
        if self.effort.shape != self.flow.shape:
            raise ValueError("effort and flow must have the same shape")

    @property
    def power(self) -> np.ndarray:
        return self.effort * self.flow

    def energy(self, sr: int) -> float:
        return float(np.sum(self.power) / float(sr))

    @classmethod
    def mechanical(cls, force, velocity, name=""):
        return cls("mechanical", force, velocity, "N", "m/s", name)

    @classmethod
    def acoustic(cls, pressure, volume_flow, name=""):
        return cls("acoustic", pressure, volume_flow, "Pa", "m^3/s", name)


@dataclass
class PassiveCoupler:
    """Bidirectional two-port coupler with explicit power transmission.

    ``transmission`` is an amplitude coupling coefficient in [0,1].  The
    coupler reflects the unused part instead of creating energy.
    """
    transmission: float = 1.0
    polarity: float = 1.0

    def connect(self, a: PhysicalPort, b: PhysicalPort) -> tuple[PhysicalPort, PhysicalPort]:
        if a.domain != b.domain:
            raise ValueError("ports must share a physical domain")
        t = float(np.clip(self.transmission, 0.0, 1.0))
        r = float(np.sqrt(max(0.0, 1.0 - t*t)))
        # Scattering-style exchange. Using an orthogonal matrix makes the
        # instantaneous quadratic signal norm non-increasing for |t|<=1.
        ae = r*a.effort + self.polarity*t*b.effort
        be = self.polarity*t*a.effort - r*b.effort
        af = r*a.flow + self.polarity*t*b.flow
        bf = self.polarity*t*a.flow - r*b.flow
        ctor = PhysicalPort.mechanical if a.domain == "mechanical" else PhysicalPort.acoustic
        return ctor(ae, af, a.name), ctor(be, bf, b.name)


@dataclass
class PhysicalGraph:
    """Small explicit graph for physical-port processing and diagnostics."""
    nodes: dict[str, PhysicalPort] = field(default_factory=dict)
    links: list[tuple[str, str, PassiveCoupler]] = field(default_factory=list)

    def add(self, name: str, port: PhysicalPort) -> PhysicalPort:
        self.nodes[name] = port
        return port

    def couple(self, a: str, b: str, transmission: float = 1.0, polarity: float = 1.0):
        if a not in self.nodes or b not in self.nodes:
            raise KeyError("both nodes must exist before coupling")
        self.links.append((a, b, PassiveCoupler(transmission, polarity)))

    def process(self) -> dict[str, PhysicalPort]:
        out = dict(self.nodes)
        for a, b, coupler in self.links:
            pa, pb = coupler.connect(out[a], out[b])
            out[a], out[b] = pa, pb
        return out

    def total_abs_energy(self, sr: int) -> float:
        return float(sum(np.sum(np.abs(p.power))/sr for p in self.nodes.values()))


@dataclass
class MechanicalImpedance:
    """Reduced mass/spring/damper mechanical impedance element."""
    mass: float = 1.0
    stiffness: float = 0.0
    damping: float = 0.0

    def force_from_velocity(self, velocity: np.ndarray, sr: int) -> np.ndarray:
        v = np.asarray(velocity, float)
        dt = 1.0/float(sr)
        accel = np.gradient(v, dt)
        disp = np.cumsum(v)*dt
        return self.mass*accel + self.damping*v + self.stiffness*disp


@dataclass
class AcousticImpedance:
    """Frequency-independent reduced acoustic impedance p = Z U."""
    impedance: float = 1.0

    def pressure_from_flow(self, flow: np.ndarray) -> np.ndarray:
        return float(self.impedance)*np.asarray(flow,float)
