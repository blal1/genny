"""High-level physically informed synthesis API.

This module is the stable Genny-facing layer above :mod:`genny.physical`, whose
48 kHz research kernels are adapted from the user-supplied Klang/procedural code.
It keeps the existing Genny API intact while exposing composable physical models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping
import math
import numpy as np
from .strings import StringPhysicalProperties, WoundStringPhysicalProperties, DispersiveString

from .core import DEFAULT_SR, resample, normalize
from .physical import modal as _modal
from .physical import particles as _particles
from .physical import waveguides as _wg
from .physical import voice as _voice
from .physical import ambience as _amb
from .interactions import (HammerStringInteraction, BowStringInteraction, ReedBoreInteraction,
                           LipBoreInteraction, SyrinxTracheaInteraction)

_PHYS_SR = 48000


def _to_sr(x: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x if sr == _PHYS_SR else resample(x, _PHYS_SR, sr)


@dataclass(frozen=True)
class Material:
    """Physical/material parameters used by high-level models.

    ``density`` is kg/m^3, ``young_modulus`` Pa, ``poisson`` dimensionless,
    ``damping_alpha`` 1/s and ``damping_beta`` s.  The damping pair follows
    Rayleigh damping: C = alpha M + beta K.
    """
    name: str = "generic"
    density: float = 1000.0
    young_modulus: float = 1.0e9
    poisson: float = 0.3
    damping_alpha: float = 1.0
    damping_beta: float = 1.0e-8
    hardness: float = 0.5
    fracture_toughness: float = 100.0


MATERIALS: dict[str, Material] = {
    "glass": Material("glass", 2500, 70e9, 0.22, 0.4, 2e-9, 0.95, 50.0),
    "steel": Material("steel", 7850, 200e9, 0.30, 0.25, 2e-9, 0.98, 120.0),
    "aluminum": Material("aluminum", 2700, 69e9, 0.33, 0.6, 4e-9, 0.8, 90.0),
    "wood": Material("wood", 650, 11e9, 0.35, 2.5, 2e-8, 0.55, 180.0),
    "stone": Material("stone", 2600, 50e9, 0.25, 1.2, 8e-9, 0.9, 80.0),
    "rubber": Material("rubber", 1100, 0.01e9, 0.49, 20.0, 2e-6, 0.12, 500.0),
}


@dataclass
class ModalBody:
    """Reusable modal body with strike-position and material controls."""
    table: str | dict
    material: Material = field(default_factory=Material)
    frequency_scale: float = 1.0
    decay_scale: float = 1.0
    gain_scale: float = 1.0

    def modes(self, *, position: float = 0.0, f0: float | None = None) -> _modal.Modes:
        return _modal.resolve(
            self.table,
            f_scale=self.frequency_scale,
            position=position,
            d_scale=self.decay_scale,
            a_scale=self.gain_scale,
            f0=f0,
        )

    def strike(self, *, duration: float = 2.0, contact_ms: float = 1.0,
               velocity: float = 1.0, position: float = 0.0, sr: int = DEFAULT_SR,
               micro_collisions: int | None = None, seed: int = 0) -> np.ndarray:
        m = self.modes(position=position)
        n = max(1, int(round(duration * _PHYS_SR)))
        force = np.zeros(n)
        p = _modal.bang(max(contact_ms / max(velocity, 0.05), 0.05), velocity)
        force[:min(len(p), n)] = p[:n]
        if micro_collisions is None:
            micro_collisions = int(round(4 * np.clip(self.material.hardness, 0, 1)))
        if micro_collisions > 0:
            _modal.micro_collisions(force, m, micro_collisions, contact_ms, velocity,
                                    np.random.default_rng(seed))
        y = _modal.render_modes(force, m)
        return _to_sr(y, sr)

    def fracture(self, *, duration: float = 4.0, toughness: float | None = None,
                 sr: int = DEFAULT_SR, seed: int = 0, **kwargs) -> np.ndarray:
        y = _modal.fracture(
            self.table, np.random.default_rng(seed), f_scale=self.frequency_scale,
            toughness=float(toughness if toughness is not None else self.material.fracture_toughness),
            **kwargs,
        )
        target = int(round(duration * _PHYS_SR))
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        else:
            y = y[:target]
        return _to_sr(y, sr)


@dataclass
class ParticleMaterial:
    preset: str = "Maraca"
    resonance_scale: float = 1.0
    damping: float = 1.0
    objects: float | None = None

    def shake(self, dur: float, *, energy: float = 0.8, sr: int = DEFAULT_SR,
              seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        curve = np.ones(int(round(dur * _PHYS_SR)), dtype=float) * float(energy)
        y = _particles.shaker(self.preset, dur, rng, energy_curve=curve, n_objects=self.objects,
                              resonance_scale=self.resonance_scale, damping=self.damping,
                              energy_mode="level")
        return _to_sr(y, sr)


@dataclass
class BubblePopulation:
    rate: float = 50.0
    radius_min: float = 0.0004
    radius_max: float = 0.006
    power: float = 2.0

    def render(self, dur: float, *, sr: int = DEFAULT_SR, seed: int = 0) -> np.ndarray:
        # Source engine exposes the calibrated Moss/Minnaert population parameters.
        y = _particles.bubble_field(dur, np.random.default_rng(seed), rate=self.rate,
                                    r_range=(self.radius_min, self.radius_max), alpha=self.power)
        return _to_sr(y, sr)


@dataclass
class StringInstrument:
    body: str = "guitar"
    brightness: float = 0.5
    position: float = 0.2
    sustain: float | None = None

    def pluck(self, frequency: float, dur: float, *, velocity: float = 0.8,
              sr: int = DEFAULT_SR, seed: int = 0) -> np.ndarray:
        y = _wg.pluck(frequency, dur, np.random.default_rng(seed), position=self.position,
                      brightness=self.brightness, body=self.body, velocity=velocity,
                      sustain=self.sustain)
        return _to_sr(y, sr)


@dataclass
class Piano:
    brightness: float = 0.55
    felt: float = 0.15
    inharmonicity: float = 4e-4
    decay: float = 0.55
    soundboard: float = 0.25

    def interaction(self) -> HammerStringInteraction:
        return HammerStringInteraction(self.felt, self.brightness, self.inharmonicity,
                                       self.decay, self.soundboard)

    def note(self, frequency: float, dur: float, *, velocity: float = 0.8,
             sr: int = DEFAULT_SR, seed: int = 0) -> np.ndarray:
        return self.interaction().solve(frequency, dur, velocity=velocity, sr=sr, seed=seed).audio

    def sympathetic_note(self, frequency: float, dur: float, *, velocity: float = 0.8,
                         sr: int = DEFAULT_SR, coupling: float = 0.055,
                         unison_detune_cents=(-1.8, 0.0, 1.6),
                         sympathetic_ratios=(0.5, 1.5, 2.0, 2.5, 3.0),
                         return_stems: bool = False):
        """Render a shared-bridge/shared-soundboard piano network.

        Unlike :meth:`note`, this reduced model explicitly transfers energy to
        unstruck strings through a common mechanical bridge and modal
        soundboard.  It is intended for sympathetic-resonance work and remains
        separate from the legacy-calibrated public note renderer.
        """
        from .networks import SympatheticPiano
        model=SympatheticPiano(float(frequency), tuple(unison_detune_cents),
                               tuple(sympathetic_ratios), coupling=float(coupling), sr=int(sr))
        return model.render(float(dur), velocity=float(velocity), return_stems=return_stems)

    def polyphonic(self, *, sr: int = DEFAULT_SR, coupling: float = 0.055,
                   sympathetic_coupling: float = 0.0028):
        """Create a stateful shared-bridge 88-key piano session."""
        from .piano import PolyphonicPiano
        return PolyphonicPiano(sr=int(sr), global_coupling=float(coupling),
                               sympathetic_coupling=float(sympathetic_coupling))


@dataclass
class Clarinet:
    breath: float = 0.6
    noise: float = 0.15

    def interaction(self) -> ReedBoreInteraction:
        return ReedBoreInteraction(self.breath, self.noise)

    def note(self, frequency: float, dur: float, *, sr: int = DEFAULT_SR,
             seed: int = 0) -> np.ndarray:
        return self.interaction().solve(frequency, dur, sr=sr, seed=seed).audio


@dataclass
class Brass:
    pressure: float = 0.55
    lip_tension: float = 0.5

    def interaction(self) -> LipBoreInteraction:
        return LipBoreInteraction(self.pressure, self.lip_tension)

    def note(self, frequency: float, dur: float, *, sr: int = DEFAULT_SR,
             seed: int = 0) -> np.ndarray:
        return self.interaction().solve(frequency, dur, sr=sr, seed=seed).audio


@dataclass
class VocalTract:
    tenseness: float = 0.6
    tongue_index: float = 12.9
    tongue_diameter: float = 2.43
    tract_scale: float = 1.0

    def phonate(self, frequency: float | np.ndarray, dur: float, *, sr: int = DEFAULT_SR,
                seed: int = 0, constrictions=()) -> np.ndarray:
        y = _voice.pink_trombone(frequency, self.tenseness, self.tongue_index,
                                 self.tongue_diameter, constrictions,
                                 tick_rate_scale=1.0 / max(self.tract_scale, 1e-4), dur=dur,
                                 rng=np.random.default_rng(seed))
        return _to_sr(y, sr)


@dataclass
class BirdSyrinx:
    pressure_pa: float = 3000.0
    trachea_length_m: float = 0.07
    trachea_radius_m: float = 0.0035

    def interaction(self) -> SyrinxTracheaInteraction:
        return SyrinxTracheaInteraction(self.pressure_pa, self.trachea_length_m,
                                        self.trachea_radius_m)

    def call(self, frequency: float | tuple[float, float], dur: float, *, sr: int = DEFAULT_SR) -> np.ndarray:
        return self.interaction().solve(frequency, dur, sr=sr).audio


@dataclass
class Environment:
    """Procedural environment generators sharing causal/statistical physical kernels."""
    seed: int = 0

    def rain(self, dur: float, intensity: float = 1.0, *, sr: int = DEFAULT_SR) -> np.ndarray:
        return _to_sr(_amb.rain(dur, np.random.default_rng(self.seed), intensity=intensity), sr)

    def fire(self, dur: float, *, sr: int = DEFAULT_SR) -> np.ndarray:
        return _to_sr(_amb.fire(dur, np.random.default_rng(self.seed)), sr)

    def stream(self, dur: float, *, sr: int = DEFAULT_SR) -> np.ndarray:
        return _to_sr(_amb.stream(dur, np.random.default_rng(self.seed)), sr)

    def thunder(self, dur: float = 12.0, distance: float = 0.0, *, sr: int = DEFAULT_SR) -> np.ndarray:
        # The research patch assumes enough tail for a one-second terminal fade.
        # Render a physically meaningful minimum window, then crop/pad to the
        # requested duration so short UI/game calls remain safe.
        rd = max(float(dur), 1.25)
        y = np.asarray(_amb.thunder(np.random.default_rng(self.seed), dur=rd, distance=distance), float)
        n48 = max(1, int(round(float(dur) * _PHYS_SR)))
        y = y[:n48]
        if len(y) < n48:
            pad = ((0, n48-len(y)), (0, 0)) if y.ndim == 2 else (0, n48-len(y))
            y = np.pad(y, pad)
        return _to_sr(y, sr)

    def surf(self, dur: float, distance: float = 120.0, *, sr: int = DEFAULT_SR) -> np.ndarray:
        return _to_sr(_amb.surf(dur, np.random.default_rng(self.seed), distance=distance), sr)

@dataclass
class BarBody:
    """Analytical Euler-Bernoulli bar/rod model rendered through Genny's modal bank.

    Dimensions are SI units.  ``boundary`` may be ``"clamped"`` (cantilever)
    or ``"free"``.  This reduced model is intended for rods, bars, chimes,
    xylophone-like elements and structural impacts; it is not a full FEM solver.
    """
    material: Material = field(default_factory=lambda: MATERIALS["steel"])
    length_m: float = 0.4
    radius_m: float | None = 0.006
    width_m: float | None = None
    boundary: str = "free"
    n_modes: int = 8

    def _radius_of_gyration(self) -> float:
        if self.radius_m is not None:
            return max(float(self.radius_m), 1e-6) / 2.0
        if self.width_m is not None:
            return max(float(self.width_m), 1e-6) / math.sqrt(12.0)
        raise ValueError("BarBody requires radius_m or width_m")

    def modes(self) -> _modal.Modes:
        L = max(float(self.length_m), 1e-5)
        Rg = self._radius_of_gyration()
        root = math.sqrt(max(self.material.young_modulus, 1.0) * Rg * Rg /
                         max(self.material.density, 1e-9))
        if self.boundary == "clamped":
            f1 = 0.5596 / (L * L) * root
            ratios = np.array([1.0, 6.267, 17.55, 34.39, 56.84, 84.35, 116.9, 154.5])
        elif self.boundary in ("free", "free_free"):
            f1 = 3.5594 / (L * L) * root
            ratios = np.array([1.0, 2.756, 5.404, 8.933, 13.34, 18.64, 24.84, 31.94])
        else:
            raise ValueError("boundary must be 'clamped' or 'free'")
        ratios = ratios[:max(1, min(int(self.n_modes), len(ratios)))]
        freqs = f1 * ratios
        w = 2.0 * np.pi * freqs
        decays = 0.5 * (self.material.damping_alpha + self.material.damping_beta * w * w)   # Rayleigh: d = (alpha + beta w^2) / 2
        # Higher bending modes usually couple less strongly to a point excitation.
        gains = 1.0 / np.sqrt(np.arange(1, len(freqs) + 1, dtype=float))
        keep = freqs < 0.45 * _PHYS_SR
        return _modal.Modes(freqs[keep], decays[keep], gains[keep], "vdd")

    def strike(self, *, duration: float = 2.0, velocity: float = 1.0,
               contact_ms: float = 0.4, sr: int = DEFAULT_SR) -> np.ndarray:
        modes = self.modes()
        n = max(1, int(round(duration * _PHYS_SR)))
        force = np.zeros(n)
        p = _modal.bang(_modal.contact_time(contact_ms, velocity), velocity)
        force[:min(n, len(p))] = p[:n]
        return _to_sr(_modal.render_modes(force, modes), sr)


@dataclass
class BowedString:
    """Bowed-string facade using the shared SI-property string core."""
    body: str = "violin"
    bow_position: float = 0.127236
    bow_pressure: float = 0.85
    bow_velocity: float = 0.20
    vibrato: bool = True
    bow_noise: float = 0.06

    def interaction(self) -> BowStringInteraction:
        return BowStringInteraction(self.body, self.bow_position, self.bow_pressure,
                                    self.bow_velocity, self.vibrato, self.bow_noise)

    def note(self, frequency: float, dur: float, *, sr: int = DEFAULT_SR,
             seed: int = 0) -> np.ndarray:
        f=max(20.0,float(frequency)); body=self.body.lower()
        if body == "cello":
            props=WoundStringPhysicalProperties.from_frequency(f,length_m=max(.62,min(.72,.69*220/f)),
                                                               core_radius_m=.00038,winding_wire_radius_m=.00018,
                                                               winding_material='nickel')
        else:
            props=StringPhysicalProperties.from_frequency(f,length_m=max(.28,min(.34,.328*440/f)),
                                                          radius_m=.00032,material='steel',t60_s=3.4)
        core=DispersiveString(props,sr=sr,bridge_coupling=.028,polarization_coupling=.016)
        y=core.render_bow(dur,bow_velocity=self.bow_velocity,bow_pressure=self.bow_pressure,
                          bow_position=self.bow_position)
        if self.bow_noise > 0 and len(y):
            rng=np.random.default_rng(seed); noise=rng.standard_normal(len(y))
            noise=np.concatenate([[0.],np.diff(noise)])
            y=y + float(self.bow_noise)*.025*noise*np.minimum(1.,np.arange(len(y))/max(1,.04*sr))
        peak=float(np.max(np.abs(y))) if len(y) else 0.
        if peak>1: y/=peak
        return y


@dataclass
class Flute:
    """Jet-driven flute/recorder waveguide."""
    model: str = "flute"
    pressure: float = 0.75
    noise: float = 0.15
    jet_ratio: float = 0.32
    vibrato: float = 0.05

    def note(self, frequency: float, dur: float, *, sr: int = DEFAULT_SR,
             seed: int = 0) -> np.ndarray:
        n = max(1, int(round(dur * _PHYS_SR)))
        env = np.ones(n) * np.clip(self.pressure, 0.0, 1.5)
        a = max(1, int(min(n, round(0.05 * _PHYS_SR))))
        env[:a] *= np.linspace(0.0, 1.0, a)
        specialized = _wg.recorder(frequency, env, dur, np.random.default_rng(seed),
                                   model=self.model, noise_gain=self.noise,
                                   jet_ratio=self.jet_ratio, vibrato_gain=self.vibrato)
        # v0.11: the shared graph contributes a true hydrodynamic jet delay and
        # passive bore/bell feedback. The specialized renderer remains the
        # dominant timbral path while migration continues.
        from .graphsolver import jet_bore
        jet_velocity = 8.0 + 18.0*np.clip(self.pressure,0.0,1.5)
        common = jet_bore(frequency, dur, jet_velocity=jet_velocity, sr=_PHYS_SR,
                          jet_length=max(.003, .012*self.jet_ratio))
        m=min(len(specialized),len(common.audio))
        y=.78*np.asarray(specialized[:m],float)+.22*np.asarray(common.audio[:m],float)
        peak=float(np.max(np.abs(y))) if len(y) else 0.0
        if peak>1.0: y/=peak
        return _to_sr(y, sr)


@dataclass
class Footstep:
    """Ground-reaction-force driven procedural footstep."""
    ground: str = "wood"
    variation: int = 0

    @staticmethod
    def grounds() -> tuple[str, ...]:
        from .physical import footsteps as _steps
        return tuple(_steps.GROUND_SPECS.keys())

    def render(self, *, sr: int = DEFAULT_SR, seed: int = 0,
               return_info: bool = False):
        from .physical import footsteps as _steps
        out = _steps.render_step(self.ground, self.variation,
                                 np.random.default_rng(seed), return_info=return_info)
        if return_info:
            y, info = out
            return _to_sr(y, sr), info
        return _to_sr(out, sr)


@dataclass
class AeroacousticSwing:
    """Distributed aeolian/vortex sound for a swept sword/rod-like object."""
    preset: str = "SWD1"
    velocity_scale: float = 1.0
    diameter_scale: float = 1.0
    natural_gain: bool = False

    def render(self, *, sr: int = DEFAULT_SR, seed: int = 0, **kwargs) -> np.ndarray:
        # The source research kernel already computes Reynolds/Strouhal-dependent
        # vortex centres, bandwidth and dipole/wake gain along the swept object.
        if isinstance(self.preset, str):
            preset = dict(_modal.md.SWORD_PRESETS["presets"][self.preset])
        else:
            preset = dict(self.preset)
        preset["TopSpeed"] *= max(float(self.velocity_scale), 1e-6)
        preset["HiltThick"] *= max(float(self.diameter_scale), 1e-6)
        preset["TipThick"] *= max(float(self.diameter_scale), 1e-6)
        # ``natural_gain`` selects the full published Reynolds/Strouhal model;
        # the perceptually calibrated LoQ path is useful for lightweight game audio.
        kwargs.setdefault("model", "patch" if self.natural_gain else "loq")
        y = _modal.sword_swing(preset, np.random.default_rng(seed), **kwargs)
        return _to_sr(y, sr)


# Extend the environment facade with additional causal generators without
# changing the methods that were present in v0.3.
def _environment_wind(self, dur: float, *, sr: int = DEFAULT_SR, scale: float = 1.0,
                      loop: bool = False, return_speed: bool = False,
                      sources=("static",)):
    # The original research patch has long causal wind-source delays (up to
    # hundreds of ms).  Render a minimum working window and crop so short unit
    # tests / one-shot requests do not create mismatched delayed buffers.
    render_dur = max(float(dur), 0.75)
    out = _amb.wind_scene(render_dur, np.random.default_rng(self.seed), loop=loop,
                          scale=scale, return_speed=return_speed, sources=sources)
    n48 = max(1, int(round(float(dur) * _PHYS_SR)))
    if return_speed:
        y, w = out
        return _to_sr(y[:n48], sr), _to_sr(w[:n48], sr)
    return _to_sr(out[:n48], sr)


def _environment_creak(self, force: np.ndarray, *, material: str = "wood",
                       sr: int = DEFAULT_SR, loop: bool = False) -> np.ndarray:
    f = np.asarray(force, float)
    if sr != _PHYS_SR:
        f = resample(f, sr, _PHYS_SR)
    y = _amb.creak(f, np.random.default_rng(self.seed), loop=loop, material=material)
    return _to_sr(y, sr)


def _environment_cave_drips(self, dur: float, *, sr: int = DEFAULT_SR,
                            sources: int = 5, loop: bool = False) -> np.ndarray:
    # Stochastic cave-drip schedules can legitimately contain no event in a very
    # short requested window. Render a longer causal scene, then select a window
    # around the first audible event so one-shot API calls still return a drip.
    rd = max(float(dur), 2.0)
    y = np.asarray(_amb.cave_drips(rd, np.random.default_rng(self.seed),
                                   loop=loop, sources=sources), float)
    n = max(1, int(round(float(dur) * _PHYS_SR)))
    amp = np.max(np.abs(y), axis=1) if y.ndim == 2 else np.abs(y)
    hit = np.flatnonzero(amp > max(1e-10, float(amp.max()) * 1e-4))
    if hit.size:
        start = max(0, int(hit[0]) - int(0.01 * _PHYS_SR))
        seg = y[start:start+n]
    else:
        # Deterministic one-shot fallback when the sparse scene produced no
        # event in the rendered interval. Keep it stereo to match cave_drips.
        drip = _particles.drip(np.random.default_rng(self.seed), dur=max(float(dur), 0.08))
        seg = np.column_stack([drip, drip * 0.93])[:n]
    if len(seg) < n:
        pad = ((0, n-len(seg)), (0, 0)) if y.ndim == 2 else (0, n-len(seg))
        seg = np.pad(seg, pad)
    return _to_sr(seg, sr)


Environment.wind = _environment_wind
Environment.creak = _environment_creak
Environment.cave_drips = _environment_cave_drips

@dataclass
class TunedPercussion:
    """Physically informed struck bar/bell family backed by modal research tables."""
    kind: str = "marimba"
    decay: float = 2.0
    hardness: float = 0.55

    def note(self, frequency: float, dur: float | None = None, *, velocity: float = 0.8,
             sr: int = DEFAULT_SR, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        kind = self.kind.lower()
        duration = float(dur if dur is not None else self.decay)
        if kind in ("marimba", "marimbabar"):
            # STK modal-bar tables include strike position and stick hardness.
            y = _modal.stk_modal_bar("Marimba", frequency, velocity,
                                 hardness=float(np.clip(self.hardness, 0, 1)), dur=duration)
        elif kind in ("vibraphone", "vibes"):
            y = _modal.stk_modal_bar("Vibraphone", frequency, velocity,
                                 hardness=float(np.clip(self.hardness, 0, 1)), dur=duration)
        elif kind in ("wood", "woodblock"):
            y = _modal.stk_modal_bar("Wood1", frequency, velocity,
                                 hardness=float(np.clip(self.hardness, 0, 1)), dur=duration)
        elif kind in ("gong", "tam-tam", "tamtam"):
            # Farnell chinatown2 gong table is a dedicated noizy model rather than
            # a generic modal table; render its inharmonic groups directly.
            tab = _modal.md.GONG
            n = max(1, int(round(duration * _PHYS_SR)))
            tt = np.arange(n) / _PHYS_SR
            y = np.zeros(n)
            ratios = np.asarray(tab["ratios"], float)
            attacks = np.asarray(tab["attack_ms"], float) * 1e-3
            decays = np.asarray(tab["decay_ms"], float) * 1e-3
            for i, (r, at, de) in enumerate(zip(ratios, attacks, decays)):
                env = np.minimum(1.0, tt/max(at, 1e-4)) * np.exp(-tt/max(de, 1e-4))
                # Slight slow FSK creates the characteristic gong shimmer.
                inst_f = frequency*r + tab["fsk_depth_hz"]*(0.04/(i+1))*np.sin(2*np.pi*0.37*tt + i)
                ph = 2*np.pi*np.cumsum(inst_f)/_PHYS_SR
                y += env*np.sin(ph + 0.7*i)/(1+0.25*i)
            y *= velocity
        elif kind in ("bell", "church_bell", "churchbell"):
            y = _modal.additive_bell("risset_bell", frequency, duration, rng) * velocity
        elif kind in ("china_bell", "chinabell"):
            y = _modal.additive_bell("chinabell", frequency, duration, rng) * velocity
        else:
            raise ValueError(f"unknown tuned percussion kind: {self.kind!r}")
        return _to_sr(y, sr)


@dataclass
class PluckedString:
    """Family facade for physical/commuted plucked strings."""
    kind: str = "guitar"
    brightness: float = 0.55
    position: float = 0.22
    detune: float = 0.995

    def note(self, frequency: float, dur: float, *, velocity: float = 0.8,
             sr: int = DEFAULT_SR, seed: int = 0) -> np.ndarray:
        """Render through the shared physical string core (v0.19)."""
        k=self.kind.lower(); f=max(20.,float(frequency))
        if k in ('bass','upright_bass','finger_bass') or f < 90:
            props=WoundStringPhysicalProperties.from_frequency(f,winding_material='nickel')
            return DispersiveString(props,sr=sr).render_pluck(dur,velocity=velocity,position=self.position)
        mat='nylon' if k in ('harp','koto','classical_guitar') else 'steel'
        props=StringPhysicalProperties.from_frequency(f,material=mat)
        if k == 'mandolin':
            a=DispersiveString(props,sr=sr).render_pluck(dur,velocity=velocity,position=self.position)
            props2=StringPhysicalProperties.from_frequency(f*float(self.detune),length_m=props.length_m,
                                                           radius_m=props.radius_m,material=mat)
            b=DispersiveString(props2,sr=sr).render_pluck(dur,velocity=velocity*.96,position=min(.95,self.position+.012))
            y=.54*a+.46*b
        else:
            y=DispersiveString(props,sr=sr).render_pluck(dur,velocity=velocity,position=self.position)
        return y

    def physical_note(self, frequency: float, dur: float, *, velocity: float = 0.8,
                      sr: int = DEFAULT_SR, material: str | None = None,
                      length_m: float | None = None, radius_m: float | None = None) -> np.ndarray:
        """Render through the shared v0.18 physical string core.

        This is additive to ``note()`` for backward compatibility.  Instrument
        kind selects a material default while explicit geometry is supported.
        """
        k=self.kind.lower()
        mat = material or ('nylon' if k in ('harp','koto') else 'steel')
        props=StringPhysicalProperties.from_frequency(frequency,length_m=length_m,
                                                      radius_m=radius_m,material=mat)
        core=DispersiveString(props,sr=sr)
        return core.render_pluck(dur,velocity=velocity,position=self.position)


    def wound_note(self, frequency: float, dur: float, *, velocity: float = 0.8, sr: int = DEFAULT_SR,
                   core_material: str = 'steel', winding_material: str = 'nickel',
                   length_m: float | None = None, core_radius_m: float | None = None,
                   winding_wire_radius_m: float | None = None) -> np.ndarray:
        props=WoundStringPhysicalProperties.from_frequency(frequency,length_m=length_m,
                    core_radius_m=core_radius_m,winding_wire_radius_m=winding_wire_radius_m,
                    core_material=core_material,winding_material=winding_material)
        return DispersiveString(props,sr=sr).render_pluck(dur,velocity=velocity,position=self.position)


@dataclass
class CombustionEngine:
    """Four-stroke engine facade with physically meaningful firing-rate control."""
    cylinders: int = 4
    rpm: float = 1500.0
    load: float = 0.5
    roughness: float = 0.25
    exhaust_delay_s: float = 0.004
    body: float = 1.0
    brightness: float = 0.0

    @property
    def firing_rate_hz(self) -> float:
        return max(self.rpm, 1.0) * max(int(self.cylinders), 1) / 120.0

    def render(self, dur: float, *, sr: int = DEFAULT_SR, seed: int = 7) -> np.ndarray:
        # Reuse Genny's combustion implementation; its pulse timing follows
        # f_fire = cylinders * RPM / 120 for a four-stroke engine.
        from .vehicle import car_engine
        return car_engine(sr=sr, rpm=self.rpm, cylinders=self.cylinders, load=self.load,
                          rough=self.roughness, pipe=self.exhaust_delay_s, body=self.body,
                          bright=self.brightness, seed=seed, dur=dur)


def _environment_pour(self, dur: float, *, sr: int = DEFAULT_SR, flow: float = 1.0,
                      fill: float = 0.0) -> np.ndarray:
    """Reduced pouring model: stream texture + bubble population + rising cavity resonance."""
    rng = np.random.default_rng(self.seed)
    rd = max(float(dur), 0.08)
    stream = np.asarray(_amb.farnell_water_voice(rd, rng, centre=650 + 450*flow), float)
    bubbles = np.asarray(_particles.bubble_field(rd, rng, rate=30 + 100*max(flow, 0.0),
                                                  r_range=(0.00035, 0.0045)), float)
    n = min(len(stream), len(bubbles))
    y = stream[:n] * 0.65 + bubbles[:n] * 0.35
    # As a container fills, its effective air column shortens and its resonance rises.
    t = np.arange(n) / _PHYS_SR
    f0 = 220.0 + 720.0 * np.clip(fill + (1.0-fill) * t/max(rd, 1e-6), 0, 1)
    # lightweight resonant coloration using a tracked sinus excited by the stream envelope
    env = np.minimum(1.0, np.abs(stream[:n]) * 6.0)
    phase = 2*np.pi*np.cumsum(f0)/_PHYS_SR
    y += 0.08 * env * np.sin(phase)
    return _to_sr(y, sr)


Environment.pour = _environment_pour

# ============================================================================
# v0.7: shared membrane / plate / contact / rotor physical models
# ============================================================================

@dataclass
class MembraneDrum:
    """Reduced circular-membrane model with strike-position and stroke controls.

    The mode ratios follow the low modes of a circular membrane (Bessel zeros),
    while ``harmonic_loading`` continuously pulls them toward an integer series
    (useful for tabla-like loaded heads).  This is a reduced modal renderer, not
    a full 2-D waveguide mesh, but preserves the excitation/resonator separation.
    """
    frequency: float = 180.0
    radius_m: float = 0.16
    damping: float = 1.0
    harmonic_loading: float = 0.0
    air_coupling: float = 0.25
    shell: str = "wood"

    _RATIOS = np.array([1.0, 1.5933, 2.1355, 2.2954, 2.6531, 2.9173, 3.1555, 3.5001, 3.6005])

    def _modes(self, position: float, stroke: str) -> _modal.Modes:
        h = float(np.clip(self.harmonic_loading, 0.0, 1.0))
        harmonic = np.arange(1, len(self._RATIOS) + 1, dtype=float)
        ratios = (1.0 - h) * self._RATIOS + h * harmonic
        f = max(15.0, float(self.frequency)) * ratios
        # Edge strikes suppress axisymmetric low modes and excite higher orders.
        p = float(np.clip(position, 0.0, 1.0))
        idx = np.arange(len(f), dtype=float)
        centre = np.exp(-0.22 * idx)
        edge = (0.28 + 0.72 * (idx / max(1.0, idx[-1]))) * np.exp(-0.08 * idx)
        gains = (1.0 - p) * centre + p * edge
        if stroke == "slap":
            gains *= 0.55 + 0.6 * idx / max(1.0, idx[-1])
        elif stroke == "bass":
            gains *= np.exp(-0.38 * idx)
        # Frequency-dependent radiation/damping; small heads decay faster.
        size = max(self.radius_m, 0.04) / 0.16
        base = max(self.damping, 0.05) * (2.0 / max(size, 0.25))
        decay = base * (1.0 + 0.13 * idx + 0.018 * idx * idx)
        keep = f < 0.45 * _PHYS_SR
        return _modal.Modes(f[keep], decay[keep], gains[keep], "membrane")

    def strike(self, dur: float = 0.8, *, velocity: float = 0.8,
               position: float = 0.25, stroke: str = "tone", sr: int = DEFAULT_SR,
               seed: int = 0) -> np.ndarray:
        dur = max(float(dur), 0.04)
        n = max(1, int(round(dur * _PHYS_SR)))
        modes = self._modes(position, stroke)
        force = np.zeros(n)
        hardness = {"bass": 3.5, "tone": 1.2, "slap": 0.22, "rim": 0.12}.get(stroke, 1.2)
        pulse = _modal.bang(_modal.contact_time(hardness, max(velocity, 0.05)), velocity)
        force[:min(n, len(pulse))] += pulse[:n]
        # Slaps contain high-frequency contact noise; it remains an excitation,
        # not a separate unrelated sound layer.
        if stroke in ("slap", "rim"):
            rng = np.random.default_rng(seed)
            m = min(n, int(0.012 * _PHYS_SR))
            noise = rng.standard_normal(m) * np.exp(-np.arange(m) / (0.0022 * _PHYS_SR))
            force[:m] += noise * (0.10 + 0.22 * velocity)
        y = _modal.render_modes(force, modes)
        # Reduced air/shell coupling: low cavity resonance plus a weak shell mode.
        t = np.arange(n) / _PHYS_SR
        cavity_f = max(45.0, self.frequency * (0.42 if stroke == "bass" else 0.62))
        cavity = np.sin(2*np.pi*cavity_f*t) * np.exp(-t / (0.08 + 0.22*self.air_coupling))
        shell_ratio = 2.8 if self.shell == "metal" else 2.15
        shell = np.sin(2*np.pi*self.frequency*shell_ratio*t) * np.exp(-t / 0.055)
        y = y + self.air_coupling * 0.22 * velocity * cavity + 0.045 * velocity * shell
        peak = np.max(np.abs(y)) + 1e-12
        y = np.tanh(y / peak * (1.1 + 0.8 * velocity)) * (0.55 + 0.35 * velocity)
        return _to_sr(y, sr)


@dataclass
class CymbalPlate:
    """Reduced nonlinear thin-plate/cymbal model.

    A dense inharmonic modal bank provides the linear body while a short
    velocity-dependent noisy excitation and weak amplitude-dependent modal FM
    approximate the well-known nonlinear spectral spreading of cymbals.
    """
    fundamental: float = 210.0
    size: float = 1.0
    damping: float = 1.0
    profile: str = "crash"   # crash | ride | hihat

    def strike(self, dur: float = 2.0, *, velocity: float = 0.8,
               bell: float = 0.0, sr: int = DEFAULT_SR, seed: int = 0) -> np.ndarray:
        dur = max(0.04, float(dur)); n = int(round(dur * _PHYS_SR))
        rng = np.random.default_rng(seed)
        # Irregular plate ratios (dense at high order), deliberately non-harmonic.
        ratios = np.array([1.00,1.43,1.91,2.37,2.86,3.31,3.88,4.47,5.12,5.89,
                           6.73,7.66,8.72,9.91,11.24,12.77,14.52,16.48,18.71,21.2])
        f = self.fundamental * ratios / max(self.size, 0.2)
        keep = f < 0.44 * _PHYS_SR; f=f[keep]; ratios=ratios[keep]
        k = np.arange(len(f), dtype=float)
        if self.profile == "ride":
            gains = 0.55/(1+0.11*k) + bell*np.exp(-0.24*k)
            decay = self.damping*(1.2 + 0.13*k)
            contact_ms = 0.28
        elif self.profile == "hihat":
            gains = 0.32 + 0.68*(k/max(1,len(k)-1))
            decay = self.damping*(7.0 + 0.65*k)
            contact_ms = 0.10
        else:
            gains = 0.85/(1+0.08*k)
            decay = self.damping*(0.75 + 0.095*k)
            contact_ms = 0.16
        modes = _modal.Modes(f, decay, gains, "plate")
        force=np.zeros(n)
        p=_modal.bang(_modal.contact_time(contact_ms, max(velocity,.05)), velocity)
        force[:min(n,len(p))]=p[:n]
        # Contact/turbulent component excites the dense upper modes.
        m=min(n,int(0.035*_PHYS_SR))
        force[:m] += rng.standard_normal(m)*np.exp(-np.arange(m)/(0.006*_PHYS_SR))*0.16*velocity
        y=_modal.render_modes(force,modes)
        # Nonlinear shimmer: weak AM/FM-like sideband cloud whose amount grows
        # supra-linearly with strike velocity, inspired by plate nonlinearity.
        t=np.arange(n)/_PHYS_SR
        shimmer=np.zeros(n)
        nl=max(0.0,float(velocity))**2
        for j, fj in enumerate(f[:min(8,len(f))]):
            fm=2.2+0.47*j
            phase=2*np.pi*fj*t + (0.0025*nl*fj)*np.sin(2*np.pi*fm*t)/(fm+1e-9)
            shimmer += np.sin(phase)*np.exp(-t/(0.18+0.05*j))*gains[j]
        y += shimmer * (0.028 + 0.06*nl)
        peak=np.max(np.abs(y))+1e-12
        return _to_sr(np.tanh(y/peak*1.35)*(0.45+0.45*velocity), sr)


@dataclass
class SurfaceContact:
    """Spatial-profile based scrape and rolling-contact generator."""
    roughness: float = 0.5
    characteristic_mm: float = 1.0
    body_frequency: float = 700.0
    body_decay: float = 18.0

    def _profile(self, n: int, speed: float, rng: np.random.Generator) -> np.ndarray:
        # Fractal-ish spatial roughness: integrate white noise then high-pass by
        # first difference. Speed maps spatial wavelength to temporal frequency.
        x=rng.standard_normal(n+2)
        smooth=np.cumsum(x)
        smooth=(smooth-np.mean(smooth))/(np.std(smooth)+1e-12)
        mix=(1-self.roughness)*x[:n] + self.roughness*np.diff(smooth[:n+1])
        # Add a physical surface periodicity: f = v/lambda.
        lam=max(self.characteristic_mm*1e-3,1e-5)
        f=min(0.42*_PHYS_SR, max(2.0, abs(speed)/lam))
        t=np.arange(n)/_PHYS_SR
        return 0.7*mix + 0.3*np.sin(2*np.pi*f*t)

    def scrape(self, dur: float, *, speed: float = 0.3, force: float = 1.0,
               sr: int = DEFAULT_SR, seed: int = 0) -> np.ndarray:
        n=max(1,int(round(dur*_PHYS_SR))); rng=np.random.default_rng(seed)
        exc=self._profile(n,speed,rng)*max(force,0.0)
        modes=_modal.Modes(np.array([self.body_frequency,self.body_frequency*1.71,self.body_frequency*2.63]),
                           np.array([self.body_decay,self.body_decay*1.4,self.body_decay*1.9]),
                           np.array([1.0,.42,.22]),"surface")
        y=_modal.render_modes(exc,modes)
        p=np.max(np.abs(y))+1e-12
        return _to_sr(np.tanh(y/p*1.15)*0.65, sr)

    def roll(self, dur: float, *, speed: float = 0.5, radius_m: float = 0.03,
             force: float = 1.0, sr: int = DEFAULT_SR, seed: int = 0) -> np.ndarray:
        n=max(1,int(round(dur*_PHYS_SR))); rng=np.random.default_rng(seed)
        rough=self._profile(n,speed,rng)
        # Rolling contact is dominated by longer spatial wavelengths.
        win=max(3,int(0.0025*_PHYS_SR)); ker=np.ones(win)/win
        exc=np.convolve(rough,ker,mode='same')
        fr=abs(speed)/(2*np.pi*max(radius_m,1e-4))
        t=np.arange(n)/_PHYS_SR
        exc += 0.35*np.sin(2*np.pi*fr*t)
        modes=_modal.Modes(np.array([self.body_frequency,self.body_frequency*1.62,self.body_frequency*2.41]),
                           np.array([self.body_decay*.75,self.body_decay,self.body_decay*1.4]),
                           np.array([1,.45,.24]),"rolling")
        y=_modal.render_modes(exc*force,modes)
        p=np.max(np.abs(y))+1e-12
        return _to_sr(np.tanh(y/p)*0.62, sr)


@dataclass
class Rotor:
    """Generic rotor/propeller/fan aeroacoustic source."""
    rpm: float = 1800.0
    blades: int = 5
    radius_m: float = 0.15
    turbulence: float = 0.45
    tonal: float = 0.55

    @property
    def rotation_hz(self) -> float:
        return max(0.0, float(self.rpm))/60.0

    @property
    def blade_passing_hz(self) -> float:
        return self.rotation_hz*max(1,int(self.blades))

    @property
    def tip_speed(self) -> float:
        return 2*np.pi*self.rotation_hz*max(self.radius_m,1e-4)

    def render(self, dur: float, *, sr: int = DEFAULT_SR, seed: int = 0,
               listener_angle: float = 0.0) -> np.ndarray:
        n=max(1,int(round(dur*_PHYS_SR))); t=np.arange(n)/_PHYS_SR
        rng=np.random.default_rng(seed)
        bpf=max(1.0,self.blade_passing_hz)
        # Pressure pulse train with harmonics; angle controls tonal/direct component.
        direction=0.25+0.75*abs(np.cos(float(listener_angle)))
        tonal=np.zeros(n)
        for h,a in ((1,1.0),(2,.34),(3,.17),(4,.08)):
            if h*bpf < .45*_PHYS_SR:
                tonal += a*np.sin(2*np.pi*h*bpf*t)
        # Broadband turbulence scales strongly with tip speed but is normalized
        # to remain safe for public rendering.
        noise=rng.standard_normal(n)
        # one-pole smoothing roughly shifts spectrum with tip speed
        fc=np.clip(300+18*self.tip_speed,350,9000)
        alpha=np.exp(-2*np.pi*fc/_PHYS_SR); lp=np.empty(n); s=0.0
        for i in range(n):
            s=(1-alpha)*noise[i]+alpha*s; lp[i]=s
        hp=noise-lp
        turb=(.65*lp+.35*hp)*(0.15+0.85*self.turbulence)
        y=self.tonal*direction*tonal + turb
        # small rotation-rate AM gives realistic cyclic loading.
        y*=0.78+0.22*np.sin(2*np.pi*self.rotation_hz*t+0.4)
        p=np.max(np.abs(y))+1e-12
        return _to_sr(np.tanh(y/p*1.35)*0.58, sr)


@dataclass
class JetEngine:
    """Reduced multi-spool turbofan/jet model: blade tones + combustion + jet turbulence."""
    rpm: float = 9000.0
    fan_blades: int = 24
    compressor_blades: int = 36
    throttle: float = 0.7

    def render(self, dur: float, *, sr: int = DEFAULT_SR, seed: int = 0) -> np.ndarray:
        n=max(1,int(round(dur*_PHYS_SR))); rng=np.random.default_rng(seed); t=np.arange(n)/_PHYS_SR
        th=float(np.clip(self.throttle,0,1))
        fan=Rotor(self.rpm*(.65+.35*th),self.fan_blades,.65,.55,.55).render(dur,sr=_PHYS_SR,seed=seed)
        comp=Rotor(self.rpm*(1.6+.5*th),self.compressor_blades,.22,.35,.42).render(dur,sr=_PHYS_SR,seed=seed+1)
        noise=rng.standard_normal(n)
        # Jet mixing broadband: emphasize low-mid roar, with HF increasing at throttle.
        fc=650+2400*th; alpha=np.exp(-2*np.pi*fc/_PHYS_SR); lp=np.empty(n); s=0.
        for i in range(n): s=(1-alpha)*noise[i]+alpha*s; lp[i]=s
        roar=lp + (noise-lp)*(0.08+.3*th)
        # Combustion rumble is correlated, not an independent white layer.
        rumble=np.sin(2*np.pi*(70+45*th)*t + .8*np.sin(2*np.pi*2.1*t))
        y=.38*fan+.22*comp+(.34+.28*th)*roar+.11*rumble
        p=np.max(np.abs(y))+1e-12
        return _to_sr(np.tanh(y/p*(1.2+.3*th))*0.65,sr)


@dataclass
class Helicopter:
    """Reduced causal helicopter model: main rotor, tail rotor, engine and blade-vortex slap."""
    main_rpm: float = 320.0
    main_blades: int = 4
    tail_rpm: float = 1600.0
    tail_blades: int = 4
    load: float = 0.55

    def render(self, dur: float, *, sr: int = DEFAULT_SR, seed: int = 0,
               listener_angle: float = 0.0) -> np.ndarray:
        n=max(1,int(round(dur*_PHYS_SR))); t=np.arange(n)/_PHYS_SR
        main=Rotor(self.main_rpm,self.main_blades,6.5,.30,.82).render(dur,sr=_PHYS_SR,seed=seed,listener_angle=listener_angle)
        tail=Rotor(self.tail_rpm,self.tail_blades,.65,.52,.42).render(dur,sr=_PHYS_SR,seed=seed+1,listener_angle=listener_angle+1.1)
        bpf=max(1.0,self.main_rpm/60*self.main_blades)
        # Blade-vortex interaction: short periodic N-ish impulses whose amount
        # rises nonlinearly with rotor loading.
        phase=(t*bpf)%1.0
        slap=(np.exp(-phase/.045)-.65*np.exp(-phase/.12))
        slap-=np.mean(slap)
        slap*=float(np.clip(self.load,0,1))**2
        engine=JetEngine(rpm=4500+2500*self.load,fan_blades=12,compressor_blades=18,
                         throttle=.35+.55*self.load).render(dur,sr=_PHYS_SR,seed=seed+2)
        y=.62*main+.20*tail+.22*engine+.33*slap
        p=np.max(np.abs(y))+1e-12
        return _to_sr(np.tanh(y/p*1.4)*.68,sr)


@dataclass
class Destruction:
    """Composite impact/fracture renderer for breakable objects."""
    material: str = "glass"
    table: str = "sy_vase"

    def render(self, dur: float = 2.0, *, energy: float = 1.0,
               sr: int = DEFAULT_SR, seed: int = 0) -> np.ndarray:
        mat=MATERIALS.get(self.material,MATERIALS["glass"])
        body=ModalBody(self.table, material=mat)
        try:
            y=body.fracture(duration=max(dur,1.0), toughness=mat.fracture_toughness,
                            sr=sr, seed=seed)
        except Exception:
            y=body.strike(duration=max(dur,1.0), velocity=max(.1,energy), sr=sr,
                          micro_collisions=max(2,int(5*mat.hardness)), seed=seed)
        n=max(1,int(round(dur*sr)))
        y=np.asarray(y,float)[:n]
        if len(y)<n: y=np.pad(y,(0,n-len(y)))
        return np.tanh(y*(.8+.5*energy))

# ============================================================================
# v0.8: bidirectional/contact-heavy models and machine/explosion families
# ============================================================================

@dataclass
class KickDrum:
    """Physical kick: membrane + enclosed-air mode + beater/contact transient."""
    frequency: float = 55.0
    radius_m: float = 0.28
    decay: float = 0.45
    beater_hardness: float = 0.55
    air_coupling: float = 0.8

    def strike(self, dur: float | None = None, *, velocity: float = 0.9,
               sr: int = DEFAULT_SR, seed: int = 0) -> np.ndarray:
        dur = float(dur if dur is not None else max(.25, self.decay*1.5))
        n = max(1, int(round(dur*_PHYS_SR)))
        rng = np.random.default_rng(seed)
        # Circular membrane fundamentals and low shell/cavity coupling.
        drum = MembraneDrum(self.frequency, self.radius_m,
                            damping=max(.3, .55/max(self.decay,.05)),
                            air_coupling=self.air_coupling)
        y = np.asarray(drum.strike(dur, velocity=velocity, position=.08,
                                   stroke="bass", sr=_PHYS_SR, seed=seed), float)
        t = np.arange(n)/_PHYS_SR
        # Beater/skin compression produces a rapid downward chirp, not a
        # separate synthetic oscillator in the physical interpretation.
        f_hi = self.frequency*(2.2 + 2.6*np.clip(self.beater_hardness,0,1))
        f_lo = self.frequency*.92
        tau = .018 + .035*(1-np.clip(self.beater_hardness,0,1))
        inst = f_lo + (f_hi-f_lo)*np.exp(-t/max(tau,1e-5))
        phase = 2*np.pi*np.cumsum(inst)/_PHYS_SR
        beater = np.sin(phase)*np.exp(-t/max(.07,self.decay*.55))*velocity
        # Very short broadband contact for hard beaters.
        m = min(n, int(.009*_PHYS_SR))
        click = np.zeros(n)
        if m:
            click[:m] = rng.standard_normal(m)*np.exp(-np.arange(m)/(.0018*_PHYS_SR))
        y = .78*y[:n] + .44*beater + .045*self.beater_hardness*click
        p=np.max(np.abs(y))+1e-12
        return _to_sr(np.tanh(y/p*1.35)*(.55+.35*velocity), sr)


@dataclass
class SnareDrum:
    """Membrane + cavity + mechanically driven snare-wire bank."""
    frequency: float = 180.0
    radius_m: float = 0.17
    decay: float = 0.28
    snare_tension: float = 0.7
    wire_count: int = 20

    def strike(self, dur: float | None = None, *, velocity: float = 0.85,
               snappy: float = 0.7, sr: int = DEFAULT_SR, seed: int = 0) -> np.ndarray:
        dur=float(dur if dur is not None else max(.14,self.decay*1.5))
        n=max(1,int(round(dur*_PHYS_SR))); rng=np.random.default_rng(seed)
        head = MembraneDrum(self.frequency,self.radius_m,
                            damping=max(.55,.8/max(self.decay,.05)), air_coupling=.35)
        membrane=np.asarray(head.strike(dur,velocity=velocity,position=.35,stroke="slap",
                                        sr=_PHYS_SR,seed=seed),float)[:n]
        # Bottom-head acceleration drives the wires.  More wires raise event
        # density, while tension moves their resonant band upward.
        acc=np.gradient(np.gradient(membrane))*_PHYS_SR*_PHYS_SR
        drive=np.abs(acc); drive/=np.max(drive)+1e-12
        events = rng.random(n) < np.clip((6e-4 + 2.2e-3*drive)
                                         * max(self.wire_count,1)/20.0, 0, .25)
        impulses = rng.standard_normal(n)*events*drive
        # Wire resonances clustered in the upper mids; a few modes are enough
        # because the collision train already supplies the dense spectrum.
        base=1300 + 1500*np.clip(self.snare_tension,0,1)
        freqs=np.array([base*.72,base,base*1.31,base*1.73,base*2.12])
        keep=freqs < .44*_PHYS_SR; freqs=freqs[keep]
        dec=np.linspace(28,64,len(freqs)); gains=np.linspace(1,.28,len(freqs))
        wires=_modal.render_modes(impulses,_modal.Modes(freqs,dec,gains,"snare_wires"))
        y=(1-.58*np.clip(snappy,0,1))*membrane + (.35+.72*np.clip(snappy,0,1))*wires
        p=np.max(np.abs(y))+1e-12
        return _to_sr(np.tanh(y/p*1.45)*(.48+.42*velocity),sr)


@dataclass
class AdvancedRollingContact:
    """Rolling contact with weak resonator→contact feedback.

    The resonator displacement changes effective normal force, so texture and
    resonance are coupled instead of rendered as two independent layers.
    """
    roughness: float = 0.5
    characteristic_mm: float = 1.0
    body_frequency: float = 700.0
    body_decay: float = 16.0
    feedback: float = 0.18

    def render(self, dur: float, *, speed: float = .5, radius_m: float = .03,
               normal_force: float = 1.0, sr: int = DEFAULT_SR, seed: int = 0) -> np.ndarray:
        n=max(1,int(round(dur*_PHYS_SR))); rng=np.random.default_rng(seed)
        surf=SurfaceContact(self.roughness,self.characteristic_mm,self.body_frequency,self.body_decay)
        base=surf._profile(n,speed,rng)
        win=max(3,int(.0025*_PHYS_SR)); base=np.convolve(base,np.ones(win)/win,mode="same")
        fr=abs(speed)/(2*np.pi*max(radius_m,1e-4)); t=np.arange(n)/_PHYS_SR
        base += .28*np.sin(2*np.pi*fr*t)
        # One resonant state closes the loop. Keep feedback <1 for passivity-ish
        # behaviour; excitation is modulated by displacement-derived load.
        w=2*np.pi*max(20.,self.body_frequency)/_PHYS_SR
        r=np.exp(-max(.1,self.body_decay)/_PHYS_SR)
        a1=2*r*np.cos(w); a2=-r*r
        y=np.zeros(n); y1=y2=0.; fb=float(np.clip(self.feedback,0,.72))
        for i in range(n):
            load=max(0.0,normal_force*(1.0 + fb*np.tanh(2.5*y1)))
            exc=base[i]*load
            yy=exc + a1*y1 + a2*y2
            y[i]=yy; y2,y1=y1,yy
        p=np.max(np.abs(y))+1e-12
        return _to_sr(np.tanh(y/p*1.2)*.62,sr)


@dataclass
class ElectricMotor:
    """Reduced brushed/BLDC motor with rotor, commutation, bearings and housing."""
    rpm: float = 6000.0
    poles: int = 6
    commutations_per_rev: int = 12
    imbalance_mass_kg: float = 0.001
    eccentricity_m: float = 0.0005
    bearing_roughness: float = 0.25
    housing_frequency: float = 900.0

    @property
    def rotation_hz(self) -> float:
        return max(0.,self.rpm)/60.0

    @property
    def commutation_hz(self) -> float:
        return self.rotation_hz*max(1,int(self.commutations_per_rev))

    @property
    def imbalance_force_n(self) -> float:
        omega=2*np.pi*self.rotation_hz
        return self.imbalance_mass_kg*self.eccentricity_m*omega*omega

    def render(self, dur: float, *, load: float = .5, startup: float = 0.0,
               sr: int = DEFAULT_SR, seed: int = 0) -> np.ndarray:
        n=max(1,int(round(dur*_PHYS_SR))); t=np.arange(n)/_PHYS_SR
        rng=np.random.default_rng(seed); target=max(1.,self.rotation_hz)
        if startup>0:
            fr=target*(1-np.exp(-t/max(startup,1e-4)))
        else: fr=np.full(n,target)
        phase=2*np.pi*np.cumsum(fr)/_PHYS_SR
        rotor=np.sin(phase)+.22*np.sin(2*phase+.4)+.08*np.sin(3*phase+.8)
        # Commutator pulse train with small timing jitter approximated by phase AM.
        cfr=fr*max(1,int(self.commutations_per_rev))
        cphase=2*np.pi*np.cumsum(cfr*(1+.002*rng.standard_normal(n)))/_PHYS_SR
        comm=np.tanh(5*np.sin(cphase))*0.22
        # Bearing broadband rises with speed and load.
        noise=rng.standard_normal(n); alpha=np.exp(-2*np.pi*np.clip(500+target*8,600,9000)/_PHYS_SR)
        lp=np.empty(n); s=0.
        for i in range(n): s=(1-alpha)*noise[i]+alpha*s; lp[i]=s
        bearing=(noise-lp*.65)*self.bearing_roughness*(.25+.75*load)
        exc=.18*rotor + .14*comm + .07*bearing
        # Structural housing resonances, including imbalance at rotation rate.
        modes=_modal.Modes(np.array([self.housing_frequency,self.housing_frequency*1.64,self.housing_frequency*2.28]),
                           np.array([14.,22.,31.]),np.array([1,.42,.22]),"motor_housing")
        housing=_modal.render_modes(exc,modes)
        imbalance=np.sin(phase)*min(2.0,self.imbalance_force_n)*.05
        y=housing + imbalance + .04*bearing
        p=np.max(np.abs(y))+1e-12
        return _to_sr(np.tanh(y/p*1.25)*.58,sr)


@dataclass
class GearTrain:
    """Gear-mesh tonal/noise model driven by shaft RPM and tooth counts."""
    rpm: float = 1800.0
    driver_teeth: int = 20
    driven_teeth: int = 40
    backlash: float = 0.15
    housing_frequency: float = 1200.0

    @property
    def mesh_hz(self) -> float:
        return max(0.,self.rpm)/60.0*max(1,int(self.driver_teeth))

    @property
    def output_rpm(self) -> float:
        return self.rpm*max(1,int(self.driver_teeth))/max(1,int(self.driven_teeth))

    def render(self, dur: float, *, load: float=.6, sr: int=DEFAULT_SR, seed: int=0) -> np.ndarray:
        n=max(1,int(round(dur*_PHYS_SR))); t=np.arange(n)/_PHYS_SR; rng=np.random.default_rng(seed)
        fm=self.mesh_hz; shaft=max(1.,self.rpm/60.)
        phase=2*np.pi*fm*t + .06*(.2+self.backlash)*np.sin(2*np.pi*shaft*t)
        mesh=np.sin(phase)+.28*np.sin(2*phase)+.11*np.sin(3*phase)
        side=.24*np.sin(phase)*np.sin(2*np.pi*shaft*t)
        impacts=(rng.random(n) < min(.08, self.backlash*.0015*fm))*rng.standard_normal(n)
        exc=(mesh+side)*(.2+.8*load)+impacts*.6
        modes=_modal.Modes(np.array([self.housing_frequency,self.housing_frequency*1.42,self.housing_frequency*2.05]),
                           np.array([18.,27.,39.]),np.array([1,.5,.25]),"gear_housing")
        y=.34*mesh+_modal.render_modes(exc,modes)
        p=np.max(np.abs(y))+1e-12
        return _to_sr(np.tanh(y/p*1.28)*.6,sr)


@dataclass
class Explosion:
    """Reduced blast: N-wave shock, low-frequency expansion, debris and distance."""
    energy: float = 1.0
    distance_m: float = 8.0
    debris: float = 0.45
    enclosure: float = 0.15

    def render(self, dur: float = 2.0, *, sr: int=DEFAULT_SR, seed: int=0) -> np.ndarray:
        dur=max(.15,float(dur)); n=max(1,int(round(dur*_PHYS_SR))); t=np.arange(n)/_PHYS_SR
        rng=np.random.default_rng(seed); e=max(.01,float(self.energy)); d=max(.5,float(self.distance_m))
        # N-wave duration increases with scale and propagation distance.
        T=np.clip(.006*e**(1/3)+.00045*d,.004,.09)
        m=min(n,max(4,int(T*_PHYS_SR))); u=np.linspace(-1,1,m)
        nw=np.zeros(n); nw[:m]=(np.sign(-u)*(1-np.abs(u)))
        # Smooth distance-dominant high frequencies by short moving average.
        smooth=max(1,int(np.clip(d*.000035*_PHYS_SR,1,80)))
        if smooth>1: nw=np.convolve(nw,np.ones(smooth)/smooth,mode='same')
        # Expansion/ground-coupled rumble.
        rumble=np.sin(2*np.pi*(38+18*np.exp(-t/.22))*t)*np.exp(-t/(.38+.2*e))
        noise=rng.standard_normal(n); env=np.exp(-t/(.18+.22*e))
        blast=noise*env
        # Debris as sparse decaying impacts, with rate tied to energy.
        debris=np.zeros(n); count=int((10+28*e)*np.clip(self.debris,0,1))
        for _ in range(count):
            k=int(rng.uniform(.03,min(dur,.75))*_PHYS_SR)
            if k>=n: continue
            L=min(n-k,int(rng.uniform(.015,.07)*_PHYS_SR))
            tt=np.arange(L)/_PHYS_SR; f=rng.uniform(350,4200)
            debris[k:k+L]+=rng.uniform(.03,.16)*np.sin(2*np.pi*f*tt)*np.exp(-tt/rng.uniform(.006,.035))
        # crude reflection taps for enclosure; direct path is distance delayed.
        y=1.0*nw + .62*rumble + .3*blast + self.debris*debris
        if self.enclosure>0:
            wet=np.zeros(n)
            for delay,g in ((.045,.36),(.082,.22),(.137,.13)):
                k=int((delay+.0007*d)*_PHYS_SR)
                if k<n: wet[k:]+=y[:n-k]*g*self.enclosure
            y+=wet
        delay=int(d/343.0*_PHYS_SR)
        if delay>0:
            out=np.zeros(n); 
            if delay<n: out[delay:]=y[:n-delay]
            y=out
        y/=max(d,1.0)**.72
        p=np.max(np.abs(y))+1e-12
        return _to_sr(np.tanh(y/p*1.5)*.78,sr)

# ============================================================================
# v0.20 convenience methods: local interactions on the shared string core
# ============================================================================
def _plucked_local_note(self, frequency: float, dur: float, *, technique: str='pick',
                        velocity: float=.8, sr: int=DEFAULT_SR, hardness: float=.7,
                        fret: float=0.0, buzz: float=.0, palm_mute: float=.0,
                        harmonic: int|None=None, seed: int=0) -> np.ndarray:
    from .strings import StringExciter, FretboardGeometry
    f=max(20.,float(frequency)); k=self.kind.lower()
    if k in ('bass','upright_bass','finger_bass') or f<90:
        props=WoundStringPhysicalProperties.from_frequency(f,winding_material='nickel')
    else:
        mat='nylon' if k in ('harp','koto','classical_guitar') else 'steel'
        props=StringPhysicalProperties.from_frequency(f,material=mat)
    core=DispersiveString(props,sr=sr)
    fb=FretboardGeometry(scale_length_m=props.length_m)
    tech=technique.lower()
    if harmonic is not None or tech in ('harmonic','natural_harmonic'):
        return core.render_natural_harmonic(harmonic or 2,dur,velocity=velocity)
    if tech in ('slap','pop'):
        return core.render_slap(dur,velocity=velocity,pop=(tech=='pop'),fretboard=fb,seed=seed)
    if buzz>0 or tech in ('buzz','fret_buzz'):
        return core.render_fret_buzz(fret,dur,velocity=velocity,fretboard=fb,buzz=max(buzz,.5),position=self.position,seed=seed)
    if fret:
        return core.render_fretted(fret,dur,velocity=velocity,fretboard=fb,hardness=hardness,
                                   position=self.position,palm_mute=palm_mute)
    kind='finger' if tech in ('finger','pizzicato') else ('nail' if tech=='nail' else 'pick')
    return core.render_exciter(dur,StringExciter(kind,hardness,self.position),velocity=velocity,palm_mute=palm_mute)


def _plucked_glissando(self, start_frequency: float, end_frequency: float, dur: float, *,
                       velocity: float=.8, sr: int=DEFAULT_SR) -> np.ndarray:
    f=max(20.,float(start_frequency)); k=self.kind.lower()
    if k in ('bass','upright_bass','finger_bass') or f<90:
        props=WoundStringPhysicalProperties.from_frequency(f,winding_material='nickel')
    else:
        props=StringPhysicalProperties.from_frequency(f,material=('nylon' if k in ('harp','koto') else 'steel'))
    return DispersiveString(props,sr=sr).render_glissando(dur,start_frequency,end_frequency,
                                                          velocity=velocity,position=self.position)

PluckedString.local_note = _plucked_local_note
PluckedString.glissando = _plucked_glissando


def _bowed_glissando(self, start_frequency: float, end_frequency: float, dur: float, *,
                     sr: int=DEFAULT_SR, velocity: float=.75) -> np.ndarray:
    # A reduced continuous-length bowed gesture: establish the same physical
    # string construction then move its delay continuously. Bow friction color
    # is reintroduced by a small high-passed component from the ordinary bow.
    f=max(20.,float(start_frequency)); body=self.body.lower()
    if body=='cello':
        props=WoundStringPhysicalProperties.from_frequency(f,length_m=max(.62,min(.72,.69*220/f)),
                    core_radius_m=.00038,winding_wire_radius_m=.00018,winding_material='nickel')
    else:
        props=StringPhysicalProperties.from_frequency(f,length_m=max(.28,min(.34,.328*440/f)),
                    radius_m=.00032,material='steel',t60_s=3.4)
    y=DispersiveString(props,sr=sr).render_glissando(dur,start_frequency,end_frequency,
                                                     velocity=velocity,position=self.bow_position)
    return y
BowedString.glissando = _bowed_glissando

# ==========================================================================
# v0.21 expressive shared-string gestures
# ==========================================================================
def _plucked_expressive_note(self, frequency: float, dur: float, *, technique: str='pick',
                             velocity: float=.8, sr: int=DEFAULT_SR, fret: float=0.0,
                             cents: float=100.0, end_frequency: float|None=None,
                             harmonic: int=3, depth_cents: float=18.0,
                             rate_hz: float=5.2, roughness: float=.45,
                             pressure: float=.5, seed: int=0) -> np.ndarray:
    from .strings import FretboardGeometry
    f=max(20.,float(frequency)); k=self.kind.lower()
    if k in ('bass','upright_bass','finger_bass') or f<90:
        props=WoundStringPhysicalProperties.from_frequency(f,winding_material='nickel')
    else:
        props=StringPhysicalProperties.from_frequency(f,material=('nylon' if k in ('harp','koto','classical_guitar') else 'steel'))
    core=DispersiveString(props,sr=sr)
    fb=FretboardGeometry(scale_length_m=props.length_m)
    tech=technique.lower().replace('-','_')
    if tech in ('bend','bending'):
        return core.render_bend(dur,cents=cents,velocity=velocity,position=self.position)
    if tech in ('vibrato','finger_vibrato'):
        return core.render_vibrato(dur,depth_cents=depth_cents,rate_hz=rate_hz,velocity=velocity,position=self.position)
    if tech in ('slide','metal_slide','bottleneck'):
        ef=float(end_frequency if end_frequency is not None else f*2**(cents/1200.))
        return core.render_slide(dur,f,ef,velocity=velocity,position=self.position,
                                 roughness=roughness,pressure=pressure,seed=seed)
    if tech in ('buzz_feedback','fret_buzz_feedback'):
        return core.render_fret_buzz_feedback(fret,dur,velocity=velocity,fretboard=fb,buzz=.75,position=self.position)
    if tech in ('hammer_on','hammeron'):
        return core.render_hammer_on(fret,dur,velocity=velocity,fretboard=fb)
    if tech in ('pull_off','pulloff'):
        return core.render_pull_off(fret,dur,velocity=velocity,fretboard=fb)
    if tech in ('tap','tapping'):
        return core.render_tapping(fret,dur,velocity=velocity,fretboard=fb)
    if tech in ('dead','dead_note','muted'):
        return core.render_dead_note(dur,velocity=velocity,position=self.position)
    if tech in ('pinch','pinch_harmonic','artificial_harmonic'):
        return core.render_pinch_harmonic(harmonic,dur,velocity=velocity)
    return self.local_note(f,dur,technique=technique,velocity=velocity,sr=sr,fret=fret,
                           harmonic=(harmonic if 'harmonic' in tech else None),seed=seed)

PluckedString.expressive_note = _plucked_expressive_note


def _bowed_vibrato_note(self, frequency: float, dur: float, *, sr: int=DEFAULT_SR,
                        depth_cents: float=16.0, rate_hz: float=5.3,
                        velocity: float=.75) -> np.ndarray:
    f=max(20.,float(frequency)); body=self.body.lower()
    if body=='cello':
        props=WoundStringPhysicalProperties.from_frequency(f,length_m=max(.62,min(.72,.69*220/f)),
                    core_radius_m=.00038,winding_wire_radius_m=.00018,winding_material='nickel')
    else:
        props=StringPhysicalProperties.from_frequency(f,length_m=max(.28,min(.34,.328*440/f)),
                    radius_m=.00032,material='steel',t60_s=3.4)
    return DispersiveString(props,sr=sr).render_vibrato(dur,depth_cents=depth_cents,
            rate_hz=rate_hz,velocity=velocity,position=self.bow_position,onset_s=.09)

BowedString.vibrato_note = _bowed_vibrato_note


# ==========================================================================
# v0.22 continuous combined string performance facade
# ==========================================================================
def _plucked_performance(self, frequency: float, dur: float, *, gestures=None,
                         velocity: float=.8, sr: int=DEFAULT_SR,
                         initial_fret: float=0.0, exciter_kind: str='pick',
                         hardness: float=.72):
    from .strings import StringPerformance, StringGesture, StringExciter
    f=max(20.,float(frequency)); k=self.kind.lower()
    if k in ('bass','upright_bass','finger_bass') or f<90:
        props=WoundStringPhysicalProperties.from_frequency(f,winding_material='nickel')
    else:
        props=StringPhysicalProperties.from_frequency(f,material=('nylon' if k in ('harp','koto','classical_guitar') else 'steel'))
    gs=[]
    for g in (gestures or []):
        if isinstance(g,StringGesture): gs.append(g)
        elif isinstance(g,dict): gs.append(StringGesture(**g))
        else: raise TypeError('gestures must contain StringGesture or dict entries')
    perf=StringPerformance(props,sr=sr)
    y=perf.render(dur,gestures=gs,velocity=velocity,initial_fret=initial_fret,
                  exciter=StringExciter(exciter_kind,hardness,self.position))
    self.last_performance_diagnostics=perf.last_diagnostics
    return y

PluckedString.performance = _plucked_performance
