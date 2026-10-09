"""Stateful reduced-order physical piano network (v0.17).

The model is intentionally a realtime surrogate rather than a FEM piano.  It
keeps the physically important topology: struck unison courses exchange energy
through one bridge and one modal soundboard, while a keyboard-wide resonance
bank responds to bridge motion. Dampers and sustain pedal change losses rather
than merely gating the final audio.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np

from .graphsolver import ThiranDelay, FrequencyDependentLoss
from .networks import BridgeImpedance, ModalSoundboard
from .strings import StringPhysicalProperties
from .graphsolver import StiffStringDispersion


def midi_frequency(note: int) -> float:
    return 440.0 * (2.0 ** ((int(note) - 69) / 12.0))


@dataclass
class FrequencyDependentBridge(BridgeImpedance):
    """Bridge with a simple frequency-dependent mechanical compliance proxy.

    The low-passed force drives the heavy bridge body while a smaller high-
    frequency component is passed directly to the soundboard.  This captures
    the fact that a real bridge is not a frequency-independent point coupler.
    """
    hf_coupling: float = 0.16
    crossover_hz: float = 900.0

    def __post_init__(self):
        super().__post_init__()
        self._lp_force = 0.0
        self._a = math.exp(-2.0 * math.pi * max(20.0, self.crossover_hz) / self.sr)
        self.last_hf_force = 0.0

    def step_split(self, force_n: float) -> tuple[float, float]:
        self._lp_force = self._a * self._lp_force + (1.0 - self._a) * float(force_n)
        hf = float(force_n) - self._lp_force
        self.last_hf_force = self.hf_coupling * hf
        return super().step(self._lp_force), self.last_hf_force

    def step_note_forces(self, note_forces: dict[int, float]) -> tuple[float, float, dict[int, float]]:
        """Drive one bridge with note-position dependent participation.

        This is a realtime surrogate for the fact that different bridge regions
        and soundboard directions have different local mobilities.  It preserves
        a single shared mechanical bridge state while giving each course a
        slightly different returned bridge velocity.
        """
        if not note_forces:
            v,hf=self.step_split(0.0)
            self.last_note_velocities={}
            return v,hf,{}
        weighted=0.0; weights={}
        for note,force in note_forces.items():
            x=float(np.clip((int(note)-21)/87.0,0.0,1.0))
            w=0.86 + 0.20*math.sin(math.pi*x) + 0.05*(x-0.5)
            weights[int(note)]=w
            weighted += w*float(force)
        v,hf=self.step_split(weighted)
        local={}
        for note,w in weights.items():
            x=float(np.clip((note-21)/87.0,0.0,1.0))
            # Position-dependent local mobility around the shared bridge state.
            local[note]=v*(0.88 + 0.18*w + 0.04*math.cos(2*math.pi*x))
        self.last_note_velocities=local
        return v,hf,local




@dataclass
class PianoHammerAction:
    """Reduced hammer/action contact model with nonlinear felt and rebound.

    The contact force follows ``F = K*x**p`` while the hammer is compressing the
    felt against the string.  The returned waveform is normalized for the
    realtime string renderer, while ``last_force_n`` retains the force-domain
    pulse for diagnostics and future fully unit-preserving coupling.
    """
    midi_note: int
    sr: int = 48000
    felt_stiffness: float | None = None
    felt_exponent: float | None = None
    hammer_mass_kg: float | None = None
    restitution: float = 0.28

    def __post_init__(self):
        pos=float(np.clip((self.midi_note-21)/87.0,0.0,1.0))
        # Reduced-order keyboard scaling: bass hammers are heavier and softer.
        self.hammer_mass_kg = float(self.hammer_mass_kg if self.hammer_mass_kg is not None
                                    else (0.012 - 0.0075*pos))
        self.felt_exponent = float(self.felt_exponent if self.felt_exponent is not None
                                   else (2.75 + 0.75*pos))
        self.felt_stiffness = float(self.felt_stiffness if self.felt_stiffness is not None
                                    else (1.1e9 * (1.0 + 1.7*pos)))
        self.last_force_n=np.zeros(0,float)
        self.last_rebound_velocity_m_s=0.0
        self.last_contact_time_s=0.0

    def strike(self, velocity: float, *, soft: bool=False) -> np.ndarray:
        vel=float(np.clip(velocity,0.0,1.0))
        # Map MIDI-like velocity to a plausible hammer speed range.
        v=(0.45 + 2.75*vel**1.35) * (0.78 if soft else 1.0)
        m=max(self.hammer_mass_kg,1e-5); k=max(self.felt_stiffness,1.0); p=max(self.felt_exponent,1.1)
        dt=1.0/self.sr; x=0.0; force=[]
        max_n=max(8,int(round(0.018*self.sr)))
        for _ in range(max_n):
            x=max(0.0,x+v*dt)
            f=k*(x**p) if x>0 else 0.0
            force.append(f)
            v += (-f/m)*dt
            if v < 0.0:
                # Follow decompression until separation.
                x=max(0.0,x+v*dt)
                if x <= 0.0:
                    break
        arr=np.asarray(force,float)
        if len(arr)==0 or not np.any(arr):
            arr=np.array([0.0],float)
        self.last_force_n=arr
        self.last_contact_time_s=len(arr)/self.sr
        self.last_rebound_velocity_m_s=-abs(v)*float(np.clip(self.restitution,0.0,1.0))
        peak=max(float(np.max(np.abs(arr))),1e-12)
        return arr/peak * (0.18 + 0.82*vel**1.4) * (0.78 if soft else 1.0)


@dataclass
class PianoStringCourse:
    """One piano note/course containing one to three slightly detuned strings."""
    midi_note: int
    sr: int = 48000
    detune_cents: tuple[float, ...] = (-1.5, 0.0, 1.4)
    impedance_n_s_m: float = 0.65
    coupling: float = 0.055
    loss_dc: float = 0.99972
    loss_hf: float = 0.990
    damper_closed_loss: float = 0.91

    def __post_init__(self):
        f0 = midi_frequency(self.midi_note)
        # Real pianos use fewer strings in the bass.
        if self.midi_note < 40:
            cents = (0.0,)
        elif self.midi_note < 52:
            cents = (-1.0, 1.0)
        else:
            cents = self.detune_cents
        self.frequencies = tuple(f0 * 2.0 ** (c / 1200.0) for c in cents)
        # v0.18: every unison string owns an explicit physical specification.
        # Geometry is a reduced keyboard scaling; the waveguide remains realtime.
        keyboard_pos = float(np.clip((self.midi_note - 21) / 87.0, 0.0, 1.0))
        length = 1.55*(1.0-keyboard_pos) + 0.19*keyboard_pos
        radius = 0.00105*(1.0-keyboard_pos) + 0.00018*keyboard_pos
        self.physical_properties = [
            StringPhysicalProperties(length_m=length, radius_m=radius, material='music_wire',
                                     frequency_hz=f, t60_s=5.8*(1.0-keyboard_pos)+2.0*keyboard_pos)
            for f in self.frequencies
        ]
        self.physical_bridge_impedance_n_s_m = float(np.mean(
            [q.characteristic_impedance_n_s_m for q in self.physical_properties]))
        self.lines = []
        self.orthogonal_lines = []
        self._x_dispersion=[]; self._y_dispersion=[]
        for f,q in zip(self.frequencies,self.physical_properties):
            d = max(2.05, self.sr / max(f, 1.0))
            self.lines.append((ThiranDelay(d, max(64, int(self.sr / 8))),
                               FrequencyDependentLoss(self.loss_dc, self.loss_hf, 0.45)))
            # Second transverse polarization with a tiny frequency split.
            f2=f*2.0**(q.polarization_split_cents/1200.0)
            d2=max(2.05,self.sr/max(f2,1.0))
            self.orthogonal_lines.append((ThiranDelay(d2,max(64,int(self.sr/8))),
                                          FrequencyDependentLoss(self.loss_dc**1.0003, self.loss_hf*0.999, 0.48)))
            da=float(np.clip(math.sqrt(max(q.inharmonicity_B,0.0))*18.0,0.0,1.0))
            self._x_dispersion.append(StiffStringDispersion(da,3))
            self._y_dispersion.append(StiffStringDispersion(min(1.0,da*1.05),3))
        self.key_down = False
        self.sustain_held = False
        self.sostenuto_held = False
        self.damper_amount = 1.0  # 0=open, 1=fully touching the string
        self.soft_pedal = False
        self._hammer = np.zeros(0, dtype=float)
        self._hammer_pos = 0
        self._hammer_gains = np.ones(len(self.lines), dtype=float)
        self.last_force_n = 0.0
        self.last_direct = 0.0

        # Hammer geometry varies over the keyboard. Bass hammers strike farther
        # from the termination, while treble hammers move closer to it.
        keyboard_pos = np.clip((self.midi_note - 21) / 87.0, 0.0, 1.0)
        self.hammer_position_ratio = float(0.145 - 0.060 * keyboard_pos)
        self.bridge_position_ratio = float(0.22 + 0.66 * keyboard_pos)
        self.hammer_action = PianoHammerAction(self.midi_note, sr=self.sr)
        self._orth_incoming=np.zeros(len(self.lines),float)
        self._polarization_coupling=float(0.012 + 0.014*keyboard_pos)
        # One reduced longitudinal mode.  It is weak but exchanges energy with
        # the transverse planes at the bridge.
        self.longitudinal_frequency_hz=min(0.38*self.sr, f0*(6.0+2.0*keyboard_pos))
        lr=10**(-3.0/max(0.42*self.sr,1.0))
        lth=2*math.pi*self.longitudinal_frequency_hz/self.sr
        self._long_a1=2*lr*math.cos(lth); self._long_a2=-(lr*lr)
        self._long_y1=0.0; self._long_y2=0.0

    @property
    def damper_open(self) -> bool:
        return self.key_down or self.sustain_held or self.sostenuto_held or self.damper_amount <= 0.0

    def note_on(self, velocity: float = 0.8, *, soft_pedal: bool = False):
        self.key_down = True
        self.soft_pedal = bool(soft_pedal)
        vel = float(np.clip(velocity, 0, 1))
        pulse = self.hammer_action.strike(vel, soft=self.soft_pedal)

        # Hammer position appears as a weak comb in the string excitation.
        f0 = midi_frequency(self.midi_note)
        pos = self.hammer_position_ratio + (0.008 if self.soft_pedal else 0.0)
        echo_delay = max(1, int(round((self.sr / max(f0, 1.0)) * pos)))
        n=len(pulse)
        exc = np.zeros(n + echo_delay, dtype=float)
        exc[:n] += pulse
        exc[echo_delay:echo_delay+n] -= pulse * (0.14 if self.soft_pedal else 0.22)
        self._hammer = exc
        self._hammer_pos = 0

        # Una corda mechanically shifts the action: treble notes strike fewer
        # strings. Bass single-string courses remain single-string courses.
        self._hammer_gains = np.zeros(len(self.lines), dtype=float)
        struck = len(self.lines)
        if self.soft_pedal and struck > 1:
            struck -= 1
        self._hammer_gains[:struck] = 1.0

    def note_off(self):
        self.key_down = False

    def set_sustain(self, held: bool):
        self.sustain_held = bool(held)

    def set_sostenuto(self, held: bool):
        self.sostenuto_held = bool(held)

    def set_damper(self, amount: float):
        self.damper_amount = float(np.clip(amount, 0.0, 1.0))

    def read_force(self) -> tuple[np.ndarray, float]:
        incoming = np.array([dl.read() for dl, _ in self.lines], dtype=float)
        orth = np.array([dl.read() for dl, _ in self.orthogonal_lines], dtype=float)
        self._orth_incoming=orth
        if self._hammer_pos < len(self._hammer):
            h = float(self._hammer[self._hammer_pos])
            self._hammer_pos += 1
            for k in range(len(incoming)):
                incoming[k] += h * self._hammer_gains[k] * (1.0 - 0.025 * k)
                orth[k] += 0.06*h*self._hammer_gains[k]
        # Weak longitudinal reaction participates in bridge force.
        force = -max(self.impedance_n_s_m, 1e-5) * float(np.sum(incoming) + 0.28*np.sum(orth) + 0.04*self._long_y1)
        self.last_force_n = force
        return incoming, force

    def write(self, incoming: np.ndarray, common_velocity: float) -> float:
        c = float(np.clip(self.coupling, 0.0, 0.24))
        pc=float(np.clip(self._polarization_coupling,0.0,0.08))
        vals=[]; ovals=[]
        for idx,((dl, loss), v) in enumerate(zip(self.lines, incoming)):
            ov=float(self._orth_incoming[idx])
            # Exchange a small amount of energy between the two transverse planes.
            pv=(1.0-c-pc)*float(v) + c*float(common_velocity) + pc*ov
            qv=(1.0-c-pc)*ov + c*(0.78*float(common_velocity)) + pc*float(v)
            pv=loss.process(self._x_dispersion[idx].process(pv))
            odl,oloss=self.orthogonal_lines[idx]
            qv=oloss.process(self._y_dispersion[idx].process(qv))
            if not self.damper_open:
                damp=self.damper_closed_loss ** self.damper_amount
                pv*=damp; qv*=damp
            dl.write(pv); odl.write(qv)
            vals.append(pv); ovals.append(qv)
        # Longitudinal mode receives bridge acceleration / polarization mismatch.
        drive=0.0018*(float(common_velocity)-0.5*(float(np.mean(vals))+float(np.mean(ovals))))
        ly=self._long_a1*self._long_y1 + self._long_a2*self._long_y2 + drive
        self._long_y2=self._long_y1; self._long_y1=ly
        self.last_direct=float(np.sum(vals) + 0.22*np.sum(ovals) + 0.035*ly)
        return self.last_direct

    def polarization_energy(self) -> float:
        vals=np.array([dl.read() for dl,_ in self.orthogonal_lines],float)
        return float(np.dot(vals,vals))

    def longitudinal_energy(self) -> float:
        return float(self._long_y1*self._long_y1 + self._long_y2*self._long_y2)

    def energy_proxy(self) -> float:
        vals = np.array([dl.read() for dl, _ in self.lines], float)
        return float(np.dot(vals, vals) + self.polarization_energy() + 0.1*self.longitudinal_energy())


@dataclass
class SympatheticResonatorBank:
    """Keyboard-wide passive resonance bank used for unstruck/duplex strings."""
    sr: int = 48000
    midi_low: int = 21
    midi_high: int = 108
    coupling: float = 0.0028
    aliquot_ratios: tuple[float, ...] = (2.0, 3.0)

    def __post_init__(self):
        self.notes = np.arange(self.midi_low, self.midi_high + 1)
        base_freq = np.array([midi_frequency(int(n)) for n in self.notes])
        self.base_count = len(base_freq)
        # Bass strings decay longer; treble strings shorter.
        frac = (self.notes - self.midi_low) / max(1, self.midi_high - self.midi_low)
        base_open = 7.0 * (1.0 - frac) + 2.0 * frac
        base_closed = 0.18 * (1.0 - frac) + 0.08 * frac
        # Aliquot strings are short, high, and intentionally undamped.  They are
        # represented as extra passive modes rather than full string courses.
        aliquots=[]
        self.aliquot_source_notes=[]
        for n in range(60, min(self.midi_high, 96) + 1):
            f0=midi_frequency(n)
            # Duplex scaling length changes gradually across the treble.  A
            # small note-dependent stretch keeps the bank from becoming a set
            # of perfectly coincident integer harmonics.
            stretch = 1.0 + 0.00045 * (n - 60)
            for r in self.aliquot_ratios:
                f=f0*float(r)*stretch
                if f < .45*self.sr:
                    aliquots.append(f)
                    self.aliquot_source_notes.append(n)
        extra=np.asarray(aliquots,float)
        self.freq=np.concatenate([base_freq,extra]) if len(extra) else base_freq
        self.t60_open=np.concatenate([base_open, np.full(len(extra),2.4)])
        self.t60_closed=np.concatenate([base_closed, np.full(len(extra),2.4)])
        self.y1 = np.zeros_like(self.freq)
        self.y2 = np.zeros_like(self.freq)
        self.key_down = np.zeros(self.base_count, dtype=bool)
        self.sustain = False
        self._phase = 2 * np.pi * self.freq / self.sr

    def set_key(self, midi_note: int, down: bool):
        if self.midi_low <= midi_note <= self.midi_high:
            self.key_down[midi_note - self.midi_low] = bool(down)

    def set_sustain(self, held: bool):
        self.sustain = bool(held)

    def step(self, bridge_velocity: float) -> tuple[float, float]:
        opened_base = self.key_down | self.sustain
        if len(self.freq) > self.base_count:
            opened = np.concatenate([opened_base, np.ones(len(self.freq)-self.base_count, dtype=bool)])
        else:
            opened = opened_base
        t60 = np.where(opened, self.t60_open, self.t60_closed)
        r = np.power(1e-3, 1.0 / np.maximum(t60 * self.sr, 1.0))
        a1 = 2.0 * r * np.cos(self._phase)
        a2 = -(r * r)
        # Weak drive prevents this bank from becoming an artificial chorus.
        drive = self.coupling * float(bridge_velocity) * np.sin(self._phase)
        y = a1 * self.y1 + a2 * self.y2 + drive
        self.y2 = self.y1
        self.y1 = y
        radiation = float(np.sum(y) / math.sqrt(len(y)))
        reaction = -float(np.sum(y - self.y2)) * 2.0e-4
        return radiation, reaction

    def note_energy(self, midi_note: int) -> float:
        if not (self.midi_low <= midi_note <= self.midi_high):
            return 0.0
        i = midi_note - self.midi_low
        return float(self.y1[i] ** 2 + self.y2[i] ** 2)


@dataclass
class PolyphonicPiano:
    """Stateful reduced-order 88-key piano with shared bridge/soundboard.

    Events are sample accurate within the rendered block.  Sustain pedal changes
    damper losses in the physical state; note-off therefore does not silence an
    already vibrating string while the pedal is held.
    """
    sr: int = 48000
    bridge: FrequencyDependentBridge | None = None
    soundboard: ModalSoundboard | None = None
    global_coupling: float = 0.055
    sympathetic_coupling: float = 0.0028
    aliquot_ratios: tuple[float, ...] = (2.0, 3.0, 4.0)

    def __post_init__(self):
        self.bridge = self.bridge or FrequencyDependentBridge(sr=self.sr)
        self.soundboard = self.soundboard or ModalSoundboard(sr=self.sr)
        self.sympathetic = SympatheticResonatorBank(sr=self.sr, coupling=self.sympathetic_coupling)
        self.courses: dict[int, PianoStringCourse] = {}
        self.sustain = False
        self.sostenuto = False
        self.soft = False
        self.sostenuto_notes: set[int] = set()
        self.damper_overrides: dict[int, float] = {}
        self.time_samples = 0
        self.last_bridge_energy_j = 0.0
        self.last_soundboard_energy = 0.0

    def _course(self, midi_note: int) -> PianoStringCourse:
        midi_note = int(midi_note)
        if midi_note not in self.courses:
            self.courses[midi_note] = PianoStringCourse(midi_note, sr=self.sr,
                                                        coupling=self.global_coupling)
            self.courses[midi_note].set_sustain(self.sustain)
            self.courses[midi_note].set_sostenuto(midi_note in self.sostenuto_notes)
            self.courses[midi_note].set_damper(self.damper_overrides.get(midi_note, 1.0))
        return self.courses[midi_note]

    def note_on(self, midi_note: int, velocity: float = 0.8):
        c = self._course(midi_note)
        c.note_on(velocity, soft_pedal=self.soft)
        self.sympathetic.set_key(int(midi_note), True)

    def note_off(self, midi_note: int):
        if int(midi_note) in self.courses:
            self.courses[int(midi_note)].note_off()
        self.sympathetic.set_key(int(midi_note), False)

    def sustain_pedal(self, down: bool):
        self.sustain = bool(down)
        self.sympathetic.set_sustain(self.sustain)
        for c in self.courses.values():
            c.set_sustain(self.sustain)

    def sostenuto_pedal(self, down: bool):
        """Capture only keys held when the sostenuto pedal goes down."""
        down = bool(down)
        if down and not self.sostenuto:
            self.sostenuto_notes = {n for n, c in self.courses.items() if c.key_down}
        elif not down:
            self.sostenuto_notes.clear()
        self.sostenuto = down
        for n, c in self.courses.items():
            c.set_sostenuto(down and n in self.sostenuto_notes)

    def una_corda(self, down: bool):
        """Set soft-pedal state for subsequent hammer strikes."""
        self.soft = bool(down)

    def set_damper(self, midi_note: int, amount: float):
        """Set per-note damper contact: 0=open, 1=fully damped."""
        note = int(midi_note)
        self.damper_overrides[note] = float(np.clip(amount, 0.0, 1.0))
        if note in self.courses:
            self.courses[note].set_damper(self.damper_overrides[note])

    def _step(self) -> float:
        reads = []
        note_forces = {}
        total_force = 0.0
        for note, course in list(self.courses.items()):
            incoming, force = course.read_force()
            reads.append((note, course, incoming))
            note_forces[int(note)]=force
            total_force += force

        # One shared bridge state, with note-position dependent local mobility.
        bridge_v, hf_force, local_bridge = self.bridge.step_note_forces(note_forces)
        board_rad, board_reaction = self.soundboard.step(-(sum(note_forces.values()) + hf_force))
        sym_rad, sym_reaction = self.sympathetic.step(bridge_v + board_reaction)

        direct = 0.0
        dead = []
        for note, course in list(self.courses.items()):
            incoming = next(arr for n,c,arr in reads if c is course)
            local_v=local_bridge.get(int(note),bridge_v) + board_reaction + sym_reaction
            direct += course.write(incoming, local_v)
            # Retire fully damped inactive voices to keep long sessions cheap.
            if (not course.key_down and not course.sustain_held and not course.sostenuto_held and
                    course.energy_proxy() < 1e-12 and course._hammer_pos >= len(course._hammer)):
                dead.append(note)
        for note in dead:
            self.courses.pop(note, None)

        self.last_bridge_energy_j = self.bridge.energy_j
        self.last_soundboard_energy = self.soundboard.modal_energy
        self.time_samples += 1
        return 0.12 * direct + 0.72 * board_rad + 0.16 * sym_rad

    def render(self, dur: float) -> np.ndarray:
        n = max(1, int(round(float(dur) * self.sr)))
        y = np.empty(n, dtype=float)
        for i in range(n):
            y[i] = self._step()
        # Stateful render intentionally does NOT normalize each block: doing so
        # would destroy relative note dynamics across successive calls.
        peak = float(np.max(np.abs(y))) if len(y) else 0.0
        if peak > 1.0:
            y /= peak
        return y

    def render_events(self, events, dur: float) -> np.ndarray:
        """Render sample-accurate event sequence.

        Accepted event forms are dictionaries, e.g.
        ``{'time': .2, 'type': 'note_on', 'note': 60, 'velocity': .8}``,
        ``{'time': 1.0, 'type': 'note_off', 'note': 60}``, and
        ``{'time': .5, 'type': 'sustain', 'down': True}``.
        """
        n = max(1, int(round(float(dur) * self.sr)))
        schedule: dict[int, list[dict]] = {}
        for ev in events:
            e = dict(ev)
            idx = int(np.clip(round(float(e.get('time', 0.0)) * self.sr), 0, n - 1))
            schedule.setdefault(idx, []).append(e)
        y = np.empty(n, dtype=float)
        for i in range(n):
            for e in schedule.get(i, ()):
                typ = e.get('type')
                if typ == 'note_on':
                    self.note_on(int(e['note']), float(e.get('velocity', .8)))
                elif typ == 'note_off':
                    self.note_off(int(e['note']))
                elif typ in ('sustain', 'pedal'):
                    self.sustain_pedal(bool(e.get('down', e.get('value', True))))
                elif typ == 'sostenuto':
                    self.sostenuto_pedal(bool(e.get('down', e.get('value', True))))
                elif typ in ('una_corda', 'soft'):
                    self.una_corda(bool(e.get('down', e.get('value', True))))
                elif typ == 'damper':
                    self.set_damper(int(e['note']), float(e.get('amount', e.get('value', 1.0))))
                else:
                    raise ValueError(f"unknown piano event type: {typ!r}")
            y[i] = self._step()
        peak = float(np.max(np.abs(y))) if len(y) else 0.0
        if peak > 1.0:
            y /= peak
        return y
