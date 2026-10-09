"""Higher-level passive waveguide networks built on the v0.11-v0.12 core.

The emphasis is reusable topology: bores with multiple side holes, coupled
strings sharing a bridge, and acoustic cavities with several ports.  These are
reduced real-time models, not full FEM/BEM solvers.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
from .graphsolver import (ThiranDelay, FractionalDelay, FrequencyDependentLoss,
                          MultiportScatteringJunction, BellRadiation)
from .ports import PhysicalPort


def _norm(x, peak=.9):
    x=np.asarray(x,float)
    m=float(np.max(np.abs(x))) if x.size else 0.0
    return x if m<=1e-12 else x*(peak/m)


@dataclass
class DynamicToneHole:
    """Side-hole model with smoothly variable opening [0,1]."""
    radius_m: float=.0035
    chimney_m: float=.004
    openness: float=0.0
    rho: float=1.2041
    c: float=343.0

    @property
    def area(self): return math.pi*max(self.radius_m,1e-6)**2

    def impedance(self, sr: int) -> float:
        # Lumped inertive hole impedance magnitude.  The large impedance when
        # closed continuously removes the side branch without a topology jump.
        o=float(np.clip(self.openness,0.0,1.0))
        leff=max(1e-5,self.chimney_m+1.5*self.radius_m)
        base=self.rho*self.c/max(self.area,1e-12)
        inertive=base*(1.0 + leff*sr/self.c)
        return inertive/max(o*o,1e-5)


@dataclass
class MultiHoleBore:
    """Cylindrical bore with a chain of continuously-openable side holes.

    ``hole_positions`` are fractions of acoustic bore length from mouth to bell.
    Each bore section is a bidirectional delay; each tone-hole junction is a
    lossless three-port scattering junction (left bore, right bore, side hole).
    """
    length_m: float=.66
    bore_radius_m: float=.007
    hole_positions: tuple[float,...]=(0.38,0.47,0.55,0.63,0.70,0.77)
    hole_radii_m: tuple[float,...]=(.003,.0032,.0033,.0035,.0037,.004)
    hole_chimneys_m: tuple[float,...]=(.004,)*6
    sr: int=48000
    loss_dc: float=.999
    loss_hf: float=.982
    bell_cutoff_hz: float=900.0

    def __post_init__(self):
        n=len(self.hole_positions)
        if len(self.hole_radii_m)!=n or len(self.hole_chimneys_m)!=n:
            raise ValueError("tone-hole tuples must have equal length")
        pos=np.asarray(self.hole_positions,float)
        if n and (np.any(pos<=0) or np.any(pos>=1) or np.any(np.diff(pos)<=0)):
            raise ValueError("hole_positions must be strictly increasing inside (0,1)")
        rho,c=1.2041,343.0
        self.zb=rho*c/(math.pi*self.bore_radius_m**2)
        boundaries=np.concatenate(([0.0],pos,[1.0]))
        lengths=np.diff(boundaries)*self.length_m
        # One-way delays per segment. A minimum >1 sample keeps interpolation safe.
        self.seg=[]
        for L in lengths:
            d=max(1.05,L/c*self.sr)
            self.seg.append({
                'lr': ThiranDelay(d,max(16,int(self.sr*.02))),
                'rl': ThiranDelay(d,max(16,int(self.sr*.02))),
                'loss_lr': FrequencyDependentLoss(self.loss_dc,self.loss_hf,.55),
                'loss_rl': FrequencyDependentLoss(self.loss_dc,self.loss_hf,.55),
            })
        self.holes=[DynamicToneHole(r,ch,0.0) for r,ch in zip(self.hole_radii_m,self.hole_chimneys_m)]
        self.bell=BellRadiation(self.bell_cutoff_hz,self.sr,1.0)

    def set_fingering(self, openings):
        vals=np.asarray(openings,float)
        if vals.shape!=(len(self.holes),): raise ValueError("fingering length mismatch")
        for h,o in zip(self.holes,vals): h.openness=float(np.clip(o,0,1))

    def _propagate(self, seg, outgoing_left, outgoing_right):
        seg['rl'].write(seg['loss_rl'].process(outgoing_left))
        seg['lr'].write(seg['loss_lr'].process(outgoing_right))

    def step(self, mouth_wave: float=0.0):
        m=len(self.holes)
        # Incoming waves at each segment end from propagation state.
        from_left=[s['lr'].read() for s in self.seg]   # wave travelling mouth->bell arriving right
        from_right=[s['rl'].read() for s in self.seg] # bell->mouth arriving left
        # Process junctions left to right using current incoming waves only.
        to_left=np.zeros(m+1); to_right=np.zeros(m+1); radiated=0.0
        # Mouth boundary: injected outgoing wave plus reflected return (weak source impedance).
        to_right[0]=float(mouth_wave) + 0.08*from_right[0]
        for j,h in enumerate(self.holes):
            inc=np.array([from_left[j], from_right[j+1], 0.0],float)
            zj=MultiportScatteringJunction(np.array([self.zb,self.zb,h.impedance(self.sr)]))
            out=zj.scatter(inc)
            to_left[j]=out[0]
            to_right[j+1]=out[1]
            # Side-port outward pressure contribution weighted by openness.
            radiated += float(out[2])*float(h.openness)*0.16
        # Bell termination on final right-going wave.
        bref,brad=self.bell.process(from_left[-1])
        to_left[-1]=bref; radiated += brad
        # Send new waves through each bore segment.
        for k,s in enumerate(self.seg):
            self._propagate(s,to_left[k],to_right[k])
        mouth_pressure=from_right[0]+mouth_wave
        return float(mouth_pressure),float(radiated)

    def render(self, drive, fingering=None):
        drive=np.asarray(drive,float)
        n=len(drive); y=np.zeros(n); p=np.zeros(n)
        if fingering is not None:
            f=np.asarray(fingering,float)
            dynamic=(f.ndim==2)
            if dynamic and f.shape!=(n,len(self.holes)): raise ValueError("dynamic fingering shape mismatch")
            if not dynamic: self.set_fingering(f)
        else: dynamic=False
        for i,x in enumerate(drive):
            if dynamic: self.set_fingering(f[i])
            p[i],rad=self.step(float(x)); y[i]=p[i]+rad
        return _norm(y),p


@dataclass
class MultiHoleClarinet:
    """Reduced reed + multi-tone-hole clarinet built on :class:`MultiHoleBore`."""
    frequency: float=220.0
    breath: float=.65
    noise: float=.015
    sr: int=48000
    bore: MultiHoleBore|None=None

    def __post_init__(self):
        if self.bore is None:
            # closed-pipe quarter-wave estimate
            L=343.0/(4.0*max(self.frequency,20.0))
            self.bore=MultiHoleBore(length_m=L,sr=self.sr,bell_cutoff_hz=max(550.,2.5*self.frequency))

    def note(self,dur:float,fingering=None,seed=0):
        n=max(1,int(round(dur*self.sr))); rng=np.random.default_rng(seed)
        pm=800.+4200.*np.clip(self.breath,0,1.5)
        drive=np.zeros(n); p_hist=np.zeros(n); flow=np.zeros(n)
        # Close all holes by default; fingering may open them.
        if fingering is None: fingering=np.zeros(len(self.bore.holes))
        y=np.zeros(n)
        rho=1.2041; reed_k=3.5e-8
        for i in range(n):
            if np.asarray(fingering).ndim==2: self.bore.set_fingering(np.asarray(fingering)[i])
            elif i==0: self.bore.set_fingering(fingering)
            pb=p_hist[i-1] if i else 0.0
            dp=max(pm-pb,0.0); opening=max(0.0,1.0-pb/max(pm*1.25,1.0))
            u=reed_k*opening*math.sqrt(2*dp/rho)
            u*=1.+self.noise*rng.standard_normal()
            wave=.5*self.bore.zb*u
            p,rad=self.bore.step(wave); p_hist[i]=p; flow[i]=u; y[i]=p+rad
        return _norm(y), PhysicalPort.acoustic(np.full(n,pm),flow,'reed'), PhysicalPort.acoustic(p_hist,-flow,'multihole_bore')


@dataclass
class MultiHoleFlute:
    """Jet-driven open bore with dynamic tone holes."""
    frequency: float=440.0
    jet_velocity: float=18.0
    sr: int=48000
    bore: MultiHoleBore|None=None

    def __post_init__(self):
        if self.bore is None:
            L=343.0/(2.0*max(self.frequency,20.0))
            self.bore=MultiHoleBore(length_m=L,bore_radius_m=.006,sr=self.sr,bell_cutoff_hz=max(700.,2.5*self.frequency))
        self.jet=FractionalDelay(max(2.,self.sr*.009/(.4*max(self.jet_velocity,.1))),max(32,int(self.sr*.03)),'lagrange3')

    def note(self,dur:float,fingering=None):
        n=max(1,int(round(dur*self.sr))); y=np.zeros(n); flow=np.zeros(n); p=np.zeros(n)
        if fingering is None: fingering=np.zeros(len(self.bore.holes))
        rho=1.2041; area=math.pi*self.bore.bore_radius_m**2
        for i in range(n):
            if np.asarray(fingering).ndim==2: self.bore.set_fingering(np.asarray(fingering)[i])
            elif i==0: self.bore.set_fingering(fingering)
            delayed=self.jet.read(); pb=p[i-1] if i else 0.0
            q=math.tanh((delayed-pb/max(self.bore.zb,1e-9))*4.0)
            u=area*.018*self.jet_velocity*q; wave=.5*self.bore.zb*u
            pp,rad=self.bore.step(wave); p[i]=pp; flow[i]=u; y[i]=pp+rad
            self.jet.write(self.jet_velocity*.025+.2*u/max(area,1e-12))
        drive_p=np.full(n,.5*rho*self.jet_velocity**2)
        return _norm(y), PhysicalPort.acoustic(drive_p,flow,'jet'), PhysicalPort.acoustic(p,-flow,'multihole_bore')


@dataclass
class CoupledStringBank:
    """Several strings coupled through a shared bridge state.

    Useful for piano unisons, mandolin courses and sympathetic strings.
    """
    frequencies: tuple[float,...]=(440.0,440.7,439.3)
    sr: int=48000
    coupling: float=.08
    loss_dc: float=.9995
    loss_hf: float=.988

    def __post_init__(self):
        self.lines=[]
        for f in self.frequencies:
            d=max(2.,self.sr/max(f,1.0))
            self.lines.append((ThiranDelay(d,max(64,int(self.sr/20))),FrequencyDependentLoss(self.loss_dc,self.loss_hf,.45)))
        self.bridge=0.0

    def render(self,dur:float,velocity=.8,excite_index:int=0):
        n=max(1,int(round(dur*self.sr))); y=np.zeros(n)
        excite=max(0,min(int(excite_index),len(self.lines)-1))
        for i in range(n):
            incoming=np.array([dl.read() for dl,_ in self.lines])
            if i < max(2,int(.003*self.sr)):
                incoming[excite]+=velocity*math.sin(math.pi*i/max(1,int(.003*self.sr)-1))
            bridge=float(np.mean(incoming)); self.bridge=.92*self.bridge+.08*bridge
            out=[]
            for k,((dl,loss),x) in enumerate(zip(self.lines,incoming)):
                coupled=(1-self.coupling)*x+self.coupling*self.bridge
                v=loss.process(coupled)
                dl.write(v); out.append(v)
            y[i]=self.bridge
        return _norm(y)


@dataclass
class AcousticCavityNetwork:
    """Lumped central cavity connected to N passive neck/port branches."""
    volume_m3: float=.02
    port_radii_m: tuple[float,...]=(.01,.008)
    port_lengths_m: tuple[float,...]=(.03,.02)
    sr: int=48000
    damping: float=.998

    def __post_init__(self):
        if len(self.port_radii_m)!=len(self.port_lengths_m): raise ValueError("port tuple mismatch")
        rho,c=1.2041,343.0
        self.rho,self.c=rho,c
        self.compliance=max(self.volume_m3,1e-7)/(rho*c*c)
        self.pressure=0.0
        self.flows=np.zeros(len(self.port_radii_m))
        self.port_mass=np.array([rho*(L+1.5*r)/(math.pi*r*r) for r,L in zip(self.port_radii_m,self.port_lengths_m)])

    def step(self,input_flow:float=0.0):
        # Acoustic mass branches driven by common cavity pressure.
        dt=1.0/self.sr
        self.flows += (-self.pressure/self.port_mass)*dt
        net=float(input_flow)-float(np.sum(self.flows))
        self.pressure += (net/self.compliance)*dt
        self.pressure *= self.damping
        return self.pressure,self.flows.copy()

    def render_impulse(self,n:int,amplitude:float=1e-5):
        y=np.zeros(int(n))
        for i in range(int(n)):
            y[i],_=self.step(amplitude if i==0 else 0.0)
        return _norm(y)

# ---------------------------------------------------------------------------
# v0.14 — coupled bridge / soundboard / sympathetic string network
# ---------------------------------------------------------------------------

@dataclass
class BridgeImpedance:
    """Reduced mechanical bridge model (mass–spring–damper).

    The state variables retain mechanical units: force [N], velocity [m/s],
    displacement [m].  It is intentionally low-order; the soundboard carries
    the dense resonance structure.
    """
    mass_kg: float = 0.018
    resonance_hz: float = 115.0
    damping_ratio: float = 0.28
    sr: int = 48000

    def __post_init__(self):
        self.mass_kg=max(float(self.mass_kg),1e-6)
        w=2.0*math.pi*max(float(self.resonance_hz),1.0)
        self.stiffness_n_m=self.mass_kg*w*w
        self.damping_n_s_m=2.0*float(self.damping_ratio)*math.sqrt(self.stiffness_n_m*self.mass_kg)
        self.displacement_m=0.0
        self.velocity_m_s=0.0

    def step(self, force_n: float) -> float:
        dt=1.0/self.sr
        # Semi-implicit Euler is dissipative and robust for this low resonance.
        a=(float(force_n)-self.damping_n_s_m*self.velocity_m_s-
           self.stiffness_n_m*self.displacement_m)/self.mass_kg
        self.velocity_m_s += a*dt
        self.displacement_m += self.velocity_m_s*dt
        return float(self.velocity_m_s)

    @property
    def energy_j(self) -> float:
        return float(.5*self.mass_kg*self.velocity_m_s**2 +
                     .5*self.stiffness_n_m*self.displacement_m**2)


@dataclass
class ModalSoundboard:
    """Shared soundboard represented by a stable bank of damped modes.

    ``frequencies_hz`` and ``t60_s`` describe the reduced modal model.  A force
    at the bridge excites every mode according to ``gains``.  ``step`` returns
    a normalized radiation proxy plus a small mechanical reaction velocity used
    to couple energy back into the bridge/string network.
    """
    frequencies_hz: tuple[float,...] = (92, 126, 171, 233, 318, 435, 596, 815,
                                         1115, 1525, 2085, 2850, 3900)
    t60_s: tuple[float,...] = (2.8,2.5,2.2,1.9,1.65,1.45,1.25,1.05,.88,.72,.58,.46,.36)
    gains: tuple[float,...] | None = None
    sr: int = 48000
    reaction_scale: float = 2.5e-5

    def __post_init__(self):
        f=np.asarray(self.frequencies_hz,float)
        t=np.asarray(self.t60_s,float)
        if f.ndim!=1 or len(f)==0 or len(t)!=len(f):
            raise ValueError("soundboard modal tuples must be non-empty and equal length")
        if self.gains is None:
            # Broadly decreasing modal participation with slight low-mid emphasis.
            g=1.0/np.sqrt(np.maximum(f/100.0,1.0))
        else:
            g=np.asarray(self.gains,float)
            if len(g)!=len(f): raise ValueError("soundboard gains length mismatch")
        self._f=f; self._g=g/max(float(np.max(np.abs(g))),1e-12)
        # -60 dB amplitude after T60: R^(T60*Fs)=1e-3
        self._r=np.power(1e-3,1.0/np.maximum(t*self.sr,1.0))
        th=2*np.pi*f/self.sr
        self._a1=2*self._r*np.cos(th)
        self._a2=-(self._r**2)
        # Scale input conservatively so a few newtons do not saturate modal states.
        self._b=self._g*self._r*np.sin(th)*1e-3
        self._y1=np.zeros_like(f); self._y2=np.zeros_like(f)
        self._prev_sum=0.0

    def step(self, force_n: float) -> tuple[float,float]:
        y=self._a1*self._y1 + self._a2*self._y2 + self._b*float(force_n)
        self._y2=self._y1; self._y1=y
        s=float(np.sum(y))
        # Radiation is a dimensionless pressure proxy; reaction is a tiny
        # velocity-like feedback derived from modal motion.
        radiation=s
        reaction=(s-self._prev_sum)*self.sr*self.reaction_scale
        self._prev_sum=s
        return float(radiation),float(reaction)

    @property
    def modal_energy(self) -> float:
        return float(np.sum(self._y1*self._y1 + self._y2*self._y2))


@dataclass
class SympatheticPiano:
    """Reduced piano network with unison strings, sympathetic strings and one
    shared bridge/soundboard.

    This is deliberately a real-time reduced model, not a full FEM piano.  Each
    string is a passive delay loop.  The strings exchange energy through a
    mechanical bridge whose force drives one modal soundboard.  Unstruck strings
    can therefore acquire energy from the struck course.
    """
    struck_frequency: float = 440.0
    unison_detune_cents: tuple[float,...] = (-1.8, 0.0, 1.6)
    sympathetic_ratios: tuple[float,...] = (0.5, 1.5, 2.0, 2.5, 3.0)
    coupling: float = 0.055
    string_impedance_n_s_m: float = 0.65
    bridge: BridgeImpedance | None = None
    soundboard: ModalSoundboard | None = None
    sr: int = 48000
    loss_dc: float = .99965
    loss_hf: float = .989

    def __post_init__(self):
        base=max(float(self.struck_frequency),20.0)
        unison=[base*(2.0**(c/1200.0)) for c in self.unison_detune_cents]
        sympathetic=[base*float(r) for r in self.sympathetic_ratios if base*float(r) < .45*self.sr]
        self.frequencies=tuple(unison+sympathetic)
        self.struck_count=len(unison)
        self.lines=[]
        for f in self.frequencies:
            d=max(2.05,self.sr/max(f,1.0))
            self.lines.append((ThiranDelay(d,max(64,int(self.sr/10))),
                               FrequencyDependentLoss(self.loss_dc,self.loss_hf,.45)))
        self.bridge = self.bridge or BridgeImpedance(sr=self.sr)
        self.soundboard = self.soundboard or ModalSoundboard(sr=self.sr)
        self.last_string_rms=np.zeros(len(self.lines))
        self.last_bridge_energy_j=0.0
        self.last_soundboard_energy=0.0

    def render(self, dur: float, velocity: float=.8, return_stems: bool=False):
        n=max(1,int(round(float(dur)*self.sr)))
        y=np.zeros(n); stems=np.zeros((len(self.lines),n)) if return_stems else None
        # Hammer/contact pulse shared by the struck unison course.  The slight
        # asymmetry prevents perfectly coherent cancellation/locking.
        hit_n=max(2,int(round(.0035*self.sr)))
        hit=np.sin(np.linspace(0,math.pi,hit_n))*float(velocity)
        z=max(float(self.string_impedance_n_s_m),1e-5)
        c=float(np.clip(self.coupling,0.0,.25))
        for i in range(n):
            incoming=np.array([dl.read() for dl,_ in self.lines],float)
            if i<hit_n:
                for k in range(self.struck_count):
                    incoming[k]+=hit[i]*(1.0-.035*k)
            # Treat line states as reduced transverse velocities.  Their net
            # reaction force drives the physical bridge.
            string_force=-z*float(np.sum(incoming))
            board_rad,board_reaction_v=self.soundboard.step(-string_force)
            bridge_v=self.bridge.step(string_force)
            common_v=bridge_v+board_reaction_v
            out=np.empty_like(incoming)
            for k,((dl,loss),v) in enumerate(zip(self.lines,incoming)):
                # Velocity scattering against a common bridge.  Convex blending
                # keeps the reduced loop passive for the supported coupling range.
                nv=(1.0-c)*v + c*common_v
                nv=loss.process(nv)
                dl.write(nv); out[k]=nv
                if stems is not None: stems[k,i]=nv
            # Bridge + board radiation; direct string component retains attack.
            y[i]=0.18*float(np.sum(out[:self.struck_count])) + 0.82*board_rad
        if stems is not None:
            self.last_string_rms=np.sqrt(np.mean(stems*stems,axis=1)+1e-30)
        else:
            # inexpensive final-state proxy if stems were not requested
            self.last_string_rms=np.abs(np.array([dl.read() for dl,_ in self.lines]))
        self.last_bridge_energy_j=self.bridge.energy_j
        self.last_soundboard_energy=self.soundboard.modal_energy
        yn=_norm(y)
        return (yn,stems) if return_stems else yn
