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
    """Side hole with a smoothly variable opening [0,1], as an impedance magnitude at ``f_ref``.

    Open: the inertance of t_e + t_i (chimney with its matching and radiation lengths, inner length) on the bore
    of radius ``bore_radius_m``. Closed: the chimney t + t_m as a quarter-wave stub, |cot k(t + t_m)|, which
    vanishes at c / 4(t + t_m): a 4 cm bassoon chimney puts an anti-formant near 2 kHz. In between, the two
    admittances blend with openness^2. ``nonlinear``: the hole's quadratic loss 0.5 (|v| / c) Z_h at the hole
    velocity of the last step. ``series_length`` is the (negative) series length of the closed hole."""
    radius_m: float=.0035
    chimney_m: float=.004
    openness: float=0.0
    rho: float=1.2041
    c: float=343.0
    bore_radius_m: float=.0075
    f_ref: float=440.0
    nonlinear: bool=False
    v_hole: float=0.0
    # ponytail: real magnitudes at one frequency (f_ref) in a resistive junction; a frequency-dependent hole needs a reactive (filter) port

    @property
    def area(self): return math.pi*max(self.radius_m,1e-6)**2

    @property
    def series_length(self) -> float:
        from .physical.winds import series_length
        b,a=self.radius_m,max(self.bore_radius_m,self.radius_m*1.01)
        return series_length(b,a,self.chimney_m,open_=False)

    def impedance(self, sr: int) -> float:
        from .physical.winds import chimney_length, inner_length, matching_length
        o=float(np.clip(self.openness,0.0,1.0))
        b,a,t=self.radius_m,max(self.bore_radius_m,self.radius_m*1.01),self.chimney_m
        zh=self.rho*self.c/max(self.area,1e-12)
        k=2*math.pi*self.f_ref/self.c
        z_open=zh*k*(chimney_length(b,a,t)+inner_length(b,a))
        if self.nonlinear: z_open+=0.5*abs(self.v_hole)/self.c*zh
        z_closed=zh*abs(1.0/math.tan(max(k*(t+matching_length(b,a)),1e-9)))
        y=o*o/max(z_open,1e-3*zh)+(1.0-o*o)/max(z_closed,1e-3*zh)
        return 1.0/y


BLOWHOLE = dict(rb=0.0075, rth=0.003, rrh=0.0015, te_k=1.4, c=347.23, rho=1.1769)


def blowhole_coefficients(fs: float, rb: float = BLOWHOLE["rb"], rth: float = BLOWHOLE["rth"], rrh: float = BLOWHOLE["rrh"],
                          xi: float = 0.0) -> dict:
    """One-pole constants of a tone hole of radius ``rth`` and of a register vent of radius ``rrh`` on a bore of radius ``rb`` at
    sample rate ``fs``: scatter = -rth^2 / (rth^2 + 2 rb^2); thCoeff = (2 te fs - c) / (2 te fs + c), te = 1.4 rth;
    vent zeta = c + 2 pi rb^2 xi / rho, psi = 2 pi rb^2 (1.4 rrh) / (pi rrh^2), rhCoeff = (zeta - 2 fs psi) / (zeta + 2 fs psi),
    rhGain = -c / (zeta + 2 fs psi)."""
    c, rho, k = BLOWHOLE["c"], BLOWHOLE["rho"], BLOWHOLE["te_k"]
    te = k * rth
    zeta = c + 2.0 * math.pi * rb * rb * xi / rho
    psi = 2.0 * math.pi * rb * rb * (k * rrh) / (math.pi * rrh * rrh)
    return dict(scatter=-rth * rth / (rth * rth + 2.0 * rb * rb), th_coeff=(2.0 * te * fs - c) / (2.0 * te * fs + c),
                rh_coeff=(zeta - 2.0 * fs * psi) / (zeta + 2.0 * fs * psi), rh_gain=-c / (zeta + 2.0 * fs * psi))


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
    f_ref_hz: float=0.0
    nonlinear_holes: bool=False
    vent_position: float=0.0
    vent_radius_m: float=BLOWHOLE["rrh"]
    vent_open: float=0.0

    def __post_init__(self):
        n=len(self.hole_positions)
        if len(self.hole_radii_m)!=n or len(self.hole_chimneys_m)!=n:
            raise ValueError("tone-hole tuples must have equal length")
        pos=np.asarray(self.hole_positions,float)
        if n and (np.any(pos<=0) or np.any(pos>=1) or np.any(np.diff(pos)<=0)):
            raise ValueError("hole_positions must be strictly increasing inside (0,1)")
        self._vent_at=None
        if self.vent_position:
            if not 0.0<self.vent_position<1.0 or np.any(pos==self.vent_position):
                raise ValueError("vent_position must lie inside (0,1), away from the tone holes")
            self._vent_at=int(np.searchsorted(pos,self.vent_position))
            pos=np.insert(pos,self._vent_at,self.vent_position)
        radii=list(self.hole_radii_m); chim=list(self.hole_chimneys_m)
        if self._vent_at is not None:
            radii.insert(self._vent_at,self.vent_radius_m); chim.insert(self._vent_at,BLOWHOLE["te_k"]*self.vent_radius_m)
        rho,c=1.2041,343.0
        self.zb=rho*c/(math.pi*self.bore_radius_m**2)
        boundaries=np.concatenate(([0.0],pos,[1.0]))
        lengths=np.diff(boundaries)*self.length_m
        fref=self.f_ref_hz or c/(4.0*self.length_m)
        self.holes=[DynamicToneHole(r,ch,0.0,bore_radius_m=self.bore_radius_m,f_ref=fref,nonlinear=self.nonlinear_holes)
                    for r,ch in zip(radii,chim)]
        self.vent=self.holes[self._vent_at] if self._vent_at is not None else None
        self.finger_holes=[h for h in self.holes if h is not self.vent]
        if self.vent is not None: self.vent.openness=float(np.clip(self.vent_open,0,1))
        for j,h in enumerate(self.holes):   # each closed hole shortens the bore by its series length (negative)
            lengths[j]+=0.5*h.series_length; lengths[j+1]+=0.5*h.series_length
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
        self.bell=BellRadiation(self.bell_cutoff_hz,self.sr,1.0)

    def set_fingering(self, openings):
        vals=np.asarray(openings,float)
        if vals.shape!=(len(self.finger_holes),): raise ValueError("fingering length mismatch")
        for h,o in zip(self.finger_holes,vals): h.openness=float(np.clip(o,0,1))

    def set_vent(self, openness: float):
        if self.vent is None: raise ValueError("this bore has no vent (vent_position)")
        self.vent.openness=float(np.clip(openness,0,1))

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
            if h.nonlinear: h.v_hole=float(out[2]+inc[2])/(h.rho*h.c)   # junction pressure / rho c
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
            if dynamic and f.shape!=(n,len(self.finger_holes)): raise ValueError("dynamic fingering shape mismatch")
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
        if fingering is None: fingering=np.zeros(len(self.bore.finger_holes))
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
        if fingering is None: fingering=np.zeros(len(self.bore.finger_holes))
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

    Useful for piano unisons, mandolin courses and sympathetic strings. Each string's reflection at the
    bridge loses the bridge motion: v_k <- v_k - coupling * bridge, bridge = low-passed mean of the
    incoming waves. The in-phase mode moves the bridge and decays fast; the antiphase mode leaves it
    still and rings on (two-stage decay).
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

    @staticmethod
    def weinreich_coupling(freq: float, n: int = 3, y_bridge: float | None = None) -> float:
        """Coupling for n unison strings of the piano key at ``freq`` from the bridge admittance: the in-phase mode
        loses eta = w sqrt(T0 mu) Y / pi per second on each string, i.e. n eta in all, and a pass loses the fraction
        c per period, so c = 1 - exp(-n eta / f)."""
        from .physical.bank_piano import Y_BRIDGE, _row
        r = _row(69.0 + 12.0 * math.log2(max(float(freq), 1.0) / 440.0))
        mu = r["rho"] * math.pi * r["d"] ** 2 / 4.0
        eta = 2.0 * math.pi * float(freq) * math.sqrt(r["T0"] * mu) * (Y_BRIDGE if y_bridge is None else float(y_bridge)) / math.pi
        return 1.0 - math.exp(-int(n) * eta / float(freq))

    @classmethod
    def from_bridge(cls, freq: float, detune_cents=(-0.9, 0.0, 0.7), sr: int = 48000, **kw):
        """A course of unison strings at ``freq`` whose coupling is the bridge's, not a free constant."""
        f = tuple(float(freq) * 2.0 ** (c / 1200.0) for c in detune_cents)
        return cls(frequencies=f, sr=sr, coupling=cls.weinreich_coupling(freq, len(f)), **kw)

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
                coupled=x-self.coupling*self.bridge
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


def loaded_junction(v_plus, R, Rb=0.0, f_d=0.0):
    """N strings of wave impedances ``R`` meeting at a bridge of real load ``Rb`` driven by the force ``f_d``: the common
    velocity v_J = (2 sum R_i v_i+ - f_d) / (sum R_i + Rb), the reflections v_i- = v_J - v_i+. ``v_plus`` is (N,) or
    (n, N); returns (v_minus, v_J). Rb -> inf is the rigid end, Rb = 0 a free one; |I - 2 R 1^T / (sum R + Rb)| <= 1."""
    v = np.asarray(v_plus, float)
    R = np.asarray(R, float)
    vj = (2.0 * (v * R).sum(axis=-1) - f_d) / (R.sum() + float(Rb))
    return vj[..., None] - v, vj


def passive_bridge(freqs, bws, g=0.9, method=1, sr=48000):
    """Bridge reflectance rho(z) = B/A of strings on a body with resonances at ``freqs`` Hz of bandwidth ``bws`` Hz,
    passive by construction (|rho| <= g <= 1 on the unit circle). Method 1: A = prod (1 - 2 R_i cos(theta_i) z^-1 +
    R_i^2 z^-2), rho = g flip(A) / A; the admittance is then (A - g flip(A)) / (A + g flip(A)), real positive.
    Method 2: admittance Gamma = (1 - 0.995 z^-1) sum 1 / (1 + a1_i z^-1 + a2_i z^-2), its phase inside +-90 degrees,
    rho = (1 - Gamma) / (1 + Gamma). Returns (b, a) of rho in direct form: with resonances far below ``sr`` (a few Hz at
    48 kHz) the coefficients lose their precision, so give a low ``sr`` or cascade the sections."""
    T = 1.0 / float(sr)
    secs = []
    for F, B in zip(np.atleast_1d(freqs), np.atleast_1d(bws)):
        r, th = math.exp(-math.pi * float(B) * T), 2.0 * math.pi * float(F) * T
        secs.append(np.array([1.0, -2.0 * r * math.cos(th), r * r]))
    if int(method) == 1:
        A = np.array([1.0])
        for s in secs:
            A = np.convolve(A, s)
        return g * A[::-1], A
    if int(method) != 2:
        raise ValueError("method must be 1 or 2")
    den = np.array([1.0])
    for s in secs:
        den = np.convolve(den, s)
    num = np.zeros(2 * len(secs) - 1)
    for k in range(len(secs)):
        rest = np.array([1.0])
        for j, t in enumerate(secs):
            if j != k:
                rest = np.convolve(rest, t)
        num += rest
    num = np.concatenate((np.convolve(num, [1.0, -0.995]), [0.0]))
    return den - num, den + num


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
    unison_detune_cents: tuple[float,...] = (-0.9, 0.0, 0.7)  # Kirk 1959, JASA 31(12):1644, 1.6 cent mean max spread
    sympathetic_ratios: tuple[float,...] = (0.5, 1.5, 2.0, 2.5, 3.0)
    coupling: float = 0.055
    string_impedance_n_s_m: float = 0.65
    bridge: BridgeImpedance | None = None
    soundboard: ModalSoundboard | None = None
    sr: int = 48000
    loss_dc: float = .99965
    loss_hf: float = .989
    junction: str = "mix"          # mix | loaded: the bridge as the loaded N-port junction (common velocity v_J)
    bridge_load: float = 20.0      # loaded: real load of the bridge in multiples of the string impedance

    def __post_init__(self):
        if self.junction not in ("mix","loaded"):
            raise ValueError("junction must be mix or loaded")
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

    def _scatter(self, incoming, z, c):
        """One sample at the bridge. The strings' summed force z * sum(v) drives the bridge and the board; each
        string's reflection loses the common velocity u they take: v_k <- v_k - c u. The in-phase mode moves
        the bridge and decays fast, the antiphase mode leaves it still and rings on."""
        if self.junction=="loaded":
            _,u=loaded_junction(incoming,np.full(len(incoming),z),self.bridge_load*z)
            board_rad,_=self.soundboard.step(self.bridge_load*z*float(u))
            nv=np.array([loss.process(float(v)-float(u)) for (dl,loss),v in zip(self.lines,incoming)])
            for (dl,_),x in zip(self.lines,nv):
                dl.write(float(x))
            return nv,board_rad
        force=z*float(np.sum(incoming))
        board_rad,board_v=self.soundboard.step(force)
        u=self.bridge.step(force)+board_v
        out=np.empty_like(incoming)
        for k,((dl,loss),v) in enumerate(zip(self.lines,incoming)):
            nv=loss.process(v-c*u)
            dl.write(nv); out[k]=nv
        return out,board_rad

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
            out,board_rad=self._scatter(incoming,z,c)
            if stems is not None: stems[:,i]=out
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
