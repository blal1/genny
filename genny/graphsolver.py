"""Sample-by-sample reduced physical graph solvers.

These solvers intentionally favor a common energy/port formulation over the
specialized, highly tuned renderers in :mod:`genny.physical`.  They are useful
for interactive coupling and as a common numerical core.  High-level models
may blend them with the specialized renderer while migration is in progress.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .ports import PhysicalPort


@dataclass
class FractionalDelay:
    """Circular fractional delay with selectable interpolation.

    ``linear`` is inexpensive and robust. ``lagrange3`` uses a four-point
    cubic Lagrange interpolator and is substantially more accurate for tuning
    and moving-delay applications.  The read pointer is defined relative to
    the *next* write location, so an integer delay of one sample returns the
    most recently written sample.
    """
    delay: float
    max_delay: int
    method: str = "linear"

    def __post_init__(self):
        self.max_delay = max(8, int(self.max_delay))
        self.buf = np.zeros(self.max_delay + 8, dtype=float)
        self.w = 0
        if self.method not in {"linear", "lagrange3"}:
            raise ValueError("FractionalDelay method must be 'linear' or 'lagrange3'")

    def _sample(self, i: int) -> float:
        return float(self.buf[i % len(self.buf)])

    def read(self, delay: float | None = None) -> float:
        d = float(self.delay if delay is None else delay)
        d = min(max(d, 1.0), self.max_delay - 3.0)
        pos = self.w - d
        i0 = math.floor(pos)
        mu = pos - i0
        if self.method == "linear":
            a = self._sample(i0)
            b = self._sample(i0 + 1)
            return (1.0-mu)*a + mu*b
        # Four-point cubic Lagrange interpolation using samples at -1,0,1,2.
        xm1, x0, x1, x2 = (self._sample(i0-1), self._sample(i0),
                           self._sample(i0+1), self._sample(i0+2))
        c_m1 = -mu*(mu-1.0)*(mu-2.0)/6.0
        c_0  =  (mu+1.0)*(mu-1.0)*(mu-2.0)/2.0
        c_1  = -(mu+1.0)*mu*(mu-2.0)/2.0
        c_2  =  (mu+1.0)*mu*(mu-1.0)/6.0
        return c_m1*xm1 + c_0*x0 + c_1*x1 + c_2*x2

    def write(self, x: float):
        self.buf[self.w % len(self.buf)] = float(x)
        self.w = (self.w + 1) % len(self.buf)


@dataclass
class ThiranDelay:
    """Integer delay followed by a first-order Thiran allpass.

    This preserves magnitude exactly while approximating the fractional group
    delay maximally flat at DC.  It is useful inside lossless/passive-ish
    waveguide loops where an FIR fractional interpolator would alter amplitude.
    """
    delay: float
    max_delay: int

    def __post_init__(self):
        self.max_delay=max(8,int(self.max_delay))
        self.buf=np.zeros(self.max_delay+8,dtype=float)
        self.w=0; self.x1=0.0; self.y1=0.0

    def read(self, delay: float | None=None) -> float:
        d=float(self.delay if delay is None else delay)
        d=min(max(d,1.0),self.max_delay-3.0)
        integer=max(1,int(math.floor(d)))
        frac=d-integer
        x=float(self.buf[(self.w-integer)%len(self.buf)])
        if frac < 1e-8:
            self.x1=x; self.y1=x
            return x
        a=(1.0-frac)/(1.0+frac)
        y=a*x + self.x1 - a*self.y1
        self.x1=x; self.y1=y
        return float(y)

    def write(self,x:float):
        self.buf[self.w%len(self.buf)]=float(x)
        self.w=(self.w+1)%len(self.buf)


@dataclass
class FrequencyDependentLoss:
    """Passive one-pole waveguide loss filter.

    ``dc_gain`` and ``hf_gain`` are constrained to [0,1]. The filter blends
    the current sample with a one-pole low-passed state, allowing high
    frequencies to decay faster than lows while never intentionally amplifying.
    """
    dc_gain: float = .9995
    hf_gain: float = .985
    smoothing: float = .35

    def __post_init__(self):
        self.dc_gain=float(np.clip(self.dc_gain,0.0,1.0))
        self.hf_gain=float(np.clip(self.hf_gain,0.0,self.dc_gain))
        self.smoothing=float(np.clip(self.smoothing,0.0,.9999)); self.state=0.0

    def process(self,x:float)->float:
        self.state=self.smoothing*self.state+(1.0-self.smoothing)*float(x)
        hi=float(x)-self.state
        return self.dc_gain*self.state + self.hf_gain*hi


@dataclass
class FirstOrderAllpass:
    """Stable first-order allpass used as a lightweight dispersion section."""
    coefficient: float=.0
    def __post_init__(self):
        self.coefficient=float(np.clip(self.coefficient,-.999,.999)); self.x1=0.; self.y1=0.
    def process(self,x:float)->float:
        a=self.coefficient
        y=a*float(x)+self.x1-a*self.y1
        self.x1=float(x); self.y1=y
        return float(y)


@dataclass
class StiffStringDispersion:
    """Small cascade of stable allpasses approximating stiff-string dispersion."""
    amount: float=.0
    sections: int=3
    def __post_init__(self):
        a=float(np.clip(self.amount,0.0,1.0))
        self.filters=[FirstOrderAllpass(.04 + .18*a*(i+1)/max(1,self.sections)) for i in range(max(1,int(self.sections)))]
    def process(self,x:float)->float:
        for f in self.filters: x=f.process(x)
        return float(x)


@dataclass
class StickSlipHysteresis:
    """Stateful static/kinetic friction law for bowed and creaking contacts."""
    static_mu: float=.85
    kinetic_mu: float=.22
    capture_velocity: float=.012
    release_force: float=1.0
    state: str="stick"
    def step(self,relative_velocity:float,normal_force:float)->tuple[float,str]:
        rv=float(relative_velocity); n=max(0.0,float(normal_force))
        if self.state=="stick":
            required=abs(rv)*max(self.release_force,1e-9)*80.0
            if required > self.static_mu*n or abs(rv) > 4*self.capture_velocity:
                self.state="slip"
        else:
            if abs(rv) < self.capture_velocity:
                self.state="stick"
        if self.state=="stick":
            f=np.clip(rv*80.0,-self.static_mu*n,self.static_mu*n)
        else:
            mu=self.kinetic_mu + (self.static_mu-self.kinetic_mu)*math.exp(-abs(rv)/max(self.capture_velocity,1e-6))
            f=mu*n*math.tanh(rv/max(self.capture_velocity,.001))
        return float(f),self.state


@dataclass
class MultiportScatteringJunction:
    """Lossless pressure-wave junction for N branches with impedances Z_i."""
    impedances: np.ndarray
    def __post_init__(self):
        self.impedances=np.asarray(self.impedances,float)
        if self.impedances.ndim!=1 or len(self.impedances)<2 or np.any(self.impedances<=0):
            raise ValueError("positive 1-D impedances are required")
    def scatter(self,incoming)->np.ndarray:
        p=np.asarray(incoming,float)
        if p.shape!=(len(self.impedances),): raise ValueError("incoming shape mismatch")
        adm=1.0/self.impedances
        pj=2.0*np.sum(p*adm)/np.sum(adm)
        return pj-p


@dataclass
class ToneHole:
    """Reduced side-hole scattering element with continuously variable opening."""
    bore_impedance: float
    hole_impedance: float
    openness: float=1.0
    def reflection(self)->float:
        o=float(np.clip(self.openness,0,1))
        zh=self.hole_impedance/max(o,1e-5)
        return float(np.clip((zh-self.bore_impedance)/(zh+self.bore_impedance),-1,1))
    def process(self,incoming:float)->tuple[float,float]:
        r=self.reflection(); reflected=r*float(incoming)
        radiated=math.sqrt(max(0.0,1-r*r))*float(incoming)
        return reflected,radiated


@dataclass
class BellRadiation:
    """Simple passive frequency-dependent bell/open-end approximation."""
    cutoff_hz: float=900.0
    sr: int=48000
    openness: float=1.0
    def __post_init__(self):
        self.state=0.0
        self.a=math.exp(-2*math.pi*max(20.,self.cutoff_hz)/self.sr)
    def process(self,x:float)->tuple[float,float]:
        self.state=self.a*self.state+(1-self.a)*float(x)
        low=self.state; high=float(x)-low
        o=float(np.clip(self.openness,0,1))
        radiated=o*(.25*low+.9*high)
        reflected=-(1.0-o*.85)*low - .08*high
        return float(reflected),float(radiated)


@dataclass
class JetDelay:
    """Hydrodynamic jet delay tau=L/U_conv with smooth saturation."""
    length_m: float=.01
    convection_ratio: float=.4
    max_delay_s: float=.02
    sr: int=48000
    def samples(self,jet_velocity_m_s:float)->float:
        u=max(.05,float(jet_velocity_m_s)*max(.05,self.convection_ratio))
        tau=min(self.max_delay_s,max(1.0/self.sr,self.length_m/u))
        return tau*self.sr


@dataclass
class CommonSolveResult:
    audio: np.ndarray
    drive: PhysicalPort
    response: PhysicalPort
    metadata: dict


def _norm(x: np.ndarray, peak=.9):
    x = np.asarray(x, float)
    m = float(np.max(np.abs(x))) if len(x) else 0.0
    return x if m <= 1e-12 else x * (peak/m)


def hammer_string(f0: float, dur: float, velocity=.8, *, sr=48000,
                  felt=.15, loss=.9985, coupling=.85) -> CommonSolveResult:
    n = max(1, int(round(dur*sr)))
    delay = max(2.0, sr/max(float(f0), 1.0))
    dl = FractionalDelay(delay, int(sr/20)+8, method='lagrange3')
    y = np.zeros(n); force=np.zeros(n); hv=np.zeros(n); sv=np.zeros(n)
    key = float(np.clip(49.0 + 12.0*math.log2(max(f0,1e-6)/440.0),1,88))
    p = 3.7 + .015*key
    k = 183.0*math.exp(.045*key) * 1e-4 * (1.0 + 2.0*felt)
    contact = max(2, min(n, int(sr*min(.012,max(.0004,.006/max(velocity,.08))))))
    z = .08*math.sqrt(max(f0,20.0)/440.0)
    prev=0.0
    losses=FrequencyDependentLoss(dc_gain=min(1.0, loss+.0008), hf_gain=max(0.0, loss-.012), smoothing=.42)
    dispersion=StiffStringDispersion(amount=min(1.0, .15 + 1.8*felt), sections=3)
    for i in range(n):
        ret = dl.read()
        if i < contact:
            q = math.sin(math.pi*i/max(contact-1,1))
            f = k*(max(q,0.0)**p)*velocity
            vh = velocity*math.cos(.5*math.pi*i/max(contact-1,1))
        else:
            f=0.0; vh=0.0
        vs = ret/max(z,1e-6)
        inject = coupling*f + (1.0-coupling)*z*(vh-vs)
        # simple one-pole HF loss in loop
        loop = losses.process(.5*ret + .5*prev) + inject
        loop = dispersion.process(loop)
        prev=ret
        dl.write(loop)
        y[i]=ret; force[i]=f; hv[i]=vh; sv[i]=vs
    return CommonSolveResult(_norm(y), PhysicalPort.mechanical(force,hv,'hammer'),
                             PhysicalPort.mechanical(-force,sv,'string'),
                             {'solver':'common-graph','delay_samples':delay,'felt_exponent':p})


def bow_string(f0: float, dur: float, pressure=.85, bow_velocity=.2, *,
               sr=48000, loss=.9992, coupling=.9) -> CommonSolveResult:
    n=max(1,int(round(dur*sr))); delay=max(2.0,sr/max(f0,1.0))
    dl=FractionalDelay(delay,int(sr/20)+8,method='lagrange3')
    y=np.zeros(n); farr=np.zeros(n); vbarr=np.zeros(n); vsarr=np.zeros(n)
    z=.11*math.sqrt(max(f0,20.0)/440.0)
    friction=StickSlipHysteresis(static_mu=.88, kinetic_mu=.2, capture_velocity=.014, release_force=1.0)
    losses=FrequencyDependentLoss(dc_gain=min(1.0,loss+.0005), hf_gain=max(0.0,loss-.008), smoothing=.5)
    states=[]
    for i in range(n):
        ret=dl.read(); vs=ret/max(z,1e-6)
        vb=bow_velocity*min(1.0,i/max(1,int(.05*sr)))
        rel=vb-vs
        f,state=friction.step(rel,1.5*max(pressure,0.0))
        if i % max(1,sr//1000)==0: states.append(state)
        outgoing=losses.process(ret) + coupling*f
        dl.write(outgoing)
        y[i]=ret; farr[i]=f; vbarr[i]=vb; vsarr[i]=vs
    return CommonSolveResult(_norm(y),PhysicalPort.mechanical(farr,vbarr,'bow'),
                             PhysicalPort.mechanical(-farr,vsarr,'string'),
                             {'solver':'common-graph','delay_samples':delay,'friction_states':states})


def reed_bore(f0: float, dur: float, breath=.6, noise=.02, *, sr=48000,
              bore_radius=.007, loss=.997, seed=0) -> CommonSolveResult:
    n=max(1,int(round(dur*sr))); delay=max(2.0,sr/max(f0,1.0))
    dl=ThiranDelay(delay,int(sr/20)+8)
    rng=np.random.default_rng(seed)
    y=np.zeros(n); mouth=np.zeros(n); flow=np.zeros(n); borep=np.zeros(n)
    rho=1.2041; c=343.; area=math.pi*bore_radius*bore_radius
    z=rho*c/max(area,1e-9)
    pm=800.0+4200.0*np.clip(breath,0,1.5)
    reed_k=3.5e-8
    losses=FrequencyDependentLoss(dc_gain=min(1.0,loss+.001),hf_gain=max(0.0,loss-.018),smoothing=.55)
    # Reduced side-hole and bell termination. Their openings can later be time-varying.
    hole=ToneHole(z, max(z*.18,1.0), openness=.35)
    bell=BellRadiation(cutoff_hz=max(500.,2.8*f0),sr=sr,openness=1.0)
    rad=np.zeros(n)
    for i in range(n):
        incoming=dl.read(); pb=incoming
        dp=max(pm-pb,0.0)
        opening=max(0.0,1.0-pb/max(pm*1.25,1.0))
        u=reed_k*opening*math.sqrt(2.0*dp/rho)
        if noise: u *= 1.0 + noise*rng.standard_normal()
        # pressure wave generated by volume flow; open-end sign inversion
        wave=0.5*z*u
        href,hrad=hole.process(incoming)
        bref,brad=bell.process(incoming-href)
        out=wave + losses.process(href+bref)
        dl.write(out)
        y[i]=pb + .35*(hrad+brad); rad[i]=hrad+brad; mouth[i]=pm; flow[i]=u; borep[i]=pb
    audio=_norm(y)
    return CommonSolveResult(audio,PhysicalPort.acoustic(mouth,flow,'reed'),
                             PhysicalPort.acoustic(borep,-flow,'bore'),
                             {'solver':'common-graph-v11','impedance':z,'delay_samples':delay,'tonehole_reflection':hole.reflection()})

class CommonPhysicalSolver:
    """Facade over the first fully shared sample-by-sample interaction solvers."""
    hammer_string = staticmethod(hammer_string)
    bow_string = staticmethod(bow_string)
    reed_bore = staticmethod(reed_bore)

def lip_bore(f0: float, dur: float, pressure=.55, lip_tension=.5, *, sr=48000,
             bore_radius=.006, loss=.998, seed=0) -> CommonSolveResult:
    n=max(1,int(round(dur*sr))); delay=max(2.0,sr/max(f0,1.0))
    dl=ThiranDelay(delay,int(sr/20)+8)
    y=np.zeros(n); pmarr=np.zeros(n); flow=np.zeros(n); parr=np.zeros(n)
    rho=1.2041; c=343.; area=math.pi*bore_radius*bore_radius; z=rho*c/max(area,1e-9)
    pm=2500.0+8500.0*np.clip(pressure,0,1.5)
    losses=FrequencyDependentLoss(dc_gain=min(1.0,loss+.001),hf_gain=max(0.0,loss-.012),smoothing=.48)
    bell=BellRadiation(cutoff_hz=max(350.,2.2*f0),sr=sr,openness=1.0)
    lip_f=max(20.0,f0*(.72+.55*np.clip(lip_tension,0,1)))
    phase=0.0; opening=2.0e-5
    for i in range(n):
        incoming=dl.read(); dp=max(pm-incoming,0.0)
        phase += 2*math.pi*lip_f/sr
        aperture=max(0.0, opening*(.45+.55*math.sin(phase)))
        u=aperture*math.sqrt(2.0*dp/rho)
        bref,brad=bell.process(incoming)
        out=.5*z*u + losses.process(bref)
        dl.write(out)
        y[i]=incoming + .4*brad; pmarr[i]=pm; flow[i]=u; parr[i]=incoming
    return CommonSolveResult(_norm(y), PhysicalPort.acoustic(pmarr,flow,'lips'),
                             PhysicalPort.acoustic(parr,-flow,'brass_bore'),
                             {'solver':'common-graph-v11','impedance':z,'delay_samples':delay,'bell_cutoff_hz':bell.cutoff_hz})


def syrinx_trachea(freq_pair, dur: float, pressure_pa=3000., *, sr=48000,
                    trachea_length=.07, trachea_radius=.0035) -> CommonSolveResult:
    f1,f2 = freq_pair if isinstance(freq_pair,tuple) else (freq_pair,freq_pair*1.01)
    n=max(1,int(round(dur*sr)))
    # round-trip tube delay from physical length, with a floor allowing small-bird models
    delay=max(2.0, 2.0*trachea_length/343.0*sr)
    dl=ThiranDelay(delay,int(sr*.02)+16)
    rho=1.2041; c=343.; area=math.pi*trachea_radius*trachea_radius; z=rho*c/max(area,1e-9)
    y=np.zeros(n); pmarr=np.zeros(n); flow=np.zeros(n); parr=np.zeros(n)
    ph1=ph2=0.0
    losses=FrequencyDependentLoss(dc_gain=.997,hf_gain=.982,smoothing=.5)
    for i in range(n):
        incoming=dl.read(); ramp=min(1.0,i/max(1,int(.02*sr))); pm=pressure_pa*ramp
        ph1 += 2*math.pi*f1/sr; ph2 += 2*math.pi*f2/sr
        membrane=.5*(math.sin(ph1)+math.sin(ph2))
        opening=max(0.0, .5+.5*membrane)
        dp=max(pm-incoming,0.0)
        u=1.2e-8*opening*math.sqrt(2.0*dp/rho)
        out=.5*z*u - losses.process(incoming)
        dl.write(out)
        y[i]=incoming; pmarr[i]=pm; flow[i]=u; parr[i]=incoming
    return CommonSolveResult(_norm(y),PhysicalPort.acoustic(pmarr,flow,'syrinx'),
                             PhysicalPort.acoustic(parr,-flow,'trachea'),
                             {'solver':'common-graph-v11','impedance':z,'delay_samples':delay})

CommonPhysicalSolver.lip_bore = staticmethod(lip_bore)
CommonPhysicalSolver.syrinx_trachea = staticmethod(syrinx_trachea)


def jet_bore(f0: float, dur: float, jet_velocity=18.0, *, sr=48000,
             bore_radius=.006, jet_length=.009, loss=.996) -> CommonSolveResult:
    """Reduced flute/edge-tone loop using a hydrodynamic jet delay and bore."""
    n=max(1,int(round(dur*sr))); bore_delay=max(2.0,sr/max(f0,1.0))
    bore=ThiranDelay(bore_delay,int(sr/20)+16)
    jet_model=JetDelay(jet_length,.4,.03,sr); jd=jet_model.samples(jet_velocity)
    jet=FractionalDelay(jd,int(sr*.04)+16,method='lagrange3')
    rho=1.2041; c=343.; area=math.pi*bore_radius*bore_radius; z=rho*c/max(area,1e-9)
    losses=FrequencyDependentLoss(min(1.0,loss+.001),max(0.0,loss-.015),.55)
    bell=BellRadiation(max(500.,2.5*f0),sr,1.0)
    y=np.zeros(n); pdrive=np.zeros(n); flow=np.zeros(n); pbore=np.zeros(n)
    drive=max(0.1,float(jet_velocity));
    for i in range(n):
        inc=bore.read(); delayed=jet.read();
        # edge interaction: bounded odd non-linearity driven by delayed jet minus bore pressure
        q=math.tanh((delayed-inc/max(z,1e-9))*4.0)
        u=area*.018*drive*q
        bref,brad=bell.process(inc)
        outgoing=.5*z*u + losses.process(bref)
        bore.write(outgoing); jet.write(drive*.025 + .2*u/area)
        y[i]=inc+.45*brad; pdrive[i]=.5*rho*drive*drive; flow[i]=u; pbore[i]=inc
    return CommonSolveResult(_norm(y),PhysicalPort.acoustic(pdrive,flow,'jet'),
                             PhysicalPort.acoustic(pbore,-flow,'bore'),
                             {'solver':'common-graph-v11','bore_delay_samples':bore_delay,'jet_delay_samples':jd})

CommonPhysicalSolver.jet_bore = staticmethod(jet_bore)

@dataclass
class WaveguideBranch:
    """Passive bidirectional branch used by :class:`WaveguideNetwork`."""
    impedance: float
    delay_samples: float
    max_delay: int
    loss_dc: float = .999
    loss_hf: float = .985
    method: str = "thiran"

    def __post_init__(self):
        self.impedance=float(self.impedance)
        if self.impedance<=0: raise ValueError("impedance must be positive")
        cls=ThiranDelay if self.method=="thiran" else FractionalDelay
        self.delay=cls(self.delay_samples,self.max_delay) if cls is ThiranDelay else cls(self.delay_samples,self.max_delay,"lagrange3")
        self.loss=FrequencyDependentLoss(self.loss_dc,self.loss_hf,.5)

    def incoming(self)->float:
        return self.delay.read()

    def send(self,x:float):
        self.delay.write(self.loss.process(float(x)))


class WaveguideNetwork:
    """Small passive multiport waveguide network.

    Each branch provides an incoming pressure wave.  A lossless scattering
    junction computes outgoing waves from branch impedances, then each branch
    propagates its outgoing wave through its own passive delay/loss section.
    """
    def __init__(self, branches):
        self.branches=list(branches)
        if len(self.branches)<2: raise ValueError("at least two branches required")
        self.junction=MultiportScatteringJunction(np.array([b.impedance for b in self.branches],float))

    def step(self, excitation=None):
        inc=np.array([b.incoming() for b in self.branches],float)
        if excitation is not None:
            ex=np.asarray(excitation,float)
            if ex.shape!=inc.shape: raise ValueError("excitation shape mismatch")
            inc=inc+ex
        out=self.junction.scatter(inc)
        for b,x in zip(self.branches,out): b.send(x)
        return inc,out

    def render_impulse(self,n:int,drive_branch:int=0,amplitude:float=1.0):
        y=np.zeros(int(n),float)
        for i in range(int(n)):
            ex=np.zeros(len(self.branches),float)
            if i==0: ex[int(drive_branch)]=float(amplitude)
            inc,out=self.step(ex)
            y[i]=inc[int(drive_branch)]
        return y
