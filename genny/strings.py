"""Shared reduced-order physical string core (v0.18).

The module collects the physical quantities that were previously implicit in
several instrument-specific waveguides.  It is deliberately a realtime
reduced-order model, not a finite-element string solver.
"""
from __future__ import annotations
from dataclasses import dataclass
import functools
import math
import numba
import numpy as np

from .graphsolver import ThiranDelay, FrequencyDependentLoss, StiffStringDispersion


def rauhala_dispersion(f0: float, B: float, sr: int, M: int = 4) -> tuple:
    """Coefficients of M identical first-order allpasses (a + z^-1)/(1 + a z^-1) that make a waveguide
    loop's partials follow f_k = k f0 sqrt(1 + B k^2): Rauhala & Valimaki's closed form (DAFx-06 eqs 3-8,
    Table 2), fitted at 44.1 kHz, so it is evaluated at the same normalised frequency f0 * 44100 / sr.
    Within ~7 cents over 15 partials for guitar and piano values up to B 3e-4. M drops until the
    cascade's DC delay M*D fits in half a period."""
    if B < 1e-7 or f0 <= 0:
        return ()
    ikey = 12.0 * math.log2(f0 * 44100.0 / sr / 27.5) + 1.0
    # ponytail: B held at 4e-4, the top of the piano data the formula was fitted on; the wound bass strings'
    # geometry gives up to 3e-3 on E1, which would put the low notes 20-30 cents sharp to the ear
    lb = math.log(min(B, 4e-4))
    kd = math.exp(-0.00179 * lb * lb - 0.0233 * lb - 2.93)
    while M > 1:
        cd = math.exp((0.0126 * math.log(M) + 0.0606) * lb - 0.00825 * math.log(M) + 1.97)
        if M * math.exp(cd - ikey * kd) < 0.5 * sr / f0:
            break
        M -= 1
    cd = math.exp((0.0126 * math.log(M) + 0.0606) * lb - 0.00825 * math.log(M) + 1.97)
    D = math.exp(cd - ikey * kd)
    return ((1.0 - D) / (1.0 + D),) * M


@dataclass(frozen=True)
class StringMaterial:
    name: str
    density_kg_m3: float
    young_pa: float
    loss_t60_s: float


STRING_MATERIALS = {
    "steel": StringMaterial("steel", 7850.0, 2.0e11, 4.8),
    "music_wire": StringMaterial("music_wire", 7850.0, 2.05e11, 5.5),
    "nylon": StringMaterial("nylon", 1140.0, 2.6e9, 2.8),
    "gut": StringMaterial("gut", 1300.0, 1.5e9, 2.3),
    "bronze": StringMaterial("bronze", 8800.0, 1.05e11, 3.8),
    "nickel": StringMaterial("nickel", 8900.0, 2.0e11, 4.2),
    "copper": StringMaterial("copper", 8960.0, 1.10e11, 3.5),
    "phosphor_bronze": StringMaterial("phosphor_bronze", 8860.0, 1.10e11, 3.9),
}


@dataclass
class StringPhysicalProperties:
    """Physical description of an idealized cylindrical musical string.

    Either ``tension_n`` or ``frequency_hz`` may be specified.  If tension is
    omitted it is inferred from the ideal-string relation
    ``f0=(1/(2L))*sqrt(T/mu)``.
    """
    length_m: float = 0.65
    radius_m: float = 0.00035
    material: str | StringMaterial = "steel"
    frequency_hz: float = 440.0
    tension_n: float | None = None
    t60_s: float | None = None
    polarization_split_cents: float = 0.32

    def __post_init__(self):
        self.length_m = max(0.02, float(self.length_m))
        self.radius_m = max(1e-5, float(self.radius_m))
        if isinstance(self.material, str):
            if self.material not in STRING_MATERIALS:
                raise ValueError(f"unknown string material: {self.material!r}")
            self.material = STRING_MATERIALS[self.material]
        self.frequency_hz = max(1.0, float(self.frequency_hz))
        self.t60_s = float(self.material.loss_t60_s if self.t60_s is None else max(0.03, self.t60_s))
        if self.tension_n is None:
            self.tension_n = (2.0 * self.length_m * self.frequency_hz) ** 2 * self.linear_density_kg_m
        else:
            self.tension_n = max(1e-4, float(self.tension_n))
            self.frequency_hz = self.ideal_frequency_hz

    @property
    def area_m2(self) -> float:
        return math.pi * self.radius_m**2

    @property
    def linear_density_kg_m(self) -> float:
        return self.material.density_kg_m3 * self.area_m2

    @property
    def second_moment_m4(self) -> float:
        # Circular section I = pi*r^4/4.
        return math.pi * self.radius_m**4 / 4.0

    @property
    def flexural_rigidity_n_m2(self) -> float:
        return self.material.young_pa * self.second_moment_m4

    @property
    def ideal_frequency_hz(self) -> float:
        return (1.0 / (2.0 * self.length_m)) * math.sqrt(self.tension_n / self.linear_density_kg_m)

    @property
    def characteristic_impedance_n_s_m(self) -> float:
        return math.sqrt(self.tension_n * self.linear_density_kg_m)

    @property
    def wave_speed_m_s(self) -> float:
        return math.sqrt(self.tension_n / self.linear_density_kg_m)

    @property
    def inharmonicity_B(self) -> float:
        # Common stiff-string approximation f_n ≈ n f0 sqrt(1 + B n²).
        return (math.pi**2 * self.flexural_rigidity_n_m2) / max(self.tension_n * self.length_m**2, 1e-18)

    @property
    def longitudinal_speed_m_s(self) -> float:
        return math.sqrt(self.material.young_pa / self.material.density_kg_m3)

    @property
    def longitudinal_frequency_hz(self) -> float:
        return self.longitudinal_speed_m_s / (2.0 * self.length_m)

    def partial_frequency(self, n: int) -> float:
        n = max(1, int(n))
        return n * self.frequency_hz * math.sqrt(1.0 + self.inharmonicity_B * n*n)

    @classmethod
    def from_frequency(cls, frequency_hz: float, *, length_m: float | None = None,
                       radius_m: float | None = None, material: str = "steel",
                       t60_s: float | None = None):
        f = max(1.0, float(frequency_hz))
        # Generic instrument-neutral scaling; explicit geometry is preferred.
        L = float(length_m if length_m is not None else np.clip(0.65 * 440.0 / f, 0.18, 1.65))
        r = float(radius_m if radius_m is not None else np.clip(0.00032 * math.sqrt(440.0 / f), 0.00016, 0.0012))
        return cls(length_m=L, radius_m=r, material=material, frequency_hz=f, t60_s=t60_s)


@dataclass
class WoundStringPhysicalProperties:
    """Reduced core+helical-winding string construction.

    ``core_radius_m`` is the load-bearing core radius and ``winding_wire_radius_m``
    is the radius of the wrapping wire.  The wrapping mass accounts for the
    helical path length.  Bending rigidity is approximated from the core plus a
    partially coupled annular winding; this is intentionally a realtime reduced
    model rather than a contact-resolved helix/FEM model.
    """
    length_m: float = 0.86
    core_radius_m: float = 0.00035
    winding_wire_radius_m: float = 0.00022
    core_material: str | StringMaterial = "steel"
    winding_material: str | StringMaterial = "nickel"
    frequency_hz: float = 110.0
    tension_n: float | None = None
    winding_pitch_m: float | None = None
    winding_layers: int = 1
    coupling_factor: float = 0.28
    t60_s: float | None = None
    polarization_split_cents: float = 0.24

    def __post_init__(self):
        self.length_m=max(.05,float(self.length_m)); self.core_radius_m=max(1e-5,float(self.core_radius_m))
        self.winding_wire_radius_m=max(1e-6,float(self.winding_wire_radius_m))
        if isinstance(self.core_material,str): self.core_material=STRING_MATERIALS[self.core_material]
        if isinstance(self.winding_material,str): self.winding_material=STRING_MATERIALS[self.winding_material]
        self.frequency_hz=max(1.,float(self.frequency_hz)); self.winding_layers=max(1,int(self.winding_layers))
        if self.winding_pitch_m is None: self.winding_pitch_m=2.05*self.winding_wire_radius_m
        self.winding_pitch_m=max(2*self.winding_wire_radius_m,float(self.winding_pitch_m))
        self.coupling_factor=float(np.clip(self.coupling_factor,0.02,1.0))
        if self.t60_s is None:
            self.t60_s=.55*self.core_material.loss_t60_s+.45*self.winding_material.loss_t60_s
        self.t60_s=max(.03,float(self.t60_s))
        if self.tension_n is None:
            self.tension_n=(2*self.length_m*self.frequency_hz)**2*self.linear_density_kg_m
        else:
            self.tension_n=max(1e-4,float(self.tension_n)); self.frequency_hz=self.ideal_frequency_hz

    @property
    def core_area_m2(self): return math.pi*self.core_radius_m**2
    @property
    def outer_radius_m(self): return self.core_radius_m+2*self.winding_layers*self.winding_wire_radius_m
    @property
    def winding_mean_radius_m(self): return self.core_radius_m+(2*self.winding_layers-1)*self.winding_wire_radius_m
    @property
    def helix_length_factor(self):
        circumference=2*math.pi*self.winding_mean_radius_m
        return math.sqrt(1+(circumference/self.winding_pitch_m)**2)
    @property
    def winding_wire_area_m2(self): return math.pi*self.winding_wire_radius_m**2
    @property
    def linear_density_kg_m(self):
        core=self.core_material.density_kg_m3*self.core_area_m2
        wrap=self.winding_material.density_kg_m3*self.winding_wire_area_m2*self.helix_length_factor*self.winding_layers
        return core+wrap
    @property
    def second_moment_m4(self):
        icore=math.pi*self.core_radius_m**4/4
        iann=math.pi*(self.outer_radius_m**4-self.core_radius_m**4)/4
        return icore+self.coupling_factor*iann
    @property
    def flexural_rigidity_n_m2(self):
        icore=math.pi*self.core_radius_m**4/4
        iann=math.pi*(self.outer_radius_m**4-self.core_radius_m**4)/4
        return self.core_material.young_pa*icore+self.coupling_factor*self.winding_material.young_pa*iann
    @property
    def ideal_frequency_hz(self): return (1/(2*self.length_m))*math.sqrt(self.tension_n/self.linear_density_kg_m)
    @property
    def characteristic_impedance_n_s_m(self): return math.sqrt(self.tension_n*self.linear_density_kg_m)
    @property
    def wave_speed_m_s(self): return math.sqrt(self.tension_n/self.linear_density_kg_m)
    @property
    def inharmonicity_B(self): return math.pi**2*self.flexural_rigidity_n_m2/max(self.tension_n*self.length_m**2,1e-18)
    @property
    def effective_longitudinal_speed_m_s(self):
        e=(self.core_material.young_pa*self.core_area_m2 + self.coupling_factor*self.winding_material.young_pa*(math.pi*(self.outer_radius_m**2-self.core_radius_m**2))) / max(math.pi*self.outer_radius_m**2,1e-18)
        rho=self.linear_density_kg_m/max(math.pi*self.outer_radius_m**2,1e-18)
        return math.sqrt(max(e,1.)/max(rho,1.))
    @property
    def longitudinal_frequency_hz(self): return self.effective_longitudinal_speed_m_s/(2*self.length_m)
    def partial_frequency(self,n:int):
        n=max(1,int(n)); return n*self.frequency_hz*math.sqrt(1+self.inharmonicity_B*n*n)

    @classmethod
    def from_frequency(cls, frequency_hz: float, *, length_m: float | None=None,
                       core_radius_m: float | None=None, winding_wire_radius_m: float | None=None,
                       core_material: str='steel', winding_material: str='nickel'):
        f=max(1.,float(frequency_hz))
        L=float(length_m if length_m is not None else np.clip(.86*110/f,.55,1.10))
        cr=float(core_radius_m if core_radius_m is not None else np.clip(.00034*math.sqrt(110/f),.00022,.0007))
        wr=float(winding_wire_radius_m if winding_wire_radius_m is not None else np.clip(.00020*math.sqrt(110/f),.00008,.00036))
        return cls(L,cr,wr,core_material,winding_material,f)


GLIDE_COUPLE_POLE = 0.97


@numba.njit(cache=True)
def _string_kernel(exc, n, dx, dy, maxd, pc, damp_gain,
                   xdc, xhf, xs, ydc, yhf, ys, xa, ya, la1, la2, dc_r, glide=0.0, couple=0.0):
    """The whole `DispersiveString.step` loop for a fresh string with a fixed length and no bridge
    input: two polarisation delay lines (integer delay + Thiran allpass), dispersion allpasses,
    loss filters, longitudinal resonator, output DC blocker. Same arithmetic as the class, compiled.

    glide > 0: tension modulation (Tolonen, Valimaki & Karjalainen, IEEE TSAP 8(3), 2000). The string's
    elongation is taken as the energy in one loop of the delay line (eq. 15 summed over a period, where
    its twice-frequency cross terms average out), smoothed by a leaky integrator with a one-loop time
    constant (eqs 11-13), and shortens both delays by the relative speed rise (eq. 18). ``glide`` is that
    rise at the first full loop (the paper fits its scale to recordings; genny's waves carry no units),
    so the pitch starts sharp by 1200 log2(1 + glide) cents and falls as the note decays. ``couple`` 0..1
    shortens that time constant (pole from exp(-1/loop) to GLIDE_COUPLE_POLE): the elongation then follows the
    twice-frequency terms, which couples the harmonics and refills those the pluck point left out."""
    size = maxd + 8
    bx = np.zeros(size); by = np.zeros(size)
    out = np.zeros(n)
    ix = int(math.floor(dx)); fx = dx - ix
    iy = int(math.floor(dy)); fy = dy - iy
    ax = (1.0-fx)/(1.0+fx); ay = (1.0-fy)/(1.0+fy)
    tx1 = 0.0; ty1 = 0.0; ux1 = 0.0; uy1 = 0.0          # Thiran states (x: tx1,ty1; y: ux1,uy1)
    dxx = np.zeros(len(xa)); dxy = np.zeros(len(xa))    # dispersion states, x plane
    dyx = np.zeros(len(ya)); dyy = np.zeros(len(ya))
    lsx = 0.0; lsy = 0.0; ly1 = 0.0; ly2 = 0.0; px = 0.0; py = 0.0
    w = 0
    esum = 0.0; esm = 0.0; kap = 0.0
    nl = max(2, int(dx))
    ap_ = -((1.0 - couple) * math.exp(-1.0 / nl) + couple * GLIDE_COUPLE_POLE)
    ref = nl + len(exc)
    for i in range(n):
        if glide > 0.0:
            if i == ref:
                kap = glide / max(esm, 1e-30)
            dd = dx * (1.0 - kap * esm); de = dy * (1.0 - kap * esm)
            ix = int(math.floor(dd - 0.5)); fx = dd - ix   # Thiran fraction kept in [0.5, 1.5)
            iy = int(math.floor(de - 0.5)); fy = de - iy
            ax = (1.0-fx)/(1.0+fx); ay = (1.0-fy)/(1.0+fy)
        v = bx[(w-ix) % size]
        if fx < 1e-8:
            x = v; tx1 = v; ty1 = v
        else:
            x = ax*v + tx1 - ax*ty1; tx1 = v; ty1 = x
        v = by[(w-iy) % size]
        if fy < 1e-8:
            y = v; ux1 = v; uy1 = v
        else:
            y = ay*v + ux1 - ay*uy1; ux1 = v; uy1 = y
        e = exc[i] if i < len(exc) else 0.0
        xv = (1.0-pc)*x + pc*y + e
        yv = (1.0-pc)*y + pc*x + 0.055*e
        for k in range(len(xa)):
            o = xa[k]*xv + dxx[k] - xa[k]*dxy[k]; dxx[k] = xv; dxy[k] = o; xv = o
        for k in range(len(ya)):
            o = ya[k]*yv + dyx[k] - ya[k]*dyy[k]; dyx[k] = yv; dyy[k] = o; yv = o
        lsx = xs*lsx + (1.0-xs)*xv; xv = xdc*lsx + xhf*(xv-lsx)
        lsy = ys*lsy + (1.0-ys)*yv; yv = ydc*lsy + yhf*(yv-lsy)
        xv *= damp_gain; yv *= damp_gain
        if glide > 0.0:
            old = bx[(w-nl) % size]
            esum += xv*xv - old*old
            esm = -ap_*esm + (1.0+ap_)*esum              # eq. 12, unit DC gain
        bx[w] = xv; by[w] = yv; w = (w+1) % size
        ly = la1*ly1 + la2*ly2 + 0.0015*(-0.5*(xv+yv)); ly2 = ly1; ly1 = ly
        raw = xv + 0.25*yv + 0.03*ly
        py = raw - px + dc_r*py; px = raw
        out[i] = py
    return out

@dataclass
class DispersiveString:
    """Stateful two-polarization dispersive string waveguide.

    The two transverse planes use slightly different loop delays.  A weak
    longitudinal resonator exchanges energy with the bridge state.  Loop loss
    is derived from T60 and dispersion amount from the physical inharmonicity B.
    """
    properties: StringPhysicalProperties
    sr: int = 48000
    bridge_coupling: float = 0.035
    polarization_coupling: float = 0.012

    def __post_init__(self):
        p = self.properties
        f2 = p.frequency_hz * 2.0 ** (p.polarization_split_cents / 1200.0)
        d1 = d2 = 2.05   # placeholders: tuned below once the loop filters exist
        maxd = max(64, int(self.sr / max(8.0, p.frequency_hz) * 2.5) + 16)
        self.x_delay = ThiranDelay(d1, maxd)
        self.y_delay = ThiranDelay(d2, maxd)
        # One round trip gain corresponding approximately to requested T60.
        roundtrip_s = 1.0 / p.frequency_hz
        loop_gain = 10.0 ** (-3.0 * roundtrip_s / max(p.t60_s, 0.03))
        hf_gain = max(0.0, loop_gain - min(0.12, 0.015 + 12.0 * p.inharmonicity_B))
        self.x_loss = FrequencyDependentLoss(loop_gain, hf_gain, 0.45)
        self.y_loss = FrequencyDependentLoss(loop_gain**1.0002, max(0.0, hf_gain*0.999), 0.48)
        disp = rauhala_dispersion(p.frequency_hz, p.inharmonicity_B, self.sr)
        self.x_disp = StiffStringDispersion(coefficients=disp)
        self.y_disp = StiffStringDispersion(coefficients=rauhala_dispersion(f2, p.inharmonicity_B, self.sr))
        self.x_delay.delay = self.loop_delay(p.frequency_hz)
        self.y_delay.delay = self.loop_delay(f2, plane="y")
        self.x_in = 0.0; self.y_in = 0.0; self._read = None
        self._dc_x = 0.0; self._dc_y = 0.0; self._dc_r = 1.0 - 2.0*math.pi*10.0/self.sr
        lf = min(0.42*self.sr, p.longitudinal_frequency_hz)
        r = 10.0 ** (-3.0 / max(0.65*self.sr, 1.0))
        th = 2.0*math.pi*lf/self.sr
        self._la1 = 2*r*math.cos(th); self._la2 = -(r*r)
        self._ly1 = 0.0; self._ly2 = 0.0
        self.last_force_n = 0.0
        self.last_velocity_m_s = 0.0

    def loop_delay(self, frequency_hz: float, plane: str = "x") -> float:
        """Delay-line length (samples) that makes the loop resonate at `frequency_hz`: one period minus
        the phase delay of the loss and dispersion filters at that frequency (without it the string
        plays flat, by more than a semitone at the top of a guitar)."""
        loss, disp = (self.x_loss, self.x_disp) if plane == "x" else (self.y_loss, self.y_disp)
        w = 2.0*math.pi*float(frequency_hz)/self.sr
        s = loss.smoothing; g = (loss.dc_gain-loss.hf_gain)*(1.0-s)
        den = 1.0-2.0*s*math.cos(w)+s*s
        phase = -math.atan2(-g*s*math.sin(w)/den, loss.hf_gain+g*(1.0-s*math.cos(w))/den)
        for f in disp.filters:
            a = f.coefficient
            phase += math.atan2(math.sin(w), a+math.cos(w)) - math.atan2(a*math.sin(w), 1.0+a*math.cos(w))
        total = max(2.05, self.sr/float(frequency_hz) - phase/w)
        # The fractional part is realised by a first-order Thiran allpass whose phase delay equals
        # the requested fraction only at DC. Ask for the fraction that gives the right delay at this
        # frequency (exact closed form), or notes above about C6 come out tens of cents sharp.
        whole = math.floor(total); frac = total - whole
        if frac > 1e-6 and w < 3.0:
            coef = math.sin((1.0-frac)*w/2.0)/math.sin((1.0+frac)*w/2.0)
            frac = (1.0-coef)/(1.0+coef)
        return whole + min(frac, 0.999999)

    @property
    def impedance(self) -> float:
        return self.properties.characteristic_impedance_n_s_m

    def read(self) -> tuple[float, float, float]:
        # Idempotent within one sample: the Thiran allpass in each delay has state, so a caller that
        # reads the string and then calls step() must not advance it twice (that made bowed notes
        # above about C7 diverge to NaN).
        if self._read is None:
            self.x_in = float(self.x_delay.read())
            self.y_in = float(self.y_delay.read())
            force = -self.impedance * (self.x_in + 0.3*self.y_in + 0.035*self._ly1)
            self.last_force_n = force
            self._read = (self.x_in, self.y_in, force)
        return self._read

    def step(self, bridge_velocity_m_s: float, excitation: float = 0.0,
             damper: float = 0.0) -> float:
        x, y, _ = self.read()
        pc = float(np.clip(self.polarization_coupling, 0.0, 0.08))
        bc = float(np.clip(self.bridge_coupling, 0.0, 0.25))
        # The bridge takes its energy through the T60 loss filter. `bc` only scales what the bridge
        # sends back: scaling the string by (1-bc) every round trip as well cost 3.5 % per period,
        # which killed every plucked note in about 0.2 s whatever T60 was asked for.
        bx = float(bridge_velocity_m_s)
        xv = (1-pc)*x + bc*bx + pc*y + float(excitation)
        yv = (1-pc)*y + 0.78*bc*bx + pc*x + 0.055*float(excitation)
        xv = self.x_loss.process(self.x_disp.process(xv))
        yv = self.y_loss.process(self.y_disp.process(yv))
        damp = float(np.clip(damper, 0.0, 1.0))
        if damp:
            g = 1.0 - 0.10*damp
            xv *= g; yv *= g
        self.x_delay.write(xv); self.y_delay.write(yv); self._read = None
        ldrive = 0.0015*(bx - 0.5*(xv+yv))
        ly = self._la1*self._ly1 + self._la2*self._ly2 + ldrive
        self._ly2=self._ly1; self._ly1=ly
        # The loop reflects without inversion, so it also resonates at 0 Hz: a one-sided pluck or a
        # steady bow leaves a large, slowly decaying offset that is not sound (a real string is
        # fixed at both ends). Blocked at the output (10 Hz), where it cannot touch the tuning.
        self.last_velocity_m_s = self._out(xv + 0.25*yv + 0.03*ly)
        return self.last_velocity_m_s

    def _out(self, raw: float) -> float:
        self._dc_y = raw - self._dc_x + self._dc_r*self._dc_y
        self._dc_x = raw
        return self._dc_y

    def render_bow(self, dur: float, *, bow_velocity: float = 0.18, bow_pressure: float = 0.75,
                   bow_position: float = 0.13, bridge_velocity: float = 0.0) -> np.ndarray:
        """Bowed excitation using the same string state and a hysteretic stick/slip law."""
        from .graphsolver import StickSlipHysteresis
        n=max(1,int(round(float(dur)*self.sr))); y=np.zeros(n,float)
        friction=StickSlipHysteresis(static_mu=.72,kinetic_mu=.34,capture_velocity=.018,release_force=1.0)
        normal=max(.02,float(bow_pressure)); bv=float(bow_velocity); pos=float(np.clip(bow_position,.02,.48))
        attack=max(1,int(.035*self.sr)); release=max(1,int(.08*self.sr))
        z=max(self.impedance,1e-9)
        for i in range(n):
            env=min(1.,i/attack)*min(1.,(n-i)/release)
            x,yplane,_=self.read(); sv=x+.22*yplane
            rel=bv*env-sv
            force,_state=friction.step(rel,normal)
            exc=(force/z)*(0.55+0.45*math.sin(math.pi*pos))
            y[i]=self.step(bridge_velocity,excitation=exc)
        peak=float(np.max(np.abs(y))) if len(y) else 0.0
        if peak>1: y/=peak
        return y

    def energy_proxy(self) -> float:
        return float(self.x_in*self.x_in + self.y_in*self.y_in + 0.1*(self._ly1*self._ly1+self._ly2*self._ly2))

    def _run(self, exc: np.ndarray, n: int, position: float, damper: float = 0.0, glide: float = 0.0,
             couple: float = 0.0) -> np.ndarray:
        """Render `n` samples of a fresh string driven by `exc` (compiled). The excitation first goes
        through the pluck-position comb: a string released at a fraction `position` of its length
        has no partials with a node there (JOS, extended Karplus-Strong)."""
        d = self.x_delay.delay
        lag = max(1, int(round(float(np.clip(position, 0.01, 0.99)) * d)))
        e = np.zeros(len(exc) + lag)
        e[:len(exc)] += exc
        e[lag:] -= exc
        xl, yl = self.x_loss, self.y_loss
        clip = lambda v: min(max(float(v), 1.0), self.x_delay.max_delay - 3.0)
        return _string_kernel(e, int(n), clip(d), clip(self.y_delay.delay), int(self.x_delay.max_delay),
                              float(np.clip(self.polarization_coupling, 0.0, 0.08)),
                              1.0 - 0.10*float(np.clip(damper, 0.0, 1.0)),
                              xl.dc_gain, xl.hf_gain, xl.smoothing, yl.dc_gain, yl.hf_gain, yl.smoothing,
                              np.array([f.coefficient for f in self.x_disp.filters]),
                              np.array([f.coefficient for f in self.y_disp.filters]),
                              self._la1, self._la2, self._dc_r, float(max(glide, 0.0)),
                              float(np.clip(couple, 0.0, 1.0)))

    def render_pluck(self, dur: float, *, velocity: float = 0.8, position: float = 0.2,
                     bridge_velocity: float = 0.0, glide_cents: float = 0.0, glide_couple: float = 0.0) -> np.ndarray:
        """glide_cents: tension-modulation pitch excess at the start of the note; glide_couple 0..1: how much that
        modulation couples the harmonics (see _string_kernel)."""
        n=max(1,int(round(float(dur)*self.sr)))
        width=max(2,int(round(0.0025*self.sr)))
        exc=float(velocity)*np.sin(np.pi*np.arange(width)/max(1,width-1))
        if bridge_velocity == 0.0:
            y=self._run(exc,n,position,glide=2.0**(max(float(glide_cents),0.0)/1200.0)-1.0,couple=glide_couple)
        else:
            y=np.zeros(n,float)
            for i in range(n):
                y[i]=self.step(bridge_velocity,excitation=float(exc[i]) if i < width else 0.0)
        peak=float(np.max(np.abs(y))) if len(y) else 0.0
        if peak>1.0: y/=peak
        return y

# ============================================================================
# v0.20: local string interactions (pick/finger/fret/buzz/mute/slide/harmonics)
# ============================================================================

FLESH_STICK_S, FLESH_SLIP_S = 0.020, 0.0008


@dataclass(frozen=True)
class StringExciter:
    """Reduced local string exciter.

    ``hardness`` controls contact duration rather than post-EQ, so a rigid pick
    injects a shorter/broader-band impulse than a fingertip.
    """
    kind: str = "pick"          # pick | finger | nail
    hardness: float = 0.7
    position: float = 0.2
    direction: float = 1.0

    def pulse(self, sr: int, velocity: float = 0.8) -> np.ndarray:
        h=float(np.clip(self.hardness,0.0,1.0))
        # ~0.25 ms hard pick to ~4.5 ms fingertip contact.
        ms=4.5*(1.0-h)+0.25*h
        if self.kind.lower()=="finger": ms=max(ms,2.2)
        # a nail releases in 0.25-1 ms, shorter the harder the stroke (min(ms, 1) pinned every guitar note to 1 ms)
        elif self.kind.lower()=="nail": ms=0.25+0.75*(1.0-h)
        if self.kind.lower()=="flesh":
            # the fingertip sticks and loads the string: force ramps over FLESH_STICK_S while the pad's hardening spring
            # S(xi) = xi1 xi / (xi2 - xi) gives the displacement xi ~ u / (1 + u) of a linear force ramp u = t / T, then
            # the string slips off in FLESH_SLIP_S
            nr=max(2,int(round(FLESH_STICK_S*sr))); ns=max(2,int(round(FLESH_SLIP_S*sr)))
            u=np.arange(nr,dtype=float)/nr
            p=np.concatenate([2.0*u/(1.0+u),np.cos(0.5*np.pi*np.arange(1,ns+1)/ns)])
            pos=float(np.clip(self.position,.01,.99))
            return p*float(velocity)*float(np.sign(self.direction) or 1.0)*(0.62+0.38*math.sin(math.pi*pos))
        n=max(2,int(round(ms*1e-3*sr)))
        x=np.arange(n,dtype=float)/max(1,n-1)
        # Smooth displacement release with finite slope at both ends.
        p=np.sin(np.pi*x)**1.25
        pos=float(np.clip(self.position,.01,.99))
        p*=float(velocity)*float(np.sign(self.direction) or 1.0)*(0.62+0.38*math.sin(math.pi*pos))
        return p


@dataclass(frozen=True)
class FretboardGeometry:
    """Equal-tempered fretboard geometry and reduced clearance model."""
    scale_length_m: float = 0.648
    action_m: float = 0.0018
    fret_height_m: float = 0.0010

    def vibrating_length(self, fret: float) -> float:
        return max(.02,float(self.scale_length_m)/(2.0**(float(fret)/12.0)))

    def frequency_ratio(self, fret: float) -> float:
        return 2.0**(float(fret)/12.0)

    def fret_position_m(self, fret: float) -> float:
        return float(self.scale_length_m)-self.vibrating_length(fret)


def _render_exciter(self: DispersiveString, dur: float, exciter: StringExciter, *,
                    velocity: float=.8, bridge_velocity: float=0.0,
                    palm_mute: float=0.0, glide_cents: float=0.0) -> np.ndarray:
    n=max(1,int(round(float(dur)*self.sr))); out=np.zeros(n,float)
    pulse=exciter.pulse(self.sr,velocity)
    mute=float(np.clip(palm_mute,0.0,1.0))
    if bridge_velocity == 0.0:
        # Palm contact is local bridge damping, not a post-render envelope.
        out=self._run(np.asarray(pulse,float),n,exciter.position,damper=.72*mute,
                      glide=2.0**(max(float(glide_cents),0.0)/1200.0)-1.0)
    else:
        for i in range(n):
            e=float(pulse[i]) if i<len(pulse) else 0.0
            out[i]=self.step(bridge_velocity,excitation=e,damper=.72*mute)
    peak=float(np.max(np.abs(out))) if len(out) else 0.
    if peak>1: out/=peak
    return out


def _clone_with_length(properties, length_m: float):
    """Preserve construction and tension while shortening vibrating length."""
    from dataclasses import replace
    return replace(properties,length_m=max(.02,float(length_m)),tension_n=float(properties.tension_n))


def _render_fretted(self: DispersiveString, fret: float, dur: float, *, velocity: float=.8,
                    fretboard: FretboardGeometry | None=None, hardness: float=.65,
                    position: float=.2, palm_mute: float=0.0) -> np.ndarray:
    fb=fretboard or FretboardGeometry(scale_length_m=self.properties.length_m)
    L=min(self.properties.length_m, fb.vibrating_length(fret))
    props=_clone_with_length(self.properties,L)
    s=DispersiveString(props,sr=self.sr,bridge_coupling=self.bridge_coupling,
                       polarization_coupling=self.polarization_coupling)
    return s.render_exciter(dur,StringExciter('pick',hardness,position),velocity=velocity,palm_mute=palm_mute)


def _render_fret_buzz(self: DispersiveString, fret: float, dur: float, *, velocity: float=.8,
                      fretboard: FretboardGeometry | None=None, buzz: float=.5,
                      position: float=.2, seed: int=0) -> np.ndarray:
    """Fretted string plus thresholded string/fret collisions.

    The contact signal is derived from the string displacement proxy crossing a
    clearance threshold.  It is intentionally reduced-order, not a geometric
    contact FEM.
    """
    fb=fretboard or FretboardGeometry(scale_length_m=self.properties.length_m)
    base=self.render_fretted(fret,dur,velocity=velocity,fretboard=fb,hardness=.72,position=position)
    n=len(base); if_buzz=float(np.clip(buzz,0.,1.))
    if n<2 or if_buzz<=0: return base
    # Lower action and stronger playing increase collision probability.
    clearance=np.clip(fb.action_m/0.0030,.18,1.5)
    thr=(.20+.42*clearance)/(max(.15,velocity)*(0.75+0.65*if_buzz))
    proxy=base/(np.max(np.abs(base))+1e-12)
    penetration=np.maximum(np.abs(proxy)-thr,0.0)
    impacts=np.maximum(0.0,np.diff(penetration,prepend=0.0))
    # Short metal fret resonances excited only by impacts.
    t=np.arange(n)/self.sr
    ring=np.zeros(n,float)
    for f,a,d in ((2400.,1.,.010),(4100.,.55,.0065),(6900.,.24,.004)):
        if f < .45*self.sr:
            # causal resonator by convolution with compact impulse response
            m=min(n,max(4,int(.035*self.sr))); tt=np.arange(m)/self.sr
            ir=np.sin(2*np.pi*f*tt)*np.exp(-tt/d)
            ring += np.convolve(impacts,ir,mode='full')[:n]*a
    y=base + ring*(.05+.13*if_buzz)
    peak=np.max(np.abs(y))+1e-12
    if peak>1: y/=peak
    return y


def _render_glissando(self: DispersiveString, dur: float, start_frequency: float,
                      end_frequency: float, *, velocity: float=.8, position: float=.2,
                      curve: float=1.0) -> np.ndarray:
    """Continuous-length/pitch render by continuously moving waveguide delay."""
    n=max(1,int(round(float(dur)*self.sr))); out=np.zeros(n,float)
    pulse=StringExciter('finger',.35,position).pulse(self.sr,velocity)
    f0=max(8.,float(start_frequency)); f1=max(8.,float(end_frequency)); c=max(.05,float(curve))
    z=max(self.impedance,1e-9)
    for i in range(n):
        u=(i/max(1,n-1))**c; f=f0*(f1/f0)**u
        dx=np.clip(self.loop_delay(f),2.05,self.x_delay.max_delay-3.)
        fy=f*2**(self.properties.polarization_split_cents/1200.)
        dy=np.clip(self.loop_delay(fy,'y'),2.05,self.y_delay.max_delay-3.)
        x=float(self.x_delay.read(dx)); yp=float(self.y_delay.read(dy))
        self.x_in=x; self.y_in=yp; self.last_force_n=-z*(x+.3*yp+.035*self._ly1)
        pc=float(np.clip(self.polarization_coupling,0,.08)); bc=float(np.clip(self.bridge_coupling,0,.25))
        e=float(pulse[i]) if i<len(pulse) else 0.
        xv=(1-pc)*x+pc*yp+e; yv=(1-pc)*yp+pc*x+.055*e
        xv=self.x_loss.process(self.x_disp.process(xv)); yv=self.y_loss.process(self.y_disp.process(yv))
        self.x_delay.write(xv); self.y_delay.write(yv)
        ld=.0015*(-.5*(xv+yv)); ly=self._la1*self._ly1+self._la2*self._ly2+ld
        self._ly2=self._ly1; self._ly1=ly
        out[i]=self._out(xv+.25*yv+.03*ly)
    peak=np.max(np.abs(out))+1e-12
    if peak>1: out/=peak
    return out


def _render_natural_harmonic(self: DispersiveString, harmonic: int, dur: float, *,
                             velocity: float=.7, touch: float=.85) -> np.ndarray:
    """Natural harmonic: weak fundamental plus a touched-node partial renderer."""
    h=max(2,int(harmonic)); touch=float(np.clip(touch,0.,1.))
    base=self.render_exciter(dur,StringExciter('finger',.28,1.0/h),velocity=velocity*(1-.82*touch))
    # A touched node strongly favours the corresponding stiff-string partial.
    target=self.properties.partial_frequency(h)
    if isinstance(self.properties,WoundStringPhysicalProperties):
        p=WoundStringPhysicalProperties.from_frequency(target,length_m=max(.05,self.properties.length_m/h),
            core_radius_m=self.properties.core_radius_m,winding_wire_radius_m=self.properties.winding_wire_radius_m,
            core_material=self.properties.core_material.name,winding_material=self.properties.winding_material.name)
    else:
        p=StringPhysicalProperties.from_frequency(target,length_m=max(.03,self.properties.length_m/h),
            radius_m=self.properties.radius_m,material=self.properties.material.name,t60_s=.72*self.properties.t60_s)
    overtone=DispersiveString(p,sr=self.sr).render_exciter(dur,StringExciter('finger',.42,.3),velocity=velocity)
    y=(1-.72*touch)*base + (.55+.35*touch)*overtone
    peak=np.max(np.abs(y))+1e-12
    if peak>1:y/=peak
    return y


def _render_slap(self: DispersiveString, dur: float, *, velocity: float=.9,
                 pop: bool=False, fretboard: FretboardGeometry|None=None, seed:int=0) -> np.ndarray:
    """Bass slap/pop as hard excitation plus distributed fret impacts."""
    hard=.92 if pop else .78; pos=.72 if pop else .48
    base=self.render_exciter(dur,StringExciter('pick',hard,pos),velocity=velocity)
    fb=fretboard or FretboardGeometry(scale_length_m=self.properties.length_m,action_m=.0016)
    # Open-string buzz proxy; slap has more fret collision, pop more initial snap.
    proxy=base/(np.max(np.abs(base))+1e-12)
    thr=.34 if pop else .23
    impacts=np.maximum(0.,np.diff(np.maximum(np.abs(proxy)-thr,0.),prepend=0.))
    n=len(base); ring=np.zeros(n)
    for f,d,a in ((1800,.012,.75),(3200,.008,.55),(5200,.005,.3)):
        if f<.45*self.sr:
            m=min(n,max(4,int(.03*self.sr))); tt=np.arange(m)/self.sr
            ring+=a*np.convolve(impacts,np.sin(2*np.pi*f*tt)*np.exp(-tt/d),mode='full')[:n]
    y=base+(.11 if pop else .075)*ring
    peak=np.max(np.abs(y))+1e-12
    if peak>1:y/=peak
    return y

# Public v0.20 methods on the shared stateful string core.
DispersiveString.render_exciter = _render_exciter
DispersiveString.render_fretted = _render_fretted
DispersiveString.render_fret_buzz = _render_fret_buzz
DispersiveString.render_glissando = _render_glissando
DispersiveString.render_natural_harmonic = _render_natural_harmonic
DispersiveString.render_slap = _render_slap

# ============================================================================
# v0.21: expressive/contact gestures on the shared physical string state
# ============================================================================

def _dynamic_string_loop(self: DispersiveString, frequencies: np.ndarray, *,
                         excitation: np.ndarray | None = None,
                         bridge_velocity: float = 0.0,
                         damper: float = 0.0,
                         fret_clearance: float | None = None,
                         collision_gain: float = 0.0,
                         friction: np.ndarray | None = None) -> np.ndarray:
    """Render one continuously retuned stateful loop.

    ``frequencies`` changes the waveguide round-trip delay sample-by-sample.
    Optional fret contact is *inside* the feedback loop: when the transverse
    displacement proxy crosses the clearance, an opposite collision impulse is
    injected into the next travelling wave rather than mixed after rendering.
    """
    freqs=np.asarray(frequencies,float).reshape(-1)
    n=len(freqs); out=np.zeros(n,float)
    exc=np.zeros(n,float) if excitation is None else np.asarray(excitation,float)
    fr=np.zeros(n,float) if friction is None else np.asarray(friction,float)
    z=max(self.impedance,1e-9)
    clearance=None if fret_clearance is None else max(1e-5,float(fret_clearance))
    c_gain=max(0.0,float(collision_gain))
    prev_disp=0.0
    for i in range(n):
        f=max(8.0,float(freqs[i]))
        dx=float(np.clip(self.loop_delay(f),2.05,self.x_delay.max_delay-3.0))
        fy=f*2.0**(self.properties.polarization_split_cents/1200.0)
        dy=float(np.clip(self.loop_delay(fy,'y'),2.05,self.y_delay.max_delay-3.0))
        x=float(self.x_delay.read(dx)); yp=float(self.y_delay.read(dy))
        self.x_in=x; self.y_in=yp
        self.last_force_n=-z*(x+.3*yp+.035*self._ly1)
        pc=float(np.clip(self.polarization_coupling,0,.08)); bc=float(np.clip(self.bridge_coupling,0,.25))
        e=float(exc[i]) if i < len(exc) else 0.0
        e += float(fr[i]) if i < len(fr) else 0.0
        # Reduced displacement proxy from local travelling-wave state.
        disp=0.5*(x+yp)
        if clearance is not None and c_gain>0.0:
            penetration=max(abs(disp)-clearance,0.0)
            prev_pen=max(abs(prev_disp)-clearance,0.0)
            # Only inject strongly on inward crossing/renewed penetration.
            impact=max(0.0,penetration-prev_pen)
            if impact>0.0:
                e -= math.copysign(c_gain*impact,disp)
        prev_disp=disp
        xv=(1-pc)*x+bc*float(bridge_velocity)+pc*yp+e
        yv=(1-pc)*yp+.78*bc*float(bridge_velocity)+pc*x+.055*e
        xv=self.x_loss.process(self.x_disp.process(xv)); yv=self.y_loss.process(self.y_disp.process(yv))
        d=float(np.clip(damper,0.,1.))
        if d:
            g=1.-.10*d; xv*=g; yv*=g
        self.x_delay.write(xv); self.y_delay.write(yv)
        ld=.0015*(float(bridge_velocity)-.5*(xv+yv))
        ly=self._la1*self._ly1+self._la2*self._ly2+ld
        self._ly2=self._ly1; self._ly1=ly
        self.last_velocity_m_s=self._out(xv+.25*yv+.03*ly)
        out[i]=self.last_velocity_m_s
    peak=float(np.max(np.abs(out))) if n else 0.0
    if peak>1.0: out/=peak
    return out


def _render_bend(self: DispersiveString, dur: float, *, cents: float=100.0,
                 velocity: float=.8, position: float=.2, curve: float=1.4,
                 release: bool=False) -> np.ndarray:
    """String bend by changing tension/round-trip delay continuously.

    For constant length and linear density, T/T0=(f/f0)^2.  We expose the
    implied tension trajectory via ``last_tension_ratio`` for diagnostics.
    """
    n=max(1,int(round(float(dur)*self.sr)))
    u=np.linspace(0.,1.,n)**max(.05,float(curve))
    if release: u=1.-u
    ratio=2.0**((float(cents)*u)/1200.0)
    freqs=self.properties.frequency_hz*ratio
    self.last_tension_ratio=float(ratio[-1]**2)
    pulse=StringExciter('finger',.38,position).pulse(self.sr,velocity)
    exc=np.zeros(n); exc[:min(n,len(pulse))]=pulse[:n]
    return self._dynamic_string_loop(freqs,excitation=exc)


def _render_vibrato(self: DispersiveString, dur: float, *, depth_cents: float=18.0,
                    rate_hz: float=5.2, velocity: float=.75, position: float=.2,
                    onset_s: float=.12) -> np.ndarray:
    """Finger vibrato as periodic tension modulation inside the waveguide."""
    n=max(1,int(round(float(dur)*self.sr))); t=np.arange(n)/self.sr
    ramp=np.clip((t-float(onset_s))/max(.03,float(onset_s)),0.,1.)
    cents=float(depth_cents)*ramp*np.sin(2*np.pi*float(rate_hz)*t)
    freqs=self.properties.frequency_hz*2.0**(cents/1200.0)
    pulse=StringExciter('finger',.35,position).pulse(self.sr,velocity)
    exc=np.zeros(n); exc[:min(n,len(pulse))]=pulse[:n]
    return self._dynamic_string_loop(freqs,excitation=exc)


def _render_slide(self: DispersiveString, dur: float, start_frequency: float,
                  end_frequency: float, *, velocity: float=.75, position: float=.25,
                  roughness: float=.45, pressure: float=.5, seed: int=0) -> np.ndarray:
    """Metal/glass slide with continuous string length and friction excitation."""
    n=max(1,int(round(float(dur)*self.sr))); t=np.arange(n)/self.sr
    u=np.linspace(0.,1.,n)
    f0=max(8.,float(start_frequency)); f1=max(8.,float(end_frequency))
    freqs=f0*(f1/f0)**u
    pulse=StringExciter('finger',.32,position).pulse(self.sr,velocity)
    exc=np.zeros(n); exc[:min(n,len(pulse))]=pulse[:n]
    rng=np.random.default_rng(seed)
    # Friction energy rises with slide speed in log-frequency space and pressure.
    speed=abs(math.log(max(f1,1e-9)/max(f0,1e-9)))/max(float(dur),1e-6)
    noise=rng.standard_normal(n)
    # Differenced noise emphasizes the scratch while a low component keeps rubbing body.
    scratch=np.concatenate([[0.0],np.diff(noise)])
    rub=.55*noise+.45*scratch
    env=np.sin(np.pi*np.clip(u,0,1))**.6
    friction=rub*env*float(np.clip(roughness,0,1))*float(np.clip(pressure,0,1))*(.002+.006*speed)
    return self._dynamic_string_loop(freqs,excitation=exc,friction=friction)


def _render_fret_buzz_feedback(self: DispersiveString, fret: float, dur: float, *,
                              velocity: float=.8, fretboard: FretboardGeometry|None=None,
                              buzz: float=.6, position: float=.2) -> np.ndarray:
    """Bidirectional/reinjected fret buzz in the travelling-wave loop."""
    fb=fretboard or FretboardGeometry(scale_length_m=self.properties.length_m)
    L=min(self.properties.length_m,fb.vibrating_length(fret))
    props=_clone_with_length(self.properties,L)
    s=DispersiveString(props,sr=self.sr,bridge_coupling=self.bridge_coupling,
                       polarization_coupling=self.polarization_coupling)
    n=max(1,int(round(float(dur)*self.sr)))
    pulse=StringExciter('pick',.74,position).pulse(self.sr,velocity)
    exc=np.zeros(n); exc[:min(n,len(pulse))]=pulse[:n]
    # Internal wave amplitudes are normalized, so convert physical action to a
    # bounded reduced clearance while preserving monotonic action dependence.
    clearance=float(np.clip(.10+.55*(fb.action_m/.003),.12,.78))
    freqs=np.full(n,props.frequency_hz)
    return s._dynamic_string_loop(freqs,excitation=exc,fret_clearance=clearance,
                                  collision_gain=.35+1.25*float(np.clip(buzz,0,1)))


def _render_hammer_on(self: DispersiveString, fret: float, dur: float, *,
                      velocity: float=.65, fretboard: FretboardGeometry|None=None) -> np.ndarray:
    """Hammer-on: local finger/fret impulse with no separate pluck."""
    fb=fretboard or FretboardGeometry(scale_length_m=self.properties.length_m)
    L=min(self.properties.length_m,fb.vibrating_length(fret))
    props=_clone_with_length(self.properties,L); s=DispersiveString(props,sr=self.sr)
    n=max(1,int(round(float(dur)*self.sr))); exc=np.zeros(n)
    m=max(2,int(.0018*self.sr)); tt=np.arange(m)/self.sr
    exc[:m]=float(velocity)*.65*np.sin(np.pi*np.arange(m)/max(1,m-1))*np.exp(-tt/.003)
    return s._dynamic_string_loop(np.full(n,props.frequency_hz),excitation=exc)


def _render_pull_off(self: DispersiveString, fret: float, dur: float, *,
                     velocity: float=.6, fretboard: FretboardGeometry|None=None) -> np.ndarray:
    """Pull-off: short lateral finger release at the new vibrating length."""
    fb=fretboard or FretboardGeometry(scale_length_m=self.properties.length_m)
    L=min(self.properties.length_m,fb.vibrating_length(fret))
    props=_clone_with_length(self.properties,L); s=DispersiveString(props,sr=self.sr)
    return s.render_exciter(dur,StringExciter('finger',.18,.12,direction=-1.),velocity=velocity)


def _render_tapping(self: DispersiveString, fret: float, dur: float, *,
                    velocity: float=.72, fretboard: FretboardGeometry|None=None) -> np.ndarray:
    """Two-handed tapping: harder local fret impact than hammer-on."""
    y=self.render_hammer_on(fret,dur,velocity=min(1.,velocity*1.12),fretboard=fretboard)
    n=len(y); m=min(n,max(4,int(.012*self.sr))); tt=np.arange(m)/self.sr
    click=np.sin(2*np.pi*2900*tt)*np.exp(-tt/.0028)
    y[:m]+=click*.035*velocity
    p=np.max(np.abs(y))+1e-12
    if p>1:y/=p
    return y


def _render_dead_note(self: DispersiveString, dur: float, *, velocity: float=.8,
                      position: float=.2, mute: float=.94) -> np.ndarray:
    """Muted/dead note: excitation with very strong in-loop finger damping."""
    return self.render_exciter(dur,StringExciter('finger',.55,position),velocity=velocity,
                               palm_mute=float(np.clip(mute,0,1)))


def _render_pinch_harmonic(self: DispersiveString, harmonic: int, dur: float, *,
                           velocity: float=.8, squeal: float=.85) -> np.ndarray:
    """Artificial/pinch harmonic: hard pick + immediate thumb-node touch."""
    h=max(2,int(harmonic)); q=float(np.clip(squeal,0,1))
    natural=self.render_natural_harmonic(h,dur,velocity=velocity,touch=.72+.25*q)
    n=len(natural); t=np.arange(n)/self.sr
    f=self.properties.partial_frequency(h)
    edge=np.sin(2*np.pi*f*t)*np.exp(-t/max(.04,.22-.12*q))*velocity
    y=(.72-.18*q)*natural+(.18+.30*q)*edge
    p=np.max(np.abs(y))+1e-12
    if p>1:y/=p
    return y

DispersiveString._dynamic_string_loop = _dynamic_string_loop
DispersiveString.render_bend = _render_bend
DispersiveString.render_vibrato = _render_vibrato
DispersiveString.render_slide = _render_slide
DispersiveString.render_fret_buzz_feedback = _render_fret_buzz_feedback
DispersiveString.render_hammer_on = _render_hammer_on
DispersiveString.render_pull_off = _render_pull_off
DispersiveString.render_tapping = _render_tapping
DispersiveString.render_dead_note = _render_dead_note
DispersiveString.render_pinch_harmonic = _render_pinch_harmonic

# ============================================================================
# v0.22: continuous combined performance + nonlinear string/fret contact
# ============================================================================

@dataclass(frozen=True)
class HertzFretContact:
    """Reduced nonlinear string/fret contact.

    The elastic term follows the Hertz-like law F = k * delta^(3/2).  A small
    penetration-dependent damping term absorbs collision energy.  This is a
    local reduced contact model, not a resolved 3-D fret/string FEM contact.
    """
    stiffness_n_m32: float = 8.0e7
    damping_n_s_m32: float = 180.0
    max_force_n: float = 80.0

    def force_n(self, penetration_m: float, normal_velocity_m_s: float = 0.0) -> float:
        d=max(0.0,float(penetration_m))
        if d <= 0.0:
            return 0.0
        elastic=max(0.0,float(self.stiffness_n_m32))*d**1.5
        # Damping only resists inward motion (positive normal velocity here).
        damping=max(0.0,float(self.damping_n_s_m32))*math.sqrt(d)*max(0.0,float(normal_velocity_m_s))
        return float(min(max(0.0,float(self.max_force_n)), elastic+damping))


@dataclass(frozen=True)
class StringGesture:
    """One time-bounded control gesture for :class:`StringPerformance`.

    ``kind`` supports bend, vibrato, slide, palm_mute, fret_contact and
    harmonic_touch.  Values are interpreted by gesture type and linearly
    interpolated from ``value`` to ``end_value`` when provided.
    """
    kind: str
    start_s: float
    end_s: float
    value: float
    end_value: float | None = None
    rate_hz: float = 5.2

    def value_at(self, t: float) -> float | None:
        if t < self.start_s or t > self.end_s:
            return None
        if self.end_s <= self.start_s:
            return float(self.end_value if self.end_value is not None else self.value)
        u=(t-self.start_s)/(self.end_s-self.start_s)
        b=float(self.value if self.end_value is None else self.end_value)
        return float(self.value)+(b-float(self.value))*float(np.clip(u,0.0,1.0))


class StringPerformance:
    """Stateful combined-performance renderer for one physical string.

    Multiple gestures are evaluated in the same sample loop.  Fret/slide
    position changes effective vibrating length, bend/vibrato changes tension,
    palm mute changes loop loss, and Hertz contact feeds collision force back
    into the travelling wave.  Diagnostics expose the final physical control
    state rather than hiding the mapping behind a post-effect.
    """
    def __init__(self, properties: StringPhysicalProperties | WoundStringPhysicalProperties,
                 *, sr: int = 48000, fretboard: FretboardGeometry | None = None,
                 contact: HertzFretContact | None = None):
        self.properties=properties
        self.sr=int(sr)
        self.core=DispersiveString(properties,sr=self.sr)
        self.fretboard=fretboard or FretboardGeometry(scale_length_m=properties.length_m)
        self.contact=contact or HertzFretContact()
        self.last_diagnostics={}

    def _lanes(self, n: int, gestures: list[StringGesture]):
        t=np.arange(n,dtype=float)/self.sr
        bend=np.zeros(n); fret=np.zeros(n); mute=np.zeros(n); contact=np.zeros(n); touch=np.zeros(n)
        for g in gestures:
            kind=g.kind.lower().replace('-','_')
            a=max(0,min(n,int(round(g.start_s*self.sr))))
            b=max(a,min(n,int(round(g.end_s*self.sr))+1))
            if b<=a: continue
            tt=t[a:b]
            if kind=='vibrato':
                # value is depth in cents, rate_hz controls periodic finger motion.
                env=np.sin(np.pi*np.clip((tt-g.start_s)/max(g.end_s-g.start_s,1e-9),0,1))**.5
                bend[a:b]+=float(g.value)*env*np.sin(2*np.pi*float(g.rate_hz)*(tt-g.start_s))
                continue
            va=float(g.value); vb=float(g.end_value if g.end_value is not None else g.value)
            lane=np.linspace(va,vb,b-a)
            if kind in ('bend','bend_cents'): bend[a:b]+=lane
            elif kind in ('slide','fret','fret_position'): fret[a:b]=lane
            elif kind in ('palm_mute','mute'): mute[a:b]=np.maximum(mute[a:b],np.clip(lane,0,1))
            elif kind in ('fret_contact','buzz','fret_buzz'): contact[a:b]=np.maximum(contact[a:b],np.clip(lane,0,1))
            elif kind in ('harmonic_touch','touch'): touch[a:b]=np.maximum(touch[a:b],np.clip(lane,0,1))
        return t,bend,fret,mute,contact,touch

    def render(self, dur: float, *, gestures: list[StringGesture] | None = None,
               velocity: float = 0.8, exciter: StringExciter | None = None,
               initial_fret: float = 0.0) -> np.ndarray:
        n=max(1,int(round(float(dur)*self.sr)))
        gestures=list(gestures or [])
        t,bend_cents,fret_lane,mute_lane,contact_lane,touch_lane=self._lanes(n,gestures)
        # Zero-valued slide lane means retain initial fret unless an explicit
        # slide/fret gesture exists at that sample.
        has_fret=any(g.kind.lower().replace('-','_') in ('slide','fret','fret_position') for g in gestures)
        if has_fret:
            # Outside active slide windows, hold the nearest specified position
            # rather than snapping to the nut.  This gives continuous geometry.
            active=np.zeros(n,bool)
            for g in gestures:
                if g.kind.lower().replace('-','_') in ('slide','fret','fret_position'):
                    a=max(0,min(n,int(round(g.start_s*self.sr)))); b=max(a,min(n,int(round(g.end_s*self.sr))+1)); active[a:b]=True
            last=float(initial_fret)
            for i in range(n):
                if active[i]: last=float(fret_lane[i])
                else: fret_lane[i]=last
        else:
            fret_lane.fill(float(initial_fret))

        # Vibrating length from fret geometry; tension trajectory from bend.
        length=self.properties.length_m/(2.0**(fret_lane/12.0))
        length=np.clip(length,0.02,self.properties.length_m)
        tension_ratio=2.0**(2.0*bend_cents/1200.0)
        # f ~ (1/2L) sqrt(T/mu), referenced to physical open-string geometry.
        freqs=self.properties.frequency_hz*(self.properties.length_m/length)*np.sqrt(tension_ratio)
        freqs=np.clip(freqs,8.0,0.42*self.sr)

        ex=exciter or StringExciter('pick',.72,.2)
        pulse=ex.pulse(self.sr,velocity)
        excitation=np.zeros(n); excitation[:min(n,len(pulse))]=pulse[:n]

        out=np.zeros(n,float); z=max(self.core.impedance,1e-9)
        prev_v=0.0; displacement=0.0; max_contact_force=0.0
        base_B=max(float(self.properties.inharmonicity_B),1e-12)
        dynamic_B=np.empty(n,float)
        for i in range(n):
            f=float(freqs[i]); dx=float(np.clip(self.core.loop_delay(f),2.05,self.core.x_delay.max_delay-3.0))
            fy=f*2.0**(self.properties.polarization_split_cents/1200.0)
            dy=float(np.clip(self.core.loop_delay(fy,'y'),2.05,self.core.y_delay.max_delay-3.0))
            x=float(self.core.x_delay.read(dx)); yp=float(self.core.y_delay.read(dy))
            pc=float(np.clip(self.core.polarization_coupling,0,.08)); bc=float(np.clip(self.core.bridge_coupling,0,.25))
            e=float(excitation[i])

            # Reduced displacement estimate from local transverse velocity.
            vlocal=x+.25*yp
            displacement=.9992*displacement+(vlocal/self.sr)*0.05
            c=float(contact_lane[i])
            if c>0.0:
                # Physical clearance shrinks with contact amount; action provides
                # the base scale.  Penetration acts against string displacement.
                clearance=max(1e-6,self.fretboard.action_m*(1.0-.995*c))
                penetration=max(abs(displacement)-clearance,0.0)
                normal_v=max(0.0,(abs(vlocal)-abs(prev_v)))
                force=self.contact.force_n(penetration,normal_v)
                max_contact_force=max(max_contact_force,force)
                if force>0.0:
                    e -= math.copysign((force/z)*0.035,displacement if displacement else vlocal)
            prev_v=vlocal

            xv=(1-pc)*x+pc*yp+e
            yv=(1-pc)*yp+pc*x+.055*e
            xv=self.core.x_loss.process(self.core.x_disp.process(xv)); yv=self.core.y_loss.process(self.core.y_disp.process(yv))
            # Palm mute is an actual loop loss. Harmonic touch damps the
            # fundamental path while leaving a weak node-selected component.
            damp=float(np.clip(mute_lane[i],0,1)); touch=float(np.clip(touch_lane[i],0,1))
            g=max(0.0,1.0-.12*damp-.045*touch)
            xv*=g; yv*=g
            if touch>0.0:
                h=max(2,int(round(2+5*touch)))
                node=math.sin(math.pi*h*float(np.clip(ex.position,.01,.99)))
                xv*=.72+.28*abs(node); yv*=.75+.25*abs(node)
            self.core.x_delay.write(xv); self.core.y_delay.write(yv)
            ld=.0015*(-.5*(xv+yv)); ly=self.core._la1*self.core._ly1+self.core._la2*self.core._ly2+ld
            self.core._ly2=self.core._ly1; self.core._ly1=ly
            out[i]=self.core._out(xv+.25*yv+.03*ly)
            # B scales approximately inversely with T*L^2.
            dynamic_B[i]=base_B/(max(tension_ratio[i],1e-9)*max((length[i]/self.properties.length_m)**2,1e-9))

        peak=float(np.max(np.abs(out))) if n else 0.0
        if peak>1.0: out/=peak
        self.last_diagnostics={
            'final_frequency_hz': float(freqs[-1]),
            'final_length_m': float(length[-1]),
            'final_tension_ratio': float(tension_ratio[-1]),
            'final_inharmonicity_B': float(dynamic_B[-1]),
            'max_contact_force_n': float(max_contact_force),
            'min_length_m': float(np.min(length)),
            'max_frequency_hz': float(np.max(freqs)),
        }
        return out


@numba.njit(cache=True)
def _admittance_reflect_kernel(vp, b1, b2, a1, a2, Yr, Yi, Y0):
    n, K = vp.shape
    R = b1.shape[0]
    M = np.linalg.inv(Yi + Y0)
    A = Yi - Y0
    vm = np.zeros((n, K))
    y1 = np.zeros((R, K))
    y2 = np.zeros((R, K))
    d1 = np.zeros(K)
    for i in range(n):
        u = A @ vp[i]
        for r in range(R):
            u += Yr[r] @ y1[r]
        vm[i] = M @ u
        d = vp[i] - vm[i]
        for r in range(R):
            for k in range(K):
                y = b1[r] * d[k] + b2[r] * d1[k] - a1[r] * y1[r, k] - a2[r] * y2[r, k]
                y2[r, k] = y1[r, k]
                y1[r, k] = y
        d1[:] = d
    return vm


def admittance_reflect(v_plus, poles, Yr, Y0):
    """Velocity waves reflected by a passive K x K bridge admittance Y(z) = sum_r Yr H_r(z), H_r = (1 - z^-2) /
    ((1 - p_r z^-1)(1 - conj(p_r) z^-1)), each Yr positive semidefinite: v- = (Y + Y0)^-1 (Y - Y0) v+, computed without a
    delay-free loop by splitting Y = Yi + z^-1 Yp. ``v_plus`` (n, K) incident waves, ``poles`` (R,) complex, ``Yr`` (R, K, K),
    ``Y0`` (K, K) the string's characteristic admittance."""
    p = np.asarray(poles, complex)
    a1, a2 = -2.0 * p.real, np.abs(p) ** 2
    Yr = np.ascontiguousarray(np.asarray(Yr, float))
    w, V = np.linalg.eigh(Yr)
    Yr = np.ascontiguousarray(np.einsum("rij,rj,rkj->rik", V, np.clip(w, 0.0, None), V))
    vp = np.ascontiguousarray(np.asarray(v_plus, float).reshape(len(v_plus), -1))
    return _admittance_reflect_kernel(vp, -a1, -1.0 - a2, a1, a2, Yr, Yr.sum(axis=0), np.ascontiguousarray(np.asarray(Y0, float)))


def pluck_scattering(M: float, R: float, K: float, Z0: float, sr: float):
    """Finger (or plectrum) on the string as a mass-damper-spring load E(s) = M s^2 + R s + K between two string segments of
    wave impedance Z0: Q(s) = (2 Z0 s - E(s)) / (2 Z0 s + E(s)), bilinear-transformed so |Q| <= 1 at every frequency
    whatever M, R, K >= 0. Scattering S = [[1 + Q, Q - 1], [Q - 1, 1 + Q]] / 2 (det S = Q); a force F0 enters both rails
    through (1 + Q) / (4 Z0). Returns (b, a) of Q(z)."""
    c = 2.0 * float(sr)
    num = np.array([-M, 2.0 * Z0 - R, -K])               # s^2, s, 1
    den = np.array([M, 2.0 * Z0 + R, K])
    def bil(p):
        return np.array([p[0] * c * c + p[1] * c + p[2], -2 * p[0] * c * c + 2 * p[2], p[0] * c * c - p[1] * c + p[2]])
    b, a = bil(num), bil(den)
    return b / a[0], a / a[0]


# ---------------------------------------------------------------- string loss and nonlinearity laws (V4)
def loss_fir(f0: float, tau, sr: int, ntaps: int | None = None, fk=None) -> np.ndarray:
    """Linear-phase FIR loop filter whose magnitude at the partials ``fk`` (default k f0) is the per-period loss
    exp(-1 / (f0 tau_k)): least squares (firls) weighted by 1 / (1 - g)^4, so the error is even in decay time
    rather than in gain; 17 taps for the lowest strings (f0 < 200 Hz), 9 above."""
    from scipy.signal import firls
    tau = np.atleast_1d(np.asarray(tau, float))
    fk = np.arange(1, tau.size + 1) * f0 if fk is None else np.asarray(fk, float)
    n = int(ntaps or (17 if f0 < 200.0 else 9))
    n += 1 - n % 2
    g = np.exp(-1.0 / (f0 * tau))
    nyq = 0.5 * sr
    fr = np.concatenate([[0.0], fk[fk < nyq * 0.98], [nyq]])
    gr = np.concatenate([[g[0]], g[fk < nyq * 0.98], [g[fk < nyq * 0.98][-1] if np.any(fk < nyq * 0.98) else g[-1]]])
    bands = np.repeat(fr, 2)[1:-1]
    des = np.repeat(gr, 2)[1:-1]
    w = 1.0 / np.maximum(1.0 - 0.5 * (gr[:-1] + gr[1:]), 1e-6) ** 4
    return firls(n, bands, des, weight=w / w.max(), fs=sr)


PIANO_LOSS = (4.4e-3, -4e-2, 1.0e-6, 1e-5)     # b1 = a f0 + b (1/s), b2 = c f0 + d (m^2/s)


def piano_loss_coeffs(f0: float) -> tuple[float, float]:
    """(b1, b2) of a piano string at f0 (Disklavier C6, mf): b1 = 4.4e-3 f0 - 4e-2, b2 = 1.0e-6 f0 + 1e-5."""
    a, b, c, d = PIANO_LOSS
    return max(a * f0 + b, 0.05), c * f0 + d


def piano_decay_rate(f, f0: float, B: float, L: float) -> np.ndarray:
    """Amplitude decay rate (1/s, positive) at f of the damped stiff string:
    -sigma = b1 + b2 pi^2 / (2 B L^2) (-1 + sqrt(1 + 4 B (w/w0)^2)); B -> 0 gives b1 + b2 (w/c)^2."""
    b1, b2 = piano_loss_coeffs(f0)
    r = np.asarray(f, float) / f0
    B = max(float(B), 1e-12)
    return b1 + b2 * math.pi ** 2 / (2.0 * B * L * L) * (-1.0 + np.sqrt(1.0 + 4.0 * B * r * r))


def loop_filter_closed_form(f, f0: float, b1: float, b2: float, B: float, L: float):
    """Waveguide loop filter F(w) that a damped stiff string implies: D = pi / w0 (one period over two),
    xi = -1 + sqrt(1 + 4 B w^2 / w0^2), |F| = exp(-D [b1 + b2 pi^2 xi / (2 B L^2)]), arg F = w D - pi sqrt(xi / (2B)).
    Returns (|F|, arg F)."""
    w = 2 * math.pi * np.asarray(f, float)
    w0 = 2 * math.pi * f0
    D = math.pi / w0
    B = max(float(B), 1e-12)
    xi = -1.0 + np.sqrt(1.0 + 4.0 * B * w * w / (w0 * w0))
    return np.exp(-D * (b1 + b2 * math.pi ** 2 * xi / (2.0 * B * L * L))), w * D - math.pi * np.sqrt(xi / (2.0 * B))


def loop_filter_from_partials(freqs, alphas):
    """Inverse problem: partial frequencies f_n (Hz) and amplitude decay rates alpha_n (1/s) of a string give the
    loop delay D = 2 pi / (w2 - w1), |F(w_n)| = exp(-alpha_n D) and arg F(w_n) = w_n D - 2 pi n. Returns (D, |F|, arg F)."""
    f = np.asarray(freqs, float)
    a = np.asarray(alphas, float)
    w = 2 * math.pi * f
    D = 2 * math.pi / (w[1] - w[0])
    n = np.arange(1, f.size + 1)
    return D, np.exp(-a * D), w * D - 2 * math.pi * n


NL_CL, NL_CT = 0.25, 10.0


def nonlinear_regime(ES: float, T0: float, slope: float = 1e-2) -> dict:
    """Which geometric nonlinearities a string needs at a slope |y'| (1e-2 = a fortissimo hammer): longitudinal force
    ~ Cl ES y'^2, nonlinear transverse ~ Ct ES y'^3, each against the linear transverse T0 y' (dB); a term counts above
    -20 dB. Also f_xi0 / f0 = sqrt(ES / T0) (3-5 nylon, 10-20 metal)."""
    lin = T0 * slope
    fl = NL_CL * ES * slope ** 2
    ft = NL_CT * ES * slope ** 3
    l_db, t_db = 20 * math.log10(fl / lin), 20 * math.log10(ft / lin)
    return {"long_db": l_db, "trans_db": t_db, "longitudinal": l_db > -20.0, "transverse": t_db > -20.0,
            "ratio": math.sqrt(ES / T0)}


def stepped_string_partials(segments, T: float, EI: float, n_modes: int = 12, N: int = 600) -> np.ndarray:
    """Partial frequencies (Hz) of a stiff string made of segments [(length m, mu kg/m), ...] (bare core, winding,
    bare core...), pinned ends: -T u'' + EI u'''' = w^2 mu(x) u by finite differences."""
    from scipy.linalg import eigh
    Ltot = sum(a for a, _ in segments)
    h = Ltot / (N + 1)
    x = (np.arange(N) + 1) * h
    edges = np.cumsum([a for a, _ in segments])
    mu = np.array([segments[min(int(np.searchsorted(edges, xi)), len(segments) - 1)][1] for xi in x])
    D2 = (np.diag(-2.0 * np.ones(N)) + np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1)) / h ** 2
    D4 = D2 @ D2
    D4[0, 0] += 1.0 / h ** 4
    D4[-1, -1] += 1.0 / h ** 4                                       # pinned ends: u'' = 0 (odd extension)
    K = -T * D2 + EI * D4
    w2 = eigh(K, np.diag(mu), eigvals_only=True, subset_by_index=[0, n_modes - 1])
    return np.sqrt(np.maximum(w2, 0.0)) / (2 * math.pi)


def fit_B(fn) -> float:
    """B of f_n = n f1' sqrt(1 + B n^2) fitted to partials (least squares on (f_n / n)^2 = f0^2 + f0^2 B n^2)."""
    fn = np.asarray(fn, float)
    n = np.arange(1, fn.size + 1)
    A = np.vstack([np.ones(fn.size), n * n]).T
    c, k = np.linalg.lstsq(A, (fn / n) ** 2, rcond=None)[0]
    return float(k / c)


def wound_bare_ends_B(f0: float, core_d: float, bare_m: float, wound_m: float, mu_core: float, mu_wound: float,
                      E: float = 2.0e11, n_modes: int = 12) -> float:
    """Inharmonicity B of a wound string whose core runs bare over ``bare_m`` at each end of the ``wound_m`` winding:
    tension set for f0, stiffness of the core alone, partials of the stepped string fitted to n sqrt(1 + B n^2)."""
    EI = E * math.pi * core_d ** 4 / 64.0
    seg = [(bare_m, mu_core), (wound_m, mu_wound), (bare_m, mu_core)]
    T = 1.0
    for _ in range(3):
        f1 = stepped_string_partials(seg, T, EI, 2)[0]
        T *= (f0 / f1) ** 2
    return fit_B(stepped_string_partials(seg, T, EI, n_modes))


@numba.njit(cache=True)
def _whirl(n, sr, w, s0, a0, b0, va0, vb0, kk2L, alpha2):
    """Two planes of modal amplitudes a_k, b_k sharing the Kirchhoff tension G = 1 + alpha^2 / 2 |u_x|^2,
    |u_x|^2 = sum (k pi / L)^2 L / 2 (a_k^2 + b_k^2); symplectic Euler. Returns the bridge-end slope sums (x, y)."""
    m = len(w)
    a = a0.copy(); b = b0.copy(); va = va0.copy(); vb = vb0.copy()
    dt = 1.0 / sr
    ox = np.zeros(n); oy = np.zeros(n)
    for i in range(n):
        e = 0.0
        for k in range(m):
            e += kk2L[k] * (a[k] * a[k] + b[k] * b[k])
        G = 1.0 + 0.5 * alpha2 * e
        sx = 0.0; sy = 0.0
        for k in range(m):
            va[k] += dt * (-w[k] * w[k] * G * a[k] - 2.0 * s0 * va[k])
            vb[k] += dt * (-w[k] * w[k] * G * b[k] - 2.0 * s0 * vb[k])
            a[k] += dt * va[k]
            b[k] += dt * vb[k]
            sx += (k + 1) * a[k]
            sy += (k + 1) * b[k]
        ox[i] = sx; oy[i] = sy
    return ox, oy


def whirling_string(freq: float, dur: float, sr: int, amp: float = 0.01, ellipse: float = 0.3, alpha: float = 30.0,
                    t60: float = 3.0, pos: float = 0.2, modes: int = 20):
    """Non-planar (whirling) Kirchhoff-Carrier string: u_tt = gamma^2 G u_xx - 2 sigma0 u_t with
    G = 1 + alpha^2 / 2 |u_x|^2 over both transverse planes, so a pluck with an out-of-plane velocity (``ellipse``,
    0 = planar) precesses. Unit length; ``amp`` initial displacement at ``pos``. Returns (x plane, y plane)."""
    k = np.arange(1, modes + 1, dtype=float)
    k = k[k * freq < 0.2 * sr]                        # symplectic Euler: w dt < 2 even with G > 1
    w = 2 * math.pi * freq * k
    shape = np.sin(k * math.pi * pos) / (k * k * pos * (1 - pos) * math.pi ** 2) * 2.0
    a0 = amp * shape
    b0 = np.zeros_like(a0)
    vb0 = float(ellipse) * w * a0
    kk2L = (k * math.pi) ** 2 * 0.5
    n = int(dur * sr)
    x, y = _whirl(n, float(sr), w, 6.91 / t60, a0, b0, np.zeros_like(a0), vb0, kk2L, float(alpha) ** 2)
    return x, y


def octave_stretch_cents(midi_lo: float, midi_hi: float, m: int, n: int) -> float:
    """Cents the upper key sits above a pure octave when its n-th partial is tuned beatless to the lower key's m-th:
    600 log2[(1 + m^2 B_lo) / (1 + n^2 B_hi)] + 1200 log2(m / 2n) (0 for m = 2n), Steinway D inharmonicity."""
    from .physical.piano_data import steinway_B
    Bl, Bh = float(steinway_B(midi_lo)), float(steinway_B(midi_hi))
    return 600.0 * math.log2((1.0 + m * m * Bl) / (1.0 + n * n * Bh)) + 1200.0 * math.log2(m / (2.0 * n))


OCTAVE_MATCH = ((12, (10, 5)), (20, (8, 4)), (28, (6, 3)), (40, (4, 2)), (78, (2, 1)), (88, (4, 2)))


def _octave_mn(key: float) -> tuple[int, int]:
    return next(v for top, v in OCTAVE_MATCH if key <= top) if key <= 88 else (4, 2)


@functools.lru_cache(maxsize=128)
def matched_tuning_cents(midi: int) -> float:
    """Tuning curve (cents from equal temperament, A4 = 0) built octave by octave from A4, each octave tuned with
    the m:n partial match of its lower key's register (keys 1-12 10:5, 13-20 8:4, 21-28 6:3, 29-40 4:2, 41-78 2:1,
    79-88 4:2); the octaves next to A4 spread evenly over their semitones."""
    k = int(round(midi)) - 20
    if 37 <= k <= 61:
        if k >= 49:
            return octave_stretch_cents(69, 81, *_octave_mn(49)) * (k - 49) / 12.0
        return -octave_stretch_cents(57, 69, *_octave_mn(37)) * (49 - k) / 12.0
    if k > 61:
        return matched_tuning_cents(midi - 12) + octave_stretch_cents(midi - 12, midi, *_octave_mn(k - 12))
    return matched_tuning_cents(midi + 12) - octave_stretch_cents(midi, midi + 12, *_octave_mn(k))
