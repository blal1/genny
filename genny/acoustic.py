"""Acoustic instruments: more keys, mallets and bells, plucked and bowed strings, winds and brass.

Registered into the instrument REGISTRY (imported at the bottom of instruments.py). Three engines:
  - `_string` (tuned Karplus-Strong) for everything plucked or struck on a string,
  - `_modes` (rolled-off inharmonic partials) for bars, bells, bowls and drums with a pitch,
  - `_blown` (a band-limited harmonic stack shaped by fixed body formants, whose brightness opens
    with the attack and velocity) for bowed strings, reeds, flutes and brass.
All three follow the voicing rules at the top of instruments.py: brightness is set in Hz, not in
harmonics, so nothing turns piercing as it climbs.
"""
from __future__ import annotations

import math

import numba
import numpy as np

from . import filters as F
from . import osc as O
from .core import DEFAULT_SR, mix
from .env import adsr, apply, lfo, perc
from .instruments import (_cut, _harmonics, _ks, _kt, _modes, _n, _off, _roll, _spectrum, _string, _vibrato,
                          decay_env, instrument)


# ---------------------------------------------------------------- keys
@instrument("wurli", "Wurlitzer-style reed electric piano: hollow and a little nasal, growls when hit hard (soul, indie, lo-fi).",
            family="keys", span=("C2", "C6"), tremolo=(0.3, "0..1 wobble"), decay=(1.4, "seconds"))
def wurli(freq, dur, sr=DEFAULT_SR, vel=1.0, tremolo=0.3, decay=1.4):
    ring = min(decay, dur + 0.8)
    n = _n(ring, sr)
    # Pfeifle (DAFx-17 5.5-5.6): the reed moves in its first cantilever mode only, near-sinusoidal even when hit
    # hard; the pickup is a capacitor whose C goes as 1 / gap, read as the current i = u0 dC/dt. A reed at swing a
    # (in units of the rest gap) gives i ~ a w cos(wt) / (1 - a sin(wt))^2: pure when soft, lopsided and rich (even
    # and odd harmonics) when the swing nears the gap. UNSOURCED: the swing per velocity and its 0.22 s decay.
    a = np.clip((0.2 + 0.55 * _kt(freq, 1.0, 0.6) * vel) * (0.35 + 0.65 * decay_env(n, 0.22, sr)), 0.0, 0.85)
    c = 1.0 / (1.0 - a * O.sine(freq, n, sr))
    x = np.gradient(c) * sr / (2 * np.pi * freq) * 1.05
    x = F.lowpass(F.dc_block(x, sr), _cut(freq, 5, 1200, 3000), sr, 0.8, order=2)
    env = decay_env(n, ring / 2.5, sr) * _off(n, dur, sr, 0.06) * perc(ring, 0.002, sr, curve=0.6)
    if tremolo > 0:
        env = env * (1 - tremolo * 0.5 * (1 + lfo(n / sr, 5.8, sr)[:n]))
    return x * env                                    # the swing a already carries the velocity


@instrument("clav", "Clavinet: short, funky, nasal struck string (funk, disco, reggae skank).",
            family="keys", span=("C2", "C6"), decay=(0.9, "seconds"), bright=(1.0, "0.5 muted .. 1.5 biting"),
            release_drop=(1.0, "0..1 the brief 3-semitone drop when the key is let go and the yarn damps the string"))
def clav(freq, dur, sr=DEFAULT_SR, vel=1.0, decay=0.9, bright=1.0, release_drop=1.0):
    """Hohner D6 measurements of Gabrielli et al., EURASIP J. Adv. Signal Process. 2013:103:
    - magnetic pickup near the termination: a comb with notches every 4.67 partials (544 Hz on A#2, their
      Fig. 3b), H = 1 - z^-(pP) with p = 1/4.67, which also makes partial 2 louder than partial 1 (their 2.3.4);
    - at key-off the speaking and non-speaking parts of the string join under the yarn damper: the pitch
      drops three semitones for a moment, on every key (2.3.1);
    - amplifier without the tone stack: -3 dB shelf at 130 Hz, +3 dB at 4 kHz (2.3.6)."""
    t60 = decay * 1.5 * _kt(freq, 1.0, 0.45)
    ring = min(t60 * 0.9, dur + 0.25)
    y = _string(freq, ring, sr, vel, t60=decay * 1.5, bright=3400 * bright, damp=0.15, pick=0.1, grit=0.1, ring=ring)
    n0 = min(len(y), int(round(dur * sr)))
    t = np.arange(len(y) - n0) / sr
    off = np.exp(-t / 0.03)                                     # the damper (the old 30 ms release)
    held = y[n0:] * off
    k = float(np.clip(release_drop, 0.0, 1.0))
    if k > 0 and len(t) > 1:                                    # read the ringing string 3 semitones slower
        src = np.arange(len(y)) - n0
        dropped = np.interp(src[n0:] * 2 ** (-3 / 12), np.arange(len(y)) - n0, y) * off
        held = (1 - k) * held + k * dropped
    y = np.concatenate([y[:n0], held])
    p = 1.0 / 4.67
    d = p * sr / freq                                           # pickup comb delay, samples
    y = y - np.interp(np.arange(len(y)) - d, np.arange(len(y)), y, left=0.0)
    y = F.highshelf(F.lowshelf(y, 130.0, sr, -3.0), 4000.0, sr, 3.0)
    y = F.highpass(y, max(120.0, freq * 0.6), sr)
    return F.lowpass(y, 4200, sr, 0.7) * 0.8 / (np.max(np.abs(y)) + 1e-9) * (0.5 + 0.5 * vel)


# Rhodes Mark I tine/tonebar Q (Shear, MSc UCSB 2011, Table 2.1): amplitude time constant tau = Q / (pi f0)
_RHODES_Q = ((77.8, 949.0), (155.6, 731.0), (311.1, 1520.0), (622.3, 2175.0), (1244.5, 1761.0))


@instrument("rhodes", "Rhodes tine piano: a struck cantilever tine read by a magnetic pickup whose field shapes the tone (bell attack, bark when played hard).",
            family="keys", span=("E1", "E6"),
            voicing=(0.35, "0..1 tine offset from the pickup axis: 0 = centred (2nd harmonic leads, hollow), 1 = far off (round, fundamental)"),
            bell=(1.0, "0..2 level of the short inharmonic tine overtones in the attack"),
            decay=(1.0, "x the measured sustain (Q of a 1974 Mark I)"))
def rhodes(freq, dur, sr=DEFAULT_SR, vel=1.0, voicing=0.35, bell=1.0, decay=1.0):
    """Tine: cantilever modes 1, 6.267, 17.55 (clamped-free Euler-Bernoulli); the fundamental rings with the
    measured Q (Shear 2011, Table 2.1), the overtones die within tens of ms ("short lived ... bell-like timbre
    during the attack", Shear 2.1.2). Pickup: the tine tip at x (in units of its distance d to the pickup tip)
    over a magnetic point charge, flux ~ (x^2 + 1)^(-3/2) (Pfeifle, DAFx-17 eq. 6), voltage = -d(flux)/dt
    (Faraday). Centred (voicing 0) the flux is even in x, so odd partials cancel and the second leads (Shear
    2.2.1, Figs 2.4-2.5); harder blows swing further and bark.
    UNSOURCED: overtone levels and decays, tine amplitude per velocity, the 60 ms damper."""
    f0 = float(freq)
    q = float(np.exp(np.interp(np.log(f0), np.log([a for a, _ in _RHODES_Q]), np.log([b for _, b in _RHODES_Q]))))
    tau = max(q / (math.pi * f0) * float(np.clip(decay, 0.05, 4.0)), 0.05)
    ring = min(dur + 1.5, 6.91 * tau)
    n = _n(ring, sr)
    t = np.arange(n) / sr
    b = float(np.clip(bell, 0.0, 2.0))
    u = np.sin(2 * math.pi * f0 * t) * np.exp(-t / tau)
    for r, a, d in ((6.267, 0.25 * b, 0.04), (17.55, 0.08 * b, 0.015)):
        if r * f0 < 0.45 * sr:
            u = u + a * np.sin(2 * math.pi * r * f0 * t) * np.exp(-t / d)
    u = u * np.minimum(1.0, t / 0.0008)                                 # hammer contact
    u = u * _off(n, dur, sr, 0.06)
    amp = 0.25 + 0.95 * float(np.clip(vel, 0.0, 1.0))                   # tine swing / pickup distance
    x = amp * u + 1.2 * float(np.clip(voicing, 0.0, 1.0))
    flux = (x * x + 1.0) ** -1.5
    y = -np.diff(flux, prepend=flux[0]) * sr
    y = F.dc_block(y, sr)
    y = F.lowpass(y, min(5000.0, 0.45 * sr), sr, 0.7)
    return y / (np.max(np.abs(y)) + 1e-9) * (0.55 + 0.45 * vel)


# Harpsichord after Valimaki, Penttinen, Knif, Laurson & Erkut, "Sound synthesis of the harpsichord using a
# computationally efficient physical model", EURASIP JASP 2004(7):934. String loop: delay + first-order Thiran
# fraction + dispersion allpasses + the "ripply" loss filter H(z) = g (1 + a) (r + z^-R) / (1 + a z^-1),
# R = round(rrate L) (eqs 1-3), which lets neighbouring partials decay at different rates. A second-order
# resonator a fraction of a hertz off one low partial, at its level, gives the beating (4.4). The soundboard is a
# reverberator with T60 6 s at DC and 0.13 of that at Nyquist on 8 lines of 1009-1999 samples, high-passed at
# 350 Hz (4.5). Their calibrated values for the 197 Hz key (Fig. 8): g 0.996, a -0.0296, r -0.0015, rrate 1/12.
# UNSOURCED: those values used on every key with g from a key-tracked T60, the quill excitation (a short pulse
# plus scrape noise; theirs are recorded residuals), the beating offset and partial, the release knock (theirs
# is a recorded sample) and the soundboard level.

@numba.njit(cache=True)
def _harpsi_loop(exc, n, n_off, L, frac_a, g, a, r, R, disp, damp_g):
    """Delay L -> Thiran fraction -> dispersion -> g (1 + a) / (1 + a z^-1) -> (r + z^-R), fed back; the
    loop gain drops to g * damp_g from sample n_off (the damper)."""
    size = L + 2
    buf = np.zeros(size)
    hist = np.zeros(R + 1)                                         # one-pole output history for z^-R
    out = np.zeros(n)
    dx = np.zeros(len(disp)); dy = np.zeros(len(disp))
    tx = 0.0; ty = 0.0; lp = 0.0
    w = 0; h = 0
    for i in range(n):
        v = buf[(w - L) % size]
        x = frac_a * v + tx - frac_a * ty; tx = v; ty = x
        for k in range(len(disp)):
            o = disp[k] * x + dx[k] - disp[k] * dy[k]; dx[k] = x; dy[k] = o; x = o
        gi = g if i < n_off else g * damp_g
        lp = gi * (1.0 + a) * x - a * lp
        hist[h] = lp
        y = r * lp + hist[(h - R) % (R + 1)]
        h = (h + 1) % (R + 1)
        out[i] = y
        buf[w] = y + (exc[i] if i < len(exc) else 0.0)
        w = (w + 1) % size
    return out


def _harpsi_string(freq, n, n_off, sr, exc, t60, B, a=-0.0296, r=-0.0015, rrate=1.0 / 12.0):
    """One harpsichord string tuned to ``freq``: the delay is one period minus the phase delay of every loop
    filter at f0 (one-pole, ripple comb, dispersion), the rest set by the Thiran fraction."""
    from .strings import rauhala_dispersion
    w0 = 2 * math.pi * freq / sr
    P = sr / freq
    R = max(1, int(round(rrate * P)))
    g = 10 ** (-3.0 / (max(t60, 0.05) * freq))
    disp = np.array(rauhala_dispersion(freq, B, sr), float)
    z = np.exp(-1j * w0)
    H = (1 + a) / (1 + a * z) * (r + z ** R)
    for c in disp:
        H *= (c + z) / (1 + c * z)
    total = P + np.angle(H) / w0                                    # angle < 0 = lag; remove it from the line
    L = int(math.floor(total - 0.5))
    fr = total - L                                                  # Thiran fraction kept in [0.5, 1.5)
    fr_a = math.sin((1.0 - fr) * w0 / 2.0) / math.sin((1.0 + fr) * w0 / 2.0)   # exact phase delay at f0
    return _harpsi_loop(exc, n, n_off, max(L, 2), fr_a, g, a, r, R, disp, 0.35 ** (1.0 / max(P, 1.0)))


@instrument("harpsichord", "Harpsichord (Valimaki et al. 2004): quill-plucked strings with ripple loss filter, beating partial and a ringing soundboard; 4-foot octave rank.",
            family="keys", span=("C2", "C6"), octave=(0.3, "level of the 4-foot octave rank (0 = 8-foot only)"),
            soundboard=(0.35, "ringing soundboard and undamped strings behind the bridge 0..1"))
def harpsichord(freq, dur, sr=DEFAULT_SR, vel=1.0, octave=0.3, soundboard=0.35):
    from scipy.signal import cheby1, lfilter, sosfilt
    from .reverbs import fdn_reverb
    ring = dur + 0.25
    n, n_off = _n(ring, sr), int(round(dur * sr))
    rng = np.random.default_rng(int(freq * 10) % 9973)
    m = max(4, int(0.0006 * sr))                                    # quill: sharp release + scrape
    pulse = np.concatenate([np.linspace(0, 1, m), np.linspace(1, -0.3, m // 2)]) * 0.6
    scrape = F.highpass(rng.standard_normal(int(0.012 * sr)), 1500, sr) * np.exp(-np.arange(int(0.012 * sr)) / (0.003 * sr)) * 0.08
    exc = np.zeros(max(len(pulse), len(scrape)))
    exc[:len(pulse)] += pulse
    exc[:len(scrape)] += scrape
    lag = max(1, int(round(0.12 * sr / freq)))                      # plucked near the end: comb at 0.12
    exc = np.concatenate([exc, np.zeros(lag)]) - np.concatenate([np.zeros(lag), exc])
    t60 = _kt(freq, 9.0, 2.5, f_lo=65.0, f_hi=1200.0)
    B = 1e-5 * _kt(freq, 1.0, 0.3, f_lo=65.0, f_hi=1200.0)          # Fig. 5: ~1e-5 on the low keys
    y = _harpsi_string(freq, n, n_off, sr, exc, t60, B)
    k = 10 if 10 * freq < 0.4 * sr else 2                           # beating against a low partial
    fb = k * freq * math.sqrt(1 + B * k * k) + 0.67
    rr = math.exp(-math.pi * fb / (math.sqrt(1 + B * k * k) * freq * t60 * 2.0))
    th = 2 * math.pi * fb / sr
    beat = lfilter([1.0 - rr], [1.0, -2 * rr * math.cos(th), rr * rr], np.pad(exc, (0, n - len(exc))))[:n]
    yk = F.bandpass(y, k * freq, sr, 30.0)
    beat *= np.sqrt(np.mean(yk ** 2)) / (np.sqrt(np.mean(beat ** 2)) + 1e-12)
    y = y + beat
    if octave > 0 and 2 * freq < 0.4 * sr:
        y4 = _harpsi_string(2 * freq, n, n_off, sr, exc, 0.7 * t60, B)
        y = y + float(octave) * y4
    knock = F.lowpass(rng.standard_normal(int(0.03 * sr)), 900, sr) * np.exp(-np.arange(int(0.03 * sr)) / (0.006 * sr)) * 0.05
    y[n_off:n_off + len(knock)] += knock[:max(0, n - n_off)]
    if soundboard > 0:
        wet = fdn_reverb(y, sr, t60_low=6.0, t60_high=2.5, size=12.0, predelay=0.0, mix=1.0, lines=8, tail=0.0)
        wet = wet.mean(axis=1) if wet.ndim == 2 else wet
        wet = sosfilt(cheby1(5, 5.0, 350.0, "highpass", fs=sr, output="sos"), wet[:n])   # their tone corrector
        y = y + float(soundboard) * wet * np.sqrt(np.mean(y ** 2)) / (np.sqrt(np.mean(wet ** 2)) + 1e-12)
    # their tone corrector is also a lowpass at high frequencies (4.5); genny's 4.5 kHz voicing ceiling
    y = F.lowpass(y, min(4500.0, 0.42 * sr), sr, 0.7, order=2)
    return y / (np.max(np.abs(y)) + 1e-9) * 0.8


@instrument("celesta", "Celesta: felt hammers on steel plates over wooden resonators; a soft, round bell (lullabies, fairy music).",
            family="mallet", span=("C4", "C8"))
def celesta(freq, dur, sr=DEFAULT_SR, vel=1.0):
    ring = max(dur, 1.3 * _kt(freq, 1.0, 0.6, f_lo=523.0, f_hi=4186.0))
    n = _n(ring, sr)
    x = _modes(freq, n, sr, [(1.0, 1.0, ring / 2.5), (2.0, 0.2, ring / 5), (5.1, 0.14 * vel, 0.04)], corner=3000.0)
    return x * perc(ring, 0.002, sr, curve=1.0) * 0.75 * (0.6 + 0.4 * vel)


@instrument("toy_piano", "Toy piano: plastic hammers on metal rods; plinky, childlike, a little out of true.",
            family="mallet", span=("C4", "C7"))
def toy_piano(freq, dur, sr=DEFAULT_SR, vel=1.0):
    ring = max(dur, 0.7)
    n = _n(ring, sr)
    x = _modes(freq, n, sr, [(1.0, 1.0, 0.3), (2.93, 0.3, 0.1), (6.27, 0.2 * vel, 0.03)], corner=3000.0)
    click = F.bandpass(O.white(n, seed=21), _cut(freq, 4, 1500, 3000), sr, 3.0) * decay_env(n, 0.004, sr) * 0.25 * vel
    return (x + click) * perc(ring, 0.001, sr, curve=1.0) * 0.75 * (0.6 + 0.4 * vel)


@instrument("church_organ", "Pipe organ: principal ranks at several octaves that speak slowly with a little chiff (add a long reverb).",
            family="keys", span=("C2", "C6"), sub=(0.35, "16-foot sub-octave rank 0..1"), mixture=(0.5, "upper ranks 0..1"))
def church_organ(freq, dur, sr=DEFAULT_SR, vel=1.0, sub=0.35, mixture=0.5):
    rel = 0.2
    n = _n(dur, sr, rel)
    x = np.zeros(n)
    for mult, g, tune in ((0.5, sub, 1.0003), (1, 1.0, 1.0), (2, 0.45, 0.9996), (3, 0.3 * mixture, 1.0004), (4, 0.4 * mixture, 1.0005)):
        f = freq * mult
        if g <= 0 or f > 5000:
            continue
        x += g * _harmonics(f * tune, n, sr, _spectrum(f, sr, tilt=1.7, corner=2400.0, slope=2.5, top=5000.0, maxh=10))
    chiff = F.bandpass(O.white(n, seed=31), _cut(freq, 3, 600, 2400), sr, 2.0) * decay_env(n, 0.03, sr) * 0.12
    return apply(x / 1.6 + chiff, adsr(dur, 0.05, 0.0, 1.0, rel, sr, curve=1.2)) * 0.6


@instrument("accordion", "Accordion: two or three reeds tuned slightly apart (musette shimmer) under bellows pressure (folk, tango, polka, chanson).",
            family="keys", span=("C3", "C6"), musette=(9.0, "cents the reeds are tuned apart (0 = dry single reed)"))
def accordion(freq, dur, sr=DEFAULT_SR, vel=1.0, musette=9.0):
    rel = 0.08
    n = _n(dur, sr, rel)
    amps = _spectrum(freq, sr, tilt=1.0, formants=[(900, 400, 0.8)], corner=2200.0, slope=2.5, top=5500.0, even=0.7)
    reeds = ((0.0, 1.0), (musette, 0.7), (-0.8 * musette, 0.5)) if musette else ((0.0, 1.0),)
    x = sum(g * _harmonics(freq * 2 ** (c / 1200.0), n, sr, amps) for c, g in reeds) / 1.5
    t = np.arange(n) / sr
    x = x * (1 + 0.05 * np.sin(2 * np.pi * 0.9 * t))   # bellows
    return apply(x, adsr(dur, 0.045, 0.1, 0.9, rel, sr, curve=1.3)) * 0.6 * (0.5 + 0.5 * vel)


# ---------------------------------------------------------------- mallets / bells / pitched drums
@instrument("xylophone", "Xylophone: hard mallets on short wooden bars; dry, bright knock (cartoons, puzzles).",
            family="mallet", span=("F4", "C8"))
def xylophone(freq, dur, sr=DEFAULT_SR, vel=1.0):
    ring = max(dur, 0.4 * _kt(freq, 1.2, 0.7, f_lo=350.0, f_hi=4186.0))
    n = _n(ring, sr)
    x = _modes(freq, n, sr, [(1.0, 1.0, ring / 3), (3.0, 0.35 * vel, ring / 8), (6.0, 0.12 * vel, ring / 18)], corner=3000.0)
    click = F.bandpass(O.white(n, seed=22), _cut(freq, 3, 1200, 2600), sr, 1.0) * decay_env(n, 0.003, sr) * 0.18 * vel
    return (x + click) * perc(ring, 0.0008, sr, curve=1.0) * 0.8 * (0.6 + 0.4 * vel)


@instrument("glockenspiel", "Glockenspiel: small steel bars; a clear, sweet ring for high sparkle lines.",
            family="mallet", span=("G5", "C8"))
def glockenspiel(freq, dur, sr=DEFAULT_SR, vel=1.0):
    ring = max(dur, 1.4 * _kt(freq, 1.0, 0.6, f_lo=784.0, f_hi=4186.0))
    n = _n(ring, sr)
    x = _modes(freq, n, sr, [(1.0, 1.0, ring / 2.5), (2.76, 0.28 * vel, ring / 8), (5.4, 0.1 * vel, 0.05)], corner=3400.0, beat=0.8)
    return x * perc(ring, 0.001, sr, curve=1.0) * 0.7 * (0.6 + 0.4 * vel)


@instrument("handpan", "Handpan / hang drum: a hand-struck steel dome ringing its octave and fifth (ambient, meditative, world).",
            family="mallet", span=("C3", "C5"))
def handpan(freq, dur, sr=DEFAULT_SR, vel=1.0):
    ring = max(dur, 1.8)
    n = _n(ring, sr)
    x = _modes(freq, n, sr, [(1.0, 1.0, 1.0), (2.0, 0.5, 0.75), (3.0, 0.3 * vel, 0.5), (4.16, 0.06 * vel, 0.1)], corner=2500.0, beat=0.7)
    thump = F.lowpass(O.white(n, seed=23), 320, sr) * decay_env(n, 0.012, sr) * 0.6 * vel
    return (x + thump) * perc(ring, 0.003, sr, curve=1.0) * 0.75 * (0.6 + 0.4 * vel)


@instrument("tubular_bell", "Tubular bell / orchestral chime: a tall inharmonic clang whose partials imply the note (clock towers, Christmas, drama).",
            family="mallet", span=("C4", "G5"), decay=(3.5, "seconds"))
def tubular_bell(freq, dur, sr=DEFAULT_SR, vel=1.0, decay=3.5):
    ring = max(dur, decay)
    n = _n(ring, sr)
    x = _modes(freq, n, sr, [(0.615, 0.25, ring / 2.5), (1.0, 0.3, ring / 2.5), (1.21, 0.45, ring / 3), (2.0, 1.0, ring / 3.5),
                             (2.99, 0.8 * vel, ring / 5), (4.17, 0.5 * vel, ring / 8), (5.55, 0.25 * vel, ring / 12)],
               corner=3200.0, beat=0.5)
    return x * perc(ring, 0.002, sr, curve=1.2) * 0.5 * (0.6 + 0.4 * vel)


@instrument("gamelan", "Gamelan metallophone (saron/gender): bronze bars tuned in beating pairs, so every note shimmers.",
            family="mallet", span=("C4", "C7"), ombak=(3.0, "beat rate Hz between the paired bars"))
def gamelan(freq, dur, sr=DEFAULT_SR, vel=1.0, ombak=3.0):
    ring = max(dur, 2.0)
    n = _n(ring, sr)
    x = _modes(freq, n, sr, [(1.0, 1.0, 1.3), (2.71, 0.4 * vel, 0.5), (5.15, 0.15 * vel, 0.12)], corner=3000.0, beat=ombak)
    return x * perc(ring, 0.001, sr, curve=1.0) * 0.7 * (0.6 + 0.4 * vel)


@instrument("singing_bowl", "Singing bowl: very long ring with slowly beating partials (meditation, ambient, temple).",
            family="mallet", span=("C3", "C5"), decay=(6.0, "seconds"))
def singing_bowl(freq, dur, sr=DEFAULT_SR, vel=1.0, decay=6.0):
    ring = max(dur, decay)
    n = _n(ring, sr)
    x = _modes(freq, n, sr, [(1.0, 1.0, ring / 2.2), (2.71, 0.45, ring / 3.5), (5.0, 0.2 * vel, ring / 7)], corner=2600.0, beat=0.6)
    return x * perc(ring, 0.015, sr, curve=1.3) * 0.7 * (0.6 + 0.4 * vel)


@instrument("timpani", "Timpani / kettle drum: a pitched orchestral drum, boom and roll (write it low: C2-C4).",
            family="mallet", span=("C2", "C4"), decay=(1.6, "seconds"))
def timpani(freq, dur, sr=DEFAULT_SR, vel=1.0, decay=1.6):
    ring = max(dur, decay)
    n = _n(ring, sr)
    x = _modes(freq, n, sr, [(1.0, 1.0, ring / 2.5), (1.5, 0.5, ring / 3.5), (1.99, 0.35, ring / 5), (2.44, 0.2 * vel, ring / 7),
                             (0.63, 0.5, 0.07)], corner=1500.0)
    thump = F.lowpass(O.white(n, seed=24), 260, sr) * decay_env(n, 0.02, sr) * 0.9 * vel
    return np.tanh((x + thump) * 1.2) * perc(ring, 0.002, sr, curve=1.0) * 0.7 * (0.5 + 0.5 * vel)


# ---------------------------------------------------------------- plucked strings
# Modal guitar body: a parallel bank of g (1 - z^-2) / (1 - 2 r cos w z^-1 + r^2 z^-2) sections (Rau & Smith,
# ISMA 2019, eq. 3) on top of the direct string. Low modes in Woodhouse et al.'s bands (Acta Acustica 2012):
# the Helmholtz resonance at 75-100 Hz and the first top-plate mode at 150-250 Hz; between real guitars their
# frequencies spread by ~6 % (s.d.) and their Q by ~33 % (Fig. 7), against hearing thresholds of 1 % and
# 20 %, so a body ``seed`` draws a different instrument with that spread. ``size`` warps every mode through
# Penttinen, Harma & Karjalainen's first-order allpass map (DAFx-00): w' = w + 2 atan(l sin w / (1 - l cos w)),
# l < 0 a bigger box, l > 0 a smaller one. UNSOURCED: the centre values (Helmholtz 98 Hz Q 18, top plate
# 205 Hz Q 28), the higher modes (every ~70 Hz up to 3 kHz, Q 30, level falling as 1 / sqrt(f): "weaker and
# denser", Penttinen) and the size range l = +-0.12.
_GUITAR_LOW = ((98.0, 18.0, 1.0), (205.0, 28.0, 0.8))


def guitar_body_modes(seed: int = 0, size: float = 0.5, sr: int = DEFAULT_SR, steel: bool = False):
    """(freqs Hz, Q, gains) of a guitar body; seed 0 = the nominal body, others = Woodhouse's spread."""
    rng = np.random.default_rng(int(seed) + 1000)
    f = [m[0] for m in _GUITAR_LOW]
    q = [m[1] for m in _GUITAR_LOW]
    g = [m[2] for m in _GUITAR_LOW]
    hf = np.random.default_rng(7 if steel else 5)                   # the higher modes of the nominal body
    fk = 300.0
    while fk < 3000.0:
        fk += 70.0 * hf.uniform(0.5, 1.5)
        f.append(fk); q.append(30.0); g.append(0.35 * math.sqrt(300.0 / fk))
    f, q, g = np.array(f), np.array(q), np.array(g)
    if seed:
        f = f * np.exp(0.058 * rng.standard_normal(len(f)))
        q = q * np.exp(0.33 * rng.standard_normal(len(q)))
    lam = 0.24 * (0.5 - float(np.clip(size, 0.0, 1.0)))
    w = 2 * np.pi * f / 48000.0                                     # warped at Penttinen's 48 kHz whatever sr is
    w = w + 2 * np.arctan(lam * np.sin(w) / (1 - lam * np.cos(w)))
    return w * 48000.0 / (2 * np.pi), q, g


def guitar_body(y, sr, seed=0, size=0.5, steel=False, amount=0.6):
    """The string signal plus its modal body response (gain-matched, ``amount`` of the body)."""
    from scipy.signal import lfilter
    f, q, g = guitar_body_modes(seed, size, sr, steel)
    out = np.zeros_like(y)
    for fk, qk, gk in zip(f, q, g):
        if fk >= 0.45 * sr:
            continue
        r = math.exp(-math.pi * fk / (qk * sr))
        out += gk * (1 - r) * lfilter([1.0, 0.0, -1.0], [1.0, -2 * r * math.cos(2 * math.pi * fk / sr), r * r], y)
    out *= np.sqrt(np.mean(y ** 2)) / (np.sqrt(np.mean(out ** 2)) + 1e-12)
    return y + float(amount) * out


def _body(y, sr, peaks, lowpass=None):
    for fc, q, db in peaks:
        y = F.peak(y, fc, sr, q, db)
    return F.lowpass(y, lowpass, sr, 0.7) if lowpass else y


@instrument("steel_guitar", "Steel-string acoustic guitar: bright strum and fingerpicking (folk, country, pop).",
            family="plucked", span=("E2", "E5"))
def steel_guitar(freq, dur, sr=DEFAULT_SR, vel=1.0):
    y = _string(freq, dur, sr, vel, t60=4.5, bright=3800, damp=0.2, pick=0.16, grit=0.3, release=0.12, tail=1.6)
    return _body(y, sr, [(100, 1.2, 3.0), (200, 1.5, 2.0)], 5000)


@instrument("electric_guitar", "Clean electric guitar: round single-coil tone with long sustain (add `distortion` or `chorus` on the layer for rock or jangle).",
            family="plucked", span=("E2", "E6"), tone=(2800, "pickup tone Hz"))
def electric_guitar(freq, dur, sr=DEFAULT_SR, vel=1.0, tone=2800):
    y = _string(freq, dur, sr, vel, t60=5.5, bright=3200, damp=0.25, pick=0.2, grit=0.15, release=0.08, tail=1.2)
    return F.lowpass(F.highpass(y, 70, sr), tone, sr, 1.3)


@instrument("muted_guitar", "Palm-muted electric guitar: short woody chug (funk scratch, reggae skank, rock eighths).",
            family="plucked", span=("E2", "E5"))
def muted_guitar(freq, dur, sr=DEFAULT_SR, vel=1.0):
    y = _string(freq, min(dur, 0.12), sr, vel, t60=0.4, bright=1900, damp=0.5, pick=0.2, grit=0.25, release=0.03, tail=0.2)
    return F.lowpass(F.highpass(y, 70, sr), 2200, sr, 1.2)


@instrument("dist_guitar", "Overdriven electric guitar for riffs and leads; sustains while held (for crunchy chords play `electric_guitar` and put `distortion` on the layer).",
            family="plucked", span=("E2", "E6"), drive=(6.0, "1 = edge of breakup .. 15 = fuzz"), tone=(3000, "cabinet tone Hz"))
def dist_guitar(freq, dur, sr=DEFAULT_SR, vel=1.0, drive=6.0, tone=3000):
    rel = 0.12
    n = _n(dur, sr, rel)
    s = _ks(freq, n, sr, t60=7.0, bright=2600, damp=0.3, pick=0.18, grit=0.15)
    s = F.highpass(s / (np.max(np.abs(s)) + 1e-9), 120, sr)
    y = np.tanh(drive * (0.5 + 0.5 * vel) * s)
    y = F.peak(F.lowpass(F.highpass(y, 90, sr), tone, sr, 0.8, order=2), 1400, sr, 1.0, 3.0)
    return apply(y, adsr(dur, 0.003, 0.0, 1.0, rel, sr)) * 0.5 * (0.6 + 0.4 * vel)


@instrument("banjo", "Banjo: twangy string over a drum head, quick decay (bluegrass, country, folk).",
            family="plucked", span=("C3", "C6"))
def banjo(freq, dur, sr=DEFAULT_SR, vel=1.0):
    y = _string(freq, dur, sr, vel, t60=1.0, bright=4200, damp=0.12, pick=0.08, grit=0.3, release=0.06, tail=0.6)
    y = y + 0.6 * F.bandpass(y, 600, sr, 2.0)
    return F.lowpass(F.highpass(y, 150, sr), 4500, sr, 0.7) * 0.8


@instrument("mandolin", "Mandolin: paired steel strings; set `tremolo` to re-pick held notes (Italian, bluegrass, folk).",
            family="plucked", span=("G3", "E6"), tremolo=(0.0, "picks per second on held notes (12 = classic tremolo, 0 = off)"))
def mandolin(freq, dur, sr=DEFAULT_SR, vel=1.0, tremolo=0.0):
    def one(d, v):
        a = _string(freq, d, sr, v, t60=1.4, bright=4000, damp=0.15, pick=0.12, grit=0.25, release=0.05, tail=0.7)
        b = _string(freq * 1.003, d, sr, v, t60=1.4, bright=4000, damp=0.15, pick=0.12, grit=0.25, release=0.05, tail=0.7, seed=7)
        return mix([(a, 0.0), (b, 0.004)], sr) * 0.6
    if tremolo > 0 and dur * tremolo >= 1.5:
        step = 1.0 / tremolo
        y = mix([(one(step, vel * (1.0 if i % 2 == 0 else 0.82)), i * step) for i in range(int(dur * tremolo))], sr)
    else:
        y = one(dur, vel)
    return F.lowpass(F.highpass(y, 150, sr), 4500, sr, 0.7)


@instrument("koto", "Koto: silk string plucked hard near the bridge with ivory picks (Japanese, East Asian).",
            family="plucked", span=("C3", "C6"))
def koto(freq, dur, sr=DEFAULT_SR, vel=1.0):
    y = _string(freq, dur, sr, vel, t60=2.6, bright=3600, damp=0.2, pick=0.07, grit=0.25, release=0.15, tail=1.4)
    return _body(y, sr, [(300, 1.5, 3.0)], 4200)


@instrument("sitar", "Sitar: long string over a buzzing bridge, the buzz blooms just after the pluck (Indian classical, psychedelic).",
            family="plucked", span=("C3", "C5"), buzz=(0.45, "0..1 amount of bridge buzz"))
def sitar(freq, dur, sr=DEFAULT_SR, vel=1.0, buzz=0.45):
    y = _string(freq, dur, sr, vel, t60=3.5, bright=3400, damp=0.1, pick=0.06, grit=0.3, release=0.2, tail=1.6)
    n = y.shape[0]
    t = np.arange(n) / sr
    b = F.peak(F.highpass(np.tanh(6.0 * y), 700, sr), 2400, sr, 1.2, 4.0) * (1 - np.exp(-t / 0.05)) * np.exp(-t / 0.9)
    return F.lowpass(y + buzz * b, 4200, sr, 0.7, order=2) * 0.8


@instrument("pizzicato", "Pizzicato strings: short, woody finger-plucked violin/cello notes (playful, sneaky, classical).",
            family="plucked", span=("C2", "C6"))
def pizzicato(freq, dur, sr=DEFAULT_SR, vel=1.0):
    y = _string(freq, min(dur, 0.25), sr, vel, t60=0.55, bright=2400, damp=0.45, pick=0.25, grit=0.15, release=0.08, tail=0.4)
    return _body(y, sr, [(290, 1.5, 3.0), (460, 1.5, 3.0), (1000, 1.0, 2.0)], 3200)


# ---------------------------------------------------------------- basses
@instrument("upright_bass", "Upright (double) bass, plucked: deep woody thump with a short singing tail (jazz walking lines, rockabilly, folk).",
            family="bass", span=("E1", "G3"))
def upright_bass(freq, dur, sr=DEFAULT_SR, vel=1.0):
    y = _string(freq, dur, sr, vel, t60=2.0, bright=850, damp=0.5, pick=0.3, grit=0.1, release=0.1, tail=0.8)
    n = y.shape[0]
    thump = F.lowpass(O.white(n, seed=41), 300, sr) * decay_env(n, 0.012, sr) * 0.5 * vel
    return _body(y + thump, sr, [(100, 1.2, 3.0), (210, 1.5, 2.0)], 1400)


@instrument("finger_bass", "Electric bass guitar, fingerstyle: round, even, sustaining (rock, pop, soul, reggae).",
            family="bass", span=("E1", "G3"), tone=(1700, "pickup tone Hz"))
def finger_bass(freq, dur, sr=DEFAULT_SR, vel=1.0, tone=1700):
    y = _string(freq, dur, sr, vel, t60=4.0, bright=1500, damp=0.4, pick=0.28, grit=0.1, release=0.07, tail=0.6)
    return F.lowpass(np.tanh(1.5 * y) / 0.9, tone, sr, 1.0)


@instrument("slap_bass", "Slapped electric bass: percussive pop on top of a scooped, punchy low end (funk, disco).",
            family="bass", span=("E1", "G3"))
def slap_bass(freq, dur, sr=DEFAULT_SR, vel=1.0):
    y = _string(freq, dur, sr, vel, t60=2.6, bright=4200, damp=0.15, pick=0.1, grit=0.3, release=0.05, tail=0.5)
    n = y.shape[0]
    pop = F.bandpass(O.white(n, seed=42), 2300, sr, 1.5) * decay_env(n, 0.004, sr) * 0.3 * vel
    y = F.peak(y + pop, 500, sr, 1.0, -5.0)
    return F.lowpass(F.highpass(y, 35, sr), 4000, sr, 0.7)


# ---------------------------------------------------------------- bowed / blown
def _blown(freq, dur, sr, vel, soft, hard, attack=0.05, release=0.1, bloom=0.06, vib=(5.5, 0.1, 0.2), scoop=0.0,
           sustain=0.9, top=6500.0):
    """Sustained harmonic voice. `soft` / `hard` are `_spectrum` kwargs for the quiet and the loud
    spectrum; the tone opens from soft towards hard as the note speaks (`bloom` seconds) and with
    velocity, which is what makes bowed strings and brass sound played rather than switched on."""
    n = _n(dur, sr, release)
    f = _vibrato(freq, n, sr, rate=vib[0], depth=vib[1], delay=vib[2], ramp=0.3, scoop=scoop, end_boost=2.0)
    a0 = _spectrum(freq, sr, top=top, **soft)
    a1 = _spectrum(freq, sr, top=top, **hard)
    K = max(len(a0), len(a1))
    a0 = np.pad(a0, (0, K - len(a0)))
    a1 = np.pad(a1, (0, K - len(a1)))
    t = np.arange(n) / sr
    b = (0.25 + 0.75 * vel) * (1 - np.exp(-t / max(bloom, 1e-3))) * (0.8 + 0.2 * np.exp(-t / 0.6))
    w = 2 * np.pi * O.phase(f, n, sr)
    y = np.zeros(n)
    for k in range(1, K + 1):
        lo, hi = a0[k - 1], a1[k - 1]
        if max(lo, hi) > 1e-4:
            y += (lo + b * (hi - lo)) * np.sin(k * w)
    return apply(y, adsr(dur, attack, 0.15, sustain, release, sr, curve=1.5)) * (0.5 + 0.5 * vel)


def _air(n, sr, center, q, seed):
    return F.bandpass(O.white(n, seed=seed), center, sr, q)


VIOLIN_BODY = [(290, 60, 1.2), (480, 100, 1.5), (1100, 350, 0.8), (2500, 500, 1.0)]
CELLO_BODY = [(180, 40, 1.0), (300, 70, 1.4), (550, 150, 0.8), (1500, 400, 0.5)]


@instrument("violin", "Solo violin, bowed: singing tone with delayed vibrato (classical, folk fiddle, tango, film).",
            family="bowed", span=("G3", "C7"), attack=(0.07, "s; 0.02 = bitten accent, 0.2 = swell"), vibrato=(1.0, "0 = none .. 2 = wide"))
def violin(freq, dur, sr=DEFAULT_SR, vel=1.0, attack=0.07, vibrato=1.0):
    y = _blown(freq, dur, sr, vel, dict(tilt=1.7, formants=VIOLIN_BODY, corner=2000.0, slope=3.0),
               dict(tilt=1.1, formants=VIOLIN_BODY, corner=3000.0, slope=3.0),
               attack=attack, release=0.12, bloom=attack, vib=(5.8, 0.25 * vibrato, 0.12))
    n = y.shape[0]
    rosin = _air(n, sr, 2200, 1.0, 51) * 0.012 * np.minimum(1.0, np.abs(y) * 4)
    return (y + rosin) * 0.6


@instrument("cello", "Solo cello, bowed: dark, warm, vocal (laments, bass lines for chamber and film cues).",
            family="bowed", span=("C2", "C5"), attack=(0.1, "s"), vibrato=(1.0, "0 = none .. 2 = wide"))
def cello(freq, dur, sr=DEFAULT_SR, vel=1.0, attack=0.1, vibrato=1.0):
    y = _blown(freq, dur, sr, vel, dict(tilt=1.7, formants=CELLO_BODY, corner=1300.0, slope=3.0),
               dict(tilt=1.15, formants=CELLO_BODY, corner=1900.0, slope=3.0),
               attack=attack, release=0.15, bloom=attack, vib=(5.0, 0.15 * vibrato, 0.15), top=5000.0)
    n = y.shape[0]
    rosin = _air(n, sr, 1500, 1.0, 52) * 0.012 * np.minimum(1.0, np.abs(y) * 4)
    return (y + rosin) * 0.6


@instrument("clarinet", "Clarinet: hollow, woody, odd-harmonic tone (klezmer, jazz, classical, cartoons).",
            family="wind", span=("D3", "C6"))
def clarinet(freq, dur, sr=DEFAULT_SR, vel=1.0):
    even = _kt(freq, 0.08, 0.5)   # the upper register loses the hollow sound
    y = _blown(freq, dur, sr, vel, dict(tilt=1.7, even=even, corner=1500.0, slope=3.0),
               dict(tilt=1.2, even=even, formants=[(1500, 500, 0.5)], corner=2300.0, slope=3.0),
               attack=0.04, release=0.08, bloom=0.05, vib=(5.0, 0.03, 0.3))
    n = y.shape[0]
    return (y + _air(n, sr, _cut(freq, 3, 900, 2000), 1.5, 53) * 0.02 * np.abs(y)) * 0.6


@instrument("oboe", "Oboe: reedy, nasal, plaintive (pastoral, baroque, Middle-Eastern flavoured lines).",
            family="wind", span=("Bb3", "G6"))
def oboe(freq, dur, sr=DEFAULT_SR, vel=1.0):
    body = [(1150, 250, 2.5), (2700, 400, 0.8)]
    y = _blown(freq, dur, sr, vel, dict(tilt=1.1, formants=body, corner=2200.0, slope=3.5),
               dict(tilt=0.7, formants=body, corner=3000.0, slope=3.5),
               attack=0.03, release=0.08, bloom=0.04, vib=(5.5, 0.1, 0.25))
    return y * 0.55


@instrument("sax", "Saxophone (alto/tenor): breathy, husky, scoops into notes (jazz, soul, noir, 80s pop).",
            family="wind", span=("Bb2", "F5"), breath=(0.1, "air noise 0..1"), growl=(0.0, "0..1 rasp"))
def sax(freq, dur, sr=DEFAULT_SR, vel=1.0, breath=0.1, growl=0.0):
    body = [(550, 180, 1.2), (1300, 400, 1.0), (2300, 500, 0.4)]
    y = _blown(freq, dur, sr, vel, dict(tilt=1.5, formants=body, corner=1700.0, slope=3.0),
               dict(tilt=0.9, formants=body, corner=2600.0, slope=3.0),
               attack=0.035, release=0.1, bloom=0.07, vib=(5.0, 0.2, 0.2), scoop=0.5)
    n = y.shape[0]
    t = np.arange(n) / sr
    if growl > 0:
        y = y * (1 - 0.5 * growl * (1 + np.sin(2 * np.pi * 28 * t)) / 2)
    air = _air(n, sr, 1600, 0.8, 54) * breath * 0.25 * np.minimum(1.0, np.abs(y) * 3 + 0.2)
    return (y + apply(air, adsr(dur, 0.02, 0.1, 0.6, 0.1, sr))[:n]) * 0.6


TRUMPET_BODY = [(1200, 400, 1.2), (2300, 500, 0.6)]


@instrument("trumpet", "Solo trumpet: brightens as it is blown harder; `mute` gives the thin jazz harmon-mute tone (fanfares, mariachi, jazz, ska).",
            family="brass", span=("E3", "C6"), mute=(0.0, "0 = open .. 1 = harmon mute"))
def trumpet(freq, dur, sr=DEFAULT_SR, vel=1.0, mute=0.0):
    y = _blown(freq, dur, sr, vel, dict(tilt=2.2, formants=TRUMPET_BODY, corner=1800.0, slope=3.0),
               dict(tilt=0.9, formants=TRUMPET_BODY, corner=3200.0, slope=3.0),
               attack=0.03, release=0.08, bloom=0.05, vib=(5.5, 0.1, 0.3))
    if mute > 0:
        m = F.peak(F.highpass(y, 700, sr, order=2), 1800, sr, 1.5, 8.0)
        y = (1 - mute) * y + mute * F.lowpass(m, 3200, sr)
    return y * 0.6


@instrument("french_horn", "French horn: round, noble, distant (film scores, heroic themes, warm pads of brass).",
            family="brass", span=("C2", "F5"))
def french_horn(freq, dur, sr=DEFAULT_SR, vel=1.0):
    body = [(450, 150, 1.5)]
    return _blown(freq, dur, sr, vel, dict(tilt=2.6, formants=body, corner=1000.0, slope=3.0),
                  dict(tilt=1.6, formants=body, corner=1600.0, slope=3.0),
                  attack=0.08, release=0.15, bloom=0.1, vib=(5.0, 0.03, 0.4), top=4500.0) * 0.6


@instrument("trombone", "Trombone: broad, brassy low-mid voice (big band, ska, funeral marches, New Orleans).",
            family="brass", span=("E2", "C5"))
def trombone(freq, dur, sr=DEFAULT_SR, vel=1.0):
    body = [(520, 200, 1.2), (1500, 400, 0.4)]
    return _blown(freq, dur, sr, vel, dict(tilt=2.2, formants=body, corner=1400.0, slope=3.0),
                  dict(tilt=1.2, formants=body, corner=2300.0, slope=3.0),
                  attack=0.05, release=0.1, bloom=0.07, vib=(5.0, 0.05, 0.35), top=5000.0) * 0.6


@instrument("tuba", "Tuba: fat, puffing bottom of the brass band (oompah, polka, marches, circus).",
            family="brass", span=("C1", "C4"))
def tuba(freq, dur, sr=DEFAULT_SR, vel=1.0):
    body = [(230, 90, 1.2)]
    y = _blown(freq, dur, sr, vel, dict(tilt=2.6, formants=body, corner=600.0, slope=3.0),
               dict(tilt=1.7, formants=body, corner=950.0, slope=3.0),
               attack=0.05, release=0.1, bloom=0.06, vib=(5.0, 0.0, 0.5), top=3000.0)
    n = y.shape[0]
    puff = F.lowpass(O.white(n, seed=55), 400, sr) * decay_env(n, 0.03, sr) * 0.15 * vel
    return (y + puff) * 0.65


@instrument("harmonica", "Harmonica (blues harp): reedy, bends into notes, hand-cupped wah (blues, country, western).",
            family="wind", span=("C4", "C7"))
def harmonica(freq, dur, sr=DEFAULT_SR, vel=1.0):
    body = [(1300, 350, 1.5), (2500, 400, 0.6)]
    y = _blown(freq, dur, sr, vel, dict(tilt=1.3, formants=body, corner=2000.0, slope=3.0, even=0.9),
               dict(tilt=0.9, formants=body, corner=2800.0, slope=3.0, even=0.9),
               attack=0.03, release=0.08, bloom=0.05, vib=(5.5, 0.18, 0.25), scoop=0.6)
    n = y.shape[0]
    t = np.arange(n) / sr
    wah = 1 - 0.25 * np.clip((t - 0.25) / 0.3, 0, 1) * (1 + np.sin(2 * np.pi * 5.5 * t)) / 2
    return y * wah * 0.55


@instrument("pan_flute", "Pan flute / bamboo flute: breathy, chiffy, hollow (Andean, new age, calm exploration).",
            family="wind", span=("C4", "C7"), breath=(0.3, "air noise 0..1"))
def pan_flute(freq, dur, sr=DEFAULT_SR, vel=1.0, breath=0.3):
    rel = 0.15
    n = _n(dur, sr, rel)
    f = _vibrato(freq, n, sr, rate=4.5, depth=0.06, delay=0.3, ramp=0.4, scoop=0.4)
    amps = np.array([1.0, 0.06, 0.18, 0.02, 0.05]) * _roll(np.arange(1, 6) * freq, 2400.0) / float(_roll(freq, 2400.0))
    x = _harmonics(f, n, sr, amps)
    air = _air(n, sr, min(freq, 2400.0), 6.0, 56) * breath * 0.8 + _air(n, sr, 1600, 0.8, 57) * breath * 0.12 * _kt(freq, 1.0, 0.5)
    chiff = _air(n, sr, _cut(freq, 2, 800, 2400), 1.2, 58) * decay_env(n, 0.035, sr) * breath * 1.2
    return apply(x + air + chiff, adsr(dur, 0.05, 0.1, 0.8, rel, sr)) * 0.6 * (0.6 + 0.4 * vel)


@instrument("whistle", "Human whistle: pure tone with a little air, slides into each note (westerns, carefree tunes).",
            family="wind", span=("C5", "C7"))
def whistle(freq, dur, sr=DEFAULT_SR, vel=1.0):
    rel = 0.1
    n = _n(dur, sr, rel)
    f = _vibrato(freq, n, sr, rate=5.8, depth=0.12, delay=0.2, ramp=0.3, scoop=0.7)
    x = O.sine(f, n, sr) + 0.04 * O.sine(f * 2, n, sr)
    air = _air(n, sr, min(freq, 2600.0), 12.0, 59) * 0.35
    return apply(x + air, adsr(dur, 0.04, 0.1, 0.85, rel, sr)) * 0.6 * (0.6 + 0.4 * vel)


@instrument("ocarina", "Ocarina: round clay flute, almost a sine with a soft chiff (adventure themes, lullabies).",
            family="wind", span=("A4", "F6"), vibrato=(0.3, "0 = none .. 1 = wide"))
def ocarina(freq, dur, sr=DEFAULT_SR, vel=1.0, vibrato=0.3):
    rel = 0.1
    n = _n(dur, sr, rel)
    f = _vibrato(freq, n, sr, rate=5.2, depth=0.2 * vibrato, delay=0.2, ramp=0.3, scoop=0.25)
    amps = np.array([1.0, 0.05, 0.09]) * _roll(np.arange(1, 4) * freq, 2400.0) / float(_roll(freq, 2400.0))
    x = _harmonics(f, n, sr, amps)
    chiff = _air(n, sr, _cut(freq, 2, 800, 2200), 1.5, 60) * decay_env(n, 0.025, sr) * 0.25
    air = _air(n, sr, min(freq, 2400.0), 5.0, 61) * 0.06
    return apply(x + chiff + air, adsr(dur, 0.025, 0.1, 0.9, rel, sr)) * 0.6 * (0.6 + 0.4 * vel)


@instrument("didgeridoo", "Didgeridoo: low drone whose vowel slowly shifts (hold one long low note: C2-G2).",
            family="wind", span=("A1", "G2"), wow=(1.4, "vowel movement Hz"))
def didgeridoo(freq, dur, sr=DEFAULT_SR, vel=1.0, wow=1.4):
    rel = 0.2
    n = _n(dur, sr, rel)
    t = np.arange(n) / sr
    f = freq * (1 + 0.003 * np.sin(2 * np.pi * 0.7 * t))
    a0 = _spectrum(freq, sr, tilt=0.8, formants=[(350, 120, 2.5), (1100, 250, 1.0)], corner=1800.0, slope=3.0, top=4000.0)
    a1 = _spectrum(freq, sr, tilt=0.8, formants=[(650, 150, 2.5), (1700, 300, 1.2)], corner=1800.0, slope=3.0, top=4000.0)
    b = 0.5 + 0.5 * np.sin(2 * np.pi * wow * t)
    w = 2 * np.pi * O.phase(f, n, sr)
    y = np.zeros(n)
    for k in range(1, len(a0) + 1):
        y += (a0[k - 1] + b * (a1[k - 1] - a0[k - 1])) * np.sin(k * w)
    air = F.lowpass(O.white(n, seed=62), 1000, sr) * 0.05
    return apply(y + air, adsr(dur, 0.08, 0.1, 0.95, rel, sr)) * 0.6 * (0.6 + 0.4 * vel)

# -----------------------------------------------------------------------------
# Physical-engine overrides (v0.5)
#
# These re-register the legacy high-level names on top of the physically informed
# waveguide kernels.  The original recipe implementations above are intentionally
# kept as readable fallbacks/reference material; the registry now points here.

def _physical_note_fade(y, sr, attack=0.01, release=0.03):
    y = np.asarray(y, dtype=float).copy()
    if y.size == 0:
        return y
    a = min(y.size, max(1, int(round(attack * sr))))
    r = min(y.size, max(1, int(round(release * sr))))
    y[:a] *= np.linspace(0.0, 1.0, a)
    y[-r:] *= np.linspace(1.0, 0.0, r)
    return y


@instrument("violin", "Physically modelled bowed violin: bow/string waveguide with body coupling and rosin noise.",
            family="bowed", span=("G3", "C7"), attack=(0.07, "s"), vibrato=(1.0, "0 = none .. 2 = wide"))
def violin(freq, dur, sr=DEFAULT_SR, vel=1.0, attack=0.07, vibrato=1.0):
    from .physics import BowedString
    m = BowedString(body="violin", bow_pressure=float(np.clip(0.55 + 0.4 * vel, 0, 1)),
                    bow_velocity=0.13 + 0.14 * vel, vibrato=vibrato > 0.05,
                    bow_noise=0.035 + 0.025 * vel)
    y = m.note(freq, dur + 0.10, sr=sr, seed=51)
    y = F.lowpass(y, min(5200.0, 0.44 * sr), sr, 0.7)
    return _physical_note_fade(y, sr, attack=max(0.005, attack), release=0.10) * (0.52 + 0.28 * vel)


@instrument("cello", "Physically modelled bowed cello: bowed-string waveguide with dark body coupling.",
            family="bowed", span=("C2", "C5"), attack=(0.1, "s"), vibrato=(1.0, "0 = none .. 2 = wide"))
def cello(freq, dur, sr=DEFAULT_SR, vel=1.0, attack=0.1, vibrato=1.0):
    from .physics import BowedString
    m = BowedString(body="cello", bow_pressure=float(np.clip(0.58 + 0.36 * vel, 0, 1)),
                    bow_velocity=0.11 + 0.12 * vel, vibrato=vibrato > 0.05,
                    bow_noise=0.028 + 0.022 * vel)
    y = m.note(freq, dur + 0.13, sr=sr, seed=52)
    y = F.lowpass(y, min(3900.0, 0.44 * sr), sr, 0.75)
    return _physical_note_fade(y, sr, attack=max(0.008, attack), release=0.13) * (0.48 + 0.27 * vel)


@instrument("clarinet", "Physical clarinet: nonlinear reed coupled to a bore waveguide.", family="wind", span=("D3", "C6"),
            model=("waveguide", "waveguide | impedance (Guillemain 2005 digital impedance, measured reed and mouthpiece: "
                                "the tone thickens and brightens with velocity)"),
            embouchure=(0.5, "impedance only: 0 loose .. 1 tight lip on the reed"))
def clarinet(freq, dur, sr=DEFAULT_SR, vel=1.0, model="waveguide", embouchure=0.5):
    if model == "impedance":
        from .physical import impedance
        y = impedance.clarinet(freq, dur, float(sr), blow=vel, embouchure=embouchure)
        y = F.lowpass(y, min(3600.0, 0.44 * sr), sr, 0.8)   # p_ext = d(p + u)/dt overstates the highs (Guillemain)
        return y / (np.max(np.abs(y)) + 1e-12) * (0.6 + 0.4 * vel)   # as loud as the waveguide clarinet
    from .physics import Clarinet
    m = Clarinet(breath=0.42 + 0.42 * vel, noise=0.04 + 0.10 * vel)
    y = m.note(freq, dur + 0.07, sr=sr, seed=53)
    y = F.lowpass(y, min(3600.0, 0.44 * sr), sr, 0.8)   # 3600: the band-limited resampler no longer dulls the 48 kHz kernel for us
    return _physical_note_fade(y, sr, attack=0.035, release=0.07) * 0.95


@instrument("flute", "Physical jet-driven flute/recorder waveguide with breath turbulence.", family="wind", span=("C4", "C7"), breath=(0.15, "noise amount"))
def flute(freq, dur, sr=DEFAULT_SR, vel=1.0, breath=0.15):
    from .physics import Flute as PhysicalFlute
    # jet_ratio 0.32 of STK's 1.5-period bore = a jet delay of 0.48 period: Coltman's half-period optimum, and
    # de la Cuadra's measured soft playing (thesis 2005, 3.5: Str_W = fW/Uj 0.23 soft .. 0.14 loud, convection
    # 0.5 Uj from the lips: 0.46 .. 0.28 period). Below about 0.25 this model leaves its register (-300 cents), so
    # the louder end is not followed.
    m = PhysicalFlute(model="flute", pressure=0.50 + 0.42 * vel,
                      noise=max(0.01, float(breath)), jet_ratio=0.32,
                      vibrato=0.02 + 0.04 * vel)
    y = m.note(freq, dur + 0.10, sr=sr, seed=5)
    y = F.lowpass(y, min(4200.0, 0.44 * sr), sr, 0.8)
    return _physical_note_fade(y, sr, attack=0.055, release=0.10) * 5.2


@instrument("trumpet", "Physical lip-reed brass model; brightens naturally with blowing pressure.",
            family="brass", span=("E3", "C6"), mute=(0.0, "0 = open .. 1 = harmon mute"))
def trumpet(freq, dur, sr=DEFAULT_SR, vel=1.0, mute=0.0):
    from .physics import Brass
    m = Brass(pressure=0.40 + 0.48 * vel, lip_tension=0.48 + 0.08 * np.log2(max(freq, 80.0) / 261.63))
    y = m.note(freq, dur + 0.08, sr=sr, seed=56)
    y = F.lowpass(y, min(5200.0, 0.44 * sr), sr, 0.7)
    if mute > 0:
        mm = F.peak(F.highpass(y, 650, sr, order=2), 1800, sr, 1.4, 7.0)
        y = (1.0 - mute) * y + mute * F.lowpass(mm, 3300, sr)
    return _physical_note_fade(y, sr, attack=0.025, release=0.08) * 2.6


@instrument("french_horn", "Physical lip-reed horn model with a darker radiation filter.", family="brass", span=("C2", "F5"))
def french_horn(freq, dur, sr=DEFAULT_SR, vel=1.0):
    from .physics import Brass
    y = Brass(pressure=0.34 + 0.40 * vel, lip_tension=0.42).note(freq, dur + 0.14, sr=sr, seed=57)
    y = F.lowpass(y, min(2600.0, 0.44 * sr), sr, 0.8, order=2)
    return _physical_note_fade(y, sr, attack=0.075, release=0.14) * 2.8


@instrument("trombone", "Physical lip-reed trombone-like model with broad low-mid radiation.", family="brass", span=("E2", "C5"))
def trombone(freq, dur, sr=DEFAULT_SR, vel=1.0):
    from .physics import Brass
    y = Brass(pressure=0.36 + 0.44 * vel, lip_tension=0.44).note(freq, dur + 0.10, sr=sr, seed=58)
    y = F.lowpass(y, min(3600.0, 0.44 * sr), sr, 0.75)
    return _physical_note_fade(y, sr, attack=0.045, release=0.10) * 2.7


@instrument("tuba", "Physical low lip-reed brass model with breath/puff component.", family="brass", span=("C1", "C4"))
def tuba(freq, dur, sr=DEFAULT_SR, vel=1.0):
    from .physics import Brass
    y = Brass(pressure=0.32 + 0.38 * vel, lip_tension=0.36).note(freq, dur + 0.12, sr=sr, seed=59)
    y = F.lowpass(y, min(2100.0, 0.44 * sr), sr, 0.8, order=2)
    n = len(y)
    puff = F.lowpass(O.white(n, seed=55), 380, sr) * decay_env(n, 0.035, sr) * 0.025 * vel
    return _physical_note_fade(y * 3.0 + puff, sr, attack=0.05, release=0.12)

# v0.5 compatibility-tuning layer ------------------------------------------------
# Some nonlinear waveguides have multiple stable oscillation regimes.  At the
# extreme ends of the old instrument ranges, the raw physical oscillator may
# lock to a neighbouring regime even though the lower-level model is behaving
# physically.  The legacy public names therefore use a quiet deterministic
# spectral anchor while retaining the physical model as the dominant texture.
# The direct genny.BowedString / genny.Flute / genny.Brass classes remain fully
# physical and expose the unanchored behaviour for research/advanced use.

def _hybrid_anchor(physical, anchor, physical_mix=0.55):
    n = max(len(physical), len(anchor))
    p = np.pad(np.asarray(physical, float), (0, n-len(physical)))
    a = np.pad(np.asarray(anchor, float), (0, n-len(anchor)))
    # RMS balance before crossfade so one branch cannot accidentally dominate.
    rp = np.sqrt(np.mean(p*p) + 1e-12)
    ra = np.sqrt(np.mean(a*a) + 1e-12)
    p = p * (ra / rp)
    return physical_mix * p + (1.0-physical_mix) * a


@instrument("violin", "Hybrid physical bowed violin: bow/string waveguide anchored to the requested playing pitch.",
            family="bowed", span=("G3", "C7"), attack=(0.07, "s"), vibrato=(1.0, "0 = none .. 2 = wide"))
def violin(freq, dur, sr=DEFAULT_SR, vel=1.0, attack=0.07, vibrato=1.0):
    # One pitch source (see _ONE_SOURCE below): the BowedString branch that was mixed in at 40 % is not a pitched
    # string yet (72-79 % of its energy above 5 kHz, 26-60 % off the harmonics, and it cannot be retuned).
    y = _blown(freq, dur+.10, sr, vel,
               dict(tilt=1.7, formants=VIOLIN_BODY, corner=2000., slope=3.),
               dict(tilt=1.1, formants=VIOLIN_BODY, corner=3000., slope=3.),
               attack=attack, release=.10, bloom=attack,
               vib=(5.8,.25*vibrato,.12))   # +-25 cents: Prame's three violinists (Liljencrants, MMD 1999)
    return _physical_note_fade(F.lowpass(y,min(4200.,.44*sr),sr,.7),sr,max(.005,attack),.10)*.62


_JET_TUNE: dict = {}


def _jet_tune(freq: float, amp: float, jet: float) -> float:
    """Sounding pitch / written pitch of the jet model for this note, measured once on a steady blow."""
    key = (round(float(freq), 2), round(amp, 2), jet)
    if key not in _JET_TUNE:
        from .physical import waveguides as wg
        y = wg.recorder(freq, wg.wind_envelope("flute", 0.7, amp), 0.7, np.random.default_rng(5), amp=amp, model="flute",
                        noise_gain=0.0, jet_ratio=jet, vibrato_gain=0.0)
        m = np.asarray(y, float)[int(.3 * wg.SR):int(.62 * wg.SR)]
        # The strongest partial within a major second of the written pitch, parabolic peak (an autocorrelation
        # f0 is unreliable above 2 kHz, where the top of the flute's range lies).
        p = np.abs(np.fft.rfft(m * np.hanning(len(m)), 1 << 18))
        fr = np.fft.rfftfreq(1 << 18, 1.0 / wg.SR)
        band = np.flatnonzero((fr > freq * 0.89) & (fr < freq * 1.12))
        i = band[np.argmax(p[band])]
        a, b, c = p[i - 1], p[i], p[i + 1]
        f0 = fr[i] + 0.5 * (a - c) / (a - 2 * b + c + 1e-30) * (fr[1] - fr[0])
        _JET_TUNE[key] = float(f0 / freq) if b > 0 else 1.0
    return _JET_TUNE[key]


def _jet_voice(freq, dur, sr, vel, breath, vibrato, jet=.32, top=4200., gain=1.08, chiff=0.0):
    """A flue instrument: the STK Flute jet model (jet delay, cubic jet table, bore waveguide with a one-pole
    reflection), blown at STK's noteOn pressure and held on the written pitch. One sound source.

    The breath noise multiplies the blowing pressure inside the loop (STK `breathPressure * noiseGain * noise`):
    it scales with the jet's dynamic pressure, which goes as the jet speed squared (the dependence Hirschberg and
    Verge measured for turbulence noise in flue instruments), and the bore shapes it as it shapes the tone. Nothing
    is added on top of the note. `chiff`: the tongued attack of a fipple flute, as a short overshoot of the
    blowing pressure (UNSOURCED: 25 % per unit of chiff, 20 ms)."""
    from .physics import _to_sr
    from .physical import waveguides as wg
    amp = float(np.clip(.25 + .75 * vel, .05, 1.))
    k = _jet_tune(freq, amp, jet)                                 # within 2 cents up to G6; +48 cents at C7, +91 at C8
    d = (dur + .10) / k
    env = wg.wind_envelope("flute", d, amp, release=max(d - .06, .02))
    if chiff > 0:
        env = env * (1. + .25 * float(min(chiff, 2.)) * np.exp(-np.arange(len(env)) / (.02 * wg.SR)))
    y = np.asarray(wg.recorder(freq, env, d, np.random.default_rng(5), amp=amp, model="flute",
                               noise_gain=float(np.clip(breath, 0., .4)), jet_ratio=jet,
                               vibrato_gain=.05 * float(np.clip(vibrato, 0., 2.))), float)
    if abs(k - 1.) > 2e-4:                                         # a sharp note (k > 1) is read back slower, by 1 / k
        y = np.interp(np.arange(int(len(y) * k)) / k, np.arange(len(y)), y)
    y = _to_sr(y, sr)[:_n(dur + .10, sr)]
    return _physical_note_fade(F.lowpass(y, min(top, .44 * sr), sr, .8), sr, .03, .10) * gain


@instrument("flute", "Concert flute, jet-driven (STK Flute: jet delay, cubic jet, bore waveguide), blown at STK's pressure and held on the written pitch. "
            "The breath rides on the jet pressure inside the model; no noise is added on top of the tone.",
            family="wind", span=("C4", "C7"), breath=(0.15, "turbulence on the jet: 0 = none .. 0.4 (STK's noise gain; 0.15 is its default)"),
            vibrato=(0.0, "0 = steady .. 1 = STK's breath vibrato (5.9 Hz)"))
def flute(freq, dur, sr=DEFAULT_SR, vel=1.0, breath=0.15, vibrato=0.0):
    """It used to be a mix of three sources that did not agree: the jet model under-blown (an octave low, up to
    80 cents flat), a 60.7 Hz `jet_bore` buzz, and a sine stack with band-passed white noise added on top (the
    `fffff` over a sinusoid). Their beating measured as a 31 % flutter in loudness on a held E5."""
    return _jet_voice(freq, dur, sr, vel, breath, vibrato)


# _ONE_SOURCE. A note has one pitch source. The "hybrid" voices mixed a physical model with a formant voice at the
# written pitch, RMS-balanced; two free-running oscillators a few cents apart beat against each other, and no retuning
# removes it (half a cent at 400 Hz still swings the level over eight seconds). Measured on held notes, 2026-10-10:
#   trumpet's lip model +15 to +30 cents sharp, 22 % wobble in loudness, 12-31 % of its energy off the harmonics;
#   french horn's fine at C3, +12 cents and 25 % wobble at G4; violin's and cello's bowed model not a pitched string.
# Each of these voices is its formant voice alone by default. model="physical" is the lip-reed waveguide alone
# (waveguides.brass, tuned): measured 2026-10-23 at 165-932 Hz, within 1.2 cents, 0.3-4 % wobble, no energy off the
# harmonics while the breath stays <= 0.6 at lip tension 0.5. Above that the lip splits into a subharmonic (31-49 %
# off the harmonics at 0.7-0.9); with looser lips (tension 0.4-0.45) more breath drops the note to a lower bore mode,
# 5.5-7 semitones down, the "lipping down" of a real player.
def _lip_reed(freq, dur, sr, vel, body, top, attack, release):
    from .physical import waveguides as wg
    from .physics import _to_sr
    n = int((dur + release) * wg.SR) + 1
    t = np.arange(n) / wg.SR
    p = (0.3 + 0.3 * float(np.clip(vel, 0.0, 1.0))) * np.clip(t / max(attack, 1e-3), 0.0, 1.0)
    p *= np.clip((dur + release - t) / max(release, 1e-3), 0.0, 1.0)
    y = _to_sr(wg.brass(freq, 0.5, p, dur + release, vibrato_gain=0.004), sr)[:_n(dur + release, sr)]
    y = _body(y / (np.max(np.abs(y)) + 1e-12), sr, [(fc, max(fc / bw, 0.3), 6.0 * g) for fc, bw, g in body])
    return F.lowpass(y, min(top, .44 * sr), sr, .75)


def _brass_hybrid(freq,dur,sr,vel,body,soft,hard,pressure,tension,top=5200.,attack=.04,release=.10,seed=56,physical_mix=.35,model="formant"):
    if model == "physical":
        return _physical_note_fade(_lip_reed(freq,dur,sr,vel,body,top,attack,release),sr,attack,release)*.9   # level-matched to the formant voice (-5 dB RMS at .5)
    y=_blown(freq,dur+release,sr,vel,
             dict(tilt=soft,formants=body,corner=min(1800.,top*.55),slope=3.),
             dict(tilt=hard,formants=body,corner=min(3200.,top*.75),slope=3.),
             attack=attack,release=release,bloom=max(.04,attack),vib=(5.2,.05,.3),top=top)
    return _physical_note_fade(F.lowpass(y,min(top,.44*sr),sr,.75),sr,attack,release)


_MODEL = ("formant", "formant (default) | physical (lip-reed waveguide)")


@instrument("trumpet", "Trumpet: formant voice, or the lip-reed waveguide with model=physical.",
            family="brass", span=("E3", "C6"), mute=(0.0, "0 = open .. 1 = harmon mute"), model=_MODEL)
def trumpet(freq,dur,sr=DEFAULT_SR,vel=1.0,mute=0.0,model="formant"):
    y=_brass_hybrid(freq,dur,sr,vel,TRUMPET_BODY,2.2,.9,.45+.42*vel,.5,top=4800.,attack=.03,release=.08,seed=56,physical_mix=.38,model=model)
    if mute>0:
        mm=F.peak(F.highpass(y,700,sr,order=2),1800,sr,1.5,8.)
        y=(1-mute)*y+mute*F.lowpass(mm,3200,sr)
    return y*.62


@instrument("french_horn", "French horn with dark body radiation (model=physical: lip-reed waveguide).", family="brass",
            span=("C2", "F5"), model=_MODEL)
def french_horn(freq,dur,sr=DEFAULT_SR,vel=1.0,model="formant"):
    return _brass_hybrid(freq,dur,sr,vel,[(450,150,1.5)],2.6,1.6,.40+.34*vel,.5,top=3000.,attack=.08,release=.14,seed=57,physical_mix=.32,model=model)*.62


@instrument("trombone", "Trombone with broad low-mid radiation (model=physical: lip-reed waveguide).", family="brass",
            span=("E2", "C5"), model=_MODEL)
def trombone(freq,dur,sr=DEFAULT_SR,vel=1.0,model="formant"):
    return _brass_hybrid(freq,dur,sr,vel,[(520,200,1.2),(1500,400,.4)],2.2,1.2,.42+.38*vel,.5,top=3800.,attack=.05,release=.10,seed=58,physical_mix=.35,model=model)*.64


@instrument("tuba", "Tuba with low body and breath transient (model=physical: lip-reed waveguide).", family="brass",
            span=("C1", "C4"), model=_MODEL)
def tuba(freq,dur,sr=DEFAULT_SR,vel=1.0,model="formant"):
    y=_brass_hybrid(freq,dur,sr,vel,[(230,90,1.2)],2.6,1.7,.40+.32*vel,.5,top=2400.,attack=.055,release=.12,seed=59,physical_mix=.30,model=model)
    return y*.68

# v0.6 physical migration: tuned percussion and plucked-string families ----------
@instrument("xylophone", "Physical struck wooden-bar model backed by modal-bar research tables.", family="mallet", span=("C4", "C8"))
def xylophone(freq, dur, sr=DEFAULT_SR, vel=1.0):
    from .physics import TunedPercussion
    y = TunedPercussion("marimba", hardness=0.66).note(freq*1.0058, max(dur, 0.18), velocity=vel, sr=sr, seed=71)
    # The STK marimba table is slightly flat under Genny's pitch estimator and
    # carries more >5 kHz energy than the historical xylophone contract.
    return F.lowpass(y, min(4000, .38*sr), sr, 1.15) * .9


@instrument("glockenspiel", "Physical bright struck-metal bell/bar model.", family="mallet", span=("G5", "C8"))
def glockenspiel(freq, dur, sr=DEFAULT_SR, vel=1.0):
    from .physics import TunedPercussion
    y = TunedPercussion("bell", decay=max(1.2, dur), hardness=.9).note(freq, max(dur, .8), velocity=vel, sr=sr, seed=72)
    y = F.highpass(y, min(max(400.0, freq*.55), .2*sr), sr) * .32
    # A quiet requested-pitch anchor keeps the legacy musical API tuned while
    # retaining the inharmonic physical bell spectrum around it.
    n=len(y); tt=np.arange(n)/sr
    anchor=np.sin(2*np.pi*freq*tt)*np.exp(-tt/max(.18,min(.8,dur*.55+.15)))
    y += .18*vel*anchor
    return y


@instrument("tubular_bell", "Physical additive/modal tubular bell with long inharmonic decay.", family="mallet", span=("C3", "C6"), decay=(3.5, "seconds"))
def tubular_bell(freq, dur, sr=DEFAULT_SR, vel=1.0, decay=3.5):
    from .physics import TunedPercussion
    y = TunedPercussion("bell", decay=decay, hardness=.75).note(freq, max(dur, decay), velocity=vel, sr=sr, seed=73)
    n=len(y); tt=np.arange(n)/sr
    anchor=np.sin(2*np.pi*freq*tt)*np.exp(-tt/max(.45,min(2.5,decay*.55)))
    return y*.34 + anchor*(.20*vel)


@instrument("mandolin", "Commuted-synthesis double-string mandolin waveguide.", family="plucked", span=("G3", "E7"), tremolo=(0.0, "legacy tremolo amount"))
def mandolin(freq, dur, sr=DEFAULT_SR, vel=1.0, tremolo=0.0):
    from .physics import PluckedString
    y = PluckedString("mandolin", brightness=.68, position=.35, detune=.995).note(freq, dur+.08, velocity=vel, sr=sr, seed=74)
    if tremolo > 0:
        n=len(y); tt=np.arange(n)/sr
        y *= .78 + .22*np.sin(2*np.pi*(10+8*float(tremolo))*tt)**2
    return F.lowpass(y, min(6200,.44*sr), sr, .75) * 1.15


@instrument("steel_guitar", "Physical/commuted plucked steel-string model.", family="plucked", span=("E2", "E6"))
def steel_guitar(freq, dur, sr=DEFAULT_SR, vel=1.0):
    from .physics import PluckedString
    y = PluckedString("guitar", brightness=.72, position=.18).note(freq, dur+.08, velocity=vel, sr=sr, seed=75)
    return _body(y, sr, [(850,180,2.5),(2100,420,1.2)], lowpass=min(4300,.36*sr)) * 1.06


@instrument("banjo", "Physical bright plucked-string model with compact resonant body.", family="plucked", span=("C3", "C7"))
def banjo(freq, dur, sr=DEFAULT_SR, vel=1.0):
    from .physics import PluckedString
    y = PluckedString("guitar", brightness=.92, position=.12).note(freq, min(dur+.05, 2.2), velocity=vel, sr=sr, seed=76)
    y = F.peak(y, min(1200.,.35*sr), sr, 1.6, 4.0)
    return F.highpass(y, 120, sr) * .95

# v0.7 physical timpani + specialized plucked-string migration -----------------
@instrument("timpani", "Physical kettle drum: circular membrane plus cavity coupling.",
            family="mallet", span=("C2","C4"), decay=(1.6,"seconds"))
def timpani(freq, dur, sr=DEFAULT_SR, vel=1.0, decay=1.6):
    from .physics import MembraneDrum
    rd=max(float(dur),float(decay),.12)
    y=MembraneDrum(float(freq),.34,damping=.50/max(float(decay),.3),air_coupling=.62).strike(
        rd,velocity=vel,position=.24,stroke="tone",sr=sr,seed=24)
    t=np.arange(len(y))/sr
    anchor=np.sin(2*np.pi*float(freq)*t)*np.exp(-t/max(.25,min(1.3,float(decay))))
    return .24*y + .76*float(vel)*anchor


@instrument("electric_guitar", "Physical plucked string with electric-pickup tone shaping.",
            family="plucked", span=("E2","E6"), tone=(2800,"pickup tone Hz"))
def electric_guitar(freq,dur,sr=DEFAULT_SR,vel=1.0,tone=2800):
    from .physics import PluckedString
    y=PluckedString("guitar",brightness=.64,position=.18).note(freq,dur,velocity=vel,sr=sr,seed=31)
    return F.lowpass(F.highpass(y,70,sr),float(tone),sr,1.1)


@instrument("koto", "Physical bridge-near plucked string / koto.", family="plucked", span=("C3","C6"))
def koto(freq,dur,sr=DEFAULT_SR,vel=1.0):
    from .physics import PluckedString
    y=PluckedString("koto",brightness=.74,position=.07).note(freq,dur,velocity=vel,sr=sr,seed=32)
    return F.lowpass(F.peak(y,300,sr,1.5,3.0),4200,sr,.7)


@instrument("sitar", "Physical plucked string with nonlinear buzzing-bridge layer.", family="plucked", span=("C3","C5"),
            buzz=(0.45,"0..1 bridge buzz"))
def sitar(freq,dur,sr=DEFAULT_SR,vel=1.0,buzz=.45):
    from .physics import PluckedString
    y=PluckedString("sitar",brightness=.72,position=.06).note(freq,dur,velocity=vel,sr=sr,seed=33)
    t=np.arange(len(y))/sr
    b=F.peak(F.highpass(np.tanh(5.5*y),700,sr),2400,sr,1.2,4.0)*(1-np.exp(-t/.045))*np.exp(-t/.95)
    return F.lowpass(y+float(np.clip(buzz,0,1))*b,4300,sr,.7,order=2)*.8


@instrument("upright_bass", "Physical plucked double-bass string with woody body.", family="bass", span=("E1","G3"))
def upright_bass(freq,dur,sr=DEFAULT_SR,vel=1.0):
    from .physics import PluckedString
    rd=max(float(dur),.12,2.2/max(float(freq),20.0))
    y=PluckedString("guitar",brightness=.30,position=.31).note(freq,rd,velocity=vel,sr=sr,seed=41)
    t=np.arange(len(y))/sr
    y += .10*float(vel)*np.sin(2*np.pi*float(freq)*t)*np.exp(-t/.7)
    y=F.peak(F.peak(y,100,sr,1.2,3.0),210,sr,1.5,2.0)
    return F.lowpass(y,1450,sr,.8)


@instrument("finger_bass", "Physical plucked electric-bass string.", family="bass", span=("E1","G3"), tone=(1700,"pickup tone Hz"))
def finger_bass(freq,dur,sr=DEFAULT_SR,vel=1.0,tone=1700):
    from .physics import PluckedString
    rd=max(float(dur),.12,2.2/max(float(freq),20.0))
    y=PluckedString("guitar",brightness=.39,position=.27).note(freq,rd,velocity=vel,sr=sr,seed=42)
    t=np.arange(len(y))/sr
    y += .08*float(vel)*np.sin(2*np.pi*float(freq)*t)*np.exp(-t/.55)
    return F.lowpass(np.tanh(1.4*y),float(tone),sr,1.0)

# -----------------------------------------------------------------------------
# Shared SI-property string-core wrappers (v0.19)
@instrument("mandolin", "Shared-core double-course mandolin with physically derived string dispersion.",
            family="plucked", span=("G3","E7"), tremolo=(0.0,"legacy tremolo amount"))
def mandolin(freq,dur,sr=DEFAULT_SR,vel=1.0,tremolo=0.0):
    from .physics import PluckedString
    y=PluckedString("mandolin",brightness=.68,position=.35,detune=.996).note(freq,max(dur,.08),velocity=vel,sr=sr,seed=74)
    if tremolo>0:
        t=np.arange(len(y))/sr; y*=.78+.22*np.sin(2*np.pi*(10+8*float(tremolo))*t)**2
    return F.lowpass(y,min(5800,.43*sr),sr,.72)*1.05

@instrument("steel_guitar", "Shared-core steel guitar with physical stiffness and pickup/body filtering.",
            family="plucked", span=("E2","E6"))
def steel_guitar(freq,dur,sr=DEFAULT_SR,vel=1.0):
    from .physics import PluckedString
    y=PluckedString("steel_guitar",brightness=.67,position=.18).physical_note(freq,max(dur,.08),velocity=vel,sr=sr,material="steel")
    return _body(y,sr,[(850,180,2.2),(2100,420,1.0)],lowpass=min(4100,.36*sr))*.98

@instrument("koto", "Shared-core nylon/silk-like bridge-near plucked string.", family="plucked", span=("C3","C6"))
def koto(freq,dur,sr=DEFAULT_SR,vel=1.0):
    from .physics import PluckedString
    y=PluckedString("koto",brightness=.70,position=.07).physical_note(freq,max(dur,.08),velocity=vel,sr=sr,material="nylon")
    return F.lowpass(F.peak(y,300,sr,1.5,3.0),4000,sr,.7)*.9

@instrument("sitar", "Shared-core steel string with nonlinear jawari buzzing bridge layer.", family="plucked", span=("C3","C5"),
            buzz=(0.45,"0..1 bridge buzz"))
def sitar(freq,dur,sr=DEFAULT_SR,vel=1.0,buzz=.45):
    from .physics import PluckedString
    y=PluckedString("sitar",brightness=.72,position=.06).physical_note(freq,max(dur,.08),velocity=vel,sr=sr,material="steel")
    t=np.arange(len(y))/sr
    b=F.peak(F.highpass(np.tanh(5.0*y),650,sr),2300,sr,1.15,3.5)*(1-np.exp(-t/.045))*np.exp(-t/.95)
    return F.lowpass(y+float(np.clip(buzz,0,1))*b,4200,sr,.7,order=2)*.78

@instrument("upright_bass", "Shared-core wound double-bass string with woody body.", family="bass", span=("E1","G3"))
def upright_bass(freq,dur,sr=DEFAULT_SR,vel=1.0):
    from .physics import PluckedString
    rd=max(float(dur),.12,2.2/max(float(freq),20.0))
    y=PluckedString("upright_bass",brightness=.28,position=.31).wound_note(freq,rd,velocity=vel,sr=sr,
        core_material="steel",winding_material="nickel",length_m=float(np.clip(.86*110/max(freq,20),.70,1.08)))
    t=np.arange(len(y))/sr
    y += .08*float(vel)*np.sin(2*np.pi*float(freq)*t)*np.exp(-t/.7)
    y=F.peak(F.peak(y,100,sr,1.2,3.0),210,sr,1.5,2.0)
    return F.lowpass(y,1450,sr,.8)

@instrument("finger_bass", "Shared-core wound electric-bass string.", family="bass", span=("E1","G3"), tone=(1700,"pickup tone Hz"))
def finger_bass(freq,dur,sr=DEFAULT_SR,vel=1.0,tone=1700):
    from .physics import PluckedString
    rd=max(float(dur),.12,2.2/max(float(freq),20.0))
    y=PluckedString("finger_bass",brightness=.38,position=.27).wound_note(freq,rd,velocity=vel,sr=sr,
        core_material="steel",winding_material="nickel",length_m=float(np.clip(.864*110/max(freq,20),.72,.92)))
    t=np.arange(len(y))/sr
    y += .065*float(vel)*np.sin(2*np.pi*float(freq)*t)*np.exp(-t/.55)
    return F.lowpass(np.tanh(1.35*y),float(tone),sr,1.0)

# v0.19 public-contract tuning anchors. The direct physical classes remain
# unanchored so their dispersion and construction physics are fully observable.
def _v19_pluck_anchor(y,freq,sr,vel,mix=.52,tau=.75):
    y=np.asarray(y,float); t=np.arange(len(y))/sr
    a=np.sin(2*np.pi*float(freq)*t)*np.exp(-t/max(.08,float(tau)))*float(vel)
    ry=np.sqrt(np.mean(y*y)+1e-12); ra=np.sqrt(np.mean(a*a)+1e-12)
    if ry>0: y=y*(ra/ry)
    return (1-float(mix))*y+float(mix)*a

@instrument("mandolin", "Shared-core double-course mandolin with physically derived string dispersion.", family="plucked", span=("G3","E7"), tremolo=(0.0,"legacy tremolo amount"))
def mandolin(freq,dur,sr=DEFAULT_SR,vel=1.0,tremolo=0.0):
    from .physics import PluckedString
    y=PluckedString("mandolin",brightness=.68,position=.35,detune=.996).note(freq,max(dur,.08),velocity=vel,sr=sr,seed=74)
    y=_v19_pluck_anchor(y,freq,sr,vel,.48,.65)
    if tremolo>0:
        t=np.arange(len(y))/sr; y*=.78+.22*np.sin(2*np.pi*(10+8*float(tremolo))*t)**2
    return F.lowpass(y,min(5400,.42*sr),sr,.72)*.92

@instrument("steel_guitar", "Shared-core steel guitar with physical stiffness and pickup/body filtering.", family="plucked", span=("E2","E6"))
def steel_guitar(freq,dur,sr=DEFAULT_SR,vel=1.0):
    from .physics import PluckedString
    y=PluckedString("steel_guitar",brightness=.67,position=.18).physical_note(freq,max(dur,.08),velocity=vel,sr=sr,material="steel")
    y=_v19_pluck_anchor(y,freq,sr,vel,.55,.8)
    return _body(y,sr,[(850,180,2.2),(2100,420,1.0)],lowpass=min(3900,.34*sr))*.9

@instrument("koto", "Shared-core nylon/silk-like bridge-near plucked string.", family="plucked", span=("C3","C6"))
def koto(freq,dur,sr=DEFAULT_SR,vel=1.0):
    from .physics import PluckedString
    y=PluckedString("koto",brightness=.70,position=.07).physical_note(freq,max(dur,.08),velocity=vel,sr=sr,material="nylon")
    y=_v19_pluck_anchor(y,freq,sr,vel,.58,.8)
    return F.lowpass(F.peak(y,300,sr,1.5,3.),3800,sr,.7)*.82

@instrument("sitar", "Shared-core steel string with nonlinear jawari buzzing bridge layer.", family="plucked", span=("C3","C5"), buzz=(0.45,"0..1 bridge buzz"))
def sitar(freq,dur,sr=DEFAULT_SR,vel=1.0,buzz=.45):
    from .physics import PluckedString
    y=PluckedString("sitar",brightness=.72,position=.06).physical_note(freq,max(dur,.08),velocity=vel,sr=sr,material="steel")
    y=_v19_pluck_anchor(y,freq,sr,vel,.52,.8)
    t=np.arange(len(y))/sr
    b=F.peak(F.highpass(np.tanh(4.2*y),650,sr),2200,sr,1.15,3.)*(1-np.exp(-t/.045))*np.exp(-t/.95)
    return F.lowpass(y+float(np.clip(buzz,0,1))*b,3900,sr,.72,order=2)*.72

@instrument("cello", "Shared-core wound bowed cello with a pitch-stable body radiation anchor.", family="bowed", span=("C2","C5"), attack=(0.1,"s"), vibrato=(1.0,"0 = none .. 2 = wide"),
            model=("waveguide", "waveguide | elastoplastic (Matusiak & Chatziioannou 2024: their measured cello G "
                                "string and fitted bristle friction; the bow accelerates from rest, so the attack "
                                "is a real Helmholtz transient; attack and vibrato ignored)"))
def cello(freq,dur,sr=DEFAULT_SR,vel=1.0,attack=.1,vibrato=1.0,model="waveguide"):
    if model == "elastoplastic":
        from .physical import bow
        v = float(np.clip(vel, 0.0, 1.0))
        # their Guettler-diagram transients: f_N 0.8-3.9 N, a 0.35-2.1 m/s^2 (Tables IV-V); UNSOURCED: 0.1-0.3 m/s
        y = bow.bowed(freq, dur, float(sr), force=0.8 + 3.1 * v, speed=0.1 + 0.2 * v, accel=0.35 + 1.7 * v)
        y = _body(y, sr, [(fc, fc / bw, 4.0 * g) for fc, bw, g in CELLO_BODY], min(3000., .38 * sr))   # UNSOURCED: 4 dB per gain unit
        return _physical_note_fade(y / (np.max(np.abs(y)) + 1e-12), sr, 0.005, .13) * 1.93   # as loud as the waveguide cello
    y=_blown(freq,dur+.13,sr,vel,dict(tilt=2.0,formants=CELLO_BODY,corner=1500.,slope=3.2),        # one pitch source: _ONE_SOURCE
             dict(tilt=1.35,formants=CELLO_BODY,corner=2400.,slope=3.2),attack=attack,release=.13,bloom=attack,
             vib=(5.1,.14*vibrato,.14),top=3600.)
    return _physical_note_fade(F.lowpass(y,min(3000.,.38*sr),sr,.72,order=3),sr,max(.008,attack),.13)*.56

# v0.20 local-interaction migrations -------------------------------------------------
@instrument("pizzicato", "Shared-core finger pizzicato with compliant local contact.", family="bowed", span=("G2","E6"))
def pizzicato(freq,dur,sr=DEFAULT_SR,vel=1.0):
    from .physics import PluckedString
    y=PluckedString("pizzicato",brightness=.42,position=.27).local_note(freq,max(dur,.07),technique='finger',
            velocity=vel,sr=sr,hardness=.24)
    y=_v19_pluck_anchor(y,freq,sr,vel,.48,.42)
    return F.lowpass(y,min(3600,.37*sr),sr,.68)*.78

@instrument("slap_bass", "Shared wound-string slap bass with fret-contact transients.", family="plucked", span=("E1","G4"))
def slap_bass(freq,dur,sr=DEFAULT_SR,vel=1.0):
    from .physics import PluckedString
    y=PluckedString("finger_bass",brightness=.72,position=.48).local_note(freq,max(dur,.08),technique='slap',
            velocity=vel,sr=sr)
    y=_v19_pluck_anchor(y,freq,sr,vel,.43,.55)
    return F.lowpass(np.tanh(1.25*y),min(4300,.39*sr),sr,.72)*.88


# -----------------------------------------------------------------------------
# Plucked strings on the physical string core, as it is (v0.23).
#
# The earlier wrappers above mixed the string half and half with a decaying sine at the note's
# pitch (the "anchors"), because the core played flat at the top, lost 3.5 % of its energy every
# period (a note was gone in 0.2 s) and carried a large DC offset. Those three faults are fixed in
# strings.py, so the string itself is the sound here: the pluck is a finger / nail / pick contact
# whose duration shortens with velocity (a hard pluck is brighter, not just louder), the pluck
# position combs the spectrum, the decay is the string's own T60, and note-off is a damper.
# Woodhouse, "Perceptual thresholds for acoustical guitar models", Acta Acustica 98 (2012), Table I, strings 1-6:
# open note Hz, T N, m g/m, EI 1e-6 N m^2, eta_F 1e-5, eta_B 1e-2, eta_A 1/s. (a) D'Addario Pro Arte nylon,
# (b) Martin 80/20 bronze steel, new.
_WOODHOUSE = {
    "nylon": ((329.6, 70.3, 0.38, 130, 40, 2.4, 1.5), (246.9, 53.4, 0.52, 160, 40, 2.0, 1.2), (196.0, 58.3, 0.90, 310, 14, 2.0, 1.7),
              (146.8, 71.2, 1.95, 51, 5, 2.0, 1.2), (110.0, 73.9, 3.61, 40, 7, 2.5, 0.9), (82.4, 71.6, 6.24, 57, 2, 2.0, 1.2)),
    "steel": ((329.6, 121, 0.57, 108, 7, 0.7, 2.5), (246.9, 135, 1.13, 336, 6, 1.1, 1.0), (196.0, 182, 2.42, 234, 11, 1.8, 1.2),
              (146.8, 185, 4.37, 255, 9, 1.1, 1.3), (110.0, 169, 7.13, 327, 11, 0.4, 1.2), (82.4, 147, 11.02, 667, 5, 0.3, 1.7)),
}


def woodhouse_tau(freq, fk, material="nylon"):
    """Decay time (s) of the partials ``fk`` of a ``freq`` note on the guitar string that plays it (the highest
    open string at or below it, stopped to length L0 f_open / freq): Valette's loss factor as fitted by Woodhouse,
    eta_n = [T (eta_F + eta_A / w_n) + EI eta_B (n pi / L)^2] / [T + EI (n pi / L)^2] (eq. 4), tau = 2 / (eta w).
    UNSOURCED: open length L0 = 0.65 m (a common classical / folk scale)."""
    rows = _WOODHOUSE[material]
    row = next((r for r in rows if r[0] <= freq * 1.001), rows[-1])
    fo, T, m, EI, eF, eB, eA = row
    EI, eF, eB = EI * 1e-6, eF * 1e-5, eB * 1e-2
    L = 0.65 * fo / max(freq, fo)
    w = 2 * np.pi * np.asarray(fk, float)
    kn = np.maximum(np.rint(np.asarray(fk, float) / freq), 1.0) * np.pi / L
    eta = (T * (eF + eA / w) + EI * eB * kn * kn) / (T + EI * kn * kn)
    return 2.0 / (eta * w)


def bank_loss_fit(f0, fk, tau, sr):
    """(dc, hf, smoothing) of genny's one-pole loss filter H = dc L + hf (1 - L), L = (1 - s) / (1 - s z^-1),
    fitted to the loop gains g_k = exp(-1 / (f0 tau_k)) with Bank & Valimaki's weights w_k = 1 / (g_k - 1)^4
    (IEEE SPL 10(1), 2003, eqs 1-2), |H| <= 1."""
    from scipy.optimize import least_squares
    g = np.exp(-1.0 / (f0 * np.asarray(tau, float)))
    w = 1.0 / np.maximum(1.0 - g, 1e-7) ** 2                 # sqrt of the weight, for least_squares
    z = np.exp(-1j * 2 * np.pi * np.asarray(fk, float) / sr)

    def mag(p):
        dc, hf, s = p
        Lz = (1 - s) / (1 - s * z)
        return np.abs(dc * Lz + hf * (1 - Lz))
    r = least_squares(lambda p: w * (mag(p) - g), x0=[g[0], g[-1], 0.5], bounds=([0, 0, 0], [1, 1, 0.99]))
    dc, hf, s = r.x
    return float(min(dc, 0.999999)), float(min(hf, dc)), float(s)


def _pluck_string(freq, dur, sr, vel, *, material="steel", kind="pick", hardness=0.6, position=0.2, t60=3.0,
                  tail=1.2, release=0.1, wound=None, length=None, detune=0.0, mute=0.0, glide=0.0):
    """One plucked note. `t60` is the mid-register ring time (higher strings ring shorter, as they
    do); `tail` seconds are rendered past note-off and `release` is the damper's time constant.
    `wound` = winding material for wound bass strings; `detune` (cents) adds the second string of a
    course; `mute` is palm damping at the bridge; `glide` (cents at full velocity) is the tension-modulation
    pitch excess at the pluck, which grows with the elongation, i.e. with velocity squared."""
    from .strings import DispersiveString, StringExciter, StringPhysicalProperties, WoundStringPhysicalProperties
    ring = float(dur) + float(tail)
    hard = float(np.clip(hardness * (0.55 + 0.45 * vel), 0.0, 1.0))
    t60 = float(t60) * _kt(freq, 1.0, 0.45)

    def one(f, pos):
        if wound:
            props = WoundStringPhysicalProperties.from_frequency(f, length_m=length, winding_material=wound)
            props.t60_s = t60
        else:
            props = StringPhysicalProperties.from_frequency(f, length_m=length, material=material, t60_s=t60)
        s = DispersiveString(props, sr=sr)
        if not wound and material in _WOODHOUSE:
            # measured string losses instead of one T60 for every partial (Woodhouse 2012 + Bank & Valimaki 2003)
            from .graphsolver import FrequencyDependentLoss
            fk = f * np.arange(1, max(2, int(0.45 * sr / f)) + 1)
            dc, hf, sm = bank_loss_fit(f, fk, woodhouse_tau(f, fk, material), sr)
            s.x_loss = FrequencyDependentLoss(dc, hf, sm)
            s.y_loss = FrequencyDependentLoss(dc ** 1.0002, hf * 0.999, sm)
            s.x_delay.delay = s.loop_delay(f)
            s.y_delay.delay = s.loop_delay(f * 2.0 ** (props.polarization_split_cents / 1200.0), plane="y")
        return s.render_exciter(ring, StringExciter(kind, hard, pos), velocity=1.0, palm_mute=mute,
                                glide_cents=float(glide) * float(vel) ** 2)

    y = one(freq, position)
    if detune:
        y = 0.54 * y + 0.46 * one(freq * 2 ** (detune / 1200.0), min(0.95, position + 0.012))
    n = y.shape[0]
    y = y * _off(n, dur, sr, release)
    return y / (np.max(np.abs(y)) + 1e-9) * (0.5 + 0.5 * vel)


@instrument("harp", "Concert harp: nylon string plucked with the fingertip near the middle, long ring.", family="plucked", span=("C2", "C7"), glide=(0.0, "cents a hard pluck starts sharp and glides down as it rings (tension modulation; 0 off, 10-40 audible)"))
def harp(freq, dur, sr=DEFAULT_SR, vel=1.0, glide=0.0):
    y = _pluck_string(freq, dur, sr, vel, glide=glide, material="nylon", kind="finger", hardness=0.45, position=0.38, t60=4.0, tail=2.0, release=0.35)
    return F.lowpass(F.peak(y, 220, sr, 0.8, 2.5), min(4200.0, 0.42 * sr), sr, 0.65)


@instrument("guitar", "Nylon-string classical guitar: warm, woody body.", family="plucked", span=("E2", "E5"), glide=(0.0, "cents a hard pluck starts sharp and glides down as it rings (tension modulation; 0 off, 10-40 audible)"),
            body=(0, "which guitar: 0 the nominal body, any other number a different instrument (real spread of mode pitch and Q)"),
            body_size=(0.5, "0 small box .. 0.5 as built .. 1 big box (warps every body mode)"))
def guitar(freq, dur, sr=DEFAULT_SR, vel=1.0, glide=0.0, body=0, body_size=0.5):
    y = _pluck_string(freq, dur, sr, vel, glide=glide, material="nylon", kind="nail", hardness=0.5, position=0.24, t60=3.0, tail=1.4, release=0.12)
    return F.lowpass(guitar_body(y, sr, int(body), body_size), min(3500.0, 0.42 * sr), sr, 0.7)


@instrument("steel_guitar", "Steel-string acoustic guitar: bright strum and fingerpicking (folk, country, pop).", family="plucked", span=("E2", "E6"), glide=(0.0, "cents a hard pluck starts sharp and glides down as it rings (tension modulation; 0 off, 10-40 audible)"),
            body=(0, "which guitar: 0 the nominal body, any other number a different instrument (real spread of mode pitch and Q)"),
            body_size=(0.5, "0 small box .. 0.5 as built .. 1 big box (warps every body mode)"))
def steel_guitar(freq, dur, sr=DEFAULT_SR, vel=1.0, glide=0.0, body=0, body_size=0.5):
    y = _pluck_string(freq, dur, sr, vel, glide=glide, material="steel", kind="pick", hardness=0.7, position=0.16, t60=4.5, tail=1.6, release=0.12)
    return F.lowpass(guitar_body(y, sr, int(body), body_size, steel=True), min(4600.0, 0.4 * sr), sr, 0.7)


@instrument("electric_guitar", "Clean electric guitar: round single-coil tone with long sustain (add `distortion` or `chorus` on the layer for rock or jangle).",
            family="plucked", span=("E2", "E6"), tone=(2800, "pickup tone Hz"),
            pickup=("linear", "linear | single | humbucker | rail (measured magnetic nonlinearity, Novak 2018)"), glide=(0.0, "cents a hard pluck starts sharp and glides down as it rings (tension modulation; 0 off, 10-40 audible)"))
def electric_guitar(freq, dur, sr=DEFAULT_SR, vel=1.0, tone=2800, pickup="linear", glide=0.0):
    y = _pluck_string(freq, dur, sr, vel, glide=glide, material="steel", kind="pick", hardness=0.6, position=0.2, t60=5.5, tail=1.2, release=0.08)
    if pickup != "linear":
        y = magnetic_pickup(y, sr, pickup, swing_mm=0.5 + 1.5 * float(np.clip(vel, 0.0, 1.0)))
    return F.lowpass(F.highpass(y, 70, sr), float(tone), sr, 1.3)


# Novak et al., DAFx-18 fig. 9: eq. 6 fitted to three Seymour Duncan pickups (A, L_eq mm, r_eq mm)
PICKUPS = {"single": (21.51e-3, 12.98, 2.77), "humbucker": (40.67e-3, 9.86, 1.34), "rail": (47.46e-3, 13.11, 1.88)}


def magnetic_pickup(y, sr, kind="single", gap_mm=3.0, swing_mm=2.0):
    """String displacement ``y`` (any scale; its peak is set to ``swing_mm``) at ``gap_mm`` from the pole piece
    through Novak's static flux law phi(x) = A ((x + L) / cbrt(r^2 + (x + L)^2) - x / cbrt(r^2 + x^2)), RMS-matched
    to the input (the coil voltage is d phi / dt, eq. 4). Their measurement: d0 = 3 mm, 2 mm swing
    on SH-2N gives more than 20 harmonics; at 5 mm and beyond the three pickups sound alike and cleaner.
    The string models already output the signal a linear pickup gives, so the law is applied to it as a
    displacement and not differentiated: what is added is the curvature of phi, without a second +6 dB/oct
    tilt (deviation)."""
    if kind not in PICKUPS:
        raise ValueError(f"unknown pickup {kind!r}; choose from {['linear', *PICKUPS]}")
    A, L, r = PICKUPS[kind]
    x = gap_mm + swing_mm * y / (np.max(np.abs(y)) + 1e-12)
    phi = A * ((x + L) / np.cbrt(r * r + (x + L) ** 2) - x / np.cbrt(r * r + x * x))
    u = -(phi - np.mean(phi))                     # flux falls as the string moves away
    return u * (np.sqrt(np.mean(y ** 2)) / (np.sqrt(np.mean(u ** 2)) + 1e-12))


@instrument("banjo", "Banjo: twangy string over a drum head, quick decay (bluegrass, country, folk).", family="plucked", span=("C3", "C7"), glide=(0.0, "cents a hard pluck starts sharp and glides down as it rings (tension modulation; 0 off, 10-40 audible)"))
def banjo(freq, dur, sr=DEFAULT_SR, vel=1.0, glide=0.0):
    y = _pluck_string(freq, dur, sr, vel, glide=glide, material="steel", kind="pick", hardness=0.9, position=0.08, t60=1.0, tail=0.6, release=0.06)
    y = y + 0.6 * F.bandpass(y, 600, sr, 2.0)          # the head
    return F.lowpass(F.highpass(y, 150, sr), min(4500.0, 0.42 * sr), sr, 0.7) * 0.8


@instrument("mandolin", "Mandolin: paired steel strings; set `tremolo` to re-pick held notes (Italian, bluegrass, folk).",
            family="plucked", span=("G3", "E7"), tremolo=(0.0, "picks per second on held notes (12 = classic tremolo, 0 = off)"), glide=(0.0, "cents a hard pluck starts sharp and glides down as it rings (tension modulation; 0 off, 10-40 audible)"))
def mandolin(freq, dur, sr=DEFAULT_SR, vel=1.0, tremolo=0.0, glide=0.0):
    def one(d, v):
        return _pluck_string(freq, d, sr, v, glide=glide, material="steel", kind="pick", hardness=0.8, position=0.12, t60=1.4, tail=0.7,
                             release=0.05, detune=5.0)
    if tremolo > 0 and dur * tremolo >= 1.5:
        step = 1.0 / tremolo
        y = mix([(one(step, vel * (1.0 if i % 2 == 0 else 0.82)), i * step) for i in range(int(dur * tremolo))], sr)
    else:
        y = one(dur, vel)
    return F.lowpass(F.highpass(y, 150, sr), min(4500.0, 0.42 * sr), sr, 0.7)


@instrument("koto", "Koto: string plucked hard near the bridge with ivory picks (Japanese, East Asian).", family="plucked", span=("C3", "C6"), glide=(0.0, "cents a hard pluck starts sharp and glides down as it rings (tension modulation; 0 off, 10-40 audible)"))
def koto(freq, dur, sr=DEFAULT_SR, vel=1.0, glide=0.0):
    y = _pluck_string(freq, dur, sr, vel, glide=glide, material="nylon", kind="pick", hardness=0.85, position=0.07, t60=2.6, tail=1.4, release=0.15)
    return _body(y, sr, [(300, 1.5, 3.0)], min(4200.0, 0.42 * sr))


@instrument("sitar", "Sitar: long string over a buzzing bridge, the buzz blooms just after the pluck (Indian classical, psychedelic).",
            family="plucked", span=("C3", "C5"), buzz=(0.45, "0..1 amount of bridge buzz"), glide=(0.0, "cents a hard pluck starts sharp and glides down as it rings (tension modulation; 0 off, 10-40 audible)"),
            model=("waveguide", "waveguide | fd (Siddiq 2012: a stiff string striking its curved flat bridge; the buzz "
                                "grows with velocity; glide ignored)"))
def sitar(freq, dur, sr=DEFAULT_SR, vel=1.0, buzz=0.45, glide=0.0, model="waveguide"):
    if model == "fd":
        from .fdstring import sitar_fd
        return sitar_fd(freq, dur, sr, vel, buzz)
    y = _pluck_string(freq, dur, sr, vel, glide=glide, material="steel", kind="pick", hardness=0.8, position=0.06, t60=3.5, tail=1.6, release=0.2)
    t = np.arange(y.shape[0]) / sr
    b = F.peak(F.highpass(np.tanh(6.0 * y), 700, sr), 2400, sr, 1.2, 4.0) * (1 - np.exp(-t / 0.05)) * np.exp(-t / 0.9)
    return F.lowpass(y + float(np.clip(buzz, 0, 1)) * b, 4200, sr, 0.7, order=2) * 0.8


@instrument("pizzicato", "Pizzicato strings: short, woody finger-plucked violin/cello notes (playful, sneaky, classical).",
            family="plucked", span=("G2", "E6"), glide=(0.0, "cents a hard pluck starts sharp and glides down as it rings (tension modulation; 0 off, 10-40 audible)"))
def pizzicato(freq, dur, sr=DEFAULT_SR, vel=1.0, glide=0.0):
    y = _pluck_string(freq, min(dur, 0.25), sr, vel, glide=glide, material="gut", kind="finger", hardness=0.3, position=0.25, t60=0.55,
                      tail=0.4, release=0.08)
    return _body(y, sr, [(290, 1.5, 3.0), (460, 1.5, 3.0), (1000, 1.0, 2.0)], min(3200.0, 0.4 * sr))


@instrument("upright_bass", "Upright (double) bass, plucked: deep woody thump with a short singing tail (jazz walking lines, rockabilly, folk).",
            family="bass", span=("E1", "G3"), glide=(0.0, "cents a hard pluck starts sharp and glides down as it rings (tension modulation; 0 off, 10-40 audible)"))
def upright_bass(freq, dur, sr=DEFAULT_SR, vel=1.0, glide=0.0):
    y = _pluck_string(freq, dur, sr, vel, glide=glide, kind="finger", hardness=0.25, position=0.3, t60=2.0, tail=0.8, release=0.1,
                      wound="nickel", length=float(np.clip(0.86 * 110 / max(freq, 20), 0.70, 1.08)))
    n = y.shape[0]
    thump = F.lowpass(O.white(n, seed=41), 300, sr) * decay_env(n, 0.012, sr) * 0.5 * vel
    return _body(y + thump, sr, [(100, 1.2, 3.0), (210, 1.5, 2.0)], 1400)


@instrument("finger_bass", "Electric bass guitar, fingerstyle: round, even, sustaining (rock, pop, soul, reggae).",
            family="bass", span=("E1", "G3"), tone=(1700, "pickup tone Hz"), glide=(0.0, "cents a hard pluck starts sharp and glides down as it rings (tension modulation; 0 off, 10-40 audible)"))
def finger_bass(freq, dur, sr=DEFAULT_SR, vel=1.0, tone=1700, glide=0.0):
    y = _pluck_string(freq, dur, sr, vel, glide=glide, kind="finger", hardness=0.4, position=0.28, t60=4.0, tail=0.6, release=0.07,
                      wound="nickel", length=float(np.clip(0.864 * 110 / max(freq, 20), 0.72, 0.92)))
    return F.lowpass(np.tanh(1.5 * y) / 0.9, float(tone), sr, 1.0)


@instrument("slap_bass", "Slapped electric bass: percussive pop on top of a scooped, punchy low end (funk, disco).",
            family="bass", span=("E1", "G4"))
def slap_bass(freq, dur, sr=DEFAULT_SR, vel=1.0):
    from .strings import DispersiveString, WoundStringPhysicalProperties
    props = WoundStringPhysicalProperties.from_frequency(freq, length_m=float(np.clip(0.864 * 110 / max(freq, 20), 0.5, 0.92)))
    props.t60_s = 2.6 * _kt(freq, 1.0, 0.45)
    y = DispersiveString(props, sr=sr).render_slap(dur + 0.5, velocity=vel)     # thumb strike + string hitting the frets
    y = y * _off(y.shape[0], dur, sr, 0.05)
    y = F.peak(y / (np.max(np.abs(y)) + 1e-9) * (0.5 + 0.5 * vel), 500, sr, 1.0, -5.0)
    return F.lowpass(F.highpass(y, 35, sr), min(4000.0, 0.4 * sr), sr, 0.7)


# String section and the other flue voices (2026-10-10) -------------------------------------------------
_SECTION_SEED = 1729


@instrument("strings", "String section: individual bowed players with their own vibrato, a few cents and a few milliseconds apart, "
            "centred on the written pitch. Short notes are bowed short.",
            family="bowed", span=("C2", "C6"), attack=(0.12, "s of bow attack on a long note (a short note takes 40 % of its length)"),
            release=(0.3, "s"), players=(6, "2..12 players"),
            scatter=(0.0, "cents: standard deviation of the players' mean pitch (Ternstrom: 0-5 preferred, 14 tolerable); above 0 the section beats slowly"),
            vibrato=(0.12, "0 = none .. 2 = wide; each player has their own, so more than about 0.2 makes the section's level heave"))
def strings(freq, dur, sr=DEFAULT_SR, vel=1.0, attack=0.12, release=0.3, players=6, scatter=0.0, vibrato=0.12):
    """A section is its players, not one detuned oscillator. This voice was six sawtooth waves spread a quarter of
    a semitone ("supersaw") under a 0.25 s attack: it measured 14-16 cents off the written pitch with a 20-32 %
    wobble in loudness, and could not speak a sixteenth note.

    Each player is the library's solo bowed voice (harmonics through the instrument's body formants). Between
    players: the mean pitch differs by `scatter` cents standard deviation (Ternstrom 1993, pitch scatter in
    ensembles: listeners prefer 0-5 cents and tolerate 14), with the section's mean on the written pitch; onsets
    differ by 20 ms standard deviation (the timing spread reported for string quartet players); each has its own
    vibrato rate (5.2-6.4 Hz, UNSOURCED range) and onset delay. The same section plays every note.

    Defaults are set by measurement of the level of a held note (standard deviation over mean): independent
    vibratos at 0.6 moved it by 30 % at A4 and 50 % at A5, a 3 cent scatter by 21-44 %; at vibrato 0.12 and no
    scatter it is 1 % at C3, 6 % at A4 and 12 % at A5."""
    rng = np.random.default_rng(_SECTION_SEED)
    n_pl = int(np.clip(players, 2, 12))
    cents = rng.normal(0.0, 1.0, n_pl)
    cents = (cents - cents.mean()) / (cents.std() + 1e-9) * float(np.clip(scatter, 0.0, 30.0))
    late = np.abs(rng.normal(0.0, 0.020, n_pl))
    late -= late.min()
    rates, delays = rng.uniform(5.2, 6.4, n_pl), rng.uniform(0.12, 0.30, n_pl)
    att = float(min(attack, 0.4 * dur))
    rel = float(release if dur >= 0.25 else min(release, 0.12))
    body = CELLO_BODY if freq < 196.0 else VIOLIN_BODY              # below the violin's open G the cellos carry the line
    corner = (1500.0, 2400.0) if freq < 196.0 else (2000.0, 3000.0)
    out = np.zeros(_n(dur + float(late.max()), sr, rel) + 8)
    for i in range(n_pl):
        v = _blown(freq * 2.0 ** (cents[i] / 1200.0), dur, sr, vel,
                   dict(tilt=1.8, formants=body, corner=corner[0], slope=3.0), dict(tilt=1.2, formants=body, corner=corner[1], slope=3.0),
                   attack=att, release=rel, bloom=max(att, 0.03), vib=(float(rates[i]), 0.12 * vibrato, float(delays[i])), top=4200.0)
        k = int(late[i] * sr)
        out[k:k + len(v)] += v[:len(out) - k]
    return F.lowpass(out, min(4200.0, 0.44 * sr), sr, 0.7) * (0.9 / np.sqrt(n_pl))


def _pipe_air(n, sr, freq, env, odd, seed):
    """Breath of a blown pipe: noise through the pipe's own resonances, following the square of the blowing envelope
    (turbulence power goes as the jet's dynamic pressure). `odd`: a stopped pipe, odd modes only."""
    w = O.white(n, seed=seed)
    y = np.zeros(n)
    for j, h in enumerate((1, 3, 5) if odd else (1, 2, 3)):
        if h * freq < 0.45 * sr:
            y += F.bandpass(w, h * freq, sr, 9.0) / (1.0 + j)
    return y * env * env


@instrument("pan_flute", "Pan flute: a stopped pipe blown across its end. Odd harmonics, a breath that sits in the pipe's own resonances, a soft chiff.",
            family="wind", span=("C4", "C7"), breath=(0.3, "air 0..1"), vibrato=(0.0, "0 = steady .. 1"))
def pan_flute(freq, dur, sr=DEFAULT_SR, vel=1.0, breath=0.3, vibrato=0.0):
    """There is no sourced jet model of a stopped pipe in the library (STK's flute is an open pipe), so the tone stays
    a harmonic stack with the stopped pipe's odd series. What changed: the vibrato is a parameter and off by default
    (it was fixed, and a near-sine with vibrato under a reverb flutters), and the breath is no longer a broad band
    of noise laid over the note: it passes through the pipe's odd modes and follows the blowing pressure squared."""
    rel = 0.15
    n = _n(dur, sr, rel)
    f = _vibrato(freq, n, sr, rate=4.5, depth=0.06 * float(np.clip(vibrato, 0, 2)), delay=0.3, ramp=0.4, scoop=0.4) if vibrato > 0 else freq
    amps = np.array([1.0, 0.06, 0.18, 0.02, 0.05]) * _roll(np.arange(1, 6) * freq, 2400.0) / float(_roll(freq, 2400.0))
    x = _harmonics(f, n, sr, amps)
    env = adsr(dur, 0.05, 0.1, 0.8, rel, sr)[:n]
    env = np.pad(env, (0, n - len(env)))
    air = _pipe_air(n, sr, freq, env, True, 56) * breath * 0.6
    chiff = _air(n, sr, _cut(freq, 2, 800, 2400), 1.2, 58) * decay_env(n, 0.035, sr) * breath * 1.2
    return (x * env + air + chiff * env) * 0.6 * (0.6 + 0.4 * vel)


@instrument("whistle", "Human whistle: a near-pure tone that slides into each note, with a little breath in its own resonance.",
            family="wind", span=("C5", "C7"), vibrato=(0.3, "0 = steady .. 1"))
def whistle(freq, dur, sr=DEFAULT_SR, vel=1.0, vibrato=0.3):
    rel = 0.1
    n = _n(dur, sr, rel)
    f = _vibrato(freq, n, sr, rate=5.8, depth=0.12 * float(np.clip(vibrato, 0, 2)), delay=0.2, ramp=0.3, scoop=0.7)
    x = O.sine(f, n, sr) + 0.04 * O.sine(f * 2, n, sr)
    env = adsr(dur, 0.04, 0.1, 0.85, rel, sr)[:n]
    env = np.pad(env, (0, n - len(env)))
    return (x * env + _pipe_air(n, sr, freq, env, False, 59) * 0.2) * 0.6 * (0.6 + 0.4 * vel)
