"""Sound chips, modelled chip by chip: the NES 2A03 APU, the Game Boy APU and the Yamaha YM2612.

Registers
    layer type  `chip`       a whole chip: one voice per hardware channel, the chip's own mixer and output filters
    instrument  `nes_pulse`, `nes_triangle`, `nes_noise`, `gb_pulse`, `gb_wave`, `gb_noise`, `ym2612`

Two ways to use a chip:
    hardware_accurate   the `chip` layer. Every channel is monophonic, pitches are the chip's timer values, the
                        channels meet in the chip's mixer (non-linear on the NES) and leave through its filters.
    retro_stylized      the instruments. One channel alone, as many notes at once as the score asks for, in tune,
                        rounded like the other retro voices (`mode: "hardware_accurate"` gives the raw channel).

Sources (pages kept in out/research/chips):
    nesdev_APU*.wiki          NESdev wiki: APU, Pulse, Triangle, Noise, Envelope, Sweep, Frame Counter, Mixer.
    pandocs_Audio*.md         Pan Docs: Audio, Audio Registers, Audio details.
    ymfm_fm.ipp, ymfm_opn.*   ymfm by Aaron Giles (BSD 3-Clause): the OPN FM core, ported below.
What a music driver does on top of the hardware (a table of volumes, duties or pitch offsets stepped once per
video frame) is not hardware and has no single source: it is marked UNSOURCED where it is defined.

Both cores step the chip's own clock and average blocks of 16 clock ticks; the result is resampled to `sr`.
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np
from scipy.signal import butter, lfilter

from . import filters as F
from .core import DEFAULT_SR, samples
from .instruments import _soften, instrument
from .notes import parse_pitch_list, parse_sequence
from .retro import _fade, _seed, _to_sr
from .spec import SpecError, layer_type

DEC = 16                 # clock ticks averaged into one sample before resampling
FRAME_HZ = 60.0          # UNSOURCED: a music driver runs once per video frame; 60 Hz is used for both machines

# ============================================================================ NES 2A03
NES_CPU = {"ntsc": 1789773.0, "pal": 1662607.0}                                      # APU: clock rates
NES_NOISE = {"ntsc": (4, 8, 16, 32, 64, 96, 128, 160, 202, 254, 380, 508, 762, 1016, 2034, 4068),
             "pal": (4, 8, 14, 30, 60, 88, 118, 148, 188, 236, 354, 472, 708, 944, 1890, 3778)}   # APU Noise: CPU cycles per LFSR clock
# APU Frame Counter, 4-step mode: quarter-frame clocks after 3728.5, 7456.5, 11185.5 and 14914.5 APU cycles
# (two CPU cycles each), the sequence repeating every 14915 (NTSC) or 16627 (PAL) APU cycles.
NES_QUARTER = {"ntsc": (7457, 14913, 22371, 29829, 29830), "pal": (8313, 16627, 24939, 33253, 33254)}
# APU Pulse: the sequencer reads these tables downward from index 0 (0, 7, 6, ...): 12.5 %, 25 %, 50 %, 25 % negated
NES_DUTY = np.array([[0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0]], np.int64)
NES_TRI = np.array(list(range(15, -1, -1)) + list(range(16)), np.int64)              # APU Triangle: the 32-step sequence


def _nes_tables():
    """APU Mixer, the exact formula: pulse_out by pulse1 + pulse2, tnd_out by (triangle, noise, dmc)."""
    pulse = np.zeros(31)
    pulse[1:] = 95.88 / (8128.0 / np.arange(1, 31) + 100.0)
    t, n, d = np.meshgrid(np.arange(16), np.arange(16), np.arange(128), indexing="ij")
    s = t / 8227.0 + n / 12241.0 + d / 22638.0
    tnd = np.zeros(s.shape)
    tnd[s > 0] = 159.79 / (1.0 / s[s > 0] + 100.0)
    return pulse, tnd.reshape(-1)


NES_PULSE_TAB, NES_TND_TAB = _nes_tables()


@nb.njit(cache=True)
def _nes_core(n_out, dec, seg, off, dmc, duty, tri, pulse_tab, tnd_tab):
    """The four tone channels, CPU cycle by CPU cycle. `seg` rows are (cycle, period, volume, aux, reset) and the
    rows of channel c are seg[off[c]:off[c + 1]]: pulse 1, pulse 2, triangle, noise. aux is the duty (pulse) or the
    short-mode flag (noise); the triangle's volume is its gate. Returns the mixer output, 0..1."""
    out = np.empty(n_out, np.float32)
    ptr = off[:4].copy()
    per = np.zeros(4, np.int64)
    vol = np.zeros(4, np.int64)
    aux = np.zeros(4, np.int64)
    tim = np.zeros(4, np.int64)
    step = np.zeros(4, np.int64)
    lfsr = 1                                               # APU Noise: loaded with 1 on power-up
    big = 1 << 62
    nxt = big
    for c in range(4):
        if ptr[c] < off[c + 1] and seg[ptr[c], 0] < nxt:
            nxt = seg[ptr[c], 0]
    cyc = 0
    for i in range(n_out):
        acc = 0.0
        for _ in range(dec):
            if cyc >= nxt:
                nxt = big
                for c in range(4):
                    while ptr[c] < off[c + 1] and seg[ptr[c], 0] <= cyc:
                        r = ptr[c]
                        per[c] = seg[r, 1]
                        vol[c] = seg[r, 2]
                        aux[c] = seg[r, 3]
                        if seg[r, 4] != 0 and c < 2:       # a write to $4003/$4007 restarts the sequencer
                            step[c] = 0
                            tim[c] = per[c]
                        ptr[c] += 1
                    if ptr[c] < off[c + 1] and seg[ptr[c], 0] < nxt:
                        nxt = seg[ptr[c], 0]
            if cyc & 1 == 0:                               # the pulse timers count APU cycles
                for c in range(2):
                    if tim[c] <= 0:
                        tim[c] = per[c]
                        step[c] = (step[c] - 1) & 7
                    else:
                        tim[c] -= 1
            if vol[2] != 0 and per[2] >= 2:                # the triangle's timer counts CPU cycles; it holds its value when gated off
                if tim[2] <= 0:                            # UNSOURCED: periods under 2 are ultrasonic and are held, as emulators do
                    tim[2] = per[2]
                    step[2] = (step[2] + 1) & 31
                else:
                    tim[2] -= 1
            tim[3] -= 1
            if tim[3] <= 0:
                tim[3] = per[3] if per[3] > 0 else 4
                fb = (lfsr ^ (lfsr >> (6 if aux[3] != 0 else 1))) & 1
                lfsr = (lfsr >> 1) | (fb << 14)
            p = 0
            for c in range(2):
                if per[c] >= 8 and duty[aux[c], step[c]] != 0:
                    p += vol[c]
            nz = vol[3] if lfsr & 1 == 0 else 0
            acc += pulse_tab[p] + tnd_tab[(tri[step[2]] << 11) | (nz << 7) | dmc]
            cyc += 1
        out[i] = acc / dec
    return out


# ============================================================================ Game Boy
GB_BASE = 2097152.0      # Pan Docs: the wave channel's period divider is clocked at 2097152 Hz, the pulse ones at half that
GB_TICK = 4096           # DIV-APU, 512 Hz: every 4096 base ticks. Envelope every 8 (64 Hz), frequency sweep every 4 (128 Hz)
# Pan Docs NR11, read off the waveform figure as digital values (the figure draws the analog output, which is inverted)
GB_DUTY = np.array([[0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 0]], np.int64)
GB_HPF = {"dmg": 0.999958, "cgb": 0.998943}                # Pan Docs: capacitor charge factor per tick of 4194304 Hz


@nb.njit(cache=True)
def _gb_core(n_out, dec, seg, off, ram, duty, left, right):
    """The four channels, one step per tick of 2097152 Hz. `seg` rows as in `_nes_core`; period is the number of
    base ticks per waveform step, aux the duty (pulse), the output level 0..3 (wave) or the 7-bit flag (noise).
    Each DAC maps digital 0..15 to analog +1..-1 (Pan Docs: the slope is negative). Returns (left, right)."""
    out = np.empty((n_out, 2), np.float32)
    ptr = off[:4].copy()
    per = np.ones(4, np.int64)
    vol = np.zeros(4, np.int64)
    aux = np.zeros(4, np.int64)
    tim = np.ones(4, np.int64)
    step = np.zeros(4, np.int64)
    dig = np.zeros(4, np.int64)
    lfsr = 0
    big = 1 << 62
    nxt = big
    for c in range(4):
        if ptr[c] < off[c + 1] and seg[ptr[c], 0] < nxt:
            nxt = seg[ptr[c], 0]
    cyc = 0
    for i in range(n_out):
        al = 0.0
        ar = 0.0
        for _ in range(dec):
            if cyc >= nxt:
                nxt = big
                for c in range(4):
                    while ptr[c] < off[c + 1] and seg[ptr[c], 0] <= cyc:
                        r = ptr[c]
                        per[c] = seg[r, 1] if seg[r, 1] > 0 else 1
                        vol[c] = seg[r, 2]
                        aux[c] = seg[r, 3]
                        if seg[r, 4] != 0:                 # trigger: the period divider is reloaded
                            tim[c] = per[c]
                            if c == 2:
                                step[2] = 0                # the wave index is reset, the sample buffer is not refilled
                            elif c == 3:
                                lfsr = 0                   # the LFSR is set to 0
                        ptr[c] += 1
                    if ptr[c] < off[c + 1] and seg[ptr[c], 0] < nxt:
                        nxt = seg[ptr[c], 0]
            for c in range(2):
                tim[c] -= 1
                if tim[c] <= 0:
                    tim[c] = per[c]
                    step[c] = (step[c] + 1) & 7
                dig[c] = vol[c] if duty[aux[c], step[c]] != 0 else 0
            if vol[2] != 0:
                tim[2] -= 1
                if tim[2] <= 0:
                    tim[2] = per[2]
                    step[2] = (step[2] + 1) & 31
                    dig[2] = ram[step[2]] >> (aux[2] - 1) if aux[2] > 0 else 0   # the level shifts the digital value
            else:
                dig[2] = 0
            if vol[3] != 0:
                tim[3] -= 1
                if tim[3] <= 0:
                    tim[3] = per[3]
                    b = 1 - ((lfsr ^ (lfsr >> 1)) & 1)     # 1 when bits 0 and 1 are identical
                    lfsr = (lfsr & 0x7FFF) | (b << 15)
                    if aux[3] != 0:
                        lfsr = (lfsr & ~0x80) | (b << 7)
                    lfsr >>= 1
                dig[3] = vol[3] if lfsr & 1 != 0 else 0
            else:
                dig[3] = 0
            for c in range(4):
                a = 1.0 - dig[c] / 7.5
                al += left[c] * a
                ar += right[c] * a
            cyc += 1
        out[i, 0] = al / (4.0 * dec)
        out[i, 1] = ar / (4.0 * dec)
    return out


GB_WAVES = {     # 32 four-bit samples. UNSOURCED: starting shapes of genny's own, wave RAM has no factory content worth copying
    "triangle": [min(i, 31 - i) for i in range(32)],
    "saw": [i // 2 for i in range(32)],
    "square": [15] * 16 + [0] * 16,
    "pulse25": [15] * 8 + [0] * 24,
    "sine": [int(round(7.5 + 7.5 * math.sin(2 * math.pi * (i + 0.5) / 32))) for i in range(32)],
    "organ": [int(round(7.5 + 4.5 * math.sin(2 * math.pi * (i + 0.5) / 32) + 3.0 * math.sin(4 * math.pi * (i + 0.5) / 32))) for i in range(32)],
    "bass": [int(round(7.5 + 5.0 * math.sin(2 * math.pi * (i + 0.5) / 32) + 2.5 * math.sin(6 * math.pi * (i + 0.5) / 32))) for i in range(32)],
}


def _wave_ram(wave) -> np.ndarray:
    if isinstance(wave, str) and wave in GB_WAVES:
        vals = GB_WAVES[wave]
    elif isinstance(wave, str):                            # 32 hex digits, as the bytes of FF30-FF3F read left to right
        vals = [int(ch, 16) for ch in wave.replace(" ", "")]
    else:
        vals = [int(v) for v in wave]
    if len(vals) != 32 or min(vals) < 0 or max(vals) > 15:
        raise SpecError(f"wave: 32 values 0..15, a 32-digit hex string or one of {sorted(GB_WAVES)}")
    return np.array(vals, np.int64)


# ============================================================================ notes -> register writes
PATCH = {"volume": 15, "vol": None, "decay": None, "loop": False, "env": 0, "duty": 2, "arp": None, "pitch": None,
         "vib": 0.0, "vib_hz": 6.0, "vib_delay": 0.15, "sweep": None, "gate": 1.0, "period": -1, "short": False,
         "linear": 0, "level": 1}
PATCH_HELP = {
    "volume": "0..15, the channel volume when no envelope runs",
    "vol": "driver table: a list of volumes 0..15, one per 1/60 s, the last one held (a plucked note: [15, 11, 8, 6, 4, 3])",
    "decay": "NES hardware envelope: divider V 0..15, the volume falls 15 -> 0 one step every (V + 1)/240 s",
    "loop": "NES hardware envelope: start again at 15 when it reaches 0",
    "env": "Game Boy hardware envelope: -1..-7 falls one step every n/64 s, 1..7 rises, 0 = off",
    "duty": "0..3 = 12.5, 25, 50, 75 %, or a driver table of them, one per 1/60 s, the last one held",
    "arp": "driver table: semitone offsets, one per 1/60 s, cycling (a chord on one channel: [0, 4, 7])",
    "pitch": "driver table: semitone offsets, one per 1/60 s, the last one held (a drum: [12, 7, 3, 0])",
    "vib": "vibrato depth in semitones, applied to the timer once per 1/60 s",
    "vib_hz": "vibrato rate", "vib_delay": "seconds before the vibrato starts",
    "sweep": "hardware sweep. NES [period P 0..7, shift 1..7, negate 0|1] every (P + 1)/120 s; Game Boy pulse 1 [pace 1..7, step 1..7, direction 1 up | -1 down] every pace/128 s",
    "gate": "fraction of the step the note sounds for",
    "period": "noise: the period index (NES 0..15) or [shift, divider] (Game Boy); -1 = from the note's pitch",
    "short": "noise: the short LFSR mode (93 or 31 steps on the NES, 7-bit on the Game Boy), a buzzing pitched noise",
    "linear": "NES triangle: linear counter, the note stops after this many 1/240 s (0 = off)",
    "level": "Game Boy wave: output level 1 = 100 %, 2 = 50 %, 3 = 25 %",
}


def _tab(m, f, cycle=False):
    if m is None:
        return 0
    if not isinstance(m, (list, tuple)):
        return m
    return m[f % len(m)] if cycle else m[min(f, len(m) - 1)]


def _note_rows(system, kind, ch, c0, c1, freq, vel, p, region, rows):
    """Appends the register states of one note to `rows` as (cycle, period, volume, aux, reset).

    The note's own hardware clocks (NES frame counter, Game Boy DIV-APU) and the driver's 60 Hz frames are walked
    in time order; a row is written each time the state of the channel changes."""
    nes = system == "nes"
    clk = NES_CPU[region] if nes else GB_BASE
    q4 = NES_QUARTER[region] if nes else None
    frame = clk / FRAME_HZ
    sweep = p["sweep"] if kind == "pulse" and (nes or ch == 0) else None
    vib = float(p["vib"])
    hard_env = p["decay"] is not None if nes else int(p["env"]) != 0

    def reg(f, c):
        """The timer value of driver frame f."""
        semis = _tab(p["arp"], f, True) + _tab(p["pitch"], f)
        t = (c - c0) / clk - float(p["vib_delay"])
        if vib and t > 0:
            semis += vib * math.sin(2 * math.pi * float(p["vib_hz"]) * t)
        hz = freq * 2.0 ** (semis / 12.0)
        if kind == "noise":
            if nes:
                tab = NES_NOISE[region]
                if p["period"] != -1:
                    return tab[int(p["period"]) & 15]
                # UNSOURCED: a note picks the period whose 93-step sequence repeats nearest its pitch (genny's convention)
                return min(tab, key=lambda v: abs(math.log(clk / v / (93.0 * hz))))
            if p["period"] != -1:
                shift, div = p["period"]
            else:                                          # UNSOURCED: likewise, with the 127-step sequence of the 7-bit mode
                shift, div = min(((s, d) for s in range(14) for d in range(8)),
                                 key=lambda sd: abs(math.log(262144.0 / ((sd[1] or 0.5) * 2 ** sd[0]) / (127.0 * hz))))
            return int(8 * (div or 0.5) * 2 ** shift)     # base ticks per LFSR clock: 2097152 / (262144 / (divider * 2^shift))
        if nes:                                            # f = fCPU / (16 (t + 1)) pulse, fCPU / (32 (t + 1)) triangle
            return int(min(max(round(clk / ((16.0 if kind == "pulse" else 32.0) * hz) - 1.0), 0), 2047))
        return int(min(max(2048 - round((131072.0 if kind == "pulse" else 65536.0) / hz), 0), 2047))   # period value

    level = 15 if nes else int(p["volume"])               # hardware envelope state
    start, div = True, 0
    sdiv, reload = 0, True
    t = reg(0, c0)
    on = True
    f = kq = 0

    def emit(c, reset=0):
        if p["vol"] is not None:
            v = int(_tab(p["vol"], f))
        elif hard_env:
            v = level
        else:
            v = int(p["volume"])
        v = min(max(int(round(v * vel)), 1 if v else 0), 15)
        if kind == "noise":
            per, aux = t, int(bool(p["short"]))
        elif nes:
            per, aux = t, int(_tab(p["duty"], f)) & 3
            if kind == "triangle":
                v = 1
            elif sweep and not sweep[2] and t + (t >> int(sweep[1])) > 0x7FF:
                v = 0                                      # APU Sweep: a target period above $7FF mutes the channel
        else:
            per = (2048 - t) * (2 if kind == "pulse" else 1)
            aux = int(_tab(p["duty"], f)) & 3 if kind == "pulse" else int(p["level"])
            if kind == "wave":
                v = 1
        row = (int(c), per, v if on else 0, aux, reset)
        if rows and rows[-1][0] == row[0] and not reset and len(rows) > mark:
            rows[-1] = row[:4] + (rows[-1][4],)
        elif len(rows) == mark or rows[-1][1:4] != row[1:4] or reset:
            rows.append(row)

    mark = len(rows)
    if not nes and sweep and int(sweep[1]) and sweep[2] > 0 and t + (t >> int(sweep[1])) > 0x7FF:
        on = False                                         # Pan Docs: the overflow check runs at the trigger
    emit(c0, 1)
    while True:
        cq = c0 + (q4[kq % 4] + (kq // 4) * q4[4] if nes else GB_TICK * (kq + 1))
        cf = c0 + int(round((f + 1) * frame))
        c = min(cq, cf)
        if c >= c1:
            break
        if cq <= cf:                                       # a hardware clock
            if nes:
                if hard_env:                               # APU Envelope, clocked every quarter frame
                    if start:
                        start, level, div = False, 15, int(p["decay"])
                    elif div == 0:
                        div = int(p["decay"])
                        level = level - 1 if level > 0 else (15 if p["loop"] else 0)
                    else:
                        div -= 1
                if kind == "triangle" and p["linear"] and kq + 1 >= int(p["linear"]):
                    on = False
                if sweep and kq % 4 in (1, 3):             # APU Sweep, clocked every half frame
                    amount = t >> int(sweep[1])
                    target = max(t - amount - (1 if ch == 0 else 0), 0) if sweep[2] else t + amount   # pulse 1 adds the ones' complement
                    if sdiv == 0 and int(sweep[1]) and t >= 8 and target <= 0x7FF:
                        t = target
                    if sdiv == 0 or reload:
                        sdiv, reload = int(sweep[0]), False
                    else:
                        sdiv -= 1
            else:
                pace = abs(int(p["env"]))
                if hard_env and kq % 8 == 7 and (kq // 8 + 1) % pace == 0:
                    level = min(max(level + (1 if p["env"] > 0 else -1), 0), 15)
                if sweep and kq % 4 == 3 and int(sweep[0]) and (kq // 4 + 1) % int(sweep[0]) == 0 and on:
                    new = t + (t >> int(sweep[1])) * (1 if sweep[2] > 0 else -1)
                    if new > 0x7FF:
                        on = False                         # the channel is turned off
                    elif int(sweep[1]):
                        t = max(new, 0)
                        if sweep[2] > 0 and t + (t >> int(sweep[1])) > 0x7FF:
                            on = False                     # the check is run again on the new period
            kq += 1
        else:                                              # a driver frame
            f += 1
            if not sweep:
                t = reg(f, c)
        emit(c)
    rows.append((int(c1), rows[-1][1], 0, rows[-1][3], 0))


def _segments(system, lanes, region):
    """lanes: per channel, a list of notes (t0, t1, hz, vel, patch) in seconds. Returns (seg, off) for the core."""
    names = SYSTEMS[system]
    clk = NES_CPU[region] if system == "nes" else GB_BASE
    rows, off = [], [0]
    for ch, name in enumerate(names):
        kind = "pulse" if name.startswith("pulse") else name
        notes = sorted(lanes.get(name, []), key=lambda n: n[0])
        for i, (t0, t1, hz, vel, p) in enumerate(notes):
            if i + 1 < len(notes):
                t1 = min(t1, notes[i + 1][0])              # one voice per channel: the next note cuts this one
            c0, c1 = int(round(t0 * clk)), int(round(t1 * clk))
            if c1 > c0:
                _note_rows(system, kind, ch, c0, c1, hz, vel, p, region, rows)
        off.append(len(rows))
    seg = np.array(rows, np.int64).reshape(-1, 5)
    return seg, np.array(off, np.int64)


SYSTEMS = {"nes": ("pulse1", "pulse2", "triangle", "noise"), "gb": ("pulse1", "pulse2", "wave", "noise")}
NES_GAIN = 2.4           # level trims: a full-volume 50 % pulse alone peaks near 0.35 on both chips
GB_GAIN = 1.4


def _patch(src: dict, where: str) -> dict:
    p = dict(PATCH)
    for k, v in src.items():
        if k in PATCH:
            p[k] = v
        elif k not in ("steps", "notes", "step", "vel", "transpose", "pan", "wave"):
            raise SpecError(f"{where}: unknown key {k!r} (known: {', '.join(sorted(PATCH))})")
    return p


def render_chip(system, lanes, dur, sr, region="ntsc", filters=None, dmc=0, wave="triangle", pan=None, model="dmg"):
    """Renders `dur` seconds of a whole chip. Returns mono (NES, or a Game Boy with every channel centred) or stereo."""
    seg, off = _segments(system, lanes, region)
    if system == "nes":
        clk = NES_CPU[region]
        x = _nes_core(int(dur * clk / DEC) + 1, DEC, seg, off, int(min(max(dmc, 0), 127)), NES_DUTY, NES_TRI,
                      NES_PULSE_TAB, NES_TND_TAB).astype(np.float64)
        y = _to_sr(x - x[0], clk / DEC, sr)
        # APU Mixer: the NES follows the DACs with first-order filters, a high-pass at 90 Hz, another at 440 Hz
        # and a low-pass at 14 kHz; the Famicom only specifies a high-pass at 37 Hz.
        chain = {"nes": ((90.0, "high"), (440.0, "high"), (14000.0, "low")), "famicom": ((37.0, "high"),),
                 "none": ((20.0, "high"),)}[filters or "nes"]
        for fc, kind in chain:
            y = lfilter(*butter(1, fc, kind, fs=sr), y)
        return y[:samples(dur, sr)] * NES_GAIN
    pan = pan or {}
    left = np.array([0.0 if pan.get(n, "C") == "R" else 1.0 for n in SYSTEMS["gb"]])
    right = np.array([0.0 if pan.get(n, "C") == "L" else 1.0 for n in SYSTEMS["gb"]])
    x = _gb_core(int(dur * GB_BASE / DEC) + 1, DEC, seg, off, _wave_ram(wave), GB_DUTY, left, right).astype(np.float64)
    rate = GB_BASE / DEC
    k = GB_HPF[model] ** (4194304.0 / rate)                # Pan Docs: the charge factor for any output rate
    x = lfilter([1.0, -1.0], [1.0, -k], x - x[0], axis=0)  # out = in - capacitor; capacitor = in - out * k
    y = np.stack([_to_sr(x[:, 0], rate, sr), _to_sr(x[:, 1], rate, sr)], axis=1)[:samples(dur, sr)] * GB_GAIN
    return y[:, 0] if np.array_equal(left, right) else y


@layer_type("chip")
def chip_layer(layer: dict, sr: int, q: float) -> np.ndarray:
    """A whole sound chip (`system`: nes | gb), hardware-accurate: one voice per channel, the chip's mixer and filters.

    `channels` maps a hardware channel (nes: pulse1, pulse2, triangle, noise; gb: pulse1, pulse2, wave, noise) to
    a lane {"steps": "...", ...patch} or to a list of lanes sharing the channel, as a driver shares the noise
    channel between its drums: the latest note wins. A chord in `steps` is played as a 60 Hz arpeggio."""
    system = str(layer.get("system", "nes")).lower()
    if system not in SYSTEMS:
        raise SpecError(f"chip layer: system must be one of {sorted(SYSTEMS)}")
    if system == "ym2612":
        return _fm_layer(layer, sr, q)
    lanes, end, pan = {}, 0.0, {}
    for name, lane_list in (layer.get("channels") or {}).items():
        if name not in SYSTEMS[system]:
            raise SpecError(f"chip layer: {system} has no channel {name!r} (it has {', '.join(SYSTEMS[system])})")
        for lane in lane_list if isinstance(lane_list, list) else [lane_list]:
            p = _patch(lane, f"chip layer, {name}")
            if lane.get("pan"):
                pan[name] = str(lane["pan"]).upper()[:1]
            t, tr = 0.0, 2.0 ** (float(lane.get("transpose", 0)) / 12.0)
            for st in parse_sequence(lane.get("steps", ""), float(lane.get("step", 0.25))) if isinstance(lane.get("steps", ""), str) else lane["steps"]:
                hz = st["pitches"] if not isinstance(st["pitches"], str) else parse_pitch_list(st["pitches"])
                if hz:
                    pp = p
                    if len(hz) > 1 and p["arp"] is None:   # a chord on one channel is an arpeggio
                        pp = dict(p, arp=[12.0 * math.log2(h / hz[0]) for h in hz])
                    lanes.setdefault(name, []).append((t * q, (t + st["dur"] * float(p["gate"])) * q, hz[0] * tr,
                                                       float(lane.get("vel", 1.0)) * float(st.get("vel", 1.0)), pp))
                t += st["dur"]
            end = max(end, t * q)
    if system == "gb" and str(layer.get("mode", "")) == "retro_stylized":
        pan = {}
    filters = layer.get("filters") or ("none" if layer.get("mode") == "retro_stylized" else "nes")
    return render_chip(system, lanes, max(end, 0.05), sr, region=str(layer.get("region", "ntsc")).lower(), filters=filters,
                       dmc=int(layer.get("dmc", 0)), wave=layer.get("wave", "triangle"), pan=pan,
                       model=str(layer.get("model", "dmg")).lower())


# ============================================================================ single-channel voices
def _voice(system, channel, freq, dur, sr, vel, mode, bright, gain, extra=None, **patch):
    """One note on one channel of the chip. `retro_stylized` (default): in tune and rounded;
    `hardware_accurate`: the timer's own pitch and the raw output."""
    p = dict(PATCH, **patch)
    rel = 0.01
    lanes = {channel: [(0.0, dur, float(freq), 1.0, p)]}
    y = render_chip(system, lanes, dur + rel, sr * 2, filters="none", **(extra or {}))
    exact = str(mode) == "hardware_accurate"
    kind = "pulse" if channel.startswith("pulse") else channel
    if not exact and kind != "noise":                      # the timer rounds the pitch: play the note back at the written one
        clk = NES_CPU["ntsc"] if system == "nes" else GB_BASE
        if system == "nes":
            got = clk / ((16.0 if kind == "pulse" else 32.0) * (round(clk / ((16.0 if kind == "pulse" else 32.0) * freq) - 1.0) + 1.0))
        else:
            ref = 131072.0 if kind == "pulse" else 65536.0
            got = ref / max(round(ref / freq), 1)
        y = _to_sr(y, sr * 2 * freq / got, sr * 2)
    y = _to_sr(y, sr * 2, sr)[:samples(dur + rel, sr)]
    y = F.dc_block(y, sr)
    if not exact and kind != "noise":
        y = _soften(y, freq, sr, harm=6, lo=2200, hi=3800, bright=bright)
    elif not exact:                                        # the raw LFSR is far brighter than any other voice of the catalog
        fc = 3000.0 * 2.0 ** ((bright - 0.5) * 4.0)
        if fc < sr * 0.45:
            y = F.lowpass(F.lowpass(y, fc, sr, 0.707), fc, sr, 0.707)    # 24 dB an octave
    return _fade(y.copy(), sr, 1.0, rel * 1000.0) * (gain * (0.5 + 0.5 * vel))


_MODE = ("retro_stylized", "retro_stylized = in tune and rounded | hardware_accurate = the chip's timer pitch, raw")
_BRIGHT = (0.5, "retro_stylized: 0 = dull .. 0.5 = default .. 1 = unfiltered")
_H = PATCH_HELP


@instrument("nes_pulse", "NES 2A03 pulse channel: 8-step sequencer at four duties, 4-bit volume, hardware envelope and sweep, driver tables.",
            family="retro", span=("C2", "C7"), duty=(2, _H["duty"]), volume=(15, _H["volume"]), vol=(None, _H["vol"]),
            decay=(None, _H["decay"]), loop=(False, _H["loop"]), arp=(None, _H["arp"]), pitch=(None, _H["pitch"]),
            vib=(0.0, _H["vib"]), vib_hz=(6.0, _H["vib_hz"]), vib_delay=(0.15, _H["vib_delay"]), sweep=(None, _H["sweep"]),
            mode=_MODE, bright=_BRIGHT)
def nes_pulse(freq, dur, sr=DEFAULT_SR, vel=1.0, duty=2, volume=15, vol=None, decay=None, loop=False, arp=None, pitch=None,
              vib=0.0, vib_hz=6.0, vib_delay=0.15, sweep=None, mode="retro_stylized", bright=0.5):
    return _voice("nes", "pulse1", freq, dur, sr, vel, mode, bright, 1.0, duty=duty, volume=volume, vol=vol, decay=decay,
                  loop=loop, arp=arp, pitch=pitch, vib=vib, vib_hz=vib_hz, vib_delay=vib_delay, sweep=sweep)


@instrument("nes_triangle", "NES 2A03 triangle channel: a 32-step, 4-bit staircase triangle with no volume control (the console's bass).",
            family="retro", span=("C1", "C6"), linear=(0, _H["linear"]), arp=(None, _H["arp"]), pitch=(None, _H["pitch"]),
            vib=(0.0, _H["vib"]), vib_hz=(6.0, _H["vib_hz"]), vib_delay=(0.15, _H["vib_delay"]), mode=_MODE, bright=(1.0, _BRIGHT[1]))
def nes_triangle(freq, dur, sr=DEFAULT_SR, vel=1.0, linear=0, arp=None, pitch=None, vib=0.0, vib_hz=6.0, vib_delay=0.15,
                 mode="retro_stylized", bright=1.0):
    return _voice("nes", "triangle", freq, dur, sr, vel, mode, bright, 1.0, linear=linear, arp=arp, pitch=pitch, vib=vib,
                  vib_hz=vib_hz, vib_delay=vib_delay)


@instrument("nes_noise", "NES 2A03 noise channel: 15-bit LFSR at 16 fixed rates; the short mode is a buzzing, pitched noise.",
            family="retro", span=("C3", "C8"), period=(-1, _H["period"]), short=(False, _H["short"]), volume=(15, _H["volume"]),
            vol=(None, _H["vol"]), decay=(None, _H["decay"]), loop=(False, _H["loop"]), pitch=(None, _H["pitch"]), mode=_MODE)
def nes_noise(freq, dur, sr=DEFAULT_SR, vel=1.0, period=-1, short=False, volume=15, vol=None, decay=None, loop=False, pitch=None,
              mode="retro_stylized"):
    return _voice("nes", "noise", freq, dur, sr, vel, mode, 0.5, 1.0, period=period, short=short, volume=volume, vol=vol,
                  decay=decay, loop=loop, pitch=pitch)


@instrument("gb_pulse", "Game Boy pulse channel: 8-step duty, 4-bit volume with the hardware envelope, frequency sweep, driver tables.",
            family="retro", span=("C2", "C7"), duty=(2, _H["duty"]), volume=(15, _H["volume"]), vol=(None, _H["vol"]),
            env=(0, _H["env"]), arp=(None, _H["arp"]), pitch=(None, _H["pitch"]), vib=(0.0, _H["vib"]), vib_hz=(6.0, _H["vib_hz"]),
            vib_delay=(0.15, _H["vib_delay"]), sweep=(None, _H["sweep"]), mode=_MODE, bright=_BRIGHT)
def gb_pulse(freq, dur, sr=DEFAULT_SR, vel=1.0, duty=2, volume=15, vol=None, env=0, arp=None, pitch=None, vib=0.0, vib_hz=6.0,
             vib_delay=0.15, sweep=None, mode="retro_stylized", bright=0.5):
    return _voice("gb", "pulse1", freq, dur, sr, vel, mode, bright, 1.0, duty=duty, volume=volume, vol=vol, env=env, arp=arp,
                  pitch=pitch, vib=vib, vib_hz=vib_hz, vib_delay=vib_delay, sweep=sweep)


@instrument("gb_wave", "Game Boy wave channel: 32 four-bit samples of wave RAM read at the note's rate (bass, leads, any shape).",
            family="retro", span=("C1", "C6"), wave=("triangle", "32 values 0..15, 32 hex digits, or " + " | ".join(GB_WAVES)),
            level=(1, _H["level"]), arp=(None, _H["arp"]), pitch=(None, _H["pitch"]), vib=(0.0, _H["vib"]), vib_hz=(6.0, _H["vib_hz"]),
            vib_delay=(0.15, _H["vib_delay"]), mode=_MODE, bright=(1.0, _BRIGHT[1]))
def gb_wave(freq, dur, sr=DEFAULT_SR, vel=1.0, wave="triangle", level=1, arp=None, pitch=None, vib=0.0, vib_hz=6.0, vib_delay=0.15,
            mode="retro_stylized", bright=1.0):
    return _voice("gb", "wave", freq, dur, sr, vel, mode, bright, 1.0, extra={"wave": wave}, level=level, arp=arp, pitch=pitch,
                  vib=vib, vib_hz=vib_hz, vib_delay=vib_delay)


@instrument("gb_noise", "Game Boy noise channel: 15-bit LFSR, or 7-bit for a metallic pitched buzz, with the hardware envelope.",
            family="retro", span=("C3", "C8"), period=(-1, _H["period"]), short=(False, _H["short"]), volume=(15, _H["volume"]),
            vol=(None, _H["vol"]), env=(0, _H["env"]), pitch=(None, _H["pitch"]), mode=_MODE)
def gb_noise(freq, dur, sr=DEFAULT_SR, vel=1.0, period=-1, short=False, volume=15, vol=None, env=0, pitch=None,
             mode="retro_stylized"):
    return _voice("gb", "noise", freq, dur, sr, vel, mode, 0.5, 1.0, period=period, short=short, volume=volume, vol=vol,
                  env=env, pitch=pitch)


# ============================================================================ Yamaha YM2612 (OPN2)
# Ported from ymfm (Aaron Giles, BSD 3-Clause; out/research/chips/ymfm_fm.ipp, ymfm_opn.cpp): the log-sine and power
# tables, the envelope generator with its increment table and key scaling, the phase generator with detune, the
# eight algorithms with operator 1 feedback, the LFO, the 9-bit channel output and its DAC discontinuity.
# Not ported: SSG-EG envelopes, the channel 3 multi-frequency mode, CSM, the channel 6 PCM DAC, timers.
OPN_CLOCK = 7670453.0    # UNSOURCED: the Mega Drive's NTSC master clock / 7 (general knowledge); the chip runs at clock / 144
# ymfm prints both tables; they equal these formulas entry for entry (checked against the file)
OPN_SIN = np.array([round(-math.log2(math.sin((i + 0.5) * math.pi / 512)) * 256) for i in range(256)], np.int64)
OPN_SIN = np.array([(OPN_SIN[(~i if i & 0x100 else i) & 0xFF]) | ((i >> 9) << 15) for i in range(1024)], np.int64)
OPN_POW = np.array([(round((2 ** ((255 - i) / 256) - 1) * 1024) | 0x400) << 2 for i in range(256)], np.int64)
OPN_INC = np.array(
    [0x00000000, 0x00000000, 0x10101010, 0x10101010, 0x10101010, 0x10101010, 0x11101110, 0x11101110]
    + [0x10101010, 0x10111010, 0x11101110, 0x11111110] * 10
    + [0x11111111, 0x21112111, 0x21212121, 0x22212221, 0x22222222, 0x42224222, 0x42424242, 0x44424442,
       0x44444444, 0x84448444, 0x84848484, 0x88848884, 0x88888888, 0x88888888, 0x88888888, 0x88888888], np.int64)
OPN_DETUNE = np.array(
    [[0, 0, 1, 2]] * 4 + [[0, 1, 2, 2], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4], [0, 1, 3, 4],
                          [0, 1, 3, 5], [0, 2, 4, 5], [0, 2, 4, 6], [0, 2, 4, 6], [0, 2, 5, 7], [0, 2, 5, 8], [0, 3, 6, 8],
                          [0, 3, 6, 9], [0, 3, 7, 10], [0, 4, 8, 11], [0, 4, 8, 12], [0, 4, 9, 13], [0, 5, 10, 14],
                          [0, 5, 11, 16], [0, 6, 12, 17], [0, 6, 13, 19], [0, 7, 14, 20]] + [[0, 8, 16, 22]] * 4, np.int64)
OPN_PM = np.array([[0x77] * 8, [0x77, 0x77, 0x77, 0x77, 0x72, 0x72, 0x72, 0x72], [0x77, 0x77, 0x77, 0x72, 0x72, 0x72, 0x17, 0x17],
                   [0x77, 0x77, 0x72, 0x72, 0x17, 0x17, 0x12, 0x12], [0x77, 0x77, 0x72, 0x17, 0x17, 0x17, 0x12, 0x07],
                   [0x77, 0x77, 0x17, 0x12, 0x07, 0x07, 0x02, 0x01], [0x77, 0x77, 0x17, 0x12, 0x07, 0x07, 0x02, 0x01],
                   [0x77, 0x77, 0x17, 0x12, 0x07, 0x07, 0x02, 0x01]], np.int64)
OPN_LFO_MAX = np.array([109, 78, 72, 68, 63, 45, 9, 6], np.int64)


def _alg(op2in, op3in, op4in, op1out, op2out, op3out):
    return op2in | (op3in << 1) | (op4in << 4) | (op1out << 7) | (op2out << 8) | (op3out << 9)


OPN_ALG = np.array([_alg(1, 2, 3, 0, 0, 0), _alg(0, 5, 3, 0, 0, 0), _alg(0, 2, 6, 0, 0, 0), _alg(1, 0, 7, 0, 0, 0),
                    _alg(1, 0, 3, 0, 1, 0), _alg(1, 1, 1, 0, 1, 1), _alg(1, 0, 0, 0, 1, 1), _alg(0, 0, 0, 1, 1, 1)], np.int64)
OPN_CARRIERS = [(3,), (3,), (3,), (3,), (1, 3), (1, 2, 3), (1, 2, 3), (0, 1, 2, 3)]     # the operators that reach the output
OP_KEYS = ("ar", "dr", "sr", "rr", "sl", "tl", "ks", "mul", "dt", "am")                  # one operator, in this column order


@nb.njit(cache=True)
def _opn_vol(ph, att, sin_tab, pow_tab):
    s = sin_tab[ph & 0x3FF]
    a = (s & 0x7FFF) + (att << 2)
    r = pow_tab[a & 0xFF] >> (a >> 8)
    return -r if s & 0x8000 else r


@nb.njit(cache=True)
def _opn_core(n, ev, patches, lfo_rate, raw_dac, sin_tab, pow_tab, inc_tab, det_tab, pm_tab, lfo_max, alg_tab):
    """Six FM channels at clock / 144. `ev` rows are (sample, channel, key 1|0, patch, block_freq, level): a key-on
    loads `patches[patch]` = (alg, fb, ams, pms, pan, 4 x OP_KEYS) and adds `level` to the carriers' total level.
    raw_dac = 1 keeps the chip's 9-bit channel output and DAC step, 0 the full 14-bit operator sum."""
    out = np.zeros((n, 2), np.float32)
    phase = np.zeros((6, 4), np.int64)
    step = np.zeros((6, 4), np.int64)
    att = np.full((6, 4), 0x3FF, np.int64)
    state = np.full((6, 4), 3, np.int64)                  # 0 attack, 1 decay, 2 sustain, 3 release
    rate = np.zeros((6, 4, 4), np.int64)
    sus = np.zeros((6, 4), np.int64)
    tl = np.zeros((6, 4), np.int64)
    am = np.zeros((6, 4), np.int64)
    det = np.zeros((6, 4), np.int64)
    mul = np.ones((6, 4), np.int64)
    alg = np.zeros(6, np.int64)
    fb = np.zeros(6, np.int64)
    ams = np.zeros(6, np.int64)
    pms = np.zeros(6, np.int64)
    pan = np.full(6, 3, np.int64)
    bf = np.zeros(6, np.int64)
    fbk = np.zeros((6, 2), np.int64)
    fin = np.zeros(6, np.int64)
    opout = np.zeros(8, np.int64)
    used = np.zeros(6, np.int64)
    envc = 0
    lfoc = 0
    e = 0
    ne = ev.shape[0]
    for i in range(n):
        while e < ne and ev[e, 0] <= i:
            c = ev[e, 1]
            if ev[e, 2] != 0:
                P = patches[ev[e, 3]]
                alg[c], fb[c], ams[c], pms[c], pan[c] = P[0], P[1], P[2], P[3], P[4]
                bf[c] = ev[e, 4]
                used[c] = 1
                keycode = ((bf[c] >> 10) & 0xF) << 1 | ((0xFE80 >> ((bf[c] >> 7) & 0xF)) & 1)
                for o in range(4):
                    q = 5 + 10 * o
                    carrier = o == 3 or (alg_tab[alg[c]] >> (7 + o)) & 1 != 0
                    tl[c, o] = min(P[q + 5] + (ev[e, 5] if carrier else 0), 127) << 3
                    s = P[q + 4]
                    sus[c, o] = (s | ((s + 1) & 0x10)) << 5
                    ksr = keycode >> (P[q + 6] ^ 3)
                    raws = (P[q] * 2, P[q + 1] * 2, P[q + 2] * 2, P[q + 3] * 4 + 2)
                    for k in range(4):
                        rate[c, o, k] = 0 if raws[k] == 0 else min(raws[k] + ksr, 63)
                    d = det_tab[keycode, P[q + 8] & 3]
                    det[c, o] = -d if P[q + 8] & 4 else d
                    mul[c, o] = P[q + 7] * 2 if P[q + 7] else 1
                    am[c, o] = P[q + 9]
                    step[c, o] = ((((((bf[c] & 0x7FF) << 1) << ((bf[c] >> 11) & 7)) >> 2) + det[c, o]) & 0x1FFFF) * mul[c, o] >> 1
                    if state[c, o] != 0:                   # start_attack
                        state[c, o] = 0
                        phase[c, o] = 0
                        if rate[c, o, 0] >= 62:
                            att[c, o] = 0
            else:
                for o in range(4):
                    if state[c, o] < 3:
                        state[c, o] = 3
            e += 1
        envc += 1
        if envc & 3 == 3:                                  # the envelope generator runs once every three samples
            envc += 1
        lfo_am = 0
        lfo_pm = 0
        if lfo_rate >= 0:
            sub = lfoc & 0xFF
            lfoc += 1
            if sub >= lfo_max[lfo_rate]:
                lfoc += 0x101 - sub
            lfo_am = (lfoc >> 8) & 0x3F
            if (lfoc >> 14) & 1 == 0:
                lfo_am ^= 0x3F
            lfo_pm = (lfoc >> 10) & 7
            if (lfoc >> 13) & 1:
                lfo_pm ^= 7
            if (lfoc >> 14) & 1:
                lfo_pm = -lfo_pm
        else:
            lfo_am = 0x3F
        al = 0
        ar = 0
        for c in range(6):
            if used[c] == 0:
                if raw_dac:
                    al += 4
                    ar += 4
                continue
            fbk[c, 0] = fbk[c, 1]
            fbk[c, 1] = fin[c]
            for o in range(4):
                if envc & 3 == 0:                          # clock_envelope
                    ec = envc >> 2
                    if state[c, o] == 0 and att[c, o] == 0:
                        state[c, o] = 1
                    if state[c, o] == 1 and att[c, o] >= sus[c, o]:
                        state[c, o] = 2
                    r = rate[c, o, state[c, o]]
                    sh = r >> 2
                    ecs = (ec << sh) & 0xFFFFFFFF
                    if ecs & 0x7FF == 0:
                        inc = (inc_tab[r] >> (4 * ((ecs >> (11 if sh <= 11 else sh)) & 7))) & 0xF
                        if state[c, o] == 0:
                            if r < 62:
                                att[c, o] += (~att[c, o] * inc) >> 4
                        else:
                            att[c, o] = min(att[c, o] + inc, 0x3FF)
                st = step[c, o]
                if pms[c] != 0 and lfo_rate >= 0:          # compute_phase_step with the LFO's pitch modulation
                    fnum = (bf[c] & 0x7FF) << 1
                    apm = -lfo_pm if lfo_pm < 0 else lfo_pm
                    shifts = pm_tab[pms[c], apm & 7]
                    fb7 = (bf[c] >> 4) & 0x7F
                    adj = (fb7 >> (shifts & 0xF)) + (fb7 >> (shifts >> 4))
                    if pms[c] > 5:
                        adj <<= pms[c] - 5
                    adj >>= 2
                    fnum = (fnum + (-adj if lfo_pm < 0 else adj)) & 0xFFF
                    st = ((((fnum << ((bf[c] >> 11) & 7)) >> 2) + det[c, o]) & 0x1FFFF) * mul[c, o] >> 1
                phase[c, o] += st
            amoff = (lfo_am << 1) >> ((1 << (ams[c] ^ 3)) - 1)      # with the LFO off its AM output rests at 0x3F, as in ymfm
            opmod = 0
            if fb[c] != 0:
                opmod = (fbk[c, 0] + fbk[c, 1]) >> (10 - fb[c])
            v = 0
            if att[c, 0] <= 0x380:
                v = _opn_vol((phase[c, 0] >> 10) + opmod, min(att[c, 0] + (amoff if am[c, 0] else 0) + tl[c, 0], 0x3FF), sin_tab, pow_tab)
            fin[c] = v
            ops = alg_tab[alg[c]]
            opout[0] = 0
            opout[1] = v
            for o in range(1, 4):
                src = opout[ops & 1] if o == 1 else (opout[(ops >> 1) & 7] if o == 2 else opout[(ops >> 4) & 7])
                v = 0
                if att[c, o] <= 0x380:
                    v = _opn_vol((phase[c, o] >> 10) + (src >> 1), min(att[c, o] + (amoff if am[c, o] else 0) + tl[c, o], 0x3FF), sin_tab, pow_tab)
                if o == 1:
                    opout[2] = v
                    opout[5] = opout[1] + v
                elif o == 2:
                    opout[3] = v
                    opout[6] = opout[1] + v
                    opout[7] = opout[2] + v
            rs = 5 if raw_dac else 0
            lim = 256 if raw_dac else 8191 * 4
            res = v >> rs
            for o in range(3):
                if (ops >> (7 + o)) & 1:
                    res = min(max(res + (opout[o + 1] >> rs), -lim - 1), lim)
            if raw_dac:
                res = res - 3 if res < 0 else res + 4      # the YM2612's DAC steps across zero
                if pan[c] & 1:
                    al += res
                else:
                    al += 4
                if pan[c] & 2:
                    ar += res
                else:
                    ar += 4
            else:
                if pan[c] & 1:
                    al += res
                if pan[c] & 2:
                    ar += res
        k = 1.0 / (6.0 * 260.0) if raw_dac else 1.0 / (6.0 * 8191.0)
        out[i, 0] = al * k
        out[i, 1] = ar * k
    return out


def _op(ar=31, dr=0, sr=0, rr=8, sl=0, tl=0, ks=0, mul=1, dt=0, am=0):
    return dict(ar=ar, dr=dr, sr=sr, rr=rr, sl=sl, tl=tl, ks=ks, mul=mul, dt=dt, am=am)


# UNSOURCED: starting patches of genny's own, written from the algorithm diagrams; not taken from any game or bank
FM_PATCHES = {
    "sine": dict(alg=7, fb=0, ops=[_op(tl=127), _op(tl=127), _op(tl=127), _op(rr=10)]),
    "epiano": dict(alg=4, fb=2, ops=[_op(dr=14, sr=6, rr=7, sl=4, tl=46, mul=14, ks=2), _op(dr=7, sr=3, rr=7, sl=2, tl=4, ks=1),
                                     _op(dr=9, sr=2, rr=7, sl=3, tl=40, mul=1), _op(dr=6, sr=2, rr=7, sl=2, tl=4, dt=1)]),
    "bass": dict(alg=0, fb=5, ops=[_op(dr=10, sr=2, rr=8, sl=2, tl=36), _op(dr=12, sr=3, rr=8, sl=3, tl=44, mul=2),
                                   _op(dr=9, sr=2, rr=8, sl=2, tl=36), _op(dr=6, sr=2, rr=9, sl=1, tl=0)]),
    "brass": dict(alg=4, fb=6, ops=[_op(ar=20, dr=6, sr=1, rr=8, sl=2, tl=30), _op(ar=22, dr=4, sr=1, rr=9, sl=1, tl=4),
                                    _op(ar=18, dr=6, sr=1, rr=8, sl=2, tl=34, mul=2), _op(ar=22, dr=4, sr=1, rr=9, sl=1, tl=6, dt=2)]),
    "bell": dict(alg=4, fb=0, ops=[_op(dr=8, sr=4, rr=5, sl=3, tl=32, mul=7), _op(dr=5, sr=3, rr=4, sl=2, tl=4, mul=1),
                                   _op(dr=10, sr=5, rr=5, sl=4, tl=38, mul=11), _op(dr=6, sr=3, rr=4, sl=2, tl=14, mul=3)]),
    "organ": dict(alg=7, fb=4, ops=[_op(rr=12, tl=8), _op(rr=12, tl=12, mul=2), _op(rr=12, tl=20, mul=4), _op(rr=12, tl=4, mul=1)]),
    "lead": dict(alg=0, fb=6, pms=3, ops=[_op(tl=34), _op(dr=4, sl=1, tl=30, mul=2), _op(dr=3, sl=1, tl=26), _op(ar=28, dr=2, sl=1, rr=10)]),
    "strings": dict(alg=5, fb=5, pms=2, ops=[_op(ar=16, dr=3, sl=1, rr=6, tl=32), _op(ar=14, rr=6, tl=6, dt=3),
                                             _op(ar=15, rr=6, tl=14, mul=2), _op(ar=14, rr=6, tl=6, dt=7)]),
    "clav": dict(alg=2, fb=4, ops=[_op(dr=12, sr=4, rr=9, sl=3, tl=28, mul=3), _op(dr=14, sr=5, rr=9, sl=4, tl=36, mul=5),
                                   _op(dr=10, sr=3, rr=9, sl=2, tl=30), _op(dr=10, sr=4, rr=10, sl=3, ks=2)]),
    "pluck": dict(alg=4, fb=3, ops=[_op(dr=16, sr=8, rr=8, sl=5, tl=26, mul=3), _op(dr=12, sr=6, rr=8, sl=4, tl=2, ks=1),
                                    _op(dr=18, sr=8, rr=8, sl=6, tl=36, mul=1), _op(dr=13, sr=6, rr=8, sl=4, tl=6, mul=2)]),
}
OP_RANGE = dict(ar=31, dr=31, sr=31, rr=15, sl=15, tl=127, ks=3, mul=15, dt=7, am=1)


def _fm_patch(patch) -> list:
    """A patch name or {"alg", "fb", "ams", "pms", "ops": [4 operators]} -> the core's row (without the pan)."""
    if isinstance(patch, str):
        if patch not in FM_PATCHES:
            raise SpecError(f"ym2612: unknown patch {patch!r} (known: {', '.join(sorted(FM_PATCHES))})")
        patch = FM_PATCHES[patch]
    ops = patch.get("ops") or []
    if len(ops) != 4:
        raise SpecError("ym2612 patch: `ops` is a list of four operators")
    row = [int(patch.get("alg", 0)) & 7, int(patch.get("fb", 0)) & 7, int(patch.get("ams", 0)) & 3, int(patch.get("pms", 0)) & 7, 3]
    for op in ops:
        bad = set(op) - set(OP_KEYS)
        if bad:
            raise SpecError(f"ym2612 operator: unknown key {sorted(bad)[0]!r} (known: {', '.join(OP_KEYS)})")
        full = _op(**op)
        row += [int(min(max(full[k], 0), OP_RANGE[k])) for k in OP_KEYS]
    return row


def render_opn(lanes, dur, sr, clock=OPN_CLOCK, lfo=None, raw_dac=True):
    """lanes: channel index 0..5 -> notes (t0, t1, hz, vel, patch, pan 'L'|'R'|'C'). Returns stereo at `sr`."""
    fs = clock / 144.0
    rows, patches = [], []
    for c, notes in lanes.items():
        notes = sorted(notes, key=lambda n: n[0])
        for i, (t0, t1, hz, vel, patch, pan) in enumerate(notes):
            if i + 1 < len(notes):
                t1 = min(t1, notes[i + 1][0] - 2.0 / fs)   # the key goes up before the next note keys on
            row = _fm_patch(patch)
            row[4] = {"L": 1, "R": 2}.get(str(pan).upper()[:1], 3)
            if row not in patches:
                patches.append(row)
            f = hz * 2.0 ** 21 / fs                        # phase step = fnum * 2^block / 2 out of 2^20
            block = int(min(max(math.ceil(math.log2(max(f, 1e-9) / 2047.0)), 0), 7))
            fnum = int(min(round(f / 2 ** block), 2047))
            level = int(round(min(-20.0 * math.log10(max(vel, 1e-3)) / 0.75, 127)))   # 0.75 dB per total-level step
            rows.append((int(round(t0 * fs)), c, 1, patches.index(row), (block << 11) | fnum, level))
            rows.append((int(round(max(t1, t0) * fs)), c, 0, 0, 0, 0))
    rows.sort(key=lambda r: (r[0], r[2]))                  # at the same sample the key-off comes first
    ev = np.array(rows, np.int64).reshape(-1, 6)
    pt = np.array(patches or [[0] * 45], np.int64)
    x = _opn_core(int(dur * fs) + 1, ev, pt, -1 if lfo is None else int(lfo) & 7, 1 if raw_dac else 0,
                  OPN_SIN, OPN_POW, OPN_INC, OPN_DETUNE, OPN_PM, OPN_LFO_MAX, OPN_ALG).astype(np.float64)
    x -= x[0]
    return np.stack([_to_sr(x[:, 0], fs, sr), _to_sr(x[:, 1], fs, sr)], axis=1)[:samples(dur, sr)]


FM_GAIN = 2.2            # level trim: one full-level carrier peaks near 0.37


def _fm_layer(layer: dict, sr: int, q: float) -> np.ndarray:
    lanes, end = {}, 0.0
    for name, lane_list in (layer.get("channels") or {}).items():
        if name not in SYSTEMS["ym2612"]:
            raise SpecError(f"chip layer: ym2612 has no channel {name!r} (it has {', '.join(SYSTEMS['ym2612'])})")
        for lane in lane_list if isinstance(lane_list, list) else [lane_list]:
            bad = set(lane) - {"steps", "step", "vel", "transpose", "pan", "patch", "gate"}
            if bad:
                raise SpecError(f"chip layer, {name}: unknown key {sorted(bad)[0]!r}")
            t, tr = 0.0, 2.0 ** (float(lane.get("transpose", 0)) / 12.0)
            for st in parse_sequence(lane.get("steps", ""), float(lane.get("step", 0.25))):
                if len(st["pitches"]) > 1:
                    raise SpecError(f"chip layer, {name}: one note at a time per FM channel (spread a chord over fm1..fm6)")
                if st["pitches"]:
                    lanes.setdefault(SYSTEMS["ym2612"].index(name), []).append(
                        (t * q, (t + st["dur"] * float(lane.get("gate", 0.95))) * q, st["pitches"][0] * tr,
                         float(lane.get("vel", 1.0)) * st["vel"], lane.get("patch", "epiano"), lane.get("pan", "C")))
                t += st["dur"]
            end = max(end, t * q)
    y = render_opn(lanes, max(end, 0.05) + float(layer.get("tail", 0.5)), sr, clock=float(layer.get("clock", OPN_CLOCK)),
                   lfo=layer.get("lfo"), raw_dac=layer.get("mode") != "retro_stylized") * FM_GAIN
    # FM at whole-number ratios carries a DC term that moves with the envelopes; the console's output is AC-coupled.
    y = lfilter(*butter(1, 20.0, "high", fs=sr), y, axis=0)     # UNSOURCED: the 20 Hz corner is genny's choice
    return y[:, 0] if np.array_equal(y[:, 0], y[:, 1]) else y


SYSTEMS["ym2612"] = ("fm1", "fm2", "fm3", "fm4", "fm5", "fm6")


@instrument("ym2612", "Yamaha YM2612 (Mega Drive) FM channel: four operators, eight algorithms, the chip's envelopes, feedback and LFO.",
            family="retro", span=("C1", "C7"),
            patch=("epiano", "a patch name (" + " | ".join(sorted(FM_PATCHES)) + ") or {alg 0..7, fb 0..7, ams 0..3, pms 0..7, ops: [4 x {ar 0..31, dr 0..31, sr 0..31, rr 0..15, sl 0..15, tl 0..127, ks 0..3, mul 0..15, dt 0..7, am 0|1}]}"),
            lfo=(None, "LFO rate 0 (slowest) .. 7 (fastest) or null = off; it acts through the patch's pms (vibrato) and ams + am (tremolo)"),
            tail=(0.6, "seconds rendered after the key goes up, for the release"),
            mode=("retro_stylized", "retro_stylized = the full-resolution operator sum | hardware_accurate = the chip's 9-bit output and DAC step"))
def ym2612(freq, dur, sr=DEFAULT_SR, vel=1.0, patch="epiano", lfo=None, tail=0.6, mode="retro_stylized"):
    y = render_opn({0: [(0.0, dur, float(freq), 1.0, patch, "C")]}, dur + float(tail), sr, lfo=lfo,
                   raw_dac=str(mode) == "hardware_accurate")[:, 0]
    return _fade(F.dc_block(y, sr), sr, 0.5, 20.0) * (FM_GAIN * (0.4 + 0.6 * vel))
