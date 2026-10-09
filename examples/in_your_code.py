"""genny inside a project: generate the sounds a program needs, from code, instead of shipping files.

    uv run python examples/in_your_code.py        # writes out_code/*.wav and prints what it made

Every call below returns a numpy float array in -1..1 (mono: shape (n,), stereo: (n, 2)) at the sample
rate you ask for. The same arguments always give the same samples, so sounds can be generated at build
time, at start-up, or on demand, and cached by their arguments.
"""
from __future__ import annotations

import io
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np

import genny  # noqa: F401  (registers every instrument, sound effect and effect)
from genny import fx, spec
from genny.core import write_wav
from genny.identify import identify_machine
from genny.instruments import render_note
from genny.sfx import render_sfx

SR = 44100
OUT = Path("out_code")


# 1. One sound, by name and physical parameters -------------------------------------------------
glass = render_sfx("impact", SR, material="glass", size=0.2)
write_wav(OUT / "glass_tap.wav", glass, SR)


# 2. A cache keyed by the arguments: ask for a sound wherever you need it ---------------------------
@lru_cache(maxsize=256)
def sound(name: str, **params) -> np.ndarray:
    return render_sfx(name, SR, **params)


def wav_bytes(x: np.ndarray, sr: int = SR) -> bytes:
    """16-bit WAV in memory, for an audio API that takes bytes (pygame.mixer.Sound(buffer=...), a web
    response, an engine's audio stream) instead of a file."""
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1 if pcm.ndim == 1 else pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# 3. An event table: the interface of an application, generated at start-up -------------------------
UI = {event: sound("cyber_ui", kind=kind) for event, kind in {
    "menu_open": "scan", "ok": "confirm", "cancel": "deny", "save": "upload", "alert": "warning"}.items()}
for event, x in UI.items():
    write_wav(OUT / f"ui_{event}.wav", x, SR)


# 4. Variations: the same object never sounds twice the same -----------------------------------------
footsteps = [sound("impact", material="wood", size=0.3, seed=i) for i in range(6)]
coins = [render_note("glockenspiel", 880.0 * 2 ** (i / 12), 0.4, sr=SR, vel=0.7) for i in (0, 4, 7, 12)]
write_wav(OUT / "coin_run.wav", np.concatenate(coins), SR)


# 5. A parameter bank: an engine the game crossfades by rpm ------------------------------------------
ENGINE_RPM = (900, 1500, 2500, 4000, 6000)
engine_bank = {rpm: sound("pipe_engine", rpm=float(rpm), cylinders=4, load=0.6, dur=2.0) for rpm in ENGINE_RPM}
for rpm, x in engine_bank.items():
    write_wav(OUT / f"engine_{rpm}.wav", x, SR)


def engine_mix(rpm: float) -> np.ndarray:
    """The two nearest loops of the bank, crossfaded (a game does this per audio block, re-pitching each)."""
    lo = max([r for r in ENGINE_RPM if r <= rpm], default=ENGINE_RPM[0])
    hi = min([r for r in ENGINE_RPM if r >= rpm], default=ENGINE_RPM[-1])
    t = 0.0 if hi == lo else (rpm - lo) / (hi - lo)
    return engine_bank[lo] * np.cos(t * np.pi / 2) + engine_bank[hi] * np.sin(t * np.pi / 2)


write_wav(OUT / "engine_at_3200.wav", engine_mix(3200.0), SR)


# 6. State drives the sound: a machine that wears out as the simulation runs --------------------------
def machine(health: float) -> np.ndarray:
    """health 1 = new, 0 = about to fail."""
    wear = 1.0 - health
    state = "healthy" if health > 0.7 else "inner" if health > 0.3 else "dry"
    gears = sound("gearbox", rpm=1500.0, wear=round(wear, 2), backlash=round(0.2 + 0.6 * wear, 2), dur=3.0)
    bearings = sound("bearing", rpm=1500.0, state=state, severity=round(0.3 + 0.7 * wear, 2), dur=3.0)
    return 0.6 * gears + 0.5 * bearings


for health in (1.0, 0.5, 0.1):
    write_wav(OUT / f"machine_health_{int(health * 100):03d}.wav", machine(health), SR)


# 7. Effects on any array: a voice line, then the same line over a failing link -----------------------
line, _ = spec.render_spec({"layers": [{"type": "speech", "text": "door unlocked", "params": {"voice": "female"}}]}, SR)
radio = fx.apply_chain(line, [{"type": "cyber_voice", "pitch": "A2", "loss": 0.1}], SR)
write_wav(OUT / "line_clean.wav", line, SR)
write_wav(OUT / "line_radio.wav", radio, SR)


# 8. A whole scene from a dict (what the JSON files hold), built by the program -----------------------
def street(cars: int, wet: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    layers = [{"type": "sfx", "kind": "transformer", "params": {"dur": 3.0 + 2.5 * cars, "flux": 0.9}, "gain": 0.1}]
    for i in range(cars):
        layers.append({"type": "sfx", "kind": "pass_by", "at": 2.5 * i, "params": {
            "dur": 5.0, "speed": float(rng.integers(40, 130)), "wet": wet, "seed": seed + i,
            "power": str(rng.choice(["combustion", "electric"])), "direction": str(rng.choice(["lr", "rl"]))}})
    y, _ = spec.render_spec({"layers": layers, "fx": [{"type": "room", "preset": "hall", "mix": 0.1}]}, SR)
    return y


write_wav(OUT / "street_3_cars_wet.wav", street(3, wet=0.7), SR)


# 9. The other direction: what is this recording? ------------------------------------------------------
guess = identify_machine(sound("gearbox", rpm=1500.0, teeth=19, dur=5.0), SR)

if __name__ == "__main__":
    print(f"glass tap: {len(glass) / SR:.2f} s, {len(wav_bytes(glass))} bytes as WAV in memory")
    print(f"interface: {', '.join(f'{k} {len(v) / SR:.2f} s' for k, v in UI.items())}")
    print(f"footsteps: {len(footsteps)} variations, all different: "
          f"{len({x.tobytes() for x in footsteps}) == len(footsteps)}")
    print(f"engine bank: {list(engine_bank)} rpm; cache holds {sound.cache_info().currsize} sounds")
    print(f"identify: {guess['kind']} {guess['params']}")
    print(f"wrote {len(list(OUT.glob('*.wav')))} files to {OUT}/")
