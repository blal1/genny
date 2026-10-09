"""Physical propagation and room acoustics for Genny."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .core import DEFAULT_SR, resample
from .physical import space as _space

_PHYS_SR = 48000

def _at48(x, sr):
    return np.asarray(x, float) if sr == _PHYS_SR else resample(np.asarray(x, float), sr, _PHYS_SR)
def _from48(x, sr):
    return np.asarray(x, float) if sr == _PHYS_SR else resample(np.asarray(x, float), _PHYS_SR, sr)

@dataclass
class Room:
    volume: float = 100.0
    surfaces: dict[str, float] = field(default_factory=lambda: {"plaster": 80.0, "wood": 20.0})

    def t60_bands(self) -> np.ndarray:
        return _space.sabine_t60(self.volume, self.surfaces)

    def reverberate(self, x: np.ndarray, *, sr: int = DEFAULT_SR, mix: float = 0.25) -> np.ndarray:
        x48 = _at48(x, sr)
        params = _space.zita_from_bands(self.t60_bands())
        wet = _space.zita_rev1(x48, **params)
        n = min(len(x48), len(wet))
        if x48.ndim == 1 and wet.ndim == 2:
            dry = np.column_stack([x48[:n], x48[:n]])
        else:
            dry = x48[:n]
        out = (1-mix)*dry + mix*wet[:n]
        return _from48(out, sr)

@dataclass
class Propagation:
    humidity: int = 50
    reference_distance: float = 1.0

    def apply(self, x: np.ndarray, distance_m: float, *, sr: int = DEFAULT_SR,
              delay: bool = True, line_source: bool = False) -> np.ndarray:
        x48 = _at48(x, sr)
        y = _space.distance(x48, distance_m, ref=self.reference_distance, rh=self.humidity,
                            delay=delay, line_source=line_source)
        return _from48(y, sr)


def sabine_t60(volume: float, surfaces: dict[str, float]) -> np.ndarray:
    return _space.sabine_t60(volume, surfaces)


def air_absorption_db_per_m(freqs, humidity: int = 50):
    return _space.air_db_per_m(np.asarray(freqs, float), humidity)

@dataclass
class MovingPropagation(Propagation):
    """Propagation for a moving source using a smoothly varying physical delay.

    ``distance_m`` is an array sampled at the input audio rate.  Time-varying
    fractional-delay resampling produces Doppler naturally instead of applying a
    separate pitch shifter.  Distance attenuation is pressure-like (1/r), with
    optional atmospheric absorption applied in short blocks through the existing
    physical propagation kernel.
    """
    speed_of_sound: float = 343.0

    def apply_trajectory(self, x: np.ndarray, distance_m: np.ndarray, *,
                         sr: int = DEFAULT_SR, block: int = 1024) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        d = np.asarray(distance_m, dtype=float).reshape(-1)
        if d.size != len(x):
            if d.size < 2:
                d = np.full(len(x), float(d[0] if d.size else self.reference_distance))
            else:
                d = np.interp(np.linspace(0, d.size - 1, len(x)), np.arange(d.size), d)
        d = np.maximum(d, 1e-3)
        # Receive sample n reads source at n - d/c. Linear interpolation is
        # deliberately local and stable; callers needing offline mastering can
        # oversample upstream.
        delay_samples = d / float(self.speed_of_sound) * sr
        read = np.arange(len(x), dtype=float) - delay_samples
        i0 = np.floor(read).astype(np.int64)
        frac = read - i0
        valid = (i0 >= 0) & (i0 + 1 < len(x))
        y = np.zeros_like(x, dtype=float)
        y[valid] = ((1.0 - frac[valid]) * x[i0[valid]] +
                    frac[valid] * x[i0[valid] + 1])
        y *= self.reference_distance / np.maximum(d, self.reference_distance)
        # Approximate frequency-dependent air loss with the existing physical
        # kernel at the block's median distance while disabling its extra delay
        # and geometric gain (already accounted for above).
        if block and len(y):
            out = np.zeros_like(y)
            for k in range(0, len(y), int(block)):
                j = min(len(y), k + int(block))
                dm = float(np.median(d[k:j]))
                seg = y[k:j]
                wet = _space.distance(seg, dm, ref=dm, rh=self.humidity,
                                      delay=False, line_source=False)
                out[k:j] = wet[:j-k]
            y = out
        return y
