"""genny - procedural and physically informed sound synthesis."""
__version__ = "1.1.0"

# High-level physical API (existing module-level APIs remain backward compatible).
from .physics import (Material, MATERIALS, ModalBody, ParticleMaterial, BubblePopulation,
                      StringInstrument, Piano, Clarinet, Brass, VocalTract, BirdSyrinx, Environment,
                      BarBody, BowedString, Flute, Footstep, AeroacousticSwing, TunedPercussion, PluckedString, CombustionEngine, MembraneDrum, CymbalPlate, SurfaceContact, Rotor, JetEngine, Helicopter, Destruction, KickDrum, SnareDrum, AdvancedRollingContact, ElectricMotor, GearTrain, Explosion)
from .acoustics import Room, Propagation, MovingPropagation
from .ports import PhysicalPort, PassiveCoupler, PhysicalGraph, MechanicalImpedance, AcousticImpedance
from .interactions import (InteractionResult, HammerStringInteraction, BowStringInteraction,
                           ReedBoreInteraction, LipBoreInteraction, SyrinxTracheaInteraction)
from .graphsolver import (CommonPhysicalSolver, FractionalDelay, ThiranDelay, FrequencyDependentLoss,
                          StiffStringDispersion, StickSlipHysteresis, MultiportScatteringJunction,
                          ToneHole, BellRadiation, JetDelay)

__all__ = [
    "Material", "MATERIALS", "ModalBody", "ParticleMaterial", "BubblePopulation",
    "StringInstrument", "Piano", "Clarinet", "Brass", "VocalTract", "BirdSyrinx",
    "Environment", "BarBody", "BowedString", "Flute", "Footstep", "AeroacousticSwing",
    "TunedPercussion", "PluckedString", "CombustionEngine", "MembraneDrum", "CymbalPlate", "SurfaceContact", "Rotor", "JetEngine", "Helicopter", "Destruction",
    "KickDrum", "SnareDrum", "AdvancedRollingContact", "ElectricMotor", "GearTrain", "Explosion",
    "Room", "Propagation", "MovingPropagation",
    "PhysicalPort", "PassiveCoupler", "PhysicalGraph", "MechanicalImpedance", "AcousticImpedance",
    "InteractionResult", "HammerStringInteraction", "BowStringInteraction", "ReedBoreInteraction",
    "LipBoreInteraction", "SyrinxTracheaInteraction",
    "CommonPhysicalSolver", "FractionalDelay", "ThiranDelay", "FrequencyDependentLoss",
    "StiffStringDispersion", "StickSlipHysteresis", "MultiportScatteringJunction",
    "ToneHole", "BellRadiation", "JetDelay",
    "DynamicToneHole", "MultiHoleBore", "MultiHoleClarinet", "MultiHoleFlute",
    "CoupledStringBank", "AcousticCavityNetwork", "BridgeImpedance", "ModalSoundboard", "SympatheticPiano",
    "PianoHammerAction", "FrequencyDependentBridge", "PianoStringCourse", "SympatheticResonatorBank", "PolyphonicPiano",
    "StringMaterial", "STRING_MATERIALS", "StringPhysicalProperties", "WoundStringPhysicalProperties", "DispersiveString", "StringExciter", "FretboardGeometry", "HertzFretContact", "StringGesture", "StringPerformance"
]

from .graphsolver import WaveguideBranch, WaveguideNetwork
from .networks import (DynamicToneHole, MultiHoleBore, MultiHoleClarinet, MultiHoleFlute, CoupledStringBank, AcousticCavityNetwork, BridgeImpedance, ModalSoundboard, SympatheticPiano)

from .piano import (PianoHammerAction, FrequencyDependentBridge, PianoStringCourse, SympatheticResonatorBank, PolyphonicPiano)
from .strings import StringMaterial, STRING_MATERIALS, StringPhysicalProperties, WoundStringPhysicalProperties, DispersiveString, StringExciter, FretboardGeometry, HertzFretContact, StringGesture, StringPerformance

# Extension modules: each registers its instruments / drums / sfx / fx / layer types on import.
# Imported last because they build on spec, fx and the registries above. A module that is not
# present yet is skipped; any other import error is raised.
import importlib as _importlib
from . import spec as _spec  # noqa: F401
for _name in ("retro", "chiptune", "contact", "friction", "fdstring", "plates", "tubes", "spectral", "dsp",
              "reverbs", "matter", "creatures", "choir", "foley", "compose", "klang", "cyber"):
    try:
        _importlib.import_module(f"{__name__}.{_name}")
    except ModuleNotFoundError as _e:
        if _e.name != f"{__name__}.{_name}":
            raise
