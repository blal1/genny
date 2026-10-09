"""Speech synthesizer: a Klatt-style formant synth with a small text front-end (English + Spanish).

No models, no downloads: text -> phonemes (built-in dictionary, number expansion, letter names,
letter-to-sound fallback) -> durations + pitch contour -> parameter tracks -> glottal source +
aspiration + frication noise through time-varying resonators.

    render_speech("fuel low", sr, voice="female")          -> mono float buffer
    render_speech(None, sr, phonemes="F Y UW1 AH0 L")      -> raw ARPAbet
    to_phonemes("level up")                                  -> "L EH1 V AH0 L | AH1 P"
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
from scipy.signal import lfilter

from .core import DEFAULT_SR, samples
from .osc import _polyblep

# ----------------------------------------------------------------------------------------------
# Phoneme inventory
# ----------------------------------------------------------------------------------------------
# kinds
VOW, DIPH, STOP, FRIC, AFFR, NAS, LIQ, GLI, ASP, TAP, TRILL, PAUSE = (
    "vowel", "diphthong", "stop", "fricative", "affricate", "nasal", "liquid", "glide", "aspirate", "tap", "trill", "pause")
SONORANT = {VOW, DIPH, LIQ, GLI}
VOWELISH = {VOW, DIPH}

F4, F5 = 3400.0, 4200.0
FNP = 270.0  # nasal pole (cancelled by the nasal zero outside nasals)

# name: kind, formants (F1,F2,F3) [, f2 end target for diphthongs], dur (inherent, min) ms, voiced, place
PH: dict[str, dict] = {
    # English vowels (Peterson & Barney / Klatt male targets)
    "IY": dict(k=VOW, f=(270, 2290, 3010), dur=(155, 55)),
    "IH": dict(k=VOW, f=(390, 1990, 2550), dur=(135, 40)),
    "EH": dict(k=VOW, f=(530, 1840, 2480), dur=(150, 70)),
    "AE": dict(k=VOW, f=(660, 1720, 2410), dur=(230, 80)),
    "AA": dict(k=VOW, f=(730, 1090, 2440), dur=(240, 100)),
    "AO": dict(k=VOW, f=(570, 840, 2410), dur=(240, 130)),
    "UH": dict(k=VOW, f=(440, 1020, 2240), dur=(160, 60)),
    "UW": dict(k=DIPH, f=(330, 1150, 2250), f2=(300, 900, 2200), dur=(210, 70)),
    "AH": dict(k=VOW, f=(640, 1190, 2390), dur=(140, 60)),
    "AX": dict(k=VOW, f=(500, 1400, 2400), dur=(120, 55)),
    "IX": dict(k=VOW, f=(420, 1800, 2500), dur=(110, 40)),
    "ER": dict(k=VOW, f=(490, 1350, 1690), dur=(180, 80)),
    "AXR": dict(k=VOW, f=(490, 1350, 1690), dur=(120, 60)),
    "EY": dict(k=DIPH, f=(480, 1760, 2500), f2=(330, 2200, 2750), dur=(190, 100)),
    "AY": dict(k=DIPH, f=(700, 1220, 2600), f2=(400, 1950, 2600), dur=(250, 150)),
    "OY": dict(k=DIPH, f=(550, 960, 2400), f2=(380, 1850, 2500), dur=(280, 150)),
    "AW": dict(k=DIPH, f=(720, 1240, 2500), f2=(450, 1000, 2350), dur=(260, 100)),
    "OW": dict(k=DIPH, f=(560, 1000, 2400), f2=(420, 850, 2300), dur=(220, 80)),
    # Spanish pure vowels (short, monophthongal)
    "A": dict(k=VOW, f=(700, 1300, 2550), dur=(150, 70)),
    "E": dict(k=VOW, f=(460, 1850, 2550), dur=(140, 60)),
    "I": dict(k=VOW, f=(290, 2250, 2950), dur=(135, 55)),
    "O": dict(k=VOW, f=(480, 900, 2450), dur=(145, 65)),
    "U": dict(k=VOW, f=(320, 780, 2300), dur=(140, 60)),
    # stops
    "P": dict(k=STOP, v=False, place="lab", dur=(90, 50)),
    "B": dict(k=STOP, v=True, place="lab", dur=(85, 60)),
    "T": dict(k=STOP, v=False, place="alv", dur=(75, 50)),
    "D": dict(k=STOP, v=True, place="alv", dur=(75, 50)),
    "K": dict(k=STOP, v=False, place="vel", dur=(80, 60)),
    "G": dict(k=STOP, v=True, place="vel", dur=(80, 60)),
    # fricatives
    "F": dict(k=FRIC, v=False, place="labd", dur=(100, 80)),
    "V": dict(k=FRIC, v=True, place="labd", dur=(60, 40)),
    "TH": dict(k=FRIC, v=False, place="dent", dur=(90, 60)),
    "DH": dict(k=FRIC, v=True, place="dent", dur=(50, 30)),
    "S": dict(k=FRIC, v=False, place="alv", dur=(105, 60)),
    "Z": dict(k=FRIC, v=True, place="alv", dur=(75, 40)),
    "SH": dict(k=FRIC, v=False, place="post", dur=(105, 80)),
    "ZH": dict(k=FRIC, v=True, place="post", dur=(70, 40)),
    "X": dict(k=FRIC, v=False, place="vel", dur=(100, 70)),   # Spanish jota
    "HH": dict(k=ASP, v=False, place="glot", dur=(80, 30)),
    # affricates
    "CH": dict(k=AFFR, v=False, place="post", dur=(120, 80)),
    "JH": dict(k=AFFR, v=True, place="post", dur=(100, 60)),
    # nasals (fnz = nasal zero during the murmur)
    "M": dict(k=NAS, v=True, place="lab", dur=(70, 60), fnz=1000),
    "N": dict(k=NAS, v=True, place="alv", dur=(60, 50), fnz=1500),
    "NG": dict(k=NAS, v=True, place="vel", dur=(95, 60), fnz=2200),
    # liquids / glides
    "L": dict(k=LIQ, f=(330, 1050, 2800), dur=(80, 40)),
    "R": dict(k=LIQ, f=(330, 1060, 1400), dur=(80, 30)),
    "W": dict(k=GLI, f=(290, 650, 2150), dur=(80, 60)),
    "Y": dict(k=GLI, f=(260, 2070, 3020), dur=(80, 40)),
    # taps / trills (American flap, Spanish r / rr)
    "DX": dict(k=TAP, v=True, place="alv", dur=(28, 22)),
    "RR": dict(k=TRILL, v=True, place="alv", dur=(95, 75)),
}
PHONEMES = list(PH)

# consonant loci (F1, F2, F3) and how far F2/F3 lean toward the neighbouring vowel
LOCUS = {"lab": (220, 850, 2200), "labd": (250, 1100, 2300), "dent": (260, 1400, 2650),
         "alv": (240, 1700, 2650), "post": (270, 1900, 2500), "glot": (500, 1500, 2500)}
COART = {STOP: 0.45, FRIC: 0.4, AFFR: 0.3, NAS: 0.35, TAP: 0.5, TRILL: 0.4}

# frication spectra: list of (centre Hz, bandwidth Hz, gain) + flat bypass gain
FRIC_SPEC = {
    "S": ([(4800, 1500, 0.55), (6500, 2000, 1.0), (8500, 3000, 0.6)], 0.0),
    "Z": ([(4800, 1500, 0.55), (6500, 2000, 1.0), (8500, 3000, 0.6)], 0.0),
    "SH": ([(2600, 800, 1.0), (3800, 1500, 0.6), (6000, 2500, 0.35)], 0.0),
    "ZH": ([(2600, 800, 1.0), (3800, 1500, 0.6), (6000, 2500, 0.35)], 0.0),
    "CH": ([(2600, 800, 1.0), (3800, 1500, 0.6), (6000, 2500, 0.35)], 0.0),
    "JH": ([(2600, 800, 1.0), (3800, 1500, 0.6), (6000, 2500, 0.35)], 0.0),
    "F": ([(2000, 3000, 0.3), (5500, 4000, 0.5), (8500, 4000, 0.4)], 0.25),
    "V": ([(2000, 3000, 0.3), (5500, 4000, 0.5), (8500, 4000, 0.4)], 0.25),
    "TH": ([(2600, 3000, 0.3), (5500, 4000, 0.5), (8000, 4000, 0.45)], 0.3),
    "DH": ([(2600, 3000, 0.3), (5500, 4000, 0.5), (8000, 4000, 0.45)], 0.3),
}
BURST_SPEC = {
    "lab": ([(1000, 1500, 0.5), (2500, 2000, 0.35), (5000, 4000, 0.2)], 0.3),
    "alv": ([(3800, 1500, 0.6), (5000, 2000, 0.8), (7000, 3000, 0.4)], 0.1),
}
# levels in dB relative to the RMS of a stressed vowel
FRIC_DB = {"S": -14, "Z": -21, "SH": -12, "ZH": -19, "F": -25, "V": -30, "TH": -27, "DH": -31, "X": -15,
           "CH": -11, "JH": -16}
BURST_DB = {"lab": -17, "alv": -9, "vel": -9}

VOICES = {
    "male": dict(pitch=115.0, formant_shift=1.0, breath=0.08, oq=0.62, tilt=5000.0, jitter=0.004, shimmer=0.035, bw=1.0),
    "female": dict(pitch=205.0, formant_shift=1.16, breath=0.18, oq=0.70, tilt=3600.0, jitter=0.004, shimmer=0.03, bw=1.15),
    "child": dict(pitch=280.0, formant_shift=1.28, breath=0.14, oq=0.68, tilt=4200.0, jitter=0.005, shimmer=0.03, bw=1.25),
    "robot": dict(pitch=100.0, formant_shift=1.0, breath=0.0, oq=0.5, tilt=7000.0, jitter=0.0, shimmer=0.0, bw=0.9,
                  flat=True, ring=0.35),
    "whisper": dict(pitch=115.0, formant_shift=1.0, breath=0.0, oq=0.62, tilt=3200.0, jitter=0.0, shimmer=0.0, bw=1.3,
                    whisper=True),
    # --- character voices (source swaps + post "character" stage, see _character) ---
    # vocoder-style synth: saw source, pitch snapped to semitones, light chorus
    "synth": dict(pitch=120.0, formant_shift=1.0, breath=0.0, oq=0.5, tilt=9000.0, jitter=0.0, shimmer=0.0, bw=0.8,
                  source="saw", quantize=True, chorus=0.5),
    # clean android: glottal, flat, snapped, faint metallic ring and chorus
    "android": dict(pitch=135.0, formant_shift=1.05, breath=0.0, oq=0.55, tilt=6500.0, jitter=0.0, shimmer=0.0, bw=0.85,
                    flat=True, quantize=True, ring=0.12, ring_hz=90.0, chorus=0.35),
    # malfunctioning robot: bit-crushed, stutters, dropouts, pitch hiccups
    "bad_robot": dict(pitch=105.0, formant_shift=1.0, breath=0.0, oq=0.45, tilt=7000.0, jitter=0.0, shimmer=0.0, bw=0.9,
                      flat=True, ring=0.25, crush_bits=6, crush_sr=9000.0, glitch=0.35, pitch_glitch=0.3),
    # evil robot: very low, octave-down sub, slow ring, overdriven
    "evil_robot": dict(pitch=62.0, formant_shift=0.86, breath=0.0, oq=0.5, tilt=6000.0, jitter=0.0, shimmer=0.0, bw=1.0,
                       flat=True, ring=0.4, ring_hz=32.0, sub=0.6, drive=2.5),
    # monster / demon: huge tract, growl (jitter/shimmer), sub, distortion
    "monster": dict(pitch=58.0, formant_shift=0.78, breath=0.25, oq=0.66, tilt=4200.0, jitter=0.03, shimmer=0.12, bw=1.3,
                    sub=0.4, drive=3.0),
    # giant: slow big natural voice
    "giant": dict(pitch=72.0, formant_shift=0.84, breath=0.12, oq=0.64, tilt=4600.0, jitter=0.006, shimmer=0.04, bw=1.1),
    # alien: small tract, high, warbling ring modulation
    "alien": dict(pitch=240.0, formant_shift=1.4, breath=0.05, oq=0.6, tilt=5000.0, jitter=0.0, shimmer=0.0, bw=1.0,
                  ring=0.45, ring_hz=180.0, wobble=0.6),
    # ghost: breathy whisper that swims (tremolo + chorus)
    "ghost": dict(pitch=150.0, formant_shift=1.08, breath=0.0, oq=0.62, tilt=3200.0, jitter=0.0, shimmer=0.0, bw=1.35,
                  whisper=True, tremolo=(4.5, 0.5), chorus=0.8),
}

PARAMS = {
    "voice": ("male", "male|female|child|robot|whisper|synth|android|bad_robot|evil_robot|monster|giant|alien|ghost preset (sets the defaults below)"),
    "pitch": (None, "base F0 in Hz or a note name; default from voice (male 115, female 205, child 280, robot 100, evil_robot 62, monster 58, alien 240)"),
    "rate": (1.0, "speaking speed multiplier (1.3 = 30% faster, 0.8 = slower)"),
    "formant_shift": (None, "scale all formants = vocal-tract size / timbre (0.85 giant .. 1.3 child); default from voice"),
    "breath": (None, "aspiration mixed into the voicing 0..1 (breathy/soft); default from voice"),
    "vibrato": (0.0, "vibrato depth in semitones (0.3 = sung / wobbly)"),
    "intonation": ("auto", "auto|statement|question|exclaim|flat (auto follows a final ? or !)"),
    "emphasis": (1.0, "pitch-accent strength (0 = level, 2 = very animated)"),
    "lang": ("en", "en|es (Spanish: letter names, numbers, regular spelling rules)"),
    "gap": (0.0, "extra silence between words in seconds (spelling / countdown style)"),
    "seed": (0, "noise/jitter seed (tiny natural variations between takes)"),
}
SPEECH_DESC = ("Formant speech synthesizer: letters, numbers, short words/phrases (English; Spanish with lang=es). "
               "Text in /slashes/ or the layer key 'phonemes' is raw ARPAbet.")

# ----------------------------------------------------------------------------------------------
# Dictionaries
# ----------------------------------------------------------------------------------------------
LETTERS_EN = {
    "a": "EY1", "b": "B IY1", "c": "S IY1", "d": "D IY1", "e": "IY1", "f": "EH1 F", "g": "JH IY1", "h": "EY1 CH",
    "i": "AY1", "j": "JH EY1", "k": "K EY1", "l": "EH1 L", "m": "EH1 M", "n": "EH1 N", "o": "OW1", "p": "P IY1",
    "q": "K Y UW1", "r": "AA1 R", "s": "EH1 S", "t": "T IY1", "u": "Y UW1", "v": "V IY1",
    "w": "D AH1 B AH0 L Y UW0", "x": "EH1 K S", "y": "W AY1", "z": "Z IY1",
}

_DICT_EN_SRC = """
zero Z IH1 R OW0|one W AH1 N|two T UW1|three TH R IY1|four F AO1 R|five F AY1 V|six S IH1 K S|seven S EH1 V AH0 N
eight EY1 T|nine N AY1 N|ten T EH1 N|eleven IH0 L EH1 V AH0 N|twelve T W EH1 L V|thirteen TH ER1 T IY1 N
fourteen F AO1 R T IY1 N|fifteen F IH1 F T IY1 N|sixteen S IH1 K S T IY1 N|seventeen S EH1 V AH0 N T IY1 N
eighteen EY1 T IY1 N|nineteen N AY1 N T IY1 N|twenty T W EH1 N T IY0|thirty TH ER1 T IY0|forty F AO1 R T IY0
fifty F IH1 F T IY0|sixty S IH1 K S T IY0|seventy S EH1 V AH0 N T IY0|eighty EY1 T IY0|ninety N AY1 N T IY0
hundred HH AH1 N D R AH0 D|thousand TH AW1 Z AH0 N D|million M IH1 L Y AH0 N
first F ER1 S T|second S EH1 K AH0 N D|third TH ER1 D|last L AE1 S T|final F AY1 N AH0 L
a AH0|the DH AH0|and AH0 N D|of AH0 V|to T UW1|is IH1 Z|it IH1 T|in IH1 N|on AA1 N|off AO1 F|out AW1 T|up AH1 P
down D AW1 N|you Y UW1|your Y AO1 R|my M AY1|me M IY1|we W IY1|all AO1 L|more M AO1 R|now N AW1|yes Y EH1 S|no N OW1
ok OW2 K EY1|okay OW2 K EY1|not N AA1 T|get G EH1 T|got G AA1 T|go G OW1|let's L EH1 T S|don't D OW1 N T|x EH1 K S
ready R EH1 D IY0|set S EH1 T|steady S T EH1 D IY0|start S T AA1 R T|stop S T AA1 P|pause P AO1 Z|resume R IH0 Z UW1 M
play P L EY1|player P L EY1 ER0|quit K W IH1 T|exit EH1 G Z IH0 T|menu M EH1 N Y UW0|select S AH0 L EH1 K T
back B AE1 K|next N EH1 K S T|continue K AH0 N T IH1 N Y UW0|new N UW1|again AH0 G EH1 N|try T R AY1
level L EH1 V AH0 L|stage S T EY1 JH|round R AW1 N D|wave W EY1 V|world W ER1 L D|mode M OW1 D|mission M IH1 SH AH0 N
complete K AH0 M P L IY1 T|clear K L IH1 R|cleared K L IH1 R D|done D AH1 N|finish F IH1 N IH0 SH|finished F IH1 N IH0 SH T
game G EY1 M|over OW1 V ER0|win W IH1 N|winner W IH1 N ER0|won W AH1 N|lose L UW1 Z|lost L AO1 S T|fail F EY1 L
failed F EY1 L D|victory V IH1 K T ER0 IY0|defeat D IH0 F IY1 T|dead D EH1 D|die D AY1|kill K IH1 L|killed K IH1 L D
time T AY1 M|times T AY1 M Z|timer T AY1 M ER0|hurry HH ER1 IY0|quick K W IH1 K|fast F AE1 S T|faster F AE1 S T ER0
slow S L OW1|wait W EY1 T|countdown K AW1 N T D AW2 N
score S K AO1 R|high HH AY1|low L OW1|best B EH1 S T|record R EH1 K ER0 D|point P OY1 N T|points P OY1 N T S
bonus B OW1 N AH0 S|extra EH1 K S T R AH0|combo K AA1 M B OW0|perfect P ER1 F IH0 K T|great G R EY1 T|good G UH1 D
nice N AY1 S|excellent EH1 K S AH0 L AH0 N T|awesome AO1 S AH0 M|amazing AH0 M EY1 Z IH0 NG|super S UW1 P ER0
double D AH1 B AH0 L|triple T R IH1 P AH0 L|streak S T R IY1 K|critical K R IH1 T IH0 K AH0 L|max M AE1 K S
life L AY1 F|lives L AY1 V Z|health HH EH1 L TH|heal HH IY1 L|shield SH IY1 L D|shields SH IY1 L D Z|armor AA1 R M ER0
power P AW1 ER0|energy EH1 N ER0 JH IY0|battery B AE1 T ER0 IY0|fuel F Y UW1 AH0 L|gas G AE1 S|oxygen AA1 K S AH0 JH AH0 N
empty EH1 M P T IY0|full F UH1 L|ammo AE1 M OW0|reload R IY0 L OW1 D|repair R IH0 P EH1 R|damage D AE1 M AH0 JH
bomb B AA1 M|bombs B AA1 M Z|magnet M AE1 G N AH0 T|turbo T ER1 B OW0|boost B UW1 S T|nitro N AY1 T R OW0
speed S P IY1 D|jump JH AH1 M P|duck D AH1 K|dodge D AA1 JH|block B L AA1 K|fire F AY1 ER0|shoot SH UW1 T|hit HH IH1 T
miss M IH1 S|missed M IH1 S T|attack AH0 T AE1 K|fight F AY1 T|boss B AO1 S|enemy EH1 N AH0 M IY0
enemies EH1 N AH0 M IY0 Z|target T AA1 R G AH0 T|incoming IH1 N K AH2 M IH0 NG|missile M IH1 S AH0 L|rocket R AA1 K AH0 T
laser L EY1 Z ER0|bullet B UH1 L AH0 T|ship SH IH1 P|boom B UW1 M|crash K R AE1 SH|drift D R IH1 F T|lap L AE1 P
laps L AE1 P S|checkpoint CH EH1 K P OY2 N T|race R EY1 S|lead L IY1 D|place P L EY1 S|position P AH0 Z IH1 SH AH0 N
engine EH1 N JH AH0 N|overheat OW1 V ER0 HH IY2 T|pit P IH1 T|turn T ER1 N|danger D EY1 N JH ER0|warning W AO1 R N IH0 NG
caution K AO1 SH AH0 N|alert AH0 L ER1 T|careful K EH1 R F AH0 L|watch W AA1 CH|look L UH1 K|safe S EY1 F|zone Z OW1 N
wall W AO1 L|hole HH OW1 L|door D AO1 R|key K IY1|exit EH1 G Z IH0 T|left L EH1 F T|right R AY1 T|center S EH1 N T ER0
front F R AH1 N T|ahead AH0 HH EH1 D|behind B IH0 HH AY1 N D|above AH0 B AH1 V|below B IH0 L OW1|north N AO1 R TH
south S AW1 TH|east IY1 S T|west W EH1 S T|move M UW1 V|run R AH1 N|hold HH OW1 L D
coin K OY1 N|coins K OY1 N Z|cash K AE1 SH|money M AH1 N IY0|gold G OW1 L D|silver S IH1 L V ER0|bronze B R AA1 N Z
bank B AE1 NG K|bankrupt B AE1 NG K R AH0 P T|bet B EH1 T|spin S P IH1 N|wheel W IY1 L|jackpot JH AE1 K P AA2 T
lucky L AH1 K IY0|free F R IY1|letter L EH1 T ER0|vowel V AW1 AH0 L|buy B AY1|solve S AA1 L V|puzzle P AH1 Z AH0 L
word W ER1 D|guess G EH1 S|correct K ER0 EH1 K T|wrong R AO1 NG|red R EH1 D|blue B L UW1|green G R IY1 N
yellow Y EH1 L OW0|white W AY1 T|black B L AE1 K|easy IY1 Z IY0|normal N AO1 R M AH0 L|hard HH AA1 R D
hello HH AH0 L OW1|welcome W EH1 L K AH0 M|goodbye G UH2 D B AY1|thanks TH AE1 NG K S|please P L IY1 Z
"""
DICT_EN: dict[str, str] = {}
for _line in _DICT_EN_SRC.strip().splitlines():
    for _entry in _line.split("|"):
        _w, _, _p = _entry.strip().partition(" ")
        if _w:
            DICT_EN[_w] = _p.strip()
WORDS = sorted(DICT_EN)

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
         "seventeen eighteen nineteen").split()
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_words_en(n: int) -> list[str]:
    if n < 20:
        return [_ONES[n]]
    if n < 100:
        return [_TENS[n // 10]] + ([_ONES[n % 10]] if n % 10 else [])
    if n < 1000:
        return [_ONES[n // 100], "hundred"] + (number_words_en(n % 100) if n % 100 else [])
    if n < 1_000_000:
        return number_words_en(n // 1000) + ["thousand"] + (number_words_en(n % 1000) if n % 1000 else [])
    if n < 1_000_000_000:
        return number_words_en(n // 1_000_000) + ["million"] + (number_words_en(n % 1_000_000) if n % 1_000_000 else [])
    return [_ONES[int(c)] for c in str(n)]


LETTERS_ES = {"a": "a", "b": "be", "c": "ce", "d": "de", "e": "e", "f": "efe", "g": "ge", "h": "hache", "i": "i",
              "j": "jota", "k": "ka", "l": "ele", "m": "eme", "n": "ene", "ñ": "eñe", "o": "o", "p": "pe", "q": "cu",
              "r": "erre", "s": "ese", "t": "te", "u": "u", "v": "uve", "w": "uve doble", "x": "equis",
              "y": "i griega", "z": "zeta"}
_ONES_ES = ("cero uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce quince dieciséis "
            "diecisiete dieciocho diecinueve veinte veintiuno veintidós veintitrés veinticuatro veinticinco "
            "veintiséis veintisiete veintiocho veintinueve").split()
_TENS_ES = ["", "", "", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta", "noventa"]
_HUND_ES = ["", "ciento", "doscientos", "trescientos", "cuatrocientos", "quinientos", "seiscientos", "setecientos",
            "ochocientos", "novecientos"]


def number_words_es(n: int) -> list[str]:
    if n < 30:
        return [_ONES_ES[n]]
    if n < 100:
        return [_TENS_ES[n // 10]] + (["y", _ONES_ES[n % 10]] if n % 10 else [])
    if n == 100:
        return ["cien"]
    if n < 1000:
        return [_HUND_ES[n // 100]] + (number_words_es(n % 100) if n % 100 else [])
    if n < 1_000_000:
        head = [] if n // 1000 == 1 else number_words_es(n // 1000)
        return head + ["mil"] + (number_words_es(n % 1000) if n % 1000 else [])
    return [_ONES_ES[int(c)] for c in str(n)]


# ----------------------------------------------------------------------------------------------
# Letter-to-sound fallbacks
# ----------------------------------------------------------------------------------------------
_V = set("aeiou")


def _lts_en(word: str) -> list[tuple[str, int]]:
    """Small English letter-to-sound rule set for words missing from the dictionary. Approximate."""
    w = re.sub(r"[^a-z]", "", word.lower())
    n = len(w)
    out: list[str] = []
    stress_mark = None  # index into out of the vowel to stress (e.g. before -tion)
    i = 0

    def at(j):
        return w[j] if 0 <= j < n else ""

    def magic_e(j):  # vowel at j + single consonant + final e
        return j + 2 == n - 1 and at(j + 2) == "e" and at(j + 1) and at(j + 1) not in _V | {"w", "x", "y"}

    while i < n:
        c, rest = w[i], w[i:]
        if rest.startswith(("tion", "sion")):
            stress_mark = len([p for p in out if p in PH and PH[p]["k"] in VOWELISH]) - 1
            out += ["SH" if c == "t" else "ZH", "AH", "N"]; i += 4; continue
        if rest.startswith("ture"):
            out += ["CH", "ER"]; i += 4; continue
        for pat, ph in (("augh", ["AO"]), ("eigh", ["EY"]), ("ough", ["AO"]), ("igh", ["AY"]), ("tch", ["CH"]),
                        ("dge", ["JH"]), ("sch", ["S", "K"]), ("ck", ["K"]), ("ch", ["CH"]), ("sh", ["SH"]),
                        ("th", ["TH"]), ("ph", ["F"]), ("wh", ["W"]), ("qu", ["K", "W"]), ("nk", ["NG", "K"]),
                        ("ng", ["NG"]), ("ee", ["IY"]), ("ea", ["IY"]), ("oo", ["UW"]), ("ou", ["AW"]),
                        ("oi", ["OY"]), ("oy", ["OY"]), ("ai", ["EY"]), ("ay", ["EY"]), ("oa", ["OW"]),
                        ("au", ["AO"]), ("aw", ["AO"]), ("ew", ["UW"]), ("ie", ["IY"]), ("ue", ["UW"]),
                        ("ar", ["AA", "R"]), ("or", ["AO", "R"]), ("er", ["ER"]), ("ir", ["ER"]), ("ur", ["ER"])):
            if rest.startswith(pat):
                if pat in ("ar", "or", "er", "ir", "ur") and at(i + 2) in _V:
                    continue  # r starts the next syllable: handled as vowel + r
                out += ph; i += len(pat); break
        else:
            if i == 0 and rest.startswith(("kn", "gn", "wr")):
                i += 1; continue
            if rest.startswith("ow"):
                out += ["OW" if i + 2 == n else "AW"]; i += 2; continue
            if rest.startswith("ey") and i + 2 == n:
                out += ["IY"]; i += 2; continue
            if c in _V:
                if c == "e" and i == n - 1 and n > 2 and any(ch in _V | {"y"} for ch in w[:i]):
                    i += 1; continue  # silent final e
                if magic_e(i):
                    out += [{"a": "EY", "e": "IY", "i": "AY", "o": "OW", "u": "UW"}[c]]
                elif c == "o" and i == n - 1:
                    out += ["OW"]
                elif c == "e" and n <= 2 and i == n - 1:
                    out += ["IY"]
                elif c == "a" and i == n - 1:
                    out += ["AH"]
                else:
                    out += [{"a": "AE", "e": "EH", "i": "IH", "o": "AA", "u": "AH"}[c]]
            elif c == "y":
                if i == 0:
                    out += ["Y"]
                elif i == n - 1:
                    nv = sum(ch in _V for ch in w)
                    out += ["AY" if nv == 0 else "IY"]
                else:
                    out += ["IH"]
            elif c == "c":
                out += ["S"] if at(i + 1) in ("e", "i", "y") else ["K"]
            elif c == "g":
                out += ["JH"] if at(i + 1) in ("e", "i", "y") and i + 1 < n - 1 else ["G"]
            elif c == "x":
                out += ["K", "S"]
            elif c == "s":
                out += ["Z"] if (0 < i < n - 1 and at(i - 1) in _V and at(i + 1) in _V) or (
                    i == n - 1 and n > 2 and at(i - 1) in "bdglmnrv") else ["S"]
            elif c == "j":
                out += ["JH"]
            elif c == "q":
                out += ["K"]
            else:
                m = {"b": "B", "d": "D", "f": "F", "h": "HH", "k": "K", "l": "L", "m": "M", "n": "N", "p": "P",
                     "r": "R", "t": "T", "v": "V", "w": "W", "z": "Z"}.get(c)
                if m and not (out and out[-1] == m and at(i - 1) == c):  # collapse doubled letters
                    out.append(m)
            i += 1
    if not out:
        return []
    # stress: first vowel, or the one before -tion
    vidx = [j for j, p in enumerate(out) if PH[p]["k"] in VOWELISH]
    target = vidx[stress_mark] if stress_mark is not None and 0 <= stress_mark < len(vidx) else (vidx[0] if vidx else -1)
    res = []
    for j, p in enumerate(out):
        if PH[p]["k"] in VOWELISH:
            if j == target:
                res.append((p, 1))
            else:
                res.append(("AH" if p in ("AE", "EH", "AA", "AH", "UH") else p, 0))
        else:
            res.append((p, 0))
    return res


_ES_ACC = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u"}
_ES_V = set("aeiouáéíóúü")


def _g2p_es(word: str) -> list[tuple[str, int]]:
    """Spanish (Castilian) spelling -> phonemes. Regular orthography makes this quite reliable."""
    w = word.lower()
    n = len(w)
    items: list[list] = []  # [phone, is_vowel, accented]
    i = 0
    while i < n:
        c = w[i]
        nx = w[i + 1] if i + 1 < n else ""
        if c in _ES_V:
            base = _ES_ACC.get(c, c)
            items.append([base.upper(), True, c in "áéíóú"])
        elif c == "c" and nx == "h":
            items.append(["CH", False, False]); i += 1
        elif c == "l" and nx == "l":
            items.append(["Y", False, False]); i += 1
        elif c == "r" and nx == "r":
            items.append(["RR", False, False]); i += 1
        elif c == "r":
            items.append(["RR" if i == 0 or w[i - 1] in "nls" else "DX", False, False])
        elif c == "q" and nx == "u":
            items.append(["K", False, False]); i += 1
        elif c == "g" and nx == "u" and i + 2 < n and w[i + 2] in "eéií":
            items.append(["G", False, False]); i += 1
        elif c == "g" and nx == "ü":
            items.append(["G", False, False]); items.append(["W", False, False]); i += 1
        elif c == "g":
            items.append(["X" if nx and nx in "eéií" else "G", False, False])
        elif c == "c":
            items.append(["TH" if nx and nx in "eéií" else "K", False, False])
        elif c == "z":
            items.append(["TH", False, False])
        elif c == "j":
            items.append(["X", False, False])
        elif c == "h":
            pass
        elif c in "bv":
            items.append(["B", False, False])
        elif c == "ñ":
            items.append(["N", False, False]); items.append(["Y", False, False])
        elif c == "x":
            items.append(["K", False, False]); items.append(["S", False, False])
        elif c == "y":
            if n == 1 or (i == n - 1) or not (nx and nx in _ES_V):
                items.append(["I", True, False])
            else:
                items.append(["Y", False, False])
        elif c == "w":
            items.append(["W", False, False])
        elif c == "k":
            items.append(["K", False, False])
        else:
            m = {"d": "D", "f": "F", "l": "L", "m": "M", "n": "N", "p": "P", "s": "S", "t": "T"}.get(c)
            if m:
                items.append([m, False, False])
        i += 1
    # glides: unaccented i/u next to another vowel
    for j, it in enumerate(items):
        if it[1] and it[0] in ("I", "U") and not it[2]:
            nxt_v = j + 1 < len(items) and items[j + 1][1]
            prv_v = j > 0 and items[j - 1][1]
            if nxt_v or prv_v:
                if sum(1 for x in items if x[1]) > 1:
                    it[0] = "Y" if it[0] == "I" else "W"
                    it[1] = False
    nuclei = [j for j, it in enumerate(items) if it[1]]
    stressed = None
    for j in nuclei:
        if items[j][2]:
            stressed = j
    if stressed is None and nuclei:
        last = re.sub(r"[^a-zñ]", "", "".join(_ES_ACC.get(ch, ch) for ch in w))[-1:]
        if len(nuclei) >= 2 and (last in "aeiouns" or last == ""):
            stressed = nuclei[-2]
        else:
            stressed = nuclei[-1]
    return [(it[0], (1 if j == stressed else 0) if it[1] else 0) for j, it in enumerate(items)]


# ----------------------------------------------------------------------------------------------
# Text front-end
# ----------------------------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"/[^/]*/|\d+|[A-Za-zÀ-ÿñÑ']+|[,.;:!?]")


def _parse_raw(s: str) -> list[dict]:
    """Raw ARPAbet: 'JH IY1', words split by '|', pauses '_' (short) or ',' ."""
    words, cur = [], []
    for tok in s.replace(",", " , ").replace("|", " | ").split():
        if tok in ("|", ",", "_"):
            if cur:
                words.append({"ph": cur, "letter": False})
                cur = []
            if tok != "|":
                words.append({"pause": 0.22 if tok == "," else 0.12})
            continue
        m = re.match(r"^([A-Za-z]+)([012])?$", tok)
        if not m or m.group(1).upper() not in PH:
            raise ValueError(f"unknown phoneme {tok!r}; known: {' '.join(PHONEMES)}")
        p = m.group(1).upper()
        st = int(m.group(2)) if m.group(2) is not None else (1 if PH[p]["k"] in VOWELISH else 0)
        cur.append((p, st))
    if cur:
        words.append({"ph": cur, "letter": False})
    return words


def _parse_ph(s: str) -> list[tuple[str, int]]:
    return [(re.sub(r"\d", "", t), int(t[-1]) if t[-1].isdigit() else 0) for t in s.split()]


def _word_en(tok: str, single: bool) -> list[dict]:
    low = tok.lower()
    if len(low) == 1 and low.isalpha():
        if low == "a" and tok == "a" and not single:
            return [{"ph": [("AH", 0)], "letter": False}]
        return [{"ph": _parse_ph(LETTERS_EN[low]), "letter": True}]
    if low in DICT_EN:
        return [{"ph": _parse_ph(DICT_EN[low]), "letter": False}]
    if low.endswith("'s") and low[:-2] in DICT_EN:
        return [{"ph": _parse_ph(DICT_EN[low[:-2]]) + [("Z", 0)], "letter": False}]
    if tok.isupper() and 2 <= len(tok) <= 5:
        return [{"ph": _parse_ph(LETTERS_EN[ch]), "letter": True} for ch in low if ch in LETTERS_EN]
    # simple plural of a known word
    if low.endswith("s") and low[:-1] in DICT_EN:
        base = _parse_ph(DICT_EN[low[:-1]])
        last = base[-1][0]
        suf = [("S", 0)] if last in ("P", "T", "K", "F", "TH") else [("Z", 0)]
        return [{"ph": base + suf, "letter": False}]
    ph = _lts_en(low)
    return [{"ph": ph, "letter": False}] if ph else []


def _word_es(tok: str, single: bool) -> list[dict]:
    low = tok.lower()
    if len(low) == 1 and low in LETTERS_ES and (single or tok.isupper()):
        return [{"ph": _g2p_es(x), "letter": True} for x in LETTERS_ES[low].split()]
    if tok.isupper() and 2 <= len(tok) <= 5:
        out = []
        for ch in low:
            out += [{"ph": _g2p_es(x), "letter": True} for x in LETTERS_ES.get(ch, ch).split()]
        return out
    ph = _g2p_es(low)
    if low == "y":
        ph = [("I", 0)]
    return [{"ph": ph, "letter": False}] if ph else []


def text_to_phrases(text: str, lang: str = "en") -> list[dict]:
    """Returns [{"words": [{"ph": [(ph, stress)], "letter": bool} | {"pause": s}], "punct": "."}, ...]."""
    if lang not in ("en", "es"):
        raise ValueError(f"lang must be 'en' or 'es', got {lang!r}")
    tokens = _TOKEN_RE.findall(text.replace("-", " "))
    words_only = [t for t in tokens if t not in ",.;:!?"]
    single = len(words_only) == 1
    phrases, cur = [], []
    for tok in tokens:
        if tok in ",.;:!?":
            if cur:
                phrases.append({"words": cur, "punct": tok})
                cur = []
            continue
        if tok.startswith("/"):
            cur += _parse_raw(tok.strip("/"))
        elif tok.isdigit():
            nums = number_words_en(int(tok)) if lang == "en" else number_words_es(int(tok))
            for wd in nums:
                cur += _word_en(wd, False) if lang == "en" else _word_es(wd, False)
        else:
            cur += _word_en(tok, single) if lang == "en" else _word_es(tok, single)
    if cur:
        phrases.append({"words": cur, "punct": ""})
    return phrases


def to_phonemes(text: str | None = None, lang: str = "en", phonemes: str | None = None) -> str:
    phrases = [{"words": _parse_raw(phonemes), "punct": ""}] if phonemes else text_to_phrases(text or "", lang)
    parts = []
    for ph in phrases:
        ws = []
        for w in ph["words"]:
            if "pause" in w:
                ws.append("_")
            else:
                ws.append(" ".join(p + (str(s) if PH[p]["k"] in VOWELISH else "") for p, s in w["ph"]))
        parts.append(" | ".join(ws) + (" " + ph["punct"] if ph["punct"] else ""))
    return "  ".join(parts)


# ----------------------------------------------------------------------------------------------
# Segments, durations
# ----------------------------------------------------------------------------------------------
def _build_segments(phrases, rate: float, gap: float, lang: str = "en") -> list[dict]:
    segs: list[dict] = []
    for pi, phr in enumerate(phrases):
        words = phr["words"]
        start_idx = len(segs)
        real = [w for w in words if "ph" in w and w["ph"]]
        for wi, w in enumerate(words):
            if "pause" in w:
                segs.append({"ph": "_", "dur": w["pause"]})
                continue
            if not w["ph"]:
                continue
            for k, (p, st) in enumerate(w["ph"]):
                segs.append({"ph": p, "st": st, "phrase": pi, "word": (pi, wi), "letter": w["letter"], "lang": lang,
                             "wstart": k == 0, "wend": k == len(w["ph"]) - 1})
            is_last = w is real[-1] if real else True
            if not is_last:
                g = gap + (0.07 if w["letter"] else 0.0)
                if g > 0:
                    segs.append({"ph": "_", "dur": g})
        # phrase-final syllable marks
        idx = [j for j in range(start_idx, len(segs)) if segs[j]["ph"] != "_"]
        vows = [j for j in idx if PH[segs[j]["ph"]]["k"] in VOWELISH]
        if vows:
            for j in idx:
                if j >= vows[-1]:
                    segs[j]["final"] = True
        # spelled letters: each is its own little phrase-final syllable
        for j in idx:
            if segs[j].get("letter") and PH[segs[j]["ph"]]["k"] in VOWELISH and segs[j]["st"] > 0:
                segs[j]["final"] = True
        if pi < len(phrases) - 1:
            segs.append({"ph": "_", "dur": 0.16 if phr["punct"] == "," else 0.28})
    _durations(segs, rate)
    return segs


def _durations(segs, rate):
    stretch = 1.12 / max(rate, 0.1)
    n = len(segs)
    # citation form: a lone word / letter is said more slowly than running speech
    words = {sg.get("word") for sg in segs if sg["ph"] != "_"}
    citation = 1.25 if len(words) == 1 else 1.0
    for i, sg in enumerate(segs):
        if sg["ph"] == "_":
            sg["dur"] = sg["dur"] / max(rate, 0.1)
            continue
        info = PH[sg["ph"]]
        inh, mn = info["dur"]
        k = info["k"]
        pct = 1.0
        nxt = segs[i + 1] if i + 1 < n and segs[i + 1]["ph"] != "_" else None
        prv = segs[i - 1] if i > 0 and segs[i - 1]["ph"] != "_" else None
        same_word_next = nxt is not None and nxt.get("word") == sg.get("word")
        if k in VOWELISH:
            if sg["st"] == 0:
                pct *= 0.5
            elif sg["st"] == 2:
                pct *= 0.8
            if sg.get("final"):
                pct *= 1.9
            if same_word_next:
                nk = PH[nxt["ph"]]
                if nk["k"] in (STOP, AFFR, FRIC) and not nk.get("v", True):
                    pct *= 0.82
                elif sg.get("final") and nk["k"] not in SONORANT:
                    pct *= 1.12
        else:
            cons_next = nxt is not None and PH[nxt["ph"]]["k"] not in VOWELISH and same_word_next
            cons_prev = prv is not None and PH[prv["ph"]]["k"] not in VOWELISH and prv.get("word") == sg.get("word")
            if cons_next or cons_prev:
                pct *= 0.8
            if sg.get("final"):
                pct *= 1.25
        sg["dur"] = (mn + (inh - mn) * pct) / 1000.0 * stretch * (citation if k in VOWELISH else 1.0)
    # stop release details / VOT
    for i, sg in enumerate(segs):
        if sg["ph"] == "_" or PH[sg["ph"]]["k"] != STOP:
            continue
        info = PH[sg["ph"]]
        nxt = segs[i + 1] if i + 1 < n and segs[i + 1]["ph"] != "_" else None
        prv = segs[i - 1] if i > 0 and segs[i - 1]["ph"] != "_" else None
        sg["burst"] = {"lab": 0.006, "alv": 0.009, "vel": 0.015}[info["place"]]
        sg["post"] = 0.0
        sg["burst_amp"] = 1.0
        if nxt is not None and PH[nxt["ph"]]["k"] in SONORANT | {TAP, TRILL}:
            if info["v"]:
                vot = 0.008 if sg.get("lang") != "es" else 0.004
                asp = 0.0
            elif sg.get("lang") == "es" or (prv is not None and prv["ph"] == "S" and prv.get("word") == sg.get("word")):
                vot, asp = 0.014, 0.5
            elif sg.get("wstart") or nxt.get("st", 0) >= 1:
                vot, asp = 0.058, 1.0
            else:
                vot, asp = 0.03, 0.8
            nxt["head"] = vot / max(rate, 0.1) ** 0.5
            nxt["head_asp"] = asp
            nxt["dur"] += nxt["head"] * 0.7
        elif nxt is None:
            sg["post"] = (0.04 if not info["v"] else 0.012) / max(rate, 0.1) ** 0.5
            sg["burst_amp"] = 0.7
            sg["dur"] += sg["post"]
        else:
            sg["burst_amp"] = 0.45
    t = 0.012
    for sg in segs:
        sg["s"] = t
        t += sg["dur"]
        sg["e"] = t


# ----------------------------------------------------------------------------------------------
# Parameter tracks
# ----------------------------------------------------------------------------------------------
def _nearest_vowel(segs, i):
    n = len(segs)
    for j in range(i + 1, min(n, i + 4)):
        if segs[j]["ph"] == "_":
            break
        if PH[segs[j]["ph"]]["k"] in VOWELISH:
            return _vowel_target(segs[j]["ph"])[0]
    for j in range(i - 1, max(-1, i - 4), -1):
        if segs[j]["ph"] == "_":
            break
        if PH[segs[j]["ph"]]["k"] in VOWELISH:
            return _vowel_target(segs[j]["ph"])[1]
    return None


def _bw(f1, f2, kind):
    if kind in VOWELISH:
        b1 = 55 + max(0.0, f1 - 300) * 0.07
        return [b1, 85 + max(0.0, f2 - 1000) * 0.02, 140, 250, 320]
    if kind == NAS:
        return [100, 300, 300, 350, 400]
    if kind in (LIQ, GLI):
        return [80, 120, 150, 250, 320]
    return [120, 150, 200, 300, 400]


def _vowel_target(p):
    info = PH[p]
    a = list(info["f"]) + [F4, F5]
    z = list(info.get("f2", info["f"])) + [F4, F5]
    return a, z


def _targets(segs):
    out = []
    for i, sg in enumerate(segs):
        p = sg["ph"]
        if p == "_":
            out.append(None)
            continue
        info = PH[p]
        k = info["k"]
        if k in VOWELISH:
            a, z = _vowel_target(p)
            out.append((a + _bw(a[0], a[1], k), z + _bw(z[0], z[1], k)))
            continue
        if k in (LIQ, GLI):
            f = list(info["f"]) + [F4, F5]
            if p == "L" and sg.get("wend") and not sg.get("wstart"):
                f[1] = 900  # dark final l
            out.append((f + _bw(*f[:2], k),) * 2)
            continue
        vt = _nearest_vowel(segs, i)
        if k == ASP:
            f = list(vt) if vt else [500, 1500, 2500, F4, F5]
            b = [250, 200, 250, 300, 400]
            out.append((f + b,) * 2)
            continue
        place = info["place"]
        if place == "vel":
            vf2 = vt[1] if vt else 1500
            f2 = float(np.clip(vf2 + 150, 1350, 2400))
            f = [250, f2, max(f2 + 450, 2200), F4, F5]
        else:
            loc = LOCUS[place]
            f = [loc[0], loc[1], loc[2], F4, F5]
            if vt is not None:
                kk = COART.get(k, 0.4)
                f[1] = loc[1] + kk * (vt[1] - loc[1])
                f[2] = loc[2] + kk * 0.6 * (vt[2] - loc[2])
        if k == NAS:
            f[0] = 280
        if k in (TAP, TRILL):
            f[0] = 300
            if vt is not None:
                f[1] = 1700 + 0.5 * (vt[1] - 1700)
        out.append((f + _bw(f[0], f[1], k),) * 2)
    return out


def _trans(ka, kb):
    """(transition seconds, fraction lying in the left segment)."""
    obs = {STOP: 0.05, AFFR: 0.04, FRIC: 0.045, ASP: 0.02, NAS: 0.035, TAP: 0.03, TRILL: 0.035}
    if ka in obs and kb in obs:
        return 0.015, 0.5
    if ka in obs:
        return obs[ka], 0.05
    if kb in obs:
        return obs[kb] + 0.005, 0.95
    if ka in (LIQ, GLI) or kb in (LIQ, GLI):
        return 0.075, 0.5
    return 0.1, 0.5


class _Tracks:
    def __init__(self):
        self.p = defaultdict(list)

    def add(self, name, t, v):
        self.p[name].append((float(t), v))

    def curve(self, name, times, default):
        pts = self.p.get(name)
        if not pts:
            return np.full(times.shape[0], float(default))
        pts = sorted(pts, key=lambda q: q[0])
        xs = np.array([q[0] for q in pts])
        ys = np.array([q[1] for q in pts], dtype=float)
        for j in range(1, xs.shape[0]):  # strictly increasing
            if xs[j] <= xs[j - 1]:
                xs[j] = xs[j - 1] + 1e-6
        return np.interp(times, xs, ys)


def _build_tracks(segs, vs, lang):
    T = _Tracks()
    n = len(segs)
    tgt = _targets(segs)
    fshift = vs["formant_shift"]
    # --- formant tracks with coarticulated transitions
    kinds = [None if s["ph"] == "_" else PH[s["ph"]]["k"] for s in segs]
    tin = [0.0] * n
    tout = [0.0] * n
    for i in range(n - 1):
        if kinds[i] is None or kinds[i + 1] is None:
            continue
        tt, fl = _trans(kinds[i], kinds[i + 1])
        tout[i] = tt * fl
        tin[i + 1] = tt * (1 - fl)
    for i, sg in enumerate(segs):
        if kinds[i] is None:
            continue
        d = sg["e"] - sg["s"]
        if sg.get("head"):
            tin[i] = max(tin[i], sg["head"] + 0.015)
        tot = tin[i] + tout[i]
        if tot > 0.85 * d:
            sc = 0.85 * d / tot
            tin[i] *= sc
            tout[i] *= sc
        a, z = tgt[i]
        t1, t2 = sg["s"] + tin[i], sg["e"] - tout[i]
        pts = [(t1, a)]
        if kinds[i] == DIPH:
            pts.append((t1 + 0.3 * (t2 - t1), a))
        pts.append((t2, z))
        if i == 0 or kinds[i - 1] is None:
            pts.insert(0, (sg["s"], a))
        if i == n - 1 or kinds[i + 1] is None:
            pts.append((sg["e"], z))
        for t, vec in pts:
            for j in range(10):
                T.add(f"FM{j}", t, vec[j] * (fshift if j < 5 else 1.0))

    # --- amplitude / source tracks
    breath = vs["breath"]
    for i, sg in enumerate(segs):
        s, e = sg["s"], sg["e"]
        d = e - s
        p = sg["ph"]
        prv_pause = i == 0 or kinds[i - 1] is None
        nxt_pause = i == n - 1 or kinds[i + 1] is None
        if p == "_":
            r = min(0.005, d / 3)
            for nm in ("AV", "AH", "AF", "AVB", "B1ADD"):
                T.add(nm, s + r, 0.0)
                T.add(nm, e - r, 0.0)
            continue
        info = PH[p]
        k = info["k"]
        r = min(0.006, d / 4)
        fnz = info.get("fnz", FNP) * fshift if k == NAS else FNP * fshift
        rz = min(0.008, d / 3)
        T.add("FNZ", s + rz, fnz)
        T.add("FNZ", e - rz, fnz)
        if k in SONORANT or k in (NAS, TAP, TRILL):
            if k in VOWELISH:
                av = 1.0 if sg["st"] > 0 else 0.78
            else:
                av = {NAS: 0.8, LIQ: 0.85, GLI: 0.88, TAP: 0.85, TRILL: 0.85}[k]
            head = min(sg.get("head", 0.0), 0.6 * d)
            if head > 0:
                asp = sg.get("head_asp", 0.0)
                T.add("AV", s, 0.0)
                T.add("AV", s + head, 0.0)
                T.add("AV", s + head + 0.007, av)
                T.add("AH", s, asp)
                T.add("AH", s + head, asp * 0.6)
                T.add("AH", s + head + 0.012, 0.0)
                T.add("B1ADD", s, 220.0)
                T.add("B1ADD", s + head, 150.0)
                T.add("B1ADD", s + head + 0.02, 0.0)
            else:
                on = min(0.03, d / 3) if prv_pause else r
                if prv_pause:
                    T.add("AV", s, 0.0)
                T.add("AV", s + on, av)
                T.add("AH", s + on, 0.0)
                T.add("B1ADD", s + on, 0.0)
            if nxt_pause:
                off = min(0.09, d * 0.45)
                T.add("AV", e - off, av)
                T.add("AV", e - off * 0.35, av * 0.35)
                T.add("AV", e, 0.0)
                T.add("AH", e - off, 0.0)
                T.add("AH", e - off * 0.3, 0.05 + breath * 0.3)
                T.add("AH", e, 0.0)
                T.add("B1ADD", e - off, 0.0)
                T.add("B1ADD", e, 150.0)
            else:
                T.add("AV", e - r, av)
                T.add("AH", e - r, 0.0)
                T.add("B1ADD", e - r, 0.0)
            for nm in ("AF", "AVB"):
                T.add(nm, s + r, 0.0)
                T.add(nm, e - r, 0.0)
            if k == TAP:
                T.add("AV", s + d * 0.5, av * 0.25)
            if k == TRILL:
                for j in range(1, 6):
                    T.add("AV", s + d * j / 6, av * (0.25 if j % 2 else 1.0))
            continue

        fl = None
        if k == STOP:
            burst = sg["burst"]
            rel = e - burst - sg["post"]
            voiced = info["v"]
            ba = 10 ** ((BURST_DB[info["place"]] - (3 if voiced else 0)) / 20) * sg["burst_amp"]
            T.add("AV", s + r, 0.0)
            T.add("AV", e - 0.001, 0.0)
            T.add("AH", s + r, 0.0)
            T.add("B1ADD", s + r, 0.0)
            if sg["post"] > 0:
                T.add("AH", rel + burst, 0.0)
                T.add("AH", rel + burst + 0.004, 0.55 if not voiced else 0.2)
                T.add("AH", e - 0.002, 0.0)
                T.add("B1ADD", rel + burst, 200.0)
            else:
                T.add("AH", e - 0.001, 0.0)
            if voiced:
                vb = 0.5 if prv_pause else 1.0
                T.add("AVB", s + min(0.012, d / 4), vb)
                T.add("AVB", rel - 0.002, vb)
                T.add("AVB", rel + 0.004, 0.0)
            else:
                T.add("AVB", s + r, 0.0)
            T.add("AVB", e, 0.0)
            T.add("AF", s + r, 0.0)
            T.add("AF", rel - 0.0004, 0.0)
            T.add("AF", rel, ba)
            T.add("AF", rel + burst, ba * 0.35)
            T.add("AF", rel + burst + 0.004, 0.0)
            if info["place"] == "vel":
                vt = _nearest_vowel(segs, i)
                fv = float(np.clip((vt[1] if vt else 1600) * 1.05 + 100, 1500, 3000))
                fl = ([(fv, 450, 1.0), (fv * 1.45, 900, 0.35), (5000, 3000, 0.1)], 0.05)
            else:
                fl = BURST_SPEC[info["place"]]
        elif k == AFFR:
            c = 0.42 * d
            voiced = info["v"]
            a = 10 ** (FRIC_DB[p] / 20)
            T.add("AV", s + r, 0.0)
            T.add("AH", s + r, 0.0)
            T.add("B1ADD", s + r, 0.0)
            T.add("AF", s + r, 0.0)
            T.add("AF", s + c - 0.0004, 0.0)
            T.add("AF", s + c, a * 1.2)
            T.add("AF", e - min(0.015, d / 4), a)
            if voiced:
                T.add("AVB", s + min(0.012, d / 4), 0.8)
                T.add("AVB", s + c, 0.6)
                T.add("AV", s + c, 0.0)
                T.add("AV", e - min(0.02, d / 4), 0.35)
                T.add("AVB", e - r, 0.3)
            else:
                T.add("AV", e - 0.002, 0.0)
                T.add("AVB", s + r, 0.0)
                T.add("AVB", e - r, 0.0)
            if nxt_pause:
                T.add("AF", e, 0.0)
            fl = FRIC_SPEC[p]
        elif k == ASP:
            ro = min(0.012, d / 3)
            T.add("AV", s + ro, 0.0)
            T.add("AV", e - 0.002, 0.0)
            T.add("AH", s + min(0.02, d / 3), 1.0)
            T.add("AH", e - 0.002, 0.85)
            T.add("B1ADD", s + ro, 150.0)
            T.add("B1ADD", e, 120.0)
            for nm in ("AF", "AVB"):
                T.add(nm, s + r, 0.0)
                T.add(nm, e - r, 0.0)
        elif k == FRIC:
            voiced = info["v"]
            rise = min(0.02, d / 3)
            a = 10 ** (FRIC_DB[p] / 20)
            ro = min(0.012, d / 3)
            if voiced:
                T.add("AV", s + rise, 0.45)
                T.add("AV", e - rise, 0.45)
                T.add("AVB", s + rise, 0.6)
                T.add("AVB", e - rise, 0.6)
            else:
                T.add("AV", s + ro, 0.0)
                T.add("AV", e - ro, 0.0)
                T.add("AVB", s + r, 0.0)
                T.add("AVB", e - r, 0.0)
            T.add("AF", s + rise, a)
            T.add("AF", e - rise, a)
            if prv_pause:
                T.add("AF", s, 0.0)
            if nxt_pause:
                T.add("AF", e, 0.0)
            asp = 0.45 if p == "X" else 0.0
            T.add("AH", s + rise, asp)
            T.add("AH", e - rise, asp)
            T.add("B1ADD", s + r, 0.0)
            T.add("B1ADD", e - r, 0.0)
            if p == "X":
                vt = _nearest_vowel(segs, i)
                fv = float(np.clip((vt[1] if vt else 1400) * 1.0 + 100, 1200, 2600))
                fl = ([(fv, 700, 1.0), (fv * 1.8, 1500, 0.5), (5000, 3000, 0.1)], 0.0)
            else:
                fl = FRIC_SPEC[p]
        if fl is not None:
            spec, byp = fl
            fsh = fshift ** 0.7
            for t in (s + 0.002, e - 0.002):
                for j, (fc, bw, g) in enumerate(spec):
                    T.add(f"FR{j}F", t, fc * fsh)
                    T.add(f"FR{j}B", t, bw * fsh)
                    T.add(f"FR{j}G", t, g)
                T.add("FRBYP", t, byp)
    return T


def _pitch_points(segs, phrases, vs, intonation, emphasis):
    """F0 breakpoints (time, Hz)."""
    base = vs["pitch"]
    pts = []
    if intonation == "flat" or vs.get("flat"):
        return [(0.0, base), (segs[-1]["e"] + 1.0, base)]
    nph = len(phrases)
    for pi, phr in enumerate(phrases):
        idx = [j for j, s in enumerate(segs) if s.get("phrase") == pi and s["ph"] != "_"]
        if not idx:
            continue
        ps, pe = segs[idx[0]]["s"], segs[idx[-1]]["e"]
        span = max(pe - ps, 1e-3)
        mode = intonation
        if mode == "auto":
            mode = {"?": "question", "!": "exclaim"}.get(phr["punct"], "statement")
        if pi < nph - 1 and phr["punct"] == "," and mode == "statement":
            mode = "continue"
        top = 1.12 if mode == "exclaim" else 1.06
        drop = 0.2 if mode == "exclaim" else 0.14

        def decl(t):
            return base * (top - drop * (t - ps) / span)

        acc_scale = emphasis * (1.8 if mode == "exclaim" else 1.0)
        vows = [j for j in idx if PH[segs[j]["ph"]]["k"] in VOWELISH]
        stressed = [j for j in vows if segs[j]["st"] >= 1] or vows[-1:]
        nuclear = stressed[-1] if stressed else None
        pts.append((ps, decl(ps) * 0.97))
        for n_acc, j in enumerate(vows):
            s, e = segs[j]["s"] + segs[j].get("head", 0.0), segs[j]["e"]
            d = e - s
            st = segs[j]["st"]
            if j == nuclear:
                break
            if st == 1:
                acc = (0.16 if j == stressed[0] else 0.1) * acc_scale
            elif st == 2:
                acc = 0.06 * acc_scale
            else:
                acc = -0.02
            pts.append((s + 0.3 * d, decl(s + 0.3 * d) * (1 + acc)))
            pts.append((e, decl(e) * (1 + acc * 0.6)))
        if nuclear is not None:
            s, e = segs[nuclear]["s"] + segs[nuclear].get("head", 0.0), segs[nuclear]["e"]
            d = e - s
            if mode == "question":
                pts.append((s + 0.25 * d, decl(s) * 0.95))
                pts.append((pe, base * (1.0 + 0.45 * max(emphasis, 0.3))))
            elif mode == "continue":
                pts.append((s + 0.3 * d, decl(s) * (1 + 0.1 * acc_scale)))
                pts.append((pe, decl(pe) * 1.04))
            else:
                acc = 0.13 * acc_scale
                pts.append((s + 0.2 * d, decl(s) * (1 + acc)))
                end = 0.72 if mode == "exclaim" else 0.8
                pts.append((pe, base * (end + (1 - end) * (1 - min(emphasis, 1.5) / 1.5) * 0.5)))
    return pts


# ----------------------------------------------------------------------------------------------
# DSP
# ----------------------------------------------------------------------------------------------
def _tv_filter(x: np.ndarray, B: np.ndarray, A: np.ndarray, blk: int) -> np.ndarray:
    """Time-varying biquad, coefficients per block. State carried as past in/out samples (direct form I),
    so coefficient changes never jump."""
    n = x.shape[0]
    y = np.empty(n)
    x1 = x2 = y1 = y2 = 0.0
    for i in range(B.shape[0]):
        s = i * blk
        if s >= n:
            break
        e = min(s + blk, n)
        b, a = B[i], A[i]
        zi = np.array([b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2, b[2] * x1 - a[2] * y1])
        xb = x[s:e]
        yb, _ = lfilter(b, a, xb, zi=zi)
        y[s:e] = yb
        if e - s >= 2:
            x1, x2, y1, y2 = xb[-1], xb[-2], yb[-1], yb[-2]
        else:
            x2, x1, y2, y1 = x1, xb[-1], y1, yb[-1]
    return y


def _reson_coefs(f, bw, sr):
    """Klatt resonator (unity DC gain) coefficients per block."""
    f = np.minimum(np.asarray(f, float), 0.47 * sr)
    bw = np.maximum(np.asarray(bw, float), 10.0)
    T = 1.0 / sr
    C = -np.exp(-2 * np.pi * bw * T)
    Bc = 2 * np.exp(-np.pi * bw * T) * np.cos(2 * np.pi * f * T)
    Ac = 1 - Bc - C
    nb = f.shape[0]
    b = np.zeros((nb, 3))
    a = np.zeros((nb, 3))
    b[:, 0] = Ac
    a[:, 0] = 1
    a[:, 1] = -Bc
    a[:, 2] = -C
    return b, a


def _anti_coefs(f, bw, sr):
    rb, ra = _reson_coefs(f, bw, sr)
    Ac, Bc, C = rb[:, 0], -ra[:, 1], -ra[:, 2]
    b = np.stack([1 / Ac, -Bc / Ac, -C / Ac], axis=1)
    a = np.zeros_like(b)
    a[:, 0] = 1
    return b, a


def _bp_coefs(f, bw, sr):
    """Constant 0 dB peak band-pass (RBJ) per block."""
    f = np.clip(np.asarray(f, float), 20, 0.46 * sr)
    q = np.maximum(f / np.maximum(np.asarray(bw, float), 20.0), 0.3)
    w0 = 2 * np.pi * f / sr
    alpha = np.sin(w0) / (2 * q)
    a0 = 1 + alpha
    b = np.stack([alpha / a0, np.zeros_like(alpha), -alpha / a0], axis=1)
    a = np.stack([np.ones_like(a0), -2 * np.cos(w0) / a0, (1 - alpha) / a0], axis=1)
    return b, a


def _smooth(x, width):
    width = int(max(1, width))
    if width <= 1:
        return x
    k = np.ones(width) / width
    pad = width // 2
    xp = np.concatenate([np.full(pad, x[0]), x, np.full(width - 1 - pad, x[-1])])
    return np.convolve(xp, k, mode="valid")


def _slow_noise(n, sr, rate_hz, rng):
    """Smooth random signal ~N(0,1) varying at about rate_hz."""
    m = max(4, int(n / sr * rate_hz) + 4)
    pts = rng.standard_normal(m)
    return np.interp(np.linspace(0, m - 3, n), np.arange(m), pts)


def _resolve(params: dict) -> dict:
    unknown = [k for k in params if k not in PARAMS]
    if unknown:
        raise ValueError(f"speech does not accept params {unknown}; allowed: {list(PARAMS)}")
    p = {k: v[0] for k, v in PARAMS.items()}
    p.update({k: v for k, v in params.items() if v is not None})
    voice = str(p["voice"]).lower()
    if voice not in VOICES:
        raise ValueError(f"unknown voice {voice!r}; choose from {list(VOICES)}")
    vs = dict(VOICES[voice])
    if p["pitch"] is not None:
        from .notes import to_hz
        vs["pitch"] = to_hz(p["pitch"])
    if p["formant_shift"] is not None:
        vs["formant_shift"] = float(p["formant_shift"])
    if p["breath"] is not None:
        vs["breath"] = float(p["breath"])
    p["intonation"] = str(p["intonation"]).lower()
    if p["intonation"] not in ("auto", "statement", "question", "exclaim", "flat"):
        raise ValueError("intonation must be auto|statement|question|exclaim|flat")
    return p, vs


def render_speech(text: str | None = None, sr: int = DEFAULT_SR, phonemes: str | None = None, **params) -> np.ndarray:
    p, vs = _resolve(params)
    lang = str(p["lang"]).lower()
    if phonemes:
        phrases = [{"words": _parse_raw(phonemes), "punct": ""}]
    else:
        if not text or not str(text).strip():
            raise ValueError("speech needs 'text' (or 'phonemes')")
        phrases = text_to_phrases(str(text), lang)
    phrases = [ph for ph in phrases if any("ph" in w and w["ph"] for w in ph["words"])]
    if not phrases:
        raise ValueError(f"nothing to say in {text!r}")
    rate = float(p["rate"])
    segs = _build_segments(phrases, rate, float(p["gap"]), lang)
    # drop trailing pause segments
    while segs and segs[-1]["ph"] == "_":
        segs.pop()
    T = _build_tracks(segs, vs, lang)
    intonation = p["intonation"]
    if intonation == "exclaim" or (intonation == "auto" and phrases[-1]["punct"] == "!"):
        vs["oq"] = max(0.45, vs["oq"] - 0.08)
    pitch_pts = _pitch_points(segs, phrases, vs, intonation, float(p["emphasis"]))
    for t, f in pitch_pts:
        T.add("F0", t, f)
    return _synth(segs, T, vs, sr, float(p["vibrato"]), int(p["seed"]))


def _character(out, vs, sr, ts, rng):
    """Post-voice character stage for the non-human presets."""
    n = len(out)
    if vs.get("ring"):
        m = vs["ring"]
        out = out * (1 - m) + out * np.sin(2 * np.pi * vs.get("ring_hz", 55.0) * ts) * m * 1.5
    if vs.get("drive"):
        d = vs["drive"]
        pk = np.max(np.abs(out)) + 1e-9
        out = np.tanh(out / pk * d) / np.tanh(d) * pk
    if vs.get("crush_bits"):
        pk = np.max(np.abs(out)) + 1e-9
        q = 2 ** (vs["crush_bits"] - 1)
        hold = max(1, int(round(sr / vs.get("crush_sr", sr))))
        held = np.repeat(out[::hold], hold)[:n]
        out = np.round(held / pk * q) / q * pk
    if vs.get("glitch"):
        # per 40 ms block: stutter (repeat the previous block), dropout, or reverse
        blk = max(1, int(0.04 * sr))
        out = out.copy()
        for i in range(blk, n - blk, blk):
            if rng.random() < vs["glitch"]:
                kind = rng.integers(0, 3)
                if kind == 0:
                    out[i:i + blk] = out[i - blk:i]
                elif kind == 1:
                    out[i:i + blk] *= 0.05
                else:
                    out[i:i + blk] = out[i:i + blk][::-1]
        # short fades at block edges would soften it; the clicks are part of the malfunction
    if vs.get("chorus"):
        c = vs["chorus"]
        delay = (0.012 + 0.003 * np.sin(2 * np.pi * 0.8 * ts)) * sr
        idx = np.arange(n) - delay
        wet = np.interp(idx, np.arange(n), out, left=0.0)
        out = out * (1 - 0.4 * c) + wet * 0.6 * c
    if vs.get("tremolo"):
        rate, depth = vs["tremolo"]
        out = out * (1 - depth * (0.5 + 0.5 * np.sin(2 * np.pi * rate * ts)))
    return out


def _synth(segs, T: _Tracks, vs: dict, sr: int, vibrato: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(1000 + seed)
    total = segs[-1]["e"] + 0.12
    n = samples(total, sr)
    blk = max(8, int(round(sr / 1400)))
    nb = (n + blk - 1) // blk
    tc = (np.arange(nb) + 0.5) * blk / sr
    ts = np.arange(n) / sr

    def up(ctl):  # control-rate -> sample-rate
        return np.interp(ts, tc, ctl)

    # ---- source
    f0c = _smooth(T.curve("F0", tc, vs["pitch"]), 0.045 * sr / blk)
    f0 = up(f0c)
    if vs["jitter"] > 0:
        f0 = f0 * (1 + vs["jitter"] * _slow_noise(n, sr, 25, rng))
        f0 = f0 * (1 + 0.004 * (np.sin(2 * np.pi * 4.7 * ts) + np.sin(2 * np.pi * 7.1 * ts + 1.0)))
    if vs.get("quantize"):
        f0 = 440.0 * 2 ** (np.round(12 * np.log2(np.maximum(f0, 1) / 440.0)) / 12)
    if not vibrato and vs.get("wobble"):
        f0 = f0 * 2 ** (vs["wobble"] / 12 * np.sin(2 * np.pi * 7.0 * ts))
    if vibrato:
        f0 = f0 * 2 ** (vibrato / 12 * np.sin(2 * np.pi * 5.5 * ts) * np.clip(ts / 0.25, 0, 1))
    if vs.get("pitch_glitch"):
        # sudden pitch hiccups: some 60 ms blocks jump up or down a few semitones
        blkn = max(1, int(0.06 * sr))
        steps = rng.choice([0, 0, 0, 3, -4, 7, -7], size=n // blkn + 1)
        steps = np.where(rng.random(len(steps)) < vs["pitch_glitch"], steps, 0)
        f0 = f0 * 2 ** (np.repeat(steps, blkn)[:n] / 12)
    f0 = np.clip(f0, 40, 0.2 * sr)
    ph = np.cumsum(f0 / sr)
    t = ph % 1.0
    oq = vs["oq"]
    tp = oq * 0.66
    tn = oq - tp
    J = np.pi / (2 * tn)
    dflow = np.zeros(n)
    flow = np.zeros(n)
    m1 = t < tp
    m2 = (t >= tp) & (t < oq)
    dflow[m1] = (np.pi / (2 * tp)) * np.sin(np.pi * t[m1] / tp)
    flow[m1] = 0.5 * (1 - np.cos(np.pi * t[m1] / tp))
    dflow[m2] = -J * np.sin(0.5 * np.pi * (t[m2] - tp) / tn)
    flow[m2] = np.cos(0.5 * np.pi * (t[m2] - tp) / tn)
    dt = np.maximum(f0 / sr, 1e-6)
    dflow += (J / 2) * _polyblep((t - oq) % 1.0, dt)  # band-limit the glottal closure
    src = dflow / J
    if vs.get("source") == "saw":
        src = -((2 * t - 1) - _polyblep(t, dt))  # buzzy vocoder carrier
    if vs.get("sub"):
        ph2 = (ph / 2) % 1.0
        sq = np.where(ph2 < 0.5, 1.0, -1.0) + _polyblep(ph2, dt / 2) - _polyblep((ph2 + 0.5) % 1.0, dt / 2)
        src = src + sq * vs["sub"] * 0.6
    if vs.get("tilt"):
        a = np.exp(-2 * np.pi * vs["tilt"] / sr)
        src = lfilter([1 - a], [1, -a], src)
        src = src / (np.sqrt(np.mean(src ** 2)) + 1e-9) * 0.35
    if vs["shimmer"] > 0:
        src = src * (1 + vs["shimmer"] * _slow_noise(n, sr, 30, rng))

    AV = up(T.curve("AV", tc, 0.0))
    AH = up(T.curve("AH", tc, 0.0))
    AF = up(T.curve("AF", tc, 0.0))
    AVB = up(T.curve("AVB", tc, 0.0))
    if vs.get("whisper"):
        AH = np.maximum(AH, AV * 0.85)
        AV = AV * 0.0
        AVB = AVB * 0.0

    asp_noise = rng.uniform(-1, 1, n)
    asp_noise = lfilter([0.5, 0.5], [1.0], asp_noise)  # tame the very top
    k_asp = 0.16
    breath = vs["breath"]
    exc = AV * src + (AH * k_asp + breath * k_asp * 0.7 * AV * (0.3 + flow)) * asp_noise

    # ---- cascade (nasal pole/zero + F5..F1)
    F = [T.curve(f"FM{j}", tc, d) for j, d in enumerate([500, 1500, 2500, F4, F5, 60, 90, 150, 250, 320])]
    F = [_smooth(f, 0.008 * sr / blk) for f in F]
    B1add = T.curve("B1ADD", tc, 0.0)
    bwk = vs["bw"]
    y = exc
    fnz = T.curve("FNZ", tc, FNP)
    b, a = _reson_coefs(np.full(nb, FNP * vs["formant_shift"]), np.full(nb, 100.0), sr)
    y = _tv_filter(y, b, a, blk)
    b, a = _anti_coefs(fnz, np.full(nb, 100.0), sr)
    y = _tv_filter(y, b, a, blk)
    for j in (4, 3, 2, 1, 0):
        bw = F[5 + j] * bwk + (B1add if j == 0 else 0.0)
        b, a = _reson_coefs(F[j], bw, sr)
        y = _tv_filter(y, b, a, blk)
    # higher vocal-tract poles (a real tract keeps resonating every ~1 kHz; without these the cascade
    # rolls off far too steeply at 44.1 kHz and sounds muffled)
    for fh, bh in ((4900.0, 600.0), (5900.0, 800.0), (7000.0, 1000.0)):
        fh *= vs["formant_shift"]
        if fh < 0.42 * sr:
            b, a = _reson_coefs(np.full(nb, fh), np.full(nb, bh), sr)
            y = _tv_filter(y, b, a, blk)
    # restore the air above ~3 kHz that the unity-gain cascade removes (higher-pole correction)
    from . import filters as FL
    y = FL.highshelf(y, 3000 * vs["formant_shift"], sr, gain_db=vs.get("air", 12.0))
    voiced_out = y

    # reference level: RMS inside stressed vowels
    mask = np.zeros(n, bool)
    for sg in segs:
        if sg["ph"] != "_" and PH[sg["ph"]]["k"] in VOWELISH and sg.get("st", 0) >= 1:
            s0 = int((sg["s"] + 0.3 * (sg["e"] - sg["s"])) * sr)
            e0 = int((sg["e"] - 0.3 * (sg["e"] - sg["s"])) * sr)
            mask[s0:e0] = True
    if not mask.any():
        mask = AV > 0.5
    ref = float(np.sqrt(np.mean(voiced_out[mask] ** 2))) if mask.any() else float(np.sqrt(np.mean(voiced_out ** 2)))
    ref = max(ref, 1e-6)

    out = voiced_out
    # ---- frication branch (parallel)
    if np.max(AF) > 0:
        noise = rng.uniform(-1, 1, n) * np.sqrt(3)  # unit variance
        # voiced fricatives: noise pulses with the glottal cycle
        vmod = np.clip(AV, 0, 1)
        noise = noise * (1 - 0.6 * vmod + 1.2 * vmod * flow)
        fr = np.zeros(n)
        power = np.zeros(nb)
        for j in range(3):
            fc = T.curve(f"FR{j}F", tc, 5000.0)
            bw = T.curve(f"FR{j}B", tc, 2000.0)
            g = T.curve(f"FR{j}G", tc, 0.0)
            b, a = _bp_coefs(fc, bw, sr)
            fr += _tv_filter(noise, b, a, blk) * up(g)
            power += g ** 2 * (np.pi / 2) * bw / (sr / 2)
        byp = T.curve("FRBYP", tc, 0.0)
        fr += noise * up(byp)
        power += byp ** 2
        fr = fr / up(np.sqrt(np.maximum(power, 1e-4)))
        # keep frication out of the rumble and the extreme top
        from . import filters as FL
        fr = FL.highpass(fr, 700, sr)
        fr = FL.lowpass(fr, min(12000, 0.45 * sr), sr)
        out = out + fr * AF * ref

    # ---- voice bar (low murmur during voiced closures / voiced fricatives)
    if np.max(AVB) > 0:
        from . import filters as FL
        vb = FL.lowpass(flow - np.mean(flow), 350 * vs["formant_shift"], sr, order=2)
        vb = vb / (np.sqrt(np.mean(vb ** 2)) + 1e-9)
        out = out + vb * AVB * ref * 10 ** (-13 / 20)

    out = _character(out, vs, sr, ts, rng)
    from . import filters as FL
    out = FL.highpass(out, 55, sr, order=2)
    peak = np.max(np.abs(out))
    if peak > 1e-9:
        out = out / peak * 0.89
    return out
