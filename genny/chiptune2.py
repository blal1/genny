"""More sound chips: SN76489, NES delta modulation, Yamaha YM2151 and YM2610, the 8580 SID and the Atari POKEY.

Registers
    chip system  `sn76489`   for the `chip` layer of genny.chiptune: tone1, tone2, tone3, noise
    instrument   `sn_tone`, `sn_noise`
    effect       `dpcm`      any sound through the NES DMC: 1-bit delta samples at one of its sixteen rates

Sources (pages kept in out/research/chips):
    smspower_SN76489.txt   SMS Power development wiki, "SN76489".
    nesdev_APU_DMC.wiki    NESdev wiki, "APU DMC".
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np
from scipy.signal import butter, lfilter

from . import filters as F
from . import chiptune as CT
from .chiptune import FRAME_HZ, _tab
from .core import DEFAULT_SR, samples
from .fx import effect
from .instruments import _soften, instrument
from .notes import parse_sequence
from .retro import _fade, _to_sr
from .spec import SpecError


# ============================================================================ shared: lanes and driver frames
def lanes_of(layer: dict, system: str, names: tuple, defaults: dict, q: float, mono_chord: str = "arp"):
    """The `channels` of a chip layer -> {channel: [(t0, t1, hz, vel, patch)]} and the length in seconds."""
    out, end = {}, 0.0
    for name, lane_list in (layer.get("channels") or {}).items():
        if name not in names:
            raise SpecError(f"chip layer: {system} has no channel {name!r} (it has {', '.join(names)})")
        for lane in lane_list if isinstance(lane_list, list) else [lane_list]:
            p = dict(defaults)
            for k, v in lane.items():
                if k in defaults:
                    p[k] = v
                elif k not in ("steps", "step", "vel", "transpose", "pan"):
                    raise SpecError(f"chip layer, {name}: unknown key {k!r} (known: {', '.join(sorted(defaults))})")
            p["pan"] = str(lane.get("pan", "C")).upper()[:1]
            t, tr = 0.0, 2.0 ** (float(lane.get("transpose", 0)) / 12.0)
            for st in parse_sequence(lane.get("steps", ""), float(lane.get("step", 0.25))):
                hz = st["pitches"]
                if hz:
                    pp = p
                    if len(hz) > 1:
                        if mono_chord != "arp":
                            raise SpecError(f"chip layer, {name}: one note at a time per channel")
                        if p.get("arp") is None:
                            pp = dict(p, arp=[12.0 * math.log2(h / hz[0]) for h in hz])
                    out.setdefault(name, []).append((t * q, (t + st["dur"] * float(p.get("gate", 1.0))) * q, hz[0] * tr,
                                                     float(lane.get("vel", 1.0)) * st["vel"], pp))
                t += st["dur"]
            end = max(end, t * q)
    return out, max(end, 0.05)


def driver_rows(c0, c1, clk, freq, vel, p, period_of, rows, aux=0):
    """One note as register rows (cycle, period, volume 0..15, aux, reset), one per 60 Hz driver frame: the chip has
    no envelope of its own, so volume, arpeggio, pitch and vibrato tables are all the music routine's."""
    f, first = 0, True
    while True:
        c = c0 + int(round(f * clk / FRAME_HZ))
        if c >= c1:
            break
        semis = _tab(p.get("arp"), f, True) + _tab(p.get("pitch"), f)
        t = (c - c0) / clk - float(p.get("vib_delay", 0.15))
        if p.get("vib") and t > 0:
            semis += float(p["vib"]) * math.sin(2 * math.pi * float(p.get("vib_hz", 6.0)) * t)
        v = int(_tab(p["vol"], f)) if p.get("vol") is not None else int(p.get("volume", 15))
        v = min(max(int(round(v * vel)), 1 if v else 0), 15)
        row = (int(c), int(period_of(freq * 2.0 ** (semis / 12.0))), v, aux, 1 if first else 0)
        if first or rows[-1][1:4] != row[1:4]:
            rows.append(row)
        first = False
        f += 1
    rows.append((int(c1), rows[-1][1], 0, aux, 0))


# ============================================================================ SN76489
SN_CLOCK = {"ntsc": 3579545.0, "pal": 3546893.0}                 # SMS Power: the colour subcarrier, divided by 16 inside the chip
SN_VOL = np.array([10.0 ** (-2.0 * (15 - v) / 20.0) if v else 0.0 for v in range(16)])   # 2 dB a step; here 15 = full, 0 = silence
# variant -> (white-noise tapped bits, shift register width). SMS Power: Sega's chips tap bits 0 and 3 of 16,
# the TI part (BBC Micro, ColecoVision, SG-1000) bits 0 and 1 of 15.
SN_VARIANTS = {"sega": (0x0009, 16), "ti": (0x0003, 15)}
SN_NAMES = ("tone1", "tone2", "tone3", "noise")
SN_PATCH = {"volume": 15, "vol": None, "arp": None, "pitch": None, "vib": 0.0, "vib_hz": 6.0, "vib_delay": 0.15, "gate": 1.0,
            "white": True, "rate": None}


@nb.njit(cache=True)
def _sn_core(n_out, dec, seg, off, vol_tab, taps, width):
    """Three tone counters and the noise, one step per tick of clock / 16. Rows as in genny.chiptune; the noise row's
    aux is 1 for white noise, 0 for "periodic", and its reset flag restarts the shift register."""
    out = np.empty(n_out, np.float32)
    ptr = off[:4].copy()
    per = np.ones(4, np.int64)
    vol = np.zeros(4, np.int64)
    aux = np.zeros(4, np.int64)
    cnt = np.ones(4, np.int64)
    bit = np.zeros(4, np.int64)
    lfsr = 1 << (width - 1)
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
                        per[c] = seg[r, 1] if seg[r, 1] > 0 else 1
                        vol[c] = seg[r, 2]
                        aux[c] = seg[r, 3]
                        if c == 3 and seg[r, 4] != 0:      # a write to the noise register: all zero but the highest bit
                            lfsr = 1 << (width - 1)
                        ptr[c] += 1
                    if ptr[c] < off[c + 1] and seg[ptr[c], 0] < nxt:
                        nxt = seg[ptr[c], 0]
            for c in range(4):
                cnt[c] -= 1
                if cnt[c] <= 0:
                    cnt[c] = per[c]
                    bit[c] ^= 1
                    if c == 3 and bit[3] == 1:             # the noise output bit clocks the shift register on its rising edge
                        if aux[3] != 0:
                            x = lfsr & taps
                            fb = 0
                            while x:
                                fb ^= x & 1
                                x >>= 1
                        else:
                            fb = lfsr & 1
                        lfsr = (lfsr >> 1) | (fb << (width - 1))
            acc += (vol_tab[vol[0]] * bit[0] + vol_tab[vol[1]] * bit[1] + vol_tab[vol[2]] * bit[2]
                    + vol_tab[vol[3]] * (lfsr & 1))
            cyc += 1
        out[i] = acc / dec
    return out


SN_GAIN = 0.55


def render_sn(lanes, dur, sr, region="ntsc", variant="sega"):
    """`dur` seconds of an SN76489. lanes: channel -> notes (t0, t1, hz, vel, patch)."""
    clk = SN_CLOCK[region] / 16.0
    taps, width = SN_VARIANTS[variant]
    rows, off = [], [0]
    for name in SN_NAMES:
        notes = sorted(lanes.get(name, []), key=lambda n: n[0])
        for i, (t0, t1, hz, vel, p) in enumerate(notes):
            if i + 1 < len(notes):
                t1 = min(t1, notes[i + 1][0])
            c0, c1 = int(round(t0 * clk)), int(round(t1 * clk))
            if c1 <= c0:
                continue
            if name != "noise":                            # f = clock / (2 x register x 16): the counter is half a period
                per = lambda f: min(max(round(clk / (2.0 * f)), 1), 1023)
                white = 0
            else:
                white = int(bool(p["white"]))
                if p["rate"] is not None:                  # the three fixed shift rates: 0x10, 0x20, 0x40
                    fixed = (0x10, 0x20, 0x40)[int(p["rate"]) % 3]
                    per = lambda f, fixed=fixed: fixed
                elif white:
                    per = lambda f: min(max(round(clk / (2.0 * f)), 1), 1023)
                else:                                      # "periodic" noise is the register looped: one pulse every `width` shifts
                    per = lambda f, width=width: min(max(round(clk / (2.0 * f * width)), 1), 1023)
            driver_rows(c0, c1, clk, hz, vel, p, per, rows, aux=white)
        off.append(len(rows))
    seg = np.array(rows, np.int64).reshape(-1, 5)
    x = _sn_core(int(dur * clk / 2) + 1, 2, seg, np.array(off, np.int64), SN_VOL, taps, width).astype(np.float64)
    y = _to_sr(x - x[0], clk / 2.0, sr)
    y = lfilter(*butter(1, 20.0, "high", fs=sr), y)        # UNSOURCED: the output is AC-coupled; 20 Hz is genny's choice
    return y[:samples(dur, sr)] * SN_GAIN


def _sn_layer(layer: dict, sr: int, q: float) -> np.ndarray:
    lanes, end = lanes_of(layer, "sn76489", SN_NAMES, SN_PATCH, q)
    variant = str(layer.get("variant", "sega")).lower()
    if variant not in SN_VARIANTS:
        raise SpecError(f"chip layer: sn76489 variant must be one of {sorted(SN_VARIANTS)}")
    return render_sn(lanes, end, sr, region=str(layer.get("region", "ntsc")).lower(), variant=variant)


CT.SYSTEM_LAYERS["sn76489"] = _sn_layer
CT.SYSTEMS["sn76489"] = SN_NAMES


def _sn_voice(channel, freq, dur, sr, vel, mode, bright, variant, **patch):
    rel = 0.01
    p = dict(SN_PATCH, **patch)
    exact = str(mode) == "hardware_accurate"
    # A write to the noise register leaves one bit set, so the chip is silent for the first 15 shifts: a third of a
    # second on a low note. The stylized voice starts the register early and plays from where the noise begins.
    pre = min(20.0 / max(freq, 1.0), 0.5) if channel == "noise" and not exact else 0.0
    y = render_sn({channel: [(0.0, dur + pre, float(freq), 1.0, p)]}, dur + pre + rel, sr * 2, variant=variant)
    y = y[int(pre * sr * 2):]
    if not exact and channel != "noise":                   # the 10-bit counter rounds the pitch
        clk = SN_CLOCK["ntsc"] / 16.0
        got = clk / (2.0 * min(max(round(clk / (2.0 * freq)), 1), 1023))
        y = _to_sr(y, sr * 2 * freq / got, sr * 2)
    y = F.dc_block(_to_sr(y, sr * 2, sr)[:samples(dur + rel, sr)], sr)
    if not exact:
        if channel != "noise":
            y = _soften(y, freq, sr, harm=6, lo=2200, hi=3800, bright=bright)
        else:
            fc = 3000.0 * 2.0 ** ((bright - 0.5) * 4.0)
            if fc < sr * 0.45:
                y = F.lowpass(F.lowpass(y, fc, sr, 0.707), fc, sr, 0.707)
    return _fade(y.copy(), sr, 1.0, rel * 1000.0) * (0.5 + 0.5 * vel)


_H = CT.PATCH_HELP


@instrument("sn_tone", "SN76489 tone channel (Master System, BBC Micro): a plain square from a 10-bit counter, 2 dB volume steps.",
            family="retro", span=("A2", "C7"), volume=(15, "0..15, 2 dB a step"), vol=(None, _H["vol"]), arp=(None, _H["arp"]),
            pitch=(None, _H["pitch"]), vib=(0.0, _H["vib"]), vib_hz=(6.0, _H["vib_hz"]), vib_delay=(0.15, _H["vib_delay"]),
            mode=CT._MODE, bright=(0.35, CT._BRIGHT[1]))
def sn_tone(freq, dur, sr=DEFAULT_SR, vel=1.0, volume=15, vol=None, arp=None, pitch=None, vib=0.0, vib_hz=6.0, vib_delay=0.15,
            mode="retro_stylized", bright=0.35):
    return _sn_voice("tone1", freq, dur, sr, vel, mode, bright, "sega", volume=volume, vol=vol, arp=arp, pitch=pitch, vib=vib,
                     vib_hz=vib_hz, vib_delay=vib_delay)


@instrument("sn_noise", "SN76489 noise channel: white noise from its shift register, or the looped register as a thin bass.",
            family="retro", span=("C2", "C8"), white=(True, "true = white noise; false = the 'periodic' mode, a 1/16 pulse an octave-and-more down (the Master System bass)"),
            rate=(None, "0, 1 or 2 = the chip's three fixed shift rates; null = from the note"), volume=(15, "0..15, 2 dB a step"),
            vol=(None, _H["vol"]), pitch=(None, _H["pitch"]), variant=("sega", "sega = 16-bit register tapped at bits 0 and 3 | ti = 15-bit, bits 0 and 1"),
            mode=CT._MODE, bright=CT._BRIGHT)
def sn_noise(freq, dur, sr=DEFAULT_SR, vel=1.0, white=True, rate=None, volume=15, vol=None, pitch=None, variant="sega",
             mode="retro_stylized", bright=0.5):
    return _sn_voice("noise", freq, dur, sr, vel, mode, bright, variant, white=white, rate=rate, volume=volume, vol=vol, pitch=pitch)


# ============================================================================ NES DMC
# APU DMC, NTSC: CPU cycles between output changes for rate index $0..$F (4181.71 Hz .. 33143.9 Hz)
DMC_RATES = (428, 380, 340, 320, 286, 254, 226, 214, 190, 160, 142, 128, 106, 84, 72, 54)


@nb.njit(cache=True)
def _dmc(x, start):
    """Encode `x` (0..127 targets, one per output clock) to 1-bit deltas and decode them as the output unit does:
    a 1 raises the 7-bit level by 2 unless it is above 125, a 0 lowers it by 2 unless it is below 2."""
    out = np.empty(x.shape[0])
    level = start
    for i in range(x.shape[0]):
        if x[i] > level:
            if level <= 125:
                level += 2
        elif level >= 2:
            level -= 2
        out[i] = level
    return out


@effect("dpcm", "NES delta-modulation channel: the sound as 1-bit delta samples at one of the 2A03's sixteen rates.",
        rate=(15, "rate index 0..15 = 4182 .. 33144 Hz (NTSC); most game samples use 12..15"),
        drive=(1.0, "level into the 7-bit counter: above 1 the slope limit is hit sooner (grittier), below 1 it is quieter and noisier"),
        mix=(1.0, "0..1"))
def dpcm(x, sr=DEFAULT_SR, rate=15, drive=1.0, mix=1.0):
    """Each output clock the level can only move by 2 of its 128 steps, so fast, loud material slope-overloads:
    that, and the sample-and-hold at the playback rate, is the sound of NES samples. Source: NESdev, APU DMC."""
    def one(ch):
        fs = CT.NES_CPU["ntsc"] / DMC_RATES[int(min(max(rate, 0), 15))]
        peak = float(np.max(np.abs(ch))) or 1.0
        t = _to_sr(np.asarray(ch, np.float64), sr, int(round(fs)))
        target = np.clip(64.0 + 63.0 * drive * t / peak, 0.0, 127.0)
        y = (_dmc(target, 64) - 64.0) / 63.0 * peak / max(drive, 1e-6)
        y = np.repeat(y, 8)                                # hold each level, then come back to the spec's rate
        y = _to_sr(y, int(round(fs)) * 8, sr)
        y = np.pad(y, (0, max(0, ch.shape[0] - y.shape[0])))[:ch.shape[0]]
        return ch * (1.0 - mix) + y * mix
    x = np.asarray(x, np.float64)
    return one(x) if x.ndim == 1 else np.stack([one(x[:, k]) for k in range(x.shape[1])], axis=1)


# ============================================================================ Yamaha YM2151 (OPM) and YM2610 (OPNB)
# Both use the operator, envelope and algorithm engine of genny.chiptune._opn_core (ported from ymfm). What differs
# is read from ymfm_opm.cpp and ymfm_fm.ipp (Aaron Giles, BSD 3-Clause):
#   YM2151  eight channels at clock / 64; the pitch is a key code and a 1/64-semitone fraction looked up in a table of
#           phase steps measured on the chip (David Viens' analysis, reproduced in ymfm), a second coarse detune
#           (DT2) per operator; 14-bit output. Not ported: the YM2151's LFO and the noise on channel 8.
#   YM2610  four FM channels of the OPN family at clock / 144 with a 13-bit output, and three SSG channels.
#           Not modelled: the ADPCM channels, which play recordings.
OPM_CLOCK = 3579545.0
OPM_NAMES = tuple(f"fm{i}" for i in range(1, 9))
OPM_DT2 = (0, (600 * 64 + 50) // 100, (781 * 64 + 50) // 100, (950 * 64 + 50) // 100)   # in 1/64 semitone: 0, 384, 500, 608


# ymfm s_phase_step, divided by 32 (every entry is a multiple of 32): phase step of each 1/64 semitone of the top octave
OPM_STEP = np.array([
    1299, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1308, 1309, 1310, 1311, 1313, 1314, 1315, 1316, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325,
    1327, 1328, 1329, 1330, 1332, 1333, 1334, 1335, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1346, 1347, 1348, 1349, 1351, 1352, 1353, 1354,
    1356, 1357, 1358, 1359, 1361, 1362, 1363, 1364, 1366, 1367, 1368, 1369, 1371, 1372, 1373, 1374, 1376, 1377, 1378, 1379, 1381, 1382, 1383, 1384,
    1386, 1387, 1388, 1389, 1391, 1392, 1393, 1394, 1396, 1397, 1398, 1399, 1401, 1402, 1403, 1404, 1406, 1407, 1408, 1409, 1411, 1412, 1413, 1414,
    1416, 1417, 1418, 1419, 1421, 1422, 1423, 1424, 1426, 1427, 1429, 1430, 1431, 1432, 1434, 1435, 1437, 1438, 1439, 1440, 1442, 1443, 1444, 1445,
    1447, 1448, 1449, 1450, 1452, 1453, 1454, 1455, 1458, 1459, 1460, 1461, 1463, 1464, 1465, 1466, 1468, 1469, 1471, 1472, 1473, 1474, 1476, 1477,
    1479, 1480, 1481, 1482, 1484, 1485, 1486, 1487, 1489, 1490, 1492, 1493, 1494, 1495, 1497, 1498, 1501, 1502, 1503, 1504, 1506, 1507, 1509, 1510,
    1512, 1513, 1514, 1515, 1517, 1518, 1520, 1521, 1523, 1524, 1525, 1526, 1528, 1529, 1531, 1532, 1534, 1535, 1536, 1537, 1539, 1540, 1542, 1543,
    1545, 1546, 1547, 1548, 1550, 1551, 1553, 1554, 1556, 1557, 1558, 1559, 1561, 1562, 1564, 1565, 1567, 1568, 1569, 1570, 1572, 1573, 1575, 1576,
    1578, 1579, 1580, 1581, 1583, 1584, 1586, 1587, 1590, 1591, 1592, 1593, 1595, 1596, 1598, 1599, 1601, 1602, 1604, 1605, 1607, 1608, 1609, 1610,
    1613, 1614, 1615, 1616, 1618, 1619, 1621, 1622, 1624, 1625, 1627, 1628, 1630, 1631, 1632, 1633, 1637, 1638, 1639, 1640, 1642, 1643, 1645, 1646,
    1648, 1649, 1651, 1652, 1654, 1655, 1656, 1657, 1660, 1661, 1663, 1664, 1666, 1667, 1669, 1670, 1672, 1673, 1675, 1676, 1678, 1679, 1681, 1682,
    1685, 1686, 1688, 1689, 1691, 1692, 1694, 1695, 1697, 1698, 1700, 1701, 1703, 1704, 1706, 1707, 1709, 1710, 1712, 1713, 1715, 1716, 1718, 1719,
    1721, 1722, 1724, 1725, 1727, 1728, 1730, 1731, 1734, 1735, 1737, 1738, 1740, 1741, 1743, 1744, 1746, 1748, 1749, 1751, 1752, 1754, 1755, 1757,
    1759, 1760, 1762, 1763, 1765, 1766, 1768, 1769, 1771, 1773, 1774, 1776, 1777, 1779, 1780, 1782, 1785, 1786, 1788, 1789, 1791, 1793, 1794, 1796,
    1798, 1799, 1801, 1802, 1804, 1806, 1807, 1809, 1811, 1812, 1814, 1815, 1817, 1819, 1820, 1822, 1824, 1825, 1827, 1828, 1830, 1832, 1833, 1835,
    1837, 1838, 1840, 1841, 1843, 1845, 1846, 1848, 1850, 1851, 1853, 1854, 1856, 1858, 1859, 1861, 1864, 1865, 1867, 1868, 1870, 1872, 1873, 1875,
    1877, 1879, 1880, 1882, 1884, 1885, 1887, 1888, 1891, 1892, 1894, 1895, 1897, 1899, 1900, 1902, 1904, 1906, 1907, 1909, 1911, 1912, 1914, 1915,
    1918, 1919, 1921, 1923, 1925, 1926, 1928, 1930, 1932, 1933, 1935, 1937, 1939, 1940, 1942, 1944, 1946, 1947, 1949, 1951, 1953, 1954, 1956, 1958,
    1960, 1961, 1963, 1965, 1967, 1968, 1970, 1972, 1975, 1976, 1978, 1980, 1982, 1983, 1985, 1987, 1989, 1990, 1992, 1994, 1996, 1997, 1999, 2001,
    2003, 2004, 2006, 2008, 2010, 2011, 2013, 2015, 2017, 2019, 2021, 2022, 2024, 2026, 2028, 2029, 2032, 2033, 2035, 2037, 2039, 2041, 2043, 2044,
    2047, 2048, 2050, 2052, 2054, 2056, 2058, 2059, 2062, 2063, 2065, 2067, 2069, 2071, 2073, 2074, 2077, 2078, 2080, 2082, 2084, 2086, 2088, 2089,
    2092, 2093, 2095, 2097, 2099, 2101, 2103, 2104, 2107, 2108, 2110, 2112, 2114, 2116, 2118, 2119, 2122, 2123, 2125, 2127, 2129, 2131, 2133, 2134,
    2137, 2139, 2141, 2142, 2145, 2146, 2148, 2150, 2153, 2154, 2156, 2158, 2160, 2162, 2164, 2165, 2168, 2170, 2172, 2173, 2176, 2177, 2179, 2181,
    2185, 2186, 2188, 2190, 2192, 2194, 2196, 2197, 2200, 2202, 2204, 2205, 2208, 2209, 2211, 2213, 2216, 2218, 2220, 2222, 2223, 2226, 2227, 2230,
    2232, 2234, 2236, 2238, 2239, 2242, 2243, 2246, 2249, 2251, 2253, 2255, 2256, 2259, 2260, 2263, 2265, 2267, 2269, 2271, 2272, 2275, 2276, 2279,
    2281, 2283, 2285, 2287, 2288, 2291, 2292, 2295, 2297, 2299, 2301, 2303, 2304, 2307, 2308, 2311, 2315, 2317, 2319, 2321, 2322, 2325, 2326, 2329,
    2331, 2333, 2335, 2337, 2338, 2341, 2342, 2345, 2348, 2350, 2352, 2354, 2355, 2358, 2359, 2362, 2364, 2366, 2368, 2370, 2371, 2374, 2375, 2378,
    2382, 2384, 2386, 2388, 2389, 2392, 2393, 2396, 2398, 2400, 2402, 2404, 2407, 2410, 2411, 2414, 2417, 2419, 2421, 2423, 2424, 2427, 2428, 2431,
    2433, 2435, 2437, 2439, 2442, 2445, 2446, 2449, 2452, 2454, 2456, 2458, 2459, 2462, 2463, 2466, 2468, 2470, 2472, 2474, 2477, 2480, 2481, 2484,
    2488, 2490, 2492, 2494, 2495, 2498, 2499, 2502, 2504, 2506, 2508, 2510, 2513, 2516, 2517, 2520, 2524, 2526, 2528, 2530, 2531, 2534, 2535, 2538,
    2540, 2542, 2544, 2546, 2549, 2552, 2553, 2556, 2561, 2563, 2565, 2567, 2568, 2571, 2572, 2575, 2577, 2579, 2581, 2583, 2586, 2589, 2590, 2593,
], np.int64) * 32


def _opm_step(block, eff):
    """ymfm opm_key_code_to_phase_step, for an octave `block` and a position `eff` in 1/64 semitone above C#."""
    if eff >= 768:
        eff -= 768
        if eff >= 768:
            block, eff = block + 1, eff - 768
        block += 1
        if block > 7:
            return int(OPM_STEP[767])
    return int(OPM_STEP[eff]) >> (block ^ 7)


def _opm_steps(hz, patch):
    """(keycode, four phase steps) for a pitch: the nearest key code and fraction, then each operator's DT2."""
    target = hz * 2.0 ** 20 / (OPM_CLOCK / 64.0)
    block = int(min(max(math.floor(math.log2(max(target, 1e-9) / float(OPM_STEP[0]))) + 7, 0), 7))
    eff = int(np.argmin(np.abs((OPM_STEP >> (block ^ 7)) - target)))
    ops = (CT.FM_PATCHES[patch] if isinstance(patch, str) else patch)["ops"]
    return (block << 2) | ((eff >> 6) // 3), [_opm_step(block, eff + OPM_DT2[int(op.get("dt2", 0)) & 3]) for op in ops]


def _no_dt2(patch):
    if isinstance(patch, str):
        return patch
    return dict(patch, ops=[{k: v for k, v in op.items() if k != "dt2"} for op in patch.get("ops") or []])


def _fm_lanes(layer, system, names, q):
    lanes, end = lanes_of(layer, system, names, {"patch": "epiano", "gate": 0.95}, q, mono_chord="error")
    return {names.index(n): [(t0, t1, hz, vel, p["patch"], p["pan"]) for t0, t1, hz, vel, p in notes] for n, notes in lanes.items()}, end


def render_opm(lanes, dur, sr, raw=True):
    lanes = {c: [(t0, t1, hz, vel, patch, pan) for t0, t1, hz, vel, patch, pan in notes] for c, notes in lanes.items()}
    steps = lambda hz, patch: _opm_steps(hz, patch)
    # the core's patch rows have no DT2 column: it has already gone into the phase steps
    stripped = {c: [(t0, t1, hz, vel, _Patch(patch), pan) for t0, t1, hz, vel, patch, pan in notes] for c, notes in lanes.items()}
    return CT.render_opn(stripped, dur, sr, clock=OPM_CLOCK, raw_dac=3 if raw else 0, divider=64.0, nch=8,
                         steps_of=lambda hz, patch: steps(hz, patch.full))


class _Patch(dict):
    """A patch as genny.chiptune._fm_patch wants it (no DT2), remembering the full one for the phase steps."""

    def __init__(self, patch):
        full = CT.FM_PATCHES[patch] if isinstance(patch, str) else patch
        super().__init__(_no_dt2(full))
        self.full = full


def _opm_layer(layer: dict, sr: int, q: float) -> np.ndarray:
    lanes, end = _fm_lanes(layer, "ym2151", OPM_NAMES, q)
    y = render_opm(lanes, end + float(layer.get("tail", 0.5)), sr, raw=layer.get("mode") != "retro_stylized") * CT.FM_GAIN
    y = lfilter(*butter(1, 20.0, "high", fs=sr), y, axis=0)
    return y[:, 0] if np.array_equal(y[:, 0], y[:, 1]) else y


CT.SYSTEM_LAYERS["ym2151"] = _opm_layer
CT.SYSTEMS["ym2151"] = OPM_NAMES


@instrument("ym2151", "Yamaha YM2151 (arcade boards, X68000) FM channel: the YM2612's four operators with the chip's own pitch table and a second detune.",
            family="retro", span=("C1", "C7"),
            patch=("epiano", "a patch name (" + " | ".join(sorted(CT.FM_PATCHES)) + ") or the YM2612 patch form; each operator may add dt2 0..3 (0, +600, +781, +950 cents: bells and metal)"),
            tail=(0.6, "seconds rendered after the key goes up, for the release"),
            mode=("retro_stylized", "retro_stylized = the full-resolution operator sum | hardware_accurate = the chip's clipped 16-bit output"))
def ym2151(freq, dur, sr=DEFAULT_SR, vel=1.0, patch="epiano", tail=0.6, mode="retro_stylized"):
    y = render_opm({0: [(0.0, dur, float(freq), 1.0, patch, "C")]}, dur + float(tail), sr, raw=str(mode) == "hardware_accurate")[:, 0]
    return _fade(F.dc_block(y, sr), sr, 0.5, 20.0) * (CT.FM_GAIN * (0.4 + 0.6 * vel))


OPNB_CLOCK = 8000000.0
OPNB_NAMES = ("fm1", "fm2", "fm3", "fm4", "ssg1", "ssg2", "ssg3")
SSG_KEYS = {"level": 15, "noise": 0.0, "noise_period": 8, "shape": -1, "env_hz": 8.0, "gate": 1.0}


def _opnb_layer(layer: dict, sr: int, q: float) -> np.ndarray:
    from . import instruments as I
    from .core import mix
    fm = {k: v for k, v in (layer.get("channels") or {}).items() if k.startswith("fm")}
    ssg = {k: v for k, v in (layer.get("channels") or {}).items() if not k.startswith("fm")}
    lanes, end_fm = _fm_lanes(dict(layer, channels=fm), "ym2610", OPNB_NAMES[:4], q)
    notes, end_ssg = lanes_of(dict(layer, channels=ssg), "ym2610", OPNB_NAMES[4:], SSG_KEYS, q, mono_chord="error")
    dur = max(end_fm, end_ssg) + float(layer.get("tail", 0.5))
    y = CT.render_opn(lanes, dur, sr, clock=OPNB_CLOCK, lfo=layer.get("lfo"), raw_dac=0 if layer.get("mode") == "retro_stylized" else 2,
                      nch=4) * CT.FM_GAIN
    y = lfilter(*butter(1, 20.0, "high", fs=sr), y, axis=0)
    parts = []
    for name, seq in notes.items():                        # the SSG is the AY-3-8910 voice of genny.retro on the chip's clock
        seq = sorted(seq, key=lambda n: n[0])
        for i, (t0, t1, hz, vel, p) in enumerate(seq):
            if i + 1 < len(seq):
                t1 = min(t1, seq[i + 1][0])
            params = {k: p[k] for k in ("level", "noise", "noise_period", "shape", "env_hz")}
            # UNSOURCED: the SSG section is clocked at a quarter of the master clock (2 MHz here)
            parts.append((I.render_note("psg", hz, max(t1 - t0, 0.01), sr, vel, clock=OPNB_CLOCK / 4.0, bright=1.0, **params) * 0.5, t0))
    if parts:
        s = mix(parts, sr)
        n = min(s.shape[0], y.shape[0])
        y[:n, 0] += s[:n]
        y[:n, 1] += s[:n]
    return y[:, 0] if np.array_equal(y[:, 0], y[:, 1]) else y


CT.SYSTEM_LAYERS["ym2610"] = _opnb_layer
CT.SYSTEMS["ym2610"] = OPNB_NAMES


# ============================================================================ MOS 8580 SID (Commodore 64)
# Written from the behaviour documented in reSID (Dag Lem; out/research/chips/resid_*): no code is taken from it.
#   oscillator  24-bit accumulator, the 16-bit frequency register added each clock cycle: f = reg x clock / 2^24.
#               sawtooth = the top 12 bits; triangle = the next 11 bits folded by the top bit; pulse = top 12 bits
#               against the 12-bit pulse width; noise = eight bits of a 23-bit register (taps 22 and 17) that
#               shifts when accumulator bit 19 rises. Hard sync and ring modulation from the voice before.
#   envelope    8-bit counter; attack, decay and release step every `rate period` cycles (sixteen periods, from 2 ms
#               to 8 s for the attack); decay and release slow down at levels 93, 54, 26, 14 and 6 (the chip's
#               piecewise exponential); sustain holds at level s x 17.
#   filter      the 8580's two-integrator filter: cutoff linear in the 11-bit register up to 12.5 kHz, sixteen
#               resonance values (reSID's 8580 table), low-pass, band-pass and high-pass outputs that can be summed.
# Not modelled: combined waveforms (they need the chip's sampled tables), the 6581's non-linear filter and
# its DC offsets, the ADSR delay bug, the external audio input.
SID_CLOCK = {"pal": 985248.0, "ntsc": 1022727.0}             # UNSOURCED here: the C64's CPU clock rates (general knowledge)
SID_RATE = np.array([8, 31, 62, 94, 148, 219, 266, 312, 391, 976, 1953, 3125, 3906, 11719, 19531, 31250], np.int64)
SID_Q = np.array([1448, 1328, 1218, 1117, 1024, 939, 861, 790, 724, 664, 609, 558, 512, 470, 431, 395], np.float64) / 1024.0
SID_WAVES = {"triangle": 0, "tri": 0, "saw": 1, "sawtooth": 1, "pulse": 2, "square": 2, "noise": 3}
SID_NAMES = ("voice1", "voice2", "voice3")
SID_PATCH = {"wave": "pulse", "pw": 0.5, "a": 0, "d": 9, "s": 0, "r": 4, "sync": False, "ring": False, "arp": None, "pitch": None,
             "vib": 0.0, "vib_hz": 6.0, "vib_delay": 0.15, "gate": 1.0}


@nb.njit(cache=True)
def _sid_core(n_out, dec, seg, off, rate_tab, q_tab, cut, frame_cycles, res, mode, routed, clk):
    """Three voices and the filter, one step per CPU cycle. `seg` rows per voice (off[v]..off[v + 1]):
    (cycle, frequency register, pulse width 0..4095, wave 0..3, gate, a, d, s, r, sync | ring << 1).
    `cut[k]` is the cutoff register during driver frame k; `mode` bits are 1 low-pass, 2 band-pass, 4 high-pass;
    `routed` bits say which voices pass through the filter."""
    out = np.empty(n_out, np.float32)
    ptr = off[:3].copy()
    acc = np.zeros(3, np.int64)
    freq = np.zeros(3, np.int64)
    pw = np.full(3, 2048, np.int64)
    wave = np.zeros(3, np.int64)
    gate = np.zeros(3, np.int64)
    adsr = np.zeros((3, 4), np.int64)
    flags = np.zeros(3, np.int64)
    lfsr = np.full(3, 0x7FFFFF, np.int64)
    env = np.zeros(3, np.int64)
    state = np.full(3, 2, np.int64)                        # 0 attack, 1 decay / sustain, 2 release
    rcount = np.zeros(3, np.int64)
    ecount = np.zeros(3, np.int64)
    eper = np.ones(3, np.int64)
    rising = np.zeros(3, np.int64)
    vlp = 0.0
    vbp = 0.0
    big = 1 << 62
    nxt = big
    for v in range(3):
        if ptr[v] < off[v + 1] and seg[ptr[v], 0] < nxt:
            nxt = seg[ptr[v], 0]
    cyc = 0
    dt = 1.0 / clk
    nframes = cut.shape[0]
    for i in range(n_out):
        a_out = 0.0
        for _ in range(dec):
            if cyc >= nxt:
                nxt = big
                for v in range(3):
                    while ptr[v] < off[v + 1] and seg[ptr[v], 0] <= cyc:
                        r = ptr[v]
                        freq[v], pw[v], wave[v] = seg[r, 1], seg[r, 2], seg[r, 3]
                        if seg[r, 4] != gate[v]:
                            gate[v] = seg[r, 4]
                            state[v] = 0 if gate[v] else 2
                            rcount[v] = 0
                        if seg[r, 4]:
                            adsr[v, 0], adsr[v, 1], adsr[v, 2], adsr[v, 3] = seg[r, 5], seg[r, 6], seg[r, 7], seg[r, 8]
                            flags[v] = seg[r, 9]
                        ptr[v] += 1
                    if ptr[v] < off[v + 1] and seg[ptr[v], 0] < nxt:
                        nxt = seg[ptr[v], 0]
            # oscillators
            for v in range(3):
                prev = acc[v]
                acc[v] = (acc[v] + freq[v]) & 0xFFFFFF
                setb = ~prev & acc[v]
                rising[v] = 1 if setb & 0x800000 else 0
                if setb & 0x080000:
                    b0 = ((lfsr[v] >> 22) ^ (lfsr[v] >> 17)) & 1
                    lfsr[v] = ((lfsr[v] << 1) | b0) & 0x7FFFFF
            for v in range(3):                             # hard sync: this voice restarts when the voice before it wraps
                if flags[v] & 1 and rising[(v + 2) % 3]:
                    acc[v] = 0
            vi = 0.0
            direct = 0.0
            for v in range(3):
                # envelope
                period = rate_tab[adsr[v, 0] if state[v] == 0 else (adsr[v, 1] if state[v] == 1 else adsr[v, 3])]
                rcount[v] += 1
                if rcount[v] >= period:
                    rcount[v] = 0
                    if state[v] == 0:
                        env[v] += 1
                        if env[v] >= 0xFF:
                            env[v] = 0xFF
                            state[v] = 1
                    else:
                        ecount[v] += 1
                        e = env[v]
                        eper[v] = 1 if e > 0x5D else (2 if e > 0x36 else (4 if e > 0x1A else (8 if e > 0x0E else (16 if e > 0x06 else 30))))
                        if ecount[v] >= eper[v]:
                            ecount[v] = 0
                            if e > 0 and not (state[v] == 1 and e <= adsr[v, 2] * 17):
                                env[v] = e - 1
                # waveform, 12 bits
                a = acc[v]
                if wave[v] == 1:
                    w = a >> 12
                elif wave[v] == 0:
                    msb = a & 0x800000
                    if flags[v] & 2:                       # ring modulation: the fold follows the other voice's top bit too
                        msb ^= acc[(v + 2) % 3] & 0x800000
                    w = ((~a if msb else a) >> 11) & 0xFFF
                elif wave[v] == 2:
                    w = 0xFFF if (a >> 12) >= pw[v] else 0
                else:
                    s = lfsr[v]
                    w = (((s & 0x100000) >> 9) | ((s & 0x040000) >> 8) | ((s & 0x004000) >> 5) | ((s & 0x000800) >> 3)
                         | ((s & 0x000200) >> 2) | ((s & 0x000020) << 1) | ((s & 0x000004) << 3) | ((s & 0x000001) << 4))
                x = (w - 2048) / 2048.0 * (env[v] / 255.0)
                if routed & (1 << v):
                    vi += x
                else:
                    direct += x
            # the 8580 filter: two integrators
            fc = cut[min(cyc // frame_cycles, nframes - 1)]
            w0 = 6.283185307179586 * 12500.0 * (fc + 1) / 2048.0
            vhp = vbp * q_tab[res] - vlp - vi
            vbp -= w0 * dt * vhp
            vlp -= w0 * dt * vbp
            f = 0.0
            if mode & 1:
                f += vlp
            if mode & 2:
                f += vbp
            if mode & 4:
                f += vhp
            a_out += direct - f                            # the integrators invert: the filtered path comes back negated
            cyc += 1
        out[i] = a_out / dec
    return out


SID_GAIN = 0.45
SID_MODES = {"off": 0, "lp": 1, "bp": 2, "hp": 4, "notch": 5, "lp+bp": 3, "bp+hp": 6}
SID_RELEASE_S = (0.006, 0.024, 0.048, 0.072, 0.114, 0.168, 0.204, 0.240, 0.300, 0.750, 1.5, 2.4, 3.0, 9.0, 15.0, 24.0)


def render_sid(lanes, dur, sr, region="pal", cutoff=1024, res=0, mode="off", routed=(1, 2, 3)):
    """`dur` seconds of a SID. lanes: voice name -> notes (t0, t1, hz, vel, patch). `cutoff` 0..2047 or a list of
    values, one per 1/60 s (the last one held): a filter sweep."""
    clk = SID_CLOCK[region]
    frame = int(round(clk / FRAME_HZ))
    rows, off = [], [0]
    for name in SID_NAMES:
        notes = sorted(lanes.get(name, []), key=lambda n: n[0])
        for i, (t0, t1, hz, vel, p) in enumerate(notes):
            if i + 1 < len(notes):
                t1 = min(t1, notes[i + 1][0] - 2.0 / FRAME_HZ * 0.05)
            c0, c1 = int(round(t0 * clk)), int(round(max(t1, t0) * clk))
            wave = SID_WAVES[str(p["wave"]).lower()]
            fl = int(bool(p["sync"])) | (int(bool(p["ring"])) << 1)
            adsr = [int(min(max(p[k], 0), 15)) for k in ("a", "d", "s", "r")]
            f, last = 0, None
            while True:
                c = c0 + f * frame
                if c >= c1:
                    break
                semis = _tab(p["arp"], f, True) + _tab(p["pitch"], f)
                t = (c - c0) / clk - float(p["vib_delay"])
                if p["vib"] and t > 0:
                    semis += float(p["vib"]) * math.sin(2 * math.pi * float(p["vib_hz"]) * t)
                reg = int(min(max(round(hz * 2.0 ** (semis / 12.0) * 2 ** 24 / clk), 0), 0xFFFF))
                width = _tab(p["pw"], f)
                width = int(min(max(round(width * 4095) if width <= 1.0 else width, 0), 4095))
                row = (c, reg, width, wave, 1, *adsr, fl)
                if row[1:] != last:
                    rows.append(row)
                    last = row[1:]
                f += 1
            rows.append((c1, rows[-1][1], rows[-1][2], wave, 0, *adsr, fl))
        off.append(len(rows))
    seg = np.array(rows, np.int64).reshape(-1, 10)
    cut = np.atleast_1d(np.asarray(cutoff, np.float64)).clip(0, 2047)
    mask = sum(1 << (int(v) - 1) for v in routed) if SID_MODES[mode] else 0
    dec = 16
    x = _sid_core(int(dur * clk / dec) + 1, dec, seg, np.array(off, np.int64), SID_RATE, SID_Q, cut, frame, int(min(max(res, 0), 15)),
                  SID_MODES[mode], mask, clk).astype(np.float64)
    y = _to_sr(x, clk / dec, sr)
    y = lfilter(*butter(1, 16.0, "high", fs=sr), y)        # the C64's output stage: high-pass 16 Hz, low-pass 16 kHz
    if 16000.0 < 0.45 * sr:
        y = lfilter(*butter(1, 16000.0, "low", fs=sr), y)
    return y[:samples(dur, sr)] * SID_GAIN


def _sid_layer(layer: dict, sr: int, q: float) -> np.ndarray:
    lanes, end = lanes_of(layer, "sid", SID_NAMES, SID_PATCH, q)
    flt = layer.get("filter") or {}
    mode = str(flt.get("mode", "off")).lower()
    if mode not in SID_MODES:
        raise SpecError(f"chip layer: sid filter mode must be one of {sorted(SID_MODES)}")
    tail = float(layer.get("tail", 0.5))
    return render_sid(lanes, end + tail, sr, region=str(layer.get("region", "pal")).lower(), cutoff=flt.get("cutoff", 1024),
                      res=int(flt.get("res", 0)), mode=mode, routed=flt.get("voices", (1, 2, 3)))


CT.SYSTEM_LAYERS["sid"] = _sid_layer
CT.SYSTEMS["sid"] = SID_NAMES


@instrument("sid", "MOS 8580 SID voice (Commodore 64): triangle, sawtooth, variable pulse or noise, the chip's ADSR and its resonant filter.",
            family="retro", span=("C1", "C7"),
            wave=("pulse", "triangle | saw | pulse | noise"),
            pw=(0.5, "pulse width 0..1, or a list of widths one per 1/60 s (the last one held): [0.1, 0.15, 0.2, ...] is the SID's moving pulse"),
            a=(0, "attack 0..15: 2 ms .. 8 s"), d=(9, "decay 0..15: 6 ms .. 24 s"), s=(0, "sustain level 0..15"), r=(4, "release 0..15: 6 ms .. 24 s"),
            cutoff=(None, "filter cutoff register 0..2047 (linear to 12.5 kHz), or a list one per 1/60 s for a sweep; null = filter off"),
            res=(8, "filter resonance 0..15"), filter=("lp", "lp | bp | hp | notch | lp+bp | bp+hp"),
            arp=(None, _H["arp"]), pitch=(None, _H["pitch"]), vib=(0.0, _H["vib"]), vib_hz=(6.0, _H["vib_hz"]), vib_delay=(0.15, _H["vib_delay"]),
            mode=CT._MODE, bright=CT._BRIGHT)
def sid(freq, dur, sr=DEFAULT_SR, vel=1.0, wave="pulse", pw=0.5, a=0, d=9, s=0, r=4, cutoff=None, res=8, filter="lp", arp=None, pitch=None,
        vib=0.0, vib_hz=6.0, vib_delay=0.15, mode="retro_stylized", bright=0.5):
    tail = min(SID_RELEASE_S[int(min(max(r, 0), 15))] + 0.02, 1.5)
    p = dict(SID_PATCH, wave=wave, pw=pw, a=a, d=d, s=s, r=r, arp=arp, pitch=pitch, vib=vib, vib_hz=vib_hz, vib_delay=vib_delay)
    y = render_sid({"voice1": [(0.0, dur, float(freq), 1.0, p)]}, dur + tail, sr, cutoff=1024 if cutoff is None else cutoff, res=res,
                   mode="off" if cutoff is None else str(filter).lower(), routed=(1,))
    y = F.dc_block(y, sr)
    if str(mode) != "hardware_accurate":
        if str(wave).lower() != "noise":
            y = _soften(y, freq, sr, harm=6, lo=2200, hi=3800, bright=bright)
        else:
            fc = 3000.0 * 2.0 ** ((bright - 0.5) * 4.0)
            if fc < sr * 0.45:
                y = F.lowpass(F.lowpass(y, fc, sr, 0.707), fc, sr, 0.707)
    return _fade(y.copy(), sr, 1.0, 10.0) * (0.5 + 0.5 * vel)


# ============================================================================ Atari POKEY (400/800, arcade boards)
# Written from the behaviour described in MAME's pokey.cpp (Brad Oliver, Eric Smith, Juergen Buchmueller,
# BSD 3-Clause; polynomial counters after Perry McFarlane and the Altirra hardware manual); no code is taken.
#   Four 8-bit dividers clocked at 63.9211 kHz (clock / 28), 15.6999 kHz (clock / 114) or the full 1.78979 MHz;
#   a divider underflows every AUDF + 1 clocks (AUDF + 4 at the full clock). On each underflow the channel either
#   toggles (a pure tone) or samples one of the chip's free-running polynomial counters (4, 9 or 17 bits), optionally
#   only when the 5-bit counter allows it. Volumes are 4-bit and linear.
# Not modelled: the 16-bit joined channels, the high-pass flip-flops, two-tone mode, volume-only (sample) output.
POKEY_CLOCK = 1789790.0
POKEY_NAMES = ("ch1", "ch2", "ch3", "ch4")
POKEY_DIV = {"64k": 28, "15k": 114, "1.79m": 1}
# AUDC bits 7..5: what an underflow does. Names are genny's; the numbers are the chip's.
POKEY_DIST = {"noise5": 0, "poly5": 1, "buzz5": 2, "noise": 4, "pure": 5, "buzz": 6}
POKEY_PATCH = {"volume": 15, "vol": None, "dist": "pure", "clock": "auto", "arp": None, "pitch": None, "vib": 0.0, "vib_hz": 6.0,
               "vib_delay": 0.15, "gate": 1.0}


def _pokey_polys():
    def p45(size):                                         # lfsr = (lfsr << 1) | ~(bit 2 ^ bit size-1)
        mask, lfsr, out = (1 << size) - 1, 0, []
        for _ in range(mask):
            lfsr = (lfsr << 1) | (~((lfsr >> 2) ^ (lfsr >> (size - 1))) & 1)
            out.append(lfsr & mask & 1)
        return np.array(out, np.int64)
    lfsr, p9 = 0x1FF, []
    for _ in range(0x1FF):                                 # in = bit 0 ^ bit 5, into bit 8
        lfsr = (lfsr >> 1) | ((((lfsr) ^ (lfsr >> 5)) & 1) << 8)
        p9.append(lfsr & 1)
    lfsr, p17 = 0x1FFFF, []
    for _ in range(0x1FFFF):                               # bit 8 ^ bit 13 into bit 7, bit 0 round into bit 16
        in8 = ((lfsr >> 8) ^ (lfsr >> 13)) & 1
        b0 = lfsr & 1
        lfsr >>= 1
        lfsr = (lfsr & 0xFF7F) | (in8 << 7) | (b0 << 16)
        p17.append(lfsr & 1)
    return p45(4), p45(5), np.array(p9, np.int64), np.array(p17, np.int64)


POKEY_P4, POKEY_P5, POKEY_P9, POKEY_P17 = _pokey_polys()


@nb.njit(cache=True)
def _pokey_core(n_out, dec, seg, off, p4, p5, p9, p17, poly9):
    """Four channels, one step per cycle of 1.78979 MHz. Rows: (cycle, AUDF, volume, distortion 0..7, prescaler)."""
    out = np.empty(n_out, np.float32)
    ptr = off[:4].copy()
    audf = np.zeros(4, np.int64)
    vol = np.zeros(4, np.int64)
    dist = np.full(4, 5, np.int64)
    pre = np.full(4, 28, np.int64)
    cnt = np.zeros(4, np.int64)
    prec = np.zeros(4, np.int64)
    bit = np.zeros(4, np.int64)
    i4 = i5 = i9 = i17 = 0
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
                        audf[c], vol[c], dist[c], pre[c] = seg[r, 1], seg[r, 2], seg[r, 3], seg[r, 4]
                        ptr[c] += 1
                    if ptr[c] < off[c + 1] and seg[ptr[c], 0] < nxt:
                        nxt = seg[ptr[c], 0]
            i4 = i4 + 1 if i4 < 14 else 0
            i5 = i5 + 1 if i5 < 30 else 0
            i9 = i9 + 1 if i9 < 510 else 0
            i17 = i17 + 1 if i17 < 131070 else 0
            for c in range(4):
                prec[c] += 1
                if prec[c] < pre[c]:
                    continue
                prec[c] = 0
                cnt[c] -= 1
                if cnt[c] < 0:
                    cnt[c] = audf[c] + (3 if pre[c] == 1 else 0)       # AUDF + 1 clocks, or AUDF + 4 at the full clock
                    d = dist[c]
                    if d & 4 or p5[i5]:                    # bit 7 of AUDC: not gated by the 5-bit counter
                        if d & 1:
                            bit[c] ^= 1
                        elif d & 2:
                            bit[c] = p4[i4]
                        elif poly9:
                            bit[c] = p9[i9]
                        else:
                            bit[c] = p17[i17]
            acc += (vol[0] * bit[0] + vol[1] * bit[1] + vol[2] * bit[2] + vol[3] * bit[3]) / 60.0
            cyc += 1
        out[i] = acc / dec
    return out


POKEY_GAIN = 1.6


def _pokey_clock(hz, clock, channel="ch1"):
    """`auto`: the finest grid that can still reach the note (an 8-bit divider at the full clock stops at 3.5 kHz)."""
    if clock != "auto":
        return clock
    if hz >= POKEY_CLOCK / (2.0 * 259) and channel in ("ch1", "ch3"):
        return "1.79m"
    return "64k" if hz >= POKEY_CLOCK / 28 / (2.0 * 256) else "15k"


def _pokey_audf(hz, clock):
    base = POKEY_CLOCK / POKEY_DIV[clock]
    extra = 4 if clock == "1.79m" else 1
    return int(min(max(round(base / (2.0 * hz) - extra), 0), 255)), base / 2.0, extra


def render_pokey(lanes, dur, sr, poly9=False):
    rows, off = [], [0]
    for name in POKEY_NAMES:
        notes = sorted(lanes.get(name, []), key=lambda n: n[0])
        for i, (t0, t1, hz, vel, p) in enumerate(notes):
            if i + 1 < len(notes):
                t1 = min(t1, notes[i + 1][0])
            clock = _pokey_clock(hz, str(p["clock"]).lower(), name)
            if clock not in POKEY_DIV:
                raise SpecError(f"pokey: clock must be one of {sorted(POKEY_DIV)}")
            if clock == "1.79m" and name not in ("ch1", "ch3"):
                raise SpecError("pokey: only ch1 and ch3 can run at the full 1.79 MHz clock")
            d = p["dist"] if isinstance(p["dist"], int) else POKEY_DIST.get(str(p["dist"]).lower())
            if d is None:
                raise SpecError(f"pokey: dist must be 0..7 or one of {sorted(POKEY_DIST)}")
            c0, c1 = int(round(t0 * POKEY_CLOCK)), int(round(t1 * POKEY_CLOCK))
            if c1 <= c0:
                continue
            mark = len(rows)
            driver_rows(c0, c1, POKEY_CLOCK, hz, vel, p, lambda f, clock=clock: _pokey_audf(f, clock)[0], rows, aux=int(d) & 7)
            for k in range(mark, len(rows)):               # the fifth column is the prescaler, not a reset flag
                rows[k] = rows[k][:4] + (POKEY_DIV[clock],)
        off.append(len(rows))
    seg = np.array(rows, np.int64).reshape(-1, 5)
    x = _pokey_core(int(dur * POKEY_CLOCK / 16) + 1, 16, seg, np.array(off, np.int64), POKEY_P4, POKEY_P5, POKEY_P9, POKEY_P17,
                    1 if poly9 else 0).astype(np.float64)
    y = _to_sr(x - x[0], POKEY_CLOCK / 16.0, sr)
    y = lfilter(*butter(1, 20.0, "high", fs=sr), y)        # UNSOURCED: AC-coupled output; the 20 Hz corner is genny's choice
    return y[:samples(dur, sr)] * POKEY_GAIN


def _pokey_layer(layer: dict, sr: int, q: float) -> np.ndarray:
    lanes, end = lanes_of(layer, "pokey", POKEY_NAMES, POKEY_PATCH, q)
    return render_pokey(lanes, end, sr, poly9=bool(layer.get("poly9", False)))


CT.SYSTEM_LAYERS["pokey"] = _pokey_layer
CT.SYSTEMS["pokey"] = POKEY_NAMES


@instrument("pokey", "Atari POKEY channel: an 8-bit divider that toggles (pure tone) or samples the chip's polynomial counters (buzz, noise).",
            family="retro", span=("C2", "C7"),
            dist=("pure", "pure | buzz (4-bit counter: the Atari bass) | noise (17-bit) | poly5 | buzz5 | noise5 (gated by the 5-bit counter), or the AUDC number 0..7"),
            clock=("auto", "auto (the finest grid that reaches the note) | 64k | 15k (the chip's usual, coarse pitch grids) | 1.79m (above 3.5 kHz; ch1 and ch3 only on the chip)"),
            volume=(15, "0..15, linear"), vol=(None, _H["vol"]), arp=(None, _H["arp"]), pitch=(None, _H["pitch"]), vib=(0.0, _H["vib"]),
            vib_hz=(6.0, _H["vib_hz"]), vib_delay=(0.15, _H["vib_delay"]), mode=CT._MODE, bright=CT._BRIGHT)
def pokey(freq, dur, sr=DEFAULT_SR, vel=1.0, dist="pure", clock="auto", volume=15, vol=None, arp=None, pitch=None, vib=0.0, vib_hz=6.0,
          vib_delay=0.15, mode="retro_stylized", bright=0.5):
    rel = 0.01
    p = dict(POKEY_PATCH, dist=dist, clock=clock, volume=volume, vol=vol, arp=arp, pitch=pitch, vib=vib, vib_hz=vib_hz, vib_delay=vib_delay)
    y = render_pokey({"ch1": [(0.0, dur, float(freq), 1.0, p)]}, dur + rel, sr * 2)
    exact = str(mode) == "hardware_accurate"
    tonal = (p["dist"] if isinstance(p["dist"], int) else POKEY_DIST[str(p["dist"]).lower()]) in (1, 3, 5, 7)
    if not exact and tonal:                                # the 8-bit divider is coarse: play the note back at the written pitch
        audf, half, extra = _pokey_audf(freq, _pokey_clock(freq, str(clock).lower()))
        y = _to_sr(y, sr * 2 * freq / (half / (audf + extra)), sr * 2)
    y = F.dc_block(_to_sr(y, sr * 2, sr)[:samples(dur + rel, sr)], sr)
    if not exact:
        if tonal:
            y = _soften(y, freq, sr, harm=6, lo=2200, hi=3800, bright=bright)
        else:
            fc = 3000.0 * 2.0 ** ((bright - 0.5) * 4.0)
            if fc < sr * 0.45:
                y = F.lowpass(F.lowpass(y, fc, sr, 0.707), fc, sr, 0.707)
    return _fade(y.copy(), sr, 1.0, rel * 1000.0) * (0.5 + 0.5 * vel)
