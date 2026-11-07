# cyber — machines, vehicles, interface cues, signal damage, textures

`genny/cyber.py` registers 8 sound effects and 7 effects for industrial and science-fiction scenes, and
`genny/identify.py` gains `identify_machine` (`genny identify file.wav`), which reads a recording of a
machine and returns the parameters of the model that explains it.

The sound effects take `seed`, work at any sample rate (rendered at 48 kHz, resampled at the output), are
DC-free and start and end at zero. All are mono except `pass_by`, which is stereo unless `pan` is 0.

Sources are named in each table and in `THIRD_PARTY_NOTICES.md`.

## Pick a sound by physics

| Machine | sfx | Governing formula | Source |
|---|---|---|---|
| Transformer core | `transformer` | Strain is not proportional to flux: its waveform holds the even harmonics of the mains, so the hum is at `2·f_mains` and its multiples. A DC bias brings back `f_mains`, `3·f_mains` | Moses et al. 2010; Shilyashki et al. 2016; Hasan 2014 for the half-cycle saturation under a DC bias (the odd lines of the sound are deduced from it, not quoted) |
| Gear pair | `gearbox` | `f_mesh = z·rpm/60`, the tone of the transmission error, loudest where it crosses a structural resonance (0.9–1.2 kHz on a car); an eccentric gear modulates amplitude and speed together, which puts sidebands `rpm/60` apart on every mesh harmonic, with index `β = ε·z` (speed fluctuation × teeth); hunting tooth `f_mesh·gcd(z₁,z₂)/(z₁z₂)` | Tosun, Yildiz & Özkan 2018 (whine, resonances); Randall 1982 (load, wear, sidebands, local faults); hunting tooth: textbook |
| Rolling bearing | `bearing` | With `r = (d/D)·cos φ` and `f_r = rpm/60`: `BPFO = (n/2)·f_r·(1−r)`, `BPFI = (n/2)·f_r·(1+r)`, `BSF = (D/2d)·f_r·(1−r²)`, `FTF = (f_r/2)·(1−r)` | Randall & Antoni 2011, equations 1 to 4; timbre fitted to recordings (below) |
| Four-stroke engine | `pipe_engine` | Firing rate `cylinders·rpm/120`; valves `sin(4πx)` on a quarter of the 720° cycle gate the reflection of an intake and an exhaust waveguide | Baldan et al. 2015 |
| Turbocharger | `turbo` | Whine at the shaft rate with its 2nd and 7th harmonics, level `a²`, `a = min(1, ω/9000)`; hiss band at `3000 + 0.35·ω` Hz | ice-simulator |
| Tyre | `tyre_squeal` | Band of noise at `900 + 700·slip` Hz per wheel | ice-simulator |
| Car on a road | `pass_by` | Rolling `L = a_R + b_R·log₁₀(v/70)`, propulsion `L = a_P + b_P·(v−70)/70` per third octave; Doppler `c/(c∓v)`; ground echo late by `(√(r²+h²)−r)/c` | Harmonoise (Jonasson et al. 2004) |

Handy numbers. A 50 Hz transformer hums at 100, 200, 300 Hz. A 23-tooth gear at 1800 rpm whines at
690 Hz with sidebands at 660 and 720 Hz. An SKF 6205 bearing (9 balls of 7.94 mm on a 39.04 mm pitch
circle) at 1750 rpm ticks 104.6 times a second for an outer-race defect, 157.9 for an inner-race defect
and 137.5 for a ball defect; its cage turns at 11.6 Hz. A four-cylinder at 1800 rpm fires at 60 Hz.

## Machines

| sfx | Parameters (default) | Physical meaning |
|---|---|---|
| `transformer` | `dur` (4), `mains` (50), `flux` (0.8), `dc_bias` (0), `load` (0.3), `loose` (0), `size` (1), `seed` | `flux` is the core induction relative to the saturation knee: 0.5 is a soft 100 Hz hum, 1.2 a harsh buzz rich in 300-600 Hz. `dc_bias` adds the odd multiples (a rougher growl). `load` adds the winding's pure `2f`. `loose` is a lamination or panel rattling once per strain cycle. `size` divides the tank resonances (212, 318, 437, 566, 801 Hz at 1). The mains is off by a constant drawn from the seed. `joints` (0..1): the core regions differ, corners and T-joints straining 2-3 times the limbs (harsher); `stress` (0..1): mechanical stress on the sheets, magnetostriction up 11 dB at 1; `packet` 1..3: the measured induction of one core packet in its five regions (0 = uniform); `grid` DE, US, EE, RU, GB, SE, ZA, ES-PM, ES-GC, IS, FO: the mains wanders as that grid (Ornstein-Uhlenbeck, 1 s increments 1.1 to 9.8 mHz; empty = steady). `tesla` (0; 0.5..2 T of core induction: sets `flux` = tesla / 2 and the level follows the measured noise rise, 0 dB at 1.5 T). |
| `gearbox` | `dur` (3), `rpm` (1800), `rpm_end` (−1), `teeth` (23), `teeth2` (41), `load` (0.6), `wear` (0.3), `eccentricity` (0.3), `backlash` (0.3), `housing` (1050), `seed` | `teeth` × `rpm` / 60 is the pitch. `load` raises the mesh line by 20 dB per decade and its second harmonic by 7 (Randall measured 21 and 7 dB on one gear between 10 % and full load): a lightly loaded gear sounds thin and high, a loaded one sings on its fundamental. `wear` flattens the roll-off of the mesh harmonics (exponent 2.2 → 0.8: wear shows first on the higher harmonics) and adds one damaged tooth, which swells the mesh for one tooth period and knocks once per turn. `eccentricity` sets the sidebands: 50 % of amplitude and a 1 % speed fluctuation at 1, so the index grows with the tooth count (0.23 rad for 23 teeth, capped at 1.5). `backlash` rattles only when `load` is low. `housing` is the first casing mode; the next are at 1.42, 2.05 and 2.9 times it; `car` = the modes 935 / 1020 / 1140 Hz decaying at 94 / 298 / 346 /s. `mount_mass` (kg) adds blocking masses on the mounts: the mesh and everything above drop as a one-pole low-pass at Z/(2πm), Z = 8√(Bρh) the point impedance of a 5 mm steel sheet (391 Hz at 1 kg). `rpm_end` glides the speed. |
| `bearing` | `dur` (3), `rpm` (1750), `rpm_end` (−1), `state` (healthy), `severity` (0.6), `balls` (9), `ball_mm` (7.94), `pitch_mm` (39.04), `resonance` (1000), `seed` | `state`: `healthy` (rolling noise and frame rumble), `worn` (more rumble and random ticks), `dry` (a squeal that breaks up once per turn), `outer` (a soft, even ticking at BPFO), `inner` (ticking at BPFI that swells once per turn, as the defect passes through the load), `ball` (bursts of ticking once per cage turn, while the damaged ball crosses the load zone; the strikes inside a burst come at twice BSF but slip 3 % each, so they hold no steady rate), `loose` (a low rumble that swells every turn and every other turn), `misaligned` (a `2×` shaft tone). `severity` sets the impacts against the rolling noise: about equal at 0.6, which is what the recordings below show; 14 dB over at 1, a far-gone bearing that ticks clearly. `resonance` is the main housing mode (up to 19000 Hz for an accelerometer's view); the others are at 0.7, 2 and 4.1 times it and all ring about 7 ms (`ring` multiplies that: 3-5 for a slow crane bearing). More states: `pitting` (uneven ticks at BPFO), `crack` (sharp inner-race clicks), `spall` (no sharp shocks: level following the shaft harmonics, noise swelling in the load zone), `polished` (the same with a small shock as each ball leaves), `cage` (irregular knocks at the cage rate). `size` 6004, 6205, 6203, 6206, 51104 (thrust, angle 90°), 51104_alt (rollers set alternately: BPF/2); `BEARING_LIMITS` holds the catalogue speed limits, open / sealed. `load` 0..1: Stribeck load zone (0 = the old cos² zone). `rotating=outer`: the outer ring turns (planet gear, wheel hub). `defect` (degrees of race): an entry and an exit shock. `mirror=1`: heard through a 90 mm parabolic mirror. `rpm` goes down to 1. |

### `bearing` against recordings

`bearing` was fitted to the FSTF sound dataset (Mendeley n9y9c7xrz3): an SKF 6004 with an inner-race,
outer-race or ball defect, loose, or healthy, at about 640, 1230 and 1810 rpm, 20 s each, recorded with
a microphone through a stethoscope. Each recording was compared with a render of the same state and
speed at the default `severity`. Ranges are over the three speeds.

| Measure | Recordings | `bearing` before | `bearing` now |
|---|---|---|---|
| Spectral centroid, faulty states | 0.6–1.6 kHz | 2.6 kHz | 0.7–1.3 kHz |
| Energy above 1.5 kHz, inner / outer / ball | 7–56 % | 98 % | 26–32 % |
| 250 Hz octave against the total | −7 to −21 dB | −38 dB | −8 to −18 dB |
| Spectral peaks | 0.7, 1.0, 1.4–2.1, 4.1 kHz | 2.4, 3.3, 4.7 kHz | 0.7, 1.0, 2.0, 4.1 kHz |
| Ring time after an impact (inner) | 6–8 ms | 1.9 ms | 5.7–5.8 ms |
| Excess kurtosis, inner | 4.9–6.6 | 21–69 | 2.4–9.0 |
| Excess kurtosis, outer | 0.1–0.3 | 6–25 | 0.0–0.2 |
| Excess kurtosis, ball | 1.1–3.7 | 11–39 | 1.4–2.8 |
| Excess kurtosis, loose | 0.0–0.2 | 56–178 | 1.2–1.3 |
| Envelope line at BPFI over the floor, inner | 31–38 dB | 42–52 dB | 33–36 dB |
| Envelope line at BPFO, outer | 12–27 dB | 33–44 dB | 20–25 dB |
| Envelope line at 2·BSF, ball | 14–15 dB | 37–49 dB | 17–26 dB |
| Envelope line at twice the cage rate, ball | 11–32 dB | not measured | 26–28 dB |
| Envelope line at BPFO, healthy | 11–19 dB | 24–26 dB | 14–19 dB |

The "now" column was measured again after the corrections taken from Randall & Antoni (2011): slip
that adds up from one impact to the next instead of a jitter about a fixed period, and a ball defect
gated once per cage turn. The lines of the model came down by 3 to 8 dB, towards the recordings.

What is still off: the strike rate of the ball defect is 2 to 11 dB too clear, and the kurtosis of the
inner-race defect falls with speed faster than in the recordings. The comparison is with one test rig
and one microphone, so part of what was fitted (the rumble under 500 Hz, the mode frequencies) belongs
to that rig. The 6004 geometry (9 balls of 6.35 mm, pitch circle 31.0 mm) is confirmed by two makers'
data sheets (GMN, NSK); the measured inner-race order, 5.39 to 5.45, agrees with its 5.42. The "before"
column was measured at the nominal speeds of the open-microphone recordings.

Two more sets of recordings were looked at, through `identify_machine` only (section "Limits" below):
the AHU parabolic-mirror set (a cylindrical roller bearing NU407, 11 rollers of 15 mm on a 68.5 mm pitch
circle, 600 to 1200 rpm, 58 files of 25 s) and a sample of SUBF v2.0 (24 files of 10 s). In AHU the
inner-race line sits at 6.78 to 6.85 orders against 6.70 from the geometry (1.1 to 2.1 % high, the
"1 to 2 %" of the tutorial), with sidebands one shaft rate away; the outer-race line at 4.27 to 4.31
against 4.30, with harmonics and no sidebands; the roller defect shows the cage rate, 0.39, and a line
at 4.1 to 4.3 with the cage rate on both sides. These are the three patterns `bearing` produces. SUBF
has no geometry and its files are dominated by the motor's hum (centroid 75 Hz, envelope lines on the
multiples of the supply): nothing in it could be compared.

`gear_frequencies(rpm, teeth, teeth2)` and `bearing_frequencies(rpm, balls, ball_mm, pitch_mm, angle_deg)`
return the rates in Hz; `magnetostriction(b)` is the strain curve.

```json
{"layers": [{"type": "sfx", "kind": "transformer", "params": {"dur": 6, "flux": 1.1, "loose": 0.3}},
            {"type": "sfx", "kind": "bearing", "params": {"dur": 6, "state": "dry", "rpm": 2800}, "gain": 0.3}],
 "fx": [{"type": "room", "preset": "hall", "mix": 0.25}]}
```

## Engine and road

| sfx | Parameters (default) | Physical meaning |
|---|---|---|
| `pipe_engine` | `dur` (3), `rpm` (1800), `rpm_end` (−1), `cylinders` (4), `load` (0.5), `header` (1.6), `pipe` (6), `lope` (0), `unequal` (0.15), `mic` (0.3), `backfire` (0), `seed` | `header` is the exhaust runner length in ms of travel (1 ms is 34 cm): short rasps, long is deep. `pipe` is the straight pipe before the four-chamber muffler: its round trip is the drone. `lope` makes the odd cylinders fire early, which brings out the half firing order (V-twin, cammed idle). `unequal` spreads the header lengths (burble). `load` strengthens and shortens the combustion pulse and opens the intake roar. `mic` moves from tailpipe (0) to air intake (1). `backfire` is the chance of a pop in the exhaust per engine cycle while the rpm falls (`rpm_end` below `rpm`); each pop squares it, so they thin out. `transients` (0..1, default 0): a short burst above 4 kHz on every firing, heard while the firing rate stays under 80 Hz, gone by 160 Hz. |
| `turbo` | `dur` (2.5), `spool` (90000), `boost` (0.7), `blowoff` (1), `flutter` (0), `seed` | `spool` is the peak shaft rpm (90 000 rpm whines at 1.5 kHz). `blowoff` ends on the valve's pshh at `1400 + 5500·boost` Hz. `flutter` chops it at 38 Hz (compressor surge). |
| `tyre_squeal` | `dur` (1.5), `slip` (0.8), `wobble` (0.5), `seed` | `slip` 0.2 is a chirp, 1.5 a full slide: it raises the pitch (four wheels 220 Hz apart) and narrows the band. |
| `pass_by` | `dur` (5), `speed` (60), `distance` (6), `power` (combustion), `cylinders` (4), `gear` (3.2), `wet` (0), `direction` (lr), `pan` (1), `seed` | `speed` in km/h sets the tyre spectrum, the wind (`∝ v^2.5`), the engine speed and the Doppler drop. `power`: `combustion` (a `pipe_engine` at `gear × 26 × km/h` rpm), `electric` (a 9:1 reduction gear whine and inverter tone, 12 dB lower) or `none`. The engine is scaled to the Harmonoise propulsion energy for that speed, so tyres win as speed rises. `distance` is the closest approach. `sources=2`: rolling noise mostly at the tyre contact (0.01 m), engine at 0.3 m, each with its own ground path. |
| `traffic` | `dur` (20), `rate` (600 per hour), `speed` (70), `distance` (8), `electric` (0.2), `seed` | Road traffic flow: Poisson arrivals in both directions (far lane 3.5 m further), a 2 s minimum gap per lane, each car a 6 s `pass_by` at 0.85-1.1 of the speed limit. Stereo. |
| `induction_motor` | `dur` (3), `mains` (50), `poles` (4), `load` (0.6), `bars` (28), `fault` (none), `severity` (0.5), `seed` | Slip s = 0.005 + 0.045·load, shaft fr = 2f(1−s)/p. Hum at 2f and 4f, rotor-slot whine at bars·fr with ±2f sidebands, shaft orders. `fault` broken_bar (sidebands f(1±2s), hum beating at 2sf), eccentric (slot line ±fr), bearing (outer-race ticking). |
| `machine_hum` | `dur` (2), `size` (120 Hz), `work` (0.5), `speed` (2 Hz), `mod_freq` (30 Hz), `seed` | Gaver's FM machine (auditory icon): pitch = size, wobble depth = how hard it works, wobble rate = how fast. |

`road_noise_levels(kmh)` returns the rolling and propulsion levels per third-octave band (20 Hz–12.5 kHz);
`_third_octave_noise(..., freqs=)` takes other band centres.

`pipe_engine` also takes: `extractor` (ms of extractor runner per header), `cylinder` (ms: the cylinder as a short
guide between the valves, about 0.2), `timing` (`"intake,exhaust"` cam phasing in fractions of a cycle), `jitter`
(each cylinder's firing wanders), `warp` (the pipe stretches with each firing), `block` (x on the block vibration:
3-5 from inside the car), `radiate=1` (tailpipe as a monopole, brighter), `units` ms | m with `gas_temp` (K, 600),
`side` (Hz of side-branch resonators that cancel their frequency, e.g. `"110,857"`), `expansion` / `expansion_ms`
(expansion chamber area ratio and length), `outlet` / `outlet_r` (tail outlet after the muffler and its end
reflection), `alternator`, `roots` (supercharger whine), `view` tail | cabin, `load="auto"` (0.3 at steady rpm, full
at 3000 rpm/s), `crank` (shared crank flutter), `norm4`, `intake_gain`, `outlet_gain`, `ends` engine | paper.
`engine_rpm(throttle, rate)` turns a throttle curve into an rpm curve for `rpm=`. Duct acoustics: `gas_speed`,
`expansion_chamber_ir`, `plane_wave_cutoff`, `bend_cents`, `step_end_correction`, `side_branch_freq`,
`helmholtz_neck`, `twin_power_gain`, `polack_ir`, `steepening`. `ui_category(dur, attack, sweep_st, noise_s, tonal,
peak_hz, ...)` names the interface category a sound's features fit.
The Harmonoise tables were one third octave low (and had two typos, cars at 40 Hz and trucks at 100 Hz); they
are corrected, which moves the road spectrum of every sound built on them (`pass_by`, `tyre_roll`,
`truck kind=truck_passby`, `bus`...) up one third octave. In `heavy_vehicles`: `cv_noise_levels(kmh, axles=4)` gives a truck's rolling and propulsion levels
(`axles` 2 = the medium-heavy category with its own propulsion; rolling + 10 lg(axles/4));
`moto_noise_levels(kmh)` a motorcycle's propulsion per octave 63 Hz–8 kHz (category 4b, CNOSSOS-EU), which
`motorcycle kind=passby` follows: `moto_propulsion(y, kmh, passes=4)` brings the engine's octave bands to that
profile, total energy kept. `chain_rate(rpm)` = 14 × rpm / 60 is the link rate that sets the `chainsaw` chain band
(a 7-tooth sprocket, about 2.2 kHz at 9400 rpm); `garden_machine kind=debroussailleuse` turns its head at the
crank speed / `TRIMMER_GEAR` (a 1.43:1 bevel).

`tyre_lateral_force(alpha, load=4000, mu=1.0, stiffness=60e3)`: the side force of a cornering tyre at slip angle
`alpha` (rad), Fiala's brush model with a parabolic pressure: C·α at small angles, saturating at μ·load.
`TYRE_MU` = (peak, sliding) grip, (1.0, 0.7) on dry asphalt.

`pipe_engine` follows Baldan, Lachambre, Delle Monache & Boussard (2015) with two parts of their model left
out: the cylinder as a waveguide whose length follows the piston (so there is no cylinder volume or
compression setting) and the outlet pipe after the muffler. Like theirs, it has less energy in the
medium-high frequencies than a recorded engine.

The `doppler` effect (see [`reverbs.md`](reverbs.md)) has two new parameters: `ground` (0..1, 0.35 for
asphalt) adds the phase-inverted reflection off the ground, a comb whose notches sweep down as the source
approaches, and `height` (1.8) is source height plus listener height in metres.

For a street: render several `pass_by` with different `seed`, `speed`, `power` and `direction` and place
them with `at`.

### Added since 1.2.1

- `gearbox friction` (flank friction: a sawtooth at the mesh rate changing sign at the pitch point);
  `transformer strip` (a lamination strip ringing near 1.5 kHz); `bearing size` (a designation such as 6004),
  `angle` (contact angle), `creep` (ball rates 1-2 % slow); `tyre_squeal model=brush` (Johnson's contact: a
  gripping tyre is 30 times quieter than a sliding one).
- `pipe_engine`: up to 16 `cylinders`, `cycle` 4 | 2 | rotary, `offsets` (firing times per cylinder), `banks` (two
  collectors and two pipes), `headers` (runner length per slot), `muffler`; `rpm` and `load` take curves.
  `cyber.rpm_curve` gives a curve from a torque model. Up to 12 cylinders it renders as before.
- `pass_by`: ground effect over asphalt and tyre-horn directivity instead of a fixed ground term (300-800 Hz
  quieter on tyre noise), and the wind noise fades in between 15 and 26 m/s: at 60 km/h it is about 6 % of its old
  level. Both change the default sound.
- `gearbox housing` also takes a list of peaks (`"935,1020,1140"`) or `car` instead of one first mode
  (`gear_housing`).
- Python helpers: `transformer_leq_db` (core noise level against induction, from measured points),
  `fluctuation_rel` (fluctuation strength against modulation rate, 1 at 4 Hz), `fluctuation_bbn` (fluctuation
  strength of amplitude-modulated broadband noise, vacil), `autocorr_zcr` (frame autocorrelation and zero-crossing
  descriptors), `soft_limit` (identity up to the knee, then a smooth approach to the ceiling).
- Whole engines, vehicles and machines now have their own modules: `engines`, `heavy_vehicles`, `road`, `rail`,
  `machines` (`catalog.md`).

## Interface cues

`cyber_ui` takes `kind`, `pitch` (1.0, transposes) and `seed` (every seed is another cue of the family).

| kind | Figure | Length |
|---|---|---|
| `scan` | A band of noise sweeping up 4–7 times in frequency under a gated square tone | 0.3–0.4 s |
| `confirm` | Root, fifth, octave, with 2:1 FM | 0.5 s |
| `deny` | The same low square note three times over a note a semitone above | 0.2 s |
| `hack` | 10–16 square blips on a pentatonic set at 20–40 per second, one in ten dropped, 6 bits | 0.5 s |
| `decrypt` | The same blips slowing by 15 % each, settling on an octave | 1–2 s |
| `upload` | 6–8 steps rising by 3, 4 or 5 semitones over rising noise | 0.5 s |
| `warning` | A minor or major third falling to the root, twice | 0.5 s |
| `target_lock` | Ticks closing in (each gap 0.7 of the last) onto a held tone | 0.6 s |
| `implant` | A click, a 60–120 Hz thump falling an octave, a faint FM ping | 0.2 s |
| `neural_link` | Two pairs of sines detuned 5–14 cents swelling in, with a copy shifted 40 Hz | 0.8 s |

Sharpness stays under 2.0 and energy above 5 kHz under 3 % for every kind and seed (tested).

These cues are tones arranged by convention (rising for success, low and repeated for refusal): what
Gaver ("Auditory icons: using sound in computer interfaces", 1986; "The SonicFinder", 1989;
"Synthesizing auditory icons", 1993) calls a symbolic mapping: arbitrary, learned by convention. His
auditory icons are the other way to do it, a nomic mapping: the sound of a physical event whose
attributes carry the information (size as pitch, type as material, progress as a vessel filling),
which he expects to be easier to learn. genny has those events already: `ui_tick`, `impact`, `scrape`,
`bounce`, `pour`, `ratchet`, `switch`. A recorded icon played the same way every time annoys and tells
little; one sound per `seed` and per parameter value is the parametric icon of Brazil & Fernström
(The Sonification Handbook, chapter 13, 2011), who also pass on Mynatt's first rule: short sounds with
a wide bandwidth.

## Effects

| effect | Parameters (default) | What it does |
|---|---|---|
| `packet_loss` | `loss` (0.1), `burst` (2), `packet_ms` (20), `conceal` (repeat), `seed` | A two-state (Gilbert) channel loses `loss` of the packets, `burst` in a row on average. A lost packet is `mute`, the previous packet at 0.7 of its level (`repeat`, so a burst stutters and fades), or its spectrum with random phase (`noise`, a frozen smear). `channel` good, fair or poor (GSM C/I 10, 7, 4 dB) replaces `loss`: frames erased 1, 15, 95 in 750 and bit errors in the samples. |
| `car_cabin` | — | Heard from the driver's seat: body panels low-pass (−6 dB/oct above 150 Hz) and the cabin's first air mode booms at 85 Hz. |
| `beat_repeat` | `grid` (0.125), `chance` (0.3), `repeats` (3), `divide` (1), `decay` (0.85), `reverse` (0.2), `seed` | With probability `chance` a `grid`-long slice is caught and played `repeats` times in place of what follows. `divide` 2, 4, 8 plays only the first part of the slice that many times (a roll). The length does not change. |
| `paulstretch` | `stretch` (8), `window` (0.25), `hold` (−1), `length` (4), `smear` (0), `drift` (0), `seed` | Random-phase stretch up to 1000 times: any sound becomes a slow wash. With `hold` ≥ 0 it freezes that instant for `length` seconds instead; `smear` blurs the held spectrum across frequency. Unlike `freeze`, the result has no phase coherence: a pad, not a held note. `drift` 0..1: the read point wanders round its place (three golden-ratio sines, up to 30 % of the source), a slowly changing wash. The `shimmer` and `shimmer_deep` recipes now take their octave layers by resampling. |
| `waveset` | `repeats` (2), `keep_length` (true) | Each cycle between upward zero crossings is played `repeats` times and the next ones skipped. |
| `formant_shift` | `semitones` (−3) | Moves the spectral envelope and leaves the pitch. |
| `cyber_voice` | `formant` (−3), `pitch` (110), `vocode` (0.6), `ring` (70), `ring_mix` (0.25), `loss` (0.06), `conceal` (noise), `radio` (1), `seed` | `formant_shift` → `vocoder` (16 bands, saw carrier at `pitch`) mixed by `vocode` → `ringmod` → `packet_loss` on 30 ms packets → `telephone`. Set a stage's amount to 0 to skip it. The output has the input's RMS. |
| `texturize` | `length` (8), `iters` (12), `seed` | Analyses the input as a texture and synthesises `length` seconds (up to 60) of a new one that loops without a seam. |
| `adaptive_fm` | `freq` (220), `index` (2), `ratio` (1), `mix` (1) | The sound reads itself through a delay swinging at fm = `freq` / `ratio`, which frequency-modulates everything in it: FM on a real input (bright, metallic). `freq` is the note the depth is set for, `index` 0..5, `ratio` 0.1..5 (an integer = harmonic). |
| `grain_delay` | `delay` (0.25 s), `delay_spread` (0.15), `size` (0.08 s), `size_spread` (0.04), `density` (20/s), `pan_spread` (0.6), `level_spread` (0.4), `mix` (0.5), `seed` | Granular delay: raised-cosine grains read from the past at a random delay, size, pan and level (each a centre plus a random spread): a scattered, glittering echo. Stereo out. |

```json
{"layers": [{"type": "speech", "text": "access granted", "params": {"voice": "female"}}],
 "fx": [{"type": "cyber_voice", "pitch": "A2", "loss": 0.1}, {"type": "reverb", "mix": 0.15}]}
```

### Textures

`texture_stats(x, sr)` and `texture_synth(stats, dur, seed, iters)` are the two halves of `texturize`.
The model is McDermott & Simoncelli (2011): 32 half-cosine bands on the ERB scale (52 Hz–8.8 kHz plus a
low and a high end band), Hilbert envelopes compressed with the exponent 0.3 and sampled at 400 Hz.

Statistics kept: the distribution of each band's envelope (64 quantiles), the correlation between every
pair of bands, the modulation power of each band in 9 octave bands from 0.5 Hz, the C1 correlations (each
octave modulation band from 2 to 128 Hz across all cochlear bands, wider than the paper's two nearest
neighbours over 3-100 Hz) and the C2 correlations (each frequency-doubled modulation band against the next
octave up, 1-128 Hz: the phase that tells an abrupt onset from an offset). They are imposed by projection
instead of the paper's conjugate gradient. `target_snr` stops
early once every class is matched that well (`genny.calibrate.texture_snr`; the paper uses 30 dB). Consequence: dense noisy
textures (rain, streams, insects, static, a room full of fans, a crowd) come out well; textures made of
sharp one-sided events (fire crackle, applause) keep their onset asymmetry through C2 but are still softer
than the original; pitched or rhythmic material does not work, in this version or in the paper's.

Realism ratings from the paper's listening test (Table S1, scale 1-7) show what to expect:

| best | rating | worst | rating |
|---|---|---|---|
| insects | 6.57 | railroad crossing | 1.93 |
| heavy rain | 6.57 | wind chimes | 1.77 |
| frogs | 6.53 | corkscrew | 1.77 |
| applause | 6.47 | gravel | 1.63 |
| stream | 6.43 | drum loops | 1.50-1.60 |
| frying eggs | 6.37 | church bells | 1.37 |
| wind | 6.33 | speech | 1.20-1.40 |
| forest fire | 6.00 | | |
| steam train | 5.87 | | |

A 4 s texture at 48 kHz takes about 1 s per iteration.

## Identifying a machine from a recording

```
genny identify hum.wav
genny identify gearbox.wav --rpm 1500
```

```python
from genny.identify import identify_machine
r = identify_machine(x, sr)          # {"kind", "sfx", "params", "evidence"}
y = genny.sfx.render_sfx(r["sfx"], sr, **r["params"])
```

| kind | Rule | Parameters returned |
|---|---|---|
| `transformer` | Over 90 % of the tonal power on multiples of 50 or 60 Hz, more on the even ones than the odd | `mains`, `dc_bias` (from the level of `f` against `2f`) |
| `bearing` | The squared envelope of the band above 1.5 kHz has a line, with its double, that is not an order, half, third or quarter order of the shaft nor a multiple of 50 or 60 Hz; the band is impulsive (excess kurtosis over 2) or the line has a third harmonic. Failing that: an impulsive band with a line at 0.34 to 0.46 of the shaft rate and its multiples (the cage) | `rpm`, `state`: `outer` if the line's double stands 12 dB above its sidebands, else `inner` if the shaft-rate pair of sidebands is the stronger and `ball` if the cage-rate pair is; `ball` for the cage rule. `balls`, except for `ball` |
| `gearbox` | A line above 150 Hz with a sideband on each side whose spacing divides it (spacings ranked by their stronger side: the two sides are rarely equal); the mesh order is the shaft harmonic that stands 10 dB out of its neighbours | `rpm`, `teeth`, `teeth2` when a second sideband family is present |
| `engine` | A harmonic series under 600 Hz; cylinders from the weaker lines at multiples of the cycle rate between its harmonics | `rpm`, `cylinders` (`evidence.cylinders_assumed` is true when no such lines were found and 4 was assumed) |

`evidence` holds what the decision used: the strongest lines, the shaft rate, the envelope lines, the fault
order, the mesh frequency. `spectral_lines(x, sr)` returns the tonal lines of any steady sound.

Separating the periodic part from the rest, before an envelope analysis (`genny.identify`):
`tsa(x, period, n)` (time synchronous average over `n` periods, fractional period allowed),
`drs(x, delay, nfft=4096)` (discrete random separation, returns deterministic and random parts),
`sanc(x, delay, order, mu=0.01)` (self-adaptive noise cancellation, normalized LMS),
`ar_residual(x, order=40)` (linear-prediction whitening: lines go, impulses stay),
`med(x, order=30, iters=30)` (Wiggins minimum entropy deconvolution: smeared impulses come back sharp),
`kurtosis4(x)` (3 for Gaussian noise, large for sparse impulses). `material_from_modes(freqs, decays)` ranks the
`contact.MATERIALS` by how well the modes' f/d fits each one's measured distribution.

What the tests measure on genny's own renders (5 s): mains frequency exact; gear `rpm` within 0.2 % and
both tooth counts exact on five gearboxes including two worn ones, one of them rattling; bearing `rpm` within 0.2 %, the
fault type right for outer, inner and ball defects (the ball defect on a 20 s render: its cage rate needs
a few hundred cage turns), 9 balls found for the 6205; a four-cylinder `pipe_engine` at 1800 rpm found
as 4 cylinders.

Limits, all observed:

- The bearing rules were set on genny's renders and on two sets of real recordings, with `rpm` given.
  They find a fault more often than they name it:

  | Recordings | Inner race | Outer race | Ball or roller | Healthy, loose |
  |---|---|---|---|---|
  | FSTF, ball bearing 6004, 36 files of 20 s | found 6 of 6, named 4 | found 0 of 6 | found and named 5 of 6 | 0 of 12 called a bearing fault; 3 called an engine |
  | AHU, roller bearing NU407, 58 files of 25 s | found 15 of 17, named 10 | found 18 of 18, named 14 | found 16 of 17, named 15 | 1 of 6 healthy files called a ball defect |
  | SUBF v2.0, 24 files of 10 s | found 0 of 8 | found 0 of 8 | none in the set | 0 of 8 called a bearing fault; all 24 files called an engine |

  The outer-race defect of FSTF is not impulsive and has no line: nothing to find. The combined-fault
  bearing of FSTF is found in 6 of 6 files. A healthy roller bearing under load is modulated at its cage
  rate like a damaged roller, hence the false alarm in AHU. In SUBF the motor's hum covers
  everything; before multiples of the supply were excluded, all 24 files, healthy ones included, were
  called an outer-race defect at 100 Hz. Renders of `bearing` at the default severity: `inner` is found,
  `outer` and `ball` only at severity 1.
- A fault whose rate falls within 1 % of a multiple of 50 or 60 Hz is not seen (that band is given up to
  motor hum).
- One steady speed. A run-up must be cut into steady pieces first.
- A machine whose strongest lines fall on multiples of 100 or 120 Hz is called a transformer (a six-cylinder
  at 2400 rpm fires at 120 Hz). Pass `rpm` to skip that test.
- The ball count assumes a ball-to-pitch diameter ratio of 0.2 and is off when the strongest envelope line
  is a sideband: an 8-ball 6200 was reported as 7, a light inner-race defect at 3000 rpm as 7 instead of 9.
- A healthy or evenly worn bearing has no periodic line and is `unknown`.
- Engines other than an even-firing four are unreliable: a loping V8 is `unknown`, a two-cylinder at full
  load was called a bearing, and `mains_hum` (whose strongest line is the mains frequency itself) a gearbox.
- No fan, electric motor or turbine rules.

## Numbers the tests check

`tests/test_cyber.py`, 72 tests.

| Prediction | Measured |
|---|---|
| Transformer lines at odd multiples of the mains, no DC bias | more than 60 dB under the `2f` line |
| The same with `dc_bias` 0.3 | `f` within 20 dB of `2f` |
| Gear sidebands at `f_mesh ± rpm/60`, eccentricity 0.5 | present, under the carrier, 10 dB over the floor; absent (−60 dB) at eccentricity 0 |
| Gear mesh line against its double, load 0.1 → 1 | gains 9 dB (14 dB at the source, less once the housing is mixed in) |
| Slip of 2 % per impact after 20 s | adds up to over 0.1 period for a bearing; stays under 0.03 for gear teeth |
| A band pulsing at 100 Hz beside a 24 Hz shaft | not called a bearing fault |
| A band pulsing at 0.4 of the shaft rate | called a ball defect, cage order 0.40 |
| 6205 bearing orders | 3.5848, 5.4152, 2.3567, 0.3983; BPFO + BPFI = 9 |
| Envelope spectrum of `outer`, `inner`, `ball` | a line at BPFO, BPFI, 2·BSF over 8 times the median |
| `pipe_engine` strongest low line | `cylinders·rpm/120` within 1 % |
| Harmonoise rolling level at 1 kHz | 90.6 dB at 70 km/h, +29.1 dB per decade of speed |
| `doppler` ground echo, 4 m away, 1.8 m heights | notch at `1/delay`, boost at `0.5/delay` |
| `packet_loss` 20 % with bursts of 3 | loss 0.2 ± 0.04, mean run 3 ± 0.6 |
| `paulstretch` × 4 of a 140 Hz tone | 4 times the length, peak still at 140 Hz |
| `formant_shift` ± 7 semitones | spectral centroid up or down by over 20 %, partials still on 110 Hz |
| Texture filter bank | squared responses sum to 1 |
| Texture synthesis | band levels correlate over 0.98 with the target; a 6 Hz modulation is reproduced within 0.15 of its share |

## Constants without a source

Each is justified in `research/SOURCES.md` (the markers are gone from the code).

- `transformer`: the strain curve `b²·exp(0.9·b²)` (its lines fall in the order Shilyashki et al. report for a model core:
  100 Hz strongest, then 200 and 300 Hz, little above; the levels were not fitted to their figure), the tank mode
  frequencies, the rattle law. The mains frequency is off by a constant drawn from the seed (sd 3.8e-4, 18.8 mHz
  at 50 Hz; a real grid holds still over a few seconds), or wanders as the chosen `grid`. Taking the radiated sound as the strain velocity
  follows the remark in Moses et al. that noise relates to vibration velocity.
- `gearbox`: the harmonic roll-off exponents, the housing mode ratios and damping (the first mode is put
  in the range Tosun et al. measured), the size of the run-out modulation (1 % of speed and 50 % of
  amplitude at eccentricity 1), the load law beyond the second harmonic, the strength of the damaged tooth.
- `bearing`: the `worn`, `dry` and `misaligned` states; 3 % of slip per strike for a ball defect (the tutorial says 1 to 2 %). The housing modes, ring time and impact levels are
  fitted to one dataset (see "`bearing` against recordings").
- `pipe_engine`: the `lope` and `unequal` laws; ignition time 0.069 of a cycle at no load falling to 0.016;
  the level of a backfire pop.
- `turbo`: the spool curve, the mass-flow scale, the 38 Hz flutter.
- `pass_by`: the wet-road spray and the whole electric drive.
- `cyber_ui`: the ten figures. Their note ranges, gaps and envelope law come from `procedural-sounds`.
- `packet_loss`: the 0.7 attenuation per concealed packet.
- `identify_machine`: every threshold (the 12 dB between a line's double and its sidebands was set on the AHU and FSTF recordings).
