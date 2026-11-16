# acoustic — physical strings, keys and winds (`genny/acoustic.py`, `strings.py`, `piano.py`)

Every name and parameter is in [`catalog.md`](catalog.md) ("Index by module"); this page covers the newer string
instruments, the options added to the existing ones and the Python helpers behind them. The models are listed in
[`physics.md`](physics.md).

## New string instruments

| name | span | what it is | params |
|---|---|---|---|
| `kantele` | D3-D5 | Finnish zither: steel strings knotted to a tuning bar; the knot's tension modulation adds a second harmonic that swells after the pluck | `knot` 1 (0..2 x the second-order nonlinearity) |
| `string_scrape` | E2-E4 | finger or pick sliding along a wound string: a picket fence of clicks, one per winding turn, pitched at speed / winding pitch | `speed` 0.4 m/s, `pitch_mm` 0.4 |
| `guitar_chord` | E2-E5 | six strings coupled at one bridge: the open strings ring in sympathy, or a strummed shape whose strings feed each other | `chord` none / maj / min / power, `strum` 0.015 s, `coupling` 1 |
| `tanpura` | C2-C4 | four long strings droning Pa Sa Sa Sa over a jawari bridge, plucked in turn | `jawari` 0.6, `cycle` 2.4 s, `first` pa / ma / ni |
| `tromba_marina` | C2-C5 | one long bowed string over a bridge whose loose foot drums on the soundboard every period | `gap` 1 (x the 20 µm rest gap) |
| `bowed_psaltery` | C4-C7 | triangular box of metal strings bowed one by one, no vibrato | |
| `zanza` | C3-C6 | thumb piano of piano wire, tongues 100 to 43 mm over a cone, with a buzzing rattle | `buzz` 0.5, `cone` 0.44 m (135 Hz) |
| `baschet` | G2-G5 | Baschet string sculpture: a string over a cylinder bridge, its halves bent in opposite directions, radiated by a metal sheet | `bend` 30 cents, `sheet` 0.5, `screw` 0 (odd-harmonic buzz) |
| `resonator_guitar` | G2-G5 | dobro: steel strings on an aluminium cone whose low modes sag when played hard; open G tuning | |
| `whirl_string` | E1-E4 | loose string plucked out of plane: both planes share one tension (Kirchhoff-Carrier), the orbit precesses and the pitch glides | `ellipse` 0.3, `slack` 1 |

## New options on existing instruments

- `guitar`: `radiation` (1 = body modes weighted by their radiation efficiency, the body radiating as a sphere),
  `air_q` (Q of the Helmholtz mode: 0 = 52 as measured on the body alone, 19-22 for a played guitar), `pluck` nail |
  finger (flesh sticks about 20 ms, wide contact) | pick, `rattle` (a screw or paper clip touching the string),
  `cloth` (cloth on the rattling stop).
- `steel_guitar`: `worn` (measured losses of played-in strings), `radiation`, `air_q`.
- `electric_guitar`: `pickup_pos2` (second pickup) and `mix` in | out (out = thin, nasal), `stiff` (pickup comb of a
  stiff string), `amp` none | bassman with the tone stack `treble`, `middle`, `bass`, `horizontal` (the pickup
  also reads the horizontal plane).
- `rhodes`: rebuilt on the tine physics; `sustain` (electromagnetic sustain while the key is held), `onset` hammer |
  swell (the coil builds the note up, no hammer), `tremolo` (the amp's, about 5 Hz), `tonebar` fork | 2dof (tine and
  aluminium tonebar as two masses coupled through the clamp), `tonebar_cents` (with 2dof: the tonebar's mistuning
  against the tine), `tuning_mass` (tuning spring at the tine tip, as a fraction of the tine's effective mass:
  lowers the tine by 1/√(1 + m)).
- `clav`: `pickup` A (upper) | B (lower) | C (both, out of phase) | D (both, in phase), `magnetic` (measured magnetic
  pickup law), `tone` (tone rockers pressed, any of S M T B: Soft and Medium darken, Treble peaks near 1.6 kHz,
  Brilliant cuts the bass; "" = none), `beat` (dB of beating on partials 1-3 below E4, 0.5-2 Hz per partial through a
  Regalia-Mitra peak; 0 = off, up to 15).
- `harpsichord ripple_partial` (partial k >= 2 whose loss the ripple filter aims at: rrate 1/k, or 1/(2k) when r < 0;
  0 = the measured 197 Hz key's 1/12 on every key).
- `dist_guitar loop=1`: the drive inside the string loop (cubic clip at twice the rate) instead of after the string.
- `flute novice=1`: blown like a beginner (jet Reynolds number 3000 at C4, x1.6 per octave up to 7000; past the laminar
  2500 the breath noise grows as Re / 2500).
- `cello body_am=1`: each partial's level follows the body resonances at its instantaneous frequency, so the vibrato
  turns into amplitude modulation.
- `guitar glide_mm`: pluck amplitude in mm at full velocity; the tension glide comes from the string (EA/T0 766): 2 mm
  about 7 cents, 4 mm about 28 (replaces `glide`).
- `harp`: span Cb1-C7; the low register is tuned stretched (`harp_stretch`: partial 1 flat by up to 12 cents so the
  heard pitch lands on the note). Default sound changes in the bass.
- `church_organ`: `chiff` classic | grey (onset noise around 2-4 x f0, 30 ms, 20 dB under the steady tone).
- `piano` (`instruments`): `pedal` 0|1 (sustain pedal: no damping, every other string rings in sympathy), `soft` 0|1
  (una corda: softer, rounder), `lid` open | half | closed (a closed lid takes the top off above 2 kHz).
- `fm_bass` (`synths`): `ratio` 1 (modulator : carrier; 2 hollow, 3 nasal, 0.5 sub growl), `model` classic | mex (MEX
  FM bass: one carrier, one modulator, index and level falling as the square of a ramp lasting 0.8 s x index²),
  `index` 0.6 (mex: modulation index at the attack, also the note length), `fb` (classic: self-feedback of both
  operators in radians; 1 gritty, 1.5 near noise).
- `soft_lead`, `square_lead`, `warm_pad`, `reese` (`synths`): `drift` (cents rms of slow analog pitch wander; 3-8
  audible). New instrument `packet_pad` (C2-C6): a five-harmonic saw whose upper harmonics close with `cutoff` (0..0.5,
  also the packet pitch) plus a windowed resonance packet restarted every cycle (`res` 0.3, `attack` 0.3 s, `release`
  0.8 s).
- `harpsichord`: `pluck` 0 = by register (12 % of the string in the bass to 50 % in the treble), `phantom`
  (longitudinal resonances: a bright, metallic onset), `timbre` 0..1 (one-pole darkening of the release knock; 0, the
  default, leaves the sound as it was: at 0.8 the centroid of the release goes from 1.8 to 1.1 kHz).
- `violin`: `model="commuted"` (body response relaunched every period), `direction` (0 facing the violin .. 1 beside
  it: the body's directivity heard through the vibrato).
- Winds: `meyer=1` on `flute` (attack by register and velocity, 120 ms release) and `clarinet` (attack), and on the
  brass also the spectral ceiling, dynamic timbre above 3 kHz and band vibrato AM; `oboe` `even` (dB on the even
  harmonics) and `precursor`, `clarinet precursor`, `flute expert` (jet speed up 1.35x per octave), `sax lattice`
  soprano | tenor (tone-hole lattice cutoff sets the spectral corner), `trumpet model="lip2d"` (Adachi's 2D lip),
  `trombone model="lip"` (one-mass lip on the bore modes).

## Effect `body_morph`

Passes the input through a guitar body: a warped all-pole model (λ 0.7, order 24) of a body, interpolated between
two bodies through their reflection coefficients, so every step is a stable, real body. `a` (0 = nominal) and `b`
(1) body seeds, `size_a` / `size_b` (0.5, 0..1), `steel` (false: steel-string body), `morph` (0.5: 0 = body a .. 1 =
body b), `mix` (1).

## Python helpers

- `warped_lpc(x, order=24, lam=0.7)` → (a, k), `lpc_from_reflection(k)`, `warped_allpole(x, a, lam)`: the warped
  all-pole filter behind `body_morph` (poles z = (1 + λw)/(w + λ) for each root w, zeros at z = λ, DC gain 1/A(1)).
- `acoustic`: `string_air_rate`, `plain_string_tau`, `finger_width_filter`, `pickup_mix`, `pickup_comb_dispersive`,
  `tonestack`, `guitar_radiation`, `rhodes_contact_time`, `rhodes_ratios`, `tonebar_length`, `beam_kirchhoff_shift`,
  `bowed_level`, `commuted_bowed`, `dynamic_timbre`, `band_vibrato_am`; `grey_chiff(n, sr, freq, steady_rms)`,
  `clav_tone(y, sr, tone)`, `rhodes_tonebar_modes(ft, fb, L, f0)`, `tonebar_freq(L)`, `harp_stretch(freq)`,
  `magnetic_pickup(horizontal=)` (the string also swings across the pole, in quadrature; across, the field is a
  bell, which rectifies the signal: even harmonics); `loss_onepole(f0, sr, tau0, f_half)` (closed-form one-pole loss:
  decay c1 + c3 ω², the decay time halving at `f_half`), `dispersion_allpass_from_partials(fk, sr, order)` (delay and
  allpasses that put each partial at its measured frequency), `tension_glide_cents(amp_m, position, length_m)`
  (Δf/f0 = α a²/(8p(1−p)L²), α = EA/T0), `ripple_rrate`, `regalia_mitra(x, sr, fc, gain, bw)`;
  `piano.PianoHammerAction(agraffe=True)`: the string is a dashpot plus the wave sent toward the agraffe, back
  inverted after `agraffe_delay_s()` = β/f0 (β the hammer position ratio), pushing the hammer off (C4 contact
  1.96 → 1.77 ms); contact then depends less on hammer mass in the bass and more on speed.
- `strings`: `loss_fir` (linear-phase loop FIR fitted to the per-period loss at the partials, least squares weighted
  by 1/(1 − g)⁴ so the error is even in decay time rather than in gain); the Bensa loss law (`piano_loss_coeffs`, `piano_decay_rate`, `loop_filter_closed_form`,
  `loop_filter_from_partials`); `nonlinear_regime`; `stepped_string_partials`, `fit_B`, `wound_bare_ends_B`;
  `whirling_string`; `octave_stretch_cents`, `matched_tuning_cents`; `StringExciter(kind="flesh")`;
  `render_pluck(glide_couple=0..1)`: how much the tension-modulation glide (`glide_cents`) couples the harmonics.
- `piano`: `PianoStringCourse(measured_loss=True)`, `hammer_impulses`, `piano_loop_loss`, `ORTH_RADIATION`,
  `SympatheticResonatorBank` (regions 8 x 8). `physical.piano_data`: `CHABASSIER_UNWOUND`, `HAMMER_STRING`,
  `FELT_C4`, `unwound_B`, `strike_ratio`, `felt_q`, `felt_K`. `physical.bow.bowed(width=, hair=, max_hz=)` (`max_hz`: oversample the bow above that pitch), `bowed(wolf=,
  wolf_hz=, wolf_q=30)` (wolf note: a body resonance at Q 30 just below the note, seen by the bow; 0.01 = a real
  wolf); `physical.sbp.mode_window`.
- `physical.waveguides`: `bridge_filter(f0, brightness, absorption)`, `BRIDGE_PRESETS` (electric, guitar, nylon,
  violin), `modular_string(f0, dur, rng, preset, sharpness, cutoff, velocity)` (a string with a bridge filter at
  each end, plucked by a noise burst).
- `physical.bank_piano.note()` options: `polarisations` 1|2, `bidirectional` (longitudinal tension fed back onto the
  transverse modes), `bridge_res` (gain of the 470 Hz bridge mode), `bridge_angle` (degrees, longitudinal coupling
  tan α), `weighting`, `long_place`, `pedal` (sympathetic strings), `sympathetic_strings` (the free top strings ring
  without the pedal), `lid_state` open | half | closed (2 kHz shelf 0 / −5 / −10 dB), `ribs` (above 1150 Hz twice
  the modes, each at half the mobility), `action_kind` tones | noise (board knock at 90 Hz and a 300-600 Hz shock,
  100 ms in the bass to 40 ms in the treble). Keys above MIDI 89 have no damper. Functions: `bridge_admittance`,
  `long_coupling`, `harmonic_weight`, `place_longitudinal`, `sympathetic`, `lid`, `damped`.

## Added in 1.3.0

- `accordion bend=` (Hz of pitch drop per kPa of bellows pressure, 2.5 with the key fully down, 5.1 half down) and
  `pressure=` (kPa); new instrument `harmonium` (Indian hand harmonium, C3-C6, from the measured free-reed source
  spectrum; `tubes.free_reed(reed="harmonium")` is its Python preset).
- `gamelan bar=saron|jegogan`: the saron has partials 1 : 2.71 : 4.84 (measured; the third was 5.15), the Balinese
  jegogan 1 : 2.7 : 5.2 : 8.4 : 12.2.
- `tubular_bell` uses measured loaded-tube partials (0.63, 1.22, 2, 2.93, 4.01, 5.21); `glockenspiel` is a steel-bar
  modal model (1 : 2.71 : 5.15 : 8.43 : 12.21); `singing_bowl` follows the measured qing modes 1 : 2.75 : 5.06 :
  7.78 : 10.83; `vibraphone` puts partials 2 and 3 where the register says and its motor vibrato is 6 dB deep;
  `glass` has partials at 1.89 and 3.21.
- New instrument `pea_whistle` (referee or police whistle with a pea, `spin` = trill, wind family). It is not the
  sports sfx of the same name (`sports.md`), which is a one-shot blast.
- `tabla`: the loaded skin has 9 modes on 5 harmonic tones (1, 2, 3, 3, 4, 4, 5, 5, 5), not 1 to 9.
- `violin sordino=1` filters with the measured mute shape: the bridge hump (1-2.5 kHz) loses 5 to 13 dB and the low
  range keeps its level.
- `physical.bank_piano.note`: `ribs` (above 1.15 kHz the table seen from the bridge has 0.4x the modes but stronger;
  above 2.5 kHz its mean mobility is 3.5x), `bridge_profile` (bridge conductance follows the key: -9 dB on the bass
  bridge, -10 dB at the top of the treble, relative to the middle), `orth_profile` (with `polarisations=2`, the
  horizontal plane radiates 27 dB under the vertical in the bass and level with it above 2 kHz instead of a flat
  -10 dB; `ORTH_PROFILE_DB`, `orth_radiation_gain`), `HORIZ_CENTS` 0.1.
- Bowed strings and brass speed their vibrato up at the end of a held note along the measured tail; `bass_drum` felt
  mallet contact is 4 ms; `trumpet model=lip2d` takes the mouth pressures of the published simulation (+0.5-1 kPa);
  `SympatheticPiano` couples at a passive bridge in the right direction (two-stage decay).
## Newer options (1.3.0)

- `koto` and `sitar loss="guitar"|"air"`; `rhodes hysteresis`; `guitar glide_couple`, `lam` = 0.5 - size; `harpsichord
  plectrum=` (`graphsolver.PlectrumJunction`, `pluck(...) -> (out, t_rel, f_pk)`); `mandolin plectrum=`;
  `church_organ voicing="light"`, `mouth=`, `grey_chiff(ping=)`; `organ crosstalk=`; `clarinet model="worman"|"map"`,
  `lattice=`; `cello` and `violin` `torsion`, `wolf` (elastoplastic); `trumpet mute_kind="straight"|"cup"|"harmon"`;
  `filled_glass partials="french"` (f0 / sqrt(1 + 2.61 fill^4), 11 semitones at full); `flute jet_peak=` (the jet noise peak
  at f = St U / h, St from 0.08 to 0.03 with the pressure); `recorder attack_pulse=`, `jet_offset`, `jet_angle`.
- Measured blowing conditions of a flue pipe: jet angle 39 +- 2 degrees, labium distance l/h 5.6, offset y/h 0.24 +- 0.25,
  flow 16 L/min, flue height 1.2 mm, jet speed 21.5 +- 2.7 m/s. With `recorder(jet_offset=0.24, jet_angle=39)` the jet sits
  that far off the labium (the even harmonics come up); `physical.winds.jet_regime(freq, W, Uj)` says which Strouhal regime
  the blow is in and `graphsolver.jet_bore(pressure=, bound=True)` keeps St in the first gain region.
- Instruments added in 1.3.0: `unison_strings` (`strings`, `detune` 1.6 cents), `slide_clarinet`, `steelpan`, `trautonium`,
  `ondes_martenot`, `harmonica_reed`, `accordion_reed`, `melodica_reed`, `sho_reed`, `khaen_reed`, `guitar_modal`,
  `violin_modal`, `rhodes_fd`; world instruments in `world.md`. `harmonium` is softened (corner 1.8-3 kHz) and `unison_strings`
  has a gain by register (+3 dB at MIDI 53-58 up to +16 dB at 84).
- Formants of the winds (Meyer): `french_horn` and `natural_horn` main formant 340 Hz (was 450), `oboe` + 4500 Hz,
  `bassoon` + 2000 and 3500 Hz, `contrabassoon` 250, 450, 800, `bass_trombone` 370 Hz (`HORN_BODY`).
- `PianoHammerAction(hysteresis=, voicing=, strike_ratio=)`; Python `acoustic.flue_harmonics`.
- `vibrato model="scanner"` and `compose emotion=`: see `dsp.md` and `compose.md`.
