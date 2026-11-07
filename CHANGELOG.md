# Changelog

## Unreleased

Catalog: 167 instruments (was 150), 53 drums (was 52), 534 sound effects (was 287), 133 effects (was 103), 14 extra
layer types.

Deep reading by lots (`reports/lecture_profonde` in the parent repository): each item below comes from the text the
lot read, or is marked UNSOURCED. Source citations are leaving the code: the paper, table or figure behind a constant,
or its UNSOURCED status, goes in `research/SOURCES.md` (done for `creatures` and `fauna`, partly elsewhere; other modules still carry
markers).

Weapons
- New modules `melee` (`crossbow`, `arrow`, `sling`, `slingshot_weapon`, `catapult`, `siege_impact` 60-250 kg up to
  66 m/s, `melee` with 10 kinds, `knife`, `armor`, `grab`, `spear`, `gouge`: Johnson's wedge contact) and
  `energy_arms` (`stun_gun`, `taser` 19 arcs/s and 55 m/s darts, `cattle_prod`, `arc_weapon`, `energy_weapon`,
  `plasma`, `railgun` with its N-wave at Mach 7.35-9.88, `coilgun`, `gauss_rifle` 400 m/s, `laser_impact`).
- `firearm` away from `pov`: the direct sound is the measured Friedlander wave (93 % of the energy within Maher's
  durations, 0.33 ms for a .22 to 2.56 ms for a 12 gauge), the gas tail is 5 ms x energy^(1/3); the body comes from
  `echo` or a `room`. Revolvers fire a cylinder-gap pre-pulse 0.2 ms ahead. MP5SD with a dominant bolt, `pov` with a
  300 ms body ring, `ring`, `weight`, muzzle delay barrel/v.
- Default sound changed: `firearm`, `minigun` and `bullet_pass` take `render` (`heard` by default, `pressure` = the
  bare pressure at the ear, the previous sound) and `place` (`street` by default, any `room` preset or `none`). Heard,
  a shot is the blast with Farnell's layers and a little weight, through an overloaded microphone and the place: about
  1.5 s and 10-20 dB louder to the ear, where the bare pressure was a click of a few ms. `gunshot_indoor` uses
  `place=none`; the `warfare` scene keeps `render=pressure` (its own street room).
- `gun_mech`: `reload_musket` (at least 20 s), `jam_clear`, `revolver_eject`; `floor`, `mag`, `rounds`. New sfx
  `casing_ricochet`; `bullet_impact level=`, energy scaled as its square root. `arms.RECIPES`.
- `ordnance`: new blast core (sinusoidal noise, Farnell's fireball, comb, ground wave), sfx `blast` (`kind`), effect
  `earplug`, complete rocket motor, supersonic flight, loaded and empty launcher, a claymore with 700 oriented balls,
  accelerating Stinger, `shell_incoming sweep`, underwater bubble pulse and surface. `explosion explosive=`
  (`ordnance.RE`), `burst=air|surface|rigid|crater` (Kinney 1962), `windows`; `explosion_layers` (Farnell's
  practical 31); `physics.blast_reflected`, `BLAST_DAMAGE`; `gunshot ground=`, `height=`.

Liquids and matter
- New module `viscous`: `mud`, `gore`, `slime`, `bone`, `blood`, `chem`, `burn_flesh`, after `viscous_bubble`.
- Bubbles: Xue's thermal damping (2023, eq. 5), `water_depth` on Minnaert, radius law by default in `bubble_field`.
  `drip`/`drops`: the first bubble 9-65 ms after the impact, rebounds at 143-350 ms. New sfx `rain_water`; `rain`
  gains Farnell's drops layer; `cave_drips regime=`; `ice_cubes` follow Jundt (n = 5.5); `clink liquid=`.
- `rain_on surface=eau rate=`, `rain_on_water(wind=)` at oblique incidence; `klang.rain_rate` goes through
  `ambience.rain_rate`.

Percussion
- New drum `bass_drum` (orchestral, circular membrane, Meyer's T60, `pos`, `mallet` felt|rute; tune 50 UNSOURCED);
  `kick model=farnell` (Farnell's fig. 3.6 swept cosine, 70 ms).
- `triangle` (`axis`, `dur`), `snare` T60 about 1 s with its peak near 320 Hz, `timpani model=measured` (Christian
  1984) and `damp`, `church_bell diameter= tune=second`, `foley.bell_nominal`, `fd_bar material=rosewood`,
  `contact.l_membrane`, `impact radiation=modes`, `modal_data.ARIUTTI_BELL`.

Music/compose
- `compose`: `swing="auto"` (Friberg & Sundström 2002), `ritard`, `melody_lead`/`offset_ms`, scales `ikeda_18` and
  `pythagorean`, Xenakis sieves, `mode=row|logistic`, Markov orders 0-4, Monte Carlo, cellular automata, L-system,
  Koenig, tendency masks, `sync` (Repp), `phase`, `drift_trend`, `expression`, `accent`, `group`, `cents` per note
  (`C4+12c`). KTH rules `high_loud`, `duration_contrast`, `double_duration`, `high_sharp`, `faster_uphill`,
  `repetition`, `phrase`, `inegales`, `leap_duration`, `melodic_charge`; `rmd`, `rmd="auto"` (Repp 1995).
- `adaptive`: `oneshot`, `next`, markers, tails, OAML conditions and tension, fades, moods `alarm` and `settle`,
  transition `layer`; stingers land in the key they move to.

Instruments
- New: `clavichord`, `double_bass`, `ukulele`, `bagpipes`, `mellotron`, `zither`, `shofar`, `traditional_horn`,
  `slide_whistle`, `waterphone`, `talking_drum`, `berimbau`, `shamisen`, `calliope` and `flexatone` (both
  UNSOURCED), `risset_arpeggio`, `gendyn` (Xenakis; also `osc.gendyn`).
- Strings: `violin`/`cello` `sordino`, `lifted`, `open_string`, `violin model=elastoplastic`; `harpsichord stops`,
  `lid`; `pizzicato size`, `vibrato`; `harp position` (five body resonances); `strings level=section`, `sordino`;
  `clav model=dwg` (Gabrielli); bow `curve="thermal"`, `torsion`, `fit=True`; pickups `pickup_comb`,
  `pickup_circuit`, `clav(flux=True)`.
- Piano: `bank_piano` Weinreich bridge coupling, `una_corda`, `stulov` hammer, `action` noise, longitudinal force,
  `note(long_full=True)`; `fit_parallel_filter` (Bank 2010); `piano unison=`; `modal.tension_glide`;
  `fdstring.StiffString.longitudinal`.
- Winds and brass: `trumpet`, `french_horn`, `trombone`, `tuba` take `lip_q` and `meyer`, the first three
  `precursor`, `trumpet bell_up`; `oboe meyer`; `flute flutter`, `noise_peaks`, `precursor`; `clarinet
  model=impedance` `reed` (typical|squeak|van_walstijn), `stiffening`, `lip_force`; `tubes.bessel_bore`,
  `tubes.causse_zr`, `WebsterTube(end_wall=)` (not used by any instrument yet).
- `chowning voice=brass|bell|drum|wood_drum`; `stk_fm` follows STK (vibrato, tremolo by `mod`, feedback, controls
  0-2).

Effects/DSP
- New effects: `wah`, `octaver`, `radio`, `balance`, `exp_delay`, `shaker_comb`, `dyno`, `expander`, `duck`, `gate`
  (hysteresis), `multiband`, `deesser`, `exciter`, `tape_stop`, `crunch`, `chebyshev`, `diode_clipper`,
  `lockhart_fold`, `delta_sigma`, `hpss`, `autotune`, `gender`, `mutate`.
- `compressor mix`, `key`, `preset` (Giannoulis eq. 17 detector); `limiter lookahead`, `true_peak`; `phaser invert`,
  notch and phase-distortion styles; `flanger damp`, `deep`; `pan itd`; `width mode=ms`; `timestretch anchors`;
  `cross_synth` stamp; `freqshift fold`; `spectral_gate` soft and subtract modes; `timestretch mode=hps`.
- Filters: `svf_band`, `onepole`, `resonz`, `korg35`, `peak_allpass`, `matched` (Vicanek), SVF allpass and
  bandshelf, `iir_sos` Chebyshev II and elliptic, `filters.ladder(kind="lpf18")`. Distortion: ADAA of orders 1 and 2
  (`dsp.adaa2` falls back on eq. 18 when two samples meet), `vicanek_shaper`, `dsf`. `quantize` fweighted and
  Lipshitz noise shaping, corrected `gsm_like`, `k_meter`. `acid model=phase_sync`, `osc.classic`.

Spaces
- `doppler path=`, `surface=` (Damiano 2024), `reverb=`, `shadow`, `horn`, `turbulence`, `law="tangent"`;
  `distance turbulence=`, `source_radius`, `angle`, `wall_distance`; `occlude path_diff`.
- `room formula`, `source_gain`, `source_radius`, `modal`, `audience`; presets `opera`, `concert_hall`,
  `practice_room`, `orchestra_studio`, `arena`, `amphitheater`; `room_metrics`.
- New effects `nrev`, `prcrev`, `costello`, `coupled_room`, `directivity`; `jcrev variant=stk`; `decorrelate
  mode=velvet`.
- Ground: 8 soils of Hess 1990 in `_GROUND_SIGMA`, `gravel` 53 and `old_snow` 16 cgs (Attenborough 2011, Hess 1990;
  only when a `ground` or `surface` is asked); `space.layered_impedance`, `ground_effect(layer=)`.

Fixes from the full test run
- Default sound changed: `skateboard`: the eccentric's load is m g (it was counted twice), and the roll is set by its
  RMS, so the joint knocks stand out again.
- Default sound changed: `industrial` (`presse`, `usine`) and `gym` strike a ringing metal box (`clang
  object=metal_box`) instead of a dull 0.12 s body, so the press beat is heard over the motor.
- Default sound changed: very lax voices (LF glottis above Rd 2.7, tenseness under 0.1) get back the Huber & Roebel
  pulse.
- `kick808 model=circuit` is in tune (the R170 loop's own frequency) and its pulse-shaper diode converges;
  `uneven_decay` bands sum back exactly; `contact.lod_gains` takes a single 1-D row; `sample_modes` splits double
  modes by 1 %; `water_tube` fully immersed; `sway_force` with a delay longer than the signal.
- `purr(voice)` is audible (a 200 Hz tone swelling on each purr cycle, on expiration); `kl_voice` fricatives at the
  lips and teeth; `primate diane` calls end on a 30 ms resonance instead of a click; `analysis.particle_count` thresholds
  per block; `story.sensitivity` follows Fant & Pauli (lossless rigid tube).
- `sync` wave: hard sync with 40 dB less aliasing (the BLEP residual is no longer interpolated across its step);
  `gunshot` silenced kinds really cut 160-1000 Hz (unity-gain band-pass); `force_pitch` moves the pitch (`psola
  sync=True`); `duck_band` notches the full 12 dB (zero-phase band-pass); `analysis.wow_flutter` windowed;
  `identify.drs` delays in time; `vbap_gains` guard; `va_ladder` `k`: oscillation at 2^(N/2) (4 or 16).
- Default sounds changed: `tromba_marina` and `bowed_psaltery` in tune (oversampled bow `bow.bowed(max_hz=)`, tuning
  pass; no more silence at 22 kHz); `spring_reverb` decays at its T60; `oboe` brighter at forte (corner 4 kHz);
  `recorder` jet centred; `trumpet model=lip2d` (lip mass, rest opening, Bernoulli sign) and `trombone model=lip`
  (tuning pass) in tune; `harpsichord stops=2` level; `interceptor_string` `gap` 0.6 (was 0.4, which locked the low
  notes 80-90 cents off pitch).
- `PianoHammerAction(agraffe=True)` keeps the re-contact (C2 double mass +61 %); free-bar tube modes from the exact
  roots of cos cosh = 1.

Voice/creatures
- New modules `human` (`burp`, `fart`, `kiss`, `spit`, `vomit`, `pee`, `yawn`, `gargle`, `hiccup`, `bodyfall`,
  `crowd` up to 24 voices, `shout`, `baby`, `blow`, `skin`, `finger_snap`, `cartoon_voice`, `whisper_bed`,
  `beatbox`, `yodel`, `rap`) and `fauna` (`bird_kind`, `chicken`, `farm_bird`, `pig`, `sheep`, `horse`, `donkey`,
  `farm_animal`, `wild_animal`, `primate`, `elephant`, `whale`, `rodent`, `reptile`, `slither`, `swarm`,
  `creature_steps`).
- `speech` `effort` (Brockmann 2008), `skew`, `tilt_db`, `h1h2`, `source=klglott88`, `trajectory=gesture`, a
  helium voice. Story's lossy tract (`tract_response`, `formants`), `MEASURED_FEMALE` areas (Story 1998),
  `min_jerk`; Huber & Röbel LF regressions up to Rd 6; `pink_trombone(excitation=)`.
- `choir coloratura`, `f1_margin`, `end_boost`; `vocalize(subharmonic=)`; new sfx `dread_choir`.
- `handclap`/`applause clap_type=`, `applause sync=`, `bark mass=`, `roar species=`, `howl biphonation=`,
  `footsteps mass=`, `accel`, `incline`; `birdsong tract=gardner`.

Retro
- `psg mixer=and`, `chip=ym2149`, `dac=mosfet`, `duty` (AY8930); ZzFXM `0.A` cells; `retro.ZZFXM_INSTRUMENTS`.
- `sfxr`: bfxr waves `rasp`, `fmsyn`, `voice`; `attack_s`, `sustain_s`, `decay_s`, `slide_oct`, `lpf_hz`, `hpf_hz`.
- `THIRD_PARTY_NOTICES.md`: STK (MIT), ZzFXM (MIT), AKWF (CC0).

Weather/fire and nature
- New module `nature`: `firework`, `firecracker`, `lava_flow`, `geothermal`, `tornado`, `volcano`, `tsunami`,
  `singing_sand`, `collapse`, `crack`, `rain_on`, `fire_out`, `storm`, `foliage`, `earthquake_stage`, `dirt`, `tree`,
  `wind_kind`, `rapids`, `fountain`. New modules `sports` (`ball_bounce`, `pea_whistle`, `snow`, `sport`,
  `skateboard`, `ice_skate`, `swim`) and `scenes` (`scene` with 46 of the 47 AMB categories, `room_tone`).
- `fire whine`, `woof`, `clatter`, `detail`; `lightning model`, `precursor`; `thunder strokes`, `walk`, `fineberg`
  (Wright-Medendorp closed form, N-wave stretched with distance); `ice_crack temp`, `salt`, `c_axis`;
  `aero.space_rocket turbulence=` (scintillation); `matter.arc(click=)`; `sway_force(species=, tree=)`,
  `tree_sway_hz`.

Mechanics and vehicles
- New modules `household` (38 sfx: locks, doors, appliances, alarms, plumbing...), `machines` (17), `engines`
  (`engine_layout` with 19 architectures, `diesel_engine`, `two_stroke`, `forced_induction`, `engine_event`,
  `race_car`, `jalopy`), `heavy_vehicles` (13), `road` (14), `rail` (`rail`, `traction_drive`, `curve_squeal`),
  `marine` (11), `aero` (13), `designed` (28: cartoon, UI, Morse, modems, alert tones...).
- `pipe_engine` `offsets`, `banks`, `headers`, `cycle=4|2|rotary`, `muffler`, up to 16 cylinders, `rpm` and `load`
  as curves; `cyber.rpm_curve`.
- `train horn`, `creep`, `corrugation`, `wavelength`; `gearbox friction`; `tyre_squeal model=brush`; `transformer
  strip`; `bearing angle`, `creep`; `stick_slip oil`, `radius`; `transporter activate`; `switch kind=relay coil=`;
  `modal_engine misfire`; `scrape stroke`; `wind_chime tube=square law=rayleigh`; `pulse_dial`,
  `phone_tone line=`; `siren_wail kind=police_rc`; `teleport style=ring`, sfx `shield_up`.
- Contact: damping laws `obrien`, `zheng`, `sterling_rayleigh`, `caughey`, `hybrid`; `impact` `thickness`, `tip`,
  `mount`, `rest_force`, `ice_temp`, `dilate`, `moisture`; `bounce yield_stress`, `restrike_jitter`, `hit_spread`;
  `roll body=can`; `scrape level=power`, `model=curvature`; `interaction sharpness`; `contact.free_disc_freq`,
  `rolling_resistance`, `loss_tangent`; `physics.natural_period`, `fireball`, `potential_collision`;
  `friction.BASCHET_MODES`.

Analysis/CLI
- `analysis.hnr`, `periodicity` (Boersma's ACF), `detect_clicks_lpc`, `loudness_range` (EBU Tech 3342), roughness,
  dissonance, annoyance, inter-channel correlation, EDR; `describe` gains five keys; `bark_to_hz` fixed above 20.1
  Bark; `analysis.wav_check`.
- `calibrate`: `compare` (a render against a recording), `fad`, `loudness_track`, `sdt`, `mushra_screen`,
  `align_stimuli`, `n_opt`, `emg_fit`; `identify.pcen`.
- `genny info --check`.

Default sound changes
- Clicks fixed at the source in `render_modes`, `strike`, bubbles, `stickslip_pulses`, `declick`, `vocalize`,
  `speech`, `sword_swing`, `aeolian_harp`, `brass`: renders change at equal seed. Pink Trombone: return phase floor
  0.2 ms, no strain spikes, other sound under tenseness 0.1. `sitar model=fd` buzzes again at D3.
- `firearm`, `minigun`: `level="relative"` and `echo=0` by default; a shorter, darker shot (Friedlander direct
  sound, 5 ms x energy^(1/3) gas tail, 9 mm centroid 3.5 to 1.7 kHz). `bullet_impact` level.
- `bounce`, `drop`, `roll`, `smash` use Hertz stiffness from the material: centroid +45 to 100 %, and so every sound
  built on them.
- Dynamics: every `compressor`, `limiter`, `phaser` with feedback, 50 ms SOLA, `gsm_like`, CVSD, `freqshift`,
  Leslie speeds, Doppler ground bounce.
- Spaces: `room preset=church` 5 s (Meyer), the late `room` level -0.85 dB per 10 m, `asphalt` 20000 cgs.
- Voice: `crowd_murmur`, `choir` diphthongs (at 35 % of the note) and breath, `cry`.
- Strings and keys: `piano` (Weinreich coupling floor), `harpsichord` (T60 20 s at C3, lid), `pizzicato` (no 0.25 s
  ceiling), `harp`, `rhodes`; `clavichord` attack glide 1 % of f0 x velocity; `piano_wg` without dispersion from
  about MIDI 102.
- Percussion: every `MembraneDrum` (pitch drop as velocity squared: `cajon`, `timbale`), `snare`, `timpani` (now
  `model=measured`), `tubular_bell` (modes (2n+1)^2), `hat808` duty 47.98 %, `cowbell` 540/800 Hz, `fd_bar`
  marimba tuning, large `church_bell` gains, `triangle` (4 s), `timestretch mode=hps` overlap-add.
- Liquids: bubble damping, `rain`, `surf` (first wave within a second), `drip`, `drops`, and the sounds built on
  `bubble_field`; `klang.rain_rate` 300 drops/s at 0.5 (was 152).
- Weather: `thunder`, `ice_crack` (loss 0.0045, also `crack kind=glace`).
- Mechanics: `servo` (steep attack and stop), `buzzer` keeps its fundamental (`beat=0.4` restores the old hum),
  `train`, `pass_by` and `heavy_vehicles` (Damiano's asphalt ground effect), `pass_by` wind fades in at 15-26 m/s,
  `tracks`, `switch state=off` (7 ms relay bounce), `whip` (Bürger and Krehl: 744 m/s in 850 µs), `pipe_engine`
  above 12 cylinders.
- Retro: `psg` YM envelope on the `ym2610` SSG, ZzFXM songs with `0.A` cells, `stk_fm` voices other than `rhodey`,
  bfxr presets hit, jump and blip, `soft_sync window` 0.35.
- Music: stingers `danger` (rising minor second), `defeat` (falling minor third), transition stingers; intensity 0
  is silent.
- Level trims regenerated (`_levels.py`, 167 entries).

Removed
- `aeolian_harp` `length` and `density` (they changed nothing).
- `firearm` outside `pov`: Farnell's body filter bank and its tanh (the shot was 30-50 ms long instead of 0.3-2.6 ms
  measured).
- `stk_fm`: the invented tremolo.
- Anti-click low-passes in `melee`, `marine`, `human`, `fauna`; the `charter` workaround in `scenes`; the
  "Buchla-type" label of `lockhart_fold`.
- Tests contradicted by measurement: `test_blast_has_a_low_body_and_a_tail`,
  `test_the_guns_differ_in_body_and_saturation`.

Earlier in this release

- Snow squeak follows the measured spectra of Arabadzhi & Rudik (Priroda 1963): `footsteps temperature=`
  (-1.5 C dull, maximum near 300 Hz; -15 C bright, a second maximum near 3 kHz and energy up to 8 kHz;
  no squeak at 0 C). Replaces the 800 Hz Crunch resonance.
- `distance ground=snow|old_snow`: flow resistivity 10 and 30 cgs (Albert, CRREL 1990, seasonal snow covers).
- Ground impedance is Miki's (J. Acoust. Soc. Jpn (E) 11, 1990) by default: physical at low flow resistivity /
  low frequency, where Delany-Bazley goes non-passive; `model="delany"` keeps the old one.
- `fd_piano stretch=0..2`: stretched tuning after a concert Steinway D (DAGA 2014: +5 cents at A1, +20 at A6).
- Singing lake ice (Romeyn et al., The Cryosphere 2021): `matter.ice_sing`, the air-coupled flexural wave
  (4.3 cm of ice sings at 724 Hz, measured 725); `ice_crack sing=0..1` puts it before the crack.
- New sfx `hail`: Poisson stones with exponential sizes (median 6.7 mm, Kopp et al., AMT 2023) and Lane's
  impact force (JASA 2006: impulse 1.9 m v, contact 0.1 D/v, hence far more energy above 10 kHz than rain)
  on any body; the shower is front-loaded, `hardness` goes from hard hail to graupel.
- `distance ground=forest|dirt|gravel|packed_earth|asphalt` (Embleton et al. 1983, values from memory, unverified).
- New module `arms`: `firearm` (33 named weapons in pascals: blast by calibre and direction, Whitham's N-wave and
  its arrival, suppressor, subsonic rounds, shooter's view), `minigun`, `gun_mech` (handling and whole reloads),
  `bullet_impact`, `ricochet`, `bullet_pass`; `casing caliber=` rings at the modes of that case on a chosen floor.
- New module `ordnance`: `grenade`, `molotov`, `mine`, `launcher`, `missile`, `rocket_salvo`, `rocket_flyby`,
  `launcher_handling`, `mortar`, `howitzer`, `tank_gun`, `shell_incoming`, `shell_crack_overhead`, `aerial_bomb`,
  `flamethrower`, `torpedo`, `depth_charge`, `sea_mine`. Blasts keep their level, start when the wave arrives
  and scale their tail as W^(1/3). Most sequence timings and Cole's underwater constants are unverified.
- `doppler c=`: speed of sound as a parameter (1500 m/s under water).
- Fixed `MovingPropagation`: it gave the moving-listener pitch (1291.5 Hz for 1 kHz at 100 m/s), now the moving
  source's 1411.5 Hz. Output of `apply_trajectory` changes.
- Fixed `read_wav`: float WAV and AIFF now load through soundfile instead of raising.
- Fixed `gunshot` bursts: the timing jitter grew with the round number.
- New drum `hat808` (six square oscillators, two high-passes; `open`, `decay`, `detune`). The four fixed frequencies
  come from RP.Sound's code and are not checked against a schematic.
- `church_bell type=measured`: the recorded true-harmonic bell `sy_bell4`, its T60s as measured at `decay` 8.
- `timpani model=premier strike=`: the ten measured modes of a Premier 2500 kettledrum (Chaigne & Kergomard Table 14.1).
- New instrument `padsynth` (Nasca): each harmonic widened into a band of random phases, a loop of 2^18 samples; `bw`.
- `marimba` and `vibraphone` now have the tube under the bar (Doutaut et al. 1998), `resonator=0` removes it. Their
  sound changes, and the level trims were regenerated.
- `filters.ladder` and effect `ladder`, `moog` or `diode` (Zavalishin); `acid model=diode`.
- New effect `dattorro` (JAES 1997): the figure-eight plate reverb.
- `texture_synth target_snr=`: stops once every statistic class is matched. `calibrate.texture_snr` now compares signed
  or complex values, so it sees C2's phase: its numbers change.
- New module `geo`, new sfx `rockfall`, `landslide`, `debris_flow`, `singing_dune`, `mud_pot`, `geyser`, `lava`, `glacier_calving`, `sand_storm`, `earthquake`, `tree_fall`,
  `snowfall`, `snow_avalanche`, `snow_squeal`, `pd_footstep` (`surface` gravel, snow, wood, dirt, grass). Their laws come from the inventory's reading of abstracts and snippets; levels and envelopes are unsourced.
- New sfx `hummer`, `cuica`, `blast_noise` (Farnell's sinusoidal noise), `bytebeat` (seven formulas; no user formulas are evaluated). `siren_wail kind=police_fr|gendarmerie|pompiers|samu|ambulance_fr`
  (French decree of 1987). `lightning strokes=` (Ribner & Roy).
- New instruments `stk_fm` (STK's FM patches on TX81Z tables), `chowning`, `jaw_harp`, `soft_sync`, `psg3` (three AY/YM channels on one load resistor, MAME's formula). `psg dac=ay8910|ym2149|westcott`.
- New effect `serge_fold`. `paulstretch` takes `phase`, `continuity`, `scan`, `loop`, `recipe`.
- `sfxr`: `link=` reads a jsfxr share link (tested on links built here, not on one from the site), `presets=bfxr`, `jump1`, `onset1`, `jump2`, `onset2`, `jump_repeat` under `engine=bfxr`. `zzfx mutate=`.
- `physics.rankine_hugoniot`, `physics.blast_ground` (Catovic & Kljuno); more explosives in `ordnance.RE`.
- `tape model=hysteresis`: Jiles-Atherton magnetisation (Faust hysteresis.lib constants, empirical).
- `"priority"` on layers: higher priorities duck lower ones, 3 dB per step, 12 at most (RP.Sound's mixer).
- New effects `gverb` (constants from ChucK's GVerb; not compared with its C source) and `binaural` (Brown & Duda 1998).
- `spring_reverb hf=`: the wide-band chirp series of Valimaki et al. 2010.
- `sax model=stk position=`: STK's Saxofony waveguide.
- New sfx `engine_knock` (`kind` knock, slap, rod): layer over `car_engine`. Constants are the mechsynth.js author's.
- Brass `model=physical` now steepens into shocks when played loud (`dsp.steepen`, Chaigne & Kergomard 8.4).
- `church_bell type=hibbert octave_nominal=` (Hibbert 2008: 2066 bells), `old_english`, `medieval`, `big_ben`; `tune=strike|nominal`.
- `bird_call species=budgie` reaches its measured 3.4 kHz trachea mode (Brittan-Powell et al. 1997), `freq` up to 3000.
- New drum `shell_crash`, `nl_plate dome=`: shallow spherical shell (Bilbao 13.3, with 2 q in the Airy equation). The edge is
  still rectangular and simply supported; the lowest modes rise about 40 % as far as the short-wave law.
- Corrections from the deep reading (`reports/Lecture profonde genny 2026-10-29.md`): `avalanche fmax` 160; `glass`
  fracture toughness 80 (Zheng & James), new `ceramic` physics material, so fewer glass pieces; choir spectral tilt 16 dB per
  unit effort (was an unsourced 36, Brockmann), so soft singing is brighter; `interaction` wobble follows the speed curve;
  PTR engines bank cylinders 1-4 / 5-8 and bound the jitter to +-40 crank degrees; bottle end correction 1.46 r
  (was 1.7 r: bottles sound slightly higher); tone-hole chimney h + 1.6 b; syrinx trachea end correction 0.6 b.

Downloaded papers and knowledge (2026-11-06): values come from the papers downloaded on 2026-11-05, or, where no source
could be read, from physics and known published measurements (`connaissance` in `research/SOURCES.md`). UNSOURCED
markers are being removed from the code module by module. Default sounds that change are flagged.
- aero: jet noise of `jet_afterburner` and `flyby_jet` now follows Tam's two similarity spectra with the measured peak
  Strouhal per angle; `space_rocket` uses the SP-8072 rocket spectrum. Default sounds change.
- ordnance: underwater blast constants read in Cole (52.39 MPa, K 2.108, depth + 10.06 m); bubble pulses 2 and 3 keep
  34 % and 54 % of the energy, so `torpedo`, `depth_charge`, `sea_mine` change. Ground arrivals of a Poisson solid
  (S = P/sqrt 3, Rayleigh = 0.919 S).
- arms: case modes add the finite-length (Love) shell modes, 6 modes instead of 3, so cases (`casing`, `gun_mech`) are
  brighter; `firearm` off-axis levels follow the measured AR15 directivity (-11 dB at 90 degrees for A 30); passing
  bullets slow with range by drag (Cd 0.3, per calibre); `minigun` motor and bolts at 100 and 120 dB at 1 m (were far
  too loud). `ricochet` unchanged.
- `kettle`: two regimes from whistle measurements, a Helmholtz note below orifice Re 2000, then a hole tone rising with
  the flow (about 2180 Hz, up to 2700 Hz at Re 3000; was a fixed 1800 Hz edge tone). `tone` is replaced by `flow`,
  `gap`, `cavity`: specs that pass `tone` now fail. Default sound changes.
- `boil`: cap-film bursts at the surface (`burst`, 0.5), rising chirps; default sound changes. New
  `particles.cap_burst`, `film_thickness`, `retraction_speed`, `particles.hinze_radius`: the `surf` bubble cut now
  follows the turbulence (about 1.0 mm, nearly unchanged).
- Glass-fill law: alpha read from the paper's figure (3.65, was 0.487); `ice_cubes` and `clink liquid=` drop further
  when full (ratio 0.906 at 0.6 full, was 0.986). Default sounds change.
- `viscous`: the bubble's Helmholtz ring is damped by radiation (Q = 4L²/πΦ², `viscous.radiation_q`), so `mud` and the
  other thick liquids ring longer. Default sounds change.
- `ship`: cavitation levels after Wittekind with a block coefficient per ship (cargo 0.65, tanker 0.78); new
  `model="randi"` (RANDI 3.1, by speed and length). `ship` with cavitation changes.
- Road noise: the Harmonoise tables were one third octave low and had two typos (cars 40 Hz, trucks 100 Hz); fixed, so
  `pass_by`, `tyre_roll`, `truck`, `bus` move up one third octave (overall rolling level at 70 km/h nearly unchanged).
  `cv_noise_levels(axles=)` (2 = medium-heavy category), `moto_noise_levels` (motorcycle category 4b).
- Ground: `ground_impedance(model="attenborough3", tortuosity=, porosity=)` with `HESS_SOILS` (13 measured soils),
  `FOREST_FLOORS` for `layered_impedance`; `distance ground=old_snow` is 30 cgs (was 16), more reflective.
- `contact.rolling_resistance(shape="sphere")`.
- `mammal` sfx: mammal call, glottal pulses with a ladder of nonlinear `complexity` (1 limit cycle .. 6 chaos) through
  a 15-section Kelly-Lochbaum `tract` (Wilkinson & Reiss 2016).
- `birdsong source="laje"`: syrinx labia driven along a pressure-tension ellipse per syllable (Laje 2002).
  `footsteps stop=N`: the last N steps slow to a halt (runner's final ritard); the step timing now varies fractally
  (human gait), so default `footsteps` change.
- Choir: the end-of-note vibrato speed-up (+15 % over the last 5 cycles, Prame 1994) is on by default (`end_boost` -1;
  0 turns it off, a positive value keeps the old Hz over 0.5 s). Default sung voices change.
- `chowning` bell, drum, wood_drum and brass envelopes follow Chowning 1973 (figs. 11-16). Default sounds change.
- Compose: KTH rules `leap_articulation`, `harmonic_charge`, `phrase_arch` (`phrase_turn`), `ritard {"model": "kth"}`;
  `leap_duration` now lengthens by 4.2 √ΔN ms (was 4.2 ΔN); `arrangement melody_lead="jazz"` delays the soloist's
  beats by 107.2 - 0.306 bpm ms (Friberg & Sundström 2002); melody `mode="ga"` (genetic-algorithm rhythm, `fit`,
  `generations`).
- Analysis: `zwicker_loudness` (ISO 532-1 stationary, sone), `zwicker_sharpness` (DIN 45692), `loudness_percentiles`
  (N5, N10), `spectral_tilt` with the -5 dB/octave reference of commercial mixes; `genny info --json` has `tilt`.
  Zwicker loudness reads analyser-shaped third-octave bands (`third_octave_power(order=3)`; 1 kHz at 40 dB = 1.02
  sone). Also `binaural_loudness`, `duration_loudness_db`, `yin`, `bark_flux`, `mfcc`, `jitter_shimmer` (Praat's
  definitions), `tempo_from_loudness`.
- `choir` style `belt` (pressed, F1 raised, narrow late vibrato). Speech: voiced stops start voicing 6 / 8 / 12 ms
  after the burst for b / d / g (was 8 ms for all), so English `speech` changes slightly. Open-vessel end
  correction 0.6133 R (was 0.62 R): `pour` into a cup about 0.1 % higher.
- New `daxophone` (friction): bowed clamped wooden tongue, pitch from the dax position, mode toggle near the dax.
- `snare808` now sits on the TR-808 bridged-T pair 249.6 / 499 Hz with their short decays (22 / 6.8 ms, were 60 /
  40 ms; `tune` default 249.6, was 180); `kick808` default 45.7 Hz (was 45) (Werner 2016). Default sounds change.
- `wall`: above the coincidence frequency the loss follows Cremer's law (11 dB dip at fc, +9 dB/octave above)
  instead of the mass law; `window`, `drywall` and `concrete` change above fc/2 (concrete from 63 Hz). Mass law
  constant is -48 dB (the page said -47).
- `convolve ir="car"`: cabin gain +12 dB/octave below the first cabin mode (60 Hz, 12 dB plateau), was +4 dB below
  120 Hz; the IR sits at 0 dB in the 1 kHz octave (an earlier peak normalisation had left the mids about 8 dB low),
  so only the lows change. Default sound changes.
- `harpsichord pluck=` (0 = by register, 12 % in the bass to 50 % in the treble): plucking point, level from the
  plectrum release (Perng 2010).
  `squeak model="additive"` (Thoret 2013); `physical` stays the default.
- STK `prcrev` and `jcrev variant=stk` use STK's exact 44.1 kHz lengths (next odd prime at other rates): they
  change. Clarinet reed stiffening Ka(dp) (`stiffening`, off by default) follows van Walstijn & Avanzini Fig. 3.
- New library pieces: `filters.phasor_reson` (Mathews-Smith), `strings.admittance_reflect` (Bank),
  `strings.pluck_scattering` (Evangelista), `drums.bridged_t`, `modal.ren_match_ratio` (Ren 2013 App. B),
  `dsp.cabin_gain`, `friction.thoret_friction`.
- UNSOURCED markers removed from the instrument and DSP modules; justifications in `research/SOURCES.md`.
- `gunshot`: the supersonic crack lasts the projectile's Whitham N-wave 3 m from its path (`foley.crack_ms`,
  `arms.n_wave_pa`): 0.18 ms rifle, 0.30 smg, 0.44 pistol, 0.59 shotgun, 2.07 cannon, rising in one sample (was 2 ms
  with a 1 ms rise). Every `kind` but `silenced` changes.
- `cork_pop`: Helmholtz end corrections 0.8216 r (flanged) + 0.6133 r (free) = 1.4349 r (was 1.46 r): slightly higher.
  New `foley.guitar_body_modes` (top and sound-hole air coupled through the box).
- New `body_hit` (a blow on flesh, stone hide, chitin shell, metal plating or bone; lower as the creature grows).
  `stab` / `slash` `weapon=` blade | heavy | spear | improvised and targets `stone`, `shell` (defaults identical).
  `casing ricochet=true`: the film case, a long ring and an accelerating rattle (`arms.casing_ricochet`).
- New `photon_torpedo` (energy_arms).
- UNSOURCED markers and citations of `foley`, `melee`, `energy_arms`, `designed` moved to `research/SOURCES.md`.
- `faucet`: drop bubbles are damped by their radius (about 1.4 ms at 5 kHz, was 4 ms) and rise over their first
  period, so they start without a click. Default sound changes.
- `swim`: kicks counted per arm cycle (six-beat crawl: 5 kicks a second, was 6). Default sound changes.
- `appliance kind=frigo`: the compressor runs at the mains frequency minus the induction motor's slip (48.3 Hz on
  50 Hz, follows `mains=60`; was 50 Hz).
- `filled_glass`: French's fill law (coefficient 2.33, was 1): a full glass drops about 10 semitones.
- `roller_drop` `buche` and `tube`: bending modes of a free bar (log 2.3 kHz, was 420 Hz; tube 880 Hz).
- `vacuum`, `hair_dryer`: level with speed by the fan law (+50 log10 speed dB). New pages `docs/household.md`,
  `docs/sports.md`; `sport kind=bowling` rolls on a Hertz contact patch (1.24 mm).
- `household`, `human`, `sports`, `scenes`: no citation or UNSOURCED marker left in the code; sources in
  `research/SOURCES.md`.
- New effect `air_speaker` (road): compressed-air horn loudspeaker, a valve on an air supply radiating dU/dt into a
  horn (`depth`, `horn`, `hiss`); loud, thin, nasal.
- `klaxon`: membrane modes are the clamped circular plate's (Leissa ratios). Default sound changes.
- `ev kind=trottinette`: hub motor with 27 slots (was 54 notches per turn). Default sound changes.
- `chainsaw`: chain band at the link rate (14 links per turn, about 2.2 kHz at 9400 rpm); `garden_machine
  kind=debroussailleuse`: head geared down 1.43:1 (line near 198 Hz). Default sounds change.
- `motorcycle kind=passby`: engine shaped to the CNOSSOS-EU category 4b octave profile (`moto_noise_levels`), energy
  kept. Default sound changes.
- `outboard listen=under`: spectrum skirts at 20 dB per decade (was 40).
- New functions `cyber.tyre_lateral_force`, `rail.carter_force`, `rail.halfspace_resonance`,
  `heavy_vehicles.moto_propulsion`, `heavy_vehicles.chain_rate`.
- No UNSOURCED marker or citation left in `marine`, `road`, `heavy_vehicles`, `rail`, `engines`, `cyber`; sources in
  `research/SOURCES.md`.
- `medical kind=moniteur` follows the IEC 60601-1-8 medium-priority pattern (three 150 ms pulses 150 ms apart, then
  2.5 s of pause), and every `medical` beep rises and falls over 15 % of its length. Default sound changes.
- `firework kind=sifflet`: the whistle falls from 6 to 2 kHz as its tube burns empty (it used to rise); this kind
  changes.
- `earthquake`: the S wave arrives 0.119 s per km after the P wave (the delay was halved before). Default sound
  changes.
- `physics.MATERIALS["rubber"]` is damped like rubber (loss factor about 0.17-0.34 instead of 0.013): rubber modal
  bodies stop ringing.
- `vowel_tube` takes `source` (`rosenberg` = Rosenberg C glottal pulse); its consonant onset is a raised-cosine
  constriction event (`tubes.event`, `tubes.events`).
- New analysis helpers: `calibrate.ir_params` (T20, T30, EDT, DRR of an impulse response),
  `physical.analysis.voice_breaks`.
- No source citation or UNSOURCED marker left in `machines`, `nature`, `chiptune`, `spectral`, `geo`, `klang`,
  `calibrate`, `physics`, `identify`, `physical/footsteps`; their sources are in `research/SOURCES.md`.
- Micro-collisions (FoleyAutomatic): the re-contact train ends within 15 ms of the first contact and is spaced by the
  periods of the four lowest modes; knocks, swords, armour and hard footsteps (`metal_web`, `ice`, `crystal`) sound
  tighter. Default sounds change in `foley`, `klang`, `melee`, `footsteps`, `physics`.
- New free-edged disc: `circ_plate_free` shape, `contact.free_disc_omega`; `free_disc_freq` uses the material's
  Poisson ratio (steel +5 %). Also in `contact`: `cut_length`, `airy_disc_roots`, `vk_disc_couplings`, and the
  truncation residual `modal_bank(residual=True)`.
- `modal_engine`: a misfire drops the combustion stroke of a cycle (probability converted per cycle); `pick="rank"`
  takes the most audible modes up to 8060 Hz (default `lowest` unchanged).
- `kick808 model="circuit"`: TR-808 pulse shaper into a bridged-T retuned during the attack (Werner 2016); `classic`
  stays the default.
- Ambience: `tear` and `stone_scrape` gestures in the place palettes; Fineberg multi-impact strikes in
  `thunder fineberg=1`; `sway_force(amplitude=)` slows and damps the sway under load.
- New `plates.thermoelastic` (Zener loss of thin metal and glass plates); `foley.guitar_body_modes` takes `volume`
  and `acoustic.guitar_uncoupled` gives the uncoupled top / sound-hole pair.
- contact: `wood:<species>` materials (32 measured woods: density, stiffness, loss), `copper`, `acrylic`, and
  measured-object shapes `ri_<object>`.
- contact: `scrape gate=1` (the scraper chatters off the surface at speed), moving `position` and `variation` (modes
  follow the contact point along the body).
- contact: `interaction prefilter`, `radius` (rolling wobble once per turn), `gamma=2` (gammatone modes);
  `impact azimuth` (mode lobes).
- contact: `roll asymmetry` is now an eccentric ball (contact path, height and load swing once per turn);
  `scrape texture=plastic` smoother (beta 2.24). Default sounds unchanged.
- contact: contact-mechanics library (elliptic Hertz, elasto-plastic indentation and crater, Mindlin, oblique impact,
  thin layers, rough surfaces, Rayleigh waves, viscoelastic contact...); `jkr_pulloff` contact radius fixed (missing
  pi).
- New `genny.physical.winds`: wind acoustics (tone holes, inharmonicity, transfer matrices, brassiness, lip and reed
  tables, flue-jet laws) and the `modal_clarinet`, `trombone_lip`, `trumpet_lip2d`, `lip_buzz`, `auvray_recorder`
  models.
- `impedance.clarinet`: conical bore, reed hardness, bite, aging, wet reed, pitch bend, squeak gesture, swept reed
  flow; stiffening works with every reed.
- `waveguides.brass(inward=)`, `waveguides.clarinet(reflection="pasp", legato=)`, `waveguides.recorder` jet offset,
  angle, tanh jet, Butterworth loop filter, convection, register hysteresis and `model="auvray"`.
- `reed_tube` / `reed_cone` `reed=`, `contact=`, `mouthpiece=`; `organ registration=`, `key_click=`;
  `brass follow=`; `voice.lf_pulse(noise_shift=)`. Defaults unchanged.
- New instruments (klang): `a_clarinet`, `eb_clarinet`, `contrabassoon`, `heckelphone`, `bass_trombone`,
  `natural_horn`, `clarino`.
- `bass_clarinet`: even partials 30 dB under the odd ones in the low register (was 24 dB). Default sound changes.
- `networks.DynamicToneHole` is a sourced hole impedance (open and closed lengths); `MultiHoleBore` corrects segment
  lengths by the series length. `LipBoreInteraction` lip area from Cullen's Table B.
- `spring_reverb`: full parametric spring (dc blocker, pre-echo and ripple taps, wandering delay, elliptic output,
  first-chirp EQ, wide-band chirp loop on by default), presets of measured springs, multi-spring tanks from
  geometry, quarter-rate `efficient` mode, in-loop `clip`, and a slow finite-difference spring (`model="fd"`).
  Default sound changes.
- `leslie`: horn back lobe at -12 dB (was -8) and a fixed -10 dB leakage path (`leak`). Default sound changes.
- `room`: new engines `moorer`, `sdn` (scattering delay network) and `raytrace`; diffusing walls (`scattering`),
  microphone pattern (`mic`), `priory` preset.
- New effects: `jpverb`, `greyhole`, `room_feedback` (Larsen), `reflection` (flutter, fence, stairs, panel),
  `huggins`, `bell_swing`, `orchestra_seat`, `pv_reverb`.
- `velvet_reverb`: dark velvet, extended DVN, sparse (multiplication-free) kinds, double-slope decay, measured hall
  presets.
- `distance`: temperature gradient (shadow zone / inversion), rolling ground, across a water surface (-29.5 dB);
  `directivity`: french horn, wall behind, podium, wider noise lobe, omnidirectional below each instrument's limit.
- `physical/space`: ray tracing, porous layers (Delany-Bazley), absorbers, QRD, room statistics, PA range, STC,
  Ambisonics / VBAP / transaural helpers; `puckette_rev` rev2 mode and capture (reverb as a source).
- osc: 4-point polyBLEP/BLAMP (triangle by default, saw/pulse/square above 2 kHz), DPW orders 1-6, BLIT (sinc,
  B-spline, Thiran), hard sync with a windowed-sinc BLEP, CZ phase distortion, GBUZZ, JP-8000 supersaw, pitched and
  held noise, Esqueda corner clip; new page `docs/osc.md`. Default sounds change (triangle everywhere; saw, pulse,
  square above 2 kHz).
- sfx: `fm_machine`, `scifi_fm`, `noizer`, `overblow`; `gunshot` `angle`, `ear`, kind `silenced_rifle`; the
  silenced pistol is now about 22 dB down with the 160-1000 Hz cut; `transporter` rebuilt (FM source, flanger,
  jcrev); `coin_spin` exponents -2/9, 4/9; `wind_chimes strike=`; dice contact time from Hertz (acrylic, about
  0.2 ms on wood). Default sounds change: `coin_spin`, `transporter`, `gunshot kind=silenced`, `dice`.
- fx: `kasfilter`, `phase_bash`, `antisat_svf`, `random_allpass`, `reverb freeze=true`, tremolo shape parameters,
  `distortion aa=adaa2`, `overdrive aa=adaa3`, oversampled final clip in `limiter`, `ladder kind=huovilainen` and
  `solver=newton|adaa`.
- filters: SVF shelves and tilt, matched first-order shelf, antisaturated SVF, atanh one-pole, Faust Q mapping,
  stretched EQ, bounded prewarping, random allpass.
- dsp: `adaa_n`, GSM lattice interpolation (`codec` sounds different), `wall leak_area=`, `convolve loop=true`, spring
  IR echo time from the helical-spring dispersion (`convolve kind=spring` changes), `factor_body`.
- New modal von Karman engine `physical/vkmodal.py` (variable thickness, imperfect shells, free edges, power-law
  damping, SAV and energy-conserving schemes) and drums `vk_cymbal`, `vk_gong`; `plates` helpers `modal_density`,
  `prestressed_modes`, `radiation_damping`, `near_field`, `point_force_pressure`, `rayleigh_t60`.
- `plate_crash`, `shell_crash`, `plate_ride`, `plate_china` follow the crash-cymbal damping law (highs shorter);
  `plate_crash` decay 6.9 s by default. `tam_tam`: lows ring 75 s, 4 s at 8 kHz, 8 s rendered (was 12 s / 5 s).
  Default sounds change.
- `gong`: a hard stroke rises up to 80 cents (`glide`), `curvature` for a domed gong; `triangle`: dense bent-rod
  mode bank peaking near 6 kHz. Default sounds change.
- `kick808 model=circuit`: TR-808 decay feedback loop and tone stage (`knob`, `tone`); new `tom808`, `conga808`.
- `fd_drum`: `berger` tension modulation, `detune` doublets; new drums `bodhran`, `water_tube`.
- tubes: new `tube_mesh` effect, a sound heard down a long square duct (3-D wave mesh, stereo): distance from the
  duct's reverberation, level falling slower than in open air.
- env: `power_adsr` (its own power curve for attack, decay and release) and `win_env` (attack/release shaped by a
  Hann, Blackman, Parzen, Tukey, Welch or Poisson window).
- piano (bank_piano): `note(soundboard="fdn")` renders the soundboard as a reverberator (long in the bass, short in
  the treble) at the same level as the default filter. Default unchanged.
- New string instruments: `kantele`, `string_scrape`, `guitar_chord`, `tanpura`, `tromba_marina`, `bowed_psaltery`,
  `zanza`, `baschet`, `resonator_guitar`, `whirl_string`; new page `docs/acoustic.md`.
- Electric guitar pickup mix, dispersive pickup comb and Fender Bassman tone stack; guitar radiation efficiency and
  finger / pick / nail plucks; `clav pickup=`, `magnetic`; `violin model=commuted`, `direction`.
- `rhodes` rebuilt on the tine physics (Hunt-Crossley contact, cantilever overtones, damper, EM sustain). Default
  sound and level change.
- `harp`, `pizzicato`: Stokes air losses and finger width (`upright_bass`, `finger_bass`: finger width);
  `harpsichord` pluck point by register; piano (`PianoStringCourse`) Bensa losses per string, multiple hammer
  impulses, horizontal plane -10 dB; elastoplastic `violin` level follows `vel`; `flute vibrato>0` adds AM above
  3 kHz. Default sounds change.
- `fd_piano tuning=`, `restrike=`; `bow.bowed(width=, hair=)`; `strings` loss, inharmonicity and whirling helpers.
- Winds: `meyer=1` on `flute` and `clarinet`, brass spectral ceiling, dynamic timbre and band vibrato AM;
  `trumpet model=lip2d`, `trombone model=lip`, `sax lattice=`, `oboe even=`, `precursor=`, `flute expert=`.
  Defaults unchanged.
- `physical.story.kl_voice`: time-domain vocal tract (normalised Kelly-Lochbaum, trachea, nose with sinuses,
  piriform fossae, turbulence noise); `tract_response(branches=)`, `liquid`, `fricative_area`, `transfer`.
- Analysis functions: Praat jitter, chaos measures, particle count, temporal masking, threshold in quiet, Rd,
  impulse-response measurement by exponential sweep, particle-swarm parameter fit.
- `master_loudness`: with no `true_peak`, the ceiling is now -2 dBTP for a target above -14 LUFS (-1 otherwise);
  new presets `apple_tv` (-24), `itunes` (-16.2), `kids` (-28) LUFS.
- `speech` key `locus="place"`; `sing` key `vib_am`.
- New instrument `stk_fmvoice`; `chowning voice=soprano`.
- `creatures`: `bark dominant=`, `stride_hz`, `creature_profile`; `physical.voice`: `octave_doubler`, beak
  filter that follows its opening (`syrinx beak_open=`), `lf_waveform variant=2`. Defaults unchanged.
- matter: bubble physics options (coupled bubbles, surface image dipole, plume sizes and collective mode, pinch /
  merge forcing, Xue amplitude law, bubble-to-ear transfer): `bubbles flow_lpm= forcing= amp_law= transfer= surface=
  vessel=`; `pour temperature= collective=` (hot water: vapour in the bubbles), `drip harmonic= crater=`,
  `splash entrain=stochastic`, `fizz pops=`, `cave_drips drop_ml=`; `underwater_ambience surf= tide= wind= ships=
  biophony=` shallow-water noise bands; `surf sizes= plume= puffs= active=`. Defaults unchanged.
- nature: `fire_out` now has a steam whine, bubbling embers, cooling creaks and a steam shriek (`detail`, 1 by
  default). **Default sound changed.**
- ambience: wind atoms (`wind_scene(sources=("atoms",), wind_type=)`), stand density index level, `diffuse_field`.
- foley `sword groove=`, geo `snow_avalanche precursor= density=`, `feedback_guitar(aperture=)`,
  `ordnance.TNT_JWL` / `jwl_pressure` / `charge_radius`, per-class calibration distance (`calibrate.class_distance`).
- Source citations removed from the code of footsteps, spectral, tubes, klang and physical/analysis (they are in
  `research/SOURCES.md`).
- friction: new instrument `musical_saw`; `cristal bend=` and `modes=4`; cristal harmonics follow finger pressure
  (unchanged at 1 N).
- friction: measured Baschet tables: `rod_bank rod=square|piano|bronto|sad|fixed|paired`, `strike=tangential`,
  `collector=`; `bowed_bar material=aluminium_b|iron|comb`; `bowed_sheet glide=`; `sheet_radiator metal= mute=`.
- friction: `coil_spring` echo round trip from its geometry (`wire`, `coil`, `turns`): default about 35 ms (was
  30 ms). **Default sound slightly changed.**
- contact: `impact rough= freq_factor= decay_factor=`; `scrape model=meso`; `bounce`/`drop restitution=0` takes the
  material's value; mode masking `ModalObject.pruned(masking=True)`.
- contact: `avalanche` swells over a fifth of its length (was 12 %). **Default sound changed.**
- physics / modal: contact-mechanics helpers; `BarBody inner_radius_m`, `thickness_m`, material `brass`;
  `fracture(mass=, size=)` energy budget; `sword_swing` BROOM and WOODSWD presets from measured geometry.
- dsp: the `wall` effect's drywall is now a double-leaf stud wall (mass-air-mass dip, stud bridging, 52 dB flanking
  ceiling): about 8 dB more loss in the mids (35 / 47 dB at 250 Hz / 1 kHz, was 27 / 39). **Default sound changed.**
- dsp: `codec kind=gsm` quantises and interpolates log-area ratios like GSM 06.10 (sounds slightly different);
  `overdrive` shapes `asym` and `tube_bendiksen`; `tape wow_rate`; PQMF bank, LTV FIR noise, envelope compression
  helpers; new sfx `complex_osc`.
- cyber: new effects `adaptive_fm`, `grain_delay`; `paulstretch drift`; the `shimmer` recipes pitch by resampling
  (slightly different); `gearbox housing` takes a list of peaks or `car`.
- spectral: new effect `spectacle`; `freqshift hz_end` glides; phase rotation, TWM pitch, Laguerre warping and
  envelope reduction helpers.
- sfx: new `scratch` (DJ); `riser pulse`; `vinyl wear`. filters: `sallen_key`, `tsk_allpass_q`. reverbs:
  `fdn_lengths(t60=)` mode-density rule.
- compose: `humanize` model `arma` (hi-hat ARIMA timing), `gamma` for the voss melody mode, `off` / `echo` /
  `min_ioi` layer keys, `poisson_grains`, Karplus-Strong `121` / `siegel` loops and multi-copy tables in
  `ks_drum_raw`.
- analysis: wow and flutter (AES6), normalised echo density and mixing time, spectral moments per frame,
  `brightness` spread, alias audibility spread, ratio pitch and virtual pitch, HNR by moving average, exposed tail,
  OVER count, temporal descriptors, TWM f0, pitch strength, SCI, spectral irregularity.
- calibrate: multi-scale spectral distance, spectral-flux error, loop filter from a recording, FAD distortion grid;
  identify: spectral kurtosis.
- spec: `headroom_only` loudness mastering (one plain gain, never past the true-peak ceiling).
- physics: recorder jet offset and Reynolds-gated turbulence (options), nonlinear passive allpass, Holger vortex
  force, clarinet iterated map and its limits, Poiseuille/Bernoulli slit flow, Bensa piano loss law, Rauhala tunable
  dispersion table, tonebar tuning, order-N Thiran; `hammer_string` now loops through a Thiran allpass.
  **Default sound changed slightly** (also any `ThiranDelay` with a fraction below 0.1).
- New `physical/fdgrid.py`: polar grids (membrane, plate, spherical shell), isotropic and implicit FD schemes, DST
  Airy solver.
- New drums `polar_shell` (polar-grid spherical shell cymbal / gong) and `perrin_bell` (63 measured bell modes with
  doublets); `tom`, `snare`: `under` / `flip` double miking (top + bottom mic).
- New instruments `mesh_vowel` (2-D waveguide mesh tract) and `box_bar` (bar over a box resonator); `tubes` radiation
  helpers (rational impedance, end reflection filter).
- `creak`: `width`, `height`, `wave_speed` door-leaf modes.
- Removed the dead duplicate `gong` definition (no sound change).
- matter: `flame(length=)` combustion-noise spectrum (peak St U / L, f^-2.5 above); `combustion_peak`,
  `combustion_spectrum`.
- reverbs: `distance(pipe=True)`: open-pipe-end directivity (Levine-Schwinger approximation: radiates behind at the
  mouth's reflection level).
- physical: `space.ground_impedance(model="slit")` slit-pore ground; `voice.syrinx(membrane="fletcher")` sounds from
  300 Pa, `radiation="piston"` beak radiation.
- Source citations removed from the docstrings of matter (moved to `research/SOURCES.md`). No default sound changed.
- `bird_call`: labial source with two coupled labia (`source="labial"`, `coupling`), latex artificial syrinx
  (`latex`), `membrane` and `radiation` options; default unchanged. `birdsong(tract="oec")`: oropharyngeal-esophageal
  cavity and beak.
- `purr(voice=)`: voicing during the purr; `primate(kind="diane")`: Diana monkey leopard alarm calls on a four-tube
  tract.
- `song`: ProTracker module import (`format="mod"`), frame-tick sequencing (`ticks`, `frame` zx48 / zx128 / pal).
- `zzfx_voice(engine="zzfxg")` and `song(format="zzfxm", engine="zzfxg")`: ZzFXM's embedded generator;
  `sfxr(strict=True)`: Bfxr's own knob laws.
- `gb_wave`: wave table built from a spectrum; `retro.beeper`: ZX Spectrum 1-bit BEEP.
- `write_wav(dither=, seed=)` and the spec key `"dither"` (+ `"dither_seed"`): TPDF (or any `dsp.quantize` mode)
  dither at export, fixed seed. Default off: every render is bit-identical.
- New effect `granular`: ring-buffer grain cloud with spray, pitch, stereo spread, freeze, feedback, Hann/Tukey/perc
  grains and `overlap="sola"` crossfades. New effect `granulate`: brassage, the sound rebuilt from grains drawn near
  the current time.
- `timestretch` and `pitch_shift` take `mode="psola"` (TD-PSOLA, formants kept) and `mode="auto"` (PSOLA on sharp
  voice pulses, vocoder elsewhere); new `spectral.psola`, `pitch_marks`, `epoch_sharpness`. Default `vocoder`
  unchanged.
- `feedback_guitar` accepts `aperture` in JSON specs (it was Python only). Default unchanged.
- compose: perform keys `feel` (soloist / bass / ride: Ellis delay and ride swing ratio by tempo), `emotion`
  palettes, `phrase_lengthen`, `nma`, `ensemble`, `accent_dip`; adaptive transitions write `level_db` / `pause_s`.
- `polar_shell(zone=bell|bow|edge)`: tapered, domed cymbal thickness law (`percussion.cymbal_thickness`). Default
  unchanged.
- `strings.render_pluck(glide_couple=)`; nonverbal: damped-oscillator laugh rhythm (`rhythm="oscillator"`),
  `moan_sova` preset, syllable vibrato.
- `stk_fm(dac=1)`: TX81Z output filters (`chiptune2.tx81z_dac`). Default `dac=0` unchanged.
- osc: MinBLEP saw, wavetable from a spectrum, perceptual alias-free limits (`dpw_order_for`).
- tubes: tone-hole length corrections (`WebsterTube(hole_kind=)`, default `lumped` unchanged), jet receptivity;
  `vowel_tube` `glide="minjerk"`, `model="kl"` (Kelly-Lochbaum tract) and `nasal`.
- choir: `peak_stretch`, `vib_dyn`, `vib_cycle`, style `amateur`.
- filters: nonlinear one-poles, ladder modal mixes, TB-303 `tbvcf`, four-stage low-pass. dsp: all-pass FDN matrix,
  unilossless check, `apdist`, waveguide oscillator, wave digital elements, variation filter, head shadow,
  decorrelation FIR, dynamic all-pass, median harmonic comb, frequency-domain modal synthesis. identify: velvet-noise
  fit of an impulse response. `marine.plume_f0`, `engines.cold_idle`.
- `genny info --json` reports `overs` (runs at full scale); the text output is unchanged.
- drums: the dead duplicate `kick` and `tom` definitions are removed (same sound; `genny list` order changes).
- `physical.bank_piano`: two polarisations, bidirectional tension feedback, bridge mode and angle, sustain pedal and
  sympathetic strings, lid, ribs, action noise. Default sound changes: keys above MIDI 89 are no longer damped at
  release.
- `piano` takes `pedal`, `soft` (una corda), `lid`.
- fdstring: θ/r implicit schemes (`fd_piano scheme`), conservative hammer (`hammer_model`), spread hammer
  (`hammer_width`), point mass (`fd_string bead`), `fd_bar` `glide` / `scan` / `scheme="implicit"`, new instrument
  `interceptor_string`, `spring_reverb wander="uniform"`.
- `fm_bass(ratio, model="mex", index)`, `bassoon(meyer, precursor)`, `distant_bell(diameter)`,
  `church_organ(chiff="grey")`, `clav(tone)`, `rhodes(tonebar="2dof", tonebar_cents, tuning_mass)`,
  `magnetic_pickup(horizontal)`, `bowed(wolf)`, `waveguides.modular_string` and `bridge_filter`, `jaw_harp` paths
  figure8 / triangle, `simulate_nl(airy="dst")`, `nl_plate(scan_order)`, `fd_drum(under, flip)`. Defaults unchanged.
- Default sound changes: `harp` now spans Cb1-C7 and its low register is tuned stretched (partial 1 up to 12 cents
  flat, so the heard pitch lands on the note); `fd_bar(tuning="xylophone")` is now 1:3:6 (was partial 2 at 3x only).
- vehicle: `car_engine` model=ptr follows the thesis (ν 0.9, load envelope, per-cycle jitter, `inharm`); new
  `model="pd003"`.
- cyber: bearing sizes, states, Stribeck load zone, outer-ring rotation, defect width, ring, mirror; transformer
  `joints`, `stress`, `packet` and `grid`; gearbox `mount_mass`; pipe_engine duct options (extractor, cylinder guide,
  timing, jitter, expansion chamber, side branches, outlet, alternator, roots, cabin view, `load="auto"` ...) and
  `engine_rpm`; `packet_loss channel`; `pass_by sources=2`; new sfx `traffic`, `induction_motor`, `machine_hum`;
  effect `car_cabin`. Default sound changes: `transformer` draws its constant mains offset from the seed instead of
  the old slow drift, so its random stream changes.
- contact: modal_engine `body="car1"` and `valves`; rolling and creep laws; engine rupture helpers; impact `uneven`.
- friction: brake noise taxonomy (`brake_squeal kind`, `hot_spots`); `whistling_blades sumtone`; new instruments
  `nail_violin` and `marloye_harp`.
- machines: `jackhammer model="rod"`; `hvac state="box_fan"`.
- physics: Helicopter multirotor, fuselage, gearbox and turbine partials; Rotor farnell style.
- road: new sfx `car_wind`.
- designed: `alert_tone` roughness (`rough`, `rough_hz`); `can_roll` second pattern and ground bumps.
- foley: broom and wooden sword swings; `wind_chime dip`; `switch noise="pnoise"`; `servo case`; `church_bell swing`;
  new sfx `relay_chatter`.
- modal: additive tables; propeller vortex harmonics and chord taper; `sy_s100` position fold.
- piano: non-rigid bridge termination (`bridge_im_admittance`, default 0); bow: exponential and hyperbolic friction
  curves; footsteps: `stick_squeal`; piano hammer validation tests.
- New effects `va_ladder` (exact VA ladders lp / hp / bp, Korg35, elliptic), `fog` (FOF granular), `tf_pad`
  (time-frequency shift pad), `lucier` (room re-recording passes); new instrument `packet_pad`.
- Physical options, defaults unchanged: syrinx `raven` set and `fletcher` radiation (`bird_call membrane="raven"`,
  `radiation="fletcher"`), `clap(model="rj")`, `handclap(rise="exp")`, `fm_bass(fb=)`, `drift=` on `soft_lead`,
  `square_lead`, `warm_pad`, `reese`, `drip(impact="franz")`, `bubbles(amp_law="pinch")`,
  `fdn_reverb(matrix="weighted", weights=, diffuser="vector")`, `lap(plume=)`, `engine_event("launch_control")`.
- acoustic: `clav(beat=)`, `harpsichord(ripple_partial=)`, `dist_guitar(loop=1)` (nonlinearity inside the loop),
  `flute(novice=1)`, `cello(body_am=1)`, `guitar(glide_mm=)`; helpers `loss_onepole`,
  `dispersion_allpass_from_partials`, `tension_glide_cents`, `ripple_rrate`, `regalia_mitra`.
- `fdstring` `rattle(gravity=)`, `plates.apnl` / `plates.wgr`, `klang.cauchy_wave` / `cauchy_gh`,
  `bank_piano.note(air="vc")`, `spectral.true_envelope(accel=True)`, `physical.voice.stk_moog`, `f0_track` public,
  `drums.vline`, `engines.launch_curve`, `marine.plume_tone`.
- Fixed: `bank_piano` falls back to uncoupled strings when a bank of close poles is ill-conditioned (cond > 1e6).
- klang: sampler layer key `hold` loops the steady part of a sample for long notes (correlation-chosen splice).
- plates: energy-conserving snare collision (`membrane(snare=)`, `fd_drum(snare=, snare_gap=)`, `collision_step`,
  `simulate_snare`).
- sfx: `helicopter` exposes rotors, spread, fuselage, box, partials (defaults unchanged).
- tubes: `tube(steepen=, silencer=)` wire `cyber.steepening` and `cyber.polack_ir`; new `steepen_through`,
  `silence_through`.
- piano: `PianoHammerAction(agraffe=True)` adds the agraffe reflection (C2 double mass, steeper contact-time vs speed).
- percussion: `agung` and `gong(preset="agung")` (bossed gong with beating doublets).
- dsp: `weighted_rw`, `wall_rw` (ISO 717-1 single-number rating).
- identify: `tsa`, `drs`, `sanc`, `med`, `ar_residual`, `kurtosis4`, `material_from_modes`.
- choir: `sing_phrase(falsetto="break")` register break with hysteresis.
- creatures: new `body_fall` and `stridulation`; `birdsong(source="gardner", "suboscine" or "tyrant", coupling=,
  gestures=)`; `footsteps(walker=, mood=)`.
- fdstring: `twang(kind="ruler", slide=)` length slide, pitch rising as 1/L².
- contact: `roll(patch=)` averages the surface over the Hertz contact patch (low-pass at U/2a); off by default,
  default sound unchanged. `contact.wave_restitution`: restitution from the elastic waves radiated into the struck
  body (Hunter's loss), no effect on defaults. `contact.patch_average`: contact-patch filter sin(ka)/(ka).
- Sources: `forge_contact_time`, `modal.contact_time`, `mindlin` and `mindlin_loss` are now read (Hunter 1957,
  Goodman & Brown 1962, Gray & Johnson) and no longer tagged `connaissance`.
- `leslie`: fast speed now 6.92 Hz horn / 5.9 Hz drum (was 6.67 / 5.67), horn spin-up time constant 1 s,
  12 dB/octave crossover and a 53 cm cabinet: **default sound changes**.
- `velvet_reverb(preset="pori")`: measured octave-band decay (2.7 s at 125 Hz to 1.1 s at 8 kHz); new
  `reverbs.velvet_ir(t60_bands=)` and `reverbs.HALL_T60`. The pori preset sounds darker.
- `strings.loss_fir`: least-squares weights 1/(1-g)^4 (even error in decay time) instead of 1/f.
- `wall`: coincidence follows Cremer / Josse-Lamure, R(0) + 10 log[(2η/π)(f/fc − 1)], with a third-octave floor at
  fc (Fahy & Gardonio eq. 5.59-5.60): deeper dip (17.5 dB instead of 11.2 at η 0.03). **Default sound change**:
  `window` about 3.5 dB thinner around 4 kHz, `concrete` 3 dB thinner at 250 Hz (Rw 54 → 50).
- Sources: double-leaf wall (mass-air-mass f0, Sharp limit, stud bridging), plate radiation efficiency, modal
  density and point-force radiation are now checked against Fahy & Gardonio (no code change).
- AGENTS.md: design recipes for dialogue ducking, ambience permutations and variation practice (Collins 2008,
  Stevens & Raybould 2011).
- `analysis.zwicker_loudness` uses the ISO 532-1 third-octave filter bank: a 1 kHz 40 dB tone now reads 1.000 sone
  (was 1.017); Table A.9 slope cell corrected (0.22).
- New `analysis.zwicker_loudness_tv` (ISO 532-1 time-varying loudness), `zwicker_band_levels`, `loudness_level`.
- Changed: `analysis.loudness_percentiles` computes N5/N10 from the time-varying loudness and adds `max`; its `win`
  parameter is gone (API change).
- `analysis.binaural_loudness`: a mono signal reads its ISO loudness (diotic), one ear alone is 2× softer at 20 phon
  and 1.4× at 80 (was always 2× the ISO value).
- `analysis.duration_loudness_db`: −10 phon per decade below 100 ms, 0 above; `temporal_masking` holds 5 ms before
  decaying; `pitch_strength` grows 2.5× from 20 to 80 dB. No default sound changes.
- compose: `leap_duration` follows Friberg's thesis erratum (upward leap: note before −4.2 √ΔN ms, landing note
  +4.2; downward: +2.4 then −4.2).
- compose: `duration_contrast` stops at 600 ms like the KTH rule (no lengthening of longer notes).
- compose: `melodic_charge` smooths the level of a neighbouring step; new keys `melodic_intonation` and
  `harmonic_intonation` (KTH, off by default).
- choir: `vib_arch` option, vibrato extent arch from Prame 1997 (off by default).
- physical.voice: `beak_end_correction`, `syrinx(beak_gape_m=)` (Fletcher & Tarnopolsky eq. 22; off by default).
- creatures: `bark(mass=)` tract length from the Riede & Fitch regression VTL = 3.15 + 11.53 log10 m (was
  interpolated between column extremes).
- `gearbox(mount_mass=)`: blocking-mass corner is Z/(2πm), not Z/(πm) (Fahy & Gardonio eq. 2.9, 2.68a): one octave
  lower, 391 Hz for 1 kg. Default sound unchanged (`mount_mass` 0).
- `plates.radiation_efficiency` / `radiation_damping`: new `stiffeners=` (total rib length, m): below coincidence
  σ × (1 + 2L/P), capped at 1 (Maidanik, in Fahy & Gardonio p. 206).
- Sources: Fahy & Gardonio now read in full; radiation damping, point mobility of an infinite plate, duct cut-off
  1.84, piston end correction, box air stiffness and T60 = 2.2/(fη) confirmed against it.
- New `tia` instrument and `tia` chip system (Atari 2600 VCS): two channels, 5-bit divider of the 31.4 kHz audio
  clock, sixteen AUDC waveforms with their own out-of-tune note grids (`mode` `hardware_accurate`), NTSC and PAL.
- `sid`: `fc_max` (Hz at register 2047, default 12500) for the lot-to-lot spread of the SID filter.
- Specs: `"variants"` + `"vary"` (pitch, gain, reseed) render a numbered set of variations for random playback.
- Layers: `"duck_band"` turns a `priority` duck into a band notch (500-4000 Hz by default).
- `adaptive`: `first` writes each state's first pass without its folded tail.
- AGENTS.md: design recipes from Collins 2008 and Stevens & Raybould 2011 (variation sets, phasing loops, ambience
  structure, adaptive transitions, UI families, weapons, physics impacts, character sets, chip music), merged with
  the dialogue / ambience / variation recipes above. No default sound changes.
- Added: `glitch_saw` instrument, `wah_pipe` sfx, `force_pitch` and `body_morph` effects.
- Added: `onepole_nl` kinds `zdf_hard`, `zdf_ota`; `dsp.butter2` (Butterworth of the 2nd kind); `calibrate.f0_l1_midi`.
- Added: `granulate` `pitch_st`, `positions`, `match`, `min_back`; `ringmod` `leak`; `modal_fit(method="vote")`;
  `pitch_clean`, `wavescan`, `vector_mix`.
- Added: aeolian Goldstein dipole and wake noise (`aeolian_tone physical/wake`), lock-in hysteresis
  (`aeolian_harp hysteresis`), bilinear strike grids.
- Added: Chadwick acceleration noise (`impact accel`), mode energy ranking, distance LOD (`smash distance`),
  superquadric cavities, strike click, cubic gong mode (`gong hardness`), NOIZER presets rain/surf/engine, thunder
  post-processing (`fineberg=2`).
- Added: `car_engine` ptr `noise`, `pipe_engine transients`, `transformer tesla`, `sympathetic tuning="random"`.
- Added: open-hole loss (`WebsterTube hole_loss`, `hole_rate`), impedance walls (`room2d absorb`), free-edge von
  Kármán plate (`nl_plate edges="free"`), string/bar end mass-spring-damper and absorbing end (`_FD.end`).
  No default sound changed.
- Sources of `arms`, `ordnance`, `aero` are in `research/SOURCES.md`; no UNSOURCED marker left in them except two in the
  frozen `ricochet`.

## 1.2.1

- Python 3.10 works again: 1.2.0 failed on import there (tomllib is 3.11+); falls back to tomli.

## 1.2.0

Catalog: 144 instruments (was 136), 50 drums (was 48), 242 sound effects (was 234), 98 effects, 14 extra layer types.

Physical-synthesis bibliography (`reports/Bibliographie synthèse physique genny.md`, roadmap P0 to P5): every value
below comes from the paper or data set named in the code, or is marked UNSOURCED there.

Keys
- `piano` is now Bank, Zambon & Fontana's modal piano (TASLP 2010): Steinway D strings and hammers (Chabassier &
  Duruflé), hereditary felt, Weinreich unisons with two-stage decay, the longitudinal bus driven by y² (phantom
  partials), parallel-filter soundboard. The waveguide piano is still there as `piano_wg`. `piano_data.py` holds the
  Steinway D table; `fd_piano` and `felt_piano` use it.
- `clav` (Gabrielli 2013 measurements), physical `rhodes`, `wurli` with Pfeifle's capacitive pickup.
- `harpsichord`: Välimäki's model (ripply loss filter, beating partial, FDN soundboard).

Strings
- Closed-form dispersion (Rauhala), tension modulation (`glide` on every plucked instrument), modal guitar body with
  Woodhouse spread (`body`) and Penttinen warp (`body_size`).
- `sitar model=fd`: Siddiq's string striking its curved flat bridge; the buzz grows with velocity.
- `cello model=elastoplastic`: Matusiak & Chatziioannou's measured cello G string and fitted elasto-plastic bow.
- New instrument `string_bridge_plate` (van Walstijn & Mehes, DAFx-17): a string driving a plate through a bridge on
  nonlinear springs, `config` body | coupled | plate | rattle | flat.

Winds and brass
- `clarinet model=impedance`: Guillemain's digital-impedance clarinet with Dalmont's measured mouthpiece, a
  tone-hole cutoff, self-tuning; `embouchure`.
- Brass `model=physical` (lip-reed waveguide, in tune); flute jet ratio checked against de la Cuadra.

Voice
- Exact LF inversion from Rd (Wang & Gobl), Hillenbrand vowel trajectories, Hawks & Miller bandwidths, jitter and
  shimmer that measure as Praat's, effort-dependent perturbation in `choir`, Story area functions with CV onsets.
- New sfx `laugh`, `scream`, `cry`, `moan` (Anikin's soundgen model with its presets; `voice`, `syllables`, `pitch`).

Percussion and plates
- New drums `steel_cymbal` and `steel_gong` (Bilbao et al., DAFx-23): clean, gliding, crashing; the gong swell; the
  scanned elliptical output. `nl_plate` takes `loss=(T60_0, f_c, T60_c)` and `scan=(R, f_o)`.
- `gong cascade=`: Chaigne & Touzé's 1:1:2 internal resonance, twins blooming after a hard strike.
- ObjectFolder / Sterling damping laws, Aramaki material blend (`impact material_angle`), Hertz contact stiffness.

Sound effects
- `thunder`: Few's tortuous channel of N-waves blended with Farnell's patch (`channel`); `distance` is now in metres
  as documented (it was fed to the patch's 0..1 control).
- `interaction` (Conan's rub / scratch / roll), `cavity_tone`, `aeolian_harp`, `edge_tone`, `propeller` rewritten,
  `explosion` (Friedlander, Kingery-Bulmash), `birdsong source=normal_form` (Amador), surf (Deane & Stokes), rain
  (Prosperetti peak).
- `car_engine model=ptr exhaust=`: the exhaust resonators Doerfler & Wyse's PTR learned (inline4, v8, v8_metal).

Space, mastering, analysis
- ISO 9613-1 air absorption, Delany-Bazley ground effect, `velvet_reverb`, TPT state-variable filters, a
  Giannoulis compressor with knee, `loudness` presets, 1/f humanize, polyBLAMP and ADAA.
- `genny info` prints the log-attack time and the spectral flux; `onset_lag`, `bright` and `warmth` on every
  instrument.
- New module `calibrate`: inharmonicity from a recording (Rauhala's PFD), ESPRIT modes with band residuals, Rayleigh
  alpha/beta, McDermott texture SNR, Daniel & Weber roughness (`pip install genny[eval]`), and `genny eval` to make a
  webMUSHRA listening test.

Constants checked against the delivered literature (about 430 texts): some 30 values now come from a source and
some 25 are confirmed with a citation, among them the 2 km thunder channel, police siren, red alert, transporter,
wind and pipe end correction (Farnell 2010), cello air mode and viola, bassoon and cor anglais formants (Meyer
2009), `felt_piano` C4 string and hammer (Chaigne & Askenfelt 1994, Bensa 2003), guitar body modes (Woodhouse
2012), room and hall decays (Kuttruff 2009), stone damping (Sterling 2019), bronze (Chadwick 2009), the /o/ vowel
and male tract length (Story 1996, 2005), propeller Q (Selfridge 2019). Values no text gives stay marked UNSOURCED.

Two more passes, papers and standards found on the web, then 29 papers delivered later. Some 50 values now come from
a source or are confirmed with a citation, among them:

- Chip clocks: PSG 1.7734 MHz, OPN 7.67 MHz, frame rate 60.0988 Hz.
- Foley: car horn 335/400 Hz (Hella), relay bounce, gun muzzle speeds, the whip's 2.5 mm thong (Krehl 1998), the
  Euler's disk exponents (Moffatt 2000).
- Materials and acoustics: ice (Gammon 1983), bronze density 8800, the Helmholtz neck correction 1.43a, NRC CBD-239
  walls and the mass law.
- Spring reverb: defaults per Välimäki, Parker & Abel 2010 (Table 1); transition frequency 3.5-5 kHz.
- Piano unisons within 1.6 cents below F#5 (Kirk 1959).
- Animals: lion, dog and cat vocal tracts (Weissengruber 2002, Riede 2000); bark intervals (Pongrácz 2006); purr at
  26.3 Hz (Sissom 1991).
- People: hand claps and applause (Jylhä & Erkut 2008, Peltola 2007), trot and gallop cadences (Heglund & Taylor 1988).
- Voice: the /o/ vowel (Hillenbrand 1995, Table V).
- Nature: tree sway 0.35 Hz (Moore & Maguire 2004).
- Retro: SNES/Amiga lo-fi rates.

Licences of borrowed numbers: see `THIRD_PARTY_NOTICES.md` (PTR exhaust values from CC BY-NC 4.0 checkpoints,
soundgen preset values from a GPL package; no code taken from either).

Module `chiptune2` (`docs/chiptune.md`, `tests/test_chiptune2.py`): six more chips for the `chip` layer and as voices.
- `sn76489` (`sn_tone`, `sn_noise`): three squares and the white or looped-register noise, Sega and TI variants;
- `ym2151` and `ym2610`: the YM2612's FM engine with the YM2151's measured pitch table and second detune, and the
  YM2610's four FM channels beside three SSG channels;
- `sid`: the 8580's oscillators (triangle, sawtooth, pulse, noise, sync, ring modulation), ADSR and filter;
- `pokey`: 8-bit dividers on the chip's three clocks, pure tones and the 4, 5, 9 and 17-bit polynomial counters;
- effect `dpcm`: any sound through the NES delta-modulation channel.

## 1.1.0

Catalog: 136 instruments (was 129), 48 drums, 234 sound effects, 96 effects, 14 extra layer types (was 13).

New module `chiptune` (`docs/chiptune.md`, `tests/test_chiptune.py`): sound chips modelled one by one.
- layer type `chip`, hardware-accurate: `system` `nes` (2A03: two pulses, triangle, noise, the non-linear mixer,
  the output filters), `gb` (Game Boy: two pulses, wave RAM, noise, hard panning, the output capacitor) or
  `ym2612` (six four-operator FM channels, a port of ymfm's OPN core, 9-bit output with the DAC step);
- instruments `nes_pulse`, `nes_triangle`, `nes_noise`, `gb_pulse`, `gb_wave`, `gb_noise`, `ym2612`: one
  channel as a polyphonic voice, in tune (`mode: "hardware_accurate"` for the raw channel);
- hardware envelopes and sweeps, and the per-frame volume, duty, arpeggio and pitch tables of a music driver.

Voices corrected (`tests/test_voice_stability.py`): `flute`, `piccolo` and `recorder` are the jet model blown at
its playing pressure and corrected in pitch; `violin`, `cello`, `trumpet`, `french_horn`, `trombone` and `tuba`
have one pitch source (they beat against a second one); `strings` is a section of individual players.

## 1.0.1

Catalog: 129 instruments, 48 drums, 234 sound effects (was 226), 96 effects (was 89), 13 extra layer types.

New module `cyber` (`docs/cyber.md`, `tests/test_cyber.py`); its sources are listed in the module and in
`THIRD_PARTY_NOTICES.md`:
- sound effects `transformer` (magnetostriction: hum at twice the mains), `gearbox` (tooth-mesh tone with
  shaft sidebands), `bearing` (healthy and seven failing states at the kinematic fault rates),
  `pipe_engine` (valve-gated intake and exhaust waveguides, backfire), `turbo`, `tyre_squeal`, `pass_by`
  (Harmonoise tyre and engine levels, Doppler, ground reflection), `cyber_ui` (ten interface cues);
- effects `packet_loss`, `beat_repeat`, `paulstretch`, `waveset`, `formant_shift`, `cyber_voice`,
  `texturize` (texture synthesis from envelope statistics, after McDermott & Simoncelli);
- `doppler` takes `ground` and `height`: the reflection off the ground.

Examples: `examples/in_your_code.py` (genny called from a program) and `examples/library/cyber.json`
(47 renders of the new sounds and effects).

`genny.identify.identify_machine` and `genny identify file.wav`: from a steady recording to the kind of
machine (transformer, gearbox, bearing fault, engine) and the sfx parameters that reproduce it. A first,
rule-based version: reliable on genny's own renders, weak on real recordings (`docs/cyber.md`, "Limits").

`bearing` is fitted to recordings (FSTF sound dataset, SKF 6004): housing modes at 0.7, 1, 2 and 4.1 kHz
ringing 7 ms, impacts level with the rolling noise at the default severity, a soft outer-race tick, a
ball defect in bursts at the cage rate, looseness as a rumble. The comparison table is in `docs/cyber.md`.

After reading Randall (1982) and Randall & Antoni (2011), which until then were cited from memory:
- `bearing`: slip adds up from impact to impact (it was a jitter about a fixed period, the model the
  tutorial calls incorrect); a ball defect is gated once per cage turn, not twice;
- `gearbox`: `load` raises the mesh line by 20 dB per decade and its double by 7 (it was one gain for all
  harmonics); the run-out index follows the tooth count; a damaged tooth also modulates the mesh;
- `identify_machine` names a ball defect by its cage rate, tells an outer-race from an inner-race defect
  by the level of the sidebands against the line's double, and no longer takes a motor's hum for a
  defect. Checked on three sets of recordings (FSTF, AHU, SUBF): table in `docs/cyber.md`, "Limits".
  Gear sideband spacings are ranked by their stronger side, since the two sides are unequal.

## 1.0.0

Catalog: 129 instruments (was 82), 48 drums, 226 sound effects, 89 effects, 13 extra layer types.

New modules, each with a page in `docs/` and tests that check the model against its physics:
`contact` (impact, bounce, roll, scrape, smash, crumple, modal bodies by shape and material), `friction`
(stick-slip, Cristal Baschet, Baschet structures), `matter` (liquids, gases, fire, electricity, ice),
`foley` (objects, weapons, machines, signals, sci-fi, bells), `creatures` (animals, insects, birds, body,
footsteps), `choir` (sung voice, sections, lyrics), `fdstring` and `plates` (finite-difference strings,
bars, membranes, linear and nonlinear plates), `tubes` (Webster tubes, reeds, vocal tract), `reverbs`
(rooms, distance, Doppler, rotary, modulation), `spectral` (time-stretch, pitch, vocoder, resynthesis),
`dsp` (convolution, steep filters, overdrive, codecs, lo-fi), `retro` (sfxr, ZzFX, AY-3-8910, tracker),
`compose` (generated parts, arrangements, Euclidean rhythms, stingers), `klang` (vehicle ports, sampler,
articulations, QA harness).

Fixed in the string core (`genny/strings.py`):
- the loop delay ignored the phase delay of its own filters: notes played up to 134 cents flat;
- the bridge term removed 3.5 % of the energy every period: a plucked note was gone in 0.2 s whatever its T60;
- the loop resonated at 0 Hz: a large DC offset in every string sound;
- a bowed string read its delay line twice per sample: NaN above C7;
- the loop is compiled (a 2 s note renders in milliseconds) and has a pluck-position comb.

Plucked instruments (guitars, harp, banjo, mandolin, koto, sitar, pizzicato, basses) no longer mix in a
sine "anchor" at the note pitch: they are the string model alone.

Also fixed: sample-rate conversion was linear interpolation (now band-limited); instrument level trims were
stale (piano 12 dB low); drums could end on a step or exceed full scale; notes carried DC; `BarBody` damping
lacked a factor 1/2; some physical scenes returned 10-25 times full scale.

New: `genny list layers`; `docs/catalog.md` and `docs/api.md` generated by `examples/build_docs.py`;
`examples/build_showcase.py` (a tune per instrument, a groove per drum, a piece per style).

Earlier versions (0.3 to 0.22): `docs/history/`.
