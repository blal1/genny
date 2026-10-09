# What genny models, and how

One row per physical phenomenon: the model used, the catalog names that reach it, the module, and the
source the model was ported from. Each module page (`docs/<module>.md`) gives parameters, recipes, the
numbers measured by its tests and its unsourced constants. `docs/catalog.md` lists every name with its
parameters; `docs/api.md` is the Python API.

"Measured" in the module pages means a test checks the model against its own physics (mode frequency,
decay time, pitch, event rate). Nothing here was validated by listening tests.

## Sources of vibration

| Phenomenon | Model | Names | Module | Source |
|---|---|---|---|---|
| Plucked string | Two-polarisation dispersive waveguide: Thiran-tuned delay, stiffness allpasses, T60 loss filter, pick-position comb, finger / nail / pick contact whose duration sets brightness, damper at note-off | `guitar` `steel_guitar` `electric_guitar` `harp` `banjo` `mandolin` `koto` `sitar` `pizzicato` `upright_bass` `finger_bass` `slap_bass` | `strings`, `acoustic` | Smith, PASP; Jaffe & Smith 1983 |
| Plucked string (classic) | Extended Karplus-Strong: stretch, loss, pick comb, dynamics filter, sympathetic string | `ks_string` `ks_drum` `pluck` `clav` `harpsichord` | `compose`, `instruments` | Karplus & Strong 1983; Jaffe & Smith 1983 |
| Stiff string, exact partials | Finite-difference scheme, two-point frequency-dependent loss (T60 at two frequencies) | `fd_string` | `fdstring` | Bilbao 2009, ch. 7 |
| Hammered string | Power-law felt hammer two-way coupled to 1-3 detuned stiff strings | `fd_piano` `piano` `felt_piano` | `fdstring`, `piano` | Bilbao ch. 7; Smith PASP |
| Prepared string | Lumped spring, damper, rattle (collision) on the string | `prepared_piano` | `fdstring` | Bilbao ch. 7.7 |
| Tension modulation | Kirchhoff-Carrier: pitch glides down after a hard pluck | `slack_string` `twang` | `fdstring` | Bilbao ch. 8 |
| Bowed string | Implicit friction solve on the FD string; waveguide bow with hysteretic stick-slip | `bowed_string_fd` `violin` `cello` `viola` `hurdy_gurdy` | `fdstring`, `strings` | Bilbao ch. 4.3, 7.4; McIntyre et al. 1983 |
| String gestures | Bend, vibrato, slide, hammer-on, pull-off, tapping, fret buzz, palm mute, harmonics as changes of length, tension and loop loss | `StringPerformance`, `PluckedString.expressive_note` (Python API) | `strings`, `physics` | Reduced-order contact models |
| Electric guitar feedback | Pickup comb, clipper, amp-to-string acoustic feedback delay | `feedback_guitar` `dist_guitar` | `reverbs` | Smith, PASP |
| Struck bar | Euler-Bernoulli bar, free / clamped / supported, mallet contact, undercut tuning | `fd_bar` `marimba` `xylophone` `vibraphone` `glockenspiel` | `fdstring`, `physics` | Bilbao ch. 7; STK ModalBar |
| Membrane | 2-D wave equation, circular head, air cavity, mallet contact; modal Bessel membrane | `fd_drum` `tom` `taiko` `kick` `snare` `conga` `bongo` `djembe` `tabla` `timpani` | `plates`, `physics` | Bilbao ch. 11 |
| Plate, linear | Kirchhoff plate, free / supported / clamped edges, orthotropic wood | `metal_plate` `glass_pane` `wood_panel` | `plates` | Bilbao ch. 12 |
| Plate, nonlinear | von Karman plate (energy-conserving variant) and Berger model: crash build-up, pitch glide | `plate_crash` `plate_ride` `plate_china` `plate_gong` `tam_tam` `thunder_sheet` | `plates` | Bilbao ch. 13 |
| Bowed plate, bar, glass | Friction on modal bodies: Bilbao bow, elasto-plastic bristle friction | `plate_bow` `bowed_bar` `bowed_sheet` `rubbed_glass` `glass_harmonica` `bowed_bowl` | `plates`, `friction`, `foley` | Bilbao; Rocchesso & Fontana 2003, ch. 8 |
| Friction-driven rod | Cristal Baschet minimal model: hyperbolic friction, stick-slip state machine | `cristal` | `friction` | Acta Acustica 2023 (minimal model of the Cristal Baschet) |
| Baschet structures | Clamped rods through a shared collector into sheet / cone / balloon radiators | `rod_bank` `whistling_blades` `tuning_fork` `coil_spring`, fx `sheet_radiator` `cone_radiator` `sympathetic` | `friction` | Ruiz i Carulla, thesis on Baschet sound sculpture |
| Bells | Measured / FEM mode tables, Risset additive bell | `church_bell` `handbell` `china_bell` `tubular_bell` `bell` `wind_chime` `hand_chime` | `foley`, `physical.modal` | Faust physmodels; STK; Farnell |
| Arbitrary solid body | Analytic modes of strings, bars, membranes, plates with gains from the mode shape at the strike point; internal-friction damping d = pi f tan(phi) | `impact`, fx `resonate`, `ShapeBody` | `contact` | van den Doel & Pai 1996; van den Doel 1998 |
| Single reed + bore | Webster's equation (arbitrary bore), dynamic beating reed, toneholes, radiation end | `reed_tube` `reed_cone` `clarinet` `bass_clarinet` `sax` `oboe` `cor_anglais` `bassoon` | `tubes`, `physical.waveguides` | Bilbao ch. 9; STK |
| Lip reed | Lip-valve waveguide | `trumpet` `trombone` `french_horn` `tuba` | `physical.waveguides` | STK Brass |
| Air jet | Jet-driven pipe, blown bottle (Helmholtz), piston-driven pipe | `flute` `recorder` `piccolo` `pan_flute` `bottle` `piston_pipe` `pipe_blow` | `tubes`, `physical.waveguides` | STK Recorder, BlowBotl |
| Voice, speech | Klatt-style formant synthesiser, English and Spanish | `speech` layer | `speech` | Klatt |
| Voice, sung | LF glottal pulse, 5-formant cascade per voice type, formant tuning, singer's formant, sung consonants; N independent singers | `voice` `vocal_choir` `hum` `chant` `boys_choir` `falsetto` `throat_singing`, layers `sing` `satb` | `choir` | Fant; Csound formant tables; Cook 2002 |
| Vocal tract tube | Webster tract with yielding walls; Pink Trombone | `vowel_tube`, fx `formant_tract` | `tubes`, `physical.voice` | Bilbao ch. 9 |
| Animal voices | Flapping multi-pulse source, tract comb, size scaling; FM and physical syrinx with song grammar | `roar` `growl` `bark` `meow` `purr` `moo` `howl` `monster` `animal` `birdsong` `bird_call` `frog` | `creatures` | Farnell 2010 |
| Insects | Stridulation pulse trains (Dolbear's law), wingbeat tones | `cricket` `cicada` `fly` `mosquito` `bee` `night_insects` `wings` | `creatures` | Farnell 2010 |

## Interactions between solids

| Phenomenon | Model | Names | Module | Source |
|---|---|---|---|---|
| Impact | Hunt-Crossley contact f = k x^a (1 + mu v) between a striker and a modal body: velocity-dependent brightness, micro-bounces | `impact` `hit` `clang` `clink` `thud` `knock` | `contact`, `foley` | Rocchesso & Fontana 2003, ch. 8 |
| Bouncing | Geometric interval and velocity series on a still-ringing body | `bounce` `drop` `casing` `dice` `coin_spin` | `contact`, `foley` | Rocchesso & Fontana ch. 9 |
| Breaking | Rupture burst plus decelerating fragment impact trains; time-varying modal fracture | `smash` `shatter` | `contact`, `physical.modal` | Rocchesso & Fontana ch. 9; Zheng & James 2010 |
| Crumpling | Power-law event energies, Poisson timing, shrinking facets | `crumple` `paper` | `contact`, `foley` | Rocchesso & Fontana ch. 9 |
| Rolling | Surface profile, rolling filter, impact with ball inertia | `roll` `ball_roll` | `contact`, `pinball` | Rocchesso & Fontana ch. 9 |
| Scraping, sliding | 1/f^beta surface profile read at sliding speed into a modal body | `scrape` `saw` `guiro` | `contact`, `foley` | van den Doel 1998 |
| Stick-slip | Elasto-plastic friction, force-driven creak | `squeak` `brake_squeal` `rub` `stick_slip` `creak` `door` | `friction`, `foley` | Rocchesso & Fontana ch. 8; Farnell |
| Many particles | PhISEM stochastic collisions in a resonant shell (25 presets) | `shake` `shaker` `tambourine` `cabasa` `sekere` `maraca` `sleigh_bells` `bamboo_chimes` | `physical.particles`, `foley` | Cook 2002; STK Shakers |
| Footsteps | Ground-reaction force into granular or modal ground (24 grounds), gait sequencer | `footstep` `footsteps` | `physical.footsteps`, `creatures` | Cook 2002; Visell et al. 2009; Farnell |

## Liquids, gases, fire, electricity, ice

| Phenomenon | Model | Names | Module | Source |
|---|---|---|---|---|
| Bubble | Minnaert resonance, thermal + viscous + radiation damping, pitch rise near the surface | `bubbles` `drip` `drops` `gurgle` `fizz` | `matter` | Zheng & James 2009; Moss et al. 2010 |
| Water in bulk | Bubble populations by rate and size (pour ~1600/s, brook ~3100/s), vessel resonance rising with fill | `pour` `babble` `stream` `splash` `drain` `boil` `waterfall` `surf` `cave_drips` `rain` `tide` | `matter`, `physical.ambience` | Zheng & James 2009; Farnell 2010 |
| Turbulent jet | Strouhal peak proportional to velocity over diameter | `steam` `air_leak` `spray` `kettle` `balloon` | `matter` | Lighthill (level law unsourced in the findings) |
| Wind | Vortex shedding around obstacles, gusting speed process | `wind` `gust` | `physical.ambience`, `matter` | Farnell 2010 |
| Aeolian tone | Reynolds-dependent Strouhal shedding along a moving edge | `swoosh` `sword` `whip` | `physical.modal`, `foley` | Selfridge et al. 2017 |
| Combustion | Flame-flux derivative below 180 Hz, f^-alpha noise extension above; crackle events | `flame` `fire` `match` | `matter` | Chadwick & James 2011 |
| Explosion, gunshot | N-wave, expansion rumble, debris, distance and reflections | `explosion` `gunshot` | `physics`, `foley` | Farnell 2010 |
| Thunder | Segmented channel with distance-dependent propagation | `thunder` `lightning` | `physical.ambience`, `matter` | Farnell 2010 |
| Electricity | Spark snap, mains-modulated arc, hum with harmonics | `spark` `arc` `mains_hum` `tesla` `neon` | `matter` | Farnell 2010 |
| Ice | Flexural-wave dispersion chirp of a thin plate, brittle cracks | `ice_crack` `ice_cubes` `freeze` | `matter` | Textbook plate dispersion (constants unsourced) |
| Engines, rotors | Firing pulses into exhaust resonances; blade-passing tones plus tip turbulence | `car_engine` `klang_car` `toy_boat` `harrier` `jet_engine` `helicopter` `fan` `propeller` `electric_motor` `gears` `modal_engine` `bicycle` | `vehicle`, `physics`, `klang`, `contact` | Farnell 2010; Klang procedural library |

## Propagation, rooms and signal processing

| Topic | Model | Names | Module | Source |
|---|---|---|---|---|
| Rooms | Feedback delay network with per-band T60, image-source early reflections, Sabine decay from dimensions and materials | `room` `fdn_reverb` `early_reflections` `zita` `reverb` | `reverbs` | Smith, PASP; Jot |
| Classic reverbs | Schroeder networks with the published delay lengths | `jcrev` `satrev` | `reverbs` | Smith, PASP |
| Plate, spring | Driven Kirchhoff plate; dispersive allpass chain | `plate_reverb` `spring_reverb` | `plates`, `fdstring` | Bilbao ch. 12 |
| Measured or generated spaces | FFT convolution with an impulse response | `convolve` | `dsp` | Smith, DSP Guide |
| Distance, motion | 1/r, air absorption, Doppler fly-by, occlusion (mass law), wall transmission | `distance` `doppler` `occlude` `wall` | `reverbs`, `dsp` | Smith, PASP |
| Ducts, cavities | Webster tube as an effect; box and sphere cavity modes | `tube` `cavity` | `tubes`, `contact` | Bilbao ch. 9; Rocchesso & Fontana |
| Rotary speaker, modulation | Doppler rotors; true-feedback flanger, allpass phaser, chorus, tape delay | `leslie` `flanger` `phaser` `chorus` `tape_delay` `shimmer` | `reverbs` | Smith, PASP |
| Time and pitch | Phase vocoder with phase locking and transient handling, SOLA, formant-preserving shift | `timestretch` `pitch_shift` `harmonizer` `freqshift` | `spectral` | Smith, SASP |
| Analysis and resynthesis | Sinusoids + noise model from a recording | layer `resynth`, `analyze` / `resynth` (Python) | `spectral` | Smith, SASP; Serra |
| Spectral effects | Channel vocoder, cross synthesis, spectral gate, freeze, blur | `vocoder` `cross_synth` `spectral_gate` `freeze` `spectral_blur` `robotize` `whisperize` | `spectral` | Smith, SASP |
| Filters | True Butterworth / Chebyshev / Bessel cascades, windowed-sinc FIR, EQ from a curve | `steep_lowpass` `steep_highpass` `steep_bandpass` `fir_eq` | `dsp` | Smith, DSP Guide; Smith, Introduction to Digital Filters |
| Nonlinearity | Oversampled waveshaping, wavefolding, tape | `overdrive` `wavefold` `tape` `distortion` | `dsp` | Smith, DSP Guide |
| Lo-fi and codecs | Mu-law, ADPCM, CVSD, dither, sample-rate reduction with hardware presets | `mulaw` `codec` `dither` `lofi` `bitcrush` | `dsp` | Smith, DSP Guide |

## Synthetic and musical

| Topic | Model | Names | Module |
|---|---|---|---|
| Retro effect generators | sfxr / Bfxr engine (sample-exact with jsfxr), ZzFX | `sfxr` `zzfx` `sfxr_voice` `zzfx_voice` | `retro` |
| Sound chips | AY-3-8910 tone dividers, LFSR noise, hardware envelopes | `psg` `chip` `chiptri` `board` | `retro`, `instruments` |
| Subtractive, FM, additive | Band-limited oscillators, key-tracked filters, capped FM | `synth` `lead` `pad` `bass` `acid` `epiano` and the rest of the synth family | `instruments`, `synths` |
| Composition | Scales and tunings, progressions, voice leading, Euclidean and style grooves, melody and bass generators, arrangement, stingers | layers `compose` `arrangement` `euclid` `arp` `stinger` `scatter` `adaptive` | `compose` |
| Tracker | Pattern / order song format, ZzFXM import | layer `song` | `retro` |
| Sampling | Folder of recordings mapped to notes; articulations | layers `sampler` `articulate` | `klang` |

## Known limits

- **Bowed strings and brass in the default catalog are hybrids.** `violin`, `cello`, `trumpet`, `trombone`,
  `french_horn`, `tuba` and `flute` mix the waveguide with an additive voice at the note's pitch, because the
  raw waveguides do not hold pitch across their range. `bowed_string_fd` is fully physical and in tune but
  takes 0.1-0.4 s to speak at ordinary bow force.
- **Nonlinear plates are flat rectangles**, not curved shells: the cymbals are darker than real ones.
- **`timpani`, `glockenspiel`, `tubular_bell`** add a sine at the note's pitch to a modal body that is not
  tuned to it.
- **No 3-D geometry**: no finite-element modes of arbitrary meshes, no fluid simulation; the fire and water
  models are driven by stochastic surrogates of the simulations in the source papers.
- **No helical-spring model**: `spring_reverb` is a dispersive allpass loop.
- Constants without a source are marked `# UNSOURCED` in the code and listed at the end of each module page.
