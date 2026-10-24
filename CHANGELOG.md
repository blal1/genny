# Changelog

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
