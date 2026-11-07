# Plates and membranes (`genny/plates.py`)

2D finite-difference models from Bilbao, *Numerical Sound Synthesis* (2009), ch. 10-13: a grid of points
is stepped sample by sample, so strike position, edge conditions, mallet hardness and (for the nonlinear
plates) strike force change the sound the way they do on the real object.

| type | name | what it is |
|---|---|---|
| drum | `fd_drum` | circular drum head, 2D wave equation, air-loaded |
| drum | `plate_crash`, `plate_ride`, `plate_china` | cymbals from a nonlinear (von Karman) plate |
| drum | `plate_gong`, `tam_tam` | gongs from the same plate: pitch glide, bloom |
| drum | `thunder_sheet` | large thin steel sheet, shaken |
| sfx | `metal_plate`, `glass_pane`, `wood_panel` | struck linear plates with material constants |
| sfx | `plate_bow` | bowed plate edge (bowed cymbal, waterphone, metal scrape) |
| fx | `plate_reverb` | the signal drives a lossy steel plate, two pickups give stereo |
| fx | `room2d` | 2D wave-equation room with rigid or absorbing (`absorb`) walls |

The existing `crash`, `ride`, `gong`, `tom` drums and the `timpani` instrument are untouched.

## Cost and `quality`

Every model runs at its own internal sample rate and is resampled to the output rate. `quality` scales
that rate (and with it the grid): 1 is the default, 0.5 is a fast draft with half the bandwidth, 2 opens
the top end. Measured on one core after JIT warm-up, seconds of CPU per second of sound at `quality` 1:
`fd_drum` 0.5, `metal_plate` 0.6, `glass_pane` 0.3, `wood_panel` 0.5, `plate_bow` 0.4, `plate_gong` 0.7,
`plate_ride` 1.6, `tam_tam` 1.7, `thunder_sheet` 1.6, `plate_china` 2.3, `plate_crash` 2.7,
`plate_reverb` 1.2, `room2d` 1.1. The first call of a session compiles the kernels (about 10 s, cached on
disk afterwards).

Internal rates: `fd_drum` 70 x gamma (6-32 kHz, about 45 x the fundamental); linear plates up to 32 kHz;
nonlinear plates `4 x kappa x 900 points`, capped at 24 kHz (crash, ride, gong 24 kHz; china 19.5 kHz;
tam-tam 12.5 kHz); `plate_reverb` 16 kHz (wet band 8 kHz); `room2d` at most 8 kHz (wet band 4 kHz).

## `fd_drum`

Circular head on a square grid with a staircase rim. Pitch is not a parameter, it follows from the head:
`f01 = 0.765 * sqrt(tension / 0.26 kg/m2) / size` (1000 N/m, 0.33 m: 144 Hz without air load).

| param | default | meaning |
|---|---|---|
| `tension` | 1000 | N/m. Pitch goes with its square root |
| `size` | 0.33 | head diameter, m |
| `decay` | 0.6 | T60 of the fundamental, s |
| `damping` | 0.5 | 0..1 extra loss at high frequency |
| `pos` | 0.35 | strike position, 0 centre (thud, few overtones) .. 1 rim |
| `hardness` | 0.5 | 0 soft felt (wide, 6 ms contact) .. 1 hard stick (narrow, under 1 ms) |
| `air` | 0.5 | 0..1 air-cavity load. Raises only the modes that change the enclosed volume: at 1 the fundamental of the default drum goes from 142 to 222 Hz while the (1,1) mode stays at 227 Hz, which is how a kettledrum gets its pitch |
| `depth` | 0.3 | shell depth, m (cavity volume) |
| `stiffness` | 0 | 0..1 bending stiffness: stretches the upper partials |
| `contact` | 0 | 1 = power-law mallet contact instead of a force pulse: contact time and brightness then follow `vel` |
| `scan` | 0 | Hz, moving pickup (slow phasing) |
| `force` | 200 | N peak stick force at `vel` 1 (with `berger`: how far the pitch is pushed up) |
| `berger` | 0 | 1 = tension modulation (Mylar head): a hard hit (`force` N) starts sharp and settles as it rings |
| `detune` | 0 | 0..0.3 tension anisotropy: splits the modes into beating doublets (0.07 = a real head) |
| `under`, `flip` | 0, 0 | second mic under the head, as on `tom` / `snare` below |
| `snare` | 0 | 1 = snare wires under the head: energy-conserving collisions give the buzz |
| `snare_gap` | 0.0 | m, rest gap between wires and head (negative = pressed in) |
| `quality` | 1 | see above |

```json
{"type": "drum", "kind": "fd_drum", "params": {"tension": 600, "size": 0.4, "pos": 0.2, "hardness": 0.3, "decay": 0.5}}
{"type": "drum", "kind": "fd_drum", "params": {"tension": 3000, "size": 0.2, "pos": 0.8, "hardness": 0.9, "air": 0.2, "decay": 0.3}}
{"type": "drum", "kind": "fd_drum", "params": {"tension": 1500, "size": 0.6, "air": 1.0, "depth": 0.45, "hardness": 0.2, "decay": 1.8, "damping": 0.3}}
```
(a floor tom, a high bongo-like head, a kettledrum.) `plates.membrane(snare=True, snare_gap=)` is the same in
Python; `collision_step(a0, b, g, kap, al)` solves one collision update (discrete gradient: r + g(Φ(r+a) − Φ(a))/r
+ b = 0, Φ = κ/(α+1)[η]₊^(α+1)) and `simulate_snare(mdl, force, exc, pickups, size, gap=0.0)` runs the head with
its wires. `plates.membrane(..., stereo=True)` in Python returns
two pickups.

## Nonlinear plates: `plate_crash`, `plate_ride`, `plate_china`, `plate_gong`, `tam_tam`

A bronze plate whose bending is coupled to its own stretching (von Karman equations, an Airy stress
function solved on the grid every sample). Three regimes as `force` rises: a clean linear ring; then a
pitch that starts sharp and glides down; then new partials and a noisy wash that keeps building after the
stick has left. The nonlinearity goes with `force / (thickness^2 x size^2)`.

| param | crash | ride | china | gong | tam_tam | meaning |
|---|---|---|---|---|---|---|
| `size` | 0.4 | 0.5 | 0.45 | 0.5 | 1.0 | diameter, m |
| `thickness` | 1.0 | 2.0 | 0.8 | 4.0 | 2.5 | mm |
| `force` | 400 | 600 | 400 | 6000 | 2000 | peak strike force in N at `vel` 1 (scales with `vel`^2) |
| `decay` | 6.9 | 7 | 3 | 9 | 75 | T60 of the low modes, s. Cymbals (and `shell_crash`) follow the crash-cymbal damping law μ = 0.007 ω^0.6 + 2: high modes die faster, T60 at 1 kHz = 0.6 x `decay`; tam-tam 4 s at 8 kHz |
| `dur` | 2.2 | 2.2 | 1.6 | 4 | 8 | rendered length, s |
| `bright` | 0.6 | 0.6 | 0.6 | 0.6 | 0.6 | output lowpass 1.5 kHz (0) .. 12 kHz (1); 0.6 = 5.2 kHz |
| `model` | vk | | | | | `vk`, `vk_simple`, `berger`, `linear` |
| `quality` | 1 | | | | | |

Fixed per preset: strike point and contact time (crash 12 ms at (0.2, 0.3); ride 2.5 ms; china 6 ms near
the edge on a 1.3:1 plate; gong 3 ms at the centre; tam-tam 10 ms off centre) and the high-frequency loss.
Cymbals are read as surface acceleration above 150 Hz, gongs as velocity.

Measured on the crash preset (centroid of the output in the first 80 ms, then 80-250 ms):
`force` 30: 293 then 319 Hz; 110: 484 then 563 Hz; 400: 1144 then 1262 Hz; 1200: 3127 Hz, already past
the point where the stress guard acts.

`model`:
- `vk` (default): von Karman, energy-conserving variant of the book's scheme. The book's explicit scheme
  (13.10) evaluates the stress term at the current step; here it is averaged over the next and previous
  step, which is still one linear solve per sample but conserves energy (drift 1e-12 over 1000 steps at
  12 thicknesses of amplitude, against 7e-3 for the explicit scheme) and does not blow up. This variant
  is not printed in the book; it was derived for this module and checked numerically.
- `vk_simple`: the book's scheme 13.10 as printed. Kept for comparison; it goes unstable when hit hard, as
  the book warns, and the amplitude guard then clamps it.
- `berger`: tension modulation only. Pitch glide, no new partials, no crash; about 5 x cheaper.
- `linear`: the same plate with the nonlinearity off.

Limits, stated plainly:
- The plate is flat, rectangular and simply supported. The book's presets use free edges, and real cymbals
  are curved shells (book sec. 13.3), which is what crowds their mid range. Neither is on this grid, so
  these are darker than a real cymbal: the default crash has its centroid near 1.2 kHz. The modal engine below
  (`vk_cymbal`, `vk_gong`) has both.
- Two guards keep extreme settings finite: the in-plane stress is capped when it would exceed what the
  time step can represent, and displacement is limited to 50 scaled units. Past roughly 3 x the default
  `force` the first one acts and more force stops adding brightness.

```json
{"type": "drum", "kind": "plate_crash", "vel": 1.0, "params": {"force": 600, "dur": 2.5}}
{"type": "drum", "kind": "plate_crash", "vel": 0.5, "params": {"thickness": 0.8, "bright": 0.8}}
{"type": "pattern", "kind": "plate_ride", "steps": "x-x-xXx-", "step": 0.5, "bpm": 120, "params": {"dur": 1.2, "force": 500}}
{"type": "drum", "kind": "plate_gong", "params": {"force": 20000, "dur": 5, "decay": 10}}
{"type": "drum", "kind": "tam_tam", "params": {"size": 1.2, "force": 4000, "dur": 6}, "fx": [{"type": "plate_reverb", "mix": 0.2}]}
```

## Modal von Karman: `vk_cymbal`, `vk_gong`

`physical/vkmodal.py` solves the von Karman plate or shallow shell on its modes instead of a grid: Rayleigh-Ritz
modes of the strain energy (variable thickness, any edge), clamped Airy modes, the coupling tensor H and an
imperfection w0, run by an SAV (explicit, energy-stable) or a conservative (energy-conserving) scheme with the
exact linear update.

- `vk_cymbal`: measured thickness profile (thick bell, thin edge), shallow-shell `dome`, free edge, power-law modal
  damping (μ = 0.007 ω^0.6 + 2); the crash wash builds up from the hit. `kind` crash | ride | splash | china,
  `dome` (rise of the cap in m, -1 = the kind's own, 0 = flat).
- `vk_gong`: `kind` opera (flat: the pitch glides down as it rings), curved (small domed gong: glides up),
  tam_tam (swells into a shimmer), rectangle (the supported 0.4 x 0.6 m steel test plate).
- Both: `force` (N at `vel` 1, 0 = the kind's own), `dur` 2.5, `bright` 0.6, `modes` 48 (more = brighter wash,
  slower; cost ~ modes²), `scheme` sav | conservative | linear.
- `vk_plate(...)` in Python is the function behind both: `profile`, `dome`, `edges`, `law`, `basis` (grid |
  ritz), `radiation` (radiation loss of each mode, on by default), `distance` (dipole near field).
- Helpers: `modal_density`, `prestressed_modes`, `coincidence`, `radiation_efficiency`, `radiation_damping`,
  `near_field`, `point_force_pressure`, `PLASTICS` / `rayleigh_t60`. `radiation_efficiency` and
  `radiation_damping` take `stiffeners` (total length in m of ribs fixed on the plate, default 0): below the
  coincidence frequency they multiply the radiation efficiency by 1 + 2 L/P (P the perimeter), capped at 1.

## `bodhran`

Irish frame drum: a slack goatskin head with no shell cavity, played with a tipper. `pos` 0.3 (0 centre, deep
thud .. 1 rim, ringing overtones), `damp` 0 (the left hand pressing the back of the skin: shorter, duller, a
little higher), `size` 0.46 m, `tension` 400 N/m.

## `polar_shell`

Cymbal or gong as a shallow spherical shell on a polar finite-difference grid (`physical/fdgrid.py`): free circular
rim, clamped or free centre, von Karman nonlinearity, so a hard hit crashes. `kind` cymbal | flat | deep | gong,
`q` curvature (−1 = the kind's; more curvature empties the bottom of the spectrum), `pos` strike radius 0..1 (−1 =
the kind's), `dur` 2 s, `fs` internal rate 32000 Hz, `rings` 16 radial grid points. `zone` bell | bow | edge: strike
at 0.12 / 0.5 / 0.92 of the radius on a domed, tapered cymbal (`percussion.cymbal_thickness(r)`, profile
`CYMBAL_TAPER`), with the force scaled by (hc/h(r))³ (`percussion.zone_drive(r)`): the thin edge is louder and
brighter. Default `""` = as before.

## `perrin_bell`

Church bell from the measured modes of one real bell: 63 modes up to 9.2 kHz, the split doublets included (slow
beats on the partials). `tune` = the hum in Hz (292.72), `dur` 8 s.

## `tom`, `snare`: double miking

`under` 0..1: level of a second mic under the drum (the head seen from below: inverted, later, darker). `flip` 1 =
that mic's polarity flipped back (full lows); 0 = raw, the lows partly cancel.

## `thunder_sheet`

A 1.1 m, 0.6 mm steel sheet driven by an irregular shake (`rate` per second, pattern from `seed`) instead
of a strike; the nonlinear plate turns the slow push into rumble and crackle. Params: `size` 1.1 m,
`thickness` 0.6 mm, `force` 25 N, `rate` 5 /s, `dur` 3 s of shaking, `decay` 4 s, `bright` 0.6, `seed` 0,
`quality` 1.

```json
{"type": "drum", "kind": "thunder_sheet", "params": {"dur": 4, "force": 40, "rate": 7, "seed": 2}}
```

## Struck plates: `metal_plate`, `glass_pane`, `wood_panel`

Linear Kirchhoff plate, struck with a short force pulse, read as surface velocity. Pitch follows from
`kappa = sqrt(E H^2 / (12 rho (1 - nu^2))) / L^2`: the lowest modes sit a few `kappa` Hz apart, so
doubling the thickness doubles every frequency and doubling the size quarters them.

| param | metal_plate | glass_pane | wood_panel | meaning |
|---|---|---|---|---|
| `size` | 0.5 m | 0.5 m | 1.0 (relative) | side length (square root of the area) |
| `thickness` | 2 mm | 4 mm | 1.0 (relative) | |
| `decay` | 3.0 | 0.9 | 0.35 | T60 at 200 Hz, s (shorter above) |
| `aspect` | 1.3 | 1.4 | fixed 1.248 | side ratio |
| `edges` | free | supported | supported | `metal_plate` only: `free`, `supported`, `clamped` |
| `pos` | 0.45 | | | strike point, 0 centre .. 1 corner |
| `hardness` | 0.6 | | | 0 soft (5 ms contact) .. 1 hard (0.5 ms) |
| `vel` | 0.8 | | | strike strength: harder is shorter and brighter |
| `bright` | 0.5 | | | output lowpass 1.5 .. 12 kHz |
| `quality` | 1 | | | |

Materials: steel and glass from `genny.physics.MATERIALS`. `wood_panel` is orthotropic, stiff along the
grain: the wooden plate of the book's Table 12.2 (lowest mode 56.5 Hz at `size` 1), scaled by
`thickness / size^2`. The explicit scheme mistunes high modes downward (a semitone by the 6th mode), so
treat these as unpitched objects.

```json
{"type": "sfx", "kind": "metal_plate", "params": {"size": 0.8, "thickness": 1.5, "decay": 4, "hardness": 0.9}}
{"type": "sfx", "kind": "metal_plate", "params": {"size": 0.25, "thickness": 6, "edges": "clamped", "decay": 0.6}}
{"type": "sfx", "kind": "glass_pane", "params": {"size": 0.3, "thickness": 3, "hardness": 0.8}}
{"type": "sfx", "kind": "wood_panel", "params": {"size": 0.5, "decay": 0.2, "hardness": 0.3}}
```

## `plate_bow`

A bow drawn across the free edge of a bronze plate whose other three edges are clamped. `force` 10 gives a
pitched tone, more force picks out a higher plate mode and adds scrape. Params: `dur` 2 s, `size` 0.3 m,
`thickness` 1.7 mm, `force` 15 (5..60, the book's scaled bow force), `speed` 1.0, `pos` 0.5 along the edge,
`decay` 4 s, `bright` 0.5.

```json
{"type": "sfx", "kind": "plate_bow", "params": {"dur": 3, "force": 12, "pos": 0.5}}
{"type": "sfx", "kind": "plate_bow", "params": {"dur": 1.5, "force": 50, "pos": 0.3, "size": 0.45}}
```

## `plate_reverb`

The input drives one point of a lossy steel plate with free edges (aspect 2, the book's Fig. 12.9
positions); two pickups give left and right. Compared with a room: no early reflections, an even, dense
tail, and a faint downward chirp on transients because high frequencies travel faster in a plate.

| param | default | meaning |
|---|---|---|
| `mix` | 0.3 | wet amount |
| `t60_low` | 3.0 | decay time at 500 Hz, s (0.3..10) |
| `t60_high` | 1.5 | decay time at 2 kHz, s |
| `size` | 1.0 | 0.4..1.5. 1 = the 2 m plate (`kappa` 2, modes 4 Hz apart); smaller is sparser and more metallic, and cheaper |
| `damping` | 0 | 0..1 damper pad: shortens both decay times |
| `predelay` | 0 | s |
| `width` | 1.0 | stereo width; 0 keeps a mono input mono |
| `tail` | auto | seconds appended (0.8 x `t60_low`) |
| `quality` | 1 | internal rate 16 kHz x quality: the wet signal has no content above half of it |

Measured: requested 2.0 s / 1.0 s gives 500 Hz and 2 kHz band decay times within 20 %.

```json
{"fx": [{"type": "plate_reverb", "mix": 0.25, "t60_low": 2.5, "t60_high": 1.2}]}
{"fx": [{"type": "plate_reverb", "mix": 0.4, "size": 0.5, "t60_low": 1.2, "t60_high": 0.5, "predelay": 0.02}]}
{"fx": [{"type": "plate_reverb", "mix": 0.5, "t60_low": 6, "t60_high": 4, "quality": 1.5}]}
```

## `room2d`

The 2D wave equation on a floor plan with rigid walls, one source and two listeners. It has real room
modes and flutter, but it is two-dimensional and band-limited: use it as a small-room colour, not as a
hall. Params: `mix` 0.3, `size` 5 m (2..15), `aspect` 1.4, `t60` 0.8 s, `damping` 0.5 (extra
high-frequency absorption), `predelay` 0, `width` 1, `tail` auto, `quality` 1, `absorb` 0 (wall absorption 0..1 at
normal incidence, one value or four for the walls x0, x1, y0, y1: impedance walls; Python `wall_impedance(alpha)`,
`wall_damping`, `_model(..., wall=)`). Rooms above about 7 m lower
the internal rate further to keep the grid under about 11 000 points.

```json
{"fx": [{"type": "room2d", "mix": 0.3, "size": 3, "t60": 0.4}]}
{"fx": [{"type": "room2d", "mix": 0.4, "size": 8, "aspect": 2.2, "t60": 1.5, "damping": 0.2}]}
```

## Python API

`nl_plate(edges="supported")` or `"free"`: von Kármán plate with free edges (Φ clamped, the force on u the adjoint
of the bracket, energy still conserved; no `dome`); `simulate_nl` accepts a "ffff" model.
`plates.membrane`, `plates.struck_plate`, `plates.nl_plate` are the functions behind the catalog names,
with every physical constant exposed (`nl_plate(..., drive=fn)` takes an arbitrary force signal, e.g. a
roll). Lower level: `_model(fs, eps, bc, kappa, gamma, sig0, sig1, nu, ortho, circle, cav)` builds a grid
(`bc` is four letters for the edges x=0, x=max, y=0, y=max: `f` free, `s` supported/fixed, `c` clamped),
`simulate` / `simulate_nl` run it (`simulate_nl(airy="dst")`: two sine-transform Poisson solves with Φ held at
the edge, faster on big grids; default `"chol"`; `nl_plate(scan_order=3)` reads the moving pickup with 4-point
interpolation instead of linear), `loss_coeffs` turns two (Hz, T60) pairs into the loss constants,
`plate_modes` / `membrane_modes` give the analytic mode frequencies and `scheme_modes` the ones the grid
actually produces. `thermoelastic(freqs, thickness, material="steel")` gives the Zener thermoelastic loss factor and
decay rate of a thin metal or glass plate (peak at 1/(2π τ), τ = ρ C h²/(k π²)); a library function, the presets
do not use it. `apnl(x, a1, a2)`: Faust's nonlinear allpass, a first-order allpass whose coefficient is a1 while the
input is positive and a2 otherwise. `wgr(f, n, sr, r)`: waveguide resonator returning a quadrature pair of unit
amplitude when r = 1.

## What differs from the book

- T60: the book's listings set `sigma = 6 ln10 / T60`, which decays 60 dB in half the stated time (a
  requested 4 s measured 1.9 s). This module uses `3 ln10 / T60`, so the times above are real 60 dB times;
  the book's `sigma0 = 1.38` is `decay` 5 here.
- Free and clamped edges are built from a discrete strain energy rather than from the book's edge
  stencils (most of which are left as exercises). Lowest free-square-plate modes come out 0.3-0.9 % under
  the published values; lossless energy is conserved to 1e-13.
- No modal fast path was needed: the linear plates are cheap enough on the grid.
- Not implemented: circular (polar-grid) plates, the curved-shell cymbal, string-soundboard coupling,
  free edges for the nonlinear and the wooden plate.
