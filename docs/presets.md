# Presets (`genny/presets.py`, `genny/presets/*.toml`)

A preset is a named set of values read from a TOML file. Weapons, cartridges, cases, engines, vehicles, scenes, doors,
horns, ships and other tables of parameters are presets, so a new object is a text file, not code. Nothing about a sound
changes until a preset is selected.

## Format

```toml
schema = "genny.preset/1"
family = "two_stroke"

[default]
rpm = 7000.0

[mx125]
rpm = 11500.0
ports = "82,113,37.8,30,54.5,110,54"
induct = "reed"

[my_kart]
extends = "mx125"       # inherit everything, change what follows
rpm = 12000.0
```

- `schema` is `genny.preset/1`; `family` names the table (see below); each `[name]` is one preset (letters, digits, `_`,
  `.`, `-`). `extends = "other"` inherits the fields of another preset of the same family. Fields not listed keep the
  family default.
- Values are numbers (finite), strings (at most 256 characters), booleans, arrays (at most 64 items), small tables, and for
  some families an array of tables (`scene`: `layers = [...]`). Units are in the comment at the top of each shipped file.
- Every file is validated against the schema its family declares: an error names the file, the table and the key
  (`weapon.toml [my_gun].barrle`: unknown field, did you mean `barrel`?). Files over 1 MB or with over 2000 presets are
  refused.

## Where presets come from

Later sources win over earlier ones: the shipped files (`genny/presets/`), the user folder (`%APPDATA%\genny\presets` on
Windows, `~/.config/genny/presets` elsewhere, `GENNY_PRESETS_USER` to move or disable it), the folders in
`GENNY_PRESETS` (path-separated), `presets.register(path)` in Python, and the `"presets"` key of a spec (a folder or file
path, or a list of them). `genny presets path` prints the sources.

## Command line

```
genny presets list [family]            # every preset and where it comes from (builtin, user, env, spec); --json
genny presets show <family> <name>     # the resolved fields (after extends)
genny presets check [file-or-folder]   # validate; exit code 1 on error
genny list presets                     # the families and their names, as JSON with --json
```

## Families

| Family | Used by |
|---|---|
| `weapon`, `cartridge`, `case`, `case_dim`, `case_extra`, `case_of`, `gun_legacy` | `firearm`, `minigun`, `bullet_pass`, `bullet_impact`, `casing`, `foley.GUNS` |
| `weapon_env` | `env=` of `firearm`, `minigun`, `bullet_pass`, `bullet_impact`: the venue field (`open_field` default, `forest`, `urban`, `range`, `indoor`) |
| `two_stroke`, `engine_layout`, `diesel`, `tracks`, `moto_2t`, `scooter`, `garden` | `two_stroke`, `engine_layout`, `diesel_engine`, `tracks`, `motorcycle`, `scooter`, `garden_machine` |
| `ev`, `avas_style`, `interior`, `air_leak`, `door`, `horn`, `tyre_surface` | `road` sounds |
| `rail_kind`, `ship`, `lap` | `rail`, `ship`, `lap` |
| `scene` | `scenes.scene kind=` (49 ambiences) |
| `lofi`, `spring`, `compressor`, `free_reed`, `bearing` | `lofi`, `spring_reverb`, `compressor preset=`, free reeds, `bearing preset=` |
| `patch` | a whole sound under a new name |

A selector that takes a name also takes an inline dict: `firearm weapon={"extends": "ar15", "barrel": 0.4}`,
`two_stroke kind={"extends": "mx125", "rpm": 12000.0}`, `tracks kind=`, `scene kind=`.

## Examples

A cartridge-and-weapon pair of your own, used in a spec:

```toml
family = "weapon"
schema = "genny.preset/1"
[my_carbine]
extends = "ar15"
barrel = 0.4
```

```json
{"out": "shot.wav", "presets": "my_presets/", "layers": [{"type": "sfx", "kind": "firearm", "params": {"weapon": "my_carbine", "distance": 50}}]}
```

A scene of your own (`scene` family: a `bed` and `layers`; every layer has `sfx` or `inst`, `db`, `wide`, `every`, `notes`, `params`):

```toml
family = "scene"
schema = "genny.preset/1"
[my_yard]
bed = "open"
layers = [
  {sfx = "noise_bed", db = -12, wide = true, params = {kind = "rumble"}},
  {sfx = "gust", db = -4, every = [6, 12], params = {through = "gap"}},
]
```

A patch: a bus, a tram or a guitar played with a pick, as a new sound name.

```toml
family = "patch"
schema = "genny.preset/1"
[tram]
sfx = "rail"
desc = "city tram"
params = {kind = "tram_roll"}
```

`genny sfx tram`, `genny render` and `genny synth` resolve it; `genny list --json` marks it `"origin": "patch"`. A patch
may not reuse the name of an existing sound and may only set parameters its parent declares.

## Adding a preset to the code base

Add a table to the family's TOML file (or a new family: declare its schema in the module that uses it), then run
`examples/preset_freeze.py` to check that the migrated tables still render bit for bit as before.
