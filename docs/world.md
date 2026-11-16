# World instruments (`genny/world.py`)

Plucked, bowed, blown and struck instruments of Chinese, Arab, Persian, Mongolian, Japanese, Indian, Turkish,
Javanese and West African traditions, built from the physical strings, bows, tubes and bars of the other modules.
`genny list instruments` has the parameters.

| Name | What it is | Notable parameters |
|---|---|---|
| `guzheng` | Chinese zither: bridges under each string, pitch bends by pressing the string behind the bridge | `position`, `press_mm`, `tension`, `t60`, `ring` |
| `oud` | fretless short-neck lute, courses C2 F2 A2 D3 G3 C4 (Arabic tuning, 61 cm; F2 was D2 before) | the common options below |
| `qanun` | trapezoid zither with mandals (microtonal levers), 26 courses of 3, range A2-E6 (was D3-A6) | `mandal` |
| `santur` | hammered dulcimer | |
| `erhu` | two-string spike fiddle with a snakeskin | `kind` erhu, gaohu, zhonghu, yehu; `force`, `speed`, `bow_dist`, `skin_tension` |
| `morin_khuur` | Mongolian horsehead fiddle, free string length 0.60 m (was 0.65) | |
| `shakuhachi` | Japanese end-blown flute | |
| `ney` | Persian and Turkish end-blown reed flute | |
| `bansuri` | Indian side-blown bamboo flute | |
| `duduk` | Armenian double-reed | `cone` |
| `bonang` | Javanese gong chime | `kind`, `ombak` (the beat between the paired instruments) |
| `balafon` | West African gourd xylophone | `buzz`, `gourd` |

All of them except `bonang`, `balafon` and `santur` take the same expressive options: `bend` (cents), `bend_at`
(fraction of the note), `bend_time`, `scoop` (cents the note starts below the pitch), `scoop_time`, `vib` (cents),
`vib_hz`, `vib_delay`.

Checked online (scales and structure, mostly from encyclopedic sources): erhu D4 A4, santur 4 strings per course, shakuhachi
D4-D6, ney, duduk, balafon structure and the guzheng pentatonic tuning agree. Tensions, Q, gains, skin radii, the bonang
ratios and the donkey bray figures stay `connaissance` (no open measurement found).

Level trims for these instruments are not in `_levels.py` yet (the peak is normalised to 0.8 x (0.35 + 0.65 vel)).
