# sports — balls, whistles, skates, swimming (`genny/sports.py`)

Every name and parameter is in [`catalog.md`](catalog.md) ("Index by module"); this page gives the physics behind
the entries whose model is not obvious from their parameters.

## Models

- **`swim`**: kicks are counted per arm cycle, as swimmers do: six in crawl and backstroke, two in butterfly, none
  in breaststroke (`swim_rates(stroke, rate)` returns arm entries and kicks per second; crawl kicks 5 times a
  second).
- **`sport kind=bowling`**: the ball rolls with the resistance of a Hertz contact of resin on maple, a patch of
  1.24 mm radius (`hertz_patch(mass, radius, e1, nu1, e2, nu2)` = (3 m g R / 4 E*)^(1/3)).

## Python

`swim_rates(stroke, rate=1.0)`, `hertz_patch(mass, radius, e1, nu1, e2, nu2)`. Sources of every constant:
`research/SOURCES.md`.
