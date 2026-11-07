"""Two @sfx with one name: the second silently replaces the first (sfx.py), so a new name must be new."""
import collections
import pathlib
import re

import genny

# Later models that replace a core sound on purpose, keeping its old parameters.
REPLACED = {"door", "engine", "explosion", "fire", "footstep", "heartbeat", "hit", "pour", "roll", "scrape", "swoosh", "thunder", "wind"}


def test_no_accidental_duplicate_sfx_names():
    root = pathlib.Path(genny.__file__).parent
    names = collections.Counter(n for f in root.rglob("*.py") for n in re.findall(r'@sfx\("([a-z_0-9]+)"', f.read_text(encoding="utf-8")))
    assert {n for n, c in names.items() if c > 1} <= REPLACED
