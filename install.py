#!/usr/bin/env python3
"""Installs the genny Claude Code skill (and optionally the `genny` command) on Windows, macOS and Linux.

    python install.py              copy skill/genny into ~/.claude/skills/genny
    python install.py --tool       also install the `genny` command with uv (editable, from this clone)
    python install.py --link       symlink the skill instead of copying (edits in the repo apply at once)
    python install.py --uninstall  remove the installed skill
    python install.py --dest DIR   install into DIR instead (the folder that will hold SKILL.md)

The skills folder is ~/.claude/skills (or $CLAUDE_CONFIG_DIR/skills when that variable is set).
Standard library only; run it with any Python 3.8+.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SOURCE = REPO / "skill" / "genny"


def skills_dir() -> Path:
    base = os.environ.get("CLAUDE_CONFIG_DIR")
    return (Path(base).expanduser() if base else Path.home() / ".claude") / "skills"


def remove(dest: Path) -> bool:
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
        return True
    if dest.is_dir():
        shutil.rmtree(dest)
        return True
    return False


def install_skill(dest: Path, link: bool) -> None:
    if not (SOURCE / "SKILL.md").is_file():
        sys.exit(f"error: {SOURCE} has no SKILL.md (run this from a clone of the genny repository)")
    if dest.resolve() == SOURCE.resolve():
        sys.exit("error: the destination is the skill's own source folder")
    dest.parent.mkdir(parents=True, exist_ok=True)
    replaced = remove(dest)
    if link:
        try:
            dest.symlink_to(SOURCE, target_is_directory=True)
        except OSError as e:   # Windows without Developer Mode / admin rights cannot create symlinks
            print(f"could not create a symlink ({e}); copying instead")
            link = False
    if not link:
        shutil.copytree(SOURCE, dest)
    files = sorted(p.name for p in dest.iterdir())
    print(f"{'updated' if replaced else 'installed'} skill: {dest}{' -> ' + str(SOURCE) if link else ''}")
    print(f"  files: {', '.join(files)}")
    print("  use it in Claude Code with:  /genny <what you want to hear>")


def install_tool() -> None:
    uv = shutil.which("uv")
    if not uv:
        sys.exit("error: uv is not on PATH. Install it from https://docs.astral.sh/uv/ and run this again,\n"
                 f"       or install the command yourself:  pip install --editable \"{REPO}\"")
    cmd = [uv, "tool", "install", "--editable", str(REPO), "--force"]
    print("running:", " ".join(cmd))
    if subprocess.run(cmd).returncode != 0:
        sys.exit("error: uv could not install genny (on a managed Windows Python, try adding "
                 "--python with the path of a system python.exe to that command)")
    exe = shutil.which("genny")
    print(f"installed command: {exe or 'genny (open a new terminal if it is not found yet)'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Install the genny Claude Code skill (and optionally the genny command).")
    ap.add_argument("--tool", action="store_true", help="also install the `genny` command with uv (editable)")
    ap.add_argument("--link", action="store_true", help="symlink the skill folder instead of copying it")
    ap.add_argument("--uninstall", action="store_true", help="remove the installed skill")
    ap.add_argument("--dest", help="skill folder to write (default: ~/.claude/skills/genny)")
    args = ap.parse_args()
    dest = Path(args.dest).expanduser() if args.dest else skills_dir() / "genny"
    if args.uninstall:
        print(f"removed {dest}" if remove(dest) else f"nothing to remove at {dest}")
        return
    if args.tool:
        install_tool()
    install_skill(dest, args.link)


if __name__ == "__main__":
    main()
