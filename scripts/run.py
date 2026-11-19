#!/usr/bin/env python3
"""Run BP with the matching hardware preset on Windows, macOS or Linux.

Usage: python scripts/run.py --profile cpu|mac|nvidia
The profile must match the one used with scripts/setup.py.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROFILES = ("cpu", "mac", "nvidia")
VENV_PY = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    args = parser.parse_args()

    if not VENV_PY.is_file():
        parser.error("Missing .venv; first run python scripts/setup.py --profile " + args.profile)

    # python-dotenv is installed by setup.py into .venv, not necessarily system Python.
    # Re-execute this script under the project venv for consistent dependencies.
    if Path(sys.executable).resolve() != VENV_PY.resolve():
        return subprocess.call([str(VENV_PY), __file__, "--profile", args.profile], cwd=ROOT)

    from dotenv import dotenv_values

    preset = ROOT / "presets" / (args.profile + ".env")
    if not preset.is_file():
        parser.error("Missing hardware preset: " + str(preset))
    values = dotenv_values(preset)
    environment = os.environ.copy()
    environment.update({key: value for key, value in values.items() if value is not None})
    print("Starting BP with profile " + args.profile + " at http://localhost:8000", flush=True)
    return subprocess.call([str(VENV_PY), str(ROOT / "app.py")], cwd=ROOT, env=environment)


if __name__ == "__main__":
    sys.exit(main())
