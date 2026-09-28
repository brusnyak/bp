#!/usr/bin/env python3
"""One-command setup for Windows / macOS / Linux (CPU). Stdlib only, no admin rights, nothing global.

    python scripts/setup.py              # venv + deps + .env + certs + voices + MT models (+ npm if present)
    python scripts/setup.py --dev        # also install test/eval dependencies (pytest, ...)
    python scripts/setup.py --skip-models   # skip the one-time MT conversion (needs torch in a throwaway venv)

Idempotent: every step is skipped when its output already exists.
"""
import argparse
import os
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PIPER_VOICES = ["en_US-ryan-medium", "sk_SK-lili-medium", "cs_CZ-jirka-medium"]
MT_PAIRS = ["en-sk", "sk-en", "en-cs"]
ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def step(msg):
    print(f"\n==> {msg}", flush=True)


def run(cmd, **kw):
    subprocess.run([str(c) for c in cmd], cwd=ROOT, env=ENV, check=True, **kw)


def pip_install(python, *args):
    """uv when available (much faster), plain pip otherwise."""
    uv = shutil.which("uv")
    if uv:
        run([uv, "pip", "install", "--python", python, *args])
    else:
        run([python, "-m", "pip", "install", "--disable-pip-version-check", *args])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="install requirements-dev.txt")
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--no-npm", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        sys.exit(f"Python 3.10-3.12 required (found {sys.version.split()[0]}). The pinned numpy/numba stack has no wheels beyond 3.12.")

    step("Virtual environment (.venv)")
    if not PY.exists():
        run([sys.executable, "-m", "venv", VENV])
    step("Dependencies")
    # The pip/setuptools bundled with `python -m venv` are old enough to carry known CVEs.
    pip_install(PY, "--upgrade", "pip", "setuptools")
    pip_install(PY, "-r", "requirements-dev.txt" if a.dev else "requirements.txt")

    step(".env (JWT secret)")
    env_file = ROOT / ".env"
    if not env_file.exists():
        env_file.write_text(f"JWT_SECRET={secrets.token_urlsafe(48)}\n", encoding="utf-8")
        print("created .env with a random JWT_SECRET")
    elif "JWT_SECRET" not in env_file.read_text(encoding="utf-8"):
        with env_file.open("a", encoding="utf-8") as f:
            f.write(f"\nJWT_SECRET={secrets.token_urlsafe(48)}\n")

    step("Self-signed localhost certificate (certs/)")
    if not (ROOT / "certs" / "cert.pem").exists():
        run([PY, "scripts/gen_cert.py"])

    step("Piper voices")
    for v in PIPER_VOICES:
        run([PY, "backend/tts/download_piper_models.py", v])

    if not a.skip_models:
        step("MT models (one-time CTranslate2 conversion, isolated env)")
        if all((ROOT / "ct2_models" / f"Helsinki-NLP--opus-mt-{p}" / "model.bin").exists() for p in MT_PAIRS):
            print("ct2_models already present")
        else:
            run([sys.executable, "scripts/convert_models.py"])

    if not a.no_npm and shutil.which("npm") and (ROOT / "package.json").exists():
        step("UI assets (npm)")
        run(["npm", "install", "--no-audit", "--no-fund"], shell=(os.name == "nt"))

    print(f"\nDone in {time.time() - t0:.0f}s. Start: {PY} app.py  ->  https://localhost:8000 (accept the self-signed cert)")


if __name__ == "__main__":
    main()
