#!/usr/bin/env python3
"""One-command setup for Windows / macOS / Linux (CPU). Stdlib only, no admin rights, nothing global.

    python scripts/setup.py              # venv + deps + .env + certs + voices + MT models (+ npm if present)
    python scripts/setup.py --dev        # also install test/eval dependencies (pytest, ...)
    python scripts/setup.py --profile mac --langs de,sk   # Apple Silicon: MLX turbo; German + Slovak (run: make run PRESET=mac)
    python scripts/setup.py --profile nvidia --langs de,fr,pl   # CUDA GPU: faster-whisper large-v3-turbo
    python scripts/setup.py --profile cpu                # any CPU: Whisper small (languages default to de,sk,cs)
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
# Per language: Piper voice and the Opus-MT pairs needed with English (en-pl does not exist on Hugging Face).
LANGS = {
    "de": {"voice": "de_DE-thorsten-medium", "pairs": ["en-de", "de-en"]},
    "sk": {"voice": "sk_SK-lili-medium", "pairs": ["en-sk", "sk-en"]},
    "cs": {"voice": "cs_CZ-jirka-medium", "pairs": ["en-cs", "cs-en"]},
    "fr": {"voice": "fr_FR-siwis-medium", "pairs": ["en-fr", "fr-en"]},
    "pl": {"voice": "pl_PL-gosia-medium", "pairs": ["pl-en"]},
}
DEFAULT_LANGS = "de,sk,cs"
PROFILES = {   # extra pip packages and the speech model each hardware profile runs (documentation/LANGUAGES.md)
    "cpu": {"pip": ["onnx-asr", "onnxruntime"], "stt": ("Systran/faster-whisper-small", "ct2_models/whisper-small")},
    "mac": {"pip": ["mlx-whisper", "onnx-asr", "onnxruntime"], "stt": None},   # MLX model is fetched from the Hugging Face hub below
    "nvidia": {"pip": ["nvidia-cublas-cu12", "nvidia-cudnn-cu12", "onnx-asr", "onnxruntime-gpu"], "stt": ("Systran/faster-whisper-large-v3-turbo", "ct2_models/whisper-large-v3-turbo")},
}


def plan(langs: str, profile: str | None):
    """(piper voices, MT pairs, extra pip packages) for a --langs list and --profile. Pure, unit-tested."""
    wanted = [l.strip() for l in langs.split(",") if l.strip()]
    bad = [l for l in wanted if l not in LANGS]
    if bad:
        raise SystemExit(f"unknown language(s) {bad}; choose from {sorted(LANGS)}")
    voices = ["en_US-john-medium"] + [LANGS[l]["voice"] for l in wanted]
    pairs = [p for l in wanted for p in LANGS[l]["pairs"]]
    return voices, pairs, (PROFILES[profile]["pip"] if profile else [])
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
    ap.add_argument("--profile", choices=sorted(PROFILES), help="hardware profile: installs its engine and fetches its speech model (then run with: make run PRESET=<profile>)")
    ap.add_argument("--langs", default=DEFAULT_LANGS, help=f"comma list of {sorted(LANGS)} (default {DEFAULT_LANGS}); fetches their voices and translation models")
    ap.add_argument("--no-npm", action="store_true", help="ignored (kept so old CI lines keep working; the npm step was dead: chart.js is vendored)")
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
    voices, pairs, extra_pip = plan(a.langs, a.profile)
    if extra_pip:
        step(f"Engine packages for profile '{a.profile}': {' '.join(extra_pip)}")
        pip_install(PY, *extra_pip)

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
    for v in voices:
        run([PY, "backend/tts/download_piper_models.py", v])

    step("Whisper base (English recognition) as a plain directory: no cache symlinks, works offline")
    if not (ROOT / "ct2_models" / "whisper-base" / "model.bin").exists():
        run([PY, "scripts/fetch_whisper.py", "Systran/faster-whisper-base", "ct2_models/whisper-base"])

    if a.profile:
        step(f"Speech model for profile '{a.profile}'")
        stt = PROFILES[a.profile]["stt"]
        if stt and not (ROOT / stt[1] / "model.bin").exists():
            run([PY, "scripts/fetch_whisper.py", *stt])
        if a.profile == "mac":
            run([PY, "-c", "from huggingface_hub import snapshot_download as s; s('mlx-community/whisper-large-v3-turbo')"])

    if not a.skip_models:
        step("Translation models (+ Slovak speech model): one-time CTranslate2 conversion in an isolated env; skips what exists")
        run([sys.executable, "scripts/convert_models.py", "--pairs", ",".join(pairs), *(["--skip-stt-model"] if "sk" not in a.langs.split(",") else [])])

    # Voice Lab ear-grade store (processed/ear_grades.json is generated + gitignored; /api/ratings and the ratings tests need it).
    step("Voice Lab grade store")
    run([PY, "scripts/grade_library.py", "--apply"])

    step("Self-test (verify models load, engines available, pipeline pass)")
    run([PY, "scripts/bp.py", "selftest"])

    print(f"\nDone in {time.time() - t0:.0f}s. Start: {PY} app.py  ->  http://localhost:8000 (plain HTTP on localhost: mic works, no certificate warning; BP_HTTPS=1 for LAN/phone)")


if __name__ == "__main__":
    main()
