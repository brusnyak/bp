#!/usr/bin/env python3
"""One-time Opus-MT -> CTranslate2 (int8) conversion in a THROWAWAY venv.

The converter needs torch + transformers (~1 GB installed); the running app does not, so they live in
.venv-convert which is deleted afterwards. Output: ct2_models/Helsinki-NLP--opus-mt-<pair>/ (model + tokenizer).
"""
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / ".venv-convert"
PY = TMP / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PAIRS = ["en-sk", "sk-en", "en-cs"]
ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def run(cmd):
    subprocess.run([str(c) for c in cmd], cwd=ROOT, env=ENV, check=True)


def main():
    todo = [p for p in PAIRS if not (ROOT / "ct2_models" / f"Helsinki-NLP--opus-mt-{p}" / "model.bin").exists()]
    if not todo:
        print("all MT models already converted")
        return
    try:
        run([sys.executable, "-m", "venv", TMP])
        run([PY, "-m", "pip", "install", "--disable-pip-version-check", "--upgrade", "pip"])
        cmd = [PY, "-m", "pip", "install", "--disable-pip-version-check", "-r", "requirements-convert.txt"]
        if platform.system() == "Linux":  # default Linux torch wheels drag in ~3 GB of CUDA libs
            cmd += ["--extra-index-url", "https://download.pytorch.org/whl/cpu"]
        run(cmd)
        for p in todo:
            run([PY, "-c",
                 "import sys; sys.setrecursionlimit(2000); import backend.mt.convert_opus_mt_to_ct2 as c; "
                 f"c.convert_model('Helsinki-NLP/opus-mt-{p}', 'ct2_models/Helsinki-NLP--opus-mt-{p}', quantization='int8')"])
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
