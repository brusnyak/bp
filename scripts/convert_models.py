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
ALL_PAIRS = ["en-sk", "sk-en", "en-cs", "cs-en", "en-de", "de-en", "en-fr", "fr-en", "pl-en", "zh-en", "hi-en", "ja-en", "en-zh", "en-hi"]
CORE_PAIRS = ["en-sk", "sk-en", "en-cs", "cs-en", "en-de", "de-en", "en-fr", "fr-en", "pl-en"]   # no opus-mt-en-pl exists
def _pairs():
    """--pairs en-de,de-en | --all | default core set (German first; zh/hi/ja are opt-in with --all)."""
    for i, a in enumerate(sys.argv):
        if a == "--pairs" and i + 1 < len(sys.argv): return [x for x in sys.argv[i + 1].split(",") if x]
    return ALL_PAIRS if "--all" in sys.argv else CORE_PAIRS
PAIRS = ALL_PAIRS
ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def run(cmd):
    subprocess.run([str(c) for c in cmd], cwd=ROOT, env=ENV, check=True)


SK_STT = ("NaiveNeuron/whisper-small-sk", "ct2_models/whisper-small-sk")  # Slovak-fine-tuned Whisper small (MIT)


def main():
    skip_stt = "--skip-stt-model" in sys.argv
    todo = [p for p in _pairs() if not (ROOT / "ct2_models" / f"Helsinki-NLP--opus-mt-{p}" / "model.bin").exists()]
    need_stt = not skip_stt and not (ROOT / SK_STT[1] / "model.bin").exists()
    if not todo and not need_stt:
        print("all models already converted")
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
        if need_stt:
            run([PY, "scripts/convert_whisper.py", *SK_STT])
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
