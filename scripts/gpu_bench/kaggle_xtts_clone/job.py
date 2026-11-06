#!/usr/bin/env python3
"""Kaggle T4 kernel: XTTS v2 zero-shot voice cloning.

Tests Coqui XTTS v2 on Kaggle Tesla T4 GPU:
- Zero-shot cloning from a 6s reference WAV (EN/SK/CS)
- RTF, first-audio latency per language
- Saves output WAVs for download and listening review

License: Coqui Public Model License (non-commercial). Outputs for evaluation only.

Run: pushed as private kernel, auto-executes on push.
Outputs: /kaggle/working/xtts_clone_{en,sk,cs}.wav + xtts_report.json
"""
import hashlib, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

import soundfile as sf

OUT = Path("/kaggle/working")
REPORT = OUT / "xtts_report.json"
CODE = OUT / "bp"
T0 = time.time()
R = {"schema": 1, "status": "running", "steps": [], "errors": [], "tts": {}, "acceptance": "route_unverified"}


def save():
    R["elapsed_s"] = round(time.time() - T0, 1)
    tmp = REPORT.with_suffix(".tmp")
    tmp.write_text(json.dumps(R, indent=1, ensure_ascii=False))
    tmp.replace(REPORT)


def step(name, **kw):
    R["steps"].append({"t": round(time.time() - T0, 1), "step": name, **kw})
    print(f"[{time.time() - T0:7.1f}s] {name} {kw}", flush=True)
    save()


def sh(cmd, timeout=600):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)[-4000:]


# ---------------------------------------------------------------------------
# 0. Environment + GPU check
# ---------------------------------------------------------------------------
step("env", python=sys.version, cwd=os.getcwd())
rc, out = sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader", 60)
step("gpu", output=out.strip())

# ---------------------------------------------------------------------------
# 1. Find staged BP repo + voice memo audio files in the dataset
# ---------------------------------------------------------------------------
step("repo_lookup")
src = next((p for p in Path("/kaggle/input").rglob("app.py")
            if (p.parent / "benchmark-provenance.json").is_file()), None)
if src is None:
    step("error", msg="no staged BP repo found in /kaggle/input")
    R["status"] = "failed"; save(); sys.exit(1)
repo = src.parent
step("repo", path=str(repo))

work = OUT / "bp"
if work.exists():
    shutil.rmtree(work)
shutil.copytree(repo, work, ignore=shutil.ignore_patterns('.env', '.env.*', 'certs', '.git', '__pycache__'))
os.chdir(work)
step("checkout", path=str(work))

# Voice memo files for cloning reference (6s clips from each language)
voice_files = {}
candidates = {
    "en": ["speaker_voices/me_en_rainbow.m4a", "The Rainbow Passage.m4a", "The Rainbow Passage 2.m4a"],
    "sk": ["speaker_voices/me_sk_trhove.m4a", "Trhove rano.m4a", "Trhove rano 2.m4a"],
    "cs": ["speaker_voices/me_cs_staromestske.m4a", "Staroměstské odpoledne.m4a", "Staroměstské odpoledne 2.m4a"],
}
for lang, paths in candidates.items():
    for p in paths:
        cand = work / p
        if cand.is_file():
            voice_files[lang] = str(cand)
            step("voice_file", lang=lang, path=p)
            break
    if lang not in voice_files:
        step("voice_file_missing", lang=lang)

# ---------------------------------------------------------------------------
# 2. Install torch CUDA + Coqui TTS (XTTS v2)
# ---------------------------------------------------------------------------
step("install_torch")
# Kaggle runs Python 3.13 but Coqui TTS needs <3.12. Install Python 3.10 standalone.
rc, out = sh("which python3.10 || (wget -q https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz -O /tmp/python310.tgz && tar -xzf /tmp/python310.tgz -C /tmp && cd /tmp/Python-3.10.12 && ./configure --enable-optimizations --prefix=/tmp/python310-install && make -j$(nproc) && make install) || true", 900)
if rc != 0:
    step("python310_warning", output=out[-1000:])
# Verify python3.10
rc, out = sh("/tmp/python310-install/bin/python3.10 --version", 30)
step("python310_check", output=out.strip())
# Install torch CUDA in python3.10
rc, out = sh("/tmp/python310-install/bin/python3.10 -m pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124", 900)
if rc != 0:
    step("error", msg="torch install failed", output=out[-2000:])
    R["status"] = "failed"; save(); sys.exit(1)
step("install_torch_done")

step("install_tts")
# Install Coqui TTS in python3.10 (requires Python <3.12)
rc, out = sh("/tmp/python310-install/bin/python3.10 -m pip install -q TTS", 600)
if rc != 0:
    step("error", msg="TTS install failed", output=out[-2000:])
    R["status"] = "failed"; save(); sys.exit(1)
step("install_tts_done")

# ---------------------------------------------------------------------------
# 3. Prepare 6s reference WAV clips for voice cloning
# ---------------------------------------------------------------------------
step("prepare_refs")
ref_wavs = {}
for lang, m4a_path in voice_files.items():
    wav_out = str(OUT / f"xtts_ref_{lang}.wav")
    rc, out = sh(f"ffmpeg -y -i {m4a_path} -ac 1 -ar 22050 -t 6 {wav_out}", 60)
    if rc != 0:
        step("ref_convert_error", lang=lang, output=out[-500:])
        continue
    ref_wavs[lang] = wav_out
    step("ref_prepared", lang=lang, wav=wav_out)

# ---------------------------------------------------------------------------
# 4. Load XTTS v2 model on CUDA (run under conda py310)
# ---------------------------------------------------------------------------
step("load_start")
xtts_script = str(OUT / "xtts_load_and_run.py")
Path(xtts_script).write_text(r'''
import sys, time, json, traceback, hashlib
from pathlib import Path
import torch
from TTS.api import TTS
import soundfile as sf
import numpy as np

OUT = Path("/kaggle/working")
REPORT = OUT / "xtts_report.json"
R = {"tts": {}, "errors": [], "load_s": None, "device": None}

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    load_dt = time.perf_counter() - t0
    sample_rate = tts.synthesizer.tts_model.audio_processor.sample_rate
    R["load_s"] = round(load_dt, 2)
    R["device"] = device
except Exception as exc:
    R["errors"].append(f"XTTS load failed: {type(exc).__name__}: {exc}")
    REPORT.write_text(json.dumps(R, indent=1))
    sys.exit(1)

test_cases = [
    ("en", "Hello, this is a test of real-time speech translation on the Tesla T4 using XTTS voice cloning."),
    ("sk", "Ahoj, toto je test prekladovania reči v reálnom čase na Tesla T4 pomocou XTTS klonovania hlasu."),
    ("cs", "Ahoj, toto je test překladu řeči v reálném čase na Tesla T4 pomocí klonování hlasu XTTS."),
]

ref_map = {
    "en": "/kaggle/working/xtts_ref_en.wav",
    "sk": "/kaggle/working/xtts_ref_sk.wav",
    "cs": "/kaggle/working/xtts_ref_cs.wav",
}

for lang, text in test_cases:
    ref_wav = ref_map.get(lang)
    t0 = time.perf_counter()
    try:
        wav_out = str(OUT / f"xtts_clone_{lang}.wav")
        tts.tts_to_file(
            text=text,
            language=lang,
            speaker_wav=ref_wav,
            file_path=wav_out,
        )
        t_first = time.perf_counter() - t0
        audio, sr = sf.read(wav_out)
        dur_s = len(audio) / sr
        rtf = t_first / dur_s if dur_s > 0 else float("inf")
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        R["tts"][lang] = {
            "text": text[:80],
            "synth_s": round(t_first, 3),
            "audio_dur_s": round(dur_s, 3),
            "rtf": round(rtf, 4),
            "cloned": bool(ref_wav),
            "device": device,
            "peak": round(peak, 4),
            "wav_path": wav_out,
        }
    except Exception as exc:
        R["tts"][lang] = {"error": f"{type(exc).__name__}: {exc}"}

REPORT.write_text(json.dumps(R, indent=1))
print("XTTS voice cloning done")
''')
step("xtts_script_written")

# Run XTTS under python3.10
rc, out = sh(f"/tmp/python310-install/bin/python3.10 {xtts_script}", 600)
if rc != 0:
    step("error", msg=f"XTTS execution failed", output=out[-2000:])
    R["status"] = "failed"; save(); sys.exit(1)

# Load results
try:
    xtts_report = json.loads(Path(OUT / "xtts_report.json").read_text())
    R["tts"] = xtts_report.get("tts", {})
    R["errors"] = xtts_report.get("errors", [])
    step("load_done", device=R["tts"].get("en", {}).get("device", "cuda"), load_s=R.get("load_s"))
except Exception as exc:
    step("error", msg=f"Failed to read XTTS report: {exc}")
    R["status"] = "failed"; save(); sys.exit(1)

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
step("done", status=R["status"], tts={k: {kk: vv for kk, vv in v.items() if kk not in ("text",)} for k, v in R["tts"].items()})
save()
print(f"XTTS voice cloning complete: {R['status']}", flush=True)
