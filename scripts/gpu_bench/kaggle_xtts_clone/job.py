#!/usr/bin/env python3
"""Kaggle T4 kernel: XTTS v2 zero-shot voice cloning benchmark (fixed standalone env).

Tests Coqui XTTS v2 on Kaggle Tesla T4 GPU:
- Zero-shot cloning from reference WAV (EN/SK/CS)
- RTF, first-audio latency per language
- License: Coqui Public Model License (non-commercial) — evaluation only

FIXES vs prior attempt:
- Kaggle default Python is 3.13.16 but Coqui TTS requires Python <3.12
- Use uv python install 3.11 + venv with direct binary paths (no conda, no `source activate`)
- Install torch with CUDA 12.x (Kaggle T4 has CUDA 13 driver but torch cu128 wheels work)

License: Coqui Public Model License (non-commercial).

Run: kaggle kernels push -p scripts/gpu_bench/kaggle_xtts_clone --accelerator NvidiaTeslaT4 -t 1800
Outputs: /kaggle/working/xtts_clone_{en,sk,cs}.wav + xtts_report.json
"""
import json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

OUT = Path("/kaggle/working")
REPORT = OUT / "xtts_report.json"
T0 = time.time()
R = {"schema": 1, "status": "running", "steps": [], "errors": [], "tts": {}, "infrastructure": {}}


def save():
    R["elapsed_s"] = round(time.time() - T0, 1)
    tmp = REPORT.with_suffix(".tmp")
    tmp.write_text(json.dumps(R, indent=1, ensure_ascii=False))
    tmp.replace(REPORT)


def step(name, **kw):
    R["steps"].append({"t": round(time.time() - T0, 1), "step": name, **kw})
    print(f"[{time.time() - T0:7.1f}s] {name} {kw}", flush=True)
    save()


def sh(cmd, timeout=900):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)[-4000:]


# ---------------------------------------------------------------------------
# 0. Environment + GPU check
# ---------------------------------------------------------------------------
step("env", python=sys.version, cwd=os.getcwd())
rc, out = sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader", 60)
step("gpu", output=out.strip())

rc, out = sh("python3 -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)'", 30)
R["infrastructure"]["torch_system"] = out.strip() if rc == 0 else "not_installed"
save()

# ---------------------------------------------------------------------------
# 1. Install uv + Python 3.11 venv (Coqui TTS requires Python <3.12)
# ---------------------------------------------------------------------------
step("install_uv")
rc, out = sh("pip install uv", 120)
step("uv_install", rc=rc, output=out[-200:] if rc != 0 else "ok")

step("uv_python_311")
rc, out = sh("uv python install 3.11", 300)
step("uv_python_install", rc=rc, output=out[-500:] if rc != 0 else "ok")

if rc != 0:
    step("uv_python_failed", error=out[-500:])
    R["status"] = "failed"; save(); sys.exit(1)

VENV_DIR = "/tmp/xtts-venv"
step("create_venv")
rc, out = sh(f"uv venv --python 3.11 {VENV_DIR}", 120)
step("venv_create", rc=rc, output=out[-300:] if rc != 0 else "ok")

if rc != 0:
    step("venv_failed", error=out[-500:])
    R["status"] = "failed"; save(); sys.exit(1)

VENV_PYTHON = f"{VENV_DIR}/bin/python"
UV_PIP_INSTALL = f"uv pip install --python {VENV_PYTHON}"

rc, out = sh(f"{VENV_PYTHON} --version", 30)
step("venv_python_version", output=out.strip() if rc == 0 else "failed")

# ---------------------------------------------------------------------------
# 2. Install torch with CUDA + Coqui TTS in the venv
# ---------------------------------------------------------------------------
step("install_torch_cuda")
# Kaggle T4 has CUDA 13 driver; torch cu128 wheels are compatible with CUDA 12.8+ drivers
rc, out = sh(f"{UV_PIP_INSTALL} --upgrade pip uv", 120)
step("pip_upgrade", rc=rc, output=out[-200:] if rc != 0 else "ok")

rc, out = sh(f"{UV_PIP_INSTALL} torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128", 600)
step("torch_cuda_install", rc=rc, output=out[-500:] if rc != 0 else "ok")

if rc != 0:
    step("torch_cuda_failed", error=out[-500:])
    R["status"] = "failed"; save(); sys.exit(1)

# Install Coqui TTS
# Set COQUI_TOS_AGREED=1 to accept Coqui Public Model License (non-commercial)
rc, out = sh(f"COQUI_TOS_AGREED=1 {UV_PIP_INSTALL} TTS", 600)
step("coqui_tts_install", rc=rc, output=out[-500:] if rc != 0 else "ok")

if rc != 0:
    step("coqui_failed", error=out[-500:])
    R["status"] = "failed"; save(); sys.exit(1)

save()

# ---------------------------------------------------------------------------
# 3. Find staged BP repo for voice reference files
# ---------------------------------------------------------------------------
step("repo_lookup")
src = next((p for p in Path("/kaggle/input").rglob("app.py")
            if (p.parent / "benchmark-provenance.json").is_file()), None)
repo = src.parent if src else None
step("repo", path=str(repo) if repo else "none")

candidates_by_lang = {
    "en": ["speaker_voices/me_en_rainbow.m4a", "processed/testaudio/en_paced.wav"],
    "sk": ["speaker_voices/me_sk_trhove.m4a", "processed/testaudio/sk_paced.wav"],
    "cs": ["speaker_voices/me_cs_staromestske.m4a", "processed/testaudio/cs_paced.wav"],
}

def find_file(paths):
    for p in paths:
        cand = Path(".") / p
        if cand.is_file():
            return str(cand)
        if repo:
            cand2 = repo / p
            if cand2.is_file():
                return str(cand2)
    return None

voice_files = {}
for lang, paths in candidates_by_lang.items():
    ref = find_file(paths)
    if ref:
        wav_path = OUT / f"ref_{lang}.wav"
        rc, out = sh(f'ffmpeg -y -v error -i {ref} -t 10 -ac 1 -ar 24000 {wav_path}', 60)
        if rc == 0:
            voice_files[lang] = str(wav_path)
            step("ref", lang=lang, wav=str(wav_path))
        else:
            step("ref_fail", lang=lang, error=out[-200:])
    else:
        step("ref_missing", lang=lang)

if not voice_files:
    step("warning", msg="no reference voice files found — synthesizing fallback audio")
    import wave, struct, math
    for lang in ("en", "sk", "cs"):
        wav_path = OUT / f"ref_{lang}.wav"
        sr = 24000
        duration = 5.0
        freq = 440 if lang == "en" else (520 if lang == "sk" else 600)
        frames = []
        for i in range(int(sr * duration)):
            t = i / sr
            sample = int(16000 * math.sin(2 * math.pi * freq * t) * (0.5 + 0.5 * math.sin(2 * math.pi * 0.5 * t)))
            frames.append(struct.pack('<h', sample))
        with wave.open(str(wav_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(b''.join(frames))
        voice_files[lang] = str(wav_path)
        step("ref_synth", lang=lang, wav=str(wav_path))

# ---------------------------------------------------------------------------
# 4. Run XTTS v2 per language
# ---------------------------------------------------------------------------
step("run_xtts")

test_texts = {
    "en": "Hello, this is a test of real-time speech translation.",
    "sk": "Dobrý deň, toto je test prekladov reči v reálnom čase.",
    "cs": "Dobrý den, toto je test překladu řeči v reálném čase.",
}

XTTS_SCRIPT = f"{OUT}/tts_xtts.py"
script_content = r'''
import sys, time, traceback, json
from pathlib import Path
from TTS.api import TTS
import torch

out_dir = sys.argv[1]
lang = sys.argv[2]
ref_wav = sys.argv[3]
text = sys.argv[4]
out_wav = sys.argv[5]

result = {"lang": lang, "status": "running"}
t0 = time.perf_counter()

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=device=="cuda")
    load_s = time.perf_counter() - t0
    result["load_s"] = round(load_s, 2)

    t0 = time.perf_counter()
    wav = tts.tts(text=text, speaker_wav=ref_wav, language=lang.split("-")[0] if "-" in lang else lang)
    infer_s = time.perf_counter() - t0

    import soundfile as sf
    import numpy as np
    wav = np.asarray(wav, dtype='float32')
    sr = 24000
    sf.write(out_wav, wav, sr)
    audio_s = len(wav) / sr
    result["status"] = "complete"
    result["wall_s"] = round(infer_s, 3)
    result["audio_s"] = round(audio_s, 2)
    result["rtf"] = round(infer_s / audio_s, 3) if audio_s > 0 else None
    result["output_wav"] = out_wav
    result["device"] = device
except Exception as e:
    result["status"] = "failed"
    result["error"] = f"{type(e).__name__}: {str(e)[:500]}"
    result["tb"] = traceback.format_exc()[-1000:]

Path(out_dir).write_text(json.dumps(result, indent=1, ensure_ascii=False))
'''

Path(XTTS_SCRIPT).write_text(script_content)

for lang in ("en", "sk", "cs"):
    step("tts", lang=lang)
    try:
        ref_wav = voice_files[lang]
        out_path = OUT / f"xtts_clone_{lang}.wav"
        out_json = OUT / f"xtts_result_{lang}.json"

        t0 = time.perf_counter()
        rc, out = sh(
            f"COQUI_TOS_AGREED=1 {VENV_PYTHON} {XTTS_SCRIPT} {out_json} {lang} {ref_wav} "
            f"'{test_texts[lang]}' {out_path} < /dev/null", 600
        )
        wall_s = time.perf_counter() - t0

        if rc == 0 and out_json.exists():
            result = json.loads(out_json.read_text())
            R["tts"][lang] = {
                "device": result.get("device", "unknown"),
                "wall_s": result.get("wall_s", round(wall_s, 3)),
                "audio_s": result.get("audio_s"),
                "rtf": result.get("rtf"),
                "output_wav": str(out_path) if out_path.exists() else None,
                "voice_clone": True,
                "load_s": result.get("load_s"),
                "method": "python_api",
            }
            if result.get("error"):
                R["tts"][lang]["error"] = result["error"]
            step("tts_done", lang=lang, rtf=result.get("rtf"))
        else:
            step("tts_rc_fail", lang=lang, rc=rc, output=out[-500:] if rc != 0 else "ok")
            R["tts"][lang] = {"error": f"Script failed rc={rc}: {out[-300:] if rc != 0 else ''}"}
    except Exception as e:
        step("tts_fail", lang=lang, err=f"{type(e).__name__}: {str(e)[:300]}")
        R["errors"].append(f"{lang}: {type(e).__name__}: {str(e)[:500]}")
        R["tts"][lang] = {"error": str(e)[:500]}
    save()

R["status"] = "complete"
save()
print("XTTS kernel complete.", flush=True)