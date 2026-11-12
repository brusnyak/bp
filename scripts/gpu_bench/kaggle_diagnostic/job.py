#!/usr/bin/env python3
"""Kaggle T4 diagnostic: environment + package install check.

Tests whether Kaggle T4 environment can:
1. See the GPU
2. Install torch CUDA
3. Install fish-speech package
4. Download Fish Audio model from HF

Minimal steps to isolate where kernels fail.
"""
import json, os, subprocess, sys, time
from pathlib import Path

OUT = Path("/kaggle/working")
REPORT = OUT / "diagnostic_report.json"
T0 = time.time()
R = {"schema": 1, "status": "running", "steps": [], "errors": {}, "acceptance": "pending"}


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
    return p.returncode, (p.stdout + p.stderr)[-3000:]


# 0. GPU
step("env", python=sys.version, cwd=os.getcwd())
rc, out = sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader", 60)
step("gpu", rc=rc, output=out.strip())

# 1. torch CUDA
step("install_torch")
rc, out = sh("pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124", 900)
step("torch_installed", rc=rc, output=out[-500:] if rc != 0 else "ok")

# 2. fish-speech package
step("install_fish_speech")
rc, out = sh("pip install -q fish-speech", 900)
step("fish_speech_installed", rc=rc, output=out[-500:] if rc != 0 else "ok")

# 3. HF model download test
step("test_hf_download")
if rc == 0:
    try:
        from huggingface_hub import snapshot_download
        t0_dl = time.time()
        path = snapshot_download(repo_id="fishaudio/fish-speech-1.5", local_dir="/tmp/fish_model_test")
        step("hf_download", rc=0, path=path, seconds=round(time.time() - t0_dl, 1))
    except Exception as exc:
        step("hf_download_failed", error=str(exc)[:500])
        R["errors"]["hf_download"] = str(exc)[:500]
else:
    step("hf_download_skipped", reason="fish-speech install failed")

# 4. Try loading model
step("test_model_load")
if rc == 0 and not R["errors"]:
    try:
        import torch
        from fishtts import Fishtts
        t0_load = time.time()
        model = Fishtts.from_pretrained("fishaudio/fish-speech-1.5", device="cuda")
        step("model_loaded", seconds=round(time.time() - t0_load, 1), device="cuda")
    except Exception as exc:
        step("model_load_failed", error=str(exc)[:500])
        R["errors"]["model_load"] = str(exc)[:500]
else:
    step("model_load_skipped", reason="install or download failed")

R["status"] = "done" if not R["errors"] else "partial"
step("done", status=R["status"], errors=R["errors"])
save()
print(f"Diagnostic complete: {R['status']}", flush=True)
