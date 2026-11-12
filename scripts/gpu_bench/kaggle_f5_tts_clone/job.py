#!/usr/bin/env python3
"""Kaggle T4 kernel: F5-TTS zero-shot voice cloning benchmark.

Tests SWivid/F5-TTS on Kaggle Tesla T4 GPU:
- Zero-shot cloning from a 6s reference WAV (EN/SK/CS)
- RTF, first-audio latency per language
- Saves output WAVs and report for download

License: MIT (PyPI package). Verify before live use.

Run: kaggle kernels push -p scripts/gpu_bench/kaggle_f5_tts_clone --accelerator NvidiaTeslaT4 -t 1800
Outputs: /kaggle/working/f5_tts_clone_{en,sk,cs}.wav + f5_tts_report.json
"""
import hashlib, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

OUT = Path("/kaggle/working")
REPORT = OUT / "f5_tts_report.json"
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
        # Also search Kaggle input directories (dataset assets)
        try:
            for input_dir in Path("/kaggle/input").iterdir():
                if not input_dir.is_dir():
                    continue
                for p in paths:
                    cand = input_dir / p
                    if cand.is_file():
                        voice_files[lang] = str(cand)
                        step("voice_file", lang=lang, path=str(cand))
                        break
                if lang in voice_files:
                    break
        except Exception:
            pass
        if lang not in voice_files:
            step("voice_file_missing", lang=lang)

# ---------------------------------------------------------------------------
# 2. Install F5-TTS (PyTorch CUDA)
# ---------------------------------------------------------------------------
step("install_torch")
rc, out = sh("pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124", 900)
if rc != 0:
    rc, out = sh("pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124", 900)
step("install_torch_done", rc=rc)

step("install_f5tts")
rc, out = sh("pip install -q f5-tts", 900)
if rc != 0:
    step("error", msg="f5-tts install failed", output=out[-2000:])
    R["status"] = "failed"; save(); sys.exit(1)
step("install_f5tts_done")

# ---------------------------------------------------------------------------
# 3. Prepare 6s reference WAV clips for voice cloning
# ---------------------------------------------------------------------------
step("prepare_refs")
import soundfile as sf
ref_wavs = {}
for lang, m4a_path in voice_files.items():
    wav_out = str(OUT / f"f5_tts_ref_{lang}.wav")
    rc, out = sh(f"ffmpeg -y -i {m4a_path} -ac 1 -ar 22050 -t 6 {wav_out}", 60)
    if rc != 0:
        step("ref_convert_error", lang=lang, output=out[-500:])
        continue
    ref_wavs[lang] = wav_out
    step("ref_prepared", lang=lang, wav=wav_out)

# ---------------------------------------------------------------------------
# 4. Load F5-TTS model on CUDA
# ---------------------------------------------------------------------------
step("load_start")
f5_script = str(OUT / "f5_tts_load_and_run.py")
Path(f5_script).write_text(r'''
import sys, time, json, traceback
from pathlib import Path
import torch
import numpy as np
import soundfile as sf

try:
    from f5_tts.api import F5TTS
except ImportError as exc:
    R = {"errors": [f"f5-tts import failed: {type(exc).__name__}: {exc}"}, "tts": {}}
    Path("/kaggle/working/f5_tts_report.json").write_text(json.dumps(R, indent=1))
    sys.exit(1)

OUT = Path("/kaggle/working")
REPORT = OUT / "f5_tts_report.json"
R = {"tts": {}, "errors": [], "load_s": None, "device": None}

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    model = F5TTS(model="F5TTS_v1_Base", device=device)
    load_dt = time.perf_counter() - t0
    R["load_s"] = round(load_dt, 2)
    R["device"] = device
except Exception as exc:
    R["errors"].append(f"F5-TTS load failed: {type(exc).__name__}: {exc}")
    REPORT.write_text(json.dumps(R, indent=1))
    sys.exit(1)

test_cases = [
    ("en", "Hello, this is a test of real-time speech translation on the Tesla T4 using F5-TTS voice cloning."),
    ("sk", "Ahoj, toto je test prekladovania reči v reálnom čase na Tesla T4 pomocou klonovania hlasu F5-TTS."),
    ("cs", "Ahoj, toto je test překladu řeči v reálném čase na Tesla T4 pomocí klonování hlasu F5-TTS."),
]

ref_map = {
    "en": "/kaggle/working/f5_tts_ref_en.wav",
    "sk": "/kaggle/working/f5_tts_ref_sk.wav",
    "cs": "/kaggle/working/f5_tts_ref_cs.wav",
}

for lang, text in test_cases:
    ref_wav = ref_map.get(lang)
    t0 = time.perf_counter()
    try:
        wav_out = str(OUT / f"f5_tts_clone_{lang}.wav")
        # F5-TTS inference with reference audio
        audio = model.infer(
            text=text,
            ref_audio=ref_wav,
            language=lang,
        )
        t_first = time.perf_counter() - t0
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        sr = 24000
        sf.write(wav_out, audio, sr)
        dur_s = len(audio) / sr
        rtf = t_first / dur_s if dur_s > 0 else float("inf")
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        R["tts"][lang] = {
            "text": text[:80],
            "synth_s": round(t_first, 3),
            "audio_dur_s": round(dur_s, 3),
            "rtf": round(rtf, 4),
            "first_audio_s": round(t_first, 3),
            "cloned": bool(ref_wav),
            "ref_wav": ref_wav,
            "device": device,
            "peak": round(peak, 4),
            "wav_path": wav_out,
        }
    except Exception as exc:
        R["tts"][lang] = {"error": f"{type(exc).__name__}: {exc}"}

REPORT.write_text(json.dumps(R, indent=1))
print("F5-TTS voice cloning done")
''')
step("f5_script_written")

# Run F5-TTS
rc, out = sh(f"{sys.executable} {f5_script}", 600)
if rc != 0:
    step("error", msg=f"F5-TTS execution failed", output=out[-2000:])
    R["status"] = "failed"; save(); sys.exit(1)

# Load results
try:
    f5_report = json.loads(Path(OUT / "f5_tts_report.json").read_text())
    R["tts"] = f5_report.get("tts", {})
    R["errors"] = f5_report.get("errors", [])
    step("load_done", device=R["tts"].get("en", {}).get("device", "cuda"), load_s=R.get("load_s"))
except Exception as exc:
    step("error", msg=f"Failed to read F5-TTS report: {exc}")

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
step("done", status=R["status"], tts={k: {kk: vv for kk, vv in v.items() if kk not in ("text",)} for k, v in R["tts"].items()})
save()
print(f"F5-TTS voice cloning complete: {R['status']}", flush=True)
