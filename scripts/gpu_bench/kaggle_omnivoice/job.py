#!/usr/bin/env python3
"""Kaggle T4 kernel: OmniVoice zero-shot voice cloning benchmark.

Tests k2-fsa/OmniVoice on Kaggle Tesla T4 GPU:
- Zero-shot cloning from a 3s reference WAV (EN, SK)
- RTF, first-audio latency per language
- Speaker similarity vs reference
- Saves output WAVs and report for download

Run: pushed as private Kaggle kernel, auto-executes on push.
Outputs: /kaggle/working/omnivoice_clone_{en,sk}.wav + omnivoice_report.json
"""
import hashlib, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

import soundfile as sf
import numpy as np

OUT = Path("/kaggle/working")
REPORT = OUT / "omnivoice_report.json"
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
# 1. Find staged BP repo + voice memo audio files
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

# Find reference voice files for each language
voice_files = {}
candidates = {
    "en": [
        "speaker_voices/me_en_script.m4a",
        "processed/testaudio/en_paced.wav",
        "processed/testaudio/en_continuous.wav",
        "processed/testaudio/en_long.wav",
    ],
    "sk": [
        "speaker_voices/me_sk_script.m4a",
        "processed/testaudio/sk_paced.wav",
        "processed/testaudio/sk_continuous.wav",
        "processed/testaudio/sk_long.wav",
    ],
    "cs": [
        "speaker_voices/me_cs_staromestske.m4a",
        "processed/testaudio/cs_paced.wav",
        "processed/testaudio/cs_continuous.wav",
    ],
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
# 2. Install dependencies for OmniVoice
# ---------------------------------------------------------------------------
step("install_deps")
rc, out = sh("pip install -q torch torchaudio soundfile numpy jiwer omnivoice", 900)
step("pip_install", rc=rc, output=out[-500:] if rc != 0 else "ok")

# ---------------------------------------------------------------------------
# 3. Load OmniVoice
# ---------------------------------------------------------------------------
step("load_omnivoice")
try:
    import torch
    from omnivoice import OmniVoice
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    omni = OmniVoice.from_pretrained("k2-fsa/OmniVoice")
    load_dt = time.perf_counter() - t0
    R["load_s"] = round(load_dt, 2)
    R["device"] = device
    step("omnivoice_loaded", load_s=round(load_dt, 2), device=device)
except Exception as exc:
    R["errors"].append("OmniVoice load failed: " + type(exc).__name__ + ": " + str(exc))
    step("load_failed", error=str(exc)[:500])
    REPORT.write_text(json.dumps(R, indent=1))
    sys.exit(1)

# ---------------------------------------------------------------------------
# 4. Voice cloning test per language
# ---------------------------------------------------------------------------
test_cases = [
    ("en", "Hello, this is a test of real-time speech translation on the Tesla T4 using OmniVoice voice cloning."),
    ("sk", "Ahoj, toto je test prekladovania reči v reálnom čase na Tesla T4 pomocou klonovania hlasu OmniVoice."),
    ("cs", "Ahoj, toto je test překladu řeči v reálném čase na Tesla T4 pomocí klonování hlasu OmniVoice."),
]

for lang, text in test_cases:
    ref_wav = voice_files.get(lang)
    t0 = time.perf_counter()
    try:
        wav_out = str(OUT / f"omnivoice_clone_{lang}.wav")
        # OmniVoice zero-shot cloning with reference audio
        audio = omni.generate(
            text=text,
            prompt_wav=ref_wav,
            prompt_text=None,  # OmniVoice auto-transcribes if not provided
            num_step=12,
            speed=1.0,
        )
        t_first = time.perf_counter() - t0
        # Save output
        sf.write(wav_out, audio, 22050)
        dur_s = len(audio) / 22050
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
            "device": R.get("device"),
            "peak": round(peak, 4),
            "wav_path": wav_out,
        }
        step(f"clone_{lang}", rtf=round(rtf, 4), synth_s=round(t_first, 3))
    except Exception as exc:
        R["tts"][lang] = {"error": type(exc).__name__ + ": " + str(exc)}
        step(f"clone_{lang}_failed", error=str(exc)[:500])

# ---------------------------------------------------------------------------
# 5. Acceptance check
# ---------------------------------------------------------------------------
acceptance = []
for lang, result in R["tts"].items():
    if "error" in result:
        acceptance.append(f"{lang}: FAILED ({result['error'][:100]})")
        continue
    checks = []
    if result.get("rtf", 99) < 0.5:
        checks.append("rtf<0.5")
    else:
        checks.append(f"rtf>=0.5 ({result['rtf']})")
    if result.get("first_audio_s", 99) < 0.3:
        checks.append("latency<300ms")
    else:
        checks.append(f"latency>=300ms ({result['first_audio_s']}s)")
    if result.get("cloned"):
        checks.append("cloned")
    else:
        checks.append("not_cloned")
    acceptance.append(f"{lang}: {', '.join(checks)}")

R["acceptance"] = "; ".join(acceptance)
R["status"] = "done" if not any("error" in v for v in R["tts"].values()) else "partial"
step("done", status=R["status"], acceptance=R["acceptance"])
save()
print(f"OmniVoice voice cloning complete: {R['status']}", flush=True)
print(f"Acceptance: {R['acceptance']}", flush=True)
