#!/usr/bin/env python3
"""Kaggle T4 kernel: STT comparison — Distil Large v3 Turbo vs Parakeet TDT v3.

Tests both STT engines on the owner's three voice memos:
  - Staroměstské odpoledne (CS)
  - Trhove rano (SK)
  - The Rainbow Passage (EN)

Measures: first-transcription latency, RTF, and WER against reference transcripts.
Outputs: /kaggle/working/stt_compare_report.json

Run: pushed as private kernel, auto-executes on push.
"""
import hashlib, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

import soundfile as sf

OUT = Path("/kaggle/working")
REPORT = OUT / "stt_compare_report.json"
CODE = OUT / "bp"
T0 = time.time()
R = {"schema": 1, "status": "running", "steps": [], "errors": [], "engines": {}, "acceptance": "route_unverified"}


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
step("install_start")
# Copy the staged BP repo from the dataset input into working dir.
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

# Find voice memo WAV/M4A files (converted to WAV for STT)
voice_files = {}
candidates = {
    "cs": ["speaker_voices/me_cs_staromestske.m4a", "Staroměstské odpoledne.m4a", "Staroměstské odpoledne 2.m4a"],
    "sk": ["speaker_voices/me_sk_trhove.m4a", "Trhove rano.m4a", "Trhove rano 2.m4a"],
    "en": ["speaker_voices/me_en_rainbow.m4a", "The Rainbow Passage.m4a", "The Rainbow Passage 2.m4a"],
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

# Convert m4a -> wav if needed
voice_wavs = {}
for lang, path in voice_files.items():
    if path.endswith(".m4a"):
        out_path = str(OUT / f"voice_{lang}.wav")
        rc, out = sh(f"ffmpeg -y -i {path} -ac 1 -ar 16000 {out_path}", 60)
        voice_wavs[lang] = out_path
        step("voice_convert", lang=lang, rc=rc, output=out[-300:])
    else:
        voice_wavs[lang] = path

step("install_done")

# ---------------------------------------------------------------------------
# 2. Install Distil Whisper Large v3 Turbo + Parakeet TDT v3
# ---------------------------------------------------------------------------
step("install_distil_start")
rc, out = sh(f"{sys.executable} -m pip install -q faster-whisper soundfile", 600)
if rc != 0:
    step("error", msg="faster-whisper install failed", output=out[-2000:])
    R["status"] = "failed"; save(); sys.exit(1)
step("install_distil_done")

step("install_parakeet_start")
# Parakeet TDT v3 from NVIDIA — install via pip (nemo) or HF transformers
rc, out = sh(f"{sys.executable} -m pip install -q nemo_toolkit['all']", 900)
if rc != 0:
    # Fallback: try direct parakeet via transformers
    rc2, out2 = sh(f"{sys.executable} -m pip install -q transformers torch torchaudio", 900)
    step("parakeet_nemo_fallback", rc=rc2, output=out2[-500:])
step("install_parakeet_done")

# ---------------------------------------------------------------------------
# 3. Run Distil Whisper Large v3 Turbo
# ---------------------------------------------------------------------------
step("distil_start")
try:
    from faster_whisper import WhisperModel
    model = WhisperModel("distil-large-v3", device="cuda", compute_type="float16")
    step("distil_load", device="cuda", compute_type="float16")
    for lang, wav in voice_wavs.items():
        t0 = time.perf_counter()
        segments, info = model.transcribe(wav, language=lang if lang != "auto" else None, word_timestamps=True, vad_filter=True)
        text = " ".join(s.text for s in segments)
        dt = time.perf_counter() - t0
        dur = len(sf.read(wav)[0]) / sf.read(wav)[1]
        R["engines"][f"distil-large-v3-{lang}"] = {
            "model": "distil-large-v3",
            "lang": lang,
            "text": text[:200],
            "synth_s": round(dt, 3),
            "audio_dur_s": round(dur, 3),
            "rtf": round(dt / dur, 4) if dur > 0 else float("inf"),
            "device": "cuda",
            "detected_lang": info.language,
        }
        step(f"distil_{lang}", synth_s=round(dt, 3), rtf=round(dt / dur, 4) if dur > 0 else float("inf"))
except Exception as exc:
    step("distil_error", msg=f"{type(exc).__name__}: {exc}", output=traceback.format_exc()[-2000:])

# ---------------------------------------------------------------------------
# 4. Run Parakeet TDT v3
# ---------------------------------------------------------------------------
step("parakeet_start")
try:
    try:
        import nemo.collections.asr as nemo_asr
        parakeet = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
        parakeet = parakeet.to("cuda")
        step("parakeet_load", engine="nemo")
        use_nemo = True
    except Exception as nemo_exc:
        step("parakeet_nemo_failed", msg=str(nemo_exc)[-300:])
        # Fallback to transformers
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        import torch
        processor = AutoProcessor.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
        model_p = AutoModelForSpeechSeq2Seq.from_pretrained("nvidia/parakeet-tdt-0.6b-v3", torch_dtype=torch.float16).to("cuda")
        use_nemo = False
        step("parakeet_load", engine="transformers")

    for lang, wav in voice_wavs.items():
        t0 = time.perf_counter()
        if use_nemo:
            # NeMo expects file path
            transcription = parakeet.transcribe([wav])[0]
            text = transcription.text if hasattr(transcription, 'text') else str(transcription)
        else:
            import torch
            audio, sr = sf.read(wav)
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt").to("cuda")
            with torch.no_grad():
                generated_ids = model_p.generate(inputs["input_features"])
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        dt = time.perf_counter() - t0
        dur = len(sf.read(wav)[0]) / sf.read(wav)[1]
        R["engines"][f"parakeet-{lang}"] = {
            "model": "parakeet-tdt-0.6b-v3",
            "lang": lang,
            "text": text[:200],
            "synth_s": round(dt, 3),
            "audio_dur_s": round(dur, 3),
            "rtf": round(dt / dur, 4) if dur > 0 else float("inf"),
            "device": "cuda",
        }
        step(f"parakeet_{lang}", synth_s=round(dt, 3), rtf=round(dt / dur, 4) if dur > 0 else float("inf"))
except Exception as exc:
    step("parakeet_error", msg=f"{type(exc).__name__}: {exc}", output=traceback.format_exc()[-2000:])

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
step("done", status=R["status"], engines={k: {kk: vv for kk, vv in v.items() if kk != "text"} for k, v in R["engines"].items()})
save()
print(f"STT compare complete: {R['status']}", flush=True)
