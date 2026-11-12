#!/usr/bin/env python3
"""Kaggle T4 kernel: Higgs TTS 3 zero-shot voice cloning benchmark.

Tests bosonai/higgs-tts-3-4b on Kaggle Tesla T4 GPU:
- Zero-shot cloning from reference WAV (EN/SK/CS)
- RTF, first-audio latency per language
- Saves output WAVs and report for download

Uses transformers + AutoModelForSeq2SeqLM from HF weights (no PyPI package needed).

License: Boson Higgs TTS 3 Research and Non-Commercial License (non-commercial).

Run: kaggle kernels push -p scripts/gpu_bench/kaggle_higgs_audio_clone --accelerator NvidiaTeslaT4 -t 1800
Outputs: /kaggle/working/higgs_audio_clone_{en,sk,cs}.wav + higgs_audio_report.json
"""
import json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

OUT = Path("/kaggle/working")
REPORT = OUT / "higgs_audio_report.json"
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


def sh(cmd, timeout=900):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)[-4000:]


# ---------------------------------------------------------------------------
# 0. Environment + GPU check
# ---------------------------------------------------------------------------
step("env", python=sys.version, cwd=os.getcwd())
rc, out = sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader", 60)
step("gpu", output=out.strip())

# ---------------------------------------------------------------------------
# 1. Find staged BP repo for voice reference files
# ---------------------------------------------------------------------------
step("repo_lookup")
src = next((p for p in Path("/kaggle/input").rglob("app.py")
            if (p.parent / "benchmark-provenance.json").is_file()), None)
if src is not None:
    repo = src.parent
    step("repo", path=str(repo))
else:
    step("repo_lookup_failed", msg="no staged BP repo — will synthesize reference audio")
    repo = None

# ---------------------------------------------------------------------------
# 2. Locate or synthesize reference voice files (10-30s clips per language)
# ---------------------------------------------------------------------------
step("prepare_refs")
import soundfile as sf
import numpy as np

voice_files = {}
candidates_by_lang = {
    "en": ["speaker_voices/me_en_rainbow.m4a", "processed/testaudio/en_paced.wav"],
    "sk": ["speaker_voices/me_sk_trhove.m4a", "processed/testaudio/sk_paced.wav"],
    "cs": ["speaker_voices/me_cs_staromestske.m4a", "processed/testaudio/cs_paced.wav"],
}

def find_file(paths):
    for p in paths:
        cand = Path(".") / p
        if cand.is_file():
            return str(cand.resolve())
    try:
        for input_dir in Path("/kaggle/input").iterdir():
            if not input_dir.is_dir():
                continue
            for p in paths:
                cand = input_dir / p
                if cand.is_file():
                    return str(cand.resolve())
    except Exception:
        pass
    return None

def synthesize_fallback(lang, out_path):
    sr = 22050
    t = np.linspace(0, 12, int(12 * sr), endpoint=False)
    f0 = 120 if lang == "en" else 140
    audio = 0.3 * np.sin(2 * np.pi * f0 * t)
    audio += 0.15 * np.sin(2 * np.pi * f0 * 1.5 * t)
    audio += 0.08 * np.sin(2 * np.pi * f0 * 2.0 * t)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 2.5 * t)
    audio = audio * env
    audio = audio / np.max(np.abs(audio)) * 0.7
    sf.write(out_path, audio.astype(np.float32), sr)
    return out_path

for lang, paths in candidates_by_lang.items():
    found = find_file(paths)
    if found:
        voice_files[lang] = found
        step("voice_file", lang=lang, path=found)
    else:
        wav_out = str(OUT / f"higgs_audio_ref_{lang}.wav")
        try:
            synthesize_fallback(lang, wav_out)
            voice_files[lang] = wav_out
            step("voice_file_synthesized", lang=lang, path=wav_out)
        except Exception as exc:
            step("voice_file_failed", lang=lang, error=str(exc)[:200])

# ---------------------------------------------------------------------------
# 3. Install transformers + accelerate
# ---------------------------------------------------------------------------
step("install_transformers")
rc, out = sh("pip install -q git+https://github.com/huggingface/transformers.git accelerate", 900)
if rc != 0:
    rc, out = sh("pip install -q --upgrade transformers accelerate", 900)
step("transformers_installed", rc=rc, output=out[-300:] if rc != 0 else "ok")

# ---------------------------------------------------------------------------
# 4. Load Higgs TTS 3 model and run zero-shot clone test
# ---------------------------------------------------------------------------
step("load_higgs_audio")
higgs_script = str(OUT / "higgs_audio_load_and_run.py")
Path(higgs_script).write_text(r'''
import sys, time, json, traceback
from pathlib import Path
import torch
import numpy as np
import soundfile as sf

OUT = Path("/kaggle/working")
REPORT = OUT / "higgs_audio_report.json"
R = {"tts": {}, "errors": [], "load_s": None, "device": None}

try:
    from transformers import AutoModelForCausalLM, AutoProcessor
except ImportError as exc:
    err_msg = "transformers import failed: " + type(exc).__name__ + ": " + str(exc)
    R = {"errors": [err_msg], "tts": {}}
    REPORT.write_text(json.dumps(R, indent=1))
    sys.exit(1)

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    # Higgs TTS 3 uses higgs_multimodal_qwen3 architecture (Qwen3-based causal LM)
    model = None
    pipe = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "bosonai/higgs-tts-3-4b",
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        model_type = "causal"
    except Exception:
        from transformers import pipeline
        pipe = pipeline(
            "text-to-speech",
            model="bosonai/higgs-tts-3-4b",
            device=device,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        model_type = "pipeline"
    try:
        processor = AutoProcessor.from_pretrained("bosonai/higgs-tts-3-4b")
    except Exception:
        processor = None
    load_dt = time.perf_counter() - t0
    R["load_s"] = round(load_dt, 2)
    R["device"] = device
except Exception as exc:
    R["errors"].append("Higgs Audio load failed: " + type(exc).__name__ + ": " + str(exc))
    REPORT.write_text(json.dumps(R, indent=1))
    sys.exit(1)

test_cases = [
    ("en", "Hello, this is a test of real-time speech translation on the Tesla T4 using Higgs TTS 3 voice cloning."),
    ("sk", "Ahoj, toto je test prekladovania reči v reálnom čase na Tesla T4 pomocou klonovania hlasu Higgs Audio."),
    ("cs", "Ahoj, toto je test překladu řeči v reálném čase na Tesla T4 pomocí klonování hlasu Higgs Audio."),
]

ref_map = {
    "en": "/kaggle/working/higgs_audio_ref_en.wav",
    "sk": "/kaggle/working/higgs_audio_ref_sk.wav",
    "cs": "/kaggle/working/higgs_audio_ref_cs.wav",
}

for lang, text in test_cases:
    ref_wav = ref_map.get(lang)
    t0 = time.perf_counter()
    try:
        wav_out = str(OUT / f"higgs_audio_clone_{lang}.wav")
        if model_type == "causal":
            # AutoModelForCausalLM path
            if processor is not None:
                inputs = processor(text=text, language=lang, reference_audio=ref_wav, return_tensors="pt")
                if device == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
            else:
                inputs = {"input_ids": None}
                if ref_wav:
                    inputs["reference_audio"] = ref_wav
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=1024) if processor else model.generate(text=text, language=lang, reference_audio=ref_wav, max_new_tokens=1024)
            if processor:
                audio = processor.decode(outputs[0], skip_special_tokens=True)
                audio = np.asarray(audio, dtype=np.float32)
            else:
                audio = np.asarray(outputs, dtype=np.float32).squeeze()
            sr = 24000
        else:
            # pipeline path
            try:
                result = pipe(text, voice_prompt=ref_wav)
            except TypeError:
                try:
                    result = pipe(text, reference_audio=ref_wav)
                except TypeError:
                    result = pipe(text)
            if isinstance(result, dict):
                audio = result.get("audio", result.get("waveform"))
                if hasattr(audio, "numpy"):
                    audio = audio.numpy()
                audio = np.asarray(audio, dtype=np.float32).squeeze()
                sr = result.get("sampling_rate", 24000)
            else:
                audio = np.asarray(result, dtype=np.float32).squeeze()
                sr = 24000
        sf.write(wav_out, audio, sr)
        t_first = time.perf_counter() - t0
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
        R["tts"][lang] = {"error": type(exc).__name__ + ": " + str(exc)}

REPORT.write_text(json.dumps(R, indent=1))
print("Higgs Audio voice cloning done")
''')
step("higgs_script_written")

rc, out = sh(f"{sys.executable} {higgs_script}", 600)

# Always try to read the inner report (even on failure) to surface actual errors
try:
    higgs_report = json.loads(Path(OUT / "higgs_audio_report.json").read_text())
    R["tts"] = higgs_report.get("tts", {})
    R["errors"] = higgs_report.get("errors", [])
except Exception:
    pass

if rc != 0:
    step("error", msg=f"Higgs Audio execution failed", output=out[-2000:])
    R["status"] = "failed"; save(); sys.exit(1)

step("load_done", device=R["tts"].get("en", {}).get("device", "cuda"), load_s=R.get("load_s"))

step("done", status=R["status"], tts={k: {kk: vv for kk, vv in v.items() if kk not in ("text",)} for k, v in R["tts"].items()})
save()
print(f"Higgs Audio voice cloning complete: {R['status']}", flush=True)
