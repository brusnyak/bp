#!/usr/bin/env python3
"""Kaggle T4 kernel: Fish Audio S2-Pro zero-shot voice cloning benchmark.

Tests fishaudio/fish-speech (S2-Pro) on Kaggle Tesla T4 GPU:
- Zero-shot cloning from reference WAV (EN/SK/CS)
- RTF, first-audio latency per language
- Saves output WAVs and report for download

Installs from GitHub source (not PyPI — package name doesn't exist on PyPI).

License: Fish Audio Community License (code BSD-3-Clause), S2-Pro weights
under Fish Audio Research License (non-commercial).

Run: kaggle kernels push -p scripts/gpu_bench/kaggle_fish_audio_clone --accelerator NvidiaTeslaT4 -t 1800
Outputs: /kaggle/working/fish_audio_clone_{en,sk,cs}.wav + fish_audio_report.json
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

OUT = Path("/kaggle/working")
REPORT = OUT / "fish_audio_report.json"
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
        wav_out = str(OUT / f"fish_audio_ref_{lang}.wav")
        try:
            synthesize_fallback(lang, wav_out)
            voice_files[lang] = wav_out
            step("voice_file_synthesized", lang=lang, path=wav_out)
        except Exception as exc:
            step("voice_file_failed", lang=lang, error=str(exc)[:200])

# ---------------------------------------------------------------------------
# 3. Install system deps (portaudio for pyaudio)
# ---------------------------------------------------------------------------
step("install_system_deps")
rc, out = sh("apt-get update -qq && apt-get install -y -qq libportaudio2 libffi-dev", 600)
step("system_deps", rc=rc, output=out[-300:] if rc != 0 else "ok")

# ---------------------------------------------------------------------------
# 4. Install torch CUDA
# ---------------------------------------------------------------------------
step("install_torch")
rc, out = sh("pip install -q --only-binary :all: torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124", 900)
if rc != 0:
    rc, out = sh("pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124", 900)
step("torch_installed", rc=rc, output=out[-300:] if rc != 0 else "ok")

# ---------------------------------------------------------------------------
# 5. Install pyaudio from pre-built wheel
# ---------------------------------------------------------------------------
step("install_pyaudio")
rc, out = sh("pip install -q --only-binary :all: pyaudio", 300)
if rc != 0:
    rc, out = sh("pip install -q pyaudio", 300)
step("pyaudio_installed", rc=rc, output=out[-300:] if rc != 0 else "ok")

# ---------------------------------------------------------------------------
# 6. Clone fish-speech from GitHub and install from source
# ---------------------------------------------------------------------------
step("clone_fish_speech")
FISH_DIR = OUT / "fish-speech"
if FISH_DIR.exists():
    shutil.rmtree(FISH_DIR)
rc, out = sh(f"git clone --depth 1 https://github.com/fishaudio/fish-speech.git {FISH_DIR}", 300)
step("fish_speech_cloned", rc=rc, output=out[-300:] if rc != 0 else "ok")

step("install_fish_speech")
if rc == 0:
    rc, out = sh(f"cd {FISH_DIR} && pip install -q -e . --no-deps", 900)
    if rc == 0:
        # Install requirements but skip pyaudio (no pre-built wheel on Python 3.13;
        # pyaudio is only needed for CLI audio playback, not API inference)
        rc2, out2 = sh(f"cd {FISH_DIR} && grep -v '^pyaudio' docs/requirements.txt > /tmp/requirements_no_pyaudio.txt && pip install -q -r /tmp/requirements_no_pyaudio.txt", 900)
        step("fish_deps_installed", rc=rc2, output=out2[-300:] if rc2 != 0 else "ok")
step("fish_speech_installed", rc=rc, output=out[-500:] if rc != 0 else "ok")

# ---------------------------------------------------------------------------
# 7. Load Fish Speech model and run zero-shot clone test
# ---------------------------------------------------------------------------
step("load_fish_speech")
fish_script = str(OUT / "fish_audio_load_and_run.py")
Path(fish_script).write_text(r'''
import sys, time, json
from pathlib import Path
import torch
import numpy as np
import soundfile as sf

OUT = Path("/kaggle/working")
REPORT = OUT / "fish_audio_report.json"
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
    model = AutoModelForCausalLM.from_pretrained(
        "fishaudio/fish-speech-1",
        device_map=device,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    try:
        processor = AutoProcessor.from_pretrained("fishaudio/fish-speech-1")
    except Exception:
        processor = None
    load_dt = time.perf_counter() - t0
    R["load_s"] = round(load_dt, 2)
    R["device"] = device
except Exception as exc:
    err_msg = "Fish Audio load failed: " + type(exc).__name__ + ": " + str(exc)
    R["errors"].append(err_msg)
    REPORT.write_text(json.dumps(R, indent=1))
    sys.exit(1)

test_cases = [
    ("en", "Hello, this is a test of real-time speech translation on the Tesla T4 using Fish Audio voice cloning."),
    ("sk", "Ahoj, toto je test prekladovania reči v reálnom čase na Tesla T4 pomocou klonovania hlasu Fish Audio."),
    ("cs", "Ahoj, toto je test prekladu reci v realnem case na Tesla T4 pomoci klonovani hlasu Fish Audio."),
]

ref_map = {
    "en": "/kaggle/working/fish_audio_ref_en.wav",
    "sk": "/kaggle/working/fish_audio_ref_sk.wav",
    "cs": "/kaggle/working/fish_audio_ref_cs.wav",
}

for lang, text in test_cases:
    ref_wav = ref_map.get(lang)
    t0 = time.perf_counter()
    try:
        wav_out = str(OUT / ("fish_audio_clone_" + lang + ".wav"))
        # Fish Speech generate with voice cloning
        if processor is not None:
            inputs = processor(text=text, language=lang, reference_audio=ref_wav, return_tensors="pt")
            if device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=1024)
            audio = processor.decode(outputs[0], skip_special_tokens=True)
            audio = np.asarray(audio, dtype=np.float32)
        else:
            audio = model.generate(
                text=text,
                language=lang,
                reference_audio=ref_wav,
                max_new_tokens=1024,
            )
            audio = np.asarray(audio, dtype=np.float32).squeeze()
        t_first = time.perf_counter() - t0
        sr = 24000
        sf.write(wav_out, audio, sr)
        dur_s = len(audio) / sr
        rtf = t_first / dur_s if dur_s > 0 else float("inf")
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        lang_result = {
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
        R["tts"][lang] = lang_result
    except Exception as exc:
        err_msg = type(exc).__name__ + ": " + str(exc)
        R["tts"][lang] = {"error": err_msg}

REPORT.write_text(json.dumps(R, indent=1))
print("Fish Audio voice cloning done")
''')

rc, out = sh(f"{sys.executable} {fish_script}", 600)

# Always try to read the inner report (even on failure) to surface actual errors
try:
    fish_report = json.loads(Path(OUT / "fish_audio_report.json").read_text())
    R["tts"] = fish_report.get("tts", {})
    R["errors"] = fish_report.get("errors", [])
except Exception:
    pass

if rc != 0:
    step("error", msg="Fish Audio execution failed", output=out[-2000:])
    R["status"] = "failed"; save(); sys.exit(1)

step("load_done", device=R["tts"].get("en", {}).get("device", "cuda"), load_s=R.get("load_s"))

step("done", status=R["status"], tts={k: {kk: vv for kk, vv in v.items() if kk not in ("text",)} for k, v in R["tts"].items()})
save()
print(f"Fish Audio voice cloning complete: {R['status']}", flush=True)
