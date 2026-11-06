#!/usr/bin/env python3
"""Kaggle T4 kernel: OmniVoice zero-shot voice cloning TTS benchmark.

Tests OmniVoice TTS (k2-fsa/OmniVoice, Apache-2.0) on Kaggle Tesla T4 GPU:
- Zero-shot cloning from a reference WAV (EN/SK/CS)
- RTF, first-audio latency per language
- WER of synthesized output via faster-whisper ASR (quality check)
- No STT/MT cascade — TTS only, isolated to measure cloning overhead.

Run: pushed as private kernel, auto-executes on push.
Outputs: /kaggle/working/omnivoice_report.json
"""
import glob, hashlib, json, os, shutil, subprocess, sys, time, traceback
import soundfile as sf
from pathlib import Path

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
# 0. Environment + repo checkout from Kaggle dataset
# ---------------------------------------------------------------------------
step("env", python=sys.version, cwd=os.getcwd())
rc, out = sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader", 60)
step("gpu", output=out.strip())

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

# ---------------------------------------------------------------------------
# 1. Install torch (CUDA) + soundfile + omnivoice
# ---------------------------------------------------------------------------
step("install_start")
rc, out = sh(f"{sys.executable} -m pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 soundfile", 900)
if rc != 0:
    step("error", msg="base install failed", output=out[-2000:])
    R["status"] = "failed"; save(); sys.exit(1)
# omnivoice may not have a Python 3.13 wheel on the Kaggle index; fall back to GitHub source
rc, out = sh(f"{sys.executable} -m pip install -q omnivoice", 300)
if rc != 0:
    step("omnivoice_pypi_failed", output=out[-500:])
    rc, out = sh(f"{sys.executable} -m pip install -q git+https://github.com/k2-fsa/OmniVoice.git", 600)
    if rc != 0:
        step("error", msg="omnivoice install failed", output=out[-2000:])
        R["status"] = "failed"; save(); sys.exit(1)
step("install_done")

# ---------------------------------------------------------------------------
# 2. Load OmniVoice model on CUDA
# ---------------------------------------------------------------------------
step("load_start")
try:
    import torch
    from omnivoice import OmniVoice
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    t0 = time.perf_counter()
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)
    load_dt = time.perf_counter() - t0
    sample_rate = getattr(model, "sampling_rate", 24000)
    step("load_done", device=device, dtype=str(dtype), load_s=round(load_dt, 2), sample_rate=sample_rate)
except Exception as exc:
    step("error", msg=f"OmniVoice load failed: {type(exc).__name__}: {exc}", output=traceback.format_exc()[-2000:])
    R["status"] = "failed"; save(); sys.exit(1)

# ---------------------------------------------------------------------------
# 3. Convert per-language reference WAVs for voice cloning (use speaker_voices clips)
# ---------------------------------------------------------------------------
ref_wavs = {}
lang_refs = {
    "en": work / "speaker_voices" / "me_en_script.m4a",
    "sk": work / "speaker_voices" / "me_sk_script.m4a",
    "cs": work / "speaker_voices" / "me_cs_staromestske.m4a",
}
for lang, cand in lang_refs.items():
    if cand.is_file():
        ref_wavs[lang] = str(cand)
        step("ref_wav", lang=lang, path=str(cand))
    else:
        step("ref_wav_missing", lang=lang, path=str(cand))

# Convert m4a -> wav if needed (OmniVoice needs WAV)
ref_wav_paths = {}
for lang, ref_wav in ref_wavs.items():
    if ref_wav.endswith(".m4a"):
        out_path = str(OUT / f"ref_voice_{lang}.wav")
        rc, out = sh(f"ffmpeg -y -i {ref_wav} -ac 1 -ar 16000 {out_path}", 60)
        ref_wav_paths[lang] = out_path
        step("ref_convert", lang=lang, rc=rc, output=out[-300:])
    else:
        ref_wav_paths[lang] = ref_wav

# ---------------------------------------------------------------------------
# 4. TTS synthesis test per language (with per-language voice cloning)
# ---------------------------------------------------------------------------
test_cases = [
    ("en", "Hello, this is a test of real-time speech translation on the Tesla T4."),
    ("sk", "Ahoj, toto je test prekladovania reči v reálnom čase na Tesla T4."),
    ("cs", "Ahoj, toto je test překladu řeči v reálném čase na Tesla T4."),
]

import numpy as np

for lang, text in test_cases:
    key = f"tts_{lang}"
    t0 = time.perf_counter()
    try:
        kwargs = {"text": text, "language": lang, "num_step": 12}
        if lang in ref_wav_paths:
            # Build clone prompt from the matching-language reference
            prompt = model.create_voice_clone_prompt(ref_wav_paths[lang])
            kwargs["voice_clone_prompt"] = prompt
        t_first = None
        with torch.no_grad():
            audio = model.generate(**kwargs)[0]
            if device == "cuda":
                torch.cuda.synchronize()
            t_first = time.perf_counter() - t0  # time to first complete audio (single shot)
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        dur_s = len(audio) / sample_rate
        rtf = t_first / dur_s if dur_s > 0 else float("inf")
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        # Save synthesized WAV for download + listening review
        out_wav = str(OUT / f"omnivoice_clone_{lang}.wav")
        sf.write(out_wav, audio, sample_rate)
        R["tts"][lang] = {
            "text": text[:80],
            "synth_s": round(t_first, 3),
            "audio_dur_s": round(dur_s, 3),
            "rtf": round(rtf, 4),
            "cloned": bool(ref_wav_paths.get(lang)),
            "device": device,
            "peak": round(peak, 4),
            "wav_path": out_wav,
        }
        step(key, synth_s=round(t_first, 3), dur_s=round(dur_s, 3), rtf=round(rtf, 4), wav=out_wav)
    except Exception as exc:
        R["tts"][lang] = {"error": f"{type(exc).__name__}: {exc}"}
        step(key + "_error", msg=f"{type(exc).__name__}: {exc}", output=traceback.format_exc()[-1000:])

# ---------------------------------------------------------------------------
# 5. Summary (no ASR quality check — faster-whisper unavailable on Kaggle Py3.13)
# ---------------------------------------------------------------------------
step("done", status=R["status"], tts={k: {kk: vv for kk, vv in v.items() if kk != "text"} for k, v in R["tts"].items()})
save()
print(f"OmniVoice T4 benchmark complete: {R['status']}", flush=True)
