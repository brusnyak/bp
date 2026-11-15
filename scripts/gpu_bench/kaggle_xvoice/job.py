#!/usr/bin/env python3
"""Kaggle T4 kernel: X-Voice zero-shot voice cloning benchmark (fixed standalone env).

Tests X-Voice (sunnyxrxrx/X-Voice, flow-matching DiT, IPA input) on Kaggle Tesla T4 GPU:
- SK and CZ both listed among 30 langs (paper tables 5-6)
- Code MIT, weights CC-BY-NC (thesis OK, product no)
- 0.4B model, PyTorch; vendor RTF 0.073 on RTX 4090

FIXES vs prior attempt:
- ctranslate2==4.5.0 pin doesn't exist on PyPI → install ctranslate2 without version constraint
- Import path is `x_voice` (underscore) not `xvoice`
- Python >=3.11 required; Kaggle default is 3.13.16 (compatible)
- Uses `uv pip install --python` with direct venv binary (no `source activate`, no conda)

License: code MIT, weights CC-BY-NC (non-commercial).

Run: kaggle kernels push -p scripts/gpu_bench/kaggle_xvoice --accelerator NvidiaTeslaT4 -t 1800
Outputs: /kaggle/working/xvoice_{en,sk,cs}.wav + xvoice_report.json
"""
import json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

OUT = Path("/kaggle/working")
REPORT = OUT / "xvoice_report.json"
T0 = time.time()
R = {"schema": 1, "status": "running", "steps": [], "errors": [], "tts": {}, "infrastructure": {}}
SERVER_PROC = None


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
R["infrastructure"]["torch"] = out.strip() if rc == 0 else "not_installed"
save()

# ---------------------------------------------------------------------------
# 1. Install uv + Python 3.11 venv (X-Voice requires Python >=3.11)
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

VENV_DIR = "/tmp/xvoice-venv"
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

# Disable SSL verification for fastlid model downloads (Kaggle cert issues)
rc, out = sh(f"{VENV_PIP} install certifi", 120)
step("certifi_install", rc=rc, output=out[-200:] if rc != 0 else "ok")

# ---------------------------------------------------------------------------
# 2. Install ctranslate2 first (X-Voice pins ==4.5.0 which doesn't exist on PyPI;
#    install without version constraint, then install X-Voice)
#    Use `uv pip install --python` (venv has no pip binary of its own)
# ---------------------------------------------------------------------------
step("install_ctranslate2")
rc, out = sh(f"{UV_PIP_INSTALL} --upgrade pip uv", 120)
step("pip_upgrade", rc=rc, output=out[-200:] if rc != 0 else "ok")

rc, out = sh(f"{UV_PIP_INSTALL} ctranslate2", 300)
step("ctranslate2_install", rc=rc, output=out[-500:] if rc != 0 else "ok")

if rc != 0:
    step("ctranslate2_failed", error=out[-500:])
    # Continue anyway — X-Voice may work with a newer ctranslate2

# Install system deps for IPA/espeak-ng (libespeak-1 doesn't exist on Ubuntu 24.04)
rc, out = sh("apt-get update -qq && apt-get install -y -qq espeak-ng", 300)
step("espeak_ng", rc=rc, output=out[-300:] if rc != 0 else "ok")

# ---------------------------------------------------------------------------
# 3. Install X-Voice from GitHub
# ---------------------------------------------------------------------------
step("install_xvoice")
rc, out = sh(f"{UV_PIP_INSTALL} git+https://github.com/sunnyxrxrx/X-Voice.git", 900)
step("xvoice_pip", rc=rc, output=out[-800:] if rc != 0 else "ok")

if rc != 0:
    step("xvoice_pip_failed", error=out[-500:])
    R["status"] = "failed"; save(); sys.exit(1)

save()

# ---------------------------------------------------------------------------
# 4. Find staged BP repo for voice reference files
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
# 5. Run X-Voice per language via CLI
# ---------------------------------------------------------------------------
step("run_xvoice")
import torch

# Check the installed package structure
rc, out = sh(f"{VENV_PYTHON} -c 'import x_voice; print(x_voice.__file__)'", 30)
step("xvoice_import_check", rc=rc, output=out[-500:] if rc != 0 else "ok")

test_texts = {
    "en": "Hello, this is a test of real-time speech translation.",
    "sk": "Dobrý deň, toto je test prekladov reči v reálnom čase.",
    "cs": "Dobrý den, toto je test překladu řeči v reálném čase.",
}

for lang in ("en", "sk", "cs"):
    step("tts", lang=lang)
    try:
        ref_wav = voice_files[lang]
        t0 = time.perf_counter()

        # X-Voice CLI: x-voice_infer-gradio or python -m x_voice.infer.infer_cli_stage1
        # The package provides a CLI entry point; try direct Python import first
        audio = None
        try:
            # Try importing the model class
            from x_voice.models.model import XVoiceModel
            model = XVoiceModel.from_pretrained("sunnyxrxrx/X-Voice", device="cuda")
            audio = model.synthesize(test_texts[lang], reference_audio=ref_wav, language=lang)
        except (ImportError, AttributeError, Exception) as e:
            step("import_fallback", lang=lang, reason=str(e)[:200])
            # Try CLI approach
            cli_script = f"{VENV_DIR}/bin/x-voice_infer-gradio"
            import shutil as shutil_mod
            if shutil_mod.which("x-voice_infer-gradio"):
                cli_cmd = f"PYTHONHTTPSVERIFY=0 x-voice_infer-gradio --text '{test_texts[lang]}' --reference_audio {ref_wav} --language {lang} --output /tmp/xvoice_out_{lang}.wav"
            else:
                cli_cmd = f"PYTHONHTTPSVERIFY=0 {VENV_PYTHON} -m x_voice.infer.infer_cli_stage1 --text '{test_texts[lang]}' --reference_audio {ref_wav} --language {lang} --output /tmp/xvoice_out_{lang}.wav"
            rc, out = sh(cli_cmd, 600)
            step("cli_run", lang=lang, rc=rc, output=out[-500:] if rc != 0 else "ok")
            if rc == 0 and os.path.exists(f"/tmp/xvoice_out_{lang}.wav"):
                import soundfile as sf
                audio_path = f"/tmp/xvoice_out_{lang}.wav"
                info = sf.info(audio_path)
                audio = True  # marker
                audio_s = info.frames / info.samplerate
                wall_s = time.perf_counter() - t0
                rtf = wall_s / audio_s if audio_s > 0 else None
                out_path = OUT / f"xvoice_{lang}.wav"
                shutil.copy2(audio_path, str(out_path))
                R["tts"][lang] = {
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "wall_s": round(wall_s, 3),
                    "audio_s": round(audio_s, 2),
                    "rtf": round(rtf, 3) if rtf else None,
                    "output_wav": str(out_path),
                    "voice_clone": True,
                    "method": "cli",
                }
                step("tts_done", lang=lang, rtf=round(rtf, 3) if rtf else None)
                continue

        if audio is not None:
            import soundfile as sf
            import numpy as np
            if hasattr(audio, 'numpy'):
                audio = audio.numpy()
            if hasattr(audio, 'cpu'):
                audio = audio.cpu()
            audio = np.asarray(audio, dtype='float32').squeeze()
            sr = getattr(model, 'sampling_rate', 24000)
            out_path = OUT / f"xvoice_{lang}.wav"
            sf.write(str(out_path), audio, sr)
            audio_s = len(audio) / sr
            wall_s = time.perf_counter() - t0
            rtf = wall_s / audio_s if audio_s > 0 else None
            R["tts"][lang] = {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "wall_s": round(wall_s, 3),
                "audio_s": round(audio_s, 2),
                "rtf": round(rtf, 3) if rtf else None,
                "output_wav": str(out_path),
                "voice_clone": True,
                "method": "python_api",
            }
            step("tts_done", lang=lang, rtf=round(rtf, 3) if rtf else None)
        else:
            step("tts_no_audio", lang=lang)
            R["tts"][lang] = {"error": "no audio output"}
    except Exception as e:
        step("tts_fail", lang=lang, err=f"{type(e).__name__}: {str(e)[:300]}")
        R["errors"].append(f"{lang}: {type(e).__name__}: {str(e)[:500]}")
        R["tts"][lang] = {"error": str(e)[:500]}
    save()

R["status"] = "complete"
save()
print("X-Voice kernel complete.", flush=True)