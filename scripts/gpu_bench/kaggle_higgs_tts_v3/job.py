#!/usr/bin/env python3
"""Kaggle T4 kernel: Higgs TTS v3 zero-shot voice cloning benchmark.

Tests bosonai/higgs-audio-v3-tts-4b via SGLang-Omni serving stack:
- Zero-shot TTS (EN/SK/CS)
- Zero-shot voice cloning from reference WAV
- Streaming time-to-first-audio
- RTF per language

Infrastructure: SGLang-Omni (pip install from source/PyPI, NOT plain transformers)
Model: bosonai/higgs-audio-v3-tts-4b
License: Boson Higgs TTS 3 Research and Non-Commercial

Run: kaggle kernels push -p scripts/gpu_bench/kaggle_higgs_tts_v3 --accelerator NvidiaTeslaT4 -t 1800
Outputs: /kaggle/working/higgs_tts_v3_report.json + WAV outputs
"""
import json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

OUT = Path("/kaggle/working")
REPORT = OUT / "higgs_tts_v3_report.json"
T0 = time.time()
R = {"schema": 1, "status": "running", "steps": [], "errors": [], "tts": {}, "acceptance": "route_unverified", "infrastructure": {}}


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

# Record what's already available
rc, torch_out = sh("python3 -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'", 30)
R["infrastructure"]["torch"] = torch_out.strip() if rc == 0 else "not_installed"
rc, cuda_out = sh("nvcc --version 2>/dev/null | tail -1 || echo 'nvcc not found'", 30)
R["infrastructure"]["cuda_compiler"] = cuda_out.strip()
save()

# ---------------------------------------------------------------------------
# 1. Install SGLang-Omni (required for Higgs TTS v3 — NOT plain transformers)
# ---------------------------------------------------------------------------
step("install_sglang_omni")
rc, out = sh("pip install --upgrade pip", 120)
step("pip_upgrade", rc=rc, output=out[-200:] if rc != 0 else "ok")

# Install base sglang + sglang-omni with all dependencies on Kaggle T4 (Python 3.13.16).
# sglang-omni metadata says <3.13 but the runtime may still work; --ignore-requires-python bypasses the check.
rc, out = sh("pip install --ignore-requires-python sglang", 900)
step("sglang_pip", rc=rc, output=out[-800:] if rc != 0 else "ok")

rc, out = sh("pip install --ignore-requires-python sglang-omni", 900)
step("sglang_omni_pip", rc=rc, output=out[-800:] if rc != 0 else "ok")

if rc != 0:
    # Fallback: install from source
    step("install_from_source")
    rc, out = sh("git clone --depth 1 https://github.com/sgl-project/sglang-omni.git /tmp/sglang-omni", 300)
    step("git_clone", rc=rc, output=out[-300:] if rc != 0 else "ok")
    if rc == 0:
        rc, out = sh("cd /tmp/sglang-omni && pip install -e . --ignore-requires-python", 900)
        step("sglang_omni_source", rc=rc, output=out[-800:] if rc != 0 else "ok")

# Verify sgl-omni CLI
rc, out = sh("which sgl-omni 2>/dev/null || python3 -m sglang_omni --help 2>/dev/null | head -3 || echo 'sgl-omni not found'", 30)
step("sgl_cli_check", output=out[-300:])

# Check if sglang-omni is importable
rc, out = sh("python3 -c 'import sglang_omni; print(sglang_omni.__version__)' 2>&1", 30)
R["infrastructure"]["sglang_omni"] = out.strip()
save()

rc_final, _ = sh("which sgl-omni 2>/dev/null; python3 -c 'import sglang_omni' 2>/dev/null && echo IMPORT_OK || echo IMPORT_FAIL", 30)
if rc_final != 0 or "IMPORT_FAIL" in out:
    step("error", msg="SGLang-Omni could not be installed — Higgs TTS v3 requires the SGLang-Omni serving stack, not plain transformers")
    R["status"] = "failed"; save(); sys.exit(1)

step("sglang_omni_ready")

# ---------------------------------------------------------------------------
# 2. Download model weights from HF
# ---------------------------------------------------------------------------
step("download_weights")
rc, out = sh("hf download bosonai/higgs-audio-v3-tts-4b --local-dir /tmp/higgs_tts_v3_weights", 900)
step("weights_downloaded", rc=rc, output=out[-300:] if rc != 0 else "ok")

# ---------------------------------------------------------------------------
# 3. Prepare reference voice files (synthesized fallback if none staged)
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
        wav_out = str(OUT / f"higgs_tts_v3_ref_{lang}.wav")
        try:
            synthesize_fallback(lang, wav_out)
            voice_files[lang] = wav_out
            step("voice_file_synthesized", lang=lang, path=wav_out)
        except Exception as exc:
            step("voice_file_failed", lang=lang, error=str(exc)[:200])

# ---------------------------------------------------------------------------
# 4. Start SGLang-Omni server for Higgs TTS v3
# ---------------------------------------------------------------------------
step("start_server")
server_log = OUT / "higgs_tts_v3_server.log"
server_proc = subprocess.Popen(
    ["sgl-omni", "serve",
     "--model-path", "bosonai/higgs-audio-v3-tts-4b",
     "--port", "8000",
     "--host", "0.0.0.0"],
    stdout=open(server_log, "w"),
    stderr=subprocess.STDOUT,
    env={**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "1"},
)
step("server_started", pid=server_proc.pid)

# Wait for server to be ready
import requests
server_ready = False
for i in range(120):  # up to 120 seconds (model is 5B, slow to load)
    try:
        resp = requests.get("http://localhost:8000/health", timeout=3)
        if resp.status_code == 200:
            server_ready = True
            step("server_ready", wait_s=i * 1)
            break
    except Exception:
        pass
    time.sleep(1)

if not server_ready:
    step("error", msg="SGLang-Omni server did not become ready", output=server_log.read_text()[-2000:] if server_log.exists() else "no log")
    R["status"] = "failed"; save(); sys.exit(1)

# ---------------------------------------------------------------------------
# 5. Zero-shot + voice cloning test per language
# ---------------------------------------------------------------------------
step("run_tests")
test_cases = [
    ("en", "Hello, this is a test of real-time speech translation on the Tesla T4 using Higgs TTS 3 voice cloning."),
    ("sk", "Ahoj, toto je test prekladovania reči v reálnom čase na Tesla T4 pomocou klonovania hlasu Higgs Audio."),
    ("cs", "Ahoj, toto je test překladu řeči v reálném čase na Tesla T4 pomocí klonování hlasu Higgs Audio."),
]

ref_map = {
    "en": voice_files.get("en"),
    "sk": voice_files.get("sk"),
    "cs": voice_files.get("cs"),
}

for lang, text in test_cases:
    ref_wav = ref_map.get(lang)
    t0 = time.perf_counter()
    try:
        wav_out = str(OUT / f"higgs_tts_v3_clone_{lang}.wav")
        payload = {
            "model": "bosonai/higgs-audio-v3-tts-4b",
            "voice": "default",
            "input": text,
            "response_format": "wav",
        }
        if ref_wav:
            payload["references"] = [{"audio_path": ref_wav, "text": ""}]

        # First-audio timing: use streaming to measure TTF
        t_first = None
        with requests.post(
            "http://localhost:8000/v1/audio/speech",
            json={**payload, "stream": True, "response_format": "pcm"},
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            sample_rate = int(resp.headers.get("x-sample-rate", 24000))
            chunks = []
            for chunk in resp.iter_content(chunk_size=None):
                if chunk:
                    if t_first is None:
                        t_first = time.perf_counter() - t0
                    chunks.append(chunk)
            audio = b"".join(chunks)
            sf.write(wav_out, np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0, sample_rate)

        t_total = time.perf_counter() - t0
        dur_s = len(audio) / (sample_rate * 2)
        rtf = t_total / dur_s if dur_s > 0 else float("inf")

        R["tts"][lang] = {
            "text": text[:80],
            "synth_s": round(t_total, 3),
            "first_audio_s": round(t_first, 3) if t_first else None,
            "audio_dur_s": round(dur_s, 3),
            "rtf": round(rtf, 4),
            "cloned": bool(ref_wav),
            "ref_wav": ref_wav,
            "sample_rate": sample_rate,
            "wav_path": wav_out,
        }
        step(f"clone_{lang}", rtf=round(rtf, 4), first_audio_s=round(t_first, 3) if t_first else None)
    except Exception as exc:
        R["tts"][lang] = {"error": type(exc).__name__ + ": " + str(exc)}
        step(f"clone_{lang}_failed", error=str(exc)[:500])

# ---------------------------------------------------------------------------
# 6. Acceptance check
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
    if result.get("first_audio_s", 99) < 0.5:
        checks.append("ttfa<500ms")
    else:
        checks.append(f"ttfa>=500ms ({result['first_audio_s']}s)")
    if result.get("cloned"):
        checks.append("cloned")
    else:
        checks.append("not_cloned")
    acceptance.append(f"{lang}: {', '.join(checks)}")

R["acceptance"] = "; ".join(acceptance)
R["status"] = "done" if not any("error" in v for v in R["tts"].values()) else "partial"
step("done", status=R["status"], acceptance=R["acceptance"])
save()
print(f"Higgs TTS v3 voice cloning complete: {R['status']}", flush=True)
print(f"Acceptance: {R['acceptance']}", flush=True)

# Shutdown server
server_proc.terminate()
try:
    server_proc.wait(timeout=10)
except subprocess.TimeoutExpired:
    server_proc.kill()
