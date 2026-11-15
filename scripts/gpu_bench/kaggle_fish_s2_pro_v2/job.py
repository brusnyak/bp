#!/usr/bin/env python3
"""Kaggle T4 kernel: Fish Audio S2-Pro TTS via SGLang-Omni (proper standalone env).

Tests fishaudio/s2-pro on Kaggle Tesla T4 GPU:
- Sets up conda Python 3.12 env (Fish Audio/SGLang-Omni require Python 3.12, NOT Kaggle's default 3.13)
- Installs SGLang-Omni from source + descript-audio-codec
- Serves the model via sgl-omni serve with s2pro config
- Tests zero-shot TTS and voice cloning (EN/SK/CS) via /v1/audio/speech API
- Measures RTF, first-audio latency

CRITICAL CONSTRAINTS:
- S2-Pro is a 5B Dual-AR model; Fish Audio recommends >=24GB VRAM. Kaggle T4 has 16GB.
- SGLang-Omni requires CUDA 13; Kaggle T4 typically has CUDA 12.x drivers.
Both may cause failure — kernel reports exactly what fails rather than silently erroring.

License: Fish Audio Community License (code BSD-3-Clause), S2-Pro weights non-commercial.

Run: kaggle kernels push -p scripts/gpu_bench/kaggle_fish_s2_pro_v2 --accelerator NvidiaTeslaT4 -t 1800
Outputs: /kaggle/working/fish_s2_pro_{en,sk,cs}.wav + fish_s2_pro_v2_report.json
"""
import json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

OUT = Path("/kaggle/working")
REPORT = OUT / "fish_s2_pro_v2_report.json"
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


def cleanup():
    global SERVER_PROC
    if SERVER_PROC:
        try:
            SERVER_PROC.terminate()
            SERVER_PROC.wait(timeout=10)
        except Exception:
            try:
                SERVER_PROC.kill()
            except Exception:
                pass


import atexit
atexit.register(cleanup)

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
# 1. Check CUDA version — SGLang-Omni requires CUDA 13
# ---------------------------------------------------------------------------
step("cuda_check")
rc, out = sh("nvcc --version 2>/dev/null | tail -3 || echo 'nvcc not found'", 30)
step("nvcc", output=out.strip()[-500:])

# ---------------------------------------------------------------------------
# 2. Create conda environment with Python 3.12
# ---------------------------------------------------------------------------
step("conda_env")
rc, out = sh("conda create -n tts312 python=3.12 -y", 300)
step("conda_create", rc=rc, output=out[-500:] if rc != 0 else "ok")
save()

CONDA_ACTIVATE = "conda activate tts312 &&"
CONDA_PYTHON = "/opt/conda/envs/tts312/bin/python"

if rc == 0:
    rc, out = sh(f"{CONDA_PYTHON} --version", 30)
    step("conda_python_version", output=out.strip() if rc == 0 else "failed")
else:
    step("conda_failed", error=out[-500:])
    R["status"] = "failed"; save(); sys.exit(1)

# ---------------------------------------------------------------------------
# 3. Install SGLang-Omni in the conda env
# ---------------------------------------------------------------------------
step("install_sglang_omni")
rc, out = sh(f"{CONDA_ACTIVATE} pip install --upgrade pip uv", 120)
step("pip_upgrade", rc=rc, output=out[-200:] if rc != 0 else "ok")

rc, out = sh(f"{CONDA_ACTIVATE} pip install --pre sglang", 900)
step("sglang_pip", rc=rc, output=out[-800:] if rc != 0 else "ok")

if rc == 0:
    rc, out = sh(f"{CONDA_ACTIVATE} pip install sglang-omni", 900)
    step("sglang_omni_pip", rc=rc, output=out[-800:] if rc != 0 else "ok")

if rc != 0:
    step("install_from_source")
    rc, out = sh(f"{CONDA_ACTIVATE} git clone --depth 1 https://github.com/sgl-project/sglang-omni.git /tmp/sglang-omni", 300)
    step("git_clone", rc=rc, output=out[-300:] if rc != 0 else "ok")
    if rc == 0:
        rc, out = sh(f"{CONDA_ACTIVATE} cd /tmp/sglang-omni && pip install -e .", 900)
        step("sglang_omni_source", rc=rc, output=out[-800:] if rc != 0 else "ok")

# Fish Audio S2-Pro needs descript-audio-codec
rc, out = sh(f"{CONDA_ACTIVATE} pip install descript-audiotools==0.7.2 descript-audio-codec==1.0.0", 300)
step("dac_deps", rc=rc, output=out[-300:] if rc != 0 else "ok")
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
# 5. Serve Fish Audio S2-Pro via SGLang-Omni
# ---------------------------------------------------------------------------
step("serve_s2pro")
MODEL_PATH = "fishaudio/s2-pro"
PORT = 8000

rc, out = sh(f"{CONDA_ACTIVATE} huggingface-cli download {MODEL_PATH} --local-dir /tmp/fish-s2-pro", 600)
step("hf_download", rc=rc, output=out[-500:] if rc != 0 else "ok")

if rc == 0:
    # CPU preprocessing for reference encoding (avoids GPU memory pressure on 16GB T4)
    server_cmd = (
        f"{CONDA_ACTIVATE} sgl-omni serve "
        f"--model-path /tmp/fish-s2-pro "
        f"--config examples/configs/s2pro_tts.yaml "
        f"--host 0.0.0.0 --port {PORT} "
        f"--allowed-local-media-path /tmp "
        f"--preprocessing.gpu null "
        f"--preprocessing.total_reserve_bytes null "
    )
    SERVER_PROC = subprocess.Popen(server_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    step("server_started", pid=SERVER_PROC.pid)

    import urllib.request, urllib.error
    server_ready = False
    for i in range(60):
        time.sleep(5)
        try:
            req = urllib.request.Request(f"http://localhost:{PORT}/v1/models")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    server_ready = True
                    break
        except Exception:
            pass
        step("waiting_server", attempt=i + 1)

    if server_ready:
        step("server_ready")
    else:
        try:
            server_out = SERVER_PROC.stdout.read() if SERVER_PROC.stdout else ""
        except Exception:
            server_out = ""
        step("server_not_ready", output=server_out[-1000:])
        R["errors"].append("SGLang-Omni server did not become ready — likely CUDA 13 requirement or VRAM insufficient on Kaggle T4")
        R["status"] = "failed"; save(); sys.exit(1)
else:
    step("hf_download_failed", error=out[-500:])
    R["status"] = "failed"; save(); sys.exit(1)

# ---------------------------------------------------------------------------
# 6. Test TTS per language via API
# ---------------------------------------------------------------------------
import requests

for lang in ("en", "sk", "cs"):
    step("tts", lang=lang)
    try:
        ref_wav = voice_files.get(lang, voice_files.get("en"))
        payload = {
            "model": MODEL_PATH,
            "voice": "default",
            "input": {
                "en": "Hello, this is a test of real-time speech translation.",
                "sk": "Dobrý deň, toto je test prekladov reči v reálnom čase.",
                "cs": "Dobrý den, toto je test překladu řeči v reálném čase.",
            }[lang],
        }
        if ref_wav:
            payload["voice"] = ref_wav

        t0 = time.perf_counter()
        resp = requests.post(
            f"http://localhost:{PORT}/v1/audio/speech",
            json=payload,
            timeout=300,
        )
        first_audio_s = time.perf_counter() - t0

        if resp.status_code == 200:
            out_path = OUT / f"fish_s2_pro_{lang}.wav"
            out_path.write_bytes(resp.content)
            import soundfile as sf
            info = sf.info(str(out_path))
            audio_s = info.frames / info.samplerate
            wall_s = time.perf_counter() - t0
            rtf = wall_s / audio_s if audio_s > 0 else None
            R["tts"][lang] = {
                "status_code": resp.status_code,
                "first_audio_s": round(first_audio_s, 3),
                "wall_s": round(wall_s, 3),
                "audio_s": round(audio_s, 2),
                "rtf": round(rtf, 3) if rtf else None,
                "output_wav": str(out_path),
                "voice_clone": bool(ref_wav),
            }
            step("tts_done", lang=lang, rtf=round(rtf, 3) if rtf else None, first_audio=round(first_audio_s, 3))
        else:
            step("tts_http_error", lang=lang, status=resp.status_code, output=resp.text[-500:])
            R["errors"].append(f"{lang}: HTTP {resp.status_code}: {resp.text[-300:]}")
            R["tts"][lang] = {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        step("tts_fail", lang=lang, err=f"{type(e).__name__}: {str(e)[:300]}")
        R["errors"].append(f"{lang}: {type(e).__name__}: {str(e)[:500]}")
        R["tts"][lang] = {"error": str(e)[:500]}
    save()

R["status"] = "complete"
save()
print("Fish Audio S2-Pro v2 kernel complete.", flush=True)