#!/usr/bin/env python3
"""Kaggle T4 kernel: Kyutai Pocket TTS benchmark (fixed — catalog voices, no HF auth needed).

Tests pocket-tts on Kaggle Tesla T4 GPU:
- Built-in catalog voices (alba, marius, etc.) — NO voice cloning, NO HF auth needed
- Cross-lingual: EN model speaking SK text
- Community Czech fine-tune via --config
- RTF, first-audio latency per language

License: MIT (code), per-voice licenses (kyutai/tts-voices).
No Slovak official model — SK tested via cross-lingual (EN model + SK text).

Run: kaggle kernels push -p scripts/gpu_bench/kaggle_pocket_tts --accelerator NvidiaTeslaT4 -t 1800
Outputs: /kaggle/working/pocket_tts_{en,sk,cs,catalog_en,catalog_sk,catalog_cs}.wav + pocket_tts_report.json
"""
import json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import soundfile as sf

OUT = Path("/kaggle/working")
REPORT = OUT / "pocket_tts_report.json"
T0 = time.time()
R = {"schema": 1, "status": "running", "steps": [], "errors": [], "tts": {}, "infrastructure": {}}


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
# 1. Install pocket-tts (CPU PyTorch wheel — no CUDA 13 needed)
# ---------------------------------------------------------------------------
step("install_pocket_tts")
rc, out = sh("pip install --upgrade pip uv", 120)
step("pip_upgrade", rc=rc, output=out[-200:] if rc != 0 else "ok")

rc, out = sh("pip install pocket-tts --extra-index-url https://download.pytorch.org/whl/cpu", 600)
step("pocket_tts_pip", rc=rc, output=out[-800:] if rc != 0 else "ok")
save()

# ---------------------------------------------------------------------------
# 2. Find staged BP repo for voice reference files (for cloning tests later)
# ---------------------------------------------------------------------------
step("repo_lookup")
src = next((p for p in Path("/kaggle/input").rglob("app.py")
            if (p.parent / "benchmark-provenance.json").is_file()), None)
repo = src.parent if src else None
step("repo", path=str(repo) if repo else "none")

# ---------------------------------------------------------------------------
# 3. Run Pocket TTS with CATALOG voices (no voice cloning, no HF auth)
# ---------------------------------------------------------------------------
from pocket_tts import TTSModel

test_texts = {
    "en": "Hello, this is a test of real-time speech translation.",
    "sk": "Dobrý deň, toto je test prekladov reči v reálnom čase.",
    "cs": "Dobrý den, toto je test překladu řeči v reálném čase.",
}

# Test each official language with its default catalog voice
official_langs = {"en": "english", "sk": "english", "cs": "english"}  # SK/CS use EN model cross-lingual

for lang, language in official_langs.items():
    step("load_catalog", lang=lang, language=language)
    try:
        model = TTSModel.load_model(language=language, quantize=True)
        # Try GPU
        try:
            model = model.to("cuda")
            device = "cuda"
        except Exception as e:
            device = "cpu"
            step("gpu_fallback", lang=lang, reason=str(e)[:200])

        # Use catalog voice "alba" (English) — no voice cloning, no HF auth
        voice_state = model.get_state_for_audio_prompt("alba")

        t0 = time.perf_counter()
        chunks = []
        for chunk in model.generate_audio_stream(voice_state, test_texts[lang]):
            chunks.append(chunk.cpu().numpy() if hasattr(chunk, 'cpu') else chunk.numpy())
        first_chunk_s = time.perf_counter() - t0

        audio = np.concatenate(chunks) if chunks else np.array([])
        audio_s = len(audio) / 24000 if len(audio) > 0 else 0
        wall_s = time.perf_counter() - t0
        rtf = wall_s / audio_s if audio_s > 0 else None

        out_path = OUT / f"pocket_tts_catalog_{lang}.wav"
        sf.write(str(out_path), audio, 24000)

        R["tts"][f"catalog_{lang}"] = {
            "device": device,
            "voice_mode": "catalog:alba",
            "language": language,
            "first_audio_s": round(first_chunk_s, 3),
            "wall_s": round(wall_s, 3),
            "audio_s": round(audio_s, 2),
            "rtf": round(rtf, 3) if rtf else None,
            "output_wav": str(out_path),
            "text": test_texts[lang],
            "note": "Catalog voice (no cloning, no HF auth)",
        }
        step("done_catalog", lang=lang, device=device, rtf=round(rtf, 3) if rtf else None)

    except Exception as e:
        step("fail_catalog", lang=lang, err=f"{type(e).__name__}: {str(e)[:300]}")
        R["errors"].append(f"catalog_{lang}: {type(e).__name__}: {str(e)[:500]}")
        R["tts"][f"catalog_{lang}"] = {"error": str(e)[:500]}
        save()

# ---------------------------------------------------------------------------
# 4. Community Czech fine-tune via --config (model load test only)
# ---------------------------------------------------------------------------
step("community_cs")
try:
    model = TTSModel.load_model(
        config="hf://vvolhejn/pocket-tts-czech/czech.yaml@7b7760dd0fe994a0800f2fdbc837dc4b8f219d1c",
        quantize=True,
    )
    try:
        model = model.to("cuda")
        device = "cuda"
    except Exception as e:
        device = "cpu"
        step("cs_gpu_fallback", reason=str(e)[:200])

    # Community Czech model: no predefined voices, test model loads only
    R["tts"]["community_cs"] = {
        "device": device,
        "voice_mode": "n/a",
        "config": "vvolhejn/pocket-tts-czech",
        "note": "Community Czech fine-tune loaded successfully; needs reference WAV for synthesis",
    }
    step("done_community_cs", device=device)

except Exception as e:
    step("fail_community_cs", err=f"{type(e).__name__}: {str(e)[:300]}")
    R["errors"].append(f"community_cs: {type(e).__name__}: {str(e)[:500]}")
    R["tts"]["community_cs"] = {"error": str(e)[:500]}
    save()

# ---------------------------------------------------------------------------
# 5. Voice cloning test with reference WAV (needs HF auth — will fail gracefully)
# ---------------------------------------------------------------------------
step("voice_clone_test")
try:
    import soundfile as sf
    import numpy as np
    # Try to find a reference WAV from the staged repo
    ref_wav = None
    if repo:
        for p in ["speaker_voices/me_en_rainbow.m4a", "processed/testaudio/en_paced.wav"]:
            cand = repo / p
            if cand.is_file():
                ref_wav = str(cand)
                break
    if ref_wav:
        wav_path = OUT / "ref_en.wav"
        sh(f'ffmpeg -y -v error -i {ref_wav} -t 10 -ac 1 -ar 24000 {wav_path}', 60)
        ref_wav = str(wav_path)

        model = TTSModel.load_model(language="english", quantize=True).to("cuda")
        try:
            voice_state = model.get_state_for_audio_prompt(ref_wav)
            voice_mode = "clone"
        except Exception as e:
            step("clone_auth_fail", reason=str(e)[:200])
            voice_state = model.get_state_for_audio_prompt("alba")
            voice_mode = "catalog:alba"

        t0 = time.perf_counter()
        chunks = []
        for chunk in model.generate_audio_stream(voice_state, test_texts["en"]):
            chunks.append(chunk.cpu().numpy() if hasattr(chunk, 'cpu') else chunk.numpy())
        wall_s = time.perf_counter() - t0
        audio = np.concatenate(chunks) if chunks else np.array([])
        audio_s = len(audio) / 24000 if len(audio) > 0 else 0

        out_path = OUT / "pocket_tts_clone_en.wav"
        sf.write(str(out_path), audio, 24000)

        R["tts"]["clone_en"] = {
            "device": "cuda",
            "voice_mode": voice_mode,
            "first_audio_s": 0,  # not measured for clone
            "wall_s": round(wall_s, 3),
            "audio_s": round(audio_s, 2),
            "rtf": round(wall_s / audio_s, 3) if audio_s > 0 else None,
            "output_wav": str(out_path),
            "text": test_texts["en"],
        }
        step("done_clone_en", voice_mode=voice_mode)
    else:
        step("clone_no_ref", msg="no reference WAV found")

except Exception as e:
    step("fail_clone", err=f"{type(e).__name__}: {str(e)[:300]}")
    R["errors"].append(f"clone_en: {type(e).__name__}: {str(e)[:500]}")
    R["tts"]["clone_en"] = {"error": str(e)[:500]}
    save()

R["status"] = "complete"
save()
print("Pocket TTS kernel complete.", flush=True)