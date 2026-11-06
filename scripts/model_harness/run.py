#!/usr/bin/env python3
"""BP model test harness — 8 GB VRAM bounded model evaluation for T4.

Tests STT/MT/TTS/S2S candidate models against fixed multi-language clips and
records WER/chrF/RTF/VRAM/latency + ear-grade wav. Declared VRAM per engine
comes from backend/registry.py Capability.resource; a model whose declared VRAM
exceeds --vram-limit is refused before loading.

Designed to run on Kaggle T4 (free, 15.6 GB but we bound to 8 GB) and locally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from backend.registry import engines as registry_engines, Capability  # noqa: E402

OUT_DIR = REPO_ROOT / "processed" / "model_harness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed multi-language test clips (must exist in speaker_voices/ or bench/audio/)
CLIPS = {
    "sk": "speaker_voices/me_sk_script.m4a",
    "en": "speaker_voices/me_en_script.m4a",
    "cs": "speaker_voices/me_cs_staromestske.m4a",
}
# Ground-truth references (owner's transcripts from speaker_voices.json)
REFS = {
    "sk": "vitajte v tejto živej ukážke prekladu reči",
    "en": "good morning and welcome to this live demonstration",
    "cs": "dobrý den a vítáme vás v této ukázce překladu",
}


def vram_gb(resource: str) -> float | None:
    """Parse declared VRAM from Capability.resource string like '~1.33 GB VRAM on T4'."""
    import re
    m = re.search(r"([\d.]+)\s*GB", resource)
    return float(m.group(1)) if m else None


def engine_vram(stage: str, name: str) -> float | None:
    cap = next((c for c in registry_engines(stage) if c.name == name), None)
    if cap is None:
        return None
    return vram_gb(cap.resource)


def out_path(stage: str, model: str, lang: str) -> Path:
    tag = hashlib.md5(f"{stage}|{model}|{lang}".encode()).hexdigest()[:8]
    return OUT_DIR / f"{stage}_{model}_{lang}_{tag}.json"


def run_stt(model: str, lang: str, clip: str, ref: str | None, device: str) -> dict:
    from backend.stt.faster_whisper_stt import FasterWhisperSTT  # noqa: E402
    import librosa  # noqa: E402
    import numpy as np  # noqa: E402
    try:
        from jiwer import wer  # noqa: E402
    except ImportError:
        wer = None  # noqa: F841

    wav, sr = librosa.load(str(REPO_ROOT / clip), sr=16000, mono=True)
    stt = FasterWhisperSTT(model_size=model, device=device, compute_type="int8")
    t0 = time.perf_counter()
    segs, stt_t, detected = stt.transcribe_audio(
        np.asarray(wav, dtype=np.float32), sr, language=lang
    )
    latency = time.perf_counter() - t0
    text = " ".join(s["text"] if isinstance(s, dict) else getattr(s, "text", "") for s in segs).strip()
    record = {
        "stage": "stt", "model": model, "lang": lang, "device": device,
        "clip": clip, "transcript": text, "detected_lang": detected,
        "latency_s": round(latency, 3), "rtf": round(latency / (len(wav) / sr), 3),
    }
    if ref and wer:
        record["wer"] = round(wer(ref, text), 4)
        record["reference"] = ref
    return record


def run_mt(model: str, text: str, src: str, tgt: str, device: str) -> dict:
    from backend.mt.ctranslate2_mt import CTranslate2MT  # noqa: E402

    mt_path = model if "/" in model else str(REPO_ROOT / "ct2_models" / model)
    mt = CTranslate2MT(model_path=mt_path, device=device)
    t0 = time.perf_counter()
    parts, _ = mt.translate_segments([text], src, tgt)
    translation = " ".join(parts).strip()
    latency = time.perf_counter() - t0
    return {
        "stage": "mt", "model": model, "src": src, "tgt": tgt, "device": device,
        "input": text, "translation": translation,
        "latency_s": round(latency, 3),
    }


def run_tts(model: str, text: str, lang: str, device: str) -> dict:
    from backend.tts.base import TTS_ENGINES  # noqa: E402
    import soundfile as sf  # noqa: E402

    tts = TTS_ENGINES[model]()
    t0 = time.perf_counter()
    audio, sr, syn_s = tts.synthesize(text, language=lang)
    latency = time.perf_counter() - t0
    out_wav = OUT_DIR / f"tts_{model}_{lang}.wav"
    sf.write(str(out_wav), audio, sr)
    return {
        "stage": "tts", "model": model, "lang": lang, "device": device,
        "text": text, "audio_out": str(out_wav),
        "syn_s": round(syn_s, 3), "audio_s": round(len(audio) / sr, 3),
        "rtf": round(syn_s / (len(audio) / sr), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(prog="model_harness")
    ap.add_argument("--stage", required=True, choices=["stt", "mt", "tts", "s2s"])
    ap.add_argument("--model", required=True, help="engine/registry name or model path")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps", "auto"])
    ap.add_argument("--vram-limit", type=float, default=8.0, help="max GB VRAM allowed")
    ap.add_argument("--clip", default=None, help="clip path (for stt)")
    ap.add_argument("--lang", default=None, help="language code")
    ap.add_argument("--ref", default=None, help="reference text for WER (stt)")
    ap.add_argument("--text", default=None, help="input text (mt/tts)")
    ap.add_argument("--src", default="en", help="source language (mt)")
    ap.add_argument("--tgt", default="sk", help="target language (mt)")
    a = ap.parse_args()

    # VRAM guard
    declared = engine_vram(a.stage, a.model)
    if declared is not None and declared > a.vram_limit:
        print(f"REFUSED: {a.stage}/{a.model} declares {declared} GB VRAM > limit {a.vram_limit} GB")
        sys.exit(1)

    if a.stage == "stt":
        clip = a.clip or CLIPS.get(a.lang or "en")
        ref = a.ref or REFS.get(a.lang)
        record = run_stt(a.model, a.lang or "en", clip, ref, a.device)
    elif a.stage == "mt":
        record = run_mt(a.model, a.text or "Good morning.", a.src, a.tgt, a.device)
    elif a.stage == "tts":
        record = run_tts(a.model, a.text or "Hello world.", a.lang or "en", a.device)
    else:
        print("s2s stage not yet implemented (SeamlessM4T v2 needs fairseq2)")
        sys.exit(1)

    path = out_path(a.stage, a.model, a.lang or a.tgt)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
