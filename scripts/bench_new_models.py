#!/usr/bin/env python3
"""New-model benchmark harness (runs in .venv-eval, NOT .venv).

Protocol (same for every candidate — no paper verdicts):
  fixed SK v2b clips -> WER/CER (+chrF for MT/S2ST) + RTF + device -> matrix JSON.

Clips (local-only, never committed):
  speaker_voices/sk_trhove_rano_v2b.m4a  (109s, ref in speaker_voices.json)
  speaker_voices/sk_script_reading.m4a   (134s, ref in speaker_voices.json)

Out: processed/new_models/<engine>_matrix.json (read by the Lab like sk_direction).

Usage (on charger):
  .venv-eval/bin/pip install -r requirements-eval.txt   # once
  .venv-eval/bin/python scripts/bench_new_models.py --engine seamless --clips sk_trhove_rano_v2b
  .venv-eval/bin/python scripts/bench_new_models.py --engine zipformer --clips sk_trhove_rano_v2b,sk_script_reading

Each --engine is implemented as a function below. Unimplemented engines exit with
a TODO + install notes instead of failing silently.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "processed", "new_models")

CLIPS = {
    "sk_trhove_rano_v2b": "speaker_voices/sk_trhove_rano_v2b.m4a",
    "sk_script_reading": "speaker_voices/sk_script_reading.m4a",
}


def load_audio(path: str):
    import librosa
    import numpy as np
    wav, _ = librosa.load(os.path.join(REPO_ROOT, path), sr=16000, mono=True)
    return np.asarray(wav, dtype=np.float32)


def load_sk_reference(clip_id: str) -> str:
    meta = json.load(open(os.path.join(REPO_ROOT, "speaker_voices", "speaker_voices.json"),
                          encoding="utf-8"))
    want = os.path.basename(CLIPS[clip_id])
    for e in meta:
        if os.path.basename(e.get("path", "")) == want:
            return (e.get("transcribed_text") or "").strip()
    raise KeyError(f"no reference for {clip_id}")


def score_wer(ref: str, hyp: str) -> dict:
    import re

    def norm(t):
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower(), flags=re.UNICODE)).strip()

    import jiwer
    return {"wer": round(jiwer.wer(norm(ref), norm(hyp)), 4),
            "cer": round(jiwer.cer(norm(ref), norm(hyp)), 4)}


def run_seamless(clips: list[str]) -> list[dict]:
    raise NotImplementedError(
        "TODO spike 1: pip install seamless_communication (fairseq2); load "
        "SeamlessM4T v2 (~2GB first download); S2ST sk->en on clips; record per-clip "
        "transcript/translation/audio + RTF. Kill fast if MPS/CPU RTF disappoints.")


def run_zipformer(clips: list[str]) -> list[dict]:
    raise NotImplementedError(
        "TODO spike 2a: sherpa-onnx streaming Zipformer (multilingual/ESB variant); "
        "check SK or CZ model availability first; stream clips in 100ms chunks;")


def run_cohere(clips: list[str]) -> list[dict]:
    raise NotImplementedError("TODO spike 2b: Cohere Transcribe ONNX; verify SK support + packaging.")


def run_chatterbox(clips: list[str]) -> list[dict]:
    raise NotImplementedError(
        "TODO spike 3a: Chatterbox zero-shot (6-10s v2b ref) -> SK synth; feed back "
        "through small-sk like scripts/engine_ab.py; compare WER vs omni_zeroshot 0.054.")


def run_kokoro(clips: list[str]) -> list[dict]:
    raise NotImplementedError("TODO spike 3b: Kokoro SK/CZ-proxy; 30-min box, keep negative if fail.")


ENGINES = {"seamless": run_seamless, "zipformer": run_zipformer, "cohere": run_cohere,
           "chatterbox": run_chatterbox, "kokoro": run_kokoro}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=sorted(ENGINES))
    ap.add_argument("--clips", default="sk_trhove_rano_v2b")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    wanted = [c for c in args.clips.split(",") if c in CLIPS]
    if not wanted:
        ap.error(f"unknown clips; choose from {sorted(CLIPS)}")
    t0 = time.perf_counter()
    try:
        rows = ENGINES[args.engine](wanted)
    except NotImplementedError as e:
        print(f"{args.engine}: {e}")
        sys.exit(3)
    out = os.path.join(OUT_DIR, f"{args.engine}_matrix.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"engine": args.engine, "clips": wanted,
                   "wall_s": round(time.perf_counter() - t0, 1),
                   "results": rows}, f, indent=2, ensure_ascii=False)
    print(f"wrote {os.path.relpath(out, REPO_ROOT)}")


if __name__ == "__main__":
    main()
