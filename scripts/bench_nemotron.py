#!/usr/bin/env python3
"""Nemotron 3.5 ASR streaming bench (needs the pure-C runtime, NOT sherpa).

Why not sherpa_onnx: 1.13.8 (latest) ships no online transducer config in
Python, and its offline from_transducer rejects the streaming INT8 bundle
('vocab_size' missing from decoder metadata). The C runtime builds the
original cache-aware streaming model directly.

Build once (2 min, /tmp is fine — binary rebuilds from source):
  git clone --depth 1 https://github.com/kdrkdrkdr/nemotron-asr-streaming.c /tmp/nemotron-c
  cd /tmp/nemotron-c && make
  .venv-eval/bin/python -c "from huggingface_hub import snapshot_download;
    snapshot_download('kdrkdrkdr/nemotron-3.5-asr-streaming-0.6b-w8a8',
    local_dir='model')"

Clips must be 16-bit PCM 16 kHz mono WAVs (ffmpeg -ac 1 -ar 16000 -sample_fmt s16).
Prompts: sk-SK for Slovak takes, en-US for the English control.

Usage: NEMOTRON_BIN=/tmp/nemotron-c/nemotron_asr \
       NEMOTRON_MODEL=/tmp/nemotron-c/model/nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin \
       .venv/bin/python scripts/bench_nemotron.py
Writes processed/new_models/nemotron_matrix.json (same protocol as bench_new_models.py).
"""

import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from bench_new_models import load_sk_reference, score_wer  # noqa: E402

BIN = os.environ.get("NEMOTRON_BIN", "/tmp/nemotron-c/nemotron_asr")
MODEL = os.environ.get("NEMOTRON_MODEL", "/tmp/nemotron-c/model/"
                        "nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin")
WAVDIR = os.environ.get("NEMOTRON_WAVDIR", "/tmp/nemotron-c")
OUT = os.path.join(REPO_ROOT, "processed", "new_models", "nemotron_matrix.json")

CLIPS = [("me_sk_trhove_b", "me_sk_trhove_b.wav", "sk-SK", False),
         ("me_sk_script", "me_sk_script.wav", "sk-SK", False),
         ("me_en_rainbow_b", "me_en_rainbow_b.wav", "en-US", True)]


def load_en_reference() -> str:
    meta = json.load(open(os.path.join(REPO_ROOT, "speaker_voices",
                                       "speaker_voices.json"), encoding="utf-8"))
    for e in meta:
        if os.path.basename(e.get("path", "")) == "me_en_rainbow_b.m4a":
            return (e.get("transcribed_text") or "").strip()
    raise KeyError("no EN reference")


def main() -> None:
    results = []
    for clip, wav, lang, is_en in CLIPS:
        ref = load_en_reference() if is_en else load_sk_reference(
            "me_sk_trhove_b" if "trhove" in clip else "me_sk_script")
        audio_s = round(os.path.getsize(os.path.join(WAVDIR, wav)) / 32000, 2)
        t1 = time.time()
        p = subprocess.run([BIN, "-m", MODEL, "-i", os.path.join(WAVDIR, wav),
                            "-l", lang, "--strip-tags"],
                           capture_output=True, text=True)
        dt = time.time() - t1
        if p.returncode != 0:
            print(f"{clip}: binary failed: {p.stderr[-300:]}", flush=True)
            sys.exit(1)
        hyp = p.stdout.strip()
        sc = score_wer(ref, hyp)
        row = {"engine": "nemotron_streaming", "clip": clip, "prompt": lang,
               "audio_s": audio_s, "infer_s": round(dt, 2),
               "rtf": round(dt / audio_s, 3), "wer": sc["wer"], "cer": sc["cer"],
               "hyp_text": hyp[:220]}
        results.append(row)
        print(f"{clip:16s} [{lang}] WER {sc['wer']:.3f} CER {sc['cer']:.3f} "
              f"{dt:.1f}s for {audio_s:.0f}s audio (RTF {dt / audio_s:.2f})",
              flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"engine": "nemotron_streaming_0.6b_560ms_w8a8",
                   "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "method": "kdrkdrkdr/nemotron-asr-streaming.c, whole-file "
                             "streaming decode, greedy",
                   "results": results}, f, indent=2, ensure_ascii=False)
    print(f"wrote {os.path.relpath(OUT, REPO_ROOT)}")


if __name__ == "__main__":
    main()
