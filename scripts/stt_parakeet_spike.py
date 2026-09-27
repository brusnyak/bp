#!/usr/bin/env python3
"""Spike: NVIDIA Parakeet-TDT-0.6B-v3 on the SK script reading.

Pass/fail vs faster-whisper small (plain-WER 0.49 on this clip).
Run: .venv-stt/bin/python scripts/stt_parakeet_spike.py [--clip sk|en]
"""

import json
import re
import string
import sys
import time

import librosa
import torch

from transformers import ParakeetForTDT, ParakeetProcessor

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
WIN_S = 60


def plain(s: str) -> str:
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation + "„“”"))
    return re.sub(r"\s+", " ", s).strip()


def wer(hyp: str, ref: str) -> float:
    h, r = hyp.split(), ref.split()
    prev = list(range(len(r) + 1))
    for i, hw in enumerate(h, 1):
        cur = [i]
        for j, rw in enumerate(r, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (hw != rw)))
        prev = cur
    return prev[-1] / max(len(r), 1)


def main() -> None:
    clip = sys.argv[sys.argv.index("--clip") + 1] if "--clip" in sys.argv else "sk"
    as_json = "--json" in sys.argv
    max_windows = (
        int(sys.argv[sys.argv.index("--max-windows") + 1]) if "--max-windows" in sys.argv else None
    )
    meta = {m["language"]: m for m in json.load(open("speaker_voices/speaker_voices.json"))}
    m = meta[clip]
    wav, _ = librosa.load(m["path"], sr=16000, mono=True)
    ref = plain(m["transcribed_text"])

    t0 = time.perf_counter()
    proc = ParakeetProcessor.from_pretrained(MODEL_ID)
    model = ParakeetForTDT.from_pretrained(MODEL_ID).eval()
    load_t = time.perf_counter() - t0
    print(f"model load: {load_t:.1f}s", flush=True)

    hyps, dec_t = [], 0.0
    windows = list(range(0, len(wav), WIN_S * 16000))
    if max_windows is not None:
        windows = windows[:max_windows]
    for i in windows:
        seg = wav[i : i + WIN_S * 16000]
        inputs = proc(seg, sampling_rate=16000, return_tensors="pt")
        t1 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs)
        dec_t += time.perf_counter() - t1
        hyps.append(proc.batch_decode(out.sequences, skip_special_tokens=True)[0])
        print(f"window {i // 16000}s done ({dec_t:.1f}s decode so far)", flush=True)

    hyp = plain(" ".join(hyps))
    audio_s = sum(len(wav[i : i + WIN_S * 16000]) for i in windows) / 16000
    if as_json:
        print(json.dumps({"text": hyp, "decode_s": round(dec_t, 2), "audio_s": round(audio_s, 1)}))
        return
    print(f"windows: {len(hyps)}, audio {audio_s:.1f}s, decode {dec_t:.1f}s (RTF {dec_t / audio_s:.3f})")
    print(f"plain-WER: {wer(hyp, ref):.4f}")
    print("HYP:", hyp[:300])


if __name__ == "__main__":
    main()
