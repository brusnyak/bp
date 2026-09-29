#!/usr/bin/env python3
"""SeamlessM4T S2ST matrix, RAM-safe: 30s chunks, one model load (.venv-eval).

Full 109s+ clips in one shot OOM this 16GB box (attention is O(n^2) + swap death);
30s chunks hold RAM flat. Extends processed/new_models/seamless_matrix.json.

Run (on charger): .venv-eval/bin/python scripts/seamless_matrix.py --clip sk_script_reading
"""
from __future__ import annotations

import argparse
import json
import os
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "processed", "new_models")
CLIPS = {
    "sk_trhove_rano_v2b": "speaker_voices/sk_trhove_rano_v2b.m4a",
    "sk_script_reading": "speaker_voices/sk_script_reading.m4a",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, choices=sorted(CLIPS))
    ap.add_argument("--chunk-s", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    import numpy as np
    import librosa
    import soundfile as sf
    import torch
    from seamless_communication.inference import Translator
    from sacrebleu import sentence_chrf

    wav, _ = librosa.load(os.path.join(REPO_ROOT, CLIPS[args.clip]), sr=16000, mono=True)
    wav = np.asarray(wav, dtype=np.float32)
    refs = json.load(open(os.path.join(OUT_DIR, "en_refs.json"), encoding="utf-8"))[args.clip]["en"]

    t0 = time.perf_counter()
    tr = Translator("seamlessM4T_v2_large", "vocoder_36langs",
                    torch.device("cpu"), dtype=torch.float32)
    load_s = round(time.perf_counter() - t0, 1)
    print(f"load_s: {load_s}", flush=True)

    hyps, outs, sr, infer_total = [], [], 16000, 0.0
    step = args.chunk_s * 16000
    for k in range(0, len(wav), step):
        ch = torch.from_numpy(wav[k:k + step])
        t0 = time.perf_counter()
        text_out, speech_out = tr.predict(ch, "s2st", "eng", "slk")
        infer_total += time.perf_counter() - t0
        hyps.append(str(text_out[0]) if text_out else "")
        if speech_out is not None and getattr(speech_out, "audio_wavs", None):
            w = np.asarray(speech_out.audio_wavs[0].detach().cpu()).flatten()
            sr = getattr(speech_out, "sample_rate", 16000)
            outs.append(w)
        print(f"chunk {k // step}: hyp: {hyps[-1][:80]}", flush=True)
    full_hyp = " ".join(hyps)
    out_path = os.path.join(OUT_DIR, f"seamless_{args.clip}_en.wav")
    full_wav = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)
    sf.write(out_path, full_wav, sr)
    audio_s = len(wav) / 16000
    row = {"clip": args.clip, "audio_s": round(audio_s, 1),
           "infer_s": round(infer_total, 1), "rtf": round(infer_total / audio_s, 3),
           "mt_chrf_vs_opusref": round(sentence_chrf(full_hyp, [refs]).score, 1),
           "hyp_text": full_hyp,
           "output_wav": os.path.relpath(out_path, REPO_ROOT), "load_s": load_s}
    mat_path = os.path.join(OUT_DIR, "seamless_matrix.json")
    try:
        mat = json.load(open(mat_path, encoding="utf-8"))
    except (OSError, ValueError):
        mat = {"engine": "seamless-chunked30", "results": []}
    mat["results"] = [r for r in mat.get("results", []) if r.get("clip") != args.clip] + [row]
    json.dump(mat, open(mat_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("MATRIX:", json.dumps({k: v for k, v in row.items() if k != "hyp_text"}), flush=True)


if __name__ == "__main__":
    main()
