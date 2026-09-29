#!/usr/bin/env python3
"""New-model benchmark harness (runs in .venv-eval, NOT .venv).

Protocol (same for every candidate — no paper verdicts):
  fixed SK v2b clips -> WER/CER (+chrF for MT/S2ST) + RTF + device -> matrix JSON.

Clips (local-only, never committed):
  speaker_voices/me_sk_trhove_b.m4a  (109s, ref in speaker_voices.json)
  speaker_voices/me_sk_script.m4a   (134s, ref in speaker_voices.json)

Out: processed/new_models/<engine>_matrix.json (read by the Lab like sk_direction).

Usage (on charger):
  .venv-eval/bin/pip install -r requirements-eval.txt   # once
  .venv-eval/bin/python scripts/bench_new_models.py --engine seamless --clips me_sk_trhove_b
  .venv-eval/bin/python scripts/bench_new_models.py --engine zipformer --clips me_sk_trhove_b,me_sk_script

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
    "me_sk_trhove_b": "speaker_voices/me_sk_trhove_b.m4a",
    "me_sk_script": "speaker_voices/me_sk_script.m4a",
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
    import time as _time

    import numpy as np
    import soundfile as sf
    import torch
    from seamless_communication.inference import Translator

    t0 = _time.perf_counter()
    translator = Translator("seamlessM4T_v2_large", "vocoder_36langs",
                            torch.device("cpu"), dtype=torch.float32)
    load_s = _time.perf_counter() - t0
    print(f"seamless model loaded in {load_s:.1f}s (cpu/float32)", flush=True)
    rows = []
    for clip_id in clips:
        wav = load_audio(CLIPS[clip_id])
        ref = load_sk_reference(clip_id)
        audio_s = len(wav) / 16000
        t0 = _time.perf_counter()
        # s2st: speech in -> translated SPEECH out (+ text alongside).
        # predict() needs a torch tensor, not numpy.
        text_out, speech_out = translator.predict(torch.from_numpy(wav), "s2st", "eng", "slk")
        infer_s = _time.perf_counter() - t0
        hyp_text = str(text_out[0]) if text_out else ""
        out_path = os.path.join(OUT_DIR, f"seamless_{clip_id}_en.wav")
        out_audio_s = None
        if speech_out is not None and getattr(speech_out, "audio_wavs", None):
            out_wav = np.asarray(speech_out.audio_wavs[0].detach().cpu()).flatten()
            out_sr = getattr(speech_out, "sample_rate", 16000)
            sf.write(out_path, out_wav, out_sr)
            out_audio_s = len(out_wav) / out_sr
        mt_score = None
        try:
            refs = json.load(open(os.path.join(OUT_DIR, "en_refs.json"), encoding="utf-8"))
            from sacrebleu import sentence_chrf
            mt_score = round(sentence_chrf(hyp_text, [refs[clip_id]["en"]]).score, 1)
        except Exception as e:
            print(f"chrF skipped: {e}", flush=True)
        rows.append({"clip": clip_id, "audio_s": round(audio_s, 1),
                     "infer_s": round(infer_s, 2),
                     "rtf": round(infer_s / audio_s, 3) if audio_s else None,
                     "mt_chrf_vs_opusref": mt_score,
                     "hyp_text": hyp_text[:300], "output_wav": out_path if out_audio_s else None,
                     "load_s": round(load_s, 1)})
        print(f"{clip_id}: RTF {infer_s / audio_s:.2f} "
              f"chrF {mt_score} hyp: {hyp_text[:100]}", flush=True)
    return rows


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
    ap.add_argument("--clips", default="me_sk_trhove_b")
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
