#!/usr/bin/env python3
"""Offline SK-direction evaluation: which STT rung feeds the SK→EN translator best?

Runs each recognizer over the existing SK recordings (script reading + market-morning
takes, including the v2b variants), then feeds every transcript through the same
Opus-MT sk-en model used live, scoring STT with WER/CER and MT with chrF (sacrebleu)
against the proofread reference column in documentation/reading_script_bilingual.md
(for the v2/v2b market-morning takes, references come from speaker_voices.json).

Rungs (configurable, all local, no network):
  small-sk   — ct2_models/whisper-small-sk  (current live default, Slovak-tuned)
  base       — ct2_models/whisper-base
  turbo      — downloads large-v3-turbo into the HF cache on first use (run with
               --include-turbo; model card itself is the cache, no repo weight)

Outputs (all under processed/sk_direction/):
  sk_direction_matrix.json     — per-clip WER/CER per rung + MT chrF + STT seconds
  sk_<rung>_<clip>.txt         — raw transcripts (auditable, diffable)

Usage:
  .venv/bin/python scripts/eval_sk_direction.py [--include-turbo] [--clips a,b]

Reads nothing from the network except the optional first-time turbo download.
Defaults mirror the live pipeline (int8 compute, language="sk", beam 5).
"""

import argparse
import json
import os
import re
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

OUT_DIR = os.path.join(REPO_ROOT, "processed", "sk_direction")
OUT_JSON = os.path.join(OUT_DIR, "sk_direction_matrix.json")

CLIPS = [
    ("me_sk_script", "speaker_voices/me_sk_script.m4a"),
    ("me_sk_trhove", "speaker_voices/me_sk_trhove.m4a"),
    ("me_sk_trhove_b", "speaker_voices/me_sk_trhove_b.m4a"),
]

BY_FILE = {c[0]: c[1] for c in CLIPS}

RUNGS = {
    "small-sk": {"model_size": "small-sk"},
    "base": {"model_size": "base"},
    "turbo": {"model_size": "large-v3-turbo"},
}


def normalize(text):
    text = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text):
    parts = re.split(r"(?<=[.?!])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def translate_sentences(mt, text, src="sk", tgt="en"):
    sentences = split_sentences(text)
    trans = []
    for s in sentences:
        t, _ = mt.translate(s, src, tgt)
        trans.append(t)
    return " ".join(trans)


def load_sk_reference(clip_id):
    meta = json.load(open(os.path.join(REPO_ROOT, "speaker_voices", "speaker_voices.json"), encoding="utf-8"))
    want = BY_FILE[clip_id]
    for e in meta:
        p = e.get("path", "") or e.get("filename", "") or ""
        if os.path.basename(p) == os.path.basename(want):
            return (e.get("transcribed_text") or "").strip()
    raise KeyError(f"no voices.json reference for {clip_id} (want {want})")


def load_en_reference(clip_id, mt, sk_ref):
    """Return ground-truth English reference if available, or clean MT-translated reference."""
    if clip_id == "me_sk_script":
        meta = json.load(open(os.path.join(REPO_ROOT, "speaker_voices", "speaker_voices.json"), encoding="utf-8"))
        for e in meta:
            p = e.get("path", "") or e.get("filename", "") or ""
            if "en_script_reading" in p:
                return (e.get("transcribed_text") or "").strip()
    # For trhove_rano, use MT translation of the clean proofread Slovak transcript as reference
    return translate_sentences(mt, sk_ref, src="sk", tgt="en")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-turbo", action="store_true", help="also run large-v3-turbo (HF download on first use)")
    ap.add_argument("--clips", default=None, help="comma-separated clip ids to run (default: all)")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    import jiwer
    import librosa
    import numpy as np
    from sacrebleu import sentence_chrf
    from backend.stt.faster_whisper_stt import FasterWhisperSTT
    from backend.mt.ctranslate2_mt import CTranslate2MT

    wanted = set(args.clips.split(",")) if args.clips else None
    rungs = dict(RUNGS)
    if not args.include_turbo:
        rungs.pop("turbo")

    mt = CTranslate2MT("Helsinki-NLP/opus-mt-sk-en")

    audio = {}
    sk_refs = {}
    en_refs = {}
    for clip_id, rel in CLIPS:
        if wanted and clip_id not in wanted:
            continue
        wav, _sr = librosa.load(os.path.join(REPO_ROOT, rel), sr=16000, mono=True)
        sk_ref = load_sk_reference(clip_id)
        en_ref = load_en_reference(clip_id, mt, sk_ref)
        audio[clip_id] = np.asarray(wav, dtype=np.float32)
        sk_refs[clip_id] = sk_ref
        en_refs[clip_id] = en_ref

    matrix = {"rungs": list(rungs), "clips": {}}
    for clip_id, wav in audio.items():
        sk_ref = sk_refs[clip_id]
        en_ref = en_refs[clip_id]
        sk_ref_n = normalize(sk_ref)
        entry = {
            "audio_s": round(len(wav) / 16000, 1),
            "ref_chars": len(sk_ref),
            "en_ref_chars": len(en_ref),
            "rungs": {},
        }
        for rung, kw in rungs.items():
            stt = FasterWhisperSTT(**kw)
            segs, stt_t, lang = stt.transcribe_audio(wav, 16000, language="sk")
            hyp = " ".join(s.text if hasattr(s, "text") else s["text"] for s in segs)
            hyp_n = normalize(hyp)
            t0 = time.perf_counter()
            translated = translate_sentences(mt, hyp, "sk", "en")
            mt_t = time.perf_counter() - t0
            entry["rungs"][rung] = {
                "wer": round(jiwer.wer(sk_ref_n, hyp_n), 4),
                "cer": round(jiwer.cer(sk_ref_n, hyp_n), 4),
                "chrf": round(sentence_chrf(translated, [en_ref]).score, 1),
                "stt_s": round(stt_t, 2),
                "mt_s": round(mt_t, 2),
                "rtf": round(stt_t / (len(wav) / 16000), 3),
                "lang": lang,
            }
            with open(os.path.join(OUT_DIR, f"sk_{rung}_{clip_id}.txt"), "w", encoding="utf-8") as f:
                f.write(hyp + "\n")
            with open(os.path.join(OUT_DIR, f"en_{rung}_{clip_id}.txt"), "w", encoding="utf-8") as f:
                f.write(translated + "\n")
            r = entry["rungs"][rung]
            print(f"{clip_id:22s} {rung:9s} WER {r['wer']:.3f} CER {r['cer']:.3f} "
                  f"chrF {r['chrf']:4.1f} STT {r['stt_s']:6.1f}s RTF {r['rtf']:.2f}", flush=True)
        matrix["clips"][clip_id] = entry

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2, ensure_ascii=False)
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()