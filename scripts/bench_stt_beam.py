#!/usr/bin/env python3
"""STT beam sweep: greedy vs wide beam on Slovak speech (small-sk live rung).

Question: does beam_size=1 (greedy) cut STT time vs beam 5 at acceptable WER?
Runs each beam over the existing SK recordings, scoring WER/CER against the
proofread references in speaker_voices/speaker_voices.json.

Outputs (under processed/stt_beam/):
  beam_matrix.json          — per-clip WER/CER/STT-seconds/RTF per beam
  sk_beam<N>_<clip>.txt    — raw transcripts (auditable, diffable)

Usage:
  .venv/bin/python scripts/bench_stt_beam.py [--beams 1,2,5] [--models small-sk]

One model instance per model_size (load once, ~20 s); beam is an instance
attribute read at transcribe time, so switching beams costs no reload.
"""
import argparse
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

OUT_DIR = os.path.join(REPO_ROOT, "processed", "stt_beam")
OUT_JSON = os.path.join(OUT_DIR, "beam_matrix.json")

CLIPS = [
    ("me_sk_script", "speaker_voices/me_sk_script.m4a"),
    ("me_sk_trhove", "speaker_voices/me_sk_trhove.m4a"),
    ("me_sk_trhove_b", "speaker_voices/me_sk_trhove_b.m4a"),
]
BY_FILE = {c[0]: c[1] for c in CLIPS}


def normalize(text):
    text = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def load_reference(clip_id):
    meta = json.load(open(os.path.join(REPO_ROOT, "speaker_voices", "speaker_voices.json"), encoding="utf-8"))
    want = BY_FILE[clip_id]
    for e in meta:
        p = e.get("path", "") or e.get("filename", "") or ""
        if os.path.basename(p) == os.path.basename(want):
            ref = (e.get("transcribed_text") or "").strip()
            if ref:
                return ref
    raise KeyError(f"no voices.json reference for {clip_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beams", default="1,2,5")
    ap.add_argument("--models", default="small-sk",
                    help="comma-separated model sizes (default: live SK rung only)")
    ap.add_argument("--chunk-s", type=float, default=0,
                    help="split audio into non-overlapping chunks of N seconds "
                         "(0 = whole clip); emulates live VAD segments")
    ap.add_argument("--prompt", default=None,
                    help="initial_prompt passed to the recognizer (e.g. Slovak "
                         "sentence with diacritics to anchor greedy decoding)")
    args = ap.parse_args()
    beams = [int(b) for b in args.beams.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    os.makedirs(OUT_DIR, exist_ok=True)

    import librosa
    import numpy as np
    import jiwer
    from backend.stt.faster_whisper_stt import FasterWhisperSTT

    audio, refs = {}, {}
    for clip_id, rel in CLIPS:
        wav, _ = librosa.load(os.path.join(REPO_ROOT, rel), sr=16000, mono=True)
        audio[clip_id] = np.asarray(wav, dtype=np.float32)
        refs[clip_id] = normalize(load_reference(clip_id))
        print(f"loaded {clip_id}: {len(audio[clip_id]) / 16000:.0f}s audio, "
              f"{len(refs[clip_id])} ref chars", flush=True)

    matrix = {"models": models, "beams": beams, "clips": {}}
    for model_size in models:
        stt = FasterWhisperSTT(model_size=model_size, compute_type="int8")
        for clip_id, wav in audio.items():
            chunks = [wav] if not args.chunk_s else [
                wav[i:i + int(args.chunk_s * 16000)]
                for i in range(0, len(wav), int(args.chunk_s * 16000))]
            entry = matrix["clips"].setdefault(
                clip_id, {"audio_s": round(len(wav) / 16000, 1), "runs": {}})
            for beam in beams:
                stt.beam_size = beam
                hyps, total_t = [], 0.0
                for ch in chunks:
                    kw = {"language": "sk"}
                    if args.prompt:
                        kw["initial_prompt"] = args.prompt
                    segs, t, lang = stt.transcribe_audio(ch, 16000, **kw)
                    hyps.append(" ".join(s.text if hasattr(s, "text") else s["text"] for s in segs))
                    total_t += t
                hyp = normalize(" ".join(hyps))
                key = f"{model_size}_beam{beam}"
                if args.prompt:
                    key += "_prompt"
                entry["runs"][key] = {
                    "wer": round(jiwer.wer(refs[clip_id], hyp), 4),
                    "cer": round(jiwer.cer(refs[clip_id], hyp), 4),
                    "stt_s": round(total_t, 2),
                    "rtf": round(total_t / (len(wav) / 16000), 3),
                    "lang": lang,
                    "chunks": len(chunks),
                }
                tag = f"sk_{key}_{clip_id}"
                if args.chunk_s:
                    tag += f"_c{int(args.chunk_s)}s"
                with open(os.path.join(OUT_DIR, f"{tag}.txt"), "w", encoding="utf-8") as f:
                    f.write(hyp + "\n")
                r = entry["runs"][key]
                print(f"{clip_id:16s} {key:18s} WER {r['wer']:.3f} CER {r['cer']:.3f} "
                      f"STT {r['stt_s']:6.1f}s RTF {r['rtf']:.2f}", flush=True)

    out_json = OUT_JSON
    if args.chunk_s or args.prompt:
        suffix = ""
        if args.chunk_s:
            suffix += f"_c{int(args.chunk_s)}s"
        if args.prompt:
            suffix += "_prompt"
        out_json = os.path.join(OUT_DIR, f"beam_matrix{suffix}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2, ensure_ascii=False)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
