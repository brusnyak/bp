#!/usr/bin/env python3
"""bp — single entry for every repeatable workflow in this repo.

Replaces the scripts/ sprawl: one CLI, one --help, same behavior (each command
reuses the original module's code path, nothing rewritten).

    venv/bin/python scripts/bp.py corpus      # normalize/dedupe/transcribe voices
    venv/bin/python scripts/bp.py qc --synthesize-only
    venv/bin/python scripts/bp.py stt --models base,small --clip sk
    venv/bin/python scripts/bp.py e2e         # full EN->SK timed run, new voice
    venv/bin/python scripts/bp.py library     # refresh Voice Lab manifest
    venv/bin/python scripts/bp.py script      # bilingual reading sheet via local MT

Trainers stay separate (own venv by design): scripts/finetune_personal_voice.py.
"""

import argparse
import os
import runpy
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

WRAPPED = {
    "corpus": "prepare_voice_corpus.py",
    "qc": "voice_similarity_qc.py",
    "e2e": "e2e_ensk_new_voice.py",
    "library": "update_voice_lab_library.py",
    "script": "make_reading_script.py",
}


def cmd_stt(args):
    """STT model × language matrix on the owner's real recordings.

    The question it answers: which STT rung for which language, with numbers.
    Ground truth: EN = script sentences; SK = proofread SK column.
    --lang-prompt: force decoder language (sk | cs | auto) — cs tests the
    Czech-as-Slovak-proxy hypothesis with zero downloads.
    """
    import time
    import unicodedata

    import librosa
    import numpy as np

    from backend.stt.faster_whisper_stt import FasterWhisperSTT
    from make_reading_script import EN_SENTENCES

    try:
        from jiwer import wer
    except ImportError:
        sys.exit("jiwer not installed in this venv")

    def norm(s):
        return " ".join(s.lower().strip().split())

    def plain(s):
        return norm("".join(
            c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)))

    def load(name):
        wav, sr = librosa.load(os.path.join(REPO_ROOT, "speaker_voices", name),
                               sr=16000, mono=True)
        return np.asarray(wav, dtype=np.float32), sr

    def hyp_text(segments):
        return norm(" ".join(
            s["text"] if isinstance(s, dict) else getattr(s, "text", "") for s in segments))

    sk_refs = []
    md = os.path.join(REPO_ROOT, "documentation", "reading_script_bilingual.md")
    for line in open(md).read().splitlines():
        if line.startswith("|") and len(line) > 2 and line[2].isdigit():
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) == 3:
                sk_refs.append(parts[2])

    clips = {
        "en": ("en_script_reading.m4a", norm(" ".join(EN_SENTENCES))),
        "sk": ("sk_script_reading.m4a", norm(" ".join(sk_refs))),
    }
    if args.clip != "both":
        clips = {args.clip: clips[args.clip]}

    import json
    results = []
    prompt_list = [p.strip() for p in args.lang_prompt.split(",")]
    for model in args.models.split(","):
        stt = FasterWhisperSTT(model_size=model.strip())
        for clip_key, (fname, ref) in clips.items():
            # --lang-prompt applies to the SK clip (the open question); EN always en.
            prompts = ["en"] if clip_key == "en" else prompt_list
            for prompt in prompts:
                wav, sr = load(fname)
                segs, t, dl = stt.transcribe_audio(
                    wav, sr, language=None if prompt == "auto" else prompt)
                hyp = hyp_text(segs)
                row = {
                    "model": model.strip(), "clip": fname, "prompt": prompt,
                    "wer": round(wer(ref, hyp), 4),
                    "wer_plain": round(wer(plain(ref), plain(hyp)), 4),
                    "stt_time_s": round(t, 1), "audio_s": round(len(wav) / sr, 1),
                    "detected_lang": dl,
                }
                results.append(row)
                print(f"{row['model']:6s} {fname:24s} prompt={prompt:4s} "
                      f"WER {row['wer']:.4f} (plain {row['wer_plain']:.4f})  {t:.1f}s")
    out = os.path.join(REPO_ROOT, "processed", "stt_matrix.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out} ({len(results)} rows)")


def main():
    parser = argparse.ArgumentParser(prog="bp", description="Single CLI for repo workflows.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in WRAPPED:
        p = sub.add_parser(name, help=f"passthrough to scripts/{WRAPPED[name]}")
        p.add_argument("rest", nargs=argparse.REMAINDER, help="args forwarded to the script")

    p = sub.add_parser("stt", help="STT model x language matrix on real recordings")
    p.add_argument("--models", default="base,small", help="comma list, e.g. base,small,medium")
    p.add_argument("--clip", default="both", choices=["en", "sk", "both"])
    p.add_argument("--lang-prompt", default="sk",
                   help="comma list of decoder prompts: sk,cs,auto (cs = Czech-proxy test)")

    args = parser.parse_args()
    if args.cmd == "stt":
        cmd_stt(args)
    else:
        sys.argv = [WRAPPED[args.cmd]] + (args.rest[1:] if args.rest[:1] == ["--"] else args.rest)
        runpy.run_path(os.path.join(REPO_ROOT, "scripts", WRAPPED[args.cmd]),
                       run_name="__main__")


if __name__ == "__main__":
    main()
