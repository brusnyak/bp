#!/usr/bin/env python3
"""Assemble a Piper train corpus dir from an omni_hq_<lang> manifest.

Reads processed/omni_hq_<lang>/manifest.json (clip wav + text), copies clips
into /tmp/voice_build/train_<lang>/ as NN.wav, writes speaker_voices.json
with matching ids/texts — the exact input shape finetune_personal_voice.py
expects (--speaker-voices-dir).

Usage: .venv/bin/python scripts/assemble_train_corpus.py --lang sk [--lang en]
"""
import argparse
import json
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["sk", "en"])
    args = ap.parse_args()

    manifest_path = os.path.join(REPO_ROOT, "processed", f"omni_hq_{args.lang}", "manifest.json")
    try:
        m = json.load(open(manifest_path, encoding="utf-8"))
    except (OSError, ValueError) as e:
        sys.exit(f"no manifest at {manifest_path}: {e}")
    clips = m.get("clips", [])
    if not clips:
        sys.exit(f"manifest {manifest_path} has no clips")

    out_dir = f"/tmp/voice_build/train_{args.lang}"
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    total_s = 0.0
    for c in clips:
        src = c["wav"]
        n = c["n"]
        dst_name = f"{n:02d}.wav"
        shutil.copy(src, os.path.join(out_dir, dst_name))
        entries.append({
            "id": f"omni_hq_{args.lang}_{n:02d}",
            "name": f"omni_hq_{args.lang}_{n:02d}",
            "language": args.lang,
            "path": dst_name,
            "transcript_source": f"omni_hq_{args.lang} manifest (OmniVoice clone of me_*, text from passage)",
            "transcribed_text": c["text"],
        })
        total_s += c.get("audio_s", 0)

    with open(os.path.join(out_dir, "speaker_voices.json"), "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"{out_dir}: {len(entries)} clips, {total_s/60:.1f}min")
    if total_s < 600:
        print(f"WARNING: {total_s/60:.1f}min below 10-min floor — trainable but expect thin quality",
              file=sys.stderr)


if __name__ == "__main__":
    main()
