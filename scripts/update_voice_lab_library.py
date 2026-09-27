#!/usr/bin/env python3
"""Build ui/voice-lab/library.json: the static manifest behind the Voice Lab page.

Scans (stdlib only — run with any python3):
  speaker_voices/*.{wav,m4a,mp3}  (+ transcript from speaker_voices.json)
  processed/voice_qc/*.wav        (+ engine/latency from candidates.json,
                                   + similarity from scores.json if present)
  test/*.wav                      (+ *_transcript.txt sidecar if present)

Writes ui/voice-lab/library.json with relative paths. The page (lab.html) reads
only this file — no backend needed. Re-run after every new recording or QC run:

    python3 scripts/update_voice_lab_library.py
"""

import argparse
import json
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_EXTS = (".wav", ".m4a", ".mp3", ".ogg", ".flac")


def audio_files(directory):
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return []
    return [n for n in names if n.lower().endswith(AUDIO_EXTS)]


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-test", action="store_true",
                    help="skip the legacy test/ clips section (test fixtures, not voice material)")
    args = ap.parse_args()
    library = {"sections": []}

    # 1. Speaker voices (reference recordings + transcripts).
    sv_dir = os.path.join(REPO_ROOT, "speaker_voices")
    meta = load_json(os.path.join(sv_dir, "speaker_voices.json")) or []
    meta_by_file = {os.path.basename(m.get("path", "")): m for m in meta if m.get("path")}
    voices = []
    for name in audio_files(sv_dir):
        m = meta_by_file.get(name, {})
        voices.append(
            {
                "name": m.get("name", os.path.splitext(name)[0]),
                "file": "../../speaker_voices/" + name,
                "language": m.get("language", "?"),
                "transcript": m.get("transcribed_text", "").strip(),
                "in_registry": name in meta_by_file,
            }
        )
    library["sections"].append(
        {"id": "voices", "title": "Speaker voices (reference recordings)", "items": voices}
    )

    # 2. QC candidates (synth engine outputs + measurements).
    qc_dir = os.path.join(REPO_ROOT, "processed", "voice_qc")
    manifest = load_json(os.path.join(qc_dir, "candidates.json")) or {}
    scores = {
        r.get("label"): r.get("cosine_similarity")
        for r in (load_json(os.path.join(qc_dir, "scores.json")) or {}).get("results", [])
    }
    by_label = {c.get("label"): c for c in manifest.get("candidates", [])}
    qc_items = []
    for name in audio_files(qc_dir):
        label = os.path.splitext(name)[0]
        c = by_label.get(label, {})
        qc_items.append(
            {
                "name": label,
                "file": "../../processed/voice_qc/" + name,
                "engine": c.get("engine", "?"),
                "language": c.get("language", "?"),
                "sample_rate": c.get("sample_rate"),
                "synthesis_latency_s": c.get("synthesis_latency_s"),
                "similarity": scores.get(label),
            }
        )
    library["sections"].append(
        {"id": "qc", "title": "QC candidates (engine outputs)", "items": qc_items}
    )

    # 3. Test clips (pipeline inputs + ground-truth transcripts).
    # Skipped with --no-test: test/*.wav are old unit-test fixtures, not voice material.
    if not args.no_test:
        test_dir = os.path.join(REPO_ROOT, "test")
        test_items = []
        for name in audio_files(test_dir):
            stem = os.path.splitext(name)[0]
            transcript = ""
            for cand in (stem + "_transcript.txt", stem + " transcript.txt"):
                p = os.path.join(test_dir, cand)
                if os.path.exists(p):
                    with open(p) as f:
                        transcript = f.read().strip()
                    break
            test_items.append(
                {"name": stem, "file": "../../test/" + name, "transcript": transcript}
            )
        library["sections"].append(
            {"id": "test", "title": "Test clips (pipeline inputs)", "items": test_items}
        )

    # 4. Conversation sim (end-to-end sentence outputs + per-stage latencies).
    conv_dir = os.path.join(REPO_ROOT, "processed", "conversation")
    sim = load_json(os.path.join(REPO_ROOT, "processed", "conversation_sim.json")) or {}
    sim_by_key = {(s.get("dir"), s.get("n")): s for s in sim.get("sections", []) or sim.get("sentences", [])}
    conv_items = []
    for name in audio_files(conv_dir):
        stem = os.path.splitext(name)[0]  # e.g. en_sk_00
        m = re.match(r"(.+)_(\d+)$", stem)
        meta = dict(sim_by_key.get((m.group(1), int(m.group(2))), {})) if m else {}
        meta.pop("dir", None)
        meta.pop("n", None)
        conv_items.append({"name": stem, "file": "../../processed/conversation/" + name,
                           "meta": meta})
    library["sections"].append(
        {"id": "conversation", "title": "Conversation sim (live-pipeline sentence outputs)", "items": conv_items}
    )

    out = os.path.join(REPO_ROOT, "ui", "voice-lab", "library.json")
    with open(out, "w") as f:
        json.dump(library, f, indent=2, ensure_ascii=False)
    counts = {s["id"]: len(s["items"]) for s in library["sections"]}
    print(f"Wrote {out}: {counts}")


if __name__ == "__main__":
    main()
