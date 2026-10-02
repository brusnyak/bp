#!/usr/bin/env python3
"""Full streaming-pipeline latency audit: can the live path do near-RT?

Streams one clip through /ws (needs `make run`) and reconstructs EVERY stage
from real events: VAD-close (parsed from the server log), transcript/translation
times (probe dump), first TTS byte, playback duration. One JSON, no simulation.

Out: processed/stream_audit/<name>_audit.json
Run: .venv/bin/python scripts/streaming_audit.py --clip <wav> --source en --target sk
     --server-log /tmp/bp_run6.log   (pass the live server log for VAD times)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "processed", "stream_audit")


def parse_vad_closes(server_log: str) -> list[float]:
    """Wall-clock times of 'Final speech segment ENDED' lines (epoch seconds)."""
    closes = []
    line_re = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*Final speech segment ENDED")
    try:
        import datetime
        with open(server_log, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = line_re.match(line)
                if m:
                    dt = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                    closes.append(dt.timestamp())
    except OSError:
        pass
    return closes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--tts", default=None)
    ap.add_argument("--server-log", default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    name = args.name or os.path.splitext(os.path.basename(args.clip))[0]

    dump = os.path.join(OUT_DIR, name + "_dump.json")
    tts_raw = os.path.join(OUT_DIR, name + "_tts.raw.wav")
    cmd = [os.path.join(REPO_ROOT, ".venv", "bin", "python"),
           os.path.join(REPO_ROOT, "scripts", "live_direction_probe.py"),
           "--source", args.source, "--target", args.target, "--clip", args.clip,
           "--settle", "14", "--trail", "3.0", "--dump", dump, "--tts-out", tts_raw]
    if args.tts:
        cmd += ["--tts", args.tts]
    print(f"streaming {args.clip} {args.source}->{args.target} …", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    print((r.stdout or "")[-800:], flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"probe failed:\n{(r.stderr or '')[-1500:]}")

    d = json.load(open(dump, encoding="utf-8"))
    n_tr = len(d["transcripts"])
    audit = {
        "clip": args.clip, "direction": f"{args.source}->{args.target}",
        "audio_s": d["audio_s"], "utterances": n_tr,
        "partials": len(d.get("partials", [])),
        "first_transcript_at": d["transcripts"][0]["at"] if n_tr else None,
        "first_translation_at": d["translations"][0]["at"] if d["translations"] else None,
        "first_tts_at": d.get("first_tts_at"),
        "vad_closes_wall": parse_vad_closes(args.server_log) if args.server_log else [],
        "tts_chunks": d.get("tts_chunks", 0),
    }
    # Stage deltas (the near-RT question in four numbers).
    fa, fb = audit["first_transcript_at"], audit["first_translation_at"]
    fc = audit["first_tts_at"]
    audit["vad_to_text_s"] = round(fa, 3) if fa is not None else None
    audit["text_to_translation_s"] = round(fb - fa, 3) if fa is not None and fb is not None else None
    audit["translation_to_audio_s"] = round(fc - fb, 3) if fb is not None and fc is not None else None
    out = os.path.join(OUT_DIR, name + "_audit.json")
    json.dump(audit, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("AUDIT:", json.dumps({k: v for k, v in audit.items()
                                if k not in ("vad_closes_wall",)}), flush=True)
    print(f"wrote {os.path.relpath(out, REPO_ROOT)}")


if __name__ == "__main__":
    main()
