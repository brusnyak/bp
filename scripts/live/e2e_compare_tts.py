#!/usr/bin/env python3
"""Compare TTS engines end-to-end on the same audio, with baselines.

Runs the E2E streaming test with multiple TTS engines and compares results
against Piper (incumbent, RTF ~0.03-0.05) and OmniVoice (batch/QC, RTF 0.40-0.65).

Baselines (from documentation/model_landscape_2026-10.md and VERIFICATION_RESULTS_2026-11-06.md):
  Piper:    RTF 0.03-0.05 on CPU, 60 MB ONNX per voice
  OmniVoice: RTF 0.40-0.65 on T4, batch/QC only (not streaming)

Usage:
  .venv/bin/python scripts/live/e2e_compare_tts.py \
      --audio processed/testaudio/en_paced.wav --src en --tgt sk \
      --tts piper,omnivoice,xtts,xvoice,pocket \
      --clone --ref speaker_voices/me_sk_trhove.m4a \
      --out processed/e2e_tts_compare.json
"""
from __future__ import annotations

import json, subprocess, sys, time
from pathlib import Path

TTS_ENGINES = ["piper", "omnivoice", "xtts", "xvoice", "pocket"]

# Baselines from measured data
BASELINES = {
    "piper": {"rtf": 0.04, "note": "CPU, 60 MB ONNX per voice, live path"},
    "omnivoice": {"rtf": 0.53, "note": "T4, batch/QC only, not streaming"},
}


def run_single(audio, src, tgt, tts, clone, ref, base, out, chunk=0.5):
    """Run one E2E streaming test and return results."""
    out_path = Path(out) / f"e2e_{tts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "scripts/live/e2e_streaming_test.py",
        "--audio", audio, "--src", src, "--tgt", tgt,
        "--tts", tts, "--base", base, "--chunk", str(chunk),
        "--out", str(out_path),
    ]
    if clone:
        cmd += ["--clone", "--ref", ref]

    print(f"\n{'='*60}")
    print(f"Running: TTS={tts}, clone={clone}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"FAILED: {result.stderr[-500:] if result.stderr else 'unknown'}")
        return None

    # Print last few lines of output
    lines = result.stdout.strip().split("\n")
    for line in lines[-5:]:
        print(f"  {line}")

    data = json.loads(out_path.read_text())
    return data["summary"]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Compare TTS engines E2E")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--tgt", required=True)
    ap.add_argument("--tts", default="piper,omnivoice,xtts,xvoice,pocket",
                    help="Comma-separated TTS engines")
    ap.add_argument("--clone", action="store_true", help="Enable voice cloning")
    ap.add_argument("--ref", help="Reference WAV for cloning")
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--out", default="processed/e2e_compare")
    args = ap.parse_args()

    engines = [e.strip() for e in args.tts.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"config": {
        "audio": args.audio, "src": args.src, "tgt": args.tgt,
        "clone": args.clone, "ref": args.ref,
    }, "baselines": BASELINES, "engines": {}}

    for tts in engines:
        summary = run_single(args.audio, args.src, args.tgt, tts,
                             args.clone, args.ref, args.base, out_dir)
        if summary:
            results["engines"][tts] = summary
            # Compare against Piper baseline
            if tts != "piper" and "piper" in results["engines"]:
                piper_tts = results["engines"]["piper"].get("tts_s_median")
                if piper_tts and summary.get("tts_s_median"):
                    ratio = summary["tts_s_median"] / piper_tts
                    summary["vs_piper_tts_ratio"] = round(ratio, 2)
                    summary["vs_piper_baseline_rtf"] = BASELINES["piper"]["rtf"]

    out_path = out_dir / "e2e_compare.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    for tts, s in results["engines"].items():
        print(f"\n{tts}:")
        for k, v in s.items():
            print(f"  {k}: {v}")

    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
