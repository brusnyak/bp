#!/usr/bin/env python3
"""Offline what-if on a saved paced_replay result: re-run the listener model at other playback rates and report the TTS-to-source duration ratio.
No server, no model, no GPU: it only reads the `events` already in the JSON.

  .venv/bin/python scripts/live/replay_whatif.py processed/vox_paced.json [--rates 1.0 1.15 1.3 1.45 adaptive]

Ratio = sum of TTS audio seconds / sum of source-span seconds over pieces whose span is a real word interval. For Voxtral spans (push times, zero length) it is
reported as n/a: use the Whisper run of the same clip for the ratio."""
import argparse, json, statistics as st, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paced_replay import listener_model


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("replay"); ap.add_argument("--rates", nargs="+", default=["1.0", "1.15", "1.3", "1.45", "adaptive"])
    a = ap.parse_args(); d = json.load(open(a.replay)); ev = d["events"]
    print(f"{a.replay}: input_end {d['input_end_s']} s, strategy {d.get('strategy')}, tts {d.get('tts')}")
    for rate in a.rates:
        out = [s for s in listener_model(ev, "adaptive" if rate == "adaptive" else float(rate))
               if "lag" in s and -5 < s["lag"] < d["input_end_s"] + 30]   # drops bogus spans (a stream-clock sentinel once read 534 s)
        lag = [s["lag"] for s in out]
        if not lag: print(f"  rate {rate}: no voiced pieces"); continue
        end = max(s["listen_end"] for s in out if "listen_end" in s)
        half = len(lag) // 2
        print(f"  rate {rate:>8}: lag median {st.median(lag):.2f} s  p90 {sorted(lag)[int(.9 * (len(lag) - 1))]:.2f} s  max {max(lag):.2f} s  "
              f"growth {st.mean(lag[half:]) - st.mean(lag[:half]):+.2f} s (2nd half minus 1st)  playback ends {end - d['input_end_s']:+.1f} s vs input end")
    spans = [(s["audio_s"], s["segment_s"]) for s in listener_model(ev, 1.0) if "audio_s" in s and 0.3 < s.get("segment_s", 0) < 15 and s.get("lag", 99) > -5]
    if spans:
        print(f"  TTS/source duration ratio {sum(x for x, _ in spans) / sum(y for _, y in spans):.2f} over {len(spans)} pieces with real spans")
    else:
        print("  TTS/source duration ratio n/a (no piece with a real source span: Voxtral push-time spans)")


if __name__ == "__main__":
    main()
