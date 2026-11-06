#!/usr/bin/env python3
"""Summarise paced_replay JSON files with one comparable metric.
word-weighted mid-span delay = listen_start - midpoint of the source words' span, weighted by word count,
under a one-at-a-time player at rate 1.0 and the adaptive rate. Baseline (non-incremental) spans use the
previous segment end as start, which overstates their delay slightly; incremental spans are exact.
usage: summarize_replay.py processed/paced_a.json processed/paced_b.json"""
import importlib.util, json, sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("p", Path(__file__).with_name("paced_replay.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for f in sys.argv[1:]:
    r = json.load(open(f))
    audio = sum(e["seconds"] for e in r["events"] if e["type"] == "tts_audio")
    out = [Path(f).stem, "audio %.0fs/%.0fs" % (audio, r["seconds"])]
    for rate in (1.0, "adaptive"):
        allsegs = m.listener_model(r["events"], rate)
        segs = [x for x in allsegs if "listen_start" in x]
        w = [max(1, len(x["text"].split())) for x in segs]
        mid = [x["listen_start"] - (x["start"] + x["source_end"]) / 2 for x in segs]
        out.append("%s: mean %.1fs worst %.1fs (%d/%d emitted segments voiced)" % (rate, sum(a * b for a, b in zip(mid, w)) / sum(w), max(mid), len(segs), len(allsegs)))
    print(" | ".join(out))
