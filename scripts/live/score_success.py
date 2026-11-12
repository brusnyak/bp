#!/usr/bin/env python3
"""Score one paced_replay run against the near-real-time success test (documentation/NEAR_REALTIME_GOAL.md). Recognition quality (criterion 4) comes from strategy_lab.py.

  PYTHONPATH=. .venv/bin/python scripts/live/score_success.py replay.json processed/testaudio/en_paced.json [--rate adaptive]

1 lag behind the speech position: median <= 2.0 s and p90 <= 4.5 s (voice start minus the end of the speech it translates)
2 early start: of sentences with 8+ words, at least half have their first voice start before the sentence ends
3 no drift: lag growth <= 0.5 s per minute of speech
4 recognition error <= 8% on the whole run (transcripts the route produced against the script)
"""
import argparse, json, re, sys


def drift_s_per_run(xs, ys):
    """Theil-Sen slope of lag against source time, in seconds of lag per second of speech (outlier-proof), times 60 = per minute via the caller's `minutes`."""
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None and -5 < y < 120]
    sl = sorted((y2 - y1) / (x2 - x1) for i, (x1, y1) in enumerate(pts) for x2, y2 in pts[i + 1:] if x2 - x1 > 1.0)
    return sl[len(sl) // 2] * 60.0 if sl else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("replay"); ap.add_argument("testaudio"); ap.add_argument("--rate", default="adaptive")
    a = ap.parse_args(); r = json.load(open(a.replay)); ta = json.load(open(a.testaudio)); x = r["rates"][a.rate]
    lag_med, lag_p90 = x["lag_median_s"], x["lag_p90_s"]
    minutes = max(r["input_end_s"], 1.0) / 60.0
    growth = drift_s_per_run(x["source_end_s"], x["lag_s"]) * minutes   # robust slope, not last-minus-first (one bad span gave -117132 s in the saved en->sk min07 run)
    ends, starts = x["source_end_s"], x["listen_start_s"]; sent = [s for s in ta.get("sentences", []) if len(s["text"].split()) >= 8]
    def belongs(end, s): return s["start"] - 0.3 <= end <= s["end"] + 0.6          # the speech this voice translates lies inside that sentence
    early = [any(belongs(e, s) and st < s["end"] for e, st in zip(ends, starts)) for s in sent]   # its voice starts before the sentence is finished
    early_frac = sum(early) / len(early) if early else None
    norm = lambda t: re.findall(r"[^\W_]+", t.lower())
    hyp = norm(" ".join(e["text"] for e in r.get("events", []) if e.get("type") == "transcription_result")); ref = norm(ta["text"])
    d = list(range(len(hyp) + 1))                                                       # word error of what the recogniser produced against the script
    for i, w in enumerate(ref, 1):
        prev = d[:]; d[0] = i
        for j, h in enumerate(hyp, 1): d[j] = min(prev[j] + 1, d[j-1] + 1, prev[j-1] + (w != h))
    err = round(100.0 * d[-1] / max(1, len(ref)), 1)
    rows = [("1 median lag <= 2.0 s", lag_med, lag_med is not None and lag_med <= 2.0), ("1 p90 lag <= 4.5 s", lag_p90, lag_p90 is not None and lag_p90 <= 4.5),
            ("2 early start >= 50% of 8+ word sentences", None if early_frac is None else round(early_frac, 2), early_frac is not None and early_frac >= 0.5),
            ("3 lag growth <= 0.5 s/min", round(growth / minutes, 2), growth / minutes <= 0.5),
            ("4 recognition error <= 8% (percent of reference words)", err, err <= 8.0)]
    print(f"strategy={r.get('strategy')} audio={r['audio']} rate={a.rate} segments={x['segments']} with_audio={x['with_audio']}")
    for name, val, ok in rows: print(f"  {'PASS' if ok else 'FAIL'}  {name}: {val}")
    sys.exit(0 if all(ok for _, _, ok in rows) else 1)


if __name__ == "__main__":
    main()
