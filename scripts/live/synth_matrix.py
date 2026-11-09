#!/usr/bin/env python3
"""Run every segmentation strategy over the generated reference speech (scripts/live/make_test_audio.py) and print one comparison table.
Recognition only, offline, computation-aware lag. Heavy: run through ~/.claude/scripts/heavy and not while someone is testing in the UI.

  PYTHONPATH=. .venv/bin/python scripts/live/synth_matrix.py [--langs sk en de] [--scenarios paced continuous long] [--strategies whole pause clause chunk2] [--out processed/testaudio/matrix.json]
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
import strategy_lab as L


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--langs", nargs="+", default=["sk", "en", "de"]); ap.add_argument("--scenarios", nargs="+", default=["paced", "continuous", "long"])
    ap.add_argument("--strategies", nargs="+", default=["whole", "pause", "clause", "chunk2"]); ap.add_argument("--engine", default="mlx"); ap.add_argument("--out", default=str(ROOT / "processed/testaudio/matrix.json"))
    a = ap.parse_args(); L.PASSES = 1; rows = []
    for lang in a.langs:
        eng = L.make_engine(a.engine, None, lang)
        for sc in a.scenarios:
            base = ROOT / "processed/testaudio" / f"{lang}_{sc}"; audio = L.load(f"{base}.wav"); ref = L.norm(json.loads(Path(f"{base}.json").read_text())["text"])
            for st in a.strategies:
                t = time.time()
                items = (L.s_whole(eng, audio, lang) if st == "whole" else L.s_chunk(eng, audio, lang) if st == "chunk2" else L.s_pause(eng, audio, lang) if st == "pause"
                         else L.s_pause(eng, audio, lang, 0.2, 4.0) if st == "clause" else L.s_agree(eng, audio, lang, float(st[5:])))
                r = L.report(st, items, len(audio) / L.SR, ref); r.update(lang=lang, scenario=sc, wall_s=round(time.time() - t, 1)); rows.append(r)
                print(f"{lang:2} {sc:10} {st:9} error {r['error_pct']:5.1f}%  word_lag {r['word_lag_median_s']}  end_lag {r['end_lag_median_s']} (max {r['end_lag_max_s']})  decodes {r['decodes']}", flush=True)
    Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
