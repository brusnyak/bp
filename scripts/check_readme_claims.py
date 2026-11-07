#!/usr/bin/env python3
"""Fail when a measured number quoted in a doc differs from the saved receipt it came from.

Mark a number in any markdown file as  <!-- claim:ID -->3.48<!-- /claim -->  (the comment pair is invisible when rendered).
This script recomputes ID from the evidence files and compares within the tolerance below.
Run: python3 scripts/check_readme_claims.py [files...]   (default: README.md documentation/*.md)  | make check-docs
Born from 2026-11-07: three README numbers (0.39 s, a Parakeet default, p50 1.86/p95 4.22 s) did not match what was measured.
"""
import glob, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "documentation/evidence/2026-11-07/lang_matrix.json"
KAGGLE = ROOT / "route_receipt.json"
PACED = ROOT / "documentation/evidence/2026-11-07/paced_replay_summary.json"
SIM = ROOT / "documentation/evidence/2026-11-07/stream_sim.json"
TOL = 0.06  # relative; also accepts <=0.06 absolute for values under 1


def pct(vals, q):
    v = sorted(vals)
    return v[min(len(v) - 1, int(q * len(v)))]


def kaggle_profile(profile, q):
    d = json.loads(KAGGLE.read_text())
    pr = next(p for p in d["profiles"] if p["profile"] == profile)
    return pct([u["onset_to_first_audio_s"] for r in pr["repeats"] for u in r["utterances"] if u.get("onset_to_first_audio_s") is not None], q)


def resolve(cid):
    """ID grammar: kaggle:<profile>:p50|p95 | stt:<model>:<lang>:wer | mt:<model>:<lang>:<to_en|from_en>:chrf | paced:<key> | sim:<cell>:wer|lat|p90"""
    k = cid.split(":")
    if k[0] == "kaggle":
        return kaggle_profile(k[1], {"p50": 0.5, "p95": 0.95}[k[2]])
    m = json.loads(MATRIX.read_text())
    if k[0] == "stt":
        return m["stt_t4"][f"{k[1]}|{k[2]}"]["WER"] * 100
    if k[0] == "mt":
        return m["mt_t4"][f"{k[1]}|{k[2]}|{k[3]}"]["chrF"]
    if k[0] == "sim":      # sim:<stream>|<engine>|<strategy>:wer|lat|p90   (T4 streaming simulation)
        c = json.loads(SIM.read_text())["cells"][k[1]]
        return {"wer": c["WER"] * 100, "lat": c["latency_median_s"], "p90": c["latency_p90_s"]}[k[2]]
    if k[0] == "paced":
        return json.loads(PACED.read_text())[k[1]]
    raise KeyError(cid)


def main(files):
    bad = n = 0
    for f in files:
        text = Path(f).read_text(errors="ignore")
        for cid, val in re.findall(r"<!--\s*claim:([^\s>]+)\s*-->\s*([-\d.]+)\s*<!--\s*/claim\s*-->", text):
            n += 1
            try:
                actual = resolve(cid)
            except Exception as e:
                print(f"FAIL {f}: claim {cid} cannot be resolved ({type(e).__name__}: {e})"); bad += 1; continue
            shown = float(val)
            if abs(shown - actual) > max(TOL * abs(actual), 0.06 if abs(actual) < 1 else 0.0):
                print(f"FAIL {f}: {cid} says {shown}, evidence says {actual:.3f}"); bad += 1
            else:
                print(f"ok   {f}: {cid} = {shown} (evidence {actual:.3f})")
    print(f"{n} claims checked, {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    files = sys.argv[1:] or ["README.md"] + sorted(glob.glob("documentation/*.md"))
    sys.exit(main(files))
