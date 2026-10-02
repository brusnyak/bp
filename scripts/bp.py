#!/usr/bin/env python3
"""bp — single entry for every repeatable workflow in this repo.

Replaces the scripts/ sprawl: one CLI, one --help, same behavior (each command
reuses the original module's code path, nothing rewritten).

    venv/bin/python scripts/bp.py corpus      # normalize/dedupe/transcribe voices
    venv/bin/python scripts/bp.py qc --synthesize-only
    venv/bin/python scripts/bp.py stt --models base,small --clip sk
    venv/bin/python scripts/bp.py e2e         # full EN->SK timed run, new voice
    venv/bin/python scripts/bp.py library     # refresh Voice Lab manifest
    venv/bin/python scripts/bp.py grades      # machine pre-grades + merge into manifest
    venv/bin/python scripts/bp.py grades --import voice-ratings.json
                                              # import Lab ear verdicts (Download my ratings)
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
    "demo-audio": "demo_audio.py",
    "meeting": "meeting_sim.py",
    "engine-ab": "engine_ab.py",
    "listen": "machine_listen_qc.py",
    "bulk": "bulk_hq.py",
}


def cmd_bench(args):
    """New-model bench on fixed owner clips (reports/model_landscape_2026-10.md §6).
    engine nemotron uses the pure-C runtime (scripts/bench_nemotron.py);
    everything else goes through scripts/bench_new_models.py."""
    if args.engine == "nemotron":
        sys.argv = ["bench_nemotron.py"]
        runpy.run_path(os.path.join(REPO_ROOT, "scripts", "bench_nemotron.py"),
                       run_name="__main__")
    else:
        sys.argv = ["bench_new_models.py", "--engine", args.engine,
                    "--clips", args.clips]
        runpy.run_path(os.path.join(REPO_ROOT, "scripts", "bench_new_models.py"),
                       run_name="__main__")


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


def cmd_grades(args):
    """Machine pre-grade the Lab + merge into the manifest.

    Default: grade_library.py --apply (never overwrites stored ear verdicts),
    then update_voice_lab_library.py --no-test so library.json carries the
    fresh machine notes. --import loads the browser JSON from the Lab's
    "Download my ratings" button into processed/ear_grades.json; re-run
    without flags afterwards to merge.
    """
    import json
    repo = REPO_ROOT
    grades_path = os.path.join(repo, "processed", "ear_grades.json")
    if args.pull:
        import urllib.request
        token = args.token or os.environ.get("BP_TOKEN")
        if not token:
            sys.exit("need --token or BP_TOKEN (login token from /ui/auth/auth.html)")
        base = args.server.rstrip("/")
        req = urllib.request.Request(
            base + "/api/ratings", headers={"Authorization": "Bearer " + token})
        with urllib.request.urlopen(req, timeout=15) as r:
            store = json.load(r)
        with open(grades_path) as f:
            local = json.load(f)
        by_name = {g["name"]: g for g in local.get("grades", [])}
        n = 0
        for g in store.get("grades", []):
            ear = g.get("ear") or {}
            if ear.get("grade") is None and ear.get("keep") is None \
                    and not ear.get("note") and not ear.get("defects"):
                continue
            tgt = by_name.get(g.get("name"))
            if tgt is None:
                continue
            if (ear.get("updated") or 0) >= ((tgt.get("ear") or {}).get("updated") or 0):
                tgt["ear"] = ear
                n += 1
        with open(grades_path, "w") as f:
            json.dump(local, f, indent=2, ensure_ascii=False)
        print(f"pulled {n} ear verdicts from {base}")
        return
    if args.import_json:
        with open(args.import_json) as f:
            browser = json.load(f)
        with open(grades_path) as f:
            store = json.load(f)
        by_name = {}
        for g in store.get("grades", []):
            by_name.setdefault(g["name"], []).append(g)
        n, ambiguous = 0, []
        for name, r in browser.items():
            targets = by_name.get(name)
            if not targets:
                continue
            if len(targets) > 1:
                ambiguous.append(name)
            defects = {k: r.get(k) for k in ("steadiness", "hiss", "muffled")
                       if r.get(k)}
            ear = {"grade": r.get("grade") or None,
                   "keep": ({1: "keep", 2: "kill"}.get(r.get("keep"))),
                   "sim": r.get("sim"),
                   "defects": defects,
                   "note": r.get("note", "")}
            if ear["grade"] == 0:
                ear["grade"] = None
            for g in targets:
                g["ear"] = ear
                n += 1
        with open(grades_path, "w") as f:
            json.dump(store, f, indent=2, ensure_ascii=False)
        print(f"imported {n} ear verdicts from {args.import_json}")
        if ambiguous:
            print("ambiguous names applied to every section: " + ", ".join(ambiguous))
        return
    for script, script_args in (("grade_library.py", ["--apply"]),
                                ("update_voice_lab_library.py", ["--no-test"])):
        sys.argv = [script] + script_args
        runpy.run_path(os.path.join(repo, "scripts", script), run_name="__main__")
        print("---")


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

    p = sub.add_parser("grades", help="machine pre-grade the Lab + merge into manifest")
    p.add_argument("--import-json", default=None, metavar="FILE",
                   help="import Lab browser ratings JSON into processed/ear_grades.json")
    p.add_argument("--pull", action="store_true",
                   help="pull ear verdicts from a running backend (/api/ratings) into processed/ear_grades.json")
    p.add_argument("--token", default=None, help="login token for --pull (or BP_TOKEN env)")
    p.add_argument("--server", default="https://localhost:8000",
                   help="backend base URL for --pull")

    p = sub.add_parser("bench", help="new-model bench on fixed owner clips")
    p.add_argument("--engine", default="seamless",
                   help="nemotron (pure-C runtime) or bench_new_models.py engine")
    p.add_argument("--clips", default="me_sk_trhove_b",
                   help="comma list for bench_new_models.py engines")

    args = parser.parse_args()
    if args.cmd == "stt":
        cmd_stt(args)
    elif args.cmd == "grades":
        cmd_grades(args)
    elif args.cmd == "bench":
        cmd_bench(args)
    else:
        sys.argv = [WRAPPED[args.cmd]] + (args.rest[1:] if args.rest[:1] == ["--"] else args.rest)
        runpy.run_path(os.path.join(REPO_ROOT, "scripts", WRAPPED[args.cmd]),
                       run_name="__main__")


if __name__ == "__main__":
    main()
