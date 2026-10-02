#!/usr/bin/env python3
"""Machine pre-grade every Voice Lab item -> processed/ear_grades.json.

Reads ui/voice-lab/library.json (source of truth for what the Lab shows) and
attaches a machine advisory per audio item. Ear verdicts stay empty for the
owner to fill in the Lab; re-running never overwrites existing ear grades.

Grading criteria per section (v1, 2026-10-01):
  voices    reference mic takes graded on clone-input fitness (registry +
            transcript present). Ear decides keep/kill per take.
  qc        engine outputs graded on measured similarity when present,
            else unscored. Ear decides character/naturalness.
  corpus_*  training clips graded on qc_wer_smallsk (<=0.10 pass, <=0.20
            review, above kill-candidate) + rms sanity flag (0.02-0.30).
  zeroshot  clone outputs, advisory from receipt rtf only. Ear A/B decides.
  spikes    benchmark numbers only, not gradable voices (gradable=false).
  conversation / demo_audio / meeting
            functional pipeline outputs, not voice grades (gradable=false,
            ear optional for translation quality).
  sk_direction / stt_input / stream_audit / charts
            evidence material, not gradable (gradable=false).

Output schema:
  {"version": 1, "grades": [{"section", "name", "gradable", "machine":
    {"grade": "pass|review|kill|unscored|na", "signals": {...}, "note": ""},
    "ear": {"grade": null, "keep": null, "note": ""}}]}

Usage: python3 scripts/grade_library.py [--apply]
  default prints a rollup; --apply writes processed/ear_grades.json
  (preserving any ear verdicts already stored).
"""

import json
import os
import struct
import sys
import wave

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBRARY = os.path.join(REPO_ROOT, "ui", "voice-lab", "library.json")
OUT = os.path.join(REPO_ROOT, "processed", "ear_grades.json")

GRADABLE = {"voices", "qc", "corpus_sk", "corpus_en", "zeroshot",
            "conversation", "demo_audio", "meeting", "spikes", "stt_input", "gpu_clones"}
EAR_OPTIONAL = {"conversation", "demo_audio", "meeting"}

WAV_DIRS = {
    "corpus_sk": "processed/omni_hq_sk",
    "corpus_en": "processed/omni_hq_en",
}


def wav_stats(section, name):
    """Stdlib PCM stats (rms peak, clip%) for corpus clips lacking STT QC.
    Returns {} when the file is missing or not PCM-readable."""
    d = WAV_DIRS.get(section)
    if not d:
        return {}
    p = os.path.join(REPO_ROOT, d, name + ".wav")
    try:
        with wave.open(p, "rb") as w:
            n, sw, ch = w.getnframes(), w.getsampwidth(), w.getnchannels()
            if sw not in (1, 2) or n == 0:
                return {}
            raw = w.readframes(n)
    except (OSError, wave.Error):
        return {}
    fmt = f"{len(raw) // sw}{'b' if sw == 1 else 'h'}"
    try:
        samples = struct.unpack("<" + fmt, raw)
    except struct.error:
        return {}
    peak = max(abs(s) for s in samples) / (128 if sw == 1 else 32768)
    mean_sq = sum(s * s for s in samples[::max(1, len(samples) // 20000)]) / max(
        1, len(samples) // max(1, len(samples) // 20000))
    rms = (mean_sq ** 0.5) / (128 if sw == 1 else 32768)
    return {"rms": round(rms, 4), "peak": round(peak, 4),
            "dur_s": round(n / w.getframerate(), 1)}


def machine_grade(section, item):
    meta = item.get("meta", {}) or {}
    if section in ("corpus_sk", "corpus_en"):
        wer = meta.get("qc_wer")
        rms = meta.get("qc_rms")
        signals = {"qc_wer": wer, "qc_rms": rms}
        notes = []
        if wer is None:
            # Fallback: stdlib audio sanity so every clip still gets a
            # machine signal; names the missing STT QC explicitly.
            st = wav_stats(section, item.get("name", ""))
            signals.update(st)
            rung = "small-sk" if section == "corpus_sk" else "EN rung"
            if not st:
                return "unscored", signals, \
                    f"no {rung} WER and wav unreadable; ear decides"
            if st["peak"] >= 0.99:
                return "review", signals, \
                    f"no {rung} WER; clipping suspected (peak {st['peak']}); ear decides"
            if not 0.02 <= st["rms"] <= 0.30:
                return "review", signals, \
                    f"no {rung} WER; rms {st['rms']} outside 0.02-0.30; ear decides"
            return "review", signals, \
                f"no {rung} WER (audio sane: rms {st['rms']}); ear decides"
        if wer <= 0.10:
            grade = "pass"
        elif wer <= 0.20:
            grade = "review"
            notes.append("WER in review band")
        else:
            grade = "kill"
            notes.append("WER above 0.20 kills training value")
        if rms is not None and not 0.02 <= rms <= 0.30:
            grade = "review" if grade == "pass" else grade
            notes.append(f"rms {rms} outside 0.02-0.30 sanity band")
        return grade, signals, "; ".join(notes) if notes else "WER in pass band"
    if section == "qc":
        sim = item.get("similarity")
        signals = {"similarity": sim,
                   "synthesis_latency_s": item.get("synthesis_latency_s")}
        if sim is None:
            return "unscored", signals, "no scores.json entry; ear decides"
        if sim >= 0.84:
            return "pass", signals, "similarity above 0.84 threshold"
        return "review", signals, "similarity below 0.84 threshold"
    if section == "voices":
        signals = {"in_registry": item.get("in_registry"),
                   "language": item.get("language"),
                   "has_transcript": bool(item.get("transcript"))}
        if item.get("in_registry") and item.get("transcript"):
            return "pass", signals, "registered reference take with transcript"
        return "review", signals, "missing registry entry or transcript"
    if section == "gpu_clones":
        return "unscored", {k: meta.get(k) for k in ("engine", "gpu")}, "GPU clone / S2ST output; ear decides"
    if section == "zeroshot":
        signals = {k: meta.get(k) for k in ("rtf", "reference")}
        return "unscored", signals, "clone output; blind A/B against Piper pair decides"
    return "na", {}, "evidence/functional material, not a voice grade"


def main():
    apply = "--apply" in sys.argv
    with open(LIBRARY) as f:
        library = json.load(f)
    old = {}
    if os.path.exists(OUT):
        try:
            with open(OUT) as f:
                old = {(g["section"], g["name"]): g.get("ear", {})
                       for g in json.load(f).get("grades", [])}
        except (OSError, ValueError):
            pass
    grades = []
    for section in library.get("sections", []):
        sid = section.get("id", "")
        for item in section.get("items", []):
            if not item.get("file"):
                continue  # table-only entries (_matrix etc.)
            grade, signals, note = machine_grade(sid, item)
            ear = old.get((sid, item.get("name")), {})
            grades.append({
                "section": sid,
                "name": item.get("name", ""),
                "gradable": sid in GRADABLE,
                "machine": {"grade": grade, "signals": signals, "note": note},
                "ear": {"grade": ear.get("grade"),
                        "keep": ear.get("keep"),
                        "sim": ear.get("sim"),
                        "defects": ear.get("defects", {}),
                        "note": ear.get("note", "")},
            })
    rollup = {}
    for g in grades:
        key = (g["section"], g["machine"]["grade"])
        rollup[key] = rollup.get(key, 0) + 1
    print(f"items: {len(grades)} "
          f"({sum(1 for g in grades if g['gradable'])} gradable)")
    for (section, grade), n in sorted(rollup.items()):
        print(f"  {section:14s} {grade:9s} {n}")
    if apply:
        with open(OUT, "w") as f:
            json.dump({"version": 1, "grades": grades}, f, indent=2,
                      ensure_ascii=False)
        print(f"wrote {os.path.relpath(OUT, REPO_ROOT)} "
              f"({sum(1 for g in grades if g['ear'].get('grade') is not None)} "
              "ear grades preserved)")
    else:
        print("dry run; re-run with --apply to write " +
              os.path.relpath(OUT, REPO_ROOT))


if __name__ == "__main__":
    main()
