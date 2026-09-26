#!/usr/bin/env python3
"""STT baseline on the owner's real recordings vs the known reading script.

Ground truth = EN_SENTENCES in scripts/make_reading_script.py (the exact text read
for the EN session) + the proofread SK column of documentation/reading_script_bilingual.md.
Metric: WER (jiwer) + wall RTF. Run: venv/bin/python scripts/stt_baseline_new_voices.py
Out: processed/stt_baseline.json — the number that decides whether a Parakeet (or any
other STT) spike is justified, and the thesis STT-accuracy table seed.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO_ROOT, "processed", "stt_baseline.json")


def normalize(s: str) -> str:
    return " ".join(s.lower().strip().split())


def normalize_plain(s: str) -> str:
    """Diacritics-stripped variant: separates orthography errors (rano vs ráno)
    from real word errors. If plain-WER << raw-WER, the model hears SK but never
    learned to spell it — a training-data gap, not an acoustic one."""
    import unicodedata
    no_dia = "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )
    return normalize(no_dia)


def main():
    from backend.stt.faster_whisper_stt import FasterWhisperSTT
    from make_reading_script import EN_SENTENCES

    try:
        from jiwer import wer
    except ImportError:
        sys.exit("jiwer not installed in this venv")

    import librosa
    import numpy as np

    def load_clip(name):
        wav, sr = librosa.load(os.path.join(REPO_ROOT, "speaker_voices", name),
                               sr=16000, mono=True)
        return np.asarray(wav, dtype=np.float32), sr

    def hyp_text(segments):
        out = []
        for s in segments:
            out.append(s["text"] if isinstance(s, dict) else getattr(s, "text", ""))
        return normalize(" ".join(out))

    stt = FasterWhisperSTT(model_size="base")
    results = []

    # EN: ground truth is exact (script sentences).
    wav, sr = load_clip("en_script_reading.m4a")
    segments, stt_time, lang = stt.transcribe_audio(wav, sr, language="en")
    hyp = hyp_text(segments)
    ref = normalize(" ".join(EN_SENTENCES))
    results.append({
        "clip": "en_script_reading.m4a", "lang": "en", "model": "base",
        "wer": round(wer(ref, hyp), 4),
        "wer_plain": round(wer(normalize_plain(ref), normalize_plain(hyp)), 4),
        "stt_time_s": round(stt_time, 2),
        "audio_s": round(len(wav) / sr, 1),
        "detected_lang": lang,
    })

    # SK: ground truth is the proofread SK column (owner-corrected 2026-09-26).
    sk_refs = []
    md = os.path.join(REPO_ROOT, "documentation", "reading_script_bilingual.md")
    for line in open(md).read().splitlines():
        if line.startswith("|") and line[2:3].isdigit():
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) == 3:
                sk_refs.append(parts[2])
    sk_path = os.path.join(REPO_ROOT, "speaker_voices", "sk_script_reading.m4a")
    wav, sr = load_clip("sk_script_reading.m4a")
    segments, stt_time, lang = stt.transcribe_audio(wav, sr, language="sk")
    hyp = hyp_text(segments)
    ref = normalize(" ".join(sk_refs))
    results.append({
        "clip": "sk_script_reading.m4a", "lang": "sk", "model": "base",
        "wer": round(wer(ref, hyp), 4),
        "wer_plain": round(wer(normalize_plain(ref), normalize_plain(hyp)), 4),
        "stt_time_s": round(stt_time, 2),
        "audio_s": round(len(wav) / sr, 1),
        "detected_lang": lang,
        "note": "SK read may paraphrase/script order may differ — inspect before citing",
    })

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    for r in results:
        print(f"{r['clip']}: WER {r['wer']} (plain {r['wer_plain']})  "
              f"STT {r['stt_time_s']}s / audio {r['audio_s']}s")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
