#!/usr/bin/env python3
"""Clean-synth STT probe: is the residual SK error acoustic or config?

Synthesizes a known Slovak passage with the default voices (jirka male,
lili female), then recognizes it under 4 configs:
  A: proper 16 kHz resample, lang=sk, whisper-vad ON   (eval regime)
  B: proper 16 kHz resample, lang=sk, whisper-vad OFF  (live regime)
  C: NO resample, 22050 Hz audio labelled 16 kHz       (suspected old-bug replay)
  D: proper 16 kHz resample, lang=auto, whisper-vad OFF (live auto regime)

Clean synth with known text should score WER ~0. If A/B are clean but the
mic clips score 0.25-0.50, the gap is acoustic (mic/reverb/accent), not
config. If C reproduces ~0.16, the old clean_synth_matrix number was a
resampling artifact.

Usage:
  .venv/bin/python scripts/bench_stt_synth.py [--text "..."] [--voices jirka,lili]

Writes processed/stt_beam/synth_probe.json
"""
import argparse
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

OUT_JSON = os.path.join(REPO_ROOT, "processed", "stt_beam", "synth_probe.json")

DEFAULT_TEXT = ("Dobrý deň, vitajte na dnešnom stretnutí o preklade reči v reálnom čase. "
                "Systém prepisuje hovorené slovo, prekladá ho a prehráva v druhom jazyku.")

VOICES = {"jirka": "cs_CZ-jirka-medium", "lili": "sk_SK-lili-medium"}


def normalize(text):
    text = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--voices", default="jirka,lili")
    ap.add_argument("--model", default="small-sk", help="STT model size / rung")
    ap.add_argument("--lang", default="sk", help="STT language tag")
    ap.add_argument("--tts-lang", default="sk", help="language tag for synthesis")
    ap.add_argument("--tag", default="", help="suffix for output file + run keys")
    args = ap.parse_args();
    out_json = OUT_JSON if not args.tag else os.path.join(
        os.path.dirname(OUT_JSON), f"synth_probe_{args.tag}.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    import numpy as np
    import jiwer
    import librosa
    from backend import hardware
    from backend.tts.piper_tts import PiperTTS
    from backend.stt.faster_whisper_stt import FasterWhisperSTT

    ref = normalize(args.text)
    stt = FasterWhisperSTT(model_size=args.model, compute_type="int8")
    out = {"text": args.text, "model": args.model, "lang": args.lang, "runs": {}}

    for voice in [v.strip() for v in args.voices.split(",")]:
        tts = PiperTTS(model_id=VOICES[voice],
                       device=hardware.detect_backend("tts_baseline"))
        wav_native, sr_native, _ = tts.synthesize(args.text, language=args.tts_lang)
        wav_native = np.asarray(wav_native, dtype=np.float32)
        wav_16k = librosa.resample(wav_native, orig_sr=sr_native, target_sr=16000).astype(np.float32)
        print(f"{voice}: native {sr_native} Hz, {len(wav_native) / sr_native:.1f}s", flush=True)

        configs = {
            "A_eval": (wav_16k, args.lang, True),
            "B_live": (wav_16k, args.lang, False),
            "C_noresample": (wav_native, args.lang, True),
            "D_auto": (wav_16k, None, False),
        }
        for cfg, (audio, lang, vad) in configs.items():
            key = f"{voice}/{cfg}"
            if args.tag:
                key = f"{args.tag}/{key}"
            try:
                segs, t, detected = stt.transcribe_audio(audio, 16000, language=lang, vad_filter=vad)
                hyp = normalize(" ".join(s.text if hasattr(s, "text") else s["text"] for s in segs))
                out["runs"][key] = {
                    "wer": round(jiwer.wer(ref, hyp), 4),
                    "cer": round(jiwer.cer(ref, hyp), 4),
                    "stt_s": round(t, 2),
                    "detected": detected,
                    "hyp": hyp,
                }
            except Exception as e:
                out["runs"][key] = {"error": f"{type(e).__name__}: {e}"}
            r = out["runs"][key]
            print(f"{key:16s} WER {r.get('wer', 'ERR')} CER {r.get('cer', '')} "
                  f"det={r.get('detected')} hyp={r.get('hyp', r.get('error'))[:80]}", flush=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
