#!/usr/bin/env python3
"""Controlled EN<->SK pipeline evaluation: source voice, recognizer, and target voice.

This answers three separate questions without conflating them:
  * Does Slovak source pronunciation change STT quality?
  * What does base vs Slovak-tuned small cost in latency and buy in WER/CER?
  * Does the personal Piper target voice cost latency versus the generic target voice?

The same manifest sentence is evaluated from generic Piper, personal Piper, and (when
recorded) real microphone audio. Output is local-only under processed/voice_control/.

Before the real-audio rows exist:
  python scripts/build_recording_set.py
  .venv/bin/python scripts/record_reading.py --lang sk
  .venv/bin/python scripts/record_reading.py --lang en

Run (first 20 sentences):
  .venv/bin/python scripts/eval_bilingual_voice_control.py --count 20
"""
import argparse, json, re, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PIPER = {
    "sk_generic": "sk_SK-lili-medium", "sk_personal": "sk_SK-personal-male-medium",
    "en_generic": "en_US-ryan-medium", "en_personal": "en_US-personal-v2",
}

def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)).strip()

def median(values):
    return round(statistics.median(values), 4) if values else None

def load_manifest(path, count):
    rows = json.loads(path.read_text(encoding="utf-8"))[:count]
    if not rows or any(not r.get("sk") or not r.get("en") for r in rows):
        raise ValueError("manifest needs non-empty sk and en fields")
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "eval_data/recording_set_v2/manifest.json"))
    ap.add_argument("--recordings", default=str(ROOT / "eval_data/recording_set_v2"), help="contains sk_00.wav and en_00.wav")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--beam-size", type=int, default=5, choices=(1, 2, 5))
    ap.add_argument("--save-examples", type=int, default=3, help="per matrix cell; keeps output listenable")
    args = ap.parse_args()
    if args.count < 1: ap.error("--count must be positive")
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        ap.error("recording manifest missing; run scripts/build_recording_set.py first")

    import jiwer, librosa, numpy as np, soundfile as sf
    from sacrebleu import sentence_chrf
    from backend.stt.faster_whisper_stt import FasterWhisperSTT
    from backend.mt.ctranslate2_mt import CTranslate2MT
    from backend.tts.piper_tts import PiperTTS

    rows = load_manifest(manifest_path, args.count)
    out = ROOT / "processed" / "voice_control"; fixtures = out / "fixtures"; examples = out / "examples"
    fixtures.mkdir(parents=True, exist_ok=True); examples.mkdir(parents=True, exist_ok=True)
    engines = {name: PiperTTS(model_id=model) for name, model in PIPER.items()}
    mt = {"en_sk": CTranslate2MT("Helsinki-NLP/opus-mt-en-sk"), "sk_en": CTranslate2MT("Helsinki-NLP/opus-mt-sk-en")}
    stt = {"en_base": FasterWhisperSTT("base", beam_size=args.beam_size),
           "sk_base": FasterWhisperSTT("base", beam_size=args.beam_size),
           "sk_small-sk": FasterWhisperSTT("small-sk", beam_size=args.beam_size)}
    rec_dir = Path(args.recordings)

    def inputs(lang):
        variants = [("generic", engines[f"{lang}_generic"]), ("personal", engines[f"{lang}_personal"])]
        result = []
        for label, engine in variants:
            clips = []
            for row in rows:
                path = fixtures / f"{lang}_{label}_{row['id']:02d}.wav"
                wav, sr, _ = engine.synthesize(row[lang], language=lang)
                sf.write(path, wav, sr); clips.append((row, path))
            result.append(("synthetic_" + label, clips))
        real = [(row, rec_dir / f"{lang}_{row['id']:02d}.wav") for row in rows]
        if all(path.exists() for _, path in real): result.append(("real_microphone", real))
        return result

    results = []
    for src, tgt in (("en", "sk"), ("sk", "en")):
        for source_kind, clips in inputs(src):
            rung_names = ["en_base"] if src == "en" else ["sk_base", "sk_small-sk"]
            for rung in rung_names:
                hypotheses, translations, stt_times, mt_times = [], [], [], []
                for row, path in clips:
                    wav, _ = librosa.load(path, sr=16000, mono=True)
                    segs, elapsed, _ = stt[rung].transcribe_audio(np.asarray(wav, dtype=np.float32), 16000, language=src)
                    hyp = " ".join(s.text for s in segs).strip(); hypotheses.append((row[src], hyp)); stt_times.append(elapsed)
                    translated, mt_elapsed = mt[src + "_" + tgt].translate(hyp, src, tgt); translations.append((row[tgt], translated)); mt_times.append(mt_elapsed)
                for target_kind in ("generic", "personal"):
                    engine = engines[f"{tgt}_{target_kind}"]; tts_times = []; sample_paths = []
                    for n, ((row, translated), (_, input_path)) in enumerate(zip(translations, clips)):
                        wav, sr, elapsed = engine.synthesize(translated, language=tgt); tts_times.append(elapsed)
                        if n < args.save_examples:
                            dst = examples / f"{src}-{tgt}_{source_kind}_{rung}_{target_kind}_{row['id']:02d}.wav"
                            sf.write(dst, wav, sr); sample_paths.append(str(dst.relative_to(ROOT)))
                    ref = " ".join(norm(a) for a, _ in hypotheses); hyp = " ".join(norm(b) for _, b in hypotheses)
                    mt_ref = " ".join(a for a, _ in translations); mt_hyp = " ".join(b for _, b in translations)
                    results.append({"direction": src + "→" + tgt, "source": source_kind, "stt": rung,
                        "target_voice": target_kind, "n": len(clips), "beam_size": args.beam_size,
                        "stt_wer": round(jiwer.wer(ref, hyp), 4), "stt_cer": round(jiwer.cer(ref, hyp), 4),
                        "mt_chrf": round(sentence_chrf(mt_hyp, [mt_ref]).score, 2), "stt_median_s": median(stt_times),
                        "mt_median_s": median(mt_times), "tts_median_s": median(tts_times),
                        "pipeline_median_s": round(median(stt_times) + median(mt_times) + median(tts_times), 4),
                        "examples": sample_paths})
                    r = results[-1]
                    print(f"{r['direction']:5} {source_kind:18} {rung:11} → {target_kind:8} WER {r['stt_wer']:.3f} chrF {r['mt_chrf']:5.1f} total {r['pipeline_median_s']:.2f}s")
    payload = {"protocol": "bilingual-voice-control-v1", "count": len(rows), "results": results,
               "note": "WER/CER score source STT; chrF scores MT against paired reference; target voice quality requires blind listening to saved examples."}
    (out / "control_matrix.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", out / "control_matrix.json")

if __name__ == "__main__": main()
