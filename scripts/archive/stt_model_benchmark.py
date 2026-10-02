#!/usr/bin/env python3
"""
Comprehensive STT model benchmark on the new voice.
Compares tiny / base / small / large-v3-turbo on piper_omni_hq.wav
Outputs: processed/stt_model_benchmark.json
"""

import json, os, sys, time
import numpy as np
import librosa
import jiwer
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from backend.stt.faster_whisper_stt import FasterWhisperSTT

WAV = os.path.join(REPO_ROOT, "processed", "engine_ab", "piper_omni_hq.wav")
OUT = os.path.join(REPO_ROOT, "processed", "stt_model_benchmark.json")
TEXT = "Včera ráno som išiel na trh kúpiť čerstvý chlieb a mlieko. Stretol som tam starého priateľa Ľuba, ktorý predával med a syry. Porozprával som mu o svojej práci a o dlhej ceste vlakom cez hory a doliny."

def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower(), flags=re.UNICODE)).strip()

def run_benchmark(model_id):
    stt = FasterWhisperSTT(model_size=model_id)
    wav, sr = librosa.load(WAV, sr=16000, mono=True)
    ref = norm(TEXT)
    
    start = time.perf_counter()
    segs, stt_s, _ = stt.transcribe_audio(np.asarray(wav, dtype=np.float32), sr, language="sk")
    wall = time.perf_counter() - start
    
    hyp = norm(" ".join(x.text for x in segs))
    wer = jiwer.wer(ref, hyp)
    cer = jiwer.cer(ref, hyp)
    audio_s = len(wav) / sr
    rtf = stt_s / audio_s
    
    return {
        "model": model_id,
        "rtf": round(rtf, 3),
        "stt_s": round(stt_s, 3),
        "wall_s": round(wall, 3),
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "hyp": hyp[:200]
    }

def main():
    models = ["tiny", "base", "small", "large-v3-turbo"]
    results = []
    print("=" * 60)
    print("STT MODEL BENCHMARK - piper_omni_hq (new SK voice)")
    print("=" * 60)
    
    for m in models:
        print(f"Testing {m}...", flush=True)
        try:
            r = run_benchmark(m)
            results.append(r)
            print(f"  RTF={r['rtf']:.3f} STT={r['stt_s']:.3f}s WER={r['wer']:.4f} CER={r['cer']:.4f}")
            print(f"  Hyp: {r['hyp'][:120]}")
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Also test the original personal voice for comparison
    print()
    print("Comparing against shipped voice (piper_personal.wav)...")
    PERSONAL_WAV = os.path.join(REPO_ROOT, "processed", "engine_ab", "piper_personal.wav")
    for m in models:
        stt = FasterWhisperSTT(model_size=m)
        wav, sr = librosa.load(PERSONAL_WAV, sr=16000, mono=True)
        ref = norm(TEXT)
        start = time.perf_counter()
        segs, stt_s, _ = stt.transcribe_audio(np.asarray(wav, dtype=np.float32), sr, language="sk")
        wall = time.perf_counter() - start
        hyp = norm(" ".join(x.text for x in segs))
        wer = jiwer.wer(ref, hyp)
        cer = jiwer.cer(ref, hyp)
        audio_s = len(wav) / sr
        rtf = stt_s / audio_s
        print(f"  {m} (personal voice): RTF={rtf:.3f} STT={stt_s:.3f}s WER={wer:.4f} CER={cer:.4f}")
    
    with open(OUT, "w") as f:
        json.dump({"text": TEXT, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {OUT}")

if __name__ == "__main__":
    main()