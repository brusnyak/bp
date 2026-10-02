#!/usr/bin/env python3
"""
Quick STT speed test comparing different Whisper models.
"""

import time
import numpy as np
import librosa
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from backend.stt.faster_whisper_stt import FasterWhisperSTT

def test_stt_speed(model_size, wav_path):
    """Test STT inference speed for a given model."""
    stt = FasterWhisperSTT(model_size=model_size)
    wav, sr = librosa.load(wav_path, sr=16000, mono=True)
    
    start = time.perf_counter()
    segs, stt_s, _ = stt.transcribe_audio(np.asarray(wav, dtype=np.float32), sr, language="sk")
    end = time.perf_counter()
    
    hyp = " ".join(x.text for x in segs)
    rtf = stt_s / (len(wav) / sr)
    
    return {
        "model": model_size,
        "rtf": rtf,
        "stt_s": stt_s,
        "wall_s": end - start,
        "hyp": hyp[:100] + "..." if len(hyp) > 100 else hyp
    }

def main():
    # Use the new voice test file
    wav_path = os.path.join(REPO_ROOT, "processed", "voice_qc", "piper_omni_hq_sk_test.wav")
    
    if not os.path.exists(wav_path):
        print(f"Test file not found: {wav_path}")
        return
    
    print(f"Testing STT speed with: {wav_path}")
    print("=" * 50)
    
    models = ["base", "small"]
    
    results = []
    for model in models:
        print(f"Testing {model} model...")
        try:
            result = test_stt_speed(model, wav_path)
            results.append(result)
            print(f"  RTF: {result['rtf']:.3f}")
            print(f"  STT time: {result['stt_s']:.3f}s")
            print(f"  Wall time: {result['wall_s']:.3f}s")
            print(f"  Hypothesis: {result['hyp']}")
            print()
        except Exception as e:
            print(f"  Error testing {model}: {e}")
            print()
    
    # Print summary
    print("SUMMARY:")
    print("-" * 50)
    for r in results:
        print(f"{r['model']:>6}: RTF={r['rtf']:.3f}, STT={r['stt_s']:.3f}s")

if __name__ == "__main__":
    main()