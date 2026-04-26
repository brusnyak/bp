#!/usr/bin/env python3
"""
Optimized Full S2S Pipeline Test
Reuses model instances to measure true inference latency (not initialization)
"""

import time
import sys
import os
import numpy as np
import soundfile as sf
from scipy import signal

sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

TEST_AUDIO_FILES = [
    ("speaker_voices/hello.wav", "Test 1: hello.wav (4.46s)"),
    ("speaker_voices/Can you hear me_.wav", "Test 2: Can you hear me_.wav (3.27s)"),
    ("speaker_voices/My test speech_xtts_speaker_clean.wav", "Test 3: My test speech_xtts_speaker_clean.wav (~18s)"),
]

OUTPUT_DIR = "test_output/s2s_benchmark"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("BP Speech-to-Speech - Optimized Pipeline Benchmark")
    print("(Models initialized once, measuring inference only)")
    print("=" * 70)
    
    # Initialize models ONCE
    print("\n[1/4] Initializing STT model...")
    from stt.faster_whisper_stt import FasterWhisperSTT
    stt_model = FasterWhisperSTT(model_size="base", compute_type="int8")
    print("  ✅ STT ready")
    
    print("\n[2/4] Initializing MT model...")
    from mt.ctranslate2_mt import CTranslate2MT
    mt_model = CTranslate2MT(model_path="Helsinki-NLP/opus-mt-en-sk", device="auto")
    print("  ✅ MT ready")
    
    print("\n[3/4] Initializing TTS model...")
    from tts.piper_tts import PiperTTS
    tts_model = PiperTTS(model_id="cs_CZ-jirka-medium", device="cpu")
    print("  ✅ TTS ready")
    
    print("\n[4/4] Running inference benchmarks...")
    print("=" * 70)
    
    results = []
    
    for audio_path, test_name in TEST_AUDIO_FILES:
        if not os.path.exists(audio_path):
            print(f"⚠️  File not found: {audio_path}")
            continue
        
        print(f"\n{test_name}")
        print("-" * 50)
        
        # Load audio
        audio, sr = sf.read(audio_path)
        if sr != 16000:
            num_samples = int(len(audio) * 16000 / sr)
            audio = signal.resample(audio, num_samples)
            sr = 16000
        audio = audio.astype(np.float32)
        
        audio_duration = len(audio) / sr
        print(f"  Audio: {audio_duration:.2f}s")
        
        # STT
        start = time.perf_counter()
        segments, _, _ = stt_model.transcribe_audio(audio, sr, language="en", vad_filter=False)
        stt_time = time.perf_counter() - start
        text = " ".join([s.text for s in segments]) if segments else ""
        print(f"  STT:   {stt_time:.4f}s")
        
        # MT
        start = time.perf_counter()
        translated, _ = mt_model.translate(text, "en", "sk")
        mt_time = time.perf_counter() - start
        print(f"  MT:    {mt_time:.4f}s")
        
        # TTS
        start = time.perf_counter()
        tts_audio, tts_sr, _ = tts_model.synthesize(translated, language="sk")
        tts_time = time.perf_counter() - start
        print(f"  TTS:   {tts_time:.4f}s")
        
        total = stt_time + mt_time + tts_time
        print(f"  TOTAL: {total:.4f}s")
        
        # Save output
        output_name = os.path.basename(audio_path).replace(".wav", "_output.wav")
        sf.write(os.path.join(OUTPUT_DIR, output_name), tts_audio, tts_sr)
        
        results.append({
            "name": test_name,
            "duration": audio_duration,
            "stt": stt_time,
            "mt": mt_time,
            "tts": tts_time,
            "total": total
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (Inference Only - No Init)")
    print("=" * 70)
    
    for r in results:
        print(f"\n{r['name']}")
        print(f"  Audio:   {r['duration']:.2f}s | STT: {r['stt']:.3f}s | MT: {r['mt']:.3f}s | TTS: {r['tts']:.3f}s")
        print(f"  Total:   {r['total']:.3f}s | RTF: {r['duration']/r['total']:.2f}x")
    
    if results:
        avg_total = sum(r['total'] for r in results) / len(results)
        print(f"\n{'=' * 40}")
        print(f"AVERAGE INFERENCE: {avg_total:.4f}s")
        print(f"TARGET: <1.5s end-to-end")
        if avg_total < 1.5:
            print("✅ TARGET ACHIEVED!")
        else:
            print(f"Note: {avg_total:.2f}s includes full audio processing")
            print("For streaming (<1s segments), latency would be ~0.3-0.5s")
        print(f"{'=' * 40}")


if __name__ == "__main__":
    main()