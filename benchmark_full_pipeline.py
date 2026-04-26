#!/usr/bin/env python3
"""
Full pipeline latency benchmark for BP Speech-to-Speech Translation Project
Measures end-to-end performance of STT -> MT -> TTS components
"""

import time
import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def benchmark_stt():
    """Benchmark Speech-to-Text performance"""
    print("=== Speech-to-Text Benchmark ===")
    
    try:
        from stt.faster_whisper_stt import FasterWhisperSTT
        
        # Initialize model
        print("Initializing STT model (FasterWhisper base)...")
        start_init = time.perf_counter()
        stt_model = FasterWhisperSTT(model_size="base", compute_type="int8")
        init_time = time.perf_counter() - start_init
        print(f"Model initialization time: {init_time:.4f}s")
        
        # We'll simulate STT processing with a dummy audio array
        # In reality, this would process actual audio data
        print("\nSimulating STT processing (using dummy audio)...")
        # Create a 1-second dummy audio signal at 16kHz
        dummy_audio = np.random.randn(16000).astype(np.float32) * 0.1
        
        latencies = []
        print("Testing STT processing:")
        for i in range(3):  # Run 3 iterations
            start = time.perf_counter()
            # This would normally transcribe the audio
            # For benchmarking, we'll just measure the function call overhead
            segments, stt_time, detected_lang = stt_model.transcribe_audio(
                dummy_audio, 
                16000, 
                language="en",
                vad_filter=False
            )
            latency = time.perf_counter() - start
            latencies.append(latency)
            transcribed_text = " ".join([s.text for s in segments]) if segments else ""
            print(f"  Run {i+1}: Processing time: {latency:.4f}s, Text: '{transcribed_text[:50]}...'")
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        
        print(f"\nSTT Performance Summary:")
        print(f"  Average latency: {avg_latency:.4f}s")
        print(f"  Min latency: {min_latency:.4f}s")
        print(f"  Max latency: {max_latency:.4f}s")
        
        return avg_latency
        
    except Exception as e:
        print(f"Error benchmarking STT: {e}")
        import traceback
        traceback.print_exc()
        return None

def benchmark_mt():
    """Benchmark Machine Translation performance"""
    print("\n=== Machine Translation Benchmark ===")
    
    try:
        from mt.ctranslate2_mt import CTranslate2MT
        
        # Initialize model
        print("Initializing MT model (Helsinki-NLP/opus-mt-en-sk)...")
        start_init = time.perf_counter()
        mt_model = CTranslate2MT(model_path="Helsinki-NLP/opus-mt-en-sk", device="auto")
        init_time = time.perf_counter() - start_init
        print(f"Model initialization time: {init_time:.4f}s")
        
        # Test translations with realistic sentences
        test_sentences = [
            "Hello world",
            "The quick brown fox jumps over the lazy dog",
            "Machine translation technology has advanced significantly in recent years",
            "Artificial intelligence is transforming how we communicate across languages",
            "Real-time speech translation enables seamless global communication"
        ]
        
        latencies = []
        print("\nTesting translations:")
        for i, sentence in enumerate(test_sentences, 1):
            start = time.perf_counter()
            translation, latency = mt_model.translate(sentence, "en", "sk")
            latencies.append(latency)
            print(f"  {i}. '{sentence}' -> '{translation}' ({latency:.4f}s)")
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        
        print(f"\nMT Performance Summary:")
        print(f"  Average latency: {avg_latency:.4f}s")
        print(f"  Min latency: {min_latency:.4f}s")
        print(f"  Max latency: {max_latency:.4f}s")
        
        return avg_latency
        
    except Exception as e:
        print(f"Error benchmarking MT: {e}")
        import traceback
        traceback.print_exc()
        return None

def benchmark_tts():
    """Benchmark Text-to-Speech performance"""
    print("\n=== Text-to-Speech Benchmark ===")
    
    try:
        from tts.piper_tts import PiperTTS
        
        # Initialize model
        print("Initializing TTS model (Piper cs_CZ-jirka-medium)...")
        start_init = time.perf_counter()
        tts_model = PiperTTS(model_id="cs_CZ-jirka-medium", device="cpu")  # Use CPU for consistency
        init_time = time.perf_counter() - start_init
        print(f"Model initialization time: {init_time:.4f}s")
        
        # Test syntheses with Slovak translations of the MT test sentences
        test_sentences = [
            "Ahoj světe",
            "Rýchla hnedá líška skáče cez lenivého psa",
            "Technológia prekladu strojového učenia výrazne pokročila v posledných rokoch",
            "Umelá inteligencia mení spôsob komunikácie medzi jazykmi.",
            "Preklad reči v reálnom čase umožňuje bezproblémovú globálnu komunikáciu"
        ]
        
        latencies = []
        print("\nTesting syntheses:")
        for i, sentence in enumerate(test_sentences, 1):
            start = time.perf_counter()
            audio, sample_rate, latency = tts_model.synthesize(sentence, language="sk")
            latencies.append(latency)
            print(f"  {i}. '{sentence}' -> {len(audio)/sample_rate:.2f}s audio ({latency:.4f}s)")
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        
        print(f"\nTTS Performance Summary:")
        print(f"  Average latency: {avg_latency:.4f}s")
        print(f"  Min latency: {min_latency:.4f}s")
        print(f"  Max latency: {max_latency:.4f}s")
        print(f"  Sample rate: {sample_rate} Hz")
        
        return avg_latency
        
    except Exception as e:
        print(f"Error benchmarking TTS: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run full pipeline benchmarks"""
    print("BP Speech-to-Speech Translation Project - Full Pipeline Latency Benchmark")
    print("=" * 80)
    
    stt_avg = benchmark_stt()
    mt_avg = benchmark_mt()
    tts_avg = benchmark_tts()
    
    print("\n" + "=" * 80)
    print("FULL PIPELINE BENCHMARK SUMMARY")
    print("=" * 80)
    
    if stt_avg is not None:
        print(f"Speech-to-Text: {stt_avg:.4f}s avg latency")
    if mt_avg is not None:
        print(f"Machine Translation (EN->SK): {mt_avg:.4f}s avg latency")
    if tts_avg is not None:
        print(f"Text-to-Speech (SK): {tts_avg:.4f}s avg latency")
    
    if all(x is not None for x in [stt_avg, mt_avg, tts_avg]):
        total_pipeline_latency = stt_avg + mt_avg + tts_avg
        print(f"\nEstimated Full Pipeline (STT+MT+TTS): {total_pipeline_latency:.4f}s")
        print(f"Target: <1.5s end-to-end latency")
        if total_pipeline_latency < 1.5:
            print("✅ Pipeline latency target ACHIEVED")
        else:
            print("⚠️  Pipeline latency target NOT YET MET")
            print(f"   Need to reduce latency by {total_pipeline_latency - 1.5:.4f}s")
    
    print("\nBenchmark completed.")

if __name__ == "__main__":
    main()