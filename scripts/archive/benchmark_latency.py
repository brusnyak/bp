#!/usr/bin/env python3
"""
Latency benchmark script for BP Speech-to-Speech Translation Project
Measures performance of MT and TTS components
"""

import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def benchmark_mt():
    """Benchmark Machine Translation performance"""
    print("=== Machine Translation Benchmark ===")
    
    try:
        from mt.ctranslate2_mt import CTranslate2MT
        
        # Initialize model
        print("Initializing MT model (Helsinki-NLP/opus-mt-en-sk)...")
        start_init = time.perf_counter()
        mt_model = CTranslate2MT(model_path="Helsinki-NLP/opus-mt-en-sk", device="auto")
        init_time = time.perf_counter() - start_init
        print(f"Model initialization time: {init_time:.4f}s")
        
        # Test translations
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
        
        # Test syntheses
        test_sentences = [
            "Ahoj světe",
            "Rýchla hnedá líška skáče cez lenivého psa",
            "Technológia prekladu strojového učenia výrazne pokročila v posledných rokoch",
            "Umelá inteligencia mení spôsob, akým komunikujeme medzi jazykmi",
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
        return None

def main():
    """Run all benchmarks"""
    print("BP Speech-to-Speech Translation Project - Latency Benchmark")
    print("=" * 60)
    
    mt_avg = benchmark_mt()
    tts_avg = benchmark_tts()
    
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    if mt_avg is not None:
        print(f"Machine Translation (EN->SK): {mt_avg:.4f}s avg latency")
    if tts_avg is not None:
        print(f"Text-to-Speech (SK): {tts_avg:.4f}s avg latency")
    
    if mt_avg is not None and tts_avg is not None:
        pipeline_latency = mt_avg + tts_avg  # Simplified pipeline
        print(f"Estimated MT+TTS pipeline: {pipeline_latency:.4f}s")
        print(f"Target: <1.5s end-to-end latency")
        if pipeline_latency < 1.5:
            print("✅ Pipeline latency target ACHIEVED")
        else:
            print("⚠️  Pipeline latency target NOT YET MET")
    
    print("\nBenchmark completed.")

if __name__ == "__main__":
    main()