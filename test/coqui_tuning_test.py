
import time
import torch
import logging
import numpy as np
from backend.tts.coqui_tts import CoquiTTS
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

SPEAKER_WAV = Path("test/Hello.wav")

def test_tuning():
    print("\n" + "="*80)
    print("COQUI TTS TUNING BENCHMARK (Threads & Speed)")
    print("="*80)
    
    # Test parameters
    thread_counts = [1, 2, 4, 8]
    speeds = [1.0, 1.2, 1.5]
    
    results = []
    
    # Initialize model once (warmup included)
    print(">>> Initializing CoquiTTS...", flush=True)
    tts = CoquiTTS(device="cpu", enable_warmup=True)
    
    text = "Hello, this is a test of the tuning benchmark. We are testing different thread counts and speed settings."
    
    for threads in thread_counts:
        print(f"\n>>> Testing with {threads} threads...", flush=True)
        torch.set_num_threads(threads)
        
        for speed in speeds:
            print(f"  > Speed: {speed}x", end="", flush=True)
            
            start_time = time.perf_counter()
            try:
                # Use streaming to measure time-to-first-chunk AND total time
                stream = tts.synthesize_stream(
                    text=text,
                    language="en",
                    speaker_wav_path=str(SPEAKER_WAV),
                    speed=speed,
                    use_cache=True
                )
                
                first_chunk_time = 0
                chunk_count = 0
                
                for i, _ in enumerate(stream):
                    if i == 0:
                        first_chunk_time = time.perf_counter() - start_time
                    chunk_count += 1
                
                total_time = time.perf_counter() - start_time
                
                print(f" -> First Chunk: {first_chunk_time:.2f}s | Total: {total_time:.2f}s", flush=True)
                
                results.append({
                    "threads": threads,
                    "speed": speed,
                    "first_chunk": first_chunk_time,
                    "total_time": total_time
                })
                
            except Exception as e:
                print(f" -> FAILED: {e}", flush=True)

    # Print Summary
    print("\n" + "="*80)
    print("TUNING RESULTS SUMMARY")
    print("="*80)
    print(f"{'Threads':<10} {'Speed':<10} {'First Chunk (s)':<20} {'Total Time (s)':<20}")
    print("-" * 60)
    
    best_latency = float('inf')
    best_config = None
    
    for r in results:
        print(f"{r['threads']:<10} {r['speed']:<10} {r['first_chunk']:<20.2f} {r['total_time']:<20.2f}")
        if r['first_chunk'] < best_latency:
            best_latency = r['first_chunk']
            best_config = r
            
    print("-" * 60)
    if best_config:
        print(f"🏆 BEST CONFIGURATION: {best_config['threads']} threads, {best_config['speed']}x speed (Latency: {best_config['first_chunk']:.2f}s)")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_tuning()
