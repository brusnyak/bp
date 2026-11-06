#!/usr/bin/env python3
"""30-minute soak test for BP streaming pipeline queue stability.

Runs continuous streaming for 30 minutes, tracking:
- Phrases produced per minute
- STT/MT/TTS latency trends
- Memory/queue growth (indicated by increasing latency)
- Error rate
"""
import sys
sys.path.insert(0, '.')
from backend.streaming.pipeline import ContinuousStreamingPipeline
from backend.hardware_scan import autoconfigure
import soundfile as sf
import asyncio
import numpy as np
import time

async def test():
    audio, sr = sf.read('/tmp/bp_e_test.wav')
    print(f'Audio: {len(audio)} samples, {sr}Hz, {len(audio)/sr:.1f}s')
    print(f'Loop duration: {len(audio)/sr:.1f}s per cycle')

    config = autoconfigure(force=True)
    print(f'STT: {config.stt.engine}/{config.stt.model_size} on {config.stt.device}')

    pipeline = ContinuousStreamingPipeline(
        speaker_id='soak_test_001',
        source_lang='en',
        target_lang='sk',
        voice='sk_SK-personal-male-medium',
        tts_engine='piper',
        config=config,
    )
    await pipeline.initialize()

    chunk_size = int(sr * 0.3)
    chunk_duration = chunk_size / sr
    total_duration = 30 * 60  # 30 minutes
    cycles_needed = int(total_duration / (len(audio) / sr)) + 1

    print(f'Target: {total_duration}s ({total_duration/60:.0f} min)')
    print(f'Audio loop: {len(audio)/sr:.1f}s, need {cycles_needed} cycles')

    results = []
    start_time = time.perf_counter()
    cycle = 0

    while (time.perf_counter() - start_time) < total_duration:
        cycle_start = time.perf_counter()
        cycle += 1

        # Stream one full audio loop
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            result = await pipeline.process_audio_chunk(chunk, sr)
            if result:
                results.append({
                    'cycle': cycle,
                    'stt_ms': result.stt_ms,
                    'mt_ms': result.mt_ms,
                    'tts_ms': result.tts_ms,
                    'elapsed': time.perf_counter() - start_time,
                })
            await asyncio.sleep(chunk_duration * 0.5)  # 2x speed for soak test

        # Flush between cycles
        await pipeline.flush_final()

        # Report every 5 cycles
        if cycle % 5 == 0:
            elapsed = time.perf_counter() - start_time
            recent = [r for r in results if r['elapsed'] > elapsed - 60]
            if recent:
                avg_stt = sum(r['stt_ms'] for r in recent) / len(recent)
                avg_mt = sum(r['mt_ms'] for r in recent) / len(recent)
                avg_tts = sum(r['tts_ms'] for r in recent) / len(recent)
                print(f"Cycle {cycle}, Elapsed {elapsed:.0f}s ({elapsed/60:.1f}min): "
                      f"phrases={len(recent)}, STT={avg_stt:.0f}ms, MT={avg_mt:.0f}ms, TTS={avg_tts:.0f}ms")

    # Final flush
    await pipeline.flush_final()

    # Summary
    total_elapsed = time.perf_counter() - start_time
    print(f"\n=== SOAK TEST SUMMARY ===")
    print(f"Total elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"Total phrases: {len(results)}")
    print(f"Phrases/min: {len(results) / (total_elapsed / 60):.1f}")

    if results:
        avg_stt = sum(r['stt_ms'] for r in results) / len(results)
        avg_mt = sum(r['mt_ms'] for r in results) / len(results)
        avg_tts = sum(r['tts_ms'] for r in results) / len(results)
        print(f"Avg STT: {avg_stt:.0f}ms, Avg MT: {avg_mt:.0f}ms, Avg TTS: {avg_tts:.0f}ms")

        # Check for latency growth (first vs last quarter)
        quarter = len(results) // 4
        if quarter > 0:
            first_q = results[:quarter]
            last_q = results[-quarter:]
            first_avg = sum(r['stt_ms'] + r['mt_ms'] + r['tts_ms'] for r in first_q) / len(first_q)
            last_avg = sum(r['stt_ms'] + r['mt_ms'] + r['tts_ms'] for r in last_q) / len(last_q)
            growth = (last_avg - first_avg) / first_avg * 100
            print(f"Latency growth (first vs last quarter): {growth:+.1f}%")
            if abs(growth) < 20:
                print("Queue stability: STABLE (latency growth < 20%)")
            else:
                print("Queue stability: UNSTABLE (latency growth > 20%)")

    await pipeline.close()

asyncio.run(test())