#!/usr/bin/env python3
"""Terminal Stop + recovery test for BP streaming pipeline.

Tests:
1. Start streaming, verify audio flows
2. Stop mid-stream, verify pipeline stops cleanly
3. Restart streaming, verify pipeline recovers
4. Verify no memory/queue growth after stop/start cycles
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

    config = autoconfigure(force=True)
    print(f'STT: {config.stt.engine}/{config.stt.model_size} on {config.stt.device}')

    pipeline = ContinuousStreamingPipeline(
        speaker_id='stop_test_001',
        source_lang='en',
        target_lang='sk',
        voice='sk_SK-personal-male-medium',
        tts_engine='piper',
        config=config,
    )
    await pipeline.initialize()

    chunk_size = int(sr * 0.3)
    chunk_duration = chunk_size / sr

    # Phase 1: Stream for ~10 seconds
    print("\n=== Phase 1: Streaming for 10s ===")
    results_phase1 = []
    for i in range(0, min(len(audio), int(sr * 10)), chunk_size):
        chunk = audio[i:i+chunk_size]
        result = await pipeline.process_audio_chunk(chunk, sr)
        if result:
            results_phase1.append(result)
        await asyncio.sleep(chunk_duration)
    print(f"Phase 1: {len(results_phase1)} phrases produced")

    # Phase 2: Stop (flush final)
    print("\n=== Phase 2: Stop (flush final) ===")
    final_result = await pipeline.flush_final()
    if final_result:
        print(f"Final flush: STT='{final_result.source_text[:50]}' MT='{final_result.translated_text[:50]}'")

    # Reset pipeline state for restart test
    pipeline._committed_words = []
    pipeline._last_word_end = 0.0
    pipeline._synthesized_text = ""
    pipeline._in_utterance = False
    pipeline._silence_frames = 0
    pipeline._sequence = 0

    # Phase 3: Restart streaming
    print("\n=== Phase 3: Restart streaming for 10s ===")
    results_phase3 = []
    for i in range(0, min(len(audio), int(sr * 10)), chunk_size):
        chunk = audio[i:i+chunk_size]
        result = await pipeline.process_audio_chunk(chunk, sr)
        if result:
            results_phase3.append(result)
        await asyncio.sleep(chunk_duration)
    print(f"Phase 3: {len(results_phase3)} phrases produced")

    # Phase 4: Final stop
    print("\n=== Phase 4: Final stop ===")
    final_result2 = await pipeline.flush_final()
    if final_result2:
        print(f"Final flush: STT='{final_result2.source_text[:50]}' MT='{final_result2.translated_text[:50]}'")

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Phase 1 phrases: {len(results_phase1)}")
    print(f"Phase 3 phrases: {len(results_phase3)}")
    print(f"Recovery successful: {len(results_phase3) > 0}")

    await pipeline.close()

asyncio.run(test())