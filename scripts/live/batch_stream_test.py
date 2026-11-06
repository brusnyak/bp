#!/usr/bin/env python3
"""Batch streaming E2E test: feed a recorded audio file in chunks through the
local WebSocket pipeline, measure per-chunk STT/MT/TTS latency.

Usage:
    .venv/bin/python scripts/live/batch_stream_test.py \
        --audio speaker_voices/me_en_script.m4a \
        --src en --tgt sk --chunks 5 --interval 0.2

Cuts audio into N sequential parts and feeds them in sequence (simulating
real-time capture). Reports per-chunk and aggregate latency from the server's
final_metrics.
"""
import asyncio, json, sys, time, wave
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import librosa
import websockets

SR = 16000
CHUNK_MS = 200  # 200ms WebSocket frames, matching browser capture

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="Input WAV/M4A file")
    ap.add_argument("--src", default="en", help="Source language")
    ap.add_argument("--tgt", default="sk", help="Target language")
    ap.add_argument("--chunks", type=int, default=5, help="Number of sequential parts")
    ap.add_argument("--interval", type=float, default=0.3, help="Seconds between chunk sends")
    ap.add_argument("--slowdown", type=float, default=2.0, help="Send audio slower than real-time (2.0 = half speed)")
    ap.add_argument("--token", default=None, help="JWT token for WebSocket auth")
    args = ap.parse_args()

    # Load and resample audio (librosa handles M4A via audioread)
    audio, sr = librosa.load(args.audio, sr=SR, mono=True)
    audio = audio.astype(np.float32)
    total_s = len(audio) / SR
    part_s = total_s / args.chunks
    print(f"Audio: {total_s:.1f}s, {args.chunks} parts of {part_s:.1f}s each")

    # Connect to WebSocket
    uri = "ws://localhost:8000/ws"
    if args.token:
        uri += f"?token={args.token}"
    results = []

    async with websockets.connect(uri, max_size=None) as ws:
        # Configure session first (server sends models_loading_status in response)
        await ws.send(json.dumps({
            "type": "config_update",
            "source_lang": args.src,
            "target_lang": args.tgt,
            "stt_model_size": "base",
            "tts_model_choice": "piper",
            "vad_enabled": True
        }))
        print(f"Configured {args.src}→{args.tgt}, waiting for models...")

        # Wait for models to load
        while True:
            raw = await ws.recv()
            if isinstance(raw, str):
                msg = json.loads(raw)
                if msg.get("type") == "models_loading_status" and msg.get("fully_loaded"):
                    print("Models loaded")
                    break
                if msg.get("type") == "error":
                    print(f"Error: {msg}")
                    return

        # Start capture
        await ws.send(json.dumps({"type": "start"}))
        print("Capture started")

        # Feed chunks sequentially
        for i in range(args.chunks):
            start = int(i * part_s * SR)
            end = int((i + 1) * part_s * SR)
            if i == args.chunks - 1:
                end = len(audio)
            part = audio[start:end]

            # Send in 200ms frames, slowed down so the server can keep up with ping/pong
            frame_samples = int(SR * CHUNK_MS / 1000)
            frame_duration = CHUNK_MS / 1000.0 * args.slowdown
            for off in range(0, len(part), frame_samples):
                frame = part[off:off + frame_samples]
                await ws.send(frame.astype(np.float32).tobytes())
                await asyncio.sleep(frame_duration)

            # Wait between parts (simulates pause between sentences)
            if i < args.chunks - 1:
                await asyncio.sleep(args.interval)

        # Stop capture
        await ws.send(json.dumps({"type": "stop"}))
        print("Capture stopped")

        # Drain results
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                break
            if isinstance(raw, bytes):
                continue  # WAV audio
            msg = json.loads(raw)
            if msg.get("type") == "final_metrics":
                m = msg["metrics"]
                results.append(m)
                print(f"  Result: stt={m.get('stt_time',0):.2f}s mt={m.get('mt_time',0):.2f}s "
                      f"tts={m.get('tts_time',0):.2f}s onset={m.get('onset_to_first_audio_s',0):.2f}s "
                      f"total={m.get('total_latency',0):.2f}s")
            elif msg.get("type") == "error":
                print(f"  Error: {msg.get('message')}")

    # Summary
    if results:
        n = len(results)
        avg_stt = sum(r.get("stt_time", 0) for r in results) / n
        avg_mt = sum(r.get("mt_time", 0) for r in results) / n
        avg_tts = sum(r.get("tts_time", 0) for r in results) / n
        avg_onset = sum(r.get("onset_to_first_audio_s", 0) for r in results) / n
        avg_total = sum(r.get("total_latency", 0) for r in results) / n
        print(f"\n=== {n} results ===")
        print(f"Avg STT:  {avg_stt:.2f}s")
        print(f"Avg MT:   {avg_mt:.2f}s")
        print(f"Avg TTS:  {avg_tts:.2f}s")
        print(f"Avg onset→audio: {avg_onset:.2f}s")
        print(f"Avg total: {avg_total:.2f}s")
    else:
        print("\nNo results received")

if __name__ == "__main__":
    asyncio.run(main())
