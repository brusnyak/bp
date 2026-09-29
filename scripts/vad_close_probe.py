"""Micro-probe: 2s tone + trailing digital zeros through the WS /ws path.

Verifies the VAD trailing-silence fix: all-zero tail chunks must flow into
the VAD frame loop so SILENCE_TIMEOUT fires live (no teardown needed).

Usage (needs `make run` in another terminal):
    .venv/bin/python scripts/vad_close_probe.py [--seconds 2] [--silence 3]

Pass criteria: server log shows "Final speech segment ENDED" BEFORE any
'stop' command / disconnect, i.e. input_to_stt_latency < silence+2s.
Fails if the segment only flushes at teardown (latency ~= total session).
"""
import argparse
import asyncio
import json
import ssl
import sys

import numpy as np
import websockets

SAMPLE_RATE = 16000
CHUNK = 960  # ~60ms at 16kHz


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--silence", type=float, default=3.0)
    ap.add_argument("--url", default="wss://localhost:8000/ws")
    args = ap.parse_args()

    tone = (0.5 * np.sin(2 * np.pi * 440.0 * np.arange(int(SAMPLE_RATE * args.seconds)) / SAMPLE_RATE)).astype(np.float32)
    zeros = np.zeros(int(SAMPLE_RATE * args.silence), dtype=np.float32)

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    print(f"connecting {args.url} ...")
    async with websockets.connect(args.url, ssl=ctx, max_size=None) as ws:
        await ws.send(json.dumps({
            "type": "config",
            "source_lang": "en", "target_lang": "sk",
            "tts_model_choice": "piper_generic_sk",
            "vad_enabled": True,
        }))
        await ws.send(json.dumps({"type": "start"}))
        # drain configs/start acks briefly
        try:
            async with asyncio.timeout(3):
                async for m in ws:
                    d = json.loads(m)
                    if d.get("type") in ("status", "models_loading_status"):
                        continue
                    break
        except (TimeoutError, StopAsyncIteration):
            pass

        async def sender():
            for i in range(0, len(tone), CHUNK):
                await ws.send(tone[i:i + CHUNK].tobytes())
                await asyncio.sleep(0.02)
            print(f"tone sent ({args.seconds}s); now {args.silence}s digital silence ...")
            for i in range(0, len(zeros), CHUNK):
                await ws.send(zeros[i:i + CHUNK].tobytes())
                await asyncio.sleep(0.02)
            print("silence sent; waiting for translation (should arrive WITHOUT stop)...")

        got_translation = False

        async def receiver():
            nonlocal got_translation
            async for m in ws:
                try:
                    d = json.loads(m)
                except Exception:
                    continue
                t = d.get("type")
                if t == "translation":
                    lat = d.get("input_to_stt_latency", "?")
                    print(f"GOT translation live: {str(d)[:220]} input_to_stt_latency={lat}")
                    got_translation = True
                    return
                if t == "final_metrics":
                    print(f"METRICS: {d}")

        send_task = asyncio.create_task(sender())
        try:
            await asyncio.wait_for(receiver(), timeout=args.seconds + args.silence + 25)
        except asyncio.TimeoutError:
            print("TIMEOUT: no live translation — trailing close still broken (or server slow).")
        await send_task
        try:
            await ws.send(json.dumps({"type": "stop"}))
        except Exception:
            pass
        print("PASS" if got_translation else "FAIL")
        return 0 if got_translation else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
