#!/usr/bin/env python3
"""T043/T046 ad-hoc verification: does a new utterance cancel a still-in-flight pipeline task
instead of both running to completion? Not a permanent test-suite addition — a one-off probe,
kept for reproducibility. Mirrors the wire protocol used by test/concurrent_performance_test.py."""
import asyncio, websockets, json, ssl, time, soundfile as sf, numpy as np

WS_URL = "wss://localhost:8000/ws"
CERT_PATH = "certs/cert.pem"
AUDIO_SAMPLE_RATE = 16000
CHUNK_SEC = 0.02


def load_chunks(path):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != AUDIO_SAMPLE_RATE:
        from scipy.signal import resample
        data = resample(data, int(len(data) * AUDIO_SAMPLE_RATE / sr))
    n = int(AUDIO_SAMPLE_RATE * CHUNK_SEC)
    return [data[i:i + n] for i in range(0, len(data), n)]


async def main():
    ctx = ssl.create_default_context()
    ctx.load_verify_locations(CERT_PATH)
    ctx.check_hostname = False

    events = []

    async def recv_loop(ws):
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                t = time.perf_counter()
                if isinstance(msg, str):
                    d = json.loads(msg)
                    if d.get("type") in ("transcription_result", "translation_result"):
                        events.append((t, d.get("type"), d.get("transcribed") or d.get("translated")))
                    elif d.get("type") == "caption_partial":
                        events.append((t, "caption_partial", d.get("text")))
                elif isinstance(msg, bytes):
                    events.append((t, "tts_audio", len(msg)))
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosedOK:
                break
            except Exception as e:
                events.append((time.perf_counter(), "recv_error", str(e)))
                break

    async with websockets.connect(WS_URL, ssl=ctx) as ws:
        await ws.send(json.dumps({"type": "config_update", "source_lang": "en", "target_lang": "sk", "tts_model_choice": "piper"}))
        await asyncio.wait_for(ws.recv(), timeout=10)
        await ws.send(json.dumps({"type": "start"}))

        recv_task = asyncio.create_task(recv_loop(ws))

        # Utterance 1: the longer clip, so its pipeline (STT->MT->TTS) is very likely still
        # running when utterance 2 starts.
        u1 = load_chunks("test/My test speech_xtts_speaker_clean.wav")
        t_u1_start = time.perf_counter()
        for c in u1:
            await ws.send(c.tobytes())
            await asyncio.sleep(CHUNK_SEC * 0.8)

        # Trailing silence long enough to cross SILENCE_TIMEOUT (0.3s) and fire task 1.
        silence_frame = np.zeros(int(AUDIO_SAMPLE_RATE * CHUNK_SEC), dtype=np.float32)
        for _ in range(30):  # ~0.48s of silence at 0.8x pacing
            await ws.send(silence_frame.tobytes())
            await asyncio.sleep(CHUNK_SEC * 0.8)
        t_task1_should_have_fired = time.perf_counter()
        print(f"[t={t_task1_should_have_fired - t_u1_start:.2f}s] silence sent, task1 should be running now")

        # Immediately start utterance 2 (short clip) — this should cancel task1's tail end.
        u2 = load_chunks("test/Hello.wav")
        t_u2_start = time.perf_counter()
        print(f"[t={t_u2_start - t_u1_start:.2f}s] starting utterance 2 (interrupt)")
        for c in u2:
            await ws.send(c.tobytes())
            await asyncio.sleep(CHUNK_SEC * 0.8)
        for _ in range(30):
            await ws.send(silence_frame.tobytes())
            await asyncio.sleep(CHUNK_SEC * 0.8)

        await asyncio.sleep(6)  # let everything settle
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(2)
        recv_task.cancel()

    print("\n--- Events received (relative to utterance-1 start) ---")
    for t, typ, data in events:
        print(f"[t={t - t_u1_start:6.2f}s] {typ}: {data}")


if __name__ == "__main__":
    asyncio.run(main())
