#!/usr/bin/env python3
"""T047 soak check: repeated interrupt/resume cycles within one session, watching server memory
for growth and confirming no hang. Not a permanent suite addition — kept for reproducibility."""
import asyncio, websockets, json, ssl, time, sys, subprocess
import soundfile as sf
import numpy as np

WS_URL = "wss://localhost:8000/ws"
CERT_PATH = "certs/cert.pem"
AUDIO_SAMPLE_RATE = 16000
CHUNK_SEC = 0.02
N_CYCLES = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SERVER_PID = sys.argv[2] if len(sys.argv) > 2 else None


def load_chunks(path):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != AUDIO_SAMPLE_RATE:
        from scipy.signal import resample
        data = resample(data, int(len(data) * AUDIO_SAMPLE_RATE / sr))
    n = int(AUDIO_SAMPLE_RATE * CHUNK_SEC)
    return [data[i:i + n] for i in range(0, len(data), n)]


def server_rss_kb(pid):
    if not pid:
        return None
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True)
        return int(out.strip())
    except Exception:
        return None


async def main():
    ctx = ssl.create_default_context()
    ctx.load_verify_locations(CERT_PATH)
    ctx.check_hostname = False

    a = load_chunks("test/Hello.wav")
    b = load_chunks("test/Can you hear me_.wav")
    silence = np.zeros(int(AUDIO_SAMPLE_RATE * CHUNK_SEC), dtype=np.float32)

    events = {"transcription_result": 0, "translation_result": 0, "tts_audio": 0, "caption_partial": 0, "error": 0}

    async def recv_loop(ws):
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                if isinstance(msg, str):
                    d = json.loads(msg)
                    t = d.get("type")
                    if t in events:
                        events[t] += 1
                    if t == "error":
                        print("  SERVER ERROR:", d.get("message"))
                elif isinstance(msg, bytes):
                    events["tts_audio"] += 1
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosedOK:
                break
            except Exception as e:
                print("  recv error:", e)
                break

    mem_before = server_rss_kb(SERVER_PID)
    print(f"Server RSS before: {mem_before} KB" if mem_before else "Server PID not tracked, skipping memory check")

    t_start = time.perf_counter()
    async with websockets.connect(WS_URL, ssl=ctx) as ws:
        await ws.send(json.dumps({"type": "config_update", "source_lang": "en", "target_lang": "sk", "tts_model_choice": "piper"}))
        await asyncio.wait_for(ws.recv(), timeout=15)
        await ws.send(json.dumps({"type": "start"}))
        recv_task = asyncio.create_task(recv_loop(ws))

        for cycle in range(N_CYCLES):
            print(f"cycle {cycle+1}/{N_CYCLES}...")
            for c in a:
                await ws.send(c.tobytes())
                await asyncio.sleep(CHUNK_SEC * 0.8)
            # short silence, then immediately interrupt with clip b before cycle's segment
            # necessarily finishes its pipeline (barge-in exercise)
            for _ in range(20):
                await ws.send(silence.tobytes())
                await asyncio.sleep(CHUNK_SEC * 0.8)
            for c in b:
                await ws.send(c.tobytes())
                await asyncio.sleep(CHUNK_SEC * 0.8)
            for _ in range(20):
                await ws.send(silence.tobytes())
                await asyncio.sleep(CHUNK_SEC * 0.8)

        await asyncio.sleep(3)
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(2)
        recv_task.cancel()

    t_total = time.perf_counter() - t_start
    mem_after = server_rss_kb(SERVER_PID)
    print(f"\nCompleted {N_CYCLES} cycles in {t_total:.1f}s, no hang.")
    print(f"Events: {events}")
    if mem_before and mem_after:
        print(f"Server RSS: {mem_before} KB -> {mem_after} KB (delta {mem_after - mem_before:+d} KB)")


if __name__ == "__main__":
    asyncio.run(main())
