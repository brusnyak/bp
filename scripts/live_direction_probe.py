#!/usr/bin/env python3
"""Live (WebSocket) probe for one direction, with WER against a reference.

Sends a clip through the real /ws pipeline the browser uses (VAD -> STT -> MT -> TTS),
prints every transcription/translation pair with timings, and — with --reference-md —
computes WER of the concatenated transcript against the proofread reference column, so
STT error and MT error stop being one unexplained number.

Needs `make run` (server on wss://localhost:8000). Examples:

  venv/bin/python scripts/live_direction_probe.py --source sk --target en \
      --clip speaker_voices/sk_script_reading.m4a \
      --reference-md documentation/reading_script_bilingual.md --reference-col 3
  venv/bin/python scripts/live_direction_probe.py --source en --target sk --clip test/Hello.wav
"""
import argparse
import asyncio
import json
import os
import re
import ssl
import sys
import time

import numpy as np
import websockets

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

WS_URL = "wss://localhost:8000/ws"
CERT_PATH = os.path.join(REPO_ROOT, "certs", "cert.pem")
SR = 16000
CHUNK_SEC = 0.02


def load_chunks(path):
    # librosa, not soundfile: the recordings are .m4a (same loader as
    # scripts/e2e_ensk_new_voice.py).
    import librosa
    data, sr = librosa.load(path, sr=SR, mono=True)
    data = np.asarray(data, dtype=np.float32)
    n = int(SR * CHUNK_SEC)
    return [data[i:i + n] for i in range(0, len(data), n)], len(data) / SR


def reference_from_md(path, col):
    """Pull one proofread column out of a markdown table (header/separator skipped)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip().startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < col or cells[0] in ("#", ""):
                continue
            if set(cells[0]) <= set("- "):
                continue
            rows.append(cells[col - 1])
    return " ".join(rows)


def normalize(text):
    text = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


async def run(args):
    ctx = ssl.create_default_context()
    ctx.load_verify_locations(CERT_PATH)
    ctx.check_hostname = False

    chunks, audio_s = load_chunks(args.clip)
    transcripts, translations = [], []
    tts_bytes = 0

    async def recv_loop(ws):
        nonlocal tts_bytes
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break
            if isinstance(msg, bytes):
                tts_bytes += len(msg)
                continue
            try:
                d = json.loads(msg)
            except ValueError:
                continue
            if d.get("type") == "transcription_result":
                transcripts.append((time.perf_counter(), (d.get("transcribed") or "").strip()))
            elif d.get("type") == "translation_result":
                translations.append((time.perf_counter(), (d.get("translated") or "").strip()))

    async with websockets.connect(WS_URL, ssl=ctx, max_size=None) as ws:
        await ws.send(json.dumps({"type": "config_update", "source_lang": args.source,
                                  "target_lang": args.target, "tts_model_choice": args.tts}))
        await asyncio.wait_for(ws.recv(), timeout=60)
        await ws.send(json.dumps({"type": "start"}))
        recv_task = asyncio.create_task(recv_loop(ws))

        t_start = time.perf_counter()
        for c in chunks:
            await ws.send(c.tobytes())
            await asyncio.sleep(CHUNK_SEC)
        silence = np.zeros(int(SR * CHUNK_SEC), dtype=np.float32)
        for _ in range(30):
            await ws.send(silence.tobytes())
            await asyncio.sleep(CHUNK_SEC)
        await asyncio.sleep(args.settle)
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(3)
        recv_task.cancel()

    print(f"\n--- {args.source}->{args.target}, clip {os.path.basename(args.clip)} "
          f"({audio_s:.1f}s), streamed in {time.perf_counter() - t_start:.1f}s, "
          f"TTS bytes {tts_bytes} ---")
    for (t1, tr), (t2, tl) in zip(transcripts, translations):
        print(f"[+{t1 - t_start:6.2f}s] {args.source}: {tr}")
        print(f"[+{t2 - t_start:6.2f}s] {args.target}: {tl}")

    hyp = normalize(" ".join(t for _, t in transcripts))
    print(f"\ntranscript: {len(transcripts)} segments, {len(hyp)} chars normalized")
    if args.reference_md:
        ref = normalize(reference_from_md(args.reference_md, args.reference_col))
        try:
            import jiwer
            print(f"reference:  {len(ref)} chars  ->  WER {jiwer.wer(ref, hyp):.4f}")
        except ImportError:
            print("jiwer not installed; skipping WER")
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump({"source": args.source, "target": args.target, "clip": args.clip,
                       "transcripts": [t for _, t in transcripts],
                       "translations": [t for _, t in translations],
                       "tts_bytes": tts_bytes}, f, ensure_ascii=False, indent=2)
        print(f"wrote {args.dump}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="sk")
    ap.add_argument("--target", default="en")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--tts", default="piper")
    ap.add_argument("--reference-md", default=None)
    ap.add_argument("--reference-col", type=int, default=3)
    ap.add_argument("--settle", type=float, default=8.0)
    ap.add_argument("--dump", default=None)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
