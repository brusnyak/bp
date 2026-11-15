#!/usr/bin/env python3
"""End-to-end streaming pipeline test: STT → MT → TTS, measured end to end.

Tests the full near-real-time route with streaming audio fed in real-time chunks,
measuring latency at each stage and comparing TTS engines (Piper vs OmniVoice vs
XTTS vs X-Voice vs Pocket TTS) including zero-shot voice cloning.

Usage:
  # Start the server first:
  BP_NO_SSL=1 BP_PORT=8013 .venv/bin/python app.py &
  # Then run:
  .venv/bin/python scripts/live/e2e_streaming_test.py \
      --audio processed/testaudio/en_paced.wav --src en --tgt sk \
      --tts piper --clone --ref speaker_voices/me_sk_trhove.m4a \
      --out processed/e2e_streaming_results.json

Metrics measured per segment:
  stt_s     - time from audio chunk end to transcription result
  mt_s      - time from transcription to translation
  tts_s     - time from translation to first audio
  total_s   - end-to-end from chunk end to first audio
  rtf       - TTS real-time factor (synthesis time / audio duration)
  first_word_lag - first audio after source segment end (what listener hears)
"""
from __future__ import annotations

import argparse, asyncio, io, json, sys, time, urllib.request, wave
from pathlib import Path

import numpy as np
import websockets

SR = 16000
CHUNK_S = 0.5  # 500 ms audio chunks, matching production route


def decode(path, seconds=None, offset=0.0):
    """Decode audio to float32 mono at 16 kHz."""
    cmd = ["ffmpeg", "-v", "error", "-ss", str(offset)]
    if seconds:
        cmd += ["-t", str(seconds)]
    cmd += ["-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    return np.frombuffer(
        subprocess.run(cmd, check=True, capture_output=True).stdout, dtype=np.float32
    )


import subprocess  # noqa: E402


def post(base, path, body):
    """POST JSON to the server REST API."""
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def wav_seconds(b):
    with wave.open(io.BytesIO(b)) as w:
        return w.getnframes() / w.getframerate()


class E2EStreamingTest:
    def __init__(self, base_url: str, src: str, tgt: str, tts_engine: str,
                 clone: bool = False, ref_wav: str | None = None):
        self.base = base_url.rstrip("/")
        self.src = src
        self.tgt = tgt
        self.tts_engine = tts_engine
        self.clone = clone
        self.ref_wav = ref_wav
        self.results = {
            "config": {
                "src": src, "tgt": tgt, "tts_engine": tts_engine,
                "clone": clone, "ref_wav": ref_wav,
            },
            "segments": [],
            "summary": {},
        }

    async def run(self, audio: np.ndarray, chunk_s: float = CHUNK_S):
        """Feed audio in real-time chunks through the WebSocket route."""
        chunk_samples = int(chunk_s * SR)
        n_chunks = (len(audio) + chunk_samples - 1) // chunk_samples

        # Initialize session via REST
        status, body = post(self.base, "/initialize", {
            "src_lang": self.src, "tgt_lang": self.tgt,
            "tts_engine": self.tts_engine,
            "voice_clone": self.clone,
            "ref_wav": self.ref_wav,
        })
        if status != 200:
            print(f"Init failed: {status} {body}")
            return

        session_id = body.get("session_id", "default")
        print(f"Session {session_id}: {n_chunks} chunks of {chunk_s}s each")

        uri = f"{self.base.replace('http', 'ws')}/ws/{session_id}"
        async with websockets.connect(uri) as ws:
            # Send initialization config
            await ws.send(json.dumps({
                "type": "init", "src": self.src, "tgt": self.tgt,
                "tts": self.tts_engine, "clone": self.clone,
                "ref_wav": self.ref_wav,
            }))

            segment_idx = 0
            for i in range(0, len(audio), chunk_samples):
                chunk = audio[i:i + chunk_samples]
                if len(chunk) < chunk_samples // 2:
                    break  # skip tiny final chunk

                t_chunk_end = time.perf_counter()
                chunk_end_s = (i + len(chunk)) / SR

                # Send audio chunk (raw f32le bytes)
                await ws.send(chunk.astype(np.float32).tobytes())

                # Collect responses for this chunk
                seg_result = {
                    "segment": segment_idx,
                    "audio_start_s": i / SR,
                    "audio_end_s": chunk_end_s,
                    "stt_s": None, "mt_s": None, "tts_s": None,
                    "total_s": None, "rtf": None,
                    "first_word_lag_s": None,
                    "transcription": "", "translation": "",
                    "tts_audio_s": 0.0,
                }

                # Wait for responses with timeout
                deadline = time.perf_counter() + 30.0
                stt_t = mt_t = tts_t = None
                stt_done = mt_done = False

                while time.perf_counter() < deadline:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    except asyncio.TimeoutError:
                        break

                    data = json.loads(msg)
                    mtype = data.get("type")
                    now = time.perf_counter()

                    if mtype == "transcription_result":
                        stt_t = now
                        seg_result["stt_s"] = round(stt_t - t_chunk_end, 3)
                        seg_result["transcription"] = data.get("text", "")
                        stt_done = True
                        mt_t = now  # MT starts after STT

                    elif mtype == "translation_result":
                        mt_t = time.perf_counter()
                        seg_result["mt_s"] = round(mt_t - stt_t, 3) if stt_t else None
                        seg_result["translation"] = data.get("text", "")
                        mt_done = True
                        tts_t = now  # TTS starts after MT

                    elif mtype == "tts_audio":
                        tts_t = time.perf_counter()
                        seg_result["tts_s"] = round(tts_t - mt_t, 3) if mt_t else None
                        audio_b64 = data.get("audio", "")
                        if audio_b64:
                            import base64
                            audio_bytes = base64.b64decode(audio_b64)
                            seg_result["tts_audio_s"] = wav_seconds(audio_bytes)
                            seg_result["rtf"] = round(
                                seg_result["tts_s"] / seg_result["tts_audio_s"], 3
                            ) if seg_result["tts_audio_s"] > 0 else None
                        seg_result["total_s"] = round(tts_t - t_chunk_end, 3)
                        seg_result["first_word_lag_s"] = seg_result["total_s"]
                        break

                    # Real-time pacing: wait until chunk audio time has elapsed
                    elapsed = now - t_chunk_end
                    chunk_duration = len(chunk) / SR
                    if elapsed < chunk_duration:
                        await asyncio.sleep(chunk_duration - elapsed)

                self.results["segments"].append(seg_result)
                segment_idx += 1

                # Progress
                if segment_idx % 5 == 0:
                    print(f"  {segment_idx}/{n_chunks} segments done")

        self._compute_summary()
        return self.results

    def _compute_summary(self):
        segs = self.results["segments"]
        if not segs:
            return

        valid = [s for s in segs if s["total_s"] is not None]
        if not valid:
            return

        self.results["summary"] = {
            "segments": len(segs),
            "completed": len(valid),
            "stt_s_median": round(
                sorted(s["stt_s"] for s in valid if s["stt_s"])[len(valid) // 2], 3
            ) if any(s["stt_s"] for s in valid) else None,
            "mt_s_median": round(
                sorted(s["mt_s"] for s in valid if s["mt_s"])[len(valid) // 2], 3
            ) if any(s["mt_s"] for s in valid) else None,
            "tts_s_median": round(
                sorted(s["tts_s"] for s in valid if s["tts_s"])[len(valid) // 2], 3
            ) if any(s["tts_s"] for s in valid) else None,
            "total_s_median": round(
                sorted(s["total_s"] for s in valid)[len(valid) // 2], 3
            ),
            "total_s_p90": round(
                sorted(s["total_s"] for s in valid)[int(len(valid) * 0.9)], 3
            ),
            "rtf_median": round(
                sorted(s["rtf"] for s in valid if s["rtf"])[len(valid) // 2], 3
            ) if any(s["rtf"] for s in valid) else None,
            "first_word_lag_median": round(
                sorted(s["first_word_lag_s"] for s in valid)[len(valid) // 2], 3
            ),
        }


def main():
    ap = argparse.ArgumentParser(description="E2E streaming pipeline test")
    ap.add_argument("--audio", required=True, help="Input WAV file")
    ap.add_argument("--src", required=True, help="Source language code")
    ap.add_argument("--tgt", required=True, help="Target language code")
    ap.add_argument("--tts", default="piper", help="TTS engine (piper, omnivoice, xtts, xvoice, pocket)")
    ap.add_argument("--clone", action="store_true", help="Enable voice cloning")
    ap.add_argument("--ref", help="Reference WAV for voice cloning")
    ap.add_argument("--base", default="http://localhost:8000", help="Server base URL")
    ap.add_argument("--chunk", type=float, default=0.5, help="Chunk size in seconds")
    ap.add_argument("--out", default="processed/e2e_streaming_results.json",
                    help="Output JSON file")
    args = ap.parse_args()

    audio = decode(args.audio)
    print(f"Audio: {len(audio)/SR:.1f}s, {args.src}→{args.tgt}, TTS={args.tts}, clone={args.clone}")

    test = E2EStreamingTest(args.base, args.src, args.tgt, args.tts,
                            clone=args.clone, ref_wav=args.ref)
    results = asyncio.run(test.run(audio, chunk_s=args.chunk))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nResults written to {out_path}")
    print(f"Summary: {json.dumps(results['summary'], indent=2)}")


if __name__ == "__main__":
    main()
