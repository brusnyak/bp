"""Kaggle T4 kernel: full STT→MT→TTS streaming cascade under continuous speech.

Runs on Tesla T4. Measures p50/p95 speech-end→first-audio latency for the
complete route, with SK→EN as the primary direction (the known bottleneck).

Usage (pushed to Kaggle):
  kaggle kernels push -p scripts/gpu_bench/kaggle_full_route --accelerator NvidiaTeslaT4 -t 1500

Outputs (saved to /tmp/kaggle_full_route/):
  route_report.json  — p50/p95 first-audio, stage timings, queue stats
  audio_chunks/      — saved TTS WAV chunks for human grading
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

# Kaggle environment: repo code is extracted to bp-code/
REPO_ROOT = Path("/tmp/bp-code") if Path("/tmp/bp-code").exists() else Path(".")
sys.path.insert(0, str(REPO_ROOT))

from backend.hardware_scan import autoconfigure  # noqa: E402
from backend.streaming.pipeline import StreamingPipeline  # noqa: E402


OUTPUT_DIR = Path("/tmp/kaggle_full_route")
AUDIO_CHUNKS_DIR = OUTPUT_DIR / "audio_chunks"


def _save_wav_chunk(audio_bytes: bytes, path: Path) -> None:
    path.write_bytes(audio_bytes)


def _wav_duration(audio_bytes: bytes, sample_rate: int = 16000) -> float:
    buf = io.BytesIO(audio_bytes)
    with wave.open(buf, "rb") as w:
        return w.getnframes() / w.getframerate()


async def run_route_async(source_lang: str, target_lang: str, audio_path: str,
                          duration_s: int = 90, voice: str = ""):
    """Run the full streaming pipeline on continuous audio (single event loop).

    Returns a report dict with p50/p95 latency metrics.
    """
    config = autoconfigure(force=True)
    pipeline = StreamingPipeline(
        speaker_id="kaggle-speaker-1",
        source_lang=source_lang,
        target_lang=target_lang,
        voice=voice,
        tts_engine=config.tts.engine,
        config=config,
    )

    # Pipeline.initialize() loads STT/MT/TTS models — the slow part (~10-30s)
    await pipeline.initialize()

    # Load and decode audio to 16kHz mono float32
    import librosa  # noqa: E402
    wav, sr = librosa.load(audio_path, sr=16000, mono=True)
    wav = np.asarray(wav, dtype=np.float32)

    # Process in 20ms VAD frames, paced at 1x real time
    frame_samples = int(16000 * 0.02)
    results = []
    t_wall_start = time.perf_counter()
    frames_sent = 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(0, len(wav), frame_samples):
        if time.perf_counter() - t_wall_start > duration_s:
            break
        frame = wav[i:i + frame_samples]
        if len(frame) < frame_samples:
            break

        result = await pipeline.process_audio_chunk(frame, 16000)
        frames_sent += 1

        if result:
            results.append({
                "t": round(time.perf_counter() - t_wall_start, 3),
                "speaker_id": result.speaker_id,
                "source_text": result.source_text,
                "translated_text": result.translated_text,
                "target_lang": result.target_lang,
                "sequence": result.sequence,
                "stt_ms": result.stt_ms,
                "mt_ms": result.mt_ms,
                "tts_ms": result.tts_ms,
                "audio_s": round(_wav_duration(result.audio, result.sample_rate), 3),
            })
            _save_wav_chunk(result.audio, AUDIO_CHUNKS_DIR / f"chunk_{result.sequence:04d}.wav")

        # Pacing: wait until the frame's real-time slot is due
        due = t_wall_start + (i + frame_samples) / 16000
        sleep_for = max(0.0, due - time.perf_counter())
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

    # Flush final fragment
    final = await pipeline.flush_final()
    if final:
        results.append({
            "t": round(time.perf_counter() - t_wall_start, 3),
            "speaker_id": final.speaker_id,
            "source_text": final.source_text,
            "translated_text": final.translated_text,
            "target_lang": final.target_lang,
            "sequence": final.sequence,
            "stt_ms": final.stt_ms,
            "mt_ms": final.mt_ms,
            "tts_ms": final.tts_ms,
            "audio_s": round(_wav_duration(final.audio, final.sample_rate), 3),
            "final": True,
        })
        _save_wav_chunk(final.audio, AUDIO_CHUNKS_DIR / f"chunk_{final.sequence:04d}.wav")

    await pipeline.close()

    # Compute metrics
    first_audio_times = [r["t"] for r in results]
    if first_audio_times:
        first_audio_times.sort()
        p50 = first_audio_times[len(first_audio_times) // 2]
        p95 = first_audio_times[int(len(first_audio_times) * 0.95)]
    else:
        p50 = p95 = None

    report = {
        "direction": f"{source_lang}->{target_lang}",
        "hardware": {
            "gpu": config.hardware.gpu,
            "gpu_name": config.hardware.gpu_name,
            "vram_gb": config.hardware.vram_gb,
            "cpu_cores": config.hardware.cpu_cores,
            "ram_gb": config.hardware.ram_gb,
        },
        "config": {
            "stt": f"{config.stt.model_size}/{config.stt.device}/{config.stt.compute_type}",
            "mt": f"{config.mt.engine}/{config.mt.device}/{config.mt.compute_type}",
            "tts": f"{config.tts.engine}/{config.tts.device}",
        },
        "frames_sent": frames_sent,
        "results_emitted": len(results),
        "first_audio_p50_s": p50,
        "first_audio_p95_s": p95,
        "first_audio_max_s": max(first_audio_times) if first_audio_times else None,
        "wall_s": round(time.perf_counter() - t_wall_start, 2),
        "results": results,
    }
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="sk")
    parser.add_argument("--target", default="en")
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    parser.add_argument("--duration", type=int, default=90)
    parser.add_argument("--voice", default="")
    parser.add_argument("--out", default=str(OUTPUT_DIR / "route_report.json"))
    args = parser.parse_args()

    report = asyncio.run(run_route_async(args.source, args.target, args.audio,
                                         duration_s=args.duration, voice=args.voice))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "direction": report["direction"],
        "results_emitted": report["results_emitted"],
        "first_audio_p50_s": report["first_audio_p50_s"],
        "first_audio_p95_s": report["first_audio_p95_s"],
        "wall_s": report["wall_s"],
    }, indent=2))