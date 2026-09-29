#!/usr/bin/env python3
"""Run one recorded clip through the local EN<->SK pipeline for a live demo.

This is deliberately an *offline* command: it proves the same STT -> MT -> TTS
stages used by the WebSocket app without requiring a browser session or login.
It accepts files uploaded by Voice Lab (normally in ``speaker_voices/``) and
writes a JSON receipt plus the translated WAV under ignored ``processed/``.

Example:
    .venv/bin/python scripts/bp.py demo-audio speaker_voices/en_script_reading.m4a \
        --source en --target sk --max-seconds 12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]


def _segment_text(segments: list[object]) -> str:
    return " ".join(
        (segment.get("text", "") if isinstance(segment, dict) else getattr(segment, "text", ""))
        for segment in segments
    ).strip()


def _voice_model(source: str, target: str, voice: str) -> str:
    if voice == "generic":
        return "sk_SK-lili-medium" if target == "sk" else "en_US-ryan-medium"
    # The fixed personal Piper models are separate per target language. This is
    # not zero-shot cloning; it is the real current default behaviour.
    return "sk_SK-personal-male-medium" if target == "sk" else "en_US-personal-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, help="audio file, e.g. speaker_voices/your_clip.m4a")
    parser.add_argument("--source", required=True, choices=("en", "sk"), help="spoken input language")
    parser.add_argument("--target", required=True, choices=("en", "sk"), help="translated output language")
    parser.add_argument("--voice", default="personal", choices=("personal", "generic"),
                        help="fixed Piper target voice (default: personal)")
    parser.add_argument("--stt-model", choices=("base", "small-sk"),
                        help="default: base for EN; small-sk for SK")
    parser.add_argument("--beam-size", default=1, choices=(1, 2, 5), type=int,
                        help="Whisper decoding beam; 1 is the live-speed candidate")
    parser.add_argument("--max-seconds", type=float,
                        help="use only this many seconds from the start; useful for a fast rehearsal")
    parser.add_argument("--name", help="output name; default comes from input filename")
    args = parser.parse_args()

    if args.source == args.target:
        parser.error("--source and --target must differ")
    audio_path = args.audio.expanduser().resolve()
    if not audio_path.is_file():
        parser.error(f"audio does not exist: {audio_path}")

    sys.path.insert(0, str(REPO_ROOT))
    from backend.mt.ctranslate2_mt import CTranslate2MT
    from backend.stt.faster_whisper_stt import FasterWhisperSTT
    from backend.tts.piper_tts import PiperTTS

    wav, sample_rate = librosa.load(audio_path, sr=16_000, mono=True)
    wav = np.asarray(wav, dtype=np.float32)
    if args.max_seconds:
        if args.max_seconds <= 0:
            parser.error("--max-seconds must be positive")
        wav = wav[: round(args.max_seconds * sample_rate)]
    if not len(wav):
        parser.error("input contains no decodable audio")

    stt_model = args.stt_model or ("base" if args.source == "en" else "small-sk")
    model_id = _voice_model(args.source, args.target, args.voice)
    out_name = args.name or audio_path.stem
    out_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in out_name)
    out_dir = REPO_ROOT / "processed" / "demo_audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_out = out_dir / f"{out_name}_{args.source}-{args.target}_{args.voice}.wav"
    json_out = wav_out.with_suffix(".json")

    started = time.perf_counter()
    print(f"Input: {audio_path.name} | {len(wav) / sample_rate:.1f}s | {args.source.upper()} -> {args.target.upper()}")
    print(f"STT: whisper-{stt_model}, beam={args.beam_size}; TTS: {model_id}")
    stt = FasterWhisperSTT(stt_model, beam_size=args.beam_size)
    segments, stt_s, detected = stt.transcribe_audio(wav, sample_rate, language=args.source)
    transcript = _segment_text(segments)
    if not transcript:
        raise RuntimeError("STT returned no speech. Choose a clip with clear speech.")

    mt = CTranslate2MT(f"Helsinki-NLP/opus-mt-{args.source}-{args.target}")
    translation, mt_s = mt.translate(transcript, args.source, args.target)
    tts = PiperTTS(model_id=model_id)
    output, output_sr, tts_s = tts.synthesize(translation, language=args.target)
    sf.write(wav_out, output, output_sr)
    total_s = time.perf_counter() - started

    receipt = {
        "input": str(audio_path), "audio_seconds": round(len(wav) / sample_rate, 3),
        "direction": f"{args.source}->{args.target}", "stt_model": stt_model,
        "beam_size": args.beam_size, "detected_language": detected, "voice": args.voice,
        "tts_model": model_id, "transcript": transcript, "translation": translation,
        "latency_seconds": {"stt": round(stt_s, 4), "mt": round(mt_s, 4),
                            "tts": round(tts_s, 4), "wall_including_load": round(total_s, 4)},
        "output_wav": str(wav_out),
    }
    json_out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"STT: {transcript}")
    print(f"MT:  {translation}")
    print("Latency: " + ", ".join(f"{key}={value:.3f}s" for key, value in receipt["latency_seconds"].items()))
    print(f"Output: {wav_out}")
    print(f"Receipt: {json_out}")


if __name__ == "__main__":
    main()
