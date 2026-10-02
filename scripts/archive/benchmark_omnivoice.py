#!/usr/bin/env python3
"""Isolated CPU benchmark for OmniVoice zero-shot TTS.

Run this only with .venv-omni; it intentionally does not register OmniVoice in
the live application.  It records cold model load, reusable reference-prompt
preparation, and synthesis latency so a subjective A/B can be made against
Piper without pretending an H100 benchmark applies to an M1.

    .venv-omni/bin/python scripts/benchmark_omnivoice.py \
      --reference test/Hello.wav --reference-text 'Hello.' --language sk
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch
from omnivoice import OmniVoice


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, default=ROOT / "test" / "My test speech_xtts_speaker_clean.wav",
                        help="clean reference clip (3-60 seconds; 3-10 seconds is the official sweet spot)")
    parser.add_argument("--reference-text", default=(ROOT / "test" / "My test speech transcript.txt").read_text(encoding="utf-8").strip(),
                        help="exact words spoken in --reference; do not guess for a real benchmark")
    parser.add_argument("--text", default="Dobrý deň. Toto je test prekladu hlasu v reálnom čase.",
                        help="target sentence to synthesize")
    parser.add_argument("--language", default="sk", choices=("en", "sk"))
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--num-step", default=16, type=int,
                        help="iterative decoding steps (official default 32; 16 is its speed setting)")
    parser.add_argument("--name", default="omnivoice_cpu")
    args = parser.parse_args()

    reference = args.reference.expanduser().resolve()
    if not reference.is_file():
        parser.error(f"reference does not exist: {reference}")
    reference_seconds = sf.info(reference).duration
    if not 2 <= reference_seconds <= 60:
        parser.error(f"reference is {reference_seconds:.1f}s; use a clean 3-60 second clip")
    if reference_seconds > 10:
        print("Warning: the reference is longer than the official 3-10 second sweet spot; this is a real-recording stress test, not the ideal clone setup.")

    # CPU is intentional. MPS has known instability reports for OmniVoice and
    # this device currently reports MPS unavailable under the isolated torch.
    device = "cpu"
    dtype = torch.float32
    output_dir = ROOT / "processed" / "omnivoice"
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / f"{args.name}_{args.language}.wav"
    receipt_path = wav_path.with_suffix(".json")

    print(f"Device: {device} | reference: {reference.name} ({reference_seconds:.1f}s)")
    print(f"Loading {args.model}; first use may download the model into the Hugging Face cache.")
    started = time.perf_counter()
    model = OmniVoice.from_pretrained(args.model, device_map=device, dtype=dtype)
    load_seconds = time.perf_counter() - started

    started = time.perf_counter()
    prompt = model.create_voice_clone_prompt(str(reference), args.reference_text)
    prompt_seconds = time.perf_counter() - started

    started = time.perf_counter()
    audio = model.generate(text=args.text, language=args.language, voice_clone_prompt=prompt,
                           num_step=args.num_step)[0]
    synthesis_seconds = time.perf_counter() - started
    sf.write(wav_path, audio, model.sampling_rate)
    audio_seconds = len(audio) / model.sampling_rate

    receipt = {
        "model": args.model, "device": device, "dtype": str(dtype), "reference": str(reference),
        "reference_text": args.reference_text, "target_text": args.text, "language": args.language,
        "sample_rate": model.sampling_rate, "num_step": args.num_step, "output_seconds": round(audio_seconds, 3),
        "latency_seconds": {"cold_load": round(load_seconds, 3), "prompt_prepare": round(prompt_seconds, 3),
                            "synthesis": round(synthesis_seconds, 3),
                            "rtf": round(synthesis_seconds / audio_seconds, 3) if audio_seconds else None},
        "output_wav": str(wav_path),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    print("Judge the resulting WAV blind beside the Piper output; RTF alone is not voice quality.")


if __name__ == "__main__":
    main()
