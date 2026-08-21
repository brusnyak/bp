"""Smoke test for the 'piper_personal' TTS_ENGINES registry entry (the user's own
fine-tuned voice). Run directly (python test/piper_personal_smoke_test.py), matching
this project's existing test/ convention for anything that boots a real model.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.tts.base import TTS_ENGINES

TEXT = "This is a short test sentence synthesized by my fine-tuned Piper voice, wired through the new engine registry."
OUT_PATH = "test_output/piper_personal_registry_smoke.wav"


def main():
    os.makedirs("test_output", exist_ok=True)
    print("Loading engine via TTS_ENGINES['piper_personal']()...")
    engine = TTS_ENGINES["piper_personal"]()
    print(f"Engine loaded. SUPPORTS_CLONING={getattr(engine, 'SUPPORTS_CLONING', None)}")

    wav, sr, latency = engine.synthesize(TEXT, language="en")
    duration = len(wav) / sr
    rtf = latency / duration if duration else float("nan")

    import soundfile as sf
    sf.write(OUT_PATH, wav, sr)

    print(f"PASS: synthesized {duration:.2f}s of audio in {latency:.4f}s (RTF {rtf:.4f}), sr={sr}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
