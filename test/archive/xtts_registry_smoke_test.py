"""One-off smoke test for the TTS_ENGINES registry's xtts entry (streaming + cloning path),
run directly (python test/xtts_registry_smoke_test.py) — not pytest, matches this project's
existing test/ convention for anything that boots a real model.
"""
import os
import sys
import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.tts.base import TTS_ENGINES

SPEAKER_WAV = "speaker_voices/voice_rec_1m.wav"
TEXT = "Toto je test klonovania hlasu cez novy registry."
OUT_PATH = "test_output/xtts_registry_smoke.wav"


def main():
    os.makedirs("test_output", exist_ok=True)
    print("Loading xtts engine via TTS_ENGINES['xtts']()...")
    engine = TTS_ENGINES["xtts"]()
    print(f"Engine loaded. SUPPORTS_STREAMING={getattr(engine, 'SUPPORTS_STREAMING', False)} "
          f"LANGUAGE_OVERRIDES={getattr(engine, 'LANGUAGE_OVERRIDES', {})} "
          f"REQUIRES_SPEAKER_WAV={getattr(engine, 'REQUIRES_SPEAKER_WAV', False)}")

    lang = engine.LANGUAGE_OVERRIDES.get("sk", "sk")
    print(f"Target language 'sk' -> proxied to '{lang}' (mirrors main.py's dispatch logic)")

    chunks = []
    for chunk in engine.synthesize_stream(text=TEXT, language=lang, speaker_wav_path=SPEAKER_WAV):
        chunks.append(chunk)
    print(f"synthesize_stream yielded {len(chunks)} chunks")

    if not chunks:
        print("FAIL: no audio chunks produced")
        sys.exit(1)

    import numpy as np
    audio = np.concatenate(chunks)
    sf.write(OUT_PATH, audio, engine.sample_rate)
    duration = len(audio) / engine.sample_rate
    print(f"PASS: wrote {duration:.2f}s of audio to {OUT_PATH} (sample_rate={engine.sample_rate})")


if __name__ == "__main__":
    main()
