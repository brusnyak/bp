#!/usr/bin/env python3
"""Full EN→SK pipeline on the NEW voice, timed per stage, artifacts saved.

STT (faster-whisper base, EN clip) → MT (Opus-MT en-sk) → TTS (Hybrid clone from
the new EN recording). Out: processed/e2e_ensk.json + sk output wav.
Run: venv/bin/python scripts/e2e_ensk_new_voice.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(REPO_ROOT, "processed", "e2e_ensk.json")
OUT_WAV = os.path.join(REPO_ROOT, "processed", "e2e_ensk_sk.wav")


def main():
    import librosa
    import numpy as np
    import soundfile as sf

    from backend.stt.faster_whisper_stt import FasterWhisperSTT
    from backend.mt.ctranslate2_mt import CTranslate2MT
    from backend.tts.base import TTS_ENGINES

    t_all = time.perf_counter()
    rep = {}

    wav, sr = librosa.load(os.path.join(REPO_ROOT, "speaker_voices", "en_script_reading.m4a"),
                           sr=16000, mono=True)
    stt = FasterWhisperSTT(model_size="base")
    segs, stt_t, lang = stt.transcribe_audio(np.asarray(wav, dtype=np.float32), sr, language="en")
    transcript = " ".join(s.text if hasattr(s, "text") else s["text"] for s in segs)
    rep["stt"] = {"time_s": round(stt_t, 2), "lang": lang, "chars": len(transcript)}
    print(f"STT: {stt_t:.1f}s, {len(transcript)} chars")

    mt = CTranslate2MT("Helsinki-NLP/opus-mt-en-sk")
    t0 = time.perf_counter()
    translated, _ = mt.translate(transcript, "en", "sk")
    mt_t = time.perf_counter() - t0
    rep["mt"] = {"time_s": round(mt_t, 2), "chars": len(translated)}
    print(f"MT: {mt_t:.1f}s, {len(translated)} chars")

    hybrid = TTS_ENGINES["hybrid"]()
    ref = os.path.join(REPO_ROOT, "speaker_voices", "en_script_reading.m4a")
    # Clone the first 600 chars (full 1300-char clip in one go risks long-tail synthesis;
    # chunking is T045's production behavior — this measures the same path).
    sample = translated[:600]
    out_wav, out_sr, tts_t = hybrid.synthesize(sample, language="sk", speaker_wav_path=ref)
    sf.write(OUT_WAV, out_wav, out_sr)
    rep["tts"] = {"time_s": round(tts_t, 2), "engine": "hybrid",
                  "audio_s": round(len(out_wav) / out_sr, 1), "chars": len(sample)}
    print(f"TTS: {tts_t:.1f}s -> {len(out_wav) / out_sr:.1f}s audio")

    rep["e2e_s"] = round(time.perf_counter() - t_all, 1)
    with open(OUT_JSON, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"E2E wall: {rep['e2e_s']}s. Wrote {OUT_JSON} + {OUT_WAV}")


if __name__ == "__main__":
    main()
