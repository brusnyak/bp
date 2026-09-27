#!/usr/bin/env python3
"""Pipeline latency probe: first-output latency + steady-state per-sentence cost.

Per-language STT routing (the near-real-time path):
  EN source -> Parakeet-TDT-v3 (.venv-stt, subprocess, JSON)
  SK source -> faster-whisper small (main venv, preloaded)
Then MT chunked batch + Piper direct TTS on the FIRST sentence only.

Measures what a live listener feels: time from speech chunk in to first
translated audio out, then per-sentence continuation cost.
Run: venv/bin/python scripts/pipeline_latency_probe.py
Out: processed/pipeline_latency.json
"""

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(REPO_ROOT, "processed", "pipeline_latency.json")

PROBE_S = 30  # first N seconds of each recording = simulated live onset


def main():
    import librosa
    import numpy as np
    import soundfile as sf

    from backend.stt.faster_whisper_stt import FasterWhisperSTT
    from backend.mt.ctranslate2_mt import CTranslate2MT
    from backend.tts.piper_tts import PiperTTS

    rep = {}
    tmp = "/tmp/latency_probe.wav"

    # ---- EN path: Parakeet (subprocess, own venv) ----
    wav, _ = librosa.load("speaker_voices/en_script_reading.m4a", sr=16000, mono=True)
    sf.write(tmp, np.asarray(wav[: PROBE_S * 16000], dtype=np.float32), 16000)
    # NOTE: spike script reads the registry clip; emulate onset by direct arg override
    t0 = time.perf_counter()
    r = subprocess.run(
        [".venv-stt/bin/python", "scripts/stt_parakeet_spike.py",
         "--clip", "en", "--max-windows", "1", "--json"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    stt_wall = time.perf_counter() - t0
    en_text = json.loads(r.stdout.strip().splitlines()[-1])["text"]
    rep["en_stt"] = {"engine": "parakeet-tdt-0.6b-v3", "wall_s": round(stt_wall, 2),
                     "note": "wall incl ~3s model load; decode-only ~3s/60s audio"}

    # ---- SK path: faster-whisper small, preloaded (warm) ----
    stt = FasterWhisperSTT(model_size="small")
    wav, _ = librosa.load("speaker_voices/sk_script_reading.m4a", sr=16000, mono=True)
    seg = np.asarray(wav[: PROBE_S * 16000], dtype=np.float32)
    t0 = time.perf_counter()
    segs, _, lang = stt.transcribe_audio(seg, 16000, language="sk")
    sk_time = time.perf_counter() - t0
    sk_text = " ".join(s.text if hasattr(s, "text") else s["text"] for s in segs)
    rep["sk_stt"] = {"engine": "faster-whisper-small", "time_s": round(sk_time, 2),
                     "lang": lang, "chars": len(sk_text)}

    # ---- MT chunked, both directions, first sentence + full probe ----
    for name, text, src, tgt in [("en_sk", en_text, "en", "sk"), ("sk_en", sk_text, "sk", "en")]:
        mt = CTranslate2MT(f"Helsinki-NLP/opus-mt-{src}-{tgt}")
        chunks = CTranslate2MT.split_chunks(text) or [text]
        t0 = time.perf_counter()
        out, _ = mt.translate_segments([chunks[0]], src, tgt)
        first_t = time.perf_counter() - t0
        t0 = time.perf_counter()
        all_out, _ = mt.translate_segments(chunks, src, tgt)
        full_t = time.perf_counter() - t0
        rep[f"mt_{name}"] = {"first_chunk_s": round(first_t, 3),
                             "all_chunks_s": round(full_t, 3),
                             "n_chunks": len(chunks),
                             "sample_out": all_out[0][:120] if all_out else ""}
        if name == "en_sk":
            first_sk_sentence = all_out[0] if all_out else ""

    # ---- TTS first audio: Piper direct (fast tier), first SK sentence ----
    piper = PiperTTS(model_id="sk_SK-lili-medium")
    t0 = time.perf_counter()
    audio, sr, _ = piper.synthesize(first_sk_sentence, language="sk")
    tts_t = time.perf_counter() - t0
    rep["tts_first_audio"] = {"engine": "piper-direct/lili", "time_s": round(tts_t, 2),
                              "audio_s": round(len(audio) / sr, 1)}

    with open(OUT_JSON, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
