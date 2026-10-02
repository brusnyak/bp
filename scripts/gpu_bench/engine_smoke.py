#!/usr/bin/env python3
"""Smoke/latency test of the shipped backend/tts/omni_tts.py on a GPU: enrollment cost, first-chunk latency of synthesize_stream,
cached-prompt reuse. Run from the repo root with PYTHONPATH=. : python scripts/gpu_bench/engine_smoke.py --inputs inputs --out results"""
import argparse, json, os, time
import numpy as np, soundfile as sf

ap = argparse.ArgumentParser(); ap.add_argument("--inputs", default="inputs"); ap.add_argument("--out", default="results")
ap.add_argument("--device", default="cuda"); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
from backend.tts.omni_tts import OmniVoiceTTS

res = {}
t0 = time.perf_counter(); eng = OmniVoiceTTS(device=a.device); res["model_load_s"] = round(time.perf_counter() - t0, 2)
SENT = {"sk": "Dobré ráno, vitajte na dnešnej prezentácii. Ukážeme vám preklad reči v reálnom čase, s vaším vlastným hlasom.",
        "en": "Good morning, and welcome to today's presentation. We will show you real-time speech translation in your own voice."}
for lang in ("sk", "en"):
    ref = os.path.join(a.inputs, f"{lang}.wav")
    t0 = time.perf_counter(); eng.prepare_voice(ref); enroll = time.perf_counter() - t0   # first call includes the one-off warm-up
    t0 = time.perf_counter(); eng.prepare_voice(ref); cached = time.perf_counter() - t0
    for steps in (16, 12):
        eng.num_step = steps
        chunks, first, t0 = [], None, time.perf_counter()
        for c in eng.synthesize_stream(SENT[lang], lang, ref):
            if first is None:
                first = time.perf_counter() - t0
            chunks.append(c)
        total = time.perf_counter() - t0
        wav = np.concatenate(chunks)
        res[f"{lang}_steps{steps}"] = {"enroll_s": round(enroll, 2), "cached_prompt_s": round(cached, 4), "first_chunk_s": round(first, 2),
                                       "total_s": round(total, 2), "audio_s": round(len(wav) / eng.sample_rate, 2),
                                       "rtf": round(total / (len(wav) / eng.sample_rate), 3)}
        sf.write(os.path.join(a.out, f"engine_{lang}_steps{steps}.wav"), wav, eng.sample_rate)
        print(lang, steps, res[f"{lang}_steps{steps}"], flush=True)
json.dump(res, open(os.path.join(a.out, "engine_smoke.json"), "w"), indent=1)
