#!/usr/bin/env python3
"""Engine A/B for the STT fix: Piper vs OmniVoice-generic vs OmniVoice zero-shot.

Same SK text through all engines -> measure synth speed (RTF) AND feed every
output back through both STT rungs (base / small-sk) scoring WER/CER.

Out (local-only): processed/engine_ab/<engine>.wav + matrix.json
Run: .venv/bin/python scripts/engine_ab.py   (omni part auto-delegates to .venv-omni)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
OUT_DIR = os.path.join(REPO_ROOT, "processed", "engine_ab")

# ~200 chars -> ~11-17s spoken: long enough that STT is not context-starved.
TEXT = ("Včera ráno som išiel na trh kúpiť čerstvý chlieb a mlieko. "
        "Stretol som tam starého priateľa Ľuba, ktorý predával med a syry. "
        "Porozprával som mu o svojej práci a o dlhej ceste vlakom cez hory a doliny.")
SK_REF = os.path.join(REPO_ROOT, "processed", "omnivoice", "ref_sk_trhove_head.wav")
SK_REF_TEXT = ("Včera ráno som išiel na trh kúpiť čerstvý chlieb a mlieko. "
               "Stretol som tam starého priateľa Ľuba, ktorý predával med a syry.")


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower(), flags=re.UNICODE)).strip()


OMNI_HELPER = """
import json, sys, time
import torch, soundfile as sf
from omnivoice import OmniVoice
out_wav, mode = sys.argv[1], sys.argv[2]
t0 = time.perf_counter()
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="mps", dtype=torch.float32)
load_s = time.perf_counter() - t0
prompt = None
prep_s = 0.0
if mode == "zeroshot":
    t0 = time.perf_counter()
    prompt = model.create_voice_clone_prompt(sys.argv[3], sys.argv[4])
    prep_s = time.perf_counter() - t0
    text = sys.argv[5]
else:
    text = sys.argv[3]
t0 = time.perf_counter()
audio = model.generate(text=text, language="sk", voice_clone_prompt=prompt, num_step=16)[0]
syn_s = time.perf_counter() - t0
sf.write(out_wav, audio, model.sampling_rate)
print(json.dumps({"load_s": round(load_s, 2), "prep_s": round(prep_s, 2),
                  "syn_s": round(syn_s, 2), "audio_s": round(len(audio) / model.sampling_rate, 2),
                  "sr": model.sampling_rate}))
"""


def synth_piper(model_id: str, name: str) -> dict:
    from backend.tts.piper_tts import PiperTTS
    import soundfile as sf
    tts = PiperTTS(model_id=model_id)
    wav, sr, syn_s = tts.synthesize(TEXT, language="sk")
    path = os.path.join(OUT_DIR, name + ".wav")
    sf.write(path, wav, sr)
    audio_s = len(wav) / sr
    return {"engine": name, "syn_s": round(syn_s, 3), "audio_s": round(audio_s, 2),
            "rtf": round(syn_s / audio_s, 3), "wav": path}


def synth_omni(name: str, mode: str) -> dict:
    path = os.path.join(OUT_DIR, name + ".wav")
    args = [os.path.join(REPO_ROOT, ".venv-omni", "bin", "python"), "-c", OMNI_HELPER,
            path, mode]
    if mode == "zeroshot":
        args += [SK_REF, SK_REF_TEXT]
    args += [TEXT]
    r = subprocess.run(args, capture_output=True, text=True, cwd=REPO_ROOT)
    if r.returncode != 0:
        raise RuntimeError(f"omni {mode} failed:\n{r.stderr[-2000:]}")
    info = json.loads(r.stdout.strip().splitlines()[-1])
    info.update({"engine": name, "rtf": round(info["syn_s"] / info["audio_s"], 3), "wav": path})
    return info


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true",
                    help="reuse existing <engine>.wav instead of re-synthesizing (cheap re-score)")
    ap.add_argument("--matrix", default="matrix.json", help="output matrix filename")
    a = ap.parse_args()
    import librosa
    import numpy as np
    import jiwer
    from backend.stt.faster_whisper_stt import FasterWhisperSTT

    # Order matters only for readability. piper_omni_hq is the voice trained on the
    # OmniVoice HQ corpus (2026-09-30) - added to test whether training on
    # OmniVoice-cloned audio inherits its STT intelligibility (see engine_ab
    # history: piper_personal scored WER 0.540 as STT input vs omni_zeroshot 0.054).
    ENGINES = [
        ("piper_generic", lambda: synth_piper("sk_SK-lili-medium", "piper_generic")),
        ("piper_personal", lambda: synth_piper("sk_SK-personal-male-medium", "piper_personal")),
        ("piper_omni_hq", lambda: synth_piper("me_omni_piper_sk", "piper_omni_hq")),
        ("omni_generic", lambda: synth_omni("omni_generic", "generic")),
        ("omni_zeroshot", lambda: synth_omni("omni_zeroshot", "zeroshot")),
    ]

    print("== synthesis (sequential, one engine at a time) ==", flush=True)
    synth = []
    for name, make in ENGINES:
        cached = os.path.join(OUT_DIR, name + ".wav")
        if a.reuse and os.path.exists(cached):
            import soundfile as sf
            audio_s = sf.info(cached).duration
            synth.append({"engine": name, "wav": cached, "audio_s": round(audio_s, 2),
                          "rtf": None, "source": "cached wav"})
            print(f"{name}: cached ({audio_s:.1f}s)", flush=True)
            continue
        s = make()
        synth.append(s)
        print(s, flush=True)

    print("== STT scoring (sequential, one rung at a time) ==", flush=True)
    ref_n = norm(TEXT)
    matrix = []
    for rung in ("base", "small-sk"):
        stt = FasterWhisperSTT("small-sk" if rung == "small-sk" else "base")
        for s in synth:
            wav, _ = librosa.load(s["wav"], sr=16000, mono=True)
            segs, stt_s, _ = stt.transcribe_audio(np.asarray(wav, dtype=np.float32), 16000,
                                                  language="sk")
            hyp = " ".join(x.text for x in segs)
            row = dict(s)
            row.update({"rung": rung, "wer": round(jiwer.wer(ref_n, norm(hyp)), 4),
                        "cer": round(jiwer.cer(ref_n, norm(hyp)), 4),
                        "stt_s": round(stt_s, 2),
                        "hyp": hyp[:200]})
            matrix.append(row)
            rtf_s = f"{s['rtf']:.3f}" if s.get("rtf") is not None else "cached"
            print(f"{s['engine']:15s} {rung:8s} synth_rtf={rtf_s} "
                  f"WER {row['wer']:.3f} CER {row['cer']:.3f} STT {stt_s:.2f}s", flush=True)
        del stt
    out_name = a.matrix
    with open(os.path.join(OUT_DIR, out_name), "w", encoding="utf-8") as f:
        json.dump({"text": TEXT, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "results": matrix}, f, indent=2, ensure_ascii=False)
    print(f"wrote {os.path.relpath(os.path.join(OUT_DIR, out_name), REPO_ROOT)}")


if __name__ == "__main__":
    main()
