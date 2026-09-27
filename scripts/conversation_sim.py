#!/usr/bin/env python3
"""Simulated live conversation both directions, per-sentence AND per-word timing.

EN->SK and SK->EN over the real script sentences: STT per sentence clip
(Parakeet EN / whisper-small SK) -> MT sentence-mode vs word-window mode
(time-to-first-chunk + total) -> TTS per sentence (male SK / ryan EN).
Saves sample outputs + metrics for the lab conversation section.
Run: venv/bin/python scripts/conversation_sim.py
Out: processed/conversation_sim.json + processed/conversation/*.wav
"""

import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))
                            ) if False else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONV_DIR = os.path.join(REPO_ROOT, "processed", "conversation")
OUT_JSON = os.path.join(REPO_ROOT, "processed", "conversation_sim.json")
SCRIPT_MD = os.path.join(REPO_ROOT, "documentation", "reading_script_bilingual.md")


def script_rows():
    rows = []
    with open(SCRIPT_MD) as f:
        for line in f:
            if line.startswith("|") and not line.startswith("| #") and "---" not in line:
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(cells) >= 3 and cells[0].isdigit():
                    rows.append((cells[1], cells[2]))
    return rows


def split_sentences(audio_path, n_expected):
    """Silence-midpoint segmentation snapped to n_expected (cf. segment_sk_dataset)."""
    import librosa
    import soundfile as sf

    p = subprocess.run(
        ["ffmpeg", "-i", audio_path, "-af", "silencedetect=noise=-35dB:d=0.4",
         "-f", "null", "-"], capture_output=True, text=True,
    )
    starts, bounds = [], []
    for line in p.stderr.splitlines():
        m = re.search(r"silence_start: ([\d.]+)", line)
        if m:
            starts.append(float(m.group(1)))
    si = 0
    for line in p.stderr.splitlines():
        m = re.search(r"silence_end: ([\d.]+) \| silence_duration: ([\d.]+)", line)
        if m:
            bounds.append((starts[si], float(m.group(1)), float(m.group(2))))
            si += 1
    wav, _ = librosa.load(audio_path, sr=16000, mono=True)
    dur = len(wav) / 16000
    pauses = [(s, e, d) for s, e, d in bounds if d >= 1.0 and s > 0.5 and e < dur - 0.5]
    cuts = sorted((s + e) / 2 for s, e, d in pauses)
    while len(cuts) + 1 > n_expected:
        idx = min(range(len(cuts)),
                  key=lambda i: next(d for s, e, d in pauses if abs((s + e) / 2 - cuts[i]) < 0.01))
        cuts.pop(idx)
    segs = [0.0] + cuts + [dur]
    while len(segs) - 1 < n_expected:
        i = max(range(len(segs) - 1), key=lambda i: segs[i + 1] - segs[i])
        segs.insert(i + 1, (segs[i] + segs[i + 1]) / 2)
    return [(segs[i], segs[i + 1]) for i in range(n_expected)], wav


def parakeet_batch(files):
    """One subprocess, model loaded once, transcribe many clips. Returns {file: text}."""
    helper = (
        "import sys,json,librosa\n"
        "from transformers import ParakeetForTDT, ParakeetProcessor\n"
        "proc = ParakeetProcessor.from_pretrained('nvidia/parakeet-tdt-0.6b-v3')\n"
        "model = ParakeetForTDT.from_pretrained('nvidia/parakeet-tdt-0.6b-v3').eval()\n"
        "import torch\n"
        "out = {}\n"
        "for f in sys.argv[1:]:\n"
        "    wav,_ = librosa.load(f, sr=16000, mono=True)\n"
        "    inp = proc(wav, sampling_rate=16000, return_tensors='pt')\n"
        "    with torch.no_grad(): r = model.generate(**inp)\n"
        "    out[f] = proc.batch_decode(r.sequences, skip_special_tokens=True)[0]\n"
        "print(json.dumps(out))"
    )
    r = subprocess.run([".venv-stt/bin/python", "-c", helper] + files,
                       capture_output=True, text=True, cwd=REPO_ROOT)
    line = r.stdout.strip().splitlines()[-1]
    return json.loads(line)


def main():
    import librosa
    import numpy as np
    import soundfile as sf

    from backend.stt.faster_whisper_stt import FasterWhisperSTT
    from backend.mt.ctranslate2_mt import CTranslate2MT
    from backend.tts.piper_tts import PiperTTS

    os.makedirs(CONV_DIR, exist_ok=True)
    rows = script_rows()
    print(f"script rows: {len(rows)}")

    # sentence clips for both languages
    en_bounds, _ = split_sentences(f"{REPO_ROOT}/speaker_voices/en_script_reading.m4a", len(rows))
    sk_bounds, _ = split_sentences(f"{REPO_ROOT}/speaker_voices/sk_script_reading.m4a", len(rows))
    en_wav, _ = librosa.load(f"{REPO_ROOT}/speaker_voices/en_script_reading.m4a", sr=16000, mono=True)
    sk_wav, _ = librosa.load(f"{REPO_ROOT}/speaker_voices/sk_script_reading.m4a", sr=16000, mono=True)

    en_clips, sk_clips = [], []
    for i, (a, b) in enumerate(en_bounds):
        p = f"/tmp/conv_en_{i:02d}.wav"
        sf.write(p, en_wav[int(a * 16000):int(b * 16000)], 16000)
        en_clips.append(p)
    for i, (a, b) in enumerate(sk_bounds):
        p = f"/tmp/conv_sk_{i:02d}.wav"
        sf.write(p, sk_wav[int(a * 16000):int(b * 16000)], 16000)
        sk_clips.append(p)

    t0 = time.perf_counter()
    en_texts = parakeet_batch(en_clips)
    en_stt_t = time.perf_counter() - t0
    print(f"EN STT all: {en_stt_t:.1f}s for {len(en_clips)} clips (incl model load)")

    stt_sk = FasterWhisperSTT(model_size="small")
    sk_texts, sk_stt_t = {}, 0.0
    for p in sk_clips:
        seg, _ = librosa.load(p, sr=16000, mono=True)
        t1 = time.perf_counter()
        segs, _, _ = stt_sk.transcribe_audio(np.asarray(seg, dtype=np.float32), 16000, language="sk")
        sk_stt_t += time.perf_counter() - t1
        sk_texts[p] = " ".join(s.text if hasattr(s, "text") else s["text"] for s in segs)
    print(f"SK STT all: {sk_stt_t:.1f}s for {len(sk_clips)} clips (warm model)")

    mt_en_sk = CTranslate2MT("Helsinki-NLP/opus-mt-en-sk")
    mt_sk_en = CTranslate2MT("Helsinki-NLP/opus-mt-sk-en")
    tts_sk = PiperTTS(model_id="sk_SK-personal-male-medium")
    tts_en = PiperTTS(model_id="en_US-ryan-medium")

    rep = {"sentences": [], "agg": {}}
    for mode, clips, texts, mt, tts, src, tgt in [
        ("en_sk", en_clips, en_texts, mt_en_sk, tts_sk, "en", "sk"),
        ("sk_en", sk_clips, sk_texts, mt_sk_en, tts_en, "sk", "en"),
    ]:
        first_s, first_w, tot_s, tot_w, tot_stt, tot_tts = [], [], [], [], [], []
        for i, p in enumerate(clips):
            text = texts[p]
            # sentence-mode MT
            t1 = time.perf_counter()
            s_out, _ = mt.translate_segments([text] if text.strip() else ["."], src, tgt)
            mt_s = time.perf_counter() - t1
            # word-window mode MT (6-word windows)
            words = text.split() or ["."]
            wins = [" ".join(words[j:j + 6]) for j in range(0, len(words), 6)]
            t1 = time.perf_counter()
            w_out, _ = mt.translate_segments(wins, src, tgt)
            mt_w = time.perf_counter() - t1
            # TTS on sentence-mode output
            t1 = time.perf_counter()
            audio, sr, _ = tts.synthesize(s_out[0][:600], language=tgt)
            tts_t = time.perf_counter() - t1
            first_s.append(mt_s)
            first_w.append(mt_w / max(len(wins), 1))
            tot_s.append(mt_s)
            tot_w.append(mt_w)
            tot_tts.append(tts_t)
            if i < 2:
                sf.write(f"{CONV_DIR}/{mode}_{i:02d}.wav", audio, sr)
            rep["sentences"].append({
                "dir": mode, "n": i, "mt_sentence_s": round(mt_s, 3),
                "mt_word_total_s": round(mt_w, 3),
                "mt_word_first_s": round(mt_w / max(len(wins), 1), 3),
                "tts_s": round(tts_t, 2), "audio_s": round(len(audio) / sr, 1),
            })
        rep["agg"][mode] = {
            "mt_sentence_p50_s": round(sorted(tot_s)[len(tot_s) // 2], 3),
            "mt_word_first_p50_s": round(sorted(first_w)[len(first_w) // 2], 3),
            "tts_p50_s": round(sorted(tot_tts)[len(tot_tts) // 2], 2),
        }
    rep["agg"]["stt"] = {
        "en_parakeet_total_s": round(en_stt_t, 1),
        "sk_whisper_total_s": round(sk_stt_t, 1),
        "note": "EN incl one model load; SK warm",
    }
    with open(OUT_JSON, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep["agg"], indent=2))


if __name__ == "__main__":
    main()
