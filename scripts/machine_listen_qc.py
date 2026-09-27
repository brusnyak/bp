#!/usr/bin/env python3
"""Machine listening panel: score synth outputs WITHOUT human ears.

Per wav: intelligibility (STT WER overall + per-third -> catches
'degrades toward the end'), acoustic health (clipping %, noise floor,
F0 std -> monotone/robotic flag, end-energy -> cutoff flag, rate).
Similarity stays in voice_similarity_qc.py (resemblyzer, separate venv).

Expected texts are the known synth inputs (see voice_similarity_qc
CANDIDATE_TEXTS + lab A/B sentence).
Run: venv/bin/python scripts/machine_listen_qc.py
Out: processed/machine_listen.json
"""

import json
import os
import re
import string
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QC_DIR = os.path.join(REPO_ROOT, "processed", "voice_qc")
OUT_JSON = os.path.join(REPO_ROOT, "processed", "machine_listen.json")

EN_TEXT = "This is a short test sentence synthesized for voice similarity quality control."
SK_TEXT = "Toto je krátka testovacia veta vytvorená na kontrolu kvality podobnosti hlasu."
SK_AB = "Dobré ráno a vitajte v tejto živej ukážke prekladu reči v reálnom čase."

TARGETS = {
    "hybrid_en.wav": ("en", EN_TEXT),
    "piper_personal_en.wav": ("en", EN_TEXT),
    "hybrid_sk.wav": ("sk", SK_TEXT),
    "piper_generic_sk.wav": ("sk", SK_TEXT),
    "piper_personal_sk_test.wav": ("sk", SK_AB),
    "piper_generic_sk_test.wav": ("sk", SK_AB),
    "piper_personal_sk_ns09.wav": ("sk", SK_AB),
    "piper_personal_sk_ns11.wav": ("sk", SK_AB),
    "piper_personal_sk_2500.wav": ("sk", SK_AB),
    "piper_male_sk_test.wav": ("sk", SK_AB),
    "sk2500_ns05.wav": ("sk", SK_AB),
    "sk2500_ns08.wav": ("sk", SK_AB),
    "enpers_ns03.wav": ("en", EN_TEXT),
    "enpers_ns05.wav": ("en", EN_TEXT),
}


def plain(s: str) -> str:
    s = s.lower().translate(str.maketrans("", "", string.punctuation + "„“”"))
    return re.sub(r"\s+", " ", s).strip()


def wer(hyp: str, ref: str) -> float:
    h, r = hyp.split(), ref.split()
    if not r:
        return 1.0
    prev = list(range(len(r) + 1))
    for i, hw in enumerate(h, 1):
        cur = [i]
        for j, rw in enumerate(r, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (hw != rw)))
        prev = cur
    return prev[-1] / len(r)


def stt_en(path: str) -> str:
    r = subprocess.run(
        [".venv-stt/bin/python", "-c",
         "import sys,librosa,torch\n"
         "from transformers import ParakeetForTDT, ParakeetProcessor\n"
         "proc = ParakeetProcessor.from_pretrained('nvidia/parakeet-tdt-0.6b-v3')\n"
         "model = ParakeetForTDT.from_pretrained('nvidia/parakeet-tdt-0.6b-v3').eval()\n"
         "wav,_ = librosa.load(sys.argv[1], sr=16000, mono=True)\n"
         "inp = proc(wav, sampling_rate=16000, return_tensors='pt')\n"
         "import torch\n"
         "with torch.no_grad(): out = model.generate(**inp)\n"
         "print(proc.batch_decode(out.sequences, skip_special_tokens=True)[0])",
         path],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""


def main():
    import librosa
    import numpy as np

    from backend.stt.faster_whisper_stt import FasterWhisperSTT

    stt_sk = FasterWhisperSTT(model_size="small")
    rep = {}
    for fname, (lang, expected) in TARGETS.items():
        path = os.path.join(QC_DIR, fname)
        if not os.path.exists(path):
            rep[fname] = {"error": "missing"}
            continue
        wav, sr = librosa.load(path, sr=22050, mono=True)
        n = len(wav)
        thirds = [wav[: n // 3], wav[n // 3 : 2 * n // 3], wav[2 * n // 3 :]]
        exp_words = plain(expected).split()
        ew = [exp_words[: len(exp_words) // 3], exp_words[len(exp_words) // 3 : 2 * len(exp_words) // 3],
              exp_words[2 * len(exp_words) // 3 :]]
        wers, hyps = [], []
        for seg, ew3 in zip(thirds, ew):
            if lang == "en":
                import soundfile as sf
                sf.write("/tmp/ml_seg.wav", seg, sr)
                hyp = plain(stt_en("/tmp/ml_seg.wav"))
            else:
                segs, _, _ = stt_sk.transcribe_audio(np.asarray(seg, dtype=np.float32), sr, language="sk")
                hyp = plain(" ".join(s.text if hasattr(s, "text") else s["text"] for s in segs))
            hyps.append(hyp)
            wers.append(round(wer(hyp, " ".join(ew3)), 3))
        full_wer = round(wer(plain(" ".join(hyps)), plain(expected)), 3)

        peak = float(np.max(np.abs(wav)))
        clip_pct = round(100 * float(np.mean(np.abs(wav) > 0.99)), 3)
        rms = float(np.sqrt(np.mean(wav ** 2)))
        quiet = float(np.percentile(np.abs(wav), 5))
        f0, _, _ = librosa.pyin(wav, fmin=librosa.note_to_hz("C2"),
                                fmax=librosa.note_to_hz("C7"), sr=sr)
        f0v = f0[~np.isnan(f0)]
        f0std = round(float(np.std(f0v)), 1) if len(f0v) else 0.0
        # tremor: mean frame-to-frame F0 jump relative to mean F0 (voiced only)
        d = np.abs(np.diff(f0v))
        jitter = round(float(np.mean(d) / (np.mean(f0v) + 1e-9)), 4) if len(f0v) > 1 else 0.0
        # hiss: harmonic-to-noise energy ratio in dB (lower = noisier/shhh)
        harm, perc = librosa.effects.hpss(wav)
        hnr = round(10 * float(np.log10((np.mean(harm ** 2) + 1e-12) / (np.mean(perc ** 2) + 1e-12))), 1)
        tail_e = float(np.mean(wav[-int(0.2 * sr) :] ** 2)) / (rms ** 2 + 1e-9)

        rep[fname] = {
            "wer_full": full_wer, "wer_thirds": wers,
            "degrades": bool(wers[2] > wers[0] + 0.15),
            "peak": round(peak, 3), "clip_pct": clip_pct,
            "noise_floor": round(quiet, 4), "f0_std_hz": f0std,
            "f0_jitter": jitter, "hnr_db": hnr,
            "tail_energy_ratio": round(tail_e, 2),
            "dur_s": round(n / sr, 1),
        }
        print(f"{fname}: WER {full_wer} thirds {wers} f0std {f0std}Hz jitter {jitter} HNR {hnr}dB")

    with open(OUT_JSON, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
