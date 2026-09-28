# Requires: python scripts/setup.py --dev, plus sk_*.wav clips (see documentation/model_evaluation_2026-09.md).
"""Tuning variants of eval_stt.py. Usage: eval_stt_tune.py <model> <threads> <beam> <prompt 0|1>"""
import glob, json, os, re, sys, time
import jiwer, librosa, numpy as np

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, B); sys.path.insert(0, os.path.join(B, "scripts"))
import conversation_sim as cs
from faster_whisper import WhisperModel

model, threads, beam, prompt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1"
PROMPT = "Dobrý deň, vitajte na konferencii. Hovorím po slovensky a prosím o pozornosť."  # generic; no script vocabulary
refs = [r[1] for r in cs.script_rows()]
clips = sorted(glob.glob(os.path.join(B, "eval_data", "sk_clips", "sk_*.wav")))
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()
m = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=threads)
w = librosa.load(clips[0], sr=16000)[0]; list(m.transcribe(w[:16000], language="sk")[0])
hyps, times, audio_s = [], [], 0.0
for c in clips:
    wav, _ = librosa.load(c, sr=16000, mono=True); audio_s += len(wav) / 16000
    t = time.perf_counter()
    segs, _ = m.transcribe(wav.astype(np.float32), language="sk", beam_size=beam, vad_filter=True,
                           initial_prompt=PROMPT if prompt else None)
    hyps.append(" ".join(s.text for s in segs).strip()); times.append(time.perf_counter() - t)
R, H = [norm(r) for r in refs], [norm(h) for h in hyps]
print(json.dumps({"model": model, "threads": threads, "beam": beam, "prompt": prompt, "wer": round(jiwer.wer(R, H), 3),
                  "cer": round(jiwer.cer(R, H), 3), "s_per_clip": round(sum(times) / len(times), 2), "rtf": round(sum(times) / audio_s, 2)}), flush=True)
