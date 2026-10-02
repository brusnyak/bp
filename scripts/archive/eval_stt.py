# Requires: python scripts/setup.py --dev, plus sk_*.wav clips (see documentation/model_evaluation_2026-09.md).
"""SK STT candidates on the 18 real sentence clips (user's own voice, 16 kHz wav).
Usage: python eval_stt.py <faster-whisper model name> [compute_type]
Same options as production FasterWhisperSTT.transcribe_audio: beam 5, vad_filter, int8, language=sk.
"""
import glob, json, os, re, sys, time
import jiwer, librosa, numpy as np

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, B); sys.path.insert(0, os.path.join(B, "scripts"))
import conversation_sim as cs
from faster_whisper import WhisperModel

model_name = sys.argv[1]; ctype = sys.argv[2] if len(sys.argv) > 2 else "int8"
refs = [r[1] for r in cs.script_rows()]
clips = sorted(glob.glob(os.path.join(os.environ.get("EVAL_CLIPS_DIR", os.path.join(B, "eval_data", "sk_clips")), "sk_*.wav")))
assert len(clips) == len(refs) == 18, (len(clips), len(refs))

norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()
t0 = time.perf_counter(); m = WhisperModel(model_name, device="cpu", compute_type=ctype); load_s = time.perf_counter() - t0
hyps, times, audio_s = [], [], 0.0
w = librosa.load(clips[0], sr=16000)[0]; list(m.transcribe(w[:16000], language="sk")[0])  # warm-up, untimed
for c in clips:
    wav, _ = librosa.load(c, sr=16000, mono=True); audio_s += len(wav) / 16000
    t = time.perf_counter()
    segs, _ = m.transcribe(wav.astype(np.float32), language="sk", beam_size=5, vad_filter=True)
    hyps.append(" ".join(s.text for s in segs).strip()); times.append(time.perf_counter() - t)
R, H = [norm(r) for r in refs], [norm(h) for h in hyps]
out = {"model": model_name, "compute": ctype, "load_s": round(load_s, 1),
       "wer": round(jiwer.wer(R, H), 3), "cer": round(jiwer.cer(R, H), 3),
       "s_per_clip": round(sum(times) / len(times), 2), "rtf": round(sum(times) / audio_s, 2), "audio_s": round(audio_s, 1),
       "hyps": hyps}
os.makedirs(os.path.join(B, "processed", "eval_out"), exist_ok=True)
tag = os.path.basename(model_name.replace("\\", "/").rstrip("/"))  # model name or local model directory
json.dump(out, open(os.path.join(B, "processed", "eval_out", f"stt_{tag}_{ctype}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print({k: v for k, v in out.items() if k != "hyps"})
