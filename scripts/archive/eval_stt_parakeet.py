# Slovak STT with NVIDIA Parakeet-TDT-0.6B-v3 (multilingual, 25 European languages incl. Slovak) via onnx-asr (no PyTorch).
# Requires: pip install "onnx-asr[cpu,hub]" jiwer   and eval_data/sk_clips/sk_00.wav ... (see documentation/model_evaluation_2026-09.md)
import glob, json, os, re, sys, time
import jiwer, soundfile as sf

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, os.path.join(B, "scripts"))
import conversation_sim as cs
import onnx_asr

quant = sys.argv[1] if len(sys.argv) > 1 else "int8"
refs = [r[1] for r in cs.script_rows()]
clips = sorted(glob.glob(os.path.join(os.environ.get("EVAL_CLIPS_DIR", os.path.join(B, "eval_data", "sk_clips")), "sk_*.wav")))
assert len(clips) == len(refs), (len(clips), len(refs))
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()

t0 = time.perf_counter()
m = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", quantization=None if quant == "fp32" else quant)
load_s = time.perf_counter() - t0
m.recognize(clips[0])  # warm-up
hyps, times, audio_s = [], [], 0.0
for c in clips:
    info = sf.info(c); audio_s += info.frames / info.samplerate
    t = time.perf_counter(); hyps.append(str(m.recognize(c)).strip()); times.append(time.perf_counter() - t)
R, H = [norm(r) for r in refs], [norm(h) for h in hyps]
out = {"model": f"parakeet-tdt-0.6b-v3-{quant}", "load_s": round(load_s, 1), "wer": round(jiwer.wer(R, H), 3), "cer": round(jiwer.cer(R, H), 3),
       "s_per_clip": round(sum(times) / len(times), 2), "rtf": round(sum(times) / audio_s, 2), "hyps": hyps}
os.makedirs(os.path.join(B, "processed", "eval_out"), exist_ok=True)
json.dump(out, open(os.path.join(B, "processed", "eval_out", f"stt_parakeet_{quant}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print({k: v for k, v in out.items() if k != "hyps"})
for i in (0, 3, 7): print(i, "REF:", refs[i], "\n   HYP:", hyps[i])
