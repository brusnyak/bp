"""Kaggle script kernel: X-Voice + OmniVoice cloning bench + SeamlessM4T v2 S2ST on the Kaggle GPU.
Inputs: private dataset yegorby/bp-gpu-bench-refs (gpu_bench.py + refs/{en,sk,cs}.{wav,txt}). Outputs: /kaggle/working/results."""
import glob, json, os, shutil, subprocess, sys, time

SEAMLESS_ONLY = os.environ.get("SEAMLESS_ONLY", "1") == "1"  # cloning bench already done on run v3
SRC = os.path.dirname(glob.glob("/kaggle/input/**/gpu_bench.py", recursive=True)[0])
print("SRC", SRC, os.listdir(SRC))
WORK = "/kaggle/working"
os.chdir(WORK)
for f in ("gpu_bench.py",):
    shutil.copy(os.path.join(SRC, f), f)
os.makedirs("refs", exist_ok=True)
for f in glob.glob(os.path.join(SRC, "*.wav")) + glob.glob(os.path.join(SRC, "*.txt")):
    shutil.copy(f, "refs")
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv; python --version", shell=True)

if SEAMLESS_ONLY:
    os.makedirs("results", exist_ok=True)
    res = {"machine": {}}
else:
    subprocess.run([sys.executable, "gpu_bench.py", "setup", "--xvoice-dir", "X-Voice"], check=False)
    subprocess.run([sys.executable, "gpu_bench.py", "run", "--xvoice-dir", "X-Voice", "--refs", "refs", "--out", "results"], check=False)
    res = json.load(open("results/gpu_bench.json"))

# ---- S2ST: SeamlessM4T v2 (speech in -> speech out); Slovak has no speech output in M4T v2
import soundfile as sf, torch
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentencepiece", "librosa"], check=False)
import librosa
from transformers import AutoProcessor, SeamlessM4Tv2Model
proc = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large", torch_dtype=torch.float16).to("cuda")
s2s = []
for src, tgt in [("sk", "eng"), ("cs", "eng"), ("en", "ces")]:
    wav, sr = sf.read(f"refs/{src}.wav")
    wav = wav.mean(1) if wav.ndim > 1 else wav
    wav = librosa.resample(wav.astype("float32"), orig_sr=sr, target_sr=16000)
    inp = proc(audio=wav, sampling_rate=16000, return_tensors="pt").to("cuda")
    inp = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inp.items()}
    model.generate(**inp, tgt_lang=tgt)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = model.generate(**inp, tgt_lang=tgt)[0].float().cpu().numpy().squeeze()
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    sf.write(f"results/seamless_{src}_to_{tgt}.wav", out, 16000)
    s2s.append({"engine": "seamless-m4t-v2", "in": src, "out": tgt, "in_s": round(len(wav) / 16000, 2),
                "out_s": round(len(out) / 16000, 2), "gen_s": round(dt, 3), "rtf_vs_input": round(dt / (len(wav) / 16000), 3)})
    print(s2s[-1], flush=True)
res["s2st"] = s2s
json.dump(res, open("results/s2st.json" if SEAMLESS_ONLY else "results/gpu_bench.json", "w"), indent=2, ensure_ascii=False)
print("RESULT_JSON", json.dumps(res)[:6000])
# keep /kaggle/working small: drop the clone + checkpoints, keep results/
shutil.rmtree("X-Voice", ignore_errors=True)
