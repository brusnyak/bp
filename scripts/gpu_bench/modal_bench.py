"""Run the GPU bench on Modal: X-Voice + OmniVoice zero-shot cloning (EN/SK/CS) and SeamlessM4T v2 S2ST.

  BENCH_GPU=T4   modal run scripts/gpu_bench/modal_bench.py     # also L4, A10G, L40S (Ada, ~4090-class), A100
Needs `modal` authorised once (`modal token new`). Refs (the owner's voice, 5-7 s each) are read from
processed/gpu_bench/refs and uploaded to Modal for the run; outputs land in processed/gpu_bench/modal_<GPU>/.
Image build (deps + X-Voice checkpoints) is cached after the first run.
"""
from __future__ import annotations

import os

import modal

GPU = os.environ.get("BENCH_GPU", "T4")
HERE = os.path.dirname(os.path.abspath(__file__))
REFS = os.path.abspath(os.path.join(HERE, "..", "..", "processed", "gpu_bench", "refs"))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "espeak-ng", "ffmpeg", "build-essential", "cmake")
    .pip_install("torch", "torchaudio", "transformers", "sentencepiece", "soundfile", "huggingface_hub",
                 "accelerate", "protobuf", "numpy")
    .add_local_file(os.path.join(HERE, "gpu_bench.py"), "/root/gpu_bench.py", copy=True)
    .run_commands("cd /root && python gpu_bench.py setup --xvoice-dir /root/X-Voice")
    .add_local_dir(REFS, "/root/refs")
)
hf_cache = modal.Volume.from_name("hf-cache-bench", create_if_missing=True)
app = modal.App("bp-gpu-bench")

SEAMLESS = [("sk", "eng"), ("cs", "eng"), ("en", "ces")]  # (input ref, output speech lang); no SK speech output in M4T v2


@app.function(image=image, gpu=GPU, timeout=3600, volumes={"/root/.cache/huggingface": hf_cache})
def bench() -> dict:
    import json, os, subprocess, time, glob
    import soundfile as sf, torch

    subprocess.run("cd /root && python gpu_bench.py run --xvoice-dir /root/X-Voice --refs /root/refs --out /root/results",
                   shell=True, check=True)
    res = json.load(open("/root/results/gpu_bench.json"))
    res["machine"]["modal_gpu"] = GPU

    # S2ST: SeamlessM4T v2 on the same reference clips (speech in -> speech out)
    from transformers import AutoProcessor, SeamlessM4Tv2Model
    import librosa  # noqa: F401  (transformers audio extras may pull it; fall back to soundfile resample below)
    proc = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large", torch_dtype=torch.float16).to("cuda")
    s2s = []
    for src, tgt in SEAMLESS:
        wav, sr = sf.read(f"/root/refs/{src}.wav")
        wav = wav.mean(1) if wav.ndim > 1 else wav
        wav = librosa.resample(wav.astype("float32"), orig_sr=sr, target_sr=16000)
        inp = proc(audios=wav, sampling_rate=16000, return_tensors="pt").to("cuda")
        inp = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inp.items()}
        model.generate(**inp, tgt_lang=tgt)  # warm-up
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = model.generate(**inp, tgt_lang=tgt)[0].float().cpu().numpy().squeeze()
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        sf.write(f"/root/results/seamless_{src}_to_{tgt}.wav", out, 16000)
        s2s.append({"engine": "seamless-m4t-v2", "in": src, "out": tgt, "in_s": round(len(wav) / 16000, 2),
                    "out_s": round(len(out) / 16000, 2), "gen_s": round(dt, 3),
                    "rtf_vs_input": round(dt / (len(wav) / 16000), 3)})
        print(s2s[-1], flush=True)
    res["s2st"] = s2s
    json.dump(res, open("/root/results/gpu_bench.json", "w"), indent=2, ensure_ascii=False)
    hf_cache.commit()
    return {"json": json.dumps(res), "files": {os.path.basename(p): open(p, "rb").read()
                                                   for p in glob.glob("/root/results/*.wav") + ["/root/results/gpu_bench.json"]}}


@app.local_entrypoint()
def main():
    out = bench.remote()
    d = os.path.join(HERE, "..", "..", "processed", "gpu_bench", f"modal_{GPU}")
    os.makedirs(d, exist_ok=True)
    for name, data in out["files"].items():
        open(os.path.join(d, name), "wb").write(data)
    print("saved", len(out["files"]), "files to", os.path.abspath(d))
