"""Kaggle script kernel: (1) smoke the SHIPPED backend OmniVoice engine on CUDA, (2) ramp re-runs with VRAM sampling:
shipped engine @16 steps, @12 steps, and raw batched generate (batch 4, 12 steps). Repo is public: clone it for the real backend code.
Inputs: dataset yegorby/bp-gpu-bench-refs (clips + refs only are needed)."""
import glob, os, shutil, subprocess, sys

WORK = "/kaggle/working"; os.chdir(WORK)
sh = lambda c, check=True: subprocess.run(c, shell=True, check=check)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv; python --version")
sh("git clone --depth 1 https://github.com/brusnyak/bp.git repo && git -C repo log --oneline -1")
SRC = os.path.dirname(glob.glob("/kaggle/input/**/lc_en_r0.wav", recursive=True)[0])
os.makedirs("inputs", exist_ok=True)
for f in glob.glob(os.path.join(SRC, "*")):
    if f.endswith((".wav", ".txt")):
        shutil.copy(f, "inputs")
sh(f"{sys.executable} -m pip install -q faster-whisper omnivoice jiwer sentencepiece sacremoses librosa soundfile")

sys.path.insert(0, os.path.join(WORK, "repo"))
from backend.mt.convert_opus_mt_to_ct2 import CompatTransformersConverter  # repo's own converter
from transformers import AutoFeatureExtractor, AutoTokenizer
for pair in ("en-sk", "sk-en"):
    out = f"ct2/Helsinki-NLP--opus-mt-{pair}"
    CompatTransformersConverter(f"Helsinki-NLP/opus-mt-{pair}").convert(out, quantization="float16", force=True)
    AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{pair}").save_pretrained(out)
out = "ct2/whisper-small-sk"
CompatTransformersConverter("NaiveNeuron/whisper-small-sk").convert(out, quantization="float16", force=True)
AutoTokenizer.from_pretrained("NaiveNeuron/whisper-small-sk").save_pretrained(out)
AutoFeatureExtractor.from_pretrained("NaiveNeuron/whisper-small-sk").save_pretrained(out)

env = f"PYTHONPATH={WORK}/repo:{WORK}/repo/scripts/gpu_bench"
sh(f"cd {WORK} && {env} {sys.executable} repo/scripts/gpu_bench/engine_smoke.py --inputs inputs --out results/engine", check=False)
LB = f"{env} {sys.executable} repo/scripts/gpu_bench/load_bench.py --models-dir ct2 --inputs inputs --only ramp --duration 60 --device cuda"
sh(f"{LB} --tts engine --steps 16 --levels 2,8,16 --out results/engine_s16", check=False)
sh(f"{LB} --tts engine --steps 12 --levels 2,4,8,12,16,24 --out results/engine_s12", check=False)
sh(f"{LB} --tts omni --batch 4 --steps 12 --levels 2,4,8,12,16,24 --out results/batch4_s12", check=False)
