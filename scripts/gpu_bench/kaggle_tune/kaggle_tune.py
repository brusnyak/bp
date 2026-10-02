"""Kaggle script kernel: full-cascade load bench (sweep / ramp / timeline) on the Kaggle GPU. Inputs: dataset yegorby/bp-gpu-bench-refs."""
import glob, os, shutil, subprocess, sys

SRC = os.path.dirname(glob.glob("/kaggle/input/**/tts_tune.py", recursive=True)[0])
WORK = "/kaggle/working"; os.chdir(WORK)
os.makedirs("inputs", exist_ok=True)
for f in glob.glob(os.path.join(SRC, "*")):
    if os.path.isfile(f):
        shutil.copy(f, "inputs" if f.endswith((".wav", ".txt")) else ".")
sh = lambda c: subprocess.run(c, shell=True, check=True)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv; python --version")
sh(f"{sys.executable} -m pip install -q faster-whisper omnivoice jiwer sentencepiece sacremoses librosa soundfile")

# convert models exactly as scripts/convert_models.py does (MT int8 -> float16 on GPU, whisper-small-sk)
sys.path.insert(0, WORK)
from convert_opus_mt_to_ct2 import CompatTransformersConverter
from transformers import AutoFeatureExtractor, AutoTokenizer
for pair in ("en-sk", "sk-en"):
    out = f"ct2/Helsinki-NLP--opus-mt-{pair}"
    CompatTransformersConverter(f"Helsinki-NLP/opus-mt-{pair}").convert(out, quantization="float16", force=True)
    AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{pair}").save_pretrained(out)
out = "ct2/whisper-small-sk"
CompatTransformersConverter("NaiveNeuron/whisper-small-sk").convert(out, quantization="float16", force=True)
AutoTokenizer.from_pretrained("NaiveNeuron/whisper-small-sk").save_pretrained(out)
AutoFeatureExtractor.from_pretrained("NaiveNeuron/whisper-small-sk").save_pretrained(out)

sh(f"{sys.executable} tts_tune.py --models-dir ct2 --inputs inputs --out results --tts omni --device cuda")
