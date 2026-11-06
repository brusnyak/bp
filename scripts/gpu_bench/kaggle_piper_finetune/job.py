#!/usr/bin/env python3
"""Kaggle T4 kernel: Piper voice fine-tuning for EN and SK.

Trains Piper TTS voices on the owner's voice recordings:
  - EN: The Rainbow Passage (me_en_rainbow.m4a)
  - SK: Trhove rano (me_sk_trhove.m4a)

Outputs fine-tuned Piper .onnx + .onnx.json models to /kaggle/working/
for download and local use.

Run: pushed as private kernel, auto-executes on push.
Outputs: /kaggle/working/piper_en.onnx, /kaggle/working/piper_sk.onnx
"""
import hashlib, json, os, shutil, subprocess, sys, tarfile, time, traceback
from pathlib import Path

import soundfile as sf

OUT = Path("/kaggle/working")
REPORT = OUT / "piper_finetune_report.json"
CODE = OUT / "bp"
PIPER_DIR = OUT / "piper"
TRAIN_DIR = OUT / "piper_train"
T0 = time.time()
R = {"schema": 1, "status": "running", "steps": [], "errors": [], "models": {}, "acceptance": "route_unverified"}


def save():
    R["elapsed_s"] = round(time.time() - T0, 1)
    tmp = REPORT.with_suffix(".tmp")
    tmp.write_text(json.dumps(R, indent=1, ensure_ascii=False))
    tmp.replace(REPORT)


def step(name, **kw):
    R["steps"].append({"t": round(time.time() - T0, 1), "step": name, **kw})
    print(f"[{time.time() - T0:7.1f}s] {name} {kw}", flush=True)
    save()


def sh(cmd, timeout=600):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)[-4000:]


# ---------------------------------------------------------------------------
# 0. Environment + GPU check
# ---------------------------------------------------------------------------
step("env", python=sys.version, cwd=os.getcwd())
rc, out = sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader", 60)
step("gpu", output=out.strip())

# ---------------------------------------------------------------------------
# 1. Find staged BP repo + voice memo audio files in the dataset
# ---------------------------------------------------------------------------
step("repo_lookup")
src = next((p for p in Path("/kaggle/input").rglob("app.py")
            if (p.parent / "benchmark-provenance.json").is_file()), None)
if src is None:
    step("error", msg="no staged BP repo found in /kaggle/input")
    R["status"] = "failed"; save(); sys.exit(1)
repo = src.parent
step("repo", path=str(repo))

work = OUT / "bp"
if work.exists():
    shutil.rmtree(work)
shutil.copytree(repo, work, ignore=shutil.ignore_patterns('.env', '.env.*', 'certs', '.git', '__pycache__'))
os.chdir(work)
step("checkout", path=str(work))

# Voice memo files for training
voice_files = {}
candidates = {
    "en": ["speaker_voices/me_en_rainbow.m4a", "The Rainbow Passage.m4a", "The Rainbow Passage 2.m4a"],
    "sk": ["speaker_voices/me_sk_trhove.m4a", "Trhove rano.m4a", "Trhove rano 2.m4a"],
}
for lang, paths in candidates.items():
    for p in paths:
        cand = work / p
        if cand.is_file():
            voice_files[lang] = str(cand)
            step("voice_file", lang=lang, path=p)
            break
    if lang not in voice_files:
        step("voice_file_missing", lang=lang)

# ---------------------------------------------------------------------------
# 2. Install TensorFlow GPU + piper training deps
# ---------------------------------------------------------------------------
step("install_tf")
rc, out = sh(f"{sys.executable} -m pip install -q tensorflow==2.15.0", 900)
if rc != 0:
    step("tf_install_warning", output=out[-1000:])
step("install_tf_done")

step("install_piper_deps")
rc, out = sh(f"{sys.executable} -m pip install -q soundfile numpy phonemizer einops", 300)
step("install_piper_deps_done", rc=rc)

# ---------------------------------------------------------------------------
# 3. Clone piper repo
# ---------------------------------------------------------------------------
step("clone_piper")
if PIPER_DIR.exists():
    shutil.rmtree(PIPER_DIR)
rc, out = sh("git clone --depth 1 https://github.com/rhasspy/piper.git " + str(PIPER_DIR), 300)
if rc != 0:
    step("error", msg="piper clone failed", output=out[-2000:])
    R["status"] = "failed"; save(); sys.exit(1)
step("piper_cloned")

# ---------------------------------------------------------------------------
# 4. Prepare training data (wav + txt pairs)
# ---------------------------------------------------------------------------
step("prepare_data")
TRAIN_DIR.mkdir(parents=True, exist_ok=True)

# Transcripts (we use the known transcript text for each recording)
transcripts = {
    "en": "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors. These take the shape of a long round arch, with its path high above and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at the end. People look, but nowhere ever finds it. When the man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow. Throughout the centuries people have explained the rainbow in various ways. Some have accepted it as a miracle without physical explanation. To the Hebrews it was a token that there would be no more universal floods. The Greeks used to imagine that it was a sign from the gods to foretell war or heavy rain. The Norsemen considered the rainbow as a bridge over which the gods passed from earth to their home in the sky. Others have tried to explain the phenomenon physically. Aristotle thought that the rainbow was caused by reflection of the sun's rays by the rain. Since then physicists have found that it is not reflection, but refraction by the raindrops which causes the rainbow. Many complicated ideas about the rainbow have been formed. The difference in the rainbow depends considerably upon the size of the drops, and the width of the rainbow increases as the size of the drops increases. The actual primary rainbow observed is said to be the effect of superposition of a number of bows. If the red of the second bow falls upon the green of the first, the result is to give a bow with an abnormally wide yellow band, since red and green light when mixed form yellow.",
    "sk": "Včera ráno som išiel na trh kúpiť čerstvý chlieb a mlieko. Stretol som tam starého priateľa Ľuba, ktorý predával med a syry. „Ahoj! Kde si sa túľal celé mesiace?\" spýtal sa ma s úsmevom. Porozprával som mu o svojej práci a o dlhej ceste vlakom cez hory a doliny, až ma bolel každý kĺb. Po chvíli sme si sadli na lavičku pod veľkú vŕbu a dali sme si kávu. „Môžem prísť aj zajtra?\" spýtal som sa. Okolo nás pobehovali deti a hádzali si farebnú loptu. Jedno dievčatko nieslo džbán plný malín a smialo sa na celé kolo. Vôňa čerstvého pečiva sa niesla vzduchom a ja som si spomenul na detstvo u starých rodičov. V diaľke vŕzgali brzdy električky. Ľubo mi potom ukázal fotografie zo svojej záhrady. Medzi riadkami cibule a cesnaku rástli ľalie, pivonky a neskutočne veľké tekvice. „Pozri, táto váži vyše deväť kíl!\" pochválil sa. Neveril som vlastným očiam. „Musím už ísť, ale v piatok prídem zas,\" povedal som a pobrali sme sa domov. Ešte dnes cítim chuť toho chleba s maslom a medom. Bolo to jedno z najkrajších rán tohto jesenného mesiaca. Koľko takýchto dní nám ešte život prinesie?",
}

for lang, wav_path in voice_files.items():
    # Convert m4a to wav
    wav_out = str(TRAIN_DIR / f"{lang}_0001.wav")
    rc, out = sh(f"ffmpeg -y -i {wav_path} -ac 1 -ar 22050 {wav_out}", 60)
    if rc != 0:
        step("error", msg=f"ffmpeg convert failed for {lang}", output=out[-1000:])
        continue
    
    # Write transcript
    txt_out = str(TRAIN_DIR / f"{lang}_0001.txt")
    transcript = transcripts.get(lang, "")
    Path(txt_out).write_text(transcript, encoding="utf-8")
    
    step("data_prepared", lang=lang, wav=wav_out, txt=txt_out, chars=len(transcript))

# ---------------------------------------------------------------------------
# 5. Train Piper (EN and SK) - limited steps for Kaggle time limit
# ---------------------------------------------------------------------------
def train_piper(lang, wav_dir, steps=500):
    """Train a Piper voice for one language."""
    key = f"piper_{lang}"
    step(f"{key}_start")
    
    model_out = str(OUT / f"piper_{lang}.onnx")
    config_out = str(OUT / f"piper_{lang}.onnx.json")
    
    # Piper training command
    # We use the piper train.py with a limited number of steps
    train_cmd = (
        f"cd {PIPER_DIR} && "
        f"{sys.executable} train.py "
        f"--data_root {wav_dir} "
        f"--output_dir {OUT / f'piper_{lang}_checkpoints'} "
        f"--max_steps {steps} "
        f"--save_every_n 100 "
        f"--eval_size 0 "
        f"--log_level info"
    )
    
    rc, out = sh(train_cmd, timeout=1200)
    if rc != 0:
        step(f"{key}_error", output=out[-2000:])
        R["models"][lang] = {"error": out[-500:]}
        return
    
    # Look for the exported model
    checkpoint_dir = OUT / f"piper_{lang}_checkpoints"
    onnx_files = list(checkpoint_dir.glob("*.onnx")) if checkpoint_dir.exists() else []
    
    if onnx_files:
        # Copy the latest onnx
        latest = max(onnx_files, key=lambda p: p.stat().st_mtime)
        shutil.copy2(str(latest), model_out)
        
        # Copy config if exists
        config_files = list(checkpoint_dir.glob("*.json"))
        if config_files:
            shutil.copy2(str(config_files[0]), config_out)
        
        R["models"][lang] = {
            "onnx": model_out,
            "config": config_out,
            "steps": steps,
            "device": "cuda",
        }
        step(f"{key}_done", model=model_out, steps=steps)
    else:
        step(f"{key}_no_model", output="No .onnx file found in checkpoints")
        R["models"][lang] = {"error": "No model output"}

# Train EN first
if "en" in voice_files:
    train_piper("en", str(TRAIN_DIR), steps=300)

# Train SK
if "sk" in voice_files:
    train_piper("sk", str(TRAIN_DIR), steps=300)

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
step("done", status=R["status"], models={k: {kk: vv for kk, vv in v.items() if kk != "text"} for k, v in R["models"].items()})
save()
print(f"Piper finetune complete: {R['status']}", flush=True)
