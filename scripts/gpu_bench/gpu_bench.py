#!/usr/bin/env python3
"""GPU speed bench for voice-cloning TTS: X-Voice 0.4B and OmniVoice, EN/SK/CS zero-shot.

Question it answers: what is the synthesis RTF (compute s / audio s) on THIS machine's GPU?
The Mac numbers (MPS) are RTF ~4.2 (X-Voice) and ~1 (OmniVoice); vendor claims 0.073 on an RTX 4090.

Runs anywhere with torch (CUDA, MPS or CPU). Intended for a free Kaggle/Colab T4 — see README.md.
Method (same as the Mac measurement): X-Voice RTF is marginal = (wall_long - wall_short) /
(audio_long - audio_short), so model load and CUDA init cancel. OmniVoice is timed in-process
after one warm-up call.

  python gpu_bench.py setup  --xvoice-dir X-Voice          # installs deps, downloads checkpoints
  python gpu_bench.py run    --xvoice-dir X-Voice --refs refs --out results
Refs: refs/{en,sk,cs}.wav + .txt (7 s clip + its exact transcript). Never commit them.
"""
from __future__ import annotations

import argparse, json, os, re, shutil, subprocess, sys, time

LONG = {
    "en": "Yesterday morning I went to the market to buy fresh bread and milk. I met an old friend there who was selling honey and cheese. I told him about my work and the long train journey over the hills and valleys.",
    "sk": "Včera ráno som išiel na trh kúpiť čerstvý chlieb a mlieko. Stretol som tam starého priateľa Ľuba, ktorý predával med a syry. Porozprával som mu o svojej práci a o dlhej ceste vlakom cez hory a doliny.",
    "cs": "Včera ráno jsem šel na trh koupit čerstvý chléb a mléko. Potkal jsem tam starého přítele Luboše, který prodával med a sýry. Povídal jsem mu o své práci a o dlouhé cestě vlakem přes hory a údolí.",
}
SHORT = {"en": "Good morning.", "sk": "Dobrý deň.", "cs": "Dobrý den."}

PKGS = ("vocos x_transformers hydra-core ema_pytorch safetensors librosa soundfile cached_path phonemizer "
        "num2words pydub pyloudnorm tomli torchdiffeq unidecode huggingface_hub transformers click pyphen "
        "regex addict accelerate matplotlib tqdm omegaconf torchcodec").split()


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=isinstance(cmd, str), text=True, **kw)


def setup(a):
    xv = os.path.abspath(a.xvoice_dir)
    if not os.path.isdir(xv):
        sh(["git", "clone", "--depth", "1", "https://github.com/sunnyxrxrx/X-Voice.git", xv], check=True)
    sh("apt-get install -y -q espeak-ng ffmpeg >/dev/null 2>&1 || true")  # Linux/Colab; harmless elsewhere
    sh([sys.executable, "-m", "pip", "install", "-q", "Cython", "setuptools", "wheel"])
    sh([sys.executable, "-m", "pip", "install", "-q", "-e", xv, "--no-deps", "--no-build-isolation"], check=True)
    sh([sys.executable, "-m", "pip", "install", "-q", *PKGS])
    # X-Voice imports CJK/Thai/etc. G2P libs unconditionally. Only EN/SK/CS are used here, so a lib that
    # fails to build (e.g. python-mecab-ko on Python 3.13) is replaced by an inert stub module.
    import site
    sp = site.getsitepackages()[0]
    stub = ("class _A:\n    def __init__(self,*a,**k): pass\n    def __call__(self,*a,**k): return self\n"
            "    def __getattr__(self,n): return _A()\ndef __getattr__(name): return _A()\n")
    def get(mod):
        pkg = {"cv2": "opencv-python", "df": "deepfilternet"}.get(mod, mod)
        r = sh([sys.executable, "-m", "pip", "install", "-q", pkg], capture_output=True)
        if r.returncode != 0:
            print("stubbing", mod)
            open(os.path.join(sp, mod + ".py"), "w").write(stub)
    for mod in "fastlid fasttext jieba pypinyin wandb datasets pythainlp pykakasi finnsyll epitran pyopenjtalk g2pk".split():
        get(mod)
    for _ in range(20):
        r = sh([sys.executable, "-m", "x_voice.infer.infer_cli_stage1", "--help"], capture_output=True, cwd=xv)
        m = re.search(r"No module named '([A-Za-z0-9_]+)", r.stderr or "")
        if not m:
            print("x_voice import chain OK" if r.returncode == 0 else r.stderr[-800:])
            break
        print("need", m.group(1))
        get(m.group(1))
    from huggingface_hub import snapshot_download
    snapshot_download("XRXRX/X-Voice", local_dir=os.path.join(xv, "ckpts_hf"))
    snapshot_download("charactr/vocos-mel-24khz", local_dir=os.path.join(xv, "my_vocoder/vocos-mel-24khz"))
    sh([sys.executable, "-m", "pip", "install", "-q", "omnivoice"])
    print("setup done")


def xvoice_run(xv, ref_wav, ref_txt, lang, text, out_wav):
    import soundfile as sf
    tpl = open(os.path.join(xv, "src/x_voice/infer/examples/basic/basic_stage1.toml"), encoding="utf-8").read()
    cfg = {"ckpt_file": "ckpts_hf/XVoice_Base_Stage1/model_600000.safetensors",
           "vocab_file": "ckpts_hf/XVoice_Base_Stage1/vocab.txt", "ref_audio": os.path.abspath(ref_wav),
           "ref_text": ref_txt, "gen_text": text, "ref_lang": lang, "gen_lang": lang,
           "auto_detect_lang": "false", "output_dir": "out", "output_file": os.path.basename(out_wav)}
    for k, v in cfg.items():
        val = v if v in ("false", "true") else json.dumps(v, ensure_ascii=False)
        tpl = re.sub(rf"^{k} = .*$", lambda m: f"{k} = {val}", tpl, flags=re.M)
    tpl = tpl.replace("load_vocoder_from_local = true", "load_vocoder_from_local = false")
    toml = os.path.join(xv, "_bench.toml")
    open(toml, "w", encoding="utf-8").write(tpl)
    t0 = time.perf_counter()
    r = sh([sys.executable, "-m", "x_voice.infer.infer_cli_stage1", "-c", toml], capture_output=True, cwd=xv)
    wall = time.perf_counter() - t0
    wav = os.path.join(xv, "out", os.path.basename(out_wav))
    if r.returncode != 0 or not os.path.exists(wav):
        return {"error": (r.stderr or "")[-600:]}
    info = sf.info(wav)
    return {"wall_s": round(wall, 2), "audio_s": round(info.duration, 2), "wav": wav}


def bench_xvoice(a, langs, outdir):
    xv = os.path.abspath(a.xvoice_dir)
    rows = []
    for lang in langs:
        ref = os.path.join(a.refs, f"{lang}.wav")
        txt = open(os.path.join(a.refs, f"{lang}.txt"), encoding="utf-8").read().strip()
        xvoice_run(xv, ref, txt, lang, SHORT[lang], f"warm_{lang}.wav")  # warm HF/disk cache, discard
        s = xvoice_run(xv, ref, txt, lang, SHORT[lang], f"short_{lang}.wav")
        l = xvoice_run(xv, ref, txt, lang, LONG[lang], f"xvoice_{lang}.wav")
        row = {"engine": "xvoice", "lang": lang, "short": {k: v for k, v in s.items() if k != "wav"},
               "long": {k: v for k, v in l.items() if k != "wav"}}
        if "error" not in s and "error" not in l and l["audio_s"] > s["audio_s"]:
            row["marginal_rtf"] = round((l["wall_s"] - s["wall_s"]) / (l["audio_s"] - s["audio_s"]), 3)
            shutil.copy(l["wav"], os.path.join(outdir, f"xvoice_{lang}.wav"))
        rows.append(row)
        print(row, flush=True)
    return rows


def bench_omni(a, langs, outdir):
    import torch, soundfile as sf
    from omnivoice import OmniVoice
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=dev, dtype=dtype)
    rows = []
    for lang in langs:
        ref = os.path.join(a.refs, f"{lang}.wav")
        txt = open(os.path.join(a.refs, f"{lang}.txt"), encoding="utf-8").read().strip()
        prompt = model.create_voice_clone_prompt(ref, txt)
        model.generate(text=SHORT[lang], language=lang, voice_clone_prompt=prompt, num_step=16)  # warm-up
        for tag, text in (("short", SHORT[lang]), ("long", LONG[lang])):
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            audio = model.generate(text=text, language=lang, voice_clone_prompt=prompt, num_step=16)[0]
            if dev == "cuda":
                torch.cuda.synchronize()
            syn = time.perf_counter() - t0
            dur = len(audio) / model.sampling_rate
            if tag == "long":
                sf.write(os.path.join(outdir, f"omni_{lang}.wav"), audio, model.sampling_rate)
            rows.append({"engine": "omnivoice", "lang": lang, "text": tag, "device": dev, "dtype": str(dtype),
                         "syn_s": round(syn, 3), "audio_s": round(dur, 2), "rtf": round(syn / dur, 3)})
            print(rows[-1], flush=True)
    return rows


def run(a):
    import torch
    os.makedirs(a.out, exist_ok=True)
    langs = a.langs.split(",")
    gpu = {"torch": torch.__version__, "cuda": torch.cuda.is_available(),
           "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())}
    if gpu["cuda"]:
        p = torch.cuda.get_device_properties(0)
        gpu.update({"name": p.name, "vram_gb": round(p.total_memory / 1e9, 1)})
    print(gpu)
    res = {"date": time.strftime("%Y-%m-%d %H:%M"), "machine": gpu, "results": []}
    if "xvoice" in a.engines:
        res["results"] += bench_xvoice(a, langs, a.out)
    if "omni" in a.engines:
        res["results"] += bench_omni(a, langs, a.out)
    json.dump(res, open(os.path.join(a.out, "gpu_bench.json"), "w"), indent=2, ensure_ascii=False)
    print("wrote", os.path.join(a.out, "gpu_bench.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["setup", "run"])
    ap.add_argument("--xvoice-dir", default="X-Voice")
    ap.add_argument("--refs", default="refs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--langs", default="en,sk,cs")
    ap.add_argument("--engines", default="xvoice,omni")
    a = ap.parse_args()
    {"setup": setup, "run": run}[a.cmd](a)
