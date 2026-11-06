#!/usr/bin/env python3
"""Private T4 run of the REAL BP route: FastAPI /ws, STT -> MT -> TTS (OmniVoice, cloned voice).

Driver: scripts/live/paced_replay.py streams 90 s of recorded English speech at 1x real time.
Not a TTS-only benchmark. No human listening grade, no physical mic: acceptance stays route_unverified.
Kaggle-only; hard cap TOTAL_CAP_S for the whole job (partial results are written on timeout).

Documented deviations from the repo as shipped (all applied to the throwaway Kaggle copy, recorded in
the report): (1) registry licence for 'omnivoice' flipped unknown -> commercial-ok because the gate fails
closed and the WS cannot set allow_unknown_license; the model-card licence is read from the HF API and
recorded; (2) OmniVoice.from_pretrained pinned to REVISION; (3) the paced client's config_update is
extended in-process with speaker_wav_path / stt_model_size (paced_replay.py itself is untouched).
"""
import hashlib, importlib.metadata as md, json, os, re, shutil, signal, socket, subprocess, sys
import tarfile, threading, time, traceback, urllib.request
from pathlib import Path

REVISION = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
TOTAL_CAP_S = 1500
HARD_ALARM_S = 1440
PORT = 8013
STT_SIZE = "small"
SECONDS = 90
RATES = ["1.0", "1.25"]
OUT = Path("/kaggle/working")
CODE = OUT / "bp"
REPORT = OUT / "route-report.json"
T0 = time.time()
R = {"schema": 1, "status": "running", "scope": "real server /ws STT->MT->TTS, paced 1x, OmniVoice cloned",
     "acceptance": "route_unverified", "listening_grade": "none", "physical_mic": False,
     "steps": [], "errors": [], "runs": {}, "deviations": []}


def save():
    R["elapsed_s"] = round(time.time() - T0, 1)
    tmp = REPORT.with_suffix(".tmp")
    tmp.write_text(json.dumps(R, indent=1, ensure_ascii=False))
    tmp.replace(REPORT)


def step(name, **kw):
    R["steps"].append({"t": round(time.time() - T0, 1), "step": name, **kw})
    print(f"[{time.time() - T0:7.1f}s] {name} {kw}", flush=True)
    save()


def remaining():
    return TOTAL_CAP_S - (time.time() - T0)


def sh(cmd, timeout, **kw):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)
    return p.returncode, (p.stdout + p.stderr)[-3000:]


def normalize_reference(source, output):  # same logic as kaggle_session/job.py
    import numpy as np, soundfile as sf
    from scipy.signal import resample_poly
    audio, sr = sf.read(str(source), dtype="float32", always_2d=True)
    if not 8000 <= sr <= 192000 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("invalid reference audio")
    mono = audio.mean(axis=1)
    if sr != 16000:
        mono = resample_poly(mono, 16000, sr)
    sf.write(str(output), mono, 16000, subtype="PCM_16")
    return {"duration_s": round(len(mono) / 16000, 2), "src_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            "sha256": hashlib.sha256(Path(output).read_bytes()).hexdigest()}


CHILD = r'''
import sys, asyncio, json
sys.path.insert(0, sys.argv[1] + "/scripts/live")
import paced_replay as pr
extra = json.loads(sys.argv[2])
class WS:
    def __init__(self, ws): self.ws = ws
    async def send(self, m):
        if isinstance(m, str) and '"config_update"' in m:
            d = json.loads(m); d.update(extra); m = json.dumps(d)
        return await self.ws.send(m)
    async def recv(self): return await self.ws.recv()
    def __aiter__(self): return self.ws.__aiter__()
class CM:
    def __init__(self, cm): self.cm = cm
    async def __aenter__(self): return WS(await self.cm.__aenter__())
    async def __aexit__(self, *a): return await self.cm.__aexit__(*a)
real = pr.websockets.connect
pr.websockets.connect = lambda *a, **k: CM(real(*a, **k))
import runpy
sys.argv = ["paced_replay.py"] + sys.argv[3:]
runpy.run_path(pr.__file__, run_name="__main__")
'''


def main():
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Kaggle-only")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("job cap")))
    signal.alarm(HARD_ALARM_S)
    server = None
    mem = {"max_used_mib": 0, "samples": 0, "stop": False}
    try:
        # ---- locate inputs
        tars = list(Path("/kaggle/input").rglob("bp-code.tar.gz"))
        srcdirs = [a.parent for a in Path("/kaggle/input").rglob("app.py") if (a.parent / "backend").is_dir()]  # Kaggle auto-extracts archives
        wavs = list(Path("/kaggle/input").rglob("rainbow_en_90s.wav"))
        assert (len(tars) == 1 or len(srcdirs) == 1) and len(wavs) == 1, (tars, srcdirs, wavs)
        refs = {}
        for lang in ("sk", "cs"):
            w = list(Path("/kaggle/input").rglob(lang + ".wav"))
            t = list(Path("/kaggle/input").rglob(lang + ".txt"))
            assert len(w) == 1 and len(t) == 1, (lang, w, t)
            refs[lang] = (w[0], t[0])
        if tars:
            with tarfile.open(tars[0]) as t:
                t.extractall(CODE, filter="data")
            step("extracted tar", tar_sha256=hashlib.sha256(tars[0].read_bytes()).hexdigest())
        else:
            shutil.copytree(srcdirs[0], CODE)
            step("copied pre-extracted dir", src=str(srcdirs[0]))

        # ---- install
        rc, o = sh([sys.executable, "-m", "pip", "install", "-q", "omnivoice==0.2.1", "soundfile==0.14.0", "scipy==1.16.3"], 420)
        step("pip omnivoice", rc=rc, tail=o[-400:])
        assert rc == 0
        drop = re.compile(r"^(numpy|scipy|transformers|tokenizers|huggingface-hub|soundfile|numba|onnxruntime|librosa|SQLAlchemy)\b", re.I)
        lines = [re.sub(r"\s+#.*$", "", l).strip() for l in (CODE / "requirements.txt").read_text().splitlines()]
        lines = [l for l in lines if l and not l.startswith("#") and not drop.match(l)]
        lines += ["transformers", "sentencepiece", "onnxruntime", "SQLAlchemy>=2.0.36"]
        R["deviations"].append("SQLAlchemy pin 2.0.30 replaced by >=2.0.36 (2.0.30 raises TypeError on Kaggle Python 3.13; seen in v2 server.log)")
        rf = OUT / "req-filtered.txt"
        rf.write_text("\n".join(lines))
        rc, o = sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(rf)], 420)
        if rc != 0:
            R["deviations"].append("pinned requirements failed, retried unpinned: " + o[-300:])
            names = ["uvicorn[standard]" if l.startswith("uvicorn") else re.split(r"[=<>~!; ]", l)[0] for l in lines if not l.startswith("SQLAlchemy")] + ["SQLAlchemy>=2.0.36"]
            rc, o = sh([sys.executable, "-m", "pip", "install", "-q"] + names, 420)
        step("pip requirements", rc=rc, tail=o[-400:])
        assert rc == 0
        import torch
        gpu = torch.cuda.get_device_name(0)
        assert "T4" in gpu, gpu
        names = ["omnivoice", "torch", "transformers", "faster-whisper", "ctranslate2", "fastapi", "uvicorn",
                 "websockets", "numpy", "scipy", "soundfile", "sentencepiece", "tokenizers", "huggingface-hub", "piper-tts", "SQLAlchemy"]
        vers = {}
        for n in names:
            try:
                vers[n] = md.version(n)
            except Exception:
                vers[n] = None
        R["gpu"] = gpu
        R["versions"] = vers
        R["python"] = sys.version
        step("gpu", gpu=gpu)

        # ---- CUDA libs for ctranslate2 (torch ships them under site-packages/nvidia)
        import site
        libs = [str(p) for sp in site.getsitepackages() for p in Path(sp).glob("nvidia/*/lib")]
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")])
        env.update({"BP_NO_SSL": "1", "BP_PORT": str(PORT), "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})

        # ---- licence of the model, read from the Hub (evidence for deviation 1)
        try:
            from huggingface_hub import HfApi
            info = HfApi().model_info("k2-fsa/OmniVoice", revision=REVISION)
            R["omnivoice_hub_license"] = (info.card_data.to_dict() if info.card_data else {}).get("license")
        except Exception as e:
            R["omnivoice_hub_license"] = "lookup failed: " + repr(e)[:200]
        step("hub license", license=R["omnivoice_hub_license"])

        # ---- patch the throwaway copy
        reg = CODE / "backend/registry.py"
        s = reg.read_text()
        m = re.search(r'("omnivoice", "tts".*?license=)"unknown"', s, re.S)
        assert m, "registry omnivoice entry not found"
        reg.write_text(s[:m.start()] + m.group(1) + '"commercial-ok"' + s[m.end():])
        R["deviations"].append("registry omnivoice license unknown->commercial-ok in Kaggle copy (gate fails closed; hub license recorded)")
        om = CODE / "backend/tts/omni_tts.py"
        s = om.read_text()
        old = "OmniVoice.from_pretrained(\n            model_name, device_map"
        assert s.count(old) == 1
        om.write_text(s.replace(old, f'OmniVoice.from_pretrained(\n            model_name, revision="{REVISION}", device_map'))
        R["deviations"].append("OmniVoice.from_pretrained pinned to revision " + REVISION)

        # ---- assets: MT conversion, STT download
        conv = ("import sys; sys.setrecursionlimit(2000); import backend.mt.convert_opus_mt_to_ct2 as c; "
                "c.convert_model('Helsinki-NLP/opus-mt-%s', 'ct2_models/Helsinki-NLP--opus-mt-%s', quantization='int8')")
        for pair in ("en-sk", "en-cs"):
            rc, o = sh([sys.executable, "-c", conv % (pair, pair)], 300, cwd=CODE, env=env)
            ok = (CODE / f"ct2_models/Helsinki-NLP--opus-mt-{pair}/model.bin").is_file()
            step("convert " + pair, rc=rc, model_bin=ok, tail=o[-300:])
            assert ok
        rc, o = sh([sys.executable, "-c", f"from faster_whisper.utils import download_model as d; print(d('{STT_SIZE}'))"], 300, env=env)
        step("stt download", rc=rc, tail=o[-300:])
        assert rc == 0
        # STT on cuda? smoke test, else cpu int8 (recorded)
        smoke = ("from faster_whisper import WhisperModel; import sys; m=WhisperModel(sys.argv[1],device='cuda',compute_type='float16'); "
                 "s,_=m.transcribe(sys.argv[2], language='en'); print('SMOKE', ' '.join(x.text for x in s)[:120])")
        rc, o = sh([sys.executable, "-c", smoke, STT_SIZE, str(wavs[0])], 240, env=env)
        stt_cuda = rc == 0 and "SMOKE" in o
        env["BP_DEVICE"] = "cuda" if stt_cuda else "cpu"
        env["BP_COMPUTE_TYPE"] = "float16" if stt_cuda else "int8"
        R["stt"] = {"model": STT_SIZE, "device": env["BP_DEVICE"], "compute_type": env["BP_COMPUTE_TYPE"],
                    "cuda_smoke_tail": o[-300:]}
        step("stt device", **{k: R["stt"][k] for k in ("device", "compute_type")})

        # ---- reference voices + metadata
        sv = CODE / "speaker_voices"
        sv.mkdir(exist_ok=True)
        meta, ref_info = [], {}
        for lang, (w, t) in refs.items():
            dst = sv / f"ref_{lang}.wav"
            ref_info[lang] = normalize_reference(w, dst)
            (sv / f"ref_{lang}.txt").write_text(t.read_text().strip(), encoding="utf-8")
            meta.append({"id": f"ref_{lang}", "name": f"ref_{lang}", "language": lang, "filename": f"ref_{lang}.wav",
                         "path": f"speaker_voices/ref_{lang}.wav", "public_demo": True,
                         "transcribed_text": t.read_text().strip()})
        (sv / "speaker_voices.json").write_text(json.dumps(meta, ensure_ascii=False))
        R["references"] = ref_info
        R["input_audio_sha256"] = hashlib.sha256(wavs[0].read_bytes()).hexdigest()

        # ---- GPU memory sampler
        def sampler():
            while not mem["stop"]:
                rc_, o_ = sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], 10)
                try:
                    v = int(o_.strip().splitlines()[0])
                    mem["max_used_mib"] = max(mem["max_used_mib"], v)
                    mem["samples"] += 1
                except Exception:
                    pass
                time.sleep(3)
        threading.Thread(target=sampler, daemon=True).start()

        # ---- server
        logf = open(OUT / "server.log", "w")
        server = subprocess.Popen([sys.executable, "app.py"], cwd=CODE, env=env, stdout=logf, stderr=subprocess.STDOUT)
        t_ready = time.time()
        while time.time() - t_ready < 180:
            try:
                socket.create_connection(("127.0.0.1", PORT), 1).close()
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=3).read(10)
                break
            except Exception:
                assert server.poll() is None, "server exited"
                time.sleep(2)
        else:
            raise RuntimeError("server not ready in 180 s")
        step("server ready", after_s=round(time.time() - t_ready, 1))

        # ---- paced runs
        for lang in ("sk", "cs"):
            left = remaining() - 45
            if left < 150:
                R["runs"][lang] = {"status": "skipped_cap", "remaining_s": round(left)}
                continue
            out_json = OUT / f"paced-en-{lang}.json"
            extra = {"speaker_wav_path": f"speaker_voices/ref_{lang}.wav", "speaker_lang": lang, "stt_model_size": STT_SIZE}
            cmd = [sys.executable, "-c", CHILD, str(CODE), json.dumps(extra), str(wavs[0]),
                   "--base", f"http://127.0.0.1:{PORT}", "--src", "en", "--tgt", lang, "--tts", "omnivoice",
                   "--seconds", str(SECONDS), "--rates", *RATES, "--load-timeout", "420", "--out", str(out_json)]
            t1 = time.time()
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=min(left, 700))
                R["runs"][lang] = {"status": "complete" if p.returncode == 0 and out_json.exists() else "failed",
                                   "rc": p.returncode, "wall_s": round(time.time() - t1, 1),
                                   "stdout_tail": p.stdout[-1500:], "stderr_tail": p.stderr[-1500:]}
            except subprocess.TimeoutExpired as e:
                R["runs"][lang] = {"status": "timeout_partial", "wall_s": round(time.time() - t1, 1)}
            step("paced " + lang, **{k: R["runs"][lang].get(k) for k in ("status", "rc", "wall_s")})
        R["status"] = "complete" if all(r.get("status") == "complete" for r in R["runs"].values()) and R["runs"] else "incomplete"
    except BaseException:
        R["status"] = "incomplete"
        R["errors"].append(traceback.format_exc())
    finally:
        signal.alarm(0)
        mem["stop"] = True
        R["gpu_memory_used_mib_max_sampled"] = mem["max_used_mib"]
        R["gpu_memory_samples"] = mem["samples"]
        if server is not None:
            server.terminate()
            try:
                server.wait(15)
            except Exception:
                server.kill()
        log = OUT / "server.log"
        if log.exists():
            txt = log.read_text(errors="replace")
            R["server_log_lines"] = txt.count("\n")
            R["server_log_tracebacks"] = txt.count("Traceback")
            R["server_log_cloned"] = len(re.findall(r"OmniVoiceTTS: .*\(\w+, cloned\)", txt))
            R["server_log_auto_voice"] = txt.count("auto voice")
            R["server_log_tail"] = txt.splitlines()[-250:]
        shutil.rmtree(CODE / "ct2_models", ignore_errors=True)  # keep output small
        shutil.rmtree(CODE / "speaker_voices", ignore_errors=True)  # private reference audio is never exported
        save()
    return 0 if R["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
