#!/usr/bin/env python3
"""Private T4 run: does incremental streaming beat short segments when STT runs on GPU?

Real server (/ws, faster-whisper small.en on CUDA fp16 -> Opus-MT en-sk -> generic Piper SK, CPU),
90 s of me_en_rainbow paced at 1x, server restarted per run. Runs A (pack defaults), B (BP_INCREMENTAL=1),
C (incremental, fast knobs), D (= A again, variance). No voice cloning, no listening grade.
Deviations in the throwaway copy only: (1) main.py 'INCR step' log line gets dt= (GPU STT step duration);
(2) requirements relaxed for Kaggle Python 3.13 (SQLAlchemy>=2.0.36); (3) the paced client's config_update is
extended in-process with stt_model_size (paced_replay.py untouched).
"""
import hashlib, importlib.metadata as md, json, os, re, shutil, signal, socket, statistics, subprocess, sys
import time, traceback, urllib.request
from pathlib import Path

TOTAL_CAP_S = 1500
HARD_ALARM_S = 1440
PORT = 8013
STT_SIZE = "large-v3-turbo"
SECONDS = 45
OUT = Path("/kaggle/working")
CODE = OUT / "bp"
REPORT = OUT / "conc-report.json"
T0 = time.time()
RUNS = [("c1", {"BP_SILENCE_RMS": "0"}, "en", "sk", "clip_en.wav", 1), ("c2", {"BP_SILENCE_RMS": "0"}, "en", "sk", "clip_en.wav", 2),
        ("c4", {"BP_SILENCE_RMS": "0"}, "en", "sk", "clip_en.wav", 4), ("c8", {"BP_SILENCE_RMS": "0"}, "en", "sk", "clip_en.wav", 8)]
R = {"schema": 1, "status": "running", "steps": [], "errors": [], "runs": {}, "deviations": [], "acceptance": "route_unverified"}


def save():
    R["elapsed_s"] = round(time.time() - T0, 1)
    tmp = REPORT.with_suffix(".tmp")
    tmp.write_text(json.dumps(R, indent=1, ensure_ascii=False))
    tmp.replace(REPORT)


def step(name, **kw):
    R["steps"].append({"t": round(time.time() - T0, 1), "step": name, **kw})
    print(f"[{time.time() - T0:7.1f}s] {name} {kw}", flush=True)
    save()


def sh(cmd, timeout, **kw):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)
    return p.returncode, (p.stdout + p.stderr)[-3000:]


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


def port_free():
    try:
        socket.create_connection(("127.0.0.1", PORT), 1).close()
        return False
    except Exception:
        return True


def main():
    if not Path("/kaggle/input").is_dir():
        raise RuntimeError("Kaggle-only")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("job cap")))
    signal.alarm(HARD_ALARM_S)
    server = None
    try:
        srcdirs = [a.parent for a in Path("/kaggle/input").rglob("app.py") if (a.parent / "backend").is_dir()]
        clips = {n: next(Path("/kaggle/input").rglob(n)) for n in ("clip_en.wav", "clip_sk.wav", "clip_cs.wav")}
        voices = [next(Path("/kaggle/input").rglob(n)) for n in ("sk_SK-lili-medium.onnx", "cs_CZ-jirka-medium.onnx", "en_US-john-medium.onnx")]
        srcdirs = [a.parent for a in Path("/kaggle/input").rglob("app.py") if (a.parent / "backend").is_dir()]
        assert len(srcdirs) == 1, srcdirs
        shutil.copytree(srcdirs[0], CODE)
        pm = CODE / "backend/tts/piper_models"
        pm.mkdir(parents=True, exist_ok=True)
        for v in voices:
            shutil.copy(v, pm); shutil.copy(str(v) + ".json", pm)
        R["tts_voice"] = "generic public Piper voices sk lili / cs jirka / en john (personal fine-tunes not uploaded)"
        R["input_audio_sha256"] = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in clips.items()}
        step("copied code", src=str(srcdirs[0]))

        drop = re.compile(r"^(numpy|scipy|transformers|tokenizers|huggingface-hub|soundfile|numba|onnxruntime|librosa|SQLAlchemy)\b", re.I)
        lines = [re.sub(r"\s+#.*$", "", l).strip() for l in (CODE / "requirements.txt").read_text().splitlines()]
        lines = [l for l in lines if l and not l.startswith("#") and not drop.match(l)]
        lines += ["transformers", "sentencepiece", "onnxruntime", "SQLAlchemy>=2.0.36", "sacrebleu", "soundfile", "scipy"]
        R["deviations"].append("requirements relaxed for Python 3.13; SQLAlchemy>=2.0.36")
        rf = OUT / "req-filtered.txt"
        rf.write_text("\n".join(lines))
        rc, o = sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(rf)], 420)
        step("pip requirements", rc=rc, tail=o[-300:])
        assert rc == 0
        import torch
        gpu = torch.cuda.get_device_name(0)
        assert "T4" in gpu, gpu
        R["gpu"] = gpu
        R["python"] = sys.version
        R["versions"] = {}
        for n in ["torch", "transformers", "faster-whisper", "ctranslate2", "fastapi", "uvicorn", "websockets", "numpy",
                  "scipy", "onnxruntime", "piper-tts", "SQLAlchemy", "sacrebleu"]:
            try:
                R["versions"][n] = md.version(n)
            except Exception:
                R["versions"][n] = None

        import site
        libs = [str(p) for sp in site.getsitepackages() for p in Path(sp).glob("nvidia/*/lib")]
        env0 = dict(os.environ)
        env0["LD_LIBRARY_PATH"] = ":".join(libs + [env0.get("LD_LIBRARY_PATH", "")])
        env0.update({"BP_SK_STT_MODEL": STT_SIZE, "BP_NO_SSL": "1", "BP_PORT": str(PORT), "PYTHONUNBUFFERED": "1"})
        for k in [k for k in env0 if k.startswith("BP_INCR") or k in ("BP_VAD_SILENCE_S", "BP_VAD_MAX_SEGMENT_S", "BP_BARGE_IN")]:
            env0.pop(k)

        for pair in ("en-sk", "en-cs", "sk-en", "cs-en"):
            rc, o = sh([sys.executable, "-c",
                        "import sys; sys.setrecursionlimit(2000); import backend.mt.convert_opus_mt_to_ct2 as c; "
                        f"c.convert_model('Helsinki-NLP/opus-mt-{pair}', 'ct2_models/Helsinki-NLP--opus-mt-{pair}', quantization='int8')"],
                       300, cwd=CODE, env=env0)
            ok = (CODE / f"ct2_models/Helsinki-NLP--opus-mt-{pair}/model.bin").is_file()
            step("convert " + pair, rc=rc, ok=ok, tail=o[-200:] if not ok else "")
            R.setdefault("mt_ok", {})[pair] = ok
        rc, o = sh([sys.executable, "-c", f"from faster_whisper.utils import download_model as d; print(d('{STT_SIZE}'))"], 300, env=env0)
        step("stt download", rc=rc, tail=o[-200:])
        assert rc == 0
        smoke = ("from faster_whisper import WhisperModel; import sys; m=WhisperModel(sys.argv[1],device='cuda',compute_type='float16'); "
                 "s,_=m.transcribe(sys.argv[2], language='en'); print('SMOKE', ' '.join(x.text for x in s)[:80])")
        rc, o = sh([sys.executable, "-c", smoke, STT_SIZE, str(clips["clip_en.wav"])], 240, env=env0)
        stt_cuda = rc == 0 and "SMOKE" in o
        env0["BP_DEVICE"] = "cuda" if stt_cuda else "cpu"
        env0["BP_COMPUTE_TYPE"] = "float16" if stt_cuda else "int8"
        R["stt"] = {"model": STT_SIZE, "device": env0["BP_DEVICE"], "compute_type": env0["BP_COMPUTE_TYPE"], "smoke_tail": o[-200:]}
        mts = ("import sys; sys.path.insert(0,'.'); from backend.mt.ctranslate2_mt import CTranslate2MT as M; "
               "m=M(model_path='ct2_models/Helsinki-NLP--opus-mt-en-sk', device='cuda'); print('MTSMOKE', m.translate('Hello world.','en','sk'))")
        rc, o = sh([sys.executable, "-c", mts], 120, cwd=CODE, env=env0)
        mt_cuda = rc == 0 and "MTSMOKE" in o
        env0["BP_MT_DEVICE"] = "cuda" if mt_cuda else "cpu"
        R["mt"] = {"device": env0["BP_MT_DEVICE"], "compute_type": "int8", "smoke_tail": o[-300:]}
        step("devices", stt=R["stt"]["device"], mt=R["mt"]["device"])

        for label, extra_env, src, tgt, clip, nclients in RUNS:
            left = TOTAL_CAP_S - (time.time() - T0)
            if left < 260:
                R["runs"][label] = {"status": "skipped_cap", "remaining_s": round(left)}
                continue
            env = {**env0, **extra_env}
            logp = OUT / f"server-{label}.log"
            assert port_free(), "port busy"
            server = subprocess.Popen([sys.executable, "app.py"], cwd=CODE, env=env, stdout=open(logp, "w"), stderr=subprocess.STDOUT)
            t1 = time.time()
            while time.time() - t1 < 120:
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=3).read(10)
                    break
                except Exception:
                    assert server.poll() is None, "server exited"
                    time.sleep(2)
            else:
                raise RuntimeError("server not ready")
            procs = []
            for ci in range(nclients):
                oj = OUT / f"paced-{label}-{ci}.json"
                cmd = [sys.executable, "-c", CHILD, str(CODE), json.dumps({"stt_model_size": STT_SIZE}), str(clips[clip]),
                       "--base", f"http://127.0.0.1:{PORT}", "--src", src, "--tgt", tgt, "--tts", "piper",
                       "--seconds", str(SECONDS), "--rates", "1.0", "--load-timeout", "300", "--out", str(oj)]
                procs.append((oj, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)))
            ok = 0; errs = []
            for oj, pr in procs:
                try:
                    o_, e_ = pr.communicate(timeout=max(60, min(left - 60, 360)))
                except subprocess.TimeoutExpired:
                    pr.kill(); errs.append("timeout"); continue
                if pr.returncode == 0 and oj.exists(): ok += 1
                else: errs.append(e_[-200:])
            lags, voiced, segs, fw = [], 0, 0, []
            for oj, _ in procs:
                if oj.exists():
                    v = json.loads(oj.read_text())["rates"]["1.0"]
                    lags += v["lag_s"]; voiced += v["with_audio"]; segs += v["segments"]; fw.append(v["first_word_lag_max_s"])
            lags.sort()
            R["runs"][label] = {"status": "complete" if ok == nclients else "failed", "clients": nclients, "clients_ok": ok,
                                "wall_s": round(time.time() - t1, 1), "segments": segs, "voiced": voiced,
                                "lag_median_s": lags[len(lags)//2] if lags else None,
                                "lag_p95_s": lags[int(0.95*(len(lags)-1))] if lags else None,
                                "lag_max_s": max(lags) if lags else None, "first_word_max_s": max(fw) if fw else None,
                                "errors": errs[:2]}
            server.terminate()
            try:
                server.wait(15)
            except Exception:
                server.kill()
            server = None
            time.sleep(2)
            step("run " + label, **{k: R["runs"][label].get(k) for k in ("status", "clients_ok", "lag_median_s", "lag_p95_s")})

        done = [l for l, *_ in RUNS if R["runs"].get(l, {}).get("status") == "complete"]
        R["status"] = "complete" if done and all(R["runs"][l]["status"] in ("complete", "skipped_cap") for l in R["runs"]) else "incomplete"
    except BaseException:
        R["status"] = "incomplete"
        R["errors"].append(traceback.format_exc())
    finally:
        signal.alarm(0)
        if server is not None:
            server.terminate()
        shutil.rmtree(CODE / "ct2_models", ignore_errors=True)
        shutil.rmtree(CODE / "backend/tts/piper_models", ignore_errors=True)
        save()
    return 0 if R["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
