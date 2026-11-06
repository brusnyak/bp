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
SECONDS = 60
OUT = Path("/kaggle/working")
CODE = OUT / "bp"
REPORT = OUT / "lang-report.json"
T0 = time.time()
RUNS = [("ensk", {"BP_SILENCE_RMS": "0"}, "en", "sk", "clip_en.wav"),
        ("encs", {"BP_SILENCE_RMS": "0"}, "en", "cs", "clip_en.wav"),
        ("sken", {"BP_SILENCE_RMS": "0"}, "sk", "en", "clip_sk.wav"),
        ("csen", {"BP_SILENCE_RMS": "0"}, "cs", "en", "clip_cs.wav"),
        ("sken_gate", {}, "sk", "en", "clip_sk.wav")]
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

        for label, extra_env, src, tgt, clip in RUNS:
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
            out_json = OUT / f"paced-{label}.json"
            cmd = [sys.executable, "-c", CHILD, str(CODE), json.dumps({"stt_model_size": STT_SIZE}), str(clips[clip]),
                   "--base", f"http://127.0.0.1:{PORT}", "--src", src, "--tgt", tgt, "--tts", "piper",
                   "--seconds", str(SECONDS), "--rates", "1.0", "--load-timeout", "300", "--out", str(out_json)]
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=min(left - 60, 330))
                st = "complete" if p.returncode == 0 and out_json.exists() else "failed"
                R["runs"][label] = {"status": st, "env": extra_env, "rc": p.returncode, "wall_s": round(time.time() - t1, 1),
                                    "stderr_tail": p.stderr[-800:]}
            except subprocess.TimeoutExpired:
                R["runs"][label] = {"status": "timeout", "env": extra_env, "wall_s": round(time.time() - t1, 1)}
            server.terminate()
            try:
                server.wait(15)
            except Exception:
                server.kill()
            server = None
            time.sleep(2)
            txt = logp.read_text(errors="replace")
            dts = [float(x) for x in re.findall(r"INCR step: dt=([0-9.]+)", txt)]
            R["runs"][label].update({
                "log_tracebacks": txt.count("Traceback"), "incr_step_errors": txt.count("incremental step error"),
                "incr_steps": len(dts), "incr_dt_median_s": round(statistics.median(dts), 3) if dts else None,
                "incr_dt_p95_s": round(sorted(dts)[int(0.95 * (len(dts) - 1))], 3) if dts else None,
                "incr_dt_max_s": max(dts) if dts else None,
                "log_stt_device_lines": [l[-160:] for l in txt.splitlines() if "STT" in l and "created" in l][:2]})
            step("run " + label, **{k: R["runs"][label].get(k) for k in ("status", "wall_s", "incr_steps", "incr_dt_median_s")})

        done = [l for l, *_ in RUNS if R["runs"].get(l, {}).get("status") == "complete"]
        files = [str(OUT / f"paced-{l}.json") for l in done]
        if files:
            rc, o = sh([sys.executable, str(CODE / "scripts/live/summarize_replay.py")] + files, 120, cwd=CODE)
            R["summarize_replay"] = o.splitlines()
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
