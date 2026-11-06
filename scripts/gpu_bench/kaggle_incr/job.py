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
STT_SIZE = "small.en"
SECONDS = 90
OUT = Path("/kaggle/working")
CODE = OUT / "bp"
REPORT = OUT / "incr-report.json"
T0 = time.time()
RUNS = [("A", {}), ("B", {"BP_INCREMENTAL": "1"}),
        ("C", {"BP_INCREMENTAL": "1", "BP_INCR_STEP_S": "0.4", "BP_INCR_MIN_WORDS": "3", "BP_INCR_MAX_WAIT_S": "1.0"}),
        ("D", {})]
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
        wavs = list(Path("/kaggle/input").rglob("rainbow_en_90s.wav"))
        onnx = list(Path("/kaggle/input").rglob("sk_SK-lili-medium.onnx"))
        ojson = list(Path("/kaggle/input").rglob("sk_SK-lili-medium.onnx.json"))
        assert len(srcdirs) == 1 and len(wavs) == 1 and len(onnx) == 1 and len(ojson) == 1, (srcdirs, wavs, onnx, ojson)
        shutil.copytree(srcdirs[0], CODE)
        pm = CODE / "backend/tts/piper_models"
        pm.mkdir(parents=True, exist_ok=True)
        shutil.copy(onnx[0], pm)
        shutil.copy(ojson[0], pm)
        R["tts_voice"] = "sk_SK-lili-medium (generic public Piper voice; personal fine-tune not uploaded)"
        R["input_audio_sha256"] = hashlib.sha256(wavs[0].read_bytes()).hexdigest()
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
        env0.update({"BP_NO_SSL": "1", "BP_PORT": str(PORT), "PYTHONUNBUFFERED": "1"})
        for k in [k for k in env0 if k.startswith("BP_INCR") or k in ("BP_VAD_SILENCE_S", "BP_VAD_MAX_SEGMENT_S", "BP_BARGE_IN")]:
            env0.pop(k)

        # patch: STT step duration in the INCR log line
        mp = CODE / "backend/main.py"
        s = mp.read_text()
        a = "        audio = np.concatenate(frames[:n])\n        times = list(frames.t[:n])\n"
        b = "        logging.info(f\"INCR step: audio="
        assert s.count(a) == 1 and s.count(b) == 1
        s = s.replace(a, a + "        _t_step = time.perf_counter()\n")
        s = s.replace(b, "        logging.info(f\"INCR step: dt={time.perf_counter() - _t_step:.3f} audio=")
        mp.write_text(s)
        R["deviations"].append("main.py INCR step log line gets dt= (time since audio concat, includes model_call_lock wait + GPU STT)")

        rc, o = sh([sys.executable, "-c",
                    "import sys; sys.setrecursionlimit(2000); import backend.mt.convert_opus_mt_to_ct2 as c; "
                    "c.convert_model('Helsinki-NLP/opus-mt-en-sk', 'ct2_models/Helsinki-NLP--opus-mt-en-sk', quantization='int8')"],
                   300, cwd=CODE, env=env0)
        mt_dir = CODE / "ct2_models/Helsinki-NLP--opus-mt-en-sk"
        step("convert en-sk", rc=rc, ok=(mt_dir / "model.bin").is_file())
        assert (mt_dir / "model.bin").is_file()
        rc, o = sh([sys.executable, "-c", f"from faster_whisper.utils import download_model as d; print(d('{STT_SIZE}'))"], 300, env=env0)
        step("stt download", rc=rc, tail=o[-200:])
        assert rc == 0
        smoke = ("from faster_whisper import WhisperModel; import sys; m=WhisperModel(sys.argv[1],device='cuda',compute_type='float16'); "
                 "s,_=m.transcribe(sys.argv[2], language='en'); print('SMOKE', ' '.join(x.text for x in s)[:80])")
        rc, o = sh([sys.executable, "-c", smoke, STT_SIZE, str(wavs[0])], 240, env=env0)
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

        for label, extra_env in RUNS:
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
            cmd = [sys.executable, "-c", CHILD, str(CODE), json.dumps({"stt_model_size": STT_SIZE}), str(wavs[0]),
                   "--base", f"http://127.0.0.1:{PORT}", "--src", "en", "--tgt", "sk", "--tts", "piper",
                   "--seconds", str(SECONDS), "--rates", "1.0", "1.25", "adaptive", "--load-timeout", "300", "--out", str(out_json)]
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

        done = [l for l, _ in RUNS if R["runs"].get(l, {}).get("status") == "complete"]
        files = [str(OUT / f"paced-{l}.json") for l in done]
        if files:
            rc, o = sh([sys.executable, str(CODE / "scripts/live/summarize_replay.py")] + files, 120, cwd=CODE)
            R["summarize_replay"] = o.splitlines()
            rc, o = sh([sys.executable, str(CODE / "scripts/live/fragment_quality.py"), "ct2_models/Helsinki-NLP--opus-mt-en-sk", "en", "sk"] + files,
                       240, cwd=CODE, env=env0)
            R["fragment_quality"] = [l for l in o.splitlines() if "chrF" in l] or o[-800:]
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
