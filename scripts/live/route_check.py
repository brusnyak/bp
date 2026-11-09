#!/usr/bin/env python3
"""One real-route check on THIS machine, polite and self-cleaning: start app.py on a spare port with a given engine, replay a recording through the
real /ws route at 1x (paced_replay.py), print the numbers, stop the server (always). Run it through `~/.claude/scripts/heavy` on a shared laptop.

  ~/.claude/scripts/heavy -- .venv/bin/python scripts/live/route_check.py --engine voxtral --strategy agree_short [--audio processed/testaudio/en_paced.wav]

Refuses to start when the 1-minute load is above --max-load or swap is above 2 GB used (a shared Mac measures the desktop, not the engine)."""
import argparse, json, os, re, signal, subprocess, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--engine", default="mlx_whisper"); ap.add_argument("--strategy", default="agree_short")
ap.add_argument("--audio", default="processed/testaudio/en_paced.wav"); ap.add_argument("--src", default="en"); ap.add_argument("--tgt", default="de")
ap.add_argument("--seconds", default="60"); ap.add_argument("--port", default="8013"); ap.add_argument("--max-load", type=float, default=6.0)
ap.add_argument("--out", default=None)
a = ap.parse_args()
load = os.getloadavg()[0]
swap = re.search(r"used = ([\d.]+)M", subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout) if sys.platform == "darwin" else None
if load > a.max_load or (swap and float(swap.group(1)) > 2048):
    sys.exit(f"refused: load {load:.1f} (max {a.max_load}), swap used {swap.group(1) if swap else 'n/a'} MB (max 2048). Close apps and retry.")
out = a.out or f"processed/route_check_{a.engine}_{a.strategy}.json"
env = {**os.environ, "BP_NO_SSL": "1", "BP_PORT": a.port, "BP_STT_ENGINE": a.engine, "BP_PROSODY": "1", "BP_LOG_PATH": f"logs/route_check_{a.engine}.log",
       "BP_SK_STT_MODEL": os.environ.get("BP_SK_STT_MODEL", "ct2_models/whisper-large-v3-turbo-sk")}
srv = subprocess.Popen([sys.executable, "app.py"], env=env, stdout=open(f"logs/route_check_{a.engine}.out", "w"), stderr=subprocess.STDOUT, preexec_fn=os.setsid)
try:
    for _ in range(180):
        try: urllib.request.urlopen(f"http://127.0.0.1:{a.port}/", timeout=2); break
        except Exception: time.sleep(2)
    subprocess.run([sys.executable, "scripts/live/paced_replay.py", a.audio, "--base", f"http://127.0.0.1:{a.port}", "--src", a.src, "--tgt", a.tgt, "--tts", "piper",
                    "--strategy", a.strategy, "--seconds", a.seconds, "--load-timeout", "600", "--out", out], env=env, capture_output=True, text=True, timeout=900)
    d = json.load(open(out)); r = d["rates"]["adaptive"]
    print(json.dumps({"engine": a.engine, "strategy": a.strategy, "segments": r["segments"], "lag_median_s": r["lag_median_s"], "lag_p90_s": r["lag_p90_s"],
                      "lag_growth_s_per_run": r["lag_growth_s"], "tail_after_input_s": r["last_listen_end_vs_input_end_s"], "load_at_end": round(os.getloadavg()[0], 1)}))
finally:
    try: os.killpg(os.getpgid(srv.pid), signal.SIGTERM); srv.wait(timeout=20)
    except Exception:
        try: os.killpg(os.getpgid(srv.pid), signal.SIGKILL)
        except Exception: pass
