"""Voxtral Mini 4B Realtime served by vLLM (CUDA graphs) on a Kaggle T4 (2026-11-09).
Why: the Transformers loop decodes Voxtral at 1.0-1.1 x real time on a T4 (no headroom, lag drifts). vLLM is Mistral's recommended server.
Does it decode faster than real time and hold a low lag? Same clips and metrics as kaggle_voxtral_t4.
Per server config (A: piecewise CUDA graphs, B: eager) and clip: burst throughput (all audio sent at once, seconds of decode per second of audio)
and paced streaming (audio sent in 128 ms pieces at real time; lag = wall time a delta arrives minus the audio position sent so far).
Writes /kaggle/working/voxtral_vllm.json after every step; a failed config records its server log tail and the run continues."""
import asyncio, base64, json, os, re, signal, subprocess, sys, time, traceback, urllib.request
from pathlib import Path

OUT = Path('/kaggle/working'); R = {'cells': {}, 'log': [], 'env': {}}; T0 = time.time()
def save(): (OUT / 'voxtral_vllm.json').write_text(json.dumps(R, ensure_ascii=False, indent=1))
def log(m, **k): R['log'].append({'t': round(time.time() - T0, 1), 'm': m, **k}); print(m, k, flush=True); save()
def sh(c, t=3000, **kw):
    p = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t, **kw); return p.returncode, (p.stdout + p.stderr)[-1500:]

R['env']['gpu'] = sh('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader')[1].strip(); log('env', **R['env'])
rc, o = sh(f'{sys.executable} -m pip install -q uv'); log('uv', rc=rc)
VENV = '/kaggle/working/v'
rc, o = sh(f'uv venv {VENV} --python 3.12 && uv pip install -p {VENV}/bin/python -q -U vllm soxr librosa soundfile websockets pybase64 numpy scipy "transformers>=5" "mistral-common[audio]"', t=3000); log('vllm install', rc=rc, tail=o[-600:])
rc, o = sh(f'{VENV}/bin/python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"'); R['env']['vllm'] = o.strip(); log('versions', v=o.strip())
if rc: raise SystemExit('vllm import failed')

import numpy as np, soundfile as sf, scipy.signal as ss
FILES = {p.stem: p for p in Path('/kaggle/input').rglob('*.wav')}
norm = lambda s: re.findall(r"[^\W_]+", s.lower())
def wer(ref, hyp):
    r, h = norm(ref), norm(hyp); d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        p, d[0] = d[0], i
        for j, b in enumerate(h, 1): p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a != b))
    return d[-1] / max(len(r), 1)
def load(name):
    a, sr = sf.read(str(FILES[name]), dtype='float32'); a = a if a.ndim == 1 else a.mean(1)
    if sr != 16000: a = ss.resample_poly(a, 16000, sr).astype('float32')
    return a, json.load(open(FILES[name].with_suffix('.json')))['text']

import websockets
MODEL = 'mistralai/Voxtral-Mini-4B-Realtime-2602'
async def transcribe(pcm16: bytes, paced: bool):
    t0 = None; deltas = []; sent = {'s': 0.0}
    async with websockets.connect('ws://127.0.0.1:8000/v1/realtime', max_size=None) as ws:
        assert json.loads(await ws.recv())['type'] == 'session.created'
        await ws.send(json.dumps({'type': 'session.update', 'model': MODEL})); await ws.send(json.dumps({'type': 'input_audio_buffer.commit'}))
        t0 = time.time(); CH = 4096
        async def send():
            for i in range(0, len(pcm16), CH):
                if paced:
                    wait = (i + CH) / 2 / 16000 - (time.time() - t0)
                    if wait > 0: await asyncio.sleep(wait)
                await ws.send(json.dumps({'type': 'input_audio_buffer.append', 'audio': base64.b64encode(pcm16[i:i + CH]).decode()}))
                sent['s'] = min(i + CH, len(pcm16)) / 2 / 16000
            await ws.send(json.dumps({'type': 'input_audio_buffer.commit', 'final': True}))
        task = asyncio.create_task(send()); text = ''
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), 120))
            if m['type'] == 'transcription.delta': deltas.append((round(sent['s'], 2), round(time.time() - t0 - sent['s'], 2), m['delta']))
            elif m['type'] == 'transcription.done': text = m['text']; break
            elif m['type'] == 'error': raise RuntimeError(str(m)[:300])
        await task
    return text, deltas, time.time() - t0

def serve(tag, extra):
    slog = OUT / f'vllm_{tag}.log'
    env = {**os.environ, 'VLLM_DISABLE_COMPILE_CACHE': '1'}
    p = subprocess.Popen(f'{VENV}/bin/vllm serve {MODEL} --dtype float16 --max-model-len 16384 --gpu-memory-utilization 0.92 --port 8000 {extra}', shell=True, env=env,
                         stdout=open(slog, 'w'), stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    for _ in range(240):
        if p.poll() is not None: return p, slog, False
        try: urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2); return p, slog, True
        except Exception: time.sleep(5)
    return p, slog, False

def stop(p):
    try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception: pass
    try: p.wait(timeout=30)
    except Exception:
        try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception: pass
    time.sleep(8)

CONFIGS = {'graphs': '--compilation_config \'{"cudagraph_mode": "PIECEWISE"}\'', 'eager': '--enforce-eager'}
for tag, extra in CONFIGS.items():
    t = time.time(); p, slog, ok = serve(tag, extra)
    log('server', tag=tag, up=ok, s=round(time.time() - t, 1), tail=slog.read_text(errors='ignore')[-800:] if not ok else '')
    if not ok: stop(p); continue
    try:
        for name in ['en_paced', 'en_continuous', 'de_paced']:
            a, ref = load(name); pcm = (np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes(); dur = len(a) / 16000; cell = {'audio_s': round(dur, 1)}
            asyncio.run(transcribe(pcm[:16000 * 2 * 3], False))   # warm-up
            text, _, wall = asyncio.run(transcribe(pcm, False)); cell['burst'] = {'decode_s_per_audio_s': round(wall / dur, 3), 'wer': round(wer(ref, text), 4)}
            text, deltas, wall = asyncio.run(transcribe(pcm, True)); lags = [d[1] for d in deltas if d[2].strip()]
            cell['paced'] = {'wer': round(wer(ref, text), 4), 'wall_minus_audio_s': round(wall - dur, 1), 'lag_median_s': round(float(np.median(lags)), 2), 'lag_p90_s': round(float(np.percentile(lags, 90)), 2),
                             'lag_max_s': max(lags), 'series': [[d[0], d[1]] for d in deltas if d[2].strip()][:200]}
            R['cells'][f'{tag}|{name}'] = cell; log('cell', cfg=tag, name=name, **{k: v for k, v in cell.items()})
    except Exception:
        log('cell failed', cfg=tag, tb=traceback.format_exc()[-1200:])
    finally:
        stop(p)
log('done')
