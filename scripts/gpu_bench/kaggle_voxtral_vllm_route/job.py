"""Voxtral Realtime served by vLLM (eager, T4) through the REAL route (2026-11-09): vLLM server as a sidecar process in its own venv, app.py with
BP_STT_ENGINE=voxtral_vllm, the owner's 60 s English reading replayed at 1x with scripts/live/paced_replay.py (same replay and metrics as kaggle_route_test and
kaggle_voxtral_route, so the earlier Whisper control run (median lag 3.16 s, error 10.7%, agree_short en->de) is the comparison).
Runs: agree_short en->de, agree en->de, agree_short en->sk.  Writes /kaggle/working/route_test.json after every run; removes the venv and the code copy at the end."""
import glob, json, os, re, secrets, shutil, site, statistics as st, subprocess, sys, time, urllib.request
from pathlib import Path

OUT = Path('/kaggle/working'); BP = OUT / 'BP'
R = {'runs': {}, 'log': [], 'env': {}}
T0 = time.time()
def save(): (OUT / 'route_test.json').write_text(json.dumps(R, ensure_ascii=False, indent=1))
def log(m, **k): R['log'].append({'t': round(time.time() - T0, 1), 'm': m, **k}); print(m, k, flush=True); save()
def sh(c, t=3000, **kw):
    p = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t, **kw); return p.returncode, (p.stdout + p.stderr)[-1200:]

src = next(p.parent for p in Path('/kaggle/input').rglob('app.py') if (p.parent / 'backend').is_dir())
audio = next(Path('/kaggle/input').rglob('en_script_reading.m4a')); ref_text = (audio.parent / 'en_ref.txt').read_text()
shutil.copytree(src, BP, dirs_exist_ok=True); log('staged', src=str(src), audio=str(audio))

# --- dependencies exactly as the earlier route kernels did (drop pins that have no wheels on Kaggle's Python 3.13) ---
drop = re.compile(r'^(numpy|scipy|transformers|tokenizers|huggingface-hub|soundfile|numba|onnxruntime|librosa|SQLAlchemy)\b', re.I)
lines = [re.sub(r'\s+#.*$', '', l).strip() for l in (BP / 'requirements.txt').read_text().splitlines()]
lines = [l for l in lines if l and not l.startswith('#') and not drop.match(l)]
lines += ['transformers==4.57.6', 'tokenizers==0.22.2', 'huggingface-hub==0.36.2', 'sentencepiece', 'SQLAlchemy>=2.0.36', 'soundfile', 'scipy', 'librosa',
          'nvidia-cublas-cu12', 'nvidia-cudnn-cu12', 'websockets', 'requests', 'faster-whisper', 'piper-tts', 'webrtcvad-wheels']
(OUT / 'route-requirements.txt').write_text('\n'.join(lines) + '\n')
rc, o = sh(f'{sys.executable} -m pip install -q -r {OUT}/route-requirements.txt'); log('pip', rc=rc, tail=o[-300:])
BASE_PIP_DONE = True
libs = [str(p) for sp in site.getsitepackages() for p in Path(sp).glob('nvidia/*/lib')]
R['env']['gpu'] = sh('nvidia-smi --query-gpu=name --format=csv,noheader')[1].strip()

# --- assets a setup.py run would fetch: Piper voices, translation models ---
base_env = {**os.environ, 'PYTHONPATH': str(BP), 'PYTHONUTF8': '1', 'LD_LIBRARY_PATH': ':'.join(libs + [os.environ.get('LD_LIBRARY_PATH', '')])}
for v in ('en_US-john-medium', 'de_DE-thorsten-medium', 'sk_SK-lili-medium'):
    rc, o = sh(f'{sys.executable} backend/tts/download_piper_models.py {v}', cwd=BP, env=base_env); log('voice', v=v, rc=rc)
for pair in ('en-de', 'en-sk'):
    rc, o = sh(f"{sys.executable} -c \"import sys; sys.setrecursionlimit(2000); import backend.mt.convert_opus_mt_to_ct2 as c; c.convert_model('Helsinki-NLP/opus-mt-{pair}', 'ct2_models/Helsinki-NLP--opus-mt-{pair}', quantization='int8')\"", cwd=BP, env=base_env)
    log('mt_convert', pair=pair, rc=rc, tail=o[-200:] if rc else '')

VENV = '/kaggle/working/v'
rc, o = sh(f'{sys.executable} -m pip install -q uv && uv venv {VENV} --python 3.12 && uv pip install -p {VENV}/bin/python -q -U vllm soxr librosa soundfile websockets pybase64 numpy scipy "transformers>=5" "mistral-common[audio]"', t=3000); log('vllm venv', rc=rc, tail=o[-300:])
vlog = OUT / 'vllm_server.log'
vsrv = subprocess.Popen(f'{VENV}/bin/vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 --dtype float16 --max-model-len 3584 --gpu-memory-utilization 0.92 --enforce-eager --port 8001', shell=True,
                        env={**os.environ, 'VLLM_DISABLE_COMPILE_CACHE': '1'}, stdout=open(vlog, 'w'), stderr=subprocess.STDOUT, preexec_fn=os.setsid)
up = False
for _ in range(240):
    if vsrv.poll() is not None: break
    try: urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=2); up = True; break
    except Exception: time.sleep(5)
log('vllm server', up=up, tail='' if up else vlog.read_text(errors='ignore')[-800:])
if not up: raise SystemExit('vllm server did not start')

norm = lambda s: re.findall(r"[^\W_]+", s.lower()); REF = norm(ref_text)
def prefix_wer(hyp_text):
    h = norm(hyp_text); best = 9.0
    for k in range(max(5, int(len(h) * 0.7)), min(len(REF), int(len(h) * 1.4)) + 1):
        r = REF[:k]; d = list(range(len(h) + 1))
        for i, a in enumerate(r, 1):
            pv = d[:]; d[0] = i
            for j, b in enumerate(h, 1): d[j] = min(pv[j] + 1, d[j-1] + 1, pv[j-1] + (a != b))
        best = min(best, d[-1] / k)
    return best

VENV_ENV = {'BP_STT_ENGINE': 'voxtral_vllm', 'BP_VOXTRAL_URL': 'ws://127.0.0.1:8001/v1/realtime'}
CONFIGS = {'vllm-agree_short': (VENV_ENV, 'agree_short'), 'vllm-agree': (VENV_ENV, 'agree')}
PLAN = [('en', 'de', 'vllm-agree_short'), ('en', 'de', 'vllm-agree'), ('en', 'sk', 'vllm-agree_short')]
PORT = 8013
def run(src_l, tgt_l, cfg):
    key = f'{src_l}-{tgt_l}|{cfg}'; slog = OUT / f'server_{src_l}-{tgt_l}_{cfg}.log'; out_json = OUT / f'replay_{src_l}-{tgt_l}_{cfg}.json'
    env = {**base_env, 'BP_NO_SSL': '1', 'BP_PORT': str(PORT), 'BP_ENABLE_RATE_LIMIT': '0', 'BP_DEVICE': 'cuda', 'BP_COMPUTE_TYPE': 'float16', 'BP_STT_MODEL': 'large-v3-turbo',
           'BP_MT_DEVICE': 'cpu', 'BP_PROSODY': '1', 'BP_SILENCE_RMS': '0', 'JWT_SECRET': secrets.token_urlsafe(32), 'BP_LOG_PATH': str(OUT / f'applog_{key.replace("|", "_")}.log'), 'BP_LOG_SESSIONS': '1', **CONFIGS[cfg][0]}
    srv = subprocess.Popen([sys.executable, 'app.py'], cwd=BP, env=env, stdout=open(slog, 'w'), stderr=subprocess.STDOUT)
    try:
        for _ in range(150):
            try: urllib.request.urlopen(f'http://127.0.0.1:{PORT}/', timeout=2); break
            except Exception: time.sleep(2)
        t = time.time()
        p = subprocess.run([sys.executable, 'scripts/live/paced_replay.py', str(audio), '--base', f'http://127.0.0.1:{PORT}', '--src', src_l, '--tgt', tgt_l, '--tts', 'piper',
                            '--seconds', '60', '--strategy', CONFIGS[cfg][1], '--load-timeout', '1500', '--out', str(out_json)], cwd=BP, env=env, capture_output=True, text=True, timeout=1500)
        if not out_json.exists(): raise RuntimeError('replay wrote no result: ' + (p.stdout + p.stderr)[-500:])
        d = json.loads(out_json.read_text()); r = d['rates']['1.0']; lag = r['lag_s']; fw = [x for x in r['first_word_lag_s'] if x is not None]
        tr = [e['text'].strip() for e in d['events'] if e['type'] == 'transcription_result' and e.get('text') and e.get('stt_time', 0) > 0]   # stt_time 0.0 = final-promotion echo, not a recognition
        tl = [e['text'].strip() for e in d['events'] if e['type'] == 'translation_result' and e.get('text')]
        dup = sum(1 for a, b in zip(tr, tr[1:]) if a == b)
        stock = sum(1 for t in tr if re.sub(r'[^a-z ]', '', t.lower()).strip() in ('thank you', 'okay', 'oh', 'you', 'bye', 'thanks'))
        slog_text = slog.read_text(errors='ignore')
        R['runs'][key] = {'segments': r['segments'], 'with_audio': r['with_audio'], 'lag_median_s': round(st.median(lag), 2), 'lag_p90_s': round(sorted(lag)[int(.9 * len(lag))], 2),
                          'lag_max_s': round(max(lag), 2), 'first_word_lag_median_s': round(st.median(fw), 2) if fw else None, 'tail_after_input_s': round(r['last_listen_end_vs_input_end_s'], 2),
                          'transcript_WER_prefix': round(prefix_wer(' '.join(tr)), 3), 'transcript_events': len(tr), 'duplicate_transcripts': dup, 'stock_phrase_segments': stock,
                          'server_errors': len(re.findall(r'Traceback|ERROR', slog_text)), 'wall_s': round(time.time() - t), 'translation_sample': tl[:5]}
        log('run', run=key, **{k: v for k, v in R['runs'][key].items() if k != 'translation_sample'})
    except Exception as e:
        import traceback; log('run_fail', run=key, err=f'{type(e).__name__}: {e}'[:300], tb=traceback.format_exc()[-400:], server_tail=slog.read_text(errors='ignore')[-600:] if slog.exists() else '')
    finally:
        srv.terminate()
        try: srv.wait(timeout=20)
        except Exception: srv.kill()
        time.sleep(3)
for s_, t_, c_ in PLAN: run(s_, t_, c_)
import signal, shutil
try: os.killpg(os.getpgid(vsrv.pid), signal.SIGTERM)
except Exception: pass
shutil.rmtree(VENV, ignore_errors=True); shutil.rmtree(BP, ignore_errors=True)   # keep the kernel output small
R['status'] = 'done'; log('done', wall_s=round(time.time() - T0))
