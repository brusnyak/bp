"""Fresh-install test on a clean Linux GPU box (Kaggle T4): what a stranger does after `git clone`.
1. Python 3.11 through uv (Kaggle ships 3.13, setup.py needs 3.10-3.12).  2. `python scripts/setup.py --profile nvidia --langs de,sk,fr` (the real installer).
3. Start the server with presets/nvidia.env only (no LD_LIBRARY_PATH: backend/cuda_libs.py must find the pip-installed CUDA libraries).
4. Replay through the real route: EN->DE (regression), DE->EN (German speech in), SK->EN (Slovak speech in, kinit model).
5. Export the converted translation models (en-de, de-en, en-fr) so the owner's Mac can use them without converting locally.
Writes /kaggle/working/fresh_install.json after every step."""
import io, json, os, re, secrets, shutil, statistics as st, subprocess, sys, time, urllib.request
from pathlib import Path

OUT = Path('/kaggle/working'); BP = OUT / 'BP'
R = {'steps': {}, 'runs': {}, 'log': [], 'env': {}}
T0 = time.time()
def save(): (OUT / 'fresh_install.json').write_text(json.dumps(R, ensure_ascii=False, indent=1))
def log(m, **k): R['log'].append({'t': round(time.time() - T0, 1), 'm': m, **k}); print(m, k, flush=True); save()
def sh(c, t=3600, **kw):
    p = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t, **kw); return p.returncode, (p.stdout + p.stderr)

src = next(p.parent for p in Path('/kaggle/input').rglob('app.py') if (p.parent / 'backend').is_dir())
owner_dir = next(Path('/kaggle/input').rglob('en_script_reading.m4a')).parent
shutil.copytree(src, BP, dirs_exist_ok=True); log('staged', src=str(src))
R['env']['gpu'] = sh('nvidia-smi --query-gpu=name --format=csv,noheader')[1].strip()

# 1. python 3.11
rc, o = sh('pip install -q uv && uv python install 3.11'); rc2, py = sh('uv python find 3.11'); py = py.strip().splitlines()[-1]
log('python311', rc=rc, py=py); R['steps']['python311'] = {'rc': rc, 'path': py}

# 2. the real installer
t = time.time(); rc, o = sh(f'{py} scripts/setup.py --profile nvidia --langs de,sk,fr', t=3300, cwd=BP, env={**os.environ, 'PATH': os.environ['PATH']})
(OUT / 'setup_output.log').write_text(o)
R['steps']['setup'] = {'rc': rc, 'wall_s': round(time.time() - t), 'tail': o[-1500:]}; log('setup', rc=rc, wall_s=round(time.time() - t))
VPY = BP / '.venv' / 'bin' / 'python'
have = {k: (BP / p).exists() for k, p in {'venv': '.venv/bin/python', 'turbo': 'ct2_models/whisper-large-v3-turbo/model.bin', 'kinit': 'ct2_models/whisper-large-v3-turbo-sk/model.bin',
        'mt_en_de': 'ct2_models/Helsinki-NLP--opus-mt-en-de/model.bin', 'mt_de_en': 'ct2_models/Helsinki-NLP--opus-mt-de-en/model.bin', 'mt_sk_en': 'ct2_models/Helsinki-NLP--opus-mt-sk-en/model.bin',
        'voice_de': 'backend/tts/piper_models/de_DE-thorsten-medium.onnx', 'voice_en': 'backend/tts/piper_models/en_US-john-medium.onnx'}.items()}
R['steps']['assets'] = have; log('assets', **have)

# 3. audio inputs (system python: pyarrow/soundfile)
sh(f'{sys.executable} -m pip install -q pyarrow soundfile scipy huggingface_hub')
import numpy as np, soundfile as sf, pyarrow.parquet as pq, scipy.signal as ss
from huggingface_hub import hf_hub_download
f = hf_hub_download('google/fleurs', 'de_de/test/0000.parquet', repo_type='dataset', revision='refs/convert/parquet')
parts, refs = [], []
for r in pq.read_table(f).to_pylist()[:14]:
    a, sr = sf.read(io.BytesIO(r['audio']['bytes']), dtype='float32'); a = a.mean(1) if a.ndim > 1 else a
    if sr != 16000: a = ss.resample_poly(a, 16000, sr).astype('float32')
    parts += [a, np.zeros(int(0.4 * 16000), dtype='float32')]; refs.append(r['transcription'])
sf.write(OUT / 'de_input.wav', np.concatenate(parts), 16000); (OUT / 'de_ref.txt').write_text(' '.join(refs))
INPUTS = {'en': (owner_dir / 'en_script_reading.m4a', (owner_dir / 'en_ref.txt').read_text()), 'de': (OUT / 'de_input.wav', ' '.join(refs)), 'sk': (owner_dir / 'sk_script_reading.m4a', (owner_dir / 'sk_ref.txt').read_text())}

norm = lambda s: re.findall(r"[^\W_]+", s.lower())
def prefix_wer(ref_text, hyp_text):
    R_, h = norm(ref_text), norm(hyp_text); best = 9.0
    for k in range(max(5, int(len(h) * 0.7)), min(len(R_), int(len(h) * 1.4)) + 1):
        r = R_[:k]; d = list(range(len(h) + 1))
        for i, a in enumerate(r, 1):
            pv = d[:]; d[0] = i
            for j, b in enumerate(h, 1): d[j] = min(pv[j] + 1, d[j-1] + 1, pv[j-1] + (a != b))
        best = min(best, d[-1] / k)
    return best
def preset_env(name):
    env = {}
    for line in (BP / 'presets' / f'{name}.env').read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line: k, v = line.split('=', 1); env[k.strip()] = v.strip()
    return env

PORT = 8013
def run(src_l, tgt_l):
    key = f'{src_l}-{tgt_l}'; slog = OUT / f'server_{key}.log'; out_json = OUT / f'replay_{key}.json'; audio, ref = INPUTS[src_l]
    env = {k: v for k, v in os.environ.items() if k != 'LD_LIBRARY_PATH'}          # the point: no manual library path
    env.update(preset_env('nvidia')); env.update({'BP_NO_SSL': '1', 'BP_PORT': str(PORT), 'BP_ENABLE_RATE_LIMIT': '0', 'BP_MT_DEVICE': 'cpu', 'BP_SILENCE_RMS': '0',
                'JWT_SECRET': secrets.token_urlsafe(32), 'BP_LOG_PATH': str(OUT / f'applog_{key}.log'), 'BP_LOG_SESSIONS': '1'})
    srv = subprocess.Popen([str(VPY), 'app.py'], cwd=BP, env=env, stdout=open(slog, 'w'), stderr=subprocess.STDOUT)
    try:
        for _ in range(150):
            try: urllib.request.urlopen(f'http://127.0.0.1:{PORT}/', timeout=2); break
            except Exception: time.sleep(2)
        t = time.time()
        p = subprocess.run([str(VPY), 'scripts/live/paced_replay.py', str(audio), '--base', f'http://127.0.0.1:{PORT}', '--src', src_l, '--tgt', tgt_l, '--tts', 'piper', '--seconds', '60',
                            '--load-timeout', '900', '--out', str(out_json)], cwd=BP, env=env, capture_output=True, text=True, timeout=1800)
        if not out_json.exists(): raise RuntimeError('replay wrote no result: ' + (p.stdout + p.stderr)[-600:])
        d = json.loads(out_json.read_text()); r = d['rates']['1.0']; lag = r['lag_s']; fw = [x for x in r['first_word_lag_s'] if x is not None]
        tr = [e['text'].strip() for e in d['events'] if e['type'] == 'transcription_result' and e.get('text') and e.get('stt_time', 0) > 0]
        tl = [e['text'].strip() for e in d['events'] if e['type'] == 'translation_result' and e.get('text')]
        stock = sum(1 for x in tr if re.sub(r'[^a-z ]', '', x.lower()).strip() in ('thank you', 'okay', 'oh', 'you', 'bye', 'thanks'))
        st_txt = slog.read_text(errors='ignore')
        R['runs'][key] = {'segments': r['segments'], 'with_audio': r['with_audio'], 'lag_median_s': round(st.median(lag), 2), 'lag_p90_s': round(sorted(lag)[int(.9 * len(lag))], 2),
                          'first_word_lag_median_s': round(st.median(fw), 2) if fw else None, 'transcript_WER_prefix': round(prefix_wer(ref, ' '.join(tr)), 3), 'stock_phrase_segments': stock,
                          'server_errors': len(re.findall(r'Traceback|ERROR', st_txt)), 'libcublas_error': 'libcublas' in st_txt and 'not found' in st_txt, 'wall_s': round(time.time() - t),
                          'transcript_sample': tr[:3], 'translation_sample': tl[:3]}
        log('run', run=key, **{k: v for k, v in R['runs'][key].items() if 'sample' not in k})
    except Exception as e:
        import traceback; log('run_fail', run=key, err=f'{type(e).__name__}: {e}'[:400], tb=traceback.format_exc()[-300:], server_tail=slog.read_text(errors='ignore')[-700:] if slog.exists() else '')
    finally:
        srv.terminate()
        try: srv.wait(timeout=20)
        except Exception: srv.kill()
        time.sleep(3)
if have['venv']:
    for s_, t_ in (('en', 'de'), ('de', 'en'), ('sk', 'en')): run(s_, t_)

# 5. export the converted translation models for the owner's Mac
for pair in ('en-de', 'de-en', 'en-fr'):
    d = BP / 'ct2_models' / f'Helsinki-NLP--opus-mt-{pair}'
    if d.exists(): shutil.copytree(d, OUT / 'export' / 'ct2_models' / d.name, dirs_exist_ok=True)
R['steps']['export'] = sorted(p.name for p in (OUT / 'export' / 'ct2_models').glob('*')) if (OUT / 'export').exists() else []
shutil.rmtree(BP / 'ct2_models', ignore_errors=True); shutil.rmtree(BP / '.venv', ignore_errors=True); shutil.rmtree(BP / '.venv-convert', ignore_errors=True)   # keep the kernel output small
R['status'] = 'done'; log('done', wall_s=round(time.time() - T0))
