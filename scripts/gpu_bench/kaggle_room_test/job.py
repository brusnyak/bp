"""Room mode on a Kaggle T4: one speaker (the owner's English reading through the real /ws route, English to German for the speaker's own session),
N listeners on /ws/listen in several languages. Measures what each listener receives, that listeners of one language get identical audio,
how the speaker's own delay changes with a room attached, and how cost scales with listeners (3 vs 12) at a fixed 3 languages.
Writes /kaggle/working/room_test.json after every run."""

import glob, json, os, re, secrets, shutil, site, statistics as st, subprocess, sys, time, urllib.request
from pathlib import Path

OUT = Path('/kaggle/working'); BP = OUT / 'BP'
R = {'runs': {}, 'log': [], 'env': {}}
T0 = time.time()
def save(): (OUT / 'room_test.json').write_text(json.dumps(R, ensure_ascii=False, indent=1))
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
libs = [str(p) for sp in site.getsitepackages() for p in Path(sp).glob('nvidia/*/lib')]
R['env']['gpu'] = sh('nvidia-smi --query-gpu=name --format=csv,noheader')[1].strip()

# --- assets a setup.py run would fetch: Piper voices, translation models ---
base_env = {**os.environ, 'PYTHONPATH': str(BP), 'PYTHONUTF8': '1', 'LD_LIBRARY_PATH': ':'.join(libs + [os.environ.get('LD_LIBRARY_PATH', '')])}
for v in ('en_US-john-medium', 'de_DE-thorsten-medium', 'sk_SK-lili-medium', 'fr_FR-siwis-medium'):
    rc, o = sh(f'{sys.executable} backend/tts/download_piper_models.py {v}', cwd=BP, env=base_env); log('voice', v=v, rc=rc)
for pair in ('en-de', 'en-sk', 'en-fr'):
    rc, o = sh(f"{sys.executable} -c \"import sys; sys.setrecursionlimit(2000); import backend.mt.convert_opus_mt_to_ct2 as c; c.convert_model('Helsinki-NLP/opus-mt-{pair}', 'ct2_models/Helsinki-NLP--opus-mt-{pair}', quantization='int8')\"", cwd=BP, env=base_env)
    log('mt_convert', pair=pair, rc=rc, tail=o[-200:] if rc else '')

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


PORT = 8013
PLAN = [('baseline', None), ('room3', 'de,sk,fr'), ('room12', 'de,de,de,de,sk,sk,sk,sk,fr,fr,fr,fr')]
def run(name, langs):
    slog = OUT / f'server_{name}.log'; out_json = OUT / f'replay_{name}.json'; lj = OUT / f'listeners_{name}.json'; room = f'kagroom{name}'
    env = {**base_env, 'BP_NO_SSL': '1', 'BP_PORT': str(PORT), 'BP_ENABLE_RATE_LIMIT': '0', 'BP_DEVICE': 'cuda', 'BP_COMPUTE_TYPE': 'float16', 'BP_STT_MODEL': 'large-v3-turbo',
           'BP_MT_DEVICE': 'cpu', 'BP_PROSODY': '1', 'BP_SILENCE_RMS': '0', 'BP_INCREMENTAL': '1', 'BP_INCR_STEP_S': '1.0', 'JWT_SECRET': secrets.token_urlsafe(32),
           'BP_LOG_PATH': str(OUT / f'applog_{name}.log')}
    srv = subprocess.Popen([sys.executable, 'app.py'], cwd=BP, env=env, stdout=open(slog, 'w'), stderr=subprocess.STDOUT)
    try:
        for _ in range(150):
            try: urllib.request.urlopen(f'http://127.0.0.1:{PORT}/', timeout=2); break
            except Exception: time.sleep(2)
        lis = None
        if langs:
            lis = subprocess.Popen([sys.executable, 'scripts/live/room_listeners.py', '--base', f'http://127.0.0.1:{PORT}', '--room', room, '--langs', langs, '--seconds', '170', '--out', str(lj)],
                                   cwd=BP, env=env, stdout=open(OUT / f'listeners_{name}.out', 'w'), stderr=subprocess.STDOUT)
            time.sleep(8)
        cmd = [sys.executable, 'scripts/live/paced_replay.py', str(audio), '--base', f'http://127.0.0.1:{PORT}', '--src', 'en', '--tgt', 'de', '--tts', 'piper', '--seconds', '60', '--load-timeout', '900', '--out', str(out_json)]
        if langs: cmd += ['--room', room]
        t = time.time(); p = subprocess.run(cmd, cwd=BP, env=env, capture_output=True, text=True, timeout=1500)
        if not out_json.exists(): raise RuntimeError('replay wrote no result: ' + (p.stdout + p.stderr)[-500:])
        d = json.loads(out_json.read_text()); r = d['rates']['1.0']; lag = r['lag_s']
        rec = {'speaker_lag_median_s': round(st.median(lag), 2), 'speaker_lag_p90_s': round(sorted(lag)[int(.9 * len(lag))], 2), 'speaker_segments': r['segments'], 'speaker_with_audio': r['with_audio'],
               'server_errors': len(re.findall(r'Traceback|ERROR', slog.read_text(errors='ignore'))), 'wall_s': round(time.time() - t)}
        if lis:
            try: lis.wait(timeout=240)
            except Exception: lis.kill()
            if lj.exists():
                L = json.loads(lj.read_text()); per = {}
                for x in L['listeners']:
                    a = per.setdefault(x['lang'], {'listeners': 0, 'captions': [], 'chunks': [], 'audio_s': [], 'first_caption_s': [], 'sample': x['sample'], 'errors': []})
                    a['listeners'] += 1; a['captions'].append(x['captions']); a['chunks'].append(x['chunks']); a['audio_s'].append(round(x['audio_s'], 1)); a['errors'] += x['errors']
                    if x['first_caption_s'] is not None: a['first_caption_s'].append(x['first_caption_s'])
                rec['listeners'] = per; rec['identical_audio_within_language'] = L['identical_audio_within_language']
        R['runs'][name] = rec; log('run', run=name, **{k: v for k, v in rec.items() if k != 'listeners'}, listener_summary={k: {kk: vv for kk, vv in v.items() if kk != 'sample'} for k, v in rec.get('listeners', {}).items()})
    except Exception as e:
        import traceback; log('run_fail', run=name, err=f'{type(e).__name__}: {e}'[:400], tb=traceback.format_exc()[-400:], server_tail=slog.read_text(errors='ignore')[-600:] if slog.exists() else '')
    finally:
        srv.terminate()
        try: srv.wait(timeout=20)
        except Exception: srv.kill()
        time.sleep(3)
for n_, l_ in PLAN: run(n_, l_)
R['status'] = 'done'; log('done', wall_s=round(time.time() - T0))
