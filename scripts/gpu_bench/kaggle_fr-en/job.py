"""Offline Kaggle runner for the PRODUCTION app.py /ws CS->EN route.

Provision a complete repo, dependencies, local STT/MT/Piper assets and the existing
CS benchmark recording as a private Kaggle input BEFORE running. Kaggle-only
dependency bootstrap; models remain offline. No fake pipeline or timing estimates.
Run: python job.py --repo /kaggle/input/bp-runtime/BP --out /kaggle/working/route_receipt.json
low_latency reproduces recoverable Mac CPU settings, not a sub-1s onset claim.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, deque
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import tempfile
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error

def kaggle_bootstrap():
    """Reuse Kaggle-lang dependency preparation, keeping models strictly offline."""
    if not Path('/kaggle/input').is_dir() or os.environ.get('BP_BENCH_BOOTSTRAPPED') == '1':
        return
    import re
    import site
    roots = [p.parent for p in Path('/kaggle/input').rglob('app.py')
             if (p.parent / 'backend').is_dir()]
    if len(roots) != 1: raise RuntimeError(f'Expected one staged BP repo, got {roots}')
    drop = re.compile(r'^(numpy|scipy|transformers|tokenizers|huggingface-hub|soundfile|numba|onnxruntime|librosa|SQLAlchemy)\b', re.I)
    lines = [re.sub(r'\s+#.*$', '', line).strip()
             for line in (roots[0] / 'requirements.txt').read_text().splitlines()]
    lines = [line for line in lines if line and not line.startswith('#') and not drop.match(line)]
    lines += ['transformers==4.57.6', 'tokenizers==0.22.2', 'huggingface-hub==0.36.2',
              'sentencepiece', 'onnxruntime', 'SQLAlchemy>=2.0.36', 'soundfile', 'scipy', 'librosa']
    req = Path('/kaggle/working/benchmark-requirements.txt')
    req.write_text('\n'.join(lines) + '\n')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(req)], check=True, timeout=600)
    libs = [str(p) for sp in site.getsitepackages() for p in Path(sp).glob('nvidia/*/lib')]
    os.environ.update(BP_BENCH_BOOTSTRAPPED='1',
                      LD_LIBRARY_PATH=':'.join(libs + [os.environ.get('LD_LIBRARY_PATH', '')]))
    os.execv(sys.executable, [sys.executable, *sys.argv])


if __name__ == '__main__':
    kaggle_bootstrap()

import numpy as np

SR = 16000
FRAME = 320
CLIP = "speaker_voices/me_en_script.m4a"  # NOTE: French source audio not available; uses EN clip as placeholder


def profile_env(profile):
    """Benchmark-only overrides; production pack/defaults remain unchanged."""
    if profile in ('default', 'low_latency'):
        return {'BP_PROFILE': profile}
    settings = {'P_short': ('1.5', '.15', '0'),
                'P_short2': ('1.0', '.12', '0'),
                'P_incr': ('3.0', '.3', '1')}
    if profile not in settings:
        raise ValueError(f'Unknown benchmark profile {profile!r}')
    maximum, silence, incremental = settings[profile]
    return dict(BP_PROFILE='default', BP_DEVICE='cuda', BP_COMPUTE_TYPE='float16',
                BP_VAD_MAX_SEGMENT_S=maximum, BP_VAD_SILENCE_S=silence,
                BP_INCREMENTAL=incremental)


def word_diff(previous, current):
    """Relative text changes, NOT reference WER or translation-quality grades."""
    before = re.findall(r'\w+', previous.lower())
    after = re.findall(r'\w+', current.lower())
    dropped, inserted = [], []
    for tag, i, j, k, l in SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        if tag in ('delete', 'replace'): dropped.extend(before[i:j])
        if tag in ('insert', 'replace'): inserted.extend(after[k:l])
    extra = Counter(after) - Counter(before)
    return {'previous_words': len(before), 'current_words': len(after),
            'dropped_words': len(dropped), 'inserted_words': len(inserted),
            'duplicated_words': sum(n for word, n in extra.items() if word in before),
            'definition': 'normalized word diff; duplicated = excess occurrences of a word present before; not truth/quality'}


def compare_texts(profiles):
    for previous, current in zip(profiles, profiles[1:]):
        current['text_comparison_previous_profile'] = previous['profile']
        for old_run, run in zip(previous['repeats'], current['repeats']):
            run['word_diff_previous'] = {key: word_diff(
                ' '.join(row[key] for row in old_run['utterances']),
                ' '.join(row[key] for row in run['utterances']))
                for key in ('recognized_cs', 'translation_en')}
            # Segmentation differs: aggregate current fragments in previous onset buckets.
            # This is approximate time alignment, not a claim of identical boundaries.
            buckets = []
            old_rows = old_run['utterances']
            for index, old in enumerate(old_rows):
                start = old['source_onset_clip_s']
                end = old_rows[index + 1]['source_onset_clip_s'] if index + 1 < len(old_rows) else run['input_s'] + 1
                matched = [row for row in run['utterances'] if start - .06 <= row['source_onset_clip_s'] < end - .06]
                buckets.append({'previous_utterance': index, 'source_window_s': [start, end],
                                'current_utterances': [row['utterance_index'] for row in matched],
                                **{key: word_diff(old[key], ' '.join(row[key] for row in matched))
                                   for key in ('recognized_cs', 'translation_en')}})
            run['word_diff_previous_onset_buckets'] = buckets


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(repo):
    paths = [repo / 'app.py', *sorted((repo / 'backend').rglob('*.py')),
             *sorted((repo / 'languages/packs').glob('*.json')), Path(__file__).resolve()]
    return {'git_hash': json.loads((repo / 'benchmark-provenance.json').read_text())['git_hash'] if (repo / 'benchmark-provenance.json').is_file() else subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
            'files_sha256': {str(p.relative_to(repo)) if p.is_relative_to(repo) else str(p): sha(p) for p in paths}}


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def decode(clip, seconds):
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(clip), '-t', str(seconds),
                                   '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'])
    audio = np.frombuffer(raw, dtype='<f4')
    if not len(audio): raise ValueError('Empty CS clip')
    return audio


def _convert_mt_runtime(source: str, target: str, repo: Path) -> None:
    """Download Opus-MT from HF and convert to CT2 if the converted dir is missing.
    Runs on Kaggle (HF internet available, ctranslate2 + transformers installed by bootstrap)."""
    model_name = f"Helsinki-NLP/opus-mt-{source}-{target}"
    ct2_dir = repo / 'ct2_models' / f'Helsinki-NLP--opus-mt-{source}-{target}'
    if ct2_dir.is_dir() and (ct2_dir / 'model.bin').is_file():
        return  # already present
    import os as _os
    _os.environ.pop('HF_HUB_OFFLINE', None)
    _os.environ.pop('TRANSFORMERS_OFFLINE', None)
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from ctranslate2.converters import TransformersConverter
    ct2_dir.mkdir(parents=True, exist_ok=True)
    converter = TransformersConverter(model_name)
    converter.convert(str(ct2_dir), quantization='int8')


def summarize(rows):
    values = [r['onset_to_first_audio_s'] for r in rows]
    if not values: raise RuntimeError('No onset-to-first-WAV measurements')
    return {'count': len(values), 'p50_s': float(np.percentile(values, 50)),
            'p95_s': float(np.percentile(values, 95))}


async def replay(base, token, audio, repeat, timeout, source='cs', target='en', drain=15):
    import websockets
    url = base.replace('http://', 'ws://') + '/ws?token=' + token
    rows, wavs, text_events = [], [], []
    pending_transcriptions, pending_results = deque(), deque()
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({'type': 'config_update', 'source_lang': source, 'target_lang': target,
                                  'stt_model_size': 'base', 'tts_model_choice': 'piper', 'vad_enabled': True}))
        async with asyncio.timeout(timeout):
            while True:
                raw = await ws.recv()
                if not isinstance(raw, str): continue
                msg = json.loads(raw)
                if msg.get('type') == 'error': raise RuntimeError(msg.get('message'))
                if msg.get('type') == 'models_loading_status' and msg.get('fully_loaded'): break
        await ws.send(json.dumps({'type': 'start'}))
        started = time.perf_counter()
        last_event = started
        async def receive():
            nonlocal last_event
            async for raw in ws:
                last_event = time.perf_counter()
                if isinstance(raw, bytes):
                    if raw[:4] != b'RIFF' or raw[8:12] != b'WAVE':
                        raise RuntimeError('Output is not WAV')
                    wavs.append({'arrival_s': last_event - started, 'bytes': len(raw),
                                 'sha256': hashlib.sha256(raw).hexdigest()})
                else:
                    msg = json.loads(raw)
                    if msg.get('type') == 'error': raise RuntimeError(msg.get('message'))
                    if msg.get('type') == 'transcription_result':
                        text = msg.get('transcribed')
                        if not isinstance(text, str): raise RuntimeError('Non-string CS result')
                        text_events.append({'type': 'transcription_result', 'text': text})
                        if text.strip(): pending_transcriptions.append(text)
                    if msg.get('type') == 'translation_result':
                        text = msg.get('translated')
                        if not isinstance(text, str): raise RuntimeError('Non-string EN result')
                        text_events.append({'type': 'translation_result', 'text': text})
                        if not pending_transcriptions: raise RuntimeError('Translation without transcription')
                        source = pending_transcriptions.popleft()
                        if text.strip(): pending_results.append((source, text))
                    if msg.get('type') == 'final_metrics':
                        metrics = msg['metrics']
                        if metrics.get('onset_to_first_audio_s') is None:
                            raise RuntimeError('Production onset metric missing')
                        if not wavs: raise RuntimeError('Metrics without received WAV')
                        if not pending_results: raise RuntimeError('Metrics without text pair')
                        source, translated = pending_results.popleft()
                        rows.append(dict(repeat=repeat, metrics=metrics,
                                         utterance_index=len(rows), recognized_cs=source, translation_en=translated,
                                         source_onset_clip_s=metrics['source_onset_time'] - started,
                                         stage_sum_s=sum(metrics[key] for key in ('stt_time', 'mt_time', 'tts_time')),
                                         onset_to_first_audio_s=metrics['onset_to_first_audio_s'],
                                         received_wav_count=len(wavs)))
        receiver = asyncio.create_task(receive())
        try:
            # Append digital silence to close the final utterance through real VAD.
            stream = np.r_[audio, np.zeros(SR, dtype=np.float32)]
            for offset in range(0, len(stream), FRAME):
                # Absolute deadlines prevent per-frame sleep drift. First packet at 20 ms
                # matches the server's receipt-minus-duration source-clock anchor.
                await asyncio.sleep(max(0, started + (offset + FRAME) / SR - time.perf_counter()))
                await ws.send(stream[offset:offset + FRAME].astype('<f4').tobytes())
                if receiver.done(): await receiver
            deadline = time.perf_counter() + drain
            while time.perf_counter() < deadline:
                if receiver.done(): await receiver
                await asyncio.sleep(.1)
            await ws.send(json.dumps({'type': 'stop'}))
        finally:
            receiver.cancel()
            try: await receiver
            except asyncio.CancelledError: pass
    if not rows: raise RuntimeError('No translated WAV/metric pairs')
    return {'repeat': repeat, 'utterances': rows, 'wavs': wavs, 'text_events': text_events,
            'unmatched_text_pairs': len(pending_results),
            'input_s': len(audio) / SR, 'drain_window_s': drain,
            'drain_complete': 'unknown; fixed observation window then Stop', 'wall_s': time.perf_counter() - started}


def serve(repo, port):
    """Production app, unchanged authentication/lifespan; separate ephemeral benchmark DB."""
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    from backend.utils import db_manager
    db_manager.SQLALCHEMY_DATABASE_URL = os.environ['BP_BENCH_DB_URL']
    import uvicorn
    # Benchmark kernel: ensure ui/ exists so the StaticFiles mount in app.py
    # does not crash when the staged dataset lacks the ui tree.
    os.makedirs("ui", exist_ok=True)
    from app import app
    uvicorn.run(app, host='127.0.0.1', port=port, log_level='warning', access_log=False)


def benchmark_profile(args, profile, audio, output):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    env = os.environ.copy()
    # Control confounding inherited knobs without reading .env or credentials.
    for key in ('BP_PROFILE', 'BP_VAD_SILENCE_S', 'BP_VAD_MAX_SEGMENT_S', 'BP_INCREMENTAL',
                'BP_STT_BEAM_SIZE', 'BP_BARGE_IN', 'BP_ENABLE_STREAMING_CAPTIONS',
                'BP_INCR_STEP_S', 'BP_INCR_MIN_WORDS', 'BP_INCR_MAX_WORDS', 'BP_INCR_MAX_WAIT_S'):
        env.pop(key, None)
    env.update(BP_PROFILE=profile, BP_DEVICE=args.device, BP_COMPUTE_TYPE=args.compute_type,
               BP_MT_DEVICE='cpu', BP_SILENCE_RMS='0', BP_HOST='127.0.0.1',
               PYTHON_DOTENV_DISABLED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               BP_BENCH_DB_URL='sqlite:///' + str(args.repo.parent / f'{profile}-{secrets.token_hex(6)}.db'))
    env.update(profile_env(profile))
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location('latency_profiles', args.repo / 'backend/latency_profiles.py')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    pack_path = args.repo / 'languages/packs' / f'{args.source}-{args.target}.json'
    if not pack_path.is_file():
        raise RuntimeError(f'Language pack not found: {pack_path}')
    pack = json.loads(pack_path.read_text())
    effective_pack, _ = module.resolve_profile(pack, env['BP_PROFILE'])
    if profile.startswith('P_'):
        effective_pack['vad'].update(max_segment_s=float(env['BP_VAD_MAX_SEGMENT_S']),
                                     silence_timeout_s=float(env['BP_VAD_SILENCE_S']),
                                     incremental=env['BP_INCREMENTAL'] == '1')
    log = output.parent / f'{profile}-server.log'
    result = {'profile': profile, 'resolved_pack': effective_pack, 'requested_stt_device': args.device,
              'effective_stt_device': 'cpu' if profile == 'low_latency' else env['BP_DEVICE'],
              'effective_stt_compute_type': 'int8' if profile == 'low_latency' else env['BP_COMPUTE_TYPE'],
              'profile_env': profile_env(profile),
              'incremental_defaults': dict(step_s=.7, min_audio_s=1., min_words=4, max_words=9, max_wait_s=1.6),
              'incremental_max_segment_note': 'production raises safety valve to >=20s while incremental; configured max remains 3s' if profile == 'P_incr' else None,
              'mt_device': 'cpu', 'tts_device': 'cpu', 'server_log': str(log), 'repeats': []}
    with log.open('w') as handle:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--serve',
                                    '--repo', str(args.repo), '--port', str(port)],
                                   env=env, cwd=args.repo, stdout=handle, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                if process.poll() is not None: raise RuntimeError(f'Server exited; see {log}')
                try:
                    with urllib.request.urlopen(base + '/api/health', timeout=1): break
                except (OSError, urllib.error.HTTPError): time.sleep(.2)
            else: raise TimeoutError(f'Server startup; see {log}')
            suffix = secrets.token_hex(6)
            account = {'username': 'bench' + suffix, 'email': f'bench{suffix}@bp.local',
                       'password': secrets.token_urlsafe(24)}
            post(base, '/api/register', account)
            login = post(base, '/api/login', {'email': account['email'], 'password': account['password']})
            token = login.get('token') or login['access_token']
            result['warmup'] = asyncio.run(replay(base, token, audio, -1, args.timeout,
                                                  args.source, args.target, args.drain))
            result['warm_state'] = 'one complete production replay before measured repeats; pooled models'
            for repeat in range(args.repeats):
                result['repeats'].append(asyncio.run(replay(base, token, audio, repeat, args.timeout,
                                                            args.source, args.target, args.drain)))
            rows = [row for run in result['repeats'] for row in run['utterances']]
            result['onset_to_first_wav'] = summarize(rows)
            result['stage_sum'] = summarize([{'onset_to_first_audio_s': r['stage_sum_s']} for r in rows])
            result['status'] = 'measured'
        except Exception as exc:
            result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        finally:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
    result['error_lines'] = [line for line in log.read_text().splitlines()
                             if re.search(r'\bERROR\b|Traceback|non-fatal', line)]
    result['stt_initialization_lines'] = [line for line in log.read_text().splitlines()
                                         if 'Initializing' in line and ('cuda' in line.lower() or 'compute_type' in line)]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, default=next((p.parent for p in Path('/kaggle/input').rglob('app.py') if (p.parent / 'benchmark-provenance.json').is_file()), Path('/kaggle/input/bp-t4-assets-20261105/BP')))
    ap.add_argument('--audio', type=Path, help='Existing source-language recording; default pack benchmark clip')
    ap.add_argument('--source', default='cs', help='Source language code (default cs)')
    ap.add_argument('--target', default='en', help='Target language code (default en)')
    ap.add_argument('--seconds', type=float, default=15)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    ap.add_argument('--profiles', default=os.environ.get('BP_BENCH_PROFILES', 'default,P_short,P_short2,P_incr'))
    ap.add_argument('--compute-type', default='float16')
    ap.add_argument('--timeout', type=float, default=180)
    ap.add_argument('--drain', type=float, default=15)
    ap.add_argument('--out', type=Path, default=Path('/kaggle/working/route_receipt.json'))
    ap.add_argument('--serve', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--port', type=int, default=8013, help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.repo = args.repo.resolve()
    if args.serve: return serve(args.repo, args.port)
    # Kaggle CLI cannot pass CLI args reliably; fall back to env vars for direction.
    args.source = os.environ.get('SOURCE_LANG', 'fr')
    args.target = os.environ.get('TARGET_LANG', 'en')
    if args.repeats < 3 or args.seconds <= 0: ap.error('Require >=3 repeats and positive duration')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    clip = args.audio or args.repo / CLIP
    receipt = {'schema': 1, 'route': 'app.py:/ws', 'direction': f'{args.source}-{args.target}', 'pace': 1.0,
               'metric': 'server first successful WAV send completion minus ingress-anchored first voiced sample',
               'excludes': ['browser playback', 'physical capture/output'],
               'host': platform.platform(), 'profiles': []}
    try:
        receipt['code'] = fingerprint(args.repo)
        import importlib.metadata as md
        receipt['runtime'] = {'python': sys.version, 'packages': {name: md.version(name) for name in ('faster-whisper', 'ctranslate2', 'transformers', 'piper-tts', 'numpy', 'SQLAlchemy')}}
        receipt['gpu_inventory'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], text=True).strip()
        if 'T4' not in receipt['gpu_inventory']: raise RuntimeError('Expected Kaggle T4')
        receipt['audio'] = {'path': str(clip), 'sha256': sha(clip), 'language': 'cs'}
        audio = decode(clip, args.seconds)
        # Kaggle datasets are read-only; production debug/session output needs a writable checkout.
        runtime = Path(tempfile.mkdtemp(prefix='bp-route-')) / 'BP'
        shutil.copytree(args.repo, runtime, ignore=shutil.ignore_patterns('.env', '.env.*', 'certs', '.git', '__pycache__'))
        args.repo = runtime
        # Convert MT model to CT2 at runtime if missing (kernel has HF internet + ctranslate2/transformers).
        _convert_mt_runtime(args.source, args.target, args.repo)
        receipt['audio']['decoded_s'] = len(audio) / SR
        for profile in args.profiles.split(','):
            receipt['profiles'].append(benchmark_profile(args, profile, audio, args.out))
            args.out.write_text(json.dumps(receipt, indent=2) + '\n')
        compare_texts(receipt['profiles'])
        receipt['status'] = 'measured' if all(p['status'] == 'measured' for p in receipt['profiles']) else 'failed'
    except Exception as exc:
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    args.out.write_text(json.dumps(receipt, indent=2) + '\n')
    if receipt['status'] != 'measured': raise SystemExit('Benchmark failed; inspect receipt')


if __name__ == '__main__':
    main()
