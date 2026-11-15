#!/usr/bin/env python3
"""Standalone Higgs TTS 3 research/non-commercial benchmark; never imports BP.

Sources inspected 2026-11-15 (installed 0.2.0 source/help is also recorded):
https://huggingface.co/bosonai/higgs-tts-3-4b/blob/main/README.md
https://github.com/sgl-project/sglang-omni/blob/main/docs/cookbook/higgs_tts.md
https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/higgs_tts/config.py
https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/higgs_tts/engine_builder.py
https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/higgs_tts/stages.py
https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/cli/serve.py
https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/serve/protocol.py
https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/serve/openai_api.py
https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/arg_groups/choices.py
https://github.com/SYSTRAN/faster-whisper/blob/master/README.md

T4/dtype/streaming are experiments, not compatibility claims. No project env.
"""
import atexit
import base64
import io
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
import unicodedata
import urllib.request
import wave
from pathlib import Path

OUT = Path('/kaggle/working')
REPORT = OUT / 'results.json'
T0 = time.time()
R = {'schema': 2, 'status': 'running', 'steps': [], 'errors': [], 'tts': {}, 'infrastructure': {}}
SERVER_PROC = None


def save():
    R['elapsed_s'] = round(time.time() - T0, 1)
    tmp = REPORT.with_suffix('.tmp')
    tmp.write_text(json.dumps(R, indent=1, ensure_ascii=False))
    tmp.replace(REPORT)


def step(name, **kw):
    R['steps'].append({'t': round(time.time() - T0, 1), 'step': name, **kw})
    print(f'[{time.time() - T0:7.1f}s] {name} {kw}', flush=True)
    save()


MODEL = 'bosonai/higgs-tts-3-4b'
BASE = 'http://127.0.0.1:8000'
WORK = Path('/tmp/higgs-tts3-eval')
TTS_PY = str(WORK / 'tts/bin/python')
ASR_PY = str(WORK / 'asr/bin/python')
TEXTS = {
    'en': {
        'short': 'Hello, this is a test of real-time speech translation.',
        'long': 'During our international meeting today, the interpreter will translate every spoken sentence clearly, preserve the original meaning, and help all participants understand each other better.',
    },
    'sk': {
        'short': 'Dobrý deň, toto je test prekladov reči v reálnom čase.',
        'long': 'Počas medzinárodného stretnutia bude tlmočník jasne prekladať každú vyslovenú vetu, zachová pôvodný význam a pomôže všetkým účastníkom lepšie porozumieť ostatným aj pri zložitých odborných témach.',
    },
    'cs': {
        'short': 'Dobrý den, toto je test překladu řeči v reálném čase.',
        'long': 'Během mezinárodního setkání bude tlumočník jasně překládat každou vyslovenou větu, zachová původní význam a pomůže všem účastníkům lépe porozumět ostatním i při složitých odborných tématech.',
    },
}
# Clean Python subprocesses cannot inherit the project import path/venv.
ENV = os.environ.copy()
# 2026-11-15 run: FlashInfer JIT link failed with "cannot find -lcuda" (no libcuda.so dev symlink on Kaggle): provide one
def _libcuda_link():
    import glob
    cands = glob.glob('/usr/lib/x86_64-linux-gnu/libcuda.so.1') + glob.glob('/usr/local/nvidia/lib64/libcuda.so.1') + glob.glob('/usr/lib64/libcuda.so.1')
    if not cands:
        return
    d = '/tmp/cudalink'
    os.makedirs(d, exist_ok=True)
    link = d + '/libcuda.so'
    if not os.path.exists(link):
        os.symlink(cands[0], link)
    ENV['LIBRARY_PATH'] = d + ':' + ENV.get('LIBRARY_PATH', '')
    ENV['LD_LIBRARY_PATH'] = d + ':' + ENV.get('LD_LIBRARY_PATH', '')
_libcuda_link()
for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'):
    ENV.pop(key, None)
ENV.update(HF_HOME=str(WORK / 'hf'), UV_CACHE_DIR=str(WORK / 'uv-cache'),
           PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES='0')
ENV['PATH'] = os.pathsep.join(p for p in ENV.get('PATH', '').split(os.pathsep)
                              if not any(x in p for x in ('.venv', '/venv/', '/envs/')))


def error(stage, exc, lang=None):
    item = {'stage': stage, 'lang': lang, 'error': f'{type(exc).__name__}: {exc}'}
    R['errors'].append(item)
    step('error', **item)
    return item


def run(argv, timeout=120, env=None):
    p = subprocess.run(argv, cwd=WORK, env=env or ENV, capture_output=True,
                       text=True, timeout=timeout)
    step('command', argv=argv, rc=p.returncode, output=(p.stdout + p.stderr)[-6000:])
    if p.returncode:
        raise RuntimeError(f'{argv[0]} exited {p.returncode}: {(p.stdout + p.stderr)[-3000:]}')
    return p.stdout


def stop_server():
    global SERVER_PROC
    if SERVER_PROC is not None:
        # Kill the session, including stage subprocesses, even if the CLI exited.
        try:
            os.killpg(SERVER_PROC.pid, signal.SIGTERM)
            SERVER_PROC.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
        try:
            os.killpg(SERVER_PROC.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        SERVER_PROC = None


atexit.register(stop_server)

GATE = r'''
import json, subprocess, torch, traceback
r = {'torch': torch.__version__, 'cuda': torch.version.cuda,
     'arch_list': torch.cuda.get_arch_list(), 'cuda_available': torch.cuda.is_available()}
r['sm_75_in_arch_list'] = 'sm_75' in r['arch_list']
r['nvidia_smi'] = subprocess.check_output(['nvidia-smi'], text=True)
r['fp16_matmul_ok'] = False
try:
    r['device'] = torch.cuda.get_device_name(0)
    r['capability'] = list(torch.cuda.get_device_capability(0))
    a = torch.ones((32, 32), device='cuda', dtype=torch.float16)
    b = a @ a
    torch.cuda.synchronize()
    r['fp16_matmul_value'] = b[0, 0].item()
    r['fp16_matmul_ok'] = bool(torch.isfinite(b).all().item() and b[0, 0].item() == 32)
except Exception:
    r['matmul_error'] = traceback.format_exc()
r['supported'] = r['sm_75_in_arch_list'] and r['fp16_matmul_ok']
print('GATE_JSON=' + json.dumps(r))
'''


def t4_gate(label):
    try:
        output = run([TTS_PY, '-I', '-c', GATE], timeout=120)
        facts = json.loads(next(line[10:] for line in output.splitlines() if line.startswith('GATE_JSON=')))
    except Exception as exc:
        facts = {'supported': False, 'error': str(exc)}
        error(label, exc)
    R['infrastructure'][label] = facts
    save()
    return facts['supported']


ASR_CODE = r'''
import json, sys, time, traceback
import numpy as np, soundfile as sf
from faster_whisper import WhisperModel

def load16k(path):
    # decode with soundfile, not PyAV: PyAV/faster-whisper kwarg clash failed twice on 2026-11-15
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        n = int(round(len(x) * 16000 / sr))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype('float32')
    return x
jobs = json.loads(sys.argv[1])
t = time.perf_counter()
# CPU int8 avoids CUDA-12 ASR libraries colliding with the CUDA-13 TTS stack.
model = WhisperModel('large-v3-turbo', device='cpu', compute_type='int8',
                     download_root=sys.argv[2])
load_s = time.perf_counter() - t
results = {}
for job in jobs:
    try:
        t = time.perf_counter()
        segments, info = model.transcribe(load16k(job['path']), language=job['lang'], beam_size=5)
        text = ' '.join(s.text.strip() for s in segments)
        results[job['path']] = {'text': text, 'wall_s': time.perf_counter() - t,
                                'language': info.language}
    except Exception:
        results[job['path']] = {'error': traceback.format_exc()}
print('ASR_JSON=' + json.dumps({'model_load_s': load_s, 'device': 'cpu',
      'compute_type': 'int8', 'model': 'large-v3-turbo', 'results': results}))
'''


def transcribe(jobs):
    output = run([ASR_PY, '-I', '-c', ASR_CODE, json.dumps(jobs), str(WORK / 'asr-models')], timeout=1200)
    return json.loads(next(s[9:] for s in output.splitlines() if s.startswith('ASR_JSON=')))


def prepare_references():
    roots = [Path('/kaggle/input'), Path.cwd()]
    # Dataset packaging differs; find real files by exact documented suffix.
    refs = {}
    for lang in TEXTS:
        try:
            candidates = []
            for root in roots:
                candidates += sorted(root.glob(f'**/speaker_voices/me_{lang}*.m4a'))
                candidates += sorted(root.glob(f'**/processed/testaudio/{lang}*_paced.wav'))
            if not candidates:
                raise FileNotFoundError(f'No real {lang} reference under {roots}')
            dest = WORK / f'ref_{lang}.wav'
            run(['ffmpeg', '-nostdin', '-y', '-v', 'error', '-i', str(candidates[0]),
                 '-t', '10', '-ac', '1', '-ar', '24000', '-c:a', 'pcm_s16le', str(dest)], 60)
            with wave.open(str(dest)) as f:
                duration = f.getnframes() / f.getframerate()
                if f.getnchannels() != 1 or abs(duration - 10) > 0.01:
                    raise ValueError(f'Reference must contain 10 seconds of real mono speech: {duration}')
            refs[lang] = {'path': str(dest), 'source': str(candidates[0]), 'audio_s': duration}
        except Exception as exc:
            error('reference', exc, lang)
            R['tts'][lang]['status'] = 'reference_failed'
    if not refs:
        raise RuntimeError('No valid reference voices; refusing synthetic fallback')
    decoded = transcribe([{'path': ref['path'], 'lang': lang} for lang, ref in refs.items()])
    R['reference_asr'] = decoded
    for lang, ref in list(refs.items()):
        try:
            row = decoded['results'][ref['path']]
            if 'error' in row or not row.get('text', '').strip():
                raise ValueError(f'Reference transcription failed: {row}')
            ref['text'] = row['text']
            R['tts'][lang]['reference'] = ref
        except Exception as exc:
            error('reference_asr', exc, lang)
            R['tts'][lang]['status'] = 'reference_failed'
            del refs[lang]
    if not refs:
        raise RuntimeError('No transcribed reference voices')
    save()
    return refs


def tail(path):
    return '\n'.join(path.read_text(errors='replace').splitlines()[-60:])


def serve():
    global SERVER_PROC
    cli = str(WORK / 'tts/bin/sgl-omni')
    run([cli, 'serve', '--help'], 90)
    # Preserve the actual installed source evidence; no assumption that main == 0.2.0.
    source = run([TTS_PY, '-I', '-c', '''
import importlib.util, pathlib
root = pathlib.Path(importlib.util.find_spec('sglang_omni').origin).parent
for rel in ['models/higgs_tts/config.py', 'models/higgs_tts/engine_builder.py',
            'models/higgs_tts/stages.py', 'serve/protocol.py', 'cli/serve.py']:
    print('SOURCE', rel)
    for line in (root / rel).read_text().splitlines():
        if any(s in line for s in ('dtype', 'attention_backend', 'process=', 'stream_format', 'parse_extra_args')):
            print(line)
'''], 90)
    R['infrastructure']['installed_source_excerpt'] = source
    R['infrastructure']['jit_strategy'] = {
        'attempt': 'tts_engine.engine.attention_backend=triton; disable_cuda_graph=true',
        'limit': 'Avoids FlashInfer attention selection; imports or other kernels may still JIT. No global no-JIT guarantee.',
        'reference_encoding': 'CPU float32; preserve cookbook default separate processes',
    }
    attempts = R['serve_attempts'] = []
    overall_deadline = time.monotonic() + 720
    for dtype in ('float16', 'bfloat16'):
        log = OUT / f'serve_{dtype}.log'
        cmd = [cli, 'serve', '--model-path', MODEL, '--port', '8000',
               '--allowed-local-media-path', str(WORK),
               '--audio_encoder.factory.device', 'cpu',
               '--audio_encoder.factory.dtype', 'float32',
               '--tts_engine.engine.dtype', dtype,
               '--vocoder.factory.dtype', dtype,
               '--tts_engine.engine.attention_backend', 'triton',
               '--tts_engine.engine.disable_cuda_graph', 'true']
        attempt = {'dtype': dtype, 'argv': cmd, 'log': str(log)}
        attempts.append(attempt)
        try:
            started = time.perf_counter()
            with log.open('w') as output:
                SERVER_PROC = subprocess.Popen(cmd, cwd=WORK, env=ENV, stdout=output,
                                               stderr=subprocess.STDOUT, start_new_session=True)
            # Reserve six minutes for fp16 if bf16 stalls; total health budget <=12m.
            deadline = min(overall_deadline, time.monotonic() + (720 if dtype == 'float16' else 350))
            step('serve_attempt', **attempt)
            while time.monotonic() < deadline:
                if SERVER_PROC.poll() is not None:
                    raise RuntimeError(f'Server exited {SERVER_PROC.returncode}')
                try:
                    with urllib.request.urlopen(BASE + '/health', timeout=2) as resp:
                        health = json.load(resp)
                    if health.get('status') == 'healthy':
                        attempt.update(status='healthy', model_load_s=time.perf_counter() - started,
                                       model_load_definition='CLI launch to healthy; includes download, initialization and server warmup',
                                       health=health)
                        step('server_ready', dtype=dtype, model_load_s=attempt['model_load_s'])
                        return attempt['model_load_s']
                except Exception as exc:
                    attempt['last_health_error'] = str(exc)
                time.sleep(min(2, max(0, deadline - time.monotonic())))
            raise TimeoutError('Server did not become healthy within attempt budget')
        except Exception as exc:
            attempt.update(status='failed', error=str(exc))
            stop_server()
            attempt['last_60_log_lines'] = tail(log) if log.exists() else ''
            error('serve_' + dtype, exc)
    R['last_60_log_lines'] = attempts[-1].get('last_60_log_lines', '')
    return None


def gpu_sample():
    output = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.used',
                                      '--format=csv,noheader,nounits'], text=True, timeout=5)
    return float(output.strip())


def wav_parts(raw):
    with wave.open(io.BytesIO(raw)) as f:
        params = (f.getnchannels(), f.getsampwidth(), f.getframerate())
        return params, f.readframes(f.getnframes())


def synthesize(text, ref, mode, output):
    payload = {'input': text, 'references': [{'audio_path': ref['path'], 'text': ref['text']}],
               'temperature': 0.8, 'top_k': 50, 'max_new_tokens': 1024}
    if mode == 'pcm':
        payload.update(stream=True, response_format='pcm')
    elif mode == 'card_sse':
        payload.update(stream=True)  # Exact model-card request; inspect actual content type.
    elif mode == 'explicit_sse':
        payload.update(stream=True, stream_format='sse', response_format='wav')
    else:
        payload.update(stream=False, response_format='wav')
    samples, monitor_errors = [], []
    done = threading.Event()

    def monitor():
        while not done.is_set():
            try:
                samples.append(gpu_sample())
            except Exception as exc:
                monitor_errors.append(str(exc))
                return
            done.wait(0.2)

    samples.append(gpu_sample())
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    first = None
    chunks = []
    params = (1, 2, 24000)
    t = time.perf_counter()
    try:
        request = urllib.request.Request(BASE + '/v1/audio/speech', data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=180) as resp:
            content_type = resp.headers.get('Content-Type', '').lower()
            if 'text/event-stream' in content_type:
                observed = 'sse_wav'
                terminal = False
                for line in resp:
                    if not line.startswith(b'data:'):
                        continue
                    data = line[5:].strip()
                    if data == b'[DONE]':
                        terminal = True
                        break
                    event = json.loads(data)
                    if event.get('error'):
                        raise RuntimeError(str(event['error']))
                    audio = event.get('audio') or {}
                    if audio.get('data'):
                        new_params, frames = wav_parts(base64.b64decode(audio['data'], validate=True))
                        if chunks and new_params != params:
                            raise ValueError('SSE chunk WAV formats differ')
                        params = new_params
                        if frames:
                            if first is None:
                                first = time.perf_counter() - t
                            chunks.append(frames)
                    if event.get('finish_reason') == 'stop':
                        terminal = True
                        break
                if not terminal:
                    raise ValueError('SSE ended without stop/DONE')
            elif 'audio/pcm' in content_type or 'audio/l16' in content_type:
                observed = 'pcm'
                params = (int(resp.headers.get('x-channels', 1)), 2,
                          int(resp.headers.get('x-sample-rate', 24000)))
                while True:
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    if first is None:
                        first = time.perf_counter() - t
                    chunks.append(chunk)
            elif 'wav' in content_type or 'octet-stream' in content_type:
                observed = 'non-streaming'
                params, frames = wav_parts(resp.read())
                chunks = [frames]
                first = time.perf_counter() - t
            else:
                raise ValueError(f'Unexpected response {content_type}: {resp.read(500)!r}')
        wall = time.perf_counter() - t
        frames = b''.join(chunks)
        if not frames or first is None or len(frames) % (params[0] * params[1]):
            raise ValueError('Empty or misaligned audio')
        with wave.open(str(output), 'wb') as f:
            f.setnchannels(params[0]); f.setsampwidth(params[1]); f.setframerate(params[2])
            f.writeframes(frames)
        duration = len(frames) / (params[0] * params[1] * params[2])
        return {'first_audio_latency_s': first, 'wall_s': wall, 'audio_s': duration,
                'rtf': wall / duration, 'streaming': observed,
                'requested_protocol': mode, 'content_type': content_type,
                'output_wav': str(output)}
    finally:
        done.set()
        thread.join(timeout=6)
        # Mutable result metadata is collected by caller even on an exception.
        R['last_gpu_sample'] = {'peak_gpu_mb': max(samples) if samples else None,
                                'gpu_memory_method': 'nvidia-smi GPU0 total used MiB, sampled every 0.2s; includes server resident memory',
                                'samples': len(samples), 'errors': monitor_errors}


def normalise(text):
    return ''.join(c for c in unicodedata.normalize('NFC', text.lower())
                   if not unicodedata.category(c).startswith('P')).split()


def wer(reference, hypothesis):
    a, b = normalise(reference), normalise(hypothesis)
    row = list(range(len(b) + 1))
    for i, word in enumerate(a, 1):
        nxt = [i]
        for j, other in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (word != other)))
        row = nxt
    return {'wer': row[-1] / len(a), 'word_errors': row[-1], 'reference_words': len(a),
            'normalised_reference': ' '.join(a), 'normalised_hypothesis': ' '.join(b)}


def benchmark(refs, model_load):
    for lang, ref in refs.items():
        row = R['tts'][lang]
        row.update(status='running', model_load_s=model_load, cases={})
        try:
            probes = row['protocol_probes'] = []
            selected = None
            for mode in ('pcm', 'card_sse', 'explicit_sse'):
                try:
                    measurement = synthesize(TEXTS[lang]['short'], ref, mode,
                                             OUT / f'higgs_{lang}_probe_{mode}.wav')
                    measurement.update(R['last_gpu_sample'])
                    probes.append({'status': 'ok', **measurement})
                    if selected is None:
                        selected = mode
                except Exception as exc:
                    probes.append({'mode': mode, 'status': 'failed', **error('protocol_' + mode, exc, lang)})
                save()
            if selected is None:
                selected = 'non-streaming'  # Explicitly measured and labelled, never called TTFC.
            for case, text in TEXTS[lang].items():
                case_row = row['cases'][case] = {'text': text, 'word_count': len(text.split()), 'runs': []}
                try:
                    case_row['warmup'] = synthesize(text, ref, selected, OUT / f'higgs_{lang}_{case}_warmup.wav')
                    for repeat in range(3):
                        try:
                            path = OUT / f'higgs_{lang}_{case}_r{repeat + 1}.wav'
                            measurement = synthesize(text, ref, selected, path)
                            measurement.update(R['last_gpu_sample'], repeat=repeat + 1, model_load_s=model_load)
                            case_row['runs'].append(measurement)
                        except Exception as exc:
                            case_row['runs'].append({'repeat': repeat + 1, **error(case, exc, lang)})
                        save()
                    good = [r for r in case_row['runs'] if 'error' not in r]
                    if len(good) != 3:
                        raise RuntimeError(f'{case}: only {len(good)}/3 repeats succeeded; no valid median')
                    keys = ('model_load_s', 'first_audio_latency_s', 'wall_s', 'audio_s', 'rtf', 'peak_gpu_mb')
                    case_row['median'] = {k: statistics.median(r[k] for r in good) for k in keys}
                    case_row.update(status='ok', streaming=good[0]['streaming'])
                    # Canonical WAV is an actual run nearest median wall time, not synthetic audio.
                    representative = min(good, key=lambda r: abs(r['wall_s'] - case_row['median']['wall_s']))
                    dest = OUT / (f'higgs_{lang}.wav' if case == 'short' else f'higgs_{lang}_long.wav')
                    shutil.copyfile(representative['output_wav'], dest)
                    case_row.update(output_wav=str(dest), representative_source=representative['output_wav'])
                except Exception as exc:
                    case_row.update(status='failed', error=str(exc))
                    error('case_' + case, exc, lang)
                save()
            row['status'] = 'ok' if all(c['status'] == 'ok' for c in row['cases'].values()) else 'failed'
        except Exception as exc:
            row.update(status='failed', error=str(exc))
            error('language', exc, lang)
        save()


def round_trip():
    jobs, targets = [], {}
    for lang, row in R['tts'].items():
        for case in row.get('cases', {}).values():
            for measurement in case['runs']:
                if 'output_wav' in measurement:
                    path = measurement['output_wav']
                    jobs.append({'path': path, 'lang': lang})
                    targets[path] = (lang, case['text'], measurement)
    if not jobs:
        return
    try:
        decoded = transcribe(jobs)
        R['round_trip_asr'] = {k: v for k, v in decoded.items() if k != 'results'}
        for path, (lang, text, measurement) in targets.items():
            try:
                result = decoded['results'][path]
                if 'error' in result:
                    raise RuntimeError(result['error'])
                measurement['asr'] = {**result, **wer(text, result['text'])}
            except Exception as exc:
                measurement['asr'] = error('round_trip', exc, lang)
                R['tts'][lang]['status'] = 'asr_failed'
        for row in R['tts'].values():
            for case in row.get('cases', {}).values():
                values = [r['asr']['wer'] for r in case['runs'] if 'wer' in r.get('asr', {})]
                if len(values) == 3 and 'median' in case:
                    case['median']['wer'] = statistics.median(values)
                    representative = next(r for r in case['runs'] if r.get('output_wav') == case['representative_source'])
                    case['asr'] = representative['asr']
    except Exception as exc:
        error('round_trip_asr', exc)
        for row in R['tts'].values():
            if row['status'] == 'ok':
                row['status'] = 'asr_failed'
    save()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    R.update(model=MODEL, license='Research and non-commercial; not a production adoption test',
             texts=TEXTS, supported_languages=['en', 'sk', 'cs'],
             support_source='Model card production-quality tier; language quality is unmeasured here',
             tts={lang: {'status': 'not_run'} for lang in TEXTS})
    step('env', python=sys.version, executable=sys.executable, cwd=os.getcwd())
    try:
        # STEP 0 FIRST: no model downloads, ASR setup or serving before this gate.
        run(['nvidia-smi'], 30)
        run([sys.executable, '-m', 'pip', 'install', 'uv'], 120)
        run(['uv', 'python', 'install', '3.12'], 300)
        run(['uv', 'venv', '--python', '3.12', str(WORK / 'tts')], 120)
        run(['uv', 'pip', 'install', '--python', TTS_PY, 'torch==2.14.1'], 900)
        if not t4_gate('initial_t4_gate'):
            R['status'] = 'unsupported_on_T4'
            save()
            return
        run(['uv', 'pip', 'install', '--python', TTS_PY, '--prerelease=allow', 'sglang-omni==0.2.0'], 1200)
        if not t4_gate('installed_stack_t4_gate'):
            R['status'] = 'unsupported_on_T4'
            save()
            return
        run(['uv', 'pip', 'freeze', '--python', TTS_PY], 30)
        run(['uv', 'venv', '--python', '3.12', str(WORK / 'asr')], 120)
        run(['uv', 'pip', 'install', '--python', ASR_PY, 'faster-whisper', 'soundfile', 'numpy'], 600)
        run(['uv', 'pip', 'freeze', '--python', ASR_PY], 30)
        refs = prepare_references()
        try:
            model_load = serve()
        except Exception as exc:
            error('serve_setup', exc)
            model_load = None
        if model_load is None:
            R['status'] = 'failed_to_serve'
            save()
            return
        benchmark(refs, model_load)
        stop_server()
        round_trip()
        R['status'] = 'complete' if all(r['status'] == 'ok' for r in R['tts'].values()) else 'partial_failure'
    except Exception as exc:
        error('fatal', exc)
        R['status'] = 'failed'
        R['traceback'] = traceback.format_exc()
    finally:
        stop_server()
        save()


if __name__ == '__main__':
    main()
