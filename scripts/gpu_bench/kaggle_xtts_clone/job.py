#!/usr/bin/env python3
"""Standalone XTTS-v2 T4 evaluation; CPML non-commercial weights.

Sources (checked 2026-11-15):
https://coqui-tts.readthedocs.io/en/latest/models/xtts.html#streaming-manually
https://github.com/idiap/coqui-ai-TTS/blob/dev/TTS/tts/models/xtts.py
https://github.com/idiap/coqui-ai-TTS/blob/dev/TTS/utils/manage.py
https://github.com/SYSTRAN/faster-whisper#usage

Only stdlib runs in the host interpreter. Both workers use independent uv venvs.
Streaming latency ends at the first nonempty yielded CUDA chunk, synchronized;
it does not include reference conditioning, playback, or transfer to the browser.
"""
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import unicodedata
import wave
from pathlib import Path

OUT = Path('/kaggle/working')
REPORT = OUT / 'results.json'
T0 = time.time()
R = {'schema': 2, 'status': 'running', 'steps': [], 'errors': [], 'tts': {}, 'infrastructure': {}}


def save():
    R['elapsed_s'] = round(time.time() - T0, 1)
    tmp = REPORT.with_suffix('.tmp')
    tmp.write_text(json.dumps(R, indent=1, ensure_ascii=False))
    tmp.replace(REPORT)


def step(name, **kw):
    R['steps'].append({'t': round(time.time() - T0, 1), 'step': name, **kw})
    print(f'[{time.time() - T0:7.1f}s] {name} {kw}', flush=True)
    save()


TEXTS = {
    'en': {'short': 'Hello, this is a test of real-time speech translation.',
           'long': 'During our meeting today, we will test whether this system can translate spoken sentences clearly while preserving the voice and delivering audio without long delays.'},
    'sk': {'short': 'Dobrý deň, toto je test prekladov reči v reálnom čase.',
           'long': 'Počas dnešného pracovného stretnutia budeme spoločne skúšať, či systém dokáže prekladať hovorené vety zrozumiteľne, zachovať pôvodný hlas a priebežne poskytovať zvuk bez dlhých zbytočných oneskorení.'},
    'cs': {'short': 'Dobrý den, toto je test překladu řeči v reálném čase.',
           'long': 'Během dnešního pracovního setkání budeme společně zkoušet, zda systém dokáže překládat mluvené věty srozumitelně, zachovat původní hlas a průběžně poskytovat zvuk bez dlouhých zbytečných prodlev.'},
}
SUPPORTED = ('en', 'cs')
TTS_ENV = Path('/tmp/xtts-venv')
ASR_ENV = Path('/tmp/xtts-asr-venv')


def run(cmd, timeout=900, env=None):
    p = subprocess.run([str(x) for x in cmd], stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, timeout=timeout, env=env)
    if p.returncode:
        raise RuntimeError(f'{cmd!r}: rc={p.returncode}\n{(p.stdout + p.stderr)[-6000:]}')
    return (p.stdout + p.stderr)[-6000:]


def error(stage, exc, target=None):
    item = {'stage': stage, 'error': f'{type(exc).__name__}: {exc}',
            'traceback': traceback.format_exc()[-6000:]}
    R['errors'].append(item)
    if target is not None:
        target.setdefault('errors', []).append(item)
        target['status'] = 'failed'
    step('error', **item)


def inventory():
    from importlib.metadata import distributions
    return {d.metadata['Name']: d.version for d in distributions()}


def reference(lang):
    # Search only real staged repo audio. No generated reference fallback.
    roots = [Path.cwd()]
    roots.extend(p.parent for p in Path('/kaggle/input').rglob('benchmark-provenance.json'))
    roots.extend(p.parent for p in Path('/kaggle/input').rglob('app.py'))
    candidates = []
    for root in dict.fromkeys(roots):
        candidates.extend(sorted((root / 'speaker_voices').glob(f'me_{lang}*.m4a')))
        candidates.extend(sorted((root / 'processed/testaudio').glob(f'{lang}*_paced.wav')))
    if not candidates:
        raise FileNotFoundError(f'No real {lang} reference under {list(map(str, roots))}')
    failures = []
    for source in dict.fromkeys(candidates):
        try:
            dest = OUT / f'ref_{lang}.wav'
            run(['ffmpeg', '-y', '-v', 'error', '-i', source, '-t', '10',
                 '-ac', '1', '-ar', '22050', '-c:a', 'pcm_s16le', dest], 60)
            with wave.open(str(dest)) as wav:
                duration = wav.getnframes() / wav.getframerate()
                if wav.getnchannels() != 1 or abs(duration - 10) > 1 / 22050:
                    raise ValueError(f'Reference must contain 10 seconds, got {duration}')
            return {'source': str(source), 'wav': str(dest), 'audio_s': duration,
                    'sample_rate': 22050, 'channels': 1, 'rejected_candidates': failures}
        except Exception as exc:
            failures.append({'source': str(source), 'error': str(exc)})
    raise RuntimeError(f'No valid 10-second {lang} reference: {failures}')


def tts_worker():
    attempt = sys.argv[2]
    R['tts'] = {'sk': {'status': 'unsupported', 'reason': 'XTTS-v2 has no Slovak language',
                       'texts': TEXTS['sk']}}
    model = None
    try:
        R['infrastructure']['packages'] = inventory()
        import torch
        import soundfile as sf
        from huggingface_hub import snapshot_download
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required: refusing a CPU benchmark')
        R['infrastructure'].update(device=torch.cuda.get_device_name(0),
                                   torch_cuda=torch.version.cuda, python=sys.version,
                                   executable=sys.executable)
        # Verify the installed manager predicate, without invoking a prompt or download.
        try:
            from TTS.utils.manage import ModelManager
            R['tos'] = {'env': os.environ.get('COQUI_TOS_AGREED'),
                        'installed_predicate_result': ModelManager.tos_agreed(
                            {'tos_required': True}, Path('/tmp/xtts-tos-probe-no-file')),
                        'high_level_prompt': 'not exercised; low-level HF checkpoint path used'}
        except Exception as exc:
            error('tos_predicate', exc)
        started = time.perf_counter()
        checkpoint = snapshot_download('coqui/XTTS-v2', cache_dir='/tmp/xtts-hf',
                                       allow_patterns=['config.json', 'model.pth', 'vocab.json',
                                                       'speakers_xtts.pth'])
        R['download_s'] = time.perf_counter() - started
        R['checkpoint_path'] = checkpoint
        config = XttsConfig()
        config.load_json(str(Path(checkpoint) / 'config.json'))
        R['checkpoint_languages'] = config.languages
        torch.cuda.synchronize()
        started = time.perf_counter()
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=checkpoint, use_deepspeed=False)
        model.cuda().eval()
        torch.cuda.synchronize()
        R['model_load_s'] = time.perf_counter() - started
        step('model_loaded', model_load_s=R['model_load_s'])
    except Exception as exc:
        error('model_setup', exc)

    for lang in SUPPORTED:
        result = R['tts'][lang] = {'status': 'running', 'texts': TEXTS[lang], 'cases': {},
                                  'model_load_s': R.get('model_load_s')}
        try:
            if model is None or 'model_load_s' not in R:
                raise RuntimeError('Model setup failed; see top-level errors')
            if lang not in config.languages:
                raise ValueError(f'{lang} absent from downloaded checkpoint languages')
            ref = reference(lang)
            result['reference'] = ref
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                latent, embedding = model.get_conditioning_latents(audio_path=[ref['wav']])
            torch.cuda.synchronize()
            result['conditioning_s'] = time.perf_counter() - started
            result['conditioning_cached'] = True
            for length, text in TEXTS[lang].items():
                for mode in ('streaming', 'plain'):
                    key = f'{length}_{mode}'
                    case = result['cases'][key] = {'status': 'running', 'text': text,
                                                   'mode': mode, 'repeats': []}
                    try:
                        # One warm-up, then three measured repeats per text and API.
                        for repeat in range(-1, 3):
                            torch.manual_seed(20261115 + max(0, repeat))
                            audio = None
                            chunks = []
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                            started = time.perf_counter()
                            first = None
                            chunks = []
                            with torch.inference_mode():
                                if mode == 'streaming':
                                    for chunk in model.inference_stream(
                                            text, lang, latent, embedding, stream_chunk_size=20):
                                        if chunk.numel():
                                            torch.cuda.synchronize()
                                            if first is None:
                                                first = time.perf_counter() - started
                                            chunks.append(chunk)
                                    if not chunks:
                                        raise RuntimeError('No audio chunks yielded')
                                    audio = torch.cat(chunks, dim=0)
                                else:
                                    audio = model.inference(text, lang, latent, embedding)['wav']
                                torch.cuda.synchronize()
                                wall = time.perf_counter() - started
                            peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
                            if mode == 'plain':
                                first = wall
                            if isinstance(audio, torch.Tensor):
                                audio = audio.detach().cpu().numpy()
                            audio = audio.reshape(-1)
                            audio_s = len(audio) / 24000
                            if audio_s <= 0:
                                raise RuntimeError('Empty synthesis output')
                            if repeat == -1:
                                case['warmup_wall_s'] = wall
                                continue
                            path = OUT / f'xtts_{lang}_{key}_r{repeat + 1}.wav'
                            sf.write(str(path), audio, 24000)
                            case['repeats'].append({
                                'repeat': repeat + 1, 'wall_s': wall, 'audio_s': audio_s,
                                'rtf': wall / audio_s, 'first_audio_latency_s': first,
                                'latency_kind': 'first-chunk' if mode == 'streaming' else 'non-streaming',
                                'peak_gpu_mb': peak, 'gpu_memory_unit': 'MiB',
                                'output_wav': str(path)})
                            save()
                        case['median'] = {field: statistics.median(r[field] for r in case['repeats'])
                                          for field in ('wall_s', 'audio_s', 'rtf',
                                                        'first_audio_latency_s', 'peak_gpu_mb')}
                        representative = min(case['repeats'],
                                             key=lambda r: abs(r['wall_s'] - case['median']['wall_s']))
                        case['representative_wav'] = representative['output_wav']
                        case['status'] = 'complete'
                        if key == 'short_streaming':
                            canonical = OUT / f'xtts_{lang}.wav'
                            shutil.copyfile(representative['output_wav'], canonical)
                            result['output_wav'] = str(canonical)
                    except Exception as exc:
                        error(f'{lang}/{key}', exc, case)
                    save()
            result['status'] = ('complete' if all(c['status'] == 'complete'
                                                for c in result['cases'].values()) else 'failed')
        except Exception as exc:
            error(f'{lang}/prepare', exc, result)
        save()
    R['status'] = 'complete' if not R['errors'] else 'failed'
    (OUT / f'xtts_attempt_{attempt}.json').write_text(json.dumps(R, indent=1, ensure_ascii=False))
    save()


def normalized_words(text):
    return ''.join(c for c in unicodedata.normalize('NFC', text).lower()
                   if not unicodedata.category(c).startswith('P')).split()


def wer(reference_text, hypothesis):
    ref, hyp = normalized_words(reference_text), normalized_words(hypothesis)
    row = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        nxt = [i]
        for j, b in enumerate(hyp, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = nxt
    return {'wer': row[-1] / len(ref), 'word_edits': row[-1], 'reference_words': len(ref),
            'normalized_reference': ' '.join(ref), 'normalized_transcript': ' '.join(hyp)}


def asr_worker():
    global R
    R = json.loads(REPORT.read_text())
    try:
        from faster_whisper import WhisperModel
        R['asr'] = {'status': 'running', 'model': 'large-v3-turbo', 'packages': inventory(),
                    'executable': sys.executable, 'compute_type': 'float16', 'device': 'cuda'}
        started = time.perf_counter()
        model = WhisperModel('large-v3-turbo', device='cuda', compute_type='float16',
                             download_root='/tmp/xtts-asr-models')
        R['asr']['model_load_including_download_s'] = time.perf_counter() - started
        for lang in SUPPORTED:
            try:
                result = R['tts'][lang]
                for case in result.get('cases', {}).values():
                    for sample in case['repeats']:
                        try:
                            started = time.perf_counter()
                            segments, _ = model.transcribe(sample['output_wav'], language=lang,
                                                           beam_size=5, condition_on_previous_text=False)
                            transcript = ' '.join(s.text for s in segments).strip()
                            sample['asr'] = {'status': 'complete', 'transcript': transcript,
                                             'wall_s': time.perf_counter() - started,
                                             **wer(case['text'], transcript)}
                            save()
                        except Exception as exc:
                            sample['asr'] = {'status': 'failed'}
                            error(f'asr/{lang}/{sample["output_wav"]}', exc, sample['asr'])
                    scores = [s['asr']['wer'] for s in case['repeats']
                              if s.get('asr', {}).get('status') == 'complete']
                    if len(scores) == 3:
                        case['median']['wer'] = statistics.median(scores)
                save()
            except Exception as exc:
                error(f'asr/{lang}', exc)
        R['asr']['status'] = 'failed' if any(e['stage'].startswith('asr/') for e in R['errors']) else 'complete'
    except Exception as exc:
        R.setdefault('asr', {})['status'] = 'failed'
        error('asr/setup', exc)
    save()


def main():
    global R
    OUT.mkdir(parents=True, exist_ok=True)
    R['license'] = 'CPML non-commercial; evaluation only'
    R['texts'] = TEXTS
    R['tts']['sk'] = {'status': 'unsupported', 'reason': 'XTTS-v2 has no Slovak language'}
    env = dict(os.environ, COQUI_TOS_AGREED='1', HF_HOME='/tmp/xtts-hf-home',
               XDG_CACHE_HOME='/tmp/xtts-cache', PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES='0')
    env.pop('PYTHONPATH', None)
    env.pop('VIRTUAL_ENV', None)
    try:
        step('env', python=sys.version, cwd=os.getcwd())
        step('gpu', output=run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                '--format=csv,noheader'], 60))
        uv = shutil.which('uv')
        if not uv:
            step('install_uv', output=run([sys.executable, '-m', 'pip', 'install', 'uv'], 120))
            uv = shutil.which('uv')
        if not uv:
            raise RuntimeError('uv executable absent after installation')
        step('python311', output=run([uv, 'python', 'install', '3.11'], 300))
        for directory in (TTS_ENV, ASR_ENV):
            step('venv', path=str(directory), output=run([uv, 'venv', '--python', '3.11', directory], 120))
        tts_python = TTS_ENV / 'bin/python'
        asr_python = ASR_ENV / 'bin/python'
        step('torch_install', output=run([uv, 'pip', 'install', '--python', tts_python,
             'torch==2.8.0', 'torchaudio==2.8.0', '--index-url',
             'https://download.pytorch.org/whl/cu128'], 900))
        step('coqui_install', output=run([uv, 'pip', 'install', '--python', tts_python,
             'coqui-tts==0.28.0', 'huggingface-hub', 'soundfile'], 900))
        script = Path(__file__).resolve()
        attempts = []
        for attempt in (1, 2):
            log = OUT / f'xtts_attempt_{attempt}.log'
            try:
                output = run([tts_python, script, '--tts-worker', str(attempt)], 1800, env)
                log.write_text(output)
            except Exception as exc:
                log.write_text(str(exc))
                error(f'tts_worker_{attempt}', exc)
            receipt = OUT / f'xtts_attempt_{attempt}.json'
            if receipt.exists():
                child = json.loads(receipt.read_text())
                attempts.append(child)
                R['tts'] = child['tts']
                R['xtts'] = child
            failed = not receipt.exists() or child['status'] != 'complete'
            retryable = receipt.exists() and any(
                e['stage'] == 'model_setup' or any(
                    e['stage'].endswith('/' + length + '_' + mode)
                    for length in ('short', 'long') for mode in ('streaming', 'plain'))
                for e in child['errors'])
            if not failed or attempt == 2 or not retryable:
                break
            # The 5.x incompatibility is not assumed proven: retain initial failure,
            # installed versions and the result after the requested bounded retry.
            R['compatibility_retry'] = {'reason': 'initial XTTS load/synthesis failed',
                                        'constraint': 'transformers>=4.57,<5',
                                        'initial_receipt': str(receipt), 'initial_log': str(log)}
            step('transformers_retry', output=run([uv, 'pip', 'install', '--python', tts_python,
                                                   'transformers>=4.57,<5'], 600))
        R['attempt_receipts'] = [str(OUT / f'xtts_attempt_{i + 1}.json') for i in range(len(attempts))]
        if not attempts or attempts[-1]['status'] != 'complete':
            R['errors'].append({'stage': 'xtts', 'error': 'XTTS evaluation incomplete; see attempt receipts/logs'})
        save()
        try:
            step('asr_install', output=run([uv, 'pip', 'install', '--python', asr_python,
                 'faster-whisper', 'nvidia-cublas-cu12', 'nvidia-cudnn-cu12'], 900))
            libs = list(ASR_ENV.glob('lib/python*/site-packages/nvidia/*/lib'))
            asr_env = dict(env, LD_LIBRARY_PATH=':'.join(map(str, libs)) + ':' + env.get('LD_LIBRARY_PATH', ''))
            output = run([asr_python, script, '--asr-worker'], 1800, asr_env)
            R = json.loads(REPORT.read_text())
            step('asr_worker', output=output)
        except Exception as exc:
            error('asr/environment', exc)
    except Exception as exc:
        error('environment', exc)
    finally:
        for lang in SUPPORTED:
            R['tts'].setdefault(lang, {'status': 'failed', 'reason': 'Environment setup failed'})
        R['status'] = 'complete' if (not R['errors'] and R.get('asr', {}).get('status') == 'complete') else 'failed'
        save()


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    if '--tts-worker' in sys.argv:
        tts_worker()
    elif '--asr-worker' in sys.argv:
        asr_worker()
    else:
        main()
