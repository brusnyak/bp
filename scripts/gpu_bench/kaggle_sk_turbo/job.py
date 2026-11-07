"""Kaggle T4: Slovak speech recognition, incumbent vs plain turbo vs kinit/whisper-large-v3-turbo-sk.
Data: FLEURS sk_sk test (60 clips) + the owner's script reading (private dataset bp-owner-scripts-20261107).
Converts the Hugging Face models to CTranslate2 (float16), scores WER and speed with faster-whisper.
Writes /kaggle/working/sk_compare.json after every cell."""
import glob, io, json, os, re, subprocess, sys, time, ctypes
from pathlib import Path

OUT = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path('.')
R = {'cells': {}, 'log': []}
T0 = time.time()
def save(): (OUT / 'sk_compare.json').write_text(json.dumps(R, ensure_ascii=False, indent=1))
def log(m, **k): R['log'].append({'t': round(time.time() - T0, 1), 'm': m, **k}); print(m, k, flush=True); save()
def sh(c, t=2400):
    p = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t); return p.returncode, (p.stdout + p.stderr)[-900:]

rc, o = sh(f'{sys.executable} -m pip install -q faster-whisper sentencepiece soundfile pyarrow scipy nvidia-cublas-cu12 nvidia-cudnn-cu12'); log('pip', rc=rc, tail=o[-200:])
import numpy as np, soundfile as sf, pyarrow.parquet as pq, scipy.signal as ss
import nvidia.cublas.lib as _cb, nvidia.cudnn.lib as _cd
for d in (list(_cb.__path__)[0], list(_cd.__path__)[0]):
    for so in sorted(glob.glob(d + '/*.so*')):
        try: ctypes.CDLL(so)
        except OSError: pass
from huggingface_hub import hf_hub_download
from faster_whisper import WhisperModel

norm = lambda s: re.findall(r"[^\W_]+", s.lower())
def wer(ref, hyp):
    r, h = norm(ref), norm(hyp); d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        p = d[:]; d[0] = i
        for j, b in enumerate(h, 1): d[j] = min(p[j] + 1, d[j-1] + 1, p[j-1] + (a != b))
    return d[-1] / max(1, len(r))

f = hf_hub_download('google/fleurs', 'sk_sk/test/0000.parquet', repo_type='dataset', revision='refs/convert/parquet')
fleurs = []
for r in pq.read_table(f).to_pylist()[:60]:
    a, sr = sf.read(io.BytesIO(r['audio']['bytes']), dtype='float32'); a = a.mean(1) if a.ndim > 1 else a
    if sr != 16000: a = ss.resample_poly(a, 16000, sr).astype('float32')
    fleurs.append((a, r['transcription']))
hit = next(iter(Path('/kaggle/input').rglob('sk_script_reading.m4a')), None)   # mount path differs between Kaggle layouts
own_dir = str(hit.parent) if hit else None
log('owner_dir', path=own_dir, tree=[str(p) for p in list(Path('/kaggle/input').glob('*'))[:6]])
owner = None
if own_dir:
    wav = '/tmp/sk_owner.wav'
    sh(f'ffmpeg -y -v error -i {own_dir}/sk_script_reading.m4a -ac 1 -ar 16000 {wav}')
    owner = (sf.read(wav, dtype='float32')[0], open(f'{own_dir}/sk_ref.txt').read())
log('data', fleurs=len(fleurs), owner=bool(owner))

def convert(hf, out, tokenizer_from=None):
    if Path(out).exists(): return out
    rc, o = sh(f'ct2-transformers-converter --model {hf} --output_dir {out} --copy_files preprocessor_config.json --quantization float16 --force')
    if rc != 0:   # some repos ship no preprocessor_config.json: convert without copying it
        rc, o = sh(f'ct2-transformers-converter --model {hf} --output_dir {out} --quantization float16 --force')
    if rc == 0:
        import shutil
        try: shutil.copy(hf_hub_download(tokenizer_from or hf, 'tokenizer.json'), f'{out}/tokenizer.json')
        except Exception as e: log('tokenizer_copy_fail', model=hf, err=str(e)[:200])
    log('convert', model=hf, rc=rc, tail=o[-300:]); return out if rc == 0 else None

MODELS = [('large-v3-turbo (plain)', lambda: 'large-v3-turbo'),
          ('kinit/whisper-large-v3-turbo-sk', lambda: convert('kinit/whisper-large-v3-turbo-sk', '/kaggle/working/ct2-kinit-turbo-sk')),
          ('NaiveNeuron/whisper-small-sk (incumbent)', lambda: convert('NaiveNeuron/whisper-small-sk', '/kaggle/working/ct2-small-sk', tokenizer_from='openai/whisper-small'))]
for name, get in MODELS:
    try:
        src = get()
        if not src: continue
        m = WhisperModel(src, device='cuda', compute_type='float16')
        ws, dt, dur = [], 0.0, 0.0
        for a, ref in fleurs:
            t = time.perf_counter(); segs, _ = m.transcribe(a, language='sk', beam_size=5, vad_filter=False, condition_on_previous_text=False)
            hyp = ' '.join(s.text for s in segs); dt += time.perf_counter() - t; dur += len(a) / 16000; ws.append(wer(ref, hyp))
        R['cells'][f'{name}|fleurs-sk'] = {'WER': round(float(np.mean(ws)), 3), 'median': round(float(np.median(ws)), 3), 'RTF': round(dt / dur, 3), 'n': len(ws)}
        log('fleurs', model=name, **R['cells'][f'{name}|fleurs-sk'])
        if owner:
            t = time.perf_counter(); segs, _ = m.transcribe(owner[0], language='sk', beam_size=5, vad_filter=True, condition_on_previous_text=False)
            hyp = ' '.join(s.text for s in segs); d = time.perf_counter() - t
            R['cells'][f'{name}|owner-sk'] = {'WER': round(wer(owner[1], hyp), 3), 'RTF': round(d / (len(owner[0]) / 16000), 3), 'hyp_head': hyp[:240]}
            log('owner', model=name, **R['cells'][f'{name}|owner-sk'])
        del m
    except Exception as e:
        import traceback; log('model_fail', model=name, err=f'{type(e).__name__}: {e}'[:300], tb=traceback.format_exc()[-400:])
R['status'] = 'done'; log('done', wall_s=round(time.time() - T0))
