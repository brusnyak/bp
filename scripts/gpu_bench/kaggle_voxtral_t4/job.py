"""Voxtral Mini 4B Realtime (Mistral, Apache 2.0) on a Kaggle T4: is a streaming recogniser fast enough, and how accurate is it?
Question (2026-11-09): on the Mac the Whisper agreement route floors at ~3 s lag (0.7 s per decode); Voxtral never revises tokens and
emits ~0.5 s behind the speech. Does it hold real time on a T4 and at what word error?
Per stream (generated English and German speech with exact reference text, /kaggle/input/bp-testaudio-20261108):
  offline    one generate() over the whole clip: compute RTF and word error (accuracy ceiling)
  streaming  audio released in real time (a chunk is available only when its end time has passed on the wall clock), Transformers'
             experimental streaming API (TextIteratorStreamer); lag = wall time a text piece arrives minus the audio position released
             at that moment (so it includes the model delay, not the chunk wait); word error of the streamed text; wall vs audio length.
Writes /kaggle/working/voxtral_t4.json after every step; a failed step records its traceback and the run continues."""
import glob, json, re, subprocess, sys, time, traceback
from pathlib import Path
from threading import Thread

OUT = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path('.')
R = {'cells': {}, 'log': [], 'env': {}}
T0 = time.time()
def save(): (OUT / 'voxtral_t4.json').write_text(json.dumps(R, ensure_ascii=False, indent=1))
def log(m, **k): R['log'].append({'t': round(time.time() - T0, 1), 'm': m, **k}); print(m, k, flush=True); save()
def sh(c, t=2400):
    p = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t); return p.returncode, (p.stdout + p.stderr)[-700:]

rc, o = sh(f'{sys.executable} -m pip install -q -U "transformers>=5.22" accelerate soundfile scipy "mistral-common[audio]"'); log('pip', rc=rc, tail=o[-300:])
import numpy as np, soundfile as sf, scipy.signal as ss, torch
import transformers
R['env'] = {'gpu': sh('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader')[1].strip(), 'torch': torch.__version__, 'transformers': transformers.__version__}
log('env', **R['env'])

norm = lambda s: re.findall(r"[^\W_]+", s.lower())
def wer(ref, hyp):
    r, h = norm(ref), norm(hyp); d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        p, d[0] = d[0], i
        for j, b in enumerate(h, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a != b))
    return d[-1] / max(len(r), 1)

REPO = 'mistralai/Voxtral-Mini-4B-Realtime-2602'
from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration, VoxtralRealtimeProcessor, TextIteratorStreamer
t = time.time()
processor = VoxtralRealtimeProcessor.from_pretrained(REPO)
model = VoxtralRealtimeForConditionalGeneration.from_pretrained(REPO, device_map='cuda', dtype=torch.float16)
log('model loaded', s=round(time.time() - t, 1), vram_gb=round(torch.cuda.memory_allocated() / 2**30, 2))

FILES = {p.stem: p for p in Path('/kaggle/input').rglob('*.wav')}   # the dataset mount path varies; find by name
log('audio found', n=len(FILES), names=sorted(FILES)[:12])
def load(name):
    a, sr = sf.read(str(FILES[name]), dtype='float32'); a = a if a.ndim == 1 else a.mean(1)
    if sr != 16000: a = ss.resample_poly(a, 16000, sr).astype('float32')
    return a, json.load(open(FILES[name].with_suffix('.json')))['text']

def offline(a):
    inputs = processor(a, return_tensors='pt').to(model.device); inputs = inputs.to(model.device, dtype=model.dtype)
    torch.cuda.synchronize(); t = time.time()
    out = model.generate(**inputs); torch.cuda.synchronize(); dt = time.time() - t
    return processor.batch_decode(out, skip_special_tokens=True)[0], dt

def streaming(a):
    a = np.pad(a, (0, processor.num_right_pad_tokens * processor.raw_audio_length_per_tok))
    first = processor(a[:processor.num_samples_first_audio_chunk], is_streaming=True, is_first_audio_chunk=True, return_tensors='pt')
    first.to(model.device, dtype=model.dtype)
    st = {'fed': 0.0, 't0': None}
    def gate(end_sample):   # a chunk exists only once its end time has passed
        wait = end_sample / 16000 - (time.time() - st['t0'])
        if wait > 0: time.sleep(wait)
        st['fed'] = end_sample / 16000
    def gen():
        gate(processor.num_samples_first_audio_chunk); yield first.input_features
        idx = processor.num_mel_frames_first_audio_chunk
        hop, win = processor.feature_extractor.hop_length, processor.feature_extractor.win_length
        start = idx * hop - win // 2
        while (end := start + processor.num_samples_per_audio_chunk) < a.shape[0]:
            x = processor(a[start:end], is_streaming=True, is_first_audio_chunk=False, return_tensors='pt'); x.to(model.device, dtype=model.dtype)
            gate(end); yield x.input_features
            idx += processor.audio_length_per_tok; start = idx * hop - win // 2
    streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    kw = {'input_ids': first.input_ids, 'input_features': gen(), 'num_delay_tokens': first.num_delay_tokens, 'streamer': streamer}
    st['t0'] = time.time(); th = Thread(target=model.generate, kwargs=kw); th.start()
    pieces = []
    for piece in streamer:
        now = time.time() - st['t0']; pieces.append((round(st['fed'], 2), round(now - st['fed'], 2), piece))
    th.join()
    return ''.join(p[2] for p in pieces), pieces, time.time() - st['t0']

_ = offline(load('en_paced')[0][:16000 * 3])   # warm-up
for name in ['en_paced', 'en_continuous', 'en_long', 'de_paced', 'de_continuous', 'de_long']:
    try:
        a, ref = load(name); dur = len(a) / 16000; cell = {'audio_s': round(dur, 1)}
        text, dt = offline(a); cell['offline'] = {'rtf': round(dt / dur, 3), 'wer': round(wer(ref, text), 4), 'text': text[:300]}
        text, pieces, wall = streaming(a)
        lags = [p[1] for p in pieces if p[2].strip()]
        cell['streaming'] = {'wer': round(wer(ref, text), 4), 'wall_s': round(wall, 1), 'wall_minus_audio_s': round(wall - dur, 1),
                             'lag_median_s': round(float(np.median(lags)), 2) if lags else None, 'lag_p90_s': round(float(np.percentile(lags, 90)), 2) if lags else None,
                             'lag_first_s': lags[0] if lags else None, 'lag_last_s': lags[-1] if lags else None, 'pieces': len(pieces), 'text': text[:300],
                             'series': [[p[0], p[1]] for p in pieces if p[2].strip()]}   # (audio released so far, wall minus that) per text piece: where do the stalls sit?
        R['cells'][name] = cell; log('cell', name=name, **{k: v for k, v in cell.items() if k != 'audio_s'})
    except Exception:
        R['cells'][name] = {'error': traceback.format_exc()[-1500:]}; log('cell failed', name=name)
log('done')
