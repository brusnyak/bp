"""Czech voice data for a Piper fine-tune (2026-11-10): OmniVoice clones the owner's Czech voice over FLEURS Czech sentences, a Czech Whisper pass
rejects clips that are not intelligible, survivors become a Piper training set (22050 Hz mono wav + `file|text` csv, peak -3 dB, edges trimmed).
Why: the owner's own recordings carry background noise; the clone is clean and can be multiplied. Output goes to /kaggle/working/data (read by the
fine-tune kernel through kernel_sources, never downloaded). Writes /kaggle/working/data_report.json after every step; models stay in /tmp.
Gate: Czech word error per clip <= 10%, duration 1.5-14 s."""
import glob, json, os, re, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np

OUT = Path('/kaggle/working'); DATA = OUT / 'data'; WAVS = DATA / 'wavs'; RAW = Path('/tmp/raw'); RAW.mkdir(exist_ok=True); WAVS.mkdir(parents=True, exist_ok=True)
R = {'log': [], 'env': {}}; T0 = time.time()
def save(): (OUT / 'data_report.json').write_text(json.dumps(R, ensure_ascii=False, indent=1))
def log(m, **k): R['log'].append({'t': round(time.time() - T0, 1), 'm': m, **k}); print(m, k, flush=True); save()
def sh(c, t=1800):
    p = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t); return p.returncode, (p.stdout + p.stderr)[-800:]
N_SENT = 800; STEPS = 16
R['env']['gpu'] = sh('nvidia-smi --query-gpu=name --format=csv,noheader')[1].strip()

# ---- reference: the cleanest 10 s window of the owner's Czech reading (highest p90/p10 frame-energy ratio) ----
src = next(Path('/kaggle/input').rglob('me_cs_staromestske.m4a'))
rc, o = sh(f'ffmpeg -y -v error -i {src} -ac 1 -ar 24000 /tmp/cs_full.wav'); log('decode', rc=rc)
import soundfile as sf
x, sr = sf.read('/tmp/cs_full.wav', dtype='float32'); fr = int(0.025 * sr); n = len(x) // fr
rms = np.sqrt((x[: n * fr].reshape(n, fr) ** 2).mean(1) + 1e-12); W = int(10 / 0.025); best = (0, -1)
for s in range(0, n - W, 20):
    w = rms[s: s + W]; sc = np.percentile(w, 90) / (np.percentile(w, 10) + 1e-6)
    if np.abs(x[s * fr:(s + W) * fr]).max() < 0.98 and sc > best[0]: best = (sc, s)
s0 = best[1] if best[1] >= 0 else 0; ref = x[s0 * fr:(s0 + W) * fr]; sf.write('/tmp/ref_cs.wav', ref, sr)
log('reference', total_s=round(len(x) / sr, 1), start_s=round(s0 * 0.025, 1), snr_ratio=round(float(best[0]), 1))

# ---- sentences: FLEURS Czech ----
import pyarrow.parquet as pq
rc, o = sh(f'{sys.executable} -m pip install -q omnivoice pyarrow'); log('pip omnivoice', rc=rc)
from huggingface_hub import hf_hub_download
sents = []
for split in ('train', 'test', 'validation'):
    try:
        f = hf_hub_download('google/fleurs', f'cs_cz/{split}/0000.parquet', repo_type='dataset', revision='refs/convert/parquet')
        for r in pq.read_table(f, columns=['raw_transcription']).to_pylist():
            t = re.sub(r'\s+', ' ', r['raw_transcription']).strip()
            if 40 <= len(t) <= 140 and t not in sents: sents.append(t)
    except Exception as e: log('fleurs_fail', split=split, err=str(e)[:150])
    if len(sents) >= N_SENT: break
sents = sents[:N_SENT]; log('sentences', n=len(sents), sample=sents[:2])

# ---- generate with OmniVoice ----
import torch
from omnivoice import OmniVoice
t = time.perf_counter(); model = OmniVoice.from_pretrained('k2-fsa/OmniVoice', device_map='cuda', dtype=torch.float16); log('omni load', s=round(time.perf_counter() - t, 1))
prompt = model.create_voice_clone_prompt('/tmp/ref_cs.wav', None); model.generate(text='Test.', language='cs', voice_clone_prompt=prompt, num_step=STEPS)
gen = []; tg = time.perf_counter(); audio_s = 0.0
for i, txt in enumerate(sents):
    try:
        a = model.generate(text=txt, language='cs', voice_clone_prompt=prompt, num_step=STEPS)[0]
        a = np.asarray(a.detach().float().cpu().numpy() if hasattr(a, 'detach') else a, dtype='float32').squeeze(); sro = getattr(model, 'sampling_rate', 24000)
        sf.write(RAW / f'{i:04d}.wav', a, sro); gen.append((i, txt, len(a) / sro)); audio_s += len(a) / sro
    except Exception as e: log('gen_fail', i=i, err=str(e)[:120])
    if i % 100 == 99: log('gen progress', done=i + 1, wall_s=round(time.perf_counter() - tg), audio_s=round(audio_s), rtf=round((time.perf_counter() - tg) / max(audio_s, 1), 3))
R['gen'] = {'clips': len(gen), 'audio_min': round(audio_s / 60, 1), 'wall_min': round((time.perf_counter() - tg) / 60, 1), 'steps': STEPS}; log('generated', **R['gen'])
del model, prompt; torch.cuda.empty_cache()

# ---- QC with a Czech Whisper pass ----
rc, o = sh(f'{sys.executable} -m pip install -q faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12'); log('pip whisper', rc=rc)
import ctypes, nvidia.cublas.lib as _cb, nvidia.cudnn.lib as _cd
for d in (list(_cb.__path__)[0], list(_cd.__path__)[0]):
    for so in sorted(glob.glob(d + '/*.so*')):
        try: ctypes.CDLL(so)
        except OSError: pass
from faster_whisper import WhisperModel
import scipy.signal as ss
asr = WhisperModel('large-v3-turbo', device='cuda', compute_type='float16')
norm = lambda s: re.findall(r"[^\W_]+", s.lower())
def wer(ref, hyp):
    r, h = norm(ref), norm(hyp); d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        p = d[:]; d[0] = i
        for j, b in enumerate(h, 1): d[j] = min(p[j] + 1, d[j-1] + 1, p[j-1] + (a != b))
    return d[-1] / max(1, len(r))
keep, rej, werl, lines = 0, {'wer': 0, 'duration': 0}, [], []
for i, txt, dur in gen:
    if not 1.5 <= dur <= 14: rej['duration'] += 1; continue
    a, sro = sf.read(RAW / f'{i:04d}.wav', dtype='float32'); a = a.mean(1) if a.ndim > 1 else a
    try:   # arrays, not paths: faster-whisper's file decoder clashed with the installed `av` package (first run, 2026-11-10)
        segs, _ = asr.transcribe(ss.resample_poly(a, 16000, int(sro)).astype('float32'), language='cs', beam_size=1, vad_filter=False); hyp = ' '.join(s.text for s in segs)
    except Exception as e: log('asr_fail', i=i, err=f'{type(e).__name__}: {e}'[:200]); rej['asr_error'] = rej.get('asr_error', 0) + 1; continue
    w = wer(txt, hyp); werl.append(w)
    if w > 0.10: rej['wer'] += 1; continue
    nz = np.where(np.abs(a) > 0.01)[0]
    if len(nz): a = a[max(0, nz[0] - int(0.1 * sro)): nz[-1] + int(0.15 * sro)]
    a = ss.resample_poly(a, 22050, int(sro)).astype('float32'); a = a / (np.abs(a).max() + 1e-6) * 0.708
    sf.write(WAVS / f'cz_{keep:04d}.wav', a, 22050, subtype='PCM_16'); lines.append(f'cz_{keep:04d}.wav|{txt}'); keep += 1
    if keep % 100 == 0: log('qc progress', checked=len(werl), kept=keep)
(DATA / 'metadata.csv').write_text('\n'.join(lines) + '\n', encoding='utf-8')
wsort = sorted(werl)
R['qc'] = {'checked': len(werl), 'kept': keep, 'rejected': rej, 'wer_median': round(wsort[len(wsort) // 2], 3) if wsort else None, 'wer_p90': round(wsort[int(len(wsort) * .9)], 3) if wsort else None,
           'kept_minutes': round(sum(len(sf.read(f, dtype='float32')[0]) / 22050 for f in sorted(WAVS.glob('*.wav'))[:keep]) / 60, 1)}
R['status'] = 'done'; log('done', wall_min=round((time.time() - T0) / 60, 1), **R['qc'])
