"""Streaming foundation simulation on a Kaggle T4 (offline, deterministic, no server, no network audio).
Question: which engine + decoding strategy gives words the earliest, without losing accuracy?
Streams: the owner's English script reading (115 s) and a German FLEURS concatenation (~2 min, reference = concatenated transcripts).
Engines: faster-whisper large-v3-turbo fp16 (GPU), Parakeet TDT v3 ONNX (CUDA).
Strategies: fixed chunks C in {1,2} s (independent, like the current /ws route) and small-step decoding with local agreement (LA-2):
decode the growing buffer every S in {0.3,0.5,1.0} s, commit the words two consecutive hypotheses agree on, trim the buffer at the last committed word.
Clock: audio arrives in real time; a decode starts when its audio is available AND the engine is free; its real measured cost is added.
Metrics per cell: median/p90 commit latency (commit time - word end in the stream), WER vs reference, backlog (max wait for a free engine), compute RTF.
Writes /kaggle/working/stream_sim.json after every cell."""
import glob, io, json, os, re, subprocess, sys, time, ctypes
from pathlib import Path
from types import SimpleNamespace

OUT = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path('.')
R = {'cells': {}, 'log': [], 'env': {}}
T0 = time.time()
def save(): (OUT / 'stream_sim.json').write_text(json.dumps(R, ensure_ascii=False, indent=1))
def log(m, **k): R['log'].append({'t': round(time.time() - T0, 1), 'm': m, **k}); print(m, k, flush=True); save()
def sh(c, t=2400):
    p = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t); return p.returncode, (p.stdout + p.stderr)[-700:]

rc, o = sh(f'{sys.executable} -m pip install -q faster-whisper soundfile pyarrow scipy nvidia-cublas-cu12 nvidia-cudnn-cu12 && {sys.executable} -m pip uninstall -y -q onnxruntime onnxruntime-gpu; {sys.executable} -m pip install -q onnxruntime-gpu "onnx-asr[hub]"'); log('pip', rc=rc)
import numpy as np, soundfile as sf, pyarrow.parquet as pq, scipy.signal as ss
import nvidia.cublas.lib as _cb, nvidia.cudnn.lib as _cd
for d in (list(_cb.__path__)[0], list(_cd.__path__)[0]):
    for so in sorted(glob.glob(d + '/*.so*')):
        try: ctypes.CDLL(so)
        except OSError: pass
from huggingface_hub import hf_hub_download, snapshot_download
from faster_whisper import WhisperModel
import onnxruntime as ort, onnx_asr
R['env']['ort'] = ort.get_available_providers(); R['env']['gpu'] = sh('nvidia-smi --query-gpu=name --format=csv,noheader')[1].strip()

norm = lambda s: re.findall(r"[^\W_]+", s.lower())
def wer(ref, hyp):
    r, h = norm(ref), norm(hyp); d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        p = d[:]; d[0] = i
        for j, b in enumerate(h, 1): d[j] = min(p[j] + 1, d[j-1] + 1, p[j-1] + (a != b))
    return d[-1] / max(1, len(r))

# ---- streams -------------------------------------------------------------
def load_owner():
    hit = next(iter(Path('/kaggle/input').rglob('en_script_reading.m4a')), None)
    if not hit: return None
    sh(f'ffmpeg -y -v error -i {hit} -ac 1 -ar 16000 /tmp/en_owner.wav'); ref = (hit.parent / 'en_ref.txt').read_text()
    return sf.read('/tmp/en_owner.wav', dtype='float32')[0], ref
def load_fleurs_concat(code, n=14):
    f = hf_hub_download('google/fleurs', f'{code}/test/0000.parquet', repo_type='dataset', revision='refs/convert/parquet')
    parts, refs = [], []
    for r in pq.read_table(f).to_pylist()[:n]:
        a, sr = sf.read(io.BytesIO(r['audio']['bytes']), dtype='float32'); a = a.mean(1) if a.ndim > 1 else a
        if sr != 16000: a = ss.resample_poly(a, 16000, sr).astype('float32')
        parts += [a, np.zeros(int(0.4 * 16000), dtype='float32')]; refs.append(r['transcription'])
    return np.concatenate(parts), ' '.join(refs)
streams = {}
o = load_owner()
if o: streams['en-owner'] = ('en', *o)
for lang, code in (('de', 'de_de'),):
    try: streams[f'{lang}-fleurs'] = (lang, *load_fleurs_concat(code))
    except Exception as e: log('stream_fail', lang=lang, err=str(e)[:160])
log('streams', **{k.replace('-', '_'): round(len(v[1]) / 16000, 1) for k, v in streams.items()})

# ---- engines: decode(audio_slice, lang) -> list of (word, start_s, end_s) relative to the slice -----
turbo = WhisperModel('large-v3-turbo', device='cuda', compute_type='float16')
def dec_turbo(a, lang):
    segs, _ = turbo.transcribe(a, language=lang, beam_size=1, vad_filter=False, condition_on_previous_text=False, word_timestamps=True)
    return [(w.word.strip(), w.start, w.end) for s in segs for w in (s.words or []) if w.word.strip()]
pdir = snapshot_download('istupakov/parakeet-tdt-0.6b-v3-onnx', local_dir='/kaggle/working/parakeet-v3-onnx')
pk = onnx_asr.load_model('nemo-parakeet-tdt-0.6b-v3', pdir, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']).with_timestamps()
def dec_parakeet(a, lang):
    r = pk.recognize(a); words, cur = [], None
    for tok, ts in zip(r.tokens, r.timestamps):
        if cur is None or tok.startswith(' '):
            if cur: words.append(cur)
            cur = [tok.strip(), float(ts), float(ts) + 0.08]
        else: cur[0] += tok; cur[2] = float(ts) + 0.08
    if cur: words.append(cur)
    return [(w, s, e) for w, s, e in words if w]
ENGINES = {'turbo': dec_turbo, 'parakeet': dec_parakeet}
for n, f in ENGINES.items():
    for _ in range(2): f(np.zeros(16000 * 2, dtype='float32') + 1e-4, 'en')       # warm-up (cuDNN autotune, graph build)

# ---- simulations ---------------------------------------------------------------
def timed(f, a, lang):
    t = time.perf_counter(); w = f(a, lang); return w, time.perf_counter() - t
def finish(lat, hyp_words, ref, busy, wait_max, dur):
    lat = np.array(lat) if lat else np.array([0.0])
    return {'latency_median_s': round(float(np.median(lat)), 2), 'latency_p90_s': round(float(np.percentile(lat, 90)), 2),
            'WER': round(wer(ref, ' '.join(hyp_words)), 3), 'max_wait_for_engine_s': round(wait_max, 2), 'compute_RTF': round(busy / dur, 3), 'words': len(hyp_words)}
def sim_chunks(f, a, lang, ref, C):
    dur = len(a) / 16000; busy_until = busy = wait_max = 0.0; lat, hyp = [], []
    for k in range(int(np.ceil(dur / C))):
        s, e = k * C, min((k + 1) * C, dur); arrive = e                        # chunk is complete when its last sample exists
        w, cost = timed(f, a[int(s * 16000):int(e * 16000)], lang)
        start = max(arrive, busy_until); wait_max = max(wait_max, start - arrive); busy_until = start + cost; busy += cost
        for word, ws, we in w: lat.append(busy_until - (s + we)); hyp.append(word)
    return finish(lat, hyp, ref, busy, wait_max, dur)
def sim_agree(f, a, lang, ref, S, max_buf=8.0):
    dur = len(a) / 16000; busy_until = busy = wait_max = 0.0; lat, committed = [], []; buf_start = 0.0; prev = []
    steps = int(np.ceil(dur / S))
    for k in range(1, steps + 1):
        t = min(k * S, dur)
        if t - buf_start < 0.5: continue
        start = max(t, busy_until); wait_max = max(wait_max, start - t)
        w, cost = timed(f, a[int(buf_start * 16000):int(t * 16000)], lang)
        busy_until = start + cost; busy += cost
        hyp = [(x, buf_start + s, buf_start + e) for x, s, e in w]
        agree = 0
        while agree < min(len(prev), len(hyp)) and norm(prev[agree][0]) == norm(hyp[agree][0]): agree += 1
        if t - buf_start > max_buf: agree = max(agree, len(hyp) - 2)           # long buffer: force-commit all but the last two words
        for word, ws, we in hyp[:agree]:
            lat.append(busy_until - we); committed.append(word)
        if agree: buf_start = hyp[agree - 1][2]; prev = []                      # trim at the last committed word
        else: prev = hyp
    w, cost = timed(f, a[int(buf_start * 16000):], lang); busy += cost         # end of stream: flush the rest
    for word, ws, we in [(x, buf_start + s, buf_start + e) for x, s, e in w]: lat.append(max(busy_until, dur) + cost - we); committed.append(word)
    return finish(lat, committed, ref, busy, wait_max, dur)

for sname, (lang, a, ref) in streams.items():
    for ename, f in ENGINES.items():
        if ename == 'parakeet' and lang not in ('en', 'de'): continue
        for kind, param in (('chunk', 1.0), ('chunk', 2.0), ('agree', 0.3), ('agree', 0.5), ('agree', 1.0)):
            key = f'{sname}|{ename}|{kind}-{param}'
            try:
                R['cells'][key] = (sim_chunks if kind == 'chunk' else sim_agree)(f, a, lang, ref, param); log('cell', cell=key, **R['cells'][key])
            except Exception as e:
                import traceback; log('cell_fail', cell=key, err=f'{type(e).__name__}: {e}'[:200], tb=traceback.format_exc()[-300:])
R['status'] = 'done'; log('done', wall_s=round(time.time() - T0))
