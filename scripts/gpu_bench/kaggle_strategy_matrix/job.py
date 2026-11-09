"""Segmentation-strategy matrix on a Kaggle T4 (offline, deterministic, no server).
Question: which way of cutting speech for recognition keeps accuracy AND gives words early, per language and engine?
Streams: generated reference speech (generic Piper voices, scripts/live/make_test_audio.py: paced / continuous / long, exact reference text), native read speech
(FLEURS sk/de/en, 14 clips joined with 0.4 s gaps), and the owner's own recordings if the dataset carries any file named owner_<lang>*.wav with owner_<lang>*.txt.
Engines: faster-whisper large-v3-turbo fp16 (plain), kinit/whisper-large-v3-turbo-sk (Slovak fine-tune, converted at run time), Parakeet TDT v3 ONNX (CUDA).
Strategies (same clock model as the stream simulation: audio arrives in real time, a decode starts when its audio exists AND the engine is free, real cost is added):
  whole        one decode after the stream ends (accuracy ceiling, no live use)
  pause        phrase closed after a 0.3 s pause or 15 s (the 2025-12 demo; UI "Phrase")
  clause       closed after a 0.2 s pause or 4 s cap (UI "Clause")
  chunk-2.0    independent 2 s windows (the old live route)
  agree-S      growing buffer decoded every S s, words two decodes agree on are committed, buffer trimmed there (UI "Agreement"), S in {0.5, 1.0}
Metrics per cell: word error vs reference, latency median/p90 (commit time minus word end in the stream), backlog, compute RTF, and the hypothesis text.
Writes /kaggle/working/strategy_matrix.json after every cell."""
import glob, io, json, os, re, subprocess, sys, time, ctypes
from pathlib import Path

OUT = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path('.')
R = {'cells': {}, 'log': [], 'env': {}}
T0 = time.time()
def save(): (OUT / 'strategy_matrix.json').write_text(json.dumps(R, ensure_ascii=False, indent=1))
def log(m, **k): R['log'].append({'t': round(time.time() - T0, 1), 'm': m, **k}); print(m, k, flush=True); save()
def sh(c, t=2400):
    p = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t); return p.returncode, (p.stdout + p.stderr)[-700:]

rc, o = sh(f'{sys.executable} -m pip install -q faster-whisper soundfile pyarrow scipy webrtcvad-wheels nvidia-cublas-cu12 nvidia-cudnn-cu12 && {sys.executable} -m pip uninstall -y -q onnxruntime onnxruntime-gpu; {sys.executable} -m pip install -q onnxruntime-gpu "onnx-asr[hub]"'); log('pip', rc=rc)
import numpy as np, soundfile as sf, pyarrow.parquet as pq, scipy.signal as ss
import nvidia.cublas.lib as _cb, nvidia.cudnn.lib as _cd
for d in (list(_cb.__path__)[0], list(_cd.__path__)[0]):
    for so in sorted(glob.glob(d + '/*.so*')):
        try: ctypes.CDLL(so)
        except OSError: pass
from huggingface_hub import hf_hub_download, snapshot_download
from faster_whisper import WhisperModel
import onnxruntime as ort, onnx_asr, webrtcvad
R['env']['ort'] = ort.get_available_providers(); R['env']['gpu'] = sh('nvidia-smi --query-gpu=name --format=csv,noheader')[1].strip()

norm = lambda s: re.findall(r"[^\W_]+", s.lower())
def wer(ref, hyp):
    r, h = norm(ref), norm(hyp); d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        p = d[:]; d[0] = i
        for j, b in enumerate(h, 1): d[j] = min(p[j] + 1, d[j-1] + 1, p[j-1] + (a != b))
    return d[-1] / max(1, len(r))

# ---- streams: name -> (lang, audio16k, reference_text) -----------------------------------
def fleurs(code, n=14):
    f = hf_hub_download('google/fleurs', f'{code}/test/0000.parquet', repo_type='dataset', revision='refs/convert/parquet')
    parts, refs = [], []
    for r in pq.read_table(f).to_pylist()[:n]:
        a, sr = sf.read(io.BytesIO(r['audio']['bytes']), dtype='float32'); a = a.mean(1) if a.ndim > 1 else a
        if sr != 16000: a = ss.resample_poly(a, 16000, sr).astype('float32')
        parts += [a, np.zeros(int(0.4 * 16000), dtype='float32')]; refs.append(r['transcription'])
    return np.concatenate(parts), ' '.join(refs)
streams = {}
for wav in sorted(Path('/kaggle/input').rglob('*.wav')):                         # generated speech and owner recordings
    meta = wav.with_suffix('.json'); txt = wav.with_suffix('.txt')
    try:
        a, sr = sf.read(wav, dtype='float32'); a = a.mean(1) if a.ndim > 1 else a
        if sr != 16000: a = ss.resample_poly(a, 16000, sr).astype('float32')
        if meta.exists(): m = json.loads(meta.read_text()); streams[wav.stem] = (m['lang'], a, m['text'])
        elif txt.exists(): streams[wav.stem] = (wav.stem.split('_')[1] if '_' in wav.stem else 'sk', a, txt.read_text())
    except Exception as e: log('stream_fail', name=wav.name, err=str(e)[:160])
for lang, code in (('sk', 'sk_sk'), ('de', 'de_de'), ('en', 'en_us')):
    try: streams[f'fleurs_{lang}'] = (lang, *fleurs(code))
    except Exception as e: log('stream_fail', lang=lang, err=str(e)[:160])
log('streams', **{k: round(len(v[1]) / 16000, 1) for k, v in streams.items()})

# ---- engines: decode(audio_slice, lang) -> [(word, start_s, end_s)] relative to the slice ----
turbo = WhisperModel('large-v3-turbo', device='cuda', compute_type='float16')
def dec_whisper(model):
    def f(a, lang):
        segs, _ = model.transcribe(a, language=lang, beam_size=1, vad_filter=False, condition_on_previous_text=False, word_timestamps=True, temperature=0.0)
        return [(w.word.strip(), w.start, w.end) for s in segs for w in (s.words or []) if w.word.strip()]
    return f
ENGINES = {'turbo': dec_whisper(turbo)}
try:
    out = '/kaggle/working/ct2-kinit-turbo-sk'
    rc, o = sh(f'ct2-transformers-converter --model kinit/whisper-large-v3-turbo-sk --output_dir {out} --copy_files preprocessor_config.json --quantization float16 --force')
    if rc != 0: rc, o = sh(f'ct2-transformers-converter --model kinit/whisper-large-v3-turbo-sk --output_dir {out} --quantization float16 --force')
    try: import shutil; shutil.copy(hf_hub_download('kinit/whisper-large-v3-turbo-sk', 'tokenizer.json'), f'{out}/tokenizer.json')
    except Exception as e: log('tokenizer_copy_fail', err=str(e)[:200])
    log('convert_kinit', rc=rc, tail=o[-200:])
    if rc == 0: ENGINES['kinit'] = dec_whisper(WhisperModel(out, device='cuda', compute_type='float16'))
except Exception as e: log('kinit_fail', err=str(e)[:200])
try:
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
    ENGINES['parakeet'] = dec_parakeet
except Exception as e: log('parakeet_fail', err=str(e)[:200])
for n, f in ENGINES.items():
    for _ in range(2): f(np.zeros(16000 * 2, dtype='float32') + 1e-4, 'en')       # warm-up (cuDNN autotune, graph build)
log('engines', names=list(ENGINES))

# ---- simulations ---------------------------------------------------------------
def timed(f, a, lang):
    t = time.perf_counter(); w = f(a, lang); return w, time.perf_counter() - t
def finish(lat, hyp_words, ref, busy, wait_max, dur):
    lat = np.array(lat) if lat else np.array([0.0])
    return {'WER': round(wer(ref, ' '.join(hyp_words)), 3), 'latency_median_s': round(float(np.median(lat)), 2), 'latency_p90_s': round(float(np.percentile(lat, 90)), 2),
            'max_wait_for_engine_s': round(wait_max, 2), 'compute_RTF': round(busy / dur, 3), 'words': len(hyp_words), 'text': ' '.join(hyp_words)[:600]}
def sim_whole(f, a, lang, ref):
    dur = len(a) / 16000; w, cost = timed(f, a, lang)
    return finish([dur + cost - we for _, _, we in w], [x for x, _, _ in w], ref, cost, 0.0, dur)
def sim_chunks(f, a, lang, ref, C):
    dur = len(a) / 16000; busy_until = busy = wait_max = 0.0; lat, hyp = [], []
    for k in range(int(np.ceil(dur / C))):
        s, e = k * C, min((k + 1) * C, dur)
        w, cost = timed(f, a[int(s * 16000):int(e * 16000)], lang)
        start = max(e, busy_until); wait_max = max(wait_max, start - e); busy_until = start + cost; busy += cost
        for word, ws, we in w: lat.append(busy_until - (s + we)); hyp.append(word)
    return finish(lat, hyp, ref, busy, wait_max, dur)
def vad_segments(a, silence, max_s, aggr=3):
    v = webrtcvad.Vad(aggr); fr = 320; segs, t0, last = [], None, None; pcm = (np.clip(a, -1, 1) * 32767).astype(np.int16)
    for k in range(len(pcm) // fr):
        t = k * 0.02
        if v.is_speech(pcm[k * fr:(k + 1) * fr].tobytes(), 16000):
            if t0 is None: t0 = t
            last = t
            if last - t0 > max_s: segs.append((t0, last)); t0 = last = None
        elif t0 is not None and last is not None and t - last > silence: segs.append((t0, last + 0.1)); t0 = last = None
    if t0 is not None and last is not None: segs.append((t0, last + 0.1))
    return segs
def sim_segments(f, a, lang, ref, silence, max_s):
    dur = len(a) / 16000; busy_until = busy = wait_max = 0.0; lat, hyp = [], []
    for t0, t1 in vad_segments(a, silence, max_s):
        t0 = max(0.0, t0 - 0.2); seg = a[int(t0 * 16000):int(t1 * 16000)]
        if len(seg) < 8000: continue
        arrive = t1 + silence; w, cost = timed(f, seg, lang)
        start = max(arrive, busy_until); wait_max = max(wait_max, start - arrive); busy_until = start + cost; busy += cost
        for word, ws, we in w: lat.append(busy_until - (t0 + we)); hyp.append(word)
    return finish(lat, hyp, ref, busy, wait_max, dur)
def sim_agree(f, a, lang, ref, S, max_buf=8.0):
    dur = len(a) / 16000; busy_until = busy = wait_max = 0.0; lat, committed = [], []; buf_start = 0.0; prev = []
    for k in range(1, int(np.ceil(dur / S)) + 1):
        t = min(k * S, dur)
        if t - buf_start < 0.5: continue
        start = max(t, busy_until); wait_max = max(wait_max, start - t)
        w, cost = timed(f, a[int(buf_start * 16000):int(t * 16000)], lang); busy_until = start + cost; busy += cost
        hyp = [(x, buf_start + s, buf_start + e) for x, s, e in w]; agree = 0
        while agree < min(len(prev), len(hyp)) and norm(prev[agree][0]) == norm(hyp[agree][0]): agree += 1
        if t - buf_start > max_buf: agree = max(agree, len(hyp) - 2)
        for word, ws, we in hyp[:agree]: lat.append(busy_until - we); committed.append(word)
        if agree: buf_start = hyp[agree - 1][2]; prev = []
        else: prev = hyp
    w, cost = timed(f, a[int(buf_start * 16000):], lang); busy += cost
    for word, ws, we in [(x, buf_start + s, buf_start + e) for x, s, e in w]: lat.append(max(busy_until, dur) + cost - we); committed.append(word)
    return finish(lat, committed, ref, busy, wait_max, dur)

STRATS = [('whole', lambda f, a, l, r: sim_whole(f, a, l, r)), ('pause', lambda f, a, l, r: sim_segments(f, a, l, r, 0.3, 15.0)),
          ('clause', lambda f, a, l, r: sim_segments(f, a, l, r, 0.2, 4.0)), ('chunk-2.0', lambda f, a, l, r: sim_chunks(f, a, l, r, 2.0)),
          ('agree-1.0', lambda f, a, l, r: sim_agree(f, a, l, r, 1.0)), ('agree-0.5', lambda f, a, l, r: sim_agree(f, a, l, r, 0.5))]
for sname, (lang, a, ref) in streams.items():
    for ename, f in ENGINES.items():
        if ename == 'kinit' and lang != 'sk': continue
        if ename == 'parakeet' and lang not in ('en', 'de', 'sk'): continue
        for kind, fn in STRATS:
            key = f'{sname}|{ename}|{kind}'
            try: R['cells'][key] = fn(f, a, lang, ref); c = R['cells'][key]; log('cell', cell=key, **{k: v for k, v in c.items() if k != 'text'})
            except Exception as e:
                import traceback; log('cell_fail', cell=key, err=f'{type(e).__name__}: {e}'[:200], tb=traceback.format_exc()[-300:])
R['status'] = 'done'; log('done', wall_s=round(time.time() - T0))
