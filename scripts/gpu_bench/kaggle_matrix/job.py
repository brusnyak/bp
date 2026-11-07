"""Kaggle T4 language matrix: STT (WER, RTF, VRAM) on FLEURS audio and MT (chrF) on FLORES text, per language.
Self-contained (internet on). Writes /kaggle/working/matrix.json after every cell, so a crash keeps partial results.
STT: faster-whisper base/small/large-v3-turbo (fp16), Parakeet TDT v3 via onnx-asr. MT: opus-mt (ct2-free, transformers) vs NLLB-600M.
"""
import io, json, os, re, subprocess, sys, time, traceback
from pathlib import Path

OUT = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path('.')
R = {'stt': {}, 'mt': {}, 'env': {}, 'log': []}
N_AUDIO = int(os.environ.get('N_AUDIO', 60)); N_TEXT = int(os.environ.get('N_TEXT', 200))
LANGS = {  # code: (fleurs, flores, opus-mt pair to en, opus-mt pair from en, nllb code, parakeet-supported)
    'de': ('de_de', 'deu_Latn', 'de-en', 'en-de', 'deu_Latn', True),
    'cs': ('cs_cz', 'ces_Latn', 'cs-en', 'en-cs', 'ces_Latn', True),
    'sk': ('sk_sk', 'slk_Latn', 'sk-en', 'en-sk', 'slk_Latn', True),
    'pl': ('pl_pl', 'pol_Latn', 'pl-en', 'en-pl', 'pol_Latn', True),
    'fr': ('fr_fr', 'fra_Latn', 'fr-en', 'en-fr', 'fra_Latn', True),
    'en': ('en_us', 'eng_Latn', None, None, 'eng_Latn', True),
}

def save(): (OUT / 'matrix.json').write_text(json.dumps(R, ensure_ascii=False, indent=1))
def log(m, **k): R['log'].append({'t': round(time.time() - T0, 1), 'm': m, **k}); print(m, k, flush=True); save()
def sh(c, t=1500):
    p = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t); return p.returncode, (p.stdout + p.stderr)[-800:]
T0 = time.time()
rc, o = sh(f'{sys.executable} -m pip install -q faster-whisper sacrebleu soundfile pyarrow "onnx-asr[gpu,hub]" sentencepiece nvidia-cublas-cu12 nvidia-cudnn-cu12'); log('pip', rc=rc, tail=o[-300:])
import numpy as np, soundfile as sf
import nvidia.cublas.lib as _cb, nvidia.cudnn.lib as _cd  # make cuBLAS/cuDNN 12 visible to ctranslate2 + onnxruntime
_dirs = [list(_cb.__path__)[0], list(_cd.__path__)[0]]  # namespace packages: __file__ is None
os.environ['LD_LIBRARY_PATH'] = ':'.join(_dirs + [os.environ.get('LD_LIBRARY_PATH', '')])
import ctypes, glob
for d in _dirs:
    for so in sorted(glob.glob(d + '/*.so*')):
        try: ctypes.CDLL(so)
        except OSError: pass
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
R['env']['gpu'] = sh('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader')[1].strip()

norm = lambda s: re.findall(r"[^\W_]+", s.lower())
def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    if not r: return None
    d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        p = d[:]; d[0] = i
        for j, b in enumerate(h, 1): d[j] = min(p[j] + 1, d[j-1] + 1, p[j-1] + (a != b))
    return d[-1] / len(r)

LANG_ONLY = tuple(os.environ.get('LANG_ONLY', 'de,cs').split(','))
def load_fleurs(code):
    f = hf_hub_download('google/fleurs', f'{code}/test/0000.parquet', repo_type='dataset', revision='refs/convert/parquet')
    t = pq.read_table(f).to_pylist()[:N_AUDIO]; out = []
    for r in t:
        a, sr = sf.read(io.BytesIO(r['audio']['bytes']), dtype='float32')
        if a.ndim > 1: a = a.mean(1)
        out.append((a, sr, r['transcription'] if 'transcription' in r else r['raw_transcription']))
    return out
clips = {}
for l, v in LANGS.items():
    if l not in LANG_ONLY: continue
    try: clips[l] = load_fleurs(v[0]); log('fleurs', lang=l, n=len(clips[l]), sec=round(sum(len(a) / s for a, s, _ in clips[l]), 1))
    except Exception as e: log('fleurs_fail', lang=l, err=str(e)[:200])

def run_stt(name, fn, vram=None):
    for l, cs in clips.items():
        if name.startswith('parakeet') and not LANGS[l][5]: continue
        try:
            fn(None, warm=True)
            ws, dt, dur = [], 0.0, 0.0
            for a, sr, ref in cs:
                t = time.perf_counter(); hyp = fn((a, sr), lang=l); d = time.perf_counter() - t
                w = wer(ref, hyp); dt += d; dur += len(a) / sr
                if w is not None: ws.append(w)
            R['stt'][f'{name}|{l}'] = {'WER': round(float(np.mean(ws)), 3), 'WER_median': round(float(np.median(ws)), 3), 'n': len(ws),
                                       'RTF': round(dt / dur, 3), 'per_clip_s': round(dt / len(cs), 3), 'audio_s': round(dur, 1)}
            log('stt', cell=f'{name}|{l}', **R['stt'][f'{name}|{l}'])
        except Exception as e: log('stt_fail', cell=f'{name}|{l}', err=f'{type(e).__name__}: {e}'[:300], tb=traceback.format_exc()[-400:])

MODE = os.environ.get('MODE', 'parakeet+mt')  # whisper results already captured in run v2
from faster_whisper import WhisperModel
for size in (('base', 'small', 'large-v3-turbo') if 'whisper' in MODE else ()):
    try:
        m = WhisperModel(size, device='cuda', compute_type='float16')
        def fn(x, lang=None, warm=False, m=m):
            if warm: return
            a, sr = x
            if sr != 16000:
                import scipy.signal as ss; a = ss.resample_poly(a, 16000, sr).astype('float32')
            segs, _ = m.transcribe(a, language=lang, beam_size=5, vad_filter=False)
            return ' '.join(s.text for s in segs)
        run_stt(f'whisper-{size}', fn); del m
    except Exception as e: log('whisper_fail', size=size, err=str(e)[:300])
try:
    import onnx_asr
    from huggingface_hub import snapshot_download
    pdir = snapshot_download('istupakov/parakeet-tdt-0.6b-v3-onnx', local_dir='/kaggle/working/parakeet-v3-onnx')
    log('parakeet_files', files=sorted(os.listdir(pdir))[:12])
    pk = onnx_asr.load_model('nemo-parakeet-tdt-0.6b-v3', pdir, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    import onnxruntime as ort; log('ort', version=ort.__version__, providers=ort.get_available_providers())
    def fnp(x, lang=None, warm=False):
        if warm: return
        a, sr = x
        if sr != 16000:
            import scipy.signal as ss; a = ss.resample_poly(a, 16000, sr).astype('float32')
        return pk.recognize(a)
    run_stt('parakeet-tdt-v3', fnp)
except Exception as e: log('parakeet_fail', err=f'{type(e).__name__}: {e}'[:300], tb=traceback.format_exc()[-500:])

# ---- MT: FLORES devtest parallel text, chrF ----
import sacrebleu, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
def opus100(l):
    for cfg in (f'en-{l}', f'{l}-en'):
        try: f = hf_hub_download('Helsinki-NLP/opus-100', f'{cfg}/test/0000.parquet', repo_type='dataset', revision='refs/convert/parquet'); break
        except Exception: f = None
    if f is None: raise FileNotFoundError(l)
    rows = [r['translation'] for r in pq.read_table(f).to_pylist()]
    rows = [r for r in rows if 4 <= len(r['en'].split()) <= 40 and 4 <= len(r[l].split()) <= 40][:N_TEXT]
    return [r[l] for r in rows], [r['en'] for r in rows]
pairs = {}
for l in LANGS:
    if l == 'en' or l not in LANG_ONLY: continue
    try: pairs[l] = opus100(l); log('opus100', lang=l, n=len(pairs[l][0]))
    except Exception as e: log('opus100_fail', lang=l, err=str(e)[:200])
def translate(tok, mdl, src, kw=None, bs=16):
    out = []
    for i in range(0, len(src), bs):
        enc = tok(src[i:i+bs], return_tensors='pt', padding=True, truncation=True, max_length=256).to('cuda')
        with torch.no_grad(): g = mdl.generate(**enc, num_beams=4, max_new_tokens=256, **(kw or {}))
        out += tok.batch_decode(g, skip_special_tokens=True)
    return out
def mt_cell(key, tok, mdl, src, ref, kw=None):
    t = time.perf_counter(); hyp = translate(tok, mdl, src, kw); dt = time.perf_counter() - t
    R['mt'][key] = {'chrF': round(sacrebleu.corpus_chrf(hyp, [ref]).score, 1), 'BLEU': round(sacrebleu.corpus_bleu(hyp, [ref]).score, 1),
                    'sent_per_s': round(len(src) / dt, 1), 'n': len(src), 'sample': hyp[:2]}
    log('mt', cell=key, **{k: v for k, v in R['mt'][key].items() if k != 'sample'})
for l, v in LANGS.items():
    if l == 'en' or l not in pairs: continue
    for d, pair in (('to_en', v[2]), ('from_en', v[3])):
        src, ref = (pairs[l][0], pairs[l][1]) if d == 'to_en' else (pairs[l][1], pairs[l][0])
        try:
            tok = AutoTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{pair}'); mdl = AutoModelForSeq2SeqLM.from_pretrained(f'Helsinki-NLP/opus-mt-{pair}').half().cuda().eval()
            mt_cell(f'opus-mt|{l}|{d}', tok, mdl, src, ref); del mdl
        except Exception as e: log('mt_fail', cell=f'opus|{l}|{d}', err=str(e)[:200])
try:
  for nm, tag in (('facebook/nllb-200-distilled-600M', 'nllb-600m'), ('facebook/nllb-200-distilled-1.3B', 'nllb-1.3b')):
    mdl = AutoModelForSeq2SeqLM.from_pretrained(nm).half().cuda().eval()
    for l, v in LANGS.items():
        if l == 'en' or l not in pairs: continue
        for d in ('to_en', 'from_en'):
            s_code, t_code = (v[4], 'eng_Latn') if d == 'to_en' else ('eng_Latn', v[4])
            tok = AutoTokenizer.from_pretrained(nm, src_lang=s_code); src, ref = (pairs[l][0], pairs[l][1]) if d == 'to_en' else (pairs[l][1], pairs[l][0])
            mt_cell(f'{tag}|{l}|{d}', tok, mdl, src, ref, kw={'forced_bos_token_id': tok.convert_tokens_to_ids(t_code)})
except Exception as e: log('nllb_fail', err=str(e)[:300])
R['status'] = 'done'; log('done', wall_s=round(time.time() - T0))
