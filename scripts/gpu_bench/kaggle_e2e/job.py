"""Kaggle T4: per-language end-to-end chain timing (STT -> MT -> TTS) on FLEURS clips, plus Parakeet on a real GPU provider.
STT: faster-whisper large-v3-turbo fp16 (kinit/whisper-large-v3-turbo-sk for Slovak). MT: Opus-MT (transformers, fp16 GPU).
TTS: Piper voices on the Kaggle CPU (same as the Mac pipeline). Per clip: stt_s, mt_s, tts_s, chain_s, audio_in_s, audio_out_s.
Writes /kaggle/working/e2e.json after every cell."""
import glob, io, json, os, re, subprocess, sys, time, ctypes, wave
from pathlib import Path

OUT = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path('.')
R = {'chain': {}, 'parakeet': {}, 'log': [], 'env': {}}
T0 = time.time(); N = int(os.environ.get('N_CLIPS', 25))
def save(): (OUT / 'e2e.json').write_text(json.dumps(R, ensure_ascii=False, indent=1))
def log(m, **k): R['log'].append({'t': round(time.time() - T0, 1), 'm': m, **k}); print(m, k, flush=True); save()
def sh(c, t=2400):
    p = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t); return p.returncode, (p.stdout + p.stderr)[-900:]

rc, o = sh(f'{sys.executable} -m pip install -q faster-whisper sentencepiece soundfile pyarrow scipy piper-tts sacremoses nvidia-cublas-cu12 nvidia-cudnn-cu12 && {sys.executable} -m pip uninstall -y -q onnxruntime onnxruntime-gpu; {sys.executable} -m pip install -q onnxruntime-gpu onnx-asr[hub]'); log('pip', rc=rc, tail=o[-200:])
import numpy as np, soundfile as sf, pyarrow.parquet as pq, scipy.signal as ss, torch
import nvidia.cublas.lib as _cb, nvidia.cudnn.lib as _cd
for d in (list(_cb.__path__)[0], list(_cd.__path__)[0]):
    for so in sorted(glob.glob(d + '/*.so*')):
        try: ctypes.CDLL(so)
        except OSError: pass
from huggingface_hub import hf_hub_download
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
R['env']['gpu'] = sh('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader')[1].strip()

# language: (fleurs code, opus pair to en, opus pair from en, piper voice path in rhasspy/piper-voices)
LANGS = {
 'de': ('de_de', 'de-en', 'en-de', 'de/de_DE/thorsten/medium/de_DE-thorsten-medium'),
 'cs': ('cs_cz', 'cs-en', 'en-cs', 'cs/cs_CZ/jirka/medium/cs_CZ-jirka-medium'),
 'sk': ('sk_sk', 'sk-en', 'en-sk', 'sk/sk_SK/lili/medium/sk_SK-lili-medium'),
 'fr': ('fr_fr', 'fr-en', 'en-fr', 'fr/fr_FR/siwis/medium/fr_FR-siwis-medium'),
 'pl': ('pl_pl', 'pl-en', None,    'pl/pl_PL/gosia/medium/pl_PL-gosia-medium'),
}
EN_VOICE = 'en/en_US/john/medium/en_US-john-medium'
from piper import PiperVoice
def piper(path):
    onnx = hf_hub_download('rhasspy/piper-voices', path + '.onnx'); hf_hub_download('rhasspy/piper-voices', path + '.onnx.json')
    return PiperVoice.load(onnx)
def tts(voice, text):
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w: voice.synthesize_wav(text, w)
    buf.seek(0); a, sr = sf.read(buf); return len(a) / sr
voices = {}
def voice_for(key, path):
    if key not in voices:
        try: voices[key] = piper(path)
        except Exception as e: log('voice_fail', key=key, err=str(e)[:200]); voices[key] = None
    return voices[key]

def clips(code):
    f = hf_hub_download('google/fleurs', f'{code}/test/0000.parquet', repo_type='dataset', revision='refs/convert/parquet')
    out = []
    for r in pq.read_table(f).to_pylist()[:N]:
        a, sr = sf.read(io.BytesIO(r['audio']['bytes']), dtype='float32'); a = a.mean(1) if a.ndim > 1 else a
        if sr != 16000: a = ss.resample_poly(a, 16000, sr).astype('float32')
        out.append(a)
    return out
def convert_kinit():
    out = '/kaggle/working/ct2-kinit-turbo-sk'
    if not Path(out).exists():
        rc, o = sh(f'ct2-transformers-converter --model kinit/whisper-large-v3-turbo-sk --output_dir {out} --copy_files preprocessor_config.json --quantization float16 --force'); log('convert_kinit', rc=rc)
        if rc == 0:
            import shutil; shutil.copy(hf_hub_download('kinit/whisper-large-v3-turbo-sk', 'tokenizer.json'), out + '/tokenizer.json')
    return out
turbo = WhisperModel('large-v3-turbo', device='cuda', compute_type='float16')
kinit = None
mts = {}
def mt(pair):
    if pair not in mts:
        try:
            tok = AutoTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{pair}'); m = AutoModelForSeq2SeqLM.from_pretrained(f'Helsinki-NLP/opus-mt-{pair}').half().cuda().eval(); mts[pair] = (tok, m)
        except Exception as e: log('mt_fail', pair=pair, err=str(e)[:160]); mts[pair] = None
    return mts[pair]
def translate(pair, text):
    tm = mt(pair)
    if tm is None: return None
    tok, m = tm
    enc = tok([text], return_tensors='pt', truncation=True, max_length=256).to('cuda')
    with torch.no_grad(): g = m.generate(**enc, num_beams=4, max_new_tokens=256)
    return tok.batch_decode(g, skip_special_tokens=True)[0]
def cuda_sync(): torch.cuda.synchronize()

for l, (fc, p_to, p_from, vpath) in LANGS.items():
    try: cs = clips(fc)
    except Exception as e: log('fleurs_fail', lang=l, err=str(e)[:160]); continue
    model = turbo
    if l == 'sk':
        try: kinit = kinit or WhisperModel(convert_kinit(), device='cuda', compute_type='float16'); model = kinit
        except Exception as e: log('kinit_fail', err=str(e)[:200])
    # direction X -> EN: STT in X, MT X->en, TTS English voice
    ven = voice_for('en', EN_VOICE)
    if ven and p_to:
        rows = []
        for a in cs:
            t0 = time.perf_counter(); segs, _ = model.transcribe(a, language=l, beam_size=5, vad_filter=False, condition_on_previous_text=False); txt = ' '.join(s.text for s in segs).strip(); t1 = time.perf_counter()
            en = translate(p_to, txt) or ''; cuda_sync(); t2 = time.perf_counter()
            out_s = tts(ven, en) if en else 0.0; t3 = time.perf_counter()
            rows.append((t1 - t0, t2 - t1, t3 - t2, len(a) / 16000, out_s))
        r = np.array(rows); R['chain'][f'{l}->en'] = {k: round(float(np.median(r[:, i])), 3) for i, k in enumerate(('stt_s', 'mt_s', 'tts_s', 'audio_in_s', 'audio_out_s'))}
        R['chain'][f'{l}->en']['chain_s_median'] = round(float(np.median(r[:, :3].sum(1))), 3); R['chain'][f'{l}->en']['n'] = len(rows)
        log('chain', direction=f'{l}->en', **R['chain'][f'{l}->en'])
    # direction EN -> X: English FLEURS audio, STT en, MT en->X, TTS X voice
    vx = voice_for(l, vpath)
    if vx and p_from:
        if 'en' not in globals().get('_en_clips', {}): _en_clips = {'en': clips('en_us')}
        rows = []
        for a in _en_clips['en']:
            t0 = time.perf_counter(); segs, _ = turbo.transcribe(a, language='en', beam_size=5, vad_filter=False, condition_on_previous_text=False); txt = ' '.join(s.text for s in segs).strip(); t1 = time.perf_counter()
            tx = translate(p_from, txt) or ''; cuda_sync(); t2 = time.perf_counter()
            out_s = tts(vx, tx) if tx else 0.0; t3 = time.perf_counter()
            rows.append((t1 - t0, t2 - t1, t3 - t2, len(a) / 16000, out_s))
        r = np.array(rows); R['chain'][f'en->{l}'] = {k: round(float(np.median(r[:, i])), 3) for i, k in enumerate(('stt_s', 'mt_s', 'tts_s', 'audio_in_s', 'audio_out_s'))}
        R['chain'][f'en->{l}']['chain_s_median'] = round(float(np.median(r[:, :3].sum(1))), 3); R['chain'][f'en->{l}']['n'] = len(rows)
        log('chain', direction=f'en->{l}', **R['chain'][f'en->{l}'])

# Parakeet on a real GPU provider
try:
    import onnxruntime as ort, onnx_asr
    from huggingface_hub import snapshot_download
    R['parakeet']['providers'] = ort.get_available_providers(); log('ort', version=ort.__version__, providers=R['parakeet']['providers'])
    pdir = snapshot_download('istupakov/parakeet-tdt-0.6b-v3-onnx', local_dir='/kaggle/working/parakeet-v3-onnx')
    pk = onnx_asr.load_model('nemo-parakeet-tdt-0.6b-v3', pdir, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    for l in ('de', 'en'):
        cs = clips(LANGS[l][0] if l != 'en' else 'en_us'); pk.recognize(cs[0])
        t0 = time.perf_counter(); [pk.recognize(a) for a in cs]; dt = time.perf_counter() - t0
        R['parakeet'][l] = {'s_per_clip': round(dt / len(cs), 3), 'RTF': round(dt / sum(len(a) / 16000 for a in cs), 3), 'session_providers': pk.asr._encoder.get_providers() if hasattr(pk, 'asr') and hasattr(pk.asr, '_encoder') else 'unknown'}
        log('parakeet', lang=l, **R['parakeet'][l])
except Exception as e:
    import traceback; log('parakeet_fail', err=f'{type(e).__name__}: {e}'[:300], tb=traceback.format_exc()[-500:])
R['status'] = 'done'; log('done', wall_s=round(time.time() - T0))
