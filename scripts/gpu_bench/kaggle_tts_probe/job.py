"""TTS probe on a Kaggle T4 (2026-11-10): is per-piece synthesis fast enough for a <2 s near-real-time route? Latency and feasibility only, no quality scoring.
Pieces: 3- and 6-word cuts of the generated Slovak and Czech test sentences (what the route's fragmenter hands to TTS).
Arm A  OmniVoice cloned voice per piece (num_step 12 and 16), reference = first 10 s of the test speaker's Slovak reading (ref text auto-transcribed).
Arm B  Piper (sk lili, cs jirka) per piece, then OpenVoice V2 tone-colour conversion to the same reference (separate venv so installs cannot break arm A).
Reports per piece wall time and audio length, sustained ratio (sum wall / sum audio, below 1.0 = the queue does not grow). Clips are saved in the output for an ear check.
Writes /kaggle/working/tts_probe.json after every step; every arm is isolated, one failure does not stop the others."""
import glob, json, os, re, shutil, numpy as np, subprocess, sys, time, traceback, wave
from pathlib import Path

OUT = Path('/kaggle/working'); R = {'arms': {}, 'log': [], 'env': {}}; T0 = time.time()
def save(): (OUT / 'tts_probe.json').write_text(json.dumps(R, ensure_ascii=False, indent=1))
def log(m, **k): R['log'].append({'t': round(time.time() - T0, 1), 'm': m, **k}); print(m, k, flush=True); save()
def sh(c, t=1800, **kw):
    p = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t, **kw); return p.returncode, (p.stdout + p.stderr)[-1500:]

R['env']['gpu'] = sh('nvidia-smi --query-gpu=name --format=csv,noheader')[1].strip()
audio = next(Path('/kaggle/input').rglob('sk_script_reading.m4a'))
ref = OUT / 'ref_sk.wav'; log('ref', rc=sh(f'ffmpeg -y -v error -t 10 -i {audio} -ac 1 -ar 24000 {ref}')[0])
def pieces(lang, n=3, per=8):
    meta = json.loads(next(Path('/kaggle/input').rglob(f'{lang}_paced.json')).read_text()); w = meta['text'].split()
    return [' '.join(w[i:i + n]) for i in range(0, len(w) - n + 1, n)][:per]
PIECES = {(lang, n): pieces(lang, n) for lang in ('sk', 'cs') for n in (3, 6)}
log('pieces', **{f'{l}{n}': len(v) for (l, n), v in PIECES.items()})
def stats(rows):
    wall = sum(r['wall_s'] for r in rows); aud = sum(r['audio_s'] for r in rows); w = sorted(r['wall_s'] for r in rows)
    return {'n': len(rows), 'wall_median_s': round(w[len(w) // 2], 3), 'wall_max_s': round(w[-1], 3), 'sustained_ratio': round(wall / aud, 3) if aud else None}

OV_ARM = r'''"""Arm B worker (own venv): OpenVoice V2 tone-colour conversion latency on Piper pieces. argv: ckpt_dir ref_wav src_dir out_dir out_json"""
import glob, json, os, sys, time, traceback
ckpt, ref, src, out, outj = sys.argv[1:6]; os.makedirs(out, exist_ok=True); R = {'pieces': [], 'errors': []}
def save(): open(outj, 'w').write(json.dumps(R, indent=1))
try:
    import torch
    from openvoice.api import ToneColorConverter
    t = time.perf_counter(); conv = ToneColorConverter(f'{ckpt}/converter/config.json', device='cuda'); conv.load_ckpt(f'{ckpt}/converter/checkpoint.pth')
    R['load_s'] = round(time.perf_counter() - t, 2)
    t = time.perf_counter(); tgt = conv.extract_se([ref]); R['extract_target_s'] = round(time.perf_counter() - t, 2)
    files = sorted(glob.glob(f'{src}/*.wav')); long_src = max(files, key=os.path.getsize)
    t = time.perf_counter(); src_se = conv.extract_se([long_src]); R['extract_source_s'] = round(time.perf_counter() - t, 2)   # fixed Piper voice: done once
    for f in files[:2]: conv.convert(audio_src_path=f, src_se=src_se, tgt_se=tgt, output_path=f'{out}/warm.wav', tau=0.3, message='default')   # warm-up
    for f in files:
        torch.cuda.synchronize(); t = time.perf_counter()
        conv.convert(audio_src_path=f, src_se=src_se, tgt_se=tgt, output_path=f'{out}/{os.path.basename(f)}', tau=0.3, message='default')
        torch.cuda.synchronize(); R['pieces'].append({'file': os.path.basename(f), 'convert_s': round(time.perf_counter() - t, 3)}); save()
except Exception as e: R['errors'].append(f'{type(e).__name__}: {e}'); R['tb'] = traceback.format_exc()[-1500:]
save()
'''
(OUT / 'ov_arm.py').write_text(OV_ARM)   # the kernel ships job.py only, so the worker is embedded

# ---------------- Arm A: OmniVoice per piece ----------------
try:
    raise RuntimeError('arm A skipped in v3: measured in v2 (omni_step12 median 0.658 s, step16 0.9 s)')
    rc, o = sh(f'{sys.executable} -m pip install -q omnivoice'); log('pip omnivoice', rc=rc, tail=o[-200:])
    import torch, soundfile as sf
    from omnivoice import OmniVoice
    t = time.perf_counter(); model = OmniVoice.from_pretrained('k2-fsa/OmniVoice', device_map='cuda', dtype=torch.float16); log('omni load', s=round(time.perf_counter() - t, 1))
    t = time.perf_counter(); prompt = model.create_voice_clone_prompt(str(ref), None); log('omni prompt', s=round(time.perf_counter() - t, 2))
    model.generate(text='Test.', language='sk', voice_clone_prompt=prompt, num_step=12)   # warm-up
    od = OUT / 'clips_omni'; od.mkdir(exist_ok=True)
    for steps in (12, 16):
        rows = []
        for (lang, n), txts in PIECES.items():
            for i, txt in enumerate(txts):
                torch.cuda.synchronize(); t = time.perf_counter()
                a = model.generate(text=txt, language=lang, voice_clone_prompt=prompt, num_step=steps)[0]
                torch.cuda.synchronize(); dt = time.perf_counter() - t
                a = np.asarray(a.detach().float().cpu().numpy() if hasattr(a, 'detach') else a, dtype='float32').squeeze(); sr = getattr(model, 'sampling_rate', 24000)
                sf.write(od / f's{steps}_{lang}_{n}w_{i}.wav', a, sr)
                rows.append({'lang': lang, 'words': n, 'wall_s': round(dt, 3), 'audio_s': round(len(a) / sr, 2)})
        R['arms'][f'omni_step{steps}'] = {'summary': stats(rows), 'by_size': {f'{l}{n}': stats([r for r in rows if (r['lang'], r['words']) == (l, n)]) for (l, n) in PIECES}, 'rows': rows}
        log('omni done', steps=steps, **R['arms'][f'omni_step{steps}']['summary'])
    del model; torch.cuda.empty_cache()
except Exception as e: log('omni_fail', err=f'{type(e).__name__}: {e}'[:300], tb=traceback.format_exc()[-600:])

# ---------------- Arm B: Piper + OpenVoice V2 ----------------
try:
    rc, o = sh(f'{sys.executable} -m pip install -q piper-tts'); log('pip piper', rc=rc, tail=o[-200:])
    pdir = OUT / 'piper_voices'; pdir.mkdir(exist_ok=True)
    for v in ('sk_SK-lili-medium', 'cs_CZ-jirka-medium'):
        rc, o = sh(f'{sys.executable} -m piper.download_voices {v} --data-dir {pdir}'); log('piper voice', v=v, rc=rc, tail=o[-200:] if rc else '')
    from piper import PiperVoice
    srcdir = OUT / 'clips_piper'; srcdir.mkdir(exist_ok=True); rows = []
    for lang, v in (('sk', 'sk_SK-lili-medium'), ('cs', 'cs_CZ-jirka-medium')):
        voice = PiperVoice.load(str(pdir / f'{v}.onnx'))
        for (l, n), txts in PIECES.items():
            if l != lang: continue
            for i, txt in enumerate(txts):
                f = srcdir / f'{lang}_{n}w_{i}.wav'; t = time.perf_counter()
                with wave.open(str(f), 'wb') as wf:
                    try: voice.synthesize_wav(txt, wf)
                    except AttributeError: voice.synthesize(txt, wf)
                dt = time.perf_counter() - t
                with wave.open(str(f)) as w: aud = w.getnframes() / w.getframerate()
                rows.append({'lang': lang, 'words': n, 'wall_s': round(dt, 3), 'audio_s': round(aud, 2)})
    R['arms']['piper'] = {'summary': stats(rows), 'rows': rows}; log('piper done', **R['arms']['piper']['summary'])
    rc, o = sh(f'{sys.executable} -m pip install -q --no-deps git+https://github.com/myshell-ai/OpenVoice.git && {sys.executable} -m pip install -q inflect unidecode jieba pypinyin cn2an eng_to_ipa langid pykakasi pydub wavmark', 1500); log('pip openvoice', rc=rc, tail=o[-400:])
    from huggingface_hub import snapshot_download
    ckpt = snapshot_download('myshell-ai/OpenVoiceV2'); log('ov ckpt', path=ckpt, files=sorted(os.listdir(ckpt))[:10])
    rc, o = sh(f'{sys.executable} {OUT / "ov_arm.py"} {ckpt} {ref} {srcdir} {OUT / "clips_openvoice"} {OUT / "ov_arm.json"}', 1500); log('ov arm', rc=rc, tail=o[-300:])
    ov = json.loads((OUT / 'ov_arm.json').read_text()); conv = [p['convert_s'] for p in ov.get('pieces', [])]
    if conv:
        by = {r['lang'] + str(r['words']) + str(i): r for i, r in enumerate(rows)}; tot_audio = sum(r['audio_s'] for r in rows)
        R['arms']['piper_plus_openvoice'] = {'load_s': ov.get('load_s'), 'extract_target_s': ov.get('extract_target_s'), 'extract_source_s': ov.get('extract_source_s'),
            'convert_median_s': sorted(conv)[len(conv) // 2], 'convert_max_s': max(conv), 'sustained_ratio_incl_piper': round((sum(conv) + sum(r['wall_s'] for r in rows)) / tot_audio, 3), 'errors': ov.get('errors')}
    else: R['arms']['piper_plus_openvoice'] = {'errors': ov.get('errors'), 'tb': ov.get('tb')}
    log('openvoice done', **{k: v for k, v in R['arms']['piper_plus_openvoice'].items() if k != 'tb'})
except Exception as e: log('arm_b_fail', err=f'{type(e).__name__}: {e}'[:300], tb=traceback.format_exc()[-600:])
shutil.rmtree('/kaggle/working/piper_voices', ignore_errors=True)
R['status'] = 'done'; log('done', wall_s=round(time.time() - T0))
