"""Piper fine-tune of the male Czech voice cs_CZ-jirka-medium on the test speaker's cloned Czech data, timed (2026-11-10).
Uses OHF-voice/piper1-gpl (`python -m piper.train fit`, maintained; rhasspy/piper is archived). Recipe from its docs/TRAINING.md: warm start from the Piper
checkpoint (`--model.warmstart_ckpt`: weights copied, fresh optimizer, epoch 0), batch 32, lr anneal derived from --trainer.max_epochs.
Phase 1 measures seconds per epoch on this T4 (4 epochs, then 8, cache warm). Phase 2 picks the epoch count that fits the time budget
(capped at 1000), trains once with that length, exports ONNX and renders Czech sample sentences for the fine-tune and the original Jirka.
Checkpoints stay in /tmp (846 MB each); /kaggle/working gets only the .onnx, config, wav samples and train_report.json."""
import glob, json, os, re, shutil, subprocess, sys, time, traceback
from pathlib import Path

OUT = Path('/kaggle/working'); TMP = Path('/tmp'); R = {'log': [], 'env': {}}; T0 = time.time()
BUDGET_S = 8.2 * 3600; MAX_EPOCHS = 1200; BATCH = 32
def save(): (OUT / 'train_report.json').write_text(json.dumps(R, ensure_ascii=False, indent=1))
def log(m, **k): R['log'].append({'t': round(time.time() - T0, 1), 'm': m, **k}); print(m, k, flush=True); save()
def sh(c, t=3600, **kw):
    p = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t, **kw); return p.returncode, (p.stdout + p.stderr)[-1500:]
R['env']['gpu'] = sh('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader')[1].strip()

N_FT = 300   # fine-tune cost scales with audio hours: use the first 300 clips that passed QC (about 25 min)
meta0 = next(Path('/kaggle/input').rglob('metadata.csv')); DATA = meta0.parent; wavs = DATA / 'wavs'
lines = [l for l in meta0.read_text(encoding='utf-8').splitlines() if l.strip()][:N_FT]; meta = TMP / 'subset.csv'; meta.write_text('\n'.join(lines) + '\n', encoding='utf-8')
n = len(lines); import soundfile as sf
secs = sum(sf.info(wavs / l.split('|')[0]).duration for l in lines); log('data', clips=n, minutes=round(secs / 60, 1), mean_s=round(secs / max(n, 1), 2))
if n < 50: raise SystemExit('too few clips')

rc, o = sh('apt-get update -qq && apt-get install -y -qq espeak-ng build-essential cmake ninja-build', 900); log('apt', rc=rc, tail=o[-200:] if rc else '')
rc, o = sh(f'git clone --depth 1 https://github.com/OHF-voice/piper1-gpl.git {TMP}/piper1', 300); log('clone', rc=rc)
rc, o = sh(f"cd {TMP}/piper1 && {sys.executable} -m pip install -q -e '.[train]'", 1800); log('pip train', rc=rc, tail=o[-400:])
rc, o = sh(f'{sys.executable} -m pip install -q scikit-build cython cmake ninja setuptools wheel onnx onnxscript onnxruntime', 900); log('pip build tools + onnx export deps', rc=rc, tail=o[-200:] if rc else '')
rc, o = sh(f'cd {TMP}/piper1 && ({"./build_monotonic_align.sh" if (TMP / "piper1/build_monotonic_align.sh").exists() else "true"}) && {sys.executable} setup.py build_ext --inplace', 1800); log('build ext', rc=rc, tail=o[-300:])
rc, o = sh(f'{sys.executable} -m piper.train fit --help | head -3; {sys.executable} -m piper.train.export_onnx --help | head -3'); log('cli check', tail=o[-300:])
from huggingface_hub import hf_hub_download
os.environ['HF_HOME'] = '/tmp/hf'
ckpt = hf_hub_download('rhasspy/piper-checkpoints', 'cs/cs_CZ/jirka/medium/epoch=8819-step=1435400.ckpt', repo_type='dataset', cache_dir='/tmp/hf'); log('ckpt', path=ckpt)
rc, o = sh(f'{sys.executable} -m piper.download_voices cs_CZ-jirka-medium --data-dir {TMP}/voices'); log('jirka voice', rc=rc)

def fit(epochs, extra=''):
    cmd = (f'cd {TMP}/piper1 && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 {sys.executable} -m piper.train fit --data.voice_name cs_owner --data.csv_path {meta} --data.audio_dir {wavs} --model.sample_rate 22050 '
           f'--data.espeak_voice cs --data.cache_dir {TMP}/cache --data.config_path {TMP}/cs_owner_config.json --data.batch_size {BATCH} '
           f'--model.warmstart_ckpt {ckpt} --trainer.max_epochs {epochs} --trainer.default_root_dir {TMP}/train_{epochs} --trainer.accelerator gpu --trainer.devices 1 {extra}')
    t = time.perf_counter(); rc, o = sh(f'({cmd}) > {TMP}/fit_{epochs}.log 2>&1', 6 * 3600)
    dt = time.perf_counter() - t; txt = Path(f'{TMP}/fit_{epochs}.log').read_text(errors='ignore')
    prog = [l for l in re.split(r'[\r\n]+', txt) if re.search(r'Epoch \d+|Vocoder coverage|it/s|s/it|Error|error|Killed', l)]
    R.setdefault('fit_logs', {})[str(epochs)] = {'wall_s': round(dt, 1), 'rc': rc, 'progress_lines': (prog[:6] + prog[-10:])}; save()
    return dt, rc, txt[-1200:]
for BATCH in (16, 8, 4):   # 32 ran out of memory on a 15 GB T4 with 10 s clips (first real run, 2026-11-10)
    t4, rc4, o4 = fit(2); log('probe 2 epochs (includes the one-time phoneme/audio cache)', batch=BATCH, s=round(t4, 1), rc=rc4, oom='OutOfMemory' in o4, tail=o4[-300:] if rc4 else '')
    if rc4 == 0 or 'OutOfMemory' not in o4: break
if rc4: raise SystemExit('fit failed: ' + o4[-300:])
t8, rc8, o8 = fit(6); per_epoch = max((t8 - t4) / 4, 1.0); startup = max(t4 - 2 * per_epoch, 0)
R['timing'] = {'s_per_epoch': round(per_epoch, 1), 'startup_s': round(startup), 'batch': BATCH}; log('timing', **R['timing'])
# dry run of the late stages on the 8-epoch checkpoint, so an export or synthesis bug costs minutes, not the whole training run
import wave
ck8 = sorted(glob.glob(f'{TMP}/train_6/**/*.ckpt', recursive=True), key=os.path.getmtime)
rc, o = sh(f'{sys.executable} -m piper.train.export_onnx --checkpoint {ck8[-1]} --output-file {TMP}/dry.onnx') if ck8 else (1, 'no checkpoint found under train_6')
log('dry export', rc=rc, tail=o[-300:])
if rc: raise SystemExit('export dry-run failed: ' + o[-200:])
shutil.copy(TMP / 'cs_owner_config.json', TMP / 'dry.onnx.json')
from piper import PiperVoice
vd = PiperVoice.load(str(TMP / 'dry.onnx'))
with wave.open(str(TMP / 'dry.wav'), 'wb') as wf:
    try: vd.synthesize_wav('Dobré ráno a vítejte.', wf)
    except AttributeError: vd.synthesize('Dobré ráno a vítejte.', wf)
log('dry synth ok', bytes=os.path.getsize(TMP / 'dry.wav'))
epochs = int(min(MAX_EPOCHS, max(50, (BUDGET_S - (time.time() - T0) - startup - 900) / per_epoch)))
R['plan'] = {'epochs': epochs, 'expected_h': round(epochs * per_epoch / 3600, 2), 'vocoder_passes_formula': round(epochs * 0.37 / (secs / n), 1)}; log('plan', **R['plan'])
tt = time.perf_counter(); tr, rcr, orr = fit(epochs, extra='--trainer.max_time 00:08:30:00'); R['train'] = {'epochs': epochs, 'wall_h': round((time.perf_counter() - tt) / 3600, 2), 'rc': rcr}; log('trained', tail=orr[-500:], **R['train'])

# ---- export + samples ----
cks = sorted(glob.glob(f'{TMP}/train_{epochs}/**/*.ckpt', recursive=True), key=os.path.getmtime)
log('ckpts', found=[Path(c).name for c in cks][-3:])
if cks:
    shutil.copy(cks[-1], TMP / 'final.ckpt')
    rc, o = sh(f'{sys.executable} -m piper.train.export_onnx --checkpoint {TMP}/final.ckpt --output-file {OUT}/cs_owner.onnx'); log('export', rc=rc, tail=o[-300:])
    shutil.copy(TMP / 'cs_owner_config.json', OUT / 'cs_owner.onnx.json')
    from piper import PiperVoice
    import wave
    S = ['Dobré ráno a vítejte u této živé ukázky překladu řeči.', 'Zpoždění je časový odstup mezi mými slovy a hlasem, který slyšíte.', 'Schůzka začíná ve čtvrt na deset v místnosti dva sedmnáct.', 'Tichá místnost je důležitější než drahý mikrofon.']
    R['samples'] = {}
    for tag, onnx in (('finetune', OUT / 'cs_owner.onnx'), ('jirka_original', TMP / 'voices/cs_CZ-jirka-medium.onnx')):
        try:
            v = PiperVoice.load(str(onnx)); tot = dur = 0.0
            for i, s in enumerate(S):
                t = time.perf_counter()
                with wave.open(str(OUT / f'sample_{tag}_{i}.wav'), 'wb') as wf:
                    try: v.synthesize_wav(s, wf)
                    except AttributeError: v.synthesize(s, wf)
                tot += time.perf_counter() - t
                with wave.open(str(OUT / f'sample_{tag}_{i}.wav')) as w: dur += w.getnframes() / w.getframerate()
            R['samples'][tag] = {'synth_s': round(tot, 2), 'audio_s': round(dur, 1), 'rtf': round(tot / dur, 3)}; log('samples', tag=tag, **R['samples'][tag])
        except Exception as e: log('sample_fail', tag=tag, err=f'{type(e).__name__}: {e}'[:200])
R['status'] = 'done'; log('done', wall_h=round((time.time() - T0) / 3600, 2))
