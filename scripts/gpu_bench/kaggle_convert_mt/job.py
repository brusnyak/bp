"""CPU-only: convert Opus-MT pairs to CTranslate2 int8 with the project's own converter and smoke-test each one, so a Mac user does not have to
install torch locally. Output: /kaggle/working/ct2_models/Helsinki-NLP--opus-mt-<pair>/ (copy into the repo's ct2_models/)."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
OUT = Path('/kaggle/working'); BP = OUT / 'BP'; R = {'pairs': {}, 'log': []}
def save(): (OUT / 'convert_mt.json').write_text(json.dumps(R, ensure_ascii=False, indent=1))
def sh(c, t=3000, **kw):
    p = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t, **kw); return p.returncode, (p.stdout + p.stderr)[-600:]
src = next(p.parent for p in Path('/kaggle/input').rglob('app.py') if (p.parent / 'backend').is_dir())
shutil.copytree(src, BP, dirs_exist_ok=True)
rc, o = sh(f'{sys.executable} -m pip install -q ctranslate2 sentencepiece transformers==4.57.6 tokenizers==0.22.2 huggingface-hub==0.36.2'); R['pip'] = rc; save()
env = {**os.environ, 'PYTHONPATH': str(BP)}
SAMPLES = {'en-de': ('Good morning, and welcome to this live demonstration.', 'en', 'de'), 'de-en': ('Guten Morgen und willkommen zu dieser Vorführung.', 'de', 'en'),
           'en-fr': ('Good morning, and welcome to this live demonstration.', 'en', 'fr')}
for pair, (text, s, t) in SAMPLES.items():
    t0 = time.time()
    rc, o = sh(f"{sys.executable} -c \"import sys; sys.setrecursionlimit(2000); import backend.mt.convert_opus_mt_to_ct2 as c; c.convert_model('Helsinki-NLP/opus-mt-{pair}', '{OUT}/ct2_models/Helsinki-NLP--opus-mt-{pair}', quantization='int8')\"", cwd=BP, env=env)
    rec = {'convert_rc': rc, 'wall_s': round(time.time() - t0), 'tail': o if rc else ''}
    if rc == 0:
        rc2, o2 = sh(f"{sys.executable} -c \"from backend.mt.ctranslate2_mt import CTranslate2MT; m = CTranslate2MT(model_path='{OUT}/ct2_models/Helsinki-NLP--opus-mt-{pair}', device='cpu'); print(m.translate({text!r}, '{s}', '{t}')[0])\"", cwd=BP, env=env)
        rec['sample_in'] = text; rec['sample_out'] = o2.strip().splitlines()[-1] if o2.strip() else ''; rec['smoke_rc'] = rc2
    R['pairs'][pair] = rec; save(); print(pair, rec, flush=True)
shutil.rmtree(BP, ignore_errors=True)
R['status'] = 'done'; save()
