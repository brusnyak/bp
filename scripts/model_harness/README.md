# BP Model Test Harness — 8 GB VRAM bounded (T4 target)

**Purpose:** test whatever STT/MT/TTS/S2S models are available on a T4-class GPU (≤8 GB VRAM) and report comparable WER/chrF/RTF/VRAM/latency + ear-grade wavs. Drives the Kaggle T4 validation (Canary AST SK→EN WER < 0.2 gate) and multi-language expansion (fr, ar, zh, hi, African languages where coverage exists).

**Usage:**
```
python scripts/model_harness/run.py --stage stt --model canary-ast --device cuda --vram-limit 8 --clip me_sk_script.m4a --lang sk --ref "transcribed text"
python scripts/model_harness/run.py --stage tts --model kokoro --device cuda --vram-limit 8 --text "Hello world" --lang en
python scripts/model_harness/run.py --stage mt --model opus-mt --device cpu --vram-limit 8 --text "Good morning" --src en --tgt sk
```

**Output:** `processed/model_harness/<date>_<stage>_<model>_<lang>.json` + ear-grade wav next to it. Never overwrites (runN suffix).

**VRAM guard:** refuses to load a model whose declared VRAM exceeds `--vram-limit`. Declared VRAM per engine comes from `backend/registry.py` Capability.resource.
