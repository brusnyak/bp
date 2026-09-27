# Personal Piper voice workflow (reusable per language)

Proven 2026-09-27 on this Mac (16GB, CPU-only): `sk_SK-personal-medium` (lili base,
val_mel 0.57→0.29) then `sk_SK-personal-male-medium` (jirka base). Total per run:
~5–8h training + minutes of prep. Inference stays Piper-native (RTF ~0.05).

## 0. Record (the only step that needs a human)

- 2–3 min per language minimum; 15–20 min for PVC-grade. Quiet room, steady pace,
  short pauses between sentences. Record HOT (peak ~0.5; our 0.14 was too quiet).
- Read Palm script: `documentation/reading_script_bilingual.md` (18 sentences).
- Drop the file into `speaker_voices/` + register it in `speaker_voices.json`
  with `transcribed_text` = the script text verbatim (never auto-STT output).

## 1. Segment into sentence clips

Why: single-blob training OOMs attention on 16GB (18GB spike observed) and learns
no alignment; 5–10s clips enable real batching + a working val split.

```bash
ffmpeg -y -i speaker_voices/<lang>_script_reading.m4a -ar 22050 -ac 1 /tmp/piper_finetune_work/voices_<lang>_<spk>/<file>.wav
venv/bin/python scripts/segment_sk_dataset.py   # silence-boundary splitter; adapt paths per language
```

Expect: N clips matching N script sentences, 5–10s each, no fallbacks triggered.

## 2. Warmstart checkpoint (never train from scratch)

- Same sample rate (22050) required; same language NOT required (Piper TRAINING.md).
- Match speaker pitch: male speaker → male base (we learned this the hard way:
  user 101Hz vs lili 198Hz wasted 2500 steps fighting an octave gap).
- Bases: SK→`cs_CZ-jirka-medium` (male, Slavic); EN→`en_US-ryan-medium` (male).
- Download one file only:

```bash
venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('rhasspy/piper-checkpoints', repo_type='dataset',
    allow_patterns=['cs/cs_CZ/jirka/medium/epoch=8819-step=1435400.ckpt'],
    local_dir='/tmp/piper_finetune_work/ckpt')"
```

## 3. Train (background, monitored)

```bash
nohup .venv-train/bin/python scripts/finetune_personal_voice.py \
  --language sk --voice-name sk_SK-personal-male-medium --espeak-voice sk \
  --speaker-voices-dir /tmp/piper_finetune_work/voices_sk_seg \
  --max-steps 2500 \
  --ckpt-path /tmp/piper_finetune_work/ckpt/cs/cs_CZ/jirka/medium/epoch=8819-step=1435400.ckpt \
  > /tmp/piper_finetune_work/train_sk_male.log 2>&1 &
```

- Machine owns the CPU while training: no spikes/probes concurrently.
- Monitor: `*/15 * * * * /tmp/piper_finetune_work/monitor.sh` (cron) → `monitor.log`;
  completion = `Done. Personal voice ready at backend/tts/piper_models/<name>.onnx`.
- Landmarks: val_mel should fall (0.57→0.29 SK); ~1000 epochs is the community
  norm for fine-tune (Piper TRAINING.md) — 200 steps sounds robotic, guaranteed.
- Gotchas fixed in-repo: checkpoint guarantee shim (empty val split used to write
  zero files); `--trainer.accelerator cpu` (MPS is 4–6x slower for VITS);
  dataset must be WAV (script rejects m4a).

## 4. QC before trusting it

```bash
venv/bin/python -c "  # A/B test sentence, new voice vs generic
from backend.tts.piper_tts import PiperTTS
import soundfile as sf
t = PiperTTS(model_id='sk_SK-personal-male-medium')
wav, sr, tts_t = t.synthesize('<test sentence>', language='sk')
sf.write('processed/voice_qc/<label>.wav', wav, sr)"
venv/bin/python scripts/machine_listen_qc.py   # jitter/HNR/F0 vs generic baseline
python3 scripts/update_voice_lab_library.py --no-test   # lab review
```

Pass bars (measured generics): F0-range ≈ 22Hz (lili), jitter ≈ 0.022, HNR ≈ +5dB.
Human ear decides timbre; numbers decide robotic/hiss/tremor. Then wire as default.

## 5. What NOT to do

- No XTTS/CosyVoice/F5 for Slovak (all SK-less or CUDA-bound); XTTS zero-shot is
  EN timbre-reference only (RTF 1.42 disqualifies live use).
- No synthetic self-extension (training on own outputs bakes in artifacts).
- No perturbation "repopulation" as a substitute for real minutes.
- XTTS-generated extension audio is legitimate ONLY for EN data (XTTS speaks EN
  well); never for SK.
- More real minutes (15–20) beats every trick if quality plateaus.
