# Voice build runbook — personal Piper voice, reproducible

Proven 2026-09-27 on this Mac (16GB, CPU-only). Goal: personal VITS voice at
Piper speed (RTF ~0.05). Each build is: record → segment → warmstart-train →
export → machine QC → ear QC in Voice Lab.

## 1. Record (the step that matters most)

- 2+ min per language to start; 15–20 min for the quality tier. Quiet room, steady
  pace, short pauses between sentences. Record HOT (peak ~0.5; our first takes
  peaked 0.14 — clean but quiet). Script: `documentation/reading_script_bilingual.md`.
- Save to `speaker_voices/<lang>_script_reading.m4a`, register in
  `speaker_voices/speaker_voices.json` with the SCRIPT text as `transcribed_text`
  (never auto-STT output — STT errors become training lies).

## 2. Segment into sentence clips

Whole-clip batches blow memory (18GB spike on 16GB Mac: VITS attention is
quadratic in frames) and train alignment poorly.

```
ffmpeg -y -i speaker_voices/sk_script_reading.m4a -ar 22050 -ac 1 /tmp/voice_build/voices_sk/sk_script_reading.wav
venv/bin/python scripts/segment_sk_dataset.py   # silence midpoints -> 18 clips + speaker_voices.json
```

Expect 18 clips, 5–10s each. (Script is SK-specific today; generalize per language.)

## 3. Warmstart checkpoint (never from scratch)

- Base must match speaker GENDER + sample rate; language family second.
  lili (F, 198Hz) → male speaker (101Hz) failed structurally across 2500 steps.
  jirka (M, Czech) is the correct SK base. EN: ryan (M).
- Same sample rate is mandatory (TRAINING.md); same language is NOT.
- `rhasspy/piper-checkpoints` (dataset) has per-voice `.ckpt`, e.g.
  `cs/cs_CZ/jirka/medium/epoch=8819-step=1435400.ckpt` (~850MB).

## 4. Train (separate venv, CPU)

```
.venv-train/bin/python scripts/finetune_personal_voice.py \
  --language sk --voice-name sk_SK-personal-male-medium --espeak-voice sk \
  --speaker-voices-dir /tmp/voice_build/voices_sk_seg \
  --max-steps 2500 \
  --ckpt-path /tmp/voice_build/ckpt/<base>.ckpt
```

- CPU beats MPS for VITS here (~8s/step). 2500 steps ≈ 6–8h. Monitor with
  `scripts` log tail; checkpoint guarantee shim in the script is REQUIRED
  (upstream val_mel/val_mos callbacks never fire on tiny datasets → zero files).
- Norms: ~1000 epochs for fine-tune (TRAINING.md). 200 steps ≈ robotic.
- Watch RAM: close browsers/heavy apps; never run STT spikes concurrently.

## 5. Export + machine QC (before ears)

Export lands `backend/tts/piper_models/<voice>.onnx` (+`.json`). Then:

```
venv/bin/python scripts/machine_listen_qc.py   # jitter, HNR, F0-range, WER thirds
```

Ship thresholds (relative to generic baseline): F0-std within ~80% of generic,
jitter ≤ ~1.5x generic, HNR within a few dB. Below that → more steps/data, not knobs
(noise_scale variants never moved jitter in our sweeps).

## 6. Ear QC in Voice Lab

```
python3 scripts/update_voice_lab_library.py --no-test   # hard-refresh browser after
```

Judge timbre identity separately from smoothness (rough-but-you = more steps;
smooth stranger = wrong base/data). Archive failures to
`processed/voice_qc_archive_<date>/` — evidence, never delete.

## Dead ends recorded (do not retry)

- XTTS fine-tune for speed: speed is architectural (autoregressive); best case
  ~0.3 RTF vs Piper 0.05. XTTS has no Slovak at all.
- XTTS-Czech reading Slovak text: F0 depressed (86Hz), Czech-accented, RTF 2.7.
- Synthetic data augmentation (XTTS-generated training clips): teaches the copy's
  artifacts, not the speaker. Real minutes only.
- noise_scale tuning for tremor: no effect on jitter. Steps fix weights; knobs don't.
