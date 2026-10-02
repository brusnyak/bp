# Piper voice training runbook (no-repeat edition)

Goal: never train blind again. Every future run follows this sheet; any
deviation gets a one-line reason in the run log. Researched 2026-10-01;
authoritative numbers come from rhasspy/piper TRAINING.md, the rest is
measured here or community-verified (sources at the bottom).

## 0. The one paragraph that saves all future nights

Official Piper guidance: ~2000 epochs from scratch, **+1000 epochs when
fine-tuning**, done when `loss_disc_all` levels off. Our Sept 30 run did
**~250 epochs** (2500 steps, batch 8, 50 clips) — 4× short of the fine-tune
target. That alone explains trembling + mumbling with clean intelligibility:
VITS learns phones early, prosody (duration predictor, flow) late. Research
2025–2026 confirms duration modeling is the known-hard part of VITS
(VITS2 adversarial DP, FNH-TTS MoE-DP). Undertrained duration = wobble.

## 1. Before any train (free, no fans)

1. **Ear-grade the corpus first.** Drop every kill, normalize survivors to one
   RMS (`scripts/grade_library.py` flags them; Lab shows the pills). Never
   train on clips you would not ship — the model learns the average,
   and the average of clean + clipped is wobble.
2. **Loudness + silence pass.** One RMS target, trimmed leading/trailing
   silence, no full-scale peaks (our corpus had peak-1.0 clipping suspects).
3. **Transcripts exact.** Every substitution in the transcript becomes a
   mispronunciation in the voice. Whisper-drafted transcripts must be
   hand-checked (rmcpantoja notebook lesson).
4. **Inference tuning BEFORE retraining.** `noise_scale` / `noise_w` in the
   exported `.onnx.json` are read at synthesis — no retrain, no re-export.
   Measured on `me_omni_piper_sk` (same sentence): HNR -11.1 (ns 1.0) →
   -9.9 (0.667) → -8.9 (0.3) → -7.4 (0.1), voiced fraction 4% → 18%.
   Lower = steadier but flatter. Render the A/B set, pick by ear in the Lab,
   then bake the winner into the JSON. Only retrain if no setting passes.
5. **Warmstart match.** Same sample rate + quality is required; language may
   differ, but closer phoneme inventory = fewer epochs. jirka(CS)→SK worked
   for intelligibility; residual accent/prosody is what extra epochs buy.

## 2. The run itself (so it finishes while you sleep once, not three nights)

- **Resume, never restart.** Checkpoints are cumulative: resume from the
  newest `*.ckpt` (our export script already selects newest-first after the
  last.ckpt/last-v1.ckpt incident). Keep top-k + always keep `last`.
- **Batch by VRAM, not hope.** Official: batch 32 + max-phoneme-ids 400 for
  24 GB. CPU Mac: batch 8 fits, ~6.7 s/step measured. Larger batch =
  fewer steps per epoch = fewer epochs per night; prefer the biggest batch
  that fits over more steps.
- **Validation split small but nonzero when affordable** (0.01 + a few test
  examples) so TensorBoard renders sample audio mid-run; on tiny corpora 0/0
  is accepted practice (official guide says so).
- **Export mid-run checkpoints.** Export every ~250 epochs worth of steps and A/B
  in the Lab — export is minutes, training is hours. Stop when ear + HNR
  plateau together, not on step count alone.
- **Stop rule (both must hold):** (a) `loss_disc_all` flat for ~100 epochs,
  (b) two consecutive checkpoint exports indistinguishable by ear.
  Step floor for fine-tune: ~1000 epochs equivalent (≈10k steps at batch 8
  on 50 clips). Our 2500-step runs are scouting runs, not voices.

## 3. The fans (concrete options, no martyrdom)

Mac CPU Torch saturates all performance cores — that is the correct behavior
for throughput, and the machine is designed for it, but three serial nights
of it is a planning failure, not a badge.

| Option | Cost | What changes |
|---|---|---|
| Biggest batch that fits + resume chain | free | fewer, longer nights; machine usable by day |
| `nice`/thread cap (e.g. `OMP_NUM_THREADS`, torch threads) for day runs | free | slower steps, quiet fans, usable desktop |
| Rented CUDA (RTX 3060-class) | ~$3–5 for the whole 15 GPU-h plan (measured math in `compute_capacity_report_2026-09.md`) | 2500 steps in ~1–1.5 h; full 10k-step run in one evening; no fan noise at home |
| University server | free if granted | same as rented + no third-party data transfer |
| Checkpoint handoff (LewisSmallwood pattern) | free | pause on Mac, resume on any CUDA box mid-run — no step lost |

Rule: no third blind overnight on the Mac. Night runs happen on CUDA (own,
loaned, or rented) or not at all.

## 4. Maintenance risks (do not upgrade blindly)

- Upstream `rhasspy/piper` was archived Oct 2025 (community CPU guide). The
  living training fork is `OHF-voice/piper1-gpl` (LightningCLI `piper.train fit`).
  Our `.venv-train` is pinned and working — treat it as frozen until a
  feature forces a move.
- ONNX export has a checkpoint race (Lightning renames top-1 mid-run);
  always copy the checkpoint to a stable path before exporting.
- `inference.noise_scale` in `.onnx.json` is deployment configuration, not
  model quality — record the shipped value next to the voice (Lab QC card).

## Sources

- rhasspy/piper TRAINING.md (epoch targets, loss_disc_all rule, batch/VRAM guide)
- OHF-voice/piper1-gpl docs (`piper.train fit` args, espeak_voice phonemization)
- LewisSmallwood/piper-tts-training (resume handoff, export race, inference.noise_scale)
- jrnwilliams/piper-voice-training-guide (CPU time table: ~6 h per 100 epochs; archive notice)
- veralvx/piper-train (val-loss monitoring variant, 3200-epoch envelope)
- kiarashQ/fa-ir-tts-piper (20-epoch GPU fine-tune sufficiency on large clean data — data quality beats step count)
- VITS2 (arXiv 2307.16430), FNH-TTS (arXiv 2508.12001), Interspeech 2024 Mehta (duration modeling is the hard part)
- Measured here: `processed/overnight_hq_2026-09-30.log`, `verification_omni_piper_sk_2026-09-30.md`, noise sweep §1.4
