# Compute capacity: what the current hardware costs us, what a CUDA box would buy

**Status:** living document — opened 2026-09-29, first refresh due 2026-10-05.
**Purpose:** support the hardware discussion with the thesis handler (possible
NVIDIA laptop and/or a university server). Every number is labelled
`MEASURED` (produced on this project's own hardware, with the file that proves
it) or `PROJECTED` (derived from a measured rate plus an architecture/price
argument). **No unlabelled numbers.** Constitution II: a projection is never
promoted to a measurement by repeating it.

## 1. The claim in one paragraph

The pipeline itself already runs inside the real-time budget on a 2020
CPU-only laptop — translation is not hardware-bound. What *is* hardware-bound
is **the development loop that produces the voice models**: a single Piper
fine-tune occupies the reference Mac for most of a working day, and the 16 GB
unified-memory ceiling forbids running an evaluation or an STT sweep at the
same time. A mid-range CUDA machine does not make the *product* faster; it
makes the *research* faster, and it is the difference between testing three
hypotheses a week and testing one.


## 2. Measured baseline (this project's own hardware)

### 2.1 Reference Mac — Apple M1 Pro, 16 GB unified, 8 cores, no discrete GPU

| Workload | Rate | Evidence |
|---|---|---|
| Piper VITS fine-tune (2500 steps, batch 8, 50 clips) | **0.07–0.08 it/s ≈ 13 s/step → ~8–9 h** | `MEASURED` 2026-09-29, `/tmp/overnight_hq.log` epoch timing, `scripts/finetune_personal_voice.py` |
| Same, on MPS instead of CPU | **0.04–0.07 it/s — slower than CPU** | `MEASURED` 2026-08, script docstring (VITS constant-padding ops leave the MPS fast path) |
| OmniVoice bulk corpus, 30 s clip | **RTF 0.93–1.22 (MPS, float32)** | `MEASURED` `processed/omni_hq_sk/manifest.json` (50 clips), `processed/omni_hq_en/manifest.json` (15 clips) |
| faster-whisper `base`, 115 s English clip | **0.52 s** (WER 0.077) | `MEASURED` `processed/stt_baseline.json` |
| Whisper `small-sk`, per Slovak sentence | **~3.0 s** | `MEASURED` `documentation/demo_report_2026-09.md` |
| Piper synthesis (fine-tuned, ONNX) | **RTF 0.0295–0.05** | `MEASURED` `documentation/personal_voice_bootstrap_2026-08-19.md` |
| `make test` (33 tests) | ~1–2 min | `MEASURED` 2026-09-28 merge gate |

### 2.2 Portability-proof laptop — Ryzen 5 8645HS, 6 cores, 14 GB, CPU only

| Workload | Rate | Evidence |
|---|---|---|
| EN → SK sentence, end to end | **~1 s** | `MEASURED` `documentation/demo_report_2026-09.md` |
| SK → EN sentence (Slovak-tuned `small`) | **~3 s** | `MEASURED` ibid. |
| SK → EN sentence (`large-v3-turbo`, old default) | **~9–11 s** | `MEASURED` ibid. |
| One-command cold setup | **~850 s** | `MEASURED` `AGENTS.md` (2026-09-28) |

**Read this table again before asking for hardware.** The *shipped product*
meets its target on hardware from 2020 with no GPU. The ask is about model
development throughput, and it should be framed that way to avoid the
"why do you need a GPU for something that already works?" objection.


## 3. Where the wall-clock actually goes

`MEASURED`, one working session 2026-09-29 (the overnight run):

| Phase | Wall time | Machine-bound? |
|---|---|---|
| EN bulk corpus, 15 clips | ~25 min (MPS) | yes (≈2× with CUDA) |
| SK corpus assembly + QC | < 1 min | no |
| **SK Piper fine-tune, 2500 steps** | **~8–9 h (CPU)** | **yes — and it is the whole cost** |
| Voice QC (WER thirds, F0, jitter, HNR) | ~2–3 min | mildly |
| Ear QC in Voice Lab | human-limited | no |

The train is ~95 % of the elapsed time and it barely uses the 8 cores. That is
the entire argument.

## 4. Projections

Method: take the measured CPU rate as the anchor, apply an architecture
multiplier, state the multiplier's basis. Where a multiplier is a community
norm rather than a measurement, it is marked as such.

### 4.1 Scenario A — mid-range NVIDIA laptop (32 GB RAM, RTX 4060 Laptop 8 GB)

| Workload | Now (M1 Pro 16 GB) | Projected | Multiplier basis |
|---|---|---|---|
| Piper 2500-step fine-tune | **~8–9 h** | **~1–1.5 h** | `PROJECTED` ~6× — VITS is ~30 M params and fits 8 GB VRAM; CUDA has no MPS padding fallback. Community norm, not measured here |
| OmniVoice bulk, 15–20 clips | ~25 min | **~10–15 min** | `PROJECTED` ~2× (CUDA vs unified MPS, float32) |
| Chatterbox AR synthesis | RTF 6.7 | **RTF ~1–2** | `PROJECTED` ~4× — autoregressive transformer is the case CUDA helps most |
| Whisper STT sweeps | serial | **parallel with training** | 32 GB removes the 16 GB contention that currently serialises every job |
| `make test` + live smoke | ~2–4 min | ~1–2 min | `PROJECTED` ~2× (faster-whisper on CUDA) |
| **Voice iteration cadence** | **~1 model/day** | **~4–6 models/day** | derived from the two rows above |

### 4.2 Scenario B — university server or rented CUDA (any modern NVIDIA card)

Same per-hour multipliers as A; the difference is operational, not
architectural:

| Item | Value | Label |
|---|---|---|
| One SK fine-tune, compute cost | ~$0.25–0.40 (RTX 3060 12 GB class, ~1.5 h) | `PROJECTED` from published hourly rates |
| The whole 2-week plan (≈5–6 trains + bulk + spikes ≈ 15 GPU-h) | **~$3–5** | `PROJECTED` |
| One-time setup | ~1–2 h: CUDA container, corpus upload, `.onnx` download | estimate |
| Data residency | voice clips leave the Mac | **policy constraint**, see §6 |

A university server is strictly better than renting *if access is granted*: no
per-hour cost, no upload of personal voice data to a third party, and
typically the fastest card available.

### 4.3 Scenario C — newer Apple Silicon (M4 Pro, 36–48 GB), for completeness

| Workload | Projected gain | Basis |
|---|---|---|
| Piper fine-tune | **~2×**, not 6× | `PROJECTED` — the MPS slowness is an op-support issue, not a generation issue, and CPU is the faster device here anyway |
| OmniVoice bulk | ~1.5× | `PROJECTED` |
| Memory ceiling | removed (36–48 GB) | architectural |

Apple wins on silence, battery and keeping voice data on-device; **CUDA wins
on price/performance for this specific stack**. If the choice is "one machine",
the NVIDIA laptop is the better research box and the Mac stays the demo box.


## 5. What this changes beyond training

1. **Hypothesis throughput.** Step-count ablations (2500 vs 5000 vs 10 000),
   base-voice comparisons (jirka vs a fresh SK base) and QC experiments each
   cost a working day today. At ~1.5 h each they become same-day — which is
   what turns a "we tried one setting" section of the thesis into a measured
   comparison.
2. **Parallelism.** 16 GB unified memory is the hard constraint; the runbook
   forbids running an STT spike during a train. 32 GB + 8 GB VRAM lets the
   eval matrix run while the trainer works.
3. **Unblocking the Chatterbox question.** Its current verdict is "quality ties
   OmniVoice, 5× slower" (RTF 6.7 on MPS). On CUDA that is testable; if the
   speedup is real, an MIT-licensed, openly trainable engine is strategically
   better than a zero-shot converter for the long term.
4. **Cold-start and portability testing** gets cheaper — more machines to test
   the one-command setup on.

## 6. Caveats and honest limits

- **The product does not need this.** A GPU changes research speed, not shipped
  latency. Any request must be framed as development capacity, not as a fix for
  a slow demo.
- **Local-first constraint.** The constitution forbids cloud services in the
  pipeline. Training on a remote machine is not the pipeline, but it does move
  personal voice recordings off the Mac — fine on a university server under the
  owner's control; worth a sentence of disclosure if it is a rented third-party
  box.
- **All Scenario A/B/C numbers are projections.** They are anchored on two
  measured rates (0.07–0.08 it/s CPU, RTF ~1.0–1.2 on MPS) plus community norms
  for CUDA. They must be re-labelled if ever measured, and thesis §5.2 may cite
  the measured rows only.
- **Voice quality will not improve from hardware.** More steps help; a faster
  box only lets us find out sooner.

## 7. Recommendation

1. Do not buy before the current overnight fine-tune is ear-QC'd. If
   `me_omni_piper_sk` scores well, the Mac has already proven sufficient for
   the thesis deliverable and the ask becomes about *research capacity*, not
   *capability*.
2. If a machine is offered, prefer **the university server** (no cost, no
   third-party data transfer), then the NVIDIA laptop as the fallback.
3. The first thing to run on new hardware is the **step-count ablation** — the
   cheapest experiment with the largest quality upside.

## 8. Refresh log

- 2026-09-29 — opened. Measured baseline captured from the overnight run;
  Scenarios A/B/C projected. Pending: completed 2500-step wall time and final
  `val_mel`, ear-QC verdict for `me_omni_piper_sk`, and a sweep of new model
  releases (STT/MT/TTS) for the 2-week report.
