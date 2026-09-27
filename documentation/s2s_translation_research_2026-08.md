# S2S / Speculative Translation Research — 2026-08-18

Scope: evaluate whether a joint speech-to-speech (S2S) model can replace the STT→MT→TTS
pipeline for this project, and whether speculative/predictive decoding is a viable
thesis contribution. One hands-on smoke test performed on this machine (M1 Pro, 16GB,
no CUDA). Existing project docs (`PERFORMANCE_TEST_RESULTS.md`, `COQUI_TTS_PERFORMANCE_REPORT.md`,
`requirements_notes.md`) were read first — this doc extends them, not repeats them.

## 1. Kyutai Hibiki — Slovak support: NO, and it's narrower than the initial search suggested

Confirmed directly from the `kyutai-labs/hibiki` GitHub README and the `hibiki-1b-mlx-bf16`
model card:

> "Hibiki currently only supports French-to-English translation."

Not French↔English — **French-to-English only, one direction**. No Slovak, no Czech, no
English-as-source. This rules Hibiki out entirely for this project's actual language
pairs (EN→SK is the hard requirement; EN↔CZ is the secondary demonstrated pair). There
is no hybrid path here — Hibiki has zero language pairs this project needs, so "S2S for
supported pairs, existing pipeline for SK" collapses to "existing pipeline for
everything," at least until a broader S2S model ships.

`Hibiki-Zero` (Feb 2026, X→EN for FR/ES/PT/DE) was already ruled out earlier — it
requires an NVIDIA GPU (8GB+ VRAM), not available on this machine.

## 2. Hands-on smoke test — Hibiki 1B MLX, on this exact machine

Ran `kyutai/hibiki-1b-mlx-bf16` via `moshi_mlx` (the only Apple-Silicon-native S2S model
found) purely to get a real feasibility number for this class of architecture, since no
usable-language-pair model exists yet. Isolated venv (Python 3.11, not the BP project's
own venv), nothing installed in the project itself.

**Setup:**
- `pip install -U moshi_mlx` — required Python 3.11 (the system default `python3` is
  3.14, which breaks `moshi_mlx`'s dependency resolution — a real, undocumented gotcha
  if this is revisited later).
- Model download: **3.7GB** on disk (`~/.cache/huggingface/hub/models--kyutai--hibiki-1b-mlx-bf16`).
- Test input: official Kyutai sample (`sample_fr_hibiki_crepes.mp3`, French, ~47.5s).

**Measured (not estimated), via `/usr/bin/time -l`:**

| Run | Wall time | Peak RSS | Notes |
|---|---|---|---|
| Cold (first run, includes model download) | 199.7s | 2.91GB | includes network fetch |
| Warm (model already cached) | 64.9s | 4.15GB | pure load + inference |

- Audio duration ≈ 47.5s, warm-run wall time 64.9s → **RTF ≈ 1.37** (slower than
  real-time) on M1 Pro CPU/MLX, single stream, offline batch mode (`run_inference`, not
  the streaming `local_web` server — true chunk-by-chunk streaming latency was **not**
  tested, that's an open unknown).
- Reported generation throughput: 11.0–11.3 tokens/sec against the model's native
  12.5Hz frame rate — running slightly under the pace needed to sustain real-time
  streaming even before accounting for I/O overhead.
- Peak RSS ~4.1–4.2GB **for one stream**. The project's own `PERFORMANCE_TEST_RESULTS.md`
  already shows the *current*, much lighter pipeline (faster-whisper + CTranslate2
  Opus-MT + Piper) saturates this same 16GB machine at 12 concurrent users via
  per-session model instantiation. A 4GB-per-stream model would hit that wall at
  roughly **2-3 concurrent users** — a large regression in the scalability this project
  already measured and documented as a requirement (+200 users target).

**Verdict:** proven, not assumed — Hibiki-class joint S2S does not fit this project's
hardware or language requirements today. Not "needs more tuning," structurally
mismatched: wrong language pair, sub-real-time on the only Apple-Silicon-native path,
and RAM footprint incompatible with the concurrency target already in this repo's own
test docs.

## 3. Hybrid architecture recommendation (per the decision to keep SK on the existing chain)

Given #1 and #2, the "hybrid" isn't S2S-for-some-pairs + old-pipeline-for-SK — it's
**keep the existing STT→MT→TTS pipeline for everything**, because no viable S2S
alternative exists yet for any pair this project uses. Effort should go into optimizing
what's already built and already measured, not swapping in an unready architecture.
Concretely, in order of effort vs. impact:

1. **STT: faster-whisper → mlx-whisper.** This project's own `requirements_notes.md`
   already flagged this in Nov 2024 and never acted on it. Current external
   benchmarks (Aug 2026) show mlx-whisper ~1.7x faster than faster-whisper on
   equivalent hardware (29.7x vs 17.3x realtime on the base model size, one public
   benchmark) because it uses native Metal instead of CTranslate2's CPU-only path.
   Lowest-effort, highest-certainty win available — swap one component, keep the rest
   of the pipeline untouched.
2. **TTS stays split, as already decided in this repo:** Piper (RTF 0.05, <1s) for
   speed-critical/generic-voice paths, Coqui XTTS v2 (RTF 1.72 CPU, officially supports
   Slovak per this project's own `coqui_tts_sections.md`) for voice-cloned SK/CZ output.
   This is already implemented, already tuned (7 debugging iterations on artifacts,
   documented), and already the right call — don't re-litigate it chasing S2S.
3. **Concurrency:** the existing docs already prescribe the fix (shared model pool
   instead of per-session instantiation) — do that before adding any new model to the
   stack, since every new model multiplies the same problem.

## 4. Speculative/predictive translation — worth prototyping, scoped correctly

SSBD (Self-Speculative Biased Decoding, arXiv 2509.21740, Sept 2025) fits this project
far better than trying to reproduce Hibiki: it's designed for exactly the
retranslation-style setup this project already has (re-running MT as new source text
arrives), reuses the previous output as a draft, and verifies in a single forward pass
against a lightweight bias — no separate draft model to train, no massive parallel
corpus, no GPU cluster. This is realistically implementable by one student as a
wrapper around the **existing CTranslate2 Opus-MT calls already in
`backend/mt/ctranslate2_mt.py`**, not a rewrite.

Recommendation: **prototype it**, scoped as a bachelor's-thesis-appropriate research
contribution — implement the draft-reuse + verify-from-divergence loop on top of the
existing Opus-MT backend, measure latency reduction and translation quality
(BLEU/SacreBLEU, already a dependency — `sacrebleu==2.5.1` is in `requirements.txt`) on
a handful of test utterances, cite Hibiki/SSBD as related work in the literature
review. Do not attempt to train or fine-tune a Hibiki-style joint model from scratch —
that needs training data and compute this project doesn't have access to, and #2 above
already shows the payoff wouldn't clear the bar even if achievable.

## 5. Contradiction found in existing docs, flagged for correction

`requirements_notes.md` (root) still describes MT as "SeamlessM4T v2 with UnitY2
architecture." Checked directly against `backend/main.py` (grep): the actual wired-in
MT is `CTranslate2MT` using Helsinki-NLP Opus-MT models
(`Helsinki-NLP/opus-mt-{src}-{tgt}`), consistent with the sub-second MT latency in
`PERFORMANCE_TEST_RESULTS.md` (avg 1.36s at 12 concurrent users) and
`COQUI_TTS_PERFORMANCE_REPORT.md` (0.23s single-user) — SeamlessM4T v2 would be far
slower than that. `requirements_notes.md` is stale/aspirational and should be updated
or removed to avoid confusing future-you or a thesis reviewer comparing code to docs.

## Open / unknown (not tested)

- Hibiki's true streaming latency (chunk-by-chunk, via `moshi_mlx.local_web`) — only
  offline batch inference was measured. If a future S2S model does add Slovak, this
  gap should be closed before trusting any RTF number for a live conference scenario.
  batch RTF is a ceiling for streaming latency, not a floor.
- mlx-whisper's actual WER/RTF specifically for Slovak and Czech audio on this
  project's own test clips — general benchmarks cited above are English-centric.
  Before committing to the STT swap, re-run this project's own `jiwer`-based WER
  evaluation (already in `requirements.txt`) against mlx-whisper using the existing
  `test/` audio + transcript pairs.
