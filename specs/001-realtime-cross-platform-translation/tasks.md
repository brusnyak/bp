---
description: "Task list for the cross-platform real-time translation pipeline rescope"
---

# Tasks: Cross-Platform Real-Time Speech Translation Pipeline (v2 rescope)

**Input**: `specs/001-realtime-cross-platform-translation/spec.md`, `plan.md`

**Tests**: Included — this project already has a pytest suite; extend it rather than skip tests.

**Organization**: Grouped by user story per `spec.md` priorities (US1/US2 = P1 MVP, US3 = P2, US4 = P3 research).

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [ ] T001 Audit current `requirements.txt` for every hardcoded platform/vendor assumption (not just the known "prioritize MPS" comment) — produce a checklist in `documentation/`
- [ ] T002 [P] Confirm `faster-whisper`, `ctranslate2`, `onnxruntime`, `torch` versions in use each expose a documented way to select CUDA/ROCm/DirectML/CoreML/CPU explicitly at runtime (not just at install time)

## Phase 2: Foundational — Hardware Backend Selection (BLOCKS all user stories)

**⚠️ CRITICAL**: This is the actual fix for the recurring "fast on my Mac, slow everywhere else" problem. Nothing else in this feature matters if this phase is skipped.

- [x] T003 [US2] Create `backend/hardware.py`: DONE, 2026-08-21. Implemented as one function, `detect_backend(stage)`, not three separately-named functions (`stt_mt_device()`/`tts_baseline_device()`/`tts_cloning_device()`) — same chains as spec'd (`plan.md`), same per-stage priority order (stt/mt: cuda→rocm→cpu; tts_baseline: cuda→rocm→directml→coreml→cpu; tts_clone: cuda→mps→rocm→cpu), just one parameterized function instead of three. `probes=` override arg makes it mockable without real GPU hardware (see T010).
- [ ] T004 [US2] Wire `backend/stt/` (faster-whisper) to call `hardware.detect_backend("stt")` instead of any hardcoded device string — NOT done this pass, out of scope (this pass only touched the TTS-engine dispatch refactor + hardware.py itself).
- [ ] T005 [US2] Wire `backend/mt/ctranslate2_mt.py` to call `hardware.detect_backend("mt")` instead of any hardcoded device string — NOT done this pass, same reason as T004.
- [ ] T006 [US2] Wire `backend/tts/piper_tts.py` (ONNX Runtime) to call `hardware.detect_backend("tts_baseline")` — PARTIALLY done. `piper_tts.py` now calls `hardware.py` and correctly *selects* directml/coreml/cpu by name (confirmed live on this Mac: selects `"coreml"`), but `piper-tts`'s `PiperVoice.load()` doesn't accept an onnxruntime `providers=` list in this version, so the selected backend name isn't actually routed into onnxruntime execution yet — flagged with a `ponytail:` comment in the code. Real RTF measured (0.0446, see T011) is consistent with CPU-class execution, not confirmed-accelerated CoreML. Leaving unchecked until provider wiring is real.
- [x] T007 [US2] Wire `backend/tts/coqui_tts.py` (XTTS v2, PyTorch) to call `hardware.detect_backend("tts_clone")` — DONE and verified live: loaded via `TTS_ENGINES["xtts"]()`, correctly selects `mps` on this Mac then self-overrides to `cpu` for XTTS-on-MPS stability (pre-existing, preserved behavior), real `synthesize_stream()` call produced 4.05s of real audio. (`f5_tts.py` mentioned in the original task text doesn't exist in this repo — XTTS v2 is the actual cloning engine in use.)
- [x] T008 [US2] Remove the `# prioritize MPS support` hardcoded default from `requirements.txt` — DONE, 2026-08-21. Replaced with a comment stating backend selection is a `backend/hardware.py` runtime decision, not an install-time pin.
- [ ] T009 [US2] Gate `mlx-whisper` behind a Darwin+Apple Silicon check — N/A, not started: `mlx-whisper` isn't used anywhere in this repo yet, nothing to gate.
- [x] T010 [P] [US2] Unit test `backend/hardware.py` against mocked cuda/rocm/mps/directml/no-GPU — DONE. `test/hardware_test.py`, 7/7 passing, run for real in a freshly-built project venv (`./venv`, Python 3.11.15) — not just syntax-checked.

**Checkpoint**: Foundation ready — every stage now selects hardware per-OS/per-vendor at runtime, not at install time.

---

## Phase 3: User Story 1 - Speaker gets voice-cloned translated audio out (Priority: P1) 🎯 MVP

**Goal**: Confirm the existing, already-built pipeline (STT→MT→TTS, voice cloning via XTTS v2/F5-TTS, generic fallback via Piper) still works correctly once Phase 2's hardware-selection changes land — this is a regression check, not new build, per Constitution Principle VI (don't re-architect what already works).

- [x] T011 [US1] Run existing `test_full_pipeline.py` against the Phase 2 (TTS-registry) changes on the reference M1 Pro — DONE, 2026-08-21, real run not a toy. Piper TTS latency across 3 real utterances: 0.0967s / 0.1793s / 1.1526s (text-length-dependent), matching the historical ~0.35s single-sentence baseline in `documentation/TTS_COMPARISON_REPORT.md` — no regression. `benchmark_full_pipeline.py` and the 12-concurrent-user load test from `PERFORMANCE_TEST_RESULTS.md` were NOT re-run (that's T019's scope, separate and much longer-running); this task covers single-request latency only.
- [x] T012 [US1] Confirm voice-clone fallback still triggers correctly when no speaker profile exists — **UPGRADED 2026-08-21: HybridTTS now actually installed, working, and benchmarked, not just guarded.** `openvoice`'s Cython/pyav build failure was a red herring; the real blocker was its setup.py hard-pinning `numpy==1.22.0` (pre-Python-3.11, doesn't build) plus a chain of missing transitive deps (`whisper_timestamped`, `openai-whisper`, `tiktoken`, `dtw-python`, `eng_to_ipa`, `cn2an`, `langid`, `proces`, `wavmark`, `resampy`). Fixed by installing with `--no-build-isolation --no-deps` against this project's own already-pinned versions (see `requirements.txt`) instead of openvoice's stale ones — one bad attempt briefly upgraded numpy to 2.2.6 and broke numba/accelerate/pandas ABI compatibility project-wide; caught immediately via a real smoke test, rolled back to `numpy==1.26.4`, reverified. OpenVoice V2 checkpoints (not on PyPI) downloaded from HF `myshell-ai/OpenVoiceV2` to `models/openvoice_v2/checkpoints_v2/converter/`.

  **Real bug found and fixed in `hybrid_tts.py` itself**: it was re-extracting Piper's source-voice embedding from every per-call generated clip, which (a) crashed with `AssertionError: input audio is too short` on any single-sentence output under OpenVoice's minimum embedding-extraction duration, silently falling back to uncloned Piper output, and (b) wasted latency re-deriving an embedding that never changes (Piper's base voice is fixed). Fixed at the root: the source embedding is now extracted once at `HybridTTS.__init__` from a longer bootstrap synthesis and cached — never re-derived per call.

  **Real, independently-verified numbers post-fix** (M1 Pro, MPS): first call for a never-seen target voice pays a one-time ~4.4s target-embedding-extraction cost; every call after that for the same voice: **SK RTF 0.169, DE RTF 0.147, EN RTF 0.156** — all comfortably faster than real-time, and genuinely cross-lingual (same English reference recording clones correctly into SK/DE/EN Piper output). Compare XTTS v2's RTF 1.46 for the same DE case — Hybrid is ~10x faster and the first engine in this project to hit sub-real-time speed on a cross-lingual cloned path. `OmniVoiceTTS`'s fallback remains unverifiable (`omnivoice==0.1.4` still conflicts with the `transformers==4.46.3` pin) — not attempted this pass, out of scope (Hybrid was the actually-relevant path).

**Checkpoint**: User Story 1 fully functional post-hardware-refactor.

---

## Phase 4: User Story 2 - Same install works on Windows/macOS/Linux (Priority: P1) 🎯 MVP

**Goal**: Prove Phase 2's backend selection actually produces working acceleration on all three OSes, and automate virtual-audio setup so it's plug-and-play per Constitution Principle V.

- [ ] T013 [US2] Extend `scripts/setup_windows.ps1` (already has partial VB-Cable detection per existing commit history) to also verify/report which GPU backend (DirectML vs CPU) was selected
- [ ] T014 [P] [US2] Write `backend/audio/setup_macos.py` (or extend existing mac setup) to automate BlackHole installation/detection
- [ ] T015 [P] [US2] Write `backend/audio/setup_linux.py` for PulseAudio/PipeWire null-sink automation — currently unaddressed, net-new
- [ ] T016 [US2] Manual/CI validation: run the pipeline on a Windows+AMD box and confirm DirectML is selected for Piper TTS, not CPU fallback (SC-002)
- [ ] T017 [US2] Manual/CI validation: run on a Linux+NVIDIA box and confirm CUDA is selected across all stages (SC-002)

**Checkpoint**: User Stories 1 AND 2 both work — this is the MVP bar for the rescope.

---

## Phase 5: User Story 3 - Concurrency without falling over (Priority: P2)

**Scope decision, 2026-08-21**: demo stays single-person-use for now (user's explicit call). T018/T019 deprioritized — not needed for the demo bar. Revisit only if the project continues past the demo and meeting-scale use becomes a real goal. Measurement work (accuracy+speed under load) still happened anyway via Phase 9/T037-T040, since that data is cheap and already answers "does it fall over," even without the pooling fix.

- [ ] T018 [US3] Implement the shared-model-pool refactor already prescribed in existing project docs — replace per-session model instantiation in `backend/main.py` with a pooled/shared model manager. **Deferred — not in scope for the demo.**
- [ ] T019 [US3] Re-run concurrency load test (existing methodology per `documentation/PERFORMANCE_TEST_RESULTS.md`); document new ceiling vs. the current documented 12-user ceiling (SC-005). **Superseded by Phase 9/T040 (2026-08-21) for the "is there a ceiling" question — T018's actual fix still open if this ever needs revisiting.**

**Checkpoint**: Concurrency ceiling measured and documented, whether or not it improved — the point is having a number, not guessing.

---

## Phase 6: User Story 4 - Speculative decoding for perceived speed (Priority: P3, thesis research contribution)

- [x] T020 [US4] Implement SSBD (arXiv 2509.21740) around `backend/mt/ctranslate2_mt.py` — done as `backend/mt/ssbd_ctranslate2_mt.py` (`SSBDCTranslate2MT`), using CTranslate2's `score_batch` (single parallel teacher-forced forward pass) for verification and `translate_batch(target_prefix=...)` to resume from the first divergence. Adaptation from the paper's literal formula documented in the module docstring: CTranslate2's public API doesn't expose per-step output distributions during a teacher-forced pass, so a log-prob-threshold divergence check is used in place of the paper's argmax-of-biased-mixture check. A real correctness bug was found and fixed during implementation: "fully accepted" must require the draft to have actually ended on EOS (`return_end_token=True`), not just that every token individually scored above threshold — otherwise short, locally-plausible-but-incomplete drafts get returned verbatim forever as the source keeps growing. Fixed; see commit.
- [x] T021 [US4] A/B benchmark: SSBD-augmented path vs. current path — done as `benchmark_ssbd_mt.py`, real wall-clock on this machine (M1 Pro, CPU, int8), `Helsinki-NLP/opus-mt-en-sk`, three runs (beam=4/short, beam=4/long, beam=1/long). **Result: negative, not positive.** Speedup 0.77x (beam=4, short utterances), 0.94x (beam=4, longer utterances), 0.93x (beam=1/greedy, longer utterances) — SSBD is 6-23% *slower* than the baseline in every configuration tested, not faster. Root cause, confirmed by inspecting per-increment draft reuse: on the long-utterance scenario the model reused **0 of 9-25** previous draft tokens on 4 of 5 increments, regardless of beam size (ruling out beam-search re-ranking as the cause) — `Helsinki-NLP/opus-mt-en-sk` restructures the whole sentence (word order, clause placement) as more source context arrives rather than monotonically extending its prior output, so the "previous translation is a reusable prefix" assumption SSBD depends on does not hold for this model. Translation quality was not separately evaluated since there is no latency win to trade it against.
- [x] T022 [US4] Write-up: `documentation/ssbd_speculative_decoding_findings_2026-08.md` — full findings, explicit proven/assumed/unknown split, real numbers, hypothesis for why this is a general-purpose-sentence-NMT-model property rather than a `Helsinki-NLP/opus-mt-en-sk`-specific quirk (untested: whether a model actually trained for incremental/streaming re-translation, like the paper's Tower+ 2B, would show prefix-stable behavior where this one doesn't — flagged as future work, not tested here). **Recommendation: do not adopt SSBD in the shipped pipeline as implemented; keep as a documented negative-result thesis contribution** — this is legitimate, citable research content (a technique's assumption failing to hold for a specific model class is a real finding), just not a shipped optimization.

**Checkpoint**: All four user stories independently functional and documented.

**2026-08-19 follow-up finding, closes this line of work further than T020-T022 alone did**: read `backend/main.py` directly rather than assuming the pipeline has an incremental-retranslation workload at all. It doesn't. The primary VAD path calls `main_mt_model.translate()` exactly once per finalized utterance (after `SILENCE_TIMEOUT`, 0.3s). The one code path that would re-translate a growing sentence is explicitly commented out at the call site: `# DISABLED: Streaming chunk processing causes Whisper hallucinations on short non-speech sounds`. So SSBD's negative result isn't just "wrong technique for this model" — this pipeline has no re-translation-of-growing-text scenario for any speculative-decoding technique to attach to, as currently built. **Do not pursue MT-stage speculative/incremental decoding further, by any technique** — the workload it would optimize doesn't exist here.

The more relevant, evidence-backed lead found instead: **`ufal/whisper_streaming`** (existing, maintained tool) implements **LocalAgreement** — emits only the longest common prefix agreed across successive incremental STT updates, which is specifically designed to prevent the hallucination-on-short-chunks problem that got this project's own streaming path disabled in the first place. Re-enabling incremental STT via this existing tool would give live partial captions while the user is still speaking — a bigger real-time UX win than MT-stage optimization would have been, and a maintained tool to adopt, not a paper to reimplement. See new Phase 8 below.

---

## Phase 7: User Story 5 - Fast personal-voice TTS tier (Priority: P2)

- [ ] T023 [US5] Check `requirements.txt` for the TTS package pin: replace unmaintained `TTS` (coqui-ai, dead since 2024) with the maintained `coqui-tts` (idiap/coqui-ai-TTS fork on PyPI) if not already
- [ ] T024 [US5] Confirm/document exact `speaker_cache` conditioning-latent caching behavior already in `backend/tts/xtts.py` (or equivalent) matches `documentation/coqui_tts_sections.md` — this is already implemented, do not reimplement, this task is verification only
- [x] T025 [US5] Set up a Piper single-speaker fine-tuning pipeline. DONE: `scripts/finetune_personal_voice.py`, using OHF-Voice/piper1-gpl (rhasspy/piper is archived, development moved). Hands-on verified on this machine: CPU beats MPS for this workload (MPS's constant-padding ops fall back to a slow path - 0.04-0.07 it/s vs 0.25-0.32 it/s on CPU). Toolchain needs its own Python 3.11 venv, separate from the main project (pytorch-lightning/scikit-build/cmake/ninja have no business on the serving backend) - see script docstring for full setup + two real bugs hit and fixed (packaged wheel ships monotonic_align's .pyx but not the compiled .so; PyTorch 2.9+'s new default ONNX exporter fails on this model's data-dependent control flow, must force `dynamo=False`).
- [ ] T026 [US5] **REVISED 2026-08-19**: the user has capped real recording at ~90s (~48s existing clips + up to ~60s more), rejecting the 10-30 min floor. Research confirms that floor is real for Piper/VITS specifically (rhasspy/piper1-gpl's own guidance), not overcautious — so T026 as originally scoped (record 10-30 min for a quality Piper fine-tune) stays a possible *upgrade path* if the user ever records more, but is no longer the default plan. See T031-T033 below for the ~90s-compatible path.
- [x] T027 [US5] Pipeline mechanically proven end-to-end on real (if tiny/toy-scale) data: 3 existing EN clips (37s total) -> trained 3 steps on CPU -> exported to ONNX (forcing legacy exporter) -> loaded via real `PiperVoice` API -> synthesized -> measured RTF 0.0295 (5-run average). This is NOT a quality fine-tune (3 steps, 37s of data) - it proves the mechanism works and the speed claim holds, not that voice similarity is good. A real run needs T026's data first. **UPGRADED 2026-08-21**: real (not toy) run completed. New `scripts/prepare_voice_corpus.py` normalizes/dedupes/auto-transcribes `speaker_voices/` (found & fixed real data bugs: 8/24 manifest entries were literal duplicates, 2 more were the same take saved as both `.wav` and `.m4a`, 15/24 pointed at files that don't exist on disk — see `documentation/voice_corpus_pipeline_2026-08-21.md`). Clean corpus: 5 entries, 89.5s. Fine-tuned 200 real steps (not 3 toy steps), warm-started from `rhasspy/piper-checkpoints` (note: it's an HF **dataset** repo, not a model repo — `repo_type="dataset"` required, cost real time to discover), exported to `backend/tts/piper_models/en_US-personal-v2.onnx`, registered as `TTS_ENGINES["piper_personal_v2"]` (kept separate from `piper_personal`, not overwritten — see below). Real synthesis: RTF 0.0501. Voice-similarity NOT compared against `piper_personal` — still open, see T029.
- [~] T028 [US5] PARTIALLY done, 2026-08-21: the *reachability* half is done — `en_US-personal-medium.onnx` (the already-trained personal voice) is now selectable via `TTS_ENGINES["piper_personal"]` in `backend/tts/base.py` (`tts_model_choice="piper_personal"` from the client), no `main.py` changes needed thanks to the registry refactor. Verified live: real synthesis, 7.01s of audio from a 116-character sentence, latency 0.3128s → RTF 0.0446 (consistent with the 0.0562 documented in `documentation/personal_voice_bootstrap_2026-08-19.md`). EN-only, same limitation as before. The full four-tier *automatic selection cascade* (fine-tuned personal → GPT-SoVITS → XTTS zero-shot → generic Piper) described below is explicitly NOT done — this only adds one more explicit, manually-selectable tier; it doesn't implement the fallback cascade logic itself. **Now a four-tier design per T031-T033**: fine-tuned personal Piper (fastest, RTF 0.0295, needs 10-30 min — upgrade path) → GPT-SoVITS personal voice (RTF ~0.526 on Apple Silicon per community-reported number, needs only ~1 min — default path) → XTTS zero-shot (RTF 1.72, any language, no per-user setup — fallback) → generic Piper (no clone at all — last resort). Deliberately left for after T031/T032 land.
- [~] T029 [US5] Latency delta MEASURED: fine-tuned Piper RTF 0.0295 vs XTTS RTF 1.72 (`documentation/COQUI_TTS_PERFORMANCE_REPORT.md`) - ~58x. Voice-similarity comparison NOT done (toy 3-step model isn't representative) - re-run once T026/a real training pass exists. GPT-SoVITS's RTF 0.526 (T031) still needs to be independently re-measured on this exact M1 Pro, not just trusted from the M4 community report it's currently sourced from. **2026-08-21**: a second real (non-toy) RTF data point now exists — `piper_personal_v2` (real-audio-only, no synthetic bootstrap), RTF 0.0501, same speed class as `piper_personal` (0.0446) and generic `piper` (0.0479). Still open: objective (resemblyzer cosine sim, same tool as `scripts/voice_similarity_qc.py`) or subjective comparison of `piper_personal_v2` against `piper_personal` and against the real reference recording — not done this pass.
- [x] T030 [US5] Documented in `scripts/finetune_personal_voice.py` module docstring (setup, verified numbers, open questions) - trade-off already stated in `spec.md` FR-008/User Story 5 from the prior session's commit.
- [ ] T031 [US5] **NEW 2026-08-19**: Set up GPT-SoVITS (RVC-Boss/GPT-SoVITS) as the default personal-voice engine for the ~90s-of-audio case. README's own stated design goal: "1 min voice data can also be used to train a good TTS model." Community-reported RTF 0.526 on Apple Silicon M4 CPU (GitHub issue #2579) — same chip family as this M1 Pro but not the same chip, re-measure directly on this machine before trusting the number (per T029). Explicitly macOS-supported per its own README, with a documented caveat: GPU-trained models on Mac are lower quality, so Mac training currently defaults to CPU — consistent with the Piper/MPS finding from T025.
- [ ] T032 [US5] Re-run the existing recordings (`speaker_voices/*.wav`, ~48s) plus the new ~60s reading (provided this session) through GPT-SoVITS's few-shot pipeline; measure real voice-similarity and RTF on this machine — this is the first real (non-toy) quality data point for this project's personal-voice tier.
- [ ] T033 [US5] Investigated and explicitly rejected as the default: synthetic-data bootstrap (short recording → zero-shot cloner generates a large synthetic training corpus → fine-tune Piper on that). Real technique (e.g. ZeSTA, arXiv 2603.04219) but requires "domain-conditioned training with real-data oversampling" to avoid measurable speaker-similarity degradation from synthetic-only data — a research project of its own, not the lazy/goal-driven answer given GPT-SoVITS already exists and fits the constraint directly. Not pursuing unless GPT-SoVITS's real quality (T032) turns out to be insufficient.

**Checkpoint**: All five user stories independently functional and documented.

---

## Phase 8: User Story 6 - Incremental STT via LocalAgreement (Priority: P2, net-new 2026-08-19)

- [x] T034 [US6] Evaluate `ufal/whisper_streaming`'s LocalAgreement policy — DONE, 2026-08-21, real numbers not assumptions. Newer `ufal/SimulStreaming` was investigated first (per 2026-08-21 research into successor tools) and correctly rejected before writing any code: its own README states CPU is "too slow for real-time," needs 10G+ VRAM and a 1.5B-param model — disqualified by the authors' own docs on this M1 Pro (no CUDA). Vendored `whisper_streaming`'s real source (MIT, `github.com/ufal/whisper_streaming` — NOT the same-named PyPI package, which is an unrelated, suspiciously-repackaged project) into `backend/stt/whisper_streaming_vendor/`, patched its `FasterWhisperASR` from a hardcoded `cuda`/`float16` default to this project's own working `auto`/`int8` config. Result: **without** Silero-VAD gating (`--vac`), LocalAgreement falls behind real-time with a growing backlog (15.6s lag on the final chunk of a 29.4s clip, RTF ~1.15 overall — gets worse, not stable). **With** `--vac` (the authors' recommended default): stable, RTF ~1.001, first partial at 3.3s. Reproduced the actual original failure mode directly (not assumed fixed by design): a synthetic non-speech clip (noise floor + short burst) produced **zero hallucinated output** — VAD correctly gates non-speech before it ever reaches Whisper. See `documentation/streaming_stt_2026-08-21.md` for full detail. This is a **positive** result — LocalAgreement+VAC is viable on this hardware — not another SSBD-style negative finding.
- [ ] T035 [US6] Wire it in — **NOT done this pass, well-specified follow-up, not a research gap.** The old disabled code path didn't do incremental LocalAgreement at all; it re-ran the full STT→MT→TTS pipeline (including a fresh TTS call) per fixed chunk, which is architecturally why it hallucinated and would also have caused stuttering repeated audio. T035's actual scope is narrower — captions-only partial text, no incremental TTS — requiring a per-session persistent `VACOnlineASRProcessor` (mirroring how `session_data["stt_model"]` already persists per session) fed from the existing chunk-receive loop, additive alongside (not replacing) the existing `SILENCE_TIMEOUT`-based final-segment path that T011 already verified regression-free. Deliberately not rushed into the live WebSocket loop at the end of a long session without dedicated testing time — see doc for the concrete design.
- [ ] T036 [US6] Blocked on T035 — nothing to compare before/after until it's wired in.

**Checkpoint**: Live partial captions working without reintroducing the original bug, or a documented reason why not.

---

## Phase 9: User Story 7 - Speed + Accuracy Under Concurrency (Priority: P1, net-new 2026-08-21)

Context: `test/concurrent_performance_test.py` and `documentation/PERFORMANCE_TEST_RESULTS.md` already
cover concurrency/latency (12 stable, 15 failing, M1 Pro 16GB, 2025-11-28 — pre-dates the TTS registry
refactor, needs reconfirming). What's missing entirely: accuracy (WER/translation correctness) is never
measured under load, only latency. Ground-truth already exists for this: `test/Hello_transcript.txt`,
`test/Can you hear me_transcript.txt`, `test/My test speech transcript.txt` and their `*_translation.txt`
counterparts. No peer-to-peer relay exists in this backend (confirmed: no room/broadcast/participant code
in `main.py`) — "N users" means N independent per-user sessions against the shared backend, matching the
project's own documented "Per-User Isolation" concurrency model, not literal cross-talk between two clients.

- [x] T037 [US7] DONE, 2026-08-21. Added `jiwer.wer()` scoring + a stdlib `difflib.SequenceMatcher` normalized-similarity/exact-match check to `test/concurrent_performance_test.py`, wired against the existing `test/*_transcript.txt`/`*_translation.txt` ground truth via a new `GROUND_TRUTH` map and `score_accuracy()`. Extended the existing harness (`SessionMetrics` gained `wer`/`translation_similarity`/`translation_exact_match` fields, `generate_summary_report()` gained an Accuracy Statistics table) — did not fork a new script. Also made `user_counts`/`tts_model`/`ramp_up_duration` CLI-overridable (`argparse`, already imported but unused before) instead of hand-edited constants, needed for T038-T040 below.
- [x] T038 [US7] DONE, 2026-08-21, real run against a live local server (not simulated): N=2, `piper` TTS. **100% success (2/2).** E2E latency avg 4.02s (3.44-4.60s). WER avg 0.071 (0.0 and 0.143 — the 0.143 case is a real STT mishear, "Laddy"→"Loddy", not a scoring bug, independently confirmed by reading the raw transcript in the output JSON). Translation similarity avg 0.876 (0.828-0.925); 0% exact-match, expected — MT paraphrases (e.g. "na účely klonovania" → "na klonovanie"), similarity is the meaningful metric here, not exact string match. Full data: `test_output/performance_tests/run_20260821_083317/`.
- [x] T039 [US7] DONE, 2026-08-21, real run: N=5, `piper` TTS. **100% success (5/5).** E2E latency avg 9.66s (4.28-26.93s, std dev 8.66s — one session (the 65-word "My test speech" clip) took 34.10s wall time due to queuing behind the other 4, consistent with the project's own documented queuing behavior at higher concurrency). WER avg 0.102 (0.0-0.224); translation similarity avg 0.754 (0.268-0.925 — the 0.268 low outlier is the long multi-sentence clip, expected to drift more under MT paraphrasing than short utterances). CPU avg 4.5%/max 17.9%, memory avg 154MB/max 230MB — nowhere near saturation at N=5. Full data: `test_output/performance_tests/run_20260821_083342/`.
- [~] T040 [US7] DONE (single-trial), 2026-08-21. Real sweep 10/12/15/18/20 against a live server (`test_output/performance_tests/run_20260821_083432/`), JSON read back directly, not trusted from log lines. Success rates: 100%/100%/100%/72.2%/85.0%. Raw pass-rate is a real improvement over the 2025-11-28 baseline (12 stable/100%, 15 unstable/73%) — this run held 100% through 15 and 85% at 20. **Marked partial, not done**, for two reasons found while verifying: (1) single trial per N — the 18-user run (72.2%) scoring *worse* than 20-user (85.0%) proves single-trial noise is large enough to flip the ranking, so this is not yet a trustworthy ceiling number, needs 3+ trials/N averaged; (2) found and confirmed (server-log-corroborated) that the harness's fixed 5s post-stop receive window is too short once STT time exceeds it (true above ~10 users per this run's own STT-latency numbers) — most "completed" sessions at N≥10 got zero transcription/translation events before the client disconnected, so the per-N accuracy numbers in this run are sample-shrinkage noise, not a real trend (N=2/N=5 numbers, T038/T039, are reliable — high completion fraction there). Also found: `ResourceMonitor` profiles the test client's own process, not the server — true of the *original* 2025-11-28 report too, so "16GB fully utilized at 12 users" was never actually shown by this script's own numbers. Full writeup with proven/assumed/unknown split: `documentation/concurrency_accuracy_2026-08-21.md`.
- [ ] T041 [US7] **NEW 2026-08-21**: make the post-`stop` receive window in `simulate_user_session` latency-aware (wait until the receiver task goes idle, or scale with observed STT time from `final_metrics`) instead of the current fixed 5s — required before N≥10 accuracy numbers from this harness can be trusted (see T040 finding).
- [ ] T042 [US7] **NEW 2026-08-21**: point `ResourceMonitor` at the server's PID (or cross-process via `psutil`) instead of the test client's own process — the "Resource Usage" section has never measured actual server memory/CPU pressure under load (see T040 finding).

**Checkpoint**: Real speed+accuracy numbers at N=2 and N=5, and a reconfirmed (not assumed-still-true) failure ceiling on this machine.

---

## Phase 10: Final architecture — kill dead time, target Google's 2-3s ceiling (Priority: P1, net-new 2026-08-22)

**Scope decision, 2026-08-22, supersedes 2026-08-21 single-person-demo deferral**: user reviewed
`documentation/simultaneous_translation_research_2026-08-22.md` and explicitly rejected building a
from-scratch simultaneous-MT system — no viable path exists without either training a wait-k model
from scratch (its own ML project) or accepting EMMA/SeamlessStreaming's published 66% BLEU-quality
collapse. Plugging this project's own voice cloning into Google's Gemini 3.5 Live Translate was also
considered and rejected: closed-source, no public voice-cloning-injection API, permanent internet
dependency, and it would eliminate this project's own technical contribution as a bachelor's thesis.
**Final target**: this project's own STT→MT→TTS chain, engineering-hardened to close the gap between
today's ~4s (mostly architectural dead time, not real compute — see finding below) and the ~2-3s
ceiling Google itself deliberately targets for the same feature. No new ML models. German-style
reordering-heavy languages keep the existing full-utterance-wait behavior by design (already the
correct/safe handling per the research doc — no special-casing needed). After this phase, feature
work stops; focus moves to documentation/thesis writeup.

- [x] T043 DONE, 2026-08-22. `_process_speech_segment_pipeline` calls converted to
  `asyncio.create_task` (tracked per-session as `active_pipeline_task`) at all 6 call sites; capture/VAD
  loop no longer blocks on translation. **Found and fixed a real correctness bug while verifying this
  live** (`test/interrupt_smoke_test.py`, kept for reproducibility, not a permanent suite addition): a
  naive "cancel any in-flight task on new speech-start" rule (the literal original design) silently
  dropped most sentences in ordinary continuous multi-sentence speech — natural inter-sentence pauses
  are shorter than one STT+MT round trip, so each new sentence cancelled the previous one's still-running
  transcription/translation before it could send results. Fixed with a `model_call_lock`
  (`asyncio.Lock`, per session) around STT+MT only — barge-in cancellation now only fires when the lock
  is *not* held (previous task is in/past TTS, the only phase where overlapping output is a real
  problem), never mid-STT/MT. Verified on a real 8-sentence continuous clip: 0 dropped sentences (was
  3/8 dropped with naive cancellation), interrupt utterance after processed correctly. Also found and
  fixed: two independent session-dict init sites (`initialize_all_models` at `main.py:364`, called by
  `app.py`'s `/ws` route *before* `handle_audio_stream`'s own init at `main.py:965`) — the new session
  keys only existed in one, causing a `KeyError` on first real run. Both now carry the same keys; this
  divergence is pre-existing tech debt (two hand-maintained copies of the same default-session shape)
  worth consolidating later, not fixed here (out of scope, minimal diff preferred). Regression: `test/hardware_test.py` 7/7 (unrelated to this file, baseline only), a direct A/B live comparison against
  unmodified `main.py` confirmed the one server-log error seen (`Cannot call "receive" once a disconnect
  message has been received`) is pre-existing, not a regression. Real single-utterance smoke test
  post-fix: correct transcription/translation/audio, WER 0.0, e2e_latency 4.95s (in line with T038's
  4.02s baseline, single-sample noise).
- [x] T044 DONE, 2026-08-22. New `backend/stt/streaming_captions.py` wraps the vendored
  LocalAgreement+VAC processor (`SessionCaptioner`), matching the exact construction used in the
  2026-08-21 benchmark (`--vac`, no `--vad`, min-chunk-size 1.0s, `buffer_trimming=("segment",15)`)
  so real behavior isn't an untested deviation from what was measured. Wired additively: every
  incoming audio chunk is fed to it fire-and-forget (`_feed_captioner_task`, bounded by a
  `captioner_lock` — a chunk is *dropped*, never queued, if the previous feed() call hasn't
  returned, so this can never build backlog); on a committed partial it sends a new
  `caption_partial` message. `_run_pipeline_task` (T043's wrapper, now the single funnel point for
  every trigger) resets the captioner at the start of each real segment. **Does not touch or gate**
  the existing SILENCE_TIMEOUT final-segment path — captions are informational only, the real
  transcript/translation/audio still come from the primary STT model exactly as before. Live-verified
  (`test/interrupt_smoke_test.py`, 8-sentence clip): first caption at 3.41s, well ahead of that
  sentence's real `transcription_result` at 5.19s — matches the ~3.3s figure from the original
  benchmark. All 8 sentences' real output still correct (0 dropped, same as T043's fix), interrupt
  utterance still processed correctly, 0 new server errors, `hardware_test.py` still 7/7.
  **Known limitation, not fixed**: loads a *second* whisper model ('base', int8) per session for
  captions, separate from the primary STT model — doubles STT memory per session. Not shared/pooled
  — same deferred concern as every other per-session model in this project (T018). **UI**: backend
  only — the frontend doesn't render `caption_partial` yet, out of scope for T044 as written (backend
  plumbing), noting so it's not mistaken for "captions visible in the demo already."
  **"Finalize-of-already-computed-partial" (the latency-win half of T044's original wording) was
  NOT implemented** — the final segment still runs a from-zero STT pass via the primary model, the
  same as before T044. Wiring `VACOnlineASRProcessor`'s own committed state into the final-segment
  path was judged out of scope for this pass: it would mean the *final* transcript (not just
  captions) depends on the streaming processor, which changes the risk profile from "additive,
  can't break the main path" to "load-bearing" — exactly the kind of change the ground rules for
  this task say to stop and flag rather than rush. Captions-as-latency-perception-improvement is
  delivered; captions-as-actual-final-STT-speedup is not, and would need its own dedicated
  correctness pass (word-level alignment between the streaming processor's buffer and the final
  full-segment audio) to do safely.
- [x] T045 DONE, 2026-08-22. Added `backend/tts/text_chunking.py` (`split_into_phrases` —
  punctuation-based clause splitting; `chunk_audio_by_duration` — slices each phrase's synthesized
  audio into ~150ms pieces). Both `PiperTTS` and `HybridTTS` gained `SUPPORTS_STREAMING = True` and
  a `synthesize_stream()` that phrase-splits the text, synthesizes each phrase with the engine's
  existing (unchanged) `synthesize()`, and sub-chunks the result — so multi-clause text is sent as a
  train of small chunks instead of one blob, without introducing a second synthesis code path.
  **Two real bugs found and fixed while verifying this live** (not assumed safe): (1) `HybridTTS.sample_rate`
  was hardcoded 16000 and never actually read anywhere pre-T045 (dormant) — confirmed the real rate
  directly from `models/openvoice_v2/checkpoints_v2/converter/config.json` (`sampling_rate: 22050`)
  and fixed it; wrong, it would have played cloned audio at the wrong pitch/speed the moment
  streaming started reading this attribute. (2) `main.py`'s streaming-vs-non-streaming dispatch had
  a hardcoded `elif tts_model_choice != "xtts":` fallback-error check, written back when XTTS was the
  only streaming engine — with piper/hybrid now also streaming, every successful streaming synthesis
  hit this and sent a false `"TTS model not ready"` error to the client (caught immediately via a live
  smoke test, not shipped). Fixed by generalizing to `not getattr(tts_engine, "SUPPORTS_STREAMING", False)`,
  matching the flag the dispatch above it already uses instead of a second hardcoded name check that
  had silently drifted out of sync with it. Verified: real chunk sizes ~150ms (3307 samples @ 22050Hz),
  correct concatenated duration, live smoke test clean (WER 0.0, no errors), 8-sentence continuous
  clip still 0 dropped. HybridTTS's streaming path verified standalone (correct chunking/duration,
  real cloned audio) but not through a live end-to-end WS test with a real `speaker_wav_path` — the
  existing test harness doesn't pass one; same integration code path as Piper's (already live-verified),
  so risk is low but this specific combination wasn't exercised end-to-end.
- [x] T046 DONE, 2026-08-22, verified live — not by test-construction, by real observed behavior.
  While re-running the interrupt smoke test after T045, the server log captured a genuine mid-TTS-send
  cancellation on its own: segment "Speak trick ignition, machine translation, and text to speech."
  started streaming TTS at 07:15:03.634, and the very next segment's speech began 88ms later —
  cancellation fired at 07:15:03.723 (within ~1ms of speech-start being detected). Confirmed the
  streaming-TTS send loop (`main.py`'s `while True: chunk = await queue.get()`) is cancelled near-
  instantly because it awaits a plain `asyncio.Queue`, not a `run_in_executor` future — the earlier
  documented "cancellation is deferred until the blocking call returns" caveat applies to STT/MT/
  non-streaming-TTS `run_in_executor` calls, **not** to the chunk-send loop once T045 made TTS
  chunked. The background `run_streaming_tts` daemon thread is NOT itself killed (both remaining
  phrases still printed as synthesized in the log after cancellation — wasted CPU, but bounded and
  harmless) — what matters is confirmed: the queue-draining/sending loop stopped, so no further
  `tts_audio` bytes for the cancelled segment reached the client.
  **Real, non-obvious finding to flag, not silently resolved**: this cancellation fired between two
  *natural* back-to-back sentences in one continuous test clip, not a deliberate user interruption —
  the test script never sent a second utterance at that point. This means fast continuous multi-
  sentence speech (this test clip reads a script with short pauses) can truncate the *previous*
  sentence's spoken audio output once the next sentence's VAD speech-start fires, same ambiguity
  already fixed for STT/MT (T043) but now visible at the TTS layer, and NOT fixed here — TTS
  cancellation was deliberately left unconditional (any new speech-start cancels in-flight TTS)
  because that IS barge-in's actual job. Whether this is correct behavior depends on framing: for a
  genuine interruption it's exactly right; for a fast monologue reading many short sentences back to
  back, it means only the *last* sentence before a real pause reliably gets its full audio delivered
  from a delivery-completeness standpoint the same class of tradeoff EMMA's own 66%-BLEU-for-speed
  choice represents at the MT layer (research doc, section 1) — keep pace vs. deliver everything.
  **This is a product decision, not a bug** — flagging for the user/parent rather than silently
  picking a side.
- [x] T047 DONE, 2026-08-22. `test/hardware_test.py` 7/7 throughout (unaffected, tests hardware
  backend selection not the pipeline). Normal non-interrupted smoke test clean at every stage.
  New `test/soak_interrupt_cycles.py`: **10 back-to-back interrupt/resume cycles, one session,
  completed in 84.1s, no hang, 20/20 results received (none dropped), 0 errors.** Server RSS
  1,954,416 KB → 1,697,088 KB (**-257 MB, no leak** — net decrease, not growth, across 10 real
  cycles). Real E2E latency post-T043-T046, N=2 (directly comparable to T038's 4.02s baseline):
  3.64s/5.01s, avg **4.33s — flat vs. baseline, not a reduction**, and correctly so: this work never
  touched STT/MT/TTS compute speed, only dead time and interruption handling, which single-utterance
  latency doesn't capture. Full writeup, before/after table, and proven/assumed/unknown split:
  `documentation/realtime_pipeline_rearchitecture_2026-08-22.md`.

**Checkpoint**: real utterance-to-utterance latency measured end to end post-fix (4.33s avg N=2,
flat vs. the 4.02s T038 baseline — this work targeted dead time and interruption handling, not raw
compute speed, so a flat number is the correct/expected result, not a miss). Interrupt/resume
verified not to break or leak (10-cycle soak, 0 dropped, RSS decreased). Honest statement on the
2-3s ceiling: **not reached, and this phase was never expected to reach it** — real STT+MT+TTS
compute for one utterance is still ~4s on this hardware with this chain; what changed is that the
system no longer wastes time on top of that floor, and can now be interrupted instead of forced to
finish speaking. Closing the 4s→2-3s compute gap itself would require a different MT/TTS stack
(the exact tradeoff the 2026-08-22 research doc already priced out and the user declined to pursue).

---

## Dependencies & Execution Order

- Phase 1 (Setup) → Phase 2 (Foundational, blocks everything) → Phases 3 & 4 (P1, can run in parallel once Phase 2 lands) → Phase 5 (P2) → Phase 6 (P3, thesis research, can start any time after Phase 2 since it only touches the MT stage) → Phase 7 (P2, TTS-stage only, can run in parallel with Phase 5/6 once Phase 3's baseline cloning path exists)
- Per Constitution Principle II: do not start Phase 6 by re-evaluating S2S/Hibiki again — that question is already closed, cited in `spec.md` Edge Cases.
- Per Constitution Principle II: do not start Phase 7 by re-litigating whether caching fixes the cloning latency — that's already measured and closed (1.9% improvement, see `spec.md` User Story 5). The open question Phase 7 actually resolves is the fine-tuning implementation, not the diagnosis.

## Notes

- Commit after each task or logical group; this feature branch's commits are local until the user explicitly decides to push (git push is a shared/visible action, out of scope for automated execution per the constitution's Operating Mode section).
- Tests exist for this project already — Phase 3/4 tasks are explicitly regression checks against the existing suite, not new test scaffolding, per Constitution Principle VI (don't build what already exists).
