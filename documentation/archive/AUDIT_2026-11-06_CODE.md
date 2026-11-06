# Code, security and self-host audit — 2026-11-06

## Verdict and scope

**REWORK before a public self-host release.** The best achievable first release is one qualified Czech/Slovak↔English route that preserves meaning, starts on a clean machine with explicit model provisioning, and stops bounded work reliably. Current code has source-loss, lifecycle, meeting delivery, packaging and security defects that must be separated from model quality. A larger model alone cannot correct dropped source audio.

Review: read-only source inspection, one new report only. No code fixes, installs, model downloads, GPU benchmark, network publication or deployment. Existing dirty worktree preserved. Audit targets the root app/backend; the `bp/` mirror is separately identified below. No claim of human translation/voice acceptance or commercial licence clearance.

## Executed evidence

1. `.venv/bin/python -m pytest tests/bp_streaming_fixes_test.py tests/bp_low_latency_profile_test.py -q -p no:cacheprovider` → **2 failed, 14 passed, 1.23s**. First failure sees every input sample but emits zero rather than two short utterances. Second failure is test-isolation: a local import of `TTS_ENGINES` bypasses the fake and actually constructs/warms local Piper. No claim that this second failure demonstrates a wrong-language production voice. Command was repeated once after the first tool return did not retain the process identifier; both runs were short.
2. `.venv/bin/python -I` AST-only inline probe (executed exact `local_stt_source` and `_Frames` definitions with a fake downloader; no app/model imports): missing STT passed `local_files_only=False`; `_Frames.add` twice then the actual legacy `.pop(0)` left **1 frame, 2 timestamps**, first timestamp still 0.02.
3. Same probe used `runpy.run_path('backend/registry.py')` (registration only, no probes/models): `target_tts_engine('piper','de'/'fr')` resolves names absent from `select`; OmniVoice default `select` raises PermissionError for unknown licence.
4. Byte comparison of root app/backend against `bp/`: **20 exact mirrors, 6 different mirrored files**. Differing: `backend/main.py`, `backend/pipeline.py`, `backend/registry.py`, `backend/streaming/pipeline.py`, `backend/stt/faster_whisper_stt.py`, `backend/tts/base.py`. `diff -u` inspected the STT/streaming/TTS-base differences.
5. `rg --files backend -g '*.py'` → 31 backend files: 28 first-party and 3 vendored. First-party review coverage listed below. Source findings are distinguished from runtime reproductions.

## Priority findings

### P1 — source loss and competing streaming implementations

**C1. Short complete utterances/tails disappear.** `backend/main.py:2476` drops every silence-closed tail shorter than 1.0s and substitutes the previous transcript if any. A fresh short utterance produces no STT/MT at all; an old utterance can be repeated. Executed test `test_ws_ingress_dispatches_all_samples_across_two_utterances` fails with 0 vs 2 dispatched utterances. It uses synthetic 20ms speech, so it proves the unconditional source-discard path rather than acceptable recognition of 20ms speech. Real short-word/negation examples still need evaluation. **Smallest fix:** never claim a short tail is already recognized without source-span coverage; pad/contextualize or buffer it and test meaningful short utterances plus exact-once source accounting.

**C2. The legacy 2s transcription branch competes with incremental mode and bypasses order/cancellation tracking.** `backend/main.py:2438` runs the streaming condition without checking `inc is None`; `:2449` pops frames without matching timestamps; `:2462` directly creates `_process_speech_segment_pipeline`, bypassing `_run_pipeline_task` sequence/task bookkeeping. `_incremental_step` concurrently snapshots and trims that same frames object (`:1980` onward). Executed `_Frames` probe verifies frame/timestamp misalignment. **Smallest fix:** one source-buffer owner and one scheduled translation path; retire or explicitly gate the competing branch and require exact-once span/order/Stop regressions. Avoid irreversible speech from unstable or contextless fragments.

**C3. Translation units throw away semantic context.** `backend/main.py:1750` and `backend/streaming/pipeline.py:281` translate each fragment independently; `backend/stt/incremental.py:39` flushes by word count/punctuation/time. StreamingPipeline uses 1s minimum audio and 0.7s hypothesis cadence without previous-text prompt. The regular VAD path appends only speech frames (`main.py:2394`), collapsing pauses. These are source-proven architectural properties; how much each contributes to observed UI errors is **unknown** without paired audio/reference traces. **Smallest experiment:** same fixed recording, oracle transcript→MT, current STT→MT, and full-sentence versus current fragment MT; grade omissions, negation, entities, numbers and latency. Keep unspoken target hypotheses revisable; voice only stable clauses. No generic model swap recommendation substitutes for that comparison.

### P1 — meeting endpoint does not meet its advertised lifecycle/delivery contract

**C4. Listener delivery is unfinished and unbounded.** `backend/session_manager.py:165` creates unbounded queues; `:175` enqueues results but `:188` broadcast start is a `pass`. `app.py:499` sends audio back to the speaker, while listener handler only waits for incoming messages. No code drains listener queues. **Smallest fix:** keep meeting UI unavailable until there is a bounded per-listener sender task with disconnect cleanup, target routing and real two-client tests. Room capacity numbers are heuristics, not demonstrated service capacity.

**C5. Meeting Stop/start gate is bypassed.** `app.py:477` calls `session.stop`, but `:488` audio handling invokes the separate pipeline regardless of `capture_active`/generation; `SpeakerSession.process_audio_chunk` has the gate but this handler bypasses it. Processing is awaited in the receive loop, so Stop cannot even be received during inference. **Smallest fix:** route meeting audio through one capture generation/task controller, check Start before accepting audio, suppress stale emissions, and test Stop during STT/MT/TTS. Flush/timer support exists in StreamingPipeline but handler never schedules idle flush/final handling.

**C6. Meeting voice selection does not perform cloning.** `backend/streaming/pipeline.py:94` stores `voice` but `:304` always synthesizes with `speaker_wav_path=None`. No factory consumes that voice. **Smallest fix:** resolve and authorize a speaker reference explicitly at preparation, pin its hash/revision, pass it to synthesis, and assert actual selected voice reaches the engine. Do not equate this endpoint with the working standalone T4 clone benchmark.

**C7. Meetings have authentication without room authorization.** `app.py:375` publicly lists room IDs; `:379` lets any authenticated user delete any room; `:441` lets a caller choose/join any room. Guest tokens are enabled by default and meeting WS lacks the guest-cap/origin check used by `/ws`. No owner/member ACL exists in the room model. **Smallest fix:** owner/member or invitation authorization, guest policy and room quotas, or mark the endpoint local experimental and disable it on shared deployments. Cross-room read/delete tests required.

### P1 — security and offline promises differ from code

**C8. Public route inventory can fetch STT weights.** `backend/main.py:551` invokes `download_model(..., local_files_only=False)`; unauthenticated `/api/routes` (`:839`) calls it per pack. Its error claims no runtime download was started. `test/readiness_regression_test.py:207` is named local-cache-only but explicitly expects False. **Smallest fix:** `local_files_only=True`, test no network on inventory/preparation, and make all provisioning an explicit separate command. OmniVoice `from_pretrained` (`backend/tts/omni_tts.py:50`) also lacks an explicit local-only revision-pinned loading contract. Optional NLLB/Coqui download on initialization as well.

**C9. Compose installs a known signing secret if the owner forgets one.** `docker-compose.yml:29` supplies `change-me-in-production`; `backend/utils/auth.py:21` accepts every nonempty env value. This negates the otherwise safer random fallback. Anyone who knows an existing account email can forge its HS256 token under the example default. **Smallest fix:** require a generated configured secret or fail startup on placeholder. Do not print secrets during the check.

**C10. The advertised non-loopback rate limiter is absent from runtime dependencies.** `app.py:16`/`backend/main.py:22` guard optional `slowapi`; `requirements.txt` does not include it. `rate_limit_enabled` can return True on exposed hosts, yet missing import silently turns decorators into no-ops. Open guest/register and expensive preparation then remain available without those limits. Model pool bounds also warn and proceed when all instances are used (`main.py:345`); HTTP `/initialize` keeps per-user sessions indefinitely. **Smallest fix:** install/verify an explicit limiter or reject shared mode when unavailable, apply inference admission quotas and strict pool/session limits, then test rate-limit responses with the limiter actually enabled.

**C11. Audio content is logged despite default text-logging protection.** `backend/mt/ctranslate2_mt.py:144` logs source and translation at INFO; `backend/tts/piper_tts.py:137` prints synthesized text. Main's `_log_text` and opt-in JSONL do not cover engine-level logs. Optional engines also log content. **Smallest fix:** one redacted logging contract across engines, recording test logs containing a sentinel phrase and verifying it is absent with `BP_LOG_SESSIONS` off.

**C12. Private research metadata/reviews remain globally accessible.** `backend/main.py:1405` exposes Voice Lab QC candidates/scores without authentication; `/api/ratings` returns the global store to any authenticated user including a guest (`:1361`), and any nonguest may overwrite any existing rating (`:1368`). Hiding `/ui/voice-lab` does not hide those API paths. Severity depends on actual metadata, **not read here**. **Smallest fix:** hide research APIs behind the same private/operator policy, scope reviews by reviewer/asset or explicitly make an approved public export.

### P1/P2 — reproducible clean-clone deployment incomplete

**C13. Docker is a shell with no complete provisioning path.** `Dockerfile:30`–`:33` copies app/backend/languages/ui but not `scripts`; `/api/routes` imports `scripts.language_packs` and therefore fails in that image. `initialize_all_models` silently loses pack settings after catching the missing scripts import. Image creates empty model directories, Compose mounts CT2 read-only, and no setup/conversion runs. Piper voices are neither provisioned nor mounted. OmniVoice/PyTorch/CUDA extras are absent. Health checks hit hardware reporting rather than route readiness. Compose does not persist the SQLite account DB (`backend/utils/db_manager.py:35`). **Smallest fix:** explicit provision command/image stage plus pinned model manifest and mounts, include required runtime modules, persist DB, provide CPU and GPU profiles, and validate a clean image with one actual translated WAV under offline runtime. No Docker build was executed in this audit.

**C14. Quality regressions are outside normal gates.** `Makefile:23` and `.github/workflows/ci.yml:43` use explicit `test/` files and exclude both `tests/bp_*` modules run above; CI also omits files included by Makefile. Core VAD tests assert output existence, not semantic accuracy or exact source preservation. MT quality fixtures skip if personal test assets are absent (`test/mt_model_tests.py:19`). `test/conftest.py:75` deletes the shared `test_output/` directory after a run, so a full-suite run is not a strict read-only operation. **Smallest fix:** one authoritative suite list/marker policy, include deterministic source/lifecycle tests, move outputs to pytest temp dirs and add licensed portable accuracy fixtures. Green counts alone cannot accept a route.

**C15. Engine/config declarations diverge.** `target_tts_engine` resolves DE/FR voices that are present in TTS_ENGINES but missing from central registry (executed probe). Unknown targets silently select English voice (`backend/registry.py:200`). OmniVoice is unknown-licensed and fail-closed by default (`:343`), with no normal WS config path to override it; its class remains registered even when `OMNIVOICE_AVAILABLE=False`. RVC adapters are registered as TTS although `synthesize` raises NotImplementedError (`backend/tts/rvc_mlx.py:100`, `:209`). Hardware scan returns CUDA without VRAM before querying Torch (`hardware_scan.py:87`), so auto-choice requiring >=4GB never sees real VRAM; cached host config is reused without a machine fingerprint (`:301`). **Smallest fix:** distinguish implemented/installed/qualified, validate per-target languages and model assets, reconcile auto-config with installed pinned routes, and test every displayed choice without loading all models. Keep OmniVoice component evidence separate from routability/licence approval.

**C16. Cancellation, pooling and async contracts still permit growing work.** `main.py:1812` creates an unbounded streaming TTS queue and daemon producer; Stop cancels consumer tasks but producer continues every phrase. Disconnect only cancels the latest `active_pipeline_task` (`:2539`), not the complete task set. Task sequencing times out and allows later audio after 20s (`:1780`). Pooled VAD references increment (`:723`) but cleanup does not release VAD (`:412`); switching TTS does not release the old pooled reference. Model factories execute synchronously inside async initialization (`:351`), and SDK translate (`backend/pipeline.py:170`) performs native STT/MT/TTS synchronously on the event loop. **Smallest fix:** bounded per-session scheduler and producer cancellation token, strict admission, paired acquire/release and offloaded inference. Test Stop under pressure and repeated connect/config/disconnect with stable model/task counts. Native inference may finish after cancellation; suppressing results alone does not reclaim compute capacity.

## Architecture recommendation from the code evidence

A consolidation is justified; a platform rewrite is not yet justified. Three paths currently disagree: regular `/ws` including the incremental/legacy branches, `StreamingPipeline` for meetings, and the SDK facade. Best target: one route configuration, model lifecycle and inference scheduler, with transports as adapters. Preserve bounded audio/context and source timestamps; produce provisional caption/MT hypotheses; commit spoken clauses only when stable; track generation, ordered output and **seconds** of queued audio. Every emitted fragment must carry source-span and route/config provenance.

Before choosing another model, capture one representative continuous CZ and SK recording plus bilingual reference, compare oracle transcript MT vs ASR cascade, whole-clause vs current slicing and current vs candidate STT, and measure human meaning errors plus actual onset/end-to-listener delay. GPU component speed does not establish full-route accuracy. A useful release threshold must name listener/use case, error types and allowed queue growth; no universal chrF threshold is implied here.

## Read coverage and explicit exclusions

Read all 28 first-party root backend Python files, root `app.py`, setup/conversion/requirements/Docker/Compose/CI/Makefile, and the listed tests through bounded `cat`/`sed` calls. Large batched outputs were truncated; core MT/TTS and main portions were subsequently reread in bounded sections. Source coverage is inspection, not execution/branch coverage.

- `backend/hardware.py` — source read
- `backend/hardware_scan.py` — source read
- `backend/latency_profiles.py` — source read
- `backend/main.py` — source read
- `backend/mt/convert_opus_mt_to_ct2.py` — source read
- `backend/mt/ctranslate2_mt.py` — source read
- `backend/mt/hybrid_mt.py` — source read
- `backend/mt/nllb_mt.py` — source read
- `backend/mt/ssbd_ctranslate2_mt.py` — source read
- `backend/pipeline.py` — source read
- `backend/registry.py` — source read
- `backend/session_manager.py` — source read
- `backend/streaming/__init__.py` — source read
- `backend/streaming/pipeline.py` — source read
- `backend/stt/faster_whisper_stt.py` — source read
- `backend/stt/incremental.py` — source read
- `backend/stt/streaming_captions.py` — source read
- `backend/tts/base.py` — source read
- `backend/tts/coqui_tts.py` — source read
- `backend/tts/download_piper_models.py` — source read
- `backend/tts/hybrid_tts.py` — source read
- `backend/tts/omni_tts.py` — source read
- `backend/tts/piper_tts.py` — source read
- `backend/tts/rvc_mlx.py` — source read
- `backend/tts/text_chunking.py` — source read
- `backend/utils/audio_utils.py` — source read
- `backend/utils/auth.py` — source read
- `backend/utils/db_manager.py` — source read

Additional complete source reads: `app.py`; `AGENTS.md`; `PRODUCT.md`; `documentation/implementation-2026-11-04/RESULTS.md`; `documentation/HANDOFF_NEXT_SESSION.md`; `scripts/setup.py`; `scripts/convert_models.py`; `requirements.txt`; `requirements-dev.txt`; `requirements-convert.txt`; `requirements-eval.txt`; `benchmark-requirements.txt`; `Dockerfile`; `docker-compose.yml`; `.github/workflows/ci.yml`; `Makefile`; `tests/bp_streaming_fixes_test.py`; `tests/bp_low_latency_profile_test.py`; `test/conftest.py`; `test/readiness_regression_test.py`; `test/incremental_tests.py`; `test/security_tests.py`; `test/backend_auth_tests.py`; `test/config_tests.py`; `test/mt_model_tests.py`; `test/vad_tests.py`.

Exclusions: three vendored whisper_streaming modules not line-reviewed; bp mirror only hash/diff-reviewed as stated, remaining different mirrors not fully audited; rest of test/, scripts/, frontend, benchmark artifacts, binary/model/audio assets and documents belong to the parent audit or remain unread here. No blanket entire-project file-to-file completion claim. No clean-machine setup, Docker build, physical UI/mic test, human listening grade, real GPU inference or commercial licence validation ran in this worker.

Roast: weakest point is the mismatch between green normal gates and source-preservation/meeting contracts; cheapest failure is a fast, fluent utterance that omits a short negation or never reaches a listener. Smallest fix: promote the failing source-span regression and a two-client Stop/delivery check into the canonical gate before increasing model complexity.

suggest: maintain one machine-readable route manifest and one test entry point; current registry, TTS factories, packs, auto-config, image and mirror drift are independently visible above.

🟢 Source inspection and AST probes ran; targeted pytest: 14 passed, 2 failed; GPU/clean-clone/human quality untested.

## Final coverage addendum — complete test source pass

Supersedes the earlier test-source exclusions: all Python source under `test/` (including archive), both Python modules under `tests/`, and `test/session_playback_test.cjs` were subsequently read in full using `cat`/bounded `sed`. `test/__init__.py` is empty. Archives were read, never executed. No remaining test-source exclusion for these extensions. Other worker owns `bp/` duplicate review; vendor/binary/generated exclusions remain as above.

**C17 (P1 if broad discovery runs): archived tests still delete owner files.** `test/archive/new_backend_voice_management_tests.py:67` deletes every voice owned by user id 1 on fixture teardown without a snapshot or test-root isolation. Its dummy fixture deletes `test/dummy_audio.wav` even when it pre-existed (`:42`). Keep archive out of runnable discovery until all paths use tmp dirs, explicit created-file ownership and restoration. `test/model_testing_framework.py:18` also attempts NLTK downloads on import; a collection pass is not necessarily offline. `test/engine_registry_tests.py:12` already contains the correct missing-registry assertion, but it is absent from Makefile/CI. Smallest fix: separate archived material from discovery and make canonical gates explicit and hermetic before broadening them.

**C18 (P1 for benchmark claims): current concurrency script can overstate service success.** `test/concurrent_performance_test.py:260` opens WS without a token; `:386` marks completed without requiring translated text/WAV or checking accumulated error. `:585` takes the maximum success rate across user counts, then attributes it to the maximum user count (`:591`). `:190` measures its own test-client process, not the backend. Its E2E is first-source-send to LAST audio receipt, which includes the recording duration. Do not reuse old capacity prose from this harness as current near-real-time/capacity proof. Smallest fix: authenticated client with explicit readiness, exact source/WAV assertions, per-count summaries, backend PID/resource measurement and correctly named first-audio versus final-completion metrics.

Per-file source-read inventory (SHA-256 prefix records bytes at audit completion; it does not imply execution):

| File | Lines | SHA-256 prefix | Review |
|---|---:|---|---|
| `test/__init__.py` | 0 | `e3b0c44298fc1c14` | source read; not run in this worker |
| `test/archive/backend_audio_test.py` | 159 | `45257f9256f7c4bc` | source read; not run in this worker |
| `test/archive/concurrent_pipeline_test.py` | 358 | `ada1f8c87d2120a9` | source read; not run in this worker |
| `test/archive/new_backend_voice_management_tests.py` | 330 | `81f65970f6b390a9` | source read; not run in this worker |
| `test/archive/piper_personal_smoke_test.py` | 34 | `01cb143b6d72d1cc` | source read; not run in this worker |
| `test/archive/soak_interrupt_cycles.py` | 110 | `92e69174325390d5` | source read; not run in this worker |
| `test/archive/standalone_piper_test.py` | 41 | `515a2e17c7f1f709` | source read; not run in this worker |
| `test/archive/standalone_vad_test.py` | 115 | `cebff05d535afa8c` | source read; not run in this worker |
| `test/archive/stt_comparison.py` | 130 | `b5f9023d21cac086` | source read; not run in this worker |
| `test/archive/stt_model_tests.py` | 54 | `23e8d9e19b7c63e0` | source read; not run in this worker |
| `test/archive/ui_charting_tests.py` | 20 | `b8c7901f335a26ac` | source read; not run in this worker |
| `test/archive/ui_functional_tests.py` | 191 | `dd19016d2caf89ef` | source read; not run in this worker |
| `test/archive/xtts_registry_smoke_test.py` | 46 | `69f39437288b78ef` | source read; not run in this worker |
| `test/backend_api_tests.py` | 181 | `603f367d78d8ede5` | source read; not run in this worker |
| `test/backend_auth_tests.py` | 155 | `a85621fc94fe3dee` | source read; not run in this worker |
| `test/concurrent_performance_test.py` | 808 | `7e1812b52c461f8e` | source read; not run in this worker |
| `test/config_tests.py` | 35 | `cd7cf99086dd1a49` | source read; not run in this worker |
| `test/conftest.py` | 167 | `f779dea3fb6e098f` | source read; not run in this worker |
| `test/engine_registry_tests.py` | 72 | `d0de2b623f66f87b` | source read; not run in this worker |
| `test/hardware_scan_tests.py` | 89 | `4c9c045f37f7bda2` | source read; not run in this worker |
| `test/hardware_test.py` | 41 | `b476ad80b78661c6` | source read; not run in this worker |
| `test/i18n_dict_test.py` | 38 | `f8d450808afa63a1` | source read; not run in this worker |
| `test/incremental_tests.py` | 36 | `14745b14d52fd2fc` | source read; not run in this worker |
| `test/language_evidence_test.py` | 183 | `a5384489551f3e5e` | source read; not run in this worker |
| `test/model_testing_framework.py` | 89 | `9a3a068cad8ccecd` | source read; not run in this worker |
| `test/mt_model_tests.py` | 85 | `2dfbf88a8c583424` | source read; not run in this worker |
| `test/readiness_regression_test.py` | 261 | `9e1afd57c6315030` | source read; not run in this worker |
| `test/security_tests.py` | 111 | `c182e63b56e89565` | source read; not run in this worker |
| `test/session_playback_test.cjs` | 140 | `d8cb028d4807bc32` | source read; not run in this worker |
| `test/ui_manifest_test.py` | 32 | `8c366e99f8ca626b` | source read; not run in this worker |
| `test/vad_tests.py` | 376 | `0a74545d607a6162` | source read; not run in this worker |
| `test/voice_similarity.py` | 33 | `c03ce2997385217d` | source read; not run in this worker |
| `tests/bp_low_latency_profile_test.py` | 104 | `cafaf14b3cdc1549` | source read; targeted pytest above |
| `tests/bp_streaming_fixes_test.py` | 276 | `4e0c6d7afbdb1bf4` | source read; targeted pytest above |

Final source count from pathlib: 34 test Python/CJS files, 4900 lines. Existing parent-run make/Node receipts are separate from this worker’s test command.
