# BP documentation / UI / packaging audit — 2026-11-06

Verdict: useful prototype, release-blocking UI lifecycle and packaging defects, contradictory current documentation. This is an audit, not implementation or release approval. Best achievable initial release: a new operator installs a declared CPU or CUDA profile, runs one evidence-qualified route with intelligible captions/speech, sees experimental limits, and can reliably Stop/recover. The largest current gap is evidence and runtime agreement, not visual polish.

## Method and scope

Read repository AGENTS.md and ~/ai-context/CLAUDE.md. Read current contracts, top-level documentation, historical reports, runbooks, specs, browser JS and HTML content, Docker/Compose/Makefile. Commands: cat, sed, nl, rg; HTMLParser initially extracted every UI HTML page's complete text and inline scripts, then complete raw HTML markup and complete CSS source were read in bounded batches. No code edits, model downloads, GPU jobs, application servers, browser actions, deployments or external sends. Only this NEW report was written. Binary models/audio/images, vendor-minified Chart.js, locale dictionaries, generated manifests and private benchmark payloads were not semantically reviewed here; parent owns benchmark/script/backend review. Source review includes all scoped CSS and raw HTML markup; visual rendering and browser accessibility acceptance remain untested.

Coverage below is deliberately granular. FULL means complete file content was returned without a known output gap. PARTIAL means targeted or truncated reading and remains explicitly unfinished. HTML-TEXT means all text/inline scripts read with targeted attributes, not every markup token. SCAN means cross-file textual/static inspection only. The final project report must not describe PARTIAL/SCAN files as fully read.

## Confirmed release blockers

### P0. Real WAV playback duration is always zero; queue protection and adaptive playback are ineffective

Source: ui/live-speech/session-playback.js:58-74. DataView.getUint32(offset,false) reads big endian but compares RIFF/WAVE/fmt/data against reversed constants. A normal RIFF/WAVE PCM file fails at line 58. enqueue:42-46 therefore cannot enforce the documented 10-second budget; adaptive rate:90 sees zero-duration backlog. test/session_playback_test.cjs:12-31 creates the same reversed magic numbers, so it tests an invalid WAV fixture and agrees with the implementation's defect.

Deterministic command: Node constructed standard byte strings RIFF/WAVE/fmt /data, PCM16 mono 16kHz, 32,000 payload bytes. SessionPlayback._wavDuration returned **0**, expected **1 second**. Parent independently reproduced. Smallest fix: correct FourCC reading/constants, use a genuinely valid WAV fixture (or Python wave-generated fixture), then exercise a >10s backlog. No fix applied.

### P0. Stop→Start allows a pending prior-generation blob to play

Source: ui/live-speech/session-playback.js:35-49. active is checked before await blob.arrayBuffer(), with no generation check after it. Deterministic Node probe started enqueue with a deferred arrayBuffer, called stop(), start(), then resolved the old buffer; **old_blob_plays_after_stop_restart=1** and current=true. This violates the file's terminal-Stop contract. Smallest fix: snapshot generation before await, reject stale resolution afterward, test Stop→Start while reading a blob. A stopped-but-not-restarted enqueue can also retain a stale queued blob until next start; the reproduced audible case is Stop→Start.

### P0 for copied shared deployment. Compose publishes a known JWT signing secret

Source: docker-compose.yml:21,29 and port mapping:19; backend/utils/auth.py:21-23 accepts any nonempty secret. `${JWT_SECRET:-change-me-in-production}` disables the safer random-secret fallback while listening on 0.0.0.0. A novice copying documented docker compose up can launch a server with a publicly known signing key. No forged token was sent. Smallest fix: require supplied secret (fail startup/Compose interpolation), loopback default or explicit shared mode, then negative auth tests. Merely warning in a comment is weaker than preventing startup.

### P1. User deletion succeeds on the server then the UI throws

Source: ui/live-speech/live-script.js:1873 sets voiceToEdit=null and :1877 immediately reads voiceToEdit.path. This is a deterministic null dereference after successful delete; the catch presents a misleading network/server error. Preserve deleted path before clearing the state. Source-confirmed; no real voice was deleted.

### P1. SDK's documented transport/install path cannot be followed

Source: documentation/sdk.md:28 recommends pip install -e ., but ls in this session found no root pyproject.toml, setup.py or setup.cfg. sdk.md:211 says PCM16 input; actual UI sends Float32Array.buffer (live-script.js:673-677), backend/main.py:2286 reads float32. sdk.md:242 sends MediaRecorder blobs, which are encoded browser containers rather than Float32 PCM; example also omits waiting for model readiness and explicit Start. REST table uses inconsistent /api prefixes, /voices/list instead of /api/voices (backend/main.py:1267), path filename instead of /api/voices/file?filename= (backend/main.py:1307). No payload was submitted. Smallest fix: one executable client contract shared with paced harness, plus one packaged or clearly repo-local entry point.

### P1. Docker is not a complete fresh-start/restore path

Dockerfile copies code/dependencies but no scripts/setup.py or ready model profile; empty CT2/voice directories are created (:33). Missing offline assets cause runtime preparation failure. Compose mounts ct2_models but not the voice-model directory, and does not persist sql_app.db (backend/utils/db_manager.py:35), so account state can disappear when the container is recreated even while voice files persist. python:3.11-slim is not a CUDA runtime; enabling a GPU device reservation alone is not proof CUDA libraries exist. Health checks hit /api/hardware (:57) rather than route readiness, so green container health can coexist with no usable translation route. .dockerignore excludes *.onnx/*.bin/audio and private outputs, avoiding inadvertent bundling of local voice binaries through COPY backend/. No Docker build/install run occurred.

### P1. Current requirement authority contradicts itself

PRODUCT.md:9 mandates English↔Slovak; constitution.md:11-12 and :31 amend core to Czech↔English and experimental Slovak on Nov 5. README/website lead EN→SK while HANDOFF_NEXT_SESSION.md says CS↔EN default. specs/003 spec:248 still requires Slovak reference and pending constitution amendment. This prevents one acceptance bar or honest cleanup decision. The current user request explicitly requires SK/CS→EN investigation, so no audit should silently drop Slovak. Smallest fix: owner-scope-preserving requirements matrix, explicit primary/experimental route status, amend current docs together; keep dated decisions as history.

### P1. Latest verification report promotes offline speed into trusted UI quality

Source: documentation/VERIFICATION_RESULTS_2026-11-06.md:4 diagnoses WebSocket ping timeout as only a harness limitation and asserts UI unaffected without a browser test; :27 and :54 call local EN→SK trusted/presentation-ready from direct inference; :38's first audio 0.39s is offline pipeline output availability, not capture-to-ear delay. :12 uses 2s windows but :85 says 1s fix. :60 says historical T4 numbers need re-verification; :125 and closing line nevertheless mark route trusted. No multilingual human assessment, long continuous physical listener route or browser E2E closes these gaps. Benchmark numbers require aggregation/metric/sample/config mapping; parent audit owns recomputation. Smallest fix: demote to inference-functional receipt; separate offline RTF, onset/offset-to-server-WAV, and actual audible playback with p50/p95, errors and quality.

### P1. Licence inventory is contradictory and cannot grant commercial clearance

LICENSE:29 calls Piper TTS/voices MIT; pinned piper-tts==1.3.0 and NOTICE.md:7 call runtime GPL-3.0-or-later. documentation/LICENSE_INVENTORY_2026-11-06.md:17 marks owner SK fine-tune commercially approved solely from ownership, while documentation/licences.md:17 says personal EN warm-start from NC ryan and SK base unreviewed. The same inventory labels Piper locally-only and not-bundled while Docker installs the dependency into its image. OmniVoice is Apache-2.0 in model_landscape addendum, unknown in inventory/NOTICE, and NC in licences.md's older unverified note. Qwen3-TTS MIT versus Apache-2.0 varies. These are repo contradictions, not independent legal determinations. Smallest fix: record exact code/checkpoint/voice/base/dataset revision + licence URL/hash + distribution decision; distinguish commercial permission from distribution obligations. Do not resolve by merely choosing newest prose. Parent researches upstream.

## Other source-confirmed cleanup / product defects

- PROJECT_INDEX.md:57,308 calls backend/hardware.py duplicate and proposes deprecation; live imports remain in backend/tts/base.py:29 and main.py:1411, plus Omni/Piper/Hybrid/Coqui. Removing it from an index recommendation would break runtime. Index names absent backend/database.py and hardware_detect.py (:56-58), incomplete packs/locales and outdated audio durations, calls a claims ledger historical while DESIGN.md requires it current. Generate structural inventory; keep judgement manually reviewed.
- documentation/INDEX.md:5 says 26 files, :6 calls documentation/README.md stale for retired components it no longer presents as default, yet later lists the same README as current. Dead recording-script/research links and an empty CLI table remain. Three overlapping indexes plus PLAN/AGENTS/current handoff compete.
- Makefile:32-35 make demo-audio validates FILE/SRC/TGT but never invokes the promised processing command; exits zero after argument checks. Source proof, no inference needed. Restore invocation or remove target promise.
- Makefile:38 serves only ui/voice-lab; lab.js:859 fetches ../../PLAN.md and library media commonly lives outside that root. Static lab cannot expose those files from this document root; old instructions instead serve whole repo, which conflicts with new private-media boundary. Provide an explicit private operator evidence server/export, rather than recommending a broad unauthenticated repo server.
- ui/test-stt/test-stt.js:2 connects /ws without JWT, :30 calls unauthenticated /initialize, never sends config_update/start; test surface remains marked Active in PROJECT_INDEX. Four-times mic gain differs from live; no worklet-to-destination pull chain. Archive or update to current protocol before recommending VAD_TUNING_GUIDE tests.
- live-script.js:1295 uses undefined voice.path/name when adding a missing sidebar voice option; usually masked because dropdown already contains all voices. Source-confirmed latent ReferenceError, not reproduced in browser.
- renderVoices installs delegated click and select handlers on every refresh (:1222,1321), causing accumulation after uploads/renames/deletes. Bind once or replace handler. Voice language/name/id are interpolated into innerHTML (:1194,1208); backend sanitizes names but speaker_lang is a form string (:1008). Treat language-derived markup as an injection candidate; no exploit attempted. Prefer textContent and validated IDs/codes.
- Live voice recorder wraps default MediaRecorder container bytes in audio/wav Blob (:1522) and uploads .wav (:1796), without actual WAV encoding. FFmpeg may sniff it successfully, but MIME/extension and bytes disagree. Use recorder mimeType and a matching extension or encode real WAV. Recording context/animation cleanup differs across close/cancel/success paths.
- caps.js:28 maps installed assets to Speech and captions and omits evidence/experimental state from /api/routes. live-langs.js only allows speech-tier routes, so installed ≠ qualified is not represented in picker status. Main route-status card partially mitigates after preparation. Preserve availability, supported modality and quality as independent fields.
- ui/auth/auth.html:109 loads Google GIS script unconditionally; cookies.html:59 says it loads where configured. cookies storage list (:48-53) omits lang/livePair/i18nDrafts (and private-lab hlas-ratings if in scope). Correct handling/description together; no legal prose edited.
- Home default guest/no-account promise is unconditional while shared hosts default guest disabled; README and FAQ qualify deployment. Home instructions say press Start, but guided runtime requires Prepare→check→isolation→Start. Show deployment-aware next step.
- DESIGN.md:44 says fonts load from Google Fonts, but UI uses self-hosted vendor fonts; current rules prohibit hardcoded language lists while live voice/modal/static fallbacks include fixed EN/SK/CS and engines. Define which fallback/recording-script lists are deliberate exceptions.

## Historical documents: preserve evidence, remove present-tense authority

Historical results are valuable but should be immutable snapshots linked from one current evidence index. Do not mass-delete references, rerun costly models or overwrite original measurements merely to make prose consistent.

- STREAMING_SPEC.md:19-30 accumulates 2s before STT, yet :44 claims ~0.5s GPU first word; acquisition/LocalAgreement time omitted from budget. ARCHITECTURE_V2 says design while PROJECT_INDEX/AGENTS promote components as true streaming. Use targets as targets and show implemented mode and measured delay separately.
- model_landscape_2026-10.md internally says no open Slovak zero-shot solution then contains successful OmniVoice SK results; first sections reject short RTF 0.40–0.65 as not real time while addenda recognize RTF<1. Contains contradictory Seamless SK output claims and numerous old READY/RUNNING states. Keep per-run cards + current recommendation ledger instead of appending incompatible conclusions.
- load_tests_2026-10.md's added production split says OmniVoice not real time at RTF 0.40–0.65; body correctly calls RTF0.18–0.34 faster than real time. VRAM measured in §5 but later Not measured still says not captured. Distinguish total generation throughput from first-chunk latency/playback backlog, and which run collected VRAM.
- PERFORMANCE_TEST_RESULTS.md says 12 verified users, 15 failure from RAM/CPU exhaustion. concurrency_accuracy_2026-08-21.md later establishes resource monitor measured the client, most high-N successful sessions had no transcript, and teardown timeouts explain failures. The later caveats must accompany earlier figures in any thesis/commercial use.
- thesis_draft.md:12 has FEI placeholder; :611 Python3.9+; :677 says Parakeet adopted EN; :687 says opposite MT direction same model reversed; :1124/:1166 and repeated :1261 assert server RAM exhausted from unmeasured client footprint. Duplicate numbered sections and repeated 12-user conclusions. Raw examples omit auth/current protocol and use old voices. Substantial English prose contradicts :1444's draft declaration that author wrote text in Slovak and AI only grammar. This declaration requires owner verification, never silent editing or acceptance by an agent. A missing DOPLNIT marker does not make the thesis ready.
- specs/003 tasks:98,128,144 list already-existing health/SDK/moved scripts open; :149 remains unchecked while following line says DONE. :112 marks system-audio capture done, but current live UI has no getDisplayMedia capture path. Reconcile by current executable evidence, preserving partial/failed acceptance states; do not tick from file existence.

## Ordered cleanup and release acceptance proposal

1. Repair playback duration/generation and Compose secret; verify real WAV backlog, Stop/restart race, no forged session from stock configuration. Keep implementation changes separate from this audit.
2. Freeze route/config and benchmark receipt schema with current code/model hashes; attach WER and bilingual semantic/intelligibility grades separately from RTF/latency. Preserve failed utterances and source-loss measurements.
3. Reconcile PRODUCT/constitution/current README/AGENTS/PLAN as one current requirements matrix. One evidence index links immutable historical reports. Remove generated-file counts from prose; generate path/link inventory and stale checks.
4. Repair SDK/CLI/debug UI contracts and lab scope. Add a verified first-sentence command per supported OS/profile, supported Python versions, asset manifest/checksum/licence, explicit download provisioning and offline failure instructions. No needless new framework.
5. Make release packaging reproducible: CPU profile first, independently supported CUDA profile, pinned resolved dependencies/assets, persistent account/voice data, health/readiness distinction, backup/restore/start/stop smoke. A Dockerfile alone does not meet clean-machine acceptance.
6. Resolve exact upstream licences/voice provenance, sensitive git-history/distribution audit, consent and deployment retention. Review legal copy with a responsible operator; source licence alone is not model clearance.
7. Named pilot: physical input→translation→selected output, 10–30min continuous source, quality across ≥3 speakers, p50/p95 listener delay, queue trend, missing/duplicate text, reconnect/device/auth faults and repeated Stop. Agree numeric tolerances before execution. Current minimum targets are proposals, not verified universal commercial bars.

Smallest architecture correction: keep cascade inspectability; stop translating/speaking contextless STT fragments, retain rolling audio context, commit stable source words and translate coherent clauses, using provisional captions separately from irrevocable speech. Benchmark against current short-segment baseline before adopting. A new unified model, GPU, room UI or microservice rewrite cannot by itself resolve stale claims, loss of context or playback state bugs. Parent owns primary-source model research.

## Coverage manifest

| File | Lines | Review | Remaining limit |
|---|---:|---|---|
| `.dockerignore` | 31 | FULL | Source-level only; runtime acceptance not implied |
| `.specify/memory/constitution.md` | 31 | FULL | Source-level only; runtime acceptance not implied |
| `AGENTS.md` | 252 | FULL | Source-level only; runtime acceptance not implied |
| `DESIGN.md` | 80 | FULL | Source-level only; runtime acceptance not implied |
| `Dockerfile` | 59 | FULL | Source-level only; runtime acceptance not implied |
| `LICENSE` | 40 | FULL | Source-level only; runtime acceptance not implied |
| `Makefile` | 44 | FULL | Source-level only; runtime acceptance not implied |
| `NOTICE.md` | 15 | FULL | Source-level only; runtime acceptance not implied |
| `PLAN.md` | 247 | FULL | Complete cumulative bounded reads; source-only |
| `PRODUCT.md` | 32 | FULL | Source-level only; runtime acceptance not implied |
| `PROJECT_INDEX.md` | 316 | FULL | Source-level only; runtime acceptance not implied |
| `README.md` | 113 | FULL | Source-level only; runtime acceptance not implied |
| `UPDATES.md` | 91 | FULL | Source-level only; runtime acceptance not implied |
| `VAD_TUNING_GUIDE.md` | 111 | FULL | Source-level only; runtime acceptance not implied |
| `docker-compose.yml` | 74 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/ARCHITECTURE_V2.md` | 241 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/AUDIT_2026-11-06_CODE.md` | 155 | EXCLUDED | Concurrent sibling report; parent synthesizes |
| `documentation/HANDOFF_2026-11-04_session2_spec_plan.md` | 130 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/HANDOFF_NEXT_SESSION.md` | 72 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/I18N.md` | 39 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/INDEX.md` | 76 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/KAGGLE_SETUP.md` | 88 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/LICENSE_INVENTORY_2026-11-06.md` | 59 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/MULTI_SPEAKER.md` | 134 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/MULTI_USER_SERVER.md` | 212 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/MULTI_USER_UI_HANDOFF.md` | 228 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/PERFORMANCE_TEST_RESULTS.md` | 134 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/PLAN_2026-11-03.md` | 138 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/PROJECT_MATRIX_2026-11-04.md` | 29 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/README.md` | 126 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/REVIEW_AND_UPDATE_PLAN_2026-11-04.md` | 86 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/ROAST_REPORT_2026-10-21.md` | 148 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/STREAMING_SPEC.md` | 126 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/TEMPORARY_NOTES_2026-11-03.md` | 53 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/TEMP_NAVIGATION_MAP.md` | 108 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/UI_REVIEW_2026-11-05.md` | 156 | FULL | Complete cumulative bounded reads; source-only |
| `documentation/VERIFICATION_RESULTS_2026-11-06.md` | 130 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/backend-fixes-2026-11-04.md` | 41 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/claims.md` | 15 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/cli_quickref.md` | 125 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/compute_capacity_report_2026-09.md` | 187 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/concurrency_accuracy_2026-08-21.md` | 94 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/demo_report_2026-09.md` | 81 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/demo_runbook_2026-09-28.md` | 111 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/documentation_audit_2026-11-03.md` | 122 | FULL | Complete cumulative bounded reads; source-only |
| `documentation/envs/README.md` | 15 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/evaluation_2026-10-18.md` | 144 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/footprint_audit_2026-09-28.md` | 102 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/fundamentals.md` | 151 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/fundamentals_deep.md` | 100 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/handoff-2026-11-03-pipeline-pivot.md` | 110 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/implementation-2026-11-04/RESULTS.md` | 68 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/implementation-2026-11-04/final-roast.md` | 34 | FULL | Complete cumulative bounded reads; source-only |
| `documentation/implementation-2026-11-04/handoff-before-fixes.md` | 101 | FULL | Complete cumulative bounded reads; source-only |
| `documentation/implementation-2026-11-04/kaggle-report.md` | 134 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/implementation-2026-11-04/kaggle-route-private/bp/languages/README.md` | 38 | FULL | Complete cumulative bounded reads; source-only |
| `documentation/implementation-2026-11-04/kaggle-t4-incremental.md` | 27 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/implementation-2026-11-04/kaggle-t4-paced-route.md` | 84 | FULL | Complete cumulative bounded reads; source-only |
| `documentation/implementation-2026-11-04/streaming-findings.md` | 70 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/kaggle_t4_results_2026-11-05.md` | 39 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/kaggle_validation_2026-11-03.md` | 91 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/licences.md` | 35 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/linux_setup_and_test.md` | 97 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/load_tests_2026-10.md` | 143 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/model_landscape_2026-10.md` | 419 | FULL | Complete cumulative bounded reads; source-only |
| `documentation/monday_test_checklist.md` | 39 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/personal_voice_bootstrap_2026-08-19.md` | 33 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/piper_training_runbook.md` | 96 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/presentation_runbook_2026-11-04.md` | 38 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/progress_since_the_break.md` | 83 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/project_overview.md` | 148 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/project_status_2026-10-20.md` | 222 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/reading_script_bilingual.md` | 29 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/recording_session_2026-09-28.md` | 106 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/research_findings_2026-11-03.md` | 144 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/review-design-evidence-2026-11-04.md` | 59 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/review-product-2026-11-04.md` | 86 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/s2s_landscape_2026-10-18.md` | 180 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/sdk.md` | 445 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/security_audit_2026-09.md` | 44 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/setup_friction_2026-10.md` | 24 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/supabase_auth_db_guide.md` | 43 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/teams_auth_ticket_2026-10-14.md` | 46 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/thesis_draft.md` | 1462 | FULL | Complete cumulative bounded reads; source-only |
| `documentation/verification_omni_piper_sk_2026-09-30.md` | 67 | FULL | Source-level only; runtime acceptance not implied |
| `documentation/voice_and_app_direction_2026-09.md` | 132 | FULL | Source-level only; runtime acceptance not implied |
| `guide.md` | 125 | FULL | Source-level only; runtime acceptance not implied |
| `requirements.txt` | 43 | FULL | Source-level only; runtime acceptance not implied |
| `specs/001-realtime-cross-platform-translation/plan.md` | 78 | FULL | Complete cumulative bounded reads; source-only |
| `specs/001-realtime-cross-platform-translation/spec.md` | 140 | FULL | Complete cumulative bounded reads; source-only |
| `specs/001-realtime-cross-platform-translation/tasks.md` | 289 | FULL | Complete cumulative bounded reads; source-only |
| `specs/002-pwa-pip-demo/plan.md` | 31 | FULL | Source-level only; runtime acceptance not implied |
| `specs/002-pwa-pip-demo/spec.md` | 66 | FULL | Source-level only; runtime acceptance not implied |
| `specs/002-pwa-pip-demo/tasks.md` | 27 | FULL | Source-level only; runtime acceptance not implied |
| `specs/003-end-to-end-completion/plan.md` | 116 | FULL | Source-level only; runtime acceptance not implied |
| `specs/003-end-to-end-completion/spec.md` | 286 | FULL | Source-level only; runtime acceptance not implied |
| `specs/003-end-to-end-completion/tasks.md` | 152 | FULL | Complete cumulative bounded reads; source-only |
| `specs/_archive/README.md` | 29 | FULL | Source-level only; runtime acceptance not implied |
| `ui/auth/auth.css` | 68 | FULL | Full CSS source; visual/accessibility runtime untested |
| `ui/auth/auth.html` | 122 | FULL | Full raw markup plus text/inline scripts; no browser render |
| `ui/auth/auth.js` | 317 | FULL | Source-level only; runtime acceptance not implied |
| `ui/auth/session-nav.js` | 25 | FULL | Source-level only; runtime acceptance not implied |
| `ui/enroll/enroll-style.css` | 55 | FULL | Full CSS source; visual/accessibility runtime untested |
| `ui/enroll/enroll.html` | 65 | FULL | Full raw markup plus text/inline scripts; no browser render |
| `ui/enroll/enroll.js` | 157 | FULL | Source-level only; runtime acceptance not implied |
| `ui/extra/about.html` | 93 | FULL | Full raw markup plus text/inline scripts; no browser render |
| `ui/extra/cookies.html` | 90 | FULL | Full raw markup plus text/inline scripts; no browser render |
| `ui/extra/doc.js` | 28 | FULL | Source-level only; runtime acceptance not implied |
| `ui/extra/extra.css` | 7 | FULL | Full CSS source; visual/accessibility runtime untested |
| `ui/extra/faq.html` | 61 | FULL | Full raw markup plus text/inline scripts; no browser render |
| `ui/extra/feedback.html` | 99 | FULL | Full raw markup plus text/inline scripts; no browser render |
| `ui/extra/privacy.html` | 106 | FULL | Full raw markup plus text/inline scripts; no browser render |
| `ui/extra/terms.html` | 100 | FULL | Full raw markup plus text/inline scripts; no browser render |
| `ui/global-styles.css` | 381 | FULL | Full CSS source; visual/accessibility runtime untested |
| `ui/home/home.css` | 247 | FULL | Full CSS source; visual/accessibility runtime untested |
| `ui/home/home.html` | 281 | FULL | Full raw markup plus text/inline scripts; no browser render |
| `ui/home/home.js` | 254 | FULL | Source-level only; runtime acceptance not implied |
| `ui/i18n/i18n.js` | 222 | FULL | Source-level only; runtime acceptance not implied |
| `ui/live-speech/account-hint.js` | 20 | FULL | Source-level only; runtime acceptance not implied |
| `ui/live-speech/audio-processor.js` | 46 | FULL | Source-level only; runtime acceptance not implied |
| `ui/live-speech/live-flow.js` | 43 | FULL | Source-level only; runtime acceptance not implied |
| `ui/live-speech/live-langs.js` | 126 | FULL | Source-level only; runtime acceptance not implied |
| `ui/live-speech/live-script.js` | 2087 | FULL | Source-level only; runtime acceptance not implied |
| `ui/live-speech/live-style.css` | 1569 | FULL | Full CSS source; visual/accessibility runtime untested |
| `ui/live-speech/live.html` | 507 | FULL | Full raw markup plus text/inline scripts; no browser render |
| `ui/live-speech/session-playback.js` | 140 | FULL | Source-level only; runtime acceptance not implied |
| `ui/shared/caps.js` | 100 | FULL | Source-level only; runtime acceptance not implied |
| `ui/shared/lang-picker.css` | 70 | FULL | Full CSS source; visual/accessibility runtime untested |
| `ui/shared/lang-picker.js` | 144 | FULL | Source-level only; runtime acceptance not implied |
| `ui/site.css` | 163 | FULL | Full CSS source; visual/accessibility runtime untested |
| `ui/sw.js` | 73 | FULL | Source-level only; runtime acceptance not implied |
| `ui/test-stt/test-stt.html` | 212 | FULL | Full raw markup plus text/inline scripts; no browser render |
| `ui/test-stt/test-stt.js` | 265 | FULL | Source-level only; runtime acceptance not implied |
| `ui/theme-toggle.js` | 46 | FULL | Source-level only; runtime acceptance not implied |
| `ui/vendor/chart.min.js` | 13 | EXCLUDED | Vendored minified library; dependency/licence audit separate |
| `ui/vendor/fonts.css` | 105 | FULL | Full CSS source; visual/accessibility runtime untested |
| `ui/vendor/material-symbols.css` | 68 | FULL | Full CSS source; visual/accessibility runtime untested |
| `ui/voice-lab/lab-style.css` | 273 | FULL | Full CSS source; visual/accessibility runtime untested |
| `ui/voice-lab/lab.html` | 119 | FULL | Full raw markup plus text/inline scripts; no browser render |
| `ui/voice-lab/lab.js` | 916 | FULL | Complete cumulative bounded reads; source-only |

No stale documentation was edited. Scoped documentation/UI source coverage is complete except the explicit excluded files and payload classes. This is not a claim of whole-project runtime, visual, generated-data, dependency-binary or model review.

## Later read completion and important GPU receipt

Completed cumulative reads of PLAN, historical review/handoff/implementation docs, full landscape, full thesis, archived spec 001 and active spec003 task list, and the previously truncated Voice Lab renderer. Coverage rows updated accordingly. Remaining source limits are visual/browser checks, vendor/minified content and generated/private payloads; sibling audit report is excluded from independent scope. `node --check ui/live-speech/live-script.js` and `node --check ui/live-speech/session-playback.js` exit 0; syntax success does not contradict the runtime defects above.

The historical **real** T4 full-route OmniVoice receipt is `documentation/implementation-2026-11-04/kaggle-t4-paced-route.md`, distinct from TTS-only synthetic-session work. EN→SK and EN→CS, 90s paced input, production /ws in a throwaway GPU copy: EN→SK 17 translations/16 voiced, first-WAV p50/p95 1.38/2.90s, modeled listener lag11.43/17.82s at1x; EN→CS16/16, first-WAV1.42/1.64s, lag3.02/5.12s. Unknown-licence gate was modified to commercial-ok in Kaggle-only copy; package pins also relaxed for3.13. No saved output audio hashes/human meaning/identity grade, physical capture or browser playback. This establishes historical GPU cloned generation/full-route execution, **not** sustained acceptable listener delay or commercial readiness. Parent owns live receipt verification.

## Completed raw HTML/CSS review and distribution notices

All 12 HTML pages and 11 CSS files were subsequently read as complete raw source (bounded `cat`/`sed` reads; live-style.css 1–500,501–1000,1001–1569; live.html1–220,221–507). Ledger now marks these FULL for source review only. Browser/desktop/390px screenshots, assistive technology, keyboard interactions and rendering remain untested.

Additional source findings:

- **P2 broken evidence link:** ui/extra/about.html:52 links `/#evidence`; ui/home/home.html contains sections hero/demo/how/start/about/faq, no evidence ID. Point the link at the actual measured section or give the canonical evidence section a stable ID.
- **P2 modal semantics/keyboard control:** ui/live-speech/live.html:374–376 voice-recording overlay uses generic divs, lacks dialog/aria-modal/title binding and its close control is a span with no tabindex/button semantics. The newer voiceAccountDialog:496 has native dialog semantics, so a consistent native-dialog implementation is a small migration. Keyboard/modal interaction was not exercised; review focus lifecycle before claiming accessibility.
- **P3 accumulated CSS:** ui/live-speech/live-style.css:358,402,723 uses Sass `darken()` in a plain stylesheet with no Sass compilation path in these UI files; browsers cannot use those background declarations. Same file repeats account-bottom-links/logout rules and appends grid→flex→grid layout overrides across dated generations. Consolidate only after browser regression evidence; do not mechanically delete old selectors without checking JS classes.
- **P1 distribution notice/provenance gap:** `rg --files ui/vendor` shows Montserrat/Playfair six WOFF2 subsets, seven MaterialSymbolsOutlined TTF weights, two CSS files and Chart.js. No accompanying licence/notice/OFL file occurs in this subtree. `ui/vendor/fonts.css:1,99` records SIL OFL1.1 and Google Fonts fetch date2026-11-05 for Montserrat/Playfair, but omits upstream revision, original copyright notices and full OFL text. `ui/vendor/material-symbols.css:1` says self-hosted, no original source/revision/licence. `ui/vendor/chart.min.js:1–6` preserves Chart.js3.7.0, copyright2021 Contributors and MIT label but not the full MIT permission/condition text. Root NOTICE.md omits these web dependencies entirely. Recover exact original distribution notices/provenance and add them to the release inventory; binary-internal metadata and actual upstream terms were not verified here, so this is a missing-distribution-proof finding, not a claim of incompatible licensing.
