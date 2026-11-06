# BP review, benchmark re-audit and release proposal — 2026-11-06

## Verdict

REWORK before a general self-host or commercial release. Useful thesis prototype with working local translation components, actual T4 synthesis and paced GPU route receipts. Speed is established for specific warm configurations. Reliable meaning, stable continuous listening and a reproducible public installation are not established. The strongest achievable first release for a university operator is one declared language direction that preserves meaning, starts from a clean install with permitted public assets, delivers bounded-delay speech/captions, and stops reliably. Current work has several competing pipelines and acceptance claims ahead of their evidence.

Interpretation: “Kackle” / “Kegel” refers to Kaggle. Current request and matching prior BP references support this interpretation. No product implementation, existing documentation, models or recordings were changed by this review. New audit artifacts only. Existing dirty worktree retained; no commit/push/deploy, new remote job, model download or local GPU run. Short existing CPU tests did run.

Preflight: reused project contracts, existing Kaggle receipts, model matrix and offline tests. Research and roast skills applied; architecture recommendation is a proposal, not an implementation approval.

## Review index and scope

- [Code/security/runtime audit](AUDIT_2026-11-06_CODE.md): app/backend, setup/deployment and tests; per-file coverage and exclusions.
- [Documentation/UI audit](AUDIT_2026-11-06_DOCS_UI.md): current and historical documents, UI source, runnable instructions and claims.
- [Scripts/evaluation/specs audit](AUDIT_2026-11-06_SCRIPTS.md): scripts, bench, language packs, specs and divergent bp mirror; per-file ledger.
- [Recomputed route and OmniVoice artifacts](AUDIT_2026-11-06_BENCHMARKS.json): receipt hashes, percentiles, stage sums, devices, changed source fingerprints; 12 TTS WAVs and three normalized references verified.
- [Rescored ASR confirmations](AUDIT_2026-11-06_ASR_CONFIRMATIONS.json): four 80-clip cases, reference/hash joins, independently recomputed word edit distance.
- [OmniVoice paced-route recomputation](AUDIT_2026-11-06_OMNI_ROUTE.json): two English-source routes, first-audio estimates and modeled listener queues.

Coverage means source inspection, not every branch tested. Binary model weights, personal recordings/database/credentials, installed packages/caches, generated private corpora and all external worktrees are not a blanket line-by-line audit. Vendor code is explicitly excluded from detailed review unless its surrounding contract was inspected. Individual worker ledgers identify any residual partial reads; do not infer “every file reviewed” from a tree listing. No visual design hand-back or fresh physical microphone/browser playback acceptance occurred.

## Fresh verification and what it establishes

| Check run this session | Result | Meaning and limit |
|---|---|---|
| `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTEST_ADDOPTS='--tb=short --show-capture=no' make test` | 87 passed, 3 warnings, 29.45s | Listed suite works in this prepared checkout; not clean-install or translation acceptance |
| `node --test test/session_playback_test.cjs` | 7 passed | Synthetic lifecycle tests; malformed WAV fixture masks duration-parser defect |
| `python3 scripts/ui_manifest.py --check` | Both generated manifests current | Structural freshness only |
| `git diff --check` | Exit 2; whitespace in pipeline.py, model_landscape and three extra HTML pages | Working tree is not whitespace-clean |
| Offline pytest `tests/bp_streaming_fixes_test.py` | 2 failed, 7 passed, 1.07s | One source-dispatch regression, one test-isolation failure; excluded from `make test` |
| Independent valid PCM WAV Node probe | Expected 1 second; parser returned 0 | Actual playback duration/budget defect, despite 7 green tests |
| Python stdlib receipt re-audit | Stored route percentiles/stage sums match; 12 OmniVoice WAV hashes, PCM formats, durations, RTF and 3 reference hashes match | Saved result integrity, not a fresh GPU inference or human quality test |
| Independent stored ASR text rescoring | Four 80-sample WER aggregates match | Component evidence; not UI translation/meeting accuracy |
| `kaggle kernels status` for bounded OmniVoice, production CS route and paced OmniVoice route | COMPLETE | Current job state; receipt contents decide acceptance |
| `kaggle kernels status` for canary/nemo confirmation jobs | COMPLETE | Existing confirmation outputs still require collection/reconciliation against their stopped queue; no duplicate submission |

The second isolated pytest failure bypasses the test’s fake TTS registry and loads existing local Piper. It is a harness isolation failure, not proof that production picks the wrong English voice. The short-utterance test feeds synthetic 20ms speech: it proves unconditional source discard, not recognizability of 20ms audio. Meaningful short-word/negation fixtures remain necessary.

## Benchmarks: corrected claims

### OmniVoice on GPU: confirmed component execution

The saved T4 receipt uses `k2-fsa/OmniVoice`, revision `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`, package 0.2.1 and 12 diffusion steps. Its raw receipt SHA256 matches the sanitized record. All 12 output WAVs and three normalized reference WAVs were reopened and hash/format checked in this session. The runner explicitly requires CUDA, loads `device_map='cuda:0'`, synchronizes CUDA and records GPU allocator usage. Receipt GPU: Tesla T4. Cold model load: 23.018s.

| Generated language | Warm phrase whole-WAV availability | Joined paragraph availability |
|---|---:|---:|
| English | 0.649–0.860s | 1.542s |
| Slovak | 0.691–1.037s | 1.938s |
| Czech | 0.558–0.713s | 1.403s |

This confirms synthesis and reference-prompt execution on T4. It does not confirm native model audio streaming, voice similarity, pronunciation or complete SK/CS→EN translation. BP’s `synthesize_stream` generates complete clauses sequentially. The current runner’s hash differs from the submitted receipt hash, so running today’s file is not reproduction of that historical job without reconstruction.

There is also stronger historical full-route evidence: `bp-route-paced-t4-20261104` ran real `/ws`, CUDA STT, CPU MT and cloned OmniVoice for 90s English→Slovak and English→Czech. Saved raw events and listener arithmetic were rechecked. The Kaggle copy changed dependencies, pinned OmniVoice, and bypassed the unknown-licence gate by marking it commercial-ok. That bypass is an experiment deviation, not commercial permission or proof unmodified BP will accept the engine. No saved output WAV hashes exist for these paced runs.

| Direction | Input / generated speech | Estimated segment-end→first-audio p50 / p95 | Modeled listener lag p50 / p95 at 1x |
|---|---|---|---|
| EN→SK, OmniVoice | 90 / 101.48s | 1.380 / 2.895s | 11.427 / 17.825s |
| EN→CS, OmniVoice | 90 / 87.87s | 1.422 / 1.643s | 3.017 / 5.124s |

These use source end inferred as transcription arrival minus STT duration; listener playback is modeled. They are not physical listening measurements. At 1.25x modeled playback, SK p95 remains 9.607s; speeding up is not a demonstrated general cure. Human meaning/identity grades remain absent.

### Czech→English on T4: fast median, meaning errors in saved rows

The newer local `route_receipt.json` exactly matches `~/Developer/bp-kaggle-out/run2/route_receipt.json`, SHA256 `c45aaf8879516a275df92a2e9a9c8821a3b86072787ef344f87afaffbe57ffb7`. Three measured repeats per profile after one excluded warmup, each using the first 15s of the same Czech recording. STT is CUDA/float16; MT and declared Piper TTS are CPU. Piper provider was not directly measured; do not promote it to GPU-proven.

| Profile | Measured utterances | Onset→first WAV p50 / p95 | STT+MT+TTS stage-sum p50 / p95 | Under 2s |
|---|---:|---:|---:|---:|
| default | 18 | 3.470 / 4.031s | 0.577 / 2.198s | 0/18 |
| P_short | 33 | 2.032 / 3.052s | 0.491 / 2.256s | 10/33 |
| P_short2 | 46 | 1.864 / 4.219s | 0.398 / 2.375s | 26/46 |
| P_incr | 15 | 11.671 / 12.884s | 0.392 / 0.678s | 0/15 |

Metric: first successful server WAV-send completion minus ingress-anchored first voiced sample. Browser playback and physical capture/output excluded. These are different emitted segmentation units, not 46 independent speakers/sentences. The fixed post-input drain window leaves completion unknown. Code fingerprints now differ in core runtime files. No current-checkout latency acceptance follows.

`documentation/kaggle_t4_results_2026-11-05.md` calls this speech-end timing and reports different counts/percentiles (e.g. P_short2 1.58/3.52). Those figures do not match the available run2 receipt. `VERIFICATION_RESULTS_2026-11-06.md` has matching later 1.86/4.22 figures but overpromotes them to trusted. Both need correction/provenance.

All profiles have `reference=null`. Reopened saved rows include malformed Czech recognition and English such as “For the steroids.” for a source fragment associated with Staroměstské náměstí. This is direct counter-evidence to “operates fine.” No numerical WER can be inferred for the full route without aligned references. GPU accelerates inference; using the same weak recognition model on GPU does not itself prove better recognition. Lower latency can come from speaking earlier with less context.

### Slovak→English: no complete near-real-time quality acceptance

The audited full-route receipt is CS→EN. The OmniVoice route runs have English as source. Neither proves SK→EN. Existing ASR and text-translation screens provide useful component comparisons, not listener-route acceptance.

Four saved confirmation receipts were independently rescored in this session, joining each row’s ID/reference/audio hash to the recorded corpus and recomputing micro WER:

| ASR configuration | Czech, 80 clips | Slovak, 80 clips |
|---|---:|---:|
| Generic CT2 small, beam5 float16 | 38.52% WER | 30.84% WER |
| CT2 large-v3-turbo | 12.71% WER | 9.81% WER |

These use FLEURS read speech; the models/configurations are not the production base model or the same UI recording. They support a GPU quality-tier experiment, not automatic promotion. KInIT’s card lists FLEURS among training data, so any KInIT result on it needs overlap qualification. Existing Canary/Nemo 80-clip jobs now show COMPLETE; their outputs were not downloaded/re-audited here. Do not claim the model-selection campaign is reconciled.

## Why accuracy currently suffers

Proven code properties:

1. `backend/main.py:2438–2464` destructively consumes frames for independent short-window decoding while the incremental path can also own them. It bypasses the ordered tracked task entry point. `.pop(0)` leaves `_Frames.t` timestamps behind. The partial-frame split also takes the suffix while retaining the prefix, reversing intended consumption order in that branch.
2. `backend/main.py:2476–2482` discards all final tails below one second and can reuse the last transcript, without proving those samples were decoded. MT then receives an incomplete source transcript.
3. Current MT receives fragments independently (`main.py:1750`, `backend/streaming/pipeline.py:285–287`). A fixed two-second window or nine-word cap is not a semantic clause boundary. Word order, agreement, negation and named entities can depend on the next words.
4. Three implementations disagree: regular/incremental `/ws`, meeting `StreamingPipeline`, and SDK `Pipeline`; `bp/` adds a divergent copied runtime. Configuration, voice, ordering and Stop differ.
5. Valid WAV durations parse as zero (`session-playback.js:58–74`), defeating the seconds budget/adaptive backlog logic. Green tests use matching reversed FourCC values rather than a standard WAV fixture.

Assumed contribution: these defects plausibly explain part of the reported UI behavior. Unknown contribution: microphone quality/resampling, exact selected route/model, STT model error versus MT context error, and actual audible delay. No fresh recording from the reported UI session was aligned and graded in this review.

The November 6 direct-pipeline report’s 0.39s first output excludes real-time source accumulation; 115 “2s” windows over 115.1s need stride/overlap explanation. RTF<1 means processing throughput, not that the user hears correct speech within that time. Its “trusted/presentation-ready” verdict exceeds its own acknowledged missing WS/physical quality evidence.

## Architecture research and recommendation

Recommendation: consolidate the existing pipeline and preserve source/context first. Keep FastAPI, browser client and engine adapters. A broad rewrite or a larger MT/LLM by default has not earned its complexity. The initial user is a university meeting/presentation operator; reliable words and bounded listening delay outrank increasing the number of advertised routes.

Primary sources opened this session:

- [Whisper-Streaming](https://github.com/ufal/whisper_streaming): explains that naive fixed windows can split words and confirms stable prefixes through LocalAgreement. It now points to SimulStreaming as its successor. This supports auditing the existing local-agreement design, not claiming BP implements it correctly.
- [SimulStreaming](https://github.com/ufal/SimulStreaming): MIT implementation supporting AlignAtt, direct Whisper translation into English and a separate LLM cascade. Its reference full stack targets 1–2 GPUs; do not assume that hardware requirement fits BP’s commodity CPU baseline.
- [CUNI IWSLT 2025 paper](https://aclanthology.org/2025.iwslt-1.41/): evaluates simultaneous Whisper with AlignAtt/context, including CS→EN. This is relevant new architecture evidence for a bounded comparison; its results do not transfer to BP’s microphones or TTS.
- [KInIT Slovak turbo model card](https://huggingface.co/kinit/whisper-large-v3-turbo-sk): Slovak-specific ASR, MIT; warns about other-language degradation and appropriate streaming chunking. Use generic multilingual turbo for Czech evaluation; do not use the Slovak fine-tune as the Czech solution.

Options:

| Option | Benefit | Cost / failure | Recommendation |
|---|---|---|---|
| Change only chunk length/beam/model | Fast experiment | Leaves competing schedulers, dropped tails and no MT context | Diagnostic only |
| One shared streaming core behind WS/SDK/meeting adapters | Correct source ownership, lifecycle, context, traceability | Requires bounded consolidation and regression tests | Recommended first |
| Replace streaming policy/core with SimulStreaming/AlignAtt, or direct Whisper→English | Evidence-backed alternative with CS→EN research | New integration/runtime fit, source-caption tradeoffs, TTS still separate | Isolated comparator after baseline is correct |

Proposed shared flow:

`timestamped audio ring → overlapping contextual ASR → stable source commits → clause/context MT → stable target commits → ordered cancellable TTS → measured listener queue`

Provisional captions can revise. Spoken output must be committed and traceable to source spans. Maintain a revision boundary for unspoken target text, rather than immediately voicing every source fragment. Keep one scheduler, cancellation generation, model lifecycle and route manifest across transports. Preserve audio through silence/short tails until accounted for; never let both decoders consume the same buffer. Bound queued source/work/audio in seconds, with explicit overflow behavior and visible omissions. Keep transcript and translation context bounded, language-aware and evaluated; do not merely concatenate repeated whole translations or speak duplicate prefixes.

### Smallest discriminating evaluation before a model decision

Use the same recordings, source timestamps and bilingual references in every cell. Begin with CS↔EN (current constitution), include SK↔EN as the requested evaluated extension. Reuse current input recordings for reproducibility; reserve held-out recordings by another speaker/domain for acceptance.

1. Gold source text → current MT on full sentences, then the exact production fragments. This isolates MT/context loss.
2. Current ASR → the same MT; compare full-context decoding against current slicing. This isolates ASR/chunking loss.
3. Existing audited CT2 turbo profile → same MT with the same commit policy. Compare quality and true device/time/memory, not model-card claims.
4. If quality still misses, compare the researched simultaneous policy/direct-English path against the corrected cascade. No broad model zoo rerun.

Score source WER/CER plus bilingual meaning: omissions, additions, negation, numbers, names, terminology and grammatical completeness. Compare paired examples, not only averages. Use chrF++/COMET as secondary diagnostics if appropriate tooling is explicitly provisioned; no universal commercial chrF threshold. Voice intelligibility and identity get separate blind listening grades.

Proposed acceptance targets, NOT approved requirements: warm p50 onset→listener first audio <2s; p95 <3s on the selected supported hardware; maximum queued speech 5s with no upward trend over a 10-minute continuous trial; no missing/duplicated source spans and no post-Stop audible output from the old generation. Onset and speech-end clocks must both be recorded and never swapped. Quality/correctness gates take priority over meeting a speed target by omitting speech. Record failures honestly and agree acceptable meaning-error thresholds with the intended listener before promotion. If commodity CPU misses the paired quality/delay acceptance targets, declare the measured capability tier and retain experimental status.

## Cleanup and documentation proposal

Do not delete based on stale indexes. Use a source inventory and import/caller checks first; `backend/hardware.py` remains actively imported despite deprecation advice in PROJECT_INDEX. Preserve raw failed receipts and personal assets. Source snapshots belong with the experiment that created them, not a second importable runtime.

| Priority | Action | Acceptance |
|---|---|---|
| P0 | Reconcile constitution, PRODUCT, README, AGENTS and active specs around default CS↔EN, requested SK scope and cloning requirement | One current requirement ID per claim; historical decisions visibly superseded; no silent scope change |
| P0 | Correct trusted/near-real-time claims and OmniVoice code-vs-weight licensing | Every performance claim names receipt hash, route, device, metric clock and tested scope |
| P1 | Add excluded deterministic regressions to one suite; fix real WAV fixture and source loss before optimization | Same authoritative local/CI gate, standard WAV and source-preservation/Stop tests pass |
| P1 | Consolidate route/model config and the runtime entry point; classify bp mirror | Import graph points to one maintained source; no deletion until ownership/provenance checked |
| P1 | Repair setup/CLI/SDK/Docker instructions against a clean checkout | Every documented primary command exercised without owner assets |
| P2 | Organize current docs as quickstart, operations, architecture, evaluation/evidence, contribution/security/licences; date/archive historical narratives | One maintained landing index, all links resolve; archive pages clearly historical |
| P2 | Separate optional ML/conversion/benchmark extras and exact reproducible GPU profile | Compatible resolved versions, immutable model revisions/hash manifests, no surprise runtime fetch |
| P2 | Move generated logs, nested code snapshots and benchmark artifacts out of ordinary source navigation | Explicit retention/provenance index; failure receipts and private assets preserved |

Recommended authoritative map: README for install/run; PRODUCT plus constitution for requirements; one architecture page for current behavior (targets separate); one evaluation index generated from validated receipts; one operations/security/licence index; dated historical records read through that index. Existing PROJECT_INDEX and documentation/INDEX are not safe as sole cleanup authorities. Add a small check for links, command paths, route declarations and current receipt fingerprints; do not build a documentation platform. A report catalog alone cannot ensure no stale documentation: each active claim must be corrected or marked unknown, then checked against code/receipts. This audit identifies that work; it does not claim the old documents have been repaired.

## What blocks “anybody can clone and start”

1. **Public, permitted default assets.** Fresh install must not depend on Yegor’s personal voices or local ignored CT2 models. Pin each engine/checkpoint/tokenizer/voice revision, hash, licence, attribution, size and download/provision command. Distinguish code, weights, voice dataset and reference-speaker consent. Public generic voice baseline must work before optional personal cloning.
2. **Honest licensing boundary.** At the exact tested OmniVoice revision, the [model card](https://huggingface.co/k2-fsa/OmniVoice/blob/c5fdb5ccb189668d56333f77ba2629f4cd7535f4/README.md) says Apache-2.0 code and CC-BY-NC pretrained weights. General commercial deployment of this checkpoint is not cleared by publishing BP under MIT. Obtain suitable permission or choose/qualify a permitted alternative; retain this checkpoint as an explicitly noncommercial optional research profile if appropriate. The [current Piper project](https://github.com/OHF-Voice/piper1-gpl) is GPL; GPL itself permits commercial use, so “locally only” in the inventory is misleading. Distribution/integration obligations need a compatible release decision, notices and corresponding source where required. This is a factual inventory finding, not legal clearance for the combined product.
3. **Install/runtime split and usable packaging.** Include imported `scripts.language_packs` in Docker; add an explicit provision step, voice/model volumes and DB persistence; qualify CPU/GPU dependency sets and native CUDA libraries. Core runtime must fail clearly without assets and avoid network calls. Current `/api/routes` can trigger `download_model(local_files_only=False)`, contradicting offline claims. `pip install -e .` needs actual package metadata. `make demo-audio` currently only checks arguments.
4. **Secure local/shared modes.** Reject the Compose placeholder JWT secret; install or fail closed on required rate limiter; restrict room ownership/membership, guest capacity and research APIs; stop logging transcript text by default. Use one authorization/retention policy for uploads/metadata/QC. TLS/routing and virtual audio install paths must be documented and actually tested for the supported deployment.
5. **A supported route that retains meaning.** Qualify exact ASR/MT/TTS/voice/hardware/streaming settings with independent references and bilingual listeners. Installed/available/fast never means accepted. Commercial API/server operation and open-source local use have different support burdens; publish a tested support matrix rather than “any language/any machine.”
6. **Actual user journey and meeting delivery.** Current meeting listeners have undrained queues, Stop is bypassed in the direct handler, and voice selection is not passed to cloning. Finish that mode with two-client tests or keep it explicitly unavailable. Validate login/guest→prepare→input/output check→Start→continuous speech→device loss/reconnect→Stop on desktop/mobile browsers and actual chosen audio devices.
7. **Clean-machine release evidence.** One release candidate, one exact repository revision, supported Python versions, no warm caches/personal files. Provision public defaults, run one audio→translation→WAV smoke with a portable fixture, verify offline runtime, then exercise Windows/macOS/Linux claims in clean environments. Back up/restore accounts/voices, upgrade/restart/recovery instructions and known limits. Docker hardware health is not route readiness.

LHC remains a separate integration acceptance: actual room mic/return channels, conference software, processing host/OS, permissions and named acceptance owner are not established by generic virtual-audio support. No new LHC facts were supplied here.

## Three ordered work packages

1. **Correctness and claim baseline — maintainer; proposal cap 1–2 focused days.** Preserve source spans, one scheduler, real WAV queue budget, Stop generation, include excluded tests, correct false current claims. Dependency: freeze the audited dirty snapshot into a reviewable change set. Accept: deterministic regressions and primary suite pass, no current claim contradicts receipt clocks or licences.
2. **Paired accuracy experiment — maintainer + bilingual listener; proposal cap one bounded free-T4 experiment after baseline.** Reuse current recordings/manifest and existing turbo evidence; oracle-MT/context/ASR ablations. Accept: paired meaning grades and onset/end/listener p50/p95, source coverage, queue trend, exact device/config/runtime and output hashes; select one route or retain experimental status. No new job was submitted by this review.
3. **Self-host release candidate — maintainer; proposal cap 2–3 focused days after route/asset selection.** Repair reproducible public provisioning, licensing inventory, local/shared security and supported-OS smoke. Accept: an uninvolved tester starts one qualified route from a clean checkout using only public permitted assets; restore/Stop/device recovery passes. These are planning estimates, not delivery promises; open licence/hardware decisions can extend them.

Roast: the cheapest failure is fluent, fast, wrong speech that drops a short negation while green tests and a “trusted” document conceal it. Smallest correction: make source preservation and bilingual meaning acceptance harder to bypass than a speed demonstration.

suggest: every voiced segment should carry a trace ID, source sample span, committed source text, target text, selected model/config fingerprints, output WAV hash, server emission time and browser playing time. That single trace resolves the current ambiguity between recognition error, translation error, queue delay and evidence drift.

## Final verification addendum

- Additional controlled omitted-suite check: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest test/engine_registry_tests.py test/i18n_dict_test.py tests/bp_low_latency_profile_test.py -q --tb=short --show-capture=no -p no:cacheprovider` → **1 failed, 18 passed**, 0.24s. `test_every_live_tts_engine_is_registered` catches missing `piper_generic_de` / `piper_generic_fr` manifest entries. This further disproves interpreting the canonical 87 green tests as all tests green.
- Independent Node Stop→Start probe with deferred `blob.arrayBuffer` → **old_blob_plays_after_stop_restart=1**. `enqueue` checks active before await but does not bind the pending blob to its originating generation; an old session can play in the next session. Require generation validation after every await, including blob parsing.
- Backend review completed all 28 first-party backend Python files and all 34 test Python/CJS files, including archives. Archived voice-management fixtures contain destructive teardown of all user-1 voices and pre-existing dummy audio: do not run broad `pytest` discovery until isolated. Tests chosen above did not execute archives.
- Workflow/bootstrap/vendor templates under `.specify/` and generated `.claude/skills/speckit-*` are third-party development scaffolding, outside runtime source review. Vendored/minified source and binary assets remain explicit exclusions, not a claim of all disk bytes semantically audited.
- `session_retro.py --today` ran; generated proposals concern other recorded sessions. Relevant practices applied here: gather named BP/OmniVoice terms, load skills, use primary web sources, preserve benchmark scope and keep detailed findings in these reports. No unrelated configuration/ledger fixes were made.

## Independent exact-report review

The code reviewer reread this final report and returned a bounded PASS for recommendations and evidence separation. One refinement adopted: predefine aligned voiced source units/spans and the first target audio carrying their meaning, including long clauses; report source-word/semantic-unit latency as well as per-emission timing. Compare matched source units across profiles, never use different emission counts to imply paired utterance improvement. Keep speech-onset and speech-end metrics distinct and report omitted/unvoiced units as failures rather than dropping them from latency denominators.

Tidy check ran: zero Orca terminals, no 20128/20129 listener output, load 2.38/2.90/2.89. System ps/swap reads were permission-restricted, so no global orphan/swap-clean assertion. This audit opened no browser tabs/servers or ongoing inference jobs; all short command sessions completed. Reports and evidence JSON are intentional retained artifacts.

Source-coverage closeout: documentation/UI reviewer completed the remaining full HTML markup and CSS source reads after the initial text/scan pass; its coverage ledger records the final status. Visual/browser/device acceptance remains untested. Vendor asset distribution notices are also overdue: font CSS claims SIL OFL provenance but bundled full notices are absent from the inspected vendor directory; Chart header names MIT without the full notice, and Material Symbols lacks sufficient pinned provenance. Reconcile these with the binary-asset manifest before distribution; no font incompatibility is inferred.

Vendor integration closeout: parent subsequently read all three vendored Python files in full (`line_packet.py` 93 lines, `silero_vad_iterator.py` 150, `whisper_online.py` 970) plus the captions wrapper. This supersedes the earlier vendor-Python source-read exclusion, but not third-party dependency/licence clearance. The active caption wrapper supplies local ASR and local Silero assets; standalone vendor alternatives still expose downloading/cloud backends and are outside the supported default. Its rolling 200-character source prompt already demonstrates a reusable bounded-context mechanism; reuse/integration requires the same source-span and lifecycle tests as the main route. Vendor minified Chart JS, binary assets and installed third-party dependencies remain excluded from semantic source audit.

Final artifact acceptance: parent independently rehashed all 189 scripts/specs/packs/mirror ledger rows: 0 mismatches. Documentation/UI ledger has 138 FULL rows. All six local links in the main review resolve; all three new audit JSON files parse. Scripts report also records 12/12 pure Canary/replay tests and a reproduced pre-inference CLI parser failure. New reports form the review index; existing stale documentation still requires the listed correction work.
