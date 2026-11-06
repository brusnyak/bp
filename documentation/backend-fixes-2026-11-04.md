# Backend repair receipt — 2026-11-04

Scope: bounded Codex backend worker; no delegation, installs, models, GPU execution, server process, commit, push or deployment. Existing dirty edits were preserved. Parent authorized expanding the packet to tokenizer/captioner and existing auth/API tests.

## Changed behavior

- `/ws?token=<JWT>` validates identity before creating a session or loading models. Missing/expired credentials receive a 4003 close (accepted solely to convey that code); invalid browser Origin is denied. Valid connection reports not-ready; explicit configuration initializes assets. HTTP `/initialize` requires a bearer user and uses a per-user preparation key, without broadcasting its configuration to other clients.
- Main live STT selects CTranslate2 CPU/CUDA capability, never Torch MPS. Missing STT resolves local cache only; missing MT model/tokenizer fails with setup-required HTTP 409. MT tokenizer loading uses `local_files_only=True`.
- Streaming partial-caption engine is opt-in (`BP_ENABLE_STREAMING_CAPTIONS=1`), requires installed base STT plus `BP_SILERO_VAD_DIR` pointing to a local Silero checkout. Its VAD injection uses `torch.hub.load(..., source='local')`; default live path does not construct this optional model. Final translation captions remain in the main path.
- Automatic Piper selects English for EN, personal/fallback Slovak for SK, Czech for CS. `cs-en.json` now selects the English voice. SDK chooses the target engine too and pools STT beam settings separately; closing releases matching pool identities. Source-language switches reset the default STT request rather than retaining the previous checkpoint.
- Stop cancels tracked pipeline tasks, clears capture buffers and suppresses post-stop/previous-generation sends. Subsequent bytes are ignored until Start. Disconnect cancels leftover work, releases model references and closes session log handles. Native inference already executing in a worker thread cannot be force-killed; output is suppressed.
- `/api/routes` returns per-pack installed assets and experimental route status without loading models. A model environment override does not promote quality. WS initialization emits `route_status`; every actual live route stays experimental pending matching live evidence.
- Research manifest `/ui/voice-lab/library.json`, QC and processed media URLs are denied in the web app. Speaker audio requires ownership with bearer/query JWT, or explicit metadata `public_demo: true`. `user_id: null` is not consent. Owned voice-reference selection is checked on HTTP/WS configuration, not just on preview. Existing UI shell remains public. No personal source clips were moved, deleted or relabelled public.
- CLI evaluator now validates target voice and local voice config, resolves converted HF-named MT paths, records MT timing plus stage sum and inference wall time, and records STT-only evidence without granting a full-route pass. Legacy WER passes cannot verify routes. Full-route records require current pack/model/runtime/code fingerprints, reviewed STT/MT/TTS/live/policy decisions, matching reference hash and matching output audio checksum. No existing receipts were promoted.

## Checks

`.venv/bin/python -m pytest test/readiness_regression_test.py test/config_tests.py test/security_tests.py test/backend_auth_tests.py test/language_evidence_test.py -k 'not hf_model_info_returns_tuple' -q`

Result: **46 passed, 1 deselected**. Deselected test makes an external HF request. Tests use isolated/mocked assets and in-memory identities; no real ML inference. Starlette/httpx and pytest-asyncio deprecation warnings remain.

`python -m py_compile` passed for app, main, registry, facade, evaluator, tokenizer wrapper, streaming caption wrapper and injected VAD adapter. `git diff --check` passed. `test/backend_api_tests.py` live smoke expected session key updated to authenticated user scope; its real-model path was **not run by this worker** (parent owns full demo gates).

## Limits and follow-up

No real browser, microphone, playback, conference routing, cloned-voice quality, full-load or remote Kaggle test ran here. Private research media is intentionally unavailable through the shared app until a sanitized public demo/export or a proper research authorization model is provided. CLI component resolution retains its historical fetchable-size semantics; `/api/routes` overrides STT availability with local-only checks. SDK STT also resolves local assets/cache only; its no-model pool test uses a stubbed installed source.

Roast: corrected configuration and passing mocked tests prove the repaired contracts, not the complete LHC meeting product. Cheapest failure remains real audio routing/capture under continuous speech; smallest next check is the parent's authenticated browser sound-check → continuous use → Stop rehearsal on installed assets.

## Follow-up: real-model VAD gate repaired

Parent's first full suite found three VAD failures: the old harness fed bytes without the newly required Start, fed trailing silence instantaneously despite a wall-clock VAD close timer, and used disconnect as an implicit final transcription flush. Production now treats disconnect/Stop as cancellation.

`test/vad_tests.py` now emits Start, sends PCM at its actual frame duration, and waits for normally VAD-triggered pipeline tasks while the mock client remains connected. Original transcript/translation assertions remain unchanged. The auth fixture now uses in-memory SQLite rather than writing a test identity into the owner's default database. Removed obsolete DIAG3 per-chunk WARNING output and downgraded normal all-zero VAD frame notices to DEBUG. `hf_model_info` now honors `HF_HUB_OFFLINE=1` before stdlib urllib so the offline suite cannot bypass the flag.

Fresh checks:

- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest test/vad_tests.py -q --tb=short --show-capture=no`: **4 passed**, 23.80 seconds, 3 deprecation warnings.
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTEST_ADDOPTS='--tb=short --show-capture=no' make test`: **68 passed**, 27.63 seconds, 3 deprecation warnings. This includes the parent's expanded Makefile, real local VAD/STT/MT/TTS tests, 17 readiness regressions and language-evidence tests. No test skipped/deselected in this run; no model download.

These are local CPU file-driven tests, not a browser/device/room audio or cloned-voice commercial acceptance run.
