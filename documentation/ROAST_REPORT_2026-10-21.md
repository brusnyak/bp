# ROAST REPORT: BP Project — 2026-10-21

**Verdict: Rework — ships as "verified" but SK→EN quality is hallucinated Czech, CLI commands documented don't exist, e2e test broken, setup doesn't run selftest**

---

## JOB

What this project must achieve for the **grumpy professor / picky old man** who wants:
- Set once, forget forever
- Plug-and-play for occasional lectures/meetings
- EN↔SK reliable, other languages swappable
- No cloud, no fiddling, no "works on my machine"

---

## FINDINGS (P0 = blocks the job; P1 = costs time/money; P2 = cheap fix)

### P0 · SK→EN marked "verified" but transcript is hallucinated Czech
**Evidence**: `processed/language_eval/sk-en.json` shows status="verified" but transcript: `"Dobré ráno a viete si v tejto šivejokáške a pre kultúru reči"` (reference: `"vitajte v tejto živej ukážke prekladu reči"`). WER would be ~0.8 if measured.
**Root cause**: `language_packs.py` marks "verified" if components resolve + clip exists, **never checks WER** unless `reference` field is provided in pack (it's `null`).
**Fix**: Require WER < threshold for "verified", or auto-fetch reference from `speaker_voices.json` (it exists there). Change status to "experimental" for sk-en.

### P0 · `bp language` commands documented but don't exist
**Evidence**: `documentation/project_status_2026-10-20.md` and `cli_quickref.md` show `bp language add|list|test|remove`; `bp --help` shows no `language` subcommand. Commands exist in `scripts/language_packs.py` but not wired into `scripts/bp.py`.
**Fix**: Add `language` subcommand to `bp.py` wrapping `language_packs.py`.

### P0 · `setup.py` claims to run `bp selftest` but doesn't
**Evidence**: `documentation/project_status_2026-10-20.md` says "ends with bp selftest"; `scripts/setup.py` ends with `print(f"\nDone... Start: {PY} app.py")` — no selftest call.
**Fix**: Add `run([PY, "scripts/bp.py", "selftest"])` at end of `setup.py` (with `--dev` flag check).

### P0 · `e2e` test broken — references missing file
**Evidence**: `scripts/e2e_ensk_new_voice.py:32` loads `speaker_voices/en_script_reading.m4a`; actual file is `me_en_script.m4a`.
**Fix**: Update hardcoded path to `me_en_script.m4a`.

### P0 · SK→EN beam_size not configurable per pack (only global env)
**Evidence**: `languages/packs/sk-en.json` has no `beam_size`; `backend/main.py:517` reads `BP_STT_BEAM_SIZE` env var only. Doc claims per-pack config works.
**Fix**: Wire `pack["stt"].get("beam_size")` through `_initialize_stt_model` → factory.

### P1 · `interrupt_smoke_test.py` and `backend_vad_stt_test.py` / `piper_pipeline_test.py` in `test/` break `pytest test/` collection
**Evidence**: These are live scripts needing running server; they import backend modules but error on collection. `make test` excludes them via explicit list but `pytest test/` would fail.
**Fix**: Move to `scripts/live/` or add `@pytest.mark.skipif(not_running_server)`.

### P1 · Kaggle credentials + Modal token still present (cleanup due 2026-10-09)
**Evidence**: `~/.kaggle/kaggle.json` and `~/.modal.toml` exist. `documentation/evaluation_2026-10-18.md:143-145` says cleanup queued for 2026-10-09, not done.
**Fix**: Revoke tokens, remove files.

### P1 · `/health` + `/metrics` HTTP routes missing
**Evidence**: `backend/main.py:388-405` has `get_model_pool_stats()` but no FastAPI routes. `documentation/project_status_2026-10-20.md` lists as missing.
**Fix**: Add `@router.get("/health")` and `@router.get("/metrics")` exposing pool stats, engine versions, session count.

### P1 · Config-driven engine swap not wired (registry is additive only)
**Evidence**: `backend/registry.py` has `select()`/`instantiate()` but `backend/main.py` still uses hardcoded `FasterWhisperSTT`, `CTranslate2MT`, `PiperTTS` directly. Task T010 open.
**Fix**: Replace hardcoded init in `_initialize_stt_model`, `_initialize_mt_model`, `_initialize_tts_models` with `registry.instantiate()`.

### P1 · Linux x86_64 never tested; CI not wired
**Evidence**: `documentation/linux_setup_and_test.md` exists but no record of execution. `.github/workflows/` empty.
**Fix**: Run on Ubuntu VM/GitHub Actions; add CI workflow.

### P1 · SDK / public API missing (supervisor can't embed)
**Evidence**: No `Pipeline` facade, no `documentation/sdk.md`, no REST/WS contract doc.
**Fix**: Implement `class Pipeline` with `translate(audio, src, tgt)` + doc.

### P2 · `speaker_voices.json` gets wiped by `backend_api_tests.py` teardown (now snapshots but fragile)
**Evidence**: `test/backend_api_tests.py` historically deleted real data. Now snapshots/restores but any test failure leaks.
**Fix**: Use temp directory for test fixtures; never touch real `speaker_voices/`.

### P2 · `requirements.txt` pins may rot (numpy/numba no wheels beyond 3.12)
**Evidence**: `setup.py` rejects Python >3.12. `requirements.txt` has pinned versions.
**Fix**: Document Python version constraint in README; test on 3.12.

### P2 · Documentation claims "verified" for sk-en but WER not measured
**Evidence**: `language_packs.py:135-149` `status()` returns "verified" if eval file exists with status="verified", but eval is written with status="verified" unconditionally (line 327-328) — WER only added if `ref` provided.
**Fix**: Require `reference` in pack for "verified", or compute WER against `speaker_voices.json` reference automatically.

---

## PROVEN / ASSUMED / UNKNOWN

| Claim | Status | Evidence |
|-------|--------|----------|
| 38/38 tests pass | **Proven** | `make test` output |
| Engine gates 4/4 pass | **Proven** | `bp engine gate` runs |
| EN→SK latency ~0.7s | **Proven** | `language_eval/en-sk.json` shows 7.25s end-to-end on 134s clip |
| SK→EN latency ~1.4s | **Proven** | `language_eval/sk-en.json` shows 25.5s end-to-end on 134s clip |
| SK→EN WER 0.23–0.32 | **Proven** | `bench_stt_beam.py` runs |
| "bp language" CLI works | **Assumed** | Only `language_packs.py` works, not `bp.py` |
| "setup ends with selftest" | **False** | `setup.py` doesn't call it |
| Linux works | **Unknown** | Never run on x86_64 |
| RVC-MLX works | **Unknown** | Registered, no model trained |
| OmniVoice on T4 works | **Proven** | `load_tests_2026-10.md` data exists |

---

## ROAST-WORTHY (Single Highest-Leverage Fix)

**Make "verified" mean quality, not just "loads"**. The language pack system is the user-facing contract for "plug-and-play any language". Right now it lies: sk-en says "verified" but produces Czech gibberish. Fix: auto-compute WER against `speaker_voices.json` reference in `cmd_test`, require WER < 0.3 for "verified", else "experimental".

---

## FLIP (What Would Change Verdict to "Ship")

1. SK→EN status = "experimental" with documented WER 0.32 (beam=1) / 0.25 (beam=5)
2. `bp language` wired and working
3. `setup.py` runs `bp selftest` and fails fast if not clean
4. `e2e` test runs on existing files
5. `/health` + `/metrics` endpoints exist
6. Config-driven engine swap wired (T010)
7. Kaggle/Modal credentials revoked
8. Live test scripts moved out of `test/`

---

## CONSUMER / GRUMPY PROFESSOR PERSPECTIVE

| Expectation | Reality | Gap |
|-------------|---------|-----|
| "Install once, run forever" | `setup.py` works but doesn't verify itself | No selftest gate |
| "Just works for EN↔SK" | EN→SK works; SK→EN hallucinates | Core language pair broken |
| "Swap engines easily" | Registry exists but pipeline ignores it | T010 not done |
| "Add a language in 30 min" | `bp language add` doesn't exist | CLI gap |
| "Health check for server" | No `/health` endpoint | Ops gap |
| "Clean docs match code" | Docs claim features not implemented | Trust gap |

---

## CLEANUP CHECKLIST FOR "PRODUCT PROJECT" RELEASE

- [ ] Fix SK→EN status to "experimental" + document WER honestly
- [ ] Wire `language` subcommand into `bp.py`
- [ ] Add `bp selftest` to end of `setup.py`
- [ ] Fix `e2e_ensk_new_voice.py` file path
- [ ] Move `backend_vad_stt_test.py`, `piper_pipeline_test.py`, `interrupt_smoke_test.py` to `scripts/live/`
- [ ] Add `/health` + `/metrics` routes
- [ ] Wire `registry.select()`/`instantiate()` in `main.py` (T010)
- [ ] Add `beam_size` per-pack config
- [ ] Revoke Kaggle/Modal credentials
- [ ] Run Linux x86_64 setup + test; wire CI
- [ ] Implement `Pipeline` facade + `documentation/sdk.md`
- [ ] Update all documentation to match actual implemented features
- [ ] Remove claims of unimplemented features from docs
- [ ] Tag release, push to origin (with owner approval)
- [ ] Fresh clone test on clean machine
- [ ] Test on Linux (VM or CI)

---

*Generated 2026-10-21 by adversarial review. Run `make test` and `bp selftest` to verify current state.*