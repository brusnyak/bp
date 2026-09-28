# BP Dev Plan — where we are, what's ahead

Branch: `main` + `merge/windows-amd-cpu` (2026-09-28, verified: Windows CPU-setup work
merged, `main` fast-forwards onto it). Two machines: Mac M1 Pro (reference) and the
Windows 11 AMD CPU-only laptop. Docs: `documentation/voice_and_app_direction_2026-09.md`,
`documentation/demo_runbook_2026-09-28.md`, `AGENTS.md`.
Thesis rules: faculty guide + §4.1 AI rules → `Deklarácia k využitiu UI` skeleton lives in
`documentation/thesis_draft.md` (one `[DOPLNIŤ]` marker left).

## Done

- [x] Core pipeline STT→MT→TTS, non-blocking + barge-in (T043–T047 — **unpushed**)
- [x] Hybrid fast cloning RTF ~0.13, cross-lingual (measured 2026-09-27)
- [x] Voice Lab static page: 2 new voices / 4-8 QC, plan panel, `--no-test`, no-store fetch
- [x] Captions in live UI — subtitle strip + PiP pop-out (spec 002)
- [x] Google login — GIS + JWT, verified live 2026-09-27 (`POST /api/auth/google 200`, user id 4)
- [x] Real JWT (HS256) replaces mock tokens; `JWT_SECRET` in local `.env`
- [x] Your recordings — EN (115s) + SK (134s) landed 2026-09-27, old clips archived
- [x] Session JSONL logging (`processed/sessions/`)
- [x] Single `bp` CLI (`corpus/qc/stt/e2e/library/script`)
- [x] PWA installable: manifest on all pages, maskable icons, shell v2, apple-touch-icon
- [x] Windows/AMD CPU-only port merged (2026-09-28): guarded Coqui import, engine-name
  write-back on the SK remap, `logging` instead of `print` on the MT hot path, UTF-8
  session logs, ctranslate2/transformers dtype compat, stale-test cleanup,
  `requirements-windows.txt` + `SETUP_WINDOWS.md` (16/16 green on stock Windows 11, e2e ~3.6s)
- [x] Demo readiness (2026-09-28): `make demo-check` pre-flight, runbook, clean WS teardown,
  fallback clip committed to git (was Mac-only)

## Measured findings 2026-09-28 (merge + demo verification, M1 Pro)

- Merged the Windows/AMD work into the main line: only `ui/voice-lab/library.json`
  (generated) conflicted; resolved by re-running the generator, so the committed manifest
  lists git-tracked assets only.
- **The merge fixed a live SK-output bug on every machine**: the `piper` →
  `piper_sk_personal` remap never wrote the effective engine name back into
  `session_config`, so `tts_engine_name != session_config["tts_model_choice"]` and both
  lookups returned `None` — translation text arrived, synthesis silently never ran.
  `test/backend_api_tests.py::test_initialize_full_pipeline_live` now pins it.
- Live WS rehearsal (`test/interrupt_smoke_test.py`): 2 utterances + barge-in →
  9 translations, 5 partial captions, 228 TTS audio chunks, final metrics
  **STT 0.51s / MT 0.10s / TTS 0.17s / total 0.78s**. Teardown: one INFO line, no ERROR.
- Test suite green: hardware 7 + VAD 4 + auth 6 + API 3 = **20 passed**, plus
  `test/piper_pipeline_test.py` (exit 0). Two defects fixed while verifying:
  `"xtts" not in engines` was a Windows-only assertion (macOS has Coqui — now
  conditional), and the API-test fixture used to wipe `speaker_voices.json` + `*.wav`
  (must stay snapshot/restore).
- Rehearsal artifact for the honest-gaps slide: MT reads literally on a misheard word
  ("into Slovak" → "do pomalého wacku") — Opus-MT behaviour on a bad transcript, not a
  new bug.

## Measured findings 2026-09-27 (evidence, not vibes)

- STT (10-row matrix): EN base 0.077 / small 0.096; **Parakeet-TDT-v3 EN 0.023** (adopt path).
  SK small 0.49 plain (keeps SK side); Parakeet SK 0.89 via transformers (no lang
  conditioning — reject this path); base+auto 1.0; Czech-proxy worse. → per-language
  STT routing: Parakeet EN / whisper-small SK.
- MT: chunked batch 18 sentences in 0.50s (3.5x faster than single-shot) + fixes
  long-input truncation; 20-word fallback for unpunctuated Parakeet output (0.11s/chunk).
- E2E EN→SK: STT 0.5s + MT 1.8s + hybrid TTS 7.1s (53s audio, RTF 0.13), wall 18.7s.
  Audible proof: `processed/e2e_ensk_sk.wav`.
- Voice: `sk_SK-personal` fine-tuned (warmstart lili, 18 sentence clips, val_mel
  0.57→0.29). Machine-listen: personal SK F0-range ~half of generic lili
  (11–13Hz vs 22Hz) = "robotic" quantified; no clipping anywhere. Overnight
  2500-step run approved → judge by ear in lab.
- Research: no local SK TTS alternative exists (Piper/lili is the only SK base);
  XTTS/CosyVoice/F5 all SK-less or CUDA-bound; Piper TRAINING.md wants ~1000
  epochs for fine-tune (we ran ~200 — undertraining explains robotic).

## Now: Accelerating SK→EN Turnaround Latency (Target: close the 2x asymmetry)

Current measured baseline from 2-speaker simulation (`scripts/demo_conversation.py`):
- **EN→SK turn latency**: ~0.67s – 0.87s (STT ~0.55s, MT ~0.08s, TTS ~0.15s)
- **SK→EN turn latency**: ~1.27s – 1.64s (STT ~1.13s – 1.43s, MT ~0.08s – 0.11s, TTS ~0.07s – 0.14s)
- **Bottleneck**: MT and TTS are virtually identical in speed (~0.2s combined). The entire asymmetry is inside **Slovak STT** (`whisper-small-sk` at ~1.3s vs English `base` at ~0.55s).

Next action items to explore and benchmark:
1. **Beam size tuning on FasterWhisper for Slovak**:
   - `beam_size=5` (default) vs `beam_size=1` (greedy) or `beam_size=2` / `best_of=1`.
   - On CTranslate2 / Whisper, greedy decoding (`beam_size=1`) can cut inference time by 30–50% while often retaining >95% accuracy on domain/colloquial speech.
   - Run offline test with `scripts/eval_sk_direction.py` comparing `beam_size=1,2,5` on `small-sk` to verify WER impact vs latency gain.
2. **Compute type & thread parallelism**:
   - Verify `compute_type="int8"` vs `int8_float16` / `float32` and `cpu_threads` settings on Apple Silicon / CPU runtime.
3. **Conditioning / Prompting (`initial_prompt`)**:
   - Supplying a short prompt with Slovak diacritics / context helps greedy decoding converge accurately without needing wide beam search.
4. **VAD chunking optimization**:
   - Tighter silence thresholds for Slovak turn completion to reduce trailing audio padded into STT.

## Next

- [ ] Land the Windows branch when it arrives (`win/*` → gates → ff `main`; protocol in
      `AGENTS.md`), then execute the footprint diet — ranked cut list with measured savings
      in `documentation/footprint_audit_2026-09-28.md` (biggest: core/extras requirements
      split ≈ −1.5 GB venv −1.6 GB browsers; `ct2_models/` out of git −225 MB)
- [ ] Linux pass on a third machine → fill `documentation/linux_setup_and_test.md` (setup,
      6-gate ladder, record sheet, unknowns)
- [ ] Thesis numbers corrected to measured + `[DOPLNIŤ]` marker filled
- [ ] Stage-demo dry run on the real machine: runbook → `documentation/demo_runbook_2026-09-28.md`
      (pre-flight, script, fallback ladder, timings); checklist → `documentation/monday_test_checklist.md`
- [ ] Delete `test/full_pipeline_test.py` (dead F5 leftover, breaks `pytest test/` collection)
- [ ] Env raw-test on clean checkout (this Mac + Windows laptop) — Windows half now done via
      `SETUP_WINDOWS.md`; Mac half parked until the project is complete

## Run things

| Command | What |
|---|---|
| `make lab` | Voice Lab review page, static only (no login/upload — those need `make run`) |
| `make run` | Full backend (https://localhost:8000) |
| `make demo-check` | Demo pre-flight (assets + live server; `scripts/demo_preflight.py --server`) |
| `make test` | Backend suite: piper pipeline + VAD + hardware + auth + API tests (20 pass, 2026-09-28) |
| `venv/bin/python test/interrupt_smoke_test.py` | Live WS rehearsal over `/ws` (needs `make run`) |
| `python3 scripts/update_voice_lab_library.py --no-test` | Refresh Voice Lab manifest |
| `venv/bin/python scripts/voice_similarity_qc.py --synthesize-only` | Synthesize QC candidates |
| `.venv-stt/bin/python scripts/stt_parakeet_spike.py --clip en\|sk` | Parakeet spike (separate venv) |
| `venv/bin/python scripts/pipeline_latency_probe.py` | First-output latency probe |
| `venv/bin/python scripts/machine_listen_qc.py` | Machine listening panel (WER thirds + acoustic health) |
