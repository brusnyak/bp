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
   - Note 2026-09-29: `FasterWhisperSTT` already accepts `beam_size` (uncommitted); `scripts/bp.py demo-audio`
     exposes `--beam-size` (default 1). Live default still 5 — change only after the WER check above.
2. **Compute type & thread parallelism**:
   - Verify `compute_type="int8"` vs `int8_float16` / `float32` and `cpu_threads` settings on Apple Silicon / CPU runtime.
3. **Conditioning / Prompting (`initial_prompt`)**:
   - Supplying a short prompt with Slovak diacritics / context helps greedy decoding converge accurately without needing wide beam search.
4. **VAD chunking optimization**:
   - Tighter silence thresholds for Slovak turn completion to reduce trailing audio padded into STT.

## Measured 2026-09-29 (STT input control + zero-shot clones + Lab showcase)

- **STT input control** (`processed/stt_input_test/clean_synth_matrix.json`): same 200-char SK passage
  synthesized by Piper (known-perfect text), then recognized. Generic-voice input: base WER 0.59,
  small-sk **0.16** (vs real-mic 0.65–0.71 / 0.23–0.35). Personal-voice input: base 0.78, small-sk **0.57** —
  the fine-tuned voice itself is hard to recognize (links the "robotic voice" and "SK STT" issues:
  personal-voice audio fed back to STT collapses both rungs). Bonus datum: personal speaks ~50% slower
  (17.3s vs 11.5s for the same text).
- **Zero-shot clones** (`processed/omnivoice/v2b_head_*_sk.wav`): OmniVoice CPU, refs trimmed to clean
  ~13–14s heads of the new v2b recordings at sentence boundaries. SK→SK RTF 2.9, EN→SK cross-lingual
  RTF 2.8 (num-step 16) — ~3× slower than real time vs Piper streaming; blind A/B vs Piper still open.
- **Lab showcase**: `scripts/build_demo_charts.py` (system-python matplotlib, not a repo dep) renders
  `ui/voice-lab/charts/` from measured JSONs; `library.json` gained `charts`, `stt_input`, `zeroshot`,
  `demo_audio` sections with matching `lab.js` renderers. Gates 2026-09-29: `make test` 33 passed,
  lab serves all 8 sections.
- **Showcase curation (owner review 2026-09-29)**: QC visible = SK A/B pair + `enpers_v2` +
  `male_alldata_denoised`; default-voice renders and superseded fine-tunes hidden from the page but kept
  with numbers. Zero-shot section shows only the 2 v2b clones (legacy hello/thirty_second + `ref_*` heads
  excluded). Voices section features the 3 v2b takes; `sk_direction` source audio behind a toggle.
  New charts: `timeline_meeting.png` (per-turn meeting simulation), `synth_by_length.png` (Piper RTF flat
  ~0.04 at 7→32s output, both voices; personal paces ~50% slower).
- **OmniVoice MPS verdict**: works on torch 2.14 MPS (old "unavailable" comment was stale) but RTF only
  2.9→2.5 — autoregressive-bound, not compute-bound. Long-term strategy stands: bulk-generate HQ audio
  offline with OmniVoice, then train a fast Piper voice (NOT another jirka fine-tune).
- **Conference playback voices**: `demo_conversation.py` used generic both directions, `conversation_sim.py`
  personal-SK + generic-EN — the reported trembling/generic-voice observations were by construction.
  Fixed 2026-09-29: personal voices both directions (`en_US-personal-v2` verified present and working);
  `processed/conversation/*` (4 wavs) + `conversation_sim.json` (36 rows + `agg`) regenerated, 3 stale
  generic-voiced orphans removed (backups in /tmp). Charts/labels adapted: new schema records STT only as
  batch direction totals, so STT bars are honestly-labeled means (EN incl. model load). Note: the sim's SK
  STT rung is generic `small`, not live's `small-sk` — open inconsistency.
- **STT input length effect (owner hypothesis confirmed) 2026-09-29** (`processed/synth_lengths/stt_by_length.json`,
  clean generic-voice synth): base WER 0.82 (7s) → 0.65 (17s) → 0.55 (24s); small-sk 0.36 → 0.32 → 0.28.
  Short clips starve Whisper of context — 7s input is measurably worst for both rungs.
- **Scripted meeting sim** (`scripts/meeting_sim.py` → `processed/meeting/`): 4 alternating turns read by the
  personal voices through real STT→MT→TTS on one 52s meeting clock (EN→SK compute ~1s/turn, SK→EN ~2–2.5s,
  STT-bound). Lab bottom section `meeting` (4 turn cards with script/heard/translation + playback audio) with
  `gantt_meeting.png` (2 speaker lanes × time: speech, STT/MT/TTS, cross-lane playback). Honest-gap material:
  SK→EN turns misheard personal-voice SK badly (e.g. "bez cloudu" → "without the reindeer").
- **Engine A/B, decisive (2026-09-29)** (`scripts/engine_ab.py` → `processed/engine_ab/matrix.json`,
  same 200-char SK text): synth RTF piper 0.04 / omni-generic 0.83 / omni-zeroshot 1.21 (MPS);
  small-sk WER omni-zeroshot **0.054** / omni-generic 0.162 / piper-generic 0.189 / piper-personal **0.540**.
  STT hears the cloned voice near-perfectly — the SK→EN mishearings are definitively the Piper personal
  voice's acoustics, not the recognizer. Further fine-tuning of that lineage is spent effort; the corpus
  for a fresh voice should come from OmniVoice bulk generation.
- **Live fixes (2026-09-29, `make test` 33 green)**: PiP waits for video metadata before
  `requestPictureInPicture` (fixes the reported `InvalidStateError`); output dropdown shares the TTS-select
  style; record modal cut to en/sk/cs with tabs+select synced (removed stale XTTS list, malformed tags, and
  a simulated consent-translator that fought the passage tabs); sidebar voice rows gained a Play button
  backed by new `GET /voices/file` (auth-aware, `_voice_path`-guarded). Speaker/headphone routing still
  needs the owner's console lines — the `setSinkId` path looks right.
- **Console-log round (2026-09-29)**: PiP root cause was a blank canvas (no frame = no metadata) — `drawPip()`
  now paints before capture, script cache-busted; outputs re-enumerate after mic grant (pre-permission
  deviceIds are empty); sinkId mismatch falls back to default with a notice; `play()` failures notify;
  voice-list filename falls back to `basename(path)`, Play disabled when no file.
- **Live TTS-voice bug, fixed (2026-09-29)**: `config_update` re-initialized models before writing the
  new direction into `session_config`, so the `piper→piper_sk_personal` remap read a stale `target_lang="sk"`
  and EN targets kept the Slovak voice (measured: English translations read by `piper_sk_personal` over /ws).
  Direction now lands before init; source-change notification compares against the saved old value.
  This was the backend half of the reported "SK→EN bad quality".
- **Label==engine invariant (2026-09-29)**: the first attempt left EN targets silent (label stuck at the
  remapped name while a different engine initialized → pipeline lookup mismatch → zero TTS bytes, masked
  in one run by stale byte files). `_initialize_tts_models` now always writes back the effective choice;
  `meeting_live.py` passes explicit per-direction engines and deletes stale dumps/raws at start. Verified
  in server log: SK→EN inits `piper_personal_v2`, EN→SK `piper_sk_personal`.
- **Turn-2 VAD anomaly (open, measured 3×)**: one EN turn's transcript arrives ~18s after speech end
  (+24.5s for 6.1s audio; clip tail is non-silent, RMS 0.057). VAD-finalization tuning question for the
  honest-gaps slide — shown as-measured in the Gantt, not smoothed over.
- **VAD close mechanism (researched 2026-09-29, fix queued)**: segments close 0.3s after the last
  webrtcvad-voiced frame (`SILENCE_TIMEOUT`, aggr. 3). EN personal-v2 clips carry a ~5× higher noise
  floor with few silent frames, so frames keep reading voiced and close only happens at teardown flush.
  Confounder: local Piper synth during sims contends CPU with server STT (wall-clock inflation).
  Research plan (after bulk): controlled stream with no local load × trailing-silence sweep, VAD
  aggressiveness/energy-gate sweep, per-segment voiced-ratio logging. Live demo remains primary.
- **Streaming meeting rework (2026-09-29)**: `meeting_sim.py` now uses the streaming-first-chunk model
  (STT streams during speech, only a stated 0.15s tail assumed; MT first-word and TTS first-chunk measured):
  first translated word 58–136ms, first audio 0.16–0.25s, playback ~0.4–0.5s after speech end. Stitched
  `meeting_mix.wav` (46s) + chapters = the whole conversation in one player; Gantt shows speech, live-STT
  overlap, markers, and cross-lane playback.
- **Chunked-streaming rework (2026-09-29, owner review)**: sentence chunks are now pipelined while the
  speaker continues — chunk playbacks audibly overlap later speech (simultaneous interpretation, not
  turn-then-translate). All 4 turns stream (first audio lands mid-speech); Gantt shows per-chunk playback
  bars overlapping speech; turn cards carry a ✓-streams flag. Turns 2–3 extended to 2 sentences so every
  turn can overlap.
- **Bulk HQ complete (2026-09-29)**: 30 clips, ~12 min, mean QC WER 0.052 — meets the 10–20 min
  fine-tune minimum. Corpus section holds all 30. Training run itself (1–5h) deferred per demo rule;
  base-voice choice (fresh lili vs male SK base vs from-scratch) parked for post-demo research.

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
