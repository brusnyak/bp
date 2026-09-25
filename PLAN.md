# BP Dev Plan — where we are, what's ahead

Branch: `voice-lab` (local, no push). Docs: `documentation/voice_and_app_direction_2026-09.md`.
Thesis rules: faculty guide + §4.1 AI rules → `Deklarácia k využitiu UI` skeleton lives in
`documentation/thesis_draft.md` (one `[DOPLNIŤ]` marker left).

## Done

- [x] Core pipeline STT→MT→TTS, non-blocking + barge-in (T043–T047, 9 commits — **unpushed**)
- [x] Hybrid fast cloning RTF 0.15–0.17, cross-lingual (measured, committed)
- [x] Per-language Piper path decided + documented (§2b: EN recording → cloned SK corpus → `sk_SK-personal`)
- [x] QC script rebuilt: live-engine candidates, `--synthesize-only/--score-only`, `scores.json` (4 candidates synthesized 2026-09-24)
- [x] Voice Lab static page: upload staging, 6 voices / 4 QC / 3 test clips, main-page styling, `make lab`
- [x] Makefile fixed: `install` (root npm), `test` (existing files only), `clean` no longer deletes `speaker_voices/`, new `lab/library/qc/qc-score` targets

## Now (one at a time)

- [x] **Captions in live UI** — subtitle strip + PiP pop-out (spec 002, committed)
- [x] **Lab plan toggle** — live PLAN.md panel (committed)
- [x] **DESIGN.md + skill** — house tokens, frontend-design global (committed)
- [ ] **Your recordings** — EN top-up + SK read, 2–3 min each, quiet room (staged via Voice Lab upload box first)
- [ ] **QC scoring** — qc venv + scoring run, then blind listening (10 raters) → thesis table

## Next

- [ ] `sk_SK-personal` build (§2b steps 2–5) → multilingual matrix green both directions
- [ ] Thesis numbers corrected to measured (~4.3s E2E, Hybrid RTF) + `[DOPLNIŤ]` marker filled
- [ ] Push `main` (9 commits) + `voice-lab` when demoable; handler update message + demo ask
- [ ] Stage-demo prep: script, pre-warmed models, soundcheck VAD, backup take
- [ ] Env raw-test on clean checkout (this Mac + 2nd ARM laptop) — parked until complete

## Run things

| Command | What |
|---|---|
| `make lab` | Voice Lab review page, no backend |
| `make run` | Full backend (https://localhost:8000) |
| `make test` | Backend suite (existing files only) |
| `python3 scripts/update_voice_lab_library.py` | Refresh Voice Lab manifest |
| `venv/bin/python scripts/voice_similarity_qc.py --synthesize-only` | Synthesize QC candidates |
| `../qc-venv/bin/python scripts/voice_similarity_qc.py --score-only` | Score QC candidates (qc venv, resemblyzer) |
