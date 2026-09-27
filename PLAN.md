# BP Dev Plan — where we are, what's ahead

Branch: `main` (local, unpushed). Docs: `documentation/voice_and_app_direction_2026-09.md`.
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

## Now (one at a time)

- [ ] Overnight SK 2500-step run → export → lab A/B listen (timbre? F0 recovery?)
- [ ] EN personal same treatment (August run likely undertrained too)
- [ ] Per-language STT router in backend (Parakeet EN / whisper-small SK)
- [ ] QC similarity scoring (resemblyzer) only after a proper clone exists
- [ ] Push `main` when demoable; handler update message + demo ask

## Next

- [ ] Thesis numbers corrected to measured + `[DOPLNIŤ]` marker filled
- [ ] Stage-demo prep: script, pre-warmed models, soundcheck VAD, backup take
- [ ] Env raw-test on clean checkout (this Mac + 2nd ARM laptop) — parked until complete

## Run things

| Command | What |
|---|---|
| `make lab` | Voice Lab review page, static only (no login/upload — those need `make run`) |
| `make run` | Full backend (https://localhost:8000) |
| `make test` | Backend suite (existing files only) |
| `python3 scripts/update_voice_lab_library.py --no-test` | Refresh Voice Lab manifest |
| `venv/bin/python scripts/voice_similarity_qc.py --synthesize-only` | Synthesize QC candidates |
| `.venv-stt/bin/python scripts/stt_parakeet_spike.py --clip en\|sk` | Parakeet spike (separate venv) |
| `venv/bin/python scripts/pipeline_latency_probe.py` | First-output latency probe |
| `venv/bin/python scripts/machine_listen_qc.py` | Machine listening panel (WER thirds + acoustic health) |
