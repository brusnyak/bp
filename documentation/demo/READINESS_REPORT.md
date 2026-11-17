# BP readiness report (2026-11-17)

## What works (VERIFIED — reproduced against saved receipts this session)

| Capability | Evidence | Status |
|---|---|---|
| EN → SK live route, median lag 0.79 s, p90 1.39 s | `documentation/evidence/2026-11-15/D_wait05.json` + `.score.txt` (PASS all 4 criteria) | VERIFIED |
| Recognition WER 5.2% (EN → SK, generated paced audio) | same receipt, criterion 4 | VERIFIED |
| Early start ≥ 50% of 8+word sentences | same receipt, criterion 2 (1.0) | VERIFIED |
| Lag growth ≤ 0.5 s/min | same receipt, criterion 3 (-0.05) | VERIFIED |
| EN → DE and EN → SK end-to-end on real server (Kaggle T4) | `route_test.json`, BENCHMARKS.md §2b | VERIFIED |
| EN → DE agreement route 1.56 s median | BENCHMARKS.md §2b | VERIFIED |
| ci.yml fix: deleted `backend_auth_tests.py` removed from test list | CI logs (exit code 4, all 3 OSes); all 16 remaining test files exist | VERIFIED (fix applied locally, not yet pushed) |
| Public site builds self-contained (no external CSS, no `tokens.css` 404) | `scripts/build_site.py`; `check-ui` 0 problems | VERIFIED locally |

## What remains experimental (EXPERIMENTAL)

| Capability | Limitation |
|---|---|
| Slovak input (SK as spoken language) | ~17–20% WER on this speaker's non-native Slovak; needs GPU or streaming-native recogniser |
| Cloned/personal voices | OmniVoice zero-shot clone is the candidate; RTF 0.18–0.34 on T4, not run end-to-end with Voxtral; non-commercial weights |
| Room mode (one speaker, many listeners) | Built + unit-tested; no real-model run with real listeners |
| Voxtral streaming through real route | 3.06–3.15 s median lag on T4 (misses 3.0 s goal); needs vLLM/faster GPU |
| Generic Slovak TTS default | Works; voice identity not the speaker's |

## What is blocked (PLANNED — not operational)

| Item | Blocker |
|---|---|
| Higgs Audio / Fish Audio S2-Pro on T4 | SGLang-Omni needs `outlines_core` Rust extension; no Rust toolchain/pre-built wheel on Kaggle Python 3.13 |
| Mac Voxtral through real route | Median lag 11.4 s / p90 20.9 s on 16 GB M1 Pro; swap pressure; not viable |
| Real-model room test | `bp-room-test-t4` queued, not run |
| Fresh-install `--profile nvidia` retest | Fix committed (`b371b31`), retest pending |

## Verified performance

- Median translation audio lag: 0.79 s (EN → SK, Mac M1 Pro, generated English, one run)
- p90 lag: 1.39 s
- Recognition WER: 5.2%
- All three figures are `check-docs`-verified against `D_wait05.score.txt`.

## Outstanding risks

- Slovak spoken input is the weakest link (accented non-native speech); do not demo SK→anything live.
- Voice cloning is not production-ready (licences + latency unproven end-to-end).
- CI is green only after removing the deleted `backend_auth_tests.py` reference; push + re-run needed to confirm.
