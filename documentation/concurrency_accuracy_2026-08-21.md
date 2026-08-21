# Concurrency + Accuracy Testing — 2026-08-21 (T037-T040)

**Environment:** M1 Pro 16GB, macOS, `./venv` (Python 3.11.15), server via `app.py` with
`DYLD_LIBRARY_PATH=/opt/homebrew/lib`, TTS engine `piper` (generic `cs_CZ-jirka-medium`).
All runs against a live local server, not simulated — real WebSocket sessions, real STT/MT/TTS.

## What was built (T037)

Extended `test/concurrent_performance_test.py` (did not fork a new script):
- `score_accuracy()`: WER via `jiwer.wer()` (already a project dependency) against
  `test/{Hello,Can you hear me_,My test speech}_transcript.txt`; translation correctness via
  stdlib `difflib.SequenceMatcher` normalized-similarity ratio + exact-match flag against the
  matching `*_translation.txt` — no new dependency (BLEU/sacrebleu was considered and rejected,
  difflib already covers "how close" without adding a package for three short reference strings).
- `SessionMetrics` gained `wer` / `translation_similarity` / `translation_exact_match`.
- `generate_summary_report()` gained an Accuracy Statistics table per user-count section.
- `main()`'s `user_counts`/`tts_model`/`ramp_up_duration` are now CLI flags
  (`--user-counts`, `--tts-model`, `--ramp-up-duration`) instead of hardcoded constants —
  needed to run N=2 and N=5 without hand-editing the file each time.

## Proven

**N=2 ("mano-a-mano" baseline, T038):** 100% success (2/2). E2E latency 3.44-4.60s (avg 4.02s).
WER 0.0 and 0.143 (the non-zero one is a real STT mishear, "Laddy"→"Loddy" — confirmed by reading
the raw transcript, not a scoring bug). Translation similarity 0.828-0.925, 0% exact-match (MT
paraphrases short phrases, e.g. "na účely klonovania"→"na klonovanie" — similarity is the
meaningful signal, exact-match is not expected to be high). Data: `test_output/performance_tests/run_20260821_083317/`.

**N=5 (small-meeting scenario, T039):** 100% success (5/5). E2E latency 4.28-26.93s (avg 9.66s,
std dev 8.66s — one long-clip session queued behind the other four). WER avg 0.102. Translation
similarity avg 0.754. CPU avg 4.5%/max 17.9%, memory avg 154MB/max 230MB (client-process
measurement — see Known Limitations below). Data: `test_output/performance_tests/run_20260821_083342/`.

**Sweep 10/12/15/18/20 (T040):** success rates 100% / 100% / 100% / 72.2% / 85.0%. This is a real
improvement in raw pass-rate over the 2025-11-28 baseline in `PERFORMANCE_TEST_RESULTS.md`
(12 stable/100%, 15 unstable/73%) — this run held 100% through 15 and still passed 85% at 20.
The 18-user run (72.2%) scoring *worse* than the 20-user run (85.0%) is itself evidence this is
noisy single-trial data, not a clean deterministic ceiling — see Unknown below.

Server-side root cause for the explicit failures ("Unexpected error: " with an empty message,
5/18 and 3/20 sessions): grepped `/tmp/bp_server.log`, found matching `ERROR - An unexpected error
occurred in handle_audio...` / `ERROR - Backend: Error sending {transcription_result,
translation_result, tts_audio, final_metrics} to client...` at the same timestamps — the server is
trying to write to a WebSocket the test client already closed. This is a client-teardown timing
issue in the test harness, not a server crash (`app.py`, PID 72550, stayed alive and serving
through the entire sweep).

## Assumed → now disproven, two real pre-existing harness limitations found

1. **The fixed 5-second post-`stop` receive window (`await asyncio.sleep(5)` in
   `simulate_user_session`, pre-existing code, not something this pass added) is too short once
   per-request STT time exceeds it** — and this run's own STT-latency numbers show that happening
   above ~10 users (STT avg 4.24s/max 11.25s at 12 users; avg 9.75s/max 11.29s at 15 users).
   Consequence: at every N≥10, a majority of "completed" sessions received **zero**
   transcription/translation events before the client cancelled its receiver and closed the
   socket — confirmed by reading the raw JSON directly, not inferred: 7/10 got any transcript at
   N=10, only 5/12 at N=12, only 4/15 at N=15, 6/18 at N=18 (minus the 5 hard failures), 5/20 at
   N=20. **The apparent "WER improves at N=20" (0.092) vs. "WER degrades at N=15" (0.509) in the
   raw table is sample-shrinkage noise, not a real accuracy trend — do not cite per-N accuracy
   numbers from this run as "accuracy under load" without this caveat.** The N=2/N=5 numbers above
   are reliable (large enough completion fraction); N≥10 accuracy numbers are not.
2. **`ResourceMonitor` in this script profiles `psutil.Process()` with no PID — i.e. the *test
   client* process, not the server.** This was true before this pass too (pre-existing code, T037
   didn't touch it). Every "Resource Usage" line in both this run and the 2025-11-28 report was
   always measuring the lightweight asyncio/websockets test script's own footprint (~150-300MB
   here), never the model-loaded server process. The 2025-11-28 report's claim "at 12 users, the
   16GB Unified Memory is likely fully utilized" was not something this instrumentation could ever
   have shown — it must have come from external observation (Activity Monitor or similar), not
   from the script's own numbers. **Real server-side RAM/CPU pressure remains unmeasured by this
   harness, then and now.**

## Unknown / not done this pass

- Whether the true ceiling is still ~12, or has moved to ~15-20: this sweep is **one trial per
  user-count**, and the 18-vs-20 non-monotonic result (72.2% < 85.0%) shows single-trial noise is
  large enough to flip the apparent ranking. A real ceiling claim needs 3+ trials per N with the
  same ramp-up strategy, averaged — not done here (out of this pass's time budget).
- Real server-side memory/CPU under load — needs `ResourceMonitor` to target the server's PID
  (or read `/proc`/`psutil` cross-process), not the client's. Not implemented this pass.
- Extending the fixed 5s receive window to be latency-aware (e.g. wait until the receiver task
  goes idle, or scale with observed STT time from `final_metrics`) instead of a hardcoded value —
  needed before N≥10 accuracy numbers from this harness can be trusted. Not implemented this pass
  (flagged, not fixed — see task additions below).
- Per project convention (see [[project_bp_translation_thesis]]/`tasks.md`), T018 (shared model
  pool refactor) is the standing, previously-identified concurrency fix and remains untouched —
  explicitly out of scope for this pass per direct instruction (demo stays single-person-use for
  now; multi-user meeting scaling deferred).

## Task backlog additions (not executed this pass, flagged in `tasks.md` Phase 9)

Two follow-up fixes to this pass's own instrumentation were identified but not built: making the
post-`stop` receive window latency-aware, and pointing `ResourceMonitor` at the server PID instead
of the client's. Both are small, scoped fixes to code this pass touched — noted as open items in
`tasks.md` rather than a new phase.
