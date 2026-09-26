# Monday demo test checklist (2026-09-28)

Owner runs this live; every row is pass/fail with an artifact. Session logs land in
`processed/sessions/*.jsonl` automatically (no setup) for post-demo review.

## A. Accuracy (measured, not eyeballed)

- [ ] **EN STT**: play `en_script_reading.m4a` through the pipeline → WER vs script.
  Baseline (2026-09-27): **0.077** (`processed/stt_baseline.json`). Pass if ≤ 0.10 live.
- [ ] **SK STT**: play `sk_script_reading.m4a` → WER vs proofread SK column.
  Baseline: **0.71 raw / 0.63 plain** — KNOWN WEAK (whisper-base SK head). Pass = honest
  number recorded, not hidden. Fix path: faster-whisper `small` rung, then Parakeet-TDT
  v3 spike (SK supported, CC-BY-4.0, ONNX CPU builds exist) — post-Monday.
- [ ] **MT EN→SK**: 18 script pairs through Opus-MT → similarity vs proofread SK column
  (paraphrase-tolerant: SequenceMatcher, not exact match). Record the number.
- [ ] **MT SK→EN**: same reversed. Expect degradation carried over from SK STT — note it.

## B. Voice (ears + numbers)

- [ ] **Hybrid EN→SK in owner's voice** (new EN recording): listen — identity? artifacts?
- [ ] **Hybrid SK→EN** (new SK recording): same.
- [ ] **QC scores**: `scores.json` present (needs qc-venv `--score-only` run before Monday).
- [ ] **Latency**: Hybrid first-call (~embedding) vs repeat-call RTF noted out loud.

## C. Live system (both laptops)

- [ ] Boot offline (wifi off): models/voices load, session initializes.
- [ ] Mic → VAD → captions appear (~3s) → final transcript → translation → audio out.
- [ ] Subtitle strip updates; **PiP pops out and follows across Zoom/browser**.
- [ ] Virtual mic selected in Meet/Zoom; other side hears translation.
- [ ] `processed/sessions/` JSONL written (transcripts, translations, latencies, captions).
- [ ] Google login button renders real GIS (needs Console localhost origin); password login fallback works.
- [ ] PWA installs (Chrome/Edge); self-signed cert accepted.

## D. Handler narrative (continued work, not finished product)

Show in order: live EN→SK → PiP → Voice Lab (voices/QC/plan) → PLAN.md + tasks.
Say out loud: SK→EN STT is the measured bottleneck (0.63), fix path funded by evidence,
recordings landed this week. Backup take on hotkey if the mic dies.
