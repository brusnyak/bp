# Monday demo test checklist (2026-09-28)

Owner runs this live; every row is pass/fail with an artifact. Session logs land in
`processed/sessions/*.jsonl` automatically (no setup) for post-demo review.

## A. Accuracy (measured, not eyeballed)

- [ ] **EN STT**: play `me_en_script.m4a` through the pipeline → WER vs script.
  Baseline (2026-09-27): **0.077** (`processed/stt_baseline.json`). Pass if ≤ 0.10 live.
- [ ] **SK STT**: play `me_sk_script.m4a` → WER vs proofread SK column.
  Baseline: **0.2143 clean / ~0.32 mic** (whisper-small-sk, Kaggle v6 T4). KNOWN WEAK — above 0.2 quality gate. Pass = honest number recorded, not hidden. SK→EN stays `experimental` until a challenger (Canary AST on GPU) crosses WER < 0.2 on real mic. **Canary AST measured 2026-11-03 on Kaggle T4: SK ASR WER 0.0477 (<0.2 ✅), but SK→EN AST chrF++ ~52 (below commercial bar) — not yet a quality pass.**
- [ ] **MT EN→SK**: 18 script pairs through Opus-MT → similarity vs proofread SK column
  (paraphrase-tolerant: SequenceMatcher, not exact match). Record the number.
- [ ] **MT SK→EN**: same reversed. Expect degradation carried over from SK STT — note it.
- [ ] **CZ STT**: play `me_cs_staromestske.m4a` → WER vs proofread CZ column.
  Baseline: Whisper small with `language=cs` (SK-tuned model as CZ proxy). Record the number — CZ should be measurably better than SK (~0.21 WER).
- [ ] **MT CZ→EN**: through Opus-MT `opus-mt-cs-en` (requires CTranslate2 conversion first — `cs-en` added to `scripts/convert_models.py` PAIRS; run `scripts/convert_models.py` to convert).

## B. Voice (ears + numbers)

- [ ] **SK voice in owner's voice** (Piper fine-tune `sk_SK-personal-male-medium`): listen — identity? artifacts? trembling?
- [ ] **EN voice** (Piper `en_US-ryan-medium` or owner fine-tune): same.
- [ ] **Latency**: first-call vs repeat-call TTS RTF noted out loud. Piper RTF ~0.04 on CPU.

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
