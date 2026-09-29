# Demo report skeleton (structure only — numbers filled after benchmarks)

Each section lists its JSON source. No hand-typed numbers ever.

1. **Pipeline latency by direction** — chart `latency_by_direction.png`
   ← `processed/conversation_sim.json`. Claim to verify: EN→SK ~1s, SK→EN STT-bound.
2. **Meeting timeline (live)** — chart `gantt_meeting.png` + mix player
   ← `processed/meeting/meeting_timeline.json`. First-audio-mid-speech proof.
3. **STT accuracy matrix** — chart `wer_synth_vs_mic.png` + tables
   ← `processed/sk_direction/`, `processed/stt_input_test/`, `processed/synth_lengths/`.
4. **Engine A/B (TTS quality as STT input)** — table
   ← `processed/engine_ab/matrix.json`. omni-zeroshot 0.054 vs piper-personal 0.540.
5. **Training corpus QC** — table + sample players
   ← `processed/bulk_hq/manifest.json`. 30 clips, ~12 min, mean WER 0.052.
6. **New-model spikes** — one subsection per engine
   ← `processed/new_models/<engine>_matrix.json`. Verdicts with kill reasons.
7. **Honest gaps** — VAD finalization lag, trailing-silence hallucinations,
   turn-2 anomaly, cold-start tax. Each with the file that proves it.
