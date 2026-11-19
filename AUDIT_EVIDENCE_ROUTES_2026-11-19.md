# BP — Consolidated Per-Language-Route Evidence Audit (read-only, 2026-11-19)

Source: repo `main` branch (commit d476939). Read-only audit; nothing modified.

## A. Full live-route tests (end-to-end through real server)

| # | Route | STT engine | MT engine | TTS engine | HW / profile | Dataset / size | Mic type | Median lag | p90 / p95 | WER | chrF / COMET | Drift | Audio queue | Status | Receipt | Measurement def |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | EN→SK | Whisper turbo (MLX) | Opus-MT | Piper | M1 Pro, `agree_short` | `en_paced.wav` 60 s (44.46 s input), generated EN | generated replay | 0.792 s | p90 1.387 s / max 1.416 | 5.2% | n/a | -0.05 s/min | all 27/27 segments got audio; queue OK | integrated/tested | `evidence/2026-11-15/D_wait05.json` + `.score.txt` | lag = first audio after source segment end |
| A2 | EN→SK | Whisper turbo (MLX) | Opus-MT | Piper | M1 Pro, `agree_short` | `en_paced.wav` 60 s | generated replay | 0.751 s | p90 1.802 s / max 2.048 | 5.2% | n/a | -0.33 s/min | all 21/21 segments got audio | integrated/tested (baseline) | `evidence/2026-11-15/A_baseline.json` + `.score.txt` | same as A1 |
| A3 | EN→SK | Whisper turbo (MLX) | Opus-MT | Piper | M1 Pro, `agree_short` | `en_paced.wav` 60 s | generated replay | 1.073 s | p90 1.573 s | 5.2% | n/a | -0.13 s/min | all 22/22 | integrated/tested | `evidence/2026-11-15/B_guard_xf.json` + `.score.txt` | same as A1 |
| A4 | EN→SK | Whisper turbo (MLX) | Opus-MT | Piper | M1 Pro, `agree_short` (step 0.35 s) | `en_paced.wav` 60 s | generated replay | 0.528 s | p90 1.5 s | 4.2% | n/a | FAIL 0.6 s/min (>0.5 goal) | all 22/22 | integrated/tested | `evidence/2026-11-15/C_step035.json` + `.score.txt` | same as A1 |
| A5 | EN→SK | Nemotron 3.5 ASR 0.6B (MLX) | Opus-MT | Piper | M1 Pro, `agree_short`, rate 1.0 | `en_real.wav` 60 s owner reading | REAL mic recording | 1.098 s | p90 2.263 s / max 2.627 | not reported | n/a | 0.288 s/min | all 26/26 | integrated/tested | `evidence/2026-11-10/route_nemotron_en_sk_real_speech_mac.json` | lag = first audio after source end |
| A6 | EN→SK | Nemotron 3.5 ASR 0.6B (MLX) | Opus-MT | Piper | M1 Pro, `agree_short`, rate 1.25 | `en_real.wav` 60 s | REAL mic recording | 0.553 s | p90 1.699 s / max 2.032 | not reported | n/a | 0.288 s/min | all 26/26 | integrated/tested | same as A5 | same as A5 |
| A7 | EN→SK | Nemotron 3.5 ASR 0.6B (MLX) | Opus-MT | Piper | M1 Pro, `agree_short`, adaptive | `en_real.wav` 60 s | REAL mic recording | not extracted | not extracted | not reported | n/a | not extracted | all 26/26 | integrated/tested | same as A5 | same as A5 |
| A8 | EN→DE | Whisper turbo | Opus-MT | Piper | Kaggle T4, 2 s chunks | owner EN reading 60 s, paced 1x | REAL mic recording | 3.89 s | p90 5.34 s | 22.1% | n/a | n/a | zero server errors; 23/34 with audio | integrated/tested | `evidence/2026-11-07/route_test.json` | lag after segment end |
| A9 | EN→DE | Whisper turbo | Opus-MT | Piper | Kaggle T4, agree every 1.0 s | owner EN reading 60 s | REAL mic recording | 1.56 s | p90 2.55 s | 21.3% | n/a | n/a | zero server errors; 24/34 | integrated/tested | same as A8 | same as A8 |
| A10 | EN→DE | Whisper turbo | Opus-MT | Piper | Kaggle T4, agree every 0.5 s | owner EN reading 60 s | REAL mic recording | 17.88 s | p90 23.54 s | 47.5% | n/a | collapses | queue blew up (8 s) | integrated/tested (negative) | same as A8 | same as A8 |
| A11 | EN→SK | Whisper turbo | Opus-MT | Piper | Kaggle T4, 2 s chunks | owner EN reading 60 s | REAL mic recording | 1.97 s | p90 2.88 s | 23.8% | n/a | n/a | zero server errors; 24/34 | integrated/tested | same as A8 | same as A8 |
| A12 | EN→SK | Whisper turbo | Opus-MT | Piper | Kaggle T4, agree every 1.0 s | owner EN reading 60 s | REAL mic recording | 1.89 s | p90 3.10 s | 23.0% | n/a | n/a | zero server errors; 24/34 | integrated/tested | same as A8 | same as A8 |
| A13 | EN→SK | Voxtral (Transformers, T4) | Opus-MT | Piper | Kaggle T4, agree_short | owner EN reading 60 s | REAL mic recording | 3.06 s | p90 6.0 s | 4.1% | n/a | grows ~6 s/min | zero server errors; 17/17 | integrated/tested | `evidence/2026-11-09/voxtral_route_t4.json` | lag after segment end |
| A14 | EN→DE | Voxtral (Transformers, T4) | Opus-MT | Piper | Kaggle T4, agree_short | owner EN reading 60 s | REAL mic recording | 3.15 s | p90 6.71 s | 4.1% | n/a | grows | zero server errors; 16/16 | integrated/tested | same as A13 | same as A13 |
| A15 | EN→SK | Voxtral vLLM (T4) | n/a (text only) | n/a | Kaggle T4, vLLM server | generated EN + German | generated | not route-level | text 0.07-0.08 s after audio | 1.0% EN paced / 3.1% EN cont. / 2.1% DE paced | n/a | no drift | n/a | component tested | `evidence/2026-11-09/voxtral_vllm_t4.json` | STT-only text latency |
| A16 | Room: baseline (no room) | not recorded | n/a | Piper de/sk/fr | Kaggle T4 x2 | EN script reading | generated/script | 1.84 s median | p90 2.96 s | n/a | n/a | n/a | zero server errors; 24/34 with audio | unit-tested (model-free) + Kaggle run | `evidence/2026-11-09/room_test_t4_log.json` | speaker lag |
| A17 | Room: 3 listeners (1 per language) | not recorded | n/a | Piper de/sk/fr | Kaggle T4 x2 | EN script reading | generated/script | 5.64 s median | p90 6.74 s | n/a | n/a | n/a | identical audio per language; zero listener errors | Kaggle run (real models, v2) | same as A16 | speaker lag |
| A18 | Room: 12 listeners (4 per language) | not recorded | n/a | Piper de/sk/fr | Kaggle T4 x2 | EN script reading | generated/script | 7.31 s median | p90 8.67 s | n/a | n/a | n/a | identical audio per language; zero listener errors | Kaggle run (real models, v2) | same as A16 | speaker lag |
| A19 | Room: 3 listeners | not recorded | n/a | Piper de/sk/fr | Kaggle T4, MT on GPU | EN script reading | generated/script | 6.96 s median | p90 8.14 s | n/a | n/a | n/a | queue 0.0; MT 0.51-0.59 s; TTS 0.39-0.46 s | Kaggle run (real models, v3) | `evidence/2026-11-10/room_test_t4_v3.json` | speaker lag |
| A20 | Room: 3 listeners MT on GPU | not recorded | n/a | Piper de/sk/fr | Kaggle T4, MT on GPU | EN script reading | generated/script | 4.0 s median | p90 5.32 s | n/a | n/a | n/a | queue 0.0 | Kaggle run (real models, v3) | same as A19 | speaker lag |
| A21 | Room: 12 listeners | not recorded | n/a | Piper de/sk/fr | Kaggle T4 | EN script reading | generated/script | 7.23 s median | p90 8.16 s | n/a | n/a | n/a | queue 0.0; MT 0.52-0.54 s; TTS 0.36-0.45 s | Kaggle run (real models, v3) | same as A19 | speaker lag |

## B. Offline component tests (not full live route)

| # | Component | Engine | HW | Dataset / size | Metric | Result | Status | Receipt |
|---|---|---|---|---|---|---|---|---|
| B1 | STT | Whisper large-v3-turbo | Kaggle T4 | FLEURS 60 clips/lang | WER | DE 4.8%, CS 11.6%, SK 10.7%, FR 6.9%, PL 7.7%, EN 4.5% | offline/tested | `evidence/2026-11-07/lang_matrix.json`, `LANGUAGES.md` |
| B2 | STT | Whisper small | Kaggle T4 | FLEURS 60 clips | WER | DE 10.9% | offline/tested | same as B1 |
| B3 | STT | Parakeet TDT v3 | Kaggle T4 | FLEURS 60 clips | WER | DE 6.3%, FR 5.0%, SK 9.9% | offline/tested | same as B1 |
| B4 | STT | kinit Slovak fine-tune | Kaggle T4 | FLEURS-sk | WER | 6.5% (native) | offline/tested | `evidence/2026-11-07/sk_compare.json`, `LANGUAGES.md` |
| B5 | STT | kinit Slovak fine-tune | Kaggle T4 | owner non-native reading | WER | 19.8% | offline/tested | `LANGUAGES.md`; `STATUS.md` |
| B6 | STT | plain Whisper turbo | Kaggle T4 | owner non-native reading | WER | 30.2% | offline/tested | `LANGUAGES.md` |
| B7 | STT | Parakeet | owner mic | Slovak reading | WER | 80.7% (wrong language) | offline/tested (negative) | `LANGUAGES.md` |
| B8 | MT | Opus-MT | Kaggle T4 | opus-100 200 sentences | chrF | DE→EN 57.9, CS→EN 59.2, FR 60.4/61.0, SK 62.1/55.6 | offline/tested | `lang_matrix.json`, `LANGUAGES.md` |
| B9 | MT | NLLB-1.3B | Kaggle T4 | opus-100 | chrF | DE→EN 54.6 | offline/tested | `LANGUAGES.md` |
| B10 | Whole chain | various | Kaggle T4 | 25 clips ~10 s | chain time | 1.15-1.57 s per utterance | offline/tested | `e2e_chain.json`, `BENCHMARKS.md` #3 |
| B11 | STT call cost | Whisper turbo MLX | M1 Pro | 12 windows | per-call | 720 ms fixed (30 s pad) regardless of length | offline/tested | `BENCHMARKS.md` #1 |
| B12 | Streaming sim | Whisper turbo | Kaggle T4 sim | EN/German FLEURS | latency/error | agree-1.0s: 2.31 s / 5.0% (EN); 2.34 s / 11.2% (DE) | offline/simulated | `stream_sim.json`, `BENCHMARKS.md` #2 |
| B13 | TTS | XTTS v2 | Kaggle T4 | EN/CS (no SK) | first audio/RTF/WER | EN first audio 0.45-0.48 s, RTF 0.52-0.63, WER EN 0%, CS 4-10% | offline/tested | `evidence/2026-11-15/kaggle_xtts_t4_results.json` |
| B14 | TTS | XTTS v2 | Kaggle T4 | SK | unsupported | no Slovak language | offline/tested (negative) | same as B13 |
| B15 | TTS | X-Voice stage1 | Kaggle T4 | EN/SK/CS | wall/RTF/WER | EN 16-step 2.2-2.7 s RTF 0.37-0.47, WER 0-11%; SK native WER 0.000 | offline/tested | `evidence/2026-11-15/kaggle_xvoice_t4_results.json` |
| B16 | TTS | Higgs Audio v3 4B | Kaggle T4 fp16 | EN/SK/CS tested | RTF/WER | RTF 1.2-1.3, WER 100% (unintelligible) | offline/tested (failed) | `evidence/2026-11-15/kaggle_higgs_t4_fp16_results.json` |
| B17 | TTS | Higgs attempt1 | Kaggle T4 | — | — | partial_failure (serve errors) | offline/tested (failed) | `evidence/2026-11-15/kaggle_higgs_t4_attempt1_results.json` |
| B18 | STT | Nemotron 3.5 ASR 0.6B | M1 Pro MLX | SK mic (trhove_b/script) | WER | SK 0.86/0.75, EN 0.19, language drift | offline/tested (negative for SK) | `model_landscape_2026-10.md` row 37; `nemotron_matrix.json` |
| B19 | STT | Canary-1B AST | Kaggle T4 | 80 FLEURS clips each SK/CS/EN | ASR WER / AST chrF++ | SK WER 0.0477, CS 0.0922, EN 0.0498; AST SK→EN chrF++ 52.49 | offline/tested | `model_landscape_2026-10.md` row 38 |
| B20 | Mic A/B | Whisper turbo MLX | M1 Pro | EN A1-A6, raw vs processed | WER | processed 6.7% whole / 15.5% pause-cut; raw 2.7% / 4.0% | offline/tested | `STATUS.md` mic A/B |

## C. Declared-only / not implemented

| # | Item | Stage | Declared in | Status | Note |
|---|---|---|---|---|---|
| C1 | qwen3-omni | unified | registry `_register_optional` | declared-only, factory=None | spec 003 T025, not built |
| C2 | canary-ast | stt | registry `_register_optional` | declared-only, factory=None | pending Kaggle T4 validation; WER<0.2 gate |
| C3 | kokoro | tts | registry `_register_optional` | declared-only | EN-only, no Slovak |
| C4 | qwen3-tts | tts | registry `_register_optional` | declared-only | EN-only cloning |
| C5 | chatterbox | tts | registry `_register_optional` | declared-only | no Slovak output |
| C6 | seamless-m4t-v2 | unified | registry `_register_optional` | declared-only | non-commercial, fail-closed |

## D. Contradictions flagged (docs vs registry vs implementation)

1. **Nemotron latency on product site vs default preset.** Site README (`index.html`/`index.template.html`) claims "Nemotron streaming recognition + Piper voice" measured 0.79/1.39/5.2 on Mac. But `mac.env` default preset is Whisper turbo (3.0-3.6 s lag, `STATUS.md`); `mac-nemotron.env` is opt-in and `setup.py` does not install mlx-audio. The README on `main` now says "separately provisioned assets; cloning does not install that configuration" — consistent, but the product site hero still implies Nemotron is the measured default. **Contradiction: site tiles show 0.79/1.39/5.2 (Whisper agree_short on generated EN), yet label the setup "Nemotron streaming recognition + Piper voice."**

2. **Nemotron SK WER negative vs site implication.** `model_landscape_2026-10.md` and registry both mark Nemotron SK WER 0.86/0.75 with language drift — rejected for SK. Yet the product site Nemotron timing (A5/A6/A7) is EN→SK on real speech with good lag (0.55-1.10 s). Lag measured but SK accuracy not validated on this route. **Contradiction: strong EN→SK latency numbers exist for Nemotron, but SK WER is a known negative (0.86/0.75) and never re-tested on the live route.**

3. **Room mode "integrated" claim vs reality.** Handler/message draft says room mode is "sestavený a integrovaný do živého prekladu" (built and integrated into live translation). Evidence: room mode is built + unit-tested (7 model-free tests) and has two Kaggle real-model runs (v2, v3), but STATUS.md explicitly says "not yet run with real models in front of real listeners" and requirement #1 (real-model room test) is open. The v2/v3 Kaggle logs (`room_test_t4_log.json`, `room_test_t4_v3.json`) show real-model runs did occur on T4 with speaker lag 5.64-7.31 s — but these are not reflected in STATUS.md's "open" status. **Contradiction: Kaggle room tests exist and passed functionally, yet docs still mark real-model room test as open/not done.**

4. **STT engine of Kaggle room test not recorded.** `room_test_t4_log.json` line 6: "STT engine of the run not recorded in the log." Room lag numbers (5.64/7.31 s) cannot be attributed to a specific STT engine.

5. **Route transcript error vs offline sim.** `BENCHMARKS.md` #2b: real-route transcript error 21-24% is ~4x the offline simulation (5.0%). Open item, not settled. README tiles quote 5.2% WER which is the *agree_short generated-EN* score (D_wait05), not the real-route score (23.0-23.8%).

6. **LANGUAGES.md chain times vs route_test.** LANGUAGES.md claims EN→SK chain 1.14 s / SK→EN 1.34 s. route_test.json real server shows EN→SK lag 1.89-1.97 s (agree10/chunk2). Different measurements (chain vs route lag), but close enough to be confusing rather than contradictory.

7. **Voxtral on Mac default.** `STATUS.md` update (2026-11-10) says Mac recogniser is Nemotron streaming (0.5-0.6 s on generated EN), but `mac.env` was not changed; default is still Whisper turbo. `evidence/2026-11-09/voxtral_route_mac.json` shows Voxtral on Mac fails (median 11.4 s). **Contradiction: STATUS update claims Nemotron is Mac recogniser, but default preset and setup.py still use Whisper.**

## E. Three most consequential evidence gaps for README publication

1. **No validated EN→SK accuracy number for the default (Mac/Whisper) path.** The README's 5.2% WER is from `agree_short` on *generated* English replay (D_wait05). The real-route score on the Kaggle server is 23.0-23.8% (route_test), and the owner's Slovak mic scores 17.7-45% depending on engine. Publishing 5.2% as the route's recognition WER overstates accuracy for real speech. Gap: need a real-speech, real-route EN→SK WER on the default Mac preset.

2. **Room mode real-model evidence is present but not documented/consolidated.** Two Kaggle room tests (v2: lag 5.64/7.31 s; v3: 6.96/4.0/7.23 s) exist and passed functionally, yet STATUS.md still marks real-model room test as open and the STT engine is unrecorded. Gap: consolidate v2/v3 into a final room-test receipt, record the STT engine, and update STATUS.md so the handler-facing claim ("integrated into live translation") is defensible.

3. **Nemotron SK accuracy gap.** Nemotron shows good EN→SK lag (0.55-1.10 s real speech, A5-A7) but SK WER 0.86/0.75 with language drift on the maintainer mic (B18), never re-tested on the live route. If Nemotron is pitched as the fast path, its SK accuracy must be validated or explicitly caveated as latency-only. Gap: Nemotron live-route SK WER, or a clear "latency measured, accuracy not validated" disclaimer.
