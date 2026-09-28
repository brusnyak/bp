# Slovak speech recognition and translation: evaluation, 2026-09 (CPU only)

Machine: Ryzen 5 8645HS (6 cores / 12 threads), 14 GB RAM, Windows 11, **no GPU**; everything int8 on CPU (faster-whisper / CTranslate2 / onnxruntime). WER and CER are computed after lower-casing and stripping punctuation. RTF = processing time / audio duration (above ~0.5 captions visibly lag).

**Decision:** Slovak input now defaults to a Slovak-fine-tuned Whisper `small` ([NaiveNeuron/whisper-small-sk](https://huggingface.co/NaiveNeuron/whisper-small-sk), MIT), converted to CTranslate2 int8 by `scripts/setup.py`. It is as fast as plain `small` and close to `large-v3-turbo` in accuracy. Override with `BP_SK_STT_MODEL`.

## 1. Which recognizer? (Slovak speech to text)

Two test sets, because a single home recording cannot separate "weak model" from "unusual recording".

**A. Public FLEURS Slovak test set, 60 utterances** (native read speech; `scripts` reproduce the numbers with the same code as B). Published reference values are from [arXiv 2509.19270](https://arxiv.org/html/2509.19270v1) (Whisper) and the [Parakeet model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).

| Recognizer | WER here | Published FLEURS-sk WER |
| --- | --- | --- |
| Whisper `small` | 0.377 | 36.1% |
| Whisper `large-v3-turbo` | 0.118 | 10.7% |
| **Whisper `small`, Slovak-tuned (CT2 int8)** | **0.129** | 10.6% |
| Parakeet-TDT v3, onnx-asr int8 | 0.202 | 8.82% (full set, fp32) |

The first three agree with the literature, which validates the evaluation code. Parakeet int8 is worse than its model card; full-precision was not tested on FLEURS (on set B it was *worse* than int8, so quantisation is not the explanation; the difference is unresolved).

**B. One speaker, 18 sentences (134 s), read from `documentation/reading_script_bilingual.md`** (AAC 44.1 kHz, SNR 53 dB, no clipping: the recording itself is clean).

| Recognizer | WER | CER | s / 7 s clip | RTF |
| --- | --- | --- | --- | --- |
| **Slovak-tuned `small`** | **0.255** | **0.080** | 2.4 | 0.33 |
| `large-v3-turbo` (beam 5) | 0.438 | 0.126 | 7.3 | 0.97 |
| `large-v3-turbo` (greedy) | 0.474 | 0.134 | 7.2 | 0.96 |
| `medium` (beam 5) | 0.464 | 0.128 | 7.2 | 0.97 |
| `medium` (greedy) | 0.510 | 0.146 | 5.8 | 0.77 |
| Parakeet-TDT v3 int8 | 0.578 | 0.253 | 0.9 | 0.12 |
| Parakeet-TDT v3 fp32 | 0.630 | 0.292 | 0.9 | n/a |
| `small` (beam 5) | 0.620 | 0.175 | 2.2 | 0.30 |
| `small` (greedy) | 0.661 | 0.197 | 1.9 | 0.26 |
| `small` + generic Slovak prompt | 0.620 | 0.178 | 2.4 | 0.32 |

Controls that were run to find out why set B is 3-6x worse than set A for the generic models:

- **Cutting the recording into sentences is not the cause:** `large-v3-turbo` on the *uncut* file scores WER 0.411 (per-clip 0.438).
- **Model support is not the cause:** Slovak is officially supported by both families (Parakeet v3 lists 25 languages incl. Slovak and Czech).
- **Clean synthetic Slovak** (Piper voice reading the same script): `small` 0.417, `large-v3-turbo` 0.135, Parakeet 0.198.
- Different architectures (Whisper, Parakeet) make the *same* substitutions on the recording ("znie" heard as "s ne", "Hlasové" as "Lasové"). What remains is a mismatch between what was said and the (machine-written, partly unnatural) script, i.e. a property of this recording, and single-speaker data cannot tell pronunciation from script issues. **A new recording of standard sentences (`scripts/build_recording_set.py`, 75 sentences with English references) is the next step.**
- Threads (6 vs 12), greedy decoding and a generic prompt do not change accuracy meaningfully; Whisper pads every clip to a 30 s window, so short utterances are not proportionally cheaper (turbo costs ~9 s even for a 2 s sentence).

## 2. What that means for the English output (Slovak to English, Opus-MT sk-en, set B)

| Recognizer | BLEU | chrF |
| --- | --- | --- |
| perfect transcript (ceiling of the MT) | 53.0 | 76.8 |
| **Slovak-tuned `small`** | **33.3** | 59.3 |
| `medium` | 27.8 | 54.5 |
| `large-v3-turbo` | 25.2 | 51.2 |
| `small` | 19.5 | 45.8 |
| Parakeet v3 int8 | 15.7 | 38.6 |

The translation model is not the bottleneck; recognition errors are. An alternative MT model (e.g. NLLB-200-distilled-600M) was not evaluated.

## 3. Two-sided conversation, measured end to end

`scripts/demo_conversation.py` plays a six-turn dialogue (A speaks English, B answers in Slovak) through the real backends and records every stage. Median time from end of speech to complete translated audio:

Two complete runs (the second while the machine was also doing other work), each with synthetic speech and with real recordings; ranges are min-max of the medians over those four conversations:

| Slovak recognizer | EN -> SK | SK -> EN | SK WER (synthetic dialogue, 3 sentences) |
| --- | --- | --- | --- |
| `large-v3-turbo` (previous default) | 1.0-1.3 s | 8.9-10.8 s | 0.07-0.12 |
| **Slovak-tuned `small` (new default)** | 1.0-1.4 s | **2.6-3.5 s** | 0.07-0.09 |
| Parakeet v3 int8 | 0.9-1.3 s | 0.4-0.9 s | 0.12-0.18 |

The WER column scores only three Slovak sentences per run and Piper's synthesis is not deterministic, so its values change between runs and the differences between recognizers there are noise; use section 1 for accuracy.

English -> Slovak costs about 1 s (STT 0.7-0.9 s, MT 0.1 s, TTS 0.1-0.3 s). Piper's very first call used to cost ~2.7 s; the engine is now warmed up at load time. Timings on a laptop vary by 10-30% run to run, so treat single numbers as approximate.

## 4. Other findings

- `FasterWhisperSTT.transcribe_audio` returned a meaningless time (~0.02 s) because it stopped the timer before consuming faster-whisper's lazy segment generator; fixed.
- The auto-upgrade of Slovak input to `large-v3-turbo` (author's earlier measurement on other hardware) makes each Slovak turn take ~9 s on a 6-core CPU; superseded by the default above.

## 5. Open items

1. Record the 75-sentence set (and ideally 1-2 more Slovak speakers), then re-run sections 1-3: `python scripts/build_recording_set.py`, `python scripts/record_reading.py`.
2. Parakeet is 3-5x faster than the tuned Whisper but less accurate here; worth re-testing on the new recording, in full precision, and with DirectML on the iGPU.
3. Compare `NaiveNeuron/whisper-medium-sk` / `-large-v3-turbo-sk` (FLEURS 7.6 / 6.4) if a GPU is available; on this CPU they would be as slow as their generic counterparts.
4. The tuned models are trained on parliamentary speech ("domain-biased", per the model card): expect the best results on formal speech.
