# Slovak -> English model evaluation, 2026-09 (CPU only)

Machine: Ryzen 5 8645HS (6 cores / 12 threads), 14 GB RAM, Windows 11, **no GPU** (faster-whisper int8 on CPU). Test set: the 18 bilingual reading-script sentences read by one speaker (134 s of audio), so this is a *single-speaker, n=18* result: good for ranking models on this machine, not a general accuracy claim. WER/CER are computed after lower-casing and stripping punctuation. Reproduce with `scripts/eval_stt.py <model>` and `scripts/eval_stt_tune.py <model> <threads> <beam> <prompt 0|1>`; they need your own per-sentence clips of `documentation/reading_script_bilingual.md` saved as `eval_data/sk_clips/sk_00.wav` ... `sk_17.wav` (16 kHz mono; personal recordings are deliberately not in the repo).

## STT (Slovak speech -> text), faster-whisper int8

| Model | Beam | WER | CER | s / clip | RTF |
| --- | --- | --- | --- | --- | --- |
| small | 5 | 0.620 | 0.175 | 2.2 | 0.30 |
| small | 1 | 0.661 | 0.197 | 1.9 | 0.26 |
| small + generic Slovak `initial_prompt` | 5 | 0.620 | 0.178 | 2.4 | 0.32 |
| medium | 5 | 0.464 | 0.128 | 7.2 | 0.97 |
| medium | 1 | 0.510 | 0.146 | 5.8 | 0.77 |
| large-v3-turbo | 5 | 0.438 | 0.126 | 7.3 (8.7 first run) | 0.97 |
| large-v3-turbo | 1 | 0.474 | 0.134 | 7.2 (6.1 with 12 threads) | 0.96 (0.82) |

RTF = processing time / audio duration; above ~0.5 the captions visibly lag behind the speaker.

Findings

- Accuracy is driven almost entirely by model size. Thread count (6 vs 12), greedy decoding and a generic prompt do not change WER meaningfully.
- On this CPU **no model is both accurate (WER < 0.5) and fast (RTF < 0.5)**. `small` is the only real-time option and is poor for Slovak (WER 0.62: e.g. "detekuje, kedy reč začína" became "dedekujú, keď reč sa čina"); `large-v3-turbo` is the most accurate but runs at about real time.
- `backend/main.py` silently upgrades any Slovak-source request for `base`/`small` to `large-v3-turbo`, so with the default settings Slovak input costs ~7 s per ~7 s sentence on a machine like this one.
- Whisper pads every clip to a 30 s encoder window, so short utterances do not get proportionally cheaper; the encoder dominates. Realistic ways to get both accuracy and speed: a GPU, a Slovak-fine-tuned small/medium Whisper model (not evaluated), or a hosted STT API.

## MT (Slovak -> English)

Not re-evaluated in this round: the baseline `opus-mt-sk-en` (CTranslate2 int8) translates a sentence in ~0.1 s and is not the bottleneck; end-to-end quality for Slovak input is dominated by the STT errors above. Comparing it against NLLB-200-distilled-600M (e.g. on the NTREX-128 news set) is still open; no MT script is shipped because none was run.

## Other measurements (same machine)

| Stage | Result |
| --- | --- |
| STT `base` (EN) | 0.67-0.80 s per short phrase |
| MT EN->SK, SK->EN (opus-mt, int8) | 0.05 s per phrase, ~0.5 s per 60-word paragraph |
| TTS Piper (warm) | 0.18-0.28 s per sentence (2.6 s on the very first call) |
| EN->SK end-to-end sim, 18 sentences | STT WER 0.105, MT ~0.09 s, TTS ~0.28 s |
| SK->EN end-to-end sim, 18 sentences | STT WER 0.62 (`small`), garbled translations follow from that |
