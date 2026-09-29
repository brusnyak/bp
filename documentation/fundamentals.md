# Voice-AI fundamentals for the owner (BP companion)

Goal: no blind execution. Every component in our pipeline maps to a concept below,
each with the file where it lives. Read top to bottom once; then use as reference.

## 1. Perceptron → MLP → backprop (the whole game in 3 ideas)

- **Perceptron**: `y = sign(w·x + b)`. A weighted vote. Only learns linearly separable
  boundaries (can't do XOR — the famous 1969 result that froze the field for a decade).
- **MLP (multi-layer perceptron)**: stack perceptron-like layers with a **nonlinearity**
  (ReLU/GELU) between them. Nonlinearity is the whole trick — without it, 100 layers
  collapse into one linear map. Universal approximation: one wide hidden layer can fit
  any continuous function, given enough neurons.
- **Backprop**: chain rule applied mechanically. Loss at the output → gradients flow
  backward, each weight learns how much it contributed to the error. Optimizer (Adam):
  per-weight step sizes from gradient history. Everything since 2012 is this, scaled.

## 2. From MLP to Transformer (why our models look the way they do)

- **CNN**: MLP with shared local weights — sees local patterns (edges, phonemes) cheaply.
- **RNN**: feeds its own output back — handles sequences but forgets long range and
  trains sequentially (slow). Mostly obsolete, but its decoding idea survives below.
- **Attention**: let every position look at every other position directly, weighted by
  relevance (`softmax(Q·K/√d)·V`). No forgetting, fully parallel. Cost: O(n²) memory —
  this is why long audio/context gets expensive.
- **Transformer**: attention + MLPs + residual connections + normalization, stacked
  12–96×. Two halves: **encoder** (reads the whole input at once) and **decoder**
  (writes output left-to-right, each step attending to input + previous outputs).
  **Autoregressive** = one token per step, can't parallelize across time. This single
  fact explains most of our latency numbers.

## 3. The four decoding families (know which game each model plays)

| Family | How it writes output | Speed | Who here |
|---|---|---|---|
| Autoregressive decoder | token-by-token, each step sees previous | slow, best quality | Whisper STT, OmniVoice, Opus-MT |
| Non-autoregressive / CTC | whole output at once + alignment rule | fast, slightly worse | SenseVoice (candidate) |
| Transducer (RNN-T/Zipformer) | streams: emits symbols or "blank" per frame | real-time streaming | Zipformer via sherpa (candidate) |
| Diffusion / flow-matching | refines noise into signal over N steps | quality ∝ steps | OmniVoice synth, Supertonic |

`beam_size` (our STT knob): autoregressive tries that many hypotheses per step and keeps
the best. beam=1 (greedy) is 30–50% faster, usually ~95% as good — the exact trade we
haven't measured yet on `small-sk`.

## 4. Audio specifics (what the models actually see)

- Speech is stored as waveforms (16,000 samples/sec in our pipeline) but models read
  **mel spectrograms**: frequency × time pictures warped to match human hearing.
  `faster_whisper_stt.py` feeds these to Whisper's encoder.
- **VAD** (webrtcvad, `backend/main.py`): tiny non-neural classifier per 20ms frame,
  speech-or-not. Aggressiveness 3 = hair-trigger. A segment "ends" 0.3s after the last
  voiced frame (`SILENCE_TIMEOUT`). Constant synth-bed noise defeats it — our measured
  failure mode, not a theory.
- **Neural audio codecs** (Mimi in Moshi, Seamless units): compress audio into discrete
  tokens so a Transformer can speak them like words. S2S models translate token-to-token
  with no text in the middle — fast and prosody-preserving, but hungry (7B+ params).

## 5. Our pipeline, component by component

- **STT — Whisper** (`backend/stt/faster_whisper_stt.py`): encoder-decoder Transformer on
  mel input. Fixed 30s window (pads short clips → waste + silence hallucinations like
  our "studio"/"m." ghosts). `small-sk` = community Slovak-tuned weights; `base` = stock.
  Parakeet (EN side) is a fast Conformer-CTC/RNNT hybrid — different family, hence faster.
- **MT — Opus-MT/CTranslate2** (`backend/mt/ctranslate2_mt.py`): small Transformer,
  int8-quantized (weights rounded to 8-bit: ~4× smaller, ~2× faster, negligible quality
  loss at this size). Runs in ~0.1s — never our bottleneck, which is why SLM swaps fail
  cost/benefit here.
- **TTS — Piper/VITS** (`backend/tts/piper_tts.py`): VAE + normalizing flows + GAN
  vocoder in 15–50M params. Monotonic alignment = robotic-but-fast (RTF 0.04). A
  fine-tune re-tunes durations/timbre toward new data; 18 clips undertrained ours
  (val_mel 0.57→0.29, F0 range halved). Base checkpoint needed because 12 min can't
  train 15M params from nothing.
- **Zero-shot TTS — OmniVoice**: diffusion-LM over audio tokens conditioned on a 13s
  reference prompt. No training per voice (in-context cloning). Cost is the ~16
  denoising steps per chunk — hence RTF ~1.0 on MPS, ~0.025 only on datacenter GPUs.
- **Hybrid (Piper+OpenVoice)**: Piper draws the phonemes fast, OpenVoice repaints the
  timbre. Two-model cost, uncanny-valley risk — parked.

## 6. CZ-proxy rule (your correction, now policy)

A model without explicit SK support is NOT rejected if it covers CZ (or multilingual
phoneme inventories): drop/neutralize the language token and benchmark on our SK v2b
clips. Precedents: XTTS-cs reading Slovak, Whisper Czech-proxy runs. Every candidate
below gets a measured SK WER + RTF before any verdict — no paper rejections.

## 7. Candidate benchmark list (research → fast iteration → verdict)

STT: Whisper turbo (already queued), Cohere Transcribe ONNX (SK? test), sherpa-onnx
Zipformer (SK/CZ streaming test), SenseVoice (expect fail on SK, 1-hour test),
Moonshine (expect fail, 1-hour test), Parakeet-SK variants if any appear.
MT: keep Opus-MT unless a candidate shows +chrF at ≤0.15s (unlikely; low priority).
TTS: Chatterbox (MIT, CZ/SK proxy test), Kokoro (proxy test), F5 (needs CUDA — likely
dead on our boxes, verify in 30 min or drop), SeamlessM4T-v2 S2ST spike (SK native,
voice-preserving; the one end-to-end bet).
Runtimes: sherpa-onnx (cross-platform), MLX fast path (Apple-only, non-default).

Protocol (same for all): fixed SK v2b clips → WER/CER + RTF + device, recorded in
`processed/` + Lab matrix. One page per candidate, numbers only.

## 8. How to read a model card (5-minute checklist)

1. Params + precision (FP16/INT8/INT4) → rough RAM: params × bytes-per-weight.
2. License (Apache/MIT = free; CC-BY-NC = thesis-ok, product-no; Coqui/XTTS = non-commercial).
3. Language list → SK? else CZ/multilingual (proxy rule, §6)?
4. Architecture family (§3) → predicts the latency shape before you run it.
5. Reported RTF + on WHAT hardware (H100 numbers are fiction for us; M1 CPU/MPS only).
