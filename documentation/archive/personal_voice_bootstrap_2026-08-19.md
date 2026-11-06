# Personal-voice bootstrap pipeline — real run, 2026-08-19

**Goal**: ~90s of real recording in, single Piper voice out, GPT-SoVITS used only offline as a synthetic-data generator, never as a live/shipped engine.

## What was built and run for real

1. `speaker_voices/voice_rec_1m.wav` (41.7s, user reading the Rainbow Passage) added to `speaker_voices.json`, plus the pre-existing ~48s of clips = ~90s real audio total.
2. GPT-SoVITS (RVC-Boss, isolated venv, base pretrained weights, zero-shot mode) used **once, offline** to synthesize 15 diverse EN sentences in the cloned voice — `synth_corpus/` (4.1MB). Never shipped, never called at runtime.
3. Real + synthetic mixed per the ZeSTA oversampling mitigation: the 5 real clips duplicated 2x, 15 synthetic clips at 1x → 24-entry training manifest, ~3.7 min effective audio.
4. `scripts/finetune_personal_voice.py` run for real — 200 training steps (not the earlier 3-step toy proof), warm-started from `rhasspy/piper-checkpoints` `en_US-ryan-medium`, on CPU (confirmed faster than MPS again, consistent with the earlier finding). ~10 min wall time for 200 steps / ~100 epochs (23 utterances, batch size 8).
5. Exported to `backend/tts/piper_models/en_US-personal-medium.onnx` (63.5MB, same size class as the project's other Piper voices).

## Real measured numbers

- **RTF 0.0562** (5-run average) on the fine-tuned, exported voice — confirms the core claim again on a real (not 3-step toy) training run: Piper-native speed, no XTTS-style per-call overhead.
- Audio sanity-checked (RMS ~3708, peak near full-scale, ~11% silence ratio consistent with natural speech pauses) — not silence or garbage output. **Voice-similarity quality was not evaluated** — that requires human listening or a speaker-verification model, neither done here. RTF proves the mechanism; it says nothing about how much the output actually sounds like the user.
- Generic `sk_SK-lili-medium` Piper voice re-confirmed still working (RTF 0.076) alongside the new personal voice — the two coexist fine via the same Piper API, consistent with the four-tier fallback design.

## Real bugs hit and fixed (toolchain, not design)

- **PyTorch 2.6+ `weights_only=True` default** rejects `rhasspy/piper-checkpoints` files (`pathlib.PosixPath` not in the safe-globals allowlist). Fixed via `torch.serialization.add_safe_globals([pathlib.PosixPath])`, not the riskier `weights_only=False`.
- **`--ckpt_path` vs `--model.warmstart_ckpt`**: `--ckpt_path` eagerly re-parses the checkpoint's saved hyperparameters and fails on a schema mismatch (`sample_bytes` unrecognized) against a checkpoint from an older piper1-gpl version. `--model.warmstart_ckpt` is the correct mechanism for initializing from a *different* checkpoint and avoids this entirely.
- **`val_mos` `ModelCheckpoint` callback hard-crashes** (`MisconfigurationException`) instead of soft-skipping when the MOS predictor doesn't log a value in time — happens even after downloading the SpeechMOS predictor and setting `num_test_examples > 0`. Not essential (the `val_mel` + `save_last` checkpoint selection still works); stripped from `piper.train.__main__._DEFAULT_CALLBACKS` at the CLI shim level.
- **Hardcoded `version_0` in `finetune_personal_voice.py`'s export step** — real bug, not toolchain: Lightning auto-increments the version directory on every run sharing the same `work_dir`, including failed retries. Was silently pointing at an earlier, far-less-trained checkpoint. Fixed to pick the highest `version_N` that actually has a `last.ckpt`.
- `core.pyx` missing entirely from the installed `piper-tts==1.7.0` wheel's `monotonic_align` package (not just the compiled `.so`, as the earlier session's docstring assumed) — fetched from the `OHF-voice/piper1-gpl` source repo directly.
- GPT-SoVITS: newer `torchaudio` requires `torchcodec`, which itself needs an FFmpeg ABI (`libavutil.56`) far older than this machine's Homebrew FFmpeg (8.1) ships — pinned `torchaudio==2.7.1` instead of chasing the FFmpeg mismatch.

## Not done, honestly flagged

- **T028-equivalent (wiring into live TTS selection)**: not done. The `.onnx`/`.onnx.json` pair sits in `backend/tts/piper_models/` like any other Piper voice but nothing in `backend/main.py`'s voice-selection path knows about it yet.
- **Full live-pipeline (STT→MT→TTS) integration smoke test**: attempted, cut short. `ctranslate2`'s Opus-MT path hit an unrelated dependency wall in this isolated venv (`transformers`' `AutoTokenizer` doesn't resolve `MarianConfig` in the installed version, and `MarianTokenizer` needs `sentencepiece`, not installed). Not a finding about the voice pipeline — a venv-scoping gap in this pass. Substituted a narrower, still-real check: confirmed the new personal voice and the existing generic SK voice both synthesize correctly through the same `PiperVoice` API, back to back.
- **SK/DE output from the personal voice**: not attempted, and would not have worked if attempted — this fine-tune is EN-only (the recording was EN-only), consistent with the already-documented per-language limitation. SK/DE stays on the generic Piper voice tier (confirmed still functional above) until/unless SK or DE reference audio is recorded and fine-tuned separately.
- Real quality/voice-similarity comparison against the earlier 3-step toy run, and against XTTS zero-shot output, not done — needs actual listening, not something to assert from RTF numbers alone.
