# Kaggle T4 paced route run — 2026-11-04

## Result

PROVEN: the real BP server (`app.py`, FastAPI `/ws`, faster-whisper STT -> Opus-MT CT2 -> OmniVoice TTS with a cloned reference) ran on a Tesla T4 and streamed translated speech for 90 s of continuous English (owner's voice, paced at 1x) for EN->SK and EN->CS. Kernel `yegorby/bp-route-paced-t4-20261104` v3: status COMPLETE (`kaggle kernels status`), report `status: complete`, job wall 368.8 s of the 1500 s cap, both runs `complete`, 0 error events, 0 `Traceback` in `server.log` (645 lines), GPU = `Tesla T4` (torch device name asserted in job).

NOT PROVEN: human listening grade, clone identity or naturalness, translation/STT quality (no WER, no meaning grade), physical mic/speaker, browser playback, CS/SK as source, a second run for variance (one run per language), commercial licence. Route stays `route_unverified`.

Headline finding: EN->SK at 1x does not keep up. Translated SK audio totals 101.5 s for 90 s of input, so the listener queue grows (lag p95 17.8 s at 1x). EN->CS (87.9 s audio) stays near 3 s median. Playback at 1.25x brings SK p50 to 3.3 s but p95 still 9.6 s.

## Tables (from `paced-en-{sk,cs}.json` events, `paced_replay.listener_model`, p50/p95 linear interpolation)

Definitions: `first_audio_after_source_end` = first tts_audio arrival minus segment source end (rate-independent). `lag` = listener start minus source end (one chunk at a time, pitch-preserving rate). `first_word_lag` = upper bound (includes the pause before the segment).

### EN->SK (omnivoice, ref `sk.wav` 7.0 s, 17 transcripts, 17 translations, 16 voiced, 31 tts_audio, 101.5 s audio)

| rate | first_audio p50 / p95 / max (s) | lag p50 / p95 / max (s) | first_word_lag max (s) | lag growth first->last (s) | last listen end minus input end (s) |
|---|---|---|---|---|---|
| 1.0 | 1.38 / 2.90 / 6.20 | 11.43 / 17.82 / 18.60 | 24.80 | +12.41 | +24.7 |
| 1.25 | 1.38 / 2.90 / 6.20 | 3.31 / 9.61 / 10.77 | 13.22 | -3.01 | +8.15 |

### EN->CS (omnivoice, ref `cs.wav` 5.6 s, 16 transcripts, 16 translations, 16 voiced, 30 tts_audio, 87.9 s audio)

| rate | first_audio p50 / p95 / max (s) | lag p50 / p95 / max (s) | first_word_lag max (s) | lag growth first->last (s) | last listen end minus input end (s) |
|---|---|---|---|---|---|
| 1.0 | 1.42 / 1.64 / 1.75 | 3.02 / 5.12 / 5.91 | 11.02 | +1.85 | +9.09 |
| 1.25 | 1.42 / 1.64 / 1.75 | 1.60 / 3.15 / 3.41 | 8.69 | +0.21 | +6.44 |

Stage times per segment (server-reported `metrics`): STT 0.15-0.63 s (small, CUDA fp16), MT 0.13-0.59 s (CPU int8). SK first segment max first_audio 6.2 s includes cold TTS start (clone prompt 2.92 s + warm-up 1.30 s were done at config_update; the 6.2 s is the first 6 s forced segment + TTS). Segments are mostly the 6 s forced cut (`Speech segment exceeded 6s` x many) since the reader does not pause; the 6.02 s forced cut is the default `max_segment_s`.

Clone check: `server.log` has 61 `OmniVoiceTTS: ... cloned)` lines (31 SK + 30 CS), 0 `auto voice` lines; clone prompts built from sidecar text (`ref text: sidecar`). That shows the cloned-prompt path ran, not that the output sounds like the owner.

Saved wav hashes: none. `paced_replay.py` records byte counts and durations only, not audio. Audio was therefore not exported (also keeps the owner's voice off disk). A human grade needs a variant that saves the tts_audio payloads.

## What ran (exact)

Local, no model loading:
- `python3 scripts/gpu_bench/kaggle_route/build_package.py /tmp/bp_route_pkg2` -> 104 files, `bp-code.tar.gz` 3,988,014 B (allowlist: backend, ui, app.py, languages, requirements*.txt, scripts/{language_packs,convert_models,convert_whisper}.py, scripts/live/paced_replay.py; excludes `.env`, certs, `sql_app.db`, `.venv*`, models, speaker_voices, processed, `*.onnx` incl. piper voices, `*.wav|m4a|bin|pt`). Plus `rainbow_en_90s.wav` (first 90 s of `me_en_rainbow.m4a`, 16 kHz mono PCM16, 2,880,078 B).
- Import check: tarball extracted to `/tmp/bp_route_chk2`, `BP_NO_SSL=1 .venv/bin/python -c 'import app'` -> `import app OK`.
- `kaggle datasets create -p /tmp/bp_route_pkg2` -> private dataset `yegorby/bp-route-code-20261104` (contains the owner's 90 s voice; delete it when no longer needed).
- `kaggle kernels push -p scripts/gpu_bench/kaggle_route --accelerator NvidiaTeslaT4 -t 1500` (private, internet on, datasets `yegorby/bp-gpu-bench-refs` + the new one). job.py sha256 of v3 push: see `scripts/gpu_bench/kaggle_route/job.py` on disk (v1 push hash 1313b741..., v3 edited afterwards: unverified hash).
- `kaggle kernels output yegorby/bp-route-paced-t4-20261104 -p documentation/implementation-2026-11-04/kaggle-route-private` (local `.gitignore` = `*`).

On Kaggle (job.py): pip `omnivoice==0.2.1 soundfile==0.14.0 scipy==1.16.3`; then `requirements.txt` minus numpy/scipy/transformers/tokenizers/huggingface-hub/soundfile/numba/onnxruntime/librosa/SQLAlchemy plus `transformers sentencepiece onnxruntime SQLAlchemy>=2.0.36`; MT conversion `backend.mt.convert_opus_mt_to_ct2.convert_model` for `opus-mt-en-sk` and `en-cs` (int8, 18-40 s each); `faster_whisper.utils.download_model('small')`; CUDA smoke test passed (so `BP_DEVICE=cuda`, `BP_COMPUTE_TYPE=float16`; MT stays CPU, repo default `BP_MT_DEVICE=cpu`); server `BP_NO_SSL=1 BP_PORT=8013 python app.py` (ready 12.1 s); then per language:

`paced_replay.py <rainbow_en_90s.wav> --base http://127.0.0.1:8013 --src en --tgt {sk|cs} --tts omnivoice --seconds 90 --rates 1.0 1.25 --load-timeout 420 --out paced-en-{lang}.json` run through an in-process wrapper that appends `speaker_wav_path=speaker_voices/ref_{lang}.wav`, `speaker_lang`, `stt_model_size=small` to the client's `config_update` (paced_replay.py itself untouched). VAD env knobs left at defaults (`BP_VAD_*`, `BP_BARGE_IN` unset).

## Versions (route-report.json)

Python 3.13 (Kaggle), torch 2.11.0+cu130, transformers 5.20.2, tokenizers 0.23.3, huggingface-hub 1.33.0, numpy 2.1.3, scipy 1.16.3, soundfile 0.14.0, ctranslate2 4.8.5, faster-whisper 1.2.0, omnivoice 0.2.1 (model revision `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`), fastapi 0.141.1, uvicorn 0.38.0, websockets 12.0, SQLAlchemy 2.1.6, piper-tts 1.3.0. These differ from the repo's pinned CPU stack (numpy 1.26.4, transformers 4.57.6, etc.) and from the Mac measurements.

GPU memory: max sampled `nvidia-smi memory.used` 4761 MiB (3 s sampling; whole-GPU, STT + TTS together).

## Deviations from the shipped code (applied only to the throwaway Kaggle copy, recorded in the report)

1. `registry.py`: `omnivoice` licence `unknown` -> `commercial-ok`. The gate fails closed and the WS cannot set `allow_unknown_license`, so unmodified the engine would be refused in this route. HF API `card_data.license` returned `None` for the pinned revision, so the licence is NOT verified here (the code docstring says Apache-2.0; unverified).
2. `omni_tts.py`: `from_pretrained(..., revision=<pinned>)`.
3. `requirements.txt` pins relaxed as listed (Python 3.13 and torch 2.11 on Kaggle). SQLAlchemy 2.0.30 crashes the server on 3.13 (`TypeError: Can't replace canonical symbol for '__firstlineno__'`), so the shipped pin is not 3.13-safe: real finding for any 3.13 deployment.
4. Reference clips are the existing private dataset's `sk.wav` / `cs.wav` (normalised to 16 kHz mono PCM16), same files as the earlier TTS job, registered as `public_demo` entries in a Kaggle-local `speaker_voices.json`.

## Failed kernel versions (root cause, one fix each)

- v1: `AssertionError` locating `bp-code.tar.gz`. Cause: Kaggle auto-extracts the archive to `bp-code/` inside the dataset. Fix: job copies the pre-extracted dir when no tar is found.
- v2: server exited at import. Cause: SQLAlchemy 2.0.30 vs Python 3.13 (above). Fix: `SQLAlchemy>=2.0.36` (got 2.1.6).
- v3: COMPLETE.

## Limits on the numbers

- One run per language, 90 s, one voice, defaults for VAD. Differences under ~1 s between SK and CS first_audio are noise-level.
- The listener model is a model, not a playback measurement; segments are mostly 6 s forced cuts, so lag includes the segment length.
- SK vs CS gap is mainly TTS audio volume (101.5 s vs 87.9 s) and the extra rest of the forced segments; the cause was not isolated.
- Quality observations (ungraded, one glance): STT "rainbows" for "raindrops"; CS MT fragment "z dlouhého kruhového oblouku, s jeho cesta vysoko nad" is ungrammatical (segment cut mid-sentence). No WER or meaning scoring was run.

## Files

- Report JSON, per-language events, server log, kernel log: `/Users/yegor/Documents/STU/BP/documentation/implementation-2026-11-04/kaggle-route-private/` (gitignored; `bp/` there is the kernel's working copy of the code, no reference audio, no ct2 models).
- Job: `/Users/yegor/Documents/STU/BP/scripts/gpu_bench/kaggle_route/{job.py,kernel-metadata.json,build_package.py}`.
- Kernel: https://www.kaggle.com/code/yegorby/bp-route-paced-t4-20261104 (private).

## Cheapest next fixes

- SK saturation: lower `OMNIVOICE_STEPS`, play at 1.25x, or shorter SK translation; test with `BP_VAD_MAX_SEGMENT_S` 3-4.5 on T4 (one more kernel run, same job).
- Export a few tts_audio WAVs to the private output so the owner can grade the clone by ear.
- Resolve the OmniVoice licence from the model card before `allow_unknown` is ever flipped.
