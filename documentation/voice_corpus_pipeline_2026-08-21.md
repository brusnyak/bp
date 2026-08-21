# Personal-voice corpus pipeline — repeatable, not one-off (2026-08-21)

Follow-up to `documentation/personal_voice_bootstrap_2026-08-19.md`, which proved by hand
that a short recording can become a real fine-tuned Piper voice. That session was manual
and one-off. This one turns the ingest side into a repeatable pipeline: point it at raw
recordings, get a clean training corpus, no hand-holding.

## The bug this fixes (found, not hypothesized)

`speaker_voices/speaker_voices.json` had real data-integrity problems before this session:

- 8 of 24 entries were literal duplicates of 4 files (`id` suffixed `_dup2`, same `path`).
  `finetune_personal_voice.py`'s `read_speaker_voices()` has no dedup logic, so the CSV it
  built for training counted those 4 clips twice each.
- `voice_rec_1m.wav` and `voice_rec_1m.m4a` are the same take (original + a format-converted
  copy) — 2 more entries, same underlying audio.
- 15 of 24 entries pointed at `speaker_voices/synth_*.wav` — the GPT-SoVITS bootstrap output
  from the 08-19 session — which don't exist on disk. The training script silently skips
  missing files, so this wasn't fatal, but it means the "24-entry manifest" that produced the
  currently-live `en_US-personal-medium.onnx` actually only had ~9 entries with real files
  behind them, over 5 unique audio files, 4 of them double-counted.

Net effect: the corpus that trained the currently-shipped voice was smaller and more skewed
toward specific clips than its own metadata implied.

## The fix: `scripts/prepare_voice_corpus.py`

New script, run against `speaker_voices/`:

1. Normalizes every input file (any of wav/m4a/mp3/webm/ogg/flac/aac) to canonical mono
   22050Hz 16-bit WAV via `ffmpeg` (arg-list subprocess call, no `shell=True`, no string
   interpolation of filenames into a shell command).
2. Dedupes by **audio content**, not filename: a 20-bucket RMS-energy fingerprint per
   normalized clip, cosine-similarity >= 0.985 within a 0.5s duration tolerance counts as the
   same take. This is what actually caught `voice_rec_1m.wav` == `voice_rec_1m.m4a` — a
   filename-stem heuristic alone would have worked for that specific pair, but content
   fingerprinting doesn't depend on naming convention discipline holding up over time.
3. Drops manifest entries with no file on disk (the 15 dangling synth_* refs).
4. Reconciles transcripts: reuses an existing `transcribed_text` where present; with
   `--auto-transcribe`, runs the project's own `FasterWhisperSTT` (same pattern as the
   existing `scripts/temp_transcribe.py`) on any clip that has none.
5. Writes a clean `speaker_voices.json` — one entry per unique take, real path, real
   transcript, explicit `transcript_source` (`existing` vs `auto-STT ... not human-verified`).

**Real bug found and fixed while building this**: the STT auto-transcribe step failed
silently on first run — `ModuleNotFoundError: No module named 'backend'` — because
`scripts/` is not on `sys.path` when run as `python scripts/prepare_voice_corpus.py`.
`scripts/temp_transcribe.py` has the identical bug (its `sys.path.insert` line is placed
*after* its `backend.*` imports, so it never takes effect). Fixed in the new script by
inserting the project root before the import, not after.

## Real run against this project's own corpus

```
$ ./venv/bin/python scripts/prepare_voice_corpus.py --source-dir speaker_voices --auto-transcribe --write

Dropping 15 manifest entries with no file on disk: [...15 synth_* entries...]
Found 6 raw audio file(s) in speaker_voices
6 file(s) -> 5 unique take(s) after dedup:
  '2_my voice 1.wav' (unique)
  'Can you hear me_.wav' (unique)
  'My test speech_xtts_speaker_clean.wav' (unique)
  'hello.wav' (unique)
  'voice_rec_1m.wav' (kept) duplicates: ['voice_rec_1m.m4a']
  Transcribing 2_my voice 1.wav ...
Clean corpus: 5 entries, 89.5s (1.49 min) of unique real audio.
Wrote speaker_voices/speaker_voices.json
```

`2_my voice 1.wav` had never been transcribed before (silently excluded from every prior
training run) — auto-STT recovered it: *"Reading these statements, I agree to provide my
voice for cloning in purposes. My voice will be used to synthesize translated speech within
this application."* Not human-verified, flagged as such in the manifest.

`speaker_voices/voice_rec_1m.m4a` (the untracked recording) was read, never modified/moved/
deleted, per instruction.

## Real fine-tune run on the cleaned corpus

`rhasspy/piper-checkpoints` (the `en_US-ryan-medium` warm-start checkpoint the 08-19 session
used) initially looked gone — `hf_hub_download(repo_id=..., repo_type="model")` 401s. It
isn't gone: it's a HF **dataset** repo, not a model repo. `repo_type="dataset"` resolves it
correctly. Worth fixing at the call site if this project ever re-downloads it elsewhere.

Rebuilt the training toolchain from scratch (no `.venv-train` survived from 08-19):
`python3.11 -m venv .venv-train`, `piper-tts[train]`, `cmake`/`ninja`/`espeak-ng` (already
on this machine via Homebrew), and the `monotonic_align` Cython extension — same missing-
`.pyx`-in-the-wheel issue documented on 08-19, fetched `core.pyx` from
`OHF-voice/piper1-gpl` again and built it.

Ran `scripts/finetune_personal_voice.py --voice-name en_US-personal-v2 --max-steps 200
--ckpt-path <ryan-medium warm-start>` against the cleaned 5-entry, 89.5s corpus (script's
`--data.num_test_examples 2` split leaves 3 clips for actual training — small, expected for
this data size). Warm-start confirmed live: `[warmstart] Copied 784 parameters`.

**Completed for real**: 200 steps (matching the 08-19 run's step count), ~21 minutes wall
time on this M1 Pro/CPU, exported successfully to `backend/tts/piper_models/en_US-personal-v2.onnx`
(63.5MB, same size class as every other Piper voice in this repo) via the legacy TorchScript
ONNX exporter (the documented `dynamo=False` workaround, still required).

Registered as a **new, separate** `TTS_ENGINES["piper_personal_v2"]` entry in
`backend/tts/base.py` — deliberately not overwriting `piper_personal`
(`en_US-personal-medium.onnx`, the GPT-SoVITS-augmented voice from 08-19), since this run has
no synthetic augmentation and is expected to sound rougher; overwriting the better-quality
default with an unverified one would be a regression, not an upgrade. Real smoke test:

```
PASS: synthesized 4.74s of audio in 0.2372s (RTF 0.0501), sr=22050
```

RTF 0.0501 — same speed class as `piper_personal` (0.0446-0.0562) and `piper` (0.0479),
confirming the pipeline mechanism (not just the original one-off run) reliably produces
Piper-native-speed output. Voice-similarity quality vs. `piper_personal` was **not**
compared (needs the same resemblyzer-cosine-similarity or human-listening pass the 08-19
session used, not done in this pass — flagged as open, not assumed either way).

## What's still open (T031-T033, unchanged by this pass)

The GPT-SoVITS synthetic-bootstrap step from 08-19 (real+synthetic ZeSTA-style oversampling,
the thing that took ~90s of real audio to a genuinely good-sounding voice) was **not**
reproduced in this pass. `GPT_SoVITS` isn't installed in any of this machine's conda envs
(`TTS`, `rvc_env`, `xtts_env`, `trade`, `trw_bot`) — the 08-19 session's isolated training
venv for it didn't survive, and its `synth_corpus/` output isn't on disk anymore either.
Standing GPT-SoVITS back up from scratch (clone the repo, fresh venv, the documented
`torchaudio==2.7.1` pin to dodge a `torchcodec`/FFmpeg-ABI mismatch, download its own
pretrained weights, run zero-shot synthesis) is real, separate, multi-step work — explicitly
out of this pass's time budget. This run is **real-audio-only**: proves the ingest pipeline
(normalize → dedupe → transcribe → fine-tune → export → register) end to end on genuine
data, without the synthetic-augmentation step that improved quality last time. Expect this
run's output to sound rougher than `en_US-personal-medium.onnx` for that reason — this was
a deliberate, documented scope cut, not an oversight. T031/T032 (GPT-SoVITS as the live
default engine, real quality/RTF measurement) remain open exactly as tasks.md already states.

## Security review (requested explicitly this pass)

- `scripts/prepare_voice_corpus.py`: all `ffmpeg` calls use `subprocess.run(cmd, ...)` with
  `cmd` as an argument list — never `shell=True`, never a string-interpolated command.
  `sanitize_stem()` strips anything derived from a filename to alnum/`_`/`-`/space before it
  can reach a filesystem path or subprocess arg, and explicitly rejects empty/`.`/`..` results.
- `backend/main.py`'s existing `/voices/upload` endpoint (the eventual real ingest point for
  this pipeline) was already doing the ffmpeg call correctly — `asyncio.create_subprocess_exec`
  with an argument list, audio piped via stdin/stdout, never a shell string. Checked the
  `voice_name`/`file.filename` sanitization for path traversal by testing
  `os.path.splitext`'s actual behavior on adversarial inputs (`a/../../b.wav`,
  `x.a/../../../tmp/evil`, etc.) rather than assuming — confirmed it's slash-aware and doesn't
  leak a `/` into the derived extension, so no traversal there.
- **Found and fixed, in that same endpoint**: no upload size limit — `contents = await
  file.read()` was unbounded, reading the entire request body into memory regardless of size.
  Added `MAX_VOICE_UPLOAD_BYTES = 50MB` with a 413 response.
- **Found and fixed, same endpoint**: the size-limit's `HTTPException(413, ...)` would have
  been silently swallowed and rewritten to a generic 500 by the function's outer
  `except Exception` clause — added an `except HTTPException: raise` guard ahead of it. This
  also fixes the pre-existing (lower-severity) issue of the inner ffmpeg-failure 500 getting
  double-wrapped by the same outer catch-all.
- No secrets, tokens, or credentials in any new file.

## Files

- `scripts/prepare_voice_corpus.py` (new)
- `speaker_voices/speaker_voices.json` (rewritten — 5 clean entries, was 24 with dupes/dangling refs)
- `backend/main.py` (upload-size cap + HTTPException-swallow fix)
- `.venv-train/` (rebuilt training toolchain, gitignored, not committed)
