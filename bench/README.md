# Frozen SK->EN STT benchmark (STT stage)

**Question:** how accurate is the repo's Slovak speech-to-text, per model and per mode (whole clip vs the live segmenter at 6 s cap), measured the same way every time?

## Clips and references
| clip | audio | reference |
|---|---|---|
| me_sk_script | `speaker_voices/me_sk_script.m4a` (134 s) | `refs/me_sk_script.txt` |
| me_sk_trhove | `speaker_voices/me_sk_trhove.m4a` (104 s) | `refs/me_sk_trhove.txt` |
| me_sk_trhove_b | `speaker_voices/me_sk_trhove_b.m4a` (109 s) | `refs/me_sk_trhove_b.txt` |

`audio/` holds symlinks to the clips (no audio copied). References are the **owner's own transcripts**, copied unchanged from the `transcribed_text` field of `speaker_voices/speaker_voices.json` (`transcript_source: existing`). Sanity check against clip length: 86 / 111 / 105 words per minute, plausible for read speech. **Flag:** `me_sk_trhove_b.txt` is byte-identical to `me_sk_trhove.txt`. It is a different recording, so if the speaker deviated while reading, `_b` carries extra error that is the reference's, not the model's. Not rewritten; a proper check needs a human listen-through.

## Metrics (`metrics.py`)
Normalisation: lowercase, strip punctuation (`[^\w\s]`), collapse whitespace. **No diacritic folding** (`deň` != `den`). WER = word-level Levenshtein / reference words. CER = same on normalised characters, spaces included. Self-test: `python bench/metrics.py`.

## Commands (run from repo root, `.venv`)
```
.venv/bin/python bench/metrics.py
.venv/bin/python bench/run_stt.py --model ct2_models/whisper-small-sk --mode whole
.venv/bin/python bench/run_stt.py --model ct2_models/whisper-small-sk --mode live --cap 6
.venv/bin/python bench/run_stt.py --model ct2_models/whisper-base --mode whole
.venv/bin/python bench/make_table.py      # regenerates RESULTS.md from results/*.json
```
Settings fixed in code: CPU, int8, language `sk`, beam 5, temperature 0, `vad_filter=False`. `live` replays the repo's main.py VAD loop in audio time (webrtcvad aggressiveness 3, 20 ms frames, 0.3 s silence, pre-VAD RMS gate 0.02, cap on speech frames), copied from `bp-review-2026-11-02/replay.py`. Models load with `local_files_only=True`: nothing is downloaded. Results go to `results/<date>_<model>_<mode>[cap].json`; repeats are saved as `_runN`, never overwritten. Each JSON records model, mode, beam, per-clip WER/CER/hypothesis/RTF, versions, git HEAD, machine descriptor, UTC time.

## Add a model
Put a CT2 directory somewhere local (or a cached faster-whisper size) and run `run_stt.py --model <path>`. Use `--tag` for a readable name. Then `make_table.py`.

## Known limitations
- One speaker (the owner), three clips of ~2 min: **not a held-out speaker**. `_b` is a second recording of the same speaker and text, not independent data. Differences of a few WER points are inside noise.
- STT only; MT and TTS are not scored here. `live` is a replay in audio time, not a real-time run (RTF is compute over audio length, single process).
- Reference text is the owner's transcript, unaudited (see flag above).
- WER is sensitive to Slovak inflection and the no-folding rule; compare only runs from this folder.
- RTF depends on machine load at run time.
