# Where the project stands (2026-11-07, end of session)

**Verdict:** one speaker can run live translation on their own machine today (English to Slovak, Czech, German). German is validated end to end through the real server on a Kaggle T4. Room mode (one speaker, many listeners, one language each) is **built and unit-tested but not yet run with real models**. The installer's GPU profile had two bugs found by a fresh-install test on a clean Linux box; both are fixed and the retest is pending. It is a measured, working base for a technical person, not a finished product.

## Works today (each line says how it was checked)

| Capability | State | Evidence |
|---|---|---|
| Default install on Linux | Works in a clean container (setup 352 s, tests pass) | `linux_setup_and_test.md` |
| `--profile nvidia` install on a fresh GPU box | **Failed on first test** (wrong turbo repo name; the box also hid its CUDA driver in the test itself). Fixed in commit `b371b31`; retest not yet run | Kaggle kernel `bp-fresh-install-t4` |
| `--profile mac|cpu` | Code and unit tests only; not run on a fresh machine | `test/setup_plan_test.py` |
| Real server on a Kaggle T4, EN to DE and EN to SK, both streaming routes | Works, zero server errors. German with agreement every 1.0 s: delay 1.56 s (2 s chunks: 3.89 s) | `route_test.json`, `BENCHMARKS.md` 2b |
| German and French voices | Were never registered (German crashed on first real run); fixed, test added | `test/tts_route_registry_test.py` |
| Live route EN to SK on a Mac, MLX turbo, pause mirror | Works; median delay 1.19 s, no errors in a 4-minute browser session | `evidence/2026-11-07/` |
| Per-language quality and timing | Measured on a T4 for German, French, Polish, Czech, Slovak, English | `LANGUAGES.md`, `BENCHMARKS.md` |
| Slovak | kinit fine-tune: 6.5% on FLEURS, 19.8% on the owner's non-native reading | `sk_compare.json` |
| Room mode (backend, `/ws/listen`, listener page `ui/listen/listen.html`, `scripts/live/room_listeners.py`) | 7 model-free tests pass (one translation and voice per language, order, pauses, dead sockets, limits). **No real-model run yet** | `test/room_tests.py`; Kaggle kernel `bp-room-test-t4` queued |
| Guards, pause mirror, rate follower | Unit-tested; used in the live runs above | `test/mlx_guard_tests.py`, `test/prosody_tests.py` |
| Docs numbers cannot drift | `make check-docs` verifies every marked number against its receipt | `scripts/check_readme_claims.py` |
| Tests and CI | 117 unit tests pass on macOS; CI on `main` (`47873ba`) passed on Ubuntu, Windows and macOS | `gh run list` |

## Streaming route

The accurate route (`BP_INCREMENTAL=1`: decode the growing buffer, commit words two decodes agree on) is validated on a T4 and is on in the `nvidia` preset. A step below 1.0 s collapses the queue. The Mac stays on 2 s chunks: turbo costs about 0.72 s per call there. Open: the real-route transcript error (21-24%) is four times the offline simulation (5%).

## Mac ceiling and what needs better compute (owner decision 2026-11-08: stop tuning on the Mac)

**Working state on the Mac (M1 Pro, 16 GB, shared with the desktop):** the Whisper turbo MLX route with `Speech cutting` = `agree` or `agree_short`, raw mic (default). Lag behind the speech 3.0-3.6 s against the 3.0 s goal; the voice starts mid-sentence; English input recognised at about 2-3% error. This is the best the repo reaches on this machine. Server command: see the handoff page. Nothing more is planned for the Mac.

**Why the Mac stops here:** Whisper decodes a padded 30 s window on every call, about 0.7 s each however short the audio, and two agreeing decodes are needed per word. Four fragment sizes all landed at 3.0-3.2 s. A streaming recogniser removes that floor, but the Mac cannot show it, because the desktop (WindowServer, Orca, browsers) holds the GPU, and swap was 10.6 of 11.3 GB used during the tests. Running more tests there overloads the machine, so none are planned.

**Voxtral Mini 4B Realtime (4-bit MLX) is installed and wired in but not proven:** `BP_STT_ENGINE=voxtral` (`backend/stt/voxtral_stt.py`; Slovak and other languages without Voxtral fall back to Whisper). Facts from 2026-11-08, all on the M1 Pro:
- Transcript of the owner's English script was clean; offline speed 0.62 x real time in two runs (38.1 s audio in 23.2 s and 23.7 s).
- Fed at real time in 0.2 s pushes: lag started at 0.3-0.5 s, then fell 3-6 s behind and stayed there (4 runs, load 5-28).
- Fed in 0.64 s and 1.28 s pushes (standalone script, not through the server): lag stayed at or below 0.7 s for the whole 38 s. The engine default is now 0.64 s (`BP_VOXTRAL_STEP_S`). **This setting was never run through the real route** (the run was stopped), so the route result below is for the old 0.2 s default.
- Real route (`paced_replay.py`, en to de, `agree_short`, 0.2 s pushes): all four automated criteria failed (median lag 8 s, recognition error 21.9%).

**Kaggle T4 result, 2026-11-09 (`documentation/evidence/2026-11-09/voxtral_t4.json`, kernel `scripts/gpu_bench/kaggle_voxtral_t4`, Voxtral Mini 4B Realtime fp16 through Transformers 5.22.2, 8.25 GB VRAM, audio released at real time, generated English and German speech, one run each):**
- Streaming holds real time: the run ends 1.7-1.9 s after the audio in 5 of 6 clips (one German clip ended 5.0 s after, with a p90 lag of 3.1 s: a single hiccup, not repeated or explained).
- Text arrives 0.04-0.08 s median after the audio chunk that contains it, on top of the model's own delay of about 0.5 s (the metric excludes that delay), so the recogniser adds roughly 0.5-0.6 s where Whisper agreement adds 2.1 s or more.
- Word error 1.0% (English paced), 4.2% (English continuous), 2.1% (German paced and continuous). The two "long" clips read 24.7% and 10.6% mostly because the model writes numbers as words ("nine forty five") and the script has digits: a scoring artefact, not mishearing (checked on the English text).
- Not yet shown: the whole chain (recognition, translation, voice) with this recogniser, and the owner's own voice. The T4 engine is not in the repo yet: `backend/stt/voxtral_stt.py` uses mlx-audio (Mac only).

**Voxtral through the real route on a T4 (2026-11-09, `documentation/evidence/2026-11-09/voxtral_route_t4.json`, same kernel, same 60 s owner English reading, Piper voices):** transcript error 4.1% for Voxtral against 10.7% for Whisper turbo with the same strategy, but median lag 3.06-3.15 s (en to sk / en to de, `agree_short`) and 4.14 s (`agree`) against 3.16 s for Whisper; p90 6.0-6.7 s against 5.8 s; lag grows about 6 s over the minute. Some pieces were voiced 0.3 s after the speech, others waited 2.6-3.9 s.

**Cause, measured:** the T4 with the Transformers loop runs Voxtral at 1.0-1.1 x real time (offline decode of whole clips, 12 clips in two runs: RTF 1.005-1.096; the model emits one token per 80 ms of audio and a step takes about 80 ms). There is no headroom: any slowdown becomes lag that never recovers. The standalone streaming lag also moved between two identical runs (median 0.06 s, then 1.27 s on the same English clip). The 4-bit MLX model on the M1 Pro decoded at 0.62 x real time offline, i.e. faster than this T4 setup. One fix not tried: serve the model with vLLM (CUDA graphs, Mistral's recommended server) or run on a faster GPU (L4, A10, A100) than Kaggle's T4.

**Mac default decided (2026-11-09, `scripts/live/route_check.py`, evidence `documentation/evidence/2026-11-09/voxtral_route_mac.json`):** Voxtral through the real route on this 16 GB M1 Pro, run alone at load 3.0 and swap 0.4 GB at the start, 60 s English reading, en to de, `agree_short`, 0.64 s pushes: median lag 11.4 s, p90 20.9 s, lag grew 16 s over the run, swap rose to 5 GB during it. It does not hold on this machine. **Default for the Mac and for lower machines stays Whisper** (`presets/mac.env`: MLX turbo, `agree_short`; CPU/Windows: `presets/cpu.env`, `phrase`). Voxtral stays opt-in (`presets/mac-voxtral.env`) for a Mac with more memory.

**vLLM fixes the T4 (2026-11-09, `documentation/evidence/2026-11-09/voxtral_vllm_t4.json`):** Voxtral served by vLLM 0.33.0 (`--enforce-eager --dtype float16 --max-model-len 3584`, own venv, Kaggle T4) with audio sent at real time: text arrives a median 0.07-0.08 s after the audio (p90 0.12 s, max 0.67 s), no drift, word error 1.0% / 3.1% (English paced / continuous) and 2.1% (German paced), run ends 1.5-1.6 s after the audio. The Transformers loop on the same T4 drifted by seconds. CUDA-graph mode did not start; the first attempt failed because the default `--max-model-len` (16384) needs more KV cache than a T4 has left (5.72 vs 4.37 GiB). Engine: `backend/stt/voxtral_vllm_stt.py` (`BP_STT_ENGINE=voxtral_vllm`, websocket client to a vLLM server; model-free test `test/voxtral_vllm_tests.py`). Through the whole route: pending (kernel `bp-voxtral-vllm-route-t4`, queued on Kaggle for 2 h+ at the time of writing; rerun it if it was cancelled).

**Needs a better processing unit (the work that is left):**
1. Rerun the real route with Voxtral on a machine where its offline RTF is well under 1.0 (check first: decode a whole clip, RTF must be 0.5 or lower) and, on the Mac, with 0.64 s pushes and a free GPU and spare memory (a 32 GB+ Apple Silicon Mac dedicated to the demo, or an NVIDIA box). `paced_replay.py processed/testaudio/en_paced.wav --strategy agree_short --src en --tgt de` then `score_success.py`. Pass = median lag at or under 3.0 s, early start at 50% or more.
2. NVIDIA: write the Transformers-based engine with the same `open_stream`/`stream_feed` interface as `voxtral_stt.py` (the T4 benchmark script already holds the chunking logic), then run `paced_replay.py` + `score_success.py` inside a Kaggle kernel.
3a. OmniVoice in the streaming route (design 2026-11-09): it is a non-autoregressive diffusion model (all frames per call, `num_step` iterations), not a streaming model, so streaming means one call per agreed piece. Knobs from its README and our T4 tests: `num_step` (12 for Slovak 4/5 by ear, 16 default, 8 degrades Slovak), `speed` and `duration` (cap, ratio 0.83-0.90), a reference clip of 3-10 s (ours was 12 s; longer slows inference), cached voice prompt (done), same-language reference for the best pronunciation. New: the route's rate follower (`length_scale`) now maps to OmniVoice `speed` (`test/omni_pace_tests.py`). Faster per-call latency: OmniVoice's FlashInfer path with CUDA graphs (README: 2.1x batch 1, 2.4x with graphs on an H100; needs a newer NVIDIA GPU, not tested on a T4). Not built: set `duration` from the source piece's span to match the speaker's pace exactly.
3. Voice: OmniVoice (`backend/tts/omni_tts.py`) already clones the speaker's voice (RTF 0.18-0.34 on a T4, about 0.8-1 on Apple MPS), so a separate voice-conversion stage is not needed on a GPU box. Copying the pitch contour across languages is research-grade and not planned. Not measured: whether OmniVoice plus Voxtral fit one T4.

Review of the measurement itself (four flaws found, one untried lever, better instruments): [`REALTIME_REVIEW_2026-11-09.md`](REALTIME_REVIEW_2026-11-09.md).

## Not validated yet (next actions, in order)

1. Rerun `bp-fresh-install-t4` (installer, kinit conversion, CUDA preload, German and Slovak as input speech).
2. Run `bp-room-test-t4` (3 and 12 listeners over 3 languages) and put the numbers in `BENCHMARKS.md`.
3. German through the browser UI and on a Mac; record the reading script (`reading_script_multilingual.md`) and score it with `scripts/score_reading.py`.
4. Push the local commits (4 ahead of `main` at the time of writing).

## Room mode: what is required before others can run a meeting with it (2026-11-08)

Room mode is built and unit-tested. It has not run with real models in front of real listeners. Everything below is a requirement, not a promise. Tick a line only with a recorded check.

| # | Requirement | State |
|---|---|---|
| 1 | Real-model room test (3 and 12 listeners, 3 languages): latency, ordering, no stalls | open (`bp-room-test-t4`) |
| 2 | Listener joins from a link or QR code with no login | built: the listener page takes a guest token itself; join by guest token tested through the real app (`test/room_join_tests.py`, 2026-11-09); not tried on a phone |
| 3 | Speaker page creates a room and shows code, link and QR | built 2026-11-09 ("Start room" button, server-made QR, link with the LAN address instead of localhost); viewed at 1440 px and 390 px in the browser, no horizontal overflow; typed room codes are no longer accepted (the speaker must press Start room) |
| 4 | Room is freed when the speaker leaves; listeners cannot create rooms by guessing an id; a speaker cannot publish into another user's room | built 2026-11-09, unit-tested (`end_room` on speaker disconnect, listener lookup `create=False`, room owner check; `test/room_tests.py`); not run with real sockets |
| 5 | One slow listener must not delay the others (send timeout per listener) | built 2026-11-09, unit-tested (concurrent sends, `BP_ROOM_SEND_TIMEOUT_S` default 3 s, slow listener dropped; `test/room_tests.py`) |
| 6 | A listener whose language pair is not declared or not installed is told when joining (a German speaker with a French listener has no `de-fr` model) | built 2026-11-09 on the listener side, unit-tested (`pair_unavailable`, `test/room_tests.py`); the speaker is not yet told which listeners were refused; checks the pack's whole asset list, STT included |
| 7 | LAN run for real phones | partly built: start with `BP_HOST=0.0.0.0` (the page warns when the server is loopback-only); listeners need no microphone, so plain HTTP works on the same Wi-Fi; HTTPS and the certificate warning matter only for a speaker on a phone; not tried on a phone |
| 8 | Listener page in the audience's language, readable on a phone at 390 px | built in the working tree, not reviewed on a device |
| 9 | Speaker page layout: room controls must not crowd the live page; design review | open |
| 10 | Fresh install passes on Linux (rerun), then Windows | open |

Live tests, 2026-11-08, Mac (session logs in `processed/sessions/`, not in the claims checker):

- **Why the translation arrived after the sentence ended.** The English-to-Slovak log shows the wait between a voice phrase being ready and playing grew from about 0 s to about 19 s over 50 s of speech. The browser queue measured every phrase as 0.0 s long (WAV tag byte order bug in `session-playback.js`, mirrored by its unit test), so the backlog budget and speed-up never worked. Fixed 2026-11-08; adaptive playback is now on.
- **Why the transcripts are poor.** The microphone is fine (RMS about 0.06, no clipping) and Parakeet via Type Whisper agrees on German. The cause is the fixed 2 s chunks: the same audio decoded whole is far better than decoded in chunks (German, Slovak, 62 chunks replayed), and short fragments produce stock hallucinations ("Ďakujem za pozornosť", "Titulky vytvořil JohnyX"). Slovak speech with the source set to `cs` is worse than with `sk`.
- **Decode retries.** Whisper's temperature fallback cost 4-5 s per junk chunk and did not improve the text: replayed decode time 43.6 s to 17.8 s (German) and 65.1 s to 37.3 s (Slovak) with retries off. Off by default now (`BP_STT_FALLBACK=1` restores).
- **Paced replay of the owner's German recording (30 s, real route, 1x):** median lag 1.3 s and worst 2.8 s after each segment's end with adaptive playback, translation ending 4.5 s after the input ended; first-word lag up to 4.8 s. The incremental route (`BP_INCREMENTAL=1`) was not better on the Mac (median lag 2.0 s). The floor with 2 s chunks is the chunk length plus 0.7 s recognition; a first-word target needs shorter or overlapping windows and has not been built.

**What changed since the 2025-12 demo (found 2026-11-08 by diffing `2a2daab` against the working tree).** The demo closed a segment only at a pause (`SILENCE_TIMEOUT` 0.3 s, 15 s safety valve) and sent the whole phrase to the recogniser; its mid-speech streaming block was commented out. Today's `main.py` runs that block live: every `STREAMING_CHUNK_LENGTH` (2.0 s) it removes frames from the open phrase and sends them as an independent chunk, so continuous reading is cut every 2 s and never reaches the pause. `BP_STREAM_CHUNKS=0 BP_VAD_MAX_SEGMENT_S=15 BP_VAD_SILENCE_S=0.3` restores the demo behaviour (verified on a replay with pauses: 3 phrases cut at the pauses, `_full.wav` written). Next: measure that mode by ear and by log, then the agreement route, on the same recordings.

**Strategy lab on the owner's Slovak recording** (57 s, reading lines A1-A4 twice; `scripts/live/strategy_lab.py`, recognition only, strict word match so a missing diacritic counts as an error; not in the claims checker):

| Engine | Whole file | Pause (demo) | 2 s chunks | Growing buffer, 1.0 s |
|---|---|---|---|---|
| Whisper turbo (MLX, generic) | 47.7% error | 60.2% | 73.4% | collapsed: lag 38 s, words rarely agree |
| Whisper turbo Slovak fine-tune (CPU) | 56.6% | 61.9% | 71.4% | not run |

Reading: the strategy order holds (whole < pause < chunks), but even the whole-file best case is near 50% on this voice, so cutting alone does not fix Slovak. The owner's Slovak is B1-B2 (`owner-facts`); ASR trained on native speech is known to degrade on accented speech. That is a hypothesis: it fits that two engines and Parakeet all fail on whole files, and it is untested (a native-speaker recording through the same lab would test it). The CPU run's timing is not valid (it ran under the `heavy` wrapper beside the server). English phrases cut at pauses were transcribed correctly in the live pause-mode session (a 9.4 s phrase, `20261108-145829`).

**Selectable cutting strategy (2026-11-08).** The live page has "Speech cutting" under Advanced: `phrase` (demo), `clause` (0.2 s pause or 4 s cap), `agree` (growing buffer, speak while talking), `chunks` (old 2 s windows). It is sent per session and written to the session log (`session_config` line), so every test says which strategy it used.

**Generated test speech (2026-11-08).** `scripts/live/make_test_audio.py` renders the reading script with the generic Piper voices (Slovak lili, English john, German thorsten) into `processed/testaudio/<lang>_<paced|continuous|long>.wav` with the exact reference text; `scripts/live/synth_matrix.py` runs every strategy over them and prints one table. This removes the need for a person to read into the UI for each comparison. First check (Slovak, paced, Whisper turbo MLX): whole file 17.7% error, pause-cut 26.2%, against 47.7% whole-file on the owner's own Slovak. The gap fits an accent effect but one synthetic voice does not prove it. The full matrix has not been run yet.

**Owner Slovak reading, direct recording (2026-11-08, 35 s, script lines A1-A6, Whisper turbo MLX, strict word match, `strategy_lab.py`):** whole file 43.8% error, pause-cut 41.0%, clause 57.4%, 2 s chunks 65.1%. The recording itself is hot: peak 1.0 and 2.5% of samples clipped (the script asks for peaks near -6 dB), which alone can cost accuracy. So the browser capture is not the main cause (the same engine does about as badly on a direct file). Earlier Kaggle numbers for the same speaker: kinit 19.8%, plain turbo 30.2% (`LANGUAGES.md`), against 6.5% and 10.7% on native FLEURS speech. A same-engine native-versus-owner comparison on identical scoring is in the Kaggle `strategy_matrix` kernel (stream `fleurs_sk` against `owner_sk_script`); not yet run.

**Known limit, decided with the owner 2026-11-08: Slovak spoken by a non-native speaker.** Whisper turbo, the Slovak fine-tune and Parakeet TDT v3 (through the TypeWhisper app) all transcribe the owner's Slovak reading badly (about 40-55% strict word error here; Parakeet's output on the owner's reading is largely garbled), while Parakeet transcribes generated Slovak speech of the same text almost perfectly. Native listeners understand the owner; the recognisers do not. The owner chose not to spend more time on it: Slovak as the spoken language is documented as limited for this speaker. The demo case (English spoken, Slovak or German heard) is unaffected. See `REALTIME_OPTIONS_2026-11-08.md` for what else exists.

**Success test for near real time (accepted by the owner 2026-11-08; full text and the reference systems in `NEAR_REALTIME_GOAL.md`).** English spoken, Slovak or German heard: (1) lag behind the speech position median at most 3.0 s, p90 at most 4.5 s; (2) of sentences with 8 or more words, at least half have their first clause voiced before the sentence ends; (3) lag grows at most 0.5 s per minute; (4) English recognition error at most 8% on whole phrases, no dropped phrase. Scored by `scripts/live/score_success.py` on a `paced_replay.py --strategy ...` run. Not a target: a voice within a second of the first word.

**Success test, first run per strategy (2026-11-08, generated English `en_paced`, 44 s, English to Slovak, real route on the Mac, Whisper turbo MLX, adaptive playback, one run each so differences of a few tenths are noise; `score_success.py`):**

| Strategy | 1 lag median / p90 (goal 3.0 / 4.5 s) | 2 early start (goal 50%) | 3 growth per min (goal 0.5 s) | 4 recognition error (goal 8%) |
|---|---|---|---|---|
| phrase | 1.71 / 2.26 pass | 0% fail | 0.4 pass | 1.0% pass |
| clause | 1.05 / 3.61 pass | 25% fail | 1.56 fail | 35.4% fail |
| agree | 3.58 / 5.70 fail | 50% pass | -3.5 pass | 2.1% pass |
| chunks (2 s) | 1.64 / 2.96 pass | 100% pass | 0.53 fail | 14.6% fail |

Reading: only the agreement route keeps accuracy and starts the voice before the sentence ends; it misses the lag goal by about 0.6 s. Phrase is accurate but cannot start early. Clause (0.2 s pause or 4 s cap) cuts words in half on clean speech and is the worst on quality; chunks start early but cost accuracy. Clause dropped 3 of 13 segments without audio (cause not yet found). A first version of the early-start check credited a voice to whichever sentence it overlapped; it was corrected the same day (a voice now belongs to the sentence it translates), which moved clause from 88% to 25% and phrase from 88% to 0%.

**Agreement route tuning on the Mac (2026-11-08, generated English to Slovak, one run per setting, run-to-run spread about 0.5 s):** baseline median lag 3.1 s (earlier run 3.6 s), 3 words and 1.0 s max wait 3.2 s, plus a 0.5 s step 3.1 s, 2 words and 0.8 s wait 3.0 s with early start 62% (the only setting to pass criterion 2). Fragment size and step barely move the lag: the floor is two agreeing decodes of about 0.7 s each plus voice time. The Kaggle simulation puts agreement at 1.0 s near 2.4 s on a T4, so the lag goal needs a faster decode than Whisper turbo on this Mac, or a GPU.

**Kaggle T4 strategy matrix (2026-11-08, `scripts/gpu_bench/kaggle_strategy_matrix`, simulation on the same clock model as the stream simulation, beam 1; word error / median commit latency; results in `processed/kaggle_matrix_json/strategy_matrix.json`):**

| Stream, engine | whole | pause | clause | 2 s chunks | agree 1.0 s | agree 0.5 s |
|---|---|---|---|---|---|---|
| English generated, turbo | 1.0% | 1.0% / 2.4 s | 2.1% / 2.0 s | 10.4% / 1.2 s | 6.2% / 2.4 s | 11.5% / 1.3 s |
| German FLEURS, turbo | 3.5% | 8.9% / 3.0 s | 12.1% / 1.9 s | 33.4% / 1.1 s | 10.4% / 2.3 s | 26.2% / 1.3 s |
| Slovak FLEURS (native), Slovak fine-tune | 8.5% | 19.0% / 3.3 s | 25.8% / 2.2 s | 27.8% / 1.5 s | 16.9% / 2.9 s | 21.4% / 10.8 s |
| Slovak, owner reading, Slovak fine-tune | 17.7% | 17.7% / 6.5 s | 33.9% / 2.4 s | 32.3% / 1.5 s | 27.4% / 3.2 s | 32.3% / 6.2 s |
| Slovak, owner reading, plain turbo | 45.2% | 41.9% | 56.5% | 67.7% | 48.4% | 66.1% |
| Slovak, owner reading, Parakeet | 90.3% | 100% | 82% | 102% | 98% | 95% |

Reading: no strategy gives under 1.5 s with under 10% error; agreement every 1.0 s is the best compromise (about 2.3-2.9 s, 6-17% error), as in the earlier simulation. The Slovak fine-tune reads the owner at 17.7% against 8.5% on native speech, about twice the native error, while plain turbo (45.2%) and Parakeet (90%) fail on the owner's Slovak. The English FLEURS pause and clause cells (75%) are not trustworthy: the voice-activity cut misbehaved on that stream and the other streams do not show it.

**Correction to the "known limit" paragraph above.** The 40-56% Slovak error figures came from plain turbo and from the owner's browser-captured audio. With the Slovak fine-tune on the owner's direct recording the error is 17.7%. The local CPU run of the same fine-tune on the browser-captured audio gave 56.6% (different sentences, different recording), so the browser capture path (echo cancellation, noise suppression and automatic gain are all on) is a suspect. The live page now accepts `?mic=raw` to switch those three off for an A/B reading; the session log records `mic_raw` or `mic_processed`.

**Mic A/B (2026-11-08, English, A1-A6 read once on the page with the browser's echo cancellation, noise suppression and automatic gain on, once with them off; offline, Whisper turbo MLX on the saved full recordings, `strategy_lab.py`):** processed capture 6.7% whole-file error and 15.5% when cut at pauses; raw capture 2.7% and 4.0%. The raw recording is much quieter (peak 0.10, rms 0.011 against peak 0.51, rms 0.051) and still recognised better, so the processing is the problem, not the level. One take each, different strategies in the live sessions (phrase against clause), so treat it as strong evidence and not proof. Raw is now the default in `live-script.js` (`?mic=processed` restores the old behaviour); the session log records `mic_raw` or `mic_processed`.

**Live page options added 2026-11-08:** `agree_short` ("Agreement, short pieces") voices 2-6 agreed words or after 0.8 s; it is the only agreement setting that passed the early-start test on the Mac (62%, one run).

Diagnostics for the next live test (`BP_LOG_SESSIONS=1`): every recognised chunk is saved as `processed/sessions/<stamp>_<client>_chunks/NNN.wav` with the raw recogniser output (before the guards), the engine and the prompt in the session `.jsonl`; the browser logs `tts_received`, `tts_play_start` and `tts_ended` with the playback queue length. A wrong transcript can be replayed offline with any engine, and playback queueing can be separated from chunking and recognition.

## Not built

1. **Cloned voice for other people.** Only a Slovak fine-tune exists (local, not distributable). Cloning needs a GPU and the weights carry non-commercial licences.
2. **Room-mode polish:** see the table above; plus authentication beyond the room code and a capacity ceiling per GPU.
3. **Operations for shared use:** retention, monitoring, a support owner.
4. The old `/ws/meeting` scaffold (`backend/session_manager.py`) is a separate, unvalidated pipeline with one target language per speaker. Room mode does not use it.

## Licences (owner decision 2026-11-07)

The project is a configuration layer. It bundles no model weights (no `.onnx`, `.bin`, `.pt` or `.safetensors` file is tracked) and its setup downloads each model from its publisher, so the user accepts that model's licence (`LICENSE_INVENTORY_2026-11-06.md`). Whoever runs it as a shared or hosted service, such as a room for other people, is the operator and should pick models whose licences allow that (Piper is GPL, OmniVoice and XTTS weights are non-commercial).

## Public site

`https://brusnyak.github.io/bp/` serves the page generated by `scripts/build_site.py` (published 2026-11-07). Regenerate and republish it to `gh-pages` `docs/index.html` after new evidence lands.
