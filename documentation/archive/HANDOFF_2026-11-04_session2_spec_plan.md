# Handoff 2026-11-04 (session 2): state, spec, plan

Read this first in the next session. Source of truth for measurements is `processed/` and the Kaggle kernels named below. Nothing is committed or pushed (working tree only; pushes are the owner's).

## 1. State (proven vs not)

### Streaming and latency (measured this session)
| Setup | Result | Evidence |
|---|---|---|
| Mac, Whisper base int8 + Opus-MT + Piper, normal priority | STT 0.35-0.5 s, MT 0.05-0.15 s, TTS 0.04-0.16 s per segment | `processed/paced_n1.json`, `paced_n2.json`, `paced_n3.json`, `paced_v1.json`, `paced_v2.json` |
| Same under `heavy` wrapper (background QoS, 3 threads) | 5-9 s STT: **artifact, do not benchmark through `heavy` unless `HEAVY_QOS=normal`** | server log of first runs |
| T4, large-v3-turbo, 4 pairs, 60 s each | all segments voiced; median lag 0.6-0.9 s; SK/CS source recognition poor | `processed/kaggle_lang_2026-11-04/` |
| T4 concurrency, EN->SK, 45 s per client | 1/2/4/8 clients: median lag 0.6/0.9/1.2/3.7 s, no failures, 8 clients degrade | `processed/kaggle_conc_2026-11-04/` |
| Incremental word-by-word mode (earlier kernel) | worse than short segments (median 1.6-1.8 s vs 0.4 s, dropped segments) | `documentation/implementation-2026-11-04/kaggle-t4-incremental.md` |

Findings: pre-VAD volume gate (0.02) discarded quiet speech (local run: 1 of ~8 segments voiced). Remaining listener lag on EN->SK is playback backlog (Slovak audio ~3 % longer than source, one-clip-at-a-time player), not compute. First-word lag is 4.5-8.5 s upper bound (segment must close first). Single runs, one speaker: indicative only.

### Code changes made (uncommitted)
- `backend/main.py`: `BP_SILENCE_RMS` env, default **0 (gate off)**; `MAX_SEGMENT_S` 3.0; `SILENCE_TIMEOUT` 0.25.
- `languages/packs/*-en.json` (7 packs): silence 0.3 s, max segment 3.0 s (was 0.4-0.5 / 6.0).
- `app.py`: plain HTTP default; `BP_HTTPS=1` for TLS. Also `scripts/setup.py`, `scripts/demo_preflight.py`, `scripts/bp.py`, `scripts/live/interrupt_smoke_test.py`, `scripts/vad_close_probe.py`, `Makefile`, `ui/test-stt/test-stt.js`, `AGENTS.md`.
- `README.md` rewritten (YouTube thumbnail, measured numbers, env table, readiness gates).
- UI: `ui/home`, `ui/live-speech`, `ui/auth` (+`session-nav.js`), `ui/enroll`, `ui/extra/{privacy,cookies}` (markdown `**` fixed), `ui/sw.js` v11. Second pass by the lead: demo thumbnail facade (click to load player), LinkedIn link fixed to `https://www.linkedin.com/in/yegor-brusnyak/`, FAQ styling restored, pipeline as 6 responsive step cards, live page diagnostics collapsed.
- Kaggle jobs added: `scripts/gpu_bench/kaggle_lang/`, `kaggle_conc/` (private dataset `yegorby/bp-lang-code-20261104` holds 60 s clips of the owner's voice; delete if not wanted).
- Tests at last run: 71 backend tests pass (subset in Makefile), 7/7 playback (`node --test test/session_playback_test.cjs`). Not re-run after the last UI pass: re-run.

### Known defects / not verified
1. **Google sign-in fails: `Error 401: deleted_client`.** The OAuth client ID in `.env` (`GOOGLE_CLIENT_ID`) was deleted in Google Cloud. Owner action: Google Cloud Console > APIs & Services > Credentials > create OAuth client ID (Web application), authorised JavaScript origin `http://localhost:8000` (add the demo host too), put the new ID in `.env` as `GOOGLE_CLIENT_ID`, restart. Do not print or commit the value. UI note: auth page should hide the Google button when the ID is missing (it shows a "not set up" note); a deleted ID cannot be detected client-side.
2. **Login is required to initialise models** by design: `/ws?token=<JWT>` and HTTP `/initialize` demand a token (`backend/main.py`). Owner decision 2026-11-04: keep login. Option for local use: loopback-only guest session (disabled when `BP_HOST` is not loopback). Not built.
3. Remaining UI: extras `terms`, `about`, `faq`, `feedback` not proofread; dark-theme Initialize button contrast not measured; phone layouts of live/enroll only checked for horizontal scroll; the auth flow was never exercised against the live backend; Google SVG button untested with a real ID.
4. Slovak/Czech recognition quality poor (Whisper base and large-v3-turbo). `kinit/whisper-large-v3-turbo-sk` not run yet.
5. Designer-agent process failure (see lessons ledger): it deleted FAQ CSS, left the demo as a blank iframe, left a tiny mermaid diagram, left markdown in legal pages, and reported pass on screenshots it did not inspect closely. Review its output with screenshots yourself before accepting.

## 2. Product spec (commercial and open source)

### 2.1 Install in 30 minutes or less (plug-and-play)
Acceptance: fresh macOS (M-series) and Ubuntu 22.04, no repo knowledge, `git clone` to first translated sentence <= 30 min on a 100 Mbit line, timed and recorded.
- One command `make install` (exists: `scripts/setup.py`); add `make doctor` (what is missing, how to fix) and `docker compose up` (CPU image with models pulled from a release, not converted on the user machine).
- Pre-converted CTranslate2 MT models and Whisper CT2 weights published as release assets or a Hugging Face repo (licence permitting) to remove the conversion step.
- Default voices must be commercially clean (see 2.4).
- `.env` generated with a random `JWT_SECRET`; Google sign-in optional.

### 2.2 LHC and conference integration: modes to support
Evidence note: I could not find a source for the Chinese classroom system the owner described (search returned XJTLU's human-interpreter lab and KIT's Lecture Translator). Closest published precedent: KIT Lecture Translator, web captions per student device. Treat the Chinese system as unverified; ask the owner for a link or name.

Architecture (one server, N listeners):
- **Mode A, single speaker, many listeners (lecture / room):** one capture stream (room mic or lectern) -> VAD -> STT -> then per distinct target language one MT + TTS. Each listener opens a WebSocket (phone browser, own earbuds), chooses a language, and receives captions and/or TTS audio. Cost scales with number of distinct languages, not listeners (broadcast the same audio per language). This matches "everyone in one room, each with an earpiece".
- **Mode B, virtual cable into Teams/Zoom (exists):** one user, one direction.
- **Mode C, multi-speaker meeting:** needs speaker diarisation or per-speaker mics (one stream per participant). Not built. Cheapest first step: one stream per participant (each joins from own device), which the current server already handles up to a measured cap.
- **Mode D, captions on a screen:** captions-only page, projector style (subtitle strip and PiP exist).
Needed facts from the owner: LHC platform (Teams/Zoom/room hardware), who provides audio, rooms and language pairs, headset hardware.

Requirements: session cap with a clean "server busy" message (cap 4 per T4 from measurement, configurable); per-session isolation of voices and transcripts; retention policy; rate limiting kept on in shared deployments.

### 2.3 Tests required before claiming meeting readiness
1. Concurrency: repeat Kaggle `kaggle_conc` with CPU Piper replaced by the chosen production TTS, 1/2/4/8/12 clients, duty cycle 0.25 and 1.0, 5 minutes each, report p50/p95 lag and failures.
2. Model pool limits: `MODEL_POOL_CONFIG` (`backend/main.py` ~l.293: 3 STT, 5 MT, 5 TTS) under mixed language pairs (e.g. 4 clients EN->SK, 3 SK->EN, 2 EN->CS): check eviction, reload latency, memory. Not run.
3. Multi-speaker: two-voice scripted meeting (`scripts/meeting_sim.py`) moved onto Kaggle with overlap and interruptions; measure dropped/merged turns.
4. End to end with real hardware: physical mic, speakers or earbuds, virtual cable, 10-minute session; record first-audio p50/p95 and queue growth.
5. Mode A broadcast prototype: 1 speaker, 6 listeners, 2 target languages.
6. Soak: 30 minutes continuous, memory and handle leaks, WebSocket reconnect.
7. Quality: WER per source language on 3+ speakers; chrF/COMET on translations; human check of 20 sentences per direction.

### 2.4 Voices and licences (commercial)
Problem: default Piper voices `en_US-ryan` is CC BY-NC-SA 4.0 (non-commercial); `lessac` licence unread; OmniVoice weights CC-BY-NC; `sk_SK-lili`, `cs_CZ-jirka` licences unverified in this session (check each `.onnx.json`/model card). A commercial build cannot ship NC voices.
Plan: build own default voices with **Piper Recording Studio** (the owner's "Voice Studio"; https://github.com/rhasspy/piper-recording-studio, exports LJSpeech format; confirm its licence file) and fine-tune Piper from a permissive base checkpoint. Guide numbers: about 1300 phrases to fine-tune, 13000 from scratch (community guides, unverified). Publish voices under CC0/MIT/CC-BY with consent records. Zero-shot cloning stays opt-in (user's own voice, local only). The owner's personal fine-tunes must not ship in the repo.
Also: Opus-MT is CC-BY-4.0 (attribution needed), faster-whisper/Whisper MIT, Piper runtime GPL-3.0 (check implications for a closed commercial product; the open-source path is fine).

## 3. Plan (ordered)
1. Owner: new Google OAuth client; answer LHC questions; approve home copy.
2. Re-test after UI pass (tests + screenshots of auth, enroll, extras, phone live page); proofread legal pages; measure dark contrast.
3. Run `kinit` Slovak STT on 3 clips (Kaggle), WER table; decide default SK model.
4. Session cap + "busy" message; guest/local mode decision.
5. Licence audit table (voice, model, runtime) in `documentation/licences.md`.
6. Voices: record with Piper Recording Studio, fine-tune, publish.
7. Concurrency, model pool, multi-speaker and real-hardware tests per 2.3.
8. Mode A broadcast prototype for LHC rooms.
9. Open source packaging: `make doctor`, Docker, release assets, clean-machine install timing, CI matrix (exists), contributing guide.
10. Thesis: reconcile `thesis_draft.md` with measured numbers; presentation EN->SK demo script (`documentation/demo_runbook_2026-09-28.md`); supervisor follow-up after 2026-11-09.

## 4. Do not repeat
- Do not benchmark through `~/.claude/scripts/heavy` without `HEAVY_QOS=normal`; background QoS gave 10x slower stages.
- Do not claim a latency gain from a single replay: variance between identical runs was 2.4-8.7 s growth.
- Do not accept a subagent's UI "pass" without viewing screenshots (home, FAQ, demo, pipeline, phone).
- Never print `.env` values; Kaggle datasets hold the owner's voice clips: keep private.

## 5. Commands
- Local replay: start `BP_NO_SSL=1 BP_PORT=8013 .venv/bin/python app.py`, then `.venv/bin/python scripts/live/paced_replay.py speaker_voices/me_en_script.m4a --base http://127.0.0.1:8013 --src en --tgt sk --tts piper --seconds 40 --rates 1.0 --out processed/x.json`
- Kaggle: `kaggle kernels push -p scripts/gpu_bench/kaggle_conc`, `kaggle kernels status yegorby/bp-conc-t4-20261104`, `kaggle kernels output ...`
- Tests: `make test`, `node --test test/session_playback_test.cjs`

## 6. Addendum (later 2026-11-04): guest mode, VoiceStudio, LHC scenarios

### 6.1 Guest mode (built, tested)
- `POST /api/guest` returns a token for the shared token-only account `guest@guest.local` (empty password hash, cannot password-login). `GET /api/auth/config` now includes `guest_enabled`.
- Enabled when `BP_GUEST=1`, or by default when `BP_HOST` is loopback; `BP_GUEST=0` always disables. A server exposed to a network must opt in.
- Guests can initialise models and translate; voice upload/rename/delete return 403 (`_deny_guest`).
- UI: live page auto-starts a guest session when no token exists (and retries once if the guest token is rejected); auth page shows "Try it as a guest"; the nav keeps "Sign in" for guests.
- Evidence: 2 new tests in `test/backend_auth_tests.py` (8 pass in file; full subset passes, 7/7 playback); real-server browser run on port 8015 with a fake microphone: no token -> guest token stored -> WebSocket connected -> Initialize reached "Initialized" with no console errors. Not tested: a real microphone, Start, translated audio in a browser.
- Open: guests share one account (shared voice list/history if any); add per-guest isolation before a public deployment.

### 6.2 VoiceStudio (https://github.com/debpalash/voicestudio)
- AGPL-3.0 desktop/server app: voice cloning, voice design, dubbing, dictation, transcription, 646 languages; default engine OmniVoice (k2-fsa); "Model Manager" shows each model's licence; weights, voices and generated output have their own licences (README "License & responsible use").
- Use: as an offline tool to design/clone and generate training audio for our default voices. Do NOT link it into BP (AGPL network-copyleft). Before using output commercially, check the licence of the exact model used (OmniVoice weights were noted CC-BY-NC in the vault, unverified here). Plan stays: record with Piper Recording Studio or generate with a permissively licensed model, then fine-tune Piper, publish under CC0/CC-BY with consent records.

### 6.3 LHC facts (primary source) and scenarios
Source: FEI STU page for the Laboratory of Hybrid Communication (LHC), https://www.fei.stuba.sk/sk/veda-a-vyskum/vyskumne-pracoviska/ustav-multimedialnych-informacnych-a-komunikacnych-technologii/laboratorium-hybridnej-komunikacie-lhc , read 2026-11-04. It says: a meeting room where participants are present physically or remotely; emphasis on every participant having their own place; remote connection for individuals and for groups or other remote rooms; development of multimedia capture and display of 3D space and objects; collaboration with Accenture on a hybrid communication platform; contact prof. Ing. Gregor Rozinaj, PhD. The page gives NO equipment list, platform, audio hardware or room number. (The "B405, administrator, on demand" text the owner pasted is a search-engine summary, not verified against the page.) The Chinese glasses-based interpretation system the owner mentioned: no source found; owner says it is a feature of smart glasses.

Scenarios (unknown = ask Rozinaj or the BP supervisor):
| # | Scenario | What BP needs | Fit today |
|---|---|---|---|
| 1 | One remote person joins the LHC meeting and needs EN<->SK | Browser session per remote person; virtual cable into the meeting software for the room side | Works for one direction per tab (Mode B) |
| 2 | Participants in the room each have a seat; each wants own language in earbuds ("per seat") | Mode A broadcast: one STT, per-language MT+TTS, one WebSocket per seat; seat = phone or small device with earbuds | Not built; architecture in section 2.2; per-seat model matches the lab's "every participant has a place" |
| 3 | Remote room or group joins | Captions-only or one audio return stream per room; echo isolation | Captions page exists; audio return not designed |
| 4 | Accenture hybrid platform integration | Unknown API/SDK; assume none until confirmed | Unknown; do not promise |
| 5 | Smart-glasses captions (owner's China example) | Captions pushed to a per-user display (WebSocket/WebVTT or a simple web view) | Captions strip/PiP exist; a glasses client is out of scope for the thesis |
Questions to send to the supervisor (draft only, owner sends): which meeting software the room uses; is there a room audio DSP/mixer with per-seat outputs; may BP run on a lab server (CPU/GPU, OS, network); can participants use their own phones/earbuds; number of rooms and languages; is the Accenture platform extensible.

## 7. Addendum 3 (end of day 2026-11-04): voices, roast, hardening

### 7.1 Voices decision (verified from model cards, `documentation/licences.md`)
- SK `sk_SK-lili` and CS `cs_CZ-jirka` are CC0: already commercially clean. EN default changed from `ryan` (CC BY-NC-SA) to `en_US-john-medium` (public domain, chain john <- kristin). Local test (6 sentences, Whisper base ASR): john WER 0.044, kristin 0.075, norman 0.080, RTF ~0.03; ryan 0.0 (non-commercial, removed). Listen before accepting: `processed/voice_candidates_2026-11-04/*.wav`.
- `lessac` (Blizzard 2013) is research-licensed: voices fine-tuned from it (amy, joe, arctic, kusal...) are not safe. Owner personal voices: `en_US-personal-v2` was warm-started from `ryan` (NC): retrain from `john`/`kristin` (EN) and `lili` (SK) before shipping. VoiceStudio (AGPL) = offline helper only.
- Open: lili intelligibility (vault note WER 0.25 on 8 sentences) not re-tested; best clean SK path = fine-tune `lili` on the owner's SK recordings (clean chain, own voice, owner consent).
- Changed: `backend/tts/base.py`, `scripts/setup.py`, `scripts/demo_audio.py`, `languages/packs/*-en.json`, Kaggle job scripts. `.onnx` files for john/kristin/norman were downloaded into `backend/tts/piper_models/` (git-ignored).

### 7.2 Roast (agent `roast`, read-only; verdict "Rework: claims and install proof ahead of evidence") and what was done
Done now: home copy no longer says "working" (now "demonstrated routes: speed measured, quality not scored"); footer matches MIT licence; mermaid CDN script removed; server-side register validation (password >= 8, email shape, reserved `@guest.local`); Google linking refused when a password account exists (409); guests denied on `/ratings`; voice upload: chunked 413 cap, ffmpeg 60 s timeout and `-t 120`, generic error text; rate limiter default ON when `BP_HOST` is not loopback (`BP_ENABLE_RATE_LIMIT` 1/0 overrides; stale `BP_DISABLE_RATE_LIMIT` now honoured as the test switch); **session logs (transcripts) are now opt-in** `BP_LOG_SESSIONS=1` (were always written to `processed/sessions/`); privacy page rewritten to the real data flow; dark Initialize contrast fixed; enroll icon font fixed; CI runs the full test subset; pack `license` fields now evidence-based; evidence JSONs copied to `documentation/evidence/2026-11-04/`; README: ffmpeg/Python note, env vars, Integration section, licence row. 7 new tests (guest x2, validation, upload cap, limiter default); subset 76 pass, node 7/7.
Not done (ordered): (1) score the headline routes: NTREX-128 >= 50 sentences EN->SK/CS, chrF++ on the MT path plus native review of 20 outputs (~2 h, local); (2) commit and push (owner), then clean-clone install timing on macOS and Ubuntu; (3) session cap + "busy" message; per-guest isolation (guests share one account and one init-session key `http_init_user_<id>`); (4) licence-check `whisper-small-sk`; (5) WS token in query string appears in access logs (uvicorn info level); `speaker_voices.json` has no lock and `id = len+1` can collide after delete; (6) a11y: aria-live on whole transcript columns, `outline: none` at live-style.css (~l.657, 1228), `auth.js:170` `.split` on a possibly missing attribute; (7) self-host fonts/icons; (8) AGENTS.md stale sections; (9) send supervisor questions (section 6.3); (10) per-seat broadcast prototype.
Roast corrections: PNG ignore claim was mostly wrong (ui/images PNGs are tracked); I added `!ui/images/*.png` anyway. README "0.4 s" lag was wrong (data says 0.6-0.9 s); fixed.
