# Voice + app direction — 2026-09-19

Owner requirements: (1) one recorded voice, registered like a Piper voice, fast and reliable,
no slow zero-shot path in the live loop; record however long is needed, system handles the rest.
(2) Multilingual from that single recording — EN↔SK both ways (plus other pairs), fast.
(3) Runs on any OS (mac, linux, windows), works anywhere (web, zoom, other apps) with
subtitles across windows — i.e. a desktop app, not a browser tab.

Project status: core pipeline completed earlier (STT→MT→TTS, Hybrid sub-real-time cloning,
non-blocking pipeline + barge-in T043–T047). Remaining work is polish + packaging, not research.

## 1. Zadanie mapping (FEI-184458-129117, deadline 05.06.2026)

The assignment asks for: tech survey (STT/MT/TTS for EN + chosen language) → architecture +
prototype with suitable tools → simple test UI (subtitles or playback) → accuracy/latency
analysis in a typical conference setting. Backend focus, prototype bar. The current browser
pipeline already satisfies every numbered task. A desktop app with system-wide subtitles and
automated virtual-mic routing is above the assignment bar — Track B below, not required
for defense. Thesis language: Slovak.

## 2. Voice decision: Hybrid registered profile is the default, Piper fine-tune is optional

Plain Piper (VITS) fine-tuning cannot satisfy requirements (1)+(2) together, and this must
stay stated plainly: a fine-tuned Piper model is bound to the phonemes of its training
language. An EN recording yields an EN voice; SK output needs SK recording + SK base
checkpoint + separate fine-tune. "Record once, get EN↔SK both ways at Piper speed" is not
a Piper-fine-tune property at any data scale. It is a voice-conversion property — which is
exactly what the already-built HybridTTS (Piper per-target-language base + OpenVoice V2
tone-color converter) does: RTF 0.15–0.17 cross-lingual on M1 Pro, seconds of reference
needed, no per-language retraining.

Resolution (spec US5 revision, same day):

- **Default "registered voice" = Hybrid speaker profile.** User records once (spec below);
  system normalizes → stores profile in `speaker_voices/speaker_voices.json` → extracts the
  tone embedding once at init and caches it (already implemented). At runtime the pipeline
  picks the Piper base for the *target* language (`sk_SK-lili` / `en_US-ryan` / `cs_CZ-jirka`)
  and stamps the user's tone on it. From the user's perspective this is "my voice, in the
  registry, fast like Piper" — because synthesis IS Piper plus a small converter overhead.
- **Piper fine-tune ONNX (`piper_personal`, `piper_personal_v2`) = optional single-language
  premium tier**, not the default. Worth it only with 10–30 min of clean audio *per language*
  (EN and SK recorded separately) + a listening/QC pass (`scripts/voice_similarity_qc.py`).
  Never from ~90s + synthetic bootstrap without a similarity measurement — RTF proves speed,
  not identity.
- **XTTS zero-shot stays the last-resort fallback** for languages with no Piper base and no
  profile. Slow (2.5–3.5s), honest ceiling, must be labelled as such in the UI.

### 2b. Sanctioned build path, added 2026-09-19: one recording → per-language Piper voices

Nothing in the toolchain prevents what the owner actually wants, once it is stated per-language
instead of as one bilingual model: record once → cross-lingual clone generates the target-language
corpus in the user's voice → fine-tune one Piper voice per target language → each registers in
`TTS_ENGINES` and ships as a standard Piper pair (`.onnx` + `.onnx.json`, same layout as any
HuggingFace Piper voice). Concrete steps per target language (e.g. SK):

1. Record once in EN (2–3 min clean; already have ~90s + Rainbow Passage).
2. Generate ~30+ min of SK audio in the user's voice with the cross-lingual cloner
   (Hybrid/OpenVoice converter — bypasses the XTTS-has-no-SK wall; XTTS-cs proxy only with
   explicit sign-off). Fixed diverse sentence set, target-language text.
3. Mix real EN clips at 2x oversampling against the synthetic SK (ZeSTA-style mitigation —
   pure-synthetic training measurably degrades speaker similarity).
4. Fine-tune warm-started from any *medium* Piper checkpoint — piper1-gpl TRAINING.md states
   `--ckpt_path` speeds training "even if the checkpoint is from a different language";
   checkpoints live at `datasets/rhasspy/piper-checkpoints` (dataset repo, not model repo).
   This unblocks the DE case from `sk_de_bootstrap_findings_2026-08-19.md`: no German
   checkpoint needed, warm-start from EN medium. `espeak-ng --voices` covers `sk`.
5. Export ONNX, drop into `backend/tts/piper_models/sk_SK-personal-medium.onnx`, register
   factory in `backend/tts/base.py`, QC before claiming identity: blind listening +
   `scripts/voice_similarity_qc.py` cosine against the real reference. RTF proves speed,
   QC proves the voice.

Honest limits that stay in force: synthetic-trained voices inherit converter artifacts (QC is
the gate, not an afterthought); each target language is a separate build (EN voice ≠ SK voice,
two registry entries, one speaker); ~90s of real audio is the floor for the Hybrid live path,
not for a quality fine-tune — volume comes from step 2, identity from steps 3+5.

## 3. Recording spec (whatever length, system handles the rest)

- Minimum for a usable Hybrid profile: ~60s clean speech, any language (EN fine), quiet room,
  16kHz+ mono. Recommended: 2–3 min with varied sentences (already have ~90s + Rainbow Passage).
- Longer recordings do not retrain anything — they improve embedding stability only, with
  diminishing returns past ~3 min. No transcription needed for Hybrid (embedding only);
  transcription required only if the user opts into the Piper fine-tune tier.
- Flow: record/upload via existing voice modal → `scripts/prepare_voice_corpus.py` normalize +
  dedupe → profile entry + embedding cached → selectable in TTS dropdown as the fast cloned
  voice for any target language. No manual checkpoint hunting, no venv juggling.
- Recording commitment 2026-09-19: no SK recording of the owner exists yet; owner will record
  on demand. Plan: one EN top-up session (varied sentences, 2–3 min total with existing clips)
  + one SK read session (varied, 2–3 min). SK session doubles as real anchor data for step 3
  of the §2b build (real SK oversampling beats EN-anchored synthetic-only) and as the future
  SK fine-tune corpus. Until recorded, SK corpus is synthesized from EN via Hybrid.

## 4. App-ification: two tracks

- **Track A (thesis-sufficient, now):** browser UI + virtual-mic select + rendered
  `caption_partial` streaming captions + latency charts. Covers zadanie task 3–4 as written.
- **Track B (post-defense unless time allows):** desktop shell (Tauri/Electron) hosting the
  same FastAPI backend: autostart daemon, per-OS audio capture (BlackHole / WASAPI loopback /
  PipeWire null-sink) automated per constitution principle V, always-on-top caption overlay
  across windows, one-click virtual-mic routing into Zoom/Meet/Teams. No pipeline model
  changes — packaging only. Must not be promised to the handler as thesis content.

## 5. Open items before handler meeting
- Push the 9 unpushed local commits (T043–T047) so the remote matches the demo.
- Render `caption_partial` in `ui/live-speech/` (backend plumbing exists, frontend missing).
- One blind listening test (Hybrid SK vs fine-tuned EN vs XTTS) to close the similarity gap
  with a number instead of an assertion.
- Confirm deadline standing: zadanie lists 05.06.2026; verify extension/standing with the
  department before presenting Track B as in-scope.

## 6. Voice Lab eval page (built 2026-09-24, branch `voice-lab`, local-only)

`https://localhost:8000/ui/voice-lab/lab.html` — one screen replacing file-hunting:
speaker voices with playback (`/api/voices` + `/speaker_voices/` mount), QC candidates
with latency + similarity (`processed/voice_qc/`, served at `/voice_qc/`), system status
(`GET /api/voice-lab/status`: engine flags, per-stage hardware backends, Piper models on
disk, QC manifest/scores presence). Read-only, no auth, no mutations. Verified live:
status 200, page 200, `hardware_test.py` 7/7. `scores.json` appears once `--score-only`
runs in the qc venv.

## 7. Env raw-test plan (deferred until project is complete)

Push full project → remove local envs/caches → clean checkout on this Mac + second ARM
laptop → single plug-and-play path (one setup script: venv, models incl. Piper/XTTS,
certs, env vars/creds) → record total footprint and every manual step as a bug. Env diet
only after (§6-context, 2026-09-24): venv 2.2G + .venv-train 1.4G + HF cache 1.8G ≈ 5.4G;
candidates are archiving `.venv-train` post-SK-build and dropping playwright browsers.
