# Demo runbook — live EN→SK in my own voice (2026-09-28)

Purpose: one page to run the thesis demo from, on either laptop, without re-deriving
anything. Companion files: `documentation/monday_test_checklist.md` (the longer
pass/fail matrix), `documentation/handler_update_2026-09.md` (the message + the
numbers), `PLAN.md` (what is done vs open).

Evidence this runbook is based on: live WS rehearsal on the M1 Pro, 2026-09-28
(`test/interrupt_smoke_test.py` → 9 translations, 5 partial captions, 222 TTS audio
chunks, final metrics **STT 0.51s / MT 0.10s / TTS 0.17s / total 0.78s**), plus the
CPU-only Windows 11 run (`79a8476`: STT 0.65s → MT 0.05s → TTS, e2e ≈ 3.6s).

## 0. Pre-flight — one command (T−15 min)

```bash
make run &                       # backend on https://localhost:8000 (self-signed cert)
make demo-check                  # = python3 scripts/demo_preflight.py --server
```

`demo-check` fails only on things the demo cannot start without (certs, `.env`
secrets, SK voice, MT models, registry, live server). Fallback assets are warnings.

| FAIL you might see | Fix |
|---|---|
| `certs/*.pem` | `make certs` |
| `.env` / `GOOGLE_CLIENT_ID` / `JWT_SECRET` | copy from the Mac by hand (never in git) — `documentation/amd_fetch_checklist.md` |
| `sk_SK-personal-male-medium.onnx` | `git checkout` it (shipped binary) — do not retrain |
| `ct2_models/...` | `make install` (or run the three `convert_opus_mt_to_ct2` commands) |
| server checks | `make run` first, then re-run; `lsof -nP -iTCP:8000 -sTCP:LISTEN` |

## 1. Pre-warm (T−10) — this is what makes the first sentence look fast

1. Open `https://localhost:8000/ui/live-speech/live.html`, accept the self-signed cert.
2. Log in (Google button, or password fallback).
3. Source `en` → target `sk`, TTS `piper` (auto-remaps to the personal SK voice), click
   **Initialize Pipeline** and wait for "initialized".
4. Speak one throwaway sentence, wait for translated audio, press **Stop**.

Now whisper, Opus-MT and the Piper SK voice are resident; the first on-stage sentence
is ~0.8s of compute instead of ~20s of model loading.

## 2. Optional (T−8) — virtual mic + PiP, if the handler wants the "real meeting" view

- Select the virtual device (BlackHole 2ch / VB-Cable) in the live page's output dropdown,
  then set it as the microphone in Meet/Zoom/Teams.
- Click **Pop out subtitles** (PiP) and drag it over the meeting window.

## 3. The stage script (8–10 min, in order)

| # | Do | Say (short) | Watch for |
|---|---|---|---|
| 1 | Speak 1 EN sentence, e.g. the script sentence | "This is translated live, locally, in my own voice." | captions appear ≈3s, translated audio follows |
| 2 | Speak a second sentence while the first is still playing | "…and a new utterance interrupts the old one" (barge-in) | first audio stops, new one plays |
| 3 | Open `lab.html` | "these are the voice candidates I compared, with numbers" | personal male SK vs generic lili |
| 4 | Play 1 lab A/B pair | "same sentence: generic Czech-base voice vs my fine-tuned SK voice" | timbre difference is audible |
| 5 | Show `PLAN.md` + `documentation/thesis_draft.md` | "what is measured, what is still open" | — |
| 6 | Close with the honest gap (below) | "SK→EN speech recognition is the binding constraint" | — |

## 4. Numbers to quote (measured, one file each)

| Claim | Value | Source |
|---|---|---|
| EN STT WER | 0.077 (`base`); 0.023 Parakeet-TDT-v3 (adopt path) | `PLAN.md`, `processed/stt_baseline.json` |
| SK STT WER | 0.41 turbo (adopted), 0.49 small | `documentation/tts_landscape_2026-09.md` |
| MT throughput | 18 sentences in 0.50s (chunked) | `PLAN.md` |
| TTS | personal SK voice, 0.28s per 6s of audio | `documentation/handler_update_2026-09.md` |
| Live sentence (rehearsal) | STT 0.51s / MT 0.10s / TTS 0.17s, total **0.78s** | `/tmp` rehearsal log, 2026-09-28 |
| Windows 11, CPU-only | e2e ≈ 3.6s (no XTTS: Coqui has no Windows wheels) | `79a8476`, `SETUP_WINDOWS.md` |

## 5. Fallback ladder — if audio dies on stage

1. **Live sentence** EN→SK in the personal voice (the primary demo).
2. **Voice Lab A/B** — `https://localhost:8000/ui/voice-lab/lab.html`, or fully static
   (`make lab` → `http://localhost:8080/ui/voice-lab/lab.html`): plays pre-rendered
   clips, no backend, works offline.
3. **`processed/e2e_ensk_sk.wav`** (53s, full EN→SK proof, plays in any player).
4. **Screenshots** — `documentation/visuals/`.
Never debug live: switch down the ladder, keep talking.

## 6. Rehearse it as a test, not by hand

```bash
make run &
venv/bin/python test/interrupt_smoke_test.py     # drives the real /ws with test wavs
```
A green run prints `transcription_result` → `translation_result` pairs and a pile of
`tts_audio` chunks, and the server log ends with a clean
`Client ... disconnected` line (no ERROR). Session JSONL for anything you do in the
browser lands in `processed/sessions/` automatically.

## 7. Rough edges — state them before they are noticed

- SK STT is the weak stage (0.41 / 0.63 plain); it is the honest bottleneck, not a bug.
- SK MT reads literally on misheard words (rehearsal produced "do pomalého wacku" for
  "into Slovak") — expected behaviour of Opus-MT on a bad transcript.
- Two machines, two capability profiles: Mac has XTTS/Hybrid (Coqui installed), the
  Windows AMD laptop is CPU-only Piper. Say which one is on screen.
- Self-signed cert: the browser warns once — click through before the audience is watching.

## 8. Teardown

```bash
# close the browser tab, then Ctrl-C the server
ls processed/sessions/          # transcripts, translations, per-stage latencies
git status && git log --oneline -5   # what changed while demoing
```
Write down anything that broke, same day, in `PLAN.md` → *Now*.
