# BP demo script — LHC technical coordinator (60–90 s video)

**Target:** English → Slovak live speech translation, measured, local pipeline.
**Hardware:** Mac M1 Pro, 16 GB (or NVIDIA box); Chrome/Edge; headphones.

## What to show (in order)

| # | Beat | What the viewer sees/hears | Evidence file |
|---|------|----------------------------|---------------|
| 1 | Intro (≈10 s) | "This is BP — local speech translation and live captions. Everything runs on this machine; no cloud speech service." Show hardware label and `localhost:8000` open. | `README.md` |
| 2 | Speak English | Read EN section A (reading_script_multilingual.md) into the microphone, e.g. "Good morning, and welcome to this live demonstration of speech translation." | reading script |
| 3 | Transcription + Slovak translation | Show incoming English transcript, then Slovak translation appearing below it. | live page (WebSocket `transcript` / `translation`) |
| 4 | Slovak speech + captions | Play the generated Slovak audio while Slovak captions display. | pipeline TTS (Piper default) |
| 5 | Measured latency | Show the lag number with defined start/end: end of English speech → first Slovak audio. Median 0.79 s, p90 1.39 s on this configuration (one run, generated audio). | `documentation/evidence/2026-11-15/D_wait05.json` + `.score.txt` |
| 6 | Technical terminology | Read EN section B with names/numbers ("Mr. Novák from Bratislava … 3,250 measurements … room 2.17"). Show transcript + Slovak. | reading script B1–B4 |
| 7 | What needs more | "Slovak input, cloned voices and room mode need a GPU and additional integration work." | STATUS.md limits |

## Exact configuration

- `make run PRESET=mac` → `http://localhost:8000`
- Direction: EN → SK (default pair from the page)
- STT: Whisper turbo (MLX), `agree_short` cutting strategy, `mic=raw` (echo cancellation/AGC off)
- MT: Opus-MT `en→sk` (CTranslate2)
- TTS: Piper generic Slovak voice (`sk_SK-lili` default; personal clone not shipped)
- Latency target: median ≤ 2.0 s, p90 ≤ 4.5 s (per `NEAR_REALTIME_GOAL.md`)

## Recording procedure

1. Open `http://localhost:8000` in Chrome/Edge. Select EN → SK. Press Start.
2. Record screen + system audio (QuickTime / OBS). Speak clearly at steady pace.
3. Capture the lag number as it settles (median/p90) — that's the measured evidence.
4. End with the limitations beat.

## What NOT to present as live

- Batch TTS, model-loading success, or synthetic replay are **not** proof of real-time cloned-voice performance.
- If the live mic run can't be reproduced reliably, use the pre-rendered proof clip `processed/e2e_ensk_sk.wav` and label it clearly as recorded input/replay.

## Evidence to attach

- `documentation/evidence/2026-11-15/D_wait05.json` — route run, median lag 0.792 s, p90 1.387 s
- `documentation/evidence/2026-11-15/D_wait05.score.txt` — scored PASS on all 4 criteria
- `documentation/evidence/2026-11-15/A_baseline.json` / `.score.txt` — previous setting comparison
- `documentation/reading_script_multilingual.md` — EN/SK/DE sentences to read
