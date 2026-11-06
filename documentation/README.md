# Live Speech Translation — Project Overview

> Current repair state (2026-11-04): start with `PRODUCT.md`, `documentation/implementation-2026-11-04/RESULTS.md` and `documentation/HANDOFF_NEXT_SESSION.md`. Earlier dated counts, readiness claims, media-access behavior and Kaggle failure explanations below are historical. Current contracts require authenticated explicit preparation, offline assets, target-language TTS, terminal Stop and reviewed full-route evidence. Full cloned-speech/LHC/commercial acceptance remains pending.

Real-time, local-first speech translation for online conferences. Captures microphone
audio, transcribes (STT), translates (MT), synthesizes the translated speech (TTS), and
plays it back — with subtitles, latency charts, and optional voice cloning. Built as a
bachelor thesis (STU FEI, EN↔SK focus) and designed to be swap-any-engine and
configurable so it can run on a laptop or a T4-class GPU server.

## What the project is (and isn't)

- ✅ **Local-first pipeline:** WebRTC VAD → faster-whisper STT → CTranslate2 Opus-MT MT →
  Piper TTS → browser playback. No cloud in the pipeline.
- ✅ **Cross-platform:** one-command setup (`python3.11 scripts/setup.py --dev`) on macOS,
  Windows (CPU-only) and Linux. No GPU required for the baseline.
- ✅ **Swap-any-engine:** every pipeline stage is behind the engine registry
  (`backend/registry.py`), so STT/MT/TTS/VAD can be swapped by config without code changes.
- ✅ **Language packs:** declarative packs in `languages/packs/*.json` (en-sk, sk-en,
  en-cs, en-zh, en-hi, …). Add a pair with `bp language add <pair>`.
- ✅ **Observability:** `/health` (public) and `/metrics` (auth) HTTP routes; `bp selftest`
  and `make test` (38 tests).
- ⚠️ **SK→EN is experimental:** the best open Slovak Whisper fine-tune scores WER ~0.21 on
  clean audio (above the 0.2 quality gate) and worse on a real mic. Do not ship SK→EN as
  production quality yet. See `documentation/research_findings_2026-11-03.md`.
- ❌ **Not "your own voice" in Slovak yet:** no open zero-shot voice-cloning TTS supports
  Slovak output (XTTS, Chatterbox, Qwen3-TTS, Kokoro all exclude SK). The only SK voice
  path is a Piper fine-tune. See §Voice cloning below.
- ❌ **Not a managed SaaS:** Teams (~9 languages, no SK) and Google Meet (~6 EN-anchored
  pairs, no SK) do not cover Slovak. KUDO AI (77 langs, incl. SK) and DeepL Voice (40+,
  incl. EU) do — but they are paid cloud services (~$0.07-1.25/min). This project is the
  local, free, Slovak-capable alternative.

## The stack (what actually runs)

| Stage | Default engine | License | Optional / GPU engines |
|-------|---------------|---------|------------------------|
| VAD | WebRTC VAD | BSD-3 | Silero VAD (torch) |
| STT (EN) | faster-whisper `base` int8 | MIT | Parakeet EN, Qwen3-ASR |
| STT (SK) | faster-whisper `small-sk` int8 | MIT | Canary AST (STT+MT collapse, untested on SK) |
| MT | Opus-MT CT2 int8 per pair | CC-BY-4.0 | MADLAD-3B (CUDA), NLLB (CC-BY-NC, blocked) |
| TTS (SK) | Piper (owner fine-tune / generic) | MIT | SeamlessM4T v2 S2S (prosody, CC-BY-NC) |
| TTS (EN) | Piper | MIT | Kokoro 82M (RTF 0.03, 1.33 GB VRAM on T4) |

All default engines are commercial-safe (MIT/BSD/Apache-2.0/CC-BY-4.0). Non-commercial
engines (XTTS CPML, NLLB CC-BY-NC, SeamlessM4T CC-BY-NC) are **fail-closed** behind the
licence gate — they cannot be selected in the default profile without explicit override.

## Measured baseline (CPU, M1 Pro / Ryzen 8645HS)

| Direction | STT | MT | TTS | End-to-end turn |
|-----------|-----|-----|-----|-----------------|
| EN→SK | whisper-base ~0.55 s | Opus-MT ~0.08 s | Piper ~0.17 s | ~0.78–1.3 s |
| SK→EN | whisper-small-sk ~1.3 s | Opus-MT ~0.08 s | Piper ~0.07 s | ~1.27–1.64 s (STT-bound) |

SK→EN is the bottleneck and the weak direction — honest about it in the thesis.

## Quick start (zero-knowledge user)

```bash
git clone <repo>
cd BP
python3.11 scripts/setup.py --dev   # ~90 s warm Mac; creates .venv + .env + certs + models
make run                            # https://localhost:8000 (self-signed cert)
# Open the UI, click Initialize Pipeline, select languages, speak
```

- `make test` → 38/38 pass
- `.venv/bin/python scripts/bp.py selftest` → full verification
- `make demo-check` → demo pre-flight (assets + live server)
- `make lab` → Voice Lab static page (no backend)

## Voice cloning — the hard part

"Live translation in your own voice" means speaking the translated output in the speaker's
voice. For Slovak output, **no open zero-shot voice-cloning TTS supports Slovak** — XTTS
(17 langs, no SK), Chatterbox (23 langs, no SK), Qwen3-TTS (10 langs, no SK), Kokoro (EN
only) all exclude it. The only SK path is a Piper fine-tune trained on the speaker's voice
(~4-8 h on Mac CPU, RTF 0.04 at inference). Personal voice models are local-only and never
committed to git.

For EN output, zero-shot cloning works with Chatterbox / Qwen3-TTS / XTTS on a GPU.

## Enterprise comparison (paid cloud)

| Service | Slovak speech translation? | Approx. cost |
|---------|---------------------------|--------------|
| Microsoft Teams Interpreter | ❌ (~9 langs) | ~$10/user/mo (Premium) |
| Google Meet Speech Translation | ❌ (~6 EN-anchored pairs) | Workspace / Google AI plan |
| KUDO AI Speech Translator | ✅ (77 langs, incl. SK) | ~$0.80-1.25/min |
| DeepL Voice-to-Voice | ✅ likely (40+ EU, incl. SK) | ~$0.07-0.20/min |
| **BP local** | ✅ EN↔SK (SK→EN experimental) | Free (hardware + electricity only) |

Local wins on cost and privacy (GDPR-friendly, no audio leaves the machine); loses on
SK→EN quality until a challenger crosses the 0.2 WER gate.

## Project structure

| Path | Purpose |
|------|---------|
| `app.py` | FastAPI app, WebSocket server, UI mounting |
| `backend/` | STT/MT/TTS/VAD engines, model pooling, registry, pipeline facade |
| `backend/registry.py` | Engine registry + capability manifests + fail-closed licence gate |
| `backend/pipeline.py` | Public `Pipeline` facade (translate(audio, src, tgt)) |
| `languages/packs/*.json` | Declarative language packs |
| `ui/` | Browser interface (plain HTML/CSS/JS, PWA, PiP) |
| `scripts/` | `setup.py`, `bp.py` CLI, bench, conversion, evaluation tooling |
| `scripts/bp.py` | Single CLI entry: `bp language|engine|stt|e2e|selftest` |
| `test/` | pytest suites (38 tests) |
| `documentation/` | Thesis draft, findings, research, runbooks |
| `specs/003-end-to-end-completion/` | Feature spec: any-language, swap-any-engine, plug-and-play |

## Key documents

- `documentation/research_findings_2026-11-03.md` — enterprise S2S, voice cloning, Qwen/
  Canary/Parakeet landscape, Slovak relevance, T4 VRAM fit
- `documentation/model_landscape_2026-10.md` — per-component model verdicts (STT/MT/TTS/S2S)
- `documentation/project_status_2026-10-20.md` — verified status, swap ladder, CLI reference
- `documentation/demo_runbook_2026-09-28.md` — stage demo pre-flight, script, fallback ladder
- `PLAN.md` — live development plan and next steps
- `AGENTS.md` — project memory for agents (read before changing anything)

## Licence

MIT (see `LICENSE`). Third-party non-commercial traps (XTTS CPML, NLLB CC-BY-NC,
SeamlessM4T CC-BY-NC) are documented and enforced fail-closed by the licence gate.
