# Contributor guide — Real-Time Speech Translation

Read `documentation/STATUS.md`, `documentation/LANGUAGES.md`,
`documentation/BENCHMARKS.md`, `PRODUCT.md`, `DESIGN.md` and
`.specify/memory/constitution.md` before changing behavior.
Dated archives describe historical configurations and are not current acceptance evidence.

## Architecture

FastAPI and WebSocket backend; plain HTML/CSS/JavaScript browser UI.
The local pipeline connects WebRTC VAD, speech recognition, Opus-MT through
CTranslate2, target-language TTS and browser playback with captions.
CPU, Apple Silicon and NVIDIA profiles have different optional engines.
Czech↔English is the primary route; Slovak input and personal voice cloning
remain experimental until the exact route passes its evidence checks.

## Setup and commands

- `python3.11 scripts/setup.py --dev`: cross-platform environment and local assets.
- `make run`: backend on `http://localhost:8000`; no public bind by default.
- `make test`: backend and model-free regression suites; some checks need installed assets.
- `make check-docs`: verify marked documentation numbers against saved receipts.
- `make check-public`: scan tracked text for private project framing.
- `python3 scripts/ui_manifest.py`: regenerate interface and route manifests.
- `python3 scripts/update_voice_lab_library.py --no-test`: regenerate the local Voice Lab library.
- `make demo-check`: preflight requiring a running local backend.
- `make lab`: private Voice Lab static server; never expose it publicly.

## Change boundaries

- Keep one cross-platform requirements/setup path. Do not restore platform-specific dependency forks.
- Never commit `.env`, credentials, personal voice models or source recordings.
  Back up irreplaceable custom assets outside the checkout before cleanup.
- A fresh clone uses generic voices until custom local assets are explicitly installed.
- Keep API test fixtures isolated; snapshot and restore any real voice data they encounter.
- Prepare assets explicitly before capture. Maintain authenticated access, target-language
  voice selection, terminal Stop and cancellation of queued playback.
- Keep effective engine names and session configuration synchronized.
- Do not add cloud recognition, translation or speech generation to the default local pipeline.
- Do not promote a route from model loading, a source-language score or isolated speed.
  Full-route evidence includes recognition, reviewed translation meaning, intelligibility,
  voice identity where applicable, licence/runtime policy and live delay/queue stability.
- Documentation numbers require saved measurements; preserve claim markers and receipt identifiers.
- Regenerate manifests after interface-copy changes and check all affected locales.

## Review and landing

Use a feature branch and a reviewed pull request. Run the relevant tests and
`make check-docs`; live rehearsals require installed assets and explicit test scope.
Do not push, deploy or publish without authorization. Do not rewrite the integration
branch or change ignored local assets as part of text-only work.
