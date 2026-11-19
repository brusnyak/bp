<p align="center">
  <a href="https://brusnyak.github.io/bp/" target="_blank">
    <img src="https://img.shields.io/badge/product%20site-brusnyak.github.io%2Fbp-blue?style=for-the-badge&logo=githubpages" alt="Product site" width="auto"/>
  </a>
  <a href="https://github.com/brusnyak/bp/actions" target="_blank">
    <img src="https://img.shields.io/github/actions/workflow/status/brusnyak/bp/.github/workflows/ci.yml?label=CI&style=for-the-badge" alt="CI status" width="auto"/>
  </a>
  <a href="LICENSE" target="_blank">
    <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="MIT license" width="auto"/>
  </a>
</p>

# BP — Live Speech Translation

Local speech translation and live captions for calls and presentations. Microphone → VAD → speech recognition → translation → voice → speakers/captions, all running on your own machine after model installation. No cloud speech service.

<p align="center">
  <a href="https://www.youtube.com/watch?v=MvxVdjCIrx8" target="_blank" rel="noopener noreferrer">
    <img src="https://img.youtube.com/vi/MvxVdjCIrx8/maxresdefault.jpg" alt="Live Speech Translation demo — English → Slovak, live captions, everything local" width="100%" max-width="860" style="border-radius:12px;border:1px solid #ddd;">
  </a>
  <br/><b>Watch the demo on YouTube — English → Slovak, live captions, everything local.</b>
</p>

<p align="center">
  <b>Deployed product site (SK/EN):</b> <a href="https://brusnyak.github.io/bp/">brusnyak.github.io/bp</a> · measured speed, voice choices, language routes, room mode.
</p>

## Features

- **Local-first** — recognition, translation and synthesis run on your machine; no cloud speech service
- **Live captions** — translated subtitles appear as you speak
- **Language routes** — English to Slovak, German and Czech (Slovak/Czech input experimental)
- **Room broadcast** — one speaker, many listeners; each listener picks their language on their phone via link or QR code (no app required, experimental)

## Quick start

Requires Git, Python 3.11 or 3.12, and ffmpeg. Setup downloads dependencies and models; allow disk space and internet access for installation.

```sh
git clone https://github.com/brusnyak/bp.git
cd bp
python3.11 scripts/setup.py --profile mac --langs cs,sk,de
make run PRESET=mac
```

| Profile | For |
|---|---|
| `mac` | Apple Silicon (Whisper default) |
| `nvidia` | NVIDIA CUDA |
| `cpu` | Other machines |

Use `mac` / `nvidia` / `cpu` to match the setup profile and `PRESET`. Add `--dev` for test dependencies. On Windows use a Python 3.11 executable for setup, then `.venv\Scripts\python.exe app.py`; the Makefile's preset loader requires a POSIX shell.

Open **http://localhost:8000** in Chrome or Edge. Select the languages, choose **Microphone** and **Output**, use **Test microphone**, wear headphones and press **Start**. Use **Stop** to end capture and queued playback.

The standard Mac setup prepares Whisper. The faster Nemotron streaming result measured on the product site uses separately provisioned assets; cloning the repository does not install that configuration automatically. Generic voices are included by setup; personal models and recordings are not shipped. Slovak fine-tune and Nemotron remain manual installs until the model router installs them.

## Choose a route

| Direction | Status |
|---|---|
| English → Slovak | Live laptop route measured; Slovak output remains experimental |
| English → Czech | Available with generic Piper voice; primary pair |
| English → German | Live route tested on a T4; check your own hardware |
| German → English | Available; component measurements do not prove sustained live performance |
| Czech → English | Route measured on a T4; latency depends on profile |
| Slovak → English | Experimental; low latency needs a faster GPU |

See [language evidence](documentation/LANGUAGES.md) and the [product site's route table](https://brusnyak.github.io/bp/#trasy). Installation and model loading alone do not establish translation meaning or speech quality.

## Connect a call or room

Select a virtual audio device under **Output**, then select its recording end as the meeting application's microphone. For listener phones, open **More → Audience broadcast (room mode) → Start room** and share the link or QR code. Room mode remains experimental — verify it with real devices before the event.

The [integration guide](documentation/INTEGRATION.md) covers BlackHole, VB-Cable, PipeWire/PulseAudio, subtitles, HTTPS, the HTTP/WebSocket contract and an AV/IT deployment checklist.

## Commands and developer reference

| Command | Purpose |
|---|---|
| `make test` | Backend regression suites; requires setup with test dependencies |
| `make check-docs` | Verify marked measurements against saved receipts |
| `python3 scripts/check_public_tree.py` | Check tracked public text |
| `.venv/bin/python scripts/build_site.py` | Rebuild the product site from its template, CSS and receipts |
| `python3 scripts/check_site_links.py` | Check local site links and anchors without a server |
| `.venv/bin/python scripts/bp.py selftest` | Check assets and configuration |

## Measured results (English → Slovak, Mac M1 Pro, one run, generated audio)

| Setting | Median lag | p90 lag | Drift | Recognition WER |
|---|---|---|---|---|
| Wait 0.5 s (current) | 0.79 s | 1.39 s | -0.05 s/min | 5.2% |
| Previous setting | 0.75 s | 1.80 s | — | 5.2% |

Measured on `D_wait05` vs `A_baseline` in `documentation/evidence/2026-11-15/`, receipt-checked by `make check-docs`. Full numbers on the product site.

[Documentation index](documentation/INDEX.md) · [API and Python SDK](documentation/sdk.md) · [Reading script](documentation/reading_script_multilingual.md) · [Benchmark methods](documentation/BENCHMARKS.md)

## Privacy and licences

The default speech pipeline runs on the processing host without cloud ASR, translation or TTS. Installation downloads models. Shared use needs access, consent and retention policies; uploaded recordings stay on the host.

Code: MIT. Models, voices and runtimes retain their own terms. Piper's runtime is GPL-3.0-or-later; each voice needs its own review. XTTS v2 weights use non-commercial CPML and X-Voice weights use CC-BY-NC. Review [NOTICE.md](NOTICE.md) and the [licence inventory](documentation/LICENSE_INVENTORY_2026-11-06.md) before distributing or operating a shared service. An engine being selectable does not grant commercial rights.

Created by Yegor Brusnyak, MIT licence
