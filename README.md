<div align="center">

# BP

**Live speech translation. On your machine.**

Speak in one language. Hear and read another — during calls, presentations, and meetings.

[**Watch demo ↗**](https://www.youtube.com/watch?v=MvxVdjCIrx8) · [**Explore live site ↗**](https://brusnyak.github.io/bp/) · [**Documentation ↗**](documentation/INDEX.md)

![CI](https://img.shields.io/github/actions/workflow/status/brusnyak/bp/ci.yml?style=flat-square&label=build)
![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?style=flat-square&logo=python&logoColor=white)
[![MIT](https://img.shields.io/badge/code-MIT-222?style=flat-square)](LICENSE)

[![BP demo: English to Slovak translation and captions](https://img.youtube.com/vi/MvxVdjCIrx8/hqdefault.jpg)](https://www.youtube.com/watch?v=MvxVdjCIrx8)

*Demo: English → Slovak, with live captions.*

</div>

## What it does

| | |
| :-- | :-- |
| **Private by design** | Speech recognition, translation, and voice generation run locally after setup. |
| **Captions + voice** | Follow translated subtitles or route generated audio to your speakers. |
| **Language routes** | English ↔ Czech; English → Slovak / German and additional experimental routes. |
| **Audience mode** | Share a link or QR code so listeners can choose a language on their phones. **Experimental.** |

```text
Microphone → Speech recognition → Translation → Voice / live captions
```

## Run locally

**Requires:** Git · Python 3.11/3.12 · ffmpeg · internet for initial model downloads

```bash
git clone https://github.com/brusnyak/bp.git
cd bp
python3.11 scripts/setup.py --profile mac --langs cs,sk,de
make run PRESET=mac
```

Open **[localhost:8000](http://localhost:8000)** in Chrome or Edge. Choose languages and audio devices, test the microphone, then select **Start**. Use headphones to avoid feedback.

| Your machine | Setup profile / run preset |
| :-- | :-- |
| Apple Silicon | `mac` |
| NVIDIA GPU | `nvidia` |
| Other / CPU | `cpu` |

Replace both `mac` values with your profile. Windows: run setup with Python 3.11, then `.venv\Scripts\python.exe app.py` (the Makefile expects a POSIX shell).

## Performance snapshot

**0.79 s** median lag · **1.39 s** p90 lag · **5.2%** recognition WER

<sub>English → Slovak · Mac M1 Pro · generated audio · one recorded run · “wait 0.5 s” setting. These measurements are not guarantees of continuous real-world performance. [Evidence](documentation/evidence/2026-11-15/) · [Methods](documentation/BENCHMARKS.md)</sub>

## Explore

[**Language support**](documentation/LANGUAGES.md) · [**Calls & room setup**](documentation/INTEGRATION.md) · [**API / SDK**](documentation/sdk.md) · [**Benchmarks**](documentation/BENCHMARKS.md) · [**Full docs**](documentation/INDEX.md)

<details>
<summary><b>For developers: checks and site build</b></summary>

```bash
make test                 # regression tests (requires development dependencies)
make check-docs           # validate benchmark receipts
python3 scripts/check_public_tree.py
python3 scripts/check_site_links.py
.venv/bin/python scripts/build_site.py
.venv/bin/python scripts/bp.py selftest
```

</details>

<details>
<summary><b>Important limitations, privacy & licences</b></summary>

- The standard Mac setup installs Whisper; the separately measured Nemotron streaming configuration is **not** included automatically.
- Slovak speech output, Slovak/Czech input, and audience mode are experimental. Route quality and speed depend on your hardware and installed models. [Route evidence](documentation/LANGUAGES.md).
- Speech processing uses local models by default; installing them downloads assets. Shared-host deployments need appropriate access, consent, and retention controls. Uploaded recordings remain on the processing host.
- **Code is MIT; models and voices are not automatically MIT.** Piper runtime, XTTS v2, X-Voice, and voice packages have separate terms, including non-commercial restrictions. Review [NOTICE](NOTICE.md) and the [licence inventory](documentation/LICENSE_INVENTORY_2026-11-06.md) before commercial or shared deployment.

</details>

---

<sub>Built by [Yegor Brusnyak](https://github.com/brusnyak) · [MIT code licence](LICENSE)</sub>
