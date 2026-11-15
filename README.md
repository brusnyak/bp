# Live Speech Translation

Local speech translation and live captions for calls and presentations. Recognition, translation and generic Piper speech run on your own machine. Personal voice cloning and Slovak input remain experimental.

**[Start with the product site](documentation/site/index.html)** — setup, language routes, measured speed, voice choices, hardware and room integration. Open the HTML file in a browser after cloning. GitHub's file viewer shows its source.

Microphone → VAD → speech recognition → translation → voice → speakers / captions. All processing is local after model installation; room listeners receive output over your network.

## Quick start

Requires Git, Python 3.11 or 3.12, and ffmpeg. Setup downloads dependencies and models; allow disk space and internet access for installation.

```sh
git clone https://github.com/brusnyak/bp.git
cd bp
python3.11 scripts/setup.py --profile mac --langs cs,sk,de
make run PRESET=mac
```

Use `mac` for Apple Silicon, `nvidia` for NVIDIA CUDA, or `cpu` for other machines; match the setup profile and `PRESET`. Add `--dev` for test dependencies. On Windows use a Python 3.11 executable for setup, then `.venv\Scripts\python.exe app.py`; the Makefile's preset loader requires a POSIX shell.

Open **http://localhost:8000** in Chrome or Edge. Select the languages, choose **Microphone** and **Output**, use **Test microphone**, wear headphones and press **Start** (it prepares the route and begins capture). Use **Stop** to end capture and queued playback.

The standard Mac setup prepares Whisper. The faster Nemotron streaming result on the product site uses separately provisioned assets; cloning the repository does not install that configuration automatically. Generic voices are included by setup; personal models and recordings are not shipped.

## Choose a route

| Direction | Status |
|---|---|
| English → Czech | Available with Piper; Czech↔English is the primary pair |
| English → Slovak | Live laptop route measured on generated English; experimental language coverage |
| English → German | Live route tested on a T4; check your own hardware |
| German → English | Available; component measurements do not prove sustained live performance |
| Czech → English | Route measured on a T4; latency depends on profile |
| Slovak → English | Experimental; low-lag operation needs faster GPU testing |

See [language evidence](documentation/LANGUAGES.md) and the [product site's route table](documentation/site/index.html#routes). Installation and model loading alone do not establish translation meaning or speech quality.

## Connect a call or room

Select a virtual audio device under **Output**, then select its recording end as the meeting application's microphone. For listener phones, open **More → Audience broadcast (room mode) → Start room** and share the link or QR code. Room mode remains experimental.

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

[Documentation index](documentation/INDEX.md) · [API and Python SDK](documentation/sdk.md) · [Reading script](documentation/reading_script_multilingual.md) · [Benchmark methods](documentation/BENCHMARKS.md)

## Privacy and licences

The default speech pipeline runs on the processing host without cloud ASR, translation or TTS. Installation downloads models. Shared use needs access, consent and retention policies; uploaded recordings stay on the host.

Code: MIT. Models, voices and runtimes retain their own terms. Piper's runtime is GPL-3.0-or-later; each voice needs its own review. XTTS v2 weights use non-commercial CPML and X-Voice weights use CC-BY-NC. Review [NOTICE.md](NOTICE.md) and the [licence inventory](documentation/LICENSE_INVENTORY_2026-11-06.md) before distributing or operating a shared service. An engine being selectable does not grant commercial rights.

## Saved measurements

<details>
<summary>Earlier measured configurations and receipt-checked values</summary>

Saved measurements from the earlier Whisper configurations. These are different engines, data and clocks from the Nemotron result on the product site. Marked numbers are checked by `make check-docs`.

- **Live route, EN→SK, M1 Pro, delay from end of speech to the first translated audio** (`scripts/live/paced_replay.py` at 1×, evidence in `documentation/evidence/2026-11-07/`): Whisper `base` median <!-- claim:paced:fw_base_lag_median -->0.82<!-- /claim --> s; `mlx-whisper` large-v3-turbo median <!-- claim:paced:mlx_lag_median -->1.01<!-- /claim --> s; turbo with the pause mirror (below) median <!-- claim:paced:mlx_prosody_lag_median -->1.19<!-- /claim --> s, p90 <!-- claim:paced:mlx_prosody_lag_p90 -->2.65<!-- /claim --> s. An earlier "0.39 s first audio" figure could not be reproduced and was removed.
- **Speech recognition, Kaggle T4, mean word error on FLEURS read speech (60 clips per language)**, German: base 18.1%, small <!-- claim:stt:whisper-small:de:wer -->10.9<!-- /claim -->%, large-v3-turbo <!-- claim:stt:whisper-large-v3-turbo:de:wer -->4.8<!-- /claim -->%, Parakeet v3 <!-- claim:stt:parakeet-tdt-v3:de:wer -->6.3<!-- /claim -->%. Czech: turbo <!-- claim:stt:whisper-large-v3-turbo:cs:wer -->11.6<!-- /claim -->% (base 67%). Slovak: turbo <!-- claim:stt:whisper-large-v3-turbo:sk:wer -->10.7<!-- /claim -->%, Parakeet <!-- claim:stt:parakeet-tdt-v3:sk:wer -->9.9<!-- /claim -->%. Turbo wins German, Czech, Polish and English; Parakeet wins French and Slovak by under 2 points. Full table: `documentation/evidence/2026-11-07/lang_matrix.json`. Parakeet ran on CPU in that run (no CUDA provider), so its speed there is not a GPU number.
- **Translation, same T4 run, chrF on 200 opus-100 test sentences (relative scores):** German→English Opus-MT <!-- claim:mt:opus-mt:de:to_en:chrf -->57.9<!-- /claim --> versus NLLB-1.3B 54.6; Opus-MT won every language/direction cell measured, 4–8× faster.
- **CS→EN on Kaggle T4, full route (`route_receipt.json`):** default profile p50 <!-- claim:kaggle:default:p50 -->3.48<!-- /claim --> s, p95 <!-- claim:kaggle:default:p95 -->4.13<!-- /claim --> s from speech onset to first audio. The shorter 1.0 s-segment profile reaches p50 1.87 s / p95 4.26 s (choppier output). Earlier figures of 1.86/4.22 s belonged to that profile, not the default.
**Streaming foundation (Kaggle T4 simulation, offline replay of the test speaker's English reading and a 2-minute German FLEURS stream; `documentation/evidence/2026-11-07/stream_sim.json`).** Latency is the time from a word being spoken until it is committed. Independent 2 s chunks (the measured configuration) with Whisper turbo: median <!-- claim:sim:en-owner|turbo|chunk-2.0:lat -->1.17<!-- /claim --> s but <!-- claim:sim:en-owner|turbo|chunk-2.0:wer -->19.1<!-- /claim -->% word error (German <!-- claim:sim:de-fleurs|turbo|chunk-2.0:wer -->33.1<!-- /claim -->%), because words are cut at chunk edges. Decoding the growing buffer every 1.0 s and committing only words two consecutive decodes agree on: median <!-- claim:sim:en-owner|turbo|agree-1.0:lat -->2.31<!-- /claim --> s and <!-- claim:sim:en-owner|turbo|agree-1.0:wer -->5.0<!-- /claim -->% error (German <!-- claim:sim:de-fleurs|turbo|agree-1.0:wer -->11.2<!-- /claim -->%). A 0.5 s step gives <!-- claim:sim:en-owner|turbo|agree-0.5:lat -->1.45<!-- /claim --> s and <!-- claim:sim:en-owner|turbo|agree-0.5:wer -->12.3<!-- /claim -->% but the engine falls up to 2 s behind; 0.3 s collapses (queue of 30 s). Parakeet is cheap per step (real-time factor 0.1–0.3) and reaches 1.2 s at 0.3 s steps, with <!-- claim:sim:en-owner|parakeet|agree-0.3:wer -->26.4<!-- /claim -->% error. No setting reached under 1 s with under 10% error.


</details>

Created by Yegor Brusnyak, MIT licence

### Local sessions and voice references

The browser automatically requests `POST /api/guest`: a distinct random guest ID
and a signed token valid for two hours. `JWT_SECRET` may be supplied by environment;
otherwise the first run generates a private, git-ignored `.jwt_secret` file.
No identity database is needed. Keep the default loopback binding. Setting
`BP_HOST=0.0.0.0` explicitly lets network clients use the tool and its reference files;
rate limits remain enabled on exposed hosts. Use `BP_HTTPS=1` for HTTPS.

In the live page, open More → Voice reference to record, upload, select or delete
local files. WAV, M4A, MP3 and WebM are accepted, up to 15 MB and 60 seconds each,
with at most five files in the git-ignored `speaker_voices/` directory. These files
are shared locally across anonymous sessions, and stay until you delete them.
Synthesis receives the selected reference filename. Optional `BP_LOG_SESSIONS=1`
saves session audio and text on the processing machine; leave it off unless needed.
