# Documentation

**[Start with the product site](site/index.html).** It covers installation, supported routes, measured speed, voice models, hardware and limits. Open the HTML file in a browser after cloning; GitHub's file viewer shows the source.

## Use the tool

| Guide | What you can do |
|---|---|
| [Quick start](../README.md#quick-start) | Clone, install a hardware profile and run locally |
| [Integration](INTEGRATION.md) | Route audio into a call or AV system; share captions and room audio |
| [Languages](LANGUAGES.md) | Check measured recognition and translation coverage by language |
| [Multilingual reading script](reading_script_multilingual.md) | Record a repeatable English, Czech, Slovak or German accuracy test |
| [API and Python SDK](sdk.md) | Embed the pipeline; compare older examples against the current wire contract in the integration guide |

## Understand the measurements

| Source | Scope |
|---|---|
| [Product-site speed section](site/index.html#speed) | Nemotron + Piper, generated English → Slovak on an M1 Pro; one run per setting |
| [Route replay and scores](evidence/2026-11-15/) | Raw route results and isolated T4 voice-model comparisons |
| [Benchmark methods](BENCHMARKS.md) | Earlier Whisper route tests, recognition and translation datasets, measurement limits |
| [Language measurements](LANGUAGES.md) | Component quality, hardware profiles and receipt-checked values |
| [Earlier receipt set](evidence/2026-11-07/) | Data used by the marked documentation-number checks |

Recognition WER, translation chrF, synthesis RTF and route lag measure different things. A component score does not establish full-route meaning, intelligibility, voice identity or room capacity.

## Develop and distribute

| Reference | Purpose |
|---|---|
| [Makefile](../Makefile) | Run and verification commands |
| [Setup script](../scripts/setup.py) | Hardware profiles and model preparation |
| [Site generator](../scripts/build_site.py) | Rebuild `site/index.html` from `site/index.template.html`, `site/site.css` and saved receipts |
| [Third-party notice](../NOTICE.md) | Runtime and checkpoint licence boundaries |
| [Licence inventory](LICENSE_INVENTORY_2026-11-06.md) | Starting point for model and voice licence review |
| [MIT licence](../LICENSE) | Repository code terms |

Regenerate after editing the site sources:

```sh
.venv/bin/python scripts/build_site.py
python3 scripts/check_site_links.py
make check-docs
python3 scripts/check_public_tree.py
```

Created by Yegor Brusnyak, MIT licence
