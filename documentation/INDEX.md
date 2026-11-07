# Documentation index (current as of 2026-11-07)

Start here: [`../README.md`](../README.md) (what it is, install, configuration), then:

| Read | When |
|---|---|
| [`STATUS.md`](STATUS.md) | Where the project stands: what works, what is measured but not built, what is not built, decisions |
| [`LANGUAGES.md`](LANGUAGES.md) | Best configuration per language and hardware, with measured quality and timing |
| [`BENCHMARKS.md`](BENCHMARKS.md) | Method, results, negative results and limits; how to rerun on Kaggle |
| [`DEMO.md`](DEMO.md) | A presentation test from a clean clone |
| [`KAGGLE_SETUP.md`](KAGGLE_SETUP.md) | Running benchmark kernels on a Kaggle T4 |
| [`linux_setup_and_test.md`](linux_setup_and_test.md) | Linux install notes and test ladder |
| [`MULTI_USER_SERVER.md`](MULTI_USER_SERVER.md), [`MULTI_SPEAKER.md`](MULTI_SPEAKER.md) | Design notes for shared use (room mode is **not built**, see STATUS) |
| [`sdk.md`](sdk.md), [`cli_quickref.md`](cli_quickref.md) | Interfaces and commands |
| [`model_landscape_2026-10.md`](model_landscape_2026-10.md), [`s2s_landscape_2026-10-18.md`](s2s_landscape_2026-10-18.md) | Engine research with `MEASURED`/`VENDOR` labels (October; superseded by BENCHMARKS where they overlap) |
| [`LICENSE_INVENTORY_2026-11-06.md`](LICENSE_INVENTORY_2026-11-06.md) | Licences of engines, models and voices |
| [`VERIFICATION_RESULTS_2026-11-06.md`](VERIFICATION_RESULTS_2026-11-06.md), [`load_tests_2026-10.md`](load_tests_2026-10.md) | Earlier verification and load measurements (read with the dates in mind) |
| [`fundamentals.md`](fundamentals.md), [`fundamentals_deep.md`](fundamentals_deep.md), [`thesis_draft.md`](thesis_draft.md), [`piper_training_runbook.md`](piper_training_runbook.md), [`supabase_auth_db_guide.md`](supabase_auth_db_guide.md) | Background, thesis text, voice training, auth |

Evidence for every quoted number: [`evidence/2026-11-07/`](evidence/2026-11-07/). `make check-docs` fails when a marked number differs from its receipt.

Archived (kept for history, not maintained): [`archive/`](archive/) and [`archive/2026-11-07-stale/`](archive/2026-11-07-stale/): the September architecture and streaming specs with contradictory latency budgets, old runbooks and handoffs, the old plan and file index.
