# BP — next session handoff (2026-11-05)

## Decisions this session

| Decision | Where recorded |
|----------|---------------|
| SK demoted from mandatory to experimental; CS↔EN is the default pair | `.specify/memory/constitution.md` v1.1.0, Principle III |
| CS→EN STT model corrected `small-sk` → `base` | `languages/packs/cs-en.json` |
| MT `opus-mt-cs-en` converted to CTranslate2 int8 | `ct2_models/Helsinki-NLP--opus-mt-cs-en/` |
| Test-only packs added: `de-en`, `en-fr` (experimental, TTS voices install-on-demand) | `languages/packs/de-en.json`, `languages/packs/en-fr.json` |
| Kaggle kernel rewritten: CPU-only, CS→EN default, configurable `--source`/`--target` | `scripts/gpu_bench/kaggle_full_route/job.py` |
| Kaggle setup documentation | `documentation/KAGGLE_SETUP.md` |
| All tests green | `make test` → 87 passed |

## What works now

- **CS→EN**: full pipeline resolves and runs end-to-end. Local CPU measurement: STT 6.20s + MT ~0.1s + TTS 2.43s ≈ 8.7s wall on a ~10s Czech clip (M1 Pro, CPU). Not near real-time on CPU — Kaggle T4 path needed.
- **EN→CS**: already worked (~2.1s wall measured previously).
- **Other language pairs**: all 15 packs resolve structurally (STT base + MT converted + TTS auto-detect). FR/DE/HI/ZH targets lack Piper voices → experimental until a voice is installed and a `processed/language_eval/<pair>.json` record passes review.

## What's broken / open

| Item | Status |
|------|--------|
| SK→EN quality | `whisper-small-sk` WER 0.25–0.32 > 0.2 gate, Czech-like hallucinations. Needs `kinit/whisper-large-v3-turbo-sk` on GPU. Now experimental, not default. |
| Near real-time (≤2s wall) | Not achieved on CPU. Target: Kaggle T4 with GPU STT/MT. |
| TTS voices for FR/DE/HI/ZH | Missing. `bp language add <pair>` scaffolds; user installs voice + converts MT on demand. |
| Multi-user meeting UI | Designed (`session_manager.py`, `/ws/meeting`, `MULTI_USER_UI_HANDOFF.md`), UI not built. |
| Kaggle push | Kernel ready, not yet pushed. 8 prior pushes failed (CUDA/libcublas, audio decoding, VAD filtering) — all addressed in the rewrite. |

## Kaggle — ready to push

Kernel: `scripts/gpu_bench/kaggle_full_route/` (`job.py`, `kernel-metadata.json`)
Default invocation:
```bash
kaggle kernels push -p scripts/gpu_bench/kaggle_full_route \
  --accelerator NvidiaTeslaT4 -t 1500
```
Then run `--source cs --target en --audio <clip>`. Supports `--source fr --target en`, `--source de --target en`, etc. CPU-only (no `libcublas.so.12`), ffmpeg audio decoding, tunable VAD RMS (`BP_VAD_RMS`).

## Session file changes (this session)

Modified:
- `.specify/memory/constitution.md` — SK demoted, CS↔EN default, v1.1.0
- `languages/packs/cs-en.json` — STT base, notes updated
- `test/language_evidence_test.py` — EXPECTED_PAIRS 13 → 15
- `scripts/gpu_bench/kaggle_full_route/job.py` — rewritten kernel
- `scripts/gpu_bench/kaggle_full_route/kernel-metadata.json` — metadata updated

New:
- `languages/packs/de-en.json` — German→English experimental pack
- `languages/packs/en-fr.json` — English→French experimental pack
- `documentation/KAGGLE_SETUP.md` — Kaggle T4 setup guide
- `ct2_models/Helsinki-NLP--opus-mt-cs-en/` — converted MT model (gitignored, local only)

Note: ~85 pre-existing unstaged changes exist in the working tree from before this session. Review `git diff` before committing. Do not push until review.

## Next steps (in order)

1. Push Kaggle kernel (`kaggle kernels push ... --accelerator NvidiaTeslaT4 -t 1500`)
2. Run CS→EN on T4, measure p50/p95 first-translation
3. If T4 meets ~2s target, promote CS→EN from experimental; keep SK→EN experimental until quality-tier STT verified
4. Add FR/DE/HI/ZH TTS voices on demand, record evals, mark verified
5. Consider SK→EN quality-tier conversion (`kinit/whisper-large-v3-turbo-sk` → CT2) only if SK needed

## Roast

**Weakest point:** CS→EN works but is ~8.7s wall on CPU — far from near real-time. Kaggle T4 is the only path to the ≤2s target, and it hasn't been pushed yet.  
**Cheapest failure:** pushing the Kaggle kernel and hitting the same environment issues as the 8 prior failures (addressed in the rewrite, but unverified until push).  
**Smallest fix:** push the kernel, run CS→EN, record p50/p95 — that single number decides whether T4 is viable or if we need GPU STT specifically.

🟢 87/87 tests pass; cs-en pipeline fixed; constitution amended; Kaggle kernel ready to push.
