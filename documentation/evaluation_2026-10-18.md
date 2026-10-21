# Engine, language and swap evaluation — 2026-10-18

Hardware: MacBook Pro M1 Pro 16 GB, CPU (CTranslate2 int8) + CoreML for Piper,
torch-free runtime (`.venv`). Everything below is **MEASURED on this machine today**
unless marked otherwise. Raw records:

- `processed/language_eval/*.json` — per-pair run records
- `processed/stt_matrix.json` — STT swap matrix
- `processed/engine_ab/matrix_tts_swap.json`, `matrix_tts_swap_all.json` — TTS swap
- `processed/gpu_bench/*` — earlier T4 GPU numbers (2026-10-02)

## 1. Multilanguage — 5 packs

| pair | status | STT s | TTS s | e2e s | engines |
|---|---|---|---|---|---|
| en-sk | **verified** | 4.11 | 3.11 | 7.21 | `base` / opus-mt-en-sk / `sk_SK-personal-male-medium` |
| sk-en | **verified** | **22.81** | 2.46 | **25.27** | `small-sk` / opus-mt-sk-en / `en_US-ryan-medium` |
| en-cs | **verified** | 4.62 | 3.66 | 8.27 | `base` / opus-mt-en-cs / `cs_CZ-jirka-medium` |
| cs-en | experimental | — | — | — | MT `opus-mt-cs-en` not converted |
| en-de | experimental | — | — | — | MT not converted + no `de_DE` Piper voice |

Three pairs run end-to-end. Czech works with zero code change (pack + one converted
MT model). The two experimental pairs fail with the *exact missing component named*,
which is the behaviour the pack contract promises.

**The one real defect the table exposes: SK→EN is 3.5× slower than EN→SK and the
transcript is visibly wrong** ("viete si v tejto šivejokáške a pre kultúru reči" for
"vitajte v tejto živej ukážke prekladu rečí"), so the English output is wrong too.
SK STT is the whole story — see §2.

## 2. STT swap — where the quality and the latency actually go

Same clips, whisper sizes swapped, language prompt varied (`processed/stt_matrix.json`):

| model | clip | prompt | WER | s |
|---|---|---|---|---|
| base | EN | en | **0.077** | **4.2** |
| small | EN | en | 0.096 | 13.9 |
| base | SK | sk | 0.714 | 8.2 |
| small | SK | sk | 0.552 | 22.2 |
| base | SK | auto | **1.000** | 7.3 |
| small | SK | auto | **1.000** | 7.3 |

Three findings that should drive decisions:

1. **EN: `base` beats `small` on both axes** (4.2 s vs 13.9 s, WER 0.077 vs 0.096).
   Going up a whisper size costs 3.3× latency and buys nothing on this clip.
2. **SK: never let the model auto-detect the language.** `auto` gives WER 1.000 —
   total failure. The pack forces `language: sk`; that is load-bearing, not a tweak.
3. **SK WER 0.55–0.71 across every rung.** The fine-tuned `small-sk` is not
   meaningfully better than plain `small` here, and it is not faster. Beating this
   needs a different SK STT (large-v3-turbo / Parakeet / a properly fine-tuned model),
   not a bigger Whisper size — that is a GPU-bench item.

## 3. TTS swap — speed is solved, voice identity costs intelligibility

Same SK sentence through five voices, synthesised, then fed back through STT to
score intelligibility (`processed/engine_ab/matrix_tts_swap_all.json`):

| voice | RTF | WER (rung `base`) | WER (rung `small-sk`) |
|---|---|---|---|
| omni_zeroshot | (MPS, cached) | 0.513 | **0.054** |
| omni_generic | (MPS, cached) | 0.568 | 0.162 |
| piper_generic (`sk_SK-lili-medium`) | 0.034 | 0.649 | 0.270 |
| piper_omni_hq (`me_omni_piper_sk`) | 0.031 | 0.676 | 0.378 |
| **piper_personal (`sk_SK-personal-male-medium`)** | 0.030 | 0.865 | **0.595** |

- Every Piper voice runs at **RTF ≈ 0.03 on CoreML** — TTS is never the bottleneck.
- **The owner's personal fine-tuned voice is the least intelligible of all five**
  (WER 0.595 vs 0.270 for the generic Piper voice). Voice identity and STT
  intelligibility trade off directly.
- OmniVoice zero-shot is the most intelligible (0.054).

Caveat: these omni rows are cached wavs from 2026-09-30 (`--reuse`), not re-synthesised
today — the `.venv-omni` env was removed in the project lightening. Piper rows are fresh.

## 4. MT swap — not yet done

Only `opus-mt` exists locally (en-sk, sk-en, en-cs). A second MT engine (NLLB-200)
needs its own converter — `scripts/convert_models.py` is opus-mt-specific — plus a
throwaway torch venv, and NLLB is licence-gated `non-commercial` in the registry.
Recorded as a task, not silently skipped: MT quality cannot be compared until it exists.

## 5. Bug ledger — found and fixed this session

| Bug | Impact | Fix |
|---|---|---|
| `bp stt` referenced renamed clip files | command dead | clips now derived from the language packs (single source of truth) |
| `scripts/free_models.py` passed `--dir` (invalid flag) | **every `opencode/*` and `openrouter/*` free model scored `fail`; the free-model ranking was silently nim-only** | run in `cwd`, and pass `PWD=` (see below) |
| `opencode run` resolves its directory from `$PWD`, not `getcwd()` | probe wrote `probe.txt` into the repo root and still reported `fail` | `env={**os.environ, "PWD": d}`; stale history/skip cleared |
| `language_packs.load/add/remove` used `sys.exit` | module not importable/testable | raise; CLI wrapper converts to exit |
| `cmd_add` hardcoded `license: commercial-ok` | would have shipped a false licence claim for NLLB/CC-BY-NC | licence inferred from the HF model card (verified: en-sk→apache-2.0→`commercial-ok`, NLLB→`non-commercial`); `--license` override; unverifiable (HTTP 401) is warned |
| `cmd_add` assumed the opus-mt model exists | silent broken pack | HF existence check + warning |
| `_detect_tts` picked the alphabetically-first voice | a pack for `sk` would get the generic voice, not the owner's | prefers `personal`/`omni` voices |
| `HF_WHISPER` hardcoded list | stale (missed `large-v3-turbo`) | runtime discovery via `faster_whisper.utils.available_models()` |

`make test` = **47/47** after all of the above.

## 6. What remains (roast of `specs/003-end-to-end-completion/spec.md`)

The spec cannot be implemented from its current text — the hard integration points
are undefined. Severity-ordered:

- **P0 — Path B has no technical design.** US5 (loopback + virtual mic + floating
  captions) is a whole product feature with zero design: no per-OS audio-routing
  commands, no caption-window tech chosen.
- **P0 — SDK (FR-012) is undefined.** "thin REST/WS contract + Python API" is a
  wish, not a spec: no message schema, no auth on the WS, no class surface.
- P1 — SC-002 claims macOS+Windows+Linux from a clean clone, but **x86_64 Linux was
  never run** (run 1 was aarch64 colima).
- P1 — model pooling (FR-009) has no config schema, memory budget or eviction policy.
- P1 — auth (FR-008) says "real auth" but names no scheme.
- P1 — the licence gate's "default profile" is never defined.
- P2 — no language-pack schema; the stage gate has no pass thresholds; SC-004
  compares T4 GPU numbers to a CPU baseline; the stale "needs the III amendment" note.
- Also open from `tasks.md`: **T010** (wire `select()`/`instantiate()` into
  `main.py` — until then the licence gate is dead code), **T012** (setup rewrite),
  T009 remainder (per-stage gate records in `processed/engine_gate/`).

## 7. Zero-shot vs trained voice (research)

Question: every TTS uses zero-shot cloning because training a voice on the Mac is
hard and each engine has its own training format. Is there a way out?

**Yes — three, and the real fix is to stop training per engine and decouple voice
identity from the TTS.**

| route | repos (★, licence, last push) | notes |
|---|---|---|
| **Voice-conversion post-layer** — train ONE voice model, run ANY TTS output through it | `RVC-Project/Retrieval-based-Voice-Conversion-WebUI` ★38.8k MIT (2026-08-04); `IAHispano/Applio` ★3.8k MIT (2026-10-17) | engine-agnostic: Piper→RVC and XTTS→RVC both give your voice; real-time proven (CUDA-graph inference, live-mic + virtual-audio routing — the same routing Path B needs); zero-shot VC also exists (MeanVC 2, 2026-06: 40 ms chunks, RTF 0.63 on one CPU core) |
| **Standardise on one engine that does both** | `RVC-Boss/GPT-SoVITS` ★63.1k MIT (pushed today) fine-tunes from ~1 min and also does zero-shot; `SWivid/F5-TTS` ★15.4k MIT (2026-10-14) | one repeatable recipe instead of five |
| **Keep the per-engine fine-tune you already have** | Piper SK fine-tune works in this repo (75-sentence set, `finetune_personal_voice.py`) | but §3 shows the personal Piper voice is the *least* intelligible — identity ≠ quality |

Avoid: `svc-develop-team/so-vits-svc` (AGPL-3.0, archived 2023).

**Recommendation:** add a **voice-conversion layer** (Applio, MIT, active). Train one
voice model once on a rented GPU (~1 h), then every TTS engine — including swappable
ones — speaks in the owner's voice with no per-engine training, and it doubles as the
Path B voice-changer. The per-engine Piper fine-tune stays as a fallback for CPU-only.

## 8. Next: GPU

Kaggle credentials are **gone** (`~/.kaggle/kaggle.json` absent, `kaggle` not on PATH).
The **Modal token still exists** (`~/.modal.toml`). Kaggle/Modal cleanup was queued
for 2026-10-09 and has not happened — confirm before spending.
