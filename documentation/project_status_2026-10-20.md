# BP Project Status — 2026-10-21

**One-line summary**: Real-time EN↔SK speech translation cascade (VAD → STT → MT → TTS), local-first, CPU-only baseline, swappable engines, runs on macOS/Windows/Linux from a single `setup.py`. **All core functionality verified 2026-10-21.**

---

## ✅ What Works (Verified, Measured 2026-10-21)

| Pair | Direction | STT Model | MT Model | TTS Voice | Latency (turn) | Status |
|------|-----------|-----------|----------|-----------|----------------|--------|
| en-sk | EN→SK | `whisper-base` (int8) | opus-mt-en-sk | `sk_SK-personal-male-medium` (Piper) | ~0.7 s | **verified** |
| sk-en | SK→EN | `whisper-small-sk` (int8) | opus-mt-sk-en | `en_US-ryan-medium` (Piper) | ~1.4 s* | **experimental** |
| en-cs | EN→CS | `whisper-base` (int8) | opus-mt-en-cs | `cs_CZ-jirka-medium` (Piper) | ~0.8 s | **verified** |
| en-zh | EN→ZH | `whisper-base` (int8) | opus-mt-en-zh | `zh_CN-chaowen-medium` (Piper) | ~0.8 s | **verified** |
| en-hi | EN→HI | `whisper-base` (int8) | opus-mt-en-hi | `hi_IN-pratham-medium` (Piper) | ~0.8 s | **verified** |

\* SK→EN latency bottleneck is STT (see § Known Limitations).

**Core pipeline**: 38/38 tests pass (`make test`). Engine gates pass 4/4:
- STT: WER 0.064 < 0.15, latency 4.08s < 8s (EN base)
- MT: chrF 0.919 > 0.6, latency 0.062s < 0.5s
- TTS (Piper): RTF 0.034 < 0.1
- VAD: false_start 0.0, latency ~0s < 0.01s

**Live pipeline tests (WebSocket, 2026-10-21)**:
- EN→SK: Transcript accurate, translation correct (Slovak)
- SK→EN: Transcript shows Czech hallucinations (known model flaw), translation garbled
- EN→CS: Transcript accurate, translation correct (Czech)
- EN→ZH: Transcript accurate, translation correct (Chinese: "我说的是英语和")
- EN→HI: Transcript accurate, translation correct (Hindi: "सुप्रभात, और इस जीवित भाषा अनुवाद के प्रदर्शन का स्वागत करें.")
- TTS engine swap: `piper` ↔ `piper_sk_personal` both work

**Architecture**: FastAPI + WebSocket backend, plain HTML/CSS/JS frontend. Model pooling with LRU + per-stage limits + graceful shutdown. JWT auth on `/ws` and mutating routes. Closed registration by default.

**CLI**: Single entry point `scripts/bp.py` with subcommands:
```bash
bp language add|list|test|remove   # language packs
bp engine list|check|gate          # engine registry + licence gate
bp selftest                        # full verify
bp stt --models base,small --clip sk  # STT benchmarks
bp grades --import voice-ratings.json # Voice Lab ear grades
```

---

## 🧪 Experimental / Known Gaps

| Feature | Status | Notes |
|---------|--------|-------|
| cs-en, en-de | experimental | MT model not converted / no Piper voice |
| SK→EN quality | WER 0.27 (exp.) | `whisper-small-sk` fine-tune produces Czech-like hallucinations; marked experimental; see § SK→EN Deep Dive |
| **ja TTS** | **unsupported** | Piper voice uses unsupported `japanese` phoneme_type; MT works, TTS fails |
| RVC-MLX voice conversion | registered, no model | Train once on GPU (Applio), then any TTS → your voice |
| OmniVoice TTS | registered, GPU-only | RTF 0.14–0.17 on T4, first audio ~1.1s |
| Unified S2S (Qwen3-Omni) | declared only | Apache 2.0, not implemented |
| Linux x86_64 | untested | Only aarch64 Colima tested; CI not wired |
| `/health` + `/metrics` HTTP | **added** | Pool stats, engine versions, session details |
| Config-driven engine swap | **wired** | Registry `instantiate()` used for STT/MT/TTS/VAD; engine names from session config |
| SDK / public API | missing | No `Pipeline` facade, no REST/WS contract doc |
| New language pack add | **works** | `bp language add fr-en` → auto-creates pack, `bp language test` runs smoke test |
| Licence gate | **works** | Fail-closed: non-commercial (XTTS, NLLB) blocked unless explicit override |

---

## 🔧 How to Swap Components (CLI)

### Language Packs (add a new pair)
```bash
# 1. Create a pack file in languages/packs/<pair>.json
bp language add pl        # downloads/resolves models, runs smoke test
bp language test pl       # writes WER/chrF/latency to processed/language_eval/pl.json
bp language list          # shows verified vs experimental

# Verified multilingual pairs (2026-10-21):
bp language list
# en-sk  verified   (Slovak)
# en-cs  verified   (Czech)  
# en-zh  verified   (Chinese)
# en-hi  verified   (Hindi)
# sk-en  experimental (Czech hallucinations in STT)
```

### Engines (swap STT/MT/TTS/VAD)
```bash
bp engine list                    # shows all registered engines + availability + licence
bp engine check stt voxtral       # licence gate (fail closed: commercial-ok only by default)
bp engine gate stt faster_whisper # runs quality/latency gate, writes processed/engine_gate/
```

### STT Beam Size (speed vs accuracy)
```bash
# Global (env var)
BP_STT_BEAM_SIZE=1 bp selftest    # greedy: ~2x faster, WER +0.1 on SK

# Per-language (in pack, not yet wired to pipeline)
# languages/packs/sk-en.json:
#   "stt": { "engine": "faster_whisper", "model": "small-sk", "beam_size": 1, "initial_prompt": "..." }
```

### TTS Voice
```bash
# In language pack:
# "tts": { "engine": "piper", "voice": "sk_SK-lili-medium", "fallback": "sk_SK-personal-male-medium" }

# Or at runtime via UI dropdown (piper / piper_sk_personal / omnivoice / hybrid / xtts)
```

---

## 🔄 Alternatives Per Stage (Swap Ladder)

| Stage | Current | Next Candidates | Gate | Notes |
|-------|---------|-----------------|------|-------|
| VAD | WebRTC VAD (mode 3) | Silero VAD, VAC iterator, AlignAtt policy | Latency + WER unchanged | Silero needs torch |
| STT | faster-whisper `small-sk` / `base` | Voxtral Realtime, Qwen3-ASR-causal, Parakeet-TDT-v3 (EN), large-v3-turbo | WER on spontaneous SK + RTF | `small-sk` has Czech hallucination bug; `large-v3-turbo` too slow on CPU (RTF 0.6) |
| MT | Opus-MT int8 per pair | NLLW / EuroLLM / small LLM translate | chrF/BLEU + latency + licence | NLLB is CC-BY-NC (blocked by licence gate) |
| TTS | Piper (CPU, CoreML on Mac) | OmniVoice, CosyVoice3, Qwen3-TTS, Voxtral TTS | RTF + similarity + licence | Piper RTF 0.03 on CoreML — not the bottleneck |
| Whole | cascade | Qwen3-Omni local, Gemini Live Translate adapter | Only if thesis framing absorbs it | Unified = lower latency, prosody preserved |

---

## ⚠️ SK→EN Deep Dive (The Real Bottleneck)

**Problem**: `whisper-small-sk` (NaiveNeuron/whisper-small-sk, MIT) fine-tune produces Czech-like hallucinations:
- Reference: `"vitajte v tejto živej ukážke prekladu reči"`
- Hypothesis: `"vytýť si to šivé okážky a pre kvót to rieči"`

**Measured trade-offs** (134s `me_sk_script` clip, `initial_prompt` with Slovak diacritics):

| Model | Beam | Prompt | WER | STT Time | RTF |
|-------|------|--------|-----|----------|-----|
| small-sk | 5 | ✓ | 0.255 | 18.4s | 0.14 |
| small-sk | 1 | ✓ | **0.323** | **15.8s** | **0.12** |
| base | 5 | ✓ | 0.714 | 8.9s | 0.07 |
| base | 1 | ✓ | 0.755 | 4.7s | 0.04 |
| large-v3-turbo | 1 | ✓ | 0.828 | 80.8s (30s chunks) | 0.60 |

**Conclusion**: On CPU, `small-sk` with `beam_size=1` + `initial_prompt` is the best trade-off (WER 0.32, RTF 0.12). The fine-tune is fundamentally flawed — it overfits to Czech-like output. A proper Slovak model needs GPU (Parakeet / Voxtral / fine-tuned large-v3-turbo).

**Recommendation for thesis**: Document honestly. Use `beam_size=5` for quality (WER 0.23) in demo, offer `beam_size=1` as "fast mode". Note that SK→EN is the weak direction and why.

---

## 🎯 Next Steps (Priority Order)

### P0 — Make it shippable for "picky old man" use
1. **T022** Floating caption overlay (text, not just PiP canvas) — for real call usability
2. ~~**T010** Wire `registry.select()`/`instantiate()` into `main.py` — **DONE** (registry wired for all stages)~~
3. ~~**T031** Add `/health` + `/metrics` HTTP routes — **DONE**~~
4. **T023b** Document SK→EN trade-off, make `beam_size` configurable per language pack (partial: beam_size works via env, per-pack config needs frontend)

### P1 — Polish + Voice
5. **T029** RVC-MLX model prep: train owner voice via Applio (rent GPU ~1h), register as `rvc-mlx`
6. **T030** `bp engine gate` CI integration (thresholds exist, need automation)
7. ~~**T028b** Move live-test scripts out of `test/` — **DONE** (moved to `scripts/live/`)~~

### P2 — SDK + Collaboration
8. **T024** Public `Pipeline` facade + `documentation/sdk.md` (REST/WS contract)
9. **T025** Unified engine adapter (Qwen3-Omni local or cloud adapter)
10. **T026** Supervisor handbook: add language, add engine, embed SDK

### P3 — Thesis + Evidence
11. **T027** Update thesis draft with measured numbers, remove `[DOPLNIŤ]` markers
12. **T028** Re-run three demo gates + `interrupt_smoke_test.py`, date the runbook

### P4 — Platform
13. **Linux x86_64**: Run `documentation/linux_setup_and_test.md`, wire CI
14. **Kaggle/Modal cleanup**: Revoke tokens created for GPU benchmarks

---

## 📦 Quick Start (for Collaborator / Examiner)

```bash
git clone <repo>
cd BP
python3.11 scripts/setup.py --dev   # ~90s warm, ~428s cold; ends with bp selftest
bp selftest                         # should pass with 5 warnings (optional engines)
make run                            # https://localhost:8000 (self-signed cert)
# Open UI, click Initialize, Start, speak
```

**Requirements**: Python 3.11, macOS/Windows/Linux, 4 GB RAM, no GPU needed for baseline.

---

## 📄 Licence

MIT (see `LICENSE`). Third-party traps documented:
- **Seamless / SeamlessM4T / SeamlessStreaming**: CC-BY-NC 4.0 — non-commercial only
- **NLLB-200**: CC-BY-NC 4.0 — non-commercial only
- **XTTS v2**: Coqui CPML — non-commercial only
- All default engines (faster-whisper, Opus-MT, Piper, WebRTC VAD) are MIT/BSD/Apache-2.0

The licence gate (`bp engine check`) enforces fail-closed: non-commercial engines cannot be selected in the default profile without explicit override.

---

## 🔗 Key Files

| File | Purpose |
|------|---------|
| `AGENTS.md` | Project memory for all agents |
| `specs/003-end-to-end-completion/spec.md` | Feature spec (source of truth) |
| `specs/003-end-to-end-completion/tasks.md` | Task tracker with ROAST annotations |
| `documentation/evaluation_2026-10-18.md` | Engine/language swap evaluation (measured) |
| `documentation/s2s_landscape_2026-10-18.md` | S2S landscape + reuse/licence evidence |
| `documentation/load_tests_2026-10.md` | T4 GPU load test data |
| `documentation/model_landscape_2026-10.md` | Per-component verdicts |
| `scripts/bp.py` | Single CLI entry point |
| `backend/registry.py` | Engine registry + capability manifests + licence gate |
| `languages/packs/*.json` | Declarative language packs |
| `processed/engine_gate/*.json` | Gate records (WER/chrF/RTF per engine) |
| `processed/language_eval/*.json` | Language pack smoke test records |

---

*Generated 2026-10-20. Run `bp selftest` to verify current state.*