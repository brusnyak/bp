# Third-party licence notice (working draft, not legal advice)

BP's own code is MIT (`LICENSE`). Components it can run have their own terms:

| Component | Terms | Consequence |
|---|---|---|
| Piper TTS runtime (`piper-tts` 1.3.0) | GPL-3.0-or-later (package metadata; upstream OHF-Voice/piper1-gpl) | Fine to run locally. Shipping Piper inside a bundle that is not GPL-compatible is a decision for the maintainer; keeping it a separately installed dependency is the cleaner boundary. Registry label: `gpl-3.0`. |
| Piper voice checkpoints | per voice, not reviewed | Do not redistribute until each voice licence is read. |
| Custom fine-tuned SK voice | personal voice model | Local only, never committed or shared (AGENTS.md hard rules). |
| OmniVoice (clone engine) | licence unknown in the registry | Research candidate; review model card before any distribution. |
| SeamlessM4T v2 | CC-BY-NC | Quality reference, **declared-only** in the current registry. Licence labels warn; current registry does not block solely on licence classification. |
| NVIDIA Canary-1B | CC-BY-4.0 | Attribution required. |
| Opus-MT (Helsinki-NLP), faster-whisper/CTranslate2 | see each model card / repo | Not re-verified this session. |

Verified in repo: `backend/registry.py` labels and `README.md` Piper note. Everything else above is from the registry notes, not re-read from upstream.

## Current integration boundary (2026-11-19)

BP is an orchestration application: it installs or invokes third-party packages and downloads selected model/voice assets through setup scripts. Its MIT licence applies to BP's own code, **not** to those dependencies or weights. The current registry warns for non-commercial or unknown classifications but is not a legal compliance check; actual rights depend on the upstream component, model card, usage, attribution, and distribution method. Do not claim that routing or separate installation automatically eliminates GPL, non-commercial or other obligations. This notice is not a completed legal audit.
