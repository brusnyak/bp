# BP License Inventory — 2026-11-06

**Purpose:** Commercial-use readiness audit. Fail-closed: non-commercial and unknown-licence
engines are blocked unless explicitly opted in. Piper is GPL-3.0-or-later (not MIT) — it is
fine to run locally but shipping it inside a non-GPL bundle is a licence decision for the owner.

## Pipeline components (local-first, no cloud in the pipeline)

| Component | Source | Licence | Commercial OK? | Notes |
|-----------|--------|---------|----------------|-------|
| faster-whisper (STT) | guillaumekln/faster-whisper | MIT | ✅ Yes | CT2 backend, no torch on Windows |
| CTranslate2 (MT inference) | OpenNMT/CTranslate2 | Apache-2.0 | ✅ Yes | CPU/CUDA, no MPS |
| Opus-MT (MT models) | Helsinki-NLP/opus-mt-* | CC-BY-4.0 | ✅ Yes | Attribution required |
| whisper-base (STT model) | openai/whisper | MIT | ✅ Yes | EN STT, base size |
| whisper-small-sk (STT model) | NaiveNeuron/whisper-small-sk | MIT | ✅ Yes | SK STT, flawed (Czech-like hallucinations, WER 0.25–0.32) |
| Piper TTS | rhasspy/piper | GPL-3.0-or-later | ⚠️ Locally only | Not bundled; user installs via `scripts/setup.py --dev`. Shipping in a non-GPL bundle requires owner licence decision. |
| sk_SK-personal-male-medium (voice) | Owner fine-tune | Owner's voice | ✅ Yes | Owner's own recording, fine-tuned from Piper base |
| en_US-john-medium (TTS voice) | rhasspy/piper voices | Public domain / CC0 | ✅ Yes | Generic English fallback |
| sk_SK-lili-medium (TTS voice) | rhasspy/piper voices | Public domain / CC0 | ✅ Yes | Generic Slovak fallback |
| WebRTC VAD | webrtcvad | BSD-style | ✅ Yes | Always CPU |
| soundfile | python-soundfile | BSD | ✅ Yes | WAV I/O |
| FastAPI | tiangolo/fastapi | MIT | ✅ Yes | Web framework |
| uvicorn | encode/uvicorn | BSD-3-Clause | ✅ Yes | ASGI server |
| WebSockets | aymeric/websockets | Apache-2.0 | ✅ Yes | WebSocket library |
| argon2 | hygge/argon2-cffi | MIT | ✅ Yes | Password hashing |
| slowapi | slowapi | BSD-3-Clause | ✅ Yes | Rate limiting (optional) |
| SQLAlchemy | sqlalchemy | MIT | ✅ Yes | Database ORM |

## GPU candidate engines (research candidates, not in live route)

| Component | Source | Licence | Commercial OK? | Notes |
|-----------|--------|---------|----------------|-------|
| Canary AST (STT) | NVIDIA/Canary | CC-BY-4.0 | ✅ Yes | SK→EN ASR WER 0.0477 on T4; chrF++ ~52 below commercial bar. Speed path, not quality path. NOT promoted to live route. |
| Kokoro (TTS) | hexgrad/Kokoro | Apache-2.0 | ✅ Yes | EN TTS, RTF 0.03, 1.33 GB VRAM on T4. Load test pending. |
| Qwen3-TTS | Qwen/Qwen3-TTS | MIT | ⚠️ GPU only | EN zero-shot cloning, 4–8 GB VRAM on T4. Research candidate. |
| Chatterbox | ResembleAI/Chatterbox | MIT | ⚠️ GPU only | EN zero-shot cloning, 4–8 GB VRAM on T4. Research candidate. |
| SeamlessM4T v2 (S2S) | facebook/seamless-m4t-v2 | CC-BY-NC | ❌ No | Non-commercial. Quality reference only, not for commercial use. |
| XTTS v2 (TTS) | Coqui/XTTS | CC-BY-NC 4.0 | ❌ No | Non-commercial. Blocked by licence gate unless explicit `allow_noncommercial=True`. |
| NLLB (MT) | facebook/nllb-200 | CC-BY-NC | ❌ No | Non-commercial. Blocked by licence gate unless explicit `allow_noncommercial=True`. |
| OmniVoice | OmniVoice | HF licence | ⚠️ TBC | HF licence to confirm. GPU-only zero-shot cloning. |

## Licence gate behaviour (backend/registry.py)

- `commercial-ok`: ships in default profile
- `gpl-3.0` (piper): fine to run locally; shipping in a non-GPL bundle is a licence decision for the owner
- `non-commercial`: blocked unless `allow_noncommercial=True` (XTTS, NLLB)
- `unknown`: blocked unless `allow_unknown=True` (confirm model card first)

Both `select` and `instantiate` raise instead of silently passing an unlabelled engine.

## Not verified

- OmniVoice HF licence not yet confirmed
- No full licence audit completed for all model weights (only engine licences above)
- Personal voice models and recordings are local-only, never committed

## Last line

🟢 Piper GPL-3.0-or-later (not bundled — user installs via setup.py) | 🟢 Opus-MT CC-BY-4.0 | 🟢 faster-whisper/CT2 MIT/Apache-2.0 | 🟡 OmniVoice HF licence unconfirmed | 🔴 XTTS/NLLB/SeamlessM4T non-commercial — blocked by gate