# BP v2 Streaming Spec — Near Real-Time STT→MT→TTS

**Status:** Implemented and tested (EN→SK, EN→CS verified on Mac M1)  
**Replaces:** v1 segment-based pipeline (wait for silence → full segment → translate)  
**Goal:** First translated audio ≤ ~1s after the speaker's word, sustained under continuous speech.  
**Preserves:** Natural pauses and prosody by synthesizing each phrase as a complete utterance.

---

## 1. Problem with v1

v1 waits for VAD silence timeout (0.25s) or max segment (3s) before running STT→MT→TTS.
- First-word delay: 3s segments → up to ~6s; 6s segments → ~8s
- Not "near real-time" — it's turn-by-turn translation
- Playback backlog grows on continuous speech (translated audio ≈ source duration)

## 2. v2 Streaming Design (Implemented)

```
Continuous audio buffer → VAD (segmentation only, never drops audio) →
STT on overlapping 2.0s windows every 0.5s → word timestamps (Parakeet/Whisper) →
Text-based dedup → phrase chunking (gap >0.3s between words = new phrase) →
MT (Opus-MT CT2) → TTS (Piper CPU or OmniVoice GPU) → WAV chunks over WebSocket →
listener playback
```

### 2.1 Per-frame processing (hot path)

Every 0.3s audio chunk from the WebSocket:

1. **Buffer** — accumulate audio continuously into a rolling buffer (never drops speech)
2. **VAD** — WebRTC VAD (aggressiveness=1) detects speech/silence for utterance boundaries only
3. **STT step** — every 0.5s, run STT on the last 2.0s of buffered audio
   - **EN source:** Parakeet TDT v3 (GPU/MPS, ~0.1s per phrase)
   - **SK/CS source:** Whisper small-sk/small (CPU, ~0.3s per phrase)
4. **Text-based dedup** — track synthesized text, only process new text from each STT call
5. **Phrase chunking** — committed words grouped into phrases by natural pauses (>0.3s gap between words)
6. **MT** — translate the phrase immediately (Opus-MT CT2, ~30-80ms)
7. **TTS** — synthesize the phrase as a complete utterance (Piper CPU, ~40-150ms)
8. **Emit** — WAV chunk sent over WebSocket as bytes; `translation_result` JSON with metadata

### 2.2 Latency budget (measured, Mac M1)

| Stage | Measured (CPU M1) |
|-------|-------------------|
| STT (Parakeet EN) | ~100-150ms |
| STT (Whisper SK/CS) | ~250-380ms |
| MT chunk | ~30-80ms |
| TTS (Piper) | ~40-150ms |
| **First audio total** | **~200-350ms** |
| Steady-state lag | ~0.3-0.5s |

### 2.3 What "near real-time" means here

- **First audio**: translated audio begins playing within ~0.3-0.5s of the speaker's phrase
- **Steady-state**: once the pipeline is warm, each phrase flows through with ~0.3-0.5s added lag
- **Phrase-based, not word-by-word**: phrases of 3-9 words are the unit of translation (MT quality needs sentence context, TTS quality needs prosodic context)
- **No segment-close wait**: audio is processed continuously, not batched into 3-6s segments
- **Natural prosody**: each phrase is synthesized as a complete utterance, preserving the speaker's pauses and intonation

---

## 3. Code Locations

| Component | File |
|-----------|------|
| Streaming pipeline | `backend/streaming/pipeline.py` |
| Parakeet STT | `backend/stt/parakeet_stt.py` |
| Whisper STT | `backend/stt/faster_whisper_stt.py` |
| MT engine | `backend/mt/ctranslate2_mt.py` |
| TTS engines | `backend/tts/base.py` (registry), `backend/tts/piper_tts.py` |
| WebSocket handler | `app.py` (`/ws`) |
| Hardware auto-config | `backend/hardware_scan.py` |
| Meeting session manager | `backend/session_manager.py` |

---

## 4. Configuration

Environment variables (overrides for auto-config):

| Env var | Default | Meaning |
|---------|---------|---------|
| `BP_DEVICE` | auto | cpu/cuda/mps |
| `BP_COMPUTE_TYPE` | int8 | int8/float16/float32 |
| `BP_MT_DEVICE` | cpu | cpu/cuda |
| `BP_STT_MODEL_SIZE` | base | STT model size |
| `BP_STT_BEAM_SIZE` | 5 | Whisper beam width |
| `BP_STT_STEP_S` | 0.5 | STT step interval (seconds) |
| `BP_STT_WINDOW_S` | 2.0 | STT audio window (seconds) |
| `BP_SILENCE_GAP_S` | 0.3 | Gap between words for phrase split (seconds) |
| `BP_UTTERANCE_SILENCE_S` | 0.8 | Silence that ends an utterance (seconds) |
| `BP_MIN_UTTERANCE_S` | 1.0 | Minimum utterance length for STT (seconds) |

---

## 5. Testing

### 5.1 Local (CPU)

```bash
# Full EN→SK demo with streaming
.venv/bin/python scripts/live/paced_replay.py speaker_voices/me_en_script.m4a \
    --src en --tgt sk

# SK→EN (bottleneck direction)
.venv/bin/python scripts/live/paced_replay.py speaker_voices/me_sk_script.m4a \
    --src sk --target en --stt-model small-sk --beam-size 5

# Stop/recovery test
.venv/bin/python scripts/test_stop_recovery.py

# 30-min soak test (queue stability)
.venv/bin/python scripts/test_soak.py
```

### 5.2 Kaggle T4 (GPU)

```bash
# Push XTTS voice clone kernel
kaggle kernels push -p scripts/gpu_bench/kaggle_xtts_clone --accelerator NvidiaTeslaT4 -t 1500

# Check status
kaggle kernels status yegorby/bp-xtts-voice-clone-t4

# Download results
kaggle kernels output yegorby/bp-xtts-voice-clone-t4 -p /tmp/bp-xtts-results
```

---

## 6. STT Error Correction

STT errors are mostly phonetic confusions (e.g., "real time" → "real term", "Latency" → "Ladency"). Mitigation options:

| Approach | Effort | Expected Impact |
|----------|--------|-----------------|
| Post-processing dictionary (map known mishearings) | Low | 30-50% reduction on domain terms |
| Initial prompt with domain vocabulary | Low | 10-20% WER reduction |
| Whisper LoRA fine-tune on speaker voice | Medium | 30-50% WER reduction |
| Speaker adaptation ( Parakeet/Whisper fine-tune) | High | 50%+ WER reduction |

**Recommended:** Start with post-processing dictionary + initial prompt. Fine-tune if WER remains >0.1 after those.

---

## 7. What Is NOT in v2 (future work)

- True word-by-word streaming (sub-300ms per word) — needs streaming STT model (Canary AST, Nemo)
- Speaker diarization (who spoke when in a shared microphone)
- Server-side audio mixing for multiple listeners with different languages
- These are v3 considerations
