# TTS Alternatives Research for Low-Latency Voice Cloning

## Problem with Current XTTS Approach
- XTTS v2 has high latency (typically 2-5+ seconds for voice cloning)
- This prevents achieving <1s end-to-end latency goal
- Not suitable for real-time conversational applications

## Evaluated Alternatives (2024-2025)

### 1. Voxtral TTS (Mistral AI)
- **Latency**: 70ms time-to-first-audio, 9.7x real-time factor
- **Voice Cloning**: Zero-shot from 3 seconds of reference audio
- **Languages**: 9 (EN, FR, DE, ES, NL, PT, IT, HI, AR)
- **License**: Open weights, Apache 2.0
- **Model Size**: 4B parameters
- **Key Benefits**: 
  - Sub-100m latency enables true real-time conversation
  - Competitive voice cloning quality with ElevenLabs
  - Open source allows self-hosting for privacy compliance
- **Considerations**: 
  - Limited language coverage (9 languages)
  - High VRAM requirement (~16GB for full model)

### 2. Qwen3-TTS (Alibaba Cloud)
- **Latency**: 97ms first-packet latency (0.6B model)
- **Voice Cloning**: Zero-shot from 3 seconds of reference audio
- **Languages**: 10 (including multilingual variant with 23 languages)
- **License**: Apache 2.0
- **Model Sizes**: 0.6B (fast) and 1.7B (high quality)
- **Key Benefits**:
  - Excellent balance of speed and quality
  - Streaming support for real-time applications
  - Good multilingual coverage
  - Lower VRAM requirements than Voxtral
- **Considerations**:
  - Slightly higher latency than Voxtral but still excellent
  - 0.6B model may have slightly lower quality than 1.7B

### 3. Chatterbox Turbo (Resemble AI)
- **Latency**: 75ms on GPU, sub-200ms on consumer GPUs
- **Voice Cloning**: Zero-shot from 5 seconds of reference audio
- **Languages**: 23+ (multilingual variant)
- **License**: MIT
- **Model Size**: 350M parameters
- **Key Benefits**:
  - Extremely fast inference (6x faster than real-time on GPU)
  - Built-in watermarking for accountability
  - Paralinguistic prompting for expressive control
  - Excellent voice cloning quality (63.75% preference vs ElevenLabs in blind tests)
- **Considerations**:
  - Requires 5 seconds of reference audio (longer than ideal)
  - English-optimized base model (multilingual variant available)

### 4. OmniVoice (K2-FSA Team)
- **Latency**: 40x real-time inference speed
- **Voice Cloning**: Zero-shot capability
- **Languages**: 600+
- **License**: Apache 2.0
- **Model Base**: Qwen3-0.6B
- **Key Benefits**:
  - Massive language coverage
  - Extremely fast inference
  - Combines diffusion and LLM advantages
- **Considerations**:
  - Newer project (3 weeks old as of Apr 2026)
  - Less community validation than established options

## Recommendation for BP Project

### Primary Recommendation: OmniVoice (K2-FSA Team)

Based on our evaluation, **OmniVoice** is the recommended TTS solution for this project:

- ✅ **Supports Czech & Slovak** (confirmed in 600+ language list)
- ⚡ **40x real-time inference** (~25ms latency vs XTTS 2-5s)
- 🎤 **State-of-the-art voice cloning** (0.830 speaker similarity)
- 🎤 **Voice Design feature** (text-controlled voice attributes)
- 🔓 **Apache 2.0 license** (commercial friendly)
- 💻 **Self-hosting capable** (privacy/compliance)
- 📊 **2.85% WER** (beats competitors)

### Integration Status

OmniVoice is now integrated into the backend:
- Backend module: `backend/tts/omni_tts.py`
- Select via TTS model choice: `"omnivoice"` in the frontend
- Requires: `pip install omnivoice`

### Fallback Options

1. **Piper TTS** - Fast synthesis, no cloning (default, lowest latency)
2. **XTTS (Coqui)** - Voice cloning, higher latency (~2-5s)
3. **Hybrid (Piper + OpenVoice)** - Voice cloning via tone conversion

For the BP speech-to-speech translation project focusing on Czech/Slovak:

1. **Primary Recommendation**: Qwen3-TTS 0.6B model
   - Provides 97ms latency suitable for <1s end-to-end goal
   - Supports 10 languages including European languages
   - Can be extended to Czech/Slovak using phonetic similarities
   - Lower VRAM requirements enable broader deployment
   - Apache 2.0 license compatible with current project

2. **Alternative**: Voxtral TTS if Czech/Slovak support can be added
   - Best latency (70ms) 
   - But currently doesn't support Czech/Slovak natively

3. **Hybrid Approach**: Use Piper TTS for regular speech + reference audio encoding
   - Keep ultra-fast Piper for regular TTS (<50ms)
   - Use reference audio to modulate prosody and timbre
   - This could achieve voice preservation with minimal latency impact

## Implementation Considerations

1. **Model Integration**:
   - Add new TTS class similar to existing PiperTTS/CoquiTTS
   - Update model initialization in backend/main.py
   - Modify TTS selection logic in _process_speech_segment_pipeline

2. **Voice Cloning Adaptation**:
   - For Czech/Slovak, consider using phonetic mapping to supported languages
   - Example: Map Slovak to Czech for voice cloning since they're closely related
   - Or use multilingual variants that support more languages

3. **Latency Optimization**:
   - Ensure streaming inference is enabled
   - Use appropriate batching for concurrent requests
   - Optimize audio buffering and preprocessing

4. **Fallback Strategy**:
   - Maintain XTTS as fallback for highest quality when latency less critical
   - Allow users to choose between speed and quality modes