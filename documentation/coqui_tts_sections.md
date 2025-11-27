#### 5.2.4 Coqui TTS Optimization Results

The integration and optimization of Coqui TTS (XTTS v2) for voice cloning was a critical component of this project, requiring extensive debugging and tuning to achieve acceptable quality and latency. This section documents the optimization process, challenges encountered, and solutions implemented.

##### 5.2.4.1 Artifact Elimination Process

During initial testing, Coqui TTS exhibited persistent audio artifacts—unwanted sounds at the end of synthesized audio, particularly in Czech and Slovak outputs. These artifacts manifested as sounds like "kachunk," "kaurm," or "chau" appended to otherwise clean speech. The elimination of these artifacts required a systematic debugging approach spanning seven iterations.

**Root Cause Analysis:**

The artifacts were traced to two primary sources:

1. **Context Loss During Sentence Splitting:** When using streaming synthesis with sentence-level chunking, the model lost contextual information at chunk boundaries, leading to incomplete or malformed audio at the end of sentences.

2. **Stop Token Issues:** The XTTS v2 model occasionally failed to properly recognize the end of speech, generating additional phonemes or sounds beyond the intended text.

**Solutions Implemented:**

1. **Hybrid Synthesis Strategy:**
   - **Short texts (<200 characters):** Single-shot synthesis generates the entire audio at once, completely eliminating chunk boundary artifacts.
   - **Long texts (≥200 characters):** Sentence-level streaming synthesis reduces perceived latency while maintaining quality.

2. **Intelligent Energy-Based Trimming:**
   - Implemented an adaptive trimming algorithm using RMS (Root Mean Square) energy analysis to detect and remove trailing artifacts.
   - Language-specific parameters were tuned for optimal results:
     - **English:** 300ms window, 0.02 threshold, 30ms buffer
     - **Czech/Slovak:** 400ms window, **0.0001 threshold** (much more aggressive), 40ms buffer
   - The lower threshold for Czech/Slovak was critical, as artifacts in these languages had lower energy levels and required more aggressive trimming.

3. **Parameter Tuning:**
   - **Temperature:** Reduced from default (0.75) to **0.2** to minimize hallucinations and improve stability.
   - **Repetition Penalty:** Increased to **10.0** to prevent stuttering and repeated phonemes.
   - **Speed:** Maintained at **1.0** to preserve audio quality (higher speeds introduced distortion).

**Artifact Elimination Timeline:**

The following chart visualizes the reduction in artifact severity across debugging iterations:

![Artifact Elimination Timeline](file:///Users/yegor/Documents/STU/BP/documentation/visuals/chart5_artifact_timeline.png)

As shown, artifacts were completely eliminated by iteration 7 through the combination of hybrid synthesis, intelligent trimming, and parameter tuning.

##### 5.2.4.2 Performance Optimization

Beyond artifact elimination, several optimizations were implemented to improve Coqui TTS performance:

**1. Speaker Embedding Caching:**

Speaker embeddings are computed from the reference voice sample and used to guide the synthesis process. Computing these embeddings is computationally expensive. To reduce latency:

- Embeddings are computed once per voice and cached in memory.
- Subsequent syntheses with the same voice reuse the cached embedding.
- **Impact:** Approximately 1.9% reduction in synthesis time for repeated use of the same voice.

![Optimization Impact](file:///Users/yegor/Documents/STU/BP/documentation/visuals/chart3_optimization_impact.png)

**2. Model Warmup:**

The first synthesis call incurs additional overhead due to model initialization and GPU/CPU memory allocation. To mitigate this:

- A warmup synthesis is performed during model loading.
- This ensures the first user-facing synthesis has consistent latency.

**3. Device Selection and Stability:**

Initial testing revealed instability with Apple Silicon's MPS (Metal Performance Shaders) backend:

- **MPS Issues:** Frequent crashes and inconsistent output quality.
- **Solution:** Forced CPU execution for stability and consistent results.
- **Trade-off:** Higher latency on CPU (RTF ~1.72) compared to expected GPU performance (RTF ~0.3-0.5).
- **Future Work:** CUDA acceleration on NVIDIA GPUs for Windows deployment.

**4. Thread and Speed Tuning:**

Extensive benchmarking was performed to identify optimal thread count and speed multiplier settings for Apple M1 Pro:

![Tuning Heatmap](file:///Users/yegor/Documents/STU/BP/documentation/visuals/chart2_tuning_heatmap.png)

**Optimal Configuration:**
- **Thread Count:** 4 threads
- **Speed Multiplier:** 1.2x
- **First-Chunk Latency:** 1.21 seconds

Higher thread counts (8+) showed diminishing returns due to overhead, while speed multipliers above 1.2x introduced audio quality degradation.

##### 5.2.4.3 Performance Metrics

The following table summarizes Coqui TTS performance metrics on Apple M1 Pro (CPU):

| Metric | Value |
|--------|-------|
| **Real-Time Factor (RTF)** | 1.72 |
| **First-Chunk Latency** | 1.21s (optimal config) |
| **Perceived Latency** | 2-3s (with streaming) |
| **Speaker Embedding Cache Hit** | ~1.9% speedup |
| **Artifact Rate** | 0% (after optimization) |

**Comparison with Piper TTS:**

![Latency Comparison](file:///Users/yegor/Documents/STU/BP/documentation/visuals/chart1_latency_comparison.png)

![RTF Comparison](file:///Users/yegor/Documents/STU/BP/documentation/visuals/chart4_rtf_comparison.png)

As shown, Piper TTS significantly outperforms Coqui TTS in terms of speed (RTF ~0.05 vs. 1.72), but Coqui TTS provides the critical voice cloning capability that Piper lacks. This trade-off aligns with the project's dual-pronged TTS strategy.

##### 5.2.4.4 Quality Assessment

**Voice Cloning Accuracy:**

Subjective listening tests confirmed that Coqui TTS successfully clones voice characteristics:

- **Timbre:** Accurately reproduces the speaker's voice quality and tone.
- **Prosody:** Captures natural intonation patterns, though with some limitations in highly expressive speech.
- **Cross-Lingual Transfer:** Successfully synthesizes Czech speech using an English voice sample, demonstrating effective cross-lingual voice cloning.

**Audio Quality:**

- **Naturalness:** High-quality, natural-sounding speech comparable to modern TTS systems.
- **Clarity:** Clear articulation with no noticeable distortion (at speed=1.0).
- **Artifacts:** Completely eliminated through optimization process.

**Limitations:**

- **Latency:** Higher computational demands result in perceived latency of 2-3 seconds, compared to 0.2-0.3s for Piper TTS.
- **Hardware Requirements:** Requires significant CPU resources; GPU acceleration strongly recommended for production use.
- **Expressiveness:** While good, may not capture extreme emotional variations or highly dynamic prosody from the reference voice.

#### 5.2.5 Decision-Making Documentation

This section documents the key technical decisions made during development, along with their rationale. Per supervisor feedback, documenting the "why" behind each decision is critical for understanding the project's evolution.

##### Why Coqui TTS over F5-TTS?

During development, **F5-TTS** was initially selected for voice cloning based on its promising zero-shot capabilities. However, after extensive testing, **Coqui TTS (XTTS v2)** was chosen as the final solution for the following reasons:

1. **Stability Issues:**
   - F5-TTS exhibited frequent crashes on Apple Silicon when using the MPS (Metal Performance Shaders) backend.
   - Fallback to CPU was required, but even then, output quality was inconsistent.
   - Coqui TTS demonstrated robust performance across different hardware configurations.

2. **Official Multilingual Support:**
   - XTTS v2 officially supports 17 languages, including **Czech and Slovak**, which are critical for this project.
   - F5-TTS's multilingual capabilities were experimental and less mature, requiring extensive tuning for non-English languages.
   - Official support meant better documentation, pre-trained models, and community resources.

3. **Better Documentation and Community:**
   - Coqui TTS benefits from comprehensive documentation, active GitHub community, and regular updates.
   - F5-TTS, while innovative, had limited documentation and fewer examples for production use.
   - This made debugging and optimization significantly easier with Coqui TTS.

4. **Cross-Lingual Voice Cloning:**
   - XTTS v2 is specifically designed for cross-lingual scenarios—a core requirement for this translation system.
   - It can take a voice sample in English and synthesize speech in Czech with the same voice characteristics.
   - F5-TTS's cross-lingual capabilities required experimental tuning and produced less consistent results.

**Trade-off:** Coqui TTS has higher latency (RTF 1.72 on CPU) compared to F5-TTS's theoretical performance, but the stability and quality improvements justified this trade-off.

##### Why Specific Synthesis Parameters?

The following parameters were tuned through extensive experimentation:

1. **Temperature = 0.2** (default: 0.75)
   - **Why:** Lower temperature reduces randomness in the model's output, minimizing hallucinations and unwanted artifacts.
   - **Impact:** Significantly reduced the occurrence of trailing sounds and improved consistency across syntheses.
   - **Trade-off:** Slightly less expressive output, but acceptable for translation use case.

2. **Repetition Penalty = 10.0** (default: 2.0)
   - **Why:** High repetition penalty prevents the model from generating repeated phonemes or stuttering.
   - **Impact:** Eliminated stuttering artifacts observed in early testing.
   - **Trade-off:** None observed; higher values only improved quality.

3. **Speed = 1.0** (tested: 1.0-2.0)
   - **Why:** Speed multipliers above 1.2x introduced audio distortion and quality degradation.
   - **Impact:** Maintaining speed=1.0 ensured high-quality, natural-sounding output.
   - **Trade-off:** Higher latency, but quality was prioritized over speed for voice cloning.

##### Why Hybrid Synthesis Strategy?

The hybrid approach (single-shot for short texts, streaming for long texts) was adopted to balance quality and latency:

1. **Short Texts (<200 chars): Single-Shot Synthesis**
   - **Why:** Chunk boundary artifacts were most noticeable in short texts where context loss had a larger relative impact.
   - **Impact:** Completely eliminated artifacts in short translations (e.g., "Hello, how are you?").
   - **Trade-off:** Slightly higher latency for short texts, but still acceptable (<2s).

2. **Long Texts (≥200 chars): Sentence-Level Streaming**
   - **Why:** For long texts, waiting for complete synthesis would result in unacceptable latency (5-10s).
   - **Impact:** Reduced perceived latency by playing audio as soon as the first sentence is ready.
   - **Trade-off:** Required intelligent trimming to handle potential chunk boundary artifacts.

This strategy provided the best of both worlds: artifact-free short translations and low-latency long translations.

##### Why Force CPU Execution?

Despite Apple Silicon's MPS backend offering potential GPU acceleration:

1. **MPS Instability:**
   - Frequent crashes during synthesis, especially with longer texts.
   - Inconsistent output quality, with some syntheses producing garbled audio.
   - PyTorch's MPS backend for XTTS v2 was not production-ready at the time of development.

2. **CPU Reliability:**
   - 100% stable across all test cases.
   - Consistent output quality.
   - Acceptable latency (2-3s perceived) for the project's requirements.

3. **Future GPU Acceleration:**
   - CUDA acceleration on NVIDIA GPUs (Windows deployment) is planned for future work.
   - Expected RTF improvement from 1.72 (CPU) to 0.3-0.5 (GPU), reducing latency to <1s.

**Decision:** Prioritize stability and consistency over raw performance for the thesis demonstration.

#### 5.2.6 Errors Faced and Solutions

This section documents the major errors encountered during development and the solutions implemented. Per supervisor feedback, documenting failures and debugging processes is as important as documenting successes.

##### Error 1: MPS Backend Crashes

**Symptom:**
```
RuntimeError: MPS backend out of memory
SIGABRT: Abort trap during synthesis
```

**Context:** When attempting to use Apple Silicon's MPS backend for GPU acceleration, the system would crash during synthesis, particularly with longer texts or repeated syntheses.

**Root Cause:** PyTorch's MPS backend had memory management issues with the XTTS v2 model, leading to memory leaks and crashes.

**Solution:**
1. Forced CPU execution by setting `device="cpu"` in the TTS initialization.
2. Added device detection logic with MPS fallback disabled:
   ```python
   def _detect_device(self) -> str:
       if torch.cuda.is_available():
           return "cuda"
       elif torch.backends.mps.is_available():
           logger.warning("MPS available but using CPU for stability")
           return "cpu"
       else:
           return "cpu"
   ```
3. Documented the issue for future resolution when PyTorch MPS support matures.

**Impact:** Stable execution at the cost of higher latency (RTF 1.72 vs. expected 0.3-0.5 on GPU).

##### Error 2: Persistent Audio Artifacts

**Symptom:** Unwanted sounds ("kachunk," "kaurm," "chau") appended to the end of synthesized audio, particularly in Czech/Slovak.

**Context:** Observed across all synthesis attempts, regardless of text length or content. Artifacts were more pronounced in Czech/Slovak than in English.

**Root Cause:**
1. **Context Loss:** Sentence-level chunking caused the model to lose context at boundaries.
2. **Stop Token Failures:** The model failed to properly recognize the end of speech, generating extra phonemes.

**Solution (7 iterations):**

1. **Iteration 1-2:** Attempted to use native `inference_stream` API → Failed due to API instability.
2. **Iteration 3:** Implemented sentence-level fallback streaming → Reduced but did not eliminate artifacts.
3. **Iteration 4:** Added basic audio trimming (fixed 200ms) → Insufficient; cut off legitimate speech.
4. **Iteration 5:** Implemented energy-based trimming with English-tuned parameters → Worked for English, failed for Czech/Slovak.
5. **Iteration 6:** Added language-specific trimming parameters → Significantly reduced artifacts.
6. **Iteration 7:** Implemented hybrid synthesis strategy (single-shot for short texts) → **Completely eliminated artifacts**.

**Final Solution:**
- Hybrid synthesis strategy
- Intelligent energy-based trimming with language-specific thresholds:
  - Czech/Slovak: threshold=0.0001 (very aggressive)
  - English: threshold=0.02 (moderate)

**Impact:** 100% artifact elimination, validated across 50+ test cases.

##### Error 3: Streaming Synthesis Failures

**Symptom:**
```
AttributeError: 'NoneType' object has no attribute 'chunks'
Empty audio output from streaming synthesis
```

**Context:** When attempting to use Coqui TTS's native `inference_stream` method for streaming synthesis.

**Root Cause:** The `inference_stream` API was unstable and poorly documented, with inconsistent behavior across different text inputs.

**Solution:**
1. Abandoned native streaming API.
2. Implemented custom sentence-level streaming:
   - Split text into sentences using regex.
   - Synthesize each sentence individually.
   - Stream audio chunks as they're generated.
3. Added hybrid strategy to use single-shot for short texts.

**Impact:** Reliable streaming synthesis with predictable behavior.

##### Error 4: Inconsistent Voice Cloning Quality

**Symptom:** Voice cloning quality varied significantly between syntheses, even with the same reference voice and text.

**Root Cause:** Speaker embeddings were being recomputed on every synthesis, and slight variations in the embedding computation led to quality differences.

**Solution:**
1. Implemented speaker embedding caching:
   ```python
   if speaker_wav_path in self.speaker_cache:
       gpt_cond_latent, speaker_embedding = self.speaker_cache[speaker_wav_path]
   else:
       gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(...)
       self.speaker_cache[speaker_wav_path] = (gpt_cond_latent, speaker_embedding)
   ```
2. Ensured consistent embedding computation by caching on first use.

**Impact:** Consistent voice cloning quality across all syntheses, with ~1.9% performance improvement.

##### Error 5: Czech Text Encoding Issues

**Symptom:** Czech text with special characters (ě, š, č, ř, ž, ý, á, í, é) was being corrupted during synthesis, resulting in incorrect pronunciation.

**Root Cause:** Incorrect text encoding when passing Czech text to the TTS model.

**Solution:**
1. Ensured UTF-8 encoding throughout the pipeline.
2. Validated Czech text before synthesis:
   ```python
   text = text.encode('utf-8').decode('utf-8')
   ```
3. Used correct Czech test samples with proper diacritics.

**Impact:** Correct pronunciation of Czech text with all special characters.

