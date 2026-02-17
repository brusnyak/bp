# ===============================================================================
# LIVE SPEECH TRANSLATION - OPTIMIZED REQUIREMENTS
# Bachelor's Thesis Project - Conference Live Translation System
# Target Hardware: Apple Silicon (M1/M2/M3)
# ===============================================================================

# === CRITICAL NOTES ===
# 1. PyTorch MPS (Metal Performance Shaders) support for Apple Silicon
# 2. faster-whisper uses CTranslate2 which is CPU-optimized (no native MPS support)
# 3. For best Apple Silicon performance, consider whisper.cpp or MLX alternatives
# 4. All versions tested for compatibility on 2024-11-06

# ===============================================================================
# CORE FRAMEWORK - PyTorch with Apple Silicon MPS Support
# ===============================================================================
# Stable versions with proven MPS compatibility for M1/M2/M3
torch==2.2.2
torchaudio==2.2.2
# NOTE: torchaudio requires FFmpeg. On macOS, install with `brew install ffmpeg`.
# --extra-index-url https://download.pytorch.org/whl/nightly/cpu # Commented out to prioritize MPS support

# ===============================================================================
# TEXT-TO-SPEECH (TTS) - CURRENT: Piper (Production)
# ===============================================================================
# Piper TTS: Fastest open-source TTS (< 1 second for short texts)
# - Real-time factor: < 1.0 (faster than real-time)
# - CPU optimized, works excellently on Apple Silicon
# - Uses ONNX Runtime for inference
# - Perfect for live translation scenarios
piper-tts==1.3.0  # Keeping 1.3.0 as it's newer and likely compatible

# ONNX Runtime for Piper inference
onnxruntime==1.19.2

# ===============================================================================
# SPEECH-TO-TEXT (STT) - faster-whisper
# ===============================================================================
# faster-whisper: Optimized Whisper implementation using CTranslate2
# NOTE: Uses CPU optimization, not native MPS
# For Apple Silicon, consider alternatives:
#   - whisper.cpp (C++ port, ~15x faster on M1)
#   - mlx-whisper (Apple MLX framework, native Metal support)
#   - insanely-fast-whisper (optimized with batching)
faster-whisper==1.1.0  # Pinned to stable version from update.txt
ctranslate2==4.5.0     # Updated to newer stable version from update.txt

# ===============================================================================
# MACHINE TRANSLATION (MT) - SeamlessM4T v2
# ===============================================================================
# SeamlessM4T v2 with UnitY2 architecture
# - 3x faster inference than v1 for speech tasks
# - Supports 101 speech input languages, 96 text languages
# - Real-time capable with proper optimization
transformers==4.38.2      # Updated for compatibility with cached-path and huggingface-hub
sentencepiece==0.2.0      # For tokenization from update.txt
accelerate==1.2.1         # Updated for better performance from update.txt
protobuf==3.20.3          # Required by sentencepiece from update.txt

# ===============================================================================
# AUDIO PROCESSING & UTILITIES
# ===============================================================================
# Core audio processing libraries
librosa==0.10.2.post1     # Audio analysis (compatible with numba) from update.txt
soundfile==0.12.1         # Audio file I/O from update.txt
scipy==1.14.1             # Scientific computing from update.txt
numba==0.60.0             # JIT compilation for librosa
av==13.1.0                # Python bindings for FFmpeg from update.txt
pydub==0.25.1             # Simple audio manipulation

# Numpy - critical compatibility constraint
# Must be < 2.0 for librosa compatibility
numpy==1.26.4

# ===============================================================================
# EVALUATION METRICS
# ===============================================================================
jiwer==3.0.4              # Word Error Rate (WER) for STT evaluation from update.txt
sacrebleu==2.5.1          # BLEU score for translation quality (updated for compatibility)
pandas==2.2.3             # Data analysis from update.txt
scikit-learn==1.5.2       # ML utilities from update.txt
nltk==3.9.2               # NLP toolkit

# ===============================================================================
# HUGGING FACE & MODEL UTILITIES
# ===============================================================================
safetensors==0.4.5        # Safe model serialization from update.txt
tokenizers==0.15.2        # Fast tokenizers (updated for compatibility with transformers 4.38.2, faster-whisper 1.1.0 and Python 3.10)

# ===============================================================================
# COMMON DEPENDENCIES
# ===============================================================================
tqdm==4.67.1              # Progress bars
pyyaml==6.0.2             # Configuration files from update.txt
regex==2024.11.6          # Regular expressions from update.txt
requests==2.32.3          # HTTP library from update.txt
packaging==24.2           # Version parsing from update.txt
jinja2==3.1.4             # Templating from update.txt
markupsafe==3.0.2         # String escaping for Jinja2 from update.txt
typing-extensions==4.12.2 # Type hints backport from update.txt
filelock==3.13.1          # File locking (updated for compatibility with cached-path)
fsspec==2024.10.0         # Filesystem abstractions from update.txt

# ===============================================================================
# WEB API / STREAMING
# ===============================================================================
fastapi==0.115.6          # Modern async web framework from update.txt
uvicorn[standard]==0.32.1 # ASGI server from update.txt
websockets==14.1          # WebSocket support for streaming from update.txt
python-multipart==0.0.20  # Form data parsing from update.txt

# ===============================================================================
# VOICE ACTIVITY DETECTION (VAD)
# ===============================================================================
webrtcvad==2.0.10         # Lightweight VAD

# ===============================================================================
# DEVELOPMENT TOOLS
# ===============================================================================
pre-commit==4.0.1         # Git hooks for code quality from update.txt
black==24.4.0             # Keeping black as a common dev tool (updated for compatibility)

# ===============================================================================
# F5-TTS - NEW, Fast & High Quality (RECOMMENDED for Voice Cloning)
# ===============================================================================
# Install from GitHub (no PyPI package yet):
git+https://github.com/SWivid/F5-TTS.git
# F5-TTS dependencies are now managed by its own setup.py/pyproject.toml
# Note: F5-TTS uses same base dependencies (torch, torchaudio, transformers)
