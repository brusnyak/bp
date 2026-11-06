# BP — Docker image for server deployment
# Build:  docker build -t bp:latest .
# Run:    docker run -p 8000:8000 -e BP_HOST=0.0.0.0 bp:latest
#         (add -e BP_HTTPS=1 for TLS with certs mounted)

FROM python:3.11-slim

WORKDIR /app

# System dependencies: FFmpeg (audio conversion), webrtcvad needs nothing special
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (core runtime only; torch-free where possible)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    faster-whisper \
    ctranslate2 \
    piper-tts \
    librosa \
    soundfile

# Copy application code
COPY app.py ./
COPY backend/ ./backend/
COPY languages/ ./languages/
COPY ui/ ./ui/

# Directories for runtime assets (gitignored on host, created at runtime)
RUN mkdir -p ct2_models speaker_voices processed/voice_qc certs

# Non-root user for security
RUN useradd -m -u 1000 bpuser && chown -R bpuser:bpuser /app
USER bpuser

# Environment defaults (overridable at runtime)
ENV BP_HOST=0.0.0.0
ENV BP_PORT=8000
ENV BP_DEVICE=auto
ENV BP_COMPUTE_TYPE=int8
ENV BP_MT_DEVICE=cpu
ENV BP_HTTPS=0
ENV BP_DISABLE_RATE_LIMIT=0
ENV BP_LOG_SESSIONS=0
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/hardware || exit 1

CMD ["python", "app.py"]