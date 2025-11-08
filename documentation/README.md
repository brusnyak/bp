# Live Speech Translation System - Bachelor's Thesis Project

## Project Overview

This project implements a real-time live speech translation system designed for conference environments. It leverages state-of-the-art open-source models for Speech-to-Text (STT), Machine Translation (MT), and Text-to-Speech (TTS) to provide low-latency, high-quality translation. The system is optimized for Apple Silicon (M1/M2/M3) hardware and features a modern web-based user interface.

**Key Features:**
*   **Real-time Performance:** Optimized for minimal end-to-end latency, targeting 2-3 seconds for standard TTS and 2.5-3.5 seconds for voice cloning.
*   **Modular Architecture:** Built with FastAPI for the backend and a responsive web UI (HTML, CSS, JavaScript) for easy interaction.
*   **Speech-to-Text (STT):** Utilizes `faster-whisper` for efficient and accurate transcription.
*   **Machine Translation (MT):** Employs `SeamlessM4T v2` for high-quality, multilingual translation.
*   **Text-to-Speech (TTS):** Integrates `Piper TTS` for fast, natural-sounding speech synthesis.
*   **Voice Cloning (Phase 2):** Designed for future integration with `F5-TTS` to enable real-time voice cloning, allowing translated speech to retain the original speaker's voice characteristics.
*   **Voice Activity Detection (VAD):** Incorporates `webrtcvad` for robust speech segment detection, crucial for streaming performance.
*   **Dynamic Language Switching:** Supports on-the-fly switching of input and output languages.
*   **Latency Visualization:** The UI includes a real-time timeline chart to visualize pipeline latency.
*   **Speaker Voice Management:** Frontend and backend support for recording, uploading, and managing speaker voice profiles for cloning.

## Architecture

The system follows a client-server architecture:

1.  **Frontend (UI):** A web application built with HTML, CSS, and JavaScript. It captures microphone audio, sends it to the backend via WebSockets, displays real-time transcriptions and translations, and plays back synthesized audio. It also manages language selection and speaker voice profiles.
2.  **Backend (FastAPI):** A Python application using FastAPI. It handles WebSocket connections, orchestrates the STT, MT, and TTS models, performs VAD, and streams results back to the frontend.

**Pipeline Flow:**
Audio Stream (Frontend) -> VAD -> STT (FasterWhisper) -> MT (SeamlessM4T v2) -> TTS (Piper/XTTS) -> Audio Playback (Frontend)

## Setup and Installation

### Prerequisites

*   **Python 3.9+**
*   **pip** (Python package installer)
*   **Git**
*   **FFmpeg** (for audio processing, usually pre-installed or easily installed via Homebrew on macOS: `brew install ffmpeg`)
*   **BlackHole 2ch** (or similar virtual audio device for macOS, recommended for routing audio output for testing: `brew install blackhole-2ch`)

### Steps

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/brusnyak/bp.git
    cd bp
    ```

2.  **Create and Activate a Virtual Environment:**
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Install the required Python packages. The `requirements.txt` file is optimized for Apple Silicon.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you encounter issues with `torch` or `torchaudio` on Apple Silicon, refer to the official PyTorch installation guide for specific commands for your macOS version and chip.*

4.  **Download Models:**
    *   **Piper TTS Models:** The system will attempt to download Piper TTS models on first initialization if they are not found locally. However, you can manually download them using the provided script:
        ```bash
        python backend/tts/download_piper_models.py en_US-ryan-medium
        python backend/tts/download_piper_models.py sk_SK-lili-medium
        python backend/tts/download_piper_models.py cs_CZ-jirka-medium
        # Download other languages as needed from PIPER_MODEL_MAPPING in backend/main.py
        ```
    *   **CTranslate2 MT Models:** You need to convert Opus-MT models to CTranslate2 format.
        ```bash
        python backend/mt/convert_opus_mt_to_ct2.py --model_name Helsinki-NLP/opus-mt-en-sk
        python backend/mt/convert_opus_mt_to_ct2.py --model_name Helsinki-NLP/opus-mt-sk-en
        python backend/mt/convert_opus-mt-to-ct2.py --model_name Helsinki-NLP/opus-mt-en-cs
        # Convert other language pairs as needed
        ```
    *   **FasterWhisper STT Model:** The `FasterWhisperSTT` model (`large-v3`) will be downloaded automatically on first use.
    *   **XTTS v2 (for Voice Cloning - Phase 2):** This model will be downloaded automatically on first use if selected.

5.  **Generate SSL Certificates (for HTTPS):**
    The FastAPI server runs with HTTPS. Generate self-signed certificates:
    ```bash
    openssl req -x509 -newkey rsa:4096 -nodes -out certs/cert.pem -keyout certs/key.pem -days 365 -subj "/CN=localhost"
    ```

6.  **Run the Application:**
    ```bash
    python app.py
    ```
    The application will start on `https://localhost:8000`. You might need to accept the self-signed certificate in your browser.

## Usage

1.  **Open in Browser:** Navigate to `https://localhost:8000` in your web browser.
2.  **Initialize Pipeline:** Click the "Initialize Pipeline" button. This will load all necessary models. The first load may take some time.
3.  **Select Languages:** Choose your desired input and output languages from the dropdowns.
4.  **Record Voice (Optional for XTTS):** If you plan to use XTTS for voice cloning, click the "Record Voice" button. Follow the prompts to record a short audio sample of your voice or upload an existing WAV file. This voice profile will be used for synthesis.
5.  **Start Speaking:** Once initialized, the system will automatically start listening for speech. Speak into your microphone.
6.  **Real-time Translation:** Observe the transcription and translation appearing in real-time. The translated speech will be played back through your selected audio output.
7.  **Monitor Latency:** The "Latency Breakdown" section and the timeline chart will show real-time performance metrics.

## Testing

A comprehensive testing framework is provided in the `test/` directory.

To run the streaming pipeline tests:
```bash
python test/streaming_pipeline_tests.py
```

**Note on Test Audio:**
For full testing, you will need to provide actual `.wav` audio files for the following paths:
*   `test/My test speech_xtts_speaker_clean.wav` (English speech for general testing)
*   `test/slovak_test_speech.wav` (Slovak speech for multi-language testing)
*   `test/Voice-Training.wav` (Speaker reference audio for XTTS voice cloning)

Ensure these files are placed in the `test/` directory. The corresponding `_transcript.txt` and `_translation.txt` files should contain the accurate text references for evaluation.

## Future Enhancements (Phase 2 & 3)

*   **F5-TTS Integration:** Fully integrate F5-TTS for superior voice cloning capabilities.
*   **Multi-speaker Support:** Extend the system to handle multiple speakers in a conference setting.
*   **Production Optimization:** Explore model quantization, `whisper.cpp` or `mlx-whisper` for STT, and cloud deployment options.
*   **`pip` Packaging:** Simplify installation by packaging the project as a Python library.

## Thesis Suggestions

Refer to `documentation/thesis_suggestions.txt` for detailed content suggestions for your bachelor's thesis, covering introduction, literature review, methodology, implementation details, results, and future work.
