# Real-Time Speech Translation System

## Project Overview

This project implements a real-time live speech translation system designed for conference environments. It leverages state-of-the-art open-source models for Speech-to-Text (STT), Machine Translation (MT), and Text-to-Speech (TTS) to provide low-latency, high-quality translation. The system is optimized for Apple Silicon (M1/M2/M3) hardware and features a modern web-based user interface.

### Live Demo & Examples

Witness the seamless real-time speech translation in action. See how our system effortlessly bridges language gaps.

[![Real-Time Speech Translation Demo](https://img.youtube.com/vi/_-jwEyGxDYs/maxresdefault.jpg)](https://www.youtube.com/watch?v=_-jwEyGxDYs)

**Key Features:**

- **Real-time Performance:** Optimized for minimal end-to-end latency, targeting <1.5 seconds for standard translation and <2 seconds for voice cloning with upcoming TTS optimizations.
- **Modular Architecture:** Built with FastAPI for the backend and a responsive web UI (HTML, CSS, JavaScript) for easy interaction.
- **Speech-to-Text (STT):** Utilizes `faster-whisper` for efficient and accurate transcription.
- **Machine Translation (MT):** Employs `CTranslate2` optimized Opus-MT models for high-quality, multilingual translation.
- **Text-to-Speech (TTS):** Integrates `Piper TTS` for fast, natural-sounding speech synthesis, with `OmniVoice` for low-latency zero-shot voice cloning (40x real-time, Apache 2.0 licensed).
- **Voice Activity Detection (VAD):** Incorporates `webrtcvad` for robust speech segment detection, crucial for streaming performance.
- **Dynamic Language Switching:** Supports on-the-fly switching of input and output languages.
- **Latency Visualization:** The UI includes a real-time timeline chart to visualize pipeline latency.
- **Speaker Voice Management:** Frontend and backend support for recording, uploading, and managing speaker voice profiles for cloning.
- **Future-Ready Design:** Modular architecture allows easy integration of end-to-end models and streaming translation techniques.

## Architecture

The system follows a client-server architecture:

1.  **Frontend (UI):** A web application built with HTML, CSS, and JavaScript. It captures microphone audio, sends it to the backend via WebSockets, displays real-time transcriptions and translations, and plays back synthesized audio. It also manages language selection and speaker voice profiles.
2.  **Backend (FastAPI):** A Python application using FastAPI. It handles WebSocket connections, orchestrates the STT, MT, and TTS models, performs VAD, and streams results back to the frontend.

**Pipeline Flow:**
Audio Stream (Frontend) -> VAD -> STT (FasterWhisper) -> MT (CTranslate2 Opus-MT) -> TTS (Piper/Experimental Zero-Shot) -> Audio Playback (Frontend)

*Note: TTS module is designed for easy swapping between Piper TTS (fast) and experimental zero-shot voice cloning models (Voxtral, Qwen3-TTS) for voice preservation.*

## Setup and Installation

### Prerequisites

- **Python 3.9+**
- **pip** (Python package installer)
- **Git**
- **FFmpeg** (for audio processing, usually pre-installed or easily installed via Homebrew on macOS: `brew install ffmpeg`)
- **BlackHole 2ch** (or similar virtual audio device for macOS, recommended for routing audio output for testing: `brew install blackhole-2ch`)

### Steps

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/brusnyak/bp.git
    cd bp
    ```

    **Windows:**
    Run the provided PowerShell setup script:

    ```powershell
    .\setup_windows.ps1
    ```

    This script will automatically install Python 3.11, FFmpeg, Node.js, create a virtual environment, and install all dependencies.

    **macOS / Linux:**

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

    _Note: If you encounter issues with `torch` or `torchaudio` on Apple Silicon, refer to the official PyTorch installation guide for specific commands for your macOS version and chip._

4.  **Download Models:**

    - **Piper TTS Models:** The system will attempt to download Piper TTS models on first initialization if they are not found locally. However, you can manually download them using the provided script:
      ```bash
      python backend/tts/download_piper_models.py en_US-ryan-medium
      python backend/tts/download_piper_models.py sk_SK-lili-medium
      python backend/tts/download_piper_models.py cs_CZ-jirka-medium
      # Download other languages as needed from PIPER_MODEL_MAPPING in backend/main.py
      ```
    - **CTranslate2 MT Models:** You need to convert Opus-MT models to CTranslate2 format.
      ```bash
      python backend/mt/convert_opus_mt_to_ct2.py --model_name Helsinki-NLP/opus-mt-en-sk
      python backend/mt/convert_opus-mt-to-ct2.py --model_name Helsinki-NLP/opus-mt-sk-en
      python backend/mt/convert_opus-mt-to-ct2.py --model_name Helsinki-NLP/opus-mt-en-cs
      # Convert other language pairs as needed
      ```
    - **FasterWhisper STT Model:** The `FasterWhisperSTT` model (`large-v3`) will be downloaded automatically on first use.
    - **F5-TTS (for Voice Cloning):** This model will be downloaded automatically on first use if selected.

5.  \*\*Generate SSL Certificates (for HTTPS):
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
4.  **Record Voice (Optional for F5-TTS):** If you plan to use F5-TTS for voice cloning, select "F5" as the TTS model, then click the "Record Voice" button. Follow the prompts to record a short audio sample of your voice or upload an existing WAV file. This voice profile will be used for synthesis.
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

- `test/My test speech_xtts_speaker_clean.wav` (English speech for general testing)
- `test/slovak_test_speech.wav` (Slovak speech for multi-language testing)
- `test/Voice-Training.wav` (Speaker reference audio for XTTS voice cloning)

Ensure these files are placed in the `test/` directory. The corresponding `_transcript.txt` and `_translation.txt` files should contain the accurate text references for evaluation.

## Future Enhancements

- **Multi-speaker Support:** Extend the system to handle multiple speakers in a conference setting.
- **Production Optimization:** Explore model quantization, `whisper.cpp` or `mlx-whisper` for STT, and cloud deployment options.
- **`pip` Packaging:** Simplify installation by packaging the project as a Python library.

## Thesis Suggestions

Refer to `documentation/thesis_suggestions.txt` for detailed content suggestions for your bachelor's thesis, covering introduction, literature review, methodology, implementation details, results, and future work.

---

**Current Development Status: F5-TTS Integration & Frontend/Backend Stability**

**Objective:** Successfully integrate F5-TTS for real-time voice cloning and resolve critical frontend and backend issues.

**Completed Actions:**

- **Frontend JavaScript Errors Resolved:**
  - Fixed `ReferenceError: loadF5Voices is not defined` in `ui/js/main.js` by correctly calling `fetchStoredVoices(populateF5VoiceSelect)`.
  - Resolved `TypeError: Cannot read properties of undefined (reading 'inputSampleRate')` in `ui/audio-processor.js` by passing `processorOptions` to `AudioWorkletNode` in `ui/js/audio_processing.js`.
  - Improved F5-TTS UI logic in `ui/index.html` and `ui/js/main.js` for better display of voice selection and record button.
- **Backend FFmpeg Integration Improved:**
  - Replaced `torchaudio.save` with `soundfile.write` in `backend/tts/f5_tts.py` to bypass `torchaudio`'s problematic FFmpeg integration, ensuring consistent audio handling with `soundfile`.
- **F5-TTS Integration:** F5-TTS is now integrated as a selectable TTS model with voice cloning capabilities.

**Next Steps:**

- **Implement UI Error Handling and Feedback:** Enhance the frontend to provide clear user feedback for backend initialization failures (e.g., F5-TTS without a voice).
- **Run Performance Tests:** Execute `test/tts_performance_test.py` to gather data on Piper vs F5-TTS performance and quality.
- **Update Documentation:** Ensure all documentation, including `documentation/thesis.docx`, reflects the current state of the project.
