# Real-Time Speech Translation System

Real-time multilingual speech translation system with voice cloning capabilities, designed for seamless integration with virtual microphones (BlackHole 2ch on macOS, VB-Cable on Windows) for use in video conferencing applications like Google Meet, Zoom, and Microsoft Teams.

## Features
- **Real-time Speech-to-Text (STT)**: faster-whisper (CTranslate2-optimized Whisper)
- **Machine Translation (MT)**: Helsinki-NLP Opus-MT models (CTranslate2-optimized)
- **Text-to-Speech (TTS)**:
  - **Piper TTS**: Fast, generic voice synthesis
  - **Coqui XTTS v2**: Zero-shot voice cloning with speaker embedding caching
- **Voice Activity Detection (VAD)**: WebRTC VAD for real-time speech detection
- **Browser-Based Virtual Mic Routing**: Route translated audio to conferencing apps via Web Audio API
- **Concurrent User Support**: Session-based model isolation with proper cleanup
- **Web Interface**: Real-time latency monitoring with Chart.js

---

## Hardware Requirements

### Recommended Hardware by Platform

| Platform | Device Mode | Minimum Requirements | Performance (XTTS RTF) | Notes |
|----------|-------------|---------------------|------------------------|-------|
| **macOS (Apple Silicon)** | CPU | 8GB RAM, M1 or later | 1.7-2.0 | Recommended for development |
| **Windows** | CUDA | NVIDIA GPU (6GB+ VRAM), CUDA 11.8+ | 0.8-1.2 | Best performance |
| **Windows** | CPU | 16GB RAM, modern CPU (i5/Ryzen 5+) | 2.5-3.5 | Slower but functional |
| **Linux** | CUDA | NVIDIA GPU (6GB+ VRAM), CUDA 11.8+ | 0.8-1.2 | Best performance |
| **Linux** | CPU | 16GB RAM | 2.5-3.5 | Slower but functional |

> **Note**: RTF (Real-Time Factor) measures synthesis speed. RTF < 2.0 means audio is synthesized faster than real-time (e.g., 1.7 RTF = 1 second of audio generated in 0.59 seconds).

### Device Detection

The system automatically detects the best available device:
1. **CUDA** (NVIDIA GPU) - Best performance
2. **MPS** (Apple Silicon) - Note: XTTS v2 has limited MPS support, automatically falls back to CPU
3. **CPU** - Universal fallback

---

## Installation

### Prerequisites

- **Python 3.9-3.11** (3.11 recommended)
- **FFmpeg** (required for audio processing)
- **Node.js** (for frontend dependencies)

### Platform-Specific Setup

#### macOS

```bash
# Install FFmpeg via Homebrew
brew install ffmpeg

# Install Python dependencies
make install

# Or manually:
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Linux (Debian/Ubuntu)

```bash
# Install FFmpeg
sudo apt update && sudo apt install ffmpeg

# Install Python dependencies
make install

# Or manually:
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project Goal

To develop a standalone, web-accessible application capable of real-time speech translation during online conferences for multiple simultaneous users. The system focuses on a robust backend solution with client-side audio routing, enabling translated audio to be seamlessly integrated into existing web conferencing platforms (Google Meet, Zoom, Teams) using virtual audio devices and browser Web Audio API.

## Key Features

*   **Real-Time Speech-to-Text (STT):** Transcribes spoken language into text using optimized models.
*   **Real-Time Machine Translation (MT):** Translates transcribed text into the target language.
*   **Real-Time Text-to-Speech (TTS):** Synthesizes translated text into speech, offering both generic and voice-cloned output.
*   **Voice Activity Detection (VAD):** Efficiently segments speech from silence to optimize processing.
*   **Dynamic Language Switching:** Users can select input and target languages on the fly.
*   **Voice Cloning:** Utilizes advanced TTS models to synthesize translated speech in the original speaker's voice.
*   **Session-Based Scalability:** Backend designed to handle multiple simultaneous users with isolated model instances.
*   **Browser-Based Audio Routing:** Uses Web Audio API to route translated audio through virtual audio devices (BlackHole/VB-Cable) directly in the browser, compatible with multi-user server deployment.
*   **Real-time Latency Visualization:** UI provides metrics and charts for pipeline performance.
*   **User Authentication & Voice Management:** Secure user registration, login, and management of speaker voice profiles.

## Core Technologies

The system is built on a modular client-server architecture:

*   **Frontend:** HTML, CSS, JavaScript, Web Audio API, WebSockets, Chart.js
*   **Backend:** Python, FastAPI, `faster-whisper` (STT), `CTranslate2` with `Helsinki-NLP Opus-MT` (MT), `Piper TTS` & `Coqui TTS` (TTS), `webrtcvad` (VAD).
*   **Deployment/Dev:** Apple Silicon (M1/M2/M3) for local development, `ffmpeg` for audio processing, `openssl` for SSL certificates.

## How to Use the Application (First-Time User Workflow)

To use the real-time speech translation system with your preferred online conferencing application (e.g., Google Meet, Zoom), follow these steps:

### 1. Install Virtual Audio Device

This application routes translated audio through a virtual microphone. You need to install a specific driver for your operating system:

*   **macOS:** Install [BlackHole](https://github.com/ExistentialAudio/BlackHole).
    *   Download and install the appropriate BlackHole package (e.g., `BlackHole 2ch`).
    *   Restart your browser after installation.
*   **Windows:** Install [VB-Cable](https://vb-audio.com/Cable/).
    *   Download and extract the VB-Cable archive.
    *   Run `VBCABLE_Setup_x64.exe` (or `VBCABLE_Setup.exe`) as administrator and click "Install Driver".
    *   Restart your browser after installation.

### 2. Run the Application

Ensure you have the project dependencies installed (refer to `Makefile` for `make install` instructions).

```bash
# Navigate to the project root directory
cd /path/to/your/bp

# Run the FastAPI backend server
make run
```
This will start the backend server, typically accessible at `https://localhost:8000`.

### 3. Access the Web Application

Open your web browser (Chrome or Edge recommended) and navigate to:
```
https://localhost:8000/ui/live-speech/live.html
```

*   **Grant Microphone Permission:** When prompted, allow the application to access your microphone.
*   **Select Audio Output Device:**
    *   Locate the "Audio Output" dropdown in the web interface (below the TTS Model selector).
    *   The system will automatically detect and pre-select BlackHole (macOS) or VB-Cable (Windows) if installed.
    *   If not auto-detected, manually select your virtual audio device from the dropdown.
    *   A green checkmark (✓) will appear if a virtual device is successfully selected.

### 4. Configure Your Conferencing Application

Open your online conferencing application (Google Meet, Zoom, Teams) in a separate browser tab or desktop app.

*   **Select Virtual Microphone Input:**
    *   Go to the audio/microphone settings in your conferencing application.
    *   Select the virtual audio device as your **microphone input**:
      - **macOS**: "BlackHole 2ch" or "BlackHole 16ch"  
      - **Windows**: "CABLE Output (VB-Audio Virtual Cable)"

### 5. Start Translation

1.  In the web application, click "Initialize Pipeline" and wait for models to load.
2.  Select your desired **input language** and **target language**.
3.  Choose your preferred **TTS model**:
    - **Piper**: Fast, generic voice
    - **XTTS**: Voice cloning (requires recording or uploading a voice sample)
4.  Click "Start" and begin speaking into your physical microphone.
5.  **Verify**: 
    - You should **NOT** hear the translation through your speakers (to prevent echo).
    - Your conferencing app should show microphone activity when translation plays.
    - Other participants will hear your translated speech.

## Project Structure

```
.
├── backend/                    # FastAPI backend for STT, MT, TTS, VAD, Auth
│   ├── main.py                 # Main FastAPI app, WebSocket handler
│   ├── stt/                    # Speech-to-Text module (Faster-Whisper)
│   ├── mt/                     # Machine Translation module (CTranslate2 Opus-MT)
│   ├── tts/                    # Text-to-Speech modules (Piper, Coqui TTS)
│   └── utils/                  # Audio utilities, DB manager, Auth functions
├── ui/                         # Frontend (HTML, CSS, JavaScript)
│   ├── home/                   # Home page, thesis content
│   ├── auth/                   # User authentication pages
│   ├── live-speech/            # Real-time translation interface
│   ├── global-styles.css       # Global styling
│   └── ...
├── documentation/              # Project documentation
│   ├── bp                      # Bachelor's Project Outline
│   ├── plan.txt                # Detailed Project Plan
│   └── thesis_draft.md         # Thesis content (expanded with justifications, results)
├── speaker_voices/             # Stores user voice profiles for Coqui TTS
├── test/                       # Unit and integration tests
│   ├── full_pipeline_test.py   # End-to-end pipeline tests
│   ├── latency_pipeline_test.py# Latency benchmarks
│   ├── stt_comparison.py       # STT model comparison
│   └── ...
├── app.py                      # Main application entry point
├── Makefile                    # Build and run commands
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Demo

A live demo showcasing the real-time translation from English to a target language, with the translated audio routed through a virtual device for other participants to hear, will be available in the home UI and linked here upon completion.

## Contact

Developed by brusnyak for Bachelor's Thesis 2024/2025.
