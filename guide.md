# Project Guide: Real-Time Speech Translation System

This document provides a concise technical overview of the Real-Time Speech Translation System, outlining its architecture, key technologies, and operational principles. It serves as a detailed guide for understanding the project's core components and design decisions, without the extensive detail of the full thesis.

## 1. Project Overview

The Real-Time Speech Translation System is a web-accessible application designed to facilitate cross-lingual communication in online conferences. It captures a speaker's voice, translates it in real-time, and synthesizes the translated speech, which is then routed to other conference participants via a virtual audio device. The primary goal is to achieve low-latency, high-quality translation with a focus on backend robustness and multi-user scalability.

## 2. System Architecture

The system employs a modular client-server architecture:

*   **Frontend (Web-based UI):** Handles microphone audio capture (via WebRTC), displays real-time transcriptions and translations, and plays back synthesized audio. It is built with plain HTML, CSS, and JavaScript for broad compatibility.
*   **Backend (FastAPI Server):** Orchestrates the core speech processing pipeline: Speech-to-Text (STT), Machine Translation (MT), and Text-to-Speech (TTS). It manages WebSocket connections for real-time data exchange and utilizes Voice Activity Detection (VAD) for efficient audio processing.

### 2.1 Detailed Pipeline Flow

1.  **Microphone Input:** User's speech is captured by the browser using WebRTC.
2.  **WebRTC VAD:** Voice Activity Detection segments speech from silence.
3.  **Faster-Whisper STT:** Speech segments are transcribed into text.
4.  **CTranslate2 MT:** Transcribed text is translated into the target language.
5.  **TTS Model Selection:** User chooses between Piper TTS (fast, generic voice) or Coqui TTS (zero-shot voice cloning using XTTS v2).
6.  **Piper TTS / Coqui TTS:** Translated text is synthesized into audio. Coqui TTS supports voice cloning for multiple languages including English and Czech (Slovak uses Czech model due to language similarity). For best performance without voice cloning, Piper TTS provides fast, high-quality generic voices.
7.  **Audio Output:** Synthesized audio is streamed back to the frontend.
8.  **Virtual Audio Device:** The frontend routes the synthesized audio to a virtual audio device (e.g., BlackHole on macOS, VB-Cable on Windows), which acts as a microphone input for other conferencing applications.

## 3. Key Technologies & Design Decisions

The technology stack is chosen for performance, scalability, and open-source availability:

*   **Frontend (HTML, CSS, JavaScript, Web Audio API, WebSockets, Chart.js):**
    *   **Why:** Standard web technologies ensure cross-platform accessibility. Web Audio API provides low-level audio control for efficient capture and processing (via Web Audio Worklets). WebSockets enable low-latency, bidirectional communication. Chart.js is used for real-time latency visualization.
    *   **Alternatives:** Heavy frontend frameworks were avoided to maintain simplicity and focus on the core pipeline.
*   **Backend (Python, FastAPI, `faster-whisper`, `CTranslate2` with `Opus-MT`, `Piper TTS`, `Coqui TTS`, `webrtcvad`):**
    *   **Why:** Python is ideal for AI/ML. FastAPI offers high performance and asynchronous capabilities for handling multiple users.
    *   `faster-whisper` (optimized Whisper) is chosen for its speed, accuracy, and multilingual support. **Based on initial testing (Phase 0.1), the "base" model size was selected as the default for its balance of speed and acceptable accuracy (WER of 0.2857 for short English phrases), prioritizing real-time performance.**
    *   `CTranslate2` with `Helsinki-NLP Opus-MT` provides efficient, broad-language machine translation. **Testing (Phase 0.2) showed a BLEU score of approximately 29.93 and a METEOR score of 0.5000 for both EN->SK and SK->EN translations, with latencies around 0.9-1.0 seconds. This demonstrates a reasonable level of accuracy and speed for the chosen models.**
    *   `Piper TTS` offers fast, high-quality generic speech, while `Coqui TTS` (XTTS v2) provides advanced zero-shot voice cloning for supported languages (e.g., English, Czech). This dual approach balances speed and personalization.
    *   `webrtcvad` is a robust and efficient VAD for real-time speech detection.
    *   **Alternatives:** Cloud APIs were avoided to minimize latency, cost, and external dependencies. Other frameworks/models were considered but deemed less optimal for real-time, local execution.
*   **Deployment/Dev (`Apple Silicon`, `ffmpeg`, `openssl`):**
    *   **Why:** The development environment is specifically optimized for **macOS with M1 Pro chip and 16GB RAM**, leveraging Apple Silicon for its high performance-per-watt, integrated GPU (MPS), and unified memory architecture. This provides a powerful local development and testing platform for AI models. `ffmpeg` is an industry-standard for robust audio processing. `openssl` secures local communication with HTTPS/WSS.

## 4. Scalability & Performance

*   **Session-Based Model Management:** The backend implements per-user isolation, where each WebSocket connection has its own dedicated model instances. This prevents global bottlenecks and ensures consistent performance for multiple simultaneous users.
*   **Asynchronous Processing:** FastAPI's `asyncio` capabilities and thread pooling offload CPU-bound tasks, preventing the main event loop from blocking.
*   **VAD and Audio Chunking:** Efficiently segments and processes audio in small chunks, reducing computational load and minimizing latency.

## 5. Testing Approach

The system will undergo rigorous testing:

*   **Functional Testing:** UI interactions, backend API endpoints (user authentication, voice management), and end-to-end pipeline verification.
*   **Virtual Audio Device Integration:** Manual verification on macOS (BlackHole) and Windows (VB-Cable in UTM VM) to confirm translated audio routing to conferencing applications.
*   **Performance & Latency Testing:** Measurement of end-to-end and component-wise latencies.
*   **Scalability Testing:** A dedicated test will simulate multiple concurrent users to assess session-based performance and resource utilization.
*   **Accuracy Evaluation:** Word Error Rate (WER) for STT and BLEU/METEOR scores for MT.
*   **TTS Quality:** Subjective listening tests for Piper and Coqui TTS.
*   **Voice Cloning Accuracy (Coqui TTS)**: Evaluate how accurately Coqui TTS replicates speaker characteristics across supported languages (e.g., Czech).

## 6. User Workflow (Virtual Microphone Setup)

**Prerequisites:**
- macOS: Install BlackHole 2ch (`brew install blackhole-2ch`) or BlackHole 16ch
- Windows: Install VB-Cable from [VB-Audio Website](https://vb-audio.com/Cable/)

**Steps:**

1.  **Install Virtual Audio Device:** Follow the installation instructions for your operating system.
2.  **Run Application:** Start the FastAPI backend server (`make run`).
3.  **Open Web Interface:** Navigate to `https://localhost:8000/ui/live-speech/live.html` in your browser (Chrome or Edge recommended for best compatibility).
4.  **Grant Permissions:** Allow microphone access when prompted.
5.  **Select Audio Output Device:** 
    - In the web interface, locate the "Audio Output" dropdown (below TTS Model selector).
    - The system will auto-detect and pre-select BlackHole 2ch (macOS) or VB-Cable (Windows) if installed.
    - If not auto-detected, manually select your virtual audio device from the dropdown.
    - Hint text will show "✓ Virtual device detected" in green if successful.
6.  **Configure Conferencing App:** 
    - In Zoom/Google Meet/Teams, open audio settings.
    - Set **Microphone** to:
      - macOS: "BlackHole 2ch" (or "BlackHole 16ch")
      - Windows: "CABLE Output (VB-Audio Virtual Cable)"
7.  **Initialize Models:** Click "Initialize Pipeline" and wait for models to load.
8.  **Start Translation:** 
    - Select input and output languages.
    - Choose TTS model (Piper for generic voice, XTTS for custom voice).
    - Click "Start" and begin speaking.
9.  **Verify Routing:** 
    - You should **NOT** hear the translation through your speakers (to avoid echo with your speech).
    - In your conferencing app, you should see microphone activity when the translation plays.
    - Other participants will hear your translated speech.

**Troubleshooting:**
- If you hear translation through speakers: Check that the correct virtual device is selected in the dropdown.
- If conferencing app doesn't receive audio: Verify the virtual device is selected as microphone input in the conferencing app.
- If "No virtual device found" warning appears: Install BlackHole or VB-Cable and refresh the page.

## Phase 0.1: STT Model Test Findings

**Test Case:** `test/Can you hear me_.wav` (Short English audio)
*   **Ground Truth:** 'Hey laddy, can you hear me well?'
*   **Predicted Transcription:** 'hey ladi, can you hear me well?'
*   **Transcription Latency:** 7.9107 seconds
*   **Faster-Whisper WER:** 0.2857

## Phase 0.2: MT Model Test Findings

**Test Case:** `test/My test speech transcript.txt` (English) and `test/My test speech translation.txt` (Slovak)

**EN -> SK Translation:**
*   **Input Text:** `In this experiment, the system converts spoken English into text, translates it into Slovak and then synthesizes it back into speech. The process involves three stages, speech recognition, machine translation, and text to speech. Each model introduces a small delay, so we will measure the overall latency. The main objective is to achieve natural communication despite language barriers.`
*   **Ground Truth (SK):** `V tomto experimente systém prevádza hovorenú angličtinu do textu, preloží ho do slovenčiny a potom ho syntetizuje späť do reči. Proces zahŕňa tri fázy: rozpoznávanie reči, strojový preklad a prevod textu na reč. Každý model zavádza malé oneskorenie, takže budeme merať celkovú latenciu. Hlavným cieľom je dosiahnuť prirodzenú komunikáciu napriek jazykovým bariéram.`
*   **Predicted Translation (SK):** `V tomto experimente systém konvertuje hovorenú angličtinu na text, prekladá ju do slovenčiny a potom ju syntetizuje späť do reči. Proces zahŕňa tri etapy, rozpoznávanie reči, strojový preklad a text do reči. Každý model predstavuje malé oneskorenie, takže zmeriame celkovú latenciu. Hlavným cieľom je dosiahnuť prirodzenú komunikáciu aj napriek jazykovým bariéram.`
*   **Translation Latency (EN->SK):** 0.9327 seconds
*   **CTranslate2MT BLEU (EN->SK):** 29.9300
*   **CTranslate2MT METEOR (EN->SK):** 0.5000

**SK -> EN Translation:**
*   **Input Text:** `V tomto experimente systém prevádza hovorenú angličtinu do textu, preloží ho do slovenčiny a potom ho syntetizuje späť do reči. Proces zahŕňa tri fázy: rozpoznávanie reči, strojový preklad a prevod textu na reč. Každý model zavádza malé oneskorenie, takže budeme merať celkovú latenciu. Hlavným cieľom je dosiahnuť prirodzenú komunikáciu napriek jazykovým bariéram.`
*   **Ground Truth (EN):** `In this experiment, the system converts spoken English into text, translates it into Slovak and then synthesizes it back into speech. The process involves three stages, speech recognition, machine translation, and text to speech. Each model introduces a small delay, so we will measure the overall latency. The main objective is to achieve natural communication despite language barriers.`
*   **Predicted Translation (EN):** `In this experiment, the system converts spoken English into text, translates it into English and then synthesizes it back into language. The process involves three phases: speech recognition, machine translation and translation of text into language. Each model introduces a small delay, so that we measure the overall latency. The main objective is to achieve natural communication despite language barriers.`
*   **Translation Latency (SK->EN):** 1.0437 seconds
*   **CTranslate2MT BLEU (SK->EN):** 29.9300
*   **CTranslate2MT METEOR (SK->EN):** 0.5000
