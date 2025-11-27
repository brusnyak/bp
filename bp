Type of work: Bachelor thesis
Topic: Preklad reči v reálnom čase počas online konferencie
Title of topic in topic in English: Real Time Speech Translation in Online Conference
Progress of registration: approved
State of topic: approved (doc. Ing. Radoslav Vargic, PhD. - Study programme supervisor)
Thesis supervisor: I. Minárik
Faculty: Faculty of Electrical Engineering and Information Technology
Supervising department: Institute of Multimedia Information and Communication Technologies - FEI
Max. no. of students: 1
Academic year: 2024/2025
Proposed by: Ing. Ivan Minárik
Summary: Online konferencie často narážajú na jazykové bariéry medzi účastníkmi. Moderné technológie na rozpoznávanie, preklad a syntézu reči umožňujú tieto prekážky prekonávať. Cieľom práce je vytvoriť prototyp systému, ktorý zabezpečí preklad reči v reálnom čase medzi angličtinou a ďalším vybraným jazykom počas online stretnutí, s využitím dostupných nástrojov a so zameraním na backendové riešenie a overenie funkčnosti.

**Project Outline: Real-Time Speech Translation in Online Conference**

**I. Introduction & Objectives**

- **Topic**: Real-Time Speech Translation in Online Conference
- **Goal**: To develop a standalone, web-accessible application capable of real-time speech translation during online conferences, specifically designed for multiple simultaneous users. The system will capture a speaker's voice, translate it, and synthesize the translated speech in the speaker's own voice (or a generic voice) for other conference participants.
- **Key Challenges**: Scalability for +200 simultaneous users, cross-platform audio handling without proprietary drivers, and maintaining low latency.
- **Development Environment**: Continue development and testing on macOS with M1 Pro chip and 16GB RAM.
- **Deployment Environment**: Anticipate deployment on various server environments. CUDA optimization will be considered later to ensure the project is not architecture/setup reliant and can run on diverse hardware.

**II. Core Functionality & Pipeline**

The application's core logic follows a clear, real-time pipeline:

1.  **Speak to Microphone**: A user speaks into their microphone.
2.  **WebRTC Audio Capture (Frontend)**: The user's microphone audio is captured directly by the browser using WebRTC (`navigator.mediaDevices.getUserMedia`).
3.  **Stream to Backend**: The captured audio stream is sent to the backend via WebSockets.
4.  **Backend Processing (STT -> MT -> TTS)**:
    - **STT (Speech-to-Text)**: The incoming audio stream is processed to transcribe the speech into text.
    - **MT (Machine Translation)**: The transcribed text is translated into the target language.
    - **TTS (Text-to-Speech)**: The translated text is synthesized into speech. This step will leverage voice cloning to speak in the original speaker's voice in the target language, or use a generic voice.
5.  **Stream to Other Participants (Backend to Frontend via WebRTC)**: The synthesized translated audio is streamed back from the backend to the frontend.
6.  **WebRTC Audio Playback (Frontend)**: The frontend receives the translated audio and plays it back through the browser's Web Audio API, routing it to a virtual audio device (e.g., BlackHole on macOS, VB-Cable on Windows) for _other conference participants_. The user initiating the translation will _not_ hear their own translated output.

**III. Backend Architecture & Components**

The backend will be built with FastAPI, leveraging asynchronous processing and a modular structure to ensure scalability and maintainability.

- **A. Core Components**:
  - **STT (Speech-to-Text)**:
    - **Models**: `Faster-Whisper` (current primary) and `MLX` (for evaluation on multi-language and complexity). The system will be designed to allow switching or dynamic selection between these based on performance and accuracy requirements.
    - **Functionality**: Real-time transcription of incoming audio segments, including language detection.
  - **MT (Machine Translation)**:
    - **Model**: `CTranslate2` (using Helsinki-NLP Opus-MT models).
    - **Functionality**: Translates transcribed text from the source language to the target language. Dynamic model conversion will be supported for various language pairs.
  - **TTS (Text-to-Speech)**:
    - **Models**: `Piper` (fast, lightweight, generic voice) and `Coqui TTS` (cross-lingual, zero-shot, higher quality, voice cloning using XTTS v2).
    - **Functionality**: Synthesizes translated text into speech. `Coqui TTS` will utilize stored speaker voice profiles for voice cloning for supported languages (e.g., Czech). For languages not supported by Coqui TTS voice cloning (e.g., Slovak), Piper will be used for generic voice synthesis.
- **B. Architectural Considerations for Scalability (+200 Users)**:
  - **Concurrency Model (Per-User Isolation)**:
    - **Challenge**: Global model instances are a bottleneck.
    - **Solution**: Implemented session-based model management. Each active user session (WebSocket connection) now has its own dedicated model instances, ensuring isolated and non-blocking processing. This prevents race conditions and ensures consistent performance per user.
  - **Asynchronous Processing**: Leveraged FastAPI's asynchronous capabilities. CPU-bound model inference tasks are offloaded to a dedicated thread pool (`loop.run_in_executor`) to prevent blocking the main event loop.
  - **Model Serving Frameworks**: For high-performance, multi-model, multi-GPU inference, **NVIDIA Triton Inference Server** will be integrated in future phases. This will handle dynamic batching, concurrent execution, and efficient resource utilization, especially crucial for CUDA deployments.
  - **Microservices Architecture**: The backend will evolve towards a microservices pattern in future phases, separating STT, MT, and TTS into distinct, independently deployable, and scalable services. A central API Gateway or orchestrator service will manage the flow between these components.
  - **Resource Management & Deployment**:
    - **Containerization (Docker)**: All backend services will be containerized for consistent environments in future phases.
    - **Orchestration (Kubernetes)**: Kubernetes will be used for deploying, managing, and automatically scaling services based on real-time load in future phases.
    - **GPU Acceleration (CUDA/MPS)**: The backend dynamically detects and utilizes available hardware accelerators (CUDA for deployment, MPS for macOS development) for model inference.
    - **Load Balancing**: Load balancing will be implemented in future phases to distribute incoming user requests across multiple backend instances.
    - **Message Queues**: Message queues (e.g., RabbitMQ, Kafka) will be integrated in future phases to decouple services, handle high throughput, and enable robust asynchronous processing of audio segments and translation results.
    - **Stateless Services**: Services are designed to be stateless where possible to facilitate horizontal scaling.
- **C. Data Management**:
  - **Speaker Voices**: Store recorded speaker voice WAV files and their metadata (display name, transcribed text, language, path) in a designated directory (`speaker_voices/`) and a JSON metadata file (`speaker_voices.json`).
  - **User Database**: Implement a database for user registration and management.

**IV. Frontend (UI) Architecture & Features**

The frontend is currently built using plain HTML, CSS, and JavaScript, providing a responsive and maintainable user interface.

- **A. Framework & Styling**:
  - **Framework**: Plain HTML, CSS, and JavaScript.
  - **Styling**: Global CSS (e.g., `ui/global-styles.css`) is used for styling.
- **B. Core UI Features**:
  1.  **Home Page Redesign**:
      - **Header**: Redesigned header with a hamburger menu (mobile), centered "🌐 Live Speech Translation" title, and "Login/Get Started" button with theme toggle (desktop). The header disappears on scroll down and reappears on scroll up.
      - **Content**: Reconfigured into full-screen sections with distinct, theme-compatible background colors. Includes sections for "Hero", "Features", "How-To" (explaining plug-and-play), "Demo" (with embedded video and examples), and "About".
      - **Footer**: Implemented with "© 2025 Lingonberry. All rights reserved." and a "Developed by brusnyak" signature linking to the project repository.
      - **Navigation**: Hero section button navigates to `ui/live-speech/live.html`. Header button navigates to `ui/auth/auth.html`. Placeholder logic for dynamic user account generation (`user#123`) and session-based deletion upon quitting.
  2.  **User Authentication**:
      - **Registration/Login**: Implement a user registration and login system with a database backend.
      - **Authentication Methods**: Support regular email/password authentication and integrate Google authentication for convenience.
      - **Design**: The UI for authentication will be simple yet stylish, based on the provided `auth.txt` content, featuring login/register tabs and Google Sign-In integration.
  3.  **Real-time Translation Controls**:
      - **TTS Model Switch**: Allow users to select between "Piper" (Faster Generic) and "XTTS" (Voice Cloning) TTS models.
      - **Language Settings**: Dropdowns for selecting input language (with auto-detect option) and target language.
      - **Start/Stop Controls**: Buttons to initialize the pipeline and start/stop real-time translation.
  4.  **Real-time Feedback & Indicators**:
      - **Transcription Display**: Live display of transcribed text.
      - **Translation Display**: Live display of translated text.
      - **State Indicators**: Visual cues for "Listening," "Translating," and "Speaking" states.
      - **Microphone Input Level**: A dynamic bar to visualize microphone input volume.
      - **Activity Log**: A scrollable log of application events and status messages.
  5.  **Voice Cloning Management (Coqui TTS Specific)**:
      - **Voice Recording**: A modal interface for users to record their voice for cloning purposes, including agreement checkbox, voice language selection (connected to backend MT for statement translation), mic input level visualization, and simple recording functionality using `MediaRecorder` API.
      - **Voice Upload**: Option to upload an existing WAV file as a speaker voice, with a hidden file input triggered by a button.
      - **Stored Voices Sidebar**: A sidebar displaying a list of stored voices, with options to select, edit (rename), and delete them.
      - **XTTS Voice Selection**: A dropdown to select a specific XTTS voice based on the language.
      - **Voice Training**: The XTTS model will be able to train on the recorded/uploaded voices.
  6.  **Theming**: Day/Night mode switch for user preference.
  7.  **Latency Metrics**: Display of latency breakdown (Input to STT, STT to MT, MT to TTS, TTS to Playback, Backend Pipeline, Total E2E Latency).
  8.  **Charting**: If Chart.js is retained, a timeline chart for latency visualization.

**V. Testing Phase & Quality Assurance**

A rigorous testing phase is crucial to ensure functionality, performance, and user experience.

- **A. Functional Testing**:
  - **End-to-End Pipeline**: Verify the entire pipeline from audio input to translated audio output, including routing to a virtual audio device.
  - **UI Interactions**: Test all buttons, dropdowns, modals, and state changes.
  - **Voice Management**: Test voice recording, upload, saving, renaming, and deletion.
  - **Language Switching**: Ensure correct behavior when changing input/output languages and TTS models.
- **B. Performance & Latency Testing**:
  - **"10s error with input -> stt"**:
    - **Investigation**: Systematically diagnose this issue, starting with disabling VAD to isolate the problem. Review `backend/main.py`'s `handle_audio_stream` and the frontend's WebRTC audio processing for buffering and segment flushing logic.
    - **Resolution**: Optimize VAD parameters and audio buffering to ensure continuous and timely processing.
  - **TTS Testing**:
    - **Quality**: Conduct subjective listening tests and explore objective metrics (e.g., MOS) for both Piper and Coqui TTS.
    - **Latency**: Measure text-to-audio latency for both TTS models.
    - **Voice Cloning Accuracy (Coqui TTS)**: Evaluate how accurately Coqui TTS replicates speaker characteristics across supported languages (e.g., Czech).
  - **Enhanced Latency Metrics**: Improve the granularity and accuracy of displayed latency metrics, including tracking input-STT segment counts and comparing actual speaker talk time against pipeline processing time.
- **C. Scalability Testing**:
  - **Load Testing**: Simulate +200 simultaneous users to identify bottlenecks and validate the backend's ability to scale.
  - **Resource Monitoring**: Monitor CPU, GPU, memory, and network usage during load tests.

**VI. Project Documentation (`bp` and `orig.txt`)**
The `bp` file has been updated to reflect the detailed plan and current implementation status, serving as the definitive project outline. Key updates include:

- **Architectural Design**: Updated to reflect the implemented session-based model management and future plans for microservices and model serving frameworks.
- **Audio Input/Output**: Explicitly states the WebRTC-based, driver-free strategy, including routing to virtual audio devices.
- **UI Framework**: Documents the current use of plain HTML, CSS, and JavaScript.
- **Testing Plan**: Outlines the refined testing strategy for latency, TTS quality, and scalability, including virtual audio device integration.
- **Deployment Considerations**: Explicitly mentions CUDA/MPS compatibility and dynamic detection.
- **Core Functionality**: Details the STT, MT, TTS models and their roles.
- **UI Features**: Lists all planned UI functionalities, including user authentication and voice management.

The `bp` file has been updated to reflect the detailed plan and current implementation status, serving as the definitive project outline. Key updates include:

- **Architectural Design**: Updated to reflect the implemented session-based model management and future plans for microservices and model serving frameworks.
- **Audio Input/Output**: Explicitly states the WebRTC-based, driver-free strategy.
- **UI Framework**: Documents the current use of plain HTML, CSS, and JavaScript.
- **Testing Plan**: Outlines the refined testing strategy for latency, TTS quality, and scalability.
- **Deployment Considerations**: Explicitly mentions CUDA/MPS compatibility and dynamic detection.
- **Core Functionality**: Details the STT, MT, TTS models and their roles.
- **UI Features**: Lists all planned UI functionalities, including user authentication and voice management.
